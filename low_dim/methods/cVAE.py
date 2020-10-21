"""
Created by Rohan Paleja on Sep 9, 2019
This is an implementation of the paper Learning a Multi-Modal Policy via Imitating Demonstrations with Mixed Behaviors
"""

from low_dim.generate_environment import create_simple_classification_dataset
# from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import RelaxedOneHotCategorical
from torch.autograd import Variable
from utils.global_utils import save_pickle

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)

epochs = 50000
state_size = 2
num_classes = 2
device = torch.device("cpu")


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[int(labels.item())].tolist()


class BiRNN(nn.Module):
    """
    bi-directional lstm with attention
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_att = nn.Linear(200, 20)  # 2 for bidirection # 20 is length of traj
        self.soft = nn.Softmax(dim=1)
        self.tanh1 = nn.Tanh()

    def forward(self, x):
        """
        THIS INCLUDES ATTENTION LAYER
        :param x:
        :return:
        """
        # Set initial states

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        shape = out.shape
        out = out.reshape([shape[0], shape[1], shape[2] // 2, 2])
        out = out.sum(dim=3)
        # new_vec = torch.empty(1,1,20)
        # # new_vec.requires_grad=True
        #
        # for n, i in enumerate(out):
        #     if n % 2 == 0:
        #         new_vec[0][0][int(n/2)] = out[0][0][n] + out[0][0][n + 1]

        # out = out.narrow(2, 0, 20)
        # Decode the hidden state of the last time step

        post_att = self.fc_att(out.reshape(1, 200))
        post_att = self.soft(self.tanh1(post_att))
        out = out.reshape((20, 10)) * post_att.reshape((20, 1))
        out = out.sum(dim=0)
        return out


class cVAE(nn.Module):
    """
    full model
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_embedding):
        super(cVAE, self).__init__()
        self.left_model = BiRNN(input_size, hidden_size, num_layers, num_classes)
        self.one_layer_MLP = nn.Linear(10, n_embedding)
        self.soft = nn.Softmax(dim=0)

        # network for predicting action

        self.h1 = nn.Linear(state_size + n_embedding, 10)
        self.h2 = nn.Linear(10, 10)
        self.out = nn.Linear(10, 2)

    def forward(self, x):
        """
        encoder pass
        :param x:
        :return:
        """
        x = self.left_model(x)
        x = self.one_layer_MLP(x)
        x = self.soft(x)
        x = RelaxedOneHotCategorical(.2, probs=x)
        x = x.rsample()
        return x

    def forward_through_decoder(self, x, z):
        """
        decoder pass
        :param x:
        :param z:
        :return:
        """
        x = torch.cat([x, z[None, :].expand(x.shape[0], 1, -1)], dim=2)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.out(x)
        # x = self.soft(x)

        return x


# noinspection PyArgumentList
class EvalcVAE:
    """
    class that handles training and evaluating this approach.
    Some helpful links
    - https://www.kaggle.com/robertke94/pytorch-bi-lstm-attention
    - https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
    """

    def __init__(self):
        input_size = 4  # 2 * 20 for state, 2 extra for action
        hidden_size = 10
        num_layers = 1
        num_classes = 2
        n_embedding = 10
        self.model = cVAE(input_size, hidden_size, num_layers, num_classes, n_embedding)

        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.001)

        # load in data

        # Training set generation
        self.num_schedules = 50
        it_1 = [True, False, False]
        it_2 = [False, True, False]
        it_3 = [False, False, True]
        it = [it_3, it_1, it_2]
        x_data, y = create_simple_classification_dataset(self.num_schedules, train=it[0][0], cv=it[0][1])

        x = []
        for each_ele in x_data:
            x.append(each_ele[2:])

        self.x = torch.Tensor(x).reshape(-1, 2)
        self.y = torch.Tensor(y).reshape((-1, 1))

        print(self.model.state_dict())

    def train(self):
        loss_array = []
        learning_rate = .001
        schedule_starts = np.linspace(0, 20 * (self.num_schedules - 1), num=self.num_schedules)
        for epoch in range(epochs):

            # choose a random schedule
            chosen_schedule_start = int(np.random.choice(schedule_starts))

            schedule_num = int(chosen_schedule_start / 20)
            # input preparation
            length_of_traj = np.random.randint(1, 21)
            chosen_schedule_end = chosen_schedule_start + length_of_traj

            encoder_input = torch.zeros(20, 1, 4)

            # create trajectory is the form of [[s,a],[s,a]...]
            for each_t in range(chosen_schedule_start, chosen_schedule_end):
                input_nn = self.x[each_t].clone()
                # input_nn.extend(one_hot_embedding(self.y[each_t], 2))
                input_nn = torch.cat([input_nn, torch.Tensor(one_hot_embedding(self.y[each_t], 2))], dim=0)
                input_nn = torch.Tensor(np.asarray(input_nn))
                encoder_input[each_t - chosen_schedule_start] = input_nn

            # get the "discrete" embedding
            z = self.model(encoder_input)
            self.opt.zero_grad()

            # prepare inputs to decoder
            decoder_input_batch = torch.zeros((length_of_traj, 1, state_size))
            decoder_output_batch = torch.zeros((length_of_traj, 1, 1))
            for e, i in enumerate(range(chosen_schedule_start, chosen_schedule_end)):
                decoder_input = self.x[i]
                decoder_truth = self.y[i]
                if torch.cuda.is_available():
                    decoder_input = Variable(
                        torch.Tensor(np.asarray(decoder_input).reshape(1, state_size)).cuda())  # change to 5 to increase batch size
                    decoder_truth = Variable(torch.Tensor(np.asarray(decoder_truth).reshape(1)).cuda().long())
                else:
                    decoder_input = Variable(torch.Tensor(np.asarray(decoder_input).reshape(1, state_size)))
                    decoder_truth = Variable(torch.Tensor(np.asarray(decoder_truth).reshape(1)).long())
                decoder_input_batch[e] = decoder_input
                decoder_output_batch[e] = decoder_truth

            output = self.model.forward_through_decoder(decoder_input_batch, z)

            loss = self.criterion(output.reshape(length_of_traj, 2), decoder_output_batch.reshape(length_of_traj).long())
            loss_array.append(loss.item())

            loss.backward()
            self.opt.step()

            # printing and saving
            if epoch % 1000 == 1:
                print('average loss for last 500: ', np.mean(loss_array[-500:]))
                print('on epoch ', epoch)

            if epoch % 50000 == 49999:
                learning_rate /= 10
                self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
                # torch.save({'nn_state_dict': self.model.state_dict()},
                #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/toy_cVAE' + str(epoch) + '.tar')

                # plt.plot(loss_array)
                # plt.show()
        # torch.save({'nn_state_dict': self.model.state_dict()},
        #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/toy_cVAE.tar')

    def test(self):
        """
        evaluate cVAE
        :return:
        """
        num_schedules = 50
        soft = nn.Softmax(dim=2)
        # create new data
        it_1 = [True, False, False]
        it_2 = [False, True, False]
        it_3 = [False, False, True]
        it = [it_3, it_1, it_2]
        x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, True, train=it[2][0], cv=it[2][1])

        x_test = []
        for each_ele in x_data_test:
            x_test.append(each_ele[2:])

        x_test = torch.Tensor(x_test).reshape(-1, 2)
        y_test = torch.Tensor(y_test).reshape((-1, 1))

        prediction_accuracy = [0]
        percentage_accuracy_top1 = []

        schedule_starts = np.linspace(0, 20 * (self.num_schedules - 1), num=self.num_schedules)

        for i in range(num_schedules):

            chosen_schedule_start = int(schedule_starts[i])
            schedule_num = int(chosen_schedule_start / 20)
            chosen_schedule_end = chosen_schedule_start + 20
            net_input = torch.zeros(20, 1, state_size + num_classes)
            for e, count in enumerate(range(chosen_schedule_start, chosen_schedule_end)):
                if e == 0:
                    pass
                else:
                    # for m,n in enumerate(range(schedule[0], schedule[0] + e)): # does not include current
                    input_nn = x_test[count - 1].clone()
                    input_nn = torch.cat([input_nn, torch.Tensor(one_hot_embedding(y_test[count - 1], 2))], dim=0)
                    net_input[e] = torch.Tensor(input_nn)
                    net_input = Variable(torch.Tensor(net_input)).reshape(20, 1, state_size + num_classes)

                # if False:
                #     pass
                #     # input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242 * 20 + 20)).cuda())  # change to 5 to increase batch size
                #     # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
                # else:
                #     input_nn = Variable(torch.Tensor(input_nn)).reshape(20, 1, 262)
                #     # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

                z = self.model(net_input)

                decoder_input = x_test[count]
                decoder_truth = y_test[count]

                if False:
                    decoder_input = Variable(
                        torch.Tensor(np.asarray(decoder_input).reshape(1, state_size)).cuda())  # change to 5 to increase batch size
                    decoder_truth = Variable(torch.Tensor(np.asarray(decoder_truth).reshape(1)).cuda().long())
                else:
                    decoder_input = Variable(torch.Tensor(np.asarray(decoder_input).reshape(1, 1, state_size)))
                    decoder_truth = Variable(torch.Tensor(np.asarray(decoder_truth).reshape(1)).long())

                pred = self.model.forward_through_decoder(decoder_input, z)
                pred = soft(pred)
                index = torch.argmax(pred).item()

                if index == decoder_truth.item():
                    prediction_accuracy[0] += 1

            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, )

            print('schedule num:', i)

            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            prediction_accuracy = [0]
        print(np.mean(percentage_accuracy_top1))
        print('done')
        # self.save_performance_results(percentage_accuracy_top1, 'toy_results_cVAE')

    def save_performance_results(self, top1, special_string):
        """
        saves performance of top1 and top3
        :return:
        """
        print('top1_mean for is : ', np.mean(top1))
        data = {'top1_mean': np.mean(top1),
                'top1_stderr': np.std(top1) / np.sqrt(len(top1))}
        save_pickle(file=data, file_location='/home/Anonymous/PycharmProjects/bayesian_prolo/base_testing_environment/additions_for_HRI/results',
                    special_string=special_string)


def main():
    """
    main
    :return:
    """
    benchmark = EvalcVAE()
    benchmark.train()
    benchmark.test()


if __name__ == '__main__':
    main()
