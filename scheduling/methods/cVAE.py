"""
Created by Rohan Paleja on Sep 7, 2019
This is an implementation of the paper Learning a Multi-Modal Policy via Imitating Demonstrations with Mixed Behaviors
"""

import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
from utils.naive_utils import load_in_naive_data, find_which_schedule_this_belongs_to
import numpy as np
from torch.autograd import Variable
from utils.global_utils import save_pickle

# import matplotlib.pyplot as plt

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")
epochs = 1000000
state_size = 242
learning_rate = .001


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels].tolist()


# Bidirectional recurrent neural network (many-to-one)
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
        self.soft = nn.Softmax()

        # network for predicting action
        self.fc1 = nn.Linear(242 + n_embedding, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(32, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, 20)
        self.soft = nn.Softmax(dim=0)

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
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc21(x)
        x = self.relu21(x)
        x = self.fc22(x)
        x = self.relu22(x)
        x = self.fc23(x)
        x = self.relu23(x)
        x = self.fc3(x)
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
        input_size = 262  # 242 * 20 for state, 20 extra for action
        hidden_size = 10
        num_layers = 1
        num_classes = 20
        n_embedding = 10
        self.model = cVAE(input_size, hidden_size, num_layers, num_classes, n_embedding)

        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.001)
        self.num_schedules = 150
        self.num_test_schedules = 100
        # load in data
        self.X_train_naive, self.Y_train_naive, self.schedule_array_train_naive, self.X_test_naive, self.Y_test_naive, self.schedule_array_test_naive = load_in_naive_data(
            250, 250)
        self.X_train_naive, \
        self.Y_train_naive, \
        self.schedule_array_train_naive, = self.sample_data(150)

        self.X_test_naive, \
        self.Y_test_naive, \
        self.schedule_array_test_naive, = self.sample_test_data(100)
        self.use_gpu = False

    def sample_data(self, size):
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250 - size)
        self.sample_min = set_of_twenty * 20
        return self.X_train_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.Y_train_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.schedule_array_train_naive[set_of_twenty:set_of_twenty + size]

    def sample_test_data(self, size):
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250 - size)
        self.sample_test_min = set_of_twenty * 20
        return self.X_test_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.Y_test_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.schedule_array_test_naive[set_of_twenty:set_of_twenty + size]



    def train(self):
        loss_array = []
        learning_rate = .001
        for epoch in range(epochs):

            # choose a random schedule
            rand_timestep_within_sched = np.random.randint(len(self.X_train_naive))
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array_train_naive, rand_timestep_within_sched+self.sample_min)

            # input preparation
            length_of_traj = np.random.randint(1, 21)

            start_of_chosen_sched = self.schedule_array_train_naive[which_schedule][0]-self.sample_min
            end_of_chosen_sched = self.schedule_array_train_naive[which_schedule][0] -self.sample_min+ length_of_traj
            encoder_input = torch.zeros(20, 1, 262)

            # create trajectory is the form of [[s,a],[s,a]...]
            for i in range(start_of_chosen_sched, end_of_chosen_sched):
                input_nn = self.X_train_naive[i].copy()
                input_nn.extend(one_hot_embedding(self.Y_train_naive[i], 20))
                input_nn = torch.Tensor(np.asarray(input_nn))
                encoder_input[i - start_of_chosen_sched] = input_nn

            # sadly, no gpu
            if self.use_gpu:
                pass
                # input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242 * 20 + 20)).cuda())  # change to 5 to increase batch size
                # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
            else:
                encoder_input = Variable(torch.Tensor(encoder_input)).reshape(20, 1, 262)
                # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

            # get the "discrete" embedding
            z = self.model(encoder_input)
            self.opt.zero_grad()

            # prepare inputs to decoder
            decoder_input_batch = torch.zeros((length_of_traj, 1, 242))
            decoder_output_batch = torch.zeros((length_of_traj, 1, 1))
            for e, i in enumerate(range(start_of_chosen_sched, end_of_chosen_sched)):
                decoder_input = self.X_train_naive[i]
                decoder_truth = self.Y_train_naive[i]
                if self.use_gpu:
                    decoder_input = Variable(
                        torch.Tensor(np.asarray(decoder_input).reshape(1, state_size)).cuda())  # change to 5 to increase batch size
                    decoder_truth = Variable(torch.Tensor(np.asarray(decoder_truth).reshape(1)).cuda().long())
                else:
                    decoder_input = Variable(torch.Tensor(np.asarray(decoder_input).reshape(1, state_size)))
                    decoder_truth = Variable(torch.Tensor(np.asarray(decoder_truth).reshape(1)).long())
                decoder_input_batch[e] = decoder_input
                decoder_output_batch[e] = decoder_truth

            output = self.model.forward_through_decoder(decoder_input_batch, z)

            loss = self.criterion(output.reshape(length_of_traj, 20), decoder_output_batch.reshape(length_of_traj).long())
            loss_array.append(loss.item())

            loss.backward()
            self.opt.step()

            # printing and saving
            if epoch % 1000 == 1:
                print('average loss for last 500: ', np.mean(loss_array[-500:]))
                print('on epoch ', epoch)

            if epoch % 200000 == 199999:
                learning_rate /= 10
                self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
                # torch.save({'nn_state_dict': self.model.state_dict()},
                #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/cVAE' + str(epoch) + '.tar')

                # plt.plot(loss_array)
                # plt.show()
        # torch.save({'nn_state_dict': self.model.state_dict()},
        #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/cVAE.tar')

    def test(self):
        """
        evaluate cVAE
        :return:
        """

        soft = nn.Softmax(dim=2)
        # load in new data

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for i, schedule in enumerate(self.schedule_array_test_naive):
            net_input = torch.zeros(20, 1, 262)
            for e, count in enumerate(range(schedule[0]-self.sample_test_min, schedule[1]-self.sample_test_min + 1)):
                if e == 0:
                    pass
                else:
                    # for m,n in enumerate(range(schedule[0], schedule[0] + e)): # does not include current
                    input_nn = []
                    input_nn.append(self.X_test_naive[count - 1].copy())
                    input_nn[-1].extend(one_hot_embedding(self.Y_test_naive[count - 1], 20))
                    net_input[e] = torch.Tensor(input_nn)
                    net_input = Variable(torch.Tensor(net_input)).reshape(20, 1, 262)

                # if torch.cuda.is_available():
                #     pass
                #     # input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242 * 20 + 20)).cuda())  # change to 5 to increase batch size
                #     # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
                # else:
                #     input_nn = Variable(torch.Tensor(input_nn)).reshape(20, 1, 262)
                #     # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

                z = self.model(net_input)

                decoder_input = self.X_test_naive[count]
                decoder_truth = self.Y_test_naive[count]

                if self.use_gpu:
                    decoder_input = Variable(
                        torch.Tensor(np.asarray(decoder_input).reshape(1, state_size)).cuda())  # change to 5 to increase batch size
                    decoder_truth = Variable(torch.Tensor(np.asarray(decoder_truth).reshape(1)).cuda().long())
                else:
                    decoder_input = Variable(torch.Tensor(np.asarray(decoder_input).reshape(1, 1, state_size)))
                    decoder_truth = Variable(torch.Tensor(np.asarray(decoder_truth).reshape(1)).long())

                pred = self.model.forward_through_decoder(decoder_input, z)
                pred = soft(pred)
                index = torch.argmax(pred).item()

                # top 3
                _, top_three = torch.topk(pred, 3)

                if index == decoder_truth.item():
                    prediction_accuracy[0] += 1

                if decoder_truth.item() in top_three.detach().cpu().tolist()[0][0]:
                    prediction_accuracy[1] += 1
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)

            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            prediction_accuracy = [0, 0]
        return np.mean(percentage_accuracy_top1)
        # self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_cVAE')

    def save_performance_results(self, top1, top3, special_string):
        """
        saves performance of top1 and top3
        :return:
        """
        print('top1_mean for is : ', np.mean(top1))
        data = {'top1_mean': np.mean(top1),
                'top3_mean': np.mean(top3),
                'top1_stderr': np.std(top1) / np.sqrt(len(top1)),
                'top3_stderr': np.std(top3) / np.sqrt(len(top3))}
        save_pickle(file=data, file_location='/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/results',
                    special_string=special_string)


def main():
    """
    main
    :return:
    """
    res = []
    for i in range(3):
        benchmark = EvalcVAE()
        benchmark.train()
        out = benchmark.test()
        res.append(out)
    print(np.mean(res))
    print(np.std(res))

if __name__ == '__main__':
    main()
