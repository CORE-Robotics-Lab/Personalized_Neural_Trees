"""
Created by Rohan Paleja on Sep 7, 2019
This is an implementation of the paper Learning a Multi-Modal Policy via Imitating Demonstrations with Mixed Behaviors
"""

import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
import numpy as np
from torch.autograd import Variable
# from utils.global_utils import save_pickle
import pickle
import os

# import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)
device = torch.device("cpu")
epochs = 1000000
state_size = 5


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
        self.fc_att = nn.Linear(430, 43)  # 2 for bidirection # 43 is length of traj
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

        post_att = self.fc_att(out.reshape(1, 430))
        post_att = self.soft(self.tanh1(post_att))
        out = out.reshape((43, 10)) * post_att.reshape((43, 1))
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
        self.fc1 = nn.Linear(5 + n_embedding, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(32, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, 3)
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


# noinspection PyArgumentList,PyTypeChecker
class EvalcVAE:
    """
    class that handles training and evaluating this approach.
    Some helpful links
    - https://www.kaggle.com/robertke94/pytorch-bi-lstm-attention
    - https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
    """

    def __init__(self):
        input_size = 8  # 5 for state, 3 extra for action
        hidden_size = 10
        num_layers = 1
        num_classes = 20
        n_embedding = 10
        self.model = cVAE(input_size, hidden_size, num_layers, num_classes, n_embedding)

        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.001)
        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()

        print(self.model.state_dict())
        self.use_gpu = False

        self.training_accuracies = []
        self.testing_accuracies = []
        self.testing_stds = []

    @staticmethod
    def load_in_data():
        """
        loads in train data
        :return:
        """
        states, actions, failed_list, mturkcodes = pickle.load(open(os.path.join('../datasets/', 'testing_data_from_all_users_2.pkl'), 'rb'))

        indices_of_failed = []
        for i in failed_list:
            if i[0] not in indices_of_failed:
                indices_of_failed.append(i[0])

        return states, actions, failed_list, mturkcodes, indices_of_failed

    @staticmethod
    def load_in_test_data():
        """
        loads in test data
        :return:
        """
        states, actions, failed_list, mturkcodes = pickle.load(open(os.path.join('../datasets/', 'training_data_from_all_users_2.pkl'), 'rb'))

        indices_of_failed = []
        for i in failed_list:
            if i[0] not in indices_of_failed:
                indices_of_failed.append(i[0])

        return states, actions, failed_list, mturkcodes, indices_of_failed

    def train(self):
        loss_array = []
        learning_rate = .001
        length_of_max_traj = 43
        for epoch in range(epochs):

            print('epoch: ', epoch)
            # sample a timestep before the cutoff for cross_validation
            which_user = np.random.choice(range(len(self.states)))
            if which_user in self.indices_of_failed:
                continue

            states = self.states[which_user]
            actions = self.actions[which_user]
            # input preparation
            length_of_traj = np.random.randint(1, len(states))

            encoder_input = torch.zeros(length_of_max_traj, 1, 8)

            # create trajectory is the form of [[s,a],[s,a]...]
            for i in range(length_of_traj):
                input_nn = states[i].copy()
                input_nn.extend(one_hot_embedding(actions[i], 3))
                input_nn = torch.Tensor(np.asarray(input_nn))
                encoder_input[i] = input_nn

            # sadly, no gpu
            if torch.cuda.is_available():
                pass
                # input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242 * 20 + 20)).cuda())  # change to 5 to increase batch size
                # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
            else:
                encoder_input = Variable(torch.Tensor(encoder_input)).reshape(length_of_max_traj, 1, 8)
                # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

            # get the "discrete" embedding
            z = self.model(encoder_input)
            self.opt.zero_grad()

            # prepare inputs to decoder
            decoder_input_batch = torch.zeros((length_of_traj, 1, 5))
            decoder_output_batch = torch.zeros((length_of_traj, 1, 1))
            for e, i in enumerate(range(length_of_traj)):
                decoder_input = states[e]
                decoder_truth = actions[e]
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

            loss = self.criterion(output.reshape(length_of_traj, 3), decoder_output_batch.reshape(length_of_traj).long())
            loss_array.append(loss.item())

            loss.backward()
            self.opt.step()

            # printing and saving
            if epoch % 10000 == 1:
                print('average loss for last 500: ', np.mean(loss_array[-500:]))
                print('on epoch ', epoch)
                self.test()
                print('testing accuracies: ', self.testing_accuracies)

            if epoch % 200000 == 199999:
                learning_rate /= 10
                self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def test(self):
        """
        evaluate cVAE
        :return:
        """

        soft = nn.Softmax(dim=2)
        # load in new data

        states = self.test_states
        actions = self.test_actions
        indices_of_failed = self.test_indices_of_failed
        accuracies = []

        for i in range(len(states)):
            if i in indices_of_failed:
                continue
            accuracy = 0
            length_of_current_game = len(states[i])
            net_input = torch.zeros(43, 1, 8)
            for e in range(length_of_current_game):
                if e == 0:
                    pass
                else:
                    # for m,n in enumerate(range(schedule[0], schedule[0] + e)): # does not include current
                    input_nn = []
                    input_nn.append(states[i][e - 1].copy())
                    input_nn[-1].extend(one_hot_embedding(actions[i][e - 1], 3))
                    net_input[e] = torch.Tensor(input_nn)
                    net_input = Variable(torch.Tensor(net_input)).reshape(43, 1, 8)

                # if torch.cuda.is_available():
                #     pass
                #     # input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242 * 20 + 20)).cuda())  # change to 5 to increase batch size
                #     # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
                # else:
                #     input_nn = Variable(torch.Tensor(input_nn)).reshape(20, 1, 262)
                #     # truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

                z = self.model(net_input)

                decoder_input = states[i][e]
                decoder_truth = actions[i][e]

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

                if index == decoder_truth.item():
                    accuracy += 1
            accuracies.append(accuracy / length_of_current_game)

        self.testing_accuracies.append(np.mean(accuracies))
        self.testing_stds.append(np.std(accuracies)/len(accuracies))



def main():
    """
    main
    :return:
    """
    benchmark = EvalcVAE()
    benchmark.train()
    benchmark.test()
    print('testing accuracies', benchmark.testing_accuracies)
    print('max val: ', np.max(benchmark.testing_accuracies), ' std', benchmark.testing_stds[int(np.argmax(benchmark.testing_accuracies))])



if __name__ == '__main__':
    main()
