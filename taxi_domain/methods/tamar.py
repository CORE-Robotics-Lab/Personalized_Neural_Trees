"""
Created by Anonymous on January 14, 2020
This is an implementation of the paper IMITATION LEARNING FROM VISUAL DATA WITH MULTIPLE INTENTIONS
Tamar et. al.
"""

import torch
import torch.nn as nn
# from torch.distributions import RelaxedOneHotCategorical
import numpy as np
from torch.autograd import Variable
# from utils.global_utils import save_pickle
# import matplotlib.pyplot as plt
# import heapq
import os
import pickle

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 100000


class SNN(nn.Module):
    """
    standard MLP
    """

    def __init__(self, state_size):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(state_size + 1, 128)
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

    def forward(self, x, z):
        """
        forward pass
        :param x: state
        :param z: random variable
        :return:
        """
        x = torch.cat([x, z.expand(x.shape[0], 1, 1)], dim=2)
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

        return x


# noinspection PyArgumentList
class EvalSNN:
    """
    class that handles training and evaluating this approach
    """

    def __init__(self):
        self.state_dim = 5
        self.output_dim = 3
        self.model = SNN(self.state_dim)

        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.001)

        # load in data
        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()

        self.priority_queue = []
        self.use_gpu = True
        self.testing_accuracies = []
        self.testing_stds = []
        print(self.model.state_dict())
        # checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/SNN999999.tar')
        # self.model.load_state_dict(checkpoint['nn_state_dict'])

    @staticmethod
    def load_in_data():
        """
        loads in train data
        :return:
        """
        states, actions, failed_list, mturkcodes = pickle.load(open(os.path.join('../', 'testing_data_from_all_users_2.pkl'), 'rb'))

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
        states, actions, failed_list, mturkcodes = pickle.load(open(os.path.join('../', 'training_data_from_all_users_2.pkl'), 'rb'))

        indices_of_failed = []
        for i in failed_list:
            if i[0] not in indices_of_failed:
                indices_of_failed.append(i[0])

        return states, actions, failed_list, mturkcodes, indices_of_failed

    def find_max_traj_length(self):
        max_length = 0
        for i in self.states:
            game_length = len(i)
            if game_length > max_length:
                max_length = game_length

        return max_length

    def train(self):
        """
        Train the network
        :return:
        """
        loss_array = []
        sampling_N = 5
        length_of_longest_game = self.find_max_traj_length()
        for epoch in range(epochs):

            print('epoch: ', epoch)
            # sample a timestep before the cutoff for cross_validation
            which_user = np.random.choice(range(len(self.states)))
            if which_user in self.indices_of_failed:
                continue

            states = self.states[which_user]
            actions = self.actions[which_user]
            length_of_current_game = len(states)


            # sample z
            sampled_z = np.random.rand(sampling_N)

            # prepare network input
            network_input = torch.zeros(length_of_current_game, 1, 5)
            network_truth = torch.zeros(length_of_current_game, 1, 1)

            z_loss_array = []
            self.opt.zero_grad()

            # load in network input
            for e, i in enumerate(states):
                state_t = states[e]
                action_t = actions[e]
                # iterate over pairwise comparisons
                if self.use_gpu:
                    network_in = torch.tensor(state_t).reshape(1, 5).cuda()
                    action_t = torch.tensor(action_t).reshape(1).cuda()
                else:
                    network_in = torch.tensor(state_t)
                    action_t = torch.tensor(action_t).reshape(1)

                network_input[e] = network_in

                truth = action_t
                network_truth[e] = truth

            # find chosen_z
            for z in sampled_z:
                network_output = self.model.forward(network_input.float(), torch.tensor(z))
                loss = self.criterion(network_output.reshape(length_of_current_game, 3), network_truth.reshape(length_of_current_game).long())
                z_loss_array.append(loss.item())
                self.priority_queue.append((loss.item(), z))

            # use lowest z to update network
            lowest_loss_ind = np.argmin(z_loss_array)
            network_output = self.model.forward(network_input.float(), torch.tensor(sampled_z[lowest_loss_ind]))
            loss = self.criterion(network_output.reshape(length_of_current_game, 3), network_truth.reshape(length_of_current_game).long())
            loss.backward()
            loss_array.append(loss.item())
            self.opt.step()

            # print and save
            if epoch % 1000 == 1:
                print('average loss for last 500: ', np.mean(loss_array[-500:]))
                print('on epoch ', epoch)
                self.test()
                print('testing accuracies: ', self.testing_accuracies)

            if epoch % 200000 == 199999:
                self.opt = torch.optim.SGD(self.model.parameters(), lr=.0001)

    # noinspection PyTypeChecker
    def test(self):
        """
        evaluate the network
        :return:
        """
        soft = nn.Softmax(dim=2)
        self.priority_queue.sort()
        # heapq.heapify(self.priority_queue)
        sampling_N = 5

        states = self.test_states
        actions = self.test_actions
        indices_of_failed = self.test_indices_of_failed
        accuracies = []
        set_of_zs = []
        for j, k in enumerate(self.priority_queue):
            if j >= 100:
                continue
            else:
                set_of_zs.append(k[1])

        for i in range(len(states)):
            if i in indices_of_failed:
                continue
            accuracy = 0
            length_of_current_game = len(states[i])

            trajectory = []
            trajectory_truths = []
            for j in range(length_of_current_game):

                if len(trajectory) == 0:
                    chosen_z = np.random.choice(set_of_zs)
                else:
                    chosen_set_of_zs = np.random.choice(set_of_zs, size=sampling_N)
                    z_loss_array = []

                    network_input = torch.zeros(len(trajectory), 1, 5)
                    network_truth = torch.zeros(len(trajectory), 1, 1)

                    for e, l in enumerate(trajectory):
                        input_nn = torch.Tensor(np.asarray(l)).clone()
                        network_input[e] = input_nn

                        truth = trajectory_truths[e]
                        truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).long())
                        network_truth[e] = truth


                    for z in chosen_set_of_zs:
                        network_output = self.model.forward(network_input, torch.tensor(z))
                        loss = self.criterion(network_output.reshape(len(trajectory), 3), network_truth.reshape(len(trajectory)).long())
                        z_loss_array.append(loss.item())

                    chosen_z = set_of_zs[np.argmin(z_loss_array)]

                trajectory.append(states[i][j])
                trajectory_truths.append(actions[i][j])
                state_t = states[i][j]
                action_t = actions[i][j]
                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, 1, self.state_dim)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, 1, self.state_dim)))
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)))
                pred = self.model.forward(input_nn.cpu(), torch.tensor(chosen_z))

                pred = soft(pred)
                index = torch.argmax(pred).item()

                if index == truth.item():
                    accuracy += 1
            accuracies.append(accuracy / length_of_current_game)

        self.testing_accuracies.append(np.mean(accuracies))
        self.testing_stds.append(np.std(accuracies)/len(accuracies))


def main():
    """
    main
    :return:
    """
    benchmark = EvalSNN()
    benchmark.train()
    benchmark.test()
    print('testing accuracies', benchmark.testing_accuracies)
    print('max val: ', np.max(benchmark.testing_accuracies), ' std', benchmark.testing_stds[int(np.argmax(benchmark.testing_accuracies))])


if __name__ == '__main__':
    main()
