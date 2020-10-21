"""
Created by Anonymous on January 13, 2020
Performance of Sammut et. al. benchmark in taxi domain
"""


import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle


sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')
import numpy as np
from torch.autograd import Variable
# sys.path.insert(0, '../')

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


class NNSmall(nn.Module):
    """
    Sammut et. al. benchmark
    """

    def __init__(self, state_dim, output_dim):
        super(NNSmall, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(128, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_dim)
        self.soft = nn.LogSoftmax(dim=0)

    def forward(self, x):
        """
        forward pass
        :param x: i_minus_j or vice versa
        :return:
        """
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


# noinspection PyTypeChecker,PyArgumentList
class NNTrain:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self):
        self.total_loss_array = []
        self.state_dim = 5
        self.output_dim = 3
        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNSmall(self.state_dim, self.output_dim).to(device)

        print(self.model.state_dict())
        self.opt = torch.optim.Adam(self.model.parameters())
        self.use_gpu = True

        self.training_accuracies = []
        self.testing_accuracies = []
        self.testing_stds = []

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

    def train(self):
        """
        Trains NN.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """

        epochs = 400000
        for epoch in range(epochs):

            print('epoch: ', epoch)
            # sample a timestep before the cutoff for cross_validation
            which_user = np.random.choice(range(len(self.states)))
            if which_user in self.indices_of_failed:
                continue

            states = self.states[which_user]
            actions = self.actions[which_user]
            length_of_current_game = len(states)

            timestep = np.random.choice(range(len(self.states[which_user])))

            # input

            state_t = states[timestep]
            action_t = actions[timestep]

            # iterate over pairwise comparisons
            if self.use_gpu:
                network_input = torch.tensor(state_t).reshape(1, 5).cuda()
                action_t = torch.tensor(action_t).reshape(1).cuda()
            else:
                network_input = torch.tensor(state_t)
                action_t = torch.tensor(action_t).reshape(1)


            self.opt.zero_grad()
            output = self.model.forward(network_input.float())
            loss = F.cross_entropy(output, action_t)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()

            if epoch % 1000 == 999:
                self.evaluate_on_data()
                print('testing accuracies: ', self.testing_accuracies)


    def evaluate_on_data(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """

        states = self.test_states
        actions = self.test_actions
        indices_of_failed = self.test_indices_of_failed
        accuracies = []

        for i in range(len(states)):
            if i in indices_of_failed:
                continue
            accuracy = 0

            length_of_current_game = len(states[i])

            for j in range(length_of_current_game):

                # input
                state_t = states[i][j]
                action_t = actions[i][j]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)))
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)))

                # forward
                output = self.model.forward(input_nn.float())

                index = torch.argmax(output).item()


                if index == truth.item():
                    accuracy += 1

            accuracies.append(accuracy / length_of_current_game)

        self.testing_accuracies.append(np.mean(accuracies))
        self.testing_stds.append(np.std(accuracies)/len(accuracies))

def main():
    """
    entry point for file
    :return:
    """

    trainer = NNTrain()
    trainer.train()
    print('testing accuracies', trainer.testing_accuracies)
    print('max val: ', np.max(trainer.testing_accuracies), ' std', trainer.testing_stds[int(np.argmax(trainer.testing_accuracies))])


if __name__ == '__main__':
    main()