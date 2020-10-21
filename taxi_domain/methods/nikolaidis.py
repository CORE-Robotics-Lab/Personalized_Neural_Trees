"""
Created by Anonymous on January 14, 2020
Nikolaidis et. al. benchmark
"""

import torch
import torch.nn.functional as F

# sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')

import numpy as np
from torch.autograd import Variable
from sklearn.cluster import KMeans
# from taxi_domain.methods.train_autoencoder_for_kmeans import Autoencoder, AutoEncoderTrain
# sys.path.insert(0, '../')
import itertools
from taxi_domain.methods.sammut import NNSmall
import os
import pickle

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


# noinspection PyTypeChecker,PyArgumentList
class NNTrain:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self):

        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()
        self.state_dim = 5
        self.output_dim = 3
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_gpu = True
        model1 = NNSmall(self.state_dim, self.output_dim).to(device)
        model2 = NNSmall(self.state_dim, self.output_dim).to(device)
        model3 = NNSmall(self.state_dim, self.output_dim).to(device)

        self.models = [model1, model2, model3]

        opt1 = torch.optim.Adam(self.models[0].parameters(), lr=.001)
        opt2 = torch.optim.Adam(self.models[1].parameters(), lr=.001)
        opt3 = torch.optim.Adam(self.models[2].parameters(), lr=.001)

        self.optimizers = [opt1, opt2, opt3]
        self.when_to_save = 1000

        schedule_matrix_load_directory = '/home/Anonymous/PycharmProjects/bayesian_prolo/taxi_domain/methods/taxi_matrixes.pkl'
        self.matrices = pickle.load(open(schedule_matrix_load_directory, "rb"))

        self.kmeans_model, self.label = self.cluster_matrices(self.matrices)
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

    def cluster_matrices(self, matrices):
        """
        clusters the matrix schedules
        :param matrices:
        :return:
        """
        # vectorize each matrix
        vectorized_set = []
        for i in matrices:
            vectorized = i.reshape(3 * 32, 1)
            vectorized_set.append(vectorized)
        kmeans = KMeans(n_clusters=3, random_state=0)  # random state makes it deterministic
        # Fitting the input data
        new_set = np.hstack(tuple(vectorized_set)).reshape(len(self.states), 3*32)
        kmeans_model = kmeans.fit(np.asarray(new_set))
        labels = kmeans_model.predict(np.asarray(new_set))
        return kmeans_model, labels

    def train(self):
        """
        Trains NN.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """
        epochs = 200000 * 3
        for epoch in range(epochs):

            print('epoch: ', epoch)
            # sample a timestep before the cutoff for cross_validation
            which_user = np.random.choice(range(len(self.states)))
            if which_user in self.indices_of_failed:
                continue
            cluster_num = self.label[which_user]
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

            self.optimizers[cluster_num].zero_grad()
            output = self.models[cluster_num].forward(network_input.float())
            loss = F.cross_entropy(output, action_t)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizers[cluster_num].step()

            if epoch % 10000 == 9999:
                self.evaluate_on_data()
                print('testing accuracies: ', self.testing_accuracies)


    def create_iterables(self):
        """
        adds all possible state combinations
        :return:
        """
        iterables = [[0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1]]
        self.iter_states = []
        for t in itertools.product(*iterables):
            self.iter_states.append(t)

        return self.iter_states


    def pass_in_embedding_out_state_ID(self, states, binary):
        """
        pass in a binary embedding, and itll return the state id
        :param states
        :param binary:
        :return:
        """
        binary_as_tuple = tuple(binary)
        index = states.index(binary_as_tuple)
        return index

    def evaluate_on_data(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """
        # confusion_matrix = np.zeros((20,20))

        iter_states = self.create_iterables()

        states = self.test_states
        actions = self.test_actions
        indices_of_failed = self.test_indices_of_failed
        accuracies = []

        mean_input = [0.4269609,  0.5730391,  0.,         0.4766131,  1.56848165]
        for i in range(len(states)):
            if i in indices_of_failed:
                continue
            accuracy = 0

            length_of_current_game = len(states[i])
            current_schedule_matrix = np.zeros((32, 3))
            for j in range(length_of_current_game):

                if current_schedule_matrix.sum() == 0:
                    cluster_num = self.kmeans_model.predict(current_schedule_matrix.reshape(1, -1))
                else:
                    matrix = np.divide(current_schedule_matrix, current_schedule_matrix.sum())
                    cluster_num = self.kmeans_model.predict(matrix.reshape(1, -1))

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
                output = self.models[int(cluster_num)].forward(input_nn.float())

                index = torch.argmax(output).item()

                if index == truth.item():
                    accuracy += 1

                # update matrix

                embedding_copy = np.zeros((1, 5))
                input_element = input_nn
                for z, each_element in enumerate(mean_input):
                    if each_element > input_element[0][z].item():
                        embedding_copy[0][z] = 0
                    else:
                        embedding_copy[0][z] = 1
                index = self.pass_in_embedding_out_state_ID(iter_states, embedding_copy[0])
                action = truth.item()
                current_schedule_matrix[index][int(action)] += 1

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
