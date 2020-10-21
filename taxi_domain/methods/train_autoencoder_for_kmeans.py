"""
trains autoencoder
"""

import torch
import sys
import torch.nn as nn
import pickle
import os
# sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')

import numpy as np
from torch.autograd import Variable
from utils.global_utils import save_pickle
import itertools

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


class Autoencoder(nn.Module):
    """
    autoencoder torch model
    """

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(242, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 11),

        )

        self.decoder = nn.Sequential(
            nn.Linear(11, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 242)
        )

    def forward(self, x):
        """
        forward pass
        :param x:
        :return:
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_only_encoding(self, x):
        """
        produce encoding
        :param x:
        :return:
        """
        z = self.encoder(x)
        return z


class AutoEncoderTrain:
    """
    create and train the autoencoder
    """

    def __init__(self):


        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()


        self.mean_embedding = None
        self.embedding_np = None
        self.matrixes = None
        self.total_binary_embeddings = None
        self.counter_splits = []
        # self.states = None


    @staticmethod
    def load_in_data():
        """
        loads in train data
        :return:
        """
        states, actions, failed_list, mturkcodes = pickle.load(open(os.path.join('../datasets/', 'training_data_from_all_users.pkl'), 'rb'))

        indices_of_failed = []
        for i in failed_list:
            if i[0] not in indices_of_failed:
                indices_of_failed.append(i[0])

        return states, actions, failed_list, mturkcodes, indices_of_failed



    # noinspection PyArgumentList
    def compute_mean(self):
        """
        computes the mean embedding by first computing all embeddings for every step of the schedule,
        adding them to a numpy array and computing the avg
        :return:
        """
        # load_in_all_parameters(self.save_directory, self.auto_encoder)
        counter = 0
        for i, data_row in enumerate(self.states):
            for e, data in enumerate(data_row):
                input_nn = data

                prediction_embedding = input_nn
                print(prediction_embedding)
                if counter == 0:
                    self.embedding_np = prediction_embedding
                else:
                    self.embedding_np = np.vstack((self.embedding_np, prediction_embedding))

                counter += 1

        self.mean_embedding = np.average(self.embedding_np, axis=0)
        print('mean embedding is ', self.mean_embedding)

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

    # noinspection PyArgumentList
    def round_each_encoding_and_create_array(self):
        """
        rounds each encoding by comparing it to the mean, and then stacks these in an array
        :return:
        """
        self.total_binary_embeddings = np.zeros((0))
        counter = 0
        for i, data_row in enumerate(self.states):
            self.counter_splits.append(counter)
            for e, data in enumerate(data_row):

                prediction_embedding = data

                embedding_copy = np.zeros((1, 5))

                for j, each_element in enumerate(self.mean_embedding):
                    if each_element > prediction_embedding[j]:
                        embedding_copy[0][j] = 0
                    else:
                        embedding_copy[0][j] = 1

                if counter == 0:
                    self.total_binary_embeddings = embedding_copy
                else:
                    self.total_binary_embeddings = np.vstack((self.total_binary_embeddings, embedding_copy))

                counter += 1
            # This should generate n schedules of binary data
        print('finished turning all elements of schedule into binary')

    def pass_in_embedding_out_state_ID(self, binary):
        """
        pass in a binary embedding, and itll return the state id
        :param binary:
        :return:
        """
        binary_as_tuple = tuple(binary)
        index = self.iter_states.index(binary_as_tuple)
        return index

    def populate_a_matrix_per_schedule(self):
        """
        creates matrixes bases on the binary embeddings
        :return:
        """
        self.matrixes = []
        for i in range(len(self.states)):
            m = np.zeros((32, 3))
            self.matrixes.append(m)
        for i, each_matrix in enumerate(self.matrixes):
            # lets look at elements of schedule 1
            for j in range(len(self.states[i])):
                binary_embedding = self.total_binary_embeddings[j]
                index = self.pass_in_embedding_out_state_ID(binary_embedding)
                # action taken at this instance
                action = self.actions[i][j]
                each_matrix[index][action] += 1
            total_sum = each_matrix.sum()
            self.matrixes[i] = np.divide(each_matrix, total_sum)

        print('n matrices have been generated')

    # def cluster_matrixes(self):
    #     # vectorize each matrix
    #     vectorized_set = []
    #     for i in self.matrixes:
    #         vectorized = i.reshape(20 * 2048, 1)
    #         vectorized_set.append(vectorized)
    #     kmeans = KMeans(n_clusters=3)
    #     # Fitting the input data
    #     new_set = np.hstack(tuple(vectorized_set)).reshape(self.num_schedules, 20 * 2048)
    #     self.kmeans = kmeans.fit(np.asarray(new_set))
    #     self.label = self.kmeans.predict(np.asarray(new_set))

    def save_matrixes(self):
        """
        saves the matrixes so these can be used to cluster in the gmm etc.
        :return:
        """
        save_pickle('/home/Anonymous/PycharmProjects/bayesian_prolo/taxi_domain/methods', self.matrixes, 'taxi_matrixes.pkl')


def main():
    """
    entry point for file
    :return:
    """
    trainer = AutoEncoderTrain()
    trainer.compute_mean()
    trainer.create_iterables()

    trainer.round_each_encoding_and_create_array()
    trainer.populate_a_matrix_per_schedule()
    trainer.save_matrixes()


if __name__ == '__main__':
    main()
