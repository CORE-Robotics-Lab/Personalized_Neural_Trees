"""
trains autoencoder
"""

import torch
import sys
import torch.nn as nn

# sys.path.insert(0, '/home/rohanpaleja/PycharmProjects/bayesian_prolo')

import numpy as np
from torch.autograd import Variable
from utils.global_utils import save_pickle
from utils.naive_utils import load_in_naive_data
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

    def __init__(self, num_schedules):

        self.num_schedules = num_schedules
        self.num_test_schedules = 100
        # load_directory = '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/scheduling_dataset/' + str(
        #         self.num_schedules) + 'EDF_DIST_8_28_2019_naive.pkl'
        # self.data = pickle.load(open(load_directory, "rb"))
        # self.X, self.Y, self.schedule_array = create_new_dataset(num_schedules=self.num_schedules, data=self.data)
        # for i, each_element in enumerate(self.X):
        #     self.X[i] = each_element + list(range(20))
        self.X_train_naive, self.Y_train_naive, self.schedule_array_train_naive, self.X_test_naive, self.Y_test_naive, self.schedule_array_test_naive = load_in_naive_data(
            self.num_schedules, self.num_test_schedules)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder().to(device)

        print(self.model.state_dict())
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.0001)
        self.mean_embedding = None
        self.embedding_np = None
        self.matrixes = None
        self.total_binary_embeddings = None
        self.states = None

    # noinspection PyArgumentList
    def train(self):
        """
        Trains an autoencoder to represent each timestep of the schedule
        :return:
        """
        loss_func = torch.nn.MSELoss()
        training_done = False
        total_loss_array = []
        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X_train_naive))
            input_nn = self.X_train_naive[rand_timestep_within_sched]

            # iterate over pairwise comparisons
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())
                truth_nn = input_nn.clone()
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                truth_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))

            self.opt.zero_grad()
            output = self.model.forward(input_nn)

            loss = loss_func(output, truth_nn)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()

            total_loss_array.append(loss.item())

            total_iterations = len(total_loss_array)

            if total_iterations % 1000 == 999:
                print('current timestep:', total_iterations, 'avg loss for last 500: ', np.mean(total_loss_array[-500:]))
                torch.save({'nn_state_dict': self.model.state_dict()},
                           '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/Autoencoder' + str(self.num_schedules) + '.tar')

            if total_iterations > 2000000:
                training_done = True
                torch.save({'nn_state_dict': self.model.state_dict()},
                           '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/Autoencoder' + str(self.num_schedules) + '.tar')

    # noinspection PyArgumentList
    def compute_mean(self):
        """
        computes the mean embedding by first computing all embeddings for every step of the schedule,
        adding them to a numpy array and computing the avg
        :return:
        """
        # load_in_all_parameters(self.save_directory, self.auto_encoder)
        for i, data_row in enumerate(self.X_train_naive):
            input_nn = data_row
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))

            prediction_embedding = self.model.forward_only_encoding(input_nn)
            print(prediction_embedding)
            if i == 0:
                self.embedding_np = prediction_embedding.data.clone().cpu().numpy()[0]
            else:
                self.embedding_np = np.vstack((self.embedding_np, prediction_embedding.data.clone().cpu().numpy()[0]))
        self.mean_embedding = np.average(self.embedding_np, axis=0)
        print('mean embedding is ', self.mean_embedding)

    def create_iterables(self):
        """
        adds all possible state combinations
        :return:
        """
        iterables = [[0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1]]
        self.states = []
        for t in itertools.product(*iterables):
            self.states.append(t)

    # noinspection PyArgumentList
    def round_each_encoding_and_create_array(self):
        """
        rounds each encoding by comparing it to the mean, and then stacks these in an array
        :return:
        """
        self.total_binary_embeddings = np.zeros((0))
        for counter, data_row in enumerate(self.X_train_naive):
            input_nn = data_row
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))

            prediction_embedding = self.model.forward_only_encoding(input_nn)

            embedding_copy = np.zeros((1, 11))

            for i, each_element in enumerate(self.mean_embedding):
                if each_element > prediction_embedding.data[0][i].item():
                    embedding_copy[0][i] = 0
                else:
                    embedding_copy[0][i] = 1

            if counter == 0:
                self.total_binary_embeddings = embedding_copy
            else:
                self.total_binary_embeddings = np.vstack((self.total_binary_embeddings, embedding_copy))

            # This should generate n schedules of binary data
        print('finished turning all elements of schedule into binary')

    def pass_in_embedding_out_state_ID(self, binary):
        """
        pass in a binary embedding, and itll return the state id
        :param binary:
        :return:
        """
        binary_as_tuple = tuple(binary)
        index = self.states.index(binary_as_tuple)
        return index

    def populate_a_matrix_per_schedule(self):
        """
        creates matrixes bases on the binary embeddings
        :return:
        """
        self.matrixes = []
        for i in range(self.num_schedules):
            m = np.zeros((2048, 20))
            self.matrixes.append(m)
        for i, each_matrix in enumerate(self.matrixes):
            # lets look at elements of schedule 1
            for j in range(self.schedule_array_train_naive[i][0], self.schedule_array_train_naive[i][1] + 1):
                binary_embedding = self.total_binary_embeddings[j]
                index = self.pass_in_embedding_out_state_ID(binary_embedding)
                # action taken at this instance
                action = self.Y_train_naive[j]
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
        save_pickle('/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/results', self.matrixes, str(self.num_schedules) + 'matrixes.pkl')


def main():
    """
    entry point for file
    :return:
    """
    num_schedules = 150
    trainer = AutoEncoderTrain(num_schedules)
    trainer.train()
    trainer.compute_mean()
    trainer.create_iterables()

    trainer.round_each_encoding_and_create_array()
    trainer.populate_a_matrix_per_schedule()
    trainer.save_matrixes()


if __name__ == '__main__':
    main()
