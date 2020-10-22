"""
Created by Rohan Paleja on September 23, 2019
Nikolaidis et. al. benchmark
"""

import torch
import torch.nn.functional as F

# sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')

import numpy as np
import pickle
from torch.autograd import Variable
from utils.naive_utils import load_in_naive_data, find_which_schedule_this_belongs_to
from utils.hri_utils import save_performance_results
from sklearn.cluster import KMeans
from scheduling.methods.train_autoencoder import Autoencoder, AutoEncoderTrain
# sys.path.insert(0, '../')
import itertools

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
from scheduling.methods.NN_naive import NNSmall


# noinspection PyTypeChecker,PyArgumentList
class NNTrain:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self):
        self.num_schedules = 150
        self.num_test_schedules = 100
        self.total_loss_array = []

        self.X_train_naive, self.Y_train_naive, self.schedule_array_train_naive, self.X_test_naive, self.Y_test_naive, self.schedule_array_test_naive = load_in_naive_data(
            250,250)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model1 = NNSmall().to(device)
        model2 = NNSmall().to(device)
        model3 = NNSmall().to(device)

        self.models = [model1, model2, model3]

        opt1 = torch.optim.Adam(self.models[0].parameters(), lr=.001)
        opt2 = torch.optim.Adam(self.models[1].parameters(), lr=.001)
        opt3 = torch.optim.Adam(self.models[2].parameters(), lr=.001)

        self.optimizers = [opt1, opt2, opt3]
        self.when_to_save = 1000

        schedule_matrix_load_directory = '/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/results/'+str(self.num_schedules) + 'matrixes.pkl'
        self.matrices = pickle.load(open(schedule_matrix_load_directory, "rb"))

        self.kmeans_model, self.label = self.cluster_matrices(self.matrices, self.num_schedules)
        self.X_train_naive, \
        self.Y_train_naive, \
        self.schedule_array_train_naive,  = self.sample_data(150)

        self.X_test_naive, \
        self.Y_test_naive, \
        self.schedule_array_test_naive,  = self.sample_test_data(100)
        self.num_test_schedules = 100

    def sample_data(self, size):
        # return self.X_train_naive[0:size * 20 * 20], \
        #        self.Y_train_naive[0:size * 20 * 20], \
        #        self.schedule_array_train_naive[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250-size)
        self.sample_min = set_of_twenty * 20
        return self.X_train_naive[set_of_twenty*20:set_of_twenty*20 + size * 20], \
               self.Y_train_naive[set_of_twenty*20:set_of_twenty*20 + size * 20], \
               self.schedule_array_train_naive[set_of_twenty:set_of_twenty+size]

    def sample_test_data(self, size):
        # return self.X_train_naive[0:size * 20 * 20], \
        #        self.Y_train_naive[0:size * 20 * 20], \
        #        self.schedule_array_train_naive[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250-size)
        self.sample_test_min = set_of_twenty * 20
        return self.X_test_naive[set_of_twenty*20:set_of_twenty*20 + size * 20], \
               self.Y_test_naive[set_of_twenty*20:set_of_twenty*20 + size * 20], \
               self.schedule_array_test_naive[set_of_twenty:set_of_twenty+size]


    @staticmethod
    def cluster_matrices(matrices, num_schedules):
        """
        clusters the matrix schedules
        :param matrices:
        :param num_schedules:
        :return:
        """
        # vectorize each matrix
        vectorized_set = []
        for i in matrices:
            vectorized = i.reshape(20 * 2048, 1)
            vectorized_set.append(vectorized)
        kmeans = KMeans(n_clusters=3, random_state=0)  # random state makes it deterministic
        # Fitting the input data
        new_set = np.hstack(tuple(vectorized_set)).reshape(num_schedules, 20 * 2048)
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
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X_train_naive))
            input_nn = self.X_train_naive[rand_timestep_within_sched]
            truth_nn = self.Y_train_naive[rand_timestep_within_sched]
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array_train_naive, rand_timestep_within_sched+self.sample_min)
            cluster_num = self.label[which_schedule]
            # iterate over pairwise comparisons
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

            self.optimizers[cluster_num].zero_grad()
            output = self.models[cluster_num].forward(input_nn)
            loss = F.cross_entropy(output, truth)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizers[cluster_num].step()

            self.total_loss_array.append(loss.item())

            if epoch % 500 == 499:
                print('loss at', epoch, ', total loss (average for each 100, averaged)', np.mean(self.total_loss_array[-100:]))
                # self.save_trained_nets(str(epoch))

    @staticmethod
    def create_iterables():
        """
        adds all possible state combinations
        :return:
        """
        iterables = [[0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1]]
        states = []
        for t in itertools.product(*iterables):
            states.append(t)
        return states

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

    def evaluate_on_test_data(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """
        # confusion_matrix = np.zeros((20,20))

        autoencoder_class = AutoEncoderTrain(self.num_schedules)
        checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/models/Autoencoder150.tar')
        autoencoder_class.model.load_state_dict(checkpoint['nn_state_dict'])
        states = self.create_iterables()

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        mean_input = [1.3277743, 0.32837677, 1.4974482, -1.3519306, -0.64621973, 0.10534518, -2.338118, -2.7345326, 1.7558736, -3.0746384, -3.485554]
        for i, schedule in enumerate(self.schedule_array_test_naive):
            current_schedule_matrix = np.zeros((2048, 20))

            for count in range(schedule[0]-self.sample_test_min, schedule[1]-self.sample_test_min + 1):
                if current_schedule_matrix.sum() == 0:
                    cluster_num = self.kmeans_model.predict(current_schedule_matrix.reshape(1, -1))
                else:
                    matrix = np.divide(current_schedule_matrix, current_schedule_matrix.sum())
                    cluster_num = self.kmeans_model.predict(matrix.reshape(1, -1))

                net_input = self.X_test_naive[count]
                truth = self.Y_test_naive[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))

                # forward
                output = self.models[int(cluster_num)].forward(input_nn)

                index = torch.argmax(output).item()

                # confusion_matrix[truth][index] += 1
                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0]:
                    prediction_accuracy[1] += 1

                    # update matrix

                embedding_copy = np.zeros((1, 11))
                input_element = autoencoder_class.model.forward_only_encoding(input_nn)
                for z, each_element in enumerate(mean_input):
                    if each_element > input_element[0][z].item():
                        embedding_copy[0][z] = 0
                    else:
                        embedding_copy[0][z] = 1
                index = self.pass_in_embedding_out_state_ID(states, embedding_copy[0])
                action = truth.item()
                current_schedule_matrix[index][int(action)] += 1

            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)

            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            prediction_accuracy = [0, 0]

        print(np.mean(percentage_accuracy_top1))
        # save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'kmeans_to_NN_naive')
        return np.mean(percentage_accuracy_top1)

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn1_state_dict': self.models[0].state_dict(),
                    'nn2_state_dict': self.models[1].state_dict(),
                    'nn3_state_dict': self.models[2].state_dict()},
                   '/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/models/k_means_NN_' + name + '.tar')


def main():
    """
    entry point for file
    :return:
    """

    res = []
    for i in range(3):
        trainer = NNTrain()
        trainer.train()
        out = trainer.evaluate_on_test_data()
        res.append(out)
    print(np.mean(res))
    print(np.std(res))


if __name__ == '__main__':
    main()
