"""
Created by Anonymous on September 23, 2019
Gombolay et. al. benchmark
"""

import torch
import sys

sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')
import numpy as np
from utils.pairwise_utils import load_in_pairwise_data
from utils.hri_utils import save_performance_results_DT_version

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


class Pairwise_DT:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self):
        # load in data
        self.num_schedules = 250
        self.num_test_schedules = 250

        self.X_train_pairwise, self.Y_train_pairwise, self.schedule_array_train_pairwise, self.start_of_each_set_twenty_train, self.X_test_pairwise, self.Y_test_pairwise, self.schedule_array_test_pairwise, self.start_of_each_set_twenty_test = load_in_pairwise_data(
            self.num_schedules, self.num_test_schedules)
        self.X_train_pairwise, \
        self.Y_train_pairwise, \
        self.schedule_array_train_pairwise, \
        self.start_of_each_set_twenty_train = self.sample_data(150)

        self.X_test_pairwise, \
        self.Y_test_pairwise, \
        self.schedule_array_test_pairwise, \
        self.start_of_each_set_twenty_test = self.sample_test_data(100)
        self.num_test_schedules = 100

    def sample_data(self, size):
        # return self.X_train_pairwise[0:size * 20 * 20], \
        #        self.Y_train_pairwise[0:size * 20 * 20], \
        #        self.schedule_array_train_pairwise[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250-size)
        self.sample_min = set_of_twenty * 400
        return self.X_train_pairwise[set_of_twenty*400:set_of_twenty*400 + size * 20 * 20], \
               self.Y_train_pairwise[set_of_twenty*400:set_of_twenty*400 + size * 20 * 20], \
               self.schedule_array_train_pairwise[set_of_twenty:set_of_twenty+size], \
               self.start_of_each_set_twenty_train[set_of_twenty*20:set_of_twenty*20+size * 20]

    def sample_test_data(self, size):
        # return self.X_train_pairwise[0:size * 20 * 20], \
        #        self.Y_train_pairwise[0:size * 20 * 20], \
        #        self.schedule_array_train_pairwise[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250-size)
        self.sample_test_min = set_of_twenty * 400
        return self.X_test_pairwise[set_of_twenty*400:set_of_twenty*400 + size * 20 * 20], \
               self.Y_test_pairwise[set_of_twenty*400:set_of_twenty*400 + size * 20 * 20], \
               self.schedule_array_test_pairwise[set_of_twenty:set_of_twenty+size], \
               self.start_of_each_set_twenty_test[set_of_twenty*20:set_of_twenty*20+size * 20]


    def generate_data(self):
        """
        Generates a bunch of counterfactual data (poorly done)
        :return:
        """

        data_matrix = []
        output_matrix = []

        # variables to keep track of loss and number of tasks trained over

        while True:
            # sample a timestep before the cutoff for cross_validation
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty_train)
            truth = self.Y_train_pairwise[set_of_twenty-self.sample_min]

            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty-self.sample_min
            phi_i = self.X_train_pairwise[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)

            # iterate over pairwise comparisons
            for counter in range(set_of_twenty-self.sample_min, set_of_twenty + 20-self.sample_min):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X_train_pairwise[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_i_numpy - phi_j_numpy
                    data_matrix.append(list(feature_input))

                    output_matrix.append(1)

            for counter in range(set_of_twenty-self.sample_min, set_of_twenty + 20-self.sample_min):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X_train_pairwise[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy

                    data_matrix.append(list(feature_input))
                    output_matrix.append(0)

            if len(data_matrix) > 300000:
                return data_matrix, output_matrix

    def generate_test_data(self):
        """
        Generates a bunch of counterfactual data (poorly done)
        :return:
        """

        data_matrix = []
        output_matrix = []

        # variables to keep track of loss and number of tasks trained over

        for j in range(0, self.num_test_schedules):
            # sample a timestep before the cutoff for cross_validation
            schedule_bounds = self.schedule_array_test_pairwise[j]
            step = schedule_bounds[0] - self.sample_test_min
            truth = self.Y_test_pairwise[step]

            # find feature vector of true action taken

            while step < schedule_bounds[1] - self.sample_test_min:
                # find feature vector of true action taken
                phi_i_num = truth + step
                phi_i = self.X_test_pairwise[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)

                # iterate over pairwise comparisons
                for counter in range(step, step + 20):
                    if counter == phi_i_num:  # if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X_test_pairwise[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy
                        data_matrix.append(list(feature_input))

                        output_matrix.append(1)

                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X_test_pairwise[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        data_matrix.append(list(feature_input))
                        output_matrix.append(0)

                        # add average loss to array
                step += 20

        return data_matrix, output_matrix

    def evaluate(self, clf):

        """
        Evaluate performance of a DT
        :return:
        """

        prediction_accuracy = [0]
        percentage_accuracy_top1 = []

        for j in range(0, self.num_test_schedules):
            schedule_bounds = self.schedule_array_test_pairwise[j]
            step = schedule_bounds[0] -self.sample_test_min
            while step < schedule_bounds[1] -self.sample_test_min:
                probability_matrix = np.zeros((20, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = self.X_test_pairwise[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagonals set to zero
                        if second_counter == counter:  # same as m = n
                            continue
                        phi_j = self.X_test_pairwise[second_counter]
                        phi_j_numpy = np.asarray(phi_j)

                        feature_input = phi_i_numpy - phi_j_numpy

                        # push through nets
                        preference_prob = clf.predict(feature_input.reshape(1, -1))
                        probability_matrix[m][n] = preference_prob
                # feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13))

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)
                highest_val = max(column_vec)
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)

                truth = self.Y_test_pairwise[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20)
            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)

            prediction_accuracy = [0]

        print(np.mean(percentage_accuracy_top1))
        print(np.std(percentage_accuracy_top1))
        # save_performance_results_DT_version(percentage_accuracy_top1, 'results_DT_pairwise')
        return np.mean(percentage_accuracy_top1)

def main():
    """
    entry point for file
    :return:
    """
    res = []
    for i in range(3):
        trainer = Pairwise_DT()
        X, Y = trainer.generate_data()
        clf = DecisionTreeClassifier(max_depth=10)
        clf.fit(X, Y)

        y_pred = clf.predict(X)
        print(accuracy_score(Y, y_pred))

        result=trainer.evaluate(clf)
        res.append(result)
    print(np.mean(res), np.std(res))
    # X_test, Y_test = trainer.generate_test_data()
    # y_pred_test = clf.predict(X_test)
    # print(accuracy_score(Y_test, y_pred_test))

    # tree.export_graphviz(clf, out_file='tree_pairwise.dot')


if __name__ == '__main__':
    main()
