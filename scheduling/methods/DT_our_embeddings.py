"""
Created by Rohan Paleja on September 23, 2019
Gombolay et. al. benchmark
"""

import torch
import sys

sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')
import numpy as np
from utils.naive_utils import  load_in_naive_data, find_which_schedule_this_belongs_to
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
        self.num_schedules = 150
        self.num_test_schedules = 100

        self.X_train_naive, self.Y_train_naive, self.schedule_array_train_naive, self.X_test_naive, self.Y_test_naive, self.schedule_array_test_naive = load_in_naive_data(
            250,250)
        self.X_train_naive, \
        self.Y_train_naive, \
        self.schedule_array_train_naive,  = self.sample_data(150)

        self.X_test_naive, \
        self.Y_test_naive, \
        self.schedule_array_test_naive,  = self.sample_test_data(100)

    def sample_data(self, size):
        # return self.X_train_naive[0:size * 20 * 20], \
        #        self.Y_train_naive[0:size * 20 * 20], \
        #        self.schedule_array_train_naive[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250 - size)
            self.start = set_of_twenty
        self.sample_min = set_of_twenty * 20
        return self.X_train_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.Y_train_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.schedule_array_train_naive[set_of_twenty:set_of_twenty + size]

    def sample_test_data(self, size):
        # return self.X_train_naive[0:size * 20 * 20], \
        #        self.Y_train_naive[0:size * 20 * 20], \
        #        self.schedule_array_train_naive[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250 - size)
            self.test_start = set_of_twenty
        self.sample_test_min = set_of_twenty * 20
        return self.X_test_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.Y_test_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.schedule_array_test_naive[set_of_twenty:set_of_twenty + size]

    def generate_data(self, i):
        """
        Generates a bunch of counterfactual data (poorly done)
        :return:
        """

        data_matrix = []
        output_matrix = []

        # variables to keep track of loss and number of tasks trained over
        checkpoint = torch.load(
            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/train_embedding_'+str(i+1)+'.tar')

        for e, i in enumerate(self.X_train_naive):
            # sample a timestep before the cutoff for cross_validation
            input_nn = self.X_train_naive[e]
            truth_nn = self.Y_train_naive[e]
            # find feature vector of true action taken


            data_matrix.append(list(input_nn))
            data_matrix[-1].extend(list(checkpoint['train_embeddings'][find_which_schedule_this_belongs_to(self.schedule_array_train_naive, e+self.sample_min)]))
            output_matrix.append(truth_nn)
        return data_matrix, output_matrix

    def evaluate(self, clf,i):

        """
        Evaluate performance of a DT
        :return:
        """

        prediction_accuracy = [0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        checkpoint = torch.load(
            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/test_embedding_'+str(i+1)+'.tar')
        for i, schedule in enumerate(self.schedule_array_test_naive):
            for count in range(schedule[0] - self.sample_test_min, schedule[1] - self.sample_test_min + 1):
                net_input = self.X_test_naive[count]
                truth = self.Y_test_naive[count]
                net_input.extend(list(checkpoint['test_embeddings'][i]))
                net_input = torch.tensor(net_input).reshape(1,-1)
                output = clf.predict(net_input)

                if output == truth:
                    prediction_accuracy[0] += 1


            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20)
            print('schedule num:', i)
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
        X, Y = trainer.generate_data(i)
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, Y)

        y_pred = clf.predict(X)
        print(accuracy_score(Y, y_pred))

        result=trainer.evaluate(clf, i)
        res.append(result)
    print(np.mean(res), np.std(res))

if __name__ == '__main__':
    main()
