"""
Created by Anonymous on January 12, 2020
Purpose: DT_pairwise benchmark in taxi domain
"""


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import torch
import os
import pickle
from torch.autograd import Variable

np.random.seed(50)


# noinspection PyArgumentList
class DT_pairwise:
    """
    class structure to train the DT with a certain alpha.
    This class handles training the DT, evaluating the DT, and saving
    """

    def __init__(self):

        self.state_dim = 5
        self.output_dim = 1

        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()

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

    def generate_data(self):
        """
        Generates a bunch of counterfactual data (poorly done)
        :return:
        """

        data_matrix = []
        output_matrix = []
        checkpoint = torch.load(
            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/train_taxi_embedding_3.tar')

        for which_user in range(len(self.states)):
            if which_user in self.indices_of_failed:
                continue
            # get game states
            states = self.states[which_user]
            actions = self.actions[which_user]
            length_of_current_game = len(states)

            for j in range(length_of_current_game):
                # input
                state_t = states[j]
                action_t = actions[j]

                data_matrix.append(state_t)
                data_matrix[-1].extend(list(checkpoint['train_embeddings'][which_user]))
                output_matrix.append(action_t)


        return data_matrix, output_matrix

    def evaluate_on_test_data(self, clf):
        """
        evaluate on a subset of training data
        :param clf: trained tree
        """

        accuracies = []
        states = self.test_states
        actions = self.test_actions
        indices_of_failed = self.test_indices_of_failed
        checkpoint = torch.load(
            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/test_taxi_embedding_3.tar')

        for i in range(len(states)):
            if i in indices_of_failed:
                continue
            accuracy = 0

            length_of_current_game = len(states[i])

            # set player specific embedding into network

            for j in range(length_of_current_game):

                # input
                state_t = states[i][j]
                action_t = actions[i][j]
                state_t.extend(list(checkpoint['test_embeddings'][i]))
                net_input = torch.tensor(state_t).reshape(1, -1)


                output = clf.predict(net_input)

                if output == action_t:
                    accuracy += 1

            # print('accuracy: ', accuracy / length_of_current_game)
            accuracies.append(accuracy / length_of_current_game)
        print('testing accuracy: ', np.mean(accuracies))
        print('stderr', np.std(accuracies)/len(accuracies))




def main():
    """
    entry point for file
    :return:
    """

    trainer = DT_pairwise()
    X, Y = trainer.generate_data()
    clf = DecisionTreeClassifier(max_depth=7)
    clf.fit(X, Y)
    y_pred = clf.predict(X)
    print('Training accuracy: ', accuracy_score(Y, y_pred))

    trainer.evaluate_on_test_data(clf)
    print('Finished')


if __name__ == '__main__':
    main()