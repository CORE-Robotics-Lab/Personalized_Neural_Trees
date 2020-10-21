"""
Created by Rohan Paleja on January 12, 2020
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
        self.action_embeddings = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        self.action_embedding_dim = len(self.action_embeddings[0])
        self.num_actions = len(self.action_embeddings)
        self.state_dim = 5 + self.num_actions
        self.output_dim = 1

        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()

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

    def generate_data(self):
        """
        Generates a bunch of counterfactual data (poorly done)
        :return:
        """

        data_matrix = []
        output_matrix = []

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
                if torch.cuda.is_available():
                    network_input = torch.tensor(state_t).reshape(1, 5).cuda()
                    action_t = torch.tensor(action_t).cuda()
                else:
                    network_input = torch.tensor(state_t)
                    action_t = torch.tensor(action_t)

                phi_i = np.asarray(self.action_embeddings[action_t])

                # positive counterfactuals
                for counter in range(self.num_actions):
                    if counter == action_t:  # if counter == phi_i_num:
                        continue
                    else:
                        phi_j = np.asarray(self.action_embeddings[counter])
                        feature_input = phi_i - phi_j

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 3)).cuda())
                            label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                            feature_input = torch.cat([feature_input, network_input.float()], dim=1)
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 3)))
                            label = Variable(torch.Tensor(torch.ones((1, 1))))
                            feature_input = torch.cat([feature_input, network_input.float()], dim=1)

                        data_matrix.append(feature_input[0].tolist())
                        output_matrix.append(int(label.item()))

                for counter in range(self.num_actions):
                    if counter == action_t:  # if counter == phi_i_num:
                        continue
                    else:
                        phi_j = np.asarray(self.action_embeddings[counter])
                        feature_input = phi_j - phi_i

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 3)).cuda())
                            label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                            feature_input = torch.cat([feature_input, network_input.float()], dim=1)
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 3)))
                            label = Variable(torch.Tensor(torch.zeros((1, 1))))
                            feature_input = torch.cat([feature_input, network_input.float()], dim=1)

                        data_matrix.append(feature_input[0].tolist())
                        output_matrix.append(int(label.item()))



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

                probability_matrix = np.zeros((self.num_actions, self.num_actions))

                # begin counterfactual reasoning
                for each_action in range(self.num_actions):
                    phi_i = self.action_embeddings[each_action]
                    phi_i_numpy = np.asarray(phi_i)

                    for each_other_action in range(self.num_actions):
                        if each_action == each_other_action:
                            continue

                        phi_j = self.action_embeddings[each_other_action]
                        phi_j_numpy = np.asarray(phi_j)
                        action_embedding_counterfactual = phi_i_numpy - phi_j_numpy


                        network_input = torch.cat(
                            [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                            dim=0).reshape(1, self.state_dim)

                        # forward
                        preference_prob = clf.predict_proba(network_input.reshape(1, -1))
                        probability_matrix[each_action][each_other_action] = preference_prob[0][1]


                # Finished all counterfactuals
                column_vec = np.sum(probability_matrix, axis=1)

                # print('preference of each action (highest number is the one you should take)', column_vec)
                highest_val = max(column_vec)

                # account for ties, if there is a tie pick a random one of the two
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
                if len(all_indexes_that_have_highest_val) > 1:
                    print('actions that are candidates: ', all_indexes_that_have_highest_val)

                # top choice
                index = np.random.choice(all_indexes_that_have_highest_val)
                if index == action_t:
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
    clf = DecisionTreeClassifier(max_depth=13)
    clf.fit(X, Y)
    y_pred = clf.predict(X)
    print('Training accuracy: ', accuracy_score(Y, y_pred))

    trainer.evaluate_on_test_data(clf)
    print('Finishe')


if __name__ == '__main__':
    main()