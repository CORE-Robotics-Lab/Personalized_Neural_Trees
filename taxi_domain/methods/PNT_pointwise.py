"""
PNT pointwise implementation on taxi_domain trees
created on February 24, 2020 by Rohan Paleja
"""
import torch
import torch.nn as nn
import numpy as np
from low_dim.prolonet import ProLoNet
from torch.autograd import Variable
import os
import pickle
from AndrewSilva.tree_nets.utils.fuzzy_to_crispy import convert_to_crisp

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)


# noinspection PyArgumentList,PyUnresolvedReferences,PyTypeChecker
class PNT_pointwise:
    """
    PNT_pairwise for taxi domain
    """

    def __init__(self):
        self.action_embeddings = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        self.action_embedding_dim = len(self.action_embeddings[0])
        self.num_actions = len(self.action_embeddings)
        self.state_dim = 5 + self.num_actions
        self.output_dim = 1
        self.embedding_dim = 3
        self.use_gpu = True
        if not self.use_gpu:
            device = 'cpu'
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ProLoNet(input_dim=self.state_dim,
                              output_dim=self.output_dim,
                              weights=None,
                              comparators=None,
                              leaves=128,
                              is_value=True,
                              bayesian_embedding_dim=self.embedding_dim,
                              alpha=1.5,
                              use_gpu=self.use_gpu,
                              vectorized=True,
                              selectors=None).to(device)

        self.sig = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()

        self.opt = torch.optim.RMSprop(
            [{'params': list(self.model.parameters())[:-1]}, {'params': self.model.bayesian_embedding.parameters(), 'lr': .1}], lr=.01)
        self.person_specific_embeddings = [torch.ones(self.embedding_dim) * 1 / 3 for _ in range(600)]

        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()

        print('Leaves is ', 128)
        print('Optimizer: RMSprop, lr_e=.001 lr=.0001')


        self.training_accuracies = []
        self.testing_accuracies = []
        self.testing_stds = []

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

    @staticmethod
    def load_in_test_data():
        """
        loads in test data
        :return:
        """
        states, actions, failed_list, mturkcodes = pickle.load(open(os.path.join('../datasets/', 'testing_data_from_all_users.pkl'), 'rb'))

        indices_of_failed = []
        for i in failed_list:
            if i[0] not in indices_of_failed:
                indices_of_failed.append(i[0])

        return states, actions, failed_list, mturkcodes, indices_of_failed

    def train_model(self):
        """
        trains a model
        :return:
        """
        # variables to keep track of loss and number of tasks trained over
        for i in range(3000):

            print('epoch: ', i)
            # sample a timestep before the cutoff for cross_validation
            which_user = np.random.choice(range(len(self.states)))
            if which_user in self.indices_of_failed:
                continue

            states = self.states[which_user]
            actions = self.actions[which_user]
            length_of_current_game = len(states)

            # set player specific embedding into network
            self.model.set_bayesian_embedding(self.person_specific_embeddings[which_user].clone())

            # pick ten timesteps within game
            for j in range(50):

                timestep = np.random.choice(range(len(self.states[which_user])))

                # input

                state_t = states[timestep]
                action_t = actions[timestep]

                if self.use_gpu:
                    network_input = torch.tensor(state_t).reshape(1, 5).cuda()
                    action_t = torch.tensor(action_t).cuda()
                else:
                    network_input = torch.tensor(state_t).reshape(1, 5)
                    action_t = torch.tensor(action_t)

                phi_i = np.asarray(self.action_embeddings[action_t])

                # positive counterfactuals
                for counter in range(self.num_actions):
                    if counter == action_t:  # if counter == phi_i_num:
                        label = torch.ones((1, 1))
                    else:
                        label = torch.zeros((1, 1))
                    phi = np.asarray(self.action_embeddings[counter])
                    feature_input = phi

                    if self.use_gpu:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 3)).cuda())
                        label = Variable(label).cuda()
                        feature_input = torch.cat([feature_input, network_input.float()], dim=1)
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 3)))
                        label = Variable(label)
                        feature_input = torch.cat([feature_input, network_input.float()], dim=1)

                    output = self.model.forward(feature_input)
                    output = self.sig(output)

                    self.opt.zero_grad()
                    loss = self.criterion(output, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()

            self.person_specific_embeddings[which_user] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())

            if i % 100 == 99:
                # print('loss is: ', np.mean(self.loss_array[-500]), 'for epoch: ', i)
                # assert not os.path.exists(log_path) or "test" in log_path, "log path already exist! "f

                self.evaluate_on_data(True)
                person_specific_embeddings = self.evaluate_on_data(False)
                # torch.save({'state_dict': self.model.state_dict(),
                #             'person_embeddings': person_specific_embeddings},
                #            '/home/Anonymous/PycharmProjects/bayesian_prolo/taxi_domain/models/PNT_pairwise_128' + str(i) + '.pkl')
                print('training accuracies: ', self.training_accuracies)
                print('testing accuracies: ', self.testing_accuracies)

            # if i > 6000 and i % 1000 == 999:
            #     self.retest_with_crisp_model()

    def evaluate_on_data(self, train=True):
        """
        evaluate on a subset of training data
        :param train: if train is false, means we are in test
        """
        embedding_optimizer = torch.optim.RMSprop([{'params': self.model.bayesian_embedding.parameters()}], lr=.1)

        accuracies = []
        person_specific_embeddings = [torch.ones(self.embedding_dim) * 1 / 3 for _ in range(600)]
        if train:
            states = self.states
            actions = self.actions
            indices_of_failed = self.indices_of_failed
        else:
            states = self.test_states
            actions = self.test_actions
            indices_of_failed = self.test_indices_of_failed

        for i in range(len(states)):
            if i in indices_of_failed:
                continue
            accuracy = 0

            length_of_current_game = len(states[i])

            # set player specific embedding into network
            self.model.set_bayesian_embedding(person_specific_embeddings[i].clone())

            for j in range(length_of_current_game):

                # input
                state_t = states[i][j]
                action_t = actions[i][j]

                probability_vector = np.zeros((1, self.num_actions))

                # begin counterfactual reasoning
                for each_action in range(self.num_actions):
                    phi_i = self.action_embeddings[each_action]
                    phi_i_numpy = np.asarray(phi_i)

                    action_embedding_in = phi_i_numpy

                    if self.use_gpu:
                        network_input = torch.cat(
                            [torch.Tensor(action_embedding_in).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                            dim=0).reshape(1, self.state_dim).cuda()


                    else:
                        network_input = torch.cat(
                            [torch.Tensor(action_embedding_in).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                            dim=0).reshape(1, self.state_dim)

                    # forward
                    preference_prob = self.sig(self.model.forward(network_input))
                    probability_vector[0][each_action] = preference_prob.data.detach()[0].item()

                # embedding update
                for each_action in range(self.num_actions):
                    if each_action == action_t:
                        label = torch.ones((1, 1))
                    else:
                        label = torch.zeros((1,1))
                    phi_j = self.action_embeddings[each_action]
                    phi_j_numpy = np.asarray(phi_j)
                    action_embedding_counterfactual = phi_j_numpy

                    if self.use_gpu:
                        network_input = torch.cat(
                            [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                            dim=0).reshape(1, self.state_dim).cuda()
                        label = Variable(label).cuda()

                    else:
                        network_input = torch.cat(
                            [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                            dim=0).reshape(1, self.state_dim)
                        label = Variable(label)

                    embedding_optimizer.zero_grad()
                    output = self.sig(self.model.forward(network_input))
                    loss = self.criterion(output, label)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    embedding_optimizer.step()

                # print('preference of each action (highest number is the one you should take)', column_vec)
                highest_val = max(probability_vector[0])

                # account for ties, if there is a tie pick a random one of the two
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(probability_vector[0])) if e == highest_val]
                if len(all_indexes_that_have_highest_val) > 1:
                    print('actions that are candidates: ', all_indexes_that_have_highest_val)

                # top choice
                index = np.random.choice(all_indexes_that_have_highest_val)
                if index == action_t:
                    accuracy += 1

            person_specific_embeddings[i] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            # print('accuracy: ', accuracy / length_of_current_game)
            accuracies.append(accuracy / length_of_current_game)
        if train:
            self.training_accuracies.append(np.mean(accuracies))
        else:
            self.testing_accuracies.append(np.mean(accuracies))
            self.testing_stds.append(np.std(accuracies)/len(accuracies))
            return person_specific_embeddings

    def retest_with_crisp_model(self, load_in=False):
        embedding_optimizer = torch.optim.RMSprop([{'params': self.model.bayesian_embedding.parameters()}], lr=.1)

        accuracies = []
        if load_in:
            # checkpoint = torch.load(
            #     '/home/Anonymous/PycharmProjects/bayesian_prolo/taxi_domain/models/PNT_pairwise_642999.pkl') good models
            checkpoint = torch.load(
                '/home/Anonymous/PycharmProjects/bayesian_prolo/taxi_domain/models/PNT_pairwise_643999.pkl')
            self.model.load_state_dict(checkpoint['state_dict'])
            person_specific_embeddings = checkpoint['person_embeddings']
        else:
            person_specific_embeddings = [torch.ones(self.embedding_dim) * 1 / 3 for _ in range(600)]

        states = self.test_states
        actions = self.test_actions
        indices_of_failed = self.test_indices_of_failed

        if load_in == False:
            for i in range(len(states)):
                if i in indices_of_failed:
                    continue
                accuracy = 0

                length_of_current_game = len(states[i])

                # set player specific embedding into network
                self.model.set_bayesian_embedding(self.person_specific_embeddings[i].clone())

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

                            if self.use_gpu:
                                network_input = torch.cat(
                                    [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                                    dim=0).reshape(1, self.state_dim).cuda()

                            else:
                                network_input = torch.cat(
                                    [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                                    dim=0).reshape(1, self.state_dim)

                            # forward
                            preference_prob = self.sig(self.model.forward(network_input))
                            probability_matrix[each_action][each_other_action] = preference_prob.data.detach()[0].item()

                    # embedding update
                    phi_i = self.action_embeddings[action_t]
                    phi_i_numpy = np.asarray(phi_i)
                    for each_action in range(self.num_actions):
                        if each_action == action_t:
                            continue
                        else:
                            phi_j = self.action_embeddings[each_action]
                            phi_j_numpy = np.asarray(phi_j)
                            action_embedding_counterfactual = phi_i_numpy - phi_j_numpy

                            if self.use_gpu:
                                network_input = torch.cat(
                                    [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                                    dim=0).reshape(1, self.state_dim).cuda()

                                label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                            else:
                                network_input = torch.cat(
                                    [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                                    dim=0).reshape(1, self.state_dim)
                                label = Variable(torch.Tensor(torch.ones((1, 1))))

                            embedding_optimizer.zero_grad()
                            output = self.sig(self.model.forward(network_input))
                            loss = self.criterion(output, label)

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                            embedding_optimizer.step()

                            phi_j = self.action_embeddings[each_action]
                            phi_j_numpy = np.asarray(phi_j)
                            action_embedding_counterfactual = phi_j_numpy - phi_i_numpy

                            if self.use_gpu:
                                network_input = torch.cat(
                                    [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                                    dim=0).reshape(1, self.state_dim).cuda()

                                label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                            else:
                                network_input = torch.cat(
                                    [torch.Tensor(action_embedding_counterfactual).reshape(self.action_embedding_dim), torch.Tensor(state_t)],
                                    dim=0).reshape(1, self.state_dim)
                                label = Variable(torch.Tensor(torch.zeros((1, 1))))

                            embedding_optimizer.zero_grad()
                            output = self.sig(self.model.forward(network_input))
                            loss = self.criterion(output, label)

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                            embedding_optimizer.step()

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

                person_specific_embeddings[i] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

                # print('accuracy: ', accuracy / length_of_current_game)
                accuracies.append(accuracy / length_of_current_game)


        # CRISPY
        model = convert_to_crisp(self.model, None)
        accuracies = []

        for i in range(len(states)):
            if i in indices_of_failed:
                continue
            accuracy = 0

            length_of_current_game = len(states[i])

            # set player specific embedding into network
            self.model.set_bayesian_embedding(person_specific_embeddings[i].clone())

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
                        preference_prob = self.sig(model.forward(network_input))
                        probability_matrix[each_action][each_other_action] = preference_prob.data.detach()[0].item()


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

            person_specific_embeddings[i] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            # print('accuracy: ', accuracy / length_of_current_game)
            accuracies.append(accuracy / length_of_current_game)

        print('crisp acc', np.mean(accuracies))
        print(np.std(accuracies) / len(accuracies))


def main():
    """
    entry point for file
    :return:
    """

    trainer = PNT_pointwise()
    trainer.train_model()
    # trainer.retest_with_crisp_model(load_in=True)
    print('Training accuracy', trainer.training_accuracies)
    print('Testing accuracies', trainer.testing_accuracies)
    print(np.max(trainer.training_accuracies))
    print(np.max(trainer.testing_accuracies))
    print('max val: ', np.max(trainer.testing_accuracies), ' std', trainer.testing_stds[int(np.argmax(trainer.testing_accuracies))])

    print('This is testing on all players games and testing on holdout set of each player \n embedding size is 12')


if __name__ == '__main__':
    main()
