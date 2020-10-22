"""
PNT standard implementation on taxi_domain trees
created on January 18, 2020 by Rohan Paleja
"""
import torch
import torch.nn as nn
import numpy as np
from low_dim.prolonet import ProLoNet
from torch.autograd import Variable
import torch.nn.functional as F

import os
import pickle

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)


# noinspection PyArgumentList,PyUnresolvedReferences,PyTypeChecker
class PNT_pairwise:
    """
    PNT_pairwise for taxi domain
    """

    def __init__(self):

        self.state_dim = 5
        self.output_dim = 5
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
                              leaves=64,
                              is_value=False,
                              bayesian_embedding_dim=self.embedding_dim,
                              alpha=1.5,
                              use_gpu=self.use_gpu,
                              vectorized=True,
                              selectors=None).to(device)

        self.sig = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()

        self.opt = torch.optim.RMSprop(
            [{'params': list(self.model.parameters())[:-1]}, {'params': self.model.bayesian_embedding.parameters(), 'lr': .01}], lr=.01)
        self.person_specific_embeddings = [torch.ones(self.embedding_dim) * 1 / 3 for _ in range(600)]

        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()

        print('Leaves is ', 64)
        print('Optimizer: RMSprop, lr_e=.001 lr=.0001')
        print('adding the bad policies')

        self.training_accuracies = []
        self.testing_accuracies = []
        self.testing_stds = []


    @staticmethod
    def load_in_data():
        """
        loads in train data
        :return:
        """
        states, actions, failed_list, mturkcodes = pickle.load(open(os.path.join('../datasets/', 'training_data_from_all_users_2.pkl'), 'rb'))

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
        states, actions, failed_list, mturkcodes = pickle.load(open(os.path.join('../datasets/', 'testing_data_from_all_users_2.pkl'), 'rb'))

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
        for i in range(30000):

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
                    action_t = torch.tensor(action_t).reshape(1).cuda()
                else:
                    network_input = torch.tensor(state_t)
                    action_t = torch.tensor(action_t).reshape(1)

                output = self.model.forward(network_input.float())

                self.opt.zero_grad()
                loss = F.cross_entropy(output, action_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()

            self.person_specific_embeddings[which_user] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())

            if i % 1000 == 999:
                # print('loss is: ', np.mean(self.loss_array[-500]), 'for epoch: ', i)
                # assert not os.path.exists(log_path) or "test" in log_path, "log path already exist! "f
                torch.save({'state_dict': self.model.state_dict(),
                            'person_embeddings': self.person_specific_embeddings},
                           '/home/Anonymous/PycharmProjects/bayesian_prolo/SGD/models/PDDT_pairwise_drone_531' + str(i) + '.pkl')
                self.evaluate_on_data(True)
                self.evaluate_on_data(False)
                print('training accuracies: ', self.training_accuracies)
                print('testing accuracies: ', self.testing_accuracies)

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

                if self.use_gpu:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)))
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)))


                # forward
                output = self.model.forward(input_nn)
                embedding_optimizer.zero_grad()
                loss = F.cross_entropy(output, truth)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                embedding_optimizer.step()


                # top choice
                index = torch.argmax(output).item()
                if index == truth.item():
                    accuracy += 1

            person_specific_embeddings[i] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            # print('accuracy: ', accuracy / length_of_current_game)
            accuracies.append(accuracy / length_of_current_game)
        if train:
            self.training_accuracies.append(np.mean(accuracies))
        else:
            self.testing_accuracies.append(np.mean(accuracies))
            self.testing_stds.append(np.std(accuracies)/len(accuracies))

    def test_again_crisp(self, train=False):
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

                if self.use_gpu:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)))
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)))


                # forward
                output = self.model.forward(input_nn)
                embedding_optimizer.zero_grad()
                loss = F.cross_entropy(output, truth)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                embedding_optimizer.step()


                # top choice
                index = torch.argmax(output).item()
                if index == truth.item():
                    accuracy += 1

            person_specific_embeddings[i] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            # print('accuracy: ', accuracy / length_of_current_game)
            accuracies.append(accuracy / length_of_current_game)
        if train:
            self.training_accuracies.append(np.mean(accuracies))
        else:
            self.testing_accuracies.append(np.mean(accuracies))
            self.testing_stds.append(np.std(accuracies)/len(accuracies))

        self.model = convert_to_crisp(model, None)

        accuracies = []
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
            self.model.set_bayesian_embedding(self.person_specific_embeddings[i].clone())

            for j in range(length_of_current_game):

                # input
                state_t = states[i][j]
                action_t = actions[i][j]

                if self.use_gpu:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)))
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)))


                # forward
                output = self.model.forward(input_nn)

                # top choice
                index = torch.argmax(output).item()
                if index == truth.item():
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

    trainer = PNT_pairwise()
    trainer.train_model()
    print('Training accuracy', trainer.training_accuracies)
    print('Testing accuracies', trainer.testing_accuracies)
    print(np.max(trainer.training_accuracies))
    print(np.max(trainer.testing_accuracies))
    print('max val: ', np.max(trainer.testing_accuracies), ' std', trainer.testing_stds[int(np.argmax(trainer.testing_accuracies))])

    print('This is testing on all players games and testing on holdout set of each player \n embedding size is 12')


if __name__ == '__main__':
    main()
