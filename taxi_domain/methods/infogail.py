"""
infogail in taxi domain
"""

import torch
import torch.nn as nn
from torch import optim
import math

import numpy as np
from torch.autograd import Variable
from utils.global_utils import save_pickle
import pickle
from torch.distributions import OneHotCategorical
import os

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    if torch.cuda.is_available():
        return y[labels].cuda()
    else:
        return y[labels]


class PolicyNetwork(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 embedding_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + embedding_dim, 128)  # 245 because mode is of type 3
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(128, 128)
        self.relu21 = nn.ReLU()
        self.fc22a = nn.Linear(128, 128)
        self.relu22a = nn.ReLU()
        self.fc23a = nn.Linear(128, 128)
        self.relu23a = nn.ReLU()
        self.fc22 = nn.Linear(128, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, action_dim)
        self.soft = nn.Softmax(dim=0)


        # self.forward_pass = nn.Sequential(
        #     nn.Linear(state_dim + embedding_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, action_dim),
        #     nn.Softmax(dim=0)
        # )

    def forward(self, state, latent_code):
        input_data = torch.cat((state, latent_code), dim=0)
        x = self.fc1(input_data)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc21(x)
        x = self.relu21(x)
        x = self.fc22(x)
        x = self.relu22(x)
        x = self.fc23(x)
        x = self.relu23(x)
        x = self.fc3(x)
        x = self.soft(x)

        return x


class Discriminator(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim):
        super(Discriminator, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        input_data = torch.cat((state, action), dim=0)
        return self.forward_pass(input_data)


class AuxiliaryDistributionPredictor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 embedding_dim):
        super(AuxiliaryDistributionPredictor, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.Softmax(dim=0)
        )

    def forward(self, state, action):
        input_data = torch.cat((state, action), dim=0)
        return self.forward_pass(input_data)


def my_entropy(dist):
    n = len(dist)
    x = 0
    for i in range(n):
        if dist[i] == 0:
            x += 0
        else:
            x += torch.sum(dist[i] * torch.log(dist[i]))
    return x


# noinspection PyTypeChecker,PyArgumentList
class InfoGAIL:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self):
        self.total_loss_array = []
        jumpstart = False
        self.states, self.actions, self.failed_list, self.mturkcodes, self.indices_of_failed = self.load_in_data()
        self.test_states, self.test_actions, self.test_failed_list, self.test_mturkcodes, self.test_indices_of_failed = self.load_in_test_data()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dimension = 5
        self.state_dim = state_dimension
        action_dimension = 3
        self.action_dim  = action_dimension
        embedding_dimension = 3
        self.gamma = .95

        # Initialise the network.
        self.policy = PolicyNetwork(
            state_dim=state_dimension,
            action_dim=action_dimension,
            embedding_dim=embedding_dimension
        ).to(device)

        self.discriminator = Discriminator(
            state_dim=state_dimension,
            action_dim=action_dimension
        ).to(device)

        self.distribution_gen = AuxiliaryDistributionPredictor(
            state_dim=state_dimension,
            action_dim=action_dimension,
            embedding_dim=embedding_dimension
        ).to(device)

        if jumpstart:
            checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/NN_jumpstart_NN999999.tar')
            self.policy.load_state_dict(checkpoint['nn_state_dict'])

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=.0001)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=.001)
        self.distro_opt = optim.Adam(self.distribution_gen.parameters(), lr=.0001)  # policy opt could be

        self.testing_accuracies = []
        self.testing_stds = []

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

    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """

        epochs = 4000
        lambda_1 = 1
        lambda_2 = 0

        for epoch in range(epochs):
            print('epoch', epoch)
            discriminator_acc = 0
            avg_policy_loss = 0
            avg_discrim_loss = 0
            # sample a timestep before the cutoff for cross_validation
            which_user = np.random.choice(range(len(self.states)))
            if which_user in self.indices_of_failed:
                continue

            # choose a random mode
            z = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])  # sample embedding from prior distribution
            if z == 0:
                mode = torch.Tensor([1, 0, 0])
            elif z == 1:
                mode = torch.Tensor([0, 1, 0])
            else:
                mode = torch.Tensor([0, 0, 1])

            if torch.cuda.is_available():
                mode = mode.cuda()
            else:
                pass

            states = self.states[which_user]
            actions = self.actions[which_user]

            for _, time in enumerate(range(len(self.states[which_user]))):
                R = 0
                for t, i in enumerate(range(time, len(self.states[which_user]))):
                    state_t = states[i]
                    action_t = actions[i]

                    if torch.cuda.is_available():
                        input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, 5)).cuda())  # change to 5 to increase batch size
                        truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).cuda().long())
                    else:
                        input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, 5)))
                        truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).long())

                    self.disc_opt.zero_grad()

                    # predict action the demonstrator would take
                    self.policy.requires_grad = False
                    self.distribution_gen.requires_grad = False
                    self.discriminator.requires_grad = True
                    action_fake = self.policy.forward(input_nn.reshape(self.state_dim), mode)

                    # Discriminator training
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()

                    # want high probability of labeling policy action as fake
                    fake_action_label = self.discriminator(input_nn.reshape(self.state_dim), one_hot_encoded_action_fake_dist) + 1 * 10 ** -6  # should be 1
                    # print(fake_action_label)
                    discriminator_acc += fake_action_label.item()
                    # loss D_part 1
                    J_D_1 = -torch.log(fake_action_label)  # if .1, will be a big negative

                    # want low probability of labeling expert as fake
                    real_action_label = self.discriminator(input_nn.reshape(self.state_dim), one_hot_embedding(truth, self.action_dim).reshape(self.action_dim)) + 1 * 10 ** -6  # should be 0
                    # print(real_action_label)
                    discriminator_acc += (1 - real_action_label.item())
                    # loss d_part 2
                    J_D_2 = -torch.log(1.01 - real_action_label)

                    # if real_action_label <= .51 + 1 * 10**-6:
                    #     J_D_3 = 10
                    # else:
                    #     J_D_3 = 0
                    loss_discrim = J_D_1 + J_D_2  # now it is positive, and gets minimized
                    # print(loss_discrim)
                    # compute gradients
                    loss_discrim.backward()
                    # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)

                    # step parameters
                    self.disc_opt.step()

                    self.policy.requires_grad = False
                    self.distribution_gen.requires_grad = True
                    self.discriminator.requires_grad = False

                    # DISTRIBUTION
                    self.distro_opt.zero_grad()
                    action_fake = self.policy.forward(input_nn.reshape(self.state_dim), mode)
                    # Discriminator training
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()
                    L_Q = -1 * lambda_1 * torch.log(
                        self.distribution_gen(input_nn.reshape(self.state_dim), one_hot_encoded_action_fake_dist)[
                            z] + 10 ** -6)  # you want to maximize the prob of actual mode
                    L_Q.backward()
                    self.distro_opt.step()
                    discriminator_acc += (1 - fake_action_label.item())

                    # POLICY
                    self.policy.requires_grad = False
                    self.distribution_gen.requires_grad = False
                    self.discriminator.requires_grad = False

                    action_fake = self.policy.forward(input_nn.reshape(self.state_dim), mode)
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()
                    fake_action_label = self.discriminator(input_nn.reshape(self.state_dim), one_hot_encoded_action_fake_dist) + 1 * 10 ** -6  # should be 0

                    L_pi_1 = -torch.log(1.01 - fake_action_label)  # we want this to be minimized now, since it should think action_fake is 0 or real
                    L_pi_2 = -1 * lambda_1 * torch.log(
                        self.distribution_gen(input_nn.reshape(self.state_dim), one_hot_encoded_action_fake_dist)[z] + 10**-6)  # you want to maximize the prob of actual mode
                    L_pi_3 = lambda_2 * -1 * my_entropy(self.policy(input_nn.reshape(self.state_dim), mode))  # you want this to be 0. maximizing entropy

                    L_pi = (L_pi_1 + L_pi_2 + L_pi_3)
                    R += self.gamma ** t * L_pi

                    if math.isnan(discriminator_acc) or math.isnan(loss_discrim.item()):
                        print('failure')
                        print(epoch)
                        exit()
                self.policy_opt.zero_grad()
                self.policy.requires_grad = True
                self.distribution_gen.requires_grad = False
                self.discriminator.requires_grad = False
                state_t = states[time]
                action_t = actions[time]
                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)).cuda())  # change to 5 to increase batch size
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(state_t).reshape(1, self.state_dim)))
                    truth = Variable(torch.Tensor(np.asarray(action_t).reshape(1)).long())
                policy_loss = -R * torch.log(self.policy(input_nn.reshape(self.state_dim), mode)[int(truth.item())])
                print(policy_loss)
                # avg_policy_loss += L_pi.item()
                policy_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm_(self.distribution_gen.parameters(), 0.5)
                self.policy_opt.step()

                # if epoch % 100 == 99 and t == 15:  # 15 chosen randomly
                #     print('at epoch {0}: D loss: {1}, pi loss: {2}, total loss: {3}'.format(epoch, loss_discrim.item(), L_pi.item(), loss_discrim.item() + L_pi.item()))
                #
                # print('epoch {0}: avg D loss {1}, avg pi loss {2}'.format(epoch, avg_discrim_loss/20, avg_policy_loss/20))
                # print(epoch, ': discriminator accuracy: ', discriminator_acc / 60)

            if epoch % 200 == 199:
                print('discrim', self.discriminator.state_dict())
                print('policy', self.policy.state_dict())
                print('dist_gen', self.distribution_gen.state_dict())
                print('epoch is ', epoch)
                # torch.save({'policy_state_dict': self.policy.state_dict(),
                #             'discrim_state_dict': self.discriminator.state_dict(),
                #             'distribution_gen_state_dict': self.distribution_gen.state_dict()},
                #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/infoGAIL' + str(epoch) + '.tar')
                self.evaluate_on_test_data()
                print('Testing accuracies', self.testing_accuracies)
        # torch.save({'policy_state_dict': self.policy.state_dict(),
        #             'discrim_state_dict': self.discriminator.state_dict(),
        #             'distribution_gen_state_dict': self.distribution_gen.state_dict()},
        #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/infoGAIL.tar')

    def evaluate_on_test_data(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
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

            # start off with a random embedding
            z = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
            if z == 0:
                mode = torch.Tensor([1, 0, 0])
            elif z == 1:
                mode = torch.Tensor([0, 1, 0])
            else:
                mode = torch.Tensor([0, 0, 1])

            if torch.cuda.is_available():
                mode = mode.cuda()
            else:
                pass

            for j in range(length_of_current_game):

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
                output = self.policy.forward(input_nn.reshape(self.state_dim), mode)

                index = torch.argmax(output).item()
                print(index, truth)

                if index == truth.item():
                    accuracy += 1

                dist = self.distribution_gen(input_nn.reshape(self.state_dim), one_hot_embedding(int(truth.item()), self.action_dim).reshape(self.action_dim))

                z = np.random.choice([0, 1, 2],
                                     p=[dist[0].item() / (dist[0].item() + dist[1].item() + dist[2].item()),
                                        dist[1].item() / (dist[0].item() + dist[1].item() + dist[2].item()),
                                        dist[2].item() / (
                                                dist[0].item() + dist[1].item() + dist[2].item())])  # sample embedding from prior distribution

                if z == 0:
                    mode = torch.Tensor([1, 0, 0])
                elif z == 1:
                    mode = torch.Tensor([0, 1, 0])
                else:
                    mode = torch.Tensor([0, 0, 1])

                if torch.cuda.is_available():
                    mode = mode.cuda()
                else:
                    pass

            # print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)

            accuracies.append(accuracy / length_of_current_game)

        self.testing_accuracies.append(np.mean(accuracies))
        self.testing_stds.append(np.std(accuracies)/len(accuracies))

        # self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_infoGAIL')

    @staticmethod
    def save_performance_results(top1, top3, special_string):
        """
        saves performance of top1 and top3
        :return:
        """
        print('top1_mean for is : ', np.mean(top1))
        data = {'top1_mean': np.mean(top1),
                'top3_mean': np.mean(top3),
                'top1_stderr': np.std(top1) / np.sqrt(len(top1)),
                'top3_stderr': np.std(top3) / np.sqrt(len(top3))}
        save_pickle(file=data, file_location='/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/results',
                    special_string=special_string)


def main():
    """
    entry point for file
    :return:
    """

    benchmark = InfoGAIL()
    benchmark.train()
    # benchmark.evaluate_on_test_data()
    print('testing accuracies', benchmark.testing_accuracies)
    print('max val: ', np.max(benchmark.testing_accuracies), ' std', benchmark.testing_stds[int(np.argmax(benchmark.testing_accuracies))])


if __name__ == '__main__':
    main()
