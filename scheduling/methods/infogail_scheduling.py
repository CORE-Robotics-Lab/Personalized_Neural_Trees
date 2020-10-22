"""
Testing the NN_small. This is expected to do much worse than the BDDT
"""

import torch
import torch.nn as nn
from torch import optim
import math

import numpy as np
from torch.autograd import Variable
from utils.global_utils import save_pickle
from utils.naive_utils import load_in_naive_data, find_which_schedule_this_belongs_to

from torch.distributions import OneHotCategorical

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
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
        self.fc3 = nn.Linear(32, 20)
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
        self.num_schedules = 150
        self.num_test_schedules = 100
        self.total_loss_array = []
        jumpstart = False
        self.X_train_naive, self.Y_train_naive, self.schedule_array_train_naive, self.X_test_naive, self.Y_test_naive, self.schedule_array_test_naive = load_in_naive_data(
            250, 250)
        self.X_train_naive, \
        self.Y_train_naive, \
        self.schedule_array_train_naive, = self.sample_data(150)

        self.X_test_naive, \
        self.Y_test_naive, \
        self.schedule_array_test_naive, = self.sample_test_data(100)

        device = torch.device("cpu")
        state_dimension = 242
        action_dimension = 20
        embedding_dimension = 3
        self.gamma = .95
        self.use_gpu = False
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
            checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/models/NN_jumpstart_NN999999.tar')
            self.policy.load_state_dict(checkpoint['nn_state_dict'])

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=.0001)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=.001)
        self.distro_opt = optim.Adam(self.distribution_gen.parameters(), lr=.0001)  # policy opt could be


    def sample_data(self, size):
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250 - size)
        self.sample_min = set_of_twenty * 20
        return self.X_train_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.Y_train_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.schedule_array_train_naive[set_of_twenty:set_of_twenty + size]

    def sample_test_data(self, size):
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250 - size)
        self.sample_test_min = set_of_twenty * 20
        return self.X_test_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.Y_test_naive[set_of_twenty * 20:set_of_twenty * 20 + size * 20], \
               self.schedule_array_test_naive[set_of_twenty:set_of_twenty + size]




    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """

        epochs = 10000
        lambda_1 = 1
        lambda_2 = 0

        for epoch in range(epochs):
            discriminator_acc = 0
            avg_policy_loss = 0
            avg_discrim_loss = 0
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X_train_naive))
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array_train_naive, rand_timestep_within_sched+self.sample_min)

            # choose a random mode
            z = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])  # sample embedding from prior distribution
            if z == 0:
                mode = torch.Tensor([1, 0, 0])
            elif z == 1:
                mode = torch.Tensor([0, 1, 0])
            else:
                mode = torch.Tensor([0, 0, 1])

            start_of_chosen_sched = self.schedule_array_train_naive[which_schedule][0]-self.sample_min
            end_of_chosen_sched = self.schedule_array_train_naive[which_schedule][1]-self.sample_min
            for _, time in enumerate(range(start_of_chosen_sched, end_of_chosen_sched + 1)):
                R = 0
                for t, i in enumerate(range(time, end_of_chosen_sched + 1)):
                    input_nn = self.X_train_naive[i]
                    truth_nn = self.Y_train_naive[i]
                    if self.use_gpu:
                        input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                        truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
                    else:
                        input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                        truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

                    self.disc_opt.zero_grad()

                    # predict action the demonstrator would take
                    self.policy.requires_grad = False
                    self.distribution_gen.requires_grad = False
                    self.discriminator.requires_grad = True
                    action_fake = self.policy.forward(input_nn.reshape(242), mode)

                    # Discriminator training
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()

                    # want high probability of labeling policy action as fake
                    fake_action_label = self.discriminator(input_nn.reshape(242), one_hot_encoded_action_fake_dist) + 1 * 10 ** -6  # should be 1
                    # print(fake_action_label)
                    discriminator_acc += fake_action_label.item()
                    # loss D_part 1
                    J_D_1 = -torch.log(fake_action_label)  # if .1, will be a big negative

                    # want low probability of labeling expert as fake
                    real_action_label = self.discriminator(input_nn.reshape(242), one_hot_embedding(truth, 20).reshape(20)) + 1 * 10 ** -6  # should be 0
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
                    action_fake = self.policy.forward(input_nn.reshape(242), mode)
                    # Discriminator training
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()
                    L_Q = -1 * lambda_1 * torch.log(
                        self.distribution_gen(input_nn.reshape(242), one_hot_encoded_action_fake_dist)[
                            z] + 10 ** -6)  # you want to maximize the prob of actual mode
                    L_Q.backward()
                    self.distro_opt.step()
                    discriminator_acc += (1 - fake_action_label.item())

                    # POLICY
                    self.policy.requires_grad = False
                    self.distribution_gen.requires_grad = False
                    self.discriminator.requires_grad = False

                    action_fake = self.policy.forward(input_nn.reshape(242), mode)
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()
                    fake_action_label = self.discriminator(input_nn.reshape(242), one_hot_encoded_action_fake_dist) + 1 * 10 ** -6  # should be 0

                    L_pi_1 = -torch.log(1.01 - fake_action_label)  # we want this to be minimized now, since it should think action_fake is 0 or real
                    L_pi_2 = -1 * lambda_1 * torch.log(
                        self.distribution_gen(input_nn.reshape(242), one_hot_encoded_action_fake_dist)[z] + 10**-6)  # you want to maximize the prob of actual mode
                    L_pi_3 = lambda_2 * -1 * my_entropy(self.policy(input_nn.reshape(242), mode))  # you want this to be 0. maximizing entropy

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
                input_nn = self.X_train_naive[time]
                truth_nn = self.Y_train_naive[time]
                if self.use_gpu:
                    input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                    truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())
                policy_loss = -R * torch.log(self.policy(input_nn.reshape(242), mode)[int(truth.item())])
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
            if epoch % 5 == 4:
                print(epoch)
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

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for i, schedule in enumerate(self.schedule_array_test_naive):

            # start off with a random embedding
            z = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
            if z == 0:
                mode = torch.Tensor([1, 0, 0])
            elif z == 1:
                mode = torch.Tensor([0, 1, 0])
            else:
                mode = torch.Tensor([0, 0, 1])

            for count in range(schedule[0]-self.sample_test_min, schedule[1]-self.sample_test_min + 1):

                net_input = self.X_test_naive[count]
                truth = self.Y_test_naive[count]

                if self.use_gpu:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())

                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))

                # forward
                output = self.policy.forward(input_nn.reshape(242), mode)

                index = torch.argmax(output).item()
                print(index, truth)
                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist():
                    prediction_accuracy[1] += 1

                dist = self.distribution_gen(input_nn.reshape(242), one_hot_embedding(int(truth.item()), 20).reshape(20))

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

            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)

            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            prediction_accuracy = [0, 0]

        print(np.mean(percentage_accuracy_top1))
        # self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_infoGAIL')
        return np.mean(percentage_accuracy_top1)

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
        save_pickle(file=data, file_location='/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/results',
                    special_string=special_string)


def main():
    """
    entry point for file
    :return:
    """
    res = []
    for i in range(3):
        benchmark = InfoGAIL()
        benchmark.train()
        out = benchmark.evaluate_on_test_data()
        res.append(out)
    print(np.mean(res))
    print(np.std(res))


if __name__ == '__main__':
    main()
