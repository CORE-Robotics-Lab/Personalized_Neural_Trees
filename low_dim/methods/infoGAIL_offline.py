"""
Created by Anonymous on September 24, 2019
infoGAIL benchmark in the toy domain
"""
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import math

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(49)  # ensures repeatability
np.random.seed(49)
from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from torch.distributions import OneHotCategorical
from base_testing_environment.utils.helper_utils import save_performance_results
from scheduling_env.additions_for_HRI.infogail_scheduling import one_hot_embedding, my_entropy


# TODO: jumpstart if does not work

class PolicyNetwork(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 embedding_dim):
        super(PolicyNetwork, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=0)
        )

    def forward(self, state, latent_code):
        input_data = torch.cat((state, latent_code), dim=0)
        return self.forward_pass(input_data)


class Discriminator(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim):
        super(Discriminator, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
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
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.Softmax(dim=0)
        )

    def forward(self, state, action):
        input_data = torch.cat((state, action), dim=0)
        return self.forward_pass(input_data)


# noinspection PyArgumentList
class InfoGAIL:
    """
    class structure to test infoGAIL benchmark
    """

    def __init__(self):
        # Training set generation
        self.num_schedules = 50
        it_1 = [True, False, False]
        it_2 = [False, True, False]
        it_3 = [False, False, True]
        it = [it_2, it_3, it_1]
        x_data, y = create_simple_classification_dataset(self.num_schedules, train=it[0][0], cv=it[0][1])

        x = []
        for each_ele in x_data:
            x.append(each_ele[2:])

        self.X = torch.Tensor(x).reshape(-1, 2)
        self.Y = torch.Tensor(y).reshape((-1, 1))

        state_dimension = 2
        action_dimension = 2
        embedding_dimension = 2

        # Initialise the network.
        device = torch.device("cpu")

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

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=.0001)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=.001)
        self.distro_opt = optim.Adam(self.distribution_gen.parameters(), lr=.0001)  # policy opt could be
        self.schedule_starts = np.linspace(0, 20 * (self.num_schedules - 1), num=self.num_schedules)
        self.gamma = .95



    def train(self):
        epochs = 50000
        lambda_1 = 1
        lambda_2 = 0
        for epoch in range(epochs):
            discriminator_acc = 0
            avg_policy_loss = 0
            avg_discrim_loss = 0
            # sample a timestep before the cutoff for cross_validation
            chosen_schedule_start = int(np.random.choice(self.schedule_starts))
            schedule_num = int(chosen_schedule_start / 20)
            # choose a random mode
            z = np.random.choice([0, 1], p=[1 / 2, 1 / 2])  # sample embedding from prior distribution
            if z == 0:
                mode = torch.Tensor([1, 0])
            else:
                mode = torch.Tensor([0, 1])

            for _, time in enumerate(range(chosen_schedule_start, chosen_schedule_start + 20)):
                R = 0
                for t, i in enumerate(range(time, chosen_schedule_start + 20)):
                    input_nn = self.X[i]
                    truth = self.Y[i]

                    self.disc_opt.zero_grad()

                    # predict action the demonstrator would take
                    self.policy.requires_grad = False
                    self.distribution_gen.requires_grad = False
                    self.discriminator.requires_grad = True
                    action_fake = self.policy.forward(input_nn.reshape(2), mode)

                    # Discriminator training
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()

                    # want high probability of labeling policy action as fake
                    fake_action_label = self.discriminator(input_nn.reshape(2), one_hot_encoded_action_fake_dist) + 1 * 10 ** -6  # should be 1
                    # print(fake_action_label)
                    discriminator_acc += fake_action_label.item()
                    # loss D_part 1
                    J_D_1 = -torch.log(fake_action_label)  # if .1, will be a big negative

                    # want low probability of labeling expert as fake
                    real_action_label = self.discriminator(input_nn.reshape(2), one_hot_embedding(int(truth.item()), 2).reshape(2)) + 1 * 10 ** -6  # should be 0
                    # print(real_action_label)
                    discriminator_acc += (1 - real_action_label.item())
                    # loss d_part 2
                    J_D_2 = -torch.log(1.01 - real_action_label)

                    # if real_action_label <= .51 + 1 * 10**-6:
                    #     J_D_3 = 10
                    # else:
                    #     J_D_3 = 0
                    loss_discrim = J_D_1 + J_D_2  # now it is positive, and gets minimized
                    print(loss_discrim)
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
                    action_fake = self.policy.forward(input_nn.reshape(2), mode)
                    # Discriminator training
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()
                    L_Q = -1 * lambda_1 * torch.log(
                        self.distribution_gen(input_nn.reshape(2), one_hot_encoded_action_fake_dist)[
                            z] + 10 ** -6)  # you want to maximize the prob of actual mode
                    L_Q.backward()
                    self.distro_opt.step()
                    discriminator_acc += (1 - fake_action_label.item())

                    # POLICY
                    self.policy.requires_grad = False
                    self.distribution_gen.requires_grad = False
                    self.discriminator.requires_grad = False

                    action_fake = self.policy.forward(input_nn.reshape(2), mode)
                    one_hot_encoded_action_fake_dist = OneHotCategorical(probs=action_fake).sample()
                    fake_action_label = self.discriminator(input_nn.reshape(2), one_hot_encoded_action_fake_dist) + 1 * 10 ** -6  # should be 0

                    L_pi_1 = -torch.log(1.01 - fake_action_label)  # we want this to be minimized now, since it should think action_fake is 0 or real
                    L_pi_2 = -1 * lambda_1 * torch.log(
                        self.distribution_gen(input_nn.reshape(2), one_hot_encoded_action_fake_dist)[
                            z] + 10 ** -6)  # you want to maximize the prob of actual mode
                    L_pi_3 = lambda_2 * -1 * my_entropy(self.policy(input_nn.reshape(2), mode))  # you want this to be 0. maximizing entropy

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
                input_nn = self.X[time]
                truth_nn = self.Y[time]

                policy_loss = -R * torch.log(self.policy(input_nn.reshape(2), mode)[int(truth_nn.item())])
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

            if epoch % 400 == 1:
                # print('discrim', self.discriminator.state_dict())
                # print('policy', self.policy.state_dict())
                # print('dist_gen', self.distribution_gen.state_dict())
                # print('epoch is ', epoch)
                # torch.save({'policy_state_dict': self.policy.state_dict(),
                #             'discrim_state_dict': self.discriminator.state_dict(),
                #             'distribution_gen_state_dict': self.distribution_gen.state_dict()},
                #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/toy_infoGAIL' + str(epoch) + '.tar')
                self.evaluate_on_test_data()
        # torch.save({'policy_state_dict': self.policy.state_dict(),
        #             'discrim_state_dict': self.discriminator.state_dict(),
        #             'distribution_gen_state_dict': self.distribution_gen.state_dict()},
        #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/toy_infoGAIL.tar')
        #


    def evaluate_on_test_data(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """

        it_1 = [True, False, False]
        it_2 = [False, True, False]
        it_3 = [False, False, True]
        it = [it_2, it_3, it_1]
        x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, True, train=it[2][0], cv=it[2][1])

        x_test = []
        for each_ele in x_data_test:
            x_test.append(each_ele[2:])

        x_test = torch.Tensor(x_test).reshape(-1, 2)
        y_test = torch.Tensor(y_test).reshape((-1, 1))
        test_accs = []
        per_schedule_test_losses, per_schedule_test_accs = [], []
        preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]

        for i in range(50):
            chosen_schedule_start = int(self.schedule_starts[i])
            z = np.random.choice([0, 1], p=[.5, .5])  # sample embedding from prior distribution
            if z == 0:
                mode = torch.Tensor([0, 1])
            else:
                mode = torch.Tensor([1, 0])

            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):

                state = x_test[each_t]
                truth = y_test[each_t]

                # forward
                output = self.policy.forward(state.reshape(2), mode)

                index = torch.argmax(output).item()
                preds[i].append(output.argmax(dim=-1).item())
                actual[i].append(y_test[each_t].item())
                print(index, truth)
                acc = (output.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()

                test_accs.append(acc.mean().item())

                dist = self.distribution_gen(state.reshape(2), one_hot_embedding(int(truth.item()), 2).reshape(2))

                z = np.random.choice([0, 1],
                                     p=[dist[0].item() / (dist[0].item() + dist[1].item()),
                                        dist[1].item() / (dist[0].item() + dist[1].item())])  # sample embedding from prior distribution

                if z == 0:
                    mode = torch.Tensor([1, 0])
                elif z == 1:
                    mode = torch.Tensor([0, 1])
            per_schedule_test_accs.append(np.mean(test_accs))
            #
            # print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)
            #
            # print('schedule num:', i)
        print('Accuracy: {}'.format(np.mean(per_schedule_test_accs)))
        print('Accuracy: {}'.format(np.std(test_accs)))

        #     percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
        #     percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
        #     prediction_accuracy = [0, 0]
        #
        # print(np.mean(percentage_accuracy_top1))
        # save_performance_results(per_schedule_test_accs, 'results_infoGAIL')


def main():
    benchmark = InfoGAIL()
    benchmark.train()
    benchmark.evaluate_on_test_data()


if __name__ == '__main__':
    main()
