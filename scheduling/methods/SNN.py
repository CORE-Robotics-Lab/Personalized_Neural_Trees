"""
Created by Rohan Paleja on Sep 8, 2019
This is an implementation of the paper IMITATION LEARNING FROM VISUAL DATA WITH MULTIPLE INTENTIONS
"""

import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
from utils.naive_utils import load_in_naive_data, find_which_schedule_this_belongs_to
import numpy as np
from torch.autograd import Variable
from utils.global_utils import save_pickle
import matplotlib.pyplot as plt
import heapq

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")
epochs = 1000000


class SNN(nn.Module):
    """
    standard MLP
    """

    def __init__(self, state_size):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(state_size + 1, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(32, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, 20)

    def forward(self, x, z):
        """
        forward pass
        :param x: state
        :param z: random variable
        :return:
        """
        x = torch.cat([x, z.expand(x.shape[0], 1, 1)], dim=2)
        x = self.fc1(x)
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

        return x


# noinspection PyArgumentList
class EvalSNN:
    """
    class that handles training and evaluating this approach
    """

    def __init__(self):
        self.state_size = 242
        self.model = SNN(self.state_size)

        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.001)

        # load in data
        self.num_schedules = 150
        self.num_test_schedules = 100
        # load in data
        self.X_train_naive, self.Y_train_naive, self.schedule_array_train_naive, self.X_test_naive, self.Y_test_naive, self.schedule_array_test_naive = load_in_naive_data(
            250, 250)
        self.X_train_naive, \
        self.Y_train_naive, \
        self.schedule_array_train_naive, = self.sample_data(150)

        self.X_test_naive, \
        self.Y_test_naive, \
        self.schedule_array_test_naive, = self.sample_test_data(100)
        self.priority_queue = []
        self.use_gpu = False

        print(self.model.state_dict())
        # checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/SNN999999.tar')
        # self.model.load_state_dict(checkpoint['nn_state_dict'])

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
        Train the network
        :return:
        """
        loss_array = []
        sampling_N = 5

        for epoch in range(epochs):
            # choose a random schedule
            rand_timestep_within_sched = np.random.randint(len(self.X_train_naive))
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array_train_naive, rand_timestep_within_sched+self.sample_min)
            start_of_chosen_sched = self.schedule_array_train_naive[which_schedule][0]-self.sample_min
            end_of_chosen_sched = self.schedule_array_train_naive[which_schedule][1]-self.sample_min

            # sample z
            sampled_z = np.random.rand(sampling_N)

            # prepare network input
            network_input = torch.zeros(20, 1, 242)
            network_truth = torch.zeros(20, 1, 1)

            z_loss_array = []
            self.opt.zero_grad()

            # load in network input
            for e, i in enumerate(range(start_of_chosen_sched, end_of_chosen_sched)):
                input_nn = self.X_train_naive[i].copy()
                input_nn = torch.Tensor(np.asarray(input_nn))
                network_input[i - start_of_chosen_sched] = input_nn

                truth = self.Y_train_naive[i]
                truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).long())
                network_truth[e] = truth

            # find chosen_z
            for z in sampled_z:
                network_output = self.model.forward(network_input, torch.tensor(z))
                loss = self.criterion(network_output.reshape(20, 20), network_truth.reshape(20).long())
                z_loss_array.append(loss.item())
                self.priority_queue.append((loss.item(), z))

            # use lowest z to update network
            lowest_loss_ind = np.argmin(z_loss_array)
            network_output = self.model.forward(network_input, torch.tensor(sampled_z[lowest_loss_ind]))
            loss = self.criterion(network_output.reshape(20, 20), network_truth.reshape(20).long())
            loss.backward()
            loss_array.append(loss.item())
            self.opt.step()

            # print and save
            if epoch % 1000 == 1:
                print('average loss for last 500: ', np.mean(loss_array[-500:]))
                print('on epoch ', epoch)

            if epoch % 200000 == 199999:
                self.opt = torch.optim.SGD(self.model.parameters(), lr=.0001)
                # torch.save({'nn_state_dict': self.model.state_dict()},
                #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/SNN_no_load' + str(epoch) + '.tar')

        # torch.save({'nn_state_dict': self.model.state_dict()},
        #            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/SNN.tar')

    def test(self):
        """
        evaluate the network
        :return:
        """
        num_schedules = 100
        soft = nn.Softmax(dim=2)
        self.priority_queue.sort()
        # heapq.heapify(self.priority_queue)
        sampling_N = 5

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        set_of_zs = []
        for j, k in enumerate(self.priority_queue):
            if j >= 100:
                continue
            else:
                set_of_zs.append(k[1])

        for i, schedule in enumerate(self.schedule_array_test_naive):
            trajectory = []
            trajectory_truths = []
            for k, count in enumerate(range(schedule[0]-self.sample_test_min, schedule[1]-self.sample_test_min + 1)):

                if len(trajectory) == 0:
                    chosen_z = np.random.choice(set_of_zs)
                else:
                    chosen_set_of_zs = np.random.choice(set_of_zs, size=sampling_N)
                    z_loss_array = []

                    network_input = torch.zeros(len(trajectory), 1, 242)
                    network_truth = torch.zeros(len(trajectory), 1, 1)

                    for e, l in enumerate(trajectory):
                        input_nn = torch.Tensor(np.asarray(l))
                        network_input[e] = input_nn

                        truth = trajectory_truths[e]
                        truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).long())
                        network_truth[e] = truth


                    for z in chosen_set_of_zs:
                        network_output = self.model.forward(network_input, torch.tensor(z))
                        loss = self.criterion(network_output.reshape(len(trajectory), 20), network_truth.reshape(len(trajectory)).long())
                        z_loss_array.append(loss.item())

                    chosen_z = set_of_zs[np.argmin(z_loss_array)]

                trajectory.append(self.X_test_naive[count])
                trajectory_truths.append(self.Y_test_naive[count])
                net_input = self.X_test_naive[count]
                truth = self.Y_test_naive[count]
                if self.use_gpu:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))
                pred = self.model.forward(input_nn, torch.tensor(chosen_z))

                pred = soft(pred)
                index = torch.argmax(pred).item()

                # top 3
                _, top_three = torch.topk(pred, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0][0]:
                    prediction_accuracy[1] += 1
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)

            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            prediction_accuracy = [0, 0]
        return np.mean(percentage_accuracy_top1)
        # self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_SNN')


    def save_performance_results(self, top1, top3, special_string):
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
    main
    :return:
    """
    res = []
    for i in range(3):
        benchmark = EvalSNN()
        benchmark.train()
        out = benchmark.test()
        res.append(out)
    print(np.mean(res))
    print(np.std(res))

if __name__ == '__main__':
    main()
