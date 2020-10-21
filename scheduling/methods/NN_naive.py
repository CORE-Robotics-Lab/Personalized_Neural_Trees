"""
Created by Anonymous on September 23, 2019
Display performance of Sammut et. al. benchmark
"""


import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')
import numpy as np
from torch.autograd import Variable
from utils.naive_utils import  load_in_naive_data
from utils.hri_utils import save_performance_results, save_trained_nets
# sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


class NNSmall(nn.Module):
    """
    Sammut et. al. benchmark
    """

    def __init__(self):
        super(NNSmall, self).__init__()
        self.fc1 = nn.Linear(242, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(128, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, 20)
        self.soft = nn.LogSoftmax(dim=0)

    def forward(self, x):
        """
        forward pass
        :param x: i_minus_j or vice versa
        :return:
        """
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
        # x = self.soft(x)

        return x


# noinspection PyTypeChecker,PyArgumentList
class NNTrain:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self):
        self.num_schedules = 150
        self.num_test_schedules = 100
        self.total_loss_array = []

        self.X_train_naive, self.Y_train_naive, self.schedule_array_train_naive, self.X_test_naive, self.Y_test_naive, self.schedule_array_test_naive = load_in_naive_data(
            250,250)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNSmall().to(device)

        print(self.model.state_dict())
        self.opt = torch.optim.Adam(self.model.parameters())
        self.when_to_save = 1000

        self.X_train_naive, \
        self.Y_train_naive, \
        self.schedule_array_train_naive,  = self.sample_data(150)

        self.X_test_naive, \
        self.Y_test_naive, \
        self.schedule_array_test_naive,  = self.sample_test_data(100)
        self.num_test_schedules = 100


    def sample_data(self, size):
        # return self.X_train_naive[0:size * 20 * 20], \
        #        self.Y_train_naive[0:size * 20 * 20], \
        #        self.schedule_array_train_naive[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250-size)
        self.sample_min = set_of_twenty * 20
        return self.X_train_naive[set_of_twenty*20:set_of_twenty*20 + size * 20], \
               self.Y_train_naive[set_of_twenty*20:set_of_twenty*20 + size * 20], \
               self.schedule_array_train_naive[set_of_twenty:set_of_twenty+size]

    def sample_test_data(self, size):
        # return self.X_train_naive[0:size * 20 * 20], \
        #        self.Y_train_naive[0:size * 20 * 20], \
        #        self.schedule_array_train_naive[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250-size)
        self.sample_test_min = set_of_twenty * 20
        return self.X_test_naive[set_of_twenty*20:set_of_twenty*20 + size * 20], \
               self.Y_test_naive[set_of_twenty*20:set_of_twenty*20 + size * 20], \
               self.schedule_array_test_naive[set_of_twenty:set_of_twenty+size]

    def train(self):
        """
        Trains NN.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """

        epochs = 400000
        for epoch in range(epochs):

            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X_train_naive))
            input_nn = self.X_train_naive[rand_timestep_within_sched]
            truth_nn = self.Y_train_naive[rand_timestep_within_sched]

            # iterate over pairwise comparisons
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

            self.opt.zero_grad()
            output = self.model.forward(input_nn)
            loss = F.cross_entropy(output, truth)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()

            self.total_loss_array.append(loss.item())

            if epoch % 1000 == 999:
                print('loss at', epoch, ', total loss (average for each 100, averaged)', np.mean(self.total_loss_array[-100:]))
                # print(self.model.state_dict())
                # save_trained_nets(self.model, 'naive_NN' + str(epoch))

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
            for count in range(schedule[0]-self.sample_test_min, schedule[1] -self.sample_test_min + 1):

                net_input = self.X_test_naive[count]
                truth = self.Y_test_naive[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))

                # forward
                output = self.model.forward(input_nn)

                index = torch.argmax(output).item()

                # confusion_matrix[truth][index] += 1
                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0]:
                    prediction_accuracy[1] += 1

            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)

            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            prediction_accuracy = [0, 0]

        print(np.mean(percentage_accuracy_top1))
        # save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_NN_naive')
        return np.mean(percentage_accuracy_top1)


def main():
    """
    entry point for file
    :return:
    """
    res = []
    for i in range(3):
        trainer = NNTrain()
        trainer.train()
        out = trainer.evaluate_on_test_data()
        res.append(out)
    print(np.mean(res))
    print(np.std(res))

if __name__ == '__main__':
    main()