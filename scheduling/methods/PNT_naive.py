"""
Created by Anonymous on September 13, 2019
Implementation of PDDT_naive since it was shown to perform so well in neurips
"""

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')
from base_testing_environment.prolonet import ProLoNet

import numpy as np
from torch.autograd import Variable
from utils.global_utils import save_pickle
from utils.naive_utils import load_in_naive_data, find_which_schedule_this_belongs_to

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


def save_performance_results(top1, top3, special_string):
    """
    saves performance of top1 and top3
    :return:
    """
    print('top1_mean is : ', np.mean(top1))
    data = {'top1_mean': np.mean(top1),
            'top3_mean': np.mean(top3),
            'top1_stderr': np.std(top1) / np.sqrt(len(top1)),
            'top3_stderr': np.std(top3) / np.sqrt(len(top3))}
    save_pickle(file=data, file_location='/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/results', special_string=special_string)


# noinspection PyTypeChecker,PyArgumentList
class ProLoTrain:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self, num_schedules):
        self.num_schedules = num_schedules
        self.total_loss_array = []

        self.num_test_schedules = 100
        self.X_train_naive, self.Y_train_naive, self.schedule_array_train_naive, self.X_test_naive, self.Y_test_naive, self.schedule_array_test_naive = load_in_naive_data(
            self.num_schedules, self.num_test_schedules)

        use_gpu = True
        self.model = ProLoNet(input_dim=len(self.X_train_naive[0]),
                              weights=None,
                              comparators=None,
                              leaves=128,
                              output_dim=20,
                              bayesian_embedding_dim=3,
                              alpha=1.5,
                              use_gpu=use_gpu,
                              vectorized=True,
                              is_value=False)

        if use_gpu:
            self.model = self.model.cuda()
        print(self.model.state_dict())
        self.opt = torch.optim.RMSprop([{'params': list(self.model.parameters())[:-1]}, {'params': self.model.bayesian_embedding.parameters(), 'lr': .01}])

        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.when_to_save = 1000

        self.embedding_list = [torch.ones(3) * 1 / 3 for _ in range(self.num_schedules)]

    def train(self):
        """
        Trains PDDT.
        :return:
        """

        epochs = 1000000
        for epoch in range(epochs):
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X_train_naive))
            input_nn = self.X_train_naive[rand_timestep_within_sched]
            truth_nn = self.Y_train_naive[rand_timestep_within_sched]

            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array_train_naive, rand_timestep_within_sched)
            self.model.set_bayesian_embedding(self.embedding_list[which_schedule].clone())

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()

            self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()).clone()  # very ugly

            # add average loss to array
            # print(list(self.model.parameters()))

            self.total_loss_array.append(loss.item())

            if epoch % 50 == 49:
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))
                # print(self.model.state_dict())

            if epoch % self.when_to_save == self.when_to_save - 1:
                self.save_trained_nets('PDDT_small_naive' + str(epoch))

    def evaluate_on_test_data(self):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient

        embedding_optimizer = torch.optim.SGD([{'params': self.model.bayesian_embedding.parameters()}], lr=.1)
        embedding_list = [torch.ones(3) * 1 / 3 for _ in range(self.num_test_schedules)]

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for i, schedule in enumerate(self.schedule_array_test_naive):
            self.model.set_bayesian_embedding(embedding_list[i])

            for count in range(schedule[0], schedule[1] + 1):

                net_input = self.X_test_naive[count]
                truth = self.Y_test_naive[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).long())

                # forward
                output = self.model.forward(input_nn)
                embedding_optimizer.zero_grad()
                loss = F.cross_entropy(output, truth)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                embedding_optimizer.step()

                index = torch.argmax(output).item()

                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0]:
                    prediction_accuracy[1] += 1

            # add average loss to array
            embedding_list[i] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        print('accuracy is ',  np.mean(percentage_accuracy_top1))
        # save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_PDDT_naive')

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'training_embeddings': self.embedding_list},
                   '/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/PDDT_' + name + '.tar')


def main():
    """
    entry point for file
    :return:
    """
    num_schedules = 150
    trainer = ProLoTrain(num_schedules)
    trainer.train()
    trainer.evaluate_on_test_data()



if __name__ == '__main__':
    main()
