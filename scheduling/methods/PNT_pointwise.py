"""
Created by Rohan Paleja on Month Day, Year
Purpose:
"""
import torch
import sys
import torch.nn as nn
from datetime import date
sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')
from low_dim.prolonet import ProLoNet
import numpy as np
from torch.autograd import Variable
from utils.pairwise_utils import load_in_pairwise_data, find_which_schedule_this_belongs_to
from utils.hri_utils import save_trained_nets, save_performance_results
sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


# noinspection PyArgumentList
class ProLoTrain:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self):
        self.total_loss_array = []
        self.bayesian_embedding_dim = 3
        self.num_schedules = 150
        self.num_test_schedules = 100
        self.X_train_pairwise, self.Y_train_pairwise, self.schedule_array_train_pairwise, self.start_of_each_set_twenty_train, self.X_test_pairwise, self.Y_test_pairwise, self.schedule_array_test_pairwise, self.start_of_each_set_twenty_test = load_in_pairwise_data(
            self.num_schedules, self.num_test_schedules)
        use_gpu = True
        self.model = ProLoNet(input_dim=len(self.X_train_pairwise[0]),
                              weights=None,
                              comparators=None,
                              leaves=32,
                              output_dim=1,
                              bayesian_embedding_dim=self.bayesian_embedding_dim,
                              alpha=1.5,
                              use_gpu=use_gpu,
                              vectorized=True,
                              is_value=True)

        if use_gpu:
            self.model = self.model.cuda()
        print(self.model.state_dict())
        self.opt = torch.optim.RMSprop(
            [{'params': list(self.model.parameters())[:-1]}, {'params': self.model.bayesian_embedding.parameters(), 'lr': .01}], lr=.01)
        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.when_to_save = 1000
        self.embedding_list = [torch.ones(self.bayesian_embedding_dim) * 1 / 3 for _ in range(self.num_schedules)]


    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """
        # loss = nn.CrossEntropyLoss()

        epochs = 50000
        criterion = torch.nn.BCELoss()

        # variables to keep track of loss and number of tasks trained over
        for epoch in range(epochs):
            # sample a timestep before the cutoff for cross_validation
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty_train)
            truth = self.Y_train_pairwise[set_of_twenty]
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array_train_pairwise, set_of_twenty)
            self.model.set_bayesian_embedding(self.embedding_list[which_schedule].clone())
            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty
            phi_i = self.X_train_pairwise[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)
            running_loss_predict_tasks = 0
            num_iterations_predict_task = 0
            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    label = torch.ones((1, 1))
                else:
                    label = torch.zeros((1, 1))
                phi_j = self.X_train_pairwise[counter]
                phi = np.asarray(phi_j)
                feature_input = phi

                if torch.cuda.is_available():
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                    label = Variable(torch.Tensor(label).cuda())
                else:
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                    label = Variable(torch.Tensor(label.reshape(1, 1)))

                output = self.model.forward(feature_input)
                sig = torch.nn.Sigmoid()
                output = sig(output)

                self.opt.zero_grad()
                loss = criterion(output, label)
                if counter == phi_i_num:
                    loss *= 20
                # print(self.total_iterations)
                if torch.isnan(loss):
                    print('nan occurred at iteration ', self.total_iterations, ' at', num_iterations_predict_task)
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()
                running_loss_predict_tasks += loss.item()
                num_iterations_predict_task += 1

            self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)
            self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()).clone()
            self.total_iterations += 1

            if self.total_iterations % 50 == 49:
                print('total loss (average for each 20, averaged) at iteration ', self.total_iterations, ' is ', np.mean(self.total_loss_array[-20:]))

            if epoch % self.when_to_save == self.when_to_save - 1:
                save_trained_nets(self.model, 'pointwise_PDDT_small_tree' + str(epoch))
                self.evaluate_on_test_data(self.model)

    def evaluate_on_test_data(self, model, load_in_model=False):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        embedding_optimizer = torch.optim.SGD([{'params': self.model.bayesian_embedding.parameters()}], lr=.01)
        criterion = torch.nn.BCELoss()
        sig = torch.nn.Sigmoid()

        if load_in_model:
            model.load_state_dict(
                torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/model_homog.tar')['nn_state_dict'])

        embedding_list = [torch.ones(self.bayesian_embedding_dim) * 1 / 3 for i in range(self.num_test_schedules)]

        for j in range(0, self.num_test_schedules):
            schedule_bounds = self.schedule_array_test_pairwise[j]
            step = schedule_bounds[0]
            model.set_bayesian_embedding(embedding_list[j])
            while step < schedule_bounds[1]:
                probability_vector = np.zeros((1, 20))
                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = self.X_test_pairwise[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    feature_input = phi_i_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                    # push through nets
                    preference_prob = model.forward(feature_input)
                    preference_prob = sig(preference_prob)
                    probability_vector[0][m] = preference_prob[0].data.detach()[
                        0].item()

                embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

                # print(probability_vector)
                highest_val = max(probability_vector[0])
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(probability_vector[0])) if e == highest_val]
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)
                # choice = np.argmax(probability_vector)

                # top 3
                _, top_three = torch.topk(torch.Tensor(probability_vector), 3)

                # Then do training update loop
                truth = self.Y_test_pairwise[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three[0]:
                    prediction_accuracy[1] += 1

                # update loop
                phi_i_num = truth + step  # old method: set_of_twenty[0] + truth
                for counter in range(step, step + 20):
                    if counter == phi_i_num:  # if counter == phi_i_num:
                        label = torch.ones((1, 1))
                    else:
                        label = torch.zeros((1, 1))
                    phi_j = self.X_test_pairwise[counter]
                    phi = np.asarray(phi_j)
                    feature_input = phi

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        label = Variable(torch.Tensor(label).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        label = Variable(torch.Tensor(label.reshape(1, 1)))

                    output = model.forward(feature_input)
                    output = sig(output)

                    embedding_optimizer.zero_grad()
                    loss = criterion(output, label)
                    if counter == phi_i_num:
                        loss *= 20
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    embedding_optimizer.step()
                    # print(model.EmbeddingList.state_dict())
                # add average loss to array
                step += 20

            # schedule finished
            # print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)
            #
            # print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'PDDT_pointwise' + str(date.today()))
        print('acc: ', np.mean(percentage_accuracy_top1))

def main():
    """
    entry point for file
    :return:
    """

    trainer = ProLoTrain()
    trainer.train()
    trainer.evaluate_on_test_data(trainer.model)


if __name__ == '__main__':
    main()
