"""
Created by Rohan Paleja on Sep 9, 2019
"""

import torch
import sys
import torch.nn as nn
from AndrewSilva.tree_nets.utils.fuzzy_to_crispy import convert_to_crisp

sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')
from base_testing_environment.prolonet import ProLoNet
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
from utils.global_utils import save_pickle
from utils.pairwise_utils import create_new_data, find_which_schedule_this_belongs_to, create_sets_of_20_from_x_for_pairwise_comparisions
from utils.pairwise_utils import load_in_pairwise_data

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

    def __init__(self, num_schedules, i):
        self.arguments = Logger()
        self.num_schedules = num_schedules
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []
        self.num_test_schedules = 100
        self.X_train_pairwise, self.Y_train_pairwise, self.schedule_array_train_pairwise, self.start_of_each_set_twenty_train, self.X_test_pairwise, self.Y_test_pairwise, self.schedule_array_test_pairwise, self.start_of_each_set_twenty_test = load_in_pairwise_data(
            250,250)

        self.X_train_pairwise, \
        self.Y_train_pairwise, \
        self.schedule_array_train_pairwise, \
        self.start_of_each_set_twenty_train = self.sample_data(150)

        self.X_test_pairwise, \
        self.Y_test_pairwise, \
        self.schedule_array_test_pairwise, \
        self.start_of_each_set_twenty_test = self.sample_test_data(100)

        use_gpu = False
        self.use_gpu = use_gpu
        self.model = ProLoNet(input_dim=len(self.X_train_pairwise[0]),
                              weights=None,
                              comparators=None,
                              leaves=32,
                              output_dim=1,
                              bayesian_embedding_dim=8,
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
        self.covergence_epsilon = .01
        self.when_to_save = 1000
        self.embedding_list = [torch.ones(8) * 1 / 3 for _ in range(self.num_schedules)]

        checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/all_PDDT_pairwise_info.tar')
        self.model.load_state_dict(checkpoint['nn_state_dict'])
        checkpoint = torch.load(
            '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/HRI/test_embedding_' + str(i + 1) + '.tar')
        # self.training_embeddings = checkpoint['training_embeddings']
        self.testing_embeddings = checkpoint['test_embeddings']

    def sample_data(self, size):
        # return self.X_train_pairwise[0:size * 20 * 20], \
        #        self.Y_train_pairwise[0:size * 20 * 20], \
        #        self.schedule_array_train_pairwise[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250-size)
        self.sample_min = set_of_twenty * 400
        return self.X_train_pairwise[set_of_twenty*400:set_of_twenty*400 + size * 20 * 20], \
               self.Y_train_pairwise[set_of_twenty*400:set_of_twenty*400 + size * 20 * 20], \
               self.schedule_array_train_pairwise[set_of_twenty:set_of_twenty+size], \
               self.start_of_each_set_twenty_train[set_of_twenty*20:set_of_twenty*20+size * 20]


    def sample_test_data(self, size):
        # return self.X_train_pairwise[0:size * 20 * 20], \
        #        self.Y_train_pairwise[0:size * 20 * 20], \
        #        self.schedule_array_train_pairwise[0:size], \
        #        self.start_of_each_set_twenty_train[0:size * 20]
        if size == 250:
            set_of_twenty = 0
        else:
            set_of_twenty = np.random.randint(250 - size)
        self.sample_test_min = set_of_twenty * 400
        return self.X_test_pairwise[set_of_twenty * 400:set_of_twenty * 400 + size * 20 * 20], \
               self.Y_test_pairwise[set_of_twenty * 400:set_of_twenty * 400 + size * 20 * 20], \
               self.schedule_array_test_pairwise[set_of_twenty:set_of_twenty + size], \
               self.start_of_each_set_twenty_test[set_of_twenty * 20:set_of_twenty * 20 + size * 20]


    def evaluate_on_test_data(self, model, load_in_model=False):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient
        num_schedules = 100
        # load in new data
        load_directory = '/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/scheduling_dataset/' + str(
            num_schedules) + 'EDF_DIST_8_28_2019_test_pairwise.pkl'
        sig = torch.nn.Sigmoid()
        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        embedding_optimizer = torch.optim.SGD([{'params': self.model.bayesian_embedding.parameters()}], lr=.01)
        criterion = torch.nn.BCELoss()

        embedding_list = [torch.ones(3) * 1 / 3 for i in range(num_schedules)]

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            model.set_bayesian_embedding(embedding_list[j])

            while step < schedule_bounds[1]:
                probability_matrix = np.zeros((20, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = X[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagonals set to zero
                        if second_counter == counter:  # same as m = n
                            continue
                        phi_j = X[second_counter]
                        phi_j_numpy = np.asarray(phi_j)

                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                        # push through nets
                        preference_prob = model.forward(feature_input)
                        sig = torch.nn.Sigmoid()
                        preference_prob = sig(preference_prob)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[
                            0].item()
                        # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)

                embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

                # top 1
                # given all inputs, and their liklihood of being scheduled, predict the output
                highest_val = max(column_vec)
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
                if len(all_indexes_that_have_highest_val) > 1:
                    print('length of indexes greater than 1: ', all_indexes_that_have_highest_val)
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)
                # choice = np.argmax(probability_vector)

                # top 3
                _, top_three = torch.topk(torch.Tensor(column_vec), 3)

                # Then do training update loop
                truth = Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # Then do training update loop

                phi_i_num = truth + step
                phi_i = X[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                # iterate over pairwise comparisons
                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.ones((1, 1))))

                        output = model(feature_input)
                        output = sig(output)
                        loss = criterion(output, label)
                        # prepare optimizer, compute gradient, update params

                        embedding_optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        embedding_optimizer.step()
                        # print(model.EmbeddingList.state_dict())

                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.zeros((1, 1))))

                        output = model.forward(feature_input)
                        output = sig(output)

                        embedding_optimizer.zero_grad()
                        loss = criterion(output, label)

                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        embedding_optimizer.step()
                        # print(model.EmbeddingList.state_dict())
                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            prediction_accuracy = [0, 0]
        # self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'PDDT_pairwise'+ str(self.num_schedules))
        return embedding_list

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/9062019_' + name + '.tar')

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
        save_pickle(file=data, file_location='/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/results',
                    special_string=special_string)

    def test_again_fuzzy(self, model, test_embeddings):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient
        num_schedules = 100
        # load in new data
        load_directory = '/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/scheduling_dataset/' + str(
            num_schedules) + 'EDF_DIST_8_28_2019_test_pairwise.pkl'
        sig = torch.nn.Sigmoid()
        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        # embedding_optimizer = torch.optim.SGD([{'params': self.model.bayesian_embedding.parameters()}], lr=.01)
        # criterion = torch.nn.BCELoss()

        embedding_list = test_embeddings

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            model.set_bayesian_embedding(embedding_list[j])

            while step < schedule_bounds[1]:
                probability_matrix = np.zeros((20, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = X[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagonals set to zero
                        if second_counter == counter:  # same as m = n
                            continue
                        phi_j = X[second_counter]
                        phi_j_numpy = np.asarray(phi_j)

                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                        # push through nets
                        preference_prob = model.forward(feature_input)
                        preference_prob = sig(preference_prob)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[
                            0].item()
                        # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)

                embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

                # top 1
                # given all inputs, and their liklihood of being scheduled, predict the output
                highest_val = max(column_vec)
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
                if len(all_indexes_that_have_highest_val) > 1:
                    print('length of indexes greater than 1: ', all_indexes_that_have_highest_val)
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)
                # choice = np.argmax(probability_vector)

                # top 3
                _, top_three = torch.topk(torch.Tensor(column_vec), 3)

                # Then do training update loop
                truth = Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            prediction_accuracy = [0, 0]
        print(np.mean(prediction_accuracy[0]))
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_PDDT_pairwise_fuzzy')

    def test_again_crisp(self, model, test_embeddings):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient

        self.model = convert_to_crisp(model, None)
        sig = torch.nn.Sigmoid()

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        # embedding_optimizer = torch.optim.SGD([{'params': self.model.bayesian_embedding.parameters()}], lr=.01)
        # criterion = torch.nn.BCELoss()

        embedding_list = test_embeddings

        for j in range(0, self.num_test_schedules):
            schedule_bounds = self.schedule_array_test_pairwise[j]
            step = schedule_bounds[0]-self.sample_test_min
            model.set_bayesian_embedding(embedding_list[j])

            while step < schedule_bounds[1]-self.sample_test_min:
                probability_matrix = np.zeros((20, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = self.X_test_pairwise[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagonals set to zero
                        if second_counter == counter:  # same as m = n
                            continue
                        phi_j = self.X_test_pairwise[second_counter]
                        phi_j_numpy = np.asarray(phi_j)

                        feature_input = phi_i_numpy - phi_j_numpy

                        if self.use_gpu:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                        # push through nets
                        preference_prob = model.forward(feature_input)
                        preference_prob = sig(preference_prob)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[
                            0].item()
                        # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)

                # embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

                # top 1
                # given all inputs, and their liklihood of being scheduled, predict the output
                highest_val = max(column_vec)
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
                if len(all_indexes_that_have_highest_val) > 1:
                    print('length of indexes greater than 1: ', all_indexes_that_have_highest_val)
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)
                # choice = np.argmax(probability_vector)

                # top 3
                _, top_three = torch.topk(torch.Tensor(column_vec), 3)

                # Then do training update loop
                truth = self.Y_test_pairwise[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            # embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            prediction_accuracy = [0, 0]
        print(np.mean(prediction_accuracy[0]))
        # self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_PDDT_pairwise_crisp')


def main():
    """
    entry point for file
    :return:
    """
    for i in range(3):
        print('on iteration', i)
        num_schedules = 150
        trainer = ProLoTrain(num_schedules, i)
        # trainer.train()
        test_embeddings = trainer.testing_embeddings
        # trainer.test_again_fuzzy(trainer.model, test_embeddings)
        trainer.test_again_crisp(trainer.model, test_embeddings)


if __name__ == '__main__':
    main()
