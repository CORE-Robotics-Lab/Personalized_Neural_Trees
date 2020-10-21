"""
File to check performance on a heterogeneous dataset. It should be near 100
"""

import torch
import sys
import torch.nn as nn

sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')
from base_testing_environment.prolonet import ProLoNet
import numpy as np
from torch.autograd import Variable
from utils.global_utils import save_pickle
from utils.pairwise_utils import find_which_schedule_this_belongs_to, load_in_pairwise_data

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

    def __init__(self, num_schedules):
        self.num_schedules = num_schedules
        self.num_test_schedules = 100
        self.total_loss_array = []

        self.X_train_pairwise, self.Y_train_pairwise, self.schedule_array_train_pairwise, self.start_of_each_set_twenty_train, self.X_test_pairwise, self.Y_test_pairwise, self.schedule_array_test_pairwise, self.start_of_each_set_twenty_test = load_in_pairwise_data(
            self.num_schedules, self.num_test_schedules)

        use_gpu = True
        att = False
        print('USING attention: ', att)

        self.bayesian_embedding_dim = 3
        self.training_accuracies = []
        self.testing_accuracies = []
        self.model = ProLoNet(input_dim=len(self.X_train_pairwise[0]),
                              weights=None,
                              comparators=None,
                              leaves=32,
                              output_dim=1,
                              bayesian_embedding_dim=self.bayesian_embedding_dim,
                              alpha=1.5,
                              use_gpu=use_gpu,
                              vectorized=True,
                              is_value=True,
                              attention=att)

        if use_gpu:
            self.model = self.model.cuda()
        print(self.model.state_dict())
        self.opt = torch.optim.RMSprop(
            [{'params': list(self.model.parameters())[:-1]}, {'params': self.model.bayesian_embedding.parameters(), 'lr': .01}], lr=.01)

        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.covergence_epsilon = .01
        self.when_to_save = 1500
        self.embedding_list = [torch.ones(self.bayesian_embedding_dim) * 1 / 3 for _ in range(self.num_schedules)]

    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """
        # loss = nn.CrossEntropyLoss()
        sig = torch.nn.Sigmoid()
        training_done = False
        criterion = torch.nn.BCELoss()

        # variables to keep track of loss and number of tasks trained over
        while not training_done:
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
                # positive counterfactuals
                if counter == phi_i_num:  # if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X_train_pairwise[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_i_numpy - phi_j_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())

                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        label = Variable(torch.Tensor(torch.ones((1, 1))))

                    output = self.model.forward(feature_input)
                    output = sig(output)

                    self.opt.zero_grad()
                    loss = criterion(output, label)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()
                    running_loss_predict_tasks += loss.item()
                    num_iterations_predict_task += 1

            # Negative counterfactuals
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X_train_pairwise[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        label = Variable(torch.Tensor(torch.zeros((1, 1))))

                    output = self.model.forward(feature_input)
                    output = sig(output)
                    loss = criterion(output, label)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()

                    running_loss_predict_tasks += loss.item()

                    num_iterations_predict_task += 1

            # add average loss to array
            # print(list(self.model.parameters()))

            self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)

            self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()).clone()  # very ugly

            self.total_iterations += 1

            if self.total_iterations % 500 == 499:
                print('total loss (average for each 40, averaged) at iteration ', self.total_iterations, ' is ', np.mean(self.total_loss_array[-40:]))

            if self.total_iterations % self.when_to_save == self.when_to_save - 1:
                # self.save_trained_nets('PDDT_small_tree' + str(self.total_iterations) + '_' + str(self.bayesian_embedding_dim))
                # train_acc = self.evaluate_on_train_data()
                test_acc = self.evaluate_on_test_data()

                # self.training_accuracies.append(train_acc)
                self.testing_accuracies.append(test_acc)
                print(self.testing_accuracies)


            if self.total_iterations > 10000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.covergence_epsilon:
                training_done = True


    def train_other(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """
        # loss = nn.CrossEntropyLoss()
        sig = torch.nn.Sigmoid()
        training_done = False
        criterion = torch.nn.BCELoss()

        # variables to keep track of loss and number of tasks trained over
        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty_train)
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array_train_pairwise, set_of_twenty)
            schedule_bounds = self.schedule_array_train_pairwise[which_schedule]
            for _ in range(10):
                set_of_twenty = np.random.choice(np.linspace(schedule_bounds[0], schedule_bounds[1] + 1, num=21).tolist()[-1:])
                set_of_twenty = int(set_of_twenty)
                truth = self.Y_train_pairwise[set_of_twenty]
                self.model.set_bayesian_embedding(self.embedding_list[which_schedule].clone())
                # find feature vector of true action taken
                phi_i_num = truth + set_of_twenty
                phi_i = self.X_train_pairwise[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                running_loss_predict_tasks = 0
                num_iterations_predict_task = 0
                # iterate over pairwise comparisons
                for counter in range(set_of_twenty, set_of_twenty + 20):
                    # positive counterfactuals
                    if counter == phi_i_num:  # if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X_train_pairwise[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.ones((1, 1))))

                        output = self.model.forward(feature_input)
                        output = sig(output)

                        self.opt.zero_grad()
                        loss = criterion(output, label)
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.opt.step()
                        running_loss_predict_tasks += loss.item()
                        num_iterations_predict_task += 1

                # Negative counterfactuals
                for counter in range(set_of_twenty, set_of_twenty + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X_train_pairwise[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.zeros((1, 1))))

                        output = self.model.forward(feature_input)
                        output = sig(output)
                        loss = criterion(output, label)
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.opt.step()

                        running_loss_predict_tasks += loss.item()

                        num_iterations_predict_task += 1

                # add average loss to array
                # print(list(self.model.parameters()))

                self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)

                self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()).clone()  # very ugly

                self.total_iterations += 1

                if self.total_iterations % 500 == 499:
                    print('total loss (average for each 40, averaged) at iteration ', self.total_iterations, ' is ', np.mean(self.total_loss_array[-40:]))

                if self.total_iterations % self.when_to_save == self.when_to_save - 1:
                    # self.save_trained_nets('PDDT_small_tree' + str(self.total_iterations) + '_' + str(self.bayesian_embedding_dim))
                    # train_acc = self.evaluate_on_train_data()
                    test_acc = self.evaluate_on_test_data()

                    # self.training_accuracies.append(train_acc)
                    self.testing_accuracies.append(test_acc)


                if self.total_iterations > 10000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                        self.total_loss_array[-500:]) < self.covergence_epsilon:
                    training_done = True

    def evaluate_on_train_data(self):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """

        sig = torch.nn.Sigmoid()
        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        embedding_optimizer = torch.optim.RMSprop([{'params': self.model.bayesian_embedding.parameters()}], lr=.1)
        criterion = torch.nn.BCELoss()

        embedding_list = [torch.ones(self.bayesian_embedding_dim) * 1 / 3 for i in range(self.num_schedules)]

        for j in range(0, self.num_schedules):
            schedule_bounds = self.schedule_array_train_pairwise[j]
            step = schedule_bounds[0]
            self.model.set_bayesian_embedding(embedding_list[j].clone())
            embeddings_throughout_trajectory = []
            while step < schedule_bounds[1]:
                probability_matrix = np.zeros((20, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = self.X_train_pairwise[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagonals set to zero
                        if second_counter == counter:  # same as m = n
                            continue
                        phi_j = self.X_train_pairwise[second_counter]
                        phi_j_numpy = np.asarray(phi_j)

                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                        # push through nets
                        preference_prob = self.model.forward(feature_input)
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
                truth = self.Y_train_pairwise[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # Then do training update loop

                phi_i_num = truth + step
                phi_i = self.X_train_pairwise[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                # iterate over pairwise comparisons
                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X_train_pairwise[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.ones((1, 1))))

                        output = self.model(feature_input)
                        output = sig(output)
                        loss = criterion(output, label)
                        # prepare optimizer, compute gradient, update params

                        embedding_optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        embedding_optimizer.step()
                        print(self.model.state_dict()['bayesian_embedding.embedding'])

                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X_train_pairwise[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.zeros((1, 1))))

                        output = self.model.forward(feature_input)
                        output = sig(output)

                        embedding_optimizer.zero_grad()
                        loss = criterion(output, label)

                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        embedding_optimizer.step()
                        print(self.model.state_dict()['bayesian_embedding.embedding'])

                embeddings_throughout_trajectory.append(torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()).clone())


                # print(model.EmbeddingList.state_dict())
                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        return np.mean(percentage_accuracy_top1)

    def evaluate_on_test_data(self, load_in_model=False):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """

        sig = torch.nn.Sigmoid()

        if load_in_model:
            checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/9222019_PDDT_extended_epochs49999_3.tar')
            self.model.load_state_dict(checkpoint['nn_state_dict'])
        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        embedding_optimizer = torch.optim.RMSprop([{'params': self.model.bayesian_embedding.parameters()}], lr=.1)
        criterion = torch.nn.BCELoss()

        embedding_list = [torch.ones(self.bayesian_embedding_dim) * 1 / 3 for i in range(self.num_test_schedules)]

        for j in range(0, self.num_test_schedules):
            schedule_bounds = self.schedule_array_test_pairwise[j]
            step = schedule_bounds[0]
            self.model.set_bayesian_embedding(embedding_list[j].clone())
            embeddings_throughout_trajectory = []
            while step < schedule_bounds[1]:
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

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                        # push through nets
                        preference_prob = self.model.forward(feature_input)
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
                truth = self.Y_test_pairwise[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # Then do training update loop

                phi_i_num = truth + step
                phi_i = self.X_test_pairwise[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                # iterate over pairwise comparisons
                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X_test_pairwise[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.ones((1, 1))))

                        output = self.model(feature_input)
                        output = sig(output)
                        loss = criterion(output, label)
                        # prepare optimizer, compute gradient, update params

                        embedding_optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        embedding_optimizer.step()
                        # print(self.model.state_dict()['bayesian_embedding.embedding'])

                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X_test_pairwise[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.zeros((1, 1))))

                        output = self.model.forward(feature_input)
                        output = sig(output)

                        embedding_optimizer.zero_grad()
                        loss = criterion(output, label)

                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        embedding_optimizer.step()
                        # print(self.model.state_dict()['bayesian_embedding.embedding'])

                embeddings_throughout_trajectory.append(torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()).clone())
                embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()).clone()  # very ugly


                # print(model.EmbeddingList.state_dict())
                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        # self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'results_PDDT_pairwise_extended_epochs_embedding_dim_3.pkl')

        torch.save({'nn_state_dict': self.model.state_dict(),
                    'training_embeddings': self.embedding_list,
                    'testing_embeddings': embedding_list},
                   '/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/302_model_for_DAgger.tar')


        return np.mean(percentage_accuracy_top1)


    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'training_embeddings': self.embedding_list},
                   '/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/additions_for_HRI/models/9222019_' + name + '.tar')

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


def main():
    """
    entry point for file
    :return:
    """

    num_schedules = 150
    trainer = ProLoTrain(num_schedules)
    trainer.train()
    trainer.evaluate_on_test_data()
    print('Training accuracy', trainer.training_accuracies)
    print('Testing accuracies', trainer.testing_accuracies)



if __name__ == '__main__':
    main()
