
import torch
from torch.autograd import Variable
import numpy as np
from utils.global_utils import save_pickle
import pickle

def create_new_data(num_schedules, data):
    """
    creates a dataset from the pkl file
    :param num_schedules:
    :return: X (contains each task feature per timestep)
             Y (contains task scheduled at each timestep)
             schedule_array (index within X where schedules start and end)
    """
    X = []
    Y = []
    schedule_array = []
    for i in range(0, num_schedules):
        timesteps_where_events_are_scheduled = find_nums_with_task_scheduled_pkl(data, i)  # should be 20 sets of 20
        if i == 0:
            start = 0
        else:
            start = schedule_array[-1][1] + 1
        end = start + len(timesteps_where_events_are_scheduled) - 1

        schedule_array.append([start, end])  # each block is of size 400
        for each_timestep in timesteps_where_events_are_scheduled:
            input_nn, output = rebuild_input_output_from_pickle(data, i, each_timestep)
            X.append(input_nn)
            Y.append(output)
    return X, Y, schedule_array


def find_nums_with_task_scheduled_pkl(data, rand_schedule):
    """
    Takes raw data and finds all indexes where a task was scheduled
    :param data:
    :param rand_schedule: the schedule being searched
    :return:
    """
    nums = []
    for i, timestep in enumerate(data[rand_schedule]):
        if data[rand_schedule][i][19] != -1:
            nums.append(i)
        else:
            continue
    return nums


def rebuild_input_output_from_pickle(data, rand_schedule, rand_timestep):
    """
    Take in a schedule and timestep, and append useful information into an array
    :param data:
    :param rand_schedule:
    :param rand_timestep:
    :return: an array of state features alongside the task that was scheduled at the timestep
    """
    schedule_timestep_data = data[rand_schedule][rand_timestep]
    state_input = []
    for i, element in enumerate(schedule_timestep_data):
        # if i == 0:
        #     if type(ast.literal_eval(element)) == float:
        #         state_input.append(ast.literal_eval(element))
        #     elif type(ast.literal_eval(element)) == int:
        #         state_input.append(ast.literal_eval(element))
        if i == 18:
            continue

        elif 19 > i > 4:
            if type(element) == float:
                state_input.append(element)
            elif type(element) == int:
                state_input.append(element)
            elif type(element) == list:
                state_input = state_input + element
            else:
                state_input.append(element)
        else:
            continue

    output = schedule_timestep_data[19]

    return state_input, output


def create_sets_of_20_from_x_for_pairwise_comparisions(X):
    """
    Create sets of 20 to denote each timestep for all schedules
    :return: range(0, length_of_X, 20)
    """
    length_of_X = len(X)
    return list(range(0, length_of_X, 20))

def find_which_schedule_this_belongs_to(schedule_array, sample_val):
    """
    Takes a sample and determines with schedule this belongs to.
    Note: A schedule is task * task sized
    :param sample_val: an int
    :return: schedule num
    """
    for i, each_array in enumerate(schedule_array):
        if each_array[0] <= sample_val <= each_array[1]:
            return i
        else:
            continue

# noinspection PyArgumentList
def transform_into_torch_vars(feature_input, epsilon, positive_example, use_gpu):
    """

    :param feature_input:
    :param P:
    :param epsilon:
    :param positive_example:
    :return:
    """
    if positive_example:
        if use_gpu:
            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
            P = Variable(torch.Tensor([1 - epsilon, epsilon]).cuda())
        else:
            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
            P = Variable(torch.Tensor([1 - epsilon, epsilon]))
    else:
        if torch.cuda.is_available():
            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
            P = Variable(torch.Tensor([epsilon, 1 - epsilon]).cuda())
        else:
            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
            P = Variable(torch.Tensor([epsilon, 1 - epsilon]))

    return feature_input, P


def load_in_pairwise_data(num_schedules, num_test_schedules):
    # load in data

    load_directory = '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/scheduling_dataset/' + str(
        num_schedules) + 'EDF_9_25_2019_old_pairwise.pkl'

    data = pickle.load(open(load_directory, "rb"))
    X_train_pairwise, Y_train_pairwise, schedule_array_train_pairwise = create_new_data(num_schedules, data)
    start_of_each_set_twenty_train = create_sets_of_20_from_x_for_pairwise_comparisions(X_train_pairwise)

    load_directory = '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/scheduling_dataset/' + str(
        num_test_schedules) + 'EDF_9_25_2019_old_test_pairwise.pkl'

    data = pickle.load(open(load_directory, "rb"))
    X_test_pairwise, Y_test_pairwise, schedule_array_test_pairwise = create_new_data(num_test_schedules, data)
    start_of_each_set_twenty_test = create_sets_of_20_from_x_for_pairwise_comparisions(X_test_pairwise)

    return X_train_pairwise, Y_train_pairwise, schedule_array_train_pairwise, start_of_each_set_twenty_train,\
           X_test_pairwise, Y_test_pairwise, schedule_array_test_pairwise, start_of_each_set_twenty_test






