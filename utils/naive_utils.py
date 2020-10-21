"""
utils for naive data loading
"""
import pickle

def create_new_dataset(data, num_schedules):
    """
    creates a dataset where each row is the twenty timesteps of a schedule alongside the chosen task
    creates a schedule array that tracks when each schedules starts and ends
    :return:
    """
    X = []
    Y = []
    schedule_array = []
    for i in range(0, num_schedules):
        rand_schedule = i
        timesteps_where_things_scheduled = find_nums_with_task_scheduled_pkl(data, rand_schedule)  # should be of size 20
        if len(timesteps_where_things_scheduled) != 20:
            print('schedule wrong size, WHY?')
            continue

        if i == 0:
            start = 0
        else:
            start = schedule_array[-1][1] + 1  # end of previous list + 1
        end = start + len(timesteps_where_things_scheduled) - 1
        schedule_array.append([start, end])
        for each_timestep in timesteps_where_things_scheduled:
            input_nn, output = rebuild_input_output_from_pickle(data, i, each_timestep)
            X.append(input_nn)
            Y.append(output)
    return X, Y, schedule_array


def rebuild_input_output_from_pickle(data, rand_schedule, rand_timestep):
        schedule_timestep_data = data[rand_schedule][rand_timestep]
        state_input = []
        for i, element in enumerate(schedule_timestep_data):
            if i == 0:
                if type(element) == float:
                    state_input.append(element)
                elif type(element) == int:
                    state_input.append(element)
            elif 131 > i > 4:
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

        output = schedule_timestep_data[131][0]

        return state_input, output


def find_nums_with_task_scheduled_pkl(data, rand_schedule):
        nums = []
        for i, data_at_timestep_i in enumerate(data[rand_schedule]):
            if data[rand_schedule][i][131][0] != -1:
                nums.append(i)
            else:
                continue
        return nums


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
            
            
def load_in_naive_data(num_schedules, num_test_schedules):
    # load in data

    load_directory = '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/scheduling_dataset/' + str(
        num_schedules) + 'EDF_9_25_2019_old_naive.pkl'

    data = pickle.load(open(load_directory, "rb"))
    X_train_naive, Y_train_naive, schedule_array_train_naive = create_new_dataset(num_schedules=num_schedules, data=data)
    for i, each_element in enumerate(X_train_naive):
        X_train_naive[i] = each_element + list(range(20))

    load_directory = '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/scheduling_dataset/' + str(
        num_test_schedules) + 'EDF_9_25_2019_old_test_naive.pkl'

    data = pickle.load(open(load_directory, "rb"))
    X_test_naive, Y_test_naive, schedule_array_test_naive = create_new_dataset(num_schedules=num_test_schedules,
                                                                                              data=data)
    for i, each_element in enumerate(X_test_naive):
        X_test_naive[i] = each_element + list(range(20))

    return X_train_naive, Y_train_naive, schedule_array_train_naive, X_test_naive, Y_test_naive, schedule_array_test_naive


