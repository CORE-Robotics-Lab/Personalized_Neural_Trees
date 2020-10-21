"""
file containing helper functions alongside the inner workings of the scheduling environment
"""

import os
import re

import torch
from torch.autograd import Variable

from scheduling_env.create_scheduling_data.agent import *
from scheduling_env.create_scheduling_data.constants import *
from scheduling_env.create_scheduling_data.graph import *
from scheduling_env.create_scheduling_data.task import Task
# from scheduling_env.updated_methodology_in_results.pairwise.NN_unimodal import NNSmall
from base_testing_environment.prolonet import ProLoNet



def randomly_initialize_tasks(m):
    """
    initializes a set of tasks (see Task class for more information)
    :param m: number of tasks to be initialized
    :return:
    """
    tasks = []
    for i in range(0, m):
        tasks.append(Task(name='task' + str(i + 1)))
    return tasks


def euclid_dist(location1, location2):
    """
    returns 2-norm between two locations
    :param location1:
    :param location2:
    :return:
    """
    return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)


def compute_angle_in_rad(location1, location2):
    """
    returns bounded angular difference between two locations
    :param location1:
    :param location2:
    :return:
    """
    return np.arctan2(location1[0] - location2[0], location1[1] - location2[1])


def compute_dist(location1, location2):
    """
    returns weighted diff between two locations
    :param location1:
    :param location2:
    :return:
    """
    alpha0 = 1
    alpha1 = 0
    alpha2 = 0
    norm = euclid_dist(location1, location2)
    angle = compute_angle_in_rad(location1, location2)
    return alpha0 * norm + alpha1 * angle + alpha2 * norm * angle


def find_nearest_unoccupied_task(cur_agent, tasks, agents):
    """
    finds nearest task that is unoccupied.
    This means if another agent is working at that location, that task will not be returned

    :param cur_agent: agent who needs a task to schedule
    :param tasks: list of all tasks
    :param agents: all other agents (this is the Agent class) so there locations are included
    :return:
    """
    current_location = cur_agent.getz()
    closest_task_distance = np.inf
    allowable_distance_to_task = .1
    closest_task = None
    for task in tasks:
        location_occupied = False
        if not task.isTaskScheduled:
            task_loc = task.getloc()
            # check if task is occupied
            for agent in agents:  # check if any agent is at the task
                if cur_agent == agent:  # don't check yourself, cuz that's fine
                    continue
                if compute_dist(agent.getz(), task_loc) < allowable_distance_to_task:
                    location_occupied = True
            if location_occupied:  # Now you know there is an agent too near to that task, thus, look at next task
                continue

            dist = euclid_dist(task_loc, current_location)
            if dist < closest_task_distance:
                closest_task_distance = dist
                closest_task = task
        else:
            continue
    return closest_task


def compute_start_and_finish_times(a, n_t, current_time):
    """
    computes start and finish times of a task given the agent's speed, current location, and task_location
    :param a: Agent
    :param n_t: task_location
    :param current_time:
    :return:
    """
    duration = n_t.getc()
    speed = a.getv()
    current_location = a.getz()
    task_loc = n_t.getloc()
    dist = np.sqrt((task_loc[0] - current_location[0]) ** 2 + (task_loc[1] - current_location[1]) ** 2)
    travel_time = dist / speed
    start_time = current_time + travel_time
    finish_time = start_time + duration
    return start_time, finish_time


def tasks_are_available(tasks):
    """
    Are there tasks that can be scheduled?
    :param tasks: list of tasks
    :return: if there are still available tasks to schedule
    """
    task_not_finished_not_scheduled_count = len(tasks)
    for task in tasks:
        if task.getisTaskFinished():
            continue
        if task.getisTaskScheduled():
            continue
        else:
            task_not_finished_not_scheduled_count -= 1
    if task_not_finished_not_scheduled_count < len(tasks):
        return True
    else:
        return False


# NOTE: ENABLED AND ALIVE ARE SWITCHED


# noinspection PyArgumentList
class World:
    """
    class that contains the scheduling world (there are a ton of parameters and loops)
    """

    def __init__(self, num_scheds, n, model, embedding_list, opt, loss_array, teacher_actions, learner_actions, what_happend_at_every_timestep,
                 timestep_terminated, number_of_decisions_before_terminal_state, num_correct_predictions_total, num_predictions_total, data_so_far):
        """

        :param num_scheds: total number of schedules
        :param n: current schedule number
        :param model: neural network model (params held globally)
        :param embedding_list: embeddings for each schedule (params held globally)
        :param opt: optimizer (params held globally)
        :param loss_array: losses for each updated (averaged over 38) (params held globally)
        :param teacher_actions: the teacher actions (heuristic) per schedule (params held globally)
        :param learner_actions: the learner actions (neural network) per schedule (params held globally)
        :param what_happend_at_every_timestep: a string saying what happens at each timestep per schedule (params held globally)
        :param timestep_terminated: the timestep the schedule was ended (params held globally)
        :param number_of_decisions_before_terminal_state: number of tasks actually scheduled
        :param num_correct_predictions_total: number of times the learner matched the expert
        :param num_predictions_total: number of times learner had the chance to match the expert
        :param data_so_far: aggregated dataset
        """
        self.global_schedule_num = n
        self.num_tasks = 20
        self.num_scheds = num_scheds
        self.size_x = grid_size_x
        self.size_y = grid_size_y
        self.w_EDR, self.w_RESOURCE, self.w_DISTANCE = self.get_random_coeffs_for_aggregate_score()
        self.tasks = randomly_initialize_tasks(self.num_tasks)  # will initialize tasks 1 to 20
        self.num_agents = 2
        self.data_done_generating = False
        self.DEBUG = False
        self.use_gpu = False

        self.pairwise = False
        # Init Features
        self.agent_locations = np.zeros((1, self.size_x * self.size_y))  # 1 x 16 vector. 1 is agent is present, else 0
        self.is_task_alive = np.ones((1, self.num_tasks))  # 1 if alive
        self.is_task_enabled = np.ones((1, self.num_tasks))  # 1 if enabled
        self.is_task_finished = np.zeros((1, self.num_tasks))  # 1 if finished
        self.task_deadlines = np.ones((1, self.num_tasks)) * 1000  # each value is deadline of corresponding task
        self.orientation = np.zeros((2, self.num_tasks))  # 2 for 2 agents
        self.how_many_tasks_in_each_square = np.zeros((1, self.size_x * self.size_y))  # add up number of tasks in each location
        self.is_agent_idle = np.ones((2, 1))
        self.travel_time_constraint_satisfied = np.ones((2, self.num_tasks))  # 1 if satisfied
        self.agent_distances = np.ones((2, self.num_tasks))
        self.is_task_in_progress = np.zeros((1, self.num_tasks))  # 1 means task in progress

        self.last_timestep_data = []
        # ALIVE, EXECUTED AND ENABLED MUST BE ON FOR TASK TO BE EXECUTED

        # Extensions of features
        self.finish_time_per_task_dict = {}
        self.task_agent_dict = {}
        self.agent_current_task = [-1, -1]
        self.task_locations = np.zeros((1, self.num_tasks))  # store where each task is

        self.task_vertex_numbers_for_end = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
        self.task_vertex_numbers_for_start = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
        self.filepath = "/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/create_scheduling_data/1_schedule.csv"
        self.writepath = "/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/create_scheduling_data/" + str(
            self.num_scheds) + "tot_naive_schedule.csv"
        self.writepath2 = "/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/create_scheduling_data/" + str(
            self.num_scheds) + "tot_pairwise_schedule.csv"
        self.second_file_path = "/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/create_scheduling_data/11_schedule.csv"
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        else:
            print('File does not exist')

        if os.path.exists(self.second_file_path):
            os.remove(self.second_file_path)
        else:
            print('File does not exist')
        self.t = 0
        self.did_schedule_fail = False
        self.do_you_like_big_tasks = self.get_whether_you_like_bigger_or_smaller_task_num()

        self.condition_dict = {0: 'agent was busy', 1: 'task scheduled could not be scheduled', 2: 'deadline passed', 3: 'no tasks available',
                               4: 'neural net task successfully scheduled'}

        # initialize 2 agents
        self.agent1 = Agent(1, (random.randint(0, grid_size_x - 1), random.randint(0, grid_size_y - 1)), name='agent1')
        self.agent2 = Agent(2, (random.randint(0, grid_size_x - 1), random.randint(0, grid_size_y - 1)), name='agent2')
        self.agents = [self.agent1, self.agent2]

        self.task_to_schedule = None
        self.alpha, self.alpha2 = self.init_hyperparameters()
        self.initialize_graph()
        while not self.graph.is_feasible():
            self.initialize_graph()
        self.initialize_finish_times_to_inf()  # at start, no tasks have finished

        self.initialization_step()
        self.naive_total_data = []
        self.pairwise_total_data = []

        # neural network stuff
        device = torch.device("cpu")
        self.model = model
        self.embedding_list = embedding_list
        self.network_state = []

        if self.DEBUG:
            print(self.model.state_dict())
        # set up main optimizer to optimize model parameters and embedding parameters with different learning rates
        # self.embedding_optimizer = torch.optim.SGD(self.model.EmbeddingList.parameters(), lr=.01)
        self.embedding_optimizer = torch.optim.RMSprop([{'params': self.model.bayesian_embedding.parameters()}], lr=.1)

        self.opt = opt
        self.loss_array = loss_array

        # other stuff we want to save
        self.teacher_actions = teacher_actions
        self.learner_actions = learner_actions
        self.what_happend_at_every_timestep = what_happend_at_every_timestep
        self.timestep_terminated = timestep_terminated
        self.number_of_decisions_before_terminal_state = number_of_decisions_before_terminal_state
        self.num_correct_predictions_total = num_correct_predictions_total
        self.num_predictions_total = num_predictions_total

        self.data_so_far = data_so_far
        if self.global_schedule_num > 0:
            self.update_network()
            # its fine if we don't reset the embedding list


    @staticmethod
    def init_hyperparameters():
        """
        hyperparameters that are used for computing distance in heuristic 3
        :return:
        """
        alpha = .8
        alpha2 = 1

        return alpha, alpha2

    def initialization_step(self):
        """
        Runs at the start
        :return: updates parameters (a little redundant, since this is done again at the start
        of the first iteration
        """
        # Update where agents are
        self.update_agent_location_vector()
        # update task locations
        self.update_task_location_vector()
        # update deadlines
        self.populate_deadline_vector()
        # update distances to each task and orientation to each task
        self.update_agent_distances_vector()
        self.update_agent_orientation_vector()

    def update_task_location_vector(self):
        """
        counts how many tasks are each of the 16 locations
        also stores which location each task is in, in another array
        :return:
        """
        for counter, task in enumerate(self.tasks):
            location = task.getloc()
            if location[0] == 0:
                vectorized_task_loc = location[1]
            elif location[0] == 1:
                vectorized_task_loc = 4 + location[1]
            elif location[0] == 2:
                vectorized_task_loc = 8 + location[1]
            else:  # location[0] == 3
                vectorized_task_loc = 12 + location[1]
            self.how_many_tasks_in_each_square[0][vectorized_task_loc] += 1
            self.task_locations[0][counter] = vectorized_task_loc
            # print(location)
        # print(self.how_many_tasks_in_each_square)

    def populate_deadline_vector(self):
        """
        Stores the current deadline for each task
        """

        count = 0
        for i in range(0, len(self.graph.vertices)):
            if i == 41:
                continue
            if (i - 1) % 2 == 0:
                self.task_deadlines[0][count] = self.graph.M[0][i]
                count += 1
        if self.DEBUG:
            print('implicit task deadlines from M is ', self.task_deadlines)

    def update_agent_location_vector(self):
        """
        This adds the agent location into vectorized format of the grid.
        Only updates if the agent is busy.
        :return: Nothing
        """

        for agent in self.agents:
            location = agent.getz()
            # print(location)
            if location[0] == 0:
                vectorized_agent_loc = location[1]
            elif location[0] == 1:
                vectorized_agent_loc = 4 + location[1]
            elif location[0] == 2:
                vectorized_agent_loc = 8 + location[1]
            else:  # location[0] == 3
                vectorized_agent_loc = 12 + location[1]

            if agent.isBusy == False:
                # remove any location if it shows it as well
                self.agent_locations[0][vectorized_agent_loc] = 0
                continue
            else:
                self.agent_locations[0][vectorized_agent_loc] = 1
        if self.DEBUG:
            print('agent location vector is ', self.agent_locations)

    @staticmethod
    def get_vectorized_location(location):
        # Helper function

        if location[0] == 0:
            vectorized_agent_loc = location[1]
        elif location[0] == 1:
            vectorized_agent_loc = 4 + location[1]
        elif location[0] == 2:
            vectorized_agent_loc = 8 + location[1]
        else:  # location[0] == 3
            vectorized_agent_loc = 12 + location[1]
        return vectorized_agent_loc

    def initialize_graph(self):
        self.graph = Graph()
        self.graph.add_vertex('start')  # initialize start and end node
        self.graph.add_vertex('end')
        self.graph.add_edge_by_name('start', 'end', 150)  # max deadline
        self.graph.add_tasks_vertex_and_edges(self.tasks)  # adds edges between start and end. Note 2-3 will be s_1 f_1
        self.graph.initialize_all_start_and_end_nodes(self.tasks)  # adds 0 edges between 2 nodes.
        self.wait_arr = self.graph.get_random_wait_constraints(self.tasks)
        self.deadline_dict = self.graph.get_random_deadline_constraints(self.tasks)
        self.graph.build_M_matrix()
        self.graph.compute_floyd_warshal()
        self.inital_M = self.graph.M

        self.graph.print_checking()
        print('graph is feasible: ', self.graph.is_feasible())
        # When two tasks have an edge, the one that is latest is currently used

    @staticmethod
    def get_random_coeffs_for_aggregate_score():
        """
        :return: omega values for scheduling preference
        The ordering is earliest deadline, most resource locations, distance
        """
        # r = [2 * random.uniform(0, 1) - 1 for i in range(1, 4)]
        # s = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        # r = [i / s for i in r]
        # print(r)
        r = [0, 0, 0]
        # r[0] = np.random.choice([-1, 0, 1])
        # r[1] = np.random.choice([-1, 0, 1])
        # r[2] = np.random.choice([-1, 0, 1])

        r[0] = np.random.uniform(low=0, high=1)
        r[1] = 0
        r[2] = np.random.uniform(low=0, high=1)

        # print('All heuristics are active')
        # print('EDF, Resource Allocation are active')
        print('EDF, distance are active')
        # h_ind = np.random.randint(0, 3)
        # print(c1)
        # print(h_ind)
        # h_ind = 0
        # if c1 < .5:
        #     r[h_ind] = 1
        # else:
        #     r[h_ind] = -1

        return r[0], r[1], r[2]

    def get_whether_you_like_bigger_or_smaller_task_num(self):
        """
        like bigger -> 1
        like smaller -> 0
        :return:
        """
        # TODO: change this to save in a global list
        what_are_you = np.random.choice([0, 1])

        file = open('what_are_you_test.txt', 'a')
        if what_are_you == 1:
            file.write(str(self.global_schedule_num) + ':higher' + '\n')
            if self.DEBUG:
                print('higher tasks preferred')
        else:
            file.write(str(self.global_schedule_num) + ':lower' + '\n')
            if self.DEBUG:
                print('lower tasks preferred')
        file.close()

        return what_are_you

    def update_agent_distances_vector(self):
        """
        updates a vector of euclidean distances to each task. If location of agent moves, this should change.
        :return: Nothing
        """
        count = 0
        for agent in self.agents:
            agent_loc = agent.getz()

            for i, each_task in enumerate(self.tasks):
                dist = euclid_dist(agent_loc, each_task.getloc())
                self.agent_distances[count][i] = dist
            count += 1
        if self.DEBUG:
            print(self.agent_distances)

    def update_agent_orientation_vector(self, DEBUG=False):
        """
        computes the angle to each task based on current locations
        :param DEBUG:
        :return:
        """
        count = 0
        for agent in self.agents:
            agent_dir = agent.getOrientation()
            agent_loc = agent.getz()
            for i, each_task in enumerate(self.tasks):
                angle_to_move_in = compute_angle_in_rad(agent_loc, each_task.getloc())
                angle_you_must_turn = angle_to_move_in - agent_dir
                angle_you_must_turn_bounded = np.arctan2(np.sin(angle_you_must_turn), np.cos(angle_you_must_turn))
                self.orientation[count][i] = angle_you_must_turn_bounded
            count += 1
        if DEBUG:
            print('orientation to all tasks is ', self.orientation)

    def update_alive_enabled_travel(self):
        """
        Updates tasks that are alive, enabled, and travel_time_enabled
        Again, has some redundancies
        """
        self.is_task_alive = np.ones((1, self.num_tasks))  # 1 if alive
        self.is_task_enabled = np.ones((1, self.num_tasks))  # 1 if enabled
        self.travel_time_constraint_satisfied = np.ones((2, self.num_tasks))  # 1 if satisfied

        # ALIVE
        for each_task, i in enumerate(self.task_vertex_numbers_for_start):
            # make sure element of first column is less than zero
            if self.graph.M[i][0] <= 0:
                name_of_task_being_checked = self.graph.names_of_vertex[i]
                # find all tasks associated with each node
                # for every element task points to
                for element in self.graph.vertices[i].points_to:
                    num = self.graph.gamma[element]
                    name_of_ele = self.graph.names_of_vertex[num]

                    if name_of_ele == 'start':
                        continue
                    elif name_of_ele == 'end':
                        continue
                    elif num == i + 1:  # is it the end of the task
                        continue
                    else:
                        # task has been found
                        c = int((re.findall('\d+', name_of_ele))[0])
                        if self.is_task_finished[0][c - 1] == 0:
                            self.is_task_alive[0][each_task] = 0
        if self.DEBUG:
            print('tasks that are alive', self.is_task_alive)

        # ENABLED
        for each_task, i in enumerate(self.task_vertex_numbers_for_start):
            # make sure element of first column is less than zero, # TODO: figure out why
            if self.graph.M[i][0] <= 0:
                name_of_task_being_checked = self.graph.names_of_vertex[i]
                # find all tasks associated with each node
                # for every element task points to
                for element in self.graph.vertices[i].points_to:
                    num = self.graph.gamma[element]  # num of vertex as in M matrix
                    name_of_ele = self.graph.names_of_vertex[num]
                    weight = self.graph.vertices[i].points_to[element]
                    if name_of_ele == 'start':
                        continue
                    elif name_of_ele == 'end':
                        continue
                    elif num == i + 1:  # is it the end of the task
                        continue
                    elif self.is_task_alive[0][each_task] == 0:
                        # if task is not alive, it cannot be enabled
                        self.is_task_enabled[0][each_task] = 0
                    else:
                        # task that is alive has been been found
                        if self.t < self.finish_time_per_task_dict[((num - 1) / 2) - 1] + np.abs(weight):
                            self.is_task_enabled[0][each_task] = 0
        if self.DEBUG:
            print('tasks that are enabled', self.is_task_enabled)

        # Travel Time Enabled
        for agent_num, agent in enumerate(self.agents):
            for each_task, i in enumerate(self.task_vertex_numbers_for_start):
                # make sure element of first column is less than zero
                if self.graph.M[i][0] <= 0:
                    name_of_task_being_checked = self.graph.names_of_vertex[i]
                    # find all tasks associated with each node
                    # for every element task points to
                    for element in self.graph.vertices[i].points_to:
                        num = self.graph.gamma[element]  # num of vertex as in M matrix
                        name_of_ele = self.graph.names_of_vertex[num]
                        weight = self.graph.vertices[i].points_to[element]
                        task_number = int((num - 1) / 2)
                        if len(self.graph.vertices[i].points_to) == 2:
                            if self.t < self.agents[agent_num].curr_finish_time + self.agent_distances[agent_num][each_task] / self.agents[
                                agent_num].getv():
                                self.travel_time_constraint_satisfied[agent_num][each_task] = 0
                                continue
                        if name_of_ele == 'start':
                            continue
                        elif name_of_ele == 'end':
                            continue
                        elif num == i + 1:  # is it the end of the task
                            continue
                        else:  # more than 2 constraints

                            if self.t < self.finish_time_per_task_dict[task_number - 1] + \
                                    self.agent_distances[agent_num][each_task] / self.agents[agent_num].getv():
                                self.travel_time_constraint_satisfied[agent_num][each_task] = 0
                            if self.t < self.agents[agent_num].curr_finish_time + self.agent_distances[agent_num][each_task] / self.agents[
                                agent_num].getv():
                                self.travel_time_constraint_satisfied[agent_num][each_task] = 0

        if self.DEBUG:
            print('tasks that are travel_constraint satisfied', self.travel_time_constraint_satisfied)

    def schedule_task(self, counter):
        """
        Schedules a task based on aggregate score
        updates agent current task based on this
        :param counter: this is the agent number
        :return: tasks
        """
        task_to_schedule = []
        each_agent = self.agents[counter]
        task_found = False
        H1_score_list = []
        H2_score_list = []
        H3_score_list = []
        H1_dict = {}
        H2_dict = {}
        H3_dict = {}

        # Agent not idle case, exit immediately
        if self.is_agent_idle[counter][0] == 0:
            print(each_agent.getName(), 'is not Idle')
            print(each_agent.getName(), 'is scheduled for null task')
            task_to_schedule.append(-1)
            self.task_to_schedule = task_to_schedule
            return task_to_schedule
            # if agent is busy,  output null task

        for task_num, each_task in enumerate(self.tasks):
            if self.is_task_finished[0][task_num] == 1:  # Can't schedule a task that has already been completed
                continue
            if self.is_task_alive[0][task_num] == 0:
                continue
            # if self.is_task_enabled[0][task_num] == 0:
            #     continue
            # if self.travel_time_constraint_satisfied[0][task_num] == 0:
            #     continue
            if self.is_task_in_progress[0][task_num] == 1:  # can't schedule the same task twice
                continue

            # All constraints satisfied
            # normalize each score separately
            deadline_score = (self.heuristic(heuristic_num=1, task_num=task_num, agent_num=counter))
            occupacity_score = (self.heuristic(heuristic_num=2, task_num=task_num, agent_num=counter))
            distance_score = (self.heuristic(heuristic_num=3, task_num=task_num, agent_num=counter)) * 150 / np.sqrt(
                32)  # The 150/np.sqrt(32) puts it in the same range as deadline score

            H1_dict[task_num] = deadline_score
            H1_score_list.append(deadline_score)

            H2_dict[task_num] = occupacity_score
            H2_score_list.append(occupacity_score)

            H3_dict[task_num] = distance_score
            H3_score_list.append(distance_score)

            task_found = True

        if not task_found:
            task_to_schedule.append(-1)
            self.task_to_schedule = task_to_schedule
            self.what_happend_at_every_timestep[self.global_schedule_num].append(self.condition_dict[3])
            return task_to_schedule

        new_dict = {}
        for key in H1_dict:
            new_dict[key] = H1_dict[key] * self.w_EDR + H2_dict[key] * self.w_RESOURCE + H3_dict[key] * self.w_DISTANCE

        highest = max(new_dict.values())  # figure out the highest score
        tasks_with_best_scores = [k for k, v in new_dict.items() if v == highest]  # find all keys associated with the highest value
        if len(tasks_with_best_scores) > 1:
            print(tasks_with_best_scores)

        if self.do_you_like_big_tasks:
            task_chosen = max(tasks_with_best_scores)
        else:
            task_chosen = min(tasks_with_best_scores)

        print('Expert: Task chosen for', each_agent.getName(), ' is ', task_chosen, ' enabled: ', self.is_task_enabled[0][task_chosen])
        self.teacher_actions[self.global_schedule_num].append(task_chosen)
        # neural net task
        self.converge_embedding_based_on_history(task_chosen, counter)
        neural_net_task = self.predict_task(task_chosen, counter)
        self.learner_actions[self.global_schedule_num].append(neural_net_task)

        print('Neural Net: Task chosen for', each_agent.getName(), ' is ', neural_net_task, ' enabled: ',
              self.is_task_enabled[0][neural_net_task])

        # all of this changed to represent neural net task

        # if self.is_task_enabled[0][neural_net_task] == 0:
        #     print('Task was not enabled, but is alive')

        if neural_net_task == task_chosen:
            self.num_correct_predictions_total[self.global_schedule_num] += 1

        self.num_predictions_total[self.global_schedule_num] += 1

        # Only do it if all of the pre-conditions are met

        if self.global_schedule_num != 0:
            location_of_task = self.tasks[neural_net_task].getloc()
            vectorized_task_num = self.get_vectorized_location(location_of_task)  # checks if current task is in a location that is occupied
            if self.is_task_alive[0][neural_net_task] == 0 or \
                    self.is_task_enabled[0][neural_net_task] == 0 or \
                    self.travel_time_constraint_satisfied[counter][neural_net_task] == 0 or \
                    self.agent_locations[0][vectorized_task_num] >= 1 or \
                    self.is_task_in_progress[0][neural_net_task]:
                task_to_schedule.append(-1)
                self.task_to_schedule = task_to_schedule
                print('Task ', neural_net_task, ' did not meet criteria of being enabled, alive, travel satisfied, or not occupied')
                self.what_happend_at_every_timestep[self.global_schedule_num].append(self.condition_dict[1])
                return task_to_schedule

        else:  # global task schedule num is zero
            if self.is_task_enabled[0][task_chosen] == 0:
                print('Task was not enabled, but is alive')
            # Only do it if all of the pre-conditions are met
            location_of_task = self.tasks[task_chosen].getloc()
            vectorized_task_num = self.get_vectorized_location(location_of_task)  # checks if current task is in a location that is occupied
            if self.is_task_alive[0][task_chosen] == 0 or \
                    self.is_task_enabled[0][task_chosen] == 0 or \
                    self.travel_time_constraint_satisfied[counter][task_chosen] == 0 or \
                    self.agent_locations[0][vectorized_task_num] >= 1:
                task_to_schedule.append(-1)
                self.task_to_schedule = task_to_schedule
                print('task ', task_chosen, ' did not meet criteria of being enabled, alive, travel satisfied, or not occupied')
                self.what_happend_at_every_timestep[self.global_schedule_num].append(self.condition_dict[1])
                return task_to_schedule

        if self.global_schedule_num != 0:
            # if self.t > self.task_deadlines[0][neural_net_task]:
            if self.has_any_deadlines_passed(neural_net_task):
                task_to_schedule.append(-1)
                print('Deadline is passed')
                # updated where this is changed
                self.did_schedule_fail = True
                self.what_happend_at_every_timestep[self.global_schedule_num].append(self.condition_dict[2])
                return task_to_schedule
        else:
            # if self.t > self.task_deadlines[0][neural_net_task]:
            if self.has_any_deadlines_passed(task_chosen):
                task_to_schedule.append(-1)
                print('Deadline is passed')
                # updated where this is changed
                self.did_schedule_fail = True
                self.what_happend_at_every_timestep[self.global_schedule_num].append(self.condition_dict[2])
                return task_to_schedule

        if self.global_schedule_num != 0:
            task_to_schedule.append(neural_net_task)
            self.agent_current_task[counter] = task_to_schedule[0]  # changes agent current task
            self.task_to_schedule = task_to_schedule
            self.what_happend_at_every_timestep[self.global_schedule_num].append(self.condition_dict[4])
            self.number_of_decisions_before_terminal_state[self.global_schedule_num] += 1
            # maybe remove the return
            print('Task scheduled for', each_agent.getName(), 'at time ', self.t, 'is ', self.task_to_schedule)
            return task_to_schedule  # ALL zero indexed
        else:
            task_to_schedule.append(task_chosen)
            self.agent_current_task[counter] = task_to_schedule[0]  # changes agent current task
            self.task_to_schedule = task_to_schedule
            self.what_happend_at_every_timestep[self.global_schedule_num].append(self.condition_dict[4])
            self.number_of_decisions_before_terminal_state[self.global_schedule_num] += 1
            # maybe remove the return
            print('Task scheduled for', each_agent.getName(), 'at time ', self.t, 'is ', self.task_to_schedule)
            return task_to_schedule  # ALL zero indexed

    def set_vector_value(self, which_vector, task_num, value):
        """
        another helper function to assist in accessing numpy vars
        :param which_vector:
        :param task_num:
        :param value:
        :return:
        """
        # unused helper function
        if which_vector == 'alive':
            self.is_task_alive[0][task_num] = value
        if which_vector == 'enabled_temporal':
            self.is_task_enabled[0][task_num] = value
        if which_vector == 'finished':
            self.is_task_finished[0][task_num] = value
        if which_vector == 'enabled_travel':
            self.travel_time_constraint_satisfied[0][task_num] = value

    def initialize_finish_times_to_inf(self):
        """
        sets finish times to inf. This basically means they are not completed
        :return:
        """
        for i in range(0, self.num_tasks):
            self.finish_time_per_task_dict[i] = np.inf

    def print_all_features(self):
        """
        "Nice" print feature to print out all relevant details at each timestep
        :return:
        """
        if self.DEBUG:
            print('weights')
            print('-------------------------')
            print('w_EDR: ', self.w_EDR)
            print('w_Resource', self.w_RESOURCE)
            print('w_Distance', self.w_DISTANCE)
            print(' ')
            print('Features')
            print('-------------------------')
            print('Agent locations at time step:', self.t, ' are ', self.agent_locations)
            print('Agents that          are idle at time step:', self.t, ' are ', self.is_agent_idle)
            print('Tasks that are          alive at time step:', self.t, ' are ', self.is_task_alive)
            print('Tasks that are        enabled at time step:', self.t, ' are ', self.is_task_enabled)
            print('Tasks that are travel_enabled at time step:', self.t, ' are ', self.travel_time_constraint_satisfied)
            print('Tasks that are    in progress at time step:', self.t, ' are ', self.is_task_in_progress)
            print('Tasks that are       finished at time step:', self.t, ' are ', self.is_task_finished)

            print("agent1 is currently at location ", self.get_vectorized_location(self.agents[0].getz()), ' and is working on ',
                  self.agents[0].curr_task)
            print("agent2 is currently at location ", self.get_vectorized_location(self.agents[1].getz()), ' and is working on ',
                  self.agents[1].curr_task)

    def heuristic(self, heuristic_num, task_num, agent_num):
        """
        computes appeal of a task
        :param heuristic_num: which heuristic are you analyzing
        :param task_num: which task
        :param agent_num: which agent
        :return:
        """
        if heuristic_num == 1:  # earliest deadline first
            deadline = self.task_deadlines[0][task_num]
            if self.DEBUG:
                print('Deadline for task ', task_num, ' is ', deadline)
            return -deadline

        if heuristic_num == 2:  # rule to mitigate resource contention
            # check task_num location
            task_loc = self.tasks[task_num].getloc()
            if task_loc[0] == 0:
                vectorized_task_loc = task_loc[1]
            elif task_loc[0] == 1:
                vectorized_task_loc = 4 + task_loc[1]
            elif task_loc[0] == 2:
                vectorized_task_loc = 8 + task_loc[1]
            else:  # location[0] == 3
                vectorized_task_loc = 12 + task_loc[1]
            return self.how_many_tasks_in_each_square[0][vectorized_task_loc]

        if heuristic_num == 3:
            combo = self.agent_distances[agent_num][task_num] + self.alpha * np.abs(self.orientation[agent_num][task_num]) + \
                    self.alpha2 * self.agent_distances[agent_num][task_num] * np.abs(self.orientation[agent_num][task_num])
            if self.DEBUG:
                print('Combo score is ', combo)
            # return a negative because you want to travel the least distance
            return -combo

    # TODO: add method to update parameters
    # TODO: clean up so only thing left is enabled,,

    def update_floyd_warshall_and_all_vectors(self):
        """
        Computes Floyd Warshalls
        Updates agent locations (if they have reached a task move there)
        Updates implicit deadlines
        Updates agent distances based on updated agent locations
        Updates which tasks are alive, enabled and travel_constraint enabled
        :return:  if schedule is feasible
        """
        self.graph.compute_floyd_warshal()
        # Update where agents are
        self.update_agent_location_vector()
        # update deadlines
        self.populate_deadline_vector()
        # update distances to each task and orientation to each task
        self.update_agent_distances_vector()
        self.update_agent_orientation_vector()

        self.update_alive_enabled_travel()

        return self.graph.is_feasible()

    def compute_task_to_schedule(self, agent_num):
        """
        Gets the task that will be scheduled.
        If agent is busy, the output will be a null task, but what the agent is currently working on
        will not change.
        :return:
        """
        task = self.schedule_task(agent_num)  # get chosen task

        agent = self.agents[agent_num]  # get current agent
        # self.write_csv_pairwise(agent_num)
        # self.write_csv(agent_num)

        if task[0] == -1:  # if null task chosen
            task_currently_working_on = self.agent_current_task[agent_num]
            if task_currently_working_on != -1 and self.is_task_finished[0][task_currently_working_on] == 0:  # task is currently in progress
                pass
            else:  # task is finished, but there is no task to schedule
                self.agent_current_task[agent_num] = -1
        else:  # tasks returned contain actual tasks

            self.agent_current_task[agent_num] = task[0]  # update current agent task
            agent.changebusy(True)  # mark agent as busy
            self.update_agent_is_idle_based_on_class()

        self.update_agent_pose_and_finish_time_and_log_event(agent_num)

    def get_features_from_timestep_data(self, counter):
        data = []
        last_timestep_state = self.network_state[-20:]
        n = last_timestep_state[counter]
        for i, element in enumerate(n):
            if i == 18:
                continue
            elif 4 < i < 19:
                data.append(element)

        return data

    def get_features_from_timestep_data_from_agg(self, timestep_data_from_agg, counter):
        data = []
        n = timestep_data_from_agg[counter]
        for i, element in enumerate(n):
            if i == 18:
                continue
            elif 4 < i < 19:
                data.append(element)

        return data

    def converge_embedding_based_on_history(self, task, agent_num):
        sig = torch.nn.Sigmoid()
        criterion = torch.nn.BCELoss()
        self.model.set_bayesian_embedding(self.embedding_list[self.global_schedule_num].clone())
        probability_matrix = np.zeros((20, 20))
        self.update_current_state(agent_num)

        # update loop
        loss_counter = 0
        phi_i_num = task
        phi_i = self.get_features_from_timestep_data(phi_i_num)
        phi_i_numpy = np.asarray(phi_i)
        # iterate over pairwise comparisons
        for counter in range(0, 0 + 20):
            if counter == phi_i_num:
                continue
            else:
                phi_j = self.get_features_from_timestep_data(counter)
                phi_j_numpy = np.asarray(phi_j)
                feature_input = phi_i_numpy - phi_j_numpy

                if self.use_gpu:
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                    label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                else:
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                    label = Variable(torch.Tensor(torch.ones((1, 1))))

                output = sig(self.model(feature_input))
                loss = criterion(output, label)
                # prepare optimizer, compute gradient, update params
                loss_counter += loss.item()
                self.embedding_optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.embedding_optimizer.step()

        for counter in range(0, 0 + 20):
            if counter == phi_i_num:
                continue
            else:
                phi_j = self.get_features_from_timestep_data(counter)
                phi_j_numpy = np.asarray(phi_j)
                feature_input = phi_j_numpy - phi_i_numpy

                if self.use_gpu:
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                    label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                else:
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                    label = Variable(torch.Tensor(torch.zeros((1, 1))))

                output = sig(self.model.forward(feature_input))

                self.embedding_optimizer.zero_grad()
                loss = criterion(output, label)
                loss_counter += loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.embedding_optimizer.step()
        self.loss_array.append(loss_counter / 38)
        self.embedding_list[self.global_schedule_num] = torch.Tensor(
            self.model.get_bayesian_embedding().detach().cpu().numpy()).clone()  # very ugly



    def predict_task(self, task, agent_num):
        """

        :param task: this is the true label
        :param agent_num:
        :return:
        """
        sig = torch.nn.Sigmoid()
        criterion = torch.nn.BCELoss()

        self.model.set_bayesian_embedding(self.embedding_list[self.global_schedule_num].clone())
        probability_matrix = np.zeros((20, 20))
        self.update_current_state(agent_num)
        for m, counter in enumerate(range(0, 0 + 20)):

            phi_i = self.get_features_from_timestep_data(counter)
            phi_i_numpy = np.asarray(phi_i)

            # for each set of twenty
            for n, second_counter in enumerate(range(0, 0 + 20)):
                # fill entire array with diagonals set to zero
                if second_counter == counter:  # same as m = n
                    continue
                phi_j = self.get_features_from_timestep_data(second_counter)
                phi_j_numpy = np.asarray(phi_j)

                feature_input = phi_i_numpy - phi_j_numpy

                if self.use_gpu:
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())

                else:
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                # push through nets to get preferences
                preference_prob = sig(self.model.forward(feature_input))
                probability_matrix[m][n] = preference_prob[0].data.detach()[
                    0].item()
                # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

        # Set of twenty is completed
        column_vec = np.sum(probability_matrix, axis=1)

        # top 1
        highest_val = max(column_vec)
        all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
        if len(all_indexes_that_have_highest_val) > 1:
            print('Length of indexes greater than 1: ', all_indexes_that_have_highest_val)
        # top 1
        choice = np.random.choice(all_indexes_that_have_highest_val)

        # # update loop
        # loss_counter = 0
        # phi_i_num = task
        # phi_i = self.get_features_from_timestep_data(phi_i_num)
        # phi_i_numpy = np.asarray(phi_i)
        # # iterate over pairwise comparisons
        # for counter in range(0, 0 + 20):
        #     if counter == phi_i_num:
        #         continue
        #     else:
        #         phi_j = self.get_features_from_timestep_data(counter)
        #         phi_j_numpy = np.asarray(phi_j)
        #         feature_input = phi_i_numpy - phi_j_numpy
        #
        #         if self.use_gpu:
        #             feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
        #             label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
        #         else:
        #             feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
        #             label = Variable(torch.Tensor(torch.ones((1, 1))))
        #
        #         output = sig(self.model(feature_input))
        #         loss = criterion(output, label)
        #         # prepare optimizer, compute gradient, update params
        #         loss_counter += loss.item()
        #         self.embedding_optimizer.zero_grad()
        #         loss.backward()
        #         # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        #         self.embedding_optimizer.step()
        #
        # for counter in range(0, 0 + 20):
        #     if counter == phi_i_num:
        #         continue
        #     else:
        #         phi_j = self.get_features_from_timestep_data(counter)
        #         phi_j_numpy = np.asarray(phi_j)
        #         feature_input = phi_j_numpy - phi_i_numpy
        #
        #         if self.use_gpu:
        #             feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
        #             label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
        #         else:
        #             feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
        #             label = Variable(torch.Tensor(torch.zeros((1, 1))))
        #
        #         output = sig(self.model.forward(feature_input))
        #
        #         self.embedding_optimizer.zero_grad()
        #         loss = criterion(output, label)
        #         loss_counter += loss.item()
        #
        #         loss.backward()
        #         # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        #         self.embedding_optimizer.step()
        # self.loss_array.append(loss_counter / 38)
        # self.embedding_list[self.global_schedule_num] = torch.Tensor(
        #     self.model.get_bayesian_embedding().detach().cpu().numpy()).clone()  # very ugly

        return choice

    def add_constraints_based_on_task(self):
        """
        adds constraints for start of element to start node
        This is what preserves order
        :return:
        """
        # Note this method is only called when a task is found
        for counter, agent in enumerate(self.agents):
            if len(agent.task_list) > 0:  # task has been chosen
                last_element = agent.task_list[-1]
                self.graph.add_movement_constraint_by_name(self.tasks[last_element].getName(), weight=self.t)

    def update_agent_is_idle_based_on_class(self):
        """
        update world traits based on agent traits
        does seem weird to have two copies though
        :return:
        """
        # method to update class based on external params
        for counter, agent in enumerate(self.agents):
            isAgentIdle = not agent.isBusy
            self.is_agent_idle[counter][0] = isAgentIdle

    @staticmethod
    def set_param(param, num, set_val):
        """
        can help you to set some numpy matrices
        :param param:
        :param num:
        :param set_val:
        :return:
        """
        param[0][num] = set_val

    def write_csv(self, agent_num):
        """
        adds data to the csv for a certain agent
        :param agent_num: agent that task was scheduled for
        :return:
        """
        data = []
        data.append(self.t)
        data.append(self.w_EDR)
        data.append(self.w_RESOURCE)
        data.append(self.w_DISTANCE)
        data.append(agent_num)
        for task_num, task in enumerate(self.tasks):
            vectorized_task_loc = self.get_vectorized_location(task.getloc())
            is_occupied = self.agent_locations[0][vectorized_task_loc]  # 1 if occupied
            data.append(is_occupied)
        # data.extend(np.ndarray.tolist(self.agent_locations)) # Feature 1
        data.extend(np.ndarray.tolist(self.is_task_finished))  # Feature 2
        data.extend(np.ndarray.tolist(self.is_task_enabled))  # Feature 3
        data.extend(np.ndarray.tolist(self.is_task_alive))  # Feature 4
        data.extend(np.ndarray.tolist(self.travel_time_constraint_satisfied[agent_num]))  # Feature 5
        data.extend(self.is_agent_idle[agent_num])  # Feature 6
        data.extend(np.ndarray.tolist(self.agent_distances[agent_num]))  # Feature 7
        for task_num, task in enumerate(self.tasks):
            vectorized_task_loc = self.get_vectorized_location(task.getloc())
            tasks_in_each_square = self.how_many_tasks_in_each_square[0][vectorized_task_loc]  # 1 if occupied
            data.append(tasks_in_each_square)
        # data.extend(np.ndarray.tolist(self.how_many_tasks_in_each_square)) # Feature 8
        data.extend(np.ndarray.tolist(self.orientation[agent_num]))  # Feature 9
        data.extend(np.ndarray.tolist(self.task_deadlines))  # Feature 10
        data.extend(np.ndarray.tolist(self.is_task_in_progress))  # Feature 11
        data.extend(np.ndarray.tolist(self.orientation[agent_num] * self.agent_distances[agent_num]))  # Feature 12
        data.append(self.task_to_schedule)  # Output
        self.naive_total_data.append(data)
        # with open('1_schedule.csv', 'a') as outfile:
        #     writer = csv.writer(outfile)
        #     writer.writerow(data)

    def write_csv_pairwise(self, agent_num):
        """
        writes a schedule in pairwise format
        That means n rows will be presented, where n is the number of tasks. Each
        row contains task specific features
        :param agent_num:
        :return:
        """
        # self.last_timestep_data = []
        for task_num, i in enumerate(self.tasks):
            current_task_data = []
            task_loc = i.getloc()
            vectorized_task_loc = self.get_vectorized_location(task_loc)
            current_task_data.append(self.t)
            current_task_data.append(self.w_EDR)
            current_task_data.append(self.w_RESOURCE)
            current_task_data.append(self.w_DISTANCE)
            current_task_data.append(agent_num)

            current_task_data.append(task_num)
            current_task_data.extend(self.is_agent_idle[agent_num])  # Feature 6
            current_task_data.append((self.is_task_finished[0][task_num]))  # Feature 2
            current_task_data.append((self.is_task_enabled[0][task_num]))  # Feature 3
            current_task_data.append((self.is_task_alive[0][task_num]))  # Feature 4
            current_task_data.append((self.travel_time_constraint_satisfied[agent_num][task_num]))  # Feature 5
            is_occupied = self.agent_locations[0][vectorized_task_loc]  # if 1 agent is there, 0 is unoccupied
            current_task_data.append((is_occupied))  # Feature 1
            current_task_data.append((self.agent_distances[agent_num][task_num]))  # Feature 7
            current_task_data.append((self.orientation[agent_num][task_num]))  # Feature 9
            current_task_data.append((self.task_deadlines[0][task_num]))  # Feature 10
            current_task_data.append((self.is_task_in_progress[0][task_num]))  # Feature 11
            current_task_data.append((
                    self.orientation[agent_num][task_num] * self.agent_distances[agent_num][task_num]))  # Feature 12
            current_task_data.append((self.how_many_tasks_in_each_square[0][vectorized_task_loc]))  # Feature 8
            if self.task_to_schedule == -1:
                null_task = 1
            else:
                null_task = 0
            current_task_data.append(null_task)
            current_task_data.append(self.task_to_schedule[0])  # Output
            self.pairwise_total_data.append(current_task_data)
            # self.last_timestep_data.append(current_task_data)
            # with open('11_schedule.csv', 'a') as outfile:
            #     writer = csv.writer(outfile)
            #     writer.writerow(current_task_data)

    def update_current_state(self, agent_num):
        """
        gets the current state
        :param agent_num:
        :return:
        """
        for task_num, i in enumerate(self.tasks):
            current_task_data = []
            task_loc = i.getloc()
            vectorized_task_loc = self.get_vectorized_location(task_loc)
            current_task_data.append(self.t)
            current_task_data.append(self.w_EDR)
            current_task_data.append(self.w_RESOURCE)
            current_task_data.append(self.w_DISTANCE)
            current_task_data.append(agent_num)

            current_task_data.append(task_num)
            current_task_data.extend(self.is_agent_idle[agent_num])  # Feature 6
            current_task_data.append((self.is_task_finished[0][task_num]))  # Feature 2
            current_task_data.append((self.is_task_enabled[0][task_num]))  # Feature 3
            current_task_data.append((self.is_task_alive[0][task_num]))  # Feature 4
            current_task_data.append((self.travel_time_constraint_satisfied[agent_num][task_num]))  # Feature 5
            is_occupied = self.agent_locations[0][vectorized_task_loc]  # if 1 agent is there, 0 is unoccupied
            current_task_data.append((is_occupied))  # Feature 1
            current_task_data.append((self.agent_distances[agent_num][task_num]))  # Feature 7
            current_task_data.append((self.orientation[agent_num][task_num]))  # Feature 9
            current_task_data.append((self.task_deadlines[0][task_num]))  # Feature 10
            current_task_data.append((self.is_task_in_progress[0][task_num]))  # Feature 11
            current_task_data.append((
                    self.orientation[agent_num][task_num] * self.agent_distances[agent_num][task_num]))  # Feature 12
            current_task_data.append((self.how_many_tasks_in_each_square[0][vectorized_task_loc]))  # Feature 8
            if self.task_to_schedule == -1:
                null_task = 1
            else:
                null_task = 0
            current_task_data.append(null_task)
            current_task_data.append(-7)  # This is not really needed, but I'll add in -7 as a bs val
            self.network_state.append(current_task_data)

    def update_agent_pose_and_finish_time_and_log_event(self, agent_num):
        """
        updates agent properties and logs event
        :return:
        """
        agent = self.agents[agent_num]
        if self.task_to_schedule[0] == -1:
            pass
        else:
            # this happens as soon as it is scheduled, i think
            scheduled_task = self.task_to_schedule[0]
            agent.curr_task = scheduled_task  # assigns inside agent class (REDUNDANCY)
            agent.set_orientation(self.orientation[agent_num][scheduled_task])
            agent.task_list.append(scheduled_task)
            agent.updateAgentLocation(self.tasks[scheduled_task].getloc())

            # Record it
            agent.task_event_dict[scheduled_task] = [self.t, self.t + self.tasks[scheduled_task].getc()]
            agent.setFinishTime(self.t + self.tasks[scheduled_task].getc())
            self.is_task_in_progress[0][self.task_to_schedule[0]] = 1

    def update_based_on_time(self):
        """
        Checks finish condition for task
        :return:
        """
        for counter, agent in enumerate(self.agents):
            if self.t >= agent.getFinishTime() and self.agent_current_task[counter] != -1:  # task is finished
                task_num = self.agent_current_task[counter]
                self.finish_time_per_task_dict[task_num] = self.t
                self.is_task_finished[0][task_num] = 1
                agent.changebusy(False)
        self.update_agent_is_idle_based_on_class()

    def check_if_schedule_finished(self):
        """
        Checks finish condition for schedule
        :return:
        """
        tot_num_tasks_scheduled = sum(self.is_task_finished[0])
        if tot_num_tasks_scheduled > 19 or self.t > 150:
            self.data_done_generating = True
            if self.t > 150:
                print('Schedule failed to create')
                print('Schedule will not be copied')
                self.did_schedule_fail = True
            else:
                print('Successful schedule created')
                # copy rows into another excel file
                # with open(self.filepath, 'r') as csvfile, open(self.writepath, 'a') as outfile:
                #     data = (csv.reader(csvfile))
                #     writer = csv.writer(outfile)
                #     for row in data:
                #         writer.writerow(row)
                #
                # with open(self.second_file_path, 'r') as csvfile, open(self.writepath2, 'a') as outfile:
                #     data = (csv.reader(csvfile))
                #     writer = csv.writer(outfile)
                #     for row in data:
                #         writer.writerow(row)

                print('1 schedule created.')

    def has_any_deadlines_passed(self, neural_net_task):
        deadlines_passed = False
        for i, each_task in enumerate(self.tasks):
            if self.t > self.task_deadlines[0][i]:
                deadlines_passed = True

        return deadlines_passed


    def update_network(self):
        """
        performs a full training on the network
        :return:
        """

        device = torch.device("cpu")
        self.model =  ProLoNet(input_dim=13,
                          weights=None,
                          comparators=None,
                          leaves=32,
                          output_dim=1,
                          bayesian_embedding_dim=8,
                          alpha=1.5,
                          use_gpu=False,
                          vectorized=True,
                          is_value=True).to(device)

        self.embedding_optimizer = torch.optim.RMSprop([{'params': self.model.bayesian_embedding.parameters()}], lr=.1)
        self.embedding_list = [torch.ones(3) * 1 / 3 for i in range(2000)]
        self.opt =  torch.optim.RMSprop(
            [{'params': list(self.model.parameters())[:-1]}, {'params': self.model.bayesian_embedding.parameters(), 'lr': .01}], lr=.01)

        criterion = torch.nn.BCELoss()

        n_epochs = 4000 + self.global_schedule_num * 3
        for epoch in range(n_epochs):
            which_schedule = np.random.randint(len(self.data_so_far))
            timestep_within_schedule = np.random.randint(len(self.teacher_actions[which_schedule]))

            index_within_network_state = timestep_within_schedule * 20
            timestep_data_from_agg = self.data_so_far[which_schedule][index_within_network_state:index_within_network_state+20]
            task = self.teacher_actions[which_schedule][timestep_within_schedule]
            # set the embedding
            self.model.set_bayesian_embedding(self.embedding_list[which_schedule].clone())
            # update loop

            phi_i_num = task
            phi_i = self.get_features_from_timestep_data_from_agg(timestep_data_from_agg, phi_i_num)
            phi_i_numpy = np.asarray(phi_i)
            loss_counter = 0
            # iterate over pairwise comparisons
            for counter in range(0, 0 + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.get_features_from_timestep_data_from_agg(timestep_data_from_agg, counter)
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_i_numpy - phi_j_numpy

                    if self.use_gpu:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        label = Variable(torch.Tensor(torch.ones((1, 1))))
                    sig = torch.nn.Sigmoid()
                    output = sig(self.model(feature_input))
                    loss = criterion(output, label)
                    # prepare optimizer, compute gradient, update params
                    loss_counter += loss.item()
                    self.opt.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()

            for counter in range(0, 0 + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.get_features_from_timestep_data_from_agg(timestep_data_from_agg, counter)
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy

                    if self.use_gpu:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        label = Variable(torch.Tensor(torch.zeros((1, 1))))
                    sig = torch.nn.Sigmoid()
                    output = sig(self.model.forward(feature_input))

                    self.opt.zero_grad()
                    loss = criterion(output, label)
                    loss_counter += loss.item()

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()
            self.loss_array.append(loss_counter / 38)
            self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()).clone()  # very ugly




