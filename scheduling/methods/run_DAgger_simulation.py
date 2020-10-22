"""
Run through a set of schedules to generate a dataset in a naive and pairwise fashion
"""
import sys

sys.path.insert(0, '/home/Anonymous/PycharmProjects/bayesian_prolo')

from scheduling.methods.PDDT_DAGGER import *
from utils.global_utils import save_pickle
import torch
import time
from low_dim.prolonet import ProLoNet


def main():
    """
    entry point of the file
    :return:
    """
    start = time.time()
    print('Starting time is ', start)
    total_schedule = []
    total_schedule_pairwise = []
    n = 0
    num_scheds = 3000
    test = True
    load_in = True
    if test:
        np.random.seed(500)
        random.seed(500)
    else:
        np.random.seed(150)
        random.seed(150)

    model = ProLoNet(input_dim=13,
                          weights=None,
                          comparators=None,
                          leaves=32,
                          output_dim=1,
                          bayesian_embedding_dim=3,
                          alpha=1.5,
                          use_gpu=False,
                          vectorized=True,
                          is_value=True)
    device = torch.device("cpu")
    # model = NNSmall().to(device)

    embedding_list = [torch.ones(3) * 1 / 3 for _ in range(2000)]
    opt = torch.optim.RMSprop(
        [{'params': list(model.parameters())[:-1]}, {'params': model.bayesian_embedding.parameters(), 'lr': .01}], lr=.01)

    # opt = torch.optim.Adam([{'params': list(model.parameters())[:-1]},
    #                         {'params': model.EmbeddingList.parameters(), 'lr': .01}], lr=.0001)
    loss_array = []
    teacher_actions = [[] for _ in range(num_scheds)]
    learner_actions = [[] for _ in range(num_scheds)]
    what_happend_at_every_timestep = [[] for _ in range(num_scheds)]
    timestep_terminated = []
    number_of_decisions_before_terminal_state = [0 for _ in range(num_scheds)]
    num_correct_predictions_total = [0 for _ in range(num_scheds)]
    num_predictions_total = [0 for _ in range(num_scheds)]
    data_so_far = []
    successful_schedules = []
    # things to save
    """
    losses !
    embedding list !
    model !
    expert schedule
    PDDT schedule
    terminated at what timestep
    what happend until that timestep
    why was task not scheduled
    
    Format: list of lists where each index is a schedule and each index within that is a half-timestep (since there are 2 agents)
    """
    # if load_in:
    #     checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/scheduling_env/models/Dagger86acc15sched20noise15.tar')
    #     model.load_state_dict(checkpoint['nn_state_dict'])

    while n < num_scheds:
        world = World(num_scheds, n, model, embedding_list, opt, loss_array, teacher_actions, learner_actions, what_happend_at_every_timestep,
                      timestep_terminated, number_of_decisions_before_terminal_state, num_correct_predictions_total, num_predictions_total, data_so_far)

        # 2 cases this breaks, either t > max duration, or 19 tasks are scheduled.
        while not world.data_done_generating:
            is_feasible = world.update_floyd_warshall_and_all_vectors()
            world.print_all_features()
            if is_feasible == False:
                print('Constraint was infeasible')
                break
            for counter, agent in enumerate(world.agents):
                world.compute_task_to_schedule(counter)

                is_feasible = world.update_floyd_warshall_and_all_vectors()

            world.add_constraints_based_on_task()
            world.check_if_schedule_finished()
            print('Current finish times', world.agents[0].getFinishTime()
                  , world.agents[1].getFinishTime())

            # Next Time Step
            world.t += 1
            print('Current time is ', world.t)
            world.update_based_on_time()

            if world.did_schedule_fail:
                timestep_terminated.append(world.t)
                model = world.model
                embedding_list = world.embedding_list
                data_so_far.append(world.network_state)
                break

        if world.did_schedule_fail:
            n += 1
            print('On schedule ', n)

            if n % 50 == 0:
                torch.save({'nn_state_dict': model.state_dict(),
                            'losses': loss_array,
                            'embedding_list': embedding_list,
                            'teacher_actions': teacher_actions,
                            'learner_actions': learner_actions,
                            'what_happend_at_every_timestep': what_happend_at_every_timestep,
                            'timestep_terminated': timestep_terminated,
                            'number_of_decisions_before_terminal_state': number_of_decisions_before_terminal_state,
                            'num_correct_predictions_total': num_correct_predictions_total,
                            'num_predictions_total': num_predictions_total,
                            'successful_schedules': successful_schedules
                            },
                           '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/dagger/509_NN_' + str(n) + '.tar')

            pass
        else:

            print('SUCCESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            successful_schedules.append(n)
            model = world.model
            embedding_list = world.embedding_list
            timestep_terminated.append(world.t)
            data_so_far.append(world.network_state)
            if n % 50 == 0:
                torch.save({'nn_state_dict': model.state_dict(),
                            'losses': loss_array,
                            'embedding_list': embedding_list,
                            'teacher_actions': teacher_actions,
                            'learner_actions': learner_actions,
                            'what_happend_at_every_timestep': what_happend_at_every_timestep,
                            'timestep_terminated': timestep_terminated,
                            'number_of_decisions_before_terminal_state': number_of_decisions_before_terminal_state,
                            'num_correct_predictions_total': num_correct_predictions_total,
                            'num_predictions_total': num_predictions_total,
                            'successful_schedules': successful_schedules
                            },
                           '/home/Anonymous/PycharmProjects/bayesian_prolo/saved_models/dagger/509_NN_' + str(n) + '.tar')

            total_schedule.append(world.naive_total_data)
            total_schedule_pairwise.append(world.pairwise_total_data)
            n += 1
            print('on schedule ', n)

    end = time.time()
    print('End time is ', end)
    print('Time to run ', end - start)


if __name__ == main():
    main()
