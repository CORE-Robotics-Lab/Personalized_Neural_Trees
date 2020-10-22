import numpy as np
from utils.global_utils import save_pickle
import torch
from torch.autograd import Variable
import torch.nn.functional as F


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
    save_pickle(file=data, file_location='/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/results',
                special_string=special_string)

def save_performance_results_DT_version(top1, special_string):
    """
    saves performance of top1 and top3
    :return:
    """
    print('top1_mean is : ', np.mean(top1))
    data = {'top1_mean': np.mean(top1),
            'top1_stderr': np.std(top1) / np.sqrt(len(top1))}
    save_pickle(file=data, file_location='/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/results',
                special_string=special_string)



def save_trained_nets(model, name):
    """
    saves the model
    :return:
    """
    torch.save({'nn_state_dict': model.state_dict()},
               '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/models/' + name + '.tar')


def save_trained_nets_and_embeddings(model, training_embeddings, name):
    """
    saves the model
    :return:
    """
    torch.save({'nn_state_dict': model.state_dict(),
                'training_embeddings': training_embeddings},
               '/home/rohanpaleja/PycharmProjects/bayesian_prolo/scheduling_env/models/' + name + '.tar')


# noinspection PyArgumentList
def train_PDDT_naive(epochs, X, Y, schedule_array, model, embedding_list, opt, total_loss_array, when_to_save):
    from utils.naive_utils import find_which_schedule_this_belongs_to

    for epoch in range(epochs):
        # sample a timestep before the cutoff for cross_validation
        rand_timestep_within_sched = np.random.randint(len(X))
        input_nn = X[rand_timestep_within_sched]
        truth_nn = Y[rand_timestep_within_sched]

        which_schedule = find_which_schedule_this_belongs_to(schedule_array, rand_timestep_within_sched)
        model.set_bayesian_embedding(embedding_list[which_schedule].clone())

        if torch.cuda.is_available():
            input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
            truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
        else:
            input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
            truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

        opt.zero_grad()
        output = model.forward(input_nn)
        loss = F.cross_entropy(output, truth)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        embedding_list[which_schedule] = torch.Tensor(model.get_bayesian_embedding().detach().cpu().numpy()).clone()  # very ugly

        # add average loss to array
        # print(list(model.parameters()))

        total_loss_array.append(loss.item())

        if epoch % 50 == 49:
            print('total loss (average for each 40, averaged)', np.mean(total_loss_array[-40:]))
            # print(model.state_dict())

        if epoch % when_to_save == when_to_save - 1:
            save_trained_nets(model, 'naive_PDDT' + str(epoch))

        return model