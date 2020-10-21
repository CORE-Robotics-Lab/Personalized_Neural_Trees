"""
NN implementation w/ embedding evaluating train and test performance on a heterogeneous dataset
created on May 17, 2019 by Anonymous
"""
from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity
from base_testing_environment.utils.helper_utils import save_performance_results
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_testing_environment.prolonet import ProLoNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)

# Training set generation
num_schedules = 50
x_data, y = create_simple_classification_dataset(num_schedules, train=True)

x = []
for each_ele in x_data:
    x.append(each_ele[2:])

x = torch.Tensor(x).reshape(-1, 1, 2)
y = torch.Tensor(y).reshape((-1, 1))

print('Toy problem generated, and data cleaned')

# Cross Validation Set
x_data_test, y_test = create_simple_classification_dataset(10, cv=True)

x_test = []
for each_ele in x_data_test:
    x_test.append(each_ele[2:])

x_test = torch.Tensor(x_test).reshape(-1, 1, 2)
y_test = torch.Tensor(y_test).reshape((-1, 1))

print('test set generated')


# embedding module
class EmbeddingModule(nn.Module):
    """
    embedding class (allows us to access parameters directly)
    """

    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Parameter(torch.randn(1, 2))

    def forward(self):
        """
        doesn't do anything
        :return:
        """
        return


input_size = 2  # Just the x dimension
hidden_size = 10  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. In this case, from 0 to 1
learning_rate = 1e-3  # The speed of convergence

# for 32 leaves, 13 epochs is the sweet spot
# for 8 leaves, 27 epochs is the sweet spot
# k-fold cross val
for _ in range(3):
    ddt = ProLoNet(input_dim=input_size,
                   output_dim=num_classes,
                   weights=None,
                   comparators=None,
                   leaves=32,
                   is_value=False,
                   bayesian_embedding_dim=2,
                   vectorized=True,
                   selectors=None)

    optimizer = torch.optim.SGD([{'params': list(ddt.parameters())[:-1]}, {'params': ddt.bayesian_embedding.parameters(), 'lr': .01}], lr=learning_rate)
    solo_embedding_optimizer = torch.optim.SGD([{'params': ddt.bayesian_embedding.parameters()}], lr=.1)

    epochs = 27
    schedule_starts = np.linspace(0, 20 * (num_schedules - 1), num=num_schedules)
    distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
    cv_distributions = [np.ones(2) * 1 / 2 for _ in range(10)]
    for epoch in range(epochs):  # loop over the dataset multiple times

        for i in range(num_schedules):
            chosen_schedule_start = int(np.random.choice(schedule_starts))
            schedule_num = int(chosen_schedule_start / 20)

            ddt.set_bayesian_embedding(list(distributions[schedule_num]))
            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                optimizer.zero_grad()
                pred = ddt(x[each_t]).reshape(1, 2)
                loss = F.cross_entropy(pred, y[each_t].long()) * 30
                loss.backward()
                optimizer.step()

            distributions[schedule_num] = list(ddt.get_bayesian_embedding().detach().numpy())  # very ugly
            # print(distributions[schedule_num])
            print(ddt.get_bayesian_embedding())
        # finished a train loop
        learning_rate /= 1.1  # lower learning rate for convergence
        test_losses, test_accs = [], []
        # cross validation
        for i in range(10):
            chosen_schedule_start = int(schedule_starts[i])
            schedule_num = int(chosen_schedule_start / 20)
            ddt.set_bayesian_embedding(list(cv_distributions[schedule_num]))
            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                solo_embedding_optimizer.zero_grad()
                pred = ddt(x_test[each_t]).reshape(1, 2)
                loss = F.cross_entropy(pred, y_test[each_t].long())
                loss.backward()
                solo_embedding_optimizer.step()
                acc = (pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
                test_losses.append(loss.item())
                test_accs.append(acc.mean().item())
        print(epoch, ' Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
    print('Finished Training')

    ### REAL TEST

    x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, True)

    x_test = []
    for each_ele in x_data_test:
        x_test.append(each_ele[2:])

    x_test = torch.Tensor(x_test).reshape(-1, 1, 2)
    y_test = torch.Tensor(y_test).reshape((-1, 1))
    test_losses, test_accs = [], []
    per_schedule_test_losses, per_schedule_test_accs = [], []
    preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]
    test_distributions = [np.ones(2) * 1 / 2 for _ in range(50)]
    for i in range(50):
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        ddt.set_bayesian_embedding(list(test_distributions[schedule_num]))
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            solo_embedding_optimizer.zero_grad()
            pred = ddt(x_test[each_t]).reshape(1, 2)
            loss = F.cross_entropy(pred, y_test[each_t].long())
            loss.backward()
            solo_embedding_optimizer.step()
            preds[i].append(pred.argmax(dim=-1).item())
            actual[i].append(y_test[each_t].item())
            print(pred.argmax(dim=-1), y_test[each_t])
            acc = (pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
            test_losses.append(loss.item())
            test_accs.append(acc.mean().item())
        per_schedule_test_accs.append(np.mean(test_accs))
    print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))


sensitivity, specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)
print('per sched accuracy: ', np.mean(per_schedule_test_accs))
print('mean sensitivity: ', sensitivity, ', mean specificity: ', specificity)

# file = open('heterogeneous_toy_env_results.txt', 'a')
# file.write('DDT w/ embedding: mean: ' +
#            str(np.mean(per_schedule_test_accs)) +
#            ', std: ' + str(np.std(per_schedule_test_accs)) +
#             ', sensitivity: ' + str(sensitivity) + ', specificity: '+ str(specificity) +
#            ', Distribution of Class: 0: ' + str(percent_of_zeros) + ', 1: ' + str(1 - percent_of_zeros) +
#            '\n')
# file.close()
# save_performance_results(per_schedule_test_accs, 'PDDT')
