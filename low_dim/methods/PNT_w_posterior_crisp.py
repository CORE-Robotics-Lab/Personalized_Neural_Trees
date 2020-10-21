"""
NN implementation w/ embedding evaluating train and test performance on a heterogeneous dataset
created on May 18,  2020 by Rohan Paleja
"""
from low_dim.generate_environment import create_simple_classification_dataset
from low_dim.utils.accuracy_measures import compute_specificity, compute_sensitivity
from low_dim.utils.helper_utils import save_performance_results
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from low_dim.prolonet import ProLoNet

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

input_size = 2  # Just the x dimension
hidden_size = 10  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. In this case, from 0 to 1
learning_rate = 1e-3  # The speed of convergence

ddt = ProLoNet(input_dim=input_size,
               output_dim=num_classes,
               weights=None,
               comparators=None,
               leaves=8,
               is_value=False,
               bayesian_embedding_dim=2,
               vectorized=True,
               selectors=None)

checkpoint = torch.load('/home/Anonymous/PycharmProjects/bayesian_prolo/base_testing_environment/additions_for_HRI/models/models_pretrain.tar')
ddt.load_state_dict(checkpoint['policy'])
posterior = ProLoNet(input_dim=4,
                     output_dim=2,
                     weights=None,
                     comparators=None,
                     leaves=16,
                     is_value=True,
                     vectorized=True,
                     selectors=None)


class Classifier_MLP(nn.Module):
    def __init__(self, ddt, in_dim, hidden_dim, out_dim):
        super(Classifier_MLP, self).__init__()
        self.ddt = ddt
        self.posterior = posterior
        # self.h1 = nn.Linear(4, hidden_dim)
        # self.h2 = nn.Linear(hidden_dim, hidden_dim)
        # self.out = nn.Linear(hidden_dim, 2)
        # self.out_dim = out_dim

    def forward_action(self, x, embedding):
        self.ddt.set_bayesian_embedding(list(embedding))
        pred = ddt(x)
        return pred

    def forward(self, x, embedding):
        self.ddt.set_bayesian_embedding(list(embedding))
        pred = ddt(x)
        x = torch.cat((x, pred.reshape(1, 2)), dim=1)
        x = self.posterior(x)
        return x


nn = Classifier_MLP(ddt, in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)

optimizer = torch.optim.SGD([{'params': list(nn.ddt.parameters())[:-1]}, {'params': nn.ddt.bayesian_embedding.parameters(), 'lr': .01}],
                            lr=learning_rate)
tot_optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)
solo_embedding_optimizer = torch.optim.SGD([{'params': nn.ddt.bayesian_embedding.parameters()}], lr=.1)

epochs = 2000
schedule_starts = np.linspace(0, 20 * (num_schedules - 1), num=num_schedules)
distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
cv_distributions = [np.ones(2) * 1 / 2 for _ in range(10)]
loss_array = []
for epoch in range(epochs):  # loop over the dataset multiple times

    for i in range(num_schedules):
        loss_sum = 0
        chosen_schedule_start = int(np.random.choice(schedule_starts))
        schedule_num = int(chosen_schedule_start / 20)

        # ddt.set_bayesian_embedding(list(distributions[schedule_num]))
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            # pred = ddt(x[each_t]).reshape(1, 2)
            tot_optimizer.zero_grad()
            output = nn.forward(x[each_t], embedding=list(distributions[schedule_num]))
            loss = (torch.Tensor(distributions[schedule_num]) - output) ** 2 * 10  # 10 is usual
            loss1 = loss.sum()
            out = nn.forward_action(x[each_t], embedding=list(distributions[schedule_num]))
            loss2 = F.cross_entropy(out, y[each_t].long()) * 5  # usual is 30
            tot_loss = loss1 + loss2
            tot_loss.backward()
            tot_optimizer.step()
            loss_sum += tot_loss.item()

            distributions[schedule_num] = list(nn.ddt.get_bayesian_embedding().detach().numpy())  # very ugly
        # print(distributions[schedule_num])
        print(ddt.get_bayesian_embedding())
        loss_array.append(loss_sum)
        print('loss ', loss_sum)
    # finished a train loop
    learning_rate /= 1.01  # lower learning rate for convergence
    test_losses, test_accs = [], []
    # cross validation
    for i in range(10):
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        nn.ddt.set_bayesian_embedding(list(cv_distributions[schedule_num]))
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            solo_embedding_optimizer.zero_grad()
            pred = nn.ddt(x_test[each_t]).reshape(1, 2)
            loss = F.cross_entropy(pred, y_test[each_t].long())
            loss.backward()
            solo_embedding_optimizer.step()
            acc = (pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
            test_losses.append(loss.item())
            test_accs.append(acc.mean().item())
    print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
    if epoch % 5 == 0:
        from matplotlib import pyplot as plt
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.cluster import KMeans

        tmp_dist = np.array(distributions)
        plt.scatter(tmp_dist[:, 0], tmp_dist[:, 1])
        plt.title('Epoch: ' + str(epoch))
        plt.xlim(0, 2)
        plt.ylim(-1, 2)
        plt.show()
        torch.save({'policy': nn.ddt.state_dict(), 'posterior': nn.posterior.state_dict()},
                   '/home/Anonymous/PycharmProjects/bayesian_prolo/base_testing_environment/additions_for_HRI/models/models_interpretable3' + str(
                       epoch) + '.tar')
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
