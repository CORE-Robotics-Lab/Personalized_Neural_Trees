"""
sklearn DT evaluating train and test performance on a heterogeneous dataset
created on May 17, 2019 by Anonymous
"""
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity
from base_testing_environment.utils.helper_utils import save_performance_results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)

it_1 = [True, False, False]
it_2 = [False, True, False]
it_3 = [False, False, True]
it = [it_3, it_1, it_2]

# Training set generation
num_schedules = 50
x_data, y = create_simple_classification_dataset(num_schedules, train=it[0][0], cv=it[0][1])

x = []
for each_ele in x_data:
    x.append(each_ele[2:])

x = torch.Tensor(x).reshape(-1, 2)
y = torch.Tensor(y).reshape((-1, 1))

print('Toy problem generated, and data cleaned')


clf = DecisionTreeClassifier(max_depth=10)
clf.fit(x, y)


### REAL TEST

x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, True, train=it[2][0], cv=it[2][1])

x_test = []
for each_ele in x_data_test:
    x_test.append(each_ele[2:])

x_test = torch.Tensor(x_test).reshape(-1, 2)
y_test = torch.Tensor(y_test).reshape((-1, 1))
schedule_starts = np.linspace(0, int(num_schedules * 20 - 20), num=num_schedules)

y_pred_test = clf.predict(x_test)

tot_accuracy = accuracy_score(y_test, y_pred_test)
print(accuracy_score(y_test, y_pred_test))

test_losses, test_accs = [], []
per_schedule_test_losses, per_schedule_test_accs = [], []
preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]
for i in range(50):
    chosen_schedule_start = int(schedule_starts[i])
    for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
        pred = clf.predict(x_test[each_t].reshape((1,-1)))
        preds[i].append(pred.item())
        actual[i].append(y_test[each_t].item())
        acc = (pred == y_test[each_t].item())
        test_accs.append(acc.mean().item())
    per_schedule_test_accs.append(np.mean(test_accs))

sensitivity, specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)
print('per sched accuracy: ', np.mean(per_schedule_test_accs))
print('mean sensitivity: ', sensitivity, ', mean specificity: ', specificity)
print('per sched accuracy: ', np.mean(per_schedule_test_accs))
print('hi')
# file = open('heterogeneous_toy_env_results.txt', 'a')
# file.write('DT: mean: ' +
#            str(np.mean(per_schedule_test_accs)) +
#            ', std: ' + str(np.std(per_schedule_test_accs)) +
#             ', sensitivity: ' + str(sensitivity) + ', specificity: '+ str(specificity) +
#            ', Distribution of Class: 0: ' + str(percent_of_zeros) + ', 1: ' + str(1 - percent_of_zeros) +
#            '\n')
# file.close()
# save_performance_results(per_schedule_test_accs, 'results_DT')