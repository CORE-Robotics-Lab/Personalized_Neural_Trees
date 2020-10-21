"""
file to generate heterogeneous dataset
created on May 18, 2019 by Rohan Paleja
"""
import numpy as np


def create_simple_classification_dataset(n, get_percent_of_zeros=False, train=False, cv=False):
    """
    Regression version:
    y = {
    z * x if assignment = 1
    (2-z) * x if assignment = 2
    Classification version:
    y = q*x

    q = {
    1 if z >= 0 and assignment = 1
    0 if z < 0 and assignment = 1
    1 if z < 0 and assignment = 2
    0 if z >= 0 and assignment = 2
    :param n:
    :return:
    """
    if train:
        np.random.seed(50)
    elif cv:
        np.random.seed(100)
    else:
        np.random.seed(150)
    # sample z from 0 to 1
    lst = [[] for _ in range(n * 20)]  # each list is a timestep
    label = [[] for _ in range(n * 20)]  # each list is a timestep
    for i in range(n):
        if i % 2 == 0:
            lam = 1  # corresponds to network [1,0]
        else:
            lam = 0  # corresponds to network [0,1]

        # lam hold throughout schedule
        for count in range(20):
            z = np.random.normal(0, 1)
            x = np.random.choice([0, 1], p=[.1, .9])
            q = None
            if lam == 1:  # assignment 1
                if z >= 0:
                    q = 1
                else:
                    q = 0
            if lam == 0:  # assignment 2
                if z < 0:
                    q = 1
                else:
                    q = 0

            y = q * x

            lst[i * 20 + count].extend([lam, q, z, x])
            label[i * 20 + count].append(y)

    count = 0
    for i in label:
        if i[0] == 0:
            count += 1
    print('percent of zeros', count / len(label))
    if get_percent_of_zeros:
        return lst, label, count / len(label)
    else:
        return lst, label


# test
def main():
    """
    run to generate 5 schedules
    :return:
    """
    x, y = create_simple_classification_dataset(5)
    count = 0
    for i in y:
        if i[0] == 0:
            count += 1
    print('percent of zeros', count / len(y))  # confirm that the distribution isn't too skewed.


if __name__ == '__main__':
    main()
