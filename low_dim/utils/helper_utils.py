"""
Created by Rohan Paleja on September 27, 2019
Purpose: help with baseline environment
"""


import numpy as np
from utils.global_utils import save_pickle


def save_performance_results(top1, special_string):
    """
    saves performance of top1 and top3
    :return:
    """
    print('top1_mean is : ', np.mean(top1))
    data = {'top1_mean': np.mean(top1),
            'top1_stderr': np.std(top1) / np.sqrt(len(top1))}
    save_pickle(file=data, file_location='/home/Anonymous/PycharmProjects/bayesian_prolo/base_testing_environment/additions_for_HRI/results',
                special_string=special_string)


