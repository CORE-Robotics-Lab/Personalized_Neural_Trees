"""
utils to help generate measure of accuracy in the toy env.
"""
import numpy as np

def compute_sensitivity(preds, actual):
    true_positives = []
    false_negative = []
    for m,i in enumerate(actual):
        per_sched_true_pos = 0
        per_sched_false_neg = 0
        # compute true positives
        for n, j in enumerate(i):
            if j == 1:  # actual is positive
                if preds[m][n] == 1:
                    per_sched_true_pos += 1
                else:
                    per_sched_false_neg += 1
        true_positives.append(per_sched_true_pos/20)
        false_negative.append(per_sched_false_neg/20)
    sensitivities = []
    for i in range(len(true_positives)):
        sensitivities.append(true_positives[i] / (true_positives[i] + false_negative[i]))

    mean_sensitivity = np.mean(sensitivities)
    return mean_sensitivity

def compute_specificity(preds, actual):
    true_negative = []
    false_positive = []
    for m,i in enumerate(actual):
        per_sched_true_neg = 0
        per_sched_false_pos = 0
        # compute true positives
        for n, j in enumerate(i):
            if j == 0:  # actual is positive
                if preds[m][n] == 0:
                    per_sched_true_neg += 1
                else:
                    per_sched_false_pos += 1
        true_negative.append(per_sched_true_neg / 20)
        false_positive.append(per_sched_false_pos / 20)
    specificities = []
    for i in range(len(true_negative)):
        specificities.append(true_negative[i] / (true_negative[i] + false_positive[i]))

    mean_specificity = np.mean(specificities)
    return mean_specificity
