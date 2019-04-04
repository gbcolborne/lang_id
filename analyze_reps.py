import os
import argparse
import numpy as np
import scipy.stats as st
from evaluate import get_scores

def mean_confidence_interval(data, confidence=0.95):
    """ Source: https://stackoverflow.com/a/15034143 """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return h

parser = argparse.ArgumentParser()
parser.add_argument("path_reps", help="path of directory containing one or more subdirectories containing results of a fine-tuning test (including training_log.txt)")
parser.add_argument("path_gold", help="path of gold labeled test set")
args = parser.parse_args()


dev_scores = []
test_scores = []
for subdir in os.listdir(args.path_reps):
    # Get dev f-scores from each training log
    path = os.path.join(args.path_reps, subdir, "training_log.txt")
    scores = []
    with open(path) as f:
        # Skip header
        next(f)
        for line in f:
            line = line.strip()
            if len(line):
                vals = line.split("\t")
                score = float(vals[-1])
                scores.append(score)
    dev_scores.append(scores)

    # Get test scores by evaluating test_pred.txt against the gold standard
    path_pred = os.path.join(args.path_reps, subdir, "test_pred.txt")    
    scores = get_scores(args.path_gold, path_pred)
    a, p, r, f, wp, wr, wf = scores
    test_scores.append(f)

# Compute mean scores
best_dev_scores = []
best_epochs = []
for i in range(len(dev_scores)):
    scores = dev_scores[i]
    best_score = float("-inf")
    best_epoch = None
    for j, score in enumerate(scores):
        if score > best_score:
            best_score = score
            best_epoch = j + 1
    best_epochs.append(best_epoch)
    best_dev_scores.append(best_score)
mean_dev_score = np.mean(best_dev_scores)
mean_dev_score_ci = mean_confidence_interval(best_dev_scores)
print("Mean dev score: {:.3f} +/- {:.3f}".format(mean_dev_score, mean_dev_score_ci))
mean_best_epoch = np.mean(best_epochs)
mean_best_epoch_ci = mean_confidence_interval(best_epochs)
print("Mean best epoch: {:.3f} +/- {:.3f}".format(mean_best_epoch, mean_best_epoch_ci))
mean_test_score = np.mean(test_scores)
mean_test_score_ci = mean_confidence_interval(test_scores)
print("Mean test score: {:.3f} +/- {:.3f}".format(mean_test_score, mean_test_score_ci))




