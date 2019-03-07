import argparse
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gold", help="tab-separated gold file")
parser.add_argument("pred", help="text file containing predicted labels (one per line)")
args = parser.parse_args()

with open(args.gold) as f:
    gold = []
    for line in f:
        line = line.strip()
        text, label = line.split("\t")
        gold.append(label)

with open(args.pred) as f:
    pred = []
    for line in f:
        label = line.strip()
        pred.append(label)

nb_test_cases = len(gold)
classes = sorted(set(pred + gold))

# Accuracy
gold = np.asarray(gold)
pred = np.asarray(pred)
nb_correct = (gold == pred).sum()
accuracy = nb_correct / len(gold)
print("Accuracy: {}".format(accuracy))

# Per-class precision, recall, and f-score
nb_correct_by_class = defaultdict(int)
nb_true_by_class = defaultdict(int)
nb_pred_by_class = defaultdict(int)
for (x,y) in zip(gold, pred):
    nb_true_by_class[x] += 1
    nb_pred_by_class[y] += 1
    if x == y:
        nb_correct_by_class[x] += 1
p_vals = []
r_vals = []
f_vals = []
for k in classes:
    if nb_correct_by_class[k] == 0 or nb_true_by_class[k] == 0:
        p = 0
        r = 0
        f = 0
    else:
        p = nb_correct_by_class[k] / nb_pred_by_class[k]
        r = nb_correct_by_class[k] / nb_true_by_class[k]
        f = 2 * p * r / (p+r)
    p_vals.append(p)
    r_vals.append(r)
    f_vals.append(f)

# Unweighted macro-averaged scores
p = np.mean(p_vals)
r = np.mean(r_vals)
f = np.mean(f_vals)
print("Unweighted macro-averaged precision: {:.5f}".format(p))
print("Unweighted macro-averaged recall: {:.5f}".format(r))
print("Unweighted macro-averaged f-score: {:.5f}".format(f))

# Weighted macro-averaged scores
weights = np.asarray([nb_true_by_class[k] / nb_test_cases for k in classes])
wp = np.sum(p_vals * weights)
wr = np.sum(r_vals * weights)
wf = np.sum(f_vals * weights)
print("Weighted macro-averaged precision: {:.5f}".format(wp))
print("Weighted macro-averaged recall: {:.5f}".format(wr))
print("Weighted macro-averaged f-score: {:.5f}".format(wf))
