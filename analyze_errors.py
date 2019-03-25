import argparse
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("path_pred")
parser.add_argument("path_gold")
parser.add_argument("path_train")
args = parser.parse_args()

def load_file(path):
    with open(path) as f:
        texts = []
        labels = []
        for line in f:
            line = line.strip()
            if len(line):
                text, label = line.split("\t")
                texts.append(text)
                labels.append(label)
    return texts, labels

texts, gold = load_file(args.path_gold)
pred = [x for x in open(args.path_pred).read().split("\n") if len(x)]

assert len(pred) == len(texts)

nb_correct = 0
nb_incorrect = 0
for p,g in zip(pred, gold):
    if p == g:
        nb_correct += 1
    else:
        nb_incorrect += 1
nb_test_cases = nb_correct + nb_incorrect
print("Nb test cases: {}".format(nb_test_cases))
print("Nb correct: {}".format(nb_correct))
print("Nb incorrect: {}".format(nb_incorrect))

# Check impact of text length
border = "-" * 21
print("\n{}\nIMPACT OF TEXT LENGTH\n{}".format(border, border))

len_sum_correct = 0
len_sum_incorrect = 0
len_to_nb_correct = defaultdict(int)
len_to_nb_incorrect = defaultdict(int)
for t,p,g in zip(texts, pred, gold):
    if p == g:
        len_sum_correct += len(t)
        len_to_nb_correct[len(t)] += 1
    else:
        len_sum_incorrect += len(t)
        len_to_nb_incorrect[len(t)] += 1
print("Avg text length: {} chars".format(np.mean([len(x) for x in texts])))
print("Avg text length (correct): {:.1f} chars".format(len_sum_correct/nb_correct))
print("Avg text length (incorrect): {:.1f} chars".format(len_sum_incorrect/nb_incorrect))

lens = set(len_to_nb_incorrect.keys())
lens.update(len_to_nb_correct.keys())
print()
for k in sorted(lens)[:10]:
    nb_errors = len_to_nb_incorrect[k]
    nb_cases = len_to_nb_correct[k] + nb_errors
    error_rate = 100 * nb_errors / nb_cases 
    print("Error rate for len = {}: {:.1f}%".format(k, error_rate)) 


total_nb_errors = nb_incorrect
cum_nb_errors = 0
print()
for k in sorted(lens)[:10]:
    nb_errors = len_to_nb_incorrect[k]
    cum_nb_errors += nb_errors
    error_coverage = 100 * cum_nb_errors / total_nb_errors
    print("% errors where len <= {}: {:.1f}%".format(k, error_coverage))


# Check impact of OOV
border = "-" * 16
print("\n{}\nIMPACT OF OOVs\n{}".format(border, border))
train_texts, train_labels = load_file(args.path_train)
char_to_freq = defaultdict(int)
for t in train_texts:
    for c in t:
        char_to_freq[c] += 1
OOV_chars = set()
nb_contains_OOV = 0
nb_all_OOV = 0
for t in texts:
    contains_OOV = False
    all_OOV = True
    for c in t:
        if char_to_freq[c] == 0:
            OOV_chars.add(c)
            contains_OOV = True
        else:
            all_OOV = False
    if contains_OOV:
        nb_contains_OOV += 1
    if all_OOV:
        nb_all_oov += 1
print("Nb OOV chars: {}".format(len(OOV_chars)))
print("Nb unique OOV chars: {}".format(len(set(OOV_chars))))
print("Nb texts containing OOV: {}".format(nb_contains_OOV))
print("Nb texts containing only OOV: {}".format(nb_all_OOV))


# Check impact of label ambiguity
border = "-" * 25
print("\n{}\nIMPACT OF LABEL AMBIGUITY\n{}".format(border, border))
text_to_labels = defaultdict(list)
for t, l in zip(train_texts, train_labels):
    text_to_labels[t].append(l)

nb_seen = 0
nb_seen_with_unseen_label = 0
nb_seen_with_ambig_labels = 0
for t, p, g in zip(texts, pred, gold):
    if t in text_to_labels:
        nb_seen += 1
        seen_labels = text_to_labels[t]
        uniq_seen_labels = set(seen_labels)
        if g not in uniq_seen_labels:
            nb_seen_with_unseen_label += 1
        else:
            if len(uniq_seen_labels) > 2:
                nb_seen_with_ambig_labels += 1
print("Nb seen texts: {}/{}".format(nb_seen, nb_test_cases))
print("Nb seen texts whose label was NOT observed during training: {}".format(nb_seen_with_unseen_label))
print("Nb seen texts whose labe WAS seen, but for which other labels were observed: {}".format(nb_seen_with_ambig_labels))
