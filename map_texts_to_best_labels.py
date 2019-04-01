import argparse
import os
from collections import defaultdict, Counter

def get_best_label(labels):
    label_freq_dist = Counter(labels)
    max_freq = 0
    best_label = None
    for label, freq in label_freq_dist.items():
        if freq > max_freq:
            max_freq = freq
            best_label = label
    return best_label


parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("output_path")
args = parser.parse_args()

text_to_labels = defaultdict(list)
with open(args.data_path) as f:
    for line in f:
        line = line.strip()
        text, label = line.split("\t")
        text_to_labels[text].append(label)

text_to_best_label = {}
for (text, labels) in text_to_labels.items():
    best_label = get_best_label(labels)
    text_to_best_label[text] = best_label
    if len(set(labels)) > 2:
        print(best_label)

with open(args.output_path, "w") as f:
    for text, label in text_to_best_label.items():
        f.write("{}\t{}\n".format(text, label))
