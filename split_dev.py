import os
import argparse
import random
from collections import defaultdict
from utils import load_labeled_data

""" Split dev set in half, maintaining class frequency distribution. """

parser = argparse.ArgumentParser()
parser.add_argument("path_dev")
parser.add_argument("dir_out", help="path of dir where we write 2 output files")
args = parser.parse_args()

# Check args
if not os.path.isdir(args.dir_out):
    raise ValueError("No directory at '{}'".format(args.dir_out))
path_dev_a = os.path.join(args.dir_out, "dev-A.txt")
path_dev_b = os.path.join(args.dir_out, "dev-B.txt")
for path in [path_dev_a, path_dev_b]:
    if os.path.exists(path):
        raise ValueError("There is already something at '{}'".format(path))

# Load data
texts, labels = load_labeled_data(args.path_dev)

# Map labels to texts
label_to_texts = defaultdict(list)
for t,l in zip(texts, labels):
    label_to_texts[l].append(t)
print("Nb texts per class:")
for k,v in label_to_texts.items():
    print("- {}: {}".format(k, len(v)))

# Split
random.seed(91500)
data_a = []
data_b = []
for k,v in label_to_texts.items():
    # Shuffle
    texts = v[:]
    random.shuffle(texts)
    # Split
    split = len(v) // 2
    for text in texts[:split]:
        data_a.append((text, k))
    for text in texts[split:]:
        data_b.append((text, k))
    
# Shuffle
random.shuffle(data_a)
random.shuffle(data_b)

# Write
with open(path_dev_a, "w") as f:
    for text, label in data_a:
        f.write("{}\t{}\n".format(text, label))
with open(path_dev_b, "w") as f:
    for text, label in data_b:
        f.write("{}\t{}\n".format(text, label))

