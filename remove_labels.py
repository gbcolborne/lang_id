import os, argparse
from collections import defaultdict

"""Remove labels from training file, and optionally split by classe. This is used to pretrain BERT.

"""

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--split", action="store_true", help="split unlabeled data by class and add empty line between classes (for pre-training on the next sentence prediction task)")
args = parser.parse_args()

# Read training data
with open(args.input) as f:
    class_to_texts = defaultdict(list) # Only used if split is true
    texts = []
    for line in f:
        text, label = line.strip().split("\t")
        texts.append(text)
        class_to_texts[label].append(text)
        
# Write unlabeled data for pretraining
with open(args.output, "w") as f:
    # Split by class
    if args.split:
        for cls, texts in class_to_texts.items():
            for text in texts:
                f.write("{}\n".format(text))
            f.write("\n")
    else:
        # No split
        for text in texts:
            f.write("{}\n".format(text))
