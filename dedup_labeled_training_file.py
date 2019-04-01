import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="path of tsv file containing labeled training data")
parser.add_argument("dict_path", help="path of tsv file that maps texts to their most frequent label")
parser.add_argument("output_path", help="path of output file")
args = parser.parse_args()

# Load dict 
with open(args.dict_path) as f:
    text_to_best_label = {}
    for line in f:
        line = line.strip()
        text, label = line.split("\t")
        text_to_best_label[text] = label

# Load data, ignore labels
with open(args.data_path) as f:
    texts = set()
    for line in f:
        line = line.strip()
        text, label = line.split("\t")
        texts.add(text)

# Write deduplicated data
with open(args.output_path, "w") as f:
    for text in texts:
        f.write("{}\t{}\n".format(text, text_to_best_label[text]))
