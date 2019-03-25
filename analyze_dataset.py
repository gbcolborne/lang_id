import argparse
import numpy as np
from collections import Counter, defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("path_data")
args = parser.parse_args()

with open(args.path_data) as f:
    texts = []
    labels = []
    for line in f:
        text, label = line.strip().split("\t")
        texts.append(text)
        labels.append(label)

print("\nNb texts: {}".format(len(texts)))
print("Nb unique texts: {}".format(len(set(texts))))
        
text_lens = [len(x) for x in texts]
avg_text_len = np.mean(text_lens)
print("\nAvg text length: {:.1f} chars".format(avg_text_len))
print("Max text length: {} chars".format(max(text_lens)))
print("20 longest texts: {}".format(sorted(text_lens)[-20:]))

text_len_freqs = Counter(text_lens)
cum_sum = 0
print("\nText length freqs (cumulative):")
for (ln, freq) in sorted(text_len_freqs.items(), key=lambda x:x[0], reverse=False)[:10]:
    cum_sum += freq
    print("<={}: {} ({:.1f}%)".format(ln, cum_sum, 100*cum_sum/len(texts)))
print("...")

class_freqs = Counter(labels)
print("\nClass freqs:")
for (cls, freq) in sorted(class_freqs.items(), key=lambda x:x[1], reverse=False):
    print("{} ({})".format(cls, freq))

# Analyze duplicates
text_to_labels = defaultdict(list)
for (text, label) in zip(texts, labels):
    text_to_labels[text].append(label)
print("\nDuplicate texts: ")
max_shown = 10
for i, (text, labs) in enumerate(sorted(text_to_labels.items(), key=lambda x:len(x[1]), reverse=True)):
    if len(labs) == 1:
        break
    if i == max_shown:
        break
    print("{}. {} labels ({} unique: {})".format(i+1, len(labs), len(set(labs)), set(labs)))

# Analyze char freqs
char_to_freq = defaultdict(int)
for text in texts:
    for char in text:
        char_to_freq[char] += 1
print("\nMost frequent chars:")
max_shown = 10
for (char, freq) in sorted(char_to_freq.items(), key=lambda x:x[1], reverse=True)[:max_shown]:
    print("{} ({})".format(hex(ord(char)), freq))

print("\nNb unique chars: {}".format(len(char_to_freq)))
for max_freq in [1, 5, 10]:
    count = sum(1 for (c,f) in char_to_freq.items() if f <= max_freq)
    print("Nb chars with freq <= {}: {}".format(max_freq, count))
