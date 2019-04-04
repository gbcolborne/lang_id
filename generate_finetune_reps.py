import argparse
import sys
import os

# Constants
NB_REPS = 5
INITIAL_SEED = 91500

# Args
parser = argparse.ArgumentParser()
parser.add_argument("bert_model_or_config_file")
parser.add_argument("output_dir")
parser.add_argument("data_dir")
parser.add_argument("nb_epochs")
args = parser.parse_args()

# Check args
if not os.path.isdir(args.output_dir):
    msg = "output_dir '{}' does not exist".format(args.output_dir)
    raise ValueError(msg)

# Set base command
BASE_CMD = "python run_BERT_classifier.py --max_seq_length 128 --do_train --do_eval --do_predict --train_batch_size 32 --learning_rate 1e-5 --warmup_proportion 0.5 --num_gpus 1"

# Add constant args to command
BASE_CMD += " --data_dir {}".format(args.data_dir)
BASE_CMD += " --bert_model_or_config_file {}".format(args.bert_model_or_config_file)
BASE_CMD += " --num_train_epochs {}".format(args.nb_epochs)

# Make commands by copying command, but using a different seed
cmds = []
seed = INITIAL_SEED
for rep in range(NB_REPS):
    cmd = BASE_CMD 
    cmd += " --seed {}".format(seed)
    output_subdir = os.path.join(args.output_dir, str(seed))
    cmd += " --output_dir {}".format(output_subdir)
    cmds.append(cmd)
    seed += 1


# Write commands
sys.stdout.write("#!/usr/bin/env bash\n")
sys.stdout.write(" ;\n".join(cmds))
