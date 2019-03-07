import sys, os, argparse
import numpy as np

""" Make file to do grid-search over the hyperparameters of model fine-tuning. """

DEBUG = False

def check_dir(path):
    if not os.path.exists(path):
        msg = "input_dir ({}) does not exist".format(path)
        raise ValueError(msg)
    if not os.path.isdir(path):
        msg = "input_dir ({}) is not a directory".format(path)
        raise ValueError(msg)

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="path of directory containing pre-trained model")
parser.add_argument("output_dir", help="path of directory we will create to store the fine-tuned models for each config")
parser.add_argument("data_dir", help="path of directory containing train.tsv and dev.tsv")
parser.add_argument("--cuda_visible_devices", type=str, required=False, help="comma-separated list of CUDA visible devices")
args = parser.parse_args()

# Make sure input dir and output dir exist
#check_dir(args.input_dir)
check_dir(args.output_dir)

# Check if output dir is empty
if len(os.listdir(args.output_dir)):
    msg = "\n\nWARNING: output_dir ({}) is not empty.\n".format(args.output_dir)
    print(msg, file=sys.stderr)

# Get number of training examples
with open(os.path.join(args.data_dir, "train.tsv")) as f:
    nb_train_examples = sum(1 for line in f if len(line.strip()))
if DEBUG:
    print("\nNb training examples: {}".format(nb_train_examples), file=sys.stderr)

# Check cuda_visible_devices
if args.cuda_visible_devices:
    dev_list = args.cuda_visible_devices.split(",")
    dev_list = [int(x) for x in dev_list]

# Set a few constant hparams
max_seq_length = 128
eval_batch_size = 32
nb_train_epochs = 10
nb_warmup_steps = 10000

if args.cuda_visible_devices:
    nb_gpus = len(dev_list)
else:
    nb_gpus = -1
seed = 91500

# Make list of prefixes for the command
prefixes = []
if args.cuda_visible_devices:
    prefixes.append("CUDA_VISIBLE_DEVICES=\"{}\"".format(",".join([str(x) for x in dev_list])))

# Create base command
base_cmd = " ".join(prefixes)
base_cmd += " python run_BERT_classifier.py"
base_cmd += " --data_dir {}".format(args.data_dir)
base_cmd += " --bert_model_or_config_file {}".format(args.input_dir)
base_cmd += " --max_seq_length {}".format(max_seq_length)
base_cmd += " --do_train --do_eval"
base_cmd += " --eval_batch_size {}".format(eval_batch_size)
base_cmd += " --num_train_epochs {}".format(nb_train_epochs)
base_cmd += " --num_gpus {}".format(nb_gpus)
base_cmd += " --seed {}".format(seed)

# Loop over hparam settings and create commands
cmds = []
for bs in ["16", "32"]:

    # Compute warmup proportion so that we do the expected number of warmup steps (10K in the BERT paper)
    steps_per_epoch = nb_train_examples // int(bs)
    wp = nb_warmup_steps / (nb_train_epochs * steps_per_epoch)
    wp = np.round(wp, decimals=3)
    # Clip at 0.5 max.
    if wp > 0.5:
        wp = 0.5
    if DEBUG:
        print("\nBatch size: {}".format(bs), file=sys.stderr)
        print("Nb steps per epoch: {}".format(steps_per_epoch), file=sys.stderr)
        print("Warmup proportion: {}".format(wp), file=sys.stderr)
        
    for lr in ["5e-5", "3e-5", "2e-5"]:
        cmd = base_cmd

        # Set warmup proportion 
        cmd += " --warmup_proportion {}".format(wp)

        # Set train batch size
        cmd += " --train_batch_size {}".format(bs)

        # Set learning rate
        cmd += " --learning_rate {}".format(lr)

        # Set output dir
        output_subdir_name = "model_bs={}_wp={}_lr={}".format(bs, wp, lr)
        output_subdir_path = os.path.join(args.output_dir, output_subdir_name)
        cmd += " --output_dir {}".format(output_subdir_path)
        cmds.append(cmd)

# Add an empty line to stderr in case stderr and stdout are the same, for prettier output
if DEBUG:
    print("\n", file=sys.stderr)
        
# Write commands to stdout
for cmd in cmds:
    print(cmd, file=sys.stdout)
