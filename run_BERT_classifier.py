# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 

Train BERT classifier. 

Code based on: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_lm_finetuning.py.

"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertForPreTraining, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from utils import CuneiformCharTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class CLIProcessor(DataProcessor):
    """Processor for the Cuneiform Language Identification datasets. """
    
    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        np.random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        return examples

    def get_test_examples(self, data_dir):
        """ Get unlabeled test examples from a file named "test.txt" located in `data_dir`. """
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test", labeled=False)
        return examples

    def get_labels(self):
        """See base class."""
        return ["SUX", "OLB", "MPB", "STB", "NEB", "LTB", "NEA"]

    def _create_examples(self, lines, set_type, labeled=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            if labeled:
                label = line[1]
            else:
                label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        if example.label:
            label_id = label_map[example.label]
        else:
            label_id = None
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {} (id = {})".format(example.label, label_id))
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def evaluate(model, eval_dataloader, device):
    pred = []
    gold = []
    eval_loss = 0
    nb_eval_steps = 0

    # Get predicted and gold labels, as well as loss
    model.eval()
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
        logits = logits.detach().cpu().numpy()
        gold += label_ids.to('cpu').numpy().tolist()
        pred += np.argmax(logits, axis=1).tolist()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    model.train()

    # Get average loss
    loss = eval_loss / nb_eval_steps

    # Get set of classes
    classes = sorted(set(pred + gold))

    # Compute accuracy
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    nb_correct = (gold == pred).sum()
    accuracy = nb_correct / len(gold)

    # Per-class f-score
    nb_correct_by_class = defaultdict(int)
    nb_true_by_class = defaultdict(int)
    nb_pred_by_class = defaultdict(int)
    for (x,y) in zip(gold, pred):
        nb_true_by_class[x] += 1
        nb_pred_by_class[y] += 1
        if x == y:
            nb_correct_by_class[x] += 1
    f_vals = []
    for k in classes:
        if nb_correct_by_class[k] == 0 or nb_true_by_class[k] == 0:
            f = 0
        else:
            p = nb_correct_by_class[k] / nb_pred_by_class[k]
            r = nb_correct_by_class[k] / nb_true_by_class[k]
            f = 2 * p * r / (p+r)
        f_vals.append(f)

    # Compute (unweighted) macro-averaged f-score
    fscore = np.mean(f_vals)

    return pred, loss, accuracy, fscore

def predict(model, test_dataloader, device):
    model.eval()
    predictions = []
    for input_ids, input_mask, segment_ids in tqdm(test_dataloader, desc="Predicting"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        logits = logits.detach().cpu().numpy()
        pred = np.argmax(logits, axis=1)
        predictions += pred.tolist()
    model.train()
    return predictions



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain files named train.tsv, dev.tsv, and/or test.txt, depending on whether we are doing training, evaluation or prediction.")
    parser.add_argument("--bert_model_or_config_file", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Directory containing pre-trained BERT model or path of configuration file (if no pre-training).")
    parser.add_argument("--task_name",
                        default="cli",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="Num GPUs to use for training (0 for none, -1 for all available)")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "cli": CLIProcessor,
    }

    num_labels_task = {
        "cli": 7,
    }

    # Check whether bert_model_or_config_file is a file or directory
    if os.path.isdir(args.bert_model_or_config_file):
        pretrained=True
        targets = [WEIGHTS_NAME, CONFIG_NAME, "tokenizer.pkl"]
        for t in targets:
            path = os.path.join(args.bert_model_or_config_file, t)
            if not os.path.exists(path):
                msg = "File '{}' not found".format(path)
                raise ValueError(msg)
        fp = os.path.join(args.bert_model_or_config_file, CONFIG_NAME)
        config = BertConfig(fp)
    else:
        pretrained=False
        config = BertConfig(args.bert_model_or_config_file)

    # What GPUs do we use?
    if args.num_gpus == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        device_ids = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")
        n_gpu = args.num_gpus
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # Check some other args
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict` must be True.")

    # Seed RNGs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare output directory
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    # Get training data
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Make tokenizer
    if pretrained:
        fp = os.path.join(args.bert_model_or_config_file, "tokenizer.pkl")
        with open(fp, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = CuneiformCharTokenizer(training_data=[x.text_a for x in train_examples])
        tokenizer.trim_vocab(config.min_freq)
        # Adapt vocab size in config
        config.vocab_size = len(tokenizer.vocab)
    print("Size of vocab: {}".format(len(tokenizer.vocab)))

    # Prepare model
    if pretrained:
        model = BertForSequenceClassification.from_pretrained(args.bert_model_or_config_file, num_labels=num_labels)
    else:
        model = BertForSequenceClassification(config, num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

            
    # Get evaluation data (we assume its validation data if do_train
    # is also true, and evaluate the model on this data after each
    # epoch)
    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Prepare for training
    global_step = 0
    nb_tr_steps = 0
    total_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Prepare log file
        output_log_file = os.path.join(args.output_dir, "training_log.txt")
        with open(output_log_file, "w") as f:
            if args.do_eval:
                f.write("Steps\tTrainLoss\tValLoss\tValAccuracy\tValFScore\n")
            else:
                f.write("Steps\tTrainLoss\n")

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            avg_loss = tr_loss / nb_tr_examples
            total_tr_steps += nb_tr_steps
            log_data = [str(total_tr_steps), "{:.5f}".format(avg_loss)]

            # Validate 
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                predictions, eval_loss, eval_accuracy, fscore = evaluate(model, eval_dataloader, device)
                log_data.append("{:.5f}".format(eval_loss))
                log_data.append("{:.5f}".format(eval_accuracy))
                log_data.append("{:.5f}".format(fscore))

            # Log
            with open(output_log_file, "a") as f:
                f.write("\t".join(log_data)+"\n")

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        output_tokenizer_file = os.path.join(args.output_dir, "tokenizer.pkl")
        with open(output_tokenizer_file, "wb") as f:
            pickle.dump(tokenizer, f)

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForSequenceClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
        with open(output_tokenizer_file, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model_or_config_file, num_labels=num_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        predictions, eval_loss, eval_accuracy, fscore = evaluate(model, eval_dataloader, device)
        loss = avg_loss if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'eval_fscore': fscore,
                  'global_step': global_step,
                  'loss': loss}

        # Write evaluation results
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # Write predictions
        output_pred_file = os.path.join(args.output_dir, "eval_pred.txt")
        with open(output_pred_file, "w") as writer:
            for label_id in predictions:
                label = label_list[label_id]
                writer.write(label + "\n")

    # Predict labels of test set
    if args.do_predict:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        predictions = predict(model, test_dataloader, device)

        # Write predictions
        output_pred_file = os.path.join(args.output_dir, "test_pred.txt")
        with open(output_pred_file, "w") as writer:
            for label_id in predictions:
                label = label_list[label_id]
                writer.write(label + "\n")

if __name__ == "__main__":
    main()
