import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

from pathlib import Path
import sys

import pandas as pd

path_root = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(path_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    HfArgumentParser, TrainingArguments,
    Trainer,
    AutoTokenizer,
    set_seed
)

from transformers.utils import check_min_version
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard

from datasets import Dataset
import datasets
import math
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, top_k_accuracy_score
logger = logging.getLogger(__name__)

from models.gex_text_dual_encoder.modeling_gex_text_dual_encoder_node2vec import GEXTextDualEncoderModel
from utils.CLIPTrainer import CLIPTrainer
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (gse92743, uci-dgex)."}
    )
    gex_column_name: Optional[str] = field(
        default='gex',
        metadata={"help": "The column name of the images in the files. If not set, will try to use 'image' or 'img'."},
    )
    text_column_name: Optional[str] = field(
        default='label',
        metadata={"help": "The column name of the text in the files. If not set, will try to use 'image' or 'img'."},
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    group_file: Optional[str] = field(default=None, metadata={"help": "A csv file containing how the genes are grouped."})
    project_name: Optional[str] = field(
        default='gext', metadata={"help": "The project name for wandb reporting."}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of predict examples to this "
                    "value if set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/feature extractor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
            "checkpoint identifier on the hub. "
            "Don't set if you want to train a model from scratch."
        },
    )
    text_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
            "checkpoint identifier on the hub. "
            "Don't set if you want to train a model from scratch."
        },
    )
    gex_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
            "checkpoint identifier on the hub. "
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store (cache) the pretrained models/datasets downloaded from the hub"},
    )
    freeze_gex_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the gex model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )

def load_dataset_from_arrow_onclass(data_dir, group_file):
    logger.debug(f'{data_dir}')

    labels_df = pd.read_csv('./labels_df.tsv', sep='\t')
    label_dict = {}
    
    ds = {}
    
    for k in ['train', 'test']:
        ds_path = os.path.join(data_dir, 'ds_cache', k)
        print(ds_path)
        ds_tmp = datasets.load_from_disk(ds_path)

        if k == 'val':
            ds['validation'] = ds_tmp
        else:
            ds[k] = ds_tmp

    for i, r in labels_df.iterrows():
        label_dict[r.ID] = r.cell_name


    gene_group = pd.read_csv(group_file,
                             sep='\t', index_col='Symbol')
    group = gene_group.to_numpy().transpose()
    ds.update({'group': group,
               'gene_list_order': list(gene_group.index),
               'labels_dict': label_dict})
    return ds

def collate_fn(examples, gene_group, gene_list, test_label_num):

    # gene_expression = torch.stack([example["gex"] for example in examples])
    # gene_input_ids = torch.stack([gene_list for _ in range(len(examples))])
    # group_mtx = torch.stack([gene_group for _ in range(len(examples))]).type(torch.FloatTensor)
    # labels = torch.tensor([example["label"] for example in examples])
    #
    # input_ids = torch.stack([example["input_ids"] for example in examples])
    # attention_mask = torch.stack([example["attention_mask"] for example in examples])

    labels_all = [example["label"] for example in examples]
    idx_l = []
    ids = []
    if len(examples[0]["input_ids"]) != test_label_num:
        for i, k in enumerate(labels_all):
            if k not in ids:
                idx_l.append(i)
                ids.append(k)
            if len(idx_l) == 32:
                break
    else:
        idx_l = list(range(len(labels_all)))
    #print(len(idx_l), len(examples[0]["input_ids"]))

    gene_expression = torch.stack([examples[i]["gex"] for i in idx_l])
    gene_input_ids = torch.stack([gene_list for _ in range(len(idx_l))])
    group_mtx = torch.stack([gene_group for _ in range(len(idx_l))]).type(torch.FloatTensor)
    labels = torch.tensor([examples[i]["label"] for i in idx_l])

    input_ids = torch.stack([examples[i]["input_ids"] for i in idx_l])
    attention_mask = torch.stack([examples[i]["attention_mask"] for i in idx_l])
    cl_node2vec = torch.stack([examples[i]["node2vec_emb"] for i in idx_l])
    
    return {"gene_expression": gene_expression, "gene_input_ids": gene_input_ids,
            "group_mtx": group_mtx
            , 'label': labels
            , "input_ids": input_ids, "attention_mask": attention_mask
            , 'cl_node2vec': cl_node2vec
            ,"return_loss": True,}
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    ds = load_dataset_from_arrow_onclass(
        data_dir=data_args.data_dir,
        group_file = data_args.group_file,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        
        # split = ds["train"].train_test_split(data_args.train_val_split, seed=7747)
        # ds["train"] = split["train"]
        # ds["validation"] = split["test"]
        
        ds["validation"] = ds["train"].shuffle(seed=training_args.seed).select(range(int(ds['train'].num_rows * data_args.train_val_split)))

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.text_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_model_name_or_path, cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir#, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if model_args.model_name_or_path:
        model = GEXTextDualEncoderModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    elif model_args.text_model_name_or_path and model_args.gex_model_name_or_path is not None:
        model = GEXTextDualEncoderModel.from_gex_text_pretrained(
            model_args.gex_model_name_or_path,
            model_args.text_model_name_or_path,
        )
    elif model_args.text_model_name_or_path and model_args.gex_model_name_or_path is None:
        print('random init GEX model!!!')
        model = GEXTextDualEncoderModel.from_gex_text_pretrained(
            text_model_name_or_path = model_args.text_model_name_or_path,
            gex_num_genes = len(ds['gene_list_order']),
            gex_vocab_size = len(ds['gene_list_order']) + 2,
        )
    else:
        raise ValueError('You must specify either model_name_or_path or both text_model_name_or_path and gex_model_name_or_path')
    
    config = model.config
    
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_gex_model:
        _freeze_params(model.gex_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)
    
    # set seed for torch dataloaders
    set_seed(training_args.seed)
    
    text_column_name = data_args.text_column_name
    gex_column_name = data_args.gex_column_name

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels_dict = ds['labels_dict']
    labels = list(labels_dict.keys())
    
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    uniq_train_label_list = list(set(ds["train"][text_column_name]))
    uniq_eval_label_list = list(set(ds["validation"][text_column_name]))
    uniq_test_label_list = []
    if "test" in ds:
        uniq_test_label_list = list(set(ds["test"][text_column_name]))
    #test_label_list = uniq_test_label_list
    #test_label_list = list(set(uniq_train_label_list + uniq_eval_label_list + uniq_test_label_list))
    #test_label_list = list(set(uniq_train_label_list + uniq_eval_label_list))
    test_label_list = labels
    print('Number of Train labels:', len(uniq_train_label_list))
    print('Number of Test labels:', len(test_label_list))
    pre_tokenized_text = {cellid:tokenizer(labels_dict[cellid], max_length=data_args.max_seq_length, padding="max_length", truncation=True) for cellid in labels}

    test_text_inputs = []
    test_labels = []
    for k, v in pre_tokenized_text.items():
        if k in test_label_list:
            test_text_inputs.append(v)
            test_labels.append(k)
    test_label2id, test_id2label = dict(), dict()
    for i, label in enumerate(test_labels):
        test_label2id[label] = str(i)
        test_id2label[str(i)] = label


    node2vec_emb = pd.read_csv('./datafiles/node2vec.emb', sep=' ', skiprows=1, header=None)
    node2vec_d = {}
    for i,r in node2vec_emb.iterrows():
        node2vec_d[r[0]] = np.array(r[1:], dtype=np.float32)
    node2vec_d['NOTFOUND'] = np.mean(np.array(list(node2vec_d.values())), 0)
    
    def preprocess_gex_text(example, mode='train'):

        if mode == 'train':
            text_inputs = [pre_tokenized_text[text] for text in example[text_column_name]]
        else:
            text_inputs = test_text_inputs

        example["input_ids"] = [torch.tensor([text_input.input_ids for text_input in text_inputs], dtype=torch.long)]
        example["attention_mask"] = [torch.tensor([text_input.attention_mask for text_input in text_inputs], dtype=torch.long)]

        example['gex'] = [torch.tensor(gex) for gex in example["gex"]]
        
        if mode == 'train':
            example['node2vec_emb'] = [torch.tensor(node2vec_d[label]) for label in example[text_column_name]]
            example['label'] = [int(label2id[label]) for label in example[text_column_name]]
        else:
            #example['node2vec_emb'] = [torch.tensor(node2vec_d.get(label, node2vec_d.get('NOTFOUND'))) for label in test_label2id.keys()]
            example['node2vec_emb'] = [torch.tensor(np.array([node2vec_d.get(label, node2vec_d.get('NOTFOUND')) for label in test_label2id.keys()]), dtype=torch.float32)]
            #example['label'] = [int(test_label2id[label]) for label in example[text_column_name]]
            example['label'] = [int(test_label2id.get(label, -1)) for label in example[text_column_name]]
        #print(example['input_ids'].shape)
        return example
    

    
    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(preprocess_gex_text)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(partial(preprocess_gex_text, mode='eval'))

    if training_args.do_predict:
        if "test" not in ds:
            raise ValueError("--do_predict requires a validation dataset")
        if data_args.max_predict_samples is not None:
            ds["test"] = (
                ds["test"].select(range(data_args.max_predict_samples))
            )
        # Set the validation transforms
        ds["test"].set_transform(partial(preprocess_gex_text, mode='predict'))
    ###todo dataset.map vs dataset.set_transform
    
    # encode gene_id to index
    vocabs = ['CLS', 'MASK'] + ds['gene_list_order']
    token2idx = {token:i for i,token in enumerate(vocabs)}
    gene_input_ids = [token2idx[geneid] for geneid in ds['gene_list_order']]
    
    # set wandb run name and project name
    if 'wandb' in training_args.report_to:
        import wandb
        wandb.init(project=data_args.project_name,
                name=training_args.run_name)
    
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        print(f'total number of predictions: {len(p.predictions[0])}')
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds_label = np.argmax(preds, axis=1)
        labels = p.label_ids
        print(f'preds_label: {preds_label.shape}')
        print(f'preds: {preds.shape}')
        print(f'labels: {len(labels)}')
        return {
            "top_1_accuracy": top_k_accuracy_score(y_true=labels, y_score=preds, labels= range(preds.shape[1]), k=1),
            "top_5_accuracy": top_k_accuracy_score(y_true=labels, y_score=preds, labels= range(preds.shape[1]), k=5)
            # "f1": f1_score(y_true=labels, y_pred=preds_label, average='micro'),
            # "recall": recall_score(y_true=labels, y_pred=preds_label, average='micro'),
            # "precision": precision_score(y_true=labels, y_pred=preds_label, average='micro')
        }
    
    print(model)
    # Initialize our trainer
    #trainer = CLIPTrainer(
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        data_collator=partial(collate_fn, gene_group=torch.tensor(ds['group']), gene_list=torch.tensor(gene_input_ids), test_label_num=len(test_label_list)),
        compute_metrics=compute_metrics,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    
     # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    with open(os.path.join(training_args.output_dir, 'labels.txt'), 'w') as f:
        f.write(f'label_id\tlabel_name\n')
        for k,v in labels_dict.items():
            f.write(f'{k}\t{v}\n')
    
    if training_args.do_predict:
        output = trainer.predict(ds['test'], metric_key_prefix="predict")
        
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)

        predictions = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions

        if trainer.is_world_process_zero():
            np.save(os.path.join(training_args.output_dir, f"predict_mat.npz"), predictions)
            np.save(os.path.join(training_args.output_dir, f"predict_mat_labels.npy"), np.array(list(test_id2label.values())))


        predictions = np.argmax(predictions, axis=1)
        predictions = [labels_dict[test_id2label[str(idx)]] for idx in predictions]
        labels = [labels_dict[test_id2label[str(idx)]] for idx in output.label_ids]

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, 'wt') as f:
                f.write(f'Gold\tPrediction\n')
                for i, p in enumerate(predictions):
                    f.write(f"{labels[i]}\t{p}\n")

if __name__ == "__main__":
    main()