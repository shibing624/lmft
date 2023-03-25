# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(default="THUDM/chatglm-6b")
    dataset_name: Optional[str] = field(
        default="shibing624/alpaca-zh", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    lora_name: str = field(default="chatglm6b-lora.pt")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
        seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
            seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1:] = mask_position
    return attention_mask, position_ids


def build_dataset(model_name, dataset_name="shibing624/alpaca-zh", max_seq_length=512):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"output": "target"})
    ds = ds.filter(lambda x: len(x["target"]) > 2, batched=False)

    def tokenize(example):
        prompt = f"Instruction: {example['instruction']}\n"
        if example.get("input", ""):
            prompt += f"Input: {example['input']}\n"
        prompt += "Answer: "
        example['prompt'] = prompt
        prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
        target_ids = tokenizer.encode(example["target"], max_length=max_seq_length, truncation=True,
                                      add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        example["input_ids"] = input_ids[:max_seq_length]
        example["seq_len"] = len(prompt_ids)
        return example

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def data_collator(batch, tokenizer) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in batch]
    longest = max(len_ids)
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, batch), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
                [-100] * (seq_len - 1)
                + ids[(seq_len - 1):]
                + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        attention_mask, position_ids = get_masks_and_position_ids(
            ids, seq_len, longest, _ids.device, gmask=False
        )
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


class FinetuneTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False, lora_name='lora.pt'):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, lora_name))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)
