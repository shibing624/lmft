# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from typing import Optional

from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm


@dataclass
class ModelArgs:
    adafactor_beta1: float = None
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
    adafactor_relative_step: bool = True
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = True
    adam_epsilon: float = 1e-8
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    config: dict = field(default_factory=dict)
    cosine_schedule_num_cycles: float = 0.5
    custom_layer_parameters: list = field(default_factory=list)
    custom_parameter_groups: list = field(default_factory=list)
    dataloader_num_workers: int = 0
    do_lower_case: bool = False
    dynamic_quantize: bool = False
    early_stopping_consider_epochs: bool = False
    early_stopping_delta: float = 0
    early_stopping_metric: str = "eval_loss"
    early_stopping_metric_minimize: bool = True
    early_stopping_patience: int = 3
    encoding: str = "utf-8"
    eval_batch_size: int = 8
    evaluate_during_training: bool = False
    evaluate_during_training_silent: bool = True
    evaluate_during_training_steps: int = 2000
    evaluate_during_training_verbose: bool = False
    evaluate_each_epoch: bool = True
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    local_rank: int = -1
    logging_steps: int = 50
    manual_seed: int = None
    max_grad_norm: float = 1.0
    max_seq_length: int = 384  # max length of input sequence
    model_name: str = None
    model_type: str = None
    multiprocessing_chunksize: int = -1
    n_gpu: int = 1
    no_cache: bool = False
    no_save: bool = False
    not_saved_args: list = field(default_factory=list)
    num_train_epochs: int = 1
    optimizer: str = "AdamW"
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = True
    polynomial_decay_schedule_lr_end: float = 1e-7
    polynomial_decay_schedule_power: float = 1.0
    process_count: int = 1
    quantized_model: bool = False
    reprocess_input_data: bool = False
    save_best_model: bool = True
    save_eval_checkpoints: bool = True
    save_model_every_epoch: bool = False
    save_optimizer_and_scheduler: bool = True
    save_steps: int = 1000
    scheduler: str = "linear_schedule_with_warmup"
    silent: bool = False
    skip_special_tokens: bool = True
    tensorboard_dir: str = None
    thread_count: int = None
    tokenizer_name: str = None
    tokenizer_type: str = None
    train_custom_parameters_only: bool = False
    use_cached_eval_features: bool = False
    use_early_stopping: bool = False
    use_hf_datasets: bool = False
    use_multiprocessing: bool = False
    use_multiprocessing_for_evaluation: bool = False
    wandb_kwargs: dict = field(default_factory=dict)
    wandb_project: str = None
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: float = 0.0

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {key: value for key, value in asdict(self).items() if key not in self.not_saved_args}
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w", encoding='utf-8') as f:
            args_dict = self.get_args_for_saving()
            if args_dict['dataset_class'] is not None and not isinstance(args_dict["dataset_class"], str):
                args_dict['dataset_class'] = type(args_dict['dataset_class']).__name__
            if args_dict["tokenizer_type"] is not None and not isinstance(args_dict["tokenizer_type"], str):
                args_dict["tokenizer_type"] = type(args_dict["tokenizer_type"]).__name__
            json.dump(args_dict, f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r", encoding='utf-8') as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


@dataclass
class ChatGLMArgs(ModelArgs):
    """
    Model args for a CopyT5Model
    """

    model_class: str = "ChatGLMArgs"
    dataset_class: Dataset = None
    debug: bool = False
    max_length = 384  # max length of the sequence to be generated
    do_sample: bool = True
    early_stopping: bool = True
    evaluate_generated_text: bool = False
    length_penalty: float = 2.0
    num_beams: int = 1
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 0.95
    special_tokens_list: list = field(default_factory=list)
    top_k: float = None
    top_p: float = 0.7
    model_name_or_path: Optional[str] = field(default="THUDM/chatglm-6b")
    dataset_name_or_path: Optional[str] = field(default="shibing624/alpaca-zh")
    use_lora: bool = True
    lora_name: str = field(default="adapter_model.bin")
    lora_rank: int = field(default=8)
    lora_alpha = 32
    lora_dropout = 0.1
    use_ppo: bool = False
    ppo_mini_batch_size = 16
    num_train_epochs = 1
    max_steps = -1
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 1
    save_total_limit = 2
    remove_unused_columns = False
    logging_steps = 50
    quantization_bit = None  # if use quantization bit, set 8 or 4, else None


def preprocess_data(data):
    instruction, input_text, target_text, tokenizer, args = data

    prompt = f"问：{instruction}\n"
    if input_text:
        prompt += f"{input_text}\n"
    prompt += "答："

    prompt_ids = tokenizer.encode(prompt, max_length=args.max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target_text, max_length=args.max_length, truncation=True,
                                  add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    input_ids = input_ids[:(args.max_seq_length + args.max_length)] + [tokenizer.eos_token_id]

    return input_ids


def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    data = (dataset["instruction"], dataset["input"], dataset["output"], tokenizer, args)
    dataset['input_ids'] = preprocess_data(data)
    return dataset


def load_hf_dataset(data, tokenizer, args):
    if isinstance(data, str):
        dataset = load_dataset(
            data,
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
        )
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        dataset = dataset["train"]
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args),
        batched=False,
    )

    dataset.set_format(type="np", columns=["input_ids"])

    return dataset["input_ids"]


class ChatGLMDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s" % cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s" % args.cache_dir)

            data = [
                (instruction, input_text, target_text, tokenizer, args)
                for instruction, input_text, target_text in zip(
                    data["instruction"], data["input"], data["output"]
                )
            ]

            if (mode == "train" and args.use_multiprocessing) or (
                    mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
