# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer
from transformers import TrainingArguments
from transformers.generation.utils import LogitsProcessorList
from transformers.trainer import TRAINING_ARGS_NAME

from .chatglm_utils import (
    ChatGLMForConditionalGeneration,
    ChatGLMArgs,
    InvalidScoreLogitsProcessor,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

has_cuda = torch.cuda.is_available()
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_CLASSES = {
    "chatglm": (AutoConfig, ChatGLMForConditionalGeneration, AutoTokenizer),
}


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)


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


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class ChatGLMTune:
    def __init__(
            self,
            model_type,
            model_name,
            args=None,
            use_cuda=has_cuda,
            cuda_device=-1,
            **kwargs,
    ):

        """
        Initializes a ChatGLMModel model.

        Args:
            model_type: The type of model (chatglm)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"
        model_type = model_type.lower()
        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ChatGLMArgs):
            self.args = args

        self.is_sweeping = False
        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"
        logger.debug(f"Device: {self.device}")

        self.results = {}
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if model_name is None:
            model_name = "THUDM/chatglm-6b"
        if torch.cuda.is_available():
            self.model = model_class.from_pretrained(model_name,
                                                     trust_remote_code=True).half().cuda()
        else:
            self.model = model_class.from_pretrained(model_name, trust_remote_code=True).float()

        self.tokenizer_class = tokenizer_class
        if self.args.tokenizer_name:
            self.tokenizer = tokenizer_class.from_pretrained(self.args.tokenizer_name, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer_class.from_pretrained(model_name, trust_remote_code=True, **kwargs)
            self.args.tokenizer_name = self.args.model_name

        if not use_cuda:
            self.args.fp16 = False

        self.args.model_type = model_type
        if model_name is None:
            self.args.model_name = "ChatGLM_from_scratch"
        else:
            self.args.model_name = model_name
        self.lora_loaded = False

    @staticmethod
    def get_masks_and_position_ids(seq_len, context_length, device, gmask=False, position_encoding_2d=True):
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

    def build_dataset(self, dataset_name_or_path="shibing624/alpaca-zh", max_seq_length=512):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.

        Args:
            dataset_name_or_path (`str`):
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        # load datasets
        if os.path.exists(dataset_name_or_path):
            ds = load_dataset("json", data_files=dataset_name_or_path)
            ds = ds['train']
        else:
            ds = load_dataset(dataset_name_or_path, split="train")
        if self.args.debug:
            ds = ds.select(range(20))
        ds = ds.rename_columns({"output": "target"})
        ds = ds.filter(lambda x: len(x["target"]) > 0, batched=False)

        def tokenize(example):
            prompt = f"Instruction: {example['instruction']}\n"
            if example.get("input", ""):
                prompt += f"Input: {example['input']}\n"
            prompt += "Answer: "
            example['prompt'] = prompt
            prompt_ids = self.tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
            target_ids = self.tokenizer.encode(example["target"], max_length=max_seq_length, truncation=True,
                                               add_special_tokens=False)
            input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
            example["input_ids"] = input_ids[:max_seq_length]
            example["seq_len"] = len(prompt_ids)
            return example

        ds = ds.map(tokenize, batched=False)
        return ds

    def data_collator(self, batch):
        len_ids = [len(example["input_ids"]) for example in batch]
        longest = max(len_ids)
        input_ids = []
        attention_mask_list = []
        position_ids_list = []
        labels_list = []
        for ids_l, feature in sorted(zip(len_ids, batch), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            seq_len = ids.index(self.tokenizer.bos_token_id)
            label_pad_token_id = -100
            labels = (
                    [label_pad_token_id] * (seq_len - 1)
                    + ids[(seq_len - 1):]
                    + [label_pad_token_id] * (longest - ids_l)
            )
            ids = ids + [self.tokenizer.pad_token_id] * (longest - ids_l)
            _ids = torch.LongTensor(ids)
            attention_mask, position_ids = self.get_masks_and_position_ids(
                seq_len, longest, _ids.device, gmask=False
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

    def train_model(
            self,
            train_data,
            output_dir=None,
            args=None,
            eval_data=None,
            verbose=True,
            **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: datasets Dataset object or path to file containing training data
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)
        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if (
                os.path.exists(output_dir)
                and os.listdir(output_dir)
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )
        # update model train config
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.is_parallelizable = True
        self.model.model_parallel = True
        self.model.lm_head = CastOutputToFloat(self.model.lm_head)
        self.model.config.use_cache = False

        # setup peft, add lora config
        if self.args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.lora_loaded = True
        self._move_model_to_device()
        os.makedirs(output_dir, exist_ok=True)

        # load dataset
        train_dataset = self.build_dataset(train_data, max_seq_length=self.args.max_seq_length)
        logger.debug(f"dataset: {train_dataset} first row: {next(iter(train_dataset))}")

        # start train
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            auto_find_batch_size=True,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            logging_dir=f"{self.args.output_dir}/logs",
            logging_steps=self.args.logging_steps,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            fp16=self.args.fp16,
            remove_unused_columns=self.args.remove_unused_columns,
            overwrite_output_dir=self.args.overwrite_output_dir,
            do_train=True,
        )
        logger.debug(f"training_args: {training_args}")
        trainer = FinetuneTrainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        trainer.train()

        self.save_model(model=self.model)

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_name, output_dir
                )
            )

    def load_lora(self):
        if self.args.use_lora:
            lora_path = os.path.join(self.args.output_dir, self.args.lora_name)
            if lora_path and os.path.exists(lora_path):
                # infer with trained lora model
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=True,
                    r=self.args.lora_rank,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                )
                self.model = get_peft_model(self.model, peft_config)
                self.model.load_state_dict(torch.load(lora_path), strict=False)
                logger.info(f"Loaded lora model from {lora_path}")
                self.lora_loaded = True

    @torch.no_grad()
    def chat(self, query: str, history: List[Tuple[str, str]] = None, logits_processor=None, **kwargs):
        """
        Chat with the model
        :param query:
        :param history:
        :param logits_processor:
        :param kwargs:
        :return: response, history
        """
        self._move_model_to_device()
        self.model.eval()
        if history is None:
            history = []
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        response = self.predict([prompt], logits_processor=logits_processor, **kwargs)[0]
        response = response.strip()
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def predict(self, sentences, logits_processor=None, **kwargs):
        """
        Performs predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. 
            logits_processor: A LogitsProcessor object that will be applied to the model's

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        if not self.lora_loaded:
            self.load_lora()
        self._move_model_to_device()
        if torch.cuda.is_available() and self.args.fp16:
            self.model = self.model.half().cuda()
        self.model.eval()

        all_outputs = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        # Batching
        for batch in tqdm(
                [
                    sentences[i: i + self.args.eval_batch_size]
                    for i in range(0, len(sentences), self.args.eval_batch_size)
                ],
                desc="Generating outputs",
                disable=self.args.silent,
        ):
            inputs = self.tokenizer(batch, padding=True, return_tensors='pt').to(self.device)
            gen_kwargs = {
                "max_length": self.args.max_length,
                "num_beams": self.args.num_beams,
                "do_sample": self.args.do_sample,
                "top_p": self.args.top_p,
                "temperature": self.args.temperature,
                "logits_processor": logits_processor,
                **kwargs
            }
            outputs = self.model.generate(**inputs, **gen_kwargs)
            all_outputs.extend(outputs.cpu().numpy())
        outputs = [self.tokenizer.decode(output_id) for output_id in all_outputs]
        return outputs

    def _move_model_to_device(self):
        self.model.to(self.device)

    def save_model(
            self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            # save model
            save_tunable_parameters(
                self.model, os.path.join(self.args.output_dir, self.args.lora_name)
            )

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = ChatGLMArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
