# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

import datasets
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer
from transformers import TrainingArguments, HfArgumentParser

sys.path.append('..')
from lmft.chatglm_tune import ModelArguments, CastOutputToFloat, ModifiedTrainer, data_collator, save_tunable_parameters
from lmft.modeling_chatglm import ChatGLMForConditionalGeneration


def main():
    model_args, training_args = HfArgumentParser(
        (ModelArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, device_map="auto"
    ).half().cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (False)

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    # load dataset
    ds = datasets.load_dataset(model_args.dataset_name, split="train")
    # ds = datasets.load_from_disk(model_args.dataset_name)

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=ds,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, model_args.lora_name)
    )


if __name__ == "__main__":
    main()
