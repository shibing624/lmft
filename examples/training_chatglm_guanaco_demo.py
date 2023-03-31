# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 在150万对话数据上训练ChatGLM模型
"""
import sys
import argparse
from loguru import logger
from datasets import load_dataset
from torch.utils.data import Dataset

sys.path.append('..')
from lmft import ChatGLMTune


def preprocess_batch_for_hf_dataset(example, tokenizer, args):
    instruction, input_text, target_text = example["instruction"], example["input"], example["output"]
    prompt = f"问：{instruction}\n"
    if input_text:
        prompt += f"{input_text}\n"
    prompt += "答："
    prompt_ids = tokenizer.encode(prompt, max_length=args.max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target_text, max_length=args.max_length, truncation=True,
                                  add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    input_ids = input_ids[:(args.max_seq_length + args.max_length)] + [tokenizer.eos_token_id]

    example['input_ids'] = input_ids
    return example


class GuanacoDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        dataset = load_dataset(data)
        dataset = dataset["train"]
        dataset = dataset.map(
            lambda x: preprocess_batch_for_hf_dataset(x, tokenizer, args),
            batched=False,
        )
        dataset.set_format(type="np", columns=["input_ids"])

        self.examples = dataset["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def finetune_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="Chinese-Vicuna/guanaco_belle_merge_v1.0", type=str,
                        help='Datasets name, such as Chinese-Vicuna/guanaco_belle_merge_v1.0')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs-guanaco/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=256, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=256, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune chatGLM model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "dataset_class": GuanacoDataset,
            'use_lora': True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "output_dir": args.output_dir,
        }
        model = ChatGLMTune(args.model_type, args.model_name, args=model_args)

        model.train_model(args.train_file)
    if args.do_predict:
        if model is None:
            model = ChatGLMTune(
                args.model_type, args.model_name,
                args={'use_lora': True, 'eval_batch_size': args.batch_size,
                      'output_dir': args.output_dir, "max_length": args.max_length, }
            )
        sents = [
            '问：用一句话描述地球为什么是独一无二的。\n答：',
            '问：给定两个数字，计算它们的平均值。 数字: 25, 36\n答：',
            '问：基于以下提示填写以下句子的空格。 提示： - 提供多种现实世界的场景 - 空格应填写一个形容词或一个形容词短语 句子: ______出去享受户外活动，包括在公园里散步，穿过树林或在海岸边散步。\n答：',
        ]
        response = model.predict(sents)
        print(response)


if __name__ == '__main__':
    finetune_demo()
