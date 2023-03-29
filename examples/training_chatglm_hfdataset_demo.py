# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import argparse
from loguru import logger

sys.path.append('..')
from lmft import ChatGLMTune


def finetune_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='shibing624/alpaca-zh', type=str,
                        help='Datasets name, eg: tatsu-lab/alpaca')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=256, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=256, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    # fine-tune chatGLM model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            'use_lora': True,
            "reprocess_input_data": False,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "output_dir": args.output_dir,
            "use_hf_datasets": True
        }
        model = ChatGLMTune(args.model_type, args.model_name, args=model_args)

        model.train_model(args.train_file)
    if args.do_predict:
        model = ChatGLMTune(args.model_type, args.model_name,
                            args={'use_lora': True, 'eval_batch_size': args.batch_size})
        response, history = model.chat("给出三个保持健康的秘诀。", history=[])
        print(response)
        response, history = model.chat("描述原子的结构。", history=history)
        print(response)
        del model

        ref_model = ChatGLMTune(args.model_type, args.model_name,
                                args={'use_lora': False, 'eval_batch_size': args.batch_size})
        response, history = ref_model.chat("给出三个保持健康的秘诀。", history=[])
        print(response)
        response, history = ref_model.chat("描述原子的结构。", history=history)
        print(response)


if __name__ == '__main__':
    finetune_demo()
