# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""

import gradio as gr
import os
import sys
import argparse
from loguru import logger

sys.path.append('..')
from lmft import ChatGlmModel

pwd_path = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str, help='Transformers model or path')
parser.add_argument('--output_dir', default='./outputs-csc/', type=str, help='Model output directory')
parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
args = parser.parse_args()
logger.info(args)

model = ChatGlmModel(
    args.model_type, args.model_name,
    args={'use_lora': True, 'eval_batch_size': args.batch_size,
          'output_dir': args.output_dir, "max_length": args.max_length, }
)
sents = ['问：对下面中文拼写纠错：\n少先队员因该为老人让坐。\n答：',
         '问：对下面中文拼写纠错：\n下个星期，我跟我朋唷打算去法国玩儿。\n答：']


def batch_correct(sentences):
    prompts = [f"问：对下面中文拼写纠错：\n{s}\n答：" for s in sentences]
    r = model.predict(prompts)
    return [s.split('\n')[0] for s in r]


response = batch_correct(sents)
print(response)


def ai_text(text):
    return batch_correct([text])[0]


if __name__ == '__main__':
    examples = [
        ['真麻烦你了。希望你们好好的跳无'],
        ['少先队员因该为老人让坐'],
        ['机七学习是人工智能领遇最能体现智能的一个分知'],
        ['今天心情很好'],
        ['他法语说的很好，的语也不错'],
        ['他们的吵翻很不错，再说他们做的咖喱鸡也好吃'],
    ]

    gr.Interface(
        ai_text,
        inputs="textbox",
        outputs=[gr.outputs.Textbox()],
        title="LMFT Model shibing624/lmft",
        description="Copy or input error Chinese question.",
        article="Link to <a href='https://github.com/shibing624/lmft' style='color:blue;' target='_blank\'>Github REPO</a>",
        examples=examples
    ).launch()
