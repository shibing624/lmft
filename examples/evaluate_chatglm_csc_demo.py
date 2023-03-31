# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install pycorrector
"""
import os
import sys
import argparse
from loguru import logger
from pycorrector.utils import eval

sys.path.append('..')
from lmft import ChatGLMTune

pwd_path = os.path.abspath(os.path.dirname(__file__))


def evaluate_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--output_dir', default='./outputs-csc/', type=str, help='Model output directory')
    parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = ChatGLMTune(
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

    eval.eval_sighan2015_by_model_batch(batch_correct, os.path.join(pwd_path, 'data/test.tsv'))
    # Sentence Level: acc:0.3864, precision:0.3263, recall:0.2284, f1:0.2687, cost time:288.23 s, total num: 1100
    # 虽然F1值远低于macbert4csc(f1:0.7742)等模型，但这个纠错结果带句子润色的效果，看结果case大多数是比ground truth结果句子更通顺流畅，
    # 我觉得是当前效果最好的纠错模型之一，比较像ChatGPT效果。


if __name__ == '__main__':
    evaluate_demo()
