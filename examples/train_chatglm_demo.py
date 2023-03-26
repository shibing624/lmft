# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import argparse

sys.path.append('..')
from lmft.chatglm_model import ChatGLMTune


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="THUDM/chatglm-6b", type=str)
    parser.add_argument("--train_data", default='shibing624/alpaca-zh', type=str)

    args = parser.parse_args()
    m = ChatGLMTune('chatglm', args.model_name)
    m.train_model(args.train_data)
    m.predict(['你是谁', '三原色是啥'])


if __name__ == '__main__':
    main()
