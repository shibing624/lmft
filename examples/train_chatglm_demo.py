# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from lmft.chatglm_model import ChatGLMTune


def main():
    m = ChatGLMTune('chatglm', "THUDM/chatglm-6b", args={'use_lora': True})
    m.train_model(train_data='shibing624/alpaca-zh')
    r = m.predict(['你是谁', '三原色是啥'])
    print(r)


if __name__ == '__main__':
    main()
