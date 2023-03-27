# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from lmft import ChatGLMTune


def finetune_demo():
    m = ChatGLMTune('chatglm', "THUDM/chatglm-6b", args={'use_lora': True})
    m.train_model(train_data='shibing624/alpaca-zh')
    r = m.predict(['你是谁', '三原色是啥'])
    print(r)
    response, history = m.chat("你好", history=[])
    print(response)
    response, history = m.chat("晚上睡不着应该怎么办", history=history)
    print(response)


def origin_chat_demo():
    m = ChatGLMTune('chatglm', "THUDM/chatglm-6b", args={'use_lora': False})
    response, history = m.chat("你好", history=[])
    print(response)
    response, history = m.chat("晚上睡不着应该怎么办", history=history)
    print(response)


if __name__ == '__main__':
    origin_chat_demo()
    finetune_demo()
