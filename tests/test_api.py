# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import pytest

sys.path.append('..')
from lmft import ChatGLMTune


def test_get_default_key():
    m = ChatGLMTune('chatglm', "THUDM/chatglm-6b", args={'use_lora': False})
    response, history = m.chat("你好", history=[])
    print(response)
    assert len(response) > 0
    response, history = m.chat("晚上睡不着应该怎么办", history=history)
    print(response)
    assert len(response) > 0


test_get_default_key()
