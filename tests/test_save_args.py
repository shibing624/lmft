# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import pytest
import os

sys.path.append('..')
from lmft.chatglm_utils import ChatGLMArgs


def test_save_args():
    args = ChatGLMArgs()
    os.makedirs('outputs/', exist_ok=True)
    print('old', args)
    args.save('outputs/')
    args.load('outputs/')
    print('new', args)


test_save_args()
