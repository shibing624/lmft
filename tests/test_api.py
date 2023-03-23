# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import pytest

sys.path.append('..')



def test_error_key():
    with pytest.raises(ValueError, match="openai api key is invalid, please check it."):
        ChatGPTBot(openai_api_key='aaa')


def test_get_default_key():
    a = '英汉互译: data science'
    m = ChatGPTBot()
    r = m.reply(a)
    print(r)
    assert len(r) > 0 and '数据科学' in r


def test_get_image_from_text():
    a = '画个猫'
    m = ChatGPTBot()
    r = m.reply(a)
    print(r)
    assert len(r) > 0


def test_get_image_from_image():
    a = '画2个猫再一起玩球'
    context = {'session_id': 'UserName2', 'type': 'IMAGE_CREATE'}
    m = ChatGPTBot()
    r = m.reply(a, context)
    print(r)
    assert len(r) > 0 and 'http' in r


test_error_key()
test_get_default_key()
test_get_image_from_text()
test_get_image_from_image()
