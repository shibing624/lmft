# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import inspect
import os
import warnings
from contextlib import contextmanager

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin
from peft import PeftConfig, PromptLearningConfig, PromptEncoder, LoraModel, PeftType
