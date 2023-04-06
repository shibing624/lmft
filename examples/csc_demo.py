# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

sys.path.append('..')

model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto')
model = PeftModel.from_pretrained(model, "shibing624/chatglm-6b-csc-zh-lora")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
if torch.cuda.is_available():
    model = model.half().cuda()
else:
    model = model.quantize(bits=4, compile_parallel_kernel=True, parallel_num=2).cpu().float()

sents = ['对下面中文拼写纠错：\n少先队员因该为老人让坐。\n答：',
         '对下面中文拼写纠错：\n下个星期，我跟我朋唷打算去法国玩儿。\n答：']
for s in sents:
    response = model.chat(tokenizer, s, max_length=128, eos_token_id=tokenizer.eos_token_id)
    print(response)
