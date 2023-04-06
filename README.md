[![PyPI version](https://badge.fury.io/py/lmft.svg)](https://badge.fury.io/py/lmft)
[![Downloads](https://pepy.tech/badge/lmft)](https://pepy.tech/project/lmft)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/lmft.svg)](https://github.com/shibing624/lmft/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/lmft.svg)](https://github.com/shibing624/lmft/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# LMFT: Language Model Fine-Tuning
Language Model Fine-Tuning, for ChatGLM, BELLE, LLaMA fine-tuning.


**lmft**实现了ChatGLM-6B的模型FineTune。


**Guide**
- [Feature](#Feature)
- [Evaluation](#Evaluation)
- [Demo](#Demo)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Reference](#reference)


# Feature
### ChatGLM
#### [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b) 模型的Finetune训练

[THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)放出的默认模型，模型以 FP16 精度加载，模型运行需要 13GB 显存，训练需要 22GB 显存(batch_size=2)。


# Evaluation

### 纠错能力比较

### 对话能力比较

# Demo

HuggingFace Demo: https://huggingface.co/spaces/shibing624/lmft

![](docs/hf.png)

run example: [examples/gradio_demo.py](examples/gradio_demo.py) to see the demo:
```shell
python examples/gradio_demo.py
```

# Install
```shell
pip install -U lmft
```

or

```shell
pip install -r requirements.txt

git clone https://github.com/shibing624/lmft.git
cd lmft
pip install --no-deps .
```

# Usage
## Use LoRA model
release lora model: 
1. 中文拼写纠错（CSC）模型 [shibing624/chatglm-6b-csc-zh-lora](https://huggingface.co/shibing624/chatglm-6b-csc-zh-lora)


```python
from lmft import ChatGlmModel
model = ChatGlmModel("chatglm", "THUDM/chatglm-6b", lora_name="shibing624/chatglm-6b-csc-zh-lora")
r = model.predict(["对下面中文拼写纠错：\n少先队员因该为老人让坐。\n答："])
print(r) # ['少先队员应该为老人让座。\n错误字：因，坐']
```

## 训练ChatGLM-6B模型

支持自定义数据集，数据集格式参考[examples/data/test.tsv](examples/data/test.tsv)。


example: [examples/training_chatglm_demo.py](examples/training_chatglm_demo.py)

```python
import sys

sys.path.append('..')
from lmft import ChatGlmModel


def finetune_demo():
    m = ChatGlmModel('chatglm', "THUDM/chatglm-6b", args={'use_lora': True})
    m.train_model(train_data='shibing624/alpaca-zh')
    r = m.predict(['给出三个保持健康的秘诀。', '描述原子的结构。'])
    print(r)
    response, history = m.chat("你好", history=[])
    print(response)
    response, history = m.chat("晚上睡不着应该怎么办", history=history)
    print(response)


def origin_chat_demo():
    m = ChatGlmModel('chatglm', "THUDM/chatglm-6b", args={'use_lora': False})
    response, history = m.chat("你好", history=[])
    print(response)
    response, history = m.chat("晚上睡不着应该怎么办", history=history)
    print(response)


if __name__ == '__main__':
    origin_chat_demo()
    finetune_demo()
```

output:
```
问:你好
答:你好

[Round 1]
问:晚上睡不着应该怎么办
答: 想要在晚上入睡,但并不容易,可以参考下述技巧:
1. 睡前放松:尝试进行一些放松的活动,如冥想、深呼吸或瑜伽,帮助放松身心,减轻压力和焦虑。
2. 创造一个舒适的睡眠环境:保持房间安静、黑暗和凉爽,使用舒适的床垫和枕头,确保床铺干净整洁。
3. 规律的睡眠时间表:保持规律的睡眠时间表,尽可能在同一时间上床,并创造一个固定的起床时间。
4. 避免刺激性食物和饮料:避免在睡前饮用含咖啡因的饮料,如咖啡、茶和可乐,以及吃辛辣、油腻或难以消化的食物。
5. 避免过度使用电子设备:避免在睡前使用电子设备,如手机、电视和电脑。这些设备会发出蓝光,干扰睡眠。
如果尝试了这些技巧仍然无法入睡,建议咨询医生或睡眠专家,获取更专业的建议和帮助。
```


## Dataset
1. 50万条中文ChatGPT指令数据集：[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
2. 100万条中文ChatGPT指令数据集：[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
3. 5万条英文ChatGPT指令数据集：[50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)
4. 2万条中文ChatGPT指令数据集：[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
5. 69万条中文指令数据集(Belle50万条+Guanaco19万条)：[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)

## FAQ
1. 问：为啥没有`int4`量化模型的Finetune训练？

答：THUDM放出了2个int4量化模型，分别是 [THUDM/chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4) 和 
[THUDM/chatglm-6b-int4-qe](https://huggingface.co/THUDM/chatglm-6b-int4-qe) 模型，是基于
[THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的int4并对Embedding量化后的模型，模型运行仅需要 4.3GB 显存。

训练方法参考官方给出的[P-tuning方法](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md)，INT4 量化模型的训练最低只需 6.7G 显存。


# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/lmft.svg)](https://github.com/shibing624/lmft/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了lmft，请按如下格式引用：

APA:
```latex
Xu, M. lmft: Lanauge Model Fine-Tuning toolkit (Version 1.1.2) [Computer software]. https://github.com/shibing624/lmft
```

BibTeX:
```latex
@misc{lmft,
  author = {Xu, Ming},
  title = {lmft: Language Model Fine-Tuning toolkit},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shibing624/lmft}},
}
```

# License


lmft授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加lmft的链接和授权协议。

- ChatGLM-6B的模型权重仅限学术研究用，具体见[MODEL_LICENSE](https://github.com/THUDM/ChatGLM-6B/blob/main/MODEL_LICENSE)
- LLAMA的模型权重仅限学术研究用，具体见[LICENSE](https://huggingface.co/decapoda-research/llama-13b-hf/blob/main/LICENSE)

# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest -v`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

# Reference
- [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
- [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [huggingface/peft](https://github.com/huggingface/peft)
