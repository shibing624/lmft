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


**lmft**实现了ChatGLM-6B的模型finetune。


**Guide**
- [Feature](#Feature)
- [Evaluation](#Evaluation)
- [Demo](#Demo)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Reference](#reference)


# Feature
### ChatGPT-6B fine-tuning
- [Word2Vec](lmft/word2vec.py)：通过腾讯AI Lab开源的大规模高质量中文[词向量数据（800万中文词轻量版）](https://pan.baidu.com/s/1La4U4XNFe8s5BJqxPQpeiQ) (文件名：light_Tencent_AILab_ChineseEmbedding.bin 密码: tawe）实现词向量检索，本项目实现了句子（词向量求平均）的word2vec向量表示
- [SBERT(Sentence-BERT)](lmft/sentencebert_model.py)：权衡性能和效率的句向量表示模型，训练时通过有监督训练上层分类函数，文本匹配预测时直接句子向量做余弦，本项目基于PyTorch复现了Sentence-BERT模型的训练和预测
- [CoSENT(Cosine Sentence)](lmft/cosent_model.py)：CoSENT模型提出了一种排序的损失函数，使训练过程更贴近预测，模型收敛速度和效果比Sentence-BERT更好，本项目基于PyTorch实现了CoSENT模型的训练和预测

# Evaluation

### 文本生成

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

## 文本生成

example: [examples/computing_embeddings_demo.py](examples/computing_embeddings_demo.py)

```python
import sys

sys.path.append('..')
from lmft import ChatGpt


def compute_emb(model):
    # Embed a list of sentences
    sentences = [
        '卡',
        '银行卡',
        'The quick brown fox jumps over the lazy dog.'
    ]
    sentence_embeddings = model.encode(sentences)
    print(type(sentence_embeddings), sentence_embeddings.shape)

    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding shape:", embedding.shape)
        print("Embedding head:", embedding[:10])
        print()


if __name__ == "__main__":
    t2v_model = ChatGpt("shibing624/lmft-base-chinese")
    compute_emb(t2v_model)
```

output:
```
<class 'numpy.ndarray'> (7, 768)
Sentence: 卡
Embedding shape: (768,)

Sentence: 银行卡
Embedding shape: (768,)
 ... 
```


#### dataset
1. [0.5M生成的中文ChatGPT结果数据](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)
2. [50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)


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


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加lmft的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest -v`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

# Reference
- [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
- [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [huggingface/peft](https://github.com/huggingface/peft)
