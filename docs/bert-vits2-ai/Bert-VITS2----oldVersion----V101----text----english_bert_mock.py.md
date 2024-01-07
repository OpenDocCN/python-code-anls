# `Bert-VITS2\oldVersion\V101\text\english_bert_mock.py`

```

# 导入 torch 库
import torch

# 根据规范化文本和单词到短语的映射，获取 BERT 特征
def get_bert_feature(norm_text, word2ph):
    # 返回一个大小为 1024x(sum(word2ph)) 的全零张量
    return torch.zeros(1024, sum(word2ph))

```