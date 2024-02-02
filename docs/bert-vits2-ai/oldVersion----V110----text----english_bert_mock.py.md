# `Bert-VITS2\oldVersion\V110\text\english_bert_mock.py`

```py
# 导入 torch 模块
import torch

# 定义函数 get_bert_feature，接受两个参数 norm_text 和 word2ph
def get_bert_feature(norm_text, word2ph):
    # 返回一个大小为 (1024, sum(word2ph)) 的全零张量
    return torch.zeros(1024, sum(word2ph))
```