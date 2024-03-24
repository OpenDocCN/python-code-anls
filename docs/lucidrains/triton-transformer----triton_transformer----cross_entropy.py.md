# `.\lucidrains\triton-transformer\triton_transformer\cross_entropy.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 导入 triton 库
import triton
# 从 triton 库中导入 language 模块并重命名为 tl
import triton.language as tl

# 定义交叉熵损失函数，接受 logits（预测值）、labels（真实标签）、ignore_index（忽略的索引，默认为0）、use_triton（是否使用 triton 加速，默认为 False）
def cross_entropy_fn(logits, labels, ignore_index = 0., use_triton = False):
    # 重新排列 logits 张量的维度，将 'b n c' 转换为 '(b n) c'
    logits = rearrange(logits, 'b n c -> (b n) c')
    # 重新排列 labels 张量的维度，将 'b n' 转换为 '(b n)'
    labels = rearrange(labels, 'b n -> (b n)')

    # 如果 use_triton 为 True，则使用 triton 库中的 cross_entropy 函数计算损失
    if use_triton:
        loss = triton.ops.cross_entropy(logits, labels)        
    # 否则使用 torch.nn.functional 库中的 cross_entropy 函数计算损失
    else:
        loss = F.cross_entropy(logits, labels, reduction = 'none')

    # 创建一个掩码，标记 labels 中不等于 ignore_index 的位置
    mask = (labels != ignore_index)
    # 返回经过掩码处理后的损失的均值
    return loss[mask].mean()
```