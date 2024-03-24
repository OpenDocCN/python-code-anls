# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\utils.py`

```
# 从 torch 模块中导入 nn 模块

from torch import nn

# 定义函数

def round_down_nearest_multiple(num, divisor):
    # 返回最接近 num 且能被 divisor 整除的数
    return num // divisor * divisor

def curtail_to_multiple(t, mult, from_left = False):
    # 获取输入张量的最后一个维度的长度
    data_len = t.shape[-1]
    # 将长度舍入到最接近的 mult 的倍数
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    # 根据 from_left 参数选择截取的方式
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    # 返回截取后的张量
    return t[..., seq_slice]

# 基类

class AudioConditionerBase(nn.Module):
    # 空的类，用于继承 nn.Module
    pass
```