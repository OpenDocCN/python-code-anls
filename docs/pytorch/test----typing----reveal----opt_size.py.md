# `.\pytorch\test\typing\reveal\opt_size.py`

```py
# 导入 PyTorch 库
import torch

# 创建一个自适应平均池化层，指定输出大小为 (1, None)，其中 None 表示该维度大小不变
avg_pool1 = torch.nn.AdaptiveAvgPool2d((1, None))
# 显示 avg_pool1 的类型信息
reveal_type(avg_pool1)  # E: {AdaptiveAvgPool2d}

# 创建一个自适应平均池化层，指定输出大小为 (None, 1)，其中 None 表示该维度大小不变
avg_pool2 = torch.nn.AdaptiveAvgPool2d((None, 1))
# 显示 avg_pool2 的类型信息
reveal_type(avg_pool2)  # E: {AdaptiveAvgPool2d}

# 创建一个自适应最大池化层，指定输出大小为 (1, None)，其中 None 表示该维度大小不变
max_pool1 = torch.nn.AdaptiveMaxPool2d((1, None))
# 显示 max_pool1 的类型信息
reveal_type(max_pool1)  # E: {AdaptiveMaxPool2d}

# 创建一个自适应最大池化层，指定输出大小为 (None, 1)，其中 None 表示该维度大小不变
max_pool2 = torch.nn.AdaptiveMaxPool2d((None, 1))
# 显示 max_pool2 的类型信息
reveal_type(max_pool2)  # E: {AdaptiveMaxPool2d}
```