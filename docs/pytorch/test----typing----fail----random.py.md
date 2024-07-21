# `.\pytorch\test\typing\fail\random.py`

```py
# 导入 torch 库，用于机器学习和深度学习任务
import torch

# 设置随机数生成器的状态，这里存在类型不匹配的问题
# set_rng_state 方法期望的参数类型是 Tensor，但实际传入的是一个包含整数的列表
torch.set_rng_state(
    [
        1,  # 第一个元素
        2,  # 第二个元素
        3,  # 第三个元素
    ]
)
```