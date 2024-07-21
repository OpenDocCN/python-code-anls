# `.\pytorch\benchmarks\distributed\rpc\parameter_server\trainer\criterions.py`

```py
# 导入 PyTorch 的神经网络模块中的 nn 类
import torch.nn as nn

# 定义一个函数 cel，用于创建一个交叉熵损失函数（CrossEntropyLoss）的准则，用于训练
def cel(rank):
    r"""A function that creates a CrossEntropyLoss
    criterion for training.
    Args:
        rank (int): worker rank
    """
    # 创建一个在 GPU 上运行的交叉熵损失函数（CrossEntropyLoss）对象，并返回
    return nn.CrossEntropyLoss().cuda(rank)
```