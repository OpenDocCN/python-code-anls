# `.\pytorch\torch\ao\pruning\_experimental\pruner\lstm_saliency_pruner.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
from typing import cast  # 从 typing 模块导入 cast 函数，用于类型转换

import torch  # 导入 PyTorch 库
from .base_structured_sparsifier import BaseStructuredSparsifier, FakeStructuredSparsity  # 导入基类和虚拟结构稀疏化类

class LSTMSaliencyPruner(BaseStructuredSparsifier):
    """
    Prune packed LSTM weights based on saliency.
    For each layer {k} inside a LSTM, we have two packed weight matrices
    - weight_ih_l{k}
    - weight_hh_l{k}

    These tensors pack the weights for the 4 linear layers together for efficiency.

    [W_ii | W_if | W_ig | W_io]

    Pruning this tensor directly will lead to weights being misassigned when unpacked.
    To ensure that each packed linear layer is pruned the same amount:
        1. We split the packed weight into the 4 constituent linear parts
        2. Update the mask for each individual piece using saliency individually

    This applies to both weight_ih_l{k} and weight_hh_l{k}.
    """

    def update_mask(self, module, tensor_name, **kwargs):
        # 获取模块中指定张量的权重
        weights = getattr(module, tensor_name)

        # 遍历参数化对象列表，找到 FakeStructuredSparsity 类型的对象
        for p in getattr(module.parametrizations, tensor_name):
            if isinstance(p, FakeStructuredSparsity):
                # 获取掩码（mask）并将其转换为 torch.Tensor 类型
                mask = cast(torch.Tensor, p.mask)

                # 根据权重的维度数量进行条件判断
                if weights.dim() <= 1:
                    raise Exception("Structured pruning can only be applied to a 2+dim weight tensor!")  # noqa: TRY002
                
                # 计算除第一个维度外的所有维度上的 L1 范数，作为权重的显著性指标
                dims = tuple(range(1, weights.dim()))
                saliency = weights.norm(dim=dims, p=1)

                # 将掩码按照预设的分组数量进行分割
                split_size = len(mask) // 4
                masks = torch.split(mask, split_size)
                saliencies = torch.split(saliency, split_size)

                # 遍历分组掩码和对应的显著性指标
                for keep_mask, sal in zip(masks, saliencies):
                    # 计算要保留的最小 k 个值，并将对应位置在掩码中设为 False，实现剪枝
                    k = int(len(keep_mask) * kwargs["sparsity_level"])
                    prune = sal.topk(k, largest=False, sorted=False).indices
                    keep_mask.data[prune] = False  # 直接修改底层的 p.mask 数据
```