# `.\pytorch\torch\ao\pruning\_experimental\pruner\saliency_pruner.py`

```
# 引入未经类型定义的函数和方法允许
# 从当前包中导入 BaseStructuredSparsifier 类
from .base_structured_sparsifier import BaseStructuredSparsifier

class SaliencyPruner(BaseStructuredSparsifier):
    """
    根据行的显著性（L1 范数）修剪权重。

    这个修剪器适用于 N 维权重张量。
    对于每一行，我们计算其显著性，即该行所有权重的 L1 范数之和。
    我们期望得到的显著性向量与我们的掩码具有相同的形状。
    然后，我们选择要移除的元素，直到达到目标稀疏度水平。
    """

    def update_mask(self, module, tensor_name, **kwargs):
        """
        更新掩码以便修剪权重。

        Args:
        - module: 包含权重的模块
        - tensor_name: 权重张量的名称
        - kwargs: 其他稀疏化配置参数，包括 sparsity_level

        Raises:
        - Exception: 如果权重张量维度小于等于1，则抛出异常
        """

        # 获取指定模块中的权重张量和相应的掩码
        weights = getattr(module, tensor_name)
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        # 如果权重张量的维度小于等于1，则无法应用结构化修剪
        if weights.dim() <= 1:
            raise Exception("Structured pruning can only be applied to a 2+dim weight tensor!")  # noqa: TRY002

        # 计算各行的负 L1 范数，以便使用 topk 函数（我们修剪掉最小的权重）
        saliency = -weights.norm(dim=tuple(range(1, weights.dim())), p=1)
        assert saliency.shape == mask.shape

        # 根据目标稀疏度级别计算需要修剪的行数
        num_to_pick = int(len(mask) * kwargs["sparsity_level"])

        # 使用 topk 函数找出要修剪的行的索引
        prune = saliency.topk(num_to_pick).indices

        # 将掩码中需要修剪的行置为 False
        mask.data[prune] = False
```