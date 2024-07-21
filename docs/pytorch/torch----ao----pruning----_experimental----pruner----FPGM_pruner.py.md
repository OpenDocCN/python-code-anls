# `.\pytorch\torch\ao\pruning\_experimental\pruner\FPGM_pruner.py`

```
# mypy: allow-untyped-defs
# 导入所需的类型和模块
from typing import Callable, Optional, Union

import torch  # 导入PyTorch模块

from .base_structured_sparsifier import BaseStructuredSparsifier  # 导入基础结构稀疏化类

__all__ = ["FPGMPruner"]  # 定义模块公开的类名列表


class FPGMPruner(BaseStructuredSparsifier):
    r"""Filter Pruning via Geometric Median (FPGM) Structured Pruner
    This sparsifier prune fliter (row) in a tensor according to distances among filters according to
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`_.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of filters (rows) that are zeroed-out.
    2. `dist` defines the distance measurement type. Default: 3 (L2 distance).
    Available options are: [1, 2, (custom callable distance function)].

    Note::
        Inputs should be a 4D convolutional tensor of shape (N, C, H, W).
            - N: output channels size
            - C: input channels size
            - H: height of kernel
            - W: width of kernel
    """

    def __init__(
        self, sparsity_level: float = 0.5, dist: Optional[Union[Callable, int]] = None
    ):
        # 定义默认参数字典
        defaults = {
            "sparsity_level": sparsity_level,
        }

        # 如果距离函数为None，默认使用L2范数（欧氏距离）
        if dist is None:
            dist = 2

        # 根据距离函数的类型，设置对应的距离计算函数
        if callable(dist):
            self.dist_fn = dist
        elif dist == 1:
            self.dist_fn = lambda x: torch.cdist(x, x, p=1)  # 使用L1范数计算距离
        elif dist == 2:
            self.dist_fn = lambda x: torch.cdist(x, x, p=2)  # 使用L2范数计算距离
        else:
            raise NotImplementedError("Distance function is not yet implemented.")  # 抛出未实现错误

        # 调用父类构造函数初始化
        super().__init__(defaults=defaults)

    def _compute_distance(self, t):
        r"""Compute distance across all entries in tensor `t` along all dimension
        except for the one identified by dim.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
        Returns:
            distance (torch.Tensor): distance computed across filtters
        """
        dim = 0  # 定义要稀疏化的维度为0（即滤波器/行）

        size = t.size(dim)
        slc = [slice(None)] * t.dim()

        # 将tensor展平到一维数组
        t_flatten = [
            t[tuple(slc[:dim] + [slice(i, i + 1)] + slc[dim + 1 :])].reshape(-1)
            for i in range(size)
        ]
        t_flatten = torch.stack(t_flatten)

        # 使用定义好的距离函数计算距离矩阵
        dist_matrix = self.dist_fn(t_flatten)

        # 计算每行的距离之和，用于确定稀疏化的程度
        distance = torch.sum(torch.abs(dist_matrix), 1)

        return distance  # 返回计算得到的距离
    # 定义一个方法，用于更新模块中指定张量的稀疏性掩码
    def update_mask(self, module, tensor_name, sparsity_level, **kwargs):
        # 获取模块中指定名称的张量
        tensor_weight = getattr(module, tensor_name)
        # 获取模块参数化对象中指定张量的掩码
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        # 根据稀疏性水平设置掩码
        if sparsity_level <= 0:
            mask.data = torch.ones_like(mask).bool()  # 如果稀疏性小于等于0，设为全真
        elif sparsity_level >= 1.0:
            mask.data = torch.zeros_like(mask).bool()  # 如果稀疏性大于等于1.0，设为全假
        else:
            # 计算张量权重之间的距离
            distance = self._compute_distance(tensor_weight)

            tensor_size = tensor_weight.shape[0]  # 获取张量的大小，即滤波器的数量（行数）
            nparams_toprune = round(sparsity_level * tensor_size)  # 计算需要裁剪的参数数目
            nparams_toprune = min(
                max(nparams_toprune, 0), tensor_size
            )  # 将裁剪数目限制在 [0, tensor_size] 范围内
            topk = torch.topk(distance, k=nparams_toprune, largest=False)  # 获取距离最小的 topk 个索引
            mask[topk.indices] = False  # 将这些索引对应的掩码设置为 False，表示不裁剪这些参数
```