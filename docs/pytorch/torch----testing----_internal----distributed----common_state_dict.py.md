# `.\pytorch\torch\testing\_internal\distributed\common_state_dict.py`

```py
# mypy: ignore-errors
# 忽略类型检查错误，mypy 是一个类型检查工具

# Owner(s): ["oncall: distributed"]
# 代码所有者标记为分布式系统的值班人员

import copy  # 导入 copy 模块，用于对象的深复制
from itertools import chain  # 导入 itertools 中的 chain 函数，用于迭代对象的串联
from typing import Any, Dict  # 导入类型提示中的 Any 和 Dict 类型

import torch  # 导入 PyTorch 模块
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块

from torch.distributed._sharded_tensor import ShardedTensor  # 导入分片张量类 ShardedTensor
from torch.distributed._state_dict_utils import _gather_state_dict  # 导入函数 _gather_state_dict
from torch.distributed._tensor import DTensor  # 导入分布式张量类 DTensor
from torch.distributed.checkpoint.state_dict import (
    _PG,  # 导入 _PG 对象
    _STATE,  # 导入 _STATE 对象
    set_state_dict,  # 导入设置状态字典的函数 set_state_dict
    StateDictOptions,  # 导入状态字典选项类 StateDictOptions
)


class VerifyStateDictMixin:
    def _compare_tensor(self, orig_tensor, dist_tensor, offload_to_cpu=False):
        # 比较原始张量和分布式张量的值
        if isinstance(dist_tensor, (DTensor, ShardedTensor)):
            # 如果 dist_tensor 是 DTensor 或 ShardedTensor 类型，则使用 _gather_state_dict 收集它们
            dist_tensor = _gather_state_dict({"mykey": dist_tensor}).pop("mykey")

        if offload_to_cpu:
            # 如果需要将张量转移到 CPU 上进行比较
            orig_tensor = orig_tensor.cpu()
            dist_tensor = dist_tensor.cpu()
        self.assertTrue(isinstance(dist_tensor, torch.Tensor))  # 确保 dist_tensor 是 torch.Tensor 类型
        self.assertTrue(torch.allclose(orig_tensor, dist_tensor))  # 使用 allclose 检查张量值是否相似

    def _verify_msd(
        self,
        msd: Dict[str, Any],  # 参数 msd 是一个字典，键是字符串，值可以是任意类型
        dist_msd: Dict[str, Any],  # 参数 dist_msd 是一个字典，键是字符串，值可以是任意类型
        options: StateDictOptions = StateDictOptions(),  # 参数 options 是 StateDictOptions 类型的选项对象，默认为空选项
        offload_to_cpu=False,  # 是否将张量转移到 CPU 进行比较，默认为 False
    ) -> None:
        if not options.ignore_frozen_params:
            # 如果选项中不忽略冻结参数，则检查两个字典的长度是否相等
            self.assertEqual(len(msd), len(dist_msd))
        for fqn, param in msd.items():
            # 遍历 msd 字典中的每一项
            dist_param = dist_msd.get(fqn, None)  # 获取 dist_msd 中对应键 fqn 的值，如果不存在则为 None
            if not options.ignore_frozen_params:
                # 如果不忽略冻结参数，则确保 dist_param 不为 None，并比较原始参数和分布式参数
                self.assertIsNotNone(dist_param, f"{fqn=}")
                self._compare_tensor(param, dist_param, offload_to_cpu)
            elif dist_param is None:
                # 如果忽略冻结参数且 dist_param 为 None，则确保原始参数不需要梯度
                self.assertFalse(param.requires_grad, f"{fqn=}")

    def _verify_osd(
        self,
        model: nn.Module,  # 参数 model 是 nn.Module 类型的神经网络模型
        optim: torch.optim.Optimizer,  # 参数 optim 是 PyTorch 优化器 Optimizer 类型
        osd: Dict[str, Any],  # 参数 osd 是一个字典，键是字符串，值可以是任意类型
        dist_osd: Dict[str, Any],  # 参数 dist_osd 是一个字典，键是字符串，值可以是任意类型
    # 定义一个方法，用于验证优化器状态字典是否正确加载
    def _verify_osd_by_load(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        new_optim: torch.optim.Optimizer,
        dist_osd: Dict[str, Any],
    ) -> None:
        # 从输入的分布式状态字典中收集新的状态字典
        new_dist_osd = _gather_state_dict(dist_osd)
        # 设置模型和优化器的状态字典
        set_state_dict(
            model,
            optimizers=new_optim,
            model_state_dict={},
            optim_state_dict=new_dist_osd,
        )
        # 断言原始优化器的状态字典与新优化器的状态字典相同
        self.assertEqual(optim.state_dict(), new_optim.state_dict())
```