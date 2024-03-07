# `.\YOLO-World\yolo_world\engine\optimizers\yolow_v5_optim_constructor.py`

```py
# 版权声明，版权归腾讯公司所有
import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import GroupNorm, LayerNorm
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper, DefaultOptimWrapperConstructor
from mmengine.utils.dl_utils import mmcv_full_available
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm

from mmyolo.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                             OPTIMIZERS)

# 注册优化器包装器构造函数
@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class YOLOWv5OptimizerConstructor(DefaultOptimWrapperConstructor):
    """YOLO World v5 constructor for optimizers."""

    # 初始化函数，接受优化器包装器配置和参数配置
    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None) -> None:
        # 调用父类的初始化函数
        super().__init__(optim_wrapper_cfg, paramwise_cfg)
        # 从参数配置中弹出'base_total_batch_size'，默认值为64
        self.base_total_batch_size = self.paramwise_cfg.pop(
            'base_total_batch_size', 64)
    # 定义一个方法，用于为模型创建优化器包装器
    def __call__(self, model: nn.Module) -> OptimWrapper:
        # 如果模型有'module'属性，则将'module'属性赋值给model
        if hasattr(model, 'module'):
            model = model.module

        # 复制优化器包装器配置
        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        # 设置默认的优化器包装器类型为'OptimWrapper'
        optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
        # 复制优化器配置
        optimizer_cfg = self.optimizer_cfg.copy()

        # 遵循原始的yolov5实现
        if 'batch_size_per_gpu' in optimizer_cfg:
            # 弹出'batch_size_per_gpu'键值对，并赋值给batch_size_per_gpu
            batch_size_per_gpu = optimizer_cfg.pop('batch_size_per_gpu')
            # 计算总批量大小
            total_batch_size = get_world_size() * batch_size_per_gpu
            # 计算累积步数
            accumulate = max(
                round(self.base_total_batch_size / total_batch_size), 1)
            # 计算缩放因子
            scale_factor = total_batch_size * \
                accumulate / self.base_total_batch_size

            # 如果缩放因子不等于1
            if scale_factor != 1:
                # 获取优化器配置中的权重衰减值
                weight_decay = optimizer_cfg.get('weight_decay', 0)
                # 根据缩放因子调整权重衰减值
                weight_decay *= scale_factor
                optimizer_cfg['weight_decay'] = weight_decay
                # 打印调整后的权重衰减值
                print_log(f'Scaled weight_decay to {weight_decay}', 'current')

        # 如果没有指定paramwise选项，则使用全局设置
        if not self.paramwise_cfg:
            # 将模型的参数设置为优化器配置的参数
            optimizer_cfg['params'] = model.parameters()
            # 构建优化器
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        else:
            # 递归设置参数的学习率和权重衰减
            params: List = []
            self.add_params(params, model)
            optimizer_cfg['params'] = params
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        # 构建优化器包装器
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        # 返回优化器包装器
        return optim_wrapper
```