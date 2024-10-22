# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\guiders.py`

```py
# 导入 logging 模块以进行日志记录
import logging
# 从 abc 模块导入 ABC 和 abstractmethod 以定义抽象基类和抽象方法
from abc import ABC, abstractmethod
# 从 typing 模块导入用于类型注解的各种类型
from typing import Dict, List, Optional, Tuple, Union
# 从 functools 导入 partial 用于部分函数应用
from functools import partial
# 导入 math 模块以进行数学运算
import math

# 导入 PyTorch 库
import torch
# 从 einops 导入 rearrange 和 repeat 用于重排和重复张量
from einops import rearrange, repeat

# 从上级目录的 util 模块导入 append_dims、default 和 instantiate_from_config 函数
from ...util import append_dims, default, instantiate_from_config


# 定义一个抽象基类 Guider，继承自 ABC
class Guider(ABC):
    # 定义一个抽象方法 __call__，接收张量和浮点数，并返回一个张量
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    # 定义准备输入的方法，接收张量、浮点数和字典，并返回一个元组
    def prepare_inputs(self, x: torch.Tensor, s: float, c: Dict, uc: Dict) -> Tuple[torch.Tensor, float, Dict]:
        pass


# 定义 VanillaCFG 类，实现并行化 CFG
class VanillaCFG:
    """
    implements parallelized CFG
    """

    # 初始化方法，接收缩放因子和可选的动态阈值配置
    def __init__(self, scale, dyn_thresh_config=None):
        # 设置缩放因子
        self.scale = scale
        # 定义缩放调度函数，独立于步骤
        scale_schedule = lambda scale, sigma: scale  # independent of step
        # 使用 partial 绑定缩放调度函数
        self.scale_schedule = partial(scale_schedule, scale)
        # 根据配置实例化动态阈值
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    # 定义调用方法，接收张量、浮点数和可选缩放因子
    def __call__(self, x, sigma, scale=None):
        # 将输入张量 x 拆分为上下文和未上下文部分
        x_u, x_c = x.chunk(2)
        # 获取缩放值，优先使用传入的缩放因子
        scale_value = default(scale, self.scale_schedule(sigma))
        # 使用动态阈值处理未上下文和上下文部分
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        # 返回预测结果
        return x_pred

    # 定义准备输入的方法
    def prepare_inputs(self, x, s, c, uc):
        # 创建一个空字典用于存储输出上下文
        c_out = dict()

        # 遍历上下文字典 c
        for k in c:
            # 如果键在特定列表中，则连接 uc 和 c 中的对应张量
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                # 否则，确保 c 和 uc 中的值相同，并将其直接赋值
                assert c[k] == uc[k]
                c_out[k] = c[k]
        # 返回两个相同的张量 x 和 s，以及输出上下文字典
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


# 定义 DynamicCFG 类，继承自 VanillaCFG
class DynamicCFG(VanillaCFG):
    # 初始化方法，接收缩放因子、指数、步骤数和可选的动态阈值配置
    def __init__(self, scale, exp, num_steps, dyn_thresh_config=None):
        # 调用父类的初始化方法
        super().__init__(scale, dyn_thresh_config)
        # 定义动态缩放调度函数
        scale_schedule = (
            lambda scale, sigma, step_index: 1 + scale * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )
        # 使用 partial 绑定缩放调度函数
        self.scale_schedule = partial(scale_schedule, scale)
        # 根据配置实例化动态阈值
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    # 定义调用方法，接收张量、浮点数、步骤索引和可选缩放因子
    def __call__(self, x, sigma, step_index, scale=None):
        # 将输入张量 x 拆分为上下文和未上下文部分
        x_u, x_c = x.chunk(2)
        # 获取缩放值，使用动态缩放调度函数
        scale_value = self.scale_schedule(sigma, step_index.item())
        # 使用动态阈值处理未上下文和上下文部分
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        # 返回预测结果
        return x_pred


# 定义 IdentityGuider 类
class IdentityGuider:
    # 定义调用方法，接收张量和浮点数，直接返回输入张量
    def __call__(self, x, sigma):
        return x

    # 定义准备输入的方法
    def prepare_inputs(self, x, s, c, uc):
        # 创建一个空字典用于存储输出上下文
        c_out = dict()

        # 遍历上下文字典 c
        for k in c:
            # 将上下文中的值直接赋值到输出上下文中
            c_out[k] = c[k]

        # 返回原始张量 x、s 和输出上下文字典
        return x, s, c_out
```