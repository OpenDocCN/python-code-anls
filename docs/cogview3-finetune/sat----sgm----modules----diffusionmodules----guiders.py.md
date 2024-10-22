# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\guiders.py`

```py
# 导入 logging 模块，用于记录日志信息
import logging
# 从 abc 模块导入 ABC 类和 abstractmethod 装饰器，用于定义抽象基类和抽象方法
from abc import ABC, abstractmethod
# 导入类型注解，方便在函数签名中定义复杂数据结构
from typing import Dict, List, Optional, Tuple, Union
# 从 functools 模块导入 partial 函数，用于部分应用函数
from functools import partial
# 导入数学模块，提供数学函数
import math

# 导入 PyTorch 库，提供张量计算功能
import torch
# 从 einops 模块导入 rearrange 和 repeat 函数，用于张量重排和重复
from einops import rearrange, repeat

# 从上层模块导入工具函数，提供一些默认值和实例化配置的功能
from ...util import append_dims, default, instantiate_from_config

# 定义一个抽象基类 Guider，继承自 ABC
class Guider(ABC):
    # 定义一个抽象方法 __call__，接受一个张量和一个浮点数，返回一个张量
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    # 定义准备输入的方法，接受多个参数并返回一个元组
    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        pass


# 定义一个类 VanillaCFG，表示基本的条件生成模型
class VanillaCFG:
    """
    implements parallelized CFG
    """

    # 初始化方法，接受比例和动态阈值配置
    def __init__(self, scale, dyn_thresh_config=None):
        # 定义一个 lambda 函数，根据 sigma 返回 scale，保持独立于步数
        scale_schedule = lambda scale, sigma: scale  # independent of step
        # 使用 partial 固定 scale 参数，创建 scale_schedule 方法
        self.scale_schedule = partial(scale_schedule, scale)
        # 实例化动态阈值对象，如果没有提供配置则使用默认配置
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
                },
            )
        )

    # 定义 __call__ 方法，使该类可以被调用，接受多个参数
    def __call__(self, x, sigma, step = None, num_steps = None, **kwargs):
        # 将输入张量 x 拆分为两个部分 x_u 和 x_c
        x_u, x_c = x.chunk(2)
        # 根据 sigma 计算 scale_value
        scale_value = self.scale_schedule(sigma)
        # 使用动态阈值处理函数进行预测，返回预测结果
        x_pred = self.dyn_thresh(x_u, x_c, scale_value, step=step, num_steps=num_steps)
        return x_pred

    # 定义准备输入的方法，接受多个参数并返回一个元组
    def prepare_inputs(self, x, s, c, uc):
        # 初始化输出字典
        c_out = dict()

        # 遍历条件字典 c 的键
        for k in c:
            # 如果键是特定值，则将 uc 和 c 中的对应张量拼接
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            # 否则确保两个字典中对应的值相等，并直接赋值
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        # 返回拼接后的张量和条件字典
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


# 定义一个类 IdentityGuider，实现一个恒等引导器
class IdentityGuider:
    # 定义 __call__ 方法，直接返回输入张量
    def __call__(self, x, sigma, **kwargs):
        return x

    # 定义准备输入的方法，返回输入和条件字典
    def prepare_inputs(self, x, s, c, uc):
        # 初始化输出字典
        c_out = dict()

        # 遍历条件字典 c 的键
        for k in c:
            # 直接将条件字典 c 的值赋给输出字典
            c_out[k] = c[k]

        # 返回输入张量和条件字典
        return x, s, c_out


# 定义一个类 LinearPredictionGuider，继承自 Guider
class LinearPredictionGuider(Guider):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        # 初始化最小和最大比例
        self.min_scale = min_scale
        self.max_scale = max_scale
        # 计算比例的线性变化，生成 num_frames 个值
        self.num_frames = num_frames
        self.scale = torch.linspace(min_scale, max_scale, num_frames).unsqueeze(0)

        # 确保 additional_cond_keys 是一个列表，如果是字符串则转换为列表
        additional_cond_keys = default(additional_cond_keys, [])
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        # 保存附加条件键
        self.additional_cond_keys = additional_cond_keys
    # 定义可调用对象的方法，接收输入张量 x 和 sigma，以及其他参数 kwargs，返回一个张量
    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        # 将输入张量 x 拆分为两部分：x_u 和 x_c
        x_u, x_c = x.chunk(2)
    
        # 重排 x_u 的维度，使其形状为 (批量大小 b, 帧数 t, ...)，t 由 num_frames 指定
        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=self.num_frames)
        # 重排 x_c 的维度，使其形状为 (批量大小 b, 帧数 t, ...)，t 由 num_frames 指定
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=self.num_frames)
        # 复制 scale 张量的维度，使其形状为 (批量大小 b, 帧数 t)
        scale = repeat(self.scale, "1 t -> b t", b=x_u.shape[0])
        # 将 scale 的维度扩展到与 x_u 的维度一致，并移动到 x_u 的设备上
        scale = append_dims(scale, x_u.ndim).to(x_u.device)
        # 将 scale 转换为与 x_u 相同的数据类型
        scale = scale.to(x_u.dtype)
    
        # 返回经过计算的结果，重排为 (批量大小 b * 帧数 t, ...)
        return rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")
    
    # 定义准备输入的函数，接收输入张量 x 和 s，以及条件字典 c 和 uc，返回一个元组
    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # 初始化一个空字典 c_out 用于存放处理后的条件
        c_out = dict()
    
        # 遍历条件字典 c 的每一个键 k
        for k in c:
            # 如果 k 是指定的条件键之一，进行拼接
            if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:
                # 将 uc[k] 和 c[k] 沿第0维拼接，并存入 c_out
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                # 确保 c[k] 与 uc[k] 相等
                assert c[k] == uc[k]
                # 将 c[k] 直接存入 c_out
                c_out[k] = c[k]
        # 返回拼接后的 x 和 s 以及处理后的条件字典 c_out
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out
```