# `.\diffusers\schedulers\scheduling_utils.py`

```py
# 版权声明，表明该文件的版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则您不能使用此文件。
# 您可以在以下地址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是按“原样”提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以了解管理权限和限制的具体条款。
import importlib  # 导入 importlib 模块以便动态导入模块
import os  # 导入 os 模块以进行操作系统交互
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from enum import Enum  # 从 enum 模块导入 Enum 类以定义枚举
from typing import Optional, Union  # 从 typing 模块导入 Optional 和 Union 类型提示

import torch  # 导入 PyTorch 库以进行深度学习计算
from huggingface_hub.utils import validate_hf_hub_args  # 导入验证 Hugging Face Hub 参数的工具函数

from ..utils import BaseOutput, PushToHubMixin  # 从上级目录导入 BaseOutput 和 PushToHubMixin 类


SCHEDULER_CONFIG_NAME = "scheduler_config.json"  # 定义调度器配置文件的名称


# 注意：将该类型定义为枚举类型，因为它简化了文档中的使用，并防止在调度器模块的 `_compatibles` 中产生循环导入。
# 当在管道中用作类型时，它实际上是一个联合类型，因为实际的调度器实例被传入。
class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1  # 定义 DDIM 调度器的枚举值
    DDPMScheduler = 2  # 定义 DDPMScheduler 调度器的枚举值
    PNDMScheduler = 3  # 定义 PNDM 调度器的枚举值
    LMSDiscreteScheduler = 4  # 定义 LMS 离散调度器的枚举值
    EulerDiscreteScheduler = 5  # 定义 Euler 离散调度器的枚举值
    HeunDiscreteScheduler = 6  # 定义 Heun 离散调度器的枚举值
    EulerAncestralDiscreteScheduler = 7  # 定义 Euler 祖先离散调度器的枚举值
    DPMSolverMultistepScheduler = 8  # 定义 DPM 求解器多步调度器的枚举值
    DPMSolverSinglestepScheduler = 9  # 定义 DPM 求解器单步调度器的枚举值
    KDPM2DiscreteScheduler = 10  # 定义 KDPM2 离散调度器的枚举值
    KDPM2AncestralDiscreteScheduler = 11  # 定义 KDPM2 祖先离散调度器的枚举值
    DEISMultistepScheduler = 12  # 定义 DEIS 多步调度器的枚举值
    UniPCMultistepScheduler = 13  # 定义 UniPC 多步调度器的枚举值
    DPMSolverSDEScheduler = 14  # 定义 DPM 求解器 SDE 调度器的枚举值
    EDMEulerScheduler = 15  # 定义 EMD Euler 调度器的枚举值


# 定义一个字典，包含不同调度器的时间步和 sigma 值
AysSchedules = {
    "StableDiffusionTimesteps": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],  # 稳定扩散的时间步列表
    "StableDiffusionSigmas": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0],  # 稳定扩散的 sigma 值列表
    "StableDiffusionXLTimesteps": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],  # 稳定扩散 XL 的时间步列表
    "StableDiffusionXLSigmas": [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0],  # 稳定扩散 XL 的 sigma 值列表
    "StableDiffusionVideoSigmas": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.0],  # 稳定扩散视频的 sigma 值列表
}


@dataclass
class SchedulerOutput(BaseOutput):  # 定义调度器输出类，继承自 BaseOutput
    """
    调度器的 `step` 函数输出的基类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像)：
            先前时间步的计算样本 `(x_{t-1})`。`prev_sample` 应用作去噪循环中的下一个模型输入。
    """

    prev_sample: torch.Tensor  # 定义 prev_sample 属性，类型为 torch.Tensor


class SchedulerMixin(PushToHubMixin):  # 定义调度器混合类，继承自 PushToHubMixin
    """
    所有调度器的基类。

    [`SchedulerMixin`] 包含所有调度器共享的常用函数，例如通用的加载和保存功能。

    [`ConfigMixin`] 负责存储传递的配置属性（如 `num_train_timesteps`）。
    # 调度器的 `__init__` 函数，属性可通过 `scheduler.config.num_train_timesteps` 访问

    # 类属性：
    # - **_compatibles** (`List[str]`) -- 兼容父调度器类的调度器类列表
    #   使用 [`~ConfigMixin.from_config`] 加载不同的兼容调度器类（应由父类重写）。
    """

    # 配置名称设为调度器配置名称常量
    config_name = SCHEDULER_CONFIG_NAME
    # 初始化兼容调度器列表为空
    _compatibles = []
    # 指示存在兼容调度器
    has_compatibles = True

    # 类方法，允许从预训练模型加载
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        # 预训练模型名称或路径（可选）
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        # 子文件夹（可选）
        subfolder: Optional[str] = None,
        # 是否返回未使用的参数
        return_unused_kwargs=False,
        # 其他参数
        **kwargs,
    # 保存调度器配置对象到指定目录
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        将调度器配置对象保存到目录，以便使用
        [`~SchedulerMixin.from_pretrained`] 类方法重新加载。

        参数：
            save_directory (`str` 或 `os.PathLike`)：
                配置 JSON 文件将保存的目录（如果不存在则创建）。
            push_to_hub (`bool`, *可选*, 默认为 `False`)：
                保存后是否将模型推送到 Hugging Face Hub。可以使用 `repo_id` 指定要推送的
                仓库（默认为 `save_directory` 的名称）。
            kwargs (`Dict[str, Any]`, *可选*)：
                其他关键词参数，传递给 [`~utils.PushToHubMixin.push_to_hub`] 方法。
        """
        # 保存配置，传入目录和推送标志及其他参数
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)

    # 属性方法，返回与该调度器兼容的所有调度器
    @property
    def compatibles(self):
        """
        返回与此调度器兼容的所有调度器

        返回：
            `List[SchedulerMixin]`: 兼容调度器的列表
        """
        # 获取兼容调度器列表
        return self._get_compatibles()

    # 类方法，获取兼容类
    @classmethod
    def _get_compatibles(cls):
        # 创建包含当前类名和兼容类名的唯一字符串列表
        compatible_classes_str = list(set([cls.__name__] + cls._compatibles))
        # 导入 diffusers 库
        diffusers_library = importlib.import_module(__name__.split(".")[0])
        # 获取存在的兼容类
        compatible_classes = [
            getattr(diffusers_library, c) for c in compatible_classes_str if hasattr(diffusers_library, c)
        ]
        # 返回兼容类列表
        return compatible_classes
```