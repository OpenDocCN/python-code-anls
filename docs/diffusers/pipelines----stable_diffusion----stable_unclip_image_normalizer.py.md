# `.\diffusers\pipelines\stable_diffusion\stable_unclip_image_normalizer.py`

```py
# 版权声明，说明版权信息和许可条款
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 该文件的使用须遵循许可证的规定
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非在适用的情况下或以书面形式达成一致，软件
# 在许可证下分发是基于 "按现状" 的基础，不提供任何形式的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 见许可证了解特定语言的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 从 typing 模块导入可选和联合类型
from typing import Optional, Union

# 导入 PyTorch 库
import torch
# 从 torch 库中导入神经网络模块
from torch import nn

# 从配置工具中导入基类和注册装饰器
from ...configuration_utils import ConfigMixin, register_to_config
# 从模型工具中导入模型基类
from ...models.modeling_utils import ModelMixin


# 定义 StableUnCLIPImageNormalizer 类，继承自 ModelMixin 和 ConfigMixin
class StableUnCLIPImageNormalizer(ModelMixin, ConfigMixin):
    """
    该类用于保存用于稳定 unCLIP 的 CLIP 嵌入器的均值和标准差。

    在应用噪声之前，用于对图像嵌入进行标准化，并在去除标准化后的噪声图像
    嵌入时使用。
    """

    # 使用注册装饰器将此方法注册到配置中
    @register_to_config
    def __init__(
        self,
        embedding_dim: int = 768,  # 初始化时设定嵌入维度，默认值为 768
    ):
        # 调用父类构造函数
        super().__init__()

        # 定义均值参数，初始化为零的张量，形状为 (1, embedding_dim)
        self.mean = nn.Parameter(torch.zeros(1, embedding_dim))
        # 定义标准差参数，初始化为一的张量，形状为 (1, embedding_dim)
        self.std = nn.Parameter(torch.ones(1, embedding_dim))

    # 定义设备转换方法，可以将均值和标准差移动到指定设备和数据类型
    def to(
        self,
        torch_device: Optional[Union[str, torch.device]] = None,  # 可选的设备参数
        torch_dtype: Optional[torch.dtype] = None,  # 可选的数据类型参数
    ):
        # 将均值参数转换到指定的设备和数据类型
        self.mean = nn.Parameter(self.mean.to(torch_device).to(torch_dtype))
        # 将标准差参数转换到指定的设备和数据类型
        self.std = nn.Parameter(self.std.to(torch_device).to(torch_dtype))
        # 返回当前对象
        return self

    # 定义缩放方法，对嵌入进行标准化
    def scale(self, embeds):
        # 根据均值和标准差标准化嵌入
        embeds = (embeds - self.mean) * 1.0 / self.std
        # 返回标准化后的嵌入
        return embeds

    # 定义反缩放方法，将标准化的嵌入恢复为原始值
    def unscale(self, embeds):
        # 根据标准差和均值反标准化嵌入
        embeds = (embeds * self.std) + self.mean
        # 返回恢复后的嵌入
        return embeds
```