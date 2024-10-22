# `.\diffusers\pipelines\dit\pipeline_dit.py`

```py
# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)  # 版权声明，说明此文件的使用限制
# William Peebles and Saining Xie  # 贡献者姓名
#
# Copyright (c) 2021 OpenAI  # 表示 OpenAI 对该文件的版权
# MIT License  # 说明该文件遵循 MIT 许可证
#
# Copyright 2024 The HuggingFace Team. All rights reserved.  # HuggingFace 团队对文件的版权声明
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 说明文件遵循 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 使用文件的前提条件
# You may obtain a copy of the License at  # 提供许可证获取链接
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证链接
#
# Unless required by applicable law or agreed to in writing, software  # 说明在特定情况下的免责条款
# distributed under the License is distributed on an "AS IS" BASIS,  # 文件按现状提供，不提供任何担保
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不承担任何明示或暗示的责任
# See the License for the specific language governing permissions and  # 提供查看许可证的建议
# limitations under the License.  # 说明许可证的限制

from typing import Dict, List, Optional, Tuple, Union  # 从 typing 模块导入类型提示功能

import torch  # 导入 PyTorch 库

from ...models import AutoencoderKL, DiTTransformer2DModel  # 从模型模块导入相关类
from ...schedulers import KarrasDiffusionSchedulers  # 从调度器模块导入 KarrasDiffusionSchedulers 类
from ...utils.torch_utils import randn_tensor  # 从工具模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从管道工具模块导入相关类


class DiTPipeline(DiffusionPipeline):  # 定义一个继承自 DiffusionPipeline 的类 DiTPipeline
    r"""  # 文档字符串，描述此类的功能和参数
    Pipeline for image generation based on a Transformer backbone instead of a UNet.  # 说明此管道用于基于 Transformer 的图像生成，而非 UNet

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods  # 指出该模型继承自 DiffusionPipeline，并建议查看父类文档
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).  # 指出可用的通用方法

    Parameters:  # 说明类的参数
        transformer ([`DiTTransformer2DModel`]):  # transformer 参数，类型为 DiTTransformer2DModel
            A class conditioned `DiTTransformer2DModel` to denoise the encoded image latents.  # 用于去噪图像潜在表示的 transformer 类
        vae ([`AutoencoderKL`]):  # vae 参数，类型为 AutoencoderKL
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.  # 用于图像编码和解码的变分自编码器模型
        scheduler ([`DDIMScheduler`]):  # scheduler 参数，类型为 DDIMScheduler
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.  # 与 transformer 结合使用以去噪的调度器
    """

    model_cpu_offload_seq = "transformer->vae"  # 定义模型 CPU 负载卸载的顺序

    def __init__(  # 定义初始化方法
        self,
        transformer: DiTTransformer2DModel,  # transformer 参数
        vae: AutoencoderKL,  # vae 参数
        scheduler: KarrasDiffusionSchedulers,  # scheduler 参数
        id2label: Optional[Dict[int, str]] = None,  # 可选的 id2label 参数，默认为 None
    ):
        super().__init__()  # 调用父类的初始化方法
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)  # 注册传入的模块

        # create a imagenet -> id dictionary for easier use  # 创建一个方便使用的 imagenet 到 id 的字典
        self.labels = {}  # 初始化标签字典
        if id2label is not None:  # 如果 id2label 参数不为 None
            for key, value in id2label.items():  # 遍历 id2label 的键值对
                for label in value.split(","):  # 分割每个值为多个标签
                    self.labels[label.lstrip().rstrip()] = int(key)  # 去除标签两端空白并存储对应的 id
            self.labels = dict(sorted(self.labels.items()))  # 对标签字典进行排序
    # 获取标签字符串对应的类 ID 的方法
    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""
    
        将 ImageNet 的标签字符串映射到对应的类 ID。
    
        参数：
            label (`str` 或 `dict` of `str`):
                要映射到类 ID 的标签字符串。
    
        返回：
            `list` of `int`:
                要被管道处理的类 ID 列表。
        """
    
        # 检查输入的 label 是否为列表，如果不是则将其转换为列表
        if not isinstance(label, list):
            label = list(label)
    
        # 遍历标签列表中的每个标签
        for l in label:
            # 检查标签是否在已知标签列表中，不存在则抛出错误
            if l not in self.labels:
                raise ValueError(
                    f"{l} 不存在。请确保选择以下标签之一： \n {self.labels}."
                )
    
        # 返回标签对应的类 ID 列表
        return [self.labels[l] for l in label]
    
    # 无梯度计算装饰器，禁止计算梯度以节省内存和计算资源
    @torch.no_grad()
    def __call__(
        # 输入的类标签列表
        class_labels: List[int],
        # 指导比例的默认值
        guidance_scale: float = 4.0,
        # 可选的随机数生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 推理步骤的默认次数
        num_inference_steps: int = 50,
        # 输出类型的默认值
        output_type: Optional[str] = "pil",
        # 是否返回字典的默认值
        return_dict: bool = True,
```