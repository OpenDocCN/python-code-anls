# `.\diffusers\pipelines\deprecated\pndm\pipeline_pndm.py`

```py
# 版权声明，声明此代码的版权所有者及使用许可信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 使用 Apache 许可证，版本 2.0 的许可声明
# Licensed under the Apache License, Version 2.0 (the "License");
# 本文件只能在遵守该许可证的前提下使用
# you may not use this file except in compliance with the License.
# 可在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 如果没有适用的法律规定或书面同意，软件在“按原样”基础上分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参阅许可证以了解特定语言所适用的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.


# 导入类型提示
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从模型模块导入 UNet2DModel 类
from ....models import UNet2DModel
# 从调度器模块导入 PNDMScheduler 类
from ....schedulers import PNDMScheduler
# 从工具模块导入随机张量生成函数
from ....utils.torch_utils import randn_tensor
# 从管道工具模块导入 DiffusionPipeline 和 ImagePipelineOutput
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput


# 定义无条件图像生成的管道类
class PNDMPipeline(DiffusionPipeline):
    r"""
    无条件图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取所有管道的通用方法
    (下载、保存、在特定设备上运行等) 的实现。

    参数：
        unet ([`UNet2DModel`]):
            用于去噪编码图像潜在值的 `UNet2DModel`。
        scheduler ([`PNDMScheduler`]):
            用于与 `unet` 结合使用以去噪编码图像的 `PNDMScheduler`。
    """

    # 定义 UNet2DModel 实例变量
    unet: UNet2DModel
    # 定义 PNDMScheduler 实例变量
    scheduler: PNDMScheduler

    # 初始化方法，接收 UNet2DModel 和 PNDMScheduler 实例
    def __init__(self, unet: UNet2DModel, scheduler: PNDMScheduler):
        # 调用父类的初始化方法
        super().__init__()

        # 从配置中创建 PNDMScheduler 实例
        scheduler = PNDMScheduler.from_config(scheduler.config)

        # 注册 unet 和 scheduler 模块
        self.register_modules(unet=unet, scheduler=scheduler)

    # 定义无梯度上下文中的可调用方法
    @torch.no_grad()
    def __call__(
        # 批处理大小，默认为 1
        self,
        batch_size: int = 1,
        # 推理步骤数，默认为 50
        num_inference_steps: int = 50,
        # 可选的随机数生成器，支持单个或多个生成器
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典形式的结果，默认为 True
        return_dict: bool = True,
        # 其他可选参数
        **kwargs,
    # 定义生成管道调用的函数，返回生成结果
    ) -> Union[ImagePipelineOutput, Tuple]:
        # 文档字符串，描述生成过程的函数调用及参数说明
        r"""
        生成管道的调用函数。
    
        参数：
            batch_size (`int`, `optional`, defaults to 1):
                生成图像的数量。
            num_inference_steps (`int`, `optional`, defaults to 50):
                去噪步骤的数量。更多的去噪步骤通常会导致更高质量的图像，但推理速度较慢。
            generator (`torch.Generator`, `optional`):
                用于生成确定性结果的 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)。
            output_type (`str`, `optional`, defaults to `"pil"`):
                生成图像的输出格式。可以选择 `PIL.Image` 或 `np.array`。
            return_dict (`bool`, *optional*, defaults to `True`):
                是否返回 [`ImagePipelineOutput`] 而不是普通元组。
    
        示例：
    
        ```py
        >>> from diffusers import PNDMPipeline
    
        >>> # 加载模型和调度器
        >>> pndm = PNDMPipeline.from_pretrained("google/ddpm-cifar10-32")
    
        >>> # 在推理中运行管道（采样随机噪声并去噪）
        >>> image = pndm().images[0]
    
        >>> # 保存图像
        >>> image.save("pndm_generated_image.png")
        ```py
    
        返回：
            [`~pipelines.ImagePipelineOutput`] 或 `tuple`:
                如果 `return_dict` 为 `True`，则返回 [`~pipelines.ImagePipelineOutput`]，否则返回一个元组，
                其中第一个元素是生成的图像列表。
        """
        # 采样方法的更多信息可以参考官方论文的算法2： https://arxiv.org/pdf/2202.09778.pdf
    
        # 生成高斯噪声以开始循环
        image = randn_tensor(
            # 创建形状为 (batch_size, 通道数, 样本大小, 样本大小) 的随机张量
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,  # 使用指定的生成器
            device=self.device,  # 设置设备为指定的设备
        )
    
        # 设置调度器的时间步长
        self.scheduler.set_timesteps(num_inference_steps)
        # 遍历进度条中的每个时间步
        for t in self.progress_bar(self.scheduler.timesteps):
            # 通过 UNet 模型进行推理，获取模型输出
            model_output = self.unet(image, t).sample
    
            # 根据调度器步骤更新图像，获取前一帧样本
            image = self.scheduler.step(model_output, t, image).prev_sample
    
        # 将图像缩放到 [0, 1] 的范围
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像从张量转为 numpy 数组，并调整维度顺序
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        # 如果输出类型为 "pil"，则将 numpy 数组转换为 PIL 图像
        if output_type == "pil":
            image = self.numpy_to_pil(image)
    
        # 如果不返回字典，返回元组格式
        if not return_dict:
            return (image,)
    
        # 返回包含生成图像的 ImagePipelineOutput 对象
        return ImagePipelineOutput(images=image)
```