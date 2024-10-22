# `.\diffusers\pipelines\deprecated\repaint\pipeline_repaint.py`

```py
# 版权声明，表明版权所有者和使用许可证
# Copyright 2024 ETH Zurich Computer Vision Lab and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律要求或书面协议另有约定，软件按“原样”分发，
# 不提供任何明示或暗示的担保或条件。
# 请参见许可证以获取有关权限和限制的具体信息。

# 导入所需的类型
from typing import List, Optional, Tuple, Union

# 导入 numpy 库用于数组操作
import numpy as np
# 导入 PIL 库用于图像处理
import PIL.Image
# 导入 PyTorch 库用于深度学习模型
import torch

# 从本地模块导入 UNet2DModel 类
from ....models import UNet2DModel
# 从本地模块导入 RePaintScheduler 类
from ....schedulers import RePaintScheduler
# 从本地模块导入工具函数
from ....utils import PIL_INTERPOLATION, deprecate, logging
# 从本地模块导入随机张量生成函数
from ....utils.torch_utils import randn_tensor
# 从本地模块导入 DiffusionPipeline 和 ImagePipelineOutput 类
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 创建一个日志记录器，用于记录信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 预处理图像的函数，支持多种输入格式
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def _preprocess_image(image: Union[List, PIL.Image.Image, torch.Tensor]):
    # 定义弃用信息，提示用户将来版本会删除此方法
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 调用弃用函数，发出警告
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 如果输入是 PyTorch 张量，直接返回
    if isinstance(image, torch.Tensor):
        return image
    # 如果输入是 PIL 图像，转换为单元素列表
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    # 如果列表中的第一个元素是 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽度和高度
        w, h = image[0].size
        # 调整宽度和高度，使其为 8 的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        # 调整图像大小并转换为 NumPy 数组，添加新维度
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 将所有图像沿第 0 维连接成一个数组
        image = np.concatenate(image, axis=0)
        # 转换为浮点型数组并归一化到 [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组维度顺序为 (批次, 通道, 高度, 宽度)
        image = image.transpose(0, 3, 1, 2)
        # 将像素值从 [0, 1] 映射到 [-1, 1]
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果列表中的第一个元素是 PyTorch 张量
    elif isinstance(image[0], torch.Tensor):
        # 将张量沿第 0 维连接
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image

# 预处理掩码的函数，支持多种输入格式
def _preprocess_mask(mask: Union[List, PIL.Image.Image, torch.Tensor]):
    # 如果输入是 PyTorch 张量，直接返回
    if isinstance(mask, torch.Tensor):
        return mask
    # 如果输入是 PIL 图像，转换为单元素列表
    elif isinstance(mask, PIL.Image.Image):
        mask = [mask]

    # 如果列表中的第一个元素是 PIL 图像
    if isinstance(mask[0], PIL.Image.Image):
        # 获取掩码的宽度和高度
        w, h = mask[0].size
        # 调整宽度和高度，使其为 32 的整数倍
        w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
        # 将掩码调整大小并转换为灰度图像的 NumPy 数组，添加新维度
        mask = [np.array(m.convert("L").resize((w, h), resample=PIL_INTERPOLATION["nearest"]))[None, :] for m in mask]
        # 将所有掩码沿第 0 维连接成一个数组
        mask = np.concatenate(mask, axis=0)
        # 转换为浮点型数组并归一化到 [0, 1]
        mask = mask.astype(np.float32) / 255.0
        # 将值小于 0.5 的部分置为 0
        mask[mask < 0.5] = 0
        # 将值大于等于 0.5 的部分置为 1
        mask[mask >= 0.5] = 1
        # 将 NumPy 数组转换为 PyTorch 张量
        mask = torch.from_numpy(mask)
    # 如果列表中的第一个元素是 PyTorch 张量
    elif isinstance(mask[0], torch.Tensor):
        # 将张量沿第 0 维连接
        mask = torch.cat(mask, dim=0)
    # 返回处理后的掩码
    return mask

# 定义 RePaintPipeline 类，继承自 DiffusionPipeline
class RePaintPipeline(DiffusionPipeline):
    # 文档字符串，描述图像修复的管道使用 RePaint 模型
        r"""
        Pipeline for image inpainting using RePaint.
    
        This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
        implemented for all pipelines (downloading, saving, running on a particular device, etc.).
    
        Parameters:
            unet ([`UNet2DModel`]):
                A `UNet2DModel` to denoise the encoded image latents.
            scheduler ([`RePaintScheduler`]):
                A `RePaintScheduler` to be used in combination with `unet` to denoise the encoded image.
        """
    
        # 定义 UNet2DModel 类型的属性，用于图像去噪
        unet: UNet2DModel
        # 定义 RePaintScheduler 类型的属性，用于图像去噪的调度
        scheduler: RePaintScheduler
        # 设置模型 CPU 卸载序列为 "unet"
        model_cpu_offload_seq = "unet"
    
        # 初始化方法，接收 unet 和 scheduler 参数
        def __init__(self, unet, scheduler):
            # 调用父类的初始化方法
            super().__init__()
            # 注册 unet 和 scheduler 模块
            self.register_modules(unet=unet, scheduler=scheduler)
    
        # 禁用梯度计算的上下文装饰器
        @torch.no_grad()
        def __call__(
            # 接收图像和掩码图像，类型可以是 Torch 张量或 PIL 图像
            image: Union[torch.Tensor, PIL.Image.Image],
            mask_image: Union[torch.Tensor, PIL.Image.Image],
            # 设置推理步骤的数量，默认为 250
            num_inference_steps: int = 250,
            # 设置 eta 参数，默认为 0.0
            eta: float = 0.0,
            # 设置跳跃长度，默认为 10
            jump_length: int = 10,
            # 设置跳跃样本数量，默认为 10
            jump_n_sample: int = 10,
            # 生成器参数，可以是一个生成器或生成器列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 输出类型参数，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典形式的结果，默认为 True
            return_dict: bool = True,
```