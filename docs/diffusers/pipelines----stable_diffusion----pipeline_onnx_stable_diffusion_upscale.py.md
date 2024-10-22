# `.\diffusers\pipelines\stable_diffusion\pipeline_onnx_stable_diffusion_upscale.py`

```py
# 版权声明，标明版权和许可信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 按照 Apache 许可证版本 2.0（"许可证"）授权； 
# 除非遵循该许可证，否则不可使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非根据适用法律或书面协议另有约定， 
# 否则根据许可证分发的软件是基于“原样”提供的，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证具体条款的信息，见许可证。
import inspect  # 导入inspect模块，用于获取对象的信息
from typing import Any, Callable, List, Optional, Union  # 导入类型注解

import numpy as np  # 导入numpy库，用于数值计算
import PIL.Image  # 导入PIL库，用于图像处理
import torch  # 导入PyTorch库，用于深度学习
from transformers import CLIPImageProcessor, CLIPTokenizer  # 从transformers库导入图像处理和标记器

from ...configuration_utils import FrozenDict  # 导入FrozenDict，用于配置管理
from ...schedulers import DDPMScheduler, KarrasDiffusionSchedulers  # 导入调度器，用于模型训练
from ...utils import deprecate, logging  # 导入工具模块，用于日志记录和弃用警告
from ..onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel  # 导入ONNX相关工具和模型类
from ..pipeline_utils import DiffusionPipeline  # 导入扩散管道类
from . import StableDiffusionPipelineOutput  # 导入稳定扩散管道输出类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def preprocess(image):  # 定义预处理函数，接收图像作为输入
    if isinstance(image, torch.Tensor):  # 检查图像是否为PyTorch张量
        return image  # 如果是，则直接返回
    elif isinstance(image, PIL.Image.Image):  # 检查图像是否为PIL图像
        image = [image]  # 将其封装为列表

    if isinstance(image[0], PIL.Image.Image):  # 检查列表中的第一个元素是否为PIL图像
        w, h = image[0].size  # 获取图像的宽度和高度
        w, h = (x - x % 64 for x in (w, h))  # 调整宽高，使其为64的整数倍

        image = [np.array(i.resize((w, h)))[None, :] for i in image]  # 调整所有图像大小并转为数组
        image = np.concatenate(image, axis=0)  # 将所有图像数组沿第0轴合并
        image = np.array(image).astype(np.float32) / 255.0  # 转换为浮点数并归一化到[0, 1]
        image = image.transpose(0, 3, 1, 2)  # 变换数组维度为[batch, channels, height, width]
        image = 2.0 * image - 1.0  # 将值归一化到[-1, 1]
        image = torch.from_numpy(image)  # 转换为PyTorch张量
    elif isinstance(image[0], torch.Tensor):  # 如果列表中的第一个元素是PyTorch张量
        image = torch.cat(image, dim=0)  # 在第0维连接所有张量

    return image  # 返回处理后的图像


class OnnxStableDiffusionUpscalePipeline(DiffusionPipeline):  # 定义ONNX稳定扩散上采样管道类，继承自DiffusionPipeline
    vae: OnnxRuntimeModel  # 定义变分自编码器模型
    text_encoder: OnnxRuntimeModel  # 定义文本编码器模型
    tokenizer: CLIPTokenizer  # 定义CLIP标记器
    unet: OnnxRuntimeModel  # 定义U-Net模型
    low_res_scheduler: DDPMScheduler  # 定义低分辨率调度器
    scheduler: KarrasDiffusionSchedulers  # 定义Karras扩散调度器
    safety_checker: OnnxRuntimeModel  # 定义安全检查模型
    feature_extractor: CLIPImageProcessor  # 定义特征提取器

    _optional_components = ["safety_checker", "feature_extractor"]  # 可选组件列表
    _is_onnx = True  # 指示该类是否为ONNX格式

    def __init__(  # 定义构造函数
        self,
        vae: OnnxRuntimeModel,  # 变分自编码器模型
        text_encoder: OnnxRuntimeModel,  # 文本编码器模型
        tokenizer: Any,  # 任意类型的标记器
        unet: OnnxRuntimeModel,  # U-Net模型
        low_res_scheduler: DDPMScheduler,  # 低分辨率调度器
        scheduler: KarrasDiffusionSchedulers,  # Karras调度器
        safety_checker: Optional[OnnxRuntimeModel] = None,  # 可选的安全检查模型
        feature_extractor: Optional[CLIPImageProcessor] = None,  # 可选的特征提取器
        max_noise_level: int = 350,  # 最大噪声级别
        num_latent_channels=4,  # 潜在通道数量
        num_unet_input_channels=7,  # U-Net输入通道数量
        requires_safety_checker: bool = True,  # 是否需要安全检查器
    # 定义一个检查输入参数的函数，确保输入有效
        def check_inputs(
            self,  # 表示该方法属于某个类
            prompt: Union[str, List[str]],  # 输入的提示，支持字符串或字符串列表
            image,  # 输入的图像，类型不固定
            noise_level,  # 噪声级别，通常用于控制生成图像的噪声程度
            callback_steps,  # 回调步数，用于更新或监控生成过程
            negative_prompt=None,  # 可选的负面提示，控制生成内容的方向
            prompt_embeds=None,  # 可选的提示嵌入，直接传入嵌入向量
            negative_prompt_embeds=None,  # 可选的负面提示嵌入，直接传入嵌入向量
        # 定义一个准备潜在变量的函数，用于生成图像的潜在表示
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
            # 定义潜在变量的形状，根据批大小、通道数、高度和宽度
            shape = (batch_size, num_channels_latents, height, width)
            # 如果没有提供潜在变量，则生成新的随机潜在变量
            if latents is None:
                latents = generator.randn(*shape).astype(dtype)  # 从生成器中生成随机潜在变量并转换为指定数据类型
            # 如果提供的潜在变量形状不符合预期，则引发错误
            elif latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
    
            return latents  # 返回准备好的潜在变量
    
        # 定义一个解码潜在变量的函数，将潜在表示转换为图像
        def decode_latents(self, latents):
            # 调整潜在变量的尺度，以匹配解码器的输入要求
            latents = 1 / 0.08333 * latents
            # 使用变分自编码器（VAE）解码潜在变量，获取生成的图像
            image = self.vae(latent_sample=latents)[0]
            # 将图像值缩放到 [0, 1] 范围内，并进行裁剪
            image = np.clip(image / 2 + 0.5, 0, 1)
            # 调整图像的维度顺序，从 (N, C, H, W) 转换为 (N, H, W, C)
            image = image.transpose((0, 2, 3, 1))
            return image  # 返回解码后的图像
    
        # 定义一个编码提示的函数，将文本提示转换为嵌入向量
        def _encode_prompt(
            self,
            prompt: Union[str, List[str]],  # 输入的提示，支持字符串或字符串列表
            num_images_per_prompt: Optional[int],  # 每个提示生成的图像数量
            do_classifier_free_guidance: bool,  # 是否进行无分类器引导
            negative_prompt: Optional[str],  # 可选的负面提示
            prompt_embeds: Optional[np.ndarray] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[np.ndarray] = None,  # 可选的负面提示嵌入
        # 定义一个调用函数，用于生成图像
        def __call__(
            self,
            prompt: Union[str, List[str]],  # 输入的提示，支持字符串或字符串列表
            image: Union[np.ndarray, PIL.Image.Image, List[PIL.Image.Image]],  # 输入的图像，可以是 ndarray 或 PIL 图像
            num_inference_steps: int = 75,  # 推理步骤的数量，默认设置为 75
            guidance_scale: float = 9.0,  # 引导的缩放因子，控制生成图像的质量
            noise_level: int = 20,  # 噪声级别，影响生成图像的随机性
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量，默认设置为 1
            eta: float = 0.0,  # 控制随机性和确定性的超参数
            generator: Optional[Union[np.random.RandomState, List[np.random.RandomState]]] = None,  # 随机数生成器
            latents: Optional[np.ndarray] = None,  # 可选的潜在变量
            prompt_embeds: Optional[np.ndarray] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[np.ndarray] = None,  # 可选的负面提示嵌入
            output_type: Optional[str] = "pil",  # 输出类型，默认设置为 PIL 图像
            return_dict: bool = True,  # 是否以字典形式返回结果
            callback: Optional[Callable[[int, int, np.ndarray], None]] = None,  # 可选的回调函数，用于处理生成过程中的状态
            callback_steps: Optional[int] = 1,  # 回调的步数，控制回调的频率
```