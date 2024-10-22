# `.\diffusers\pipelines\kandinsky\pipeline_kandinsky_inpaint.py`

```py
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy  # 从 copy 模块导入 deepcopy 函数，用于深拷贝对象
from typing import Callable, List, Optional, Union  # 导入类型提示功能，用于函数签名

import numpy as np  # 导入 numpy 库，通常用于数组和矩阵操作
import PIL.Image  # 导入 PIL.Image 模块，用于图像处理
import torch  # 导入 PyTorch 库，深度学习框架
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from packaging import version  # 导入 version 模块，用于版本管理
from PIL import Image  # 导入 PIL.Image 模块，用于图像操作
from transformers import (  # 从 transformers 库导入以下组件
    XLMRobertaTokenizer,  # 导入 XLM-RoBERTa 的分词器
)

from ... import __version__  # 导入当前模块的版本信息
from ...models import UNet2DConditionModel, VQModel  # 导入模型类
from ...schedulers import DDIMScheduler  # 导入 DDIM 调度器
from ...utils import (  # 从 utils 模块导入以下工具
    logging,  # 导入日志模块
    replace_example_docstring,  # 导入用于替换示例文档字符串的函数
)
from ...utils.torch_utils import randn_tensor  # 从 torch_utils 导入 randn_tensor 函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 导入扩散管道和图像管道输出类
from .text_encoder import MultilingualCLIP  # 导入多语言 CLIP 文本编码器

logger = logging.get_logger(__name__)  # 创建日志记录器，用于记录模块日志，名称为当前模块名
# pylint: disable=invalid-name  # 禁用 pylint 对无效名称的警告

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，提供代码示例
    Examples:
        ```py
        >>> from diffusers import KandinskyInpaintPipeline, KandinskyPriorPipeline  # 从 diffusers 导入管道类
        >>> from diffusers.utils import load_image  # 从 utils 导入图像加载函数
        >>> import torch  # 导入 PyTorch
        >>> import numpy as np  # 导入 numpy

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained(  # 从预训练模型创建管道
        ...     "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16  # 指定模型和数据类型
        ... )
        >>> pipe_prior.to("cuda")  # 将管道移动到 CUDA 设备

        >>> prompt = "a hat"  # 定义生成图像的提示词
        >>> image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)  # 获取图像嵌入和零图像嵌入

        >>> pipe = KandinskyInpaintPipeline.from_pretrained(  # 从预训练模型创建图像修复管道
        ...     "kandinsky-community/kandinsky-2-1-inpaint", torch_dtype=torch.float16  # 指定模型和数据类型
        ... )
        >>> pipe.to("cuda")  # 将管道移动到 CUDA 设备

        >>> init_image = load_image(  # 加载初始图像
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"  # 图像的 URL
        ...     "/kandinsky/cat.png"  # 图像文件名
        ... )

        >>> mask = np.zeros((768, 768), dtype=np.float32)  # 创建一个全零的掩膜
        >>> mask[:250, 250:-250] = 1  # 在掩膜中设置特定区域为 1

        >>> out = pipe(  # 调用管道进行图像生成
        ...     prompt,  # 提示词
        ...     image=init_image,  # 初始图像
        ...     mask_image=mask,  # 掩膜图像
        ...     image_embeds=image_emb,  # 图像嵌入
        ...     negative_image_embeds=zero_image_emb,  # 负图像嵌入
        ...     height=768,  # 输出图像的高度
        ...     width=768,  # 输出图像的宽度
        ...     num_inference_steps=50,  # 推理步骤数
        ... )

        >>> image = out.images[0]  # 获取输出的第一张图像
        >>> image.save("cat_with_hat.png")  # 保存生成的图像
        ```py
"""


def get_new_h_w(h, w, scale_factor=8):  # 定义函数，计算新的高和宽
    new_h = h // scale_factor**2  # 计算新的高度，使用整除
    if h % scale_factor**2 != 0:  # 如果原高度不能被 scale_factor^2 整除
        new_h += 1  # 高度加一
    new_w = w // scale_factor**2  # 计算新的宽度，使用整除
    if w % scale_factor**2 != 0:  # 如果原宽度不能被 scale_factor^2 整除
        new_w += 1  # 宽度加一
    return new_h * scale_factor, new_w * scale_factor  # 返回新的高和宽，均乘以 scale_factor
# 准备掩码
def prepare_mask(masks):
    # 初始化一个空列表以存储准备好的掩码
    prepared_masks = []
    # 遍历所有输入掩码
    for mask in masks:
        # 深拷贝当前掩码以避免修改原始数据
        old_mask = deepcopy(mask)
        # 遍历掩码的每一行
        for i in range(mask.shape[1]):
            #
    # 检查 image 是否为 torch.Tensor 类型
        if isinstance(image, torch.Tensor):
            # 如果 mask 不是 torch.Tensor，则抛出类型错误
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")
    
            # 如果 image 是单张图像（3 个维度）
            if image.ndim == 3:
                # 断言图像的通道数为 3，确保形状符合要求
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                # 添加一个维度以表示批量
                image = image.unsqueeze(0)
    
            # 如果 mask 是单张图像（2 个维度），则添加通道维度
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
    
            # 检查 mask 的维度
            if mask.ndim == 3:
                # 如果 mask 是单个批量图像，且没有通道维度
                if mask.shape[0] == 1:
                    mask = mask.unsqueeze(0)
    
                # 如果 mask 是批量图像且没有通道维度
                else:
                    mask = mask.unsqueeze(1)
    
            # 断言 image 和 mask 都必须有 4 个维度
            assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
            # 断言 image 和 mask 的空间维度相同
            assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
            # 断言 image 和 mask 的批量大小相同
            assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"
    
            # 检查 image 的值是否在 [-1, 1] 范围内
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")
    
            # 检查 mask 的值是否在 [0, 1] 范围内
            if mask.min() < 0 or mask.max() > 1:
                raise ValueError("Mask should be in [0, 1] range")
    
            # 将 mask 二值化
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
    
            # 将 image 转换为 float32 类型
            image = image.to(dtype=torch.float32)
        # 如果 mask 是 torch.Tensor，但 image 不是，则抛出类型错误
        elif isinstance(mask, torch.Tensor):
            raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # 处理图像
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            # 将单个图像转换为列表形式
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # 根据传入的高度和宽度调整所有图像的大小
            image = [i.resize((width, height), resample=Image.BICUBIC, reducing_gap=1) for i in image]
            # 将每个图像转换为 RGB 数组，并添加一个维度
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            # 将多个图像数组在第一个维度上连接起来
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            # 将多个 ndarray 在第一个维度上连接起来
            image = np.concatenate([i[None, :] for i in image], axis=0)

        # 调整图像维度顺序，从 (N, H, W, C) 到 (N, C, H, W)
        image = image.transpose(0, 3, 1, 2)
        # 将 numpy 数组转换为 PyTorch 张量，并进行归一化处理
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # 处理掩码
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            # 将单个掩码转换为列表形式
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            # 根据传入的高度和宽度调整所有掩码的大小
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            # 将每个掩码转换为灰度数组，并添加两个维度
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            # 将掩码数组的值归一化到 [0, 1]
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            # 将多个 ndarray 在第一个维度上连接起来
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        # 将掩码进行二值化处理，低于 0.5 设为 0，高于等于 0.5 设为 1
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        # 将 numpy 数组转换为 PyTorch 张量
        mask = torch.from_numpy(mask)

    # 将掩码取反
    mask = 1 - mask

    # 返回掩码和图像
    return mask, image
# 定义Kandinsky图像修复管道，继承自DiffusionPipeline
class KandinskyInpaintPipeline(DiffusionPipeline):
    """
    使用Kandinsky2.1进行文本引导的图像修复的管道

    该模型继承自[`DiffusionPipeline`]。请查看超类文档以获取库为所有管道实现的通用方法
    （例如下载或保存，运行在特定设备上等）。

    参数：
        text_encoder ([`MultilingualCLIP`]):
            冻结的文本编码器。
        tokenizer ([`XLMRobertaTokenizer`]):
            类的分词器
        scheduler ([`DDIMScheduler`]):
            用于与`unet`结合生成图像潜在表示的调度器。
        unet ([`UNet2DConditionModel`]):
            用于去噪图像嵌入的条件U-Net架构。
        movq ([`VQModel`]):
            MoVQ图像编码器和解码器
    """

    # 定义模型CPU卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->movq"

    def __init__(
        self,
        text_encoder: MultilingualCLIP,
        movq: VQModel,
        tokenizer: XLMRobertaTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
    ):
        # 初始化父类
        super().__init__()

        # 注册模型组件
        self.register_modules(
            text_encoder=text_encoder,
            movq=movq,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        # 计算MoVQ的缩放因子
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)
        # 初始化警告标志
        self._warn_has_been_called = False

    # 从diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline复制的方法，用于准备潜在表示
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果没有给定潜在表示，则随机生成
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查潜在表示的形状是否符合预期
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在表示转移到指定设备
            latents = latents.to(device)

        # 使用调度器的初始噪声标准差调整潜在表示
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在表示
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
    # 定义一个可调用的类方法
        def __call__(
            self,
            # 接受一个字符串或字符串列表作为提示
            prompt: Union[str, List[str]],
            # 接受一个图像张量或 PIL 图像作为输入图像
            image: Union[torch.Tensor, PIL.Image.Image],
            # 接受一个掩膜图像，可以是张量、PIL 图像或 NumPy 数组
            mask_image: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
            # 接受一个图像嵌入的张量
            image_embeds: torch.Tensor,
            # 接受一个负图像嵌入的张量
            negative_image_embeds: torch.Tensor,
            # 可选参数，接受一个字符串或字符串列表作为负提示
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 设置输出图像的高度，默认为 512
            height: int = 512,
            # 设置输出图像的宽度，默认为 512
            width: int = 512,
            # 设置推理步骤的数量，默认为 100
            num_inference_steps: int = 100,
            # 设置引导比例，默认为 4.0
            guidance_scale: float = 4.0,
            # 设置每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 可选参数，接受一个随机数生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选参数，接受潜在空间的张量
            latents: Optional[torch.Tensor] = None,
            # 可选参数，设置输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 可选参数，接受一个回调函数，用于处理图像生成的步骤
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 设置回调函数调用的步数，默认为 1
            callback_steps: int = 1,
            # 设置返回字典的布尔值，默认为 True
            return_dict: bool = True,
```