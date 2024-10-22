# `.\diffusers\pipelines\kandinsky\pipeline_kandinsky_img2img.py`

```py
# 版权信息，表明该文件的所有权及相关许可信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版的条款提供该文件，用户需遵循此许可证
# Licensed under the Apache License, Version 2.0 (the "License");
# 仅在符合许可证的情况下才能使用此文件
# 可以在以下网址获得许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件在"原样"基础上分发，不提供任何形式的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入类型注解，定义可调用对象、列表、可选项和联合类型
from typing import Callable, List, Optional, Union

# 导入 numpy 库
import numpy as np
# 导入 PIL 库中的 Image 模块
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 PIL 导入 Image 类
from PIL import Image
# 从 transformers 库导入 XLMRobertaTokenizer
from transformers import (
    XLMRobertaTokenizer,
)

# 从相对路径导入 UNet2DConditionModel 和 VQModel
from ...models import UNet2DConditionModel, VQModel
# 从相对路径导入 DDIMScheduler
from ...schedulers import DDIMScheduler
# 从相对路径导入 logging 和 replace_example_docstring
from ...utils import (
    logging,
    replace_example_docstring,
)
# 从相对路径导入 randn_tensor
from ...utils.torch_utils import randn_tensor
# 从相对路径导入 DiffusionPipeline 和 ImagePipelineOutput
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
# 从相对路径导入 MultilingualCLIP
from .text_encoder import MultilingualCLIP

# 创建一个日志记录器，使用模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用该模块的功能
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyImg2ImgPipeline, KandinskyPriorPipeline
        >>> from diffusers.utils import load_image
        >>> import torch

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior.to("cuda")

        >>> prompt = "A red cartoon frog, 4k"
        >>> image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)

        >>> pipe = KandinskyImg2ImgPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")

        >>> init_image = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/frog.png"
        ... )

        >>> image = pipe(
        ...     prompt,
        ...     image=init_image,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=100,
        ...     strength=0.2,
        ... ).images

        >>> image[0].save("red_frog.png")
        ```py
"""

# 定义一个函数，用于计算新的高度和宽度
def get_new_h_w(h, w, scale_factor=8):
    # 根据给定的高度和缩放因子计算新的高度
    new_h = h // scale_factor**2
    # 如果高度不能被缩放因子平方整除，增加新的高度
    if h % scale_factor**2 != 0:
        new_h += 1
    # 根据给定的宽度和缩放因子计算新的宽度
    new_w = w // scale_factor**2
    # 如果宽度不能被缩放因子平方整除，增加新的宽度
    if w % scale_factor**2 != 0:
        new_w += 1
    # 返回新的高度和宽度，乘以缩放因子
    return new_h * scale_factor, new_w * scale_factor

# 定义一个函数，用于准备图像
def prepare_image(pil_image, w=512, h=512):
    # 调整 PIL 图像的大小，使用双三次插值法
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    # 将 PIL 图像转换为 NumPy 数组并转换为 RGB
    arr = np.array(pil_image.convert("RGB"))
    # 将数组数据类型转换为 float32，并归一化到 [-1, 1] 范围
    arr = arr.astype(np.float32) / 127.5 - 1
    # 调整数组维度，从 (H, W, C) 转为 (C, H, W)
    arr = np.transpose(arr, [2, 0, 1])
    # 将 NumPy 数组转换为 PyTorch 张量，并在第一个维度上添加一个维度
        image = torch.from_numpy(arr).unsqueeze(0)
        # 返回处理后的张量
        return image
# Kandinsky 图像到图像生成管道类，继承自 DiffusionPipeline
class KandinskyImg2ImgPipeline(DiffusionPipeline):
    """
    使用 Kandinsky 进行图像生成的管道

    此模型继承自 [`DiffusionPipeline`]。查看超类文档以获取库为所有管道实现的通用方法（如下载或保存、在特定设备上运行等）。

    参数:
        text_encoder ([`MultilingualCLIP`]):
            冻结的文本编码器。
        tokenizer ([`XLMRobertaTokenizer`]):
            词汇表的类。
        scheduler ([`DDIMScheduler`]):
            用于与 `unet` 结合生成图像潜在值的调度器。
        unet ([`UNet2DConditionModel`]):
            用于去噪图像嵌入的条件 U-Net 架构。
        movq ([`VQModel`]):
            MoVQ 图像编码器和解码器。
    """

    # 定义模型的 CPU 卸载顺序
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

        # 注册模块，包括文本编码器、分词器、U-Net、调度器和 MoVQ
        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        # 设置 MoVQ 的缩放因子
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    def get_timesteps(self, num_inference_steps, strength, device):
        # 使用 init_timestep 获取原始时间步
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始时间步
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中获取时间步
        timesteps = self.scheduler.timesteps[t_start:]

        # 返回时间步和剩余推理步骤
        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, latents, latent_timestep, shape, dtype, device, generator, scheduler):
        # 如果没有潜在值，则生成随机潜在值
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查潜在值的形状是否符合预期
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在值转移到目标设备
            latents = latents.to(device)

        # 将潜在值乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma

        # 获取潜在值的形状
        shape = latents.shape
        # 生成与潜在值相同形状的随机噪声
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 向潜在值添加噪声
        latents = self.add_noise(latents, noise, latent_timestep)
        # 返回处理后的潜在值
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    #  添加噪声的方法覆盖调度器中的同名方法，因为它使用不同的 beta 调度进行添加噪声与采样
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    # 返回一个张量类型的输出
    ) -> torch.Tensor:
        # 生成 0.0001 到 0.02 之间的 1000 个等间隔的 beta 值
        betas = torch.linspace(0.0001, 0.02, 1000, dtype=torch.float32)
        # 计算 alpha 值，等于 1 减去 beta 值
        alphas = 1.0 - betas
        # 计算 alpha 的累积乘积
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # 将累积乘积转换为原样本的设备和数据类型
        alphas_cumprod = alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        # 将时间步转换为原样本的设备
        timesteps = timesteps.to(original_samples.device)
    
        # 计算 sqrt(alpha) 的乘积，取出对应时间步的值
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将其展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果维度少于原样本，增加维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
        # 计算 sqrt(1 - alpha) 的乘积，取出对应时间步的值
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将其展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果维度少于原样本，增加维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
        # 根据加权公式生成带噪声的样本
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    
        # 返回生成的带噪声的样本
        return noisy_samples
    
    # 装饰器，表示不需要计算梯度
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 接收的提示，可以是字符串或字符串列表
        self,
        prompt: Union[str, List[str]],
        # 接收的图像，可以是张量或 PIL 图像
        image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
        # 图像的嵌入表示
        image_embeds: torch.Tensor,
        # 负图像的嵌入表示
        negative_image_embeds: torch.Tensor,
        # 可选的负提示
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 输出图像的高度，默认 512
        height: int = 512,
        # 输出图像的宽度，默认 512
        width: int = 512,
        # 推理步骤的数量，默认 100
        num_inference_steps: int = 100,
        # 强度参数，默认 0.3
        strength: float = 0.3,
        # 指导比例，默认 7.0
        guidance_scale: float = 7.0,
        # 每个提示生成的图像数量，默认 1
        num_images_per_prompt: int = 1,
        # 可选的生成器
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 输出类型，默认是 "pil"
        output_type: Optional[str] = "pil",
        # 可选的回调函数
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调的步骤间隔，默认 1
        callback_steps: int = 1,
        # 是否返回字典，默认 True
        return_dict: bool = True,
```