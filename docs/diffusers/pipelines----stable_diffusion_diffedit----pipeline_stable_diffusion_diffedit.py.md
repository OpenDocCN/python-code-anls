# `.\diffusers\pipelines\stable_diffusion_diffedit\pipeline_stable_diffusion_diffedit.py`

```py
# 版权声明，表明代码的作者及版权信息
# Copyright 2024 DiffEdit Authors and Pix2Pix Zero Authors and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版进行授权（“许可证”）； 
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意， 
# 否则根据许可证分发的软件是在“按现状”基础上分发的， 
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证所管辖的权限和限制的具体信息，请参阅许可证。
import inspect  # 导入 inspect 模块，用于获取有关对象的信息
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器，用于简化类定义
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示相关的类型

import numpy as np  # 导入 numpy 库，用于数值计算
import PIL.Image  # 导入 PIL.Image，用于图像处理
import torch  # 导入 PyTorch 库，用于深度学习
from packaging import version  # 导入 version 模块，用于版本控制
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer  # 导入 CLIP 相关模型和处理器

from ...configuration_utils import FrozenDict  # 从上层模块导入 FrozenDict，用于配置管理
from ...image_processor import VaeImageProcessor  # 从上层模块导入 VaeImageProcessor，用于图像处理
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入混合类用于加载器
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 LORA 规模的函数
from ...schedulers import DDIMInverseScheduler, KarrasDiffusionSchedulers  # 导入调度器
from ...utils import (  # 导入各种工具函数和常量
    PIL_INTERPOLATION,  # PIL 图像插值常量
    USE_PEFT_BACKEND,  # 使用 PEFT 后端的常量
    BaseOutput,  # 基础输出类
    deprecate,  # 用于标记已弃用的函数
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 缩放 LORA 层的函数
    unscale_lora_layers,  # 取消缩放 LORA 层的函数
)
from ...utils.torch_utils import randn_tensor  # 从工具模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合类
from ..stable_diffusion import StableDiffusionPipelineOutput  # 导入稳定扩散管道输出类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入安全检查器

logger = logging.get_logger(__name__)  # 创建一个名为当前模块的日志记录器，便于调试和信息记录

@dataclass  # 使用 dataclass 装饰器定义一个数据类
class DiffEditInversionPipelineOutput(BaseOutput):  # 继承基础输出类
    """
    Stable Diffusion 管道的输出类。

    参数：
        latents (`torch.Tensor`)
            反转的潜变量张量
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            一个 PIL 图像的列表，长度为 `num_timesteps * batch_size` 或形状为 `(num_timesteps,
            batch_size, height, width, num_channels)` 的 numpy 数组。PIL 图像或 numpy 数组表示
            扩散管道的去噪图像。
    """

    latents: torch.Tensor  # 定义潜变量属性，类型为张量
    images: Union[List[PIL.Image.Image], np.ndarray]  # 定义图像属性，类型为图像列表或 numpy 数组
# 示例文档字符串，包含代码示例
EXAMPLE_DOC_STRING = """

        ```py
        >>> import PIL  # 导入PIL库用于图像处理
        >>> import requests  # 导入requests库用于HTTP请求
        >>> import torch  # 导入PyTorch库用于深度学习
        >>> from io import BytesIO  # 从io模块导入BytesIO类用于字节流处理

        >>> from diffusers import StableDiffusionDiffEditPipeline  # 从diffusers库导入StableDiffusionDiffEditPipeline类


        >>> def download_image(url):  # 定义下载图像的函数，接受URL作为参数
        ...     response = requests.get(url)  # 使用requests库获取指定URL的响应
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")  # 将响应内容转为字节流，打开为图像并转换为RGB模式


        >>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"  # 图像的URL

        >>> init_image = download_image(img_url).resize((768, 768))  # 下载图像并调整大小为768x768

        >>> pipeline = StableDiffusionDiffEditPipeline.from_pretrained(  # 从预训练模型加载StableDiffusionDiffEditPipeline
        ...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16  # 指定模型名称和数据类型为float16
        ... )

        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)  # 设置调度器为DDIM调度器
        >>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)  # 设置逆调度器为DDIM逆调度器
        >>> pipeline.enable_model_cpu_offload()  # 启用模型的CPU卸载以节省内存

        >>> mask_prompt = "A bowl of fruits"  # 定义遮罩提示词
        >>> prompt = "A bowl of pears"  # 定义生成提示词

        >>> mask_image = pipeline.generate_mask(image=init_image, source_prompt=prompt, target_prompt=mask_prompt)  # 生成遮罩图像
        >>> image_latents = pipeline.invert(image=init_image, prompt=mask_prompt).latents  # 对初始图像进行反向处理，获取潜在图像
        >>> image = pipeline(prompt=prompt, mask_image=mask_image, image_latents=image_latents).images[0]  # 生成最终图像
        ```py
"""

# 反转示例文档字符串，包含代码示例
EXAMPLE_INVERT_DOC_STRING = """
        ```py
        >>> import PIL  # 导入PIL库用于图像处理
        >>> import requests  # 导入requests库用于HTTP请求
        >>> import torch  # 导入PyTorch库用于深度学习
        >>> from io import BytesIO  # 从io模块导入BytesIO类用于字节流处理

        >>> from diffusers import StableDiffusionDiffEditPipeline  # 从diffusers库导入StableDiffusionDiffEditPipeline类


        >>> def download_image(url):  # 定义下载图像的函数，接受URL作为参数
        ...     response = requests.get(url)  # 使用requests库获取指定URL的响应
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")  # 将响应内容转为字节流，打开为图像并转换为RGB模式


        >>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"  # 图像的URL

        >>> init_image = download_image(img_url).resize((768, 768))  # 下载图像并调整大小为768x768

        >>> pipeline = StableDiffusionDiffEditPipeline.from_pretrained(  # 从预训练模型加载StableDiffusionDiffEditPipeline
        ...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16  # 指定模型名称和数据类型为float16
        ... )

        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)  # 设置调度器为DDIM调度器
        >>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)  # 设置逆调度器为DDIM逆调度器
        >>> pipeline.enable_model_cpu_offload()  # 启用模型的CPU卸载以节省内存

        >>> prompt = "A bowl of fruits"  # 定义生成提示词

        >>> inverted_latents = pipeline.invert(image=init_image, prompt=prompt).latents  # 对初始图像进行反向处理，获取潜在图像
        ```py
"""


def auto_corr_loss(hidden_states, generator=None):  # 定义自相关损失函数，接受隐藏状态和生成器作为参数
    reg_loss = 0.0  # 初始化正则化损失为0.0
    # 遍历隐藏状态的第一个维度
        for i in range(hidden_states.shape[0]):
            # 遍历隐藏状态的第二个维度
            for j in range(hidden_states.shape[1]):
                # 提取当前隐藏状态的一个子块
                noise = hidden_states[i : i + 1, j : j + 1, :, :]
                # 进入循环以处理噪声
                while True:
                    # 随机选择一个滚动的数量
                    roll_amount = torch.randint(noise.shape[2] // 2, (1,), generator=generator).item()
                    # 计算第一个方向的正则化损失并累加
                    reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
                    # 计算第二个方向的正则化损失并累加
                    reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2
    
                    # 如果噪声的宽度小于等于8，退出循环
                    if noise.shape[2] <= 8:
                        break
                    # 对噪声进行平均池化处理
                    noise = torch.nn.functional.avg_pool2d(noise, kernel_size=2)
        # 返回计算的正则化损失
        return reg_loss
# 计算隐藏状态的 Kullback-Leibler 散度
def kl_divergence(hidden_states):
    # 计算隐藏状态的方差，加上均值的平方，再减去1，再减去方差加上一个小常数的对数
    return hidden_states.var() + hidden_states.mean() ** 2 - 1 - torch.log(hidden_states.var() + 1e-7)


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess 复制而来
def preprocess(image):
    # 定义弃用信息，提示用户使用新的预处理方法
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 调用弃用函数，输出警告信息
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 如果输入是 PyTorch 张量，则直接返回
    if isinstance(image, torch.Tensor):
        return image
    # 如果输入是 PIL 图像，将其放入列表中
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    # 如果列表中的第一个元素是 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽和高
        w, h = image[0].size
        # 将宽和高调整为8的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        # 将每个图像调整为新大小，并转换为 NumPy 数组
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 沿第0维连接所有图像数组
        image = np.concatenate(image, axis=0)
        # 将数据类型转换为 float32，并归一化到[0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组维度顺序
        image = image.transpose(0, 3, 1, 2)
        # 将像素值缩放到[-1, 1]范围
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果列表中的第一个元素是 PyTorch 张量
    elif isinstance(image[0], torch.Tensor):
        # 沿第0维连接所有张量
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image


def preprocess_mask(mask, batch_size: int = 1):
    # 如果输入的 mask 不是 PyTorch 张量
    if not isinstance(mask, torch.Tensor):
        # 处理 mask
        # 如果是 PIL 图像或 NumPy 数组，将其放入列表中
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        # 如果 mask 是列表
        if isinstance(mask, list):
            # 如果列表中的第一个元素是 PIL 图像
            if isinstance(mask[0], PIL.Image.Image):
                # 将每个图像转换为灰度并归一化到[0, 1]
                mask = [np.array(m.convert("L")).astype(np.float32) / 255.0 for m in mask]
            # 如果列表中的第一个元素是 NumPy 数组
            if isinstance(mask[0], np.ndarray):
                # 根据维度堆叠或连接数组
                mask = np.stack(mask, axis=0) if mask[0].ndim < 3 else np.concatenate(mask, axis=0)
                # 将 NumPy 数组转换为 PyTorch 张量
                mask = torch.from_numpy(mask)
            # 如果列表中的第一个元素是 PyTorch 张量
            elif isinstance(mask[0], torch.Tensor):
                # 堆叠或连接张量
                mask = torch.stack(mask, dim=0) if mask[0].ndim < 3 else torch.cat(mask, dim=0)

    # 如果 mask 是二维，添加批次和通道维度
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)

    # 如果 mask 是三维
    if mask.ndim == 3:
        # 如果是单一批次的 mask，且没有通道维度或单一 mask 但有通道维度
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)

        # 对于没有通道维度的批次 mask
        else:
            mask = mask.unsqueeze(1)

    # 检查 mask 的形状
    if batch_size > 1:
        # 如果 mask 只有一个元素，复制以匹配 batch_size
        if mask.shape[0] == 1:
            mask = torch.cat([mask] * batch_size)
        # 如果 mask 的形状与 batch_size 不一致，则引发错误
        elif mask.shape[0] > 1 and mask.shape[0] != batch_size:
            raise ValueError(
                f"`mask_image` with batch size {mask.shape[0]} cannot be broadcasted to batch size {batch_size} "
                f"inferred by prompt inputs"
            )

    # 检查 mask 是否具有单通道
    if mask.shape[1] != 1:
        raise ValueError(f"`mask_image` must have 1 channel, but has {mask.shape[1]} channels")

    # 检查 mask 的值是否在 [0, 1] 之间
    # 检查掩码的最小值是否小于0，或最大值是否大于1
    if mask.min() < 0 or mask.max() > 1:
        # 如果条件满足，抛出值错误异常，提示掩码图像应在[0, 1]范围内
        raise ValueError("`mask_image` should be in [0, 1] range")

    # 二值化掩码，低于0.5的值设为0
    mask[mask < 0.5] = 0
    # 大于等于0.5的值设为1
    mask[mask >= 0.5] = 1

    # 返回处理后的掩码
    return mask
# 定义一个稳定扩散图像编辑管道类，继承多个混入类
class StableDiffusionDiffEditPipeline(
    # 继承扩散管道、稳定扩散和文本逆转加载的功能
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin
):
    r"""
    <Tip warning={true}>
    # 提示用户该特性是实验性的
    This is an experimental feature!
    </Tip>

    # 使用稳定扩散和DiffEdit进行文本引导的图像修补的管道。
    Pipeline for text-guided image inpainting using Stable Diffusion and DiffEdit.

    # 该模型继承自DiffusionPipeline。检查超类文档以获取所有管道实现的通用方法（下载、保存、在特定设备上运行等）。
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    # 该管道还继承以下加载和保存方法：
        # 加载文本逆转嵌入的方法
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        # 加载和保存LoRA权重的方法
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights

    # 构造函数参数说明：
        vae ([`AutoencoderKL`]):
            # 变分自编码器模型，用于将图像编码和解码为潜在表示。
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            # 冻结的文本编码器，使用特定的CLIP模型。
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # 用于将文本进行分词的CLIP分词器。
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            # 用于去噪编码后图像潜在表示的UNet模型。
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            # 与UNet结合使用的调度器，用于去噪图像潜在表示。
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        inverse_scheduler ([`DDIMInverseScheduler`]):
            # 与UNet结合使用的调度器，用于填补输入潜在表示的未掩蔽部分。
            A scheduler to be used in combination with `unet` to fill in the unmasked part of the input latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            # 用于评估生成图像是否可能被视为冒犯或有害的分类模块。
            Classification module that estimates whether generated images could be considered offensive or harmful.
            # 参考模型卡以获取关于模型潜在危害的更多细节。
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            # 用于提取生成图像特征的CLIP图像处理器；作为输入传递给安全检查器。
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    # 定义模型的CPU卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件
    _optional_components = ["safety_checker", "feature_extractor", "inverse_scheduler"]
    # 定义排除在CPU卸载之外的组件
    _exclude_from_cpu_offload = ["safety_checker"]

    # 构造函数
    def __init__(
        # 定义构造函数的参数，包括各种模型
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        inverse_scheduler: DDIMInverseScheduler,
        requires_safety_checker: bool = True,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制而来
    def _encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 指定的设备（如 CPU 或 GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的 LORA 缩放因子
            **kwargs,  # 其他任意关键字参数
    ):
            # 生成弃用消息，提示用户使用新的 encode_prompt() 函数
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用弃用函数警告
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt() 以获取提示嵌入的元组
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,  # 传递提示文本
                device=device,  # 传递设备
                num_images_per_prompt=num_images_per_prompt,  # 传递每个提示的图像数量
                do_classifier_free_guidance=do_classifier_free_guidance,  # 传递无分类器引导参数
                negative_prompt=negative_prompt,  # 传递负面提示文本
                prompt_embeds=prompt_embeds,  # 传递提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,  # 传递负面提示嵌入
                lora_scale=lora_scale,  # 传递 LORA 缩放因子
                **kwargs,  # 传递其他参数
            )
    
            # 将提示嵌入元组中的两个部分连接为一个张量
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回合并后的提示嵌入
            return prompt_embeds
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制而来
    def encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 指定的设备（如 CPU 或 GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的 LORA 缩放因子
            clip_skip: Optional[int] = None,  # 可选的剪辑跳过参数
    ):
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制而来
        def run_safety_checker(self, image, device, dtype):  # 定义运行安全检查器的函数
            # 如果没有安全检查器，则将 NSFW 概念标记为 None
            if self.safety_checker is None:
                has_nsfw_concept = None
            else:
                # 如果输入图像是张量，进行后处理为 PIL 格式
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果不是张量，则将 NumPy 数组转换为 PIL 格式
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 获取安全检查器的输入，将其转换为张量并移到指定设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 运行安全检查器，检查图像的 NSFW 概念
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像和 NSFW 概念标记
            return image, has_nsfw_concept
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制而来
    # 定义一个方法来准备额外的参数，用于调度器的步骤
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外关键字参数，因为并非所有调度器的签名相同
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 其值应在 [0, 1] 之间

        # 检查调度器的步骤函数是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个空字典以存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta 参数，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤函数是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator 参数，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的方法，用于解码潜在变量
    def decode_latents(self, latents):
        # 定义弃用消息，说明 decode_latents 方法将于 1.0.0 版本中移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用函数，发出警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 使用 VAE 配置的缩放因子来调整潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量，返回图像数据
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像值从 [-1, 1] 范围转换到 [0, 1] 范围，并限制其范围
        image = (image / 2 + 0.5).clamp(0, 1)
        # 始终转换为 float32，因为这不会造成显著开销且与 bfloat16 兼容
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回解码后的图像
        return image

    # 定义一个方法检查输入参数
    def check_inputs(
        self,
        prompt,  # 输入提示
        strength,  # 强度参数
        callback_steps,  # 回调步骤数
        negative_prompt=None,  # 可选的负面提示
        prompt_embeds=None,  # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负面提示嵌入
    ):
        # 检查 strength 参数是否为 None 或者不在 [0, 1] 范围内
        if (strength is None) or (strength is not None and (strength < 0 or strength > 1)):
            # 如果不符合条件，抛出 ValueError 异常，并给出详细错误信息
            raise ValueError(
                f"The value of `strength` should in [0.0, 1.0] but is, but is {strength} of type {type(strength)}."
            )

        # 检查 callback_steps 参数是否为 None 或者不符合条件（非正整数）
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            # 如果不符合条件，抛出 ValueError 异常，并给出详细错误信息
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查同时传入 prompt 和 prompt_embeds 是否为 None
        if prompt is not None and prompt_embeds is not None:
            # 如果同时传入，抛出 ValueError 异常
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否均为 None
        elif prompt is None and prompt_embeds is None:
            # 如果均为 None，抛出 ValueError 异常
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 类型是否为 str 或 list
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果不符合条件，抛出 ValueError 异常
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查同时传入 negative_prompt 和 negative_prompt_embeds 是否为 None
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果同时传入，抛出 ValueError 异常
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否同时不为 None
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            # 检查它们的形状是否相同
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 如果形状不同，抛出 ValueError 异常，并给出详细错误信息
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # 定义 check_source_inputs 方法，检查源输入的有效性
    def check_source_inputs(
        self,
        # 定义 source_prompt 参数，默认为 None
        source_prompt=None,
        # 定义 source_negative_prompt 参数，默认为 None
        source_negative_prompt=None,
        # 定义 source_prompt_embeds 参数，默认为 None
        source_prompt_embeds=None,
        # 定义 source_negative_prompt_embeds 参数，默认为 None
        source_negative_prompt_embeds=None,
    ):
        # 检查 source_prompt 和 source_prompt_embeds 是否同时提供
        if source_prompt is not None and source_prompt_embeds is not None:
            # 抛出错误，提示不能同时传递这两个参数
            raise ValueError(
                f"Cannot forward both `source_prompt`: {source_prompt} and `source_prompt_embeds`: {source_prompt_embeds}."
                "  Please make sure to only forward one of the two."
            )
        # 检查是否同时未提供 source_prompt 和 source_prompt_embeds
        elif source_prompt is None and source_prompt_embeds is None:
            # 抛出错误，提示至少要提供一个参数
            raise ValueError(
                "Provide either `source_image` or `source_prompt_embeds`. Cannot leave all both of the arguments undefined."
            )
        # 检查 source_prompt 是否不是字符串或列表
        elif source_prompt is not None and (
            not isinstance(source_prompt, str) and not isinstance(source_prompt, list)
        ):
            # 抛出错误，提示 source_prompt 类型错误
            raise ValueError(f"`source_prompt` has to be of type `str` or `list` but is {type(source_prompt)}")

        # 检查 source_negative_prompt 和 source_negative_prompt_embeds 是否同时提供
        if source_negative_prompt is not None and source_negative_prompt_embeds is not None:
            # 抛出错误，提示不能同时传递这两个参数
            raise ValueError(
                f"Cannot forward both `source_negative_prompt`: {source_negative_prompt} and `source_negative_prompt_embeds`:"
                f" {source_negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 source_prompt_embeds 和 source_negative_prompt_embeds 是否同时提供且形状不同
        if source_prompt_embeds is not None and source_negative_prompt_embeds is not None:
            if source_prompt_embeds.shape != source_negative_prompt_embeds.shape:
                # 抛出错误，提示两个参数的形状不匹配
                raise ValueError(
                    "`source_prompt_embeds` and `source_negative_prompt_embeds` must have the same shape when passed"
                    f" directly, but got: `source_prompt_embeds` {source_prompt_embeds.shape} !="
                    f" `source_negative_prompt_embeds` {source_negative_prompt_embeds.shape}."
                )

    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步长，使用 num_inference_steps 和 strength
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始时间步长，确保不小于零
        t_start = max(num_inference_steps - init_timestep, 0)
        # 获取调度器的时间步长切片
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # 返回时间步长和剩余的推理步骤
        return timesteps, num_inference_steps - t_start

    def get_inverse_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步长，使用 num_inference_steps 和 strength
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始时间步长，确保不小于零
        t_start = max(num_inference_steps - init_timestep, 0)

        # 安全检查以防止 t_start 溢出，避免空切片
        if t_start == 0:
            return self.inverse_scheduler.timesteps, num_inference_steps
        # 获取逆调度器的时间步长切片
        timesteps = self.inverse_scheduler.timesteps[:-t_start]

        # 返回时间步长和剩余的推理步骤
        return timesteps, num_inference_steps - t_start

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 中复制的
    # 准备潜在变量，返回调整后的张量
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，根据输入参数计算
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且长度与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 若不匹配，抛出值错误
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果潜在变量为 None，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果潜在变量不为 None，则将其移动到指定设备
            latents = latents.to(device)
    
        # 将初始噪声按调度器所需的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents
    # 准备图像的潜在表示，接受图像、批次大小、数据类型、设备和可选生成器
    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        # 检查输入的 image 是否为有效类型（torch.Tensor、PIL.Image.Image 或 list）
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                # 如果类型不匹配，则抛出错误并提示实际类型
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # 将图像数据转移到指定的设备上，并转换为指定的数据类型
        image = image.to(device=device, dtype=dtype)

        # 如果图像的通道数为4，直接使用该图像作为潜在表示
        if image.shape[1] == 4:
            latents = image

        else:
            # 如果生成器是列表且其长度与批次大小不匹配，抛出错误
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    # 抛出错误，提示生成器列表的长度与请求的批次大小不匹配
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # 如果生成器是列表，则逐个处理图像
            if isinstance(generator, list):
                # 对于每个图像，编码并从对应生成器中采样潜在表示
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                # 将潜在表示沿第0维度拼接成一个张量
                latents = torch.cat(latents, dim=0)
            else:
                # 如果生成器不是列表，直接编码图像并采样潜在表示
                latents = self.vae.encode(image).latent_dist.sample(generator)

            # 根据配置的缩放因子调整潜在表示
            latents = self.vae.config.scaling_factor * latents

        # 检查生成的潜在表示与请求的批次大小是否匹配
        if batch_size != latents.shape[0]:
            # 如果请求的批次大小可以整除当前潜在表示的大小
            if batch_size % latents.shape[0] == 0:
                # 扩展潜在表示以匹配批次大小
                deprecation_message = (
                    # 构造警告消息，提示用户图像数量与文本提示数量不匹配
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                # 发出过时警告
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                # 计算每个图像需要复制的次数
                additional_latents_per_image = batch_size // latents.shape[0]
                # 复制潜在表示以满足批次大小
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                # 如果无法按请求大小复制潜在表示，抛出错误
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            # 如果匹配，则将潜在表示转换为张量形式
            latents = torch.cat([latents], dim=0)

        # 返回处理后的潜在表示
        return latents
    # 根据模型输出、样本和时间步长获取 epsilon 值
        def get_epsilon(self, model_output: torch.Tensor, sample: torch.Tensor, timestep: int):
            # 获取预测类型配置
            pred_type = self.inverse_scheduler.config.prediction_type
            # 获取时间步长对应的 alpha 乘积值
            alpha_prod_t = self.inverse_scheduler.alphas_cumprod[timestep]
    
            # 计算 beta 乘积值
            beta_prod_t = 1 - alpha_prod_t
    
            # 根据预测类型返回不同的计算结果
            if pred_type == "epsilon":
                return model_output
            elif pred_type == "sample":
                # 根据样本和模型输出计算并返回生成样本
                return (sample - alpha_prod_t ** (0.5) * model_output) / beta_prod_t ** (0.5)
            elif pred_type == "v_prediction":
                # 根据 alpha 和 beta 乘积值返回加权模型输出和样本
                return (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
            else:
                # 抛出错误，指明无效的预测类型
                raise ValueError(
                    f"prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`"
                )
    
        # 不计算梯度
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 生成掩码的函数定义
        def generate_mask(
            # 输入图像，可以是张量或PIL图像
            image: Union[torch.Tensor, PIL.Image.Image] = None,
            # 目标提示，单个字符串或字符串列表
            target_prompt: Optional[Union[str, List[str]]] = None,
            # 目标负提示，单个字符串或字符串列表
            target_negative_prompt: Optional[Union[str, List[str]]] = None,
            # 目标提示的嵌入表示，张量形式
            target_prompt_embeds: Optional[torch.Tensor] = None,
            # 目标负提示的嵌入表示，张量形式
            target_negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 源提示，单个字符串或字符串列表
            source_prompt: Optional[Union[str, List[str]]] = None,
            # 源负提示，单个字符串或字符串列表
            source_negative_prompt: Optional[Union[str, List[str]]] = None,
            # 源提示的嵌入表示，张量形式
            source_prompt_embeds: Optional[torch.Tensor] = None,
            # 源负提示的嵌入表示，张量形式
            source_negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 每个掩码生成的映射数量
            num_maps_per_mask: Optional[int] = 10,
            # 掩码编码强度
            mask_encode_strength: Optional[float] = 0.5,
            # 掩码阈值比例
            mask_thresholding_ratio: Optional[float] = 3.0,
            # 推理步骤数
            num_inference_steps: int = 50,
            # 引导比例
            guidance_scale: float = 7.5,
            # 随机数生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 输出类型，默认是numpy格式
            output_type: Optional[str] = "np",
            # 跨注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 不计算梯度
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_INVERT_DOC_STRING)
        # 反转操作的函数定义
        def invert(
            # 提示内容，单个字符串或字符串列表
            prompt: Optional[Union[str, List[str]]] = None,
            # 输入图像，可以是张量或PIL图像
            image: Union[torch.Tensor, PIL.Image.Image] = None,
            # 推理步骤数
            num_inference_steps: int = 50,
            # 反向处理强度
            inpaint_strength: float = 0.8,
            # 引导比例
            guidance_scale: float = 7.5,
            # 负提示，单个字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 随机数生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 提示嵌入表示，张量形式
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入表示，张量形式
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 是否解码潜变量
            decode_latents: bool = False,
            # 输出类型，默认是PIL格式
            output_type: Optional[str] = "pil",
            # 是否返回字典格式
            return_dict: bool = True,
            # 回调函数，用于每个步骤的反馈
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调步骤间隔
            callback_steps: Optional[int] = 1,
            # 跨注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 自动相关的惩罚系数
            lambda_auto_corr: float = 20.0,
            # KL散度的惩罚系数
            lambda_kl: float = 20.0,
            # 正则化步骤数
            num_reg_steps: int = 0,
            # 自动相关的滚动次数
            num_auto_corr_rolls: int = 5,
        # 不计算梯度
    # 使用装饰器替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义可调用方法，接受多个参数以生成图像
        def __call__(
            # 提示文本，可以是字符串或字符串列表
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            # 待处理的掩码图像，可以是张量或PIL图像
            mask_image: Union[torch.Tensor, PIL.Image.Image] = None,
            # 图像潜变量，可以是张量或PIL图像
            image_latents: Union[torch.Tensor, PIL.Image.Image] = None,
            # 图像修补强度，默认值为0.8
            inpaint_strength: Optional[float] = 0.8,
            # 推理步骤数量，默认值为50
            num_inference_steps: int = 50,
            # 指导缩放因子，默认值为7.5
            guidance_scale: float = 7.5,
            # 负提示文本，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认值为1
            num_images_per_prompt: Optional[int] = 1,
            # 噪声系数，默认值为0.0
            eta: float = 0.0,
            # 随机数生成器，可以是单个生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜变量，可以是张量
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，张量类型
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入，张量类型
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为“pil”
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为True
            return_dict: bool = True,
            # 回调函数，接受步骤和张量
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调调用的步骤间隔，默认为1
            callback_steps: int = 1,
            # 跨注意力的关键字参数，可选字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 跳过的剪辑数量，默认为None
            clip_skip: int = None,
```