# `.\diffusers\pipelines\stable_diffusion_xl\pipeline_stable_diffusion_xl_instruct_pix2pix.py`

```py
# 版权声明，标识此代码的版权归属
# Copyright 2024 Harutatsu Akiyama and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 本文件只能在遵守许可证的情况下使用
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有规定，否则本软件以“原样”方式分发，不提供任何形式的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解有关许可和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块，用于检查活跃的对象
import inspect
# 导入类型提示相关的类和函数
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PIL 图像处理库
import PIL.Image
# 导入 PyTorch 库
import torch
# 导入 CLIP 模型和分词器
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# 从本地模块导入图像处理和加载器类
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin
# 从本地模型导入相关类
from ...models import AutoencoderKL, UNet2DConditionModel
# 导入注意力处理器
from ...models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    XFormersAttnProcessor,
)
# 导入 Lora 相关调整函数
from ...models.lora import adjust_lora_scale_text_encoder
# 导入扩散调度器
from ...schedulers import KarrasDiffusionSchedulers
# 从本地工具库导入常用工具函数
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
)
# 从 torch_utils 导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 导入扩散管道及其混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 导入管道输出类
from .pipeline_output import StableDiffusionXLPipelineOutput

# 检查是否可以使用隐形水印功能
if is_invisible_watermark_available():
    # 如果可用，导入水印类
    from .watermark import StableDiffusionXLWatermarker

# 检查是否可以使用 Torch XLA
if is_torch_xla_available():
    # 如果可用，导入 XLA 核心模型
    import torch_xla.core.xla_model as xm

    # 设置标志，指示 XLA 可用
    XLA_AVAILABLE = True
else:
    # 设置标志，指示 XLA 不可用
    XLA_AVAILABLE = False

# 创建一个日志记录器，用于记录当前模块的信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，包含代码示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLInstructPix2PixPipeline
        >>> from diffusers.utils import load_image

        >>> resolution = 768  # 设置图像分辨率
        >>> image = load_image(
        ...     "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
        ... ).resize((resolution, resolution))  # 加载并调整图像大小
        >>> edit_instruction = "Turn sky into a cloudy one"  # 编辑指令

        >>> pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        ...     "diffusers/sdxl-instructpix2pix-768", torch_dtype=torch.float16
        ... ).to("cuda")  # 加载预训练管道并移动到 GPU

        >>> edited_image = pipe(
        ...     prompt=edit_instruction,  # 传入编辑指令
        ...     image=image,  # 传入待编辑的图像
        ...     height=resolution,  # 设置图像高度
        ...     width=resolution,  # 设置图像宽度
        ...     guidance_scale=3.0,  # 设置引导比例
        ...     image_guidance_scale=1.5,  # 设置图像引导比例
        ...     num_inference_steps=30,  # 设置推理步数
        ... ).images[0]  # 获取编辑后的图像
        >>> edited_image  # 输出编辑后的图像
        ```py
"""
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 中复制
def retrieve_latents(
    # 输入参数：编码器输出的张量，生成器（可选），采样模式（默认为 "sample"）
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 如果编码器输出具有 latent_dist 属性且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 返回从 latent_dist 中采样的结果
        return encoder_output.latent_dist.sample(generator)
    # 如果编码器输出具有 latent_dist 属性且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的模式值
        return encoder_output.latent_dist.mode()
    # 如果编码器输出具有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 latents 属性的值
        return encoder_output.latents
    # 如果都没有，抛出 AttributeError
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 guidance_rescale 重新缩放 `noise_cfg`。基于 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的发现。见第 3.4 节
    """
    # 计算噪声预测文本的标准差，保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差，保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据文本的标准差与配置的标准差的比例，重新缩放噪声预测（修正过曝）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 通过引导缩放因子将重新缩放的结果与原始结果混合，以避免图像 "过于简单"
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的噪声配置
    return noise_cfg


class StableDiffusionXLInstructPix2PixPipeline(
    # 继承自 DiffusionPipeline 和其他混合类
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
):
    r"""
    基于文本指令的像素级图像编辑管道。基于 Stable Diffusion XL。

    该模型继承自 [`DiffusionPipeline`]。查看超类文档以了解库为所有管道实现的通用方法
    （例如下载或保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
    # 文档字符串，描述函数的参数
        Args:
            vae ([`AutoencoderKL`]):
                变分自编码器（VAE）模型，用于将图像编码和解码为潜在表示。
            text_encoder ([`CLIPTextModel`]):
                冻结的文本编码器。Stable Diffusion XL 使用
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 的文本部分，
                特别是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
            text_encoder_2 ([` CLIPTextModelWithProjection`]):
                第二个冻结文本编码器。Stable Diffusion XL 使用
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection) 的文本和池部分，
                特别是
                [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
                变体。
            tokenizer (`CLIPTokenizer`):
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 类的分词器。
            tokenizer_2 (`CLIPTokenizer`):
                第二个 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 类的分词器。
            unet ([`UNet2DConditionModel`]): 条件 U-Net 架构，用于去噪编码后的图像潜在表示。
            scheduler ([`SchedulerMixin`]):
                与 `unet` 结合使用的调度器，用于去噪编码的图像潜在表示。可以是
                [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
            requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):
                `unet` 是否需要在推理期间传递审美评分条件。另见
                `stabilityai/stable-diffusion-xl-refiner-1-0` 的配置。
            force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
                是否强制负提示嵌入始终设置为 0。另见
                `stabilityai/stable-diffusion-xl-base-1-0` 的配置。
            add_watermarker (`bool`, *optional*):
                是否使用 [invisible_watermark 库](https://github.com/ShieldMnt/invisible-watermark/) 对输出图像进行水印处理。
                如果未定义，且包已安装，则默认为 True，否则不使用水印。
            is_cosxl_edit (`bool`, *optional*):
                当设置时，图像潜在表示被缩放。
        """
    
        # 定义模型 CPU 卸载顺序的字符串
        model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
        # 定义可选组件的列表
        _optional_components = ["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2"]
    # 初始化方法，用于创建类的实例
        def __init__(
            self,
            # 自动编码器模型
            vae: AutoencoderKL,
            # 文本编码器模型
            text_encoder: CLIPTextModel,
            # 第二个文本编码器模型，带投影
            text_encoder_2: CLIPTextModelWithProjection,
            # 第一个分词器
            tokenizer: CLIPTokenizer,
            # 第二个分词器
            tokenizer_2: CLIPTokenizer,
            # 条件生成的 UNet 模型
            unet: UNet2DConditionModel,
            # Karras 扩散调度器
            scheduler: KarrasDiffusionSchedulers,
            # 是否在空提示时强制使用零
            force_zeros_for_empty_prompt: bool = True,
            # 是否添加水印，默认为 None
            add_watermarker: Optional[bool] = None,
            # 是否为 cosxl 编辑，默认为 False
            is_cosxl_edit: Optional[bool] = False,
        ):
            # 调用父类的初始化方法
            super().__init__()
    
            # 注册模型模块
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
            )
            # 将参数注册到配置中
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 设置默认样本大小
            self.default_sample_size = self.unet.config.sample_size
            # 记录是否为 cosxl 编辑
            self.is_cosxl_edit = is_cosxl_edit
    
            # 判断是否添加水印，如果未提供，则根据可用性设置
            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
    
            # 如果需要添加水印，创建水印对象
            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                # 否则设置为 None
                self.watermark = None
    
        # 编码提示的方法
        def encode_prompt(
            self,
            # 输入的提示字符串
            prompt: str,
            # 可选的第二个提示字符串
            prompt_2: Optional[str] = None,
            # 可选的设备
            device: Optional[torch.device] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: int = 1,
            # 是否进行无分类器自由引导
            do_classifier_free_guidance: bool = True,
            # 可选的负面提示字符串
            negative_prompt: Optional[str] = None,
            # 可选的第二个负面提示字符串
            negative_prompt_2: Optional[str] = None,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的池化提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面池化提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 LoRA 缩放因子
            lora_scale: Optional[float] = None,
        # 从 diffusers 中复制的额外步骤参数准备方法
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，因为并非所有调度器的签名相同
            # eta（η）仅用于 DDIMScheduler，对于其他调度器将被忽略。
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 应在 [0, 1] 之间
    
            # 检查调度器是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                # 如果接受，添加 eta 参数
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            if accepts_generator:
                # 如果接受，添加 generator 参数
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数
            return extra_step_kwargs
    # 定义一个检查输入参数的函数
        def check_inputs(
            self,  # 当前对象实例
            prompt,  # 正向提示词
            callback_steps,  # 回调步骤的频率
            negative_prompt=None,  # 负向提示词，可选
            prompt_embeds=None,  # 正向提示词的嵌入表示，可选
            negative_prompt_embeds=None,  # 负向提示词的嵌入表示，可选
            callback_on_step_end_tensor_inputs=None,  # 回调时的张量输入，可选
        ):
            # 检查 callback_steps 是否为正整数
            if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
                # 如果不是，则抛出值错误
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )
    
            # 检查 callback_on_step_end_tensor_inputs 是否在预定义的输入中
            if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
            ):
                # 如果有不在其中的，则抛出值错误
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
                )
    
            # 检查 prompt 和 prompt_embeds 是否同时提供
            if prompt is not None and prompt_embeds is not None:
                # 如果同时提供，则抛出值错误
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查 prompt 和 prompt_embeds 是否都未提供
            elif prompt is None and prompt_embeds is None:
                # 如果都未提供，则抛出值错误
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查 prompt 的类型是否正确
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                # 如果类型不正确，则抛出值错误
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查 negative_prompt 和 negative_prompt_embeds 是否同时提供
            if negative_prompt is not None and negative_prompt_embeds is not None:
                # 如果同时提供，则抛出值错误
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查 prompt_embeds 和 negative_prompt_embeds 是否同时提供
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                # 如果它们的形状不同，则抛出值错误
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的代码
    # 准备潜在变量的函数，接受批量大小、通道数、高度、宽度等参数
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，计算后高和宽根据 VAE 的缩放因子调整
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表，并验证其长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                # 抛出错误，说明生成器列表长度与请求的批量大小不一致
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜在变量，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在变量，则将其移动到指定的设备上
            latents = latents.to(device)

        # 按照调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在变量
        return latents

    # 准备图像潜在变量的函数，接受图像及其他参数
    def prepare_image_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    # 该代码块的最后一部分开始于这个方法的结束
        ):
            # 检查传入的 image 是否为有效类型，如果不是则抛出错误
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    # 报告 image 的实际类型
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )
    
            # 将 image 移动到指定设备并转换为指定数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 根据每个提示的图像数量计算批大小
            batch_size = batch_size * num_images_per_prompt
    
            # 如果 image 的通道数为 4，则直接使用它作为潜在表示
            if image.shape[1] == 4:
                image_latents = image
            else:
                # 确保 VAE 在 float32 模式下，以避免在 float16 中溢出
                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
                if needs_upcasting:
                    # 将图像转换为 float32
                    image = image.float()
                    # 执行 VAE 的上采样
                    self.upcast_vae()
    
                # 编码图像并获取潜在表示
                image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
    
                # 如果需要，将 VAE 数据类型转换回 fp16
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)
    
            # 如果批大小大于潜在表示的数量且可以整除
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # 扩展潜在表示以匹配批大小
                deprecation_message = (
                    # 提示用户传入的提示数量与图像数量不匹配
                    f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                # 发出弃用警告
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                # 计算每个提示所需的额外图像数量
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                # 复制潜在表示以匹配批大小
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            # 如果批大小大于潜在表示的数量但不能整除
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                # 抛出错误，说明无法复制图像
                raise ValueError(
                    # 报告无法复制的批大小
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                # 将潜在表示扩展到批次维度
                image_latents = torch.cat([image_latents], dim=0)
    
            # 如果使用无分类器自由引导
            if do_classifier_free_guidance:
                # 创建与潜在表示相同形状的零张量
                uncond_image_latents = torch.zeros_like(image_latents)
                # 将潜在表示与无条件潜在表示合并
                image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
    
            # 如果潜在表示的数据类型与 VAE 不匹配，则进行转换
            if image_latents.dtype != self.vae.dtype:
                image_latents = image_latents.to(dtype=self.vae.dtype)
    
            # 如果为 COSXL 编辑模式，则应用缩放因子
            if self.is_cosxl_edit:
                image_latents = image_latents * self.vae.config.scaling_factor
    
            # 返回最终的潜在表示
            return image_latents
    
        # 该方法用于获取附加时间 ID
        # 复制自 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
        def _get_add_time_ids(
            # 定义方法所需的参数
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    # 定义函数的结束部分
        ):
            # 创建包含原始大小、裁剪坐标和目标大小的列表
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
            # 计算传入的附加时间嵌入维度
            passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
            )
            # 获取模型预期的附加嵌入维度
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    
            # 检查预期和实际的嵌入维度是否一致
            if expected_add_embed_dim != passed_add_embed_dim:
                # 如果不一致，抛出错误并提供相关信息
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )
    
            # 将时间 ID 转换为张量，指定数据类型
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
            # 返回添加的时间 ID 张量
            return add_time_ids
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.upcast_vae 复制
        def upcast_vae(self):
            # 获取 VAE 的数据类型
            dtype = self.vae.dtype
            # 将 VAE 转换为 float32 类型
            self.vae.to(dtype=torch.float32)
            # 检查是否使用了 Torch 2.0 或 Xformers
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                    FusedAttnProcessor2_0,
                ),
            )
            # 如果使用 xformers 或 torch_2_0，注意力模块不需要使用 float32，以节省内存
            if use_torch_2_0_or_xformers:
                # 将后量化卷积转换为指定数据类型
                self.vae.post_quant_conv.to(dtype)
                # 将输入卷积转换为指定数据类型
                self.vae.decoder.conv_in.to(dtype)
                # 将中间块转换为指定数据类型
                self.vae.decoder.mid_block.to(dtype)
    
        # 关闭梯度计算
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法，接受多个参数进行处理
        def __call__(
            # 输入提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 第二个提示，默认为 None
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 输入图像，类型为 PipelineImageInput，默认为 None
            image: PipelineImageInput = None,
            # 图像高度，默认为 None
            height: Optional[int] = None,
            # 图像宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤数，默认为 100
            num_inference_steps: int = 100,
            # 去噪结束的浮点值，默认为 None
            denoising_end: Optional[float] = None,
            # 引导比例，默认为 5.0
            guidance_scale: float = 5.0,
            # 图像引导比例，默认为 1.5
            image_guidance_scale: float = 1.5,
            # 负面提示，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负面提示，默认为 None
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # ETA 参数，默认为 0.0
            eta: float = 0.0,
            # 随机生成器，可以是 torch.Generator 或其列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在张量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化的提示嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面的池化提示嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 返回字典的标志，默认为 True
            return_dict: bool = True,
            # 回调函数，默认为 None
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调步骤，默认为 1
            callback_steps: int = 1,
            # 交叉注意力的关键字参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 引导重标定，默认为 0.0
            guidance_rescale: float = 0.0,
            # 原始图像大小，默认为 None
            original_size: Tuple[int, int] = None,
            # 左上角裁剪坐标，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标图像大小，默认为 None
            target_size: Tuple[int, int] = None,
```