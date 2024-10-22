# `.\diffusers\pipelines\pag\pipeline_pag_sd_xl.py`

```py
# 版权声明，标识本文件的版权信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可声明，说明文件使用的许可证
# Licensed under the Apache License, Version 2.0 (the "License");
# 仅在遵守许可证的情况下才能使用该文件
# you may not use this file except in compliance with the License.
# 获取许可证的链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 指定除非适用，否则以 "AS IS" 基础分发软件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 没有任何形式的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证的具体语言以及权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 Python 内置的 inspect 模块
import inspect
# 从 typing 模块导入类型注解
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIP 相关的类
from transformers import (
    CLIPImageProcessor,  # 图像处理器
    CLIPTextModel,  # 文本模型
    CLIPTextModelWithProjection,  # 带投影的文本模型
    CLIPTokenizer,  # 标记器
    CLIPVisionModelWithProjection,  # 带投影的视觉模型
)

# 从相对路径导入自定义模块
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 图像输入和变分自编码器处理器
from ...loaders import (
    FromSingleFileMixin,  # 单文件加载混合类
    IPAdapterMixin,  # IP 适配器混合类
    StableDiffusionXLLoraLoaderMixin,  # Stable Diffusion XL LoRA 加载混合类
    TextualInversionLoaderMixin,  # 文本反转加载混合类
)
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 各种模型
from ...models.attention_processor import (
    AttnProcessor2_0,  # 注意力处理器版本 2.0
    FusedAttnProcessor2_0,  # 融合注意力处理器版本 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 调整文本编码器的 LoRA 比例
from ...schedulers import KarrasDiffusionSchedulers  # Karras 扩散调度器
from ...utils import (
    USE_PEFT_BACKEND,  # 是否使用 PEFT 后端
    is_invisible_watermark_available,  # 检查隐形水印功能是否可用
    is_torch_xla_available,  # 检查 Torch XLA 是否可用
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的工具
    scale_lora_layers,  # 调整 LoRA 层
    unscale_lora_layers,  # 反调整 LoRA 层
)
from ...utils.torch_utils import randn_tensor  # 随机生成张量的工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 扩散管道和混合类
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput  # Stable Diffusion XL 输出
from .pag_utils import PAGMixin  # PAG 混合类

# 如果隐形水印可用，则导入相关的水印模块
if is_invisible_watermark_available():
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker  # Stable Diffusion XL 水印处理器

# 检查并导入 XLA 相关模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 模型模块

    XLA_AVAILABLE = True  # 设置 XLA 可用标志为真
else:
    XLA_AVAILABLE = False  # 设置 XLA 可用标志为假

# 创建日志记录器，使用模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，提供用法示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipe = AutoPipelineForText2Image.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0",
        ...     torch_dtype=torch.float16,
        ...     enable_pag=True,
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt, pag_scale=0.3).images[0]
        ```py
"""

# 从 diffusers 库复制的函数，重标定噪声配置
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 guidance_rescale 重新标定 `noise_cfg`。基于对 [Common Diffusion Noise Schedules 和
    # 文档引用，说明样本步骤存在缺陷，详细信息请参见文档第3.4节
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)  # 计算噪声预测文本的标准差，维度上保留原始形状
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)  # 计算噪声配置的标准差，维度上保留原始形状
    # 通过标准化结果来重新缩放指导结果，修正过度曝光的问题
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)  # 计算重新缩放后的噪声预测，按标准差进行调整
    # 通过引导缩放因子与原始指导结果混合，避免生成“普通”外观的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg  # 将重新缩放的结果与噪声配置混合
    return noise_cfg  # 返回最终调整后的噪声配置
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的函数
def retrieve_timesteps(
    # 调度器对象
    scheduler,
    # 用于生成样本的推理步骤数量，可选
    num_inference_steps: Optional[int] = None,
    # 要移动到的设备，可选
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步，可选
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 值，可选
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器获取时间步。处理自定义时间步。任何关键字参数将被传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            获取时间步的调度器。
        num_inference_steps (`int`):
            用于生成样本的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`，*可选*):
            要移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`，*可选*):
            自定义时间步，用于覆盖调度器的时间步间距策略。如果传递 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`，*可选*):
            自定义 sigma 值，用于覆盖调度器的时间步间距策略。如果传递 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是调度器的时间步安排，第二个元素是推理步骤的数量。
    """
    # 如果同时提供了时间步和 sigma，则抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果提供了时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，则抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果提供了 sigma
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，则抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果不是前面的条件，则执行以下代码块
        else:
            # 设置调度器的时间步长，使用指定的推理步骤数和设备，传入额外参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器的时间步长并存储到变量中
            timesteps = scheduler.timesteps
        # 返回时间步长和推理步骤数
        return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionXLPAGPipeline 的类，继承多个混合类
class StableDiffusionXLPAGPipeline(
    # 继承自 DiffusionPipeline，提供基本扩散管道功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin，提供稳定扩散特性
    StableDiffusionMixin,
    # 继承自 FromSingleFileMixin，支持从单文件加载模型
    FromSingleFileMixin,
    # 继承自 StableDiffusionXLLoraLoaderMixin，支持加载 LoRA 权重
    StableDiffusionXLLoraLoaderMixin,
    # 继承自 TextualInversionLoaderMixin，支持加载文本反转嵌入
    TextualInversionLoaderMixin,
    # 继承自 IPAdapterMixin，支持加载 IP 适配器
    IPAdapterMixin,
    # 继承自 PAGMixin，提供 PAG 相关功能
    PAGMixin,
):
    # 文档字符串，描述该管道的功能和继承关系
    r"""
    用于使用 Stable Diffusion XL 的文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解库为所有管道实现的通用方法
    （如下载或保存，特定设备上的运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    """
    # 定义函数参数的文档字符串，说明每个参数的用途和类型
        Args:
            vae ([`AutoencoderKL`]):  # 变分自编码器（VAE）模型，用于将图像编码和解码为潜在表示
                Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
            text_encoder ([`CLIPTextModel`]):  # 冻结的文本编码器，Stable Diffusion XL 使用 CLIP 的文本部分
                Frozen text-encoder. Stable Diffusion XL uses the text portion of
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)，具体是
                [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
            text_encoder_2 ([` CLIPTextModelWithProjection`]):  # 第二个冻结文本编码器，使用 CLIP 的文本和池部分
                Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection)，
                具体是
                [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
                变体。
            tokenizer (`CLIPTokenizer`):  # CLIPTokenizer 类的分词器，用于将文本转换为标记
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            tokenizer_2 (`CLIPTokenizer`):  # 第二个 CLIPTokenizer 类的分词器
                Second Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            unet ([`UNet2DConditionModel`]):  # 条件 U-Net 结构，用于去噪编码的图像潜在表示
                Conditional U-Net architecture to denoise the encoded image latents.
            scheduler ([`SchedulerMixin`]):  # 与 U-Net 结合使用的调度器，用于去噪编码的图像潜在表示
                A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
                [`DDIMScheduler`], [`LMSDiscreteScheduler`]，或 [`PNDMScheduler`]。
            force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):  # 是否强制将负提示嵌入设置为 0
                Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
                `stabilityai/stable-diffusion-xl-base-1-0`.
            add_watermarker (`bool`, *optional*):  # 是否使用不可见水印库对输出图像进行水印处理
                Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
                watermark output images. If not defined, it will default to True if the package is installed, otherwise no
                watermarker will be used.
        """
    
        # 定义模型在 CPU 上的卸载顺序，便于管理资源
        model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
        # 定义可选组件的列表，便于灵活使用不同模块
        _optional_components = [
            "tokenizer",  # 第一个分词器
            "tokenizer_2",  # 第二个分词器
            "text_encoder",  # 第一个文本编码器
            "text_encoder_2",  # 第二个文本编码器
            "image_encoder",  # 图像编码器
            "feature_extractor",  # 特征提取器
        ]
        # 定义回调的张量输入列表，方便进行数据管理
        _callback_tensor_inputs = [
            "latents",  # 潜在表示
            "prompt_embeds",  # 提示嵌入
            "negative_prompt_embeds",  # 负提示嵌入
            "add_text_embeds",  # 添加的文本嵌入
            "add_time_ids",  # 添加的时间 ID
            "negative_pooled_prompt_embeds",  # 负池化提示嵌入
            "negative_add_time_ids",  # 负添加时间 ID
        ]
    # 初始化类，设置所需参数
        def __init__(
            # 变分自编码器
            self,
            vae: AutoencoderKL,
            # 文本编码器
            text_encoder: CLIPTextModel,
            # 第二个文本编码器，带投影
            text_encoder_2: CLIPTextModelWithProjection,
            # 第一个分词器
            tokenizer: CLIPTokenizer,
            # 第二个分词器
            tokenizer_2: CLIPTokenizer,
            # UNet 2D 条件模型
            unet: UNet2DConditionModel,
            # 调度器
            scheduler: KarrasDiffusionSchedulers,
            # 可选的图像编码器
            image_encoder: CLIPVisionModelWithProjection = None,
            # 可选的特征提取器
            feature_extractor: CLIPImageProcessor = None,
            # 当提示为空时，强制使用零
            force_zeros_for_empty_prompt: bool = True,
            # 可选的水印添加标志
            add_watermarker: Optional[bool] = None,
            # 应用层的选择，默认为"mid"
            pag_applied_layers: Union[str, List[str]] = "mid",  # ["mid"],["down.block_1"],["up.block_0.attentions_0"]
        ):
            # 调用父类初始化
            super().__init__()
    
            # 注册所需模块，便于后续调用
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
            # 将配置参数注册到对象中
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 初始化图像处理器
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
            # 设置默认的采样大小
            self.default_sample_size = self.unet.config.sample_size
    
            # 如果未指定水印标志，则检查水印可用性
            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
    
            # 如果需要添加水印，则初始化水印对象
            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                # 否则将水印设置为 None
                self.watermark = None
    
            # 设置应用层
            self.set_pag_applied_layers(pag_applied_layers)
    
        # 从 StableDiffusionXLPipeline 复制的编码提示函数
        def encode_prompt(
            # 输入提示
            self,
            prompt: str,
            # 可选的第二个提示
            prompt_2: Optional[str] = None,
            # 可选的设备配置
            device: Optional[torch.device] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: int = 1,
            # 是否进行无分类器引导
            do_classifier_free_guidance: bool = True,
            # 可选的负提示
            negative_prompt: Optional[str] = None,
            # 可选的第二个负提示
            negative_prompt_2: Optional[str] = None,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的池化提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负池化提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 Lora 缩放因子
            lora_scale: Optional[float] = None,
            # 可选的剪辑跳过参数
            clip_skip: Optional[int] = None,
        # 从 StableDiffusionPipeline 复制的编码图像函数
    # 定义编码图像的方法，输入图像、设备、每个提示的图像数量及可选的隐藏状态输出
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入是否为张量，如果不是则通过特征提取器转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并转换为指定数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 获取图像编码器的隐藏状态并选择倒数第二层
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 根据每个提示数量重复隐藏状态
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建一个与图像相同形状的零张量，获取其编码器的隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 根据每个提示数量重复未条件化的隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码图像和未条件化图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要隐藏状态，直接获取图像的嵌入表示
                image_embeds = self.image_encoder(image).image_embeds
                # 根据每个提示数量重复图像嵌入
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的零张量作为未条件化图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和未条件化图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的方法
        def prepare_ip_adapter_image_embeds(
            # 定义方法参数，包括适配器图像、图像嵌入、设备、每个提示的图像数量和分类器自由引导标志
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用分类器自由引导，则初始化一个空列表用于负图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 如果输入适配器图像不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像的长度是否与 IP 适配器的数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 抛出错误，说明输入图像数量与 IP 适配器数量不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历输入适配器图像和对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 检查图像投影层是否为 ImageProjection 类型
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个适配器图像，获取嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用分类器自由引导，添加负图像嵌入
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 遍历已存在的输入适配器图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用分类器自由引导，则拆分图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负图像嵌入添加到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储最终的适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历图像嵌入的索引和内容
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入按每个提示的数量进行扩展
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用分类器自由引导，扩展负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入和正图像嵌入连接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 添加到最终的适配器图像嵌入列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回最终的适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 中复制
    # 准备调度器步骤的额外参数，因不同调度器的参数签名可能不同
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器会忽略该参数
        # eta 对应 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 应在 [0, 1] 范围内

        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 创建一个字典用于存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外步骤参数的字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.check_inputs 复制而来
    def check_inputs(
        # 定义方法所需的输入参数
        self,
        prompt,  # 主提示词
        prompt_2,  # 备用提示词
        height,  # 图像高度
        width,  # 图像宽度
        callback_steps,  # 回调步骤
        negative_prompt=None,  # 负提示词
        negative_prompt_2=None,  # 备用负提示词
        prompt_embeds=None,  # 提示词嵌入
        negative_prompt_embeds=None,  # 负提示词嵌入
        pooled_prompt_embeds=None,  # 池化的提示词嵌入
        negative_pooled_prompt_embeds=None,  # 负池化的提示词嵌入
        ip_adapter_image=None,  # 图像适配器输入
        ip_adapter_image_embeds=None,  # 图像适配器嵌入
        callback_on_step_end_tensor_inputs=None,  # 步骤结束时的张量输入
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制而来
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义拉丁变换的形状
        shape = (
            batch_size,  # 批次大小
            num_channels_latents,  # 拉丁变量通道数
            int(height) // self.vae_scale_factor,  # 高度按 VAE 缩放因子调整
            int(width) // self.vae_scale_factor,  # 宽度按 VAE 缩放因子调整
        )
        # 如果生成器是列表且其长度不等于批次大小，则抛出错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果拉丁变量为空，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果已有拉丁变量，则将其移动到指定设备
            latents = latents.to(device)

        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的拉丁变量
        return latents

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids 复制而来
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    # 生成添加时间 ID 列表，包括原始大小、裁剪坐标和目标大小
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # 计算通过添加时间嵌入维度和文本编码器投影维度得到的总维度
    passed_add_embed_dim = (
        self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    )
    # 获取模型期望的添加时间嵌入维度
    expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

    # 检查期望维度与实际维度是否一致
    if expected_add_embed_dim != passed_add_embed_dim:
        # 抛出错误，提示模型配置不正确
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    # 将添加时间 ID 转换为张量，指定数据类型
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    # 返回添加时间 ID 张量
    return add_time_ids

    # 从 StableDiffusionXLPipeline 中复制的方法，用于上采样 VAE
    def upcast_vae(self):
        # 获取 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为 float32 类型
        self.vae.to(dtype=torch.float32)
        # 检查是否使用了 Torch 2.0 或 XFormers 的注意力处理器
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # 如果使用了 XFormers 或 Torch 2.0，注意力块不需要为 float32，节省内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积层转换为指定数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将输入卷积层转换为指定数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将中间块转换为指定数据类型
            self.vae.decoder.mid_block.to(dtype)

    # 从 LatentConsistencyModelPipeline 中复制的方法，用于获取引导缩放嵌入
    def get_guidance_scale_embedding(
        # 输入参数：权重张量、嵌入维度（默认为512）、数据类型（默认为 float32）
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:  # 声明返回类型为 torch.Tensor
        """  # 开始文档字符串
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298  # 文档字符串中提供的链接
        
        Args:  # 参数说明部分
            w (`torch.Tensor`):  # 输入参数 w，类型为 torch.Tensor
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.  # 描述输入参数的作用
            embedding_dim (`int`, *optional*, defaults to 512):  # 可选参数 embedding_dim，默认值为 512
                Dimension of the embeddings to generate.  # 描述 embedding_dim 的作用
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):  # 可选参数 dtype，默认值为 torch.float32
                Data type of the generated embeddings.  # 描述 dtype 的作用

        Returns:  # 返回值说明部分
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.  # 返回一个形状为 (len(w), embedding_dim) 的 tensor
        """  # 结束文档字符串
        assert len(w.shape) == 1  # 确保 w 的形状是一维的
        w = w * 1000.0  # 将 w 的值放大 1000 倍

        half_dim = embedding_dim // 2  # 计算 embedding_dim 的一半
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)  # 计算对数缩放因子
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)  # 生成基于指数衰减的嵌入向量
        emb = w.to(dtype)[:, None] * emb[None, :]  # 将 w 转换为指定数据类型，并与 emb 进行广播相乘
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # 将正弦和余弦值沿维度 1 连接
        if embedding_dim % 2 == 1:  # 如果 embedding_dim 是奇数
            emb = torch.nn.functional.pad(emb, (0, 1))  # 在最后一维填充零以保持形状一致
        assert emb.shape == (w.shape[0], embedding_dim)  # 确保最终嵌入的形状正确
        return emb  # 返回生成的嵌入

    @property  # 将方法定义为属性
    def guidance_scale(self):  # 定义 guidance_scale 属性
        return self._guidance_scale  # 返回存储的指导比例

    @property  # 将方法定义为属性
    def guidance_rescale(self):  # 定义 guidance_rescale 属性
        return self._guidance_rescale  # 返回存储的重新调整比例

    @property  # 将方法定义为属性
    def clip_skip(self):  # 定义 clip_skip 属性
        return self._clip_skip  # 返回存储的剪辑跳过值

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)  # 说明 guidance_scale 类似于方程 (2) 中的指导权重 w
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`  # 说明 guidance_scale = 1 时无分类器自由引导
    # corresponds to doing no classifier free guidance.  # 进一步解释

    @property  # 将方法定义为属性
    def do_classifier_free_guidance(self):  # 定义 do_classifier_free_guidance 属性
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None  # 返回是否进行分类器自由引导的布尔值

    @property  # 将方法定义为属性
    def cross_attention_kwargs(self):  # 定义 cross_attention_kwargs 属性
        return self._cross_attention_kwargs  # 返回交叉注意力的关键字参数

    @property  # 将方法定义为属性
    def denoising_end(self):  # 定义 denoising_end 属性
        return self._denoising_end  # 返回去噪结束的标记

    @property  # 将方法定义为属性
    def num_timesteps(self):  # 定义 num_timesteps 属性
        return self._num_timesteps  # 返回时间步数

    @property  # 将方法定义为属性
    def interrupt(self):  # 定义 interrupt 属性
        return self._interrupt  # 返回中断标志

    @torch.no_grad()  # 指定后续操作不计算梯度
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 使用装饰器替换示例文档字符串
    # 定义可调用对象的方法，接受多个参数以生成图像或处理输入
        def __call__(
            # 主提示文本，可以是单个字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 第二个提示文本，默认为 None
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 图像的高度，默认为 None
            height: Optional[int] = None,
            # 图像的宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 指定的时间步列表，默认为 None
            timesteps: List[int] = None,
            # 噪声级别的列表，默认为 None
            sigmas: List[float] = None,
            # 去噪结束值，默认为 None
            denoising_end: Optional[float] = None,
            # 引导比例，默认为 5.0
            guidance_scale: float = 5.0,
            # 负面提示文本，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负面提示文本，默认为 None
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 额外的超参数，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在表示，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化的提示嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面池化提示嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 图像适配器的输入，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 图像适配器的嵌入列表，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 交叉注意力参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 引导重标定，默认为 0.0
            guidance_rescale: float = 0.0,
            # 原始图像的大小，默认为 None
            original_size: Optional[Tuple[int, int]] = None,
            # 图像左上角的裁剪坐标，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标大小，默认为 None
            target_size: Optional[Tuple[int, int]] = None,
            # 负面图像的原始大小，默认为 None
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负面图像左上角的裁剪坐标，默认为 (0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负面图像的目标大小，默认为 None
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 跳过的剪辑数量，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 步骤结束时的张量输入回调，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # PAG 缩放因子，默认为 3.0
            pag_scale: float = 3.0,
            # 自适应 PAG 缩放因子，默认为 0.0
            pag_adaptive_scale: float = 0.0,
```