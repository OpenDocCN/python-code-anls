# `.\diffusers\pipelines\stable_diffusion_xl\pipeline_stable_diffusion_xl.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache License, Version 2.0（"许可证"）许可；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，
# 否则根据许可证分发的软件是以“按现状”基础分发的，
# 不提供任何种类的担保或条件，无论是明示或暗示的。
# 有关许可证所管理的权限和限制的具体信息，请参见许可证。

import inspect  # 导入 inspect 模块，用于获取对象的接口信息
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 从 typing 导入类型提示所需的类型

import torch  # 导入 PyTorch 库，进行张量运算和深度学习任务
from transformers import (  # 从 transformers 库导入以下组件
    CLIPImageProcessor,  # CLIP 图像处理器
    CLIPTextModel,  # CLIP 文本模型
    CLIPTextModelWithProjection,  # 带投影的 CLIP 文本模型
    CLIPTokenizer,  # CLIP 分词器
    CLIPVisionModelWithProjection,  # 带投影的 CLIP 视觉模型
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 从回调模块导入多管道回调和管道回调
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 从图像处理模块导入管道图像输入和变分自编码器图像处理器
from ...loaders import (  # 从加载器模块导入以下混合类
    FromSingleFileMixin,  # 从单文件加载的混合类
    IPAdapterMixin,  # IP 适配器混合类
    StableDiffusionXLLoraLoaderMixin,  # 稳定扩散 XL Lora 加载器混合类
    TextualInversionLoaderMixin,  # 文本反演加载器混合类
)
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 从模型模块导入自编码器、图像投影和条件 UNet 模型
from ...models.attention_processor import (  # 从注意力处理模块导入以下处理器
    AttnProcessor2_0,  # 注意力处理器版本 2.0
    FusedAttnProcessor2_0,  # 融合的注意力处理器版本 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 从 Lora 模型导入调整文本编码器的 Lora 比例函数
from ...schedulers import KarrasDiffusionSchedulers  # 从调度器模块导入 Karras 扩散调度器
from ...utils import (  # 从工具模块导入以下工具函数和常量
    USE_PEFT_BACKEND,  # 使用 PEFT 后端的标志
    deprecate,  # 用于标记过时功能的工具
    is_invisible_watermark_available,  # 检查隐形水印是否可用的函数
    is_torch_xla_available,  # 检查 Torch XLA 是否可用的函数
    logging,  # 日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的工具
    scale_lora_layers,  # 调整 Lora 层规模的函数
    unscale_lora_layers,  # 恢复 Lora 层规模的函数
)
from ...utils.torch_utils import randn_tensor  # 从 PyTorch 工具模块导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从管道工具模块导入扩散管道和稳定扩散混合类
from .pipeline_output import StableDiffusionXLPipelineOutput  # 从管道输出模块导入稳定扩散 XL 管道输出类


if is_invisible_watermark_available():  # 如果隐形水印可用
    from .watermark import StableDiffusionXLWatermarker  # 导入稳定扩散 XL 水印工具

if is_torch_xla_available():  # 如果 Torch XLA 可用
    import torch_xla.core.xla_model as xm  # 导入 XLA 模型核心

    XLA_AVAILABLE = True  # 设置 XLA 可用标志为真
else:
    XLA_AVAILABLE = False  # 设置 XLA 可用标志为假


logger = logging.get_logger(__name__)  # 创建一个记录器，使用当前模块的名称作为标识

EXAMPLE_DOC_STRING = """  # 示例文档字符串，用于展示代码示例
    Examples:  # 示例说明
        ```py  # Python 代码示例开始
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableDiffusionXLPipeline  # 从 diffusers 导入稳定扩散 XL 管道

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(  # 从预训练模型加载管道
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16  # 指定模型名称和数据类型
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道转移到 GPU

        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义生成图像的提示
        >>> image = pipe(prompt).images[0]  # 生成图像并获取第一个图像
        ```py
"""


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制的函数
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):  # 定义重标定噪声配置的函数，接受噪声配置、预测文本和指导重标定参数
    """
    根据 `guidance_rescale` 重标定 `noise_cfg`。基于[常见扩散噪声调度和样本步骤存在缺陷](https://arxiv.org/pdf/2305.08891.pdf)中的发现。见第 3.4 节
    """  # 函数文档字符串，解释功能和引用文献
    # 计算噪声预测文本的标准差，沿指定维度进行计算，并保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差，沿指定维度进行计算，并保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据标准差重新缩放来自指导的结果，以修正过度曝光问题
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 通过指导缩放因子将重新缩放的噪声与原始指导结果混合，以避免生成“普通”外观的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回最终的噪声配置
    return noise_cfg
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 复制的代码
def retrieve_timesteps(
    # 调度器对象，用于获取时间步
    scheduler,
    # 生成样本时使用的扩散步骤数量，默认为 None
    num_inference_steps: Optional[int] = None,
    # 时间步要移动到的设备，可以是字符串或 torch.device 类型，默认为 None
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步，默认为 None
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 值，默认为 None
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数，将传递给调度器的 set_timesteps 方法
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器检索时间步。处理
    自定义时间步。任何关键字参数将被提供给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`，*可选*):
            时间步要移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`，*可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递了 `timesteps`，
            则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`，*可选*):
            自定义 sigma 值，用于覆盖调度器的时间步间隔策略。如果传递了 `sigmas`，
            则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是来自调度器的时间步调度，
        第二个元素是推理步骤的数量。
    """
    # 如果同时传入了自定义时间步和 sigma，抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传入了自定义时间步
    if timesteps is not None:
        # 检查当前调度器是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持自定义时间步，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义时间步，移动到指定设备
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传入了自定义 sigma
    elif sigmas is not None:
        # 检查当前调度器是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持自定义 sigma，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义 sigma，移动到指定设备
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果不是特定条件，设置调度器的时间步长
    else:
        # 调用调度器的方法，设置推理步骤数和设备信息，同时传递其他关键字参数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取调度器的时间步长
        timesteps = scheduler.timesteps
    # 返回时间步长和推理步骤数
    return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionXLPipeline 的类，继承多个混合类以实现功能
class StableDiffusionXLPipeline(
    # 继承自 DiffusionPipeline，提供基础的扩散模型功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin，添加与稳定扩散相关的方法
    StableDiffusionMixin,
    # 继承自 FromSingleFileMixin，支持从单个文件加载模型
    FromSingleFileMixin,
    # 继承自 StableDiffusionXLLoraLoaderMixin，添加加载和保存 LoRA 权重的方法
    StableDiffusionXLLoraLoaderMixin,
    # 继承自 TextualInversionLoaderMixin，添加加载文本反转嵌入的方法
    TextualInversionLoaderMixin,
    # 继承自 IPAdapterMixin，支持加载 IP 适配器
    IPAdapterMixin,
):
    # 文档字符串，描述该类的功能
    r"""
    使用 Stable Diffusion XL 进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。查看超类文档以了解库实现的所有管道的通用方法
    （例如下载或保存，在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    # 参数说明
    Args:
        # 变分自编码器模型，用于将图像编码为潜在表示并解码回图像
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        # 冻结的文本编码器，Stable Diffusion XL 使用 CLIP 模型的文本部分
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        # 第二个冻结的文本编码器，Stable Diffusion XL 使用 CLIP 模型的文本和池化部分
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        # CLIP 的分词器，用于将文本转换为模型可处理的输入
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # 第二个 CLIP 分词器
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # 条件 U-Net 架构，用于对编码的图像潜在表示进行去噪
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        # 调度器，与 U-Net 结合使用以去噪编码的图像潜在表示
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        # 是否强制将负提示嵌入设置为 0
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        # 是否使用隐形水印库对输出图像进行水印处理
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """
    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    # 可选组件的列表，包含一些可选的模型组件
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    # 回调张量输入的列表，包含模型输入张量的名称
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]
    # 初始化方法，用于创建类的实例并设置属性
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器模型
            text_encoder: CLIPTextModel,  # 文本编码器模型
            text_encoder_2: CLIPTextModelWithProjection,  # 第二文本编码器模型
            tokenizer: CLIPTokenizer,  # 文本分词器
            tokenizer_2: CLIPTokenizer,  # 第二文本分词器
            unet: UNet2DConditionModel,  # U-Net 模型用于生成图像
            scheduler: KarrasDiffusionSchedulers,  # 扩散调度器
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选的图像编码器
            feature_extractor: CLIPImageProcessor = None,  # 可选的特征提取器
            force_zeros_for_empty_prompt: bool = True,  # 是否对空提示强制使用零
            add_watermarker: Optional[bool] = None,  # 可选的水印添加标志
        ):
            super().__init__()  # 调用父类的初始化方法
    
            # 注册模块，初始化必要的组件
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
            # 注册配置参数，包含强制零的标志
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 计算 VAE 的缩放因子，基于块输出通道的数量
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器，使用计算得出的缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
            # 设置默认采样大小，来自 U-Net 的配置
            self.default_sample_size = self.unet.config.sample_size
    
            # 如果没有提供水印标志，则根据可用性决定
            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
    
            # 根据水印标志，初始化水印器
            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()  # 初始化水印器
            else:
                self.watermark = None  # 不使用水印
    
        # 编码提示的方法，用于生成嵌入
        def encode_prompt(
            self,
            prompt: str,  # 主提示字符串
            prompt_2: Optional[str] = None,  # 可选的第二提示字符串
            device: Optional[torch.device] = None,  # 设备信息，默认为 None
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            do_classifier_free_guidance: bool = True,  # 是否进行无分类器引导
            negative_prompt: Optional[str] = None,  # 可选的负提示字符串
            negative_prompt_2: Optional[str] = None,  # 可选的第二负提示字符串
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的池化提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负池化提示嵌入
            lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
            clip_skip: Optional[int] = None,  # 可选的跳过 Clip 层
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制
    # 定义编码图像的函数，接受图像、设备、每个提示的图像数量和可选的隐藏状态
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，则使用特征提取器处理并返回张量格式
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像转移到指定设备并转换为正确的数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果请求输出隐藏状态，则获取编码后的图像隐藏状态
            if output_hidden_states:
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 通过重复每个图像的隐藏状态来匹配提示数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对于未条件图像，获取对应的编码隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 同样地重复未条件图像的隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回条件和未条件的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要隐藏状态，直接获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 通过重复嵌入来匹配提示数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的零张量作为未条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回条件和未条件的图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 StableDiffusionPipeline 中复制的函数，用于准备图像嵌入
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用分类器自由引导，则初始化一个空列表，用于存储负面图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 检查输入适配器图像是否为列表，如果不是则转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像的长度是否与 IP 适配器数量匹配
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果不匹配，抛出错误并提供详细信息
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个单独的适配器图像和其对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断是否输出隐藏状态
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 对单个适配器图像进行编码，返回图像嵌入和负面图像嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用分类器自由引导，则将负面图像嵌入添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果已有输入适配器图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用分类器自由引导，将负面和正面图像嵌入分开
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 添加负面图像嵌入到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将正面图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储最终的 IP 适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历每个图像嵌入
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入重复 num_images_per_prompt 次
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用分类器自由引导，处理负面图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负面图像嵌入与正面图像嵌入拼接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到结果列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回所有的 IP 适配器图像嵌入
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的内容
    # 准备调度器步骤的额外参数，因为并非所有调度器的参数签名相同
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略此参数
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 并且应在 [0, 1] 范围内
    
        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数字典
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数字典
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数字典
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
    # 检查输入参数的有效性
    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的部分
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在张量的形状，包括批量大小和通道数
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器是否为列表，并且其长度与批量大小一致
            if isinstance(generator, list) and len(generator) != batch_size:
                # 如果不一致，抛出值错误
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果没有提供潜在张量，则随机生成一个
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了潜在张量，则将其移动到指定设备
                latents = latents.to(device)
    
            # 按照调度器所需的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在张量
            return latents
    
        # 定义获取时间标识的函数
        def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        # 创建一个包含原始大小、裁剪左上角坐标和目标大小的列表
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        # 计算通过添加时间嵌入维度得到的总嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取 UNet 中添加嵌入的预期输入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查预期的和实际的嵌入维度是否匹配
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将添加的时间 ID 转换为张量，指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 返回添加的时间 ID 张量
        return add_time_ids

    def upcast_vae(self):
        # 获取 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为浮点数类型
        self.vae.to(dtype=torch.float32)
        # 检查是否使用 Torch 2.0 或 XFormers 的注意力处理器
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # 如果使用 XFormers 或 Torch 2.0，注意力块不需要为浮点32，这样可以节省大量内存
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img 中复制的方法
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        查看 https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        参数:
            w (`torch.Tensor`):
                使用指定的引导比例生成嵌入向量，以丰富时间步嵌入。
            embedding_dim (`int`, *可选*, 默认值为 512):
                要生成的嵌入的维度。
            dtype (`torch.dtype`, *可选*, 默认值为 `torch.float32`):
                生成的嵌入的数据类型。

        返回:
            `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
        """
        # 确保输入张量 w 是一维的
        assert len(w.shape) == 1
        # 将 w 的值扩大 1000 倍
        w = w * 1000.0

        # 计算一半的嵌入维度
        half_dim = embedding_dim // 2
        # 计算嵌入的基础值
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 计算指数以生成嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 生成最终的嵌入，通过将 w 和基础值相乘
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 合并正弦和余弦嵌入
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度是奇数，进行零填充
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保生成的嵌入形状正确
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb

    @property
    # 定义方法以获取指导比例
    def guidance_scale(self):
        # 返回指导比例的值
        return self._guidance_scale

    # 将指导重缩放定义为属性
    @property
    def guidance_rescale(self):
        # 返回重缩放值
        return self._guidance_rescale

    # 将剪切跳过定义为属性
    @property
    def clip_skip(self):
        # 返回剪切跳过的值
        return self._clip_skip

    # 定义属性以判断是否执行无分类器自由引导
    @property
    def do_classifier_free_guidance(self):
        # 判断指导比例是否大于 1 且时间条件投影维度是否为 None
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 将交叉注意力关键字参数定义为属性
    @property
    def cross_attention_kwargs(self):
        # 返回交叉注意力关键字参数
        return self._cross_attention_kwargs

    # 将去噪结束定义为属性
    @property
    def denoising_end(self):
        # 返回去噪结束的值
        return self._denoising_end

    # 将时间步数定义为属性
    @property
    def num_timesteps(self):
        # 返回时间步数
        return self._num_timesteps

    # 将中断定义为属性
    @property
    def interrupt(self):
        # 返回中断状态
        return self._interrupt

    # 使用装饰器禁用梯度计算
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用方法以处理输入
    def __call__(
        # 输入提示，可以是字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 第二个提示，可选
        prompt_2: Optional[Union[str, List[str]]] = None,
        # 可选的高度
        height: Optional[int] = None,
        # 可选的宽度
        width: Optional[int] = None,
        # 推理步骤的数量，默认值为 50
        num_inference_steps: int = 50,
        # 时间步列表，可选
        timesteps: List[int] = None,
        # sigma 列表，可选
        sigmas: List[float] = None,
        # 可选的去噪结束值
        denoising_end: Optional[float] = None,
        # 指导比例，默认为 5.0
        guidance_scale: float = 5.0,
        # 可选的负提示，可以是字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 第二个负提示，可选
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # eta 值，默认为 0.0
        eta: float = 0.0,
        # 随机生成器，可选
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在表示，可选
        latents: Optional[torch.Tensor] = None,
        # 提示嵌入，可选
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示嵌入，可选
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 池化的提示嵌入，可选
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 负池化的提示嵌入，可选
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 输入适配器图像，可选
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 输入适配器图像嵌入，可选
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 返回字典的标志，默认为 True
        return_dict: bool = True,
        # 交叉注意力关键字参数，可选
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 原始重缩放值，默认为 0.0
        guidance_rescale: float = 0.0,
        # 可选的原始大小
        original_size: Optional[Tuple[int, int]] = None,
        # 左上角裁剪坐标，默认为 (0, 0)
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 可选的目标大小
        target_size: Optional[Tuple[int, int]] = None,
        # 可选的负原始大小
        negative_original_size: Optional[Tuple[int, int]] = None,
        # 负裁剪左上角坐标，默认为 (0, 0)
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 可选的负目标大小
        negative_target_size: Optional[Tuple[int, int]] = None,
        # 可选的剪切跳过值
        clip_skip: Optional[int] = None,
        # 可选的回调函数，在步骤结束时调用
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        # 默认输入张量名称列表
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 其他关键字参数
        **kwargs,
```