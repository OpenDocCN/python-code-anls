# `.\diffusers\pipelines\stable_diffusion_xl\pipeline_stable_diffusion_xl_inpaint.py`

```py
# 版权声明，指明该代码属于 HuggingFace 团队，受版权保护
# 本文件根据 Apache 许可证第 2.0 版授权，使用需遵循许可证条款
# 许可证的副本可以在以下网址获取
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，软件按 "原样" 方式分发，不提供任何形式的保证或条件
# 详见许可证中关于权限和限制的具体条款

# 导入 inspect 模块，用于获取活跃的对象的源代码和文档
import inspect
# 从 typing 模块导入多种类型提示工具
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 numpy 库，通常用于数值计算和数组操作
import numpy as np
# 导入 PIL.Image 库，用于处理图像
import PIL.Image
# 导入 torch 库，PyTorch 深度学习框架
import torch
# 从 transformers 库导入各种 CLIP 模型和处理器
from transformers import (
    CLIPImageProcessor,  # 图像处理器
    CLIPTextModel,  # 文本模型
    CLIPTextModelWithProjection,  # 带投影的文本模型
    CLIPTokenizer,  # 文本分词器
    CLIPVisionModelWithProjection,  # 带投影的视觉模型
)

# 导入回调函数和处理器类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 导入图像处理器类
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 导入加载器类，负责从文件加载模型
from ...loaders import (
    FromSingleFileMixin,  # 从单个文件加载混合
    IPAdapterMixin,  # IP 适配器混合
    StableDiffusionXLLoraLoaderMixin,  # 稳定扩散 XL Lora 加载器混合
    TextualInversionLoaderMixin,  # 文本反演加载器混合
)
# 导入模型类，用于图像生成
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
# 导入注意力处理器类
from ...models.attention_processor import (
    AttnProcessor2_0,  # 注意力处理器版本 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
# 导入调度器类，用于控制生成过程
from ...schedulers import KarrasDiffusionSchedulers
# 导入实用工具，提供额外功能
from ...utils import (
    USE_PEFT_BACKEND,  # 使用 PEFT 后端的标志
    deprecate,  # 标记已弃用的功能
    is_invisible_watermark_available,  # 检查是否支持隐形水印
    is_torch_xla_available,  # 检查是否可用 Torch XLA
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串
    scale_lora_layers,  # 调整 Lora 层的缩放
    unscale_lora_layers,  # 恢复 Lora 层的缩放
)
# 从 torch_utils 模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 导入扩散管道和稳定扩散混合的基类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 导入管道输出类
from .pipeline_output import StableDiffusionXLPipelineOutput

# 如果隐形水印可用，则导入水印处理器
if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker

# 如果 Torch XLA 可用，则导入相关核心功能
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 Torch XLA 核心模块

    XLA_AVAILABLE = True  # 设置标志表示 XLA 可用
else:
    XLA_AVAILABLE = False  # 设置标志表示 XLA 不可用

# 创建日志记录器，用于记录模块内的日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，用于说明示例代码或函数的作用
EXAMPLE_DOC_STRING = """
``` 
```py  
```  
```py  
```  
```py  
```  
```py  
```  
```py  
    # 示例代码使用了 Stable Diffusion XL 进行图像修复
        Examples:
            ```py
            # 导入 PyTorch 库
            >>> import torch
            # 从 diffusers 库导入 StableDiffusionXLInpaintPipeline 类
            >>> from diffusers import StableDiffusionXLInpaintPipeline
            # 从 diffusers.utils 导入 load_image 函数
            >>> from diffusers.utils import load_image
    
            # 创建 StableDiffusionXLInpaintPipeline 的实例，加载预训练模型
            >>> pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            ...     "stabilityai/stable-diffusion-xl-base-1.0",  # 模型的名称
            ...     torch_dtype=torch.float16,  # 使用半精度浮点数
            ...     variant="fp16",  # 指定模型变种为 fp16
            ...     use_safetensors=True,  # 启用安全张量
            ... )
            # 将管道转移到 CUDA 设备以加速计算
            >>> pipe.to("cuda")
    
            # 定义要修复的图像的 URL
            >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
            # 定义图像的掩码 URL
            >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    
            # 加载并转换初始图像为 RGB 格式
            >>> init_image = load_image(img_url).convert("RGB")
            # 加载并转换掩码图像为 RGB 格式
            >>> mask_image = load_image(mask_url).convert("RGB")
    
            # 定义提示文本，用于指导图像生成
            >>> prompt = "A majestic tiger sitting on a bench"
            # 使用管道生成图像，输入提示、初始图像、掩码图像，并设置推理步骤和强度
            >>> image = pipe(
            ...     prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80
            ... ).images[0]  # 获取生成的第一张图像
"""
# 多行字符串，可能用于文档说明或注释，内容未给出

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制的函数
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 对 `noise_cfg` 进行重新缩放。基于[Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)的研究结果。详见第 3.4 节
    """
    # 计算 noise_pred_text 沿所有维度（除第 0 维）标准差，保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算 noise_cfg 沿所有维度（除第 0 维）标准差，保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 重新缩放指导结果（修正过曝问题）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 根据指导缩放因子 mix 原始结果，避免生成“普通”的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的噪声配置
    return noise_cfg


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def mask_pil_to_torch(mask, height, width):
    # 预处理 mask
    if isinstance(mask, (PIL.Image.Image, np.ndarray)):
        # 如果 mask 是 PIL 图像或 NumPy 数组，将其放入列表中
        mask = [mask]

    if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
        # 如果 mask 是 PIL 图像列表，调整每个图像的尺寸
        mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
        # 将图像转换为灰度并堆叠成 NumPy 数组
        mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
        # 将数组的数据类型转换为 float32，并归一化到 [0, 1] 范围
        mask = mask.astype(np.float32) / 255.0
    elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
        # 如果 mask 是 NumPy 数组列表，堆叠数组
        mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

    # 将 NumPy 数组转换为 PyTorch 张量
    mask = torch.from_numpy(mask)
    # 返回转换后的张量
    return mask


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 复制的函数
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 如果 encoder_output 有 latent_dist 属性并且采样模式是“sample”
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 返回采样的潜在分布
        return encoder_output.latent_dist.sample(generator)
    # 如果 encoder_output 有 latent_dist 属性并且采样模式是“argmax”
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回潜在分布的众数
        return encoder_output.latent_dist.mode()
    # 如果 encoder_output 有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回潜在向量
        return encoder_output.latents
    else:
        # 如果都没有，抛出属性错误
        raise AttributeError("Could not access latents of provided encoder_output")


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 复制的函数
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器获取时间步。处理自定义时间步。任何关键字参数将传递给 `scheduler.set_timesteps`。
    # 参数说明部分
    Args:
        scheduler (`SchedulerMixin`):  # 调度器，用于获取时间步
            The scheduler to get timesteps from.  # 描述调度器的功能
        num_inference_steps (`int`):  # 生成样本时使用的扩散步骤数
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.  # 如果使用此参数，timesteps必须为None
        device (`str` or `torch.device`, *optional*):  # 指定要移动时间步的设备
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.  # 如果为None，则不移动时间步
        timesteps (`List[int]`, *optional*):  # 自定义时间步，用于覆盖调度器的时间步间距策略
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.  # 如果提供该参数，则num_inference_steps和sigmas必须为None
        sigmas (`List[float]`, *optional*):  # 自定义sigmas，用于覆盖调度器的时间步间距策略
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.  # 如果提供该参数，则num_inference_steps和timesteps必须为None

    Returns:  # 返回值说明部分
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.  # 返回一个元组，包含时间步调度和推断步骤数
    """
    # 检查是否同时提供了timesteps和sigmas
    if timesteps is not None and sigmas is not None:  # 如果两个参数都不为None
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")  # 抛出错误，告知只能选择一个参数
    if timesteps is not None:  # 如果timesteps不为None
        # 检查当前调度器是否支持timesteps
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())  # 检查set_timesteps方法的参数中是否包含timesteps
        if not accepts_timesteps:  # 如果不支持
            raise ValueError(  # 抛出错误，告知不支持自定义时间步
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义的时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)  # 调用调度器设置时间步
        timesteps = scheduler.timesteps  # 更新时间步为调度器中的值
        num_inference_steps = len(timesteps)  # 计算推断步骤数
    elif sigmas is not None:  # 如果sigmas不为None
        # 检查当前调度器是否支持sigmas
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())  # 检查set_timesteps方法的参数中是否包含sigmas
        if not accept_sigmas:  # 如果不支持
            raise ValueError(  # 抛出错误，告知不支持自定义sigmas
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义的sigmas
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)  # 调用调度器设置sigmas
        timesteps = scheduler.timesteps  # 更新时间步为调度器中的值
        num_inference_steps = len(timesteps)  # 计算推断步骤数
    else:  # 如果两个参数都为None
        # 设置默认的推断步骤
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)  # 调用调度器设置默认推断步骤
        timesteps = scheduler.timesteps  # 更新时间步为调度器中的值
    return timesteps, num_inference_steps  # 返回时间步和推断步骤数
# 定义一个名为 StableDiffusionXLInpaintPipeline 的类，继承多个基类
class StableDiffusionXLInpaintPipeline(
    # 继承自 DiffusionPipeline，提供扩散过程的基本功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin，提供与稳定扩散相关的功能
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin，允许加载文本反演嵌入
    TextualInversionLoaderMixin,
    # 继承自 StableDiffusionXLLoraLoaderMixin，允许加载和保存 LoRA 权重
    StableDiffusionXLLoraLoaderMixin,
    # 继承自 FromSingleFileMixin，允许从单个文件加载模型
    FromSingleFileMixin,
    # 继承自 IPAdapterMixin，允许加载 IP 适配器
    IPAdapterMixin,
):
    # 文档字符串，描述该类的用途和继承的功能
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters
    # 参数定义部分，说明各参数的用途
    Args:
        vae ([`AutoencoderKL`]):  # 变分自编码器模型，用于将图像编码和解码为潜在表示
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):  # 冻结的文本编码器，使用 CLIP 的文本部分
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):  # 第二个冻结文本编码器，使用 CLIP 的文本和池化部分
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):  # CLIPTokenizer 类的分词器
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):  # 第二个 CLIPTokenizer 类的分词器
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]):  # 条件 U-Net 架构，用于去噪编码后的图像潜在表示
            Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):  # 用于与 `unet` 结合使用的调度器
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):  # 是否需要美学评分条件
            Whether the `unet` requires a aesthetic_score condition to be passed during inference. Also see the config
            of `stabilityai/stable-diffusion-xl-refiner-1-0`.
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):  # 是否将负提示嵌入强制设为 0
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):  # 是否使用隐形水印库对输出图像进行水印处理
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """  # 文档字符串的结束

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"

    # 可选组件的列表，用于后续处理
    _optional_components = [
        "tokenizer",  # 分词器
        "tokenizer_2",  # 第二个分词器
        "text_encoder",  # 文本编码器
        "text_encoder_2",  # 第二个文本编码器
        "image_encoder",  # 图像编码器
        "feature_extractor",  # 特征提取器
    ]
    # 定义一个回调张量输入的列表，包含多个输入项的名称
    _callback_tensor_inputs = [
        "latents",  # 潜在表示
        "prompt_embeds",  # 提示嵌入
        "negative_prompt_embeds",  # 负提示嵌入
        "add_text_embeds",  # 附加文本嵌入
        "add_time_ids",  # 附加时间标识
        "negative_pooled_prompt_embeds",  # 负池化提示嵌入
        "add_neg_time_ids",  # 附加负时间标识
        "mask",  # 掩码
        "masked_image_latents",  # 被掩蔽的图像潜在表示
    ]

    # 初始化方法，设置各个组件和参数
    def __init__(
        # 定义自编码器
        vae: AutoencoderKL,
        # 定义文本编码器
        text_encoder: CLIPTextModel,
        # 定义第二个文本编码器
        text_encoder_2: CLIPTextModelWithProjection,
        # 定义分词器
        tokenizer: CLIPTokenizer,
        # 定义第二个分词器
        tokenizer_2: CLIPTokenizer,
        # 定义 UNet 模型
        unet: UNet2DConditionModel,
        # 定义调度器
        scheduler: KarrasDiffusionSchedulers,
        # 定义可选的图像编码器
        image_encoder: CLIPVisionModelWithProjection = None,
        # 定义可选的特征提取器
        feature_extractor: CLIPImageProcessor = None,
        # 定义是否需要美学评分的标志
        requires_aesthetics_score: bool = False,
        # 强制空提示为零的标志
        force_zeros_for_empty_prompt: bool = True,
        # 可选的水印添加标志
        add_watermarker: Optional[bool] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个模块到当前实例中
        self.register_modules(
            vae=vae,  # 注册自编码器
            text_encoder=text_encoder,  # 注册文本编码器
            text_encoder_2=text_encoder_2,  # 注册第二个文本编码器
            tokenizer=tokenizer,  # 注册分词器
            tokenizer_2=tokenizer_2,  # 注册第二个分词器
            unet=unet,  # 注册 UNet 模型
            image_encoder=image_encoder,  # 注册图像编码器
            feature_extractor=feature_extractor,  # 注册特征提取器
            scheduler=scheduler,  # 注册调度器
        )
        # 将强制空提示为零的标志注册到配置中
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        # 将是否需要美学评分的标志注册到配置中
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
        # 计算 VAE 缩放因子，基于块输出通道的数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化图像处理器，用于处理 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 初始化掩码处理器，进行特殊处理
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        # 确定是否添加水印，如果未指定则根据是否可用来设置
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        # 根据标志设置水印处理器
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()  # 创建水印处理器
        else:
            self.watermark = None  # 不创建水印处理器

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的编码图像的方法
    # 定义一个方法，用于编码图像
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入的图像是否为张量类型
            if not isinstance(image, torch.Tensor):
                # 如果不是，则使用特征提取器处理图像并返回张量
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定的设备上，并转换为正确的数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 对图像进行编码，获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态重复以匹配每个提示的图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对全零的图像进行编码，获取其隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将全零图像的隐藏状态重复以匹配每个提示的图像数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码的图像隐藏状态和全零图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要输出隐藏状态，则直接获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 将图像嵌入重复以匹配每个提示的图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入形状相同的全零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的代码
        def prepare_ip_adapter_image_embeds(
            # 定义方法参数，包括适配器图像、适配器图像嵌入、设备、每个提示的图像数量和是否执行分类器自由引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    # 开始处理图像嵌入的逻辑
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用分类器自由引导，初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果适配器图像嵌入为空
            if ip_adapter_image_embeds is None:
                # 检查输入图像是否为列表，如果不是则转换为列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
    
                # 确保适配器图像长度与 IP 适配器数量一致
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
    
                # 遍历每个适配器图像和对应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断输出隐藏状态是否需要
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码单个适配器图像，获取嵌入和负嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
    
                    # 将单个图像嵌入添加到列表
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用分类器自由引导，添加负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 如果已有适配器图像嵌入，遍历它们
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用分类器自由引导，分离负图像嵌入和正图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 添加正图像嵌入到列表
                    image_embeds.append(single_image_embeds)
    
            # 初始化最终的适配器图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历图像嵌入
            for i, single_image_embeds in enumerate(image_embeds):
                # 扩展每个图像嵌入至每个提示所需的数量
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用分类器自由引导，扩展负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    # 将负图像嵌入和正图像嵌入拼接在一起
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将处理后的图像嵌入添加到最终列表
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回适配器图像嵌入
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt 复制的代码
    # 定义一个编码提示的函数，接受多个参数来生成图像
    def encode_prompt(
        self,
        # 主提示字符串
        prompt: str,
        # 可选的第二个提示字符串
        prompt_2: Optional[str] = None,
        # 可选的设备参数，指定计算的设备
        device: Optional[torch.device] = None,
        # 每个提示生成的图像数量，默认为1
        num_images_per_prompt: int = 1,
        # 是否进行分类器自由引导，默认为True
        do_classifier_free_guidance: bool = True,
        # 可选的负面提示字符串
        negative_prompt: Optional[str] = None,
        # 可选的第二个负面提示字符串
        negative_prompt_2: Optional[str] = None,
        # 可选的提示嵌入，预先计算的提示向量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面提示嵌入，预先计算的负向量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的池化提示嵌入，预先计算的池化向量
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的池化负面提示嵌入，预先计算的负池化向量
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的Lora缩放因子
        lora_scale: Optional[float] = None,
        # 可选的跳过剪辑参数
        clip_skip: Optional[int] = None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的
    # 准备额外的步骤关键字参数，适用于调度器步骤，因为并非所有调度器都有相同的签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅用于 DDIMScheduler，对于其他调度器将被忽略
        # eta 在 DDIM 论文中的对应值应在 [0, 1] 范围内
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数的字典
        extra_step_kwargs = {}
        if accepts_eta:
            # 如果调度器接受eta参数，则将其添加到字典中
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受生成器
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果调度器接受生成器参数，则将其添加到字典中
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数
        return extra_step_kwargs

    # 定义检查输入的函数，用于验证各种输入参数的有效性
    def check_inputs(
        self,
        # 主提示字符串
        prompt,
        # 可选的第二个提示字符串
        prompt_2,
        # 输入图像
        image,
        # 输入掩码图像
        mask_image,
        # 图像高度
        height,
        # 图像宽度
        width,
        # 强度参数
        strength,
        # 回调步骤
        callback_steps,
        # 输出类型
        output_type,
        # 可选的负面提示字符串
        negative_prompt=None,
        # 可选的第二个负面提示字符串
        negative_prompt_2=None,
        # 可选的提示嵌入
        prompt_embeds=None,
        # 可选的负面提示嵌入
        negative_prompt_embeds=None,
        # 可选的适配器图像
        ip_adapter_image=None,
        # 可选的适配器图像嵌入
        ip_adapter_image_embeds=None,
        # 可选的在步骤结束时回调的张量输入
        callback_on_step_end_tensor_inputs=None,
        # 可选的填充掩码裁剪参数
        padding_mask_crop=None,
    # 定义准备潜在变量的函数，用于生成潜在表示
    def prepare_latents(
        # 批大小
        batch_size,
        # 潜在通道数
        num_channels_latents,
        # 图像高度
        height,
        # 图像宽度
        width,
        # 数据类型
        dtype,
        # 设备参数
        device,
        # 随机数生成器
        generator,
        # 可选的潜在变量
        latents=None,
        # 可选的输入图像
        image=None,
        # 时间步
        timestep=None,
        # 是否为最大强度，默认为True
        is_strength_max=True,
        # 是否添加噪声，默认为True
        add_noise=True,
        # 是否返回噪声，默认为False
        return_noise=False,
        # 是否返回图像潜在表示，默认为False
        return_image_latents=False,
    ):
        # 定义输出张量的形状，包含批量大小、通道数、以及缩放后的高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表，并且其长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出值错误，提示生成器长度与请求的批量大小不符
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 检查图像或时间步是否为 None，且强度不为最大值
        if (image is None or timestep is None) and not is_strength_max:
            # 抛出值错误，提示图像或噪声时间步未提供
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        # 检查图像的通道数是否为 4
        if image.shape[1] == 4:
            # 将图像转换为指定设备和数据类型的张量
            image_latents = image.to(device=device, dtype=dtype)
            # 重复图像张量以匹配批量大小
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        # 如果需要返回图像潜变量或潜变量为 None 且强度不为最大值
        elif return_image_latents or (latents is None and not is_strength_max):
            # 将图像转换为指定设备和数据类型的张量
            image = image.to(device=device, dtype=dtype)
            # 编码图像为 VAE 潜变量
            image_latents = self._encode_vae_image(image=image, generator=generator)
            # 重复图像潜变量以匹配批量大小
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        # 如果潜变量为 None 且需要添加噪声
        if latents is None and add_noise:
            # 生成随机噪声张量
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # 如果强度为 1，初始化潜变量为噪声，否则初始化为图像和噪声的组合
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # 如果强度为最大，则按调度器的初始化 sigma 缩放潜变量
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        # 如果需要添加噪声但潜变量不为 None
        elif add_noise:
            # 将潜变量转换为指定设备的张量
            noise = latents.to(device)
            # 按调度器的初始化 sigma 缩放潜变量
            latents = noise * self.scheduler.init_noise_sigma
        # 如果不需要添加噪声
        else:
            # 生成随机噪声张量
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # 将图像潜变量转换为指定设备的张量
            latents = image_latents.to(device)

        # 创建输出元组，初始包含潜变量
        outputs = (latents,)

        # 如果需要返回噪声，将其添加到输出元组中
        if return_noise:
            outputs += (noise,)

        # 如果需要返回图像潜变量，将其添加到输出元组中
        if return_image_latents:
            outputs += (image_latents,)

        # 返回最终的输出元组
        return outputs
    # 定义一个编码 VAE 图像的私有方法，接受图像和生成器作为参数
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        # 获取输入图像的数值类型
        dtype = image.dtype
        # 如果配置强制提升类型，则将图像转换为浮点型，并将 VAE 转为浮点32类型
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)
    
        # 检查生成器是否为列表
        if isinstance(generator, list):
            # 对于每个图像，编码并检索其潜变量
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            # 将所有潜变量在第0维度上连接成一个张量
            image_latents = torch.cat(image_latents, dim=0)
        else:
            # 如果生成器不是列表，直接编码图像并检索潜变量
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)
    
        # 如果配置强制提升类型，则将 VAE 恢复到原来的数值类型
        if self.vae.config.force_upcast:
            self.vae.to(dtype)
    
        # 将潜变量转换为原来的数据类型
        image_latents = image_latents.to(dtype)
        # 根据配置的缩放因子调整潜变量
        image_latents = self.vae.config.scaling_factor * image_latents
    
        # 返回处理后的潜变量
        return image_latents
    
    # 定义一个准备掩码潜变量的公共方法，接受多个参数
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # 将掩膜调整为与潜在变量形状一致，以便将掩膜与潜在变量拼接
        # 在转换数据类型之前执行此操作，以避免在使用 cpu_offload 和半精度时出现问题
        mask = torch.nn.functional.interpolate(
            # 调整掩膜大小，使其与潜在变量匹配，缩放因子由 self.vae_scale_factor 控制
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        # 将掩膜移动到指定设备并设置数据类型
        mask = mask.to(device=device, dtype=dtype)

        # 为每个提示生成重复掩膜和被掩膜图像潜在变量，使用适合 mps 的方法
        if mask.shape[0] < batch_size:
            # 检查掩膜数量是否能够整除批量大小
            if not batch_size % mask.shape[0] == 0:
                # 如果不匹配，抛出错误提示
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            # 通过重复掩膜以匹配批量大小
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        # 如果进行无分类器引导，则将掩膜复制两次；否则保持不变
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        # 检查被掩膜图像是否存在且其通道数为4
        if masked_image is not None and masked_image.shape[1] == 4:
            # 如果是，则直接将其赋值给 masked_image_latents
            masked_image_latents = masked_image
        else:
            # 否则初始化为 None
            masked_image_latents = None

        # 如果被掩膜图像存在
        if masked_image is not None:
            # 如果潜在变量为 None，则编码被掩膜图像
            if masked_image_latents is None:
                # 将被掩膜图像移动到指定设备并设置数据类型
                masked_image = masked_image.to(device=device, dtype=dtype)
                # 使用 VAE 编码器将被掩膜图像编码为潜在变量
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

            # 检查编码后的潜在变量数量是否能够整除批量大小
            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    # 如果不匹配，抛出错误提示
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                # 通过重复潜在变量以匹配批量大小
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            # 如果进行无分类器引导，则将潜在变量复制两次；否则保持不变
            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # 将潜在变量移动到指定设备并设置数据类型，以防在与潜在模型输入拼接时出现设备错误
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        # 返回处理后的掩膜和被掩膜图像的潜在变量
        return mask, masked_image_latents

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps 复制
    # 获取时间步长的函数，计算推理步骤、强度和设备相关的时间步长
        def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
            # 如果没有指定去噪开始时间，则计算初始时间步
            if denoising_start is None:
                # 根据强度和推理步骤计算初始时间步，确保不超过总推理步骤
                init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
                # 计算开始时间步，确保不小于0
                t_start = max(num_inference_steps - init_timestep, 0)
            else:
                # 如果指定了去噪开始时间，开始时间步为0
                t_start = 0
    
            # 根据开始时间步和调度器的顺序获取时间步长
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
    
            # 如果指定了去噪开始时间，则强度不再重要，直接使用去噪开始时间
            if denoising_start is not None:
                # 计算离散时间步截止点
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_start * self.scheduler.config.num_train_timesteps)
                    )
                )
    
                # 计算有效的推理步骤数量
                num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
                # 如果调度器是二阶调度器且推理步骤为偶数，则需要加1以确保正确性
                if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                    # 添加1以确保去噪过程在调度器的二阶导数步骤之后结束
                    num_inference_steps = num_inference_steps + 1
    
                # 从末尾切片获取推理步骤的时间步
                timesteps = timesteps[-num_inference_steps:]
                # 返回时间步和推理步骤数量
                return timesteps, num_inference_steps
    
            # 如果没有去噪开始时间，返回时间步和调整后的推理步骤数量
            return timesteps, num_inference_steps - t_start
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids 复制的函数
        def _get_add_time_ids(
            self,
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype,
            text_encoder_projection_dim=None,
    ):
        # 检查配置是否需要美学评分
        if self.config.requires_aesthetics_score:
            # 创建包含原始尺寸、裁剪坐标左上角和美学评分的列表
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            # 创建负样本的原始尺寸、裁剪坐标左上角和负美学评分的列表
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            # 创建包含原始尺寸、裁剪坐标左上角和目标尺寸的列表
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            # 创建负样本的原始尺寸、裁剪坐标左上角和负目标尺寸的列表
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        # 计算传入的附加嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取模型期望的附加嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查期望的附加嵌入维度是否大于实际传入的维度
        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出错误，提示需要启用美学评分
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        # 检查期望的附加嵌入维度是否小于实际传入的维度
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出错误，提示需要禁用美学评分
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        # 检查传入的附加嵌入维度是否与期望的不同
        elif expected_add_embed_dim != passed_add_embed_dim:
            # 抛出错误，提示模型配置不正确
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将 add_time_ids 转换为张量，并指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 将 add_neg_time_ids 转换为张量，并指定数据类型
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        # 返回添加的时间 ID 和负样本的时间 ID
        return add_time_ids, add_neg_time_ids

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae 复制的代码
    # 定义一个方法来将 VAE 模型的类型上升到指定的浮点数精度
        def upcast_vae(self):
            # 获取当前 VAE 的数据类型
            dtype = self.vae.dtype
            # 将 VAE 转换为浮点32类型
            self.vae.to(dtype=torch.float32)
            # 检查 VAE 解码器的中间块注意力处理器是否为指定的类型
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                ),
            )
            # 如果使用的是 xformers 或 torch_2_0，则注意力块不需要为浮点32，这可以节省大量内存
            if use_torch_2_0_or_xformers:
                # 将后量化卷积转换为当前数据类型
                self.vae.post_quant_conv.to(dtype)
                # 将解码器的输入卷积转换为当前数据类型
                self.vae.decoder.conv_in.to(dtype)
                # 将解码器的中间块转换为当前数据类型
                self.vae.decoder.mid_block.to(dtype)
    
        # 从 LatentConsistencyModelPipeline 中复制的方法，用于获取引导缩放嵌入
        def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
            """
            参考链接：模型的指导缩放嵌入生成方法。
    
            参数：
                w (`torch.Tensor`):
                    生成带有指定指导缩放的嵌入向量，以随后丰富时间步嵌入。
                embedding_dim (`int`, *可选*, 默认为 512):
                    要生成的嵌入维度。
                dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                    生成嵌入的数据类型。
    
            返回：
                `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
            """
            # 确保输入张量是一个一维向量
            assert len(w.shape) == 1
            # 将 w 的值乘以 1000.0 以增强缩放
            w = w * 1000.0
    
            # 计算嵌入的一半维度
            half_dim = embedding_dim // 2
            # 计算对数的常数值
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 生成指数衰减的嵌入
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 生成嵌入的最终形状，通过广播机制结合 w
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 将正弦和余弦嵌入合并到一起
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度为奇数，则进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保生成的嵌入形状符合预期
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回最终生成的嵌入
            return emb
    
        # 返回当前的指导缩放值
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 返回当前的指导重缩放值
        @property
        def guidance_rescale(self):
            return self._guidance_rescale
    
        # 返回当前的跳过剪辑值
        @property
        def clip_skip(self):
            return self._clip_skip
    
        # 定义不使用分类器自由指导的条件
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 返回交叉注意力的额外参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 返回去噪结束的值
        @property
        def denoising_end(self):
            return self._denoising_end
    
        # 这里省略了剩余的属性定义
    # 定义去噪的起始方法
        def denoising_start(self):
            # 返回去噪的起始值
            return self._denoising_start
    
        # 定义 num_timesteps 属性的获取方法
        @property
        def num_timesteps(self):
            # 返回时间步数
            return self._num_timesteps
    
        # 定义 interrupt 属性的获取方法
        @property
        def interrupt(self):
            # 返回中断状态
            return self._interrupt
    
        # 关闭梯度计算以提高性能，并替换示例文档字符串
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义可调用方法，接受多种输入参数
        def __call__(
            # 输入提示文本，支持字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 第二个提示文本，支持字符串或字符串列表
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 输入图像
            image: PipelineImageInput = None,
            # 输入掩码图像
            mask_image: PipelineImageInput = None,
            # 掩码图像的潜在张量
            masked_image_latents: torch.Tensor = None,
            # 图像高度
            height: Optional[int] = None,
            # 图像宽度
            width: Optional[int] = None,
            # 填充掩码裁剪参数
            padding_mask_crop: Optional[int] = None,
            # 强度参数
            strength: float = 0.9999,
            # 推理步骤数量
            num_inference_steps: int = 50,
            # 时间步列表
            timesteps: List[int] = None,
            # sigma值列表
            sigmas: List[float] = None,
            # 去噪开始值
            denoising_start: Optional[float] = None,
            # 去噪结束值
            denoising_end: Optional[float] = None,
            # 引导比例
            guidance_scale: float = 7.5,
            # 负提示文本，支持字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负提示文本，支持字符串或字符串列表
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: Optional[int] = 1,
            # eta值
            eta: float = 0.0,
            # 随机数生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在张量
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化后的提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负池化提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 图像适配器输入
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 图像适配器嵌入
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式
            return_dict: bool = True,
            # 交叉注意力的额外参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 引导重缩放参数
            guidance_rescale: float = 0.0,
            # 原始图像尺寸
            original_size: Tuple[int, int] = None,
            # 裁剪坐标，默认为(0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标图像尺寸
            target_size: Tuple[int, int] = None,
            # 负样本的原始尺寸
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负样本裁剪坐标，默认为(0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负样本目标尺寸
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 美学评分
            aesthetic_score: float = 6.0,
            # 负样本美学评分
            negative_aesthetic_score: float = 2.5,
            # 跳过的剪辑层
            clip_skip: Optional[int] = None,
            # 每步结束时的回调
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 回调时的张量输入列表
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 其他额外参数
            **kwargs,
```