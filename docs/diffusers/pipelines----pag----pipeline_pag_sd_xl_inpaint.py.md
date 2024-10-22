# `.\diffusers\pipelines\pag\pipeline_pag_sd_xl_inpaint.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，软件按“原样”分发，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 有关许可证下的特定权限和限制，请参阅许可证。

import inspect  # 导入 inspect 模块以检查对象的活动
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示工具

import PIL.Image  # 导入 PIL 库中的 Image 模块，用于图像处理
import torch  # 导入 PyTorch 库
from transformers import (  # 从 transformers 库中导入必要的类和函数
    CLIPImageProcessor,  # 导入 CLIP 图像处理器
    CLIPTextModel,  # 导入 CLIP 文本模型
    CLIPTextModelWithProjection,  # 导入带投影的 CLIP 文本模型
    CLIPTokenizer,  # 导入 CLIP 分词器
    CLIPVisionModelWithProjection,  # 导入带投影的 CLIP 视觉模型
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入多管道回调和管道回调
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入管道图像输入和 VAE 图像处理器
from ...loaders import (  # 从 loaders 模块导入多个加载器混入类
    FromSingleFileMixin,  # 从单文件加载混入类
    IPAdapterMixin,  # IP 适配器混入类
    StableDiffusionXLLoraLoaderMixin,  # 稳定扩散 XL Lora 加载混入类
    TextualInversionLoaderMixin,  # 文本反转加载混入类
)
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入各种模型
from ...models.attention_processor import (  # 从注意力处理器模块导入注意力处理器类
    AttnProcessor2_0,  # 版本 2.0 的注意力处理器
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 缩放文本编码器的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (  # 导入多个工具函数
    USE_PEFT_BACKEND,  # 指示是否使用 PEFT 后端
    is_invisible_watermark_available,  # 检查是否可用隐形水印功能
    is_torch_xla_available,  # 检查是否可用 Torch XLA
    logging,  # 导入日志记录功能
    replace_example_docstring,  # 替换示例文档字符串的功能
    scale_lora_layers,  # 缩放 Lora 层的功能
    unscale_lora_layers,  # 取消缩放 Lora 层的功能
)
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混入
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput  # 导入稳定扩散 XL 管道输出
from .pag_utils import PAGMixin  # 导入 PAG 混入类


if is_invisible_watermark_available():  # 检查隐形水印功能是否可用
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker  # 导入稳定扩散 XL 水印类

if is_torch_xla_available():  # 检查 Torch XLA 是否可用
    import torch_xla.core.xla_model as xm  # 导入 Torch XLA 核心模块

    XLA_AVAILABLE = True  # 设置 XLA 可用标志为 True
else:
    XLA_AVAILABLE = False  # 设置 XLA 可用标志为 False


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，命名为模块名

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串的开始
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档字符串的结束
```  # 文档字符串的结束
```py  # 文档
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库，用于深度学习计算
        >>> from diffusers import AutoPipelineForInpainting  # 从 diffusers 库导入自动化图像修复管道
        >>> from diffusers.utils import load_image  # 从 diffusers 库导入图像加载工具

        >>> pipe = AutoPipelineForInpainting.from_pretrained(  # 从预训练模型创建图像修复管道
        ...     "stabilityai/stable-diffusion-xl-base-1.0",  # 指定使用的预训练模型路径
        ...     torch_dtype=torch.float16,  # 设置使用的张量数据类型为 float16 以节省内存
        ...     variant="fp16",  # 指定模型变体为 fp16
        ...     enable_pag=True,  # 启用分页功能以优化性能
        ... )
        >>> pipe.to("cuda")  # 将管道移动到 GPU 以加速计算

        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"  # 定义初始图像的 URL
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"  # 定义掩码图像的 URL

        >>> init_image = load_image(img_url).convert("RGB")  # 从 URL 加载初始图像并转换为 RGB 格式
        >>> mask_image = load_image(mask_url).convert("RGB")  # 从 URL 加载掩码图像并转换为 RGB 格式

        >>> prompt = "A majestic tiger sitting on a bench"  # 定义用于图像生成的提示文本
        >>> image = pipe(  # 调用图像修复管道进行图像生成
        ...     prompt=prompt,  # 提供提示文本
        ...     image=init_image,  # 输入初始图像
        ...     mask_image=mask_image,  # 输入掩码图像
        ...     num_inference_steps=50,  # 设置推理步骤数
        ...     strength=0.80,  # 设置修复强度
        ...     pag_scale=0.3,  # 设置分页缩放比例
        ... ).images[0]  # 获取生成的图像
        ```py 
# 注释部分是代码示例的文档字符串，用于说明函数的目的和用法
"""
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制而来
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    # 根据 guidance_rescale 对 noise_cfg 进行重新缩放，参考文献的第 3.4 节
    """
    # 计算 noise_pred_text 的标准差，沿指定维度保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算 noise_cfg 的标准差，沿指定维度保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 重新缩放来自指导的结果，以修正过度曝光
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 按照指导缩放因子与原始结果混合，以避免产生“普通”外观的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的噪声配置
    return noise_cfg


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制而来
def retrieve_latents(
    # 输入的编码器输出，类型为 torch.Tensor，可选的随机生成器，默认采样模式为 "sample"
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 检查 encoder_output 是否具有 latent_dist 属性，并且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 返回从 latent_dist 中采样的潜在变量
        return encoder_output.latent_dist.sample(generator)
    # 检查 encoder_output 是否具有 latent_dist 属性，并且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的众数
        return encoder_output.latent_dist.mode()
    # 检查 encoder_output 是否具有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 返回 encoder_output 中的潜在变量
        return encoder_output.latents
    # 如果没有找到所需的属性，抛出异常
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 复制而来
def retrieve_timesteps(
    # 调度器对象，推理步数，可选设备，时间步数，标准差列表，其他关键字参数
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    # 调用调度器的 set_timesteps 方法，并在调用后从调度器获取时间步数
    """
    # 处理自定义时间步数，将任何关键字参数传递给 scheduler.set_timesteps 方法
    # 定义函数的参数说明
    Args:
        scheduler (`SchedulerMixin`):
            # 要从中获取时间步的调度器
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            # 生成样本时使用的扩散步数。如果使用该参数，则 `timesteps` 必须为 `None`
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            # 指定时间步要移动到的设备。如果为 `None`，则时间步不会移动
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            # 自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递了 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            # 自定义 sigma 值，用于覆盖调度器的时间步间隔策略。如果传递了 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    # 函数返回值说明
    Returns:
        `Tuple[torch.Tensor, int]`: 
            # 返回一个元组，第一个元素是来自调度器的时间步调度，第二个元素是推理步数
            A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
    """
    # 检查是否同时传递了时间步和 sigma
    if timesteps is not None and sigmas is not None:
        # 如果同时传递，抛出错误
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了时间步
    if timesteps is not None:
        # 检查调度器是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步数
        num_inference_steps = len(timesteps)
    # 如果传递了 sigma
    elif sigmas is not None:
        # 检查调度器是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步数
        num_inference_steps = len(timesteps)
    # 如果没有传递时间步和 sigma
    else:
        # 设置调度器的时间步数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步数
    return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionXLPAGInpaintPipeline 的类，继承多个混合类
class StableDiffusionXLPAGInpaintPipeline(
    # 继承 DiffusionPipeline 类，提供基本的扩散管道功能
    DiffusionPipeline,
    # 继承 StableDiffusionMixin 类，提供稳定扩散的特性
    StableDiffusionMixin,
    # 继承 TextualInversionLoaderMixin 类，提供文本反演加载功能
    TextualInversionLoaderMixin,
    # 继承 StableDiffusionXLLoraLoaderMixin 类，提供 LoRA 加载功能
    StableDiffusionXLLoraLoaderMixin,
    # 继承 FromSingleFileMixin 类，提供从单个文件加载的功能
    FromSingleFileMixin,
    # 继承 IPAdapterMixin 类，提供 IP 适配器加载功能
    IPAdapterMixin,
    # 继承 PAGMixin 类，提供 PAG 特性
    PAGMixin,
):
    # 文档字符串，说明该类用于使用 Stable Diffusion XL 进行文本到图像生成
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    # 说明该模型继承自 DiffusionPipeline，并提到可以查阅超类文档了解通用方法
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    # 该管道还继承了一些加载方法的说明
    The pipeline also inherits the following loading methods:
        # 文本反演嵌入加载方法
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        # 从 .ckpt 文件加载的方法
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        # 加载 LoRA 权重的方法
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        # 保存 LoRA 权重的方法
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        # 加载 IP 适配器的方法
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters
    # 参数说明，提供每个参数的类型和功能
    Args:
        # 变分自编码器模型，用于对图像进行编码和解码
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        # 冻结的文本编码器，Stable Diffusion XL 使用 CLIP 的文本部分
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        # 第二个冻结的文本编码器，使用 CLIP 的文本和池部分
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        # CLIP 的分词器，用于将文本转换为模型可接受的格式
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # 第二个分词器，用于处理文本输入
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # 条件 U-Net 架构，用于去噪编码后的图像潜表示
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        # 调度器，用于与 U-Net 结合去噪图像潜表示
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        # 是否需要在推断期间传递美学评分的条件
        requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):
            Whether the `unet` requires a aesthetic_score condition to be passed during inference. Also see the config
            of `stabilityai/stable-diffusion-xl-refiner-1-0`.
        # 是否强制将负提示嵌入设置为 0
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        # 是否使用隐形水印库对输出图像进行水印处理
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"

    # 可选组件列表，包含所有可选参数的名称
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    # 定义一个包含回调张量输入的列表
    _callback_tensor_inputs = [
        # 潜在变量
        "latents",
        # 提示词嵌入
        "prompt_embeds",
        # 负面提示词嵌入
        "negative_prompt_embeds",
        # 附加文本嵌入
        "add_text_embeds",
        # 附加时间ID
        "add_time_ids",
        # 负面聚合提示词嵌入
        "negative_pooled_prompt_embeds",
        # 附加负面时间ID
        "add_neg_time_ids",
        # 掩码
        "mask",
        # 被掩码的图像潜在变量
        "masked_image_latents",
    ]

    # 初始化函数，接受多个模型及参数
    def __init__(
        # VAE模型
        vae: AutoencoderKL,
        # 文本编码器模型
        text_encoder: CLIPTextModel,
        # 第二个文本编码器模型
        text_encoder_2: CLIPTextModelWithProjection,
        # 词汇编码器
        tokenizer: CLIPTokenizer,
        # 第二个词汇编码器
        tokenizer_2: CLIPTokenizer,
        # UNet模型
        unet: UNet2DConditionModel,
        # 调度器
        scheduler: KarrasDiffusionSchedulers,
        # 可选的图像编码器
        image_encoder: CLIPVisionModelWithProjection = None,
        # 可选的特征提取器
        feature_extractor: CLIPImageProcessor = None,
        # 是否需要美学评分
        requires_aesthetics_score: bool = False,
        # 是否对空提示强制使用零
        force_zeros_for_empty_prompt: bool = True,
        # 可选的水印设置
        add_watermarker: Optional[bool] = None,
        # 应用层设置，默认是“mid”
        pag_applied_layers: Union[str, List[str]] = "mid",  # ["mid"], ["down.block_1", "up.block_0.attentions_0"]
    ):
        # 调用父类初始化方法
        super().__init__()

        # 注册多个模块
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            scheduler=scheduler,
        )
        # 注册配置项：强制零设置
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        # 注册配置项：美学评分需求
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
        # 计算VAE缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器实例
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 创建掩码处理器实例
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        # 如果水印未指定，则根据可用性设置
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        # 根据水印设置初始化水印对象
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        # 设置应用层
        self.set_pag_applied_layers(pag_applied_layers)

    # 从diffusers库中复制的编码图像的方法
    # 定义编码图像的函数，接受图像、设备、每个提示的图像数量和输出隐藏状态参数
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，将其转换为张量并提取像素值
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备，并转换为相应的数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态，进行隐藏状态的编码
            if output_hidden_states:
                # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 复制隐藏状态以匹配每个提示的图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 编码全零图像以获取无条件的隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 复制无条件隐藏状态以匹配每个提示的图像数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回图像和无条件的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 复制图像嵌入以匹配每个提示的图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入形状相同的全零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件嵌入
                return image_embeds, uncond_image_embeds
    
        # 从稳定扩散管道复制的函数，用于准备图像嵌入
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化存储图像嵌入的列表
        image_embeds = []
        # 如果启用分类器自由引导，则初始化负图像嵌入的列表
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 如果输入适配器图像不是列表，则转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像的长度是否与 IP 适配器的数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果不相同，抛出值错误
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历输入适配器图像和图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 检查当前图像投影层是否为图像投影的实例
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码当前图像，获取嵌入和负嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将当前图像嵌入添加到列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用分类器自由引导，则添加负图像嵌入
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 遍历输入适配器图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用分类器自由引导，则将嵌入拆分为负嵌入和图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化存储最终图像适配器嵌入的列表
        ip_adapter_image_embeds = []
        # 遍历图像嵌入和其索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 将当前图像嵌入复制指定次数
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用分类器自由引导，则处理负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入与图像嵌入连接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入转移到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回最终的图像适配器嵌入
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt 复制的代码
    # 定义一个编码提示的函数，接受多个参数以生成图像
        def encode_prompt(
            self,
            prompt: str,  # 主提示字符串，用于生成图像
            prompt_2: Optional[str] = None,  # 可选的第二个提示字符串
            device: Optional[torch.device] = None,  # 指定设备（如CPU或GPU）
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            do_classifier_free_guidance: bool = True,  # 是否使用无分类器引导
            negative_prompt: Optional[str] = None,  # 可选的负面提示字符串
            negative_prompt_2: Optional[str] = None,  # 可选的第二个负面提示字符串
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的池化提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面池化提示嵌入
            lora_scale: Optional[float] = None,  # 可选的LORA缩放因子
            clip_skip: Optional[int] = None,  # 可选的跳过剪辑的步数
        # 从diffusers库中复制的函数，用于准备额外的步骤参数
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外的参数，因为不同调度器的签名不同
            # eta（η）仅在DDIMScheduler中使用，其他调度器将忽略该值
            # eta对应于DDIM论文中的η，范围应在[0, 1]之间
    
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())  # 检查调度器是否接受eta参数
            extra_step_kwargs = {}  # 初始化额外步骤参数字典
            if accepts_eta:
                extra_step_kwargs["eta"] = eta  # 如果接受eta，则将其添加到字典
    
            # 检查调度器是否接受生成器
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())  # 检查生成器参数
            if accepts_generator:
                extra_step_kwargs["generator"] = generator  # 如果接受生成器，则将其添加到字典
            return extra_step_kwargs  # 返回准备好的额外步骤参数
    
        # 从diffusers库中复制的函数，用于检查输入参数的有效性
        def check_inputs(
            self,
            prompt,  # 主提示字符串
            prompt_2,  # 可选的第二个提示字符串
            image,  # 输入图像
            mask_image,  # 输入的遮罩图像
            height,  # 图像的高度
            width,  # 图像的宽度
            strength,  # 强度参数，用于控制图像生成
            callback_steps,  # 回调步骤数
            output_type,  # 输出类型（如图像或张量）
            negative_prompt=None,  # 可选的负面提示字符串
            negative_prompt_2=None,  # 可选的第二个负面提示字符串
            prompt_embeds=None,  # 可选的提示嵌入
            negative_prompt_embeds=None,  # 可选的负面提示嵌入
            ip_adapter_image=None,  # 可选的适配器图像
            ip_adapter_image_embeds=None,  # 可选的适配器图像嵌入
            callback_on_step_end_tensor_inputs=None,  # 在步骤结束时的回调张量输入
            padding_mask_crop=None,  # 可选的填充遮罩裁剪参数
        # 从diffusers库中复制的函数，用于准备潜在的变量
        def prepare_latents(
            self,
            batch_size,  # 批量大小
            num_channels_latents,  # 潜在通道数量
            height,  # 图像的高度
            width,  # 图像的宽度
            dtype,  # 数据类型（如float32）
            device,  # 指定设备（如CPU或GPU）
            generator,  # 生成器，用于随机数生成
            latents=None,  # 可选的潜在变量
            image=None,  # 可选的输入图像
            timestep=None,  # 可选的时间步长
            is_strength_max=True,  # 是否将强度设置为最大
            add_noise=True,  # 是否在潜在变量中添加噪声
            return_noise=False,  # 是否返回噪声
            return_image_latents=False,  # 是否返回图像潜在变量
    # 定义形状，包含批处理大小、通道数、调整后的高度和宽度
        ):
            shape = (
                batch_size,  # 批处理大小
                num_channels_latents,  # 潜在变量通道数
                int(height) // self.vae_scale_factor,  # 调整后的高度
                int(width) // self.vae_scale_factor,  # 调整后的宽度
            )
            # 检查生成器列表的长度是否与批处理大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(  # 抛出值错误，提示生成器长度与批处理大小不匹配
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 检查图像或时间步是否为空，且强度不为最大值
            if (image is None or timestep is None) and not is_strength_max:
                raise ValueError(  # 抛出值错误，提示初始化潜在变量时缺少必要参数
                    "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                    "However, either the image or the noise timestep has not been provided."
                )
    
            # 检查图像的通道数是否为4
            if image.shape[1] == 4:
                # 将图像转换为所需设备和数据类型
                image_latents = image.to(device=device, dtype=dtype)
                # 根据批处理大小重复图像潜在变量
                image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
            # 如果需要返回图像潜在变量或潜在变量为空且强度不为最大值
            elif return_image_latents or (latents is None and not is_strength_max):
                # 将图像转换为所需设备和数据类型
                image = image.to(device=device, dtype=dtype)
                # 使用 VAE 编码图像以获得图像潜在变量
                image_latents = self._encode_vae_image(image=image, generator=generator)
                # 根据批处理大小重复图像潜在变量
                image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
    
            # 如果潜在变量为空且需要添加噪声
            if latents is None and add_noise:
                # 创建随机噪声张量
                noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                # 如果强度为1，则初始化潜在变量为噪声，否则初始化为图像与噪声的组合
                latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
                # 如果强度为最大值，则将潜在变量乘以调度器的初始 sigma
                latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
            # 如果需要添加噪声但潜在变量不为空
            elif add_noise:
                # 将潜在变量转换为所需设备
                noise = latents.to(device)
                # 将潜在变量乘以调度器的初始 sigma
                latents = noise * self.scheduler.init_noise_sigma
            # 如果不添加噪声
            else:
                # 创建随机噪声张量
                noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                # 将图像潜在变量转换为所需设备
                latents = image_latents.to(device)
    
            # 将潜在变量打包为输出元组
            outputs = (latents,)
    
            # 如果需要返回噪声，将其添加到输出元组
            if return_noise:
                outputs += (noise,)
    
            # 如果需要返回图像潜在变量，将其添加到输出元组
            if return_image_latents:
                outputs += (image_latents,)
    
            # 返回最终的输出元组
            return outputs
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._encode_vae_image 复制的代码
    # 定义一个私有方法，用于编码变分自编码器的图像
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        # 获取输入图像的数据类型
        dtype = image.dtype
        # 如果配置强制上溯，则将图像转换为浮点类型，并将 VAE 转换为浮点32类型
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)
    
        # 如果生成器是列表，则对每个图像编码并获取潜在表示
        if isinstance(generator, list):
            image_latents = [
                # 调用 VAE 编码函数并获取每个图像的潜在表示
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            # 将所有潜在表示在第0维合并
            image_latents = torch.cat(image_latents, dim=0)
        else:
            # 对单个图像编码并获取潜在表示
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)
    
        # 如果配置强制上溯，则将 VAE 恢复为原数据类型
        if self.vae.config.force_upcast:
            self.vae.to(dtype)
    
        # 将潜在表示转换回原数据类型
        image_latents = image_latents.to(dtype)
        # 根据配置的缩放因子调整潜在表示
        image_latents = self.vae.config.scaling_factor * image_latents
    
        # 返回最终的潜在表示
        return image_latents
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline.prepare_mask_latents 复制的方法
        def prepare_mask_latents(
            # 定义输入参数，包括掩码、被遮蔽图像、批量大小、高度、宽度、数据类型、设备、生成器及分类器自由引导标志
            self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # 将掩码调整为与潜在空间形状相同，以便后续拼接
        # 在转换数据类型之前进行调整，避免在使用 CPU 卸载和半精度时出错
        mask = torch.nn.functional.interpolate(
            # 将掩码的大小调整为高度和宽度除以 VAE 缩放因子的结果
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        # 将掩码移动到指定设备并转换为指定数据类型
        mask = mask.to(device=device, dtype=dtype)

        # 为每个生成的提示复制掩码和被掩盖的图像潜在空间，使用适合 MPS 的方法
        if mask.shape[0] < batch_size:
            # 检查传入的掩码数量是否能整除所需的批处理大小
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    # 抛出错误，说明掩码数量与批处理大小不匹配
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            # 根据批处理大小复制掩码
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        # 根据是否进行无分类器自由引导拼接掩码
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        # 检查被掩盖的图像是否存在且有四个通道
        if masked_image is not None and masked_image.shape[1] == 4:
            # 将被掩盖的图像潜在空间设置为原始被掩盖图像
            masked_image_latents = masked_image
        else:
            # 如果条件不满足，则设置为 None
            masked_image_latents = None

        # 如果被掩盖的图像存在
        if masked_image is not None:
            # 如果被掩盖的图像潜在空间为 None
            if masked_image_latents is None:
                # 将被掩盖的图像移动到指定设备并转换为指定数据类型
                masked_image = masked_image.to(device=device, dtype=dtype)
                # 编码被掩盖的图像为 VAE 潜在空间
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

            # 检查被掩盖的图像潜在空间的数量是否小于批处理大小
            if masked_image_latents.shape[0] < batch_size:
                # 检查图像数量是否能整除批处理大小
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        # 抛出错误，说明图像数量与批处理大小不匹配
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                # 根据批处理大小复制被掩盖的图像潜在空间
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            # 根据是否进行无分类器自由引导拼接被掩盖的图像潜在空间
            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # 调整设备以防止在与潜在模型输入拼接时出现设备错误
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        # 返回处理后的掩码和被掩盖的图像潜在空间
        return mask, masked_image_latents

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps 复制的代码
    # 获取推理步骤的时间戳
        def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
            # 使用 init_timestep 获取原始时间戳
            if denoising_start is None:
                # 计算初始时间戳，取 num_inference_steps 与 strength 的乘积或 num_inference_steps 的最小值
                init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
                # 计算开始时间戳，确保不小于 0
                t_start = max(num_inference_steps - init_timestep, 0)
            else:
                # 如果 denoising_start 不为 None，开始时间戳为 0
                t_start = 0
    
            # 获取从 t_start 开始的时间戳序列
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
    
            # 如果直接请求一个时间戳开始，则强度与 denoising_start 相关
            if denoising_start is not None:
                # 计算离散时间戳截止值
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_start * self.scheduler.config.num_train_timesteps)
                    )
                )
    
                # 计算推理步骤数，依据时间戳小于截止值的总和
                num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
                if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                    # 如果调度器为二阶调度器，推理步骤数为偶数则加 1
                    num_inference_steps = num_inference_steps + 1
    
                # 因为 t_n+1 >= t_n，从末尾切片时间戳
                timesteps = timesteps[-num_inference_steps:]
                return timesteps, num_inference_steps
    
            # 返回剩余时间戳和计算后的推理步骤数
            return timesteps, num_inference_steps - t_start
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img 中复制
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
            # 创建包含原始尺寸、裁剪坐标和美学评分的列表
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            # 创建包含负原始尺寸、裁剪坐标和负美学评分的列表
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            # 创建包含原始尺寸、裁剪坐标和目标尺寸的列表
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            # 创建包含负原始尺寸、裁剪坐标和负目标尺寸的列表
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        # 计算传递的附加嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取预期的附加嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查预期维度大于传递维度且差值等于附加时间嵌入维度
        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出值错误，提示维度不匹配
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        # 检查预期维度小于传递维度且差值等于附加时间嵌入维度
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出值错误，提示维度不匹配
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        # 检查预期维度与传递维度不相等
        elif expected_add_embed_dim != passed_add_embed_dim:
            # 抛出值错误，提示模型配置不正确
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将添加的时间 ID 转换为张量
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 将添加的负时间 ID 转换为张量
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        # 返回添加的时间 ID 和添加的负时间 ID
        return add_time_ids, add_neg_time_ids

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae 中复制
    # 将变分自编码器（VAE）向上转换到指定数据类型
    def upcast_vae(self):
        # 获取当前 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为 float32 数据类型
        self.vae.to(dtype=torch.float32)
        # 判断是否使用了 torch 2.0 或 xformers 的注意力处理器
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # 如果使用了 xformers 或 torch 2.0，注意力块可以不使用 float32，从而节省内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积转换为原始数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将输入卷积转换为原始数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将中间块转换为原始数据类型
            self.vae.decoder.mid_block.to(dtype)
    
        # 从 LatentConsistencyModelPipeline 获取指导尺度嵌入
        def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
            """
            查看 https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
            参数:
                w (`torch.Tensor`):
                    生成指定指导尺度的嵌入向量，以丰富时间步嵌入。
                embedding_dim (`int`, *可选*, 默认值为 512):
                    生成嵌入的维度。
                dtype (`torch.dtype`, *可选*, 默认值为 `torch.float32`):
                    生成嵌入的数据类型。
    
            返回:
                `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
            """
            # 确保 w 的形状为一维
            assert len(w.shape) == 1
            # 将 w 乘以 1000
            w = w * 1000.0
    
            # 计算一半的维度
            half_dim = embedding_dim // 2
            # 计算嵌入的比例
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 生成指数衰减的嵌入
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 根据 w 和嵌入计算最终的嵌入
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 连接正弦和余弦值
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度为奇数，则进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保最终嵌入的形状符合预期
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回嵌入
            return emb
    
        # 返回当前指导尺度
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 返回当前指导重缩放
        @property
        def guidance_rescale(self):
            return self._guidance_rescale
    
        # 返回当前剪切跳过设置
        @property
        def clip_skip(self):
            return self._clip_skip
    
        # 判断是否进行无分类器指导，基于指导尺度和 UNet 配置
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 返回交叉注意力参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 返回去噪结束的设置
        @property
        def denoising_end(self):
            return self._denoising_end
    
        # 返回其他属性
    # 定义去噪过程的起始点的 getter 方法
        def denoising_start(self):
            # 返回去噪过程的起始点
            return self._denoising_start
    
    # 定义时间步数的属性
        @property
        def num_timesteps(self):
            # 返回时间步数
            return self._num_timesteps
    
    # 定义中断的属性
        @property
        def interrupt(self):
            # 返回中断状态
            return self._interrupt
    
    # 以无梯度的方式定义可调用的方法
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 输入提示字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 第二个提示字符串或字符串列表（可选）
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 输入图像（可选）
            image: PipelineImageInput = None,
            # 掩码图像（可选）
            mask_image: PipelineImageInput = None,
            # 掩码图像的潜在张量（可选）
            masked_image_latents: torch.Tensor = None,
            # 图像高度（可选）
            height: Optional[int] = None,
            # 图像宽度（可选）
            width: Optional[int] = None,
            # 填充掩码裁剪的大小（可选）
            padding_mask_crop: Optional[int] = None,
            # 去噪强度
            strength: float = 0.9999,
            # 推理步骤数量
            num_inference_steps: int = 50,
            # 时间步列表（可选）
            timesteps: List[int] = None,
            # sigma 值列表（可选）
            sigmas: List[float] = None,
            # 去噪起始点（可选）
            denoising_start: Optional[float] = None,
            # 去噪结束点（可选）
            denoising_end: Optional[float] = None,
            # 指导比例
            guidance_scale: float = 7.5,
            # 负提示字符串或字符串列表（可选）
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负提示字符串或字符串列表（可选）
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量（可选）
            num_images_per_prompt: Optional[int] = 1,
            # eta 值
            eta: float = 0.0,
            # 随机生成器（可选）
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在张量（可选）
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入（可选）
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入（可选）
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化的提示嵌入（可选）
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负池化的提示嵌入（可选）
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # IP 适配器图像（可选）
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # IP 适配器图像嵌入（可选）
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型（可选，默认是 "pil"）
            output_type: Optional[str] = "pil",
            # 是否返回字典形式的结果（默认是 True）
            return_dict: bool = True,
            # 交叉注意力的关键字参数（可选）
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 指导重新缩放
            guidance_rescale: float = 0.0,
            # 原始图像大小（可选）
            original_size: Tuple[int, int] = None,
            # 裁剪的左上角坐标（默认是 (0, 0)）
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标图像大小（可选）
            target_size: Tuple[int, int] = None,
            # 负原始图像大小（可选）
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负裁剪的左上角坐标（默认是 (0, 0)）
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负目标图像大小（可选）
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 美学评分
            aesthetic_score: float = 6.0,
            # 负美学评分
            negative_aesthetic_score: float = 2.5,
            # 跳过的剪辑数（可选）
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数（可选）
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 结束步骤时的张量输入名称列表（默认是 ["latents"]）
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # PAG 缩放因子
            pag_scale: float = 3.0,
            # PAG 自适应缩放因子
            pag_adaptive_scale: float = 0.0,
```