# `.\diffusers\pipelines\animatediff\pipeline_animatediff_sdxl.py`

```py
# 版权声明，标明此文件由 HuggingFace 团队编写并保留所有权利
# 
# 根据 Apache 2.0 许可协议进行许可（“许可协议”）；
# 除非遵循该许可协议，否则不得使用此文件。
# 可在以下地址获取许可协议副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非根据适用法律或书面协议另有约定，
# 否则根据许可协议分发的软件按“原样”提供，
# 不附带任何形式的保证或条件，无论是明示还是暗示。
# 有关许可协议所涉及权限和限制的具体信息，
# 请参见许可协议。
# 
# 导入 inspect 模块以进行对象的检查和获取信息
import inspect
# 从 typing 模块导入类型注解相关的类
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 torch 库以支持深度学习操作
import torch
# 从 transformers 库导入多个 CLIP 相关模型和处理器
from transformers import (
    CLIPImageProcessor,  # 用于处理图像的 CLIP 处理器
    CLIPTextModel,  # 文本模型
    CLIPTextModelWithProjection,  # 带有投影功能的文本模型
    CLIPTokenizer,  # CLIP 的分词器
    CLIPVisionModelWithProjection,  # 带有投影功能的视觉模型
)

# 从本地模块导入图像处理相关的类
from ...image_processor import PipelineImageInput
# 从加载器模块导入多个混合类
from ...loaders import (
    FromSingleFileMixin,  # 从单个文件加载的混合类
    IPAdapterMixin,  # 适配器混合类
    StableDiffusionXLLoraLoaderMixin,  # 稳定扩散模型加载器混合类
    TextualInversionLoaderMixin,  # 文本反转加载器混合类
)
# 从模型模块导入多个深度学习模型
from ...models import AutoencoderKL, ImageProjection, MotionAdapter, UNet2DConditionModel, UNetMotionModel
# 从注意力处理器模块导入不同的注意力处理器类
from ...models.attention_processor import (
    AttnProcessor2_0,  # 注意力处理器版本 2.0
    FusedAttnProcessor2_0,  # 融合的注意力处理器版本 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
# 从 LoRA 模块导入调整文本编码器 LoRA 权重的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 从调度器模块导入多种调度器类
from ...schedulers import (
    DDIMScheduler,  # DDIM 调度器
    DPMSolverMultistepScheduler,  # DPM 多步调度器
    EulerAncestralDiscreteScheduler,  # 欧拉祖先离散调度器
    EulerDiscreteScheduler,  # 欧拉离散调度器
    LMSDiscreteScheduler,  # LMS 离散调度器
    PNDMScheduler,  # PNDM 调度器
)
# 从工具模块导入多个实用功能和常量
from ...utils import (
    USE_PEFT_BACKEND,  # 指示是否使用 PEFT 后端的标志
    logging,  # 导入日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的功能
    scale_lora_layers,  # 调整 LoRA 层的规模的功能
    unscale_lora_layers,  # 恢复 LoRA 层规模的功能
)
# 从 PyTorch 工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 导入视频处理器类
from ...video_processor import VideoProcessor
# 导入自由初始化相关的混合类
from ..free_init_utils import FreeInitMixin
# 导入扩散管道和稳定扩散相关的类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 导入管道输出模块中的 AnimateDiffPipelineOutput 类
from .pipeline_output import AnimateDiffPipelineOutput

# 获取当前模块的日志记录器实例
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，未给出具体内容
EXAMPLE_DOC_STRING = """
``` 
```py  # 文档字符串开始的标记
``` 
```py  # 文档字符串结束的标记
``` 
```py  # 另一个标记，表明示例文档字符串未结束
``` 
```py  # 文档字符串结束的标记
``` 
```py  # 文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标记
``` 
```py  # 示例文档字符串的结束标
    # 示例代码展示如何使用Diffusers库生成动画
        Examples:
            ```py
            # 导入所需的库和模块
            >>> import torch
            >>> from diffusers.models import MotionAdapter
            >>> from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
            >>> from diffusers.utils import export_to_gif
    
            # 从预训练模型加载运动适配器，使用半精度浮点数
            >>> adapter = MotionAdapter.from_pretrained(
            ...     "a-r-r-o-w/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16
            ... )
    
            # 定义基础模型ID
            >>> model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            # 从预训练模型加载调度器，配置相关参数
            >>> scheduler = DDIMScheduler.from_pretrained(
            ...     model_id,
            ...     subfolder="scheduler",
            ...     clip_sample=False,
            ...     timestep_spacing="linspace",
            ...     beta_schedule="linear",
            ...     steps_offset=1,
            ... )
            # 从预训练模型加载动画管道，并将其移至GPU
            >>> pipe = AnimateDiffSDXLPipeline.from_pretrained(
            ...     model_id,
            ...     motion_adapter=adapter,
            ...     scheduler=scheduler,
            ...     torch_dtype=torch.float16,
            ...     variant="fp16",
            ... ).to("cuda")
    
            # 启用内存节省功能
            >>> # enable memory savings
            >>> pipe.enable_vae_slicing()
            >>> pipe.enable_vae_tiling()
    
            # 调用管道生成动画，提供提示和参数
            >>> output = pipe(
            ...     prompt="a panda surfing in the ocean, realistic, high quality",
            ...     negative_prompt="low quality, worst quality",
            ...     num_inference_steps=20,
            ...     guidance_scale=8,
            ...     width=1024,
            ...     height=1024,
            ...     num_frames=16,
            ... )
    
            # 提取生成的第一帧
            >>> frames = output.frames[0]
            # 将帧导出为GIF动画文件
            >>> export_to_gif(frames, "animation.gif")
            ```py 
"""
# 该模块提供噪声配置的调整和时间步检索功能
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 调整 `noise_cfg`。基于论文 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 中的发现。见第 3.4 节
    """
    # 计算噪声预测文本的标准差，沿指定维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差，沿指定维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据标准差调整噪声预测结果，以修正过曝问题
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 将调整后的噪声与原始噪声混合，通过指导重缩因子避免“平淡”的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回调整后的噪声配置
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器检索时间步。处理自定义时间步。任何关键字参数将被传递给 `scheduler.set_timesteps`。

    Args:
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`，*可选*):
            要移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`，*可选*):
            用于覆盖调度器时间步间距策略的自定义时间步。如果传递了 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`，*可选*):
            用于覆盖调度器时间步间距策略的自定义 sigma。如果传递了 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    Returns:
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是来自调度器的时间步调度，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了自定义时间步和 sigma，若是则抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果提供了时间步长参数
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受时间步长参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受时间步长参数，抛出异常
        if not accepts_timesteps:
            raise ValueError(
                # 提示当前调度器类不支持自定义时间步长调度
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步长，传入设备和其他参数
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取设置后的时间步长
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果提供了 sigma 参数
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受 sigma 参数，抛出异常
        if not accept_sigmas:
            raise ValueError(
                # 提示当前调度器类不支持自定义 sigma 调度
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步长，传入 sigma、设备和其他参数
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取设置后的时间步长
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果既没有时间步长也没有 sigma 参数
    else:
        # 设置调度器的时间步长为推理步骤数量，传入设备和其他参数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 从调度器获取设置后的时间步长
        timesteps = scheduler.timesteps
    # 返回时间步长和推理步骤的数量
    return timesteps, num_inference_steps
# 定义一个名为 AnimateDiffSDXLPipeline 的类，继承多个基类
class AnimateDiffSDXLPipeline(
    # 继承 DiffusionPipeline 类，提供扩散管道功能
    DiffusionPipeline,
    # 继承 StableDiffusionMixin 类，提供稳定扩散相关功能
    StableDiffusionMixin,
    # 继承 FromSingleFileMixin 类，允许从单一文件加载
    FromSingleFileMixin,
    # 继承 StableDiffusionXLLoraLoaderMixin 类，允许加载 LoRA 权重
    StableDiffusionXLLoraLoaderMixin,
    # 继承 TextualInversionLoaderMixin 类，允许加载文本反转嵌入
    TextualInversionLoaderMixin,
    # 继承 IPAdapterMixin 类，提供 IP 适配器加载功能
    IPAdapterMixin,
    # 继承 FreeInitMixin 类，提供自由初始化相关功能
    FreeInitMixin,
):
    # 文档字符串，描述该类的用途和继承的功能
    r"""
    Pipeline for text-to-video generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters
    # 参数说明文档字符串
    Args:
        # 变分自编码器模型，用于将图像编码为潜在表示并解码回图像
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        # 冻结的文本编码器，Stable Diffusion XL 使用 CLIP 的文本部分
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        # 第二个冻结文本编码器，Stable Diffusion XL 使用 CLIP 的文本和池部分
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        # CLIPTokenizer 类的分词器
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # 第二个 CLIPTokenizer 类的分词器
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # 条件 U-Net 架构，用于去噪编码后的图像潜在表示
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the encoded image latents.
        # 调度器，与 U-Net 结合使用以去噪编码的图像潜在表示
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        # 是否强制将负提示嵌入设置为 0 的布尔值，可选，默认为 "True"
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
    """

    # 定义模型在 CPU 上卸载的顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    # 可选组件的列表
    _optional_components = [
        # 分词器
        "tokenizer",
        # 第二个分词器
        "tokenizer_2",
        # 文本编码器
        "text_encoder",
        # 第二个文本编码器
        "text_encoder_2",
        # 图像编码器
        "image_encoder",
        # 特征提取器
        "feature_extractor",
    ]
    # 回调张量输入的列表
    _callback_tensor_inputs = [
        # 潜在表示
        "latents",
        # 提示嵌入
        "prompt_embeds",
        # 负提示嵌入
        "negative_prompt_embeds",
        # 附加文本嵌入
        "add_text_embeds",
        # 附加时间 ID
        "add_time_ids",
        # 负池化提示嵌入
        "negative_pooled_prompt_embeds",
        # 负附加时间 ID
        "negative_add_time_ids",
    ]
    # 初始化类的构造函数
        def __init__(
            self,
            vae: AutoencoderKL,  # 定义变分自编码器
            text_encoder: CLIPTextModel,  # 定义第一个文本编码器
            text_encoder_2: CLIPTextModelWithProjection,  # 定义第二个文本编码器，带投影
            tokenizer: CLIPTokenizer,  # 定义第一个分词器
            tokenizer_2: CLIPTokenizer,  # 定义第二个分词器
            unet: Union[UNet2DConditionModel, UNetMotionModel],  # 定义 UNet 模型，可以是条件模型或运动模型
            motion_adapter: MotionAdapter,  # 定义运动适配器
            scheduler: Union[  # 定义调度器，支持多种类型
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选的图像编码器，带投影
            feature_extractor: CLIPImageProcessor = None,  # 可选的特征提取器
            force_zeros_for_empty_prompt: bool = True,  # 空提示时是否强制使用零
        ):
            # 调用父类构造函数
            super().__init__()
    
            # 如果 UNet 是二维条件模型，则将其转换为运动模型
            if isinstance(unet, UNet2DConditionModel):
                unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
    
            # 注册各个模块，以便在模型中使用
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                motion_adapter=motion_adapter,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
            # 将空提示强制使用零的设置注册到配置中
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 初始化视频处理器
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor)
    
            # 设置默认采样大小
            self.default_sample_size = self.unet.config.sample_size
    
        # 从 StableDiffusionXLPipeline 复制的 encode_prompt 方法，修改参数为 num_videos_per_prompt
        def encode_prompt(
            self,
            prompt: str,  # 输入的提示文本
            prompt_2: Optional[str] = None,  # 可选的第二个提示文本
            device: Optional[torch.device] = None,  # 可选的设备设置
            num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量
            do_classifier_free_guidance: bool = True,  # 是否使用无分类器引导
            negative_prompt: Optional[str] = None,  # 可选的负面提示文本
            negative_prompt_2: Optional[str] = None,  # 可选的第二个负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的池化提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面池化嵌入
            lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
            clip_skip: Optional[int] = None,  # 可选的跳过剪辑
        # 从 StableDiffusionPipeline 复制的 encode_image 方法
    # 定义一个方法用于编码图像
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入的图像是否为张量，如果不是，则使用特征提取器将其转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并转换为相应的数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果要求输出隐藏状态，则进行相应的处理
            if output_hidden_states:
                # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态根据每个提示的图像数量进行重复
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对全零张量进行编码以获取无条件隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将无条件隐藏状态根据每个提示的图像数量进行重复
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码后的图像隐藏状态和无条件隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要输出隐藏状态，直接获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 将图像嵌入根据每个提示的图像数量进行重复
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入形状相同的全零张量作为无条件嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的方法
        def prepare_ip_adapter_image_embeds(
            # 定义参数，包括适配器图像、图像嵌入、设备、每个提示的图像数量和分类器自由引导的标志
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    # 定义一个函数，用于处理图像嵌入
        ):
            # 初始化一个空列表用于存储图像嵌入
            image_embeds = []
            # 如果启用了分类器自由引导，初始化一个空列表用于存储负图像嵌入
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果输入适配器图像嵌入为空
            if ip_adapter_image_embeds is None:
                # 如果输入适配器图像不是列表，将其转换为列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
    
                # 检查输入适配器图像的数量是否与 IP 适配器数量一致
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        # 抛出值错误，说明输入适配器图像的数量与 IP 适配器的数量不匹配
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
    
                # 遍历输入适配器图像和图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断当前图像投影层是否是 ImageProjection 的实例
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码当前单张适配器图像，获取图像嵌入和负图像嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
    
                    # 将单张图像嵌入添加到图像嵌入列表
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用了分类器自由引导，将负图像嵌入添加到负图像嵌入列表
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 遍历已提供的适配器图像嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用了分类器自由引导，分离出负图像嵌入和图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        # 将负图像嵌入添加到负图像嵌入列表
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将图像嵌入添加到图像嵌入列表
                    image_embeds.append(single_image_embeds)
    
            # 初始化一个空列表用于存储适配器图像嵌入
            ip_adapter_image_embeds = []
            # 遍历图像嵌入及其索引
            for i, single_image_embeds in enumerate(image_embeds):
                # 根据每个提示的图像数量扩展单张图像嵌入
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用了分类器自由引导，扩展负图像嵌入并与图像嵌入连接
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将处理后的图像嵌入添加到适配器图像嵌入列表
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回最终的适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.text_to_video_synthesis/pipeline_text_to_video_synth.TextToVideoSDPipeline.decode_latents 复制的方法
        def decode_latents(self, latents):
            # 根据 VAE 的缩放因子调整潜在向量
            latents = 1 / self.vae.config.scaling_factor * latents
    
            # 获取潜在向量的形状信息
            batch_size, channels, num_frames, height, width = latents.shape
            # 调整潜在向量的维度以适配解码过程
            latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
    
            # 解码潜在向量以生成图像
            image = self.vae.decode(latents).sample
            # 将解码后的图像调整为视频格式
            video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
            # 始终将视频转换为 float32 格式，以保证兼容性并减少开销
            video = video.float()
            # 返回解码后的视频数据
            return video
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 拷贝而来
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为并不是所有的调度器都有相同的参数签名
        # eta（η）仅在 DDIMScheduler 中使用，其他调度器会忽略该参数。
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 之间

        # 检查调度器的 step 方法的参数中是否接受 eta
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个空字典，用于存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法的参数中是否接受 generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数字典
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    # 从 diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents 拷贝而来
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        # 计算生成的潜在张量的形状
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 如果生成器是列表且长度与批量大小不匹配，则引发值错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜在张量，则根据指定形状生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在张量，则将其移动到指定设备
            latents = latents.to(device)

        # 根据调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回缩放后的潜在张量
        return latents

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        # 创建一个包含原始大小、裁剪坐标和目标大小的列表
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        # 计算通过时间嵌入维度和文本编码器维度得出的传递嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取 UNet 模型中线性层的输入特征维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 如果预期的和传递的嵌入维度不一致，则引发错误
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将时间 ID 转换为张量，指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 返回处理后的时间 ID 张量
        return add_time_ids

    def upcast_vae(self):
        # 获取 VAE 的数据类型
        dtype = self.vae.dtype
        # 将 VAE 转换为 float32 类型
        self.vae.to(dtype=torch.float32)
        # 检查 VAE 解码器中是否使用了指定的处理器
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # 如果使用了 xformers 或 torch_2_0，则将相关层转换为相应数据类型以节省内存
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img 导入的函数
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        参见 GitHub 链接，获取引导尺度嵌入向量

        参数:
            w (`torch.Tensor`):
                用于生成嵌入向量的引导尺度。
            embedding_dim (`int`, *可选*, 默认值为 512):
                要生成的嵌入维度。
            dtype (`torch.dtype`, *可选*, 默认值为 `torch.float32`):
                生成嵌入的数值类型。

        返回:
            `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
        """
        # 确保输入张量是 1D 的
        assert len(w.shape) == 1
        # 将输入 w 放大 1000 倍
        w = w * 1000.0

        # 计算嵌入的一半维度
        half_dim = embedding_dim // 2
        # 计算每个嵌入位置的缩放因子
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成指数衰减的嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将输入 w 转换为指定 dtype，并与嵌入相乘
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 将正弦和余弦嵌入组合在一起
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度是奇数，进行零填充
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保最终嵌入的形状是正确的
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb

    @property
    # 定义获取指导比例的函数
        def guidance_scale(self):
            # 返回私有变量 _guidance_scale 的值
            return self._guidance_scale
    
        # 属性装饰器，定义获取指导重标定的属性
        @property
        def guidance_rescale(self):
            # 返回私有变量 _guidance_rescale 的值
            return self._guidance_rescale
    
        # 属性装饰器，定义获取剪辑跳过的属性
        @property
        def clip_skip(self):
            # 返回私有变量 _clip_skip 的值
            return self._clip_skip
    
        # 定义无分类器自由指导的属性，参照 Imagen 论文中的指导比例
        @property
        def do_classifier_free_guidance(self):
            # 判断指导比例是否大于 1 且时间条件投影维度是否为 None
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 属性装饰器，定义获取交叉注意力参数的属性
        @property
        def cross_attention_kwargs(self):
            # 返回私有变量 _cross_attention_kwargs 的值
            return self._cross_attention_kwargs
    
        # 属性装饰器，定义获取去噪结束的属性
        @property
        def denoising_end(self):
            # 返回私有变量 _denoising_end 的值
            return self._denoising_end
    
        # 属性装饰器，定义获取时间步数的属性
        @property
        def num_timesteps(self):
            # 返回私有变量 _num_timesteps 的值
            return self._num_timesteps
    
        # 属性装饰器，定义获取中断状态的属性
        @property
        def interrupt(self):
            # 返回私有变量 _interrupt 的值
            return self._interrupt
    
        # 禁用梯度计算的装饰器，减少内存使用
        @torch.no_grad()
        # 用于替换示例文档字符串的装饰器
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义可调用的方法，支持多种参数
        def __call__(
            # 提示文本，支持字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 第二个提示文本，支持字符串或字符串列表
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 视频帧数量，默认为 16
            num_frames: int = 16,
            # 图像高度，默认为 None
            height: Optional[int] = None,
            # 图像宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤数量，默认为 50
            num_inference_steps: int = 50,
            # 自定义时间步列表，默认为 None
            timesteps: List[int] = None,
            # 噪声参数列表，默认为 None
            sigmas: List[float] = None,
            # 去噪结束时间，默认为 None
            denoising_end: Optional[float] = None,
            # 指导比例，默认为 5.0
            guidance_scale: float = 5.0,
            # 负面提示文本，支持字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负面提示文本，支持字符串或字符串列表
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成视频的数量，默认为 1
            num_videos_per_prompt: Optional[int] = 1,
            # 噪声的 eta 值，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，支持单个或多个生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在空间张量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化的提示嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面池化的提示嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像嵌入，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 交叉注意力参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 指导重标定，默认为 0.0
            guidance_rescale: float = 0.0,
            # 原始图像尺寸，默认为 None
            original_size: Optional[Tuple[int, int]] = None,
            # 图像裁剪左上角坐标，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标图像尺寸，默认为 None
            target_size: Optional[Tuple[int, int]] = None,
            # 负面原始图像尺寸，默认为 None
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负面裁剪左上角坐标，默认为 (0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负面目标图像尺寸，默认为 None
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 剪辑跳过参数，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 结束时输入的张量名称列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```