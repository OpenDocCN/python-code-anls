# `.\diffusers\pipelines\stable_diffusion_xl\pipeline_stable_diffusion_xl_img2img.py`

```py
# 版权声明，表明版权属于 HuggingFace 团队，所有权利保留
# 
# 根据 Apache 2.0 许可证授权（"许可证"）； 
# 除非遵循许可证，否则不得使用此文件。
# 可通过以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定， 
# 否则根据许可证分发的软件是以“原样”基础分发，不附有任何类型的保证或条件，
# 无论是明示或暗示的。
# 请参见许可证以获取关于权限和限制的具体语言。

import inspect  # 导入 inspect 模块，用于检查对象
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示，定义多种数据类型

import PIL.Image  # 导入 PIL.Image 模块，用于图像处理
import torch  # 导入 PyTorch 库，用于深度学习操作
from transformers import (  # 从 transformers 库导入多个模型和处理器
    CLIPImageProcessor,  # CLIP 图像处理器
    CLIPTextModel,  # CLIP 文本模型
    CLIPTextModelWithProjection,  # 带投影的 CLIP 文本模型
    CLIPTokenizer,  # CLIP 标记器
    CLIPVisionModelWithProjection,  # 带投影的 CLIP 视觉模型
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入回调相关的类
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关的类
from ...loaders import (  # 导入加载器相关的混入类
    FromSingleFileMixin,  # 单文件加载混入
    IPAdapterMixin,  # IP 适配器混入
    StableDiffusionXLLoraLoaderMixin,  # StableDiffusion XL Lora 加载混入
    TextualInversionLoaderMixin,  # 文本反转加载混入
)
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入模型相关的类
from ...models.attention_processor import (  # 导入注意力处理器相关的类
    AttnProcessor2_0,  # 注意力处理器版本 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 规模的文本编码器的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (  # 导入工具类中的多个函数和常量
    USE_PEFT_BACKEND,  # 是否使用 PEFT 后端
    deprecate,  # 用于标记已弃用功能的装饰器
    is_invisible_watermark_available,  # 检查是否可用隐形水印
    is_torch_xla_available,  # 检查是否可用 Torch XLA
    logging,  # 导入日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的工具
    scale_lora_layers,  # 缩放 Lora 层的函数
    unscale_lora_layers,  # 反缩放 Lora 层的函数
)
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合类
from .pipeline_output import StableDiffusionXLPipelineOutput  # 导入稳定扩散 XL 管道输出类

# 检查隐形水印功能是否可用，如果可用，则导入水印处理器
if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker  # 导入稳定扩散 XL 水印处理器

# 检查 Torch XLA 是否可用，若可用，则导入其核心模型
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 模型核心

    XLA_AVAILABLE = True  # 设置标志，指示 XLA 可用
else:
    XLA_AVAILABLE = False  # 设置标志，指示 XLA 不可用

logger = logging.get_logger(__name__)  # 初始化日志记录器，使用当前模块名作为标识

EXAMPLE_DOC_STRING = """  # 示例文档字符串，展示如何使用管道
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableDiffusionXLImg2ImgPipeline  # 导入 StableDiffusion XL 图像到图像管道
        >>> from diffusers.utils import load_image  # 导入加载图像的工具

        >>> pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(  # 从预训练模型创建管道
        ...     "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16  # 指定模型和数据类型
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU
        >>> url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"  # 图像的 URL

        >>> init_image = load_image(url).convert("RGB")  # 加载并转换图像为 RGB 模式
        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义生成图像的提示
        >>> image = pipe(prompt, image=init_image).images[0]  # 使用管道生成图像
        ```py
"""

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制的内容
# 根据给定的噪声配置和文本预测噪声进行重新缩放
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 重新缩放 `noise_cfg`。基于论文 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的发现。参见第 3.4 节
    """
    # 计算噪声预测文本的标准差，保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差，保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据标准差调整噪声预测，修正过曝问题
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 将调整后的噪声与原始噪声按 `guidance_rescale` 因子混合，以避免图像看起来“普通”
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的噪声配置
    return noise_cfg


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 如果 encoder_output 有 latent_dist 属性并且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从潜在分布中采样并返回
        return encoder_output.latent_dist.sample(generator)
    # 如果 encoder_output 有 latent_dist 属性并且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回潜在分布的众数
        return encoder_output.latent_dist.mode()
    # 如果 encoder_output 有 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 直接返回潜在向量
        return encoder_output.latents
    # 如果都没有，抛出属性错误
    else:
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
    调用调度器的 `set_timesteps` 方法并在调用后从调度器获取时间步。处理自定义时间步。任何额外参数将传递给 `scheduler.set_timesteps`。

    参数：
        scheduler (`SchedulerMixin`):
            获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`，*可选*):
            将时间步移动到的设备。如果为 `None`，则时间步不移动。
        timesteps (`List[int]`，*可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传入 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`，*可选*):
            自定义 sigmas，用于覆盖调度器的时间步间隔策略。如果传入 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。
    # 返回值说明：返回一个元组，第一个元素是调度器的时间步调度，第二个元素是推理步骤的数量
    """
    # 检查是否同时传入了时间步和 sigma，抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果时间步不为空，进行相应处理
    if timesteps is not None:
        # 检查调度器是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误提示
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取当前调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果 sigma 不为空，进行相应处理
    elif sigmas is not None:
        # 检查调度器是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误提示
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取当前调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果都不为空，则设置默认的时间步
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取当前调度器的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤的数量
    return timesteps, num_inference_steps
# 定义一个类，名为 StableDiffusionXLImg2ImgPipeline，继承多个父类
class StableDiffusionXLImg2ImgPipeline(
    # 继承 DiffusionPipeline 类
    DiffusionPipeline,
    # 继承 StableDiffusionMixin 类
    StableDiffusionMixin,
    # 继承 TextualInversionLoaderMixin 类
    TextualInversionLoaderMixin,
    # 继承 FromSingleFileMixin 类
    FromSingleFileMixin,
    # 继承 StableDiffusionXLLoraLoaderMixin 类
    StableDiffusionXLLoraLoaderMixin,
    # 继承 IPAdapterMixin 类
    IPAdapterMixin,
):
    # 文档字符串，描述该管道的功能
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
    """
    # 文档字符串，描述函数的参数及其作用
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
                特别是 [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 变体。
            tokenizer (`CLIPTokenizer`):
                类 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 的分词器。
            tokenizer_2 (`CLIPTokenizer`):
                第二个类 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 的分词器。
            unet ([`UNet2DConditionModel`]): 条件 U-Net 架构，用于去噪编码后的图像潜在表示。
            scheduler ([`SchedulerMixin`]):
                用于与 `unet` 结合使用的调度器，以去噪编码后的图像潜在表示。可以是
                [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
            requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):
                `unet` 是否需要在推理过程中传递 `aesthetic_score` 条件。另见
                `stabilityai/stable-diffusion-xl-refiner-1-0` 的配置。
            force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
                是否强制将负提示嵌入始终设置为 0。另见
                `stabilityai/stable-diffusion-xl-base-1-0` 的配置。
            add_watermarker (`bool`, *optional*):
                是否使用 [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) 为输出图像加水印。如果未定义，
                如果安装了该包，则默认为 True，否则不使用水印。
        """
    
        # 定义模型的 CPU 卸载顺序
        model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
        # 定义可选组件列表
        _optional_components = [
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
            "feature_extractor",
        ]
    # 存储输入张量的回调名称列表
        _callback_tensor_inputs = [
            "latents",  # 潜在变量的名称
            "prompt_embeds",  # 正面提示的嵌入
            "negative_prompt_embeds",  # 负面提示的嵌入
            "add_text_embeds",  # 附加文本的嵌入
            "add_time_ids",  # 附加时间标识
            "negative_pooled_prompt_embeds",  # 负面池化提示的嵌入
            "add_neg_time_ids",  # 附加负时间标识
        ]
    
        # 初始化函数，设置模型及其参数
        def __init__(
            self,
            vae: AutoencoderKL,  # 变分自编码器模型
            text_encoder: CLIPTextModel,  # 文本编码器模型
            text_encoder_2: CLIPTextModelWithProjection,  # 第二文本编码器，带有投影
            tokenizer: CLIPTokenizer,  # 主要的分词器
            tokenizer_2: CLIPTokenizer,  # 第二分词器
            unet: UNet2DConditionModel,  # 条件 U-Net 模型
            scheduler: KarrasDiffusionSchedulers,  # 扩散调度器
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选的图像编码器，带有投影
            feature_extractor: CLIPImageProcessor = None,  # 可选的特征提取器
            requires_aesthetics_score: bool = False,  # 是否需要美学评分
            force_zeros_for_empty_prompt: bool = True,  # 是否对空提示强制设置为零
            add_watermarker: Optional[bool] = None,  # 可选的水印添加标志
        ):
            # 调用父类的初始化方法
            super().__init__()
    
            # 注册各个模块，便于后续调用
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
            # 将强制空提示为零的配置注册到系统中
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 将美学评分的配置注册到系统中
            self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器实例，使用缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
            # 根据条件决定是否添加水印
            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
    
            # 根据是否添加水印，初始化水印对象
            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()  # 创建水印对象
            else:
                self.watermark = None  # 不添加水印
    
        # 从稳定扩散 XL 管道复制的提示编码函数
        def encode_prompt(
            self,
            prompt: str,  # 正面提示字符串
            prompt_2: Optional[str] = None,  # 可选的第二个提示字符串
            device: Optional[torch.device] = None,  # 设备选择（CPU或GPU）
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            do_classifier_free_guidance: bool = True,  # 是否执行无分类器引导
            negative_prompt: Optional[str] = None,  # 可选的负面提示字符串
            negative_prompt_2: Optional[str] = None,  # 可选的第二个负面提示字符串
            prompt_embeds: Optional[torch.Tensor] = None,  # 预先计算的正面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 预先计算的负面提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 预先计算的池化提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 预先计算的负面池化提示嵌入
            lora_scale: Optional[float] = None,  # 可选的 LORA 缩放因子
            clip_skip: Optional[int] = None,  # 可选的跳过的 CLIP 层数
        # 从稳定扩散管道复制的额外步骤参数准备函数
    # 准备额外的参数以供调度器步骤使用，因为不同的调度器具有不同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # 检查调度器的步骤函数是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数的字典
        extra_step_kwargs = {}
        # 如果调度器接受 eta 参数，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的步骤函数是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator 参数，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
    # 检查输入参数的有效性和类型
    def check_inputs(
        self,
        prompt,  # 输入的提示文本
        prompt_2,  # 第二个输入的提示文本
        strength,  # 强度参数，控制效果的强弱
        num_inference_steps,  # 推理步骤的数量
        callback_steps,  # 回调步骤，控制回调的频率
        negative_prompt=None,  # 可选的负面提示文本
        negative_prompt_2=None,  # 第二个可选的负面提示文本
        prompt_embeds=None,  # 可选的提示嵌入表示
        negative_prompt_embeds=None,  # 可选的负面提示嵌入表示
        ip_adapter_image=None,  # 可选的适配器图像
        ip_adapter_image_embeds=None,  # 可选的适配器图像嵌入表示
        callback_on_step_end_tensor_inputs=None,  # 可选的回调输入张量
    # 获取时间步长，用于推理过程
        def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
            # 根据初始时间步计算原始时间步
            if denoising_start is None:
                # 计算初始时间步，取 num_inference_steps 与 strength 的乘积和 num_inference_steps 的较小值
                init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
                # 计算起始时间步，确保不小于零
                t_start = max(num_inference_steps - init_timestep, 0)
            else:
                # 如果给定 denoising_start，起始时间步设为零
                t_start = 0
    
            # 从调度器中获取时间步，基于起始时间步和调度器的顺序
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
    
            # 如果提供了 denoising_start，强度将不相关
            if denoising_start is not None:
                # 计算离散时间步截止点
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_start * self.scheduler.config.num_train_timesteps)
                    )
                )
    
                # 计算有效的推理时间步数
                num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
                # 检查是否为二阶调度器，并处理时间步数的奇偶性
                if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                    # 若时间步数为偶数，增加1以避免中断导致错误结果
                    num_inference_steps = num_inference_steps + 1
    
                # 从时间步的末尾切片，确保顺序正确
                timesteps = timesteps[-num_inference_steps:]
                return timesteps, num_inference_steps
    
            # 返回未经过 denoising_start 调整的时间步和时间步数
            return timesteps, num_inference_steps - t_start
    
        # 准备潜在变量的函数
        def prepare_latents(
            self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制
    # 定义一个方法，用于对输入图像进行编码
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，使用特征提取器将其转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定的设备，并转换为适当的数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 通过图像编码器编码图像，并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态沿第0维重复指定的次数
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建一个与图像形状相同的全零张量，并通过编码器获取其隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将无条件图像的隐藏状态沿第0维重复指定的次数
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回有条件和无条件的图像编码隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要输出隐藏状态，直接编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 将图像嵌入沿第0维重复指定的次数
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建一个与图像嵌入形状相同的全零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回有条件和无条件的图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的方法
        def prepare_ip_adapter_image_embeds(
            # 定义方法参数，包括输入适配器图像、适配器图像嵌入、设备、每个提示的图像数量和是否进行分类器自由引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    # 方法的开始，接收各种参数
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用无分类器自由引导，则初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果 IP 适配器图像嵌入为空
            if ip_adapter_image_embeds is None:
                # 检查 ip_adapter_image 是否为列表，如果不是则将其转为列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
    
                # 检查 ip_adapter_image 的长度是否与 IP 适配器数量一致
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
                
                # 遍历每个 IP 适配器图像和对应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断输出隐藏状态是否为真，若图像投影层不是 ImageProjection 类型
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 对单个图像进行编码，返回图像嵌入和负图像嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
    
                    # 将单个图像嵌入添加到图像嵌入列表
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用无分类器自由引导，则添加负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 如果 IP 适配器图像嵌入不为空，则遍历其内容
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用无分类器自由引导，则将单个图像嵌入拆分为负和正
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将图像嵌入添加到列表
                    image_embeds.append(single_image_embeds)
    
            # 初始化 IP 适配器图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历图像嵌入
            for i, single_image_embeds in enumerate(image_embeds):
                # 根据每个提示的图像数量重复图像嵌入
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用无分类器自由引导，则重复负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将处理后的图像嵌入添加到 IP 适配器图像嵌入列表
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回 IP 适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 定义获取附加时间 ID 的方法
        def _get_add_time_ids(
            self,
            # 原始图像的大小
            original_size,
            # 裁剪区域的左上角坐标
            crops_coords_top_left,
            # 目标图像大小
            target_size,
            # 审美分数
            aesthetic_score,
            # 负审美分数
            negative_aesthetic_score,
            # 负原始图像大小
            negative_original_size,
            # 负裁剪区域的左上角坐标
            negative_crops_coords_top_left,
            # 负目标图像大小
            negative_target_size,
            # 数据类型
            dtype,
            # 文本编码器投影维度（可选）
            text_encoder_projection_dim=None,
    # 结束当前方法定义
    ):
        # 检查配置是否需要美学评分
        if self.config.requires_aesthetics_score:
            # 组合原始尺寸、裁剪坐标和美学评分，转换为列表
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            # 组合负原始尺寸、负裁剪坐标和负美学评分，转换为列表
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            # 组合原始尺寸、裁剪坐标和目标尺寸，转换为列表
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            # 组合负原始尺寸、裁剪坐标和负目标尺寸，转换为列表
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        # 计算传递的附加嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取期望的附加嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查期望的附加嵌入维度是否大于传递的维度，并且两者差值符合配置要求
        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出值错误，提示维度不匹配，需启用美学评分
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        # 检查期望的附加嵌入维度是否小于传递的维度，并且两者差值符合配置要求
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 抛出值错误，提示维度不匹配，需禁用美学评分
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        # 检查期望的附加嵌入维度是否与传递的维度不相等
        elif expected_add_embed_dim != passed_add_embed_dim:
            # 抛出值错误，提示模型配置不正确
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将附加时间 ID 转换为张量，指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 将附加负时间 ID 转换为张量，指定数据类型
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        # 返回附加时间 ID 和附加负时间 ID
        return add_time_ids, add_neg_time_ids

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae 中复制
    # 定义一个用于上行转换 VAE 的方法
        def upcast_vae(self):
            # 获取 VAE 的数据类型
            dtype = self.vae.dtype
            # 将 VAE 转换为 float32 类型
            self.vae.to(dtype=torch.float32)
            # 判断是否使用 Torch 2.0 或 XFormers 的注意力处理器
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                ),
            )
            # 如果使用 xformers 或 torch_2_0，注意力块无需为 float32，可以节省大量内存
            if use_torch_2_0_or_xformers:
                # 将后量化卷积层转换为原始数据类型
                self.vae.post_quant_conv.to(dtype)
                # 将解码器的输入卷积层转换为原始数据类型
                self.vae.decoder.conv_in.to(dtype)
                # 将解码器中间块转换为原始数据类型
                self.vae.decoder.mid_block.to(dtype)
    
        # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img 中复制的方法
        def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
            """
            参见 https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
            参数:
                w (`torch.Tensor`):
                    用指定的引导比例生成嵌入向量，以随后丰富时间步嵌入。
                embedding_dim (`int`, *可选*, 默认为 512):
                    要生成的嵌入维度。
                dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                    生成嵌入的类型。
    
            返回:
                `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
            """
            # 确保输入张量 w 的维度为 1
            assert len(w.shape) == 1
            # 将 w 扩大 1000 倍
            w = w * 1000.0
    
            # 计算半维度
            half_dim = embedding_dim // 2
            # 计算嵌入的缩放因子
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 生成指数衰减的嵌入
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 计算最终嵌入矩阵
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 将正弦和余弦嵌入合并
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度为奇数，则进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保最终嵌入的形状正确
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回生成的嵌入
            return emb
    
        # 定义一个属性，返回当前的引导比例
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 定义一个属性，返回当前的引导重标定值
        @property
        def guidance_rescale(self):
            return self._guidance_rescale
    
        # 定义一个属性，返回当前的剪切跳过值
        @property
        def clip_skip(self):
            return self._clip_skip
    
        # 这里定义的 `guidance_scale` 类似于公式 (2) 中的引导权重 `w`
        # 来源于 Imagen 论文: https://arxiv.org/pdf/2205.11487.pdf。`guidance_scale = 1`
        # 表示不进行分类器自由引导。
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 定义一个属性，返回当前的交叉注意力关键字参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 定义一个属性，返回当前的去噪结束值
        @property
        def denoising_end(self):
            return self._denoising_end
    
        # 定义一个属性
    # 定义方法 denoising_start，返回去噪开始的值
    def denoising_start(self):
        return self._denoising_start

    # 定义属性 num_timesteps，返回时间步数的值
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 定义属性 interrupt，返回中断状态的值
    @property
    def interrupt(self):
        return self._interrupt

    # 装饰器，禁用梯度计算以减少内存使用
    @torch.no_grad()
    # 替换文档字符串为示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用方法 __call__，接收多个参数以执行具体操作
    def __call__(
        # 输入提示，可以是字符串或字符串列表
        self,
        prompt: Union[str, List[str]] = None,
        # 第二个提示，可以是字符串或字符串列表
        prompt_2: Optional[Union[str, List[str]]] = None,
        # 输入图像，可以是特定格式的图像输入
        image: PipelineImageInput = None,
        # 去噪强度，默认值为 0.3
        strength: float = 0.3,
        # 推理步骤数，默认值为 50
        num_inference_steps: int = 50,
        # 时间步列表，默认为 None
        timesteps: List[int] = None,
        # Sigma 值列表，默认为 None
        sigmas: List[float] = None,
        # 可选的去噪开始值
        denoising_start: Optional[float] = None,
        # 可选的去噪结束值
        denoising_end: Optional[float] = None,
        # 引导比例，默认值为 5.0
        guidance_scale: float = 5.0,
        # 可选的负提示，可以是字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 可选的第二个负提示
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # eta 值，默认值为 0.0
        eta: float = 0.0,
        # 随机数生成器，可以是单个或多个生成器
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在变量，可以是 PyTorch 张量
        latents: Optional[torch.Tensor] = None,
        # 提示的嵌入表示，可以是 PyTorch 张量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示的嵌入表示，可以是 PyTorch 张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 池化后的提示嵌入，可以是 PyTorch 张量
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 负池化提示嵌入，可以是 PyTorch 张量
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的图像输入适配器图像
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 可选的图像适配器嵌入列表
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 输出类型，默认值为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典，默认为 True
        return_dict: bool = True,
        # 可选的交叉注意力参数字典
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 引导重新缩放，默认值为 0.0
        guidance_rescale: float = 0.0,
        # 原始大小的元组，默认为 None
        original_size: Tuple[int, int] = None,
        # 左上角裁剪坐标，默认值为 (0, 0)
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 目标大小的元组，默认为 None
        target_size: Tuple[int, int] = None,
        # 可选的负原始大小元组
        negative_original_size: Optional[Tuple[int, int]] = None,
        # 负裁剪左上角坐标，默认值为 (0, 0)
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 可选的负目标大小元组
        negative_target_size: Optional[Tuple[int, int]] = None,
        # 美学分数，默认值为 6.0
        aesthetic_score: float = 6.0,
        # 负美学分数，默认值为 2.5
        negative_aesthetic_score: float = 2.5,
        # 可选的跳过裁剪的数量
        clip_skip: Optional[int] = None,
        # 可选的步骤结束时回调
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        # 默认的步骤结束时回调输入的张量名称
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 可接受的其他参数
        **kwargs,
```