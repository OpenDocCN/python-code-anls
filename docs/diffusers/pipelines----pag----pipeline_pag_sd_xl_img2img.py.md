# `.\diffusers\pipelines\pag\pipeline_pag_sd_xl_img2img.py`

```py
# 版权声明，标识文件归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获得许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是在“按现状”基础上分发的，不提供任何明示或暗示的担保或条件。
# 请参见许可证以了解管理权限和
# 限制的具体条款。

# 导入 inspect 模块以检查对象的类型
import inspect
# 从 typing 模块导入所需的类型注解
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PIL.Image 用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 transformers 库中导入必要的 CLIP 模型和处理器
from transformers import (
    CLIPImageProcessor,  # 图像处理器
    CLIPTextModel,  # 文本模型
    CLIPTextModelWithProjection,  # 带投影的文本模型
    CLIPTokenizer,  # 分词器
    CLIPVisionModelWithProjection,  # 带投影的视觉模型
)

# 从相对路径导入各类回调和图像处理器
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import (
    FromSingleFileMixin,  # 单文件加载混合器
    IPAdapterMixin,  # IP 适配器混合器
    StableDiffusionXLLoraLoaderMixin,  # 稳定扩散 XL Lora 加载器混合器
    TextualInversionLoaderMixin,  # 文本反转加载器混合器
)
# 从模型模块导入必要的模型
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from ...models.attention_processor import (
    AttnProcessor2_0,  # 注意力处理器 2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 调整 Lora 的文本编码器比例
from ...schedulers import KarrasDiffusionSchedulers  # Karras 扩散调度器
from ...utils import (
    USE_PEFT_BACKEND,  # 是否使用 PEFT 后端
    is_invisible_watermark_available,  # 检查不可见水印是否可用
    is_torch_xla_available,  # 检查 Torch XLA 是否可用
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的工具
    scale_lora_layers,  # 缩放 Lora 层
    unscale_lora_layers,  # 反缩放 Lora 层
)
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合器
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput  # 导入稳定扩散 XL 管道输出
from .pag_utils import PAGMixin  # 导入 PAG 混合器

# 检查不可见水印是否可用，如果可用则导入水印模块
if is_invisible_watermark_available():
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

# 检查 Torch XLA 是否可用，如果可用则导入 XLA 模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 核心模型

    XLA_AVAILABLE = True  # 设置 XLA 可用标志
else:
    XLA_AVAILABLE = False  # 设置 XLA 不可用标志

# 使用日志记录模块获取当前模块的记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义示例文档字符串
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AutoPipelineForImage2Image
        >>> from diffusers.utils import load_image

        >>> pipe = AutoPipelineForImage2Image.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-refiner-1.0",
        ...     torch_dtype=torch.float16,
        ...     enable_pag=True,
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU
        >>> url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

        >>> init_image = load_image(url).convert("RGB")  # 加载并转换初始图像为 RGB
        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义提示语
        >>> image = pipe(prompt, image=init_image, pag_scale=0.3).images[0]  # 生成图像
        ```py
"""
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 对 `noise_cfg` 进行重新缩放。基于 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的发现。参见第 3.4 节
    """
    # 计算噪声预测文本的标准差，维度保持不变
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差，维度保持不变
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据标准差缩放噪声预测，修复过度曝光问题
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 根据指导的缩放因子，将缩放后的噪声与原始噪声混合，避免生成“平淡”的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的噪声配置
    return noise_cfg


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 如果 encoder_output 包含 latent_dist 且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 latent_dist 中采样
        return encoder_output.latent_dist.sample(generator)
    # 如果 encoder_output 包含 latent_dist 且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 latent_dist 的众数
        return encoder_output.latent_dist.mode()
    # 如果 encoder_output 包含 latents
    elif hasattr(encoder_output, "latents"):
        # 返回 encoder_output 的 latents
        return encoder_output.latents
    # 如果都不满足，抛出属性错误
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 复制
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器中检索时间步。处理自定义时间步。任何关键字参数将传递给 `scheduler.set_timesteps`。
    # 函数参数说明
    Args:
        scheduler (`SchedulerMixin`):
            # 用于获取时间步的调度器
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            # 生成样本时使用的扩散步数。如果使用此参数，则 `timesteps` 必须为 `None`
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            # 用于移动时间步的设备。如果为 `None`，则时间步不会被移动
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            # 自定义时间步，覆盖调度器的时间步间隔策略。如果传入 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            # 自定义 sigma，覆盖调度器的时间步间隔策略。如果传入 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    # 返回值说明
    Returns:
        `Tuple[torch.Tensor, int]`: 
            # 返回一个元组，第一个元素是来自调度器的时间步调度，第二个元素是推理步数
            A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
    """
    # 检查是否同时传入了 `timesteps` 和 `sigmas`
    if timesteps is not None and sigmas is not None:
        # 如果同时传入，抛出错误提示
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    
    # 检查是否传入了 `timesteps`
    if timesteps is not None:
        # 检查当前调度器类是否支持自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            # 如果不支持，抛出错误提示
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器中获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步数
        num_inference_steps = len(timesteps)
    
    # 检查是否传入了 `sigmas`
    elif sigmas is not None:
        # 检查当前调度器类是否支持自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            # 如果不支持，抛出错误提示
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器中获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步数
        num_inference_steps = len(timesteps)
    
    # 如果既没有传入 `timesteps` 也没有 `sigmas`
    else:
        # 根据推理步数设置时间步
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 从调度器中获取设置后的时间步
        timesteps = scheduler.timesteps
    
    # 返回时间步和推理步数
    return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionXLPAGImg2ImgPipeline 的类，继承多个混入类
class StableDiffusionXLPAGImg2ImgPipeline(
    # 继承自 DiffusionPipeline 类，提供基本的扩散管道功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin 类，提供与 Stable Diffusion 相关的功能
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin 类，提供文本反演加载功能
    TextualInversionLoaderMixin,
    # 继承自 FromSingleFileMixin 类，提供从单个文件加载的功能
    FromSingleFileMixin,
    # 继承自 StableDiffusionXLLoraLoaderMixin 类，提供加载 LoRA 权重的功能
    StableDiffusionXLLoraLoaderMixin,
    # 继承自 IPAdapterMixin 类，提供 IP 适配器加载功能
    IPAdapterMixin,
    # 继承自 PAGMixin 类，提供 PAG 相关的功能
    PAGMixin,
):
    # 文档字符串，描述该管道的用途和功能
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
    # 文档字符串，描述各参数的功能和用途
        Args:
            vae ([`AutoencoderKL`]):
                变分自编码器（VAE）模型，用于对图像进行编码和解码，生成潜在表示。
            text_encoder ([`CLIPTextModel`]):
                冻结的文本编码器。Stable Diffusion XL 使用 CLIP 的文本部分，
                特别是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
            text_encoder_2 ([` CLIPTextModelWithProjection`]):
                第二个冻结的文本编码器。Stable Diffusion XL 使用 CLIP 的文本和池部分，
                特别是 [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 变体。
            tokenizer (`CLIPTokenizer`):
                `CLIPTokenizer` 类的分词器。
            tokenizer_2 (`CLIPTokenizer`):
                第二个 `CLIPTokenizer` 类的分词器。
            unet ([`UNet2DConditionModel`]): 条件 U-Net 架构，用于去噪编码后的图像潜在表示。
            scheduler ([`SchedulerMixin`]):
                与 `unet` 一起使用的调度器，用于去噪编码后的图像潜在表示。可以是
                [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
            requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):
                `unet` 是否需要在推理过程中传递 `aesthetic_score` 条件。另见
                `stabilityai/stable-diffusion-xl-refiner-1-0` 的配置。
            force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
                是否强制将负提示嵌入始终设置为 0。另见
                `stabilityai/stable-diffusion-xl-base-1-0` 的配置。
            add_watermarker (`bool`, *optional*):
                是否使用 [invisible_watermark 库](https://github.com/ShieldMnt/invisible-watermark/) 对输出图像进行水印。如果未定义，默认将设置为 True（如果安装了该包），否则不使用水印。
        """
    
        # 定义模型在 CPU 上的卸载顺序
        model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
        # 定义可选组件列表
        _optional_components = [
            "tokenizer",  # 第一个分词器
            "tokenizer_2",  # 第二个分词器
            "text_encoder",  # 第一个文本编码器
            "text_encoder_2",  # 第二个文本编码器
            "image_encoder",  # 图像编码器
            "feature_extractor",  # 特征提取器
        ]
    # 定义回调张量输入的名称列表
    _callback_tensor_inputs = [
        "latents",  # 潜在表示
        "prompt_embeds",  # 提示嵌入
        "negative_prompt_embeds",  # 负向提示嵌入
        "add_text_embeds",  # 添加的文本嵌入
        "add_time_ids",  # 添加的时间标识
        "negative_pooled_prompt_embeds",  # 负向池化提示嵌入
        "add_neg_time_ids",  # 添加的负向时间标识
    ]

    # 初始化方法，定义模型的各个组件
    def __init__(
        self,
        vae: AutoencoderKL,  # 变分自编码器
        text_encoder: CLIPTextModel,  # 文本编码器
        text_encoder_2: CLIPTextModelWithProjection,  # 第二个文本编码器，带投影
        tokenizer: CLIPTokenizer,  # 文本分词器
        tokenizer_2: CLIPTokenizer,  # 第二个文本分词器
        unet: UNet2DConditionModel,  # UNet2D条件模型
        scheduler: KarrasDiffusionSchedulers,  # Karras扩散调度器
        image_encoder: CLIPVisionModelWithProjection = None,  # 图像编码器，带投影，默认为None
        feature_extractor: CLIPImageProcessor = None,  # 特征提取器，默认为None
        requires_aesthetics_score: bool = False,  # 是否需要美学评分
        force_zeros_for_empty_prompt: bool = True,  # 是否强制将空提示的值设为零
        add_watermarker: Optional[bool] = None,  # 添加水印的选项
        pag_applied_layers: Union[str, List[str]] = "mid",  # 应用层的选项，默认为"mid"
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模型的各个模块
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
        # 将配置信息注册到类中
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
        # 计算变分自编码器的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器实例
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # 如果没有指定水印选项，检查水印是否可用
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        # 根据水印选项创建水印对象
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()  # 创建水印实例
        else:
            self.watermark = None  # 不使用水印

        # 设置应用的层
        self.set_pag_applied_layers(pag_applied_layers)

    # 从StableDiffusionXLPipeline复制的编码提示方法
    def encode_prompt(
        self,
        prompt: str,  # 主要提示字符串
        prompt_2: Optional[str] = None,  # 第二个提示字符串，默认为None
        device: Optional[torch.device] = None,  # 指定的设备，默认为None
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
        do_classifier_free_guidance: bool = True,  # 是否使用无分类器自由引导
        negative_prompt: Optional[str] = None,  # 负向提示字符串，默认为None
        negative_prompt_2: Optional[str] = None,  # 第二个负向提示字符串，默认为None
        prompt_embeds: Optional[torch.Tensor] = None,  # 提示嵌入的张量，默认为None
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负向提示嵌入的张量，默认为None
        pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 池化提示嵌入的张量，默认为None
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 负向池化提示嵌入的张量，默认为None
        lora_scale: Optional[float] = None,  # LoRA缩放因子，默认为None
        clip_skip: Optional[int] = None,  # 跳过的CLIP层数，默认为None
    # 从StableDiffusionPipeline复制的准备额外步骤关键字的方法
    # 为调度器步骤准备额外的关键字参数，不同调度器的签名可能不同
    def prepare_extra_step_kwargs(self, generator, eta):
        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化一个空字典用于存储额外的步骤关键字参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器步骤是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数字典
        return extra_step_kwargs
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.check_inputs 复制而来
        def check_inputs(
            self,
            prompt,
            prompt_2,
            strength,
            num_inference_steps,
            callback_steps,
            # 可选的负向提示参数
            negative_prompt=None,
            negative_prompt_2=None,
            # 提示嵌入参数
            prompt_embeds=None,
            negative_prompt_embeds=None,
            # 输入适配器图像及其嵌入参数
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            # 处理步骤结束时的回调输入
            callback_on_step_end_tensor_inputs=None,
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps 复制而来
    # 获取时间步，考虑推理步骤、强度、设备及去噪起始时间
        def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
            # 根据去噪起始时间确定初始时间步
            if denoising_start is None:
                # 计算初始时间步，确保不超过推理步骤总数
                init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
                # 计算开始时间步，确保不小于零
                t_start = max(num_inference_steps - init_timestep, 0)
            else:
                # 如果有去噪起始时间，开始时间步设为零
                t_start = 0
    
            # 从调度器获取时间步，从计算的起始时间步开始
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
    
            # 如果请求特定时间步，强度不再相关，由去噪起始时间决定
            if denoising_start is not None:
                # 计算离散时间步截止点
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_start * self.scheduler.config.num_train_timesteps)
                    )
                )
    
                # 计算在截止点之前的推理步骤数量
                num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
                if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                    # 如果调度器为二阶调度器，调整推理步骤数量以避免错误
                    num_inference_steps = num_inference_steps + 1
    
                # 从时间步数组末尾切片，确保符合推理步骤数量
                timesteps = timesteps[-num_inference_steps:]
                return timesteps, num_inference_steps
    
            # 返回时间步和推理步骤数，考虑开始时间步
            return timesteps, num_inference_steps - t_start
    
        # 从图像生成管道准备潜变量的函数，参数包括图像、时间步、批量大小等
        def prepare_latents(
            self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
        # 从图像编码管道复制的函数
    # 编码图像并返回图像嵌入或隐藏状态
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，使用特征提取器处理图像
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像转移到指定设备并转换为目标数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态，进行编码并处理隐藏状态
            if output_hidden_states:
                # 获取图像的隐藏状态，并选取倒数第二层的输出
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 重复隐藏状态以适应每个提示的图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对于无条件图像，使用零张量进行编码并获取隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 重复无条件隐藏状态以适应每个提示的图像数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回图像和无条件图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 重复图像嵌入以适应每个提示的图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 StableDiffusionPipeline 复制的函数，用于准备 IP 适配器的图像嵌入
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    # 定义处理图像嵌入的代码块
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用无分类器引导，则初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果输入适配器图像嵌入为 None
            if ip_adapter_image_embeds is None:
                # 检查输入适配器图像是否为列表
                if not isinstance(ip_adapter_image, list):
                    # 如果不是，转换为单元素列表
                    ip_adapter_image = [ip_adapter_image]
    
                # 确保输入适配器图像的数量与 IP 适配器数量相同
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    # 如果不匹配，抛出值错误
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
    
                # 遍历每个适配器图像及其对应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断输出隐藏状态是否为布尔值
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码单个图像，获取图像嵌入和负图像嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
    
                    # 将图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用无分类器引导，添加负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 遍历已存在的图像嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用无分类器引导，则分离负图像嵌入和图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds)
    
            # 初始化输入适配器图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历图像嵌入及其索引
            for i, single_image_embeds in enumerate(image_embeds):
                # 将单个图像嵌入重复指定次数
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用无分类器引导，重复负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    # 将负图像嵌入与图像嵌入连接
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入移动到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将处理后的图像嵌入添加到列表中
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回输入适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline 复制的函数
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
            # 生成包含原始大小、裁剪坐标和美学评分的列表
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            # 生成包含负样本的原始大小、裁剪坐标和负美学评分的列表
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            # 生成包含原始大小、裁剪坐标和目标大小的列表
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            # 生成包含负样本的原始大小、裁剪坐标和负目标大小的列表
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        # 计算通过时间嵌入的维度与文本编码器投影维度的乘积
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取模型预期的时间嵌入维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查预期的嵌入维度是否大于实际传入的维度，并确保差值等于配置中的时间嵌入维度
        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 如果条件不满足，抛出错误并提示用户检查配置
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        # 检查预期的嵌入维度是否小于实际传入的维度，并确保差值等于配置中的时间嵌入维度
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            # 如果条件不满足，抛出错误并提示用户检查配置
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        # 检查预期的嵌入维度是否与实际传入的维度不相等
        elif expected_add_embed_dim != passed_add_embed_dim:
            # 如果条件不满足，抛出错误并提示用户检查配置
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将时间 ID 列表转换为张量，并指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 将负时间 ID 列表转换为张量，并指定数据类型
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        # 返回生成的时间 ID 和负时间 ID 张量
        return add_time_ids, add_neg_time_ids

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae 复制
    # 将 VAE 模型的参数类型转换为指定的数据类型
    def upcast_vae(self):
        # 获取当前 VAE 模型的参数数据类型
        dtype = self.vae.dtype
        # 将 VAE 模型的参数转换为 float32 类型
        self.vae.to(dtype=torch.float32)
        # 判断是否使用了 Torch 2.0 或 xformers 的注意力处理器
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # 如果使用 xformers 或 Torch 2.0，则注意力块不需要使用 float32
        # 这可以节省大量内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积层的参数转换为原始数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将输入卷积层的参数转换为原始数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将中间块的参数转换为原始数据类型
            self.vae.decoder.mid_block.to(dtype)

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img 中复制的方法
    def get_guidance_scale_embedding(
        # 输入参数：指导权重的张量
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        参考链接：https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        参数：
            w (`torch.Tensor`):
                生成具有指定指导比例的嵌入向量，以随后丰富时间步长嵌入。
            embedding_dim (`int`, *可选*, 默认为 512):
                要生成的嵌入的维度。
            dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                生成的嵌入的数据类型。

        返回：
            `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
        """
        # 确保输入张量 w 只有一个维度
        assert len(w.shape) == 1
        # 将 w 的值放大 1000 倍
        w = w * 1000.0

        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算用于生成嵌入的基础值
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成用于时间步长嵌入的指数值
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 计算最终的嵌入向量
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 生成 sin 和 cos 的组合嵌入
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度为奇数，则在最后填充一个零
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保生成的嵌入形状与预期一致
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入向量
        return emb

    # 属性：获取指导比例
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 属性：获取指导重标定
    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # 属性：获取剪切跳过参数
    @property
    def clip_skip(self):
        return self._clip_skip

    # 这里的 `guidance_scale` 类似于方程 (2) 中的指导权重 `w`
    # 来自于 Imagen 论文：https://arxiv.org/pdf/2205.11487.pdf ，`guidance_scale = 1`
    # 表示不进行无分类器引导。
    @property
    def do_classifier_free_guidance(self):
        # 判断是否需要进行无分类器引导
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 属性：获取交叉注意力的关键字参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 属性：获取去噪结束的参数
    @property
    def denoising_end(self):
        return self._denoising_end

    # 属性定义尚未完成
    # 定义一个方法，返回当前去噪开始的阈值
    def denoising_start(self):
        return self._denoising_start

    # 定义属性，返回时间步数
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 定义属性，返回中断状态
    @property
    def interrupt(self):
        return self._interrupt

    # 使用装饰器禁用梯度计算以节省内存
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用方法，处理各种输入参数
    def __call__(
        # 提示文本，可以是字符串或字符串列表
        self,
        prompt: Union[str, List[str]] = None,
        # 第二个提示文本，可以是字符串或字符串列表
        prompt_2: Optional[Union[str, List[str]]] = None,
        # 输入图像
        image: PipelineImageInput = None,
        # 去噪强度，默认值为0.3
        strength: float = 0.3,
        # 推理步骤数，默认值为50
        num_inference_steps: int = 50,
        # 时间步列表
        timesteps: List[int] = None,
        # Sigma值列表
        sigmas: List[float] = None,
        # 可选的去噪开始阈值
        denoising_start: Optional[float] = None,
        # 可选的去噪结束阈值
        denoising_end: Optional[float] = None,
        # 指导比例，默认值为5.0
        guidance_scale: float = 5.0,
        # 可选的负面提示文本
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 可选的第二个负面提示文本
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为1
        num_images_per_prompt: Optional[int] = 1,
        # 预设的η值，默认值为0.0
        eta: float = 0.0,
        # 随机数生成器，可选
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 可选的潜在张量
        latents: Optional[torch.Tensor] = None,
        # 可选的提示嵌入张量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面提示嵌入张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的聚合提示嵌入张量
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面聚合提示嵌入张量
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的图像适配器输入
        ip_adapter_image: Optional[PipelineImageInput] = None,
        # 可选的图像适配器嵌入张量列表
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # 输出类型，默认值为"pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式，默认为True
        return_dict: bool = True,
        # 可选的跨注意力参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 指导重标定值，默认值为0.0
        guidance_rescale: float = 0.0,
        # 原始图像的尺寸
        original_size: Tuple[int, int] = None,
        # 裁剪左上角坐标，默认值为(0, 0)
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 目标图像的尺寸
        target_size: Tuple[int, int] = None,
        # 可选的负面原始尺寸
        negative_original_size: Optional[Tuple[int, int]] = None,
        # 可选的负面裁剪左上角坐标，默认值为(0, 0)
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        # 可选的负面目标尺寸
        negative_target_size: Optional[Tuple[int, int]] = None,
        # 审美评分，默认值为6.0
        aesthetic_score: float = 6.0,
        # 负面审美评分，默认值为2.5
        negative_aesthetic_score: float = 2.5,
        # 可选的剪切跳过参数
        clip_skip: Optional[int] = None,
        # 步骤结束时的回调函数，可选
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        # 结束步骤时的张量输入回调参数，默认为["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # pag比例，默认值为3.0
        pag_scale: float = 3.0,
        # pag自适应比例，默认值为0.0
        pag_adaptive_scale: float = 0.0,
```