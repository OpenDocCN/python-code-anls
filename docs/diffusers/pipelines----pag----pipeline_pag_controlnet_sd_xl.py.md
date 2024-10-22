# `.\diffusers\pipelines\pag\pipeline_pag_controlnet_sd_xl.py`

```py
# 版权声明，说明此代码由 HuggingFace 团队版权所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 您不得在未遵守许可证的情况下使用此文件。
# 您可以在以下网址获取许可证副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，
# 否则根据许可证分发的软件按“原样”提供，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证，以获取有关许可和
# 限制的特定语言。

# 导入 inspect 模块，用于获取对象的信息
import inspect
# 从 typing 模块导入类型提示所需的多个类型
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 numpy 库，通常用于数组和数值计算
import numpy as np
# 导入 PIL 库中的 Image 模块，用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 PyTorch 中导入功能性模块
import torch.nn.functional as F
# 从 transformers 库中导入 CLIP 相关的处理器和模型
from transformers import (
    CLIPImageProcessor,  # CLIP 图像处理器
    CLIPTextModel,  # CLIP 文本模型
    CLIPTextModelWithProjection,  # 带投影的 CLIP 文本模型
    CLIPTokenizer,  # CLIP 分词器
    CLIPVisionModelWithProjection,  # 带投影的 CLIP 视觉模型
)

# 从 diffusers.utils 中导入一个工具函数，检查水印是否可用
from diffusers.utils.import_utils import is_invisible_watermark_available

# 从当前目录的回调模块中导入回调相关类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 从当前目录的图像处理模块中导入图像输入和 VAE 图像处理器
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 从当前目录的加载器模块中导入多个加载器类
from ...loaders import (
    FromSingleFileMixin,  # 单文件加载混合器
    IPAdapterMixin,  # IP 适配器混合器
    StableDiffusionXLLoraLoaderMixin,  # 稳定扩散 XL Lora 加载器混合器
    TextualInversionLoaderMixin,  # 文本反转加载器混合器
)
# 从当前目录的模型模块中导入多种模型类
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
# 从当前目录的注意力处理器模块中导入多个注意力处理器
from ...models.attention_processor import (
    AttnProcessor2_0,  # 版本 2.0 的注意力处理器
    XFormersAttnProcessor,  # XFormers 注意力处理器
)
# 从当前目录的 Lora 模型中导入 Lora 相关函数
from ...models.lora import adjust_lora_scale_text_encoder
# 从当前目录的调度器模块中导入 Karras 扩散调度器
from ...schedulers import KarrasDiffusionSchedulers
# 从当前目录的工具模块中导入多个工具函数和常量
from ...utils import (
    USE_PEFT_BACKEND,  # 使用 PEFT 后端的标志
    logging,  # 日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 缩放 Lora 层的函数
    unscale_lora_layers,  # 取消缩放 Lora 层的函数
)
# 从当前目录的 torch_utils 模块中导入 PyTorch 相关的工具函数
from ...utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
# 从当前目录的管道工具模块中导入扩散管道和稳定扩散混合器
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从稳定扩散 XL 的管道输出模块中导入输出类
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
# 从当前目录的 pag_utils 模块中导入 PAG 相关混合器
from .pag_utils import PAGMixin

# 如果不可见水印可用，则导入相应的水印处理器
if is_invisible_watermark_available():
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

# 从控制网模块中导入多控制网模型
from ..controlnet.multicontrolnet import MultiControlNetModel

# 创建日志记录器，用于记录模块中的日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义示例文档字符串，用于说明代码示例的格式
EXAMPLE_DOC_STRING = """
``` 
```py 
``` 
```py 
``` 
    # 示例代码
        Examples:
            ```py
            >>> # 安装所需库
            >>> # !pip install opencv-python transformers accelerate
            >>> # 导入所需的库
            >>> from diffusers import AutoPipelineForText2Image, ControlNetModel, AutoencoderKL
            >>> from diffusers.utils import load_image
            >>> import numpy as np
            >>> import torch
    
            >>> import cv2
            >>> from PIL import Image
    
            >>> # 设置生成图像的提示信息
            >>> prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
            >>> # 设置负面提示信息
            >>> negative_prompt = "low quality, bad quality, sketches"
    
            >>> # 下载一张图像
            >>> image = load_image(
            ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
            ... )
    
            >>> # 初始化模型和管道
            >>> controlnet_conditioning_scale = 0.5  # 推荐用于良好的泛化
            >>> # 从预训练模型加载 ControlNet
            >>> controlnet = ControlNetModel.from_pretrained(
            ...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
            ... )
            >>> # 从预训练模型加载自动编码器
            >>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            >>> # 从预训练模型加载文本到图像的自动管道
            >>> pipe = AutoPipelineForText2Image.from_pretrained(
            ...     "stabilityai/stable-diffusion-xl-base-1.0",
            ...     controlnet=controlnet,
            ...     vae=vae,
            ...     torch_dtype=torch.float16,
            ...     enable_pag=True,
            ... )
            >>> # 启用模型的 CPU 内存卸载
            >>> pipe.enable_model_cpu_offload()
    
            >>> # 获取 Canny 边缘检测图像
            >>> image = np.array(image)
            >>> # 应用 Canny 边缘检测
            >>> image = cv2.Canny(image, 100, 200)
            >>> # 增加维度以适应图像格式
            >>> image = image[:, :, None]
            >>> # 将单通道图像扩展为三通道
            >>> image = np.concatenate([image, image, image], axis=2)
            >>> # 从数组创建图像对象
            >>> canny_image = Image.fromarray(image)
    
            >>> # 生成图像
            >>> image = pipe(
            ...     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image, pag_scale=0.3
            ... ).images[0]
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 导入的函数
def retrieve_timesteps(
    # 调度器对象，用于获取时间步
    scheduler,
    # 用于生成样本的推理步骤数量，可选
    num_inference_steps: Optional[int] = None,
    # 指定时间步移动的设备，可选
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步，可选
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 值，可选
    sigmas: Optional[List[float]] = None,
    # 额外的关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器获取时间步。处理自定义时间步。
    任何关键字参数将被传递给 `scheduler.set_timesteps`。
    
    参数：
        scheduler (`SchedulerMixin`):
            要获取时间步的调度器。
        num_inference_steps (`int`):
            用于生成样本的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            要移动时间步的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义 sigma 值，用于覆盖调度器的时间步间隔策略。如果传递 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回：
        `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是调度器的时间步计划，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传入了时间步和 sigma 值
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传入了时间步
    if timesteps is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受时间步参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持时间步，则抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法设置时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传入了 sigma 值
    elif sigmas is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受 sigma 参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持 sigma，则抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法设置 sigma 值
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果不是第一个条件的情况
        else:
            # 设置推理步骤的时间步，指定设备和其他参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器中的时间步
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤的数量
        return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionXLControlNetPAGPipeline 的类，继承多个混合类
class StableDiffusionXLControlNetPAGPipeline(
    # 继承自 DiffusionPipeline 类，提供扩散管道的基本功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin 类，增加稳定扩散相关功能
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin 类，用于加载文本反演嵌入
    TextualInversionLoaderMixin,
    # 继承自 StableDiffusionXLLoraLoaderMixin 类，用于加载和保存 LoRA 权重
    StableDiffusionXLLoraLoaderMixin,
    # 继承自 IPAdapterMixin 类，用于加载 IP 适配器
    IPAdapterMixin,
    # 继承自 FromSingleFileMixin 类，用于从单个文件加载模型
    FromSingleFileMixin,
    # 继承自 PAGMixin 类，提供与 PAG 相关的功能
    PAGMixin,
):
    # 文档字符串，描述此类的用途
    r"""
    使用 Stable Diffusion XL 和 ControlNet 引导进行文本到图像生成的管道。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取所有管道的通用方法
    （下载、保存、在特定设备上运行等）。

    该管道还继承了以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    """
    # 函数参数说明
    Args:
        # 变分自编码器模型，用于编码和解码图像与潜在表示之间的转换
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        # 冻结的文本编码器，用于处理文本输入
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        # 第二个冻结文本编码器，提供更多文本处理能力
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):
            Second frozen text-encoder
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        # 用于将文本标记化的 CLIPTokenizer
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        # 另一个用于文本标记化的 CLIPTokenizer
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        # 用于去噪编码后图像潜在表示的 UNet 模型
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        # 提供额外条件给 UNet 的 ControlNet 模型，可以是单个或多个
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        # 与 UNet 配合使用的调度器，用于去噪图像潜在表示
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        # 是否始终将负面提示嵌入设置为0的布尔值，默认为真
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings should always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        # 是否使用隐形水印库在输出图像上添加水印的布尔值
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark](https://github.com/ShieldMnt/invisible-watermark/) library to
            watermark output images. If not defined, it defaults to `True` if the package is installed; otherwise no
            watermarker is used.
    """

    # 有意不包括 controlnet，因为它与 unet 迭代
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    # 可选组件列表，包括不同的编码器和标记器
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "feature_extractor",
        "image_encoder",
    ]
    # 回调张量输入的列表，包含潜在向量和嵌入
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]
    # 初始化方法，定义对象的基本属性和参数
        def __init__(
            self,
            vae: AutoencoderKL,  # 自动编码器模型
            text_encoder: CLIPTextModel,  # 文本编码器模型
            text_encoder_2: CLIPTextModelWithProjection,  # 第二个文本编码器，带投影功能
            tokenizer: CLIPTokenizer,  # 用于文本分词的工具
            tokenizer_2: CLIPTokenizer,  # 第二个分词工具
            unet: UNet2DConditionModel,  # UNet模型，用于生成任务
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],  # 控制网络模型，可以是单个或多个模型
            scheduler: KarrasDiffusionSchedulers,  # 用于调度的扩散调度器
            force_zeros_for_empty_prompt: bool = True,  # 控制是否对空提示强制零
            add_watermarker: Optional[bool] = None,  # 可选参数，用于添加水印
            feature_extractor: CLIPImageProcessor = None,  # 可选的图像特征提取器
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选的图像编码器，带投影功能
            pag_applied_layers: Union[str, List[str]] = "mid",  # 应用的层，默认为“mid”
        ):
            super().__init__()  # 调用父类的初始化方法
    
            # 如果 controlnet 是列表或元组，将其转换为 MultiControlNetModel
            if isinstance(controlnet, (list, tuple)):
                controlnet = MultiControlNetModel(controlnet)
    
            # 注册模型组件到当前对象
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 初始化图像处理器，用于 VAE
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
            # 初始化控制图像处理器，用于 VAE
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            # 确定是否添加水印，如果未提供则根据可用性设置
            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
    
            # 根据参数决定是否创建水印对象
            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()  # 创建水印对象
            else:
                self.watermark = None  # 不创建水印对象
    
            # 将配置注册到当前对象
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 设置应用的层
            self.set_pag_applied_layers(pag_applied_layers)
    
        # 从外部库复制的编码提示方法
        def encode_prompt(
            self,
            prompt: str,  # 主提示字符串
            prompt_2: Optional[str] = None,  # 可选的第二个提示字符串
            device: Optional[torch.device] = None,  # 可选的设备参数
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            do_classifier_free_guidance: bool = True,  # 是否使用无分类器引导
            negative_prompt: Optional[str] = None,  # 可选的负提示字符串
            negative_prompt_2: Optional[str] = None,  # 可选的第二个负提示字符串
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入张量
            pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的池化提示嵌入张量
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的池化负提示嵌入张量
            lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
            clip_skip: Optional[int] = None,  # 可选的跳过剪辑层参数
        # 从外部库复制的编码图像方法
    # 定义编码图像的函数，输入包括图像、设备、每个提示的图像数量和可选的隐藏状态输出
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入图像不是张量，则使用特征提取器将其转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并转换为所需的数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态，则获取编码后的隐藏状态
            if output_hidden_states:
                # 通过图像编码器编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 重复隐藏状态以匹配每个提示的图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对于未条件的图像，通过零张量编码并获取隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 重复未条件隐藏状态以匹配每个提示的图像数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码后的图像和未条件的图像隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要输出隐藏状态，则获取编码后的图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 重复图像嵌入以匹配每个提示的图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的零张量作为未条件的图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回编码后的图像嵌入和未条件的图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 StableDiffusionPipeline 类中复制的函数，用于准备 IP 适配器的图像嵌入
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用分类器自由引导，则初始化一个空列表，用于存储负图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器的图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 如果输入适配器图像不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像的长度是否与 IP 适配器的层数匹配
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果长度不匹配，抛出值错误
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历输入适配器图像和对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断当前图像投影层是否是 ImageProjection 类型，输出隐藏状态的标志
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 对单个输入适配器图像进行编码，获取嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到图像嵌入列表中，并扩展维度
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用分类器自由引导，添加负图像嵌入
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果输入适配器的图像嵌入不为空，遍历这些嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用分类器自由引导，将嵌入拆分为负嵌入和正嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 添加负图像嵌入
                    negative_image_embeds.append(single_negative_image_embeds)
                # 添加正图像嵌入
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储最终的适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历每个图像嵌入及其索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入扩展到 num_images_per_prompt 的数量
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用分类器自由引导，将负图像嵌入扩展
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入和正图像嵌入合并
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到最终列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回适配器图像嵌入的列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    # 准备额外的参数以供调度器步骤使用，因为并非所有调度器具有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # 检查调度器的步骤函数是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数字典
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的步骤函数是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回额外参数字典
        return extra_step_kwargs
    
    # 从 diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.check_inputs 复制
    def check_inputs(
        self,
        prompt,
        prompt_2,
        image,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        negative_pooled_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
        # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image 复制
    # 检查图像的类型和尺寸，确保其与提示的批量大小一致
        def check_image(self, image, prompt, prompt_embeds):
            # 判断图像是否为 PIL 图片类型
            image_is_pil = isinstance(image, PIL.Image.Image)
            # 判断图像是否为 Torch 张量类型
            image_is_tensor = isinstance(image, torch.Tensor)
            # 判断图像是否为 NumPy 数组类型
            image_is_np = isinstance(image, np.ndarray)
            # 判断图像是否为 PIL 图片的列表
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            # 判断图像是否为 Torch 张量的列表
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            # 判断图像是否为 NumPy 数组的列表
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
            # 如果图像不是以上任何一种类型，则抛出类型错误
            if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
            ):
                raise TypeError(
                    f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
                )
    
            # 如果图像是 PIL 图片，设置批量大小为 1
            if image_is_pil:
                image_batch_size = 1
            else:
                # 否则，批量大小为图像的长度
                image_batch_size = len(image)
    
            # 如果提示不为空且为字符串，设置提示的批量大小为 1
            if prompt is not None and isinstance(prompt, str):
                prompt_batch_size = 1
            # 如果提示为列表，设置提示的批量大小为列表的长度
            elif prompt is not None and isinstance(prompt, list):
                prompt_batch_size = len(prompt)
            # 如果提示嵌入不为空，设置批量大小为提示嵌入的形状的第一维
            elif prompt_embeds is not None:
                prompt_batch_size = prompt_embeds.shape[0]
    
            # 如果图像批量大小不为 1 且与提示批量大小不相等，抛出值错误
            if image_batch_size != 1 and image_batch_size != prompt_batch_size:
                raise ValueError(
                    f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
                )
    
        # 处理图像以适应模型输入的准备过程
        def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            # 预处理图像，调整大小并转换为指定的数据类型
            image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            # 获取处理后图像的批量大小
            image_batch_size = image.shape[0]
    
            # 如果图像批量大小为 1，重复次数为 batch_size
            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # 否则，重复次数为每个提示的图像数量
                repeat_by = num_images_per_prompt
    
            # 按照重复次数扩展图像维度
            image = image.repeat_interleave(repeat_by, dim=0)
    
            # 将图像转移到指定的设备和数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果启用分类器自由引导且不在猜测模式下，将图像重复两次
            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)
    
            # 返回处理后的图像
            return image
    
        # 复制自 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    # 准备潜在向量，定义形状和其他参数
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在向量的形状，考虑批量大小和维度缩放
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器列表长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果没有提供潜在向量，则生成随机潜在向量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 将现有潜在向量转移到指定设备
                latents = latents.to(device)
    
            # 根据调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在向量
            return latents
    
        # 从 diffusers.pipelines.stable_diffusion_xl 复制的方法，获取添加的时间ID
        def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
        ):
            # 创建包含原始大小、裁剪坐标和目标大小的时间ID列表
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
            # 计算通过添加时间嵌入维度的总维度
            passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
            )
            # 获取模型期望的添加时间嵌入维度
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    
            # 检查通过的维度与期望维度是否匹配
            if expected_add_embed_dim != passed_add_embed_dim:
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )
    
            # 将添加时间ID转换为指定类型的张量
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
            # 返回添加时间ID的张量
            return add_time_ids
    
        # 从 diffusers.pipelines.latent_consistency_models 复制的方法，提升 VAE 精度
        def upcast_vae(self):
            # 获取 VAE 的数据类型
            dtype = self.vae.dtype
            # 将 VAE 转换为 float32 类型
            self.vae.to(dtype=torch.float32)
            # 检查是否使用 Torch 2.0 或 XFormers 处理器
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                ),
            )
            # 如果使用了 XFormers 或 Torch 2.0，则无需将注意力块保持为 float32，可以节省大量内存
            if use_torch_2_0_or_xformers:
                # 将各个部分转换为之前保存的 dtype
                self.vae.post_quant_conv.to(dtype)
                self.vae.decoder.conv_in.to(dtype)
                self.vae.decoder.mid_block.to(dtype)
    
        # 从 diffusers.pipelines.latent_consistency_models 复制的方法，获取指导缩放嵌入
    # 定义获取指导尺度嵌入的函数，接受输入张量 w，嵌入维度和数据类型
    def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
            # 文档字符串，提供函数的链接和参数说明
            """
            See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
            Args:
                w (`torch.Tensor`):
                    Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
                embedding_dim (`int`, *optional*, defaults to 512):
                    Dimension of the embeddings to generate.
                dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                    Data type of the generated embeddings.
    
            Returns:
                `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
            """
            # 确保输入张量 w 是一维的
            assert len(w.shape) == 1
            # 将 w 乘以 1000.0，以调整指导尺度
            w = w * 1000.0
    
            # 计算嵌入维度的一半
            half_dim = embedding_dim // 2
            # 计算指数衰减的常数
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 生成衰减的嵌入向量
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 根据输入张量和嵌入向量生成最终的嵌入
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 将正弦和余弦值拼接成最终的嵌入
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度为奇数，则进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保输出的嵌入形状符合预期
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回生成的嵌入张量
            return emb
    
        # 定义指导尺度的属性，返回内部变量
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 定义跳过剪辑的属性，返回内部变量
        @property
        def clip_skip(self):
            return self._clip_skip
    
        # 定义是否进行无分类器指导的属性，基于指导尺度的值
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 定义交叉注意力的参数属性，返回内部变量
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 定义去噪结束的属性，返回内部变量
        @property
        def denoising_end(self):
            return self._denoising_end
    
        # 定义时间步数的属性，返回内部变量
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 禁用梯度计算装饰器，确保在不计算梯度的情况下运行
        @torch.no_grad()
        # 替换示例文档字符串的装饰器
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用方法，允许对象被当作函数使用
        def __call__(
            # 提示文本，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 第二个提示文本，默认为 None
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 输入图像，类型为 PipelineImageInput
            image: PipelineImageInput = None,
            # 输出图像高度，默认为 None
            height: Optional[int] = None,
            # 输出图像宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤数，默认为 50
            num_inference_steps: int = 50,
            # 时间步列表，默认为 None
            timesteps: List[int] = None,
            # σ 值列表，默认为 None
            sigmas: List[float] = None,
            # 去噪结束值，默认为 None
            denoising_end: Optional[float] = None,
            # 指导比例，默认为 5.0
            guidance_scale: float = 5.0,
            # 负提示文本，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负提示文本，默认为 None
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # η 值，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在表示，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化的提示嵌入，默认为 None
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负池化提示嵌入，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像嵌入列表，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 交叉注意力的参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # ControlNet 条件缩放，默认为 1.0
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            # 控制指导开始值，默认为 0.0
            control_guidance_start: Union[float, List[float]] = 0.0,
            # 控制指导结束值，默认为 1.0
            control_guidance_end: Union[float, List[float]] = 1.0,
            # 原始图像大小，默认为 None
            original_size: Tuple[int, int] = None,
            # 裁剪坐标的左上角，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标图像大小，默认为 None
            target_size: Tuple[int, int] = None,
            # 负原始图像大小，默认为 None
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负裁剪坐标的左上角，默认为 (0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负目标图像大小，默认为 None
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 剪切跳过的数量，默认为 None
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 结束时张量输入的回调，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # PAG 缩放，默认为 3.0
            pag_scale: float = 3.0,
            # 自适应 PAG 缩放，默认为 0.0
            pag_adaptive_scale: float = 0.0,
```