# `.\diffusers\pipelines\controlnet\pipeline_controlnet_sd_xl.py`

```py
# 版权声明，2024年HuggingFace团队所有权利
# 
# 根据Apache许可证第2.0版（“许可证”）授权；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件
# 根据许可证分发是基于“原样”基础，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参见许可证以获取特定语言的权限和
# 限制。

# 导入inspect模块，用于检查对象的属性和方法
import inspect
# 从typing模块导入类型注释所需的各种类型
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入numpy库，通常用于数值计算
import numpy as np
# 导入PIL库中的Image模块，用于图像处理
import PIL.Image
# 导入PyTorch库
import torch
# 导入PyTorch中的函数式API，用于深度学习操作
import torch.nn.functional as F
# 从transformers库导入多个CLIP相关模型和处理器
from transformers import (
    CLIPImageProcessor,  # 图像处理器
    CLIPTextModel,  # 文本模型
    CLIPTextModelWithProjection,  # 带有投影的文本模型
    CLIPTokenizer,  # 文本分词器
    CLIPVisionModelWithProjection,  # 带有投影的视觉模型
)

# 从diffusers.utils导入检查隐形水印可用性的函数
from diffusers.utils.import_utils import is_invisible_watermark_available

# 导入多个回调和处理器类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 导入图像处理器相关的类
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 导入加载器的多个混合类
from ...loaders import (
    FromSingleFileMixin,  # 单文件加载混合类
    IPAdapterMixin,  # IP适配器混合类
    StableDiffusionXLLoraLoaderMixin,  # 稳定扩散XL Lora加载混合类
    TextualInversionLoaderMixin,  # 文本反转加载混合类
)
# 导入多个模型类
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
# 导入注意力处理器相关的类
from ...models.attention_processor import (
    AttnProcessor2_0,  # 注意力处理器2.0
    XFormersAttnProcessor,  # XFormers注意力处理器
)
# 从Lora模型中导入调整文本编码器的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 导入调度器相关的类
from ...schedulers import KarrasDiffusionSchedulers
# 从utils模块导入多个实用函数和常量
from ...utils import (
    USE_PEFT_BACKEND,  # 指示是否使用PEFT后端
    deprecate,  # 用于标记弃用的函数
    logging,  # 日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的函数
    scale_lora_layers,  # 调整Lora层的比例
    unscale_lora_layers,  # 反调整Lora层的比例
)
# 从torch_utils导入与PyTorch相关的实用工具
from ...utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
# 从pipeline_utils导入扩散管道及其混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从stable_diffusion_xl导入管道输出类
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

# 如果隐形水印可用，则导入水印类
if is_invisible_watermark_available():
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

# 导入多控制网模型类
from .multicontrolnet import MultiControlNetModel

# 创建一个记录器，用于记录模块中的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，可能包含使用示例
EXAMPLE_DOC_STRING = """
```  
```py  
```  
```py  
```  
```py  
```  
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> # 导入必要的库
        >>> from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
        >>> from diffusers.utils import load_image  # 导入加载图像的工具
        >>> import numpy as np  # 导入 NumPy 库用于数组操作
        >>> import torch  # 导入 PyTorch 库用于深度学习

        >>> import cv2  # 导入 OpenCV 库用于计算机视觉操作
        >>> from PIL import Image  # 导入 PIL 库用于图像处理

        >>> # 定义生成图像的提示文本
        >>> prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
        >>> # 定义负面提示文本以避免生成低质量图像
        >>> negative_prompt = "low quality, bad quality, sketches"

        >>> # 下载一张图像
        >>> image = load_image(
        ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
        ... )  # 从指定URL加载图像

        >>> # 初始化模型和管道
        >>> controlnet_conditioning_scale = 0.5  # 设置控制网的条件缩放比例，推荐用于良好的泛化
        >>> controlnet = ControlNetModel.from_pretrained(
        ...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        ... )  # 从预训练模型加载控制网模型
        >>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)  # 从预训练模型加载变分自编码器
        >>> pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        ... )  # 从预训练模型加载稳定扩散管道并指定控制网和变分自编码器
        >>> pipe.enable_model_cpu_offload()  # 启用模型的 CPU 卸载以节省内存

        >>> # 获取 Canny 边缘检测图像
        >>> image = np.array(image)  # 将图像转换为 NumPy 数组
        >>> image = cv2.Canny(image, 100, 200)  # 使用 Canny 算法进行边缘检测
        >>> image = image[:, :, None]  # 将数组维度扩展以便于后续处理
        >>> image = np.concatenate([image, image, image], axis=2)  # 将单通道图像转换为三通道图像
        >>> canny_image = Image.fromarray(image)  # 从 NumPy 数组创建 PIL 图像

        >>> # 生成图像
        >>> image = pipe(
        ...     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
        ... ).images[0]  # 使用提示和 Canny 图像生成新图像，并提取结果
        ```  
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 导入的 retrieve_timesteps 函数
def retrieve_timesteps(
    # 调度器对象
    scheduler,
    # 推理步骤的数量，默认为 None
    num_inference_steps: Optional[int] = None,
    # 设备类型，默认为 None
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步列表，默认为 None
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 列表，默认为 None
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中获取时间步。处理自定义时间步。任何关键字参数将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步要移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义时间步。如果传递 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义 sigma。如果传递 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是调度器的时间步计划，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了时间步和 sigma
    if timesteps is not None and sigmas is not None:
        # 抛出错误，提示只能选择一个
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了时间步
    if timesteps is not None:
        # 检查调度器是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，自定义时间步不被支持，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器设置时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传递了 sigma
    elif sigmas is not None:
        # 检查调度器是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，自定义 sigma 不被支持，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器设置 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 否则分支，用于设置调度器的时间步
        else:
            # 调用调度器设置推理步数，并指定设备和其他参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器中的时间步列表
            timesteps = scheduler.timesteps
        # 返回时间步列表和推理步数
        return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionXLControlNetPipeline 的类，继承多个父类
class StableDiffusionXLControlNetPipeline(
    # 继承 DiffusionPipeline 类
    DiffusionPipeline,
    # 继承 StableDiffusionMixin 类
    StableDiffusionMixin,
    # 继承 TextualInversionLoaderMixin 类
    TextualInversionLoaderMixin,
    # 继承 StableDiffusionXLLoraLoaderMixin 类
    StableDiffusionXLLoraLoaderMixin,
    # 继承 IPAdapterMixin 类
    IPAdapterMixin,
    # 继承 FromSingleFileMixin 类
    FromSingleFileMixin,
):
    # 文档字符串，描述该管道的功能和用途
    r"""
    使用 Stable Diffusion XL 进行文本到图像生成，并结合 ControlNet 指导。

    该模型继承自 [`DiffusionPipeline`]。有关所有管道的通用方法的文档，请查看超类文档
    （下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    # 文档字符串，说明函数参数及其类型和作用
    Args:
        vae ([`AutoencoderKL`]):  # 变分自编码器模型，用于将图像编码和解码为潜在表示
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):  # 冻结的文本编码器模型，使用 CLIP 进行文本处理
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):  # 第二个冻结的文本编码器
            Second frozen text-encoder
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        tokenizer ([`~transformers.CLIPTokenizer`]):  # 用于文本分词的 CLIP 分词器
            A `CLIPTokenizer` to tokenize text.
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):  # 第二个用于文本分词的 CLIP 分词器
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):  # 用于去噪编码图像潜在表示的 UNet 模型
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):  # 提供额外条件以帮助 UNet 在去噪过程中
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):  # 用于与 UNet 结合使用的调度器，帮助去噪
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):  # 指定负提示嵌入是否应始终设为0
            Whether the negative prompt embeddings should always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):  # 指定是否使用水印库为输出图像添加水印
            Whether to use the [invisible_watermark](https://github.com/ShieldMnt/invisible-watermark/) library to
            watermark output images. If not defined, it defaults to `True` if the package is installed; otherwise no
            watermarker is used.
    """

    # 有意不包含 controlnet，因为它会与 unet 进行迭代
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"  # 定义模型的 CPU 卸载顺序
    _optional_components = [  # 可选组件列表，包含可选使用的模型部分
        "tokenizer",  # 第一个分词器
        "tokenizer_2",  # 第二个分词器
        "text_encoder",  # 第一个文本编码器
        "text_encoder_2",  # 第二个文本编码器
        "feature_extractor",  # 特征提取器
        "image_encoder",  # 图像编码器
    ]
    _callback_tensor_inputs = [  # 回调张量输入列表，定义输入的张量名称
        "latents",  # 潜在表示
        "prompt_embeds",  # 提示嵌入
        "negative_prompt_embeds",  # 负提示嵌入
        "add_text_embeds",  # 添加的文本嵌入
        "add_time_ids",  # 添加的时间 ID
        "negative_pooled_prompt_embeds",  # 负池化提示嵌入
        "negative_add_time_ids",  # 负添加时间 ID
    ]
    # 初始化类的构造函数，接收多个参数
        def __init__(
            self,
            # VAE 模型，用于图像编码和解码
            vae: AutoencoderKL,
            # 文本编码器模型，用于将文本转换为特征
            text_encoder: CLIPTextModel,
            # 另一个文本编码器，带有投影层
            text_encoder_2: CLIPTextModelWithProjection,
            # 文本分词器，将文本分割为词元
            tokenizer: CLIPTokenizer,
            # 另一个文本分词器
            tokenizer_2: CLIPTokenizer,
            # UNet 模型，用于图像生成
            unet: UNet2DConditionModel,
            # 控制网络，可以是单个或多个模型
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
            # 调度器，用于控制生成过程
            scheduler: KarrasDiffusionSchedulers,
            # 是否强制空提示的输出为零
            force_zeros_for_empty_prompt: bool = True,
            # 是否添加水印的可选参数
            add_watermarker: Optional[bool] = None,
            # 特征提取器，用于处理图像特征
            feature_extractor: CLIPImageProcessor = None,
            # 图像编码器，带有投影层
            image_encoder: CLIPVisionModelWithProjection = None,
        ):
            # 调用父类的构造函数
            super().__init__()
    
            # 如果 controlnet 是列表或元组，则创建多控制网络模型
            if isinstance(controlnet, (list, tuple)):
                controlnet = MultiControlNetModel(controlnet)
    
            # 注册各个模块到当前对象
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
            # 初始化图像处理器，转换 RGB 图像
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
            # 初始化控制图像处理器，不进行归一化
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            # 如果没有提供水印参数，则使用可用性检测
            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
    
            # 根据是否添加水印的条件初始化水印对象
            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                self.watermark = None
    
            # 注册配置参数，处理空提示的设置
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
    
        # 从 StableDiffusionXLPipeline 复制的编码提示方法
        def encode_prompt(
            # 输入的提示字符串
            prompt: str,
            # 可选的第二个提示字符串
            prompt_2: Optional[str] = None,
            # 可选的设备参数
            device: Optional[torch.device] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: int = 1,
            # 是否使用分类器自由引导
            do_classifier_free_guidance: bool = True,
            # 可选的负面提示字符串
            negative_prompt: Optional[str] = None,
            # 可选的第二个负面提示字符串
            negative_prompt_2: Optional[str] = None,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的聚合提示嵌入
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面聚合提示嵌入
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 LoRA 缩放因子
            lora_scale: Optional[float] = None,
            # 可选的跳过参数，用于剪辑
        # 从 StableDiffusionPipeline 复制的编码图像方法
    # 定义一个方法用于编码图像，接收图像及其他参数
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的 image 不是张量，则使用特征提取器处理它
            if not isinstance(image, torch.Tensor):
                # 通过特征提取器将图像转换为张量，并返回像素值
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备，并转换为指定数据类型
            image = image.to(device=device, dtype=dtype)
            
            # 如果要求输出隐藏状态
            if output_hidden_states:
                # 使用图像编码器处理图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态在第一维上重复 num_images_per_prompt 次
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 生成与图像大小相同的零张量，并获取其隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 将未条件化的隐藏状态重复 num_images_per_prompt 次
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码后的图像隐藏状态和未条件化的图像隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 使用图像编码器处理图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 将图像嵌入在第一维上重复 num_images_per_prompt 次
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入大小相同的零张量作为未条件化的图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和未条件化的图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的方法
        def prepare_ip_adapter_image_embeds(
            # 定义方法的参数，包括适配器图像、图像嵌入、设备、每个提示的图像数量以及是否进行无分类器自由引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用了分类器自由引导，则初始化一个空列表，用于存储负图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 检查 ip_adapter_image_embeds 是否为 None
        if ip_adapter_image_embeds is None:
            # 如果 ip_adapter_image 不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查 ip_adapter_image 的长度是否与 IP 适配器的数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果不相同，则抛出值错误
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个单独的 IP 适配器图像及其对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断输出隐藏状态是否为真，取决于图像投影层的类型
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 对单个图像进行编码，获取嵌入和负嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用了分类器自由引导，则将负图像嵌入添加到负嵌入列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果已经存在图像嵌入，则遍历这些嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用了分类器自由引导，则分割单个嵌入为负嵌入和正嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将单个嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储 IP 适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历图像嵌入，索引从 i 开始
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入复制 num_images_per_prompt 次
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用了分类器自由引导，则处理负嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负嵌入与正嵌入连接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将单个图像嵌入移动到指定设备上
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回处理后的 IP 适配器图像嵌入
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    # 准备额外的步骤关键字参数，用于调度器的步骤，不同调度器的签名可能不同
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 的取值应在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 创建一个字典用于存储额外的步骤关键字参数
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs

    # 检查输入参数的有效性和完整性
    def check_inputs(
        self,
        prompt,  # 输入的提示词
        prompt_2,  # 第二个输入的提示词
        image,  # 输入的图像
        callback_steps,  # 回调步骤数
        negative_prompt=None,  # 可选的负面提示词
        negative_prompt_2=None,  # 第二个可选的负面提示词
        prompt_embeds=None,  # 提示词的嵌入表示
        negative_prompt_embeds=None,  # 负面提示词的嵌入表示
        pooled_prompt_embeds=None,  # 池化后的提示词嵌入表示
        ip_adapter_image=None,  # 输入适配器的图像
        ip_adapter_image_embeds=None,  # 输入适配器图像的嵌入表示
        negative_pooled_prompt_embeds=None,  # 负面池化提示词的嵌入表示
        controlnet_conditioning_scale=1.0,  # ControlNet 条件缩放因子，默认为 1.0
        control_guidance_start=0.0,  # ControlNet 指导开始的比例，默认为 0.0
        control_guidance_end=1.0,  # ControlNet 指导结束的比例，默认为 1.0
        callback_on_step_end_tensor_inputs=None,  # 步骤结束时的回调张量输入
    # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image 复制
    # 检查输入图像及其相关提示的类型和大小
        def check_image(self, image, prompt, prompt_embeds):
            # 判断输入图像是否为 PIL 图片类型
            image_is_pil = isinstance(image, PIL.Image.Image)
            # 判断输入图像是否为 PyTorch 张量类型
            image_is_tensor = isinstance(image, torch.Tensor)
            # 判断输入图像是否为 NumPy 数组类型
            image_is_np = isinstance(image, np.ndarray)
            # 判断输入是否为 PIL 图片列表
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            # 判断输入是否为 PyTorch 张量列表
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            # 判断输入是否为 NumPy 数组列表
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
            # 如果输入不符合任何图像类型，抛出类型错误
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
    
            # 如果输入为 PIL 图片，则批处理大小为 1
            if image_is_pil:
                image_batch_size = 1
            else:
                # 否则，获取输入图像的批处理大小
                image_batch_size = len(image)
    
            # 如果提示不为空且为字符串，批处理大小为 1
            if prompt is not None and isinstance(prompt, str):
                prompt_batch_size = 1
            # 如果提示为列表，批处理大小为列表长度
            elif prompt is not None and isinstance(prompt, list):
                prompt_batch_size = len(prompt)
            # 如果提示嵌入不为空，获取其批处理大小
            elif prompt_embeds is not None:
                prompt_batch_size = prompt_embeds.shape[0]
    
            # 如果图像批处理大小与提示批处理大小不一致，抛出值错误
            if image_batch_size != 1 and image_batch_size != prompt_batch_size:
                raise ValueError(
                    f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
                )
    
        # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image 复制
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
            # 使用控制图像处理器预处理图像，并转换为浮点32类型
            image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            # 获取图像的批处理大小
            image_batch_size = image.shape[0]
    
            # 如果图像批处理大小为 1，则重复次数为批大小
            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # 否则，重复次数为每个提示的图像数量
                repeat_by = num_images_per_prompt
    
            # 按指定维度重复图像
            image = image.repeat_interleave(repeat_by, dim=0)
    
            # 将图像移动到指定设备并转换为指定类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果启用分类器自由引导且未启用猜测模式，重复图像两次
            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)
    
            # 返回处理后的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    # 准备潜在变量，返回适当形状的张量
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状，包括批量大小和通道数
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器的数量是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果没有传入潜在变量，则随机生成
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果传入潜在变量，将其移动到指定设备
                latents = latents.to(device)
    
            # 根据调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 从 StableDiffusionXLPipeline 复制的函数，获取附加时间 ID
        def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
        ):
            # 创建附加时间 ID 列表，合并原始大小、裁剪坐标和目标大小
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
            # 计算传递的附加嵌入维度
            passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
            )
            # 获取期望的附加嵌入维度
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    
            # 检查传递的维度与期望的维度是否匹配
            if expected_add_embed_dim != passed_add_embed_dim:
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )
    
            # 将附加时间 ID 转换为张量
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
            # 返回附加时间 ID 张量
            return add_time_ids
    
        # 从 StableDiffusionUpscalePipeline 复制的函数，升维 VAE
        def upcast_vae(self):
            # 获取 VAE 的数据类型
            dtype = self.vae.dtype
            # 将 VAE 转换为浮点32位
            self.vae.to(dtype=torch.float32)
            # 检查是否使用了 Torch 2.0 或 XFormers
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                ),
            )
            # 如果使用了 XFormers 或 Torch 2.0，注意力模块不需要保持浮点32位，节省内存
            if use_torch_2_0_or_xformers:
                self.vae.post_quant_conv.to(dtype)
                self.vae.decoder.conv_in.to(dtype)
                self.vae.decoder.mid_block.to(dtype)
    
        # 从 LatentConsistencyModelPipeline 复制的函数，获取引导缩放嵌入
    # 定义一个方法，获取具有引导尺度的嵌入向量
    def get_guidance_scale_embedding(
        # 输入的张量 w
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        参考链接：https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        参数：
            w (`torch.Tensor`):
                生成具有指定引导尺度的嵌入向量，以随后丰富时间步嵌入。
            embedding_dim (`int`, *可选*, 默认值为 512):
                要生成的嵌入的维度。
            dtype (`torch.dtype`, *可选*, 默认值为 `torch.float32`):
                生成的嵌入的数据类型。

        返回：
            `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
        """
        # 确保输入张量 w 是一维的
        assert len(w.shape) == 1
        # 将 w 的值乘以 1000.0
        w = w * 1000.0

        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算每个嵌入的基础值
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成一个指数衰减的嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将 w 转换为指定的数据类型，并与嵌入相乘
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 将正弦和余弦嵌入连接在一起
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度为奇数，则在最后填充一个零
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保输出嵌入的形状为 (w.shape[0], embedding_dim)
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb

    # 定义一个属性，用于获取引导尺度
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义一个属性，用于获取剪切跳过的值
    @property
    def clip_skip(self):
        return self._clip_skip

    # 定义一个属性，判断是否进行无分类器引导
    # 此处的 `guidance_scale` 定义类似于 Imagen 论文中的引导权重 `w`（公式 (2)）
    # `guidance_scale = 1` 对应于不进行无分类器引导。
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 定义一个属性，用于获取交叉注意力的参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 定义一个属性，用于获取去噪结束的值
    @property
    def denoising_end(self):
        return self._denoising_end

    # 定义一个属性，用于获取时间步的数量
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 装饰器，禁止在这个方法内计算梯度
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的方法，接受多个参数以执行特定功能
        def __call__(
            # 提示信息，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 第二个提示信息，选填，可以是字符串或字符串列表
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 输入图像，可以是特定类型
            image: PipelineImageInput = None,
            # 输出图像的高度，选填
            height: Optional[int] = None,
            # 输出图像的宽度，选填
            width: Optional[int] = None,
            # 推理步骤的数量，默认值为50
            num_inference_steps: int = 50,
            # 定义时间步，选填，默认为None
            timesteps: List[int] = None,
            # 噪声标准差列表，选填，默认为None
            sigmas: List[float] = None,
            # 去噪结束的时间点，选填，默认为None
            denoising_end: Optional[float] = None,
            # 指导缩放因子，默认值为5.0
            guidance_scale: float = 5.0,
            # 负向提示信息，选填，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负向提示信息，选填，可以是字符串或字符串列表
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 影响采样过程的参数，默认为0.0
            eta: float = 0.0,
            # 随机数生成器，选填，可以是单个或列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 先前生成的潜在表示，选填
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，选填
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负向提示嵌入，选填
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 聚合后的提示嵌入，选填
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负向聚合后的提示嵌入，选填
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像，选填
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像的嵌入，选填
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为True
            return_dict: bool = True,
            # 跨注意力相关的参数，选填
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # ControlNet条件缩放因子，默认为1.0
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            # 猜测模式，默认为False
            guess_mode: bool = False,
            # ControlNet指导开始时间，默认为0.0
            control_guidance_start: Union[float, List[float]] = 0.0,
            # ControlNet指导结束时间，默认为1.0
            control_guidance_end: Union[float, List[float]] = 1.0,
            # 原始图像大小，选填
            original_size: Tuple[int, int] = None,
            # 图像左上角的裁剪坐标，默认为(0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标图像大小，选填
            target_size: Tuple[int, int] = None,
            # 负向图像的原始大小，选填
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负向图像的左上角裁剪坐标，默认为(0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负向图像的目标大小，选填
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 跳过剪辑的参数，选填
            clip_skip: Optional[int] = None,
            # 在步骤结束时调用的回调函数，选填
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 在步骤结束时的张量输入回调，默认为["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 接收额外关键字参数
            **kwargs,
```