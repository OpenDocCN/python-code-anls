# `.\diffusers\pipelines\t2i_adapter\pipeline_stable_diffusion_xl_adapter.py`

```py
# 版权声明，表明该代码归腾讯ARC和HuggingFace团队所有
# 
# 根据Apache许可证第2.0版（“许可证”）许可；
# 除非遵循许可证，否则不得使用此文件。
# 可在以下网址获得许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面同意，否则根据许可证分发的软件
# 是按“原样”基础提供的，不附带任何明示或暗示的保证或条件。
# 有关许可证下特定权限和限制的更多信息，请参阅许可证。

import inspect  # 导入inspect模块，用于获取对象的签名和文档等信息
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 从typing模块导入类型注解

import numpy as np  # 导入numpy库，用于数值计算
import PIL.Image  # 导入PIL库的Image模块，用于图像处理
import torch  # 导入PyTorch库，用于深度学习
from transformers import (  # 从transformers库导入多个模型和处理器
    CLIPImageProcessor,  # 导入CLIP图像处理器
    CLIPTextModel,  # 导入CLIP文本模型
    CLIPTextModelWithProjection,  # 导入带有投影的CLIP文本模型
    CLIPTokenizer,  # 导入CLIP分词器
    CLIPVisionModelWithProjection,  # 导入带有投影的CLIP视觉模型
)

from ...image_processor import PipelineImageInput, VaeImageProcessor  # 从相对路径导入图像处理器相关类
from ...loaders import (  # 从相对路径导入加载器相关混合类
    FromSingleFileMixin,  # 导入单文件加载混合类
    IPAdapterMixin,  # 导入IP适配器混合类
    StableDiffusionXLLoraLoaderMixin,  # 导入稳定扩散XL Lora加载混合类
    TextualInversionLoaderMixin,  # 导入文本反转加载混合类
)
from ...models import AutoencoderKL, ImageProjection, MultiAdapter, T2IAdapter, UNet2DConditionModel  # 从相对路径导入多个模型
from ...models.attention_processor import (  # 从相对路径导入注意力处理器
    AttnProcessor2_0,  # 导入2.0版本的注意力处理器
    XFormersAttnProcessor,  # 导入XFormers注意力处理器
)
from ...models.lora import adjust_lora_scale_text_encoder  # 从相对路径导入调整Lora规模的文本编码器函数
from ...schedulers import KarrasDiffusionSchedulers  # 从相对路径导入Karras扩散调度器
from ...utils import (  # 从相对路径导入多个实用工具
    PIL_INTERPOLATION,  # 导入PIL插值方法
    USE_PEFT_BACKEND,  # 导入是否使用PEFT后端的标志
    logging,  # 导入日志模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入缩放Lora层的函数
    unscale_lora_layers,  # 导入反缩放Lora层的函数
)
from ...utils.torch_utils import randn_tensor  # 从相对路径导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从相对路径导入扩散管道和稳定扩散混合类
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput  # 从相对路径导入稳定扩散XL管道输出类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例
EXAMPLE_DOC_STRING = """  # 定义示例文档字符串
``` 
    # 示例代码，展示如何使用 T2IAdapter 进行图像生成
    Examples:
        ```py
        # 导入 PyTorch 库
        >>> import torch
        # 从 diffusers 库导入必要的类
        >>> from diffusers import T2IAdapter, StableDiffusionXLAdapterPipeline, DDPMScheduler
        >>> from diffusers.utils import load_image

        # 加载并转换指定 URL 的图像为灰度图像
        >>> sketch_image = load_image("https://huggingface.co/Adapter/t2iadapter/resolve/main/sketch.png").convert("L")

        # 指定模型的 ID
        >>> model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        # 从预训练模型加载适配器
        >>> adapter = T2IAdapter.from_pretrained(
        ...     "Adapter/t2iadapter",  # 适配器的路径
        ...     subfolder="sketch_sdxl_1.0",  # 子文件夹名
        ...     torch_dtype=torch.float16,  # 设置数据类型为 float16
        ...     adapter_type="full_adapter_xl",  # 适配器类型
        ... )
        # 从预训练模型加载调度器
        >>> scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # 创建 StableDiffusionXLAdapterPipeline 对象并将其移动到 GPU
        >>> pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        ...     model_id,  # 模型 ID
        ...     adapter=adapter,  # 加载的适配器
        ...     torch_dtype=torch.float16,  # 设置数据类型为 float16
        ...     variant="fp16",  # 变体设置为 fp16
        ...     scheduler=scheduler  # 使用的调度器
        ... ).to("cuda")  # 将管道移动到 GPU

        # 设置随机种子以确保结果可复现
        >>> generator = torch.manual_seed(42)
        # 使用管道生成图像，提供提示和负面提示
        >>> sketch_image_out = pipe(
        ...     prompt="a photo of a dog in real world, high quality",  # 正面提示
        ...     negative_prompt="extra digit, fewer digits, cropped, worst quality, low quality",  # 负面提示
        ...     image=sketch_image,  # 输入的草图图像
        ...     generator=generator,  # 随机数生成器
        ...     guidance_scale=7.5,  # 引导比例
        ... ).images[0]  # 获取生成的图像
        ```
"""
# 预处理适配器图像的函数定义，接受图像、高度和宽度作为参数
def _preprocess_adapter_image(image, height, width):
    # 检查图像是否为 PyTorch 张量
    if isinstance(image, torch.Tensor):
        # 如果是，直接返回该张量
        return image
    # 检查图像是否为 PIL 图像
    elif isinstance(image, PIL.Image.Image):
        # 将单个图像转换为列表
        image = [image]

    # 如果图像的第一个元素是 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 将每个图像调整为指定的高度和宽度，并转换为 NumPy 数组
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        # 扩展数组的维度，以确保其格式为 [b, h, w, c]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        # 将所有图像沿第一个轴拼接成一个大数组
        image = np.concatenate(image, axis=0)
        # 将图像数据类型转换为浮点型并进行归一化
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组的维度顺序
        image = image.transpose(0, 3, 1, 2)
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果图像的第一个元素是 PyTorch 张量
    elif isinstance(image[0], torch.Tensor):
        # 检查维度并堆叠或拼接张量
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            # 如果维度不符合预期，则抛出错误
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    # 返回处理后的图像
    return image


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制的函数定义
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 对 `noise_cfg` 进行重新缩放。基于 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 的发现。参见第 3.4 节
    """
    # 计算噪声预测文本的标准差
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 通过标准差进行结果的重新缩放（修复过曝问题）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 按照 `guidance_rescale` 比例混合原始结果，以避免“普通”图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的噪声配置
    return noise_cfg


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 复制的函数定义
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器中检索时间步。处理自定义时间步。所有的关键字参数将传递给 `scheduler.set_timesteps`。
    # 函数参数说明
    Args:
        scheduler (`SchedulerMixin`):  # 调度器，用于获取时间步
            The scheduler to get timesteps from.
        num_inference_steps (`int`):  # 生成样本时使用的扩散步数
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):  # 指定移动时间步的设备
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):  # 自定义时间步，覆盖调度器的时间步策略
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):  # 自定义sigma，覆盖调度器的时间步策略
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    # 返回一个元组，包含时间步调度和推理步骤数
    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    # 如果同时传入时间步和sigma，抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传入了时间步
    if timesteps is not None:
        # 检查调度器是否支持自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，抛出错误
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
    # 如果传入了sigma
    elif sigmas is not None:
        # 检查调度器是否支持自定义sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取当前调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果都没有传入，使用默认推理步骤数
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取当前调度器的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤数
    return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionXLAdapterPipeline 的类，该类继承多个混入类和主类
class StableDiffusionXLAdapterPipeline(
    # 从 DiffusionPipeline 继承，提供扩散管道的基础功能
    DiffusionPipeline,
    # 从 StableDiffusionMixin 继承，提供稳定扩散相关的功能
    StableDiffusionMixin,
    # 从 TextualInversionLoaderMixin 继承，提供文本反转加载功能
    TextualInversionLoaderMixin,
    # 从 StableDiffusionXLLoraLoaderMixin 继承，提供 LoRA 权重加载和保存功能
    StableDiffusionXLLoraLoaderMixin,
    # 从 IPAdapterMixin 继承，提供 IP 适配器加载功能
    IPAdapterMixin,
    # 从 FromSingleFileMixin 继承，提供从单个文件加载功能
    FromSingleFileMixin,
):
    # 文档字符串，说明该管道的用途和背景
    r"""
    使用增强 T2I-Adapter 的 Stable Diffusion 进行文本到图像生成的管道
    https://arxiv.org/abs/2302.08453

    此模型继承自 [`DiffusionPipeline`]。查看超类文档以获取库为所有管道实现的通用方法（例如下载、保存、在特定设备上运行等）

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    """
    # 参数说明部分，描述类的初始化所需参数
        Args:
            adapter ([`T2IAdapter`] or [`MultiAdapter`] or `List[T2IAdapter]`):
                提供在去噪过程中对 unet 的额外条件。如果将多个适配器作为列表设置，
                则每个适配器的输出将相加以创建一个合并的额外条件。
            adapter_weights (`List[float]`, *optional*, defaults to None):
                表示每个适配器输出前加权的浮点数列表，权重将在相加前乘以各适配器的输出。
            vae ([`AutoencoderKL`]):
                变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。
            text_encoder ([`CLIPTextModel`]):
                冻结的文本编码器。稳定扩散使用
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 的文本部分，
                具体是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
            tokenizer (`CLIPTokenizer`):
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 类的分词器。
            unet ([`UNet2DConditionModel`]): 条件 U-Net 架构，用于去噪编码后的图像潜在值。
            scheduler ([`SchedulerMixin`]):
                用于与 `unet` 结合使用的调度器，以去噪编码的图像潜在值。可以是
                [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的一个。
            safety_checker ([`StableDiffusionSafetyChecker`]):
                分类模块，估计生成的图像是否可能被认为是冒犯性或有害的。
                请参考 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5) 获取详细信息。
            feature_extractor ([`CLIPImageProcessor`]):
                从生成图像中提取特征的模型，用作 `safety_checker` 的输入。
        """
    
        # 定义模型的 CPU 卸载顺序
        model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
        # 可选组件的列表，包括多个可能使用的模型组件
        _optional_components = [
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2",
            "feature_extractor",
            "image_encoder",
        ]
    
        # 初始化方法，设置模型的各种组件
        def __init__(
            self,
            vae: AutoencoderKL,  # 初始化变分自编码器
            text_encoder: CLIPTextModel,  # 初始化文本编码器
            text_encoder_2: CLIPTextModelWithProjection,  # 初始化第二个文本编码器
            tokenizer: CLIPTokenizer,  # 初始化第一个分词器
            tokenizer_2: CLIPTokenizer,  # 初始化第二个分词器
            unet: UNet2DConditionModel,  # 初始化条件 U-Net 模型
            adapter: Union[T2IAdapter, MultiAdapter, List[T2IAdapter]],  # 初始化适配器
            scheduler: KarrasDiffusionSchedulers,  # 初始化调度器
            force_zeros_for_empty_prompt: bool = True,  # 是否为空提示强制使用零
            feature_extractor: CLIPImageProcessor = None,  # 初始化特征提取器，默认为 None
            image_encoder: CLIPVisionModelWithProjection = None,  # 初始化图像编码器，默认为 None
    # 调用父类的初始化方法
    ):
        super().__init__()

        # 注册各种模块，包括 VAE、文本编码器等
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            adapter=adapter,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        # 将配置项注册，包括强制对空提示的零处理
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化图像处理器，使用 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 获取默认样本大小
        self.default_sample_size = self.unet.config.sample_size

    # 从 StableDiffusionXLPipeline 复制的 encode_prompt 方法
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    # 从 StableDiffusionPipeline 复制的 encode_image 方法
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果输入的图像不是张量，使用特征提取器转换
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像移动到指定设备并设置数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果要求输出隐藏状态
        if output_hidden_states:
            # 编码图像并获取倒数第二个隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 重复隐藏状态以适应每个提示的图像数量
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 编码空图像并获取其隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 重复空图像的隐藏状态
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
            # 创建与图像嵌入形状相同的零张量作为无条件嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回图像嵌入和无条件嵌入
            return image_embeds, uncond_image_embeds
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制而来
    def prepare_ip_adapter_image_embeds(
        # 定义方法，接受适配器图像、图像嵌入、设备、每个提示的图像数量和是否进行无分类器自由引导的标志
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化图像嵌入列表
        image_embeds = []
        # 如果启用无分类器自由引导
        if do_classifier_free_guidance:
            # 初始化负图像嵌入列表
            negative_image_embeds = []
        # 如果图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 如果输入的图像不是列表，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入图像数量与适配器数量是否匹配
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    # 抛出错误，提示输入图像数量与适配器数量不匹配
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历输入图像和图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 检查图像投影层是否为 ImageProjection 类型
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像，获取嵌入和负嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用无分类器自由引导，将负嵌入添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果图像嵌入已提供
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用无分类器自由引导，分割负嵌入和图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负嵌入添加到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化适配器图像嵌入列表
        ip_adapter_image_embeds = []
        # 遍历图像嵌入列表
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入复制指定次数
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用无分类器自由引导，复制负嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负嵌入和图像嵌入拼接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 添加到适配器图像嵌入列表
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回适配器图像嵌入列表
        return ip_adapter_image_embeds
    # 准备额外的参数用于调度器步骤，因为并非所有调度器都有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅用于 DDIMScheduler，在其他调度器中将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 之间

        # 检查调度器的 step 方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数的字典
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的 step 方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.check_inputs 复制的
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
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 根据批量大小、通道数、高度和宽度构建形状元组
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 如果 generator 是列表且长度与批量大小不匹配，则引发错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供 latents，则生成随机噪声
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了 latents，则将其转移到指定设备
            latents = latents.to(device)

        # 将初始噪声按调度器要求的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜变量
        return latents

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids 复制的
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    # 定义参数
        ):
            # 将原始尺寸、裁剪坐标和目标尺寸合并为一个列表
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
            # 计算通过传递的时间嵌入维度和文本编码器投影维度得出的总嵌入维度
            passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
            )
            # 获取模型期望的添加时间嵌入维度
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    
            # 检查实际嵌入维度是否与期望值相等
            if expected_add_embed_dim != passed_add_embed_dim:
                # 如果不相等，抛出值错误并提供调试信息
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )
    
            # 将添加的时间 ID 转换为张量，并指定数据类型
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
            # 返回添加的时间 ID
            return add_time_ids
    
        # 从 StableDiffusionUpscalePipeline 类复制的方法
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
            # 如果使用了 XFormers 或 Torch 2.0，则注意力块不需要为浮点32位，从而节省内存
            if use_torch_2_0_or_xformers:
                # 将 VAE 的后量化卷积层转换为相应的数据类型
                self.vae.post_quant_conv.to(dtype)
                # 将 VAE 的输入卷积层转换为相应的数据类型
                self.vae.decoder.conv_in.to(dtype)
                # 将 VAE 的中间块转换为相应的数据类型
                self.vae.decoder.mid_block.to(dtype)
    
        # 从 StableDiffusionAdapterPipeline 类复制的方法
        def _default_height_width(self, height, width, image):
            # 注意：图像列表中的每个图像可能具有不同的维度
            # 所以只检查第一个图像并不完全正确，但比较简单
            while isinstance(image, list):
                # 获取图像列表的第一个图像
                image = image[0]
    
            # 如果高度为 None，则根据图像获取高度
            if height is None:
                if isinstance(image, PIL.Image.Image):
                    # 获取 PIL 图像的高度
                    height = image.height
                elif isinstance(image, torch.Tensor):
                    # 获取张量的高度
                    height = image.shape[-2]
    
                # 向下取整到 `self.adapter.downscale_factor` 的最近倍数
                height = (height // self.adapter.downscale_factor) * self.adapter.downscale_factor
    
            # 如果宽度为 None，则根据图像获取宽度
            if width is None:
                if isinstance(image, PIL.Image.Image):
                    # 获取 PIL 图像的宽度
                    width = image.width
                elif isinstance(image, torch.Tensor):
                    # 获取张量的宽度
                    width = image.shape[-1]
    
                # 向下取整到 `self.adapter.downscale_factor` 的最近倍数
                width = (width // self.adapter.downscale_factor) * self.adapter.downscale_factor
    
            # 返回最终的高度和宽度
            return height, width
    
        # 从 LatentConsistencyModelPipeline 类复制的方法
    # 定义获取指导尺度嵌入的函数，接收权重张量、嵌入维度和数据类型
        def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
            # 参见指定链接中的文档说明
            """
            See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
            Args:
                w (`torch.Tensor`):
                    生成具有指定指导尺度的嵌入向量，以丰富时间步嵌入。
                embedding_dim (`int`, *optional*, defaults to 512):
                    要生成的嵌入的维度。
                dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                    生成的嵌入的数据类型。
    
            Returns:
                `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
            """
            # 确保输入权重张量是一维的
            assert len(w.shape) == 1
            # 将权重乘以1000.0以增强数值范围
            w = w * 1000.0
    
            # 计算嵌入的一半维度
            half_dim = embedding_dim // 2
            # 计算缩放因子
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 生成指数衰减的嵌入
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 计算最终嵌入，通过权重张量调整嵌入
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 连接正弦和余弦嵌入
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度为奇数，则进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保生成的嵌入形状正确
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回生成的嵌入
            return emb
    
        # 定义属性以获取指导尺度
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 这里定义的 `guidance_scale` 类似于论文中方程 (2) 的指导权重 `w` 
        # `guidance_scale = 1` 表示不进行无分类器指导。
        @property
        def do_classifier_free_guidance(self):
            # 返回是否需要进行无分类器指导的布尔值
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 装饰器表示不计算梯度
        @torch.no_grad()
        # 替换示例文档字符串的装饰器
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的方法，允许多个输入参数
        def __call__(
            # 输入的提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 第二个输入提示，选填，可以是字符串或字符串列表
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 输入的图像，可以是特定格式
            image: PipelineImageInput = None,
            # 输出图像的高度，选填
            height: Optional[int] = None,
            # 输出图像的宽度，选填
            width: Optional[int] = None,
            # 推理步骤的数量，默认为50
            num_inference_steps: int = 50,
            # 指定时间步，选填
            timesteps: List[int] = None,
            # 指定 sigma 值，选填
            sigmas: List[float] = None,
            # 去噪结束的值，选填
            denoising_end: Optional[float] = None,
            # 指导比例，默认为5.0
            guidance_scale: float = 5.0,
            # 负面提示，选填，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 第二个负面提示，选填
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # eta 值，默认为0.0
            eta: float = 0.0,
            # 生成器，选填，可以是单个或多个 torch.Generator 对象
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，选填，可以是 torch.Tensor 对象
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，选填，可以是 torch.Tensor 对象
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入，选填，可以是 torch.Tensor 对象
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化提示嵌入，选填，可以是 torch.Tensor 对象
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面池化提示嵌入，选填，可以是 torch.Tensor 对象
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像，选填
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像嵌入，选填，可以是 torch.Tensor 列表
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式，默认为True
            return_dict: bool = True,
            # 回调函数，选填，用于特定操作
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调执行的步骤，默认为1
            callback_steps: int = 1,
            # 交叉注意力参数，选填
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 指导重缩放值，默认为0.0
            guidance_rescale: float = 0.0,
            # 原始图像尺寸，选填
            original_size: Optional[Tuple[int, int]] = None,
            # 裁剪左上角坐标，默认为(0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标图像尺寸，选填
            target_size: Optional[Tuple[int, int]] = None,
            # 负面原始图像尺寸，选填
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负面裁剪左上角坐标，默认为(0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负面目标图像尺寸，选填
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 适配器条件缩放，默认为1.0
            adapter_conditioning_scale: Union[float, List[float]] = 1.0,
            # 适配器条件因子，默认为1.0
            adapter_conditioning_factor: float = 1.0,
            # 跳过的剪辑步数，选填
            clip_skip: Optional[int] = None,
```