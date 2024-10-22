# `.\diffusers\pipelines\deprecated\alt_diffusion\pipeline_alt_diffusion_img2img.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有约定，依据许可证分发的软件
# 在“按原样”基础上提供，没有任何形式的保证或条件，
# 无论是明示或暗示的。有关许可证的特定权限和
# 限制，请参见许可证。
import inspect  # 导入inspect模块，用于获取对象的实时信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型注解，用于类型提示

import numpy as np  # 导入numpy库，通常用于数值计算
import PIL.Image  # 导入PIL库的Image模块，用于图像处理
import torch  # 导入PyTorch库，用于深度学习
from packaging import version  # 导入version模块，用于处理版本字符串
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, XLMRobertaTokenizer  # 导入transformers库中的图像处理和模型类

from ....configuration_utils import FrozenDict  # 从配置工具导入FrozenDict，用于不可变字典
from ....image_processor import PipelineImageInput, VaeImageProcessor  # 从图像处理模块导入相关类
from ....loaders import (  # 从加载器模块导入多个混合类
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from ....models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 从模型模块导入多个模型类
from ....models.lora import adjust_lora_scale_text_encoder  # 从Lora模块导入调整Lora规模的函数
from ....schedulers import KarrasDiffusionSchedulers  # 从调度器模块导入Karras扩散调度器
from ....utils import (  # 从工具模块导入多个工具函数和常量
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ....utils.torch_utils import randn_tensor  # 从torch工具模块导入生成随机张量的函数
from ...pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从管道工具模块导入扩散管道和稳定扩散混合类
from ...stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 从稳定扩散模块导入安全检查器
from .modeling_roberta_series import RobertaSeriesModelWithTransformation  # 从Roberta系列建模模块导入模型
from .pipeline_output import AltDiffusionPipelineOutput  # 从管道输出模块导入输出类


logger = logging.get_logger(__name__)  # 创建当前模块的日志记录器，禁用pylint对名称的警告

EXAMPLE_DOC_STRING = """  # 定义一个示例文档字符串，展示用法示例
    Examples:
        ```py
        >>> import requests  # 导入requests库，用于发送HTTP请求
        >>> import torch  # 导入PyTorch库，用于深度学习
        >>> from PIL import Image  # 从PIL库导入Image模块，用于图像处理
        >>> from io import BytesIO  # 从io模块导入BytesIO，用于字节流操作

        >>> from diffusers import AltDiffusionImg2ImgPipeline  # 从diffusers库导入图像到图像的扩散管道类

        >>> device = "cuda"  # 设置设备为CUDA（GPU）
        >>> model_id_or_path = "BAAI/AltDiffusion-m9"  # 指定模型的ID或路径
        >>> pipe = AltDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)  # 从预训练模型加载管道，并设置数据类型为float16
        >>> pipe = pipe.to(device)  # 将管道移动到指定设备

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"  # 指定输入图像的URL

        >>> response = requests.get(url)  # 发送HTTP GET请求以获取输入图像
        >>> init_image = Image.open(BytesIO(response.content)).convert("RGB")  # 将响应内容转为图像并转换为RGB模式
        >>> init_image = init_image.resize((768, 512))  # 调整图像大小为768x512

        >>> # "A fantasy landscape, trending on artstation"  # 提示字符串，用于生成图像
        >>> prompt = "幻想风景, artstation"  # 设置生成图像的描述性提示

        >>> images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images  # 调用管道生成图像
        >>> images[0].save("幻想风景.png")  # 保存生成的图像
        ```py  # 结束示例文档字符串
"""
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 复制
def retrieve_latents(
    # 定义函数接收编码器输出（张量），可选的生成器，和采样模式字符串
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    # 检查编码器输出是否具有 'latent_dist' 属性，并且采样模式为 'sample'
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从 'latent_dist' 中进行采样并返回结果
        return encoder_output.latent_dist.sample(generator)
    # 检查编码器输出是否具有 'latent_dist' 属性，并且采样模式为 'argmax'
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回 'latent_dist' 的众数
        return encoder_output.latent_dist.mode()
    # 检查编码器输出是否具有 'latents' 属性
    elif hasattr(encoder_output, "latents"):
        # 直接返回 'latents'
        return encoder_output.latents
    else:
        # 如果没有有效的属性，抛出属性错误
        raise AttributeError("Could not access latents of provided encoder_output")


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 复制
def preprocess(image):
    # 定义弃用消息，指明该方法将在 diffusers 1.0.0 中被移除
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 发出弃用警告
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 检查图像是否为张量类型
    if isinstance(image, torch.Tensor):
        # 如果是张量，直接返回
        return image
    # 检查图像是否为 PIL 图像
    elif isinstance(image, PIL.Image.Image):
        # 如果是单个 PIL 图像，转为列表
        image = [image]

    # 检查列表中的第一个元素是否为 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽和高
        w, h = image[0].size
        # 将宽和高调整为8的整数倍
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        # 调整每个图像的大小并转换为 NumPy 数组
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 将所有图像沿第0维连接
        image = np.concatenate(image, axis=0)
        # 归一化图像数组
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组维度顺序
        image = image.transpose(0, 3, 1, 2)
        # 将像素值缩放到 [-1, 1]
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为张量
        image = torch.from_numpy(image)
    # 检查列表中的第一个元素是否为张量
    elif isinstance(image[0], torch.Tensor):
        # 沿着第0维连接所有张量
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image


# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 复制
def retrieve_timesteps(
    # 定义函数接收调度器、可选的推理步骤数量、设备、时间步和标准差列表以及额外参数
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器检索时间步。处理自定义时间步。任何 kwargs 将被传递给 `scheduler.set_timesteps`。
    # 参数说明部分
    Args:
        scheduler (`SchedulerMixin`):  # 接受一个调度器混合类实例，用于获取时间步
            The scheduler to get timesteps from.
        num_inference_steps (`int`):  # 用于生成样本的扩散步数
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):  # 指定时间步所移动到的设备，若为 None，则不移动
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):  # 自定义时间步，覆盖调度器的时间步间隔策略
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):  # 自定义 sigma，覆盖调度器的时间步间隔策略
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    # 返回值说明
    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    # 检查是否同时传入时间步和 sigma，抛出异常
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传入时间步
    if timesteps is not None:
        # 检查调度器是否支持自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(  # 抛出错误提示当前调度器不支持自定义时间步
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推断步骤数量
        num_inference_steps = len(timesteps)
    # 如果传入 sigma
    elif sigmas is not None:
        # 检查调度器是否支持自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(  # 抛出错误提示当前调度器不支持自定义 sigma
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推断步骤数量
        num_inference_steps = len(timesteps)
    # 如果没有传入时间步和 sigma
    else:
        # 根据推断步骤数量设置调度器的时间步
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取设置后的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推断步骤数量
    return timesteps, num_inference_steps
# 定义一个名为 AltDiffusionImg2ImgPipeline 的类，继承自多个父类
class AltDiffusionImg2ImgPipeline(
    # 继承自 DiffusionPipeline
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin
    TextualInversionLoaderMixin,
    # 继承自 IPAdapterMixin
    IPAdapterMixin,
    # 继承自 StableDiffusionLoraLoaderMixin
    StableDiffusionLoraLoaderMixin,
    # 继承自 FromSingleFileMixin
    FromSingleFileMixin,
):
    # 文档字符串，描述该类的作用和参数
    r"""
    Pipeline for text-guided image-to-image generation using Alt Diffusion.

    # 描述该模型的功能：基于文本指导的图像到图像生成
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    # 说明该管道继承了一些加载方法
    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    # 定义构造函数的参数
    Args:
        vae ([`AutoencoderKL`]):
            # Variational Auto-Encoder (VAE) 模型，用于图像的编码和解码
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.RobertaSeriesModelWithTransformation`]):
            # 冻结的文本编码器，用于处理输入文本
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.XLMRobertaTokenizer`]):
            # 用于对文本进行分词的工具
            A `XLMRobertaTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            # 用于去噪图像潜在表示的 UNet 模型
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            # 与 UNet 结合使用的调度器，用于去噪
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            # 评估生成图像是否可能被视为冒犯或有害的分类模块
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            # 用于从生成图像中提取特征的处理器
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 可选组件列表
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 从 CPU 卸载中排除的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    # 用于回调的张量输入列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化方法，用于创建类的实例
        def __init__(
            # VAE（变分自编码器）实例，用于数据编码
            self,
            vae: AutoencoderKL,
            # 文本编码器，用于处理文本输入
            text_encoder: RobertaSeriesModelWithTransformation,
            # 分词器，用于将文本转换为标记
            tokenizer: XLMRobertaTokenizer,
            # UNet 模型，用于图像生成
            unet: UNet2DConditionModel,
            # 调度器，用于控制生成过程中的时间步
            scheduler: KarrasDiffusionSchedulers,
            # 安全检查器，用于过滤不安全的内容
            safety_checker: StableDiffusionSafetyChecker,
            # 特征提取器，用于图像处理
            feature_extractor: CLIPImageProcessor,
            # 可选的图像编码器，用于处理图像输入
            image_encoder: CLIPVisionModelWithProjection = None,
            # 指示是否需要安全检查器的布尔值，默认为 True
            requires_safety_checker: bool = True,
        # 编码提示的方法
        def _encode_prompt(
            self,
            # 提示文本，输入的描述信息
            prompt,
            # 设备类型，指定计算设备（CPU或GPU）
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否执行分类器自由引导的布尔值
            do_classifier_free_guidance,
            # 可选的负面提示文本
            negative_prompt=None,
            # 可选的提示嵌入，提前计算的提示表示
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入，提前计算的负面提示表示
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 LoRA 比例，用于模型调优
            lora_scale: Optional[float] = None,
            # 其他可选参数
            **kwargs,
        ):
            # 警告信息，说明该方法已弃用并将在未来版本中删除，建议使用新的方法
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用弃用函数的警告方法，记录弃用信息
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用新的编码方法，获取提示嵌入元组
            prompt_embeds_tuple = self.encode_prompt(
                # 提示文本
                prompt=prompt,
                # 计算设备
                device=device,
                # 每个提示生成的图像数量
                num_images_per_prompt=num_images_per_prompt,
                # 是否执行分类器自由引导
                do_classifier_free_guidance=do_classifier_free_guidance,
                # 可选的负面提示文本
                negative_prompt=negative_prompt,
                # 可选的提示嵌入
                prompt_embeds=prompt_embeds,
                # 可选的负面提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,
                # 可选的 LoRA 比例
                lora_scale=lora_scale,
                # 传递其他参数
                **kwargs,
            )
    
            # 将嵌入元组的两个部分连接在一起，便于后续处理
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回连接后的提示嵌入
            return prompt_embeds
    
        # 编码提示的方法，用于处理输入的提示文本
        def encode_prompt(
            self,
            # 提示文本
            prompt,
            # 计算设备
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否执行分类器自由引导
            do_classifier_free_guidance,
            # 可选的负面提示文本
            negative_prompt=None,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 LoRA 比例
            lora_scale: Optional[float] = None,
            # 可选的跳过剪辑层的参数
            clip_skip: Optional[int] = None,
    # 定义编码图像的方法，接受图像、设备、每个提示的图像数量和隐藏状态输出参数
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入是否为张量，若不是则进行特征提取并转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像移动到指定设备并转换为相应数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 通过图像编码器编码图像，并获取倒数第二个隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 根据每个提示的图像数量重复隐藏状态
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对于无条件图像，创建零张量并编码以获取隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 同样根据每个提示的图像数量重复无条件隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码后的图像隐藏状态和无条件隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 根据每个提示的图像数量重复图像嵌入
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 定义运行安全检查器的方法，接受图像、设备和数据类型
        def run_safety_checker(self, image, device, dtype):
            # 如果安全检查器不存在，设置无NSFW概念标志为None
            if self.safety_checker is None:
                has_nsfw_concept = None
            else:
                # 检查图像是否为张量，若是则进行后处理
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 将NumPy数组转换为PIL图像
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 提取特征并移动到设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 运行安全检查器并返回处理后的图像和NSFW概念标志
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像和NSFW概念标志
            return image, has_nsfw_concept
    
        # 定义解码潜在变量的方法
        def decode_latents(self, latents):
            # 定义弃用提示信息
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            # 发出弃用警告
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 对潜在变量进行缩放
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在变量以生成图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 将图像进行归一化处理并限制范围
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为float32格式，方便与bfloat16兼容且不造成显著开销
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回处理后的图像
            return image
    # 准备调度器步骤所需的额外参数，因不同调度器的参数签名可能不同
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅用于 DDIMScheduler，其他调度器将忽略它
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 的值应在 [0, 1] 之间
    
        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数字典
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
    # 检查输入参数的有效性
    def check_inputs(
        self,
        prompt,
        strength,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # 检查 strength 参数是否在有效范围内 (0 到 1)
        if strength < 0 or strength > 1:
            # 如果不在范围内，抛出值错误
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # 检查 callback_steps 是否为正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            # 如果不是正整数，抛出值错误
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查 callback_on_step_end_tensor_inputs 中的每个键是否在 self._callback_tensor_inputs 中
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 如果有键不在列表中，抛出值错误
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        # 检查是否同时提供了 prompt 和 prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            # 如果同时提供，抛出值错误
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都为 None
        elif prompt is None and prompt_embeds is None:
            # 如果都为 None，抛出值错误
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 的类型是否为 str 或 list
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 如果类型不正确，抛出值错误
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了 negative_prompt 和 negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            # 如果同时提供，抛出值错误
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查 prompt_embeds 和 negative_prompt_embeds 是否都不为 None
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            # 检查它们的形状是否一致
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                # 如果形状不一致，抛出值错误
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_timesteps(self, num_inference_steps, strength, device):
        # 根据初始化时间步计算原始时间步
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算时间步的起始位置
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中获取相应的时间步
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # 返回计算的时间步和剩余推理步骤
        return timesteps, num_inference_steps - t_start
    # 准备潜在向量（latents）以便于后续处理
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        # 检查输入的 image 是否为指定类型之一：torch.Tensor、PIL.Image.Image 或列表
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                # 抛出异常，告知 image 的类型不符合要求
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # 将 image 转移到指定的设备并转换为指定的数据类型
        image = image.to(device=device, dtype=dtype)

        # 计算有效的 batch_size，即每个提示的图像数量
        batch_size = batch_size * num_images_per_prompt

        # 如果 image 的通道数为 4，直接将其赋值为初始化潜在向量
        if image.shape[1] == 4:
            init_latents = image

        else:
            # 如果 generator 是列表且其长度与 batch_size 不匹配，抛出异常
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    # 抛出异常，告知 generator 的长度与请求的 batch_size 不匹配
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # 如果 generator 是列表，遍历每个生成器以获取潜在向量
            elif isinstance(generator, list):
                init_latents = [
                    # 对每个图像进行编码并检索对应的潜在向量
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                # 将所有的潜在向量合并成一个张量
                init_latents = torch.cat(init_latents, dim=0)
            else:
                # 如果 generator 不是列表，直接对整个图像进行编码并检索潜在向量
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            # 将潜在向量乘以配置中的缩放因子
            init_latents = self.vae.config.scaling_factor * init_latents

        # 检查请求的 batch_size 是否大于初始化潜在向量的数量并且能够整除
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # 扩展 init_latents 以满足 batch_size
            deprecation_message = (
                # 构造弃用警告消息
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            # 记录弃用警告
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            # 计算每个提示需要的额外图像数量
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            # 复制初始化潜在向量以满足 batch_size
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        # 如果 batch_size 大于初始化潜在向量的数量但不能整除，抛出异常
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                # 抛出异常，告知不能将潜在向量重复以满足 batch_size
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            # 如果 batch_size 合法，则将 init_latents 扩展为单一张量
            init_latents = torch.cat([init_latents], dim=0)

        # 获取初始化潜在向量的形状
        shape = init_latents.shape
        # 生成与潜在向量形状相同的随机噪声
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 将噪声添加到潜在向量中
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        # 将处理后的潜在向量赋值给 latents
        latents = init_latents

        # 返回最终的潜在向量
        return latents
    # 定义获取指导尺度嵌入的函数，接收输入参数 w、嵌入维度和数据类型
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        查看关于嵌入生成的更多信息，链接到相关 GitHub 页面

        参数:
            timesteps (`torch.Tensor`):
                在这些时间步生成嵌入向量
            embedding_dim (`int`, *可选*, 默认值为 512):
                生成的嵌入的维度
            dtype:
                生成嵌入的数据类型

        返回:
            `torch.Tensor`: 形状为 `(len(timesteps), embedding_dim)` 的嵌入向量
        """
        # 确保输入的 w 是一维的
        assert len(w.shape) == 1
        # 将 w 放大 1000 倍
        w = w * 1000.0

        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算 log(10000) 除以 (half_dim - 1) 的值，用于后续计算
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成从 0 到 half_dim 的指数衰减的嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将 w 转换为指定数据类型，并生成与 emb 组合的嵌入
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 将正弦和余弦嵌入连接在一起
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度是奇数，则在最后填充一个零
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保最终嵌入的形状与预期一致
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb

    # 定义属性，返回内部的指导尺度
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 定义属性，返回内部的剪辑跳过值
    @property
    def clip_skip(self):
        return self._clip_skip

    # 这里 `guidance_scale` 类似于公式 (2) 中的指导权重 `w`
    # 在 Imagen 论文中: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # 对应于不进行分类器自由指导。
    @property
    def do_classifier_free_guidance(self):
        # 检查指导尺度是否大于 1 且时间条件投影维度是否为空
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 定义属性，返回交叉注意力的关键字参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 定义属性，返回时间步的数量
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 在不计算梯度的情况下，装饰后续函数
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的方法，允许通过不同参数进行调用
        def __call__(
            # 输入的提示文本，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输入的图像，类型为 PipelineImageInput
            image: PipelineImageInput = None,
            # 控制生成强度的参数，默认值为 0.8
            strength: float = 0.8,
            # 推理步骤的数量，默认为 50
            num_inference_steps: Optional[int] = 50,
            # 可选的时间步列表
            timesteps: List[int] = None,
            # 可选的 sigma 值列表
            sigmas: List[float] = None,
            # 引导缩放因子，默认为 7.5
            guidance_scale: Optional[float] = 7.5,
            # 可选的负提示文本，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 可选的 eta 值，默认为 0.0
            eta: Optional[float] = 0.0,
            # 随机数生成器，可以是单个或列表形式
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的提示嵌入，类型为 torch.Tensor
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负提示嵌入，类型为 torch.Tensor
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的图像输入，用于 IP 适配器
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 可选的交叉注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的跳过剪辑参数
            clip_skip: int = None,
            # 在步骤结束时的可选回调函数
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 在步骤结束时的张量输入列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 其他可变参数
            **kwargs,
```