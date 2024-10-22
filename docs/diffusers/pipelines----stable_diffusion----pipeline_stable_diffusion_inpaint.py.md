# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion_inpaint.py`

```py
# 版权信息，声明该代码的所有权和许可信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律要求或书面同意，否则根据许可证分发的软件是按“原样”基础分发的，
# 不附有任何形式的明示或暗示的担保或条件。
# 有关许可证的具体条款和权限限制，请参阅许可证。

# 导入 inspect 模块，用于获取对象的签名和信息
import inspect
# 导入类型相关的类，用于类型注解
from typing import Any, Callable, Dict, List, Optional, Union

# 导入图像处理库 PIL
import PIL.Image
# 导入 PyTorch 库
import torch
# 导入版本管理工具
from packaging import version
# 导入 Hugging Face Transformers 中的相关类
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 导入其他模块和类，涉及回调、配置、图像处理等
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...configuration_utils import FrozenDict
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AsymmetricAutoencoderKL, AutoencoderKL, ImageProjection, UNet2DConditionModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker

# 初始化日志记录器，以当前模块的名称为标识
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义函数，从编码器输出中检索潜在变量
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,  # 输入的编码器输出，类型为张量
    generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
    sample_mode: str = "sample"  # 采样模式，默认为 "sample"
):
    # 如果编码器输出有潜在分布且采样模式为 "sample"
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从潜在分布中采样并返回结果
        return encoder_output.latent_dist.sample(generator)
    # 如果编码器输出有潜在分布且采样模式为 "argmax"
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回潜在分布的众数
        return encoder_output.latent_dist.mode()
    # 如果编码器输出有潜在变量
    elif hasattr(encoder_output, "latents"):
        # 直接返回潜在变量
        return encoder_output.latents
    # 如果以上条件都不满足，抛出属性错误
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

# 定义函数，从调度器中检索时间步
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,  # 调度器对象
    num_inference_steps: Optional[int] = None,  # 可选的推理步骤数量
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备信息
    timesteps: Optional[List[int]] = None,  # 可选的时间步列表
    sigmas: Optional[List[float]] = None,  # 可选的 sigma 值列表
    **kwargs,  # 其他可选参数
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中检索时间步。
    处理自定义时间步。任何其他参数都将传递给 `scheduler.set_timesteps`。
    # 定义函数参数的文档字符串
    Args:
        scheduler (`SchedulerMixin`):
            # 调度器，用于获取时间步
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            # 生成样本时使用的扩散步骤数量，如果使用此参数，则 `timesteps` 必须为 `None`
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            # 要将时间步移动到的设备，如果为 `None`，则时间步不移动
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            # 自定义时间步，用于覆盖调度器的时间步间隔策略，如果提供了 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            # 自定义 sigma 值，用于覆盖调度器的时间步间隔策略，如果提供了 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        # 返回一个元组，第一个元素是来自调度器的时间步调度，第二个元素是推理步骤的数量
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    # 检查是否同时传入了 `timesteps` 和 `sigmas`
    if timesteps is not None and sigmas is not None:
        # 如果同时存在，则引发错误
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    
    # 检查是否传入了 `timesteps`
    if timesteps is not None:
        # 检查调度器是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，则引发错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    
    # 检查是否传入了 `sigmas`
    elif sigmas is not None:
        # 检查调度器是否接受自定义 sigma 值
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持，则引发错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置调度器的 sigma 值
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    
    # 如果都没有传入，则使用默认推理步骤设置时间步
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取调度器的时间步
        timesteps = scheduler.timesteps
    
    # 返回时间步和推理步骤数量
    return timesteps, num_inference_steps
# 定义一个用于文本引导图像修复的管道类，继承自多个基类
class StableDiffusionInpaintPipeline(
    # 继承自 DiffusionPipeline，提供通用的管道功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin，增加稳定扩散特性
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin，支持文本反转加载
    TextualInversionLoaderMixin,
    # 继承自 IPAdapterMixin，支持 IP 适配器加载
    IPAdapterMixin,
    # 继承自 StableDiffusionLoraLoaderMixin，支持 LoRA 权重加载
    StableDiffusionLoraLoaderMixin,
    # 继承自 FromSingleFileMixin，支持从单个文件加载
    FromSingleFileMixin,
):
    # 文档字符串，描述管道的功能和参数
    r"""
    Pipeline for text-guided image inpainting using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`, `AsymmetricAutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    # 定义模型的 CPU 卸载顺序，指定组件的处理顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 可选组件列表，包含安全检查器和特征提取器
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 排除 CPU 卸载的组件，安全检查器不会被卸载
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义需要回调的张量输入列表，包含潜在变量和提示嵌入
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "mask", "masked_image_latents"]
    # 初始化方法，用于创建类的实例
        def __init__(
            self,
            vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],  # VAE模型，支持两种类型
            text_encoder: CLIPTextModel,  # 文本编码器，处理文本输入
            tokenizer: CLIPTokenizer,  # 分词器，将文本转换为标记
            unet: UNet2DConditionModel,  # UNet模型，处理生成任务
            scheduler: KarrasDiffusionSchedulers,  # 调度器，用于调整生成过程
            safety_checker: StableDiffusionSafetyChecker,  # 安全检查器，确保生成内容的安全性
            feature_extractor: CLIPImageProcessor,  # 特征提取器，处理图像输入
            image_encoder: CLIPVisionModelWithProjection = None,  # 可选图像编码器，用于图像的额外处理
            requires_safety_checker: bool = True,  # 是否需要安全检查器的标志
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制
        def _encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 设备信息（CPU或GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否执行无分类器引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的LORA缩放因子
            **kwargs,  # 其他可选参数
        ):
            # 弃用消息，提示用户该方法将来会被移除
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 发出弃用警告
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 方法并获取嵌入元组
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,  # 提示文本
                device=device,  # 设备信息
                num_images_per_prompt=num_images_per_prompt,  # 每个提示生成的图像数量
                do_classifier_free_guidance=do_classifier_free_guidance,  # 无分类器引导标志
                negative_prompt=negative_prompt,  # 负面提示文本
                prompt_embeds=prompt_embeds,  # 提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,  # 负面提示嵌入
                lora_scale=lora_scale,  # LORA缩放因子
                **kwargs,  # 其他参数
            )
    
            # 将嵌入元组的内容拼接以兼容旧版
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回拼接后的提示嵌入
            return prompt_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制
        def encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 设备信息（CPU或GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否执行无分类器引导
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的LORA缩放因子
            clip_skip: Optional[int] = None,  # 可选的剪切参数
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制
    # 定义一个方法用于编码图像，接收图像、设备、每个提示的图像数量和可选的隐藏状态输出
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量类型，则使用特征提取器处理图像并返回张量格式
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像转移到指定设备，并转换为正确的数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 将隐藏状态在第0维上重复，数量为每个提示的图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对一个全零的图像进行编码以获取无条件图像的隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 同样在第0维上重复无条件图像的隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回有条件和无条件的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 如果不需要输出隐藏状态，直接编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 在第0维上重复图像嵌入，数量为每个提示的图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入相同形状的全零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回有条件和无条件的图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 中复制的方法，用于准备图像嵌入
        def prepare_ip_adapter_image_embeds(
            # 接收输入适配器图像、适配器图像嵌入、设备、每个提示的图像数量和分类器自由引导的开关
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    # 定义函数体的结束
    ):
        # 初始化一个空列表以存储图像嵌入
        image_embeds = []
        # 如果启用无分类器自由引导，则初始化负图像嵌入列表
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 检查输入适配器图像是否为列表，如果不是则转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查输入适配器图像与 IP 适配器数量是否匹配
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果不匹配，则抛出值错误
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历输入适配器图像与图像投影层的组合
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断输出隐藏状态是否为 True
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像以获取图像嵌入和负图像嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用无分类器自由引导，则添加负图像嵌入
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 遍历现有的输入适配器图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用无分类器自由引导，则分割嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负图像嵌入添加到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表以存储 IP 适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历图像嵌入和它们的索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 将单个图像嵌入按数量扩展
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用无分类器自由引导，则扩展负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入与正图像嵌入合并
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定设备
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回 IP 适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的代码
    # 运行安全检查器以确保图像符合安全标准
        def run_safety_checker(self, image, device, dtype):
            # 检查安全检查器是否存在
            if self.safety_checker is None:
                # 如果没有安全检查器，设置NSFW概念为None
                has_nsfw_concept = None
            else:
                # 如果输入是张量，则处理为PIL格式
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果输入是numpy数组，将其转换为PIL格式
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 提取特征并将其传输到指定设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 运行安全检查器，并返回处理后的图像和NSFW概念
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像和NSFW概念
            return image, has_nsfw_concept
    
        # 从StableDiffusionPipeline复制的函数，准备额外的调度器步骤参数
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外的关键字参数，因为并非所有调度器具有相同的签名
            # eta仅在DDIM调度器中使用，其他调度器会忽略
            # eta对应于DDIM论文中的η，应在[0, 1]之间
    
            # 检查调度器是否接受eta参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 创建额外步骤参数的字典
            extra_step_kwargs = {}
            if accepts_eta:
                # 如果接受eta，则将其添加到字典中
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受generator参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            if accepts_generator:
                # 如果接受generator，则将其添加到字典中
                extra_step_kwargs["generator"] = generator
            # 返回额外步骤参数字典
            return extra_step_kwargs
    
        # 检查输入参数的有效性和完整性
        def check_inputs(
            self,
            prompt,
            image,
            mask_image,
            height,
            width,
            strength,
            callback_steps,
            output_type,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            callback_on_step_end_tensor_inputs=None,
            padding_mask_crop=None,
        # 准备潜在变量的函数
        def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
            image=None,
            timestep=None,
            is_strength_max=True,
            return_noise=False,
            return_image_latents=False,
    ):
        # 定义形状，包含批次大小、通道数、调整后的高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器列表的长度是否与批次大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 验证图像和时间步是否提供，且强度不为最大值
        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        # 根据条件处理图像潜在变量
        if return_image_latents or (latents is None and not is_strength_max):
            # 将图像移动到指定设备和数据类型
            image = image.to(device=device, dtype=dtype)

            # 如果图像有4个通道，则直接使用图像潜在变量
            if image.shape[1] == 4:
                image_latents = image
            else:
                # 使用 VAE 编码图像以获取潜在变量
                image_latents = self._encode_vae_image(image=image, generator=generator)
            # 根据批次大小重复潜在变量
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        # 如果潜在变量为空，则生成噪声
        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # 根据强度初始化潜在变量
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # 如果强度为最大值，则按调度器的初始 sigma 缩放潜在变量
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            # 将现有潜在变量转换到设备上
            noise = latents.to(device)
            # 按调度器的初始 sigma 缩放潜在变量
            latents = noise * self.scheduler.init_noise_sigma

        # 输出结果，包括潜在变量
        outputs = (latents,)

        # 如果需要返回噪声，则添加噪声到输出
        if return_noise:
            outputs += (noise,)

        # 如果需要返回图像潜在变量，则添加到输出
        if return_image_latents:
            outputs += (image_latents,)

        # 返回最终输出
        return outputs

    # 编码 VAE 图像以获取潜在变量
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        # 检查生成器是否为列表
        if isinstance(generator, list):
            # 为每个图像批次编码潜在变量并检索
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            # 将所有潜在变量合并为一个张量
            image_latents = torch.cat(image_latents, dim=0)
        else:
            # 使用单个生成器编码图像并检索潜在变量
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        # 按配置的缩放因子缩放潜在变量
        image_latents = self.vae.config.scaling_factor * image_latents

        # 返回缩放后的潜在变量
        return image_latents

    # 准备掩膜潜在变量的方法
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # 将掩码调整为与潜在特征图的形状相同，以便将掩码与潜在特征图拼接
        # 在转换数据类型之前进行此操作，以避免在使用 cpu_offload 和半精度时出现问题
        mask = torch.nn.functional.interpolate(
            # 使用插值方法调整掩码的大小，目标尺寸为根据 VAE 缩放因子调整后的高度和宽度
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        # 将掩码移动到指定设备，并转换为指定数据类型
        mask = mask.to(device=device, dtype=dtype)

        # 将掩码图像移动到指定设备，并转换为指定数据类型
        masked_image = masked_image.to(device=device, dtype=dtype)

        # 检查掩码图像的通道数是否为4
        if masked_image.shape[1] == 4:
            # 如果是4通道，则将其直接赋值给潜在特征图
            masked_image_latents = masked_image
        else:
            # 否则，使用 VAE 编码掩码图像以获取潜在特征图
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # 针对每个提示生成，重复掩码和潜在特征图以适应批量大小，使用对 MPS 友好的方法
        if mask.shape[0] < batch_size:
            # 如果掩码的数量小于批量大小，则检查批量大小是否能被掩码数量整除
            if not batch_size % mask.shape[0] == 0:
                # 如果不能整除，抛出值错误
                raise ValueError(
                    "传入的掩码与所需的批量大小不匹配。掩码应复制到"
                    f" 总批量大小 {batch_size}，但传入了 {mask.shape[0]} 个掩码。请确保传入的掩码数量"
                    " 能被所请求的总批量大小整除。"
                )
            # 通过重复掩码，调整其数量以匹配批量大小
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        # 同样检查潜在特征图的数量
        if masked_image_latents.shape[0] < batch_size:
            # 检查批量大小是否能被潜在特征图数量整除
            if not batch_size % masked_image_latents.shape[0] == 0:
                # 如果不能整除，抛出值错误
                raise ValueError(
                    "传入的图像与所需的批量大小不匹配。图像应复制到"
                    f" 总批量大小 {batch_size}，但传入了 {masked_image_latents.shape[0]} 个图像。"
                    " 请确保传入的图像数量能被所请求的总批量大小整除。"
                )
            # 通过重复潜在特征图，调整其数量以匹配批量大小
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # 如果启用分类器自由引导，则重复掩码以进行拼接
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        # 如果启用分类器自由引导，则重复潜在特征图以进行拼接
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # 确保潜在特征图的设备与潜在模型输入一致，以避免设备错误
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        # 返回掩码和潜在特征图
        return mask, masked_image_latents

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps 复制
    # 定义获取时间步长的函数，接受推理步数、强度和设备作为参数
    def get_timesteps(self, num_inference_steps, strength, device):
        # 根据给定的推理步数和强度计算初始时间步
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算时间步开始的索引，确保不小于零
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中获取时间步长，从 t_start 开始
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # 如果调度器具有设置开始索引的方法，则调用该方法
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        # 返回时间步长和剩余的推理步数
        return timesteps, num_inference_steps - t_start

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img 中复制的函数
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        参考链接: https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        参数:
            w (`torch.Tensor`):
                生成带有指定引导尺度的嵌入向量，以后丰富时间步嵌入。
            embedding_dim (`int`, *可选*, 默认为 512):
                要生成的嵌入维度。
            dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                生成的嵌入的数据类型。

        返回:
            `torch.Tensor`: 嵌入向量，形状为 `(len(w), embedding_dim)`。
        """
        # 确保输入的张量 w 是一维的
        assert len(w.shape) == 1
        # 将 w 扩大 1000 倍
        w = w * 1000.0

        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算嵌入的基数
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 计算指数衰减的嵌入值
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 将 w 转换为目标数据类型，并与嵌入值相乘
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 将正弦和余弦值连接成最终嵌入
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度是奇数，则在最后填充一个零
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保最终的嵌入形状是正确的
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回计算得到的嵌入
        return emb

    # 获取引导尺度的属性
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 获取剪辑跳过的属性
    @property
    def clip_skip(self):
        return self._clip_skip

    # 判断是否进行无分类器引导的属性
    # 这里的 `guidance_scale` 类似于公式 (2) 中的引导权重 `w`
    # 参见 Imagen 论文: https://arxiv.org/pdf/2205.11487.pdf 。`guidance_scale = 1`
    # 表示不进行分类器自由引导。
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # 获取交叉注意力的关键字参数
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    # 获取时间步数的属性
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 获取中断状态的属性
    @property
    def interrupt(self):
        return self._interrupt

    # 在计算梯度时不追踪
    @torch.no_grad()
    # 定义一个可调用的类方法，允许传入多个参数
        def __call__(
            # 提示信息，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输入图像，用于处理的管道图像
            image: PipelineImageInput = None,
            # 用于掩蔽的图像
            mask_image: PipelineImageInput = None,
            # 掩蔽图像的潜在表示，Tensor 类型
            masked_image_latents: torch.Tensor = None,
            # 输出图像的高度，默认为 None
            height: Optional[int] = None,
            # 输出图像的宽度，默认为 None
            width: Optional[int] = None,
            # 填充掩码裁剪的大小，默认为 None
            padding_mask_crop: Optional[int] = None,
            # 强度参数，默认为 1.0
            strength: float = 1.0,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 预定义的时间步列表，默认为 None
            timesteps: List[int] = None,
            # sigma 值列表，默认为 None
            sigmas: List[float] = None,
            # 指导缩放因子，默认为 7.5
            guidance_scale: float = 7.5,
            # 负提示信息，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # eta 参数，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可以是单个或列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在表示，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示的嵌入表示，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的嵌入表示，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像的嵌入列表，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 交叉注意力的参数字典，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 跳过的剪辑步骤数，默认为 None
            clip_skip: int = None,
            # 在每个步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 回调时输入的张量名称列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 额外的关键字参数
            **kwargs,
```