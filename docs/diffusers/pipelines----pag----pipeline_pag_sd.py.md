# `.\diffusers\pipelines\pag\pipeline_pag_sd.py`

```py
# 版权声明，指明版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行授权；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是按“现状”基础分发的，
# 不附有任何形式的明示或暗示的担保或条件。
# 有关许可证下特定权限和限制的更多信息，请参见许可证。
import inspect  # 导入 inspect 模块，用于获取活跃对象的详细信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示相关的工具

import torch  # 导入 PyTorch 库
from packaging import version  # 导入版本管理工具
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入 CLIP 相关的处理器和模型

from ...configuration_utils import FrozenDict  # 导入 FrozenDict，可能用于不可变字典配置
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关的类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入不同的加载器混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入用于调整 Lora 模型的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (  # 从工具模块导入多个实用函数
    USE_PEFT_BACKEND,  # 指示是否使用 PEFT 后端
    deprecate,  # 用于标记弃用的功能
    logging,  # 导入日志记录工具
    replace_example_docstring,  # 用于替换示例文档字符串的函数
    scale_lora_layers,  # 用于缩放 Lora 层的函数
    unscale_lora_layers,  # 用于取消缩放 Lora 层的函数
)
from ...utils.torch_utils import randn_tensor  # 导入随机张量生成工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和稳定扩散混合类
from ..stable_diffusion.pipeline_output import StableDiffusionPipelineOutput  # 导入稳定扩散管道的输出类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入稳定扩散安全检查器
from .pag_utils import PAGMixin  # 导入 PAG 相关的混合类

logger = logging.get_logger(__name__)  # 创建一个日志记录器，记录当前模块的信息；禁用 pylint 的无效名称警告

EXAMPLE_DOC_STRING = """  # 定义一个示例文档字符串
    Examples:  # 示例部分
        ```py  # 示例代码块开始
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import AutoPipelineForText2Image  # 从 diffusers 导入自动文本到图像管道

        >>> pipe = AutoPipelineForText2Image.from_pretrained(  # 从预训练模型加载管道
        ...     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, enable_pag=True  # 指定模型路径、数据类型及启用 PAG
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 CUDA 设备

        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义生成图像的提示
        >>> image = pipe(prompt, pag_scale=0.3).images[0]  # 生成图像并获取第一张图像
        ```py  # 示例代码块结束
"""

# 从稳定扩散管道复制的函数，调整噪声配置
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):  # 定义噪声配置重缩放函数
    """
    根据 `guidance_rescale` 重缩放 `noise_cfg`。基于文献[Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)的发现。见第 3.4 节
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)  # 计算文本噪声预测的标准差
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)  # 计算噪声配置的标准差
    # 根据引导结果重缩放噪声配置（修复过度曝光问题）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)  # 使用标准差比重缩放噪声配置
    # 将经过缩放的噪声预测与原始结果混合，以避免图像显得"平淡"
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        # 返回混合后的噪声配置
        return noise_cfg
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的
def retrieve_timesteps(
    # 调度器，用于获取时间步
    scheduler,
    # 可选的推理步骤数量
    num_inference_steps: Optional[int] = None,
    # 可选的设备，指定时间步移动的目标设备
    device: Optional[Union[str, torch.device]] = None,
    # 可选的自定义时间步
    timesteps: Optional[List[int]] = None,
    # 可选的自定义 sigmas
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器获取时间步。处理
    自定义时间步。所有关键字参数将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            用于获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps`
            必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步应移动到的设备。如果为 `None`，则时间步不移动。
        timesteps (`List[int]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义时间步。如果传入 `timesteps`，
            则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            用于覆盖调度器的时间步间隔策略的自定义 sigmas。如果传入 `sigmas`，
            则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是来自调度器的时间步计划，
        第二个元素是推理步骤的数量。
    """
    # 如果同时传入了自定义时间步和 sigmas，抛出异常
    if timesteps is not None and sigmas is not None:
        raise ValueError("只能传入 `timesteps` 或 `sigmas` 中的一个。请选择一个以设置自定义值")
    # 如果传入了自定义时间步
    if timesteps is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持自定义时间步，抛出异常
        if not accepts_timesteps:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义"
                f" 时间步计划。请检查是否使用了正确的调度器。"
            )
        # 设置调度器的时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传入了自定义 sigmas
    elif sigmas is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受自定义 sigmas
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不支持自定义 sigmas，抛出异常
        if not accept_sigmas:
            raise ValueError(
                f"当前调度器类 {scheduler.__class__} 的 `set_timesteps` 不支持自定义"
                f" sigmas 时间步计划。请检查是否使用了正确的调度器。"
            )
        # 设置调度器的 sigmas
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果不是满足条件的情况，执行以下操作
        else:
            # 设置推理步骤的时间步，指定设备并传递额外参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器中的时间步列表
            timesteps = scheduler.timesteps
        # 返回时间步和推理步骤的数量
        return timesteps, num_inference_steps
# 定义一个类 StableDiffusionPAGPipeline，继承自多个基类以实现特定功能
class StableDiffusionPAGPipeline(
    # 继承自 DiffusionPipeline 基类，提供基本的扩散管道功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin，添加 Stable Diffusion 特有的功能
    StableDiffusionMixin,
    # 继承自 TextualInversionLoaderMixin，支持文本反演加载功能
    TextualInversionLoaderMixin,
    # 继承自 StableDiffusionLoraLoaderMixin，支持加载 LoRA 权重
    StableDiffusionLoraLoaderMixin,
    # 继承自 IPAdapterMixin，支持加载 IP 适配器
    IPAdapterMixin,
    # 继承自 FromSingleFileMixin，支持从单一文件加载模型
    FromSingleFileMixin,
    # 继承自 PAGMixin，添加特定于 PAG 的功能
    PAGMixin,
):
    # 类文档字符串，描述该管道的功能
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
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

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义可选组件的列表，包含可能的附加功能
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义不进行 CPU 卸载的组件列表，确保安全检查器始终在 CPU 上
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义回调张量输入的列表，指定需要监控的输入数据
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # 初始化类的构造函数，接收多个参数用于模型的配置
        def __init__(
            self,
            # VAE模型，负责生成和重构图像
            vae: AutoencoderKL,
            # 文本编码器，使用CLIP进行文本嵌入
            text_encoder: CLIPTextModel,
            # 分词器，将文本转换为模型可处理的格式
            tokenizer: CLIPTokenizer,
            # UNet模型，用于图像生成
            unet: UNet2DConditionModel,
            # 调度器，控制生成过程中的采样
            scheduler: KarrasDiffusionSchedulers,
            # 安全检查器，用于确保生成内容的安全性
            safety_checker: StableDiffusionSafetyChecker,
            # 特征提取器，处理输入图像
            feature_extractor: CLIPImageProcessor,
            # 可选的图像编码器，用于图像的额外处理
            image_encoder: CLIPVisionModelWithProjection = None,
            # 是否需要安全检查的标志
            requires_safety_checker: bool = True,
            # 应用层的配置，控制图像处理的层次
            pag_applied_layers: Union[str, List[str]] = "mid",
        # 从稳定扩散管道中复制的函数，编码提示文本
        def encode_prompt(
            self,
            # 输入的文本提示
            prompt,
            # 设备类型（CPU或GPU）
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否执行分类器自由引导
            do_classifier_free_guidance,
            # 可选的负面提示
            negative_prompt=None,
            # 可选的提示嵌入，若已预先计算
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入，若已预先计算
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的LoRA缩放参数
            lora_scale: Optional[float] = None,
            # 可选的跳过CLIP层的参数
            clip_skip: Optional[int] = None,
        # 从稳定扩散管道中复制的函数，编码输入图像
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入不是张量，则通过特征提取器处理
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像转移到指定设备，并设置数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态，则进行隐藏状态的编码
            if output_hidden_states:
                # 编码图像并获取倒数第二层的隐藏状态
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 重复嵌入以匹配图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 对无条件图像编码
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 重复无条件图像嵌入以匹配图像数量
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回图像和无条件图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 对图像进行编码，获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 重复图像嵌入以匹配图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建与图像嵌入形状相同的零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从稳定扩散管道中复制的函数，准备图像嵌入
        def prepare_ip_adapter_image_embeds(
            # 输入的IP适配器图像
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):  # 方法结束括号，开始代码块
        image_embeds = []  # 初始化图像嵌入列表
        if do_classifier_free_guidance:  # 检查是否进行分类器自由引导
            negative_image_embeds = []  # 初始化负图像嵌入列表
        if ip_adapter_image_embeds is None:  # 检查 IP 适配器图像嵌入是否为空
            if not isinstance(ip_adapter_image, list):  # 确保 ip_adapter_image 是列表
                ip_adapter_image = [ip_adapter_image]  # 将其转换为单元素列表

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):  # 检查列表长度
                raise ValueError(  # 引发值错误异常
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."  # 异常信息，显示不匹配的长度
                )

            for single_ip_adapter_image, image_proj_layer in zip(  # 遍历 IP 适配器图像和图像投影层
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers  # 将两者打包在一起
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)  # 判断是否输出隐藏状态
                single_image_embeds, single_negative_image_embeds = self.encode_image(  # 编码单个图像并获取嵌入
                    single_ip_adapter_image, device, 1, output_hidden_state  # 调用图像编码方法
                )

                image_embeds.append(single_image_embeds[None, :])  # 将单个图像嵌入添加到列表
                if do_classifier_free_guidance:  # 如果进行分类器自由引导
                    negative_image_embeds.append(single_negative_image_embeds[None, :])  # 添加负图像嵌入
        else:  # 如果 ip_adapter_image_embeds 不是 None
            for single_image_embeds in ip_adapter_image_embeds:  # 遍历已有的图像嵌入
                if do_classifier_free_guidance:  # 如果进行分类器自由引导
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)  # 分割嵌入
                    negative_image_embeds.append(single_negative_image_embeds)  # 添加负图像嵌入
                image_embeds.append(single_image_embeds)  # 添加图像嵌入

        ip_adapter_image_embeds = []  # 初始化最终的 IP 适配器图像嵌入列表
        for i, single_image_embeds in enumerate(image_embeds):  # 遍历图像嵌入
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)  # 重复嵌入以匹配每个提示的数量
            if do_classifier_free_guidance:  # 如果进行分类器自由引导
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)  # 重复负图像嵌入
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)  # 连接负图像和正图像嵌入

            single_image_embeds = single_image_embeds.to(device=device)  # 将嵌入移动到指定设备
            ip_adapter_image_embeds.append(single_image_embeds)  # 添加到最终嵌入列表

        return ip_adapter_image_embeds  # 返回最终的 IP 适配器图像嵌入

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker  # 复制自其他模块的安全检查器方法
    # 运行安全检查器，验证图像的内容是否安全
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则将不安全内容标志设为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入的图像是张量，进行后处理以转换为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            # 如果输入的图像不是张量，将其从 numpy 数组转换为 PIL 格式
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理图像，并将其转换为设备上的张量
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，检查图像并返回处理后的图像和不安全内容标志
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和不安全内容标志
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为并非所有调度器都有相同的参数签名
        # eta (η) 仅用于 DDIMScheduler，其他调度器将忽略此参数。
        # eta 对应于 DDIM 论文中的 η：https://arxiv.org/abs/2010.02502
        # 应在 [0, 1] 范围内

        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 创建一个字典以存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数字典
        return extra_step_kwargs

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs 复制
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    # 准备潜在变量，接受批量大小、通道数、高度、宽度、数据类型、设备、生成器和可选的潜在变量
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器列表的长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果潜在变量未提供，生成随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了潜在变量，则将其移动到指定设备
                latents = latents.to(device)
    
            # 按调度器所需的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回最终的潜在变量
            return latents
    
        # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img 导入的函数
        def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
            """
            参见 https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
            参数:
                w (`torch.Tensor`):
                    使用指定的引导比例生成嵌入向量，以随后丰富时间步嵌入。
                embedding_dim (`int`, *可选*, 默认为 512):
                    生成的嵌入的维度。
                dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                    生成嵌入的数据类型。
    
            返回:
                `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
            """
            # 确保输入张量是一维的
            assert len(w.shape) == 1
            # 将 w 乘以 1000.0 进行缩放
            w = w * 1000.0
    
            # 计算嵌入的半维度
            half_dim = embedding_dim // 2
            # 计算用于生成嵌入的基数
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 生成衰减因子
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 根据输入张量和衰减因子生成嵌入
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 将正弦和余弦值连接到嵌入向量中
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度为奇数，则进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保生成的嵌入形状正确
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回生成的嵌入向量
            return emb
    
        # 返回引导比例的属性
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 返回引导重标定的属性
        @property
        def guidance_rescale(self):
            return self._guidance_rescale
    
        # 返回跳过剪辑的属性
        @property
        def clip_skip(self):
            return self._clip_skip
    
        # 这里的 `guidance_scale` 按照 Imagen 论文中的引导权重 `w` 定义
        # `guidance_scale = 1` 对应于不进行无分类器引导
        @property
    # 定义一个方法用于无分类器自由引导
        def do_classifier_free_guidance(self):
            # 检查引导比例是否大于1且时间条件投影维度为None
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 定义一个属性，用于获取交叉注意力的参数
        @property
        def cross_attention_kwargs(self):
            # 返回交叉注意力的参数
            return self._cross_attention_kwargs
    
        # 定义一个属性，用于获取时间步数
        @property
        def num_timesteps(self):
            # 返回时间步数的值
            return self._num_timesteps
    
        # 定义一个属性，用于获取中断状态
        @property
        def interrupt(self):
            # 返回中断状态的值
            return self._interrupt
    
        # 使用装饰器禁用梯度计算以提高性能
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义调用方法，接收多个参数
        def __call__(
            # 输入的提示文本，可以是字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 高度参数，默认为None
            height: Optional[int] = None,
            # 宽度参数，默认为None
            width: Optional[int] = None,
            # 推理步骤数量，默认为50
            num_inference_steps: int = 50,
            # 时间步列表，默认为None
            timesteps: List[int] = None,
            # Sigma值列表，默认为None
            sigmas: List[float] = None,
            # 引导比例，默认为7.5
            guidance_scale: float = 7.5,
            # 负提示文本，可以是字符串或字符串列表，默认为None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # Eta值，默认为0.0
            eta: float = 0.0,
            # 随机数生成器，默认为None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，默认为None
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，默认为None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入，默认为None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 图像适配器输入，默认为None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 图像适配器嵌入，默认为None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为True
            return_dict: bool = True,
            # 交叉注意力的参数，默认为None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 引导重标定值，默认为0.0
            guidance_rescale: float = 0.0,
            # 跳过的剪辑次数，默认为None
            clip_skip: Optional[int] = None,
            # 每步结束时的回调函数，默认为None
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 每步结束时的张量输入回调列表，默认为["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # PAG比例，默认为3.0
            pag_scale: float = 3.0,
            # 自适应PAG比例，默认为0.0
            pag_adaptive_scale: float = 0.0,
```