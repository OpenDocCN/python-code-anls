# `.\diffusers\pipelines\stable_diffusion_ldm3d\pipeline_stable_diffusion_ldm3d.py`

```py
# 版权声明，说明版权所有者及相关团队
# Copyright 2024 The Intel Labs Team Authors and the HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0（“许可证”）许可； 
# 除非符合许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定， 
# 否则根据许可证分发的软件在“原样”基础上提供， 
# 不提供任何形式的担保或条件。
# 请参见许可证，以了解有关权限和限制的具体内容。

import inspect  # 导入inspect模块以进行对象获取和检查
from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
from typing import Any, Callable, Dict, List, Optional, Union  # 导入常用类型提示

import numpy as np  # 导入NumPy库以进行数值计算
import PIL.Image  # 导入PIL库中的Image模块以处理图像
import torch  # 导入PyTorch库以进行深度学习操作
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 导入Transformers库中的CLIP相关模型和处理器

from ...image_processor import PipelineImageInput, VaeImageProcessorLDM3D  # 从相对路径导入图像处理相关类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入不同类型的加载器混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入不同的模型类
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整Lora文本编码器的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入Karras扩散调度器
from ...utils import (  # 从utils模块导入多个工具函数和常量
    USE_PEFT_BACKEND,
    BaseOutput,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor  # 从torch_utils模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从pipeline_utils模块导入扩散管道和稳定扩散混合类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 从stable_diffusion模块导入安全检查器

logger = logging.get_logger(__name__)  # 创建一个记录器，用于记录模块中的日志信息，禁用pylint对名称无效的警告

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，用于展示如何使用StableDiffusionLDM3DPipeline
    Examples:
        ```python
        >>> from diffusers import StableDiffusionLDM3DPipeline  # 从diffusers模块导入StableDiffusionLDM3DPipeline

        >>> pipe = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-4c")  # 从预训练模型加载管道
        >>> pipe = pipe.to("cuda")  # 将管道移动到GPU设备上

        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义生成图像的提示
        >>> output = pipe(prompt)  # 使用提示生成图像输出
        >>> rgb_image, depth_image = output.rgb, output.depth  # 从输出中提取RGB图像和深度图像
        >>> rgb_image[0].save("astronaut_ldm3d_rgb.jpg")  # 保存RGB图像
        >>> depth_image[0].save("astronaut_ldm3d_depth.png")  # 保存深度图像
        ```py
"""

# 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion模块复制的函数
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):  # 定义一个函数，用于根据guidance_rescale重缩放噪声配置
    """
    根据`guidance_rescale`重缩放`noise_cfg`。基于[Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)中的发现。见第3.4节
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)  # 计算文本预测噪声的标准差
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)  # 计算噪声配置的标准差
    # 根据引导结果重缩放噪声（修复过度曝光问题）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)  # 通过标准差比例重缩放噪声配置
    # 将原始结果与通过因子 guidance_rescale 指导的结果混合，以避免“平淡无奇”的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回混合后的噪声配置
    return noise_cfg
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 复制的函数
def retrieve_timesteps(
    # 调度器对象，用于获取时间步
    scheduler,
    # 推理步骤的数量，可选参数，默认为 None
    num_inference_steps: Optional[int] = None,
    # 设备类型，可选参数，默认为 None
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步，可选参数，默认为 None
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 值，可选参数，默认为 None
    sigmas: Optional[List[float]] = None,
    # 其他关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器检索时间步。处理自定义时间步。
    任何 kwargs 将被传递给 `scheduler.set_timesteps`。
    
    参数：
        scheduler (`SchedulerMixin`):
            获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数。如果使用，`timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步要移动到的设备。如果为 `None`，时间步将不被移动。
        timesteps (`List[int]`, *可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递了 `timesteps`，
            则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义 sigma，用于覆盖调度器的时间步间隔策略。如果传递了 `sigmas`，
            则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回：
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是来自调度器的时间步调度，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了自定义时间步和自定义 sigma
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了自定义时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取当前的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传递了自定义 sigma
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取当前的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    else:  # 如果前面的条件不满足，则执行以下代码
        # 设置调度器的时间步长，传入推理步骤数量和设备参数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取当前调度器的时间步长
        timesteps = scheduler.timesteps
    # 返回时间步长和推理步骤数量
    return timesteps, num_inference_steps
# 定义 LDM3D 输出数据类，继承自 BaseOutput
@dataclass
class LDM3DPipelineOutput(BaseOutput):
    """
    输出类，用于稳定扩散管道。

    参数:
        rgb (`List[PIL.Image.Image]` 或 `np.ndarray`)
            表示去噪后的 PIL 图像列表，长度为 `batch_size` 或形状为 `(batch_size, height, width,
            num_channels)` 的 NumPy 数组。
        depth (`List[PIL.Image.Image]` 或 `np.ndarray`)
            表示去噪后的 PIL 图像列表，长度为 `batch_size` 或形状为 `(batch_size, height, width,
            num_channels)` 的 NumPy 数组。
        nsfw_content_detected (`List[bool]`)
            表示相应生成图像是否包含“不适合工作” (nsfw) 内容的列表，如果无法执行安全检查则为 `None`。
    """

    # 定义 rgb 属性，可以是 PIL 图像列表或 NumPy 数组
    rgb: Union[List[PIL.Image.Image], np.ndarray]
    # 定义 depth 属性，可以是 PIL 图像列表或 NumPy 数组
    depth: Union[List[PIL.Image.Image], np.ndarray]
    # 定义 nsfw_content_detected 属性，表示每个图像的安全性检测结果
    nsfw_content_detected: Optional[List[bool]]


# 定义稳定扩散 LDM3D 管道类，继承多个混合类
class StableDiffusionLDM3DPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    FromSingleFileMixin,
):
    r"""
    用于文本到图像和 3D 生成的管道，使用 LDM3D。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取所有管道的通用方法（下载、保存、在特定设备上运行等）。

    此管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器
    # 定义参数说明部分
        Args:
            vae ([`AutoencoderKL`]):
                Variational Auto-Encoder (VAE) 模型，用于编码和解码图像到潜在表示。
            text_encoder ([`~transformers.CLIPTextModel`]):
                冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
            tokenizer ([`~transformers.CLIPTokenizer`]):
                用于将文本标记化的 `CLIPTokenizer`。
            unet ([`UNet2DConditionModel`]):
                用于对编码的图像潜在表示进行去噪的 `UNet2DConditionModel`。
            scheduler ([`SchedulerMixin`]):
                与 `unet` 结合使用的调度器，用于对编码的图像潜在表示进行去噪。可以是
                [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
            safety_checker ([`StableDiffusionSafetyChecker`]):
                分类模块，用于评估生成的图像是否可能被视为冒犯或有害。
                有关模型潜在危害的更多详细信息，请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
            feature_extractor ([`~transformers.CLIPImageProcessor`]):
                用于从生成图像中提取特征的 `CLIPImageProcessor`；作为输入用于 `safety_checker`。
        """
    
        # 定义模型在 CPU 上的卸载顺序
        model_cpu_offload_seq = "text_encoder->unet->vae"
        # 定义可选组件的列表
        _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
        # 定义在 CPU 卸载时排除的组件
        _exclude_from_cpu_offload = ["safety_checker"]
        # 定义回调张量输入的列表
        _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    
        # 初始化方法定义
        def __init__(
            # 初始化所需的 VAE 模型
            vae: AutoencoderKL,
            # 初始化所需的文本编码器
            text_encoder: CLIPTextModel,
            # 初始化所需的标记器
            tokenizer: CLIPTokenizer,
            # 初始化所需的 UNet 模型
            unet: UNet2DConditionModel,
            # 初始化所需的调度器
            scheduler: KarrasDiffusionSchedulers,
            # 初始化所需的安全检查器
            safety_checker: StableDiffusionSafetyChecker,
            # 初始化所需的特征提取器
            feature_extractor: CLIPImageProcessor,
            # 可选的图像编码器
            image_encoder: Optional[CLIPVisionModelWithProjection],
            # 指示是否需要安全检查器的布尔值
            requires_safety_checker: bool = True,
    # 初始化父类
        ):
            super().__init__()
    
            # 检查是否禁用安全检查器且需要安全检查器
            if safety_checker is None and requires_safety_checker:
                # 记录警告信息，提醒用户有关安全检查器的使用建议
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 检查是否提供了安全检查器但未提供特征提取器
            if safety_checker is not None and feature_extractor is None:
                # 引发错误，提示用户需定义特征提取器
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 注册各个模块到当前对象
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器，使用 VAE 缩放因子
            self.image_processor = VaeImageProcessorLDM3D(vae_scale_factor=self.vae_scale_factor)
            # 将需要的配置注册到当前对象
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 从 StableDiffusionPipeline 复制的方法，用于编码提示
        def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            lora_scale: Optional[float] = None,
            **kwargs,
    # 定义一个方法，可能是类中的一个部分
        ):
            # 创建一个关于 `_encode_prompt()` 方法被弃用的提示信息
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用 deprecate 函数，记录弃用信息和版本
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 方法，传入一系列参数，获取提示嵌入元组
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,  # 输入提示
                device=device,  # 设备类型（CPU或GPU）
                num_images_per_prompt=num_images_per_prompt,  # 每个提示生成的图像数量
                do_classifier_free_guidance=do_classifier_free_guidance,  # 是否使用无分类器引导
                negative_prompt=negative_prompt,  # 负面提示
                prompt_embeds=prompt_embeds,  # 提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,  # 负面提示嵌入
                lora_scale=lora_scale,  # Lora缩放因子
                **kwargs,  # 其他关键字参数
            )
    
            # 将提示嵌入元组中的两个元素连接成一个张量，用于向后兼容
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回合并后的提示嵌入
            return prompt_embeds
    
        # 从 StableDiffusionPipeline 复制的 encode_prompt 方法定义
        def encode_prompt(
            self,
            prompt,  # 输入的提示文本
            device,  # 设备类型（CPU或GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=None,  # 负面提示，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
            clip_skip: Optional[int] = None,  # 可选的剪辑跳过参数
        # 从 StableDiffusionPipeline 复制的 encode_image 方法定义
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入图像是否为张量，如果不是则进行转换
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像数据移动到指定设备并转换为指定类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态，则处理隐藏状态
            if output_hidden_states:
                # 获取编码后的图像隐藏状态，并重复图像以适应生成数量
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 获取无条件图像的编码隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 重复无条件图像隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回图像和无条件图像的编码隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 获取编码后的图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 重复图像嵌入以适应生成数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建一个与图像嵌入大小相同的零张量作为无条件图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回图像嵌入和无条件图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 StableDiffusionPipeline 复制的 prepare_ip_adapter_image_embeds 方法定义
    # 准备 IP 适配器图像的嵌入表示
    def prepare_ip_adapter_image_embeds(
        # 定义方法的参数，包括图像、图像嵌入、设备、每个提示的图像数量和分类自由引导标志
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化图像嵌入列表
        image_embeds = []
        # 如果启用分类自由引导
        if do_classifier_free_guidance:
            # 初始化负图像嵌入列表
            negative_image_embeds = []
        # 如果图像嵌入为空
        if ip_adapter_image_embeds is None:
            # 如果传入的图像不是列表，将其转为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查图像数量与 IP 适配器的数量是否一致
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果不一致，抛出值错误
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历图像和相应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 检查图像投影层是否为 ImageProjection 类型
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 编码单个图像，返回图像嵌入和负图像嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将单个图像嵌入添加到嵌入列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用分类自由引导，将负图像嵌入添加到列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果已提供图像嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用分类自由引导，分离负图像嵌入和图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负图像嵌入添加到列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将图像嵌入添加到列表中
                image_embeds.append(single_image_embeds)

        # 初始化 IP 适配器图像嵌入列表
        ip_adapter_image_embeds = []
        # 遍历图像嵌入列表
        for i, single_image_embeds in enumerate(image_embeds):
            # 复制嵌入以生成每个提示的图像数量
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用分类自由引导
            if do_classifier_free_guidance:
                # 复制负图像嵌入以生成每个提示的图像数量
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入和图像嵌入拼接在一起
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入转移到指定设备上
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到结果列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回 IP 适配器图像嵌入列表
        return ip_adapter_image_embeds
    # 定义运行安全检查器的方法，接收图像、设备和数据类型作为参数
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则没有 NSFW 概念
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果图像是一个张量，则进行后处理以转换为 PIL 图像
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果图像不是张量，则将 NumPy 数组转换为 PIL 图像
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取第一个图像作为 RGB 特征输入
            rgb_feature_extractor_input = feature_extractor_input[0]
            # 使用特征提取器将 RGB 输入转换为张量并移动到指定设备
            safety_checker_input = self.feature_extractor(rgb_feature_extractor_input, return_tensors="pt").to(device)
            # 使用安全检查器检查图像，返回处理后的图像和 NSFW 概念的标识
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 NSFW 概念的标识
        return image, has_nsfw_concept

    # 从 StableDiffusionPipeline 复制的准备额外步骤参数的方法
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外参数，因为不是所有调度器具有相同的签名
        # eta（η）仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 之间

        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外步骤参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外步骤参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数字典
        return extra_step_kwargs

    # 从 StableDiffusionPipeline 复制的检查输入的方法
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
    # 准备潜在变量，用于生成模型的输入
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状，依据批量大小和通道数
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且其长度与批量大小一致
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出值错误，提示生成器长度与批量大小不匹配
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果未提供潜在变量，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将提供的潜在变量转移到指定设备
            latents = latents.to(device)
    
        # 将初始噪声按调度器要求的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents
    
    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img 的 LatentConsistencyModelPipeline 复制
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        查看链接 https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
        参数:
            w (`torch.Tensor`):
                生成带有指定引导缩放的嵌入向量，以丰富时间步嵌入。
            embedding_dim (`int`, *可选*, 默认为 512):
                要生成的嵌入维度。
            dtype (`torch.dtype`, *可选*, 默认为 `torch.float32`):
                生成的嵌入数据类型。
    
        返回:
            `torch.Tensor`: 嵌入向量，形状为 `(len(w), embedding_dim)`。
        """
        # 确保输入张量 w 是一维的
        assert len(w.shape) == 1
        # 将 w 乘以 1000.0
        w = w * 1000.0
    
        # 计算半维度
        half_dim = embedding_dim // 2
        # 计算嵌入的缩放因子
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 生成指数衰减的嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        # 计算最终嵌入，按 w 进行缩放
        emb = w.to(dtype)[:, None] * emb[None, :]
        # 连接正弦和余弦变换的嵌入
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 如果嵌入维度为奇数，则进行零填充
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        # 确保生成的嵌入形状正确
        assert emb.shape == (w.shape[0], embedding_dim)
        # 返回生成的嵌入
        return emb
    
    # 定义属性，返回引导缩放因子
    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    # 定义属性，返回引导重缩放因子
    @property
    def guidance_rescale(self):
        return self._guidance_rescale
    
    # 定义属性，返回跳过剪辑的标志
    @property
    def clip_skip(self):
        return self._clip_skip
    
    # 这里的 `guidance_scale` 定义类似于 Imagen 论文中方程 (2) 的引导权重 `w`
    # https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # 对应于不进行分类器自由引导。
    @property
    # 定义一个方法，用于判断是否进行无分类器引导
        def do_classifier_free_guidance(self):
            # 判断引导比例是否大于1并且时间条件投影维度是否为None
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 定义一个属性，返回交叉注意力的关键字参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 定义一个属性，返回时间步数
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 定义一个属性，返回中断状态
        @property
        def interrupt(self):
            return self._interrupt
    
        # 使用无梯度计算装饰器，定义调用方法
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 定义调用方法的参数，prompt可以是字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 定义可选的高度参数
            height: Optional[int] = None,
            # 定义可选的宽度参数
            width: Optional[int] = None,
            # 设置推理步骤的默认值为49
            num_inference_steps: int = 49,
            # 定义可选的时间步数列表
            timesteps: List[int] = None,
            # 定义可选的sigma值列表
            sigmas: List[float] = None,
            # 设置引导比例的默认值为5.0
            guidance_scale: float = 5.0,
            # 定义可选的负面提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 定义每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 设置eta的默认值为0.0
            eta: float = 0.0,
            # 定义可选的生成器，可以是单个或多个torch.Generator
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 定义可选的潜在向量
            latents: Optional[torch.Tensor] = None,
            # 定义可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 定义可选的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 定义可选的图像输入适配器
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 定义可选的图像适配器嵌入列表
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 设置输出类型的默认值为"pil"
            output_type: Optional[str] = "pil",
            # 设置返回字典的默认值为True
            return_dict: bool = True,
            # 定义可选的交叉注意力关键字参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 设置引导重标定的默认值为0.0
            guidance_rescale: float = 0.0,
            # 定义可选的剪切跳过参数
            clip_skip: Optional[int] = None,
            # 定义可选的步骤结束回调函数
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 定义步骤结束时的张量输入回调参数，默认包括"latents"
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 接收其他关键字参数
            **kwargs,
```