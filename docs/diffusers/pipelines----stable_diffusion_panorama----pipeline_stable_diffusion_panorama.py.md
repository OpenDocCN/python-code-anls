# `.\diffusers\pipelines\stable_diffusion_panorama\pipeline_stable_diffusion_panorama.py`

```py
# 版权声明，说明该文件的版权所有者及相关信息
# Copyright 2024 MultiDiffusion Authors and The HuggingFace Team. All rights reserved."
# 根据 Apache 许可证第 2.0 版（“许可证”）进行授权；
# 您不得在未遵守许可证的情况下使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 按照“按现状”基础分发，没有任何形式的保证或条件，
# 明示或暗示。
# 请参阅许可证以获取有关权限和
# 限制的具体说明。

# 导入用于复制对象的库
import copy
# 导入用于检查对象的库
import inspect
# 导入用于类型提示的库
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 CLIP 模型相关的处理器和模型
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 从本地模块导入图像处理和加载器相关的类
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 从本地模块导入模型相关的类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from ...models.lora import adjust_lora_scale_text_encoder
# 导入调度器
from ...schedulers import DDIMScheduler
# 导入一些实用工具
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
# 从实用工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入扩散管道和稳定扩散相关的混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从稳定扩散模块导入输出类
from ..stable_diffusion import StableDiffusionPipelineOutput
# 从安全检查器模块导入稳定扩散安全检查器
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# 初始化日志记录器，指定当前模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，提供如何使用该模块的示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

        >>> model_ckpt = "stabilityai/stable-diffusion-2-base"
        >>> scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        >>> pipe = StableDiffusionPanoramaPipeline.from_pretrained(
        ...     model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
        ... )

        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of the dolomites"
        >>> image = pipe(prompt).images[0]
        ```py
"""

# 从稳定扩散的管道中复制的函数，用于重新缩放噪声配置
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 重新缩放 `noise_cfg`。基于 [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 中的发现。见第 3.4 节
    """
    # 计算噪声预测文本的标准差，沿指定维度保持维度
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差，沿指定维度保持维度
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 根据文本标准差和配置标准差重新缩放噪声配置，以修正过曝问题
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 将原始结果与通过因子 guidance_rescale 指导的结果混合，以避免图像“过于平淡”
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回混合后的噪声配置
    return noise_cfg
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 复制而来，用于检索时间步
def retrieve_timesteps(
    scheduler,  # 调度器，用于获取时间步
    num_inference_steps: Optional[int] = None,  # 生成样本时使用的扩散步骤数，可选
    device: Optional[Union[str, torch.device]] = None,  # 指定将时间步移动到的设备，可选
    timesteps: Optional[List[int]] = None,  # 自定义时间步，用于覆盖调度器的时间步策略，可选
    sigmas: Optional[List[float]] = None,  # 自定义 sigma，用于覆盖调度器的时间步策略，可选
    **kwargs,  # 其他关键字参数，将传递给 `scheduler.set_timesteps`
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器中检索时间步。处理自定义时间步。
    任何 kwargs 将被传递到 `scheduler.set_timesteps`。

    Args:
        scheduler (`SchedulerMixin`): 
            用于获取时间步的调度器。
        num_inference_steps (`int`): 
            生成样本时使用的扩散步骤数。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` or `torch.device`, *optional*): 
            将时间步移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *optional*): 
            自定义时间步，用于覆盖调度器的时间步策略。如果传递 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *optional*): 
            自定义 sigma，用于覆盖调度器的时间步策略。如果传递 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    Returns:
        `Tuple[torch.Tensor, int]`: 一个元组，第一元素为调度器的时间步调度，第二元素为推理步骤数。
    """
    # 检查是否同时传递了自定义时间步和自定义 sigma
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了自定义时间步
    if timesteps is not None:
        # 检查当前调度器是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器中获取当前时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤数
        num_inference_steps = len(timesteps)
    # 如果传递了自定义 sigma
    elif sigmas is not None:
        # 检查当前调度器是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器中获取当前时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤数
        num_inference_steps = len(timesteps)
    # 否则分支处理
        else:
            # 设置调度器的推理步数，指定设备，并传递额外参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 获取调度器的时间步
            timesteps = scheduler.timesteps
        # 返回时间步和推理步数
        return timesteps, num_inference_steps
# 定义一个名为 StableDiffusionPanoramaPipeline 的类，继承自多个基类
class StableDiffusionPanoramaPipeline(
    # 继承 DiffusionPipeline 类
    DiffusionPipeline,
    # 继承 StableDiffusionMixin 类
    StableDiffusionMixin,
    # 继承 TextualInversionLoaderMixin 类
    TextualInversionLoaderMixin,
    # 继承 StableDiffusionLoraLoaderMixin 类
    StableDiffusionLoraLoaderMixin,
    # 继承 IPAdapterMixin 类
    IPAdapterMixin,
):
    # 文档字符串，描述该管道的用途和相关信息
    r"""
    用于通过 MultiDiffusion 生成文本到图像的管道。

    该模型继承自 [`DiffusionPipeline`]。有关所有管道的通用方法（下载、保存、在特定设备上运行等）的文档，请查看父类文档。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反演嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行标记化的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码后的图像潜在值的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            结合 `unet` 用于去噪编码后的图像潜在值的调度器。可以是 [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            分类模块，用于估计生成的图像是否可能被认为是冒犯性或有害的。
            请参考 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5) 了解有关模型潜在危害的更多详细信息。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于从生成图像中提取特征的 `CLIPImageProcessor`；作为 `safety_checker` 的输入。
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义不允许 CPU 卸载的组件
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义回调张量输入列表
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # 初始化方法，接受多个参数以初始化类
    def __init__(
        # VAE 模型参数，类型为 AutoencoderKL
        self,
        vae: AutoencoderKL,
        # 文本编码器参数，类型为 CLIPTextModel
        text_encoder: CLIPTextModel,
        # 标记化器参数，类型为 CLIPTokenizer
        tokenizer: CLIPTokenizer,
        # UNet 模型参数，类型为 UNet2DConditionModel
        unet: UNet2DConditionModel,
        # 调度器参数，类型为 DDIMScheduler
        scheduler: DDIMScheduler,
        # 安全检查器参数，类型为 StableDiffusionSafetyChecker
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器参数，类型为 CLIPImageProcessor
        feature_extractor: CLIPImageProcessor,
        # 可选参数，图像编码器，类型为 CLIPVisionModelWithProjection 或 None
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        # 可选参数，指示是否需要安全检查器，默认为 True
        requires_safety_checker: bool = True,
    ):
        # 调用父类的构造函数，初始化基类
        super().__init__()

        # 检查安全检查器是否为 None，且要求使用安全检查器
        if safety_checker is None and requires_safety_checker:
            # 记录警告，提示用户禁用了安全检查器，并强调遵守相关许可证条件
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查安全检查器不为 None，且特征提取器为 None
        if safety_checker is not None and feature_extractor is None:
            # 抛出错误，提示用户必须定义特征提取器以使用安全检查器
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 注册各个模块，便于后续使用
        self.register_modules(
            vae=vae,  # 注册变分自编码器
            text_encoder=text_encoder,  # 注册文本编码器
            tokenizer=tokenizer,  # 注册分词器
            unet=unet,  # 注册 UNet 模型
            scheduler=scheduler,  # 注册调度器
            safety_checker=safety_checker,  # 注册安全检查器
            feature_extractor=feature_extractor,  # 注册特征提取器
            image_encoder=image_encoder,  # 注册图像编码器
        )
        # 计算 VAE 的缩放因子，基于其输出通道数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化图像处理器，使用计算出的缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 将安全检查器的需求注册到配置中
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从 StableDiffusionPipeline 类复制的编码提示的方法
    def _encode_prompt(
        self,
        prompt,  # 输入的提示文本
        device,  # 设备类型，例如 'cpu' 或 'cuda'
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 负提示文本，表示不希望生成的内容
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
        **kwargs,  # 其他可选参数
    ):
        # 定义弃用消息，提醒用户该函数已弃用并将来会被移除，建议使用新的函数
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用弃用警告函数，传入函数名、版本号和警告消息
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法，传递多个参数并获取返回的元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,  # 提供的提示文本
            device=device,  # 设备（CPU或GPU）
            num_images_per_prompt=num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance=do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=negative_prompt,  # 负面提示文本
            prompt_embeds=prompt_embeds,  # 已编码的提示张量
            negative_prompt_embeds=negative_prompt_embeds,  # 已编码的负面提示张量
            lora_scale=lora_scale,  # Lora缩放因子
            **kwargs,  # 其他可选参数
        )

        # 将返回的元组中的两个张量连接起来，以兼容旧版
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回连接后的提示嵌入
        return prompt_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制的方法
    def encode_prompt(
        self,
        prompt,  # 提示文本
        device,  # 设备（CPU或GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 负面提示文本（可选）
        prompt_embeds: Optional[torch.Tensor] = None,  # 已编码的提示张量（可选）
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 已编码的负面提示张量（可选）
        lora_scale: Optional[float] = None,  # Lora缩放因子（可选）
        clip_skip: Optional[int] = None,  # 剪辑跳过（可选）
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 复制的方法
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果输入图像不是张量，则使用特征提取器将其转换为张量
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像移动到指定设备，并转换为相应的数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 获取图像编码的隐藏状态，取倒数第二个
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 重复以生成每个提示相应数量的隐藏状态
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 获取无条件图像的隐藏状态，使用全零图像
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 重复以生成每个提示相应数量的无条件隐藏状态
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回有条件和无条件的图像编码隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 如果不需要隐藏状态，获取图像编码的嵌入
            image_embeds = self.image_encoder(image).image_embeds
            # 重复以生成每个提示相应数量的图像嵌入
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建与图像嵌入相同形状的全零无条件图像嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回有条件和无条件的图像嵌入
            return image_embeds, uncond_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 复制的方法
    # 准备图像适配器图像嵌入的函数
        def prepare_ip_adapter_image_embeds(
            # 输入参数：图像适配器图像、图像嵌入、设备、每个提示的图像数量、是否进行无分类器自由引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用无分类器自由引导，初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果图像嵌入为 None，则进行处理
            if ip_adapter_image_embeds is None:
                # 如果输入的图像不是列表，则转换为列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
                # 检查图像数量与适配器数量是否匹配
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        # 抛出错误，提示图像数量与适配器数量不匹配
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
                # 遍历图像和适配器层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 输出隐藏状态的标志
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码单个图像，获取嵌入和负嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
                    # 将单个图像嵌入添加到列表
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用无分类器自由引导，添加负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 如果已有图像嵌入，遍历每个嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用无分类器自由引导，分割负嵌入和图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 添加图像嵌入到列表
                    image_embeds.append(single_image_embeds)
    
            # 初始化最终图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历图像嵌入
            for i, single_image_embeds in enumerate(image_embeds):
                # 将单个图像嵌入重复指定次数
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用无分类器自由引导，重复负嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    # 连接负嵌入和图像嵌入
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入转移到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将结果添加到最终列表
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回最终的图像适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制的代码
    # 运行安全检查器，检测给定图像的安全性
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则设置 NSFW 概念为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入图像是张量，处理为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入图像是 NumPy 数组，转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征并将其转移到指定设备上
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，并返回处理后的图像和 NSFW 概念
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 NSFW 概念
        return image, has_nsfw_concept
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的解码潜在变量的方法
    def decode_latents(self, latents):
        # 定义弃用提示信息，告知用户该方法将在未来版本中移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 发出弃用警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 根据缩放因子调整潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量，获取生成的图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 对图像进行归一化处理，并限制其值在 0 到 1 之间
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 格式，以兼容 bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回解码后的图像
        return image
    
    # 解码潜在变量并添加填充，以便进行循环推理
    def decode_latents_with_padding(self, latents: torch.Tensor, padding: int = 8) -> torch.Tensor:
        """
        Decode the given latents with padding for circular inference.
    
        Args:
            latents (torch.Tensor): The input latents to decode.
            padding (int, optional): The number of latents to add on each side for padding. Defaults to 8.
    
        Returns:
            torch.Tensor: The decoded image with padding removed.
    
        Notes:
            - The padding is added to remove boundary artifacts and improve the output quality.
            - This would slightly increase the memory usage.
            - The padding pixels are then removed from the decoded image.
    
        """
        # 根据缩放因子调整潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 获取潜在变量的左侧填充部分
        latents_left = latents[..., :padding]
        # 获取潜在变量的右侧填充部分
        latents_right = latents[..., -padding:]
        # 将填充部分与原始潜在变量合并
        latents = torch.cat((latents_right, latents, latents_left), axis=-1)
        # 解码合并后的潜在变量，获取生成的图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 计算去除填充后图像的边界像素
        padding_pix = self.vae_scale_factor * padding
        # 去除填充像素，返回最终图像
        image = image[..., padding_pix:-padding_pix]
        return image
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的准备额外步骤参数的方法
    # 准备调度器步骤的额外参数，因为并非所有调度器都有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta（η）仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 的值应在 [0, 1] 之间
    
        # 检查调度器的步骤方法是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数字典
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的步骤方法是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
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
    ):
        # 检查输入的有效性，确保参数满足要求
    
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义生成潜在变量的形状
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 如果传入的生成器是列表且长度与批量大小不匹配，则抛出错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果未提供潜在变量，则随机生成
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供潜在变量，将其移动到指定设备
            latents = latents.to(device)
    
        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回生成的潜在变量
        return latents
    
    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding 复制
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ):
        # 获取引导缩放嵌入的函数，接受张量和嵌入维度
    # 该函数返回一个形状为 (len(w), embedding_dim) 的嵌入向量
    ) -> torch.Tensor:
            """
            # 函数文档字符串，提供函数的详细信息和参数说明
            See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    
            Args:
                w (`torch.Tensor`):
                    # 输入张量，用于生成带有指定引导比例的嵌入向量，以丰富时间步嵌入
                    Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
                embedding_dim (`int`, *optional*, defaults to 512):
                    # 生成的嵌入维度，默认为 512
                    Dimension of the embeddings to generate.
                dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                    # 生成的嵌入数据类型，默认为 torch.float32
                    Data type of the generated embeddings.
    
            Returns:
                `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
            """
            # 确保输入张量 w 只有一个维度
            assert len(w.shape) == 1
            # 将 w 的值放大 1000.0
            w = w * 1000.0
    
            # 计算嵌入维度的一半
            half_dim = embedding_dim // 2
            # 计算缩放因子，用于后续的嵌入计算
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 计算每个位置的嵌入值的指数
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 将 w 转换为指定 dtype 并与嵌入值相乘
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 将正弦和余弦值拼接在一起，形成最终的嵌入
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度为奇数，则在最后进行零填充
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保最终嵌入的形状与预期一致
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回生成的嵌入
            return emb
    
        # 定义获取视图的函数，接收全景图的高度和宽度
        def get_views(
            self,
            panorama_height: int,
            panorama_width: int,
            window_size: int = 64,
            stride: int = 8,
            circular_padding: bool = False,
    # 返回视图坐标列表的函数签名，返回值类型为包含四个整数的元组列表
    ) -> List[Tuple[int, int, int, int]]:
        # 生成视图的文档字符串，说明函数的参数和返回值
        """
        # 将全景图的高度除以 8，得到缩放后的高度
        panorama_height /= 8
        # 将全景图的宽度除以 8，得到缩放后的宽度
        panorama_width /= 8
        # 计算块的高度，如果全景图高度大于窗口大小，则计算块数，否则返回 1
        num_blocks_height = (panorama_height - window_size) // stride + 1 if panorama_height > window_size else 1
        # 如果应用循环填充，计算块的宽度
        if circular_padding:
            num_blocks_width = panorama_width // stride if panorama_width > window_size else 1
        # 否则，根据宽度计算块的数量
        else:
            num_blocks_width = (panorama_width - window_size) // stride + 1 if panorama_width > window_size else 1
        # 计算总块数
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        # 初始化视图列表
        views = []
        # 遍历每个块，计算其起始和结束坐标
        for i in range(total_num_blocks):
            # 计算当前块的高度起始坐标
            h_start = int((i // num_blocks_width) * stride)
            # 计算当前块的高度结束坐标
            h_end = h_start + window_size
            # 计算当前块的宽度起始坐标
            w_start = int((i % num_blocks_width) * stride)
            # 计算当前块的宽度结束坐标
            w_end = w_start + window_size
            # 将当前块的坐标添加到视图列表中
            views.append((h_start, h_end, w_start, w_end))
        # 返回所有视图的坐标列表
        return views
    
    # 定义一个属性，用于获取引导比例
    @property
    def guidance_scale(self):
        # 返回内部存储的引导比例值
        return self._guidance_scale
    
    # 定义一个属性，用于获取引导重标定值
    @property
    def guidance_rescale(self):
        # 返回内部存储的引导重标定值
        return self._guidance_rescale
    
    # 定义一个属性，用于获取交叉注意力的关键字参数
    @property
    def cross_attention_kwargs(self):
        # 返回内部存储的交叉注意力关键字参数
        return self._cross_attention_kwargs
    
    # 定义一个属性，用于获取剪辑跳过的值
    @property
    def clip_skip(self):
        # 返回内部存储的剪辑跳过值
        return self._clip_skip
    
    # 定义一个属性，用于判断是否进行分类器自由引导
    @property
    def do_classifier_free_guidance(self):
        # 始终返回 False，表示不进行分类器自由引导
        return False
    
    # 定义一个属性，用于获取时间步数
    @property
    def num_timesteps(self):
        # 返回内部存储的时间步数值
        return self._num_timesteps
    
    # 定义一个属性，用于获取中断标志
    @property
    def interrupt(self):
        # 返回内部存储的中断标志
        return self._interrupt
    
    # 取消梯度计算的上下文装饰器，优化内存使用
    @torch.no_grad()
    # 替换文档字符串的装饰器，提供示例文档
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的方法，允许对象像函数一样被调用
        def __call__(
            # 输入提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输出图像的高度，默认为 512
            height: Optional[int] = 512,
            # 输出图像的宽度，默认为 2048
            width: Optional[int] = 2048,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 时间步的列表，可以用于控制推理过程
            timesteps: List[int] = None,
            # 指导比例，默认为 7.5
            guidance_scale: float = 7.5,
            # 视图批量大小，默认为 1
            view_batch_size: int = 1,
            # 负向提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 额外参数，用于控制生成过程的随机性，默认为 0.0
            eta: float = 0.0,
            # 生成器，可以是单个生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，可以用于控制生成图像的特征
            latents: Optional[torch.Tensor] = None,
            # 提示的嵌入表示，可以用于优化生成过程
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负向提示的嵌入表示
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 图像输入，可能用于适配器
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 图像嵌入的列表，可能用于适配器
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil" 图像格式
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的输出，默认为 True
            return_dict: bool = True,
            # 交叉注意力的额外参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 指导再缩放的值，默认为 0.0
            guidance_rescale: float = 0.0,
            # 是否使用循环填充，默认为 False
            circular_padding: bool = False,
            # 可选的剪切跳过参数
            clip_skip: Optional[int] = None,
            # 步骤结束时的回调函数
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 步骤结束时的张量输入列表，默认为包含 "latents"
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 其他任意关键字参数
            **kwargs: Any,
```