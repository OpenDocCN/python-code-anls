# `.\diffusers\pipelines\kolors\pipeline_kolors_img2img.py`

```py
# 版权信息，说明文件的所有权及使用许可
# Copyright 2024 Stability AI, Kwai-Kolors Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）许可使用此文件；
# 除非遵循许可证，否则不得使用此文件。
# 可在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是在“按现状”基础上分发的，
# 不附带任何种类的保证或条件。
# 有关许可证的具体条款和条件，请参阅许可证。
import inspect  # 导入 inspect 模块，用于获取对象的信息
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示，定义可用的类型

import PIL.Image  # 导入 PIL.Image，用于图像处理
import torch  # 导入 PyTorch 库，进行深度学习相关操作
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # 导入 CLIP 相关的图像处理和模型

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 从回调模块导入多管道回调和管道回调类
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关的类
from ...loaders import IPAdapterMixin, StableDiffusionXLLoraLoaderMixin  # 导入加载器混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入模型类
from ...models.attention_processor import AttnProcessor2_0, FusedAttnProcessor2_0, XFormersAttnProcessor  # 导入注意力处理器
from ...schedulers import KarrasDiffusionSchedulers  # 导入调度器类
from ...utils import is_torch_xla_available, logging, replace_example_docstring  # 导入工具函数
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合类
from .pipeline_output import KolorsPipelineOutput  # 导入管道输出类
from .text_encoder import ChatGLMModel  # 导入文本编码器模型
from .tokenizer import ChatGLMTokenizer  # 导入文本分词器

if is_torch_xla_available():  # 检查是否可用 Torch XLA
    import torch_xla.core.xla_model as xm  # 导入 XLA 模块

    XLA_AVAILABLE = True  # 设置 XLA 可用标志为 True
else:
    XLA_AVAILABLE = False  # 如果不支持，则设置为 False

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

EXAMPLE_DOC_STRING = """  # 示例文档字符串
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import KolorsImg2ImgPipeline  # 从 diffusers 导入图像到图像管道
        >>> from diffusers.utils import load_image  # 从工具模块导入加载图像的函数

        >>> pipe = KolorsImg2ImgPipeline.from_pretrained(  # 从预训练模型创建管道实例
        ...     "Kwai-Kolors/Kolors-diffusers", variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU
        >>> url = (  # 定义图像的 URL
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kolors/bunny_source.png"
        ... )

        >>> init_image = load_image(url)  # 从 URL 加载初始图像
        >>> prompt = "high quality image of a capybara wearing sunglasses. In the background of the image there are trees, poles, grass and other objects. At the bottom of the object there is the road., 8k, highly detailed."  # 定义生成图像的提示
        >>> image = pipe(prompt, image=init_image).images[0]  # 使用提示和初始图像生成新图像
        ```py
"""

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 复制的函数
def retrieve_latents(  # 定义函数以检索潜在变量
    encoder_output: torch.Tensor,  # 输入的编码器输出，类型为张量
    generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
    sample_mode: str = "sample"  # 采样模式，默认为“sample”
):
    # 检查 encoder_output 是否具有 latent_dist 属性，并且样本模式为 "sample"
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            # 从 latent_dist 中获取样本
            return encoder_output.latent_dist.sample(generator)
        # 检查 encoder_output 是否具有 latent_dist 属性，并且样本模式为 "argmax"
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            # 返回 latent_dist 的众数
            return encoder_output.latent_dist.mode()
        # 检查 encoder_output 是否具有 latents 属性
        elif hasattr(encoder_output, "latents"):
            # 返回 latents 属性的值
            return encoder_output.latents
        # 如果以上条件都不满足，抛出属性错误
        else:
            raise AttributeError("Could not access latents of provided encoder_output")
# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion 中复制的
def retrieve_timesteps(
    # 调度器对象，负责时间步的管理
    scheduler,
    # 用于生成样本的推理步骤数量，默认为 None
    num_inference_steps: Optional[int] = None,
    # 指定设备类型，默认为 None
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步列表，默认为 None
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 列表，默认为 None
    sigmas: Optional[List[float]] = None,
    # 额外的关键字参数
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器获取时间步。处理自定义时间步。
    任何关键字参数将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`):
            获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数量。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            时间步要移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            自定义时间步，用于覆盖调度器的时间步间隔策略。如果传递了 `timesteps`，
            `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义 sigma，用于覆盖调度器的时间步间隔策略。如果传递了 `sigmas`，
            `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，元组的第一个元素是调度器的时间步调度，
        第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了时间步和 sigma，若是则抛出异常
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了时间步
    if timesteps is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出异常
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法设置时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取当前的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传递了 sigma
    elif sigmas is not None:
        # 检查调度器的 `set_timesteps` 方法是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，则抛出异常
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的 `set_timesteps` 方法设置 sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取当前的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    else:  # 如果不满足前面的条件，则执行此代码块
        # 设置调度器的时间步数，使用指定的设备和其他参数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取调度器的时间步数
        timesteps = scheduler.timesteps
    # 返回时间步数和推理步数
    return timesteps, num_inference_steps
# 定义 KolorsImg2ImgPipeline 类，继承多个父类以实现文本到图像生成的功能
class KolorsImg2ImgPipeline(DiffusionPipeline, StableDiffusionMixin, StableDiffusionXLLoraLoaderMixin, IPAdapterMixin):
    r"""
    使用 Kolors 进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查阅父类文档以获取库为所有管道实现的通用方法（如下载、保存、在特定设备上运行等）。

    该管道还继承以下加载方法：
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`ChatGLMModel`]):
            冻结的文本编码器。Kolors 使用 [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b)。
        tokenizer (`ChatGLMTokenizer`):
            [ChatGLMTokenizer](https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py) 类的分词器。
        unet ([`UNet2DConditionModel`]): 条件 U-Net 架构，用于去噪编码后的图像潜在表示。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于去噪编码的图像潜在表示。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
        force_zeros_for_empty_prompt (`bool`, *可选*, 默认为 `"False"`):
            是否始终将负提示嵌入强制设置为 0。另请参见 `Kwai-Kolors/Kolors-diffusers` 的配置。
    """

    # 定义模型在 CPU 上的卸载顺序
    model_cpu_offload_seq = "text_encoder->image_encoder-unet->vae"
    # 定义可选组件列表
    _optional_components = [
        "image_encoder",
        "feature_extractor",
    ]
    # 定义回调张量输入列表
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    # 初始化方法，设置管道的参数
    def __init__(
        self,
        vae: AutoencoderKL,  # VAE 模型，用于图像的编码和解码
        text_encoder: ChatGLMModel,  # 文本编码器，负责处理输入文本
        tokenizer: ChatGLMTokenizer,  # 分词器，用于将文本转换为模型可处理的格式
        unet: UNet2DConditionModel,  # U-Net 模型，用于图像去噪
        scheduler: KarrasDiffusionSchedulers,  # 调度器，控制去噪过程
        image_encoder: CLIPVisionModelWithProjection = None,  # 可选的图像编码器
        feature_extractor: CLIPImageProcessor = None,  # 可选的特征提取器
        force_zeros_for_empty_prompt: bool = False,  # 是否将空提示的负嵌入强制为 0
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个模块，包括 VAE、文本编码器、分词器、UNet、调度器、图像编码器和特征提取器
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        # 将配置参数注册到对象中，强制为空提示时使用零
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        # 计算 VAE 的缩放因子，根据 VAE 配置的块输出通道数决定值，若无 VAE 则默认为 8
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # 创建图像处理器，使用之前计算的 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # 获取 UNet 配置中的默认采样大小
        self.default_sample_size = self.unet.config.sample_size

    # 从 diffusers.pipelines.kolors.pipeline_kolors.KolorsPipeline 复制的 encode_prompt 方法
    def encode_prompt(
        self,
        # 提示文本输入
        prompt,
        # 可选的设备参数
        device: Optional[torch.device] = None,
        # 每个提示生成的图像数量
        num_images_per_prompt: int = 1,
        # 是否使用无分类器自由引导
        do_classifier_free_guidance: bool = True,
        # 可选的负提示文本
        negative_prompt=None,
        # 可选的提示嵌入
        prompt_embeds: Optional[torch.FloatTensor] = None,
        # 可选的池化提示嵌入
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        # 可选的负池化提示嵌入
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # 提示的最大序列长度
        max_sequence_length: int = 256,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的 encode_image 方法
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果输入的图像不是张量，使用特征提取器将其转换为张量格式
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像转移到指定设备并转换为指定数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 获取图像编码器的隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 根据每个提示的图像数量重复隐藏状态
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 对无条件图像进行编码以获取隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 根据每个提示的图像数量重复无条件图像的隐藏状态
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回图像编码的隐藏状态和无条件图像的隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 编码图像以获取图像嵌入
            image_embeds = self.image_encoder(image).image_embeds
            # 根据每个提示的图像数量重复图像嵌入
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建与图像嵌入同样形状的全零张量作为无条件图像嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回图像嵌入和无条件图像嵌入
            return image_embeds, uncond_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的 prepare_ip_adapter_image_embeds 方法
    # 准备 IP 适配器图像嵌入的函数
        def prepare_ip_adapter_image_embeds(
            # 输入参数：IP 适配器图像、图像嵌入、设备、每个提示的图像数量、是否进行无分类器自由引导
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果进行无分类器自由引导，则初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果图像嵌入为 None
            if ip_adapter_image_embeds is None:
                # 如果输入的图像不是列表，则将其转换为列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
    
                # 检查输入图像的数量与 IP 适配器数量是否一致
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        # 抛出值错误，说明输入图像数量与 IP 适配器数量不匹配
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
    
                # 遍历每个单一 IP 适配器图像及其对应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断是否输出隐藏状态
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 编码图像，获取单一图像嵌入和单一负图像嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
    
                    # 将单一图像嵌入添加到嵌入列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果进行无分类器自由引导，则添加单一负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 遍历已提供的图像嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果进行无分类器自由引导，则拆分负图像嵌入和正图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        # 将负图像嵌入添加到列表中
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 将图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds)
    
            # 初始化最终的 IP 适配器图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历每个图像嵌入
            for i, single_image_embeds in enumerate(image_embeds):
                # 将单一图像嵌入重复 num_images_per_prompt 次
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果进行无分类器自由引导，则重复负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    # 连接负图像嵌入和正图像嵌入
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
    
                # 将图像嵌入转移到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 添加处理后的图像嵌入到最终列表中
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回 IP 适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的代码
    # 准备额外参数用于调度器步骤，因为并非所有调度器具有相同的参数签名
    def prepare_extra_step_kwargs(self, generator, eta):
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略该参数
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 之间
    
        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        # 如果调度器接受 eta，添加到额外步骤参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，添加到额外步骤参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回包含额外参数的字典
        return extra_step_kwargs
    
    # 检查输入参数的有效性
    def check_inputs(
        self,
        prompt,
        strength,
        num_inference_steps,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps 复制
    # 获取时间步长，参数包括推理步骤数、强度、设备和去噪起始时间
        def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
            # 获取原始时间步，使用 init_timestep 计算
            if denoising_start is None:
                # 计算初始时间步，确保不超过总推理步骤数
                init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
                # 计算起始时间步，确保不小于0
                t_start = max(num_inference_steps - init_timestep, 0)
            else:
                # 如果有去噪起始时间，则从0开始
                t_start = 0
    
            # 从调度器中获取时间步，从计算的起始位置开始
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
    
            # 如果指定了去噪起始时间，则强度不再相关
            if denoising_start is not None:
                # 计算离散时间步截止值，基于去噪起始时间
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_start * self.scheduler.config.num_train_timesteps)
                    )
                )
    
                # 计算有效的推理步骤数
                num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
                # 如果调度器为二阶调度器，检查推理步骤是否为偶数
                if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                    # 处理偶数步骤的情况，确保去噪过程结束在正确的导数步骤
                    num_inference_steps = num_inference_steps + 1
    
                # 从最后开始切片时间步
                timesteps = timesteps[-num_inference_steps:]
                # 返回时间步和有效推理步骤数
                return timesteps, num_inference_steps
    
            # 如果没有去噪起始时间，返回时间步和推理步骤数减去起始步
            return timesteps, num_inference_steps - t_start
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.prepare_latents 复制
        def prepare_latents(
            self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids 复制
        def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    # 追加时间 ID 列表，合并原始大小、裁剪坐标和目标大小
        ):
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
            # 计算通过模型配置生成的嵌入维度
            passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
            )
            # 获取模型期望的嵌入维度
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    
            # 检查生成的嵌入维度与期望值是否匹配
            if expected_add_embed_dim != passed_add_embed_dim:
                # 如果不匹配，抛出值错误
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )
    
            # 将追加时间 ID 转换为张量，并指定数据类型
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
            # 返回处理后的追加时间 ID
            return add_time_ids
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.upcast_vae 复制的函数
        def upcast_vae(self):
            # 获取 VAE 的数据类型
            dtype = self.vae.dtype
            # 将 VAE 转换为 float32 数据类型
            self.vae.to(dtype=torch.float32)
            # 检查是否使用了 Torch 2.0 或 Xformers 处理器
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                    FusedAttnProcessor2_0,
                ),
            )
            # 如果使用了 Xformers 或 Torch 2.0，注意力块不需要为 float32，从而节省内存
            if use_torch_2_0_or_xformers:
                # 将后量化卷积层转换为指定数据类型
                self.vae.post_quant_conv.to(dtype)
                # 将输入卷积层转换为指定数据类型
                self.vae.decoder.conv_in.to(dtype)
                # 将中间块转换为指定数据类型
                self.vae.decoder.mid_block.to(dtype)
    
        # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding 复制的函数
        def get_guidance_scale_embedding(
            # 输入张量和嵌入维度，默认为 512，数据类型默认为 float32
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    # 此函数返回生成的嵌入向量，具有指定的引导尺度
        ) -> torch.Tensor:
            """
            参见指定链接以获取详细文档
    
            参数:
                w (`torch.Tensor`):
                    使用指定的引导尺度生成嵌入向量，以后用于丰富时间步嵌入。
                embedding_dim (`int`, *可选*, 默认值为 512):
                    要生成的嵌入的维度。
                dtype (`torch.dtype`, *可选*, 默认值为 `torch.float32`):
                    生成的嵌入的数据类型。
    
            返回:
                `torch.Tensor`: 形状为 `(len(w), embedding_dim)` 的嵌入向量。
            """
            # 确保输入张量 w 的形状是一维的
            assert len(w.shape) == 1
            # 将 w 的值放大 1000 倍
            w = w * 1000.0
    
            # 计算嵌入维度的一半
            half_dim = embedding_dim // 2
            # 计算用于缩放的对数值
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            # 生成半个维度的嵌入值并取指数
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            # 根据输入 w 转换数据类型并扩展维度
            emb = w.to(dtype)[:, None] * emb[None, :]
            # 将正弦和余弦值串联在一起
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            # 如果嵌入维度是奇数，则填充零
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            # 确保最终嵌入的形状符合预期
            assert emb.shape == (w.shape[0], embedding_dim)
            # 返回生成的嵌入
            return emb
    
        # 返回当前的引导尺度
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 此属性定义与 Imagen 论文中的引导权重 w 类似的 `guidance_scale`
        # `guidance_scale = 1` 表示不进行无分类器引导
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 返回当前的交叉注意力参数
        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs
    
        # 返回去噪的起始点
        @property
        def denoising_start(self):
            return self._denoising_start
    
        # 返回去噪的结束点
        @property
        def denoising_end(self):
            return self._denoising_end
    
        # 返回时间步的数量
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 返回中断标志
        @property
        def interrupt(self):
            return self._interrupt
    
        # 禁用梯度计算并替换示例文档字符串
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用的方法，接受多个参数用于处理图像生成任务
        def __call__(
            # 提示文本，可以是单个字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输入图像，用于生成的图像输入
            image: PipelineImageInput = None,
            # 控制生成强度的参数，默认为0.3
            strength: float = 0.3,
            # 输出图像的高度，可选
            height: Optional[int] = None,
            # 输出图像的宽度，可选
            width: Optional[int] = None,
            # 推理步骤的数量，默认为50
            num_inference_steps: int = 50,
            # 指定时间步列表，可选
            timesteps: List[int] = None,
            # 指定 sigma 值的列表，可选
            sigmas: List[float] = None,
            # 去噪开始的值，可选
            denoising_start: Optional[float] = None,
            # 去噪结束的值，可选
            denoising_end: Optional[float] = None,
            # 指导比例，默认为5.0
            guidance_scale: float = 5.0,
            # 负面提示文本，可选，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 影响生成的 eta 值，默认为0.0
            eta: float = 0.0,
            # 随机数生成器，可选，可以是单个生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，可选，形状为张量
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入，可选，形状为张量
            prompt_embeds: Optional[torch.Tensor] = None,
            # 池化的提示嵌入，可选，形状为张量
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示嵌入，可选，形状为张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面池化的提示嵌入，可选，形状为张量
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 输入适配器图像，可选
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 输入适配器图像嵌入，可选，张量列表
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，可选，默认为“pil”
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为True
            return_dict: bool = True,
            # 交叉注意力参数，可选，字典类型
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 原始图像的尺寸，可选，元组类型
            original_size: Optional[Tuple[int, int]] = None,
            # 图像裁剪的左上角坐标，默认为(0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标尺寸，可选，元组类型
            target_size: Optional[Tuple[int, int]] = None,
            # 负面原始图像的尺寸，可选，元组类型
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负面图像裁剪的左上角坐标，默认为(0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负面目标尺寸，可选，元组类型
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 步骤结束时的回调函数，可选，支持多种类型
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 结束步骤的张量输入列表，默认为["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 最大序列长度，默认为256
            max_sequence_length: int = 256,
```