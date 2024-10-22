# `.\diffusers\pipelines\kolors\pipeline_kolors.py`

```py
# 版权声明，表示代码的版权所有者和保留权利
# Copyright 2024 Stability AI, Kwai-Kolors Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵守许可证的情况下才能使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非根据适用法律或书面协议另有规定，否则按“原样”提供软件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 参见许可证以获取特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect  # 导入inspect模块以便进行对象检查
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示工具

import torch  # 导入PyTorch库以进行张量计算
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # 导入transformers库中的图像处理器和模型

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # 导入回调相关的模块
from ...image_processor import PipelineImageInput, VaeImageProcessor  # 导入图像处理相关的模块
from ...loaders import IPAdapterMixin, StableDiffusionXLLoraLoaderMixin  # 导入加载器相关的混合类
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel  # 导入模型相关的类
from ...models.attention_processor import AttnProcessor2_0, FusedAttnProcessor2_0, XFormersAttnProcessor  # 导入注意力处理器
from ...schedulers import KarrasDiffusionSchedulers  # 导入调度器
from ...utils import is_torch_xla_available, logging, replace_example_docstring  # 导入工具函数
from ...utils.torch_utils import randn_tensor  # 导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入管道相关的类
from .pipeline_output import KolorsPipelineOutput  # 导入管道输出相关的类
from .text_encoder import ChatGLMModel  # 导入文本编码模型
from .tokenizer import ChatGLMTokenizer  # 导入聊天GLM的分词器


if is_torch_xla_available():  # 检查是否可用torch_xla
    import torch_xla.core.xla_model as xm  # 导入torch_xla相关模块

    XLA_AVAILABLE = True  # 如果可用，设置标志为True
else:
    XLA_AVAILABLE = False  # 否则设置为False


logger = logging.get_logger(__name__)  # 创建当前模块的日志记录器，禁用pylint警告


EXAMPLE_DOC_STRING = """  # 示例文档字符串，用于展示如何使用代码
    Examples:
        ```py
        >>> import torch  # 导入torch库
        >>> from diffusers import KolorsPipeline  # 从diffusers导入Kolors管道

        >>> pipe = KolorsPipeline.from_pretrained(  # 从预训练模型创建Kolors管道
        ...     "Kwai-Kolors/Kolors-diffusers", variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到GPU设备

        >>> prompt = (  # 定义生成图像的提示
        ...     "A photo of a ladybug, macro, zoom, high quality, film, holding a wooden sign with the text 'KOLORS'"
        ... )
        >>> image = pipe(prompt).images[0]  # 生成图像并获取第一张图像
        ```py
"""


# 从stable_diffusion管道复制的函数，用于检索时间步
def retrieve_timesteps(
    scheduler,  # 调度器对象
    num_inference_steps: Optional[int] = None,  # 可选的推理步骤数量
    device: Optional[Union[str, torch.device]] = None,  # 可选的设备信息
    timesteps: Optional[List[int]] = None,  # 可选的时间步列表
    sigmas: Optional[List[float]] = None,  # 可选的sigma值列表
    **kwargs,  # 其他关键字参数
):
    """
    调用调度器的`set_timesteps`方法并在调用后从调度器获取时间步。处理
    自定义时间步。任何关键字参数将传递给`scheduler.set_timesteps`。
    # 参数说明
    Args:
        scheduler (`SchedulerMixin`):  # 调度器，用于获取时间步
            The scheduler to get timesteps from.  # 从调度器中获取时间步
        num_inference_steps (`int`):  # 生成样本时使用的扩散步骤数
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.  # 如果使用此参数，`timesteps`必须为`None`
        device (`str` or `torch.device`, *optional*):  # 指定时间步移动到的设备
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.  # 如果为`None`，则不移动时间步
        timesteps (`List[int]`, *optional*):  # 自定义时间步，用于覆盖调度器的时间步间隔策略
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.  # 如果传递了`timesteps`，则`num_inference_steps`和`sigmas`必须为`None`
        sigmas (`List[float]`, *optional*):  # 自定义sigmas，用于覆盖调度器的时间步间隔策略
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.  # 如果传递了`sigmas`，则`num_inference_steps`和`timesteps`必须为`None`

    Returns:
        `Tuple[torch.Tensor, int]`:  # 返回一个元组
        A tuple where the first element is the timestep schedule from the scheduler and the  # 第一个元素是来自调度器的时间步安排
        second element is the number of inference steps.  # 第二个元素是推理步骤的数量
    """
    # 检查`timesteps`和`sigmas`是否同时存在
    if timesteps is not None and sigmas is not None:
        # 抛出错误，提示只能传递`timesteps`或`sigmas`中的一个
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果`timesteps`存在
    if timesteps is not None:
        # 检查调度器的`set_timesteps`方法是否接受`timesteps`参数
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的`set_timesteps`方法，设置时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果`sigmas`存在
    elif sigmas is not None:
        # 检查调度器的`set_timesteps`方法是否接受`sigmas`参数
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        # 如果不接受，抛出错误
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 调用调度器的`set_timesteps`方法，设置sigmas
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果`timesteps`和`sigmas`都不存在
    else:
        # 调用调度器的`set_timesteps`方法，使用默认的推理步骤数量
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取设置后的时间步
        timesteps = scheduler.timesteps
    # 返回时间步和推理步骤数量
    return timesteps, num_inference_steps
# 定义 KolorsPipeline 类，继承自多个父类以实现文本到图像生成
class KolorsPipeline(DiffusionPipeline, StableDiffusionMixin, StableDiffusionXLLoraLoaderMixin, IPAdapterMixin):
    # 文档字符串，说明该管道的用途和继承关系
    r"""
    Pipeline for text-to-image generation using Kolors.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`ChatGLMModel`]):
            Frozen text-encoder. Kolors uses [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b).
        tokenizer (`ChatGLMTokenizer`):
            Tokenizer of class
            [ChatGLMTokenizer](https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"False"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `Kwai-Kolors/Kolors-diffusers`.
    """

    # 定义模型在 CPU 上卸载的顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 定义可选组件的列表
    _optional_components = [
        "image_encoder",
        "feature_extractor",
    ]
    # 定义回调张量输入的列表
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    # 初始化方法，定义类的属性和参数
    def __init__(
        # 定义变分自编码器模型
        self,
        vae: AutoencoderKL,
        # 定义冻结的文本编码器
        text_encoder: ChatGLMModel,
        # 定义分词器
        tokenizer: ChatGLMTokenizer,
        # 定义条件 U-Net 模型
        unet: UNet2DConditionModel,
        # 定义调度器，用于图像去噪
        scheduler: KarrasDiffusionSchedulers,
        # 可选的图像编码器
        image_encoder: CLIPVisionModelWithProjection = None,
        # 可选的特征提取器
        feature_extractor: CLIPImageProcessor = None,
        # 是否强制将空提示的负向嵌入设置为零
        force_zeros_for_empty_prompt: bool = False,
    # 初始化父类
        ):
            super().__init__()
    
            # 注册模块，包括 VAE、文本编码器、分词器等
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
            # 将配置中的强制零填充选项注册到配置中
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            # 计算 VAE 的缩放因子，如果存在 VAE 则取其通道数的块数
            self.vae_scale_factor = (
                2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
            )
            # 初始化图像处理器，使用 VAE 缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
            # 设置默认样本大小，从 UNet 的配置中获取
            self.default_sample_size = self.unet.config.sample_size
    
        # 编码提示的函数，处理各种输入参数
        def encode_prompt(
            self,
            prompt,
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            max_sequence_length: int = 256,
        # 从 diffusers 库中复制的编码图像的函数
        def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 如果输入的图像不是张量，则进行特征提取
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(image, return_tensors="pt").pixel_values
    
            # 将图像数据移动到指定设备，并设置数据类型
            image = image.to(device=device, dtype=dtype)
            # 如果需要输出隐藏状态，进行相应的编码
            if output_hidden_states:
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                # 重复编码的隐藏状态以匹配每个提示的图像数量
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
                # 编码未条件化图像的隐藏状态
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
                # 重复未条件化图像的隐藏状态
                uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                # 返回编码后的图像和未条件化图像的隐藏状态
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                # 编码图像并获取图像嵌入
                image_embeds = self.image_encoder(image).image_embeds
                # 重复图像嵌入以匹配每个提示的图像数量
                image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                # 创建未条件化的图像嵌入
                uncond_image_embeds = torch.zeros_like(image_embeds)
    
                # 返回编码后的图像嵌入和未条件化的图像嵌入
                return image_embeds, uncond_image_embeds
    
        # 从 diffusers 库中复制的准备 IP 适配器图像嵌入的函数
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        # 初始化一个空列表，用于存储图像嵌入
        image_embeds = []
        # 如果启用无分类器自由引导，则初始化一个空列表，用于存储负图像嵌入
        if do_classifier_free_guidance:
            negative_image_embeds = []
        # 如果输入适配器的图像嵌入为 None
        if ip_adapter_image_embeds is None:
            # 检查 ip_adapter_image 是否为列表，如果不是，则将其转换为列表
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 检查 ip_adapter_image 的长度是否与 IP 适配器的数量相同
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                # 如果长度不匹配，抛出 ValueError
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 遍历每个适配器图像及其对应的图像投影层
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 检查当前图像投影层是否为 ImageProjection 的实例，决定是否输出隐藏状态
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                # 对单个适配器图像进行编码，返回嵌入
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # 将当前图像嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds[None, :])
                # 如果启用无分类器自由引导，则将负图像嵌入添加到负图像嵌入列表中
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # 如果输入适配器的图像嵌入不为 None，则遍历这些嵌入
            for single_image_embeds in ip_adapter_image_embeds:
                # 如果启用无分类器自由引导，则将当前图像嵌入拆分为负图像嵌入和正图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    # 将负图像嵌入添加到负图像嵌入列表中
                    negative_image_embeds.append(single_negative_image_embeds)
                # 将正图像嵌入添加到图像嵌入列表中
                image_embeds.append(single_image_embeds)

        # 初始化一个空列表，用于存储最终的适配器图像嵌入
        ip_adapter_image_embeds = []
        # 遍历图像嵌入及其索引
        for i, single_image_embeds in enumerate(image_embeds):
            # 将每个图像嵌入复制 num_images_per_prompt 次
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            # 如果启用无分类器自由引导，则复制负图像嵌入
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                # 将负图像嵌入与正图像嵌入拼接
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            # 将图像嵌入移动到指定的设备上
            single_image_embeds = single_image_embeds.to(device=device)
            # 将处理后的图像嵌入添加到最终列表中
            ip_adapter_image_embeds.append(single_image_embeds)

        # 返回最终的适配器图像嵌入列表
        return ip_adapter_image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    # 准备额外参数用于调度器步骤，因为并非所有调度器具有相同的参数签名
        def prepare_extra_step_kwargs(self, generator, eta):
            # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 值应在 [0, 1] 之间
    
            # 检查调度器步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外步骤参数的字典
            extra_step_kwargs = {}
            # 如果调度器接受 eta，添加其值到字典
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果调度器接受 generator，添加其值到字典
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 检查输入参数的有效性
        def check_inputs(
            self,
            prompt,
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
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 计算生成的张量形状
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查传入的 generator 列表长度是否与 batch_size 匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果未提供 latents，生成随机张量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了 latents，将其移动到指定设备
                latents = latents.to(device)
    
            # 将初始噪声按调度器所需的标准差进行缩放
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的噪声张量
            return latents
    
        # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids 复制
        def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        # 将原始尺寸、裁剪坐标和目标尺寸合并为一个列表
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        # 计算通过添加时间嵌入维度和文本编码器投影维度得到的总嵌入维度
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        # 获取模型期望的添加时间嵌入的输入特征维度
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # 检查计算得到的维度与期望的维度是否相等
        if expected_add_embed_dim != passed_add_embed_dim:
            # 如果不相等，抛出错误提示
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        # 将添加的时间 ID 转换为张量，指定数据类型
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        # 返回添加的时间 ID 张量
        return add_time_ids

    # 从 diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.upcast_vae 复制而来
    def upcast_vae(self):
        # 获取 VAE 模型的数据类型
        dtype = self.vae.dtype
        # 将 VAE 模型转换为 float32 类型
        self.vae.to(dtype=torch.float32)
        # 检查是否使用 torch 2.0 或 xformers
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # 如果使用 xformers 或 torch 2.0，则注意力模块不需要在 float32 中，从而节省大量内存
        if use_torch_2_0_or_xformers:
            # 将后量化卷积层转换为指定数据类型
            self.vae.post_quant_conv.to(dtype)
            # 将解码器输入卷积层转换为指定数据类型
            self.vae.decoder.conv_in.to(dtype)
            # 将解码器中间块转换为指定数据类型
            self.vae.decoder.mid_block.to(dtype)

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding 复制而来
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:  # 定义函数返回类型为 torch.Tensor
        """  # 文档字符串开始，描述函数的功能
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298  # 参考链接
        
        Args:  # 参数说明开始
            w (`torch.Tensor`):  # 输入的张量 w，用于生成嵌入向量
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.  # 生成嵌入向量以丰富时间步嵌入
            embedding_dim (`int`, *optional*, defaults to 512):  # 嵌入维度，可选，默认为512
                Dimension of the embeddings to generate.  # 生成的嵌入维度
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):  # 数据类型，可选，默认为 torch.float32
                Data type of the generated embeddings.  # 生成嵌入的数值类型
        
        Returns:  # 返回值说明开始
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.  # 返回形状为 (len(w), embedding_dim) 的嵌入张量
        """  # 文档字符串结束
        assert len(w.shape) == 1  # 断言 w 的形状是一维的
        w = w * 1000.0  # 将 w 的值放大1000倍

        half_dim = embedding_dim // 2  # 计算嵌入维度的一半
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)  # 计算对数并归一化
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)  # 生成指数衰减的嵌入
        emb = w.to(dtype)[:, None] * emb[None, :]  # 将 w 转换为目标数据类型并进行广播乘法
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # 将正弦和余弦嵌入在维度1上拼接
        if embedding_dim % 2 == 1:  # 如果嵌入维度为奇数
            emb = torch.nn.functional.pad(emb, (0, 1))  # 在最后一维进行零填充
        assert emb.shape == (w.shape[0], embedding_dim)  # 断言嵌入的形状符合预期
        return emb  # 返回生成的嵌入张量

    @property  # 将以下方法声明为属性
    def guidance_scale(self):  # 定义 guidance_scale 属性
        return self._guidance_scale  # 返回内部存储的引导比例

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)  # 解释 guidance_scale 的定义
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`  # 说明相关文献
    # corresponds to doing no classifier free guidance.  # 指出值为1时不进行无分类器引导
    @property  # 将以下方法声明为属性
    def do_classifier_free_guidance(self):  # 定义 do_classifier_free_guidance 属性
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None  # 返回是否启用无分类器引导的布尔值

    @property  # 将以下方法声明为属性
    def cross_attention_kwargs(self):  # 定义 cross_attention_kwargs 属性
        return self._cross_attention_kwargs  # 返回交叉注意力参数

    @property  # 将以下方法声明为属性
    def denoising_end(self):  # 定义 denoising_end 属性
        return self._denoising_end  # 返回去噪结束位置

    @property  # 将以下方法声明为属性
    def num_timesteps(self):  # 定义 num_timesteps 属性
        return self._num_timesteps  # 返回时间步数

    @property  # 将以下方法声明为属性
    def interrupt(self):  # 定义 interrupt 属性
        return self._interrupt  # 返回中断状态

    @torch.no_grad()  # 禁用梯度计算，提升推理性能
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 用示例文档字符串替换当前文档字符串
    # 定义可调用对象的方法，允许使用一系列参数进行处理
        def __call__(
            # 提示文本，可以是字符串或字符串列表，默认为 None
            self,
            prompt: Union[str, List[str]] = None,
            # 输出图像的高度，默认为 None
            height: Optional[int] = None,
            # 输出图像的宽度，默认为 None
            width: Optional[int] = None,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 时间步列表，默认为 None
            timesteps: List[int] = None,
            # 噪声级别的列表，默认为 None
            sigmas: List[float] = None,
            # 去噪结束的阈值，默认为 None
            denoising_end: Optional[float] = None,
            # 引导尺度，默认为 5.0
            guidance_scale: float = 5.0,
            # 负提示文本，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 用于控制生成过程的参数，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，可以是单个或多个，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量的张量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示的嵌入张量，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 经过池化的提示嵌入张量，默认为 None
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的嵌入张量，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示的池化嵌入张量，默认为 None
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 图像适配器输入，默认为 None
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 图像适配器的嵌入张量列表，默认为 None
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 跨注意力的参数，默认为 None
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 原始图像的尺寸，默认为 None
            original_size: Optional[Tuple[int, int]] = None,
            # 裁剪坐标的左上角，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 目标尺寸，默认为 None
            target_size: Optional[Tuple[int, int]] = None,
            # 负样本的原始尺寸，默认为 None
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 负样本的裁剪坐标的左上角，默认为 (0, 0)
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 负样本的目标尺寸，默认为 None
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 步骤结束时的回调函数，默认为 None
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 用于步骤结束回调的张量输入列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 最大序列长度，默认为 256
            max_sequence_length: int = 256,
```