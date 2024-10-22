# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_unclip.py`

```py
# 版权信息，表示此文件的版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证发布的声明
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵循许可证的情况下才能使用此文件
# you may not use this file except in compliance with the License.
# 可在以下链接获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律适用或书面同意，否则本软件按 "现状" 方式分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 详见许可证中对权限和限制的具体规定
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块，用于检查对象的属性和方法
import inspect
# 从 typing 模块导入各种类型注解，方便类型检查
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 CLIP 文本模型和分词器
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
# 从 CLIP 模型中导入文本模型输出的类型
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

# 导入图像处理器
from ...image_processor import VaeImageProcessor
# 导入加载器的混合类，用于处理特定加载逻辑
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 导入自动编码器和其他模型
from ...models import AutoencoderKL, PriorTransformer, UNet2DConditionModel
# 导入时间步嵌入函数
from ...models.embeddings import get_timestep_embedding
# 导入 LoRA 相关的调整函数
from ...models.lora import adjust_lora_scale_text_encoder
# 导入 Karras 扩散调度器
from ...schedulers import KarrasDiffusionSchedulers
# 导入工具函数，处理各种实用功能
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
# 导入用于生成随机张量的函数
from ...utils.torch_utils import randn_tensor
# 导入扩散管道相关类
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin
# 导入稳定反向图像归一化器
from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

# 创建日志记录器，记录该模块的日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用 StableUnCLIPPipeline
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableUnCLIPPipeline

        >>> pipe = StableUnCLIPPipeline.from_pretrained(
        ...     "fusing/stable-unclip-2-1-l", torch_dtype=torch.float16
        ... )  # TODO update model path
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> images = pipe(prompt).images
        >>> images[0].save("astronaut_horse.png")
        ```py
"""

# 定义 StableUnCLIPPipeline 类，继承多个混合类
class StableUnCLIPPipeline(
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin
):
    """
    使用稳定反向 CLIP 的文本到图像生成管道。

    此模型继承自 [`DiffusionPipeline`]。请查阅超类文档以获取所有管道实现的通用方法
    (下载、保存、在特定设备上运行等)。

    该管道还继承以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
    # 定义函数参数
    Args:
        prior_tokenizer ([`CLIPTokenizer`]):  # 指定用于文本的 CLIP 标记器
            A [`CLIPTokenizer`].  # 说明这是一个 CLIP 标记器
        prior_text_encoder ([`CLIPTextModelWithProjection`]):  # 指定冻结的 CLIP 文本编码器
            Frozen [`CLIPTextModelWithProjection`] text-encoder.  # 说明这是一个冻结的文本编码器
        prior ([`PriorTransformer`]):  # 指定用于图像嵌入的 unCLIP 先验模型
            The canonical unCLIP prior to approximate the image embedding from the text embedding.  # 说明这是一个标准的 unCLIP 先验，用于从文本嵌入近似图像嵌入
        prior_scheduler ([`KarrasDiffusionSchedulers`]):  # 指定用于去噪过程的调度器
            Scheduler used in the prior denoising process.  # 说明这是在先验去噪过程中使用的调度器
        image_normalizer ([`StableUnCLIPImageNormalizer`]):  # 指定用于标准化图像嵌入的标准化器
            Used to normalize the predicted image embeddings before the noise is applied and un-normalize the image  # 说明用于在应用噪声之前标准化预测的图像嵌入，并在应用噪声后反标准化图像嵌入
            embeddings after the noise has been applied.  # 继续说明标准化的过程
        image_noising_scheduler ([`KarrasDiffusionSchedulers`]):  # 指定用于添加噪声的调度器
            Noise schedule for adding noise to the predicted image embeddings. The amount of noise to add is determined  # 说明这是用于对预测的图像嵌入添加噪声的调度器，噪声量由 `noise_level` 决定
            by the `noise_level`.  # 说明噪声量的确定依据
        tokenizer ([`CLIPTokenizer`]):  # 指定用于文本的 CLIP 标记器
            A [`CLIPTokenizer`].  # 说明这是一个 CLIP 标记器
        text_encoder ([`CLIPTextModel`]):  # 指定冻结的 CLIP 文本编码器
            Frozen [`CLIPTextModel`] text-encoder.  # 说明这是一个冻结的文本编码器
        unet ([`UNet2DConditionModel`]):  # 指定用于去噪的 UNet 模型
            A [`UNet2DConditionModel`] to denoise the encoded image latents.  # 说明这是一个用于去噪编码图像潜变量的 UNet 模型
        scheduler ([`KarrasDiffusionSchedulers`]):  # 指定与 UNet 结合使用的调度器
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.  # 说明这是与 UNet 结合使用的去噪调度器
        vae ([`AutoencoderKL`]):  # 指定变分自编码器模型
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.  # 说明这是一个变分自编码器模型，用于将图像编码和解码为潜在表示
    """  # 结束文档字符串

    _exclude_from_cpu_offload = ["prior", "image_normalizer"]  # 指定不进行 CPU 卸载的组件列表
    model_cpu_offload_seq = "text_encoder->prior_text_encoder->unet->vae"  # 定义模型的 CPU 卸载顺序

    # prior components  # 注释说明以下是先验组件
    prior_tokenizer: CLIPTokenizer  # 声明 prior_tokenizer 为 CLIPTokenizer 类型
    prior_text_encoder: CLIPTextModelWithProjection  # 声明 prior_text_encoder 为 CLIPTextModelWithProjection 类型
    prior: PriorTransformer  # 声明 prior 为 PriorTransformer 类型
    prior_scheduler: KarrasDiffusionSchedulers  # 声明 prior_scheduler 为 KarrasDiffusionSchedulers 类型

    # image noising components  # 注释说明以下是图像噪声组件
    image_normalizer: StableUnCLIPImageNormalizer  # 声明 image_normalizer 为 StableUnCLIPImageNormalizer 类型
    image_noising_scheduler: KarrasDiffusionSchedulers  # 声明 image_noising_scheduler 为 KarrasDiffusionSchedulers 类型

    # regular denoising components  # 注释说明以下是常规去噪组件
    tokenizer: CLIPTokenizer  # 声明 tokenizer 为 CLIPTokenizer 类型
    text_encoder: CLIPTextModel  # 声明 text_encoder 为 CLIPTextModel 类型
    unet: UNet2DConditionModel  # 声明 unet 为 UNet2DConditionModel 类型
    scheduler: KarrasDiffusionSchedulers  # 声明 scheduler 为 KarrasDiffusionSchedulers 类型

    vae: AutoencoderKL  # 声明 vae 为 AutoencoderKL 类型

    def __init__(  # 定义构造函数
        self,  # 指向实例本身
        # prior components  # 注释说明以下是先验组件
        prior_tokenizer: CLIPTokenizer,  # 指定构造函数参数 prior_tokenizer 为 CLIPTokenizer 类型
        prior_text_encoder: CLIPTextModelWithProjection,  # 指定构造函数参数 prior_text_encoder 为 CLIPTextModelWithProjection 类型
        prior: PriorTransformer,  # 指定构造函数参数 prior 为 PriorTransformer 类型
        prior_scheduler: KarrasDiffusionSchedulers,  # 指定构造函数参数 prior_scheduler 为 KarrasDiffusionSchedulers 类型
        # image noising components  # 注释说明以下是图像噪声组件
        image_normalizer: StableUnCLIPImageNormalizer,  # 指定构造函数参数 image_normalizer 为 StableUnCLIPImageNormalizer 类型
        image_noising_scheduler: KarrasDiffusionSchedulers,  # 指定构造函数参数 image_noising_scheduler 为 KarrasDiffusionSchedulers 类型
        # regular denoising components  # 注释说明以下是常规去噪组件
        tokenizer: CLIPTokenizer,  # 指定构造函数参数 tokenizer 为 CLIPTokenizer 类型
        text_encoder: CLIPTextModelWithProjection,  # 指定构造函数参数 text_encoder 为 CLIPTextModelWithProjection 类型
        unet: UNet2DConditionModel,  # 指定构造函数参数 unet 为 UNet2DConditionModel 类型
        scheduler: KarrasDiffusionSchedulers,  # 指定构造函数参数 scheduler 为 KarrasDiffusionSchedulers 类型
        # vae  # 注释说明以下是变分自编码器
        vae: AutoencoderKL,  # 指定构造函数参数 vae 为 AutoencoderKL 类型
    # 结束括号，表示类构造函数的参数列表结束
    ):
        # 调用父类构造函数
        super().__init__()

        # 注册所需模块及其参数
        self.register_modules(
            prior_tokenizer=prior_tokenizer,  # 注册先前的分词器
            prior_text_encoder=prior_text_encoder,  # 注册先前的文本编码器
            prior=prior,  # 注册先前模型
            prior_scheduler=prior_scheduler,  # 注册先前的调度器
            image_normalizer=image_normalizer,  # 注册图像归一化器
            image_noising_scheduler=image_noising_scheduler,  # 注册图像噪声调度器
            tokenizer=tokenizer,  # 注册当前的分词器
            text_encoder=text_encoder,  # 注册当前的文本编码器
            unet=unet,  # 注册 U-Net 模型
            scheduler=scheduler,  # 注册调度器
            vae=vae,  # 注册变分自编码器
        )

        # 计算 VAE 的缩放因子，基于其配置中的输出通道数
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器，使用 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # 从 UnCLIPPipeline 复制的方法，调整为处理先前的提示
    def _encode_prior_prompt(
        self,
        prompt,  # 输入提示
        device,  # 设备信息
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器自由引导
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,  # 可选的文本模型输出
        text_attention_mask: Optional[torch.Tensor] = None,  # 可选的文本注意力掩码
    # 从 StableDiffusionPipeline 复制的方法，调整为处理当前的提示
    def _encode_prompt(
        self,
        prompt,  # 输入提示
        device,  # 设备信息
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器自由引导
        negative_prompt=None,  # 可选的负面提示
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
        lora_scale: Optional[float] = None,  # 可选的 Lora 缩放因子
        **kwargs,  # 其他可选参数
    ):
        # 发出弃用警告，提示将来版本中移除此方法
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用新的编码提示方法，生成嵌入元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,  # 输入提示
            device=device,  # 设备信息
            num_images_per_prompt=num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance=do_classifier_free_guidance,  # 是否使用无分类器自由引导
            negative_prompt=negative_prompt,  # 负面提示
            prompt_embeds=prompt_embeds,  # 提示嵌入
            negative_prompt_embeds=negative_prompt_embeds,  # 负面提示嵌入
            lora_scale=lora_scale,  # Lora 缩放因子
            **kwargs,  # 其他参数
        )

        # 将嵌入元组的内容连接以兼容旧版本
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回合并后的提示嵌入
        return prompt_embeds

    # 从 StableDiffusionPipeline 复制的方法，用于编码提示
    # 定义一个编码提示的函数
        def encode_prompt(
            # 提示内容
            self,
            prompt,
            # 设备信息
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否进行无分类器自由引导
            do_classifier_free_guidance,
            # 可选的负面提示
            negative_prompt=None,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 Lora 缩放因子
            lora_scale: Optional[float] = None,
            # 可选的剪辑跳过参数
            clip_skip: Optional[int] = None,
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
        def decode_latents(self, latents):
            # 定义过时消息
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            # 调用 deprecate 函数以显示警告信息
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 根据 VAE 配置缩放因子调整潜变量
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜变量生成图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 将图像数据归一化到 [0, 1] 之间
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为 float32 格式以便兼容 bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回解码后的图像
            return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
        def prepare_prior_extra_step_kwargs(self, generator, eta):
            # 准备用于 prior_scheduler 步骤的额外参数，因为并非所有的 prior_schedulers 具有相同的参数签名
            # eta 仅在 DDIMScheduler 中使用，其他 prior_schedulers 会忽略它
            # eta 在 DDIM 论文中的对应符号为 η: https://arxiv.org/abs/2010.02502
            # eta 的值应在 [0, 1] 之间
    
            # 检查 prior_scheduler 是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.prior_scheduler.step).parameters.keys())
            # 初始化额外参数字典
            extra_step_kwargs = {}
            # 如果接受 eta，添加到额外参数字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查 prior_scheduler 是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.prior_scheduler.step).parameters.keys())
            # 如果接受 generator，添加到额外参数字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数
            return extra_step_kwargs
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    # 定义准备额外参数的函数，供调度器步骤使用
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤所需的额外参数，因为不同调度器的参数签名不同
        # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 其值应在 [0, 1] 之间
    
        # 检查调度器的步骤函数是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化存储额外参数的字典
        extra_step_kwargs = {}
        # 如果接受 eta 参数，则将其添加到额外参数字典中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器的步骤函数是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator 参数，则将其添加到额外参数字典中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回构建好的额外参数字典
        return extra_step_kwargs
    
    # 定义检查输入参数的函数
    def check_inputs(
        self,
        prompt,  # 输入提示
        height,  # 生成图像的高度
        width,   # 生成图像的宽度
        callback_steps,  # 回调的步数
        noise_level,  # 噪声级别
        negative_prompt=None,  # 可选的负面提示
        prompt_embeds=None,  # 可选的提示嵌入
        negative_prompt_embeds=None,  # 可选的负面提示嵌入
    ):
        # 检查高度和宽度是否为8的倍数，如果不是，则抛出值错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步骤是否为正整数，如果不是，则抛出值错误
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查是否同时提供了提示和提示嵌入，如果是，则抛出值错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Please make sure to define only one of the two."
            )

        # 检查提示和提示嵌入是否都未定义，如果是，则抛出值错误
        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )

        # 检查提示是否为字符串或列表，如果不是，则抛出值错误
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了负提示和负提示嵌入，如果是，则抛出值错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Provide either `negative_prompt` or `negative_prompt_embeds`. Cannot leave both `negative_prompt` and `negative_prompt_embeds` undefined."
            )

        # 检查是否同时提供了提示和负提示，如果是，则检查类型是否一致
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        # 检查提示嵌入和负提示嵌入的形状是否一致，如果不一致则抛出值错误
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # 检查噪声级别是否在有效范围内，如果不在，则抛出值错误
        if noise_level < 0 or noise_level >= self.image_noising_scheduler.config.num_train_timesteps:
            raise ValueError(
                f"`noise_level` must be between 0 and {self.image_noising_scheduler.config.num_train_timesteps - 1}, inclusive."
            )

    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents 复制的代码
    # 准备潜在向量，参数包括形状、数据类型、设备、生成器、潜在向量和调度器
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果没有给定潜在向量，则随机生成一个
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查给定潜在向量的形状是否与预期形状一致
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在向量移动到指定设备
            latents = latents.to(device)

        # 将潜在向量与调度器的初始噪声标准差相乘
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在向量
        return latents

    # 为图像嵌入添加噪声，噪声量由噪声级别控制
    def noise_image_embeddings(
        self,
        image_embeds: torch.Tensor,
        noise_level: int,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        向图像嵌入添加噪声，噪声量由 `noise_level` 输入控制。较高的
        `noise_level` 会增加最终未去噪图像的方差。

        噪声的应用有两种方式：
        1. 噪声调度直接应用于嵌入。
        2. 将正弦时间嵌入向量附加到输出。

        在这两种情况下，噪声量均由相同的 `noise_level` 控制。

        嵌入在应用噪声之前会被归一化，应用噪声后会被反归一化。
        """
        # 如果没有提供噪声，则随机生成噪声
        if noise is None:
            noise = randn_tensor(
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )

        # 创建与图像嵌入数量相同的噪声级别张量
        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        # 将图像归一化器移动到图像嵌入所在设备
        self.image_normalizer.to(image_embeds.device)
        # 对图像嵌入进行归一化处理
        image_embeds = self.image_normalizer.scale(image_embeds)

        # 使用噪声调度器将噪声添加到图像嵌入
        image_embeds = self.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

        # 对图像嵌入进行反归一化处理
        image_embeds = self.image_normalizer.unscale(image_embeds)

        # 获取时间步嵌入，控制噪声的时间步
        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        # 将时间步嵌入转换为与图像嵌入相同的数据类型
        noise_level = noise_level.to(image_embeds.dtype)

        # 将时间步嵌入与图像嵌入拼接在一起
        image_embeds = torch.cat((image_embeds, noise_level), 1)

        # 返回包含噪声的图像嵌入
        return image_embeds

    # 禁用梯度计算以提高推理性能
    @torch.no_grad()
    # 用示例文档字符串替换当前文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的方法，用于处理去噪过程
    def __call__(
        self,
        # 正常的去噪过程参数
        # 提示文本，可以是字符串或字符串列表，默认为 None
        prompt: Optional[Union[str, List[str]]] = None,
        # 输出图像的高度，默认为 None
        height: Optional[int] = None,
        # 输出图像的宽度，默认为 None
        width: Optional[int] = None,
        # 推理步骤的数量，默认为 20
        num_inference_steps: int = 20,
        # 引导比例，默认为 10.0
        guidance_scale: float = 10.0,
        # 负提示文本，可以是字符串或字符串列表，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 控制噪声的参数，默认为 0.0
        eta: float = 0.0,
        # 用于随机数生成的生成器，默认为 None
        generator: Optional[torch.Generator] = None,
        # 潜在变量张量，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 提示嵌入张量，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示嵌入张量，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式，默认为 True
        return_dict: bool = True,
        # 可选的回调函数，接收步骤和张量，默认为 None
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调函数调用的步骤间隔，默认为 1
        callback_steps: int = 1,
        # 跨注意力的可选参数，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 噪声水平，默认为 0
        noise_level: int = 0,
        # 先验参数
        # 先验推理步骤的数量，默认为 25
        prior_num_inference_steps: int = 25,
        # 先验引导比例，默认为 4.0
        prior_guidance_scale: float = 4.0,
        # 先验潜在变量张量，默认为 None
        prior_latents: Optional[torch.Tensor] = None,
        # 可选的跳过剪辑步骤，默认为 None
        clip_skip: Optional[int] = None,
```