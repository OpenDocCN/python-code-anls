# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_unclip_img2img.py`

```py
# 版权声明，2024年由 HuggingFace 团队保留所有权利。
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守该许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面协议另有约定，根据许可证分发的软件是按“原样”提供的，
# 不提供任何种类的担保或条件，无论是明示还是暗示。
# 请参阅许可证以了解特定语言规定的权限和限制。

import inspect  # 导入 inspect 模块以获取有关活跃对象的信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示，便于静态类型检查

import PIL.Image  # 导入 PIL.Image 用于图像处理
import torch  # 导入 PyTorch 以使用深度学习功能
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # 从 transformers 导入 CLIP 相关模型和处理器

from ...image_processor import VaeImageProcessor  # 导入 VAE 图像处理器
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入稳定扩散和文本反转加载混合器
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入模型类
from ...models.embeddings import get_timestep_embedding  # 导入获取时间步嵌入的函数
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 LoRA 缩放的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入 Karras 扩散调度器
from ...utils import (  # 导入工具函数
    USE_PEFT_BACKEND,  # 导入用于 PEFT 后端的常量
    deprecate,  # 导入用于标记弃用功能的装饰器
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入用于替换示例文档字符串的函数
    scale_lora_layers,  # 导入缩放 LoRA 层的函数
    unscale_lora_layers,  # 导入取消缩放 LoRA 层的函数
)
from ...utils.torch_utils import randn_tensor  # 从自定义 PyTorch 工具中导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin  # 导入扩散管道及其输出类
from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer  # 导入稳定 UnCLIP 图像归一化器

logger = logging.get_logger(__name__)  # 创建一个用于当前模块的日志记录器，禁用 pylint 名称检查

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串，用于展示如何使用管道
    Examples:  # 示例的说明
        ```py  # 示例代码块的开始
        >>> import requests  # 导入 requests 库用于发起 HTTP 请求
        >>> import torch  # 导入 PyTorch 库
        >>> from PIL import Image  # 从 PIL 导入图像处理模块
        >>> from io import BytesIO  # 导入 BytesIO 用于处理字节流

        >>> from diffusers import StableUnCLIPImg2ImgPipeline  # 从 diffusers 导入 StableUnCLIPImg2ImgPipeline 类

        >>> pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(  # 从预训练模型加载管道
        ...     "stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16  # 指定模型名称和数据类型
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 CUDA 设备以加速计算

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"  # 定义要加载的图像 URL

        >>> response = requests.get(url)  # 发起 GET 请求以获取图像
        >>> init_image = Image.open(BytesIO(response.content)).convert("RGB")  # 打开图像并转换为 RGB 格式
        >>> init_image = init_image.resize((768, 512))  # 将图像调整为指定大小

        >>> prompt = "A fantasy landscape, trending on artstation"  # 定义生成图像的提示文本

        >>> images = pipe(init_image, prompt).images  # 使用管道生成图像
        >>> images[0].save("fantasy_landscape.png")  # 保存生成的图像为 PNG 文件
        ```py  # 示例代码块的结束
"""

class StableUnCLIPImg2ImgPipeline(  # 定义 StableUnCLIPImg2ImgPipeline 类
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin  # 继承多个混合器以实现功能组合
):
    """
    使用稳定的 unCLIP 进行文本引导的图像到图像生成的管道。  # 类文档字符串，说明该类的功能

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解所有管道的通用方法（下载、保存、在特定设备上运行等）。  # 提供有关超类的信息
    # 管道还继承以下加载方法：
    # - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
    # - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
    # - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
    
    # 参数说明：
    # feature_extractor ([`CLIPImageProcessor`]):
    # 图像预处理的特征提取器，编码之前使用。
    # image_encoder ([`CLIPVisionModelWithProjection`]):
    # CLIP 视觉模型，用于编码图像。
    # image_normalizer ([`StableUnCLIPImageNormalizer`]):
    # 用于在添加噪声之前规范化预测的图像嵌入，并在添加噪声后反规范化图像嵌入。
    # image_noising_scheduler ([`KarrasDiffusionSchedulers`]):
    # 添加噪声到预测的图像嵌入的噪声调度器，噪声量由 `noise_level` 决定。
    # tokenizer (`~transformers.CLIPTokenizer`):
    # 一个 [`~transformers.CLIPTokenizer`]。
    # text_encoder ([`~transformers.CLIPTextModel`]):
    # 冻结的 [`~transformers.CLIPTextModel`] 文本编码器。
    # unet ([`UNet2DConditionModel`]):
    # 用于去噪编码图像潜变量的 [`UNet2DConditionModel`]。
    # scheduler ([`KarrasDiffusionSchedulers`]):
    # 与 `unet` 结合使用的调度器，用于去噪编码图像潜变量。
    # vae ([`AutoencoderKL`]):
    # 变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。
    # """
    
    # 定义模型在 CPU 上卸载的顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    # 要从 CPU 卸载中排除的组件
    _exclude_from_cpu_offload = ["image_normalizer"]
    
    # 图像编码组件
    feature_extractor: CLIPImageProcessor
    image_encoder: CLIPVisionModelWithProjection
    
    # 图像加噪声组件
    image_normalizer: StableUnCLIPImageNormalizer
    image_noising_scheduler: KarrasDiffusionSchedulers
    
    # 常规去噪组件
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    scheduler: KarrasDiffusionSchedulers
    
    # 变分自编码器
    vae: AutoencoderKL
    
    # 初始化方法
    def __init__(
        # 图像编码组件
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        # 图像加噪声组件
        image_normalizer: StableUnCLIPImageNormalizer,
        image_noising_scheduler: KarrasDiffusionSchedulers,
        # 常规去噪组件
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        # 变分自编码器
        vae: AutoencoderKL,
    # 初始化父类
        ):
            super().__init__()
    
            # 注册模块，包括特征提取器、图像编码器等
            self.register_modules(
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
                image_normalizer=image_normalizer,
                image_noising_scheduler=image_noising_scheduler,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                scheduler=scheduler,
                vae=vae,
            )
    
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器，使用 VAE 的缩放因子
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
        # 从 StableDiffusionPipeline 复制的编码提示函数
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
        ):
            # 警告用户该函数已弃用，未来版本将被移除
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 函数以编码提示
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                **kwargs,
            )
    
            # 连接提示嵌入以便于向后兼容
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回连接后的提示嵌入
            return prompt_embeds
    
        # 编码图像的函数
        def _encode_image(
            self,
            image,
            device,
            batch_size,
            num_images_per_prompt,
            do_classifier_free_guidance,
            noise_level,
            generator,
            image_embeds,
    ):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 检查输入的图像是否为 PIL 图像类型
        if isinstance(image, PIL.Image.Image):
            # 如果是 PIL 图像，则重复次数与批量大小相同
            repeat_by = batch_size
        else:
            # 否则，假设图像输入已经正确批处理，只需重复以匹配每个提示的图像数量
            #
            # 注意：这可能缺少一些边缘情况，比如已批处理和未批处理的 `image_embeds`。
            # 如果这些情况比较常见，需要仔细考虑输入的预期维度及其编码处理。
            repeat_by = num_images_per_prompt

        # 检查图像嵌入是否为 None
        if image_embeds is None:
            # 如果输入图像不是张量，则使用特征提取器将其转换为张量
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            # 将图像转移到指定设备，并转换为适当的数据类型
            image = image.to(device=device, dtype=dtype)
            # 使用图像编码器对图像进行编码，得到图像嵌入
            image_embeds = self.image_encoder(image).image_embeds

        # 对图像嵌入应用噪声图像嵌入处理
        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )

        # 为每个生成的提示复制图像嵌入，使用适合 mps 的方法
        image_embeds = image_embeds.unsqueeze(1)
        # 获取嵌入的批量大小、序列长度和最后一个维度
        bs_embed, seq_len, _ = image_embeds.shape
        # 根据 repeat_by 重复图像嵌入
        image_embeds = image_embeds.repeat(1, repeat_by, 1)
        # 重新调整图像嵌入的形状
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        # 挤出多余的维度
        image_embeds = image_embeds.squeeze(1)

        # 如果需要进行无分类器引导
        if do_classifier_free_guidance:
            # 创建与图像嵌入形状相同的零张量作为负提示嵌入
            negative_prompt_embeds = torch.zeros_like(image_embeds)

            # 对于无分类器引导，我们需要进行两次前向传播
            # 这里将无条件和文本嵌入拼接到一个批次中，以避免进行两次前向传播
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        # 返回处理后的图像嵌入
        return image_embeds

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制的函数
    def encode_prompt(
        self,
        # 输入的提示文本
        prompt,
        # 设备类型
        device,
        # 每个提示的图像数量
        num_images_per_prompt,
        # 是否进行无分类器引导
        do_classifier_free_guidance,
        # 可选的负提示文本
        negative_prompt=None,
        # 可选的提示嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的 LoRA 缩放因子
        lora_scale: Optional[float] = None,
        # 可选的剪切跳过参数
        clip_skip: Optional[int] = None,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制的部分
    # 解码潜在向量
    def decode_latents(self, latents):
        # 定义过时警告信息，提示用户该方法将在 1.0.0 中被移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 触发过时警告，告知用户替代方法
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
        # 根据缩放因子调整潜在向量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在向量，返回的第一个元素是图像数据
        image = self.vae.decode(latents, return_dict=False)[0]
        # 归一化图像数据到 [0, 1] 范围
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像数据从 GPU 转移到 CPU，调整维度顺序并转换为 float32 格式，返回为 NumPy 数组
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制的方法
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外的参数，因为并非所有调度器都有相同的签名
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将被忽略
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 应该在 [0, 1] 范围内
    
            # 检查调度器是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 创建额外参数字典
            extra_step_kwargs = {}
            # 如果接受 eta，添加到额外参数字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，添加到额外参数字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回额外参数字典
            return extra_step_kwargs
    
        # 检查输入参数
        def check_inputs(
            self,
            prompt,  # 输入的提示文本
            image,  # 输入的图像数据
            height,  # 图像高度
            width,  # 图像宽度
            callback_steps,  # 回调步骤
            noise_level,  # 噪声水平
            negative_prompt=None,  # 可选的负面提示文本
            prompt_embeds=None,  # 可选的提示嵌入
            negative_prompt_embeds=None,  # 可选的负面提示嵌入
            image_embeds=None,  # 可选的图像嵌入
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的部分
    # 准备潜在空间的张量，输入包括批量大小、通道数、图像高度和宽度等参数
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 根据输入参数计算潜在张量的形状
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且长度与批量大小匹配，若不匹配则抛出错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
    
        # 如果未提供潜在张量，则生成一个随机的潜在张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在张量，则将其移动到指定设备
            latents = latents.to(device)
    
        # 根据调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在张量
        return latents
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_unclip.StableUnCLIPPipeline.noise_image_embeddings 复制的函数
        def noise_image_embeddings(
            self,
            image_embeds: torch.Tensor,
            noise_level: int,
            noise: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
    ):
        """
        向图像嵌入添加噪声。噪声的量由 `noise_level` 输入控制。较高的
        `noise_level` 增加最终无噪声图像的方差。

        噪声通过两种方式应用：
        1. 噪声调度直接应用于嵌入。
        2. 一个正弦时间嵌入向量附加到输出中。

        在这两种情况下，噪声的量由相同的 `noise_level` 控制。

        在应用噪声之前，嵌入会被归一化，在应用噪声之后再进行反归一化。
        """
        # 如果未提供噪声，则生成与图像嵌入形状相同的随机噪声
        if noise is None:
            noise = randn_tensor(
                # 生成随机噪声的形状与图像嵌入相同
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )

        # 创建一个与图像嵌入数量相同的噪声水平张量
        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        # 将图像归一化器移动到与图像嵌入相同的设备
        self.image_normalizer.to(image_embeds.device)
        # 对图像嵌入进行归一化处理
        image_embeds = self.image_normalizer.scale(image_embeds)

        # 向图像嵌入添加噪声，使用噪声调度器
        image_embeds = self.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

        # 对添加噪声后的图像嵌入进行反归一化处理
        image_embeds = self.image_normalizer.unscale(image_embeds)

        # 获取时间步嵌入，并将其应用于图像嵌入
        noise_level = get_timestep_embedding(
            # 传入时间步、嵌入维度等参数来生成时间步嵌入
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        # `get_timestep_embeddings` 不包含任何权重，始终返回 f32 张量，
        # 但我们可能在 fp16 中运行，因此需要在这里转换类型。
        # 可能有更好的封装方式。
        noise_level = noise_level.to(image_embeds.dtype)

        # 将时间步嵌入与图像嵌入在维度1上进行拼接
        image_embeds = torch.cat((image_embeds, noise_level), 1)

        # 返回处理后的图像嵌入
        return image_embeds

    # 不计算梯度以节省内存和加快计算
    @torch.no_grad()
    # 用于替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 接收的图像，可以是张量或 PIL 图像
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        # 用户提供的提示文本，可以是字符串或字符串列表
        prompt: Union[str, List[str]] = None,
        # 生成图像的高度，可选
        height: Optional[int] = None,
        # 生成图像的宽度，可选
        width: Optional[int] = None,
        # 推理步骤的数量，默认为20
        num_inference_steps: int = 20,
        # 引导缩放因子，控制生成图像与提示的匹配程度
        guidance_scale: float = 10,
        # 可选的负面提示文本，可以是字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为1
        num_images_per_prompt: Optional[int] = 1,
        # 调整生成图像的随机性参数，默认为0.0
        eta: float = 0.0,
        # 可选的随机数生成器
        generator: Optional[torch.Generator] = None,
        # 可选的潜在张量，通常用于输入
        latents: Optional[torch.Tensor] = None,
        # 可选的提示嵌入张量
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负面提示嵌入张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型，可选，默认为"pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式的输出，默认为True
        return_dict: bool = True,
        # 可选的回调函数，在特定步骤调用
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调函数调用的步数间隔
        callback_steps: int = 1,
        # 可选的交叉注意力参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 噪声水平，控制噪声的强度
        noise_level: int = 0,
        # 可选的图像嵌入张量
        image_embeds: Optional[torch.Tensor] = None,
        # 可选的跳过的剪辑步骤
        clip_skip: Optional[int] = None,
```