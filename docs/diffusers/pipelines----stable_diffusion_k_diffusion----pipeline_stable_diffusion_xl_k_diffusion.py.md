# `.\diffusers\pipelines\stable_diffusion_k_diffusion\pipeline_stable_diffusion_xl_k_diffusion.py`

```py
# 版权所有 2024 HuggingFace 团队。所有权利保留。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非符合许可证，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，按“原样”分发的软件不提供任何形式的保证或条件，
# 无论是明示或暗示的。有关许可证下的特定权限和限制，请参阅许可证。

import importlib  # 导入用于动态导入模块的库
import inspect  # 导入用于检查对象的库
from typing import List, Optional, Tuple, Union  # 导入类型注解

import torch  # 导入 PyTorch 库
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser  # 导入外部去噪模型
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras  # 导入采样器和获取函数
from transformers import (  # 导入 Transformers 库中的模型和分词器
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from ...image_processor import VaeImageProcessor  # 导入变分自编码器图像处理器
from ...loaders import (  # 导入不同加载器的混合器
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入模型类
from ...models.attention_processor import (  # 导入注意力处理器
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    XFormersAttnProcessor,
)
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 规模的函数
from ...schedulers import KarrasDiffusionSchedulers, LMSDiscreteScheduler  # 导入调度器
from ...utils import (  # 导入实用工具
    USE_PEFT_BACKEND,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合器
from ..stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput  # 导入管道输出类


logger = logging.get_logger(__name__)  # 初始化日志记录器，使用模块名

EXAMPLE_DOC_STRING = """  # 示例文档字符串，演示用法
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableDiffusionXLKDiffusionPipeline  # 导入扩散管道

        >>> pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained(  # 从预训练模型加载管道
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16  # 设置模型和数据类型
        ... )
        >>> pipe = pipe.to("cuda")  # 将管道移动到 GPU
        >>> pipe.set_scheduler("sample_dpmpp_2m_sde")  # 设置调度器

        >>> prompt = "a photo of an astronaut riding a horse on mars"  # 定义生成图像的提示
        >>> image = pipe(prompt).images[0]  # 生成图像并获取第一张
        ```py
"""


# 从 diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.ModelWrapper 复制的类
class ModelWrapper:  # 定义模型包装器类
    def __init__(self, model, alphas_cumprod):  # 初始化方法，接受模型和累积 alpha 值
        self.model = model  # 保存模型
        self.alphas_cumprod = alphas_cumprod  # 保存累积 alpha 值

    def apply_model(self, *args, **kwargs):  # 应用模型的方法
        if len(args) == 3:  # 如果传入三个位置参数
            encoder_hidden_states = args[-1]  # 获取最后一个参数作为编码器隐藏状态
            args = args[:2]  # 保留前两个参数
        if kwargs.get("cond", None) is not None:  # 如果关键字参数中有“cond”
            encoder_hidden_states = kwargs.pop("cond")  # 从关键字参数中提取并删除“cond”
        return self.model(*args, encoder_hidden_states=encoder_hidden_states, **kwargs).sample  # 调用模型并返回样本
# 定义一个类 StableDiffusionXLKDiffusionPipeline，继承自多个基类
class StableDiffusionXLKDiffusionPipeline(
    # 继承自 DiffusionPipeline 基类，提供扩散模型功能
    DiffusionPipeline,
    # 继承自 StableDiffusionMixin，提供稳定扩散特性
    StableDiffusionMixin,
    # 继承自 FromSingleFileMixin，支持从单个文件加载
    FromSingleFileMixin,
    # 继承自 StableDiffusionXLLoraLoaderMixin，支持加载 LoRA 权重
    StableDiffusionXLLoraLoaderMixin,
    # 继承自 TextualInversionLoaderMixin，支持加载文本反转嵌入
    TextualInversionLoaderMixin,
    # 继承自 IPAdapterMixin，支持加载 IP 适配器
    IPAdapterMixin,
):
    # 文档字符串，描述该管道的功能和用途
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL and k-diffusion.

    # 该模型继承自 `DiffusionPipeline`。请查看超类文档以了解库为所有管道实现的通用方法
    # （例如下载或保存、在特定设备上运行等）

    # 该管道还继承了以下加载方法：
        # [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        # [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载 `.ckpt` 文件
        # [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        # [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重
        # [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载 IP 适配器

    # 参数：
        # vae ([`AutoencoderKL`] ):
        # 变分自编码器 (VAE) 模型，用于编码和解码图像到潜在表示。
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        # text_encoder ([`CLIPTextModel`]):
        # 冻结的文本编码器。Stable Diffusion XL 使用 CLIP 的文本部分
        # [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)，具体为
        # [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        # text_encoder_2 ([` CLIPTextModelWithProjection`]):
        # 第二个冻结文本编码器。Stable Diffusion XL 使用 CLIP 的文本和池部分
        # [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection)，
        # 具体为
        # [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
        # 变体。
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        # tokenizer (`CLIPTokenizer`):
        # CLIP 的分词器
        # Tokenizer of class
        # [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # tokenizer_2 (`CLIPTokenizer`):
        # 第二个 CLIP 分词器
        # Second Tokenizer of class
        # [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # unet ([`UNet2DConditionModel`]): 条件 U-Net 架构，用于去噪编码后的图像潜变量。
        # Conditional U-Net architecture to denoise the encoded image latents.
        unet ([`UNet2DConditionModel`]): 
            Conditional U-Net architecture to denoise the encoded image latents.
        # scheduler ([`SchedulerMixin`]):
        # 用于与 `unet` 结合使用的调度器，以去噪编码的图像潜变量。可以是
        # [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 中的一个。
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        # force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
        # 是否将负提示嵌入强制设置为 0。还请参见
        # `stabilityai/stable-diffusion-xl-base-1-0` 的配置。
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
    """
    # 文档字符串，通常用于描述类或方法的功能

    # 定义模型中 CPU 卸载的顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    # 定义可选组件列表
    _optional_components = [
        "tokenizer",  # 词元化器
        "tokenizer_2",  # 第二个词元化器
        "text_encoder",  # 文本编码器
        "text_encoder_2",  # 第二个文本编码器
        "feature_extractor",  # 特征提取器
    ]

    # 初始化方法，接收多个参数
    def __init__(
        self,
        vae: AutoencoderKL,  # 变分自编码器
        text_encoder: CLIPTextModel,  # 文本编码器
        text_encoder_2: CLIPTextModelWithProjection,  # 带投影的第二文本编码器
        tokenizer: CLIPTokenizer,  # 词元化器
        tokenizer_2: CLIPTokenizer,  # 第二个词元化器
        unet: UNet2DConditionModel,  # UNet 模型
        scheduler: KarrasDiffusionSchedulers,  # 调度器
        force_zeros_for_empty_prompt: bool = True,  # 空提示时强制使用零
    ):
        super().__init__()  # 调用父类初始化方法

        # 从 LMS 获取正确的 sigma 值
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)  # 根据配置创建调度器
        # 注册模块到类中
        self.register_modules(
            vae=vae,  # 注册变分自编码器
            text_encoder=text_encoder,  # 注册文本编码器
            text_encoder_2=text_encoder_2,  # 注册第二文本编码器
            tokenizer=tokenizer,  # 注册词元化器
            tokenizer_2=tokenizer_2,  # 注册第二个词元化器
            unet=unet,  # 注册 UNet 模型
            scheduler=scheduler,  # 注册调度器
        )
        # 将配置中的参数注册到类中
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        # 计算 VAE 缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 VAE 图像处理器
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # 获取 UNet 的默认样本大小
        self.default_sample_size = self.unet.config.sample_size

        # 创建模型包装器
        model = ModelWrapper(unet, scheduler.alphas_cumprod)
        # 根据调度器配置选择 K Diffusion 模型
        if scheduler.config.prediction_type == "v_prediction":
            self.k_diffusion_model = CompVisVDenoiser(model)  # 使用 V 预测模型
        else:
            self.k_diffusion_model = CompVisDenoiser(model)  # 使用普通去噪模型

    # 从 StableDiffusionKDiffusionPipeline 复制的方法，设置调度器
    def set_scheduler(self, scheduler_type: str):
        library = importlib.import_module("k_diffusion")  # 动态导入 k_diffusion 库
        sampling = getattr(library, "sampling")  # 获取 sampling 模块
        try:
            # 根据调度器类型设置采样器
            self.sampler = getattr(sampling, scheduler_type)
        except Exception:
            valid_samplers = []  # 初始化有效采样器列表
            # 遍历 sampling 模块中的属性，查找有效采样器
            for s in dir(sampling):
                if "sample_" in s:
                    valid_samplers.append(s)

            # 抛出无效调度器类型的异常，并提供有效选择
            raise ValueError(f"Invalid scheduler type {scheduler_type}. Please choose one of {valid_samplers}.")

    # 从 StableDiffusionXLPipeline 复制的方法，编码提示
    # 定义一个编码提示的函数，接受多个参数以生成图像
        def encode_prompt(
            self,
            prompt: str,  # 主提示字符串
            prompt_2: Optional[str] = None,  # 可选的第二个提示字符串
            device: Optional[torch.device] = None,  # 可选的设备信息
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
            do_classifier_free_guidance: bool = True,  # 是否执行无分类器引导
            negative_prompt: Optional[str] = None,  # 可选的负提示字符串
            negative_prompt_2: Optional[str] = None,  # 可选的第二个负提示字符串
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负提示嵌入张量
            pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的池化提示嵌入张量
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的池化负提示嵌入张量
            lora_scale: Optional[float] = None,  # 可选的 LoRA 缩放因子
            clip_skip: Optional[int] = None,  # 可选的剪切跳过参数
        # 定义一个检查输入的函数，确保所有必要参数有效
        def check_inputs(
            self,
            prompt,  # 主提示
            prompt_2,  # 第二个提示
            height,  # 高度参数
            width,  # 宽度参数
            negative_prompt=None,  # 负提示
            negative_prompt_2=None,  # 第二个负提示
            prompt_embeds=None,  # 提示嵌入
            negative_prompt_embeds=None,  # 负提示嵌入
            pooled_prompt_embeds=None,  # 池化提示嵌入
            negative_pooled_prompt_embeds=None,  # 池化负提示嵌入
        # 准备潜在变量的函数，根据输入生成张量
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 根据批量大小、通道数、高度和宽度定义形状
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,  # 计算缩放后的高度
                int(width) // self.vae_scale_factor,  # 计算缩放后的宽度
            )
            # 检查生成器列表长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            # 如果潜在变量为空，则生成随机张量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:  # 否则，将潜在变量转移到指定设备
                latents = latents.to(device)
    
            return latents  # 返回生成或处理后的潜在变量
    
        # 从 StableDiffusionXLPipeline 复制的函数，用于获取添加时间ID
        def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
        ):
            # 创建添加时间ID列表，由原始大小、裁剪坐标和目标大小组成
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
            # 计算实际添加嵌入维度
            passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
            )
            # 获取期望的添加嵌入维度
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    
            # 检查实际和期望的维度是否匹配
            if expected_add_embed_dim != passed_add_embed_dim:
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )
    
            # 将添加时间ID转换为张量并返回
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
            return add_time_ids
    
        # 从 StableDiffusionXLPipeline 复制的函数，用于上采样 VAE
    # 定义一个方法，用于将 VAE 的数据类型提升到 float32
        def upcast_vae(self):
            # 获取当前 VAE 的数据类型
            dtype = self.vae.dtype
            # 将 VAE 转换为 float32 数据类型
            self.vae.to(dtype=torch.float32)
            # 检查当前使用的是否是 Torch 2.0 或 XFormers 的注意力处理器
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                    FusedAttnProcessor2_0,
                ),
            )
            # 如果使用的是 XFormers 或 Torch 2.0，注意力块不需要为 float32，这样可以节省大量内存
            if use_torch_2_0_or_xformers:
                # 将后量化卷积层转换为原始数据类型
                self.vae.post_quant_conv.to(dtype)
                # 将输入卷积层转换为原始数据类型
                self.vae.decoder.conv_in.to(dtype)
                # 将中间块转换为原始数据类型
                self.vae.decoder.mid_block.to(dtype)
    
        # 定义一个属性，用于获取引导缩放因子
        @property
        def guidance_scale(self):
            # 返回引导缩放因子的值
            return self._guidance_scale
    
        # 定义一个属性，用于获取剪切跳过的值
        @property
        def clip_skip(self):
            # 返回剪切跳过的值
            return self._clip_skip
    
        # 定义一个属性，指示是否执行无分类器引导
        # 该引导等同于 Imagen 论文中的指导权重 w
        @property
        def do_classifier_free_guidance(self):
            # 判断引导缩放因子是否大于 1 且时间条件投影维度为 None
            return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
        # 装饰器，表示在推理过程中不计算梯度
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义可调用方法，接收多种参数
        def __call__(
            # 接收一个或多个提示文本
            prompt: Union[str, List[str]] = None,
            # 接收第二个提示文本
            prompt_2: Optional[Union[str, List[str]]] = None,
            # 可选的图像高度
            height: Optional[int] = None,
            # 可选的图像宽度
            width: Optional[int] = None,
            # 指定推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 指定引导缩放因子的值，默认为 5.0
            guidance_scale: float = 5.0,
            # 可选的负提示文本
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 可选的第二个负提示文本
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 可选的随机数生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量张量
            latents: Optional[torch.Tensor] = None,
            # 可选的提示嵌入张量
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负提示嵌入张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的聚合提示嵌入张量
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负聚合提示嵌入张量
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 指示是否返回字典形式的结果，默认为 True
            return_dict: bool = True,
            # 可选的原始图像尺寸
            original_size: Optional[Tuple[int, int]] = None,
            # 默认的裁剪坐标，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 可选的目标尺寸
            target_size: Optional[Tuple[int, int]] = None,
            # 可选的负原始尺寸
            negative_original_size: Optional[Tuple[int, int]] = None,
            # 可选的负裁剪坐标
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 可选的负目标尺寸
            negative_target_size: Optional[Tuple[int, int]] = None,
            # 可选的 Karras sigma 使用标志
            use_karras_sigmas: Optional[bool] = False,
            # 可选的噪声采样器种子
            noise_sampler_seed: Optional[int] = None,
            # 可选的剪切跳过值
            clip_skip: Optional[int] = None,
```