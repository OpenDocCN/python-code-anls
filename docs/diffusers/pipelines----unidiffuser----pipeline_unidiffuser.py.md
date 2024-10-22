# `.\diffusers\pipelines\unidiffuser\pipeline_unidiffuser.py`

```py
# 导入 inspect 模块，用于检查对象的内部结构和属性
import inspect
# 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from dataclasses import dataclass
# 从 typing 模块导入类型注释工具
from typing import Callable, List, Optional, Union

# 导入 numpy 库，用于数值计算和数组操作
import numpy as np
# 导入 PIL.Image，用于图像处理
import PIL.Image
# 导入 torch 库，用于深度学习和张量操作
import torch
# 从 transformers 库导入多个类，用于处理 CLIP 模型
from transformers import (
    CLIPImageProcessor,  # 处理图像的 CLIP 处理器
    CLIPTextModel,       # 处理文本的 CLIP 模型
    CLIPTokenizer,       # CLIP 模型的分词器
    CLIPVisionModelWithProjection,  # CLIP 视觉模型
    GPT2Tokenizer,       # GPT-2 模型的分词器
)

# 从相对路径导入 VaeImageProcessor 类，用于变分自编码器图像处理
from ...image_processor import VaeImageProcessor
# 从相对路径导入加载器混合类，用于加载不同模型
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 从相对路径导入自编码器类
from ...models import AutoencoderKL
# 从相对路径导入调整 Lora 的函数，用于文本编码器
from ...models.lora import adjust_lora_scale_text_encoder
# 从相对路径导入 Karras 扩散调度器类
from ...schedulers import KarrasDiffusionSchedulers
# 从相对路径导入工具函数和常量
from ...utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
# 从相对路径导入基础输出类
from ...utils.outputs import BaseOutput
# 从相对路径导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从相对路径导入扩散管道类
from ..pipeline_utils import DiffusionPipeline
# 从相对路径导入文本解码器模型类
from .modeling_text_decoder import UniDiffuserTextDecoder
# 从相对路径导入 UViT 模型类
from .modeling_uvit import UniDiffuserModel

# 创建一个日志记录器实例，用于记录该模块的日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个新的基类输出子类，用于联合图像-文本输出
@dataclass
class ImageTextPipelineOutput(BaseOutput):
    """
    联合图像-文本管道的输出类。

    参数：
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            长度为 `batch_size` 的去噪 PIL 图像列表或形状为 `(batch_size, height, width,
            num_channels)` 的 NumPy 数组。
        text (`List[str]` 或 `List[List[str]]`)
            长度为 `batch_size` 的生成文本字符串列表或外层列表长度为
            `batch_size` 的字符串列表。
    """

    # 可选的图像输出，可以是 PIL 图像列表或 NumPy 数组
    images: Optional[Union[List[PIL.Image.Image], np.ndarray]]
    # 可选的文本输出，可以是字符串列表或字符串列表的列表
    text: Optional[Union[List[str], List[List[str]]]]

# 定义联合扩散管道类，继承自 DiffusionPipeline
class UniDiffuserPipeline(DiffusionPipeline):
    r"""
    用于双模态图像-文本模型的管道，支持无条件文本和图像生成、文本条件图像生成、
    图像条件文本生成以及联合图像-文本生成。

    该模型继承自 [`DiffusionPipeline`]。查看超类文档以了解所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。
    # 定义参数的文档字符串，描述每个参数的作用
        Args:
            vae ([`AutoencoderKL`]):
                # 变分自编码器模型，用于将图像编码和解码为潜在表示
                Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations. This
                is part of the UniDiffuser image representation along with the CLIP vision encoding.
            text_encoder ([`CLIPTextModel`]):
                # 冻结的文本编码器，使用特定的 CLIP 模型进行文本编码
                Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
            image_encoder ([`CLIPVisionModel`]):
                # CLIP 视觉模型，用于将图像编码为其表示的一部分
                A [`~transformers.CLIPVisionModel`] to encode images as part of its image representation along with the VAE
                latent representation.
            image_processor ([`CLIPImageProcessor`]):
                # CLIP 图像处理器，用于在编码之前对图像进行预处理
                [`~transformers.CLIPImageProcessor`] to preprocess an image before CLIP encoding it with `image_encoder`.
            clip_tokenizer ([`CLIPTokenizer`]):
                # CLIP 分词器，用于在文本编码之前对提示进行分词
                 A [`~transformers.CLIPTokenizer`] to tokenize the prompt before encoding it with `text_encoder`.
            text_decoder ([`UniDiffuserTextDecoder`]):
                # 冻结的文本解码器，用于从 UniDiffuser 嵌入生成文本
                Frozen text decoder. This is a GPT-style model which is used to generate text from the UniDiffuser
                embedding.
            text_tokenizer ([`GPT2Tokenizer`]):
                # GPT2 分词器，用于文本生成的解码，与文本解码器一起使用
                A [`~transformers.GPT2Tokenizer`] to decode text for text generation; used along with the `text_decoder`.
            unet ([`UniDiffuserModel`]):
                # U-ViT 模型，具有 UNNet 风格的跳跃连接，用于去噪编码的图像潜在表示
                A [U-ViT](https://github.com/baofff/U-ViT) model with UNNet-style skip connections between transformer
                layers to denoise the encoded image latents.
            scheduler ([`SchedulerMixin`]):
                # 调度器，与 UNet 一起使用以去噪编码的图像和/或文本潜在表示
                A scheduler to be used in combination with `unet` to denoise the encoded image and/or text latents. The
                original UniDiffuser paper uses the [`DPMSolverMultistepScheduler`] scheduler.
        """
    
        # TODO: 支持启用模型 CPU 离线加载的组件的子模块移动
        model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae->text_decoder"
    
        # 初始化方法，接受多个模型组件作为参数
        def __init__(
            self,
            # 变分自编码器模型
            vae: AutoencoderKL,
            # 文本编码器模型
            text_encoder: CLIPTextModel,
            # 图像编码器模型
            image_encoder: CLIPVisionModelWithProjection,
            # CLIP 图像处理器
            clip_image_processor: CLIPImageProcessor,
            # CLIP 分词器
            clip_tokenizer: CLIPTokenizer,
            # 文本解码器模型
            text_decoder: UniDiffuserTextDecoder,
            # GPT2 分词器
            text_tokenizer: GPT2Tokenizer,
            # U-ViT 模型
            unet: UniDiffuserModel,
            # 调度器模型
            scheduler: KarrasDiffusionSchedulers,
    ):
        # 初始化父类
        super().__init__()

        # 检查文本编码器的隐藏层大小与文本解码器的前缀内维度是否相同
        if text_encoder.config.hidden_size != text_decoder.prefix_inner_dim:
            # 抛出值错误，提示二者不匹配
            raise ValueError(
                f"The text encoder hidden size and text decoder prefix inner dim must be the same, but"
                f" `text_encoder.config.hidden_size`: {text_encoder.config.hidden_size} and `text_decoder.prefix_inner_dim`: {text_decoder.prefix_inner_dim}"
            )

        # 注册模块，包括 VAE、文本编码器、图像编码器等
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            clip_image_processor=clip_image_processor,
            clip_tokenizer=clip_tokenizer,
            text_decoder=text_decoder,
            text_tokenizer=text_tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 VAE 图像处理器
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # 获取潜在空间的通道数
        self.num_channels_latents = vae.config.latent_channels
        # 获取文本编码器的最大序列长度
        self.text_encoder_seq_len = text_encoder.config.max_position_embeddings
        # 获取文本编码器的隐藏层大小
        self.text_encoder_hidden_size = text_encoder.config.hidden_size
        # 获取图像编码器的投影维度
        self.image_encoder_projection_dim = image_encoder.config.projection_dim
        # 获取 U-Net 的分辨率
        self.unet_resolution = unet.config.sample_size

        # 设置文本中间维度，默认为文本编码器的隐藏层大小
        self.text_intermediate_dim = self.text_encoder_hidden_size
        # 如果文本解码器的前缀隐藏维度不为 None，则使用该维度
        if self.text_decoder.prefix_hidden_dim is not None:
            self.text_intermediate_dim = self.text_decoder.prefix_hidden_dim

        # 初始化模式属性为 None
        self.mode = None

        # TODO: 处理安全检查？
        self.safety_checker = None

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外参数，因为并非所有调度器都有相同的签名
        # eta（η）仅用于 DDIMScheduler，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应该在 [0, 1] 之间

        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        # 如果接受 eta，添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回额外参数字典
        return extra_step_kwargs
    # 定义一个方法，用于根据输入推断生成任务的模式
    def _infer_mode(self, prompt, prompt_embeds, image, latents, prompt_latents, vae_latents, clip_latents):
        r"""
        从 `__call__` 的输入中推断生成任务（'mode'）。如果模式已手动设置，则使用设置的模式。
        """
        # 检查 prompt 或 prompt_embeds 是否可用
        prompt_available = (prompt is not None) or (prompt_embeds is not None)
        # 检查 image 是否可用
        image_available = image is not None
        # 判断输入是否可用（prompt 或 image 至少一个可用）
        input_available = prompt_available or image_available

        # 检查 prompt_latents 是否可用
        prompt_latents_available = prompt_latents is not None
        # 检查 vae_latents 是否可用
        vae_latents_available = vae_latents is not None
        # 检查 clip_latents 是否可用
        clip_latents_available = clip_latents is not None
        # 检查 latents 是否可用
        full_latents_available = latents is not None
        # 判断图像 latents 是否可用（同时有 vae_latents 和 clip_latents）
        image_latents_available = vae_latents_available and clip_latents_available
        # 判断所有单独的 latents 是否可用（有 prompt_latents 和图像 latents）
        all_indv_latents_available = prompt_latents_available and image_latents_available

        # 如果用户已设置模式，则优先使用该模式
        if self.mode is not None:
            mode = self.mode
        # 如果 prompt 可用，则设置模式为 "text2img"
        elif prompt_available:
            mode = "text2img"
        # 如果 image 可用，则设置模式为 "img2text"
        elif image_available:
            mode = "img2text"
        else:
            # 如果既没有提供 prompt 也没有提供 image，则根据 latents 的可用性推断模式
            if full_latents_available or all_indv_latents_available:
                mode = "joint"
            elif prompt_latents_available:
                mode = "text"
            elif image_latents_available:
                mode = "img"
            else:
                # 没有可用的输入或 latents
                mode = "joint"

        # 对模糊的情况给予警告
        if self.mode is None and prompt_available and image_available:
            logger.warning(
                f"You have supplied both a text prompt and image to the pipeline and mode has not been set manually,"
                f" defaulting to mode '{mode}'."
            )

        # 如果没有设置模式且没有输入可用
        if self.mode is None and not input_available:
            if vae_latents_available != clip_latents_available:
                # 只有一个 vae_latents 或 clip_latents 被提供
                logger.warning(
                    f"You have supplied exactly one of `vae_latents` and `clip_latents`, whereas either both or none"
                    f" are expected to be supplied. Defaulting to mode '{mode}'."
                )
            elif not prompt_latents_available and not vae_latents_available and not clip_latents_available:
                # 没有提供输入或 latents
                logger.warning(
                    f"No inputs or latents have been supplied, and mode has not been manually set,"
                    f" defaulting to mode '{mode}'."
                )

        # 返回推断得到的模式
        return mode

    # 从 diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_slicing 复制
    # 启用切片 VAE 解码
        def enable_vae_slicing(self):
            r""" 
            启用切片 VAE 解码。当该选项启用时，VAE 将输入张量拆分为多个切片，以
            分步计算解码。这有助于节省内存并允许更大的批量大小。
            """
            # 调用 VAE 的 enable_slicing 方法启用切片解码
            self.vae.enable_slicing()
    
        # 从 diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_slicing 复制
        def disable_vae_slicing(self):
            r""" 
            禁用切片 VAE 解码。如果之前启用了 `enable_vae_slicing`，该方法将恢复为
            一步计算解码。
            """
            # 调用 VAE 的 disable_slicing 方法禁用切片解码
            self.vae.disable_slicing()
    
        # 从 diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_tiling 复制
        def enable_vae_tiling(self):
            r""" 
            启用平铺 VAE 解码。当该选项启用时，VAE 将输入张量拆分为平铺以
            分步计算解码和编码。这有助于节省大量内存并允许处理更大图像。
            """
            # 调用 VAE 的 enable_tiling 方法启用平铺解码
            self.vae.enable_tiling()
    
        # 从 diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_tiling 复制
        def disable_vae_tiling(self):
            r""" 
            禁用平铺 VAE 解码。如果之前启用了 `enable_vae_tiling`，该方法将恢复为
            一步计算解码。
            """
            # 调用 VAE 的 disable_tiling 方法禁用平铺解码
            self.vae.disable_tiling()
    
        # 手动设置模式的函数
        def set_text_mode(self):
            r""" 
            手动将生成模式设置为无条件（"边际"）文本生成。
            """
            # 将模式属性设置为 "text"
            self.mode = "text"
    
        def set_image_mode(self):
            r""" 
            手动将生成模式设置为无条件（"边际"）图像生成。
            """
            # 将模式属性设置为 "img"
            self.mode = "img"
    
        def set_text_to_image_mode(self):
            r""" 
            手动将生成模式设置为基于文本的图像生成。
            """
            # 将模式属性设置为 "text2img"
            self.mode = "text2img"
    
        def set_image_to_text_mode(self):
            r""" 
            手动将生成模式设置为基于图像的文本生成。
            """
            # 将模式属性设置为 "img2text"
            self.mode = "img2text"
    
        def set_joint_mode(self):
            r""" 
            手动将生成模式设置为无条件联合图像-文本生成。
            """
            # 将模式属性设置为 "joint"
            self.mode = "joint"
    
        def reset_mode(self):
            r""" 
            移除手动设置的模式；调用此方法后，管道将从输入推断模式。
            """
            # 将模式属性重置为 None
            self.mode = None
    
        def _infer_batch_size(
            self,
            mode,
            prompt,
            prompt_embeds,
            image,
            num_images_per_prompt,
            num_prompts_per_image,
            latents,
            prompt_latents,
            vae_latents,
            clip_latents,
    # 定义文档字符串，说明该函数用于推断批处理大小和乘数
    ):
        r"""Infers the batch size and multiplier depending on mode and supplied arguments to `__call__`."""
        # 如果每个提示的图像数量未指定，则默认为1
        if num_images_per_prompt is None:
            num_images_per_prompt = 1
        # 如果每个图像的提示数量未指定，则默认为1
        if num_prompts_per_image is None:
            num_prompts_per_image = 1

        # 确保每个提示的图像数量为正整数
        assert num_images_per_prompt > 0, "num_images_per_prompt must be a positive integer"
        # 确保每个图像的提示数量为正整数
        assert num_prompts_per_image > 0, "num_prompts_per_image must be a positive integer"

        # 如果模式为“text2img”
        if mode in ["text2img"]:
            # 如果提供了提示且类型为字符串，则批处理大小为1
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            # 如果提供了提示且类型为列表，则批处理大小为提示数量
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                # 对于“text2img”，必须提供提示或提示嵌入
                batch_size = prompt_embeds.shape[0]
            # 乘数设为每个提示的图像数量
            multiplier = num_images_per_prompt
        # 如果模式为“img2text”
        elif mode in ["img2text"]:
            # 如果图像为PIL图像，则批处理大小为1
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            else:
                # 图像必须是PIL图像或torch.Tensor类型，不支持image_embeds
                batch_size = image.shape[0]
            # 乘数设为每个图像的提示数量
            multiplier = num_prompts_per_image
        # 如果模式为“img”
        elif mode in ["img"]:
            # 如果VAE潜变量存在，则批处理大小为VAE潜变量的数量
            if vae_latents is not None:
                batch_size = vae_latents.shape[0]
            # 如果CLIP潜变量存在，则批处理大小为CLIP潜变量的数量
            elif clip_latents is not None:
                batch_size = clip_latents.shape[0]
            else:
                # 否则，默认为1
                batch_size = 1
            # 乘数设为每个提示的图像数量
            multiplier = num_images_per_prompt
        # 如果模式为“text”
        elif mode in ["text"]:
            # 如果提示潜变量存在，则批处理大小为提示潜变量的数量
            if prompt_latents is not None:
                batch_size = prompt_latents.shape[0]
            else:
                # 否则，默认为1
                batch_size = 1
            # 乘数设为每个图像的提示数量
            multiplier = num_prompts_per_image
        # 如果模式为“joint”
        elif mode in ["joint"]:
            # 如果潜变量存在，则批处理大小为潜变量的数量
            if latents is not None:
                batch_size = latents.shape[0]
            elif prompt_latents is not None:
                batch_size = prompt_latents.shape[0]
            elif vae_latents is not None:
                batch_size = vae_latents.shape[0]
            elif clip_latents is not None:
                batch_size = clip_latents.shape[0]
            else:
                # 否则，默认为1
                batch_size = 1

            # 如果每个提示的图像数量与每个图像的提示数量相等，则乘数等于该数量
            if num_images_per_prompt == num_prompts_per_image:
                multiplier = num_images_per_prompt
            else:
                # 否则，乘数为二者中的较小值，并发出警告
                multiplier = min(num_images_per_prompt, num_prompts_per_image)
                logger.warning(
                    f"You are using mode `{mode}` and `num_images_per_prompt`: {num_images_per_prompt} and"
                    f" num_prompts_per_image: {num_prompts_per_image} are not equal. Using batch size equal to"
                    f" `min(num_images_per_prompt, num_prompts_per_image) = {batch_size}."
                )
        # 返回计算得出的批处理大小和乘数
        return batch_size, multiplier

    # 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt复制的代码
    # 定义编码提示的私有方法，接收多个参数用于处理提示信息
        def _encode_prompt(
            self,
            prompt,  # 提示文本
            device,  # 设备类型（如 CPU 或 GPU）
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=None,  # 负面提示文本（可选）
            prompt_embeds: Optional[torch.Tensor] = None,  # 提示的嵌入表示（可选）
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示的嵌入表示（可选）
            lora_scale: Optional[float] = None,  # LoRA 的缩放因子（可选）
            **kwargs,  # 其他关键字参数
        ):
            # 警告信息，表明该方法已弃用，并建议使用 `encode_prompt()` 代替
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用弃用警告函数，显示该方法的弃用信息
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用编码提示的方法，获取提示嵌入的元组
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,  # 提示文本
                device=device,  # 设备类型
                num_images_per_prompt=num_images_per_prompt,  # 图像数量
                do_classifier_free_guidance=do_classifier_free_guidance,  # 无分类器引导标志
                negative_prompt=negative_prompt,  # 负面提示
                prompt_embeds=prompt_embeds,  # 提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,  # 负面提示嵌入
                lora_scale=lora_scale,  # LoRA 缩放因子
                **kwargs,  # 其他参数
            )
    
            # 将提示嵌入元组中的两个部分进行拼接，以支持向后兼容
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回拼接后的提示嵌入
            return prompt_embeds
    
        # 从 StableDiffusionPipeline 的 encode_prompt 方法复制，替换了 tokenizer 为 clip_tokenizer
        def encode_prompt(
            self,
            prompt,  # 提示文本
            device,  # 设备类型
            num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance,  # 是否使用无分类器引导
            negative_prompt=None,  # 负面提示文本（可选）
            prompt_embeds: Optional[torch.Tensor] = None,  # 提示的嵌入表示（可选）
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示的嵌入表示（可选）
            lora_scale: Optional[float] = None,  # LoRA 的缩放因子（可选）
            clip_skip: Optional[int] = None,  # 可选的跳过步骤（可选）
        # 从 StableDiffusionInstructPix2PixPipeline 的 prepare_image_latents 方法修改而来
        # 添加 num_prompts_per_image 参数，从自动编码器的瞬时分布中采样
        def encode_image_vae_latents(
            self,
            image,  # 输入图像
            batch_size,  # 批量大小
            num_prompts_per_image,  # 每个图像的提示数量
            dtype,  # 数据类型
            device,  # 设备类型
            do_classifier_free_guidance,  # 是否使用无分类器引导
            generator=None,  # 随机数生成器（可选）
    # 定义一个函数，以便对图像进行处理和编码
        ):
            # 检查输入的图像类型是否为指定的几种类型之一
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                # 抛出错误，提示图像类型不正确
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )
    
            # 将图像转换到指定的设备和数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 计算有效批量大小
            batch_size = batch_size * num_prompts_per_image
            # 检查生成器列表的长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                # 抛出错误，提示生成器长度与请求的批量大小不符
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果生成器是列表，则逐个编码图像并生成潜在向量
            if isinstance(generator, list):
                image_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator=generator[i])
                    * self.vae.config.scaling_factor
                    for i in range(batch_size)
                ]
                # 将潜在向量按维度0拼接成一个大的张量
                image_latents = torch.cat(image_latents, dim=0)
            else:
                # 否则直接编码图像并生成潜在向量
                image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)
                # 按照 VAE 的缩放因子对潜在向量进行缩放
                image_latents = image_latents * self.vae.config.scaling_factor
    
            # 检查批量大小是否大于潜在向量的形状并且能够被整除
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # 如果条件满足，则构建弃用警告信息
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                # 记录弃用信息
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                # 计算每个提示所需的额外图像数量
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                # 将潜在向量按额外图像数量进行重复拼接
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            # 如果批量大小大于潜在向量的形状且不能被整除，抛出错误
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                # 将潜在向量按维度0拼接
                image_latents = torch.cat([image_latents], dim=0)
    
            # 如果启用了无分类器自由引导，则构造无条件潜在向量
            if do_classifier_free_guidance:
                uncond_image_latents = torch.zeros_like(image_latents)
                # 拼接无条件潜在向量以形成最终的潜在向量
                image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
    
            # 返回最终的图像潜在向量
            return image_latents
    
        # 定义编码图像 CLIP 潜在向量的函数
        def encode_image_clip_latents(
            self,
            image,
            batch_size,
            num_prompts_per_image,
            dtype,
            device,
            generator=None,
    ):
        # 将图像映射到 CLIP 嵌入。
        # 检查输入的 image 是否为有效类型：torch.Tensor、PIL.Image.Image 或 list
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            # 如果类型不匹配，抛出值错误并显示当前类型
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # 使用 clip_image_processor 预处理图像，并返回张量格式
        preprocessed_image = self.clip_image_processor.preprocess(
            image,
            return_tensors="pt",
        )
        # 将预处理后的图像移动到指定设备并设置数据类型
        preprocessed_image = preprocessed_image.to(device=device, dtype=dtype)

        # 根据提示数和每个图像的提示数计算批处理大小
        batch_size = batch_size * num_prompts_per_image
        # 如果生成器是列表，逐个处理每个预处理图像
        if isinstance(generator, list):
            image_latents = [
                # 使用 image_encoder 对每个预处理图像进行编码，获取图像嵌入
                self.image_encoder(**preprocessed_image[i : i + 1]).image_embeds for i in range(batch_size)
            ]
            # 将所有图像嵌入在第0维上拼接成一个张量
            image_latents = torch.cat(image_latents, dim=0)
        else:
            # 如果生成器不是列表，直接对预处理图像进行编码
            image_latents = self.image_encoder(**preprocessed_image).image_embeds

        # 如果批处理大小大于图像嵌入数量并且可以整除
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # 扩展 image_latents 以匹配批处理大小
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            # 发出弃用警告，提示用户更新代码
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            # 计算每个提示需要的额外图像数量
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            # 将图像嵌入重复以匹配批处理大小
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        # 如果批处理大小大于图像嵌入数量但不能整除，抛出值错误
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            # 将 image_latents 包装成一个张量
            image_latents = torch.cat([image_latents], dim=0)

        # 如果生成器是列表且其长度与批处理大小不匹配，抛出值错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 返回最终的图像嵌入张量
        return image_latents

    def prepare_text_latents(
        # 准备文本嵌入的函数定义，参数包括批处理大小、每个提示的图像数、序列长度、隐藏层大小、数据类型、设备、生成器和潜在变量
        self, batch_size, num_images_per_prompt, seq_len, hidden_size, dtype, device, generator, latents=None
    ):
        # 准备用于 CLIP 嵌入提示的潜在变量
        shape = (batch_size * num_images_per_prompt, seq_len, hidden_size)  # 定义潜在变量的形状
        # 检查生成器是否为列表且长度是否与批大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )  # 抛出错误，提示生成器长度与批大小不匹配

        # 如果潜在变量为 None，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # 生成随机潜在变量
        else:
            # 假设潜在变量具有形状 (B, L, D)
            latents = latents.repeat(num_images_per_prompt, 1, 1)  # 根据每个提示的图像数量重复潜在变量
            latents = latents.to(device=device, dtype=dtype)  # 将潜在变量转移到指定设备和数据类型

        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma  # 缩放潜在变量
        return latents  # 返回处理后的潜在变量

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 修改而来
    # 将 prepare_latents 重命名为 prepare_image_vae_latents，并添加 num_prompts_per_image 参数。
    def prepare_image_vae_latents(
        self,
        batch_size,
        num_prompts_per_image,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # 定义潜在变量的形状
        shape = (
            batch_size * num_prompts_per_image,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且长度是否与批大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )  # 抛出错误，提示生成器长度与批大小不匹配

        # 如果潜在变量为 None，则生成随机潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # 生成随机潜在变量
        else:
            # 假设潜在变量具有形状 (B, C, H, W)
            latents = latents.repeat(num_prompts_per_image, 1, 1, 1)  # 根据每个图像的提示数量重复潜在变量
            latents = latents.to(device=device, dtype=dtype)  # 将潜在变量转移到指定设备和数据类型

        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma  # 缩放潜在变量
        return latents  # 返回处理后的潜在变量

    def prepare_image_clip_latents(
        self, batch_size, num_prompts_per_image, clip_img_dim, dtype, device, generator, latents=None
    ):
        # 准备 CLIP 嵌入图像的潜在表示
        shape = (batch_size * num_prompts_per_image, 1, clip_img_dim)  # 定义潜在张量的形状
        # 检查生成器列表长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )  # 抛出值错误，说明生成器列表与批量大小不匹配

        # 如果潜在张量为 None，则生成随机潜在张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # 生成随机张量
        else:
            # 假设潜在张量的形状为 (B, L, D)
            latents = latents.repeat(num_prompts_per_image, 1, 1)  # 按提示数量重复潜在张量
            latents = latents.to(device=device, dtype=dtype)  # 将潜在张量转移到指定设备和数据类型

        # 按调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma  # 缩放潜在张量
        return latents  # 返回处理后的潜在张量

    def decode_text_latents(self, text_latents, device):
        # 生成输出标记列表和序列长度
        output_token_list, seq_lengths = self.text_decoder.generate_captions(
            text_latents, self.text_tokenizer.eos_token_id, device=device
        )  # 调用文本解码器生成文本输出
        output_list = output_token_list.cpu().numpy()  # 将输出转移到 CPU 并转换为 NumPy 数组
        # 解码输出标记，生成文本
        generated_text = [
            self.text_tokenizer.decode(output[: int(length)], skip_special_tokens=True)
            for output, length in zip(output_list, seq_lengths)
        ]  # 逐个解码每个输出
        return generated_text  # 返回生成的文本列表

    def _split(self, x, height, width):
        r"""
        将形状为 (B, C * H * W + clip_img_dim) 的扁平化嵌入 x 拆分为两个张量，形状为 (B, C, H, W)
        和 (B, 1, clip_img_dim)
        """
        batch_size = x.shape[0]  # 获取批量大小
        latent_height = height // self.vae_scale_factor  # 计算潜在高度
        latent_width = width // self.vae_scale_factor  # 计算潜在宽度
        img_vae_dim = self.num_channels_latents * latent_height * latent_width  # 计算 VAE 图像维度

        # 根据指定维度拆分张量
        img_vae, img_clip = x.split([img_vae_dim, self.image_encoder_projection_dim], dim=1)  # 拆分为 VAE 和 CLIP 图像

        img_vae = torch.reshape(img_vae, (batch_size, self.num_channels_latents, latent_height, latent_width))  # 重塑 VAE 图像
        img_clip = torch.reshape(img_clip, (batch_size, 1, self.image_encoder_projection_dim))  # 重塑 CLIP 图像
        return img_vae, img_clip  # 返回拆分后的两个张量

    def _combine(self, img_vae, img_clip):
        r"""
        将形状为 (B, C, H, W) 的潜在图像 img_vae 和形状为 (B, 1,
        clip_img_dim) 的 CLIP 嵌入图像 img_clip 组合成一个形状为 (B, C * H * W + clip_img_dim) 的张量。
        """
        img_vae = torch.reshape(img_vae, (img_vae.shape[0], -1))  # 将 VAE 图像重塑为一维张量
        img_clip = torch.reshape(img_clip, (img_clip.shape[0], -1))  # 将 CLIP 图像重塑为一维张量
        return torch.concat([img_vae, img_clip], dim=-1)  # 按最后一个维度连接两个张量
    # 将扁平化的嵌入 x 拆分为 (img_vae, img_clip, text)
    def _split_joint(self, x, height, width):
        r"""
        拆分形状为 (B, C * H * W + clip_img_dim + text_seq_len * text_dim) 的扁平化嵌入 x 为 (img_vae,
        img_clip, text)，其中 img_vae 形状为 (B, C, H, W)，img_clip 形状为 (B, 1, clip_img_dim)，text 形状为
        (B, text_seq_len, text_dim)。
        """
        # 获取输入 x 的批量大小
        batch_size = x.shape[0]
        # 计算潜在空间的高度
        latent_height = height // self.vae_scale_factor
        # 计算潜在空间的宽度
        latent_width = width // self.vae_scale_factor
        # 计算 img_vae 的维度
        img_vae_dim = self.num_channels_latents * latent_height * latent_width
        # 计算 text 的维度
        text_dim = self.text_encoder_seq_len * self.text_intermediate_dim

        # 根据指定的维度拆分 x
        img_vae, img_clip, text = x.split([img_vae_dim, self.image_encoder_projection_dim, text_dim], dim=1)

        # 将 img_vae 重新塑形为 (B, C, H, W)
        img_vae = torch.reshape(img_vae, (batch_size, self.num_channels_latents, latent_height, latent_width))
        # 将 img_clip 重新塑形为 (B, 1, clip_img_dim)
        img_clip = torch.reshape(img_clip, (batch_size, 1, self.image_encoder_projection_dim))
        # 将 text 重新塑形为 (B, text_seq_len, text_intermediate_dim)
        text = torch.reshape(text, (batch_size, self.text_encoder_seq_len, self.text_intermediate_dim))
        # 返回拆分后的 img_vae、img_clip 和 text
        return img_vae, img_clip, text

    # 将 img_vae、img_clip 和 text 组合成一个单一的嵌入 x
    def _combine_joint(self, img_vae, img_clip, text):
        r"""
        将形状为 (B, C, H, W) 的潜在图像 img_vae，形状为 (B, L_img,
        clip_img_dim) 的 CLIP 嵌入图像 img_clip，以及形状为 (B, L_text, text_dim) 的文本嵌入 text
        组合成形状为 (B, C * H * W + L_img * clip_img_dim + L_text * text_dim) 的单一嵌入 x。
        """
        # 将 img_vae 重塑为 (B, C * H * W)
        img_vae = torch.reshape(img_vae, (img_vae.shape[0], -1))
        # 将 img_clip 重塑为 (B, L_img * clip_img_dim)
        img_clip = torch.reshape(img_clip, (img_clip.shape[0], -1))
        # 将 text 重塑为 (B, L_text * text_dim)
        text = torch.reshape(text, (text.shape[0], -1))
        # 将 img_vae、img_clip 和 text 沿最后一个维度连接
        return torch.concat([img_vae, img_clip, text], dim=-1)

    # 获取噪声预测的核心函数
    def _get_noise_pred(
        self,
        mode,
        latents,
        t,
        prompt_embeds,
        img_vae,
        img_clip,
        max_timestep,
        data_type,
        guidance_scale,
        generator,
        device,
        height,
        width,
    # 检查潜在变量的形状是否符合预期
    def check_latents_shape(self, latents_name, latents, expected_shape):
        # 获取潜在变量的形状
        latents_shape = latents.shape
        # 计算预期维度，包括批量维度
        expected_num_dims = len(expected_shape) + 1  # expected dimensions plus the batch dimension
        # 生成预期形状的字符串
        expected_shape_str = ", ".join(str(dim) for dim in expected_shape)
        # 检查潜在变量维度数量是否符合预期
        if len(latents_shape) != expected_num_dims:
            raise ValueError(
                f"`{latents_name}` 应具有形状 (batch_size, {expected_shape_str})，但当前形状"
                f" {latents_shape} 有 {len(latents_shape)} 维度。"
            )
        # 遍历每个维度进行逐一检查
        for i in range(1, expected_num_dims):
            # 检查每个维度是否与预期匹配
            if latents_shape[i] != expected_shape[i - 1]:
                raise ValueError(
                    f"`{latents_name}` 应具有形状 (batch_size, {expected_shape_str})，但当前形状"
                    f" {latents_shape} 在维度 {i} 有 {latents_shape[i]} != {expected_shape[i - 1]}。"
                )
    # 定义输入检查方法
        def check_inputs(
            self,  # 当前实例对象
            mode,  # 模式参数，指示当前操作的类型
            prompt,  # 提示文本，用于生成内容
            image,  # 输入图像，可能用于处理或生成
            height,  # 输出图像的高度
            width,  # 输出图像的宽度
            callback_steps,  # 回调步骤的频率
            negative_prompt=None,  # 可选的负面提示，用于限制生成内容
            prompt_embeds=None,  # 可选的提示嵌入，用于直接输入嵌入向量
            negative_prompt_embeds=None,  # 可选的负面提示嵌入
            latents=None,  # 可选的潜在变量，用于生成过程
            prompt_latents=None,  # 可选的提示潜在变量
            vae_latents=None,  # 可选的变分自编码器潜在变量
            clip_latents=None,  # 可选的 CLIP 潜在变量
        @torch.no_grad()  # 禁用梯度计算，以节省内存和加快推理速度
        def __call__(  # 定义调用方法，使对象可调用
            self,  # 当前实例对象
            prompt: Optional[Union[str, List[str]]] = None,  # 可选的提示文本，可以是字符串或字符串列表
            image: Optional[Union[torch.Tensor, PIL.Image.Image]] = None,  # 可选的输入图像，可以是张量或图像对象
            height: Optional[int] = None,  # 可选的输出图像高度
            width: Optional[int] = None,  # 可选的输出图像宽度
            data_type: Optional[int] = 1,  # 数据类型，默认值为 1
            num_inference_steps: int = 50,  # 推理步骤的数量，默认 50
            guidance_scale: float = 8.0,  # 引导尺度，控制生成内容的自由度
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量，默认 1
            num_prompts_per_image: Optional[int] = 1,  # 每个图像的提示数量，默认 1
            eta: float = 0.0,  # 噪声参数，控制生成过程的随机性
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 可选的随机数生成器
            latents: Optional[torch.Tensor] = None,  # 可选的潜在变量
            prompt_latents: Optional[torch.Tensor] = None,  # 可选的提示潜在变量
            vae_latents: Optional[torch.Tensor] = None,  # 可选的变分自编码器潜在变量
            clip_latents: Optional[torch.Tensor] = None,  # 可选的 CLIP 潜在变量
            prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入
            output_type: Optional[str] = "pil",  # 输出类型，默认为 PIL 图像
            return_dict: bool = True,  # 是否返回字典格式的结果，默认是
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 可选的回调函数
            callback_steps: int = 1,  # 回调的步骤数，默认是 1
```