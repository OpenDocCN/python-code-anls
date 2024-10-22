# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion_image_variation.py`

```py
# 版权声明，标识文件的版权所有者和相关条款
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件按“原样”提供，
# 不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解适用的权限和限制。
#
# 导入 inspect 模块，用于检查函数签名和源代码
import inspect
# 从 typing 模块导入类型提示所需的类
from typing import Callable, List, Optional, Union

# 导入 PIL.Image 模块，用于处理图像
import PIL.Image
# 导入 PyTorch 库
import torch
# 导入 version 模块用于处理版本信息
from packaging import version
# 导入 CLIP 相关的图像处理器和模型
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# 从相对路径导入 FrozenDict 配置类
from ...configuration_utils import FrozenDict
# 从相对路径导入图像处理器
from ...image_processor import VaeImageProcessor
# 从相对路径导入自动编码器和 UNet 模型
from ...models import AutoencoderKL, UNet2DConditionModel
# 从相对路径导入 Karras 扩散调度器
from ...schedulers import KarrasDiffusionSchedulers
# 从相对路径导入工具函数
from ...utils import deprecate, logging
# 从工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入扩散管道和稳定扩散混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从当前目录导入稳定扩散管道输出类
from . import StableDiffusionPipelineOutput
# 从当前目录导入安全检查器
from .safety_checker import StableDiffusionSafetyChecker

# 创建日志记录器，便于记录调试信息和警告
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个类，用于生成图像变体，继承自扩散管道和稳定扩散混合类
class StableDiffusionImageVariationPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    管道用于从输入图像生成图像变体，使用稳定扩散模型。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解所有管道的通用方法
    （下载、保存、在特定设备上运行等）。
    # 函数参数说明
    Args:
        vae ([`AutoencoderKL`]):  # 变分自编码器（VAE）模型，用于将图像编码为潜在表示，并从中解码图像。
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`] ):  # 冻结的 CLIP 图像编码器，具体为 clip-vit-large-patch14。
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_encoder ([`~transformers.CLIPTextModel`]):  # 冻结的文本编码器，具体为 clip-vit-large-patch14。
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):  # 用于对文本进行分词的 CLIP 分词器。
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):  # 用于去噪已编码图像潜在表示的 UNet 模型。
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):  # 用于与 UNet 结合使用以去噪已编码图像潜在表示的调度器，可以是 DDIMScheduler、LMSDiscreteScheduler 或 PNDMScheduler。
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):  # 分类模块，用于评估生成的图像是否可能被认为是冒犯性或有害的。
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):  # CLIP 图像处理器，用于从生成的图像中提取特征；作为安全检查器的输入。
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    # TODO: feature_extractor 是必需的，以便编码图像（如果它们是 PIL 格式），
    # 如果管道没有 feature_extractor，我们应该给出描述性消息。
    _optional_components = ["safety_checker"]  # 可选组件列表，包含安全检查器。
    model_cpu_offload_seq = "image_encoder->unet->vae"  # 模型在 CPU 卸载时的顺序。
    _exclude_from_cpu_offload = ["safety_checker"]  # 在 CPU 卸载时排除的组件，安全检查器不会被卸载。

    def __init__(  # 初始化方法，定义类的构造函数。
        self,
        vae: AutoencoderKL,  # 传入变分自编码器实例。
        image_encoder: CLIPVisionModelWithProjection,  # 传入图像编码器实例。
        unet: UNet2DConditionModel,  # 传入 UNet 实例。
        scheduler: KarrasDiffusionSchedulers,  # 传入调度器实例。
        safety_checker: StableDiffusionSafetyChecker,  # 传入安全检查器实例。
        feature_extractor: CLIPImageProcessor,  # 传入图像处理器实例。
        requires_safety_checker: bool = True,  # 是否需要安全检查器的标志，默认值为 True。
    # 定义一个私有方法用于编码图像，接收多个参数
        def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
            # 获取图像编码器参数的数据类型
            dtype = next(self.image_encoder.parameters()).dtype
    
            # 检查输入是否为张量，如果不是，则使用特征提取器处理图像
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
    
            # 将图像转移到指定设备并转换为所需数据类型
            image = image.to(device=device, dtype=dtype)
            # 通过图像编码器生成图像嵌入
            image_embeddings = self.image_encoder(image).image_embeds
            # 增加一个维度以便于后续处理
            image_embeddings = image_embeddings.unsqueeze(1)
    
            # 针对每个提示生成图像嵌入的副本，使用适合 MPS 的方法
            bs_embed, seq_len, _ = image_embeddings.shape
            # 重复图像嵌入以匹配每个提示生成的图像数量
            image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
            # 重新调整图像嵌入的形状
            image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
    
            # 如果需要无分类器引导，则创建零向量的负提示嵌入
            if do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros_like(image_embeddings)
    
                # 对于无分类器引导，我们需要进行两次前向传递
                # 这里将无条件和文本嵌入拼接到一个批次中，以避免两次前向传递
                image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
    
            # 返回最终的图像嵌入
            return image_embeddings
    
        # 从 StableDiffusionPipeline 复制的方法，用于运行安全检查器
        def run_safety_checker(self, image, device, dtype):
            # 如果安全检查器未定义，则将标记设为 None
            if self.safety_checker is None:
                has_nsfw_concept = None
            else:
                # 检查图像是否为张量，如果是则处理为 PIL 格式
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 使用特征提取器处理图像并转移到指定设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 运行安全检查器，返回处理后的图像和 NSFW 概念标记
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像及 NSFW 概念标记
            return image, has_nsfw_concept
    
        # 从 StableDiffusionPipeline 复制的方法，用于解码潜在变量
        def decode_latents(self, latents):
            # 显示弃用提示，告知用户该方法将在未来版本中移除
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 按照配置的缩放因子调整潜在变量
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在变量以生成图像
            image = self.vae.decode(latents, return_dict=False)[0]
            # 归一化图像数据并限制其值在 0 到 1 之间
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像转换为 float32 格式，以确保兼容性并避免显著开销
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回解码后的图像
            return image
    
        # 从 StableDiffusionPipeline 复制的方法，用于准备额外步骤的关键字参数
    # 准备额外参数以便于调度器步骤，因不同调度器的签名可能不同
    def prepare_extra_step_kwargs(self, generator, eta):
        # 检查调度器步骤是否接受 eta 参数，eta 仅在 DDIMScheduler 中使用
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # eta 应该在 [0, 1] 范围内
    
        # 判断调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外参数字典
        extra_step_kwargs = {}
        # 如果调度器接受 eta，添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
    
        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回额外参数字典
        return extra_step_kwargs
    
    # 检查输入的有效性，包括图像、高度、宽度和回调步数
    def check_inputs(self, image, height, width, callback_steps):
        # 确保图像类型为 torch.Tensor 或 PIL.Image.Image 或图像列表
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )
    
        # 确保高度和宽度都是8的倍数
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    
        # 确保回调步骤是正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 如果传入的生成器列表长度与批量大小不匹配，抛出异常
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果没有提供潜在变量，生成随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果提供了潜在变量，将其移动到指定设备
                latents = latents.to(device)
    
            # 根据调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 禁用梯度计算，以节省内存
        @torch.no_grad()
    # 定义一个可调用的方法，用于处理图像输入
        def __call__(
            self,
            # 输入图像，可以是单个 PIL 图片、图片列表或 PyTorch 张量
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
            # 目标高度，可选参数
            height: Optional[int] = None,
            # 目标宽度，可选参数
            width: Optional[int] = None,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 引导缩放因子，默认为 7.5
            guidance_scale: float = 7.5,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 噪声控制参数，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，默认为 None，可以是单个生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 预定义的潜在张量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 可选的回调函数，接收步骤、图像索引和张量
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调函数调用的步数，默认为 1
            callback_steps: int = 1,
```