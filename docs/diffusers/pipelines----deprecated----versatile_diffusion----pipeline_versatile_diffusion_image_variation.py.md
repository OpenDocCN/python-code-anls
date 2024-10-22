# `.\diffusers\pipelines\deprecated\versatile_diffusion\pipeline_versatile_diffusion_image_variation.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件
# 以“原样”基础分发，不提供任何形式的保证或条件，
# 明示或暗示。请参阅许可证以获取有关权限的特定语言
# 和限制。

import inspect  # 导入 inspect 模块以获取对象的内部信息
from typing import Callable, List, Optional, Union  # 从 typing 导入用于类型注释的各种类型

import numpy as np  # 导入 numpy 作为数值计算库
import PIL.Image  # 导入 PIL.Image 用于处理图像
import torch  # 导入 PyTorch 框架以进行深度学习
import torch.utils.checkpoint  # 导入 checkpoint 以进行内存优化的反向传播
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # 从 transformers 导入图像处理和视觉模型

from ....image_processor import VaeImageProcessor  # 从相对路径导入 VaeImageProcessor
from ....models import AutoencoderKL, UNet2DConditionModel  # 导入自动编码器和 UNet 模型
from ....schedulers import KarrasDiffusionSchedulers  # 导入 Karras Diffusion 调度器
from ....utils import deprecate, logging  # 导入 deprecate 和 logging 工具
from ....utils.torch_utils import randn_tensor  # 从 torch_utils 导入 randn_tensor 函数
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从 pipeline_utils 导入 DiffusionPipeline 和 ImagePipelineOutput

logger = logging.get_logger(__name__)  # 初始化日志记录器，使用当前模块名称

class VersatileDiffusionImageVariationPipeline(DiffusionPipeline):  # 定义一个用于图像变换的管道类，继承自 DiffusionPipeline
    r"""  # 开始文档字符串，描述此类的作用
    Pipeline for image variation using Versatile Diffusion.  # 声明这是一个用于图像变换的管道

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).  # 说明该模型继承自 DiffusionPipeline，并可查看其文档

    Parameters:  # 参数说明
        vqvae ([`VQModel`]):  # vqvae 参数，类型为 VQModel
            Vector-quantized (VQ) model to encode and decode images to and from latent representations.  # 描述 vqvae 的功能
        bert ([`LDMBertModel`]):  # bert 参数，类型为 LDMBertModel
            Text-encoder model based on [`~transformers.BERT`].  # 描述 bert 的功能
        tokenizer ([`~transformers.BertTokenizer`]):  # tokenizer 参数，类型为 BertTokenizer
            A `BertTokenizer` to tokenize text.  # 描述 tokenizer 的功能
        unet ([`UNet2DConditionModel`]):  # unet 参数，类型为 UNet2DConditionModel
            A `UNet2DConditionModel` to denoise the encoded image latents.  # 描述 unet 的功能
        scheduler ([`SchedulerMixin`]):  # scheduler 参数，类型为 SchedulerMixin
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].  # 描述 scheduler 的功能
    """

    model_cpu_offload_seq = "bert->unet->vqvae"  # 定义模型在 CPU 上的卸载顺序

    image_feature_extractor: CLIPImageProcessor  # 声明图像特征提取器的类型
    image_encoder: CLIPVisionModelWithProjection  # 声明图像编码器的类型
    image_unet: UNet2DConditionModel  # 声明 UNet 的类型
    vae: AutoencoderKL  # 声明变分自编码器的类型
    scheduler: KarrasDiffusionSchedulers  # 声明调度器的类型

    def __init__(  # 定义初始化方法
        self,  # 当前实例
        image_feature_extractor: CLIPImageProcessor,  # 图像特征提取器参数
        image_encoder: CLIPVisionModelWithProjection,  # 图像编码器参数
        image_unet: UNet2DConditionModel,  # UNet 参数
        vae: AutoencoderKL,  # 变分自编码器参数
        scheduler: KarrasDiffusionSchedulers,  # 调度器参数
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 注册模块，包括图像特征提取器、图像编码器、图像 UNet、VAE 和调度器
        self.register_modules(
            image_feature_extractor=image_feature_extractor,
            image_encoder=image_encoder,
            image_unet=image_unet,
            vae=vae,
            scheduler=scheduler,
        )
        # 计算 VAE 的缩放因子，基于输出通道的数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建 VAE 图像处理器实例，传入缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # 从 StableDiffusionPipeline 复制的解码潜在变量的方法
    def decode_latents(self, latents):
        # 警告消息，表明该方法已弃用
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用警告函数，通知用户该方法即将被移除
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 根据 VAE 配置缩放潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量，获取生成的图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 归一化图像数据，确保其在 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像数据转换为 float32 格式，并将维度顺序调整为 (批量, 高, 宽, 通道)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image

    # 从 StableDiffusionPipeline 复制的准备额外步骤参数的方法
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的参数，因为并非所有调度器的参数签名相同
        # eta 仅在 DDIMScheduler 中使用，其他调度器将忽略该参数
        # eta 在 DDIM 论文中的值应在 [0, 1] 范围内

        # 检查调度器步骤是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 初始化额外步骤参数字典
        extra_step_kwargs = {}
        if accepts_eta:
            # 如果接受 eta，则将其添加到额外参数中
            extra_step_kwargs["eta"] = eta

        # 检查调度器步骤是否接受生成器参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            # 如果接受生成器，则将其添加到额外参数中
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外步骤参数
        return extra_step_kwargs

    # 从 StableDiffusionImageVariationPipeline 复制的输入检查方法
    # 检查输入参数的有效性
    def check_inputs(self, image, height, width, callback_steps):
        # 确保 image 是 torch.Tensor、PIL.Image.Image 或者列表类型
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            # 如果类型不符合，抛出值错误
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )
    
        # 确保 height 和 width 是 8 的倍数
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    
        # 检查 callback_steps 是否为正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            # 如果不符合，抛出值错误
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 定义潜在张量的形状
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器列表的长度是否与 batch_size 匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果 latents 为 None，则随机生成潜在张量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 否则将 latents 转移到指定设备
                latents = latents.to(device)
    
            # 按调度器所需的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在张量
            return latents
    
        # 禁用梯度计算
        @torch.no_grad()
        def __call__(
            # 定义输入参数，包括图像、尺寸、推理步数等
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            callback_steps: int = 1,
            **kwargs,
```