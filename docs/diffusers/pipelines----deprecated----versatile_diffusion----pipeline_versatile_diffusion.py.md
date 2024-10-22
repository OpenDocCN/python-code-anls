# `.\diffusers\pipelines\deprecated\versatile_diffusion\pipeline_versatile_diffusion.py`

```py
# 导入检查模块，用于获取对象的成员信息
import inspect
# 从 typing 模块导入常用类型，方便类型注解
from typing import Callable, List, Optional, Union

# 导入 PIL 库中的 Image 模块，用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 transformers 库中导入用于处理和生成图像的模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

# 从本地模块导入模型
from ....models import AutoencoderKL, UNet2DConditionModel
# 从本地模块导入调度器
from ....schedulers import KarrasDiffusionSchedulers
# 从本地模块导入日志记录工具
from ....utils import logging
# 从本地模块导入扩散管道的工具
from ...pipeline_utils import DiffusionPipeline
# 从本地模块导入多种引导的扩散管道
from .pipeline_versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
from .pipeline_versatile_diffusion_image_variation import VersatileDiffusionImageVariationPipeline
from .pipeline_versatile_diffusion_text_to_image import VersatileDiffusionTextToImagePipeline

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个名为 VersatileDiffusionPipeline 的类，继承自 DiffusionPipeline
class VersatileDiffusionPipeline(DiffusionPipeline):
    r"""
    使用稳定扩散进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。有关所有管道的通用方法的文档（下载、保存、在特定设备上运行等），请查看超类文档。

    参数:
        vae ([`AutoencoderKL`]):
            用于编码和解码图像与潜在表示的变分自编码器（VAE）模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            一个 `CLIPTokenizer` 用于对文本进行分词。
        unet ([`UNet2DConditionModel`]):
            用于对编码后的图像潜在数据进行去噪的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于对编码的图像潜在数据进行去噪。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            分类模块，估计生成的图像是否可能被视为冒犯或有害。
            有关模型潜在危害的更多详细信息，请参阅 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            一个 `CLIPImageProcessor` 用于从生成的图像中提取特征；用于作为 `safety_checker` 的输入。
    """

    # 定义类属性，用于存储不同组件的实例
    tokenizer: CLIPTokenizer  # 用于文本分词的 CLIPTokenizer 实例
    image_feature_extractor: CLIPImageProcessor  # 用于从图像中提取特征的 CLIPImageProcessor 实例
    text_encoder: CLIPTextModel  # 文本编码器的 CLIPTextModel 实例
    image_encoder: CLIPVisionModel  # 图像编码器的 CLIPVisionModel 实例
    image_unet: UNet2DConditionModel  # 用于图像去噪的 UNet2DConditionModel 实例
    text_unet: UNet2DConditionModel  # 用于文本去噪的 UNet2DConditionModel 实例
    vae: AutoencoderKL  # 变分自编码器的实例
    scheduler: KarrasDiffusionSchedulers  # 调度器的实例
    # 初始化类的构造函数，接收多个模型组件作为参数
        def __init__(
            self,
            tokenizer: CLIPTokenizer,  # 文本分词器，用于处理文本输入
            image_feature_extractor: CLIPImageProcessor,  # 图像特征提取器，用于处理图像输入
            text_encoder: CLIPTextModel,  # 文本编码器，将文本转换为向量表示
            image_encoder: CLIPVisionModel,  # 图像编码器，将图像转换为向量表示
            image_unet: UNet2DConditionModel,  # 图像生成的UNet模型
            text_unet: UNet2DConditionModel,  # 文本生成的UNet模型
            vae: AutoencoderKL,  # 变分自编码器，用于图像重建
            scheduler: KarrasDiffusionSchedulers,  # 调度器，控制生成过程的时间步
        ):
            # 调用父类的构造函数
            super().__init__()
    
            # 注册各个模块，使其可用
            self.register_modules(
                tokenizer=tokenizer,
                image_feature_extractor=image_feature_extractor,
                text_encoder=text_encoder,
                image_encoder=image_encoder,
                image_unet=image_unet,
                text_unet=text_unet,
                vae=vae,
                scheduler=scheduler,
            )
            # 计算变分自编码器的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    
        # 装饰器，禁止梯度计算，提高推理速度
        @torch.no_grad()
        def image_variation(
            self,
            image: Union[torch.Tensor, PIL.Image.Image],  # 输入图像，可以是张量或PIL图像
            height: Optional[int] = None,  # 可选的输出图像高度
            width: Optional[int] = None,  # 可选的输出图像宽度
            num_inference_steps: int = 50,  # 推理步骤的数量
            guidance_scale: float = 7.5,  # 引导比例，控制生成效果
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量
            eta: float = 0.0,  # 控制噪声的参数
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
            latents: Optional[torch.Tensor] = None,  # 先验潜在向量
            output_type: Optional[str] = "pil",  # 输出类型，默认为PIL图像
            return_dict: bool = True,  # 是否返回字典格式的结果
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 可选的回调函数
            callback_steps: int = 1,  # 回调的步数
        # 装饰器，禁止梯度计算，提高推理速度
        @torch.no_grad()
        def text_to_image(
            self,
            prompt: Union[str, List[str]],  # 输入提示，可以是单个字符串或字符串列表
            height: Optional[int] = None,  # 可选的输出图像高度
            width: Optional[int] = None,  # 可选的输出图像宽度
            num_inference_steps: int = 50,  # 推理步骤的数量
            guidance_scale: float = 7.5,  # 引导比例，控制生成效果
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量
            eta: float = 0.0,  # 控制噪声的参数
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
            latents: Optional[torch.Tensor] = None,  # 先验潜在向量
            output_type: Optional[str] = "pil",  # 输出类型，默认为PIL图像
            return_dict: bool = True,  # 是否返回字典格式的结果
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 可选的回调函数
            callback_steps: int = 1,  # 回调的步数
        # 装饰器，禁止梯度计算，提高推理速度
        @torch.no_grad()
        def dual_guided(
            self,
            prompt: Union[PIL.Image.Image, List[PIL.Image.Image]],  # 输入提示，可以是图像或图像列表
            image: Union[str, List[str]],  # 输入图像路径，可以是单个字符串或字符串列表
            text_to_image_strength: float = 0.5,  # 文本到图像的强度
            height: Optional[int] = None,  # 可选的输出图像高度
            width: Optional[int] = None,  # 可选的输出图像宽度
            num_inference_steps: int = 50,  # 推理步骤的数量
            guidance_scale: float = 7.5,  # 引导比例，控制生成效果
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量
            eta: float = 0.0,  # 控制噪声的参数
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
            latents: Optional[torch.Tensor] = None,  # 先验潜在向量
            output_type: Optional[str] = "pil",  # 输出类型，默认为PIL图像
            return_dict: bool = True,  # 是否返回字典格式的结果
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 可选的回调函数
            callback_steps: int = 1,  # 回调的步数
```