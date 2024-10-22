# `.\diffusers\pipelines\stable_diffusion\pipeline_onnx_stable_diffusion_img2img.py`

```py
# 版权信息，表示该代码的所有权归 HuggingFace 团队所有
# 许可信息，表明该文件在 Apache 2.0 许可下分发
# 除非遵守该许可，否则不得使用此文件
# 提供许可的获取地址
# 除非适用法律或书面同意，否则按照 "按现状" 基础分发软件，没有任何明示或暗示的担保
# 详细信息见许可中关于权限和限制的部分

import inspect  # 导入 inspect 模块，用于获取对象的信息
from typing import Callable, List, Optional, Union  # 导入类型提示，定义函数参数和返回值类型

import numpy as np  # 导入 numpy，用于数组和矩阵操作
import PIL.Image  # 导入 PIL.Image，用于图像处理
import torch  # 导入 PyTorch，支持深度学习计算
from transformers import CLIPImageProcessor, CLIPTokenizer  # 从 transformers 库导入 CLIP 图像处理器和分词器

from ...configuration_utils import FrozenDict  # 从配置工具导入 FrozenDict，用于不可变字典
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler  # 导入调度器类，用于模型训练调度
from ...utils import PIL_INTERPOLATION, deprecate, logging  # 导入工具类，处理 PIL 插值、弃用警告和日志记录
from ..onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel  # 从 ONNX 工具导入转换类型和模型类
from ..pipeline_utils import DiffusionPipeline  # 导入 DiffusionPipeline，基础管道类
from . import StableDiffusionPipelineOutput  # 导入稳定扩散管道的输出类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess 中复制的 preprocess 函数，调整尺寸从 8 变为 64
def preprocess(image):
    # 弃用消息，通知用户该方法将在未来版本中删除
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    # 触发弃用警告，提醒用户使用替代方法
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    # 如果输入是 PyTorch 张量，则直接返回
    if isinstance(image, torch.Tensor):
        return image
    # 如果输入是 PIL 图像，将其封装到列表中
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    # 如果输入的第一个元素是 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽度和高度
        w, h = image[0].size
        # 将宽和高调整为64的整数倍
        w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64

        # 调整图像大小并转换为 NumPy 数组
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        # 沿着第0维连接图像数组
        image = np.concatenate(image, axis=0)
        # 将图像数据类型转换为浮点型并归一化到 [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组的维度顺序
        image = image.transpose(0, 3, 1, 2)
        # 将图像数据缩放到 [-1, 1]
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果输入的第一个元素是 PyTorch 张量
    elif isinstance(image[0], torch.Tensor):
        # 沿着第0维连接多个张量
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image

# 定义一个用于文本引导的图像到图像生成的管道类，使用稳定扩散模型
class OnnxStableDiffusionImg2ImgPipeline(DiffusionPipeline):
    r"""
    用于文本引导的图像到图像生成的管道，基于稳定扩散模型。

    该模型继承自 [`DiffusionPipeline`]。查看超类文档，了解库为所有管道实现的通用方法
    （例如下载或保存、在特定设备上运行等）。
    # 参数说明
    Args:
        vae ([`AutoencoderKL`]):  # 变分自编码器模型，用于将图像编码和解码为潜在表示。
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):  # 冻结的文本编码器，Stable Diffusion 使用 CLIP 的文本部分。
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)，具体是
            [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
        tokenizer (`CLIPTokenizer`):  # CLIPTokenizer 类的分词器。
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]):  # 条件 U-Net 结构，用于去噪编码的图像潜在表示。
            Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):  # 与 `unet` 结合使用的调度器，用于去噪编码的图像潜在表示。
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):  # 分类模块，估计生成的图像是否可能被视为冒犯或有害。
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):  # 从生成的图像中提取特征，以作为 `safety_checker` 的输入。
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # 定义变量类型
    vae_encoder: OnnxRuntimeModel  # VAE 编码器的类型
    vae_decoder: OnnxRuntimeModel  # VAE 解码器的类型
    text_encoder: OnnxRuntimeModel  # 文本编码器的类型
    tokenizer: CLIPTokenizer  # 分词器的类型
    unet: OnnxRuntimeModel  # U-Net 模型的类型
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]  # 调度器的类型
    safety_checker: OnnxRuntimeModel  # 安全检查器的类型
    feature_extractor: CLIPImageProcessor  # 特征提取器的类型

    # 可选组件列表
    _optional_components = ["safety_checker", "feature_extractor"]  # 包含可选组件的列表
    _is_onnx = True  # 指示是否使用 ONNX 模型

    # 构造函数初始化各个组件
    def __init__(  # 初始化方法
        self,
        vae_encoder: OnnxRuntimeModel,  # 传入的 VAE 编码器
        vae_decoder: OnnxRuntimeModel,  # 传入的 VAE 解码器
        text_encoder: OnnxRuntimeModel,  # 传入的文本编码器
        tokenizer: CLIPTokenizer,  # 传入的分词器
        unet: OnnxRuntimeModel,  # 传入的 U-Net 模型
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],  # 传入的调度器
        safety_checker: OnnxRuntimeModel,  # 传入的安全检查器
        feature_extractor: CLIPImageProcessor,  # 传入的特征提取器
        requires_safety_checker: bool = True,  # 是否需要安全检查器的标志
    # 从 diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion.OnnxStableDiffusionPipeline._encode_prompt 复制
    def _encode_prompt(  # 编码提示的方法
        self,
        prompt: Union[str, List[str]],  # 提示文本，可以是字符串或字符串列表
        num_images_per_prompt: Optional[int],  # 每个提示生成的图像数量
        do_classifier_free_guidance: bool,  # 是否进行无分类器引导
        negative_prompt: Optional[str],  # 可选的负面提示文本
        prompt_embeds: Optional[np.ndarray] = None,  # 可选的提示嵌入
        negative_prompt_embeds: Optional[np.ndarray] = None,  # 可选的负面提示嵌入
    # 定义一个检查输入参数的函数
        def check_inputs(
            self,  # 类实例自身
            prompt: Union[str, List[str]],  # 提示信息，可以是字符串或字符串列表
            callback_steps: int,  # 回调步骤的整数值
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示，字符串或列表
            prompt_embeds: Optional[np.ndarray] = None,  # 可选的提示嵌入，NumPy 数组
            negative_prompt_embeds: Optional[np.ndarray] = None,  # 可选的负面提示嵌入，NumPy 数组
        ):
            # 检查回调步骤是否为 None 或非正整数
            if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
            ):
                # 如果回调步骤不符合要求，则抛出值错误
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )
    
            # 检查是否同时提供了提示和提示嵌入
            if prompt is not None and prompt_embeds is not None:
                # 如果同时提供了，则抛出值错误
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查提示和提示嵌入是否都未提供
            elif prompt is None and prompt_embeds is None:
                # 如果都未提供，则抛出值错误
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查提示类型是否为字符串或列表
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                # 如果类型不符合，则抛出值错误
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查是否同时提供了负面提示和负面提示嵌入
            if negative_prompt is not None and negative_prompt_embeds is not None:
                # 如果同时提供了，则抛出值错误
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查提示嵌入和负面提示嵌入的形状是否一致
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    # 如果形状不一致，则抛出值错误
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
        # 定义可调用的方法，处理提示和生成图像
        def __call__(
            self,  # 类实例自身
            prompt: Union[str, List[str]],  # 提示信息，可以是字符串或字符串列表
            image: Union[np.ndarray, PIL.Image.Image] = None,  # 可选的图像输入，可以是 NumPy 数组或 PIL 图像
            strength: float = 0.8,  # 图像强度的浮点值，默认为 0.8
            num_inference_steps: Optional[int] = 50,  # 可选的推理步骤数，默认为 50
            guidance_scale: Optional[float] = 7.5,  # 可选的引导尺度，默认为 7.5
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负面提示，字符串或列表
            num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量，默认为 1
            eta: Optional[float] = 0.0,  # 可选的 eta 值，默认为 0.0
            generator: Optional[np.random.RandomState] = None,  # 可选的随机数生成器
            prompt_embeds: Optional[np.ndarray] = None,  # 可选的提示嵌入，NumPy 数组
            negative_prompt_embeds: Optional[np.ndarray] = None,  # 可选的负面提示嵌入，NumPy 数组
            output_type: Optional[str] = "pil",  # 输出类型，默认为 'pil'
            return_dict: bool = True,  # 是否返回字典格式，默认为 True
            callback: Optional[Callable[[int, int, np.ndarray], None]] = None,  # 可选的回调函数
            callback_steps: int = 1,  # 回调步骤的整数值，默认为 1
```