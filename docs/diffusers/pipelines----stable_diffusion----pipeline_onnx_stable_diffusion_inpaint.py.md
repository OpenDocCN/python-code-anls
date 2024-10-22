# `.\diffusers\pipelines\stable_diffusion\pipeline_onnx_stable_diffusion_inpaint.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件按“现状”基础进行分发，
# 不附有任何明示或暗示的担保或条件。
# 请参见许可证以获取管理权限的特定语言和
# 限制条款。

import inspect  # 导入 inspect 模块，用于检查对象的属性和方法
from typing import Callable, List, Optional, Union  # 导入类型注释，便于类型检查

import numpy as np  # 导入 NumPy 库，用于数组和矩阵操作
import PIL.Image  # 导入 PIL 图像处理库
import torch  # 导入 PyTorch 库，用于深度学习
from transformers import CLIPImageProcessor, CLIPTokenizer  # 导入 Transformers 库中的图像处理和标记器

from ...configuration_utils import FrozenDict  # 导入 FrozenDict，用于不可变字典
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler  # 导入调度器类
from ...utils import PIL_INTERPOLATION, deprecate, logging  # 导入工具函数和日志模块
from ..onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel  # 导入 ONNX 相关工具
from ..pipeline_utils import DiffusionPipeline  # 导入扩散管道基类
from . import StableDiffusionPipelineOutput  # 导入稳定扩散管道输出类


logger = logging.get_logger(__name__)  # 创建一个记录器，使用当前模块名称进行日志记录

NUM_UNET_INPUT_CHANNELS = 9  # 定义 UNet 输入通道的数量
NUM_LATENT_CHANNELS = 4  # 定义潜在通道的数量


def prepare_mask_and_masked_image(image, mask, latents_shape):  # 定义准备掩模和掩模图像的函数
    # 将输入图像转换为 RGB 格式，并调整大小以适应潜在形状
    image = np.array(image.convert("RGB").resize((latents_shape[1] * 8, latents_shape[0] * 8)))
    # 调整数组形状以适配深度学习模型的输入要求
    image = image[None].transpose(0, 3, 1, 2)
    # 将图像数据类型转换为 float32，并归一化到 [-1, 1] 范围
    image = image.astype(np.float32) / 127.5 - 1.0

    # 将掩模图像转换为灰度并调整大小
    image_mask = np.array(mask.convert("L").resize((latents_shape[1] * 8, latents_shape[0] * 8)))
    # 应用掩模到图像，得到掩模图像
    masked_image = image * (image_mask < 127.5)

    # 调整掩模大小以匹配潜在形状，并转换为灰度格式
    mask = mask.resize((latents_shape[1], latents_shape[0]), PIL_INTERPOLATION["nearest"])
    mask = np.array(mask.convert("L"))
    # 将掩模数据类型转换为 float32，并归一化到 [0, 1] 范围
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]  # 添加维度以匹配模型输入要求
    # 将小于 0.5 的值设为 0
    mask[mask < 0.5] = 0
    # 将大于等于 0.5 的值设为 1
    mask[mask >= 0.5] = 1

    return mask, masked_image  # 返回处理后的掩模和掩模图像


class OnnxStableDiffusionInpaintPipeline(DiffusionPipeline):  # 定义用于图像修补的扩散管道类
    r"""
    使用稳定扩散进行文本引导的图像修补管道。*这是一个实验特性*。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档，以获取库为所有管道实现的通用方法
    （例如下载或保存，在特定设备上运行等）。
    # 文档字符串，定义类的参数和它们的类型
    Args:
        vae ([`AutoencoderKL`]):  # 变分自编码器模型，用于编码和解码图像及其潜在表示
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):  # 冻结的文本编码器，用于处理文本输入
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):  # 处理文本的标记器
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]):  # 条件 U-Net 架构，用于去噪编码的图像潜在表示
            Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):  # 调度器，与 `unet` 一起用于去噪图像潜在表示
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):  # 分类模块，评估生成图像是否可能被视为冒犯或有害
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):  # 从生成图像中提取特征的模型，用于 `safety_checker` 的输入
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # 定义多个模型的类型，用于保存各个组件的实例
    vae_encoder: OnnxRuntimeModel  # 编码器模型的类型
    vae_decoder: OnnxRuntimeModel  # 解码器模型的类型
    text_encoder: OnnxRuntimeModel  # 文本编码器模型的类型
    tokenizer: CLIPTokenizer  # 文本标记器的类型
    unet: OnnxRuntimeModel  # U-Net 模型的类型
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]  # 调度器的类型，可以是多种类型之一
    safety_checker: OnnxRuntimeModel  # 安全检查器模型的类型
    feature_extractor: CLIPImageProcessor  # 特征提取器模型的类型

    _optional_components = ["safety_checker", "feature_extractor"]  # 可选组件的列表
    _is_onnx = True  # 指示当前模型是否为 ONNX 格式

    # 构造函数，用于初始化类的实例
    def __init__(
        self,
        vae_encoder: OnnxRuntimeModel,  # 传入编码器模型实例
        vae_decoder: OnnxRuntimeModel,  # 传入解码器模型实例
        text_encoder: OnnxRuntimeModel,  # 传入文本编码器模型实例
        tokenizer: CLIPTokenizer,  # 传入文本标记器实例
        unet: OnnxRuntimeModel,  # 传入 U-Net 模型实例
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],  # 传入调度器实例
        safety_checker: OnnxRuntimeModel,  # 传入安全检查器模型实例
        feature_extractor: CLIPImageProcessor,  # 传入特征提取器模型实例
        requires_safety_checker: bool = True,  # 指示是否需要安全检查器的布尔参数
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion.OnnxStableDiffusionPipeline._encode_prompt
    # 编码提示的函数
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],  # 输入的提示，可以是字符串或字符串列表
        num_images_per_prompt: Optional[int],  # 每个提示生成的图像数量，默认为可选
        do_classifier_free_guidance: bool,  # 是否执行无分类器自由引导的布尔参数
        negative_prompt: Optional[str],  # 可选的负面提示
        prompt_embeds: Optional[np.ndarray] = None,  # 可选的提示嵌入，默认为 None
        negative_prompt_embeds: Optional[np.ndarray] = None,  # 可选的负面提示嵌入，默认为 None
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion.OnnxStableDiffusionPipeline.check_inputs
    # 定义一个检查输入参数的函数，确保所有输入都符合预期
        def check_inputs(
            self,  # 指向类实例的引用
            prompt: Union[str, List[str]],  # 输入的提示，类型为字符串或字符串列表
            height: Optional[int],  # 图像高度，类型为可选整数
            width: Optional[int],  # 图像宽度，类型为可选整数
            callback_steps: int,  # 回调步骤数，类型为整数
            negative_prompt: Optional[str] = None,  # 负提示，类型为可选字符串
            prompt_embeds: Optional[np.ndarray] = None,  # 提示的嵌入表示，类型为可选numpy数组
            negative_prompt_embeds: Optional[np.ndarray] = None,  # 负提示的嵌入表示，类型为可选numpy数组
        ):
            # 检查高度和宽度是否都能被8整除
            if height % 8 != 0 or width % 8 != 0:
                # 如果不能整除，抛出值错误
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    
            # 检查回调步骤是否为正整数
            if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
            ):
                # 如果不是正整数，抛出值错误
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )
    
            # 检查提示和提示嵌入是否同时存在
            if prompt is not None and prompt_embeds is not None:
                # 如果两者都存在，抛出值错误
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查提示和提示嵌入是否同时为None
            elif prompt is None and prompt_embeds is None:
                # 如果都是None，抛出值错误
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查提示的类型是否为字符串或列表
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                # 如果不是，抛出值错误
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查负提示和负提示嵌入是否同时存在
            if negative_prompt is not None and negative_prompt_embeds is not None:
                # 如果同时存在，抛出值错误
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查提示嵌入和负提示嵌入的形状是否一致
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    # 如果形状不一致，抛出值错误
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
        # 在不计算梯度的情况下进行操作，节省内存和计算资源
        @torch.no_grad()
    # 定义一个可调用的类方法，用于处理图像生成
        def __call__(
            self,
            # 用户输入的提示，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 输入的图像，类型为 PIL.Image.Image
            image: PIL.Image.Image,
            # 掩模图像，类型为 PIL.Image.Image
            mask_image: PIL.Image.Image,
            # 输出图像的高度，默认为 512
            height: Optional[int] = 512,
            # 输出图像的宽度，默认为 512
            width: Optional[int] = 512,
            # 推理步骤的数量，默认为 50
            num_inference_steps: int = 50,
            # 指导尺度，默认为 7.5
            guidance_scale: float = 7.5,
            # 负提示，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 噪声的 eta 值，默认为 0.0
            eta: float = 0.0,
            # 随机数生成器，默认为 None
            generator: Optional[np.random.RandomState] = None,
            # 潜在表示，默认为 None
            latents: Optional[np.ndarray] = None,
            # 提示嵌入，默认为 None
            prompt_embeds: Optional[np.ndarray] = None,
            # 负提示嵌入，默认为 None
            negative_prompt_embeds: Optional[np.ndarray] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为 True
            return_dict: bool = True,
            # 回调函数，默认为 None
            callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
            # 回调步骤的间隔，默认为 1
            callback_steps: int = 1,
```