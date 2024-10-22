# `.\diffusers\pipelines\deprecated\stable_diffusion_variants\pipeline_onnx_stable_diffusion_inpaint_legacy.py`

```py
# 导入inspect模块，用于获取当前的堆栈信息和函数参数
import inspect
# 导入类型提示相关的类型，包含可调用对象、列表、可选项和联合类型
from typing import Callable, List, Optional, Union

# 导入numpy库，用于数组操作和数值计算
import numpy as np
# 导入PIL库中的Image模块，用于图像处理
import PIL.Image
# 导入torch库，用于深度学习相关操作
import torch
# 从transformers库导入CLIP图像处理器和分词器
from transformers import CLIPImageProcessor, CLIPTokenizer

# 从配置工具导入FrozenDict，用于处理不可变字典
from ....configuration_utils import FrozenDict
# 从调度器导入不同类型的调度器
from ....schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
# 从工具模块导入弃用装饰器和日志记录功能
from ....utils import deprecate, logging
# 从onnx工具导入ORT到NumPy类型映射和OnnxRuntime模型
from ...onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
# 从管道工具导入DiffusionPipeline类
from ...pipeline_utils import DiffusionPipeline
# 从稳定扩散输出模块导入StableDiffusionPipelineOutput类
from ...stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

# 创建一个日志记录器，用于记录当前模块的日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义图像预处理函数
def preprocess(image):
    # 获取输入图像的宽度和高度
    w, h = image.size
    # 将宽度和高度调整为32的整数倍
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    # 按指定大小调整图像，使用LANCZOS重采样
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # 将图像转换为NumPy数组并归一化到[0, 1]区间
    image = np.array(image).astype(np.float32) / 255.0
    # 调整数组的维度，将通道维移到第二个位置
    image = image[None].transpose(0, 3, 1, 2)
    # 将数组值从[0, 1]范围缩放到[-1, 1]
    return 2.0 * image - 1.0

# 定义掩膜预处理函数
def preprocess_mask(mask, scale_factor=8):
    # 将掩膜图像转换为灰度模式
    mask = mask.convert("L")
    # 获取掩膜的宽度和高度
    w, h = mask.size
    # 将宽度和高度调整为32的整数倍
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    # 按比例缩放掩膜图像并使用最近邻重采样
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=PIL.Image.NEAREST)
    # 将掩膜转换为NumPy数组并归一化到[0, 1]区间
    mask = np.array(mask).astype(np.float32) / 255.0
    # 将掩膜数组复制四次以匹配通道数
    mask = np.tile(mask, (4, 1, 1))
    # 调整数组的维度
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    # 将掩膜值反转，将白色变为黑色，黑色保持不变
    mask = 1 - mask  # repaint white, keep black
    # 返回处理后的掩膜
    return mask

# 定义一个类，用于稳定扩散的图像修复管道，继承自DiffusionPipeline
class OnnxStableDiffusionInpaintPipelineLegacy(DiffusionPipeline):
    r"""
    使用稳定扩散进行文本引导图像修复的管道。此功能为*遗留功能*，用于ONNX管道，以
    提供与StableDiffusionInpaintPipelineLegacy的兼容性，未来可能会被移除。

    该模型继承自[`DiffusionPipeline`]。有关库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等），请查看超类文档。
    """
    # 参数说明部分，描述每个参数的用途
    Args:
        vae ([`AutoencoderKL`]):
            # 变分自编码器模型，用于对图像进行编码和解码，将图像转化为潜在表示
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            # 冻结的文本编码器，Stable Diffusion 使用 CLIP 的文本部分，具体为 clip-vit-large-patch14 变体
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            # 用于将文本转化为标记的 CLIPTokenizer 类
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): 
            # 条件 U-Net 架构，用于去噪编码后的图像潜在表示
            Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            # 与 unet 结合使用的调度器，用于去噪编码后的图像潜在表示，可以是 DDIMScheduler、LMSDiscreteScheduler 或 PNDMScheduler
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            # 分类模块，用于评估生成图像是否可能被视为冒犯或有害，详情请参考模型卡
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            # 从生成图像中提取特征的模型，用于作为安全检查器的输入
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # 定义可选组件，包含安全检查器和特征提取器
    _optional_components = ["safety_checker", "feature_extractor"]
    # 标记该模型为 ONNX 格式
    _is_onnx = True

    # 声明各个模型组件的类型
    vae_encoder: OnnxRuntimeModel
    vae_decoder: OnnxRuntimeModel
    text_encoder: OnnxRuntimeModel
    tokenizer: CLIPTokenizer
    unet: OnnxRuntimeModel
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
    safety_checker: OnnxRuntimeModel
    feature_extractor: CLIPImageProcessor

    # 构造函数，初始化模型组件
    def __init__(
        self,
        vae_encoder: OnnxRuntimeModel,
        vae_decoder: OnnxRuntimeModel,
        text_encoder: OnnxRuntimeModel,
        tokenizer: CLIPTokenizer,
        unet: OnnxRuntimeModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: OnnxRuntimeModel,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    # 从 diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion.OnnxStableDiffusionPipeline._encode_prompt 复制的函数
    # 用于编码输入提示的函数
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],  # 输入提示，可以是字符串或字符串列表
        num_images_per_prompt: Optional[int],  # 每个提示生成的图像数量
        do_classifier_free_guidance: bool,  # 是否使用无分类器引导
        negative_prompt: Optional[str],  # 可选的负面提示
        prompt_embeds: Optional[np.ndarray] = None,  # 输入提示的嵌入表示，可选
        negative_prompt_embeds: Optional[np.ndarray] = None,  # 负面提示的嵌入表示，可选
    # 检查输入有效性的函数
    def check_inputs(
        self,
        prompt,  # 输入提示
        callback_steps,  # 回调步骤
        negative_prompt=None,  # 可选的负面提示
        prompt_embeds=None,  # 输入提示的嵌入表示，可选
        negative_prompt_embeds=None,  # 负面提示的嵌入表示，可选
    # 函数定义结束，开始处理参数验证
        ):
            # 检查回调步骤是否为 None 或者为负值或非整数
            if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
            ):
                # 如果不符合条件，抛出 ValueError 异常
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )
    
            # 检查同时提供 prompt 和 prompt_embeds 的情况
            if prompt is not None and prompt_embeds is not None:
                # 如果两者都存在，抛出 ValueError 异常
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查是否同时未提供 prompt 和 prompt_embeds
            elif prompt is None and prompt_embeds is None:
                # 如果都未提供，抛出 ValueError 异常
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查 prompt 的类型
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                # 如果类型不正确，抛出 ValueError 异常
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查同时提供 negative_prompt 和 negative_prompt_embeds 的情况
            if negative_prompt is not None and negative_prompt_embeds is not None:
                # 如果两者都存在，抛出 ValueError 异常
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查 prompt_embeds 和 negative_prompt_embeds 是否都存在
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                # 检查这两个数组的形状是否一致
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    # 如果形状不一致，抛出 ValueError 异常
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
        # 定义 __call__ 方法，以便该类的实例可以被调用
        def __call__(
            # 定义 prompt 参数，支持字符串或字符串列表类型
            prompt: Union[str, List[str]],
            # 定义可选的图像输入
            image: Union[np.ndarray, PIL.Image.Image] = None,
            # 定义可选的掩膜图像输入
            mask_image: Union[np.ndarray, PIL.Image.Image] = None,
            # 定义强度参数，默认值为 0.8
            strength: float = 0.8,
            # 定义推理步骤数，默认值为 50
            num_inference_steps: Optional[int] = 50,
            # 定义引导比例，默认值为 7.5
            guidance_scale: Optional[float] = 7.5,
            # 定义可选的负向提示
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 定义每个提示生成的图像数量，默认值为 1
            num_images_per_prompt: Optional[int] = 1,
            # 定义 eta 参数，默认值为 0.0
            eta: Optional[float] = 0.0,
            # 定义可选的随机数生成器
            generator: Optional[np.random.RandomState] = None,
            # 定义可选的提示嵌入
            prompt_embeds: Optional[np.ndarray] = None,
            # 定义可选的负向提示嵌入
            negative_prompt_embeds: Optional[np.ndarray] = None,
            # 定义输出类型，默认值为 "pil"
            output_type: Optional[str] = "pil",
            # 定义是否返回字典，默认值为 True
            return_dict: bool = True,
            # 定义可选的回调函数
            callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
            # 定义回调步骤，默认值为 1
            callback_steps: int = 1,
```