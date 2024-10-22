# `.\diffusers\pipelines\controlnet\pipeline_controlnet_img2img.py`

```py
# 版权声明，表示该文件的所有权和使用条款
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache License 2.0 许可证授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件是按“现状”提供的，
# 不附带任何明示或暗示的担保或条件
# 查看许可证以获取有关权限和限制的具体信息

import inspect  # 导入 inspect 模块以获取对象的信息
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型注释工具

import numpy as np  # 导入 numpy 库，用于数值计算
import PIL.Image  # 导入 PIL 库的 Image 模块，用于图像处理
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数式神经网络模块

# 从 transformers 库导入 CLIP 相关的图像处理和模型类
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 导入回调相关的类和混合类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 导入图像处理器
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 导入加载器的混合类
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
# 导入模型相关的类
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
# 从 LoRA 模型导入调整文本编码器的函数
from ...models.lora import adjust_lora_scale_text_encoder
# 导入调度器
from ...schedulers import KarrasDiffusionSchedulers
# 导入实用工具函数
from ...utils import (
    USE_PEFT_BACKEND,  # 导入用于选择后端的常量
    deprecate,  # 导入用于标记过时功能的装饰器
    logging,  # 导入日志记录工具
    replace_example_docstring,  # 导入用于替换示例文档字符串的函数
    scale_lora_layers,  # 导入用于缩放 LoRA 层的函数
    unscale_lora_layers,  # 导入用于取消缩放 LoRA 层的函数
)
# 从 utils.torch_utils 导入与 PyTorch 相关的工具函数
from ...utils.torch_utils import is_compiled_module, randn_tensor
# 从 pipeline_utils 导入扩散管道和稳定扩散混合类
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# 从稳定扩散模块导入输出类
from ..stable_diffusion import StableDiffusionPipelineOutput
# 从稳定扩散安全检查器导入类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 从 multicontrolnet 导入多控制网模型
from .multicontrolnet import MultiControlNetModel

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串的初始部分
EXAMPLE_DOC_STRING = """
``` 
```py  # 开始示例文档字符串的结束部分
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
    # 示例代码说明
        Examples:
            ```py
            >>> # 安装所需的库
            >>> # !pip install opencv-python transformers accelerate
            >>> # 从 diffusers 库导入所需的类
            >>> from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
            >>> # 导入加载图像的工具
            >>> from diffusers.utils import load_image
            >>> # 导入 NumPy 库
            >>> import numpy as np
            >>> # 导入 PyTorch 库
            >>> import torch
    
            >>> # 导入 OpenCV 库
            >>> import cv2
            >>> # 导入 PIL 图像处理库
            >>> from PIL import Image
    
            >>> # 下载图像
            >>> image = load_image(
            ...     # 指定图像的 URL
            ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
            ... )
            >>> # 将图像转换为 NumPy 数组
            >>> np_image = np.array(image)
    
            >>> # 使用 Canny 边缘检测算法处理图像
            >>> np_image = cv2.Canny(np_image, 100, 200)
            >>> # 将处理后的图像增加一个维度
            >>> np_image = np_image[:, :, None]
            >>> # 复制通道以形成彩色图像
            >>> np_image = np.concatenate([np_image, np_image, np_image], axis=2)
            >>> # 从 NumPy 数组创建图像对象
            >>> canny_image = Image.fromarray(np_image)
    
            >>> # 加载控制网络和稳定扩散模型 v1-5
            >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
            >>> pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            ...     # 指定模型和控制网络，使用半精度浮点
            ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            ... )
    
            >>> # 使用更快的调度器加速扩散过程并优化内存
            >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            >>> # 启用模型的 CPU 离线加载
            >>> pipe.enable_model_cpu_offload()
    
            >>> # 生成图像
            >>> generator = torch.manual_seed(0)  # 设置随机种子以确保可重复性
            >>> image = pipe(
            ...     # 输入生成图像的提示
            ...     "futuristic-looking woman",
            ...     num_inference_steps=20,  # 设置推理步骤数
            ...     generator=generator,  # 使用指定的随机生成器
            ...     image=image,  # 使用初始图像
            ...     control_image=canny_image,  # 使用 Canny 图像作为控制图像
            ... ).images[0]  # 获取生成的图像
            ```py
# 该函数用于从编码器输出中提取潜在向量
def retrieve_latents(
    # 输入参数：编码器输出，类型为张量
    encoder_output: torch.Tensor, 
    # 可选参数：随机数生成器
    generator: Optional[torch.Generator] = None, 
    # 采样模式，默认为“sample”
    sample_mode: str = "sample"
):
    # 如果编码器输出具有潜在分布，并且采样模式为“sample”
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 返回潜在分布的样本
        return encoder_output.latent_dist.sample(generator)
    # 如果编码器输出具有潜在分布，并且采样模式为“argmax”
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回潜在分布的众数
        return encoder_output.latent_dist.mode()
    # 如果编码器输出具有潜在向量
    elif hasattr(encoder_output, "latents"):
        # 返回潜在向量
        return encoder_output.latents
    # 如果都不满足，抛出异常
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# 准备图像数据的函数
def prepare_image(image):
    # 如果输入是张量
    if isinstance(image, torch.Tensor):
        # 对单张图像进行批处理
        if image.ndim == 3:
            # 增加一个维度以形成批次
            image = image.unsqueeze(0)

        # 转换为32位浮点类型
        image = image.to(dtype=torch.float32)
    else:
        # 预处理图像
        # 如果输入是PIL图像或NumPy数组
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            # 将单张图像放入列表
            image = [image]

        # 如果输入是图像列表且元素为PIL图像
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # 将每张图像转换为RGB格式的NumPy数组，并增加一个维度
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            # 将所有图像沿第0维拼接
            image = np.concatenate(image, axis=0)
        # 如果输入是图像列表且元素为NumPy数组
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            # 将每个图像增加一个维度后拼接
            image = np.concatenate([i[None, :] for i in image], axis=0)

        # 调整维度顺序为(batch, channels, height, width)
        image = image.transpose(0, 3, 1, 2)
        # 转换为张量，归一化到[-1, 1]范围
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    # 返回处理后的图像
    return image


# 定义用于图像到图像生成的Stable Diffusion管道类
class StableDiffusionControlNetImg2ImgPipeline(
    # 继承多个基类
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    # 文档字符串，描述该管道的功能
    r"""
    使用Stable Diffusion和ControlNet指导进行图像到图像生成的管道。

    此模型继承自[`DiffusionPipeline`]。有关通用方法的实现，请查看超类文档（下载、保存、在特定设备上运行等）。

    该管道还继承了以下加载方法：
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 用于加载文本反转嵌入
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载LoRA权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存LoRA权重
        - [`~loaders.FromSingleFileMixin.from_single_file`] 用于加载`.ckpt`文件
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] 用于加载IP适配器
    # 参数定义部分
    Args:
        # 变分自编码器模型，用于将图像编码为潜在表示并解码
        vae ([`AutoencoderKL']):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        # 冻结的文本编码器模型，用于处理文本输入
        text_encoder ([`~transformers.CLIPTextModel']):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        # 文本分词器，用于将文本转换为可处理的格式
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        # 去噪声的 UNet 模型，用于处理编码后的图像潜在表示
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        # 控制网络，为 UNet 在去噪过程提供额外的条件
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        # 调度器，与 UNet 一起使用以去噪编码后的图像潜在表示
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        # 安全检查器模块，用于估计生成的图像是否可能被认为是攻击性或有害的
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        # 特征提取器，从生成的图像中提取特征，作为安全检查器的输入
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    # 定义 CPU 离线加载顺序，依次为文本编码器、UNet 和 VAE
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义可选组件列表，包含安全检查器和特征提取器
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # 定义从 CPU 离线加载中排除的组件，当前为安全检查器
    _exclude_from_cpu_offload = ["safety_checker"]
    # 定义需要回调的张量输入列表，包括潜在表示和文本嵌入
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # 初始化方法，接收多个模型和组件作为参数
    def __init__(
        # 变分自编码器
        vae: AutoencoderKL,
        # 文本编码器
        text_encoder: CLIPTextModel,
        # 文本分词器
        tokenizer: CLIPTokenizer,
        # UNet 模型
        unet: UNet2DConditionModel,
        # 控制网络，可以是单个或多个模型
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        # 调度器
        scheduler: KarrasDiffusionSchedulers,
        # 安全检查器
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器
        feature_extractor: CLIPImageProcessor,
        # 可选的图像编码器，默认为 None
        image_encoder: CLIPVisionModelWithProjection = None,
        # 是否需要安全检查器，默认为 True
        requires_safety_checker: bool = True,
    ):
        # 调用父类构造函数
        super().__init__()

        # 检查安全检查器的状态，若未定义且需要，则记录警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                # 输出警告信息，提醒用户关于安全检查器的使用和许可证要求
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查安全检查器是否定义，且特征提取器未定义，则抛出错误
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                # 提示用户在加载时确保定义特征提取器，否则可以禁用安全检查器
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 如果 controlnet 是列表或元组，则将其转换为 MultiControlNetModel
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        # 注册模块，设置模型组件
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 初始化 VAE 图像处理器
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        # 初始化控制图像处理器
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        # 将所需的安全检查器注册到配置中
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的 _encode_prompt 方法
    def _encode_prompt(
        self,
        # 提示信息
        prompt,
        # 设备类型
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
        # 可选的 LORA 缩放因子
        lora_scale: Optional[float] = None,
        # 其他可选参数
        **kwargs,
    ):
        # 定义一个关于 `_encode_prompt()` 函数已被弃用的警告消息，提示用户使用 `encode_prompt()` 代替
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 函数，记录 `_encode_prompt()` 的弃用信息，版本号为 "1.0.0"
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法生成提示嵌入元组，传入多个参数
        prompt_embeds_tuple = self.encode_prompt(
            # 提示文本
            prompt=prompt,
            # 指定设备
            device=device,
            # 每个提示生成的图像数量
            num_images_per_prompt=num_images_per_prompt,
            # 是否使用无分类器自由引导
            do_classifier_free_guidance=do_classifier_free_guidance,
            # 负提示文本
            negative_prompt=negative_prompt,
            # 提示嵌入
            prompt_embeds=prompt_embeds,
            # 负提示嵌入
            negative_prompt_embeds=negative_prompt_embeds,
            # Lora 的缩放因子
            lora_scale=lora_scale,
            # 其他关键字参数
            **kwargs,
        )

        # 将提示嵌入元组的两个部分连接起来，以便与之前的实现兼容
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回合并后的提示嵌入
        return prompt_embeds

    # 复制自 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 方法
    def encode_prompt(
        # 提示文本
        self,
        prompt,
        # 设备
        device,
        # 每个提示生成的图像数量
        num_images_per_prompt,
        # 是否使用无分类器自由引导
        do_classifier_free_guidance,
        # 可选的负提示文本
        negative_prompt=None,
        # 可选的提示嵌入
        prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的负提示嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 可选的 Lora 缩放因子
        lora_scale: Optional[float] = None,
        # 可选的跳过剪辑参数
        clip_skip: Optional[int] = None,
    # 复制自 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image 方法
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 获取图像编码器参数的数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果输入的图像不是张量，则使用特征提取器将其转换为张量
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将图像张量移动到指定设备并转换为正确的数据类型
        image = image.to(device=device, dtype=dtype)
        # 如果要求输出隐藏状态
        if output_hidden_states:
            # 通过图像编码器获取图像的隐藏状态，提取倒数第二层的隐藏状态
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            # 重复图像的隐藏状态，生成指定数量的图像
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            # 对于无条件图像，使用零填充张量并获取其隐藏状态
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            # 重复无条件图像的隐藏状态，生成指定数量的图像
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            # 返回图像和无条件图像的隐藏状态
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            # 如果不要求输出隐藏状态，直接获取图像嵌入
            image_embeds = self.image_encoder(image).image_embeds
            # 重复图像嵌入，生成指定数量的图像
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 创建一个与图像嵌入形状相同的零填充张量，作为无条件图像嵌入
            uncond_image_embeds = torch.zeros_like(image_embeds)

            # 返回图像嵌入和无条件图像嵌入
            return image_embeds, uncond_image_embeds

    # 复制自 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds 方法
    # 准备 IP 适配器图像嵌入的函数定义
        def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
        ):
            # 初始化图像嵌入列表
            image_embeds = []
            # 如果启用无分类器自由引导，初始化负图像嵌入列表
            if do_classifier_free_guidance:
                negative_image_embeds = []
            # 如果图像嵌入为 None，处理输入图像
            if ip_adapter_image_embeds is None:
                # 如果输入图像不是列表，转换为列表
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]
                # 检查输入图像数量是否与 IP 适配器数量匹配
                if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )
                # 遍历每个输入图像和对应的图像投影层
                for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
                ):
                    # 判断输出隐藏状态是否为 False
                    output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                    # 对单个图像进行编码，获取图像嵌入
                    single_image_embeds, single_negative_image_embeds = self.encode_image(
                        single_ip_adapter_image, device, 1, output_hidden_state
                    )
                    # 将单个图像嵌入添加到列表中
                    image_embeds.append(single_image_embeds[None, :])
                    # 如果启用无分类器自由引导，添加负图像嵌入
                    if do_classifier_free_guidance:
                        negative_image_embeds.append(single_negative_image_embeds[None, :])
            else:
                # 遍历已有的图像嵌入
                for single_image_embeds in ip_adapter_image_embeds:
                    # 如果启用无分类器自由引导，分离负图像嵌入
                    if do_classifier_free_guidance:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    # 添加单个图像嵌入到列表中
                    image_embeds.append(single_image_embeds)
    
            # 初始化最终的图像嵌入列表
            ip_adapter_image_embeds = []
            # 遍历图像嵌入，进行数量扩展
            for i, single_image_embeds in enumerate(image_embeds):
                # 根据每个提示的数量重复图像嵌入
                single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
                # 如果启用无分类器自由引导，重复负图像嵌入
                if do_classifier_free_guidance:
                    single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                    # 合并负图像嵌入和正图像嵌入
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
                # 将图像嵌入转移到指定设备
                single_image_embeds = single_image_embeds.to(device=device)
                # 将处理后的图像嵌入添加到最终列表
                ip_adapter_image_embeds.append(single_image_embeds)
    
            # 返回最终的 IP 适配器图像嵌入列表
            return ip_adapter_image_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
    # 定义运行安全检查器的方法，接受图像、设备和数据类型作为参数
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，则将 NSFW 概念设置为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入图像是张量，则进行后处理以转换为 PIL 图像
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入图像是 NumPy 数组，则将其转换为 PIL 图像
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理图像并转换为张量，转移到指定设备上
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 执行安全检查，返回处理后的图像和 NSFW 概念标志
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和 NSFW 概念标志
        return image, has_nsfw_concept

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的解码潜在变量的方法
    def decode_latents(self, latents):
        # 设置弃用警告信息，指示该方法将在 1.0.0 版本中移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # 调用弃用函数，记录弃用警告
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 使用 VAE 的缩放因子反向缩放潜在变量
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在变量，获取图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将图像像素值缩放到 [0, 1] 范围内
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float32 类型以兼容 bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 复制的准备额外步骤参数的方法
    def prepare_extra_step_kwargs(self, generator, eta):
        # 准备调度器步骤的额外参数，因为并非所有调度器的签名相同
        # eta（η）仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
        # eta 对应于 DDIM 论文中的 η，应在 [0, 1] 之间

        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 创建一个字典以存储额外的步骤参数
        extra_step_kwargs = {}
        # 如果调度器接受 eta，则将其添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果调度器接受 generator，则将其添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数
        return extra_step_kwargs

    # 定义输入检查的方法，接受多个参数
    def check_inputs(
        self,
        prompt,
        image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    # 从 diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline 复制的检查图像的方法
    # 检查输入图像和提示的有效性
        def check_image(self, image, prompt, prompt_embeds):
            # 判断输入是否为 PIL 图像
            image_is_pil = isinstance(image, PIL.Image.Image)
            # 判断输入是否为 PyTorch 张量
            image_is_tensor = isinstance(image, torch.Tensor)
            # 判断输入是否为 NumPy 数组
            image_is_np = isinstance(image, np.ndarray)
            # 判断输入是否为 PIL 图像列表
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            # 判断输入是否为 PyTorch 张量列表
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            # 判断输入是否为 NumPy 数组列表
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
            # 如果输入不符合任何有效类型，抛出类型错误
            if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
            ):
                raise TypeError(
                    f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
                )
    
            # 如果输入为 PIL 图像，批处理大小为 1
            if image_is_pil:
                image_batch_size = 1
            else:
                # 否则，批处理大小为输入的长度
                image_batch_size = len(image)
    
            # 如果提示存在且为字符串，批处理大小为 1
            if prompt is not None and isinstance(prompt, str):
                prompt_batch_size = 1
            # 如果提示为列表，批处理大小为列表长度
            elif prompt is not None and isinstance(prompt, list):
                prompt_batch_size = len(prompt)
            # 如果提示嵌入存在，批处理大小为嵌入的第一维长度
            elif prompt_embeds is not None:
                prompt_batch_size = prompt_embeds.shape[0]
    
            # 如果图像批处理大小不为 1，且与提示批处理大小不一致，抛出值错误
            if image_batch_size != 1 and image_batch_size != prompt_batch_size:
                raise ValueError(
                    f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
                )
    
        # 处理控制图像以适应模型输入
        def prepare_control_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            # 使用控制图像处理器预处理输入图像，并转换为指定数据类型
            image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            # 获取图像批处理大小
            image_batch_size = image.shape[0]
    
            # 如果图像批处理大小为 1，按批大小重复图像
            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # 图像批处理大小与提示批处理大小相同
                repeat_by = num_images_per_prompt
    
            # 根据重复因子扩展图像
            image = image.repeat_interleave(repeat_by, dim=0)
    
            # 将图像转移到指定设备和数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果启用无分类器引导，且未猜测模式，则重复图像
            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)
    
            # 返回处理后的图像
            return image
    
        # 从稳定扩散图像到图像管道获取时间步
    # 定义获取时间步长的函数，接收推理步骤数量、强度和设备
        def get_timesteps(self, num_inference_steps, strength, device):
            # 使用初始化时间步长计算原始时间步长
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            # 计算时间步长的起始索引，确保不小于0
            t_start = max(num_inference_steps - init_timestep, 0)
            # 根据起始索引获取调度器的时间步长
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            # 如果调度器有设置开始索引的方法，则调用它
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
            # 返回时间步长和剩余的推理步骤数量
            return timesteps, num_inference_steps - t_start
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img 中复制的属性
        @property
        def guidance_scale(self):
            # 返回引导比例
            return self._guidance_scale
    
        @property
        def clip_skip(self):
            # 返回剪辑跳过的值
            return self._clip_skip
    
        # 定义分类器自由引导的属性，基于引导比例
        @property
        def do_classifier_free_guidance(self):
            # 判断引导比例是否大于1以决定是否使用分类器自由引导
            return self._guidance_scale > 1
    
        @property
        def cross_attention_kwargs(self):
            # 返回交叉注意力参数
            return self._cross_attention_kwargs
    
        @property
        def num_timesteps(self):
            # 返回时间步长的数量
            return self._num_timesteps
    
        # 在不计算梯度的上下文中定义调用方法
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 接收提示字符串或字符串列表作为输入
            prompt: Union[str, List[str]] = None,
            # 接收图像输入
            image: PipelineImageInput = None,
            # 接收控制图像输入
            control_image: PipelineImageInput = None,
            # 接收图像高度
            height: Optional[int] = None,
            # 接收图像宽度
            width: Optional[int] = None,
            # 设置强度，默认值为0.8
            strength: float = 0.8,
            # 设置推理步骤数量，默认值为50
            num_inference_steps: int = 50,
            # 设置引导比例，默认值为7.5
            guidance_scale: float = 7.5,
            # 接收负提示字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认值为1
            num_images_per_prompt: Optional[int] = 1,
            # 设置eta值，默认值为0.0
            eta: float = 0.0,
            # 接收生成器，支持单个或多个
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 接收潜在变量
            latents: Optional[torch.Tensor] = None,
            # 接收提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 接收负提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 接收图像适配器输入
            ip_adapter_image: Optional[PipelineImageInput] = None,
            # 接收图像适配器嵌入列表
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            # 设置输出类型，默认值为"pil"
            output_type: Optional[str] = "pil",
            # 设置是否返回字典，默认值为True
            return_dict: bool = True,
            # 接收交叉注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 设置控制网络条件比例，默认值为0.8
            controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
            # 设置是否猜测模式，默认值为False
            guess_mode: bool = False,
            # 设置控制引导开始值，默认值为0.0
            control_guidance_start: Union[float, List[float]] = 0.0,
            # 设置控制引导结束值，默认值为1.0
            control_guidance_end: Union[float, List[float]] = 1.0,
            # 接收剪辑跳过的值
            clip_skip: Optional[int] = None,
            # 接收在步骤结束时的回调函数
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 设置在步骤结束时回调的张量输入，默认为["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 接收额外的关键字参数
            **kwargs,
```