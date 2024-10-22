# `.\diffusers\pipelines\t2i_adapter\pipeline_stable_diffusion_adapter.py`

```py
# 版权所有声明，包含版权信息和许可证详情
# Copyright 2024 TencentARC and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（"许可证"）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件在许可证下分发时按“原样”提供，
# 不附带任何明示或暗示的保证或条件。
# 有关许可证下的特定权限和限制，请参阅许可证。

import inspect  # 导入 inspect 模块以获取对象的内部信息
from dataclasses import dataclass  # 从 dataclasses 导入 dataclass 装饰器以简化类定义
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示

import numpy as np  # 导入 numpy 作为 np，用于数值计算
import PIL.Image  # 导入 PIL 的 Image 模块以处理图像
import torch  # 导入 PyTorch 库，用于深度学习
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer  # 从 transformers 导入相关模型和处理器

from ...image_processor import VaeImageProcessor  # 从本地模块导入 VaeImageProcessor
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入混合类用于加载
from ...models import AutoencoderKL, MultiAdapter, T2IAdapter, UNet2DConditionModel  # 导入模型
from ...models.lora import adjust_lora_scale_text_encoder  # 导入函数用于调整 LORA 模型的缩放
from ...schedulers import KarrasDiffusionSchedulers  # 导入调度器
from ...utils import (  # 从 utils 导入多个工具函数和常量
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    BaseOutput,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor  # 从工具模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 导入扩散管道和混合类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入安全检查器

@dataclass  # 使用 dataclass 装饰器简化数据类定义
class StableDiffusionAdapterPipelineOutput(BaseOutput):  # 定义 StableDiffusionAdapterPipelineOutput 类，继承自 BaseOutput
    """
    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            图像列表，包含去噪后的 PIL 图像，长度为 `batch_size`，或形状为 `(batch_size, height, width,
            num_channels)` 的 numpy 数组。PIL 图像或 numpy 数组表示扩散管道生成的去噪图像。
        nsfw_content_detected (`List[bool]`)
            布尔值标志列表，表示对应生成图像是否可能包含“不安全内容”
            （nsfw），如果无法执行安全检查则为 `None`。
    """

    images: Union[List[PIL.Image.Image], np.ndarray]  # 定义 images 属性，类型为图像列表或 numpy 数组
    nsfw_content_detected: Optional[List[bool]]  # 定义 nsfw_content_detected 属性，类型为可选布尔列表


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
# pylint: disable=invalid-name  # 禁用 pylint 对无效名称的警告

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串的开始
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
```py  # 文档字符串的结束标记
```  # 文档字符串的结束标记
    # 示例代码，展示如何使用图像和适配器进行稳定扩散处理
    Examples:
        ```py
        # 导入必要的库
        >>> from PIL import Image  # 导入图像处理库 PIL
        >>> from diffusers.utils import load_image  # 从 diffusers 库导入加载图像的函数
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableDiffusionAdapterPipeline, T2IAdapter  # 导入稳定扩散适配器和管道

        # 加载图像，使用指定的 URL
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_ref.png"  # 指定图像的 URL
        ... )

        # 将加载的图像调整为 8x8 的大小，创建色彩调色板
        >>> color_palette = image.resize((8, 8))  
        # 将色彩调色板调整为 512x512 的大小，使用最近邻插值
        >>> color_palette = color_palette.resize((512, 512), resample=Image.Resampling.NEAREST)

        # 从预训练的模型中加载 T2IAdapter，指定数据类型为 float16
        >>> adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_color_sd14v1", torch_dtype=torch.float16)
        # 从预训练的稳定扩散模型中加载管道，指定适配器和数据类型
        >>> pipe = StableDiffusionAdapterPipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4",  # 指定稳定扩散模型的名称
        ...     adapter=adapter,  # 使用上面加载的适配器
        ...     torch_dtype=torch.float16,  # 指定数据类型为 float16
        ... )

        # 将管道转移到 GPU
        >>> pipe.to("cuda")

        # 使用管道生成图像，提供提示和调色板
        >>> out_image = pipe(
        ...     "At night, glowing cubes in front of the beach",  # 提供生成图像的文本提示
        ...     image=color_palette,  # 使用之前调整的色彩调色板
        ... ).images[0]  # 获取生成图像的第一个元素
"""
# 文档字符串，描述该函数的功能和参数
def _preprocess_adapter_image(image, height, width):
    # 检查输入是否为 PyTorch 张量，如果是则直接返回
    if isinstance(image, torch.Tensor):
        return image
    # 如果输入是 PIL 图像，则将其放入列表中
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    # 检查列表中的第一个元素是否为 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 将每个图像调整为指定的高度和宽度，并转换为 NumPy 数组
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        # 扩展图像维度：如果是 [h, w]，则变为 [b, h, w, 1]；如果是 [h, w, c]，则变为 [b, h, w, c]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        # 沿第一个维度（批次维度）连接所有图像
        image = np.concatenate(image, axis=0)
        # 转换为浮点数并归一化到 [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        # 调整维度顺序为 [b, c, h, w]
        image = image.transpose(0, 3, 1, 2)
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果列表中的第一个元素是 PyTorch 张量
    elif isinstance(image[0], torch.Tensor):
        # 如果张量的维度是 3，沿第一个维度堆叠
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        # 如果张量的维度是 4，沿第一个维度连接
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        # 如果维度不正确，则抛出错误
        else:
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    # 返回处理后的图像
    return image


# 文档字符串，描述该函数的功能和参数
# 从调度器中检索时间步
def retrieve_timesteps(
    # 调度器对象，用于获取时间步
    scheduler,
    # 推理步骤数，指定生成样本时使用的扩散步骤
    num_inference_steps: Optional[int] = None,
    # 指定将时间步移动到的设备
    device: Optional[Union[str, torch.device]] = None,
    # 自定义时间步的列表
    timesteps: Optional[List[int]] = None,
    # 自定义 sigma 的列表
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法，并在调用后从调度器检索时间步。处理自定义时间步。
    所有额外参数将传递给 `scheduler.set_timesteps`。

    参数：
        scheduler (`SchedulerMixin`):
            从中获取时间步的调度器。
        num_inference_steps (`int`):
            生成样本时使用的扩散步骤数。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*):
            将时间步移动到的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *可选*):
            自定义时间步，覆盖调度器的时间步间隔策略。如果传递 `timesteps`，`num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*):
            自定义 sigma，覆盖调度器的时间步间隔策略。如果传递 `sigmas`，`num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回：
        `Tuple[torch.Tensor, int]`: 一个元组，第一个元素是来自调度器的时间步安排，第二个元素是推理步骤数。
    """
    # 如果同时传递了自定义时间步和 sigma，则抛出错误
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 检查 timesteps 是否不为 None
        if timesteps is not None:
            # 检查当前调度器的 set_timesteps 方法是否接受 timesteps 参数
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            # 如果不接受 timesteps，则抛出错误
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            # 调用调度器的 set_timesteps 方法，设置 timesteps、设备和其他参数
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            # 从调度器中获取当前的 timesteps
            timesteps = scheduler.timesteps
            # 计算推理步骤的数量
            num_inference_steps = len(timesteps)
        # 检查 sigmas 是否不为 None
        elif sigmas is not None:
            # 检查当前调度器的 set_timesteps 方法是否接受 sigmas 参数
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            # 如果不接受 sigmas，则抛出错误
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            # 调用调度器的 set_timesteps 方法，设置 sigmas、设备和其他参数
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            # 从调度器中获取当前的 timesteps
            timesteps = scheduler.timesteps
            # 计算推理步骤的数量
            num_inference_steps = len(timesteps)
        # 如果 timesteps 和 sigmas 都为 None
        else:
            # 调用调度器的 set_timesteps 方法，设置推理步骤数量、设备和其他参数
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            # 从调度器中获取当前的 timesteps
            timesteps = scheduler.timesteps
        # 返回 timesteps 和推理步骤数量
        return timesteps, num_inference_steps
# 继承自 DiffusionPipeline 和 StableDiffusionMixin 的稳定扩散适配器管道类
class StableDiffusionAdapterPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    用于使用增强 T2I-Adapter 的文本到图像生成的管道
    https://arxiv.org/abs/2302.08453

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档，以了解库为所有管道实现的通用方法
    （例如下载或保存、在特定设备上运行等）。

    参数：
        adapter ([`T2IAdapter`] 或 [`MultiAdapter`] 或 `List[T2IAdapter]`):
            在去噪过程中为 unet 提供额外的条件。如果将多个适配器设置为列表，
            每个适配器的输出将相加以创建一个组合的额外条件。
        adapter_weights (`List[float]`, *可选*, 默认值为 None):
            表示每个适配器的输出相加之前所乘的权重的浮点数列表。
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型，用于将图像编码和解码为潜在表示。
        text_encoder ([`CLIPTextModel`]):
            冻结的文本编码器。稳定扩散使用
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 的文本部分，
            特别是 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 变体。
        tokenizer (`CLIPTokenizer`):
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 类的标记器。
        unet ([`UNet2DConditionModel`]): 条件 U-Net 架构，用于去噪编码图像潜在表示。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用的调度器，以去噪编码图像潜在表示。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            分类模块，用于评估生成的图像是否可能被认为是冒犯性或有害的。
            请参阅 [模型卡](https://huggingface.co/runwayml/stable-diffusion-v1-5) 以获取详细信息。
        feature_extractor ([`CLIPImageProcessor`]):
            从生成的图像中提取特征的模型，用作 `safety_checker` 的输入。
    """

    # 定义模型 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->adapter->unet->vae"
    # 定义可选组件
    _optional_components = ["safety_checker", "feature_extractor"]
    # 初始化类的构造函数，接受多个参数以配置模型
        def __init__(
            self,
            # 变分自编码器，用于生成图像
            vae: AutoencoderKL,
            # 文本编码器，用于将文本转换为嵌入
            text_encoder: CLIPTextModel,
            # 标记器，用于文本的标记化处理
            tokenizer: CLIPTokenizer,
            # 条件生成模型，用于生成图像
            unet: UNet2DConditionModel,
            # 适配器，用于不同类型的输入适配
            adapter: Union[T2IAdapter, MultiAdapter, List[T2IAdapter]],
            # 调度器，用于控制生成过程的时间步骤
            scheduler: KarrasDiffusionSchedulers,
            # 安全检查器，用于过滤不安全内容
            safety_checker: StableDiffusionSafetyChecker,
            # 特征提取器，用于处理输入图像
            feature_extractor: CLIPImageProcessor,
            # 是否需要安全检查器的标志，默认为 True
            requires_safety_checker: bool = True,
        ):
            # 调用父类的构造函数
            super().__init__()
    
            # 检查安全检查器是否为 None，并且要求安全检查器
            if safety_checker is None and requires_safety_checker:
                # 记录警告信息，提醒用户安全检查器已被禁用
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 检查安全检查器存在，但特征提取器为 None
            if safety_checker is not None and feature_extractor is None:
                # 抛出错误，要求提供特征提取器
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 如果适配器是列表或元组，则将其转换为 MultiAdapter 实例
            if isinstance(adapter, (list, tuple)):
                adapter = MultiAdapter(adapter)
    
            # 注册模型的各个模块
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                adapter=adapter,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器实例，用于 VAE 缩放
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            # 将是否需要安全检查器的配置注册到类中
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt 复制的方法
        def _encode_prompt(
            # 提示文本，用于生成图像的输入
            prompt,
            # 设备类型（如 GPU 或 CPU）
            device,
            # 每个提示生成的图像数量
            num_images_per_prompt,
            # 是否执行无分类器自由引导
            do_classifier_free_guidance,
            # 可选的负提示文本，用于反向指导生成
            negative_prompt=None,
            # 可选的提示嵌入，预先计算的提示表示
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负提示嵌入，预先计算的负提示表示
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的 LoRA 缩放因子，用于微调
            lora_scale: Optional[float] = None,
            # 其他关键字参数
            **kwargs,
    # 声明一个过时警告信息，提醒用户此方法将在未来版本中删除
        ):
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            # 调用 deprecate 函数发出过时警告，指定版本和消息
            deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调用 encode_prompt 方法生成提示嵌入的元组
            prompt_embeds_tuple = self.encode_prompt(
                # 设置提示内容
                prompt=prompt,
                # 指定设备类型
                device=device,
                # 每个提示生成的图像数量
                num_images_per_prompt=num_images_per_prompt,
                # 是否进行分类自由引导
                do_classifier_free_guidance=do_classifier_free_guidance,
                # 设置负提示内容
                negative_prompt=negative_prompt,
                # 设置已有的提示嵌入
                prompt_embeds=prompt_embeds,
                # 设置已有的负提示嵌入
                negative_prompt_embeds=negative_prompt_embeds,
                # 设置 LoRA 缩放因子
                lora_scale=lora_scale,
                # 接收额外的关键字参数
                **kwargs,
            )
    
            # 为了向后兼容，将元组中的两个元素连接成一个张量
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    
            # 返回合并后的提示嵌入
            return prompt_embeds
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt 复制
        def encode_prompt(
            # 定义 encode_prompt 方法的参数
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            # 负提示内容的可选参数
            negative_prompt=None,
            # 提示嵌入的可选参数
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入的可选参数
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # LoRA 缩放因子的可选参数
            lora_scale: Optional[float] = None,
            # 可选的跳过剪辑参数
            clip_skip: Optional[int] = None,
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker 复制
        def run_safety_checker(self, image, device, dtype):
            # 检查安全检查器是否为 None
            if self.safety_checker is None:
                # 如果没有安全检查器，标记为 None
                has_nsfw_concept = None
            else:
                # 如果输入图像是张量格式
                if torch.is_tensor(image):
                    # 后处理图像以便特征提取
                    feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
                else:
                    # 如果是其他格式，转换为 PIL 格式
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                # 生成安全检查器输入，并将其移动到指定设备
                safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
                # 调用安全检查器检查图像和特征输入
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            # 返回处理后的图像和 NSFW 概念标记
            return image, has_nsfw_concept
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents 复制
    # 解码潜在变量并生成图像
        def decode_latents(self, latents):
            # 定义弃用警告信息，提示用户此方法将在1.0.0中移除
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            # 调用弃用警告函数，标记此方法已弃用
            deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
    
            # 调整潜在变量的尺度以便解码
            latents = 1 / self.vae.config.scaling_factor * latents
            # 解码潜在变量，返回图像数据
            image = self.vae.decode(latents, return_dict=False)[0]
            # 将图像数据归一化到[0, 1]范围
            image = (image / 2 + 0.5).clamp(0, 1)
            # 将图像数据转换为float32，保证兼容性
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            # 返回生成的图像
            return image
    
        # 准备额外步骤的参数，用于调度器
        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，不同调度器的参数签名可能不同
            # eta (η) 仅在DDIMScheduler中使用，其他调度器将忽略
            # eta对应于DDIM论文中的η，取值应在[0, 1]之间
    
            # 检查调度器是否接受eta参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外参数字典
            extra_step_kwargs = {}
            # 如果接受eta，则将其添加到字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受generator参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受generator，则将其添加到字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回包含额外参数的字典
            return extra_step_kwargs
    
        # 检查输入参数的有效性
        def check_inputs(
            self,
            prompt,
            height,
            width,
            callback_steps,
            image,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        # 检查高度和宽度是否都能被8整除，不符合则抛出错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查回调步骤是否有效，必须为正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查是否同时提供了提示和提示嵌入，不能同时存在
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查是否未提供任何提示或提示嵌入，必须至少提供一个
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查提示的类型，必须为字符串或列表
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查是否同时提供了负提示和负提示嵌入，不能同时存在
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查如果提供了提示嵌入，负提示嵌入也被提供，二者形状必须一致
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # 检查适配器类型是否为多适配器，且图像参数必须为列表
        if isinstance(self.adapter, MultiAdapter):
            if not isinstance(image, list):
                raise ValueError(
                    "MultiAdapter is enabled, but `image` is not a list. Please pass a list of images to `image`."
                )

            # 检查传入的图像数量与适配器数量是否匹配
            if len(image) != len(self.adapter.adapters):
                raise ValueError(
                    f"MultiAdapter requires passing the same number of images as adapters. Given {len(image)} images and {len(self.adapter.adapters)} adapters."
                )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的代码
    # 准备潜在变量（latents），根据给定的参数设置形状
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 计算潜在变量的形状，基于输入的批大小和通道数，以及经过 VAE 缩放因子的高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器是否为列表且其长度与批大小不匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 抛出错误，提示生成器的长度必须与请求的批大小一致
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果潜在变量为 None，生成随机的潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜在变量，将其转移到指定的设备上
            latents = latents.to(device)

        # 将初始噪声按调度器要求的标准差进行缩放
        latents = latents * self.scheduler.init_noise_sigma
        # 返回准备好的潜在变量
        return latents

    # 默认高度和宽度的设置，根据给定的高度、宽度和图像进行调整
    def _default_height_width(self, height, width, image):
        # 注意：可能传入的图像列表中每个图像的维度不同， 
        # 所以仅检查第一个图像并不完全准确，但这样处理较为简单
        while isinstance(image, list):
            # 如果图像是列表，取第一个图像进行处理
            image = image[0]

        # 如果高度未指定，进行计算
        if height is None:
            # 如果图像是 PIL 图像，获取其高度
            if isinstance(image, PIL.Image.Image):
                height = image.height
            # 如果图像是张量，获取其形状中的高度
            elif isinstance(image, torch.Tensor):
                height = image.shape[-2]

            # 向下取整至最近的 `self.adapter.downscale_factor` 的倍数
            height = (height // self.adapter.downscale_factor) * self.adapter.downscale_factor

        # 如果宽度未指定，进行计算
        if width is None:
            # 如果图像是 PIL 图像，获取其宽度
            if isinstance(image, PIL.Image.Image):
                width = image.width
            # 如果图像是张量，获取其形状中的宽度
            elif isinstance(image, torch.Tensor):
                width = image.shape[-1]

            # 向下取整至最近的 `self.adapter.downscale_factor` 的倍数
            width = (width // self.adapter.downscale_factor) * self.adapter.downscale_factor

        # 返回调整后的高度和宽度
        return height, width

    # 从 diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline 获取引导尺度嵌入
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:  # 定义一个返回类型为 torch.Tensor 的函数
        """  # 函数的文档字符串，描述功能和参数
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298  # 相关链接，提供更多信息

        Args:  # 参数说明部分开始
            w (`torch.Tensor`):  # 参数 w，类型为 torch.Tensor
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.  # 用于生成嵌入向量，指定引导比例以丰富时间步嵌入
            embedding_dim (`int`, *optional*, defaults to 512):  # 可选参数，嵌入的维度，默认值为 512
                Dimension of the embeddings to generate.  # 嵌入向量的维度
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):  # 可选参数，数据类型，默认为 torch.float32
                Data type of the generated embeddings.  # 生成的嵌入的类型

        Returns:  # 返回值说明部分开始
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.  # 返回形状为 (len(w), embedding_dim) 的嵌入向量
        """
        assert len(w.shape) == 1  # 确保 w 是一维张量
        w = w * 1000.0  # 将 w 的值放大 1000 倍

        half_dim = embedding_dim // 2  # 计算嵌入维度的一半
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)  # 计算嵌入的基准值
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)  # 生成对数空间的嵌入值
        emb = w.to(dtype)[:, None] * emb[None, :]  # 将 w 转换为指定 dtype 并与 emb 进行广播相乘
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # 将 sin 和 cos 值沿着维度 1 拼接
        if embedding_dim % 2 == 1:  # 如果嵌入维度为奇数
            emb = torch.nn.functional.pad(emb, (0, 1))  # 在最后一维进行零填充
        assert emb.shape == (w.shape[0], embedding_dim)  # 确保输出的形状正确
        return emb  # 返回计算得到的嵌入

    @property  # 将方法转换为属性
    def guidance_scale(self):  # 定义 guidance_scale 属性
        return self._guidance_scale  # 返回内部存储的引导比例

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)  # 解释 guidance_scale 的定义
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`  # 说明在特定论文中的应用
    # corresponds to doing no classifier free guidance.  # 解释当 guidance_scale 为 1 时的含义
    @property  # 将方法转换为属性
    def do_classifier_free_guidance(self):  # 定义 do_classifier_free_guidance 属性
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None  # 检查引导比例是否大于 1 且配置是否为 None

    @torch.no_grad()  # 在不计算梯度的上下文中执行
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 用示例文档字符串替换当前文档
    def __call__(  # 定义可调用对象的方法
        self,  # 对象自身的引用
        prompt: Union[str, List[str]] = None,  # 提示，可以是字符串或字符串列表，默认为 None
        image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]] = None,  # 输入图像，可以是 Tensor 或 PIL 图像，默认为 None
        height: Optional[int] = None,  # 可选参数，图像高度
        width: Optional[int] = None,  # 可选参数，图像宽度
        num_inference_steps: int = 50,  # 推理步骤的数量，默认为 50
        timesteps: List[int] = None,  # 可选参数，时间步列表
        sigmas: List[float] = None,  # 可选参数，sigma 列表
        guidance_scale: float = 7.5,  # 引导比例，默认为 7.5
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选参数，负提示
        num_images_per_prompt: Optional[int] = 1,  # 每个提示生成的图像数量，默认为 1
        eta: float = 0.0,  # 可选参数，eta 值，默认为 0.0
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 可选参数，随机数生成器
        latents: Optional[torch.Tensor] = None,  # 可选参数，潜在张量
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选参数，提示的嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选参数，负提示的嵌入
        output_type: Optional[str] = "pil",  # 可选参数，输出类型，默认为 "pil"
        return_dict: bool = True,  # 是否返回字典，默认为 True
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 可选参数，回调函数
        callback_steps: int = 1,  # 回调步骤，默认为 1
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选参数，交叉注意力的关键字参数
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,  # 可选参数，适配器条件比例，默认为 1.0
        clip_skip: Optional[int] = None,  # 可选参数，剪辑跳过的步骤
```