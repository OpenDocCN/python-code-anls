# `.\diffusers\pipelines\stable_diffusion\pipeline_onnx_stable_diffusion.py`

```py
# 版权声明，表明该代码的版权所有者及相关条款
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件在 "按原样" 基础上分发，
# 不提供任何明示或暗示的担保或条件。
# 请参见许可证以获取有关特定语言治理权限和
# 限制的更多信息。

# 导入 inspect 模块以进行获取对象的文档字符串和源代码
import inspect
# 从 typing 模块导入类型提示所需的工具
from typing import Callable, List, Optional, Union

# 导入 numpy 库用于数值计算
import numpy as np
# 导入 torch 库用于深度学习模型的构建和训练
import torch
# 从 transformers 库导入 CLIP 图像处理器和 CLIP 分词器
from transformers import CLIPImageProcessor, CLIPTokenizer

# 从配置工具导入 FrozenDict 用于处理不可变字典
from ...configuration_utils import FrozenDict
# 从调度器导入不同类型的调度器
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
# 从工具模块导入去deprecated功能和日志记录
from ...utils import deprecate, logging
# 从 onnx_utils 导入 ONNX 相关的类型和模型
from ..onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
# 从 pipeline_utils 导入扩散管道
from ..pipeline_utils import DiffusionPipeline
# 导入 StableDiffusionPipelineOutput 模块
from . import StableDiffusionPipelineOutput

# 创建一个日志记录器，用于记录该模块的日志信息
logger = logging.get_logger(__name__)

# 定义 OnnxStableDiffusionPipeline 类，继承自 DiffusionPipeline
class OnnxStableDiffusionPipeline(DiffusionPipeline):
    # 声明类的各个成员变量，表示使用的模型组件
    vae_encoder: OnnxRuntimeModel
    vae_decoder: OnnxRuntimeModel
    text_encoder: OnnxRuntimeModel
    tokenizer: CLIPTokenizer
    unet: OnnxRuntimeModel
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
    safety_checker: OnnxRuntimeModel
    feature_extractor: CLIPImageProcessor

    # 定义可选组件的列表，包括安全检查器和特征提取器
    _optional_components = ["safety_checker", "feature_extractor"]
    # 标记该管道为 ONNX 格式
    _is_onnx = True

    # 初始化函数，设置各个组件的参数
    def __init__(
        self,
        vae_encoder: OnnxRuntimeModel,  # VAE 编码器模型
        vae_decoder: OnnxRuntimeModel,  # VAE 解码器模型
        text_encoder: OnnxRuntimeModel,  # 文本编码器模型
        tokenizer: CLIPTokenizer,        # CLIP 分词器
        unet: OnnxRuntimeModel,          # U-Net 模型
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],  # 调度器
        safety_checker: OnnxRuntimeModel,  # 安全检查器模型
        feature_extractor: CLIPImageProcessor,  # 特征提取器
        requires_safety_checker: bool = True,  # 是否需要安全检查器
    ):
    # 定义用于编码提示的私有方法
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],  # 输入的提示文本，可以是字符串或字符串列表
        num_images_per_prompt: Optional[int],  # 每个提示生成的图像数量
        do_classifier_free_guidance: bool,  # 是否使用无分类器引导
        negative_prompt: Optional[str],  # 负面提示文本
        prompt_embeds: Optional[np.ndarray] = None,  # 提示的嵌入表示
        negative_prompt_embeds: Optional[np.ndarray] = None,  # 负面提示的嵌入表示
    ):
    # 定义检查输入有效性的私有方法
    def check_inputs(
        self,
        prompt: Union[str, List[str]],  # 输入的提示文本
        height: Optional[int],  # 图像高度
        width: Optional[int],  # 图像宽度
        callback_steps: int,  # 回调步骤数量
        negative_prompt: Optional[str] = None,  # 负面提示文本
        prompt_embeds: Optional[np.ndarray] = None,  # 提示的嵌入表示
        negative_prompt_embeds: Optional[np.ndarray] = None,  # 负面提示的嵌入表示
    # 进行一系列参数检查，以确保输入值的有效性
        ):
            # 检查高度和宽度是否都能被8整除
            if height % 8 != 0 or width % 8 != 0:
                # 如果不能整除，抛出值错误异常，提示当前高度和宽度
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    
            # 检查回调步骤是否为正整数
            if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
            ):
                # 如果条件不满足，抛出值错误异常，提示当前回调步骤的类型和值
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )
    
            # 检查同时传入 prompt 和 prompt_embeds
            if prompt is not None and prompt_embeds is not None:
                # 如果同时传入，抛出值错误异常，提示只能传入其中一个
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查 prompt 和 prompt_embeds 是否均未提供
            elif prompt is None and prompt_embeds is None:
                # 抛出值错误异常，提示必须提供至少一个
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查 prompt 的类型
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                # 如果类型不匹配，抛出值错误异常，提示类型不符合
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查同时传入 negative_prompt 和 negative_prompt_embeds
            if negative_prompt is not None and negative_prompt_embeds is not None:
                # 如果同时传入，抛出值错误异常，提示只能传入其中一个
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查 prompt_embeds 和 negative_prompt_embeds 是否都提供
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                # 确保它们的形状相同
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    # 如果形状不匹配，抛出值错误异常，提示它们的形状
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
        # 定义调用方法，接受多个参数
        def __call__(
            # 提供的提示，类型为字符串或字符串列表
            prompt: Union[str, List[str]] = None,
            # 图像高度，默认为512
            height: Optional[int] = 512,
            # 图像宽度，默认为512
            width: Optional[int] = 512,
            # 推理步骤的数量，默认为50
            num_inference_steps: Optional[int] = 50,
            # 指导尺度，默认为7.5
            guidance_scale: Optional[float] = 7.5,
            # 负提示，类型为字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 额外的随机因素，默认为0.0
            eta: Optional[float] = 0.0,
            # 随机生成器，默认为None
            generator: Optional[np.random.RandomState] = None,
            # 潜在变量，默认为None
            latents: Optional[np.ndarray] = None,
            # 提示的嵌入表示，默认为None
            prompt_embeds: Optional[np.ndarray] = None,
            # 负提示的嵌入表示，默认为None
            negative_prompt_embeds: Optional[np.ndarray] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典，默认为True
            return_dict: bool = True,
            # 回调函数，默认为None
            callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
            # 回调步骤，默认为1
            callback_steps: int = 1,
# 定义一个名为 StableDiffusionOnnxPipeline 的类，继承自 OnnxStableDiffusionPipeline
class StableDiffusionOnnxPipeline(OnnxStableDiffusionPipeline):
    # 初始化方法，接受多个模型和处理器作为参数
    def __init__(
        self,
        vae_encoder: OnnxRuntimeModel,  # VAE 编码器模型
        vae_decoder: OnnxRuntimeModel,  # VAE 解码器模型
        text_encoder: OnnxRuntimeModel,  # 文本编码器模型
        tokenizer: CLIPTokenizer,        # 分词器
        unet: OnnxRuntimeModel,          # U-Net 模型
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],  # 调度器，可以是多种类型
        safety_checker: OnnxRuntimeModel,  # 安全检查模型
        feature_extractor: CLIPImageProcessor,  # 特征提取器
    ):
        # 定义弃用消息，提醒用户使用替代类
        deprecation_message = "Please use `OnnxStableDiffusionPipeline` instead of `StableDiffusionOnnxPipeline`."
        # 调用弃用函数，记录弃用警告
        deprecate("StableDiffusionOnnxPipeline", "1.0.0", deprecation_message)
        # 调用父类的初始化方法，传入所有参数
        super().__init__(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
```