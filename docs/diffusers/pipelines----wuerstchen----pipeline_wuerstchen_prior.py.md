# `.\diffusers\pipelines\wuerstchen\pipeline_wuerstchen_prior.py`

```py
# 版权信息，声明此代码归 HuggingFace 团队所有，保留所有权利
# 许可证声明，使用此文件需遵守 Apache 许可证 2.0
# 提供许可证的获取地址
# 许可证说明，未按适用法律或书面协议另行约定的情况下，软件在“按现状”基础上分发
# 提供许可证详细信息的地址

from dataclasses import dataclass  # 导入数据类装饰器，用于简化类的定义
from math import ceil  # 导入向上取整函数
from typing import Callable, Dict, List, Optional, Union  # 导入类型注解

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习
from transformers import CLIPTextModel, CLIPTokenizer  # 导入 CLIP 模型和分词器

from ...loaders import StableDiffusionLoraLoaderMixin  # 导入加载 LoRA 权重的混合类
from ...schedulers import DDPMWuerstchenScheduler  # 导入调度器
from ...utils import BaseOutput, deprecate, logging, replace_example_docstring  # 导入工具类和函数
from ...utils.torch_utils import randn_tensor  # 导入生成随机张量的工具函数
from ..pipeline_utils import DiffusionPipeline  # 导入扩散管道基类
from .modeling_wuerstchen_prior import WuerstchenPrior  # 导入 Wuerstchen 先验模型

logger = logging.get_logger(__name__)  # 创建日志记录器

DEFAULT_STAGE_C_TIMESTEPS = list(np.linspace(1.0, 2 / 3, 20)) + list(np.linspace(2 / 3, 0.0, 11))[1:]  # 设置默认的时间步，分段线性生成

EXAMPLE_DOC_STRING = """  # 示例文档字符串，提供使用示例
    Examples:
        ```py  # Python 代码块开始
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import WuerstchenPriorPipeline  # 导入 WuerstchenPriorPipeline 类

        >>> prior_pipe = WuerstchenPriorPipeline.from_pretrained(  # 从预训练模型加载管道
        ...     "warp-ai/wuerstchen-prior", torch_dtype=torch.float16  # 指定模型路径和数据类型
        ... ).to("cuda")  # 将管道移动到 GPU

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"  # 定义生成图像的提示
        >>> prior_output = pipe(prompt)  # 生成图像并返回结果
        ```py  # Python 代码块结束
"""

@dataclass  # 使用数据类装饰器定义输出类
class WuerstchenPriorPipelineOutput(BaseOutput):  # 定义 WuerstchenPriorPipeline 的输出类
    """
    输出类用于 WuerstchenPriorPipeline。

    Args:
        image_embeddings (`torch.Tensor` or `np.ndarray`)  # 图像嵌入数据的类型说明
            Prior image embeddings for text prompt  # 为文本提示生成的图像嵌入

    """

    image_embeddings: Union[torch.Tensor, np.ndarray]  # 定义图像嵌入属性

class WuerstchenPriorPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):  # 定义 WuerstchenPriorPipeline 类，继承自扩散管道和加载器
    """
    用于生成 Wuerstchen 图像先验的管道。

    此模型继承自 [`DiffusionPipeline`]。查看超类文档以获取库实现的所有管道的通用方法（例如下载、保存、在特定设备上运行等）

    该管道还继承以下加载方法：
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] 用于加载 LoRA 权重
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] 用于保存 LoRA 权重

```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
```py  # 文档结束
```  # 文档结束
    # 文档字符串，说明构造函数参数及其作用
    Args:
        prior ([`Prior`]):
            # 指定用于从文本嵌入近似图像嵌入的标准 unCLIP 先验
        text_encoder ([`CLIPTextModelWithProjection`]):
            # 冻结的文本编码器
        tokenizer (`CLIPTokenizer`):
            # 用于文本处理的标记器，详细信息见 CLIPTokenizer 文档
        scheduler ([`DDPMWuerstchenScheduler`]):
            # 与 `prior` 结合使用的调度器，用于生成图像嵌入
        latent_mean ('float', *optional*, defaults to 42.0):
            # 潜在扩散器的均值
        latent_std ('float', *optional*, defaults to 1.0):
            # 潜在扩散器的标准差
        resolution_multiple ('float', *optional*, defaults to 42.67):
            # 生成多个图像时的默认分辨率
    """

    # 定义 unet 的名称为 "prior"
    unet_name = "prior"
    # 定义文本编码器的名称
    text_encoder_name = "text_encoder"
    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->prior"
    # 定义回调张量输入的列表
    _callback_tensor_inputs = ["latents", "text_encoder_hidden_states", "negative_prompt_embeds"]
    # 定义可加载的 LoRA 模块
    _lora_loadable_modules = ["prior", "text_encoder"]

    # 初始化函数，设置类的属性
    def __init__(
        self,
        # 初始化所需的标记器
        tokenizer: CLIPTokenizer,
        # 初始化所需的文本编码器
        text_encoder: CLIPTextModel,
        # 初始化所需的 unCLIP 先验
        prior: WuerstchenPrior,
        # 初始化所需的调度器
        scheduler: DDPMWuerstchenScheduler,
        # 设置潜在均值，默认值为 42.0
        latent_mean: float = 42.0,
        # 设置潜在标准差，默认值为 1.0
        latent_std: float = 1.0,
        # 设置生成图像的默认分辨率倍数，默认值为 42.67
        resolution_multiple: float = 42.67,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 注册所需的模块
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prior=prior,
            scheduler=scheduler,
        )
        # 将配置注册到类中
        self.register_to_config(
            latent_mean=latent_mean, latent_std=latent_std, resolution_multiple=resolution_multiple
        )

    # 从指定的管道准备潜在张量
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果未提供潜在张量，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查潜在张量的形状是否匹配
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在张量移动到指定设备
            latents = latents.to(device)

        # 将潜在张量乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回准备好的潜在张量
        return latents

    # 编码提示信息，处理正向和负向提示
    def encode_prompt(
        self,
        # 指定设备
        device,
        # 每个提示生成的图像数量
        num_images_per_prompt,
        # 是否进行无分类器自由引导
        do_classifier_free_guidance,
        # 正向提示文本
        prompt=None,
        # 负向提示文本
        negative_prompt=None,
        # 提示的嵌入张量，若有则提供
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负向提示的嵌入张量，若有则提供
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    # 检查输入的有效性
    def check_inputs(
        self,
        # 正向提示文本
        prompt,
        # 负向提示文本
        negative_prompt,
        # 推理步骤的数量
        num_inference_steps,
        # 是否进行无分类器自由引导
        do_classifier_free_guidance,
        # 提示的嵌入张量，若有则提供
        prompt_embeds=None,
        # 负向提示的嵌入张量，若有则提供
        negative_prompt_embeds=None,
    # 检查 prompt 和 prompt_embeds 是否同时存在
        ):
            if prompt is not None and prompt_embeds is not None:
                # 抛出异常，提示不能同时提供 prompt 和 prompt_embeds
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            # 检查 prompt 和 prompt_embeds 是否都未定义
            elif prompt is None and prompt_embeds is None:
                # 抛出异常，提示必须提供 prompt 或 prompt_embeds
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            # 检查 prompt 是否为有效类型
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                # 抛出异常，提示 prompt 类型不正确
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            # 检查 negative_prompt 和 negative_prompt_embeds 是否同时存在
            if negative_prompt is not None and negative_prompt_embeds is not None:
                # 抛出异常，提示不能同时提供 negative_prompt 和 negative_prompt_embeds
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            # 检查 prompt_embeds 和 negative_prompt_embeds 是否同时存在
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                # 验证这两个张量的形状是否一致
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    # 抛出异常，提示形状不匹配
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
            # 检查 num_inference_steps 是否为整数
            if not isinstance(num_inference_steps, int):
                # 抛出异常，提示 num_inference_steps 类型不正确
                raise TypeError(
                    f"'num_inference_steps' must be of type 'int', but got {type(num_inference_steps)}\
                               In Case you want to provide explicit timesteps, please use the 'timesteps' argument."
                )
    
        # 定义属性 guidance_scale，返回该类的 _guidance_scale 值
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 定义属性 do_classifier_free_guidance，判断是否执行无分类器引导
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1
    
        # 定义属性 num_timesteps，返回该类的 _num_timesteps 值
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 定义可调用方法，执行主要功能
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 定义方法的参数，包括 prompt 和其他配置
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            height: int = 1024,
            width: int = 1024,
            num_inference_steps: int = 60,
            timesteps: List[float] = None,
            guidance_scale: float = 8.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pt",
            return_dict: bool = True,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
```