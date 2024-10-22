# `.\diffusers\pipelines\pixart_alpha\pipeline_pixart_sigma.py`

```py
# 版权声明，标明版权所有者和团队
# Copyright 2024 PixArt-Sigma Authors and The HuggingFace Team. All rights reserved.
#
# 按照 Apache 2.0 许可证许可使用本文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 仅可在遵守许可证的情况下使用此文件
# you may not use this file except in compliance with the License.
# 可在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，本软件按“现状”基础分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

import html  # 导入 html 模块，用于处理 HTML 实体
import inspect  # 导入 inspect 模块，用于获取对象的信息
import re  # 导入 re 模块，用于正则表达式匹配
import urllib.parse as ul  # 导入 urllib.parse 模块并重命名为 ul，用于处理 URL
from typing import Callable, List, Optional, Tuple, Union  # 导入类型注解，便于定义类型

import torch  # 导入 PyTorch 库，用于深度学习
from transformers import T5EncoderModel, T5Tokenizer  # 从 transformers 导入 T5 编码器模型和分词器

from ...image_processor import PixArtImageProcessor  # 从相对路径导入 PixArt 图像处理器
from ...models import AutoencoderKL, PixArtTransformer2DModel  # 从相对路径导入模型
from ...schedulers import KarrasDiffusionSchedulers  # 从相对路径导入 Karras 采样调度器
from ...utils import (  # 从相对路径导入多个工具函数
    BACKENDS_MAPPING,  # 后端映射
    deprecate,  # 标记弃用的函数
    is_bs4_available,  # 检查 BeautifulSoup 是否可用
    is_ftfy_available,  # 检查 ftfy 是否可用
    logging,  # 日志记录工具
    replace_example_docstring,  # 替换示例文档字符串的函数
)
from ...utils.torch_utils import randn_tensor  # 从相对路径导入 randn_tensor 函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从相对路径导入扩散管道和图像输出
from .pipeline_pixart_alpha import (  # 从相对路径导入多个常量
    ASPECT_RATIO_256_BIN,  # 256 比例常量
    ASPECT_RATIO_512_BIN,  # 512 比例常量
    ASPECT_RATIO_1024_BIN,  # 1024 比例常量
)

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 pylint 的无效名称警告

if is_bs4_available():  # 检查 BeautifulSoup 是否可用
    from bs4 import BeautifulSoup  # 导入 BeautifulSoup 库以解析 HTML

if is_ftfy_available():  # 检查 ftfy 库是否可用
    import ftfy  # 导入 ftfy 库用于文本修复

# 定义一个字典，用于存储不同宽高比的尺寸
ASPECT_RATIO_2048_BIN = {
    "0.25": [1024.0, 4096.0],  # 宽高比为 0.25 时的宽高尺寸
    "0.26": [1024.0, 3968.0],  # 宽高比为 0.26 时的宽高尺寸
    "0.27": [1024.0, 3840.0],  # 宽高比为 0.27 时的宽高尺寸
    "0.28": [1024.0, 3712.0],  # 宽高比为 0.28 时的宽高尺寸
    "0.32": [1152.0, 3584.0],  # 宽高比为 0.32 时的宽高尺寸
    "0.33": [1152.0, 3456.0],  # 宽高比为 0.33 时的宽高尺寸
    "0.35": [1152.0, 3328.0],  # 宽高比为 0.35 时的宽高尺寸
    "0.4": [1280.0, 3200.0],  # 宽高比为 0.4 时的宽高尺寸
    "0.42": [1280.0, 3072.0],  # 宽高比为 0.42 时的宽高尺寸
    "0.48": [1408.0, 2944.0],  # 宽高比为 0.48 时的宽高尺寸
    "0.5": [1408.0, 2816.0],  # 宽高比为 0.5 时的宽高尺寸
    "0.52": [1408.0, 2688.0],  # 宽高比为 0.52 时的宽高尺寸
    "0.57": [1536.0, 2688.0],  # 宽高比为 0.57 时的宽高尺寸
    "0.6": [1536.0, 2560.0],  # 宽高比为 0.6 时的宽高尺寸
    "0.68": [1664.0, 2432.0],  # 宽高比为 0.68 时的宽高尺寸
    "0.72": [1664.0, 2304.0],  # 宽高比为 0.72 时的宽高尺寸
    "0.78": [1792.0, 2304.0],  # 宽高比为 0.78 时的宽高尺寸
    "0.82": [1792.0, 2176.0],  # 宽高比为 0.82 时的宽高尺寸
    "0.88": [1920.0, 2176.0],  # 宽高比为 0.88 时的宽高尺寸
    "0.94": [1920.0, 2048.0],  # 宽高比为 0.94 时的宽高尺寸
    "1.0": [2048.0, 2048.0],  # 宽高比为 1.0 时的宽高尺寸
    "1.07": [2048.0, 1920.0],  # 宽高比为 1.07 时的宽高尺寸
    "1.13": [2176.0, 1920.0],  # 宽高比为 1.13 时的宽高尺寸
    "1.21": [2176.0, 1792.0],  # 宽高比为 1.21 时的宽高尺寸
    "1.29": [2304.0, 1792.0],  # 宽高比为 1.29 时的宽高尺寸
    "1.38": [2304.0, 1664.0],  # 宽高比为 1.38 时的宽高尺寸
    "1.46": [2432.0, 1664.0],  # 宽高比为 1.46 时的宽高尺寸
    "1.67": [2560.0, 1536.0],  # 宽高比为 1.67 时的宽高尺寸
    "1.75": [2688.0, 1536.0],  # 宽高比为 1.75 时的宽高尺寸
    "2.0": [2816.0, 1408.0],  # 宽高比为 2.0 时的宽高尺寸
    "2.09": [2944.0, 1408.0],  # 宽高比为 2.09 时的宽高尺寸
    "2.4": [3072.0, 1280.0],  # 宽高比为 2.4 时的宽高尺寸
    "2.5": [3200.0, 1280.0],  # 宽高比为 2.5 时的宽高尺寸
    "2.89": [3328.0, 1152.0],  # 宽高比为 2.89 时的宽高尺寸
    "3.0": [3456.0, 1152.0],  # 宽高比为 3.0 时的宽
    # 示例代码部分
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库，以便进行张量操作和深度学习
        >>> from diffusers import PixArtSigmaPipeline  # 从 diffusers 库导入 PixArtSigmaPipeline 类

        >>> # 你可以将检查点 ID 替换为 "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
        >>> pipe = PixArtSigmaPipeline.from_pretrained(  # 从预训练模型加载 PixArtSigmaPipeline
        ...     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16  # 指定模型路径及张量数据类型为 float16
        ... )
        >>> # 启用内存优化
        >>> # pipe.enable_model_cpu_offload()  # 可选：启用模型的 CPU 卸载以节省内存

        >>> prompt = "A small cactus with a happy face in the Sahara desert."  # 设置生成图像的提示文本
        >>> image = pipe(prompt).images[0]  # 生成图像并提取第一张图像
        ```py 
"""
# 该函数用于从调度器中检索时间步
# 复制自 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,  # 调度器对象，用于获取时间步
    num_inference_steps: Optional[int] = None,  # 生成样本时使用的扩散步骤数量，默认为 None
    device: Optional[Union[str, torch.device]] = None,  # 要移动时间步的设备，默认为 None
    timesteps: Optional[List[int]] = None,  # 自定义时间步，用于覆盖调度器的时间步间距策略，默认为 None
    sigmas: Optional[List[float]] = None,  # 自定义 sigma，用于覆盖调度器的时间步间距策略，默认为 None
    **kwargs,  # 其他可选参数，传递给调度器的 set_timesteps 方法
):
    """
    调用调度器的 `set_timesteps` 方法并在调用后从调度器检索时间步。处理
    自定义时间步。任何 kwargs 将被传递给 `scheduler.set_timesteps`。

    Args:
        scheduler (`SchedulerMixin`): 需要从中获取时间步的调度器。
        num_inference_steps (`int`): 生成样本时使用的扩散步骤数量。如果使用，`timesteps` 必须为 `None`。
        device (`str` or `torch.device`, *optional*): 要移动时间步的设备。如果为 `None`，则不移动时间步。
        timesteps (`List[int]`, *optional*): 自定义时间步，用于覆盖调度器的时间步间距策略。如果传递了 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *optional*): 自定义 sigma，用于覆盖调度器的时间步间距策略。如果传递了 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    Returns:
        `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是调度器的时间步调度，第二个元素是推理步骤的数量。
    """
    # 检查是否同时传递了自定义时间步和 sigma
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 如果传递了自定义时间步
    if timesteps is not None:
        # 检查调度器的 set_timesteps 方法是否接受自定义时间步
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:  # 如果不支持，抛出错误
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义时间步并移动到指定设备
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    # 如果传递了自定义 sigma
    elif sigmas is not None:
        # 检查调度器的 set_timesteps 方法是否接受自定义 sigma
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:  # 如果不支持，抛出错误
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置自定义 sigma 并移动到指定设备
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 从调度器获取设置后的时间步
        timesteps = scheduler.timesteps
        # 计算推理步骤的数量
        num_inference_steps = len(timesteps)
    else:  # 如果条件不满足，则执行下面的代码
        # 设置调度器的时间步数，传入推理步数和设备参数，可能还包含其他关键字参数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取当前调度器的时间步数
        timesteps = scheduler.timesteps
    # 返回时间步数和推理步数
    return timesteps, num_inference_steps
# 定义一个名为 PixArtSigmaPipeline 的类，继承自 DiffusionPipeline 类
class PixArtSigmaPipeline(DiffusionPipeline):
    r"""
    使用 PixArt-Sigma 进行文本到图像生成的管道。
    """

    # 编译一个正则表达式，用于匹配不良标点符号
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    # 定义可选组件的名称列表
    _optional_components = ["tokenizer", "text_encoder"]
    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # 初始化方法，接受多个组件作为参数
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: PixArtTransformer2DModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册传入的模块
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器，使用计算出的缩放因子
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # 从 PixArtAlphaPipeline 复制的方法，用于编码提示信息，最大序列长度从 120 改为 300
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        **kwargs,
    # 从 StableDiffusionPipeline 复制的方法，用于准备额外的调度步骤参数
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为调度器步骤准备额外的关键字参数，因为并不是所有调度器都有相同的签名
        # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略它。
        # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
        # 应在 [0, 1] 之间

        # 检查调度器是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        # 如果接受 eta，则将其添加到额外参数中
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        # 如果接受 generator，则将其添加到额外参数中
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        # 返回准备好的额外参数字典
        return extra_step_kwargs

    # 从 PixArtAlphaPipeline 复制的方法，用于检查输入参数
    # 定义检查输入参数的函数
        def check_inputs(
            self,
            prompt,  # 提示文本
            height,  # 图像高度
            width,   # 图像宽度
            negative_prompt,  # 负面提示文本
            callback_steps,  # 回调步数
            prompt_embeds=None,  # 提示嵌入（可选）
            negative_prompt_embeds=None,  # 负面提示嵌入（可选）
            prompt_attention_mask=None,  # 提示注意力掩码（可选）
            negative_prompt_attention_mask=None,  # 负面提示注意力掩码（可选）
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制
        def _text_preprocessing(self, text, clean_caption=False):  # 文本预处理函数，带有清理标志
            if clean_caption and not is_bs4_available():  # 检查是否需要清理且 bs4 库不可用
                logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))  # 记录警告
                logger.warning("Setting `clean_caption` to False...")  # 记录清理被设置为 False 的警告
                clean_caption = False  # 设置清理标志为 False
    
            if clean_caption and not is_ftfy_available():  # 检查是否需要清理且 ftfy 库不可用
                logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))  # 记录警告
                logger.warning("Setting `clean_caption` to False...")  # 记录清理被设置为 False 的警告
                clean_caption = False  # 设置清理标志为 False
    
            if not isinstance(text, (tuple, list)):  # 检查文本类型是否为元组或列表
                text = [text]  # 将文本包装为列表
    
            def process(text: str):  # 定义处理文本的内部函数
                if clean_caption:  # 如果需要清理文本
                    text = self._clean_caption(text)  # 清理文本
                    text = self._clean_caption(text)  # 再次清理文本
                else:  # 如果不需要清理文本
                    text = text.lower().strip()  # 将文本转为小写并去除空格
                return text  # 返回处理后的文本
    
            return [process(t) for t in text]  # 对每个文本项进行处理并返回结果列表
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption 复制
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):  # 准备潜在变量的函数
            shape = (  # 定义潜在变量的形状
                batch_size,  # 批量大小
                num_channels_latents,  # 潜在变量的通道数
                int(height) // self.vae_scale_factor,  # 缩放后的高度
                int(width) // self.vae_scale_factor,  # 缩放后的宽度
            )
            if isinstance(generator, list) and len(generator) != batch_size:  # 检查生成器列表的长度是否与批量大小匹配
                raise ValueError(  # 抛出值错误
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"  # 提示生成器长度与批量大小不匹配
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."  # 提示用户检查匹配
                )
    
            if latents is None:  # 如果未提供潜在变量
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # 生成随机潜在变量
            else:  # 如果提供了潜在变量
                latents = latents.to(device)  # 将潜在变量移动到指定设备
    
            # 根据调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma  # 缩放潜在变量
            return latents  # 返回潜在变量
    
        @torch.no_grad()  # 在无梯度模式下运行
        @replace_example_docstring(EXAMPLE_DOC_STRING)  # 替换示例文档字符串
    # 定义一个可调用的类方法，接受多种参数
        def __call__(
            # 用户输入的提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 用户的负面提示，默认为空字符串
            negative_prompt: str = "",
            # 推理步骤的数量，默认为20
            num_inference_steps: int = 20,
            # 时间步列表，默认为None
            timesteps: List[int] = None,
            # Sigma值列表，默认为None
            sigmas: List[float] = None,
            # 指导比例，默认为4.5
            guidance_scale: float = 4.5,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 输出图像的高度，默认为None
            height: Optional[int] = None,
            # 输出图像的宽度，默认为None
            width: Optional[int] = None,
            # Eta值，默认为0.0
            eta: float = 0.0,
            # 随机数生成器，可为单个或多个torch.Generator，默认为None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在向量，默认为None
            latents: Optional[torch.Tensor] = None,
            # 提示的嵌入表示，默认为None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 提示的注意力掩码，默认为None
            prompt_attention_mask: Optional[torch.Tensor] = None,
            # 负面提示的嵌入表示，默认为None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 负面提示的注意力掩码，默认为None
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            # 输出类型，默认为'pil'
            output_type: Optional[str] = "pil",
            # 是否返回字典格式，默认为True
            return_dict: bool = True,
            # 回调函数，接受三个参数，默认为None
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调步骤的数量，默认为1
            callback_steps: int = 1,
            # 是否清理标题，默认为True
            clean_caption: bool = True,
            # 是否使用分辨率分箱，默认为True
            use_resolution_binning: bool = True,
            # 最大序列长度，默认为300
            max_sequence_length: int = 300,
            # 其他关键字参数
            **kwargs,
```