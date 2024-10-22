# `.\diffusers\pipelines\wuerstchen\pipeline_wuerstchen_combined.py`

```py
# 版权信息，表明版权所有者和许可信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证版本 2.0 进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 本文件只能在遵循许可证的情况下使用
# you may not use this file except in compliance with the License.
# 可以在以下地址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“原样”基础分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的保证或条件，无论是明示或暗示
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参见许可证以了解管理权限和限制的具体语言
# See the License for the specific language governing permissions and
# limitations under the License.
# 导入所需的类型提示
from typing import Callable, Dict, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIP 文本模型和分词器
from transformers import CLIPTextModel, CLIPTokenizer

# 从自定义调度器导入 DDPMWuerstchenScheduler
from ...schedulers import DDPMWuerstchenScheduler
# 从自定义工具导入去除过时函数和替换示例文档字符串的函数
from ...utils import deprecate, replace_example_docstring
# 从管道工具导入 DiffusionPipeline 基类
from ..pipeline_utils import DiffusionPipeline
# 从模型模块导入 PaellaVQModel
from .modeling_paella_vq_model import PaellaVQModel
# 从模型模块导入 WuerstchenDiffNeXt
from .modeling_wuerstchen_diffnext import WuerstchenDiffNeXt
# 从模型模块导入 WuerstchenPrior
from .modeling_wuerstchen_prior import WuerstchenPrior
# 从管道模块导入 WuerstchenDecoderPipeline
from .pipeline_wuerstchen import WuerstchenDecoderPipeline
# 从管道模块导入 WuerstchenPriorPipeline
from .pipeline_wuerstchen_prior import WuerstchenPriorPipeline

# 文档字符串示例，用于展示如何使用文本转图像的管道
TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusions import WuerstchenCombinedPipeline

        >>> pipe = WuerstchenCombinedPipeline.from_pretrained("warp-ai/Wuerstchen", torch_dtype=torch.float16).to(
        ...     "cuda"
        ... )
        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> images = pipe(prompt=prompt)
        ```
"""

# 定义一个结合文本到图像生成的管道类
class WuerstchenCombinedPipeline(DiffusionPipeline):
    """
    使用 Wuerstchen 进行文本到图像生成的组合管道

    该模型继承自 [`DiffusionPipeline`]。查看父类文档以了解库为所有管道实现的通用方法
    (如下载或保存，运行在特定设备等)。

    参数:
        tokenizer (`CLIPTokenizer`):
            用于文本输入的解码器分词器。
        text_encoder (`CLIPTextModel`):
            用于文本输入的解码器文本编码器。
        decoder (`WuerstchenDiffNeXt`):
            用于图像生成管道的解码器模型。
        scheduler (`DDPMWuerstchenScheduler`):
            用于图像生成管道的调度器。
        vqgan (`PaellaVQModel`):
            用于图像生成管道的 VQGAN 模型。
        prior_tokenizer (`CLIPTokenizer`):
            用于文本输入的先前分词器。
        prior_text_encoder (`CLIPTextModel`):
            用于文本输入的先前文本编码器。
        prior_prior (`WuerstchenPrior`):
            用于先前管道的先前模型。
        prior_scheduler (`DDPMWuerstchenScheduler`):
            用于先前管道的调度器。
    """

    # 标志，表示是否加载连接的管道
    _load_connected_pipes = True
    # 初始化类的构造函数，接收多个模型和调度器作为参数
        def __init__(
            self,
            tokenizer: CLIPTokenizer,  # 词汇处理器
            text_encoder: CLIPTextModel,  # 文本编码器
            decoder: WuerstchenDiffNeXt,  # 解码器模型
            scheduler: DDPMWuerstchenScheduler,  # 调度器
            vqgan: PaellaVQModel,  # VQGAN模型
            prior_tokenizer: CLIPTokenizer,  # 先验词汇处理器
            prior_text_encoder: CLIPTextModel,  # 先验文本编码器
            prior_prior: WuerstchenPrior,  # 先验模型
            prior_scheduler: DDPMWuerstchenScheduler,  # 先验调度器
        ):
            super().__init__()  # 调用父类的构造函数
    
            # 注册各个模型和调度器到当前实例
            self.register_modules(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                decoder=decoder,
                scheduler=scheduler,
                vqgan=vqgan,
                prior_prior=prior_prior,
                prior_text_encoder=prior_text_encoder,
                prior_tokenizer=prior_tokenizer,
                prior_scheduler=prior_scheduler,
            )
            # 初始化先验管道，用于处理先验相关操作
            self.prior_pipe = WuerstchenPriorPipeline(
                prior=prior_prior,  # 先验模型
                text_encoder=prior_text_encoder,  # 先验文本编码器
                tokenizer=prior_tokenizer,  # 先验词汇处理器
                scheduler=prior_scheduler,  # 先验调度器
            )
            # 初始化解码器管道，用于处理解码相关操作
            self.decoder_pipe = WuerstchenDecoderPipeline(
                text_encoder=text_encoder,  # 文本编码器
                tokenizer=tokenizer,  # 词汇处理器
                decoder=decoder,  # 解码器
                scheduler=scheduler,  # 调度器
                vqgan=vqgan,  # VQGAN模型
            )
    
        # 启用节省内存的高效注意力机制
        def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
            # 在解码器管道中启用高效注意力机制
            self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)
    
        # 启用模型的CPU卸载，减少内存使用
        def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
            r"""
            使用accelerate将所有模型卸载到CPU，减少内存使用且对性能影响较小。
            此方法在调用模型的`forward`方法时将整个模型移到GPU，模型将在下一个模型运行之前保持在GPU上。
            相比于`enable_sequential_cpu_offload`，内存节省较少，但性能更佳。
            """
            # 在先验管道中启用模型的CPU卸载
            self.prior_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
            # 在解码器管道中启用模型的CPU卸载
            self.decoder_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
    
        # 启用顺序CPU卸载，显著减少内存使用
        def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
            r"""
            使用🤗Accelerate将所有模型卸载到CPU，显著减少内存使用。
            模型被移动到`torch.device('meta')`，并仅在调用特定子模块的`forward`方法时加载到GPU。
            卸载是基于子模块进行的，内存节省比使用`enable_model_cpu_offload`高，但性能较低。
            """
            # 在先验管道中启用顺序CPU卸载
            self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
            # 在解码器管道中启用顺序CPU卸载
            self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
    # 定义进度条方法，接受可迭代对象和总计数作为参数
    def progress_bar(self, iterable=None, total=None):
        # 在 prior_pipe 上更新进度条，传入可迭代对象和总计数
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        # 在 decoder_pipe 上更新进度条，传入可迭代对象和总计数
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)

    # 定义设置进度条配置的方法，接收任意关键字参数
    def set_progress_bar_config(self, **kwargs):
        # 在 prior_pipe 上设置进度条配置，传入关键字参数
        self.prior_pipe.set_progress_bar_config(**kwargs)
        # 在 decoder_pipe 上设置进度条配置，传入关键字参数
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    # 使用 torch.no_grad() 装饰器，表示在此上下文中不计算梯度
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)
    # 定义调用方法，处理文本到图像的转换
    def __call__(
        # 接受提示文本，支持字符串或字符串列表，默认为 None
        prompt: Optional[Union[str, List[str]]] = None,
        # 图像高度，默认为 512
        height: int = 512,
        # 图像宽度，默认为 512
        width: int = 512,
        # prior 阶段推理步骤数，默认为 60
        prior_num_inference_steps: int = 60,
        # prior 阶段时间步，默认为 None
        prior_timesteps: Optional[List[float]] = None,
        # prior 阶段引导比例，默认为 4.0
        prior_guidance_scale: float = 4.0,
        # decoder 阶段推理步骤数，默认为 12
        num_inference_steps: int = 12,
        # decoder 阶段时间步，默认为 None
        decoder_timesteps: Optional[List[float]] = None,
        # decoder 阶段引导比例，默认为 0.0
        decoder_guidance_scale: float = 0.0,
        # 负提示文本，支持字符串或字符串列表，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 提示嵌入，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示嵌入，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: int = 1,
        # 随机数生成器，默认为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在表示，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典格式，默认为 True
        return_dict: bool = True,
        # prior 阶段的回调函数，默认为 None
        prior_callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # prior 阶段回调函数输入的张量名称列表，默认为 ["latents"]
        prior_callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # decoder 阶段的回调函数，默认为 None
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # decoder 阶段回调函数输入的张量名称列表，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 接受其他任意关键字参数
        **kwargs,
```