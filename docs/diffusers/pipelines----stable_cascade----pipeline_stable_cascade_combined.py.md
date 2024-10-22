# `.\diffusers\pipelines\stable_cascade\pipeline_stable_cascade_combined.py`

```py
# 版权声明，说明该代码的所有权
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 使用 Apache License 2.0 进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件只能在遵循许可证的情况下使用
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件在“原样”基础上分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定语言的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
# 导入所需的类型定义
from typing import Callable, Dict, List, Optional, Union

# 导入图像处理库
import PIL
# 导入 PyTorch
import torch
# 从 transformers 库导入 CLIP 模型及处理器
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# 从本地模型中导入 StableCascadeUNet
from ...models import StableCascadeUNet
# 从调度器中导入 DDPMWuerstchenScheduler
from ...schedulers import DDPMWuerstchenScheduler
# 导入工具函数
from ...utils import is_torch_version, replace_example_docstring
# 从管道工具中导入 DiffusionPipeline
from ..pipeline_utils import DiffusionPipeline
# 从 VQ 模型中导入 PaellaVQModel
from ..wuerstchen.modeling_paella_vq_model import PaellaVQModel
# 导入 StableCascade 解码器管道
from .pipeline_stable_cascade import StableCascadeDecoderPipeline
# 导入 StableCascade 优先管道
from .pipeline_stable_cascade_prior import StableCascadePriorPipeline


# 文档字符串示例，展示如何使用文本转图像功能
TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableCascadeCombinedPipeline

        # 从预训练模型创建管道实例
        >>> pipe = StableCascadeCombinedPipeline.from_pretrained(
        ...     "stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.bfloat16
        ... )
        # 启用模型的 CPU 离线加载
        >>> pipe.enable_model_cpu_offload()
        # 定义图像生成的提示
        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        # 生成图像
        >>> images = pipe(prompt=prompt)
        ```
"""

# 定义稳定级联组合管道类，用于文本到图像生成
class StableCascadeCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for text-to-image generation using Stable Cascade.

    该模型继承自 [`DiffusionPipeline`]。检查父类文档以了解库为所有管道实现的通用方法
    (例如下载或保存、在特定设备上运行等。)
    # 文档字符串，描述初始化方法参数的含义
        Args:
            tokenizer (`CLIPTokenizer`):
                用于文本输入的解码器分词器。
            text_encoder (`CLIPTextModel`):
                用于文本输入的解码器文本编码器。
            decoder (`StableCascadeUNet`):
                用于解码器图像生成管道的解码模型。
            scheduler (`DDPMWuerstchenScheduler`):
                用于解码器图像生成管道的调度器。
            vqgan (`PaellaVQModel`):
                用于解码器图像生成管道的 VQGAN 模型。
            feature_extractor ([`~transformers.CLIPImageProcessor`]):
                从生成图像中提取特征的模型，作为 `image_encoder` 的输入。
            image_encoder ([`CLIPVisionModelWithProjection`]):
                冻结的 CLIP 图像编码器（[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）。
            prior_prior (`StableCascadeUNet`):
                用于先验管道的先验模型。
            prior_scheduler (`DDPMWuerstchenScheduler`):
                用于先验管道的调度器。
        """
    
        # 设置加载连接管道的标志为 True
        _load_connected_pipes = True
        # 定义可选组件的列表
        _optional_components = ["prior_feature_extractor", "prior_image_encoder"]
    
        # 初始化方法
        def __init__(
            # 定义参数类型及名称
            self,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            decoder: StableCascadeUNet,
            scheduler: DDPMWuerstchenScheduler,
            vqgan: PaellaVQModel,
            prior_prior: StableCascadeUNet,
            prior_text_encoder: CLIPTextModel,
            prior_tokenizer: CLIPTokenizer,
            prior_scheduler: DDPMWuerstchenScheduler,
            prior_feature_extractor: Optional[CLIPImageProcessor] = None,
            prior_image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        ):
            # 调用父类初始化方法
            super().__init__()
    
            # 注册多个模块以便于管理
            self.register_modules(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                decoder=decoder,
                scheduler=scheduler,
                vqgan=vqgan,
                prior_text_encoder=prior_text_encoder,
                prior_tokenizer=prior_tokenizer,
                prior_prior=prior_prior,
                prior_scheduler=prior_scheduler,
                prior_feature_extractor=prior_feature_extractor,
                prior_image_encoder=prior_image_encoder,
            )
            # 初始化先验管道
            self.prior_pipe = StableCascadePriorPipeline(
                prior=prior_prior,
                text_encoder=prior_text_encoder,
                tokenizer=prior_tokenizer,
                scheduler=prior_scheduler,
                image_encoder=prior_image_encoder,
                feature_extractor=prior_feature_extractor,
            )
            # 初始化解码器管道
            self.decoder_pipe = StableCascadeDecoderPipeline(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                decoder=decoder,
                scheduler=scheduler,
                vqgan=vqgan,
            )
    # 启用 xformers 的内存高效注意力机制
        def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
            # 调用解码管道以启用内存高效的注意力机制
            self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)
    
    # 启用模型的 CPU 离线加载
        def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
            r"""
            使用 accelerate 将所有模型转移到 CPU，降低内存使用，性能影响较小。与 `enable_sequential_cpu_offload` 相比，该方法在调用模型的 `forward` 方法时一次移动整个模型到 GPU，并在下一个模型运行之前保持在 GPU 中。内存节省低于 `enable_sequential_cpu_offload`，但由于 `unet` 的迭代执行，性能更好。
            """
            # 启用 CPU 离线加载到优先管道
            self.prior_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
            # 启用 CPU 离线加载到解码管道
            self.decoder_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
    
    # 启用顺序 CPU 离线加载
        def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
            r"""
            使用 🤗 Accelerate 将所有模型（`unet`、`text_encoder`、`vae` 和 `safety checker` 状态字典）转移到 CPU，显著减少内存使用。模型被移动到 `torch.device('meta')`，仅在调用其特定子模块的 `forward` 方法时加载到 GPU。离线加载是基于子模块进行的。内存节省高于使用 `enable_model_cpu_offload`，但性能较低。
            """
            # 启用顺序 CPU 离线加载到优先管道
            self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
            # 启用顺序 CPU 离线加载到解码管道
            self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
    
    # 处理进度条的显示
        def progress_bar(self, iterable=None, total=None):
            # 在优先管道中显示进度条
            self.prior_pipe.progress_bar(iterable=iterable, total=total)
            # 在解码管道中显示进度条
            self.decoder_pipe.progress_bar(iterable=iterable, total=total)
    
    # 设置进度条的配置
        def set_progress_bar_config(self, **kwargs):
            # 设置优先管道的进度条配置
            self.prior_pipe.set_progress_bar_config(**kwargs)
            # 设置解码管道的进度条配置
            self.decoder_pipe.set_progress_bar_config(**kwargs)
    
    # 禁用梯度计算以节省内存
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)
    # 定义可调用方法，允许使用多个参数进行推理
        def __call__(
            self,
            # 输入的提示，可以是字符串或字符串列表
            prompt: Optional[Union[str, List[str]]] = None,
            # 输入的图像，可以是张量或 PIL 图像，支持列表形式
            images: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]] = None,
            # 生成图像的高度，默认值为 512
            height: int = 512,
            # 生成图像的宽度，默认值为 512
            width: int = 512,
            # 推理步骤的数量，用于先验模型，默认值为 60
            prior_num_inference_steps: int = 60,
            # 先验指导尺度，控制生成的样式强度，默认值为 4.0
            prior_guidance_scale: float = 4.0,
            # 推理步骤的数量，控制图像生成的细致程度，默认值为 12
            num_inference_steps: int = 12,
            # 解码器指导尺度，影响图像的多样性，默认值为 0.0
            decoder_guidance_scale: float = 0.0,
            # 负面提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 提示嵌入，供模型使用的预计算张量
            prompt_embeds: Optional[torch.Tensor] = None,
            # 池化后的提示嵌入，增强模型的理解能力
            prompt_embeds_pooled: Optional[torch.Tensor] = None,
            # 负面提示嵌入，供模型使用的预计算张量
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 池化后的负面提示嵌入，增强模型的理解能力
            negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,
            # 每个提示生成的图像数量，默认值为 1
            num_images_per_prompt: int = 1,
            # 随机数生成器，控制生成过程的随机性
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，用于图像生成的输入张量
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认使用 PIL 图像格式
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认值为 True
            return_dict: bool = True,
            # 先验回调函数，处理每个步骤结束时的操作
            prior_callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 先验回调使用的张量输入，默认包含 'latents'
            prior_callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 回调函数，处理每个步骤结束时的操作
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 回调使用的张量输入，默认包含 'latents'
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```