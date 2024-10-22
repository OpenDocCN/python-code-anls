# `.\diffusers\pipelines\amused\pipeline_amused.py`

```py
# 版权声明，表明该文件的版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0（"许可证"）授权；
# 除非遵循许可证，否则不得使用此文件。
# 您可以在以下地址获得许可证副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件
# 在许可证下分发的，均按"原样"提供，没有任何明示或暗示的担保或条件。
# 请参阅许可证，以了解有关权限和
# 限制的具体条款。

# 从 typing 模块导入所需的类型提示
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库中导入 CLIP 文本模型和标记器
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# 从当前目录的上级导入 VAE 图像处理器
from ...image_processor import VaeImageProcessor
# 从当前目录的上级导入 UVit2D 模型和 VQ 模型
from ...models import UVit2DModel, VQModel
# 从当前目录的上级导入 Amused 调度器
from ...schedulers import AmusedScheduler
# 从当前目录的上级导入替换示例文档字符串的工具
from ...utils import replace_example_docstring
# 从上级目录的管道工具导入 DiffusionPipeline 和 ImagePipelineOutput
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 示例文档字符串，提供使用示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AmusedPipeline

        >>> pipe = AmusedPipeline.from_pretrained("amused/amused-512", variant="fp16", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```py
"""

# 定义 AmusedPipeline 类，继承自 DiffusionPipeline
class AmusedPipeline(DiffusionPipeline):
    # 声明图像处理器属性
    image_processor: VaeImageProcessor
    # 声明 VQ 模型属性
    vqvae: VQModel
    # 声明标记器属性
    tokenizer: CLIPTokenizer
    # 声明文本编码器属性
    text_encoder: CLIPTextModelWithProjection
    # 声明转换器属性
    transformer: UVit2DModel
    # 声明调度器属性
    scheduler: AmusedScheduler

    # 定义 CPU 预加载顺序
    model_cpu_offload_seq = "text_encoder->transformer->vqvae"

    # 初始化方法，接收各个组件作为参数
    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: UVit2DModel,
        scheduler: AmusedScheduler,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个组件
        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
        # 计算 VAE 缩放因子
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        # 初始化图像处理器
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

    # 装饰器，表示该方法不需要计算梯度
    @torch.no_grad()
    # 使用替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法，允许使用不同参数生成结果
    def __call__(
        # 用户输入的提示，可以是字符串或字符串列表，默认为 None
        self,
        prompt: Optional[Union[List[str], str]] = None,
        # 输出图像的高度，默认为 None
        height: Optional[int] = None,
        # 输出图像的宽度，默认为 None
        width: Optional[int] = None,
        # 推理步骤的数量，默认为 12
        num_inference_steps: int = 12,
        # 指导比例，影响生成图像与提示的一致性，默认为 10.0
        guidance_scale: float = 10.0,
        # 负提示，可以是字符串或字符串列表，默认为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: Optional[int] = 1,
        # 用于生成的随机数生成器，默认为 None
        generator: Optional[torch.Generator] = None,
        # 潜在空间的张量，默认为 None
        latents: Optional[torch.IntTensor] = None,
        # 提示的嵌入张量，默认为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 编码器的隐藏状态，默认为 None
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 负提示的嵌入张量，默认为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 负编码器的隐藏状态，默认为 None
        negative_encoder_hidden_states: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"
        output_type="pil",
        # 是否返回字典格式的输出，默认为 True
        return_dict: bool = True,
        # 回调函数，用于在生成过程中执行特定操作，默认为 None
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调执行的步骤间隔，默认为 1
        callback_steps: int = 1,
        # 跨注意力的额外参数，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 微调条件的美学评分，默认为 6
        micro_conditioning_aesthetic_score: int = 6,
        # 微调条件的裁剪坐标，默认为 (0, 0)
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        # 温度参数，用于控制生成多样性，默认为 (2, 0)
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
```