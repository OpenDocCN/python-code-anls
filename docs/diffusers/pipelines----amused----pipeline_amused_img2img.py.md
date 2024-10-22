# `.\diffusers\pipelines\amused\pipeline_amused_img2img.py`

```py
# 版权信息，声明该文件的版权所有者和许可证信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 使用 Apache License, Version 2.0（“许可证”）进行授权；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件在许可证下以“按现状”基础分发，
# 不附带任何类型的明示或暗示的担保或条件。
# 有关许可证下的具体权限和限制，请参见许可证。

# 从 typing 模块导入类型注释
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIP 相关模型和标记器
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# 从本地模块导入图像处理相关的类
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 从本地模块导入模型类
from ...models import UVit2DModel, VQModel
# 从本地模块导入调度器类
from ...schedulers import AmusedScheduler
# 从本地模块导入文档字符串替换工具
from ...utils import replace_example_docstring
# 从本地模块导入扩散管道和图像输出相关类
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 示例文档字符串，用于展示如何使用该管道的示例代码
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AmusedImg2ImgPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = AmusedImg2ImgPipeline.from_pretrained(
        ...     "amused/amused-512", variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "winter mountains"
        >>> input_image = (
        ...     load_image(
        ...         "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains.jpg"
        ...     )
        ...     .resize((512, 512))
        ...     .convert("RGB")
        ... )
        >>> image = pipe(prompt, input_image).images[0]
        ```py
"""

# 定义 AmusedImg2ImgPipeline 类，继承自 DiffusionPipeline
class AmusedImg2ImgPipeline(DiffusionPipeline):
    # 声明类属性，表示图像处理器
    image_processor: VaeImageProcessor
    # 声明类属性，表示 VQ 模型
    vqvae: VQModel
    # 声明类属性，表示 CLIP 标记器
    tokenizer: CLIPTokenizer
    # 声明类属性，表示 CLIP 文本编码器
    text_encoder: CLIPTextModelWithProjection
    # 声明类属性，表示变换模型
    transformer: UVit2DModel
    # 声明类属性，表示调度器
    scheduler: AmusedScheduler

    # 定义模型 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->transformer->vqvae"

    # TODO - 处理 self.vqvae.quantize 时使用 self.vqvae.quantize.embedding.weight，
    # 该调用在 self.vqvae.quantize 的前向方法之前，因此钩子不会被调用以将参数
    # 从 meta 设备移除。需要找到解决方法，而不是仅仅不卸载它
    _exclude_from_cpu_offload = ["vqvae"]

    # 初始化方法，接收多个模型和调度器作为参数
    def __init__(
        self,
        # VQ 模型
        vqvae: VQModel,
        # CLIP 标记器
        tokenizer: CLIPTokenizer,
        # CLIP 文本编码器
        text_encoder: CLIPTextModelWithProjection,
        # UVit 变换模型
        transformer: UVit2DModel,
        # Amused 调度器
        scheduler: AmusedScheduler,
    # 初始化父类
        ):
            super().__init__()
    
            # 注册多个模块以供使用
            self.register_modules(
                vqvae=vqvae,  # 注册变分量化自编码器
                tokenizer=tokenizer,  # 注册分词器
                text_encoder=text_encoder,  # 注册文本编码器
                transformer=transformer,  # 注册变换器
                scheduler=scheduler,  # 注册调度器
            )
            # 计算 VAE 的缩放因子，基于块输出通道数
            self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
            # 创建图像处理器实例，未进行归一化处理
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)
    
        # 禁止梯度计算以节省内存和加快推理
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 接收可选的提示，支持单个字符串或字符串列表
            prompt: Optional[Union[List[str], str]] = None,
            # 可选的输入图像
            image: PipelineImageInput = None,
            # 强度参数，控制生成图像的混合程度
            strength: float = 0.5,
            # 推理步骤的数量
            num_inference_steps: int = 12,
            # 引导比例，影响生成图像的风格
            guidance_scale: float = 10.0,
            # 可选的负面提示
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量
            num_images_per_prompt: Optional[int] = 1,
            # 可选的随机数生成器
            generator: Optional[torch.Generator] = None,
            # 可选的提示嵌入
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面编码器隐藏状态
            negative_encoder_hidden_states: Optional[torch.Tensor] = None,
            # 输出类型，默认为 PIL 格式
            output_type="pil",
            # 是否返回字典格式的输出
            return_dict: bool = True,
            # 可选的回调函数，处理中间结果
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调的步骤间隔
            callback_steps: int = 1,
            # 可选的交叉注意力参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 微调条件的美学评分，默认为 6
            micro_conditioning_aesthetic_score: int = 6,
            # 微调条件的裁剪坐标，默认为 (0, 0)
            micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
            # 温度参数，控制生成的多样性
            temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
```