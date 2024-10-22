# `.\diffusers\pipelines\kandinsky\pipeline_kandinsky.py`

```py
# 版权声明，标识该文件的版权信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 使用 Apache License, Version 2.0 进行许可
# 只有在遵守许可证的情况下，您才能使用此文件
# 许可证的副本可以在以下网址获取
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，
# 否则根据许可证分发的软件是“按原样”提供的，
# 不附带任何明示或暗示的担保或条件
# 有关许可证的具体条款，请参阅许可证文档

# 导入类型提示相关的模块
from typing import Callable, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库中导入 XLMRobertaTokenizer
from transformers import (
    XLMRobertaTokenizer,
)

# 导入自定义模型和调度器
from ...models import UNet2DConditionModel, VQModel
from ...schedulers import DDIMScheduler, DDPMScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .text_encoder import MultilingualCLIP

# 获取日志记录器实例，供后续使用
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用该管道
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyPipeline, KandinskyPriorPipeline
        >>> import torch

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained("kandinsky-community/Kandinsky-2-1-prior")
        >>> pipe_prior.to("cuda")

        >>> prompt = "red cat, 4k photo"
        >>> out = pipe_prior(prompt)
        >>> image_emb = out.image_embeds
        >>> negative_image_emb = out.negative_image_embeds

        >>> pipe = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1")
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     prompt,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=negative_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=100,
        ... ).images

        >>> image[0].save("cat.png")
        ```py
"""

# 定义函数，根据输入的高度和宽度计算新的高度和宽度
def get_new_h_w(h, w, scale_factor=8):
    # 计算新的高度，向下取整除以 scale_factor^2
    new_h = h // scale_factor**2
    # 如果 h 不能被 scale_factor^2 整除，则高度加 1
    if h % scale_factor**2 != 0:
        new_h += 1
    # 计算新的宽度，向下取整除以 scale_factor^2
    new_w = w // scale_factor**2
    # 如果 w 不能被 scale_factor^2 整除，则宽度加 1
    if w % scale_factor**2 != 0:
        new_w += 1
    # 返回新的高度和宽度，乘以 scale_factor
    return new_h * scale_factor, new_w * scale_factor

# 定义 KandinskyPipeline 类，继承自 DiffusionPipeline
class KandinskyPipeline(DiffusionPipeline):
    """
    用于使用 Kandinsky 进行文本到图像生成的管道

    此模型继承自 [`DiffusionPipeline`]。有关所有管道实现的通用方法（例如下载、保存、在特定设备上运行等）的文档，请查看超类文档。
    # 函数参数说明
    Args:
        text_encoder ([`MultilingualCLIP`]):  # 冻结的文本编码器
            Frozen text-encoder.
        tokenizer ([`XLMRobertaTokenizer`]):  # 类的分词器
            Tokenizer of class
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):  # 用于与 `unet` 结合生成图像潜在变量的调度器
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):  # 用于去噪图像嵌入的条件 U-Net 架构
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):  # MoVQ 解码器，用于从潜在变量生成图像
            MoVQ Decoder to generate the image from the latents.
    """

    # 定义模型的 CPU 离线加载顺序
    model_cpu_offload_seq = "text_encoder->unet->movq"

    # 初始化函数，设置模型的各个组件
    def __init__(
        self,
        text_encoder: MultilingualCLIP,  # 文本编码器
        tokenizer: XLMRobertaTokenizer,  # 分词器
        unet: UNet2DConditionModel,  # U-Net 模型
        scheduler: Union[DDIMScheduler, DDPMScheduler],  # 调度器
        movq: VQModel,  # MoVQ 解码器
    ):
        super().__init__()  # 调用父类初始化函数

        # 注册模型模块
        self.register_modules(
            text_encoder=text_encoder,  # 注册文本编码器
            tokenizer=tokenizer,  # 注册分词器
            unet=unet,  # 注册 U-Net 模型
            scheduler=scheduler,  # 注册调度器
            movq=movq,  # 注册 MoVQ 解码器
        )
        # 计算 MoVQ 的缩放因子
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline 复制的函数，用于准备潜在变量
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果潜在变量为空，则随机生成潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查潜在变量形状是否符合预期
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在变量移动到指定设备
            latents = latents.to(device)

        # 乘以调度器的初始化噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 编码提示信息
    def _encode_prompt(
        self,
        prompt,  # 提示文本
        device,  # 设备类型
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器引导
        negative_prompt=None,  # 可选的负提示文本
    ):
    @torch.no_grad()  # 在不计算梯度的情况下执行
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 替换示例文档字符串
    # 调用函数，生成图像
    def __call__(
        self,
        prompt: Union[str, List[str]],  # 提示文本，可以是单个字符串或字符串列表
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],  # 图像嵌入，可以是张量或张量列表
        negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],  # 负图像嵌入
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 可选的负提示文本
        height: int = 512,  # 生成图像的高度
        width: int = 512,  # 生成图像的宽度
        num_inference_steps: int = 100,  # 推理步骤的数量
        guidance_scale: float = 4.0,  # 引导比例
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 可选的随机数生成器
        latents: Optional[torch.Tensor] = None,  # 可选的潜在变量
        output_type: Optional[str] = "pil",  # 输出类型，默认为 PIL 图像
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 可选的回调函数
        callback_steps: int = 1,  # 每隔多少步骤调用一次回调
        return_dict: bool = True,  # 是否返回字典格式的结果
```