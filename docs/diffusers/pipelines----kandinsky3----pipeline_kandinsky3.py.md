# `.\diffusers\pipelines\kandinsky3\pipeline_kandinsky3.py`

```py
# 从 typing 模块导入类型注解工具
from typing import Callable, Dict, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 T5 编码器模型和分词器
from transformers import T5EncoderModel, T5Tokenizer

# 从本地模块导入混合加载器
from ...loaders import StableDiffusionLoraLoaderMixin
# 从本地模块导入 Kandinsky3UNet 和 VQModel
from ...models import Kandinsky3UNet, VQModel
# 从本地模块导入 DDPMScheduler
from ...schedulers import DDPMScheduler
# 从本地模块导入工具函数
from ...utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
# 从本地工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从本地管道工具模块导入 DiffusionPipeline 和 ImagePipelineOutput
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 创建一个记录日志的 logger 实例，名称为当前模块名
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示如何使用该模块的功能
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import AutoPipelineForText2Image
        >>> import torch

        >>> pipe = AutoPipelineForText2Image.from_pretrained(
        ...     "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

        >>> generator = torch.Generator(device="cpu").manual_seed(0)
        >>> image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
        ```py

"""

# 定义 downscale_height_and_width 函数，用于缩放高度和宽度
def downscale_height_and_width(height, width, scale_factor=8):
    # 计算缩放后的新高度
    new_height = height // scale_factor**2
    # 如果高度不能被缩放因子平方整除，则向上取整
    if height % scale_factor**2 != 0:
        new_height += 1
    # 计算缩放后的新宽度
    new_width = width // scale_factor**2
    # 如果宽度不能被缩放因子平方整除，则向上取整
    if width % scale_factor**2 != 0:
        new_width += 1
    # 返回缩放后的高度和宽度
    return new_height * scale_factor, new_width * scale_factor

# 定义 Kandinsky3Pipeline 类，继承自 DiffusionPipeline 和 StableDiffusionLoraLoaderMixin
class Kandinsky3Pipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):
    # 定义模型 CPU 卸载的顺序
    model_cpu_offload_seq = "text_encoder->unet->movq"
    # 定义需要回调的张量输入列表
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "negative_attention_mask",
        "attention_mask",
    ]

    # 初始化方法，接收多个模型组件作为参数
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: Kandinsky3UNet,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 注册模型组件
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, unet=unet, scheduler=scheduler, movq=movq
        )

    # 定义处理嵌入的函数
    def process_embeds(self, embeddings, attention_mask, cut_context):
        # 如果需要裁剪上下文
        if cut_context:
            # 将 attention_mask 为 0 的嵌入置为零
            embeddings[attention_mask == 0] = torch.zeros_like(embeddings[attention_mask == 0])
            # 计算最大序列长度
            max_seq_length = attention_mask.sum(-1).max() + 1
            # 裁剪嵌入和 attention_mask
            embeddings = embeddings[:, :max_seq_length]
            attention_mask = attention_mask[:, :max_seq_length]
        # 返回处理后的嵌入和 attention_mask
        return embeddings, attention_mask

    # 禁用梯度计算，以节省内存
    @torch.no_grad()
    # 定义一个用于编码提示的函数
    def encode_prompt(
        # 提示内容
        self,
        prompt,
        # 是否进行无分类器引导
        do_classifier_free_guidance=True,
        # 每个提示生成的图像数量
        num_images_per_prompt=1,
        # 设备类型（CPU/GPU）
        device=None,
        # 负面提示内容
        negative_prompt=None,
        # 提示的嵌入张量（可选）
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负面提示的嵌入张量（可选）
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 是否截断上下文
        _cut_context=False,
        # 注意力掩码（可选）
        attention_mask: Optional[torch.Tensor] = None,
        # 负面提示的注意力掩码（可选）
        negative_attention_mask: Optional[torch.Tensor] = None,
    # 准备潜在变量的函数
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果没有潜在变量，则随机生成
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查潜在变量的形状是否与预期匹配
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在变量移动到指定设备
            latents = latents.to(device)

        # 通过调度器的初始噪声标准差缩放潜在变量
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 检查输入有效性的函数
    def check_inputs(
        # 提示内容
        self,
        prompt,
        # 回调步骤
        callback_steps,
        # 负面提示内容（可选）
        negative_prompt=None,
        # 提示的嵌入张量（可选）
        prompt_embeds=None,
        # 负面提示的嵌入张量（可选）
        negative_prompt_embeds=None,
        # 结束时的回调张量输入（可选）
        callback_on_step_end_tensor_inputs=None,
        # 注意力掩码（可选）
        attention_mask=None,
        # 负面提示的注意力掩码（可选）
        negative_attention_mask=None,
    # 返回引导缩放因子的属性
    @property
    def guidance_scale(self):
        # 返回内部引导缩放因子的值
        return self._guidance_scale

    # 返回是否进行无分类器引导的属性
    @property
    def do_classifier_free_guidance(self):
        # 判断引导缩放因子是否大于 1
        return self._guidance_scale > 1

    # 返回时间步数的属性
    @property
    def num_timesteps(self):
        # 返回内部时间步数的值
        return self._num_timesteps

    # 禁用梯度计算的调用函数
    @torch.no_grad()
    # 替换示例文档字符串的装饰器
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 提示内容（可以是字符串或字符串列表）
        self,
        prompt: Union[str, List[str]] = None,
        # 推理步骤的数量
        num_inference_steps: int = 25,
        # 引导缩放因子的值
        guidance_scale: float = 3.0,
        # 负面提示内容（可选）
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量（可选）
        num_images_per_prompt: Optional[int] = 1,
        # 图像高度（可选）
        height: Optional[int] = 1024,
        # 图像宽度（可选）
        width: Optional[int] = 1024,
        # 随机生成器（可选）
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 提示的嵌入张量（可选）
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负面提示的嵌入张量（可选）
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 注意力掩码（可选）
        attention_mask: Optional[torch.Tensor] = None,
        # 负面提示的注意力掩码（可选）
        negative_attention_mask: Optional[torch.Tensor] = None,
        # 输出类型（默认为 PIL 格式）
        output_type: Optional[str] = "pil",
        # 是否返回字典格式的结果（默认为 True）
        return_dict: bool = True,
        # 潜在变量（可选）
        latents=None,
        # 步骤结束时的回调函数（可选）
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 结束时的回调张量输入（默认值为 ["latents"]）
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 其他关键字参数
        **kwargs,
```