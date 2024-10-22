# `.\diffusers\pipelines\kandinsky2_2\pipeline_kandinsky2_2.py`

```py
# 版权声明，2024年由 HuggingFace 团队保留所有权利
# 
# 根据 Apache 许可证 2.0 版（"许可证"）进行许可；
# 除非遵守许可证，否则不得使用此文件。
# 可在以下地址获得许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有规定，软件
# 以“按原样”方式分发，不附带任何形式的保证或条件，
# 明示或暗示。
# 请参见许可证以获取与权限和
# 限制相关的具体信息。

from typing import Callable, Dict, List, Optional, Union  # 从 typing 模块导入类型提示相关类

import torch  # 导入 PyTorch 库

# 从模型模块导入 UNet2DConditionModel 和 VQModel
from ...models import UNet2DConditionModel, VQModel  
# 从调度器模块导入 DDPMScheduler
from ...schedulers import DDPMScheduler  
# 从工具模块导入实用功能
from ...utils import deprecate, logging, replace_example_docstring  
# 从工具中的 torch_utils 模块导入 randn_tensor
from ...utils.torch_utils import randn_tensor  
# 从管道工具导入 DiffusionPipeline 和 ImagePipelineOutput
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  

# 创建日志记录器，用于记录信息和错误
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，包含使用该管道的示例代码
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
        >>> import torch

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")  # 从预训练模型创建管道
        >>> pipe_prior.to("cuda")  # 将管道转移到 CUDA 设备
        >>> prompt = "red cat, 4k photo"  # 设置生成图像的提示文本
        >>> out = pipe_prior(prompt)  # 使用提示生成图像嵌入
        >>> image_emb = out.image_embeds  # 提取生成的图像嵌入
        >>> zero_image_emb = out.negative_image_embeds  # 提取负图像嵌入
        >>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")  # 创建解码器管道
        >>> pipe.to("cuda")  # 将解码器管道转移到 CUDA 设备
        >>> image = pipe(  # 使用图像嵌入生成图像
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,  # 设置生成图像的高度
        ...     width=768,  # 设置生成图像的宽度
        ...     num_inference_steps=50,  # 设置推理步骤数量
        ... ).images  # 提取生成的图像
        >>> image[0].save("cat.png")  # 保存生成的图像为 PNG 文件
        ```py
"""

# 定义一个函数，用于根据给定的高度和宽度进行缩放
def downscale_height_and_width(height, width, scale_factor=8):  # 接受高度、宽度和缩放因子
    new_height = height // scale_factor**2  # 计算新的高度
    if height % scale_factor**2 != 0:  # 检查高度是否不能被缩放因子整除
        new_height += 1  # 如果不能整除，则增加高度
    new_width = width // scale_factor**2  # 计算新的宽度
    if width % scale_factor**2 != 0:  # 检查宽度是否不能被缩放因子整除
        new_width += 1  # 如果不能整除，则增加宽度
    return new_height * scale_factor, new_width * scale_factor  # 返回缩放后的高度和宽度

# 定义 KandinskyV22Pipeline 类，继承自 DiffusionPipeline
class KandinskyV22Pipeline(DiffusionPipeline):
    """
    用于使用 Kandinsky 进行文本到图像生成的管道

    该模型继承自 [`DiffusionPipeline`]。有关库为所有管道实现的通用方法的文档，请参见父类文档（例如下载或保存，运行在特定设备等）。

    Args:
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            与 `unet` 结合使用以生成图像潜在特征的调度器。
        unet ([`UNet2DConditionModel`]):
            条件 U-Net 结构，用于去噪图像嵌入。
        movq ([`VQModel`]):
            MoVQ 解码器，用于从潜在特征生成图像。
    """
    # 定义模型的 CPU 卸载顺序，这里是将 unet 的输出移动到 q 形式
    model_cpu_offload_seq = "unet->movq"
    # 定义输入张量的名称列表
    _callback_tensor_inputs = ["latents", "image_embeds", "negative_image_embeds"]

    # 初始化方法，用于创建类的实例
    def __init__(
        # UNet2DConditionModel 实例，用于生成模型
        unet: UNet2DConditionModel,
        # DDPMScheduler 实例，用于调度过程
        scheduler: DDPMScheduler,
        # VQModel 实例，用于处理量化
        movq: VQModel,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模型组件
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        # 计算 movq 的缩放因子，基于块输出通道数
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline 复制的准备潜在变量的方法
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果没有提供潜在变量，则随机生成一个
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供的潜在变量形状不符合预期，则抛出错误
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在变量转移到指定设备
            latents = latents.to(device)

        # 将潜在变量乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 获取指导缩放因子的属性
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # 获取是否进行无分类器指导的属性，基于指导缩放因子
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # 获取时间步数的属性
    @property
    def num_timesteps(self):
        return self._num_timesteps

    # 禁用梯度计算，防止在推理过程中计算梯度
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 图像嵌入，可以是张量或张量列表
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        # 负图像嵌入，可以是张量或张量列表
        negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        # 输出图像的高度，默认 512
        height: int = 512,
        # 输出图像的宽度，默认 512
        width: int = 512,
        # 推理步骤数，默认 100
        num_inference_steps: int = 100,
        # 指导缩放因子，默认 4.0
        guidance_scale: float = 4.0,
        # 每个提示生成的图像数量，默认 1
        num_images_per_prompt: int = 1,
        # 随机数生成器，可以是单个生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 可选的潜在变量张量
        latents: Optional[torch.Tensor] = None,
        # 输出类型，默认是 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典，默认是 True
        return_dict: bool = True,
        # 结束时的回调函数，可选
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 结束时张量输入的回调名称列表，默认是 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 其他可选关键字参数
        **kwargs,
```