# `.\diffusers\pipelines\shap_e\pipeline_shap_e_img2img.py`

```py
# 版权声明，说明该文件的版权所有者及其权利
# Copyright 2024 Open AI and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证，版本 2.0（"许可证"）进行许可；
# 除非遵守许可证，否则不得使用此文件。
# 可在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件
# 是在“按原样”基础上分发的，没有任何明示或暗示的担保或条件。
# 有关许可证的具体权限和限制，请参见许可证。
#
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入 List、Optional 和 Union 类型
from typing import List, Optional, Union

# 导入 numpy 库并简化为 np
import numpy as np
# 导入 PIL.Image 模块以处理图像
import PIL.Image
# 导入 torch 库用于深度学习
import torch
# 从 transformers 库中导入 CLIP 图像处理器和 CLIP 视觉模型
from transformers import CLIPImageProcessor, CLIPVisionModel

# 从本地模型导入 PriorTransformer 类
from ...models import PriorTransformer
# 从调度器模块导入 HeunDiscreteScheduler 类
from ...schedulers import HeunDiscreteScheduler
# 从 utils 模块导入多个实用工具
from ...utils import (
    BaseOutput,         # 基类输出
    logging,           # 日志记录
    replace_example_docstring,  # 替换示例文档字符串的函数
)
# 从 torch_utils 模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从 pipeline_utils 模块导入 DiffusionPipeline 类
from ..pipeline_utils import DiffusionPipeline
# 从 renderer 模块导入 ShapERenderer 类
from .renderer import ShapERenderer

# 创建一个日志记录器，用于记录当前模块的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，展示了如何使用 DiffusionPipeline
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from PIL import Image  # 导入 PIL 图像处理库
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import DiffusionPipeline  # 导入 DiffusionPipeline 类
        >>> from diffusers.utils import export_to_gif, load_image  # 导入实用函数

        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测可用设备（CUDA或CPU）

        >>> repo = "openai/shap-e-img2img"  # 定义模型的仓库名称
        >>> pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)  # 从预训练模型加载管道
        >>> pipe = pipe.to(device)  # 将管道转移到指定设备上

        >>> guidance_scale = 3.0  # 设置引导比例
        >>> image_url = "https://hf.co/datasets/diffusers/docs-images/resolve/main/shap-e/corgi.png"  # 定义图像 URL
        >>> image = load_image(image_url).convert("RGB")  # 加载并转换图像为 RGB 格式

        >>> images = pipe(  # 调用管道生成图像
        ...     image,  # 输入图像
        ...     guidance_scale=guidance_scale,  # 使用的引导比例
        ...     num_inference_steps=64,  # 推理步骤数量
        ...     frame_size=256,  # 设置帧大小
        ... ).images  # 获取生成的图像列表

        >>> gif_path = export_to_gif(images[0], "corgi_3d.gif")  # 将生成的第一张图像导出为 GIF
        ```py
"""

# 定义 ShapEPipelineOutput 类，继承自 BaseOutput，用于表示输出
@dataclass
class ShapEPipelineOutput(BaseOutput):
    """
    [`ShapEPipeline`] 和 [`ShapEImg2ImgPipeline`] 的输出类。

    Args:
        images (`torch.Tensor`)  # 图像的张量列表，用于 3D 渲染
            A list of images for 3D rendering.
    """

    # 定义输出的图像属性，可以是 PIL.Image.Image 或 np.ndarray 类型
    images: Union[PIL.Image.Image, np.ndarray]


# 定义 ShapEImg2ImgPipeline 类，继承自 DiffusionPipeline
class ShapEImg2ImgPipeline(DiffusionPipeline):
    """
    从图像生成 3D 资产的潜在表示并使用 NeRF 方法进行渲染的管道。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。
    # 参数说明
    Args:
        prior ([`PriorTransformer`]):
            用于近似文本嵌入生成图像嵌入的标准 unCLIP 先验。
        image_encoder ([`~transformers.CLIPVisionModel`]):
            冻结的图像编码器。
        image_processor ([`~transformers.CLIPImageProcessor`]):
             用于处理图像的 `CLIPImageProcessor`。
        scheduler ([`HeunDiscreteScheduler`]):
            与 `prior` 模型结合使用以生成图像嵌入的调度器。
        shap_e_renderer ([`ShapERenderer`]):
            Shap-E 渲染器将生成的潜在变量投影为 MLP 参数，以使用 NeRF 渲染方法创建 3D 对象。
    """

    # 定义模型在 CPU 上卸载的顺序
    model_cpu_offload_seq = "image_encoder->prior"
    # 定义在 CPU 卸载中排除的模块
    _exclude_from_cpu_offload = ["shap_e_renderer"]

    # 初始化方法
    def __init__(
        self,
        prior: PriorTransformer,  # 定义 prior 参数
        image_encoder: CLIPVisionModel,  # 定义图像编码器参数
        image_processor: CLIPImageProcessor,  # 定义图像处理器参数
        scheduler: HeunDiscreteScheduler,  # 定义调度器参数
        shap_e_renderer: ShapERenderer,  # 定义 Shap-E 渲染器参数
    ):
        super().__init__()  # 调用父类初始化方法

        # 注册模块以供后续使用
        self.register_modules(
            prior=prior,  # 注册 prior 模块
            image_encoder=image_encoder,  # 注册图像编码器模块
            image_processor=image_processor,  # 注册图像处理器模块
            scheduler=scheduler,  # 注册调度器模块
            shap_e_renderer=shap_e_renderer,  # 注册 Shap-E 渲染器模块
        )

    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents 复制的准备潜在变量方法
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 检查潜在变量是否为 None
        if latents is None:
            # 生成随机潜在变量
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查潜在变量形状是否与预期一致
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在变量移动到指定设备
            latents = latents.to(device)

        # 将潜在变量与调度器的初始噪声标准差相乘
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 定义编码图像的方法
    def _encode_image(
        self,
        image,  # 输入图像
        device,  # 指定设备
        num_images_per_prompt,  # 每个提示的图像数量
        do_classifier_free_guidance,  # 是否进行无分类器引导
    ):
        # 检查输入的 image 是否是一个列表且列表的第一个元素是 torch.Tensor
        if isinstance(image, List) and isinstance(image[0], torch.Tensor):
            # 如果第一个元素的维度是4，使用 torch.cat 沿着第0维连接，否则使用 torch.stack 叠加
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

        # 如果 image 不是 torch.Tensor 类型，进行处理
        if not isinstance(image, torch.Tensor):
            # 使用图像处理器处理 image，将结果转换为 PyTorch 张量，并提取 pixel_values 的第一个元素
            image = self.image_processor(image, return_tensors="pt").pixel_values[0].unsqueeze(0)

        # 将 image 转换为指定的数据类型和设备（CPU或GPU）
        image = image.to(dtype=self.image_encoder.dtype, device=device)

        # 使用图像编码器对图像进行编码，获取最后隐藏状态
        image_embeds = self.image_encoder(image)["last_hidden_state"]
        # 取出最后隐藏状态的切片，忽略第一个维度，并确保内存连续
        image_embeds = image_embeds[:, 1:, :].contiguous()  # batch_size, dim, 256

        # 对图像嵌入进行重复扩展，以适应每个提示的图像数量
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        # 如果需要进行无分类器引导
        if do_classifier_free_guidance:
            # 创建与 image_embeds 形状相同的零张量作为负样本嵌入
            negative_image_embeds = torch.zeros_like(image_embeds)

            # 对于无分类器引导，我们需要进行两次前向传播
            # 在这里将无条件和文本嵌入拼接成一个批次，以避免进行两次前向传播
            image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 返回最终的图像嵌入
        return image_embeds

    # 禁用梯度计算，以提高推理速度并节省内存
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 定义可调用函数的输入参数，支持 PIL 图像或图像列表
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        # 每个提示生成的图像数量，默认为1
        num_images_per_prompt: int = 1,
        # 推理步骤的数量，默认为25
        num_inference_steps: int = 25,
        # 随机数生成器，可以是单个生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在变量的张量，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 引导比例，默认为4.0
        guidance_scale: float = 4.0,
        # 帧的大小，默认为64
        frame_size: int = 64,
        # 输出类型，可选：'pil'、'np'、'latent'、'mesh'
        output_type: Optional[str] = "pil",  # pil, np, latent, mesh
        # 是否返回字典格式的结果，默认为 True
        return_dict: bool = True,
```