# `.\diffusers\pipelines\kandinsky2_2\pipeline_kandinsky2_2_controlnet.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，
# 根据许可证分发的软件均按“原样”提供，
# 不附有任何明示或暗示的担保或条件。
# 有关许可证所管理权限和限制的具体内容，请参阅许可证。

from typing import Callable, List, Optional, Union  # 从 typing 模块导入可调用、列表、可选和联合类型

import torch  # 导入 PyTorch 库

from ...models import UNet2DConditionModel, VQModel  # 从模型模块导入 UNet2DConditionModel 和 VQModel 类
from ...schedulers import DDPMScheduler  # 从调度器模块导入 DDPMScheduler 类
from ...utils import (  # 从 utils 模块导入 logging 工具
    logging,
)
from ...utils.torch_utils import randn_tensor  # 从 torch_utils 导入 randn_tensor 函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从 pipeline_utils 导入 DiffusionPipeline 和 ImagePipelineOutput 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 pylint 命名检查
    # 示例代码，展示如何使用深度估计和图像生成管道
    Examples:
        ```py
        # 导入 PyTorch 和 NumPy 库
        >>> import torch
        >>> import numpy as np
    
        # 导入 Kandinsky V22 模型的管道和其他工具
        >>> from diffusers import KandinskyV22PriorPipeline, KandinskyV22ControlnetPipeline
        >>> from transformers import pipeline
        >>> from diffusers.utils import load_image
    
        # 定义生成深度提示的函数
        >>> def make_hint(image, depth_estimator):
            # 使用深度估计器获取图像的深度信息
            ...     image = depth_estimator(image)["depth"]
            # 将深度信息转换为 NumPy 数组
            ...     image = np.array(image)
            # 为深度图像增加一个维度
            ...     image = image[:, :, None]
            # 将深度图像复制三次，形成 RGB 格式
            ...     image = np.concatenate([image, image, image], axis=2)
            # 将 NumPy 数组转换为 PyTorch 张量并归一化
            ...     detected_map = torch.from_numpy(image).float() / 255.0
            # 调整张量的维度顺序
            ...     hint = detected_map.permute(2, 0, 1)
            # 返回生成的提示
            ...     return hint
    
        # 创建深度估计器管道
        >>> depth_estimator = pipeline("depth-estimation")
    
        # 加载 Kandinsky V22 先验管道，并指定数据类型为 float16
        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        # 将管道移动到 CUDA 设备
        >>> pipe_prior = pipe_prior.to("cuda")
    
        # 加载 Kandinsky V22 控制管道，并指定数据类型为 float16
        >>> pipe = KandinskyV22ControlnetPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype=torch.float16
        ... )
        # 将管道移动到 CUDA 设备
        >>> pipe = pipe.to("cuda")
    
        # 从 URL 加载图像并调整大小
        >>> img = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... ).resize((768, 768))
    
        # 生成深度提示，并调整张量维度和数据类型
        >>> hint = make_hint(img, depth_estimator).unsqueeze(0).half().to("cuda")
    
        # 定义生成图像的提示内容
        >>> prompt = "A robot, 4k photo"
        # 定义负提示内容，用于排除不希望出现的特征
        >>> negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
    
        # 创建随机数生成器并设置种子
        >>> generator = torch.Generator(device="cuda").manual_seed(43)
    
        # 使用先验管道生成图像嵌入和零图像嵌入
        >>> image_emb, zero_image_emb = pipe_prior(
        ...     prompt=prompt, negative_prompt=negative_prior_prompt, generator=generator
        ... ).to_tuple()
    
        # 使用控制管道生成最终图像
        >>> images = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     hint=hint,
        ...     num_inference_steps=50,
        ...     generator=generator,
        ...     height=768,
        ...     width=768,
        ... ).images
    
        # 保存生成的图像
        >>> images[0].save("robot_cat.png")
        ```py  
"""
# 文档字符串，说明该模块的功能和用途

# 从 diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2.downscale_height_and_width 复制的函数
def downscale_height_and_width(height, width, scale_factor=8):
    # 根据给定的高度和宽度以及缩放因子计算新的高度
    new_height = height // scale_factor**2
    # 如果高度不能被缩放因子平方整除，则增加高度
    if height % scale_factor**2 != 0:
        new_height += 1
    # 根据给定的高度和宽度以及缩放因子计算新的宽度
    new_width = width // scale_factor**2
    # 如果宽度不能被缩放因子平方整除，则增加宽度
    if width % scale_factor**2 != 0:
        new_width += 1
    # 返回新的高度和宽度，乘以缩放因子以恢复到原始比例
    return new_height * scale_factor, new_width * scale_factor


class KandinskyV22ControlnetPipeline(DiffusionPipeline):
    """
    文档字符串，描述使用 Kandinsky 进行文本到图像生成的管道

    该模型继承自 [`DiffusionPipeline`]。查看超类文档以获取库实现的通用方法（例如下载或保存、在特定设备上运行等）

    参数：
        scheduler ([`DDIMScheduler`]):
            用于与 `unet` 结合生成图像潜变量的调度器。
        unet ([`UNet2DConditionModel`]):
            用于去噪图像嵌入的条件 U-Net 架构。
        movq ([`VQModel`]):
            MoVQ 解码器，用于从潜变量生成图像。
    """

    # 定义模型中 CPU 卸载的顺序
    model_cpu_offload_seq = "unet->movq"

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        # 调用父类构造函数进行初始化
        super().__init__()

        # 注册模块以便在管道中使用
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        # 计算 MoVQ 的缩放因子
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    # 从 diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents 复制的函数
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果未提供潜变量，则生成随机潜变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供的潜变量形状不匹配，则抛出异常
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜变量移动到指定设备
            latents = latents.to(device)

        # 将潜变量乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜变量
        return latents

    @torch.no_grad()
    def __call__(
        self,
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        hint: torch.Tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
```