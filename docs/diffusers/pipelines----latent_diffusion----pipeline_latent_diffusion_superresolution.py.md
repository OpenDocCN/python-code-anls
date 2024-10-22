# `.\diffusers\pipelines\latent_diffusion\pipeline_latent_diffusion_superresolution.py`

```py
# 导入 inspect 模块以便于获取有关对象的信息
import inspect
# 从 typing 模块导入 List, Optional, Tuple 和 Union 类型
from typing import List, Optional, Tuple, Union

# 导入 numpy 库，通常用于数组和矩阵操作
import numpy as np
# 导入 PIL.Image 库，用于图像处理
import PIL.Image
# 导入 PyTorch 库，用于深度学习
import torch
# 导入 PyTorch 的检查点工具，用于内存优化
import torch.utils.checkpoint

# 从上层模块导入 UNet2DModel 和 VQModel 模型
from ...models import UNet2DModel, VQModel
# 从上层模块导入不同的调度器
from ...schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
# 从上层模块导入 PIL_INTERPOLATION 用于图像插值
from ...utils import PIL_INTERPOLATION
# 从上层模块导入 randn_tensor 用于生成随机张量
from ...utils.torch_utils import randn_tensor
# 从上层模块导入 DiffusionPipeline 和 ImagePipelineOutput
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


# 定义一个图像预处理函数
def preprocess(image):
    # 获取图像的宽度和高度
    w, h = image.size
    # 将宽度和高度调整为32的整数倍
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    # 使用兰索斯插值法调整图像大小
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    # 将图像转换为 numpy 数组并归一化到 [0, 1] 范围
    image = np.array(image).astype(np.float32) / 255.0
    # 增加一个维度并调整数组的轴顺序
    image = image[None].transpose(0, 3, 1, 2)
    # 将 numpy 数组转换为 PyTorch 张量
    image = torch.from_numpy(image)
    # 将图像值范围从 [0, 1] 转换到 [-1, 1]
    return 2.0 * image - 1.0


# 定义一个用于图像超分辨率的潜在扩散管道类
class LDMSuperResolutionPipeline(DiffusionPipeline):
    r"""
    使用潜在扩散进行图像超分辨率的管道。

    该模型继承自 [`DiffusionPipeline`]。有关所有管道实现的通用方法的文档（下载、保存、在特定设备上运行等），请查看超类文档。

    参数：
        vqvae ([`VQModel`]):
            用于将图像编码和解码为潜在表示的矢量量化（VQ）模型。
        unet ([`UNet2DModel`]):
            用于去噪编码图像的 `UNet2DModel`。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用的调度器，以去噪编码图像的潜在表示。可以是以下之一：
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerDiscreteScheduler`],
            [`EulerAncestralDiscreteScheduler`], [`DPMSolverMultistepScheduler`] 或 [`PNDMScheduler`]。
    """

    # 初始化方法，设置 VQ 模型、UNet 模型和调度器
    def __init__(
        self,
        vqvae: VQModel,
        unet: UNet2DModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 注册 VQ 模型、UNet 模型和调度器
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    # 在不计算梯度的情况下调用方法
    @torch.no_grad()
    def __call__(
        self,
        # 输入图像，可以是 PyTorch 张量或 PIL 图像
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        # 批处理大小的可选参数，默认为 1
        batch_size: Optional[int] = 1,
        # 可选的推理步骤数，默认为 100
        num_inference_steps: Optional[int] = 100,
        # 可选的 eta 值，默认为 0.0
        eta: Optional[float] = 0.0,
        # 可选的随机生成器，可以是单个生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 输出类型的可选参数，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典形式的结果，默认为 True
        return_dict: bool = True,
```