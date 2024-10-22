# `.\diffusers\pipelines\deprecated\latent_diffusion_uncond\pipeline_latent_diffusion_uncond.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件在“按原样”基础上分发，
# 不提供任何形式的担保或条件，明示或暗示。
# 请参阅许可证以了解有关权限和
# 限制的具体信息。

# 导入检查模块，用于获取对象的源代码或签名信息
import inspect
# 从 typing 模块导入类型注解
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从相对路径导入 VQModel 和 UNet2DModel 模型
from ....models import UNet2DModel, VQModel
# 从相对路径导入 DDIMScheduler 调度器
from ....schedulers import DDIMScheduler
# 从工具模块导入随机张量生成函数
from ....utils.torch_utils import randn_tensor
# 从管道工具模块导入扩散管道和图像输出类
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput


# 定义一个用于无条件图像生成的潜在扩散管道类
class LDMPipeline(DiffusionPipeline):
    r"""
    使用潜在扩散进行无条件图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取所有管道的通用方法（下载、保存、在特定设备上运行等）。

    参数：
        vqvae ([`VQModel`]):
            用于编码和解码图像到潜在表示的向量量化（VQ）模型。
        unet ([`UNet2DModel`]):
            用于对编码的图像潜在数据进行去噪的 `UNet2DModel`。
        scheduler ([`SchedulerMixin`]):
            [`DDIMScheduler`] 与 `unet` 一起使用以去噪编码的图像潜在数据。
    """

    # 初始化方法，用于创建 LDMPipeline 实例
    def __init__(self, vqvae: VQModel, unet: UNet2DModel, scheduler: DDIMScheduler):
        # 调用父类的初始化方法
        super().__init__()
        # 注册模型模块
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    # 禁用梯度计算，优化内存使用
    @torch.no_grad()
    def __call__(
        # 默认批次大小为 1
        self,
        batch_size: int = 1,
        # 可选的生成器，支持单个或多个生成器
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 默认的 eta 参数设置为 0.0
        eta: float = 0.0,
        # 默认推理步骤数量为 50
        num_inference_steps: int = 50,
        # 输出类型默认为 "pil"
        output_type: Optional[str] = "pil",
        # 返回字典标志，默认为 True
        return_dict: bool = True,
        # 接受额外参数
        **kwargs,
```