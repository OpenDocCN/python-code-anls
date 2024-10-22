# `.\diffusers\pipelines\ddim\pipeline_ddim.py`

```py
# 版权声明，注明该代码的版权信息及使用许可
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版进行许可（“许可证”）； 
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，按“原样”基础分发软件，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和
# 限制的具体语言。

# 从 typing 模块导入必要的类型
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从调度器模块导入 DDIMScheduler 类
from ...schedulers import DDIMScheduler
# 从工具模块导入 randn_tensor 函数
from ...utils.torch_utils import randn_tensor
# 从管道工具模块导入 DiffusionPipeline 和 ImagePipelineOutput 类
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 定义 DDIMPipeline 类，继承自 DiffusionPipeline
class DDIMPipeline(DiffusionPipeline):
    r"""
    用于图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。有关为所有管道实现的通用方法的文档，请查看超类文档（下载、保存、在特定设备上运行等）。

    参数：
        unet ([`UNet2DModel`]):
            用于去噪编码图像潜变量的 `UNet2DModel`。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用以去噪编码图像的调度器。可以是 [`DDPMScheduler`] 或 [`DDIMScheduler`] 之一。
    """

    # 定义模型在 CPU 上的卸载序列
    model_cpu_offload_seq = "unet"

    # 初始化方法，接受 unet 和 scheduler 参数
    def __init__(self, unet, scheduler):
        # 调用父类的初始化方法
        super().__init__()

        # 确保调度器始终可以转换为 DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        # 注册 unet 和 scheduler 模块
        self.register_modules(unet=unet, scheduler=scheduler)

    # 禁用梯度计算的上下文管理器
    @torch.no_grad()
    def __call__(
        # 定义批处理大小，默认为 1
        batch_size: int = 1,
        # 可选的随机数生成器，可以是单个生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # eta 参数，默认为 0.0
        eta: float = 0.0,
        # 进行推理的步骤数，默认为 50
        num_inference_steps: int = 50,
        # 可选的布尔值，用于指示是否使用裁剪的模型输出
        use_clipped_model_output: Optional[bool] = None,
        # 可选的输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 返回字典的布尔值，默认为 True
        return_dict: bool = True,
```