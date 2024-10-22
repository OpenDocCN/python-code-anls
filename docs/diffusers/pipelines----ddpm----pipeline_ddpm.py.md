# `.\diffusers\pipelines\ddpm\pipeline_ddpm.py`

```py
# 版权声明，表明版权所有者和版权年份
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可声明，说明根据 Apache 2.0 许可证使用该文件的条件
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 说明如何获取许可证的链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 免责声明，声明在适用法律下软件以“按现状”方式分发，不提供任何形式的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 说明许可证的具体条款和条件
# See the License for the specific language governing permissions and
# limitations under the License.

# 从 typing 模块导入需要的类型注解
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从 utils.torch_utils 模块导入 randn_tensor 函数
from ...utils.torch_utils import randn_tensor
# 从 pipeline_utils 模块导入 DiffusionPipeline 和 ImagePipelineOutput 类
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 定义 DDPMPipeline 类，继承自 DiffusionPipeline 类
class DDPMPipeline(DiffusionPipeline):
    r"""
    用于图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解为所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    参数:
        unet ([`UNet2DModel`]):
            用于去噪编码图像潜变量的 `UNet2DModel`。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于去噪编码图像。可以是
            [`DDPMScheduler`] 或 [`DDIMScheduler`] 的其中之一。
    """

    # 定义一个类变量，表示 CPU 卸载的模型序列
    model_cpu_offload_seq = "unet"

    # 初始化方法，接受 unet 和 scheduler 作为参数
    def __init__(self, unet, scheduler):
        # 调用父类的初始化方法
        super().__init__()
        # 注册 unet 和 scheduler 模块
        self.register_modules(unet=unet, scheduler=scheduler)

    # 使用装饰器，表示该方法在执行时不需要计算梯度
    @torch.no_grad()
    def __call__(
        # 定义批量大小，默认为 1
        batch_size: int = 1,
        # 可选参数，生成器，可以是单个或多个 torch.Generator 实例
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 定义推理步骤的数量，默认为 1000
        num_inference_steps: int = 1000,
        # 定义输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 定义是否返回字典格式的输出，默认为 True
        return_dict: bool = True,
```