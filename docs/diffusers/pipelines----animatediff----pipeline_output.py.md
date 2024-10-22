# `.\diffusers\pipelines\animatediff\pipeline_output.py`

```py
# 从数据类模块导入数据类装饰器
from dataclasses import dataclass
# 导入用于类型提示的 List 和 Union
from typing import List, Union

# 导入 NumPy 库
import numpy as np
# 导入图像处理库 PIL
import PIL.Image
# 导入 PyTorch 库
import torch

# 从上级目录导入 BaseOutput 类
from ...utils import BaseOutput

# 定义 AnimateDiffPipelineOutput 数据类，继承自 BaseOutput
@dataclass
class AnimateDiffPipelineOutput(BaseOutput):
    r"""
     输出类，用于 AnimateDiff 管道。

    参数：
         frames (`torch.Tensor`, `np.ndarray` 或 List[List[PIL.Image.Image]]):
             视频输出的列表 - 可以是一个嵌套列表，长度为 `batch_size`，每个子列表包含去噪后的
             PIL 图像序列，长度为 `num_frames`。也可以是形状为
    `(batch_size, num_frames, channels, height, width)` 的 NumPy 数组或 Torch 张量。
    """

    # 定义 frames 属性，可以是不同类型的数据结构
    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
```