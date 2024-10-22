# `.\diffusers\pipelines\text_to_video_synthesis\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入 List 和 Union 类型
from typing import List, Union

# 导入 numpy 库并简写为 np
import numpy as np
# 导入 PIL 库，用于图像处理
import PIL
# 导入 PyTorch 库
import torch

# 从上级模块的 utils 导入 BaseOutput 类
from ...utils import (
    BaseOutput,
)

# 定义一个数据类 TextToVideoSDPipelineOutput 继承自 BaseOutput
@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
     文本到视频管道的输出类。

    参数:
         frames (`torch.Tensor`, `np.ndarray` 或 List[List[PIL.Image.Image]]):
             视频输出的列表 - 可以是长度为 `batch_size` 的嵌套列表，
             每个子列表包含去噪后的
             PIL 图像序列，长度为 `num_frames`。也可以是形状为
    `(batch_size, num_frames, channels, height, width)` 的 NumPy 数组或 Torch 张量
    """

    # 定义一个 frames 属性，可以是 Torch 张量、NumPy 数组或嵌套的 PIL 图像列表
    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
```