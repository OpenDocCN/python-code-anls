# `.\diffusers\pipelines\cogvideo\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass

# 导入 PyTorch 库
import torch

# 从 diffusers.utils 导入 BaseOutput 基类
from diffusers.utils import BaseOutput


# 定义 CogVideoXPipelineOutput 类，继承自 BaseOutput
@dataclass
class CogVideoXPipelineOutput(BaseOutput):
    r"""
    CogVideo 管道的输出类。

    参数:
        frames (`torch.Tensor`, `np.ndarray`, 或 List[List[PIL.Image.Image]]):
            视频输出的列表 - 可以是长度为 `batch_size` 的嵌套列表，每个子列表包含
            去噪的 PIL 图像序列，长度为 `num_frames`。也可以是形状为
            `(batch_size, num_frames, channels, height, width)` 的 NumPy 数组或 Torch 张量。
    """

    # 定义输出的帧，类型为 torch.Tensor
    frames: torch.Tensor
```