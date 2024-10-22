# `.\diffusers\pipelines\ledits_pp\pipeline_output.py`

```py
# 从 dataclass 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入 List, Optional, 和 Union 类型提示
from typing import List, Optional, Union

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 PIL.Image 模块
import PIL.Image

# 从上级目录的 utils 模块导入 BaseOutput 类
from ...utils import BaseOutput

# 定义 LEditsPPDiffusionPipelineOutput 类，继承自 BaseOutput
@dataclass
class LEditsPPDiffusionPipelineOutput(BaseOutput):
    """
    LEdits++ Diffusion 流水线的输出类。

    参数：
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            长度为 `batch_size` 的去噪 PIL 图像列表或形状为 `(batch_size, height, width,
            num_channels)` 的 NumPy 数组。
        nsfw_content_detected (`List[bool]`)
            表示生成图像是否包含“不适合工作” (nsfw) 内容的列表，如果无法进行安全检查则为 `None`。
    """

    # 图像属性，可以是 PIL 图像列表或 NumPy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
    # NSFW 内容检测结果，可能是布尔值列表或 None
    nsfw_content_detected: Optional[List[bool]]

# 定义 LEditsPPInversionPipelineOutput 类，继承自 BaseOutput
@dataclass
class LEditsPPInversionPipelineOutput(BaseOutput):
    """
    LEdits++ Diffusion 流水线的输出类。

    参数：
        input_images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            长度为 `batch_size` 的裁剪和调整大小后的输入图像列表，或形状为 `
            (batch_size, height, width, num_channels)` 的 NumPy 数组。
        vae_reconstruction_images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            所有输入图像的 VAE 重建列表，作为长度为 `batch_size` 的 PIL 图像列表，或形状为
            ` (batch_size, height, width, num_channels)` 的 NumPy 数组。
    """

    # 输入图像属性，可以是 PIL 图像列表或 NumPy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
    # VAE 重建图像属性，可以是 PIL 图像列表或 NumPy 数组
    vae_reconstruction_images: Union[List[PIL.Image.Image], np.ndarray]
```