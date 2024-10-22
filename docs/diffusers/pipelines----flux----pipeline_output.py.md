# `.\diffusers\pipelines\flux\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入 List 和 Union 类型注解
from typing import List, Union

# 导入 numpy 库并简化为 np
import numpy as np
# 导入 PIL.Image 模块用于图像处理
import PIL.Image

# 从上级目录的 utils 模块导入 BaseOutput 类
from ...utils import BaseOutput


# 定义 FluxPipelineOutput 类，继承自 BaseOutput
@dataclass
class FluxPipelineOutput(BaseOutput):
    """
    Stable Diffusion 流水线的输出类。

    Args:
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            长度为 `batch_size` 的去噪 PIL 图像列表或形状为 `(batch_size, height, width,
            num_channels)` 的 numpy 数组。PIL 图像或 numpy 数组表示扩散流水线的去噪图像。
    """

    # 定义 images 属性，可以是 PIL 图像列表或 numpy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
```