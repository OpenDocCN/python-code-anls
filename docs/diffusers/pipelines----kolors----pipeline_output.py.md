# `.\diffusers\pipelines\kolors\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from dataclasses import dataclass
# 从 typing 模块导入 List 和 Union 类型，用于类型注解
from typing import List, Union

# 导入 numpy 库，通常用于数组操作
import numpy as np
# 导入 PIL.Image 模块，用于处理图像
import PIL.Image

# 从上级模块导入 BaseOutput 类，可能是用于输出的基类
from ...utils import BaseOutput

# 定义 KolorsPipelineOutput 类，继承自 BaseOutput
@dataclass
class KolorsPipelineOutput(BaseOutput):
    """
    Kolors 管道输出类。

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            图像列表，包含去噪后的 PIL 图像，长度为 `batch_size` 或形状为 `(batch_size, height, width,
            num_channels)` 的 numpy 数组。PIL 图像或 numpy 数组表示扩散管道的去噪图像。
    """

    # 定义 images 属性，可以是 PIL 图像列表或 numpy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
```