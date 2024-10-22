# `.\diffusers\pipelines\stable_diffusion_3\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入 List 和 Union 类型提示
from typing import List, Union

# 导入 numpy 库并简化为 np
import numpy as np
# 导入 PIL.Image 模块，用于处理图像
import PIL.Image

# 从上级模块的 utils 中导入 BaseOutput 类
from ...utils import BaseOutput


# 定义一个数据类 StableDiffusion3PipelineOutput，继承自 BaseOutput
@dataclass
class StableDiffusion3PipelineOutput(BaseOutput):
    """
    Stable Diffusion 管道的输出类。

    参数:
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            长度为 `batch_size` 的去噪 PIL 图像列表或形状为 `(batch_size, height, width,
            num_channels)` 的 numpy 数组。PIL 图像或 numpy 数组表示扩散管道的去噪图像。
    """

    # 定义一个属性 images，类型为列表或 numpy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
```