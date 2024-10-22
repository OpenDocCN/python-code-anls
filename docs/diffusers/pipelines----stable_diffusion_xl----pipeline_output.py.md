# `.\diffusers\pipelines\stable_diffusion_xl\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from dataclasses import dataclass
# 从 typing 模块导入 List 和 Union 类型提示
from typing import List, Union

# 导入 numpy 库，通常用于数值计算
import numpy as np
# 导入 PIL.Image 模块，用于处理图像
import PIL.Image

# 从上级模块导入 BaseOutput 类和 is_flax_available 函数
from ...utils import BaseOutput, is_flax_available


# 定义一个数据类，用于存储 Stable Diffusion 管道的输出
@dataclass
class StableDiffusionXLPipelineOutput(BaseOutput):
    """
    Stable Diffusion 管道的输出类。

    参数:
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            包含去噪后的 PIL 图像的列表，长度为 `batch_size`，或形状为 `(batch_size, height, width,
            num_channels)` 的 numpy 数组。PIL 图像或 numpy 数组表示扩散管道的去噪图像。
    """

    # 定义一个属性 images，可以是 PIL 图像列表或 numpy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]


# 检查是否可用 Flax 库
if is_flax_available():
    # 导入 flax 库，通常用于深度学习框架
    import flax

    # 定义一个数据类，用于存储 Flax Stable Diffusion XL 管道的输出
    @flax.struct.dataclass
    class FlaxStableDiffusionXLPipelineOutput(BaseOutput):
        """
        Flax Stable Diffusion XL 管道的输出类。

        参数:
            images (`np.ndarray`)
                形状为 `(batch_size, height, width, num_channels)` 的数组，包含来自扩散管道的图像。
        """

        # 定义一个属性 images，类型为 numpy 数组
        images: np.ndarray
```