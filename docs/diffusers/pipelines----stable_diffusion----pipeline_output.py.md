# `.\diffusers\pipelines\stable_diffusion\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入 List, Optional, Union 类型注解
from typing import List, Optional, Union

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 PIL.Image 模块
import PIL.Image

# 从上层模块导入 BaseOutput 和 is_flax_available 函数
from ...utils import BaseOutput, is_flax_available


# 定义一个数据类，作为 Stable Diffusion 管道的输出
@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Stable Diffusion 管道的输出类。

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            包含去噪后的 PIL 图像列表，长度为 `batch_size`，或形状为 `(batch_size, height, width,
            num_channels)` 的 NumPy 数组。
        nsfw_content_detected (`List[bool]`)
            指示对应生成图像是否包含“不可安全观看” (nsfw) 内容的列表，若无法进行安全检查则为 `None`。
    """

    # 存储图像，类型为 PIL 图像列表或 NumPy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
    # 存储 nsfw 内容检测结果的可选列表
    nsfw_content_detected: Optional[List[bool]]


# 检查是否可用 Flax 库
if is_flax_available():
    # 导入 flax 库
    import flax

    # 定义一个数据类，作为 Flax 基于 Stable Diffusion 管道的输出
    @flax.struct.dataclass
    class FlaxStableDiffusionPipelineOutput(BaseOutput):
        """
        Flax 基于 Stable Diffusion 管道的输出类。

        Args:
            images (`np.ndarray`):
                形状为 `(batch_size, height, width, num_channels)` 的去噪图像数组。
            nsfw_content_detected (`List[bool]`):
                指示对应生成图像是否包含“不可安全观看” (nsfw) 内容的列表，
                或 `None` 如果无法进行安全检查。
        """

        # 存储图像，类型为 NumPy 数组
        images: np.ndarray
        # 存储 nsfw 内容检测结果的列表
        nsfw_content_detected: List[bool]
```