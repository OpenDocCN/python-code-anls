# `.\diffusers\pipelines\deprecated\alt_diffusion\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入 List、Optional 和 Union 类型
from typing import List, Optional, Union

# 导入 numpy 库并简写为 np
import numpy as np
# 导入 PIL.Image 模块
import PIL.Image

# 从上级模块导入 BaseOutput 类
from ....utils import (
    BaseOutput,
)


@dataclass
# 注释说明：从 diffusers.pipelines.stable_diffusion.pipeline_output 导入的类，Stable 替换为 Alt
class AltDiffusionPipelineOutput(BaseOutput):
    """
    Alt Diffusion 管道的输出类。

    参数：
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            包含去噪的 PIL 图像的列表，长度为 `batch_size` 或形状为 `(batch_size, height, width,
            num_channels)` 的 NumPy 数组。
        nsfw_content_detected (`List[bool]`)
            列表，指示对应生成的图像是否包含 "不安全内容" (nsfw)，
            如果无法进行安全检查，则为 `None`。
    """

    # 定义 images 属性，类型为列表或 NumPy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
    # 定义 nsfw_content_detected 属性，类型为可选列表
    nsfw_content_detected: Optional[List[bool]]
```