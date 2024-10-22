# `.\diffusers\pipelines\semantic_stable_diffusion\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入 List、Optional 和 Union 类型提示
from typing import List, Optional, Union

# 导入 numpy 库并简写为 np
import numpy as np
# 导入 PIL.Image 模块，用于处理图像
import PIL.Image

# 从上层目录的 utils 模块导入 BaseOutput 基类
from ...utils import BaseOutput


# 定义一个数据类 SemanticStableDiffusionPipelineOutput，继承自 BaseOutput
@dataclass
class SemanticStableDiffusionPipelineOutput(BaseOutput):
    """
    Stable Diffusion 流水线的输出类。

    参数:
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            包含去噪后 PIL 图像的列表，长度为 `batch_size`，或形状为 `(batch_size, height, width,
            num_channels)` 的 NumPy 数组。
        nsfw_content_detected (`List[bool]`)
            列表，指示相应生成的图像是否包含“非安全内容”（nsfw），
            如果无法执行安全检查，则为 `None`。
    """

    # 定义 images 属性，可以是 PIL 图像列表或 NumPy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
    # 定义 nsfw_content_detected 属性，表示安全检查结果，类型为可选布尔列表
    nsfw_content_detected: Optional[List[bool]]
```