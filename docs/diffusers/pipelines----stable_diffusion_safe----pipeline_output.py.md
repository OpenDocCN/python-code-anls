# `.\diffusers\pipelines\stable_diffusion_safe\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from dataclasses import dataclass
# 从 typing 模块导入 List、Optional 和 Union 类型提示，用于类型注解
from typing import List, Optional, Union

# 导入 numpy 库，常用于数值计算和处理数组
import numpy as np
# 导入 PIL.Image 模块，用于图像处理
import PIL.Image

# 从相对路径的 utils 模块中导入 BaseOutput 类，作为输出类的基类
from ...utils import (
    BaseOutput,
)


# 定义 StableDiffusionSafePipelineOutput 类，继承自 BaseOutput
@dataclass
class StableDiffusionSafePipelineOutput(BaseOutput):
    """
    Safe Stable Diffusion 管道的输出类。

    参数:
        images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            长度为 `batch_size` 的去噪 PIL 图像列表，或形状为 `(batch_size, height, width,
            num_channels)` 的 numpy 数组。PIL 图像或 numpy 数组表示扩散管道的去噪图像。
        nsfw_content_detected (`List[bool]`)
            标志列表，表示对应生成的图像是否可能代表“成人内容”
            (nsfw) 的内容，如果无法执行安全检查，则为 `None`。
        unsafe_images (`List[PIL.Image.Image]` 或 `np.ndarray`)
            被安全检查器标记的去噪 PIL 图像列表，可能包含“成人内容”
            (nsfw) 的图像，或如果未执行安全检查或未标记图像，则为 `None`。
        applied_safety_concept (`str`)
            应用的安全概念，用于安全指导，如果禁用安全指导，则为 `None`
    """

    # 定义类属性，images 可以是 PIL 图像列表或 numpy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
    # 定义可选的 nsfw_content_detected 属性，表示安全检查结果
    nsfw_content_detected: Optional[List[bool]]
    # 定义可选的 unsafe_images 属性，表示被标记为不安全的图像
    unsafe_images: Optional[Union[List[PIL.Image.Image], np.ndarray]]
    # 定义可选的 applied_safety_concept 属性，表示应用的安全概念
    applied_safety_concept: Optional[str]
```