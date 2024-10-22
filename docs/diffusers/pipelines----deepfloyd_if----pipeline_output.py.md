# `.\diffusers\pipelines\deepfloyd_if\pipeline_output.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from dataclasses import dataclass
# 从 typing 模块导入类型注解，用于类型提示
from typing import List, Optional, Union

# 导入 numpy 库，通常用于数组操作和数值计算
import numpy as np
# 导入 PIL.Image，用于处理图像
import PIL.Image

# 从上级模块导入 BaseOutput 基类，用于输出类的继承
from ...utils import BaseOutput


# 定义 IFPipelineOutput 类，继承自 BaseOutput
@dataclass
class IFPipelineOutput(BaseOutput):
    """
    Args:
    Output class for Stable Diffusion pipelines.
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content or a watermark. `None` if safety checking could not be performed.
        watermark_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely has a watermark. `None` if safety
            checking could not be performed.
    """

    # 定义 images 属性，可以是 PIL 图像列表或 numpy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
    # 定义 nsfw_detected 属性，可选的布尔列表，用于标记是否检测到不安全内容
    nsfw_detected: Optional[List[bool]]
    # 定义 watermark_detected 属性，可选的布尔列表，用于标记是否检测到水印
    watermark_detected: Optional[List[bool]]
```