# `.\transformers\tools\image_segmentation.py`

```
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# 版权归 The HuggingFace Inc. 团队所有。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0 进行许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得基于许可证分发软件
# 软件按"原样"分发，不附带任何保证或条件，无论是明示或暗示的。
# 有关特定语言管理权限和限制，请参阅许可证。
# 导入依赖库
import numpy as np
import torch

# 从本地导入模块
from ..models.clipseg import CLIPSegForImageSegmentation
from ..utils import is_vision_available, requires_backends
from .base import PipelineTool

# 检查视觉库是否可用，如果可用则导入 PIL 库
if is_vision_available():
    from PIL import Image

# 定义图像分割工具类
class ImageSegmentationTool(PipelineTool):
    # 工具描述
    description = (
        "This is a tool that creates a segmentation mask of an image according to a label. It cannot create an image. "
        "It takes two arguments named `image` which should be the original image, and `label` which should be a text "
        "describing the elements what should be identified in the segmentation mask. The tool returns the mask."
    )
    # 默认检查点
    default_checkpoint = "CIDAS/clipseg-rd64-refined"
    # 工具名称
    name = "image_segmenter"
    # 模型类
    model_class = CLIPSegForImageSegmentation

    # 输入和输出参数
    inputs = ["image", "text"]
    outputs = ["image"]

    # 构造函数
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
        super().__init__(*args, **kwargs)

    # 编码方法，将图像和标签编码成模型输入
    def encode(self, image: "Image", label: str):
        return self.pre_processor(text=[label], images=[image], padding=True, return_tensors="pt")

    # 前向传播，获取模型输出
    def forward(self, inputs):
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits

    # 解码方法，将模型输出解码成图像
    def decode(self, outputs):
        array = outputs.cpu().detach().numpy()
        array[array <= 0] = 0
        array[array > 0] = 1
        return Image.fromarray((array * 255).astype(np.uint8))
```