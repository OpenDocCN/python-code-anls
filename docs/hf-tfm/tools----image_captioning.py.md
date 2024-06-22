# `.\transformers\tools\image_captioning.py`

```py
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# 2023 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可以
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律要求或书面同意，否则根据许可证分发的软件
# 均以“原样”分发，无任何明示或暗示的担保或条件。
# 有关特定语言对权限和限制的详细信息，请参阅许可证。
from typing import TYPE_CHECKING

from ..models.auto import AutoModelForVision2Seq
from ..utils import requires_backends
from .base import PipelineTool

# 检查类型是否仅为检查目的
if TYPE_CHECKING:
    from PIL import Image


class ImageCaptioningTool(PipelineTool):
    # 默认检查点
    default_checkpoint = "Salesforce/blip-image-captioning-base"
    description = (
        "This is a tool that generates a description of an image. It takes an input named `image` which should be the "
        "image to caption, and returns a text that contains the description in English."
    )
    name = "image_captioner"
    model_class = AutoModelForVision2Seq

    # 输入输出
    inputs = ["image"]
    outputs = ["text"]

    # 初始化方法
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
        super().__init__(*args, **kwargs)

    # 编码方法
    def encode(self, image: "Image"):
        return self.pre_processor(images=image, return_tensors="pt")

    # 前向传播方法
    def forward(self, inputs):
        return self.model.generate(**inputs)

    # 解码方法
    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
```