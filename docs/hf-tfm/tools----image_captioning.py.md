# `.\tools\image_captioning.py`

```py
#!/usr/bin/env python
# coding=utf-8

# 导入 TYPE_CHECKING 模块，用于静态类型检查
from typing import TYPE_CHECKING

# 导入 AutoModelForVision2Seq 类，用于视觉到序列任务的自动模型加载
from ..models.auto import AutoModelForVision2Seq
# 导入 requires_backends 函数，用于检查所需的后端库是否安装
from ..utils import requires_backends
# 导入 PipelineTool 基类，作为工具类的基础类
from .base import PipelineTool

# 如果 TYPE_CHECKING 为 True，导入 Image 类
if TYPE_CHECKING:
    from PIL import Image

# 定义 ImageCaptioningTool 类，继承自 PipelineTool 基类
class ImageCaptioningTool(PipelineTool):
    # 默认的模型检查点路径
    default_checkpoint = "Salesforce/blip-image-captioning-base"
    # 工具描述信息，生成图像描述的工具
    description = (
        "This is a tool that generates a description of an image. It takes an input named `image` which should be the "
        "image to caption, and returns a text that contains the description in English."
    )
    # 工具名称
    name = "image_captioner"
    # 模型类
    model_class = AutoModelForVision2Seq

    # 输入要求，图像输入
    inputs = ["image"]
    # 输出要求，文本输出
    outputs = ["text"]

    # 初始化方法，检查视觉后端的必要性
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
        super().__init__(*args, **kwargs)

    # 编码方法，将图像进行编码
    def encode(self, image: "Image"):
        return self.pre_processor(images=image, return_tensors="pt")

    # 前向推理方法，生成描述文本
    def forward(self, inputs):
        return self.model.generate(**inputs)

    # 解码方法，解析生成的文本结果
    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
```