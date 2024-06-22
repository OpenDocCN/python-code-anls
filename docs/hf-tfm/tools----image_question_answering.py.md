# `.\transformers\tools\image_question_answering.py`

```py
#!/usr/bin/env python
# coding=utf-8

# 导入必要的模块和库
from typing import TYPE_CHECKING
import torch

# 导入自定义模块
from ..models.auto import AutoModelForVisualQuestionAnswering, AutoProcessor
from ..utils import requires_backends
from .base import PipelineTool

# 类型检查
if TYPE_CHECKING:
    from PIL import Image

# 定义图像问答工具类
class ImageQuestionAnsweringTool(PipelineTool):
    # 默认检查点
    default_checkpoint = "dandelin/vilt-b32-finetuned-vqa"
    # 描述
    description = (
        "This is a tool that answers a question about an image. It takes an input named `image` which should be the "
        "image containing the information, as well as a `question` which should be the question in English. It "
        "returns a text that is the answer to the question."
    )
    # 工具名称
    name = "image_qa"
    # 预处理器类
    pre_processor_class = AutoProcessor
    # 模型类
    model_class = AutoModelForVisualQuestionAnswering

    # 输入和输出的类型
    inputs = ["image", "text"]
    outputs = ["text"]

    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 检查后端
        requires_backends(self, ["vision"])
        super().__init__(*args, **kwargs)

    # 编码方法
    def encode(self, image: "Image", question: str):
        # 调用预处理器
        return self.pre_processor(image, question, return_tensors="pt")

    # 前向传播方法
    def forward(self, inputs):
        # 禁用梯度追踪
        with torch.no_grad():
            # 使用模型进行推理
            return self.model(**inputs).logits

    # 解码方法
    def decode(self, outputs):
        # 获取最高得分的索引
        idx = outputs.argmax(-1).item()
        # 返回对应的标签
        return self.model.config.id2label[idx]
```