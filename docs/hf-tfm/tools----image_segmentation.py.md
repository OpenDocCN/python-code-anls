# `.\tools\image_segmentation.py`

```py
# 导入必要的库和模块
import numpy as np
import torch

# 从自定义的模块中导入CLIPSegForImageSegmentation模型
from ..models.clipseg import CLIPSegForImageSegmentation
# 从自定义的模块中导入必要的工具函数和类
from ..utils import is_vision_available, requires_backends
# 从本地模块中导入基础工具类PipelineTool
from .base import PipelineTool

# 如果视觉功能可用，则导入PIL库中的Image类
if is_vision_available():
    from PIL import Image

# 定义一个图像分割工具类，继承自PipelineTool基类
class ImageSegmentationTool(PipelineTool):
    # 工具描述信息
    description = (
        "This is a tool that creates a segmentation mask of an image according to a label. It cannot create an image. "
        "It takes two arguments named `image` which should be the original image, and `label` which should be a text "
        "describing the elements what should be identified in the segmentation mask. The tool returns the mask."
    )
    # 默认的模型检查点路径
    default_checkpoint = "CIDAS/clipseg-rd64-refined"
    # 工具名称
    name = "image_segmenter"
    # 使用的模型类
    model_class = CLIPSegForImageSegmentation

    # 输入参数列表
    inputs = ["image", "text"]
    # 输出参数列表
    outputs = ["image"]

    # 初始化方法，检查视觉后端支持
    def __init__(self, *args, **kwargs):
        # 检查并确保视觉后端可用
        requires_backends(self, ["vision"])
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    # 编码方法，将图像和标签转换为模型输入格式
    def encode(self, image: "Image", label: str):
        # 使用预处理器处理文本和图像，返回PyTorch张量
        return self.pre_processor(text=[label], images=[image], padding=True, return_tensors="pt")

    # 前向传播方法，执行模型推理
    def forward(self, inputs):
        # 使用无梯度计算环境执行模型推理，获取logits
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits

    # 解码方法，将模型输出转换为图像
    def decode(self, outputs):
        # 将输出张量转换为NumPy数组
        array = outputs.cpu().detach().numpy()
        # 将数组中小于等于0的值设为0，大于0的值设为1
        array[array <= 0] = 0
        array[array > 0] = 1
        # 将数组转换为PIL图像，并返回
        return Image.fromarray((array * 255).astype(np.uint8))
```