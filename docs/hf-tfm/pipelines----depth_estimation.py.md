# `.\transformers\pipelines\depth_estimation.py`

```py
# 从 typing 模块导入 List 和 Union 类型提示
from typing import List, Union

# 导入 numpy 库并用别名 np 引用
import numpy as np

# 从 ..utils 模块中导入 add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends 函数
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
# 从 .base 模块中导入 PIPELINE_INIT_ARGS, Pipeline 类
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果 is_vision_available() 返回 True，则导入 PIL 库中的 Image 类和 ..image_utils 模块中的 load_image 函数
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果 is_torch_available() 返回 True，则导入 torch 库
if is_torch_available():
    import torch
    # 从 ..models.auto.modeling_auto 模块中导入 MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES 字典
    from ..models.auto.modeling_auto import MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 使用 @add_end_docstrings(PIPELINE_INIT_ARGS) 装饰器来添加文档字符串
class DepthEstimationPipeline(Pipeline):
    """
    Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
    >>> output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
    >>> # This is a tensor with the values being the depth expressed in meters for each pixel
    >>> output["predicted_depth"].shape
    torch.Size([1, 384, 384])
    ```py

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This depth estimation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"depth-estimation"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=depth-estimation).
    """

    # 初始化方法，接受任意位置和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类 Pipeline 的初始化方法
        super().__init__(*args, **kwargs)
        # 检查当前 Pipeline 对象的后端是否包含 vision
        requires_backends(self, "vision")
        # 检查当前 Pipeline 对象的模型类型是否在 MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES 中
        self.check_model_type(MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)
    # 定义一个方法，将标签分配给传入的图片
    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        为传入的图像赋予标签。

        参数:
            images (str, List[str], PIL.Image 或 List[PIL.Image]):
                该管道处理三种类型的图像:

                - 包含指向图像的 http 链接的字符串
                - 包含指向图像的本地路径的字符串
                - 直接在 PIL 中加载的图像

                该管道接受单个图像或一批图像，后者必须以字符串的形式传递。
                一批图像必须都是相同格式的: 为全部 http 链接，全部本地路径，或全部是 PIL 图像。
            top_k (int, 可选, 默认为 5):
                该管道将返回的前几个标签数。如果提供的数字高于模型配置中的标签数量，它将默认为标签数量。
            timeout (float, 可选, 默认为 None):
                从网络获取图像的最长等待时间，单位为秒。如果为 None，表示没有设置超时，调用可能会一直阻塞。

        返回:
            包含结果的字典或字典列表。如果输入是单个图像，则返回一个字典；如果输入是一组图像，则返回对应图像的字典列表。

            字典包含以下键:

            - **label** (str) -- 模型识别的标签。
            - **score** (int) -- 模型为该标签评定的分数。
        """
        return super().__call__(images, **kwargs)

    # 处理参数，确保超时参数传递到预处理参数中
    def _sanitize_parameters(self, timeout=None, **kwargs):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        return preprocess_params, {}, {}

    # 对图像进行预处理，加载图像并获取模型输入
    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout)
        self.image_size = image.size
        model_inputs = self.image_processor(images=image, return_tensors=self.framework)
        return model_inputs

    # 将模型输入传递给模型进行前向传播
    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    # 对模型输出进行后处理，将预测深度格式化为图像并返回
    def postprocess(self, model_outputs):
        predicted_depth = model_outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=self.image_size[::-1], mode="bicubic", align_corners=False
        )
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        output_dict = {}
        output_dict["predicted_depth"] = predicted_depth
        output_dict["depth"] = depth
        return output_dict
```