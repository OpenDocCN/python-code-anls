# `.\pipelines\depth_estimation.py`

```
# 从 typing 模块导入 List 和 Union 类型
from typing import List, Union

# 导入 numpy 库并使用 np 别名
import numpy as np

# 从 ..utils 模块导入指定函数和类
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends

# 从 .base 模块导入 Pipeline 类和 build_pipeline_init_args 函数
from .base import Pipeline, build_pipeline_init_args

# 如果 torch 可用，则执行条件语句块
if is_torch_available():
    # 导入 torch 库
    import torch

    # 从 ..models.auto.modeling_auto 模块导入 MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES 常量
    from ..models.auto.modeling_auto import MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES

# 使用 logging 模块获取名为 __name__ 的 logger 对象
logger = logging.get_logger(__name__)

# 使用装饰器 add_end_docstrings 构造类 DepthEstimationPipeline，继承自 Pipeline 类
@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class DepthEstimationPipeline(Pipeline):
    """
    Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
    >>> output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
    >>> # This is a tensor with the values being the depth expressed in meters for each pixel
    >>> output["predicted_depth"].shape
    torch.Size([1, 384, 384])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This depth estimation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"depth-estimation"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=depth-estimation).
    """

    # 构造函数，初始化对象
    def __init__(self, *args, **kwargs):
        # 调用父类 Pipeline 的构造函数进行初始化
        super().__init__(*args, **kwargs)
        # 要求视觉后端库可用，否则引发异常
        requires_backends(self, "vision")
        # 检查模型类型是否匹配 MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES 中的映射名称
        self.check_model_type(MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)
    # 调用函数，用于预测输入图像的深度
    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        Predict the depth(s) of the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **predicted_depth** (`torch.Tensor`) -- The predicted depth by the model as a `torch.Tensor`.
            - **depth** (`PIL.Image`) -- The predicted depth by the model as a `PIL.Image`.
        """
        return super().__call__(images, **kwargs)

    # 将超时参数转换为预处理参数字典
    def _sanitize_parameters(self, timeout=None, **kwargs):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        return preprocess_params, {}, {}

    # 对输入图像进行预处理，返回模型输入
    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout)  # 加载图像
        self.image_size = image.size  # 记录图像尺寸
        model_inputs = self.image_processor(images=image, return_tensors=self.framework)  # 图像处理并返回模型输入
        return model_inputs

    # 对模型输入进行前向传播，返回模型输出
    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)  # 模型前向传播
        return model_outputs

    # 对模型输出进行后处理，生成预测深度图和格式化后的深度图像
    def postprocess(self, model_outputs):
        predicted_depth = model_outputs.predicted_depth  # 提取预测深度
        # 对预测深度进行双三次插值，并转换为 PIL 图像
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=self.image_size[::-1], mode="bicubic", align_corners=False
        )
        output = prediction.squeeze().cpu().numpy()  # 去除多余维度并转为 numpy 数组
        formatted = (output * 255 / np.max(output)).astype("uint8")  # 格式化深度图像
        depth = Image.fromarray(formatted)  # 转换为 PIL 图像
        output_dict = {}
        output_dict["predicted_depth"] = predicted_depth  # 存储预测深度张量
        output_dict["depth"] = depth  # 存储格式化后的深度图像
        return output_dict
```