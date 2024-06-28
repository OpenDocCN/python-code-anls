# `.\pipelines\image_to_image.py`

```py
# 导入所需的模块和函数
from typing import List, Union

import numpy as np

# 导入通用工具函数和类
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)

# 导入基础类和函数
from .base import Pipeline, build_pipeline_init_args

# 如果视觉处理可用，则导入必要的图像处理库和函数
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果 PyTorch 可用，则导入图像到图像映射模型名称列表
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 添加文档字符串到类 ImageToImagePipeline，并设置 has_image_processor=True
@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ImageToImagePipeline(Pipeline):
    """
    Image to Image pipeline using any `AutoModelForImageToImage`. This pipeline generates an image based on a previous
    image input.

    Example:

    ```
    >>> from PIL import Image
    >>> import requests

    >>> from transformers import pipeline

    >>> upscaler = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x2-64")
    >>> img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    >>> img = img.resize((64, 64))
    >>> upscaled_img = upscaler(img)
    >>> img.size
    (64, 64)

    >>> upscaled_img.size
    (144, 144)
    ```

    This image to image pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-to-image"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=image-to-image).
    """

    # 初始化 ImageToImagePipeline 类的实例
    def __init__(self, *args, **kwargs):
        # 调用父类 Pipeline 的初始化方法
        super().__init__(*args, **kwargs)
        # 检查是否有必要的后端支持（这里是视觉处理）
        requires_backends(self, "vision")
        # 检查模型类型是否为图像到图像映射模型
        self.check_model_type(MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)

    # 清理参数的辅助函数，返回预处理、前向和后处理参数字典
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}

        # 如果传入了超时参数，则设置到预处理参数中
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        # 如果传入了头部遮罩参数，则设置到前向参数中
        if "head_mask" in kwargs:
            forward_params["head_mask"] = kwargs["head_mask"]

        return preprocess_params, forward_params, postprocess_params

    # 调用实例时的方法，接收图像或图像列表作为输入
    def __call__(
        self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs
    ) -> Union["Image.Image", List["Image.Image"]]:
        """
        Transform the image(s) passed as inputs.

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
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and
                the call may block forever.

        Return:
            An image (Image.Image) or a list of images (List["Image.Image"]) containing result(s). If the input is a
            single image, the return will be also a single image, if the input is a list of several images, it will
            return a list of transformed images.
        """
        # 调用父类的 __call__ 方法来处理输入的图像数据
        return super().__call__(images, **kwargs)

    def _forward(self, model_inputs):
        # 使用模型进行前向推断，获取模型输出
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def preprocess(self, image, timeout=None):
        # 加载图像并进行预处理，将图像转换为模型接受的输入格式
        image = load_image(image, timeout=timeout)
        # 使用图像处理器处理预处理后的图像，返回模型输入
        inputs = self.image_processor(images=[image], return_tensors="pt")
        return inputs

    def postprocess(self, model_outputs):
        # 初始化空列表来存储后处理后的图像
        images = []
        # 如果模型输出中包含 "reconstruction" 键
        if "reconstruction" in model_outputs.keys():
            # 获取重建的输出
            outputs = model_outputs.reconstruction
        # 遍历每个输出
        for output in outputs:
            # 将输出数据转换为浮点数并在 CPU 上进行操作
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            # 调整输出的轴顺序，将通道放到最后一个维度
            output = np.moveaxis(output, source=0, destination=-1)
            # 将浮点数转换为 uint8 类型的像素值
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            # 将处理后的像素值数组转换为 PIL 图像并添加到 images 列表中
            images.append(Image.fromarray(output))

        # 如果 images 列表中有多个图像，则返回图像列表；否则返回单个图像
        return images if len(images) > 1 else images[0]
```