# `.\transformers\pipelines\image_to_image.py`

```py
# 版权声明及许可证信息
# 版权声明
# 许可证：Apache许可证2.0
# 获取许可证的方式
# 若非法律要求或书面同意，不得使用该文件
# 在"原样"的基础上分发软件，不附加任何保证或条件，无论明示或暗示
# 请参见许可证，获取特定语言的权限和限制


# 导入模块
from typing import List, Union
import numpy as np
# 导入自定义工具包中的模块
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import PIPELINE_INIT_ARGS, Pipeline

# 如果视觉可用，导入PIL模块
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

# 如果 Torch 可用，导入自动图像到图像映射模型
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES

# 获取logger对象
logger = logging.get_logger(__name__)

# 使用装饰器添加端到端文档注释
@add_end_docstrings(PIPELINE_INIT_ARGS)

# 定义ImageToImagePipeline类，继承自Pipeline类
class ImageToImagePipeline(Pipeline):
    """
    图像到图像的管道，使用任何`AutoModelForImageToImage`。该管道基于前一个图像输入生成一个图像。

    示例:

    ```python
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
    ```py

    目前可以使用该图像到图像管道从[`pipeline`]加载，使用以下任务标识符：
    `"image-to-image"`。

    查看[huggingface.co/models](https://huggingface.co/models?filter=image-to-image)上可用模型的列表。
    """

    # 初始化函数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 确保必要后端可用，否则引发异常
        requires_backends(self, "vision")
        # 检查模型类型是否匹配图像到图像映射模型
        self.check_model_type(MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)

    # 参数清理函数
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}

        # 如果参数中包含"timeout"，则设置预处理参数"timeout"
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        # 如果参数中包含"head_mask"，则设置前向参数"head_mask"
        if "head_mask" in kwargs:
            forward_params["head_mask"] = kwargs["head_mask"]

        return preprocess_params, forward_params, postprocess_params

    # 调用函数
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
        # 调用父类的方法，并返回结果
        return super().__call__(images, **kwargs)

    def _forward(self, model_inputs):
        # 使用模型处理输入，并返回模型输出
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def preprocess(self, image, timeout=None):
        # 加载图像，并将其处理为模型输入
        image = load_image(image, timeout=timeout)
        inputs = self.image_processor(images=[image], return_tensors="pt")
        return inputs

    def postprocess(self, model_outputs):
        images = []
        # 若模型输出中包含"reconstruction"字段
        if "reconstruction" in model_outputs.keys():
            # 取出"reconstruction"字段的值
            outputs = model_outputs.reconstruction
        for output in outputs:
            # 对输出进行后处理，将其转换为图像
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.moveaxis(output, source=0, destination=-1)
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            images.append(Image.fromarray(output))

        # 若输出多张图像，则返回图像列表；若只有一张图像，则返回单张图像
        return images if len(images) > 1 else images[0]
```