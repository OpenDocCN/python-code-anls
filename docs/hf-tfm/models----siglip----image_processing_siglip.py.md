# `.\models\siglip\image_processing_siglip.py`

```
# 导入所需模块和类
from typing import Dict, List, Optional, Union

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,         # 导入图像处理所需的标准均值
    IMAGENET_STANDARD_STD,          # 导入图像处理所需的标准标准差
    ChannelDimension,               # 导入通道维度相关的枚举
    ImageInput,                     # 导入图像输入的类型定义
    PILImageResampling,             # 导入 PIL 图像的重采样方式枚举
    infer_channel_dimension_format, # 推断通道维度格式的函数
    is_scaled_image,                # 判断图像是否已经缩放的函数
    make_list_of_images,            # 将输入转换为图像列表的函数
    to_numpy_array,                 # 将图像转换为 NumPy 数组的函数
    valid_images,                   # 检查图像有效性的函数
    validate_kwargs,                # 验证关键字参数的函数
    validate_preprocess_arguments,  # 验证预处理参数的函数
)
from ...utils import TensorType, is_vision_available, logging  # 导入张量类型、可视化库是否可用的函数和日志记录器


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


if is_vision_available():  # 如果可视化库可用
    import PIL  # 导入 PIL 库用于图像处理


class SiglipImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SigLIP image processor.
    """
    # 定义一个类，用于预处理图像数据以供模型输入使用
    class ImagePreprocessing:
        # 模型的输入名称列表
        model_input_names = ["pixel_values"]
    
        # 初始化方法，设置各种图像预处理参数的默认值，并允许通过关键字参数进一步覆盖
        def __init__(
            self,
            do_resize: bool = True,  # 是否调整图像大小的标志，默认为True
            size: Dict[str, int] = None,  # 调整大小后的图像尺寸，默认为{"height": 224, "width": 224}
            resample: PILImageResampling = PILImageResampling.BICUBIC,  # 调整图像大小时使用的重采样滤波器，默认为BICUBIC
            do_rescale: bool = True,  # 是否对图像进行重新缩放的标志，默认为True
            rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放图像时的缩放因子，默认为1/255
            do_normalize: bool = True,  # 是否对图像进行归一化的标志，默认为True
            image_mean: Optional[Union[float, List[float]]] = None,  # 归一化图像时使用的均值，默认为[0.5, 0.5, 0.5]
            image_std: Optional[Union[float, List[float]]] = None,  # 归一化图像时使用的标准差，默认为[0.5, 0.5, 0.5]
            **kwargs,  # 其他可能的关键字参数
        ):
    # 构造函数初始化，继承父类并传递关键字参数
    def __init__(
        self,
        **kwargs,
    ) -> None:
        # 调用父类初始化方法，传递所有关键字参数
        super().__init__(**kwargs)
        # 如果提供了 size 参数，则使用提供的值；否则使用默认尺寸 (height: 224, width: 224)
        size = size if size is not None else {"height": 224, "width": 224}
        # 如果提供了 image_mean 参数，则使用提供的值；否则使用 IMAGENET_STANDARD_MEAN 值
        image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        # 如果提供了 image_std 参数，则使用提供的值；否则使用 IMAGENET_STANDARD_STD 值

        image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

        # 设置对象的属性，用于预处理图像
        self.do_resize = do_resize  # 是否执行重新调整尺寸操作的标志
        self.size = size  # 图像的目标尺寸
        self.resample = resample  # 重新调整尺寸时的重采样方法
        self.do_rescale = do_rescale  # 是否执行重新缩放的标志
        self.rescale_factor = rescale_factor  # 图像重新缩放的因子
        self.do_normalize = do_normalize  # 是否执行归一化的标志
        self.image_mean = image_mean  # 归一化时的均值
        self.image_std = image_std  # 归一化时的标准差

        # 定义可以接受的预处理关键字参数列表
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 图像预处理方法
    def preprocess(
        self,
        images: ImageInput,  # 待处理的图像数据
        do_resize: bool = None,  # 是否执行重新调整尺寸的标志
        size: Dict[str, int] = None,  # 图像的目标尺寸
        resample: PILImageResampling = None,  # 重新调整尺寸时的重采样方法
        do_rescale: bool = None,  # 是否执行重新缩放的标志
        rescale_factor: float = None,  # 图像重新缩放的因子
        do_normalize: bool = None,  # 是否执行归一化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 归一化时的均值
        image_std: Optional[Union[float, List[float]]] = None,  # 归一化时的标准差
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量的格式
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,  # 数据的通道顺序
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的通道顺序
        **kwargs,  # 其他未命名参数
```