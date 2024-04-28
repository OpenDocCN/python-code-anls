# `.\models\dpt\image_processing_dpt.py`

```
# 设置文件编码为utf-8
# 版权声明和许可信息
"""Image processor class for DPT."""  # 为DPT设计的图像处理器类

# 导入所需的包和模块
import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import pad, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_torch_available,
    is_torch_tensor,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_vision_available, logging

# 如果torch可用，则导入torch模块
if is_torch_available():
    import torch

# 如果vision可用，则导入PIL模块
if is_vision_available():
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一个函数，用于计算调整大小后的输出图像大小
def get_resize_output_image_size(
    input_image: np.ndarray,  # 输入的图像数据
     output_size: Union[int, Iterable[int]],  # 输出的图像尺寸，可以是单个整数或一个包含两个整数的可迭代对象
    keep_aspect_ratio: bool,  # 是否保持宽高比
    multiple: int,  # 调整大小的倍数
    input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图像的数据格式，可选参数
) -> Tuple[int, int]:  # 返回调整大小后的图像高度和宽度的元组类型
    # 定义一个函数，将值约束为某个数的倍数
    def constraint_to_multiple_of(val, multiple, min_val=0, max_val=None):
        x = round(val / multiple) * multiple

        if max_val is not None and x > max_val:
            x = math.floor(val / multiple) * multiple

        if x < min_val:
            x = math.ceil(val / multiple) * multiple

        return x

    output_size = (output_size, output_size) if isinstance(output_size, int) else output_size  # 将输出尺寸转换为元组形式

    input_height, input_width = get_image_size(input_image, input_data_format)  # 获取输入图像的高度和宽度
    output_height, output_width = output_size  # 获取输出图像的高度和宽度

    # 确定新的高度和宽度
    scale_height = output_height / input_height  # 计算高度的缩放比例
    scale_width = output_width / input_width  # 计算宽度的缩放比例

    if keep_aspect_ratio:  # 如果需要保持宽高比
        # 尽量缩放得更少
        if abs(1 - scale_width) < abs(1 - scale_height):
            # 适应宽度
            scale_height = scale_width
        else:
            # 适应高度
            scale_width = scale_height

    new_height = constraint_to_multiple_of(scale_height * input_height, multiple=multiple)  # 根据倍数约束新的高度
    new_width = constraint_to_multiple_of(scale_width * input_width, multiple=multiple)  # 根据倍数约束新的宽度

    return (new_height, new_width)  # 返回调整大小后的图像高度和宽度的元组

# 定义一个DPTImageProcessor类，继承自BaseImageProcessor类
class DPTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DPT image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            # 是否调整图像的（高度，宽度）尺寸。可以在`preprocess`中用`do_resize`覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 384, "width": 384}`):
            # 调整大小后的图像尺寸。可以在`preprocess`中用`size`来覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            # 如果调整图像大小，则定义要使用的重采样滤波器。可以在`preprocess`中用`resample`来覆盖。
        keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
            # 如果为`True`，则将图像调整为尺寸最大的尺寸，使得纵横比保持不变。可以在`preprocess`中用`keep_aspect_ratio`来覆盖。
        ensure_multiple_of (`int`, *optional*, defaults to 1):
            # 如果`do_resize`为`True`，则将图像调整为此值的倍数大小。可以在`preprocess`中用`ensure_multiple_of`来覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            # 是否按指定比例`rescale_factor`对图像进行重新缩放。可以在`preprocess`中用`do_rescale`来覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            # 如果重新缩放图像，将使用的缩放因子。可以在`preprocess`中用`rescale_factor`来覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            # 是否对图像进行归一化。可以在`preprocess`的`do_normalize`参数中覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            # 如果对图像进行归一化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在`preprocess`中用`image_mean`来覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            # 如果对图像进行归一化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在`preprocess`中用`image_std`来覆盖。
        do_pad (`bool`, *optional*, defaults to `False`):
            # 是否应用中心填充。这在DINOv2论文中引入，该论文与DPT模型结合使用。
        size_divisor (`int`, *optional*):
            # 如果`do_pad`为`True`，则将图像尺寸填充为此值的倍数。这在DINOv2论文中引入，该论文与DPT模型结合使用。
    """

    model_input_names = ["pixel_values"]
    # 初始化方法定义，包括大量的图像处理参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整的标志
        size: Dict[str, int] = None,  # 调整后的图像尺寸，字典格式，包含宽和高
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像重采样方法，默认双三次插值法
        keep_aspect_ratio: bool = False,  # 是否保持图像的长宽比
        ensure_multiple_of: int = 1,  # 确保图像尺寸是某个数的倍数
        do_rescale: bool = True,  # 是否对图像进行缩放
        rescale_factor: Union[int, float] = 1 / 255,  # 缩放系数，默认为1/255，常用于归一化处理
        do_normalize: bool = True,  # 是否进行图像标准化处理
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像标准化时使用的平均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准化时使用的标准差
        do_pad: bool = False,  # 是否对图像进行填充处理
        size_divisor: int = None,  # 对尺寸进行处理的除数，用于确定填充量
        **kwargs,  # 其他任意关键字参数
    ) -> None:
        super().__init__(**kwargs)  # 调用父类的初始化方法
        size = size if size is not None else {"height": 384, "width": 384}  # 如果未指定尺寸则默认384x384
        size = get_size_dict(size)  # 获取规范化的尺寸字典
        self.do_resize = do_resize  # 存储是否需要调整尺寸的设置
        self.size = size  # 存储调整后的尺寸
        self.keep_aspect_ratio = keep_aspect_ratio  # 存储是否保持长宽比的设置
        self.ensure_multiple_of = ensure_multiple_of  # 存储尺寸倍数要求
        self.resample = resample  # 存储图像重采样方法
        self.do_rescale = do_rescale  # 存储是否进行缩放的设置
        self.rescale_factor = rescale_factor  # 存储缩放系数
        self.do_normalize = do_normalize  # 存储是否进行标准化处理的设置
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 设置或使用默认的图像平均值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 设置或使用默认的图像标准差
        self.do_pad = do_pad  # 存储是否进行填充处理的设置
        self.size_divisor = size_divisor  # 存储尺寸处理除数

    # 定义 resize 方法，用于调整图像大小
    def resize(
        self,
        image: np.ndarray,  # 输入的图像，NumPy 数组格式
        size: Dict[str, int],  # 目标尺寸，字典格式
        keep_aspect_ratio: bool = False,  # 是否保持长宽比
        ensure_multiple_of: int = 1,  # 确保尺寸是某个数的倍数
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 图像数据格式（通道顺序）
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的图像格式
        **kwargs,  # 接收任意额外的关键字参数
    ) -> np.ndarray:
        """
        将图像调整为目标大小 `(size["height"], size["width"])`。如果 `keep_aspect_ratio` 设为 `True`，则将图像调整为保持宽高比的最大可能大小。如果设置了 `ensure_multiple_of`，则将图像调整为此值的倍数。

        Args:
            image (`np.ndarray`):
                要调整大小的图像。
            size (`Dict[str, int]`):
                输出图像的目标大小。
            keep_aspect_ratio (`bool`, *可选*, 默认为 `False`):
                如果为 `True`，则将图像调整为保持宽高比的最大可能大小。
            ensure_multiple_of (`int`, *可选*, 默认为 1):
                将图像调整为此值的倍数大小。
            resample (`PILImageResampling`, *可选*, 默认为 `PILImageResampling.BICUBIC`):
                如果调整图像大小，则定义要使用的重采样滤波器。否则，将按`size`参数指定的大小调整图像。
            data_format (`str` 或 `ChannelDimension`, *可选*):
                图像的通道维度格式。如果未提供，则将使用输入图像的相同格式。
            input_data_format (`str` 或 `ChannelDimension`, *可选*):
                输入图像的通道维度格式。如果未提供，则将推断出格式。
        """
        # 将size参数转换为标准化的大小字典
        size = get_size_dict(size)
        # 检查size字典是否包含'height'和'width'键
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must contain the keys 'height' and 'width'. Got {size.keys()}")

        # 获取调整后图像的输出大小
        output_size = get_resize_output_image_size(
            image,
            output_size=(size["height"], size["width"]),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
            input_data_format=input_data_format,
        )
        # 调整图像大小并返回
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def pad_image(
        self,
        image: np.array,
        size_divisor: int,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def _get_pad(size, size_divisor):
        # 计算将尺寸调整为 size_divisor 的倍数后的新尺寸
        new_size = math.ceil(size / size_divisor) * size_divisor
        # 计算需要填充的尺寸
        pad_size = new_size - size
        # 计算左侧填充尺寸
        pad_size_left = pad_size // 2
        # 计算右侧填充尺寸
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right
    
    # 如果输入图片的数据格式未指定，从输入图片中推断出通道维度格式
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    
    # 获取图片的高度和宽度
    height, width = get_image_size(image, input_data_format)
    
    # 计算高度方向和宽度方向的填充尺寸
    pad_size_left, pad_size_right = _get_pad(height, size_divisor)
    pad_size_top, pad_size_bottom = _get_pad(width, size_divisor)
    
    # 对图片进行填充
    return pad(image, ((pad_size_left, pad_size_right), (pad_size_top, pad_size_bottom)), data_format=data_format)
    
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: int = None,
        keep_aspect_ratio: bool = None,
        ensure_multiple_of: int = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = None,
        size_divisor: int = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 从 transformers.models.beit.image_processing_beit.BeitImageProcessor.post_process_semantic_segmentation 复制，并将 Beit 替换为 DPT
    # 后处理语义分割结果，将 `DPTForSemanticSegmentation` 的输出转换为语义分割图。仅支持 PyTorch。
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`DPTForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`DPTForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        # TODO: add support for other frameworks

        # 获取模型输出的 logits
        logits = outputs.logits

        # 调整 logits 的大小并计算语义分割图
        if target_sizes is not None:
            # 检查目标大小数量是否与 logits 的批处理维度数量相匹配
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # 将 target_sizes 转换为 numpy 数组（如果是 torch.Tensor 类型）
            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            # 初始化语义分割结果列表
            semantic_segmentation = []

            # 遍历每个 logits，并调整大小以及计算语义分割图
            for idx in range(len(logits)):
                # 调整 logits 大小
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # 提取语义分割图
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # 如果未提供目标大小，则直接从 logits 计算语义分割图
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # 返回语义分割结果
        return semantic_segmentation
```