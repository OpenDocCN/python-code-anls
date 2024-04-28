# `.\transformers\models\owlv2\image_processing_owlv2.py`

```
# 设置 UTF-8 编码
# 版权声明和许可协议，表示代码版权和使用许可
# 仅在遵循许可协议的情况下可使用该文件
# 获取许可协议的副本
# 许可协议网址
# 根据适用法律或书面协议约定而不必要，本软件以“原样”分发，没有任何形式的担保或条件
# 根据许可协议，本软件分发时不提供任何明示或暗示的保证或条件
# 有关许可协议的特定语言的权限和限制，请参阅许可协议
"""OWLv2 的图像处理器类。"""

# 引入警告模块
import warnings
# 引入类型提示模块
from typing import Dict, List, Optional, Tuple, Union

# 引入 NumPy 库
import numpy as np

# 引入图像处理工具函数和图像转换函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    center_to_corners_format,
    pad,
    to_channel_dimension_format,
)
# 引入图像工具函数
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
# 引入工具函数
from ...utils import (
    TensorType,
    is_scipy_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)

# 如果 PyTorch 可用，则引入 PyTorch 库
if is_torch_available():
    import torch

# 如果 Vision 库可用，则引入 PIL 库
if is_vision_available():
    import PIL

# 如果 SciPy 库可用，则引入 SciPy 库
if is_scipy_available():
    from scipy import ndimage as ndi

# 获取日志记录器
logger = logging.get_logger(__name__)


# 从 transformers 模型的 OWLv2 图像处理器模块中复制的 _upcast 函数
def _upcast(t):
    # 防止乘法导致数值溢出，将数据类型转换为等效的更高类型
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# 从 transformers 模型的 OWLv2 图像处理器模块中复制的 box_area 函数
def box_area(boxes):
    """
    计算一组边界框的面积，这些边界框由其 (x1, y1, x2, y2) 坐标指定。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            要计算面积的边界框。它们应该以 (x1, y1, x2, y2) 格式给出，其中 `0 <= x1 < x2` 且 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个框的面积的张量。
    """
    # 将输入张量数据类型转换为适当类型
    boxes = _upcast(boxes)
    # 计算边界框的宽度和高度并返回面积
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# 从 transformers 模型的 OWLv2 图像处理器模块中复制的 box_iou 函数
def box_iou(boxes1, boxes2):
    # 计算边界框的面积
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 计算两组边界框的交集
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # 计算交集的宽度和高度
    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    # 计算两个矩形框的交集面积
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]
    
    # 计算两个矩形框的并集面积
    union = area1[:, None] + area2 - inter
    
    # 计算两个矩形框的 IoU (Intersection over Union)
    iou = inter / union
    return iou, union
# 预处理函数，用于验证调整大小后的输出形状是否符合输入图像
def _preprocess_resize_output_shape(image, output_shape):
    # 将输出形状转换为元组
    output_shape = tuple(output_shape)
    # 获取输出形状的维度
    output_ndim = len(output_shape)
    # 获取输入图像的形状
    input_shape = image.shape
    # 如果输出维度大于输入维度，向输入形状追加维度
    if output_ndim > image.ndim:
        input_shape += (1,) * (output_ndim - image.ndim)
        # 重塑输入图像
        image = np.reshape(image, input_shape)
    # 如果输出维度减1等于输入维度，为多通道情况，追加最后一轴的形状
    elif output_ndim == image.ndim - 1:
        output_shape = output_shape + (image.shape[-1],)
    # 如果输出维度小于输入维度，抛出数值错误
    elif output_ndim < image.ndim:
        raise ValueError("output_shape length cannot be smaller than the " "image number of dimensions")
    # 返回经处理后的图像和输出形状
    return image, output_shape


# 函数用于将输出图像剪裁到输入图像的数值范围内
def _clip_warp_output(input_image, output_image):
    # 找到输入图像的最小值
    min_val = np.min(input_image)
    # 如果存在 NaN 值，使用 NaN 安全的最小/最大值处理函数
    if np.isnan(min_val):
        min_func = np.nanmin
        max_func = np.nanmax
        min_val = min_func(input_image)
    else:
        min_func = np.min
        max_func = np.max
    # 找到输入图像的最大值
    max_val = max_func(input_image)
    # 将输出图像的值剪裁到输入图像的范围内
    output_image = np.clip(output_image, min_val, max_val)
    # 返回剪裁后的输出图像
    return output_image


# OWLv2 图像处理器类
class Owlv2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs an OWLv2 image processor.
    """
    # 初始化图像预处理器类
    """
    Args:
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定比例`rescale_factor`对图像进行重新缩放。可以在`preprocess`方法中被`do_rescale`参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像，要使用的比例因子。可以在`preprocess`方法中被`rescale_factor`参数覆盖。
        do_pad (`bool`, *optional*, defaults to `True`):
            是否将图像填充为一个灰色底的正方形。可以在`preprocess`方法中被`do_pad`参数覆盖。
        do_resize (`bool`, *optional*, defaults to `True`):
            控制是否将图像的（高度，宽度）尺寸调整为指定的`size`。可以在`preprocess`方法中被`do_resize`参数覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 960, "width": 960}`):
            要调整图像大小到的尺寸。可以在`preprocess`方法中被`size`参数覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            如果调整图像大小，要使用的重采样方法。可以在`preprocess`方法中被`resample`参数覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以在`preprocess`方法中被`do_normalize`参数覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            如果对图像进行归一化，要使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在`preprocess`方法中被`image_mean`参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
            如果对图像进行归一化，要使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在`preprocess`方法中被`image_std`参数覆盖。
    """

    # 模型输入名称列表
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_pad: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
    def __init__(self, *args, **kwargs) -> None:
        # 调用父类的构造函数，并传入kwargs参数
        super().__init__(**kwargs)

        # 初始化各种数据处理参数
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 960, "width": 960}
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

    def pad(
        self,
        image: np.array,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pad an image to a square with gray pixels on the bottom and the right, as per the original OWLv2
        implementation.

        Args:
            image (`np.ndarray`):
                Image to pad.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional`):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        # 获取图像的高度和宽度
        height, width = get_image_size(image)
        # 计算需要pad的边长
        size = max(height, width)
        # 进行图像的pad操作
        image = pad(
            image=image,
            padding=((0, size - height), (0, size - width)),
            constant_values=0.5,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return image

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        anti_aliasing: bool = True,
        anti_aliasing_sigma=None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image as per the original implementation.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary containing the height and width to resize the image to.
            anti_aliasing (`bool`, *optional*, defaults to `True`):
                Whether to apply anti-aliasing when downsampling the image.
            anti_aliasing_sigma (`float`, *optional*, defaults to `None`):
                Standard deviation for Gaussian kernel when downsampling the image. If `None`, it will be calculated
                automatically.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        # 确保所需的后端库存在
        requires_backends(self, "scipy")

        # 输出形状为指定高度和宽度
        output_shape = (size["height"], size["width"])
        # 调整图像通道维度格式为最后一个维度
        image = to_channel_dimension_format(image, ChannelDimension.LAST)
        # 预处理调整输出形状参数
        image, output_shape = _preprocess_resize_output_shape(image, output_shape)
        # 输入图像形状
        input_shape = image.shape
        # 计算缩放因子
        factors = np.divide(input_shape, output_shape)

        # 将 np.pad 使用的模式转换为 scipy.ndimage 使用的模式
        ndi_mode = "mirror"
        cval = 0
        order = 1
        # 如果需要反锯齿处理
        if anti_aliasing:
            # 如果未提供反锯齿标准差，则自动计算
            if anti_aliasing_sigma is None:
                anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
            else:
                anti_aliasing_sigma = np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
                # 检查标准差是否为非负值
                if np.any(anti_aliasing_sigma < 0):
                    raise ValueError("Anti-aliasing standard deviation must be " "greater than or equal to zero")
                # 检查是否同时存在非零标准差和小于等于1的缩放因子
                elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                    warnings.warn(
                        "Anti-aliasing standard deviation greater than zero but " "not down-sampling along all axes"
                    )
            # 使用高斯滤波器进行反锯齿处理
            filtered = ndi.gaussian_filter(image, anti_aliasing_sigma, cval=cval, mode=ndi_mode)
        else:
            filtered = image

        # 计算缩放因子并在滤波后进行缩放
        zoom_factors = [1 / f for f in factors]
        out = ndi.zoom(filtered, zoom_factors, order=order, mode=ndi_mode, cval=cval, grid_mode=True)

        # 调整处理后的输出图像的像素值范围
        image = _clip_warp_output(image, out)

        # 将通道维度格式转换为指定的格式
        image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)
        # 如果提供了数据格式，则将图像格式转换为指定格式
        image = (
            to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        )
        # 返回处理后的图像
        return image
    # 图像预处理函数
    def preprocess(
        self,
        # 图像输入数据，可以是文件路径、PIL.Image 对象或者张量
        images: ImageInput,
        # 是否进行填充操作
        do_pad: bool = None,
        # 是否进行调整大小操作
        do_resize: bool = None,
        # 调整大小后的目标尺寸
        size: Dict[str, int] = None,
        # 是否进行重新缩放操作
        do_rescale: bool = None,
        # 重新缩放因子
        rescale_factor: float = None,
        # 是否进行标准化操作
        do_normalize: bool = None,
        # 图像均值
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, List[float]]] = None,
        # 返回张量还是 NumPy 数组
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 数据格式，通道维度在前还是在后
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据的格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他参数
        **kwargs,
    # 从 OwlViTImageProcessor.post_process_object_detection 复制的后处理函数
    def post_process_object_detection(
        self, 
        # 模型输出
        outputs, 
        # 目标检测置信度阈值
        threshold: float = 0.1, 
        # 目标尺寸
        target_sizes: Union[TensorType, List[Tuple]] = None
    def post_process_image_guided_detection(
        self,
        outputs,
        threshold=0.5,
        target_sizes=None,
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.
    
        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # Extract the logits and bounding boxes from the model outputs
        logits, boxes = outputs.logits, outputs.pred_boxes
    
        # If target sizes are provided, ensure the batch size matches the logits
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
    
        # Convert the logits to probabilities and get the class labels
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices
    
        # Convert the bounding boxes from center-based format to corner-based format
        boxes = center_to_corners_format(boxes)
    
        # Convert the bounding boxes from relative coordinates [0, 1] to absolute coordinates [0, height] based on the target sizes
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)
    
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]
    
        # Filter the results based on the score threshold and pack them into a list of dictionaries
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})
    
        return results
```