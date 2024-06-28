# `.\models\beit\image_processing_beit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指出代码版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，只有在遵循许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则按“现状”提供软件，不附带任何明示或暗示的担保或条件。
# 有关许可证详细信息，请参见许可证文本。

"""Beit 的图像处理类。"""

# 导入警告模块
import warnings
# 导入类型提示模块
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np

# 导入图像处理工具类和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换函数
from ...image_transforms import resize, to_channel_dimension_format
# 导入图像处理工具函数
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,  # 导入常用的图像均值
    IMAGENET_STANDARD_STD,   # 导入常用的图像标准差
    ChannelDimension,        # 导入通道维度类
    ImageInput,              # 导入图像输入类
    PILImageResampling,      # 导入 PIL 图像重采样方法枚举
    infer_channel_dimension_format,  # 推断通道维度格式函数
    is_scaled_image,         # 判断图像是否经过缩放函数
    make_list_of_images,     # 将图像转换为图像列表函数
    to_numpy_array,          # 将图像转换为 NumPy 数组函数
    valid_images,            # 检验有效图像函数
    validate_kwargs,         # 验证关键字参数函数
    validate_preprocess_arguments,  # 验证预处理参数函数
)
# 导入通用工具函数和类型
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging

# 如果 PyTorch 可用，导入 PyTorch 模块
if is_torch_available():
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 BeitImageProcessor 类，继承自 BaseImageProcessor 类
class BeitImageProcessor(BaseImageProcessor):
    r"""
    构建 BEiT 图像处理器。

    """

    # 模型输入名称列表，仅包含像素值
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,                   # 是否进行调整大小的标志
        size: Dict[str, int] = None,              # 图像大小的字典，包含宽和高
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # PIL 图像重采样方法
        do_center_crop: bool = True,              # 是否进行中心裁剪的标志
        crop_size: Dict[str, int] = None,         # 裁剪尺寸的字典，包含宽和高
        rescale_factor: Union[int, float] = 1 / 255,  # 图像缩放因子
        do_rescale: bool = True,                  # 是否进行图像缩放的标志
        do_normalize: bool = True,                # 是否进行图像标准化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值
        image_std: Optional[Union[float, List[float]]] = None,   # 图像标准差
        do_reduce_labels: bool = False,           # 是否减少标签的标志
        **kwargs,                                 # 其他关键字参数
    ):
        # 调用父类的构造函数
        super().__init__(**kwargs)
    ) -> None:
        # 如果 kwargs 中包含 "reduce_labels" 参数，则发出警告，并将其值赋给 do_reduce_labels
        if "reduce_labels" in kwargs:
            warnings.warn(
                "The `reduce_labels` parameter is deprecated and will be removed in a future version. Please use"
                " `do_reduce_labels` instead.",
                FutureWarning,
            )
            do_reduce_labels = kwargs.pop("reduce_labels")
        # 调用父类的初始化方法，传入所有的 kwargs
        super().__init__(**kwargs)
        # 设置 size 变量，如果未指定则使用默认值 {"height": 256, "width": 256}
        size = size if size is not None else {"height": 256, "width": 256}
        # 调用 get_size_dict 函数，确保 size 是一个字典
        size = get_size_dict(size)
        # 设置 crop_size 变量，如果未指定则使用默认值 {"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 调用 get_size_dict 函数，确保 crop_size 是一个字典，参数名为 "crop_size"
        crop_size = get_size_dict(crop_size, param_name="crop_size")
        # 设置对象的成员变量
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        # 设置对象的成员变量，如果未指定 image_mean 则使用 IMAGENET_STANDARD_MEAN
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        # 设置对象的成员变量，如果未指定 image_std 则使用 IMAGENET_STANDARD_STD
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        # 设置对象的成员变量 do_reduce_labels
        self.do_reduce_labels = do_reduce_labels
        # 设置对象的成员变量 _valid_processor_keys，包含所有可能的处理器参数键名
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_reduce_labels",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure `reduce_labels` is updated if image processor
        is created using from_dict and kwargs e.g. `BeitImageProcessor.from_pretrained(checkpoint, reduce_labels=True)`
        """
        # 复制 image_processor_dict，确保原始字典不受影响
        image_processor_dict = image_processor_dict.copy()
        # 如果 kwargs 中包含 "reduce_labels" 参数，则将其值更新到 image_processor_dict 中
        if "reduce_labels" in kwargs:
            image_processor_dict["reduce_labels"] = kwargs.pop("reduce_labels")
        # 调用父类的 from_dict 方法，传入更新后的 image_processor_dict 和其他 kwargs
        return super().from_dict(image_processor_dict, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def _preprocess(
        self,
        image: ImageInput,
        do_reduce_labels: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocesses an image based on specified operations.

        Args:
            image (`ImageInput`):
                The input image to be preprocessed.
            do_reduce_labels (`bool`, optional):
                Whether to reduce labels using `reduce_label` method.
            do_resize (`bool`, optional):
                Whether to resize the image.
            size (`Dict[str, int]`, optional):
                Target size (height and width) for resizing.
            resample (`PILImageResampling`, optional):
                Resampling filter for resizing the image.
            do_center_crop (`bool`, optional):
                Whether to perform center cropping.
            crop_size (`Dict[str, int]`, optional):
                Size for center cropping (height and width).
            do_rescale (`bool`, optional):
                Whether to rescale the image.
            rescale_factor (`float`, optional):
                Factor for rescaling the image.
            do_normalize (`bool`, optional):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, optional):
                Mean values for normalizing the image.
            image_std (`float` or `List[float]`, optional):
                Standard deviation values for normalizing the image.
            input_data_format (`str` or `ChannelDimension`, optional):
                Format of the input image data.

        Returns:
            `np.ndarray`: Preprocessed image based on the specified operations.
        """
        if do_reduce_labels:
            # Reduce label values using the `reduce_label` method
            image = self.reduce_label(image)

        if do_resize:
            # Resize the image using specified size and resampling filter
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            # Perform center cropping on the image
            image = self.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            # Rescale the image using the specified factor
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            # Normalize the image using mean and standard deviation
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        return image
    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # 转换输入图像为 numpy 数组
        image = to_numpy_array(image)
        
        # 如果输入图像已经进行了缩放且设置了 do_rescale=True，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        
        # 推断输入数据格式的通道维度
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        
        # 调用 _preprocess 方法，对图像进行预处理
        image = self._preprocess(
            image,
            do_reduce_labels=False,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
        )
        
        # 如果指定了 data_format，将图像转换为指定的通道维度格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        
        # 返回预处理后的图像数组
        return image

    def _preprocess_segmentation_map(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_reduce_labels: bool = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocesses a single segmentation map.

        """
        # All transformations expect numpy arrays.
        segmentation_map = to_numpy_array(segmentation_map)
        # Add an axis to the segmentation maps for transformations.
        if segmentation_map.ndim == 2:
            segmentation_map = segmentation_map[None, ...]
            added_dimension = True
            input_data_format = ChannelDimension.FIRST
        else:
            added_dimension = False
            # If input_data_format is not specified, infer it based on the segmentation map.
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_reduce_labels=do_reduce_labels,
            do_resize=do_resize,
            resample=resample,
            size=size,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_normalize=False,
            do_rescale=False,
            input_data_format=ChannelDimension.FIRST,
        )
        # Remove extra axis if added
        if added_dimension:
            segmentation_map = np.squeeze(segmentation_map, axis=0)
        segmentation_map = segmentation_map.astype(np.int64)
        return segmentation_map

    def __call__(self, images, segmentation_maps=None, **kwargs):
        """
        Overrides the `__call__` method of the `Preprocessor` class such that the images and segmentation maps can both
        be passed in as positional arguments.
        """
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
        ):
        """
        Handles preprocessing of images and segmentation maps with various options for transformations and adjustments.

        """
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`BeitForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`BeitForSemanticSegmentation`]):
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
        
        # Extract logits from the model outputs
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps if target_sizes is provided
        if target_sizes is not None:
            # Check if the number of logits matches the number of target sizes
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # Convert target_sizes to numpy array if it's a torch tensor
            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            # Initialize an empty list for storing semantic segmentation maps
            semantic_segmentation = []

            # Iterate over each element in logits and perform interpolation
            for idx in range(len(logits)):
                # Resize logits using bilinear interpolation
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # Compute the semantic map by taking the argmax along the channel dimension
                semantic_map = resized_logits[0].argmax(dim=0)
                # Append the computed semantic map to the list
                semantic_segmentation.append(semantic_map)
        else:
            # Compute semantic segmentation by taking the argmax over the channel dimension
            semantic_segmentation = logits.argmax(dim=1)
            # Convert the result to a list of tensors
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # Return the list of semantic segmentation maps
        return semantic_segmentation
```