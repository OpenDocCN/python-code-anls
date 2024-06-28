# `.\models\segformer\image_processing_segformer.py`

```
# 导入警告模块，用于可能的警告信息输出
import warnings
# 导入类型提示相关模块
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 NumPy 库，用于处理数组等数值计算
import numpy as np

# 导入基础图像处理工具类和相关方法
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换相关方法
from ...image_transforms import resize, to_channel_dimension_format
# 导入图像处理中的常用方法和常量
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
# 导入通用工具函数和类型相关模块
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging

# 如果使用视觉相关功能，导入 PIL 图像模块
if is_vision_available():
    import PIL.Image

# 如果使用 PyTorch 相关功能，导入 PyTorch 库
if is_torch_available():
    import torch

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# SegformerImageProcessor 类，继承自 BaseImageProcessor 类
class SegformerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Segformer image processor.
    
    """
    # 定义函数参数和默认值，用于图像预处理
    Args:
        # 是否调整图像大小到指定尺寸(size["height"], size["width"])，可以在 preprocess 方法中通过 do_resize 参数覆盖
        do_resize (`bool`, *optional*, defaults to `True`):
        size (`Dict[str, int]` *optional*, defaults to `{"height": 512, "width": 512}`):
            # 调整后的图像大小，可以在 preprocess 方法中通过 size 参数覆盖
            Size of the output image after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            # 调整图像大小时使用的重采样滤波器，在 preprocess 方法中可以通过 resample 参数覆盖
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            # 是否按照指定比例 rescale_factor 对图像进行重新缩放，可以在 preprocess 方法中通过 do_rescale 参数覆盖
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            # 图像归一化的缩放因子，可以在 preprocess 方法中通过 rescale_factor 参数覆盖
            Whether to normalize the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            # 是否对图像进行标准化，可以在 preprocess 方法中通过 do_normalize 参数覆盖
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            # 图像标准化的均值，可以在 preprocess 方法中通过 image_mean 参数覆盖
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            # 图像标准化的标准差，可以在 preprocess 方法中通过 image_std 参数覆盖
            Standard deviation to use if normalizing the image.
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            # 是否减少分割图中所有标签值的值，通常用于数据集中 0 表示背景的情况，可以在 preprocess 方法中通过 do_reduce_labels 参数覆盖
            Whether or not to reduce all label values of segmentation maps by 1.
    """

    # 模型输入的名称列表
    model_input_names = ["pixel_values"]
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: bool = False,
        **kwargs,
    ) -> None:
        """
        Constructor method for initializing the ViTImageProcessor.

        Args:
            do_resize (bool): Whether to resize images.
            size (Dict[str, int], optional): Desired image size as a dictionary of height and width.
            resample (PILImageResampling, optional): Resampling method for resizing images.
            do_rescale (bool): Whether to rescale image pixel values.
            rescale_factor (Union[int, float]): Factor to scale image pixel values.
            do_normalize (bool): Whether to normalize image pixel values.
            image_mean (Optional[Union[float, List[float]]]): Mean values for image normalization.
            image_std (Optional[Union[float, List[float]]]): Standard deviation values for image normalization.
            do_reduce_labels (bool): Whether to reduce image labels.
            **kwargs: Additional keyword arguments.
        """
        if "reduce_labels" in kwargs:
            # Issue a warning if 'reduce_labels' is passed via kwargs (deprecated).
            warnings.warn(
                "The `reduce_labels` parameter is deprecated and will be removed in a future version. Please use "
                "`do_reduce_labels` instead.",
                FutureWarning,
            )
            # Set `do_reduce_labels` based on the value passed via kwargs.
            do_reduce_labels = kwargs.pop("reduce_labels")

        # Call the superclass initializer with any remaining kwargs.
        super().__init__(**kwargs)

        # Set default image size if not provided.
        size = size if size is not None else {"height": 512, "width": 512}
        # Normalize the size dictionary.
        size = get_size_dict(size)

        # Initialize instance variables.
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_reduce_labels = do_reduce_labels

        # List of valid processor keys for validation and configuration purposes.
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "do_resize",
            "size",
            "resample",
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
        Overrides the `from_dict` method from the base class to ensure `do_reduce_labels` is updated if the image
        processor is created using `from_dict` and kwargs (e.g., `SegformerImageProcessor.from_pretrained(checkpoint,
        reduce_labels=True)`).

        Args:
            image_processor_dict (Dict[str, Any]): Dictionary containing image processor configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            cls: Initialized instance of the class based on the provided dictionary and kwargs.
        """
        image_processor_dict = image_processor_dict.copy()
        if "reduce_labels" in kwargs:
            # Update 'reduce_labels' key in the dictionary with the value from kwargs.
            image_processor_dict["reduce_labels"] = kwargs.pop("reduce_labels")
        # Call the superclass's from_dict method with updated dictionary and remaining kwargs.
        return super().from_dict(image_processor_dict, **kwargs)

    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # Resize an image to the specified dimensions using various options.
    #
    # Args:
    #     image (`np.ndarray`): The input image to be resized.
    #     size (`Dict[str, int]`): A dictionary specifying the target size in the format {"height": int, "width": int}.
    #     resample (`PILImageResampling`, optional): The resampling filter to use during resizing.
    #         Defaults to `PILImageResampling.BILINEAR`.
    #     data_format (`ChannelDimension` or `str`, optional): Specifies the output image format regarding channel dimensions.
    #         Defaults to the format of the input image.
    #     input_data_format (`ChannelDimension` or `str`, optional): Specifies the input image format regarding channel dimensions.
    #         Defaults to the format inferred from the input image.
    #
    # Returns:
    #     `np.ndarray`: The resized image.
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample=PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format=None,
        **kwargs,
    ) -> np.ndarray:
        size = get_size_dict(size)  # Ensure `size` is in the correct dictionary format
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.reduce_label
    # Reduce the label image, converting 0 to 255, subtracting 1, and converting 254 to 255.
    #
    # Args:
    #     label (ImageInput): The input label image to be processed.
    #
    # Returns:
    #     `np.ndarray`: The processed label image.
    def reduce_label(self, label: ImageInput) -> np.ndarray:
        label = to_numpy_array(label)  # Convert label to a NumPy array
        label[label == 0] = 255  # Set all 0s to 255 to avoid underflow issues
        label = label - 1  # Subtract 1 from all values
        label[label == 254] = 255  # Set all 254s to 255
        return label  # Return the processed label image
    # 图像预处理函数，用于处理图像数据
    def _preprocess(
        self,
        image: ImageInput,
        do_reduce_labels: bool,
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # 如果需要减少标签，调用 reduce_label 函数处理图像
        if do_reduce_labels:
            image = self.reduce_label(image)

        # 如果需要调整大小，调用 resize 函数调整图像大小
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        # 如果需要重新缩放，调用 rescale 函数重新缩放图像
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        # 如果需要归一化，调用 normalize 函数归一化图像
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        # 返回预处理后的图像数据
        return image

    # 单个图像的预处理函数
    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """对单个图像进行预处理。"""
        # 将图像转换为 numpy 数组
        image = to_numpy_array(image)

        # 如果图像已经进行了缩放且需要重新缩放，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        # 推断输入数据的通道格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # 调用 _preprocess 函数进行图像预处理
        image = self._preprocess(
            image=image,
            do_reduce_labels=False,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
        )

        # 如果指定了输出数据格式，则将图像转换为指定格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回预处理后的图像数据
        return image

    # 预处理分割标签图像的函数
    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_reduce_labels: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # TODO: Implement preprocessing for segmentation maps (if needed in future)
        pass
    ) -> np.ndarray:
        """
        Preprocesses a single mask.

        Args:
            segmentation_map (np.ndarray): The input segmentation map to preprocess.

        Returns:
            np.ndarray: The preprocessed segmentation map as numpy array.
        """
        # Convert segmentation_map to numpy array if it's not already
        segmentation_map = to_numpy_array(segmentation_map)

        # Add channel dimension if missing - needed for certain transformations
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]  # Add a new axis at the beginning
            input_data_format = ChannelDimension.FIRST  # Set input data format to channel first
        else:
            added_channel_dim = False
            # If input_data_format is not specified, infer the channel dimension format
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)

        # Preprocess the segmentation map using the _preprocess method
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_reduce_labels=do_reduce_labels,
            do_resize=do_resize,
            resample=PILImageResampling.NEAREST,
            size=size,
            do_rescale=False,
            do_normalize=False,
            input_data_format=input_data_format,
        )

        # Remove extra channel dimension if it was added for processing
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)  # Squeeze out the added dimension

        segmentation_map = segmentation_map.astype(np.int64)  # Convert to np.int64 type
        return segmentation_map

    def __call__(self, images, segmentation_maps=None, **kwargs):
        """
        Preprocesses a batch of images and optionally segmentation maps.

        Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be
        passed in as positional arguments.

        Args:
            images: Batch of input images to preprocess.
            segmentation_maps: Optional batch of segmentation maps to preprocess.
            **kwargs: Additional keyword arguments passed to the superclass `__call__` method.

        Returns:
            The processed batch of images and segmentation maps.
        """
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocesses a batch of images and optionally segmentation maps.

        Args:
            images: Batch of input images to preprocess.
            segmentation_maps: Optional batch of segmentation maps to preprocess.
            do_resize: Optional flag indicating whether to resize images/maps.
            size: Optional dictionary specifying target size for resizing.
            resample: Resampling method for resizing images/maps.
            do_rescale: Optional flag indicating whether to rescale images/maps.
            rescale_factor: Optional factor for rescaling images/maps.
            do_normalize: Optional flag indicating whether to normalize images/maps.
            image_mean: Optional mean value(s) for image normalization.
            image_std: Optional standard deviation value(s) for image normalization.
            do_reduce_labels: Optional flag indicating whether to reduce labels.
            return_tensors: Optional flag indicating desired output tensor format.
            data_format: Channel dimension format for images/maps.
            input_data_format: Optional specific format for channel dimension.

            **kwargs: Additional keyword arguments for flexibility.
        """
        # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.post_process_semantic_segmentation with Beit->Segformer
    # 后处理语义分割模型输出，将模型输出转换为语义分割图。仅支持 PyTorch。
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`SegformerForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`SegformerForSemanticSegmentation`]):
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

        # 获取模型输出的逻辑层结果
        logits = outputs.logits

        # 调整逻辑层大小并计算语义分割图
        if target_sizes is not None:
            # 检查目标大小与逻辑层数量是否匹配
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # 如果目标大小是 PyTorch 张量，则转换为 NumPy 数组
            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            # 初始化语义分割结果列表
            semantic_segmentation = []

            # 遍历每个样本的逻辑层结果
            for idx in range(len(logits)):
                # 对逻辑层进行插值调整大小
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # 取调整大小后的逻辑层结果中最大值所在的索引，即语义分割图
                semantic_map = resized_logits[0].argmax(dim=0)
                # 将语义分割图添加到结果列表中
                semantic_segmentation.append(semantic_map)
        else:
            # 如果未指定目标大小，直接计算逻辑层中每个样本的最大值索引作为语义分割图
            semantic_segmentation = logits.argmax(dim=1)
            # 将每个样本的语义分割图转换为列表形式
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # 返回语义分割结果列表
        return semantic_segmentation
```