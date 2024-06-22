# `.\transformers\models\beit\image_processing_beit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2022 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，不附带任何明示或暗示的担保，
# 包括但不限于特定用途的适销性和适用性的任何担保。
# 请查看许可证以获取特定语言的权限和限制。
"""Beit 的图像处理器类。"""

# 引入警告模块
import warnings
# 引入类型提示模块
from typing import Any, Dict, List, Optional, Tuple, Union

# 引入 numpy 模块
import numpy as np

# 引入图像处理工具模块
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 引入图像变换模块
from ...image_transforms import resize, to_channel_dimension_format
# 引入图像工具模块
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
# 引入通用工具模块
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging

# 如果有视觉模块可用
if is_vision_available():
    # 引入 PIL 模块
    import PIL

# 如果有 Torch 模块可用
if is_torch_available():
    # 引入 Torch 模块
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 Beit 图像处理器类，继承自 BaseImageProcessor
class BeitImageProcessor(BaseImageProcessor):
    r"""
    构建一个 BEiT 图像处理器。

    """

    # 模型输入名称为像素值
    model_input_names = ["pixel_values"]

    # 初始化方法
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: bool = False,
        **kwargs,
    # 定义一个方法，用于设置图像处理器的参数
    ) -> None:
        # 如果 kwargs 中包含"reduce_labels"参数
        if "reduce_labels" in kwargs:
            # 发出警告，提示"reduce_labels"参数已被弃用，并将在将来的版本中移除，建议使用"do_reduce_labels"代替
            warnings.warn(
                "The `reduce_labels` parameter is deprecated and will be removed in a future version. Please use"
                " `do_reduce_labels` instead.",
                FutureWarning,
            )
            # 将"reduce_labels"参数的值赋给"do_reduce_labels"，并将其从 kwargs 中移除
            do_reduce_labels = kwargs.pop("reduce_labels")
        # 调用父类的初始化方法，传入所有的关键字参数
        super().__init__(**kwargs)
        # 如果 size 参数不为 None，则将其设为指定的值，否则设为默认值
        size = size if size is not None else {"height": 256, "width": 256}
        # 将 size 转换为字典形式
        size = get_size_dict(size)
        # 如果 crop_size 参数不为 None，则将其设为指定的值，否则设为默认值
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 将 crop_size 转换为字典形式
        crop_size = get_size_dict(crop_size, param_name="crop_size")
        # 初始化图像处理器的各个参数
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_reduce_labels = do_reduce_labels

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure `reduce_labels` is updated if image processor
        is created using from_dict and kwargs e.g. `BeitImageProcessor.from_pretrained(checkpoint, reduce_labels=True)`
        """
        # 复制传入的图像处理器参数字典
        image_processor_dict = image_processor_dict.copy()
        # 如果 kwargs 中包含"reduce_labels"参数
        if "reduce_labels" in kwargs:
            # 将传入的"reduce_labels"参数的值更新到图像处理器参数字典中
            image_processor_dict["reduce_labels"] = kwargs.pop("reduce_labels")
        # 调用父类的 from_dict 方法，传入更新后的图像处理器参数字典和其他关键字参数
        return super().from_dict(image_processor_dict, **kwargs)

    # 定义一个调整图像大小的方法
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数据
        size: Dict[str, int],  # 目标大小的字典，包括高度和宽度
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 输出数据格式，可选参数，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选参数，默认为 None
        **kwargs,  # 其他关键字参数
``` 
    ) -> np.ndarray:
        """
        Resize an image to (size["height"], size["width"]).

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PIL.Image.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 将size参数转换为字典形式，如果未提供height和width，则默认为正方形
        size = get_size_dict(size, default_to_square=True, param_name="size")
        # 检查size字典中是否包含height和width键，如果没有则引发异常
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` argument must contain `height` and `width` keys. Got {size.keys()}")
        # 调用resize函数，对图像进行调整大小操作
        return resize(
            image,
            size=(size["height"], size["width"]),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def reduce_label(self, label: ImageInput) -> np.ndarray:
        # 将label转换为numpy数组
        label = to_numpy_array(label)
        # 避免使用下溢转换，将值为0的像素值设为255
        label[label == 0] = 255
        # 将label值减1
        label = label - 1
        # 将值为254的像素值设为255
        label[label == 254] = 255
        return label

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
        image_std: Optional[Union[float, List[float]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # 如果需要减少标签，则调用reduce_label函数
        if do_reduce_labels:
            image = self.reduce_label(image)

        # 如果需要调整大小，则调用resize函数
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        # 如果需要中心裁剪，则调用center_crop函数
        if do_center_crop:
            image = self.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

        # 如果需要重新缩放，则调用rescale函数
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        # 如果需要归一化，则调用normalize函数
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        return image
    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,  # 是否执行调整大小的操作，默认为 None
        size: Dict[str, int] = None,  # 调整大小的目标尺寸，默认为 None
        resample: PILImageResampling = None,  # 图像调整大小时的重采样方法，默认为 None
        do_center_crop: bool = None,  # 是否执行中心裁剪操作，默认为 None
        crop_size: Dict[str, int] = None,  # 中心裁剪的目标尺寸，默认为 None
        do_rescale: bool = None,  # 是否执行重新缩放操作，默认为 None
        rescale_factor: float = None,  # 重新缩放的因子，默认为 None
        do_normalize: bool = None,  # 是否执行归一化操作，默认为 None
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像的均值，默认为 None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像的标准差，默认为 None
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 输出数据格式，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为 None
    ) -> np.ndarray:
        """Preprocesses a single image."""  # 预处理单张图像
        # 将输入图像转换为 numpy 数组
        image = to_numpy_array(image)
        # 如果输入图像已经缩放，并且设置了重新缩放操作，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        # 推断输入数据的通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        # 对图像进行预处理
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
        # 如果指定了输出数据格式，则转换为指定格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        # 返回预处理后的图像
        return image

    def _preprocess_segmentation_map(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,  # 是否执行调整大小的操作，默认为 None
        size: Dict[str, int] = None,  # 调整大小的目标尺寸，默认为 None
        resample: PILImageResampling = None,  # 图像调整大小时的重采样方法，默认为 None
        do_center_crop: bool = None,  # 是否执行中心裁剪操作，默认为 None
        crop_size: Dict[str, int] = None,  # 中心裁剪的目标尺寸，默认为 None
        do_reduce_labels: bool = None,  # 是否执行标签减少操作，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为 None
    ):
        """Preprocesses a single segmentation map."""
        # 将分割地图预处理成单个分割地图
        # 将分割地图转换为 numpy 数组
        segmentation_map = to_numpy_array(segmentation_map)
        # 为了进行转换，为分割地图添加一个轴
        if segmentation_map.ndim == 2:
            segmentation_map = segmentation_map[None, ...]
            added_dimension = True
            input_data_format = ChannelDimension.FIRST
        else:
            added_dimension = False
            # 如果输入数据格式未指定，则推断通道维度的格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        # 调用 _preprocess 方法对分割地图进行预处理
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
        # 如果添加了额外的轴，则删除它
        if added_dimension:
            segmentation_map = np.squeeze(segmentation_map, axis=0)
        # 将分割地图转换为 int64 类型
        segmentation_map = segmentation_map.astype(np.int64)
        # 返回预处理后的分割地图
        return segmentation_map

    def __call__(self, images, segmentation_maps=None, **kwargs):
        # 重写 Preprocessor 类的 `__call__` 方法，使得可以将图像和分割地图都作为位置参数传入
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
        # 获取模型输出的logits
        logits = outputs.logits

        # 调整logits的大小并计算语义分割图
        if target_sizes is not None:
            # 如果传入的target_sizes不为空
            if len(logits) != len(target_sizes):
                # 检查传入的target_sizes数量是否与logits的batch维度数量相匹配
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # 如果target_sizes是torch张量，则转换为numpy数组
            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            # 初始化存储语义分割图的列表
            semantic_segmentation = []

            # 遍历每个logits并进行处理
            for idx in range(len(logits)):
                # 调整logits的大小为目标大小，并使用双线性插值方法进行插值
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # 获取调整大小后的logits中最大值所在的索引，即语义类别的标识
                semantic_map = resized_logits[0].argmax(dim=0)
                # 将语义类别标识添加到语义分割图列表中
                semantic_segmentation.append(semantic_map)
        else:
            # 如果未传入target_sizes，则直接对logits进行处理
            # 获取logits中每个像素最大概率对应的类别标识
            semantic_segmentation = logits.argmax(dim=1)
            # 将每个样本的语义分割结果添加到列表中
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # 返回语义分割图列表
        return semantic_segmentation
```