# `.\transformers\models\mobilenet_v2\image_processing_mobilenet_v2.py`

```
# 导入必要的模块和类
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ChannelDimension, ImageInput,
    PILImageResampling, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, is_torch_available, is_torch_tensor, logging

这一部分是代码导入和 Python 基础的各种类型定义和工具类的导入。
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否将图像的（高度，宽度）尺寸调整为指定的 `size`。可以通过 `preprocess` 方法中的 `do_resize` 参数进行覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            调整后图像的尺寸。图像的最短边将被调整为 size["shortest_edge"]，最长边将被调整以保持输入的纵横比。可以通过
            `preprocess` 方法中的 `size` 参数进行覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            如果调整图像大小，要使用的重采样滤波器。可以通过 `preprocess` 方法中的 `resample` 参数进行覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否对图像进行中心裁剪。如果输入尺寸小于 `crop_size` 的任何边，图像将填充为 0，然后进行中心裁剪。可以通过
            `preprocess` 方法中的 `do_center_crop` 参数进行覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            在应用中心裁剪时期望的输出尺寸。仅在 `do_center_crop` 设置为 `True` 时有效。可以通过 `preprocess` 方法中的
            `crop_size` 参数进行覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按照指定的比例 `rescale_factor` 进行重新缩放图像。可以通过 `preprocess` 方法中的 `do_rescale` 参数进行覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像，则要使用的缩放因子。可以通过 `preprocess` 方法中的 `rescale_factor` 参数进行覆盖。
        do_normalize:
            是否对图像进行归一化。可以通过 `preprocess` 方法中的 `do_normalize` 参数进行覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果对图像进行归一化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以通过 `preprocess` 方法中的
            `image_mean` 参数进行覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果对图像进行归一化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以通过 `preprocess` 方法中的
            `image_std` 参数进行覆盖。
    """

    model_input_names = ["pixel_values"]
    # 初始化方法，用于创建一个新的图像处理器对象
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整，默认为True
        size: Optional[Dict[str, int]] = None,  # 图像大小的字典，键为尺寸参数，值为整数，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整过程中使用的重采样方法，默认为双线性插值
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪尺寸的字典，键为裁剪参数，值为整数，默认为None
        do_rescale: bool = True,  # 是否进行图像重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像重新缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，默认为None
        **kwargs,  # 其他关键字参数
    ) -> None:  # 返回空值
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果没有指定图像大小，则设置默认的最短边为256的尺寸
        size = size if size is not None else {"shortest_edge": 256}
        # 获取调整后的图像大小字典，保证不会失真
        size = get_size_dict(size, default_to_square=False)
        # 如果没有指定裁剪尺寸，则设置默认的高度和宽度都为224的尺寸
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 获取调整后的裁剪尺寸字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")
        # 初始化各参数
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
    
    # 从transformers.models.mobilenet_v1.image_processing_mobilenet_v1.MobileNetV1ImageProcessor.resize复制而来的方法
    def resize(
        self,
        image: np.ndarray,  # 待调整大小的图像数组
        size: Dict[str, int],  # 调整后的尺寸字典
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为None
        **kwargs,  # 其他关键字参数
    )

这是一个方法的定义，方法名为`resize_image`，它接受以下参数：
- `self`：当前实例对象
- `image`：要调整大小的图像，类型为`np.ndarray`
- `size`：输出图像的大小，类型为`Dict[str, int]`
- `resample`：调整大小时使用的重新采样滤波器，默认值为`PILImageResampling.BICUBIC`，类型为`PILImageResampling`
- `data_format`：图像的通道维度格式，类型为`str`或`ChannelDimension`
- `input_data_format`：输入图像的通道维度格式，类型为`ChannelDimension`或`str`

        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """

这是`resize_image`方法的文档字符串，其中详细说明了方法的作用、参数和返回值。

        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

首先，将`default_to_square`变量设置为`True`。然后使用条件语句判断`size`字典中的内容：
- 如果`size`字典中包含键`shortest_edge`，则将`size`赋值为`size["shortest_edge"]`，同时将`default_to_square`设置为`False`。
- 否则，如果`size`字典中包含`height`和`width`两个键，则将`size`赋值为元组`(size["height"], size["width"])`。
- 否则，抛出`ValueError`异常，提示`size`字典必须包含`shortest_edge`或`height`和`width`。

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )

调用名为`get_resize_output_image_size`的函数，传入以下参数：
- `image`：要调整大小的图像
- `size`：输出图像的大小
- `default_to_square`：是否默认为方形
- `input_data_format`：输入图���的通道维度格式
将返回值赋值给`output_size`变量。

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

调用名为`resize`的函数，传入以下参数：
- `image`：要调整大小的图像
- `size`：输出图像的大小
- `resample`：重新采样滤波器
- `data_format`：图像的通道维度格式
- `input_data_format`：输入图像的通道维度格式
同时，使用`**kwargs`传递任意其他关键字参数。将函数的返回值作为结果进行返回。

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.post_process_semantic_segmentation with Beit->MobileNetV2

这是一个方法的定义，方法名为`preprocess`，它接受很多参数，包括：
- `self`：当前实例对象
- `images`：图像输入
- `do_resize`：是否调整大小的标志
- `size`：输出图像的大小
- `resample`：重新采样滤波器
- `do_center_crop`：是否进行中心裁剪的标志
- `crop_size`：裁剪后的大小
- `do_rescale`：是否进行重新缩放的标志
- `rescale_factor`：重新缩放的因子
- `do_normalize`：是否进行规范化的标志
- `image_mean`：图像均值
- `image_std`：图像标准差
- `return_tensors`：返回的张量类型
- `data_format`：图像的通道维度格式
- `input_data_format`：输入图像的通道维度格式
- `**kwargs`：任意其他关键字参数。
    # 后处理语义分割结果，将模型输出转换为语义分割图。仅支持 PyTorch。

    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`MobileNetV2ForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`MobileNetV2ForSemanticSegmentation`]):
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
        
        # 从输出中提取 logits
        logits = outputs.logits

        # 调整 logits 的大小并计算语义分割图
        if target_sizes is not None:
            # 如果目标大小不为空
            if len(logits) != len(target_sizes):
                # 如果 logits 的数量与目标大小的数量不一致，则引发 ValueError
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # 将目标大小转换为 NumPy 数组，如果目标大小是张量的话
            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            # 存储语义分割图的列表
            semantic_segmentation = []

            # 遍历每个样本的 logits
            for idx in range(len(logits)):
                # 调整 logits 的大小
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # 计算每个像素点的语义类别
                semantic_map = resized_logits[0].argmax(dim=0)
                # 将语义分割图添加到结果列表中
                semantic_segmentation.append(semantic_map)
        else:
            # 如果目标大小为空
            # 计算整个 batch 的语义分割图
            semantic_segmentation = logits.argmax(dim=1)
            # 将每个样本的语义分割图存储为单独的张量
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # 返回语义分割图列表
        return semantic_segmentation
```