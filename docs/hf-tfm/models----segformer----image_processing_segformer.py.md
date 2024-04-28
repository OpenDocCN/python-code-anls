# `.\transformers\models\segformer\image_processing_segformer.py`

```py
# 设置编码格式为 utf-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非遵守许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”基础分发软件，没有任何担保或条件，无论是明示还是暗示
# 请参阅许可证以获取权限和限制
"""Segformer 的图像处理器类。"""

# 引入警告模块
import warnings
# 引入类型提示模块
from typing import Any, Dict, List, Optional, Tuple, Union
# 引入 numpy 模块
import numpy as np

# 引入图像处理实用程序模块
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 引入图像变换模块
from ...image_transforms import resize, to_channel_dimension_format
# 引入图像工具包模块
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
)
# 引入实用工具模块
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging

# 如果有 torch 模块可用，则引入 torch 模块
if is_torch_available():
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 SegformerImageProcessor 类，继承自 BaseImageProcessor 类
class SegformerImageProcessor(BaseImageProcessor):
    r"""
    构建一个 Segformer 图像处理器。
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的（高度，宽度）尺寸到指定的 `(size["height"], size["width"])`。可以通过 `preprocess` 方法中的 `do_resize` 参数进行覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 512, "width": 512}`):
            调整后输出图像的尺寸。可以通过 `preprocess` 方法中的 `size` 参数进行覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            调整图像时要使用的重采样滤波器。可以通过 `preprocess` 方法中的 `resample` 参数进行覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按照指定比例 `rescale_factor` 进行重新缩放图像。可以通过 `preprocess` 方法中的 `do_rescale` 参数进行覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            是否对图像进行归一化。可以通过 `preprocess` 方法中的 `do_normalize` 参数进行覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以通过 `preprocess` 方法中的 `do_normalize` 参数进行覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            在归一化图像时使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以通过 `preprocess` 方法中的 `image_mean` 参数进行覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            在归一化图像时使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以通过 `preprocess` 方法中的 `image_std` 参数进行覆盖。
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            是否将所有分割地图的标签值减去1。通常用于数据集中将0用作背景，并且背景本身不包含在数据集的所有类中的情况（例如ADE20k）。背景标签将被替换为255。可以通过 `preprocess` 方法中的 `do_reduce_labels` 参数进行覆盖。
    """

    model_input_names = ["pixel_values"]
    # 初始化函数，设置各种参数的默认值
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行调整大小，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样算法，默认为双线性插值
        do_rescale: bool = True,  # 是否进行重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否进行归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，默认为None
        do_reduce_labels: bool = False,  # 是否减少标签数量，默认为False
        **kwargs,
    ) -> None:
        # 如果kwargs中包含"reduce_labels"，则弹出警告信息
        if "reduce_labels" in kwargs:
            warnings.warn(
                "The `reduce_labels` parameter is deprecated and will be removed in a future version. Please use "
                "`do_reduce_labels` instead.",
                FutureWarning,
            )
            do_reduce_labels = kwargs.pop("reduce_labels")

        # 调用父类的初始化方法
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 512, "width": 512}  # 如果size为None，则设置默认值为{"height": 512, "width": 512}
        size = get_size_dict(size)  # 获取调整后的大小字典
        # 设置各参数的取值
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_reduce_labels = do_reduce_labels

    # 从字典创建实例的方法
    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure `do_reduce_labels` is updated if image
        processor is created using from_dict and kwargs e.g. `SegformerImageProcessor.from_pretrained(checkpoint,
        reduce_labels=True)`
        """
        image_processor_dict = image_processor_dict.copy()
        if "reduce_labels" in kwargs:
            image_processor_dict["reduce_labels"] = kwargs.pop("reduce_labels")
        return super().from_dict(image_processor_dict, **kwargs)

    # 图像调整大小的方法
    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],  # 调整后的大小字典
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样算法，默认为双线性插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为None
        **kwargs,
    # 此函数负责将输入的图像(image)根据指定的大小(size)进行调整尺寸
    # 它接受以下参数:
    #   - image: 待调整大小的图像,格式为numpy数组
    #   - size: 一个字典,包含"height"和"width"两个键,指定输出图像的目标尺寸
    #   - resample: 可选参数,指定调整大小时使用的resampling算法,默认为BILINEAR
    #   - data_format: 可选参数,指定输出图像的通道顺序,"channels_first","channels_last"或"none"
    #   - input_data_format: 可选参数,指定输入图像的通道顺序,"channels_first","channels_last"或"none"
    def resize(
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ) -> np.ndarray:
        # 检查size字典是否包含"height"和"width"两个键
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 根据size字典设置输出图像的尺寸
        output_size = (size["height"], size["width"])
        # 调用resize函数对图像进行缩放,并返回缩放后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    
    # 此函数用于将输入的标签(label)图像减小到0-255的范围
    # 它接受以下参数:
    #   - label: 待处理的标签图像,格式为numpy数组
    def reduce_label(self, label: ImageInput) -> np.ndarray:
        # 将输入标签图像转换为numpy数组
        label = to_numpy_array(label)
        # 将标签中值为0的像素点设置为255
        label[label == 0] = 255
        # 将标签中的值减1
        label = label - 1
        # 将标签中值为254的像素点设置为255
        label[label == 254] = 255
        # 返回处理后的标签图像
        return label
    # 对图像进行预处理
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
        # 如果需要减少标签，则调用reduce_label方法进行处理
        if do_reduce_labels:
            image = self.reduce_label(image)

        # 如果需要调整大小，则调用resize方法进行处理
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        # 如果需要重新缩放，则调用rescale方法进行处理
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        # 如果需要归一化，则调用normalize方法进行处理
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        # 返回预处理后的图像
        return image

    # 对单个图像进行预处理
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
        """预处理单个图像。"""
        # 所有的转换都期望numpy数组作为输入
        image = to_numpy_array(image)

        # 如果图像已经缩放，并且需要重新缩放，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        # 如果输入数据格式为None，则推断出通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # 调用_preprocess方法进行图像预处理
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

        # 如果指定了数据格式，则转换成相应的通道维度格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回预处理后的图像
        return image

    # 对掩模进行预处理
    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_reduce_labels: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        ) -> np.ndarray:
        """Preprocesses a single mask."""
        # 将分割图像转换为 NumPy 数组
        segmentation_map = to_numpy_array(segmentation_map)
        # 如果分割图像缺少通道维度，则添加通道维度 - 在某些转换中需要
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            # 如果未指定输入数据格式，则推断通道维度格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        # 如果需要，减少零标签
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
        # 如果为处理而添加了额外的通道维度，则移除它
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        # 将分割图像数据类型转换为 int64
        segmentation_map = segmentation_map.astype(np.int64)
        # 返回预处理后的分割图像
        return segmentation_map

    def __call__(self, images, segmentation_maps=None, **kwargs):
        """
        Preprocesses a batch of images and optionally segmentation maps.

        Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be
        passed in as positional arguments.
        """
        # 调用父类的 __call__ 方法，并传递参数 images, segmentation_maps 和 kwargs
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
        image_std: Optional[Union[float, List[float]] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 从 transformers.models.beit.image_processing_beit.BeitImageProcessor.post_process_semantic_segmentation 复制，将 Beit 改为 Segformer
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
        # 从模型输出中获取 logits
        logits = outputs.logits

        # 调整 logits 的大小并计算语义分割图
        if target_sizes is not None:
            # 如果目标大小已指定
            if len(logits) != len(target_sizes):
                # 确保目标大小的数量与 logits 的批量维度一致
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            if is_torch_tensor(target_sizes):
                # 如果目标大小是 PyTorch 张量，则转换为 NumPy 数组
                target_sizes = target_sizes.numpy()

            semantic_segmentation = []

            for idx in range(len(logits)):
                # 对每个样本进行循环
                # 调整 logits 的大小以匹配目标大小
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # 计算语义分割图，取最大值作为类别
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # 如果未指定目标大小
            # 对 logits 进行 argmax 操作得到语义分割图
            semantic_segmentation = logits.argmax(dim=1)
            # 将语义分割图存储在列表中
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # 返回语义分割图列表
        return semantic_segmentation
```  
```