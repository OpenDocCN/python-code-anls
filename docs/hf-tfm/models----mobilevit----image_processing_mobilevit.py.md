# `.\transformers\models\mobilevit\image_processing_mobilevit.py`

```
# 设定文件编码为 UTF-8
# 版权声明
# 定义了一个用于MobileViT的图像处理类
# 引入必要的库和模块
"""Image processor class for MobileViT."""  # MobileViT的图像处理类

from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示所需的模块

import numpy as np  # 导入NumPy库，用于数组处理

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 导入图像处理相关的模块
from ...image_transforms import flip_channel_order, get_resize_output_image_size, resize, to_channel_dimension_format  # 导入图像变换相关的模块
from ...image_utils import (  # 导入图像处理工具相关的模块
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging  # 导入相关的工具和判断模块

# 如果有torch库的话，导入torch
if is_vision_available():
    import PIL  # 如果有视觉库可用，导入PIL库

if is_torch_available():
    import torch  # 如果有torch可用，导入torch库

# 获取 logger
logger = logging.get_logger(__name__)
# 创建MobileViT的图像处理类，继承自BaseImageProcessor
class MobileViTImageProcessor(BaseImageProcessor):
    # 构造函数
    r"""
    Constructs a MobileViT image processor.
```  # 构造一个 MobileViT 图像处理类。
    # 定义类的初始化方法
        Args:
            do_resize (`bool`, *optional*, defaults to `True`):
                Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
                `do_resize` parameter in the `preprocess` method.
            size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
                Controls the size of the output image after resizing. Can be overridden by the `size` parameter in the
                `preprocess` method.
            resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
                Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter
                in the `preprocess` method.
            do_rescale (`bool`, *optional*, defaults to `True`):
                Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
                parameter in the `preprocess` method.
            rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
                Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
                `preprocess` method.
            do_center_crop (`bool`, *optional*, defaults to `True`):
                Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
                image is padded with 0's and then center cropped. Can be overridden by the `do_center_crop` parameter in
                the `preprocess` method.
            crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
                Desired output size `(size["height"], size["width"])` when applying center-cropping. Can be overridden by
                the `crop_size` parameter in the `preprocess` method.
            do_flip_channel_order (`bool`, *optional*, defaults to `True`):
                Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order`
                parameter in the `preprocess` method.
        """
    
        # 模型输入的名称列表
        model_input_names = ["pixel_values"]
    
        # 定义类的初始化方法
        def __init__(
            self,
            do_resize: bool = True,
            size: Dict[str, int] = None,
            resample: PILImageResampling = PILImageResampling.BILINEAR,
            do_rescale: bool = True,
            rescale_factor: Union[int, float] = 1 / 255,
            do_center_crop: bool = True,
            crop_size: Dict[str, int] = None,
            do_flip_channel_order: bool = True,
            **kwargs,
    # 这是一个图像处理类的初始化方法
    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Union[Dict[str, int], int]] = {"shortest_edge": 224},
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_center_crop: bool = True,
        crop_size: Optional[Union[Dict[str, int], int]] = {"height": 256, "width": 256},
        do_flip_channel_order: bool = False,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置图像大小，如果未指定则默认为最短边长224像素
        size = size if size is not None else {"shortest_edge": 224}
        # 将图像大小转换为字典格式
        size = get_size_dict(size, default_to_square=False)
        # 设置裁剪大小，如果未指定则默认为256x256
        crop_size = crop_size if crop_size is not None else {"height": 256, "width": 256}
        # 将裁剪大小转换为字典格式
        crop_size = get_size_dict(crop_size, param_name="crop_size")
    
        # 将各种参数赋值给实例属性
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_flip_channel_order = do_flip_channel_order
    
    # 定义了一个resize方法，用于调整图像大小
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.
    
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 如果size中包含"shortest_edge"键，则根据shortest_edge调整图像大小，并保持原有长宽比
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        # 如果size中包含"height"和"width"键，则根据它们调整图像大小
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        # 否则抛出ValueError异常
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")
    
        # 根据图像尺寸和指定的大小计算输出图像的尺寸
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        # 调用resize函数调整图像大小并返回结果
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    def flip_channel_order(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Flip the color channels from RGB to BGR or vice versa.

        Args:
            image (`np.ndarray`):
                The image, represented as a numpy array.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        return flip_channel_order(image, data_format=data_format, input_data_format=input_data_format)

    def __call__(self, images, segmentation_maps=None, **kwargs):
        """
        Preprocesses a batch of images and optionally segmentation maps.

        Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be
        passed in as positional arguments.
        """
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool,
        do_rescale: bool,
        do_center_crop: bool,
        do_flip_channel_order: bool,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        rescale_factor: Optional[float] = None,
        crop_size: Optional[Dict[str, int]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        if do_resize:
            # 如果需要改变尺寸，则调用resize方法
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_rescale:
            # 如果需要重新缩放，则调用rescale方法
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_center_crop:
            # 如果需要中心裁剪，则调用center_crop方法
            image = self.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

        if do_flip_channel_order:
            # 如果需要翻转通道顺序，则调用flip_channel_order方法
            image = self.flip_channel_order(image, input_data_format=input_data_format)

        return image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_flip_channel_order: bool = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # 所有的转换都需要 numpy 数组作为输入
        image = to_numpy_array(image)
        # 如果输入图像已经缩放，并且需要重新缩放，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # 推断输入图像通道维度的格式
            input_data_format = infer_channel_dimension_format(image)

        # 预处理图像
        image = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_flip_channel_order=do_flip_channel_order,
            input_data_format=input_data_format,
        )

        # 调整图像通道维度格式
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        return image

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single mask."""
        segmentation_map = to_numpy_array(segmentation_map)
        # 如果分割图是二维的，添加通道维度
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            if input_data_format is None:
                # 推断分割图通道维度的格式
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)

        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            size=size,
            resample=PILImageResampling.NEAREST,
            do_rescale=False,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_flip_channel_order=False,
            input_data_format=input_data_format,
        )
        # 如果为处理而添加了额外的通道维度，则移除
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        segmentation_map = segmentation_map.astype(np.int64)
        return segmentation_map
    # 定义一个预处理函数，用于对输入的图像进行预处理
    def preprocess(
        self,
        images: ImageInput,  # 输入的图像数据
        segmentation_maps: Optional[ImageInput] = None,  # 分割图的数据，可选
        do_resize: bool = None,  # 是否调整大小
        size: Dict[str, int] = None,  # 调整的大小
        resample: PILImageResampling = None,  # 重采样方法
        do_rescale: bool = None,  # 是否调整比例
        rescale_factor: float = None,  # 调整的比例因子
        do_center_crop: bool = None,  # 是否中心裁剪
        crop_size: Dict[str, int] = None,  # 裁剪的大小
        do_flip_channel_order: bool = None,  # 是否翻转通道顺序
        return_tensors: Optional[Union[str, TensorType]] = None,  # 是否返回张量
        data_format: ChannelDimension = ChannelDimension.FIRST,  # 数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式
        **kwargs,  # 其他参数
    # 从transformers.models.beit.image_processing_beit.BeitImageProcessor.post_process_semantic_segmentation中复制并将Beit->MobileViT
    # 定义一个用于后处理语义分割输出的函数
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`MobileViTForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`MobileViTForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.  # 目标尺寸，如果未设置，则表示预测结果不会被调整大小

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.  # 返回语义分割列表，每个项都是一个语义分割图

        # TODO: add support for other frameworks  # 添加对其他框架的支持
        logits = outputs.logits  # 获取输出的logits

        # 调整logits的大小并计算语义分割图  # 调整logits的大小并计算语义分割图
        if target_sizes is not None:  # 如果存在目标尺寸
            if len(logits) != len(target_sizes):  # 如果logits的长度与目标尺寸的长度不一致
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )  # 抛出值错误

            if is_torch_tensor(target_sizes):  # 如果目标尺寸是torch张量
                target_sizes = target_sizes.numpy()  # 转换为numpy格式

            semantic_segmentation = []  # 初始化语义分割列表

            for idx in range(len(logits)):  # 遍历logits
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )  # 插值调整logits大小
                semantic_map = resized_logits[0].argmax(dim=0)  # 计算语义分割图
                semantic_segmentation.append(semantic_map)  # 添加到语义分割列表中
        else:  # 如果不存在目标尺寸
            semantic_segmentation = logits.argmax(dim=1)  # 计算logits的最大值索引
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]  # 重新组织成列表形式

        return semantic_segmentation  # 返回语义分割结果
```