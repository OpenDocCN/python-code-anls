# `.\models\sam\image_processing_sam.py`

```
# 指定编码格式为 UTF-8
# 版权声明和许可信息
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 获取许可证的详细信息，请访问指定的 URL
# 如果适用法律要求或书面同意，本软件按“原样”分发，不提供任何明示或暗示的担保或条件
# 请查阅许可证了解具体语言和限制条款
"""SAM 的图像处理类。"""
# 导入所需的库和模块
import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 NumPy 库，并使用 np 别名
import numpy as np

# 导入 Hugging Face 库中的图像处理相关工具和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入 Hugging Face 库中的图像转换函数和工具
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
# 导入 Hugging Face 库中的图像处理工具函数
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
# 导入 Hugging Face 库中的通用工具和函数
from ...utils import (
    TensorType,
    is_tf_available,
    is_torch_available,
    is_torchvision_available,
    logging,
    requires_backends,
)

# 如果 Torch 可用，则导入 Torch 库和 Torch 的功能模块
if is_torch_available():
    import torch
    import torch.nn.functional as F

# 如果 TorchVision 可用，则从 TorchVision 中导入批量 NMS 函数
if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms

# 如果 TensorFlow 可用，则导入 TensorFlow 库和实验性的 NumPy 模块
if is_tf_available():
    import tensorflow as tf
    from tensorflow.experimental import numpy as tnp
    # 导入 Hugging Face 库中的 TensorFlow 相关工具函数
    from ...tf_utils import flatten, shape_list

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 SAM 图像处理器类，继承自 BaseImageProcessor 类
class SamImageProcessor(BaseImageProcessor):
    r"""
    构造 SAM 图像处理器。
    """

    # 模型输入名称列表，此处仅包含 'pixel_values'
    model_input_names = ["pixel_values"]

    # SAM 图像处理器的初始化方法
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        mask_size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        pad_size: int = None,
        mask_pad_size: int = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        # 调用父类初始化方法，传入任意关键字参数
        super().__init__(**kwargs)
        # 如果 size 为 None，则设置默认最长边为 1024 的大小字典
        size = size if size is not None else {"longest_edge": 1024}
        # 如果 size 不是字典，则调用函数获取大小字典，不默认转换为正方形
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size

        # 如果 pad_size 为 None，则设置默认高度和宽度都为 1024 的大小字典
        pad_size = pad_size if pad_size is not None else {"height": 1024, "width": 1024}
        # 调用函数获取大小字典，默认转换为正方形
        pad_size = get_size_dict(pad_size, default_to_square=True)

        # 如果 mask_size 为 None，则设置默认最长边为 256 的大小字典
        mask_size = mask_size if mask_size is not None else {"longest_edge": 256}
        # 如果 mask_size 不是字典，则调用函数获取大小字典，不默认转换为正方形
        mask_size = (
            get_size_dict(max_size=mask_size, default_to_square=False)
            if not isinstance(mask_size, dict)
            else mask_size
        )

        # 如果 mask_pad_size 为 None，则设置默认高度和宽度都为 256 的大小字典
        mask_pad_size = mask_pad_size if mask_pad_size is not None else {"height": 256, "width": 256}
        # 调用函数获取大小字典，默认转换为正方形
        mask_pad_size = get_size_dict(mask_pad_size, default_to_square=True)

        # 设置属性，是否进行 resize 操作
        self.do_resize = do_resize
        # 设置属性，图片大小的大小字典
        self.size = size
        # 设置属性，mask 的大小字典
        self.mask_size = mask_size
        # 设置属性，重采样方式
        self.resample = resample
        # 设置属性，是否进行 rescale 操作
        self.do_rescale = do_rescale
        # 设置属性，rescale 的因子
        self.rescale_factor = rescale_factor
        # 设置属性，是否进行 normalize 操作
        self.do_normalize = do_normalize
        # 设置属性，图片的均值，默认为 IMAGENET 的默认均值
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        # 设置属性，图片的标准差，默认为 IMAGENET 的默认标准差
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        # 设置属性，是否进行 pad 操作
        self.do_pad = do_pad
        # 设置属性，pad 的大小字典
        self.pad_size = pad_size
        # 设置属性，mask 的 pad 的大小字典
        self.mask_pad_size = mask_pad_size
        # 设置属性，是否进行 RGB 转换
        self.do_convert_rgb = do_convert_rgb
        # 设置属性，有效的处理器键列表
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "do_resize",
            "size",
            "mask_size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_pad",
            "pad_size",
            "mask_pad_size",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    def pad_image(
        self,
        image: np.ndarray,
        pad_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Pad an image to `(pad_size["height"], pad_size["width"])` with zeros to the right and bottom.

        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`Dict[str, int]`):
                Size of the output image after padding.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the image. Can be either "channels_first" or "channels_last". If `None`, the
                `data_format` of the `image` will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取目标填充后的输出高度和宽度
        output_height, output_width = pad_size["height"], pad_size["width"]
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        # 计算需要填充的宽度和高度
        pad_width = output_width - input_width
        pad_height = output_height - input_height

        # 使用零填充图像到目标大小
        padded_image = pad(
            image,
            ((0, pad_height), (0, pad_width)),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return padded_image

    def _get_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int):
        """
        Compute the output size given input size and target long side length.
        """
        # 获取输入形状的高度和宽度
        oldh, oldw = old_shape
        # 计算缩放比例，确保最长边达到目标长度
        scale = longest_edge * 1.0 / max(oldh, oldw)
        # 计算新的高度和宽度
        newh, neww = oldh * scale, oldw * scale
        # 四舍五入并转换为整数
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
        ):
        """
        Resize an image to a specific size.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Target size of the output image after resizing, specified as {'height': height, 'width': width}.
            resample (`PILImageResampling`, *optional*):
                Resampling method. Default is `PILImageResampling.BICUBIC`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the image. Can be either "channels_first" or "channels_last". If `None`, the
                `data_format` of the `image` will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # TODO: Add implementation for image resizing
        pass
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest
                edge of the image will be resized to the specified size, while the other edge will be resized to
                maintain the aspect ratio.
            resample:
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)  # 调用函数 get_size_dict 将 size 转换为标准格式的尺寸字典
        if "longest_edge" not in size:
            raise ValueError(f"The `size` dictionary must contain the key `longest_edge`. Got {size.keys()}")
        input_size = get_image_size(image, channel_dim=input_data_format)  # 获取输入图片的尺寸信息
        output_height, output_width = self._get_preprocess_shape(input_size, size["longest_edge"])  # 根据输入和输出尺寸计算预处理后的图像尺寸
        return resize(
            image,
            size=(output_height, output_width),  # 调整图像大小为指定的输出尺寸
            resample=resample,  # 使用指定的重采样方法
            data_format=data_format,  # 输出图像的通道顺序格式
            input_data_format=input_data_format,  # 输入图像的通道顺序格式
            **kwargs,  # 其它可选参数
        )

    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # 如果需要调整大小，则调用 resize 方法调整图像大小
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        # 获取调整后图像的大小
        reshaped_input_size = get_image_size(image, channel_dim=input_data_format)

        # 如果需要重新缩放，则调用 rescale 方法重新缩放图像
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        # 如果需要归一化，则调用 normalize 方法对图像进行归一化处理
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        # 如果需要填充，则调用 pad_image 方法对图像进行填充处理
        if do_pad:
            image = self.pad_image(image=image, pad_size=pad_size, input_data_format=input_data_format)

        # 返回预处理后的图像及其调整前后的大小信息
        return image, reshaped_input_size

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        # 将图像转换为 numpy 数组
        image = to_numpy_array(image)

        # 如果需要将 PIL RGBA 图像转换为 RGB 格式
        if do_convert_rgb:
            image = convert_to_rgb(image)

        # 所有的转换操作都期望输入为 numpy 数组
        image = to_numpy_array(image)

        # 如果输入图像已经进行了缩放并且需要重新缩放，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        # 推断输入数据的通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # 获取原始图像的大小
        original_size = get_image_size(image, channel_dim=input_data_format)

        # 对图像进行预处理，并获取预处理后的图像及其调整后的大小信息
        image, reshaped_input_size = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            pad_size=pad_size,
            input_data_format=input_data_format,
        )

        # 如果指定了输出数据格式，则将图像转换为该格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回最终的预处理结果，包括图像及其原始大小和调整后的大小信息
        return image, original_size, reshaped_input_size
    # 对分割地图进行预处理，返回处理后的分割地图和原始尺寸
    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: Optional[bool] = None,
        mask_size: Dict[str, int] = None,
        do_pad: Optional[bool] = None,
        mask_pad_size: Optional[Dict[str, int]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        # 将分割地图转换为 NumPy 数组
        segmentation_map = to_numpy_array(segmentation_map)

        # 如果分割地图是二维的，则添加通道维度，某些转换需要此维度
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]  # 添加通道维度
            input_data_format = ChannelDimension.FIRST  # 设置数据格式为通道维度在最前面
        else:
            added_channel_dim = False
            if input_data_format is None:
                # 推断通道维度格式，确保一维通道格式
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)

        # 获取原始图像尺寸
        original_size = get_image_size(segmentation_map, channel_dim=input_data_format)

        # 对分割地图进行预处理，包括调整大小、填充等操作
        segmentation_map, _ = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            size=mask_size,
            resample=PILImageResampling.NEAREST,
            do_rescale=False,
            do_normalize=False,
            do_pad=do_pad,
            pad_size=mask_pad_size,
            input_data_format=input_data_format,
        )

        # 如果之前添加了额外的通道维度，则在处理完成后去除
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)  # 去除添加的通道维度
        segmentation_map = segmentation_map.astype(np.int64)  # 将分割地图转换为整型

        # 返回处理后的分割地图和原始尺寸
        return segmentation_map, original_size

    # 对图像及其分割地图进行预处理，支持多种处理选项
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        mask_size: Optional[Dict[str, int]] = None,
        resample: Optional["PILImageResampling"] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        mask_pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        # 函数用于图像和分割地图的预处理，支持多种选项和参数配置
        pass

    # 对预处理后的掩模进行后处理，包括阈值处理和二值化等
    def post_process_masks(
        self,
        masks,
        original_sizes,
        reshaped_input_sizes,
        mask_threshold=0.0,
        binarize=True,
        pad_size=None,
        return_tensors="pt",
        **kwargs,
    ):
        # 对预处理后的掩模进行后处理，支持阈值处理、二值化和填充操作
        pass
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[torch.Tensor], List[np.ndarray], List[tf.Tensor]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, tf.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            reshaped_input_sizes (`Union[torch.Tensor, tf.Tensor, List[Tuple[int,int]]]`):
                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                If `"pt"`, return PyTorch tensors. If `"tf"`, return TensorFlow tensors.
        Returns:
            (`Union[torch.Tensor, tf.Tensor]`): Batched masks in batch_size, num_channels, height, width) format, where
            (height, width) is given by original_size.
        """
        # 根据 return_tensors 参数选择使用 PyTorch 或 TensorFlow 的后处理函数
        if return_tensors == "pt":
            return self._post_process_masks_pt(
                masks=masks,
                original_sizes=original_sizes,
                reshaped_input_sizes=reshaped_input_sizes,
                mask_threshold=mask_threshold,
                binarize=binarize,
                pad_size=pad_size,
            )
        elif return_tensors == "tf":
            return self._post_process_masks_tf(
                masks=masks,
                original_sizes=original_sizes,
                reshaped_input_sizes=reshaped_input_sizes,
                mask_threshold=mask_threshold,
                binarize=binarize,
                pad_size=pad_size,
            )
        else:
            # 如果 return_tensors 参数既不是 "pt" 也不是 "tf"，抛出数值错误异常
            raise ValueError("return_tensors must be either 'pt' or 'tf'")

    def _post_process_masks_pt(
        self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[torch.Tensor], List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            reshaped_input_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        requires_backends(self, ["torch"])  # 确保当前环境支持 torch
        pad_size = self.pad_size if pad_size is None else pad_size  # 如果未指定 pad_size，则使用类的默认值
        target_image_size = (pad_size["height"], pad_size["width"])  # 获取目标图像尺寸 (height, width)

        if isinstance(original_sizes, (torch.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()  # 将 original_sizes 转换为列表形式
        if isinstance(reshaped_input_sizes, (torch.Tensor, np.ndarray)):
            reshaped_input_sizes = reshaped_input_sizes.tolist()  # 将 reshaped_input_sizes 转换为列表形式

        output_masks = []  # 初始化空列表，用于存储输出的 masks

        for i, original_size in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])  # 如果 masks[i] 是 np.ndarray，则转换为 torch.Tensor
            elif not isinstance(masks[i], torch.Tensor):
                raise ValueError("Input masks should be a list of `torch.tensors` or a list of `np.ndarray`")

            # 插值操作，将 masks[i] 缩放到 target_image_size 大小
            interpolated_mask = F.interpolate(masks[i], target_image_size, mode="bilinear", align_corners=False)
            # 截取插值后的结果，保留至 reshaped_input_sizes[i] 大小
            interpolated_mask = interpolated_mask[..., : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1]]
            # 再次插值，将 interpolated_mask 缩放至 original_size 大小
            interpolated_mask = F.interpolate(interpolated_mask, original_size, mode="bilinear", align_corners=False)

            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold  # 根据阈值进行二值化处理

            output_masks.append(interpolated_mask)  # 将处理后的 mask 添加到输出列表中

        return output_masks  # 返回处理后的输出 masks
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`tf.Tensor`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`tf.Tensor`):
                The original size of the images before resizing for input to the model, in (height, width) format.
            reshaped_input_sizes (`tf.Tensor`):
                The size of the image input to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
        Returns:
            (`tf.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width) is
            given by original_size.
        """
        # Ensure the necessary backend for operations
        requires_backends(self, ["tf"])
        
        # Determine the padding size to use
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size["height"], pad_size["width"])

        output_masks = []
        for i, original_size in enumerate(original_sizes):
            # Transpose masks to NHWC format as required by tf.image functions
            mask = tf.transpose(masks[i], perm=[0, 2, 3, 1])
            
            # Resize masks to match target_image_size
            interpolated_mask = tf.image.resize(mask, target_image_size, method="bilinear")
            
            # Remove padding from resized masks based on reshaped_input_sizes
            interpolated_mask = interpolated_mask[:, : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1], :]
            
            # Resize masks to original_size
            interpolated_mask = tf.image.resize(interpolated_mask, original_size, method="bilinear")
            
            # Binarize masks if specified
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            
            # Transpose masks back to original NCHW format
            output_masks.append(tf.transpose(interpolated_mask, perm=[0, 3, 1, 2]))

        return output_masks
        ):
        """
        Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

        Args:
            all_masks (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all predicted segmentation masks
            all_scores (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all predicted iou scores
            all_boxes (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all bounding boxes of the predicted masks
            crops_nms_thresh (`float`):
                Threshold for NMS (Non Maximum Suppression) algorithm.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        if return_tensors == "pt":
            # 如果返回类型是 `pt`，调用 _postprocess_for_mg 函数对预测的mask进行后处理
            return _postprocess_for_mg(all_masks, all_scores, all_boxes, crops_nms_thresh)
        elif return_tensors == "tf":
            # 如果返回类型是 `tf`，调用 _postprocess_for_mg_tf 函数对预测的mask进行后处理
            return _postprocess_for_mg_tf(all_masks, all_scores, all_boxes, crops_nms_thresh)

    def generate_crop_boxes(
        self,
        image,
        target_size,
        crop_n_layers: int = 0,
        overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        crop_n_points_downscale_factor: Optional[List[int]] = 1,
        device: Optional["torch.device"] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        return_tensors: str = "pt",
    ):
        """
        Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

        Args:
            image (`np.array`):
                Input original image
            target_size (`int`):
                Target size of the resized image
            crop_n_layers (`int`, *optional*, defaults to 0):
                If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where
                each layer has 2**i_layer number of image crops.
            overlap_ratio (`float`, *optional*, defaults to 512/1500):
                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            points_per_crop (`int`, *optional*, defaults to 32):
                Number of points to sample from each crop.
            crop_n_points_downscale_factor (`List[int]`, *optional*, defaults to 1):
                The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            device (`torch.device`, *optional*, defaults to None):
                Device to use for the computation. If None, cpu will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        # Generate crop boxes, sample points, cropped images, and labels from the input image
        crop_boxes, points_per_crop, cropped_images, input_labels = _generate_crop_boxes(
            image,
            target_size,
            crop_n_layers,
            overlap_ratio,
            points_per_crop,
            crop_n_points_downscale_factor,
            input_data_format,
        )
        # Convert outputs to PyTorch tensors if return_tensors is 'pt'
        if return_tensors == "pt":
            # If device is not specified, default to CPU
            if device is None:
                device = torch.device("cpu")
            # Convert crop boxes, points_per_crop, and input_labels to PyTorch tensors
            crop_boxes = torch.tensor(crop_boxes, device=device)
            points_per_crop = torch.tensor(points_per_crop, device=device)
            # cropped_images remains as NumPy array
            input_labels = torch.tensor(input_labels, device=device)

        # Convert outputs to TensorFlow tensors if return_tensors is 'tf'
        elif return_tensors == "tf":
            # TensorFlow does not support device specification in this context
            if device is not None:
                raise ValueError("device is not a supported argument when return_tensors is tf!")
            # Convert crop boxes, points_per_crop, and input_labels to TensorFlow tensors
            crop_boxes = tf.convert_to_tensor(crop_boxes)
            points_per_crop = tf.convert_to_tensor(points_per_crop)
            # cropped_images remains as NumPy array
            input_labels = tf.convert_to_tensor(input_labels)
        else:
            # Raise an error if return_tensors is neither 'pt' nor 'tf'
            raise ValueError("return_tensors must be either 'pt' or 'tf'.")
        # Return generated crop boxes, points per crop, cropped images, and input labels
        return crop_boxes, points_per_crop, cropped_images, input_labels
        """
        根据给定的条件过滤预测的掩码，并执行必要的转换和填充操作。

        Args:
            masks (`Union[torch.Tensor, tf.Tensor]`):
                输入的掩码张量。
            iou_scores (`Union[torch.Tensor, tf.Tensor]`):
                IoU（Intersection over Union）分数的列表。
            original_size (`Tuple[int,int]`):
                原始图像的尺寸。
            cropped_box_image (`np.array`):
                裁剪后的图像数组。
            pred_iou_thresh (`float`, *optional*, 默认为 0.88):
                IoU 分数的阈值。
            stability_score_thresh (`float`, *optional*, 默认为 0.95):
                稳定性分数的阈值。
            mask_threshold (`float`, *optional*, 默认为 0):
                预测掩码的阈值。
            stability_score_offset (`float`, *optional*, 默认为 1):
                在 `_compute_stability_score` 方法中使用的稳定性分数的偏移量。
            return_tensors (`str`, *optional*, 默认为 `pt`):
                如果是 `pt`，返回 `torch.Tensor`；如果是 `tf`，返回 `tf.Tensor`。
        """
        if return_tensors == "pt":
            # 调用基于 PyTorch 的掩码过滤方法
            return self._filter_masks_pt(
                masks=masks,
                iou_scores=iou_scores,
                original_size=original_size,
                cropped_box_image=cropped_box_image,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                mask_threshold=mask_threshold,
                stability_score_offset=stability_score_offset,
            )
        elif return_tensors == "tf":
            # 调用基于 TensorFlow 的掩码过滤方法
            return self._filter_masks_tf(
                masks=masks,
                iou_scores=iou_scores,
                original_size=original_size,
                cropped_box_image=cropped_box_image,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                mask_threshold=mask_threshold,
                stability_score_offset=stability_score_offset,
            )
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`torch.Tensor`):
                Input masks.
            iou_scores (`torch.Tensor`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the original image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the IoU scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.

        """
        # Ensure the torch backend is available
        requires_backends(self, ["torch"])
        
        # Extract dimensions of the original image
        original_height, original_width = original_size
        
        # Flatten masks and IoU scores for easier manipulation
        iou_scores = iou_scores.flatten(0, 1)
        masks = masks.flatten(0, 1)
        
        # Check if the number of masks matches the number of IoU scores
        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError("masks and iou_scores must have the same batch size.")
        
        # Ensure masks and IoU scores are on the same device
        if masks.device != iou_scores.device:
            iou_scores = iou_scores.to(masks.device)
        
        # Determine batch size from the flattened masks
        batch_size = masks.shape[0]
        
        # Initialize a mask to keep all masks (defaulting to True)
        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)
        
        # Apply filtering based on IoU threshold
        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)
        
        # Compute stability scores and filter based on stability score threshold
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_pt(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)
        
        # Select scores and masks that meet the criteria
        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]
        
        # Binarize masks and convert them to bounding boxes
        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box(masks)
        
        # Check if boxes are near the cropped image edges and filter accordingly
        keep_mask = ~_is_box_near_crop_edge(
            converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
        )
        
        # Select final scores, masks, and converted boxes
        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]
        
        # Pad masks to original image dimensions and convert to RLE format
        masks = _pad_masks(masks, cropped_box_image, original_height, original_width)
        masks = _mask_to_rle_pytorch(masks)
        
        return masks, scores, converted_boxes
    # 定义一个私有方法 `_filter_masks_tf`，用于在 TensorFlow 中过滤掩码
    # 参数 `masks`: 掩码数据
    # 参数 `iou_scores`: IoU（交并比）分数
    # 参数 `original_size`: 原始图像尺寸
    # 参数 `cropped_box_image`: 裁剪后的图像框
    # 参数 `pred_iou_thresh`: 预测 IoU 阈值，默认为 0.88
    # 参数 `stability_score_thresh`: 稳定性分数阈值，默认为 0.95
    # 参数 `mask_threshold`: 掩码阈值，默认为 0
    # 参数 `stability_score_offset`: 稳定性分数偏移，默认为 1
    ):
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`tf.Tensor`):
                Input masks.
            iou_scores (`tf.Tensor`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the orginal image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.

        """
        # Ensure necessary backend support for TensorFlow
        requires_backends(self, ["tf"])
        # Extract dimensions of the original image
        original_height, original_width = original_size
        # Reshape IoU scores tensor for processing
        iou_scores = tf.reshape(iou_scores, [iou_scores.shape[0] * iou_scores.shape[1], iou_scores.shape[2:]])
        # Reshape masks tensor for processing
        masks = tf.reshape(masks, [masks.shape[0] * masks.shape[1], masks.shape[2:]])

        # Check if batch sizes of masks and IoU scores match
        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError("masks and iou_scores must have the same batch size.")

        # Retrieve batch size from masks tensor
        batch_size = masks.shape[0]

        # Initialize a mask to keep all elements
        keep_mask = tf.ones(batch_size, dtype=tf.bool)

        # Apply filter based on IoU threshold if specified
        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)

        # Compute stability scores and apply filter based on stability score threshold if specified
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_tf(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)

        # Filter out masks and scores based on the keep_mask
        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]

        # Binarize masks
        masks = masks > mask_threshold

        # Convert masks to bounding boxes
        converted_boxes = _batched_mask_to_box_tf(masks)

        # Filter out boxes near the cropped image edges
        keep_mask = ~_is_box_near_crop_edge_tf(
            converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
        )

        # Refilter masks, scores, and converted boxes based on the updated keep_mask
        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]

        # Pad masks to match original image dimensions
        masks = _pad_masks_tf(masks, cropped_box_image, original_height, original_width)

        # Convert masks to RLE format for non-maximum suppression
        masks = _mask_to_rle_tf(masks)

        # Return filtered masks, scores, and converted boxes
        return masks, scores, converted_boxes
# 计算两个掩码之间的稳定性评分的函数，对于每个掩码，使用阈值和偏移量计算与另一个掩码的交集并集比率。

def _compute_stability_score_pt(masks: "torch.Tensor", mask_threshold: float, stability_score_offset: int):
    # 计算两个掩码之间的交集数量，避免不必要的cast到torch.int64，使用int16和int32作为中间类型以节省内存。
    intersections = (
        (masks > (mask_threshold + stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    )
    
    # 计算两个掩码之间的并集数量，使用相同的中间数据类型作为内存优化措施。
    unions = (masks > (mask_threshold - stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    
    # 使用交集和并集数量计算稳定性分数。
    stability_scores = intersections / unions
    
    # 返回稳定性分数。
    return stability_scores


# 使用TensorFlow进行相同功能的计算。将mask_threshold和stability_score_offset转换为浮点类型来确保正确进行除法。
def _compute_stability_score_tf(masks: "tf.Tensor", mask_threshold: float, stability_score_offset: int):
    intersections = tf.count_nonzero(
        masks > (mask_threshold + stability_score_offset), axis=[-1, -2], dtype=tf.float32
    )
    unions = tf.count_nonzero(masks > (mask_threshold - stability_score_offset), axis=[-1, -2], dtype=tf.float32)
    stability_scores = intersections / unions
    
    # 返回稳定性分数。
    return stability_scores


# 生成2D网格点列表，这些点在[0,1]x[0,1]区间用等间距插入。
def _build_point_grid(n_per_side: int) -> np.ndarray:
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


# 对坐标进行标准化，以适应给定的目标尺寸，考虑到原始尺寸和所需的乘法因子。
def _normalize_coordinates(
    target_size: int, coords: np.ndarray, original_size: Tuple[int, int], is_bounding_box=False
) -> np.ndarray:
    old_height, old_width = original_size
    scale = target_size * 1.0 / max(old_height, old_width)
    new_height, new_width = old_height * scale, old_width * scale
    new_width, new_height = int(new_width + 0.5), int(new_height + 0.5)

    # 复制输入数组并将其转换为float类型。
    coords = deepcopy(coords).astype(float)

    if is_bounding_box:
        coords = coords.reshape(-1, 2, 2)

    # 标准化坐标值。
    coords[..., 0] = coords[..., 0] * (new_width / old_width)
    coords[..., 1] = coords[..., 1] * (new_height / old_height)

    if is_bounding_box:
        coords = coords.reshape(-1, 4)

    # 返回标准化的坐标。
    return coords


# 定义生成截取方形盒子的函数，该函数支持自定义层的数量、重叠比率、点的数量和解 xuống因素的参数。
def _generate_crop_boxes(
    image,
    target_size: int,  # 在此处目标尺寸应该是整数还是元组并不是特别清晰, 但是通常此参数代表目标预处理尺寸的大小。
    crop_n_layers: int = 0,
    overlap_ratio: float = 512 / 1500,
    points_per_crop: Optional[int] = 32,
    crop_n_points_downscale_factor: Optional[List[int]] = 1,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    根据指定的参数生成不同的大小的截取框列表，每个层级不同大小的包含 (2^i)^2 个框。
    """
    # 如果输入的图像是列表，则抛出数值错误，仅支持单张图像进行裁剪生成
    if isinstance(image, list):
        raise ValueError("Only one image is allowed for crop generation.")
    
    # 将图像转换为 numpy 数组格式，确保后续处理的统一性
    image = to_numpy_array(image)
    
    # 获取原始图像的尺寸，根据输入数据格式获取
    original_size = get_image_size(image, input_data_format)
    
    # 初始化一个空列表，用于存储各层次的点网格
    points_grid = []
    for i in range(crop_n_layers + 1):
        # 计算每个裁剪区域的采样点数，根据指定的下采样因子进行缩放
        n_points = int(points_per_crop / (crop_n_points_downscale_factor**i))
        # 构建当前层次的点网格并添加到列表中
        points_grid.append(_build_point_grid(n_points))
    
    # 生成裁剪框和层次索引，确定各个裁剪区域在原始图像中的位置和层次
    crop_boxes, layer_idxs = _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size)
    
    # 根据生成的裁剪框和点网格，裁剪原始图像并生成裁剪后的图像以及对应的点网格
    cropped_images, point_grid_per_crop = _generate_crop_images(
        crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format
    )
    
    # 将裁剪框转换为 numpy 数组格式，并将数据类型设置为 float32
    crop_boxes = np.array(crop_boxes)
    crop_boxes = crop_boxes.astype(np.float32)
    
    # 将每个裁剪区域的点网格转换为 numpy 数组格式，并调整维度顺序以匹配后续处理的要求
    points_per_crop = np.array([point_grid_per_crop])
    points_per_crop = np.transpose(points_per_crop, axes=(0, 2, 1, 3))
    
    # 生成输入标签，初始化为与点网格相同大小的全 1 数组，数据类型为 int64
    input_labels = np.ones_like(points_per_crop[:, :, :, 0], dtype=np.int64)
    
    # 返回生成的裁剪框、点网格、裁剪后的图像和对应的输入标签
    return crop_boxes, points_per_crop, cropped_images, input_labels
# 生成每层裁剪框，以XYWH格式表示。XYWH格式包含以下必需索引：
#   - X：边界框左上角的X坐标
#   - Y：边界框左上角的Y坐标
#   - W：边界框的宽度
#   - H：边界框的高度
def _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size):
    crop_boxes, layer_idxs = [], []  # 初始化裁剪框列表和层索引列表
    im_height, im_width = original_size  # 获取原始图像的高度和宽度
    short_side = min(im_height, im_width)  # 计算图像的较短边

    # 原始图像
    crop_boxes.append([0, 0, im_width, im_height])  # 将整个图像作为一个裁剪框添加到列表中
    layer_idxs.append(0)  # 第一层的索引为0

    # 对于每一层裁剪
    for i_layer in range(crop_n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)  # 计算每边的裁剪数量
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))  # 计算重叠区域大小

        # 计算裁剪框的宽度和高度
        crop_width = int(math.ceil((overlap * (n_crops_per_side - 1) + im_width) / n_crops_per_side))
        crop_height = int(math.ceil((overlap * (n_crops_per_side - 1) + im_height) / n_crops_per_side))

        # 计算每个裁剪框的左上角坐标
        crop_box_x0 = [int((crop_width - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_height - overlap) * i) for i in range(n_crops_per_side)]

        # 对每个左上角坐标组合进行裁剪框的生成
        for left, top in product(crop_box_x0, crop_box_y0):
            box = [left, top, min(left + crop_width, im_width), min(top + crop_height, im_height)]
            crop_boxes.append(box)  # 将裁剪框添加到裁剪框列表中
            layer_idxs.append(i_layer + 1)  # 添加相应层索引到层索引列表中

    return crop_boxes, layer_idxs  # 返回裁剪框列表和层索引列表


# 生成裁剪图像
def _generate_crop_images(
    crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format=None
):
    cropped_images = []  # 初始化裁剪后的图像列表
    total_points_per_crop = []  # 初始化每个裁剪中的总点数列表

    # 遍历所有裁剪框
    for i, crop_box in enumerate(crop_boxes):
        left, top, right, bottom = crop_box  # 获取裁剪框的左上角和右下角坐标

        # 推断通道维度格式
        channel_dim = infer_channel_dimension_format(image, input_data_format)
        if channel_dim == ChannelDimension.LAST:
            cropped_im = image[top:bottom, left:right, :]  # 切片裁剪图像（通道在最后）
        else:
            cropped_im = image[:, top:bottom, left:right]  # 切片裁剪图像（通道在最前）

        cropped_images.append(cropped_im)  # 将裁剪后的图像添加到列表中

        cropped_im_size = get_image_size(cropped_im, channel_dim)  # 获取裁剪后图像的大小
        points_scale = np.array(cropped_im_size)[None, ::-1]  # 计算点的比例缩放

        points = points_grid[layer_idxs[i]] * points_scale  # 缩放对应的点
        normalized_points = _normalize_coordinates(target_size, points, original_size)  # 标准化坐标
        total_points_per_crop.append(normalized_points)  # 添加总点数到列表中

    return cropped_images, total_points_per_crop  # 返回裁剪后的图像列表和总点数列表


# 填充掩模
def _pad_masks(masks, crop_box: List[int], orig_height: int, orig_width: int):
    left, top, right, bottom = crop_box  # 获取裁剪框的左上角和右下角坐标
    if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
        return masks  # 如果裁剪框与原始图像大小相同，直接返回掩模

    # 坐标变换掩模
    pad_x, pad_y = orig_width - (right - left), orig_height - (bottom - top)  # 计算填充量
    pad = (left, pad_x - left, top, pad_y - top)  # 构建填充元组
    # 使用 PyTorch 的 nn.functional 模块中的 pad 函数对 masks 进行填充操作
    # 参数 masks：需要填充的张量
    # 参数 pad：填充的大小，可以是单个整数表示每个维度填充相同的量，或者是元组表示每个维度填充的前后数量
    # 参数 value：填充时使用的值，默认为 0
    return torch.nn.functional.pad(masks, pad, value=0)
# 对输入的 masks 进行填充，以适应给定的裁剪框大小
def _pad_masks_tf(masks, crop_box: List[int], orig_height: int, orig_width: int):
    left, top, right, bottom = crop_box
    # 如果裁剪框与原始图像大小一致，则直接返回 masks，无需填充
    if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
        return masks
    # 计算需要填充的宽度和高度
    pad_x, pad_y = orig_width - (right - left), orig_height - (bottom - top)
    # 构建填充参数，格式为(left, right, top, bottom)
    pad = (left, pad_x - left, top, pad_y - top)
    # 使用 TensorFlow 的 pad 函数对 masks 进行填充，填充值为常数0
    return tf.pad(masks, pad, constant_values=0)


# 检查边界框是否接近裁剪边缘，但不接近原始图像边缘
def _is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    # 将裁剪框和原始框转换为 Torch 张量
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)

    left, top, _, _ = crop_box
    # 创建偏移量张量，并将其添加到 boxes 张量中
    offset = torch.tensor([[left, top, left, top]], device=boxes.device)
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    boxes = (boxes + offset).float()

    # 检查 boxes 是否接近裁剪边缘和图像边缘，使用 torch.isclose 函数进行比较
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    # 检查是否有任何接近裁剪边缘的边界框，并返回结果
    return torch.any(near_crop_edge, dim=1)


# 检查边界框是否接近裁剪边缘，但不接近原始图像边缘（使用 TensorFlow）
def _is_box_near_crop_edge_tf(boxes, crop_box, orig_box, atol=20.0):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    # 将裁剪框和原始框转换为 TensorFlow 张量
    crop_box_tf = tf.convert_to_tensor(crop_box, dtype=tf.float32)
    orig_box_tf = tf.convert_to_tensor(orig_box, dtype=tf.float32)

    left, top, _, _ = crop_box
    # 创建偏移量张量，并将其添加到 boxes 张量中
    offset = tf.convert_to_tensor([[left, top, left, top]])
    if len(boxes.shape) == 3:
        offset = tf.expand_dims(offset, 1)
    boxes = tf.cast(boxes + offset, tf.float32)

    # 检查 boxes 是否接近裁剪边缘和图像边缘，使用 tfp.math.isclose 函数进行比较
    near_crop_edge = tfp.math.is_close(boxes, crop_box_tf[None, :], atol=atol, rtol=0)
    near_image_edge = tfp.math.is_close(boxes, orig_box_tf[None, :], atol=atol, rtol=0)
    near_crop_edge = tf.math.logical_and(near_crop_edge, ~near_image_edge)
    # 检查是否有任何接近裁剪边缘的边界框，并返回结果
    return tf.reduce_any(near_crop_edge, axis=1)


# 将批量的 masks 转换为包围框（使用 Torch）
def _batched_mask_to_box(masks: "torch.Tensor"):
    """
    Computes the bounding boxes around the given input masks. The bounding boxes are in the XYXY format which
    corresponds the following required indices:
        - LEFT: left hand side of the bounding box
        - TOP: top of the bounding box
        - RIGHT: right of the bounding box
        - BOTTOM: bottom of the bounding box

    Return [0,0,0,0] for an empty mask. For input shape channel_1 x channel_2 x ... x height x width, the output shape
    is channel_1 x channel_2 x ... x 4.

    Args:
        - masks (`torch.Tensor` of shape `(batch, nb_mask, height, width)`)
    """
    # 如果 masks 张量为空，则返回形状相同的零张量
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    # 将 masks 张量的形状规范化为 Cxheightxwidth 的格式
    shape = masks.shape
    height, width = shape[-2:]
    
    # 获取顶部和底部边界
    in_height, _ = torch.max(masks, dim=-1)
    # 创建高度坐标矩阵
    in_height_coords = in_height * torch.arange(height, device=in_height.device)[None, :]
    # 计算底部边界
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    # 更新高度坐标矩阵，将非边界位置的坐标置为零
    in_height_coords = in_height_coords + height * (~in_height)
    # 计算顶部边界
    top_edges, _ = torch.min(in_height_coords, dim=-1)
    
    # 获取左侧和右侧边界
    in_width, _ = torch.max(masks, dim=-2)
    # 创建宽度坐标矩阵
    in_width_coords = in_width * torch.arange(width, device=in_width.device)[None, :]
    # 计算右侧边界
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    # 更新宽度坐标矩阵，将非边界位置的坐标置为零
    in_width_coords = in_width_coords + width * (~in_width)
    # 计算左侧边界
    left_edges, _ = torch.min(in_width_coords, dim=-1)
    
    # 如果掩码为空，右边界将在左边界左侧。将这些框替换为 [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    # 构建边界框数组
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    # 将空框对应的边界框置为零
    out = out * (~empty_filter).unsqueeze(-1)
    
    # 恢复到原始形状
    out = out.reshape(*shape[:-2], 4)
    # 返回边界框张量
    return out
# 将输入的掩码数据重新排列为Fortran顺序，并展平高度和宽度维度
batch_size, height, width = input_mask.shape

# 计算掩码数据在高度和宽度方向上的变化索引
input_mask = tf.transpose(input_mask, perm=[0, 2, 1])  # 将高度和宽度维度交换位置
input_mask = tf.reshape(input_mask, [batch_size, -1])  # 展平高度和宽度维度

# 计算掩码数据的变化位置
diff = input_mask[:, 1:] ^ input_mask[:, :-1]  # 计算相邻像素之间的不同
change_indices = tf.where(diff)  # 获取变化位置的索引

# 编码成运行长度编码（RLE）格式，符合pycocotools期望的格式
out = []
for i in tf.range(batch_size):
    cur_idxs = tf.boolean_mask(change_indices[:, 1], change_indices[:, 0] == i) + 1
    btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
    counts = [] if input_mask[i, 0] == 0 else [0]  # 如果第一个像素为0，则起始计数为0
    counts += [cur_idxs[0].numpy().item()] + btw_idxs.numpy().tolist() + [height * width - cur_idxs[-1]]
    out.append({"size": [height, width], "counts": counts})

return out
    # 将输入掩码进行转置，然后展开为二维数组
    input_mask = flatten(tf.transpose(input_mask, perm=(0, 2, 1)), 1)

    # 计算变化的索引位置
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    # 找出发生变化的位置的索引
    change_indices = tf.where(diff)

    # 编码运行长度
    out = []
    for i in range(batch_size):
        # 找出当前批次中第 i 行发生变化的索引
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        # 计算变化点之间的距离
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        # 如果第一列的值为 0，counts 列表以空列表开始，否则以 [0] 开始
        counts = [] if input_mask[i, 0] == 0 else [0]
        # 添加第一个变化点前面的零以及变化点之间的距离
        counts += [cur_idxs[0].item()] + btw_idxs.tolist() + [height * width - cur_idxs[-1]]
        # 将结果添加到输出列表中
        out.append({"size": [height, width], "counts": counts})
    # 返回最终结果列表
    return out
# 将非压缩的 RLE（Run-Length Encoding）转换为二进制掩码（mask）
def _rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    # 获取尺寸信息
    height, width = rle["size"]
    # 创建一个空的布尔类型数组，用于存储掩码
    mask = np.empty(height * width, dtype=bool)
    idx = 0
    parity = False
    # 根据 RLE 中的 counts 数组填充掩码数组
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity = not parity
    # 将平铺的掩码数组重新形状化为原始尺寸的掩码
    mask = mask.reshape(width, height)
    return mask.transpose()  # 将掩码转置为原始形状


# 对于 TensorFlow 版本的后处理，执行非极大值抑制（NMS）
def _postprocess_for_mg_tf(rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
    """
    Perform NMS (Non Maximum Suppression) on the outputs.

    Args:
            rle_masks (`tf.Tensor`):
                binary masks in the RLE format
            iou_scores (`tf.Tensor` of shape (nb_masks, 1)):
                iou_scores predicted by the model
            mask_boxes (`tf.Tensor`):
                The bounding boxes corresponding to segmentation masks
            amg_crops_nms_thresh (`float`, *optional*, defaults to 0.7):
                NMS threshold.
    """
    # 使用 TensorFlow 提供的 combined_non_max_suppression 函数执行 NMS
    keep_by_nms = tf.image.combined_non_max_suppression(
        boxes=mask_boxes.float(),
        scores=iou_scores,
        idxs=torch.zeros(mask_boxes.shape[0]),  # 使用零填充，因为在 TensorFlow 中没有对应的 idxs 参数
        iou_threshold=amg_crops_nms_thresh,
    )

    # 根据 NMS 的结果进行筛选
    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    # 将每个 RLE 格式的掩码转换为二进制掩码
    masks = [_rle_to_mask(rle) for rle in rle_masks]

    return masks, iou_scores, rle_masks, mask_boxes
```