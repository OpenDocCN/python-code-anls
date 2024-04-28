# `.\transformers\models\sam\image_processing_sam.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 使用文件，禁止未经授权使用
# 在以下链接获取许可证信息：http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，根据许可证分发的软件基于“AS IS”基础，
# 没有任何明示或暗示的担保或条件
# 详细参考许可证中说明的特定语言规定以及限制
"""SAM 的图像处理器类"""
# 导入必要的库文件
import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 导入自定义的图像处理工具，如 BaseImageProcessor 类、BatchFeature、get_size_dict 等
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换函数，如 convert_to_rgb、pad、resize、to_channel_dimension_format
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
# 导入一些图像工具函数，如 IMAGENET_DEFAULT_MEAN、IMAGENET_DEFAULT_STD 等
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
)
# 导入通用工具函数，如 TensorType、is_tf_available、is_torch_available、is_torchvision_available、logging、requires_backends
from ...utils import (
    TensorType,
    is_tf_available,
    is_torch_available,
    is_torchvision_available,
    logging,
    requires_backends,
)

# 如果 Torch 可用，则导入 torch 包和 torch.nn.functional 模块
if is_torch_available():
    import torch
    import torch.nn.functional as F

# 如果 Torchvision 可用，则从 torchvision.ops 导入 batched_nms 函数
if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms

# 如果 TensorFlow 可用，则导入 tensorflow 包和 experimental.numpy 模块
if is_tf_available():
    import tensorflow as tf
    from tensorflow.experimental import numpy as tnp

    # 导入 TF 相关的 flatten、shape_list 等函数
    from ...tf_utils import flatten, shape_list

# 获取日志对象
logger = logging.get_logger(__name__)

class SamImageProcessor(BaseImageProcessor):
    r"""
    构造一个 SAM 图像处理器。
    """

    # 模型的输入名称为 "pixel_values"
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,  # 是否调整大小
        size: Dict[str, int] = None,  # 调整大小的目标尺寸
        mask_size: Dict[str, int] = None,  # 蒙版调整大小的目标尺寸
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法
        do_rescale: bool = True,  # 是否缩放
        rescale_factor: Union[int, float] = 1 / 255,  # 缩放因子
        do_normalize: bool = True,  # 是否归一化
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值
        image_std: Optional[Union[float, List[float]] = None,  # 图像标准差
        do_pad: bool = True,  # 是否填充
        pad_size: int = None,  # 填充尺寸
        mask_pad_size: int = None,  # 蒙版填充尺寸
        do_convert_rgb: bool = True,  # 是否转换为 RGB 格式
        **kwargs,  # 其他参数
    # 定义一个构造函数，传入参数及其注释
    def __init__(self,
                 size: Optional[Union[Dict[str, int], int]] = None,
                 pad_size: Optional[Union[Dict[str, int], int]] = None,
                 mask_size: Optional[Union[Dict[str, int], int]] = None,
                 mask_pad_size: Optional[Union[Dict[str, int], int]] = None,
                 do_resize: bool = False,
                 resample: Union[str, int] = "bilinear",
                 do_rescale: bool = False,
                 rescale_factor: Optional[float] = None,
                 do_normalize: bool = False,
                 image_mean: Optional[Union[float, Tuple[float, float, float]]] = None,
                 image_std: Optional[Union[float, Tuple[float, float, float]]] = None,
                 do_pad: bool = False,
                 do_convert_rgb: bool = False,
                 **kwargs) -> None:
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 设置图片大小参数，默认值为长边1024
        size = size if size is not None else {"longest_edge": 1024}
        # 处理图片大小参数，确保其为字典类型
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size

        # 设置填充大小参数，默认高度和宽度均为1024
        pad_size = pad_size if pad_size is not None else {"height": 1024, "width": 1024}
        # 处理填充大小参数，确保其为字典类型，并默认为正方形
        pad_size = get_size_dict(pad_size, default_to_square=True)

        # 设置蒙版大小参数，默认值为长边256
        mask_size = mask_size if mask_size is not None else {"longest_edge": 256}
        # 处理蒙版大小参数，确保其为字典类型
        mask_size = (
            get_size_dict(max_size=mask_size, default_to_square=False)
            if not isinstance(mask_size, dict)
            else mask_size
        )

        # 设置蒙版填充大小参数，默认高度和宽度均为256
        mask_pad_size = mask_pad_size if mask_pad_size is not None else {"height": 256, "width": 256}
        # 处理蒙版填充大小参数，确保其为字典类型，并默认为正方形
        mask_pad_size = get_size_dict(mask_pad_size, default_to_square=True)

        # 初始化各参数值
        self.do_resize = do_resize
        self.size = size
        self.mask_size = mask_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.mask_pad_size = mask_pad_size
        self.do_convert_rgb = do_convert_rgb

    # 定义图片填充函数，传入参数及其注释
    def pad_image(
            self,
            image: np.ndarray,
            pad_size: Dict[str, int],
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
            **kwargs,
    ) -> np.ndarray:
        """
        在右下角用零填充图像至指定大小`(pad_size["height"], pad_size["width"])`。

        Args:
            image (`np.ndarray`):
                需要填充的图像。
            pad_size (`Dict[str, int]`):
                填充后图像的大小。
            data_format (`str` 或 `ChannelDimension`, *可选*):
                图像的数据格式。可以是"channels_first"或"channels_last"。若为`None`，将使用图像的`data_format`。
            input_data_format (`str` 或 `ChannelDimension`, *可选*):
                输入图像的通道维度格式。如果未提供，将进行推断。
        """
        # 获取填充后图像的高度和宽度
        output_height, output_width = pad_size["height"], pad_size["width"]
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        # 计算需要填充的宽度和高度
        pad_width = output_width - input_width
        pad_height = output_height - input_height

        # 对图像进行填充操作
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
        # 获取旧形状的高度和宽度
        oldh, oldw = old_shape
        # 计算长边的缩放比例
        scale = longest_edge * 1.0 / max(oldh, oldw)
        # 根据缩放比例计算新的高度和宽度
        newh, neww = oldh * scale, oldw * scale
        # 对新的高度和宽度进行四舍五入取整
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        # 返回新的高度和宽度
        return (newh, neww)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
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
        # 将 size 转换为标准的 {"height": int, "width": int} 格式
        size = get_size_dict(size)
        # 检查 size 字典中是否包含 "longest_edge" 键，如果没有则抛出异常
        if "longest_edge" not in size:
            raise ValueError(f"The `size` dictionary must contain the key `longest_edge`. Got {size.keys()}")
        # 获取输入图像的尺寸
        input_size = get_image_size(image, channel_dim=input_data_format)
        # 根据目标长边长度调整输入图像的尺寸
        output_height, output_width = self._get_preprocess_shape(input_size, size["longest_edge"])
        # 返回调整大小后的图像
        return resize(
            image,
            size=(output_height, output_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
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
    # 对图像进行预处理的函数，包括 resizing, rescaling, normalization, padding等等
    # 参数image是输入的图像
    # 参数do_resize表示是否要进行resizing
    # 参数do_rescale表示是否要进行rescaling
    # 参数do_normalize表示是否要进行normalization
    # 参数size是一个可选的字典，用于指定 resizing 的目标尺寸
    # 参数resample是一个可选的PILImageResampling枚举值，用于指定resizing时的采样方法
    # 参数rescale_factor是一个可选的float值，用于指定 rescaling 的比例因子
    # 参数image_mean是一个可选的float或float列表，用于指定 normalization 的均值
    # 参数image_std是一个可选的float或float列表，用于指定 normalization 的标准差
    # 参数do_pad表示是否进行padding
    # 参数pad_size是一个可选的字典，用于指定padding的大小
    # 参数input_data_format是一个可选的字符串或ChannelDimension类，用于指定输入数据的格式

        # 如果需要resizing
        if do_resize:
            # 调用resize函数对输入的图像进行resizing，并将结果赋给image变量
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        # 获取经过resizing之后的图像尺寸
        reshaped_input_size = get_image_size(image, channel_dim=input_data_format)

        # 如果需要rescaling
        if do_rescale:
            # 调用rescale函数对输入的图像进行rescaling，并将结果赋给image变量
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        # 如果需要normalization
        if do_normalize:
            # 调用normalize函数对输入的图像进行normalization，并将结果赋给image变量
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        # 如果需要padding
        if do_pad:
            # 调用pad_image函数对输入的图像进行padding，并将结果赋给image变量
            image = self.pad_image(image=image, pad_size=pad_size, input_data_format=input_data_format)

        # 返回预处理后的图像和经过resizing之后的图像尺寸
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
    ):
    # 对图像进行预处理的函数，包括 resizing, rescaling, normalization, padding等等
    # 参数image是输入的图像
    # 参数do_resize表示是否要进行resizing
    # 参数size是一个可选的字典，用于指定 resizing 的目标尺寸
    # 参数resample是一个可选的PILImageResampling枚举值，用于指定resizing时的采样方法
    # 参数do_rescale表示是否要进行rescaling
    # 参数rescale_factor是一个可选的float值，用于指定 rescaling 的比例因子
    # 参数do_normalize表示是否要进行normalization
    # 参数image_mean是一个可选的float或float列表，用于指定 normalization 的均值
    # 参数image_std是一个可选的float或float列表，用于指定 normalization 的标准差
    # 参数do_pad表示是否进行padding
    # 参数pad_size是一个可选的字典，用于指定padding的大小
    # 参数do_convert_rgb表示是否要将输入图像转换成RGB格式
    # 参数data_format是一个可选的字符串或ChannelDimension类，用于指定输出数据的格式
    # 参数input_data_format是一个可选的字符串或ChannelDimension类，用于指定输入数据的格式

        # 调用_preprocess函数对输入的图像进行预处理，并将结果返回
        return self._preprocess(
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
            input_data_format=input_data_format
        )
    # 定义方法，对输入的图像进行预处理，返回处理后的图像、原始尺寸和重塑后的输入尺寸
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]:
        # 将输入的图像转换为 numpy 数组
        image = to_numpy_array(image)

        # 如果需要将 PIL RGBA 图像转换为 RGB
        if do_convert_rgb:
            image = convert_to_rgb(image)

        # 所有的转换都期望 numpy 数组的输入
        image = to_numpy_array(image)

        # 如果 input_data_format 未指定，根据图像推断通道维度格式
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # 获取原始图像尺寸
        original_size = get_image_size(image, channel_dim=input_data_format)

        # 对图像进行预处理
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

        # 如果 data_format 不为空，则根据要求重新调整通道维度格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回处理后的图像、原始尺寸和重塑后的输入尺寸
        return image, original_size, reshaped_input_size

    # 定义方法，对输入的分割图进行预处理
    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: Optional[bool] = None,
        mask_size: Dict[str, int] = None,
        do_pad: Optional[bool] = None,
        mask_pad_size: Optional[Dict[str, int]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        # 将分割图转换为 NumPy 数组
        segmentation_map = to_numpy_array(segmentation_map)

        # 如果分割图缺少通道维度，则添加通道维度，某些转换需要这个维度
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            # 如果未指定输入数据格式，则根据分割图推断通道维度格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)

        # 获取原始图像尺寸
        original_size = get_image_size(segmentation_map, channel_dim=input_data_format)

        # 预处理分割图像
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

        # 如果为了处理而添加了额外的通道维度，则去除该维度
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        # 将分割图转换为 np.int64 类型
        segmentation_map = segmentation_map.astype(np.int64)

        # 返回预处理后的分割图和原始图像尺寸
        return segmentation_map, original_size

    # 对图像进行预处理
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
    # 后处理分割掩模
    def post_process_masks(
        self,
        masks,
        original_sizes,
        reshaped_input_sizes,
        mask_threshold=0.0,
        binarize=True,
        pad_size=None,
        return_tensors="pt",
    # 该函数用于从预测的掩码中删除填充并将其扩展到原始图像大小
    def post_process_masks(
        self,
        masks,
        original_sizes,
        reshaped_input_sizes,
        mask_threshold=0.0,
        binarize=True,
        pad_size=None,
        return_tensors="pt",
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
        # 根据 return_tensors 参数选择处理掩码的不同实现
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
            raise ValueError("return_tensors must be either 'pt' or 'tf'")
    
    # 该函数用于对 PyTorch 张量进行掩码后处理
    def _post_process_masks_pt(
        self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None
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
        requires_backends(self, ["torch"])
        # 根据是否传入了 pad_size 来设置目标的 padding 尺寸
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size["height"], pad_size["width"])
        # 将 original_sizes 转换为 list 格式
        if isinstance(original_sizes, (torch.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()
        # 将 reshaped_input_sizes 转换为 list 格式
        if isinstance(reshaped_input_sizes, (torch.Tensor, np.ndarray)):
            reshaped_input_sizes = reshaped_input_sizes.tolist()
        output_masks = []
        # 对每个 mask 进行处理
        for i, original_size in enumerate(original_sizes):
            # 判断 masks[i] 类型，如果是 np.ndarray 则转换为 torch.Tensor
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])
            # 如果 masks[i] 类型不是 torch.Tensor 抛出错误
            elif not isinstance(masks[i], torch.Tensor):
                raise ValueError("Input masks should be a list of `torch.tensors` or a list of `np.ndarray`")
            # 对 mask 进行插值，调整到目标 image 尺寸
            interpolated_mask = F.interpolate(masks[i], target_image_size, mode="bilinear", align_corners=False)
            interpolated_mask = interpolated_mask[..., : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1]]
            # 对插值后的 mask 进行进一步插值，调整到原始 image 尺寸
            interpolated_mask = F.interpolate(interpolated_mask, original_size, mode="bilinear", align_corners=False)
            # 如果需要二值化，根据 mask_threshold 进行二值化处理
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)

        return output_masks

    def _post_process_masks_tf(
        self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None
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
        requires_backends(self, ["tf"])
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size["height"], pad_size["width"])

        output_masks = []
        for i, original_size in enumerate(original_sizes):
            # tf.image expects NHWC, we transpose the NCHW inputs for it
            mask = tf.transpose(masks[i], perm=[0, 2, 3, 1])
            interpolated_mask = tf.image.resize(mask, target_image_size, method="bilinear")
            interpolated_mask = interpolated_mask[:, : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1], :]
            interpolated_mask = tf.image.resize(interpolated_mask, original_size, method="bilinear")
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            # And then we transpose them back at the end
            output_masks.append(tf.transpose(interpolated_mask, perm=[0, 3, 1, 2]))

        return output_masks

    def post_process_for_mask_generation(
        self, all_masks, all_scores, all_boxes, crops_nms_thresh, return_tensors="pt"
    ):
        """
        后处理通过调用预测掩码的非最大抑制算法生成的掩码。

        Args:
            all_masks (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                所有预测分割掩码的列表
            all_scores (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                所有预测的 IoU 分数列表
            all_boxes (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                所有预测掩码的边界框列表
            crops_nms_thresh (`float`):
                非最大抑制（NMS）算法的阈值
            return_tensors (`str`, *optional*, defaults to `pt`):
                如果是 `pt`，则返回 `torch.Tensor`。如果是 `tf`，则返回 `tf.Tensor`。
        """
        # 如果 return_tensors 为 "pt"，则调用 _postprocess_for_mg 函数并返回 torch.Tensor
        if return_tensors == "pt":
            return _postprocess_for_mg(all_masks, all_scores, all_boxes, crops_nms_thresh)
        # 如果 return_tensors 为 "tf"，则调用 _postprocess_for_mg_tf 函数并返回 tf.Tensor
        elif return_tensors == "tf":
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
    def _generate_crops(
                # 生成不同大小的裁剪框列表。每个层都有(2**i)**2 个框，i表示层的深度
                image (`np.array`):
                    # 输入原始图像
                target_size (`int`):
                    # 调整后图像的目标尺寸
                crop_n_layers (`int`, *optional*, defaults to 0):
                    # 如果>0，将重新对图像进行裁剪以进行掩膜预测。设置要运行的层数，其中每一层有2**i_layer个图像裁剪。
                overlap_ratio (`float`, *optional*, defaults to 512/1500):
                    # 设置裁剪重叠的程度。在第一层裁剪中，裁剪将重叠这个图像长度的比例。更多裁剪的层将缩小这种重叠。
                points_per_crop (`int`, *optional*, defaults to 32):
                    # 每个裁剪中采样的点数。
                crop_n_points_downscale_factor (`List[int]`, *optional*, defaults to 1):
                    # 在第n层采样的每边点数按照 crop_n_points_downscale_factor**n 缩小。
                device (`torch.device`, *optional*, defaults to None):
                    # 用于计算的设备。如果为None，则使用cpu。
                input_data_format (`str` or `ChannelDimension`, *optional*):
                    # 输入图像的通道维度格式。如果未提供，将会推断。
                return_tensors (`str`, *optional*, defaults to `pt`):
                    # 如果是`pt`，返回`torch.Tensor`。如果是`tf`，返回`tf.Tensor`。
            """
            # 调用_internal函数生成裁剪框、每个裁剪中的点数、裁剪后的图像和输入标签
            crop_boxes, points_per_crop, cropped_images, input_labels = _generate_crop_boxes(
                image,
                target_size,
                crop_n_layers,
                overlap_ratio,
                points_per_crop,
                crop_n_points_downscale_factor,
                input_data_format,
            )
            if return_tensors == "pt":
                if device is None:
                    device = torch.device("cpu")
                # 将裁剪框和每个裁剪中的点数转换为torch.Tensor
                crop_boxes = torch.tensor(crop_boxes, device=device)
                points_per_crop = torch.tensor(points_per_crop, device=device)
                # cropped_images保持为np
                input_labels = torch.tensor(input_labels, device=device)
    
            elif return_tensors == "tf":
                if device is not None:
                    raise ValueError("device is not a supported argument when return_tensors is tf!")
                # 将裁剪框和每个裁剪中的点数转换为tf.Tensor
                crop_boxes = tf.convert_to_tensor(crop_boxes)
                points_per_crop = tf.convert_to_tensor(points_per_crop)
                # cropped_images 保持为np
                input_labels = tf.convert_to_tensor(input_labels)
            else:
                raise ValueError("return_tensors must be either 'pt' or 'tf'.")
            return crop_boxes, points_per_crop, cropped_images, input_labels
    def filter_masks(
        self,
        masks,
        iou_scores,
        original_size,
        cropped_box_image,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
        return_tensors="pt",
    ):
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`Union[torch.Tensor, tf.Tensor]`):
                Input masks.
            iou_scores (`Union[torch.Tensor, tf.Tensor]`):
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
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        # 根据 return_tensors 的不同取值，调用对应的私有方法进行预测掩码过滤
        if return_tensors == "pt":
            # 调用私有方法 _filter_masks_pt 进行预测掩码过滤，返回结果
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
            # 调用私有方法 _filter_masks_tf 进行预测掩码过滤，返回结果
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

    def _filter_masks_pt(
        self,
        masks,
        iou_scores,
        original_size,
        cropped_box_image,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
    ):
        """
        Filters the predicted masks by selecting only the ones that meet several criteria. The first criterion being
        that the IoU scores need to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pads the predicted masks if necessary.

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
        # Check if torch backend is available
        requires_backends(self, ["torch"])
        # Get the original height and width of the image
        original_height, original_width = original_size
        # Flatten the IoU scores and masks tensors
        iou_scores = iou_scores.flatten(0, 1)
        masks = masks.flatten(0, 1)

        # Check if the batch sizes of masks and IoU scores match
        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError("masks and iou_scores must have the same batch size.")

        # Ensure both tensors are on the same device
        if masks.device != iou_scores.device:
            iou_scores = iou_scores.to(masks.device)

        # Get the batch size
        batch_size = masks.shape[0]

        # Initialize a mask to keep track of which masks to keep
        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)

        # Apply thresholding based on predicted IoU scores
        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)

        # Compute stability scores and apply thresholding
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_pt(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)

        # Filter out masks and scores based on the keep mask
        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]

        # Binarize masks
        masks = masks > mask_threshold
        # Convert masks to bounding boxes
        converted_boxes = _batched_mask_to_box(masks)

        # Check if boxes are near the edge of the cropped image and filter out accordingly
        keep_mask = ~_is_box_near_crop_edge(
            converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
        )

        # Update masks, scores, and converted_boxes based on the new keep mask
        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]

        # Pad masks if necessary
        masks = _pad_masks(masks, cropped_box_image, original_height, original_width)
        # Convert masks to run-length encoding for non-maximum suppression
        masks = _mask_to_rle_pytorch(masks)

        return masks, scores, converted_boxes
    # 此函数用于过滤输入的 masks 和 iou_scores，根据预定义的阈值进行处理
    def _filter_masks_tf(
        self,
        # 输入的 masks 列表
        masks,
        # 每个 mask 对应的 iou 得分列表
        iou_scores,
        # 原始图像尺寸
        original_size,
        # 裁剪后图像区域
        cropped_box_image,
        # iou 得分阈值，低于此值的 mask 会被过滤
        pred_iou_thresh=0.88,
        # 稳定性得分阈值，低于此值的 mask 会被过滤
        stability_score_thresh=0.95,
        # 稳定性得分偏移量
        stability_score_offset=1,
        # mask 阈值，低于此值的 mask 会被过滤
        mask_threshold=0,
    ):
    # 根据预测的掩膜来过滤它们，选择满足几个条件的掩膜
    def _filter_predicted_masks(
        masks, # 输入的预测掩膜
        iou_scores, # 每个掩膜的 IoU 分数
        original_size, # 原始图像尺寸
        cropped_box_image, # 裁剪后的图像
        pred_iou_thresh=0.88, # IoU 阈值
        stability_score_thresh=0.95, # 稳定性分数阈值
        mask_threshold=0, # 掩膜阈值
        stability_score_offset=1 # 稳定性分数偏移量
    ):
        # 确保后端是 TensorFlow
        requires_backends(self, ["tf"])
        
        # 获取原始图像的宽高
        original_height, original_width = original_size
        
        # 将输入的掩膜和 IoU 分数展平为批量
        iou_scores = tf.reshape(iou_scores, [iou_scores.shape[0] * iou_scores.shape[1], iou_scores.shape[2:]])
        masks = tf.reshape(masks, [masks.shape[0] * masks.shape[1], masks.shape[2:]])
        
        # 检查掩膜和 IoU 分数的批量大小是否匹配
        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError("masks and iou_scores must have the same batch size.")
        
        # 获取批量大小
        batch_size = masks.shape[0]
        
        # 初始化一个全 True 的保留掩膜
        keep_mask = tf.ones(batch_size, dtype=tf.bool)
        
        # 根据 IoU 分数阈值更新保留掩膜
        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)
        
        # 计算稳定性分数
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_tf(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)
        
        # 根据保留掩膜更新 IoU 分数和掩膜
        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]
        
        # 二值化掩膜
        masks = masks > mask_threshold
        
        # 将掩膜转换为边界框
        converted_boxes = _batched_mask_to_box_tf(masks)
        
        # 移除靠近裁剪边缘的边界框
        keep_mask = ~_is_box_near_crop_edge_tf(
            converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
        )
        
        # 更新 IoU 分数、掩膜和边界框
        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]
        
        # 将掩膜扩展到原始图像尺寸
        masks = _pad_masks_tf(masks, cropped_box_image, original_height, original_width)
        
        # 将掩膜转换为 RLE 格式
        masks = _mask_to_rle_tf(masks)
        
        # 返回过滤后的掩膜、分数和边界框
        return masks, scores, converted_boxes
# 计算稳定性得分（PyTorch版本）
def _compute_stability_score_pt(masks: "torch.Tensor", mask_threshold: float, stability_score_offset: int):
    # 计算每个掩膜与给定阈值的交集元素个数，以节约内存
    intersections = (
        (masks > (mask_threshold + stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    )
    # 计算每个掩膜与给定阈值的并集元素个数
    unions = (masks > (mask_threshold - stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    # 计算稳定性得分
    stability_scores = intersections / unions
    return stability_scores


# 计算稳定性得分（TensorFlow版本）
def _compute_stability_score_tf(masks: "tf.Tensor", mask_threshold: float, stability_score_offset: int):
    # 由于 Torch 进行 Py3 风格的除法而 TF 使用整数进行 floor 除法，所以在 TF 中强制转换为 float32 以获取正确的除法结果
    intersections = tf.count_nonzero(
        masks > (mask_threshold + stability_score_offset), axis=[-1, -2], dtype=tf.float32
    )
    unions = tf.count_nonzero(masks > (mask_threshold - stability_score_offset), axis=[-1, -2], dtype=tf.float32)
    stability_scores = intersections / unions
    return stability_scores


# 生成均匀分布的二维点网格数组
def _build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


# 标准化坐标值
def _normalize_coordinates(
    target_size: int, coords: np.ndarray, original_size: Tuple[int, int], is_bounding_box=False
) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the original image size in (height, width) format.
    """
    old_height, old_width = original_size

    # 计算缩放比例
    scale = target_size * 1.0 / max(old_height, old_width)
    new_height, new_width = old_height * scale, old_width * scale
    new_width = int(new_width + 0.5)
    new_height = int(new_height + 0.5)

    coords = deepcopy(coords).astype(float)

    if is_bounding_box:
        coords = coords.reshape(-1, 2, 2)

    # 标准化坐标值
    coords[..., 0] = coords[..., 0] * (new_width / old_width)
    coords[..., 1] = coords[..., 1] * (new_height / old_height)

    if is_bounding_box:
        coords = coords.reshape(-1, 4)

    return coords


# 生成裁剪框列表
def _generate_crop_boxes(
    image,
    target_size: int,  # Is it tuple here?
    crop_n_layers: int = 0,
    overlap_ratio: float = 512 / 1500,
    points_per_crop: Optional[int] = 32,
    crop_n_points_downscale_factor: Optional[List[int]] = 1,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.
    """
    # 定义一个函数，用于生成给定图像的裁剪
    Args:
        image (Union[numpy.ndarray, PIL.Image, torch.Tensor]):
            要生成裁剪的图像。
        target_size (int):
            最小裁剪尺寸。
        crop_n_layers (int, *optional*):
            如果`crops_n_layers>0`，将再次在图像的裁剪上运行掩码预测。设置要运行的层数，其中每一层都有2**i_layer个图像裁剪。
        overlap_ratio (int, *optional*):
            设置裁剪重叠的程度。在第一层裁剪中，裁剪将重叠这个比例的图像长度。后续具有更多裁剪的图层会缩小此重叠。
        points_per_crop (int, *optional*):
            每个裁剪中要采样的点数。
        crop_n_points_downscale_factor (int, *optional*):
            第n层中采样的每边点数按crop_n_points_downscale_factor**n缩小。
        input_data_format (str or ChannelDimension, *optional*):
            输入图像的通道维度格式。如果未提供，将被推断。
    
    if isinstance(image, list):
        # 如果图像是列表，则抛出值错误，只允许一个图像用于裁剪生成。
        raise ValueError("Only one image is allowed for crop generation.")
    # 将图像转换为 numpy 数组
    image = to_numpy_array(image)
    # 获取原始图像尺寸
    original_size = get_image_size(image, input_data_format)
    
    points_grid = []
    # 循环生成不同层的点网格
    for i in range(crop_n_layers + 1):
        # 计算每个裁剪中要采样的点数
        n_points = int(points_per_crop / (crop_n_points_downscale_factor**i))
        # 构建点网格
        points_grid.append(_build_point_grid(n_points))
    
    # 生成各层裁剪框和层索引
    crop_boxes, layer_idxs = _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size)
    
    # 生成裁剪后的图像及每个裁剪的点网格
    cropped_images, point_grid_per_crop = _generate_crop_images(
        crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format
    )
    # 将裁剪框转换为 numpy 数组并转换为浮点型
    crop_boxes = np.array(crop_boxes)
    crop_boxes = crop_boxes.astype(np.float32)
    # 转置点网格以匹配尺寸
    points_per_crop = np.array([point_grid_per_crop])
    points_per_crop = np.transpose(points_per_crop, axes=(0, 2, 1, 3))
    
    # 创建输入标签，所有值初始化为1，与点网格尺寸相同
    input_labels = np.ones_like(points_per_crop[:, :, :, 0], dtype=np.int64)
    
    # 返回裁剪框、点网格、裁剪后的图像和输入标签
    return crop_boxes, points_per_crop, cropped_images, input_labels
    ```  
# 生成每个层级的裁剪框，根据给定的裁剪层数和重叠比例，裁剪原始尺寸的图像
def _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size):
    # 初始化裁剪框列表和层级索引列表
    crop_boxes, layer_idxs = [], []
    im_height, im_width = original_size
    short_side = min(im_height, im_width)

    # 原始图像的裁剪框
    crop_boxes.append([0, 0, im_width, im_height])
    layer_idxs.append(0)
    for i_layer in range(crop_n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_width = int(math.ceil((overlap * (n_crops_per_side - 1) + im_width) / n_crops_per_side))
        crop_height = int(math.ceil((overlap * (n_crops_per_side - 1) + im_height) / n_crops_per_side))

        crop_box_x0 = [int((crop_width - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_height - overlap) * i) for i in range(n_crops_per_side)]

        for left, top in product(crop_box_x0, crop_box_y0):
            box = [left, top, min(left + crop_width, im_width), min(top + crop_height, im_height)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs

# 生成裁剪后的图像，根据裁剪框和原始图像上的点网格，返回裁剪后的图像列表和各个裁剪图像上的点
def _generate_crop_images(
    crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format=None
):
    cropped_images = []
    total_points_per_crop = []
    for i, crop_box in enumerate(crop_boxes):
        left, top, right, bottom = crop_box

        channel_dim = infer_channel_dimension_format(image, input_data_format)
        if channel_dim == ChannelDimension.LAST:
            cropped_im = image[top:bottom, left:right, :]
        else:
            cropped_im = image[:, top:bottom, left:right]

        cropped_images.append(cropped_im)

        cropped_im_size = get_image_size(cropped_im, channel_dim)
        points_scale = np.array(cropped_im_size)[None, ::-1]

        points = points_grid[layer_idxs[i]] * points_scale
        normalized_points = _normalize_coordinates(target_size, points, original_size)
        total_points_per_crop.append(normalized_points)

    return cropped_images, total_points_per_crop

# 对于不完全覆盖原始图像的裁剪框，使用零填充进行扩展
def _pad_masks(masks, crop_box: List[int], orig_height: int, orig_width: int):
    left, top, right, bottom = crop_box
    if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
        return masks
    # 坐标变换掩码
    pad_x, pad_y = orig_width - (right - left), orig_height - (bottom - top)
    pad = (left, pad_x - left, top, pad_y - top)
    # 使用 PyTorch 中的 nn.functional 模块中的 pad 函数对输入的张量进行填充
    # 参数 masks: 待填充的张量
    # 参数 pad: 指定在每个维度上填充的长度
    # 参数 value: 填充的值，默认为 0
    return torch.nn.functional.pad(masks, pad, value=0)
# 对输入的掩码图像进行适当的填充，使其适应给定的裁剪框
def _pad_masks_tf(masks, crop_box: List[int], orig_height: int, orig_width: int):
    # 获取裁剪框的坐标信息
    left, top, right, bottom = crop_box
    # 如果裁剪框涵盖了整个原始图像，则直接返回输入的掩码图像
    if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
        return masks
    # 计算需要填充的宽度和高度
    pad_x, pad_y = orig_width - (right - left), orig_height - (bottom - top)
    # 根据裁剪框的坐标信息进行填充
    pad = (left, pad_x - left, top, pad_y - top)
    return tf.pad(masks, pad, constant_values=0)


# 过滤靠近裁剪边缘但不靠近原始图像边缘的框
def _is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    """过滤掩码在裁剪边缘但不在原始图像边缘的区域"""
    # 将输入的坐标转换为tensor
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)

    # 计算偏移量
    left, top, _, _ = crop_box
    offset = torch.tensor([[left, top, left, top]], device=boxes.device)
    # 检查boxes是否有通道维度
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    boxes = (boxes + offset).float()

    # 判断boxes是否靠近裁剪边缘但不靠近原始图像边缘
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


# 过滤靠近裁剪边缘但不靠近原始图像边缘的框（TensorFlow版本）
def _is_box_near_crop_edge_tf(boxes, crop_box, orig_box, atol=20.0):
    """过滤掩码在裁剪边缘但不在原始图像边缘的区域"""
    # 将输入的坐标转换为TensorFlow tensor
    crop_box_tf = tf.convert_to_tensor(crop_box, dtype=tf.float32)
    orig_box_tf = tf.convert_to_tensor(orig_box, dtype=tf.float32)

    # 计算偏移量
    left, top, _, _ = crop_box
    offset = tf.convert_to_tensor([[left, top, left, top]])
    # 检查boxes是否有通道维度
    if len(boxes.shape) == 3:
        offset = tf.expand_dims(offset, 1)
    boxes = tf.cast(boxes + offset, tf.float32)

    # 判断boxes是否靠近裁剪边缘但不靠近原始图像边缘
    near_crop_edge = tnp.isclose(boxes, crop_box_tf[None, :], atol=atol, rtol=0)
    near_image_edge = tnp.isclose(boxes, orig_box_tf[None, :], atol=atol, rtol=0)
    near_crop_edge = tf.math.logical_and(near_crop_edge, ~near_image_edge)
    return tf.reduce_any(near_crop_edge, axis=1)


# 计算给定掩码图像的边界框
def _batched_mask_to_box(masks: "torch.Tensor"):
    """
    计算给定输入掩码的边界框。边界框以XYXY格式表示，对应以下索引:
        - LEFT: 边界框的左侧
        - TOP: 边界框的顶部
        - RIGHT: 边界框的右侧
        - BOTTOM: 边界框的底部

    如果掩码为空，返回 [0,0,0,0]。对于输入形状 channel_1 x channel_2 x ... x height x width，输出形状为 channel_1 x channel_2 x ... x 4。

    参数:
        - masks (`torch.Tensor` of shape `(batch, nb_mask, height, width)`)
    """
    # 如果输入为空，直接返回全零张量
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    # 将 masks 的形状标准化为 Cxheightxwidth
    shape = masks.shape
    height, width = shape[-2:]

    # 获取顶部和底部边缘
    # 沿着最后两个维度(dim=-1)找到 masks 中的最大值，即高度信息
    in_height, _ = torch.max(masks, dim=-1)
    # 构建高度坐标矩阵
    in_height_coords = in_height * torch.arange(height, device=in_height.device)[None, :]
    # 获取底部边缘
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    # 根据高度信息和是否存在值来更新高度坐标矩阵
    in_height_coords = in_height_coords + height * (~in_height)
    # 获取顶部边缘
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # 获取左边和右边边缘
    # 沿着倒数第二个维度(dim=-2)找到 masks 中的最大值，即宽度信息
    in_width, _ = torch.max(masks, dim=-2)
    # 构建宽度坐标矩阵
    in_width_coords = in_width * torch.arange(width, device=in_width.device)[None, :]
    # 获取右边边缘
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    # 根据宽度信息和是否存在值来更新宽度坐标矩阵
    in_width_coords = in_width_coords + width * (~in_width)
    # 获取左边边缘
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # 如果掩码为空，则右边边缘将位于左边边缘的左侧。
    # 用 [0, 0, 0, 0] 替换这些框
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    # 将左边、顶部、右边和底部边缘堆叠成一个张量
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    # 将空边框部分置为零
    out = out * (~empty_filter).unsqueeze(-1)

    # 返回到原始形状
    out = out.reshape(*shape[:-2], 4)
    return out
def _batched_mask_to_box_tf(masks: "tf.Tensor"):
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
        - masks (`tf.Tensor` of shape `(batch, nb_mask, height, width)`)
    """

    # Check if the input masks tensor is empty, if so return tensor of zeros
    if tf.size(masks) == 0:
        return tf.zeros([*masks.shape[:-2], 4])

    # Normalize the shape to Cxheightxwidth
    shape = shape_list(masks)
    height, width = shape[-2:]

    # Get top and bottom edges
    in_height = tf.reduce_max(masks, axis=-1)
    in_height_coords = in_height * tf.range(height)[None, :]
    bottom_edges = tf.reduce_max(in_height_coords, axis=-1)
    in_height_coords = in_height_coords + height * (~in_height)
    top_edges = tf.reduce_min(in_height_coords, axis=-1)

    # Get left and right edges
    in_width, _ = tf.reduce_max(masks, axis=-2)
    in_width_coords = in_width * tf.range(width)[None, :]
    right_edges, _ = tf.reduce_max(in_width_coords, axis=-1)
    in_width_coords = in_width_coords + width * (~in_width)
    left_edges, _ = tf.reduce_min(in_width_coords, axis=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = tf.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    out = out * tf.expand_dims(~empty_filter, -1)

    # Return to original shape
    out = tf.reshape(out, *shape[:-2], 4)
    return out


def _mask_to_rle_pytorch(input_mask: "torch.Tensor"):
    """
    Encodes masks into run-length encoding (RLE), in the format expected by pycoco tools.
    """
    # Put in fortran order and flatten height and width
    batch_size, height, width = input_mask.shape
    input_mask = input_mask.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(batch_size):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if input_mask[i, 0] == 0 else [0]
        counts += [cur_idxs[0].item()] + btw_idxs.tolist() + [height * width - cur_idxs[-1]]
        out.append({"size": [height, width], "counts": counts})
    return out


def _mask_to_rle_tf(input_mask: "tf.Tensor"):
    """
    Encodes masks into run-length encoding (RLE), in the format expected by pycoco tools.
    """
    # Put in fortran order and flatten height and width
    batch_size, height, width = input_mask.shape
    # 将输入掩码转置并展平为二维矩阵
    input_mask = flatten(tf.transpose(input_mask, perm=(0, 2, 1)), 1)
    
    # 计算掩码中值变化的索引位置
    # 将输入掩码的相邻两列进行异或操作，得到值变化的位置
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    # 使用 tf.where() 获取值变化位置的索引
    change_indices = tf.where(diff)
    
    # 对变化位置进行行程编码
    out = []
    for i in range(batch_size):
        # 获取当前样本中值变化位置的索引
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        # 计算相邻变化位置之间的长度
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        # 构建行程编码列表
        counts = [] if input_mask[i, 0] == 0 else [0]
        counts += [cur_idxs[0].item()] + btw_idxs.tolist() + [height * width - cur_idxs[-1]]
        # 将当前样本的行程编码信息添加到输出列表
        out.append({"size": [height, width], "counts": counts})
    # 返回所有样本的行程编码信息
    return out
# 将未压缩的 RLE 格式转换为二进制掩码
def _rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    # 从 RLE 中提取高度和宽度
    height, width = rle["size"]
    # 创建一个空的布尔数组，用于存储掩码
    mask = np.empty(height * width, dtype=bool)
    # 初始化索引和奇偶性
    idx = 0
    parity = False
    # 遍历 RLE 中的每个计数值
    for count in rle["counts"]:
        # 将计数范围内的像素值设置为当前奇偶性
        mask[idx : idx + count] = parity
        idx += count
        # 切换奇偶性
        parity = not parity
    # 将一维掩码数组重新整形为二维掩码数组
    mask = mask.reshape(width, height)
    # 返回转置后的掩码，以恢复原始形状
    return mask.transpose()  # Reshape to original shape


# 对模型输出进行 NMS（非最大抑制）
def _postprocess_for_mg(rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
    """
    Perform NMS (Non Maximum Suppression) on the outputs.

    Args:
            rle_masks (`torch.Tensor`):
                binary masks in the RLE format
            iou_scores (`torch.Tensor` of shape (nb_masks, 1)):
                iou_scores predicted by the model
            mask_boxes (`torch.Tensor`):
                The bounding boxes corresponding to segmentation masks
            amg_crops_nms_thresh (`float`, *optional*, defaults to 0.7):
                NMS threshold.
    """
    # 使用批处理的 NMS 算法对输出进行非最大抑制
    keep_by_nms = batched_nms(
        boxes=mask_boxes.float(),
        scores=iou_scores,
        idxs=torch.zeros(mask_boxes.shape[0]),
        iou_threshold=amg_crops_nms_thresh,
    )

    # 根据 NMS 结果过滤输出
    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [_rle_to_mask(rle) for rle in rle_masks]

    return masks, iou_scores, rle_masks, mask_boxes


# 对 TensorFlow 模型输出进行 NMS（非最大抑制）
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
    # 使用 TensorFlow 的组合 NMS 算法对输出进行非最大抑制
    keep_by_nms = tf.image.combined_non_max_suppression(
        boxes=mask_boxes.float(),
        scores=iou_scores,
        idxs=torch.zeros(mask_boxes.shape[0]),
        iou_threshold=amg_crops_nms_thresh,
    )

    # 根据 NMS 结果过滤输出
    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [_rle_to_mask(rle) for rle in rle_masks]

    return masks, iou_scores, rle_masks, mask_boxes
```