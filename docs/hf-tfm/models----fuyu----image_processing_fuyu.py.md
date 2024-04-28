# `.\models\fuyu\image_processing_fuyu.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 许可协议说明
"""Fuyu的图像处理类Image processor class for Fuyu"""

# 导入模块
import math
from typing import Dict, List, Optional, Union

import numpy as np

# 导入自定义模块
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    pad,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
)
from ...utils import (
    TensorType,
    is_torch_available,
    is_torch_device,
    is_torch_dtype,
    logging,
    requires_backends,
)

# 当torch模块可用时导入torch
if is_torch_available():
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义函数make_list_of_list_of_images，接收一个参数images，返回一个列表的列表
def make_list_of_list_of_images(
    images: Union[List[List[ImageInput]], List[ImageInput], ImageInput],
) -> List[List[ImageInput]]:
    # 如果images是有效的图像，则返回包含该图像的列表的列表
    if is_valid_image(images):
        return [[images]]

    # 如果images是列表，并且列表中的元素都是列表，则直接返回images
    if isinstance(images, list) and all(isinstance(image, list) for image in images):
        return images

    # 如果images是列表，但列表中的元素不是列表，则对列表中的每个元素调用make_list_of_images函数
    if isinstance(images, list):
        return [make_list_of_images(image) for image in images]

    # 如果images不满足以上条件，则抛出数值错误
    raise ValueError("images must be a list of list of images or a list of images or an image.")

# 定义FuyuBatchFeature类，继承自BatchFeature类
class FuyuBatchFeature(BatchFeature):
    """
    BatchFeature class for Fuyu image processor and processor.

    The outputs dictionary from the processors contains a mix of tensors and lists of tensors.
    """
    # 将内部内容转换为张量

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        # 如果未指定张量类型，直接返回当前对象
        if tensor_type is None:
            return self

        # 获取判断元素是否为张量和将元素转换为张量的函数
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type=tensor_type)

        # 将元素转换为张量的函数
        def _convert_tensor(elem):
            if is_tensor(elem):
                return elem
            return as_tensor(elem)

        # 安全地尝试将元素转换为张量
        def _safe_convert_tensor(elem):
            try:
                return _convert_tensor(elem)
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        # 在批处理中进行张量转换
        for key, value in self.items():
            # 如果值是列表且列表的第一个元素也是列表
            if isinstance(value, list) and isinstance(value[0], list):
                # List[List[Any]] -> List[List[Tensor]]
                # 对于嵌套列表，将每个元素转换为张量
                self[key] = [[_safe_convert_tensor(elem) for elem in elems] for elems in value]
            # 如果值是列表
            elif isinstance(value, list):
                # List[Any] -> List[Tensor]
                # 对于列表，将每个元素转换为张量
                self[key] = [_safe_convert_tensor(elem) for elem in value]
            else:
                # Any -> Tensor
                # 对于单个元素，直接转换为张量
                self[key] = _safe_convert_tensor(value)
        # 返回转换后的对象
        return self
    # 将 BatchFeature 实例中的所有值发送到指定设备，使用 PyTorch 的 `to` 方法进行操作
    def to(self, *args, **kwargs) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        # 检查是否需要导入 PyTorch
        requires_backends(self, ["torch"])
        import torch  # noqa

        # 创建一个空字典用于存储转换后的数据
        new_data = {}
        # 获取设备信息
        device = kwargs.get("device")
        # 检查参数是否是设备或数据类型
        if device is None and len(args) > 0:
            # 设备应始终是第一个参数
            arg = args[0]
            if is_torch_dtype(arg):
                # 第一个参数是数据类型
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                # 第一个参数是设备信息
                device = arg
            else:
                # 参数类型不受支持
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")

        # 定义内部函数，用于转换和发送元素
        def _to(elem):
            # 检查是否是浮点数类型
            if torch.is_floating_point(elem):
                # 转换并发送到指定设备
                return elem.to(*args, **kwargs)
            if device is not None:
                # 发送到指定设备
                return elem.to(device=device)

            return elem

        # 仅将浮点数张量转换以避免标记器将 `LongTensor` 转换为 `FloatTensor` 的问题
        for k, v in self.items():
            if isinstance(v, list) and isinstance(v[0], list):
                # 数据结构是列表的列表
                new_v = []
                for elems in v:
                    # 遍历列表中的元素，进行转换和发送
                    new_v.append([_to(elem) for elem in elems])
                new_data[k] = new_v
            elif isinstance(v, list):
                # 数据结构是列表
                new_data[k] = [_to(elem) for elem in v]
            else:
                # 单个元素，进行转换和发送
                new_data[k] = _to(v)
        # 更新 BatchFeature 实例的数据
        self.data = new_data
        # 返回更新后的 BatchFeature 实例
        return self
# FuyuImageProcessor类继承自BaseImageProcessor类，用于处理图像
class FuyuImageProcessor(BaseImageProcessor):
    """
    This class should handle the image processing part before the main FuyuForCausalLM. In particular, it should
    handle:

    - Processing Images:
        Taking a batch of images as input. If the images are variable-sized, it resizes them based on the desired patch
        dimensions. The image output is always img_h, img_w of (1080, 1920)
        对图像进行处理：
        以一批图像为输入。如果图像的大小是可变的，则根据期望的分块尺寸进行调整大小。图像输出的大小始终为img_h，img_w为（1080, 1920）
        
        Then, it patches up these images using the patchify_image function.
        然后，使用patchify_image函数拼接这些图像的块。
        
    - Creating Image Input IDs:
        For each patch, a placeholder ID is given to identify where these patches belong in a token sequence. For
        variable-sized images, each line of patches is terminated with a newline ID.
        创建图像输入ID：
        对于每个分块，我们给他们一个占位ID，以在标记序列中标识这些分块所属的位置。对于可变大小的图像，每行分块以一个换行符ID结束。
        
    - Image Patch Indices:
        For each image patch, the code maintains an index where these patches should be inserted in a token stream.
        图像分块索引：
        对于每个图像分块，代码维护了一个索引，用于确定这些分块应该在标记串中插入的位置。

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image to `size`.
            是否将图像调整为`size`大小。
        size (`Dict[str, int]`, *optional*, defaults to `{"height": 1080, "width": 1920}`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            以`{"height": int, "width": int}`格式的字典指定输出图像的大小。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            调整图像大小时要使用的`PILImageResampling`滤波器，例如`PILImageResampling.BILINEAR`。
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to `size`.
            是否将图像填充为`size`的大小。
        padding_value (`float`, *optional*, defaults to 1.0):
            The value to pad the image with.
            用于填充图像的值。
        padding_mode (`str`, *optional*, defaults to `"constant"`):
            The padding mode to use when padding the image.
            在填充图像时要使用的填充模式。
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
            是否对图像进行归一化。
        image_mean (`float`, *optional*, defaults to 0.5):
            The mean to use when normalizing the image.
            归一化图像时要使用的均值。
        image_std (`float`, *optional*, defaults to 0.5):
            The standard deviation to use when normalizing the image.
            归一化图像时要使用的标准差。
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image.
            是否对图像进行重新缩放。
        rescale_factor (`float`, *optional*, defaults to `1 / 255`):
            The factor to use when rescaling the image.
            重新缩放图像时使用的系数。
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 30, "width": 30}`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
            以`{"height": int, "width": int}`格式的字典指定分块的大小。
    """

    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]
    # 定义模型输入名称的列表
    # 初始化函数，用于创建一个图像处理器对象
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行调整大小的标志，默认为 True
        size: Optional[Dict[str, int]] = None,  # 调整大小的目标尺寸，默认为 None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        do_pad: bool = True,  # 是否进行填充的标志，默认为 True
        padding_value: float = 1.0,  # 填充值，默认为 1.0
        padding_mode: str = "constant",  # 填充模式，默认为常量填充
        do_normalize: bool = True,  # 是否进行归一化的标志，默认为 True
        image_mean: Union[float, List[float]] = 0.5,  # 图像均值，默认为 0.5
        image_std: Union[float, List[float]] = 0.5,  # 图像标准差，默认为 0.5
        do_rescale: bool = True,  # 是否进行重新缩放的标志，默认为 True
        rescale_factor: float = 1 / 255,  # 重新缩放因子，默认为 1/255
        patch_size: Optional[Dict[str, int]] = None,  # 补丁尺寸，默认为 None
        **kwargs,
    ):
        # 调用父类初始化函数
        super().__init__(**kwargs)
        # 设置属性值
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 1080, "width": 1920}  # 若未指定尺寸，则设置默认尺寸
        self.resample = resample  # 设置重采样方法
        self.do_pad = do_pad  # 设置是否进行填充的标志
        self.padding_value = padding_value  # 设置填充值
        self.padding_mode = padding_mode  # 设置填充模式
        self.do_normalize = do_normalize  # 设置是否进行归一化的标志
        self.image_mean = image_mean  # 设置图像均值
        self.image_std = image_std  # 设置图像标准差
        self.do_rescale = do_rescale  # 设置是否进行重新缩放的标志
        self.rescale_factor = rescale_factor  # 设置重新缩放因子
        self.patch_size = patch_size if patch_size is not None else {"height": 30, "width": 30}  # 若未指定补丁尺寸，则设置默认补丁尺寸
    
    # 调整图像大小的方法
    def resize(
        self,
        image: np.ndarray,  # 输入图像的数组表示
        size: Dict[str, int],  # 目标尺寸的字典表示
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为 None
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        # 获取输入图像的高度和宽度
        image_height, image_width = get_image_size(image, input_data_format)
        # 获取目标图像的高度和宽度
        target_height, target_width = size["height"], size["width"]

        # 如果输入图像的宽度和高度均小于或等于目标宽度和高度，则直接返回原图像
        if image_width <= target_width and image_height <= target_height:
            return image

        # 计算高度和宽度的缩放因子
        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        # 选择最小的缩放因子作为最优缩放因子
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        # 计算新的高度和宽度
        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        # 调用resize函数进行图像缩放
        scaled_image = resize(
            image=image,
            size=(new_height, new_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        # 返回缩放后的图像
        return scaled_image

    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        mode: str = "constant",
        constant_values: float = 1.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def pad_image(
        image: np.ndarray,    # 输入图片数据
        size: Dict[str, int],  # 输出图片的高度和宽度
        data_format: Union[ChannelDimension, str] = ChannelDimension.LAST,  # 输出图片的数据格式，默认为输入图片的格式
        input_data_format: Union[ChannelDimension, str] = None  # 输入图片的通道维度格式，如果未指定，将被推断
    ) -> np.ndarray:  # 返回填充后的图片数据
        """
        Pad an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取输入图片的高度和宽度
        image_height, image_width = get_image_size(image, input_data_format)
        # 获取输出图片的目标高度和宽度
        target_height, target_width = size["height"], size["width"]
        # 计算上、左、下、右的填充值
        padding_top = 0
        padding_left = 0
        padding_bottom = target_height - image_height
        padding_right = target_width - image_width
        # 对图片进行填充
        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=mode,  # 填充模式
            constant_values=constant_values,  # 填充的常数值
            data_format=data_format,  # 输出图片的数据格式
            input_data_format=input_data_format  # 输入图片的通道维度格式
        )
        # 返回填充后的图片数据
        return padded_image

    def preprocess(
        self,
        images,  # 输入的图片数据
        do_resize: Optional[bool] = None,  # 是否执行调整大小的标志
        size: Optional[Dict[str, int]] = None,  # 输出图片的尺寸
        resample: Optional[PILImageResampling] = None,  # 重采样方法
        do_pad: Optional[bool] = None,  # 是否执行填充的标志
        padding_value: Optional[float] = None,  # 填充值
        padding_mode: Optional[str] = None,  # 填充模式
        do_normalize: Optional[bool] = None,  # 是否执行归一化的标志
        image_mean: Optional[float] = None,  # 图片的均值
        image_std: Optional[float] = None,  # 图片的标准差
        do_rescale: Optional[bool] = None,  # 是否执行重新缩放的标志
        rescale_factor: Optional[float] = None,  # 重新缩放因子
        patch_size: Optional[Dict[str, int]] = None,  # 图像块的尺寸
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,  # 输出图片的数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入图片的通道维度格式
        return_tensors: Optional[TensorType] = None,  # 返回的张量类型
    def get_num_patches(self, image_height: int, image_width: int, patch_size: Dict[str, int] = None) -> int:
        """
        Calculate number of patches required to encode an image.

        Args:
            image_height (`int`):
                Height of the image.
            image_width (`int`):
                Width of the image.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
        # 如果未提供 patch_size 参数，则使用默认值 self.patch_size
        patch_size = patch_size if patch_size is not None else self.patch_size
        # 提取 patch_size 中的高度和宽度
        patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]

        # 如果图像的高度不能被 patch 的高度整除，则抛出 ValueError 异常
        if image_height % patch_height != 0:
            raise ValueError(f"{image_height=} must be divisible by {patch_height}")
        # 如果图像的宽度不能被 patch 的宽度整除，则抛出 ValueError 异常
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width=} must be divisible by {patch_width}")

        # 计算每个维度上的 patch 数量
        num_patches_per_dim_h = image_height // patch_height
        num_patches_per_dim_w = image_width // patch_width
        # 计算总的 patch 数量
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w
        return num_patches

    def patchify_image(self, image: "torch.Tensor", patch_size: Optional[Dict[str, int]] = None) -> "torch.Tensor":
        """
        Convert an image into a tensor of patches.

        Args:
            image (`torch.Tensor`):
                Image to convert. Shape: [batch, channels, height, width]
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
        # 确保引入了 torch 后端
        requires_backends(self, ["torch"])
        # 如果未提供 patch_size 参数，则使用默认值 self.patch_size
        patch_size = patch_size if patch_size is not None else self.patch_size
        # 提取 patch_size 中的高度和宽度
        patch_height, patch_width = patch_size["height"], patch_size["width"]

        # 根据指定的 patch 大小对图像进行切片并展开，得到图像的 patch 张量
        batch_size, channels, _, _ = image.shape
        unfolded_along_height = image.unfold(2, patch_height, patch_height)
        patches = unfolded_along_height.unfold(3, patch_width, patch_width)
        patches = patches.contiguous()
        patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 4, 1)
        patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)
        return patches

    def preprocess_with_tokenizer_info(
        self,
        image_input: "torch.Tensor",
        image_present: "torch.Tensor",
        image_unpadded_h: "torch.Tensor",
        image_unpadded_w: "torch.Tensor",
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
        patch_size: Optional[Dict[str, int]] = None,
        ):
```