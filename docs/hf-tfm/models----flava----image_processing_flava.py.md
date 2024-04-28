# `.\models\flava\image_processing_flava.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
#
# 根据 Apache 许可证 2.0 版本授权
#
# Flava 图像处理类
#
# 导入所需库
import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

# 导入自定义模块和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_vision_available, logging

# 判断视觉库是否可用，并导入必要的模块
if is_vision_available():
    import PIL

# 获取日志记录器
logger = logging.get_logger(__name__)

# 以下变量值来源于 CLIP 模型
FLAVA_IMAGE_MEAN = OPENAI_CLIP_MEAN
FLAVA_IMAGE_STD = OPENAI_CLIP_STD
FLAVA_CODEBOOK_MEAN = [0.0, 0.0, 0.0]
FLAVA_CODEBOOK_STD = [1.0, 1.0, 1.0]
LOGIT_LAPLACE_EPS: float = 0.1

# FlavaMaskingGenerator 类，实现图像遮挡生成
#
# 受 https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py 启发
class FlavaMaskingGenerator:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 14,
        total_mask_patches: int = 75,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_patches: int = 16,
        mask_group_min_aspect_ratio: Optional[float] = 0.3,
        mask_group_max_aspect_ratio: float = None,
    ):
        # 如果 input_size 不是元组，则转换为元组
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        # 获取输入高度和宽度
        self.height, self.width = input_size

        # 计算总路径数
        self.num_patches = self.height * self.width
        self.total_mask_patches = total_mask_patches

        # 设置最小和最大遮挡 patches 数量
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = total_mask_patches if mask_group_max_patches is None else mask_group_max_patches

        # 计算最小和最大遮挡 patches 长宽比的对数值范围
        mask_group_max_aspect_ratio = mask_group_max_aspect_ratio or 1 / mask_group_min_aspect_ratio
        self.log_aspect_ratio = (math.log(mask_group_min_aspect_ratio), math.log(mask_group_max_aspect_ratio))
    def __repr__(self):
        # 返回对象的字符串表示，包括一些属性的信息
        repr_str = "MaskingGenerator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.mask_group_min_patches,
            self.mask_group_max_patches,
            self.total_mask_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str
    
    def get_shape(self):
        # 返回对象的高度和宽度
        return self.height, self.width
    
    def _mask(self, mask, max_mask_patches):
        # 生成遮罩效果，使随机位置被遮罩掉
        delta = 0
        for _attempt in range(10):
            target_area = random.uniform(self.mask_group_min_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)
    
                num_masked = mask[top : top + height, left : left + width].sum()
                # 检查遮罩是否重叠
                if 0 < height * width - num_masked <= max_mask_patches:
                    for i in range(top, top + height):
                        for j in range(left, left + width):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
    
                if delta > 0:
                    break
        return delta
    
    def __call__(self):
        # 创建一个全零数组作为遮罩
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        mask_count = 0
        while mask_count < self.total_mask_patches:
            max_mask_patches = self.total_mask_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.mask_group_max_patches)
    
            # 生成遮罩效果，并且更新计数
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
    
        return mask
# 定义 Flava 图像处理器类，继承自 BaseImageProcessor
class FlavaImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Flava image processor.

    """

    # 定义模型输入的名称为 "pixel_values"
    model_input_names = ["pixel_values"]

    # 初始化函数，接受多个参数
    def __init__(
        self,
        # 是否进行图片调整的标志
        do_resize: bool = True,
        # 图片调整的尺寸
        size: Dict[str, int] = None,
        # 图片调整的重采样方法
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        # 是否进行中心裁剪的标志
        do_center_crop: bool = True,
        # 中心裁剪的尺寸
        crop_size: Dict[str, int] = None,
        # 是否进行重新缩放的标志
        do_rescale: bool = True,
        # 重新缩放的因子
        rescale_factor: Union[int, float] = 1 / 255,
        # 是否进行正则化的标志
        do_normalize: bool = True,
        # 图像均值
        image_mean: Optional[Union[float, Iterable[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, Iterable[float]]] = None
        # Mask 相关参数
        return_image_mask: bool = False,
        input_size_patches: int = 14,
        total_mask_patches: int = 75,
        mask_group_min_patches: int = 16,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_aspect_ratio: float = 0.3,
        mask_group_max_aspect_ratio: Optional[float] = None,
        # Codebook 相关参数
        return_codebook_pixels: bool = False,
        codebook_do_resize: bool = True,
        codebook_size: bool = None,
        codebook_resample: int = PILImageResampling.LANCZOS,
        codebook_do_center_crop: bool = True,
        codebook_crop_size: int = None,
        codebook_do_rescale: bool = True,
        codebook_rescale_factor: Union[int, float] = 1 / 255,
        codebook_do_map_pixels: bool = True,
        codebook_do_normalize: bool = True,
        codebook_image_mean: Optional[Union[float, Iterable[float]]] = None,
        codebook_image_std: Optional[Union[float, Iterable[float]]] = None,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法，传入关键字参数
        super().__init__(**kwargs)
        # 设置图片大小，默认为{"height": 224, "width": 224}
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        # 设置裁剪大小，默认为{"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 设置码本大小，默认为{"height": 112, "width": 112}
        codebook_size = codebook_size if codebook_size is not None else {"height": 112, "width": 112}
        codebook_size = get_size_dict(codebook_size, param_name="codebook_size")
        # 设置码本裁剪大小，默认为{"height": 112, "width": 112}
        codebook_crop_size = codebook_crop_size if codebook_crop_size is not None else {"height": 112, "width": 112}
        codebook_crop_size = get_size_dict(codebook_crop_size, param_name="codebook_crop_size")

        # 设置一系列属性
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else FLAVA_IMAGE_MEAN
        self.image_std = image_std if image_std is not None else FLAVA_IMAGE_STD

        self.return_image_mask = return_image_mask
        self.input_size_patches = input_size_patches
        self.total_mask_patches = total_mask_patches
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = mask_group_max_patches
        self.mask_group_min_aspect_ratio = mask_group_min_aspect_ratio
        self.mask_group_max_aspect_ratio = mask_group_max_aspect_ratio

        self.return_codebook_pixels = return_codebook_pixels
        self.codebook_do_resize = codebook_do_resize
        self.codebook_size = codebook_size
        self.codebook_resample = codebook_resample
        self.codebook_do_center_crop = codebook_do_center_crop
        self.codebook_crop_size = codebook_crop_size
        self.codebook_do_rescale = codebook_do_rescale
        self.codebook_rescale_factor = codebook_rescale_factor
        self.codebook_do_map_pixels = codebook_do_map_pixels
        self.codebook_do_normalize = codebook_do_normalize
        self.codebook_image_mean = codebook_image_mean
        # 设置码本图像均值，默认为FLAVA_IMAGE_MEAN
        self.codebook_image_mean = codebook_image_mean if codebook_image_mean is not None else FLAVA_CODEBOOK_MEAN
        # 设置码本图像标准差，默认为FLAVA_IMAGE_STD
        self.codebook_image_std = codebook_image_std if codebook_image_std is not None else FLAVA_CODEBOOK_STD

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        重写基类的 `from_dict` 方法，确保参数在使用 `from_dict` 和 kwargs 创建图像处理器时更新，
        例如 `FlavaImageProcessor.from_pretrained(checkpoint, codebook_size=600)`
        """
        # 复制传入的图像处理器字典，以免改变原始输入
        image_processor_dict = image_processor_dict.copy()
        # 如果 kwargs 中包含 `codebook_size` 参数，则更新图像处理器字典中的 `codebook_size`
        if "codebook_size" in kwargs:
            image_processor_dict["codebook_size"] = kwargs.pop("codebook_size")
        # 如果 kwargs 中包含 `codebook_crop_size` 参数，则更新图像处理器字典中的 `codebook_crop_size`
        if "codebook_crop_size" in kwargs:
            image_processor_dict["codebook_crop_size"] = kwargs.pop("codebook_crop_size")
        # 调用基类的 from_dict 方法，传入更新后的图像处理器字典和 kwargs
        return super().from_dict(image_processor_dict, **kwargs)

    @lru_cache()
    def masking_generator(
        self,
        input_size_patches,
        total_mask_patches,
        mask_group_min_patches,
        mask_group_max_patches,
        mask_group_min_aspect_ratio,
        mask_group_max_aspect_ratio,
    ) -> FlavaMaskingGenerator:
        # 创建一个 FlavaMaskingGenerator 实例，并返回
        return FlavaMaskingGenerator(
            input_size=input_size_patches,
            total_mask_patches=total_mask_patches,
            mask_group_min_patches=mask_group_min_patches,
            mask_group_max_patches=mask_group_max_patches,
            mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,
            mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,
        )

    # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor.resize 复制，并将 PILImageResampling.BILINEAR->PILImageResampling.BICUBIC
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
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
        # 将输入的 size 字典格式化
        size = get_size_dict(size)
        # 检查 size 字典是否包含 height 和 width 键
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 获取输出的尺寸
        output_size = (size["height"], size["width"])
        # 调用 resize 函数进行图片大小调整，并返回结果
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 映射像素值的方法
    def map_pixels(self, image: np.ndarray) -> np.ndarray:
        # 对图像像素值进行映射
        return (1 - 2 * LOGIT_LAPLACE_EPS) * image + LOGIT_LAPLACE_EPS

    # 图像预处理方法
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
        do_map_pixels: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[ChannelDimension] = None,
    # 定义一个方法，用于预处理单个图像
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses a single image."""
        
        # 如果需要调整大小，并且大小或重采样未指定，则引发数值错误
        if self.do_resize and (self.size is None or self.resample is None):
            raise ValueError("Size and resample must be specified if do_resize is True.")

        # 如果需要重新缩放，并且缩放因子未指定，则引发数值错误
        if self.do_rescale and self.rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        # 如果需要归一化并且图像均值和标准差未指定，则引发数值错误
        if self.do_normalize and (self.image_mean is None or self.image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # 所有变换都需要输入 numpy 数组
        image = to_numpy_array(image)

        # 如果图像已经被缩放且需要重新缩放，发出警告
        if is_scaled_image(image) and self.do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        # 如果输入数据格式未指定，则推断所有图像具有相同的通道维度格式
        if self.input_data_format is None:
            self.input_data_format = infer_channel_dimension_format(image)

        # 如果需要调整大小，则调用 resize 方法
        if self.do_resize:
            image = self.resize(image=image, size=self.size, resample=self.resample, input_data_format=self.input_data_format)

        # 如果需要中心裁剪，则调用 center_crop 方法
        if self.do_center_crop:
            image = self.center_crop(image=image, size=self.crop_size, input_data_format=self.input_data_format)

        # 如果需要重新缩放，则调用 rescale 方法
        if self.do_rescale:
            image = self.rescale(image=image, scale=self.rescale_factor, input_data_format=self.input_data_format)

        # 如果需要归一化，则调用 normalize 方法
        if self.do_normalize:
            image = self.normalize(image=image, mean=self.image_mean, std=self.image_std, input_data_format=self.input_data_format)

        # 如果需要映射像素值，则调用 map_pixels 方法
        if self.do_map_pixels:
            image = self.map_pixels(image)

        # 如果数据格式非空，则调用 to_channel_dimension_format 方法转换数据格式
        if self.data_format is not None:
            image = to_channel_dimension_format(image, self.data_format, input_channel_dim=self.input_data_format)
        
        # 返回预处理后的图像
        return image
    # 预处理函数，对输入的图像进行预处理操作
    def preprocess(
        self,
        # 输入的图像数据
        images: ImageInput,
        # 是否进行调整尺寸
        do_resize: Optional[bool] = None,
        # 调整的尺寸
        size: Dict[str, int] = None,
        # 重采样方法
        resample: PILImageResampling = None,
        # 是否进行中心裁剪
        do_center_crop: Optional[bool] = None,
        # 裁剪尺寸
        crop_size: Optional[Dict[str, int]] = None,
        # 是否进行尺度调整
        do_rescale: Optional[bool] = None,
        # 尺度调整因子
        rescale_factor: Optional[float] = None,
        # 是否进行归一化
        do_normalize: Optional[bool] = None,
        # 图像均值
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, List[float]]] = None,
        # 掩码相关参数
        return_image_mask: Optional[bool] = None,
        # 输入大小的补丁
        input_size_patches: Optional[int] = None,
        # 总的掩码补丁数
        total_mask_patches: Optional[int] = None,
        # 掩码分组的最小补丁数
        mask_group_min_patches: Optional[int] = None,
        # 掩码分组的最大补丁数
        mask_group_max_patches: Optional[int] = None,
        # 掩码分组的最小长宽比
        mask_group_min_aspect_ratio: Optional[float] = None,
        # 掩码分组的最大长宽比
        mask_group_max_aspect_ratio: Optional[float] = None,
        # 代码本相关参数
        return_codebook_pixels: Optional[bool] = None,
        # 是否调整代码本的尺寸
        codebook_do_resize: Optional[bool] = None,
        # 代码本的尺寸
        codebook_size: Optional[Dict[str, int]] = None,
        # 代码本的重采样方法
        codebook_resample: Optional[int] = None,
        # 是否进行代码本的中心裁剪
        codebook_do_center_crop: Optional[bool] = None,
        # 代码本的裁剪尺寸
        codebook_crop_size: Optional[Dict[str, int]] = None,
        # 是否进行代码本的尺度调整
        codebook_do_rescale: Optional[bool] = None,
        # 代码本的尺度调整因子
        codebook_rescale_factor: Optional[float] = None,
        # 是否进行像素映射
        codebook_do_map_pixels: Optional[bool] = None,
        # 是否进行代码本的归一化
        codebook_do_normalize: Optional[bool] = None,
        # 代码本的图像均值
        codebook_image_mean: Optional[Iterable[float]] = None,
        # 代码本的图像标准差
        codebook_image_std: Optional[Iterable[float]] = None,
        # 返回的张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 数据格式的通道维度
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据的格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他关键字参数
        **kwargs,
```