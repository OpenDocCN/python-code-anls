# `.\transformers\image_processing_utils.py`

```
# 设置编码格式为 UTF-8

# 版权声明
# Copyright 2022 The HuggingFace Inc. team.
#
# 根据 Apache 许可 2.0 版本使用此文件（“许可”）;
# 除非符合许可，否则您不得使用此文件。
# 您可以在以下网址获取许可的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“按原样”基础分发，
# 没有任何形式的明示或暗示保证或条件。
# 有关特定语言的权限，请参阅许可证。

# 导入模块
import copy  # 导入 copy 模块
import json  # 导入 json 模块
import os  # 导入 os 模块
import warnings  # 导入 warnings 模块
from io import BytesIO  # 从 io 模块中导入 BytesIO 类
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库
import requests  # 导入 requests 模块

# 导入自定义模块
from .dynamic_module_utils import custom_object_save  # 从当前包中导入 custom_object_save 函数
from .feature_extraction_utils import BatchFeature as BaseBatchFeature  # 从当前包中导入 BatchFeature 类并重命名为 BaseBatchFeature
from .image_transforms import center_crop, normalize, rescale  # 从当前包中导入 center_crop、normalize 和 rescale 函数
from .image_utils import ChannelDimension  # 从当前包中导入 ChannelDimension 类
from .utils import (  # 从当前包中导入以下函数和类
    IMAGE_PROCESSOR_NAME,  # 导入 IMAGE_PROCESSOR_NAME 常量
    PushToHubMixin,  # 导入 PushToHubMixin 类
    add_model_info_to_auto_map,  # 导入 add_model_info_to_auto_map 函数
    cached_file,  # 导入 cached_file 函数
    copy_func,  # 导入 copy_func 函数
    download_url,  # 导入 download_url 函数
    is_offline_mode,  # 导入 is_offline_mode 函数
    is_remote_url,  # 导入 is_remote_url 函数
    is_vision_available,  # 导入 is_vision_available 函数
    logging,  # 导入 logging 模块
)

# 如果视觉功能可用，则导入 PIL 库的 Image 模块
if is_vision_available():
    from PIL import Image  # 从 PIL 库中导入 Image 类

# 获取 logger 对象
logger = logging.get_logger(__name__)

# TODO: 移动 BatchFeature 类使其被 image_processing_utils 和 image_processing_utils 导入
# 我们在这里覆盖了类字符串，但逻辑是相同的。
# BatchFeature 类，用于保存图像处理器的特定 '__call__' 方法的输出。
# 该类派生自 Python 字典，并且可以像字典一样使用。
class BatchFeature(BaseBatchFeature):
    r"""
    Holds the output of the image processor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """


# TODO: (Amy) - factor out the common parts of this and the feature extractor
# ImageProcessingMixin 类，用于提供顺序和图像特征提取器的保存/加载功能。
class ImageProcessingMixin(PushToHubMixin):
    """
    This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """

    _auto_class = None  # 设置自动类属性为 None
```  
    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # 从参数 `kwargs` 中设置属性
        # 由于我们现在使用 `XXXImageProcessor` 进行图像处理，因此删除此属性及其值会导致误导
        kwargs.pop("feature_extractor_type", None)
        # 将 "processor_class" 弹出，因为应该将其保存为私有属性
        self._processor_class = kwargs.pop("processor_class", None)
        # 没有默认值的其他属性
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        # 设置处理器类作为属性
        self._processor_class = processor_class

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~image_processing_utils.ImageProcessingMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the image processor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        # Pop out the 'use_auth_token' argument if exists
        use_auth_token = kwargs.pop("use_auth_token", None)

        # If 'use_auth_token' is provided, issue a warning and use 'token' instead
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        # Check if the provided save_directory is a file, raise an error if it is
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        # Create the save_directory if it does not exist
        os.makedirs(save_directory, exist_ok=True)

        # If push_to_hub is True, prepare for pushing the model to the Hub
        if push_to_hub:
            # Pop out optional arguments for pushing to Hub
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1]) # Default repo_id is the last part of save_directory
            # Create or get the repo_id for the model
            repo_id = self._create_repo(repo_id, **kwargs)
            # Get the timestamps of files in the save_directory
            files_timestamps = self._get_files_timestamps(save_directory)

        # If a custom config exists, save it in the save_directory
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # Save the image processor JSON file in the save_directory
        output_image_processor_file = os.path.join(save_directory, IMAGE_PROCESSOR_NAME)
        self.to_json_file(output_image_processor_file)
        logger.info(f"Image processor saved in {output_image_processor_file}")

        # If push_to_hub is True, upload modified files to the Hub
        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        # Return the path of the saved image processor file
        return [output_image_processor_file]

    @classmethod
    @classmethod
    # 类方法：从预训练模型名称或路径实例化图像处理器字典
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        """
        Instantiates a type of [`~image_processing_utils.ImageProcessingMixin`] from a Python dictionary of parameters.

        Args:
            image_processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the image processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~image_processing_utils.ImageProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the image processor object.

        Returns:
            [`~image_processing_utils.ImageProcessingMixin`]: The image processor object instantiated from those
            parameters.
        """
        # 复制图像处理器字典，以防止修改原始参数
        image_processor_dict = image_processor_dict.copy()
        # 弹出 return_unused_kwargs 参数，默认为 False
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # `size` 参数是一个字典，在特征提取器中以前是一个 int 或元组。
        # 在这里，我们直接将 `size` 设置为 `image_processor_dict` 中，以便在图像处理器内转换为适当的字典，
        # 如果 `size` 作为 kwarg 传入，则不会被覆盖。
        if "size" in kwargs and "size" in image_processor_dict:
            image_processor_dict["size"] = kwargs.pop("size")
        if "crop_size" in kwargs and "crop_size" in image_processor_dict:
            image_processor_dict["crop_size"] = kwargs.pop("crop_size")

        # 用给定的参数实例化图像处理器对象
        image_processor = cls(**image_processor_dict)

        # 如果需要，使用 kwargs 更新 image_processor
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(image_processor, key):
                setattr(image_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # 记录图像处理器对象
        logger.info(f"Image processor {image_processor}")
        if return_unused_kwargs:
            return image_processor, kwargs
        else:
            return image_processor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this image processor instance.
        """
        # 深拷贝实例的属性字典
        output = copy.deepcopy(self.__dict__)
        # 将图像处理器类型添加到输出字典中
        output["image_processor_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        从指向参数 JSON 文件的路径实例化一个类型为 [`~image_processing_utils.ImageProcessingMixin`] 的图像处理器。

        Args:
            json_file (`str` or `os.PathLike`):
                包含参数的 JSON 文件的路径。

        Returns:
            一个类型为 [`~image_processing_utils.ImageProcessingMixin`] 的图像处理器对象：从该 JSON 文件实例化的图像处理器对象。
        """
        # 打开 JSON 文件，读取其内容为文本
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        # 将文本解析为 JSON 格式，得到图像处理器字典
        image_processor_dict = json.loads(text)
        # 使用得到的参数字典实例化图像处理器对象并返回
        return cls(**image_processor_dict)

    def to_json_string(self) -> str:
        """
        将此实例序列化为 JSON 字符串。

        Returns:
            `str`: 包含构成此特征提取器实例的所有属性的字符串，格式为 JSON。
        """
        # 将实例转换为字典格式
        dictionary = self.to_dict()

        # 将 numpy 数组转换为列表形式
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # 确保私有名称 "_processor_class" 被正确保存为 "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        # 将字典格式化为 JSON 字符串，并增加缩进和按键排序
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        将此实例保存为 JSON 文件。

        Args:
            json_file_path (`str` or `os.PathLike`):
                要保存此图像处理器实例参数的 JSON 文件的路径。
        """
        # 将实例参数保存为 JSON 字符串，并写入指定路径的文件中
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        # 返回此实例的字符串表示，包含其 JSON 格式参数
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoImageProcessor"):
        """
        将此类注册到指定的自动类中。这仅适用于自定义图像处理器，因为库中的图像处理器已与 `AutoImageProcessor` 映射。

        <Tip warning={true}>

        此 API 是实验性的，并且在下个版本中可能会有一些轻微的更改。

        </Tip>

        Args:
            auto_class (`str` or `type`, *可选*, 默认为 `"AutoImageProcessor "`):
                要将此新图像处理器注册到的自动类。
        """
        # 如果 auto_class 不是字符串，则转换为类名字符串
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入 transformers.models.auto 模块
        import transformers.models.auto as auto_module

        # 检查 auto_class 是否是有效的自动类
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} 不是有效的自动类。")

        # 将自动类名称保存到 _auto_class 属性中
        cls._auto_class = auto_class
    # 定义一个方法用于获取图片，接受一个图片 URL 或 URL 列表作为输入参数，并返回相应的 PIL.Image 对象或对象列表
    def fetch_images(self, image_url_or_urls: Union[str, List[str]]):
        """
        Convert a single or a list of urls into the corresponding `PIL.Image` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        # 定义 HTTP 请求头，模拟浏览器行为
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                " Safari/537.36"
            )
        }
        # 如果传入的是一个 URL 列表，则递归调用 fetch_images 方法对每个 URL 进行获取图片操作
        if isinstance(image_url_or_urls, list):
            return [self.fetch_images(x) for x in image_url_or_urls]
        # 如果传入的是一个单独的 URL 字符串，则发送 HTTP GET 请求获取图片，并返回对应的 PIL.Image 对象
        elif isinstance(image_url_or_urls, str):
            response = requests.get(image_url_or_urls, stream=True, headers=headers)
            # 如果 HTTP 响应状态码不是 200，则抛出异常
            response.raise_for_status()
            # 从 HTTP 响应内容中创建 PIL.Image 对象并返回
            return Image.open(BytesIO(response.content))
        # 如果传入的既不是字符串也不是字符串列表，则抛出 ValueError 异常
        else:
            raise ValueError(f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}")
class BaseImageProcessor(ImageProcessingMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法

    def __call__(self, images, **kwargs) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, **kwargs)  # 调用 preprocess 方法进行图像预处理

    def preprocess(self, images, **kwargs) -> BatchFeature:
        raise NotImplementedError("Each image processor must implement its own preprocess method")  # 抛出未实现异常

    def rescale(
        self,
        image: np.ndarray,
        scale: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            data_format (`str` or `ChannelDimension`, *optional*):
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
            `np.ndarray`: The rescaled image.
        """
        return rescale(image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs)  # 调用 rescale 函数进行图像缩放

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image by subtracting mean and dividing by std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `Iterable[float]`):
                Mean values to subtract from the image.
            std (`float` or `Iterable[float]`):
                Standard deviation values to divide the image by.
            data_format (`str` or `ChannelDimension`, *optional*):
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
            `np.ndarray`: The normalized image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, input_data_format=input_data_format, **kwargs)  # 调用 normalize 函数进行图像归一化
    ) -> np.ndarray:
        """
        标准化图像。image = (image - image_mean) / image_std。

        Args:
            image (`np.ndarray`):
                待标准化的图像。
            mean (`float` or `Iterable[float]`):
                用于标准化的图像均值。
            std (`float` or `Iterable[float]`):
                用于标准化的图像标准差。
            data_format (`str` or `ChannelDimension`, *optional*):
                输出图像的通道维度格式。如果未设置，则使用输入图像的通道维度格式。可以是以下之一：
                - `"channels_first"` or `ChannelDimension.FIRST`: 图像以 (num_channels, height, width) 格式。
                - `"channels_last"` or `ChannelDimension.LAST`: 图像以 (height, width, num_channels) 格式。
            input_data_format (`ChannelDimension` or `str`, *optional*):
                输入图像的通道维度格式。如果未设置，则从输入图像推断通道维度格式。可以是以下之一：
                - `"channels_first"` or `ChannelDimension.FIRST`: 图像以 (num_channels, height, width) 格式。
                - `"channels_last"` or `ChannelDimension.LAST`: 图像以 (height, width, num_channels) 格式。

        Returns:
            `np.ndarray`: 标准化后的图像。
        """
        return normalize(
            image, mean=mean, std=std, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

    def center_crop(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Dict[str, int]`):
                Size of the output image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        # 将 size 转换为字典格式
        size = get_size_dict(size)
        # 检查 size 字典中是否包含 'height' 和 'width' 键，如果没有则引发 ValueError 异常
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        # 返回中心裁剪后的图像
        return center_crop(
            image,
            size=(size["height"], size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
# 定义有效的尺寸字典的键的组合
VALID_SIZE_DICT_KEYS = ({"height", "width"}, {"shortest_edge"}, {"shortest_edge", "longest_edge"}, {"longest_edge"})


# 检查给定的尺寸字典是否有效
def is_valid_size_dict(size_dict):
    # 如果不是字典类型，则返回 False
    if not isinstance(size_dict, dict):
        return False

    # 获取尺寸字典的键集合
    size_dict_keys = set(size_dict.keys())
    # 遍历有效的尺寸字典键组合，如果匹配则返回 True
    for allowed_keys in VALID_SIZE_DICT_KEYS:
        if size_dict_keys == allowed_keys:
            return True
    # 如果没有匹配的键组合，则返回 False
    return False


# 将尺寸转换为尺寸字典
def convert_to_size_dict(
    size, max_size: Optional[int] = None, default_to_square: bool = True, height_width_order: bool = True
):
    # 默认情况下，如果尺寸是整数，则假定表示 (size, size) 的元组
    if isinstance(size, int) and default_to_square:
        # 如果同时指定了整数尺寸和 default_to_square=True 以及 max_size，则引发 ValueError
        if max_size is not None:
            raise ValueError("Cannot specify both size as an int, with default_to_square=True and max_size")
        return {"height": size, "width": size}
    # 在其他配置中，如果尺寸是整数且 default_to_square 为 False，则尺寸表示调整大小后最短边的长度
    elif isinstance(size, int) and not default_to_square:
        size_dict = {"shortest_edge": size}
        if max_size is not None:
            size_dict["longest_edge"] = max_size
        return size_dict
    # 否则，如果尺寸是元组，则可能是 (height, width) 或 (width, height)
    elif isinstance(size, (tuple, list)) and height_width_order:
        return {"height": size[0], "width": size[1]}
    elif isinstance(size, (tuple, list)) and not height_width_order:
        return {"height": size[1], "width": size[0]}
    # 如果尺寸为 None 且 max_size 不为 None，则根据 default_to_square 返回相应的尺寸字典
    elif size is None and max_size is not None:
        if default_to_square:
            raise ValueError("Cannot specify both default_to_square=True and max_size")
        return {"longest_edge": max_size}

    # 如果无法将尺寸输入转换为尺寸字典，则引发 ValueError
    raise ValueError(f"Could not convert size input to size dict: {size}")


# 获取尺寸字典
def get_size_dict(
    size: Union[int, Iterable[int], Dict[str, int]] = None,
    max_size: Optional[int] = None,
    height_width_order: bool = True,
    default_to_square: bool = True,
    param_name="size",
) -> dict:
    """
    Converts the old size parameter in the config into the new dict expected in the config. This is to ensure backwards
    compatibility with the old image processor configs and removes ambiguity over whether the tuple is in (height,
    width) or (width, height) format.

    - If `size` is tuple, it is converted to `{"height": size[0], "width": size[1]}` or `{"height": size[1], "width":
    size[0]}` if `height_width_order` is `False`.
    - If `size` is an int, and `default_to_square` is `True`, it is converted to `{"height": size, "width": size}`.
    - If `size` is an int and `default_to_square` is False, it is converted to `{"shortest_edge": size}`. If `max_size`
      is set, it is added to the dict as `{"longest_edge": max_size}`.
    """
    Args:
        size (`Union[int, Iterable[int], Dict[str, int]]`, *optional*):
            The `size` parameter to be cast into a size dictionary.
        max_size (`Optional[int]`, *optional*):
            The `max_size` parameter to be cast into a size dictionary.
        height_width_order (`bool`, *optional*, defaults to `True`):
            If `size` is a tuple, whether it's in (height, width) or (width, height) order.
        default_to_square (`bool`, *optional*, defaults to `True`):
            If `size` is an int, whether to default to a square image or not.
    """
    # 如果 size 不是字典类型，则将其转换为字典类型
    if not isinstance(size, dict):
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
        # 记录日志，提示参数应为字典类型，给出转换后的字典
        logger.info(
            f"{param_name} should be a dictionary on of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size}."
            f" Converted to {size_dict}.",
        )
    else:
        size_dict = size

    # 检查 size_dict 是否为有效的尺寸字典，如果不是则抛出 ValueError 异常
    if not is_valid_size_dict(size_dict):
        raise ValueError(
            f"{param_name} must have one of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size_dict.keys()}"
        )
    # 返回处理后的 size_dict
    return size_dict
# 将 ImageProcessingMixin 类的 push_to_hub 方法复制一份
ImageProcessingMixin.push_to_hub = copy_func(ImageProcessingMixin.push_to_hub)
# 如果 push_to_hub 方法有文档字符串
if ImageProcessingMixin.push_to_hub.__doc__ is not None:
    # 使用格式化字符串将文档字符串中的占位符替换为相应的内容
    ImageProcessingMixin.push_to_hub.__doc__ = ImageProcessingMixin.push_to_hub.__doc__.format(
        object="image processor", object_class="AutoImageProcessor", object_files="image processor file"
    )
```