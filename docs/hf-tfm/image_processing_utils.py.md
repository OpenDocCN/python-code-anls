# `.\image_processing_utils.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy  # 导入深拷贝模块
import json  # 导入 JSON 模块
import os  # 导入操作系统功能模块
import warnings  # 导入警告模块
from io import BytesIO  # 从 io 模块导入 BytesIO 类
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 模块
import requests  # 导入请求模块

from .dynamic_module_utils import custom_object_save  # 从当前包导入动态模块工具中的 custom_object_save 函数
from .feature_extraction_utils import BatchFeature as BaseBatchFeature  # 从当前包导入特征提取工具中的 BatchFeature 类并重命名为 BaseBatchFeature
from .image_transforms import center_crop, normalize, rescale  # 从当前包导入图像转换模块中的三个函数
from .image_utils import ChannelDimension  # 从当前包导入图像工具模块中的 ChannelDimension 类
from .utils import (
    IMAGE_PROCESSOR_NAME,  # 从当前包导入工具模块中的 IMAGE_PROCESSOR_NAME 常量
    PushToHubMixin,  # 从当前包导入工具模块中的 PushToHubMixin 类
    add_model_info_to_auto_map,  # 从当前包导入工具模块中的 add_model_info_to_auto_map 函数
    cached_file,  # 从当前包导入工具模块中的 cached_file 函数
    copy_func,  # 从当前包导入工具模块中的 copy_func 函数
    download_url,  # 从当前包导入工具模块中的 download_url 函数
    is_offline_mode,  # 从当前包导入工具模块中的 is_offline_mode 函数
    is_remote_url,  # 从当前包导入工具模块中的 is_remote_url 函数
    is_vision_available,  # 从当前包导入工具模块中的 is_vision_available 函数
    logging,  # 从当前包导入工具模块中的 logging 模块
)


if is_vision_available():  # 如果视觉功能可用
    from PIL import Image  # 从 PIL 模块导入 Image 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# TODO: Move BatchFeature to be imported by both image_processing_utils and image_processing_utils
# We override the class string here, but logic is the same.
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
class ImageProcessingMixin(PushToHubMixin):
    """
    This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """

    _auto_class = None
    # 初始化方法，用于设置对象的属性
    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # 由于图片处理现在使用 `XXXImageProcessor`，不再使用 `XXXFeatureExtractor`，因此删除此属性
        kwargs.pop("feature_extractor_type", None)
        
        # 将 "processor_class" 弹出并保存为私有属性 `_processor_class`
        self._processor_class = kwargs.pop("processor_class", None)
        
        # 遍历剩余的关键字参数，设置为对象的属性
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                # 如果设置属性失败，则记录错误日志并抛出异常
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    # 设置处理器类的方法，将传入的字符串参数 `processor_class` 设置为 `_processor_class` 属性
    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    # 类方法，用于从预训练模型名或路径创建实例
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
        use_auth_token = kwargs.pop("use_auth_token", None)  # 获取并移除 use_auth_token 参数

        if use_auth_token is not None:  # 如果 use_auth_token 不为 None，发出警告
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:  # 如果同时指定了 token 和 use_auth_token，则抛出错误
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token  # 将 use_auth_token 赋给 token 参数

        if os.path.isfile(save_directory):  # 如果 save_directory 是一个文件路径，则抛出错误
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)  # 创建 save_directory 目录，如果不存在则创建

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)  # 获取并移除 commit_message 参数
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])  # 获取并移除 repo_id 参数，如果不存在则默认为 save_directory 的最后一部分名称
            repo_id = self._create_repo(repo_id, **kwargs)  # 创建或获取指定名称的 repository

            # 获取保存目录中文件的时间戳列表
            files_timestamps = self._get_files_timestamps(save_directory)

        # 如果有自定义配置 (_auto_class 不为 None)，将当前对象以及其配置保存到目录中
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # 将图像处理器对象保存为 JSON 文件
        output_image_processor_file = os.path.join(save_directory, IMAGE_PROCESSOR_NAME)
        self.to_json_file(output_image_processor_file)
        logger.info(f"Image processor saved in {output_image_processor_file}")

        if push_to_hub:
            # 上传修改后的文件到指定的 repository
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        # 返回保存的文件路径列表
        return [output_image_processor_file]

    @classmethod
    @classmethod
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        """
        Creates a dictionary of parameters (`image_processor_dict`) needed to instantiate an image processor.

        Args:
            cls: Class method descriptor.
            pretrained_model_name_or_path (Union[str, os.PathLike]):
                Name or path of the pretrained model for the image processor.
            kwargs (Dict[str, Any]):
                Additional keyword arguments to customize the image processor.

        Returns:
            Dict[str, Any]: Dictionary of parameters (`image_processor_dict`) required to instantiate
                            the image processor.
        """
        image_processor_dict = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            **kwargs
        }
        return image_processor_dict

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates an image processor object from a dictionary of parameters.

        Args:
            image_processor_dict (Dict[str, Any]):
                Dictionary containing parameters to instantiate the image processor.
                Typically obtained from a pretrained checkpoint using `to_dict` method.
            kwargs (Dict[str, Any]):
                Additional parameters to initialize the image processor object.

        Returns:
            ImageProcessingMixin: The instantiated image processor object.
        """
        image_processor_dict = image_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Ensure `size` and `crop_size` are correctly set from kwargs if provided
        if "size" in kwargs and "size" in image_processor_dict:
            image_processor_dict["size"] = kwargs.pop("size")
        if "crop_size" in kwargs and "crop_size" in image_processor_dict:
            image_processor_dict["crop_size"] = kwargs.pop("crop_size")

        # Instantiate the image processor object
        image_processor = cls(**image_processor_dict)

        # Update image_processor attributes with remaining kwargs if applicable
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(image_processor, key):
                setattr(image_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # Log information about the instantiated image processor
        logger.info(f"Image processor {image_processor}")

        # Return the instantiated image processor object with optional unused kwargs
        if return_unused_kwargs:
            return image_processor, kwargs
        else:
            return image_processor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the instance attributes of this image processor to a Python dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all attributes of the image processor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["image_processor_type"] = self.__class__.__name__

        return output
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        从包含参数的 JSON 文件路径实例化一个 `~image_processing_utils.ImageProcessingMixin` 类型的图像处理器。

        Args:
            json_file (`str` or `os.PathLike`):
                包含参数的 JSON 文件路径。

        Returns:
            `~image_processing_utils.ImageProcessingMixin` 类型的图像处理器：从指定 JSON 文件实例化的图像处理器对象。
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        # 将 JSON 文本解析为字典
        image_processor_dict = json.loads(text)
        # 使用解析出的字典参数实例化当前类对象
        return cls(**image_processor_dict)

    def to_json_string(self) -> str:
        """
        将当前实例序列化为 JSON 字符串。

        Returns:
            `str`: 包含当前特征提取器实例所有属性的 JSON 格式字符串。
        """
        # 将当前实例转换为字典形式
        dictionary = self.to_dict()

        # 如果值为 numpy 数组，则转换为列表形式
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # 确保私有名称 "_processor_class" 被正确保存为 "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        # 将字典转换为格式化的 JSON 字符串，并进行排序
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        将当前实例保存到 JSON 文件中。

        Args:
            json_file_path (`str` or `os.PathLike`):
                将保存此图像处理器实例参数的 JSON 文件路径。
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            # 将实例转换为 JSON 字符串并写入文件
            writer.write(self.to_json_string())

    def __repr__(self):
        """
        返回当前实例的字符串表示形式。

        Returns:
            `str`: 包含当前实例 JSON 格式化字符串的类名。
        """
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoImageProcessor"):
        """
        使用给定的自动类注册此类。这仅适用于自定义图像处理器，因为库中的图像处理器已与 `AutoImageProcessor` 映射。

        <Tip warning={true}>
        此 API 是实验性的，可能在未来版本中有些微的破坏性更改。
        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, 默认为 `"AutoImageProcessor"`):
                要将此新图像处理器注册到的自动类。
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入自动模块
        import transformers.models.auto as auto_module

        # 检查是否存在指定的自动类
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} 不是有效的自动类。")

        # 将自动类名称存储在 `_auto_class` 属性中
        cls._auto_class = auto_class
    def fetch_images(self, image_url_or_urls: Union[str, List[str]]):
        """
        Convert a single or a list of URLs into corresponding `PIL.Image` objects.

        If a single URL is passed, the return value will be a single object. If a list is passed, a list of objects is
        returned.
        """
        # 设置 HTTP 请求的头部信息，模拟浏览器行为
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                " Safari/537.36"
            )
        }
        
        # 如果传入的参数是列表，则递归调用 fetch_images 处理列表中的每个 URL
        if isinstance(image_url_or_urls, list):
            return [self.fetch_images(x) for x in image_url_or_urls]
        # 如果传入的参数是字符串，则发送 HTTP 请求获取图片内容，并返回 PIL.Image 对象
        elif isinstance(image_url_or_urls, str):
            # 发送带有自定义头部信息的 HTTP GET 请求
            response = requests.get(image_url_or_urls, stream=True, headers=headers)
            # 如果响应状态码不是 200，则抛出异常
            response.raise_for_status()
            # 将响应内容封装为 PIL.Image 对象并返回
            return Image.open(BytesIO(response.content))
        else:
            # 如果传入的既不是字符串也不是列表，则抛出值错误异常
            raise ValueError(f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}")
class BaseImageProcessor(ImageProcessingMixin):
    # 初始化函数，调用父类的初始化方法
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # 调用对象时调用的方法，用于预处理单张或批量图片
    def __call__(self, images, **kwargs) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, **kwargs)

    # 预处理方法的抽象定义，子类必须实现具体逻辑
    def preprocess(self, images, **kwargs) -> BatchFeature:
        raise NotImplementedError("Each image processor must implement its own preprocess method")

    # 图片按比例缩放的方法
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
        return rescale(image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs)

    # 图片标准化的方法
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
        Normalize an image by subtracting mean and dividing by standard deviation.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `Iterable[float]`):
                Mean value(s) for normalization.
            std (`float` or `Iterable[float]`):
                Standard deviation value(s) for normalization.
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
        return normalize(image, mean=mean, std=std, data_format=data_format, input_data_format=input_data_format, **kwargs)
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `Iterable[float]`):
                Image mean to use for normalization.
            std (`float` or `Iterable[float]`):
                Image standard deviation to use for normalization.
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
        # 调用 `normalize` 函数对图像进行归一化处理，并返回处理后的图像
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
        ):
        """
        Perform center cropping on the image.

        Args:
            image (`np.ndarray`):
                Image to crop.
            size (`Dict[str, int]`):
                Dictionary containing the target size for cropping, with keys 'height' and 'width'.
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
            `np.ndarray`: Cropped image.
        """
        # 执行图像的中心裁剪操作，并返回裁剪后的图像
        # 使用给定的尺寸参数对图像进行中心裁剪
        return center_crop(
            image, size=size, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
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
        # 根据传入的 size 参数，获取确保其为字典格式的大小信息
        size = get_size_dict(size)
        # 检查 size 字典中是否包含 'height' 和 'width' 键，若不包含则引发 ValueError 异常
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        # 调用 center_crop 函数，对输入的 image 进行中心裁剪，并返回裁剪后的图像
        return center_crop(
            image,
            size=(size["height"], size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
# 定义一个元组，包含多个集合，每个集合都是合法的尺寸字典的键集合
VALID_SIZE_DICT_KEYS = ({"height", "width"}, {"shortest_edge"}, {"shortest_edge", "longest_edge"}, {"longest_edge"})


def is_valid_size_dict(size_dict):
    # 判断输入的 size_dict 是否为字典类型，如果不是则返回 False
    if not isinstance(size_dict, dict):
        return False

    # 获取 size_dict 的键集合
    size_dict_keys = set(size_dict.keys())
    # 遍历预定义的合法尺寸字典键集合
    for allowed_keys in VALID_SIZE_DICT_KEYS:
        # 如果 size_dict 的键集合与某个预定义的合法键集合相同，则返回 True
        if size_dict_keys == allowed_keys:
            return True
    # 如果遍历完所有合法键集合后未找到匹配的，则返回 False
    return False


def convert_to_size_dict(
    size, max_size: Optional[int] = None, default_to_square: bool = True, height_width_order: bool = True
):
    # 默认情况下，如果 size 是整数且 default_to_square 为 True，则返回一个表示正方形尺寸的字典
    if isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError("Cannot specify both size as an int, with default_to_square=True and max_size")
        return {"height": size, "width": size}
    
    # 在其他配置下，如果 size 是整数且 default_to_square 为 False，则返回一个表示最短边长度的字典
    elif isinstance(size, int) and not default_to_square:
        size_dict = {"shortest_edge": size}
        if max_size is not None:
            size_dict["longest_edge"] = max_size
        return size_dict
    
    # 如果 size 是元组且 height_width_order 为 True，则返回一个表示高度和宽度的字典
    elif isinstance(size, (tuple, list)) and height_width_order:
        return {"height": size[0], "width": size[1]}
    
    # 如果 size 是元组且 height_width_order 为 False，则返回一个表示高度和宽度的字典（顺序相反）
    elif isinstance(size, (tuple, list)) and not height_width_order:
        return {"height": size[1], "width": size[0]}
    
    # 如果 size 为 None 且 max_size 不为 None，则返回一个表示最长边长度的字典
    elif size is None and max_size is not None:
        if default_to_square:
            raise ValueError("Cannot specify both default_to_square=True and max_size")
        return {"longest_edge": max_size}

    # 如果 size 不满足以上任何条件，则抛出异常
    raise ValueError(f"Could not convert size input to size dict: {size}")


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
    # 调用 convert_to_size_dict 函数，将 size 转换为合适的尺寸字典
    return convert_to_size_dict(size, max_size, default_to_square, height_width_order)
    """
    Casts the `size` parameter into a standardized size dictionary.

    Args:
        size (`Union[int, Iterable[int], Dict[str, int]]`, *optional*):
            The `size` parameter to be cast into a size dictionary.
        max_size (`Optional[int]`, *optional*):
            The `max_size` parameter to be cast into a size dictionary.
        height_width_order (`bool`, *optional*, defaults to `True`):
            If `size` is a tuple, specifies whether it's in (height, width) or (width, height) order.
        default_to_square (`bool`, *optional*, defaults to `True`):
            If `size` is an int, specifies whether to default to a square image or not.
    """
    # 如果 `size` 不是字典类型，则调用函数将其转换为标准化的大小字典
    if not isinstance(size, dict):
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
        # 记录日志，指出参数应该是一个包含指定键集合的字典，如果不是则进行了转换
        logger.info(
            f"{param_name} should be a dictionary on of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size}."
            f" Converted to {size_dict}.",
        )
    else:
        size_dict = size

    # 检查生成的大小字典是否有效，如果不是则抛出 ValueError 异常
    if not is_valid_size_dict(size_dict):
        raise ValueError(
            f"{param_name} must have one of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size_dict.keys()}"
        )
    # 返回标准化后的大小字典
    return size_dict
def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    # 解包原始尺寸
    original_height, original_width = original_size
    # 初始化最佳匹配为None
    best_fit = None
    # 初始化最大有效分辨率为0
    max_effective_resolution = 0
    # 初始化最小浪费分辨率为无穷大
    min_wasted_resolution = float("inf")

    # 遍历可能的分辨率
    for height, width in possible_resolutions:
        # 计算缩放比例
        scale = min(width / original_width, height / original_height)
        # 计算缩小后的宽度和高度
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        # 计算有效分辨率
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        # 计算浪费分辨率
        wasted_resolution = (width * height) - effective_resolution

        # 更新最佳匹配条件：如果有效分辨率大于最大有效分辨率，或者有效分辨率相等且浪费分辨率小于最小浪费分辨率
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    # 返回最佳匹配分辨率
    return best_fit

# 下面是一个稍微不同的注释块
ImageProcessingMixin.push_to_hub = copy_func(ImageProcessingMixin.push_to_hub)
if ImageProcessingMixin.push_to_hub.__doc__ is not None:
    # 格式化文档字符串，替换对象描述中的占位符
    ImageProcessingMixin.push_to_hub.__doc__ = ImageProcessingMixin.push_to_hub.__doc__.format(
        object="image processor", object_class="AutoImageProcessor", object_files="image processor file"
    )
```