# `.\transformers\feature_extraction_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 导入必要的库
import copy
import json
import os
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import numpy as np

# 导入自定义模块
from .dynamic_module_utils import custom_object_save
from .utils import (
    FEATURE_EXTRACTOR_NAME,
    PushToHubMixin,
    TensorType,
    add_model_info_to_auto_map,
    cached_file,
    copy_func,
    download_url,
    is_flax_available,
    is_jax_tensor,
    is_numpy_array,
    is_offline_mode,
    is_remote_url,
    is_tf_available,
    is_torch_available,
    is_torch_device,
    is_torch_dtype,
    logging,
    requires_backends,
)

# 如果是类型检查，导入 torch 库
if TYPE_CHECKING:
    if is_torch_available():
        import torch  # noqa

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 PreTrainedFeatureExtractor 类型别名
PreTrainedFeatureExtractor = Union["SequenceFeatureExtractor"]  # noqa: F821

# 定义 BatchFeature 类，继承自 UserDict
class BatchFeature(UserDict):
    r"""
    Holds the output of the [`~SequenceFeatureExtractor.pad`] and feature extractor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    # 初始化方法
    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        # 将数据转换为张量
        self.convert_to_tensors(tensor_type=tensor_type)

    # 获取字典中的值
    def __getitem__(self, item: str) -> Union[Any]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_values', 'attention_mask',
        etc.).
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError("Indexing with integers is not available when using Python based feature extractors")

    # 获取属性值
    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    # 获取对象状态
    def __getstate__(self):
        return {"data": self.data}
    # 定义对象的 __setstate__ 方法，用于反序列化对象状态
    def __setstate__(self, state):
        # 检查状态中是否包含键为"data"的项
        if "data" in state:
            # 如果包含，将对象的数据属性设置为状态中"data"键对应的值
            self.data = state["data"]

    # 定义 BatchEncoding 对象的 keys 方法，返回数据字典的键
    # 该方法被复制自 transformers.tokenization_utils_base.BatchEncoding.keys
    def keys(self):
        # 返回数据字典的键
        return self.data.keys()

    # 定义 BatchEncoding 对象的 values 方法，返回数据字典的值
    # 该方法被复制自 transformers.tokenization_utils_base.BatchEncoding.values
    def values(self):
        # 返回数据字典的值
        return self.data.values()

    # 定义 BatchEncoding 对象的 items 方法，返回数据字典的键值对
    # 该方法被复制自 transformers.tokenization_utils_base.BatchEncoding.items
    def items(self):
        # 返回数据字典的键值对
        return self.data.items()

    # 定义 BatchEncoding 对象的 _get_is_as_tensor_fns 方法，根据指定的张量类型返回相应的转换函数
    def _get_is_as_tensor_fns(self, tensor_type: Optional[Union[str, TensorType]] = None):
        # 如果未指定张量类型，返回空值
        if tensor_type is None:
            return None, None

        # 将 tensor_type 转换为 TensorType 枚举类型
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # 根据指定的张量类型获取相应的框架函数引用
        if tensor_type == TensorType.TENSORFLOW:
            # 如果 TensorFlow 不可用，引发 ImportError
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            # 导入 TensorFlow 库
            import tensorflow as tf

            # 定义 TensorFlow 下的 as_tensor 和 is_tensor 函数
            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            # 如果 PyTorch 不可用，引发 ImportError
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            # 导入 PyTorch 库
            import torch  # noqa

            # 定义 PyTorch 下的 as_tensor 函数
            def as_tensor(value):
                # 如果值是列表或元组，且第一个元素是 NumPy 数组
                if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    # 转换列表中的 NumPy 数组为 PyTorch 张量
                    value = np.array(value)
                # 返回 PyTorch 张量
                return torch.tensor(value)

            # 定义 PyTorch 下的 is_tensor 函数
            is_tensor = torch.is_tensor
        elif tensor_type == TensorType.JAX:
            # 如果 JAX 不可用，引发 ImportError
            if not is_flax_available():
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            # 导入 JAX 库
            import jax.numpy as jnp  # noqa: F811

            # 定义 JAX 下的 as_tensor 函数
            as_tensor = jnp.array
            # 使用 is_jax_tensor 函数检查是否为 JAX 张量
            is_tensor = is_jax_tensor
        else:
            # 定义默认的 as_tensor 函数
            def as_tensor(value, dtype=None):
                # 如果值是列表、元组，且其中包含列表、元组或 NumPy 数组
                if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                    # 计算值中每个列表、元组或数组的长度
                    value_lens = [len(val) for val in value]
                    # 如果存在长度不同的子列表、子元组或数组，并且未指定数据类型
                    if len(set(value_lens)) > 1 and dtype is None:
                        # 处理不规则列表，将其转换为对象数组
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                # 返回 NumPy 数组表示的值
                return np.asarray(value, dtype=dtype)

            # 使用 is_numpy_array 函数检查是否为 NumPy 数组
            is_tensor = is_numpy_array
        # 返回判断是否为张量和转换为张量的函数
        return is_tensor, as_tensor
    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        # 如果未指定要转换的张量类型，则直接返回原始对象
        if tensor_type is None:
            return self

        # 获取用于判断和转换张量的函数
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        # 在批处理中进行张量转换
        for key, value in self.items():
            try:
                # 如果值不是张量，则进行转换
                if not is_tensor(value):
                    tensor = as_tensor(value)

                    # 更新字典中的值为张量
                    self[key] = tensor
            # 捕获异常，可能是因为无法创建张量或其他错误
            except:  # noqa E722
                # 如果键是"overflowing_values"，则引发特定异常
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                # 否则引发通用异常
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        return self
    # 将批量特征中的所有值发送到设备，调用 `v.to(*args, **kwargs)` 函数（仅适用于 PyTorch）。支持在不同的 `dtype` 中进行类型转换，并将 `BatchFeature` 发送到不同的 `device`。

    def to(self, *args, **kwargs) -> "BatchFeature":
        # 确保需要的后端库可用
        requires_backends(self, ["torch"])
        # 导入 PyTorch 库
        import torch  # noqa

        # 创建新的数据字典
        new_data = {}
        # 获取设备参数
        device = kwargs.get("device")
        # 检查参数是否为设备或数据类型
        if device is None and len(args) > 0:
            # 设备应始终是第一个参数
            arg = args[0]
            if is_torch_dtype(arg):
                # 第一个参数是数据类型
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                # 参数是设备
                device = arg
            else:
                # 参数是其他类型
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # 仅将浮点张量转换，以避免与分词器将 `LongTensor` 转换为 `FloatTensor` 的问题
        for k, v in self.items():
            # 检查 v 是否为浮点数
            if torch.is_floating_point(v):
                # 转换并发送到设备
                new_data[k] = v.to(*args, **kwargs)
            elif device is not None:
                # 发送到指定设备
                new_data[k] = v.to(device=device)
            else:
                # 不进行转换
                new_data[k] = v
        # 更新数据
        self.data = new_data
        # 返回修改后的实例
        return self
class FeatureExtractionMixin(PushToHubMixin):
    """
    This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """

    _auto_class = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # Pop "processor_class" as it should be saved as private attribute
        # 弹出 "processor_class"，因为它应该作为私有属性保存
        self._processor_class = kwargs.pop("processor_class", None)
        # Additional attributes without default values
        # 附加没有默认值的属性
        for key, value in kwargs.items():
            try:
                # 设置实例属性
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        # 将处理器类设置为属性
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
```  
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a feature_extractor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        # Pop the 'use_auth_token' argument from kwargs, if present
        use_auth_token = kwargs.pop("use_auth_token", None)

        # Check if 'use_auth_token' is provided, and issue a warning if it is
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            # Check if 'token' is also provided, raise error if both are present
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            # Set 'token' to the value of 'use_auth_token'
            kwargs["token"] = use_auth_token

        # Check if the provided save_directory is a file, raise AssertionError if it is
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        # Create the save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # If push_to_hub is True, prepare for pushing the model to the Hugging Face model hub
        if push_to_hub:
            # Pop optional arguments for pushing to Hub
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])  # Default repo_id to directory name
            # Create or get the repository ID for the model on Hub
            repo_id = self._create_repo(repo_id, **kwargs)
            # Get timestamps for files in the save directory
            files_timestamps = self._get_files_timestamps(save_directory)

        # If a custom config is provided, save it in the save_directory for loading from Hub
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # Save the feature extractor JSON file in the save_directory
        output_feature_extractor_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)
        self.to_json_file(output_feature_extractor_file)
        logger.info(f"Feature extractor saved in {output_feature_extractor_file}")

        # If push_to_hub is True, upload modified files to the Hub
        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        # Return the path of the saved feature extractor JSON file
        return [output_feature_extractor_file]

    @classmethod
    @classmethod
    # 类方法：用于创建特征提取器字典
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """
        # 是否返回未使用的关键字参数，默认为 False
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # 使用传入的特征提取器字典创建特征提取器对象
        feature_extractor = cls(**feature_extractor_dict)

        # 如果有需要，使用关键字参数更新特征提取器对象
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # 记录特征提取器对象信息
        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        # 深拷贝实例的字典属性
        output = copy.deepcopy(self.__dict__)
        # 添加特征提取器类型到字典中
        output["feature_extractor_type"] = self.__class__.__name__
        # 如果存在'mel_filters'属性，从字典中删除
        if "mel_filters" in output:
            del output["mel_filters"]
        # 如果存在'window'属性，从字典中删除
        if "window" in output:
            del output["window"]
        # 返回序列化后的字典
        return output

    @classmethod
    # 类方法：从 JSON 文件中创建特征提取器对象
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedFeatureExtractor:
        """
        Instantiates a feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] from the path to
        a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature_extractor
            object instantiated from that JSON file.
        """
        # 从 JSON 文件中读取参数
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)
        # 使用参数创建特征提取器对象
        return cls(**feature_extractor_dict)
    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        # 将实例序列化为一个字典
        dictionary = self.to_dict()

        # 将所有 numpy 数组转换为列表
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # 确保私有属性 "_processor_class" 被正确保存为 "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        # 将字典转换为 JSON 字符串，并进行格式化和排序
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this feature_extractor instance's parameters will be saved.
        """
        # 将实例保存到 JSON 文件中
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoFeatureExtractor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoFeatureExtractor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoFeatureExtractor"`):
                The auto class to register this new feature extractor with.
        """
        # 如果 auto_class 不是字符串，则转换为类名字符串
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入自动类模块
        import transformers.models.auto as auto_module

        # 检查 auto_class 是否是有效的自动类
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将新的特征提取器注册到指定的自动类中
        cls._auto_class = auto_class
# 将 FeatureExtractionMixin 类的 push_to_hub 方法复制一份，并赋值给原方法
FeatureExtractionMixin.push_to_hub = copy_func(FeatureExtractionMixin.push_to_hub)
# 如果 push_to_hub 方法有文档字符串
if FeatureExtractionMixin.push_to_hub.__doc__ is not None:
    # 格式化 push_to_hub 方法的文档字符串，替换其中的占位符
    FeatureExtractionMixin.push_to_hub.__doc__ = FeatureExtractionMixin.push_to_hub.__doc__.format(
        object="feature extractor", object_class="AutoFeatureExtractor", object_files="feature extractor file"
    )
```