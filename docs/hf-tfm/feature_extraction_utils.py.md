# `.\feature_extraction_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可信息
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证要求或书面同意，否则禁止使用此文件
# 可以通过访问 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
#
# 除非适用法律要求或书面同意，本软件是基于“按现状”提供的，不提供任何形式的担保或条件，无论是明示的还是默示的
# 有关许可证的详细信息，请参阅许可证文本

"""
用于常见特征提取器的特征提取保存/加载的类。
"""

import copy  # 导入深复制模块
import json  # 导入 JSON 处理模块
import os  # 导入操作系统模块
import warnings  # 导入警告模块
from collections import UserDict  # 导入用户字典模块
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 库

from .dynamic_module_utils import custom_object_save  # 导入自定义对象保存函数
from .utils import (  # 导入工具函数
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

if TYPE_CHECKING:  # 如果是类型检查阶段
    if is_torch_available():  # 如果 Torch 可用
        import torch  # 导入 Torch 库（用于类型检查）

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

PreTrainedFeatureExtractor = Union["SequenceFeatureExtractor"]  # 预训练特征提取器类型定义

class BatchFeature(UserDict):  # 批次特征类，继承自用户字典
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

    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)  # 调用父类的初始化方法
        self.convert_to_tensors(tensor_type=tensor_type)  # 将数据转换为张量类型

    def __getitem__(self, item: str) -> Union[Any]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_values', 'attention_mask',
        etc.).
        """
        if isinstance(item, str):  # 如果索引是字符串类型
            return self.data[item]  # 返回字典中与键关联的值
        else:
            raise KeyError("Indexing with integers is not available when using Python based feature extractors")  # 抛出索引错误

    def __getattr__(self, item: str):
        try:
            return self.data[item]  # 返回属性对应的数据项
        except KeyError:
            raise AttributeError  # 抛出属性错误

    def __getstate__(self):
        return {"data": self.data}  # 返回对象的状态信息
    # 实现对象状态的反序列化方法，如果状态中包含"data"字段，则将其赋值给当前对象的"data"属性
    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    # 从self.data中获取所有键的方法，模仿transformers.tokenization_utils_base.BatchEncoding.keys的功能
    def keys(self):
        return self.data.keys()

    # 从self.data中获取所有值的方法，模仿transformers.tokenization_utils_base.BatchEncoding.values的功能
    def values(self):
        return self.data.values()

    # 从self.data中获取所有键值对的方法，模仿transformers.tokenization_utils_base.BatchEncoding.items的功能
    def items(self):
        return self.data.items()

    # 根据指定的tensor_type获取对应的转换和判断函数
    def _get_is_as_tensor_fns(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return None, None

        # 将tensor_type转换为TensorType类型
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # 根据tensor_type选择合适的框架，并获取相应的转换和判断函数
        if tensor_type == TensorType.TENSORFLOW:
            # 如果选择的是TensorFlow，则检查TensorFlow是否可用，若不可用则抛出ImportError异常
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            as_tensor = tf.constant  # 定义TensorFlow下的转换函数
            is_tensor = tf.is_tensor  # 定义TensorFlow下的判断函数
        elif tensor_type == TensorType.PYTORCH:
            # 如果选择的是PyTorch，则检查PyTorch是否可用，若不可用则抛出ImportError异常
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch  # noqa

            # 定义PyTorch下的转换函数
            def as_tensor(value):
                if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    value = np.array(value)
                return torch.tensor(value)

            is_tensor = torch.is_tensor  # 定义PyTorch下的判断函数
        elif tensor_type == TensorType.JAX:
            # 如果选择的是JAX，则检查JAX是否可用，若不可用则抛出ImportError异常
            if not is_flax_available():
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # noqa: F811

            as_tensor = jnp.array  # 定义JAX下的转换函数
            is_tensor = is_jax_tensor  # 定义JAX下的判断函数
        else:
            # 如果未知的tensor_type，则使用通用的转换函数
            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # 处理不规则列表
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array  # 定义通用的判断函数

        return is_tensor, as_tensor  # 返回判断函数和转换函数
    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        # 如果 tensor_type 为 None，则直接返回当前对象，不进行任何修改
        if tensor_type is None:
            return self

        # 获取适合转换成指定类型张量的函数
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        # 在批量处理中进行张量转换
        for key, value in self.items():
            try:
                # 如果当前值不是张量，则尝试转换为指定类型的张量
                if not is_tensor(value):
                    tensor = as_tensor(value)

                    # 更新当前键对应的值为转换后的张量
                    self[key] = tensor
            except:  # noqa E722
                # 处理异常情况，特别是针对不同长度的溢出值处理
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        # 返回转换后的对象
        return self
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
        # Ensure that PyTorch backend is available
        requires_backends(self, ["torch"])
        import torch  # noqa
        
        # Initialize a new dictionary for modified data
        new_data = {}
        
        # Retrieve the device from kwargs if available
        device = kwargs.get("device")
        
        # Check if the first argument in args is a device or dtype
        if device is None and len(args) > 0:
            # If device is not specified, the first argument in args is used
            arg = args[0]
            if is_torch_dtype(arg):
                # If the first argument is a PyTorch dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                # If the first argument is a device or a dtype specifier
                device = arg
            else:
                # If the first argument is of an unsupported type
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        
        # Iterate over key-value pairs in the current instance
        for k, v in self.items():
            # Check if the value v is a floating point tensor
            if torch.is_floating_point(v):
                # If v is floating point, cast and send it to the specified device or dtype
                new_data[k] = v.to(*args, **kwargs)
            elif device is not None:
                # If a device is specified, send v to that device
                new_data[k] = v.to(device=device)
            else:
                # Otherwise, retain v as it is
                new_data[k] = v
        
        # Update the data attribute of the instance with the modified data
        self.data = new_data
        
        # Return the modified instance of BatchFeature
        return self
    """
    # 这是一个特征提取的 Mixin 类，用于为顺序数据和图像特征提取器提供保存和加载功能。
    """

    _auto_class = None

    def __init__(self, **kwargs):
        """
        # 初始化方法，将 kwargs 中的元素设置为对象的属性。
        """
        # 弹出 "processor_class" 作为私有属性，用于保存处理器类信息
        self._processor_class = kwargs.pop("processor_class", None)
        # 处理额外的属性，这些属性没有默认值
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """
        # 设置处理器类作为对象的属性。
        """
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
    ):
        """
        # 从预训练模型或路径加载类实例，并配置相关参数。
        """
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
        use_auth_token = kwargs.pop("use_auth_token", None)

        # Handle deprecated `use_auth_token` argument
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            # Raise an error if both `token` and `use_auth_token` are specified
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        # Assert that the provided path is a directory, not a file
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        # Create the directory if it does not exist
        os.makedirs(save_directory, exist_ok=True)

        # If push_to_hub is True, prepare to push the model to the model hub
        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            # Determine the repository ID from the save_directory name
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            # Create or get the repository ID for the model
            repo_id = self._create_repo(repo_id, **kwargs)
            # Get timestamps of files in save_directory for tracking changes
            files_timestamps = self._get_files_timestamps(save_directory)

        # If there's a custom config, save it in the directory
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # Save the feature extractor JSON file in save_directory
        output_feature_extractor_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)
        self.to_json_file(output_feature_extractor_file)
        logger.info(f"Feature extractor saved in {output_feature_extractor_file}")

        # If push_to_hub is True, upload modified files to the model hub
        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        # Return the list containing the path to the saved feature extractor file
        return [output_feature_extractor_file]

    @classmethod
    @classmethod
    # 类方法：从预训练模型名称或路径和其他关键字参数中获取特征提取器字典
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
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # 使用 feature_extractor_dict 字典创建特征提取器对象
        feature_extractor = cls(**feature_extractor_dict)

        # 如果需要，用 kwargs 更新 feature_extractor
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # 记录日志，展示创建的特征提取器对象
        logger.info(f"Feature extractor {feature_extractor}")

        # 如果需要返回未使用的关键字参数，则返回特征提取器对象和未使用的 kwargs
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        # 深拷贝对象的属性到 output 字典
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__

        # 如果存在 "mel_filters" 属性，则从 output 字典中删除
        if "mel_filters" in output:
            del output["mel_filters"]

        # 如果存在 "window" 属性，则从 output 字典中删除
        if "window" in output:
            del output["window"]

        return output

    @classmethod
    # 类方法：从 JSON 文件中实例化特征提取器对象
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
        # 从 JSON 文件中读取参数文本
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()

        # 将 JSON 文本解析为字典形式的特征提取器参数
        feature_extractor_dict = json.loads(text)

        # 使用特征提取器参数字典创建特征提取器对象并返回
        return cls(**feature_extractor_dict)
    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        # 将对象转换为字典表示
        dictionary = self.to_dict()

        # 将所有 numpy 数组转换为 Python 列表
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # 确保私有名称 "_processor_class" 保存为 "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        # 将字典转换为带有缩进和排序的 JSON 字符串，最后加上换行符
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this feature_extractor instance's parameters will be saved.
        """
        # 将对象序列化为 JSON 字符串，写入指定路径的文件中
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        # 返回类的字符串表示形式，包括其 JSON 序列化的内容
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
        # 如果 auto_class 不是字符串，则取其类名作为字符串
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入自动模块，并检查是否存在指定的 auto_class
        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            # 如果找不到对应的 auto_class，则抛出错误
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将 auto_class 设置为类属性 _auto_class
        cls._auto_class = auto_class
# 将 FeatureExtractionMixin 类中的 push_to_hub 方法复制一份，使其成为独立的新函数
FeatureExtractionMixin.push_to_hub = copy_func(FeatureExtractionMixin.push_to_hub)

# 如果 push_to_hub 方法已经有文档字符串（即注释），则对其进行格式化，填充特定的对象信息
if FeatureExtractionMixin.push_to_hub.__doc__ is not None:
    FeatureExtractionMixin.push_to_hub.__doc__ = FeatureExtractionMixin.push_to_hub.__doc__.format(
        object="feature extractor", object_class="AutoFeatureExtractor", object_files="feature extractor file"
    )
```