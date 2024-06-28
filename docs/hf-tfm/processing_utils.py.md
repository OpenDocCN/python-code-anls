# `.\processing_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，声明代码的版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证版本 2.0 使用此文件，除非遵守许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据"原样"提供，不附带任何形式的明示或暗示的担保或条件
# 有关详细信息，请参阅许可证
"""
通用处理器的保存/加载类。
"""

import copy  # 导入复制模块
import inspect  # 导入检查模块
import json  # 导入 JSON 模块
import os  # 导入操作系统模块
import warnings  # 导入警告模块
from pathlib import Path  # 导入 Path 类
from typing import Any, Dict, Optional, Tuple, Union  # 导入类型提示

from .dynamic_module_utils import custom_object_save  # 从动态模块工具导入自定义对象保存函数
from .tokenization_utils_base import PreTrainedTokenizerBase  # 从基础标记化工具导入预训练分词器基类
from .utils import (
    PROCESSOR_NAME,  # 从工具模块导入处理器名称常量
    PushToHubMixin,  # 从工具模块导入推送至 Hub 的 Mixin 类
    add_model_info_to_auto_map,  # 从工具模块导入将模型信息添加到自动映射的函数
    cached_file,  # 从工具模块导入缓存文件函数
    copy_func,  # 从工具模块导入复制函数函数
    direct_transformers_import,  # 从工具模块导入直接导入 Transformers 模块的函数
    download_url,  # 从工具模块导入下载 URL 函数
    is_offline_mode,  # 从工具模块导入检查是否为离线模式的函数
    is_remote_url,  # 从工具模块导入检查是否为远程 URL 的函数
    logging,  # 从工具模块导入日志记录对象
)

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# 动态导入 Transformers 模块，以获取处理器类的属性类
transformers_module = direct_transformers_import(Path(__file__).parent)

# 自动映射到基类的映射表，用于自动模型加载时的类关联
AUTO_TO_BASE_CLASS_MAPPING = {
    "AutoTokenizer": "PreTrainedTokenizerBase",  # 自动分词器映射到基础分词器基类
    "AutoFeatureExtractor": "FeatureExtractionMixin",  # 自动特征提取器映射到特征提取混合类
    "AutoImageProcessor": "ImageProcessingMixin",  # 自动图像处理器映射到图像处理混合类
}


class ProcessorMixin(PushToHubMixin):
    """
    这是一个 Mixin 类，用于为所有处理器类提供保存/加载功能。
    """

    attributes = ["feature_extractor", "tokenizer"]  # 处理器类中需要保存的属性列表
    # 对应属性列表中的类属性定义
    feature_extractor_class = None  # 特征提取器类属性初始化为空
    tokenizer_class = None  # 分词器类属性初始化为空
    _auto_class = None  # 自动加载的类属性初始化为空

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        # 对传入的参数和关键字参数进行清理和验证
        for key in kwargs:
            # 检查关键字参数是否在对象的属性列表中，否则引发异常
            if key not in self.attributes:
                raise TypeError(f"Unexpected keyword argument {key}.")
        
        for arg, attribute_name in zip(args, self.attributes):
            # 检查位置参数是否与属性名匹配的关键字参数冲突，如果有冲突则引发异常
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.attributes):
            # 检查最终的关键字参数数量是否与对象属性数量匹配，不匹配则引发数值错误异常
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead."
            )

        # 检查每个参数是否属于其对应的预期类别，这也会捕获用户错误顺序初始化的情况
        for attribute_name, arg in kwargs.items():
            class_name = getattr(self, f"{attribute_name}_class")
            # 如果类名为"AutoXxx"，则检查其对应的基类
            class_name = AUTO_TO_BASE_CLASS_MAPPING.get(class_name, class_name)
            if isinstance(class_name, tuple):
                # 如果类名是元组，则获取模块中对应的类列表
                proper_class = tuple(getattr(transformers_module, n) for n in class_name if n is not None)
            else:
                # 否则直接获取模块中的类
                proper_class = getattr(transformers_module, class_name)

            # 检查参数是否属于预期的类别，不属于则引发数值错误异常
            if not isinstance(arg, proper_class):
                raise ValueError(
                    f"Received a {type(arg).__name__} for argument {attribute_name}, but a {class_name} was expected."
                )

            # 将参数设置为对象的属性
            setattr(self, attribute_name, arg)
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        # Create a deep copy of the instance's __dict__ to prevent unintended modifications
        output = copy.deepcopy(self.__dict__)

        # Retrieve the signature of the __init__ method to get its parameters
        sig = inspect.signature(self.__init__)
        
        # Filter out attributes that are not listed in the __init__ parameters
        attrs_to_save = sig.parameters
        attrs_to_save = [x for x in attrs_to_save if x not in self.__class__.attributes]
        
        # Add "auto_map" to the list of attributes to be saved
        attrs_to_save += ["auto_map"]

        # Filter the output dictionary to include only the attributes to be saved
        output = {k: v for k, v in output.items() if k in attrs_to_save}

        # Add the class name of the processor instance to the output dictionary
        output["processor_class"] = self.__class__.__name__

        # Remove specific attributes that should not be included in the output
        if "tokenizer" in output:
            del output["tokenizer"]
        if "image_processor" in output:
            del output["image_processor"]
        if "feature_extractor" in output:
            del output["feature_extractor"]

        # Filter out attributes with names indicating objects not suitable for serialization
        output = {
            k: v
            for k, v in output.items()
            if not (isinstance(v, PushToHubMixin) or v.__class__.__name__ == "BeamSearchDecoderCTC")
        }

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        # Convert the instance to a dictionary
        dictionary = self.to_dict()

        # Serialize the dictionary to a JSON string with formatting
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        # Open the JSON file for writing
        with open(json_file_path, "w", encoding="utf-8") as writer:
            # Write the instance's JSON representation to the file
            writer.write(self.to_json_string())

    def __repr__(self):
        """
        Returns a string representation of the processor instance.

        Returns:
            `str`: String representation of the processor instance, including key attributes and JSON serialization.
        """
        # Generate representations of all attributes specified in self.attributes
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.attributes]
        
        # Concatenate attribute representations into a single string
        attributes_repr = "\n".join(attributes_repr)
        
        # Return a formatted string including class name, attributes, and JSON serialization
        return f"{self.__class__.__name__}:\n{attributes_repr}\n\n{self.to_json_string()}"

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        """
        Placeholder method for defining how to get processor dictionary.

        This method is not implemented in the provided code snippet.
        """
        pass
    def from_args_and_dict(cls, args, processor_dict: Dict[str, Any], **kwargs):
        """
        从参数字典和额外关键字参数实例化一个 [`~processing_utils.ProcessingMixin`] 类型的对象。

        Args:
            processor_dict (`Dict[str, Any]`):
                用于实例化处理器对象的参数字典。可以利用预训练检查点的
                [`~processing_utils.ProcessingMixin.to_dict`] 方法来获取这样一个字典。
            kwargs (`Dict[str, Any]`):
                初始化处理器对象的额外参数。

        Returns:
            [`~processing_utils.ProcessingMixin`]: 从这些参数实例化的处理器对象。
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # 不像图像处理器或特征提取器那样，处理器的 `__init__` 方法不接受 `kwargs`。
        # 我们必须弹出一些未使用的（但是特定的）参数才能使其正常工作。
        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        if "auto_map" in processor_dict:
            del processor_dict["auto_map"]

        # 使用给定的 `args` 和 `processor_dict` 实例化处理器对象
        processor = cls(*args, **processor_dict)

        # 如果需要，使用 `kwargs` 更新处理器对象
        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        # 记录处理器对象的信息
        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        else:
            return processor

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
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], image processor
        [`~image_processing_utils.ImageProcessingMixin`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        # Check and handle deprecated use_auth_token argument
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        # If token is provided, set it in kwargs
        if token is not None:
            kwargs["token"] = token

        # Get arguments from pretrained model and process kwargs
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Obtain processor dictionary and update kwargs
        processor_dict, kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)

        # Instantiate the class using obtained arguments and processor dictionary
        return cls.from_args_and_dict(args, processor_dict, **kwargs)

    @classmethod
    # 注册一个自动类别名，用于自定义特征提取器，这应仅用于自定义的特征提取器，因为库中的提取器已经与 `AutoProcessor` 映射好了。
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入 transformers.models.auto 模块，用于检查 auto_class 是否存在
        import transformers.models.auto as auto_module

        # 如果 auto_module 中没有找到指定的 auto_class，则抛出 ValueError
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将 auto_class 赋值给当前类的 _auto_class 属性
        cls._auto_class = auto_class

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 初始化一个空列表，用于存储从预训练模型中获取的参数
        args = []
        # 遍历类的 attributes 列表
        for attribute_name in cls.attributes:
            # 获取当前属性对应的类名
            class_name = getattr(cls, f"{attribute_name}_class")

            # 如果 class_name 是一个元组
            if isinstance(class_name, tuple):
                # 从 transformers_module 中获取类，如果为 None 则跳过
                classes = tuple(getattr(transformers_module, n) if n is not None else None for n in class_name)
                # 获取 kwargs 中的 use_fast 参数，默认为 True
                use_fast = kwargs.get("use_fast", True)
                # 如果 use_fast 为 True 并且 classes[1] 不为 None，则使用 classes[1]，否则使用 classes[0]
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                # 如果 class_name 不是元组，则直接从 transformers_module 中获取对应的类
                attribute_class = getattr(transformers_module, class_name)

            # 使用 from_pretrained 方法从预训练模型加载参数，并添加到 args 列表中
            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args

    @property
    def model_input_names(self):
        # 获取当前对象的第一个属性，并尝试获取其 model_input_names 属性，如果不存在则返回 None
        first_attribute = getattr(self, self.attributes[0])
        return getattr(first_attribute, "model_input_names", None)
# 将 ProcessorMixin 类的 push_to_hub 方法复制一份，赋值给原方法
ProcessorMixin.push_to_hub = copy_func(ProcessorMixin.push_to_hub)
# 检查 push_to_hub 方法的文档字符串是否不为空
if ProcessorMixin.push_to_hub.__doc__ is not None:
    # 如果文档字符串不为空，使用格式化字符串将文档字符串中的占位符替换为指定的内容
    ProcessorMixin.push_to_hub.__doc__ = ProcessorMixin.push_to_hub.__doc__.format(
        object="processor", object_class="AutoProcessor", object_files="processor files"
    )
```