# `.\transformers\processing_utils.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""
用于常见处理器的处理保存/加载类。
"""

import copy
import inspect
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# 导入自定义模块工具函数
from .dynamic_module_utils import custom_object_save
# 导入基础分词器类
from .tokenization_utils_base import PreTrainedTokenizerBase
# 导入工具函数
from .utils import (
    PROCESSOR_NAME,
    PushToHubMixin,
    add_model_info_to_auto_map,
    cached_file,
    copy_func,
    direct_transformers_import,
    download_url,
    is_offline_mode,
    is_remote_url,
    logging,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 动态导入 Transformers 模块，以获取处理器的属性类
transformers_module = direct_transformers_import(Path(__file__).parent)

# 自动映射基类
AUTO_TO_BASE_CLASS_MAPPING = {
    "AutoTokenizer": "PreTrainedTokenizerBase",
    "AutoFeatureExtractor": "FeatureExtractionMixin",
    "AutoImageProcessor": "ImageProcessingMixin",
}

# 处理器混合类，用于提供所有处理器类的保存/加载功能
class ProcessorMixin(PushToHubMixin):
    """
    This is a mixin used to provide saving/loading functionality for all processor classes.
    """

    # 需要保存/加载的属性列表
    attributes = ["feature_extractor", "tokenizer"]
    # 属性类的名称，需要与 attributes 属性中的属性名称对应
    feature_extractor_class = None
    tokenizer_class = None
    _auto_class = None

    # 参数必须与 attributes 类属性匹配
    # 初始化函数，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 清理参数列表，确保关键字参数在预期范围内
        for key in kwargs:
            # 检查是否存在不在属性列表中的关键字参数，若存在则引发 TypeError 异常
            if key not in self.attributes:
                raise TypeError(f"Unexpected keyword argument {key}.")
        # 检查位置参数与关键字参数的一致性
        for arg, attribute_name in zip(args, self.attributes):
            # 检查是否有重复的参数，若有则引发 TypeError 异常
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                # 将位置参数映射到关键字参数中
                kwargs[attribute_name] = arg

        # 检查参数数量是否正确
        if len(kwargs) != len(self.attributes):
            # 若参数数量不正确，则引发 ValueError 异常
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead."
            )

        # 检查每个参数是否是正确的类的实例（也会捕获用户错误的初始化顺序）
        for attribute_name, arg in kwargs.items():
            # 获取属性对应的类名
            class_name = getattr(self, f"{attribute_name}_class")
            # 检查是否为特殊的 "AutoXxx" 类型，若是则检查其对应的基类
            class_name = AUTO_TO_BASE_CLASS_MAPPING.get(class_name, class_name)
            # 如果类名是元组，则分别获取每个类并组成新的元组
            if isinstance(class_name, tuple):
                proper_class = tuple(getattr(transformers_module, n) for n in class_name if n is not None)
            else:
                # 获取对应的类对象
                proper_class = getattr(transformers_module, class_name)

            # 检查参数是否为正确的类的实例，若不是则引发 ValueError 异常
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
        # 深拷贝当前实例的属性字典
        output = copy.deepcopy(self.__dict__)

        # 获取 `__init__` 中的参数
        sig = inspect.signature(self.__init__)
        # 只保存在 `__init__` 中出现的属性
        attrs_to_save = sig.parameters
        # 不保存像 `tokenizer`, `image processor` 等属性
        attrs_to_save = [x for x in attrs_to_save if x not in self.__class__.attributes]
        # 额外需要保存的属性
        attrs_to_save += ["auto_map"]

        # 仅保留需要保存的属性
        output = {k: v for k, v in output.items() if k in attrs_to_save}

        output["processor_class"] = self.__class__.__name__

        if "tokenizer" in output:
            del output["tokenizer"]
        if "image_processor" in output:
            del output["image_processor"]
        if "feature_extractor" in output:
            del output["feature_extractor"]

        # 一些属性可能有不同的名称，但包含的对象不是简单的字符串
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
        # 将实例序列化为字典
        dictionary = self.to_dict()

        # 将字典转换为 JSON 字符串
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        # 将实例保存到 JSON 文件中
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        # 生成实例的字符串表示形式
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.attributes]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}\n\n{self.to_json_string()}"

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    @classmethod
    # 从参数和字典实例化一个 [`~processing_utils.ProcessingMixin`] 类型的对象
    def from_args_and_dict(cls, args, processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        # 复制参数字典以防止对原字典的修改
        processor_dict = processor_dict.copy()
        # 是否返回未使用的关键字参数
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # 与接受 `kwargs` 的图像处理器或特征提取器不同，处理器没有 `kwargs`
        # 我们必须弹出一些未使用的（但特定的）参数来使其工作
        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        if "auto_map" in processor_dict:
            del processor_dict["auto_map"]

        # 使用处理器字典实例化处理器对象
        processor = cls(*args, **processor_dict)

        # 如果需要，使用关键字参数更新处理器
        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        # 记录处理器对象信息
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
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a pretrained model.

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]):
                The model name or path to instantiate the processor object from.
            cache_dir (Optional[Union[str, os.PathLike]], optional):
                The directory to cache the model downloaded parameters.
            force_download (bool, optional):
                Whether or not to force redownload the model parameters.
            local_files_only (bool, optional):
                Whether or not to only look at local files for loading the model parameters.
            token (Optional[Union[str, bool]], optional):
                The token to use for authentication headers when downloading the model parameters.
            revision (str, optional):
                The specific revision of the model to load.
            **kwargs:
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those parameters.
        """
        pass  # 这里是占位符，具体实现被省略了

    @classmethod
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
        # 如果 auto_class 不是字符串，转换成类名字符串
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入 transformers.models.auto 模块
        import transformers.models.auto as auto_module

        # 检查 auto_class 是否是有效的自动类
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 设置类属性 _auto_class 为 auto_class
        cls._auto_class = auto_class

    @classmethod
    # 从预训练的模型名或路径中获取参数
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 初始化参数列表
        args = []
        # 遍历类的属性
        for attribute_name in cls.attributes:
            # 获取属性的类名
            class_name = getattr(cls, f"{attribute_name}_class")
            # 如果类名是元组
            if isinstance(class_name, tuple):
                # 获取类名对应的类对象，如果其中一个为 None，则对应位置也为 None
                classes = tuple(getattr(transformers_module, n) if n is not None else None for n in class_name)
                # 获取是否使用快速模式的标志，默认为 True
                use_fast = kwargs.get("use_fast", True)
                # 如果使用快速模式且第二个类对象不为 None，则选择第二个类对象，否则选择第一个
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                # 获取属性对应的类对象
                attribute_class = getattr(transformers_module, class_name)

            # 调用类对象的 from_pretrained 方法，获取预训练参数，并添加到参数列表中
            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        
        # 返回参数列表
        return args

    # 获取模型输入的名称
    @property
    def model_input_names(self):
        # 获取第一个属性
        first_attribute = getattr(self, self.attributes[0])
        # 如果第一个属性存在 model_input_names 属性，则返回其值，否则返回 None
        return getattr(first_attribute, "model_input_names", None)
# 将 ProcessorMixin 类的 push_to_hub 方法复制一份，并赋值给 ProcessorMixin.push_to_hub
ProcessorMixin.push_to_hub = copy_func(ProcessorMixin.push_to_hub)
# 如果 push_to_hub 方法的文档字符串不为空
if ProcessorMixin.push_to_hub.__doc__ is not None:
    # 使用格式化字符串将文档字符串中的占位符替换为相应的内容
    ProcessorMixin.push_to_hub.__doc__ = ProcessorMixin.push_to_hub.__doc__.format(
        object="processor", object_class="AutoProcessor", object_files="processor files"
    )
```