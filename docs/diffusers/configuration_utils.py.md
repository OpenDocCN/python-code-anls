# `.\diffusers\configuration_utils.py`

```py
# 指定编码为 UTF-8
# 版权声明，指明版权归 HuggingFace Inc. 团队所有
# 版权声明，指明版权归 NVIDIA CORPORATION 所有
#
# 根据 Apache License, Version 2.0 授权本文件
# 只能在遵循许可证的情况下使用该文件
# 可在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 本文件按“原样”提供，未提供任何明示或暗示的保证或条件
# 参见许可证中关于权限和限制的具体条款
"""配置混合类的基类及其工具函数。"""

# 导入必要的库和模块
import dataclasses  # 提供数据类支持
import functools  # 提供高阶函数的工具
import importlib  # 提供导入模块的功能
import inspect  # 提供获取对象信息的功能
import json  # 提供JSON数据的处理
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式支持
from collections import OrderedDict  # 提供有序字典支持
from pathlib import Path  # 提供路径操作支持
from typing import Any, Dict, Tuple, Union  # 提供类型提示支持

import numpy as np  # 导入 NumPy 库
from huggingface_hub import create_repo, hf_hub_download  # 从 Hugging Face Hub 导入相关函数
from huggingface_hub.utils import (  # 导入 Hugging Face Hub 工具中的异常和验证函数
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from requests import HTTPError  # 导入处理 HTTP 错误的类

from . import __version__  # 导入当前模块的版本信息
from .utils import (  # 从工具模块导入常用工具
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    DummyObject,
    deprecate,
    extract_commit_hash,
    http_user_agent,
    logging,
)

# 创建日志记录器实例
logger = logging.get_logger(__name__)

# 编译正则表达式用于匹配配置文件名
_re_configuration_file = re.compile(r"config\.(.*)\.json")


class FrozenDict(OrderedDict):  # 定义一个不可变字典类，继承自有序字典
    def __init__(self, *args, **kwargs):  # 初始化方法，接收任意参数
        super().__init__(*args, **kwargs)  # 调用父类初始化方法

        for key, value in self.items():  # 遍历字典中的每个键值对
            setattr(self, key, value)  # 将每个键值对作为属性设置

        self.__frozen = True  # 标记字典为不可变状态

    def __delitem__(self, *args, **kwargs):  # 禁止使用 del 删除字典项
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")  # 抛出异常

    def setdefault(self, *args, **kwargs):  # 禁止使用 setdefault 方法
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")  # 抛出异常

    def pop(self, *args, **kwargs):  # 禁止使用 pop 方法
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")  # 抛出异常

    def update(self, *args, **kwargs):  # 禁止使用 update 方法
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")  # 抛出异常

    def __setattr__(self, name, value):  # 重写设置属性的方法
        if hasattr(self, "__frozen") and self.__frozen:  # 检查是否已被冻结
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")  # 抛出异常
        super().__setattr__(name, value)  # 调用父类方法设置属性

    def __setitem__(self, name, value):  # 重写设置字典项的方法
        if hasattr(self, "__frozen") and self.__frozen:  # 检查是否已被冻结
            raise Exception(f"You cannot use ``__setitem__`` on a {self.__class__.__name__} instance.")  # 抛出异常
        super().__setitem__(name, value)  # 调用父类方法设置字典项


class ConfigMixin:  # 定义配置混合类
    r"""  # 类文档字符串，描述类的用途
    Base class for all configuration classes. All configuration parameters are stored under `self.config`. Also
    provides the [`~ConfigMixin.from_config`] and [`~ConfigMixin.save_config`] methods for loading, downloading, and
    # 保存从 `ConfigMixin` 继承的类的配置。
    # 类属性：
    # - **config_name** (`str`) -- 应该在调用 `~ConfigMixin.save_config` 时存储的配置文件名（应由父类重写）。
    # - **ignore_for_config** (`List[str]`) -- 不应在配置中保存的属性列表（应由子类重写）。
    # - **has_compatibles** (`bool`) -- 类是否有兼容的类（应由子类重写）。
    # - **_deprecated_kwargs** (`List[str]`) -- 已废弃的关键字参数。注意，`init` 函数只有在至少有一个参数被废弃时才应具有 `kwargs` 参数（应由子类重写）。
    class ConfigMixin:
        config_name = None  # 配置文件名初始化为 None
        ignore_for_config = []  # 不保存到配置的属性列表初始化为空
        has_compatibles = False  # 默认没有兼容类
    
        _deprecated_kwargs = []  # 已废弃的关键字参数列表初始化为空
    
        def register_to_config(self, **kwargs):
            # 检查 config_name 是否已定义
            if self.config_name is None:
                raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
            # 针对用于废弃警告的特殊情况
            # TODO: 当移除废弃警告和 `kwargs` 参数时删除此处
            kwargs.pop("kwargs", None)  # 从 kwargs 中移除 "kwargs"
    
            # 如果没有 _internal_dict 则初始化
            if not hasattr(self, "_internal_dict"):
                internal_dict = kwargs  # 直接使用 kwargs
            else:
                previous_dict = dict(self._internal_dict)  # 复制之前的字典
                # 合并之前的字典和新的 kwargs
                internal_dict = {**self._internal_dict, **kwargs}
                logger.debug(f"Updating config from {previous_dict} to {internal_dict}")  # 记录更新日志
    
            self._internal_dict = FrozenDict(internal_dict)  # 将内部字典冻结以防修改
    
        def __getattr__(self, name: str) -> Any:
            """覆盖 `getattr` 的唯一原因是优雅地废弃直接访问配置属性。
            参见 https://github.com/huggingface/diffusers/pull/3129
    
            此函数主要复制自 PyTorch 的 `__getattr__` 重写。
            """
            # 检查是否在 _internal_dict 中，并且该名称存在
            is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
            is_attribute = name in self.__dict__  # 检查是否为直接属性
    
            # 如果在配置中但不是属性，则发出废弃警告
            if is_in_config and not is_attribute:
                deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'scheduler.config.{name}'."
                deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)
                return self._internal_dict[name]  # 通过 _internal_dict 返回该属性
    
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")  # 引发属性错误
    # 定义一个保存配置的方法，接受保存目录和其他可选参数
    def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        保存配置对象到指定目录 `save_directory`，以便使用
        [`~ConfigMixin.from_config`] 类方法重新加载。

        参数:
            save_directory (`str` 或 `os.PathLike`):
                保存配置 JSON 文件的目录（如果不存在则会创建）。
            push_to_hub (`bool`, *可选*, 默认值为 `False`):
                保存后是否将模型推送到 Hugging Face Hub。可以用 `repo_id` 指定
                要推送的仓库（默认为 `save_directory` 的名称）。
            kwargs (`Dict[str, Any]`, *可选*):
                额外的关键字参数，将传递给 [`~utils.PushToHubMixin.push_to_hub`] 方法。
        """
        # 如果提供的路径是文件，则抛出异常，要求是目录
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        # 创建目录，存在时不会报错
        os.makedirs(save_directory, exist_ok=True)

        # 根据预定义名称保存时，可以使用 `from_config` 加载
        output_config_file = os.path.join(save_directory, self.config_name)

        # 将配置写入 JSON 文件
        self.to_json_file(output_config_file)
        # 记录保存配置的日志信息
        logger.info(f"Configuration saved in {output_config_file}")

        # 如果需要推送到 Hub
        if push_to_hub:
            # 从 kwargs 中弹出提交信息
            commit_message = kwargs.pop("commit_message", None)
            # 从 kwargs 中弹出私有标志
            private = kwargs.pop("private", False)
            # 从 kwargs 中弹出创建 PR 的标志
            create_pr = kwargs.pop("create_pr", False)
            # 从 kwargs 中弹出令牌
            token = kwargs.pop("token", None)
            # 从 kwargs 中获取 repo_id，默认为保存目录名称
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            # 创建仓库，若存在则不报错，并返回仓库 ID
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

            # 上传文件夹到指定的仓库
            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    # 定义一个类方法，获取配置字典
    @classmethod
    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        # 生成废弃消息，提醒用户此方法将被移除
        deprecation_message = (
            f" The function get_config_dict is deprecated. Please use {cls}.load_config instead. This function will be"
            " removed in version v1.0.0"
        )
        # 调用废弃警告函数
        deprecate("get_config_dict", "1.0.0", deprecation_message, standard_warn=False)
        # 返回加载的配置
        return cls.load_config(*args, **kwargs)

    # 定义一个类方法，用于加载配置
    @classmethod
    @validate_hf_hub_args
    def load_config(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        return_unused_kwargs=False,
        return_commit_hash=False,
        **kwargs,
    # 定义一个静态方法，获取初始化所需的关键字
    @staticmethod
    def _get_init_keys(input_class):
        # 返回类初始化方法的参数名称集合
        return set(dict(inspect.signature(input_class.__init__).parameters).keys())

    # 额外的类方法
    @classmethod
    @classmethod
    # 从 JSON 文件创建字典
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        # 打开指定的 JSON 文件，使用 UTF-8 编码读取内容
        with open(json_file, "r", encoding="utf-8") as reader:
            # 读取文件内容并存储到变量 text 中
            text = reader.read()
        # 将读取的 JSON 字符串解析为字典并返回
        return json.loads(text)

    # 返回类的字符串表示形式
    def __repr__(self):
        # 使用类名和 JSON 字符串表示配置实例返回字符串
        return f"{self.__class__.__name__} {self.to_json_string()}"

    # 定义一个只读属性 config
    @property
    def config(self) -> Dict[str, Any]:
        """
        返回类的配置作为一个不可变字典

        Returns:
            `Dict[str, Any]`: 类的配置字典。
        """
        # 返回内部字典 _internal_dict
        return self._internal_dict

    # 将配置实例序列化为 JSON 字符串
    def to_json_string(self) -> str:
        """
        将配置实例序列化为 JSON 字符串。

        Returns:
            `str`:
                包含配置实例的所有属性的 JSON 格式字符串。
        """
        # 检查是否存在 _internal_dict，若不存在则使用空字典
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        # 将类名添加到配置字典中
        config_dict["_class_name"] = self.__class__.__name__
        # 将当前版本添加到配置字典中
        config_dict["_diffusers_version"] = __version__

        # 定义一个用于将值转换为可保存的 JSON 格式的函数
        def to_json_saveable(value):
            # 如果值是 numpy 数组，则转换为列表
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # 如果值是 Path 对象，则转换为 POSIX 路径字符串
            elif isinstance(value, Path):
                value = value.as_posix()
            # 返回转换后的值
            return value

        # 对配置字典中的每个键值对进行转换
        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        # 从字典中移除 "_ignore_files" 和 "_use_default_values" 项
        config_dict.pop("_ignore_files", None)
        config_dict.pop("_use_default_values", None)

        # 将配置字典转换为格式化的 JSON 字符串并返回
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    # 将配置实例的参数保存到 JSON 文件
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        将配置实例的参数保存到 JSON 文件。

        Args:
            json_file_path (`str` or `os.PathLike`):
                要保存配置实例参数的 JSON 文件路径。
        """
        # 打开指定的 JSON 文件进行写入，使用 UTF-8 编码
        with open(json_file_path, "w", encoding="utf-8") as writer:
            # 将配置实例转换为 JSON 字符串并写入文件
            writer.write(self.to_json_string())
# 装饰器，用于应用在继承自 [`ConfigMixin`] 的类的初始化方法上，自动将所有参数发送到 `self.register_for_config`
def register_to_config(init):
    # 文档字符串，描述装饰器的功能和警告
    r"""
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    # 包装原始初始化方法，以便在其上添加功能
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # 忽略初始化方法中的私有关键字参数
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        # 提取私有关键字参数以供后续使用
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        # 检查当前类是否继承自 ConfigMixin
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )

        # 获取需要忽略的配置参数列表
        ignore = getattr(self, "ignore_for_config", [])
        # 对齐位置参数与关键字参数
        new_kwargs = {}
        # 获取初始化方法的签名
        signature = inspect.signature(init)
        # 提取参数名和默认值，排除忽略的参数
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore
        }
        # 将位置参数映射到新关键字参数
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # 更新新关键字参数，加入所有未被忽略的关键字参数
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )

        # 记录未在配置中出现的参数
        if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
            new_kwargs["_use_default_values"] = list(set(new_kwargs.keys()) - set(init_kwargs))

        # 合并配置初始化参数和新关键字参数
        new_kwargs = {**config_init_kwargs, **new_kwargs}
        # 调用类的注册方法，将参数发送到配置
        getattr(self, "register_to_config")(**new_kwargs)
        # 调用原始初始化方法
        init(self, *args, **init_kwargs)

    # 返回包装后的初始化方法
    return inner_init


# 装饰器函数，用于在类上注册配置功能
def flax_register_to_config(cls):
    # 保存原始初始化方法
    original_init = cls.__init__

    # 包装原始初始化方法，以便在其上添加功能
    @functools.wraps(original_init)
    # 定义初始化方法，接受可变位置和关键字参数
        def init(self, *args, **kwargs):
            # 检查当前实例是否继承自 ConfigMixin
            if not isinstance(self, ConfigMixin):
                raise RuntimeError(
                    # 抛出异常，提示类未继承 ConfigMixin
                    f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                    "not inherit from `ConfigMixin`."
                )
    
            # 忽略私有关键字参数，获取所有传入的属性
            init_kwargs = dict(kwargs.items())
    
            # 获取默认值
            fields = dataclasses.fields(self)
            default_kwargs = {}
            for field in fields:
                # 忽略 flax 特定属性
                if field.name in self._flax_internal_args:
                    continue
                # 检查字段的默认值是否缺失
                if type(field.default) == dataclasses._MISSING_TYPE:
                    default_kwargs[field.name] = None
                else:
                    # 获取字段的默认值
                    default_kwargs[field.name] = getattr(self, field.name)
    
            # 确保 init_kwargs 可以覆盖默认值
            new_kwargs = {**default_kwargs, **init_kwargs}
            # 从 new_kwargs 中移除 dtype，确保它仅在 init_kwargs 中
            if "dtype" in new_kwargs:
                new_kwargs.pop("dtype")
    
            # 获取与关键字参数对齐的位置参数
            for i, arg in enumerate(args):
                name = fields[i].name
                new_kwargs[name] = arg
    
            # 记录未在加载配置中出现的参数
            if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
                new_kwargs["_use_default_values"] = list(set(new_kwargs.keys()) - set(init_kwargs))
    
            # 调用 register_to_config 方法，传入新构建的关键字参数
            getattr(self, "register_to_config")(**new_kwargs)
            # 调用原始初始化方法
            original_init(self, *args, **kwargs)
    
        # 将自定义初始化方法赋值给类的 __init__ 方法
        cls.__init__ = init
        return cls
# 定义一个名为 LegacyConfigMixin 的类，它是 ConfigMixin 的子类
class LegacyConfigMixin(ConfigMixin):
    r"""
    该类是 `ConfigMixin` 的子类，用于将旧类（如 `Transformer2DModel`）映射到更
    特定于管道的类（如 `DiTTransformer2DModel`）。
    """

    # 定义一个类方法 from_config，接收配置和其他可选参数
    @classmethod
    def from_config(cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs):
        # 为了防止依赖导入问题，从指定模块导入函数
        from .models.model_loading_utils import _fetch_remapped_cls_from_config

        # 调用函数，解析类的映射关系
        remapped_class = _fetch_remapped_cls_from_config(config, cls)

        # 返回映射后的类使用配置和其他参数进行的实例化
        return remapped_class.from_config(config, return_unused_kwargs, **kwargs)
```