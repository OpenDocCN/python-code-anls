# `.\models\auto\feature_extraction_auto.py`

```
# 设置脚本的编码格式为 UTF-8
# 版权声明，使用 Apache License Version 2.0 授权许可
# 只有遵循许可证的条款，才能使用该文件
# 可以从 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 除非适用法律要求或书面同意，否则不得使用此文件
# 此软件根据 "原样" 分发，不提供任何形式的明示或暗示担保或条件
# 有关详细信息，请参阅许可证
""" AutoFeatureExtractor class."""

# 导入必要的模块
import importlib  # 动态导入模块的功能
import json  # 处理 JSON 数据的模块
import os  # 提供与操作系统相关的功能
import warnings  # 控制警告信息的输出
from collections import OrderedDict  # 提供有序字典的数据结构
from typing import Dict, Optional, Union  # 导入类型提示所需的类型

# 导入 transformers 库中的其他模块和函数
from ...configuration_utils import PretrainedConfig  # 预训练配置类
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code  # 动态模块相关工具函数
from ...feature_extraction_utils import FeatureExtractionMixin  # 特征提取混合类
from ...utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, get_file_from_repo, logging  # 提供各种实用功能的工具函数
from .auto_factory import _LazyAutoMapping  # 自动工厂类的延迟映射
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,  # 配置映射名称列表
    AutoConfig,  # 自动配置类
    model_type_to_module_name,  # 模型类型到模块名称的映射函数
    replace_list_option_in_docstrings,  # 在文档字符串中替换列表选项的函数
)

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 特征提取器映射名称的有序字典定义
FEATURE_EXTRACTOR_MAPPING_NAMES = OrderedDict(
    [
        # 此处应该有一些特征提取器的映射条目，但代码片段中省略了具体内容
    ]
)

# 基于配置映射名称和特征提取器映射名称创建特征提取器映射对象
FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)


def feature_extractor_class_from_name(class_name: str):
    """
    根据特征提取器类名获取对应的特征提取器类对象。

    Args:
        class_name (str): 特征提取器类的名称。

    Returns:
        type or None: 如果找到匹配的特征提取器类，则返回该类对象；否则返回 None。
    """
    # 遍历特征提取器映射名称字典中的模块名称和特征提取器类列表
    for module_name, extractors in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
        # 如果 class_name 在当前模块的特征提取器类列表中
        if class_name in extractors:
            # 将模型类型转换为相应的模块名称
            module_name = model_type_to_module_name(module_name)
            # 在 transformers.models 下动态导入对应的模块
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                # 返回特征提取器类对象
                return getattr(module, class_name)
            except AttributeError:
                continue

    # 在额外内容中查找特征提取器对象
    for _, extractor in FEATURE_EXTRACTOR_MAPPING._extra_content.items():
        # 如果特征提取器对象的 __name__ 属性等于 class_name
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # 如果在当前模块中找不到特征提取器类，可能是由于依赖项丢失，此时返回适当的 dummy 类以获得适当的错误消息
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    # 如果找不到匹配的特征提取器类，则返回 None
    return None


def get_feature_extractor_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],  # 预训练模型名称或路径
    cache_dir: Optional[Union[str, os.PathLike]] = None,  # 缓存目录，可选
    force_download: bool = False,  # 是否强制下载
    resume_download: bool = False,  # 是否恢复下载
    proxies: Optional[Dict[str, str]] = None,  # 代理设置
    token: Optional[Union[bool, str]] = None,  # 访问令牌，可选
    revision: Optional[str] = None,  # 仓库的版本号，可选
    local_files_only: bool = False,  # 仅使用本地文件
    **kwargs,  # 其他关键字参数
):
    """
    从预训练模型加载特征提取器的配置信息。

    Args:
        pretrained_model_name_or_path (Union[str, os.PathLike]): 预训练模型的名称或路径。
        cache_dir (Optional[Union[str, os.PathLike]], optional): 缓存目录路径，可选参数。默认为 None。
        force_download (bool, optional): 是否强制下载，默认为 False。
        resume_download (bool, optional): 是否恢复下载，默认为 False。
        proxies (Optional[Dict[str, str]], optional): 代理设置，可选参数。默认为 None。
        token (Optional[Union[bool, str]], optional): 访问令牌，可选参数。默认为 None。
        revision (Optional[str], optional): 仓库的版本号，可选参数。默认为 None。
        local_files_only (bool, optional): 是否仅使用本地文件，默认为 False。
        **kwargs: 其他关键字参数。

    Returns:
        None
    """
    pass  # 函数体未实现，仅有文档字符串提示函数用途
    # 从参数中获取 `use_auth_token`，如果存在则弹出并赋值给 `use_auth_token` 变量，否则设置为 `None`
    use_auth_token = kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 参数不为 None，则发出警告，说明该参数在将来版本中会被移除
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果同时指定了 token 参数，则抛出数值错误，提示只能设置 `token` 参数
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 token 参数设置为 use_auth_token 的值
        token = use_auth_token

    # 获取预训练模型名称或路径对应的特征提取器配置文件路径
    resolved_config_file = get_file_from_repo(
        pretrained_model_name_or_path,
        FEATURE_EXTRACTOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
    )
    # 如果未找到特征提取器配置文件，则记录日志并返回空字典
    if resolved_config_file is None:
        logger.info(
            "Could not locate the feature extractor configuration file, will try to use the model config instead."
        )
        return {}

    # 使用 UTF-8 编码打开特征提取器配置文件，并加载为 JSON 格式返回
    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)
class AutoFeatureExtractor:
    r"""
    This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
    library when created with the [`AutoFeatureExtractor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        # 抛出环境错误，阻止直接通过 __init__() 实例化该类
        raise EnvironmentError(
            "AutoFeatureExtractor is designed to be instantiated "
            "using the `AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(FEATURE_EXTRACTOR_MAPPING_NAMES)
    @staticmethod
    def register(config_class, feature_extractor_class, exist_ok=False):
        """
        Register a new feature extractor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            feature_extractor_class ([`FeatureExtractorMixin`]): The feature extractor to register.
        """
        # 使用 FEATURE_EXTRACTOR_MAPPING 的 register 方法注册新的特征提取器类
        FEATURE_EXTRACTOR_MAPPING.register(config_class, feature_extractor_class, exist_ok=exist_ok)
```