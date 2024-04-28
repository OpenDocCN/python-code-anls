# `.\transformers\models\auto\feature_extraction_auto.py`

```
# 导入所需的模块和库
# 这些模块包括了用于动态导入和加载类、配置管理、日志记录等功能
import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Union

# 从 Transformers 库中导入一些实用函数和类
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, get_file_from_repo, logging

# 导入特定于自动特征提取器的工厂和配置
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 构建所有特征提取器的映射关系
FEATURE_EXTRACTOR_MAPPING_NAMES = OrderedDict(
    # 这里应该填写特征提取器名称和对应的模块名称，但代码中此处留空了
)

# 将特征提取器映射关系封装成 LazyAutoMapping 对象
FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)


# 根据特征提取器类名获取特征提取器类
def feature_extractor_class_from_name(class_name: str):
    # 遍历特征提取器映射关系，寻找与给定类名匹配的特征提取器类
    for module_name, extractors in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
        if class_name in extractors:
            # 将模块名称转换为对应的模型名称
            module_name = model_type_to_module_name(module_name)
            # 动态导入模块
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                # 获取特征提取器类并返回
                return getattr(module, class_name)
            except AttributeError:
                continue

    # 如果在映射关系中未找到特征提取器类，则尝试在额外内容中查找
    for _, extractor in FEATURE_EXTRACTOR_MAPPING._extra_content.items():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # 如果仍未找到特征提取器类，则尝试从主要的 init 文件中查找
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


# 加载预训练模型的特征提取器配置
def get_feature_extractor_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.
    加载预训练模型的特征提取器配置。

    """
    # 此函数负责加载特征提取器的配置，但代码中此处留空，需要补充实现
```  
    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            模型名称或路径，可以是以下之一：

            - 字符串，预训练模型配置的*模型标识*，托管在huggingface.co上的模型存储库中。有效的模型标识可以位于根级别，如`bert-base-uncased`，或者在用户或组织名称下的命名空间中，如`dbmdz/bert-base-german-cased`。
            - 包含使用[`~PreTrainedTokenizer.save_pretrained`]方法保存的配置文件的*目录*路径，例如，`./my_model_directory/`。

        cache_dir (`str` or `os.PathLike`, *optional*):
            下载的预训练模型配置应该缓存在其中的目录路径，如果不想使用标准缓存，则使用该选项。
        force_download (`bool`, *optional*, 默认为 `False`):
            是否强制（重新）下载配置文件，并覆盖缓存的版本（如果存在）。
        resume_download (`bool`, *optional*, 默认为 `False`):
            是否删除未完全接收的文件。如果存在这样的文件，尝试恢复下载。
        proxies (`Dict[str, str]`, *optional*):
            要使用的代理服务器字典，按协议或端点，例如，`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`。代理服务器在每个请求上使用。
        token (`str` or *bool*, *optional*):
            用作远程文件的HTTP令牌的令牌。如果为`True`，将使用在运行`huggingface-cli login`时生成的令牌（存储在`~/.huggingface`中）。
        revision (`str`, *optional*, 默认为 `"main"`):
            要使用的特定模型版本。它可以是分支名称、标签名称或提交ID，因为我们在huggingface.co上使用基于git的系统存储模型和其他工件，所以`revision`可以是git允许的任何标识符。
        local_files_only (`bool`, *optional*, 默认为 `False`):
            如果为`True`，则仅尝试从本地文件加载分词器配置。

    <Tip>

    当您想使用私有模型时，需要传递`token=True`。

    </Tip>

    Returns:
        `Dict`: 分词器的配置。

    Examples:

    ```python
    # 从huggingface.co下载配置并缓存。
    tokenizer_config = get_tokenizer_config("bert-base-uncased")
    # 此模型没有分词器配置，因此结果将是一个空字典。
    tokenizer_config = get_tokenizer_config("xlm-roberta-base")

    # 将预训练分词器保存到本地，然后可以重新加载其配置
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""
    # 从kwargs中弹出use_auth_token参数
    use_auth_token = kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 不为 None，则发出警告
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 token 不为 None，则抛出数值错误
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 token 设置为 use_auth_token
        token = use_auth_token

    # 从仓库中获取预训练模型的配置文件
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
    # 如果未找到特征提取器配置文件，则记录信息并返回空字典
    if resolved_config_file is None:
        logger.info(
            "Could not locate the feature extractor configuration file, will try to use the model config instead."
        )
        return {}

    # 打开解析后的配置文件，使用 utf-8 编码，加载为 JSON 格式并返回
    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)
class AutoFeatureExtractor:
    r"""
    This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
    library when created with the [`AutoFeatureExtractor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        # 抛出环境错误，提示用户使用 `AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)` 方法实例化
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
        # 将配置类和特征提取器类注册到特征提取器映射中
        FEATURE_EXTRACTOR_MAPPING.register(config_class, feature_extractor_class, exist_ok=exist_ok)
```