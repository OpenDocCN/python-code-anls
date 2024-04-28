# `.\transformers\models\auto\tokenization_auto.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2018 年的 HuggingFace 公司团队
#
# 根据 Apache 许可证 2.0 版（"许可证"）获得许可;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据"原样"分发，
# 没有任何担保或条件。
# 有关特定语言的权限，请参阅许可证。
""" 自动标记器类。"""

# 导入所需模块
import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

# 导入相关的配置和工具函数
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...utils import (
    cached_file,
    extract_commit_hash,
    is_g2p_en_available,
    is_sentencepiece_available,
    is_tokenizers_available,
    logging,
)

# 导入编码解码配置和自动工厂
from ..encoder_decoder import EncoderDecoderConfig
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    config_class_to_model_type,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)

# 检查是否安装了 Tokenizers 库，若安装则导入相应模块，否则置为 None
if is_tokenizers_available():
    from ...tokenization_utils_fast import PreTrainedTokenizerFast
else:
    PreTrainedTokenizerFast = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果是类型检查模式，则定义 TOKENIZER_MAPPING_NAMES，否则置为 None
if TYPE_CHECKING:
    # 这将显著提高使用 Microsoft 的 Pylance 语言服务器时的完成建议性能。
    TOKENIZER_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    # 否则，设置 TOKENIZER_MAPPING_NAMES 为空字典
    TOKENIZER_MAPPING_NAMES = {}

# 创建 TOKENIZER_MAPPING，映射配置到标记器类名
TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

# 创建配置到类型的映射字典
CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}

# 根据类名获取标记器类
def tokenizer_class_from_name(class_name: str):
    # 若类名为 "PreTrainedTokenizerFast"，则返回 PreTrainedTokenizerFast 类
    if class_name == "PreTrainedTokenizerFast":
        return PreTrainedTokenizerFast

    # 遍历 TOKENIZER_MAPPING_NAMES
    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        # 若类名在 tokenizers 中，则导入对应模块并返回该类
        if class_name in tokenizers:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    # 遍历 TOKENIZER_MAPPING 的额外内容
    for config, tokenizers in TOKENIZER_MAPPING._extra_content.items():
        for tokenizer in tokenizers:
            # 如果 tokenizer 的 __name__ 与 class_name 匹配，则返回该 tokenizer
            if getattr(tokenizer, "__name__", None) == class_name:
                return tokenizer

    # 如果找不到类，则可能是因为缺少依赖。在这种情况下，该类将在主模块中。
    # 初始化并返回适当的虚拟对象，以获取适当的错误消息。
    main_module = importlib.import_module("transformers")
    # 检查主模块是否包含指定的类名
    if hasattr(main_module, class_name):
        # 返回主模块中指定类名的属性
        return getattr(main_module, class_name)
    
    # 如果未找到指定类名，则返回 None
    return None
def get_tokenizer_config(
    # 定义函数，获取预训练模型的分词器配置信息
    pretrained_model_name_or_path: Union[str, os.PathLike],  # 预训练模型名称或路径
    cache_dir: Optional[Union[str, os.PathLike]] = None,  # 缓存目录，可选
    force_download: bool = False,  # 是否强制重新下载配置文件，默认为False
    resume_download: bool = False,  # 是否恢复下载，默认为False
    proxies: Optional[Dict[str, str]] = None,  # 代理服务器字典，可选
    token: Optional[Union[bool, str]] = None,  # HTTP令牌，可选
    revision: Optional[str] = None,  # 模型版本，可选，默认为"main"
    local_files_only: bool = False,  # 是否仅从本地文件加载，默认为False
    subfolder: str = "",  # 模型存储库的子文件夹名称，默认为空字符串
    **kwargs,  # 其他关键字参数
):
    """
    从预训练模型的分词器配置文件中加载分词器配置。

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            可以是以下之一：

            - 字符串，托管在huggingface.co模型存储库中的预训练模型配置的*模型ID*。有效的模型ID可以位于根级别，
              如`bert-base-uncased`，或者在用户或组织名称下命名空间，如`dbmdz/bert-base-german-cased`。
            - 包含使用[`~PreTrainedTokenizer.save_pretrained`]方法保存的配置文件的*目录*的路径，例如，`./my_model_directory/`。

        cache_dir (`str` or `os.PathLike`, *optional*):
            如果不使用标准缓存，则应将下载的预训练模型配置缓存到其中的目录路径。
        force_download (`bool`, *optional*, 默认为`False`):
            是否强制重新下载配置文件并覆盖缓存版本。
        resume_download (`bool`, *optional*, 默认为`False`):
            是否删除接收不完整的文件。如果存在此类文件，则尝试恢复下载。
        proxies (`Dict[str, str]`, *optional*):
            一个代理服务器字典，用于协议或端点，例如，`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`。
            代理服务器将在每个请求上使用。
        token (`str` or *bool*, *optional*):
            用作远程文件的HTTP bearer授权的令牌。如果为`True`，则在运行`huggingface-cli login`时生成的令牌将被使用（存储在`~/.huggingface`中）。
        revision (`str`, *optional*, 默认为`"main"`):
            要使用的特定模型版本。它可以是分支名称、标签名称或提交ID，因为我们使用基于git的系统在huggingface.co上存储模型和其他文件，
            所以`revision`可以是git允许的任何标识符。
        local_files_only (`bool`, *optional*, 默认为`False`):
            如果为`True`，将仅尝试从本地文件加载分词器配置。
        subfolder (`str`, *optional*, 默认为`""`):
            如果分词器配置位于huggingface.co模型存储库的子文件夹中，您可以在此处指定文件夹名称。

    <Tip>

    当您想要使用私有模型时，传递`token=True`是必需的。

    </Tip>
    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    tokenizer_config = get_tokenizer_config("bert-base-uncased")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""
    # 从kwargs中取出use_auth_token参数，如果存在则弹出
    use_auth_token = kwargs.pop("use_auth_token", None)
    # 如果use_auth_token参数不为空，发出警告，该参数即将被移除
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果token也不为空，则抛出数值错误，不允许同时设置token和use_auth_token
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将token设置为use_auth_token的值
        token = use_auth_token

    # 获取_commit_hash参数的值，默认为None
    commit_hash = kwargs.get("_commit_hash", None)
    # 通过cached_file函数获取tokenizer配置文件的路径
    resolved_config_file = cached_file(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        _commit_hash=commit_hash,
    )
    # 如果无法找到tokenizer配置文件，则记录日志并返回空字典
    if resolved_config_file is None:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}
    # 从配置文件中提取commit_hash
    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)

    # 打开配置文件，读取其中内容到result字典中
    with open(resolved_config_file, encoding="utf-8") as reader:
        result = json.load(reader)
    # 将commit_hash存入result字典中
    result["_commit_hash"] = commit_hash
    # 返回result字典，即tokenizer的配置信息
    return result
class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the [`AutoTokenizer.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        # 抛出环境错误，不能直接实例化此类
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(TOKENIZER_MAPPING_NAMES)
    def register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False):
        """
        Register a new tokenizer in this mapping.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            slow_tokenizer_class ([`PretrainedTokenizer`], *optional*):
                The slow tokenizer to register.
            fast_tokenizer_class ([`PretrainedTokenizerFast`], *optional*):
                The fast tokenizer to register.
        """
        # 如果既未传入慢速分词器类也未传入快速分词器类，则抛出值错误
        if slow_tokenizer_class is None and fast_tokenizer_class is None:
            raise ValueError("You need to pass either a `slow_tokenizer_class` or a `fast_tokenizer_class")
        # 如果传入的慢速分词器类为快速分词器的子类，则抛出值错误
        if slow_tokenizer_class is not None and issubclass(slow_tokenizer_class, PreTrainedTokenizerFast):
            raise ValueError("You passed a fast tokenizer in the `slow_tokenizer_class`.")
        # 如果传入的快速分词器类为慢速分词器的子类，则抛出值错误
        if fast_tokenizer_class is not None and issubclass(fast_tokenizer_class, PreTrainedTokenizer):
            raise ValueError("You passed a slow tokenizer in the `fast_tokenizer_class`.")

        # 如果同时传入慢速分词器类和快速分词器类，并且快速分词器类的慢速分词器类属性与传入的慢速分词器类不一致，则抛出值错误
        if (
            slow_tokenizer_class is not None
            and fast_tokenizer_class is not None
            and issubclass(fast_tokenizer_class, PreTrainedTokenizerFast)
            and fast_tokenizer_class.slow_tokenizer_class != slow_tokenizer_class
        ):
            raise ValueError(
                "The fast tokenizer class you are passing has a `slow_tokenizer_class` attribute that is not "
                "consistent with the slow tokenizer class you passed (fast tokenizer has "
                f"{fast_tokenizer_class.slow_tokenizer_class} and you passed {slow_tokenizer_class}. Fix one of those "
                "so they match!"
            )

        # 如果在TOKENIZER_MAPPING._extra_content中存在配置类，则更新慢速和快速分词器类
        if config_class in TOKENIZER_MAPPING._extra_content:
            existing_slow, existing_fast = TOKENIZER_MAPPING[config_class]
            if slow_tokenizer_class is None:
                slow_tokenizer_class = existing_slow
            if fast_tokenizer_class is None:
                fast_tokenizer_class = existing_fast

        # 将配置类及其慢速和快速分词器类注册到TOKENIZER_MAPPING中
        TOKENIZER_MAPPING.register(config_class, (slow_tokenizer_class, fast_tokenizer_class), exist_ok=exist_ok)
```