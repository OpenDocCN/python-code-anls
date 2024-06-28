# `.\models\auto\tokenization_auto.py`

```
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Tokenizer class."""

import importlib  # 导入用于动态导入模块的标准库
import json  # 导入处理 JSON 格式数据的标准库
import os  # 导入与操作系统交互的标准库
import warnings  # 导入警告处理相关的标准库
from collections import OrderedDict  # 导入有序字典的标准库
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union  # 导入类型提示相关的标准库

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code  # 导入动态模块相关工具函数
from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器基类
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE  # 导入分词器配置文件常量
from ...utils import (  # 导入一些工具函数
    cached_file,
    extract_commit_hash,
    is_g2p_en_available,
    is_sentencepiece_available,
    is_tokenizers_available,
    logging,
)
from ..encoder_decoder import EncoderDecoderConfig  # 导入编码器解码器配置类
from .auto_factory import _LazyAutoMapping  # 导入自动工厂相关类
from .configuration_auto import (  # 导入自动配置相关的模块
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    config_class_to_model_type,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)

if is_tokenizers_available():
    from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 如果有安装 tokenizers，导入快速分词器
else:
    PreTrainedTokenizerFast = None  # 否则将快速分词器设为 None

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

if TYPE_CHECKING:
    # 定义一个有序字典，用于存储分词器名称及其对应的模块名和类名元组
    TOKENIZER_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    # 如果不是类型检查模式，则 TOKENIZER_MAPPING_NAMES 初始化为空
    TOKENIZER_MAPPING_NAMES = OrderedDict()

# 使用 _LazyAutoMapping 类初始化 TOKENIZER_MAPPING
TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

# 通过 CONFIG_MAPPING_NAMES 创建反向映射字典，用于从映射名称到类型的转换
CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


def tokenizer_class_from_name(class_name: str):
    # 根据类名返回相应的分词器类对象
    if class_name == "PreTrainedTokenizerFast":
        return PreTrainedTokenizerFast

    # 遍历 TOKENIZER_MAPPING_NAMES，查找与 class_name 匹配的分词器类
    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        if class_name in tokenizers:
            module_name = model_type_to_module_name(module_name)

            # 动态导入 transformers.models 下的指定模块
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)  # 返回指定模块下的类对象
            except AttributeError:
                continue

    # 如果在 TOKENIZER_MAPPING 中找不到对应的类，尝试从 _extra_content 中查找
    for config, tokenizers in TOKENIZER_MAPPING._extra_content.items():
        for tokenizer in tokenizers:
            if getattr(tokenizer, "__name__", None) == class_name:
                return tokenizer

    # 如果以上方法都无法找到指定类，则返回 None
    # 这段代码用于处理未能找到类的情况，可能是由于依赖项缺失导致的
    # 在这种情况下，该类应该在主要的模块中
    # 导入 importlib 模块，并使用它来导入名为 "transformers" 的模块
    main_module = importlib.import_module("transformers")
    # 检查在导入的模块中是否存在名为 class_name 的属性
    if hasattr(main_module, class_name):
        # 如果存在，则返回该属性对应的对象或函数
        return getattr(main_module, class_name)

    # 如果不存在名为 class_name 的属性，则返回 None
    return None
# 加载预训练模型的分词器配置信息
def get_tokenizer_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    **kwargs,
):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the tokenizer config is located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # 从huggingface.co下载配置文件并进行缓存
    ```
    # 获取指定预训练模型的分词器配置信息
    tokenizer_config = get_tokenizer_config("google-bert/bert-base-uncased")
    # 由于这个模型没有分词器配置，所以结果将会是一个空字典。
    tokenizer_config = get_tokenizer_config("FacebookAI/xlm-roberta-base")

    # 导入transformers库中的AutoTokenizer类，用于自动获取预训练模型的分词器
    from transformers import AutoTokenizer

    # 从预训练模型路径中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    # 将分词器保存到本地目录"tokenizer-test"中
    tokenizer.save_pretrained("tokenizer-test")
    # 获取保存的分词器配置信息
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```
    # 处理`use_auth_token`参数的兼容性警告和错误处理逻辑
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        # 发出将在Transformers v5中移除`use_auth_token`参数的警告
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果同时指定了`token`和`use_auth_token`参数，则抛出错误
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将`use_auth_token`参数的值赋给`token`变量
        token = use_auth_token

    # 获取kwargs中的_commit_hash参数值
    commit_hash = kwargs.get("_commit_hash", None)
    # 解析和缓存预训练模型的tokenizer配置文件
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
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        _commit_hash=commit_hash,
    )
    # 如果未能定位tokenizer配置文件，则记录日志并返回空字典
    if resolved_config_file is None:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}
    
    # 提取配置文件的提交哈希值
    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)

    # 打开配置文件并加载其内容到result字典中
    with open(resolved_config_file, encoding="utf-8") as reader:
        result = json.load(reader)
    # 将提取的提交哈希值存入result字典中的"_commit_hash"键
    result["_commit_hash"] = commit_hash
    return result
class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the [`AutoTokenizer.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        # 抛出环境错误，阻止直接实例化该类
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
        # 检查是否提供了慢速或快速的分词器类，否则抛出值错误
        if slow_tokenizer_class is None and fast_tokenizer_class is None:
            raise ValueError("You need to pass either a `slow_tokenizer_class` or a `fast_tokenizer_class`.")
        # 如果在`slow_tokenizer_class`中传入了快速分词器类，则抛出值错误
        if slow_tokenizer_class is not None and issubclass(slow_tokenizer_class, PreTrainedTokenizerFast):
            raise ValueError("You passed a fast tokenizer in the `slow_tokenizer_class`.")
        # 如果在`fast_tokenizer_class`中传入了慢速分词器类，则抛出值错误
        if fast_tokenizer_class is not None and issubclass(fast_tokenizer_class, PreTrainedTokenizer):
            raise ValueError("You passed a slow tokenizer in the `fast_tokenizer_class`.")

        # 如果同时提供了慢速和快速分词器类，并且快速分词器类有一个与传入的慢速分词器类不一致的`slow_tokenizer_class`属性，则抛出值错误
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

        # 如果已经在TOKENIZER_MAPPING._extra_content中注册了config_class，则尝试使用现有的慢速和快速分词器类
        if config_class in TOKENIZER_MAPPING._extra_content:
            existing_slow, existing_fast = TOKENIZER_MAPPING[config_class]
            if slow_tokenizer_class is None:
                slow_tokenizer_class = existing_slow
            if fast_tokenizer_class is None:
                fast_tokenizer_class = existing_fast

        # 在TOKENIZER_MAPPING中注册config_class与其对应的慢速和快速分词器类的映射
        TOKENIZER_MAPPING.register(config_class, (slow_tokenizer_class, fast_tokenizer_class), exist_ok=exist_ok)
```