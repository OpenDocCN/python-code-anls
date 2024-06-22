# `.\transformers\models\auto\image_processing_auto.py`

```py
# 导入必要的库和模块
import importlib  # 动态导入模块的库
import json  # 处理 JSON 格式数据的库
import os  # 处理文件路径的库
import warnings  # 提供警告功能的库
from collections import OrderedDict  # 提供有序字典功能的库
from typing import Dict, Optional, Union  # 提供类型提示功能的库

# 导入 HuggingFace 框架的相关模块和函数
from ...configuration_utils import PretrainedConfig  # 预训练模型配置的相关函数
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code  # 动态模块相关的函数
from ...image_processing_utils import ImageProcessingMixin  # 图像处理相关的函数和类
from ...utils import CONFIG_NAME, IMAGE_PROCESSOR_NAME, get_file_from_repo, logging  # 框架的工具函数和日志记录函数
from .auto_factory import _LazyAutoMapping  # 自动工厂的相关函数和类
from .configuration_auto import (  # 自动配置的相关函数和类
    CONFIG_MAPPING_NAMES,  # 配置映射名称的列表
    AutoConfig,  # 自动配置类
    model_type_to_module_name,  # 模型类型到模块名称的映射函数
    replace_list_option_in_docstrings,  # 替换文档字符串中列表选项的函数
)

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 图像处理器映射名称的有序字典
IMAGE_PROCESSOR_MAPPING_NAMES = OrderedDict(
    # 留空，将由 _LazyAutoMapping 自动填充
)

# 图像处理器映射对象
IMAGE_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, IMAGE_PROCESSOR_MAPPING_NAMES)


def image_processor_class_from_name(class_name: str):
    # 遍历图像处理器映射名称的项
    for module_name, extractors in IMAGE_PROCESSOR_MAPPING_NAMES.items():
        # 如果类名在映射中
        if class_name in extractors:
            # 将模块名称转换为实际模块名称
            module_name = model_type_to_module_name(module_name)

            # 动态导入模块
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                # 获取类对象并返回
                return getattr(module, class_name)
            except AttributeError:
                continue

    # 检查额外内容中的类对象
    for _, extractor in IMAGE_PROCESSOR_MAPPING._extra_content.items():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # 如果找不到类对象，检查是否是因为缺少依赖，若是，则返回相应的虚拟类以获取适当的错误消息
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_image_processor_config(
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
    Loads the image processor configuration from a pretrained model image processor configuration.

    """
    def get_image_processor_config(
        pretrained_model_name_or_path,
        cache_dir=None,
        force_download=False,
        resume_download=False,
        proxies=None,
        token=None,
        revision="main",
        local_files_only=False,
    ):
        """
        获取图像处理器的配置。
    
        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                可以是以下之一：
    
                - 字符串，托管在 huggingface.co 模型库中的预训练模型配置的*模型标识*。有效的模型标识可以位于根级别，如 `bert-base-uncased`，或者命名空间下的用户或组织名称，如 `dbmdz/bert-base-german-cased`。
                - 包含使用 [`~PreTrainedTokenizer.save_pretrained`] 方法保存的配置文件的*目录*路径，例如 `./my_model_directory/`。
    
            cache_dir (`str` or `os.PathLike`, *可选*):
                如果不使用标准缓存，则指定一个目录，其中应该缓存下载的预训练模型配置。
            force_download (`bool`, *可选*, 默认为 `False`):
                是否强制重新下载配置文件并覆盖已缓存的版本（如果存在）。
            resume_download (`bool`, *可选*, 默认为 `False`):
                是否删除接收不完整的文件。如果存在这样的文件，则尝试恢复下载。
            proxies (`Dict[str, str]`, *可选*):
                一个代理服务器的字典，按协议或端点使用，例如 `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`。代理服务器在每个请求上使用。
            token (`str` or *bool*, *可选*):
                用作远程文件的 HTTP bearer 授权的令牌。如果为 `True`，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
            revision (`str`, *可选*, 默认为 `"main"`):
                要使用的特定模型版本。它可以是分支名称、标签名称或提交 ID，因为我们使用基于 git 的系统在 huggingface.co 上存储模型和其他资源，所以 `revision` 可以是 git 允许的任何标识符。
            local_files_only (`bool`, *可选*, 默认为 `False`):
                如果为 `True`，则仅尝试从本地文件加载图像处理器配置。
    
        <Tip>
    
        当想要使用私有模型时，传递 `token=True` 是必需的。
    
        </Tip>
    
        Returns:
            `Dict`: 图像处理器的配置。
    
        Examples:
    
        ```python
        # 从 huggingface.co 下载配置并缓存。
        image_processor_config = get_image_processor_config("bert-base-uncased")
        # 此模型没有图像处理器配置，因此结果将是一个空字典。
        image_processor_config = get_image_processor_config("xlm-roberta-base")
    
        # 本地保存一个预训练的图像处理器，然后可以重新加载其配置
        from transformers import AutoTokenizer
    
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image_processor.save_pretrained("image-processor-test")
        ```py
        """
        pass
    # 获取图像处理器配置信息
    image_processor_config = get_image_processor_config("image-processor-test")
    
    # 检查是否使用授权令牌，如果是则发出警告
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果同时指定了`token`和`use_auth_token`，则引发错误
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # 从仓库中获取解析后的配置文件
    resolved_config_file = get_file_from_repo(
        pretrained_model_name_or_path,
        IMAGE_PROCESSOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
    )
    
    # 如果未找到解析后的配置文件，则记录日志并返回空字典
    if resolved_config_file is None:
        logger.info(
            "Could not locate the image processor configuration file, will try to use the model config instead."
        )
        return {}

    # 打开解析后的配置文件，加载其中的 JSON 数据并返回
    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)
class AutoImageProcessor:
    r"""
    This is a generic image processor class that will be instantiated as one of the image processor classes of the
    library when created with the [`AutoImageProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        # 抛出环境错误，防止直接实例化该类，提示应该使用 `AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)` 方法
        raise EnvironmentError(
            "AutoImageProcessor is designed to be instantiated "
            "using the `AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(IMAGE_PROCESSOR_MAPPING_NAMES)
    @staticmethod
    def register(config_class, image_processor_class, exist_ok=False):
        """
        Register a new image processor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            image_processor_class ([`ImageProcessingMixin`]): The image processor to register.
        """
        # 注册一个新的图像处理器到这个类中
        IMAGE_PROCESSOR_MAPPING.register(config_class, image_processor_class, exist_ok=exist_ok)
```