# `.\models\auto\image_processing_auto.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，指明代码版权归 HuggingFace Inc. 团队所有
# 使用 Apache License, Version 2.0 许可协议，详见链接
# 除非法律另有规定或书面同意，否则不得使用本文件
# 详细信息请查看许可协议：http://www.apache.org/licenses/LICENSE-2.0
# 引入 warnings 库，用于发出警告信息
import warnings
# collections 模块中的 OrderedDict 类，用于创建有序字典
from collections import OrderedDict
# typing 模块，用于类型提示
from typing import Dict, Optional, Union

# 从相应模块中导入函数和类
# configuration_utils 模块中的 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# dynamic_module_utils 中的函数，用于从动态模块中获取类
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
# image_processing_utils 中的 ImageProcessingMixin 类
from ...image_processing_utils import ImageProcessingMixin
# utils 中的各种实用函数和常量
from ...utils import CONFIG_NAME, IMAGE_PROCESSOR_NAME, get_file_from_repo, logging
# 从当前包中导入 auto_factory 模块的 _LazyAutoMapping 类
from .auto_factory import _LazyAutoMapping
# 从当前包中导入 configuration_auto 模块中的若干变量和函数
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 IMAGE_PROCESSOR_MAPPING_NAMES 为有序字典
IMAGE_PROCESSOR_MAPPING_NAMES = OrderedDict(
    # 这里原本应该有具体的映射关系，由开发者补充完整
    # 类似 {'module_name': ['extractor1', 'extractor2']}
    # 用于存储映射关系
)

# 使用 _LazyAutoMapping 类创建 IMAGE_PROCESSOR_MAPPING 对象
IMAGE_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, IMAGE_PROCESSOR_MAPPING_NAMES)

# 根据类名从 IMAGE_PROCESSOR_MAPPING_NAMES 中获取对应的处理器类
def image_processor_class_from_name(class_name: str):
    for module_name, extractors in IMAGE_PROCESSOR_MAPPING_NAMES.items():
        # 遍历映射字典，查找匹配的类名
        if class_name in extractors:
            # 将模块名转换为模块的实际名称
            module_name = model_type_to_module_name(module_name)
            # 动态导入相应模块
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                # 返回模块中对应的类对象
                return getattr(module, class_name)
            except AttributeError:
                continue

    # 如果在 IMAGE_PROCESSOR_MAPPING_NAMES 中未找到对应类名，则遍历额外内容
    for _, extractor in IMAGE_PROCESSOR_MAPPING._extra_content.items():
        # 检查额外内容中是否包含与类名匹配的对象
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # 若以上方法均未找到匹配的类名，则从主模块中导入，返回对应的类对象或 None
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None

# 加载预训练模型的图像处理器配置信息
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
    从预训练模型的图像处理器配置中加载图像处理器配置信息。
    """
    # 函数体内容尚未给出，需由开发者补充完整
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
            If `True`, will only try to load the image processor configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the image processor.

    Examples:

    ```
    # Download configuration from huggingface.co and cache.
    image_processor_config = get_image_processor_config("google-bert/bert-base-uncased")
    # This model does not have a image processor config so the result will be an empty dict.
    image_processor_config = get_image_processor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained image processor locally and you can reload its config
    from transformers import AutoTokenizer

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_processor.save_pretrained("image-processor-test")
    image_processor_config = get_image_processor_config("image-processor-test")
    ```
"""
    use_auth_token = kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 参数不为 None，则发出警告，提醒该参数将在 Transformers v5 版本中被移除
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果同时指定了 token 参数和 use_auth_token 参数，则抛出数值错误
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 token 参数设置为 use_auth_token 参数的值
        token = use_auth_token

    # 从指定的预训练模型名或路径中获取配置文件路径
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
    # 如果未能定位到图像处理器配置文件，则记录信息并返回空字典
    if resolved_config_file is None:
        logger.info(
            "Could not locate the image processor configuration file, will try to use the model config instead."
        )
        return {}

    # 打开配置文件并以 UTF-8 编码读取其中的内容，解析为 JSON 格式返回
    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)
class AutoImageProcessor:
    r"""
    This is a generic image processor class that will be instantiated as one of the image processor classes of the
    library when created with the [`AutoImageProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        # 抛出环境错误，阻止直接实例化该类
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
        # 调用全局注册函数，将给定的配置类和图像处理器类注册到映射表中
        IMAGE_PROCESSOR_MAPPING.register(config_class, image_processor_class, exist_ok=exist_ok)
```