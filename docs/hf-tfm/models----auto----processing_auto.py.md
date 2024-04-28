# `.\transformers\models\auto\processing_auto.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
""" AutoProcessor class."""
# 导入模块
import importlib
# 导入模块
import inspect
# 导入模块
import json
# 导入模块
import os
# 导入模块
import warnings
# 导入模块
from collections import OrderedDict

# 导入模块
from ...configuration_utils import PretrainedConfig
# 导入模块
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
# 导入模块
from ...feature_extraction_utils import FeatureExtractionMixin
# 导入模块
from ...image_processing_utils import ImageProcessingMixin
# 导入模块
from ...processing_utils import ProcessorMixin
# 导入模块
from ...tokenization_utils import TOKENIZER_CONFIG_FILE
# 导入模块
from ...utils import FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME, get_file_from_repo, logging
# 导入模块
from .auto_factory import _LazyAutoMapping
# 导入模块
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)
# 导入模块
from .feature_extraction_auto import AutoFeatureExtractor
# 导入模块
from .image_processing_auto import AutoImageProcessor
# 导入模块
from .tokenization_auto import AutoTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义处理器映射名称的有序字典
PROCESSOR_MAPPING_NAMES = OrderedDict(
    # 创建一个包含处理器名称和对应处理器类的元组列表
    [
        ("align", "AlignProcessor"),  # 处理器名称为"align"，对应处理器类为"AlignProcessor"
        ("altclip", "AltCLIPProcessor"),  # 处理器名称为"altclip"，对应处理器类为"AltCLIPProcessor"
        ("bark", "BarkProcessor"),  # 处理器名称为"bark"，对应处理器类为"BarkProcessor"
        # 其余元组依此类推，每个元组包含处理器名称和对应处理器类
    ]
# 导入模块
)

# 使用_LazyAutoMapping函数将CONFIG_MAPPING_NAMES和PROCESSOR_MAPPING_NAMES映射为PROCESSOR_MAPPING
PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)


# 根据类名获取对应的处理器类
def processor_class_from_name(class_name: str):
    # 遍历PROCESSOR_MAPPING_NAMES中的模块名和处理器列表
    for module_name, processors in PROCESSOR_MAPPING_NAMES.items():
        # 如果class_name在processors列表中
        if class_name in processors:
            # 将模块名转换为模型类型的模块名
            module_name = model_type_to_module_name(module_name)

            # 动态导入模块
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                # 返回模块中的类对象
                return getattr(module, class_name)
            except AttributeError:
                continue

    # 遍历PROCESSOR_MAPPING._extra_content中的处理器对象
    for processor in PROCESSOR_MAPPING._extra_content.values():
        # 如果处理器对象的__name__属性与class_name相同
        if getattr(processor, "__name__", None) == class_name:
            # 返回该处理器对象
            return processor

    # 如果未找到对应的类，检查是否因为缺少依赖而未找到，如果在主初始化文件中找到了对应的类，则返回该类
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    # 未找到对应的类，返回None
    return None


# 自动处理器类
class AutoProcessor:
    r"""
    This is a generic processor class that will be instantiated as one of the processor classes of the library when
    created with the [`AutoProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    # 初始化方法，抛出EnvironmentError异常
    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    # 类方法，用于注册新的处理器类
    @classmethod
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
    @staticmethod
    def register(config_class, processor_class, exist_ok=False):
        """
        Register a new processor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            processor_class ([`FeatureExtractorMixin`]): The processor to register.
        """
        # 调用PROCESSOR_MAPPING的register方法注册新的处理器类
        PROCESSOR_MAPPING.register(config_class, processor_class, exist_ok=exist_ok)
```