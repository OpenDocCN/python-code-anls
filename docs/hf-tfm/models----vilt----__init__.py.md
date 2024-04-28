# `.\transformers\models\vilt\__init__.py`

```
# 版权声明及授权许可
# 版权声明
# 根据 Apache 许可证 2.0 版 (下文简称“许可证”) 许可，除非符合许可证的约定，否则不得使用本文件。
# 你可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非根据适用法律或书面同意要求，否则依据“如实”原则发布的软件是基于"原样"的基础，没有明示或暗示的任何保证或条件。
# 请参阅许可证以了解特定于语言的权限和限制

# 导入类型提示模块
from typing import TYPE_CHECKING
# 导入可选依赖未安装异常类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 设置导入结构
_import_structure = {"configuration_vilt": ["VILT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViltConfig"]}

# 尝试检测视觉库是否可用，若不可用则抛出可选依赖未安装异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若视觉库可用，则添加导入结构
    _import_structure["feature_extraction_vilt"] = ["ViltFeatureExtractor"]
    _import_structure["image_processing_vilt"] = ["ViltImageProcessor"]
    _import_structure["processing_vilt"] = ["ViltProcessor"]

# 尝试检测torch库是否可用，若不可用则抛出可选依赖未安装异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch库可用，则添加导入结构
    _import_structure["modeling_vilt"] = [
        "VILT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViltForImageAndTextRetrieval",
        "ViltForImagesAndTextClassification",
        "ViltForTokenClassification",
        "ViltForMaskedLM",
        "ViltForQuestionAnswering",
        "ViltLayer",
        "ViltModel",
        "ViltPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关内容
    from .configuration_vilt import VILT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViltConfig

    try:
        # 尝试检测视觉库是否可用，若不可用则抛出可选依赖未安装异常
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若视觉库可用，则导入相关内容
        from .feature_extraction_vilt import ViltFeatureExtractor
        from .image_processing_vilt import ViltImageProcessor
        from .processing_vilt import ViltProcessor

    try:
        # 尝试检测torch库是否可用，若不可用则抛出可选依赖未安装异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若torch库可用，则导入相关内容
        from .modeling_vilt import (
            VILT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViltForImageAndTextRetrieval,
            ViltForImagesAndTextClassification,
            ViltForMaskedLM,
            ViltForQuestionAnswering,
            ViltForTokenClassification,
            ViltLayer,
            ViltModel,
            ViltPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys
    # 为当前模块创建懒加载模块对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```