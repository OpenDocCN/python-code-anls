# `.\models\poolformer\__init__.py`

```py
# 版权声明和许可信息，指明代码版权及使用许可
# 版权所有 2022 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的条款，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关具体语言的权限，请参阅许可证。

# 导入类型检查
from typing import TYPE_CHECKING

# 从 utils 模块导入相关依赖和工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构的字典
_import_structure = {
    "configuration_poolformer": [
        "POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PoolFormerConfig",
        "PoolFormerOnnxConfig",
    ]
}

# 检查视觉处理库是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 feature_extraction_poolformer 模块的导入项
    _import_structure["feature_extraction_poolformer"] = ["PoolFormerFeatureExtractor"]
    # 添加 image_processing_poolformer 模块的导入项
    _import_structure["image_processing_poolformer"] = ["PoolFormerImageProcessor"]

# 检查 PyTorch 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_poolformer 模块的导入项
    _import_structure["modeling_poolformer"] = [
        "POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PoolFormerForImageClassification",
        "PoolFormerModel",
        "PoolFormerPreTrainedModel",
    ]

# 如果当前环境支持类型检查（如 Mypy），执行以下导入
if TYPE_CHECKING:
    # 从 configuration_poolformer 模块导入相关类和常量
    from .configuration_poolformer import (
        POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PoolFormerConfig,
        PoolFormerOnnxConfig,
    )

    # 检查视觉处理库是否可用，若可用则导入 feature_extraction_poolformer 和 image_processing_poolformer 模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_poolformer import PoolFormerFeatureExtractor
        from .image_processing_poolformer import PoolFormerImageProcessor

    # 检查 PyTorch 是否可用，若可用则导入 modeling_poolformer 模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_poolformer import (
            POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            PoolFormerForImageClassification,
            PoolFormerModel,
            PoolFormerPreTrainedModel,
        )

# 如果当前环境不支持类型检查，则使用懒加载模块代理导入结构
else:
    import sys

    # 将当前模块替换为懒加载模块的实例，该实例根据需要延迟加载具体模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```