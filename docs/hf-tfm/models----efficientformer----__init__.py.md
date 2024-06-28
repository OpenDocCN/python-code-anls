# `.\models\efficientformer\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_efficientformer": [
        "EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EfficientFormerConfig",
    ]
}

# 检查视觉处理模块是否可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将视觉处理模块加入导入结构
    _import_structure["image_processing_efficientformer"] = ["EfficientFormerImageProcessor"]

# 检查是否Torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将Torch的模型相关类加入导入结构
    _import_structure["modeling_efficientformer"] = [
        "EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "EfficientFormerForImageClassification",
        "EfficientFormerForImageClassificationWithTeacher",
        "EfficientFormerModel",
        "EfficientFormerPreTrainedModel",
    ]

# 检查是否TensorFlow可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将TensorFlow的模型相关类加入导入结构
    _import_structure["modeling_tf_efficientformer"] = [
        "TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFEfficientFormerForImageClassification",
        "TFEfficientFormerForImageClassificationWithTeacher",
        "TFEfficientFormerModel",
        "TFEfficientFormerPreTrainedModel",
    ]

# 若是类型检查环境，则导入必要的类型和类
if TYPE_CHECKING:
    from .configuration_efficientformer import EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, EfficientFormerConfig

    # 检查视觉处理模块是否可用，若不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入视觉处理模块中的类
        from .image_processing_efficientformer import EfficientFormerImageProcessor

    # 检查是否Torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入Torch模型相关类
        from .modeling_efficientformer import (
            EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            EfficientFormerForImageClassification,
            EfficientFormerForImageClassificationWithTeacher,
            EfficientFormerModel,
            EfficientFormerPreTrainedModel,
        )
    # 尝试检查是否可用 TensorFlow，如果不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果 TensorFlow 不可用，则捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 如果依赖不可用，什么都不做，继续执行后续代码
        pass
    # 如果没有引发异常，则执行以下代码块
    else:
        # 从 TensorFlow 版本的 EfficientFormer 模型导入相关内容
        from .modeling_tf_efficientformer import (
            TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFEfficientFormerForImageClassification,
            TFEfficientFormerForImageClassificationWithTeacher,
            TFEfficientFormerModel,
            TFEfficientFormerPreTrainedModel,
        )
else:
    # 导入sys模块，用于操作Python解释器的相关功能
    import sys

    # 将当前模块注册到sys.modules中，使用_LazyModule进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```