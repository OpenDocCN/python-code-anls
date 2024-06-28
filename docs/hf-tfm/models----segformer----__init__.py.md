# `.\models\segformer\__init__.py`

```py
# 导入必要的模块和函数声明
from typing import TYPE_CHECKING  # 导入类型检查模块

# 导入必要的依赖项和函数
from ...utils import (
    OptionalDependencyNotAvailable,  # 导入自定义的可选依赖未安装异常
    _LazyModule,  # 导入自定义的懒加载模块
    is_tf_available,  # 导入检查 TensorFlow 是否可用的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
    is_vision_available,  # 导入检查视觉处理模块是否可用的函数
)

# 定义模块的导入结构字典
_import_structure = {
    "configuration_segformer": ["SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "SegformerConfig", "SegformerOnnxConfig"]
}

# 检查视觉处理模块是否可用，若不可用则抛出自定义的可选依赖未安装异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若视觉处理模块可用，则导入相应的特征提取和图像处理函数
    _import_structure["feature_extraction_segformer"] = ["SegformerFeatureExtractor"]
    _import_structure["image_processing_segformer"] = ["SegformerImageProcessor"]

# 检查 PyTorch 是否可用，若不可用则抛出自定义的可选依赖未安装异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 PyTorch 可用，则导入相应的 Segformer 模型组件
    _import_structure["modeling_segformer"] = [
        "SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SegformerDecodeHead",
        "SegformerForImageClassification",
        "SegformerForSemanticSegmentation",
        "SegformerLayer",
        "SegformerModel",
        "SegformerPreTrainedModel",
    ]

# 检查 TensorFlow 是否可用，若不可用则抛出自定义的可选依赖未安装异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 TensorFlow 可用，则导入相应的 TensorFlow Segformer 模型组件
    _import_structure["modeling_tf_segformer"] = [
        "TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSegformerDecodeHead",
        "TFSegformerForImageClassification",
        "TFSegformerForSemanticSegmentation",
        "TFSegformerModel",
        "TFSegformerPreTrainedModel",
    ]

# 如果是类型检查阶段，则进一步导入类型相关的模块和函数
if TYPE_CHECKING:
    from .configuration_segformer import SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, SegformerConfig, SegformerOnnxConfig

    # 检查视觉处理模块是否可用，在类型检查阶段导入相应的特征提取和图像处理函数
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_segformer import SegformerFeatureExtractor
        from .image_processing_segformer import SegformerImageProcessor

    # 检查 PyTorch 是否可用，在类型检查阶段导入相关的模型组件
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不是 TensorFlow 环境，则导入本地的 Segformer 模型相关模块
    else:
        from .modeling_segformer import (
            SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SegformerDecodeHead,
            SegformerForImageClassification,
            SegformerForSemanticSegmentation,
            SegformerLayer,
            SegformerModel,
            SegformerPreTrainedModel,
        )
    try:
        # 检查当前环境是否可用 TensorFlow
        if not is_tf_available():
            # 如果 TensorFlow 不可用，则抛出 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果发生 OptionalDependencyNotAvailable 异常，不做任何处理，继续执行
        pass
    else:
        # 如果 TensorFlow 可用，则导入 TensorFlow 版本的 Segformer 模型相关模块
        from .modeling_tf_segformer import (
            TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSegformerDecodeHead,
            TFSegformerForImageClassification,
            TFSegformerForSemanticSegmentation,
            TFSegformerModel,
            TFSegformerPreTrainedModel,
        )
else:
    # 如果不满足前面的条件，即不是第一次导入模块时执行的分支
    import sys
    # 导入 sys 模块，用于操作 Python 解释器相关的功能

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 封装
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # 创建一个 _LazyModule 对象并将其赋值给当前模块的键名，设置模块的相关属性
```