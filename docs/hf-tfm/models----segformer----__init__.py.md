# `.\transformers\models\segformer\__init__.py`

```
# 该文件包含 HuggingFace 团队开发的 SegFormer 模型的相关配置和功能定义
# 版权归 HuggingFace 团队所有，遵循 Apache 2.0 License 开源协议

# 导入所需的类型检查相关模块
from typing import TYPE_CHECKING

# 导入 HuggingFace 团队自定义的工具函数
from ...utils import (
    OptionalDependencyNotAvailable, # 可选依赖缺失异常
    _LazyModule, # 延迟加载模块的工具类
    is_tf_available, # 检查 TensorFlow 是否可用
    is_torch_available, # 检查 PyTorch 是否可用 
    is_vision_available # 检查计算机视觉相关依赖是否可用
)

# 定义待延迟导入的模块结构
_import_structure = {
    "configuration_segformer": ["SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "SegformerConfig", "SegformerOnnxConfig"]
}

# 检查计算机视觉相关依赖是否可用，如果不可用则忽略相关功能的导入
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_segformer"] = ["SegformerFeatureExtractor"]
    _import_structure["image_processing_segformer"] = ["SegformerImageProcessor"]

# 检查 PyTorch 是否可用，如果不可用则忽略相关功能的导入 
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_segformer"] = [
        "SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SegformerDecodeHead",
        "SegformerForImageClassification",
        "SegformerForSemanticSegmentation",
        "SegformerLayer",
        "SegformerModel",
        "SegformerPreTrainedModel",
    ]

# 检查 TensorFlow 是否可用，如果不可用则忽略相关功能的导入
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_segformer"] = [
        "TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSegformerDecodeHead",
        "TFSegformerForImageClassification",
        "TFSegformerForSemanticSegmentation",
        "TFSegformerModel",
        "TFSegformerPreTrainedModel",
    ]

# 如果进行类型检查，导入相关类型定义
if TYPE_CHECKING:
    from .configuration_segformer import SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, SegformerConfig, SegformerOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_segformer import SegformerFeatureExtractor
        from .image_processing_segformer import SegformerImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果条件不满足，则执行以下代码块
    else:
        # 从模块中导入以下模型相关内容
        from .modeling_segformer import (
            SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SegformerDecodeHead,
            SegformerForImageClassification,
            SegformerForSemanticSegmentation,
            SegformerLayer,
            SegformerModel,
            SegformerPreTrainedModel,
        )
    # 尝试执行以下代码块
    try:
        # 如果 TensorFlow 不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果发生 OptionalDependencyNotAvailable 异常，则执行以下代码块
    except OptionalDependencyNotAvailable:
        pass
    # 否则执行以下代码块
    else:
        # 从模块中导入以下 TensorFlow 相关的模型内容
        from .modeling_tf_segformer import (
            TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSegformerDecodeHead,
            TFSegformerForImageClassification,
            TFSegformerForSemanticSegmentation,
            TFSegformerModel,
            TFSegformerPreTrainedModel,
        )
else:
    # 如果之前的条件不满足，则执行下面的代码
    import sys  # 导入 sys 模块来访问系统相关的功能

    # 使用 _LazyModule 类创建一个懒加载模块，并赋值给当前模块的全局模块字典
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```