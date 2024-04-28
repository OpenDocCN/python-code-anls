# `.\models\deit\__init__.py`

```
# 导入必要的包和模块
from typing import TYPE_CHECKING  # 引入类型提示

from ...utils import (  # 导入自定义的工具模块
    OptionalDependencyNotAvailable,  # 引入可选依赖不可用异常
    _LazyModule,  # 引入延迟加载模块
    is_tf_available,  # 引入判断 TensorFlow 是否可用的函数
    is_torch_available,  # 引入判断 PyTorch 是否可用的函数
    is_vision_available,  # 引入判断 torchvision 是否可用的函数
)


# 定义了一个字典，用于记录待导入的模块和其中的对象/变量
_import_structure = {
    "configuration_deit": ["DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeiTConfig", "DeiTOnnxConfig"]  # 导入 configuration_deit 模块中的多个对象/变量
}

# 检查 torchvision 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable
# 如果可用，则将相关模块和对象/变量添加到 _import_structure 中
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_deit"] = ["DeiTFeatureExtractor"]
    _import_structure["image_processing_deit"] = ["DeiTImageProcessor"]

# 检查 PyTorch 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable
# 如果可用，则将相关模块和对象/变量添加到 _import_structure 中
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_deit"] = [
        "DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DeiTForImageClassification",
        "DeiTForImageClassificationWithTeacher",
        "DeiTForMaskedImageModeling",
        "DeiTModel",
        "DeiTPreTrainedModel",
    ]

# 检查 TensorFlow 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable
# 如果可用，则将相关模块和对象/变量添加到 _import_structure 中
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_deit"] = [
        "TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDeiTForImageClassification",
        "TFDeiTForImageClassificationWithTeacher",
        "TFDeiTForMaskedImageModeling",
        "TFDeiTModel",
        "TFDeiTPreTrainedModel",
    ]


# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入类型提示用到的模块和对象/变量
    from .configuration_deit import DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, DeiTConfig, DeiTOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_deit import DeiTFeatureExtractor
        from .image_processing_deit import DeiTImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_deit import (
            DEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DeiTForImageClassification,
            DeiTForImageClassificationWithTeacher,
            DeiTForMaskedImageModeling,
            DeiTModel,
            DeiTPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_deit import (
            TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDeiTForImageClassification,
            TFDeiTForImageClassificationWithTeacher,
            TFDeiTForMaskedImageModeling,
            TFDeiTModel,
            TFDeiTPreTrainedModel,
        )


以上是给定代码块的注释。
    # 如果发生 OptionalDependencyNotAvailable 异常，则忽略，不做任何处理
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有发生 OptionalDependencyNotAvailable 异常，则执行以下操作
    else:
        # 从当前目录中导入相关模块和类
        from .modeling_tf_deit import (
            TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入 TF DEIT 预训练模型的列表
            TFDeiTForImageClassification,  # 导入用于图像分类的 TF DEIT 模型
            TFDeiTForImageClassificationWithTeacher,  # 导入带有 teacher 的图像分类 TF DEIT 模型
            TFDeiTForMaskedImageModeling,  # 导入用于图像蒙版建模的 TF DEIT 模型
            TFDeiTModel,  # 导入 TF DEIT 模型
            TFDeiTPreTrainedModel,  # 导入 TF DEIT 预训练模型的基类
        )
# 如果未满足前面 if 语句的条件，则执行下面的代码块
    # 导入 sys 模块
    import sys
    # 将当前模块的名称赋值给 sys.modules 中的 __name__ 键对应的值
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```