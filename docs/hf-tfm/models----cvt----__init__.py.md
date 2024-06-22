# `.\models\cvt\__init__.py`

```py
# 版权声明和许可证信息
# 根据 Apache 许可证 2.0 版本进行许可
# 如果不遵从许可证，不得使用该文件
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available 

# 设置模块的导入结构
_import_structure = {"configuration_cvt": ["CVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CvtConfig"]}

# 检查是否 Torch 可用
try:
    if not is_torch_available():
        # 如果 Torch 不可用，则引发 OptionalDependencyNotAvailable 异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果出现 OptionalDependencyNotAvailable 异常，则不进行下面的操作
    pass
else:
    # 如果 Torch 可用，则设置导入结构
    _import_structure["modeling_cvt"] = [
        "CVT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CvtForImageClassification",
        "CvtModel",
        "CvtPreTrainedModel",
    ]

# 检查是否 TensorFlow 可用
try:
    if not is_tf_available():
        # 如果 TensorFlow 不可用，则引发 OptionalDependencyNotAvailable 异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果出现 OptionalDependencyNotAvailable 异常，则不进行下面的操作
    pass
else:
    # 如果 TensorFlow 可用，则设置导入结构
    _import_structure["modeling_tf_cvt"] = [
        "TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFCvtForImageClassification",
        "TFCvtModel",
        "TFCvtPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入相关模块
    from .configuration_cvt import CVT_PRETRAINED_CONFIG_ARCHIVE_MAP, CvtConfig
    try:
        if not is_torch_available():
            # 如果 Torch 不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果出现 OptionalDependencyNotAvailable 异常，则不进行下面的操作
        pass
    else:
        # 如果 Torch 可用，则导入对应模块
        from .modeling_cvt import (
            CVT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CvtForImageClassification,
            CvtModel,
            CvtPreTrainedModel,
        )
    try:
        if not is_tf_available():
            # 如果 TensorFlow 不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果出现 OptionalDependencyNotAvailable 异常，则不进行下面的操作
        pass
    else:
        # 如果 TensorFlow 可用，则导入对应模块
        from .modeling_tf_cvt import (
            TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCvtForImageClassification,
            TFCvtModel,
            TFCvtPreTrainedModel,
        )
# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys
    # 设置当前模块为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```