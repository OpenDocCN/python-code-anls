# `.\models\cvt\__init__.py`

```
# 版权声明和许可证信息
#
# 根据 Apache 许可证版本 2.0 授权使用此文件
# 除非遵守许可证，否则不得使用此文件
# 可以在以下链接处获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果法律要求或书面同意，本软件根据“原样”分发，无任何明示或暗示的担保或条件。
# 请查阅许可证获取更多信息。

from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义导入结构
_import_structure = {"configuration_cvt": ["CVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CvtConfig"]}

# 检查是否 Torch 可用，否则抛出可选依赖项不可用的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加以下模块到导入结构中
    _import_structure["modeling_cvt"] = [
        "CVT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CvtForImageClassification",
        "CvtModel",
        "CvtPreTrainedModel",
    ]

# 检查是否 TensorFlow 可用，否则抛出可选依赖项不可用的异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，则添加以下模块到导入结构中
    _import_structure["modeling_tf_cvt"] = [
        "TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFCvtForImageClassification",
        "TFCvtModel",
        "TFCvtPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置和模型相关模块（Torch 或 TensorFlow 可能会存在）
    from .configuration_cvt import CVT_PRETRAINED_CONFIG_ARCHIVE_MAP, CvtConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_cvt import (
            CVT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CvtForImageClassification,
            CvtModel,
            CvtPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_cvt import (
            TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCvtForImageClassification,
            TFCvtModel,
            TFCvtPreTrainedModel,
        )

# 非类型检查阶段
else:
    import sys

    # 将当前模块替换为懒加载模块，以便动态导入指定模块结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```