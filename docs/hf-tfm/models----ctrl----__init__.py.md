# `.\models\ctrl\__init__.py`

```py
# 版权声明和许可信息
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下链接获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 如果适用法律要求或书面同意，软件将根据“原样”基础分发，不附带任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
#

from typing import TYPE_CHECKING

# 引入自定义工具模块和依赖检查函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_ctrl": ["CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CTRLConfig"],
    "tokenization_ctrl": ["CTRLTokenizer"],
}

# 检查是否可用 Torch，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 Torch 模块到导入结构
    _import_structure["modeling_ctrl"] = [
        "CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CTRLForSequenceClassification",
        "CTRLLMHeadModel",
        "CTRLModel",
        "CTRLPreTrainedModel",
    ]

# 检查是否可用 TensorFlow，如果不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 TensorFlow 模块到导入结构
    _import_structure["modeling_tf_ctrl"] = [
        "TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFCTRLForSequenceClassification",
        "TFCTRLLMHeadModel",
        "TFCTRLModel",
        "TFCTRLPreTrainedModel",
    ]

# 如果是类型检查阶段，引入具体模块的类型和常量
if TYPE_CHECKING:
    from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
    from .tokenization_ctrl import CTRLTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_ctrl import (
            CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
            CTRLForSequenceClassification,
            CTRLLMHeadModel,
            CTRLModel,
            CTRLPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_ctrl import (
            TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCTRLForSequenceClassification,
            TFCTRLLMHeadModel,
            TFCTRLModel,
            TFCTRLPreTrainedModel,
        )

# 如果不是类型检查阶段，使用延迟加载模块
else:
    import sys

    # 将当前模块替换为延迟加载模块对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```