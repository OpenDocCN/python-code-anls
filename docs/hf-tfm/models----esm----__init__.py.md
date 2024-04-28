# `.\models\esm\__init__.py`

```
# 版权声明
# 版权所有2022年Facebook和HuggingFace团队。保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）许可
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或以书面形式同意，软件
# 根据许可证分发，以“按原样”基础分发，
# 没有任何种类的明示或暗示的担保或条件。
# 有关特定语言规定权限和
# 许可下的限制。
from typing import TYPE_CHECKING
# 从自定义的utils模块中导入OptionalDependencyNotAvailable，_LazyModule，is_tf_available，is_torch_available
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_esm": ["ESM_PRETRAINED_CONFIG_ARCHIVE_MAP", "EsmConfig"],
    "tokenization_esm": ["EsmTokenizer"],
}

# 检查是否torch可用，如果不可用，则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则添加以下模块到导入结构中
    _import_structure["modeling_esm"] = [
        "ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "EsmForMaskedLM",
        "EsmForSequenceClassification",
        "EsmForTokenClassification",
        "EsmModel",
        "EsmPreTrainedModel",
    ]
    # 添加modeling_esmfold模块到导入结构中
    _import_structure["modeling_esmfold"] = ["EsmForProteinFolding", "EsmFoldPreTrainedModel"]

# 检查是否tf可用，如果不可用，则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果tf可用，则添加以下模块到导入结构中
    _import_structure["modeling_tf_esm"] = [
        "TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFEsmForMaskedLM",
        "TFEsmForSequenceClassification",
        "TFEsmForTokenClassification",
        "TFEsmModel",
        "TFEsmPreTrainedModel",
    ]

# 如果是类型检查，导入相关模块
if TYPE_CHECKING:
    from .configuration_esm import ESM_PRETRAINED_CONFIG_ARCHIVE_MAP, EsmConfig
    from .tokenization_esm import EsmTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_esm import (
            ESM_PRETRAINED_MODEL_ARCHIVE_LIST,
            EsmForMaskedLM,
            EsmForSequenceClassification,
            EsmForTokenClassification,
            EsmModel,
            EsmPreTrainedModel,
        )
        from .modeling_esmfold import EsmFoldPreTrainedModel, EsmForProteinFolding

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_esm import (
            TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFEsmForMaskedLM,
            TFEsmForSequenceClassification,
            TFEsmForTokenClassification,
            TFEsmModel,
            TFEsmPreTrainedModel,
        )
# 如果不是类型检查，使用LazyModule懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```