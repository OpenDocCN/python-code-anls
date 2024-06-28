# `.\models\esm\__init__.py`

```py
# 版权声明和许可证信息
#
# 版权所有 2022 年 Facebook 和 HuggingFace 团队。保留所有权利。
# 
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 请查阅许可证获取具体语言的权限和限制。
from typing import TYPE_CHECKING

# 从相对路径导入必要的模块和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_esm": ["ESM_PRETRAINED_CONFIG_ARCHIVE_MAP", "EsmConfig"],
    "tokenization_esm": ["EsmTokenizer"],
}

# 检查是否 Torch 可用，否则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，导入 Torch 版本的 ESM 模块
    _import_structure["modeling_esm"] = [
        "ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "EsmForMaskedLM",
        "EsmForSequenceClassification",
        "EsmForTokenClassification",
        "EsmModel",
        "EsmPreTrainedModel",
    ]
    _import_structure["modeling_esmfold"] = ["EsmForProteinFolding", "EsmFoldPreTrainedModel"]

# 检查是否 TensorFlow 可用，否则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，导入 TensorFlow 版本的 ESM 模块
    _import_structure["modeling_tf_esm"] = [
        "TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFEsmForMaskedLM",
        "TFEsmForSequenceClassification",
        "TFEsmForTokenClassification",
        "TFEsmModel",
        "TFEsmPreTrainedModel",
    ]

# 如果是类型检查阶段，导入必要的类型
if TYPE_CHECKING:
    from .configuration_esm import ESM_PRETRAINED_CONFIG_ARCHIVE_MAP, EsmConfig
    from .tokenization_esm import EsmTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 Torch 版本的 ESM 模块（仅用于类型检查）
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
        # 导入 TensorFlow 版本的 ESM 模块（仅用于类型检查）
        from .modeling_tf_esm import (
            TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFEsmForMaskedLM,
            TFEsmForSequenceClassification,
            TFEsmForTokenClassification,
            TFEsmModel,
            TFEsmPreTrainedModel,
        )

# 如果不是类型检查阶段，将当前模块设为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```