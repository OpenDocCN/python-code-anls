# `.\models\bros\__init__.py`

```py
# 版权声明及许可证声明，指明代码作者及授权条款
# 版权所有 2023-present NAVER Corp，Microsoft Research Asia LayoutLM Team 作者和 HuggingFace Inc. 团队。
# 根据 Apache 许可证 2.0 版本发布，除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，本软件按"原样"分发，不附带任何明示或暗示的保证或条件。
# 有关详细的权利和限制，请参阅许可证。

# 引入类型检查模块
from typing import TYPE_CHECKING

# 从内部模块引入相关依赖
# utils 模块来自上层目录的 "..."，这里是一个占位符
# OptionalDependencyNotAvailable 是一个自定义的异常类
# _LazyModule 是一个懒加载模块类
# is_tokenizers_available 和 is_torch_available 是检查依赖是否可用的函数

# 定义导入结构，字典包含不同模块和它们的导入项
_import_structure = {
    "configuration_bros": ["BROS_PRETRAINED_CONFIG_ARCHIVE_MAP", "BrosConfig"],
}

# 检查 tokenizers 是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 processing_bros 模块到导入结构中
    _import_structure["processing_bros"] = ["BrosProcessor"]

# 检查 torch 是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_bros 模块到导入结构中，并列出具体导入项
    _import_structure["modeling_bros"] = [
        "BROS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BrosPreTrainedModel",
        "BrosModel",
        "BrosForTokenClassification",
        "BrosSpadeEEForTokenClassification",
        "BrosSpadeELForTokenClassification",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 configuration_bros 模块导入相关内容
    from .configuration_bros import BROS_PRETRAINED_CONFIG_ARCHIVE_MAP, BrosConfig

    # 检查 tokenizers 是否可用，如果可用则从 processing_bros 模块导入 BrosProcessor
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .processing_bros import BrosProcessor

    # 检查 torch 是否可用，如果可用则从 modeling_bros 模块导入多个类和常量
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bros import (
            BROS_PRETRAINED_MODEL_ARCHIVE_LIST,
            BrosForTokenClassification,
            BrosModel,
            BrosPreTrainedModel,
            BrosSpadeEEForTokenClassification,
            BrosSpadeELForTokenClassification,
        )

# 如果不在类型检查模式下
else:
    import sys

    # 将当前模块注册为懒加载模块
    # _LazyModule 是一个辅助类，用于在需要时加载模块
    # __name__ 是当前模块的名称
    # globals()["__file__"] 返回当前文件的路径
    # _import_structure 是定义好的导入结构
    # module_spec=__spec__ 指定模块的规范

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```