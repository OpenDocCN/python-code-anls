# `.\transformers\models\mra\__init__.py`

```py
# flake8: noqa
# 禁用 flake8 对代码的语法检测
# 在此模块中，无法忽略 "F401 '...' imported but unused" 警告，但可以保留其他警告。因此，完全不对这个模块进行检查。

# 版权声明
# 2023年 HuggingFace 团队保留所有权利
#
# 根据 Apache 许可证，版本 2.0 授权
# 除非符合许可证，否则不能使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律要求或经书面同意，否则依据本许可证分发的软件是在“现状”基础上分发的，
# 没有任何明示或暗示的保证或条件，包括但不限于适销性或特定用途的保证。
# 有关特定语言的权限和限制，请参阅许可证
from typing import TYPE_CHECKING
# 导入类型检查模块

# rely on isort to merge the imports
# 依赖 isort 来合并导入语句
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available
# 从工具包中导入必要的依赖项：OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义导入结构字典
_import_structure = {"configuration_mra": ["MRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "MraConfig"]}

try:
    # 检查是否存在 torch 库，如果不存在则引发异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加下面的模块到导入结构中
    _import_structure["modeling_mra"] = [
        "MRA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MraForMaskedLM",
        "MraForMultipleChoice",
        "MraForQuestionAnswering",
        "MraForSequenceClassification",
        "MraForTokenClassification",
        "MraLayer",
        "MraModel",
        "MraPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从配置模块中导入特定名称
    from .configuration_mra import MRA_PRETRAINED_CONFIG_ARCHIVE_MAP, MraConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从建模模块中导入特定名称
        from .modeling_mra import (
            MRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            MraForMaskedLM,
            MraForMultipleChoice,
            MraForQuestionAnswering,
            MraForSequenceClassification,
            MraForTokenClassification,
            MraLayer,
            MraModel,
            MraPreTrainedModel,
        )
else:
    # 如果不是类型检查模式，导入 sys 模块
    import sys

    # 将当前模块设为 lazy 模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```