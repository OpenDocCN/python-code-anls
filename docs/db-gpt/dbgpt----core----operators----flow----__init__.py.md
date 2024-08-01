# `.\DB-GPT-src\dbgpt\core\operators\flow\__init__.py`

```py
"""Flow operators package."""
# 导入流操作符模块的相关内容

from .composer_operator import (  # noqa: F401
    ConversationComposerOperator,
    PromptFormatDictBuilderOperator,
)
# 从composer_operator模块导入ConversationComposerOperator和PromptFormatDictBuilderOperator类
# 使用 noqa: F401 告诉 linter 忽略“未使用”警告

from .dict_operator import MergeStringToDictOperator  # noqa: F401
# 从dict_operator模块导入MergeStringToDictOperator类
# 使用 noqa: F401 告诉 linter 忽略“未使用”警告

__ALL__ = [
    "ConversationComposerOperator",
    "PromptFormatDictBuilderOperator",
    "MergeStringToDictOperator",
]
# 设置 __ALL__ 列表，定义了在使用 from package import * 时应该导入的符号名的列表
```