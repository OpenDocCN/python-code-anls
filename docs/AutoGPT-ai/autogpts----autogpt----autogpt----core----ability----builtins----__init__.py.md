# `.\AutoGPT\autogpts\autogpt\autogpt\core\ability\builtins\__init__.py`

```py
# 从指定路径导入模块中的类或函数
from autogpt.core.ability.builtins.create_new_ability import CreateNewAbility
from autogpt.core.ability.builtins.query_language_model import QueryLanguageModel

# 定义一个包含内置能力名称和对应类的字典
BUILTIN_ABILITIES = {
    QueryLanguageModel.name(): QueryLanguageModel,
}

# 定义一个包含需要导出的变量名的列表
__all__ = [
    "BUILTIN_ABILITIES",
    "CreateNewAbility",
    "QueryLanguageModel",
]
```