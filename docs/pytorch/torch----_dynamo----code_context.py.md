# `.\pytorch\torch\_dynamo\code_context.py`

```
# mypy: allow-untyped-defs
# 导入types模块，用于处理Python对象的类型信息
import types

# 从当前包中导入ExactWeakKeyDictionary工具类
from .utils import ExactWeakKeyDictionary

# 定义一个名为CodeContextDict的类，用于管理代码上下文的字典
class CodeContextDict:
    def __init__(self):
        # 初始化一个ExactWeakKeyDictionary实例，用于存储代码上下文信息
        self.code_context = ExactWeakKeyDictionary()

    # 检查给定的code对象是否存在于code_context字典中
    def has_context(self, code: types.CodeType):
        return code in self.code_context

    # 获取给定code对象的上下文信息，如果不存在则创建一个空字典并返回
    def get_context(self, code: types.CodeType):
        ctx = self.code_context.get(code)
        if ctx is None:
            ctx = {}
            self.code_context[code] = ctx
        return ctx

    # 弹出给定code对象的上下文信息，并从字典中删除对应的条目
    def pop_context(self, code: types.CodeType):
        ctx = self.get_context(code)
        self.code_context._remove_id(id(code))
        return ctx

    # 清空code_context字典中的所有条目
    def clear(self):
        self.code_context.clear()

# 创建一个全局的CodeContextDict实例，命名为code_context，用于在程序中管理代码的上下文信息
code_context = CodeContextDict()
```