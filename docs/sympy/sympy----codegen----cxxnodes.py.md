# `D:\src\scipysrc\sympy\sympy\codegen\cxxnodes.py`

```
"""
AST nodes specific to C++.
"""

# 导入 sympy.codegen.ast 模块中的 Attribute, String, Token, Type, none 符号
from sympy.codegen.ast import Attribute, String, Token, Type, none

# 定义一个名为 using 的类，继承自 Token 类，表示 C++ 中的 'using' 语句
class using(Token):
    """ Represents a 'using' statement in C++ """
    # 使用 __slots__ 定义实例的属性，_fields 定义类的字段
    __slots__ = _fields = ('type', 'alias')
    # 默认的属性值，alias 默认为 none
    defaults = {'alias': none}
    # _construct_type 属性构造函数类型，_construct_alias 属性构造函数别名
    _construct_type = Type
    _construct_alias = String

# 创建一个名为 constexpr 的 Attribute 对象
constexpr = Attribute('constexpr')
```