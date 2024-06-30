# `D:\src\scipysrc\sympy\sympy\strategies\util.py`

```
# 从 sympy.core.basic 模块导入 Basic 类
from sympy.core.basic import Basic

# 复制 Basic 类的 __new__ 方法引用给 new 变量
new = Basic.__new__

# 定义一个函数 assoc，用于在字典 d 中关联键 k 和值 v，并返回副本
def assoc(d, k, v):
    d = d.copy()  # 复制字典 d，以便修改不影响原字典
    d[k] = v  # 将键 k 关联的值设置为 v
    return d  # 返回修改后的字典副本

# 定义一个字典 basic_fns，包含不同的键值对：
# - 'op': type，映射到 type 函数
# - 'new': Basic.__new__，映射到 Basic 类的 __new__ 方法引用
# - 'leaf': 一个 lambda 函数，用于检查 x 是否不是 Basic 类型或者是 Atom 类型的实例
# - 'children': 一个 lambda 函数，用于获取 x 的参数列表 args
basic_fns = {'op': type,
             'new': Basic.__new__,
             'leaf': lambda x: not isinstance(x, Basic) or x.is_Atom,
             'children': lambda x: x.args}

# 调用 assoc 函数，将 'new' 键的值修改为一个新的 lambda 函数，
# 这个 lambda 函数接受 op 和 args 作为参数，并调用 op(*args) 来创建新的 Basic 对象
expr_fns = assoc(basic_fns, 'new', lambda op, *args: op(*args))
```