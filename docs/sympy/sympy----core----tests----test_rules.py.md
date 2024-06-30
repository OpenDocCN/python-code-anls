# `D:\src\scipysrc\sympy\sympy\core\tests\test_rules.py`

```
# 从 sympy.core.rules 模块导入 Transform 类
from sympy.core.rules import Transform
# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises

# 定义测试函数 test_Transform
def test_Transform():
    # 创建 Transform 对象 add1，使用 lambda 函数定义转换规则和判断条件
    add1 = Transform(lambda x: x + 1, lambda x: x % 2 == 1)
    # 断言在 add1 中索引 1 的结果为 2
    assert add1[1] == 2
    # 断言 add1 中包含值为 1 的键
    assert (1 in add1) is True
    # 断言调用 add1 的 get 方法获取键 1 的结果为 2
    assert add1.get(1) == 2

    # 使用 raises 函数断言 KeyError 异常在 lambda 函数中被触发
    raises(KeyError, lambda: add1[2])
    # 断言 add1 中不包含值为 2 的键
    assert (2 in add1) is False
    # 断言调用 add1 的 get 方法获取键 2 返回 None
    assert add1.get(2) is None
```