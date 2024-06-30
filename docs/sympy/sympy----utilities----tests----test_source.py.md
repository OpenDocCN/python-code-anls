# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_source.py`

```
# 从 sympy.utilities.source 模块中导入 get_mod_func 和 get_class 函数
from sympy.utilities.source import get_mod_func, get_class

# 定义测试函数 test_get_mod_func，用于测试 get_mod_func 函数
def test_get_mod_func():
    # 断言调用 get_mod_func 函数，传入参数 'sympy.core.basic.Basic'，期望返回 ('sympy.core.basic', 'Basic')
    assert get_mod_func('sympy.core.basic.Basic') == ('sympy.core.basic', 'Basic')

# 定义测试函数 test_get_class，用于测试 get_class 函数
def test_get_class():
    # 调用 get_class 函数，传入参数 'sympy.core.basic.Basic'，返回结果赋给 _basic 变量
    _basic = get_class('sympy.core.basic.Basic')
    # 断言 _basic 对象的 __name__ 属性是否等于 'Basic'
    assert _basic.__name__ == 'Basic'
```