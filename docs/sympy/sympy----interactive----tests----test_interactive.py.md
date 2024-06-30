# `D:\src\scipysrc\sympy\sympy\interactive\tests\test_interactive.py`

```
# 从 sympy.interactive.session 模块导入 int_to_Integer 函数
from sympy.interactive.session import int_to_Integer

# 定义测试函数 test_int_to_Integer
def test_int_to_Integer():
    # 断言 int_to_Integer 函数对字符串 "1 + 2.2 + 0x3 + 40" 的返回值是否等于预期值
    assert int_to_Integer("1 + 2.2 + 0x3 + 40") == 'Integer (1 )+2.2 +Integer (0x3 )+Integer (40 )'
    # 断言 int_to_Integer 函数对字符串 "0b101" 的返回值是否等于预期值
    assert int_to_Integer("0b101") == 'Integer (0b101 )'
    # 断言 int_to_Integer 函数对字符串 "ab1 + 1 + '1 + 2'" 的返回值是否等于预期值
    assert int_to_Integer("ab1 + 1 + '1 + 2'") == "ab1 +Integer (1 )+'1 + 2'"
    # 断言 int_to_Integer 函数对字符串 "(2 + \n3)" 的返回值是否等于预期值
    assert int_to_Integer("(2 + \n3)") == '(Integer (2 )+\nInteger (3 ))'
    # 断言 int_to_Integer 函数对字符串 "2 + 2.0 + 2j + 2e-10" 的返回值是否等于预期值
    assert int_to_Integer("2 + 2.0 + 2j + 2e-10") == 'Integer (2 )+2.0 +2j +2e-10 '
```