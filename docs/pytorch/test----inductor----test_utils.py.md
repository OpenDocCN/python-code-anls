# `.\pytorch\test\inductor\test_utils.py`

```
# Owner(s): ["module: inductor"]

# 导入符号运算库中的符号对象
from sympy import Symbol

# 导入自定义测试框架的测试用例类和函数
from torch._inductor.test_case import run_tests, TestCase
# 导入自定义工具函数中的符号替换函数
from torch._inductor.utils import sympy_subs

# 定义测试类 TestUtils，继承自 TestCase
class TestUtils(TestCase):
    # 定义测试方法 testSympySubs
    def testSympySubs(self):
        # 测试用例：替换单一符号对象
        # 创建符号对象 'x'
        expr = Symbol("x")
        # 调用符号替换函数，将 'x' 替换为 'y'
        result = sympy_subs(expr, {expr: "y"})
        # 断言替换后的符号名称为 'y'
        self.assertEqual(result.name, "y")
        # 断言替换后的符号是否为整数及非负数的属性保持不变
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

        # 测试用例：替换带有整数和非负属性的符号对象
        # 创建整数且非负属性的符号对象 'x'
        expr = Symbol("x", integer=True, nonnegative=False)
        # 调用符号替换函数，将 'x' 替换为 'y'
        result = sympy_subs(expr, {expr: "y"})
        # 断言替换后的符号名称为 'y'
        self.assertEqual(result.name, "y")
        # 断言替换后的符号整数属性为 True，非负属性为 False
        self.assertEqual(result.is_integer, True)
        self.assertEqual(result.is_nonnegative, False)

        # 测试用例：无效的替换
        # 创建整数属性的符号对象 'x'
        expr = Symbol("x", integer=True)
        # 尝试用不同的符号对象 'x' 替换 'x'
        result = sympy_subs(expr, {Symbol("x"): Symbol("y")})
        # 断言替换后的符号名称仍为 'x'
        self.assertEqual(result.name, "x")

        # 测试用例：有效的替换（属性匹配）
        # 创建整数属性的符号对象 'x'
        expr = Symbol("x", integer=True)
        # 用具有相同整数属性的符号对象 'y' 替换 'x'
        result = sympy_subs(expr, {Symbol("x", integer=True): Symbol("y")})
        # 断言替换后的符号名称为 'y'
        self.assertEqual(result.name, "y")

        # 测试用例：无效的替换（属性不匹配）
        # 创建整数属性为 None 的符号对象 'x'
        expr = Symbol("x", integer=None)
        # 尝试用整数属性为 False 的符号对象 'x' 替换 'x'
        result = sympy_subs(expr, {Symbol("x", integer=False): Symbol("y")})
        # 断言替换后的符号名称仍为 'x'
        self.assertEqual(result.name, "x")

        # 测试用例：替换不能是字符串
        # 断言在替换中传递字符串时引发 AssertionError 异常
        self.assertRaises(AssertionError, sympy_subs, expr, {"x": "y"})

        # 测试用例：替换可以是表达式
        # 创建符号对象 'x'，并对其应用绝对值函数
        expr = Symbol("x")
        expr = abs(expr)
        # 断言绝对值函数后的符号整数属性为 None，非负属性为 None
        self.assertEqual(expr.is_integer, None)
        self.assertEqual(expr.is_nonnegative, None)
        # 替换表达式 abs(x) 为符号对象 'y'
        result = sympy_subs(expr, {expr: Symbol("y")})
        # 断言替换后的符号名称为 'y'
        self.assertEqual(result.name, "y")
        # 断言替换后的符号整数属性为 None，非负属性为 None
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

# 如果该脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```