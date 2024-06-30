# `D:\src\scipysrc\sympy\sympy\plotting\intervalmath\tests\test_interval_membership.py`

```
from sympy.core.symbol import Symbol  # 导入 Symbol 类
from sympy.plotting.intervalmath import interval  # 导入 interval 函数
from sympy.plotting.intervalmath.interval_membership import intervalMembership  # 导入 intervalMembership 函数
from sympy.plotting.experimental_lambdify import experimental_lambdify  # 导入 experimental_lambdify 函数
from sympy.testing.pytest import raises  # 导入 raises 函数


def test_creation():
    assert intervalMembership(True, True)  # 测试 intervalMembership 函数返回 True
    raises(TypeError, lambda: intervalMembership(True))  # 测试传递一个参数给 intervalMembership 会引发 TypeError
    raises(TypeError, lambda: intervalMembership(True, True, True))  # 测试传递三个参数给 intervalMembership 会引发 TypeError


def test_getitem():
    a = intervalMembership(True, False)  # 创建一个 intervalMembership 对象
    assert a[0] is True  # 验证第一个元素为 True
    assert a[1] is False  # 验证第二个元素为 False
    raises(IndexError, lambda: a[2])  # 测试访问第三个元素会引发 IndexError


def test_str():
    a = intervalMembership(True, False)  # 创建一个 intervalMembership 对象
    assert str(a) == 'intervalMembership(True, False)'  # 验证 str 方法的输出
    assert repr(a) == 'intervalMembership(True, False)'  # 验证 repr 方法的输出


def test_equivalence():
    a = intervalMembership(True, True)  # 创建两个 intervalMembership 对象
    b = intervalMembership(True, False)
    assert (a == b) is False  # 验证对象不相等
    assert (a != b) is True  # 验证对象不相等

    a = intervalMembership(True, False)  # 创建两个相同的 intervalMembership 对象
    b = intervalMembership(True, False)
    assert (a == b) is True  # 验证对象相等
    assert (a != b) is False  # 验证对象相等


def test_not():
    x = Symbol('x')  # 创建符号变量 x

    r1 = x > -1  # 创建一个不等式
    r2 = x <= -1  # 创建另一个不等式

    i = interval  # 将 interval 函数赋给变量 i

    f1 = experimental_lambdify((x,), r1)  # 使用 experimental_lambdify 创建函数 f1
    f2 = experimental_lambdify((x,), r2)  # 使用 experimental_lambdify 创建函数 f2

    tt = i(-0.1, 0.1, is_valid=True)  # 创建 interval 对象
    tn = i(-0.1, 0.1, is_valid=None)
    tf = i(-0.1, 0.1, is_valid=False)

    assert f1(tt) == ~f2(tt)  # 验证 f1 和 ~f2 的结果相等
    assert f1(tn) == ~f2(tn)
    assert f1(tf) == ~f2(tf)

    nt = i(0.9, 1.1, is_valid=True)
    nn = i(0.9, 1.1, is_valid=None)
    nf = i(0.9, 1.1, is_valid=False)

    assert f1(nt) == ~f2(nt)
    assert f1(nn) == ~f2(nn)
    assert f1(nf) == ~f2(nf)

    ft = i(1.9, 2.1, is_valid=True)
    fn = i(1.9, 2.1, is_valid=None)
    ff = i(1.9, 2.1, is_valid=False)

    assert f1(ft) == ~f2(ft)
    assert f1(fn) == ~f2(fn)
    assert f1(ff) == ~f2(ff)


def test_boolean():
    # 有 9*9 个测试用例的全排列，但我们只考虑简化后的 3*3 情况
    s = [
        intervalMembership(False, False),  # 创建 intervalMembership 对象
        intervalMembership(None, None),
        intervalMembership(True, True)
    ]

    # 减少测试 'And' 操作
    a1 = [
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(None, None),
        intervalMembership(None, None),
        intervalMembership(False, False),
        intervalMembership(None, None),
        intervalMembership(True, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        for j in range(len(s)):
            assert s[i] & s[j] == next(a1_iter)

    # Reduced tests for 'Or'
    # 创建包含 intervalMembership 对象的列表 a1，表示按位运算的结果
    a1 = [
        intervalMembership(False, False),    # s[i] | s[j] 的预期结果为 False
        intervalMembership(None, False),     # s[i] | s[j] 的预期结果为 None
        intervalMembership(True, False),     # s[i] | s[j] 的预期结果为 True
        intervalMembership(None, False),     # s[i] | s[j] 的预期结果为 None
        intervalMembership(None, None),      # s[i] | s[j] 的预期结果为 None
        intervalMembership(True, None),      # s[i] | s[j] 的预期结果为 True
        intervalMembership(True, False),     # s[i] | s[j] 的预期结果为 True
        intervalMembership(True, None),      # s[i] | s[j] 的预期结果为 True
        intervalMembership(True, True)       # s[i] | s[j] 的预期结果为 True
    ]
    # 创建 a1 列表的迭代器
    a1_iter = iter(a1)
    # 遍历字符串 s 的索引范围
    for i in range(len(s)):
        # 遍历字符串 s 的索引范围
        for j in range(len(s)):
            # 断言按位或运算 s[i] | s[j] 的结果与预期的下一个 a1 迭代元素相等
            assert s[i] | s[j] == next(a1_iter)

    # 缩减版的按位异或运算测试
    a1 = [
        intervalMembership(False, False),    # s[i] ^ s[j] 的预期结果为 False
        intervalMembership(None, False),     # s[i] ^ s[j] 的预期结果为 None
        intervalMembership(True, False),     # s[i] ^ s[j] 的预期结果为 True
        intervalMembership(None, False),     # s[i] ^ s[j] 的预期结果为 None
        intervalMembership(None, None),      # s[i] ^ s[j] 的预期结果为 None
        intervalMembership(None, None),      # s[i] ^ s[j] 的预期结果为 None
        intervalMembership(True, False),     # s[i] ^ s[j] 的预期结果为 True
        intervalMembership(None, None),      # s[i] ^ s[j] 的预期结果为 None
        intervalMembership(False, True)      # s[i] ^ s[j] 的预期结果为 True
    ]
    # 创建 a1 列表的迭代器
    a1_iter = iter(a1)
    # 遍历字符串 s 的索引范围
    for i in range(len(s)):
        # 遍历字符串 s 的索引范围
        for j in range(len(s)):
            # 断言按位异或运算 s[i] ^ s[j] 的结果与预期的下一个 a1 迭代元素相等
            assert s[i] ^ s[j] == next(a1_iter)

    # 缩减版的按位非运算测试
    a1 = [
        intervalMembership(True, False),     # ~s[i] 的预期结果为 True
        intervalMembership(None, None),      # ~s[i] 的预期结果为 None
        intervalMembership(False, True)      # ~s[i] 的预期结果为 False
    ]
    # 创建 a1 列表的迭代器
    a1_iter = iter(a1)
    # 遍历字符串 s 的索引范围
    for i in range(len(s)):
        # 断言按位非运算 ~s[i] 的结果与预期的下一个 a1 迭代元素相等
        assert ~s[i] == next(a1_iter)
# 定义一个测试函数，用于测试布尔区间的错误情况
def test_boolean_errors():
    # 调用 intervalMembership 函数，传入 True 和 True 作为参数，返回结果赋给变量 a
    a = intervalMembership(True, True)
    # 断言：当尝试对 a 和整数 1 进行按位与操作时，引发 ValueError 异常
    raises(ValueError, lambda: a & 1)
    # 断言：当尝试对 a 和整数 1 进行按位或操作时，引发 ValueError 异常
    raises(ValueError, lambda: a | 1)
    # 断言：当尝试对 a 和整数 1 进行按位异或操作时，引发 ValueError 异常
    raises(ValueError, lambda: a ^ 1)
```