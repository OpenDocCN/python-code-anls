# `D:\src\scipysrc\sympy\sympy\core\tests\test_logic.py`

```
from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
    fuzzy_or, _fuzzy_group, _torf, fuzzy_nand, fuzzy_xor)
    # 导入逻辑模块的各种函数和类

from sympy.testing.pytest import raises
    # 导入 pytest 中的 raises 函数用于测试异常

from itertools import product
    # 导入 itertools 中的 product 函数，用于生成迭代器的笛卡尔积

T = True
F = False
U = None
    # 定义 True, False, None 的简写

def test_torf():
    v = [T, F, U]
    # 列表 v 包含 True, False, None 三个元素
    for i in product(*[v]*3):
        # 遍历 v 三个元素的笛卡尔积
        assert _torf(i) is (True if all(j for j in i) else
                            (False if all(j is False for j in i) else None))
        # 使用 _torf 函数测试笛卡尔积元组 i 的布尔值组合是否符合条件

def test_fuzzy_group():
    v = [T, F, U]
    # 列表 v 包含 True, False, None 三个元素
    for i in product(*[v]*3):
        # 遍历 v 三个元素的笛卡尔积
        assert _fuzzy_group(i) is (None if None in i else
                                   (True if all(j for j in i) else False))
        # 使用 _fuzzy_group 函数测试笛卡尔积元组 i 的模糊逻辑组合结果
        assert _fuzzy_group(i, quick_exit=True) is \
            (None if (i.count(False) > 1) else
             (None if None in i else (True if all(j for j in i) else False)))
        # 使用 _fuzzy_group 函数测试笛卡尔积元组 i 的模糊逻辑组合结果，带有快速退出选项
    it = (True if (i == 0) else None for i in range(2))
    # 创建一个生成器 it，包含两个元素 True 和 None
    assert _torf(it) is None
    # 使用 _torf 函数测试生成器 it 的布尔值组合是否符合条件
    it = (True if (i == 1) else None for i in range(2))
    # 创建一个生成器 it，包含两个元素 None 和 True
    assert _torf(it) is None
    # 使用 _torf 函数测试生成器 it 的布尔值组合是否符合条件

def test_fuzzy_not():
    assert fuzzy_not(T) == F
    # 断言 fuzzy_not 函数应用于 True 的结果为 False
    assert fuzzy_not(F) == T
    # 断言 fuzzy_not 函数应用于 False 的结果为 True
    assert fuzzy_not(U) == U
    # 断言 fuzzy_not 函数应用于 None 的结果为 None

def test_fuzzy_and():
    assert fuzzy_and([T, T]) == T
    # 断言 fuzzy_and 函数应用于 [True, True] 的结果为 True
    assert fuzzy_and([T, F]) == F
    # 断言 fuzzy_and 函数应用于 [True, False] 的结果为 False
    assert fuzzy_and([T, U]) == U
    # 断言 fuzzy_and 函数应用于 [True, None] 的结果为 None
    assert fuzzy_and([F, F]) == F
    # 断言 fuzzy_and 函数应用于 [False, False] 的结果为 False
    assert fuzzy_and([F, U]) == F
    # 断言 fuzzy_and 函数应用于 [False, None] 的结果为 False
    assert fuzzy_and([U, U]) == U
    # 断言 fuzzy_and 函数应用于 [None, None] 的结果为 None
    assert [fuzzy_and([w]) for w in [U, T, F]] == [U, T, F]
    # 断言对于列表中的每个元素 w，fuzzy_and 函数应用于 [w] 的结果与预期一致
    assert fuzzy_and([T, F, U]) == F
    # 断言 fuzzy_and 函数应用于 [True, False, None] 的结果为 False
    assert fuzzy_and([]) == T
    # 断言 fuzzy_and 函数应用于空列表的结果为 True
    raises(TypeError, lambda: fuzzy_and())
    # 断言调用 fuzzy_and 函数时会引发 TypeError 异常

def test_fuzzy_or():
    assert fuzzy_or([T, T]) == T
    # 断言 fuzzy_or 函数应用于 [True, True] 的结果为 True
    assert fuzzy_or([T, F]) == T
    # 断言 fuzzy_or 函数应用于 [True, False] 的结果为 True
    assert fuzzy_or([T, U]) == T
    # 断言 fuzzy_or 函数应用于 [True, None] 的结果为 True
    assert fuzzy_or([F, F]) == F
    # 断言 fuzzy_or 函数应用于 [False, False] 的结果为 False
    assert fuzzy_or([F, U]) == U
    # 断言 fuzzy_or 函数应用于 [False, None] 的结果为 None
    assert fuzzy_or([U, U]) == U
    # 断言 fuzzy_or 函数应用于 [None, None] 的结果为 None
    assert [fuzzy_or([w]) for w in [U, T, F]] == [U, T, F]
    # 断言对于列表中的每个元素 w，fuzzy_or 函数应用于 [w] 的结果与预期一致
    assert fuzzy_or([T, F, U]) == T
    # 断言 fuzzy_or 函数应用于 [True, False, None] 的结果为 True
    assert fuzzy_or([]) == F
    # 断言 fuzzy_or 函数应用于空列表的结果为 False
    raises(TypeError, lambda: fuzzy_or())
    # 断言调用 fuzzy_or 函数时会引发 TypeError 异常

def test_logic_cmp():
    l1 = And('a', Not('b'))
    # 创建逻辑表达式 l1 = 'a' 与 非 'b'
    l2 = And('a', Not('b'))
    # 创建逻辑表达式 l2 = 'a' 与 非 'b'

    assert hash(l1) == hash(l2)
    # 断言逻辑表达式 l1 和 l2 的哈希值相等
    assert (l1 == l2) == T
    # 断言逻辑表达式 l1 和 l2 相等为 True
    assert (l1 != l2) == F
    # 断言逻辑表达式 l1 和 l2 不等为 False

    assert And('a', 'b', 'c') == And('b', 'a', 'c')
    # 断言逻辑表达式 'a' 与 'b' 与 'c' 与运算的结果与 'b' 与 'a' 与 'c' 与运算的结果相等
    assert And('a', 'b', 'c') == And('c', 'b', 'a')
    # 断言逻辑表达式 'a' 与 'b' 与 'c' 与运算的结果与 'c' 与 'b' 与 'a' 与运算的结果相等
    assert And('a', 'b', 'c') == And('c', 'a', 'b')
    # 断言逻辑表达式 'a' 与 'b' 与 'c' 与运算的结果与 'c' 与 'a' 与 'b' 与运算的结果相等

    assert Not('a') < Not('b')
    # 断言逻辑表达式 非 'a' 小于 非 'b'
    assert (Not('b') < Not('a')) is False
    # 断言逻辑表达式 非 'b' 小于 非 'a' 结果为 False
    assert (Not('a') < 2) is False
    # 断言逻辑表达式 非 'a' 小于 2 的结果为 False

def test_logic_onearg():
    assert And() is True
    # 断言不带参数调用 And 函数的结果为 True
    assert Or() is False
    # 断言不带参数
# 定义测试逻辑运算组合函数
def test_logic_combine_args():
    # 断言两个 And 对象相等，去重其中的重复参数
    assert And('a', 'b', 'a') == And('a', 'b')
    # 断言两个 Or 对象相等，去重其中的重复参数
    assert Or('a', 'b', 'a') == Or('a', 'b')

    # 断言两个 And 对象的组合等于合并它们的参数
    assert And(And('a', 'b'), And('c', 'd')) == And('a', 'b', 'c', 'd')
    # 断言两个 Or 对象的组合等于合并它们的参数
    assert Or(Or('a', 'b'), Or('c', 'd')) == Or('a', 'b', 'c', 'd')

    # 断言一个复杂的 Or 对象的化简结果正确
    assert Or('t', And('n', 'p', 'r'), And('n', 'r'), And('n', 'p', 'r'), 't',
              And('n', 'r')) == Or('t', And('n', 'p', 'r'), And('n', 'r'))


# 定义测试逻辑展开函数
def test_logic_expand():
    # 创建 And 对象并展开，断言展开结果正确
    t = And(Or('a', 'b'), 'c')
    assert t.expand() == Or(And('a', 'c'), And('b', 'c'))

    # 创建 And 对象并展开，断言展开结果正确
    t = And(Or('a', Not('b')), 'b')
    assert t.expand() == And('a', 'b')

    # 创建 And 对象并展开，断言展开结果正确
    t = And(Or('a', 'b'), Or('c', 'd'))
    assert t.expand() == \
        Or(And('a', 'c'), And('a', 'd'), And('b', 'c'), And('b', 'd'))


# 定义测试从字符串解析逻辑表达式的函数
def test_logic_fromstring():
    S = Logic.fromstring

    # 断言从字符串 'a' 解析的结果正确
    assert S('a') == 'a'
    # 断言从字符串 '!a' 解析的结果正确
    assert S('!a') == Not('a')
    # 断言从字符串 'a & b' 解析的结果正确
    assert S('a & b') == And('a', 'b')
    # 断言从字符串 'a | b' 解析的结果正确
    assert S('a | b') == Or('a', 'b')
    # 断言从字符串 'a | b & c' 解析的结果正确
    assert S('a | b & c') == And(Or('a', 'b'), 'c')
    # 断言从字符串 'a & b | c' 解析的结果正确
    assert S('a & b | c') == Or(And('a', 'b'), 'c')
    # 断言从字符串 'a & b & c' 解析的结果正确
    assert S('a & b & c') == And('a', 'b', 'c')
    # 断言从字符串 'a | b | c' 解析的结果正确
    assert S('a | b | c') == Or('a', 'b', 'c')

    # 断言解析非法字符串会引发 ValueError 异常
    raises(ValueError, lambda: S('| a'))
    raises(ValueError, lambda: S('& a'))
    raises(ValueError, lambda: S('a | | b'))
    raises(ValueError, lambda: S('a | & b'))
    raises(ValueError, lambda: S('a & & b'))
    raises(ValueError, lambda: S('a |'))
    raises(ValueError, lambda: S('a|b'))
    raises(ValueError, lambda: S('!'))
    raises(ValueError, lambda: S('! a'))
    raises(ValueError, lambda: S('!(a + 1)'))
    raises(ValueError, lambda: S(''))


# 定义测试逻辑取反函数
def test_logic_not():
    # 断言取反操作的结果正确
    assert Not('a') != '!a'
    assert Not('!a') != 'a'
    assert Not(True) == False
    assert Not(False) == True

    # 注：也许应该将默认的 Not 行为改变并放入某个方法中
    # 断言对 And 对象取反的结果正确
    assert Not(And('a', 'b')) == Or(Not('a'), Not('b'))
    # 断言对 Or 对象取反的结果正确
    assert Not(Or('a', 'b')) == And(Not('a'), Not('b'))

    # 断言对非法参数取反会引发 ValueError 异常
    raises(ValueError, lambda: Not(1))


# 定义测试格式化函数
def test_formatting():
    S = Logic.fromstring
    # 断言对非法格式化字符串会引发 ValueError 异常
    raises(ValueError, lambda: S('a&b'))
    raises(ValueError, lambda: S('a|b'))
    raises(ValueError, lambda: S('! a'))


# 定义测试模糊逻辑 XOR 函数
def test_fuzzy_xor():
    # 断言对给定参数的模糊 XOR 结果正确
    assert fuzzy_xor((None,)) is None
    assert fuzzy_xor((None, True)) is None
    assert fuzzy_xor((None, False)) is None
    assert fuzzy_xor((True, False)) is True
    assert fuzzy_xor((True, True)) is False
    assert fuzzy_xor((True, True, False)) is False
    assert fuzzy_xor((True, True, False, True)) is True


# 定义测试模糊逻辑 NAND 函数
def test_fuzzy_nand():
    # 遍历给定参数列表，断言模糊 NAND 函数的结果与模糊 AND 的非结果一致
    for args in [(1, 0), (1, 1), (0, 0)]:
        assert fuzzy_nand(args) == fuzzy_not(fuzzy_and(args))
```