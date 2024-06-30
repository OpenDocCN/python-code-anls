# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_generators.py`

```
# 导入模块中的函数和类
from sympy.combinatorics.generators import symmetric, cyclic, alternating, \
    dihedral, rubik
# 导入 Permutation 类
from sympy.combinatorics.permutations import Permutation
# 导入 raises 函数用于测试异常
from sympy.testing.pytest import raises

# 定义测试函数 test_generators，用于测试生成器函数的输出是否符合预期
def test_generators():

    # 断言循环群生成器生成的置换列表是否正确
    assert list(cyclic(6)) == [
        Permutation([0, 1, 2, 3, 4, 5]),
        Permutation([1, 2, 3, 4, 5, 0]),
        Permutation([2, 3, 4, 5, 0, 1]),
        Permutation([3, 4, 5, 0, 1, 2]),
        Permutation([4, 5, 0, 1, 2, 3]),
        Permutation([5, 0, 1, 2, 3, 4])]

    # 断言循环群生成器生成更大规模的置换列表是否正确
    assert list(cyclic(10)) == [
        Permutation([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        Permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        Permutation([2, 3, 4, 5, 6, 7, 8, 9, 0, 1]),
        Permutation([3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
        Permutation([4, 5, 6, 7, 8, 9, 0, 1, 2, 3]),
        Permutation([5, 6, 7, 8, 9, 0, 1, 2, 3, 4]),
        Permutation([6, 7, 8, 9, 0, 1, 2, 3, 4, 5]),
        Permutation([7, 8, 9, 0, 1, 2, 3, 4, 5, 6]),
        Permutation([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]),
        Permutation([9, 0, 1, 2, 3, 4, 5, 6, 7, 8])]

    # 断言交错群生成器生成的置换列表是否正确
    assert list(alternating(4)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([0, 2, 3, 1]),
        Permutation([0, 3, 1, 2]),
        Permutation([1, 0, 3, 2]),
        Permutation([1, 2, 0, 3]),
        Permutation([1, 3, 2, 0]),
        Permutation([2, 0, 1, 3]),
        Permutation([2, 1, 3, 0]),
        Permutation([2, 3, 0, 1]),
        Permutation([3, 0, 2, 1]),
        Permutation([3, 1, 0, 2]),
        Permutation([3, 2, 1, 0])]

    # 断言对称群生成器生成的置换列表是否正确
    assert list(symmetric(3)) == [
        Permutation([0, 1, 2]),
        Permutation([0, 2, 1]),
        Permutation([1, 0, 2]),
        Permutation([1, 2, 0]),
        Permutation([2, 0, 1]),
        Permutation([2, 1, 0])]

    # 断言对称群生成器生成更大规模的置换列表是否正确
    assert list(symmetric(4)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([0, 1, 3, 2]),
        Permutation([0, 2, 1, 3]),
        Permutation([0, 2, 3, 1]),
        Permutation([0, 3, 1, 2]),
        Permutation([0, 3, 2, 1]),
        Permutation([1, 0, 2, 3]),
        Permutation([1, 0, 3, 2]),
        Permutation([1, 2, 0, 3]),
        Permutation([1, 2, 3, 0]),
        Permutation([1, 3, 0, 2]),
        Permutation([1, 3, 2, 0]),
        Permutation([2, 0, 1, 3]),
        Permutation([2, 0, 3, 1]),
        Permutation([2, 1, 0, 3]),
        Permutation([2, 1, 3, 0]),
        Permutation([2, 3, 0, 1]),
        Permutation([2, 3, 1, 0]),
        Permutation([3, 0, 1, 2]),
        Permutation([3, 0, 2, 1]),
        Permutation([3, 1, 0, 2]),
        Permutation([3, 1, 2, 0]),
        Permutation([3, 2, 0, 1]),
        Permutation([3, 2, 1, 0])]

    # 断言二面角群生成器生成的置换列表是否正确
    assert list(dihedral(1)) == [
        Permutation([0, 1]), Permutation([1, 0])]

    # 断言二面角群生成更大规模的置换列表是否正确
    assert list(dihedral(2)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([1, 0, 3, 2]),
        Permutation([2, 3, 0, 1]),
        Permutation([3, 2, 1, 0])]
    # 断言：验证 dihedral(3) 函数返回的结果是否与期望的列表相匹配
    assert list(dihedral(3)) == [
        Permutation([0, 1, 2]),
        Permutation([2, 1, 0]),
        Permutation([1, 2, 0]),
        Permutation([0, 2, 1]),
        Permutation([2, 0, 1]),
        Permutation([1, 0, 2])]
    
    # 断言：验证 dihedral(5) 函数返回的结果是否与期望的列表相匹配
    assert list(dihedral(5)) == [
        Permutation([0, 1, 2, 3, 4]),
        Permutation([4, 3, 2, 1, 0]),
        Permutation([1, 2, 3, 4, 0]),
        Permutation([0, 4, 3, 2, 1]),
        Permutation([2, 3, 4, 0, 1]),
        Permutation([1, 0, 4, 3, 2]),
        Permutation([3, 4, 0, 1, 2]),
        Permutation([2, 1, 0, 4, 3]),
        Permutation([4, 0, 1, 2, 3]),
        Permutation([3, 2, 1, 0, 4])]
    
    # 断言：验证 rubik(1) 函数调用是否会引发 ValueError 异常
    raises(ValueError, lambda: rubik(1))
```