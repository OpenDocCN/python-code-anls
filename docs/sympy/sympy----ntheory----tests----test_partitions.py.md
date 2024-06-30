# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_partitions.py`

```
# 导入 sympy.ntheory.partitions_ 模块中的 npartitions, _partition_rec, _partition 函数
from sympy.ntheory.partitions_ import npartitions, _partition_rec, _partition


# 测试 _partition_rec 函数
def test__partition_rec():
    # A000041 是整数分割数列的前几个值
    A000041 = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135,
               176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575]
    # 遍历 A000041 中的索引和值
    for n, val in enumerate(A000041):
        # 断言 _partition_rec(n) 的返回值与 A000041 中对应位置的值相等
        assert _partition_rec(n) == val


# 测试 _partition 函数
def test__partition():
    # 断言 _partition(k) 对于 k 在 0 到 12 之间的返回值与指定列表相等
    assert [_partition(k) for k in range(13)] == \
        [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77]
    # 断言 _partition(100) 的返回值为指定值
    assert _partition(100) == 190569292
    # 断言 _partition(200) 的返回值为指定值
    assert _partition(200) == 3972999029388
    # 断言 _partition(1000) 的返回值为指定值
    assert _partition(1000) == 24061467864032622473692149727991
    # 断言 _partition(1001) 的返回值为指定值
    assert _partition(1001) == 25032297938763929621013218349796
    # 断言 _partition(2000) 的返回值为指定值
    assert _partition(2000) == 4720819175619413888601432406799959512200344166
    # 断言 _partition(10000) 除以 10^10 的余数与指定值相等
    assert _partition(10000) % 10**10 == 6916435144
    # 断言 _partition(100000) 除以 10^10 的余数与指定值相等
    assert _partition(100000) % 10**10 == 9421098519


# 测试已废弃的 ntheory.symbolic.functions 函数
def test_deprecated_ntheory_symbolic_functions():
    # 导入 warns_deprecated_sympy 函数用于捕获 sympy 的已废弃警告
    from sympy.testing.pytest import warns_deprecated_sympy

    # 使用 warns_deprecated_sympy 上下文管理器捕获已废弃警告
    with warns_deprecated_sympy():
        # 断言 npartitions(0) 的返回值为 1，即整数分割数列中 n = 0 时的分割数为 1
        assert npartitions(0) == 1
```