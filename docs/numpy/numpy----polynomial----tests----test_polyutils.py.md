# `.\numpy\numpy\polynomial\tests\test_polyutils.py`

```
"""Tests for polyutils module.

"""
# 导入必要的库和模块
import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )

# 定义测试类 TestMisc
class TestMisc:

    # 测试函数 test_trimseq
    def test_trimseq(self):
        # 设置目标结果
        tgt = [1]
        # 遍历不同的尾部零个数
        for num_trailing_zeros in range(5):
            # 调用 trimseq 函数并断言结果与目标结果相等
            res = pu.trimseq([1] + [0] * num_trailing_zeros)
            assert_equal(res, tgt)

    # 测试函数 test_trimseq_empty_input
    def test_trimseq_empty_input(self):
        # 遍历空输入序列的不同类型
        for empty_seq in [[], np.array([], dtype=np.int32)]:
            # 调用 trimseq 函数并断言结果与输入序列相等
            assert_equal(pu.trimseq(empty_seq), empty_seq)

    # 测试函数 test_as_series
    def test_as_series(self):
        # 检查异常情况
        assert_raises(ValueError, pu.as_series, [[]])
        assert_raises(ValueError, pu.as_series, [[[1, 2]]])
        assert_raises(ValueError, pu.as_series, [[1], ['a']])
        
        # 检查常见类型
        types = ['i', 'd', 'O']
        for i in range(len(types)):
            for j in range(i):
                ci = np.ones(1, types[i])
                cj = np.ones(1, types[j])
                [resi, resj] = pu.as_series([ci, cj])
                assert_(resi.dtype.char == resj.dtype.char)
                assert_(resj.dtype.char == types[i])

    # 测试函数 test_trimcoef
    def test_trimcoef(self):
        coef = [2, -1, 1, 0]
        # 测试异常情况
        assert_raises(ValueError, pu.trimcoef, coef, -1)
        # 测试结果
        assert_equal(pu.trimcoef(coef), coef[:-1])
        assert_equal(pu.trimcoef(coef, 1), coef[:-3])
        assert_equal(pu.trimcoef(coef, 2), [0])

    # 测试函数 test_vander_nd_exception
    def test_vander_nd_exception(self):
        # n_dims != len(points)
        assert_raises(ValueError, pu._vander_nd, (), (1, 2, 3), [90])
        # n_dims != len(degrees)
        assert_raises(ValueError, pu._vander_nd, (), (), [90.65])
        # n_dims == 0
        assert_raises(ValueError, pu._vander_nd, (), (), [])

    # 测试函数 test_div_zerodiv
    def test_div_zerodiv(self):
        # c2[-1] == 0
        assert_raises(ZeroDivisionError, pu._div, pu._div, (1, 2, 3), [0])

    # 测试函数 test_pow_too_large
    def test_pow_too_large(self):
        # power > maxpower
        assert_raises(ValueError, pu._pow, (), [1, 2, 3], 5, 4)

# 定义测试类 TestDomain
class TestDomain:

    # 测试函数 test_getdomain
    def test_getdomain(self):
        # 测试实数值情况
        x = [1, 10, 3, -1]
        tgt = [-1, 10]
        res = pu.getdomain(x)
        assert_almost_equal(res, tgt)

        # 测试复数值情况
        x = [1 + 1j, 1 - 1j, 0, 2]
        tgt = [-1j, 2 + 1j]
        res = pu.getdomain(x)
        assert_almost_equal(res, tgt)
    # 定义测试方法 test_mapdomain，用于测试 mapdomain 函数的不同情况
    def test_mapdomain(self):
        # 测试实数值情况
        
        # 定义第一个定义域 dom1
        dom1 = [0, 4]
        # 定义第二个定义域 dom2
        dom2 = [1, 3]
        # 定义目标值 tgt 为 dom2
        tgt = dom2
        # 调用 pu.mapdomain 函数，将 dom1 映射到 dom2，并返回结果 res
        res = pu.mapdomain(dom1, dom1, dom2)
        # 断言 res 与目标值 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 测试复数值情况
        
        # 重新定义 dom1 为复数列表
        dom1 = [0 - 1j, 2 + 1j]
        # 定义第二个定义域 dom2
        dom2 = [-2, 2]
        # 目标值 tgt 为 dom2
        tgt = dom2
        # 定义 x 为 dom1
        x = dom1
        # 调用 pu.mapdomain 函数，将 x 映射到 dom2，并返回结果 res
        res = pu.mapdomain(x, dom1, dom2)
        # 断言 res 与目标值 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 测试多维数组情况
        
        # 重新定义 dom1 和 dom2 为数组
        dom1 = [0, 4]
        dom2 = [1, 3]
        # 目标值 tgt 为 dom2 组成的 numpy 数组
        tgt = np.array([dom2, dom2])
        # 定义 x 为 dom1 组成的 numpy 数组
        x = np.array([dom1, dom1])
        # 调用 pu.mapdomain 函数，将 x 中的每个元素映射到 dom2，并返回结果 res
        res = pu.mapdomain(x, dom1, dom2)
        # 断言 res 与目标值 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 测试子类型保持不变的情况
        
        # 定义 MyNDArray 类，继承自 np.ndarray
        class MyNDArray(np.ndarray):
            pass
        
        # 重新定义 dom1 和 dom2
        dom1 = [0, 4]
        dom2 = [1, 3]
        # 定义 x 为 dom1 组成的 numpy 数组，转换为 MyNDArray 类型
        x = np.array([dom1, dom1]).view(MyNDArray)
        # 调用 pu.mapdomain 函数，将 x 映射到 dom2，并返回结果 res
        res = pu.mapdomain(x, dom1, dom2)
        # 断言 res 的类型为 MyNDArray 类型
        assert_(isinstance(res, MyNDArray))

    # 定义测试方法 test_mapparms，用于测试 mapparms 函数的不同情况
    def test_mapparms(self):
        # 测试实数值情况
        
        # 定义第一个定义域 dom1
        dom1 = [0, 4]
        # 定义第二个定义域 dom2
        dom2 = [1, 3]
        # 定义目标值 tgt 为列表 [1, .5]
        tgt = [1, .5]
        # 调用 pu.mapparms 函数，将 dom1 映射到 dom2，并返回结果 res
        res = pu.mapparms(dom1, dom2)
        # 断言 res 与目标值 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 测试复数值情况
        
        # 重新定义 dom1 为复数列表
        dom1 = [0 - 1j, 2 + 1j]
        # 定义第二个定义域 dom2
        dom2 = [-2, 2]
        # 定义目标值 tgt 为复数列表 [-1 + 1j, 1 - 1j]
        tgt = [-1 + 1j, 1 - 1j]
        # 调用 pu.mapparms 函数，将 dom1 映射到 dom2，并返回结果 res
        res = pu.mapparms(dom1, dom2)
        # 断言 res 与目标值 tgt 几乎相等
        assert_almost_equal(res, tgt)
```