# `.\numpy\numpy\lib\tests\test_regression.py`

```py
import os  # 导入操作系统模块

import numpy as np  # 导入NumPy库
from numpy.testing import (  # 导入NumPy测试模块中的函数和类
    assert_, assert_equal, assert_array_equal, assert_array_almost_equal,
    assert_raises, _assert_valid_refcount,
    )
import pytest  # 导入pytest测试框架


class TestRegression:
    def test_poly1d(self):
        # Ticket #28
        # 测试 np.poly1d 函数的行为，验证多项式减法
        assert_equal(np.poly1d([1]) - np.poly1d([1, 0]),
                     np.poly1d([-1, 1]))

    def test_cov_parameters(self):
        # Ticket #91
        # 创建随机矩阵 x，并复制到 y
        x = np.random.random((3, 3))
        y = x.copy()
        # 分别计算 x 和 y 的协方差矩阵，验证结果一致性
        np.cov(x, rowvar=True)
        np.cov(y, rowvar=False)
        assert_array_equal(x, y)

    def test_mem_digitize(self):
        # Ticket #95
        # 循环进行数字化处理，验证 np.digitize 函数的内存使用情况
        for i in range(100):
            np.digitize([1, 2, 3, 4], [1, 3])
            np.digitize([0, 1, 2, 3, 4], [1, 3])

    def test_unique_zero_sized(self):
        # Ticket #205
        # 测试空数组的唯一值，验证 np.unique 函数的行为
        assert_array_equal([], np.unique(np.array([])))

    def test_mem_vectorise(self):
        # Ticket #325
        # 使用 np.vectorize 函数创建向量化函数，并验证其内存使用情况
        vt = np.vectorize(lambda *args: args)
        vt(np.zeros((1, 2, 1)), np.zeros((2, 1, 1)), np.zeros((1, 1, 2)))
        vt(np.zeros((1, 2, 1)), np.zeros((2, 1, 1)), np.zeros((1,
           1, 2)), np.zeros((2, 2)))

    def test_mgrid_single_element(self):
        # Ticket #339
        # 验证 np.mgrid 在只有一个元素时的行为
        assert_array_equal(np.mgrid[0:0:1j], [0])
        assert_array_equal(np.mgrid[0:0], [])

    def test_refcount_vectorize(self):
        # Ticket #378
        # 定义函数 p，并使用 np.vectorize 进行向量化处理，验证其引用计数
        def p(x, y):
            return 123
        v = np.vectorize(p)
        _assert_valid_refcount(v)

    def test_poly1d_nan_roots(self):
        # Ticket #396
        # 创建具有 NaN 根的多项式，验证 np.poly1d 函数的异常处理
        p = np.poly1d([np.nan, np.nan, 1], r=False)
        assert_raises(np.linalg.LinAlgError, getattr, p, "r")

    def test_mem_polymul(self):
        # Ticket #448
        # 验证空列表输入时 np.polymul 的内存使用情况
        np.polymul([], [1.])

    def test_mem_string_concat(self):
        # Ticket #469
        # 创建空数组 x，并向其附加字符串，验证 np.append 函数的行为
        x = np.array([])
        np.append(x, 'asdasd\tasdasd')

    def test_poly_div(self):
        # Ticket #553
        # 创建两个多项式 u 和 v，并验证 np.polydiv 函数的行为
        u = np.poly1d([1, 2, 3])
        v = np.poly1d([1, 2, 3, 4, 5])
        q, r = np.polydiv(u, v)
        assert_equal(q*v + r, u)

    def test_poly_eq(self):
        # Ticket #554
        # 创建两个多项式 x 和 y，并验证其相等性
        x = np.poly1d([1, 2, 3])
        y = np.poly1d([3, 4])
        assert_(x != y)
        assert_(x == x)
    def test_polyfit_build(self):
        # Ticket #628
        # 参考值，多项式拟合的期望系数数组
        ref = [-1.06123820e-06, 5.70886914e-04, -1.13822012e-01,
               9.95368241e+00, -3.14526520e+02]
        # x 数据点
        x = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
             104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
             116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 129,
             130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
             146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
             158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
             170, 171, 172, 173, 174, 175, 176]
        # y 数据点
        y = [9.0, 3.0, 7.0, 4.0, 4.0, 8.0, 6.0, 11.0, 9.0, 8.0, 11.0, 5.0,
             6.0, 5.0, 9.0, 8.0, 6.0, 10.0, 6.0, 10.0, 7.0, 6.0, 6.0, 6.0,
             13.0, 4.0, 9.0, 11.0, 4.0, 5.0, 8.0, 5.0, 7.0, 7.0, 6.0, 12.0,
             7.0, 7.0, 9.0, 4.0, 12.0, 6.0, 6.0, 4.0, 3.0, 9.0, 8.0, 8.0,
             6.0, 7.0, 9.0, 10.0, 6.0, 8.0, 4.0, 7.0, 7.0, 10.0, 8.0, 8.0,
             6.0, 3.0, 8.0, 4.0, 5.0, 7.0, 8.0, 6.0, 6.0, 4.0, 12.0, 9.0,
             8.0, 8.0, 8.0, 6.0, 7.0, 4.0, 4.0, 5.0, 7.0]
        # 测试多项式拟合
        tested = np.polyfit(x, y, 4)
        # 断言拟合结果与参考值接近
        assert_array_almost_equal(ref, tested)

    def test_polydiv_type(self):
        # 使 polydiv 支持复数类型
        msg = "Wrong type, should be complex"
        x = np.ones(3, dtype=complex)
        # 对复数类型进行多项式除法
        q, r = np.polydiv(x, x)
        # 断言结果的数据类型是复数
        assert_(q.dtype == complex, msg)
        msg = "Wrong type, should be float"
        x = np.ones(3, dtype=int)
        # 对整数类型进行多项式除法
        q, r = np.polydiv(x, x)
        # 断言结果的数据类型是浮点数
        assert_(q.dtype == float, msg)

    def test_histogramdd_too_many_bins(self):
        # Ticket 928.
        # 检查 np.histogramdd 处理过多的 bins 时是否引发 ValueError
        assert_raises(ValueError, np.histogramdd, np.ones((1, 10)), bins=2**10)

    def test_polyint_type(self):
        # Ticket #944
        msg = "Wrong type, should be complex"
        x = np.ones(3, dtype=complex)
        # 对复数类型进行积分操作
        assert_(np.polyint(x).dtype == complex, msg)
        msg = "Wrong type, should be float"
        x = np.ones(3, dtype=int)
        # 对整数类型进行积分操作
        assert_(np.polyint(x).dtype == float, msg)

    def test_ndenumerate_crash(self):
        # Ticket 1140
        # 不应该导致崩溃的测试：对空数组使用 np.ndenumerate
        list(np.ndenumerate(np.array([[]])))

    def test_large_fancy_indexing(self):
        # 大规模的 fancy indexing，在 64 位系统上可能会失败
        nbits = np.dtype(np.intp).itemsize * 8
        thesize = int((2**nbits)**(1.0/5.0)+1)

        def dp():
            n = 3
            a = np.ones((n,)*5)
            i = np.random.randint(0, n, size=thesize)
            # 在数组 a 上进行大规模 fancy indexing
            a[np.ix_(i, i, i, i, i)] = 0

        def dp2():
            n = 3
            a = np.ones((n,)*5)
            i = np.random.randint(0, n, size=thesize)
            # 尝试进行大规模 fancy indexing
            a[np.ix_(i, i, i, i, i)]

        # 断言大规模 fancy indexing 会引发 ValueError
        assert_raises(ValueError, dp)
        assert_raises(ValueError, dp2)

    def test_void_coercion(self):
        dt = np.dtype([('a', 'f4'), ('b', 'i4')])
        x = np.zeros((1,), dt)
        # 测试结构化数组的拼接
        assert_(np.r_[x, x].dtype == dt)
    def test_include_dirs(self):
        # 检查函数 get_include 是否包含合理的内容，作为健全性检查
        # 相关于 ticket #1405 的部分
        include_dirs = [np.get_include()]
        # 遍历 include_dirs 中的每个路径
        for path in include_dirs:
            # 断言路径是字符串类型
            assert_(isinstance(path, str))
            # 断言路径不为空字符串
            assert_(path != '')

    def test_polyder_return_type(self):
        # Ticket #1249 的测试
        # 断言 np.polyder 函数返回的对象类型为 np.poly1d
        assert_(isinstance(np.polyder(np.poly1d([1]), 0), np.poly1d))
        # 断言 np.polyder 函数对于列表输入返回的对象类型为 np.ndarray
        assert_(isinstance(np.polyder([1], 0), np.ndarray))
        # 断言 np.polyder 函数进行一阶导数操作后返回的对象类型为 np.poly1d
        assert_(isinstance(np.polyder(np.poly1d([1]), 1), np.poly1d))
        # 断言 np.polyder 函数对于列表输入进行一阶导数操作返回的对象类型为 np.ndarray
        assert_(isinstance(np.polyder([1], 1), np.ndarray))

    def test_append_fields_dtype_list(self):
        # Ticket #1676 的测试
        from numpy.lib.recfunctions import append_fields

        # 创建一个基础数组
        base = np.array([1, 2, 3], dtype=np.int32)
        # 字段名称列表
        names = ['a', 'b', 'c']
        # 数据是单位矩阵的整数形式
        data = np.eye(3).astype(np.int32)
        # 数据类型列表
        dlist = [np.float64, np.int32, np.int32]
        try:
            # 尝试使用 append_fields 函数
            append_fields(base, names, data, dlist)
        except Exception:
            # 如果出现异常则抛出断言错误
            raise AssertionError()

    def test_loadtxt_fields_subarrays(self):
        # 对 ticket #1936 的测试
        from io import StringIO

        # 定义结构化数据类型
        dt = [("a", 'u1', 2), ("b", 'u1', 2)]
        # 使用 loadtxt 从字符串流读取数据到结构化数组
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        # 断言 x 的内容与给定的数组数据相等
        assert_equal(x, np.array([((0, 1), (2, 3))], dtype=dt))

        # 更复杂的结构化数据类型
        dt = [("a", [("a", 'u1', (1, 3)), ("b", 'u1')])]
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([(((0, 1, 2), 3),)], dtype=dt))

        # 具有多维形状的结构化数据类型
        dt = [("a", 'u1', (2, 2))]
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([(((0, 1), (2, 3)),)], dtype=dt))

        # 更复杂的多维结构化数据类型
        dt = [("a", 'u1', (2, 3, 2))]
        x = np.loadtxt(StringIO("0 1 2 3 4 5 6 7 8 9 10 11"), dtype=dt)
        data = [((((0, 1), (2, 3), (4, 5)), ((6, 7), (8, 9), (10, 11))),)]
        assert_equal(x, np.array(data, dtype=dt))

    def test_nansum_with_boolean(self):
        # 对 gh-2978 的测试
        # 创建一个布尔类型的零数组
        a = np.zeros(2, dtype=bool)
        try:
            # 尝试使用 np.nansum 函数
            np.nansum(a)
        except Exception:
            # 如果出现异常则抛出断言错误
            raise AssertionError()

    def test_py3_compat(self):
        # 对 gh-2561 的测试
        # 测试在 Python 3 中是否绕过了旧式类测试
        class C():
            """Python 2 中的旧式类，在 Python 3 中是普通类"""
            pass

        # 打开空设备文件，用于输出
        out = open(os.devnull, 'w')
        try:
            # 尝试使用 np.info 函数
            np.info(C(), output=out)
        except AttributeError:
            # 如果出现属性错误则抛出断言错误
            raise AssertionError()
        finally:
            # 关闭输出文件
            out.close()
```