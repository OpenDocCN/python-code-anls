# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_cython_blas.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.testing import (assert_allclose,  # 从 NumPy 的测试模块中导入断言函数，用于比较数组是否接近
                           assert_equal)  # 导入断言函数，用于比较数组是否相等
import scipy.linalg.cython_blas as blas  # 导入 Cython 实现的 BLAS 函数接口

class TestDGEMM:
    
    def test_transposes(self):
        # 创建一个 3x2 的浮点数数组 a，包含前两行和前两列的元素
        a = np.arange(12, dtype='d').reshape((3, 4))[:2,:2]
        # 创建一个 2x2 的浮点数数组 b，包含前两行和前两列的元素
        b = np.arange(1, 13, dtype='d').reshape((4, 3))[:2,:2]
        # 创建一个 2x4 的空浮点数数组 c
        c = np.empty((2, 4))[:2,:2]

        # 调用 blas 模块的 _test_dgemm 函数进行矩阵乘法操作
        blas._test_dgemm(1., a, b, 0., c)
        # 使用断言函数 assert_allclose 检查 c 是否等于 a 与 b 的矩阵乘积
        assert_allclose(c, a.dot(b))

        # 使用转置矩阵进行相同的操作
        blas._test_dgemm(1., a.T, b, 0., c)
        assert_allclose(c, a.T.dot(b))

        blas._test_dgemm(1., a, b.T, 0., c)
        assert_allclose(c, a.dot(b.T))

        blas._test_dgemm(1., a.T, b.T, 0., c)
        assert_allclose(c, a.T.dot(b.T))

        blas._test_dgemm(1., a, b, 0., c.T)
        assert_allclose(c, a.dot(b).T)

        blas._test_dgemm(1., a.T, b, 0., c.T)
        assert_allclose(c, a.T.dot(b).T)

        blas._test_dgemm(1., a, b.T, 0., c.T)
        assert_allclose(c, a.dot(b.T).T)

        blas._test_dgemm(1., a.T, b.T, 0., c.T)
        assert_allclose(c, a.T.dot(b.T).T)
    
    def test_shapes(self):
        # 创建一个 3x2 的浮点数数组 a
        a = np.arange(6, dtype='d').reshape((3, 2))
        # 创建一个 2x4 的浮点数数组 b
        b = np.arange(-6, 2, dtype='d').reshape((2, 4))
        # 创建一个 3x4 的空浮点数数组 c
        c = np.empty((3, 4))

        blas._test_dgemm(1., a, b, 0., c)
        assert_allclose(c, a.dot(b))

        # 使用转置矩阵进行相同的操作
        blas._test_dgemm(1., b.T, a.T, 0., c.T)
        assert_allclose(c, b.T.dot(a.T).T)
        
class TestWfuncPointers:
    """ Test the function pointers that are expected to fail on
    Mac OS X without the additional entry statement in their definitions
    in fblas_l1.pyf.src. """

    def test_complex_args(self):
        # 创建一个复数数组 cx
        cx = np.array([.5 + 1.j, .25 - .375j, 12.5 - 4.j], np.complex64)
        # 创建一个复数数组 cy
        cy = np.array([.8 + 2.j, .875 - .625j, -1. + 2.j], np.complex64)

        # 使用断言函数 assert_allclose 检查复数的点积运算结果
        assert_allclose(blas._test_cdotc(cx, cy),
                        -17.6468753815+21.3718757629j)
        assert_allclose(blas._test_cdotu(cx, cy),
                        -6.11562538147+30.3156242371j)

        # 使用断言函数 assert_equal 检查返回的最大模值的索引
        assert_equal(blas._test_icamax(cx), 3)

        # 使用断言函数 assert_allclose 检查复数的绝对和与 Euclidean 范数
        assert_allclose(blas._test_scasum(cx), 18.625)
        assert_allclose(blas._test_scnrm2(cx), 13.1796483994)

        # 对数组的每隔一个元素进行相同操作
        assert_allclose(blas._test_cdotc(cx[::2], cy[::2]),
                        -18.1000003815+21.2000007629j)
        assert_allclose(blas._test_cdotu(cx[::2], cy[::2]),
                        -6.10000038147+30.7999992371j)
        assert_allclose(blas._test_scasum(cx[::2]), 18.)
        assert_allclose(blas._test_scnrm2(cx[::2]), 13.1719398499)
    # 测试函数，用于测试接受双精度浮点数参数的 BLAS 函数

    x = np.array([5., -3, -.5], np.float64)  # 创建双精度浮点数数组 x
    y = np.array([2, 1, .5], np.float64)     # 创建双精度浮点数数组 y

    # 断言测试 _test_dasum 函数对 x 的计算结果接近 8.5
    assert_allclose(blas._test_dasum(x), 8.5)
    # 断言测试 _test_ddot 函数对 x 和 y 的内积计算结果接近 6.75
    assert_allclose(blas._test_ddot(x, y), 6.75)
    # 断言测试 _test_dnrm2 函数对 x 的二范数计算结果接近 5.85234975815
    assert_allclose(blas._test_dnrm2(x), 5.85234975815)

    # 断言测试 _test_dasum 函数对 x 的每隔一个元素的子集计算结果接近 5.5
    assert_allclose(blas._test_dasum(x[::2]), 5.5)
    # 断言测试 _test_ddot 函数对 x 和 y 的每隔一个元素的子集的内积计算结果接近 9.75
    assert_allclose(blas._test_ddot(x[::2], y[::2]), 9.75)
    # 断言测试 _test_dnrm2 函数对 x 的每隔一个元素的子集的二范数计算结果接近 5.0249376297
    assert_allclose(blas._test_dnrm2(x[::2]), 5.0249376297)

    # 断言测试 _test_idamax 函数对 x 返回的最大元素索引是否为 1
    assert_equal(blas._test_idamax(x), 1)


    # 测试函数，用于测试接受单精度浮点数参数的 BLAS 函数

    x = np.array([5., -3, -.5], np.float32)  # 创建单精度浮点数数组 x
    y = np.array([2, 1, .5], np.float32)     # 创建单精度浮点数数组 y

    # 断言测试 _test_isamax 函数对 x 返回的最大元素索引是否为 1
    assert_equal(blas._test_isamax(x), 1)

    # 断言测试 _test_sasum 函数对 x 的计算结果接近 8.5
    assert_allclose(blas._test_sasum(x), 8.5)
    # 断言测试 _test_sdot 函数对 x 和 y 的内积计算结果接近 6.75
    assert_allclose(blas._test_sdot(x, y), 6.75)
    # 断言测试 _test_snrm2 函数对 x 的二范数计算结果接近 5.85234975815
    assert_allclose(blas._test_snrm2(x), 5.85234975815)

    # 断言测试 _test_sasum 函数对 x 的每隔一个元素的子集计算结果接近 5.5
    assert_allclose(blas._test_sasum(x[::2]), 5.5)
    # 断言测试 _test_sdot 函数对 x 和 y 的每隔一个元素的子集的内积计算结果接近 9.75
    assert_allclose(blas._test_sdot(x[::2], y[::2]), 9.75)
    # 断言测试 _test_snrm2 函数对 x 的每隔一个元素的子集的二范数计算结果接近 5.0249376297
    assert_allclose(blas._test_snrm2(x[::2]), 5.0249376297)


    # 测试函数，用于测试接受双精度复数参数的 BLAS 函数

    cx = np.array([.5 + 1.j, .25 - .375j, 13. - 4.j], np.complex128)  # 创建双精度复数数组 cx
    cy = np.array([.875 + 2.j, .875 - .625j, -1. + 2.j], np.complex128)  # 创建双精度复数数组 cy

    # 断言测试 _test_izamax 函数对 cx 返回的最大元素索引是否为 3
    assert_equal(blas._test_izamax(cx), 3)

    # 断言测试 _test_zdotc 函数对 cx 和 cy 的共轭内积计算结果接近 -18.109375+22.296875j
    assert_allclose(blas._test_zdotc(cx, cy), -18.109375+22.296875j)
    # 断言测试 _test_zdotu 函数对 cx 和 cy 的非共轭内积计算结果接近 -6.578125+31.390625j
    assert_allclose(blas._test_zdotu(cx, cy), -6.578125+31.390625j)

    # 断言测试 _test_zdotc 函数对 cx 和 cy 的每隔一个元素的子集的共轭内积计算结果接近 -18.5625+22.125j
    assert_allclose(blas._test_zdotc(cx[::2], cy[::2]), -18.5625+22.125j)
    # 断言测试 _test_zdotu 函数对 cx 和 cy 的每隔一个元素的子集的非共轭内积计算结果接近 -6.5625+31.875j
    assert_allclose(blas._test_zdotu(cx[::2], cy[::2]), -6.5625+31.875j)
```