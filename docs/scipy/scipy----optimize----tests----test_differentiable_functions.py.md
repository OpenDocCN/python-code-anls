# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_differentiable_functions.py`

```
# 导入 pytest 模块，用于编写和运行测试用例
import pytest
# 导入 platform 模块，用于获取系统平台信息
import platform
# 导入 numpy 库，并将其命名为 np，用于数值计算
import numpy as np
# 导入 numpy.testing 模块中的测试工具和断言函数
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_, assert_allclose,
                           assert_equal)
# 导入 scipy._lib._gcutils 模块中的函数 assert_deallocated
from scipy._lib._gcutils import assert_deallocated
# 导入 scipy.sparse 库中的 csr_matrix 类
from scipy.sparse import csr_matrix
# 导入 scipy.sparse.linalg 库中的 LinearOperator 类
from scipy.sparse.linalg import LinearOperator
# 导入 scipy.optimize._differentiable_functions 模块中的各种函数和类
from scipy.optimize._differentiable_functions import (ScalarFunction,
                                                      VectorFunction,
                                                      LinearVectorFunction,
                                                      IdentityVectorFunction)
# 导入 scipy.optimize 库中的 rosen, rosen_der, rosen_hess 函数
from scipy.optimize import rosen, rosen_der, rosen_hess
# 导入 scipy.optimize._hessian_update_strategy 模块中的 BFGS 类
from scipy.optimize._hessian_update_strategy import BFGS


# 自定义的 ScalarFunction 类，用于定义标量函数及其梯度和黑塞矩阵
class ExScalarFunction:

    def __init__(self):
        self.nfev = 0  # 初始化函数调用次数计数器
        self.ngev = 0  # 初始化梯度计算次数计数器
        self.nhev = 0  # 初始化黑塞矩阵计算次数计数器

    # 定义标量函数 fun，计算特定公式的值
    def fun(self, x):
        self.nfev += 1  # 每调用一次 fun，增加函数调用次数计数器
        return 2*(x[0]**2 + x[1]**2 - 1) - x[0]

    # 定义梯度函数 grad，计算函数 fun 在某点的梯度
    def grad(self, x):
        self.ngev += 1  # 每调用一次 grad，增加梯度计算次数计数器
        return np.array([4*x[0]-1, 4*x[1]])

    # 定义黑塞矩阵函数 hess，计算函数 fun 在某点的黑塞矩阵
    def hess(self, x):
        self.nhev += 1  # 每调用一次 hess，增加黑塞矩阵计算次数计数器
        return 4*np.eye(2)


# 测试类 TestScalarFunction，用于测试 ScalarFunction 类的功能
class TestScalarFunction(TestCase):

    # 测试函数 test_fun_and_grad，测试标量函数和其梯度的计算
    def test_fun_and_grad(self):
        ex = ExScalarFunction()  # 创建 ExScalarFunction 的实例 ex

        # 自定义函数 fg_allclose，用于比较两个向量是否几乎相等
        def fg_allclose(x, y):
            assert_allclose(x[0], y[0])  # 断言 x[0] 几乎等于 y[0]
            assert_allclose(x[1], y[1])  # 断言 x[1] 几乎等于 y[1]

        x0 = [2.0, 0.3]  # 初始点 x0

        # 创建 ScalarFunction 实例 analit，使用解析梯度
        analit = ScalarFunction(ex.fun, x0, (), ex.grad,
                                ex.hess, None, (-np.inf, np.inf))

        fg = ex.fun(x0), ex.grad(x0)  # 计算函数值和梯度值
        fg_allclose(analit.fun_and_grad(x0), fg)  # 检查 analit 计算结果与预期是否一致
        assert analit.ngev == 1  # 断言 analit 的梯度计算次数为 1

        x0[1] = 1.  # 修改 x0 的第二个分量为 1
        fg = ex.fun(x0), ex.grad(x0)  # 重新计算函数值和梯度值
        fg_allclose(analit.fun_and_grad(x0), fg)  # 检查 analit 计算结果与预期是否一致

        # 创建 ScalarFunction 实例 sf，使用有限差分梯度
        x0 = [2.0, 0.3]  # 初始点 x0
        sf = ScalarFunction(ex.fun, x0, (), '3-point',
                                ex.hess, None, (-np.inf, np.inf))
        assert sf.ngev == 1  # 断言 sf 的梯度计算次数为 1
        fg = ex.fun(x0), ex.grad(x0)  # 计算函数值和梯度值
        fg_allclose(sf.fun_and_grad(x0), fg)  # 检查 sf 计算结果与预期是否一致
        assert sf.ngev == 1  # 再次断言 sf 的梯度计算次数为 1

        x0[1] = 1.  # 修改 x0 的第二个分量为 1
        fg = ex.fun(x0), ex.grad(x0)  # 重新计算函数值和梯度值
        fg_allclose(sf.fun_and_grad(x0), fg)  # 检查 sf 计算结果与预期是否一致
    def test_x_storage_overlap(self):
        # Scalar_Function应该不存储数组的引用，而应该存储副本 - 这里检查原地更新数组是否会导致Scalar_Function.x也被更新。

        def f(x):
            return np.sum(np.asarray(x) ** 2)

        x = np.array([1., 2., 3.])
        sf = ScalarFunction(f, x, (), '3-point', lambda x: x, None, (-np.inf, np.inf))

        assert x is not sf.x  # 检查x是否不等于sf.x
        assert_equal(sf.fun(x), 14.0)  # 断言计算sf.fun(x)的结果是否等于14.0
        assert x is not sf.x  # 再次检查x是否不等于sf.x

        x[0] = 0.
        f1 = sf.fun(x)
        assert_equal(f1, 13.0)  # 断言计算sf.fun(x)的结果是否等于13.0

        x[0] = 1
        f2 = sf.fun(x)
        assert_equal(f2, 14.0)  # 断言计算sf.fun(x)的结果是否等于14.0
        assert x is not sf.x  # 检查x是否不等于sf.x

        # 现在测试指定HessianUpdate策略的情况
        hess = BFGS()
        x = np.array([1., 2., 3.])
        sf = ScalarFunction(f, x, (), '3-point', hess, None, (-np.inf, np.inf))

        assert x is not sf.x  # 检查x是否不等于sf.x
        assert_equal(sf.fun(x), 14.0)  # 断言计算sf.fun(x)的结果是否等于14.0
        assert x is not sf.x  # 再次检查x是否不等于sf.x

        x[0] = 0.
        f1 = sf.fun(x)
        assert_equal(f1, 13.0)  # 断言计算sf.fun(x)的结果是否等于13.0

        x[0] = 1
        f2 = sf.fun(x)
        assert_equal(f2, 14.0)  # 断言计算sf.fun(x)的结果是否等于14.0
        assert x is not sf.x  # 检查x是否不等于sf.x

        # gh13740 x在用户函数中被修改
        def ff(x):
            x *= x    # 覆盖x的值
            return np.sum(x)

        x = np.array([1., 2., 3.])
        sf = ScalarFunction(
            ff, x, (), '3-point', lambda x: x, None, (-np.inf, np.inf)
        )
        assert x is not sf.x  # 检查x是否不等于sf.x
        assert_equal(sf.fun(x), 14.0)  # 断言计算sf.fun(x)的结果是否等于14.0
        assert_equal(sf.x, np.array([1., 2., 3.]))  # 断言sf.x是否仍然是初始值np.array([1., 2., 3.])
        assert x is not sf.x  # 再次检查x是否不等于sf.x

    def test_lowest_x(self):
        # ScalarFunction应该记住访问过的最低func(x)值。
        x0 = np.array([2, 3, 4])
        sf = ScalarFunction(rosen, x0, (), rosen_der, rosen_hess,
                            None, None)
        sf.fun([1, 1, 1])
        sf.fun(x0)
        sf.fun([1.01, 1, 1.0])
        sf.grad([1.01, 1, 1.0])
        assert_equal(sf._lowest_f, 0.0)  # 断言_sf._lowest_f是否等于0.0
        assert_equal(sf._lowest_x, [1.0, 1.0, 1.0])  # 断言_sf._lowest_x是否等于[1.0, 1.0, 1.0]

        sf = ScalarFunction(rosen, x0, (), '2-point', rosen_hess,
                            None, (-np.inf, np.inf))
        sf.fun([1, 1, 1])
        sf.fun(x0)
        sf.fun([1.01, 1, 1.0])
        sf.grad([1.01, 1, 1.0])
        assert_equal(sf._lowest_f, 0.0)  # 断言_sf._lowest_f是否等于0.0
        assert_equal(sf._lowest_x, [1.0, 1.0, 1.0])  # 断言_sf._lowest_x是否等于[1.0, 1.0, 1.0]

    def test_float_size(self):
        x0 = np.array([2, 3, 4]).astype(np.float32)

        # 检查ScalarFunction/approx_derivative是否总是发送正确的float宽度
        def rosen_(x):
            assert x.dtype == np.float32
            return rosen(x)

        sf = ScalarFunction(rosen_, x0, (), '2-point', rosen_hess,
                            None, (-np.inf, np.inf))
        res = sf.fun(x0)
        assert res.dtype == np.float32  # 断言res的数据类型是否为np.float32
class ExVectorialFunction:

    def __init__(self):
        # 初始化计数器
        self.nfev = 0  # 记录函数调用次数
        self.njev = 0  # 记录雅可比矩阵计算次数
        self.nhev = 0  # 记录黑塞矩阵计算次数

    def fun(self, x):
        # 计算函数值并更新计数器
        self.nfev += 1
        return np.array([2*(x[0]**2 + x[1]**2 - 1) - x[0],
                         4*(x[0]**3 + x[1]**2 - 4) - 3*x[0]], dtype=x.dtype)

    def jac(self, x):
        # 计算雅可比矩阵并更新计数器
        self.njev += 1
        return np.array([[4*x[0]-1, 4*x[1]],
                         [12*x[0]**2-3, 8*x[1]]], dtype=x.dtype)

    def hess(self, x, v):
        # 计算黑塞矩阵乘以向量并更新计数器
        self.nhev += 1
        return v[0]*4*np.eye(2) + v[1]*np.array([[24*x[0], 0],
                                                 [0, 8]])


class TestVectorialFunction(TestCase):

    def test_x_storage_overlap(self):
        # 测试向量函数不应存储数组引用，而应存储副本 - 这个测试检查原地更新数组是否会导致 Scalar_Function.x 被更新。
        ex = ExVectorialFunction()
        x0 = np.array([1.0, 0.0])

        vf = VectorFunction(ex.fun, x0, '3-point', ex.hess, None, None,
                            (-np.inf, np.inf), None)

        assert x0 is not vf.x  # 检查是否复制了输入向量 x0
        assert_equal(vf.fun(x0), ex.fun(x0))  # 检查函数值是否正确
        assert x0 is not vf.x  # 再次确认是否复制了输入向量 x0

        x0[0] = 2.
        assert_equal(vf.fun(x0), ex.fun(x0))  # 检查更新后的函数值是否正确
        assert x0 is not vf.x  # 确认更新后是否复制了输入向量 x0

        x0[0] = 1.
        assert_equal(vf.fun(x0), ex.fun(x0))  # 恢复原始值后再次检查函数值是否正确
        assert x0 is not vf.x  # 确认恢复原始值后是否复制了输入向量 x0

        # 现在使用指定的 HessianUpdate 策略进行测试
        hess = BFGS()
        x0 = np.array([1.0, 0.0])
        vf = VectorFunction(ex.fun, x0, '3-point', hess, None, None,
                            (-np.inf, np.inf), None)

        with pytest.warns(UserWarning):
            # 过滤掉 UserWarning，因为 ExVectorialFunction 是线性的，Hessian 使用拟牛顿逼近。
            assert x0 is not vf.x  # 检查是否复制了输入向量 x0
            assert_equal(vf.fun(x0), ex.fun(x0))  # 检查函数值是否正确
            assert x0 is not vf.x  # 再次确认是否复制了输入向量 x0

            x0[0] = 2.
            assert_equal(vf.fun(x0), ex.fun(x0))  # 检查更新后的函数值是否正确
            assert x0 is not vf.x  # 确认更新后是否复制了输入向量 x0

            x0[0] = 1.
            assert_equal(vf.fun(x0), ex.fun(x0))  # 恢复原始值后再次检查函数值是否正确
            assert x0 is not vf.x  # 确认恢复原始值后是否复制了输入向量 x0

    def test_float_size(self):
        ex = ExVectorialFunction()
        x0 = np.array([1.0, 0.0]).astype(np.float32)

        vf = VectorFunction(ex.fun, x0, ex.jac, ex.hess, None, None,
                            (-np.inf, np.inf), None)

        res = vf.fun(x0)
        assert res.dtype == np.float32  # 检查函数返回结果的数据类型是否为 np.float32

        res = vf.jac(x0)
        assert res.dtype == np.float32  # 检查雅可比矩阵返回结果的数据类型是否为 np.float32


def test_LinearVectorFunction():
    A_dense = np.array([
        [-1, 2, 0],
        [0, 4, 2]
    ])
    x0 = np.zeros(3)
    A_sparse = csr_matrix(A_dense)
    x = np.array([1, -1, 0])
    v = np.array([-1, 1])
    Ax = np.array([-3, -4])

    f1 = LinearVectorFunction(A_dense, x0, None)
    assert_(not f1.sparse_jacobian)  # 检查是否为稠密雅可比矩阵

    f2 = LinearVectorFunction(A_dense, x0, True)
    assert_(f2.sparse_jacobian)  # 检查是否为稀疏雅可比矩阵

    f3 = LinearVectorFunction(A_dense, x0, False)
    # 断言检查 f3 的稀疏雅可比矩阵属性是否为 False
    assert_(not f3.sparse_jacobian)
    
    # 使用稀疏矩阵 A_sparse 和初始向量 x0 创建线性向量函数 f4，断言其稀疏雅可比矩阵属性为 True
    f4 = LinearVectorFunction(A_sparse, x0, None)
    assert_(f4.sparse_jacobian)
    
    # 使用稀疏矩阵 A_sparse 和初始向量 x0 创建线性向量函数 f5，并显式设置其稀疏雅可比矩阵属性为 True，断言其稀疏雅可比矩阵属性为 True
    f5 = LinearVectorFunction(A_sparse, x0, True)
    assert_(f5.sparse_jacobian)
    
    # 使用稀疏矩阵 A_sparse 和初始向量 x0 创建线性向量函数 f6，并显式设置其稀疏雅可比矩阵属性为 False，断言其稀疏雅可比矩阵属性为 False
    f6 = LinearVectorFunction(A_sparse, x0, False)
    assert_(not f6.sparse_jacobian)
    
    # 断言调用 f1 的函数 fun，并将向量 x 传递给它，检查其返回的结果与预期的 Ax 是否相等
    assert_array_equal(f1.fun(x), Ax)
    
    # 断言调用 f2 的函数 fun，并将向量 x 传递给它，检查其返回的结果与预期的 Ax 是否相等
    assert_array_equal(f2.fun(x), Ax)
    
    # 断言调用 f1 的函数 jac，并将向量 x 传递给它，检查其返回的雅可比矩阵与预期的 A_dense 是否相等
    assert_array_equal(f1.jac(x), A_dense)
    
    # 断言调用 f2 的函数 jac，并将向量 x 传递给它，检查其返回的雅可比矩阵（转换为稀疏矩阵形式）与预期的 A_sparse 是否相等
    assert_array_equal(f2.jac(x).toarray(), A_sparse.toarray())
    
    # 断言调用 f1 的函数 hess，并将向量 x 和向量 v 传递给它，检查其返回的 Hessian 矩阵（转换为稀疏矩阵形式）是否与预期的全零矩阵相等
    assert_array_equal(f1.hess(x, v).toarray(), np.zeros((3, 3)))
# 定义测试函数 `test_LinearVectorFunction_memoization`，用于测试线性向量函数的记忆化功能
def test_LinearVectorFunction_memoization():
    # 创建一个 2x3 的 numpy 数组 A，表示线性向量函数的系数矩阵
    A = np.array([[-1, 2, 0], [0, 4, 2]])
    # 创建一个长度为 3 的 numpy 数组 x0，表示线性向量函数的初始向量
    x0 = np.array([1, 2, -1])
    # 使用系数矩阵 A 和初始向量 x0 创建线性向量函数对象 fun
    fun = LinearVectorFunction(A, x0, False)

    # 断言初始向量 x0 等于 fun 对象的 x 属性
    assert_array_equal(x0, fun.x)
    # 断言系数矩阵 A 乘以初始向量 x0 等于 fun 对象的 f 属性
    assert_array_equal(A.dot(x0), fun.f)

    # 创建一个长度为 3 的 numpy 数组 x1，表示新的测试向量
    x1 = np.array([-1, 3, 10])
    # 断言系数矩阵 A 等于在 fun 对象上调用 jac 方法，并传入 x1 作为参数的结果
    assert_array_equal(A, fun.jac(x1))
    # 断言 x1 等于 fun 对象的 x 属性
    assert_array_equal(x1, fun.x)
    # 断言系数矩阵 A 乘以初始向量 x0 等于 fun 对象的 f 属性（验证记忆化是否生效）
    assert_array_equal(A.dot(x0), fun.f)
    # 断言系数矩阵 A 乘以 x1 等于在 fun 对象上调用 fun 方法，并传入 x1 作为参数的结果
    assert_array_equal(A.dot(x1), fun.fun(x1))
    # 再次断言系数矩阵 A 乘以 x1 等于 fun 对象的 f 属性（验证记忆化是否继续生效）
    assert_array_equal(A.dot(x1), fun.f)


# 定义测试函数 `test_IdentityVectorFunction`，用于测试恒等向量函数的不同参数配置
def test_IdentityVectorFunction():
    # 创建一个长度为 3 的全零 numpy 数组 x0
    x0 = np.zeros(3)

    # 创建三个不同配置的恒等向量函数对象 f1, f2, f3
    f1 = IdentityVectorFunction(x0, None)
    f2 = IdentityVectorFunction(x0, False)
    f3 = IdentityVectorFunction(x0, True)

    # 断言 f1 对象的 sparse_jacobian 属性为 True
    assert_(f1.sparse_jacobian)
    # 断言 f2 对象的 sparse_jacobian 属性为 False
    assert_(not f2.sparse_jacobian)
    # 断言 f3 对象的 sparse_jacobian 属性为 True
    assert_(f3.sparse_jacobian)

    # 创建一个长度为 3 的 numpy 数组 x，用于测试向量函数的计算
    x = np.array([-1, 2, 1])
    # 创建一个长度为 3 的 numpy 数组 v，用于测试向量函数的计算
    v = np.array([-2, 3, 0])

    # 断言 f1 对象在 x 上调用 fun 方法的结果等于 x 本身
    assert_array_equal(f1.fun(x), x)
    # 断言 f2 对象在 x 上调用 fun 方法的结果等于 x 本身
    assert_array_equal(f2.fun(x), x)

    # 断言 f1 对象在 x 上调用 jac 方法的结果转换成稀疏矩阵后等于 3x3 单位矩阵
    assert_array_equal(f1.jac(x).toarray(), np.eye(3))
    # 断言 f2 对象在 x 上调用 jac 方法的结果等于 3x3 单位矩阵
    assert_array_equal(f2.jac(x), np.eye(3))

    # 断言 f1 对象在 x, v 上调用 hess 方法的结果转换成稀疏矩阵后等于 3x3 全零矩阵
    assert_array_equal(f1.hess(x, v).toarray(), np.zeros((3, 3)))


# 使用 pytest.mark.skipif 标记的测试函数，用于检测在 PyPy 上的特定条件下是否需要跳过测试
@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="assert_deallocate not available on PyPy"
)
def test_ScalarFunctionNoReferenceCycle():
    """Regression test for gh-20768."""
    # 创建一个 ExScalarFunction 的实例 ex
    ex = ExScalarFunction()
    # 创建一个长度为 3 的全零 numpy 数组 x0
    x0 = np.zeros(3)
    # 使用 assert_deallocated 确保在创建 ScalarFunction 对象时不产生引用循环
    with assert_deallocated(lambda: ScalarFunction(ex.fun, x0, (), ex.grad,
                            ex.hess, None, (-np.inf, np.inf))):
        pass


# 使用 pytest.mark.skipif 和 pytest.mark.xfail 标记的测试函数，用于检测在 PyPy 上的特定条件下和标记为 xfail 的预期失败
@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="assert_deallocate not available on PyPy"
)
@pytest.mark.xfail(reason="TODO remove reference cycle from VectorFunction")
def test_VectorFunctionNoReferenceCycle():
    """Regression test for gh-20768."""
    # 创建一个 ExVectorialFunction 的实例 ex
    ex = ExVectorialFunction()
    # 创建一个包含两个元素的列表 x0
    x0 = [1.0, 0.0]
    # 使用 assert_deallocated 确保在创建 VectorFunction 对象时不产生引用循环
    with assert_deallocated(lambda: VectorFunction(ex.fun, x0, ex.jac,
                            ex.hess, None, None, (-np.inf, np.inf), None)):
        pass


# 使用 pytest.mark.skipif 标记的测试函数，用于检测在 PyPy 上的特定条件下是否需要跳过测试
@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="assert_deallocate not available on PyPy"
)
def test_LinearVectorFunctionNoReferenceCycle():
    """Regression test for gh-20768."""
    # 创建一个 2x3 的稀疏矩阵 A_sparse
    A_dense = np.array([
        [-1, 2, 0],
        [0, 4, 2]
    ])
    x0 = np.zeros(3)
    A_sparse = csr_matrix(A_dense)
    # 使用 assert_deallocated 确保在创建 LinearVectorFunction 对象时不产生引用循环
    with assert_deallocated(lambda: LinearVectorFunction(A_sparse, x0, None)):
        pass
```