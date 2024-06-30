# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__linprog_clean_inputs.py`

```
"""
Unit test for Linear Programming via Simplex Algorithm.
"""
# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_, assert_allclose, assert_equal  # 导入测试相关的断言函数
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 函数，用于断言异常
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem  # 导入线性规划相关的函数和类
from scipy._lib._util import VisibleDeprecationWarning  # 导入可见性警告相关内容
from copy import deepcopy  # 导入深拷贝函数 deepcopy
from datetime import date  # 导入日期模块


def test_aliasing():
    """
    Test for ensuring that no objects referred to by `lp` attributes,
    `c`, `A_ub`, `b_ub`, `A_eq`, `b_eq`, `bounds`, have been modified
    by `_clean_inputs` as a side effect.
    """
    # 创建 _LPProblem 对象 lp，并设置初始属性
    lp = _LPProblem(
        c=1,
        A_ub=[[1]],
        b_ub=[1],
        A_eq=[[1]],
        b_eq=[1],
        bounds=(-np.inf, np.inf)
    )
    lp_copy = deepcopy(lp)  # 深拷贝 lp 对象，用于后续比较

    _clean_inputs(lp)  # 调用 _clean_inputs 函数，对 lp 进行清理和检查

    # 使用断言检查 lp 的各属性是否被修改
    assert_(lp.c == lp_copy.c, "c modified by _clean_inputs")
    assert_(lp.A_ub == lp_copy.A_ub, "A_ub modified by _clean_inputs")
    assert_(lp.b_ub == lp_copy.b_ub, "b_ub modified by _clean_inputs")
    assert_(lp.A_eq == lp_copy.A_eq, "A_eq modified by _clean_inputs")
    assert_(lp.b_eq == lp_copy.b_eq, "b_eq modified by _clean_inputs")
    assert_(lp.bounds == lp_copy.bounds, "bounds modified by _clean_inputs")


def test_aliasing2():
    """
    Similar purpose as `test_aliasing` above.
    """
    # 创建 _LPProblem 对象 lp，并设置初始属性（使用 NumPy 数组）
    lp = _LPProblem(
        c=np.array([1, 1]),
        A_ub=np.array([[1, 1], [2, 2]]),
        b_ub=np.array([[1], [1]]),
        A_eq=np.array([[1, 1]]),
        b_eq=np.array([1]),
        bounds=[(-np.inf, np.inf), (None, 1)]
    )
    lp_copy = deepcopy(lp)  # 深拷贝 lp 对象，用于后续比较

    _clean_inputs(lp)  # 调用 _clean_inputs 函数，对 lp 进行清理和检查

    # 使用 assert_allclose 断言检查 NumPy 数组的属性是否被修改
    assert_allclose(lp.c, lp_copy.c, err_msg="c modified by _clean_inputs")
    assert_allclose(lp.A_ub, lp_copy.A_ub, err_msg="A_ub modified by _clean_inputs")
    assert_allclose(lp.b_ub, lp_copy.b_ub, err_msg="b_ub modified by _clean_inputs")
    assert_allclose(lp.A_eq, lp_copy.A_eq, err_msg="A_eq modified by _clean_inputs")
    assert_allclose(lp.b_eq, lp_copy.b_eq, err_msg="b_eq modified by _clean_inputs")
    assert_(lp.bounds == lp_copy.bounds, "bounds modified by _clean_inputs")


def test_missing_inputs():
    # 定义测试用例中的变量
    c = [1, 2]
    A_ub = np.array([[1, 1], [2, 2]])
    b_ub = np.array([1, 1])
    A_eq = np.array([[1, 1], [2, 2]])
    b_eq = np.array([1, 1])

    # 使用 assert_raises 断言检查不同情况下 _clean_inputs 函数的异常处理
    assert_raises(TypeError, _clean_inputs)
    assert_raises(TypeError, _clean_inputs, _LPProblem(c=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=A_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=A_ub, b_ub=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, b_ub=b_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=None, b_ub=b_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=A_eq))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=A_eq, b_eq=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, b_eq=b_eq))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=None, b_eq=b_eq))
# 测试函数，用于检查输入数组维度过多的情况
def test_too_many_dimensions():
    # 创建一个普通的一维数组
    cb = [1, 2, 3, 4]
    # 创建一个随机的 4x4 数组
    A = np.random.rand(4, 4)
    # 创建一个二维数组（不符合要求的输入）
    bad2D = [[1, 2], [3, 4]]
    # 创建一个三维数组（不符合要求的输入）
    bad3D = np.random.rand(4, 4, 4)
    # 检查是否会引发 ValueError 异常，期望检测到维度不匹配的问题
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=bad2D, A_ub=A, b_ub=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_ub=bad3D, b_ub=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_ub=A, b_ub=bad2D))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_eq=bad3D, b_eq=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_eq=A, b_eq=bad2D))


# 测试函数，用于检查输入数组维度过少的情况
def test_too_few_dimensions():
    # 创建一个扁平化后的数组（维度过少）
    bad = np.random.rand(4, 4).ravel()
    # 创建一个长度为 4 的随机数组
    cb = np.random.rand(4)
    # 检查是否会引发 ValueError 异常，期望检测到维度不匹配的问题
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_ub=bad, b_ub=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_eq=bad, b_eq=cb))


# 测试函数，用于检查输入数组维度不一致的情况
def test_inconsistent_dimensions():
    # 定义矩阵行数和列数
    m = 2
    n = 4
    c = [1, 2, 3, 4]

    # 创建一个 m x n 的随机数组（维度符合要求）
    Agood = np.random.rand(m, n)
    # 创建一个 m x (n+1) 的随机数组（列数过多，维度不一致）
    Abad = np.random.rand(m, n + 1)
    # 创建一个长度为 m+1 的随机数组（行数过多，维度不一致）
    bgood = np.random.rand(m)
    bbad = np.random.rand(m + 1)
    # 创建一个长度为 n+1 的元组列表（列数过多，维度不一致）
    boundsbad = [(0, 1)] * (n + 1)
    # 检查是否会引发 ValueError 异常，期望检测到维度不匹配的问题
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=Abad, b_ub=bgood))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=Agood, b_ub=bbad))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=Abad, b_eq=bgood))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=Agood, b_eq=bbad))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, bounds=boundsbad))
    with np.testing.suppress_warnings() as sup:
        sup.filter(VisibleDeprecationWarning, "Creating an ndarray from ragged")
        assert_raises(ValueError, _clean_inputs,
                      _LPProblem(c=c, bounds=[[1, 2], [2, 3], [3, 4], [4, 5, 6]]))


# 测试函数，用于检查输入类型错误的情况
def test_type_errors():
    # 创建一个 _LPProblem 对象，包含正确的输入
    lp = _LPProblem(
        c=[1, 2],
        A_ub=np.array([[1, 1], [2, 2]]),
        b_ub=np.array([1, 1]),
        A_eq=np.array([[1, 1], [2, 2]]),
        b_eq=np.array([1, 1]),
        bounds=[(0, 1)]
    )
    # 创建一个字符串对象（类型错误）
    bad = "hello"

    # 检查是否会引发 TypeError 异常，期望检测到输入类型错误的问题
    assert_raises(TypeError, _clean_inputs, lp._replace(c=bad))
    assert_raises(TypeError, _clean_inputs, lp._replace(A_ub=bad))
    assert_raises(TypeError, _clean_inputs, lp._replace(b_ub=bad))
    assert_raises(TypeError, _clean_inputs, lp._replace(A_eq=bad))
    assert_raises(TypeError, _clean_inputs, lp._replace(b_eq=bad))

    # 检查是否会引发 ValueError 异常，期望检测到输入值错误的问题
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=bad))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds="hi"))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=["hi"]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[("hi")]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, "")]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2), (1, "")]))
    assert_raises(TypeError, _clean_inputs,
                  lp._replace(bounds=[(1, date(2020, 2, 29))]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[[[1, 2]]]))
# 定义一个测试函数，用于测试处理非有限值错误的情况
def test_non_finite_errors():
    # 创建一个线性规划问题对象，包含有限值的参数
    lp = _LPProblem(
        c=[1, 2],  # 目标函数的系数向量
        A_ub=np.array([[1, 1], [2, 2]]),  # 不等式约束的系数矩阵
        b_ub=np.array([1, 1]),  # 不等式约束的右侧常数向量
        A_eq=np.array([[1, 1], [2, 2]]),  # 等式约束的系数矩阵
        b_eq=np.array([1, 1]),  # 等式约束的右侧常数向量
        bounds=[(0, 1)]  # 变量的边界条件列表
    )
    # 对于不同的非有限值情况，检查是否引发 ValueError 异常
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[0, None]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[np.inf, 0]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[0, -np.inf]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[np.nan, 0]))

    assert_raises(ValueError, _clean_inputs, lp._replace(A_ub=[[1, 2], [None, 1]]))
    assert_raises(ValueError, _clean_inputs, lp._replace(b_ub=[np.inf, 1]))
    assert_raises(ValueError, _clean_inputs, lp._replace(A_eq=[[1, 2], [1, -np.inf]]))
    assert_raises(ValueError, _clean_inputs, lp._replace(b_eq=[1, np.nan]))


# 定义一个测试函数，用于测试_clean_inputs函数的用例1
def test__clean_inputs1():
    # 创建一个线性规划问题对象，包含规范的参数
    lp = _LPProblem(
        c=[1, 2],  # 目标函数的系数向量
        A_ub=[[1, 1], [2, 2]],  # 不等式约束的系数矩阵
        b_ub=[1, 1],  # 不等式约束的右侧常数向量
        A_eq=[[1, 1], [2, 2]],  # 等式约束的系数矩阵
        b_eq=[1, 1],  # 等式约束的右侧常数向量
        bounds=None  # 变量的边界条件（未指定）
    )

    # 清理参数并返回一个新的线性规划问题对象
    lp_cleaned = _clean_inputs(lp)

    # 检查清理后的参数是否符合预期的数值
    assert_allclose(lp_cleaned.c, np.array(lp.c))
    assert_allclose(lp_cleaned.A_ub, np.array(lp.A_ub))
    assert_allclose(lp_cleaned.b_ub, np.array(lp.b_ub))
    assert_allclose(lp_cleaned.A_eq, np.array(lp.A_eq))
    assert_allclose(lp_cleaned.b_eq, np.array(lp.b_eq))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)

    # 检查清理后的参数形状是否符合预期
    assert_(lp_cleaned.c.shape == (2,), "")
    assert_(lp_cleaned.A_ub.shape == (2, 2), "")
    assert_(lp_cleaned.b_ub.shape == (2,), "")
    assert_(lp_cleaned.A_eq.shape == (2, 2), "")
    assert_(lp_cleaned.b_eq.shape == (2,), "")


# 定义一个测试函数，用于测试_clean_inputs函数的用例2
def test__clean_inputs2():
    # 创建一个线性规划问题对象，包含规范的参数（较简单的情况）
    lp = _LPProblem(
        c=1,  # 目标函数的系数（标量）
        A_ub=[[1]],  # 不等式约束的系数矩阵
        b_ub=1,  # 不等式约束的右侧常数（标量）
        A_eq=[[1]],  # 等式约束的系数矩阵
        b_eq=1,  # 等式约束的右侧常数（标量）
        bounds=(0, 1)  # 变量的边界条件（单个元组）
    )

    # 清理参数并返回一个新的线性规划问题对象
    lp_cleaned = _clean_inputs(lp)

    # 检查清理后的参数是否符合预期的数值
    assert_allclose(lp_cleaned.c, np.array(lp.c))
    assert_allclose(lp_cleaned.A_ub, np.array(lp.A_ub))
    assert_allclose(lp_cleaned.b_ub, np.array(lp.b_ub))
    assert_allclose(lp_cleaned.A_eq, np.array(lp.A_eq))
    assert_allclose(lp_cleaned.b_eq, np.array(lp.b_eq))
    assert_equal(lp_cleaned.bounds, [(0, 1)])

    # 检查清理后的参数形状是否符合预期
    assert_(lp_cleaned.c.shape == (1,), "")
    assert_(lp_cleaned.A_ub.shape == (1, 1), "")
    assert_(lp_cleaned.b_ub.shape == (1,), "")
    assert_(lp_cleaned.A_eq.shape == (1, 1), "")
    assert_(lp_cleaned.b_eq.shape == (1,), "")


# 定义一个测试函数，用于测试_clean_inputs函数的用例3
def test__clean_inputs3():
    # 创建一个线性规划问题对象，包含规范的参数（多维度情况）
    lp = _LPProblem(
        c=[[1, 2]],  # 目标函数的系数向量
        A_ub=np.random.rand(2, 2),  # 不等式约束的系数矩阵
        b_ub=[[1], [2]],  # 不等式约束的右侧常数向量
        A_eq=np.random.rand(2, 2),  # 等式约束的系数矩阵
        b_eq=[[1], [2]],  # 等式约束的右侧常数向量
        bounds=[(0, 1)]  # 变量的边界条件列表
    )

    # 清理参数并返回一个新的线性规划问题对象
    lp_cleaned = _clean_inputs(lp)

    # 检查清理后的参数是否符合预期的数值
    assert_allclose(lp_cleaned.c, np.array([1, 2]))
    assert_allclose(lp_cleaned.b_ub, np.array([1, 2]))
    assert_allclose(lp_cleaned.b_eq, np.array([1, 2]))
    assert_equal(lp_cleaned.bounds, [(0, 1)] * 2)

    # 检查清理后的参数形状是否符合预期
    assert_(lp_cleaned.c.shape == (2,), "")
    assert_(lp_cleaned.b_ub.shape == (2,), "")
    # 断言语句，用于验证条件是否为真，如果不为真则抛出异常
    assert_(lp_cleaned.b_eq.shape == (2,), "")
# 定义测试函数，用于测试不良的边界条件处理
def test_bad_bounds():
    # 创建一个 LP 问题对象，目标系数为 [1, 2]
    lp = _LPProblem(c=[1, 2])

    # 断言抛出 ValueError 异常，因为 bounds 参数元组长度不正确
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=(1, 2, 2)))
    
    # 断言抛出 ValueError 异常，因为 bounds 参数列表中有一个元组的长度不正确
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2, 2)]))
    
    # 忽略 VisibleDeprecationWarning 警告，断言抛出 ValueError 异常，因为其中一个元组长度不正确
    with np.testing.suppress_warnings() as sup:
        sup.filter(VisibleDeprecationWarning, "Creating an ndarray from ragged")
        assert_raises(ValueError, _clean_inputs,
                      lp._replace(bounds=[(1, 2), (1, 2, 2)]))
    
    # 断言抛出 ValueError 异常，因为 bounds 参数列表中有三个元组，其中一个元组的长度不正确
    assert_raises(ValueError, _clean_inputs,
                  lp._replace(bounds=[(1, 2), (1, 2), (1, 2)]))

    # 更新 LP 问题对象，目标系数为 [1, 2, 3, 4]
    lp = _LPProblem(c=[1, 2, 3, 4])

    # 断言抛出 ValueError 异常，因为 bounds 参数列表中有两个元组，其中一个元组的长度不正确
    assert_raises(ValueError, _clean_inputs,
                  lp._replace(bounds=[(1, 2, 3, 4), (1, 2, 3, 4)]))


# 定义测试函数，用于测试良好的边界条件处理
def test_good_bounds():
    # 创建一个 LP 问题对象，目标系数为 [1, 2]
    lp = _LPProblem(c=[1, 2])

    # 测试默认情况下，使用 _clean_inputs 处理 LP 对象后，bounds 被设置为 [(0, np.inf)] * 2
    lp_cleaned = _clean_inputs(lp)  # lp.bounds 默认为 None
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)

    # 测试当 bounds 被设置为空列表时，bounds 被设置为 [(0, np.inf)] * 2
    lp_cleaned = _clean_inputs(lp._replace(bounds=[]))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)

    # 测试当 bounds 被设置为包含一个空列表的列表时，bounds 被设置为 [(0, np.inf)] * 2
    lp_cleaned = _clean_inputs(lp._replace(bounds=[[]]))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)

    # 测试当 bounds 被设置为一个元组 (1, 2) 时，bounds 被设置为 [(1, 2)] * 2
    lp_cleaned = _clean_inputs(lp._replace(bounds=(1, 2)))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 2)

    # 测试当 bounds 被设置为包含一个元组 (1, 2) 的列表时，bounds 被设置为 [(1, 2)] * 2
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, 2)]))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 2)

    # 测试当 bounds 被设置为一个元组 (1, None) 的列表时，bounds 被设置为 [(1, np.inf)] * 2
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, None)]))
    assert_equal(lp_cleaned.bounds, [(1, np.inf)] * 2)

    # 测试当 bounds 被设置为一个元组 (None, 1) 的列表时，bounds 被设置为 [(-np.inf, 1)] * 2
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, 1)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, 1)] * 2)

    # 测试当 bounds 被设置为包含两个元组 (None, None) 和 (-np.inf, None) 的列表时，bounds 被设置为 [(-np.inf, np.inf)] * 2
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, None), (-np.inf, None)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, np.inf)] * 2)

    # 更新 LP 问题对象，目标系数为 [1, 2, 3, 4]
    lp = _LPProblem(c=[1, 2, 3, 4])

    # 测试默认情况下，使用 _clean_inputs 处理 LP 对象后，bounds 被设置为 [(0, np.inf)] * 4
    lp_cleaned = _clean_inputs(lp)  # lp.bounds 默认为 None
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 4)

    # 测试当 bounds 被设置为一个元组 (1, 2) 时，bounds 被设置为 [(1, 2)] * 4
    lp_cleaned = _clean_inputs(lp._replace(bounds=(1, 2)))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 4)

    # 测试当 bounds 被设置为包含一个元组 (1, 2) 的列表时，bounds 被设置为 [(1, 2)] * 4
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, 2)]))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 4)

    # 测试当 bounds 被设置为一个元组 (1, None) 的列表时，bounds 被设置为 [(1, np.inf)] * 4
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, None)]))
    assert_equal(lp_cleaned.bounds, [(1, np.inf)] * 4)

    # 测试当 bounds 被设置为一个元组 (None, 1) 的列表时，bounds 被设置为 [(-np.inf, 1)] * 4
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, 1)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, 1)] * 4)

    # 测试当 bounds 被设置为包含四个元组的列表时，bounds 被设置为 [(-np.inf, np.inf)] * 4
    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, None),
                                                   (-np.inf, None),
                                                   (None, np.inf),
                                                   (-np.inf, np.inf)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, np.inf)] * 4)
```