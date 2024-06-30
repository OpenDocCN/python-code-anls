# `D:\src\scipysrc\scipy\scipy\special\tests\test_sf_error.py`

```
# 导入必要的模块和库
import sys  # 系统模块
import warnings  # 警告模块

import numpy as np  # 导入 NumPy 库，命名为 np
from numpy.testing import assert_, assert_equal, IS_PYPY  # 导入 NumPy 测试模块中的函数和变量
import pytest  # 导入 pytest 测试框架
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 函数，命名为 assert_raises

import scipy.special as sc  # 导入 SciPy 的 special 子模块，命名为 sc
from scipy.special._ufuncs import _sf_error_test_function  # 导入特定函数 _sf_error_test_function

# 定义特定错误码到错误类型的映射字典
_sf_error_code_map = {
    'singular': 1,
    'underflow': 2,
    'overflow': 3,
    'slow': 4,
    'loss': 5,
    'no_result': 6,
    'domain': 7,
    'arg': 8,
    'other': 9
}

# 定义可用的错误处理动作列表
_sf_error_actions = [
    'ignore',
    'warn',
    'raise'
]

# 检查特定函数在给定参数和错误处理动作下的行为
def _check_action(fun, args, action):
    # 将参数转换为 NumPy 的 long 类型数组
    args = np.asarray(args, dtype=np.dtype("long"))
    if action == 'warn':
        # 如果动作为 'warn'，则检查是否发出特定警告
        with pytest.warns(sc.SpecialFunctionWarning):
            fun(*args)
    elif action == 'raise':
        # 如果动作为 'raise'，则检查是否抛出特定异常
        with assert_raises(sc.SpecialFunctionError):
            fun(*args)
    else:
        # 如果动作为 'ignore'，确保没有警告或异常被触发
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fun(*args)

# 测试函数：检查当前错误设置是否在有效范围内
def test_geterr():
    err = sc.geterr()
    for key, value in err.items():
        assert_(key in _sf_error_code_map)  # 检查错误类型是否在预定义的字典中
        assert_(value in _sf_error_actions)  # 检查错误处理动作是否在预定义的列表中

# 测试函数：设置和恢复不同错误处理动作的状态
def test_seterr():
    entry_err = sc.geterr()
    try:
        for category, error_code in _sf_error_code_map.items():
            for action in _sf_error_actions:
                geterr_olderr = sc.geterr()
                seterr_olderr = sc.seterr(**{category: action})
                assert_(geterr_olderr == seterr_olderr)
                newerr = sc.geterr()
                assert_(newerr[category] == action)
                geterr_olderr.pop(category)
                newerr.pop(category)
                assert_(geterr_olderr == newerr)
                _check_action(_sf_error_test_function, (error_code,), action)
    finally:
        sc.seterr(**entry_err)

# 跳过条件性测试：PyPy 下不执行此测试
@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_sf_error_special_refcount():
    # 检查特定条件下的回归测试
    # 检查在引发 SpecialFunctionError 时，scipy.special 的引用计数未增加
    refcount_before = sys.getrefcount(sc)
    with sc.errstate(all='raise'):
        with pytest.raises(sc.SpecialFunctionError, match='domain error'):
            sc.ndtri(2.0)
    refcount_after = sys.getrefcount(sc)
    assert refcount_after == refcount_before

# 测试函数：基本的错误状态管理测试（Python 实现）
def test_errstate_pyx_basic():
    olderr = sc.geterr()
    with sc.errstate(singular='raise'):
        with assert_raises(sc.SpecialFunctionError):
            sc.loggamma(0)
    assert_equal(olderr, sc.geterr())

# 测试函数：基本的错误状态管理测试（C 实现）
def test_errstate_c_basic():
    olderr = sc.geterr()
    with sc.errstate(domain='raise'):
        with assert_raises(sc.SpecialFunctionError):
            sc.spence(-1)
    assert_equal(olderr, sc.geterr())

# 测试函数：基本的错误状态管理测试（C++ 实现）
def test_errstate_cpp_basic():
    olderr = sc.geterr()
    # 进入一个上下文管理器，设置特定数学函数的错误处理状态为引发下溢错误
    with sc.errstate(underflow='raise'):
        # 在特定错误条件下，使用 assert_raises 检查特殊函数调用是否引发了特定的异常
        with assert_raises(sc.SpecialFunctionError):
            # 调用 wrightomega 函数，传入参数 -1000
            sc.wrightomega(-1000)
    # 确保上下文管理器结束后，错误状态已经恢复到旧的状态
    assert_equal(olderr, sc.geterr())
# 测试函数：验证当特定错误状态设置时，特定的 SciPy 特殊函数会引发异常

def test_errstate_cpp_scipy_special():
    # 保存当前的错误状态
    olderr = sc.geterr()
    # 设置特定错误状态：当出现奇异性时抛出异常
    with sc.errstate(singular='raise'):
        # 使用 assert_raises 断言，验证调用 lambertw(0, 1) 时会引发 sc.SpecialFunctionError 异常
        with assert_raises(sc.SpecialFunctionError):
            sc.lambertw(0, 1)
    # 断言：当前的错误状态与进入测试前保持一致
    assert_equal(olderr, sc.geterr())


def test_errstate_cpp_alt_ufunc_machinery():
    # 保存当前的错误状态
    olderr = sc.geterr()
    # 设置特定错误状态：当出现奇异性时抛出异常
    with sc.errstate(singular='raise'):
        # 使用 assert_raises 断言，验证调用 gammaln(0) 时会引发 sc.SpecialFunctionError 异常
        with assert_raises(sc.SpecialFunctionError):
            sc.gammaln(0)
    # 断言：当前的错误状态与进入测试前保持一致
    assert_equal(olderr, sc.geterr())


def test_errstate():
    # 遍历所有错误类别和对应的错误码
    for category, error_code in _sf_error_code_map.items():
        # 遍历所有错误操作
        for action in _sf_error_actions:
            # 保存当前的错误状态
            olderr = sc.geterr()
            # 设置特定的错误状态
            with sc.errstate(**{category: action}):
                # 调用 _check_action 函数，检查特定条件下的错误行为
                _check_action(_sf_error_test_function, (error_code,), action)
            # 断言：当前的错误状态与进入测试前保持一致
            assert_equal(olderr, sc.geterr())


def test_errstate_all_but_one():
    # 保存当前的错误状态
    olderr = sc.geterr()
    # 设置所有错误状态为 'raise'，除奇异性错误外都忽略
    with sc.errstate(all='raise', singular='ignore'):
        # 调用 gammaln(0)，不应该引发异常
        sc.gammaln(0)
        # 使用 assert_raises 断言，验证调用 spence(-1.0) 时会引发 sc.SpecialFunctionError 异常
        with assert_raises(sc.SpecialFunctionError):
            sc.spence(-1.0)
    # 断言：当前的错误状态与进入测试前保持一致
    assert_equal(olderr, sc.geterr())
```