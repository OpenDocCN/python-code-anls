# `D:\src\scipysrc\scipy\scipy\_lib\tests\test__util.py`

```
# 导入多进程相关模块中的进程池类
from multiprocessing import Pool
# 导入另一个命名冲突的进程池类，并用PWL作为别名
from multiprocessing.pool import Pool as PWL
# 导入正则表达式模块
import re
# 导入数学模块
import math
# 导入分数模块
from fractions import Fraction

# 导入NumPy库并使用np作为别名
import numpy as np
# 导入NumPy的测试模块中的断言函数
from numpy.testing import assert_equal, assert_
# 导入pytest模块，并使用raises作为别名引用assert_raises函数
import pytest
from pytest import raises as assert_raises
# 导入hypothesis模块中的部分子模块和函数
import hypothesis.extra.numpy as npst
# 导入hypothesis模块中的given、strategies、reproduce_failure函数
from hypothesis import given, strategies, reproduce_failure  # noqa: F401
# 从scipy的conftest模块中导入array_api_compatible和skip_xp_invalid_arg函数
from scipy.conftest import array_api_compatible, skip_xp_invalid_arg

# 从scipy库内部导入一些数组API相关函数和工具函数
from scipy._lib._array_api import (xp_assert_equal, xp_assert_close, is_numpy,
                                   copy as xp_copy, is_array_api_strict)
# 从scipy库内部导入一些工具函数
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
                              getfullargspec_no_self, FullArgSpec,
                              rng_integers, _validate_int, _rename_parameter,
                              _contains_nan, _rng_html_rewrite, _lazywhere)

# 创建一个pytest的标记，用于标记慢速测试
skip_xp_backends = pytest.mark.skip_xp_backends


@pytest.mark.slow
# 定义名为test__aligned_zeros的测试函数
def test__aligned_zeros():
    # 设置迭代次数
    niter = 10

    # 定义名为check的内部函数，用于检查_aligned_zeros函数的行为
    def check(shape, dtype, order, align):
        # 生成错误消息字符串
        err_msg = repr((shape, dtype, order, align))
        # 调用_aligned_zeros函数生成数组x
        x = _aligned_zeros(shape, dtype, order, align=align)
        # 如果align参数为None，则使用dtype的alignment
        if align is None:
            align = np.dtype(dtype).alignment
        # 断言数组x的数据指针地址能够被align整除
        assert_equal(x.__array_interface__['data'][0] % align, 0)
        # 如果shape具有长度属性，则断言x的形状与shape相等，否则与(shape,)相等
        if hasattr(shape, '__len__'):
            assert_equal(x.shape, shape, err_msg)
        else:
            assert_equal(x.shape, (shape,), err_msg)
        # 断言数组x的数据类型与dtype相等
        assert_equal(x.dtype, dtype)
        # 根据order参数的不同进行进一步的断言
        if order == "C":
            assert_(x.flags.c_contiguous, err_msg)
        elif order == "F":
            if x.size > 0:
                # 对于非空数组，断言其为Fortran连续存储
                assert_(x.flags.f_contiguous, err_msg)
        elif order is None:
            assert_(x.flags.c_contiguous, err_msg)
        else:
            # 抛出值错误异常，应对未知的order参数情况
            raise ValueError()

    # 针对不同的align、n、order、dtype和shape参数进行多层嵌套的测试
    for align in [1, 2, 3, 4, 8, 16, 32, 64, None]:
        for n in [0, 1, 3, 11]:
            for order in ["C", "F", None]:
                for dtype in [np.uint8, np.float64]:
                    for shape in [n, (1, 2, 3, n)]:
                        # 对每种组合进行niter次的测试
                        for j in range(niter):
                            check(shape, dtype, order, align)


# 定义名为test_check_random_state的测试函数
def test_check_random_state():
    # 如果seed参数为None，则返回由np.random使用的RandomState实例
    # 如果seed参数为整数，则返回一个以seed为种子的新RandomState实例
    # 如果seed参数已经是RandomState实例，则直接返回该实例
    # 否则抛出值错误异常
    rsi = check_random_state(1)
    assert_equal(type(rsi), np.random.RandomState)
    rsi = check_random_state(rsi)
    assert_equal(type(rsi), np.random.RandomState)
    rsi = check_random_state(None)
    assert_equal(type(rsi), np.random.RandomState)
    assert_raises(ValueError, check_random_state, 'a')
    rg = np.random.Generator(np.random.PCG64())
    rsi = check_random_state(rg)
    assert_equal(type(rsi), np.random.Generator)


# 定义名为test_getfullargspec_no_self的测试函数
def test_getfullargspec_no_self():
    # 创建 MapWrapper 类的实例 p，参数为 1
    p = MapWrapper(1)
    # 使用 getfullargspec_no_self 函数获取 p.__init__ 方法的参数规范
    argspec = getfullargspec_no_self(p.__init__)
    # 断言验证 p.__init__ 方法的参数规范是否符合预期
    assert_equal(argspec, FullArgSpec(['pool'], None, None, (1,), [],
                                      None, {}))
    # 使用 getfullargspec_no_self 函数获取 p.__call__ 方法的参数规范
    argspec = getfullargspec_no_self(p.__call__)
    # 断言验证 p.__call__ 方法的参数规范是否符合预期
    assert_equal(argspec, FullArgSpec(['func', 'iterable'], None, None, None,
                                      [], None, {}))

    # 定义内部类 _rv_generic
    class _rv_generic:
        # 定义 _rvs 方法，包含多种参数形式
        def _rvs(self, a, b=2, c=3, *args, size=None, **kwargs):
            return None

    # 创建 _rv_generic 类的实例 rv_obj
    rv_obj = _rv_generic()
    # 使用 getfullargspec_no_self 函数获取 rv_obj._rvs 方法的参数规范
    argspec = getfullargspec_no_self(rv_obj._rvs)
    # 断言验证 rv_obj._rvs 方法的参数规范是否符合预期
    assert_equal(argspec, FullArgSpec(['a', 'b', 'c'], 'args', 'kwargs',
                                      (2, 3), ['size'], {'size': None}, {}))
# 测试 MapWrapper 类在串行模式下的功能
def test_mapwrapper_serial():
    # 创建输入参数，包含从 0 到 9 的浮点数数组
    in_arg = np.arange(10.)
    # 计算输入数组的正弦值，作为预期输出
    out_arg = np.sin(in_arg)

    # 使用 MapWrapper 类创建对象 p，指定并行度为 1
    p = MapWrapper(1)
    # 断言 _mapfunc 属性是 Python 内置的 map 函数
    assert_(p._mapfunc is map)
    # 断言 pool 属性为 None，表示未使用外部进程池
    assert_(p.pool is None)
    # 断言 _own_pool 属性为 False，表示未拥有独立的进程池
    assert_(p._own_pool is False)
    # 使用 MapWrapper 对象 p 执行并行映射，计算输入数组的正弦值
    out = list(p(np.sin, in_arg))
    # 断言计算结果与预期输出相等
    assert_equal(out, out_arg)

    # 使用 assert_raises 断言在创建并行度为 0 的 MapWrapper 对象时会抛出 RuntimeError 异常
    with assert_raises(RuntimeError):
        p = MapWrapper(0)


# 测试 Pool 类在并行模式下的功能
def test_pool():
    # 使用 Pool 类创建具有 2 个进程的进程池对象 p
    with Pool(2) as p:
        # 使用进程池对象 p 并行计算列表中元素的正弦值
        p.map(math.sin, [1, 2, 3, 4])


# 测试 MapWrapper 类在并行模式下的功能
def test_mapwrapper_parallel():
    # 创建输入参数，包含从 0 到 9 的浮点数数组
    in_arg = np.arange(10.)
    # 计算输入数组的正弦值，作为预期输出
    out_arg = np.sin(in_arg)

    # 使用 MapWrapper 类创建对象 p，指定并行度为 2
    with MapWrapper(2) as p:
        # 使用 MapWrapper 对象 p 执行并行映射，计算输入数组的正弦值
        out = p(np.sin, in_arg)
        # 断言计算结果与预期输出相等
        assert_equal(list(out), out_arg)

        # 断言 _own_pool 属性为 True，表示拥有独立的进程池
        assert_(p._own_pool is True)
        # 断言 pool 属性为 PWL 类的实例，表示内部使用特定类型的进程池
        assert_(isinstance(p.pool, PWL))
        # 断言 _mapfunc 属性不为 None，表示映射函数已设置
        assert_(p._mapfunc is not None)

    # 使用 assert_raises 断言在上下文管理器关闭后调用 MapWrapper 对象 p 会抛出 ValueError 异常
    with assert_raises(Exception) as excinfo:
        p(np.sin, in_arg)

    assert_(excinfo.type is ValueError)

    # 创建具有 2 个进程的进程池对象 p
    with Pool(2) as p:
        # 使用 Pool 对象 p 创建 MapWrapper 对象 q，指定映射函数为 p.map
        q = MapWrapper(p.map)

        # 断言 _own_pool 属性为 False，表示未拥有独立的进程池
        assert_(q._own_pool is False)
        # 关闭 PoolWrapper 对象 q
        q.close()

        # 断言关闭 PoolWrapper 对象 q 不会关闭内部进程池 p
        # 因为 q 没有创建该进程池
        out = p.map(np.sin, in_arg)
        assert_equal(list(out), out_arg)


# 测试 rng_integers 函数生成随机整数数组的功能
def test_rng_integers():
    # 创建 NumPy 随机数生成器 rng
    rng = np.random.RandomState()

    # 测试生成的随机整数包含上限值
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 2
    assert arr.shape == (100, )

    # 测试生成的随机整数包含上限值
    arr = rng_integers(rng, low=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 0
    assert arr.shape == (100, )

    # 测试生成的随机整数不包含上限值
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 2
    assert arr.shape == (100, )

    # 测试生成的随机整数不包含上限值
    arr = rng_integers(rng, low=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 0
    assert arr.shape == (100, )

    # 使用 np.random.default_rng 创建随机数生成器 rng
    try:
        rng = np.random.default_rng()
    except AttributeError:
        return

    # 测试生成的随机整数包含上限值
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 2
    assert arr.shape == (100, )

    # 测试生成的随机整数包含上限值
    arr = rng_integers(rng, low=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 0
    assert arr.shape == (100, )

    # 测试生成的随机整数不包含上限值
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 2
    assert arr.shape == (100, )
    # 使用随机数生成器rng生成一个包含100个元素的整数数组，其中元素取值范围为[0, 4)
    arr = rng_integers(rng, low=5, size=100, endpoint=False)
    
    # 断言：数组arr的最大值应该是4
    assert np.max(arr) == 4
    
    # 断言：数组arr的最小值应该是0
    assert np.min(arr) == 0
    
    # 断言：数组arr的形状应该是(100,)
    assert arr.shape == (100, )
class TestValidateInt:
    # 测试用例类 TestValidateInt

    @pytest.mark.parametrize('n', [4, np.uint8(4), np.int16(4), np.array(4)])
    # 使用 pytest 的参数化装饰器，对参数 n 进行多组测试
    def test_validate_int(self, n):
        # 测试函数 test_validate_int，验证 _validate_int 函数对整数的验证
        n = _validate_int(n, 'n')
        # 断言 n 的返回值应为 4
        assert n == 4

    @pytest.mark.parametrize('n', [4.0, np.array([4]), Fraction(4, 1)])
    # 使用 pytest 的参数化装饰器，对参数 n 进行多组测试
    def test_validate_int_bad(self, n):
        # 测试函数 test_validate_int_bad，验证 _validate_int 函数对非整数的处理
        with pytest.raises(TypeError, match='n must be an integer'):
            # 断言调用 _validate_int 函数会引发 TypeError 异常，异常消息应包含 'n must be an integer'
            _validate_int(n, 'n')

    def test_validate_int_below_min(self):
        # 测试函数 test_validate_int_below_min，验证 _validate_int 函数对小于零的整数处理
        with pytest.raises(ValueError, match='n must be an integer not less than 0'):
            # 断言调用 _validate_int 函数会引发 ValueError 异常，异常消息应包含 'n must be an integer not less than 0'
            _validate_int(-1, 'n', 0)


class TestRenameParameter:
    # 测试用例类 TestRenameParameter
    # 检查 `_rename_parameter` 包装器对向后兼容关键字重命名的正确工作

    # 仍接受关键字 `old` 的示例方法/函数
    @_rename_parameter("old", "new")
    # 使用 `_rename_parameter` 装饰器将 old 参数重命名为 new
    def old_keyword_still_accepted(self, new):
        # 方法 old_keyword_still_accepted，返回参数 new
        return new

    # 关键字 `old` 已弃用的示例方法/函数
    @_rename_parameter("old", "new", dep_version="1.9.0")
    # 使用 `_rename_parameter` 装饰器将 old 参数重命名为 new，并指定弃用版本为 1.9.0
    def old_keyword_deprecated(self, new):
        # 方法 old_keyword_deprecated，返回参数 new
        return new

    def test_old_keyword_still_accepted(self):
        # 测试函数 test_old_keyword_still_accepted

        # 位置参数和两个关键字应完全相同工作
        res1 = self.old_keyword_still_accepted(10)
        res2 = self.old_keyword_still_accepted(new=10)
        res3 = self.old_keyword_still_accepted(old=10)
        # 断言 res1、res2、res3 都等于 10
        assert res1 == res2 == res3 == 10

        # 意外的关键字引发错误
        message = re.escape("old_keyword_still_accepted() got an unexpected")
        with pytest.raises(TypeError, match=message):
            # 断言调用 old_keyword_still_accepted 函数会引发 TypeError 异常，异常消息符合正则表达式 message
            self.old_keyword_still_accepted(unexpected=10)

        # 同一参数的多个值引发错误
        message = re.escape("old_keyword_still_accepted() got multiple")
        with pytest.raises(TypeError, match=message):
            # 断言调用 old_keyword_still_accepted 函数会引发 TypeError 异常，异常消息符合正则表达式 message
            self.old_keyword_still_accepted(10, new=10)
        with pytest.raises(TypeError, match=message):
            # 断言调用 old_keyword_still_accepted 函数会引发 TypeError 异常，异常消息符合正则表达式 message
            self.old_keyword_still_accepted(10, old=10)
        with pytest.raises(TypeError, match=message):
            # 断言调用 old_keyword_still_accepted 函数会引发 TypeError 异常，异常消息符合正则表达式 message
            self.old_keyword_still_accepted(new=10, old=10)
    # 定义测试方法，用于测试旧关键字在方法中的使用情况
    def test_old_keyword_deprecated(self):
        # positional argument and both keyword work identically,
        # but use of old keyword results in DeprecationWarning
        # 定义提示信息，指出使用旧关键字参数 `old` 已经被废弃
        dep_msg = "Use of keyword argument `old` is deprecated"
        # 调用方法，传入位置参数 10，返回结果赋给 res1
        res1 = self.old_keyword_deprecated(10)
        # 调用方法，传入关键字参数 new=10，返回结果赋给 res2
        res2 = self.old_keyword_deprecated(new=10)
        # 使用旧关键字 old=10 调用方法，在抛出 DeprecationWarning 的情况下返回结果赋给 res3
        with pytest.warns(DeprecationWarning, match=dep_msg):
            res3 = self.old_keyword_deprecated(old=10)
        # 断言三次调用的结果均为 10
        assert res1 == res2 == res3 == 10

        # unexpected keyword raises an error
        # 定义错误消息，指出方法不应接受未预期的关键字参数
        message = re.escape("old_keyword_deprecated() got an unexpected")
        # 使用 pytest.raises 检测到 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=message):
            self.old_keyword_deprecated(unexpected=10)

        # multiple values for the same parameter raises an error and,
        # if old keyword is used, results in DeprecationWarning
        # 定义错误消息，指出方法不能接受同一参数的多个值
        message = re.escape("old_keyword_deprecated() got multiple")
        # 使用 pytest.raises 检测到 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=message):
            self.old_keyword_deprecated(10, new=10)
        # 同时使用旧关键字 old=10 调用方法，在抛出 DeprecationWarning 的情况下，再次检测到 TypeError 异常
        with pytest.raises(TypeError, match=message), \
                pytest.warns(DeprecationWarning, match=dep_msg):
            self.old_keyword_deprecated(10, old=10)
        # 同时使用关键字参数 new=10 和 old=10 调用方法，在抛出 DeprecationWarning 的情况下，再次检测到 TypeError 异常
        with pytest.raises(TypeError, match=message), \
                pytest.warns(DeprecationWarning, match=dep_msg):
            self.old_keyword_deprecated(new=10, old=10)
class TestContainsNaNTest:

    def test_policy(self):
        # 创建一个包含 NaN 值的 NumPy 数组
        data = np.array([1, 2, 3, np.nan])

        # 测试 nan_policy="propagate" 时的函数调用结果
        contains_nan, nan_policy = _contains_nan(data, nan_policy="propagate")
        assert contains_nan  # 确保包含 NaN 值
        assert nan_policy == "propagate"  # 确保 nan_policy 被正确传播

        # 测试 nan_policy="omit" 时的函数调用结果
        contains_nan, nan_policy = _contains_nan(data, nan_policy="omit")
        assert contains_nan  # 确保包含 NaN 值
        assert nan_policy == "omit"  # 确保 NaN 值被省略

        # 测试 nan_policy="raise" 时应抛出 ValueError 异常
        msg = "The input contains nan values"
        with pytest.raises(ValueError, match=msg):
            _contains_nan(data, nan_policy="raise")

        # 测试 nan_policy="nan" 时应抛出 ValueError 异常
        msg = "nan_policy must be one of"
        with pytest.raises(ValueError, match=msg):
            _contains_nan(data, nan_policy="nan")

    def test_contains_nan(self):
        # 测试不包含 NaN 值的数组
        data1 = np.array([1, 2, 3])
        assert not _contains_nan(data1)[0]

        # 测试包含 NaN 值的数组
        data2 = np.array([1, 2, 3, np.nan])
        assert _contains_nan(data2)[0]

        # 测试包含多个 NaN 值的数组
        data3 = np.array([np.nan, 2, 3, np.nan])
        assert _contains_nan(data3)[0]

        # 测试二维数组，不包含 NaN 值
        data4 = np.array([[1, 2], [3, 4]])
        assert not _contains_nan(data4)[0]

        # 测试二维数组，包含 NaN 值
        data5 = np.array([[1, 2], [3, np.nan]])
        assert _contains_nan(data5)[0]

    @skip_xp_invalid_arg
    def test_contains_nan_with_strings(self):
        # 测试包含字符串的数组，"nan" 被转换为字符串
        data1 = np.array([1, 2, "3", np.nan])  # converted to string "nan"
        assert not _contains_nan(data1)[0]

        # 测试包含字符串的数组，使用 'object' 数据类型
        data2 = np.array([1, 2, "3", np.nan], dtype='object')
        assert _contains_nan(data2)[0]

        # 测试二维数组包含字符串，"nan" 被转换为字符串
        data3 = np.array([["1", 2], [3, np.nan]])  # converted to string "nan"
        assert not _contains_nan(data3)[0]

        # 测试二维数组包含字符串，使用 'object' 数据类型
        data4 = np.array([["1", 2], [3, np.nan]], dtype='object')
        assert _contains_nan(data4)[0]

    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    @pytest.mark.parametrize("nan_policy", ['propagate', 'omit', 'raise'])
    def test_array_api(self, xp, nan_policy):
        # 生成随机数数组 x0，并将其转换为 xp 数组（可能是 NumPy 或者 JAX）
        rng = np.random.default_rng(932347235892482)
        x0 = rng.random(size=(2, 3, 4))
        x = xp.asarray(x0)
        x_nan = xp_copy(x, xp=xp)
        x_nan[1, 2, 1] = np.nan

        # 测试 _contains_nan 函数对数组 x 的返回结果
        contains_nan, nan_policy_out = _contains_nan(x, nan_policy=nan_policy)
        assert not contains_nan  # 确保不包含 NaN 值
        assert nan_policy_out == nan_policy  # 确保 nan_policy 被正确传播

        # 根据 nan_policy 的不同情况进行进一步测试
        if nan_policy == 'raise':
            message = 'The input contains...'
            with pytest.raises(ValueError, match=message):
                _contains_nan(x_nan, nan_policy=nan_policy)
        elif nan_policy == 'omit' and not is_numpy(xp):
            message = "`nan_policy='omit' is incompatible..."
            with pytest.raises(ValueError, match=message):
                _contains_nan(x_nan, nan_policy=nan_policy)
        elif nan_policy == 'propagate':
            # 再次测试 _contains_nan 函数对包含 NaN 值的数组 x_nan 的返回结果
            contains_nan, nan_policy_out = _contains_nan(
                x_nan, nan_policy=nan_policy)
            assert contains_nan  # 确保包含 NaN 值
            assert nan_policy_out == nan_policy  # 确保 nan_policy 被正确传播
# 定义测试函数 test__rng_html_rewrite()
def test__rng_html_rewrite():
    
    # 定义内部函数 mock_str()，返回预设的字符串列表
    def mock_str():
        lines = [
            'np.random.default_rng(8989843)',  # 第一个字符串
            'np.random.default_rng(seed)',    # 第二个字符串，包含变量 seed
            'np.random.default_rng(0x9a71b21474694f919882289dc1559ca)',  # 第三个字符串
            ' bob ',                          # 第四个字符串，不包含 np.random.default_rng() 调用
        ]
        return lines

    # 调用 _rng_html_rewrite() 函数，并执行其返回的结果
    res = _rng_html_rewrite(mock_str)()
    
    # 预期的输出结果列表
    ref = [
        'np.random.default_rng()',          # 预期输出结果的第一个字符串
        'np.random.default_rng(seed)',      # 预期输出结果的第二个字符串，保持不变
        'np.random.default_rng()',          # 预期输出结果的第三个字符串
        ' bob ',                            # 预期输出结果的第四个字符串，不变
    ]

    # 断言实际结果与预期结果相符
    assert res == ref


# 定义测试类 TestLazywhere
class TestLazywhere:
    
    # 定义类属性 n_arrays，使用 strategies.integers() 生成一个介于 1 到 3 之间的整数
    n_arrays = strategies.integers(min_value=1, max_value=3)
    
    # 定义类属性 rng_seed，使用 strategies.integers() 生成一个 10 位数的随机整数
    rng_seed = strategies.integers(min_value=1000000000, max_value=9999999999)
    
    # 定义类属性 dtype，从 np.float32 和 np.float64 中随机选择一个数据类型
    dtype = strategies.sampled_from((np.float32, np.float64))
    
    # 定义类属性 p，使用 strategies.floats() 生成一个介于 0 到 1 之间的浮点数
    p = strategies.floats(min_value=0, max_value=1)
    
    # 定义类属性 data，使用 strategies.data() 生成数据
    data = strategies.data()

    # 使用 pytest.mark.fail_slow(10) 标记的测试方法
    @pytest.mark.fail_slow(10)
    
    # 忽略 RuntimeWarning 的警告
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    
    # 跳过特定的 XP 后端，原因是 JAX arrays 不支持项赋值
    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    
    # 使用 fixture "skip_xp_backends"
    @pytest.mark.usefixtures("skip_xp_backends")
    
    # 标记为兼容数组 API
    @array_api_compatible
    
    # 使用给定的参数执行测试方法
    @given(n_arrays=n_arrays, rng_seed=rng_seed, dtype=dtype, p=p, data=data)
    # 定义一个测试函数，用于测试_lazywhere函数的基本功能
    def test_basic(self, n_arrays, rng_seed, dtype, p, data, xp):
        # 生成互相广播的形状列表，并获取输入形状和结果形状
        mbs = npst.mutually_broadcastable_shapes(num_shapes=n_arrays+1,
                                                 min_side=0)
        input_shapes, result_shape = data.draw(mbs)
        # 解包条件形状和其余形状
        cond_shape, *shapes = input_shapes
        # 定义一个元素字典，用于填充值，并且禁止子正常值
        elements = {'allow_subnormal': False}  # cupy/cupy#8382
        # 从数据生成一个填充值，并将其转换为xp的数组格式
        fillvalue = xp.asarray(data.draw(npst.arrays(dtype=dtype, shape=tuple(),
                                                     elements=elements)))
        # 将填充值转换为浮点数格式
        float_fillvalue = float(fillvalue)
        # 生成多个数组，每个数组的形状从shapes中选择
        arrays = [xp.asarray(data.draw(npst.arrays(dtype=dtype, shape=shape)))
                  for shape in shapes]

        # 定义一个简单的求和函数
        def f(*args):
            return sum(arg for arg in args)

        # 定义一个求和后除以2的函数
        def f2(*args):
            return sum(arg for arg in args) / 2

        # 使用给定的随机种子生成一个随机数生成器
        rng = np.random.default_rng(rng_seed)
        # 生成一个条件数组，判断是否大于概率p，并转换为xp的数组格式
        cond = xp.asarray(rng.random(size=cond_shape) > p)

        # 调用_lazywhere函数，计算结果res1、res2和res3
        res1 = _lazywhere(cond, arrays, f, fillvalue)
        res2 = _lazywhere(cond, arrays, f, f2=f2)
        if not is_array_api_strict(xp):
            res3 = _lazywhere(cond, arrays, f, float_fillvalue)

        # 确保数组至少是1维，以遵循类型提升规则。当最低支持的NumPy版本是2.0时，可以移除此部分代码。
        if xp == np:
            cond, fillvalue, *arrays = np.atleast_1d(cond, fillvalue, *arrays)

        # 计算参考结果ref1、ref2和ref3，用于验证_lazywhere函数的输出
        ref1 = xp.where(cond, f(*arrays), fillvalue)
        ref2 = xp.where(cond, f(*arrays), f2(*arrays))
        if not is_array_api_strict(xp):
            # 当fillvalue是Python标量时，Array API标准当前未定义行为。当定义后，可以使用array_api_strict运行测试。
            ref3 = xp.where(cond, f(*arrays), float_fillvalue)

        # 如果xp是NumPy，因为我们确保了数组至少是1维
        if xp == np:
            # 重新整形ref1、ref2和ref3，以匹配结果形状
            ref1 = ref1.reshape(result_shape)
            ref2 = ref2.reshape(result_shape)
            ref3 = ref3.reshape(result_shape)

        # 断言测试结果与参考结果的接近程度
        xp_assert_close(res1, ref1, rtol=2e-16, allow_0d=True)
        xp_assert_equal(res2, ref2, allow_0d=True)
        if not is_array_api_strict(xp):
            xp_assert_equal(res3, ref3, allow_0d=True)
```