# `.\numpy\numpy\_core\tests\test_nep50_promotions.py`

```py
"""
This file adds basic tests to test the NEP 50 style promotion compatibility
mode.  Most of these test are likely to be simply deleted again once NEP 50
is adopted in the main test suite.  A few may be moved elsewhere.
"""

# 引入需要的库和模块
import operator  # 操作符模块，用于进行运算操作
import threading  # 线程模块，用于多线程支持
import warnings  # 警告模块，用于处理警告信息

import numpy as np  # 引入NumPy库，命名为np，用于数值计算

import pytest  # 引入pytest测试框架
import hypothesis  # 引入假设测试框架
from hypothesis import strategies  # 引入假设测试的策略

from numpy.testing import assert_array_equal, IS_WASM  # 引入NumPy的测试工具和平台判断


@pytest.fixture(scope="module", autouse=True)
def _weak_promotion_enabled():
    # 设置模块级别的夯升状态为"weak_and_warn"，并在测试完成后恢复到原状态
    state = np._get_promotion_state()
    np._set_promotion_state("weak_and_warn")
    yield
    np._set_promotion_state(state)


@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for fp errors")
def test_nep50_examples():
    # 测试NEP 50样式的推广模式是否正常工作
    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.uint8(1) + 2
    assert res.dtype == np.uint8  # 断言结果的数据类型为np.uint8

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1], np.uint8) + np.int64(1)
    assert res.dtype == np.int64  # 断言结果的数据类型为np.int64

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1], np.uint8) + np.array(1, dtype=np.int64)
    assert res.dtype == np.int64  # 断言结果的数据类型为np.int64

    with pytest.warns(UserWarning, match="result dtype changed"):
        # 注意: 对于"weak_and_warn"推广状态，不幸的是不会给出溢出警告（因为使用了完整的数组路径）。
        with np.errstate(over="raise"):
            res = np.uint8(100) + 200
    assert res.dtype == np.uint8  # 断言结果的数据类型为np.uint8

    with pytest.warns(Warning) as recwarn:
        res = np.float32(1) + 3e100

    # 检查在一次调用中是否给出了两个警告：
    warning = str(recwarn.pop(UserWarning).message)
    assert warning.startswith("result dtype changed")
    warning = str(recwarn.pop(RuntimeWarning).message)
    assert warning.startswith("overflow")
    assert len(recwarn) == 0  # 没有更多的警告
    assert np.isinf(res)
    assert res.dtype == np.float32  # 断言结果的数据类型为np.float32

    # 发生了变化，但我们不会为此发出警告（太吵闹了）
    res = np.array([0.1], np.float32) == np.float64(0.1)
    assert res[0] == False  # 断言结果为False

    # 由于上述操作消除了警告，因此进行额外的测试：
    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([0.1], np.float32) + np.float64(0.1)
    assert res.dtype == np.float64  # 断言结果的数据类型为np.float64

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1.], np.float32) + np.int64(3)
    assert res.dtype == np.float64  # 断言结果的数据类型为np.float64


@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
def test_nep50_weak_integers(dtype):
    # 测试NEP 50中弱类型整数的表现
    # 设置推广状态为"weak"，避免警告（对标量使用不同的代码路径）
    np._set_promotion_state("weak")
    scalar_type = np.dtype(dtype).type

    maxint = int(np.iinfo(dtype).max)

    with np.errstate(over="warn"):
        with pytest.warns(RuntimeWarning):
            res = scalar_type(100) + maxint
    assert res.dtype == dtype  # 断言结果的数据类型为输入的dtype

    # 预期数组操作不会引发警告，但结果应相同
    # 创建一个包含单个元素 100 的 NumPy 数组，指定数据类型为参数 dtype 所指定的类型
    res = np.array(100, dtype=dtype) + maxint
    # 断言语句，检查 res 数组的数据类型是否与参数 dtype 指定的类型相同
    assert res.dtype == dtype
# 使用 pytest.mark.parametrize 装饰器为 test_nep50_weak_integers_with_inexact 函数参数 dtype 提供多个测试参数
@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
def test_nep50_weak_integers_with_inexact(dtype):
    # 设置 NumPy 的类型提升状态为 "weak"
    np._set_promotion_state("weak")
    # 将 dtype 转换为 NumPy 的数据类型对象，并获取其标量类型
    scalar_type = np.dtype(dtype).type

    # 计算大于 dtype 的最大值的两倍
    too_big_int = int(np.finfo(dtype).max) * 2

    if dtype in "dDG":
        # 对于 'd', 'D', 'G' 类型，这些类型目前在内部转换为 Python 浮点数，会引发 OverflowError
        # 其他类型会溢出到 inf
        # 注意：统一行为可能更合理！
        with pytest.raises(OverflowError):
            scalar_type(1) + too_big_int

        with pytest.raises(OverflowError):
            np.array(1, dtype=dtype) + too_big_int
    else:
        # NumPy 使用 `int -> string -> longdouble` 进行转换。但是对于巨大的整数，Python 可能会拒绝 `str(int)` 转换。
        # 在这种情况下，RuntimeWarning 是正确的，但转换会更早失败（似乎发生在 32 位 Linux 上，可能仅在调试模式下）。
        if dtype in "gG":
            try:
                str(too_big_int)
            except ValueError:
                pytest.skip("`huge_int -> string -> longdouble` 失败")

        # 否则，我们会溢出到无穷大：
        with pytest.warns(RuntimeWarning):
            res = scalar_type(1) + too_big_int
        assert res.dtype == dtype
        assert res == np.inf

        with pytest.warns(RuntimeWarning):
            # 这里强制指定 dtype，因为在 Windows 上可能会选择 double 而不是 longdouble 循环。
            # 这会导致稍微不同的结果（int 的转换会失败，如上所述）。
            res = np.add(np.array(1, dtype=dtype), too_big_int, dtype=dtype)
        assert res.dtype == dtype
        assert res == np.inf


# 使用 pytest.mark.parametrize 装饰器为 test_weak_promotion_scalar_path 函数参数 op 提供多个测试参数
@pytest.mark.parametrize("op", [operator.add, operator.pow])
def test_weak_promotion_scalar_path(op):
    # 一些额外的路径，测试弱类型提升的标量
    np._set_promotion_state("weak")

    # 整数路径：
    res = op(np.uint8(3), 5)
    assert res == op(3, 5)
    assert res.dtype == np.uint8 or res.dtype == bool

    with pytest.raises(OverflowError):
        op(np.uint8(3), 1000)

    # 浮点路径：
    res = op(np.float32(3), 5.)
    assert res == op(3., 5.)
    assert res.dtype == np.float32 or res.dtype == bool


# 测试 nep50_complex_promotion 函数，测试复杂类型的类型提升
def test_nep50_complex_promotion():
    np._set_promotion_state("weak")

    with pytest.warns(RuntimeWarning, match=".*overflow"):
        res = np.complex64(3) + complex(2**300)

    assert type(res) == np.complex64


# 测试 nep50_integer_conversion_errors 函数，测试整数转换错误
def test_nep50_integer_conversion_errors():
    # 这里不用担心警告（自动 fixture 会重置）。
    np._set_promotion_state("weak")
    # 错误路径的实现大部分是缺失的（截至撰写时）
    with pytest.raises(OverflowError, match=".*uint8"):
        np.array([1], np.uint8) + 300

    with pytest.raises(OverflowError, match=".*uint8"):
        np.uint8(1) + 300
    # 使用 pytest 的断言来测试是否会抛出 OverflowError 异常
    with pytest.raises(OverflowError,
            match="Python integer -1 out of bounds for uint8"):
        # 尝试将 np.uint8 类型的整数 1 加上 -1，预期会引发溢出错误
        np.uint8(1) + -1
def test_nep50_integer_regression():
    # 测试旧的整数提升规则。当整数过大时，需要继续使用旧式提升规则。
    np._set_promotion_state("legacy")
    # 创建一个包含整数1的 NumPy 数组
    arr = np.array(1)
    # 断言：加上 2 的 63 次方的结果的数据类型应为 np.float64
    assert (arr + 2**63).dtype == np.float64
    # 断言：使用 arr[()] 加上 2 的 63 次方的结果的数据类型应为 np.float64
    assert (arr[()] + 2**63).dtype == np.float64


def test_nep50_with_axisconcatenator():
    # 在 1.25 版本的发布说明中，承诺这将来会成为一个错误；测试这一点（NEP 50 的选择使得弃用成为错误）。
    np._set_promotion_state("weak")

    # 使用 pytest 检查是否会抛出 OverflowError 异常
    with pytest.raises(OverflowError):
        np.r_[np.arange(5, dtype=np.int8), 255]


@pytest.mark.parametrize("ufunc", [np.add, np.power])
@pytest.mark.parametrize("state", ["weak", "weak_and_warn"])
def test_nep50_huge_integers(ufunc, state):
    # 非常大的整数会很复杂，因为它们会变成 uint64 或者对象类型。这些测试覆盖了几种可能的情况（其中一些不能给出 NEP 50 警告）。
    np._set_promotion_state(state)

    # 使用 pytest 检查是否会抛出 OverflowError 异常
    with pytest.raises(OverflowError):
        ufunc(np.int64(0), 2**63)  # 2**63 对于 int64 来说太大了

    if state == "weak_and_warn":
        # 使用 pytest 检查是否会抛出 UserWarning，并匹配指定的字符串
        with pytest.warns(UserWarning,
                match="result dtype changed.*float64.*uint64"):
            with pytest.raises(OverflowError):
                ufunc(np.uint64(0), 2**64)
    else:
        # 使用 pytest 检查是否会抛出 OverflowError 异常
        with pytest.raises(OverflowError):
            ufunc(np.uint64(0), 2**64)  # 2**64 无法被 uint64 表示

    # 然而，2**63 可以被 uint64 表示（并且会被使用）：
    if state == "weak_and_warn":
        # 使用 pytest 检查是否会抛出 UserWarning，并匹配指定的字符串
        with pytest.warns(UserWarning,
                match="result dtype changed.*float64.*uint64"):
            res = ufunc(np.uint64(1), 2**63)
    else:
        res = ufunc(np.uint64(1), 2**63)

    assert res.dtype == np.uint64
    assert res == ufunc(1, 2**63, dtype=object)

    # 以下路径未能正确警告关于类型变化的情况：
    with pytest.raises(OverflowError):
        ufunc(np.int64(1), 2**63)  # np.array(2**63) 会变成 uint

    with pytest.raises(OverflowError):
        ufunc(np.int64(1), 2**100)  # np.array(2**100) 会变成对象类型

    # 这会变成对象类型，因此是 Python 的 float，而不是 NumPy 的 float：
    res = ufunc(1.0, 2**100)
    assert isinstance(res, np.float64)


def test_nep50_in_concat_and_choose():
    np._set_promotion_state("weak_and_warn")

    # 使用 pytest 检查是否会抛出 UserWarning，并匹配指定的字符串
    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.concatenate([np.float32(1), 1.], axis=None)
    assert res.dtype == "float32"

    # 使用 pytest 检查是否会抛出 UserWarning，并匹配指定的字符串
    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.choose(1, [np.float32(1), 1.])
    assert res.dtype == "float32"
@pytest.mark.parametrize("expected,dtypes,optional_dtypes", [
        # 参数化测试用例：定义期望结果、主要数据类型、可选数据类型
        (np.float32, [np.float32],
            [np.float16, 0.0, np.uint16, np.int16, np.int8, 0]),
        (np.complex64, [np.float32, 0j],
            [np.float16, 0.0, np.uint16, np.int16, np.int8, 0]),
        (np.float32, [np.int16, np.uint16, np.float16],
            [np.int8, np.uint8, np.float32, 0., 0]),
        (np.int32, [np.int16, np.uint16],
            [np.int8, np.uint8, 0, np.bool]),
        ])
@hypothesis.given(data=strategies.data())
def test_expected_promotion(expected, dtypes, optional_dtypes, data):
    np._set_promotion_state("weak")

    # 设置 NumPy 的类型提升状态为弱
    # 从数据中随机抽取以确保 "dtypes" 总是存在：
    optional = data.draw(strategies.lists(
            strategies.sampled_from(dtypes + optional_dtypes)))
    all_dtypes = dtypes + optional
    dtypes_sample = data.draw(strategies.permutations(all_dtypes))

    # 通过抽取的数据类型样本计算结果类型
    res = np.result_type(*dtypes_sample)
    assert res == expected


@pytest.mark.parametrize("sctype",
        # 参数化测试用例：定义整数类型
        [np.int8, np.int16, np.int32, np.int64,
         np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("other_val",
        # 参数化测试用例：定义不同的比较值
        [-2*100, -1, 0, 9, 10, 11, 2**63, 2*100])
@pytest.mark.parametrize("comp",
        # 参数化测试用例：定义比较运算符
        [operator.eq, operator.ne, operator.le, operator.lt,
         operator.ge, operator.gt])
def test_integer_comparison(sctype, other_val, comp):
    np._set_promotion_state("weak")

    # 测试整数（特别是超出范围的整数）的比较是否正确
    val_obj = 10
    val = sctype(val_obj)
    # 检查标量与 Python 整数的比较行为是否一致
    assert comp(10, other_val) == comp(val, other_val)
    assert comp(val, other_val) == comp(10, other_val)
    # 除了结果类型不同：
    assert type(comp(val, other_val)) is np.bool

    # 检查整数数组和对象数组的比较行为是否一致
    val_obj = np.array([10, 10], dtype=object)
    val = val_obj.astype(sctype)
    assert_array_equal(comp(val_obj, other_val), comp(val, other_val))
    assert_array_equal(comp(other_val, val_obj), comp(other_val, val))


@pytest.mark.parametrize("comp",
        # 参数化测试用例：定义 NumPy 的整数比较通用函数
        [np.equal, np.not_equal, np.less_equal, np.less,
         np.greater_equal, np.greater])
def test_integer_integer_comparison(comp):
    np._set_promotion_state("weak")

    # 测试 NumPy 的比较通用函数是否可以处理大整数
    assert comp(2**200, -2**200) == comp(2**200, -2**200, dtype=object)


def create_with_scalar(sctype, value):
    return sctype(value)


def create_with_array(sctype, value):
    return np.array([value], dtype=sctype)


@pytest.mark.parametrize("sctype",
        # 参数化测试用例：定义整数类型
        [np.int8, np.int16, np.int32, np.int64,
         np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("create", [create_with_scalar, create_with_array])
def test_oob_creation(sctype, create):
    iinfo = np.iinfo(sctype)

    # 测试溢出创建情况是否会抛出 OverflowError 异常
    with pytest.raises(OverflowError):
        create(sctype, iinfo.min - 1)
    # 测试函数 create 在特定情况下是否会引发 OverflowError 异常
    
    # 测试当传入的值超过整数类型的最大值时是否会触发 OverflowError 异常
    with pytest.raises(OverflowError):
        create(sctype, iinfo.max + 1)
    
    # 测试当传入的值低于整数类型的最小值，并且以字符串形式传入时是否会触发 OverflowError 异常
    with pytest.raises(OverflowError):
        create(sctype, str(iinfo.min - 1))
    
    # 测试当传入的值超过整数类型的最大值，并且以字符串形式传入时是否会触发 OverflowError 异常
    with pytest.raises(OverflowError):
        create(sctype, str(iinfo.max + 1))
    
    # 断言当传入的值为整数类型的最小值时，create 函数的返回值应该等于整数类型的最小值
    assert create(sctype, iinfo.min) == iinfo.min
    
    # 断言当传入的值为整数类型的最大值时，create 函数的返回值应该等于整数类型的最大值
    assert create(sctype, iinfo.max) == iinfo.max
# 使用 pytest.mark.skipif 标记来跳过测试，如果 IS_WASM 为 True，说明环境是 wasm，不支持多线程
@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for threads")
def test_thread_local_promotion_state():
    # 创建一个 Barrier 对象，用于同步两个线程
    b = threading.Barrier(2)

    # 定义一个函数 legacy_no_warn
    def legacy_no_warn():
        # 设置 NumPy 的促升状态为 "legacy"
        np._set_promotion_state("legacy")
        # 等待直到所有线程都达到 Barrier
        b.wait()
        # 断言当前 NumPy 的促升状态为 "legacy"
        assert np._get_promotion_state() == "legacy"
        # 将警告转换为错误，此处不应该因为旧的促升状态而警告
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # 进行一个操作，如果有警告会转为错误
            np.float16(1) + 131008

    # 定义一个函数 weak_warn
    def weak_warn():
        # 设置 NumPy 的促升状态为 "weak"
        np._set_promotion_state("weak")
        # 等待直到所有线程都达到 Barrier
        b.wait()
        # 断言当前 NumPy 的促升状态为 "weak"
        assert np._get_promotion_state() == "weak"
        # 期望捕获到 RuntimeWarning 异常
        with pytest.raises(RuntimeWarning):
            # 进行一个操作，预期会产生 RuntimeWarning
            np.float16(1) + 131008

    # 创建两个线程，分别执行 legacy_no_warn 和 weak_warn 函数
    task1 = threading.Thread(target=legacy_no_warn)
    task2 = threading.Thread(target=weak_warn)

    # 启动两个线程
    task1.start()
    task2.start()
    # 等待两个线程执行完成
    task1.join()
    task2.join()
```