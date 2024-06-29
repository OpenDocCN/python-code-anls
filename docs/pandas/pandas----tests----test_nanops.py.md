# `D:\src\scipysrc\pandas\pandas\tests\test_nanops.py`

```
# 从 functools 模块中导入 partial 函数，用于创建带有预设参数的新函数
from functools import partial

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 pytest 测试框架
import pytest

# 导入 pandas 库中的测试装饰器模块
import pandas.util._test_decorators as td

# 从 pandas.core.dtypes.common 模块中导入 is_integer_dtype 函数
from pandas.core.dtypes.common import is_integer_dtype

# 导入 pandas 库，并使用别名 pd
import pandas as pd

# 从 pandas 模块中导入 Series 和 isna 函数
from pandas import (
    Series,
    isna,
)

# 导入 pandas 内部的测试工具模块
import pandas._testing as tm

# 从 pandas.core 模块中导入 nanops 模块
from pandas.core import nanops

# 设置一个变量 use_bn，指向 nanops 模块中的 _USE_BOTTLENECK 常量
use_bn = nanops._USE_BOTTLENECK

# 定义一个用于禁用 bottleneck 的 fixture，使用 monkeypatch 参数
@pytest.fixture
def disable_bottleneck(monkeypatch):
    # 在上下文中使用 monkeypatch，将 _USE_BOTTLENECK 设置为 False
    with monkeypatch.context() as m:
        m.setattr(nanops, "_USE_BOTTLENECK", False)
        # 使用 yield 返回 fixture 的值
        yield

# 定义一个返回数组形状 (11, 7) 的 fixture
@pytest.fixture
def arr_shape():
    return 11, 7

# 定义一个返回浮点数组的 fixture，依赖于 arr_shape fixture
@pytest.fixture
def arr_float(arr_shape):
    return np.random.default_rng(2).standard_normal(arr_shape)

# 定义一个返回复数数组的 fixture，依赖于 arr_float fixture
@pytest.fixture
def arr_complex(arr_float):
    return arr_float + arr_float * 1j

# 定义一个返回整数数组的 fixture，依赖于 arr_shape fixture
@pytest.fixture
def arr_int(arr_shape):
    return np.random.default_rng(2).integers(-10, 10, arr_shape)

# 定义一个返回布尔数组的 fixture，依赖于 arr_shape fixture
@pytest.fixture
def arr_bool(arr_shape):
    return np.random.default_rng(2).integers(0, 2, arr_shape) == 0

# 定义一个返回字符串数组的 fixture，依赖于 arr_float fixture
@pytest.fixture
def arr_str(arr_float):
    return np.abs(arr_float).astype("S")

# 定义一个返回 Unicode 字符串数组的 fixture，依赖于 arr_float fixture
@pytest.fixture
def arr_utf(arr_float):
    return np.abs(arr_float).astype("U")

# 定义一个返回日期时间数组的 fixture，依赖于 arr_shape fixture
@pytest.fixture
def arr_date(arr_shape):
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype("M8[ns]")

# 定义一个返回时间差数组的 fixture，依赖于 arr_shape fixture
@pytest.fixture
def arr_tdelta(arr_shape):
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype("m8[ns]")

# 定义一个返回包含 NaN 值的数组的 fixture，依赖于 arr_shape fixture
@pytest.fixture
def arr_nan(arr_shape):
    return np.tile(np.nan, arr_shape)

# 定义一个返回浮点 NaN 值数组的 fixture，依赖于 arr_float 和 arr_nan fixtures
@pytest.fixture
def arr_float_nan(arr_float, arr_nan):
    return np.vstack([arr_float, arr_nan])

# 定义一个返回 NaN 值浮点数组的 fixture，依赖于 arr_nan 和 arr_float fixtures
@pytest.fixture
def arr_nan_float1(arr_nan, arr_float):
    return np.vstack([arr_nan, arr_float])

# 定义一个返回包含 NaN 值的 NaN 值数组的 fixture，依赖于 arr_nan fixture
@pytest.fixture
def arr_nan_nan(arr_nan):
    return np.vstack([arr_nan, arr_nan])

# 定义一个返回包含无穷大值的数组的 fixture，依赖于 arr_float fixture
@pytest.fixture
def arr_inf(arr_float):
    return arr_float * np.inf

# 定义一个返回包含浮点和无穷大值的数组的 fixture，依赖于 arr_float 和 arr_inf fixtures
@pytest.fixture
def arr_float_inf(arr_float, arr_inf):
    return np.vstack([arr_float, arr_inf])

# 定义一个返回包含 NaN 值和无穷大值的数组的 fixture，依赖于 arr_nan 和 arr_inf fixtures
@pytest.fixture
def arr_nan_inf(arr_nan, arr_inf):
    return np.vstack([arr_nan, arr_inf])

# 定义一个返回包含浮点、NaN 值和无穷大值的数组的 fixture，依赖于 arr_float、arr_nan 和 arr_inf fixtures
@pytest.fixture
def arr_float_nan_inf(arr_float, arr_nan, arr_inf):
    return np.vstack([arr_float, arr_nan, arr_inf])

# 定义一个返回包含 NaN 值、NaN 值和无穷大值的数组的 fixture，依赖于 arr_nan 和 arr_inf fixtures
@pytest.fixture
def arr_nan_nan_inf(arr_nan, arr_inf):
    return np.vstack([arr_nan, arr_nan, arr_inf])

# 定义一个返回包含多种对象类型的数组的 fixture，依赖于多个不同类型的数组 fixtures
@pytest.fixture
def arr_obj(
    arr_float, arr_int, arr_bool, arr_complex, arr_str, arr_utf, arr_date, arr_tdelta
):
    return np.vstack(
        [
            arr_float.astype("O"),
            arr_int.astype("O"),
            arr_bool.astype("O"),
            arr_complex.astype("O"),
            arr_str.astype("O"),
            arr_utf.astype("O"),
            arr_date.astype("O"),
            arr_tdelta.astype("O"),
        ]
    )

# 定义一个返回包含 NaN 和 NaN 值的数组的 fixture，使用 np.errstate 来忽略无效的操作
@pytest.fixture
def arr_nan_nanj(arr_nan):
    with np.errstate(invalid="ignore"):
        return arr_nan + arr_nan * 1j

# 定义一个返回包含复数和 NaN 值的数组的 fixture，使用 np.errstate 来忽略无效的操作
@pytest.fixture
def arr_complex_nan(arr_complex, arr_nan_nanj):
    with np.errstate(invalid="ignore"):
        return np.vstack([arr_complex, arr_nan_nanj])

# 定义一个返回包含无穷大和 NaN 值的数组的 fixture，依赖于 arr_inf fixture
@pytest.fixture
def arr_nan_infj(arr_inf):
    # 在此代码块中，设置了一个上下文管理器，用于忽略 NumPy 中的无效操作错误
    with np.errstate(invalid="ignore"):
        # 返回复数形式的 arr_inf，其中实部为零，虚部为单位复数（1j）
        return arr_inf * 1j
# 定义一个 Pytest 的测试夹具，用于生成包含复数和NaN/Inf值的数组
@pytest.fixture
def arr_complex_nan_infj(arr_complex, arr_nan_infj):
    # 忽略无效值错误状态，将复数数组和NaN/Inf值数组垂直堆叠起来
    with np.errstate(invalid="ignore"):
        return np.vstack([arr_complex, arr_nan_infj])


# 定义一个 Pytest 的测试夹具，用于生成包含浮点数的一维数组
@pytest.fixture
def arr_float_1d(arr_float):
    # 返回输入浮点数数组的第一列，形成一维数组
    return arr_float[:, 0]


# 定义一个 Pytest 的测试夹具，用于生成包含NaN值的一维数组
@pytest.fixture
def arr_nan_1d(arr_nan):
    # 返回输入NaN值数组的第一列，形成一维数组
    return arr_nan[:, 0]


# 定义一个 Pytest 的测试夹具，用于生成包含浮点数和NaN值的一维数组
@pytest.fixture
def arr_float_nan_1d(arr_float_nan):
    # 返回输入浮点数和NaN值混合数组的第一列，形成一维数组
    return arr_float_nan[:, 0]


# 定义一个 Pytest 的测试夹具，用于生成包含浮点数和NaN值的一维数组
@pytest.fixture
def arr_float1_nan_1d(arr_float1_nan):
    # 返回输入浮点数和NaN值混合数组的第一列，形成一维数组
    return arr_float1_nan[:, 0]


# 定义一个 Pytest 的测试夹具，用于生成包含NaN值和浮点数的一维数组
@pytest.fixture
def arr_nan_float1_1d(arr_nan_float1):
    # 返回输入NaN值和浮点数混合数组的第一列，形成一维数组
    return arr_nan_float1[:, 0]


# 定义一个测试类，用于对 nanopsDataFrame 的功能进行单元测试
class TestnanopsDataFrame:
    # 设置测试方法的准备阶段
    def setup_method(self):
        # 关闭 pandas 的优化引擎（如果开启了的话）
        nanops._USE_BOTTLENECK = False

        # 定义一个数组的形状
        arr_shape = (11, 7)

        # 创建一个浮点数数组，形状为 arr_shape
        self.arr_float = np.random.default_rng(2).standard_normal(arr_shape)
        # 创建另一个与 self.arr_float 相同的浮点数数组
        self.arr_float1 = np.random.default_rng(2).standard_normal(arr_shape)
        # 创建一个复数数组，由 self.arr_float 和 self.arr_float1 构成
        self.arr_complex = self.arr_float + self.arr_float1 * 1j
        # 创建一个整数数组，取值范围为 [-10, 10)，形状为 arr_shape
        self.arr_int = np.random.default_rng(2).integers(-10, 10, arr_shape)
        # 创建一个布尔数组，形状为 arr_shape，随机确定每个元素是否为 False
        self.arr_bool = np.random.default_rng(2).integers(0, 2, arr_shape) == 0
        # 创建一个字符串数组，元素为 self.arr_float 的绝对值的字符串表示
        self.arr_str = np.abs(self.arr_float).astype("S")
        # 创建一个Unicode字符串数组，元素为 self.arr_float 的绝对值的Unicode字符串表示
        self.arr_utf = np.abs(self.arr_float).astype("U")
        # 创建一个日期时间数组，元素为 [0, 20000) 范围内的整数，转换为日期时间类型
        self.arr_date = (
            np.random.default_rng(2).integers(0, 20000, arr_shape).astype("M8[ns]")
        )
        # 创建一个时间增量数组，元素为 [0, 20000) 范围内的整数，转换为时间增量类型
        self.arr_tdelta = (
            np.random.default_rng(2).integers(0, 20000, arr_shape).astype("m8[ns]")
        )

        # 创建一个与 self.arr_float 形状相同的 NaN 填充的数组
        self.arr_nan = np.tile(np.nan, arr_shape)
        # 将 self.arr_float 和 self.arr_nan 垂直堆叠形成一个新数组
        self.arr_float_nan = np.vstack([self.arr_float, self.arr_nan])
        # 将 self.arr_float1 和 self.arr_nan 垂直堆叠形成一个新数组
        self.arr_float1_nan = np.vstack([self.arr_float1, self.arr_nan])
        # 将 self.arr_nan 和 self.arr_float1 垂直堆叠形成一个新数组
        self.arr_nan_float1 = np.vstack([self.arr_nan, self.arr_float1])
        # 将两个 self.arr_nan 垂直堆叠形成一个新数组
        self.arr_nan_nan = np.vstack([self.arr_nan, self.arr_nan])

        # 创建一个与 self.arr_float 形状相同的无穷大填充的数组
        self.arr_inf = self.arr_float * np.inf
        # 将 self.arr_float 和 self.arr_inf 垂直堆叠形成一个新数组
        self.arr_float_inf = np.vstack([self.arr_float, self.arr_inf])

        # 将 self.arr_nan 和 self.arr_inf 垂直堆叠形成一个新数组
        self.arr_nan_inf = np.vstack([self.arr_nan, self.arr_inf])
        # 将 self.arr_float、self.arr_nan 和 self.arr_inf 垂直堆叠形成一个新数组
        self.arr_float_nan_inf = np.vstack([self.arr_float, self.arr_nan, self.arr_inf])
        # 将 self.arr_nan、self.arr_nan 和 self.arr_inf 垂直堆叠形成一个新数组
        self.arr_nan_nan_inf = np.vstack([self.arr_nan, self.arr_nan, self.arr_inf])
        # 将浮点数、整数、布尔、复数、字符串、Unicode字符串、日期时间和时间增量数组堆叠形成一个对象数组
        self.arr_obj = np.vstack(
            [
                self.arr_float.astype("O"),
                self.arr_int.astype("O"),
                self.arr_bool.astype("O"),
                self.arr_complex.astype("O"),
                self.arr_str.astype("O"),
                self.arr_utf.astype("O"),
                self.arr_date.astype("O"),
                self.arr_tdelta.astype("O"),
            ]
        )

        # 忽略错误状态为无效时，将 self.arr_nan 加上自身乘以虚数单位，得到复数数组
        with np.errstate(invalid="ignore"):
            self.arr_nan_nanj = self.arr_nan + self.arr_nan * 1j
            # 将 self.arr_complex 和 self.arr_nan_nanj 垂直堆叠形成一个新数组
            self.arr_complex_nan = np.vstack([self.arr_complex, self.arr_nan_nanj])

            # 将 self.arr_inf 乘以虚数单位得到 self.arr_nan_infj
            self.arr_nan_infj = self.arr_inf * 1j
            # 将 self.arr_complex 和 self.arr_nan_infj 垂直堆叠形成一个新数组
            self.arr_complex_nan_infj = np.vstack([self.arr_complex, self.arr_nan_infj])

        # 将 self.arr_float 赋值给 self.arr_float_2d
        self.arr_float_2d = self.arr_float
        # 将 self.arr_float1 赋值给 self.arr_float1_2d
        self.arr_float1_2d = self.arr_float1

        # 将 self.arr_nan 赋值给 self.arr_nan_2d
        self.arr_nan_2d = self.arr_nan
        # 将 self.arr_float_nan 赋值给 self.arr_float_nan_2d
        self.arr_float_nan_2d = self.arr_float_nan
        # 将 self.arr_float1_nan 赋值给 self.arr_float1_nan_2d
        self.arr_float1_nan_2d = self.arr_float1_nan
        # 将 self.arr_nan_float1 赋值给 self.arr_nan_float1_2d
        self.arr_nan_float1_2d = self.arr_nan_float1

        # 将 self.arr_float 的第一列赋值给 self.arr_float_1d
        self.arr_float_1d = self.arr_float[:, 0]
        # 将 self.arr_float1 的第一列赋值给 self.arr_float1_1d
        self.arr_float1_1d = self.arr_float1[:, 0]

        # 将 self.arr_nan 的第一列赋值给 self.arr_nan_1d
        self.arr_nan_1d = self.arr_nan[:, 0]
        # 将 self.arr_float_nan 的第一列赋值给 self.arr_float_nan_1d
        self.arr_float_nan_1d = self.arr_float_nan[:, 0]
        # 将 self.arr_float1_nan 的第一列赋值给 self.arr_float1_nan_1d
        self.arr_float1_nan_1d = self.arr_float1_nan[:, 0]
        # 将 self.arr_nan_float1 的第一列赋值给 self.arr_nan_float1_1d
        self.arr_nan_float1_1d = self.arr_nan_float1[:, 0]

    # 设置测试方法的结束阶段
    def teardown_method(self):
        # 恢复 pandas 的优化引擎（如果之前开启了的话）
        nanops._USE_BOTTLENECK = use_bn
    # 检查结果的一致性，并根据需要将 res 转换为其 asm8 属性的值
    res = getattr(res, "asm8", res)

    # 如果 axis 不为 0，同时 targ 对象具有 shape 属性并且 ndim 不为 0，且 targ 的形状与 res 的形状不同
    if (
        axis != 0
        and hasattr(targ, "shape")
        and targ.ndim
        and targ.shape != res.shape
    ):
        # 根据 axis 对 res 进行分割，使其与 targ 的形状相匹配
        res = np.split(res, [targ.shape[0]], axis=0)[0]

    try:
        # 使用 tm.assert_almost_equal 函数检查 targ 和 res 的近似相等性
        tm.assert_almost_equal(targ, res, check_dtype=check_dtype)
    except AssertionError:
        # 处理 timedelta 类型的 dtype
        if hasattr(targ, "dtype") and targ.dtype == "m8[ns]":
            raise

        # 对于复数和对象类型的 dtype，有时会出现舍入误差
        # 如果不是这两种类型，则重新引发错误
        if not hasattr(res, "dtype") or res.dtype.kind not in ["c", "O"]:
            raise
        
        # 将对象类型的 dtype 转换为能够分解为实部和虚部的类型
        if res.dtype.kind == "O":
            if targ.dtype.kind != "O":
                res = res.astype(targ.dtype)
            else:
                # 根据 numpy 的版本选择合适的复数类型
                cast_dtype = "c16" if hasattr(np, "complex128") else "f8"
                res = res.astype(cast_dtype)
                targ = targ.astype(cast_dtype)
        
        # 不应出现 numpy 返回对象类型但 nanops 没有的情况，抛出异常
        elif targ.dtype.kind == "O":
            raise
        
        # 分别检查 targ 和 res 的实部和虚部的近似相等性
        tm.assert_almost_equal(np.real(targ), np.real(res), check_dtype=check_dtype)
        tm.assert_almost_equal(np.imag(targ), np.imag(res), check_dtype=check_dtype)
    ):
        # 遍历目标值的维度列表以及一个额外的 None 值
        for axis in list(range(targarval.ndim)) + [None]:
            # 根据 skipna 参数选择合适的目标值进行处理
            targartempval = targarval if skipna else testarval
            # 如果 skipna 为真且存在空的目标函数，并且目标值中所有值都是 NA，则调用空目标函数
            if skipna and empty_targfunc and isna(targartempval).all():
                targ = empty_targfunc(targartempval, axis=axis, **kwargs)
            else:
                # 否则调用正常的目标函数处理目标值
                targ = targfunc(targartempval, axis=axis, **kwargs)

            # 如果目标值的数据类型是对象并且目标函数是 np.any 或 np.all
            if targartempval.dtype == object and (
                targfunc is np.any or targfunc is np.all
            ):
                # numpy 函数会保留例如浮点性
                if isinstance(targ, np.ndarray):
                    targ = targ.astype(bool)
                else:
                    targ = bool(targ)

            # 如果测试函数的名称在 ["nanargmax", "nanargmin"] 中，并且测试值以 "arr_nan" 开头
            # 或者测试值以 "nan" 结尾并且（不跳过 NA 或者轴为 1）
            if testfunc.__name__ in ["nanargmax", "nanargmin"] and (
                testar.startswith("arr_nan")
                or (testar.endswith("nan") and (not skipna or axis == 1))
            ):
                # 使用 pytest 检测是否抛出 ValueError，并匹配特定的错误信息
                with pytest.raises(ValueError, match="Encountered .* NA value"):
                    testfunc(testarval, axis=axis, skipna=skipna, **kwargs)
                # 直接返回，不再继续执行
                return

            # 使用测试函数处理测试值，得到结果
            res = testfunc(testarval, axis=axis, skipna=skipna, **kwargs)

            # 如果目标值和结果都是 np.complex128 类型，并且结果是 float 类型并且都是 NaN
            if (
                isinstance(targ, np.complex128)
                and isinstance(res, float)
                and np.isnan(targ)
                and np.isnan(res)
            ):
                # GH#18463，目标值设为结果值
                targ = res

            # 检查处理后的目标值和结果，包括指定的轴和数据类型检查
            self.check_results(targ, res, axis, check_dtype=check_dtype)

            # 如果跳过 NA 值，则再次使用测试函数处理测试值并检查结果
            if skipna:
                res = testfunc(testarval, axis=axis, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)

            # 如果轴为 None，则使用测试函数处理测试值并检查结果
            if axis is None:
                res = testfunc(testarval, skipna=skipna, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)

            # 如果跳过 NA 值并且轴为 None，则使用测试函数处理测试值并检查结果
            if skipna and axis is None:
                res = testfunc(testarval, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)

        # 如果测试值的维度小于等于 1，则直接返回
        if testarval.ndim <= 1:
            return

        # 对较低维度进行递归处理
        testarval2 = np.take(testarval, 0, axis=-1)
        targarval2 = np.take(targarval, 0, axis=-1)
        # 调用自身检查函数，处理较低维度的数据
        self.check_fun_data(
            testfunc,
            targfunc,
            testar,
            testarval2,
            targarval2,
            skipna=skipna,
            check_dtype=check_dtype,
            empty_targfunc=empty_targfunc,
            **kwargs,
        )

    # 定义检查函数，用于对给定的测试函数、目标函数和参数进行检查
    def check_fun(
        self, testfunc, targfunc, testar, skipna, empty_targfunc=None, **kwargs
    ):
        targar = testar  # 将当前的测试目标赋给目标变量
        # 如果测试目标以 "_nan" 结尾并且类属性中存在去除 "_nan" 后的同名属性，则将目标变量改为去除 "_nan" 后的属性名
        if testar.endswith("_nan") and hasattr(self, testar[:-4]):
            targar = testar[:-4]

        testarval = getattr(self, testar)  # 获取当前测试目标的值
        targarval = getattr(self, targar)  # 获取目标变量的值
        self.check_fun_data(
            testfunc,  # 测试函数
            targfunc,  # 目标函数
            testar,    # 当前测试目标的名称
            testarval,  # 当前测试目标的值
            targarval,  # 目标变量的值
            skipna=skipna,  # 跳过 NaN 值的标志
            empty_targfunc=empty_targfunc,  # 空目标函数标志
            **kwargs,  # 其他关键字参数
        )

    def check_funs(
        self,
        testfunc,
        targfunc,
        skipna,
        allow_complex=True,
        allow_all_nan=True,
        allow_date=True,
        allow_tdelta=True,
        allow_obj=True,
        **kwargs,
    ):
        # 对不同类型数组执行 check_fun 检查
        self.check_fun(testfunc, targfunc, "arr_float", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_float_nan", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_int", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_bool", skipna, **kwargs)
        objs = [
            self.arr_float.astype("O"),
            self.arr_int.astype("O"),
            self.arr_bool.astype("O"),
        ]

        if allow_all_nan:
            self.check_fun(testfunc, targfunc, "arr_nan", skipna, **kwargs)

        if allow_complex:
            self.check_fun(testfunc, targfunc, "arr_complex", skipna, **kwargs)
            self.check_fun(testfunc, targfunc, "arr_complex_nan", skipna, **kwargs)
            if allow_all_nan:
                self.check_fun(testfunc, targfunc, "arr_nan_nanj", skipna, **kwargs)
            objs += [self.arr_complex.astype("O")]

        if allow_date:
            targfunc(self.arr_date)  # 执行目标函数以处理日期数组
            self.check_fun(testfunc, targfunc, "arr_date", skipna, **kwargs)
            objs += [self.arr_date.astype("O")]

        if allow_tdelta:
            try:
                targfunc(self.arr_tdelta)  # 尝试执行目标函数以处理时间增量数组
            except TypeError:
                pass
            else:
                self.check_fun(testfunc, targfunc, "arr_tdelta", skipna, **kwargs)
                objs += [self.arr_tdelta.astype("O")]

        if allow_obj:
            self.arr_obj = np.vstack(objs)  # 堆叠各类型数组以形成对象数组
            # 一些 nanops 处理对象 dtype 的能力比其 numpy 对应函数更好，因此需要向 numpy 函数提供其它输入
            if allow_obj == "convert":
                targfunc = partial(
                    self._badobj_wrap, func=targfunc, allow_complex=allow_complex
                )
            self.check_fun(testfunc, targfunc, "arr_obj", skipna, **kwargs)

    def _badobj_wrap(self, value, func, allow_complex=True, **kwargs):
        if value.dtype.kind == "O":  # 如果值的 dtype 是对象类型
            if allow_complex:
                value = value.astype("c16")  # 如果允许复数，则转换为复数类型
            else:
                value = value.astype("f8")  # 否则转换为浮点数类型
        return func(value, **kwargs)

    @pytest.mark.parametrize(
        "nan_op,np_op", [(nanops.nanany, np.any), (nanops.nanall, np.all)]
    )
    # 测试 NaN 函数的通用方法，调用 check_funs 函数进行测试
    def test_nan_funcs(self, nan_op, np_op, skipna):
        self.check_funs(nan_op, np_op, skipna, allow_all_nan=False, allow_date=False)

    # 测试 nansum 函数，调用 check_funs 函数进行测试
    def test_nansum(self, skipna):
        self.check_funs(
            nanops.nansum,
            np.sum,
            skipna,
            allow_date=False,
            check_dtype=False,
            empty_targfunc=np.nansum,
        )

    # 测试 nanmean 函数，调用 check_funs 函数进行测试
    def test_nanmean(self, skipna):
        self.check_funs(
            nanops.nanmean, np.mean, skipna, allow_obj=False, allow_date=False
        )

    # 忽略 RuntimeWarning 警告，测试 nanmedian 函数，调用 check_funs 函数进行测试
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanmedian(self, skipna):
        self.check_funs(
            nanops.nanmedian,
            np.median,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
        )

    # 使用参数化的方式测试 nanvar 函数，调用 check_funs 函数进行测试
    @pytest.mark.parametrize("ddof", range(3))
    def test_nanvar(self, ddof, skipna):
        self.check_funs(
            nanops.nanvar,
            np.var,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
            ddof=ddof,
        )

    # 使用参数化的方式测试 nanstd 函数，调用 check_funs 函数进行测试
    @pytest.mark.parametrize("ddof", range(3))
    def test_nanstd(self, ddof, skipna):
        self.check_funs(
            nanops.nanstd,
            np.std,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
            ddof=ddof,
        )

    # 使用参数化的方式测试 nansem 函数，调用 check_funs 函数进行测试
    @pytest.mark.parametrize("ddof", range(3))
    def test_nansem(self, ddof, skipna):
        sp_stats = pytest.importorskip("scipy.stats")

        # 忽略无效操作的错误，调用 check_funs 函数进行测试
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nansem,
                sp_stats.sem,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
                allow_obj="convert",
                ddof=ddof,
            )

    # 忽略 RuntimeWarning 警告，参数化测试 nanmin 和 nanmax 函数，调用 check_funs 函数进行测试
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize(
        "nan_op,np_op", [(nanops.nanmin, np.min), (nanops.nanmax, np.max)]
    )
    def test_nanops_with_warnings(self, nan_op, np_op, skipna):
        self.check_funs(nan_op, np_op, skipna, allow_obj=False)

    # 封装函数 _argminmax_wrap，调用 func 函数对 value 进行操作
    def _argminmax_wrap(self, value, axis=None, func=None):
        res = func(value, axis)
        nans = np.min(value, axis)
        nullnan = isna(nans)
        # 根据条件处理结果 res
        if res.ndim:
            res[nullnan] = -1
        elif (
            hasattr(nullnan, "all")
            and nullnan.all()
            or not hasattr(nullnan, "all")
            and nullnan
        ):
            res = -1
        return res

    # 忽略 RuntimeWarning 警告，测试 nanargmax 函数，调用 check_funs 函数进行测试
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanargmax(self, skipna):
        func = partial(self._argminmax_wrap, func=np.argmax)
        self.check_funs(nanops.nanargmax, func, skipna, allow_obj=False)
    # 定义测试方法，用于测试 nanargmin 函数
    def test_nanargmin(self, skipna):
        # 创建函数对象，使用 np.argmin 作为参数的部分函数
        func = partial(self._argminmax_wrap, func=np.argmin)
        # 调用通用检查函数，测试 nanops.nanargmin 函数
        self.check_funs(nanops.nanargmin, func, skipna, allow_obj=False)

    # 封装函数，用于处理 skew 和 kurtosis 相关计算
    def _skew_kurt_wrap(self, values, axis=None, func=None):
        # 如果输入的值不是浮点类型，则转换为 float64 类型
        if not isinstance(values.dtype.type, np.floating):
            values = values.astype("f8")
        # 使用给定的函数 func 计算结果，不进行偏置调整
        result = func(values, axis=axis, bias=False)
        # 修复处理某些情况下轴向上所有元素都相同的问题
        if isinstance(result, np.ndarray):
            # 如果最大值等于最小值，则将结果设置为 0
            result[np.max(values, axis=axis) == np.min(values, axis=axis)] = 0
            return result
        elif np.max(values) == np.min(values):
            return 0.0
        return result

    # 定义测试方法，用于测试 nanskew 函数
    def test_nanskew(self, skipna):
        # 导入 scipy.stats 模块，如果不存在则引发异常
        sp_stats = pytest.importorskip("scipy.stats")
        # 创建函数对象，使用 sp_stats.skew 作为参数的部分函数
        func = partial(self._skew_kurt_wrap, func=sp_stats.skew)
        # 在忽略无效值的情况下调用通用检查函数，测试 nanops.nanskew 函数
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nanskew,
                func,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
            )

    # 定义测试方法，用于测试 nankurt 函数
    def test_nankurt(self, skipna):
        # 导入 scipy.stats 模块，如果不存在则引发异常
        sp_stats = pytest.importorskip("scipy.stats")
        # 创建函数对象，使用 sp_stats.kurtosis(fisher=True) 作为参数的部分函数
        func1 = partial(sp_stats.kurtosis, fisher=True)
        func = partial(self._skew_kurt_wrap, func=func1)
        # 在忽略无效值的情况下调用通用检查函数，测试 nanops.nankurt 函数
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nankurt,
                func,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
            )

    # 定义测试方法，用于测试 nanprod 函数
    def test_nanprod(self, skipna):
        # 调用通用检查函数，测试 nanops.nanprod 函数
        self.check_funs(
            nanops.nanprod,
            np.prod,
            skipna,
            allow_date=False,
            allow_tdelta=False,
            empty_targfunc=np.nanprod,
        )
    # 检查给定的二维数据集的相关性和协方差，使用指定的检查函数和参数
    def check_nancorr_nancov_2d(self, checkfun, targ0, targ1, **kwargs):
        # 对未包含 NaN 值的两个二维浮点数组执行检查函数
        res00 = checkfun(self.arr_float_2d, self.arr_float1_2d, **kwargs)
        # 对未包含 NaN 值的两个二维浮点数组执行检查函数，指定最小周期为数组长度减1
        res01 = checkfun(
            self.arr_float_2d,
            self.arr_float1_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        # 断言未包含 NaN 值的两个二维浮点数组的检查结果与预期值 targ0 相近
        tm.assert_almost_equal(targ0, res00)
        # 断言未包含 NaN 值的两个二维浮点数组的检查结果与预期值 targ0 相近
        tm.assert_almost_equal(targ0, res01)

        # 对包含 NaN 值的两个二维浮点数组执行检查函数
        res10 = checkfun(self.arr_float_nan_2d, self.arr_float1_nan_2d, **kwargs)
        # 对包含 NaN 值的两个二维浮点数组执行检查函数，指定最小周期为数组长度减1
        res11 = checkfun(
            self.arr_float_nan_2d,
            self.arr_float1_nan_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        # 断言包含 NaN 值的两个二维浮点数组的检查结果与预期值 targ1 相近
        tm.assert_almost_equal(targ1, res10)
        # 断言包含 NaN 值的两个二维浮点数组的检查结果与预期值 targ1 相近
        tm.assert_almost_equal(targ1, res11)

        # 定义 NaN 值的目标值
        targ2 = np.nan
        # 对一个数组包含 NaN 值，另一个数组不含 NaN 值执行检查函数
        res20 = checkfun(self.arr_nan_2d, self.arr_float1_2d, **kwargs)
        # 对一个数组不含 NaN 值，另一个数组包含 NaN 值执行检查函数
        res21 = checkfun(self.arr_float_2d, self.arr_nan_2d, **kwargs)
        # 对两个数组都包含 NaN 值执行检查函数
        res22 = checkfun(self.arr_nan_2d, self.arr_nan_2d, **kwargs)
        # 对一个数组包含 NaN 值，另一个数组部分包含 NaN 值执行检查函数
        res23 = checkfun(self.arr_float_nan_2d, self.arr_nan_float1_2d, **kwargs)
        # 对一个数组部分包含 NaN 值，另一个数组部分包含 NaN 值执行检查函数，指定最小周期为数组长度减1
        res24 = checkfun(
            self.arr_float_nan_2d,
            self.arr_nan_float1_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        # 对未包含 NaN 值的两个二维浮点数组执行检查函数，指定最小周期为数组长度加1
        res25 = checkfun(
            self.arr_float_2d,
            self.arr_float1_2d,
            min_periods=len(self.arr_float_2d) + 1,
            **kwargs,
        )
        # 断言检查 NaN 值的结果与目标值 targ2（NaN）相近
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)
    # 定义一个方法，用于检查一维数据的相关性和协方差计算函数的结果
    def check_nancorr_nancov_1d(self, checkfun, targ0, targ1, **kwargs):
        # 调用给定的检查函数计算两组一维浮点数组的相关性和协方差，不考虑缺失值
        res00 = checkfun(self.arr_float_1d, self.arr_float1_1d, **kwargs)
        # 调用给定的检查函数计算两组一维浮点数组的相关性和协方差，指定最小有效期为数组长度减一
        res01 = checkfun(
            self.arr_float_1d,
            self.arr_float1_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        # 断言目标值与计算结果的近似相等性
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)

        # 调用给定的检查函数计算两组包含NaN的一维浮点数组的相关性和协方差
        res10 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, **kwargs)
        # 调用给定的检查函数计算两组包含NaN的一维浮点数组的相关性和协方差，指定最小有效期为数组长度减一
        res11 = checkfun(
            self.arr_float_nan_1d,
            self.arr_float1_nan_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        # 断言目标值与计算结果的近似相等性
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)

        # 设置目标值为NaN
        targ2 = np.nan
        # 调用给定的检查函数计算包含NaN的一维浮点数组与非NaN数组的相关性和协方差
        res20 = checkfun(self.arr_nan_1d, self.arr_float1_1d, **kwargs)
        # 调用给定的检查函数计算两组一维浮点数组中一个包含NaN的相关性和协方差
        res21 = checkfun(self.arr_float_1d, self.arr_nan_1d, **kwargs)
        # 调用给定的检查函数计算两组包含NaN的一维浮点数组的相关性和协方差
        res22 = checkfun(self.arr_nan_1d, self.arr_nan_1d, **kwargs)
        # 调用给定的检查函数计算包含NaN的一维浮点数组与包含NaN的浮点数组的相关性和协方差
        res23 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, **kwargs)
        # 调用给定的检查函数计算两组包含NaN的一维浮点数组的相关性和协方差，指定最小有效期为数组长度减一
        res24 = checkfun(
            self.arr_float_nan_1d,
            self.arr_nan_float1_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        # 调用给定的检查函数计算两组一维浮点数组的相关性和协方差，指定最小有效期为数组长度加一
        res25 = checkfun(
            self.arr_float_1d,
            self.arr_float1_1d,
            min_periods=len(self.arr_float_1d) + 1,
            **kwargs,
        )
        # 断言目标值与计算结果的近似相等性
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)
    # 测试使用 Kendall 方法计算 NaN 安全相关性
    def test_nancorr_kendall(self):
        # 导入并检查 scipy.stats 库是否可用，否则跳过测试
        sp_stats = pytest.importorskip("scipy.stats")

        # 计算二维数组 self.arr_float_2d 和 self.arr_float1_2d 的 Kendall tau 相关系数的第一个返回值
        targ0 = sp_stats.kendalltau(self.arr_float_2d, self.arr_float1_2d)[0]
        # 计算扁平化后的二维数组的 Kendall tau 相关系数的第一个返回值
        targ1 = sp_stats.kendalltau(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        # 调用 self.check_nancorr_nancov_2d 方法检查二维数组的 NaN 安全相关性计算结果
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="kendall")

        # 计算一维数组 self.arr_float_1d 和 self.arr_float1_1d 的 Kendall tau 相关系数的第一个返回值
        targ0 = sp_stats.kendalltau(self.arr_float_1d, self.arr_float1_1d)[0]
        # 计算扁平化后的一维数组的 Kendall tau 相关系数的第一个返回值
        targ1 = sp_stats.kendalltau(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        # 调用 self.check_nancorr_nancov_1d 方法检查一维数组的 NaN 安全相关性计算结果
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="kendall")

    # 测试使用 Spearman 方法计算 NaN 安全相关性
    def test_nancorr_spearman(self):
        # 导入并检查 scipy.stats 库是否可用，否则跳过测试
        sp_stats = pytest.importorskip("scipy.stats")

        # 计算二维数组 self.arr_float_2d 和 self.arr_float1_2d 的 Spearman 相关系数的第一个返回值
        targ0 = sp_stats.spearmanr(self.arr_float_2d, self.arr_float1_2d)[0]
        # 计算扁平化后的二维数组的 Spearman 相关系数的第一个返回值
        targ1 = sp_stats.spearmanr(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        # 调用 self.check_nancorr_nancov_2d 方法检查二维数组的 NaN 安全相关性计算结果
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="spearman")

        # 计算一维数组 self.arr_float_1d 和 self.arr_float1_1d 的 Spearman 相关系数的第一个返回值
        targ0 = sp_stats.spearmanr(self.arr_float_1d, self.arr_float1_1d)[0]
        # 计算扁平化后的一维数组的 Spearman 相关系数的第一个返回值
        targ1 = sp_stats.spearmanr(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        # 调用 self.check_nancorr_nancov_1d 方法检查一维数组的 NaN 安全相关性计算结果
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="spearman")

    # 测试使用无效方法时的异常情况
    def test_invalid_method(self):
        # 导入并检查 scipy 库是否可用，否则跳过测试
        pytest.importorskip("scipy")

        # 计算二维数组 self.arr_float_2d 和 self.arr_float1_2d 的相关系数矩阵后取其第一行第二列的值
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        # 计算扁平化后的二维数组的相关系数矩阵后取其第一行第二列的值
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        # 准备一个匹配异常信息的消息字符串
        msg = "Unknown method 'foo', expected one of 'kendall', 'spearman'"
        # 使用 pytest 的断言，期望引发 ValueError 异常，并匹配预定义的消息
        with pytest.raises(ValueError, match=msg):
            self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="foo")

    # 测试使用 NaN 安全协方差计算
    def test_nancov(self):
        # 计算二维数组 self.arr_float_2d 和 self.arr_float1_2d 的协方差矩阵后取其第一行第二列的值
        targ0 = np.cov(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        # 计算扁平化后的二维数组的协方差矩阵后取其第一行第二列的值
        targ1 = np.cov(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        # 调用 self.check_nancorr_nancov_2d 方法检查二维数组的 NaN 安全协方差计算结果
        self.check_nancorr_nancov_2d(nanops.nancov, targ0, targ1)

        # 计算一维数组 self.arr_float_1d 和 self.arr_float1_1d 的协方差矩阵后取其第一行第二列的值
        targ0 = np.cov(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        # 计算扁平化后的一维数组的协方差矩阵后取其第一行第二列的值
        targ1 = np.cov(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        # 调用 self.check_nancorr_nancov_1d 方法检查一维数组的 NaN 安全协方差计算结果
        self.check_nancorr_nancov_1d(nanops.nancov, targ0, targ1)
@pytest.mark.parametrize(
    "arr, correct",
    [  # 定义参数化测试的参数数组，每个元素是一个元组，包含数组名和预期结果
        ("arr_complex", False),  # 测试复杂数组，预期结果为 False
        ("arr_int", False),  # 测试整数数组，预期结果为 False
        ("arr_bool", False),  # 测试布尔数组，预期结果为 False
        ("arr_str", False),  # 测试字符串数组，预期结果为 False
        ("arr_utf", False),  # 测试UTF-8编码字符串数组，预期结果为 False
        ("arr_complex_nan", False),  # 测试包含NaN的复杂数组，预期结果为 False
        ("arr_nan_nanj", False),  # 测试包含NaN和NaNj的数组，预期结果为 False
        ("arr_nan_infj", True),  # 测试包含NaN和Infj的数组，预期结果为 True
        ("arr_complex_nan_infj", True),  # 测试包含复杂数NaN和Infj的数组，预期结果为 True
    ],
)
def test_has_infs_non_float(request, arr, correct, disable_bottleneck):
    val = request.getfixturevalue(arr)  # 获取测试夹具中的数组数据
    while getattr(val, "ndim", True):  # 循环直到数组维度为0
        res0 = nanops._has_infs(val)  # 调用函数检查数组是否包含无穷大
        if correct:
            assert res0  # 如果预期结果为 True，则断言检查结果为 True
        else:
            assert not res0  # 如果预期结果为 False，则断言检查结果为 False

        if not hasattr(val, "ndim"):  # 如果数组没有维度属性，则跳出循环
            break

        # 减少维度，为下一个循环步骤做准备
        val = np.take(val, 0, axis=-1)


@pytest.mark.parametrize(
    "arr, correct",
    [  # 定义参数化测试的参数数组，每个元素是一个元组，包含数组名和预期结果
        ("arr_float", False),  # 测试浮点数数组，预期结果为 False
        ("arr_nan", False),  # 测试包含NaN的数组，预期结果为 False
        ("arr_float_nan", False),  # 测试包含浮点数NaN的数组，预期结果为 False
        ("arr_nan_nan", False),  # 测试包含NaN和NaN的数组，预期结果为 False
        ("arr_float_inf", True),  # 测试包含无穷大的浮点数数组，预期结果为 True
        ("arr_inf", True),  # 测试包含无穷大的数组，预期结果为 True
        ("arr_nan_inf", True),  # 测试包含NaN和无穷大的数组，预期结果为 True
        ("arr_float_nan_inf", True),  # 测试包含浮点数NaN和无穷大的数组，预期结果为 True
        ("arr_nan_nan_inf", True),  # 测试包含NaN和NaN及无穷大的数组，预期结果为 True
    ],
)
@pytest.mark.parametrize("astype", [None, "f4", "f2"])
def test_has_infs_floats(request, arr, correct, astype, disable_bottleneck):
    val = request.getfixturevalue(arr)  # 获取测试夹具中的数组数据
    if astype is not None:
        val = val.astype(astype)  # 将数组转换为指定类型（如果指定了类型）
    while getattr(val, "ndim", True):  # 循环直到数组维度为0
        res0 = nanops._has_infs(val)  # 调用函数检查数组是否包含无穷大
        if correct:
            assert res0  # 如果预期结果为 True，则断言检查结果为 True
        else:
            assert not res0  # 如果预期结果为 False，则断言检查结果为 False

        if not hasattr(val, "ndim"):  # 如果数组没有维度属性，则跳出循环
            break

        # 减少维度，为下一个循环步骤做准备
        val = np.take(val, 0, axis=-1)


@pytest.mark.parametrize(
    "fixture", ["arr_float", "arr_complex", "arr_int", "arr_bool", "arr_str", "arr_utf"]
)
def test_bn_ok_dtype(fixture, request, disable_bottleneck):
    obj = request.getfixturevalue(fixture)  # 获取测试夹具中的数组数据
    assert nanops._bn_ok_dtype(obj.dtype, "test")  # 断言数组数据类型在处理函数中符合预期


@pytest.mark.parametrize(
    "fixture",
    [
        "arr_date",  # 测试日期数组，预期结果是数据类型不符合处理函数要求
        "arr_tdelta",  # 测试时间差数组，预期结果是数据类型不符合处理函数要求
        "arr_obj",  # 测试对象数组，预期结果是数据类型不符合处理函数要求
    ],
)
def test_bn_not_ok_dtype(fixture, request, disable_bottleneck):
    obj = request.getfixturevalue(fixture)  # 获取测试夹具中的数组数据
    assert not nanops._bn_ok_dtype(obj.dtype, "test")  # 断言数组数据类型在处理函数中不符合预期


class TestEnsureNumeric:
    def test_numeric_values(self):
        # 测试整数
        assert nanops._ensure_numeric(1) == 1

        # 测试浮点数
        assert nanops._ensure_numeric(1.1) == 1.1

        # 测试复数
        assert nanops._ensure_numeric(1 + 2j) == 1 + 2j
    # 定义一个测试方法，用于测试 nanops._ensure_numeric 函数的不同输入情况

    # 测试数值类型的 ndarray
    values = np.array([1, 2, 3])
    assert np.allclose(nanops._ensure_numeric(values), values)

    # 测试对象类型的 ndarray
    o_values = values.astype(object)
    assert np.allclose(nanops._ensure_numeric(o_values), values)

    # 测试可转换为数值的字符串类型 ndarray
    s_values = np.array(["1", "2", "3"], dtype=object)
    msg = r"Could not convert \['1' '2' '3'\] to numeric"
    with pytest.raises(TypeError, match=msg):
        nanops._ensure_numeric(s_values)

    # 测试不可转换为数值的字符串类型 ndarray
    s_values = np.array(["foo", "bar", "baz"], dtype=object)
    msg = r"Could not convert .* to numeric"
    with pytest.raises(TypeError, match=msg):
        nanops._ensure_numeric(s_values)

    # 测试 nanops._ensure_numeric 函数处理可转换的单个字符串值时的异常情况
    with pytest.raises(TypeError, match="Could not convert string '1' to numeric"):
        nanops._ensure_numeric("1")
    with pytest.raises(
        TypeError, match="Could not convert string '1.1' to numeric"
    ):
        nanops._ensure_numeric("1.1")
    with pytest.raises(
        TypeError, match=r"Could not convert string '1\+1j' to numeric"
    ):
        nanops._ensure_numeric("1+1j")

    # 测试 nanops._ensure_numeric 函数处理不可转换的单个字符串值时的异常情况
    msg = "Could not convert string 'foo' to numeric"
    with pytest.raises(TypeError, match=msg):
        nanops._ensure_numeric("foo")

    # 测试 nanops._ensure_numeric 函数处理不支持的数据类型时的异常情况
    msg = "argument must be a string or a number"
    with pytest.raises(TypeError, match=msg):
        nanops._ensure_numeric({})
    with pytest.raises(TypeError, match=msg):
        nanops._ensure_numeric([])
# 定义一个测试类 TestNanvarFixedValues，用于测试 nanvar 函数的固定数值情况
class TestNanvarFixedValues:
    # 标识符 xref GH10242，关联到 GitHub issue 10242
    # 从正态分布中采样样本数据
    @pytest.fixture
    def variance(self):
        return 3.0

    # 根据给定的方差生成样本数据
    @pytest.fixture
    def samples(self, variance):
        return self.prng.normal(scale=variance**0.5, size=100000)

    # 测试 nanvar 函数在所有数据都为有限值时的情况
    def test_nanvar_all_finite(self, samples, variance):
        # 计算样本数据的方差
        actual_variance = nanops.nanvar(samples)
        # 断言计算得到的方差接近于预期方差值，相对误差容差为 1e-2
        tm.assert_almost_equal(actual_variance, variance, rtol=1e-2)

    # 测试 nanvar 函数在数据中含有 NaN 值时的情况
    def test_nanvar_nans(self, samples, variance):
        # 创建一个含有 NaN 值的测试样本数据
        samples_test = np.nan * np.ones(2 * samples.shape[0])
        samples_test[::2] = samples

        # 计算含有 NaN 值的样本数据的方差，跳过 NaN 值
        actual_variance = nanops.nanvar(samples_test, skipna=True)
        # 断言计算得到的方差接近于预期方差值，相对误差容差为 1e-2
        tm.assert_almost_equal(actual_variance, variance, rtol=1e-2)

        # 计算含有 NaN 值的样本数据的方差，不跳过 NaN 值
        actual_variance = nanops.nanvar(samples_test, skipna=False)
        # 断言计算得到的方差应为 NaN，相对误差容差为 1e-2
        tm.assert_almost_equal(actual_variance, np.nan, rtol=1e-2)

    # 测试 nanstd 函数在数据中含有 NaN 值时的情况
    def test_nanstd_nans(self, samples, variance):
        # 创建一个含有 NaN 值的测试样本数据
        samples_test = np.nan * np.ones(2 * samples.shape[0])
        samples_test[::2] = samples

        # 计算含有 NaN 值的样本数据的标准差，跳过 NaN 值
        actual_std = nanops.nanstd(samples_test, skipna=True)
        # 断言计算得到的标准差接近于预期标准差值，相对误差容差为 1e-2
        tm.assert_almost_equal(actual_std, variance**0.5, rtol=1e-2)

        # 计算含有 NaN 值的样本数据的方差，不跳过 NaN 值
        actual_std = nanops.nanvar(samples_test, skipna=False)
        # 断言计算得到的方差应为 NaN，相对误差容差为 1e-2
        tm.assert_almost_equal(actual_std, np.nan, rtol=1e-2)

    # 测试 nanvar 函数在指定轴向上的计算情况
    def test_nanvar_axis(self, samples, variance):
        # 生成一些样本数据
        samples_unif = self.prng.uniform(size=samples.shape[0])
        samples = np.vstack([samples, samples_unif])

        # 计算样本数据在指定轴向上的方差
        actual_variance = nanops.nanvar(samples, axis=1)
        # 断言计算得到的方差数组接近于预期数组，相对误差容差为 1e-2
        tm.assert_almost_equal(
            actual_variance, np.array([variance, 1.0 / 12]), rtol=1e-2
        )

    # 测试 nanvar 函数在指定自由度修正值下的计算情况
    def test_nanvar_ddof(self):
        n = 5
        samples = self.prng.uniform(size=(10000, n + 1))
        samples[:, -1] = np.nan  # 强制使用我们自定义的算法

        # 计算样本数据在指定轴向上的方差，跳过 NaN 值，使用指定的自由度修正值
        variance_0 = nanops.nanvar(samples, axis=1, skipna=True, ddof=0).mean()
        variance_1 = nanops.nanvar(samples, axis=1, skipna=True, ddof=1).mean()
        variance_2 = nanops.nanvar(samples, axis=1, skipna=True, ddof=2).mean()

        # 断言未偏估计值
        var = 1.0 / 12
        tm.assert_almost_equal(variance_1, var, rtol=1e-2)

        # 断言低估方差值
        tm.assert_almost_equal(variance_0, (n - 1.0) / n * var, rtol=1e-2)

        # 断言高估方差值
        tm.assert_almost_equal(variance_2, (n - 1.0) / (n - 2.0) * var, rtol=1e-2)

    # 使用参数化测试来测试不同轴向和自由度修正值下的情况
    @pytest.mark.parametrize("axis", range(2))
    @pytest.mark.parametrize("ddof", range(3))
    # 定义一个测试方法，用于验证与使用 Numpy 预先计算的值是否一致。
    def test_ground_truth(self, axis, ddof):
        # 创建一个 4x4 的空数组作为样本数据集
        samples = np.empty((4, 4))
        # 将部分数值填充到样本数组中
        samples[:3, :3] = np.array(
            [
                [0.97303362, 0.21869576, 0.55560287],
                [0.72980153, 0.03109364, 0.99155171],
                [0.09317602, 0.60078248, 0.15871292],
            ]
        )
        # 将第四行和第四列设置为 NaN 值
        samples[3] = samples[:, 3] = np.nan

        # 预先计算的方差值，分别针对 axis=0, 1 和 ddof=0, 1, 2
        variance = np.array(
            [
                [
                    [0.13762259, 0.05619224, 0.11568816],
                    [0.20643388, 0.08428837, 0.17353224],
                    [0.41286776, 0.16857673, 0.34706449],
                ],
                [
                    [0.09519783, 0.16435395, 0.05082054],
                    [0.14279674, 0.24653093, 0.07623082],
                    [0.28559348, 0.49306186, 0.15246163],
                ],
            ]
        )

        # 使用 nanops 模块中的 nanvar 方法计算样本的方差，跳过 NaN 值，指定 axis 和 ddof 参数
        var = nanops.nanvar(samples, skipna=True, axis=axis, ddof=ddof)
        # 断言计算得到的方差结果与预计结果相近，对前三个元素进行验证
        tm.assert_almost_equal(var[:3], variance[axis, ddof])
        # 断言第四个元素为 NaN
        assert np.isnan(var[3])

        # 使用 nanops 模块中的 nanstd 方法计算样本的标准差，跳过 NaN 值，指定 axis 和 ddof 参数
        std = nanops.nanstd(samples, skipna=True, axis=axis, ddof=ddof)
        # 断言计算得到的标准差结果与预计结果的平方根相近，对前三个元素进行验证
        tm.assert_almost_equal(std[:3], variance[axis, ddof] ** 0.5)
        # 断言第四个元素为 NaN
        assert np.isnan(std[3])

    # 使用 pytest 的参数化装饰器，为 ddof 参数提供 0, 1, 2 的三种取值，以多次运行同一个测试方法
    @pytest.mark.parametrize("ddof", range(3))
    def test_nanstd_roundoff(self, ddof):
        # GH 10242 的回归测试（测试数据来源于 GH 10489）。确保方差计算的稳定性。
        # 创建一个 Series 对象，其中的数据是 766897346 重复十次
        data = Series(766897346 * np.ones(10))
        # 计算数据的标准差，指定 ddof 参数
        result = data.std(ddof=ddof)
        # 断言计算得到的标准差结果为 0.0
        assert result == 0.0

    # 属性方法，用于返回一个基于默认种子值 2 的随机数生成器对象
    @property
    def prng(self):
        return np.random.default_rng(2)
class TestNanskewFixedValues:
    # 定义一个测试类 TestNanskewFixedValues，用于测试 nanskew 函数的固定值情况
    # xref GH 11974，参考 GitHub 上的 issue 编号 11974

    # 返回一个包含正弦值的样本数据，共 200 个点
    @pytest.fixture
    def samples(self):
        return np.sin(np.linspace(0, 1, 200))

    # 返回预期的偏度值 -0.1875895205961754
    @pytest.fixture
    def actual_skew(self):
        return -0.1875895205961754

    # 使用参数化测试，验证对于给定的几个常数，nanskew 函数的输出为 0.0
    @pytest.mark.parametrize("val", [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        # xref GH 11974，参考 GitHub 上的 issue 编号 11974
        # 创建一个包含 300 个重复值为 val 的数据
        data = val * np.ones(300)
        # 计算该数据的偏度
        skew = nanops.nanskew(data)
        # 断言计算出的偏度应该等于 0.0
        assert skew == 0.0

    # 测试所有数据均为有限值时的情况
    def test_all_finite(self):
        # 设置两个 beta 分布的参数值
        alpha, beta = 0.3, 0.1
        # 生成一个左尾的 beta 分布样本数据，大小为 100
        left_tailed = self.prng.beta(alpha, beta, size=100)
        # 断言左尾 beta 分布的数据偏度应小于 0
        assert nanops.nanskew(left_tailed) < 0

        alpha, beta = 0.1, 0.3
        # 生成一个右尾的 beta 分布样本数据，大小为 100
        right_tailed = self.prng.beta(alpha, beta, size=100)
        # 断言右尾 beta 分布的数据偏度应大于 0
        assert nanops.nanskew(right_tailed) > 0

    # 测试样本数据与预期偏度值进行比较
    def test_ground_truth(self, samples, actual_skew):
        # 计算样本数据的偏度
        skew = nanops.nanskew(samples)
        # 使用近似相等断言，验证计算出的偏度与预期值应近似相等
        tm.assert_almost_equal(skew, actual_skew)

    # 测试在特定轴向上的偏度计算
    def test_axis(self, samples, actual_skew):
        # 将样本数据与一列 NaN 值堆叠在一起，构成一个矩阵
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        # 计算矩阵在 axis=1 方向上的偏度
        skew = nanops.nanskew(samples, axis=1)
        # 使用近似相等断言，验证计算出的偏度与预期值矩阵应近似相等
        tm.assert_almost_equal(skew, np.array([actual_skew, np.nan]))

    # 测试包含 NaN 值的样本数据，验证 skipna=False 时的偏度计算
    def test_nans(self, samples):
        # 将样本数据与一个 NaN 值合并在一起，构成一个数组
        samples = np.hstack([samples, np.nan])
        # 计算包含 NaN 值的数组在 skipna=False 情况下的偏度
        skew = nanops.nanskew(samples, skipna=False)
        # 断言计算出的偏度应为 NaN
        assert np.isnan(skew)

    # 测试包含 NaN 值的样本数据，验证 skipna=True 时的偏度计算
    def test_nans_skipna(self, samples, actual_skew):
        # 将样本数据与一个 NaN 值合并在一起，构成一个数组
        samples = np.hstack([samples, np.nan])
        # 计算包含 NaN 值的数组在 skipna=True 情况下的偏度
        skew = nanops.nanskew(samples, skipna=True)
        # 使用近似相等断言，验证计算出的偏度与预期值应近似相等
        tm.assert_almost_equal(skew, actual_skew)

    # 属性方法，返回一个基于默认种子的随机数生成器
    @property
    def prng(self):
        return np.random.default_rng(2)


class TestNankurtFixedValues:
    # 定义一个测试类 TestNankurtFixedValues，用于测试 nankurt 函数的固定值情况
    # xref GH 11974，参考 GitHub 上的 issue 编号 11974

    # 返回一个包含正弦值的样本数据，共 200 个点
    @pytest.fixture
    def samples(self):
        return np.sin(np.linspace(0, 1, 200))

    # 返回预期的峰度值 -1.2058303433799713
    @pytest.fixture
    def actual_kurt(self):
        return -1.2058303433799713

    # 使用参数化测试，验证对于给定的几个常数，nankurt 函数的输出为 0.0
    @pytest.mark.parametrize("val", [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        # xref GH 11974，参考 GitHub 上的 issue 编号 11974
        # 创建一个包含 300 个重复值为 val 的数据
        data = val * np.ones(300)
        # 计算该数据的峰度
        kurt = nanops.nankurt(data)
        # 断言计算出的峰度应该等于 0.0
        assert kurt == 0.0

    # 测试所有数据均为有限值时的情况
    def test_all_finite(self):
        # 设置两个 beta 分布的参数值
        alpha, beta = 0.3, 0.1
        # 生成一个左尾的 beta 分布样本数据，大小为 100
        left_tailed = self.prng.beta(alpha, beta, size=100)
        # 断言左尾 beta 分布的数据峰度应小于 2
        assert nanops.nankurt(left_tailed) < 2

        alpha, beta = 0.1, 0.3
        # 生成一个右尾的 beta 分布样本数据，大小为 100
        right_tailed = self.prng.beta(alpha, beta, size=100)
        # 断言右尾 beta 分布的数据峰度应小于 0
        assert nanops.nankurt(right_tailed) < 0

    # 测试样本数据与预期峰度值进行比较
    def test_ground_truth(self, samples, actual_kurt):
        # 计算样本数据的峰度
        kurt = nanops.nankurt(samples)
        # 使用近似相等断言，验证计算出的峰度与预期值应近似相等
        tm.assert_almost_equal(kurt, actual_kurt)

    # 测试在特定轴向上的峰度计算
    def test_axis(self, samples, actual_kurt):
        # 将样本数据与一列 NaN 值堆叠在一起，构成一个矩阵
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        # 计算矩阵在 axis=1 方向上的峰度
        kurt = nanops.nankurt(samples, axis=1)
        # 使用近似相等断言，验证计算出的峰度与预期值矩阵应近似相等
        tm.assert_almost_equal(kurt, np.array([actual_kurt, np.nan]))
    # 测试函数，用于检查处理含有 NaN（Not a Number）的情况
    def test_nans(self, samples):
        # 在 samples 数组末尾添加一个 NaN 值
        samples = np.hstack([samples, np.nan])
        # 调用 nankurt 函数计算样本的峰度，不跳过 NaN 值
        kurt = nanops.nankurt(samples, skipna=False)
        # 使用断言确保计算结果为 NaN
        assert np.isnan(kurt)

    # 测试函数，用于检查处理含有 NaN 的情况，并验证计算结果
    def test_nans_skipna(self, samples, actual_kurt):
        # 在 samples 数组末尾添加一个 NaN 值
        samples = np.hstack([samples, np.nan])
        # 调用 nankurt 函数计算样本的峰度，跳过 NaN 值
        kurt = nanops.nankurt(samples, skipna=True)
        # 使用测试工具函数确保计算结果与预期的实际峰度值几乎相等
        tm.assert_almost_equal(kurt, actual_kurt)

    # 属性方法，返回一个新的随机数生成器对象
    @property
    def prng(self):
        return np.random.default_rng(2)
# 定义一个测试类 TestDatetime64NaNOps，用于测试 Pandas 的日期时间操作
class TestDatetime64NaNOps:
    
    # 测试 nanmean 方法
    # unit 参数指定时间单位，生成日期时间索引
    def test_nanmean(self, unit):
        # 生成一个日期时间索引对象 dti，从 "2016-01-01" 开始，包含3个时间点，时间单位由 unit 指定
        dti = pd.date_range("2016-01-01", periods=3).as_unit(unit)
        # 预期的值是 dti 的第二个时间点
        expected = dti[1]

        # 遍历 dti 和 dti._data 对象
        for obj in [dti, dti._data]:
            # 计算对象 obj 的 nanmean（忽略 NaN 值的平均值）
            result = nanops.nanmean(obj)
            # 断言计算结果与预期值相等
            assert result == expected

        # 在 dti 的第1个位置插入 NaT（Not a Time，表示缺失时间）
        dti2 = dti.insert(1, pd.NaT)

        # 再次遍历 dti2 和 dti2._data 对象
        for obj in [dti2, dti2._data]:
            # 计算对象 obj 的 nanmean
            result = nanops.nanmean(obj)
            # 断言计算结果与预期值相等
            assert result == expected

    # 使用 pytest 的参数化装饰器，测试 nanmean 方法中的 skipna=False 的情况
    # constructor 参数指定数据类型的构造器，unit 参数指定时间单位
    @pytest.mark.parametrize("constructor", ["M8", "m8"])
    def test_nanmean_skipna_false(self, constructor, unit):
        # 构造 dtype，指定数据类型和时间单位
        dtype = f"{constructor}[{unit}]"
        # 创建一个 4x3 的数组，数据类型为 dtype，包含整数从0到11
        arr = np.arange(12).astype(np.int64).view(dtype).reshape(4, 3)

        # 将数组最后一个元素设为 "NaT"（表示缺失时间）
        arr[-1, -1] = "NaT"

        # 计算数组 arr 的 nanmean，skipna=False 表示不忽略 NaN 值
        result = nanops.nanmean(arr, skipna=False)
        # 断言结果是 NaN 时间
        assert np.isnat(result)
        # 断言结果的数据类型符合预期的 dtype
        assert result.dtype == dtype

        # 沿着 axis=0 计算 arr 的 nanmean，skipna=False
        result = nanops.nanmean(arr, axis=0, skipna=False)
        # 期望的结果是一个数组，第三个元素为 "NaT"
        expected = np.array([4, 5, "NaT"], dtype=arr.dtype)
        # 使用 test machinery (tm) 的方法断言两个数组相等
        tm.assert_numpy_array_equal(result, expected)

        # 沿着 axis=1 计算 arr 的 nanmean，skipna=False
        result = nanops.nanmean(arr, axis=1, skipna=False)
        # 期望的结果是一个数组，包含每行第二个元素和最后一个元素
        expected = np.array([arr[0, 1], arr[1, 1], arr[2, 1], arr[-1, -1]])
        # 使用 test machinery 的方法断言两个数组相等
        tm.assert_numpy_array_equal(result, expected)


# 测试是否使用了 bottleneck 库
def test_use_bottleneck():
    # 如果已安装 bottleneck 库
    if nanops._BOTTLENECK_INSTALLED:
        # 在上下文中设置 Pandas 使用 bottleneck 库，验证选项生效
        with pd.option_context("use_bottleneck", True):
            # 断言 Pandas 是否正在使用 bottleneck 库
            assert pd.get_option("use_bottleneck")

        # 在上下文中设置 Pandas 不使用 bottleneck 库，验证选项生效
        with pd.option_context("use_bottleneck", False):
            # 断言 Pandas 是否没有使用 bottleneck 库
            assert not pd.get_option("use_bottleneck")


# 参数化测试不同的 numpy 操作
@pytest.mark.parametrize(
    "numpy_op, expected",
    [
        (np.sum, 10),
        (np.nansum, 10),
        (np.mean, 2.5),
        (np.nanmean, 2.5),
        (np.median, 2.5),
        (np.nanmedian, 2.5),
        (np.min, 1),
        (np.max, 4),
        (np.nanmin, 1),
        (np.nanmax, 4),
    ],
)
def test_numpy_ops(numpy_op, expected):
    # GH8383
    # 对 Series([1, 2, 3, 4]) 执行 numpy_op 操作
    result = numpy_op(Series([1, 2, 3, 4]))
    # 断言操作的结果与预期值相等
    assert result == expected


# 参数化测试不同的 nanops 操作
@pytest.mark.parametrize(
    "operation",
    [
        nanops.nanany,
        nanops.nanall,
        nanops.nansum,
        nanops.nanmean,
        nanops.nanmedian,
        nanops.nanstd,
        nanops.nanvar,
        nanops.nansem,
        nanops.nanargmax,
        nanops.nanargmin,
        nanops.nanmax,
        nanops.nanmin,
        nanops.nanskew,
        nanops.nankurt,
        nanops.nanprod,
    ],
)
def test_nanops_independent_of_mask_param(operation):
    # GH22764
    # 创建包含 NaN 的 Series 对象
    ser = Series([1, 2, np.nan, 3, np.nan, 4])
    # 创建 mask，标记 NaN 值的位置
    mask = ser.isna()
    # 计算不考虑 mask 的操作 operation 在 ser._values 上的结果
    median_expected = operation(ser._values)
    # 计算考虑 mask 的操作 operation 在 ser._values 上的结果
    median_result = operation(ser._values, mask=mask)
    # 断言两种计算方式得到的结果相等
    assert median_expected == median_result


# 参数化测试检查 below_min_count 函数的负值或零值 min_count 参数
@pytest.mark.parametrize("min_count", [-1, 0])
def test_check_below_min_count_negative_or_zero_min_count(min_count):
    # GH35227
    # 调用 check_below_min_count 函数，检查结果是否为 False
    result = nanops.check_below_min_count((21, 37), None, min_count)
    # 断言结果为预期的 False
    expected_result = False
    assert result == expected_result
# 使用 pytest 的 mark.parametrize 装饰器，对 test_check_below_min_count_positive_min_count 函数参数化测试
@pytest.mark.parametrize(
    "mask", [None, np.array([False, False, True]), np.array([True] + 9 * [False])]
)
# 再次使用 parametrize 装饰器，对 min_count 和 expected_result 参数化测试
@pytest.mark.parametrize("min_count, expected_result", [(1, False), (101, True)])
# 定义测试函数 test_check_below_min_count_positive_min_count，用于测试 nanops.check_below_min_count 函数
def test_check_below_min_count_positive_min_count(mask, min_count, expected_result):
    # GH35227: GitHub issue reference
    # 定义 shape 变量，表示数组形状为 (10, 10)
    shape = (10, 10)
    # 调用 nanops.check_below_min_count 函数，获取结果
    result = nanops.check_below_min_count(shape, mask, min_count)
    # 断言结果是否符合预期
    assert result == expected_result


# 使用自定义的 skip_if_windows 和 skip_if_32bit 装饰器来跳过特定条件下的测试
@td.skip_if_windows
@td.skip_if_32bit
# 使用 parametrize 装饰器，对 min_count 和 expected_result 参数化测试
@pytest.mark.parametrize("min_count, expected_result", [(1, False), (2812191852, True)])
# 定义测试函数 test_check_below_min_count_large_shape，用于测试 nanops.check_below_min_count 函数
def test_check_below_min_count_large_shape(min_count, expected_result):
    # GH35227 large shape used to show that the issue is fixed
    # 定义 shape 变量，表示数组形状为 (2244367, 1253)
    shape = (2244367, 1253)
    # 调用 nanops.check_below_min_count 函数，传入大数组形状的参数，获取结果
    result = nanops.check_below_min_count(shape, mask=None, min_count=min_count)
    # 断言结果是否符合预期
    assert result == expected_result


# 使用 parametrize 装饰器，对 func 参数化测试，依次传入 "nanmean" 和 "nansum"
@pytest.mark.parametrize("func", ["nanmean", "nansum"])
# 定义测试函数 test_check_bottleneck_disallow，用于测试 nanops._bn_ok_dtype 函数
def test_check_bottleneck_disallow(any_real_numpy_dtype, func):
    # GH 42878 bottleneck sometimes produces unreliable results for mean and sum
    # 断言 nanops._bn_ok_dtype 函数对给定 dtype 和 func 返回 False
    assert not nanops._bn_ok_dtype(np.dtype(any_real_numpy_dtype).type, func)


# 使用 parametrize 装饰器，对 val 参数化测试，依次传入 2**55, -(2**55), 20150515061816532
@pytest.mark.parametrize("val", [2**55, -(2**55), 20150515061816532])
# 定义测试函数 test_nanmean_overflow，用于测试 Series 对象的 mean 方法
def test_nanmean_overflow(disable_bottleneck, val):
    # GH 10155: GitHub issue reference
    # 创建一个包含指定值和数据类型的 Series 对象
    ser = Series(val, index=range(500), dtype=np.int64)
    # 调用 Series 的 mean 方法，获取计算结果
    result = ser.mean()
    # 使用 numpy 计算 Series 值的均值
    np_result = ser.values.mean()
    # 断言计算结果与预期值相等
    assert result == val
    assert result == np_result
    # 断言结果的数据类型为 np.float64
    assert result.dtype == np.float64


# 使用 parametrize 装饰器，对 dtype 参数化测试，依次传入不同的 numpy 数据类型
@pytest.mark.parametrize(
    "dtype",
    [
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        getattr(np, "float128", None),
    ],
)
# 使用 parametrize 装饰器，对 method 参数化测试，依次传入 "mean", "std", "var", "skew", "kurt", "min", "max"
@pytest.mark.parametrize("method", ["mean", "std", "var", "skew", "kurt", "min", "max"])
# 定义测试函数 test_returned_dtype，用于测试 Series 对象在不同方法下返回的数据类型
def test_returned_dtype(disable_bottleneck, dtype, method):
    if dtype is None:
        pytest.skip("np.float128 not available")

    # 创建一个包含整数序列的 Series 对象，数据类型由参数 dtype 指定
    ser = Series(range(10), dtype=dtype)
    # 调用 getattr 方法获取对应的方法名 method，并执行该方法
    result = getattr(ser, method)()
    # 如果数据类型为整数，并且方法不是 "min" 或 "max"，则断言结果的数据类型为 np.float64
    if is_integer_dtype(dtype) and method not in ["min", "max"]:
        assert result.dtype == np.float64
    else:
        # 否则断言结果的数据类型与传入的 dtype 一致
        assert result.dtype == dtype
```