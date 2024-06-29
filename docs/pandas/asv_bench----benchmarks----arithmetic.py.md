# `D:\src\scipysrc\pandas\asv_bench\benchmarks\arithmetic.py`

```
# 导入操作符模块，用于进行各种操作符的运算
import operator
# 导入警告模块，用于管理或处理可能的警告信息
import warnings

# 导入 NumPy 库，并简称为 np
import numpy as np

# 导入 Pandas 库，并简称为 pd
import pandas as pd
# 从 Pandas 库中导入特定的类和函数
from pandas import (
    DataFrame,      # 数据帧类，用于操作二维表格数据
    Index,          # 索引对象，用于标记和引用数据
    Series,         # 系列类，用于操作一维标记数据
    Timestamp,      # 时间戳类，表示时间点
    date_range,     # 时间范围生成函数
    to_timedelta,   # 将数据转换为时间差类型
)

# 从自定义模块中导入指定内容
from .pandas_vb_common import numeric_dtypes

# 尝试导入 pandas.core.computation.expressions 模块，若失败则导入 pandas.computation.expressions 模块
try:
    import pandas.core.computation.expressions as expr
except ImportError:
    import pandas.computation.expressions as expr

# 尝试导入 pandas.tseries.holiday 模块，若失败则忽略
try:
    import pandas.tseries.holiday
except ImportError:
    pass


# 定义一个带有参数化属性的类
class IntFrameWithScalar:
    # 参数化的列表，包含不同的数据类型、标量值和操作符
    params = [
        [np.float64, np.int64],  # 数据类型
        [2, 3.0, np.int32(4), np.float64(5)],  # 标量值
        [  # 操作符列表
            operator.add, operator.sub, operator.mul, operator.truediv,
            operator.floordiv, operator.pow, operator.mod, operator.eq,
            operator.ne, operator.gt, operator.ge, operator.lt, operator.le,
        ],
    ]
    param_names = ["dtype", "scalar", "op"]  # 参数名称列表

    # 设置方法，在测试执行前准备数据
    def setup(self, dtype, scalar, op):
        # 创建一个随机数组，转换为指定的数据类型，并创建数据帧对象
        arr = np.random.randn(20000, 100)
        self.df = DataFrame(arr.astype(dtype))

    # 测试方法，执行数据帧和标量之间的操作
    def time_frame_op_with_scalar(self, dtype, scalar, op):
        op(self.df, scalar)


# 定义一个带有填充值操作的类
class OpWithFillValue:
    # 设置方法，在测试执行前准备数据
    def setup(self):
        # 创建一个包含一百万元素的数组，并使用其创建数据帧对象和系列对象
        arr = np.arange(10**6)
        df = DataFrame({"A": arr})
        ser = df["A"]

        self.df = df
        self.ser = ser

    # 测试方法，执行数据帧和填充值的操作（无缺失值情况）
    def time_frame_op_with_fill_value_no_nas(self):
        self.df.add(self.df, fill_value=4)

    # 测试方法，执行系列对象和填充值的操作（无缺失值情况）
    def time_series_op_with_fill_value_no_nas(self):
        self.ser.add(self.ser, fill_value=4)


# 定义一个带有混合数据类型和系列对象操作的类
class MixedFrameWithSeriesAxis:
    params = [
        [  # 操作名列表
            "eq", "ne", "lt", "le", "ge", "gt", "add", "sub", "truediv", "floordiv", "mul", "pow",
        ]
    ]
    param_names = ["opname"]  # 参数名称列表

    # 设置方法，在测试执行前准备数据
    def setup(self, opname):
        # 创建一个包含一百万元素的数组，并根据需求创建数据帧对象、系列对象和行对象
        arr = np.arange(10**6).reshape(1000, -1)
        df = DataFrame(arr)
        df["C"] = 1.0
        self.df = df
        self.ser = df[0]
        self.row = df.iloc[0]

    # 测试方法，执行数据帧和系列对象在 axis=0 方向上的操作
    def time_frame_op_with_series_axis0(self, opname):
        getattr(self.df, opname)(self.ser, axis=0)

    # 测试方法，执行数据帧和系列对象在 axis=1 方向上的操作
    def time_frame_op_with_series_axis1(self, opname):
        getattr(operator, opname)(self.df, self.ser)

    # 排除比较操作，因为它们不执行对齐操作而会引发异常
    time_frame_op_with_series_axis1.params = [params[0][6:]]


# 定义一个包含大量列和混合数据类型的类
class FrameWithFrameWide:
    # 参数化列表，包含不同的操作符和数据形状
    params = [
        [  # 操作符列表
            operator.add, operator.floordiv, operator.gt,
        ],
        [  # 形状列表，表示（行数，列数）
            (1_000_000, 10), (100_000, 100), (10_000, 1000), (1000, 10_000),
        ],
    ]
    param_names = ["op", "shape"]  # 参数名称列表
    # 设置函数，用于初始化对象状态，根据给定的操作和形状
    def setup(self, op, shape):
        # 根据形状确定行数和列数
        n_rows, n_cols = shape

        # 如果操作是整除运算符（//），则减少行数以降低数据量
        if op is operator.floordiv:
            n_rows = n_rows // 10

        # 生成两个不同数据类型的随机数组成的数据帧
        arr1 = np.random.randn(n_rows, n_cols // 2).astype("f8")  # 使用双精度浮点数类型生成随机数组
        arr2 = np.random.randn(n_rows, n_cols // 2).astype("f4")  # 使用单精度浮点数类型生成随机数组
        df = pd.concat([DataFrame(arr1), DataFrame(arr2)], axis=1, ignore_index=True)
        # 确保数据帧已经被巩固以提高性能
        df._consolidate_inplace()

        # TODO: GH#33198 这里的设置应该不需要两步
        # 再次生成不同数据类型的随机数组成的数据帧
        arr1 = np.random.randn(n_rows, max(n_cols // 4, 3)).astype("f8")  # 使用双精度浮点数类型生成随机数组
        arr2 = np.random.randn(n_rows, n_cols // 2).astype("i8")           # 使用64位整数类型生成随机数组
        arr3 = np.random.randn(n_rows, n_cols // 4).astype("f8")           # 使用双精度浮点数类型生成随机数组
        df2 = pd.concat(
            [DataFrame(arr1), DataFrame(arr2), DataFrame(arr3)],
            axis=1,
            ignore_index=True,
        )
        # 确保数据帧已经被巩固以提高性能
        df2._consolidate_inplace()

        # 将生成的数据帧分别赋值给对象的左右属性
        self.left = df
        self.right = df2

    # 执行时间测量函数，用于对两个数据帧执行给定操作
    def time_op_different_blocks(self, op, shape):
        # 左右数据帧的数据块和数据类型不完全对齐
        op(self.left, self.right)

    # 执行时间测量函数，用于对同一个数据帧执行给定操作
    def time_op_same_blocks(self, op, shape):
        # 左右数据帧的数据块和数据类型完全对齐
        op(self.left, self.left)
class Ops:
    # 定义类变量 params 为包含两个列表的列表，用于参数设置
    params = [[True, False], ["default", 1]]
    # 定义类变量 param_names，包含参数的名称列表
    param_names = ["use_numexpr", "threads"]

    # 设置方法，初始化数据框和数据框2，并根据参数设置更新运算库线程和使用情况
    def setup(self, use_numexpr, threads):
        # 创建大小为20000x100的随机数据框
        self.df = DataFrame(np.random.randn(20000, 100))
        # 创建大小为20000x100的随机数据框2
        self.df2 = DataFrame(np.random.randn(20000, 100))

        # 如果线程参数不是"default"，设置表达式运算线程数
        if threads != "default":
            expr.set_numexpr_threads(threads)
        # 如果不使用 numexpr，设置为 False
        if not use_numexpr:
            expr.set_use_numexpr(False)

    # 数据框加法计时方法
    def time_frame_add(self, use_numexpr, threads):
        # 对数据框和数据框2进行加法操作
        self.df + self.df2

    # 数据框乘法计时方法
    def time_frame_mult(self, use_numexpr, threads):
        # 对数据框和数据框2进行乘法操作
        self.df * self.df2

    # 数据框与操作计时方法
    def time_frame_multi_and(self, use_numexpr, threads):
        # 执行数据框大于0且数据框2大于0的布尔运算
        self.df[(self.df > 0) & (self.df2 > 0)]

    # 数据框比较计时方法
    def time_frame_comparison(self, use_numexpr, threads):
        # 执行数据框大于数据框2的布尔运算
        self.df > self.df2

    # 清理方法，恢复 numexpr 使用并设置默认线程数
    def teardown(self, use_numexpr, threads):
        # 恢复使用 numexpr
        expr.set_use_numexpr(True)
        # 设置默认线程数
        expr.set_numexpr_threads()


class Ops2:
    # 设置方法，初始化数据框和数据框2，以及整数数据框和整数数据框2
    def setup(self):
        N = 10**3
        # 创建大小为N*N的随机数据框
        self.df = DataFrame(np.random.randn(N, N))
        # 创建大小为N*N的随机数据框2
        self.df2 = DataFrame(np.random.randn(N, N))

        # 创建大小为N*N的随机整数数据框
        self.df_int = DataFrame(
            np.random.randint(
                np.iinfo(np.int16).min, np.iinfo(np.int16).max, size=(N, N)
            )
        )
        # 创建大小为N*N的随机整数数据框2
        self.df2_int = DataFrame(
            np.random.randint(
                np.iinfo(np.int16).min, np.iinfo(np.int16).max, size=(N, N)
            )
        )

        # 创建大小为N的随机数据序列
        self.s = Series(np.random.randn(N))

    # 浮点数除法计时方法
    def time_frame_float_div(self):
        # 执行数据框的浮点数除法
        self.df // self.df2

    # 浮点数除零计时方法
    def time_frame_float_div_by_zero(self):
        # 执行数据框除以零的计算
        self.df / 0

    # 浮点数向下取整除零计时方法
    def time_frame_float_floor_by_zero(self):
        # 执行数据框向下取整除以零的计算
        self.df // 0

    # 整数除零计时方法
    def time_frame_int_div_by_zero(self):
        # 执行整数数据框除以零的计算
        self.df_int / 0

    # 整数取模计时方法
    def time_frame_int_mod(self):
        # 执行整数数据框取模操作
        self.df_int % self.df2_int

    # 浮点数取模计时方法
    def time_frame_float_mod(self):
        # 执行数据框取模操作
        self.df % self.df2

    # 点乘计时方法
    def time_frame_dot(self):
        # 执行数据框的点乘操作
        self.df.dot(self.df2)

    # 序列点乘计时方法
    def time_series_dot(self):
        # 执行序列的点乘操作
        self.s.dot(self.s)

    # 数据框序列点乘计时方法
    def time_frame_series_dot(self):
        # 执行数据框与序列的点乘操作
        self.df.dot(self.s)


class Timeseries:
    # 定义类变量 params 为包含时区设置的列表
    params = [None, "US/Eastern"]
    # 定义类变量 param_names，包含时区参数的名称
    param_names = ["tz"]

    # 设置方法，根据时区参数初始化时间序列和不同分辨率的时间戳
    def setup(self, tz):
        N = 10**6
        halfway = (N // 2) - 1
        # 创建包含N个日期的时间序列，频率为分钟，带有时区设置
        self.s = Series(date_range("20010101", periods=N, freq="min", tz=tz))
        # 获取时间序列的中间时间戳
        self.ts = self.s[halfway]

        # 创建包含N个日期的时间序列，频率为秒，带有时区设置
        self.s2 = Series(date_range("20010101", periods=N, freq="s", tz=tz))
        # 创建不同分辨率的时间戳，带有时区设置
        self.ts_different_reso = Timestamp("2001-01-02", tz=tz)

    # 时间序列时间戳比较计时方法
    def time_series_timestamp_compare(self, tz):
        # 执行时间序列与时间戳比较操作
        self.s <= self.ts

    # 不同分辨率时间戳比较计时方法
    def time_series_timestamp_different_reso_compare(self, tz):
        # 执行时间序列与不同分辨率时间戳比较操作
        self.s <= self.ts_different_reso

    # 时间戳时间序列比较计时方法
    def time_timestamp_series_compare(self, tz):
        # 执行时间戳与时间序列比较操作
        self.ts >= self.s

    # 时间戳差分计算计时方法
    def time_timestamp_ops_diff(self, tz):
        # 执行时间序列的差分计算
        self.s2.diff()

    # 时间戳差分与移位计算计时方法
    def time_timestamp_ops_diff_with_shift(self, tz):
        # 执行时间序列与其移位的差分计算
        self.s - self.s.shift()


class IrregularOps:
    # 这里是未提供的类定义，没有需要添加的注释
    pass
    # 定义一个设置方法，用于初始化数据
    def setup(self):
        # 设置数据长度为 100000
        N = 10**5
        # 生成一个时间索引，从 "2000-01-01 00:00:00" 开始，频率为每秒一次，共 N 个时间点
        idx = date_range(start="1/1/2000", periods=N, freq="s")
        # 生成一个随机数序列，长度为 N，并以 idx 为索引
        s = Series(np.random.randn(N), index=idx)
        # 将随机数序列 s 随机抽样，并按抽样结果赋值给 self.left
        self.left = s.sample(frac=1)
        # 再次将随机数序列 s 随机抽样，并按抽样结果赋值给 self.right
        self.right = s.sample(frac=1)

    # 定义一个时间加法方法
    def time_add(self):
        # 计算 self.left 和 self.right 的加法，但未对结果进行任何操作
        self.left + self.right
class TimedeltaOps:
    # 设置测试数据
    def setup(self):
        # 创建一个包含1000000个元素的时间增量数组
        self.td = to_timedelta(np.arange(1000000))
        # 创建一个时间戳对象，表示为2000年
        self.ts = Timestamp("2000")

    # 测试时间增量与时间戳相加的性能
    def time_add_td_ts(self):
        self.td + self.ts


class CategoricalComparisons:
    # 参数化测试方法
    params = ["__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__"]
    param_names = ["op"]

    # 设置测试数据
    def setup(self, op):
        N = 10**5
        # 创建一个包含大量数据的有序分类数据
        self.cat = pd.Categorical(list("aabbcd") * N, ordered=True)

    # 测试分类数据的比较操作性能
    def time_categorical_op(self, op):
        getattr(self.cat, op)("b")


class IndexArithmetic:
    # 参数化测试类
    params = ["float", "int"]
    param_names = ["dtype"]

    # 设置测试数据
    def setup(self, dtype):
        N = 10**6
        # 根据参数化类型创建索引对象
        if dtype == "float":
            self.index = Index(np.arange(N), dtype=np.float64)
        elif dtype == "int":
            self.index = Index(np.arange(N), dtype=np.int64)

    # 测试索引对象的加法性能
    def time_add(self, dtype):
        self.index + 2

    # 测试索引对象的减法性能
    def time_subtract(self, dtype):
        self.index - 2

    # 测试索引对象的乘法性能
    def time_multiply(self, dtype):
        self.index * 2

    # 测试索引对象的除法性能
    def time_divide(self, dtype):
        self.index / 2

    # 测试索引对象的取模性能
    def time_modulo(self, dtype):
        self.index % 2


class NumericInferOps:
    # 参数化测试类，来源于 GH 7332
    params = numeric_dtypes
    param_names = ["dtype"]

    # 设置测试数据
    def setup(self, dtype):
        N = 5 * 10**5
        # 创建一个包含两列数据的数据帧，数据类型由参数决定
        self.df = DataFrame(
            {"A": np.arange(N).astype(dtype), "B": np.arange(N).astype(dtype)}
        )

    # 测试数据帧的列相加性能
    def time_add(self, dtype):
        self.df["A"] + self.df["B"]

    # 测试数据帧的列相减性能
    def time_subtract(self, dtype):
        self.df["A"] - self.df["B"]

    # 测试数据帧的列相乘性能
    def time_multiply(self, dtype):
        self.df["A"] * self.df["B"]

    # 测试数据帧的列相除性能
    def time_divide(self, dtype):
        self.df["A"] / self.df["B"]

    # 测试数据帧的列取模性能
    def time_modulo(self, dtype):
        self.df["A"] % self.df["B"]


class DateInferOps:
    # 来自 GH 7332
    def setup_cache(self):
        N = 5 * 10**5
        # 创建包含两列数据的数据帧，第一列是 datetime64 类型数据，第二列是 timedelta 类型数据
        df = DataFrame({"datetime64": np.arange(N).astype("datetime64[ms]")})
        df["timedelta"] = df["datetime64"] - df["datetime64"]
        return df

    # 测试日期时间列的减法性能
    def time_subtract_datetimes(self, df):
        df["datetime64"] - df["datetime64"]

    # 测试时间增量与日期时间相加的性能
    def time_timedelta_plus_datetime(self, df):
        df["timedelta"] + df["datetime64"]

    # 测试时间增量列相加的性能
    def time_add_timedeltas(self, df):
        df["timedelta"] + df["timedelta"]


hcal = pd.tseries.holiday.USFederalHolidayCalendar()
# 创建不适用 .apply_index() 方法的偏移量列表
non_apply = [
    pd.offsets.Day(),
    pd.offsets.BYearEnd(),
    pd.offsets.BYearBegin(),
    pd.offsets.BQuarterEnd(),
    pd.offsets.BQuarterBegin(),
    pd.offsets.BMonthEnd(),
    pd.offsets.BMonthBegin(),
    pd.offsets.CustomBusinessDay(),
    pd.offsets.CustomBusinessDay(calendar=hcal),
    pd.offsets.CustomBusinessMonthBegin(calendar=hcal),
    pd.offsets.CustomBusinessMonthEnd(calendar=hcal),
    pd.offsets.CustomBusinessMonthEnd(calendar=hcal),
]
# 创建其他偏移量列表
other_offsets = [
    pd.offsets.YearEnd(),
    pd.offsets.YearBegin(),
    pd.offsets.QuarterEnd(),
    pd.offsets.QuarterBegin(),
    pd.offsets.MonthEnd(),
]
    # 创建一个日期偏移对象，表示每月的开始日期
    pd.offsets.MonthBegin(),

    # 创建一个日期偏移对象，表示每两个月零两天的日期偏移量
    pd.offsets.DateOffset(months=2, days=2),

    # 创建一个日期偏移对象，表示工作日的日期偏移量
    pd.offsets.BusinessDay(),

    # 创建一个日期偏移对象，表示半月末的日期偏移量
    pd.offsets.SemiMonthEnd(),

    # 创建一个日期偏移对象，表示半月初的日期偏移量
    pd.offsets.SemiMonthBegin(),
# 定义一个列表 `offsets`，将 `non_apply` 和 `other_offsets` 合并
offsets = non_apply + other_offsets


class OffsetArrayArithmetic:
    # 参数 `params` 设置为 `offsets`，用于性能测试时的参数化
    params = offsets
    # 参数名称设置为 `"offset"`，用于性能测试时标识参数的名称
    param_names = ["offset"]

    # 设置方法，在性能测试之前准备数据
    def setup(self, offset):
        # 创建一个包含 10000 个时间戳的日期范围 `rng`
        N = 10000
        rng = date_range(start="1/1/2000", periods=N, freq="min")
        # 将日期范围保存到实例变量 `self.rng`
        self.rng = rng
        # 创建一个 Pandas Series，以 `rng` 为数据
        self.ser = Series(rng)

    # 测试方法，测量将 `self.ser` 与 `offset` 相加的时间
    def time_add_series_offset(self, offset):
        # 捕获可能的警告信息
        with warnings.catch_warnings(record=True):
            self.ser + offset

    # 测试方法，测量将 `self.rng` 与 `offset` 相加的时间
    def time_add_dti_offset(self, offset):
        # 捕获可能的警告信息
        with warnings.catch_warnings(record=True):
            self.rng + offset


class ApplyIndex:
    # 参数 `params` 设置为 `other_offsets`，用于性能测试时的参数化
    params = other_offsets
    # 参数名称设置为 `"offset"`，用于性能测试时标识参数的名称
    param_names = ["offset"]

    # 设置方法，在性能测试之前准备数据
    def setup(self, offset):
        # 创建一个包含 10000 个时间戳的日期范围 `rng`
        N = 10000
        rng = date_range(start="1/1/2000", periods=N, freq="min")
        # 将日期范围保存到实例变量 `self.rng`
        self.rng = rng

    # 测试方法，测量将 `self.rng` 与 `offset` 相加的时间
    def time_apply_index(self, offset):
        self.rng + offset


class BinaryOpsMultiIndex:
    # 参数 `params` 设置为包含四种操作的列表，用于性能测试时的参数化
    params = ["sub", "add", "mul", "div"]
    # 参数名称设置为 `"func"`，用于性能测试时标识参数的名称
    param_names = ["func"]

    # 设置方法，在性能测试之前准备数据
    def setup(self, func):
        # 创建一个包含从 "20200101 00:00" 到 "20200102 0:00" 的时间戳数组 `array`
        array = date_range("20200101 00:00", "20200102 0:00", freq="s")
        # 创建一个包含 30 个字符串的列表 `level_0_names`
        level_0_names = [str(i) for i in range(30)]
        # 使用 `level_0_names` 和 `array` 创建一个多级索引 `index`
        index = pd.MultiIndex.from_product([level_0_names, array])
        # 创建一个包含两列随机数据的 DataFrame `self.df`，以 `index` 为索引
        column_names = ["col_1", "col_2"]
        self.df = DataFrame(
            np.random.rand(len(index), 2), index=index, columns=column_names
        )
        # 创建一个包含随机整数数据的 DataFrame `self.arg_df`，以 `level_0_names` 为索引
        self.arg_df = DataFrame(
            np.random.randint(1, 10, (len(level_0_names), 2)),
            index=level_0_names,
            columns=column_names,
        )

    # 测试方法，测量在多级索引 DataFrame `self.df` 上执行二元操作 `func` 的时间
    def time_binary_op_multiindex(self, func):
        # 使用 `getattr` 动态调用 `self.df` 的方法 `func`，传递 `self.arg_df` 作为参数
        getattr(self.df, func)(self.arg_df, level=0)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
```