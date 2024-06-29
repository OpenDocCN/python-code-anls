# `D:\src\scipysrc\pandas\pandas\tests\extension\decimal\test_decimal.py`

```
# 从未来导入类型注解功能，使得可以在类中使用类型注解
from __future__ import annotations

# 导入 decimal 模块，用于处理高精度的十进制数运算
import decimal
# 导入 operator 模块，提供了一系列对各种 Python 数据类型进行操作的函数
import operator

# 导入 numpy 库，用于支持大规模数据数组和矩阵运算
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas 库，提供了用于数据操作和分析的数据结构和工具
import pandas as pd
# 导入 pandas 内部测试工具模块
import pandas._testing as tm
# 导入 pandas 扩展测试模块的基类
from pandas.tests.extension import base
# 导入 pandas 扩展 decimal 数组相关模块
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
    make_data,
    to_decimal,
)


# 为 decimal 类型创建一个固定的测试数据类型对象
@pytest.fixture
def dtype():
    return DecimalDtype()


# 生成一个包含随机数据的 DecimalArray 实例作为测试数据
@pytest.fixture
def data():
    return DecimalArray(make_data())


# 生成一个包含值为 2 的 DecimalArray 实例作为测试数据
@pytest.fixture
def data_for_twos():
    return DecimalArray([decimal.Decimal(2) for _ in range(100)])


# 生成一个包含 NaN 和 1 的 DecimalArray 实例作为测试数据
@pytest.fixture
def data_missing():
    return DecimalArray([decimal.Decimal("NaN"), decimal.Decimal(1)])


# 生成一个用于排序的 DecimalArray 实例，包含 1、2、0 三个数值
@pytest.fixture
def data_for_sorting():
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("2"), decimal.Decimal("0")]
    )


# 生成一个用于排序的 DecimalArray 实例，包含 1、NaN、0 三个数值
@pytest.fixture
def data_missing_for_sorting():
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("NaN"), decimal.Decimal("0")]
    )


# 定义一个用于比较 NaN 的比较函数
@pytest.fixture
def na_cmp():
    return lambda x, y: x.is_nan() and y.is_nan()


# 生成一个用于分组操作的 DecimalArray 实例，包含多个数字和 NaN 值
@pytest.fixture
def data_for_grouping():
    b = decimal.Decimal("1.0")
    a = decimal.Decimal("0.0")
    c = decimal.Decimal("2.0")
    na = decimal.Decimal("NaN")
    return DecimalArray([b, b, na, na, a, a, b, c])


# DecimalArray 扩展测试的类，继承自 base.ExtensionTests
class TestDecimalArray(base.ExtensionTests):
    
    # 返回预期的异常类型，这里总是返回 None
    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | None:
        return None

    # 判断是否支持对 Series 进行指定的减少操作
    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return True

    # 执行减少操作的检查，如果是 "count" 操作则调用基类方法，否则比较结果与预期是否接近
    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        if op_name == "count":
            return super().check_reduce(ser, op_name, skipna)
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(np.asarray(ser), op_name)()
            tm.assert_almost_equal(result, expected)

    # 对 Series 进行数值减少操作的测试，根据 all_numeric_reductions 参数进行选择性测试
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        if all_numeric_reductions in ["kurt", "skew", "sem", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    # 对 DataFrame 进行数值减少操作的测试，根据 all_numeric_reductions 参数进行选择性测试
    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        op_name = all_numeric_reductions
        if op_name in ["skew", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)

        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    # 对单个标量值与 Series 进行比较操作的测试
    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0.5)
    # 定义一个测试方法，用于比较数组的值和给定的比较操作符
    def test_compare_array(self, data, comparison_op):
        # 创建一个 Pandas Series 对象，使用传入的数据
        ser = pd.Series(data)

        # 使用随机数生成器创建一个随机数组，数组长度与输入数据相同
        alter = np.random.default_rng(2).choice([-1, 0, 1], len(data))
        # 随机将值乘以2、除以2或保持不变
        other = pd.Series(data) * [decimal.Decimal(pow(2.0, i)) for i in alter]
        # 调用内部方法，比较 ser 与 data，使用指定的比较操作符，并与 other 比较
        self._compare_other(ser, data, comparison_op, other)

    # 定义一个测试方法，用于将序列与数组进行算术运算
    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # 获取算术运算符的名称
        op_name = all_arithmetic_operators
        # 创建一个 Pandas Series 对象，使用传入的数据
        ser = pd.Series(data)

        # 获取当前 Decimal 上下文
        context = decimal.getcontext()
        # 保存当前除零异常设置和无效操作异常设置
        divbyzerotrap = context.traps[decimal.DivisionByZero]
        invalidoptrap = context.traps[decimal.InvalidOperation]
        # 禁用除零异常和无效操作异常
        context.traps[decimal.DivisionByZero] = 0
        context.traps[decimal.InvalidOperation] = 0

        # 创建一个新的 Pandas Series 对象，其元素为原始数据乘以100后转换为整数
        other = pd.Series([int(d * 100) for d in data])
        # 调用内部方法，检查 ser 与 op_name 指定的操作符，在 other 上的操作结果
        self.check_opname(ser, op_name, other)

        # 如果操作符列表中不包含 "mod"，再检查 ser 与 op_name 操作符在 ser 自身乘以2后的结果
        if "mod" not in op_name:
            self.check_opname(ser, op_name, ser * 2)

        # 分别检查 ser 与 op_name 操作符在值为0和5时的结果
        self.check_opname(ser, op_name, 0)
        self.check_opname(ser, op_name, 5)

        # 恢复之前保存的除零异常设置和无效操作异常设置
        context.traps[decimal.DivisionByZero] = divbyzerotrap
        context.traps[decimal.InvalidOperation] = invalidoptrap

    # 定义一个测试方法，测试在 DataFrame 中使用 fillna 方法时的行为
    def test_fillna_frame(self, data_missing):
        # 设置警告消息
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        # 断言产生 DeprecationWarning 警告，匹配给定的消息，不检查堆栈级别
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            # 调用父类的 test_fillna_frame 方法，传入缺失数据
            super().test_fillna_frame(data_missing)

    # 定义一个测试方法，测试在 Series 中使用 fillna 方法时的行为
    def test_fillna_series(self, data_missing):
        # 设置警告消息
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        # 断言产生 DeprecationWarning 警告，匹配给定的消息，不检查堆栈级别
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            # 调用父类的 test_fillna_series 方法，传入缺失数据
            super().test_fillna_series(data_missing)

    # 定义一个测试方法，测试在填充 None 时的行为
    def test_fillna_with_none(self, data_missing):
        # GH#57723
        # EAs that don't have special logic for None will raise, unlike pandas'
        # which interpret None as the NA value for the dtype.
        # 设置异常消息
        msg = "conversion from NoneType to Decimal is not supported"
        # 断言产生 TypeError 异常，匹配给定的消息
        with pytest.raises(TypeError, match=msg):
            # 调用父类的 test_fillna_with_none 方法，传入缺失数据
            super().test_fillna_with_none(data_missing)

    # 定义一个测试方法，测试在 DataFrame 中使用 fillna 方法时，设置填充限制的行为
    def test_fillna_limit_frame(self, data_missing):
        # GH#58001
        # 设置警告消息
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        # 断言产生 DeprecationWarning 警告，匹配给定的消息，不检查堆栈级别
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            # 调用父类的 test_fillna_limit_frame 方法，传入缺失数据
            super().test_fillna_limit_frame(data_missing)

    # 定义一个测试方法，测试在 Series 中使用 fillna 方法时，设置填充限制的行为
    def test_fillna_limit_series(self, data_missing):
        # GH#58001
        # 设置警告消息
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        # 断言产生 DeprecationWarning 警告，匹配给定的消息，不检查堆栈级别
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            # 调用父类的 test_fillna_limit_series 方法，传入缺失数据
            super().test_fillna_limit_series(data_missing)

    # 使用 pytest.mark.parametrize 装饰器，参数化 dropna 参数为 True 和 False
    # 测试 Series 对象的 value_counts 方法，检查频数统计是否正确
    def test_value_counts(self, all_data, dropna):
        # 取前 10 行数据进行测试
        all_data = all_data[:10]
        
        # 如果 dropna 参数为 True，则从 all_data 中筛选出非空值构成的数组
        if dropna:
            other = np.array(all_data[~all_data.isna()])
        else:
            # 如果 dropna 参数为 False，则直接使用 all_data
            other = all_data

        # 对 all_data 进行 value_counts 统计，根据 dropna 参数决定是否包含 NaN 值
        vcs = pd.Series(all_data).value_counts(dropna=dropna)
        # 对 other 进行 value_counts 统计，根据 dropna 参数决定是否包含 NaN 值
        vcs_ex = pd.Series(other).value_counts(dropna=dropna)

        # 设置 decimal 的本地上下文，避免在比较 Decimal("NAN") 和 Decimal(2) 时引发异常
        with decimal.localcontext() as ctx:
            ctx.traps[decimal.InvalidOperation] = False

            # 对 vcs 进行按索引排序
            result = vcs.sort_index()
            # 对 vcs_ex 进行按索引排序
            expected = vcs_ex.sort_index()

        # 使用测试框架的方法比较两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    # 测试 Series 对象的字符串表示形式（repr）
    def test_series_repr(self, data):
        # 创建一个 Series 对象
        ser = pd.Series(data)
        # 断言 Series 对象的字符串表示中包含数据类型的名称
        assert data.dtype.name in repr(ser)
        # 断言 Series 对象的字符串表示中包含特定的自定义信息 "Decimal: "
        assert "Decimal: " in repr(ser)

    # 标记该测试为预期失败，并说明失败的原因
    @pytest.mark.xfail(reason="Inconsistent array-vs-scalar behavior")
    # 使用参数化测试，对给定的一元通用函数进行等价性测试
    @pytest.mark.parametrize("ufunc", [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data, ufunc):
        # 调用父类方法，测试给定数据和一元通用函数的等价性
        super().test_unary_ufunc_dunder_equivalence(data, ufunc)
def test_take_na_value_other_decimal():
    # 创建一个 DecimalArray 对象，包含两个 Decimal 类型的数据：1.0 和 2.0
    arr = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    # 使用 take 方法，选择索引为 0 和 -1 的元素，允许填充并填充值为 -1.0
    result = arr.take([0, -1], allow_fill=True, fill_value=decimal.Decimal("-1.0"))
    # 创建一个预期的 DecimalArray 对象，包含两个 Decimal 类型的数据：1.0 和 -1.0
    expected = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("-1.0")])
    # 使用 assert_extension_array_equal 方法比较 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)


def test_series_constructor_coerce_data_to_extension_dtype():
    # 创建一个 DecimalDtype 类型的对象
    dtype = DecimalDtype()
    # 创建一个 Pandas Series 对象，其数据为 [0, 1, 2]，数据类型为 DecimalDtype
    ser = pd.Series([0, 1, 2], dtype=dtype)

    # 创建一个 DecimalArray 对象，包含三个 Decimal 类型的数据：0, 1, 2，数据类型为 DecimalDtype
    arr = DecimalArray(
        [decimal.Decimal(0), decimal.Decimal(1), decimal.Decimal(2)],
        dtype=dtype,
    )
    # 创建一个预期的 Pandas Series 对象，数据为 arr
    exp = pd.Series(arr)
    # 使用 assert_series_equal 方法比较 ser 和 exp 是否相等
    tm.assert_series_equal(ser, exp)


def test_series_constructor_with_dtype():
    # 创建一个 DecimalArray 对象，包含一个 Decimal 类型的数据：10.0
    arr = DecimalArray([decimal.Decimal("10.0")])
    # 使用 DecimalDtype 类型创建一个 Pandas Series 对象，数据为 arr
    result = pd.Series(arr, dtype=DecimalDtype())
    # 创建一个预期的 Pandas Series 对象，数据为 arr
    expected = pd.Series(arr)
    # 使用 assert_series_equal 方法比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 使用 "int64" 数据类型创建一个 Pandas Series 对象，数据为 [10]
    result = pd.Series(arr, dtype="int64")
    # 创建一个预期的 Pandas Series 对象，数据为 [10]
    expected = pd.Series([10])
    # 使用 assert_series_equal 方法比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_dataframe_constructor_with_dtype():
    # 创建一个 DecimalArray 对象，包含一个 Decimal 类型的数据：10.0
    arr = DecimalArray([decimal.Decimal("10.0")])

    # 使用 DecimalDtype 类型创建一个 Pandas DataFrame 对象，包含一列名为 "A"，数据为 arr
    result = pd.DataFrame({"A": arr}, dtype=DecimalDtype())
    # 创建一个预期的 Pandas DataFrame 对象，包含一列名为 "A"，数据为 arr
    expected = pd.DataFrame({"A": arr})
    # 使用 assert_frame_equal 方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 使用 "int64" 数据类型创建一个 Pandas DataFrame 对象，包含一列名为 "A"，数据为 [10]
    result = pd.DataFrame({"A": arr}, dtype="int64")
    # 创建一个预期的 Pandas DataFrame 对象，包含一列名为 "A"，数据为 [10]
    expected = pd.DataFrame({"A": [10]})
    # 使用 assert_frame_equal 方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_astype_dispatches(frame_or_series):
    # 这是一个特定于数据类型的测试，确保 Series[decimal].astype 能够顺利调用到 ExtensionArray.astype
    # 设计一个可靠的 smoke test 以适用于任意数据类型是困难的。
    # 创建一个 Pandas Series 对象，数据为 DecimalArray([decimal.Decimal(2)])，名称为 "a"
    data = pd.Series(DecimalArray([decimal.Decimal(2)]), name="a")
    # 创建一个 decimal.Context 对象
    ctx = decimal.Context()
    ctx.prec = 5

    # 将 data 转换为 frame_or_series 类型的对象
    data = frame_or_series(data)

    # 将数据类型转换为 DecimalDtype(ctx)
    result = data.astype(DecimalDtype(ctx))

    # 如果 frame_or_series 是 pd.DataFrame，则仅保留 "a" 列
    if frame_or_series is pd.DataFrame:
        result = result["a"]

    # 断言 result 的数据类型的精度是否等于 ctx 的精度
    assert result.dtype.context.prec == ctx.prec


class DecimalArrayWithoutFromSequence(DecimalArray):
    """用于测试 _from_sequence 错误处理的辅助类。"""

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # 抛出 KeyError 异常，用于测试
        raise KeyError("For the test")


class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):
    @classmethod
    def _create_arithmetic_method(cls, op):
        # 返回 _create_method 方法的调用结果，禁用到 dtype 的强制转换
        return cls._create_method(op, coerce_to_dtype=False)


DecimalArrayWithoutCoercion._add_arithmetic_ops()


def test_combine_from_sequence_raises(monkeypatch):
    # https://github.com/pandas-dev/pandas/issues/22850
    # 设置 cls 变量为 DecimalArrayWithoutFromSequence 类
    cls = DecimalArrayWithoutFromSequence

    @classmethod
    def construct_array_type(cls):
        # 返回 DecimalArrayWithoutFromSequence 类型的对象
        return DecimalArrayWithoutFromSequence

    # 使用 monkeypatch 设置 DecimalDtype.construct_array_type 方法为 construct_array_type
    monkeypatch.setattr(DecimalDtype, "construct_array_type", construct_array_type)

    # 创建一个 DecimalArrayWithoutFromSequence 对象，包含两个 Decimal 类型的数据：1.0 和 2.0
    arr = cls([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    # 创建一个 Pandas Series 对象，数据为 arr
    ser = pd.Series(arr)
    # 使用 combine 方法对 ser 和 ser 执行 operator.add 操作
    result = ser.combine(ser, operator.add)

    # 注意：数据类型为 object
    # 创建一个预期的 Pandas Series 对象，包含两个 Decimal 对象作为数据，数据类型为 object
    expected = pd.Series(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    # 使用 Pandas Testing 模块中的 assert_series_equal 函数，比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "class_", [DecimalArrayWithoutFromSequence, DecimalArrayWithoutCoercion]
)
def test_scalar_ops_from_sequence_raises(class_):
    # 测试标量操作是否能够正确处理特定类型的数组实例
    # 如果操作返回一个数组实例，应当返回该实例；否则返回一个 ndarray
    arr = class_([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    # 执行数组加法操作
    result = arr + arr
    # 期望的结果是一个包含指定十进制数的 ndarray
    expected = np.array(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    # 使用测试工具验证结果是否符合预期
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "reverse, expected_div, expected_mod",
    [(False, [0, 1, 1, 2], [1, 0, 1, 0]), (True, [2, 1, 0, 0], [0, 0, 2, 2])],
)
def test_divmod_array(reverse, expected_div, expected_mod):
    # https://github.com/pandas-dev/pandas/issues/22930
    # 创建十进制数组
    arr = to_decimal([1, 2, 3, 4])
    if reverse:
        # 反向执行 divmod 操作
        div, mod = divmod(2, arr)
    else:
        # 正常执行 divmod 操作
        div, mod = divmod(arr, 2)
    # 转换为预期的十进制数组
    expected_div = to_decimal(expected_div)
    expected_mod = to_decimal(expected_mod)

    # 使用测试工具验证 div 和 mod 的值是否符合预期
    tm.assert_extension_array_equal(div, expected_div)
    tm.assert_extension_array_equal(mod, expected_mod)


def test_ufunc_fallback(data):
    # 从数据中选择前五个元素创建数组
    a = data[:5]
    # 创建带索引的 Pandas Series
    s = pd.Series(a, index=range(3, 8))
    # 对 Series 中的元素执行绝对值操作
    result = np.abs(s)
    # 创建预期结果的 Pandas Series
    expected = pd.Series(np.abs(a), index=range(3, 8))
    # 使用测试工具验证结果是否符合预期
    tm.assert_series_equal(result, expected)


def test_array_ufunc():
    # 创建十进制数组
    a = to_decimal([1, 2, 3])
    # 对数组执行指数函数
    result = np.exp(a)
    # 创建预期的十进制数组
    expected = to_decimal(np.exp(a._data))
    # 使用测试工具验证结果是否符合预期
    tm.assert_extension_array_equal(result, expected)


def test_array_ufunc_series():
    # 创建十进制数组
    a = to_decimal([1, 2, 3])
    # 创建带有十进制数组的 Pandas Series
    s = pd.Series(a)
    # 对 Series 中的元素执行指数函数
    result = np.exp(s)
    # 创建预期的 Pandas Series
    expected = pd.Series(to_decimal(np.exp(a._data)))
    # 使用测试工具验证结果是否符合预期
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_scalar_other():
    # 检查 _HANDLED_TYPES
    # 创建十进制数组
    a = to_decimal([1, 2, 3])
    # 创建带有十进制数组的 Pandas Series
    s = pd.Series(a)
    # 对 Series 中的元素执行加法操作
    result = np.add(s, decimal.Decimal(1))
    # 创建预期的 Pandas Series
    expected = pd.Series(np.add(a, decimal.Decimal(1)))
    # 使用测试工具验证结果是否符合预期
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_defer():
    # 创建十进制数组
    a = to_decimal([1, 2, 3])
    # 创建带有十进制数组的 Pandas Series
    s = pd.Series(a)

    # 创建预期的 Pandas Series
    expected = pd.Series(to_decimal([2, 4, 6]))
    # 执行两种顺序的加法操作
    r1 = np.add(s, a)
    r2 = np.add(a, s)

    # 使用测试工具验证结果是否符合预期
    tm.assert_series_equal(r1, expected)
    tm.assert_series_equal(r2, expected)


def test_groupby_agg():
    # 确保 agg 函数的结果推断为十进制类型
    # https://github.com/pandas-dev/pandas/issues/29141

    # 创建部分数据
    data = make_data()[:5]
    # 创建包含十进制数组的 DataFrame
    df = pd.DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data)}
    )

    # 单个键，选择列
    # 创建预期的 Pandas Series
    expected = pd.Series(to_decimal([data[0], data[3]]))
    # 执行 groupby 操作并使用 agg 函数
    result = df.groupby("id1")["decimals"].agg(lambda x: x.iloc[0])
    # 使用测试工具验证结果是否符合预期，关闭名称检查
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby(df["id1"]).agg(lambda x: x.iloc[0])
    # 使用测试工具验证结果是否符合预期，关闭名称检查
    tm.assert_series_equal(result, expected, check_names=False)

    # 多个键，选择列
    # 创建一个期望的 Series，包含特定的数据和多重索引
    expected = pd.Series(
        to_decimal([data[0], data[1], data[3]]),  # 将给定的数据转换为 Decimal 类型，并组成 Series
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 1)]),  # 创建一个多重索引的 Series
    )
    # 使用 DataFrame 的 groupby 方法按指定的列进行分组，并使用聚合函数取第一个元素
    result = df.groupby(["id1", "id2"])["decimals"].agg(lambda x: x.iloc[0])
    # 使用测试框架的函数来比较两个 Series 是否相等，忽略索引名的检查
    tm.assert_series_equal(result, expected, check_names=False)

    # 再次进行类似的操作，但是将 DataFrame 的一列分组后转换为 Series，再进行相同的聚合操作
    result = df["decimals"].groupby([df["id1"], df["id2"]]).agg(lambda x: x.iloc[0])
    # 使用测试框架的函数来比较两个 Series 是否相等，忽略索引名的检查
    tm.assert_series_equal(result, expected, check_names=False)

    # 对于包含多列的情况
    # 创建一个期望的 DataFrame，包含特定的数据和列
    expected = pd.DataFrame({"id2": [0, 1], "decimals": to_decimal([data[0], data[3]])})
    # 使用 DataFrame 的 groupby 方法按指定的列进行分组，并使用聚合函数取第一个元素
    result = df.groupby("id1").agg(lambda x: x.iloc[0])
    # 使用测试框架的函数来比较两个 DataFrame 是否相等，忽略列名的检查
    tm.assert_frame_equal(result, expected, check_names=False)
# 确保聚合结果被推断为十进制数据类型
# 参考：https://github.com/pandas-dev/pandas/issues/29141
def test_groupby_agg_ea_method(monkeypatch):
    # 定义自定义方法 DecimalArray__my_sum，用于 DecimalArray 类型的求和操作
    def DecimalArray__my_sum(self):
        return np.sum(np.array(self))

    # 将自定义的 DecimalArray__my_sum 方法设置为 DecimalArray 类的 my_sum 属性
    monkeypatch.setattr(DecimalArray, "my_sum", DecimalArray__my_sum, raising=False)

    # 创建测试数据
    data = make_data()[:5]
    # 使用 make_data 创建数据，并取前5个元素
    df = pd.DataFrame({"id": [0, 0, 0, 1, 1], "decimals": DecimalArray(data)})
    # 创建包含 DecimalArray 类型的 decimals 列的 DataFrame
    expected = pd.Series(to_decimal([data[0] + data[1] + data[2], data[3] + data[4]]))
    # 创建预期结果的 Series，将 data 按指定规则转换为十进制数组并求和后封装成 Series

    # 使用 groupby 对 id 列进行分组，并对 decimals 列应用自定义的 my_sum 方法
    result = df.groupby("id")["decimals"].agg(lambda x: x.values.my_sum())
    # 检查 result 是否与 expected 相等，忽略名称检查
    tm.assert_series_equal(result, expected, check_names=False)

    # 创建 DecimalArray 类型的 Series
    s = pd.Series(DecimalArray(data))
    # 创建与 s 相同的 grouper 数组
    grouper = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    # 使用 groupby 对 grouper 进行分组，并对 Series 应用自定义的 my_sum 方法
    result = s.groupby(grouper).agg(lambda x: x.values.my_sum())
    # 检查 result 是否与 expected 相等，忽略名称检查
    tm.assert_series_equal(result, expected, check_names=False)


def test_indexing_no_materialize(monkeypatch):
    # 查看 https://github.com/pandas-dev/pandas/issues/29708
    # 确保索引操作不会不必要地实体化（将 ExtensionArray 转换为 numpy 数组）

    # 定义 DecimalArray 类的 __array__ 方法，抛出异常以防止将 DecimalArray 转换为 numpy 数组
    def DecimalArray__array__(self, dtype=None):
        raise Exception("tried to convert a DecimalArray to a numpy array")

    # 将自定义的 DecimalArray__array__ 方法设置为 DecimalArray 类的 __array__ 属性
    monkeypatch.setattr(DecimalArray, "__array__", DecimalArray__array__, raising=False)

    # 创建测试数据
    data = make_data()
    # 创建 DecimalArray 类型的 Series
    s = pd.Series(DecimalArray(data))
    # 创建 DataFrame，包含一个由 s 和 range(len(s)) 组成的字典
    df = pd.DataFrame({"a": s, "b": range(len(s))})

    # 确保以下操作不会引发错误
    s[s > 0.5]  # 对 Series s 执行条件索引
    df[s > 0.5]  # 对 DataFrame df 执行条件索引
    s.at[0]  # 使用 at 方法访问 Series s 的元素
    df.at[0, "a"]  # 使用 at 方法访问 DataFrame df 的元素


def test_to_numpy_keyword():
    # 测试额外的关键字参数

    # 创建 Decimal 对象的列表
    values = [decimal.Decimal("1.1111"), decimal.Decimal("2.2222")]
    # 创建期望的 numpy 数组，使用 object 类型存储 Decimal 对象
    expected = np.array(
        [decimal.Decimal("1.11"), decimal.Decimal("2.22")], dtype="object"
    )
    # 使用 pd.array 创建包含 Decimal 对象的数组 a
    a = pd.array(values, dtype="decimal")
    # 调用 a 的 to_numpy 方法，使用 decimals=2 的关键字参数
    result = a.to_numpy(decimals=2)
    # 检查 result 是否与 expected 相等
    tm.assert_numpy_array_equal(result, expected)

    # 创建 Series，使用 DecimalArray 类型的数组 a
    result = pd.Series(a).to_numpy(decimals=2)
    # 检查 result 是否与 expected 相等
    tm.assert_numpy_array_equal(result, expected)


def test_array_copy_on_write():
    # 创建 DataFrame，包含对象类型为 Decimal 的 a 列
    df = pd.DataFrame({"a": [decimal.Decimal(2), decimal.Decimal(3)]}, dtype="object")
    # 将 DataFrame df 转换为 DecimalDtype 类型的 DataFrame df2
    df2 = df.astype(DecimalDtype())
    # 修改 df 中的第一个元素为 0
    df.iloc[0, 0] = 0
    # 创建期望的 DataFrame，包含 DecimalDtype 类型的 a 列
    expected = pd.DataFrame(
        {"a": [decimal.Decimal(2), decimal.Decimal(3)]}, dtype=DecimalDtype()
    )
    # 检查 df2 和 expected 的值是否相等
    tm.assert_equal(df2.values, expected.values)
```