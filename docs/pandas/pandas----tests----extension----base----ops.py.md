# `D:\src\scipysrc\pandas\pandas\tests\extension\base\ops.py`

```
# 从未来导入注释以支持类型提示的自引用
from __future__ import annotations

# 导入必要的类型
from typing import final

# 导入第三方库 numpy 和 pytest
import numpy as np
import pytest

# 导入 pandas 内部使用的 pyarrow 字符串类型检查函数
from pandas._config import using_pyarrow_string_dtype

# 导入 pandas 中用于判断是否为字符串类型的函数
from pandas.core.dtypes.common import is_string_dtype

# 导入 pandas 并命名为 pd
import pandas as pd
# 导入 pandas 内部用于测试的模块
import pandas._testing as tm
# 导入 pandas 核心操作模块
from pandas.core import ops


class BaseOpsUtil:
    # 定义类变量，用于存储不同操作的异常类型或 None
    series_scalar_exc: type[Exception] | None = TypeError
    frame_scalar_exc: type[Exception] | None = TypeError
    series_array_exc: type[Exception] | None = TypeError
    divmod_exc: type[Exception] | None = TypeError

    # 获取期望的异常类型，用于指定操作名称、对象和另一个操作数
    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | None:
        # 查找预期引发的异常类型，当调用 obj.__op_name__(other) 时

        # 不建议使用 self.obj_bar_exc 这种模式，因为它可能依赖于 op_name 或 dtypes，但我们在这里用它来保持向后兼容性。
        if op_name in ["__divmod__", "__rdivmod__"]:
            result = self.divmod_exc
        elif isinstance(obj, pd.Series) and isinstance(other, pd.Series):
            result = self.series_array_exc
        elif isinstance(obj, pd.Series):
            result = self.series_scalar_exc
        else:
            result = self.frame_scalar_exc

        # 如果正在使用 pyarrow 的字符串类型，且 result 不为 None，则进一步处理
        if using_pyarrow_string_dtype() and result is not None:
            import pyarrow as pa

            result = (  # type: ignore[assignment]
                result,
                pa.lib.ArrowNotImplementedError,
                NotImplementedError,
            )
        return result

    # 将点对点操作的结果进行类型转换，以确保与向量化操作 obj.__op_name__(other) 的结果匹配
    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        # 在 _check_op 中，我们检查点对点操作的结果（通过 _combine 找到）是否与向量化操作 obj.__op_name__(other) 的结果匹配。
        # 在某些情况下，即使两个操作行为“正确”，pandas 对标量结果的 dtype 推断可能不会给出匹配的 dtype。在这些情况下，需要在此执行额外的必要转换。
        return pointwise_result

    # 根据操作名称获取操作函数
    def get_op_from_name(self, op_name: str):
        return tm.get_op_from_name(op_name)

    # 最终方法，子类不应该重写 check_opname、_check_op、_check_divmod_op 或 _combine。
    # 理想情况下，任何相关的重写可以在 _cast_pointwise_result、get_op_from_name 和 exc 的规范中完成。
    # 如果您发现还需要重写 _check_op 或 _combine，请告诉我们：github.com/pandas-dev/pandas/issues
    @final
    def check_opname(self, ser: pd.Series, op_name: str, other):
        # 获取预期的异常类型
        exc = self._get_expected_exception(op_name, ser, other)
        # 获取操作函数
        op = self.get_op_from_name(op_name)
        # 检查操作的有效性
        self._check_op(ser, op, other, op_name, exc)

    # 与 check_opname 上的注释类似
    @final
    def _check_op(self, ser: pd.Series, op, other, op_name: str, exc):
        pass
    # 定义一个方法 _combine，用于根据对象类型进行合并操作
    def _combine(self, obj, other, op):
        # 如果 obj 是 DataFrame 类型
        if isinstance(obj, pd.DataFrame):
            # 如果 DataFrame 的列数不为1，则抛出未实现错误
            if len(obj.columns) != 1:
                raise NotImplementedError
            # 否则，预期的结果是将 obj 的第一列与 other 执行 op 操作，并转换为 DataFrame
            expected = obj.iloc[:, 0].combine(other, op).to_frame()
        else:
            # 对于非 DataFrame 对象，直接执行 combine 操作
            expected = obj.combine(other, op)
        return expected

    # 修饰器 @final，见 check_opname 的注释
    @final
    def _check_op(
        self, ser: pd.Series, op, other, op_name: str, exc=NotImplementedError
    ):
        # 检查 Series/DataFrame 的算术/比较方法是否与 _combine 的逐点结果匹配

        # 如果 exc 为 None
        if exc is None:
            # 执行 op 操作，计算结果
            result = op(ser, other)
            # 使用 _combine 方法计算预期的结果
            expected = self._combine(ser, other, op)
            # 将预期结果转换为逐点结果，见 _cast_pointwise_result 方法的注释
            expected = self._cast_pointwise_result(op_name, ser, other, expected)
            # 断言结果的类型与 ser 的类型相同
            assert isinstance(result, type(ser))
            # 使用测试工具函数 tm.assert_equal 检查 result 是否等于 expected
            tm.assert_equal(result, expected)
        else:
            # 如果 exc 不为 None，预期会引发特定的异常
            with pytest.raises(exc):
                op(ser, other)

    # 修饰器 @final，见 check_opname 的注释
    @final
    def _check_divmod_op(self, ser: pd.Series, op, other):
        # 检查 divmod 的行为是否与 floordiv+mod 的行为匹配
        if op is divmod:
            # 获取预期的异常，见 _get_expected_exception 方法的注释
            exc = self._get_expected_exception("__divmod__", ser, other)
        else:
            exc = self._get_expected_exception("__rdivmod__", ser, other)
        if exc is None:
            # 执行 divmod 操作，分别得到商和余数的结果
            result_div, result_mod = op(ser, other)
            # 如果 op 是 divmod，则预期的结果是 ser // other 和 ser % other
            if op is divmod:
                expected_div, expected_mod = ser // other, ser % other
            else:
                # 否则，预期的结果是 other // ser 和 other % ser
                expected_div, expected_mod = other // ser, other % ser
            # 使用测试工具函数 tm.assert_series_equal 检查结果_div 和 expected_div 是否相等
            tm.assert_series_equal(result_div, expected_div)
            # 使用测试工具函数 tm.assert_series_equal 检查结果_mod 和 expected_mod 是否相等
            tm.assert_series_equal(result_mod, expected_mod)
        else:
            # 如果 exc 不为 None，预期会引发特定的异常
            with pytest.raises(exc):
                divmod(ser, other)
class BaseArithmeticOpsTests(BaseOpsUtil):
    """
    Various Series and DataFrame arithmetic ops methods.

    Subclasses supporting various ops should set the class variables
    to indicate that they support ops of that kind

    * series_scalar_exc = TypeError
    * frame_scalar_exc = TypeError
    * series_array_exc = TypeError
    * divmod_exc = TypeError
    """

    series_scalar_exc: type[Exception] | None = TypeError  # 异常类型，用于指示操作错误
    frame_scalar_exc: type[Exception] | None = TypeError   # 异常类型，用于指示操作错误
    series_array_exc: type[Exception] | None = TypeError   # 异常类型，用于指示操作错误
    divmod_exc: type[Exception] | None = TypeError          # 异常类型，用于指示操作错误

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        # series & scalar
        if all_arithmetic_operators == "__rmod__" and is_string_dtype(data.dtype):
            pytest.skip("Skip testing Python string formatting")
        
        op_name = all_arithmetic_operators   # 操作名，根据参数确定要测试的特定算术操作
        ser = pd.Series(data)   # 创建一个 pandas Series 对象
        self.check_opname(ser, op_name, ser.iloc[0])   # 调用方法检查操作名在 Series 上的效果

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        # frame & scalar
        if all_arithmetic_operithmetic_operators == "__rmod__" and is_string_dtype(data.dtype):
            pytest.skip("Skip testing Python string formatting")
        
        op_name = all_arithmetic_operators   # 操作名，根据参数确定要测试的特定算术操作
        df = pd.DataFrame({"A": data})   # 创建一个包含数据的 DataFrame 对象
        self.check_opname(df, op_name, data[0])   # 调用方法检查操作名在 DataFrame 上的效果

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # ndarray & other series
        op_name = all_arithmetic_operators   # 操作名，根据参数确定要测试的特定算术操作
        ser = pd.Series(data)   # 创建一个 pandas Series 对象
        self.check_opname(ser, op_name, pd.Series([ser.iloc[0]] * len(ser)))   # 调用方法检查操作名在 Series 上的效果

    def test_divmod(self, data):
        ser = pd.Series(data)   # 创建一个 pandas Series 对象
        self._check_divmod_op(ser, divmod, 1)   # 调用方法检查 divmod 操作在 Series 上的效果
        self._check_divmod_op(1, ops.rdivmod, ser)   # 调用方法检查 rdivmod 操作在 Series 上的效果

    def test_divmod_series_array(self, data, data_for_twos):
        ser = pd.Series(data)   # 创建一个 pandas Series 对象
        self._check_divmod_op(ser, divmod, data)   # 调用方法检查 divmod 操作在 Series 上的效果

        other = data_for_twos   # 创建另一个数据对象
        self._check_divmod_op(other, ops.rdivmod, ser)   # 调用方法检查 rdivmod 操作在 Series 上的效果

        other = pd.Series(other)   # 将数据对象转换为 pandas Series 对象
        self._check_divmod_op(other, ops.rdivmod, ser)   # 调用方法检查 rdivmod 操作在 Series 上的效果

    def test_add_series_with_extension_array(self, data):
        # Check adding an ExtensionArray to a Series of the same dtype matches
        # the behavior of adding the arrays directly and then wrapping in a
        # Series.

        ser = pd.Series(data)   # 创建一个 pandas Series 对象

        exc = self._get_expected_exception("__add__", ser, data)   # 获取预期的异常类型
        if exc is not None:
            with pytest.raises(exc):
                ser + data   # 执行加法操作并验证是否抛出异常
            return

        result = ser + data   # 执行加法操作
        expected = pd.Series(data + data)   # 创建预期结果的 Series 对象
        tm.assert_series_equal(result, expected)   # 断言操作结果与预期结果相等

    @pytest.mark.parametrize("box", [pd.Series, pd.DataFrame, pd.Index])
    @pytest.mark.parametrize(
        "op_name",
        [
            x
            for x in tm.arithmetic_dunder_methods + tm.comparison_dunder_methods
            if not x.startswith("__r")
        ],
    )
    def test_direct_arith_with_ndframe_returns_not_implemented(
        self, data, box, op_name
    ):
        # EAs should return NotImplemented for ops with Series/DataFrame/Index
        # EAs (Extension Arrays) should return NotImplemented when operating with Series, DataFrame, or Index objects.
        # Pandas takes care of unboxing the series and calling the EA's op.
        # Pandas handles converting the Series into native types and invokes the operation defined in the Extension Array.
        other = box(data)
        # Convert data using the box function, which is expected to handle specific data types.

        if hasattr(data, op_name):
            # Check if the data object has the specified operation defined.
            result = getattr(data, op_name)(other)
            # Invoke the specified operation on the data object with the converted 'other' data.
            assert result is NotImplemented
            # Assert that the result of the operation is NotImplemented.
    # BaseComparisonOpsTests 类，继承自 BaseOpsUtil 类，用于测试 Series 和 DataFrame 的比较操作方法
    class BaseComparisonOpsTests(BaseOpsUtil):

        # _compare_other 方法，用于比较 Series 和其他数据（标量或数组）的操作
        def _compare_other(self, ser: pd.Series, data, op, other):
            # 如果操作是 "eq" 或 "ne"，则进行逐点比较
            if op.__name__ in ["eq", "ne"]:
                # 执行操作并比较结果
                result = op(ser, other)
                # 使用 combine 方法生成期望结果
                expected = ser.combine(other, op)
                # 转换逐点比较的结果类型
                expected = self._cast_pointwise_result(op.__name__, ser, other, expected)
                # 使用 assert_series_equal 断言结果与期望一致
                tm.assert_series_equal(result, expected)
    
            else:
                exc = None
                try:
                    # 执行操作并捕获可能的异常
                    result = op(ser, other)
                except Exception as err:
                    exc = err
    
                if exc is None:
                    # 没有发生异常，则进行逐点比较
                    expected = ser.combine(other, op)
                    # 转换逐点比较的结果类型
                    expected = self._cast_pointwise_result(
                        op.__name__, ser, other, expected
                    )
                    # 使用 assert_series_equal 断言结果与期望一致
                    tm.assert_series_equal(result, expected)
                else:
                    # 如果发生异常，则使用 pytest.raises 断言异常类型
                    with pytest.raises(type(exc)):
                        ser.combine(other, op)

        # test_compare_scalar 方法，用于测试与标量的比较操作
        def test_compare_scalar(self, data, comparison_op):
            # 创建 Series 对象
            ser = pd.Series(data)
            # 调用 _compare_other 方法进行比较
            self._compare_other(ser, data, comparison_op, 0)

        # test_compare_array 方法，用于测试与数组的比较操作
        def test_compare_array(self, data, comparison_op):
            # 创建 Series 对象
            ser = pd.Series(data)
            # 创建另一个与第一个元素相同的 Series 对象作为比较对象
            other = pd.Series([data[0]] * len(data), dtype=data.dtype)
            # 调用 _compare_other 方法进行比较
            self._compare_other(ser, data, comparison_op, other)


    # BaseUnaryOpsTests 类，继承自 BaseOpsUtil 类，用于测试一元操作方法
    class BaseUnaryOpsTests(BaseOpsUtil):

        # test_invert 方法，测试按位取反操作
        def test_invert(self, data):
            # 创建 Series 对象
            ser = pd.Series(data, name="name")
            try:
                # 对数据的前 10 个元素进行按位取反操作
                [~x for x in data[:10]]
            except TypeError:
                # 如果数据类型不支持按位取反，则预期会抛出 TypeError 异常
                with pytest.raises(TypeError):
                    ~ser
                with pytest.raises(TypeError):
                    ~data
            else:
                # 如果数据类型支持按位取反，则比较操作后的结果与预期结果
                result = ~ser
                expected = pd.Series(~data, name="name")
                # 使用 assert_series_equal 断言结果与期望一致
                tm.assert_series_equal(result, expected)

        # 使用 pytest.mark.parametrize 注解，参数化测试以下三个一元操作函数：np.positive, np.negative, np.abs
        @pytest.mark.parametrize("ufunc", [np.positive, np.negative, np.abs])
    # 测试一元通用函数与魔术方法的等价性
    def test_unary_ufunc_dunder_equivalence(self, data, ufunc):
        # 如果 __pos__ 能正常工作，那么 np.positive 也应当如此，同理适用于 __neg__/np.negative 和 __abs__/np.abs
        attr = {np.positive: "__pos__", np.negative: "__neg__", np.abs: "__abs__"}[
            ufunc
        ]

        exc = None
        try:
            # 调用对象的指定魔术方法
            result = getattr(data, attr)()
        except Exception as err:
            exc = err

            # 如果调用 __pos__ 引发异常，那么通用函数 ufunc 也应该引发相同类型的异常或 TypeError
            with pytest.raises((type(exc), TypeError)):
                ufunc(data)
        else:
            # 否则，对于 ufunc(data) 和 getattr(data, attr)() 应当返回相同结果
            alt = ufunc(data)
            tm.assert_extension_array_equal(result, alt)
```