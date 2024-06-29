# `D:\src\scipysrc\pandas\pandas\tests\extension\base\reduce.py`

```
from typing import final  # 导入 final 类型提示

import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 pandas 库并简称为 pd
import pandas._testing as tm  # 导入 pandas 内部测试模块并简称为 tm
from pandas.api.types import is_numeric_dtype  # 从 pandas.api.types 模块导入 is_numeric_dtype 函数

class BaseReduceTests:
    """
    Reduction specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # 指定是否预期此减少操作成功
        return False

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # 我们对 np.float64 类型数据执行相同操作，并检查结果是否匹配。
        # 如果需要转换为除 float64 之外的其他类型，请覆盖此方法。
        res_op = getattr(ser, op_name)

        try:
            alt = ser.astype("float64")
        except (TypeError, ValueError):
            # 例如，Interval 类型无法转换 (TypeError)，StringArray 类型无法转换 (ValueError)，
            # 因此我们转换为 object 类型并逐点执行减少操作。
            alt = ser.astype(object)

        exp_op = getattr(alt, op_name)
        if op_name == "count":
            result = res_op()
            expected = exp_op()
        else:
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        # 在对 DataFrame 列进行给定减少操作时，找到此数组的预期 dtype。
        # 默认情况下假设类似 float64 的行为，即保留 dtype。
        return arr.dtype

    # 我们预期作者不需要覆盖 check_reduce_frame 方法，
    # 但应能够在 _get_expected_reduction_dtype 方法中进行必要的覆盖。
    # 如果您有例外情况，请在 github.com/pandas-dev/pandas/issues 上通知我们。
    @final
    def check_reduce_frame(self, ser: pd.Series, op_name: str, skipna: bool):
        # 检查在数据帧减少中是否与序列减少的结果类似
        arr = ser.array
        # 创建包含序列数据的数据帧
        df = pd.DataFrame({"a": arr})

        # 如果操作名是 "var" 或 "std"，设置 ddof=1；否则为空字典
        kwargs = {"ddof": 1} if op_name in ["var", "std"] else {}

        # 获取预期的减少操作结果的数据类型
        cmp_dtype = self._get_expected_reduction_dtype(arr, op_name, skipna)

        # 数据帧方法只是以 keepdims=True 调用 arr._reduce，因此这个检查是例行公事
        result1 = arr._reduce(op_name, skipna=skipna, keepdims=True, **kwargs)
        # 调用数据帧的操作名方法，并获取其序列数组形式的结果
        result2 = getattr(df, op_name)(skipna=skipna, **kwargs).array
        # 断言两个扩展数组的相等性
        tm.assert_extension_array_equal(result1, result2)

        # 检查二维减少是否类似于包装过的一维减少
        if not skipna and ser.isna().any():
            # 如果不跳过缺失值且序列中有缺失值，则预期结果为一个 NA 值的序列
            expected = pd.array([pd.NA], dtype=cmp_dtype)
        else:
            # 否则，计算在删除缺失值后的序列上应用操作名操作后的预期值
            exp_value = getattr(ser.dropna(), op_name)()
            expected = pd.array([exp_value], dtype=cmp_dtype)

        # 断言两个扩展数组的相等性
        tm.assert_extension_array_equal(result1, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        op_name = all_boolean_reductions
        ser = pd.Series(data)

        if not self._supports_reduction(ser, op_name):
            # TODO: 这里检查的消息实际上没有起到任何作用
            msg = (
                "[Cc]annot perform|Categorical is not ordered for operation|"
                "does not support operation|"
            )

            # 使用 pytest 来检查操作不支持的异常情况
            with pytest.raises(TypeError, match=msg):
                getattr(ser, op_name)(skipna=skipna)

        else:
            # 调用自定义的检查函数来验证序列的减少操作
            self.check_reduce(ser, op_name, skipna)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        op_name = all_numeric_reductions
        ser = pd.Series(data)

        if not self._supports_reduction(ser, op_name):
            # TODO: 这里检查的消息实际上没有起到任何作用
            msg = (
                "[Cc]annot perform|Categorical is not ordered for operation|"
                "does not support operation|"
            )

            # 使用 pytest 来检查操作不支持的异常情况
            with pytest.raises(TypeError, match=msg):
                getattr(ser, op_name)(skipna=skipna)

        else:
            # 调用自定义的检查函数来验证序列的减少操作
            # 对于空数据进行最小值/最大值操作会产生 NumPy 警告
            self.check_reduce(ser, op_name, skipna)

    @pytest.mark.parametrize("skipna", [True, False])
    # 定义一个测试方法，用于测试数据帧的降维操作
    def test_reduce_frame(self, data, all_numeric_reductions, skipna):
        # 将所有数值类型的降维方法的名称赋给变量 op_name
        op_name = all_numeric_reductions
        # 根据输入的数据创建一个 pandas Series 对象
        ser = pd.Series(data)
        # 如果 Series 对象的数据类型不是数值类型，则跳过测试，并输出相应信息
        if not is_numeric_dtype(ser.dtype):
            pytest.skip(f"{ser.dtype} is not numeric dtype")

        # 如果降维方法的名称在 ["count", "kurt", "sem"] 中，则跳过测试，并输出相应信息
        if op_name in ["count", "kurt", "sem"]:
            pytest.skip(f"{op_name} not an array method")

        # 如果当前数据类型不支持指定的降维方法，则跳过测试，并输出相应信息
        if not self._supports_reduction(ser, op_name):
            pytest.skip(f"Reduction {op_name} not supported for this dtype")

        # 调用类中的方法，检查并执行数据帧的降维操作
        self.check_reduce_frame(ser, op_name, skipna)
```