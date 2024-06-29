# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_shift.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 中导入特定的子模块和类
    CategoricalIndex,
    DataFrame,
    Index,
    NaT,
    Series,
    date_range,
    offsets,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestDataFrameShift:
    def test_shift_axis1_with_valid_fill_value_one_array(self):
        # Case with axis=1 that does not go through the "len(arrays)>1" path
        # in DataFrame.shift
        data = np.random.default_rng(2).standard_normal((5, 3))  # 生成一个 5x3 的随机标准正态分布数组
        df = DataFrame(data)  # 创建 DataFrame 对象
        res = df.shift(axis=1, periods=1, fill_value=12345)  # 对 DataFrame 进行水平方向的位移操作，填充值为 12345
        expected = df.T.shift(periods=1, fill_value=12345).T  # 期望结果也进行了相同的操作
        tm.assert_frame_equal(res, expected)  # 使用测试工具比较结果是否相等

        # same but with an 1D ExtensionArray backing it
        df2 = df[[0]].astype("Float64")  # 选择 DataFrame 中的第一列并转换为 Float64 类型
        res2 = df2.shift(axis=1, periods=1, fill_value=12345)  # 对新的 DataFrame 进行水平方向的位移操作，填充值为 12345
        expected2 = DataFrame([12345] * 5, dtype="Float64")  # 创建期望结果的 DataFrame
        tm.assert_frame_equal(res2, expected2)  # 使用测试工具比较结果是否相等

    def test_shift_disallow_freq_and_fill_value(self, frame_or_series):
        # Can't pass both!
        obj = frame_or_series(
            np.random.default_rng(2).standard_normal(5),  # 创建一个随机数据对象
            index=date_range("1/1/2000", periods=5, freq="h"),  # 创建一个日期时间索引对象
        )

        msg = "Passing a 'freq' together with a 'fill_value'"
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 来测试是否抛出 ValueError 异常
            obj.shift(1, fill_value=1, freq="h")

        if frame_or_series is DataFrame:
            obj.columns = date_range("1/1/2000", periods=1, freq="h")  # 设置 DataFrame 的列为日期时间索引
            with pytest.raises(ValueError, match=msg):  # 再次测试是否抛出 ValueError 异常
                obj.shift(1, axis=1, fill_value=1, freq="h")

    @pytest.mark.parametrize(
        "input_data, output_data",
        [(np.empty(shape=(0,)), []), (np.ones(shape=(2,)), [np.nan, 1.0])],
    )
    def test_shift_non_writable_array(self, input_data, output_data, frame_or_series):
        # GH21049 Verify whether non writable numpy array is shiftable
        input_data.setflags(write=False)  # 设置输入数据为不可写

        result = frame_or_series(input_data).shift(1)  # 对输入数据进行位移操作
        if frame_or_series is not Series:
            # need to explicitly specify columns in the empty case
            expected = frame_or_series(
                output_data,
                index=range(len(output_data)),
                columns=range(1),
                dtype="float64",
            )
        else:
            expected = frame_or_series(output_data, dtype="float64")

        tm.assert_equal(result, expected)  # 使用测试工具比较结果是否相等

    def test_shift_mismatched_freq(self, frame_or_series):
        ts = frame_or_series(
            np.random.default_rng(2).standard_normal(5),  # 创建一个随机时间序列对象
            index=date_range("1/1/2000", periods=5, freq="h"),  # 创建一个小时频率的日期时间索引
        )

        result = ts.shift(1, freq="5min")  # 对时间序列进行位移操作，更改频率为 5 分钟
        exp_index = ts.index.shift(1, freq="5min")  # 期望结果中索引的频率也变为 5 分钟
        tm.assert_index_equal(result.index, exp_index)  # 使用测试工具比较结果的索引是否相等

        # GH#1063, multiple of same base
        result = ts.shift(1, freq="4h")  # 对时间序列进行位移操作，更改频率为 4 小时
        exp_index = ts.index + offsets.Hour(4)  # 期望结果中索引增加 4 小时的偏移量
        tm.assert_index_equal(result.index, exp_index)  # 使用测试工具比较结果的索引是否相等
    @pytest.mark.parametrize(
        "obj",
        [
            Series([np.arange(5)]),  # 参数化测试对象为包含 numpy 数组的 Series
            date_range("1/1/2011", periods=24, freq="h"),  # 参数化测试对象为日期范围，频率为每小时
            Series(range(5), index=date_range("2017", periods=5)),  # 参数化测试对象为具有日期索引的 Series
        ],
    )
    @pytest.mark.parametrize("shift_size", [0, 1, 2])
    def test_shift_always_copy(self, obj, shift_size, frame_or_series):
        # GH#22397
        # 如果 frame_or_series 不是 Series 类型，则将 obj 转换为 DataFrame
        if frame_or_series is not Series:
            obj = obj.to_frame()
        # 断言 obj.shift(shift_size) 不等于 obj
        assert obj.shift(shift_size) is not obj

    def test_shift_object_non_scalar_fill(self):
        # shift 要求除了对象类型之外，fill_value 必须是标量
        ser = Series(range(3))
        # 使用 pytest 的断言检查是否会引发 ValueError，并匹配错误信息
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            ser.shift(1, fill_value=[])

        df = ser.to_frame()
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            df.shift(1, fill_value=np.arange(3))

        obj_ser = ser.astype(object)
        # 对象类型的 Series 使用空字典作为 fill_value 进行 shift
        result = obj_ser.shift(1, fill_value={})
        assert result[0] == {}

        obj_df = obj_ser.to_frame()
        # 对象类型的 DataFrame 使用空字典作为 fill_value 进行 shift
        result = obj_df.shift(1, fill_value={})
        assert result.iloc[0, 0] == {}

    def test_shift_int(self, datetime_frame, frame_or_series):
        # 将 datetime_frame 转换为整数类型，并进行 shift 操作
        ts = tm.get_obj(datetime_frame, frame_or_series).astype(int)
        shifted = ts.shift(1)
        expected = ts.astype(float).shift(1)
        tm.assert_equal(shifted, expected)

    @pytest.mark.parametrize("dtype", ["int32", "int64"])
    def test_shift_32bit_take(self, frame_or_series, dtype):
        # 32 位整数的取值范围测试
        # GH#8129
        index = date_range("2000-01-01", periods=5)
        arr = np.arange(5, dtype=dtype)
        s1 = frame_or_series(arr, index=index)
        p = arr[1]
        result = s1.shift(periods=p)
        expected = frame_or_series([np.nan, 0, 1, 2, 3], index=index)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("periods", [1, 2, 3, 4])
    def test_shift_preserve_freqstr(self, periods, frame_or_series):
        # 保持频率字符串不变的 shift 测试
        # GH#21275
        obj = frame_or_series(
            range(periods),
            index=date_range("2016-1-1 00:00:00", periods=periods, freq="h"),
        )

        result = obj.shift(1, "2h")

        expected = frame_or_series(
            range(periods),
            index=date_range("2016-1-1 02:00:00", periods=periods, freq="h"),
        )
        tm.assert_equal(result, expected)
    # 定义一个测试方法，用于测试时区转换下的时间偏移
    def test_shift_dst(self, frame_or_series):
        # GH#13926
        # 创建一个包含时区信息的日期范围，从"2016-11-06"开始，每小时一个时间点，共10个时间点
        dates = date_range("2016-11-06", freq="h", periods=10, tz="US/Eastern")
        # 使用传入的 frame_or_series 函数创建一个时间序列或数据帧对象
        obj = frame_or_series(dates)

        # 测试时间序列或数据帧对象不偏移
        res = obj.shift(0)
        tm.assert_equal(res, obj)
        # 断言结果的数据类型为"datetime64[ns, US/Eastern]"
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

        # 测试时间序列或数据帧对象向前偏移1个单位
        res = obj.shift(1)
        # 创建期望的偏移后的值列表，其中第一个值是 NaT（Not a Time），其余为 dates 列表的前9个元素
        exp_vals = [NaT] + dates.astype(object).values.tolist()[:9]
        exp = frame_or_series(exp_vals)
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

        # 测试时间序列或数据帧对象向后偏移2个单位
        res = obj.shift(-2)
        # 创建期望的偏移后的值列表，从 dates 列表的第3个元素开始，最后两个值为 NaT
        exp_vals = dates.astype(object).values.tolist()[2:] + [NaT, NaT]
        exp = frame_or_series(exp_vals)
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

    @pytest.mark.parametrize("ex", [10, -10, 20, -20])
    # 定义一个参数化测试方法，用于测试超出范围的时间偏移
    def test_shift_dst_beyond(self, frame_or_series, ex):
        # GH#13926
        # 创建一个包含时区信息的日期范围，从"2016-11-06"开始，每小时一个时间点，共10个时间点
        dates = date_range("2016-11-06", freq="h", periods=10, tz="US/Eastern")
        # 使用传入的 frame_or_series 函数创建一个时间序列或数据帧对象
        obj = frame_or_series(dates)
        # 对时间序列或数据帧对象进行 ex 指定的偏移
        res = obj.shift(ex)
        # 创建期望的结果对象，包含10个 NaT 值，数据类型为"datetime64[ns, US/Eastern]"
        exp = frame_or_series([NaT] * 10, dtype="datetime64[ns, US/Eastern]")
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

    # 测试时间偏移为0的情况
    def test_shift_by_zero(self, datetime_frame, frame_or_series):
        # shift by 0
        # 从 datetime_frame 和 frame_or_series 函数获取相应的对象
        obj = tm.get_obj(datetime_frame, frame_or_series)
        # 对对象进行0偏移
        unshifted = obj.shift(0)
        tm.assert_equal(unshifted, obj)

    # 测试时间偏移
    def test_shift(self, datetime_frame):
        # naive shift
        # 获取数据帧 datetime_frame 中的时间序列 "A"
        ser = datetime_frame["A"]

        # 对整个数据帧进行向前偏移5个单位
        shifted = datetime_frame.shift(5)
        tm.assert_index_equal(shifted.index, datetime_frame.index)

        # 对时间序列 "A" 进行向前偏移5个单位
        shifted_ser = ser.shift(5)
        tm.assert_series_equal(shifted["A"], shifted_ser)

        # 对整个数据帧进行向后偏移5个单位
        shifted = datetime_frame.shift(-5)
        tm.assert_index_equal(shifted.index, datetime_frame.index)

        # 对时间序列 "A" 进行向后偏移5个单位
        shifted_ser = ser.shift(-5)
        tm.assert_series_equal(shifted["A"], shifted_ser)

        # 测试偏移5个单位后再向后偏移5个单位，结果应与原始数据相等
        unshifted = datetime_frame.shift(5).shift(-5)
        tm.assert_numpy_array_equal(
            unshifted.dropna().values, datetime_frame.values[:-5]
        )

        # 测试时间序列 "A" 偏移5个单位后再向后偏移5个单位，结果应与原始时间序列 "A" 的值相等
        unshifted_ser = ser.shift(5).shift(-5)
        tm.assert_numpy_array_equal(unshifted_ser.dropna().values, ser.values[:-5])
    # 定义一个测试方法，用于测试在日期框架和数据帧/系列中进行偏移操作
    def test_shift_by_offset(self, datetime_frame, frame_or_series):
        # shift by DateOffset
        # 从测试管理器获取对象，该对象可能是日期框架或数据帧/系列
        obj = tm.get_obj(datetime_frame, frame_or_series)
        # 创建一个工作日偏移对象
        offset = offsets.BDay()

        # 对象向后偏移5个工作日
        shifted = obj.shift(5, freq=offset)
        # 断言偏移后的长度与原对象长度相等
        assert len(shifted) == len(obj)
        # 将偏移后的对象再向前偏移5个工作日
        unshifted = shifted.shift(-5, freq=offset)
        # 断言还原后的对象与原对象相等
        tm.assert_equal(unshifted, obj)

        # 使用字符串'B'再次向后偏移5个工作日
        shifted2 = obj.shift(5, freq="B")
        # 断言两次偏移结果相等
        tm.assert_equal(shifted, shifted2)

        # 将对象不偏移（偏移量为0）并使用工作日偏移再次向后偏移
        unshifted = obj.shift(0, freq=offset)
        # 断言还原后的对象与原对象相等
        tm.assert_equal(unshifted, obj)

        # 获取对象索引的第一个日期
        d = obj.index[0]
        # 计算该日期向后偏移5个工作日后的日期
        shifted_d = d + offset * 5
        # 根据数据帧或系列类型进行不同的断言
        if frame_or_series is DataFrame:
            # 断言序列的值相等，忽略名称检查
            tm.assert_series_equal(obj.xs(d), shifted.xs(shifted_d), check_names=False)
        else:
            # 断言浮点数近似相等
            tm.assert_almost_equal(obj.at[d], shifted.at[shifted_d])

    # 定义一个测试方法，用于测试在周期索引中进行偏移操作
    def test_shift_with_periodindex(self, frame_or_series):
        # Shifting with PeriodIndex
        # 创建一个包含四个浮点数的数据帧，使用周期索引从"2020-01-01"开始，周期为4
        ps = DataFrame(
            np.arange(4, dtype=float), index=pd.period_range("2020-01-01", periods=4)
        )
        # 从测试管理器获取对象，该对象可能是数据帧或系列
        ps = tm.get_obj(ps, frame_or_series)

        # 对象中的数据向后偏移1个周期
        shifted = ps.shift(1)
        # 将偏移后的对象再向前偏移1个周期
        unshifted = shifted.shift(-1)
        # 断言偏移后的索引与原对象的索引相等
        tm.assert_index_equal(shifted.index, ps.index)
        tm.assert_index_equal(unshifted.index, ps.index)

        # 根据数据帧或系列类型进行不同的断言
        if frame_or_series is DataFrame:
            # 断言第一列的数值相等，忽略NaN值
            tm.assert_numpy_array_equal(
                unshifted.iloc[:, 0].dropna().values, ps.iloc[:-1, 0].values
            )
        else:
            # 断言数值相等，忽略NaN值
            tm.assert_numpy_array_equal(unshifted.dropna().values, ps.values[:-1])

        # 使用字符串'D'再次向后偏移1天
        shifted2 = ps.shift(1, "D")
        # 使用工作日偏移再次向后偏移1天
        shifted3 = ps.shift(1, offsets.Day())
        # 断言两次偏移结果相等
        tm.assert_equal(shifted2, shifted3)
        # 断言数据帧与向后偏移1天后的结果相等
        tm.assert_equal(ps, shifted2.shift(-1, "D"))

        # 使用频率"W"尝试进行偏移，预期会引发值错误异常
        msg = "does not match PeriodIndex freq"
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq="W")

        # 为了保留兼容性，使用字符串'D'再次向后偏移1天
        shifted4 = ps.shift(1, freq="D")
        # 断言两次偏移结果相等
        tm.assert_equal(shifted2, shifted4)

        # 使用工作日偏移再次向后偏移1天
        shifted5 = ps.shift(1, freq=offsets.Day())
        # 断言两次偏移结果相等
        tm.assert_equal(shifted5, shifted4)

    # 定义一个测试方法，用于测试在数据帧中对其它轴进行偏移操作
    def test_shift_other_axis(self):
        # shift other axis
        # GH#6371
        # 创建一个随机生成的10行5列数据帧
        df = DataFrame(np.random.default_rng(2).random((10, 5)))
        # 创建期望的结果，将每行的第一列移动到每行的最后
        expected = pd.concat(
            [DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:-1]],
            ignore_index=True,
            axis=1,
        )
        # 对数据帧进行沿着列轴的偏移操作
        result = df.shift(1, axis=1)
        # 断言偏移后的结果与期望的结果相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试在数据帧中对命名轴进行偏移操作
    def test_shift_named_axis(self):
        # shift named axis
        # GH#6371
        # 创建一个随机生成的10行5列数据帧
        df = DataFrame(np.random.default_rng(2).random((10, 5)))
        # 创建期望的结果，将每行的第一列移动到每行的最后
        expected = pd.concat(
            [DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:-1]],
            ignore_index=True,
            axis=1,
        )
        # 对数据帧进行沿着命名为'columns'的轴的偏移操作
        result = df.shift(1, axis="columns")
        # 断言偏移后的结果与期望的结果相等
        tm.assert_frame_equal(result, expected)
    # 定义测试方法，用于测试在指定频率下对 DataFrame 或 Series 进行轴向偏移
    def test_shift_other_axis_with_freq(self, datetime_frame):
        # 将 datetime_frame 转置为 obj
        obj = datetime_frame.T
        # 定义偏移量为一个工作日
        offset = offsets.BDay()

        # GH#47039
        # 对 obj 进行轴向（axis=1）偏移5个单位，使用 offset 作为频率
        shifted = obj.shift(5, freq=offset, axis=1)
        # 断言偏移后的长度应与原始对象相同
        assert len(shifted) == len(obj)
        # 对偏移后的对象再次进行反向偏移，应当与原始对象相等
        unshifted = shifted.shift(-5, freq=offset, axis=1)
        tm.assert_equal(unshifted, obj)

    # 定义测试方法，用于测试对包含布尔值的 DataFrame 进行偏移
    def test_shift_bool(self):
        # 创建包含布尔值的 DataFrame
        df = DataFrame({"high": [True, False], "low": [False, False]})
        # 对 df 进行向下偏移1行
        rs = df.shift(1)
        # 创建期望结果 xp，将第一行移动到最后一行
        xp = DataFrame(
            np.array([[np.nan, np.nan], [True, False]], dtype=object),
            columns=["high", "low"],
        )
        # 断言偏移后的结果应与期望结果 xp 相等
        tm.assert_frame_equal(rs, xp)

    # 定义测试方法，用于测试对包含分类数据的 Series 或 DataFrame 进行偏移
    def test_shift_categorical1(self, frame_or_series):
        # GH#9416
        # 创建包含分类数据的对象 obj
        obj = frame_or_series(["a", "b", "c", "d"], dtype="category")

        # 对 obj 进行向下偏移1行，再向上偏移1行
        rt = obj.shift(1).shift(-1)
        # 断言结果应与 obj 的所有行除最后一行外相等
        tm.assert_equal(obj.iloc[:-1], rt.dropna())

        # 定义获取分类值的函数
        def get_cat_values(ndframe):
            # 对于 Series 可以直接访问 ._values；对于 DataFrame 则需要更复杂的处理
            return ndframe._mgr.blocks[0].values

        # 获取 obj 的分类值
        cat = get_cat_values(obj)

        # 对 obj 进行向下偏移1行
        sp1 = obj.shift(1)
        # 断言索引应与 obj 相同
        tm.assert_index_equal(obj.index, sp1.index)
        # 断言偏移后的第一行分类代码应为 -1
        assert np.all(get_cat_values(sp1).codes[:1] == -1)
        # 断言偏移后的分类代码应与原始分类代码相符
        assert np.all(cat.codes[:-1] == get_cat_values(sp1).codes[1:])

        # 对 obj 进行向上偏移2行
        sn2 = obj.shift(-2)
        # 断言索引应与 obj 相同
        tm.assert_index_equal(obj.index, sn2.index)
        # 断言偏移后的倒数第二行分类代码应为 -1
        assert np.all(get_cat_values(sn2).codes[-2:] == -1)
        # 断言偏移后的分类代码应与原始分类代码相符
        assert np.all(cat.codes[2:] == get_cat_values(sn2).codes[:-2])

        # 断言分类的类别应与偏移后的类别相同
        tm.assert_index_equal(cat.categories, get_cat_values(sp1).categories)
        tm.assert_index_equal(cat.categories, get_cat_values(sn2).categories)

    # 定义测试方法，用于测试对包含分类数据的 DataFrame 进行偏移
    def test_shift_categorical(self):
        # GH#9416
        # 创建包含分类数据的 Series s1 和 s2
        s1 = Series(["a", "b", "c"], dtype="category")
        s2 = Series(["A", "B", "C"], dtype="category")
        # 创建包含 s1 和 s2 的 DataFrame df
        df = DataFrame({"one": s1, "two": s2})
        # 对 df 进行向下偏移1行
        rs = df.shift(1)
        # 创建期望结果 xp，分别对 s1 和 s2 进行向下偏移1行
        xp = DataFrame({"one": s1.shift(1), "two": s2.shift(1)})
        # 断言偏移后的结果应与期望结果 xp 相等
        tm.assert_frame_equal(rs, xp)

    # 定义测试方法，用于测试在填充值的情况下对包含分类数据的 Series 进行偏移
    def test_shift_categorical_fill_value(self, frame_or_series):
        # 创建包含分类数据的 Series ts
        ts = frame_or_series(["a", "b", "c", "d"], dtype="category")
        # 对 ts 进行向下偏移1行，使用 "a" 填充缺失值
        res = ts.shift(1, fill_value="a")
        # 创建期望结果 expected，指定类别和顺序
        expected = frame_or_series(
            pd.Categorical(
                ["a", "a", "b", "c"], categories=["a", "b", "c", "d"], ordered=False
            )
        )
        # 断言偏移后的结果应与期望结果 expected 相等
        tm.assert_equal(res, expected)

        # 检查使用不正确的填充值时是否会引发异常
        msg = r"Cannot setitem on a Categorical with a new category \(f\)"
        with pytest.raises(TypeError, match=msg):
            # 对 ts 进行向下偏移1行，使用 "f" 填充缺失值，预期会引发 TypeError 异常
            ts.shift(1, fill_value="f")
    # 定义一个测试方法，用于测试带有填充值的数据位移操作
    def test_shift_fill_value(self, frame_or_series):
        # GH#24128
        # 创建一个日期时间索引，从 "1/1/2000" 开始，每小时增加一条记录，共计5条记录
        dti = date_range("1/1/2000", periods=5, freq="h")

        # 使用给定的数据和索引创建一个DataFrame或Series对象
        ts = frame_or_series([1.0, 2.0, 3.0, 4.0, 5.0], index=dti)
        # 创建一个期望结果，将数据向后移动一个位置，并在缺失的位置填充0.0
        exp = frame_or_series([0.0, 1.0, 2.0, 3.0, 4.0], index=dti)
        # 检查填充值是否生效
        result = ts.shift(1, fill_value=0.0)
        tm.assert_equal(result, exp)

        # 创建另一个期望结果，将数据向后移动两个位置，并在缺失的位置填充0.0
        exp = frame_or_series([0.0, 0.0, 1.0, 2.0, 3.0], index=dti)
        result = ts.shift(2, fill_value=0.0)
        tm.assert_equal(result, exp)

        # 使用整数数据创建一个新的Series对象
        ts = frame_or_series([1, 2, 3])
        # 将该Series对象的数据向后移动两个位置，并在缺失的位置填充0
        res = ts.shift(2, fill_value=0)
        # 检查结果的数据类型是否与原始数据的数据类型相同
        assert tm.get_dtype(res) == tm.get_dtype(ts)

        # 创建一个带有整数数据的DataFrame或Series对象，并指定日期时间索引
        obj = frame_or_series([1, 2, 3, 4, 5], index=dti)
        # 创建一个期望结果，将数据向后移动一个位置，并在缺失的位置填充0
        exp = frame_or_series([0, 1, 2, 3, 4], index=dti)
        result = obj.shift(1, fill_value=0)
        tm.assert_equal(result, exp)

        # 创建另一个期望结果，将数据向后移动两个位置，并在缺失的位置填充0
        exp = frame_or_series([0, 0, 1, 2, 3], index=dti)
        result = obj.shift(2, fill_value=0)
        tm.assert_equal(result, exp)

    # 定义一个测试方法，用于测试空数据框的位移操作
    def test_shift_empty(self):
        # GH#8019
        # 创建一个空的DataFrame对象，只包含名为 "foo" 的列
        df = DataFrame({"foo": []})
        # 对该空数据框进行向前位移一位的操作
        rs = df.shift(-1)
        # 检查位移后的结果是否与原始数据框相同
        tm.assert_frame_equal(df, rs)

    # 定义一个测试方法，用于验证在存在重复列时基于位置的位移操作是否正常工作
    def test_shift_duplicate_columns(self):
        # GH#9092; 验证在存在重复列的情况下，基于位置的位移操作是否正常工作
        # 定义多个列列表，每个列表包含不同的列名或索引
        column_lists = [list(range(5)), [1] * 5, [1, 1, 2, 2, 1]]
        # 使用随机数据填充一个5x20的DataFrame对象
        data = np.random.default_rng(2).standard_normal((20, 5))

        # 创建一个空列表，用于存储位移后的DataFrame对象
        shifted = []
        # 遍历每个列列表
        for columns in column_lists:
            # 根据当前列列表复制数据创建一个DataFrame对象
            df = DataFrame(data.copy(), columns=columns)
            # 对每一列进行位置依次递增的位移操作
            for s in range(5):
                df.iloc[:, s] = df.iloc[:, s].shift(s + 1)
            # 将列名重置为0到4的连续整数
            df.columns = range(5)
            # 将位移后的DataFrame对象添加到列表中
            shifted.append(df)

        # 对基础情况进行验证，检查空值的数量是否符合预期
        nulls = shifted[0].isna().sum()
        tm.assert_series_equal(nulls, Series(range(1, 6), dtype="int64"))

        # 检查所有结果是否相同
        tm.assert_frame_equal(shifted[0], shifted[1])
        tm.assert_frame_equal(shifted[0], shifted[2])
    # 定义测试方法，用于测试在 axis=1 上进行多个块的移位操作
    def test_shift_axis1_multiple_blocks(self):
        # GH#35488: 测试标识号，可能是GitHub上的问题跟踪编号
        # 创建包含随机整数的 DataFrame df1
        df1 = DataFrame(np.random.default_rng(2).integers(1000, size=(5, 3)))
        # 创建包含随机整数的 DataFrame df2，与 df1 使用相同的随机种子
        df2 = DataFrame(np.random.default_rng(2).integers(1000, size=(5, 2)))
        # 沿着 axis=1 连接 df1 和 df2，形成 df3
        df3 = pd.concat([df1, df2], axis=1)
        # 断言 df3 内部块的数量为 2
        assert len(df3._mgr.blocks) == 2

        # 在 axis=1 上将 df3 向右移动 2 个位置，生成 result
        result = df3.shift(2, axis=1)

        # 从 df3 中取出指定列索引，组成期望的 DataFrame expected
        expected = df3.take([-1, -1, 0, 1, 2], axis=1)
        # 显式将结果转换为 float 类型，避免设置 NaN 时的隐式类型转换问题
        expected = expected.pipe(
            lambda df: df.set_axis(range(df.shape[1]), axis=1)
            .astype({0: "float", 1: "float"})
            .set_axis(df.columns, axis=1)
        )
        # 将期望结果的前两列设置为 NaN
        expected.iloc[:, :2] = np.nan
        # 设置期望结果的列名与 df3 的列名相同
        expected.columns = df3.columns

        # 断言 result 与 expected 的内容相等
        tm.assert_frame_equal(result, expected)

        # 处理周期小于 0 的情况
        # 重新构建 df3，因为上面的 `take` 调用已经整合了数据
        df3 = pd.concat([df1, df2], axis=1)
        # 再次断言 df3 内部块的数量为 2
        assert len(df3._mgr.blocks) == 2
        # 在 axis=1 上将 df3 向左移动 2 个位置，生成 result
        result = df3.shift(-2, axis=1)

        # 从 df3 中取出指定列索引，组成期望的 DataFrame expected
        expected = df3.take([2, 3, 4, -1, -1], axis=1)
        # 显式将结果转换为 float 类型，避免设置 NaN 时的隐式类型转换问题
        expected = expected.pipe(
            lambda df: df.set_axis(range(df.shape[1]), axis=1)
            .astype({3: "float", 4: "float"})
            .set_axis(df.columns, axis=1)
        )
        # 将期望结果的后两列设置为 NaN
        expected.iloc[:, -2:] = np.nan
        # 设置期望结果的列名与 df3 的列名相同
        expected.columns = df3.columns

        # 断言 result 与 expected 的内容相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试在 axis=1 上进行多个块的移位操作，并使用整数填充
    def test_shift_axis1_multiple_blocks_with_int_fill(self):
        # GH#42719: 测试标识号，可能是GitHub上的问题跟踪编号
        rng = np.random.default_rng(2)
        # 创建包含随机整数的 DataFrame df1
        df1 = DataFrame(rng.integers(1000, size=(5, 3), dtype=int))
        # 创建包含随机整数的 DataFrame df2，与 df1 使用相同的随机种子
        df2 = DataFrame(rng.integers(1000, size=(5, 2), dtype=int))
        # 从 df1 和 df2 中选择部分行和列，组成 df3
        df3 = pd.concat([df1.iloc[:4, 1:3], df2.iloc[:4, :]], axis=1)
        # 在 axis=1 上将 df3 向右移动 2 个位置，使用整数 0 填充缺失值，生成 result
        result = df3.shift(2, axis=1, fill_value=np.int_(0))
        # 断言 df3 内部块的数量为 2
        assert len(df3._mgr.blocks) == 2

        # 从 df3 中取出指定列索引，组成期望的 DataFrame expected
        expected = df3.take([-1, -1, 0, 1], axis=1)
        # 将期望结果的前两列设置为整数 0
        expected.iloc[:, :2] = np.int_(0)
        # 设置期望结果的列名与 df3 的列名相同
        expected.columns = df3.columns

        # 断言 result 与 expected 的内容相等
        tm.assert_frame_equal(result, expected)

        # 处理周期小于 0 的情况
        df3 = pd.concat([df1.iloc[:4, 1:3], df2.iloc[:4, :]], axis=1)
        # 在 axis=1 上将 df3 向左移动 2 个位置，使用整数 0 填充缺失值，生成 result
        result = df3.shift(-2, axis=1, fill_value=np.int_(0))
        # 断言 df3 内部块的数量为 2
        assert len(df3._mgr.blocks) == 2

        # 从 df3 中取出指定列索引，组成期望的 DataFrame expected
        expected = df3.take([2, 3, -1, -1], axis=1)
        # 将期望结果的后两列设置为整数 0
        expected.iloc[:, -2:] = np.int_(0)
        # 设置期望结果的列名与 df3 的列名相同
        expected.columns = df3.columns

        # 断言 result 与 expected 的内容相等
        tm.assert_frame_equal(result, expected)
    # 测试方法：测试周期性索引帧（DataFrame 或 Series）使用指定频率进行位移操作
    def test_period_index_frame_shift_with_freq(self, frame_or_series):
        # 创建一个周期性索引的 DataFrame ps，索引从 "2020-01-01" 开始，持续 4 个周期
        ps = DataFrame(range(4), index=pd.period_range("2020-01-01", periods=4))
        # 使用测试辅助函数获取特定类型的对象（DataFrame 或 Series）
        ps = tm.get_obj(ps, frame_or_series)

        # 使用周期性偏移1个单位，频率为"infer"
        shifted = ps.shift(1, freq="infer")
        # 再次进行反向偏移1个单位，频率为"infer"
        unshifted = shifted.shift(-1, freq="infer")
        # 断言反向偏移后的结果与原始 ps 相等
        tm.assert_equal(unshifted, ps)

        # 使用频率为"D"进行偏移操作
        shifted2 = ps.shift(freq="D")
        # 断言两次偏移结果相等
        tm.assert_equal(shifted, shifted2)

        # 使用 pandas 的偏移对象 Day() 进行偏移操作
        shifted3 = ps.shift(freq=offsets.Day())
        # 断言结果与之前的 shifted 结果相等
        tm.assert_equal(shifted, shifted3)

    # 测试方法：测试日期时间索引帧（DataFrame 或 Series）使用指定频率进行位移操作
    def test_datetime_frame_shift_with_freq(self, datetime_frame, frame_or_series):
        # 使用测试辅助函数获取特定类型的对象（DataFrame 或 Series）
        dtobj = tm.get_obj(datetime_frame, frame_or_series)

        # 使用日期时间偏移1个单位，频率为"infer"
        shifted = dtobj.shift(1, freq="infer")
        # 再次进行反向偏移1个单位，频率为"infer"
        unshifted = shifted.shift(-1, freq="infer")
        # 断言反向偏移后的结果与原始 dtobj 相等
        tm.assert_equal(dtobj, unshifted)

        # 使用当前索引的频率进行偏移操作
        shifted2 = dtobj.shift(freq=dtobj.index.freq)
        # 断言两次偏移结果相等
        tm.assert_equal(shifted, shifted2)

        # 创建一个使用日期时间索引的 DataFrame inferred_ts，与 datetime_frame 相似
        inferred_ts = DataFrame(
            datetime_frame.values,
            Index(np.asarray(datetime_frame.index)),
            columns=datetime_frame.columns,
        )
        inferred_ts = tm.get_obj(inferred_ts, frame_or_series)
        # 使用推断的频率进行偏移操作
        shifted = inferred_ts.shift(1, freq="infer")
        # 创建预期的偏移结果 expected，与原始 dtobj 偏移1个单位频率为"infer"相同
        expected = dtobj.shift(1, freq="infer")
        # 将预期结果的索引频率设为 None，以便与 shifted 的索引频率比较
        expected.index = expected.index._with_freq(None)
        # 断言推断后的偏移结果与预期结果相等
        tm.assert_equal(shifted, expected)

        # 再次进行反向偏移1个单位，频率为"infer"
        unshifted = shifted.shift(-1, freq="infer")
        # 断言反向偏移后的结果与 inferred_ts 相等
        tm.assert_equal(unshifted, inferred_ts)

    # 测试方法：测试周期性索引帧（DataFrame 或 Series）使用错误的频率引发异常
    def test_period_index_frame_shift_with_freq_error(self, frame_or_series):
        # 创建一个周期性索引的 DataFrame ps，索引从 "2020-01-01" 开始，持续 4 个周期
        ps = DataFrame(range(4), index=pd.period_range("2020-01-01", periods=4))
        # 使用测试辅助函数获取特定类型的对象（DataFrame 或 Series）
        ps = tm.get_obj(ps, frame_or_series)
        # 期望引发 ValueError 异常，并检查异常消息是否为指定信息
        msg = "Given freq M does not match PeriodIndex freq D"
        with pytest.raises(ValueError, match=msg):
            # 使用错误的频率 "M" 进行偏移操作，应引发异常
            ps.shift(freq="M")

    # 测试方法：测试日期时间索引帧（DataFrame 或 Series）使用错误的频率引发异常
    def test_datetime_frame_shift_with_freq_error(
        self, datetime_frame, frame_or_series
    ):
        # 使用测试辅助函数获取特定类型的对象（DataFrame 或 Series）
        dtobj = tm.get_obj(datetime_frame, frame_or_series)
        # 选择部分没有设置频率的行构成 no_freq
        no_freq = dtobj.iloc[[0, 5, 7]]
        # 期望引发 ValueError 异常，并检查异常消息是否为指定信息
        msg = "Freq was not set in the index hence cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            # 使用频率为 "infer" 的操作引发异常
            no_freq.shift(freq="infer")
    # 定义一个测试方法，用于测试 shift 方法在处理 numpy datetime64 数据时的行为
    def test_shift_dt64values_int_fill_deprecated(self):
        # GH#31971：GitHub issue 编号，表示这段代码解决了对应的问题
        # 创建一个包含两个 Timestamp 的 Series
        ser = Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])

        # 使用 pytest 断言捕获 TypeError 异常，验证 shift 方法在 fill_value 参数为整数时会抛出异常
        with pytest.raises(TypeError, match="value should be a"):
            ser.shift(1, fill_value=0)

        # 将 Series 转换为 DataFrame
        df = ser.to_frame()
        # 同样使用 pytest 断言捕获 TypeError 异常，验证 DataFrame 的 shift 方法在 fill_value 参数为整数时会抛出异常
        with pytest.raises(TypeError, match="value should be a"):
            df.shift(1, fill_value=0)

        # axis = 1 方向的操作
        # 创建一个包含两列的 DataFrame
        df2 = DataFrame({"A": ser, "B": ser})
        # 对 DataFrame 进行内部整合
        df2._consolidate_inplace()

        # 对 DataFrame 进行 axis=1 方向的 shift 操作，使用 fill_value=0 进行填充
        result = df2.shift(1, axis=1, fill_value=0)
        # 预期的结果 DataFrame
        expected = DataFrame({"A": [0, 0], "B": df2["A"]})
        # 使用 pytest 工具断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 对没有内部整合的 DataFrame 进行相同的操作；在 2.0 之前会有不同的行为
        df3 = DataFrame({"A": ser})
        df3["B"] = ser
        # 断言 DataFrame 的内部块数量为 2
        assert len(df3._mgr.blocks) == 2
        # 对 DataFrame 进行 axis=1 方向的 shift 操作，使用 fill_value=0 进行填充
        result = df3.shift(1, axis=1, fill_value=0)
        # 使用 pytest 工具断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 参数化测试函数，测试不同的输入参数组合
    @pytest.mark.parametrize(
        "as_cat",
        [
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
            False,
        ],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            date_range("2020-01-01", periods=2),  # 生成日期范围的时间戳
            date_range("2020-01-01", periods=2, tz="US/Pacific"),  # 生成带时区的日期范围
            pd.period_range("2020-01-01", periods=2, freq="D"),  # 生成周期范围的时间段
            pd.timedelta_range("2020 Days", periods=2, freq="D"),  # 生成时间差范围
            pd.interval_range(0, 3, periods=2),  # 生成间隔范围
            pytest.param(
                pd.array([1, 2], dtype="Int64"),
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
            pytest.param(
                pd.array([1, 2], dtype="Float32"),
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
        ],
        ids=lambda x: str(x.dtype),  # 将每个参数的数据类型作为其标识符
    )
    def test_shift_dt64values_axis1_invalid_fill(self, vals, as_cat):
        # GH#44564
        # 创建 Series 对象
        ser = Series(vals)
        # 如果需要转换为分类数据类型，则进行转换
        if as_cat:
            ser = ser.astype("category")

        # 创建 DataFrame 对象，包含单列 A
        df = DataFrame({"A": ser})
        # 对 DataFrame 进行列方向上的向左移动操作，填充值为 "foo"
        result = df.shift(-1, axis=1, fill_value="foo")
        # 创建期望结果的 DataFrame 对象
        expected = DataFrame({"A": ["foo", "foo"]})
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建包含多列 A 和 B 的 DataFrame 对象
        df2 = DataFrame({"A": ser, "B": ser})
        # 在原地整理 DataFrame 对象
        df2._consolidate_inplace()

        # 对 DataFrame 进行列方向上的向左移动操作，填充值为 "foo"
        result = df2.shift(-1, axis=1, fill_value="foo")
        # 创建期望结果的 DataFrame 对象
        expected = DataFrame({"A": df2["B"], "B": ["foo", "foo"]})
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建包含列 A 和 B 的 DataFrame 对象，但未整理
        df3 = DataFrame({"A": ser})
        df3["B"] = ser
        # 断言 DataFrame 的内部数据块数量是否为 2
        assert len(df3._mgr.blocks) == 2
        # 对 DataFrame 进行列方向上的向左移动操作，填充值为 "foo"
        result = df3.shift(-1, axis=1, fill_value="foo")
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_shift_axis1_categorical_columns(self):
        # GH#38434
        # 创建分类索引对象
        ci = CategoricalIndex(["a", "b", "c"])
        # 创建包含指定索引和列的 DataFrame 对象
        df = DataFrame(
            {"a": [1, 3], "b": [2, 4], "c": [5, 6]}, index=ci[:-1], columns=ci
        )
        # 对 DataFrame 进行列方向上的向左移动操作
        result = df.shift(axis=1)

        # 创建期望结果的 DataFrame 对象
        expected = DataFrame(
            {"a": [np.nan, np.nan], "b": [1, 3], "c": [2, 4]}, index=ci[:-1], columns=ci
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 设置 periods 不等于 1 的情况下，对 DataFrame 进行列方向上的向左移动操作
        result = df.shift(2, axis=1)
        # 创建期望结果的 DataFrame 对象
        expected = DataFrame(
            {"a": [np.nan, np.nan], "b": [np.nan, np.nan], "c": [1, 3]},
            index=ci[:-1],
            columns=ci,
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_shift_axis1_many_periods(self):
        # GH#44978 periods > len(columns)
        # 创建具有随机数据的 DataFrame 对象
        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        # 对 DataFrame 进行列方向上的向左移动操作，periods 大于列数，填充值为 None
        shifted = df.shift(6, axis=1, fill_value=None)

        # 创建期望结果的 DataFrame 对象，全部为 NaN
        expected = df * np.nan
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(shifted, expected)

        # 对 DataFrame 进行列方向上的向右移动操作，periods 大于列数，填充值为 None
        shifted2 = df.shift(-6, axis=1, fill_value=None)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(shifted2, expected)

    def test_shift_with_offsets_freq(self):
        # 创建包含时间序列数据的 DataFrame 对象
        df = DataFrame({"x": [1, 2, 3]}, index=date_range("2000", periods=3))
        # 对 DataFrame 进行时间偏移量为 "1MS" 的操作
        shifted = df.shift(freq="1MS")
        # 创建期望结果的 DataFrame 对象
        expected = DataFrame(
            {"x": [1, 2, 3]},
            index=date_range(start="02/01/2000", end="02/01/2000", periods=3),
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(shifted, expected)

    def test_shift_with_iterable_basic_functionality(self):
        # GH#44424
        # 创建包含数据的字典对象
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        # 创建偏移量的列表
        shifts = [0, 1, 2]

        # 创建 DataFrame 对象
        df = DataFrame(data)
        # 对 DataFrame 进行多个偏移量的操作
        shifted = df.shift(shifts)

        # 创建期望结果的 DataFrame 对象
        expected = DataFrame(
            {
                "a_0": [1, 2, 3],
                "b_0": [4, 5, 6],
                "a_1": [np.nan, 1.0, 2.0],
                "b_1": [np.nan, 4.0, 5.0],
                "a_2": [np.nan, np.nan, 1.0],
                "b_2": [np.nan, np.nan, 4.0],
            }
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(expected, shifted)
    def test_shift_with_iterable_series(self):
        # GH#44424
        # 准备测试数据，包含一个字典和一个列表
        data = {"a": [1, 2, 3]}
        shifts = [0, 1, 2]

        # 创建 DataFrame 对象
        df = DataFrame(data)
        # 从 DataFrame 中选择列 'a'
        s = df["a"]
        # 断言 DataFrame 列 'a' 向前移动后与整体 DataFrame 向前移动结果相等
        tm.assert_frame_equal(s.shift(shifts), df.shift(shifts))

    def test_shift_with_iterable_freq_and_fill_value(self):
        # GH#44424
        # 创建一个包含随机数据的 DataFrame，时间索引为每小时频率
        df = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=date_range("1/1/2000", periods=5, freq="h"),
        )

        # 断言：使用填充值1，将 DataFrame 向前移动一个单位，并重命名列名
        tm.assert_frame_equal(
            df.shift([1], fill_value=1).rename(columns=lambda x: int(x[0])),
            df.shift(1, fill_value=1),
        )

        # 断言：使用频率参数将 DataFrame 按小时向前移动一个单位，并重命名列名
        tm.assert_frame_equal(
            df.shift([1], freq="h").rename(columns=lambda x: int(x[0])),
            df.shift(1, freq="h"),
        )

    def test_shift_with_iterable_check_other_arguments(self):
        # GH#44424
        # 准备测试数据，包含两个列的字典和一个列表
        data = {"a": [1, 2], "b": [4, 5]}
        shifts = [0, 1]
        df = DataFrame(data)

        # 测试后缀参数：将 'a' 列向前移动，并添加 '_suffix' 后缀
        shifted = df[["a"]].shift(shifts, suffix="_suffix")
        expected = DataFrame({"a_suffix_0": [1, 2], "a_suffix_1": [np.nan, 1.0]})
        tm.assert_frame_equal(shifted, expected)

        # 检查错误输入情况：当进行多个列向前移动时，axis 参数不能为 1
        msg = "If `periods` contains multiple shifts, `axis` cannot be 1."
        with pytest.raises(ValueError, match=msg):
            df.shift(shifts, axis=1)

        # 检查错误输入情况：当 periods 参数为非整数时抛出 TypeError
        msg = "Periods must be integer, but s is <class 'str'>."
        with pytest.raises(TypeError, match=msg):
            df.shift(["s"])

        # 检查错误输入情况：当 periods 参数为空列表时抛出 ValueError
        msg = "If `periods` is an iterable, it cannot be empty."
        with pytest.raises(ValueError, match=msg):
            df.shift([])

        # 检查错误输入情况：当 periods 参数为整数时不能指定 suffix 参数
        msg = "Cannot specify `suffix` if `periods` is an int."
        with pytest.raises(ValueError, match=msg):
            df.shift(1, suffix="fails")

    def test_shift_axis_one_empty(self):
        # GH#57301
        # 创建一个空的 DataFrame
        df = DataFrame()
        # 在 axis=1 上进行向前移动一个单位，预期结果与原 DataFrame 相等
        result = df.shift(1, axis=1)
        tm.assert_frame_equal(result, df)
```