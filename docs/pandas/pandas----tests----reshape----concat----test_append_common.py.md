# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_append_common.py`

```
import zoneinfo  # 导入zoneinfo模块，用于处理时区信息

import numpy as np  # 导入numpy库，用于数值计算
import pytest  # 导入pytest库，用于编写和执行单元测试

import pandas as pd  # 导入pandas库，用于数据处理和分析
from pandas import (  # 从pandas库中导入特定子模块和类
    Categorical,
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm  # 导入pandas内部测试工具模块

@pytest.fixture(  # 定义一个pytest的测试夹具
    params=list(  # 使用参数化装饰器，为测试提供多组输入参数
        {
            "bool": [True, False, True],  # 布尔类型的数据列表
            "int64": [1, 2, 3],  # 64位整数类型的数据列表
            "float64": [1.1, np.nan, 3.3],  # 64位浮点数类型的数据列表，包含NaN
            "category": Categorical(["X", "Y", "Z"]),  # 分类类型的数据列表
            "object": ["a", "b", "c"],  # 对象类型的数据列表
            "datetime64[s]": [  # 时间日期类型的数据列表（秒精度）
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timestamp("2011-01-03"),
            ],
            "datetime64[s, US/Eastern]": [  # 带时区的时间日期类型的数据列表（秒精度）
                pd.Timestamp("2011-01-01", tz="US/Eastern"),
                pd.Timestamp("2011-01-02", tz="US/Eastern"),
                pd.Timestamp("2011-01-03", tz="US/Eastern"),
            ],
            "timedelta64[ns]": [  # 时间差类型的数据列表（纳秒精度）
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
                pd.Timedelta("3 days"),
            ],
            "period[M]": [  # 时间段类型的数据列表（月度）
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Period("2011-03", freq="M"),
            ],
        }.items()  # 将字典转换为列表以便参数化
    )
)
def item(request):  # 定义item夹具，用于传递参数化的测试数据
    key, data = request.param  # 从参数中获取测试数据的键和值
    return key, data  # 返回键和对应的数据


@pytest.fixture  # 定义另一个pytest的测试夹具
def item2(item):  # 使用item夹具作为参数
    return item  # 返回item夹具的结果


class TestConcatAppendCommon:  # 定义一个测试类，用于测试concat和append的通用dtype强制规则
    """
    Test common dtype coercion rules between concat and append.
    """

    def test_dtypes(self, item, index_or_series, using_infer_string):
        # to confirm test case covers intended dtypes
        typ, vals = item  # 获取参数化测试数据的类型和值
        obj = index_or_series(vals)  # 根据vals创建一个索引或者系列对象
        if typ == "object" and using_infer_string:  # 如果类型是"object"并且使用了字符串推断
            typ = "string"  # 将类型更改为"string"
        if isinstance(obj, Index):  # 如果obj是索引对象
            assert obj.dtype == typ  # 断言索引对象的数据类型与typ相同
        elif isinstance(obj, Series):  # 如果obj是系列对象
            if typ.startswith("period"):  # 如果类型以"period"开头
                assert obj.dtype == "Period[M]"  # 断言系列对象的数据类型为"Period[M]"
            else:
                assert obj.dtype == typ  # 否则断言系列对象的数据类型与typ相同
    # 定义一个测试函数，用于测试不同数据类型的连接和强制类型转换
    def test_concatlike_dtypes_coercion(self, item, item2, request):
        # GH 13660
        # 从元组中解包出类型和值
        typ1, vals1 = item
        typ2, vals2 = item2

        # 将第二个元组的值赋给第三个变量
        vals3 = vals2

        # 推断期望的索引数据类型和系列数据类型
        exp_index_dtype = None
        exp_series_dtype = None

        # 根据类型判断是否跳过测试
        if typ1 == typ2:
            pytest.skip("same dtype is tested in test_concatlike_same_dtypes")
        elif typ1 == "category" or typ2 == "category":
            pytest.skip("categorical type tested elsewhere")

        # 指定期望的数据类型
        if typ1 == "bool" and typ2 in ("int64", "float64"):
            # 系列按照 numpy 规则转换为数值类型
            # 索引不会，因为布尔类型是对象类型
            exp_series_dtype = typ2
            # 对于此情况，标记为预期失败，原因是类型转换为对象类型
            mark = pytest.mark.xfail(reason="GH#39187 casting to object")
            request.applymarker(mark)
        elif typ2 == "bool" and typ1 in ("int64", "float64"):
            exp_series_dtype = typ1
            mark = pytest.mark.xfail(reason="GH#39187 casting to object")
            request.applymarker(mark)
        elif typ1 in {"datetime64[ns, US/Eastern]", "timedelta64[ns]"} or typ2 in {
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        }:
            # 对于日期时间或时间间隔类型，期望索引和系列的数据类型为对象类型
            exp_index_dtype = object
            exp_series_dtype = object

        # 计算预期的数据
        exp_data = vals1 + vals2
        exp_data3 = vals1 + vals2 + vals3

        # ----- Index ----- #

        # index.append
        # GH#39817
        # 将 vals2 添加到 vals1 的索引上，并验证结果与期望相等
        res = Index(vals1).append(Index(vals2))
        exp = Index(exp_data, dtype=exp_index_dtype)
        tm.assert_index_equal(res, exp)

        # 3 elements
        # 将 vals2 和 vals3 添加到 vals1 的索引上，并验证结果与期望相等
        res = Index(vals1).append([Index(vals2), Index(vals3)])
        exp = Index(exp_data3, dtype=exp_index_dtype)
        tm.assert_index_equal(res, exp)

        # ----- Series ----- #

        # series._append
        # GH#39817
        # 将 vals2 添加到 vals1 的系列上，并验证结果与期望相等，忽略索引
        res = Series(vals1)._append(Series(vals2), ignore_index=True)
        exp = Series(exp_data, dtype=exp_series_dtype)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # concat
        # GH#39817
        # 使用 pd.concat 将 vals1 和 vals2 连接为系列，并验证结果与期望相等，忽略索引类型检查
        res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
        tm.assert_series_equal(res, exp, check_index_type=True)

        # 3 elements
        # 将 vals2 和 vals3 添加到 vals1 的系列上，并验证结果与期望相等，忽略索引
        res = Series(vals1)._append([Series(vals2), Series(vals3)], ignore_index=True)
        exp = Series(exp_data3, dtype=exp_series_dtype)
        tm.assert_series_equal(res, exp)

        # GH#39817
        # 使用 pd.concat 将 vals1、vals2 和 vals3 连接为系列，并验证结果与期望相等，忽略索引类型检查
        res = pd.concat(
            [Series(vals1), Series(vals2), Series(vals3)],
            ignore_index=True,
        )
        tm.assert_series_equal(res, exp)
    def test_concatlike_common_coerce_to_pandas_object(self):
        # GH 13626
        # result must be Timestamp/Timedelta, not datetime.datetime/timedelta
        # 创建一个 DatetimeIndex 对象，包含两个日期字符串
        dti = pd.DatetimeIndex(["2011-01-01", "2011-01-02"])
        # 创建一个 TimedeltaIndex 对象，包含两个时间差字符串
        tdi = pd.TimedeltaIndex(["1 days", "2 days"])

        # 期望的结果是一个 Index 对象，包含日期时间戳和时间差对象
        exp = Index(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
            ]
        )

        # 将两个 Index 对象合并
        res = dti.append(tdi)
        # 检查结果是否符合期望
        tm.assert_index_equal(res, exp)
        # 检查结果的第一个元素是否为 Timestamp 对象
        assert isinstance(res[0], pd.Timestamp)
        # 检查结果的最后一个元素是否为 Timedelta 对象
        assert isinstance(res[-1], pd.Timedelta)

        # 创建两个 Series 对象，分别包含 DatetimeIndex 和 TimedeltaIndex
        dts = Series(dti)
        tds = Series(tdi)
        # 使用内部方法 _append 合并两个 Series 对象
        res = dts._append(tds)
        # 检查合并后的 Series 是否符合期望
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        # 检查合并后的 Series 的第一个元素是否为 Timestamp 对象
        assert isinstance(res.iloc[0], pd.Timestamp)
        # 检查合并后的 Series 的最后一个元素是否为 Timedelta 对象
        assert isinstance(res.iloc[-1], pd.Timedelta)

        # 使用 pd.concat 方法合并两个 Series 对象
        res = pd.concat([dts, tds])
        # 检查合并后的 Series 是否符合期望
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        # 检查合并后的 Series 的第一个元素是否为 Timestamp 对象
        assert isinstance(res.iloc[0], pd.Timestamp)
        # 检查合并后的 Series 的最后一个元素是否为 Timedelta 对象
        assert isinstance(res.iloc[-1], pd.Timedelta)

    def test_concatlike_datetimetz(self, tz_aware_fixture):
        # GH 7795
        # 创建一个带有时区信息的 DatetimeIndex 对象
        tz = tz_aware_fixture
        dti1 = pd.DatetimeIndex(["2011-01-01", "2011-01-02"], tz=tz)
        dti2 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"], tz=tz)

        # 期望的结果是一个带有时区信息的 DatetimeIndex 对象
        exp = pd.DatetimeIndex(
            ["2011-01-01", "2011-01-02", "2012-01-01", "2012-01-02"], tz=tz
        )

        # 将两个带有时区信息的 DatetimeIndex 对象合并
        res = dti1.append(dti2)
        # 检查结果是否符合期望
        tm.assert_index_equal(res, exp)

        # 创建两个 Series 对象，分别包含带有时区信息的 DatetimeIndex
        dts1 = Series(dti1)
        dts2 = Series(dti2)
        # 使用内部方法 _append 合并两个 Series 对象
        res = dts1._append(dts2)
        # 检查合并后的 Series 是否符合期望
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # 使用 pd.concat 方法合并两个 Series 对象
        res = pd.concat([dts1, dts2])
        # 检查合并后的 Series 是否符合期望
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo", "EST5EDT"])
    def test_concatlike_datetimetz_short(self, tz):
        # GH#7795
        # 创建一个带有时区信息的日期范围
        ix1 = pd.date_range(start="2014-07-15", end="2014-07-17", freq="D", tz=tz)
        ix2 = pd.DatetimeIndex(["2014-07-11", "2014-07-21"], tz=tz)
        # 创建两个 DataFrame 对象，带有相同的时区信息
        df1 = DataFrame(0, index=ix1, columns=["A", "B"])
        df2 = DataFrame(0, index=ix2, columns=["A", "B"])

        # 期望的结果是一个带有时区信息的 DatetimeIndex 对象，精确到纳秒
        exp_idx = pd.DatetimeIndex(
            ["2014-07-15", "2014-07-16", "2014-07-17", "2014-07-11", "2014-07-21"],
            tz=tz,
        ).as_unit("ns")
        exp = DataFrame(0, index=exp_idx, columns=["A", "B"])

        # 使用 DataFrame 的内部方法 _append 合并两个 DataFrame 对象
        tm.assert_frame_equal(df1._append(df2), exp)
        # 使用 pd.concat 方法合并两个 DataFrame 对象
        tm.assert_frame_equal(pd.concat([df1, df2]), exp)
    def test_concatlike_datetimetz_to_object(self, tz_aware_fixture):
        # 定义一个测试函数，接受一个带时区信息的参数
        tz = tz_aware_fixture
        # 将参数赋值给变量tz
        # GH 13660
        # 标识 GitHub issue 编号

        # different tz coerces to object
        # 不同的时区强制转换为对象
        dti1 = pd.DatetimeIndex(["2011-01-01", "2011-01-02"], tz=tz)
        # 创建带时区信息的日期时间索引对象dti1
        dti2 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"])
        # 创建不带时区信息的日期时间索引对象dti2

        exp = Index(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.Timestamp("2011-01-02", tz=tz),
                pd.Timestamp("2012-01-01"),
                pd.Timestamp("2012-01-02"),
            ],
            dtype=object,
        )
        # 创建期望的索引对象exp，包含带时区和不带时区的时间戳

        res = dti1.append(dti2)
        # 将dti2追加到dti1上，返回新的索引对象res
        tm.assert_index_equal(res, exp)
        # 断言res与exp相等

        dts1 = Series(dti1)
        # 创建带时区信息的Series对象dts1
        dts2 = Series(dti2)
        # 创建不带时区信息的Series对象dts2
        res = dts1._append(dts2)
        # 将dts2追加到dts1上，返回新的Series对象res
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        # 断言res与期望的Series对象相等

        res = pd.concat([dts1, dts2])
        # 将dts1和dts2进行连接，返回新的Series对象res
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        # 断言res与期望的Series对象相等

        # different tz
        # 不同的时区
        tz_diff = zoneinfo.ZoneInfo("US/Hawaii")
        # 创建一个不同时区的ZoneInfo对象tz_diff
        dti3 = pd.DatetimeIndex(["2012-01-01", "2012-01-02"], tz=tz_diff)
        # 创建带不同时区信息的日期时间索引对象dti3

        exp = Index(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.Timestamp("2011-01-02", tz=tz),
                pd.Timestamp("2012-01-01", tz=tz_diff),
                pd.Timestamp("2012-01-02", tz=tz_diff),
            ],
            dtype=object,
        )
        # 创建期望的索引对象exp，包含带不同时区的时间戳

        res = dti1.append(dti3)
        # 将dti3追加到dti1上，返回新的索引对象res
        tm.assert_index_equal(res, exp)
        # 断言res与exp相等

        dts1 = Series(dti1)
        # 创建带时区信息的Series对象dts1
        dts3 = Series(dti3)
        # 创建带不同时区信息的Series对象dts3
        res = dts1._append(dts3)
        # 将dts3追加到dts1上，返回新的Series对象res
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        # 断言res与期望的Series对象相等

        res = pd.concat([dts1, dts3])
        # 将dts1和dts3进行连接，返回新的Series对象res
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        # 断言res与期望的Series对象相等

    def test_concatlike_common_period(self):
        # GH 13660
        # 标识 GitHub issue 编号
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        # 创建周期索引对象pi1
        pi2 = pd.PeriodIndex(["2012-01", "2012-02"], freq="M")
        # 创建周期索引对象pi2

        exp = pd.PeriodIndex(["2011-01", "2011-02", "2012-01", "2012-02"], freq="M")
        # 创建期望的周期索引对象exp

        res = pi1.append(pi2)
        # 将pi2追加到pi1上，返回新的索引对象res
        tm.assert_index_equal(res, exp)
        # 断言res与exp相等

        ps1 = Series(pi1)
        # 创建周期索引对象pi1的Series对象ps1
        ps2 = Series(pi2)
        # 创建周期索引对象pi2的Series对象ps2
        res = ps1._append(ps2)
        # 将ps2追加到ps1上，返回新的Series对象res
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        # 断言res与期望的Series对象相等

        res = pd.concat([ps1, ps2])
        # 将ps1和ps2进行连接，返回新的Series对象res
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        # 断言res与期望的Series对象相等
    def test_concatlike_common_period_diff_freq_to_object(self):
        # GH 13221
        # 创建一个 PeriodIndex 对象 pi1，包含两个月份的周期，频率为月份
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        # 创建一个 PeriodIndex 对象 pi2，包含两个日期的周期，频率为天
        pi2 = pd.PeriodIndex(["2012-01-01", "2012-02-01"], freq="D")

        # 期望的结果是一个 Index 对象，包含四个周期对象，其中两个为月份，两个为日期
        exp = Index(
            [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Period("2012-01-01", freq="D"),
                pd.Period("2012-02-01", freq="D"),
            ],
            dtype=object,
        )

        # 将 pi1 和 pi2 合并成一个新的 PeriodIndex 对象 res，并验证其是否与期望的 exp 相等
        res = pi1.append(pi2)
        tm.assert_index_equal(res, exp)

        # 创建 Series 对象 ps1 和 ps2，分别使用 pi1 和 pi2 作为其数据
        ps1 = Series(pi1)
        ps2 = Series(pi2)

        # 将 ps1 和 ps2 合并成一个新的 Series 对象 res，并验证其是否与期望的 exp 相等
        res = ps1._append(ps2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # 使用 pd.concat 将 ps1 和 ps2 合并成一个新的 Series 对象 res，并验证其是否与期望的 exp 相等
        res = pd.concat([ps1, ps2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period_mixed_dt_to_object(self):
        # GH 13221
        # 创建一个 PeriodIndex 对象 pi1，包含两个月份的周期，频率为月份
        pi1 = pd.PeriodIndex(["2011-01", "2011-02"], freq="M")
        # 创建一个 TimedeltaIndex 对象 tdi，包含两个日期的 timedelta 值
        tdi = pd.TimedeltaIndex(["1 days", "2 days"])

        # 期望的结果是一个 Index 对象，包含四个对象，其中两个为月份周期，两个为 timedelta 值
        exp = Index(
            [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
            ],
            dtype=object,
        )

        # 将 pi1 和 tdi 合并成一个新的 Index 对象 res，并验证其是否与期望的 exp 相等
        res = pi1.append(tdi)
        tm.assert_index_equal(res, exp)

        # 创建 Series 对象 ps1 和 tds，分别使用 pi1 和 tdi 作为其数据
        ps1 = Series(pi1)
        tds = Series(tdi)

        # 将 ps1 和 tds 合并成一个新的 Series 对象 res，并验证其是否与期望的 exp 相等
        res = ps1._append(tds)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # 使用 pd.concat 将 ps1 和 tds 合并成一个新的 Series 对象 res，并验证其是否与期望的 exp 相等
        res = pd.concat([ps1, tds])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # 创建反向顺序的期望结果 exp
        exp = Index(
            [
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02", freq="M"),
            ],
            dtype=object,
        )

        # 将 tdi 和 pi1 合并成一个新的 Index 对象 res，并验证其是否与期望的 exp 相等
        res = tdi.append(pi1)
        tm.assert_index_equal(res, exp)

        # 再次创建 Series 对象 ps1 和 tds，分别使用 pi1 和 tdi 作为其数据
        ps1 = Series(pi1)
        tds = Series(tdi)

        # 将 tds 和 ps1 合并成一个新的 Series 对象 res，并验证其是否与期望的 exp 相等
        res = tds._append(ps1)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

        # 使用 pd.concat 将 tds 和 ps1 合并成一个新的 Series 对象 res，并验证其是否与期望的 exp 相等
        res = pd.concat([tds, ps1])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
    def test_concat_categorical(self):
        # GH 13524
        # 测试连接分类数据的函数，这里是一个特定的 GitHub 问题编号

        # 创建第一个 Series，包含数字和 NaN，使用分类数据类型
        s1 = Series([1, 2, np.nan], dtype="category")
        
        # 创建第二个 Series，包含数字，使用分类数据类型
        s2 = Series([2, 1, 2], dtype="category")

        # 期望的结果，将两个 Series 连接成一个，忽略索引
        exp = Series([1, 2, np.nan, 2, 1, 2], dtype="category")
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        
        # 调用 s1 的内部方法 _append 连接 s2，忽略索引
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        # 创建第一个 Series，部分分类数据不同
        s1 = Series([3, 2], dtype="category")
        
        # 创建第二个 Series，部分分类数据不同
        s2 = Series([2, 1], dtype="category")

        # 期望的结果，连接两个 Series，忽略索引，结果不是分类数据类型
        exp = Series([3, 2, 2, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        
        # 调用 s1 的内部方法 _append 连接 s2，忽略索引，结果不是分类数据类型
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        # 创建第一个 Series，完全不同的分类数据，但数据类型相同
        s1 = Series([10, 11, np.nan], dtype="category")
        
        # 创建第二个 Series，完全不同的分类数据，但数据类型相同
        s2 = Series([np.nan, 1, 3, 2], dtype="category")

        # 期望的结果，连接两个 Series，忽略索引，结果不是分类数据类型
        exp = Series([10, 11, np.nan, np.nan, 1, 3, 2], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        
        # 调用 s1 的内部方法 _append 连接 s2，忽略索引，结果不是分类数据类型
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)


    def test_union_categorical_same_categories_different_order(self):
        # https://github.com/pandas-dev/pandas/issues/19096
        # 测试连接具有相同分类不同顺序的 Series 的函数

        # 创建第一个 Series，包含分类数据并指定分类的顺序
        a = Series(Categorical(["a", "b", "c"], categories=["a", "b", "c"]))
        
        # 创建第二个 Series，包含分类数据并指定不同的分类顺序
        b = Series(Categorical(["a", "b", "c"], categories=["b", "a", "c"]))

        # 连接两个 Series，忽略索引
        result = pd.concat([a, b], ignore_index=True)
        
        # 期望的结果，连接后的 Series 包含完整的分类并保持初始分类顺序
        expected = Series(
            Categorical(["a", "b", "c", "a", "b", "c"], categories=["a", "b", "c"])
        )
        tm.assert_series_equal(result, expected)
    def test_concat_categorical_coercion(self):
        # 测试用例函数：测试分类数据的合并转换

        # 类别数据 + 非类别数据 => 非类别数据
        s1 = Series([1, 2, np.nan], dtype="category")  # 创建一个类别数据的 Series 对象
        s2 = Series([2, 1, 2])  # 创建一个普通数据的 Series 对象

        exp = Series([1, 2, np.nan, 2, 1, 2], dtype=np.float64)  # 预期结果为合并后的 Series 对象
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)  # 断言合并后的结果与预期结果相等
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)  # 断言使用 _append 方法的结果与预期结果相等

        # 结果不应受第一个元素的 dtype 影响
        exp = Series([2, 1, 2, 1, 2, np.nan], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

        # 所有值都不是类别数据 => 非类别数据
        s1 = Series([3, 2], dtype="category")
        s2 = Series([2, 1])

        exp = Series([3, 2, 2, 1])  # 预期结果为合并后的 Series 对象
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)  # 断言合并后的结果与预期结果相等
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)  # 断言使用 _append 方法的结果与预期结果相等

        exp = Series([2, 1, 3, 2])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)

        # 完全不同的类别 => 非类别数据
        s1 = Series([10, 11, np.nan], dtype="category")
        s2 = Series([1, 3, 2])

        exp = Series([10, 11, np.nan, 1, 3, 2], dtype=np.float64)
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)

        exp = Series([1, 3, 2, 10, 11, np.nan], dtype=np.float64)
    def test_concat_categorical_3elem_coercion(self):
        # 测试用例函数：test_concat_categorical_3elem_coercion

        # GH 13524
        # GitHub issue编号: 13524

        # mixed dtypes => not-category
        # 混合数据类型 => 非分类数据
        s1 = Series([1, 2, np.nan], dtype="category")
        # 创建一个Series，包含整数和缺失值，数据类型为"category"
        s2 = Series([2, 1, 2], dtype="category")
        # 创建一个Series，包含整数，数据类型为"category"
        s3 = Series([1, 2, 1, 2, np.nan])
        # 创建一个Series，包含整数和缺失值

        exp = Series([1, 2, np.nan, 2, 1, 2, 1, 2, 1, 2, np.nan], dtype="float")
        # 期望的结果Series，包含浮点数，用于对比测试
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        # 使用pd.concat合并s1, s2, s3，忽略索引，并与期望的结果进行比较
        tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)
        # 使用s1._append合并s2和s3，忽略索引，并与期望的结果进行比较

        exp = Series([1, 2, 1, 2, np.nan, 1, 2, np.nan, 2, 1, 2], dtype="float")
        # 另一个期望的结果Series，包含浮点数，用于对比测试
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        # 使用pd.concat合并s3, s1, s2，忽略索引，并与期望的结果进行比较
        tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)
        # 使用s3._append合并s1和s2，忽略索引，并与期望的结果进行比较

        # values are all in either category => not-category
        # 所有值都在分类中 => 非分类数据
        s1 = Series([4, 5, 6], dtype="category")
        # 创建一个Series，包含整数，数据类型为"category"
        s2 = Series([1, 2, 3], dtype="category")
        # 创建一个Series，包含整数，数据类型为"category"
        s3 = Series([1, 3, 4])
        # 创建一个Series，包含整数

        exp = Series([4, 5, 6, 1, 2, 3, 1, 3, 4])
        # 期望的结果Series，包含整数，用于对比测试
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        # 使用pd.concat合并s1, s2, s3，忽略索引，并与期望的结果进行比较
        tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)
        # 使用s1._append合并s2和s3，忽略索引，并与期望的结果进行比较

        exp = Series([1, 3, 4, 4, 5, 6, 1, 2, 3])
        # 另一个期望的结果Series，包含整数，用于对比测试
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        # 使用pd.concat合并s3, s1, s2，忽略索引，并与期望的结果进行比较
        tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)
        # 使用s3._append合并s1和s2，忽略索引，并与期望的结果进行比较

        # values are all in either category => not-category
        # 所有值都在分类中 => 非分类数据
        s1 = Series([4, 5, 6], dtype="category")
        # 创建一个Series，包含整数，数据类型为"category"
        s2 = Series([1, 2, 3], dtype="category")
        # 创建一个Series，包含整数，数据类型为"category"
        s3 = Series([10, 11, 12])
        # 创建一个Series，包含整数

        exp = Series([4, 5, 6, 1, 2, 3, 10, 11, 12])
        # 期望的结果Series，包含整数，用于对比测试
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        # 使用pd.concat合并s1, s2, s3，忽略索引，并与期望的结果进行比较
        tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)
        # 使用s1._append合并s2和s3，忽略索引，并与期望的结果进行比较

        exp = Series([10, 11, 12, 4, 5, 6, 1, 2, 3])
        # 另一个期望的结果Series，包含整数，用于对比测试
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        # 使用pd.concat合并s3, s1, s2，忽略索引，并与期望的结果进行比较
        tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)
        # 使用s3._append合并s1和s2，忽略索引，并与期望的结果进行比较
    def test_concat_categorical_ordered(self):
        # GH 13524
        # 定义一个测试方法，用于验证有序分类数据的拼接行为

        s1 = Series(Categorical([1, 2, np.nan], ordered=True))
        # 创建一个包含有序分类数据的 Series 对象 s1，包含值 1、2 和 NaN

        s2 = Series(Categorical([2, 1, 2], ordered=True))
        # 创建另一个包含有序分类数据的 Series 对象 s2，包含值 2、1 和 2

        exp = Series(Categorical([1, 2, np.nan, 2, 1, 2], ordered=True))
        # 创建预期的结果 Series 对象 exp，包含 s1 和 s2 拼接后的有序分类数据

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        # 使用 pandas 的 concat 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
        # 使用 Series 对象的 _append 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        exp = Series(Categorical([1, 2, np.nan, 2, 1, 2, 1, 2, np.nan], ordered=True))
        # 创建另一个预期的结果 Series 对象 exp，包含 s1、s2 和 s1 再次拼接后的有序分类数据

        tm.assert_series_equal(pd.concat([s1, s2, s1], ignore_index=True), exp)
        # 使用 pandas 的 concat 方法将 s1、s2 和 s1 按行拼接，并忽略索引，验证结果是否等于 exp

        tm.assert_series_equal(s1._append([s2, s1], ignore_index=True), exp)
        # 使用 Series 对象的 _append 方法将 s1、s2 和 s1 按行拼接，并忽略索引，验证结果是否等于 exp

    def test_concat_categorical_coercion_nan(self):
        # GH 13524
        # 定义另一个测试方法，用于验证分类数据在包含 NaN 值时的拼接行为

        # some edge cases
        # category + not-category => not category
        # 创建一个边缘情况的测试用例，验证分类数据与非分类数据拼接后的数据类型

        s1 = Series(np.array([np.nan, np.nan], dtype=np.float64), dtype="category")
        # 创建一个包含 NaN 的 numpy 数组，并指定其为分类数据类型的 Series 对象 s1

        s2 = Series([np.nan, 1])
        # 创建另一个包含 NaN 和整数 1 的 Series 对象 s2

        exp = Series([np.nan, np.nan, np.nan, 1])
        # 创建预期的结果 Series 对象 exp，包含 s1 和 s2 拼接后的数据

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        # 使用 pandas 的 concat 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
        # 使用 Series 对象的 _append 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        s1 = Series([1, np.nan], dtype="category")
        # 创建一个包含整数 1 和 NaN 的分类数据类型的 Series 对象 s1

        s2 = Series([np.nan, np.nan])
        # 创建另一个包含 NaN 的 Series 对象 s2

        exp = Series([1, np.nan, np.nan, np.nan], dtype="float")
        # 创建预期的结果 Series 对象 exp，包含 s1 和 s2 拼接后的数据，并指定数据类型为浮点数

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        # 使用 pandas 的 concat 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
        # 使用 Series 对象的 _append 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        # mixed dtype, all nan-likes => not-category
        # 创建一个混合数据类型且全部为 NaN 类似值的测试用例，验证拼接后数据类型为非分类数据

        s1 = Series([np.nan, np.nan], dtype="category")
        # 创建一个包含 NaN 的分类数据类型的 Series 对象 s1

        s2 = Series([np.nan, np.nan])
        # 创建另一个包含 NaN 的 Series 对象 s2

        exp = Series([np.nan, np.nan, np.nan, np.nan])
        # 创建预期的结果 Series 对象 exp，包含 s1 和 s2 拼接后的数据

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        # 使用 pandas 的 concat 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
        # 使用 Series 对象的 _append 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        # 使用 pandas 的 concat 方法将 s2 和 s1 按行拼接，并忽略索引，验证结果是否等于 exp

        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)
        # 使用 Series 对象的 _append 方法将 s2 和 s1 按行拼接，并忽略索引，验证结果是否等于 exp

        # all category nan-likes => category
        # 创建一个全部为分类数据类型且 NaN 类似值的测试用例，验证拼接后数据类型为分类数据

        s1 = Series([np.nan, np.nan], dtype="category")
        # 创建一个包含 NaN 的分类数据类型的 Series 对象 s1

        s2 = Series([np.nan, np.nan], dtype="category")
        # 创建另一个包含 NaN 的分类数据类型的 Series 对象 s2

        exp = Series([np.nan, np.nan, np.nan, np.nan], dtype="category")
        # 创建预期的结果 Series 对象 exp，包含 s1 和 s2 拼接后的数据，并指定数据类型为分类数据

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        # 使用 pandas 的 concat 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp

        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
        # 使用 Series 对象的 _append 方法将 s1 和 s2 按行拼接，并忽略索引，验证结果是否等于 exp
    def test_concat_categorical_empty(self):
        # GH 13524
        # 测试空的分类 Series 合并

        s1 = Series([], dtype="category")
        # 创建一个空的分类 Series s1
        s2 = Series([1, 2], dtype="category")
        # 创建一个含有元素 [1, 2] 的分类 Series s2
        exp = s2.astype(object)
        # 将 s2 转换为 object 类型，作为期望结果
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        # 使用 pd.concat 将 s1 和 s2 合并，忽略索引，比较结果是否与 exp 相等
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
        # 使用 s1._append 方法将 s2 追加到 s1，忽略索引，比较结果是否与 exp 相等

        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        # 使用 pd.concat 将 s2 和 s1 合并，忽略索引，比较结果是否与 exp 相等
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)
        # 使用 s2._append 方法将 s1 追加到 s2，忽略索引，比较结果是否与 exp 相等

        s1 = Series([], dtype="category")
        # 创建一个空的分类 Series s1
        s2 = Series([], dtype="category")
        # 创建一个空的分类 Series s2

        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        # 使用 pd.concat 将 s1 和 s2 合并，忽略索引，比较结果是否与 s2 相等
        tm.assert_series_equal(s1._append(s2, ignore_index=True), s2)
        # 使用 s1._append 方法将 s2 追加到 s1，忽略索引，比较结果是否与 s2 相等

        s1 = Series([], dtype="category")
        # 创建一个空的分类 Series s1
        s2 = Series([], dtype="object")
        # 创建一个空的 object 类型 Series s2

        # different dtype => not-category
        # 不同的数据类型 => 非分类类型
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        # 使用 pd.concat 将 s1 和 s2 合并，忽略索引，比较结果是否与 s2 相等
        tm.assert_series_equal(s1._append(s2, ignore_index=True), s2)
        # 使用 s1._append 方法将 s2 追加到 s1，忽略索引，比较结果是否与 s2 相等
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), s2)
        # 使用 pd.concat 将 s2 和 s1 合并，忽略索引，比较结果是否与 s2 相等
        tm.assert_series_equal(s2._append(s1, ignore_index=True), s2)
        # 使用 s2._append 方法将 s1 追加到 s2，忽略索引，比较结果是否与 s2 相等

        s1 = Series([], dtype="category")
        # 创建一个空的分类 Series s1
        s2 = Series([np.nan, np.nan])
        # 创建一个包含 np.nan 的 object 类型 Series s2

        exp = Series([np.nan, np.nan], dtype=object)
        # 作为期望结果，创建一个包含 np.nan 的 object 类型 Series exp
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        # 使用 pd.concat 将 s1 和 s2 合并，忽略索引，比较结果是否与 exp 相等
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
        # 使用 s1._append 方法将 s2 追加到 s1，忽略索引，比较结果是否与 exp 相等

        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        # 使用 pd.concat 将 s2 和 s1 合并，忽略索引，比较结果是否与 exp 相等
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)
        # 使用 s2._append 方法将 s1 追加到 s2，忽略索引，比较结果是否与 exp 相等

    def test_categorical_concat_append(self):
        cat = Categorical(["a", "b"], categories=["a", "b"])
        # 创建一个分类变量 cat，包含值 ["a", "b"]，指定可能的分类为 ["a", "b"]
        vals = [1, 2]
        # 创建一个数值列表 vals，包含 [1, 2]
        df = DataFrame({"cats": cat, "vals": vals})
        # 创建一个 DataFrame df，包含列 'cats' 和 'vals'，对应 cat 和 vals
        cat2 = Categorical(["a", "b", "a", "b"], categories=["a", "b"])
        # 创建另一个分类变量 cat2，包含值 ["a", "b", "a", "b"]，指定可能的分类为 ["a", "b"]
        vals2 = [1, 2, 1, 2]
        # 创建一个数值列表 vals2，包含 [1, 2, 1, 2]
        exp = DataFrame({"cats": cat2, "vals": vals2}, index=Index([0, 1, 0, 1]))
        # 创建期望结果 DataFrame exp，包含列 'cats' 和 'vals'，对应 cat2 和 vals2，指定索引为 [0, 1, 0, 1]

        tm.assert_frame_equal(pd.concat([df, df]), exp)
        # 使用 pd.concat 将 df 和 df 合并，比较结果是否与 exp 相等
        tm.assert_frame_equal(df._append(df), exp)
        # 使用 df._append 方法将 df 追加到自身，比较结果是否与 exp 相等

        # GH 13524 can concat different categories
        # GH 13524 可以合并不同的分类变量

        cat3 = Categorical(["a", "b"], categories=["a", "b", "c"])
        # 创建一个分类变量 cat3，包含值 ["a", "b"]，指定可能的分类为 ["a", "b", "c"]
        vals3 = [1, 2]
        # 创建一个数值列表 vals3，包含 [1, 2]
        df_different_categories = DataFrame({"cats": cat3, "vals": vals3})
        # 创建一个 DataFrame df_different_categories，包含列 'cats' 和 'vals'，对应 cat3 和 vals3

        res = pd.concat([df, df_different_categories], ignore_index=True)
        # 使用 pd.concat 将 df 和 df_different_categories 合并，忽略索引
        exp = DataFrame({"cats": list("abab"), "vals": [1, 2, 1, 2]})
        # 创建期望结果 DataFrame exp，包含列 'cats' 和 'vals'，'cats' 为 "abab"，'vals' 为 [1, 2, 1, 2]
        tm.assert_frame_equal(res, exp)
        # 比较结果 res 是否与 exp 相等

        res = df._append(df_different_categories, ignore_index=True)
        # 使用 df._append 方法将 df_different_categories 追加到 df，忽略索引
        tm.assert_frame_equal(res, exp)
        # 比较结果 res 是否与 exp 相等
```