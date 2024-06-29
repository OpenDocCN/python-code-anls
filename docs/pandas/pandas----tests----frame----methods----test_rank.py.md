# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_rank.py`

```
# 从 datetime 模块导入 datetime 和 timedelta 类
from datetime import (
    datetime,
    timedelta,
)

# 导入 numpy 库，并使用 np 作为别名
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas._libs.algos 模块导入 Infinity 和 NegInfinity
from pandas._libs.algos import (
    Infinity,
    NegInfinity,
)

# 从 pandas 模块导入 DataFrame、Index 和 Series 类
from pandas import (
    DataFrame,
    Index,
    Series,
)

# 导入 pandas._testing 模块并使用 tm 作为别名
import pandas._testing as tm


# 定义一个测试类 TestRank
class TestRank:
    # 创建一个 Series 对象 s 和一个 DataFrame 对象 df，其中 DataFrame 包含 s 的两列 A 和 B
    s = Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])
    df = DataFrame({"A": s, "B": s})

    # 定义一个结果字典 results，包含不同排名方式的预期结果
    results = {
        "average": np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5]),
        "min": np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5]),
        "max": np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6]),
        "first": np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6]),
        "dense": np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3]),
    }

    # 定义一个测试方法 test_rank，接受参数 float_frame
    def test_rank(self, float_frame):
        # 导入 scipy.stats，并确保导入成功，否则跳过测试
        sp_stats = pytest.importorskip("scipy.stats")

        # 在 float_frame 的列 "A" 上每隔两行设置为 NaN
        float_frame.loc[::2, "A"] = np.nan
        # 在 float_frame 的列 "B" 上每隔三行设置为 NaN
        float_frame.loc[::3, "B"] = np.nan
        # 在 float_frame 的列 "C" 上每隔四行设置为 NaN
        float_frame.loc[::4, "C"] = np.nan
        # 在 float_frame 的列 "D" 上每隔五行设置为 NaN
        float_frame.loc[::5, "D"] = np.nan

        # 对 float_frame 进行默认排名，得到 ranks0
        ranks0 = float_frame.rank()
        # 对 float_frame 按行进行排名，得到 ranks1
        ranks1 = float_frame.rank(1)
        # 创建一个掩码，标记 float_frame 中的 NaN 值
        mask = np.isnan(float_frame.values)

        # 将 float_frame 中的 NaN 值用 np.inf 填充，并存储为 fvals
        fvals = float_frame.fillna(np.inf).values

        # 按列对 fvals 应用 scipy.stats.rankdata 函数，得到 exp0
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fvals)
        # 将掩码中的位置还原为 NaN
        exp0[mask] = np.nan

        # 按行对 fvals 应用 scipy.stats.rankdata 函数，得到 exp1
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fvals)
        # 将掩码中的位置还原为 NaN
        exp1[mask] = np.nan

        # 断言 ranks0 的值与 exp0 接近
        tm.assert_almost_equal(ranks0.values, exp0)
        # 断言 ranks1 的值与 exp1 接近
        tm.assert_almost_equal(ranks1.values, exp1)

        # 创建一个包含随机整数的 DataFrame df
        df = DataFrame(
            np.random.default_rng(2).integers(0, 5, size=40).reshape((10, 4))
        )

        # 对 df 进行默认排名，得到 result，并与 exp 进行比较
        result = df.rank()
        exp = df.astype(float).rank()
        tm.assert_frame_equal(result, exp)

        # 对 df 按行进行排名，得到 result，并与 exp 进行比较
        result = df.rank(1)
        exp = df.astype(float).rank(1)
        tm.assert_frame_equal(result, exp)
    # 测试在第二维度上对 DataFrame 进行排名，并将排名结果转换为百分比
    def test_rank2(self):
        # 创建一个 DataFrame 对象
        df = DataFrame([[1, 3, 2], [1, 2, 3]])
        # 创建一个期望的 DataFrame 对象，对原 DataFrame 进行排名并除以总数
        expected = DataFrame([[1.0, 3.0, 2.0], [1, 2, 3]]) / 3.0
        # 在第一维度上对 DataFrame 进行排名并转换为百分比
        result = df.rank(1, pct=True)
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

        # 以下类似的测试用例省略注释，均为对 DataFrame 进行排名操作并进行断言比较

    # 测试排名操作不会改变原 DataFrame
    def test_rank_does_not_mutate(self):
        # 创建一个随机数填充的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), dtype="float64"
        )
        # 复制一份原 DataFrame 作为期望结果
        expected = df.copy()
        # 对 DataFrame 进行排名操作
        df.rank()
        # 将排名后的 DataFrame 与原 DataFrame 进行比较
        result = df
        tm.assert_frame_equal(result, expected)

    # 测试混合类型的 DataFrame 对象进行排名操作
    def test_rank_mixed_frame(self, float_string_frame):
        # 向 DataFrame 中添加 datetime 和 timedelta 类型的数据
        float_string_frame["datetime"] = datetime.now()
        float_string_frame["timedelta"] = timedelta(days=1, seconds=1)

        # 对 DataFrame 进行排名操作，不限制只对数值类型进行排名
        float_string_frame.rank(numeric_only=False)
        # 使用 pytest 断言捕获 TypeError 异常
        with pytest.raises(TypeError, match="not supported between instances of"):
            # 在第一维度上对 DataFrame 进行排名操作
            float_string_frame.rank(axis=1)
    # 定义一个测试方法，用于测试带有缺失值的 DataFrame 的排名操作
    def test_rank_na_option(self, float_frame):
        # 导入 pytest 模块，如果未安装则跳过测试
        sp_stats = pytest.importorskip("scipy.stats")

        # 将每隔两行的'A'列设为 NaN
        float_frame.loc[::2, "A"] = np.nan
        # 将每隔三行的'B'列设为 NaN
        float_frame.loc[::3, "B"] = np.nan
        # 将每隔四行的'C'列设为 NaN
        float_frame.loc[::4, "C"] = np.nan
        # 将每隔五行的'D'列设为 NaN
        float_frame.loc[::5, "D"] = np.nan

        # 使用 'bottom' na_option 进行排名
        ranks0 = float_frame.rank(na_option="bottom")
        ranks1 = float_frame.rank(1, na_option="bottom")

        # 将 DataFrame 中的 NaN 值填充为无穷大，然后获取其值
        fvals = float_frame.fillna(np.inf).values

        # 期望的排名结果，沿着列应用 scipy.stats.rankdata 函数
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fvals)
        # 期望的排名结果，沿着行应用 scipy.stats.rankdata 函数
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fvals)

        # 使用测试工具库 tm 来比较实际排名和期望排名的近似性
        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # 使用 'top' na_option 进行排名
        ranks0 = float_frame.rank(na_option="top")
        ranks1 = float_frame.rank(1, na_option="top")

        # 将 DataFrame 中的 NaN 值填充为各列最小值减一的字典形式，然后获取其值
        fval0 = float_frame.fillna((float_frame.min() - 1).to_dict()).values
        # 转置 DataFrame，填充 NaN 值为各行最小值减一的字典形式，再转置，最后获取其值
        fval1 = float_frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values

        # 期望的排名结果，沿着列应用 scipy.stats.rankdata 函数
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fval0)
        # 期望的排名结果，沿着行应用 scipy.stats.rankdata 函数
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fval1)

        # 使用测试工具库 tm 来比较实际排名和期望排名的近似性
        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # 使用 'top' na_option 和降序进行排名
        ranks0 = float_frame.rank(na_option="top", ascending=False)
        ranks1 = float_frame.rank(1, na_option="top", ascending=False)

        # 将 DataFrame 中的 NaN 值填充为无穷大，然后获取其值
        fvals = float_frame.fillna(np.inf).values

        # 期望的排名结果，沿着列应用 scipy.stats.rankdata 函数，且结果取负值
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, -fvals)
        # 期望的排名结果，沿着行应用 scipy.stats.rankdata 函数，且结果取负值
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, -fvals)

        # 使用测试工具库 tm 来比较实际排名和期望排名的近似性
        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # 使用 'bottom' na_option 和降序进行排名
        ranks0 = float_frame.rank(na_option="bottom", ascending=False)
        ranks1 = float_frame.rank(1, na_option="bottom", ascending=False)

        # 将 DataFrame 中的 NaN 值填充为各列最小值减一的字典形式，然后获取其值
        fval0 = float_frame.fillna((float_frame.min() - 1).to_dict()).values
        # 转置 DataFrame，填充 NaN 值为各行最小值减一的字典形式，再转置，最后获取其值
        fval1 = float_frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values

        # 期望的排名结果，沿着列应用 scipy.stats.rankdata 函数，且结果取负值
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, -fval0)
        # 期望的排名结果，沿着行应用 scipy.stats.rankdata 函数，且结果取负值
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, -fval1)

        # 使用测试工具库 tm 来比较实际排名和期望排名的近似性
        tm.assert_numpy_array_equal(ranks0.values, exp0)
        tm.assert_numpy_array_equal(ranks1.values, exp1)

        # 当 na_option 参数值非法时，应抛出 ValueError 异常，匹配指定的错误消息
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"
        with pytest.raises(ValueError, match=msg):
            float_frame.rank(na_option="bad", ascending=False)

        # 当 na_option 参数类型非法时，应抛出 ValueError 异常，匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            float_frame.rank(na_option=True, ascending=False)

    # 定义一个测试方法，用于验证排名函数中使用轴名称和索引进行操作时的一致性
    def test_rank_axis(self):
        # 创建一个测试 DataFrame
        df = DataFrame([[2, 1], [4, 3]])

        # 使用 axis=0 和 axis="index" 进行排名，并比较结果是否相等
        tm.assert_frame_equal(df.rank(axis=0), df.rank(axis="index"))
        
        # 使用 axis=1 和 axis="columns" 进行排名，并比较结果是否相等
        tm.assert_frame_equal(df.rank(axis=1), df.rank(axis="columns"))
    # 使用 pytest 的参数化功能，定义了一个测试函数 test_rank_methods_frame，测试 DataFrame 的排名方法
    @pytest.mark.parametrize("ax", [0, 1])
    def test_rank_methods_frame(self, ax, rank_method):
        # 导入 scipy.stats 模块，如果不存在则跳过测试
        sp_stats = pytest.importorskip("scipy.stats")

        # 生成一个随机的 100x26 的整数数组，并归一化到 [-1, 1] 的范围内
        xs = np.random.default_rng(2).integers(0, 21, (100, 26))
        xs = (xs - 10.0) / 10.0
        # 创建列标签，从 'z' 到 'a'
        cols = [chr(ord("z") - i) for i in range(xs.shape[1])]

        # 对于三种不同的数值变换方式进行循环测试
        for vals in [xs, xs + 1e6, xs * 1e-6]:
            # 创建 DataFrame 对象 df，用随机生成的数据填充，列名为 cols
            df = DataFrame(vals, columns=cols)

            # 计算 DataFrame 的排名结果，按照给定的轴和排名方法
            result = df.rank(axis=ax, method=rank_method)
            # 使用 scipy.stats.rankdata 函数计算预期的排名结果
            sprank = np.apply_along_axis(
                sp_stats.rankdata,
                ax,
                vals,
                rank_method if rank_method != "first" else "ordinal",
            )
            sprank = sprank.astype(np.float64)
            # 创建预期的 DataFrame 结果，列名与 df 相同，数据类型为 float64
            expected = DataFrame(sprank, columns=cols).astype("float64")
            # 使用 pytest 的测试工具 tm 来比较结果 DataFrame 和预期 DataFrame 是否相等
            tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化功能，定义了一个测试函数 test_rank_descending，测试 DataFrame 的降序排名方法
    @pytest.mark.parametrize("dtype", ["O", "f8", "i8"])
    def test_rank_descending(self, rank_method, dtype):
        # 根据数据类型 dtype 对 self.df 进行处理，如果包含整数则删除 NaN 值并转换数据类型
        if "i" in dtype:
            df = self.df.dropna().astype(dtype)
        else:
            df = self.df.astype(dtype)

        # 计算 DataFrame df 的降序排名结果
        res = df.rank(ascending=False)
        # 计算预期的降序排名结果
        expected = (df.max() - df).rank()
        # 使用 pytest 的测试工具 tm 来比较结果 DataFrame 和预期 DataFrame 是否相等
        tm.assert_frame_equal(res, expected)

        # 使用指定的排名方法计算预期的降序排名结果
        expected = (df.max() - df).rank(method=rank_method)

        # 如果数据类型不是对象类型，则进行进一步的测试
        if dtype != "O":
            # 计算带有指定排名方法的降序排名结果，仅对数值列进行计算
            res2 = df.rank(method=rank_method, ascending=False, numeric_only=True)
            # 使用 pytest 的测试工具 tm 来比较结果 DataFrame 和预期 DataFrame 是否相等
            tm.assert_frame_equal(res2, expected)

        # 计算带有指定排名方法的降序排名结果，对所有列进行计算
        res3 = df.rank(method=rank_method, ascending=False, numeric_only=False)
        # 使用 pytest 的测试工具 tm 来比较结果 DataFrame 和预期 DataFrame 是否相等
        tm.assert_frame_equal(res3, expected)

    # 使用 pytest 的参数化功能，定义了一个测试函数 test_rank_2d_tie_methods，测试二维 DataFrame 的排名方法和并列值处理
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("dtype", [None, object])
    def test_rank_2d_tie_methods(self, rank_method, axis, dtype):
        # 使用已有的 DataFrame self.df 进行测试
        df = self.df

        # 定义内部函数 _check2d，用于验证二维 DataFrame 的排名结果是否符合预期
        def _check2d(df, expected, method="average", axis=0):
            # 创建预期的 DataFrame 结果，包含列 A 和 B，数据从 expected 中获取
            exp_df = DataFrame({"A": expected, "B": expected})

            # 如果 axis 为 1，则转置 DataFrame 进行排名计算
            if axis == 1:
                df = df.T
                exp_df = exp_df.T

            # 计算 DataFrame df 的排名结果，按照指定的排名方法和轴
            result = df.rank(method=rank_method, axis=axis)
            # 使用 pytest 的测试工具 tm 来比较结果 DataFrame 和预期 DataFrame 是否相等
            tm.assert_frame_equal(result, exp_df)

        # 根据 dtype 是否为 None 来选择 DataFrame frame
        frame = df if dtype is None else df.astype(dtype)
        # 使用内部函数 _check2d 进行排名测试，预期结果从 self.results[rank_method] 中获取
        _check2d(frame, self.results[rank_method], method=rank_method, axis=axis)
    @pytest.mark.parametrize(
        "rank_method,exp",
        [   # 使用 pytest 的参数化装饰器，定义不同的排名方法和对应的预期结果列表
            ("dense", [[1.0, 1.0, 1.0], [1.0, 0.5, 2.0 / 3], [1.0, 0.5, 1.0 / 3]]),
            (
                "min",
                [
                    [1.0 / 3, 1.0, 1.0],
                    [1.0 / 3, 1.0 / 3, 2.0 / 3],
                    [1.0 / 3, 1.0 / 3, 1.0 / 3],
                ],
            ),
            (
                "max",
                [[1.0, 1.0, 1.0], [1.0, 2.0 / 3, 2.0 / 3], [1.0, 2.0 / 3, 1.0 / 3]],
            ),
            (
                "average",
                [[2.0 / 3, 1.0, 1.0], [2.0 / 3, 0.5, 2.0 / 3], [2.0 / 3, 0.5, 1.0 / 3]],
            ),
            (
                "first",
                [
                    [1.0 / 3, 1.0, 1.0],
                    [2.0 / 3, 1.0 / 3, 2.0 / 3],
                    [3.0 / 3, 2.0 / 3, 1.0 / 3],
                ],
            ),
        ],
    )
    def test_rank_pct_true(self, rank_method, exp):
        # 用于测试 DataFrame 的 rank 方法，在每种排名方法下验证其百分位数计算是否正确
        # see gh-15630.  # 关联 GitHub 问题编号

        df = DataFrame([[2012, 66, 3], [2012, 65, 2], [2012, 65, 1]])  # 创建包含年份和数据的 DataFrame
        result = df.rank(method=rank_method, pct=True)  # 对 DataFrame 应用排名方法并计算百分位数

        expected = DataFrame(exp)  # 创建预期结果的 DataFrame
        tm.assert_frame_equal(result, expected)  # 使用 pytest 的测试辅助工具验证结果与预期是否一致

    @pytest.mark.single_cpu
    def test_pct_max_many_rows(self):
        # 用于测试在大量行的情况下，DataFrame 的 rank 方法是否正确计算最大百分位数
        # GH 18271  # 关联 GitHub 问题编号
        df = DataFrame({"A": np.arange(2**24 + 1), "B": np.arange(2**24 + 1, 0, -1)})  # 创建具有大量数据的 DataFrame
        result = df.rank(pct=True).max()  # 对 DataFrame 应用排名方法并计算最大百分位数
        assert (result == 1).all()  # 断言最大百分位数是否全部为 1
    @pytest.mark.parametrize(
        "contents,dtype",
        [
            (
                [
                    -np.inf,
                    -50,
                    -1,
                    -1e-20,
                    -1e-25,
                    -1e-50,
                    0,
                    1e-40,
                    1e-20,
                    1e-10,
                    2,
                    40,
                    np.inf,
                ],
                "float64",
            ),
            (
                [
                    -np.inf,
                    -50,
                    -1,
                    -1e-20,
                    -1e-25,
                    -1e-45,
                    0,
                    1e-40,
                    1e-20,
                    1e-10,
                    2,
                    40,
                    np.inf,
                ],
                "float32",
            ),
            ([np.iinfo(np.uint8).min, 1, 2, 100, np.iinfo(np.uint8).max], "uint8"),
            (
                [
                    np.iinfo(np.int64).min,
                    -100,
                    0,
                    1,
                    9999,
                    100000,
                    1e10,
                    np.iinfo(np.int64).max,
                ],
                "int64",
            ),
            ([NegInfinity(), "1", "A", "BA", "Ba", "C", Infinity()], "object"),
            (
                [datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 5)],
                "datetime64",
            ),
        ],
    )
    # 使用 pytest 的 @pytest.mark.parametrize 装饰器标记这个测试函数，提供多组参数化输入
    def test_rank_inf_and_nan(self, contents, dtype, frame_or_series):
        # 定义 dtype 到缺失值映射关系的字典
        dtype_na_map = {
            "float64": np.nan,
            "float32": np.nan,
            "object": None,
            "datetime64": np.datetime64("nat"),
        }
        # 如果 dtype 存在于 dtype_na_map 中，则插入 NaN 值到随机位置
        # 然后调整期望的排序顺序，相应地添加 NaN 值，用于测试排序计算是否受 NaN 值影响
        values = np.array(contents, dtype=dtype)
        exp_order = np.array(range(len(values)), dtype="float64") + 1.0
        if dtype in dtype_na_map:
            na_value = dtype_na_map[dtype]
            nan_indices = np.random.default_rng(2).choice(range(len(values)), 5)
            values = np.insert(values, nan_indices, na_value)
            exp_order = np.insert(exp_order, nan_indices, np.nan)

        # 将测试数组和期望结果以相同的方式随机打乱顺序
        random_order = np.random.default_rng(2).permutation(len(values))
        # 使用 frame_or_series 将打乱顺序后的 values 构造为对象
        obj = frame_or_series(values[random_order])
        # 构造期望的 frame_or_series 对象，数据类型为 float64
        expected = frame_or_series(exp_order[random_order], dtype="float64")
        # 对 obj 调用 rank() 方法，得到结果
        result = obj.rank()
        # 使用 pytest 的 tm.assert_equal() 断言 result 和 expected 相等
        tm.assert_equal(result, expected)
    def test_df_series_inf_nan_consistency(self):
        # 定义测试函数，检查DataFrame和Series中包含inf和nan的一致性
        # GH#32593

        # 创建DataFrame的索引
        index = [5, 4, 3, 2, 1, 6, 7, 8, 9, 10]
        
        # 创建两列数据，其中col1没有缺失值，col2包含np.nan和np.inf
        col1 = [5, 4, 3, 5, 8, 5, 2, 1, 6, 6]
        col2 = [5, 4, np.nan, 5, 8, 5, np.inf, np.nan, 6, -np.inf]
        
        # 使用数据创建DataFrame对象df
        df = DataFrame(
            data={
                "col1": col1,
                "col2": col2,
            },
            index=index,
            dtype="f8",
        )
        
        # 使用DataFrame的rank()方法得到排名后的结果df_result
        df_result = df.rank()

        # 创建一个df的副本series_result，用于存储处理后的结果
        series_result = df.copy()
        
        # 对col1列进行rank()处理，更新到series_result中
        series_result["col1"] = df["col1"].rank()
        
        # 对col2列进行rank()处理，更新到series_result中
        series_result["col2"] = df["col2"].rank()
        
        # 使用assert_frame_equal比较df_result和series_result，确认它们相等
        tm.assert_frame_equal(df_result, series_result)

    def test_rank_both_inf(self):
        # 定义测试函数，检查包含正负无穷大的DataFrame排名处理
        # GH#32593

        # 创建一个包含正负无穷大的DataFrame对象df
        df = DataFrame({"a": [-np.inf, 0, np.inf]})
        
        # 创建期望的DataFrame对象expected，其中列a包含按顺序排名的结果
        expected = DataFrame({"a": [1.0, 2.0, 3.0]})
        
        # 使用DataFrame的rank()方法进行排名处理，结果存储在result中
        result = df.rank()
        
        # 使用assert_frame_equal比较result和expected，确认它们相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "na_option,ascending,expected",
        [
            ("top", True, [3.0, 1.0, 2.0]),
            ("top", False, [2.0, 1.0, 3.0]),
            ("bottom", True, [2.0, 3.0, 1.0]),
            ("bottom", False, [1.0, 3.0, 2.0]),
        ],
    )
    def test_rank_inf_nans_na_option(
        self, frame_or_series, rank_method, na_option, ascending, expected
    ):
        # 使用参数化测试方式，测试处理包含正负无穷大及NaN值的DataFrame或Series对象的排名
        obj = frame_or_series([np.inf, np.nan, -np.inf])
        
        # 使用rank()方法对obj进行排名处理，传入方法rank_method、na_option、ascending作为参数
        result = obj.rank(method=rank_method, na_option=na_option, ascending=ascending)
        
        # 创建期望的结果对象expected，作为对比
        expected = frame_or_series(expected)
        
        # 使用tm.assert_equal比较result和expected，确认它们相等
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "na_option,ascending,expected",
        [
            ("bottom", True, [1.0, 2.0, 4.0, 3.0]),
            ("bottom", False, [1.0, 2.0, 4.0, 3.0]),
            ("top", True, [2.0, 3.0, 1.0, 4.0]),
            ("top", False, [2.0, 3.0, 1.0, 4.0]),
        ],
    )
    def test_rank_object_first(
        self, frame_or_series, na_option, ascending, expected, using_infer_string
    ):
        # 使用参数化测试方式，测试处理包含对象及正负无穷大的DataFrame或Series对象的排名
        obj = frame_or_series(["foo", "foo", None, "foo"])
        
        # 使用rank()方法对obj进行排名处理，传入方法"first"、na_option、ascending作为参数
        result = obj.rank(method="first", na_option=na_option, ascending=ascending)
        
        # 创建期望的结果对象expected，作为对比
        expected = frame_or_series(expected)
        
        # 如果使用了infer_string并且obj是Series对象，则将expected转换为uint64类型
        if using_infer_string and isinstance(obj, Series):
            expected = expected.astype("uint64")
        
        # 使用tm.assert_equal比较result和expected，确认它们相等
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                {"a": [1, 2, "a"], "b": [4, 5, 6]},
                DataFrame({"b": [1.0, 2.0, 3.0]}, columns=Index(["b"], dtype=object)),
            ),
            ({"a": [1, 2, "a"]}, DataFrame(index=range(3), columns=[])),
        ],
    )
    def test_rank_mixed_axis_zero(self, data, expected):
        # 定义测试函数，检查在axis=0上处理包含混合数据类型的DataFrame的排名
        
        # 根据输入数据创建DataFrame对象df
        df = DataFrame(data, columns=Index(list(data.keys()), dtype=object))
        
        # 使用pytest.raises检测是否抛出TypeError异常，匹配指定的错误信息
        with pytest.raises(TypeError, match="'<' not supported between instances of"):
            df.rank()
        
        # 使用DataFrame的rank()方法对numeric_only=True进行排名处理，结果存储在result中
        result = df.rank(numeric_only=True)
        
        # 使用tm.assert_frame_equal比较result和expected，确认它们相等
        tm.assert_frame_equal(result, expected)
    @pytest.mark.parametrize(
        "dtype, exp_dtype",
        [("string[pyarrow]", "Int64"), ("string[pyarrow_numpy]", "float64")],
    )
    # 定义测试方法，用于验证不同数据类型的序列排名功能
    def test_rank_string_dtype(self, dtype, exp_dtype):
        # GH#55362
        # 导入pyarrow库，如果库不可用则跳过测试
        pytest.importorskip("pyarrow")
        # 创建一个包含字符串和空值的Series对象，使用指定的数据类型
        obj = Series(["foo", "foo", None, "foo"], dtype=dtype)
        # 对Series对象执行排名操作，指定使用"first"方法
        result = obj.rank(method="first")
        # 创建期望的排名结果Series对象，使用预期的数据类型
        expected = Series([1, 2, None, 3], dtype=exp_dtype)
        # 使用测试框架的函数验证实际结果与期望结果是否相等
        tm.assert_series_equal(result, expected)
```