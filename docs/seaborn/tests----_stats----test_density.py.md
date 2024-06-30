# `D:\src\scipysrc\seaborn\tests\_stats\test_density.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理和分析

import pytest  # 导入pytest库，用于编写和运行测试
from numpy.testing import assert_array_equal, assert_array_almost_equal  # 导入NumPy测试工具函数

from seaborn._core.groupby import GroupBy  # 导入Seaborn的GroupBy类
from seaborn._stats.density import KDE, _no_scipy  # 导入Seaborn的KDE类和_no_scipy函数
from seaborn._compat import groupby_apply_include_groups  # 导入Seaborn的兼容性函数

class TestKDE:

    @pytest.fixture
    def df(self, rng):
        # 定义一个测试数据生成的fixture函数
        n = 100
        return pd.DataFrame(dict(
            x=rng.uniform(0, 7, n).round(),  # 生成在[0, 7)范围内均匀分布的随机数，并四舍五入
            y=rng.normal(size=n),  # 生成服从标准正态分布的随机数
            color=rng.choice(["a", "b", "c"], n),  # 从["a", "b", "c"]中随机选择n个元素作为颜色数据
            alpha=rng.choice(["x", "y"], n),  # 从["x", "y"]中随机选择n个元素作为alpha数据
        ))

    def get_groupby(self, df, orient):
        # 定义一个根据指定方向orient进行分组的函数
        cols = [c for c in df if c != orient]  # 获取除去orient列之外的所有列名
        return GroupBy([*cols, "group"])  # 返回一个GroupBy对象，对指定列进行分组

    def integrate(self, y, x):
        # 定义一个数值积分函数
        y = np.asarray(y)  # 将输入的y转换为NumPy数组
        x = np.asarray(x)  # 将输入的x转换为NumPy数组
        dx = np.diff(x)  # 计算x的差分
        return (dx * y[:-1] + dx * y[1:]).sum() / 2  # 计算积分近似值并返回

    @pytest.mark.parametrize("ori", ["x", "y"])
    def test_columns(self, df, ori):
        # 测试函数：测试KDE生成的DataFrame的列是否符合预期

        df = df[[ori, "alpha"]]  # 选择df的ori列和alpha列作为测试数据
        gb = self.get_groupby(df, ori)  # 调用get_groupby方法生成GroupBy对象
        res = KDE()(df, gb, ori, {})  # 调用KDE类进行计算
        other = {"x": "y", "y": "x"}[ori]  # 根据ori确定other列的名称
        expected = [ori, "alpha", "density", other]  # 期望的结果列名列表
        assert list(res.columns) == expected  # 断言KDE计算结果的列名是否与期望一致

    @pytest.mark.parametrize("gridsize", [20, 30, None])
    def test_gridsize(self, df, gridsize):
        # 测试函数：测试KDE生成的DataFrame的长度是否符合预期

        ori = "y"  # 指定测试的列为"y"
        df = df[[ori]]  # 选择df的ori列作为测试数据
        gb = self.get_groupby(df, ori)  # 调用get_groupby方法生成GroupBy对象
        res = KDE(gridsize=gridsize)(df, gb, ori, {})  # 调用KDE类进行计算
        if gridsize is None:
            assert_array_equal(res[ori], df[ori])  # 断言当gridsize为None时，KDE计算的结果与输入数据一致
        else:
            assert len(res) == gridsize  # 断言KDE计算的结果长度与gridsize一致

    @pytest.mark.parametrize("cut", [1, 2])
    def test_cut(self, df, cut):
        # 测试函数：测试KDE计算结果的边界是否符合预期

        ori = "y"  # 指定测试的列为"y"
        df = df[[ori]]  # 选择df的ori列作为测试数据
        gb = self.get_groupby(df, ori)  # 调用get_groupby方法生成GroupBy对象
        res = KDE(cut=cut, bw_method=1)(df, gb, ori, {})  # 调用KDE类进行计算

        vals = df[ori]  # 获取输入数据的ori列
        bw = vals.std()  # 计算ori列数据的标准差作为带宽
        assert res[ori].min() == pytest.approx(vals.min() - bw * cut, abs=1e-2)  # 断言KDE计算结果的最小值是否符合预期
        assert res[ori].max() == pytest.approx(vals.max() + bw * cut, abs=1e-2)  # 断言KDE计算结果的最大值是否符合预期

    @pytest.mark.parametrize("common_grid", [True, False])
    def test_common_grid(self, df, common_grid):
        # 测试函数：测试KDE计算结果在不同条件下是否一致

        ori = "y"  # 指定测试的列为"y"
        df = df[[ori, "alpha"]]  # 选择df的ori列和alpha列作为测试数据
        gb = self.get_groupby(df, ori)  # 调用get_groupby方法生成GroupBy对象
        res = KDE(common_grid=common_grid)(df, gb, ori, {})  # 调用KDE类进行计算

        vals = df["alpha"].unique()  # 获取alpha列的唯一值
        a = res.loc[res["alpha"] == vals[0], ori].to_numpy()  # 获取alpha列为vals[0]时的ori列数据
        b = res.loc[res["alpha"] == vals[1], ori].to_numpy()  # 获取alpha列为vals[1]时的ori列数据
        if common_grid:
            assert_array_equal(a, b)  # 断言在使用共同网格时，不同alpha值对应的结果是否相等
        else:
            assert np.not_equal(a, b).all()  # 断言在不使用共同网格时，不同alpha值对应的结果是否不全相等
    # 定义测试函数，用于测试 KDE 类的普通情况
    def test_common_norm(self, df, common_norm):
        # 设定原始数据列名为 "y"
        ori = "y"
        # 从数据框中选择 "y" 和 "alpha" 列，并重新赋值给 df
        df = df[[ori, "alpha"]]
        # 调用 self.get_groupby 方法，按照 "y" 列对数据框进行分组
        gb = self.get_groupby(df, ori)
        # 使用 KDE 类的实例，使用指定的 common_norm 参数进行核密度估计
        res = KDE(common_norm=common_norm)(df, gb, ori, {})

        # 对结果进行分组，计算每个组的面积
        areas = (
            res.groupby("alpha")
            .apply(
                lambda x: self.integrate(x["density"], x[ori]),
                **groupby_apply_include_groups(False),
            )
        )

        # 如果 common_norm 为 True，则断言总面积接近于 1
        if common_norm:
            assert areas.sum() == pytest.approx(1, abs=1e-3)
        # 如果 common_norm 为 False，则断言各组的面积几乎等于 [1, 1]
        else:
            assert_array_almost_equal(areas, [1, 1], decimal=3)

    # 定义测试函数，用于测试 KDE 类在指定变量条件下的情况
    def test_common_norm_variables(self, df):
        # 设定原始数据列名为 "y"
        ori = "y"
        # 从数据框中选择 "y", "alpha" 和 "color" 列，并重新赋值给 df
        df = df[[ori, "alpha", "color"]]
        # 调用 self.get_groupby 方法，按照 "y" 列对数据框进行分组
        gb = self.get_groupby(df, ori)
        # 使用 KDE 类的实例，使用 ["alpha"] 作为 common_norm 参数进行核密度估计
        res = KDE(common_norm=["alpha"])(df, gb, ori, {})

        # 定义按颜色分组并计算面积的函数
        def integrate_by_color_and_sum(x):
            return (
                x.groupby("color")
                .apply(
                    lambda y: self.integrate(y["density"], y[ori]),
                    **groupby_apply_include_groups(False)
                )
                .sum()
            )

        # 对结果进行分组，按 "alpha" 列分组并应用上面定义的函数
        areas = (
            res
            .groupby("alpha")
            .apply(integrate_by_color_and_sum, **groupby_apply_include_groups(False))
        )
        # 断言各组的面积几乎等于 [1, 1]
        assert_array_almost_equal(areas, [1, 1], decimal=3)

    # 使用 pytest.mark.parametrize 参数化测试，测试 KDE 类的输入检查
    @pytest.mark.parametrize("param", ["norm", "grid"])
    def test_common_input_checks(self, df, param):
        # 设定原始数据列名为 "y"
        ori = "y"
        # 从数据框中选择 "y" 和 "alpha" 列，并重新赋值给 df
        df = df[[ori, "alpha"]]
        # 调用 self.get_groupby 方法，按照 "y" 列对数据框进行分组
        gb = self.get_groupby(df, ori)
        # 构造预期的警告消息，检查是否会触发警告
        msg = rf"Undefined variable\(s\) passed for KDE.common_{param}"
        with pytest.warns(UserWarning, match=msg):
            # 使用指定的 common_param 参数构造 KDE 类的实例，并调用进行核密度估计
            KDE(**{f"common_{param}": ["color", "alpha"]})(df, gb, ori, {})

        # 构造预期的错误消息，检查是否会抛出异常
        msg = f"KDE.common_{param} must be a boolean or list of strings"
        with pytest.raises(TypeError, match=msg):
            # 使用错误的 common_param 参数构造 KDE 类的实例，并调用进行核密度估计
            KDE(**{f"common_{param}": "alpha"})(df, gb, ori, {})

    # 定义测试函数，用于测试 KDE 类的带宽调整功能
    def test_bw_adjust(self, df):
        # 设定原始数据列名为 "y"
        ori = "y"
        # 从数据框中选择 "y" 列，并重新赋值给 df
        df = df[[ori]]
        # 调用 self.get_groupby 方法，按照 "y" 列对数据框进行分组
        gb = self.get_groupby(df, ori)
        # 使用带指定带宽调整系数的 KDE 类的实例进行核密度估计
        res1 = KDE(bw_adjust=0.5)(df, gb, ori, {})
        res2 = KDE(bw_adjust=2.0)(df, gb, ori, {})

        # 计算密度估计结果的绝对值差的均值
        mad1 = res1["density"].diff().abs().mean()
        mad2 = res2["density"].diff().abs().mean()
        # 断言带宽调整系数较小的结果的差异均值大于带宽调整系数较大的结果的差异均值
        assert mad1 > mad2

    # 定义测试函数，用于测试 KDE 类的带宽方法（标量形式）功能
    def test_bw_method_scalar(self, df):
        # 设定原始数据列名为 "y"
        ori = "y"
        # 从数据框中选择 "y" 列，并重新赋值给 df
        df = df[[ori]]
        # 调用 self.get_groupby 方法，按照 "y" 列对数据框进行分组
        gb = self.get_groupby(df, ori)
        # 使用指定带宽方法的 KDE 类的实例进行核密度估计
        res1 = KDE(bw_method=0.5)(df, gb, ori, {})
        res2 = KDE(bw_method=2.0)(df, gb, ori, {})

        # 计算密度估计结果的绝对值差的均值
        mad1 = res1["density"].diff().abs().mean()
        mad2 = res2["density"].diff().abs().mean()
        # 断言带宽方法参数较小的结果的差异均值大于带宽方法参数较大的结果的差异均值
        assert mad1 > mad2

    # 使用 pytest.mark.skipif 根据条件跳过测试，测试 KDE 类的累积功能
    @pytest.mark.skipif(_no_scipy, reason="KDE.cumulative requires scipy")
    @pytest.mark.parametrize("common_norm", [True, False])
    # 测试累积分布函数（KDE）的功能，对给定数据框 df 和通用归一化参数 common_norm 进行测试
    def test_cumulative(self, df, common_norm):
    
        # 设置原始列名为 "y"
        ori = "y"
        # 从数据框中选择 "y" 和 "alpha" 列组成新的数据框 df
        df = df[[ori, "alpha"]]
        # 调用 self.get_groupby 方法，按照 "y" 列进行分组
        gb = self.get_groupby(df, ori)
        # 调用带有累积参数 cumulative=True 和通用归一化参数 common_norm 的 KDE 类，并对数据进行评估
        res = KDE(cumulative=True, common_norm=common_norm)(df, gb, ori, {})
    
        # 遍历结果 res，按照 "alpha" 列进行分组
        for _, group_res in res.groupby("alpha"):
            # 断言每个分组的密度函数的差分不小于零
            assert (group_res["density"].diff().dropna() >= 0).all()
            # 如果 common_norm 为 False，则断言每个分组的密度函数最大值近似为 1，精度为 1e-3
            if not common_norm:
                assert group_res["density"].max() == pytest.approx(1, abs=1e-3)
    
    # 测试累积分布函数（KDE）在缺少 scipy 模块时的行为
    def test_cumulative_requires_scipy(self):
    
        # 如果没有安装 scipy 模块
        if _no_scipy:
            # 抛出 RuntimeError 异常，异常信息为 "Cumulative KDE evaluation requires scipy"
            err = "Cumulative KDE evaluation requires scipy"
            with pytest.raises(RuntimeError, match=err):
                # 调用带有 cumulative=True 参数的 KDE 类
                KDE(cumulative=True)
    
    # 测试在给定数据 df 和 vals 列表的情况下，KDE 的行为
    @pytest.mark.parametrize("vals", [[], [1], [1] * 5, [1929245168.06679] * 18])
    def test_singular(self, df, vals):
    
        # 创建新的数据框 df1，包含 "y" 列和 "alpha" 列
        df1 = pd.DataFrame({"y": vals, "alpha": ["z"] * len(vals)})
        # 调用 self.get_groupby 方法，按照 "y" 列进行分组
        gb = self.get_groupby(df1, "y")
        # 调用不带参数的 KDE 类，并对数据进行评估
        res = KDE()(df1, gb, "y", {})
        # 断言结果 res 为空
    
        # 将 df1 和原始数据 df[["y", "alpha"]] 合并为新的数据框 df2
        df2 = pd.concat([df[["y", "alpha"]], df1], ignore_index=True)
        # 再次调用 self.get_groupby 方法，按照 "y" 列进行分组
        gb = self.get_groupby(df2, "y")
        # 再次调用不带参数的 KDE 类，并对数据进行评估
        res = KDE()(df2, gb, "y", {})
        # 断言结果 res 的 "alpha" 列集合与原始数据 df 的 "alpha" 列集合相同
    
    # 测试在给定数据 df 和列 col 的情况下，KDE 的行为
    @pytest.mark.parametrize("col", ["y", "weight"])
    def test_missing(self, df, col):
    
        # 设置变量 val 和 ori 的值为 "xy"
        val, ori = "xy"
        # 将数据框 df 中的 "weight" 列设置为 1
        df["weight"] = 1
        # 从数据框中选择 ori 列和 "weight" 列，并将其转换为浮点数
        df = df[[ori, "weight"]].astype(float)
        # 将数据框中前 5 行的 col 列设置为 NaN
        df.loc[:4, col] = np.nan
        # 调用 self.get_groupby 方法，按照 ori 列进行分组
        gb = self.get_groupby(df, ori)
        # 调用不带参数的 KDE 类，并对数据进行评估
        res = KDE()(df, gb, ori, {})
        # 断言对结果 res 中的 val 列和 ori 列的积分近似为 1，精度为 1e-3
        assert self.integrate(res[val], res[ori]) == pytest.approx(1, abs=1e-3)
```