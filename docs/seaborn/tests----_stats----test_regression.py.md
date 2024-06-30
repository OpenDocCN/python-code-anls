# `D:\src\scipysrc\seaborn\tests\_stats\test_regression.py`

```
# 导入必要的库
import numpy as np
import pandas as pd

# 导入测试相关的库和函数
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal

# 导入 seaborn 库中的 GroupBy 和 PolyFit 类
from seaborn._core.groupby import GroupBy
from seaborn._stats.regression import PolyFit

# 定义测试类 TestPolyFit
class TestPolyFit:

    # 定义 pytest 的 fixture，用于生成测试数据
    @pytest.fixture
    def df(self, rng):
        n = 100
        return pd.DataFrame(dict(
            x=rng.normal(0, 1, n),
            y=rng.normal(0, 1, n),
            color=rng.choice(["a", "b", "c"], n),
            group=rng.choice(["x", "y"], n),
        ))

    # 测试函数：当没有分组条件时
    def test_no_grouper(self, df):
        # 创建 GroupBy 对象，根据 "group" 列分组
        groupby = GroupBy(["group"])
        # 调用 PolyFit 对象，拟合二维数据 df[["x", "y"]]，以 "x" 列作为自变量，空字典作为参数
        res = PolyFit(order=1, gridsize=100)(df[["x", "y"]], groupby, "x", {})

        # 断言返回结果的列名应为 ["x", "y"]
        assert_array_equal(res.columns, ["x", "y"])

        # 生成等间距的网格，用于检查结果中的 "x" 列
        grid = np.linspace(df["x"].min(), df["x"].max(), 100)
        assert_array_equal(res["x"], grid)

        # 断言返回结果中 "y" 列的差分两次后的结果应为全零（用于检查拟合效果）
        assert_array_almost_equal(
            res["y"].diff().diff().dropna(), np.zeros(grid.size - 2)
        )

    # 测试函数：当只有一个分组条件时
    def test_one_grouper(self, df):
        # 创建 GroupBy 对象，根据 "group" 列分组
        groupby = GroupBy(["group"])
        gridsize = 50
        # 调用 PolyFit 对象，拟合数据集 df，以 "x" 列作为自变量，空字典作为参数
        res = PolyFit(gridsize=gridsize)(df, groupby, "x", {})

        # 断言返回结果的列名应为 ["x", "y", "group"]
        assert res.columns.to_list() == ["x", "y", "group"]

        # 计算分组数目
        ngroups = df["group"].nunique()
        # 断言返回结果的索引应为长度为 ngroups * gridsize 的连续整数数组
        assert_array_equal(res.index, np.arange(ngroups * gridsize))

        # 对每个分组进行检查
        for _, part in res.groupby("group"):
            # 生成等间距的网格，用于检查每个分组中的 "x" 列
            grid = np.linspace(part["x"].min(), part["x"].max(), gridsize)
            assert_array_equal(part["x"], grid)
            # 断言每个分组中 "y" 列经过两次差分后的结果都大于零（用于检查拟合效果）
            assert part["y"].diff().diff().dropna().abs().gt(0).all()

    # 测试函数：处理缺失数据的情况
    def test_missing_data(self, df):
        # 创建 GroupBy 对象，根据 "group" 列分组
        groupby = GroupBy(["group"])
        # 在数据集的第 5 至 9 行设置缺失值
        df.iloc[5:10] = np.nan
        # 分别用缺失值处理前后的数据集调用 PolyFit 对象，以 "x" 列作为自变量，空字典作为参数
        res1 = PolyFit()(df[["x", "y"]], groupby, "x", {})
        res2 = PolyFit()(df[["x", "y"]].dropna(), groupby, "x", {})
        # 断言两次调用的结果应该一致
        assert_frame_equal(res1, res2)
```