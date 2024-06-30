# `D:\src\scipysrc\seaborn\tests\_stats\test_order.py`

```
# 导入必要的库
import numpy as np
import pandas as pd

# 导入 pytest 库和需要的测试函数
import pytest
from numpy.testing import assert_array_equal

# 导入 seaborn 库中的特定模块和函数
from seaborn._core.groupby import GroupBy
from seaborn._stats.order import Perc
from seaborn.utils import _version_predates


# 定义一个 Fixtures 类，用于生成测试数据
class Fixtures:

    @pytest.fixture
    def df(self, rng):
        # 创建一个包含随机数据的 DataFrame
        return pd.DataFrame(dict(x="", y=rng.normal(size=30)))

    def get_groupby(self, df, orient):
        # 根据 orient 参数选择性地组织列
        other = {"x": "y", "y": "x"}[orient]
        cols = [c for c in df if c != other]
        # 返回 GroupBy 对象，用于进一步的分组操作
        return GroupBy(cols)


# 定义 TestPerc 类，用于测试 Perc 类的不同方法
class TestPerc(Fixtures):

    def test_int_k(self, df):
        # 测试基于整数 k 的 Perc 类方法
        ori = "x"
        gb = self.get_groupby(df, ori)
        res = Perc(3)(df, gb, ori, {})
        percentiles = [0, 50, 100]
        # 检查结果中的百分位数列是否与预期相同
        assert_array_equal(res["percentile"], percentiles)
        # 检查计算出的百分位数列是否与预期相同
        assert_array_equal(res["y"], np.percentile(df["y"], percentiles))

    def test_list_k(self, df):
        # 测试基于列表 k 的 Perc 类方法
        ori = "x"
        gb = self.get_groupby(df, ori)
        percentiles = [0, 20, 100]
        res = Perc(k=percentiles)(df, gb, ori, {})
        # 检查结果中的百分位数列是否与预期相同
        assert_array_equal(res["percentile"], percentiles)
        # 检查计算出的百分位数列是否与预期相同
        assert_array_equal(res["y"], np.percentile(df["y"], percentiles))

    def test_orientation(self, df):
        # 测试不同列名方向的 Perc 类方法
        df = df.rename(columns={"x": "y", "y": "x"})
        ori = "y"
        gb = self.get_groupby(df, ori)
        res = Perc(k=3)(df, gb, ori, {})
        # 检查结果中的百分位数列是否与预期相同
        assert_array_equal(res["x"], np.percentile(df["x"], [0, 50, 100]))

    def test_method(self, df):
        # 测试 Perc 类方法中不同的插值方法
        ori = "x"
        gb = self.get_groupby(df, ori)
        method = "nearest"
        res = Perc(k=5, method=method)(df, gb, ori, {})
        percentiles = [0, 25, 50, 75, 100]
        # 根据 numpy 版本选择相应的百分位数计算方法
        if _version_predates(np, "1.22.0"):
            expected = np.percentile(df["y"], percentiles, interpolation=method)
        else:
            expected = np.percentile(df["y"], percentiles, method=method)
        # 检查计算出的百分位数列是否与预期相同
        assert_array_equal(res["y"], expected)

    def test_grouped(self, df, rng):
        # 测试分组数据上的 Perc 类方法
        ori = "x"
        df = df.assign(x=rng.choice(["a", "b", "c"], len(df)))
        gb = self.get_groupby(df, ori)
        k = [10, 90]
        res = Perc(k)(df, gb, ori, {})
        # 遍历每个分组，检查每个分组的百分位数列是否与预期相同
        for x, res_x in res.groupby("x"):
            assert_array_equal(res_x["percentile"], k)
            expected = np.percentile(df.loc[df["x"] == x, "y"], k)
            # 检查计算出的每个分组的百分位数列是否与预期相同
            assert_array_equal(res_x["y"], expected)

    def test_with_na(self, df):
        # 测试包含缺失值的 Perc 类方法
        ori = "x"
        df.loc[:5, "y"] = np.nan
        gb = self.get_groupby(df, ori)
        k = [10, 90]
        res = Perc(k)(df, gb, ori, {})
        # 计算出不包含缺失值的数据集上的百分位数列
        expected = np.percentile(df["y"].dropna(), k)
        # 检查计算出的百分位数列是否与预期相同
        assert_array_equal(res["y"], expected)
```