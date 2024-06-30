# `D:\src\scipysrc\seaborn\tests\test_regression.py`

```
import warnings
# 导入警告模块，用于管理警告信息

import numpy as np
# 导入NumPy库，用于数值计算

import matplotlib as mpl
import matplotlib.pyplot as plt
# 导入Matplotlib库及其模块，用于绘图功能

import pandas as pd
# 导入Pandas库，用于数据处理和分析

import pytest
# 导入Pytest库，用于编写和运行测试

import numpy.testing as npt
import pandas.testing as pdt
# 导入NumPy和Pandas的测试模块，用于测试数组和数据帧的相等性

try:
    import statsmodels.regression.linear_model as smlm
    _no_statsmodels = False
except ImportError:
    _no_statsmodels = True
# 尝试导入Statsmodels库中的线性模型，如果导入失败则标记_no_statsmodels为True

from seaborn import regression as lm
from seaborn.palettes import color_palette
# 从Seaborn库中导入回归模块和调色板模块

rs = np.random.RandomState(0)
# 创建一个随机状态对象，用于生成随机数种子

class TestLinearPlotter:
    rs = np.random.RandomState(77)
    # 创建另一个随机状态对象，用于生成随机数种子

    df = pd.DataFrame(dict(x=rs.normal(size=60),
                           d=rs.randint(-2, 3, 60),
                           y=rs.gamma(4, size=60),
                           s=np.tile(list("abcdefghij"), 6)))
    # 创建一个包含随机数据的DataFrame对象

    df["z"] = df.y + rs.randn(60)
    df["y_na"] = df.y.copy()
    df.loc[[10, 20, 30], 'y_na'] = np.nan
    # 修改DataFrame，添加新列，并在某些行中插入NaN值

    def test_establish_variables_from_frame(self):
        p = lm._LinearPlotter()
        # 创建_LinearPlotter对象，用于回归分析

        p.establish_variables(self.df, x="x", y="y")
        # 从DataFrame中建立变量x和y

        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y, self.df.y)
        pdt.assert_frame_equal(p.data, self.df)
        # 使用Pandas.testing模块进行断言，确保建立的变量与DataFrame中的数据一致

    def test_establish_variables_from_series(self):
        p = lm._LinearPlotter()
        # 创建_LinearPlotter对象，用于回归分析

        p.establish_variables(None, x=self.df.x, y=self.df.y)
        # 从Series对象中建立变量x和y

        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y, self.df.y)
        assert p.data is None
        # 使用Pandas.testing模块进行断言，确保建立的变量与Series对象中的数据一致，并验证data为None

    def test_establish_variables_from_array(self):
        p = lm._LinearPlotter()
        # 创建_LinearPlotter对象，用于回归分析

        p.establish_variables(None,
                              x=self.df.x.values,
                              y=self.df.y.values)
        # 从数组中建立变量x和y

        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y)
        assert p.data is None
        # 使用NumPy.testing模块进行断言，确保建立的变量与数组中的数据一致，并验证data为None

    def test_establish_variables_from_lists(self):
        p = lm._LinearPlotter()
        # 创建_LinearPlotter对象，用于回归分析

        p.establish_variables(None,
                              x=self.df.x.values.tolist(),
                              y=self.df.y.values.tolist())
        # 从列表中建立变量x和y

        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y)
        assert p.data is None
        # 使用NumPy.testing模块进行断言，确保建立的变量与列表中的数据一致，并验证data为None

    def test_establish_variables_from_mix(self):
        p = lm._LinearPlotter()
        # 创建_LinearPlotter对象，用于回归分析

        p.establish_variables(self.df, x="x", y=self.df.y)
        # 从混合数据类型中建立变量x和y

        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y, self.df.y)
        pdt.assert_frame_equal(p.data, self.df)
        # 使用Pandas.testing模块进行断言，确保建立的变量与混合数据类型中的数据一致

    def test_establish_variables_from_bad(self):
        p = lm._LinearPlotter()
        # 创建_LinearPlotter对象，用于回归分析

        with pytest.raises(ValueError):
            p.establish_variables(None, x="x", y=self.df.y)
        # 测试使用错误的参数建立变量时是否会引发ValueError异常

    def test_dropna(self):
        p = lm._LinearPlotter()
        # 创建_LinearPlotter对象，用于回归分析

        p.establish_variables(self.df, x="x", y_na="y_na")
        # 从DataFrame中建立变量x和y_na

        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y_na, self.df.y_na)
        # 使用Pandas.testing模块进行断言，确保建立的变量与DataFrame中的数据一致

        p.dropna("x", "y_na")
        # 删除NaN值

        mask = self.df.y_na.notnull()
        # 创建一个布尔掩码，标识y_na列中非NaN值的位置

        pdt.assert_series_equal(p.x, self.df.x[mask])
        pdt.assert_series_equal(p.y_na, self.df.y_na[mask])
        # 使用Pandas.testing模块进行断言，确保删除NaN值后的变量与DataFrame中的数据一致


class TestRegressionPlotter:
    pass
# 定义一个空的测试类，用于回归分析绘图器的测试
    # 创建一个指定种子的随机数生成器对象
    rs = np.random.RandomState(49)

    # 在区间[-3, 3]上生成一个等间隔的包含30个元素的数组
    grid = np.linspace(-3, 3, 30)
    # 设置Bootstrap抽样的次数
    n_boot = 100
    # 数值型变量的分箱数
    bins_numeric = 3
    # 给定的分箱边界
    bins_given = [-1, 0, 1]

    # 创建包含随机数据的DataFrame对象
    df = pd.DataFrame(dict(x=rs.normal(size=60),
                           d=rs.randint(-2, 3, 60),
                           y=rs.gamma(4, size=60),
                           s=np.tile(list(range(6)), 10)))
    # 添加一列z，为y加上随机噪声
    df["z"] = df.y + rs.randn(60)
    # 复制一列y的数据到y_na列
    df["y_na"] = df.y.copy()

    # 根据s列的值，生成随机误差
    bw_err = rs.randn(6)[df.s.values] * 2
    # 将y列加上随机误差
    df.y += bw_err

    # 计算逻辑回归的概率
    p = 1 / (1 + np.exp(-(df.x * 2 + rs.randn(60))))
    # 添加一个二项分布的列c
    df["c"] = [rs.binomial(1, p_i) for p_i in p]
    # 将y_na列的第10、20、30行设置为NaN
    df.loc[[10, 20, 30], 'y_na'] = np.nan

    # 测试从DataFrame对象中获取变量的方法
    def test_variables_from_frame(self):

        # 使用RegressionPlotter类初始化p对象，传入变量名和数据源
        p = lm._RegressionPlotter("x", "y", data=self.df, units="s")

        # 断言p对象的x属性与self.df的x列相等
        pdt.assert_series_equal(p.x, self.df.x)
        # 断言p对象的y属性与self.df的y列相等
        pdt.assert_series_equal(p.y, self.df.y)
        # 断言p对象的units属性与self.df的s列相等
        pdt.assert_series_equal(p.units, self.df.s)
        # 断言p对象的data属性与self.df相等
        pdt.assert_frame_equal(p.data, self.df)

    # 测试从Series对象中获取变量的方法
    def test_variables_from_series(self):

        # 使用RegressionPlotter类初始化p对象，传入Series对象作为变量和数据源
        p = lm._RegressionPlotter(self.df.x, self.df.y, units=self.df.s)

        # 断言p对象的x属性与self.df的x列的值相等
        npt.assert_array_equal(p.x, self.df.x)
        # 断言p对象的y属性与self.df的y列的值相等
        npt.assert_array_equal(p.y, self.df.y)
        # 断言p对象的units属性与self.df的s列的值相等
        npt.assert_array_equal(p.units, self.df.s)
        # 断言p对象的data属性为None
        assert p.data is None

    # 测试从混合数据源获取变量的方法
    def test_variables_from_mix(self):

        # 使用RegressionPlotter类初始化p对象，传入变量名、表达式和数据源
        p = lm._RegressionPlotter("x", self.df.y + 1, data=self.df)

        # 断言p对象的x属性与self.df的x列的值相等
        npt.assert_array_equal(p.x, self.df.x)
        # 断言p对象的y属性与self.df的y列的值加1后的值相等
        npt.assert_array_equal(p.y, self.df.y + 1)
        # 断言p对象的data属性与self.df相等
        pdt.assert_frame_equal(p.data, self.df)

    # 测试变量必须是一维的情况
    def test_variables_must_be_1d(self):

        # 创建二维数组和一维数组
        array_2d = np.random.randn(20, 2)
        array_1d = np.random.randn(20)
        # 使用RegressionPlotter类初始化对象时，传入不符合条件的参数，预期抛出异常
        with pytest.raises(ValueError):
            lm._RegressionPlotter(array_2d, array_1d)
        with pytest.raises(ValueError):
            lm._RegressionPlotter(array_1d, array_2d)

    # 测试在处理缺失值时的情况
    def test_dropna(self):

        # 使用RegressionPlotter类初始化p对象，传入变量名和带有NaN的数据源self.df
        p = lm._RegressionPlotter("x", "y_na", data=self.df)
        # 断言p对象的x属性的长度等于self.df的y_na列中非NaN值的数量
        assert len(p.x) == pd.notnull(self.df.y_na).sum()

        # 使用RegressionPlotter类初始化p对象，传入变量名、带有NaN的数据源self.df和dropna=False
        p = lm._RegressionPlotter("x", "y_na", data=self.df, dropna=False)
        # 断言p对象的x属性的长度等于self.df的y_na列的长度
        assert len(p.x) == len(self.df.y_na)

    # 使用参数化测试装饰器，测试RegressionPlotter类的单例情况
    @pytest.mark.parametrize("x,y",
                             [([1.5], [2]),
                              (np.array([1.5]), np.array([2])),
                              (pd.Series(1.5), pd.Series(2))])
    def test_singleton(self, x, y):
        # 使用RegressionPlotter类初始化p对象，传入单一值的变量
        p = lm._RegressionPlotter(x, y)
        # 断言p对象的fit_reg属性为False，表示不进行回归拟合
        assert not p.fit_reg

    # 测试置信区间ci的情况
    def test_ci(self):

        # 使用RegressionPlotter类初始化p对象，传入变量名、置信区间ci和数据源self.df
        p = lm._RegressionPlotter("x", "y", data=self.df, ci=95)
        # 断言p对象的ci属性值为95
        assert p.ci == 95
        # 断言p对象的x_ci属性值为95
        assert p.x_ci == 95

        # 使用RegressionPlotter类初始化p对象，传入变量名、置信区间ci和x_ci，以及数据源self.df
        p = lm._RegressionPlotter("x", "y", data=self.df, ci=95, x_ci=68)
        # 断言p对象的ci属性值为95
        assert p.ci == 95
        # 断言p对象的x_ci属性值为68
        assert p.x_ci == 68

        # 使用RegressionPlotter类初始化p对象，传入变量名、置信区间ci为95、x_ci为"sd"，以及数据源self.df
        p = lm._RegressionPlotter("x", "y", data=self.df, ci=95, x_ci="sd")
        # 断言p对象的ci属性值为95
        assert p.ci == 95
        # 断言p对象的x_ci属性值为"sd"

    # 根据条件是否支持statsmodels库来跳过测试
    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_fast_regression(self):
        # 创建 RegressionPlotter 对象 p，用于快速线性回归分析
        p = lm._RegressionPlotter("x", "y", data=self.df, n_boot=self.n_boot)

        # 使用“快速”函数进行拟合，该函数只进行线性代数运算
        yhat_fast, _ = p.fit_fast(self.grid)

        # 使用 statsmodels 函数进行拟合，使用 OLS 模型
        yhat_smod, _ = p.fit_statsmodels(self.grid, smlm.OLS)

        # 比较 y_hat 值的向量
        npt.assert_array_almost_equal(yhat_fast, yhat_smod)

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_regress_poly(self):
        # 创建 RegressionPlotter 对象 p，用于多项式回归分析
        p = lm._RegressionPlotter("x", "y", data=self.df, n_boot=self.n_boot)

        # 拟合一个一阶多项式
        yhat_poly, _ = p.fit_poly(self.grid, 1)

        # 使用 statsmodels 函数进行拟合，使用 OLS 模型
        yhat_smod, _ = p.fit_statsmodels(self.grid, smlm.OLS)

        # 比较 y_hat 值的向量
        npt.assert_array_almost_equal(yhat_poly, yhat_smod)

    @pytest.mark.parametrize("option", ["logistic", "robust", "lowess"])
    @pytest.mark.skipif(not _no_statsmodels, reason="statsmodels installed")
    def test_statsmodels_missing_errors(self, long_df, option):
        # 使用 pytest 来检测当 statsmodels 缺失时的错误情况
        with pytest.raises(RuntimeError, match=rf"`{option}=True` requires"):
            lm.regplot(long_df, x="x", y="y", **{option: True})

    def test_regress_logx(self):
        # 创建 RegressionPlotter 对象 p，用于对数线性回归分析
        x = np.arange(1, 10)
        y = np.arange(1, 10)
        grid = np.linspace(1, 10, 100)
        p = lm._RegressionPlotter(x, y, n_boot=self.n_boot)

        # 使用“快速”函数进行拟合
        yhat_lin, _ = p.fit_fast(grid)

        # 使用对数 x 的方式进行拟合
        yhat_log, _ = p.fit_logx(grid)

        # 断言：确保对数线性拟合在特定索引处比线性拟合效果更好
        assert yhat_lin[0] > yhat_log[0]
        assert yhat_log[20] > yhat_lin[20]
        assert yhat_lin[90] > yhat_log[90]

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_regress_n_boot(self):
        # 创建 RegressionPlotter 对象 p，用于回归分析并进行自举
        p = lm._RegressionPlotter("x", "y", data=self.df, n_boot=self.n_boot)

        # 快速版本（线性代数）拟合
        _, boots_fast = p.fit_fast(self.grid)
        npt.assert_equal(boots_fast.shape, (self.n_boot, self.grid.size))

        # 较慢版本（np.polyfit）拟合
        _, boots_poly = p.fit_poly(self.grid, 1)
        npt.assert_equal(boots_poly.shape, (self.n_boot, self.grid.size))

        # 最慢版本（statsmodels）拟合
        _, boots_smod = p.fit_statsmodels(self.grid, smlm.OLS)
        npt.assert_equal(boots_smod.shape, (self.n_boot, self.grid.size))

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    # 定义测试方法，用于测试无自举重采样的回归绘图器功能
    def test_regress_without_bootstrap(self):

        # 创建回归绘图器对象，使用快速线性代数版本
        p = lm._RegressionPlotter("x", "y", data=self.df,
                                  n_boot=self.n_boot, ci=None)

        # 调用快速拟合方法
        _, boots_fast = p.fit_fast(self.grid)
        # 断言快速拟合结果为None
        assert boots_fast is None

        # 调用多项式拟合方法
        _, boots_poly = p.fit_poly(self.grid, 1)
        # 断言多项式拟合结果为None
        assert boots_poly is None

        # 调用statsmodels拟合方法
        _, boots_smod = p.fit_statsmodels(self.grid, smlm.OLS)
        # 断言statsmodels拟合结果为None
        assert boots_smod is None

    # 定义测试方法，用于测试带种子的自举重采样的回归绘图器功能
    def test_regress_bootstrap_seed(self):

        # 设定种子
        seed = 200
        # 创建两个具有相同种子的回归绘图器对象
        p1 = lm._RegressionPlotter("x", "y", data=self.df,
                                   n_boot=self.n_boot, seed=seed)
        p2 = lm._RegressionPlotter("x", "y", data=self.df,
                                   n_boot=self.n_boot, seed=seed)

        # 对第一个回归绘图器对象进行快速拟合
        _, boots1 = p1.fit_fast(self.grid)
        # 对第二个回归绘图器对象进行快速拟合
        _, boots2 = p2.fit_fast(self.grid)
        # 断言两次快速拟合的结果数组完全相等
        npt.assert_array_equal(boots1, boots2)

    # 定义测试方法，用于测试数值型预测变量分箱的回归绘图器功能
    def test_numeric_bins(self):

        # 创建回归绘图器对象
        p = lm._RegressionPlotter(self.df.x, self.df.y)
        # 对预测变量进行数值型分箱处理
        x_binned, bins = p.bin_predictor(self.bins_numeric)
        # 断言分箱后的分箱数与预期相等
        npt.assert_equal(len(bins), self.bins_numeric)
        # 断言分箱后的唯一值数组与分箱数相等
        npt.assert_array_equal(np.unique(x_binned), bins)

    # 定义测试方法，用于测试给定分箱方案的回归绘图器功能
    def test_provided_bins(self):

        # 创建回归绘图器对象
        p = lm._RegressionPlotter(self.df.x, self.df.y)
        # 对预测变量进行给定分箱处理
        x_binned, bins = p.bin_predictor(self.bins_given)
        # 断言分箱后的唯一值数组与给定分箱数组相等
        npt.assert_array_equal(np.unique(x_binned), self.bins_given)

    # 定义测试方法，用于测试分箱结果的回归绘图器功能
    def test_bin_results(self):

        # 创建回归绘图器对象
        p = lm._RegressionPlotter(self.df.x, self.df.y)
        # 对预测变量进行给定分箱处理
        x_binned, bins = p.bin_predictor(self.bins_given)
        # 断言第一个分箱区间的最小值大于上一个分箱区间的最大值
        assert self.df.x[x_binned == 0].min() > self.df.x[x_binned == -1].max()
        # 断言第二个分箱区间的最小值大于第一个分箱区间的最大值
        assert self.df.x[x_binned == 1].min() > self.df.x[x_binned == 0].max()

    # 定义测试方法，用于测试散点图数据提取的回归绘图器功能
    def test_scatter_data(self):

        # 创建回归绘图器对象，提取散点图数据
        p = lm._RegressionPlotter(self.df.x, self.df.y)
        x, y = p.scatter_data
        # 断言提取的散点图数据x与数据框self.df中的x列完全相等
        npt.assert_array_equal(x, self.df.x)
        # 断言提取的散点图数据y与数据框self.df中的y列完全相等
        npt.assert_array_equal(y, self.df.y)

        # 创建回归绘图器对象，提取散点图数据（使用不同的预测变量）
        p = lm._RegressionPlotter(self.df.d, self.df.y)
        x, y = p.scatter_data
        # 断言提取的散点图数据x与数据框self.df中的d列完全相等
        npt.assert_array_equal(x, self.df.d)
        # 断言提取的散点图数据y与数据框self.df中的y列完全相等
        npt.assert_array_equal(y, self.df.y)

        # 创建回归绘图器对象，提取散点图数据（使用带有x抖动的不同预测变量）
        p = lm._RegressionPlotter(self.df.d, self.df.y, x_jitter=.1)
        x, y = p.scatter_data
        # 断言提取的散点图数据x不完全等于数据框self.df中的d列
        assert (x != self.df.d).any()
        # 断言提取的散点图数据x的绝对差值小于0.1的比例与数据框self.df中d列的绝对差值小于0.1的比例相等
        npt.assert_array_less(np.abs(self.df.d - x), np.repeat(.1, len(x)))
        # 断言提取的散点图数据y与数据框self.df中的y列完全相等
        npt.assert_array_equal(y, self.df.y)

        # 创建回归绘图器对象，提取散点图数据（使用带有y抖动的不同预测变量）
        p = lm._RegressionPlotter(self.df.d, self.df.y, y_jitter=.05)
        x, y = p.scatter_data
        # 断言提取的散点图数据x与数据框self.df中的d列完全相等
        npt.assert_array_equal(x, self.df.d)
        # 断言提取的散点图数据y的绝对差值小于0.05的比例与数据框self.df中y列的绝对差值小于0.1的比例相等
        npt.assert_array_less(np.abs(self.df.y - y), np.repeat(.1, len(y)))
    # 定义测试函数，用于测试 RegressionPlotter 类的 estimate_data 方法
    def test_estimate_data(self):
        
        # 初始化 RegressionPlotter 对象，使用 self.df 中的 d 列作为 x 数据，y 列作为 y 数据，使用 np.mean 作为 x 估计器
        p = lm._RegressionPlotter(self.df.d, self.df.y, x_estimator=np.mean)
        
        # 调用 estimate_data 方法，返回 x, y, ci 三个变量
        x, y, ci = p.estimate_data
        
        # 断言 x 数组等于按升序排序后的 self.df.d 中唯一值的数组
        npt.assert_array_equal(x, np.sort(np.unique(self.df.d)))
        
        # 断言 y 数组近似等于按 self.df.d 分组后的 y 列的均值数组
        npt.assert_array_almost_equal(y, self.df.groupby("d").y.mean())
        
        # 断言 ci 数组中每个区间的下限小于对应的 y 值
        npt.assert_array_less(np.array(ci)[:, 0], y)
        
        # 断言 ci 数组中每个区间的上限大于对应的 y 值
        npt.assert_array_less(y, np.array(ci)[:, 1])

    # 定义测试函数，用于测试 RegressionPlotter 类的 estimate_data 方法
    def test_estimate_cis(self):
        
        # 设定种子值为 123
        seed = 123
        
        # 初始化 RegressionPlotter 对象，使用 self.df 中的 d 列作为 x 数据，y 列作为 y 数据，使用 np.mean 作为 x 估计器，置信区间为 95%，种子值为 123
        p = lm._RegressionPlotter(self.df.d, self.df.y,
                                  x_estimator=np.mean, ci=95, seed=seed)
        _, _, ci_big = p.estimate_data
        
        # 初始化 RegressionPlotter 对象，使用 self.df 中的 d 列作为 x 数据，y 列作为 y 数据，使用 np.mean 作为 x 估计器，置信区间为 50%，种子值为 123
        p = lm._RegressionPlotter(self.df.d, self.df.y,
                                  x_estimator=np.mean, ci=50, seed=seed)
        _, _, ci_wee = p.estimate_data
        
        # 断言 ci_wee 的区间长度比 ci_big 的区间长度小
        npt.assert_array_less(np.diff(ci_wee), np.diff(ci_big))
        
        # 初始化 RegressionPlotter 对象，使用 self.df 中的 d 列作为 x 数据，y 列作为 y 数据，使用 np.mean 作为 x 估计器，不设定置信区间
        p = lm._RegressionPlotter(self.df.d, self.df.y,
                                  x_estimator=np.mean, ci=None)
        _, _, ci_nil = p.estimate_data
        
        # 断言 ci_nil 数组中的每个元素都为 None
        npt.assert_array_equal(ci_nil, [None] * len(ci_nil))

    # 定义测试函数，用于测试 RegressionPlotter 类的 estimate_data 方法
    def test_estimate_units(self):
        
        # 在本地设置随机数生成器的种子值为 345
        seed = 345
        
        # 初始化 RegressionPlotter 对象，使用 self.df 中的 "x" 列作为 x 数据，"y" 列作为 y 数据，data 为 self.df，单位为 "s"，种子值为 345，x_bins 为 3
        p = lm._RegressionPlotter("x", "y", data=self.df,
                                  units="s", seed=seed, x_bins=3)
        _, _, ci_big = p.estimate_data
        ci_big = np.diff(ci_big, axis=1)
        
        # 初始化 RegressionPlotter 对象，使用 self.df 中的 "x" 列作为 x 数据，"y" 列作为 y 数据，data 为 self.df，种子值为 345，x_bins 为 3
        p = lm._RegressionPlotter("x", "y", data=self.df, seed=seed, x_bins=3)
        _, _, ci_wee = p.estimate_data
        ci_wee = np.diff(ci_wee, axis=1)
        
        # 断言 ci_wee 的区间长度比 ci_big 的区间长度小
        npt.assert_array_less(ci_wee, ci_big)

    # 定义测试函数，用于测试 RegressionPlotter 类的 estimate_data 方法
    def test_partial(self):
        
        # 生成长度为 100 的随机数列 x, y, z
        x = self.rs.randn(100)
        y = x + self.rs.randn(100)
        z = x + self.rs.randn(100)
        
        # 初始化 RegressionPlotter 对象，使用 y, z 作为数据
        p = lm._RegressionPlotter(y, z)
        _, r_orig = np.corrcoef(p.x, p.y)[0]
        
        # 初始化 RegressionPlotter 对象，使用 y, z 作为数据，使用 x 作为 y 的部分回归变量
        p = lm._RegressionPlotter(y, z, y_partial=x)
        _, r_semipartial = np.corrcoef(p.x, p.y)[0]
        
        # 断言半偏相关系数 r_semipartial 小于原始相关系数 r_orig
        assert r_semipartial < r_orig
        
        # 初始化 RegressionPlotter 对象，使用 y, z 作为数据，使用 x 作为 x 的部分回归变量，使用 x 作为 y 的部分回归变量
        p = lm._RegressionPlotter(y, z, x_partial=x, y_partial=x)
        _, r_partial = np.corrcoef(p.x, p.y)[0]
        
        # 断言部分相关系数 r_partial 小于原始相关系数 r_orig
        assert r_partial < r_orig
        
        # 将 x, y 转换为 pandas.Series 对象
        x = pd.Series(x)
        y = pd.Series(y)
        
        # 初始化 RegressionPlotter 对象，使用 y, z 作为数据，使用 x 作为 x 的部分回归变量，使用 x 作为 y 的部分回归变量
        p = lm._RegressionPlotter(y, z, x_partial=x, y_partial=x)
        _, r_partial = np.corrcoef(p.x, p.y)[0]
        
        # 断言部分相关系数 r_partial 小于原始相关系数 r_orig
        assert r_partial < r_orig

    # 用于测试 Logistic 回归功能的函数，如果没有 statsmodels 库，则跳过此测试
    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_logistic_regression(self):
        
        # 初始化 RegressionPlotter 对象，使用 self.df 中的 "x" 列作为 x 数据，"c" 列作为 y 数据，data 为 self.df，进行 Logistic 回归，bootstrap 次数为 self.n_boot
        p = lm._RegressionPlotter("x", "c", data=self.df,
                                  logistic=True, n_boot=self.n_boot)
        _, yhat, _ = p.fit_regression(x_range=(-3, 3))
        
        # 断言 yhat 中的所有值小于 1
        npt.assert_array_less(yhat, 1)
        
        # 断言 yhat 中的所有值大于 0
        npt.assert_array_less(0, yhat)

    # 用于测试 Logistic 回归功能的函数，如果没有 statsmodels 库，则跳过此测试
    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    # 定义一个测试函数，用于检验逻辑回归中的完美分离情况
    def test_logistic_perfect_separation(self):
        # 根据数据集中 x 列的平均值生成布尔序列 y
        y = self.df.x > self.df.x.mean()
        # 创建一个 RegressionPlotter 对象 p，用于绘制回归图，采用 logistic 回归，进行 bootstrap 抽样
        p = lm._RegressionPlotter("x", y, data=self.df,
                                  logistic=True, n_boot=10)
        # 忽略 RuntimeWarning 警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # 对 p 进行回归拟合，返回拟合结果的预测值 yhat
            _, yhat, _ = p.fit_regression(x_range=(-3, 3))
        # 断言预测值 yhat 中的所有值是否为 NaN
        assert np.isnan(yhat).all()

    # 根据条件跳过测试，若无 statsmodels 库则跳过
    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    # 定义一个测试函数，用于检验鲁棒回归
    def test_robust_regression(self):
        # 创建一个 RegressionPlotter 对象 p_ols，用普通最小二乘法进行回归
        p_ols = lm._RegressionPlotter("x", "y", data=self.df,
                                      n_boot=self.n_boot)
        # 对 p_ols 进行回归拟合，返回拟合结果的预测值 ols_yhat
        _, ols_yhat, _ = p_ols.fit_regression(x_range=(-3, 3))

        # 创建一个 RegressionPlotter 对象 p_robust，用鲁棒回归进行回归
        p_robust = lm._RegressionPlotter("x", "y", data=self.df,
                                         robust=True, n_boot=self.n_boot)
        # 对 p_robust 进行回归拟合，返回拟合结果的预测值 robust_yhat
        _, robust_yhat, _ = p_robust.fit_regression(x_range=(-3, 3))

        # 断言普通最小二乘法和鲁棒回归的预测值长度是否相等
        assert len(ols_yhat) == len(robust_yhat)

    # 根据条件跳过测试，若无 statsmodels 库则跳过
    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    # 定义一个测试函数，用于检验低通滤波回归
    def test_lowess_regression(self):
        # 创建一个 RegressionPlotter 对象 p，采用低通滤波进行回归
        p = lm._RegressionPlotter("x", "y", data=self.df, lowess=True)
        # 对 p 进行回归拟合，返回拟合结果的网格点 grid、预测值 yhat 和误差带 err_bands
        grid, yhat, err_bands = p.fit_regression(x_range=(-3, 3))

        # 断言网格点 grid 和预测值 yhat 的长度是否相等
        assert len(grid) == len(yhat)
        # 断言误差带 err_bands 是否为 None
        assert err_bands is None

    # 定义一个测试函数，用于检验回归选项设置
    def test_regression_options(self):
        # 断言低通滤波和二阶多项式回归不可同时使用时会引发 ValueError 异常
        with pytest.raises(ValueError):
            lm._RegressionPlotter("x", "y", data=self.df,
                                  lowess=True, order=2)

        # 断言低通滤波和 logistic 回归不可同时使用时会引发 ValueError 异常
        with pytest.raises(ValueError):
            lm._RegressionPlotter("x", "y", data=self.df,
                                  lowess=True, logistic=True)

    # 定义一个测试函数，用于检验回归限制条件
    def test_regression_limits(self):
        # 创建一个散点图 f, ax，并以 self.df 中的 x 和 y 列进行填充
        f, ax = plt.subplots()
        ax.scatter(self.df.x, self.df.y)
        # 创建一个 RegressionPlotter 对象 p，用于绘制回归图，采用普通最小二乘法回归
        p = lm._RegressionPlotter("x", "y", data=self.df)
        # 对 p 进行回归拟合，返回拟合结果的网格点 grid、预测值和误差带（这里用不到）
        grid, _, _ = p.fit_regression(ax)
        # 获取当前轴的 x 范围
        xlim = ax.get_xlim()
        # 断言网格点 grid 的最小值是否等于轴的最小值
        assert grid.min() == xlim[0]
        # 断言网格点 grid 的最大值是否等于轴的最大值
        assert grid.max() == xlim[1]

        # 创建一个 RegressionPlotter 对象 p，用于绘制回归图，采用截断回归
        p = lm._RegressionPlotter("x", "y", data=self.df, truncate=True)
        # 对 p 进行回归拟合，返回拟合结果的网格点 grid、预测值和误差带（这里用不到）
        grid, _, _ = p.fit_regression()
        # 断言网格点 grid 的最小值是否等于 self.df 中 x 列的最小值
        assert grid.min() == self.df.x.min()
        # 断言网格点 grid 的最大值是否等于 self.df 中 x 列的最大值
        assert grid.max() == self.df.x.max()
# 定义一个名为 TestRegressionPlots 的测试类
class TestRegressionPlots:

    # 使用随机种子 56 初始化随机状态生成器
    rs = np.random.RandomState(56)
    
    # 创建包含 x, y, z, g, h, u 列的 DataFrame
    df = pd.DataFrame(dict(x=rs.randn(90),
                           y=rs.randn(90) + 5,
                           z=rs.randint(0, 1, 90),
                           g=np.repeat(list("abc"), 30),
                           h=np.tile(list("xy"), 45),
                           u=np.tile(np.arange(6), 15)))
    
    # 在 df 的 y 列上添加随机误差
    bw_err = rs.randn(6)[df.u.values]
    df.y += bw_err

    # 定义一个测试函数 test_regplot_basic
    def test_regplot_basic(self):
        # 创建一个新的图形 f 和坐标系 ax
        f, ax = plt.subplots()
        # 绘制回归图，并将其绘制的线条数量与集合数量与预期值进行比较
        lm.regplot(x="x", y="y", data=self.df)
        assert len(ax.lines) == 1
        assert len(ax.collections) == 2
        
        # 获取第一个集合的偏移量，并与 df 的 x 和 y 列进行比较
        x, y = ax.collections[0].get_offsets().T
        npt.assert_array_equal(x, self.df.x)
        npt.assert_array_equal(y, self.df.y)

    # 定义一个测试函数 test_regplot_selective
    def test_regplot_selective(self):
        # 创建一个新的图形 f 和坐标系 ax，并绘制指定参数的回归图
        f, ax = plt.subplots()
        ax = lm.regplot(x="x", y="y", data=self.df, scatter=False, ax=ax)
        assert len(ax.lines) == 1
        assert len(ax.collections) == 1
        ax.clear()

        # 创建一个新的图形 f 和坐标系 ax，并绘制指定参数的回归图
        f, ax = plt.subplots()
        ax = lm.regplot(x="x", y="y", data=self.df, fit_reg=False)
        assert len(ax.lines) == 0
        assert len(ax.collections) == 1
        ax.clear()

        # 创建一个新的图形 f 和坐标系 ax，并绘制指定参数的回归图
        f, ax = plt.subplots()
        ax = lm.regplot(x="x", y="y", data=self.df, ci=None)
        assert len(ax.lines) == 1
        assert len(ax.collections) == 1
        ax.clear()

    # 定义一个测试函数 test_regplot_scatter_kws_alpha
    def test_regplot_scatter_kws_alpha(self):
        # 创建一个新的图形 f 和坐标系 ax，并绘制指定参数的回归图
        f, ax = plt.subplots()
        # 设置散点的颜色和透明度，并检查透明度是否与预期相符
        color = np.array([[0.3, 0.8, 0.5, 0.5]])
        ax = lm.regplot(x="x", y="y", data=self.df,
                        scatter_kws={'color': color})
        assert ax.collections[0]._alpha is None
        assert ax.collections[0]._facecolors[0, 3] == 0.5

        # 创建一个新的图形 f 和坐标系 ax，并绘制指定参数的回归图
        f, ax = plt.subplots()
        color = np.array([[0.3, 0.8, 0.5]])
        ax = lm.regplot(x="x", y="y", data=self.df,
                        scatter_kws={'color': color})
        assert ax.collections[0]._alpha == 0.8

        # 创建一个新的图形 f 和坐标系 ax，并绘制指定参数的回归图
        f, ax = plt.subplots()
        color = np.array([[0.3, 0.8, 0.5]])
        ax = lm.regplot(x="x", y="y", data=self.df,
                        scatter_kws={'color': color, 'alpha': 0.4})
        assert ax.collections[0]._alpha == 0.4

        # 创建一个新的图形 f 和坐标系 ax，并绘制指定参数的回归图
        f, ax = plt.subplots()
        color = 'r'
        ax = lm.regplot(x="x", y="y", data=self.df,
                        scatter_kws={'color': color})
        assert ax.collections[0]._alpha == 0.8

        # 创建一个新的图形 f 和坐标系 ax，并绘制指定参数的回归图
        f, ax = plt.subplots()
        alpha = .3
        ax = lm.regplot(x="x", y="y", data=self.df,
                        x_bins=5, fit_reg=False,
                        scatter_kws={"alpha": alpha})
        for line in ax.lines:
            assert line.get_alpha() == alpha

    # 定义一个测试函数 test_regplot_binned
    def test_regplot_binned(self):
        # 绘制分箱后的回归图，并检查线条和集合的数量是否符合预期
        ax = lm.regplot(x="x", y="y", data=self.df, x_bins=5)
        assert len(ax.lines) == 6
        assert len(ax.collections) == 2
    # 测试 lmplot 函数在没有数据情况下是否抛出 TypeError 异常
    def test_lmplot_no_data(self):

        # 使用 pytest.raises 检查是否抛出 TypeError 异常
        with pytest.raises(TypeError):
            # 调用 lmplot 函数，但没有提供必需的关键字参数 `data`
            lm.lmplot(x="x", y="y")

    # 测试 lmplot 函数的基本功能
    def test_lmplot_basic(self):

        # 调用 lmplot 函数，创建 lmplot 对象 g，使用数据框 self.df
        g = lm.lmplot(x="x", y="y", data=self.df)
        # 获取 lmplot 对象 g 的第一个子图 axes[0, 0]
        ax = g.axes[0, 0]
        # 断言子图中包含的线条数量为 1
        assert len(ax.lines) == 1
        # 断言子图中包含的集合数量为 2
        assert len(ax.collections) == 2

        # 从第一个集合中获取偏移量，并与数据框 self.df 中的 x 和 y 列进行比较
        x, y = ax.collections[0].get_offsets().T
        npt.assert_array_equal(x, self.df.x)
        npt.assert_array_equal(y, self.df.y)

    # 测试 lmplot 函数的 hue 参数
    def test_lmplot_hue(self):

        # 调用 lmplot 函数，使用数据框 self.df，并指定 hue 参数为 "h"
        g = lm.lmplot(x="x", y="y", data=self.df, hue="h")
        # 获取 lmplot 对象 g 的第一个子图 axes[0, 0]
        ax = g.axes[0, 0]

        # 断言子图中包含的线条数量为 2
        assert len(ax.lines) == 2
        # 断言子图中包含的集合数量为 4
        assert len(ax.collections) == 4

    # 测试 lmplot 函数的 markers 参数
    def test_lmplot_markers(self):

        # 调用 lmplot 函数，使用数据框 self.df，并指定 hue 和 markers 参数
        g1 = lm.lmplot(x="x", y="y", data=self.df, hue="h", markers="s")
        # 断言 g1 的 hue_kws 属性与预期字典相等
        assert g1.hue_kws == {"marker": ["s", "s"]}

        # 再次调用 lmplot 函数，使用数据框 self.df，并指定 hue 和 markers 参数
        g2 = lm.lmplot(x="x", y="y", data=self.df, hue="h", markers=["o", "s"])
        # 断言 g2 的 hue_kws 属性与预期字典相等
        assert g2.hue_kws == {"marker": ["o", "s"]}

        # 使用 pytest.raises 检查是否抛出 ValueError 异常
        with pytest.raises(ValueError):
            # 调用 lmplot 函数，使用数据框 self.df，并指定过多的 markers 参数
            lm.lmplot(x="x", y="y", data=self.df, hue="h",
                      markers=["o", "s", "d"])

    # 测试 lmplot 函数的 markers 和 fit_reg 参数
    def test_lmplot_marker_linewidths(self):

        # 调用 lmplot 函数，使用数据框 self.df，并指定 hue、markers 和 fit_reg 参数
        g = lm.lmplot(x="x", y="y", data=self.df, hue="h",
                      fit_reg=False, markers=["o", "+"])
        # 获取 g 的第一个子图 axes[0, 0] 的 collections
        c = g.axes[0, 0].collections
        # 断言第二个集合的线宽与默认线宽相等
        assert c[1].get_linewidths()[0] == mpl.rcParams["lines.linewidth"]

    # 测试 lmplot 函数的 facet 参数
    def test_lmplot_facets(self):

        # 调用 lmplot 函数，使用数据框 self.df，并指定 row 和 col 参数
        g = lm.lmplot(x="x", y="y", data=self.df, row="g", col="h")
        # 断言 g 的子图形状为 (3, 2)
        assert g.axes.shape == (3, 2)

        # 再次调用 lmplot 函数，使用数据框 self.df，并指定 col 和 col_wrap 参数
        g = lm.lmplot(x="x", y="y", data=self.df, col="u", col_wrap=4)
        # 断言 g 的子图形状为 (6,)
        assert g.axes.shape == (6,)

        # 再次调用 lmplot 函数，使用数据框 self.df，并指定 hue 和 col 参数
        g = lm.lmplot(x="x", y="y", data=self.df, hue="h", col="u")
        # 断言 g 的子图形状为 (1, 6)
        assert g.axes.shape == (1, 6)

    # 测试 lmplot 函数的 hue 和 col 参数同时指定时，是否禁用图例
    def test_lmplot_hue_col_nolegend(self):

        # 调用 lmplot 函数，使用数据框 self.df，并指定 col 和 hue 参数
        g = lm.lmplot(x="x", y="y", data=self.df, col="h", hue="h")
        # 断言 g 的 _legend 属性为 None，即禁用了图例
        assert g._legend is None

    # 测试 lmplot 函数的 scatter_kws 参数
    def test_lmplot_scatter_kws(self):

        # 调用 lmplot 函数，使用数据框 self.df，并指定 hue 和 scatter_kws 参数
        g = lm.lmplot(x="x", y="y", hue="h", data=self.df, ci=None)
        # 获取 g 的第一个子图 axes[0, 0] 的 collections 中的两个 scatter 对象
        red_scatter, blue_scatter = g.axes[0, 0].collections

        # 获取红色和蓝色的 RGB 值
        red, blue = color_palette(n_colors=2)
        # 断言 scatter 对象的颜色与预期值相等
        npt.assert_array_equal(red, red_scatter.get_facecolors()[0, :3])
        npt.assert_array_equal(blue, blue_scatter.get_facecolors()[0, :3])

    # 使用 pytest.mark.parametrize 测试 lmplot 函数的 facet 参数和 truncate 参数
    @pytest.mark.parametrize("sharex", [True, False])
    def test_lmplot_facet_truncate(self, sharex):

        # 调用 lmplot 函数，使用数据框 self.df，并指定 x、y、hue、col 和 truncate 参数
        g = lm.lmplot(
            data=self.df, x="x", y="y", hue="g", col="h",
            truncate=False, facet_kws=dict(sharex=sharex),
        )

        # 遍历所有子图，并检查线条的 x 数据是否在子图的 x 范围内
        for ax in g.axes.flat:
            for line in ax.lines:
                xdata = line.get_xdata()
                assert ax.get_xlim() == tuple(xdata[[0, -1]])
    # 定义测试函数 test_lmplot_sharey，用于测试 lmplot 函数的 sharey 参数
    def test_lmplot_sharey(self):

        # 创建包含 x, y, z 列的 DataFrame
        df = pd.DataFrame(dict(
            x=[0, 1, 2, 0, 1, 2],
            y=[1, -1, 0, -100, 200, 0],
            z=["a", "a", "a", "b", "b", "b"],
        ))

        # 检测是否会产生 UserWarning 警告
        with pytest.warns(UserWarning):
            # 调用 lmplot 绘制图形，并指定 x, y, col 参数，设置 sharey=False
            g = lm.lmplot(data=df, x="x", y="y", col="z", sharey=False)
        
        # 获取绘制的子图对象
        ax1, ax2 = g.axes.flat
        # 断言第一个子图的 y 轴限制的下限大于第二个子图的 y 轴限制的下限
        assert ax1.get_ylim()[0] > ax2.get_ylim()[0]
        # 断言第一个子图的 y 轴限制的上限小于第二个子图的 y 轴限制的上限
        assert ax1.get_ylim()[1] < ax2.get_ylim()[1]

    # 定义测试函数 test_lmplot_facet_kws，用于测试 lmplot 函数的 facet_kws 参数
    def test_lmplot_facet_kws(self):

        # 设定 x 轴的限制范围
        xlim = -4, 20
        # 调用 lmplot 绘制图形，并指定 x, y, col 参数，以及 facet_kws 参数设置 x 轴限制范围
        g = lm.lmplot(
            data=self.df, x="x", y="y", col="h", facet_kws={"xlim": xlim}
        )
        
        # 遍历所有子图对象
        for ax in g.axes.flat:
            # 断言每个子图的 x 轴限制与预期的 xlim 相同
            assert ax.get_xlim() == xlim

    # 定义测试函数 test_residplot，用于测试 residplot 函数
    def test_residplot(self):

        # 从 self.df 中获取 x 和 y 数据
        x, y = self.df.x, self.df.y
        # 调用 residplot 绘制残差图，返回子图对象
        ax = lm.residplot(x=x, y=y)

        # 计算残差值
        resid = y - np.polyval(np.polyfit(x, y, 1), x)
        # 获取绘制的点集合的 x, y 坐标
        x_plot, y_plot = ax.collections[0].get_offsets().T

        # 断言 x 值与绘制的 x 值相等
        npt.assert_array_equal(x, x_plot)
        # 断言残差值与绘制的 y 值相近
        npt.assert_array_almost_equal(resid, y_plot)

    # 当未安装 statsmodels 时，跳过此测试
    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    # 定义测试函数 test_residplot_lowess，用于测试 residplot 函数的 lowess 参数
    def test_residplot_lowess(self):

        # 调用 residplot 绘制低通滤波的残差图，返回子图对象
        ax = lm.residplot(x="x", y="y", data=self.df, lowess=True)
        # 断言子图中有两条线
        assert len(ax.lines) == 2

        # 获取第二条线的 x, y 数据
        x, y = ax.lines[1].get_xydata().T
        # 断言 x 数据与 self.df.x 排序后相同
        npt.assert_array_equal(x, np.sort(self.df.x))

    # 使用参数化测试，测试 residplot 函数的 robust 和 lowess 参数缺失时是否抛出 RuntimeError 异常
    @pytest.mark.parametrize("option", ["robust", "lowess"])
    @pytest.mark.skipif(not _no_statsmodels, reason="statsmodels installed")
    def test_residplot_statsmodels_missing_errors(self, long_df, option):
        # 使用 pytest.raises 检测是否抛出 RuntimeError 异常
        with pytest.raises(RuntimeError, match=rf"`{option}=True` requires"):
            # 调用 residplot 函数，传入 long_df, x, y 参数以及 option 参数
            lm.residplot(long_df, x="x", y="y", **{option: True})

    # 定义测试函数 test_three_point_colors，用于测试 regplot 函数的颜色参数
    def test_three_point_colors(self):

        # 生成三个随机的 x, y 数据点
        x, y = np.random.randn(2, 3)
        # 调用 regplot 绘制散点图，指定颜色为红色
        ax = lm.regplot(x=x, y=y, color=(1, 0, 0))
        # 获取散点的颜色
        color = ax.collections[0].get_facecolors()
        # 断言第一个散点的颜色与预期的红色相近
        npt.assert_almost_equal(color[0, :3],
                                (1, 0, 0))

    # 定义测试函数 test_regplot_xlim，用于测试 regplot 函数的 truncate 参数
    def test_regplot_xlim(self):

        # 创建一个图形和子图对象
        f, ax = plt.subplots()
        # 生成三组随机数据点
        x, y1, y2 = np.random.randn(3, 50)
        # 调用 regplot 绘制回归图，不截断数据
        lm.regplot(x=x, y=y1, truncate=False)
        lm.regplot(x=x, y=y2, truncate=False)
        # 获取第一条和第二条线的 x 数据
        line1, line2 = ax.lines
        # 断言两条线的 x 数据相等
        assert np.array_equal(line1.get_xdata(), line2.get_xdata())
```