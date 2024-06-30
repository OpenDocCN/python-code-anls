# `D:\src\scipysrc\seaborn\tests\test_matrix.py`

```
import tempfile  # 导入临时文件处理模块
import copy  # 导入复制模块

import numpy as np  # 导入NumPy库并使用别名np
import matplotlib as mpl  # 导入matplotlib库并使用别名mpl
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块并使用别名plt
import pandas as pd  # 导入Pandas库并使用别名pd

try:
    from scipy.spatial import distance  # 尝试导入scipy的空间距离模块
    from scipy.cluster import hierarchy  # 尝试导入scipy的层次聚类模块
    _no_scipy = False  # 设置_scipy导入成功的标志为False
except ImportError:
    _no_scipy = True  # 如果导入失败，则将_scipy导入成功的标志设置为True

try:
    import fastcluster  # 尝试导入fastcluster模块
    assert fastcluster  # 断言确保fastcluster模块导入成功
    _no_fastcluster = False  # 设置_fastcluster导入成功的标志为False
except ImportError:
    _no_fastcluster = True  # 如果导入失败，则将_fastcluster导入成功的标志设置为True

import numpy.testing as npt  # 导入NumPy的测试模块，并使用别名npt
import pandas.testing as pdt  # 导入Pandas的测试模块，并使用别名pdt
import pytest  # 导入pytest测试框架

from seaborn import matrix as mat  # 从seaborn库中导入matrix模块，并使用别名mat
from seaborn import color_palette  # 从seaborn库中导入color_palette函数
from seaborn._compat import get_colormap  # 从seaborn库的_compat模块中导入get_colormap函数
from seaborn._testing import assert_colors_equal  # 从seaborn库的_testing模块中导入assert_colors_equal函数

class TestHeatmap:  # 定义测试类TestHeatmap

    rs = np.random.RandomState(sum(map(ord, "heatmap")))  # 创建一个随机数生成器实例

    x_norm = rs.randn(4, 8)  # 生成一个4行8列的标准正态分布随机数矩阵
    letters = pd.Series(["A", "B", "C", "D"], name="letters")  # 创建一个包含字母的Pandas系列
    df_norm = pd.DataFrame(x_norm, index=letters)  # 使用随机数矩阵和字母系列创建DataFrame

    x_unif = rs.rand(20, 13)  # 生成一个20行13列的均匀分布随机数矩阵
    df_unif = pd.DataFrame(x_unif)  # 使用均匀分布随机数矩阵创建DataFrame

    default_kws = dict(vmin=None, vmax=None, cmap=None, center=None,
                       robust=False, annot=False, fmt=".2f", annot_kws=None,
                       cbar=True, cbar_kws=None, mask=None)  # 创建默认参数字典

    def test_ndarray_input(self):  # 定义测试数组输入的方法

        p = mat._HeatMapper(self.x_norm, **self.default_kws)  # 使用默认参数创建热图映射器实例
        npt.assert_array_equal(p.plot_data, self.x_norm)  # 断言绘图数据与输入数据相等
        pdt.assert_frame_equal(p.data, pd.DataFrame(self.x_norm))  # 断言数据框架与输入数据框架相等

        npt.assert_array_equal(p.xticklabels, np.arange(8))  # 断言x轴刻度标签与预期的范围数组相等
        npt.assert_array_equal(p.yticklabels, np.arange(4))  # 断言y轴刻度标签与预期的范围数组相等

        assert p.xlabel == ""  # 断言x轴标签为空字符串
        assert p.ylabel == ""  # 断言y轴标签为空字符串

    def test_df_input(self):  # 定义测试数据框架输入的方法

        p = mat._HeatMapper(self.df_norm, **self.default_kws)  # 使用默认参数创建热图映射器实例
        npt.assert_array_equal(p.plot_data, self.x_norm)  # 断言绘图数据与标准化数据相等
        pdt.assert_frame_equal(p.data, self.df_norm)  # 断言数据框架与输入数据框架相等

        npt.assert_array_equal(p.xticklabels, np.arange(8))  # 断言x轴刻度标签与预期的范围数组相等
        npt.assert_array_equal(p.yticklabels, self.letters.values)  # 断言y轴刻度标签与字母系列值相等

        assert p.xlabel == ""  # 断言x轴标签为空字符串
        assert p.ylabel == "letters"  # 断言y轴标签为"letters"

    def test_df_multindex_input(self):  # 定义测试多重索引数据框架输入的方法

        df = self.df_norm.copy()  # 复制标准化数据框架
        index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2),
                                           ("C", 3), ("D", 4)],
                                          names=["letter", "number"])  # 创建多重索引
        index.name = "letter-number"  # 设置多重索引名称
        df.index = index  # 将数据框架的索引设置为多重索引

        p = mat._HeatMapper(df, **self.default_kws)  # 使用默认参数创建热图映射器实例

        combined_tick_labels = ["A-1", "B-2", "C-3", "D-4"]  # 组合的刻度标签列表
        npt.assert_array_equal(p.yticklabels, combined_tick_labels)  # 断言y轴刻度标签与组合标签列表相等
        assert p.ylabel == "letter-number"  # 断言y轴标签为"letter-number"

        p = mat._HeatMapper(df.T, **self.default_kws)  # 使用默认参数创建热图映射器实例（转置）

        npt.assert_array_equal(p.xticklabels, combined_tick_labels)  # 断言x轴刻度标签与组合标签列表相等
        assert p.xlabel == "letter-number"  # 断言x轴标签为"letter-number"

    @pytest.mark.parametrize("dtype", [float, np.int64, object])  # 使用pytest的参数化标记定义数据类型参数
    # 测试函数：测试带有输入掩码的情况
    def test_mask_input(self, dtype):
        # 复制默认关键字参数字典
        kws = self.default_kws.copy()

        # 创建布尔掩码，选择大于零的元素
        mask = self.x_norm > 0
        # 将掩码存储在关键字参数中
        kws['mask'] = mask
        # 将 self.x_norm 转换为指定的数据类型，并存储在 data 中
        data = self.x_norm.astype(dtype)
        # 创建 mat._HeatMapper 实例 p，传入数据和关键字参数
        p = mat._HeatMapper(data, **kws)
        # 使用掩码在数据中创建一个 masked array 对象
        plot_data = np.ma.masked_where(mask, data)

        # 断言 p 的 plot_data 与 plot_data 对象相等
        npt.assert_array_equal(p.plot_data, plot_data)

    # 测试函数：确保掩码的单元格不参与极值计算
    def test_mask_limits(self):
        """Make sure masked cells are not used to calculate extremes"""

        # 复制默认关键字参数字典
        kws = self.default_kws.copy()

        # 创建布尔掩码，选择大于零的元素
        mask = self.x_norm > 0
        # 将掩码存储在关键字参数中
        kws['mask'] = mask
        # 创建 mat._HeatMapper 实例 p，传入数据和关键字参数
        p = mat._HeatMapper(self.x_norm, **kws)

        # 断言 p 的 vmax 等于 masked array 中未掩码部分的最大值
        assert p.vmax == np.ma.array(self.x_norm, mask=mask).max()
        # 断言 p 的 vmin 等于 masked array 中未掩码部分的最小值
        assert p.vmin == np.ma.array(self.x_norm, mask=mask).min()

        # 创建布尔掩码，选择小于零的元素
        mask = self.x_norm < 0
        # 更新关键字参数中的掩码
        kws['mask'] = mask
        # 创建 mat._HeatMapper 实例 p，传入数据和更新后的关键字参数
        p = mat._HeatMapper(self.x_norm, **kws)

        # 断言 p 的 vmin 等于 masked array 中未掩码部分的最小值
        assert p.vmin == np.ma.array(self.x_norm, mask=mask).min()
        # 断言 p 的 vmax 等于 masked array 中未掩码部分的最大值
        assert p.vmax == np.ma.array(self.x_norm, mask=mask).max()

    # 测试函数：测试默认的 vmin 和 vmax 值
    def test_default_vlims(self):

        # 创建 mat._HeatMapper 实例 p，传入数据和默认关键字参数
        p = mat._HeatMapper(self.df_unif, **self.default_kws)
        # 断言 p 的 vmin 等于 self.x_unif 中的最小值
        assert p.vmin == self.x_unif.min()
        # 断言 p 的 vmax 等于 self.x_unif 中的最大值
        assert p.vmax == self.x_unif.max()

    # 测试函数：测试鲁棒的 vmin 和 vmax 值
    def test_robust_vlims(self):

        # 复制默认关键字参数字典
        kws = self.default_kws.copy()
        # 添加 robust 参数设置为 True
        kws["robust"] = True
        # 创建 mat._HeatMapper 实例 p，传入数据和更新后的关键字参数
        p = mat._HeatMapper(self.df_unif, **kws)

        # 断言 p 的 vmin 等于 self.x_unif 中百分位数为 2 的值
        assert p.vmin == np.percentile(self.x_unif, 2)
        # 断言 p 的 vmax 等于 self.x_unif 中百分位数为 98 的值
        assert p.vmax == np.percentile(self.x_unif, 98)

    # 测试函数：测试自定义顺序 vlims
    def test_custom_sequential_vlims(self):

        # 复制默认关键字参数字典
        kws = self.default_kws.copy()
        # 添加自定义的 vmin 和 vmax 参数
        kws["vmin"] = 0
        kws["vmax"] = 1
        # 创建 mat._HeatMapper 实例 p，传入数据和更新后的关键字参数
        p = mat._HeatMapper(self.df_unif, **kws)

        # 断言 p 的 vmin 等于 0
        assert p.vmin == 0
        # 断言 p 的 vmax 等于 1
        assert p.vmax == 1

    # 测试函数：测试自定义分散 vlims
    def test_custom_diverging_vlims(self):

        # 复制默认关键字参数字典
        kws = self.default_kws.copy()
        # 添加自定义的 vmin、vmax 和 center 参数
        kws["vmin"] = -4
        kws["vmax"] = 5
        kws["center"] = 0
        # 创建 mat._HeatMapper 实例 p，传入数据和更新后的关键字参数
        p = mat._HeatMapper(self.df_norm, **kws)

        # 断言 p 的 vmin 等于 -4
        assert p.vmin == -4
        # 断言 p 的 vmax 等于 5
        assert p.vmax == 5

    # 测试函数：测试包含 NaN 的数组情况
    def test_array_with_nans(self):

        # 创建随机数组 x1
        x1 = self.rs.rand(10, 10)
        # 创建一个含有 NaN 的列，并与 x1 拼接成 x2
        nulls = np.zeros(10) * np.nan
        x2 = np.c_[x1, nulls]

        # 创建 mat._HeatMapper 实例 m1，传入 x1 和默认关键字参数
        m1 = mat._HeatMapper(x1, **self.default_kws)
        # 创建 mat._HeatMapper 实例 m2，传入 x2 和默认关键字参数
        m2 = mat._HeatMapper(x2, **self.default_kws)

        # 断言 m1 的 vmin 等于 m2 的 vmin
        assert m1.vmin == m2.vmin
        # 断言 m1 的 vmax 等于 m2 的 vmax
        assert m1.vmax == m2.vmax

    # 测试函数：测试掩码功能
    def test_mask(self):

        # 创建 DataFrame df，含有 NaN 值
        df = pd.DataFrame(data={'a': [1, 1, 1],
                                'b': [2, np.nan, 2],
                                'c': [3, 3, np.nan]})

        # 复制默认关键字参数字典
        kws = self.default_kws.copy()
        # 添加 mask 参数，将 NaN 值设置为 True
        kws["mask"] = np.isnan(df.values)

        # 创建 mat._HeatMapper 实例 m，传入 df 和更新后的关键字参数
        m = mat._HeatMapper(df, **kws)

        # 断言 m 的 plot_data.data 中的 NaN 值与 m 的 plot_data.mask 中的掩码相等
        npt.assert_array_equal(np.isnan(m.plot_data.data),
                               m.plot_data.mask)

    # 测试函数：测试自定义 colormap
    def test_custom_cmap(self):

        # 复制默认关键字参数字典
        kws = self.default_kws.copy()
        # 添加自定义的 colormap 参数
        kws["cmap"] = "BuGn"
        # 创建 mat._HeatMapper 实例 p，传入 self.df_unif 和更新后的关键字参数
        p = mat._HeatMapper(self.df_unif, **kws)
        # 断言 p 的 colormap 等于 mpl.cm.BuGn
        assert p.cmap == mpl.cm.BuGn
    # 定义一个测试方法，用于测试 HeatMapper 对象的 vmin 和 vmax 是否正确设置
    def test_centered_vlims(self):

        # 复制默认参数字典
        kws = self.default_kws.copy()
        # 设置参数字典中的 center 值为 0.5
        kws["center"] = .5

        # 创建 HeatMapper 对象 p，使用 self.df_unif 数据和指定参数
        p = mat._HeatMapper(self.df_unif, **kws)

        # 断言 HeatMapper 对象的 vmin 属性是否等于 self.df_unif 数据的最小值
        assert p.vmin == self.df_unif.values.min()
        # 断言 HeatMapper 对象的 vmax 属性是否等于 self.df_unif 数据的最大值

        assert p.vmax == self.df_unif.values.max()

    # 定义一个测试方法，用于测试 heatmap 方法的默认颜色设置
    def test_default_colors(self):

        # 生成一个包含9个元素的均匀分布的数组
        vals = np.linspace(.2, 1, 9)
        # 设置 colormap 为 binary
        cmap = mpl.cm.binary
        # 调用 mat 的 heatmap 方法，生成一个带有 colormap 的 heatmap
        ax = mat.heatmap([vals], cmap=cmap)
        # 获取 heatmap 中第一个 collection 的颜色
        fc = ax.collections[0].get_facecolors()
        # 生成一个线性分布的值数组
        cvals = np.linspace(0, 1, 9)
        # 使用断言检查获取的颜色值是否与 colormap 对应的颜色值近似相等
        npt.assert_array_almost_equal(fc, cmap(cvals), 2)

    # 定义一个测试方法，用于测试 heatmap 方法的自定义 vmin 和 colormap 设置
    def test_custom_vlim_colors(self):

        # 生成一个包含9个元素的均匀分布的数组
        vals = np.linspace(.2, 1, 9)
        # 设置 colormap 为 binary
        cmap = mpl.cm.binary
        # 调用 mat 的 heatmap 方法，生成一个带有自定义 vmin 和 colormap 的 heatmap
        ax = mat.heatmap([vals], vmin=0, cmap=cmap)
        # 获取 heatmap 中第一个 collection 的颜色
        fc = ax.collections[0].get_facecolors()
        # 使用断言检查获取的颜色值是否与 colormap 对应的颜色值近似相等
        npt.assert_array_almost_equal(fc, cmap(vals), 2)

    # 定义一个测试方法，用于测试 heatmap 方法的自定义 center 和 colormap 设置
    def test_custom_center_colors(self):

        # 生成一个包含9个元素的均匀分布的数组
        vals = np.linspace(.2, 1, 9)
        # 设置 colormap 为 binary
        cmap = mpl.cm.binary
        # 调用 mat 的 heatmap 方法，生成一个带有自定义 center 和 colormap 的 heatmap
        ax = mat.heatmap([vals], center=.5, cmap=cmap)
        # 获取 heatmap 中第一个 collection 的颜色
        fc = ax.collections[0].get_facecolors()
        # 使用断言检查获取的颜色值是否与 colormap 对应的颜色值近似相等
        npt.assert_array_almost_equal(fc, cmap(vals), 2)

    # 定义一个测试方法，用于测试 HeatMapper 对象在指定 colormap 和属性设置情况下的行为
    def test_cmap_with_properties(self):

        # 复制默认参数字典
        kws = self.default_kws.copy()
        # 复制 colormap "BrBG" 并设置无效值的颜色为红色
        cmap = copy.copy(get_colormap("BrBG"))
        cmap.set_bad("red")
        # 将设置好的 colormap 放入参数字典中
        kws["cmap"] = cmap
        # 创建 HeatMapper 对象 hm，使用 self.df_unif 数据和指定参数
        hm = mat._HeatMapper(self.df_unif, **kws)
        # 使用断言检查无效值（NaN）的颜色映射是否正确
        npt.assert_array_equal(
            cmap(np.ma.masked_invalid([np.nan])),
            hm.cmap(np.ma.masked_invalid([np.nan])))

        # 更新参数字典中的 center 属性为 0.5
        kws["center"] = 0.5
        # 创建新的 HeatMapper 对象 hm，使用相同的数据和更新后的参数
        hm = mat._HeatMapper(self.df_unif, **kws)
        # 使用断言检查无效值（NaN）的颜色映射是否正确
        npt.assert_array_equal(
            cmap(np.ma.masked_invalid([np.nan])),
            hm.cmap(np.ma.masked_invalid([np.nan])))

        # 复制默认参数字典
        kws = self.default_kws.copy()
        # 复制 colormap "BrBG" 并设置低于最小值的颜色为红色
        cmap = copy.copy(get_colormap("BrBG"))
        cmap.set_under("red")
        # 将设置好的 colormap 放入参数字典中
        kws["cmap"] = cmap
        # 创建 HeatMapper 对象 hm，使用 self.df_unif 数据和指定参数
        hm = mat._HeatMapper(self.df_unif, **kws)
        # 使用断言检查负无穷的颜色映射是否正确
        npt.assert_array_equal(cmap(-np.inf), hm.cmap(-np.inf))

        # 更新参数字典中的 center 属性为 0.5
        kws["center"] = .5
        # 创建新的 HeatMapper 对象 hm，使用相同的数据和更新后的参数
        hm = mat._HeatMapper(self.df_unif, **kws)
        # 使用断言检查负无穷的颜色映射是否正确
        npt.assert_array_equal(cmap(-np.inf), hm.cmap(-np.inf))

        # 复制默认参数字典
        kws = self.default_kws.copy()
        # 复制 colormap "BrBG" 并设置高于最大值的颜色为红色
        cmap = copy.copy(get_colormap("BrBG"))
        cmap.set_over("red")
        # 将设置好的 colormap 放入参数字典中
        kws["cmap"] = cmap
        # 创建 HeatMapper 对象 hm，使用 self.df_unif 数据和指定参数
        hm = mat._HeatMapper(self.df_unif, **kws)
        # 使用断言检查正无穷的颜色映射是否正确
        npt.assert_array_equal(cmap(-np.inf), hm.cmap(-np.inf))

        # 更新参数字典中的 center 属性为 0.5
        kws["center"] = .5
        # 创建新的 HeatMapper 对象 hm，使用相同的数据和更新后的参数
        hm = mat._HeatMapper(self.df_unif, **kws)
        # 使用断言检查正无穷的颜色映射是否正确
        npt.assert_array_equal(cmap(np.inf), hm.cmap(np.inf))

    # 定义一个测试方法，用于测试 heatmap 方法的显式 None 标准化设置
    def test_explicit_none_norm(self):

        # 生成一个包含9个元素的均匀分布的数组
        vals = np.linspace(.2, 1, 9)
        # 设置 colormap 为 binary
        cmap = mpl.cm.binary
        # 创建一个包含两个子图的图形对象，并返回子图的数组
        _, (ax1, ax2) = plt.subplots(2)

        # 在第一个子图中绘制 heatmap，使用自定义 vmin 和 colormap，但使用默认标准化
        mat.heatmap([vals], vmin=0, cmap=cmap, ax=ax1)
        # 获取第一个子图中 heatmap 的颜色
        fc_default_norm = ax1.collections[0].get_facecolors()

        # 在第二个子图中绘制 heatmap，使用自定义 vmin 和 colormap，并显式设置为无标准化
        mat.heatmap([vals], vmin=0, norm=None, cmap=cmap, ax=ax2)
        # 获取第二个子图中 heatmap 的颜色
        fc_explicit_norm = ax2.collections[0].get_facecolors()

        # 使用断言检查两个子图中的颜色是否近似相等
        npt.assert_array_almost_equal(fc_default_norm, fc_explicit_norm, 2)
    # 测试函数，验证禁用坐标轴标签后 HeatMapper 对象的行为
    def test_ticklabels_off(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        # 禁用 x 和 y 轴的标签显示
        kws['xticklabels'] = False
        kws['yticklabels'] = False
        # 创建 HeatMapper 对象
        p = mat._HeatMapper(self.df_norm, **kws)
        # 断言 xticklabels 和 yticklabels 均为空列表
        assert p.xticklabels == []
        assert p.yticklabels == []

    # 测试函数，验证自定义坐标轴标签后 HeatMapper 对象的行为
    def test_custom_ticklabels(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        # 按照列数和行数自定义 x 和 y 轴的标签
        xticklabels = list('iheartheatmaps'[:self.df_norm.shape[1]])
        yticklabels = list('heatmapsarecool'[:self.df_norm.shape[0]])
        kws['xticklabels'] = xticklabels
        kws['yticklabels'] = yticklabels
        # 创建 HeatMapper 对象
        p = mat._HeatMapper(self.df_norm, **kws)
        # 断言 HeatMapper 对象的 xticklabels 和 yticklabels 与设置的自定义标签相同
        assert p.xticklabels == xticklabels
        assert p.yticklabels == yticklabels

    # 测试函数，验证自定义坐标轴标签间隔后 HeatMapper 对象的行为
    def test_custom_ticklabel_interval(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        # 设置 x 和 y 轴标签的间隔步长
        xstep, ystep = 2, 3
        kws['xticklabels'] = xstep
        kws['yticklabels'] = ystep
        # 创建 HeatMapper 对象
        p = mat._HeatMapper(self.df_norm, **kws)
        # 获取数据框的转置形状
        nx, ny = self.df_norm.T.shape
        # 断言 HeatMapper 对象的 xticks 和 yticks 等于预期的数组
        npt.assert_array_equal(p.xticks, np.arange(0, nx, xstep) + .5)
        npt.assert_array_equal(p.yticks, np.arange(0, ny, ystep) + .5)
        npt.assert_array_equal(p.xticklabels,
                               self.df_norm.columns[0:nx:xstep])
        npt.assert_array_equal(p.yticklabels,
                               self.df_norm.index[0:ny:ystep])

    # 测试函数，验证带注释的热图行为
    def test_heatmap_annotation(self):
        # 生成带注释的热图，并验证注释的文本和字体大小
        ax = mat.heatmap(self.df_norm, annot=True, fmt=".1f",
                         annot_kws={"fontsize": 14})
        for val, text in zip(self.x_norm.flat, ax.texts):
            assert text.get_text() == f"{val:.1f}"
            assert text.get_fontsize() == 14

    # 测试函数，验证带自定义注释属性的热图行为
    def test_heatmap_annotation_overwrite_kws(self):
        # 定义自定义的注释属性，并验证热图注释的文本颜色、水平对齐和垂直对齐属性
        annot_kws = dict(color="0.3", va="bottom", ha="left")
        ax = mat.heatmap(self.df_norm, annot=True, fmt=".1f",
                         annot_kws=annot_kws)
        for text in ax.texts:
            assert text.get_color() == "0.3"
            assert text.get_ha() == "left"
            assert text.get_va() == "bottom"

    # 测试函数，验证带屏蔽区域的热图注释行为
    def test_heatmap_annotation_with_mask(self):
        # 创建包含 NaN 值的数据框，并生成带屏蔽区域的热图
        df = pd.DataFrame(data={'a': [1, 1, 1],
                                'b': [2, np.nan, 2],
                                'c': [3, 3, np.nan]})
        mask = np.isnan(df.values)
        df_masked = np.ma.masked_where(mask, df)
        ax = mat.heatmap(df, annot=True, fmt='.1f', mask=mask)
        # 断言屏蔽区域内的数据点与注释文本数量相同
        assert len(df_masked.compressed()) == len(ax.texts)
        for val, text in zip(df_masked.compressed(), ax.texts):
            assert f"{val:.1f}" == text.get_text()

    # 测试函数，验证带网格颜色的热图行为
    def test_heatmap_annotation_mesh_colors(self):
        # 生成带注释的热图，并验证网格的颜色数量与数据点数量相同
        ax = mat.heatmap(self.df_norm, annot=True)
        mesh = ax.collections[0]
        assert len(mesh.get_facecolors()) == self.df_norm.values.size
        # 关闭所有图形窗口
        plt.close("all")
    # 测试热力图注释其他数据的情况
    def test_heatmap_annotation_other_data(self):
        # 生成带有加法操作的标注数据
        annot_data = self.df_norm + 10

        # 调用热力图函数绘制热力图，并在每个单元格上标注annot_data中的数据，格式化为小数点后一位
        ax = mat.heatmap(self.df_norm, annot=annot_data, fmt=".1f",
                         annot_kws={"fontsize": 14})

        # 遍历标注数据的每个值和对应的文本对象，验证文本显示的内容与预期值一致，并且字体大小为14
        for val, text in zip(annot_data.values.flat, ax.texts):
            assert text.get_text() == f"{val:.1f}"
            assert text.get_fontsize() == 14

    # 测试热力图注释数据形状不同的情况
    def test_heatmap_annotation_different_shapes(self):
        # 选择除最后一行外的数据作为标注数据，预期应该抛出值错误异常
        annot_data = self.df_norm.iloc[:-1]
        with pytest.raises(ValueError):
            # 调用热力图函数尝试绘制热力图，传入不匹配的标注数据
            mat.heatmap(self.df_norm, annot=annot_data)

    # 测试热力图注释有限的刻度标签情况
    def test_heatmap_annotation_with_limited_ticklabels(self):
        # 调用热力图函数绘制热力图，格式化数据为小数点后两位，并且不显示刻度标签
        ax = mat.heatmap(self.df_norm, fmt=".2f", annot=True,
                         xticklabels=False, yticklabels=False)
        # 遍历热力图的每个单元格的值和对应的文本对象，验证文本显示的内容与预期值一致
        for val, text in zip(self.x_norm.flat, ax.texts):
            assert text.get_text() == f"{val:.2f}"

    # 测试热力图的颜色条
    def test_heatmap_cbar(self):
        # 测试绘制默认包含颜色条的热力图，验证图形对象的数量为2（包括主图和颜色条）
        f = plt.figure()
        mat.heatmap(self.df_norm)
        assert len(f.axes) == 2
        plt.close(f)

        # 测试绘制不包含颜色条的热力图，验证图形对象的数量为1（只包括主图）
        f = plt.figure()
        mat.heatmap(self.df_norm, cbar=False)
        assert len(f.axes) == 1
        plt.close(f)

        # 测试绘制具有自定义颜色条位置的热力图，验证图形对象的数量为2（包括主图和自定义位置的颜色条）
        f, (ax1, ax2) = plt.subplots(2)
        mat.heatmap(self.df_norm, ax=ax1, cbar_ax=ax2)
        assert len(f.axes) == 2
        plt.close(f)

    # 测试热力图的坐标轴属性
    def test_heatmap_axes(self):
        # 调用热力图函数绘制热力图，并获取绘图对象
        ax = mat.heatmap(self.df_norm)

        # 获取和验证 x 轴刻度标签的文本内容，应该与 self.df_norm 的列标签列表相同
        xtl = [int(l.get_text()) for l in ax.get_xticklabels()]
        assert xtl == list(self.df_norm.columns)

        # 获取和验证 y 轴刻度标签的文本内容，应该与 self.df_norm 的行索引列表相同
        ytl = [l.get_text() for l in ax.get_yticklabels()]
        assert ytl == list(self.df_norm.index)

        # 验证 x 轴和 y 轴的标签为空字符串和"letters"，以及 x 轴和 y 轴的限制范围
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == "letters"
        assert ax.get_xlim() == (0, 8)
        assert ax.get_ylim() == (4, 0)

    # 测试热力图刻度标签的旋转
    def test_heatmap_ticklabel_rotation(self):
        # 绘制具有指定刻度标签旋转角度的热力图，验证 x 轴刻度标签的旋转角度为0度，y 轴刻度标签的旋转角度为90度
        f, ax = plt.subplots(figsize=(2, 2))
        mat.heatmap(self.df_norm, xticklabels=1, yticklabels=1, ax=ax)

        for t in ax.get_xticklabels():
            assert t.get_rotation() == 0

        for t in ax.get_yticklabels():
            assert t.get_rotation() == 90

        plt.close(f)

        # 使用复制的数据帧绘制热力图，验证 x 轴刻度标签的旋转角度为90度，y 轴刻度标签的旋转角度为0度
        df = self.df_norm.copy()
        df.columns = [str(c) * 10 for c in df.columns]
        df.index = [i * 10 for i in df.index]

        f, ax = plt.subplots(figsize=(2, 2))
        mat.heatmap(df, xticklabels=1, yticklabels=1, ax=ax)

        for t in ax.get_xticklabels():
            assert t.get_rotation() == 90

        for t in ax.get_yticklabels():
            assert t.get_rotation() == 0

        plt.close(f)

    # 测试热力图的内部线条
    def test_heatmap_inner_lines(self):
        # 绘制具有指定线宽和颜色的热力图，验证热力图对象的网格线宽度和颜色
        c = (0, 0, 1, 1)
        ax = mat.heatmap(self.df_norm, linewidths=2, linecolor=c)
        mesh = ax.collections[0]
        assert mesh.get_linewidths()[0] == 2
        assert tuple(mesh.get_edgecolor()[0]) == c

    # 测试热力图的方形纵横比
    def test_square_aspect(self):
        # 绘制具有方形纵横比的热力图，验证热力图对象的纵横比是否为1
        ax = mat.heatmap(self.df_norm, square=True)
        npt.assert_equal(ax.get_aspect(), 1)
    # 定义一个测试方法，用于验证矩阵掩码的功能
    def test_mask_validation(self):

        # 调用矩阵掩码函数，生成一个掩码
        mask = mat._matrix_mask(self.df_norm, None)
        # 断言掩码的形状与原始数据的形状相同
        assert mask.shape == self.df_norm.shape
        # 断言掩码中的值的总和为零，即所有值都被掩盖
        assert mask.values.sum() == 0

        # 使用 pytest 断言引发 ValueError 异常，因为传入了不合法的数组掩码
        with pytest.raises(ValueError):
            bad_array_mask = self.rs.randn(3, 6) > 0
            mat._matrix_mask(self.df_norm, bad_array_mask)

        # 使用 pytest 断言引发 ValueError 异常，因为传入了不合法的 DataFrame 掩码
        with pytest.raises(ValueError):
            bad_df_mask = pd.DataFrame(self.rs.randn(4, 8) > 0)
            mat._matrix_mask(self.df_norm, bad_df_mask)

    # 定义一个测试方法，用于验证处理缺失数据时的掩码功能
    def test_missing_data_mask(self):

        # 创建一个包含缺失数据的 DataFrame
        data = pd.DataFrame(np.arange(4, dtype=float).reshape(2, 2))
        data.loc[0, 0] = np.nan
        # 生成一个数据掩码，用于标识缺失数据的位置
        mask = mat._matrix_mask(data, None)
        # 使用 numpy.testing.assert_array_equal 进行数组相等性断言，验证生成的掩码
        npt.assert_array_equal(mask, [[True, False], [False, False]])

        # 定义一个输入掩码数组
        mask_in = np.array([[False, True], [False, False]])
        # 使用输入掩码数组生成一个新的数据掩码
        mask_out = mat._matrix_mask(data, mask_in)
        # 使用 numpy.testing.assert_array_equal 进行数组相等性断言，验证生成的掩码
        npt.assert_array_equal(mask_out, [[True, True], [False, False]])

    # 定义一个测试方法，用于验证热图绘制时的 colorbar 刻度设置功能
    def test_cbar_ticks(self):

        # 创建一个包含两个子图的图形窗口
        f, (ax1, ax2) = plt.subplots(2)
        # 调用热图绘制函数，在第一个子图中绘制热图，在第二个子图中添加 colorbar
        mat.heatmap(self.df_norm, ax=ax1, cbar_ax=ax2,
                    cbar_kws=dict(drawedges=True))
        # 断言第二个子图的集合对象数量为2，确认成功添加了 colorbar
        assert len(ax2.collections) == 2
# 使用 pytest.mark.skipif 标记的测试类 TestDendrogram，当条件 _no_scipy 为真时跳过测试
@pytest.mark.skipif(_no_scipy, reason="Test requires scipy")
class TestDendrogram:

    # 创建随机数生成器，种子为字符串 "dendrogram" 的 ASCII 码之和
    rs = np.random.RandomState(sum(map(ord, "dendrogram")))

    # 默认参数字典，包含聚类链接、距离度量、方法等信息
    default_kws = dict(linkage=None, metric='euclidean', method='single',
                       axis=1, label=True, rotate=False)

    # 生成一个 4x8 的随机数矩阵 x_norm，并在每列上加上列索引的偏移量
    x_norm = rs.randn(4, 8) + np.arange(8)
    x_norm = (x_norm.T + np.arange(4)).T

    # 创建包含字母的 Series，名称为 "letters"
    letters = pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"],
                        name="letters")

    # 基于 x_norm 和 letters 创建 DataFrame df_norm
    df_norm = pd.DataFrame(x_norm, columns=letters)

    # 如果不是 _no_scipy 状态，进入条件分支
    if not _no_scipy:
        # 如果 _no_fastcluster 为真，计算 x_norm 的欧几里德距离
        if _no_fastcluster:
            x_norm_distances = distance.pdist(x_norm.T, metric='euclidean')
            # 使用单链接法计算层次聚类
            x_norm_linkage = hierarchy.linkage(x_norm_distances, method='single')
        else:
            # 使用 fastcluster 库的向量化链接方法计算层次聚类
            x_norm_linkage = fastcluster.linkage_vector(x_norm.T,
                                                        metric='euclidean',
                                                        method='single')

        # 根据层次聚类结果绘制树状图，不显示图形，设置颜色阈值为负无穷
        x_norm_dendrogram = hierarchy.dendrogram(x_norm_linkage, no_plot=True,
                                                 color_threshold=-np.inf)
        # 获取叶节点的索引
        x_norm_leaves = x_norm_dendrogram['leaves']
        # 将 DataFrame 的列名按照叶节点的顺序转换为数组
        df_norm_leaves = np.asarray(df_norm.columns[x_norm_leaves])

    # 定义测试方法 test_ndarray_input
    def test_ndarray_input(self):
        # 创建 _DendrogramPlotter 对象 p，传入 x_norm 和 default_kws
        p = mat._DendrogramPlotter(self.x_norm, **self.default_kws)
        # 断言 p.array 的转置与 self.x_norm 相等
        npt.assert_array_equal(p.array.T, self.x_norm)
        # 断言 p.data 的转置与 DataFrame(self.x_norm) 相等
        pdt.assert_frame_equal(p.data.T, pd.DataFrame(self.x_norm))

        # 断言 p.linkage 与 self.x_norm_linkage 相等
        npt.assert_array_equal(p.linkage, self.x_norm_linkage)
        # 断言 p.dendrogram 与 self.x_norm_dendrogram 相等
        assert p.dendrogram == self.x_norm_dendrogram

        # 断言 p.reordered_ind 与 self.x_norm_leaves 相等
        npt.assert_array_equal(p.reordered_ind, self.x_norm_leaves)

        # 断言 p.xticklabels 与 self.x_norm_leaves 相等
        npt.assert_array_equal(p.xticklabels, self.x_norm_leaves)
        # 断言 p.yticklabels 为空数组
        npt.assert_array_equal(p.yticklabels, [])

        # 断言 p.xlabel 为 None
        assert p.xlabel is None
        # 断言 p.ylabel 为空字符串
        assert p.ylabel == ''

    # 定义测试方法 test_df_input
    def test_df_input(self):
        # 创建 _DendrogramPlotter 对象 p，传入 df_norm 和 default_kws
        p = mat._DendrogramPlotter(self.df_norm, **self.default_kws)
        # 断言 p.array 的转置与 np.asarray(self.df_norm) 相等
        npt.assert_array_equal(p.array.T, np.asarray(self.df_norm))
        # 断言 p.data 的转置与 self.df_norm 相等
        pdt.assert_frame_equal(p.data.T, self.df_norm)

        # 断言 p.linkage 与 self.x_norm_linkage 相等
        npt.assert_array_equal(p.linkage, self.x_norm_linkage)
        # 断言 p.dendrogram 与 self.x_norm_dendrogram 相等
        assert p.dendrogram == self.x_norm_dendrogram

        # 断言 p.xticklabels 与 np.asarray(self.df_norm.columns)[self.x_norm_leaves] 相等
        npt.assert_array_equal(p.xticklabels,
                               np.asarray(self.df_norm.columns)[
                                   self.x_norm_leaves])
        # 断言 p.yticklabels 为空数组
        npt.assert_array_equal(p.yticklabels, [])

        # 断言 p.xlabel 为 'letters'
        assert p.xlabel == 'letters'
        # 断言 p.ylabel 为空字符串
        assert p.ylabel == ''
    def test_df_multindex_input(self):
        # 复制规范化后的数据框
        df = self.df_norm.copy()
        # 创建多级索引，每个元组包含一个字母和一个数字，指定索引名称为"letter-number"
        index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2),
                                           ("C", 3), ("D", 4)],
                                          names=["letter", "number"])
        index.name = "letter-number"
        # 将数据框的索引设置为新创建的多级索引
        df.index = index
        # 复制默认关键字参数，设置'label'为True
        kws = self.default_kws.copy()
        kws['label'] = True

        # 创建_DendrogramPlotter对象p，传入转置后的数据框df.T和关键字参数kws
        p = mat._DendrogramPlotter(df.T, **kws)

        # 设置x轴刻度标签
        xticklabels = ["A-1", "B-2", "C-3", "D-4"]
        # 根据p.reordered_ind重新排序xticklabels列表
        xticklabels = [xticklabels[i] for i in p.reordered_ind]
        # 断言p对象的x轴刻度标签与重新排序后的xticklabels列表相等
        npt.assert_array_equal(p.xticklabels, xticklabels)
        # 断言p对象的y轴刻度标签为空列表
        npt.assert_array_equal(p.yticklabels, [])
        # 断言p对象的x轴标签为"letter-number"
        assert p.xlabel == "letter-number"

    def test_axis0_input(self):
        # 复制默认关键字参数，设置'axis'为0
        kws = self.default_kws.copy()
        kws['axis'] = 0
        # 创建_DendrogramPlotter对象p，传入数据框self.df_norm.T和关键字参数kws
        p = mat._DendrogramPlotter(self.df_norm.T, **kws)

        # 断言p对象的array属性与self.df_norm.T转换成的NumPy数组相等
        npt.assert_array_equal(p.array, np.asarray(self.df_norm.T))
        # 断言p对象的data属性与self.df_norm.T相等
        pdt.assert_frame_equal(p.data, self.df_norm.T)

        # 断言p对象的linkage属性与self.x_norm_linkage相等
        npt.assert_array_equal(p.linkage, self.x_norm_linkage)
        # 断言p对象的dendrogram属性与self.x_norm_dendrogram相等
        assert p.dendrogram == self.x_norm_dendrogram

        # 断言p对象的x轴刻度标签与self.df_norm_leaves相等
        npt.assert_array_equal(p.xticklabels, self.df_norm_leaves)
        # 断言p对象的y轴刻度标签为空列表
        npt.assert_array_equal(p.yticklabels, [])

        # 断言p对象的x轴标签为'letters'
        assert p.xlabel == 'letters'
        # 断言p对象的y轴标签为空字符串
        assert p.ylabel == ''

    def test_rotate_input(self):
        # 复制默认关键字参数，设置'rotate'为True
        kws = self.default_kws.copy()
        kws['rotate'] = True
        # 创建_DendrogramPlotter对象p，传入数据框self.df_norm和关键字参数kws
        p = mat._DendrogramPlotter(self.df_norm, **kws)
        # 断言p对象的array属性的转置与self.df_norm转换成的NumPy数组相等
        npt.assert_array_equal(p.array.T, np.asarray(self.df_norm))
        # 断言p对象的data属性的转置与self.df_norm相等
        pdt.assert_frame_equal(p.data.T, self.df_norm)

        # 断言p对象的x轴刻度标签为空列表
        npt.assert_array_equal(p.xticklabels, [])
        # 断言p对象的y轴刻度标签与self.df_norm_leaves相等
        npt.assert_array_equal(p.yticklabels, self.df_norm_leaves)

        # 断言p对象的x轴标签为空字符串
        assert p.xlabel == ''
        # 断言p对象的y轴标签为'letters'
        assert p.ylabel == 'letters'

    def test_rotate_axis0_input(self):
        # 复制默认关键字参数，设置'rotate'和'axis'为True和0
        kws = self.default_kws.copy()
        kws['rotate'] = True
        kws['axis'] = 0
        # 创建_DendrogramPlotter对象p，传入数据框self.df_norm.T和关键字参数kws
        p = mat._DendrogramPlotter(self.df_norm.T, **kws)

        # 断言p对象的reordered_ind属性与self.x_norm_leaves相等
        npt.assert_array_equal(p.reordered_ind, self.x_norm_leaves)

    def test_custom_linkage(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()

        try:
            # 尝试导入fastcluster库
            import fastcluster

            # 使用fastcluster库计算self.x_norm的链接向量，设置method='single'和metric='euclidean'
            linkage = fastcluster.linkage_vector(self.x_norm, method='single',
                                                 metric='euclidean')
        except ImportError:
            # 如果导入失败，使用scipy库计算self.x_norm的欧几里得距离矩阵，设置method='single'
            d = distance.pdist(self.x_norm, metric='euclidean')
            linkage = hierarchy.linkage(d, method='single')
        
        # 使用linkage绘制树状图，不绘制到图上，设置color_threshold为负无穷
        dendrogram = hierarchy.dendrogram(linkage, no_plot=True,
                                          color_threshold=-np.inf)
        # 将linkage添加到关键字参数kws中的'linkage'项
        kws['linkage'] = linkage
        # 创建_DendrogramPlotter对象p，传入数据框self.df_norm和关键字参数kws
        p = mat._DendrogramPlotter(self.df_norm, **kws)

        # 断言p对象的linkage属性与linkage相等
        npt.assert_array_equal(p.linkage, linkage)
        # 断言p对象的dendrogram属性与dendrogram相等
        assert p.dendrogram == dendrogram
    # 测试禁用标签功能的情况
    def test_label_false(self):
        # 复制默认关键字参数，并设置'label'为False
        kws = self.default_kws.copy()
        kws['label'] = False
        # 使用修改后的关键字参数创建 _DendrogramPlotter 实例
        p = mat._DendrogramPlotter(self.df_norm, **kws)
        # 断言各属性是否符合预期空值或空字符串
        assert p.xticks == []
        assert p.yticks == []
        assert p.xticklabels == []
        assert p.yticklabels == []
        assert p.xlabel == ""
        assert p.ylabel == ""

    # 测试使用 scipy 计算链接的情况
    def test_linkage_scipy(self):
        # 使用默认关键字参数创建 _DendrogramPlotter 实例
        p = mat._DendrogramPlotter(self.x_norm, **self.default_kws)

        # 调用对象方法计算 scipy 的链接
        scipy_linkage = p._calculate_linkage_scipy()

        # 导入 scipy 中的距离计算和层次聚类模块
        from scipy.spatial import distance
        from scipy.cluster import hierarchy

        # 使用给定的距离度量计算 x_norm 的成对距离
        dists = distance.pdist(self.x_norm.T,
                               metric=self.default_kws['metric'])
        # 使用给定方法计算层次聚类的链接
        linkage = hierarchy.linkage(dists, method=self.default_kws['method'])

        # 断言 scipy 计算的链接与对象方法计算的链接是否一致
        npt.assert_array_equal(scipy_linkage, linkage)

    # 在 fastcluster 安装的情况下，测试使用其他方法的 fastcluster
    @pytest.mark.skipif(_no_fastcluster, reason="fastcluster not installed")
    def test_fastcluster_other_method(self):
        # 导入 fastcluster 模块
        import fastcluster

        # 复制默认关键字参数，并设置'method'为'average'
        kws = self.default_kws.copy()
        kws['method'] = 'average'
        # 使用 fastcluster 计算层次聚类的链接
        linkage = fastcluster.linkage(self.x_norm.T, method='average',
                                      metric='euclidean')
        # 使用修改后的关键字参数创建 _DendrogramPlotter 实例
        p = mat._DendrogramPlotter(self.x_norm, **kws)
        # 断言对象属性的链接与 fastcluster 计算的链接是否一致
        npt.assert_array_equal(p.linkage, linkage)

    # 在 fastcluster 安装的情况下，测试使用非欧氏距离的 fastcluster
    @pytest.mark.skipif(_no_fastcluster, reason="fastcluster not installed")
    def test_fastcluster_non_euclidean(self):
        # 导入 fastcluster 模块
        import fastcluster

        # 复制默认关键字参数，并设置'metric'为'cosine'，'method'为'average'
        kws = self.default_kws.copy()
        kws['metric'] = 'cosine'
        kws['method'] = 'average'
        # 使用 fastcluster 计算指定距离度量和方法的层次聚类的链接
        linkage = fastcluster.linkage(self.x_norm.T, method=kws['method'],
                                      metric=kws['metric'])
        # 使用修改后的关键字参数创建 _DendrogramPlotter 实例
        p = mat._DendrogramPlotter(self.x_norm, **kws)
        # 断言对象属性的链接与 fastcluster 计算的链接是否一致
        npt.assert_array_equal(p.linkage, linkage)

    # 测试生成树状图的情况
    def test_dendrogram_plot(self):
        # 使用默认关键字参数生成 x_norm 的树状图
        d = mat.dendrogram(self.x_norm, **self.default_kws)

        # 获取当前轴对象
        ax = plt.gca()
        # 获取当前 x 轴的限制范围
        xlim = ax.get_xlim()
        # 10 来自于 scipy.cluster.hierarchy 中的 _plot_dendrogram 方法
        xmax = len(d.reordered_ind) * 10

        # 断言 x 轴的范围是否符合预期
        assert xlim[0] == 0
        assert xlim[1] == xmax

        # 断言绘图对象的路径数量与依赖坐标数量是否一致
        assert len(ax.collections[0].get_paths()) == len(d.dependent_coord)

    # 测试旋转树状图的情况
    def test_dendrogram_rotate(self):
        # 复制默认关键字参数，并设置'rotate'为True
        kws = self.default_kws.copy()
        kws['rotate'] = True

        # 使用修改后的关键字参数生成 x_norm 的树状图
        d = mat.dendrogram(self.x_norm, **kws)

        # 获取当前轴对象
        ax = plt.gca()
        # 获取当前 y 轴的限制范围
        ylim = ax.get_ylim()

        # 10 来自于 scipy.cluster.hierarchy 中的 _plot_dendrogram 方法
        ymax = len(d.reordered_ind) * 10

        # 由于 y 轴是反向的，所以 ylim 是 (80, 0)，而不是正常的 (0, 80)
        assert ylim[1] == 0
        assert ylim[0] == ymax
    # 定义测试方法，用于检查树状图的刻度标签旋转
    def test_dendrogram_ticklabel_rotation(self):
        # 创建一个小尺寸的图和坐标轴对象
        f, ax = plt.subplots(figsize=(2, 2))
        # 使用规范化后的数据绘制树状图到指定的坐标轴
        mat.dendrogram(self.df_norm, ax=ax)

        # 遍历 X 轴刻度标签
        for t in ax.get_xticklabels():
            # 断言 X 轴刻度标签的旋转角度为 0 度
            assert t.get_rotation() == 0

        # 关闭当前图形
        plt.close(f)

        # 复制规范化后的数据框，并修改列名和索引
        df = self.df_norm.copy()
        df.columns = [str(c) * 10 for c in df.columns]
        df.index = [i * 10 for i in df.index]

        # 创建一个新的小尺寸图和坐标轴对象
        f, ax = plt.subplots(figsize=(2, 2))
        # 使用修改后的数据绘制树状图到指定的坐标轴
        mat.dendrogram(df, ax=ax)

        # 遍历 X 轴刻度标签
        for t in ax.get_xticklabels():
            # 断言 X 轴刻度标签的旋转角度为 90 度
            assert t.get_rotation() == 90

        # 关闭当前图形
        plt.close(f)

        # 创建一个新的小尺寸图和坐标轴对象
        f, ax = plt.subplots(figsize=(2, 2))
        # 使用转置后的数据绘制树状图到指定的坐标轴，且允许旋转
        mat.dendrogram(df.T, axis=0, rotate=True)

        # 遍历 Y 轴刻度标签
        for t in ax.get_yticklabels():
            # 断言 Y 轴刻度标签的旋转角度为 0 度
            assert t.get_rotation() == 0

        # 关闭当前图形
        plt.close(f)
@pytest.mark.skipif(_no_scipy, reason="Test requires scipy")
class TestClustermap:
    # 创建一个随机数生成器，用于后续随机数的生成
    rs = np.random.RandomState(sum(map(ord, "clustermap")))

    # 生成一个带有噪声的数据集，并进行逐列偏移处理
    x_norm = rs.randn(4, 8) + np.arange(8)
    x_norm = (x_norm.T + np.arange(4)).T

    # 创建一个包含字母的 Series 对象
    letters = pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"], name="letters")

    # 根据生成的数据和字母创建 DataFrame 对象
    df_norm = pd.DataFrame(x_norm, columns=letters)

    # 设置默认的参数字典，包含各种可选参数
    default_kws = dict(pivot_kws=None, z_score=None, standard_scale=None,
                       figsize=(10, 10), row_colors=None, col_colors=None,
                       dendrogram_ratio=.2, colors_ratio=.03,
                       cbar_pos=(0, .8, .05, .2))

    # 设置默认的绘图参数字典，包含聚类相关的参数
    default_plot_kws = dict(metric='euclidean', method='average',
                            colorbar_kws=None,
                            row_cluster=True, col_cluster=True,
                            row_linkage=None, col_linkage=None,
                            tree_kws=None)

    # 根据数据行数生成行颜色列表
    row_colors = color_palette('Set2', df_norm.shape[0])

    # 根据数据列数生成列颜色列表
    col_colors = color_palette('Dark2', df_norm.shape[1])

    # 如果没有禁用 scipy
    if not _no_scipy:
        # 如果禁用了 fastcluster，则使用 scipy 中的层次聚类计算距离和链接
        if _no_fastcluster:
            x_norm_distances = distance.pdist(x_norm.T, metric='euclidean')
            x_norm_linkage = hierarchy.linkage(x_norm_distances, method='single')
        # 否则使用 fastcluster 进行距离和链接的计算
        else:
            x_norm_linkage = fastcluster.linkage_vector(x_norm.T,
                                                        metric='euclidean',
                                                        method='single')

        # 计算数据的树状图信息，但不绘制图形
        x_norm_dendrogram = hierarchy.dendrogram(x_norm_linkage, no_plot=True,
                                                 color_threshold=-np.inf)
        # 获取树状图的叶子节点顺序
        x_norm_leaves = x_norm_dendrogram['leaves']
        # 根据叶子节点顺序重新排列 DataFrame 列名
        df_norm_leaves = np.asarray(df_norm.columns[x_norm_leaves])

    # 测试函数：测试以 ndarray 输入的情况
    def test_ndarray_input(self):
        # 创建 ClusterGrid 对象，使用默认参数
        cg = mat.ClusterGrid(self.x_norm, **self.default_kws)
        # 断言生成的数据框与输入数据一致
        pdt.assert_frame_equal(cg.data, pd.DataFrame(self.x_norm))
        # 断言图形对象的轴数为4
        assert len(cg.fig.axes) == 4
        # 断言行颜色轴为空
        assert cg.ax_row_colors is None
        # 断言列颜色轴为空
        assert cg.ax_col_colors is None

    # 测试函数：测试以 DataFrame 输入的情况
    def test_df_input(self):
        # 创建 ClusterGrid 对象，使用默认参数
        cg = mat.ClusterGrid(self.df_norm, **self.default_kws)
        # 断言生成的数据框与输入数据一致
        pdt.assert_frame_equal(cg.data, self.df_norm)

    # 测试函数：测试以相关系数 DataFrame 输入的情况
    def test_corr_df_input(self):
        # 计算输入数据的相关系数矩阵
        df = self.df_norm.corr()
        # 创建 ClusterGrid 对象，使用默认参数
        cg = mat.ClusterGrid(df, **self.default_kws)
        # 绘制聚类热图，使用默认绘图参数
        cg.plot(**self.default_plot_kws)
        # 检查对角线元素是否接近于1
        diag = cg.data2d.values[np.diag_indices_from(cg.data2d)]
        npt.assert_array_almost_equal(diag, np.ones(cg.data2d.shape[0]))

    # 测试函数：测试以长格式 DataFrame 输入的情况
    def test_pivot_input(self):
        # 复制数据并设置索引名
        df_norm = self.df_norm.copy()
        df_norm.index.name = 'numbers'
        # 将 DataFrame 转换为长格式
        df_long = pd.melt(df_norm.reset_index(), var_name='letters',
                          id_vars='numbers')
        # 复制默认参数字典并设置数据透视的参数
        kws = self.default_kws.copy()
        kws['pivot_kws'] = dict(index='numbers', columns='letters',
                                values='value')
        # 创建 ClusterGrid 对象，使用设定的参数
        cg = mat.ClusterGrid(df_long, **kws)

        # 断言生成的二维数据与原始数据框一致
        pdt.assert_frame_equal(cg.data2d, df_norm)
    # 测试接受颜色参数的方法
    def test_colors_input(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()

        # 设置行颜色和列颜色参数
        kws['row_colors'] = self.row_colors
        kws['col_colors'] = self.col_colors

        # 创建 ClusterGrid 对象
        cg = mat.ClusterGrid(self.df_norm, **kws)
        # 断言行颜色和列颜色与预期相等
        npt.assert_array_equal(cg.row_colors, self.row_colors)
        npt.assert_array_equal(cg.col_colors, self.col_colors)

        # 断言图中包含6个子图
        assert len(cg.fig.axes) == 6

    # 测试接受分类颜色参数的方法
    def test_categorical_colors_input(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()

        # 将行颜色和列颜色转换为分类数据类型
        row_colors = pd.Series(self.row_colors, dtype="category")
        col_colors = pd.Series(self.col_colors, dtype="category", index=self.df_norm.columns)

        # 设置行颜色和列颜色参数
        kws['row_colors'] = row_colors
        kws['col_colors'] = col_colors

        # 期望的行颜色和列颜色转换为 RGB 值
        exp_row_colors = list(map(mpl.colors.to_rgb, row_colors))
        exp_col_colors = list(map(mpl.colors.to_rgb, col_colors))

        # 创建 ClusterGrid 对象
        cg = mat.ClusterGrid(self.df_norm, **kws)
        # 断言行颜色和列颜色与预期相等
        npt.assert_array_equal(cg.row_colors, exp_row_colors)
        npt.assert_array_equal(cg.col_colors, exp_col_colors)

        # 断言图中包含6个子图
        assert len(cg.fig.axes) == 6

    # 测试接受嵌套颜色参数的方法
    def test_nested_colors_input(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()

        # 设置嵌套的行颜色和列颜色参数
        row_colors = [self.row_colors, self.row_colors]
        col_colors = [self.col_colors, self.col_colors]
        kws['row_colors'] = row_colors
        kws['col_colors'] = col_colors

        # 创建 ClusterGrid 对象
        cm = mat.ClusterGrid(self.df_norm, **kws)
        # 断言行颜色和列颜色与预期相等
        npt.assert_array_equal(cm.row_colors, row_colors)
        npt.assert_array_equal(cm.col_colors, col_colors)

        # 断言图中包含6个子图
        assert len(cm.fig.axes) == 6

    # 测试接受自定义颜色映射参数的方法
    def test_colors_input_custom_cmap(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()

        # 设置自定义颜色映射和行列颜色参数
        kws['cmap'] = mpl.cm.PRGn
        kws['row_colors'] = self.row_colors
        kws['col_colors'] = self.col_colors

        # 创建 clustermap 对象
        cg = mat.clustermap(self.df_norm, **kws)
        # 断言行颜色和列颜色与预期相等
        npt.assert_array_equal(cg.row_colors, self.row_colors)
        npt.assert_array_equal(cg.col_colors, self.col_colors)

        # 断言图中包含6个子图
        assert len(cg.fig.axes) == 6

    # 测试 z-score 标准化的方法
    def test_z_score(self):
        # 复制数据帧并进行 z-score 标准化
        df = self.df_norm.copy()
        df = (df - df.mean()) / df.std()
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        kws['z_score'] = 1

        # 创建 ClusterGrid 对象
        cg = mat.ClusterGrid(self.df_norm, **kws)
        # 断言数据帧与预期相等
        pdt.assert_frame_equal(cg.data2d, df)

    # 测试在轴0上进行 z-score 标准化的方法
    def test_z_score_axis0(self):
        # 复制数据帧并转置后进行 z-score 标准化
        df = self.df_norm.copy()
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        kws['z_score'] = 0

        # 创建 ClusterGrid 对象
        cg = mat.ClusterGrid(self.df_norm, **kws)
        # 断言数据帧与预期相等
        pdt.assert_frame_equal(cg.data2d, df)

    # 测试标准缩放的方法
    def test_standard_scale(self):
        # 复制数据帧并进行标准缩放
        df = self.df_norm.copy()
        df = (df - df.min()) / (df.max() - df.min())
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        kws['standard_scale'] = 1

        # 创建 ClusterGrid 对象
        cg = mat.ClusterGrid(self.df_norm, **kws)
        # 断言数据帧与预期相等
        pdt.assert_frame_equal(cg.data2d, df)
    def test_standard_scale_axis0(self):
        df = self.df_norm.copy()  # 复制 self.df_norm 数据框
        df = df.T  # 转置数据框 df
        df = (df - df.min()) / (df.max() - df.min())  # 对每列进行标准化处理
        df = df.T  # 再次将数据框 df 转置回原始形状
        kws = self.default_kws.copy()  # 复制默认参数字典
        kws['standard_scale'] = 0  # 设置标准化参数为 0

        cg = mat.ClusterGrid(self.df_norm, **kws)  # 创建 ClusterGrid 对象
        pdt.assert_frame_equal(cg.data2d, df)  # 断言 ClusterGrid 对象的数据与 df 相等

    def test_z_score_standard_scale(self):
        kws = self.default_kws.copy()  # 复制默认参数字典
        kws['z_score'] = True  # 设置 z_score 参数为 True
        kws['standard_scale'] = True  # 设置标准化参数为 True
        with pytest.raises(ValueError):  # 断言 ValueError 异常被抛出
            mat.ClusterGrid(self.df_norm, **kws)  # 创建 ClusterGrid 对象

    def test_color_list_to_matrix_and_cmap(self):
        # 注意：此处使用名为 col_colors 的属性，但测试行颜色
        matrix, cmap = mat.ClusterGrid.color_list_to_matrix_and_cmap(
            self.col_colors, self.x_norm_leaves, axis=0)  # 调用 color_list_to_matrix_and_cmap 方法

        for i, leaf in enumerate(self.x_norm_leaves):
            color = self.col_colors[leaf]  # 获取 leaf 对应的颜色
            assert_colors_equal(cmap(matrix[i, 0]), color)  # 断言颜色相等

    def test_nested_color_list_to_matrix_and_cmap(self):
        # 注意：此处使用名为 col_colors 的属性，但测试行颜色
        colors = [self.col_colors, self.col_colors[::-1]]  # 创建颜色列表
        matrix, cmap = mat.ClusterGrid.color_list_to_matrix_and_cmap(
            colors, self.x_norm_leaves, axis=0)  # 调用 color_list_to_matrix_and_cmap 方法

        for i, leaf in enumerate(self.x_norm_leaves):
            for j, color_row in enumerate(colors):
                color = color_row[leaf]  # 获取 leaf 对应的颜色
                assert_colors_equal(cmap(matrix[i, j]), color)  # 断言颜色相等

    def test_color_list_to_matrix_and_cmap_axis1(self):
        matrix, cmap = mat.ClusterGrid.color_list_to_matrix_and_cmap(
            self.col_colors, self.x_norm_leaves, axis=1)  # 调用 color_list_to_matrix_and_cmap 方法

        for j, leaf in enumerate(self.x_norm_leaves):
            color = self.col_colors[leaf]  # 获取 leaf 对应的颜色
            assert_colors_equal(cmap(matrix[0, j]), color)  # 断言颜色相等

    def test_color_list_to_matrix_and_cmap_different_sizes(self):
        colors = [self.col_colors, self.col_colors * 2]  # 创建不同大小的颜色列表
        with pytest.raises(ValueError):  # 断言 ValueError 异常被抛出
            matrix, cmap = mat.ClusterGrid.color_list_to_matrix_and_cmap(
                colors, self.x_norm_leaves, axis=1)  # 调用 color_list_to_matrix_and_cmap 方法

    def test_savefig(self):
        # 不确定这种测试方式是否正确....
        cg = mat.ClusterGrid(self.df_norm, **self.default_kws)  # 创建 ClusterGrid 对象
        cg.plot(**self.default_plot_kws)  # 调用 plot 方法
        cg.savefig(tempfile.NamedTemporaryFile(), format='png')  # 保存图形到临时文件

    def test_plot_dendrograms(self):
        cm = mat.clustermap(self.df_norm, **self.default_kws)  # 创建 clustermap 对象

        assert len(cm.ax_row_dendrogram.collections[0].get_paths()) == len(
            cm.dendrogram_row.independent_coord
        )  # 断言行树图路径数量与独立坐标数量相等
        assert len(cm.ax_col_dendrogram.collections[0].get_paths()) == len(
            cm.dendrogram_col.independent_coord
        )  # 断言列树图路径数量与独立坐标数量相等
        data2d = self.df_norm.iloc[cm.dendrogram_row.reordered_ind,
                                   cm.dendrogram_col.reordered_ind]  # 获取重排后的数据
        pdt.assert_frame_equal(cm.data2d, data2d)  # 断言 clustermap 对象的数据与 data2d 相等
    # 定义一个单元测试方法，用于测试在禁用行和列聚类的情况下的集群图表
    def test_cluster_false(self):
        # 复制默认关键字参数，并禁用行和列聚类
        kws = self.default_kws.copy()
        kws['row_cluster'] = False
        kws['col_cluster'] = False

        # 创建一个集群图表对象，使用归一化后的数据和修改后的关键字参数
        cm = mat.clustermap(self.df_norm, **kws)

        # 断言行和列的树状图中线的数量为零
        assert len(cm.ax_row_dendrogram.lines) == 0
        assert len(cm.ax_col_dendrogram.lines) == 0

        # 断言行和列树状图的刻度线的数量为零
        assert len(cm.ax_row_dendrogram.get_xticks()) == 0
        assert len(cm.ax_row_dendrogram.get_yticks()) == 0
        assert len(cm.ax_col_dendrogram.get_xticks()) == 0
        assert len(cm.ax_col_dendrogram.get_yticks()) == 0

        # 断言归一化后的数据与预期数据框相等
        pdt.assert_frame_equal(cm.data2d, self.df_norm)

    # 定义一个单元测试方法，用于测试行和列颜色功能
    def test_row_col_colors(self):
        # 复制默认关键字参数，并设置行和列颜色
        kws = self.default_kws.copy()
        kws['row_colors'] = self.row_colors
        kws['col_colors'] = self.col_colors

        # 创建一个集群图表对象，使用归一化后的数据和修改后的关键字参数
        cm = mat.clustermap(self.df_norm, **kws)

        # 断言行颜色和列颜色的集合数量为一
        assert len(cm.ax_row_colors.collections) == 1
        assert len(cm.ax_col_colors.collections) == 1

    # 定义一个单元测试方法，用于测试在禁用行和列聚类，并设置行和列颜色的情况下的集群图表
    def test_cluster_false_row_col_colors(self):
        # 复制默认关键字参数，并禁用行和列聚类，并设置行和列颜色
        kws = self.default_kws.copy()
        kws['row_cluster'] = False
        kws['col_cluster'] = False
        kws['row_colors'] = self.row_colors
        kws['col_colors'] = self.col_colors

        # 创建一个集群图表对象，使用归一化后的数据和修改后的关键字参数
        cm = mat.clustermap(self.df_norm, **kws)

        # 断言行和列的树状图中线的数量为零
        assert len(cm.ax_row_dendrogram.lines) == 0
        assert len(cm.ax_col_dendrogram.lines) == 0

        # 断言行和列树状图的刻度线的数量为零
        assert len(cm.ax_row_dendrogram.get_xticks()) == 0
        assert len(cm.ax_row_dendrogram.get_yticks()) == 0
        assert len(cm.ax_col_dendrogram.get_xticks()) == 0
        assert len(cm.ax_col_dendrogram.get_yticks()) == 0

        # 断言行颜色和列颜色的集合数量为一
        assert len(cm.ax_row_colors.collections) == 1
        assert len(cm.ax_col_colors.collections) == 1

        # 断言归一化后的数据与预期数据框相等
        pdt.assert_frame_equal(cm.data2d, self.df_norm)

    # 定义一个单元测试方法，用于测试在设置行和列颜色为数据框的情况下的集群图表
    def test_row_col_colors_df(self):
        # 复制默认关键字参数，并设置行和列颜色为数据框
        kws = self.default_kws.copy()
        kws['row_colors'] = pd.DataFrame({'row_1': list(self.row_colors),
                                          'row_2': list(self.row_colors)},
                                         index=self.df_norm.index,
                                         columns=['row_1', 'row_2'])
        kws['col_colors'] = pd.DataFrame({'col_1': list(self.col_colors),
                                          'col_2': list(self.col_colors)},
                                         index=self.df_norm.columns,
                                         columns=['col_1', 'col_2'])

        # 创建一个集群图表对象，使用归一化后的数据和修改后的关键字参数
        cm = mat.clustermap(self.df_norm, **kws)

        # 获取行颜色标签和列颜色标签，并进行断言比较
        row_labels = [l.get_text() for l in cm.ax_row_colors.get_xticklabels()]
        assert cm.row_color_labels == ['row_1', 'row_2']
        assert row_labels == cm.row_color_labels

        col_labels = [l.get_text() for l in cm.ax_col_colors.get_yticklabels()]
        assert cm.col_color_labels == ['col_1', 'col_2']
        assert col_labels == cm.col_color_labels
    def test_row_col_colors_df_shuffled(self):
        # 测试行和列的颜色是否正确匹配，即使顺序不正确也能匹配

        # 获取 DataFrame 的行数 m 和列数 n
        m, n = self.df_norm.shape
        
        # 创建行索引的打乱顺序列表
        shuffled_inds = [self.df_norm.index[i] for i in
                         list(range(0, m, 2)) + list(range(1, m, 2))]
        
        # 创建列索引的打乱顺序列表
        shuffled_cols = [self.df_norm.columns[i] for i in
                         list(range(0, n, 2)) + list(range(1, n, 2))]

        # 复制默认的参数字典
        kws = self.default_kws.copy()

        # 创建行颜色 DataFrame，将其与行索引对应
        row_colors = pd.DataFrame({'row_annot': list(self.row_colors)},
                                  index=self.df_norm.index)
        # 将打乱顺序后的行颜色应用到参数字典中的行颜色
        kws['row_colors'] = row_colors.loc[shuffled_inds]

        # 创建列颜色 DataFrame，将其与列索引对应
        col_colors = pd.DataFrame({'col_annot': list(self.col_colors)},
                                  index=self.df_norm.columns)
        # 将打乱顺序后的列颜色应用到参数字典中的列颜色
        kws['col_colors'] = col_colors.loc[shuffled_cols]

        # 使用参数字典调用 clustermap 函数创建聚类热图
        cm = mat.clustermap(self.df_norm, **kws)
        
        # 断言：检查生成的聚类热图的列颜色是否与原始数据的列颜色匹配
        assert list(cm.col_colors)[0] == list(self.col_colors)
        # 断言：检查生成的聚类热图的行颜色是否与原始数据的行颜色匹配
        assert list(cm.row_colors)[0] == list(self.row_colors)

    def test_row_col_colors_df_missing(self):
        # 测试处理缺失行和列颜色的情况

        # 复制默认的参数字典
        kws = self.default_kws.copy()
        
        # 创建行颜色 DataFrame，将其与行索引对应
        row_colors = pd.DataFrame({'row_annot': list(self.row_colors)},
                                  index=self.df_norm.index)
        # 删除第一个行索引对应的行颜色，并将剩余的行颜色应用到参数字典中的行颜色
        kws['row_colors'] = row_colors.drop(self.df_norm.index[0])

        # 创建列颜色 DataFrame，将其与列索引对应
        col_colors = pd.DataFrame({'col_annot': list(self.col_colors)},
                                  index=self.df_norm.columns)
        # 删除第一个列索引对应的列颜色，并将剩余的列颜色应用到参数字典中的列颜色
        kws['col_colors'] = col_colors.drop(self.df_norm.columns[0])

        # 使用参数字典调用 clustermap 函数创建聚类热图
        cm = mat.clustermap(self.df_norm, **kws)

        # 断言：检查生成的聚类热图的列颜色是否与缺少第一个元素后的原始数据的列颜色匹配
        assert list(cm.col_colors)[0] == [(1.0, 1.0, 1.0)] + list(self.col_colors[1:])
        # 断言：检查生成的聚类热图的行颜色是否与缺少第一个元素后的原始数据的行颜色匹配
        assert list(cm.row_colors)[0] == [(1.0, 1.0, 1.0)] + list(self.row_colors[1:])
    # 定义一个测试方法，用于测试带有行列注释的情况（使用单个轴）。
    def test_row_col_colors_df_one_axis(self):
        # 复制默认的关键字参数
        kws1 = self.default_kws.copy()
        # 创建包含行注释的DataFrame，使用self.row_colors列表作为数据，self.df_norm的索引作为行索引，列名为'row_1'和'row_2'
        kws1['row_colors'] = pd.DataFrame({'row_1': list(self.row_colors),
                                           'row_2': list(self.row_colors)},
                                          index=self.df_norm.index,
                                          columns=['row_1', 'row_2'])

        # 调用clustermap方法生成聚类图，并传入参数kws1
        cm1 = mat.clustermap(self.df_norm, **kws1)

        # 获取聚类图中行颜色标签的文本内容
        row_labels = [l.get_text() for l in cm1.ax_row_colors.get_xticklabels()]
        # 断言聚类图的行颜色标签与预期的['row_1', 'row_2']相同
        assert cm1.row_color_labels == ['row_1', 'row_2']
        # 断言获取的行标签与聚类图的行颜色标签相同
        assert row_labels == cm1.row_color_labels

        # 创建包含列注释的DataFrame，使用self.col_colors列表作为数据，self.df_norm的列索引作为列索引，列名为'col_1'和'col_2'
        kws2 = self.default_kws.copy()
        kws2['col_colors'] = pd.DataFrame({'col_1': list(self.col_colors),
                                           'col_2': list(self.col_colors)},
                                          index=self.df_norm.columns,
                                          columns=['col_1', 'col_2'])

        # 调用clustermap方法生成聚类图，并传入参数kws2
        cm2 = mat.clustermap(self.df_norm, **kws2)

        # 获取聚类图中列颜色标签的文本内容
        col_labels = [l.get_text() for l in cm2.ax_col_colors.get_yticklabels()]
        # 断言聚类图的列颜色标签与预期的['col_1', 'col_2']相同
        assert cm2.col_color_labels == ['col_1', 'col_2']
        # 断言获取的列标签与聚类图的列颜色标签相同
        assert col_labels == cm2.col_color_labels

    # 定义一个测试方法，用于测试带有行列注释的情况（使用Series）
    def test_row_col_colors_series(self):
        # 复制默认的关键字参数
        kws = self.default_kws.copy()
        # 创建包含行注释的Series，使用self.row_colors列表作为数据，名称为'row_annot'，self.df_norm的索引作为索引
        kws['row_colors'] = pd.Series(list(self.row_colors), name='row_annot',
                                      index=self.df_norm.index)
        # 创建包含列注释的Series，使用self.col_colors列表作为数据，名称为'col_annot'，self.df_norm的列索引作为索引
        kws['col_colors'] = pd.Series(list(self.col_colors), name='col_annot',
                                      index=self.df_norm.columns)

        # 调用clustermap方法生成聚类图，并传入参数kws
        cm = mat.clustermap(self.df_norm, **kws)

        # 获取聚类图中行颜色标签的文本内容
        row_labels = [l.get_text() for l in cm.ax_row_colors.get_xticklabels()]
        # 断言聚类图的行颜色标签与预期的['row_annot']相同
        assert cm.row_color_labels == ['row_annot']
        # 断言获取的行标签与聚类图的行颜色标签相同
        assert row_labels == cm.row_color_labels

        # 获取聚类图中列颜色标签的文本内容
        col_labels = [l.get_text() for l in cm.ax_col_colors.get_yticklabels()]
        # 断言聚类图的列颜色标签与预期的['col_annot']相同
        assert cm.col_color_labels == ['col_annot']
        # 断言获取的列标签与聚类图的列颜色标签相同
        assert col_labels == cm.col_color_labels
    def test_row_col_colors_series_shuffled(self):
        # 测试行和列颜色是否正确匹配，即使顺序不正确也能正确处理

        # 获取数据框的行数和列数
        m, n = self.df_norm.shape
        # 创建行索引的洗牌列表
        shuffled_inds = [self.df_norm.index[i] for i in
                         list(range(0, m, 2)) + list(range(1, m, 2))]
        # 创建列索引的洗牌列表
        shuffled_cols = [self.df_norm.columns[i] for i in
                         list(range(0, n, 2)) + list(range(1, n, 2))]

        # 复制默认参数字典
        kws = self.default_kws.copy()

        # 创建行颜色的序列，并将其赋值给参数字典中的'row_colors'键
        row_colors = pd.Series(list(self.row_colors), name='row_annot',
                               index=self.df_norm.index)
        kws['row_colors'] = row_colors.loc[shuffled_inds]

        # 创建列颜色的序列，并将其赋值给参数字典中的'col_colors'键
        col_colors = pd.Series(list(self.col_colors), name='col_annot',
                               index=self.df_norm.columns)
        kws['col_colors'] = col_colors.loc[shuffled_cols]

        # 聚类映射并生成聚类图
        cm = mat.clustermap(self.df_norm, **kws)

        # 断言列颜色顺序是否与预期一致
        assert list(cm.col_colors) == list(self.col_colors)
        # 断言行颜色顺序是否与预期一致
        assert list(cm.row_colors) == list(self.row_colors)

    def test_row_col_colors_series_missing(self):
        # 测试当行和列颜色序列中存在缺失值时的处理

        # 复制默认参数字典
        kws = self.default_kws.copy()

        # 创建行颜色的序列，并将其赋值给参数字典中的'row_colors'键，删除第一个索引
        row_colors = pd.Series(list(self.row_colors), name='row_annot',
                               index=self.df_norm.index)
        kws['row_colors'] = row_colors.drop(self.df_norm.index[0])

        # 创建列颜色的序列，并将其赋值给参数字典中的'col_colors'键，删除第一个索引
        col_colors = pd.Series(list(self.col_colors), name='col_annot',
                               index=self.df_norm.columns)
        kws['col_colors'] = col_colors.drop(self.df_norm.columns[0])

        # 聚类映射并生成聚类图
        cm = mat.clustermap(self.df_norm, **kws)
        # 断言列颜色顺序是否与预期一致（首位加入白色）
        assert list(cm.col_colors) == [(1.0, 1.0, 1.0)] + list(self.col_colors[1:])
        # 断言行颜色顺序是否与预期一致（首位加入白色）
        assert list(cm.row_colors) == [(1.0, 1.0, 1.0)] + list(self.row_colors[1:])

    def test_row_col_colors_ignore_heatmap_kwargs(self):
        # 测试忽略热图关键字参数时的行列颜色处理

        # 生成随机数据矩阵并聚类映射
        g = mat.clustermap(self.rs.uniform(0, 200, self.df_norm.shape),
                           row_colors=self.row_colors,
                           col_colors=self.col_colors,
                           cmap="Spectral",
                           norm=mpl.colors.LogNorm(),
                           vmax=100)

        # 断言行颜色是否正确对应重排后的索引
        assert np.array_equal(
            np.array(self.row_colors)[g.dendrogram_row.reordered_ind],
            g.ax_row_colors.collections[0].get_facecolors()[:, :3]
        )

        # 断言列颜色是否正确对应重排后的索引
        assert np.array_equal(
            np.array(self.col_colors)[g.dendrogram_col.reordered_ind],
            g.ax_col_colors.collections[0].get_facecolors()[:, :3]
        )

    def test_row_col_colors_raise_on_mixed_index_types(self):
        # 测试当行和列索引类型不一致时是否引发异常

        # 创建行颜色的序列，使用数据框的行索引
        row_colors = pd.Series(
            list(self.row_colors), name="row_annot", index=self.df_norm.index
        )

        # 创建列颜色的序列，使用数据框的列索引
        col_colors = pd.Series(
            list(self.col_colors), name="col_annot", index=self.df_norm.columns
        )

        # 使用不匹配类型的行索引调用聚类映射，预期引发TypeError异常
        with pytest.raises(TypeError):
            mat.clustermap(self.x_norm, row_colors=row_colors)

        # 使用不匹配类型的列索引调用聚类映射，预期引发TypeError异常
        with pytest.raises(TypeError):
            mat.clustermap(self.x_norm, col_colors=col_colors)
    # 定义测试方法，用于测试热图中的掩码重组功能
    def test_mask_reorganization(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        # 将"mask"关键字设置为大于0的布尔DataFrame
        kws["mask"] = self.df_norm > 0

        # 生成聚类热图对象，并传入参数kws
        g = mat.clustermap(self.df_norm, **kws)
        # 断言热图的数据行索引与掩码的行索引相等
        npt.assert_array_equal(g.data2d.index, g.mask.index)
        # 断言热图的数据列索引与掩码的列索引相等
        npt.assert_array_equal(g.data2d.columns, g.mask.columns)

        # 断言掩码的行索引与原始数据的行索引经过重排序后的索引相等
        npt.assert_array_equal(g.mask.index,
                               self.df_norm.index[
                                   g.dendrogram_row.reordered_ind])
        # 断言掩码的列索引与原始数据的列索引经过重排序后的索引相等
        npt.assert_array_equal(g.mask.columns,
                               self.df_norm.columns[
                                   g.dendrogram_col.reordered_ind])

    # 定义测试方法，用于测试刻度标签的重组功能
    def test_ticklabel_reorganization(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        # 创建从0到列数的数组作为刻度标签
        xtl = np.arange(self.df_norm.shape[1])
        # 将"xticklabels"关键字设置为刻度标签列表
        kws["xticklabels"] = list(xtl)
        # 使用字母表作为行刻度标签
        ytl = self.letters.loc[:self.df_norm.shape[0]]
        kws["yticklabels"] = ytl

        # 生成聚类热图对象，并传入参数kws
        g = mat.clustermap(self.df_norm, **kws)

        # 获取热图横轴刻度标签的文本列表
        xtl_actual = [t.get_text() for t in g.ax_heatmap.get_xticklabels()]
        # 获取热图纵轴刻度标签的文本列表
        ytl_actual = [t.get_text() for t in g.ax_heatmap.get_yticklabels()]

        # 获取按照重排序索引后的横轴刻度标签期望值
        xtl_want = xtl[g.dendrogram_col.reordered_ind].astype("<U1")
        # 获取按照重排序索引后的纵轴刻度标签期望值
        ytl_want = ytl[g.dendrogram_row.reordered_ind].astype("<U1")

        # 断言实际横轴刻度标签与期望值相等
        npt.assert_array_equal(xtl_actual, xtl_want)
        # 断言实际纵轴刻度标签与期望值相等
        npt.assert_array_equal(ytl_actual, ytl_want)

    # 定义测试方法，用于测试无刻度标签的情况
    def test_noticklabels(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        # 将"xticklabels"和"yticklabels"关键字均设置为False
        kws["xticklabels"] = False
        kws["yticklabels"] = False

        # 生成聚类热图对象，并传入参数kws
        g = mat.clustermap(self.df_norm, **kws)

        # 获取热图横轴刻度标签的文本列表
        xtl_actual = [t.get_text() for t in g.ax_heatmap.get_xticklabels()]
        # 获取热图纵轴刻度标签的文本列表
        ytl_actual = [t.get_text() for t in g.ax_heatmap.get_yticklabels()]

        # 断言横轴刻度标签为空列表
        assert xtl_actual == []
        # 断言纵轴刻度标签为空列表
        assert ytl_actual == []
    def test_size_ratios(self):
        # 在 GridSpec 中，wspace/hspace 的工作方式使得从输入比例到每个轴的实际宽度/高度的映射复杂，
        # 因此这个测试仅仅断言比较关系

        # 创建第一个设置字典，复制默认设置并更新相关参数
        kws1 = self.default_kws.copy()
        kws1.update(dendrogram_ratio=.2, colors_ratio=.03,
                    col_colors=self.col_colors, row_colors=self.row_colors)

        # 创建第二个设置字典，复制第一个设置并更新相关参数
        kws2 = kws1.copy()
        kws2.update(dendrogram_ratio=.3, colors_ratio=.05)

        # 创建两个集群图
        g1 = mat.clustermap(self.df_norm, **kws1)
        g2 = mat.clustermap(self.df_norm, **kws2)

        # 断言比较关系：第二个图的列向谱图的高度大于第一个图的列向谱图的高度
        assert (g2.ax_col_dendrogram.get_position().height
                > g1.ax_col_dendrogram.get_position().height)

        # 断言比较关系：第二个图的列颜色条的高度大于第一个图的列颜色条的高度
        assert (g2.ax_col_colors.get_position().height
                > g1.ax_col_colors.get_position().height)

        # 断言比较关系：第二个图的热图的高度小于第一个图的热图的高度
        assert (g2.ax_heatmap.get_position().height
                < g1.ax_heatmap.get_position().height)

        # 断言比较关系：第二个图的行向谱图的宽度大于第一个图的行向谱图的宽度
        assert (g2.ax_row_dendrogram.get_position().width
                > g1.ax_row_dendrogram.get_position().width)

        # 断言比较关系：第二个图的行颜色条的宽度大于第一个图的行颜色条的宽度
        assert (g2.ax_row_colors.get_position().width
                > g1.ax_row_colors.get_position().width)

        # 断言比较关系：第二个图的热图的宽度小于第一个图的热图的宽度
        assert (g2.ax_heatmap.get_position().width
                < g1.ax_heatmap.get_position().width)

        # 更新设置字典，仅设置列颜色条
        kws1 = self.default_kws.copy()
        kws1.update(col_colors=self.col_colors)

        # 复制更新后的设置字典
        kws2 = kws1.copy()
        kws2.update(col_colors=[self.col_colors, self.col_colors])

        # 创建两个集群图
        g1 = mat.clustermap(self.df_norm, **kws1)
        g2 = mat.clustermap(self.df_norm, **kws2)

        # 断言比较关系：第二个图的列颜色条的高度大于第一个图的列颜色条的高度
        assert (g2.ax_col_colors.get_position().height
                > g1.ax_col_colors.get_position().height)

        # 更新设置字典，设置行向谱图比例为 (.2, .2)
        kws1 = self.default_kws.copy()
        kws1.update(dendrogram_ratio=(.2, .2))

        # 复制更新后的设置字典
        kws2 = kws1.copy()
        kws2.update(dendrogram_ratio=(.2, .3))

        # 创建两个集群图
        g1 = mat.clustermap(self.df_norm, **kws1)
        g2 = mat.clustermap(self.df_norm, **kws2)

        # Fails on pinned matplotlib?
        # 断言比较关系：第二个图的行向谱图的宽度等于第一个图的行向谱图的宽度
        # assert (g2.ax_row_dendrogram.get_position().width
        #         == g1.ax_row_dendrogram.get_position().width)
        # 断言两个集群图的宽度比例相同
        assert g1.gs.get_width_ratios() == g2.gs.get_width_ratios()

        # 断言比较关系：第二个图的列向谱图的高度大于第一个图的列向谱图的高度
        assert (g2.ax_col_dendrogram.get_position().height
                > g1.ax_col_dendrogram.get_position().height)

    def test_cbar_pos(self):
        # 创建设置字典，复制默认设置并设置颜色条位置
        kws = self.default_kws.copy()
        kws["cbar_pos"] = (.2, .1, .4, .3)

        # 创建集群图
        g = mat.clustermap(self.df_norm, **kws)
        # 获取颜色条位置
        pos = g.ax_cbar.get_position()

        # 断言颜色条位置的近似值等于设置的颜色条位置的前两个值
        assert pytest.approx(tuple(pos.p0)) == kws["cbar_pos"][:2]
        # 断言颜色条位置的宽度近似等于设置的颜色条位置的第三个值
        assert pytest.approx(pos.width) == kws["cbar_pos"][2]
        # 断言颜色条位置的高度近似等于设置的颜色条位置的第四个值
        assert pytest.approx(pos.height) == kws["cbar_pos"][3]

        # 清空颜色条位置设置
        kws["cbar_pos"] = None
        # 创建集群图
        g = mat.clustermap(self.df_norm, **kws)
        # 断言颜色条不存在
        assert g.ax_cbar is None
    # 定义一个测试方法，用于测试带有警告的正方形设置
    def test_square_warning(self):
        # 复制默认关键字参数
        kws = self.default_kws.copy()
        # 创建第一个热图并聚类
        g1 = mat.clustermap(self.df_norm, **kws)
    
        # 断言在设置正方形标志时会触发用户警告
        with pytest.warns(UserWarning):
            # 修改关键字参数以设置正方形标志
            kws["square"] = True
            # 创建第二个热图并聚类
            g2 = mat.clustermap(self.df_norm, **kws)
    
        # 获取第一个和第二个热图的热图轴形状
        g1_shape = g1.ax_heatmap.get_position().get_points()
        g2_shape = g2.ax_heatmap.get_position().get_points()
        # 断言两个热图的形状相等
        assert np.array_equal(g1_shape, g2_shape)
    
    # 定义一个测试方法，用于测试簇图的注释
    def test_clustermap_annotation(self):
        # 创建带有注释的簇图
        g = mat.clustermap(self.df_norm, annot=True, fmt=".1f")
        # 遍历数据矩阵中的每个值及其对应的文本对象
        for val, text in zip(np.asarray(g.data2d).flat, g.ax_heatmap.texts):
            # 断言文本对象的内容与格式化后的数据值一致
            assert text.get_text() == f"{val:.1f}"
    
        # 创建带有自身数据作为注释的簇图
        g = mat.clustermap(self.df_norm, annot=self.df_norm, fmt=".1f")
        # 遍历数据矩阵中的每个值及其对应的文本对象
        for val, text in zip(np.asarray(g.data2d).flat, g.ax_heatmap.texts):
            # 断言文本对象的内容与格式化后的数据值一致
            assert text.get_text() == f"{val:.1f}"
    
    # 定义一个测试方法，用于测试树形图的关键字参数设置
    def test_tree_kws(self):
        # 定义 RGB 颜色元组
        rgb = (1, .5, .2)
        # 创建带有树形图关键字参数设置的簇图
        g = mat.clustermap(self.df_norm, tree_kws=dict(color=rgb))
        # 遍历列和行树形图的轴对象
        for ax in [g.ax_col_dendrogram, g.ax_row_dendrogram]:
            # 获取树形对象集合并断言其颜色与预期的 RGB 颜色元组一致
            tree, = ax.collections
            assert tuple(tree.get_color().squeeze())[:3] == rgb
# 如果没有导入 scipy 库，则定义测试函数 test_required_scipy_errors()
if _no_scipy:
    # 生成一个 10x10 的服从正态分布的随机矩阵 x
    x = np.random.normal(0, 1, (10, 10))

    # 使用 pytest 断言捕获 RuntimeError 异常，检测 mat.clustermap(x) 是否触发异常
    with pytest.raises(RuntimeError):
        mat.clustermap(x)

    # 使用 pytest 断言捕获 RuntimeError 异常，检测 mat.ClusterGrid(x) 是否触发异常
    with pytest.raises(RuntimeError):
        mat.ClusterGrid(x)

    # 使用 pytest 断言捕获 RuntimeError 异常，检测 mat.dendrogram(x) 是否触发异常
    with pytest.raises(RuntimeError):
        mat.dendrogram(x)
```