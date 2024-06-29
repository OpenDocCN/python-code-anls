# `D:\src\scipysrc\pandas\pandas\tests\plotting\frame\test_hist_box_by.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

from pandas import DataFrame  # 从pandas库导入DataFrame类
import pandas._testing as tm  # 导入pandas的测试工具模块
from pandas.tests.plotting.common import (  # 从pandas的测试绘图公共模块导入以下函数
    _check_axes_shape,
    _check_plot_works,
    get_x_axis,
    get_y_axis,
)

pytest.importorskip("matplotlib")  # 确保matplotlib库已导入，否则跳过测试

@pytest.fixture
def hist_df():
    # 创建一个包含随机数据的DataFrame对象
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 2)), columns=["A", "B"]
    )
    # 向DataFrame添加列'C'，并用随机选择的字符串填充
    df["C"] = np.random.default_rng(2).choice(["a", "b", "c"], 30)
    # 向DataFrame添加列'D'，并用随机选择的字符串填充
    df["D"] = np.random.default_rng(2).choice(["a", "b", "c"], 30)
    return df  # 返回创建的DataFrame对象


class TestHistWithBy:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, titles, legends",
        [  # 参数化测试用例列表
            ("C", "A", ["a", "b", "c"], [["A"]] * 3),  # 参数组合1
            ("C", ["A", "B"], ["a", "b", "c"], [["A", "B"]] * 3),  # 参数组合2
            ("C", None, ["a", "b", "c"], [["A", "B"]] * 3),  # 参数组合3
            (
                ["C", "D"],
                "A",
                ["(a, a)", "(b, b)", "(c, c)"],
                [["A"]] * 3,
            ),  # 参数组合4
            (
                ["C", "D"],
                ["A", "B"],
                ["(a, a)", "(b, b)", "(c, c)"],
                [["A", "B"]] * 3,
            ),  # 参数组合5
            (
                ["C", "D"],
                None,
                ["(a, a)", "(b, b)", "(c, c)"],
                [["A", "B"]] * 3,
            ),  # 参数组合6
        ],
    )
    def test_hist_plot_by_argument(self, by, column, titles, legends, hist_df):
        # GH 15079：检查GitHub问题号
        # 调用_check_plot_works函数检查直方图绘制是否正常工作
        axes = _check_plot_works(
            hist_df.plot.hist, column=column, by=by, default_axes=True
        )
        # 获取每个子图的标题
        result_titles = [ax.get_title() for ax in axes]
        # 获取每个子图的图例文本
        result_legends = [
            [legend.get_text() for legend in ax.get_legend().texts] for ax in axes
        ]

        # 断言：检查绘图结果的标题是否与预期相符
        assert result_titles == titles
        # 断言：检查绘图结果的图例是否与预期相符
        assert result_legends == legends

    @pytest.mark.parametrize(
        "by, column, titles, legends",
        [
            (0, "A", ["a", "b", "c"], [["A"]] * 3),  # 参数化测试用例1
            (0, None, ["a", "b", "c"], [["A", "B"]] * 3),  # 参数化测试用例2
            (
                [0, "D"],
                "A",
                ["(a, a)", "(b, b)", "(c, c)"],
                [["A"]] * 3,
            ),  # 参数化测试用例3
        ],
    )
    # 测试直方图绘制函数，对于指定的数据列和分组键进行测试
    def test_hist_plot_by_0(self, by, column, titles, legends, hist_df):
        # GH 15079
        # 复制直方图数据框
        df = hist_df.copy()
        # 将列名 "C" 改为 0
        df = df.rename(columns={"C": 0})

        # 调用 _check_plot_works 函数，检查直方图绘制结果，并获取绘图对象 axes
        axes = _check_plot_works(df.plot.hist, default_axes=True, column=column, by=by)
        # 获取每个轴的标题列表
        result_titles = [ax.get_title() for ax in axes]
        # 获取每个轴的图例文本列表
        result_legends = [
            [legend.get_text() for legend in ax.get_legend().texts] for ax in axes
        ]

        # 断言每个轴的图例文本与预期的 legends 相同
        assert result_legends == legends
        # 断言每个轴的标题与预期的 titles 相同
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column",
        [
            ([], ["A"]),
            ([], ["A", "B"]),
            ((), None),
            ((), ["A", "B"]),
        ],
    )
    # 测试直方图绘制函数对于空列表、字符串、元组的分组键的处理
    def test_hist_plot_empty_list_string_tuple_by(self, by, column, hist_df):
        # GH 15079
        msg = "No group keys passed"
        # 使用 pytest 的断言检查 ValueError 异常是否被正确抛出，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(
                hist_df.plot.hist, default_axes=True, column=column, by=by
            )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, layout, axes_num",
        [
            (["C"], "A", (2, 2), 3),
            ("C", "A", (2, 2), 3),
            (["C"], ["A"], (1, 3), 3),
            ("C", None, (3, 1), 3),
            ("C", ["A", "B"], (3, 1), 3),
            (["C", "D"], "A", (9, 1), 3),
            (["C", "D"], "A", (3, 3), 3),
            (["C", "D"], ["A"], (5, 2), 3),
            (["C", "D"], ["A", "B"], (9, 1), 3),
            (["C", "D"], None, (9, 1), 3),
            (["C", "D"], ["A", "B"], (5, 2), 3),
        ],
    )
    # 测试带有分组键的直方图绘制函数，检查不同布局情况下的绘图结果
    def test_hist_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
        # GH 15079
        # _check_plot_works 在绘制时可能会添加一个轴，因此捕获警告信息
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works 函数，检查直方图绘制结果，并获取绘图对象 axes
            axes = _check_plot_works(
                hist_df.plot.hist, column=column, by=by, layout=layout
            )
        # 检查绘图对象 axes 的形状是否符合预期
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize(
        "msg, by, layout",
        [
            ("larger than required size", ["C", "D"], (1, 1)),
            (re.escape("Layout must be a tuple of (rows, columns)"), "C", (1,)),
            ("At least one dimension of layout must be positive", "C", (-1, -1)),
        ],
    )
    # 测试当给定无效布局时是否能正确引发 ValueError 异常
    def test_hist_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df):
        # GH 15079, test if error is raised when invalid layout is given

        # 使用 pytest 的断言检查 ValueError 异常是否被正确抛出，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            hist_df.plot.hist(column=["A", "B"], by=by, layout=layout)

    @pytest.mark.slow
    # 测试函数，用于测试在共享 x 轴的情况下绘制直方图
    def test_axis_share_x_with_by(self, hist_df):
        # 调用 hist_df 对象的 plot.hist 方法，绘制关于列 "A" 的直方图，
        # 按列 "C" 进行分组，并且共享 x 轴
        ax1, ax2, ax3 = hist_df.plot.hist(column="A", by="C", sharex=True)
    
        # 验证共享 x 轴
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)
        assert get_x_axis(ax3).joined(ax1, ax3)
        assert get_x_axis(ax3).joined(ax2, ax3)
    
        # 验证不共享 y 轴
        assert not get_y_axis(ax1).joined(ax1, ax2)
        assert not get_y_axis(ax2).joined(ax1, ax2)
        assert not get_y_axis(ax3).joined(ax1, ax3)
        assert not get_y_axis(ax3).joined(ax2, ax3)
    
    # 测试函数，用于测试在共享 y 轴的情况下绘制直方图
    @pytest.mark.slow
    def test_axis_share_y_with_by(self, hist_df):
        # 调用 hist_df 对象的 plot.hist 方法，绘制关于列 "A" 的直方图，
        # 按列 "C" 进行分组，并且共享 y 轴
        ax1, ax2, ax3 = hist_df.plot.hist(column="A", by="C", sharey=True)
    
        # 验证共享 y 轴
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)
        assert get_y_axis(ax3).joined(ax1, ax3)
        assert get_y_axis(ax3).joined(ax2, ax3)
    
        # 验证不共享 x 轴
        assert not get_x_axis(ax1).joined(ax1, ax2)
        assert not get_x_axis(ax2).joined(ax1, ax2)
        assert not get_x_axis(ax3).joined(ax1, ax3)
        assert not get_x_axis(ax3).joined(ax2, ax3)
    
    # 测试函数，用于测试在指定尺寸下绘制直方图
    @pytest.mark.parametrize("figsize", [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(self, figsize, hist_df):
        # 调用 hist_df 对象的 plot.hist 方法，绘制关于列 "A" 的直方图，
        # 按列 "C" 进行分组，并设置图形的尺寸为 figsize
        axes = hist_df.plot.hist(column="A", by="C", figsize=figsize)
        # 验证生成的子图数量为 3，并且图形的尺寸为指定的 figsize
        _check_axes_shape(axes, axes_num=3, figsize=figsize)
class TestBoxWithBy:
    @pytest.mark.parametrize(
        "by, column, titles, xticklabels",
        [
            ("C", "A", ["A"], [["a", "b", "c"]]),  # 参数化测试用例：使用字符串"C"作为by，"A"作为column，标题为["A"]，xticklabels为[["a", "b", "c"]]
            (
                ["C", "D"],
                "A",
                ["A"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ],
            ),  # 参数化测试用例：使用列表["C", "D"]作为by，"A"作为column，标题为["A"]，xticklabels为[[["(a, a)", "(b, b)", "(c, c)"]]]
            ("C", ["A", "B"], ["A", "B"], [["a", "b", "c"]] * 2),  # 参数化测试用例：使用字符串"C"作为by，["A", "B"]作为column，标题为["A", "B"]，xticklabels为[["a", "b", "c"]]的两倍
            (
                ["C", "D"],
                ["A", "B"],
                ["A", "B"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ]
                * 2,
            ),  # 参数化测试用例：使用列表["C", "D"]作为by，["A", "B"]作为column，标题为["A", "B"]，xticklabels为[[["(a, a)", "(b, b)", "(c, c)"]]]的两倍
            (["C"], None, ["A", "B"], [["a", "b", "c"]] * 2),  # 参数化测试用例：使用列表["C"]作为by，None作为column，标题为["A", "B"]，xticklabels为[["a", "b", "c"]]的两倍
        ],
    )
    def test_box_plot_by_argument(self, by, column, titles, xticklabels, hist_df):
        # GH 15079
        axes = _check_plot_works(
            hist_df.plot.box, default_axes=True, column=column, by=by
        )  # 调用_check_plot_works函数，生成箱线图，设置默认坐标轴，传入column和by参数
        result_titles = [ax.get_title() for ax in axes]  # 获取每个轴的标题
        result_xticklabels = [
            [label.get_text() for label in ax.get_xticklabels()] for ax in axes
        ]  # 获取每个轴的X轴刻度标签的文本内容

        assert result_xticklabels == xticklabels  # 断言X轴刻度标签内容与期望值xticklabels相等
        assert result_titles == titles  # 断言标题内容与期望值titles相等

    @pytest.mark.parametrize(
        "by, column, titles, xticklabels",
        [
            (0, "A", ["A"], [["a", "b", "c"]]),  # 参数化测试用例：使用整数0作为by，"A"作为column，标题为["A"]，xticklabels为[["a", "b", "c"]]
            (
                [0, "D"],
                "A",
                ["A"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ],
            ),  # 参数化测试用例：使用列表[0, "D"]作为by，"A"作为column，标题为["A"]，xticklabels为[[["(a, a)", "(b, b)", "(c, c)"]]]
            (0, None, ["A", "B"], [["a", "b", "c"]] * 2),  # 参数化测试用例：使用整数0作为by，None作为column，标题为["A", "B"]，xticklabels为[["a", "b", "c"]]的两倍
        ],
    )
    def test_box_plot_by_0(self, by, column, titles, xticklabels, hist_df):
        # GH 15079
        df = hist_df.copy()  # 复制hist_df数据框
        df = df.rename(columns={"C": 0})  # 重命名列"C"为0

        axes = _check_plot_works(df.plot.box, default_axes=True, column=column, by=by)  # 调用_check_plot_works函数，生成箱线图，设置默认坐标轴，传入column和by参数
        result_titles = [ax.get_title() for ax in axes]  # 获取每个轴的标题
        result_xticklabels = [
            [label.get_text() for label in ax.get_xticklabels()] for ax in axes
        ]  # 获取每个轴的X轴刻度标签的文本内容

        assert result_xticklabels == xticklabels  # 断言X轴刻度标签内容与期望值xticklabels相等
        assert result_titles == titles  # 断言标题内容与期望值titles相等

    @pytest.mark.parametrize(
        "by, column",
        [
            ([], ["A"]),  # 参数化测试用例：使用空列表作为by，["A"]作为column
            ((), "A"),  # 参数化测试用例：使用空元组作为by，"A"作为column
            ([], None),  # 参数化测试用例：使用空列表作为by，None作为column
            ((), ["A", "B"]),  # 参数化测试用例：使用空元组作为by，["A", "B"]作为column
        ],
    )
    def test_box_plot_with_none_empty_list_by(self, by, column, hist_df):
        # GH 15079
        msg = "No group keys passed"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)  # 调用_check_plot_works函数，生成箱线图，设置默认坐标轴，传入column和by参数

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, layout, axes_num",
        [   # 参数化测试用例，定义了不同的参数组合
            (["C"], "A", (1, 1), 1),        # 参数组合1
            ("C", "A", (1, 1), 1),          # 参数组合2
            ("C", None, (2, 1), 2),         # 参数组合3
            ("C", ["A", "B"], (1, 2), 2),   # 参数组合4
            (["C", "D"], "A", (1, 1), 1),   # 参数组合5
            (["C", "D"], None, (1, 2), 2),  # 参数组合6
        ],
    )
    def test_box_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
        # GH 15079
        # 调用 _check_plot_works 函数，验证绘图函数的工作情况
        axes = _check_plot_works(
            hist_df.plot.box, default_axes=True, column=column, by=by, layout=layout
        )
        # 调用 _check_axes_shape 函数，验证生成的图形轴的形状
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize(
        "msg, by, layout",
        [   # 参数化测试用例，定义了不同的参数组合
            ("larger than required size", ["C", "D"], (1, 1)),  # 参数组合1
            (re.escape("Layout must be a tuple of (rows, columns)"), "C", (1,)),  # 参数组合2
            ("At least one dimension of layout must be positive", "C", (-1, -1)),  # 参数组合3
        ],
    )
    def test_box_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df):
        # GH 15079, test if error is raised when invalid layout is given
        # 使用 pytest.raises 来检查是否引发 ValueError 异常，并匹配给定的错误消息
        with pytest.raises(ValueError, match=msg):
            hist_df.plot.box(column=["A", "B"], by=by, layout=layout)

    @pytest.mark.parametrize("figsize", [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(self, figsize, hist_df):
        # GH 15079
        # 绘制柱状图，指定图形大小为 figsize
        axes = hist_df.plot.box(column="A", by="C", figsize=figsize)
        # 调用 _check_axes_shape 函数，验证生成的图形轴的形状
        _check_axes_shape(axes, axes_num=1, figsize=figsize)
```