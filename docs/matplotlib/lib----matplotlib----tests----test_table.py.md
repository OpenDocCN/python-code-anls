# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_table.py`

```
import datetime  # 导入处理日期和时间的模块
from unittest.mock import Mock  # 从unittest.mock模块中导入Mock类

import numpy as np  # 导入数值计算库numpy
import pytest  # 导入用于编写和运行测试用例的pytest库

import matplotlib.pyplot as plt  # 导入用于绘制图表的matplotlib.pyplot模块
import matplotlib as mpl  # 导入matplotlib库的主要接口
from matplotlib.path import Path  # 从matplotlib.path模块中导入Path类
from matplotlib.table import CustomCell, Table  # 从matplotlib.table模块中导入CustomCell和Table类
from matplotlib.testing.decorators import image_comparison, check_figures_equal  # 导入用于图像比较和检查的装饰器
from matplotlib.transforms import Bbox  # 从matplotlib.transforms模块中导入Bbox类
import matplotlib.units as munits  # 导入用于处理度量单位的模块matplotlib.units


def test_non_square():
    # 检查创建非方形表格是否正常工作
    cellcolors = ['b', 'r']
    plt.table(cellColours=cellcolors)


@image_comparison(['table_zorder.png'], remove_text=True)
def test_zorder():
    data = [[66386, 174296],
            [58230, 381139]]

    colLabels = ('Freeze', 'Wind')
    rowLabels = ['%d year' % x for x in (100, 50)]

    cellText = []
    yoff = np.zeros(len(colLabels))
    for row in reversed(data):
        yoff += row
        cellText.append(['%1.1f' % (x/1000.0) for x in yoff])

    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(t, np.cos(t), lw=4, zorder=2)  # 绘制余弦曲线，设置图层顺序为2

    plt.table(cellText=cellText,
              rowLabels=rowLabels,
              colLabels=colLabels,
              loc='center',
              zorder=-2,  # 设置表格的图层顺序为-2
              )

    plt.table(cellText=cellText,
              rowLabels=rowLabels,
              colLabels=colLabels,
              loc='upper center',
              zorder=4,  # 设置另一个表格的图层顺序为4
              )
    plt.yticks([])  # 设置y轴刻度为空


@image_comparison(['table_labels.png'])
def test_label_colours():
    dim = 3

    c = np.linspace(0, 1, dim)
    colours = plt.cm.RdYlGn(c)
    cellText = [['1'] * dim] * dim

    fig = plt.figure()

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.axis('off')
    ax1.table(cellText=cellText,
              rowColours=colours,
              loc='best')  # 在最佳位置添加带颜色的表格

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.axis('off')
    ax2.table(cellText=cellText,
              rowColours=colours,
              rowLabels=['Header'] * dim,
              loc='best')  # 在最佳位置添加带行标签和颜色的表格

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.axis('off')
    ax3.table(cellText=cellText,
              colColours=colours,
              loc='best')  # 在最佳位置添加带列颜色的表格

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.axis('off')
    ax4.table(cellText=cellText,
              colColours=colours,
              colLabels=['Header'] * dim,
              loc='best')  # 在最佳位置添加带列标签和颜色的表格


@image_comparison(['table_cell_manipulation.png'], remove_text=True)
def test_diff_cell_table():
    cells = ('horizontal', 'vertical', 'open', 'closed', 'T', 'R', 'B', 'L')
    cellText = [['1'] * len(cells)] * 2
    colWidths = [0.1] * len(cells)

    _, axs = plt.subplots(nrows=len(cells), figsize=(4, len(cells)+1))
    for ax, cell in zip(axs, cells):
        ax.table(
                colWidths=colWidths,
                cellText=cellText,
                loc='center',
                edges=cell,
                )
        ax.axis('off')
    plt.tight_layout()


def test_customcell():
    types = ('horizontal', 'vertical', 'open', 'closed', 'T', 'R', 'B', 'L')
    # 定义一个元组 codes，包含多个元组，每个元组代表一组路径指令
    codes = (
        (Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO),
        (Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO),
        (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO),
        (Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY),
        (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO),
        (Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.MOVETO),
        (Path.MOVETO, Path.LINETO, Path.MOVETO, Path.MOVETO, Path.MOVETO),
        (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.LINETO),
    )

    # 遍历 types 和 codes 的元组，并进行比较
    for t, c in zip(types, codes):
        # 创建一个 CustomCell 对象，定义其可见边、宽度和高度
        cell = CustomCell((0, 0), visible_edges=t, width=1, height=1)
        # 从 CustomCell 对象的路径中提取出路径指令序列，组成元组 code
        code = tuple(s for _, s in cell.get_path().iter_segments())
        # 断言路径指令序列与预期的代码序列 c 相等
        assert c == code
@image_comparison(['table_auto_column.png'])
def test_auto_column():
    fig = plt.figure()

    # 创建子图ax1，位置为(4行, 1列, 第1个位置)
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.axis('off')
    # 在ax1中创建表格tb1，设置单元格文本、行标签、列标签和位置
    tb1 = ax1.table(
        cellText=[['Fit Text', 2],
                  ['very long long text, Longer text than default', 1]],
        rowLabels=["A", "B"],
        colLabels=["Col1", "Col2"],
        loc="center")
    tb1.auto_set_font_size(False)
    tb1.set_fontsize(12)
    # 自动设置列宽度
    tb1.auto_set_column_width([-1, 0, 1])

    # 创建子图ax2，位置为(4行, 1列, 第2个位置)
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.axis('off')
    # 在ax2中创建表格tb2，设置单元格文本、行标签、列标签和位置
    tb2 = ax2.table(
        cellText=[['Fit Text', 2],
                  ['very long long text, Longer text than default', 1]],
        rowLabels=["A", "B"],
        colLabels=["Col1", "Col2"],
        loc="center")
    tb2.auto_set_font_size(False)
    tb2.set_fontsize(12)
    # 自动设置列宽度
    tb2.auto_set_column_width((-1, 0, 1))

    # 创建子图ax3，位置为(4行, 1列, 第3个位置)
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.axis('off')
    # 在ax3中创建表格tb3，设置单元格文本、行标签、列标签和位置
    tb3 = ax3.table(
        cellText=[['Fit Text', 2],
                  ['very long long text, Longer text than default', 1]],
        rowLabels=["A", "B"],
        colLabels=["Col1", "Col2"],
        loc="center")
    tb3.auto_set_font_size(False)
    tb3.set_fontsize(12)
    # 逐列自动设置列宽度
    tb3.auto_set_column_width(-1)
    tb3.auto_set_column_width(0)
    tb3.auto_set_column_width(1)

    # 创建子图ax4，位置为(4行, 1列, 第4个位置)
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.axis('off')
    # 在ax4中创建表格tb4，设置单元格文本、行标签、列标签和位置
    tb4 = ax4.table(
        cellText=[['Fit Text', 2],
                  ['very long long text, Longer text than default', 1]],
        rowLabels=["A", "B"],
        colLabels=["Col1", "Col2"],
        loc="center")
    tb4.auto_set_font_size(False)
    tb4.set_fontsize(12)
    # 使用警告测试检查自动设置列宽时的异常情况
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match="'col' must be an int or sequence of ints"):
        tb4.auto_set_column_width("-101")  # type: ignore [arg-type]
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match="'col' must be an int or sequence of ints"):
        tb4.auto_set_column_width(["-101"])  # type: ignore [list-item]


def test_table_cells():
    fig, ax = plt.subplots()
    table = Table(ax)

    # 添加自定义单元格到表格，并进行断言确认
    cell = table.add_cell(1, 2, 1, 1)
    assert isinstance(cell, CustomCell)
    assert cell is table[1, 2]

    # 创建自定义单元格并设置到表格，进行断言确认
    cell2 = CustomCell((0, 0), 1, 2, visible_edges=None)
    table[2, 1] = cell2
    assert table[2, 1] is cell2

    # 确保表格的getitem支持未被破坏，并调用properties和setp方法
    table.properties()
    plt.setp(table)


@check_figures_equal(extensions=["png"])
def test_table_bbox(fig_test, fig_ref):
    data = [[2, 3],
            [4, 5]]

    col_labels = ('Foo', 'Bar')
    row_labels = ('Ada', 'Bob')

    cell_text = [[f"{x}" for x in row] for row in data]

    # 创建子图列表
    ax_list = fig_test.subplots()
    # 在 ax_list 对象上创建一个表格，并设置其参数
    ax_list.table(cellText=cell_text,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center',
                  bbox=[0.1, 0.2, 0.8, 0.6]
                  )

    # 在 fig_ref 对象的子图上创建一个表格，并设置其参数
    ax_bbox = fig_ref.subplots()
    ax_bbox.table(cellText=cell_text,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center',
                  bbox=Bbox.from_extents(0.1, 0.2, 0.9, 0.8)
                  )
@check_figures_equal(extensions=['png'])
def test_table_unit(fig_test, fig_ref):
    # 定义一个测试函数，用于验证表格是否正确处理单位机制，而不是使用repr/str方法

    class FakeUnit:
        def __init__(self, thing):
            pass
        
        def __repr__(self):
            return "Hello"

    # 创建一个假的单位转换器对象
    fake_convertor = munits.ConversionInterface()

    # 设置convert方法的模拟实现，返回固定值0
    fake_convertor.convert = Mock(side_effect=lambda v, u, a: 0)

    # 设置default_units方法的模拟实现，返回None
    fake_convertor.default_units = Mock(side_effect=lambda v, a: None)

    # 设置axisinfo方法的模拟实现，返回AxisInfo对象
    fake_convertor.axisinfo = Mock(side_effect=lambda u, a: munits.AxisInfo())

    # 将FakeUnit类注册到munits.registry中，对应的转换器是fake_convertor
    munits.registry[FakeUnit] = fake_convertor

    # 创建测试用的数据，包含FakeUnit对象和其他数据类型
    data = [[FakeUnit("yellow"), FakeUnit(42)],
            [FakeUnit(datetime.datetime(1968, 8, 1)), FakeUnit(True)]]

    # 在测试图表(fig_test)上创建子图，并在子图上创建表格，填充测试数据
    fig_test.subplots().table(data)

    # 在参考图表(fig_ref)上创建子图，并在子图上创建表格，填充预期数据("Hello"字符串)
    fig_ref.subplots().table([["Hello", "Hello"], ["Hello", "Hello"]])

    # 绘制测试图表的画布
    fig_test.canvas.draw()

    # 检查fake_convertor.convert方法未被调用
    fake_convertor.convert.assert_not_called()

    # 从munits.registry中移除FakeUnit类
    munits.registry.pop(FakeUnit)

    # 断言munits.registry中不再存在FakeUnit类的转换器
    assert not munits.registry.get_converter(FakeUnit)
```