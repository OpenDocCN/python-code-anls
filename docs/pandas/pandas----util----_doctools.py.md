# `D:\src\scipysrc\pandas\pandas\util\_doctools.py`

```
# 从未来版本导入类型注解
from __future__ import annotations

# 如果是类型检查，则导入类型检查相关的模块
from typing import TYPE_CHECKING

# 导入 numpy 库并重命名为 np
import numpy as np

# 导入 pandas 库并重命名为 pd
import pandas as pd

# 如果是类型检查，则导入 Iterable 和 Figure 类
if TYPE_CHECKING:
    from collections.abc import Iterable
    from matplotlib.figure import Figure

# 定义一个类 TablePlotter，用于布局和绘制 DataFrames，用于 merging.rst 文档
class TablePlotter:
    """
    Layout some DataFrames in vertical/horizontal layout for explanation.
    Used in merging.rst
    """

    # 初始化方法，设置单元格宽度、高度和字体大小，默认为 0.37、0.25 和 7.5
    def __init__(
        self,
        cell_width: float = 0.37,
        cell_height: float = 0.25,
        font_size: float = 7.5,
    ) -> None:
        self.cell_width = cell_width  # 设置单元格宽度
        self.cell_height = cell_height  # 设置单元格高度
        self.font_size = font_size  # 设置字体大小

    # 内部方法，计算 DataFrame 的形状（行数和列数），考虑索引级别
    def _shape(self, df: pd.DataFrame) -> tuple[int, int]:
        """
        Calculate table shape considering index levels.
        """
        row, col = df.shape  # 获取 DataFrame 的行数和列数
        return row + df.columns.nlevels, col + df.index.nlevels  # 返回考虑索引级别后的行数和列数

    # 内部方法，根据左右数据计算适当的图形大小
    def _get_cells(self, left, right, vertical) -> tuple[int, int]:
        """
        Calculate appropriate figure size based on left and right data.
        """
        if vertical:
            # 如果是垂直布局，计算所需的单元格数
            vcells = max(sum(self._shape(df)[0] for df in left), self._shape(right)[0])  # 计算垂直方向上的单元格数
            hcells = max(self._shape(df)[1] for df in left) + self._shape(right)[1]  # 计算水平方向上的单元格数
        else:
            # 如果是水平布局，计算所需的单元格数
            vcells = max([self._shape(df)[0] for df in left] + [self._shape(right)[0]])  # 计算垂直方向上的单元格数
            hcells = sum([self._shape(df)[1] for df in left] + [self._shape(right)[1]])  # 计算水平方向上的单元格数
        return hcells, vcells  # 返回水平和垂直方向上的单元格数

    # 绘制方法，用于绘制左右数据的表格，支持标签和垂直/水平布局
    def plot(
        self, left, right, labels: Iterable[str] = (), vertical: bool = True
    ):
    ) -> Figure:
        """
        Plot left / right DataFrames in specified layout.

        Parameters
        ----------
        left : list of DataFrames before operation is applied
            List of DataFrames to be plotted on the left side of the figure.
        right : DataFrame of operation result
            DataFrame to be plotted on the right side of the figure.
        labels : list of str to be drawn as titles of left DataFrames
            List of strings representing titles for each DataFrame in `left`.
        vertical : bool, default True
            If True, use vertical layout. If False, use horizontal layout.
        """
        from matplotlib import gridspec
        import matplotlib.pyplot as plt

        if not isinstance(left, list):
            left = [left]
        # Convert each DataFrame in `left` using `_conv` method
        left = [self._conv(df) for df in left]
        # Convert `right` DataFrame using `_conv` method
        right = self._conv(right)

        # Determine number of horizontal and vertical cells in the figure
        hcells, vcells = self._get_cells(left, right, vertical)

        if vertical:
            figsize = self.cell_width * hcells, self.cell_height * vcells
        else:
            # Calculate figure size with margin for titles
            figsize = self.cell_width * hcells, self.cell_height * vcells
        # Create a new figure with calculated `figsize`
        fig = plt.figure(figsize=figsize)

        if vertical:
            # Create a grid layout for vertical arrangement
            gs = gridspec.GridSpec(len(left), hcells)
            # Plot each DataFrame in `left` on the left side of the figure
            max_left_cols = max(self._shape(df)[1] for df in left)
            max_left_rows = max(self._shape(df)[0] for df in left)
            for i, (_left, _label) in enumerate(zip(left, labels)):
                # Add subplot for each DataFrame in `left`
                ax = fig.add_subplot(gs[i, 0:max_left_cols])
                # Create a table for the current DataFrame
                self._make_table(ax, _left, title=_label, height=1.0 / max_left_rows)
            # Plot `right` DataFrame on the right side of the figure
            ax = plt.subplot(gs[:, max_left_cols:])
            self._make_table(ax, right, title="Result", height=1.05 / vcells)
            # Adjust subplot parameters to fit the figure
            fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
        else:
            # Create a grid layout for horizontal arrangement
            max_rows = max(self._shape(df)[0] for df in left + [right])
            height = 1.0 / np.max(max_rows)
            gs = gridspec.GridSpec(1, hcells)
            i = 0
            # Plot each DataFrame in `left` horizontally
            for df, _label in zip(left, labels):
                sp = self._shape(df)
                ax = fig.add_subplot(gs[0, i : i + sp[1]])
                self._make_table(ax, df, title=_label, height=height)
                i += sp[1]
            # Plot `right` DataFrame at the end of the horizontal layout
            ax = plt.subplot(gs[0, i:])
            self._make_table(ax, right, title="Result", height=height)
            # Adjust subplot parameters to fit the figure
            fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)

        # Return the created figure object
        return fig

    def _conv(self, data):
        """
        Convert each input to appropriate format for table plotting.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input data to be converted.

        Returns
        -------
        pd.DataFrame
            Converted DataFrame suitable for plotting.
        """
        if isinstance(data, pd.Series):
            if data.name is None:
                data = data.to_frame(name="")
            else:
                data = data.to_frame()
        # Fill NaN values in the DataFrame with "NaN"
        data = data.fillna("NaN")
        return data
    # 定义一个私有方法 _insert_index，用于处理数据插入操作
    def _insert_index(self, data):
        # insert is destructive
        # 复制数据，因为 insert 操作是破坏性的
        data = data.copy()
        # 获取索引的层级数
        idx_nlevels = data.index.nlevels
        # 如果索引只有一级
        if idx_nlevels == 1:
            # 在第一列插入名为 "Index" 的索引列
            data.insert(0, "Index", data.index)
        else:
            # 多级索引情况下，逐级插入索引值作为新列
            for i in range(idx_nlevels):
                data.insert(i, f"Index{i}", data.index._get_level_values(i))

        # 获取列的层级数
        col_nlevels = data.columns.nlevels
        # 如果列数大于 1
        if col_nlevels > 1:
            # 获取第一级列名和其他级别的值
            col = data.columns._get_level_values(0)
            values = [
                data.columns._get_level_values(i)._values for i in range(1, col_nlevels)
            ]
            # 创建包含多级列的 DataFrame
            col_df = pd.DataFrame(values)
            data.columns = col_df.columns  # 设置新的列名
            data = pd.concat([col_df, data])  # 拼接 DataFrame
            data.columns = col  # 恢复原始的列名

        return data  # 返回处理后的 DataFrame

    # 定义一个私有方法 _make_table，用于在给定的坐标轴上创建表格
    def _make_table(self, ax, df, title: str, height: float | None = None) -> None:
        # 如果 DataFrame 为空，则隐藏坐标轴并返回
        if df is None:
            ax.set_visible(False)
            return

        from pandas import plotting

        # 获取索引和列的层级数，并调用 _insert_index 方法处理 DataFrame
        idx_nlevels = df.index.nlevels
        col_nlevels = df.columns.nlevels
        df = self._insert_index(df)

        # 在给定的坐标轴上创建表格，并设置表格字体大小
        tb = plotting.table(ax, df, loc=9)
        tb.set_fontsize(self.font_size)

        # 如果未指定高度，则计算默认行高
        if height is None:
            height = 1.0 / (len(df) + 1)

        # 获取表格属性，并根据索引和列的层级数进行单元格颜色设置和隐藏操作
        props = tb.properties()
        for (r, c), cell in props["celld"].items():
            if c == -1:
                cell.set_visible(False)  # 隐藏最后一列
            elif r < col_nlevels and c < idx_nlevels:
                cell.set_visible(False)  # 隐藏索引与列之间的单元格
            elif r < col_nlevels or c < idx_nlevels:
                cell.set_facecolor("#AAAAAA")  # 设置部分单元格背景颜色
            cell.set_height(height)  # 设置单元格高度

        ax.set_title(title, size=self.font_size)  # 设置坐标轴标题和字体大小
        ax.axis("off")  # 关闭坐标轴显示
# 主函数入口，程序的执行从这里开始
def main() -> None:
    # 导入 matplotlib 库中的 pyplot 模块，并重命名为 plt
    import matplotlib.pyplot as plt

    # 创建 TablePlotter 类的实例 p
    p = TablePlotter()

    # 创建两个 DataFrame 对象 df1 和 df2，分别包含不同的列和数据
    df1 = pd.DataFrame({"A": [10, 11, 12], "B": [20, 21, 22], "C": [30, 31, 32]})
    df2 = pd.DataFrame({"A": [10, 12], "C": [30, 32]})

    # 调用 p 对象的 plot 方法，绘制图表
    # 参数包括一个 DataFrame 列表 [df1, df2]，以及它们的合并结果 pd.concat([df1, df2])
    # labels 参数指定每个 DataFrame 的标签，vertical=True 表示垂直显示
    p.plot([df1, df2], pd.concat([df1, df2]), labels=["df1", "df2"], vertical=True)
    
    # 显示绘制的图表
    plt.show()

    # 创建第三个 DataFrame 对象 df3，包含不同的列和数据
    df3 = pd.DataFrame({"X": [10, 12], "Z": [30, 32]})

    # 再次调用 p 对象的 plot 方法，绘制第二个图表
    # 参数包括一个 DataFrame 列表 [df1, df3]，以及它们的水平合并结果 pd.concat([df1, df3], axis=1)
    # labels 参数指定每个 DataFrame 的标签，vertical=False 表示水平显示
    p.plot(
        [df1, df3], pd.concat([df1, df3], axis=1), labels=["df1", "df2"], vertical=False
    )
    
    # 显示第二个绘制的图表
    plt.show()

    # 创建 MultiIndex 对象 idx 和 column
    idx = pd.MultiIndex.from_tuples(
        [(1, "A"), (1, "B"), (1, "C"), (2, "A"), (2, "B"), (2, "C")]
    )
    column = pd.MultiIndex.from_tuples([(1, "A"), (1, "B")])
    
    # 创建第三个 DataFrame 对象 df3，包含不同的列、数据和 MultiIndex
    df3 = pd.DataFrame({"v1": [1, 2, 3, 4, 5, 6], "v2": [5, 6, 7, 8, 9, 10]}, index=idx)
    df3.columns = column
    
    # 调用 p 对象的 plot 方法，绘制第三个图表
    # 参数包括 df3 作为两个表的数据源
    # labels 参数指定每个 DataFrame 的标签
    p.plot(df3, df3, labels=["df3"])
    
    # 显示第三个绘制的图表
    plt.show()

# 如果脚本作为主程序执行，则调用 main 函数
if __name__ == "__main__":
    main()
```