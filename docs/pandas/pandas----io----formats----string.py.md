# `D:\src\scipysrc\pandas\pandas\io\formats\string.py`

```
"""
Module`
"""
Module for formatting output data in console (to string).
"""

from __future__ import annotations  # 导入未来的注解语法，确保类型注解兼容未来版本的 Python

from shutil import get_terminal_size  # 从 shutil 模块导入 get_terminal_size 函数，获取终端窗口大小
from typing import TYPE_CHECKING  # 从 typing 模块导入 TYPE_CHECKING，帮助在类型检查时引用类型

import numpy as np  # 导入 numpy 库

from pandas.io.formats.printing import pprint_thing  # 从 pandas 库的打印格式中导入 pprint_thing 函数，用于打印格式化输出

if TYPE_CHECKING:
    from collections.abc import Iterable  # 导入 Iterable，表示所有可迭代对象
    from pandas.io.formats.format import DataFrameFormatter  # 从 pandas 库的格式化模块中导入 DataFrameFormatter 类

class StringFormatter:
    """Formatter for string representation of a dataframe."""

    def __init__(self, fmt: DataFrameFormatter, line_width: int | None = None) -> None:
        # 初始化 StringFormatter 类，传入 DataFrameFormatter 实例和可选的行宽参数
        self.fmt = fmt  # 保存 DataFrameFormatter 实例
        self.adj = fmt.adj  # 保存对齐方式
        self.frame = fmt.frame  # 保存数据框架
        self.line_width = line_width  # 保存行宽参数

    def to_string(self) -> str:
        # 将数据框架转换为字符串形式
        text = self._get_string_representation()  # 获取数据框架的字符串表示
        if self.fmt.should_show_dimensions:
            text = f"{text}{self.fmt.dimensions_info}"  # 如果需要显示维度信息，添加维度信息
        return text

    def _get_strcols(self) -> list[list[str]]:
        # 获取字符串列，处理数据框架的字符串列
        strcols = self.fmt.get_strcols()  # 获取数据框架的字符串列
        if self.fmt.is_truncated:
            strcols = self._insert_dot_separators(strcols)  # 如果数据框架被截断，插入省略符
        return strcols

    def _get_string_representation(self) -> str:
        # 获取数据框架的字符串表示
        if self.fmt.frame.empty:
            return self._empty_info_line  # 如果数据框架为空，返回空信息行

        strcols = self._get_strcols()  # 获取字符串列

        if self.line_width is None:
            # 如果行宽未指定，直接返回整行数据框架的字符串表示
            return self.adj.adjoin(1, *strcols)

        if self._need_to_wrap_around:
            return self._join_multiline(strcols)  # 如果需要换行，返回多行字符串表示

        return self._fit_strcols_to_terminal_width(strcols)  # 返回适应终端宽度的字符串表示

    @property
    def _empty_info_line(self) -> str:
        # 返回空数据框架的字符串信息行
        return (
            f"Empty {type(self.frame).__name__}\n"  # 数据框架为空，显示其类型
            f"Columns: {pprint_thing(self.frame.columns)}\n"  # 显示列信息
            f"Index: {pprint_thing(self.frame.index)}"  # 显示索引信息
        )

    @property
    def _need_to_wrap_around(self) -> bool:
        # 判断是否需要换行
        return bool(self.fmt.max_cols is None or self.fmt.max_cols > 0)  # 如果最大列数未指定或大于0，返回 True

    def _insert_dot_separators(self, strcols: list[list[str]]) -> list[list[str]]:
        # 插入省略符，处理被截断的数据框架
        str_index = self.fmt._get_formatted_index(self.fmt.tr_frame)  # 获取格式化的索引
        index_length = len(str_index)  # 获取索引的长度

        if self.fmt.is_truncated_horizontally:
            strcols = self._insert_dot_separator_horizontal(strcols, index_length)  # 如果水平截断，插入水平省略符

        if self.fmt.is_truncated_vertically:
            strcols = self._insert_dot_separator_vertical(strcols, index_length)  # 如果垂直截断，插入垂直省略符

        return strcols

    @property
    def _adjusted_tr_col_num(self) -> int:
        # 获取调整后的列数
        return self.fmt.tr_col_num + 1 if self.fmt.index else self.fmt.tr_col_num  # 如果有索引，列数加1，否则为列数

    def _insert_dot_separator_horizontal(
        self, strcols: list[list[str]], index_length: int
    ) -> list[list[str]]:
        # 插入水平省略符
        strcols.insert(self._adjusted_tr_col_num, [" ..."] * index_length)  # 在指定位置插入省略符
        return strcols

    def _insert_dot_separator_vertical(
        self, strcols: list[list[str]], index_length: int
    ) -> list[list[str]]:
        # 插入垂直省略符
        # 省略符插入逻辑待实现
    # 定义一个方法 _join_multiline，接受一个 strcols_input 参数，类型为可迭代的列表，其中每个元素是一个字符串列表
    def _join_multiline(self, strcols_input: Iterable[list[str]]) -> str:
        # 获取当前对象的行宽度
        lwidth = self.line_width
        # 定义用于连接字符串列的宽度
        adjoin_width = 1
        # 将输入的 strcols_input 转换为列表 strcols
        strcols = list(strcols_input)

        # 如果格式中包含索引列
        if self.fmt.index:
            # 从 strcols 中弹出索引列，并计算剩余行宽度
            idx = strcols.pop(0)
            lwidth -= np.array([self.adj.len(x) for x in idx]).max() + adjoin_width

        # 计算每列的最大宽度，并将结果存储在 col_widths 中
        col_widths = [
            np.array([self.adj.len(x) for x in col]).max() if len(col) > 0 else 0
            for col in strcols
        ]

        # 断言确保 lwidth 不为 None
        assert lwidth is not None
        # 对列宽度进行分组，以便宽度不超过 lwidth
        col_bins = _binify(col_widths, lwidth)
        # 计算分组后的列数
        nbins = len(col_bins)

        # 初始化字符串列表 str_lst
        str_lst = []
        start = 0
        # 遍历列分组
        for i, end in enumerate(col_bins):
            # 获取当前分组的行
            row = strcols[start:end]
            # 如果有索引列，则在行首插入索引列
            if self.fmt.index:
                row.insert(0, idx)
            # 如果有多个列分组，则在末尾添加连接线
            if nbins > 1:
                nrows = len(row[-1])
                if end <= len(strcols) and i < nbins - 1:
                    row.append([" \\"] + ["  "] * (nrows - 1))
                else:
                    row.append([" "] * nrows)
            # 使用 self.adj.adjoin 方法连接当前行，并将结果添加到 str_lst 中
            str_lst.append(self.adj.adjoin(adjoin_width, *row))
            start = end

        # 将 str_lst 中的各个连接结果用两个换行符连接成最终的字符串返回
        return "\n\n".join(str_lst)
    # 定义一个方法 _fit_strcols_to_terminal_width，用于调整字符串列以适应终端宽度
    def _fit_strcols_to_terminal_width(self, strcols: list[list[str]]) -> str:
        # 导入 Series 类
        from pandas import Series

        # 使用 adj.adjoin 方法将 strcols 中的字符串列连接成一个字符串，然后按换行符分割为列表 lines
        lines = self.adj.adjoin(1, *strcols).split("\n")
        
        # 计算 lines 中最长行的长度
        max_len = Series(lines).str.len().max()
        
        # 获取终端的宽度和高度信息
        width, _ = get_terminal_size()
        
        # 计算最长行长度与终端宽度之间的差值
        dif = max_len - width
        
        # 为了避免过宽的表示（GH PR #17023），在 dif 上加 1
        adj_dif = dif + 1
        
        # 计算每列的最大长度
        col_lens = Series([Series(ele).str.len().max() for ele in strcols])
        
        # 获取列数
        n_cols = len(col_lens)
        
        # 初始化计数器
        counter = 0
        
        # 当调整差值大于 0 且列数大于 1 时进行循环
        while adj_dif > 0 and n_cols > 1:
            counter += 1
            
            # 计算中间列的索引和长度
            mid = round(n_cols / 2)
            mid_ix = col_lens.index[mid]
            col_len = col_lens[mid_ix]
            
            # adjoin 方法会添加一个字符，因此在 adj_dif 上减去列长度加 1
            adj_dif -= col_len + 1
            
            # 删除长度最大的中间列
            col_lens = col_lens.drop(mid_ix)
            
            # 更新列数
            n_cols = len(col_lens)

        # 减去索引列
        max_cols_fitted = n_cols - self.fmt.index
        
        # 确保至少打印两列（GH-21180）
        max_cols_fitted = max(max_cols_fitted, 2)
        
        # 将计算出的最大列数赋给 self.fmt.max_cols_fitted
        self.fmt.max_cols_fitted = max_cols_fitted

        # 再次调用 _truncate 方法以适当地截取帧
        self.fmt.truncate()
        
        # 获取字符串列
        strcols = self._get_strcols()
        
        # 使用 adj.adjoin 方法将调整后的字符串列连接成一个字符串并返回
        return self.adj.adjoin(1, *strcols)
# 定义一个函数 `_binify`，接收两个参数：列宽度列表 `cols` 和行宽度 `line_width`，返回一个整数列表
def _binify(cols: list[int], line_width: int) -> list[int]:
    # 设置相邻列之间的间隔宽度为 1
    adjoin_width = 1
    # 初始化一个空列表用来存储分组的索引
    bins = []
    # 初始化当前行已使用的宽度为 0
    curr_width = 0
    # 获取最后一列的索引
    i_last_column = len(cols) - 1
    
    # 遍历列宽度列表 `cols`
    for i, w in enumerate(cols):
        # 计算当前列加上相邻列间隔后的宽度
        w_adjoined = w + adjoin_width
        # 将当前列宽度加到当前行已使用宽度中
        curr_width += w_adjoined
        
        # 判断是否需要换行
        if i_last_column == i:
            # 如果是最后一列，则判断是否超过行宽度，且不是第一列
            wrap = curr_width + 1 > line_width and i > 0
        else:
            # 如果不是最后一列，则判断是否超过行宽度加上两个间隔宽度，且不是第一列
            wrap = curr_width + 2 > line_width and i > 0
        
        # 如果需要换行
        if wrap:
            # 将当前列索引加入到 bins 列表中
            bins.append(i)
            # 重置当前行已使用宽度为当前列加上相邻列间隔后的宽度
            curr_width = w_adjoined
    
    # 将最后一列的索引加入到 bins 列表中
    bins.append(len(cols))
    # 返回 bins 列表，其中包含需要换行的列索引
    return bins
```