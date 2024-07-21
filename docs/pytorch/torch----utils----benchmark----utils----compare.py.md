# `.\pytorch\torch\utils\benchmark\utils\compare.py`

```
# mypy: allow-untyped-defs
"""Display class to aggregate and print the results of many measurements."""
# 导入必要的模块和库
import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple

# 导入 torch 相关模块
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
import operator

# ANSI 控制字符，用于终端输出不同颜色和格式的文本
__all__ = ["Colorize", "Compare"]

BEST = "\033[92m"       # 绿色，代表最佳结果
GOOD = "\033[34m"       # 蓝色，代表好的结果
BAD = "\033[2m\033[91m" # 红色，代表较差的结果
VERY_BAD = "\033[31m"   # 红色，代表很差的结果
BOLD = "\033[1m"        # 粗体
TERMINATE = "\033[0m"   # 结束ANSI控制，重置文本格式

# 枚举类 Colorize，定义颜色化方式
class Colorize(enum.Enum):
    NONE = "none"
    COLUMNWISE = "columnwise"
    ROWWISE = "rowwise"

# 用于内部管理的类，分离内部数据处理和渲染逻辑
class _Column:
    def __init__(
        self,
        grouped_results: List[Tuple[Optional[common.Measurement], ...]], # 分组的测量结果列表
        time_scale: float,         # 时间缩放比例
        time_unit: str,            # 时间单位
        trim_significant_figures: bool,  # 是否修剪有效数字
        highlight_warnings: bool,  # 是否高亮显示警告
    ):
        self._grouped_results = grouped_results   # 分组后的测量结果
        self._flat_results = list(it.chain(*grouped_results))  # 扁平化的测量结果列表
        self._time_scale = time_scale     # 时间缩放比例
        self._time_unit = time_unit       # 时间单位
        self._trim_significant_figures = trim_significant_figures   # 是否修剪有效数字
        self._highlight_warnings = (
            highlight_warnings
            and any(r.has_warnings for r in self._flat_results if r)
        )   # 是否高亮显示警告，条件是有警告存在且高亮开启

        # 计算显示模板的相关参数
        leading_digits = [
            int(_tensor(r.median / self._time_scale).log10().ceil()) if r else None
            for r in self._flat_results
        ]
        unit_digits = max(d for d in leading_digits if d is not None)
        decimal_digits = min(
            max(m.significant_figures - digits, 0)
            for digits, m in zip(leading_digits, self._flat_results)
            if (m is not None) and (digits is not None)
        ) if self._trim_significant_figures else 1
        length = unit_digits + decimal_digits + (1 if decimal_digits else 0)
        self._template = f"{{:>{length}.{decimal_digits}f}}{{:>{7 if self._highlight_warnings else 0}}}"

    def get_results_for(self, group):
        return self._grouped_results[group]    # 返回指定分组的测量结果列表

    def num_to_str(self, value: Optional[float], estimated_sigfigs: int, spread: Optional[float]):
        if value is None:
            return " " * len(self.num_to_str(1, estimated_sigfigs, None))

        if self._trim_significant_figures:
            value = common.trim_sigfig(value, estimated_sigfigs)  # 根据有效数字修剪测量值

        return self._template.format(
            value,
            f" (! {spread * 100:.0f}%)" if self._highlight_warnings and spread is not None else ""
        )   # 格式化数字和可能的警告信息

# 计算可选序列的最小值，返回最小值或 None
def optional_min(seq):
    l = list(seq)
    return None if len(l) == 0 else min(l)

# 用于内部管理的类，分离内部数据处理和渲染逻辑
class _Row:
    # 初始化方法，接受多个参数以设置对象的初始状态
    def __init__(self, results, row_group, render_env, env_str_len,
                 row_name_str_len, time_scale, colorize, num_threads=None):
        # 调用父类的初始化方法
        super().__init__()
        # 将结果存储到对象的私有属性中
        self._results = results
        # 存储行组信息到对象的私有属性中
        self._row_group = row_group
        # 存储渲染环境到对象的私有属性中
        self._render_env = render_env
        # 存储环境字符串长度到对象的私有属性中
        self._env_str_len = env_str_len
        # 存储行名字符串长度到对象的私有属性中
        self._row_name_str_len = row_name_str_len
        # 存储时间比例尺到对象的私有属性中
        self._time_scale = time_scale
        # 存储颜色化选项到对象的私有属性中
        self._colorize = colorize
        # 初始化空元组，用于存储列对象
        self._columns: Tuple[_Column, ...] = ()
        # 存储线程数到对象的私有属性中，可选参数
        self._num_threads = num_threads

    # 注册列方法，接受一个元组参数，将其存储到对象的私有属性中
    def register_columns(self, columns: Tuple[_Column, ...]):
        self._columns = columns

    # 将结果以列字符串的形式返回的方法
    def as_column_strings(self):
        # 过滤出非空结果的列表
        concrete_results = [r for r in self._results if r is not None]
        # 根据渲染环境选择是否使用环境字符串
        env = f"({concrete_results[0].env})" if self._render_env else ""
        # 将环境字符串左对齐到指定长度加四个空格
        env = env.ljust(self._env_str_len + 4)
        # 初始化输出列表，包含环境字符串和第一个结果的行名
        output = ["  " + env + concrete_results[0].as_row_name]
        # 遍历结果和列对象，生成每列的字符串表示并加入输出列表
        for m, col in zip(self._results, self._columns or ()):
            if m is None:
                # 如果结果为空，使用列对象的方法生成空值的字符串表示
                output.append(col.num_to_str(None, 1, None))
            else:
                # 否则，使用列对象的方法生成结果的字符串表示
                output.append(col.num_to_str(
                    m.median / self._time_scale,
                    m.significant_figures,
                    m.iqr / m.median if m.has_warnings else None
                ))
        return output

    # 静态方法，根据值和最佳值返回对应的着色段
    @staticmethod
    def color_segment(segment, value, best_value):
        if value <= best_value * 1.01 or value <= best_value + 100e-9:
            return BEST + BOLD + segment + TERMINATE * 2
        if value <= best_value * 1.1:
            return GOOD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 5:
            return VERY_BAD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 2:
            return BAD + segment + TERMINATE * 2

        return segment

    # 行分隔方法，根据整体宽度返回包含线程数信息的列表
    def row_separator(self, overall_width):
        return (
            [f"{self._num_threads} threads: ".ljust(overall_width, "-")]
            if self._num_threads is not None else []
        )
    def finalize_column_strings(self, column_strings, col_widths):
        # 初始化一个列表，用于存储每列最佳值的占位符
        best_values = [-1 for _ in column_strings]
        
        # 如果颜色设置为按行着色
        if self._colorize == Colorize.ROWWISE:
            # 计算所有结果中的中位数的最小值
            row_min = min(r.median for r in self._results if r is not None)
            # 将所有列的最佳值设置为该最小值
            best_values = [row_min for _ in column_strings]
        
        # 如果颜色设置为按列着色
        elif self._colorize == Colorize.COLUMNWISE:
            # 遍历每一列，获取该列在当前行组中的结果，并计算中位数的最小值
            best_values = [
                optional_min(r.median for r in column.get_results_for(self._row_group) if r is not None)
                for column in (self._columns or ())
            ]

        # 初始化行内容列表，包含第一列的字符串，左对齐到指定宽度
        row_contents = [column_strings[0].ljust(col_widths[0])]
        
        # 遍历除第一列外的其他列的字符串、对应的宽度、结果和最佳值
        for col_str, width, result, best_value in zip(column_strings[1:], col_widths[1:], self._results, best_values):
            # 将列字符串居中到指定宽度
            col_str = col_str.center(width)
            # 如果颜色设置不为无且结果和最佳值都不为空
            if self._colorize != Colorize.NONE and result is not None and best_value is not None:
                # 对列字符串进行颜色分段处理，基于结果的中位数和最佳值
                col_str = self.color_segment(col_str, result.median, best_value)
            # 将处理后的列字符串添加到行内容列表中
            row_contents.append(col_str)
        
        # 返回最终的行内容列表
        return row_contents
# 定义名为 Table 的类，用于展示测量结果的表格
class Table:
    # 初始化方法，接受多个参数来初始化对象
    def __init__(
            self,
            results: List[common.Measurement],  # 包含 common.Measurement 对象的列表，表示测量结果
            colorize: Colorize,  # 用于着色的对象
            trim_significant_figures: bool,  # 是否修剪有效数字
            highlight_warnings: bool  # 是否突出显示警告
    ):
        # 断言所有结果的标签相同
        assert len({r.label for r in results}) == 1

        # 将参数赋值给对象属性
        self.results = results
        self._colorize = colorize
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = highlight_warnings
        self.label = results[0].label  # 设置表格的标签为第一个测量结果的标签
        # 选择时间单位和时间规模，基于结果中所有测量的中位数
        self.time_unit, self.time_scale = common.select_unit(
            min(r.median for r in results)
        )

        # 生成行键列表，基于每个结果对象通过 row_fn 方法转换得到的元组
        self.row_keys = common.ordered_unique([self.row_fn(i) for i in results])
        # 根据切片(2)的方式对行键列表进行排序，以保留语句的顺序
        self.row_keys.sort(key=operator.itemgetter(slice(2)))
        # 生成列键列表，基于每个结果对象通过 col_fn 方法转换得到的字符串
        self.column_keys = common.ordered_unique([self.col_fn(i) for i in results])
        # 调用方法来填充表格的行和列数据
        self.rows, self.columns = self.populate_rows_and_columns()

    # 静态方法，根据 common.Measurement 对象返回一个元组，用于作为行的键
    @staticmethod
    def row_fn(m: common.Measurement) -> Tuple[int, Optional[str], str]:
        return m.num_threads, m.env, m.as_row_name

    # 静态方法，根据 common.Measurement 对象返回一个可选的描述字符串，用作列的键
    @staticmethod
    def col_fn(m: common.Measurement) -> Optional[str]:
        return m.description
    # 定义方法，用于填充行和列数据，并返回行和列的元组
    def populate_rows_and_columns(self) -> Tuple[Tuple[_Row, ...], Tuple[_Column, ...]]:
        # 初始化行和列的空列表
        rows: List[_Row] = []
        columns: List[_Column] = []
        
        # 初始化有序结果列表，使用 None 填充
        ordered_results: List[List[Optional[common.Measurement]]] = [
            [None for _ in self.column_keys]  # 每一行使用 None 初始化
            for _ in self.row_keys  # 根据行的数量进行迭代
        ]
        
        # 创建行和列的位置字典，映射关系为 键: 索引
        row_position = {key: i for i, key in enumerate(self.row_keys)}
        col_position = {key: i for i, key in enumerate(self.column_keys)}
        
        # 将结果按照行和列的索引位置填充到有序结果列表中
        for r in self.results:
            i = row_position[self.row_fn(r)]  # 获取行索引
            j = col_position[self.col_fn(r)]  # 获取列索引
            ordered_results[i][j] = r
        
        # 确定结果中唯一的环境值集合
        unique_envs = {r.env for r in self.results}
        render_env = len(unique_envs) > 1  # 判断是否需要渲染环境信息
        env_str_len = max(len(i) for i in unique_envs) if render_env else 0  # 计算环境信息的最大长度
        
        # 计算结果中行名称的最大长度
        row_name_str_len = max(len(r.as_row_name) for r in self.results)
        
        # 初始化先前的线程数、环境和行组索引
        prior_num_threads = -1
        prior_env = ""
        row_group = -1
        rows_by_group: List[List[List[Optional[common.Measurement]]]] = []
        
        # 遍历行键和有序结果列表，并创建行对象
        for (num_threads, env, _), row in zip(self.row_keys, ordered_results):
            thread_transition = (num_threads != prior_num_threads)
            
            # 如果线程数发生变化，则更新先前的线程数和环境信息，并增加行组索引
            if thread_transition:
                prior_num_threads = num_threads
                prior_env = ""
                row_group += 1
                rows_by_group.append([])
            
            # 创建行对象，并添加到行列表中
            rows.append(
                _Row(
                    results=row,
                    row_group=row_group,
                    render_env=(render_env and env != prior_env),
                    env_str_len=env_str_len,
                    row_name_str_len=row_name_str_len,
                    time_scale=self.time_scale,
                    colorize=self._colorize,
                    num_threads=num_threads if thread_transition else None,
                )
            )
            
            # 将行添加到当前行组的列表中
            rows_by_group[-1].append(row)
            prior_env = env  # 更新先前的环境信息
        
        # 遍历列键列表，并根据行组数据创建列对象
        for i in range(len(self.column_keys)):
            grouped_results = [tuple(row[i] for row in g) for g in rows_by_group]
            column = _Column(
                grouped_results=grouped_results,
                time_scale=self.time_scale,
                time_unit=self.time_unit,
                trim_significant_figures=self._trim_significant_figures,
                highlight_warnings=self._highlight_warnings,
            )
            columns.append(column)  # 将列对象添加到列列表中
        
        # 将行和列列表转换为元组
        rows_tuple, columns_tuple = tuple(rows), tuple(columns)
        
        # 为每行注册列对象
        for ri in rows_tuple:
            ri.register_columns(columns_tuple)
        
        # 返回行和列的元组
        return rows_tuple, columns_tuple
    # 渲染表格并返回其字符串表示
    def render(self) -> str:
        # 创建一个二维列表，第一行为表头，包含一个空字符串和所有列的键名
        string_rows = [[""] + self.column_keys]
        
        # 遍历每一行数据对象，将其转换为字符串列表，并加入到二维列表中
        for r in self.rows:
            string_rows.append(r.as_column_strings())
        
        # 计算最大列数
        num_cols = max(len(i) for i in string_rows)
        
        # 对每行数据列表进行扩展，使其长度与最大列数相等，用空字符串填充
        for sr in string_rows:
            sr.extend(["" for _ in range(num_cols - len(sr))])

        # 计算每列的最大宽度
        col_widths = [max(len(j) for j in i) for i in zip(*string_rows)]
        
        # 将每行的列数据根据列宽度进行居中处理，并用"  |  "连接，形成最终的列字符串
        finalized_columns = ["  |  ".join(i.center(w) for i, w in zip(string_rows[0], col_widths))]
        
        # 计算总体宽度
        overall_width = len(finalized_columns[0])
        
        # 遍历每一行数据对象，对每一行进行分隔处理，并将处理后的结果加入到最终列字符串中
        for string_row, row in zip(string_rows[1:], self.rows):
            finalized_columns.extend(row.row_separator(overall_width))
            finalized_columns.append("  |  ".join(row.finalize_column_strings(string_row, col_widths)))

        # 创建一个换行符
        newline = "\n"
        
        # 检查是否需要高亮显示警告，如果需要且任何结果对象中存在警告，则设置为True
        has_warnings = self._highlight_warnings and any(ri.has_warnings for ri in self.results)
        
        # 返回格式化的表格字符串，包括前面的空行和表格数据部分
        return f"""
[{(' ' + (self.label or '') + ' ').center(overall_width - 2, '-')}]
# 创建一个列表，包含一个字符串，该字符串根据 self.label 的内容居中显示在一个指定宽度的横线中间，横线宽度为 overall_width - 2。

{newline.join(finalized_columns)}
# 将 finalized_columns 中的每个元素用 newline 连接成一个字符串。

Times are in {common.unit_to_english(self.time_unit)}s ({self.time_unit}).
# 打印一条消息，显示时间单位和单位名称，使用 common.unit_to_english 方法将单位转换为英文。

{'(! XX%) Measurement has high variance, where XX is the IQR / median * 100.' + newline if has_warnings else ""}
# 如果 has_warnings 为 True，则返回一个描述测量具有高方差的警告信息的字符串，其中 XX 是 IQR / median * 100；否则返回空字符串。
"""[1:]
# 从上述字符串的第二个字符（索引为1）开始，截取到结尾，忽略开头的三个引号。
    # 定义一个私有方法 `_layout`，接收一个名为 `results` 的列表参数，列表中的元素是 `common.Measurement` 类型的对象
    def _layout(self, results: List[common.Measurement]):
        # 创建一个 Table 对象，用于展示结果
        table = Table(
            results,  # 将 results 列表传递给 Table 对象，用于填充表格内容
            self._colorize,  # 调用 self._colorize 方法，用于为表格内容着色
            self._trim_significant_figures,  # 调用 self._trim_significant_figures 方法，用于修剪表格内容的有效数字位数
            self._highlight_warnings  # 调用 self._highlight_warnings 方法，用于突出显示表格中的警告信息
        )
        # 调用 Table 对象的 render 方法，生成最终的表格展示结果，并返回
        return table.render()
```