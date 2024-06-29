# `D:\src\scipysrc\pandas\pandas\core\config_init.py`

```
"""
This module is imported from the pandas package __init__.py file
in order to ensure that the core.config options registered here will
be available as soon as the user loads the package. if register_option
is invoked inside specific modules, they will not be registered until that
module is imported, which may or may not be a problem.

If you need to make sure options are available even before a certain
module is imported, register them here rather than in the module.

"""

from __future__ import annotations  # 导入模块的 annotations 特性，用于类型提示

from collections.abc import Callable  # 导入 Callable 类型
import os  # 导入操作系统接口

import pandas._config.config as cf  # 导入 pandas 配置模块
from pandas._config.config import (  # 导入以下函数和工厂函数
    is_bool,
    is_callable,
    is_instance_factory,
    is_int,
    is_nonnegative_int,
    is_one_of_factory,
    is_str,
    is_text,
)

# compute

use_bottleneck_doc = """
: bool
    Use the bottleneck library to accelerate if it is installed,
    the default is True
    Valid values: False,True
"""

def use_bottleneck_cb(key: str) -> None:
    from pandas.core import nanops

    nanops.set_use_bottleneck(cf.get_option(key))  # 设置使用 bottleneck 库的选项

use_numexpr_doc = """
: bool
    Use the numexpr library to accelerate computation if it is installed,
    the default is True
    Valid values: False,True
"""

def use_numexpr_cb(key: str) -> None:
    from pandas.core.computation import expressions

    expressions.set_use_numexpr(cf.get_option(key))  # 设置使用 numexpr 库的选项

use_numba_doc = """
: bool
    Use the numba engine option for select operations if it is installed,
    the default is False
    Valid values: False,True
"""

def use_numba_cb(key: str) -> None:
    from pandas.core.util import numba_

    numba_.set_use_numba(cf.get_option(key))  # 设置使用 numba 引擎的选项

with cf.config_prefix("compute"):
    cf.register_option(
        "use_bottleneck",
        True,
        use_bottleneck_doc,
        validator=is_bool,
        cb=use_bottleneck_cb,
    )
    cf.register_option(
        "use_numexpr", True, use_numexpr_doc, validator=is_bool, cb=use_numexpr_cb
    )
    cf.register_option(
        "use_numba", False, use_numba_doc, validator=is_bool, cb=use_numba_cb
    )
#
# options from the "display" namespace

pc_precision_doc = """
: int
    Floating point output precision in terms of number of places after the
    decimal, for regular formatting as well as scientific notation. Similar
    to ``precision`` in :meth:`numpy.set_printoptions`.
"""

pc_max_rows_doc = """
: int
    If max_rows is exceeded, switch to truncate view. Depending on
    `large_repr`, objects are either centrally truncated or printed as
    a summary view. 'None' value means unlimited.

    In case python/IPython is running in a terminal and `large_repr`
    equals 'truncate' this can be set to 0 and pandas will auto-detect
    the height of the terminal and print a truncated object which fits
    the screen height. The IPython notebook, IPython qtconsole, or
    IDLE do not run in a terminal and hence it is not possible to do
    correct auto-detection.
"""

pc_min_rows_doc = """
: int
"""
    The numbers of rows to show in a truncated view (when `max_rows` is
    exceeded). Ignored when `max_rows` is set to None or 0. When set to
    None, follows the value of `max_rows`.
"""

pc_max_cols_doc = """
: int
    如果超过 max_cols，则切换到截断视图。根据 `large_repr` 的设置，对象要么居中截断，要么以摘要视图打印。'None' 表示无限制。

    如果 Python/IPython 在终端中运行且 `large_repr` 设置为 'truncate'，则可以将其设置为 0 或 None，pandas 将自动检测终端的宽度并打印适合屏幕宽度的截断对象。IPython 笔记本、IPython qtconsole 或 IDLE 不在终端运行，因此无法进行正确的自动检测，默认为 20。
"""

pc_max_categories_doc = """
: int
    设置 pandas 在打印 `Categorical` 或 dtype 为 "category" 的 Series 时应输出的最大类别数。
"""

pc_max_info_cols_doc = """
: int
    max_info_columns 用于 DataFrame.info 方法，决定是否打印每列的信息。
"""

pc_nb_repr_h_doc = """
: boolean
    当为 True 时，IPython 笔记本将使用 HTML 表示来显示 pandas 对象（如果可用）。
"""

pc_pprint_nest_depth = """
: int
    控制在漂亮打印时要处理的嵌套级别数。
"""

pc_multi_sparse_doc = """
: boolean
    控制是否“稀疏化”MultiIndex显示（在组内不显示外部级别中的重复元素）。
"""

float_format_doc = """
: callable
    这个可调用对象应接受一个浮点数并返回希望的数值格式的字符串。在一些地方如 SeriesFormatter 中使用。
    参见 formats.format.EngFormatter 以获取示例。
"""

max_colwidth_doc = """
: int or None
    在 pandas 数据结构的 repr 中，列的最大宽度（以字符数表示）。当列溢出时，输出中嵌入一个 "..." 占位符。'None' 表示无限制。
"""

colheader_justify_doc = """
: 'left'/'right'
    控制列标题的对齐方式。由 DataFrameFormatter 使用。
"""

pc_expand_repr_doc = """
: boolean
    是否为宽数据框打印完整的 DataFrame repr，可以跨多行展示，但仍遵循 `max_columns` 的限制，如果宽度超过 `display.width`，输出将会换行显示多个“页”。
"""

pc_show_dimensions_doc = """
: boolean or 'truncate'
    是否在 DataFrame 的 repr 结尾处打印维度信息。如果指定为 'truncate'，则只在 DataFrame 被截断时（例如未显示所有行和/或列时）打印维度。
"""

pc_east_asian_width_doc = """
: boolean
    是否使用 Unicode 的东亚宽度来计算显示文本的宽度。
    启用此选项可能会影响性能（默认为 False）。
"""

pc_table_schema_doc = """
: boolean
    是否为支持的前端发布表模式表示。
    （默认为 False）
"""

pc_html_border_doc = """
: int
    DataFrame 的 HTML repr 中 `<table>` 标签中插入的 `border=value` 属性。
"""
"""
pc_html_use_mathjax_doc = """\
: boolean
    When True, Jupyter notebook will process table contents using MathJax,
    rendering mathematical expressions enclosed by the dollar symbol.
    (default: True)
"""
"""

pc_max_dir_items = """\
: int
    The number of items that will be added to `dir(...)`. 'None' value means
    unlimited. Because dir is cached, changing this option will not immediately
    affect already existing dataframes until a column is deleted or added.

    This is for instance used to suggest columns from a dataframe to tab
    completion.
"""
"""

pc_width_doc = """
: int
    Width of the display in characters. In case python/IPython is running in
    a terminal this can be set to None and pandas will correctly auto-detect
    the width.
    Note that the IPython notebook, IPython qtconsole, or IDLE do not run in a
    terminal and hence it is not possible to correctly detect the width.
"""
"""

pc_chop_threshold_doc = """
: float or None
    if set to a float value, all float values smaller than the given threshold
    will be displayed as exactly 0 by repr and friends.
"""
"""

pc_max_seq_items = """
: int or None
    When pretty-printing a long sequence, no more then `max_seq_items`
    will be printed. If items are omitted, they will be denoted by the
    addition of "..." to the resulting string.

    If set to None, the number of items to be printed is unlimited.
"""
"""

pc_max_info_rows_doc = """
: int
    df.info() will usually show null-counts for each column.
    For large frames this can be quite slow. max_info_rows and max_info_cols
    limit this null check only to frames with smaller dimensions than
    specified.
"""
"""

pc_large_repr_doc = """
: 'truncate'/'info'
    For DataFrames exceeding max_rows/max_cols, the repr (and HTML repr) can
    show a truncated table, or switch to the view from
    df.info() (the behaviour in earlier versions of pandas).
"""
"""

pc_memory_usage_doc = """
: bool, string or None
    This specifies if the memory usage of a DataFrame should be displayed when
    df.info() is called. Valid values True,False,'deep'
"""


def table_schema_cb(key: str) -> None:
    """
    Enable a specific data resource formatter for the display option.

    :param key: The key to configure the display option.
    """
    from pandas.io.formats.printing import enable_data_resource_formatter

    enable_data_resource_formatter(cf.get_option(key))


def is_terminal() -> bool:
    """
    Detect if Python is running in a terminal.

    Returns True if Python is running in a terminal or False if not.
    """
    try:
        # error: Name 'get_ipython' is not defined
        ip = get_ipython()  # type: ignore[name-defined]
    except NameError:  # assume standard Python interpreter in a terminal
        return True
    else:
        if hasattr(ip, "kernel"):  # IPython as a Jupyter kernel
            return False
        else:  # IPython in a terminal
            return True


with cf.config_prefix("display"):
    """
    Register the 'precision' option under the 'display' configuration prefix.

    This option controls the number of decimal places displayed for floating point numbers.
    """
    cf.register_option("precision", 6, pc_precision_doc, validator=is_nonnegative_int)
    # 注册 "float_format" 选项，没有默认值，使用自定义的浮点格式文档
    cf.register_option(
        "float_format",
        None,
        float_format_doc,
        validator=is_one_of_factory([None, is_callable]),
    )
    # 注册 "max_info_rows" 选项，默认值为 1690785，使用预定义的最大信息行文档，验证器为整数
    cf.register_option(
        "max_info_rows",
        1690785,
        pc_max_info_rows_doc,
        validator=is_int,
    )
    # 注册 "max_rows" 选项，默认值为 60，使用预定义的最大行数文档，验证器为非负整数
    cf.register_option("max_rows", 60, pc_max_rows_doc, validator=is_nonnegative_int)
    # 注册 "min_rows" 选项，默认值为 10，使用预定义的最小行数文档，验证器接受 None 或整数类型
    cf.register_option(
        "min_rows",
        10,
        pc_min_rows_doc,
        validator=is_instance_factory((type(None), int)),
    )
    # 注册 "max_categories" 选项，默认值为 8，使用预定义的最大类别数文档，验证器为整数
    cf.register_option("max_categories", 8, pc_max_categories_doc, validator=is_int)

    # 注册 "max_colwidth" 选项，默认值为 50，使用预定义的最大列宽文档，验证器为非负整数
    cf.register_option(
        "max_colwidth",
        50,
        max_colwidth_doc,
        validator=is_nonnegative_int,
    )
    # 如果是在终端环境下，自动确定最优列数为 0；否则，设置最大列数为 20
    if is_terminal():
        max_cols = 0  # automatically determine optimal number of columns
    else:
        max_cols = 20  # cannot determine optimal number of columns
    # 注册 "max_columns" 选项，使用最优列数变量 max_cols，使用预定义的最大列数文档，验证器为非负整数
    cf.register_option(
        "max_columns", max_cols, pc_max_cols_doc, validator=is_nonnegative_int
    )
    # 注册 "large_repr" 选项，默认值为 "truncate"，使用预定义的大型表示文档，验证器接受 "truncate" 或 "info"
    cf.register_option(
        "large_repr",
        "truncate",
        pc_large_repr_doc,
        validator=is_one_of_factory(["truncate", "info"]),
    )
    # 注册 "max_info_columns" 选项，默认值为 100，使用预定义的最大信息列数文档，验证器为整数
    cf.register_option("max_info_columns", 100, pc_max_info_cols_doc, validator=is_int)
    # 注册 "colheader_justify" 选项，默认值为 "right"，使用预定义的列标题对齐文档，验证器为文本类型
    cf.register_option(
        "colheader_justify", "right", colheader_justify_doc, validator=is_text
    )
    # 注册 "notebook_repr_html" 选项，默认值为 True，使用预定义的 Notebook HTML 表示文档，验证器为布尔类型
    cf.register_option("notebook_repr_html", True, pc_nb_repr_h_doc, validator=is_bool)
    # 注册 "pprint_nest_depth" 选项，默认值为 3，使用预定义的 pretty print 嵌套深度文档，验证器为整数
    cf.register_option("pprint_nest_depth", 3, pc_pprint_nest_depth, validator=is_int)
    # 注册 "multi_sparse" 选项，默认值为 True，使用预定义的多重稀疏表示文档，验证器为布尔类型
    cf.register_option("multi_sparse", True, pc_multi_sparse_doc, validator=is_bool)
    # 注册 "expand_frame_repr" 选项，默认值为 True，使用预定义的扩展数据框表示文档
    cf.register_option("expand_frame_repr", True, pc_expand_repr_doc)
    # 注册 "show_dimensions" 选项，默认值为 "truncate"，使用预定义的显示维度文档，验证器接受 True、False 或 "truncate"
    cf.register_option(
        "show_dimensions",
        "truncate",
        pc_show_dimensions_doc,
        validator=is_one_of_factory([True, False, "truncate"]),
    )
    # 注册 "chop_threshold" 选项，默认值为 None，使用预定义的裁剪阈值文档
    cf.register_option("chop_threshold", None, pc_chop_threshold_doc)
    # 注册 "max_seq_items" 选项，默认值为 100，使用预定义的最大序列项文档
    cf.register_option("max_seq_items", 100, pc_max_seq_items)
    # 注册 "width" 选项，默认值为 80，使用预定义的宽度文档，验证器接受 None 或整数类型
    cf.register_option(
        "width", 80, pc_width_doc, validator=is_instance_factory((type(None), int))
    )
    # 注册 "memory_usage" 选项，默认值为 True，使用预定义的内存使用文档，验证器接受 None、True、False 或 "deep"
    cf.register_option(
        "memory_usage",
        True,
        pc_memory_usage_doc,
        validator=is_one_of_factory([None, True, False, "deep"]),
    )
    # 注册 "unicode.east_asian_width" 选项，默认值为 False，使用预定义的 Unicode 东亚宽度文档，验证器为布尔类型
    cf.register_option(
        "unicode.east_asian_width", False, pc_east_asian_width_doc, validator=is_bool
    )
    # 注册 "unicode.ambiguous_as_wide" 选项，默认值为 False，使用预定义的 Unicode 模糊字符作为宽字符文档，验证器为布尔类型
    cf.register_option(
        "unicode.ambiguous_as_wide", False, pc_east_asian_width_doc, validator=is_bool
    )
    # 注册 "html.table_schema" 选项，默认值为 False，使用预定义的 HTML 表格结构文档，验证器为布尔类型，带有回调函数 table_schema_cb
    cf.register_option(
        "html.table_schema",
        False,
        pc_table_schema_doc,
        validator=is_bool,
        cb=table_schema_cb,
    )
    # 注册 "html.border" 选项，默认值为 1，使用预定义的 HTML 边框文档，验证器为整数类型
    cf.register_option("html.border", 1, pc_html_border_doc, validator=is_int)
    # 注册 "html.use_mathjax" 选项，默认值为 True，使用预定义的 HTML 使用 MathJax 文档，验证器为布尔类型
    cf.register_option(
        "html.use_mathjax", True, pc_html_use_mathjax_doc, validator=is_bool
    )
    # 注册 "max_dir_items" 选项，默认值为 100，使用预定义的最大目录项文档，验证器为非负整数
    cf.register_option(
        "max_dir_items", 100, pc_max_dir_items, validator=is_nonnegative_int
    )
# 定义一个文档字符串，描述了一个布尔类型的配置选项，指示是否模拟交互模式以进行测试目的
tc_sim_interactive_doc = """
: boolean
    Whether to simulate interactive mode for purposes of testing
"""

# 使用上下文管理器注册一个名为 "mode" 的配置前缀，为名为 "sim_interactive" 的选项注册默认值和文档字符串
with cf.config_prefix("mode"):
    cf.register_option("sim_interactive", False, tc_sim_interactive_doc)


# TODO better name?
# 定义一个文档字符串，描述了一个布尔类型的配置选项，指示是否启用新的Copy-on-Write的复制视图行为，默认为False
copy_on_write_doc = """
: bool
    Use new copy-view behaviour using Copy-on-Write. Defaults to False,
    unless overridden by the 'PANDAS_COPY_ON_WRITE' environment variable
    (if set to "1" for True, needs to be set before pandas is imported).
"""

# 使用上下文管理器注册一个名为 "mode" 的配置前缀，为名为 "copy_on_write" 的选项注册默认值和文档字符串，
# 根据环境变量 'PANDAS_COPY_ON_WRITE' 的设置来决定默认值，并指定一个验证器函数
with cf.config_prefix("mode"):
    cf.register_option(
        "copy_on_write",
        # Get the default from an environment variable, if set, otherwise defaults
        # to False. This environment variable can be set for testing.
        "warn"
        if os.environ.get("PANDAS_COPY_ON_WRITE", "0") == "warn"
        else os.environ.get("PANDAS_COPY_ON_WRITE", "0") == "1",
        copy_on_write_doc,
        validator=is_one_of_factory([True, False, "warn"]),
    )


# user warnings
# 定义一个文档字符串，描述了一个字符串类型的配置选项，用于指定尝试使用链式赋值时的处理方式，默认为警告
chained_assignment = """
: string
    Raise an exception, warn, or no action if trying to use chained assignment,
    The default is warn
"""

# 使用上下文管理器注册一个名为 "mode" 的配置前缀，为名为 "chained_assignment" 的选项注册默认值和文档字符串，
# 并指定一个验证器函数
with cf.config_prefix("mode"):
    cf.register_option(
        "chained_assignment",
        "warn",
        chained_assignment,
        validator=is_one_of_factory([None, "warn", "raise"]),
    )


# 定义一个文档字符串，描述了一个布尔类型的配置选项，指示是否显示或隐藏性能警告
performance_warnings = """
: boolean
    Whether to show or hide PerformanceWarnings.
"""

# 使用上下文管理器注册一个名为 "mode" 的配置前缀，为名为 "performance_warnings" 的选项注册默认值和文档字符串，
# 并指定一个验证器函数
with cf.config_prefix("mode"):
    cf.register_option(
        "performance_warnings",
        True,
        performance_warnings,
        validator=is_bool,
    )


# 定义一个文档字符串，描述了一个字符串类型的配置选项，指定StringDtype的默认存储方式
# 如果 'future.infer_string' 设置为 True，则此选项被忽略
string_storage_doc = """
: string
    The default storage for StringDtype. This option is ignored if
    ``future.infer_string`` is set to True.
"""

# 使用上下文管理器注册一个名为 "mode" 的配置前缀，为名为 "string_storage" 的选项注册默认值和文档字符串，
# 并指定一个验证器函数
with cf.config_prefix("mode"):
    cf.register_option(
        "string_storage",
        "python",
        string_storage_doc,
        validator=is_one_of_factory(["python", "pyarrow", "pyarrow_numpy"]),
    )


# 定义一个文档字符串，描述了一个字符串类型的配置选项，指定Excel读取器引擎的默认选择
reader_engine_doc = """
: string
    The default Excel reader engine for '{ext}' files. Available options:
    auto, {others}.
"""

# 定义不同扩展名的可选读取器引擎列表
_xls_options = ["xlrd", "calamine"]
_xlsm_options = ["xlrd", "openpyxl", "calamine"]
_xlsx_options = ["xlrd", "openpyxl", "calamine"]
_ods_options = ["odf", "calamine"]
_xlsb_options = ["pyxlsb", "calamine"]

# 使用上下文管理器注册一个名为 "io.excel.xls" 的配置前缀，为名为 "reader" 的选项注册默认值和文档字符串，
# 并指定一个验证器函数
with cf.config_prefix("io.excel.xls"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xls", others=", ".join(_xls_options)),
        validator=is_one_of_factory(_xls_options + ["auto"]),
    )

# 使用上下文管理器注册一个名为 "io.excel.xlsm" 的配置前缀，为名为 "reader" 的选项注册默认值和文档字符串，
# 并指定一个验证器函数
with cf.config_prefix("io.excel.xlsm"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xlsm", others=", ".join(_xlsm_options)),
        validator=is_one_of_factory(_xlsm_options + ["auto"]),
    )

# 继续下一个代码块的注册
with cf.config_prefix("io.excel.xlsx"):
    # 注册一个名为 "reader" 的配置选项到配置工厂 cf 中，初始值为 "auto"
    cf.register_option(
        "reader",
        "auto",
        # 使用格式化字符串将特定的文档模板应用到 reader 引擎的说明文档中，其中包括 xlsx 扩展名和其他选项的列表
        reader_engine_doc.format(ext="xlsx", others=", ".join(_xlsx_options)),
        # 使用 is_one_of_factory 函数创建一个验证器，验证器的功能是确保选项值是 _xlsx_options 和 "auto" 中的一个
        validator=is_one_of_factory(_xlsx_options + ["auto"]),
    )
# 使用特定前缀配置'io.excel.ods'，设置相关选项
with cf.config_prefix("io.excel.ods"):
    # 注册名为'reader'的选项，设置默认值为'auto'，包含有关'ods'文件的读取引擎文档和验证器
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="ods", others=", ".join(_ods_options)),
        validator=is_one_of_factory(_ods_options + ["auto"]),
    )

# 使用特定前缀配置'io.excel.xlsb'，设置相关选项
with cf.config_prefix("io.excel.xlsb"):
    # 注册名为'reader'的选项，设置默认值为'auto'，包含有关'xlsb'文件的读取引擎文档和验证器
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xlsb", others=", ".join(_xlsb_options)),
        validator=is_one_of_factory(_xlsb_options + ["auto"]),
    )

# 设置'io.excel'特定的写入器配置
writer_engine_doc = """
: string
    默认的 Excel 写入引擎，用于 '{ext}' 文件。可用选项有：
    auto, {others}.
"""

# 定义不同类型的Excel文件可选的写入引擎选项
_xlsm_options = ["openpyxl"]
_xlsx_options = ["openpyxl", "xlsxwriter"]
_ods_options = ["odf"]

# 使用特定前缀配置'io.excel.xlsm'，设置相关选项
with cf.config_prefix("io.excel.xlsm"):
    # 注册名为'writer'的选项，设置默认值为'auto'，包含有关'xlsm'文件的写入引擎文档和验证器
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="xlsm", others=", ".join(_xlsm_options)),
        validator=str,
    )

# 使用特定前缀配置'io.excel.xlsx'，设置相关选项
with cf.config_prefix("io.excel.xlsx"):
    # 注册名为'writer'的选项，设置默认值为'auto'，包含有关'xlsx'文件的写入引擎文档和验证器
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="xlsx", others=", ".join(_xlsx_options)),
        validator=str,
    )

# 使用特定前缀配置'io.excel.ods'，设置相关选项
with cf.config_prefix("io.excel.ods"):
    # 注册名为'writer'的选项，设置默认值为'auto'，包含有关'ods'文件的写入引擎文档和验证器
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="ods", others=", ".join(_ods_options)),
        validator=str,
    )

# 设置'io.parquet'特定的配置
parquet_engine_doc = """
: string
    默认的 parquet 读写引擎。可用选项有：
    'auto', 'pyarrow', 'fastparquet'，默认为'auto'
"""

# 使用特定前缀配置'io.parquet'，设置相关选项
with cf.config_prefix("io.parquet"):
    # 注册名为'engine'的选项，设置默认值为'auto'，包含有关parquet文件的引擎文档和验证器
    cf.register_option(
        "engine",
        "auto",
        parquet_engine_doc,
        validator=is_one_of_factory(["auto", "pyarrow", "fastparquet"]),
    )

# 设置'io.sql'特定的配置
sql_engine_doc = """
: string
    默认的 SQL 读写引擎。可用选项有：
    'auto', 'sqlalchemy'，默认为'auto'
"""

# 使用特定前缀配置'io.sql'，设置相关选项
with cf.config_prefix("io.sql"):
    # 注册名为'engine'的选项，设置默认值为'auto'，包含有关SQL文件的引擎文档和验证器
    cf.register_option(
        "engine",
        "auto",
        sql_engine_doc,
        validator=is_one_of_factory(["auto", "sqlalchemy"]),
    )

# --------
# 绘图
# ---------

# 绘图后端文档说明
plotting_backend_doc = """
: str
    要使用的绘图后端。默认值为 "matplotlib"，pandas 提供的后端。可以通过提供实现后端的模块名称来指定其他后端。
"""

# 注册绘图后端选项
def register_plotting_backend_cb(key: str | None) -> None:
    if key == "matplotlib":
        # 对于默认的matplotlib后端，推迟验证
        return
    from pandas.plotting._core import _get_plot_backend

    _get_plot_backend(key)

# 使用特定前缀配置'plotting'，设置绘图后端选项
with cf.config_prefix("plotting"):
    cf.register_option(
        "backend",
        defval="matplotlib",
        doc=plotting_backend_doc,
        validator=register_plotting_backend_cb,  # type: ignore[arg-type]
    )

# 注册转换器选项文档
register_converter_doc = """
: bool 或 'auto'。
    Whether to register converters with matplotlib's units registry for
    dates, times, datetimes, and Periods. Toggling to False will remove
    the converters, restoring any converters that pandas overwrote.


    # 是否向 matplotlib 的单位注册表注册转换器，用于处理日期、时间、日期时间和周期。
    # 将其设置为 False 将移除这些转换器，恢复 pandas 可能覆盖的任何转换器。
"""


def register_converter_cb(key: str) -> None:
    from pandas.plotting import (
        deregister_matplotlib_converters,
        register_matplotlib_converters,
    )

    # 检查配置文件中的选项是否启用了Matplotlib转换器注册
    if cf.get_option(key):
        # 如果启用了，注册Matplotlib转换器
        register_matplotlib_converters()
    else:
        # 如果未启用，取消注册Matplotlib转换器
        deregister_matplotlib_converters()


with cf.config_prefix("plotting.matplotlib"):
    cf.register_option(
        "register_converters",
        "auto",
        register_converter_doc,
        validator=is_one_of_factory(["auto", True, False]),
        cb=register_converter_cb,
    )

# ------
# Styler
# ------

styler_sparse_index_doc = """
: bool
    Whether to sparsify the display of a hierarchical index. Setting to False will
    display each explicit level element in a hierarchical key for each row.
"""

styler_sparse_columns_doc = """
: bool
    Whether to sparsify the display of hierarchical columns. Setting to False will
    display each explicit level element in a hierarchical key for each column.
"""

styler_render_repr = """
: str
    Determine which output to use in Jupyter Notebook in {"html", "latex"}.
"""

styler_max_elements = """
: int
    The maximum number of data-cell (<td>) elements that will be rendered before
    trimming will occur over columns, rows or both if needed.
"""

styler_max_rows = """
: int, optional
    The maximum number of rows that will be rendered. May still be reduced to
    satisfy ``max_elements``, which takes precedence.
"""

styler_max_columns = """
: int, optional
    The maximum number of columns that will be rendered. May still be reduced to
    satisfy ``max_elements``, which takes precedence.
"""

styler_precision = """
: int
    The precision for floats and complex numbers.
"""

styler_decimal = """
: str
    The character representation for the decimal separator for floats and complex.
"""

styler_thousands = """
: str, optional
    The character representation for thousands separator for floats, int and complex.
"""

styler_na_rep = """
: str, optional
    The string representation for values identified as missing.
"""

styler_escape = """
: str, optional
    Whether to escape certain characters according to the given context; html or latex.
"""

styler_formatter = """
: str, callable, dict, optional
    A formatter object to be used as default within ``Styler.format``.
"""

styler_multirow_align = """
: {"c", "t", "b"}
    The specifier for vertical alignment of sparsified LaTeX multirows.
"""

styler_multicol_align = r"""
: {"r", "c", "l", "naive-l", "naive-r"}
    The specifier for horizontal alignment of sparsified LaTeX multicolumns. Pipe
    decorators can also be added to non-naive values to draw vertical
    rules, e.g. "\|r" will draw a rule on the left side of right aligned merged cells.
"""

styler_hrules = """
: bool
    Whether to add horizontal rules on top and bottom and below the headers.
"""

styler_environment = """
: str
    The environment to replace ``\\begin{table}``. If "longtable" is used results
    # 定义一个名为 read_latex_table 的函数，接收两个参数：文件名和列数
    def read_latex_table(filename, columns):
        # 打开指定文件，读取内容作为一个字符串
        with open(filename, 'r') as f:
            # 从文件中读取所有行并存储在变量 lines 中
            lines = f.readlines()
        
        # 使用列表解析将 lines 列表中的每一行转换为一个列表，每个列表代表一行中的单元格内容
        table = [line.strip().split('&') for line in lines]
        
        # 如果指定的列数大于每行的单元格数量，抛出 ValueError 异常
        if any(len(row) < columns for row in table):
            raise ValueError('Not enough columns in some rows of the table')
        
        # 返回处理后的表格数据，其中每一行由一个列表表示，每个列表中包含了相应的单元格内容
        return table
"""
Configuration for pandas-styling options using `cf` (presumably a configuration handler).
"""

# Register option for "styler.sparse.index" with default value True and documentation
cf.register_option("styler.sparse.index", True, styler_sparse_index_doc, validator=is_bool)

# Register option for "styler.sparse.columns" with default value True and documentation
cf.register_option("styler.sparse.columns", True, styler_sparse_columns_doc, validator=is_bool)

# Register option for "styler.render.repr" with default value "html", supporting "html" and "latex" outputs
cf.register_option("styler.render.repr", "html", styler_render_repr, validator=is_one_of_factory(["html", "latex"]))

# Register option for "styler.render.max_elements" with default value 2**18 (262144)
cf.register_option("styler.render.max_elements", 2**18, styler_max_elements, validator=is_nonnegative_int)

# Register option for "styler.render.max_rows" with default value None
cf.register_option("styler.render.max_rows", None, styler_max_rows, validator=is_nonnegative_int)

# Register option for "styler.render.max_columns" with default value None
cf.register_option("styler.render.max_columns", None, styler_max_columns, validator=is_nonnegative_int)

# Register option for "styler.render.encoding" with default value "utf-8"
cf.register_option("styler.render.encoding", "utf-8", styler_encoding, validator=is_str)

# Register option for "styler.format.decimal" with default value "."
cf.register_option("styler.format.decimal", ".", styler_decimal, validator=is_str)

# Register option for "styler.format.precision" with default value 6
cf.register_option("styler.format.precision", 6, styler_precision, validator=is_nonnegative_int)

# Register option for "styler.format.thousands" with default value None
cf.register_option("styler.format.thousands", None, styler_thousands, validator=is_instance_factory((type(None), str)))

# Register option for "styler.format.na_rep" with default value None
cf.register_option("styler.format.na_rep", None, styler_na_rep, validator=is_instance_factory((type(None), str)))

# Register option for "styler.format.escape" with default value None, supporting None, "html", "latex", "latex-math"
cf.register_option("styler.format.escape", None, styler_escape, validator=is_one_of_factory([None, "html", "latex", "latex-math"]))

# Register option for "styler.format.formatter" with default value None, supporting None, dict, Callable, str
cf.register_option("styler.format.formatter", None, styler_formatter, validator=is_instance_factory((type(None), dict, Callable, str)))

# Register option for "styler.html.mathjax" with default value True
cf.register_option("styler.html.mathjax", True, styler_mathjax, validator=is_bool)

# Register option for "styler.latex.multirow_align" with default value "c", supporting "c", "t", "b", "naive"
cf.register_option("styler.latex.multirow_align", "c", styler_multirow_align, validator=is_one_of_factory(["c", "t", "b", "naive"]))

# Define valid values for "styler.latex.multicol_align"
val_mca = ["r", "|r|", "|r", "r|", "c", "|c|", "|c", "c|", "l", "|l|", "|l", "l|"]
val_mca += ["naive-l", "naive-r"]

# Register option for "styler.latex.multicol_align" with default value "r", supporting predefined values
cf.register_option("styler.latex.multicol_align", "r", styler_multicol_align, validator=is_one_of_factory(val_mca))

# Register option for "styler.latex.hrules" with default value False
cf.register_option("styler.latex.hrules", False, styler_hrules, validator=is_bool)

# Register option for "styler.latex.environment" with default value None
cf.register_option("styler.latex.environment", None, styler_environment, validator=is_instance_factory((type(None), str)))
    # 注册配置选项 "infer_string"
    cf.register_option(
        "infer_string",
        False,
        "Whether to infer sequence of str objects as pyarrow string "
        "dtype, which will be the default in pandas 3.0 "
        "(at which point this option will be deprecated).",
        validator=is_one_of_factory([True, False]),
    )

    # 注册配置选项 "no_silent_downcasting"
    cf.register_option(
        "no_silent_downcasting",
        False,
        "Whether to opt-in to the future behavior which will *not* silently "
        "downcast results from Series and DataFrame `where`, `mask`, and `clip` "
        "methods. "
        "Silent downcasting will be removed in pandas 3.0 "
        "(at which point this option will be deprecated).",
        validator=is_one_of_factory([True, False]),
    )
```