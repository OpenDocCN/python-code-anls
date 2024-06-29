# `D:\src\scipysrc\pandas\scripts\validate_rst_title_capitalization.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import re  # 正则表达式库，用于文本匹配操作
import sys  # 系统相关的功能模块
from typing import TYPE_CHECKING  # 强类型提示检查

if TYPE_CHECKING:
    from collections.abc import Iterable  # 强类型提示：导入Iterable接口用于类型检查

# 特殊的小写单词集合，用于标题校验例外
CAPITALIZATION_EXCEPTIONS = {
    "pandas", "pd", "Python", "IPython", "PyTables", "Excel", "JSON", "HTML", "SAS",
    "SQL", "BigQuery", "STATA", "Interval", "IntervalArray", "PEP8", "Period", "Series",
    "Index", "DataFrame", "DataFrames", "C", "Git", "GitHub", "NumPy", "Apache", "Arrow",
    "Parquet", "MultiIndex", "NumFOCUS", "sklearn", "Docker", "PeriodIndex", "NA", "NaN",
    "NaT", "ValueError", "Boolean", "BooleanArray", "KeyError", "API", "FAQ", "IO",
    "Timedelta", "TimedeltaIndex", "DatetimeIndex", "IntervalIndex", "Categorical",
    "CategoricalIndex", "GroupBy", "DataFrameGroupBy", "SeriesGroupBy", "SPSS", "ORC", "R",
    "HDF5", "HDFStore", "CDay", "CBMonthBegin", "CBMonthEnd", "BMonthBegin", "BMonthEnd",
    "BDay", "FY5253Quarter", "FY5253", "YearBegin", "YearEnd", "BYearBegin", "BYearEnd",
    "YearOffset", "QuarterBegin", "QuarterEnd", "BQuarterBegin", "BQuarterEnd",
    "QuarterOffset", "LastWeekOfMonth", "WeekOfMonth", "SemiMonthBegin", "SemiMonthEnd",
    "SemiMonthOffset", "CustomBusinessMonthBegin", "CustomBusinessMonthEnd",
    "BusinessMonthBegin", "BusinessMonthEnd", "MonthBegin", "MonthEnd", "MonthOffset",
    "CustomBusinessHour", "CustomBusinessDay", "BusinessHour", "BusinessDay", "DateOffset",
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "Float64Index", "FloatIndex", "TZ", "GIL", "strftime",
    "XPORT", "Unicode", "East", "Asian", "None", "URLs", "UInt64", "SciPy", "Matplotlib",
    "PyPy", "SparseDataFrame", "Google", "CategoricalDtype", "UTC", "False", "Styler", "os",
    "str", "msgpack", "ExtensionArray", "LZMA", "Numba", "Timestamp", "PyArrow", "Gitpod",
    "Liveserve", "I", "VSCode"
}

# 创建小写例外单词到原始单词的映射字典
CAP_EXCEPTIONS_DICT = {word.lower(): word for word in CAPITALIZATION_EXCEPTIONS}

# 错误信息字符串
err_msg = "Heading capitalization formatted incorrectly. Please correctly capitalize"

# 标题中允许的符号字符
symbols = ("*", "=", "-", "^", "~", "#", '"')

def correct_title_capitalization(title: str) -> str:
    """
    # 算法用于创建给定标题的正确大写形式。

    # 如果标题以“:”开头，则无论如何跳过修改，以排除用于构建链接的特定语法。
    if title[0] == ":":
        return title

    # 从标题开头到第一个单词字符之前删除所有非单词字符。
    correct_title: str = re.sub(r"^\W*", "", title).capitalize()

    # 从标题中移除 URL。我们这样做是因为 URL 中的单词必须保持小写，即使它们是大写的例外情况。
    removed_https_title = re.sub(r"<https?:\/\/.*[\r\n]*>", "", correct_title)

    # 使用非单词字符作为分隔符将标题拆分为列表。
    word_list = re.split(r"\W", removed_https_title)

    # 遍历标题中的单词列表。
    for word in word_list:
        # 如果单词在 CAP_EXCEPTIONS_DICT 中，则替换为其大写例外值。
        if word.lower() in CAP_EXCEPTIONS_DICT:
            correct_title = re.sub(
                rf"\b{word}\b", CAP_EXCEPTIONS_DICT[word.lower()], correct_title
            )

    # 返回正确的标题
    return correct_title
def find_titles(rst_file: str) -> Iterable[tuple[str, int]]:
    """
    Algorithm to identify particular text that should be considered headings in an
    RST file.

    See <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html> for details
    on what constitutes a string as a heading in RST.

    Parameters
    ----------
    rst_file : str
        RST file to scan through for headings.

    Yields
    -------
    title : str
        A heading found in the rst file.

    line_number : int
        The corresponding line number of the heading.
    """

    # 打开指定的 RST 文件，使用 utf-8 编码读取
    with open(rst_file, encoding="utf-8") as fd:
        previous_line = ""  # 初始化前一行为空字符串
        # 遍历文件中的每一行，同时获取行号 i
        for i, line in enumerate(fd):
            line_no_last_elem = line[:-1]  # 去掉行末尾的换行符
            line_chars = set(line_no_last_elem)  # 获取行中出现的字符集合
            # 判断当前行是否符合特定条件，表示可能是标题行
            if (
                len(line_chars) == 1
                and line_chars.pop() in symbols
                and len(line_no_last_elem) == len(previous_line)
            ):
                # 如果符合条件，使用正则表达式去除特定字符后，作为标题输出
                yield re.sub(r"[`\*_]", "", previous_line), i
            previous_line = line_no_last_elem  # 更新前一行内容为当前行内容


def main(source_paths: list[str]) -> int:
    """
    The main method to print all headings with incorrect capitalization.

    Parameters
    ----------
    source_paths : str
        List of directories to validate, provided through command line arguments.

    Returns
    -------
    int
        Number of incorrect headings found overall.
    """

    number_of_errors: int = 0  # 初始化错误计数器为零

    # 遍历所有给定的源路径
    for filename in source_paths:
        # 遍历每个文件中的标题和行号
        for title, line_number in find_titles(filename):
            # 检查标题的大写是否正确，如果不正确则输出错误信息
            if title != correct_title_capitalization(title):
                # 打印错误信息，包括文件名、行号和修改后的标题
                print(
                    f"""{filename}:{line_number}:{err_msg} "{title}" to "{
                    correct_title_capitalization(title)}" """
                )
                number_of_errors += 1  # 错误计数器加一

    return number_of_errors  # 返回总的错误数


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate heading capitalization")

    parser.add_argument(
        "paths", nargs="*", help="Source paths of file/directory to check."
    )

    args = parser.parse_args()  # 解析命令行参数
    sys.exit(main(args.paths))  # 执行主函数并退出程序
```