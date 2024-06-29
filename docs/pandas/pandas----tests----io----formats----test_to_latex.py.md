# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_to_latex.py`

```
import codecs  # 导入codecs模块，用于处理文件编码
from datetime import datetime  # 导入datetime模块中的datetime类
from textwrap import dedent  # 导入textwrap模块中的dedent函数

import pytest  # 导入pytest测试框架

import pandas as pd  # 导入pandas库，并用pd作为别名
from pandas import (  # 从pandas中导入DataFrame和Series类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入pandas内部测试模块作为tm别名

pytest.importorskip("jinja2")  # 检查并导入jinja2模块，如果不存在则跳过


def _dedent(string):
    """Dedent without new line in the beginning.

    Built-in textwrap.dedent would keep new line character in the beginning
    of multi-line string starting from the new line.
    This version drops the leading new line character.
    """
    return dedent(string).lstrip()  # 对输入的字符串进行去缩进处理，去掉开头可能的空格和换行符


@pytest.fixture
def df_short():
    """Short dataframe for testing table/tabular/longtable LaTeX env."""
    return DataFrame({"a": [1, 2], "b": ["b1", "b2"]})  # 返回一个简短的DataFrame用于测试LaTeX环境的表格


class TestToLatex:
    def test_to_latex_to_file(self, float_frame):
        with tm.ensure_clean("test.tex") as path:  # 使用tm.ensure_clean确保在路径"test.tex"上的文件存在并且是干净的
            float_frame.to_latex(path)  # 将float_frame写入到指定路径的LaTeX文件中
            with open(path, encoding="utf-8") as f:  # 打开指定路径的文件，使用UTF-8编码
                assert float_frame.to_latex() == f.read()  # 断言生成的LaTeX内容与文件内容一致

    def test_to_latex_to_file_utf8_with_encoding(self):
        # test with utf-8 and encoding option (GH 7061)
        df = DataFrame([["au\xdfgangen"]])  # 创建一个包含特殊字符的DataFrame
        with tm.ensure_clean("test.tex") as path:  # 使用tm.ensure_clean确保在路径"test.tex"上的文件存在并且是干净的
            df.to_latex(path, encoding="utf-8")  # 将DataFrame写入到指定路径的LaTeX文件中，指定UTF-8编码
            with codecs.open(path, "r", encoding="utf-8") as f:  # 使用codecs打开文件，指定UTF-8编码
                assert df.to_latex() == f.read()  # 断言生成的LaTeX内容与文件内容一致

    def test_to_latex_to_file_utf8_without_encoding(self):
        # test with utf-8 without encoding option
        df = DataFrame([["au\xdfgangen"]])  # 创建一个包含特殊字符的DataFrame
        with tm.ensure_clean("test.tex") as path:  # 使用tm.ensure_clean确保在路径"test.tex"上的文件存在并且是干净的
            df.to_latex(path)  # 将DataFrame写入到指定路径的LaTeX文件中，默认使用系统编码（UTF-8）
            with codecs.open(path, "r", encoding="utf-8") as f:  # 使用codecs打开文件，指定UTF-8编码
                assert df.to_latex() == f.read()  # 断言生成的LaTeX内容与文件内容一致

    def test_to_latex_tabular_with_index(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})  # 创建一个DataFrame
        result = df.to_latex()  # 生成DataFrame的LaTeX表示
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )  # 预期的LaTeX表格内容，使用_dedent函数去除前导空格和换行符
        assert result == expected  # 断言生成的LaTeX内容与预期内容一致

    def test_to_latex_tabular_without_index(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})  # 创建一个DataFrame
        result = df.to_latex(index=False)  # 生成DataFrame的LaTeX表示，不包含索引
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            a & b \\
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )  # 预期的LaTeX表格内容，使用_dedent函数去除前导空格和换行符
        assert result == expected  # 断言生成的LaTeX内容与预期内容一致

    @pytest.mark.parametrize(
        "bad_column_format",
        [5, 1.2, ["l", "r"], ("r", "c"), {"r", "c", "l"}, {"a": "r", "b": "l"}],
    )
    def test_to_latex_bad_column_format(self, bad_column_format):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})  # 创建一个DataFrame
        msg = r"`column_format` must be str or unicode"  # 错误消息内容
        with pytest.raises(ValueError, match=msg):  # 使用pytest断言捕获ValueError，并匹配错误消息
            df.to_latex(column_format=bad_column_format)  # 尝试使用不合法的column_format参数调用to_latex方法
    # 测试函数，验证 DataFrame 的 to_latex 方法在指定 column_format 时的功能
    def test_to_latex_column_format_just_works(self, float_frame):
        # GH Bug #9402：验证在指定 column_format 为 "lcr" 时，to_latex 方法正常工作
        float_frame.to_latex(column_format="lcr")

    # 测试函数，验证 DataFrame 的 to_latex 方法在不同数据输入下生成正确的 LaTeX 表格
    def test_to_latex_column_format(self):
        # 创建一个 DataFrame 对象 df，包含两列 'a' 和 'b'
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 调用 df 的 to_latex 方法，指定 column_format 为 "lcr"，生成 LaTeX 格式的表格字符串
        result = df.to_latex(column_format="lcr")
        # 预期的 LaTeX 格式表格字符串，使用 _dedent 函数去除首尾空格和换行
        expected = _dedent(
            r"""
            \begin{tabular}{lcr}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期结果一致
        assert result == expected

    # 测试函数，验证 Series 的 to_latex 方法在包含不同类型数据时的表现
    def test_to_latex_float_format_object_col(self):
        # 创建一个 Series 对象 ser，包含一个浮点数和一个字符串
        ser = Series([1000.0, "test"])
        # 调用 ser 的 to_latex 方法，指定 float_format 为 "{:,.0f}"，生成 LaTeX 格式的表格字符串
        result = ser.to_latex(float_format="{:,.0f}".format)
        # 预期的 LaTeX 格式表格字符串，使用 _dedent 函数去除首尾空格和换行
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & 1,000 \\
            1 & test \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期结果一致
        assert result == expected

    # 测试函数，验证空 DataFrame 的 to_latex 方法生成正确的空表格
    def test_to_latex_empty_tabular(self):
        # 创建一个空的 DataFrame 对象 df
        df = DataFrame()
        # 调用 df 的 to_latex 方法，生成 LaTeX 格式的空表格字符串
        result = df.to_latex()
        # 预期的 LaTeX 格式空表格字符串，使用 _dedent 函数去除首尾空格和换行
        expected = _dedent(
            r"""
            \begin{tabular}{l}
            \toprule
            \midrule
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期结果一致
        assert result == expected

    # 测试函数，验证 Series 的 to_latex 方法在不同数据输入下生成正确的 LaTeX 表格
    def test_to_latex_series(self):
        # 创建一个 Series 对象 s，包含三个字符串元素
        s = Series(["a", "b", "c"])
        # 调用 s 的 to_latex 方法，生成 LaTeX 格式的表格字符串
        result = s.to_latex()
        # 预期的 LaTeX 格式表格字符串，使用 _dedent 函数去除首尾空格和换行
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & a \\
            1 & b \\
            2 & c \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期结果一致
        assert result == expected

    # 测试函数，验证 DataFrame 的 to_latex 方法在指定 index_names 时生成正确的 LaTeX 表格
    def test_to_latex_midrule_location(self):
        # 创建一个 DataFrame 对象 df，包含一列 'a'，索引名设置为 "foo"
        df = DataFrame({"a": [1, 2]})
        df.index.name = "foo"
        # 调用 df 的 to_latex 方法，指定 index_names 为 False，生成 LaTeX 格式的表格字符串
        result = df.to_latex(index_names=False)
        # 预期的 LaTeX 格式表格字符串，使用 _dedent 函数去除首尾空格和换行
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & a \\
            \midrule
            0 & 1 \\
            1 & 2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期结果一致
        assert result == expected
class TestToLatexLongtable:
    # 测试空的 DataFrame 转换为长表格 LaTeX 的情况
    def test_to_latex_empty_longtable(self):
        df = DataFrame()  # 创建一个空的 DataFrame 对象
        result = df.to_latex(longtable=True)  # 将 DataFrame 转换为长表格 LaTeX 字符串
        expected = _dedent(  # 生成预期的长表格 LaTeX 字符串
            r"""
            \begin{longtable}{l}
            \toprule
            \midrule
            \endfirsthead
            \toprule
            \midrule
            \endhead
            \midrule
            \multicolumn{0}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            \end{longtable}
            """
        )
        assert result == expected  # 断言实际输出与预期输出一致

    # 测试带索引的 DataFrame 转换为长表格 LaTeX 的情况
    def test_to_latex_longtable_with_index(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})  # 创建带有两列的 DataFrame 对象
        result = df.to_latex(longtable=True)  # 将 DataFrame 转换为长表格 LaTeX 字符串
        expected = _dedent(  # 生成预期的长表格 LaTeX 字符串
            r"""
            \begin{longtable}{lrl}
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected  # 断言实际输出与预期输出一致

    # 测试不带索引的 DataFrame 转换为长表格 LaTeX 的情况
    def test_to_latex_longtable_without_index(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})  # 创建带有两列的 DataFrame 对象
        result = df.to_latex(index=False, longtable=True)  # 将 DataFrame 转换为长表格 LaTeX 字符串，不包含索引
        expected = _dedent(  # 生成预期的长表格 LaTeX 字符串
            r"""
            \begin{longtable}{rl}
            \toprule
            a & b \\
            \midrule
            \endfirsthead
            \toprule
            a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{2}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            1 & b1 \\
            2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected  # 断言实际输出与预期输出一致

    # 使用参数化测试来验证带有不同列数的 DataFrame 转换为长表格 LaTeX 的情况
    @pytest.mark.parametrize(
        "df_data, expected_number",
        [
            ({"a": [1, 2]}, 1),  # 测试单列 DataFrame 的情况
            ({"a": [1, 2], "b": [3, 4]}, 2),  # 测试两列 DataFrame 的情况
            ({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, 3),  # 测试三列 DataFrame 的情况
        ],
    )
    def test_to_latex_longtable_continued_on_next_page(self, df_data, expected_number):
        df = DataFrame(df_data)  # 根据传入的数据创建 DataFrame 对象
        result = df.to_latex(index=False, longtable=True)  # 将 DataFrame 转换为长表格 LaTeX 字符串，不包含索引
        assert rf"\multicolumn{{{expected_number}}}" in result  # 断言预期的列数标志在输出中存在


class TestToLatexHeader:
    # 下一个测试类的注释应该继续在这里
    def test_to_latex_no_header_with_index(self):
        # 测试函数：test_to_latex_no_header_with_index，用于测试 DataFrame.to_latex() 方法
        # GH 7124：GitHub issue编号，指明这个测试用例关联的问题
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 创建一个测试用的 DataFrame，包含两列：a 和 b
        result = df.to_latex(header=False)
        # 将 DataFrame 转换成 LaTeX 格式，不包含表头
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 预期输出的 LaTeX 格式字符串
        assert result == expected
        # 断言结果与预期输出一致

    def test_to_latex_no_header_without_index(self):
        # 测试函数：test_to_latex_no_header_without_index，用于测试 DataFrame.to_latex() 方法
        # GH 7124：GitHub issue编号，指明这个测试用例关联的问题
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 创建一个测试用的 DataFrame，包含两列：a 和 b
        result = df.to_latex(index=False, header=False)
        # 将 DataFrame 转换成 LaTeX 格式，不包含表头和索引
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 预期输出的 LaTeX 格式字符串
        assert result == expected
        # 断言结果与预期输出一致

    def test_to_latex_specified_header_with_index(self):
        # 测试函数：test_to_latex_specified_header_with_index，用于测试 DataFrame.to_latex() 方法
        # GH 7124：GitHub issue编号，指明这个测试用例关联的问题
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 创建一个测试用的 DataFrame，包含两列：a 和 b
        result = df.to_latex(header=["AA", "BB"])
        # 将 DataFrame 转换成 LaTeX 格式，指定表头为 ["AA", "BB"]
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & AA & BB \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 预期输出的 LaTeX 格式字符串
        assert result == expected
        # 断言结果与预期输出一致

    def test_to_latex_specified_header_without_index(self):
        # 测试函数：test_to_latex_specified_header_without_index，用于测试 DataFrame.to_latex() 方法
        # GH 7124：GitHub issue编号，指明这个测试用例关联的问题
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 创建一个测试用的 DataFrame，包含两列：a 和 b
        result = df.to_latex(header=["AA", "BB"], index=False)
        # 将 DataFrame 转换成 LaTeX 格式，指定表头为 ["AA", "BB"]，并去除索引
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            AA & BB \\
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 预期输出的 LaTeX 格式字符串
        assert result == expected
        # 断言结果与预期输出一致

    @pytest.mark.parametrize(
        "header, num_aliases",
        [
            (["A"], 1),
            (("B",), 1),
            (("Col1", "Col2", "Col3"), 3),
            (("Col1", "Col2", "Col3", "Col4"), 4),
        ],
    )
    def test_to_latex_number_of_items_in_header_missmatch_raises(
        self,
        header,
        num_aliases,
    ):
        # 参数化测试函数：test_to_latex_number_of_items_in_header_missmatch_raises，
        # 用于测试 DataFrame.to_latex() 方法在不同表头数量时是否正确抛出 ValueError 异常
        # GH 7124：GitHub issue编号，指明这个测试用例关联的问题
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 创建一个测试用的 DataFrame，包含两列：a 和 b
        msg = f"Writing 2 cols but got {num_aliases} aliases"
        # 预期的错误信息，表明期望写入的列数与给定表头的别名数量不匹配
        with pytest.raises(ValueError, match=msg):
            # 期望抛出 ValueError 异常，异常信息匹配 msg
            df.to_latex(header=header)

    def test_to_latex_decimal(self):
        # 测试函数：test_to_latex_decimal，用于测试 DataFrame.to_latex() 方法在处理小数时的输出
        # GH 12031：GitHub issue编号，指明这个测试用例关联的问题
        df = DataFrame({"a": [1.0, 2.1], "b": ["b1", "b2"]})
        # 创建一个测试用的 DataFrame，包含两列：a 和 b，其中 a 包含小数
        result = df.to_latex(decimal=",")
        # 将 DataFrame 转换成 LaTeX 格式，指定小数分隔符为 ","
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1,000000 & b1 \\
            1 & 2,100000 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 预期输出的 LaTeX 格式字符串
        assert result == expected
        # 断言结果与预期输出一致
class TestToLatexBold:
    def test_to_latex_bold_rows(self):
        # 创建一个包含两列的 DataFrame，列名分别为 'a' 和 'b'，数据为 [1, 2] 和 ['b1', 'b2']
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 调用 DataFrame 的 to_latex 方法，设置 bold_rows=True，生成 LaTeX 格式的表格
        result = df.to_latex(bold_rows=True)
        # 预期的 LaTeX 格式字符串，包含一个带有加粗行标题的表格
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            \textbf{0} & 1 & b1 \\
            \textbf{1} & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的 LaTeX 结果与预期结果相同
        assert result == expected

    def test_to_latex_no_bold_rows(self):
        # 创建一个包含两列的 DataFrame，列名分别为 'a' 和 'b'，数据为 [1, 2] 和 ['b1', 'b2']
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 调用 DataFrame 的 to_latex 方法，设置 bold_rows=False，生成 LaTeX 格式的表格
        result = df.to_latex(bold_rows=False)
        # 预期的 LaTeX 格式字符串，不包含加粗行标题的表格
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的 LaTeX 结果与预期结果相同
        assert result == expected


class TestToLatexCaptionLabel:
    @pytest.fixture
    def caption_table(self):
        """表格/表格环境的标题"""
        return "a table in a \\texttt{table/tabular} environment"

    @pytest.fixture
    def short_caption(self):
        """用于测试 \\caption[short_caption]{full_caption} 的短标题"""
        return "a table"

    @pytest.fixture
    def label_table(self):
        """表格/表格环境的标签"""
        return "tab:table_tabular"

    @pytest.fixture
    def caption_longtable(self):
        """长表格/longtable 环境的标题"""
        return "a table in a \\texttt{longtable} environment"

    @pytest.fixture
    def label_longtable(self):
        """长表格/longtable 环境的标签"""
        return "tab:longtable"

    def test_to_latex_caption_only(self, df_short, caption_table):
        # GH 25436
        # 调用 DataFrame 的 to_latex 方法，仅设置标题为 caption_table，生成 LaTeX 格式的表格
        result = df_short.to_latex(caption=caption_table)
        # 预期的 LaTeX 格式字符串，包含一个带有标题的表格环境
        expected = _dedent(
            r"""
            \begin{table}
            \caption{a table in a \texttt{table/tabular} environment}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        # 断言生成的 LaTeX 结果与预期结果相同
        assert result == expected

    def test_to_latex_label_only(self, df_short, label_table):
        # GH 25436
        # 调用 DataFrame 的 to_latex 方法，仅设置标签为 label_table，生成 LaTeX 格式的表格
        result = df_short.to_latex(label=label_table)
        # 预期的 LaTeX 格式字符串，包含一个带有标签的表格环境
        expected = _dedent(
            r"""
            \begin{table}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        # 断言生成的 LaTeX 结果与预期结果相同
        assert result == expected
    # 定义一个测试方法，用于测试将 DataFrame 转换为 LaTeX 格式的表格，并验证生成的 LaTeX 代码是否符合预期
    def test_to_latex_caption_and_label(self, df_short, caption_table, label_table):
        # GH 25436
        # 调用 DataFrame 的 to_latex 方法，生成带有指定标题和标签的 LaTeX 代码
        result = df_short.to_latex(caption=caption_table, label=label_table)
        # 期望的正确的 LaTeX 代码，使用 _dedent 进行格式化
        expected = _dedent(
            r"""
            \begin{table}
            \caption{a table in a \texttt{table/tabular} environment}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        # 断言实际生成的 LaTeX 代码与期望的一致
        assert result == expected

    # 定义一个测试方法，用于测试将 DataFrame 转换为 LaTeX 格式的表格，并验证生成的 LaTeX 代码是否符合预期（包含简短标题）
    def test_to_latex_caption_and_shortcaption(
        self,
        df_short,
        caption_table,
        short_caption,
    ):
        # 调用 DataFrame 的 to_latex 方法，生成带有指定标题和简短标题的 LaTeX 代码
        result = df_short.to_latex(caption=(caption_table, short_caption))
        # 期望的正确的 LaTeX 代码，使用 _dedent 进行格式化
        expected = _dedent(
            r"""
            \begin{table}
            \caption[a table]{a table in a \texttt{table/tabular} environment}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        # 断言实际生成的 LaTeX 代码与期望的一致
        assert result == expected

    # 定义一个测试方法，用于测试将 DataFrame 转换为 LaTeX 格式的表格，并验证使用列表形式的标题是否正常工作
    def test_to_latex_caption_and_shortcaption_list_is_ok(self, df_short):
        # 定义一个长标题和短标题的元组
        caption = ("Long-long-caption", "Short")
        # 调用 DataFrame 的 to_latex 方法，生成带有指定标题和短标题的 LaTeX 代码（元组形式）
        result_tuple = df_short.to_latex(caption=caption)
        # 调用 DataFrame 的 to_latex 方法，生成带有指定标题和短标题的 LaTeX 代码（列表形式）
        result_list = df_short.to_latex(caption=list(caption))
        # 断言两种形式生成的 LaTeX 代码结果相同
        assert result_tuple == result_list

    # 定义一个测试方法，用于测试将 DataFrame 转换为 LaTeX 格式的表格，并验证生成的 LaTeX 代码是否符合预期（包含简短标题和标签）
    def test_to_latex_caption_shortcaption_and_label(
        self,
        df_short,
        caption_table,
        short_caption,
        label_table,
    ):
        # 调用 DataFrame 的 to_latex 方法，生成带有指定标题、简短标题和标签的 LaTeX 代码
        result = df_short.to_latex(
            caption=(caption_table, short_caption),
            label=label_table,
        )
        # 期望的正确的 LaTeX 代码，使用 _dedent 进行格式化
        expected = _dedent(
            r"""
            \begin{table}
            \caption[a table]{a table in a \texttt{table/tabular} environment}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        # 断言实际生成的 LaTeX 代码与期望的一致
        assert result == expected

    # 使用 pytest.mark.parametrize 标记的参数化测试，测试当提供不正确参数组合时的行为
    @pytest.mark.parametrize(
        "bad_caption",
        [
            ("full_caption", "short_caption", "extra_string"),  # 混合字符串组合
            ("full_caption", "short_caption", 1),  # 包含整数的组合
            ("full_caption", "short_caption", None),  # 包含 None 的组合
            ("full_caption",),  # 只提供长标题
            (None,),  # 只提供 None
        ],
    )
    # 测试当传入错误的标题参数时是否引发异常
    def test_to_latex_bad_caption_raises(self, bad_caption):
        # 创建一个包含单列的数据框
        df = DataFrame({"a": [1]})
        # 准备匹配的错误消息
        msg = "`caption` must be either a string or 2-tuple of strings"
        # 断言调用 df.to_latex() 时会引发 ValueError 异常，并且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            df.to_latex(caption=bad_caption)

    # 测试当传入两个字符的标题时是否正常处理
    def test_to_latex_two_chars_caption(self, df_short):
        # 调用 df_short.to_latex() 生成 LaTeX 表格代码，使用 "xy" 作为标题
        result = df_short.to_latex(caption="xy")
        # 准备预期的 LaTeX 表格代码
        expected = _dedent(
            r"""
            \begin{table}
            \caption{xy}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        # 断言生成的 LaTeX 代码与预期结果一致
        assert result == expected

    # 测试在使用 longtable 参数时，只提供长表格的标题，不提供标签
    def test_to_latex_longtable_caption_only(self, df_short, caption_longtable):
        # 调用 df_short.to_latex() 生成长表格的 LaTeX 代码，使用 caption_longtable 作为标题
        result = df_short.to_latex(longtable=True, caption=caption_longtable)
        # 准备预期的 LaTeX 表格代码
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \caption{a table in a \texttt{longtable} environment} \\
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \caption[]{a table in a \texttt{longtable} environment} \\
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        # 断言生成的 LaTeX 代码与预期结果一致
        assert result == expected

    # 测试在使用 longtable 参数时，只提供长表格的标签，不提供标题
    def test_to_latex_longtable_label_only(self, df_short, label_longtable):
        # 调用 df_short.to_latex() 生成长表格的 LaTeX 代码，使用 label_longtable 作为标签
        result = df_short.to_latex(longtable=True, label=label_longtable)
        # 准备预期的 LaTeX 表格代码
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \label{tab:longtable} \\
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        # 断言生成的 LaTeX 代码与预期结果一致
        assert result == expected

    # 测试在使用 longtable 参数时，同时提供长表格的标题和标签
    def test_to_latex_longtable_caption_and_label(
        self,
        df_short,
        caption_longtable,
        label_longtable,
    ):
        # 调用 df_short.to_latex() 生成长表格的 LaTeX 代码，使用 caption_longtable 作为标题，label_longtable 作为标签
        result = df_short.to_latex(longtable=True, caption=caption_longtable, label=label_longtable)
        # 准备预期的 LaTeX 表格代码
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \caption{a table in a \texttt{longtable} environment} \\
            \label{tab:longtable} \\
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        # 断言生成的 LaTeX 代码与预期结果一致
        assert result == expected
    ):
        # GH 25436
        # 调用 DataFrame 的 to_latex 方法生成长表格的 LaTeX 格式字符串
        result = df_short.to_latex(
            longtable=True,                 # 使用 longtable 环境生成长表格
            caption=caption_longtable,       # 设置表格标题
            label=label_longtable,           # 设置表格标签
        )
        # 期望的 LaTeX 格式字符串，包含一个长表格环境的示例
        expected = _dedent(
            r"""
        \begin{longtable}{lrl}
        \caption{a table in a \texttt{longtable} environment} \label{tab:longtable} \\
        \toprule
         & a & b \\
        \midrule
        \endfirsthead
        \caption[]{a table in a \texttt{longtable} environment} \\
        \toprule
         & a & b \\
        \midrule
        \endhead
        \midrule
        \multicolumn{3}{r}{Continued on next page} \\
        \midrule
        \endfoot
        \bottomrule
        \endlastfoot
        0 & 1 & b1 \\
        1 & 2 & b2 \\
        \end{longtable}
        """
        )
        # 断言生成的 LaTeX 字符串与期望的字符串相同
        assert result == expected

    def test_to_latex_longtable_caption_shortcaption_and_label(
        self,
        df_short,
        caption_longtable,
        short_caption,
        label_longtable,
    ):
        # test when the caption, the short_caption and the label are provided
        # 调用 DataFrame 的 to_latex 方法生成长表格的 LaTeX 格式字符串，包括提供了标题、短标题和标签的情况
        result = df_short.to_latex(
            longtable=True,                 # 使用 longtable 环境生成长表格
            caption=(caption_longtable, short_caption),  # 设置表格标题和短标题
            label=label_longtable,           # 设置表格标签
        )
        # 期望的 LaTeX 格式字符串，这部分在下一个代码块中继续
\begin{longtable}{lrl}
\caption[a table]{a table in a \texttt{longtable} environment} \label{tab:longtable} \\
\toprule
 & a & b \\
\midrule
\endfirsthead
\caption[]{a table in a \texttt{longtable} environment} \\
\toprule
 & a & b \\
\midrule
\endhead
\midrule
\multicolumn{3}{r}{Continued on next page} \\
\midrule
\endfoot
\bottomrule
\endlastfoot
0 & 1 & b1 \\
1 & 2 & b2 \\
\end{longtable}
"""
        )
        assert result == expected


class TestToLatexEscape:
    @pytest.fixture
    def df_with_symbols(self):
        """Dataframe with special characters for testing chars escaping."""
        # 定义包含特殊字符的 DataFrame 用于测试字符转义
        a = "a"
        b = "b"
        return DataFrame({"co$e^x$": {a: "a", b: "b"}, "co^l1": {a: "a", b: "b"}})

    def test_to_latex_escape_false(self, df_with_symbols):
        # 测试转义设置为 False 的情况
        result = df_with_symbols.to_latex(escape=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             & co$e^x$ & co^l1 \\
            \midrule
            a & a & a \\
            b & b & b \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_escape_default(self, df_with_symbols):
        # 测试转义默认设置的情况
        # gh50871: in v2.0 escape is False by default (styler.format.escape=None)
        default = df_with_symbols.to_latex()
        specified_true = df_with_symbols.to_latex(escape=True)
        assert default != specified_true

    def test_to_latex_special_escape(self):
        # 测试包含特殊字符的转义
        df = DataFrame([r"a\b\c", r"^a^b^c", r"~a~b~c"])
        result = df.to_latex(escape=True)
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & a\textbackslash b\textbackslash c \\
            1 & \textasciicircum a\textasciicircum b\textasciicircum c \\
            2 & \textasciitilde a\textasciitilde b\textasciitilde c \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_escape_special_chars(self):
        # 测试转义特殊字符的情况
        special_characters = ["&", "%", "$", "#", "_", "{", "}", "~", "^", "\\"]
        df = DataFrame(data=special_characters)
        result = df.to_latex(escape=True)
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & \& \\
            1 & \% \\
            2 & \$ \\
            3 & \# \\
            4 & \_ \\
            5 & \{ \\
            6 & \} \\
            7 & \textasciitilde  \\
            8 & \textasciicircum  \\
            9 & \textbackslash  \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected
    # 定义测试方法，测试 DataFrame 对象生成 LaTeX 表格时不转义特殊字符的情况
    def test_to_latex_specified_header_special_chars_without_escape(self):
        # GH 7124：GitHub 上的 issue 编号
        # 创建一个 DataFrame 对象，包含两列数据："a"列是整数列表，"b"列是字符串列表
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 调用 DataFrame 的 to_latex 方法生成 LaTeX 格式的表格
        # 指定表头为 ["$A$", "$B$"]，并且不对特殊字符进行转义
        result = df.to_latex(header=["$A$", "$B$"], escape=False)
        # 期望的生成的 LaTeX 字符串，使用 _dedent 函数去除多余的缩进
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & $A$ & $B$ \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的 LaTeX 字符串与期望的字符串相等
        assert result == expected
class TestToLatexPosition:
    def test_to_latex_position(self):
        # 设定 LaTeX 表格的位置为 "h"
        the_position = "h"
        # 创建一个包含两列（'a' 和 'b'）的 DataFrame
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 将 DataFrame 转换为 LaTeX 格式的表格，并指定位置为 the_position
        result = df.to_latex(position=the_position)
        # 期望的 LaTeX 表格格式，使用 _dedent 去除字符串前面的缩进
        expected = _dedent(
            r"""
            \begin{table}[h]
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        # 断言结果与期望值相等
        assert result == expected

    def test_to_latex_longtable_position(self):
        # 设定 LaTeX 长表格的位置为 "t"
        the_position = "t"
        # 创建一个包含两列（'a' 和 'b'）的 DataFrame
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        # 将 DataFrame 转换为 LaTeX 格式的长表格，并指定长表格和位置参数
        result = df.to_latex(longtable=True, position=the_position)
        # 期望的 LaTeX 长表格格式，使用 _dedent 去除字符串前面的缩进
        expected = _dedent(
            r"""
            \begin{longtable}[t]{lrl}
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        # 断言结果与期望值相等
        assert result == expected


class TestToLatexFormatters:
    def test_to_latex_with_formatters(self):
        # 创建一个包含不同数据类型的 DataFrame
        df = DataFrame(
            {
                "datetime64": [
                    datetime(2016, 1, 1),
                    datetime(2016, 2, 5),
                    datetime(2016, 3, 3),
                ],
                "float": [1.0, 2.0, 3.0],
                "int": [1, 2, 3],
                "object": [(1, 2), True, False],
            }
        )

        # 定义列数据的格式化函数字典
        formatters = {
            "datetime64": lambda x: x.strftime("%Y-%m"),
            "float": lambda x: f"[{x: 4.1f}]",
            "int": lambda x: f"0x{x:x}",
            "object": lambda x: f"-{x!s}-",
            "__index__": lambda x: f"index: {x}",
        }
        # 将 DataFrame 转换为 LaTeX 格式的表格，使用给定的格式化函数
        result = df.to_latex(formatters=dict(formatters))

        # 期望的 LaTeX 表格格式，使用 _dedent 去除字符串前面的缩进
        expected = _dedent(
            r"""
            \begin{tabular}{llrrl}
            \toprule
             & datetime64 & float & int & object \\
            \midrule
            index: 0 & 2016-01 & [ 1.0] & 0x1 & -(1, 2)- \\
            index: 1 & 2016-02 & [ 2.0] & 0x2 & -True- \\
            index: 2 & 2016-03 & [ 3.0] & 0x3 & -False- \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言结果与期望值相等
        assert result == expected
    # 定义一个测试方法，测试生成不带固定宽度的浮点数格式化 LaTeX 输出，主题为 GH 21625
    def test_to_latex_float_format_no_fixed_width_3decimals(self):
        # 创建一个包含单个浮点数列 'x' 的 DataFrame
        df = DataFrame({"x": [0.19999]})
        # 调用 DataFrame 的 to_latex 方法，使用格式化字符串 "%.3f" 来格式化浮点数
        result = df.to_latex(float_format="%.3f")
        # 期望的 LaTeX 输出字符串，使用 _dedent 函数格式化
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & x \\
            \midrule
            0 & 0.200 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与期望的输出一致
        assert result == expected

    # 定义一个测试方法，测试生成不带固定宽度的整数格式化 LaTeX 输出，主题为 GH 22270
    def test_to_latex_float_format_no_fixed_width_integer(self):
        # 创建一个包含单个整数列 'x' 的 DataFrame
        df = DataFrame({"x": [100.0]})
        # 调用 DataFrame 的 to_latex 方法，使用格式化字符串 "%.0f" 来格式化整数
        result = df.to_latex(float_format="%.0f")
        # 期望的 LaTeX 输出字符串，使用 _dedent 函数格式化
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & x \\
            \midrule
            0 & 100 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与期望的输出一致
        assert result == expected

    # 使用 pytest 的参数化装饰器，测试生成带不同 na_rep 和浮点数格式化的 LaTeX 输出
    @pytest.mark.parametrize("na_rep", ["NaN", "Ted"])
    def test_to_latex_na_rep_and_float_format(self, na_rep):
        # 创建一个包含两列 'Group' 和 'Data' 的 DataFrame，其中 'Data' 包含浮点数和空值
        df = DataFrame(
            [
                ["A", 1.2225],
                ["A", None],
            ],
            columns=["Group", "Data"],
        )
        # 调用 DataFrame 的 to_latex 方法，使用格式化字符串 "{:.2f}" 来格式化浮点数，并指定 na_rep
        result = df.to_latex(na_rep=na_rep, float_format="{:.2f}".format)
        # 期望的 LaTeX 输出字符串，使用 _dedent 函数格式化
        expected = _dedent(
            rf"""
            \begin{{tabular}}{{llr}}
            \toprule
             & Group & Data \\
            \midrule
            0 & A & 1.22 \\
            1 & A & {na_rep} \\
            \bottomrule
            \end{{tabular}}
            """
        )
        # 断言生成的结果与期望的输出一致
        assert result == expected
class TestToLatexMultiindex:
    @pytest.fixture
    def multiindex_frame(self):
        """返回一个用于测试多行 LaTeX 宏的多级索引数据框架"""
        return DataFrame.from_dict(
            {
                ("c1", 0): Series({x: x for x in range(4)}),
                ("c1", 1): Series({x: x + 4 for x in range(4)}),
                ("c2", 0): Series({x: x for x in range(4)}),
                ("c2", 1): Series({x: x + 4 for x in range(4)}),
                ("c3", 0): Series({x: x for x in range(4)}),
            }
        ).T

    @pytest.fixture
    def multicolumn_frame(self):
        """返回一个用于测试多列 LaTeX 宏的多列数据框架"""
        return DataFrame(
            {
                ("c1", 0): {x: x for x in range(5)},
                ("c1", 1): {x: x + 5 for x in range(5)},
                ("c2", 0): {x: x for x in range(5)},
                ("c2", 1): {x: x + 5 for x in range(5)},
                ("c3", 0): {x: x for x in range(5)},
            }
        )

    def test_to_latex_multindex_header(self):
        # GH 16718
        # 创建一个简单的数据框架 df，设置索引为 ["a", "b"]
        df = DataFrame({"a": [0], "b": [1], "c": [2], "d": [3]})
        df = df.set_index(["a", "b"])
        # 调用 to_latex 方法生成 LaTeX 表格字符串，设置表头为 ["r1", "r2"]，禁用多行功能
        observed = df.to_latex(header=["r1", "r2"], multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llrr}
            \toprule
             &  & r1 & r2 \\
            a & b &  &  \\
            \midrule
            0 & 1 & 2 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert observed == expected

    def test_to_latex_multiindex_empty_name(self):
        # GH 18669
        # 创建一个空名称的多级索引 mi，创建数据框架 df
        mi = pd.MultiIndex.from_product([[1, 2]], names=[""])
        df = DataFrame(-1, index=mi, columns=range(4))
        # 调用 to_latex 方法生成 LaTeX 表格字符串
        observed = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrr}
            \toprule
             & 0 & 1 & 2 & 3 \\
             &  &  &  &  \\
            \midrule
            1 & -1 & -1 & -1 & -1 \\
            2 & -1 & -1 & -1 & -1 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert observed == expected

    def test_to_latex_multiindex_column_tabular(self):
        # 创建一个简单的数据框架 df
        df = DataFrame({("x", "y"): ["a"]})
        # 调用 to_latex 方法生成 LaTeX 表格字符串
        result = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & x \\
             & y \\
            \midrule
            0 & a \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_small_tabular(self):
        # 创建一个简单的数据框架 df，转置后进行处理
        df = DataFrame({("x", "y"): ["a"]}).T
        # 调用 to_latex 方法生成 LaTeX 表格字符串，禁用多行功能
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             &  & 0 \\
            \midrule
            x & y & a \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected
    def test_to_latex_multiindex_tabular(self, multiindex_frame):
        # 使用 DataFrame 的 to_latex 方法将多索引框架转换为 LaTeX 格式的表格
        result = multiindex_frame.to_latex(multirow=False)
        # 预期的 LaTeX 表格字符串，使用 _dedent 函数移除首尾空白
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrr}
            \toprule
             &  & 0 & 1 & 2 & 3 \\
            \midrule
            c1 & 0 & 0 & 1 & 2 & 3 \\
             & 1 & 4 & 5 & 6 & 7 \\
            c2 & 0 & 0 & 1 & 2 & 3 \\
             & 1 & 4 & 5 & 6 & 7 \\
            c3 & 0 & 0 & 1 & 2 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言结果与预期相符
        assert result == expected

    def test_to_latex_multicolumn_tabular(self, multiindex_frame):
        # GH 14184
        # 转置多索引框架，设置列名称，并使用 to_latex 方法生成 LaTeX 格式的表格
        df = multiindex_frame.T
        df.columns.names = ["a", "b"]
        result = df.to_latex(multirow=False)
        # 预期的 LaTeX 表格字符串，使用 _dedent 函数移除首尾空白
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
            a & \multicolumn{2}{r}{c1} & \multicolumn{2}{r}{c2} & c3 \\
            b & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 4 & 0 & 4 & 0 \\
            1 & 1 & 5 & 1 & 5 & 1 \\
            2 & 2 & 6 & 2 & 6 & 2 \\
            3 & 3 & 7 & 3 & 7 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言结果与预期相符
        assert result == expected

    def test_to_latex_index_has_name_tabular(self):
        # GH 10660
        # 创建包含带有名称的索引的 DataFrame，并使用 set_index 方法生成 LaTeX 格式的表格
        df = DataFrame({"a": [0, 0, 1, 1], "b": list("abab"), "c": [1, 2, 3, 4]})
        result = df.set_index(["a", "b"]).to_latex(multirow=False)
        # 预期的 LaTeX 表格字符串，使用 _dedent 函数移除首尾空白
        expected = _dedent(
            r"""
            \begin{tabular}{llr}
            \toprule
             &  & c \\
            a & b &  \\
            \midrule
            0 & a & 1 \\
             & b & 2 \\
            1 & a & 3 \\
             & b & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言结果与预期相符
        assert result == expected

    def test_to_latex_groupby_tabular(self):
        # GH 10660
        # 创建 DataFrame，对其进行分组统计，并使用 to_latex 方法生成 LaTeX 格式的表格
        df = DataFrame({"a": [0, 0, 1, 1], "b": list("abab"), "c": [1, 2, 3, 4]})
        result = (
            df.groupby("a")
            .describe()
            .to_latex(float_format="{:.1f}".format, escape=True)
        )
        # 预期的 LaTeX 表格字符串，使用 _dedent 函数移除首尾空白
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrrrrr}
            \toprule
             & \multicolumn{8}{r}{c} \\
             & count & mean & std & min & 25\% & 50\% & 75\% & max \\
            a &  &  &  &  &  &  &  &  \\
            \midrule
            0 & 2.0 & 1.5 & 0.7 & 1.0 & 1.2 & 1.5 & 1.8 & 2.0 \\
            1 & 2.0 & 3.5 & 0.7 & 3.0 & 3.2 & 3.5 & 3.8 & 4.0 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言结果与预期相符
        assert result == expected
    def test_to_latex_multiindex_dupe_level(self):
        # 测试函数：test_to_latex_multiindex_dupe_level
        # 测试目标：检查多重索引中重复级别的处理
        # 详细描述：
        # 如果一个索引在后续的行中重复出现，应该在创建的表格中用空白替换。
        # 这种情况只会发生在所有更高阶的索引（在左边）也相等的情况下。
        # 在此测试中，'c' 必须两次打印，因为更高阶的索引 'A' != 'B'。
        df = DataFrame(
            index=pd.MultiIndex.from_tuples([("A", "c"), ("B", "c")]), columns=["col"]
        )
        # 调用 to_latex 方法生成 LaTeX 格式的表格
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             &  & col \\
            \midrule
            A & c & NaN \\
            B & c & NaN \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期的 LaTeX 表格一致
        assert result == expected

    def test_to_latex_multicolumn_default(self, multicolumn_frame):
        # 测试函数：test_to_latex_multicolumn_default
        # 测试目标：检查多列合并（multicolumn）的默认行为
        result = multicolumn_frame.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
             & \multicolumn{2}{r}{c1} & \multicolumn{2}{r}{c2} & c3 \\
             & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 5 & 0 & 5 & 0 \\
            1 & 1 & 6 & 1 & 6 & 1 \\
            2 & 2 & 7 & 2 & 7 & 2 \\
            3 & 3 & 8 & 3 & 8 & 3 \\
            4 & 4 & 9 & 4 & 9 & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期的 LaTeX 表格一致
        assert result == expected

    def test_to_latex_multicolumn_false(self, multicolumn_frame):
        # 测试函数：test_to_latex_multicolumn_false
        # 测试目标：检查关闭多列合并（multicolumn）时的表格格式
        result = multicolumn_frame.to_latex(multicolumn=False, multicolumn_format="l")
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
             & c1 & & c2 & & c3 \\
             & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 5 & 0 & 5 & 0 \\
            1 & 1 & 6 & 1 & 6 & 1 \\
            2 & 2 & 7 & 2 & 7 & 2 \\
            3 & 3 & 8 & 3 & 8 & 3 \\
            4 & 4 & 9 & 4 & 9 & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期的 LaTeX 表格一致
        assert result == expected

    def test_to_latex_multirow_true(self, multicolumn_frame):
        # 测试函数：test_to_latex_multirow_true
        # 测试目标：检查多行合并（multirow）为真时的表格格式
        result = multicolumn_frame.T.to_latex(multirow=True)
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrrr}
            \toprule
             &  & 0 & 1 & 2 & 3 & 4 \\
            \midrule
            \multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            \multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\
            \cline{1-7}
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的结果与预期的 LaTeX 表格一致
        assert result == expected
    def test_to_latex_multicolumnrow_with_multicol_format(self, multicolumn_frame):
        # 将索引设置为转置后的索引，以便正确处理多行多列的情况
        multicolumn_frame.index = multicolumn_frame.T.index
        # 将转置后的 DataFrame 转换为 LaTeX 格式的表格，支持多行和多列格式
        result = multicolumn_frame.T.to_latex(
            multirow=True,
            multicolumn=True,
            multicolumn_format="c",
        )
        # 期望的 LaTeX 表格内容，使用 _dedent 函数处理字符串的缩进
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrrr}
            \toprule
             &  & \multicolumn{2}{c}{c1} & \multicolumn{2}{c}{c2} & c3 \\
             &  & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            \multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            \multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\
            \cline{1-7}
            \bottomrule
            \end{tabular}
            """
        )
        # 断言生成的 LaTeX 表格与期望的表格内容一致
        assert result == expected

    @pytest.mark.parametrize("name0", [None, "named0"])
    @pytest.mark.parametrize("name1", [None, "named1"])
    @pytest.mark.parametrize("axes", [[0], [1], [0, 1]])
    def test_to_latex_multiindex_names(self, name0, name1, axes):
        # GH 18667 GitHub 上的 issue 编号，这里是为了记录问题来源
        names = [name0, name1]
        # 创建一个多级索引，从产品中生成多级索引
        mi = pd.MultiIndex.from_product([[1, 2], [3, 4]])
        # 创建一个所有值为 -1 的 DataFrame，索引和列均使用 mi
        df = DataFrame(-1, index=mi.copy(), columns=mi.copy())
        # 针对每个轴上的索引，设置对应的名称
        for idx in axes:
            df.axes[idx].names = names

        # 将名称中的 None 替换为空字符串，以便在 LaTeX 中正确显示
        idx_names = tuple(n or "" for n in names)
        # 根据轴的情况，生成 LaTeX 表格中的行名称
        idx_names_row = (
            f"{idx_names[0]} & {idx_names[1]} &  &  &  &  \\\\\n"
            if (0 in axes and any(names))
            else ""
        )
        # 根据轴的情况，生成 LaTeX 表格中的列名称
        col_names = [n if (bool(n) and 1 in axes) else "" for n in names]
        # 转换 DataFrame 为 LaTeX 格式的表格，不使用多行格式
        observed = df.to_latex(multirow=False)
        # 期望的 LaTeX 表格内容
        expected = r"""\begin{tabular}{llrrrr}
# 测试函数，用于测试生成带有多级索引的 DataFrame 转换为 LaTeX 表格是否正确
def test_to_latex_multiindex_multirow():
    # 创建一个多级索引
    mi = pd.MultiIndex.from_product(
        [[0.0, 1.0], [3.0, 2.0, 1.0], ["0", "1"]],
        names=["i", "val0", "val1"]
    )
    # 创建一个空的 DataFrame，以多级索引作为索引
    df = DataFrame(index=mi)
    # 将 DataFrame 转换为 LaTeX 格式的表格，允许多行合并，不对特殊字符进行转义
    result = df.to_latex(multirow=True, escape=False)
    # 预期的 LaTeX 格式表格字符串，包含了具体的多级索引结构和数据
    expected = _dedent(
        r"""
        \begin{tabular}{lll}
        \toprule
        i & val0 & val1 \\
        \midrule
        \multirow[t]{6}{*}{0.000000} & \multirow[t]{2}{*}{3.000000} & 0 \\
         &  & 1 \\
        \cline{2-3}
         & \multirow[t]{2}{*}{2.000000} & 0 \\
         &  & 1 \\
        \cline{2-3}
         & \multirow[t]{2}{*}{1.000000} & 0 \\
         &  & 1 \\
        \cline{1-3} \cline{2-3}
        \multirow[t]{6}{*}{1.000000} & \multirow[t]{2}{*}{3.000000} & 0 \\
         &  & 1 \\
        \cline{2-3}
         & \multirow[t]{2}{*}{2.000000} & 0 \\
         &  & 1 \\
        \cline{2-3}
         & \multirow[t]{2}{*}{1.000000} & 0 \\
         &  & 1 \\
        \cline{1-3} \cline{2-3}
        \bottomrule
        \end{tabular}
        """
    )
    # 断言观察到的结果与预期结果一致
    assert result == expected
```