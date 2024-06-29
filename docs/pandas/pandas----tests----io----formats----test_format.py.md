# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_format.py`

```
"""
Tests for the file pandas.io.formats.format, *not* tests for general formatting
of pandas objects.
"""

# 从标准库中导入 datetime 类
from datetime import datetime
# 从 io 模块中导入 StringIO 类
from io import StringIO
# 导入正则表达式模块
import re
# 从 shutil 库中导入 get_terminal_size 函数
from shutil import get_terminal_size

# 导入第三方库 numpy，并简写为 np
import numpy as np
# 导入 pytest 测试框架
import pytest

# 从 pandas 库中导入 using_pyarrow_string_dtype 设置
from pandas._config import using_pyarrow_string_dtype

# 导入 pandas 库，并简写为 pd
import pandas as pd
# 从 pandas 库中导入以下对象和函数
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
    get_option,
    option_context,
    read_csv,
    reset_option,
)

# 从 pandas.io.formats 模块中导入 printing
from pandas.io.formats import printing
# 从 pandas.io.formats.format 模块中导入 fmt
import pandas.io.formats.format as fmt


# 检查 DataFrame 是否有 info 的表示方法
def has_info_repr(df):
    # 获得 DataFrame 的字符串表示形式
    r = repr(df)
    # 检查第一行是否以 "<class" 开头
    c1 = r.split("\n")[0].startswith("<class")
    # 检查第一行是否以 "&lt;class" 开头（用于 HTML 表示）
    c2 = r.split("\n")[0].startswith(r"&lt;class")
    # 返回是否满足以上任一条件
    return c1 or c2


# 检查 DataFrame 是否有非详细的 info 表示方法
def has_non_verbose_info_repr(df):
    # 检查 DataFrame 是否有 info 的表示方法
    has_info = has_info_repr(df)
    # 获得 DataFrame 的字符串表示形式
    r = repr(df)
    # 检查字符串是否正好分为六行
    nv = len(r.split("\n")) == 6
    # 返回是否有 info 的表示方法并且是否非详细
    return has_info and nv


# 检查 DataFrame 是否水平截断的表示方法
def has_horizontally_truncated_repr(df):
    try:  # 检查表头行
        # 将 DataFrame 的字符串表示形式按行拆分，并转换成数组
        fst_line = np.array(repr(df).splitlines()[0].split())
        # 找到 "..." 所在的列索引
        cand_col = np.where(fst_line == "...")[0][0]
    except IndexError:
        return False
    # 确保每一行都在同一列上有 "..."
    r = repr(df)
    for ix, _ in enumerate(r.splitlines()):
        if not r.split()[cand_col] == "...":
            return False
    return True


# 检查 DataFrame 是否垂直截断的表示方法
def has_vertically_truncated_repr(df):
    # 获得 DataFrame 的字符串表示形式
    r = repr(df)
    only_dot_row = False
    for row in r.splitlines():
        # 使用正则表达式检查是否每行只有 "." 或空格组成
        if re.match(r"^[\.\ ]+$", row):
            only_dot_row = True
    return only_dot_row


# 检查 DataFrame 是否截断的表示方法（水平或垂直）
def has_truncated_repr(df):
    return has_horizontally_truncated_repr(df) or has_vertically_truncated_repr(df)


# 检查 DataFrame 是否同时水平和垂直截断的表示方法
def has_doubly_truncated_repr(df):
    return has_horizontally_truncated_repr(df) and has_vertically_truncated_repr(df)


# 检查 DataFrame 是否有扩展的表示方法
def has_expanded_repr(df):
    # 获得 DataFrame 的字符串表示形式
    r = repr(df)
    for line in r.split("\n"):
        # 检查是否有以 "\\" 结尾的行
        if line.endswith("\\"):
            return True
    return False


# 测试 DataFrame 格式化的测试类
class TestDataFrameFormatting:
    def test_repr_truncation(self):
        # 设置最大长度为20
        max_len = 20
        # 在上下文中设置选项"display.max_colwidth"为max_len
        with option_context("display.max_colwidth", max_len):
            # 创建一个DataFrame对象df，包含列"A"和"B"
            df = DataFrame(
                {
                    "A": np.random.default_rng(2).standard_normal(10),
                    "B": [
                        "a"
                        * np.random.default_rng(2).integers(max_len - 1, max_len + 1)
                        for _ in range(10)
                    ],
                }
            )
            # 获取DataFrame对象df的字符串表示，并去除第一个换行符后的部分
            r = repr(df)
            r = r[r.find("\n") + 1 :]

            # 获取当前打印调整设置
            adj = printing.get_adjustment()

            # 遍历r的每一行和df["B"]中的每个值
            for line, value in zip(r.split("\n"), df["B"]):
                # 如果值value的长度加1大于max_len，则断言行line包含"..."
                if adj.len(value) + 1 > max_len:
                    assert "..." in line
                else:
                    # 否则断言行line不包含"..."
                    assert "..." not in line

        # 在上下文中设置选项"display.max_colwidth"为999999
        with option_context("display.max_colwidth", 999999):
            # 断言DataFrame对象df的字符串表示中不包含"..."
            assert "..." not in repr(df)

        # 在上下文中设置选项"display.max_colwidth"为max_len + 2
        with option_context("display.max_colwidth", max_len + 2):
            # 断言DataFrame对象df的字符串表示中不包含"..."
            assert "..." not in repr(df)

    def test_repr_truncation_preserves_na(self):
        # 创建一个包含10个NA值的DataFrame对象df
        df = DataFrame({"a": [pd.NA for _ in range(10)]})
        # 在上下文中设置选项"display.max_rows"为2和"display.show_dimensions"为False
        with option_context("display.max_rows", 2, "display.show_dimensions", False):
            # 断言DataFrame对象df的字符串表示符合预期格式
            assert repr(df) == "       a\n0   <NA>\n..   ...\n9   <NA>"

    def test_max_colwidth_negative_int_raises(self):
        # 断言使用选项"display.max_colwidth"为-1时会引发ValueError异常，并且异常信息匹配指定内容
        with pytest.raises(
            ValueError, match="Value must be a nonnegative integer or None"
        ):
            with option_context("display.max_colwidth", -1):
                pass

    def test_repr_chop_threshold(self):
        # 创建一个DataFrame对象df，包含两行两列的数据
        df = DataFrame([[0.1, 0.5], [0.5, -0.1]])
        # 重置选项"display.chop_threshold"为默认值None
        reset_option("display.chop_threshold")  # default None
        # 断言DataFrame对象df的字符串表示符合预期格式
        assert repr(df) == "     0    1\n0  0.1  0.5\n1  0.5 -0.1"

        # 在上下文中设置选项"display.chop_threshold"为0.2
        with option_context("display.chop_threshold", 0.2):
            # 断言DataFrame对象df的字符串表示符合预期格式
            assert repr(df) == "     0    1\n0  0.0  0.5\n1  0.5  0.0"

        # 在上下文中设置选项"display.chop_threshold"为0.6
        with option_context("display.chop_threshold", 0.6):
            # 断言DataFrame对象df的字符串表示符合预期格式
            assert repr(df) == "     0    1\n0  0.0  0.0\n1  0.0  0.0"

        # 在上下文中设置选项"display.chop_threshold"为None
        with option_context("display.chop_threshold", None):
            # 断言DataFrame对象df的字符串表示符合预期格式
            assert repr(df) == "     0    1\n0  0.1  0.5\n1  0.5 -0.1"
    # 测试函数，用于测试在chop_threshold列值低于阈值时的情况
    def test_repr_chop_threshold_column_below(self):
        # GH 6839: validation case

        # 创建一个DataFrame对象，包含两列数据
        df = DataFrame([[10, 20, 30, 40], [8e-10, -1e-11, 2e-9, -2e-11]]).T

        # 设置display.chop_threshold选项为0，验证DataFrame对象的字符串表示是否符合预期
        with option_context("display.chop_threshold", 0):
            assert repr(df) == (
                "      0             1\n"
                "0  10.0  8.000000e-10\n"
                "1  20.0 -1.000000e-11\n"
                "2  30.0  2.000000e-09\n"
                "3  40.0 -2.000000e-11"
            )

        # 设置display.chop_threshold选项为1e-8，验证DataFrame对象的字符串表示是否符合预期
        with option_context("display.chop_threshold", 1e-8):
            assert repr(df) == (
                "      0             1\n"
                "0  10.0  0.000000e+00\n"
                "1  20.0  0.000000e+00\n"
                "2  30.0  0.000000e+00\n"
                "3  40.0  0.000000e+00"
            )

        # 设置display.chop_threshold选项为5e-11，验证DataFrame对象的字符串表示是否符合预期
        with option_context("display.chop_threshold", 5e-11):
            assert repr(df) == (
                "      0             1\n"
                "0  10.0  8.000000e-10\n"
                "1  20.0  0.000000e+00\n"
                "2  30.0  2.000000e-09\n"
                "3  40.0  0.000000e+00"
            )

    # 测试函数，用于测试在没有反斜杠的情况下的字符串表示
    def test_repr_no_backslash(self):
        # 设置mode.sim_interactive选项为True
        with option_context("mode.sim_interactive", True):
            # 创建一个包含随机数据的DataFrame对象
            df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
            # 验证DataFrame对象的字符串表示中是否不包含反斜杠
            assert "\\" not in repr(df)

    # 测试函数，用于测试expand_frame_repr选项
    def test_expand_frame_repr(self):
        # 创建不同形状的DataFrame对象
        df_small = DataFrame("hello", index=[0], columns=[0])
        df_wide = DataFrame("hello", index=[0], columns=range(10))
        df_tall = DataFrame("hello", index=range(30), columns=range(5))

        # 设置mode.sim_interactive、display.max_columns、display.width、display.max_rows、display.show_dimensions选项
        with option_context("mode.sim_interactive", True):
            with option_context(
                "display.max_columns",
                10,
                "display.width",
                20,
                "display.max_rows",
                20,
                "display.show_dimensions",
                True,
            ):
                # 设置display.expand_frame_repr选项为True，验证DataFrame对象的字符串表示是否符合预期
                with option_context("display.expand_frame_repr", True):
                    assert not has_truncated_repr(df_small)
                    assert not has_expanded_repr(df_small)
                    assert not has_truncated_repr(df_wide)
                    assert has_expanded_repr(df_wide)
                    assert has_vertically_truncated_repr(df_tall)
                    assert has_expanded_repr(df_tall)

                # 设置display.expand_frame_repr选项为False，验证DataFrame对象的字符串表示是否符合预期
                with option_context("display.expand_frame_repr", False):
                    assert not has_truncated_repr(df_small)
                    assert not has_expanded_repr(df_small)
                    assert not has_horizontally_truncated_repr(df_wide)
                    assert not has_expanded_repr(df_wide)
                    assert has_vertically_truncated_repr(df_tall)
                    assert not has_expanded_repr(df_tall)
    # 定义测试函数，验证在非交互模式下，不依赖于终端自动大小检测的结果
    def test_repr_non_interactive(self):
        # 创建一个包含1000行和5列的DataFrame对象，索引为0到999，列名为0到4
        df = DataFrame("hello", index=range(1000), columns=range(5))

        # 使用上下文管理器设置多个显示选项，确保模拟非交互式模式
        with option_context(
            "mode.sim_interactive", False, "display.width", 0, "display.max_rows", 5000
        ):
            # 断言DataFrame对象在截断表示和展开表示方面都不被截断
            assert not has_truncated_repr(df)
    # 定义测试函数，用于测试在最大列数和行数条件下的 DataFrame 表现
    def test_repr_max_columns_max_rows(self):
        # 获取终端的宽度和高度
        term_width, term_height = get_terminal_size()
        # 如果终端宽度或高度小于10，则跳过测试，并输出相应信息
        if term_width < 10 or term_height < 10:
            pytest.skip(f"terminal size too small, {term_width} x {term_height}")

        # 定义生成指定行数的 DataFrame 的辅助函数
        def mkframe(n):
            # 生成索引列表，格式为 '00000' 到 'n-1'
            index = [f"{i:05d}" for i in range(n)]
            # 创建一个值为0的 DataFrame，使用 index 作为行和列的标签
            return DataFrame(0, index, index)

        # 创建两个不同行数的 DataFrame
        df6 = mkframe(6)
        df10 = mkframe(10)

        # 在 "mode.sim_interactive" 为 True 的上下文中执行以下代码块
        with option_context("mode.sim_interactive", True):
            # 设置显示宽度为终端宽度的两倍
            with option_context("display.width", term_width * 2):
                # 设置最大显示行数和列数分别为5
                with option_context("display.max_rows", 5, "display.max_columns", 5):
                    # 断言 DataFrame mkframe(4) 没有扩展的表示形式
                    assert not has_expanded_repr(mkframe(4))
                    # 断言 DataFrame mkframe(5) 没有扩展的表示形式
                    assert not has_expanded_repr(mkframe(5))
                    # 断言 DataFrame df6 没有扩展的表示形式
                    assert not has_expanded_repr(df6)
                    # 断言 DataFrame df6 有双重截断的表示形式
                    assert has_doubly_truncated_repr(df6)

                # 设置最大显示行数为20，最大显示列数为10
                with option_context("display.max_rows", 20, "display.max_columns", 10):
                    # 断言 DataFrame df6 没有扩展的表示形式
                    assert not has_expanded_repr(df6)
                    # 断言 DataFrame df6 没有截断的表示形式
                    assert not has_truncated_repr(df6)

                # 设置最大显示行数为9，最大显示列数为10
                with option_context("display.max_rows", 9, "display.max_columns", 10):
                    # 断言 DataFrame df10 没有扩展的表示形式
                    assert not has_expanded_repr(df10)
                    # 断言 DataFrame df10 有垂直截断的表示形式
                    assert has_vertically_truncated_repr(df10)

            # 当终端宽度为 None 时，自动检测宽度
            with option_context(
                "display.max_columns",
                100,
                "display.max_rows",
                term_width * 20,
                "display.width",
                None,
            ):
                # 根据终端宽度计算需要生成的 DataFrame 的行数
                df = mkframe((term_width // 7) - 2)
                # 断言该 DataFrame 没有扩展的表示形式
                assert not has_expanded_repr(df)
                # 根据终端宽度计算需要生成的 DataFrame 的行数
                df = mkframe((term_width // 7) + 2)
                # 打印并断言该 DataFrame 有扩展的表示形式
                printing.pprint_thing(df._repr_fits_horizontal_())
                assert has_expanded_repr(df)
    # 定义测试方法 test_repr_min_rows
    def test_repr_min_rows(self):
        # 创建一个包含'a'列的 DataFrame，包含数字0到19
        df = DataFrame({"a": range(20)})

        # 默认设置下，即使超过最小行数也不截断
        assert ".." not in repr(df)
        assert ".." not in df._repr_html_()

        # 创建一个包含'a'列的 DataFrame，包含数字0到60
        df = DataFrame({"a": range(61)})

        # 当 max_rows 默认值为60时，超过最大行数会触发截断
        assert ".." in repr(df)
        assert ".." in df._repr_html_()

        # 使用 option_context 设置 display.max_rows 为10，display.min_rows 为4
        with option_context("display.max_rows", 10, "display.min_rows", 4):
            # 在前两行之后进行截断
            assert ".." in repr(df)
            assert "2  " not in repr(df)
            assert "..." in df._repr_html_()
            assert "<td>2</td>" not in df._repr_html_()

        # 使用 option_context 设置 display.max_rows 为12，display.min_rows 为None
        with option_context("display.max_rows", 12, "display.min_rows", None):
            # 当 min_rows 设置为 None 时，遵循 max_rows 的值
            assert "5    5" in repr(df)
            assert "<td>5</td>" in df._repr_html_()

        # 使用 option_context 设置 display.max_rows 为10，display.min_rows 为12
        with option_context("display.max_rows", 10, "display.min_rows", 12):
            # 当 min_rows 设置值大于 max_rows 时，使用最小值
            assert "5    5" not in repr(df)
            assert "<td>5</td>" not in df._repr_html_()

        # 使用 option_context 设置 display.max_rows 为None，display.min_rows 为12
        with option_context("display.max_rows", None, "display.min_rows", 12):
            # 当 max_rows 设置为 None 时，永不截断
            assert ".." not in repr(df)
            assert ".." not in df._repr_html_()

    # 定义测试方法 test_str_max_colwidth，用于测试字符串最大列宽度
    def test_str_max_colwidth(self):
        # 创建一个包含两行数据的 DataFrame，每行包含'a', 'b', 'c', 'd'四列
        df = DataFrame(
            [
                {
                    "a": "foo",
                    "b": "bar",
                    "c": "uncomfortably long line with lots of stuff",
                    "d": 1,
                },
                {"a": "foo", "b": "bar", "c": "stuff", "d": 1},
            ]
        )
        # 将 DataFrame 设置索引为'a', 'b', 'c'
        df.set_index(["a", "b", "c"])
        
        # 验证 DataFrame 的字符串表示是否符合预期
        assert str(df) == (
            "     a    b                                           c  d\n"
            "0  foo  bar  uncomfortably long line with lots of stuff  1\n"
            "1  foo  bar                                       stuff  1"
        )
        
        # 使用 option_context 设置 max_colwidth 为20
        with option_context("max_colwidth", 20):
            # 验证 DataFrame 的字符串表示在设置最大列宽度后是否符合预期
            assert str(df) == (
                "     a    b                    c  d\n"
                "0  foo  bar  uncomfortably lo...  1\n"
                "1  foo  bar                stuff  1"
            )
    # 定义一个测试方法，用于自动检测终端尺寸
    def test_auto_detect(self):
        # 获取终端的宽度和高度
        term_width, term_height = get_terminal_size()
        # 设置一个任意大的因子以确保超过终端宽度
        fac = 1.05
        # 创建一个列范围，长度为终端宽度乘以 fac 的整数部分
        cols = range(int(term_width * fac))
        # 创建一个索引范围为 0 到 9 的数据框
        index = range(10)
        df = DataFrame(index=index, columns=cols)
        
        # 进入选项上下文，设置模式为交互式显示
        with option_context("mode.sim_interactive", True):
            # 进入选项上下文，设置最大行数显示为无限制
            with option_context("display.max_rows", None):
                # 进入选项上下文，设置最大列数显示为无限制
                with option_context("display.max_columns", None):
                    # 断言 DataFrame 是否具有扩展表示形式
                    assert has_expanded_repr(df)
            
            # 进入选项上下文，设置最大行数显示为 0
            with option_context("display.max_rows", 0):
                # 进入选项上下文，设置最大列数显示为 0
                with option_context("display.max_columns", 0):
                    # 断言 DataFrame 是否具有水平截断的表示形式
                    assert has_horizontally_truncated_repr(df)

            # 更新索引范围为终端高度乘以 fac 的整数部分
            index = range(int(term_height * fac))
            df = DataFrame(index=index, columns=cols)
            
            # 进入选项上下文，设置最大行数显示为 0
            with option_context("display.max_rows", 0):
                # 进入选项上下文，设置最大列数显示为无限制
                with option_context("display.max_columns", None):
                    # 断言 DataFrame 是否具有扩展表示形式
                    assert has_expanded_repr(df)
                    # 断言 DataFrame 是否具有垂直截断的表示形式
                    assert has_vertically_truncated_repr(df)

            # 进入选项上下文，设置最大行数显示为无限制
            with option_context("display.max_rows", None):
                # 进入选项上下文，设置最大列数显示为 0
                with option_context("display.max_columns", 0):
                    # 断言 DataFrame 是否具有水平截断的表示形式
                    assert has_horizontally_truncated_repr(df)

    # 定义一个测试方法，测试字符串表示和 Unicode 编码
    def test_to_string_repr_unicode2(self):
        # 创建一个索引对象，包含包含 Unicode 字符的字符串
        idx = Index(["abc", "\u03c3a", "aegdvg"])
        # 创建一个 Series 对象，其中数据为随机标准正态分布，索引为 idx
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        # 获取 Series 对象的字符串表示，并按行分割
        rs = repr(ser).split("\n")
        # 记录第一行的长度
        line_len = len(rs[0])
        # 遍历每一行（除第一行外）
        for line in rs[1:]:
            try:
                # 尝试解码每行以获取显示编码
                line = line.decode(get_option("display.encoding"))
            except AttributeError:
                pass
            # 如果行不以 "dtype:" 开头，断言其长度与第一行相同
            if not line.startswith("dtype:"):
                assert len(line) == line_len

    # 定义一个测试方法，测试字符串缓冲和 Unicode 编码的所有情况
    def test_to_string_buffer_all_unicode(self):
        # 创建一个字符串缓冲
        buf = StringIO()

        # 创建一个空的 DataFrame，包含一个具有 Unicode 字符的列
        empty = DataFrame({"c/\u03c3": Series(dtype=object)})
        # 创建一个非空的 DataFrame，包含一个数值列和一个具有 Unicode 字符的列
        nonempty = DataFrame({"c/\u03c3": Series([1, 2, 3])})

        # 将空的 DataFrame 和非空的 DataFrame 输出到字符串缓冲中
        print(empty, file=buf)
        print(nonempty, file=buf)

        # 断言可以从缓冲中获取值
        buf.getvalue()

    # 使用 pytest 的参数化装饰器定义多个参数化测试用例
    @pytest.mark.parametrize(
        "index_scalar",
        [
            "a" * 10,
            1,
            Timestamp(2020, 1, 1),
            pd.Period("2020-01-01"),
        ],
    )
    # 使用 pytest 的参数化装饰器定义多个参数化测试用例
    @pytest.mark.parametrize("h", [10, 20])
    # 使用 pytest 的参数化装饰器定义多个参数化测试用例
    @pytest.mark.parametrize("w", [10, 20])
    # 定义一个测试方法，用于测试在不同条件下DataFrame对象的字符串表示是否被截断
    def test_to_string_truncate_indices(self, index_scalar, h, w):
        # 设置选项上下文，禁用展开行表示
        with option_context("display.expand_frame_repr", False):
            # 创建一个DataFrame对象，行索引为单个标量的重复，列索引为以数字串形式表示的列名
            df = DataFrame(
                index=[index_scalar] * h, columns=[str(i) * 10 for i in range(w)]
            )
            # 设置选项上下文，限制最大显示行数为15
            with option_context("display.max_rows", 15):
                # 根据条件判断是否应该垂直截断DataFrame的表示，并进行断言
                if h == 20:
                    assert has_vertically_truncated_repr(df)
                else:
                    assert not has_vertically_truncated_repr(df)
            # 设置选项上下文，限制最大显示列数为15
            with option_context("display.max_columns", 15):
                # 根据条件判断是否应该水平截断DataFrame的表示，并进行断言
                if w == 20:
                    assert has_horizontally_truncated_repr(df)
                else:
                    assert not has_horizontally_truncated_repr(df)
            # 设置选项上下文，限制最大显示行数和列数均为15
            with option_context("display.max_rows", 15, "display.max_columns", 15):
                # 根据条件判断是否应该同时垂直和水平截断DataFrame的表示，并进行断言
                if h == 20 and w == 20:
                    assert has_doubly_truncated_repr(df)
                else:
                    assert not has_doubly_truncated_repr(df)

    # 定义一个测试方法，用于测试在多级索引下DataFrame对象的字符串表示是否被截断
    def test_to_string_truncate_multilevel(self):
        # 创建一个包含多级索引的DataFrame对象
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        df = DataFrame(index=arrays, columns=arrays)
        # 设置选项上下文，限制最大显示行数和列数均为7
        with option_context("display.max_rows", 7, "display.max_columns", 7):
            # 断言多级索引下DataFrame的表示是否被同时垂直和水平截断
            assert has_doubly_truncated_repr(df)

    # 使用pytest的参数化装饰器定义测试方法，用于测试不同数据类型的Series对象的字符串表示是否被截断
    @pytest.mark.parametrize("dtype", ["object", "datetime64[us]"])
    def test_truncate_with_different_dtypes(self, dtype):
        # 创建一个包含不同数据类型的Series对象
        ser = Series(
            [datetime(2012, 1, 1)] * 10
            + [datetime(1012, 1, 2)]
            + [datetime(2012, 1, 3)] * 10,
            dtype=dtype,
        )
        # 设置选项上下文，限制最大显示行数为8
        with option_context("display.max_rows", 8):
            # 将Series对象转换为字符串表示，并进行断言检查特定数据类型是否存在于结果中
            result = str(ser)
        assert dtype in result

    # 定义一个测试方法，用于测试包含不同数据类型的DataFrame对象的字符串表示是否被截断
    def test_truncate_with_different_dtypes2(self):
        # 创建一个包含对象数据类型列的DataFrame对象
        df = DataFrame({"text": ["some words"] + [None] * 9}, dtype=object)
        # 设置选项上下文，限制最大显示行数为8，最大显示列数为3
        with option_context("display.max_rows", 8, "display.max_columns", 3):
            # 将DataFrame对象转换为字符串表示，并进行断言检查结果中是否包含特定内容
            result = str(df)
            assert "None" in result
            assert "NaN" not in result

    # 定义一个测试方法，用于测试多级索引DataFrame对象的部分截断字符串表示
    def test_truncate_with_different_dtypes_multiindex(self):
        # 创建一个DataFrame对象，并在其基础上构建具有多级索引的新对象
        df = DataFrame({"Vals": range(100)})
        frame = pd.concat([df], keys=["Sweep"], names=["Sweep", "Index"])
        # 获取完整DataFrame的字符串表示
        result = repr(frame)
        # 获取部分截断DataFrame的字符串表示
        result2 = repr(frame.iloc[:5])
        # 断言部分截断表示是否为完整表示的前缀
        assert result.startswith(result2)
    def test_datetimelike_frame(self):
        # 定义一个测试方法，用于测试日期时间类型的DataFrame
        df = DataFrame({"date": [Timestamp("20130101").tz_localize("UTC")] + [NaT] * 5})
        # 创建一个DataFrame对象，包含一个日期时间列和5个NaT（Not a Time）值

        with option_context("display.max_rows", 5):
            # 设置上下文环境，指定最大显示行数为5
            result = str(df)
            # 将DataFrame转换为字符串
            assert "2013-01-01 00:00:00+00:00" in result
            # 断言字符串中包含指定的日期时间格式
            assert "NaT" in result
            # 断言字符串中包含NaT
            assert "..." in result
            # 断言字符串中包含省略号
            assert "[6 rows x 1 columns]" in result
            # 断言字符串中包含特定的行列数信息

        dts = [Timestamp("2011-01-01", tz="US/Eastern")] * 5 + [NaT] * 5
        # 创建一个包含5个东部时间戳和5个NaT的日期时间列表
        df = DataFrame({"dt": dts, "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        # 创建一个包含日期时间列和另一个列的DataFrame对象

        with option_context("display.max_rows", 5):
            # 设置上下文环境，指定最大显示行数为5
            expected = (
                "                          dt   x\n"
                "0  2011-01-01 00:00:00-05:00   1\n"
                "1  2011-01-01 00:00:00-05:00   2\n"
                "..                       ...  ..\n"
                "8                        NaT   9\n"
                "9                        NaT  10\n\n"
                "[10 rows x 2 columns]"
            )
            # 期望的字符串，展示了DataFrame的特定格式和行列信息
            assert repr(df) == expected
            # 断言DataFrame的表现形式与期望的字符串相等

        dts = [NaT] * 5 + [Timestamp("2011-01-01", tz="US/Eastern")] * 5
        # 创建一个包含5个NaT和5个东部时间戳的日期时间列表
        df = DataFrame({"dt": dts, "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        # 创建一个包含日期时间列和另一个列的DataFrame对象

        with option_context("display.max_rows", 5):
            # 设置上下文环境，指定最大显示行数为5
            expected = (
                "                          dt   x\n"
                "0                        NaT   1\n"
                "1                        NaT   2\n"
                "..                       ...  ..\n"
                "8  2011-01-01 00:00:00-05:00   9\n"
                "9  2011-01-01 00:00:00-05:00  10\n\n"
                "[10 rows x 2 columns]"
            )
            # 期望的字符串，展示了DataFrame的特定格式和行列信息
            assert repr(df) == expected
            # 断言DataFrame的表现形式与期望的字符串相等

        dts = [Timestamp("2011-01-01", tz="Asia/Tokyo")] * 5 + [
            Timestamp("2011-01-01", tz="US/Eastern")
        ] * 5
        # 创建一个包含5个东京时间戳和5个东部时间戳的日期时间列表
        df = DataFrame({"dt": dts, "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        # 创建一个包含日期时间列和另一个列的DataFrame对象

        with option_context("display.max_rows", 5):
            # 设置上下文环境，指定最大显示行数为5
            expected = (
                "                           dt   x\n"
                "0   2011-01-01 00:00:00+09:00   1\n"
                "1   2011-01-01 00:00:00+09:00   2\n"
                "..                        ...  ..\n"
                "8   2011-01-01 00:00:00-05:00   9\n"
                "9   2011-01-01 00:00:00-05:00  10\n\n"
                "[10 rows x 2 columns]"
            )
            # 期望的字符串，展示了DataFrame的特定格式和行列信息
            assert repr(df) == expected
            # 断言DataFrame的表现形式与期望的字符串相等

    @pytest.mark.parametrize(
        "start_date",
        [
            "2017-01-01 23:59:59.999999999",
            "2017-01-01 23:59:59.99999999",
            "2017-01-01 23:59:59.9999999",
            "2017-01-01 23:59:59.999999",
            "2017-01-01 23:59:59.99999",
            "2017-01-01 23:59:59.9999",
        ],
    )
    # 使用pytest的参数化装饰器，传递不同的起始日期字符串作为参数
    def test_datetimeindex_highprecision(self, start_date):
        # GH19030
        # 检查高精度时间值（例如一天结束时的时间）是否包含在 DatetimeIndex 的表示中
        df = DataFrame({"A": date_range(start=start_date, freq="D", periods=5)})
        result = str(df)
        assert start_date in result

        # 创建一个日期范围，并使用其作为索引创建 DataFrame
        dti = date_range(start=start_date, freq="D", periods=5)
        df = DataFrame({"A": range(5)}, index=dti)
        result = str(df.index)
        assert start_date in result

    def test_string_repr_encoding(self, datapath):
        # 从指定路径读取 CSV 文件，使用 Latin-1 编码解析
        filepath = datapath("io", "parser", "data", "unicode_series.csv")
        df = read_csv(filepath, header=None, encoding="latin1")
        # 获取 DataFrame 的字符串表示形式
        repr(df)
        # 获取 DataFrame 列的字符串表示形式
        repr(df[1])

    def test_repr_corner(self):
        # 表示包含无穷大值（np.inf）的 DataFrame 不会出现问题
        df = DataFrame({"foo": [-np.inf, np.inf]})
        repr(df)

    def test_frame_info_encoding(self):
        # 使用特定的索引创建 DataFrame
        index = ["'Til There Was You (1997)", "ldum klaka (Cold Fever) (1994)"]
        with option_context("display.max_rows", 1):
            df = DataFrame(columns=["a", "b", "c"], index=index)
            # 获取 DataFrame 的字符串表示形式
            repr(df)
            # 获取 DataFrame 转置后的字符串表示形式
            repr(df.T)

    def test_wide_repr(self):
        # 设置特定的显示选项
        with option_context(
            "mode.sim_interactive",
            True,
            "display.show_dimensions",
            True,
            "display.max_columns",
            20,
        ):
            max_cols = get_option("display.max_columns")
            # 创建一个包含大量列的 DataFrame
            df = DataFrame([["a" * 25] * (max_cols - 1)] * 10)
            with option_context("display.expand_frame_repr", False):
                # 获取 DataFrame 紧凑表示形式的字符串
                rep_str = repr(df)

            # 断言确保在紧凑和扩展表示形式之间有差异
            assert f"10 rows x {max_cols - 1} columns" in rep_str
            with option_context("display.expand_frame_repr", True):
                # 获取 DataFrame 扩展表示形式的字符串
                wide_repr = repr(df)
            assert rep_str != wide_repr

            with option_context("display.width", 120):
                # 获取宽度调整后的扩展表示形式字符串
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_wide_columns(self):
        # 设置特定的显示选项
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            # 创建一个具有非常宽列名的 DataFrame
            df = DataFrame(
                np.random.default_rng(2).standard_normal((5, 3)),
                columns=["a" * 90, "b" * 90, "c" * 90],
            )
            # 获取 DataFrame 的字符串表示形式
            rep_str = repr(df)

            # 断言确保字符串表示形式的行数为 20
            assert len(rep_str.splitlines()) == 20
    # 定义一个测试方法，测试在宽格式下 DataFrame 的字符串表示是否正确显示命名索引。
    def test_wide_repr_named(self):
        # 使用上下文管理器设置一些显示选项，模拟交互模式和最大列数。
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            # 获取当前的最大列数设置
            max_cols = get_option("display.max_columns")
            # 创建一个 DataFrame，填充内容为单个字符串的列表，行数为10，列数为最大列数减1
            df = DataFrame([["a" * 25] * (max_cols - 1)] * 10)
            # 设置 DataFrame 的索引名称为 "DataFrame Index"
            df.index.name = "DataFrame Index"
            # 在显示选项上下文中，将 DataFrame 转换为字符串表示
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)
            # 再次在显示选项上下文中，使用展开的 DataFrame 表示形式生成字符串
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            # 断言展开的字符串表示与普通的字符串表示不同
            assert rep_str != wide_repr

            # 在更宽的显示宽度设置下，再次生成更宽的字符串表示
            with option_context("display.width", 150):
                wider_repr = repr(df)
                # 断言更宽的字符串表示比之前的展开的字符串表示要短
                assert len(wider_repr) < len(wide_repr)

            # 遍历展开的字符串表示的每13行，断言每行包含 "DataFrame Index"
            for line in wide_repr.splitlines()[1::13]:
                assert "DataFrame Index" in line

    # 定义一个测试方法，测试在宽格式下 DataFrame 的字符串表示是否正确显示多级索引。
    def test_wide_repr_multiindex(self):
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            # 创建一个多级索引，每级包含单个字符串，共两级，每级包含10个元素
            midx = MultiIndex.from_arrays([["a" * 5] * 10] * 2)
            # 获取当前的最大列数设置
            max_cols = get_option("display.max_columns")
            # 创建一个 DataFrame，填充内容为单个字符串的列表，行数为10，列数为最大列数减1，使用上述多级索引
            df = DataFrame([["a" * 25] * (max_cols - 1)] * 10, index=midx)
            # 设置 DataFrame 的索引级别名称为 ["Level 0", "Level 1"]
            df.index.names = ["Level 0", "Level 1"]
            # 在显示选项上下文中，将 DataFrame 转换为字符串表示
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)
            # 再次在显示选项上下文中，使用展开的 DataFrame 表示形式生成字符串
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            # 断言展开的字符串表示与普通的字符串表示不同
            assert rep_str != wide_repr

            # 在更宽的显示宽度设置下，再次生成更宽的字符串表示
            with option_context("display.width", 150):
                wider_repr = repr(df)
                # 断言更宽的字符串表示比之前的展开的字符串表示要短
                assert len(wider_repr) < len(wide_repr)

            # 遍历展开的字符串表示的每13行，断言每行包含 "Level 0 Level 1"
            for line in wide_repr.splitlines()[1::13]:
                assert "Level 0 Level 1" in line

    # 定义一个测试方法，测试在宽格式下 DataFrame 的字符串表示是否正确显示多级索引和多级列索引。
    def test_wide_repr_multiindex_cols(self):
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            # 获取当前的最大列数设置
            max_cols = get_option("display.max_columns")
            # 创建一个多级索引，每级包含单个字符串，共两级，每级包含10个元素
            midx = MultiIndex.from_arrays([["a" * 5] * 10] * 2)
            # 创建一个多级列索引，每级包含单个字符串，列数为最大列数减1，共两级
            mcols = MultiIndex.from_arrays([["b" * 3] * (max_cols - 1)] * 2)
            # 创建一个 DataFrame，填充内容为单个字符串的列表，行数为10，列数为最大列数减1，使用上述多级索引和多级列索引
            df = DataFrame([["c" * 25] * (max_cols - 1)] * 10, index=midx, columns=mcols)
            # 设置 DataFrame 的索引级别名称为 ["Level 0", "Level 1"]
            df.index.names = ["Level 0", "Level 1"]
            # 在显示选项上下文中，将 DataFrame 转换为字符串表示
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)
            # 再次在显示选项上下文中，使用展开的 DataFrame 表示形式生成字符串
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            # 断言展开的字符串表示与普通的字符串表示不同
            assert rep_str != wide_repr

        # 在更宽的显示宽度设置下，再次生成更宽的字符串表示
        with option_context("display.width", 150, "display.max_columns", 20):
            wider_repr = repr(df)
            # 断言更宽的字符串表示比之前的展开的字符串表示要短
            assert len(wider_repr) < len(wide_repr)
    # 测试宽格式的Unicode表示是否正常工作
    def test_wide_repr_unicode(self):
        # 在指定选项环境下，设置模式为交互式，并设置最大列数为20
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            max_cols = 20
            # 创建一个DataFrame，包含多行每行都是包含25个字符'a'的列表，共(max_cols-1)行
            df = DataFrame([["a" * 25] * 10] * (max_cols - 1))
            # 在关闭行扩展显示的选项环境下，获取DataFrame的表示形式
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)
            # 在开启行扩展显示的选项环境下，获取DataFrame的表示形式
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            # 断言两种表示形式不相同
            assert rep_str != wide_repr

            # 在设置显示宽度为150的选项环境下，获取DataFrame的表示形式
            with option_context("display.width", 150):
                wider_repr = repr(df)
                # 断言宽表示形式的长度小于行扩展显示的宽表示形式的长度
                assert len(wider_repr) < len(wide_repr)

    # 测试宽格式的长列是否正常工作
    def test_wide_repr_wide_long_columns(self):
        # 在指定选项环境下，设置模式为交互式
        with option_context("mode.sim_interactive", True):
            # 创建一个DataFrame，包含两列，其中一列有30个字符'a'，另一列有70或80个字符'c'或'd'
            df = DataFrame({"a": ["a" * 30, "b" * 30], "b": ["c" * 70, "d" * 80]})

            # 获取DataFrame的表示形式
            result = repr(df)
            # 断言结果中包含字符串'ccccc'和'ddddd'
            assert "ccccc" in result
            assert "ddddd" in result

    # 测试长序列是否正常工作
    def test_long_series(self):
        n = 1000
        # 创建一个包含1000个随机整数的Series，索引为形如's0000'到's0999'的字符串
        s = Series(
            np.random.default_rng(2).integers(-50, 50, n),
            index=[f"s{x:04d}" for x in range(n)],
            dtype="int64",
        )

        # 获取Series的字符串表示形式
        str_rep = str(s)
        # 在字符串表示形式中查找字符串'dtype'出现的次数
        nmatches = len(re.findall("dtype", str_rep))
        # 断言找到'dtype'的次数为1
        assert nmatches == 1

    # 测试to_string方法在ASCII错误时的行为
    def test_to_string_ascii_error(self):
        data = [
            (
                "0  ",
                "                        .gitignore ",
                "     5 ",
                " \xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2",
            )
        ]
        # 创建一个包含ASCII错误的DataFrame
        df = DataFrame(data)

        # 获取DataFrame的表示形式
        repr(df)
    # 定义一个测试函数，用于测试显示数据框的维度设置

    def test_show_dimensions(self):
        # 创建一个数据框，内容为数字123，行索引为10到14，列索引为0到29
        df = DataFrame(123, index=range(10, 15), columns=range(30))

        # 设置上下文选项，调整数据框的显示选项，包括最大显示行数、最大显示列数、显示宽度等
        with option_context(
            "display.max_rows",   # 设置最大显示行数为10行
            10,
            "display.max_columns",   # 设置最大显示列数为40列
            40,
            "display.width",   # 设置显示宽度为500像素
            500,
            "display.expand_frame_repr",   # 设置扩展显示数据框的格式为"info"
            "info",
            "display.show_dimensions",   # 开启显示数据框的维度信息
            True,
        ):
            # 断言数据框的字符串表示中包含"5 rows"
            assert "5 rows" in str(df)
            # 断言数据框的 HTML 表示中包含"5 rows"
            assert "5 rows" in df._repr_html_()

        # 更改上下文选项，关闭显示数据框的维度信息
        with option_context(
            "display.max_rows",   # 设置最大显示行数为10行
            10,
            "display.max_columns",   # 设置最大显示列数为40列
            40,
            "display.width",   # 设置显示宽度为500像素
            500,
            "display.expand_frame_repr",   # 设置扩展显示数据框的格式为"info"
            "info",
            "display.show_dimensions",   # 关闭显示数据框的维度信息
            False,
        ):
            # 断言数据框的字符串表示中不包含"5 rows"
            assert "5 rows" not in str(df)
            # 断言数据框的 HTML 表示中不包含"5 rows"
            assert "5 rows" not in df._repr_html_()

        # 使用不同的上下文选项，设置最大显示行数和列数，并将数据框进行截断显示
        with option_context(
            "display.max_rows",   # 设置最大显示行数为2行
            2,
            "display.max_columns",   # 设置最大显示列数为2列
            2,
            "display.width",   # 设置显示宽度为500像素
            500,
            "display.expand_frame_repr",   # 设置扩展显示数据框的格式为"info"
            "info",
            "display.show_dimensions",   # 截断显示数据框的维度信息
            "truncate",
        ):
            # 断言数据框的字符串表示中包含"5 rows"
            assert "5 rows" in str(df)
            # 断言数据框的 HTML 表示中包含"5 rows"
            assert "5 rows" in df._repr_html_()

        # 再次更改上下文选项，设置最大显示行数和列数，并关闭数据框的维度信息显示
        with option_context(
            "display.max_rows",   # 设置最大显示行数为10行
            10,
            "display.max_columns",   # 设置最大显示列数为40列
            40,
            "display.width",   # 设置显示宽度为500像素
            500,
            "display.expand_frame_repr",   # 设置扩展显示数据框的格式为"info"
            "info",
            "display.show_dimensions",   # 关闭显示数据框的维度信息
            "truncate",
        ):
            # 断言数据框的字符串表示中不包含"5 rows"
            assert "5 rows" not in str(df)
            # 断言数据框的 HTML 表示中不包含"5 rows"
            assert "5 rows" not in df._repr_html_()

    # 定义一个测试函数，用于测试数据框的信息显示
    def test_info_repr(self):
        # 获取终端的尺寸，用于适应终端的显示设置
        term_width, term_height = get_terminal_size()

        # 设置最大显示行数和列数，根据终端宽度调整最大列数
        max_rows = 60
        max_cols = 20 + (max(term_width, 80) - 80) // 4

        # 创建一个高度超出显示范围的数据框
        h, w = max_rows + 1, max_cols - 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        # 断言数据框是否垂直截断显示
        assert has_vertically_truncated_repr(df)

        # 使用上下文选项设置数据框的大数据显示格式为"info"
        with option_context("display.large_repr", "info"):
            # 断言数据框是否使用"info"格式显示
            assert has_info_repr(df)

        # 创建一个宽度超出显示范围的数据框
        h, w = max_rows - 1, max_cols + 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        # 断言数据框是否水平截断显示
        assert has_horizontally_truncated_repr(df)

        # 使用上下文选项设置数据框的大数据显示格式为"info"，并限制最大列数
        with option_context(
            "display.large_repr",   # 设置大数据显示格式为"info"
            "info",
            "display.max_columns",   # 设置最大显示列数为max_cols
            max_cols
        ):
            # 断言数据框是否使用"info"格式显示
            assert has_info_repr(df)
    # 定义测试方法 test_info_repr_max_cols，用于测试 DataFrame 的显示选项设置
    def test_info_repr_max_cols(self):
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        # 使用上下文管理器设置显示选项，以检查是否有非详细信息的表示形式
        with option_context(
            "display.large_repr",
            "info",
            "display.max_columns",
            1,
            "display.max_info_columns",
            4,
        ):
            # 断言 DataFrame 是否具有非详细信息的表示形式
            assert has_non_verbose_info_repr(df)

        # 使用上下文管理器设置不同的显示选项，以检查是否没有非详细信息的表示形式
        with option_context(
            "display.large_repr",
            "info",
            "display.max_columns",
            1,
            "display.max_info_columns",
            5,
        ):
            # 断言 DataFrame 是否没有非详细信息的表示形式
            assert not has_non_verbose_info_repr(df)

        # FIXME: 不要留下被注释掉的代码
        # 测试详细信息覆盖
        # set_option('display.max_info_columns', 4)  # 超过了限制

    # 定义测试方法 test_pprint_pathological_object，用于测试特定情况下的对象打印
    def test_pprint_pathological_object(self):
        """
        如果测试失败，至少不会挂起。
        """
        # 定义一个简单的类 A，重载 __getitem__ 方法返回固定值 3
        class A:
            def __getitem__(self, key):
                return 3  # 显然是简化的示例

        # 创建一个包含 A 类实例的 DataFrame
        df = DataFrame([A()])
        # 打印 DataFrame 的表示形式，目的是确保不会导致崩溃
        repr(df)  # 只需确保不崩溃即可

    # 定义测试方法 test_float_trim_zeros，用于测试浮点数显示中的零修剪
    def test_float_trim_zeros(self):
        # 定义一组浮点数值
        vals = [
            2.08430917305e10,
            3.52205017305e10,
            2.30674817305e10,
            2.03954217305e10,
            5.59897817305e10,
        ]
        # 初始化跳过标志为 True
        skip = True
        # 对 DataFrame 的表示形式进行逐行检查
        for line in repr(DataFrame({"A": vals})).split("\n")[:-2]:
            # 如果行以 "dtype:" 开头，则跳过该行
            if line.startswith("dtype:"):
                continue
            # 根据条件检查行中是否包含期望的浮点数表示格式
            if _three_digit_exp():
                assert ("+010" in line) or skip
            else:
                assert ("+10" in line) or skip
            skip = False

    # 使用 pytest 的参数化装饰器标记的测试方法，测试不同输入下的字符串浮点数截断
    @pytest.mark.parametrize(
        "data, expected",
        [
            (["3.50"], "0    3.50\ndtype: object"),
            ([1.20, "1.00"], "0     1.2\n1    1.00\ndtype: object"),
            ([np.nan], "0   NaN\ndtype: float64"),
            ([None], "0    None\ndtype: object"),
            (["3.50", np.nan], "0    3.50\n1     NaN\ndtype: object"),
            ([3.50, np.nan], "0    3.5\n1    NaN\ndtype: float64"),
            ([3.50, np.nan, "3.50"], "0     3.5\n1     NaN\n2    3.50\ndtype: object"),
            ([3.50, None, "3.50"], "0     3.5\n1    None\n2    3.50\ndtype: object"),
        ],
    )
    # 定义测试方法 test_repr_str_float_truncation，用于测试字符串浮点数的截断表示
    def test_repr_str_float_truncation(self, data, expected, using_infer_string):
        # GH#38708
        # 创建一个 Series 对象，数据为输入的 data，数据类型根据数据内容决定
        series = Series(data, dtype=object if "3.50" in data else None)
        # 获取 Series 对象的字符串表示形式
        result = repr(series)
        # 断言获取的结果与期望结果一致
        assert result == expected

    # 使用 pytest 的参数化装饰器标记的测试方法，测试不同浮点数格式化下对象列的表示形式
    @pytest.mark.parametrize(
        "float_format,expected",
        [
            ("{:,.0f}".format, "0   1,000\n1    test\ndtype: object"),
            ("{:.4f}".format, "0   1000.0000\n1        test\ndtype: object"),
        ],
    )
    # 定义测试方法 test_repr_float_format_in_object_col，用于测试对象列中浮点数的格式化显示
    def test_repr_float_format_in_object_col(self, float_format, expected):
        # GH#40024
        # 创建一个包含浮点数和字符串的 Series 对象
        df = Series([1000.0, "test"])
        # 使用上下文管理器设置显示选项，包括浮点数格式化选项
        with option_context("display.float_format", float_format):
            # 获取 Series 对象的字符串表示形式
            result = repr(df)

        # 断言获取的结果与期望结果一致
        assert result == expected
    def test_period(self):
        # GH 12615
        # 创建一个 DataFrame 对象，包含三列数据，其中列"A"使用 pd.period_range 生成时间周期，列"B"包含不同形式的时间周期
        df = DataFrame(
            {
                "A": pd.period_range("2013-01", periods=4, freq="M"),
                "B": [
                    pd.Period("2011-01", freq="M"),
                    pd.Period("2011-02-01", freq="D"),
                    pd.Period("2011-03-01 09:00", freq="h"),
                    pd.Period("2011-04", freq="M"),
                ],
                "C": list("abcd"),
            }
        )
        # 期望的 DataFrame 字符串表示，用于断言测试结果是否符合期望
        exp = (
            "         A                 B  C\n"
            "0  2013-01           2011-01  a\n"
            "1  2013-02        2011-02-01  b\n"
            "2  2013-03  2011-03-01 09:00  c\n"
            "3  2013-04           2011-04  d"
        )
        # 断言 DataFrame 转换为字符串后是否与期望的字符串相同
        assert str(df) == exp

    @pytest.mark.parametrize(
        "length, max_rows, min_rows, expected",
        [
            (10, 10, 10, 10),
            (10, 10, None, 10),
            (10, 8, None, 8),
            (20, 30, 10, 30),  # max_rows > len(frame), hence max_rows
            (50, 30, 10, 10),  # max_rows < len(frame), hence min_rows
            (100, 60, 10, 10),  # same
            (60, 60, 10, 60),  # edge case
            (61, 60, 10, 10),  # edge case
        ],
    )
    def test_max_rows_fitted(self, length, min_rows, max_rows, expected):
        """Check that display logic is correct.

        GH #37359

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options
        """
        # 创建 DataFrameFormatter 对象，用于格式化 DataFrame 的显示
        formatter = fmt.DataFrameFormatter(
            DataFrame(np.random.default_rng(2).random((length, 3))),
            max_rows=max_rows,
            min_rows=min_rows,
        )
        # 获取最适合显示的行数
        result = formatter.max_rows_fitted
        # 断言结果是否与预期相同
        assert result == expected
def gen_series_formatting():
    # 创建包含 100 个 "a" 的 Series 对象
    s1 = Series(["a"] * 100)
    # 创建包含 100 个 "ab" 的 Series 对象
    s2 = Series(["ab"] * 100)
    # 创建包含字符串列表的 Series 对象
    s3 = Series(["a", "ab", "abc", "abcd", "abcde", "abcdef"])
    # 创建 s3 的逆序 Series 对象
    s4 = s3[::-1]
    # 组成包含多个 Series 的字典
    test_sers = {"onel": s1, "twol": s2, "asc": s3, "desc": s4}
    # 返回包含不同格式 Series 对象的字典
    return test_sers


class TestSeriesFormatting:
    def test_freq_name_separation(self):
        # 创建一个包含随机数的 Series 对象，设置索引为日期范围
        s = Series(
            np.random.default_rng(2).standard_normal(10),
            index=date_range("1/1/2000", periods=10),
            name=0,
        )

        # 调用 repr 函数，将 Series 对象转换为字符串表示形式
        result = repr(s)
        # 断言字符串中包含指定的文本 "Freq: D, Name: 0"
        assert "Freq: D, Name: 0" in result

    def test_unicode_name_in_footer(self):
        # 创建一个带有 Unicode 名称的 Series 对象
        s = Series([1, 2], name="\u05e2\u05d1\u05e8\u05d9\u05ea")
        # 创建 SeriesFormatter 对象，设置名称为 Unicode 名称
        sf = fmt.SeriesFormatter(s, name="\u05e2\u05d1\u05e8\u05d9\u05ea")
        # 调用 _get_footer 方法，验证不引发异常
        sf._get_footer()  # should not raise exception

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="Fixup when arrow is default"
    )
    def test_float_trim_zeros(self):
        # 创建包含浮点数的列表
        vals = [
            2.08430917305e10,
            3.52205017305e10,
            2.30674817305e10,
            2.03954217305e10,
            5.59897817305e10,
        ]
        # 对 Series 对象的 repr 字符串按行进行处理
        for line in repr(Series(vals)).split("\n"):
            if line.startswith("dtype:"):
                continue
            # 根据条件断言字符串中是否包含指定内容
            if _three_digit_exp():
                assert "+010" in line
            else:
                assert "+10" in line

    @pytest.mark.parametrize(
        "start_date",
        [
            "2017-01-01 23:59:59.999999999",
            "2017-01-01 23:59:59.99999999",
            "2017-01-01 23:59:59.9999999",
            "2017-01-01 23:59:59.999999",
            "2017-01-01 23:59:59.99999",
            "2017-01-01 23:59:59.9999",
        ],
    )
    def test_datetimeindex_highprecision(self, start_date):
        # GH19030
        # 检查高精度时间值是否包含在 DatetimeIndex 的 repr 中
        s1 = Series(date_range(start=start_date, freq="D", periods=5))
        result = str(s1)
        # 断言结果字符串中包含指定的起始日期
        assert start_date in result

        # 创建 DatetimeIndex 对象
        dti = date_range(start=start_date, freq="D", periods=5)
        s2 = Series(3, index=dti)
        result = str(s2.index)
        # 断言结果字符串中包含指定的起始日期
        assert start_date in result

    def test_mixed_datetime64(self):
        # 创建包含整数和日期字符串的 DataFrame 对象
        df = DataFrame({"A": [1, 2], "B": ["2012-01-01", "2012-01-02"]})
        # 将 DataFrame 中的 "B" 列转换为 datetime64 类型
        df["B"] = pd.to_datetime(df.B)

        # 调用 repr 函数，将 DataFrame 的一行转换为字符串表示形式
        result = repr(df.loc[0])
        # 断言结果字符串中包含指定的日期字符串
        assert "2012-01-01" in result
    # 定义一个测试方法，测试期间数据的显示和格式化

    def test_period(self):
        # GH 12615
        # 创建一个时间段范围，从"2013-01"开始，周期为6个月，频率为每月一次
        index = pd.period_range("2013-01", periods=6, freq="M")
        # 创建一个 Series 对象，使用整数数组作为数据，时间段作为索引
        s = Series(np.arange(6, dtype="int64"), index=index)
        # 期望的字符串表示
        exp = (
            "2013-01    0\n"
            "2013-02    1\n"
            "2013-03    2\n"
            "2013-04    3\n"
            "2013-05    4\n"
            "2013-06    5\n"
            "Freq: M, dtype: int64"
        )
        # 断言期望的字符串与 Series 对象转换的字符串相等
        assert str(s) == exp

        # 创建一个只包含索引的 Series 对象
        s = Series(index)
        # 期望的字符串表示
        exp = (
            "0    2013-01\n"
            "1    2013-02\n"
            "2    2013-03\n"
            "3    2013-04\n"
            "4    2013-05\n"
            "5    2013-06\n"
            "dtype: period[M]"
        )
        # 断言期望的字符串与 Series 对象转换的字符串相等
        assert str(s) == exp

        # 测试包含混合频率的时间段
        s = Series(
            [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02-01", freq="D"),
                pd.Period("2011-03-01 09:00", freq="h"),
            ]
        )
        # 期望的字符串表示
        exp = (
            "0             2011-01\n1          2011-02-01\n"
            "2    2011-03-01 09:00\ndtype: object"
        )
        # 断言期望的字符串与 Series 对象转换的字符串相等
        assert str(s) == exp

    # 测试多级索引的最大显示数量
    def test_max_multi_index_display(self):
        # GH 7101

        # 文档示例（indexing.rst）

        # 创建多级索引
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = list(zip(*arrays))
        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        # 创建一个 Series 对象，使用随机正态分布的数据，多级索引
        s = Series(np.random.default_rng(2).standard_normal(8), index=index)

        # 使用不同的最大行数设置，断言字符串表示的行数
        with option_context("display.max_rows", 10):
            assert len(str(s).split("\n")) == 10
        with option_context("display.max_rows", 3):
            assert len(str(s).split("\n")) == 5
        with option_context("display.max_rows", 2):
            assert len(str(s).split("\n")) == 5
        with option_context("display.max_rows", 1):
            assert len(str(s).split("\n")) == 4
        with option_context("display.max_rows", 0):
            assert len(str(s).split("\n")) == 10

        # 创建一个没有索引的 Series 对象
        s = Series(np.random.default_rng(2).standard_normal(8), None)

        # 使用不同的最大行数设置，断言字符串表示的行数
        with option_context("display.max_rows", 10):
            assert len(str(s).split("\n")) == 9
        with option_context("display.max_rows", 3):
            assert len(str(s).split("\n")) == 4
        with option_context("display.max_rows", 2):
            assert len(str(s).split("\n")) == 4
        with option_context("display.max_rows", 1):
            assert len(str(s).split("\n")) == 3
        with option_context("display.max_rows", 0):
            assert len(str(s).split("\n")) == 9

    # 确保问题 #8532 已修复
    # 定义测试方法，验证 Series 对象的字符串表示格式是否一致
    def test_consistent_format(self):
        # 创建一个包含大量重复元素的 Series 对象
        s = Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9999, 1, 1] * 10)
        # 设置显示选项：最大行数为 10，不显示维度信息
        with option_context("display.max_rows", 10, "display.show_dimensions", False):
            # 获取 Series 对象的字符串表示
            res = repr(s)
        # 预期的字符串表示，包含多行数值和 dtype 信息
        exp = (
            "0      1.0000\n1      1.0000\n2      1.0000\n3      "
            "1.0000\n4      1.0000\n        ...  \n125    "
            "1.0000\n126    1.0000\n127    0.9999\n128    "
            "1.0000\n129    1.0000\ndtype: float64"
        )
        # 断言实际结果与预期结果一致
        assert res == exp

    # 定义方法检查 Series 对象的字符串表示是否具有相同的列数
    def chck_ncols(self, s):
        # 将 Series 对象的字符串表示按行分割，并过滤掉带有连续小数点的行
        lines = [
            line for line in repr(s).split("\n") if not re.match(r"[^\.]*\.+", line)
        ][:-1]
        # 计算不同行长度的集合，即列数的集合
        ncolsizes = len({len(line.strip()) for line in lines})
        # 断言只有一个不同长度的列存在
        assert ncolsizes == 1

    # 使用 pytest.mark.xfail 标记，当条件 using_pyarrow_string_dtype() 成立时，标记为预期失败
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="change when arrow is default"
    )
    # 定义测试方法，验证特定格式的 Series 对象的字符串表示
    def test_format_explicit(self):
        # 生成一组格式化的 Series 对象
        test_sers = gen_series_formatting()
        # 对每个 Series 对象执行以下操作：
        with option_context("display.max_rows", 4, "display.show_dimensions", False):
            # 获取 Series 对象的字符串表示
            res = repr(test_sers["onel"])
            # 预期的字符串表示，包含一定格式的数值和 dtype 信息
            exp = "0     a\n1     a\n     ..\n98    a\n99    a\ndtype: object"
            # 断言实际结果与预期结果一致
            assert exp == res
            # 重复以上步骤，针对不同的 Series 对象
            res = repr(test_sers["twol"])
            exp = "0     ab\n1     ab\n      ..\n98    ab\n99    ab\ndtype: object"
            assert exp == res
            res = repr(test_sers["asc"])
            exp = (
                "0         a\n1        ab\n      ...  \n4     abcde\n5    "
                "abcdef\ndtype: object"
            )
            assert exp == res
            res = repr(test_sers["desc"])
            exp = (
                "5    abcdef\n4     abcde\n      ...  \n1        ab\n0         "
                "a\ndtype: object"
            )
            assert exp == res

    # 定义测试方法，验证生成的多个 Series 对象的字符串表示是否具有相同的列数
    def test_ncols(self):
        # 生成一组格式化的 Series 对象
        test_sers = gen_series_formatting()
        # 对每个 Series 对象执行列数检查
        for s in test_sers.values():
            self.chck_ncols(s)

    # 定义测试方法，验证当 display.max_rows 设置为 1 时，Series 对象的字符串表示是否正确截断和省略
    def test_max_rows_eq_one(self):
        # 创建一个包含整数的 Series 对象
        s = Series(range(10), dtype="int64")
        # 设置显示选项：最大行数为 1
        with option_context("display.max_rows", 1):
            # 获取 Series 对象的字符串表示，并按行分割
            strrepr = repr(s).split("\n")
        # 第一行预期结果为 ["0", "0"]
        exp1 = ["0", "0"]
        # 实际结果的第一行
        res1 = strrepr[0].split()
        # 断言第一行实际结果与预期结果一致
        assert exp1 == res1
        # 第二行预期结果为 [".."]
        exp2 = [".."]
        # 实际结果的第二行
        res2 = strrepr[1].split()
        # 断言第二行实际结果与预期结果一致
        assert exp2 == res2

    # 定义测试方法，验证当显示选项设置为最大行数为 2 时，Series 对象的字符串表示是否正确截断和省略
    def test_truncate_ndots(self):
        # 定义函数，返回字符串中连续小数点的数量
        def getndots(s):
            return len(re.match(r"[^\.]*(\.*)", s).groups()[0])

        # 创建一个包含整数的 Series 对象
        s = Series([0, 2, 3, 6])
        # 设置显示选项：最大行数为 2
        with option_context("display.max_rows", 2):
            # 获取 Series 对象的字符串表示，并将换行符替换为空格
            strrepr = repr(s).replace("\n", "")
        # 断言字符串中连续小数点的数量为 2
        assert getndots(strrepr) == 2

        # 创建一个包含整数的 Series 对象
        s = Series([0, 100, 200, 400])
        # 设置显示选项：最大行数为 2
        with option_context("display.max_rows", 2):
            # 获取 Series 对象的字符串表示，并将换行符替换为空格
            strrepr = repr(s).replace("\n", "")
        # 断言字符串中连续小数点的数量为 3
        assert getndots(strrepr) == 3
    # 定义测试函数 test_show_dimensions，用于测试显示设置是否正确
    def test_show_dimensions(self):
        # gh-7117，标识这个测试与GitHub上的 issue-7117 相关联

        # 创建一个包含0到4的Series对象
        s = Series(range(5))

        # 断言在s的字符串表示中不包含"Length"
        assert "Length" not in repr(s)

        # 使用选项上下文设置"display.max_rows"为4，断言在s的字符串表示中包含"Length"
        with option_context("display.max_rows", 4):
            assert "Length" in repr(s)

        # 使用选项上下文设置"display.show_dimensions"为True，断言在s的字符串表示中包含"Length"
        with option_context("display.show_dimensions", True):
            assert "Length" in repr(s)

        # 使用选项上下文设置"display.max_rows"为4和"display.show_dimensions"为False，
        # 断言在s的字符串表示中不包含"Length"
        with option_context("display.max_rows", 4, "display.show_dimensions", False):
            assert "Length" not in repr(s)

    # 定义测试函数 test_repr_min_rows，用于测试在不同的行数设置下字符串表示的截断效果
    def test_repr_min_rows(self):
        # 创建一个包含0到19的Series对象
        s = Series(range(20))

        # 默认设置下，即使超过min_rows，字符串表示中不应包含".."
        assert ".." not in repr(s)

        # 创建一个包含0到60的Series对象
        s = Series(range(61))

        # 默认设置下，当max_rows为60时，如果超过，则字符串表示中应包含".."
        assert ".." in repr(s)

        # 使用选项上下文设置"display.max_rows"为10和"display.min_rows"为4，
        # 预期结果是前两行后会被截断
        with option_context("display.max_rows", 10, "display.min_rows", 4):
            assert ".." in repr(s)  # 字符串表示中应包含".."
            assert "2  " not in repr(s)  # 字符串表示中不应包含"2  "

        # 使用选项上下文设置"display.max_rows"为12和"display.min_rows"为None，
        # 当min_rows设置为None时，表示最小行数会跟随max_rows的值
        with option_context("display.max_rows", 12, "display.min_rows", None):
            assert "5      5" in repr(s)  # 字符串表示中应包含"5      5"

        # 使用选项上下文设置"display.max_rows"为10和"display.min_rows"为12，
        # 当min_rows设置值高于max_rows时，应该使用max_rows作为最小值
        with option_context("display.max_rows", 10, "display.min_rows", 12):
            assert "5      5" not in repr(s)  # 字符串表示中不应包含"5      5"

        # 使用选项上下文设置"display.max_rows"为None和"display.min_rows"为12，
        # 当max_rows为None时，字符串表示不应该被截断
        with option_context("display.max_rows", None, "display.min_rows", 12):
            assert ".." not in repr(s)  # 字符串表示中不应包含".."
class TestGenericArrayFormatter:
    def test_1d_array(self):
        # 使用 _GenericArrayFormatter 处理没有专用格式化器的类型，如 np.bool_
        obj = fmt._GenericArrayFormatter(np.array([True, False]))
        # 获取格式化后的结果
        res = obj.get_result()
        # 断言结果长度为2
        assert len(res) == 2
        # 结果应右对齐
        assert res[0] == "  True"
        assert res[1] == " False"

    def test_2d_array(self):
        # 使用 _GenericArrayFormatter 处理二维数组
        obj = fmt._GenericArrayFormatter(np.array([[True, False], [False, True]]))
        # 获取格式化后的结果
        res = obj.get_result()
        # 断言结果长度为2
        assert len(res) == 2
        # 断言结果内容
        assert res[0] == " [True, False]"
        assert res[1] == " [False, True]"

    def test_3d_array(self):
        # 使用 _GenericArrayFormatter 处理三维数组
        obj = fmt._GenericArrayFormatter(
            np.array([[[True, True], [False, False]], [[False, True], [True, False]]])
        )
        # 获取格式化后的结果
        res = obj.get_result()
        # 断言结果长度为2
        assert len(res) == 2
        # 断言结果内容
        assert res[0] == " [[True, True], [False, False]]"
        assert res[1] == " [[False, True], [True, False]]"

    def test_2d_extension_type(self):
        # GH 33770

        # 定义一个存根扩展类型，仅包含足够的代码以运行 Series.__repr__()
        class DtypeStub(pd.api.extensions.ExtensionDtype):
            @property
            def type(self):
                return np.ndarray

            @property
            def name(self):
                return "DtypeStub"

        class ExtTypeStub(pd.api.extensions.ExtensionArray):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, ix):
                return [ix == 1, ix == 0]

            @property
            def dtype(self):
                return DtypeStub()

        # 创建一个 Series 对象，使用 ExtTypeStub 作为其数据源
        series = Series(ExtTypeStub(), copy=False)
        # 获取 Series 的字符串表示形式，此行在修复 #33770 之前会导致崩溃
        res = repr(series)
        # 期望的输出
        expected = "\n".join(
            ["0    [False True]", "1    [True False]", "dtype: DtypeStub"]
        )
        # 断言结果符合预期输出
        assert res == expected


def _three_digit_exp():
    # 返回结果是否为科学计数法表示的三位数字
    return f"{1.7e8:.4g}" == "1.7e+008"


class TestFloatArrayFormatter:
    def test_misc(self):
        # 使用 FloatArrayFormatter 处理空的 float64 数组
        obj = fmt.FloatArrayFormatter(np.array([], dtype=np.float64))
        # 获取格式化后的结果
        result = obj.get_result()
        # 断言结果长度为0
        assert len(result) == 0

    def test_format(self):
        # 使用 FloatArrayFormatter 处理包含整数的 float64 数组
        obj = fmt.FloatArrayFormatter(np.array([12, 0], dtype=np.float64))
        # 获取格式化后的结果
        result = obj.get_result()
        # 断言结果内容
        assert result[0] == " 12.0"
        assert result[1] == "  0.0"

    def test_output_display_precision_trailing_zeroes(self):
        # Issue #20359: 在没有小数点的情况下修剪尾随零

        # 当显示精度设置为零时发生
        with option_context("display.precision", 0):
            # 创建一个 Series 对象
            s = Series([840.0, 4200.0])
            # 期望的输出
            expected_output = "0     840\n1    4200\ndtype: float64"
            # 断言 Series 对象的字符串表示形式符合预期输出
            assert str(s) == expected_output
    @pytest.mark.parametrize(
        "value,expected",
        [
            ([9.4444], "   0\n0  9"),
            ([0.49], "       0\n0  5e-01"),
            ([10.9999], "    0\n0  11"),
            ([9.5444, 9.6], "    0\n0  10\n1  10"),
            ([0.46, 0.78, -9.9999], "       0\n0  5e-01\n1  8e-01\n2 -1e+01"),
        ],
    )
    # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试数据
    def test_set_option_precision(self, value, expected):
        # Issue #30122
        # 对于问题编号为 #30122 的问题进行测试

        # 使用 option_context 设置显示精度为 0
        with option_context("display.precision", 0):
            # 创建 DataFrame 对象 df_value，传入测试数据 value
            df_value = DataFrame(value)
            # 断言 DataFrame 对象转换为字符串后与期望字符串 expected 相等
            assert str(df_value) == expected
    def test_output_significant_digits(self):
        # Issue #9764

        # In case default display precision changes:
        with option_context("display.precision", 6):
            # 创建一个 DataFrame，用于展示 Issue #9764 中的例子
            d = DataFrame(
                {
                    "col1": [
                        9.999e-8,
                        1e-7,
                        1.0001e-7,
                        2e-7,
                        4.999e-7,
                        5e-7,
                        5.0001e-7,
                        6e-7,
                        9.999e-7,
                        1e-6,
                        1.0001e-6,
                        2e-6,
                        4.999e-6,
                        5e-6,
                        5.0001e-6,
                        6e-6,
                    ]
                }
            )

            # 预期的输出结果，包含不同的行范围和显示精度
            expected_output = {
                (0, 6): "           col1\n"
                "0  9.999000e-08\n"
                "1  1.000000e-07\n"
                "2  1.000100e-07\n"
                "3  2.000000e-07\n"
                "4  4.999000e-07\n"
                "5  5.000000e-07",
                (1, 6): "           col1\n"
                "1  1.000000e-07\n"
                "2  1.000100e-07\n"
                "3  2.000000e-07\n"
                "4  4.999000e-07\n"
                "5  5.000000e-07",
                (1, 8): "           col1\n"
                "1  1.000000e-07\n"
                "2  1.000100e-07\n"
                "3  2.000000e-07\n"
                "4  4.999000e-07\n"
                "5  5.000000e-07\n"
                "6  5.000100e-07\n"
                "7  6.000000e-07",
                (8, 16): "            col1\n"
                "8   9.999000e-07\n"
                "9   1.000000e-06\n"
                "10  1.000100e-06\n"
                "11  2.000000e-06\n"
                "12  4.999000e-06\n"
                "13  5.000000e-06\n"
                "14  5.000100e-06\n"
                "15  6.000000e-06",
                (9, 16): "        col1\n"
                "9   0.000001\n"
                "10  0.000001\n"
                "11  0.000002\n"
                "12  0.000005\n"
                "13  0.000005\n"
                "14  0.000005\n"
                "15  0.000006",
            }

            # 遍历预期输出结果并断言 DataFrame 的字符串表示与预期相同
            for (start, stop), v in expected_output.items():
                assert str(d[start:stop]) == v

    def test_too_long(self):
        # GH 10451

        # 当默认显示精度改变时：
        with option_context("display.precision", 4):
            # 需要一个大于 1e6 的数字和通常格式化后长度大于 display.precision + 6 的内容
            df = DataFrame({"x": [12345.6789]})
            # 断言 DataFrame 的字符串表示与预期相同
            assert str(df) == "            x\n0  12345.6789"
            df = DataFrame({"x": [2e6]})
            # 断言 DataFrame 的字符串表示与预期相同
            assert str(df) == "           x\n0  2000000.0"
            df = DataFrame({"x": [12345.6789, 2e6]})
            # 断言 DataFrame 的字符串表示与预期相同
            assert str(df) == "            x\n0  1.2346e+04\n1  2.0000e+06"
class TestTimedelta64Formatter:
    # 测试类，用于测试时间间隔格式化功能

    def test_days(self):
        # 测试处理天数单位的时间间隔格式化
        x = pd.to_timedelta(list(range(5)) + [NaT], unit="D")._values
        # 将时间间隔转换为 Timedelta64 格式，并获取其内部值
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == "0 days"
        assert result[1].strip() == "1 days"

        result = fmt._Timedelta64Formatter(x[1:2]).get_result()
        assert result[0].strip() == "1 days"

        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == "0 days"
        assert result[1].strip() == "1 days"

        result = fmt._Timedelta64Formatter(x[1:2]).get_result()
        assert result[0].strip() == "1 days"

    def test_days_neg(self):
        # 测试处理负数天数单位的时间间隔格式化
        x = pd.to_timedelta(list(range(5)) + [NaT], unit="D")._values
        result = fmt._Timedelta64Formatter(-x).get_result()
        assert result[0].strip() == "0 days"
        assert result[1].strip() == "-1 days"

    def test_subdays(self):
        # 测试处理秒数单位的时间间隔格式化
        y = pd.to_timedelta(list(range(5)) + [NaT], unit="s")._values
        result = fmt._Timedelta64Formatter(y).get_result()
        assert result[0].strip() == "0 days 00:00:00"
        assert result[1].strip() == "0 days 00:00:01"

    def test_subdays_neg(self):
        # 测试处理负数秒数单位的时间间隔格式化
        y = pd.to_timedelta(list(range(5)) + [NaT], unit="s")._values
        result = fmt._Timedelta64Formatter(-y).get_result()
        assert result[0].strip() == "0 days 00:00:00"
        assert result[1].strip() == "-1 days +23:59:59"

    def test_zero(self):
        # 测试处理零时间间隔格式化
        x = pd.to_timedelta(list(range(1)) + [NaT], unit="D")._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == "0 days"

        x = pd.to_timedelta(list(range(1)), unit="D")._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == "0 days"


class TestDatetime64Formatter:
    # 测试类，用于测试日期时间格式化功能

    def test_mixed(self):
        # 测试混合日期时间格式化
        x = Series([datetime(2013, 1, 1), datetime(2013, 1, 1, 12), NaT])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == "2013-01-01 00:00:00"
        assert result[1].strip() == "2013-01-01 12:00:00"

    def test_dates(self):
        # 测试日期格式化
        x = Series([datetime(2013, 1, 1), datetime(2013, 1, 2), NaT])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == "2013-01-01"
        assert result[1].strip() == "2013-01-02"

    def test_date_nanos(self):
        # 测试带纳秒的日期时间格式化
        x = Series([Timestamp(200)])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == "1970-01-01 00:00:00.000000200"
    def test_dates_display(self):
        # 测试日期显示函数

        # 创建包含日期范围的Series对象，频率为每天一次，从"20130101 09:00:00"开始
        x = Series(date_range("20130101 09:00:00", periods=5, freq="D"))
        # 将第二个日期置为NaN
        x.iloc[1] = np.nan
        # 使用_Datetime64Formatter类对日期值进行格式化处理，并获取结果
        result = fmt._Datetime64Formatter(x._values).get_result()
        # 断言第一个结果的字符串应为"2013-01-01 09:00:00"
        assert result[0].strip() == "2013-01-01 09:00:00"
        # 断言第二个结果的字符串应为"NaT" (Not a Time)
        assert result[1].strip() == "NaT"
        # 断言第五个结果的字符串应为"2013-01-05 09:00:00"
        assert result[4].strip() == "2013-01-05 09:00:00"

        # 创建包含日期范围的Series对象，频率为每秒一次，从"20130101 09:00:00"开始
        x = Series(date_range("20130101 09:00:00", periods=5, freq="s"))
        # 将第二个日期置为NaN
        x.iloc[1] = np.nan
        # 使用_Datetime64Formatter类对日期值进行格式化处理，并获取结果
        result = fmt._Datetime64Formatter(x._values).get_result()
        # 断言第一个结果的字符串应为"2013-01-01 09:00:00"
        assert result[0].strip() == "2013-01-01 09:00:00"
        # 断言第二个结果的字符串应为"NaT"
        assert result[1].strip() == "NaT"
        # 断言第五个结果的字符串应为"2013-01-01 09:00:04"
        assert result[4].strip() == "2013-01-01 09:00:04"

        # 创建包含日期范围的Series对象，频率为每毫秒一次，从"20130101 09:00:00"开始
        x = Series(date_range("20130101 09:00:00", periods=5, freq="ms"))
        # 将第二个日期置为NaN
        x.iloc[1] = np.nan
        # 使用_Datetime64Formatter类对日期值进行格式化处理，并获取结果
        result = fmt._Datetime64Formatter(x._values).get_result()
        # 断言第一个结果的字符串应为"2013-01-01 09:00:00.000"
        assert result[0].strip() == "2013-01-01 09:00:00.000"
        # 断言第二个结果的字符串应为"NaT"
        assert result[1].strip() == "NaT"
        # 断言第五个结果的字符串应为"2013-01-01 09:00:00.004"
        assert result[4].strip() == "2013-01-01 09:00:00.004"

        # 创建包含日期范围的Series对象，频率为每微秒一次，从"20130101 09:00:00"开始
        x = Series(date_range("20130101 09:00:00", periods=5, freq="us"))
        # 将第二个日期置为NaN
        x.iloc[1] = np.nan
        # 使用_Datetime64Formatter类对日期值进行格式化处理，并获取结果
        result = fmt._Datetime64Formatter(x._values).get_result()
        # 断言第一个结果的字符串应为"2013-01-01 09:00:00.000000"
        assert result[0].strip() == "2013-01-01 09:00:00.000000"
        # 断言第二个结果的字符串应为"NaT"
        assert result[1].strip() == "NaT"
        # 断言第五个结果的字符串应为"2013-01-01 09:00:00.000004"
        assert result[4].strip() == "2013-01-01 09:00:00.000004"

        # 创建包含日期范围的Series对象，频率为每纳秒一次，从"20130101 09:00:00"开始
        x = Series(date_range("20130101 09:00:00", periods=5, freq="ns"))
        # 将第二个日期置为NaN
        x.iloc[1] = np.nan
        # 使用_Datetime64Formatter类对日期值进行格式化处理，并获取结果
        result = fmt._Datetime64Formatter(x._values).get_result()
        # 断言第一个结果的字符串应为"2013-01-01 09:00:00.000000000"
        assert result[0].strip() == "2013-01-01 09:00:00.000000000"
        # 断言第二个结果的字符串应为"NaT"
        assert result[1].strip() == "NaT"
        # 断言第五个结果的字符串应为"2013-01-01 09:00:00.000000004"
        assert result[4].strip() == "2013-01-01 09:00:00.000000004"

    def test_datetime64formatter_yearmonth(self):
        # 测试年月格式化函数

        # 创建包含两个日期对象的Series对象
        x = Series([datetime(2016, 1, 1), datetime(2016, 2, 2)])._values

        # 定义一个格式化函数，将日期对象转换为"%Y-%m"格式的字符串
        def format_func(x):
            return x.strftime("%Y-%m")

        # 使用_Datetime64Formatter类对日期值进行格式化处理，并获取结果，使用自定义的格式化函数
        formatter = fmt._Datetime64Formatter(x, formatter=format_func)
        # 断言格式化后的结果应为["2016-01", "2016-02"]
        result = formatter.get_result()
        assert result == ["2016-01", "2016-02"]

    def test_datetime64formatter_hoursecond(self):
        # 测试时分秒格式化函数

        # 创建包含两个时间对象的Series对象，使用"%H:%M:%S.%f"格式
        x = Series(
            pd.to_datetime(["10:10:10.100", "12:12:12.120"], format="%H:%M:%S.%f")
        )._values

        # 定义一个格式化函数，将时间对象转换为"%H:%M"格式的字符串
        def format_func(x):
            return x.strftime("%H:%M")

        # 使用_Datetime64Formatter类对时间值进行格式化处理，并获取结果，使用自定义的格式化函数
        formatter = fmt._Datetime64Formatter(x, formatter=format_func)
        # 断言格式化后的结果应为["10:10", "12:12"]
        result = formatter.get_result()
        assert result == ["10:10", "12:12"]

    def test_datetime64formatter_tz_ms(self):
        # 测试带时区毫秒格式化函数

        # 创建包含三个日期对象的Series对象，数据类型为datetime64[ms]，并将时区设置为"US/Pacific"
        x = (
            Series(
                np.array(["2999-01-01", "2999-01-02", "NaT"], dtype="datetime64[ms]")
            )
            .dt.tz_localize("US/Pacific")
            ._values
        )

        # 使用_Datetime64TZFormatter类对带时区的日期值进行格式化处理，并获取结果
        result = fmt._Datetime64TZFormatter(x).get_result()
        # 断言格式化后的第一个结果应为"2999-01-01 00:00:00-08:00"
        assert result[0].strip() == "2999-01-01 00:00:00-08:00"
        # 断言格式化后的第二个结果应为"2999-01-02 00:00:00-08:00"
        assert result[1].strip() == "2999-01-02 00:00:00-08:00"
class TestFormatPercentiles:
    # 参数化测试用例，验证百分位数格式化函数的输出是否符合预期
    @pytest.mark.parametrize(
        "percentiles, expected",
        [
            (
                [0.01999, 0.02001, 0.5, 0.666666, 0.9999],
                ["1.999%", "2.001%", "50%", "66.667%", "99.99%"],
            ),
            (
                [0, 0.5, 0.02001, 0.5, 0.666666, 0.9999],
                ["0%", "50%", "2.0%", "50%", "66.67%", "99.99%"],
            ),
            ([0.281, 0.29, 0.57, 0.58], ["28.1%", "29%", "57%", "58%"]),
            ([0.28, 0.29, 0.57, 0.58], ["28%", "29%", "57%", "58%"]),
            (
                [0.9, 0.99, 0.999, 0.9999, 0.99999],
                ["90%", "99%", "99.9%", "99.99%", "99.999%"],
            ),
        ],
    )
    def test_format_percentiles(self, percentiles, expected):
        # 调用被测试的百分位数格式化函数并断言结果是否符合预期
        result = fmt.format_percentiles(percentiles)
        assert result == expected

    # 参数化测试用例，验证当输入百分位数不在 [0,1] 范围内时，是否会引发 ValueError 异常
    @pytest.mark.parametrize(
        "percentiles",
        [
            ([0.1, np.nan, 0.5]),
            ([-0.001, 0.1, 0.5]),
            ([2, 0.1, 0.5]),
            ([0.1, 0.5, "a"]),
        ],
    )
    def test_error_format_percentiles(self, percentiles):
        # 验证调用百分位数格式化函数时是否会抛出特定的 ValueError 异常
        msg = r"percentiles should all be in the interval \[0,1\]"
        with pytest.raises(ValueError, match=msg):
            fmt.format_percentiles(percentiles)

    # 测试整数索引下的百分位数格式化，验证是否能正确处理整数索引情况
    def test_format_percentiles_integer_idx(self):
        # Issue #26660
        # 调用百分位数格式化函数并断言结果是否符合预期
        result = fmt.format_percentiles(np.linspace(0, 1, 10 + 1))
        expected = [
            "0%",
            "10%",
            "20%",
            "30%",
            "40%",
            "50%",
            "60%",
            "70%",
            "80%",
            "90%",
            "100%",
        ]
        assert result == expected


@pytest.mark.parametrize("method", ["to_string", "to_html", "to_latex"])
@pytest.mark.parametrize(
    "encoding, data",
    [(None, "abc"), ("utf-8", "abc"), ("gbk", "造成输出中文显示乱码"), ("foo", "abc")],
)
@pytest.mark.parametrize("filepath_or_buffer_id", ["string", "pathlike", "buffer"])
def test_filepath_or_buffer_arg(
    method,
    tmp_path,
    encoding,
    data,
    filepath_or_buffer_id,
):
    # 根据不同的 filepath_or_buffer_id 类型选择不同的文件路径或缓冲区
    if filepath_or_buffer_id == "buffer":
        filepath_or_buffer = StringIO()
    elif filepath_or_buffer_id == "pathlike":
        filepath_or_buffer = tmp_path / "foo"
    else:
        filepath_or_buffer = str(tmp_path / "foo")

    # 创建包含单行数据的 DataFrame 对象
    df = DataFrame([data])

    # 如果方法为 "to_latex"，则需要导入 jinja2 模块
    if method in ["to_latex"]:  # uses styler implementation
        pytest.importorskip("jinja2")

    # 如果 filepath_or_buffer_id 不是 "string" 或 "pathlike"，且指定了编码，则应抛出 ValueError 异常
    if filepath_or_buffer_id not in ["string", "pathlike"] and encoding is not None:
        with pytest.raises(
            ValueError, match="buf is not a file name and encoding is specified."
        ):
            getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
    # 如果编码为 "foo"，则应抛出 LookupError 异常
    elif encoding == "foo":
        with pytest.raises(LookupError, match="unknown encoding"):
            getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
    else:
        # 使用 getattr 函数获取 DataFrame 对象 df 的方法 method 的返回值，作为预期结果
        expected = getattr(df, method)()
        # 使用 getattr 函数调用 DataFrame 对象 df 的方法 method，并传入文件路径或缓冲区以及编码参数
        getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
        # 如果未指定编码，将编码设置为 "utf-8"
        encoding = encoding or "utf-8"
        # 如果 filepath_or_buffer_id 为 "string"
        if filepath_or_buffer_id == "string":
            # 使用指定编码打开文件，并读取其内容
            with open(filepath_or_buffer, encoding=encoding) as f:
                result = f.read()
        # 如果 filepath_or_buffer_id 为 "pathlike"
        elif filepath_or_buffer_id == "pathlike":
            # 使用指定编码读取 filepath_or_buffer 对象的文本内容
            result = filepath_or_buffer.read_text(encoding=encoding)
        # 如果 filepath_or_buffer_id 为 "buffer"
        elif filepath_or_buffer_id == "buffer":
            # 获取 filepath_or_buffer 缓冲区的所有值
            result = filepath_or_buffer.getvalue()
        # 断言结果与预期值相等
        assert result == expected
    # 如果 filepath_or_buffer_id 为 "buffer"，断言缓冲区未关闭
    if filepath_or_buffer_id == "buffer":
        assert not filepath_or_buffer.closed
# 使用 pytest 模块的 parametrize 装饰器，指定测试函数的参数化
@pytest.mark.parametrize("method", ["to_string", "to_html", "to_latex"])
# 定义测试函数，检查给定方法在使用错误参数时是否引发异常
def test_filepath_or_buffer_bad_arg_raises(float_frame, method):
    # 如果方法是 'to_latex'，则需要引入 jinja2 库
    if method in ["to_latex"]:  # uses styler implementation
        pytest.importorskip("jinja2")
    # 错误消息，指示缓冲区既不是文件名也没有写入方法
    msg = "buf is not a file name and it has no write method"
    # 使用 pytest.raises 检查是否引发 TypeError 异常，并匹配特定的错误消息
    with pytest.raises(TypeError, match=msg):
        # 调用 float_frame 对象的指定方法（根据 parametrize 提供的参数），传入错误的缓冲区对象
        getattr(float_frame, method)(buf=object())
```