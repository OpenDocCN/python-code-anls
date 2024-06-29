# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_comment.py`

```
"""
Tests that comments are properly handled during parsing
for all of the parsers defined in parsers.py
"""

# 从 io 模块中导入 StringIO 类
from io import StringIO

# 导入 numpy 和 pytest 库
import numpy as np
import pytest

# 从 pandas 库中导入 DataFrame 类，并导入 pandas._testing 库作为 tm 别名
from pandas import DataFrame
import pandas._testing as tm

# 使用 pytest 的 parametrize 装饰器，定义测试函数 test_comment，测试读取 CSV 文件时的评论处理
@pytest.mark.parametrize("na_values", [None, ["NaN"]])
def test_comment(all_parsers, na_values):
    # 获取 all_parsers 夹具
    parser = all_parsers
    # 定义测试数据字符串
    data = """A,B,C
1,2.,4.#hello world
5.,NaN,10.0
"""
    # 定义期望的 DataFrame 结果
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    # 如果使用的解析引擎是 'pyarrow'
    if parser.engine == "pyarrow":
        # 抛出 ValueError 异常，提示不支持 'pyarrow' 引擎中的 'comment' 选项
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", na_values=na_values)
        return
    # 使用 parser 对象读取 CSV 数据，并进行比较
    result = parser.read_csv(StringIO(data), comment="#", na_values=na_values)
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，定义测试函数 test_line_comment，测试读取 CSV 文件时的行评论处理
@pytest.mark.parametrize("read_kwargs", [{}, {"lineterminator": "*"}, {"sep": r"\s+"}])
def test_line_comment(all_parsers, read_kwargs):
    # 获取 all_parsers 夹具
    parser = all_parsers
    # 定义测试数据字符串
    data = """# empty
A,B,C
1,2.,4.#hello world
#ignore this line
5.,NaN,10.0
"""
    # 根据 read_kwargs 中的选项修改 data 字符串格式
    if read_kwargs.get("sep"):
        data = data.replace(",", " ")
    elif read_kwargs.get("lineterminator"):
        data = data.replace("\n", read_kwargs.get("lineterminator"))

    # 将 "comment" 选项设置为 "#"
    read_kwargs["comment"] = "#"

    # 如果使用的解析引擎是 'pyarrow'
    if parser.engine == "pyarrow":
        # 根据不同的情况抛出 ValueError 异常
        if "lineterminator" in read_kwargs:
            msg = "The 'lineterminator' option is not supported with the 'pyarrow' engine"
        else:
            msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **read_kwargs)
        return
    # 如果解析引擎是 'python' 并且 read_kwargs 包含 "lineterminator" 选项
    elif parser.engine == "python" and read_kwargs.get("lineterminator"):
        # 抛出 ValueError 异常，提示暂不支持自定义行终止符
        msg = r"Custom line terminators not supported in python parser \(yet\)"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **read_kwargs)
        return

    # 使用 parser 对象读取 CSV 数据，并进行比较
    result = parser.read_csv(StringIO(data), **read_kwargs)
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_comment_skiprows，测试读取 CSV 文件时的跳过行处理
def test_comment_skiprows(all_parsers):
    # 获取 all_parsers 夹具
    parser = all_parsers
    # 定义测试数据字符串
    data = """# empty
random line
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # 应该忽略前四行（包括注释行）
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    # 如果使用的解析引擎是 'pyarrow'
    if parser.engine == "pyarrow":
        # 抛出 ValueError 异常，提示不支持 'pyarrow' 引擎中的 'comment' 选项
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", skiprows=4)
        return

    # 使用 parser 对象读取 CSV 数据，并进行比较
    result = parser.read_csv(StringIO(data), comment="#", skiprows=4)
    tm.assert_frame_equal(result, expected)
def test_comment_header(all_parsers):
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 设置测试数据，包含带有注释的CSV格式数据
    data = """# empty
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # 预期结果为一个DataFrame对象，包含特定的数据和列名
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    # 如果解析器引擎为"pyarrow"，则抛出特定的异常
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            # 通过解析器读取CSV数据，验证异常是否符合预期
            parser.read_csv(StringIO(data), comment="#", header=1)
        return
    # 使用解析器读取CSV数据，跳过第一行注释，从第二行开始作为表头
    result = parser.read_csv(StringIO(data), comment="#", header=1)
    # 验证读取结果与预期是否相符
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows_header(all_parsers):
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 设置测试数据，包含带有注释和空行的CSV格式数据
    data = """# empty
# second empty line
# third empty line
X,Y,Z
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # 预期结果为一个DataFrame对象，包含特定的数据和列名
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    # 如果解析器引擎为"pyarrow"，则抛出特定的异常
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            # 通过解析器读取CSV数据，验证异常是否符合预期
            parser.read_csv(StringIO(data), comment="#", skiprows=4, header=1)
        return
    # 使用解析器读取CSV数据，跳过前四行（包括注释），从第五行开始作为表头
    result = parser.read_csv(StringIO(data), comment="#", skiprows=4, header=1)
    # 验证读取结果与预期是否相符
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("comment_char", ["#", "~", "&", "^", "*", "@"])
def test_custom_comment_char(all_parsers, comment_char):
    # 从参数中获取所有解析器对象和自定义的注释字符
    parser = all_parsers
    # 设置包含自定义注释字符的测试数据
    data = "a,b,c\n1,2,3#ignore this!\n4,5,6#ignorethistoo"

    # 如果解析器引擎为"pyarrow"，则抛出特定的异常
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            # 通过解析器读取CSV数据，验证异常是否符合预期
            parser.read_csv(
                StringIO(data.replace("#", comment_char)), comment=comment_char
            )
        return
    # 使用解析器读取CSV数据，根据自定义的注释字符过滤注释行
    result = parser.read_csv(
        StringIO(data.replace("#", comment_char)), comment=comment_char
    )

    # 预期结果为一个DataFrame对象，包含特定的数据和列名
    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    # 验证读取结果与预期是否相符
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("header", ["infer", None])
def test_comment_first_line(all_parsers, header):
    # see gh-4623
    # 从参数中获取所有解析器对象和表头参数
    parser = all_parsers
    # 设置带有注释的测试数据
    data = "# notes\na,b,c\n# more notes\n1,2,3"

    # 根据不同的表头参数，设置预期的DataFrame对象
    if header is None:
        expected = DataFrame({0: ["a", "1"], 1: ["b", "2"], 2: ["c", "3"]})
    else:
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    # 如果解析器引擎为"pyarrow"，则抛出特定的异常
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            # 通过解析器读取CSV数据，验证异常是否符合预期
            parser.read_csv(StringIO(data), comment="#", header=header)
        return
    # 使用解析器读取CSV数据，根据注释过滤注释行，并根据表头参数处理数据
    result = parser.read_csv(StringIO(data), comment="#", header=header)
    # 验证读取结果与预期是否相符
    tm.assert_frame_equal(result, expected)
def test_comment_char_in_default_value(all_parsers, request):
    # GH#34002
```