# `D:\src\scipysrc\pandas\pandas\tests\io\parser\usecols\test_strings.py`

```
"""
Tests the usecols functionality during parsing
for all of the parsers defined in parsers.py
"""

from io import StringIO  # 导入 StringIO 类来创建内存中的文件对象

import pytest  # 导入 pytest 测试框架

from pandas import DataFrame  # 导入 DataFrame 类用于数据框操作
import pandas._testing as tm  # 导入 pandas 内部测试工具模块

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)  # 设置 pytest 的标记，忽略特定的警告信息


def test_usecols_with_unicode_strings(all_parsers):
    # 测试用例：使用 Unicode 字符串作为 usecols 参数
    # 参见 GitHub 问题 gh-13219
    data = """AAA,BBB,CCC,DDD
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers  # 从参数中获取数据解析器

    exp_data = {
        "AAA": {
            0: 0.056674972999999997,
            1: 2.6132309819999997,
            2: 3.5689350380000002,
        },
        "BBB": {0: 8, 1: 2, 2: 7},
    }
    expected = DataFrame(exp_data)  # 创建期望的 DataFrame 对象

    result = parser.read_csv(StringIO(data), usecols=["AAA", "BBB"])  # 解析 CSV 数据，仅选择指定列
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果和期望的 DataFrame


def test_usecols_with_single_byte_unicode_strings(all_parsers):
    # 测试用例：使用单字节 Unicode 字符串作为 usecols 参数
    # 参见 GitHub 问题 gh-13219
    data = """A,B,C,D
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers  # 从参数中获取数据解析器

    exp_data = {
        "A": {
            0: 0.056674972999999997,
            1: 2.6132309819999997,
            2: 3.5689350380000002,
        },
        "B": {0: 8, 1: 2, 2: 7},
    }
    expected = DataFrame(exp_data)  # 创建期望的 DataFrame 对象

    result = parser.read_csv(StringIO(data), usecols=["A", "B"])  # 解析 CSV 数据，仅选择指定列
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果和期望的 DataFrame


@pytest.mark.parametrize("usecols", [["AAA", b"BBB"], [b"AAA", "BBB"]])
def test_usecols_with_mixed_encoding_strings(all_parsers, usecols):
    # 测试用例：使用混合编码字符串作为 usecols 参数
    data = """AAA,BBB,CCC,DDD
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers  # 从参数中获取数据解析器
    _msg_validate_usecols_arg = (
        "'usecols' must either be list-like "
        "of all strings, all unicode, all "
        "integers or a callable."
    )
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols=usecols)  # 期望抛出 ValueError 异常，捕获错误信息验证参数类型


def test_usecols_with_multi_byte_characters(all_parsers):
    # 测试用例：使用多字节字符作为 usecols 参数
    data = """あああ,いい,ううう,ええええ
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers  # 从参数中获取数据解析器

    exp_data = {
        "あああ": {
            0: 0.056674972999999997,
            1: 2.6132309819999997,
            2: 3.5689350380000002,
        },
        "いい": {0: 8, 1: 2, 2: 7},
    }
    expected = DataFrame(exp_data)  # 创建期望的 DataFrame 对象

    result = parser.read_csv(StringIO(data), usecols=["あああ", "いい"])  # 解析 CSV 数据，仅选择指定列
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果和期望的 DataFrame
```