# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_decimal.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入所需的模块和库
from io import StringIO  # 导入 StringIO 类，用于在内存中操作字符串数据
import pytest  # 导入 pytest 测试框架

from pandas import DataFrame  # 导入 DataFrame 类，用于处理和操作数据框
import pandas._testing as tm  # 导入 pandas 内部测试工具模块

# 忽略特定警告，以便测试能正常执行
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


# 参数化测试函数，测试千位分隔和小数点不同格式的 CSV 数据解析
@pytest.mark.parametrize(
    "data,thousands,decimal",
    [
        (
            """A|B|C
1|2,334.01|5
10|13|10.
""",
            ",",
            ".",
        ),
        (
            """A|B|C
1|2.334,01|5
10|13|10,
""",
            ".",
            ",",
        ),
    ],
)
def test_1000_sep_with_decimal(all_parsers, data, thousands, decimal):
    # 获取解析器对象
    parser = all_parsers
    # 预期的 DataFrame 结果
    expected = DataFrame({"A": [1, 10], "B": [2334.01, 13], "C": [5, 10.0]})

    # 当解析器引擎为 'pyarrow' 时，验证是否会抛出预期的 ValueError 异常
    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), sep="|", thousands=thousands, decimal=decimal
            )
        return

    # 使用解析器读取 CSV 数据并比较结果是否与预期相符
    result = parser.read_csv(
        StringIO(data), sep="|", thousands=thousands, decimal=decimal
    )
    tm.assert_frame_equal(result, expected)


# 测试欧洲小数格式的 CSV 数据解析
def test_euro_decimal_format(all_parsers):
    # 获取解析器对象
    parser = all_parsers
    # 欧洲小数格式的数据
    data = """Id;Number1;Number2;Text1;Text2;Number3
1;1521,1541;187101,9543;ABC;poi;4,738797819
2;121,12;14897,76;DEF;uyt;0,377320872
3;878,158;108013,434;GHI;rez;2,735694704"""

    # 预期的 DataFrame 结果
    expected = DataFrame(
        [
            [1, 1521.1541, 187101.9543, "ABC", "poi", 4.738797819],
            [2, 121.12, 14897.76, "DEF", "uyt", 0.377320872],
            [3, 878.158, 108013.434, "GHI", "rez", 2.735694704],
        ],
        columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
    )

    # 使用解析器读取 CSV 数据并比较结果是否与预期相符
    result = parser.read_csv(StringIO(data), sep=";", decimal=",")
    tm.assert_frame_equal(result, expected)
```