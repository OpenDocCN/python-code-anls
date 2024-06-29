# `D:\src\scipysrc\pandas\pandas\tests\io\parser\usecols\test_parse_dates.py`

```
"""
Tests the usecols functionality during parsing
for all of the parsers defined in parsers.py
"""

# 引入必要的模块和库
from io import StringIO  # 从 io 模块中导入 StringIO 类
import pytest  # 导入 pytest 测试框架
from pandas import (  # 从 pandas 库中导入以下对象
    DataFrame,  # 数据框对象
    Index,  # 索引对象
    Timestamp,  # 时间戳对象
)
import pandas._testing as tm  # 导入 pandas 内部的测试工具

# 忽略特定警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")  # 标记用例对于 pyarrow 的失败预期
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")  # 标记用例对于 pyarrow 的跳过

_msg_pyarrow_requires_names = (
    "The pyarrow engine does not allow 'usecols' to be integer column "
    "positions. Pass a list of string column names instead."
)

@skip_pyarrow  # 标记当前测试用例为 pyarrow 跳过
def test_usecols_with_parse_dates2(all_parsers):
    # 测试用例：验证在解析过程中 usecols 功能的使用（见 gh-13604）

    parser = all_parsers  # 使用给定的解析器对象
    data = """2008-02-07 09:40,1032.43
2008-02-07 09:50,1042.54
2008-02-07 10:00,1051.65"""

    names = ["date", "values"]  # 指定列名列表
    usecols = names[:]  # 使用所有列名作为 usecols
    parse_dates = [0]  # 将第一列解析为日期类型

    index = Index(
        [
            Timestamp("2008-02-07 09:40"),
            Timestamp("2008-02-07 09:50"),
            Timestamp("2008-02-07 10:00"),
        ],
        name="date",
    )  # 创建日期索引对象
    cols = {"values": [1032.43, 1042.54, 1051.65]}  # 列数据字典
    expected = DataFrame(cols, index=index)  # 期望的数据框对象

    # 执行 CSV 解析，并验证结果是否与期望一致
    result = parser.read_csv(
        StringIO(data),
        parse_dates=parse_dates,
        index_col=0,
        usecols=usecols,
        header=None,
        names=names,
    )
    tm.assert_frame_equal(result, expected)  # 断言结果与期望一致

def test_usecols_with_parse_dates3(all_parsers):
    # 测试用例：验证在解析过程中 usecols 功能的使用（见 gh-14792）

    parser = all_parsers  # 使用给定的解析器对象
    data = """a,b,c,d,e,f,g,h,i,j
2016/09/21,1,1,2,3,4,5,6,7,8"""

    usecols = list("abcdefghij")  # 使用所有字母作为列名列表
    parse_dates = [0]  # 将第一列解析为日期类型

    cols = {
        "a": Timestamp("2016-09-21"),
        "b": [1],
        "c": [1],
        "d": [2],
        "e": [3],
        "f": [4],
        "g": [5],
        "h": [6],
        "i": [7],
        "j": [8],
    }  # 列数据字典
    expected = DataFrame(cols, columns=usecols)  # 期望的数据框对象

    # 执行 CSV 解析，并验证结果是否与期望一致
    result = parser.read_csv(StringIO(data), usecols=usecols, parse_dates=parse_dates)
    tm.assert_frame_equal(result, expected)  # 断言结果与期望一致
```