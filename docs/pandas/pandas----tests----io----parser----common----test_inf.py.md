# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_inf.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入所需的库和模块
from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm

# 设置 pytest 标记，忽略特定的警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 使用 xfail_pyarrow 标记，表示在 pyarrow 环境下预期会失败
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")

# 在 xfail_pyarrow 标记下，使用参数化测试来测试不同的 na_filter 参数
@xfail_pyarrow  # AssertionError: DataFrame.index are different
@pytest.mark.parametrize("na_filter", [True, False])
def test_inf_parsing(all_parsers, na_filter):
    # 从 all_parsers 获取解析器
    parser = all_parsers
    # 定义测试数据
    data = """\
,A
a,inf
b,-inf
c,+Inf
d,-Inf
e,INF
f,-INF
g,+INf
h,-INf
i,inF
j,-inF"""
    # 期望的 DataFrame 结果
    expected = DataFrame(
        {"A": [float("inf"), float("-inf")] * 5},
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )
    # 使用解析器读取 CSV 数据并进行断言比较
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)


# 同样在 xfail_pyarrow 标记下，使用参数化测试来测试不同的 na_filter 参数
@xfail_pyarrow  # AssertionError: DataFrame.index are different
@pytest.mark.parametrize("na_filter", [True, False])
def test_infinity_parsing(all_parsers, na_filter):
    # 从 all_parsers 获取解析器
    parser = all_parsers
    # 定义测试数据
    data = """\
,A
a,Infinity
b,-Infinity
c,+Infinity
"""
    # 期望的 DataFrame 结果
    expected = DataFrame(
        {"A": [float("infinity"), float("-infinity"), float("+infinity")]},
        index=["a", "b", "c"],
    )
    # 使用解析器读取 CSV 数据并进行断言比较
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)


这段代码是一组用于测试 CSV 解析功能的单元测试。它使用了 `pytest` 框架来管理测试，通过参数化和标记来组织不同的测试情况，并使用 `pandas._testing` 中的工具来比较预期结果和实际结果的 DataFrame 是否相等。
```