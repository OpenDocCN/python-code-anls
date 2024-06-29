# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_float.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入需要的模块和库
from io import StringIO  # 导入字符串IO模块中的StringIO类

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas.compat import is_platform_linux  # 导入pandas兼容模块中的is_platform_linux函数

from pandas import DataFrame  # 从pandas库中导入DataFrame类，用于处理数据表
import pandas._testing as tm  # 导入pandas内部测试模块，用于测试框架相关的工具函数和类

# 忽略特定警告信息的pytest标记
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")  # 使用pyarrow_xfail标记，用于测试时预期的失败情况
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")  # 使用pyarrow_skip标记，用于跳过与pyarrow相关的测试


@skip_pyarrow  # 使用skip_pyarrow标记，跳过与pyarrow相关的测试
def test_float_parser(all_parsers):
    # 以gh-9565为例
    parser = all_parsers  # 设置parser变量为all_parsers的值
    data = "45e-1,4.5,45.,inf,-inf"  # 定义包含特定浮点数的数据字符串
    result = parser.read_csv(StringIO(data), header=None)  # 使用parser解析数据字符串，生成DataFrame对象

    expected = DataFrame([[float(s) for s in data.split(",")]])  # 生成预期的DataFrame对象，包含数据字符串中的浮点数
    tm.assert_frame_equal(result, expected)  # 断言result与expected的DataFrame对象是否相等


def test_scientific_no_exponent(all_parsers_all_precisions):
    # 以gh-12215为例
    df = DataFrame.from_dict({"w": ["2e"], "x": ["3E"], "y": ["42e"], "z": ["632E"]})  # 从字典创建DataFrame对象
    data = df.to_csv(index=False)  # 将DataFrame对象转换为CSV格式的数据字符串
    parser, precision = all_parsers_all_precisions  # 获取all_parsers_all_precisions中的parser和precision

    df_roundtrip = parser.read_csv(StringIO(data), float_precision=precision)  # 使用parser解析CSV数据，指定浮点数精度
    tm.assert_frame_equal(df_roundtrip, df)  # 断言解析后的DataFrame与原始DataFrame对象是否相等


@pytest.mark.parametrize(
    "neg_exp",
    [
        -617,
        -100000,
        pytest.param(-99999999999999999, marks=pytest.mark.skip_ubsan),
    ],
)
def test_very_negative_exponent(all_parsers_all_precisions, neg_exp):
    # GH#38753
    parser, precision = all_parsers_all_precisions  # 获取all_parsers_all_precisions中的parser和precision

    data = f"data\n10E{neg_exp}"  # 根据指定的负指数创建数据字符串
    result = parser.read_csv(StringIO(data), float_precision=precision)  # 使用parser解析数据字符串，指定浮点数精度
    expected = DataFrame({"data": [0.0]})  # 创建预期的DataFrame对象，包含零值
    tm.assert_frame_equal(result, expected)  # 断言解析后的DataFrame与预期DataFrame对象是否相等


@pytest.mark.skip_ubsan  # 使用skip_ubsan标记，跳过与ubsan相关的测试
@xfail_pyarrow  # 使用xfail_pyarrow标记，预期与pyarrow相关的测试失败
@pytest.mark.parametrize("exp", [999999999999999999, -999999999999999999])
def test_too_many_exponent_digits(all_parsers_all_precisions, exp, request):
    # GH#38753
    parser, precision = all_parsers_all_precisions  # 获取all_parsers_all_precisions中的parser和precision
    data = f"data\n10E{exp}"  # 根据指定的指数创建数据字符串
    result = parser.read_csv(StringIO(data), float_precision=precision)  # 使用parser解析数据字符串，指定浮点数精度
    if precision == "round_trip":  # 如果精度为"round_trip"
        if exp == 999999999999999999 and is_platform_linux():  # 如果指数为999999999999999999且当前系统为Linux
            mark = pytest.mark.xfail(reason="GH38794, on Linux gives object result")  # 添加xfail标记，说明在Linux下的特定情况
            request.applymarker(mark)  # 应用标记到当前测试用例

        value = np.inf if exp > 0 else 0.0  # 根据指数确定期望值，大于0时为无穷大，否则为0
        expected = DataFrame({"data": [value]})  # 创建预期的DataFrame对象，包含计算后的值
    else:
        expected = DataFrame({"data": [f"10E{exp}"]})  # 创建预期的DataFrame对象，包含原始数据字符串

    tm.assert_frame_equal(result, expected)  # 断言解析后的DataFrame与预期DataFrame对象是否相等
```