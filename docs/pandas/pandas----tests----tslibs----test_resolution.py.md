# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_resolution.py`

```
# 导入 datetime 模块，用于处理日期时间相关操作
import datetime

# 导入 numpy 库，并将其命名为 np，用于科学计算和数组操作
import numpy as np

# 导入 pytest 库，用于编写和执行测试用例
import pytest

# 从 pandas._libs.tslibs 中导入 Resolution 和 get_resolution 函数
from pandas._libs.tslibs import (
    Resolution,
    get_resolution,
)

# 从 pandas._libs.tslibs.dtypes 中导入 NpyDatetimeUnit，用于处理 NumPy datetime 单位
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit

# 导入 pandas._testing 模块，并将其命名为 tm，用于 pandas 测试辅助功能
import pandas._testing as tm


# 定义测试函数 test_get_resolution_nano，测试获取纳秒分辨率的情况
def test_get_resolution_nano():
    # 创建一个包含单个整数的 NumPy 数组，数据类型为 int64
    arr = np.array([1], dtype=np.int64)
    # 调用 get_resolution 函数，获取数组的时间分辨率
    res = get_resolution(arr)
    # 断言获取的分辨率为纳秒 RESO_NS
    assert res == Resolution.RESO_NS


# 定义测试函数 test_get_resolution_non_nano_data，测试非纳秒分辨率数据的情况
def test_get_resolution_non_nano_data():
    # 创建一个包含单个整数的 NumPy 数组，数据类型为 int64
    arr = np.array([1], dtype=np.int64)
    # 调用 get_resolution 函数，传入自定义的时间单位 NPY_FR_us（微秒），获取时间分辨率
    res = get_resolution(arr, None, NpyDatetimeUnit.NPY_FR_us.value)
    # 断言获取的分辨率为微秒 RESO_US
    assert res == Resolution.RESO_US

    # 调用 get_resolution 函数，传入时区信息和时间单位 NPY_FR_us（微秒），获取时间分辨率
    res = get_resolution(arr, datetime.timezone.utc, NpyDatetimeUnit.NPY_FR_us.value)
    # 断言获取的分辨率为微秒 RESO_US
    assert res == Resolution.RESO_US


# 定义参数化测试函数 test_get_attrname_from_abbrev，测试根据频率字符串获取属性名的情况
@pytest.mark.parametrize(
    "freqstr,expected",
    [
        ("Y", "year"),
        ("Q", "quarter"),
        ("M", "month"),
        ("D", "day"),
        ("h", "hour"),
        ("min", "minute"),
        ("s", "second"),
        ("ms", "millisecond"),
        ("us", "microsecond"),
        ("ns", "nanosecond"),
    ],
)
def test_get_attrname_from_abbrev(freqstr, expected):
    # 调用 Resolution 类的 get_reso_from_freqstr 方法，根据频率字符串获取分辨率对象 reso
    reso = Resolution.get_reso_from_freqstr(freqstr)
    # 断言分辨率对象的属性缩写为频率字符串 freqstr
    assert reso.attr_abbrev == freqstr
    # 断言分辨率对象的属性名为预期的全名 expected
    assert reso.attrname == expected


# 定义参数化测试函数 test_units_H_S_deprecated_from_attrname_to_abbrevs，测试从属性名到缩写的转换，并检查是否弃用
@pytest.mark.parametrize("freq", ["H", "S"])
def test_units_H_S_deprecated_from_attrname_to_abbrevs(freq):
    # 构造将要移除的警告消息
    msg = f"'{freq}' is deprecated and will be removed in a future version."
    
    # 使用 pytest 的 assert_produces_warning 上下文管理器，检查是否生成 FutureWarning 警告，并匹配消息内容
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # 调用 Resolution 类的 get_reso_from_freqstr 方法，根据频率字符串 freq 获取分辨率对象
        Resolution.get_reso_from_freqstr(freq)
```