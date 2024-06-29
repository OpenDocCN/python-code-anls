# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_api.py`

```
"""Tests that the tslibs API is locked down"""

# 从 pandas._libs 导入 tslibs 模块
from pandas._libs import tslibs

# 定义一个测试函数，验证 tslibs 模块的命名空间是否符合预期
def test_namespace():
    # 定义子模块列表
    submodules = [
        "base",
        "ccalendar",
        "conversion",
        "dtypes",
        "fields",
        "nattype",
        "np_datetime",
        "offsets",
        "parsing",
        "period",
        "strptime",
        "vectorized",
        "timedeltas",
        "timestamps",
        "timezones",
        "tzconversion",
    ]

    # 定义 API 列表
    api = [
        "BaseOffset",
        "NaT",
        "NaTType",
        "iNaT",
        "nat_strings",
        "OutOfBoundsDatetime",
        "OutOfBoundsTimedelta",
        "Period",
        "IncompatibleFrequency",
        "Resolution",
        "Tick",
        "Timedelta",
        "dt64arr_to_periodarr",
        "Timestamp",
        "is_date_array_normalized",
        "ints_to_pydatetime",
        "normalize_i8_timestamps",
        "get_resolution",
        "delta_to_nanoseconds",
        "ints_to_pytimedelta",
        "localize_pydatetime",
        "tz_convert_from_utc",
        "tz_convert_from_utc_single",
        "to_offset",
        "tz_compare",
        "is_unitless",
        "astype_overflowsafe",
        "get_unit_from_dtype",
        "periods_per_day",
        "periods_per_second",
        "guess_datetime_format",
        "add_overflowsafe",
        "get_supported_dtype",
        "is_supported_dtype",
    ]

    # 预期的名称集合为子模块和 API 名称的并集
    expected = set(submodules + api)

    # 获取 tslibs 模块中所有不以双下划线开头的对象名称列表
    names = [x for x in dir(tslibs) if not x.startswith("__")]

    # 断言实际的名称集合与预期的名称集合相等
    assert set(names) == expected
```