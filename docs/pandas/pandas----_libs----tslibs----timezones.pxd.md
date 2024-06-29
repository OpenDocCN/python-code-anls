# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timezones.pxd`

```
# 导入需要的 cpython.datetime 模块中的类型和函数
from cpython.datetime cimport (
    datetime,       # 导入 datetime 类型
    timedelta,      # 导入 timedelta 类型
    tzinfo,         # 导入 tzinfo 类型
)

# 声明一个名为 utc_stdlib 的 tzinfo 对象
cdef tzinfo utc_stdlib

# 定义一个公共的函数 is_utc，用来检查给定的 tzinfo 对象是否表示 UTC 时间
cpdef bint is_utc(tzinfo tz)

# 定义一个私有的函数 is_tzlocal，用来检查给定的 tzinfo 对象是否是本地时间
cdef bint is_tzlocal(tzinfo tz)

# 定义一个私有的函数 is_zoneinfo，用来检查给定的 tzinfo 对象是否是区域信息
cdef bint is_zoneinfo(tzinfo tz)

# 定义一个私有的函数 treat_tz_as_pytz，用来指示是否应该将给定的 tzinfo 对象视为 pytz 类型
cdef bint treat_tz_as_pytz(tzinfo tz)

# 定义一个公共的函数 tz_compare，用来比较两个 tzinfo 对象的顺序
cpdef bint tz_compare(tzinfo start, tzinfo end)

# 定义一个公共的函数 get_timezone，根据给定的 tzinfo 对象获取时区信息
cpdef object get_timezone(tzinfo tz)

# 定义一个公共的函数 maybe_get_tz，根据给定的对象获取其可能关联的 tzinfo 对象
cpdef tzinfo maybe_get_tz(object tz)

# 定义一个私有的函数 get_utcoffset，根据给定的 tzinfo 对象和 datetime 对象获取 UTC 偏移量
cdef timedelta get_utcoffset(tzinfo tz, datetime obj)

# 定义一个公共的函数 is_fixed_offset，用来检查给定的 tzinfo 对象是否表示固定偏移量
cpdef bint is_fixed_offset(tzinfo tz)

# 定义一个私有的函数 get_dst_info，根据给定的 tzinfo 对象获取夏令时信息
cdef object get_dst_info(tzinfo tz)
```