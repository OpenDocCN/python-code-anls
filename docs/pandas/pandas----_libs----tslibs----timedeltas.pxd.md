# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timedeltas.pxd`

```
# 导入必要的 Cython 头文件，从 cpython.datetime 模块导入 timedelta 类型
from cpython.datetime cimport timedelta
# 从 numpy 模块的 Cython 接口中导入 int64_t 类型
from numpy cimport int64_t

# 从当前目录的 np_datetime 模块的 Cython 接口中导入 NPY_DATETIMEUNIT 常量
from .np_datetime cimport NPY_DATETIMEUNIT


# 声明一个 Cython 公共函数，用于 tslib，不对外部使用
cpdef int64_t get_unit_for_round(freq, NPY_DATETIMEUNIT creso) except? -1

# 声明一个 Cython 公共函数，用于将时间差转换为纳秒，可以处理舍入（round_ok）
# delta: 时间差
# reso: 时间单位，默认为未指定
# round_ok: 布尔值，表示是否可以舍入，默认为未指定
) except? -1

# 声明一个 Cython 私有函数，用于将对象 ts 转换为 timedelta64 类型，使用指定的单位
# ts: 时间戳对象
# unit: 时间单位字符串
cdef convert_to_timedelta64(object ts, str unit)

# 声明一个 Cython 私有函数，用于检查对象是否为任意 timedelta 标量
# obj: 待检查对象
cdef bint is_any_td_scalar(object obj)


# 定义一个 Cython 类 _Timedelta，继承自 Python 的 timedelta 类
cdef class _Timedelta(timedelta):
    # 声明只读属性
    cdef readonly:
        int64_t _value      # 储存纳秒级别的时间差值
        bint _is_populated  # 标志位，指示组件是否已填充
        int64_t _d, _h, _m, _s, _ms, _us, _ns  # 天、小时、分钟、秒、毫秒、微秒、纳秒组件
        NPY_DATETIMEUNIT _creso  # 时间单位（精度）

    # 声明一个公共方法，将 _Timedelta 类型转换为 Python 的 timedelta 类型
    cpdef timedelta to_pytimedelta(_Timedelta self)
    
    # 声明一个私有方法，用于检查 _Timedelta 实例是否具有纳秒（nanoseconds）精度
    cdef bint _has_ns(self)
    
    # 声明一个私有方法，用于检查 _Timedelta 实例是否在 Python timedelta 类型的界限内
    cdef bint _is_in_pytimedelta_bounds(self)
    
    # 声明一个私有方法，确保 _Timedelta 实例的组件已填充
    cdef _ensure_components(_Timedelta self)
    
    # 声明一个私有方法，比较两个 _Timedelta 实例在时间单位上的不匹配情况
    # other: 另一个 _Timedelta 实例
    # op: 比较操作符
    cdef bint _compare_mismatched_resos(self, _Timedelta other, op)
    
    # 声明一个私有方法，将 _Timedelta 实例转换为指定时间单位（精度）的实例
    # reso: 时间单位
    # round_ok: 是否允许舍入
    cdef _Timedelta _as_creso(self, NPY_DATETIMEUNIT reso, bint round_ok=*)
    
    # 声明一个公共方法，尝试将当前 _Timedelta 实例转换为与另一个 _Timedelta 实例相匹配的时间单位（精度）
    cpdef _maybe_cast_to_matching_resos(self, _Timedelta other)
```