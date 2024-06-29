# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\fields.pyi`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

from pandas._typing import npt  # 从 pandas._typing 导入 NumPy 相关类型定义

def build_field_sarray(
    dtindex: npt.NDArray[np.int64],  # 参数 dtindex 是一个 int64 类型的 NumPy 数组
    reso: int,  # 参数 reso 是一个整数，表示时间分辨率单位
) -> np.ndarray:  # 返回类型是一个 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释

def month_position_check(fields, weekdays) -> str | None:
    ...  # 函数体未实现，暂时不做具体注释

def get_date_name_field(
    dtindex: npt.NDArray[np.int64],  # 参数 dtindex 是一个 int64 类型的 NumPy 数组
    field: str,  # 字符串类型的字段名称
    locale: str | None = ...,  # 可选参数，表示语言环境，默认为 None
    reso: int = ...,  # 参数 reso 是一个整数，表示时间分辨率单位，默认值未指定
) -> npt.NDArray[np.object_]:  # 返回类型是一个对象类型的 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释

def get_start_end_field(
    dtindex: npt.NDArray[np.int64],  # 参数 dtindex 是一个 int64 类型的 NumPy 数组
    field: str,  # 字符串类型的字段名称
    freq_name: str | None = ...,  # 可选参数，表示频率名称，默认为 None
    month_kw: int = ...,  # 整数类型的参数，月份关键词，默认值未指定
    reso: int = ...,  # 参数 reso 是一个整数，表示时间分辨率单位，默认值未指定
) -> npt.NDArray[np.bool_]:  # 返回类型是一个布尔类型的 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释

def get_date_field(
    dtindex: npt.NDArray[np.int64],  # 参数 dtindex 是一个 int64 类型的 NumPy 数组
    field: str,  # 字符串类型的字段名称
    reso: int = ...,  # 参数 reso 是一个整数，表示时间分辨率单位，默认值未指定
) -> npt.NDArray[np.int32]:  # 返回类型是一个 int32 类型的 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释

def get_timedelta_field(
    tdindex: npt.NDArray[np.int64],  # 参数 tdindex 是一个 int64 类型的 NumPy 数组
    field: str,  # 字符串类型的字段名称
    reso: int = ...,  # 参数 reso 是一个整数，表示时间分辨率单位，默认值未指定
) -> npt.NDArray[np.int32]:  # 返回类型是一个 int32 类型的 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释

def get_timedelta_days(
    tdindex: npt.NDArray[np.int64],  # 参数 tdindex 是一个 int64 类型的 NumPy 数组
    reso: int = ...,  # 参数 reso 是一个整数，表示时间分辨率单位，默认值未指定
) -> npt.NDArray[np.int64]:  # 返回类型是一个 int64 类型的 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释

def isleapyear_arr(
    years: np.ndarray,  # 参数 years 是一个 NumPy 数组，表示年份列表
) -> npt.NDArray[np.bool_]:  # 返回类型是一个布尔类型的 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释

def build_isocalendar_sarray(
    dtindex: npt.NDArray[np.int64],  # 参数 dtindex 是一个 int64 类型的 NumPy 数组
    reso: int,  # 参数 reso 是一个整数，表示时间分辨率单位
) -> np.ndarray:  # 返回类型是一个 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释

def _get_locale_names(name_type: str, locale: str | None = ...):
    ...  # 函数体未实现，暂时不做具体注释

class RoundTo:
    @property
    def MINUS_INFTY(self) -> int: ...
    @property
    def PLUS_INFTY(self) -> int: ...
    @property
    def NEAREST_HALF_EVEN(self) -> int: ...
    @property
    def NEAREST_HALF_PLUS_INFTY(self) -> int: ...
    @property
    def NEAREST_HALF_MINUS_INFTY(self) -> int: ...
    # RoundTo 类定义了一些属性，但没有具体的实现，这些属性可能用于特定的舍入模式

def round_nsint64(
    values: npt.NDArray[np.int64],  # 参数 values 是一个 int64 类型的 NumPy 数组
    mode: RoundTo,  # 参数 mode 是一个 RoundTo 类的实例，表示舍入模式
    nanos: int,  # 参数 nanos 是一个整数，表示纳秒
) -> npt.NDArray[np.int64]:  # 返回类型是一个 int64 类型的 NumPy 数组
    ...  # 函数体未实现，暂时不做具体注释
```