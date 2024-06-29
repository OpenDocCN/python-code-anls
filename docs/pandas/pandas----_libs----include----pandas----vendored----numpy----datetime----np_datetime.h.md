# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\vendored\numpy\datetime\np_datetime.h`

```
/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Copyright (c) 2005-2011, NumPy Developers
All rights reserved.

This file is derived from NumPy 1.7. See NUMPY_LICENSE.txt
*/

// 使用 #pragma once 指令确保头文件只被编译一次
#pragma once

// 定义 NPY_NO_DEPRECATED_API 宏，防止使用废弃的 API 版本
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif // NPY_NO_DEPRECATED_API

// 引入 NumPy 的 ndarraytypes.h 头文件
#include <numpy/ndarraytypes.h>

// 定义 pandas_timedeltastruct 结构体，包含了时间增量的各个单位
typedef struct {
  npy_int64 days;        // 天数
  npy_int32 hrs, min, sec, ms, us, ns, seconds, microseconds, nanoseconds;  // 小时、分钟、秒、毫秒、微秒、纳秒等
} pandas_timedeltastruct;

// 定义各种最小时间戳常量
static const npy_datetimestruct _AS_MIN_DTS = {1969, 12,     31,     23,    59,
                                               50,   776627, 963145, 224193};
static const npy_datetimestruct _FS_MIN_DTS = {1969, 12,     31,     21,    26,
                                               16,   627963, 145224, 193000};
static const npy_datetimestruct _PS_MIN_DTS = {1969, 9,      16,     5, 57,
                                               7,    963145, 224193, 0};
static const npy_datetimestruct _NS_MIN_DTS = {1677, 9,      21,     0, 12,
                                               43,   145224, 193000, 0};
static const npy_datetimestruct _US_MIN_DTS = {-290308, 12,     21, 19, 59,
                                               05,      224193, 0,  0};
static const npy_datetimestruct _MS_MIN_DTS = {-292275055, 5,      16, 16, 47,
                                               4,          193000, 0,  0};
static const npy_datetimestruct _S_MIN_DTS = {
    -292277022657, 1, 27, 8, 29, 53, 0, 0, 0};
static const npy_datetimestruct _M_MIN_DTS = {
    -17536621475646, 5, 4, 5, 53, 0, 0, 0, 0};

// 定义各种最大时间戳常量
static const npy_datetimestruct _AS_MAX_DTS = {1970, 1,      1,     0,     0,
                                               9,    223372, 36854, 775807};
static const npy_datetimestruct _FS_MAX_DTS = {1970, 1,      1,      2,     33,
                                               43,   372036, 854775, 807000};
static const npy_datetimestruct _PS_MAX_DTS = {1970, 4,     17,     18, 2,
                                               52,   36854, 775807, 0};
static const npy_datetimestruct _NS_MAX_DTS = {2262, 4,      11,     23, 47,
                                               16,   854775, 807000, 0};
static const npy_datetimestruct _US_MAX_DTS = {294247, 1,      10, 4, 0,
                                               54,     775807, 0,  0};
static const npy_datetimestruct _MS_MAX_DTS = {292278994, 8,      17, 7, 12,
                                               55,        807000, 0,  0};
static const npy_datetimestruct _S_MAX_DTS = {
    292277026596, 12, 4, 15, 30, 7, 0, 0, 0};
static const npy_datetimestruct _M_MAX_DTS = {
    17536621479585, 8, 30, 18, 7, 0, 0, 0, 0};

// pandas 需要的其他函数声明
// ----------------------------------------------------------------------------
PyObject *extract_utc_offset(PyObject *obj);
// 转换 numpy 的日期时间结构为标准的 datetime 对象
npy_datetime npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT base,
                                            const npy_datetimestruct *dts);

// 将 pandas 的 datetime 值转换为 numpy 的日期时间结构
void pandas_datetime_to_datetimestruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                       npy_datetimestruct *result);

// 将 pandas 的 timedelta 值转换为 pandas 的时间差结构
void pandas_timedelta_to_timedeltastruct(npy_timedelta val, NPY_DATETIMEUNIT fr,
                                         pandas_timedeltastruct *result);

// 外部声明：每月天数的表格，包含平年和闰年
extern const int days_per_month_table[2][12];

// numpy 派生代码所需的头文件内容
// ----------------------------------------------------------------------------

// 判断给定年份是否为闰年
int is_leapyear(npy_int64 year);

/*
 * 计算从 1970 年纪元开始到指定日期时间结构的天数偏移量。
 */
npy_int64 get_datetimestruct_days(const npy_datetimestruct *dts);

/*
 * 比较两个 npy_datetimestruct 对象的时间先后顺序。
 */
int cmp_npy_datetimestruct(const npy_datetimestruct *a,
                           const npy_datetimestruct *b);

/*
 * 根据分钟偏移量调整给定的日期时间结构。假定当前值有效。
 */
void add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes);

/*
 * 此函数从提供的 datetime dtype 中获取 DateTimeMetaData。
 */
PyArray_DatetimeMetaData get_datetime_metadata_from_dtype(PyArray_Descr *dtype);
```