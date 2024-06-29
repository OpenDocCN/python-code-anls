# `D:\src\scipysrc\pandas\pandas\_libs\src\datetime\date_conversions.c`

```
/*
Copyright (c) 2020, PyData Development Team
All rights reserved.
Distributed under the terms of the BSD Simplified License.
The full license is in the LICENSE file, distributed with this software.
*/

// 包含必要的头文件：日期时间转换相关
#include "pandas/datetime/date_conversions.h"
#include "pandas/vendored/numpy/datetime/np_datetime.h"
#include "pandas/vendored/numpy/datetime/np_datetime_strings.h"

/*
 * Function: scaleNanosecToUnit
 * -----------------------------
 *
 * 将表示纳秒时间的整数值转换为指定单位。
 *
 * 直接修改提供的值。成功时返回0，错误时返回非零值。
 */
int scaleNanosecToUnit(int64_t *value, NPY_DATETIMEUNIT unit) {
  switch (unit) {
  case NPY_FR_ns:
    // 如果单位为纳秒，无需变换
    break;
  case NPY_FR_us:
    // 如果单位为微秒，将值除以1000
    *value /= 1000LL;
    break;
  case NPY_FR_ms:
    // 如果单位为毫秒，将值除以1000000
    *value /= 1000000LL;
    break;
  case NPY_FR_s:
    // 如果单位为秒，将值除以1000000000
    *value /= 1000000000LL;
    break;
  default:
    // 对于其他未知单位，返回错误
    return -1;
  }

  return 0;
}

/* Converts the int64_t representation of a datetime to ISO; mutates len */
char *int64ToIso(int64_t value, NPY_DATETIMEUNIT valueUnit,
                 NPY_DATETIMEUNIT base, size_t *len) {
  npy_datetimestruct dts;
  int ret_code;

  // 转换 int64_t 类型的日期时间到 datetime 结构体
  pandas_datetime_to_datetimestruct(value, valueUnit, &dts);

  // 计算 ISO 8601 格式日期时间字符串的长度
  *len = (size_t)get_datetime_iso_8601_strlen(0, base);
  char *result = PyObject_Malloc(*len);

  if (result == NULL) {
    // 内存分配失败时设置异常并返回空指针
    PyErr_NoMemory();
    return NULL;
  }

  // 将 datetime 结构体转换为 ISO 8601 格式字符串
  ret_code = make_iso_8601_datetime(&dts, result, *len, 0, base);
  if (ret_code != 0) {
    // 如果转换失败，设置值错误异常并释放内存
    PyErr_SetString(PyExc_ValueError,
                    "Could not convert datetime value to string");
    PyObject_Free(result);
  }

  // 注意 get_datetime_iso_8601_strlen 仅提供 ISO 字符串的通用长度，
  // 并非实际使用的大小
  *len = strlen(result);
  return result;
}

/* Converts the int64_t representation of a duration to ISO; mutates len */
char *int64ToIsoDuration(int64_t value, size_t *len) {
  pandas_timedeltastruct tds;
  int ret_code;

  // 转换 int64_t 类型的时长到 timedelta 结构体
  pandas_timedelta_to_timedeltastruct(value, NPY_FR_ns, &tds);

  // ISO 8601 时长的最大理论长度，包括最大为 64 位整数的天数
  char *result = PyObject_Malloc(71);
  if (result == NULL) {
    // 内存分配失败时设置异常并返回空指针
    PyErr_NoMemory();
    return NULL;
  }

  // 将 timedelta 结构体转换为 ISO 8601 时长字符串
  ret_code = make_iso_8601_timedelta(&tds, result, len);
  if (ret_code == -1) {
    // 如果转换失败，设置值错误异常并释放内存
    PyErr_SetString(PyExc_ValueError,
                    "Could not convert timedelta value to string");
    PyObject_Free(result);
    return NULL;
  }

  return result;
}
```