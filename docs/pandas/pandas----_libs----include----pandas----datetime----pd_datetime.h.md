# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\datetime\pd_datetime.h`

```
/*
   This block includes copyright information and license terms for the code.
   It defines some macros and includes necessary headers for datetime operations.
*/
#pragma once

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif // NPY_NO_DEPRECATED_API

#include "pandas/datetime/date_conversions.h"
#include "pandas/vendored/numpy/datetime/np_datetime.h"
#include "pandas/vendored/numpy/datetime/np_datetime_strings.h"
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
   Definition of PandasDateTime_CAPI struct that contains function pointers for datetime operations.
*/
typedef struct {
  npy_datetime (*npy_datetimestruct_to_datetime)(NPY_DATETIMEUNIT,
                                                 const npy_datetimestruct *);
  int (*scaleNanosecToUnit)(int64_t *, NPY_DATETIMEUNIT);
  char *(*int64ToIso)(int64_t, NPY_DATETIMEUNIT, NPY_DATETIMEUNIT, size_t *);
  char *(*PyDateTimeToIso)(PyObject *, NPY_DATETIMEUNIT, size_t *);
  npy_datetime (*PyDateTimeToEpoch)(PyObject *, NPY_DATETIMEUNIT);
  char *(*int64ToIsoDuration)(int64_t, size_t *);
  void (*pandas_datetime_to_datetimestruct)(npy_datetime, NPY_DATETIMEUNIT,
                                            npy_datetimestruct *);
  void (*pandas_timedelta_to_timedeltastruct)(npy_datetime, NPY_DATETIMEUNIT,
                                              pandas_timedeltastruct *);
  int (*convert_pydatetime_to_datetimestruct)(PyObject *, npy_datetimestruct *);
  int (*cmp_npy_datetimestruct)(const npy_datetimestruct *,
                                const npy_datetimestruct *);
  PyArray_DatetimeMetaData (*get_datetime_metadata_from_dtype)(PyArray_Descr *);
  int (*parse_iso_8601_datetime)(const char *, int, int, npy_datetimestruct *,
                                 NPY_DATETIMEUNIT *, int *, int *, const char *,
                                 int, FormatRequirement);
  int (*get_datetime_iso_8601_strlen)(int, NPY_DATETIMEUNIT);
  int (*make_iso_8601_datetime)(npy_datetimestruct *, char *, size_t, int,
                                NPY_DATETIMEUNIT);
  int (*make_iso_8601_timedelta)(pandas_timedeltastruct *, char *, size_t *);
} PandasDateTime_CAPI;

/*
   Definition of the capsule name for the Pandas datetime API, limited by module.attribute.
*/
#define PandasDateTime_CAPSULE_NAME "pandas._pandas_datetime_CAPI"

/* block used as part of public API */
#ifndef _PANDAS_DATETIME_IMPL
/*
   Definition of PandasDateTimeAPI as a static variable initialized to NULL.
   It will be set to the imported capsule of the Pandas datetime API.
*/
static PandasDateTime_CAPI *PandasDateTimeAPI = NULL;

/*
   Macro to import PandasDateTimeAPI using PyCapsule_Import based on PandasDateTime_CAPSULE_NAME.
*/
#define PandasDateTime_IMPORT                                                  \
  PandasDateTimeAPI =                                                          \
      (PandasDateTime_CAPI *)PyCapsule_Import(PandasDateTime_CAPSULE_NAME, 0)
/*
定义了一系列的宏，用于将日期时间数据转换为不同的格式或单位，基于 PandasDateTimeAPI 的函数调用。
每个宏对应一个特定的日期时间转换或操作，包括不同单位之间的转换、ISO 格式的日期时间处理、比较等。
这些宏的目的是通过调用 PandasDateTimeAPI 提供的功能来实现日期时间操作的标准化和抽象化。
*/

#define npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT, npy_datetimestruct)   \
  PandasDateTimeAPI->npy_datetimestruct_to_datetime((NPY_DATETIMEUNIT),        \
                                                    (npy_datetimestruct))

#define scaleNanosecToUnit(value, unit)                                        \
  PandasDateTimeAPI->scaleNanosecToUnit((value), (unit))

#define int64ToIso(value, valueUnit, base, len)                                \
  PandasDateTimeAPI->int64ToIso((value), (valueUnit), (base), (len))

#define NpyDateTimeToEpoch(dt, base)                                           \
  PandasDateTimeAPI->NpyDateTimeToEpoch((dt), (base))

#define PyDateTimeToIso(obj, base, len)                                        \
  PandasDateTimeAPI->PyDateTimeToIso((obj), (base), (len))

#define PyDateTimeToEpoch(dt, base)                                            \
  PandasDateTimeAPI->PyDateTimeToEpoch((dt), (base))

#define int64ToIsoDuration(value, len)                                         \
  PandasDateTimeAPI->int64ToIsoDuration((value), (len))

#define pandas_datetime_to_datetimestruct(dt, base, out)                       \
  PandasDateTimeAPI->pandas_datetime_to_datetimestruct((dt), (base), (out))

#define pandas_timedelta_to_timedeltastruct(td, base, out)                     \
  PandasDateTimeAPI->pandas_timedelta_to_timedeltastruct((td), (base), (out))

#define convert_pydatetime_to_datetimestruct(dtobj, out)                       \
  PandasDateTimeAPI->convert_pydatetime_to_datetimestruct((dtobj), (out))

#define cmp_npy_datetimestruct(a, b)                                           \
  PandasDateTimeAPI->cmp_npy_datetimestruct((a), (b))

#define get_datetime_metadata_from_dtype(dtype)                                \
  PandasDateTimeAPI->get_datetime_metadata_from_dtype((dtype))

#define parse_iso_8601_datetime(str, len, want_exc, out, out_bestunit,         \
                                out_local, out_tzoffset, format, format_len,   \
                                format_requirement)                            \
  PandasDateTimeAPI->parse_iso_8601_datetime(                                  \
      (str), (len), (want_exc), (out), (out_bestunit), (out_local),            \
      (out_tzoffset), (format), (format_len), (format_requirement))

#define get_datetime_iso_8601_strlen(local, base)                              \
  PandasDateTimeAPI->get_datetime_iso_8601_strlen((local), (base))

#define make_iso_8601_datetime(dts, outstr, outlen, utc, base)                 \
  PandasDateTimeAPI->make_iso_8601_datetime((dts), (outstr), (outlen), (utc),  \
                                            (base))

#define make_iso_8601_timedelta(tds, outstr, outlen)                           \
  PandasDateTimeAPI->make_iso_8601_timedelta((tds), (outstr), (outlen))

#endif /* !defined(_PANDAS_DATETIME_IMPL) */

#ifdef __cplusplus
}
#endif
```