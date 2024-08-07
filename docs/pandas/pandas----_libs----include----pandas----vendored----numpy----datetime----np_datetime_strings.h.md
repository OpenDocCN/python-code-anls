# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\vendored\numpy\datetime\np_datetime_strings.h`

```
/*
 * Copyright (c) 2016, PyData Development Team
 * All rights reserved.
 *
 * Distributed under the terms of the BSD Simplified License.
 *
 * The full license is in the LICENSE file, distributed with this software.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * Copyright (c) 2005-2011, NumPy Developers
 * All rights reserved.
 *
 * See NUMPY_LICENSE.txt for the license.
 *
 * This file implements string parsing and creation for NumPy datetime.
 */

#pragma once

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif // NPY_NO_DEPRECATED_API

/* 'format_requirement' can be one of three values:
 *      * PARTIAL_MATCH : Only require a partial match with 'format'.
 *           For example, if the string is '2020-01-01 05:00:00' and
 *           'format' is '%Y-%m-%d', then parse '2020-01-01';
 *      * EXACT_MATCH : require an exact match with 'format'. If the
 *           string is '2020-01-01', then the only format which will
 *           be able to parse it without error is '%Y-%m-%d';
 *      * INFER_FORMAT: parse without comparing 'format' (i.e. infer it).
 */
typedef enum { PARTIAL_MATCH, EXACT_MATCH, INFER_FORMAT } FormatRequirement;

/*
 * Parses (almost) standard ISO 8601 date strings. The differences are:
 *
 * + The date "20100312" is parsed as the year 20100312, not as
 *   equivalent to "2010-03-12". The '-' in the dates are not optional.
 * + Only seconds may have a decimal point, with up to 18 digits after it
 *   (maximum attoseconds precision).
 * + Either a 'T' as in ISO 8601 or a ' ' may be used to separate
 *   the date and the time. Both are treated equivalently.
 * + Doesn't (yet) handle the "YYYY-DDD" or "YYYY-Www" formats.
 * + Doesn't handle leap seconds (seconds value has 60 in these cases).
 * + Doesn't handle 24:00:00 as synonym for midnight (00:00:00) tomorrow
 * + Accepts special values "NaT" (not a time), "Today", (current
 *   day according to local time) and "Now" (current time in UTC).
 *
 * 'str' must be a NULL-terminated string, and 'len' must be its length.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_local' gets whether returned value contains timezone. 0 for UTC, 1 for
 * local time. 'out_tzoffset' gets set to timezone offset by minutes if the
 * parsed time was in local time, to 0 otherwise. The values 'now' and 'today'
 * don't get counted as local, and neither do UTC +/-#### timezone offsets,
 * because they aren't using the computer's local timezone offset.
 *
 * Returns 0 on success, -1 on failure.
 */
int parse_iso_8601_datetime(const char *str, int len, int want_exc,
                            npy_datetimestruct *out,
                            NPY_DATETIMEUNIT *out_bestunit, int *out_local,
                            int *out_tzoffset, const char *format,
                            int format_len,
                            FormatRequirement format_requirement);
/*
 * 提供一个字符串长度，用于根据给定的本地和单位设置转换 datetime 对象。
 */
int get_datetime_iso_8601_strlen(int local, NPY_DATETIMEUNIT base);

/*
 * 将 npy_datetimestruct 转换为（几乎）带有时区 Z（UTC）的 ISO 8601 格式的
 * NULL 结尾字符串。
 *
 * 'base' 限制输出到该单位。将 'base' 设置为 -1 可以自动检测一个基准，其后所有值都为零。
 *
 * 返回值：成功返回 0，失败返回 -1（例如，如果输出字符串太短）。
 */
int make_iso_8601_datetime(npy_datetimestruct *dts, char *outstr, size_t outlen,
                           int utc, NPY_DATETIMEUNIT base);

/*
 * 将 pandas_timedeltastruct 转换为 ISO 8601 格式的字符串。
 *
 * 改变 outlen 来提供非 NULL 结尾字符串的大小。
 *
 * 目前没有错误处理。
 */
int make_iso_8601_timedelta(pandas_timedeltastruct *tds, char *outstr,
                            size_t *outlen);
```