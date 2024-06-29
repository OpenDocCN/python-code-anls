# `.\numpy\numpy\_core\src\multiarray\datetime.c`

```
/*
 * This file implements core functionality for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpyos.h"

#include "npy_config.h"

#include "common.h"
#include "numpy/arrayscalars.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "convert_datatype.h"
#include "array_method.h"
#include "dtypemeta.h"
#include "usertypes.h"

#include "dtype_transfer.h"
#include "lowlevel_strided_loops.h"

#include <datetime.h>
#include <time.h>

/*
 * Computes the python `ret, d = divmod(d, unit)`.
 *
 * Note that GCC is smart enough at -O2 to eliminate the `if(*d < 0)` branch
 * for subsequent calls to this command - it is able to deduce that `*d >= 0`.
 */
static inline
npy_int64 extract_unit_64(npy_int64 *d, npy_int64 unit) {
    assert(unit > 0);
    npy_int64 div = *d / unit;
    npy_int64 mod = *d % unit;
    if (mod < 0) {
        mod += unit;
        div -= 1;
    }
    assert(mod >= 0);
    *d = mod;
    return div;
}

static inline
npy_int32 extract_unit_32(npy_int32 *d, npy_int32 unit) {
    assert(unit > 0);
    npy_int32 div = *d / unit;
    npy_int32 mod = *d % unit;
    if (mod < 0) {
        mod += unit;
        div -= 1;
    }
    assert(mod >= 0);
    *d = mod;
    return div;
}

/*
 * Imports the PyDateTime functions so we can create these objects.
 * This is called during module initialization
 */
NPY_NO_EXPORT void
numpy_pydatetime_import(void)
{
    PyDateTime_IMPORT;
}

/* Exported as DATETIMEUNITS in multiarraymodule.c */
NPY_NO_EXPORT char const *_datetime_strings[NPY_DATETIME_NUMUNITS] = {
    "Y",
    "M",
    "W",
    "<invalid>",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "ns",
    "ps",
    "fs",
    "as",
    "generic"
};

/* Days per month, regular year and leap year */
NPY_NO_EXPORT int _days_per_month_table[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

/*
 * Returns 1 if the given year is a leap year, 0 otherwise.
 */
NPY_NO_EXPORT int
is_leapyear(npy_int64 year)
{
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 ||
            (year % 400) == 0);
}

/*
 * Calculates the days offset from the 1970 epoch.
 */
NPY_NO_EXPORT npy_int64
get_datetimestruct_days(const npy_datetimestruct *dts)
{
    int i, month;
    npy_int64 year, days = 0;
    int *month_lengths;

    year = dts->year - 1970;
    days = year * 365;

    /* Adjust for leap years */
    for (i = 1970; i < dts->year; i++) {
        if (is_leapyear(i))
            days++;
    }

    month_lengths = _days_per_month_table[is_leapyear(dts->year)];
    for (month = 1; month < dts->month; month++) {
        days += month_lengths[month - 1];
    }

    days += dts->day - 1;

    return days;
}
    // 如果天数大于等于0，则表示是当前年份之后的日期计算
    if (days >= 0) {
        /*
         * 1968 is the closest leap year before 1970.
         * Exclude the current year, so add 1.
         */
        // 将年份加一，因为1968年是1970年之前最近的闰年
        year += 1;
        /* Add one day for each 4 years */
        // 每4年增加一天
        days += year / 4;
        /* 1900 is the closest previous year divisible by 100 */
        // 1900年是最接近1970年之前可被100整除的年份
        // 每100年减少一天
        year += 68;
        /* Subtract one day for each 100 years */
        days -= year / 100;
        /* 1600 is the closest previous year divisible by 400 */
        // 1600年是最接近1970年之前可被400整除的年份
        // 每400年增加一天
        year += 300;
        /* Add one day for each 400 years */
        days += year / 400;
    }
    else {
        /*
         * 1972 is the closest later year after 1970.
         * Include the current year, so subtract 2.
         */
        // 如果天数小于0，则表示是当前年份之前的日期计算
        year -= 2;
        /* Subtract one day for each 4 years */
        // 每4年减少一天
        days += year / 4;
        /* 2000 is the closest later year divisible by 100 */
        // 2000年是最接近1970年之后可被100整除的年份
        // 每100年增加一天
        year -= 28;
        /* Add one day for each 100 years */
        days -= year / 100;
        /* 2000 is also the closest later year divisible by 400 */
        // 2000年也是最接近1970年之后可被400整除的年份
        // 每400年减少一天
        days += year / 400;
    }

    // 获取当前年份是否为闰年的月份长度表
    month_lengths = _days_per_month_table[is_leapyear(dts->year)];
    // 获取日期结构体中的月份值，并转换为数组下标（从0开始）
    month = dts->month - 1;

    /* Add the months */
    // 累加月份之前的天数
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    // 累加日期结构体中的日数
    days += dts->day - 1;

    // 返回累计的天数作为结果
    return days;
/*
 * Calculates the minutes offset from the 1970 epoch.
 */
NPY_NO_EXPORT npy_int64
get_datetimestruct_minutes(const npy_datetimestruct *dts)
{
    // 计算天数偏移量乘以每天的分钟数，从而得到分钟数偏移量
    npy_int64 days = get_datetimestruct_days(dts) * 24 * 60;
    // 加上小时对应的分钟数
    days += dts->hour * 60;
    // 加上分钟数偏移量
    days += dts->min;

    return days;
}

/*
 * Modifies '*days_' to be the day offset within the year,
 * and returns the year.
 */
static npy_int64
days_to_yearsdays(npy_int64 *days_)
{
    // 定义每400年的天数
    const npy_int64 days_per_400years = (400*365 + 100 - 4 + 1);
    /* 调整以相对于2000年的年份（2000年可以被400整除） */
    npy_int64 days = (*days_) - (365*30 + 7);
    npy_int64 year;

    /* 将400年周期拆分为年份和年内天数 */
    year = 400 * extract_unit_64(&days, days_per_400years);

    /* 计算在400年周期内的年份和年内天数 */
    if (days >= 366) {
        year += 100 * ((days-1) / (100*365 + 25 - 1));
        days = (days-1) % (100*365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days+1) / (4*365 + 1));
            days = (days+1) % (4*365 + 1);
            if (days >= 366) {
                year += (days-1) / 365;
                days = (days-1) % 365;
            }
        }
    }

    *days_ = days;
    return year + 2000;
}

/* Extracts the month number from a 'datetime64[D]' value */
NPY_NO_EXPORT int
days_to_month_number(npy_datetime days)
{
    npy_int64 year;
    int *month_lengths, i;

    year = days_to_yearsdays(&days);
    month_lengths = _days_per_month_table[is_leapyear(year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            // 返回月份编号，从1开始
            return i + 1;
        }
        else {
            days -= month_lengths[i];
        }
    }

    /* Should never get here */
    return 1;
}

/*
 * Fills in the year, month, day in 'dts' based on the days
 * offset from 1970.
 */
static void
set_datetimestruct_days(npy_int64 days, npy_datetimestruct *dts)
{
    int *month_lengths, i;

    // 填充年份和年内天数到结构体 'dts' 中
    dts->year = days_to_yearsdays(&days);
    month_lengths = _days_per_month_table[is_leapyear(dts->year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            // 填充月份和日期到结构体 'dts' 中
            dts->month = i + 1;
            dts->day = (int)days + 1;
            return;
        }
        else {
            days -= month_lengths[i];
        }
    }
}

/*NUMPY_API
 *
 * Converts a datetime from a datetimestruct to a datetime based
 * on some metadata. The date is assumed to be valid.
 *
 * TODO: If meta->num is really big, there could be overflow
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
NpyDatetime_ConvertDatetimeStructToDatetime64(PyArray_DatetimeMetaData *meta,
                                    const npy_datetimestruct *dts,
                                    npy_datetime *out)
{
    npy_datetime ret;
    NPY_DATETIMEUNIT base = meta->base;

    /* If the datetimestruct is NaT, return NaT */
    if (dts->year == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        return 0;
    }
}
    /* 如果日期时间对象的基础单位是通用单位（generic units），则不能实例化一个 datetime 对象 */
    if (meta->base == NPY_FR_GENERIC) {
        /* 抛出一个值错误异常，指示不能使用通用单位创建非 NaT 的 NumPy datetime */
        PyErr_SetString(PyExc_ValueError,
                    "Cannot create a NumPy datetime other than NaT "
                    "with generic units");
        // 返回 -1 表示操作失败
        return -1;
    }

    // 如果基础单位是年
    if (base == NPY_FR_Y) {
        /* 截断到年份 */
        // 计算相对于 1970 年的年数差
        ret = dts->year - 1970;
    }
    // 如果基础单位是月
    else if (base == NPY_FR_M) {
        /* 截断到月份 */
        // 计算相对于 1970 年的月数差
        ret = 12 * (dts->year - 1970) + (dts->month - 1);
    }
    else {
        /* 否则计算开始日期到现在的天数 */
        npy_int64 days = get_datetimestruct_days(dts);

        switch (base) {
            case NPY_FR_W:
                /* 截断到周 */
                if (days >= 0) {
                    ret = days / 7;
                }
                else {
                    ret = (days - 6) / 7;
                }
                break;
            case NPY_FR_D:
                ret = days;
                break;
            case NPY_FR_h:
                ret = days * 24 +
                      dts->hour;
                break;
            case NPY_FR_m:
                ret = (days * 24 +
                      dts->hour) * 60 +
                      dts->min;
                break;
            case NPY_FR_s:
                ret = ((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec;
                break;
            case NPY_FR_ms:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000 +
                      dts->us / 1000;
                break;
            case NPY_FR_us:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us;
                break;
            case NPY_FR_ns:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000 +
                      dts->ps / 1000;
                break;
            case NPY_FR_ps:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps;
                break;
            case NPY_FR_fs:
                /* 只有2.6小时 */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000 +
                      dts->as / 1000;
                break;
            case NPY_FR_as:
                /* 只有9.2秒 */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000000 +
                      dts->as;
                break;
            default:
                /* 某些数据损坏 */
                PyErr_SetString(PyExc_ValueError,
                        "NumPy datetime metadata with corrupt unit value");
                return -1;
        }
    }
    /* 如果元数据中的数字大于1，则进行除法运算 */
    if (meta->num > 1) {
        /* 如果返回值大于等于0，则对返回值进行除法操作 */
        if (ret >= 0) {
            ret /= meta->num;
        }
        /* 如果返回值小于0，则对返回值进行特定的除法操作 */
        else {
            ret = (ret - meta->num + 1) / meta->num;
        }
    }

    /* 将处理后的结果存入输出参数中 */
    *out = ret;

    /* 返回成功状态 */
    return 0;
/*NUMPY_API
 *
 * Converts a datetime based on the given metadata into a datetimestruct
 */
NPY_NO_EXPORT int
NpyDatetime_ConvertDatetime64ToDatetimeStruct(
        PyArray_DatetimeMetaData *meta, npy_datetime dt,
        npy_datetimestruct *out)
{
    npy_int64 days;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->year = 1970;
    out->month = 1;
    out->day = 1;

    /* NaT is signaled in the year */
    if (dt == NPY_DATETIME_NAT) {
        out->year = NPY_DATETIME_NAT;
        return 0;
    }

    /* Datetimes can't be in generic units */
    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot convert a NumPy datetime value other than NaT "
                    "with generic units");
        return -1;
    }

    /* TODO: Change to a mechanism that avoids the potential overflow */
    dt *= meta->num;

    /*
     * Note that care must be taken with the / and % operators
     * for negative values.
     */
    }

    return 0;
}


注释：
    // 将字符串转换为长整型数，使用指定的基数（进制）
    return strtol(str, (char**)endptr, base);
/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit multiplier + enum value, which are populated
 * into out_meta. Other metadata is left along.
 *
 * 'metastr' is only used in the error message, and may be NULL.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_datetime_extended_unit_from_string(char const *str, Py_ssize_t len,
                                    char const *metastr,
                                    PyArray_DatetimeMetaData *out_meta)
{
    char const *substr = str, *substrend = NULL;
    int den = 1;
    npy_longlong true_meta_val;

    /* First comes an optional integer multiplier */
    out_meta->num = (int)strtol_const(substr, &substrend, 10);
    if (substr == substrend) {
        out_meta->num = 1;  // 设置默认值为 1
    }
    else {
        // 检查是否存在32位整数溢出
        char *endptr = NULL;
        true_meta_val = NumPyOS_strtoll(substr, &endptr, 10);
        if (true_meta_val > INT_MAX || true_meta_val < 0) {
            goto bad_input;  // 如果溢出，跳转到错误处理
        }
    }
    substr = substrend;

    /* Next comes the unit itself, followed by either '/' or the string end */
    substrend = substr;
    while (substrend-str < len && *substrend != '/') {
        ++substrend;
    }
    if (substr == substrend) {
        goto bad_input;  // 如果没有单位，跳转到错误处理
    }
    out_meta->base = parse_datetime_unit_from_string(substr,
                                                     substrend - substr,
                                                     metastr);
    if (out_meta->base == NPY_FR_ERROR ) {
        return -1;  // 如果解析单位失败，返回错误
    }
    substr = substrend;

    /* Next comes an optional integer denominator */
    if (substr-str < len && *substr == '/') {
        substr++;
        den = (int)strtol_const(substr, &substrend, 10);
        /* If the '/' exists, there must be a number followed by ']' */
        if (substr == substrend || *substrend != ']') {
            goto bad_input;  // 如果没有找到预期的分母格式，跳转到错误处理
        }
        substr = substrend + 1;
    }
    else if (substr-str != len) {
        goto bad_input;  // 如果不是最后一个字符，但没有预期的分母格式，跳转到错误处理
    }

    if (den != 1) {
        if (convert_datetime_divisor_to_multiple(
                                out_meta, den, metastr) < 0) {
            return -1;  // 如果转换分母到倍数失败，返回错误
        }
    }

    return 0;  // 成功解析并填充 out_meta 结构体

bad_input:
    if (metastr != NULL) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\" at position %zd",
                metastr, substr-metastr);  // 格式化错误消息，包含 metastr 和位置信息
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\"",
                str);  // 格式化错误消息，包含 str
    }

    return -1;  // 返回错误状态
}

/*
 * Parses the metadata string into the metadata C structure.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_datetime_metadata_from_metastr(char const *metastr, Py_ssize_t len,
                                    PyArray_DatetimeMetaData *out_meta)
{
    char const *substr = metastr, *substrend = NULL;

    /* Treat the empty string as generic units */
    // 如果长度为0，设置输出元数据的基本类型为通用类型，并且设置数量为1，然后返回0
    if (len == 0) {
        out_meta->base = NPY_FR_GENERIC;
        out_meta->num = 1;

        return 0;
    }

    /* 元数据字符串必须以'['开头 */
    // 如果长度小于3或者子字符串指针所指向的字符不是'['，跳转到输入格式错误处理
    if (len < 3 || *substr++ != '[') {
        goto bad_input;
    }

    substrend = substr;
    // 找到']'或者达到字符串末尾为止
    while (substrend - metastr < len && *substrend != ']') {
        ++substrend;
    }
    // 如果字符串遍历到末尾或者子字符串指针没有移动，表示输入格式错误，跳转到输入格式错误处理
    if (substrend - metastr == len || substr == substrend) {
        substr = substrend;
        goto bad_input;
    }

    /* 解析'[]'中的扩展单位 */
    // 调用函数从字符串中解析扩展单位，如果解析失败则返回-1
    if (parse_datetime_extended_unit_from_string(substr, substrend-substr,
                                                metastr, out_meta) < 0) {
        return -1;
    }

    substr = substrend+1;

    // 如果子字符串指针和元数据字符串指针之差不等于长度，表示输入格式错误，跳转到输入格式错误处理
    if (substr - metastr != len) {
        goto bad_input;
    }

    return 0;
bad_input:
    # 如果 substr 和 metastr 不相等，说明存在错误的元数据字符串位置，生成相应的错误信息
    if (substr != metastr) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\" at position %zd",
                metastr, substr - metastr);
    }
    # 否则，生成错误信息表示整体元数据字符串存在问题
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\"",
                metastr);
    }

    # 返回 -1 表示解析失败
    return -1;
}

/*
 * Converts a datetype dtype string into a dtype descr object.
 * The "type" string should be NULL-terminated.
 */
NPY_NO_EXPORT PyArray_Descr *
parse_dtype_from_datetime_typestr(char const *typestr, Py_ssize_t len)
{
    # 创建日期时间元数据结构体和元数据字符串指针
    PyArray_DatetimeMetaData meta;
    char const *metastr = NULL;
    int is_timedelta = 0;
    Py_ssize_t metalen = 0;

    # 检查字符串长度是否小于 2，生成相应的错误信息并返回空指针
    if (len < 2) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime typestr \"%s\"",
                typestr);
        return NULL;
    }

    /*
     * First validate that the root is correct,
     * and get the metadata string address
     */
    # 检查字符串开头是否匹配特定格式，确定是否为 timedelta 或 datetime 类型
    if (typestr[0] == 'm' && typestr[1] == '8') {
        is_timedelta = 1;
        metastr = typestr + 2;
        metalen = len - 2;
    }
    else if (typestr[0] == 'M' && typestr[1] == '8') {
        is_timedelta = 0;
        metastr = typestr + 2;
        metalen = len - 2;
    }
    else if (len >= 11 && strncmp(typestr, "timedelta64", 11) == 0) {
        is_timedelta = 1;
        metastr = typestr + 11;
        metalen = len - 11;
    }
    else if (len >= 10 && strncmp(typestr, "datetime64", 10) == 0) {
        is_timedelta = 0;
        metastr = typestr + 10;
        metalen = len - 10;
    }
    else {
        # 如果不匹配任何已知格式，生成相应的错误信息并返回空指针
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime typestr \"%s\"",
                typestr);
        return NULL;
    }

    /* Parse the metadata string into a metadata struct */
    # 解析元数据字符串到元数据结构体中
    if (parse_datetime_metadata_from_metastr(metastr, metalen, &meta) < 0) {
        return NULL;
    }

    # 根据解析的元数据创建日期时间类型描述对象并返回
    return create_datetime_dtype(is_timedelta ? NPY_TIMEDELTA : NPY_DATETIME,
                                    &meta);
}

static NPY_DATETIMEUNIT _multiples_table[16][4] = {
    {12, 52, 365},                            /* NPY_FR_Y */
    {NPY_FR_M, NPY_FR_W, NPY_FR_D},
    {4,  30, 720},                            /* NPY_FR_M */
    {NPY_FR_W, NPY_FR_D, NPY_FR_h},
    {7,  168, 10080},                         /* NPY_FR_W */
    {NPY_FR_D, NPY_FR_h, NPY_FR_m},
    {0},                                      /* Gap for removed NPY_FR_B */
    {0},
    {24, 1440, 86400},                        /* NPY_FR_D */
    {NPY_FR_h, NPY_FR_m, NPY_FR_s},
    {60, 3600},                               /* NPY_FR_h */
    {NPY_FR_m, NPY_FR_s},
    {60, 60000},                              /* NPY_FR_m */
    {NPY_FR_s, NPY_FR_ms},
    {1000, 1000000},                          /* >=NPY_FR_s */
    {0, 0}
};
/*
 * Translate divisors into multiples of smaller units.
 * 'metastr' is used for the error message if the divisor doesn't work,
 * and can be NULL if the metadata didn't come from a string.
 *
 * This function only affects the 'base' and 'num' values in the metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_divisor_to_multiple(PyArray_DatetimeMetaData *meta,
                                    int den, char const *metastr)
{
    int i, num, ind;
    NPY_DATETIMEUNIT *totry;
    NPY_DATETIMEUNIT *baseunit;
    int q, r;

    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
            "Can't use 'den' divisor with generic units");
        return -1;
    }

    num = 3;
    if (meta->base == NPY_FR_W) {
        num = 4;
    }
    else if (meta->base > NPY_FR_D) {
        num = 2;
    }
    if (meta->base >= NPY_FR_s) {
        /* _multiplies_table only has entries up to NPY_FR_s */
        ind = ((int)NPY_FR_s - (int)NPY_FR_Y)*2;
        totry = _multiples_table[ind];
        baseunit = _multiples_table[ind + 1];
        baseunit[0] = meta->base + 1;
        baseunit[1] = meta->base + 2;
        if (meta->base == NPY_FR_as - 1) {
            num = 1;
        }
        if (meta->base == NPY_FR_as) {
            num = 0;
        }
    }
    else {
        ind = ((int)meta->base - (int)NPY_FR_Y)*2;
        totry = _multiples_table[ind];
        baseunit = _multiples_table[ind + 1];
    }

    for (i = 0; i < num; i++) {
        q = totry[i] / den;  // Calculate quotient of division
        r = totry[i] % den;  // Calculate remainder of division
        if (r == 0) {
            break;  // Found a valid multiple
        }
    }
    if (i == num) {
        if (metastr == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "divisor (%d) is not a multiple of a lower-unit "
                    "in datetime metadata", den);
        }
        else {
            PyErr_Format(PyExc_ValueError,
                    "divisor (%d) is not a multiple of a lower-unit "
                    "in datetime metadata \"%s\"", den, metastr);
        }
        return -1;  // Return failure if no valid multiple found
    }
    meta->base = baseunit[i];  // Update metadata with the base unit
    meta->num *= q;  // Update metadata with the new multiplier

    return 0;  // Return success
}

/*
 * Lookup table for factors between datetime units, except
 * for years and months.
 */
static npy_uint32
_datetime_factors[] = {
    1,  /* Years - not used */
    1,  /* Months - not used */
    7,  /* Weeks -> Days */
    1,  /* Business Days - was removed but a gap still exists in the enum */
    24, /* Days -> Hours */
    60, /* Hours -> Minutes */
    60, /* Minutes -> Seconds */
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1,   /* Attoseconds are the smallest base unit */
    0    /* Generic units don't have a conversion */
};

/*
 * Returns the scale factor between the units. Does not validate
 * that bigbase represents larger units than littlebase, or that
 * the units are not generic.
 *
 * Returns 0 if there is an overflow.
 */
static npy_uint64
/*
 * 计算从大时间单位到小时间单位的转换因子
 */
get_datetime_units_factor(NPY_DATETIMEUNIT bigbase, NPY_DATETIMEUNIT littlebase)
{
    // 初始化转换因子为1，从大时间单位开始计算
    npy_uint64 factor = 1;
    NPY_DATETIMEUNIT unit = bigbase;

    // 循环直到大时间单位达到或超过小时间单位
    while (unit < littlebase) {
        // 将当前时间单位的因子乘到转换因子上
        factor *= _datetime_factors[unit];

        /*
         * 检测是否发生溢出，通过禁止最高16位为1来检测。
         * 这样的误差边界比任何时间因子都大得多。
         */
        if (factor & 0xff00000000000000ULL) {
            return 0; // 如果溢出，返回0表示失败
        }
        ++unit; // 继续下一个时间单位
    }
    return factor; // 返回计算得到的转换因子
}

/* 欧几里德算法求两个正整数的最大公约数 */
static npy_uint64
_uint64_euclidean_gcd(npy_uint64 x, npy_uint64 y)
{
    npy_uint64 tmp;

    // 确保 x <= y
    if (x > y) {
        tmp = x;
        x = y;
        y = tmp;
    }
    // 使用欧几里德算法求最大公约数
    while (x != y && y != 0) {
        tmp = x % y;
        x = y;
        y = tmp;
    }

    return x; // 返回最大公约数
}

/*
 * 计算从 'src_meta' 元数据转换为 'dst_meta' 元数据的转换因子。
 *
 * 如果发生溢出，将同时设置 out_num 和 out_denom 为0，但不设置错误。
 */
NPY_NO_EXPORT void
get_datetime_conversion_factor(PyArray_DatetimeMetaData *src_meta,
                                PyArray_DatetimeMetaData *dst_meta,
                                npy_int64 *out_num, npy_int64 *out_denom)
{
    int src_base, dst_base, swapped;
    npy_uint64 num = 1, denom = 1, tmp, gcd;

    /* 对于通用单位，到目标单位的转换因子为1 */
    if (src_meta->base == NPY_FR_GENERIC) {
        *out_num = 1;
        *out_denom = 1;
        return;
    }
    /*
     * 从特定单位转换到通用单位是一个错误。
     */
    else if (dst_meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot convert from specific units to generic "
                    "units in NumPy datetimes or timedeltas");
        *out_num = 0;
        *out_denom = 0;
        return;
    }

    // 确定基础时间单位的顺序和是否交换
    if (src_meta->base <= dst_meta->base) {
        src_base = src_meta->base;
        dst_base = dst_meta->base;
        swapped = 0;
    }
    else {
        src_base = dst_meta->base;
        dst_base = src_meta->base;
        swapped = 1;
    }
    if (src_base != dst_base) {
        /*
         * Conversions between years/months and other units use
         * the factor averaged over the 400 year leap year cycle.
         */
        // 如果源时间单位不等于目标时间单位，则进行单位转换计算
        if (src_base == NPY_FR_Y) {
            // 如果源时间单位是年（NPY_FR_Y）
            if (dst_base == NPY_FR_M) {
                // 如果目标时间单位是月（NPY_FR_M），将年数转换为月数
                num *= 12;
            }
            else if (dst_base == NPY_FR_W) {
                // 如果目标时间单位是周（NPY_FR_W）
                // 年转周的计算，考虑400年闰年周期的平均值
                num *= (97 + 400*365);
                denom *= 400*7;
            }
            else {
                /* Year -> Day */
                // 如果目标时间单位是日（NPY_FR_D）
                // 年转日的计算，考虑400年闰年周期的平均值
                num *= (97 + 400*365);
                denom *= 400;
                // 将日转换为目标时间单位
                num *= get_datetime_units_factor(NPY_FR_D, dst_base);
            }
        }
        else if (src_base == NPY_FR_M) {
            // 如果源时间单位是月（NPY_FR_M）
            if (dst_base == NPY_FR_W) {
                // 如果目标时间单位是周（NPY_FR_W）
                // 月转周的计算，考虑400年闰年周期的平均值
                num *= (97 + 400*365);
                denom *= 400*12*7;
            }
            else {
                /* Month -> Day */
                // 如果目标时间单位是日（NPY_FR_D）
                // 月转日的计算，考虑400年闰年周期的平均值
                num *= (97 + 400*365);
                denom *= 400*12;
                // 将日转换为目标时间单位
                num *= get_datetime_units_factor(NPY_FR_D, dst_base);
            }
        }
        else {
            // 对于其他情况，使用通用的时间单位转换因子
            num *= get_datetime_units_factor(src_base, dst_base);
        }
    }

    /* If something overflowed, make both num and denom 0 */
    // 如果发生溢出，则将分子和分母都设为0
    if (num == 0) {
        PyErr_Format(PyExc_OverflowError,
                    "Integer overflow while computing the conversion "
                    "factor between NumPy datetime units %s and %s",
                    _datetime_strings[src_base],
                    _datetime_strings[dst_base]);
        *out_num = 0;
        *out_denom = 0;
        return;
    }

    /* Swap the numerator and denominator if necessary */
    // 如果需要交换分子和分母，则进行交换
    if (swapped) {
        tmp = num;
        num = denom;
        denom = tmp;
    }

    // 根据源和目标的元数据调整分子和分母
    num *= src_meta->num;
    denom *= dst_meta->num;

    /* Return as a fraction in reduced form */
    // 返回化简后的分数形式
    gcd = _uint64_euclidean_gcd(num, denom);
    *out_num = (npy_int64)(num / gcd);
    *out_denom = (npy_int64)(denom / gcd);
/*
 * Determines whether the 'divisor' metadata divides evenly into
 * the 'dividend' metadata.
 */
NPY_NO_EXPORT npy_bool
datetime_metadata_divides(
                        PyArray_DatetimeMetaData *dividend,
                        PyArray_DatetimeMetaData *divisor,
                        int strict_with_nonlinear_units)
{
    npy_uint64 num1, num2;

    /*
     * Any unit can always divide into generic units. In other words, we
     * should be able to convert generic units into any more specific unit.
     */
    if (dividend->base == NPY_FR_GENERIC) {
        return 1;
    }
    /*
     * However, generic units cannot always divide into more specific units.
     * We cannot safely convert datetimes with units back into generic units.
     */
    else if (divisor->base == NPY_FR_GENERIC) {
        return 0;
    }

    num1 = (npy_uint64)dividend->num;
    num2 = (npy_uint64)divisor->num;

    /* If the bases are different, factor in a conversion */
    if (dividend->base != divisor->base) {
        /*
         * Years and Months are incompatible with
         * all other units (except years and months are compatible
         * with each other).
         */
        if (dividend->base == NPY_FR_Y) {
            if (divisor->base == NPY_FR_M) {
                num1 *= 12;
            }
            else if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }
        else if (divisor->base == NPY_FR_Y) {
            if (dividend->base == NPY_FR_M) {
                num2 *= 12;
            }
            else if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }
        else if (dividend->base == NPY_FR_M || divisor->base == NPY_FR_M) {
            if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }

        /* Take the greater base (unit sizes are decreasing in enum) */
        if (dividend->base > divisor->base) {
            num2 *= get_datetime_units_factor(divisor->base, dividend->base);
            if (num2 == 0) {
                return 0;
            }
        }
        else {
            num1 *= get_datetime_units_factor(dividend->base, divisor->base);
            if (num1 == 0) {
                return 0;
            }
        }
    }

    /* Crude, incomplete check for overflow */
    if (num1&0xff00000000000000LL || num2&0xff00000000000000LL ) {
        return 0;
    }

    return (num1 % num2) == 0;
}
/*
 * Check if casting between two datetime units is allowed based on the specified casting mode.
 */
can_cast_datetime64_units(NPY_DATETIMEUNIT src_unit,
                          NPY_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting)
{
    switch (casting) {
        /* Allow any casting without checking types */
        case NPY_UNSAFE_CASTING:
            return 1;

        /*
         * Allow casting between all datetime units with 'same_kind' casting,
         * except if either unit is NPY_FR_GENERIC, in which case, only allow
         * if src_unit is NPY_FR_GENERIC.
         */
        case NPY_SAME_KIND_CASTING:
            if (src_unit == NPY_FR_GENERIC || dst_unit == NPY_FR_GENERIC) {
                return src_unit == NPY_FR_GENERIC;
            }
            else {
                return 1;
            }

        /*
         * Allow casting towards more precise datetime units with 'safe' casting,
         * and enforce a barrier between 'generic' and other units.
         */
        case NPY_SAFE_CASTING:
            if (src_unit == NPY_FR_GENERIC || dst_unit == NPY_FR_GENERIC) {
                return src_unit == NPY_FR_GENERIC;
            }
            else {
                return (src_unit <= dst_unit);
            }

        /* Enforce strict equality for 'no' or 'equiv' casting */
        default:
            return src_unit == dst_unit;
    }
}

/*
 * Provide casting rules for timedelta64 data type units.
 * Enforce barriers between 'date units' and 'time units' with specific casting modes.
 */
NPY_NO_EXPORT npy_bool
can_cast_timedelta64_units(NPY_DATETIMEUNIT src_unit,
                          NPY_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting)
{
    switch (casting) {
        /* Allow any casting without checking types */
        case NPY_UNSAFE_CASTING:
            return 1;

        /*
         * Enforce the 'date units' vs 'time units' barrier with 'same_kind' casting,
         * ensuring src_unit and dst_unit are either both date units or both time units.
         * If either unit is NPY_FR_GENERIC, only allow if src_unit is NPY_FR_GENERIC.
         */
        case NPY_SAME_KIND_CASTING:
            if (src_unit == NPY_FR_GENERIC || dst_unit == NPY_FR_GENERIC) {
                return src_unit == NPY_FR_GENERIC;
            }
            else {
                return (src_unit <= NPY_FR_M && dst_unit <= NPY_FR_M) ||
                       (src_unit > NPY_FR_M && dst_unit > NPY_FR_M);
            }

        /*
         * Enforce the 'date units' vs 'time units' barrier with 'safe' casting,
         * and allow casting towards more precise units. Also, ensure src_unit
         * and dst_unit follow the 'date units' vs 'time units' rule.
         */
        case NPY_SAFE_CASTING:
            if (src_unit == NPY_FR_GENERIC || dst_unit == NPY_FR_GENERIC) {
                return src_unit == NPY_FR_GENERIC;
            }
            else {
                return (src_unit <= dst_unit) &&
                       ((src_unit <= NPY_FR_M && dst_unit <= NPY_FR_M) ||
                        (src_unit > NPY_FR_M && dst_unit > NPY_FR_M));
            }

        /* Enforce strict equality for 'no' or 'equiv' casting */
        default:
            return src_unit == dst_unit;
    }
}
# 根据给定的转换规则检查是否可以安全地将 datetime64 数据类型从源元数据转换为目标元数据。
# 返回值为 True 表示可以转换，返回值为 False 表示不能转换或者出现异常。
NPY_NO_EXPORT int
raise_if_datetime64_metadata_cast_error(char *object_type,
                            PyArray_DatetimeMetaData *src_meta,
                            PyArray_DatetimeMetaData *dst_meta,
                            NPY_CASTING casting)
{
    # 调用 can_cast_datetime64_metadata 函数检查是否可以进行不同的转换类型
    if (can_cast_datetime64_metadata(src_meta, dst_meta, casting)) {
        # 如果可以转换，则返回 0 表示没有错误
        return 0;
    }
    else {
        # 如果不能转换，获取源元数据的描述字符串
        PyObject *src = metastr_to_unicode(src_meta, 0);
        if (src == NULL) {
            # 如果获取失败，返回 -1 表示异常
            return -1;
        }
        # 获取目标元数据的描述字符串
        PyObject *dst = metastr_to_unicode(dst_meta, 0);
        if (dst == NULL) {
            # 如果获取失败，释放源描述字符串，返回 -1 表示异常
            Py_DECREF(src);
            return -1;
        }
        # 抛出类型错误异常，说明无法按照指定规则从源到目标进行转换
        PyErr_Format(PyExc_TypeError,
            "Cannot cast %s from metadata %S to %S according to the rule %s",
            object_type, src, dst, npy_casting_to_string(casting));
        # 释放源和目标描述字符串
        Py_DECREF(src);
        Py_DECREF(dst);
        # 返回 -1 表示异常
        return -1;
    }
}
/*
 * Tests whether a timedelta64 can be cast from the source metadata
 * to the destination metadata according to the specified casting rule.
 *
 * Returns -1 if an exception was raised, 0 otherwise.
 */
NPY_NO_EXPORT int
raise_if_timedelta64_metadata_cast_error(char *object_type,
                            PyArray_DatetimeMetaData *src_meta,
                            PyArray_DatetimeMetaData *dst_meta,
                            NPY_CASTING casting)
{
    // 检查是否可以根据指定的转换规则从源元数据(src_meta)到目标元数据(dst_meta)进行timedelta64的转换
    if (can_cast_timedelta64_metadata(src_meta, dst_meta, casting)) {
        return 0; // 可以转换，返回成功状态码0
    }
    else {
        // 获取源元数据(src_meta)对应的Unicode字符串对象
        PyObject *src = metastr_to_unicode(src_meta, 0);
        if (src == NULL) {
            return -1; // 获取失败，返回异常状态码-1
        }
        // 获取目标元数据(dst_meta)对应的Unicode字符串对象
        PyObject *dst = metastr_to_unicode(dst_meta, 0);
        if (dst == NULL) {
            Py_DECREF(src);
            return -1; // 获取失败，返回异常状态码-1
        }
        // 抛出类型错误异常，指示无法根据给定规则将object_type从src到dst进行转换
        PyErr_Format(PyExc_TypeError,
             "Cannot cast %s from metadata %S to %S according to the rule %s",
             object_type, src, dst, npy_casting_to_string(casting));
        Py_DECREF(src);
        Py_DECREF(dst);
        return -1; // 返回异常状态码-1
    }
}

/*
 * Computes the GCD of the two date-time metadata values. Raises
 * an exception if there is no reasonable GCD, such as with
 * years and days.
 *
 * The result is placed in 'out_meta'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
compute_datetime_metadata_greatest_common_divisor(
                        PyArray_DatetimeMetaData *meta1,
                        PyArray_DatetimeMetaData *meta2,
                        PyArray_DatetimeMetaData *out_meta,
                        int strict_with_nonlinear_units1,
                        int strict_with_nonlinear_units2)
{
    NPY_DATETIMEUNIT base;
    npy_uint64 num1, num2, num;

    /* If either unit is generic, adopt the metadata from the other one */
    // 如果其中一个单位是通用的(NPY_FR_GENERIC)，则采用另一个单位的元数据
    if (meta1->base == NPY_FR_GENERIC) {
        *out_meta = *meta2;
        return 0; // 返回成功状态码0
    }
    else if (meta2->base == NPY_FR_GENERIC) {
        *out_meta = *meta1;
        return 0; // 返回成功状态码0
    }

    num1 = (npy_uint64)meta1->num;
    num2 = (npy_uint64)meta2->num;

    /* First validate that the units have a reasonable GCD */
    // 首先验证单位是否有合理的最大公约数(GCD)
    if (meta1->base == meta2->base) {
        base = meta1->base;
    }
    else {
        /*
         * Years and Months are incompatible with
         * all other units (except years and months are compatible
         * with each other).
         */
        如果条件不满足，执行以下代码块，处理单位不兼容情况
        if (meta1->base == NPY_FR_Y) {
            如果第一个元数据基础单位是年
            if (meta2->base == NPY_FR_M) {
                如果第二个元数据基础单位是月
                base = NPY_FR_M;
                num1 *= 12;
            }
            else if (strict_with_nonlinear_units1) {
                如果非线性单位严格模式为真，跳转到不兼容单位处理
                goto incompatible_units;
            }
            else {
                否则使用第二个元数据的基础单位
                base = meta2->base;
                /* 不乘以 num1，因为没有整数倍数 */
            }
        }
        else if (meta2->base == NPY_FR_Y) {
            如果第二个元数据基础单位是年
            if (meta1->base == NPY_FR_M) {
                如果第一个元数据基础单位是月
                base = NPY_FR_M;
                num2 *= 12;
            }
            else if (strict_with_nonlinear_units2) {
                如果非线性单位严格模式为真，跳转到不兼容单位处理
                goto incompatible_units;
            }
            else {
                否则使用第一个元数据的基础单位
                base = meta1->base;
                /* 不乘以 num2，因为没有整数倍数 */
            }
        }
        else if (meta1->base == NPY_FR_M) {
            如果第一个元数据基础单位是月
            if (strict_with_nonlinear_units1) {
                如果非线性单位严格模式为真，跳转到不兼容单位处理
                goto incompatible_units;
            }
            else {
                否则使用第二个元数据的基础单位
                base = meta2->base;
                /* 不乘以 num1，因为没有整数倍数 */
            }
        }
        else if (meta2->base == NPY_FR_M) {
            如果第二个元数据基础单位是月
            if (strict_with_nonlinear_units2) {
                如果非线性单位严格模式为真，跳转到不兼容单位处理
                goto incompatible_units;
            }
            else {
                否则使用第一个元数据的基础单位
                base = meta1->base;
                /* 不乘以 num2，因为没有整数倍数 */
            }
        }

        /* 取较大的基础单位（枚举中单位尺寸递减） */
        如果第一个元数据的基础单位大于第二个元数据的基础单位
        if (meta1->base > meta2->base) {
            base = meta1->base;
            num2 *= get_datetime_units_factor(meta2->base, meta1->base);
            如果 num2 等于 0
            如果结果乘以第二个元数据基础单位和第一个元数据基础单位之间的时间单位因子，跳转到单位溢出处理
            if (num2 == 0) {
                goto units_overflow;
            }
        }
        else {
            否则使用第二个元数据的基础单位
            base = meta2->base;
            num1 *= get_datetime_units_factor(meta1->base, meta2->base);
            如果 num1 等于 0
            如果结果乘以第一个元数据基础单位和第二个元数据基础单位之间的时间单位因子，跳转到单位溢出处理
            if (num1 == 0) {
                goto units_overflow;
            }
        }
    }

    /* 计算结果乘数的最大公约数 */
    num = _uint64_euclidean_gcd(num1, num2);

    /* 填充 'out_meta' 结构的值 */
    out_meta->base = base;
    out_meta->num = (int)num;
    如果 out_meta->num 小于等于 0 或者 num 不等于 (npy_uint64)out_meta->num
    如果结果不是有效的有符号整数或者是无符号 64 位整数的值，跳转到单位溢出处理
    if (out_meta->num <= 0 || num != (npy_uint64)out_meta->num) {
        goto units_overflow;
    }

    返回 0;

    /*
     * We do not use `DTypePromotionError` below.  The reason this is that a
     * `DTypePromotionError` indicates that `arr_dt1 != arr_dt2` for
     * all values, but this is wrong for "0".  This could be changed but
     * for now we consider them errors that occur _while_ promoting.
     */
incompatible_units: {
    // 将 meta1 转换为 Unicode 对象
    PyObject *umeta1 = metastr_to_unicode(meta1, 0);
    // 如果转换失败，返回 -1
    if (umeta1 == NULL) {
        return -1;
    }
    // 将 meta2 转换为 Unicode 对象
    PyObject *umeta2 = metastr_to_unicode(meta2, 0);
    // 如果转换失败，释放 umeta1 并返回 -1
    if (umeta2 == NULL) {
        Py_DECREF(umeta1);
        return -1;
    }
    // 抛出类型错误异常，说明 Numpy 日期时间元数据 umeta1 和 umeta2 的基本时间单位不兼容
    PyErr_Format(PyExc_TypeError,
        "Cannot get a common metadata divisor for Numpy datetime "
        "metadata %S and %S because they have incompatible nonlinear "
        "base time units.", umeta1, umeta2);
    // 释放 umeta1 和 umeta2
    Py_DECREF(umeta1);
    Py_DECREF(umeta2);
    // 返回 -1
    return -1;
}

units_overflow: {
    // 将 meta1 转换为 Unicode 对象
    PyObject *umeta1 = metastr_to_unicode(meta1, 0);
    // 如果转换失败，返回 -1
    if (umeta1 == NULL) {
        return -1;
    }
    // 将 meta2 转换为 Unicode 对象
    PyObject *umeta2 = metastr_to_unicode(meta2, 0);
    // 如果转换失败，释放 umeta1 并返回 -1
    if (umeta2 == NULL) {
        Py_DECREF(umeta1);
        return -1;
    }
    // 抛出溢出错误异常，说明获取 Numpy 日期时间元数据 umeta1 和 umeta2 的公共元数据除数时发生整数溢出
    PyErr_Format(PyExc_OverflowError,
        "Integer overflow getting a common metadata divisor for "
        "NumPy datetime metadata %S and %S.", umeta1, umeta2);
    // 释放 umeta1 和 umeta2
    Py_DECREF(umeta1);
    Py_DECREF(umeta2);
    // 返回 -1
    return -1;
}

/*
 * Both type1 and type2 must be either NPY_DATETIME or NPY_TIMEDELTA.
 * Applies the type promotion rules between the two types, returning
 * the promoted type.
 */
NPY_NO_EXPORT PyArray_Descr *
datetime_type_promotion(PyArray_Descr *type1, PyArray_Descr *type2)
{
    // 获取 type1 和 type2 的类型编号
    int type_num1, type_num2;
    type_num1 = type1->type_num;
    type_num2 = type2->type_num;

    // 判断 type1 或 type2 是否为日期时间类型
    int is_datetime;
    is_datetime = (type_num1 == NPY_DATETIME || type_num2 == NPY_DATETIME);

    /* Create a DATETIME or TIMEDELTA dtype */
    // 根据 is_datetime 的值创建 DATETIME 或 TIMEDELTA 类型的描述符
    PyArray_Descr *dtype;
    dtype = PyArray_DescrNewFromType(is_datetime ? NPY_DATETIME : NPY_TIMEDELTA);
    // 如果创建失败，返回空指针
    if (dtype == NULL) {
        return NULL;
    }

    /*
     * Get the metadata GCD, being strict about nonlinear units for
     * timedelta and relaxed for datetime.
     */
    // 计算日期时间元数据的最大公约数，对于 timedelta 类型严格对待非线性单位，对于 datetime 类型放宽要求
    if (compute_datetime_metadata_greatest_common_divisor(
                                get_datetime_metadata_from_dtype(type1),
                                get_datetime_metadata_from_dtype(type2),
                                get_datetime_metadata_from_dtype(dtype),
                                type_num1 == NPY_TIMEDELTA,
                                type_num2 == NPY_TIMEDELTA) < 0) {
        // 如果计算失败，释放 dtype 并返回空指针
        Py_DECREF(dtype);
        return NULL;
    }

    // 返回计算得到的描述符
    return dtype;
}

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit enum value. The 'metastr' parameter
 * is used for error messages, and may be NULL.
 *
 * Returns NPY_DATETIMEUNIT on success, NPY_FR_ERROR on failure.
 */
NPY_NO_EXPORT NPY_DATETIMEUNIT
parse_datetime_unit_from_string(char const *str, Py_ssize_t len, char const *metastr)
{
    /* Use switch statements so the compiler can make it fast */
    # 如果长度为1，检查单字符时间单位
    if (len == 1) {
        # 根据单字符时间单位返回对应的 NumPy 时间单位常量
        switch (str[0]) {
            case 'Y':
                return NPY_FR_Y;
            case 'M':
                return NPY_FR_M;
            case 'W':
                return NPY_FR_W;
            case 'D':
                return NPY_FR_D;
            case 'h':
                return NPY_FR_h;
            case 'm':
                return NPY_FR_m;
            case 's':
                return NPY_FR_s;
        }
    }
    # 如果长度为2，并且第二个字符为's'，检查两字符时间单位的变体
    /* All the two-letter units are variants of seconds */
    else if (len == 2 && str[1] == 's') {
        # 根据两字符时间单位的第一个字符返回对应的 NumPy 时间单位常量
        switch (str[0]) {
            case 'm':
                return NPY_FR_ms;
            case 'u':
                return NPY_FR_us;
            case 'n':
                return NPY_FR_ns;
            case 'p':
                return NPY_FR_ps;
            case 'f':
                return NPY_FR_fs;
            case 'a':
                return NPY_FR_as;
        }
    }
    # 如果长度为3，并且字符串以特定的 UTF-8 编码的字符串开头，返回微秒时间单位
    else if (len == 3 && !strncmp(str, "\xce\xbcs", 3)) {
        /* greek small letter mu, utf8-encoded */
        return NPY_FR_us;
    }
    # 如果长度为7，并且字符串为"generic"，返回通用时间单位
    else if (len == 7 && !strncmp(str, "generic", 7)) {
        return NPY_FR_GENERIC;
    }

    # 如果未匹配到任何条件，抛出类型错误异常，指示无效的时间单位字符串
    /* If nothing matched, it's an error */
    if (metastr == NULL) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime unit \"%s\" in metadata",
                str);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime unit in metadata string \"%s\"",
                metastr);
    }
    # 返回 NumPy 错误时间单位常量
    return NPY_FR_ERROR;
/*
 * 将 datetime 元数据转换为元组对象。
 *
 * 返回值：成功返回包含转换后元数据的元组对象，失败返回 NULL。
 */
NPY_NO_EXPORT PyObject *
convert_datetime_metadata_to_tuple(PyArray_DatetimeMetaData *meta)
{
    PyObject *dt_tuple; // 定义一个 PyObject 指针变量 dt_tuple

    dt_tuple = PyTuple_New(2); // 创建一个长度为 2 的元组对象 dt_tuple
    if (dt_tuple == NULL) { // 检查元组创建是否成功
        return NULL; // 失败则返回 NULL
    }

    PyTuple_SET_ITEM(dt_tuple, 0,
            PyUnicode_FromString(_datetime_strings[meta->base])); // 将 meta->base 对应的日期时间字符串转为 PyUnicode 对象，并设置为元组的第一个元素
    PyTuple_SET_ITEM(dt_tuple, 1,
            PyLong_FromLong(meta->num)); // 将 meta->num 转为 PyLong 对象，并设置为元组的第二个元素

    return dt_tuple; // 返回创建的元组对象
}

/*
 * 将元组对象转换为 datetime 元数据的 C 结构体。
 *
 * 参数：
 *   tuple: 待转换的元组对象。
 *   out_meta: 存储转换后元数据的 PyArray_DatetimeMetaData 结构体指针。
 *   from_pickle: 是否从 pickle 中加载。
 *
 * 返回值：成功返回 0，失败返回 -1。
 */
NPY_NO_EXPORT int
convert_datetime_metadata_tuple_to_datetime_metadata(PyObject *tuple,
                                        PyArray_DatetimeMetaData *out_meta,
                                        npy_bool from_pickle)
{
    int den = 1; // 初始化一个整型变量 den，值为 1

    if (!PyTuple_Check(tuple)) { // 检查 tuple 是否为元组对象
        PyErr_Format(PyExc_TypeError,
                "Require tuple for tuple to NumPy "
                "datetime metadata conversion, not %R", tuple); // 若不是，设置错误信息并返回 -1
        return -1;
    }

    Py_ssize_t tuple_size = PyTuple_GET_SIZE(tuple); // 获取元组的大小
    if (tuple_size < 2 || tuple_size > 4) { // 检查元组大小是否在 2 到 4 之间
        PyErr_SetString(PyExc_TypeError,
                        "Require tuple of size 2 to 4 for "
                        "tuple to NumPy datetime metadata conversion"); // 若不在范围内，设置错误信息并返回 -1
        return -1;
    }

    PyObject *unit_str = PyTuple_GET_ITEM(tuple, 0); // 获取元组的第一个元素作为 unit_str
    if (PyBytes_Check(unit_str)) { // 检查 unit_str 是否为字节对象
        /* Allow bytes format strings: convert to unicode */
        PyObject *tmp = PyUnicode_FromEncodedObject(unit_str, NULL, NULL); // 将字节对象转为 Unicode 对象
        if (tmp == NULL) {
            return -1; // 转换失败则返回 -1
        }
        unit_str = tmp; // 更新 unit_str 为转换后的对象
    }
    else {
        Py_INCREF(unit_str); // 增加对 unit_str 的引用计数
    }

    Py_ssize_t len;
    char const *basestr = PyUnicode_AsUTF8AndSize(unit_str, &len); // 将 Unicode 对象 unit_str 转为 UTF-8 编码的 C 字符串
    if (basestr == NULL) {
        Py_DECREF(unit_str); // 若转换失败，释放 unit_str 的引用
        return -1;
    }

    out_meta->base = parse_datetime_unit_from_string(basestr, len, NULL); // 解析 basestr 表示的日期时间单位，并存储到 out_meta->base 中
    if (out_meta->base == NPY_FR_ERROR) { // 检查解析是否失败
        Py_DECREF(unit_str); // 失败时释放 unit_str 的引用
        return -1;
    }

    Py_DECREF(unit_str); // 释放 unit_str 的引用

    /* Convert the values to longs */
    out_meta->num = PyLong_AsLong(PyTuple_GET_ITEM(tuple, 1)); // 将元组的第二个元素转为 long 类型，并存储到 out_meta->num 中
    if (error_converting(out_meta->num)) { // 检查转换是否出错
        return -1; // 出错则返回 -1
    }

    /*
     * The event metadata was removed way back in numpy 1.7 (cb4545), but was
     * not deprecated at the time.
     */

    /* (unit, num, event) */
    if (tuple_size == 3) { // 如果元组大小为 3
        /* Numpy 1.14, 2017-08-11 */
        if (DEPRECATE(
                "When passing a 3-tuple as (unit, num, event), the event "
                "is ignored (since 1.7) - use (unit, num) instead") < 0) { // 发出弃用警告信息
            return -1; // 若出错则返回 -1
        }
    }
    /* (unit, num, den, event) */
    else if (tuple_size == 4) {
        // 获取元组中索引为3的元素作为事件数据对象
        PyObject *event = PyTuple_GET_ITEM(tuple, 3);
        // 如果使用 pickle 加载，并且事件数据为 1
        if (from_pickle) {
            /* if (event == 1) */
            // 创建一个整数对象表示数字 1
            PyObject *one = PyLong_FromLong(1);
            // 检查对象创建是否成功
            if (one == NULL) {
                return -1;
            }
            // 比较事件对象与数字 1 是否相等
            int equal_one = PyObject_RichCompareBool(event, one, Py_EQ);
            // 释放整数对象
            Py_DECREF(one);
            // 检查比较操作是否出错
            if (equal_one == -1) {
                return -1;
            }

            /* if the event data is not 1, it had semantics different to how
             * datetime types now behave, which are no longer respected.
             */
            // 如果事件数据不等于 1，发出警告说明其语义与当前 datetime 类型的行为不同，并且已被忽略自 1.7 版本以来
            if (!equal_one) {
                // 发出用户警告，指出加载的 pickle 文件包含了非默认事件数据，已自 1.7 版本起忽略
                if (PyErr_WarnEx(PyExc_UserWarning,
                        "Loaded pickle file contains non-default event data "
                        "for a datetime type, which has been ignored since 1.7",
                        1) < 0) {
                    return -1;
                }
            }
        }
        // 如果不是使用 pickle 加载，并且事件数据不为空
        else if (event != Py_None) {
            /* Numpy 1.14, 2017-08-11 */
            // 发出警告，说明在传递 4 元组作为 (unit, num, den, event) 时，事件参数已自 1.7 版本起不再被使用，应设为 None
            if (DEPRECATE(
                    "When passing a 4-tuple as (unit, num, den, event), the "
                    "event argument is ignored (since 1.7), so should be None"
                    ) < 0) {
                return -1;
            }
        }
        // 将索引为2的元组元素转换为长整型，并赋给 den
        den = PyLong_AsLong(PyTuple_GET_ITEM(tuple, 2));
        // 检查转换过程是否出错
        if (error_converting(den)) {
            return -1;
        }
    }

    // 如果输出元数据的 num 小于等于 0，或者 den 小于等于 0
    if (out_meta->num <= 0 || den <= 0) {
        // 设置异常信息，说明元组值对于 NumPy datetime 元数据转换无效
        PyErr_SetString(PyExc_TypeError,
                        "Invalid tuple values for "
                        "tuple to NumPy datetime metadata conversion");
        return -1;
    }

    // 如果 den 不等于 1
    if (den != 1) {
        // 将 datetime 的除数转换为其倍数形式
        if (convert_datetime_divisor_to_multiple(out_meta, den, NULL) < 0) {
            return -1;
        }
    }

    // 返回操作成功的标志
    return 0;
/*
 * Adjusts a datetimestruct based on a seconds offset. Assumes
 * the current values are valid.
 */
NPY_NO_EXPORT void
/*
 * 给指定的日期时间结构体增加指定秒数。
 * 参数:
 *     dts: 指向日期时间结构体的指针
 *     seconds: 需要增加的秒数
 */
add_seconds_to_datetimestruct(npy_datetimestruct *dts, int seconds)
{
    int minutes;

    // 增加秒数到当前秒数
    dts->sec += seconds;
    // 提取出分钟数，并将其添加到日期时间结构体中
    minutes = extract_unit_32(&dts->sec, 60);
    add_minutes_to_datetimestruct(dts, minutes);
}

/*
 * 根据分钟偏移调整日期时间结构体。假定当前值有效。
 * 参数:
 *     dts: 指向日期时间结构体的指针
 *     minutes: 需要增加的分钟数
 */
NPY_NO_EXPORT void
add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes)
{
    int isleap;

    // 增加分钟到当前分钟数
    dts->min += minutes;

    /* 将无效的分钟传播到小时和日的变化中 */
    dts->hour += extract_unit_32(&dts->min,  60);
    dts->day  += extract_unit_32(&dts->hour, 24);

    /* 将无效的日传播到月和年的变化中 */
    if (dts->day < 1) {
        // 如果日数小于1，减少月数，如果月数小于1，减少年数，将月数设为12
        dts->month--;
        if (dts->month < 1) {
            dts->year--;
            dts->month = 12;
        }
        // 判断是否是闰年，并更新日数
        isleap = is_leapyear(dts->year);
        dts->day += _days_per_month_table[isleap][dts->month-1];
    }
    else if (dts->day > 28) {
        // 如果日数大于28，根据月份和是否是闰年更新日数
        isleap = is_leapyear(dts->year);
        if (dts->day > _days_per_month_table[isleap][dts->month-1]) {
            dts->day -= _days_per_month_table[isleap][dts->month-1];
            dts->month++;
            if (dts->month > 12) {
                dts->year++;
                dts->month = 1;
            }
        }
    }
}

/*NUMPY_API
 *
 * 测试并将Python datetime.datetime或datetime.date对象转换为NumPy npy_datetimestruct。
 *
 * 尽管C API有PyDate_*和PyDateTime_*函数，以下实现只需请求属性，因此支持datetime鸭子类型。
 * tzinfo时区转换将需要此样式的访问。
 *
 * 'out_bestunit'基于对象是否为datetime.date或datetime.datetime对象，给出建议的单位。
 *
 * 如果'apply_tzinfo'为1，则此函数使用tzinfo将其转换为UTC时间，否则返回具有本地时间的结构。
 *
 * 返回:
 *     -1：错误
 *      0：成功
 *      1：obj缺少所需的日期或日期时间属性，但没有设置错误。
 */
NPY_NO_EXPORT int
NpyDatetime_ConvertPyDateTimeToDatetimeStruct(
        PyObject *obj, npy_datetimestruct *out, NPY_DATETIMEUNIT *out_bestunit,
        int apply_tzinfo)
{
    PyObject *tmp;
    int isleap;

    /* 将输出初始化为全零 */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;

    /* 需要至少年/月/日属性 */
    if (!PyObject_HasAttrString(obj, "year") ||
            !PyObject_HasAttrString(obj, "month") ||
            !PyObject_HasAttrString(obj, "day")) {
        return 1;
    }

    /* 获取年份 */
    tmp = PyObject_GetAttrString(obj, "year");
    if (tmp == NULL) {
        return -1;
    }
    out->year = PyLong_AsLong(tmp);
    if (error_converting(out->year)) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* 获取月份 */
    tmp = PyObject_GetAttrString(obj, "month");
    if (tmp == NULL) {
        return -1;
    }
    out->month = PyLong_AsLong(tmp);
    if (error_converting(out->month)) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);
    /* Check if tmp pointer is NULL */
    if (tmp == NULL) {
        return -1;
    }
    /* Convert Python object tmp to a long integer and assign to out->month */
    out->month = PyLong_AsLong(tmp);
    /* Check for conversion error */
    if (error_converting(out->month)) {
        Py_DECREF(tmp);
        return -1;
    }
    /* Decrement reference count for tmp */
    Py_DECREF(tmp);

    /* Get the day attribute from Python object obj */
    tmp = PyObject_GetAttrString(obj, "day");
    /* Check if tmp pointer is NULL */
    if (tmp == NULL) {
        return -1;
    }
    /* Convert Python object tmp to a long integer and assign to out->day */
    out->day = PyLong_AsLong(tmp);
    /* Check for conversion error */
    if (error_converting(out->day)) {
        Py_DECREF(tmp);
        return -1;
    }
    /* Decrement reference count for tmp */
    Py_DECREF(tmp);

    /* Validate that the month and day are within valid ranges */
    if (out->month < 1 || out->month > 12) {
        goto invalid_date;
    }
    /* Determine if the year is a leap year */
    isleap = is_leapyear(out->year);
    /* Validate that the day is within the valid range for the month and year */
    if (out->day < 1 ||
                out->day > _days_per_month_table[isleap][out->month-1]) {
        goto invalid_date;
    }

    /* Check if time attributes (hour, minute, second, microsecond) exist */
    if (!PyObject_HasAttrString(obj, "hour") ||
            !PyObject_HasAttrString(obj, "minute") ||
            !PyObject_HasAttrString(obj, "second") ||
            !PyObject_HasAttrString(obj, "microsecond")) {
        /* Set the best unit for date if requested */
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_D;
        }
        /* Return success (0) indicating a valid date */
        return 0;
    }

    /* Get the hour attribute from Python object obj */
    tmp = PyObject_GetAttrString(obj, "hour");
    /* Check if tmp pointer is NULL */
    if (tmp == NULL) {
        return -1;
    }
    /* Convert Python object tmp to a long integer and assign to out->hour */
    out->hour = PyLong_AsLong(tmp);
    /* Check for conversion error */
    if (error_converting(out->hour)) {
        Py_DECREF(tmp);
        return -1;
    }
    /* Decrement reference count for tmp */
    Py_DECREF(tmp);

    /* Get the minute attribute from Python object obj */
    tmp = PyObject_GetAttrString(obj, "minute");
    /* Check if tmp pointer is NULL */
    if (tmp == NULL) {
        return -1;
    }
    /* Convert Python object tmp to a long integer and assign to out->min */
    out->min = PyLong_AsLong(tmp);
    /* Check for conversion error */
    if (error_converting(out->min)) {
        Py_DECREF(tmp);
        return -1;
    }
    /* Decrement reference count for tmp */
    Py_DECREF(tmp);

    /* Get the second attribute from Python object obj */
    tmp = PyObject_GetAttrString(obj, "second");
    /* Check if tmp pointer is NULL */
    if (tmp == NULL) {
        return -1;
    }
    /* Convert Python object tmp to a long integer and assign to out->sec */
    out->sec = PyLong_AsLong(tmp);
    /* Check for conversion error */
    if (error_converting(out->sec)) {
        Py_DECREF(tmp);
        return -1;
    }
    /* Decrement reference count for tmp */
    Py_DECREF(tmp);

    /* Get the microsecond attribute from Python object obj */
    tmp = PyObject_GetAttrString(obj, "microsecond");
    /* Check if tmp pointer is NULL */
    if (tmp == NULL) {
        return -1;
    }
    /* Convert Python object tmp to a long integer and assign to out->us */
    out->us = PyLong_AsLong(tmp);
    /* Check for conversion error */
    if (error_converting(out->us)) {
        Py_DECREF(tmp);
        return -1;
    }
    /* Decrement reference count for tmp */
    Py_DECREF(tmp);

    /* Validate that hour, minute, second, and microsecond are within valid ranges */
    if (out->hour < 0 || out->hour >= 24 ||
            out->min < 0 || out->min >= 60 ||
            out->sec < 0 || out->sec >= 60 ||
            out->us < 0 || out->us >= 1000000) {
        goto invalid_time;
    }

    /* Apply the time zone offset if it exists */
    # 如果需要应用时区信息，并且对象具有 "tzinfo" 属性
    if (apply_tzinfo && PyObject_HasAttrString(obj, "tzinfo")) {
        # 获取对象的 "tzinfo" 属性
        tmp = PyObject_GetAttrString(obj, "tzinfo");
        # 如果获取失败，返回错误
        if (tmp == NULL) {
            return -1;
        }
        # 如果 "tzinfo" 属性为 None，则释放 tmp 引用
        if (tmp == Py_None) {
            Py_DECREF(tmp);
        }
        else {
            PyObject *offset;
            int seconds_offset, minutes_offset;
            # 发出用户警告，指示 numpy.datetime64 没有显式的时区表示
            if (PyErr_WarnEx(PyExc_UserWarning,
                "no explicit representation of timezones available for np.datetime64",
                1) < 0) {
                    return -1;
                }

            /* 调用 "utcoffset" 方法获取时区偏移量 */
            offset = PyObject_CallMethod(tmp, "utcoffset", "O", obj);
            # 如果调用失败，释放 tmp 引用并返回错误
            if (offset == NULL) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /*
             * timedelta 对象应该有一个 "total_seconds" 方法
             * 返回我们需要的偏移量值
             */
            tmp = PyObject_CallMethod(offset, "total_seconds", "");
            Py_DECREF(offset);
            # 如果调用失败，返回错误
            if (tmp == NULL) {
                return -1;
            }
            /* 在这里进行四舍五入不会比下面的整数除法更糟糕。
             * numpy 只支持整分钟的偏移量。
             */
            seconds_offset = (int)PyFloat_AsDouble(tmp);
            # 如果转换错误，释放 tmp 引用并返回错误
            if (error_converting(seconds_offset)) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /* 转换为分钟偏移量并应用它 */
            minutes_offset = seconds_offset / 60;

            add_minutes_to_datetimestruct(out, -minutes_offset);
        }
    }

    /* Python datetime 的分辨率是 'us'（微秒） */
    if (out_bestunit != NULL) {
        *out_bestunit = NPY_FR_us;
    }

    # 返回成功
    return 0;
invalid_date:
    PyErr_Format(PyExc_ValueError,
            "Invalid date (%" NPY_INT64_FMT ",%" NPY_INT32_FMT ",%" NPY_INT32_FMT ") when converting to NumPy datetime",
            out->year, out->month, out->day);
    return -1;


# 在日期无效时，生成一个带有错误消息的异常对象，用于转换为 NumPy datetime
invalid_time:
    PyErr_Format(PyExc_ValueError,
            "Invalid time (%" NPY_INT32_FMT ",%" NPY_INT32_FMT ",%" NPY_INT32_FMT ",%" NPY_INT32_FMT ") when converting "
            "to NumPy datetime",
            out->hour, out->min, out->sec, out->us);
    return -1;
}


# 在时间无效时，生成一个带有错误消息的异常对象，用于转换为 NumPy datetime
/*
 * Gets a tzoffset in minutes by calling the fromutc() function on
 * the Python datetime.tzinfo object.
 */
NPY_NO_EXPORT int
get_tzoffset_from_pytzinfo(PyObject *timezone_obj, npy_datetimestruct *dts)
{
    PyObject *dt, *loc_dt;
    npy_datetimestruct loc_dts;

    /* 创建一个 Python datetime 对象用于传递给时区对象 */
    dt = PyDateTime_FromDateAndTime((int)dts->year, dts->month, dts->day,
                            dts->hour, dts->min, 0, 0);
    if (dt == NULL) {
        return -1;
    }

    /* 将 datetime 从 UTC 转换为本地时间 */
    loc_dt = PyObject_CallMethod(timezone_obj, "fromutc", "O", dt);
    Py_DECREF(dt);
    if (loc_dt == NULL) {
        return -1;
    }

    /* 将本地 datetime 转换为 datetimestruct 结构体 */
    if (NpyDatetime_ConvertPyDateTimeToDatetimeStruct(loc_dt, &loc_dts, NULL, 0) < 0) {
        Py_DECREF(loc_dt);
        return -1;
    }

    Py_DECREF(loc_dt);

    /* 计算时区偏移量，即两个 datetime 之间的差异 */
    return (int)(get_datetimestruct_minutes(&loc_dts) -
                 get_datetimestruct_minutes(dts));
}


# 通过调用 Python datetime.tzinfo 对象的 fromutc() 函数获取时区偏移量（以分钟表示）
/*
 * Converts a PyObject * into a datetime, in any of the forms supported.
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * The 'casting' parameter is used to control what kinds of inputs
 * are accepted, and what happens. For example, with 'unsafe' casting,
 * unrecognized inputs are converted to 'NaT' instead of throwing an error,
 * while with 'safe' casting an error will be thrown if any precision
 * from the input will be thrown away.
 *
 * Returns -1 on error, 0 on success.
 */
NPY_NO_EXPORT int
convert_pyobject_to_datetime(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                NPY_CASTING casting, npy_datetime *out)
{


# 将 PyObject * 转换为 datetime 对象，支持各种形式的输入。
#
# 如果单位元数据事先未知，请将 meta->base 设置为 -1，此函数将填充 meta
# 使用默认值或输入对象中的值。
#
# 'casting' 参数用于控制接受何种类型的输入以及相应的处理方式。例如，使用 'unsafe' 转换时，
# 无法识别的输入将转换为 'NaT' 而不是抛出错误；使用 'safe' 转换时，如果输入的任何精度将被丢弃，则会抛出错误。
#
# 返回 -1 表示出错，返回 0 表示成功。
    if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        PyObject *utf8 = NULL;

        /* Convert to an UTF8 string for the date parser */
        // 如果对象是字节字符串或 Unicode 字符串
        if (PyBytes_Check(obj)) {
            // 如果是字节字符串，将其转换为 UTF-8 编码的 Unicode 对象
            utf8 = PyUnicode_FromEncodedObject(obj, NULL, NULL);
            if (utf8 == NULL) {
                return -1;
            }
        }
        else {
            // 如果是 Unicode 字符串，直接使用
            utf8 = obj;
            Py_INCREF(utf8);
        }

        Py_ssize_t len = 0;
        char const *str = PyUnicode_AsUTF8AndSize(utf8, &len);
        if (str == NULL) {
            Py_DECREF(utf8);
            return -1;
        }

        /* Parse the ISO date */
        // 解析 ISO 日期
        npy_datetimestruct dts;
        NPY_DATETIMEUNIT bestunit = NPY_FR_ERROR;
        if (NpyDatetime_ParseISO8601Datetime(
                str, len, meta->base, casting,
                &dts, &bestunit, NULL) < 0) {
            Py_DECREF(utf8);
            return -1;
        }

        /* Use the detected unit if none was specified */
        // 如果没有指定基础单位，则使用检测到的最佳单位
        if (meta->base == NPY_FR_ERROR) {
            meta->base = bestunit;
            meta->num = 1;
        }

        // 将解析后的日期时间结构转换为 datetime64 类型
        if (NpyDatetime_ConvertDatetimeStructToDatetime64(meta, &dts, out) < 0) {
            Py_DECREF(utf8);
            return -1;
        }

        Py_DECREF(utf8);
        return 0;
    }
    /* Do no conversion on raw integers */
    // 如果是原始整数，不做转换
    else if (PyLong_Check(obj)) {
        /* Don't allow conversion from an integer without specifying a unit */
        // 如果没有指定单位，不允许从整数转换
        if (meta->base == NPY_FR_ERROR || meta->base == NPY_FR_GENERIC) {
            PyErr_SetString(PyExc_ValueError, "Converting an integer to a "
                            "NumPy datetime requires a specified unit");
            return -1;
        }
        // 将整数转换为长长整型
        *out = PyLong_AsLongLong(obj);
        if (error_converting(*out)) {
            return -1;
        }
        return 0;
    }
    /* Datetime scalar */
    // 如果是日期时间标量
    else if (PyArray_IsScalar(obj, Datetime)) {
        PyDatetimeScalarObject *dts = (PyDatetimeScalarObject *)obj;

        /* Copy the scalar directly if units weren't specified */
        // 如果没有指定单位，直接复制标量
        if (meta->base == NPY_FR_ERROR) {
            *meta = dts->obmeta;
            *out = dts->obval;

            return 0;
        }
        /* Otherwise do a casting transformation */
        // 否则进行类型转换
        else {
            /* Allow NaT (not-a-time) values to slip through any rule */
            // 允许 NaT (not-a-time) 值绕过任何规则
            if (dts->obval != NPY_DATETIME_NAT &&
                        raise_if_datetime64_metadata_cast_error(
                                "NumPy timedelta64 scalar",
                                &dts->obmeta, meta, casting) < 0) {
                return -1;
            }
            else {
                // 执行日期时间到日期时间转换
                return cast_datetime_to_datetime(&dts->obmeta, meta,
                                                    dts->obval, out);
            }
        }
    }
    /* Datetime zero-dimensional array */
    // 如果是零维数组的日期时间
    # 如果 obj 是一个 NumPy 数组，并且是零维的日期时间数组
    # 并且数组的数据类型是 NPY_DATETIME 类型
    else if (PyArray_Check(obj) &&
              PyArray_NDIM((PyArrayObject *)obj) == 0 &&
              PyArray_DESCR((PyArrayObject *)obj)->type_num == NPY_DATETIME) {
        # 将 obj 强制转换为 PyArrayObject 类型的数组对象
        PyArrayObject *arr = (PyArrayObject *)obj;
        # 创建一个指向日期时间元数据的指针
        PyArray_DatetimeMetaData *arr_meta;
        # 初始化日期时间变量 dt 为 0
        npy_datetime dt = 0;

        # 从数组的数据类型中获取日期时间的元数据
        arr_meta = get_datetime_metadata_from_dtype(PyArray_DESCR(arr));
        # 如果获取的元数据为空，则返回错误
        if (arr_meta == NULL) {
            return -1;
        }
        # 根据数组的数据类型描述，拷贝或交换数据到 dt 中
        PyDataType_GetArrFuncs(PyArray_DESCR(arr))->copyswap(&dt,
                                PyArray_DATA(arr),
                                PyArray_ISBYTESWAPPED(arr),
                                obj);

        /* 如果未指定单位，直接复制值 */
        if (meta->base == NPY_FR_ERROR) {
            *meta = *arr_meta;  # 复制数组的元数据到 meta 中
            *out = dt;  # 将 dt 赋值给 out

            return 0;  # 返回成功
        }
        /* 否则进行类型转换 */
        else {
            /* 允许 NaT（非时间）值通过任何规则 */
            if (dt != NPY_DATETIME_NAT &&
                        raise_if_datetime64_metadata_cast_error(
                                "NumPy timedelta64 scalar",
                                arr_meta, meta, casting) < 0) {
                return -1;  # 如果转换错误则返回错误
            }
            else {
                return cast_datetime_to_datetime(arr_meta, meta, dt, out);  # 否则进行日期时间的类型转换
            }
        }
    }
    /* 如果是从 Python 的日期或日期时间对象进行转换 */
    else {
        int code;  # 定义一个整型变量 code
        npy_datetimestruct dts;  # 定义一个日期时间结构体变量 dts
        NPY_DATETIMEUNIT bestunit = NPY_FR_ERROR;  # 初始化最佳单位为 NPY_FR_ERROR

        # 调用函数将 Python 的日期或日期时间对象转换为日期时间结构体
        code = NpyDatetime_ConvertPyDateTimeToDatetimeStruct(obj, &dts, &bestunit, 1);
        # 如果转换失败，则返回错误
        if (code == -1) {
            return -1;
        }
        # 如果转换成功
        else if (code == 0) {
            /* 如果没有指定单位，则使用检测到的最佳单位 */
            if (meta->base == NPY_FR_ERROR) {
                meta->base = bestunit;  # 将最佳单位赋值给 meta 的基础单位
                meta->num = 1;  # 设置 meta 的数量为 1
            }
            else {
                # 创建一个新的日期时间元数据对象
                PyArray_DatetimeMetaData obj_meta;
                obj_meta.base = bestunit;  # 设置新对象的基础单位为最佳单位
                obj_meta.num = 1;  # 设置新对象的数量为 1

                # 如果日期时间转换错误，则返回错误
                if (raise_if_datetime64_metadata_cast_error(
                                bestunit == NPY_FR_D ? "datetime.date object"
                                                 : "datetime.datetime object",
                                &obj_meta, meta, casting) < 0) {
                    return -1;
                }
            }

            # 调用函数将日期时间结构体转换为 datetime64 类型
            return NpyDatetime_ConvertDatetimeStructToDatetime64(meta, &dts, out);
        }
    }

    /*
     * 对于不安全的类型转换，将无法识别的对象转换为 NaT（非时间）
     * 对于相同类型的转换，将 None 转换为 NaT
     */
    if (casting == NPY_UNSAFE_CASTING ||
            (obj == Py_None && casting == NPY_SAME_KIND_CASTING)) {
        if (meta->base == NPY_FR_ERROR) {
            meta->base = NPY_FR_GENERIC;  # 将 meta 的基础单位设置为通用单位
            meta->num = 1;  # 设置 meta 的数量为 1
        }
        *out = NPY_DATETIME_NAT;  # 将 out 设置为 NaT（非时间）
        return 0;  # 返回成功
    }
    # 如果无法将对象转换为 NumPy 的日期时间类型，设置错误信息
    PyErr_SetString(PyExc_ValueError,
            "Could not convert object to NumPy datetime");
    # 返回错误码 -1
    return -1;
}

/*
 * Converts a PyObject * into a timedelta, in any of the forms supported
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * The 'casting' parameter is used to control what kinds of inputs
 * are accepted, and what happens. For example, with 'unsafe' casting,
 * unrecognized inputs are converted to 'NaT' instead of throwing an error,
 * while with 'safe' casting an error will be thrown if any precision
 * from the input will be thrown away.
 *
 * Returns -1 on error, 0 on success.
 */
NPY_NO_EXPORT int
convert_pyobject_to_timedelta(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                NPY_CASTING casting, npy_timedelta *out)
{
    // 如果对象是字节或者Unicode字符串
    if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        PyObject *utf8 = NULL;
        int succeeded = 0;

        /* Convert to an UTF8 string for the date parser */
        // 如果对象是字节字符串，将其转换为UTF8编码的Unicode字符串
        if (PyBytes_Check(obj)) {
            utf8 = PyUnicode_FromEncodedObject(obj, NULL, NULL);
            if (utf8 == NULL) {
                return -1;
            }
        }
        else {
            // 否则，使用原始的Unicode对象
            utf8 = obj;
            Py_INCREF(utf8);
        }

        Py_ssize_t len = 0;
        // 将Unicode字符串转换为UTF8编码，并获取其长度和内容
        char const *str = PyUnicode_AsUTF8AndSize(utf8, &len);
        if (str == NULL) {
            Py_DECREF(utf8);
            return -1;
        }

        /* Check for a NaT string */
        // 检查是否为NaT字符串
        if (len <= 0 || (len == 3 &&
                        tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'a' &&
                        tolower(str[2]) == 't')) {
            // 如果是NaT字符串，将输出设置为NPY_DATETIME_NAT
            *out = NPY_DATETIME_NAT;
            succeeded = 1;
        }
        /* Parse as an integer */
        else {
            // 否则尝试将字符串解析为整数
            char *strend = NULL;
            *out = strtol(str, &strend, 10);
            if (strend - str == len) {
                succeeded = 1;
            }
        }
        Py_DECREF(utf8);

        if (succeeded) {
            /* Use generic units if none was specified */
            // 如果没有指定时间单位，使用通用单位
            if (meta->base == NPY_FR_ERROR) {
                meta->base = NPY_FR_GENERIC;
                meta->num = 1;
            }

            return 0;
        }
    }
    /* Do no conversion on raw integers */
    // 如果对象是长整型
    else if (PyLong_Check(obj)) {
        /* Use the default unit if none was specified */
        // 如果没有指定时间单位，使用默认单位
        if (meta->base == NPY_FR_ERROR) {
            meta->base = NPY_DATETIME_DEFAULTUNIT;
            meta->num = 1;
        }

        // 将长整型对象转换为长长整型，并存储到输出中
        *out = PyLong_AsLongLong(obj);
        if (error_converting(*out)) {
            return -1;
        }
        return 0;
    }
    /* Timedelta scalar */
    else if (PyArray_IsScalar(obj, Timedelta)) {
        // 检查 obj 是否是一个 NumPy 时间间隔标量对象
        PyTimedeltaScalarObject *dts = (PyTimedeltaScalarObject *)obj;

        /* Copy the scalar directly if units weren't specified */
        // 如果没有指定单位，直接复制标量
        if (meta->base == NPY_FR_ERROR) {
            // 如果 meta 的基础时间单位是错误状态，则将其设置为 dts 的元数据并复制值到 out
            *meta = dts->obmeta;
            *out = dts->obval;

            return 0;
        }
        /* Otherwise do a casting transformation */
        // 否则进行类型转换
        else {
            /* Allow NaT (not-a-time) values to slip through any rule */
            // 允许 NaT (not-a-time) 值通过任何规则
            if (dts->obval != NPY_DATETIME_NAT &&
                        raise_if_timedelta64_metadata_cast_error(
                                "NumPy timedelta64 scalar",
                                &dts->obmeta, meta, casting) < 0) {
                return -1;
            }
            else {
                // 否则进行 timedelta 到 timedelta 的类型转换
                return cast_timedelta_to_timedelta(&dts->obmeta, meta,
                                                    dts->obval, out);
            }
        }
    }
    /* Timedelta zero-dimensional array */
    else if (PyArray_Check(obj) &&
             PyArray_NDIM((PyArrayObject *)obj) == 0 &&
             PyArray_DESCR((PyArrayObject *)obj)->type_num == NPY_TIMEDELTA) {
        // 检查 obj 是否是一个 NumPy 时间间隔零维数组
        PyArrayObject *arr = (PyArrayObject *)obj;
        PyArray_DatetimeMetaData *arr_meta;
        npy_timedelta dt = 0;

        // 从数组的 dtype 中获取日期时间元数据
        arr_meta = get_datetime_metadata_from_dtype(PyArray_DESCR(arr));
        if (arr_meta == NULL) {
            return -1;
        }
        // 使用数组的数据进行拷贝或交换
        PyDataType_GetArrFuncs(PyArray_DESCR(arr))->copyswap(&dt,
                                PyArray_DATA(arr),
                                PyArray_ISBYTESWAPPED(arr),
                                obj);

        /* Copy the value directly if units weren't specified */
        // 如果没有指定单位，直接复制值
        if (meta->base == NPY_FR_ERROR) {
            // 如果 meta 的基础时间单位是错误状态，则将其设置为数组的元数据并复制值到 out
            *meta = *arr_meta;
            *out = dt;

            return 0;
        }
        /* Otherwise do a casting transformation */
        // 否则进行类型转换
        else {
            /* Allow NaT (not-a-time) values to slip through any rule */
            // 允许 NaT (not-a-time) 值通过任何规则
            if (dt != NPY_DATETIME_NAT &&
                        raise_if_timedelta64_metadata_cast_error(
                                "NumPy timedelta64 scalar",
                                arr_meta, meta, casting) < 0) {
                return -1;
            }
            else {
                // 否则进行 timedelta 到 timedelta 的类型转换
                return cast_timedelta_to_timedelta(arr_meta, meta, dt, out);
            }
        }
    }
    /* Convert from a Python timedelta object */
    else if (PyObject_HasAttrString(obj, "days") &&
                PyObject_HasAttrString(obj, "seconds") &&
                PyObject_HasAttrString(obj, "microseconds")) {
        PyObject *tmp;
        PyArray_DatetimeMetaData us_meta;
        npy_timedelta td;
        npy_int64 days;
        int seconds = 0, useconds = 0;

        /* 获取对象的 'days' 属性 */
        tmp = PyObject_GetAttrString(obj, "days");
        if (tmp == NULL) {
            return -1;  // 如果获取失败，返回错误
        }
        days = PyLong_AsLongLong(tmp);  // 将 'days' 属性转换为长整型
        if (error_converting(days)) {  // 检查转换是否出错
            Py_DECREF(tmp);
            return -1;  // 如果出错，返回错误
        }
        Py_DECREF(tmp);  // 释放临时变量

        /* 获取对象的 'seconds' 属性 */
        tmp = PyObject_GetAttrString(obj, "seconds");
        if (tmp == NULL) {
            return -1;  // 如果获取失败，返回错误
        }
        seconds = PyLong_AsLong(tmp);  // 将 'seconds' 属性转换为整型
        if (error_converting(seconds)) {  // 检查转换是否出错
            Py_DECREF(tmp);
            return -1;  // 如果出错，返回错误
        }
        Py_DECREF(tmp);  // 释放临时变量

        /* 获取对象的 'microseconds' 属性 */
        tmp = PyObject_GetAttrString(obj, "microseconds");
        if (tmp == NULL) {
            return -1;  // 如果获取失败，返回错误
        }
        useconds = PyLong_AsLong(tmp);  // 将 'microseconds' 属性转换为整型
        if (error_converting(useconds)) {  // 检查转换是否出错
            Py_DECREF(tmp);
            return -1;  // 如果出错，返回错误
        }
        Py_DECREF(tmp);  // 释放临时变量

        // 计算总微秒数，包括天、秒和微秒
        td = days*(24*60*60*1000000LL) + seconds*1000000LL + useconds;

        /* 如果目标时间单位是错误（NPY_FR_ERROR），则使用微秒 */
        if (meta->base == NPY_FR_ERROR) {
            meta->base = NPY_FR_us;  // 设置时间单位为微秒
            meta->num = 1;  // 设置单位数为1

            *out = td;  // 将计算结果赋值给输出

            return 0;  // 返回成功
        }
        else {
            /*
             * 检测最大的时间单位，使得之后的所有值为零，
             * 例如，如果微秒为零，则安全地转换为秒。
             */
            if (td % 1000LL != 0) {
                us_meta.base = NPY_FR_us;  // 微秒
            }
            else if (td % 1000000LL != 0) {
                us_meta.base = NPY_FR_ms;  // 毫秒
            }
            else if (td % (60*1000000LL) != 0) {
                us_meta.base = NPY_FR_s;  // 秒
            }
            else if (td % (60*60*1000000LL) != 0) {
                us_meta.base = NPY_FR_m;  // 分钟
            }
            else if (td % (24*60*60*1000000LL) != 0) {
                us_meta.base = NPY_FR_h;  // 小时
            }
            else if (td % (7*24*60*60*1000000LL) != 0) {
                us_meta.base = NPY_FR_D;  // 天
            }
            else {
                us_meta.base = NPY_FR_W;  // 周
            }
            us_meta.num = 1;  // 设置单位数为1

            // 如果转换操作出错，返回错误；否则，进行类型转换
            if (raise_if_timedelta64_metadata_cast_error(
                                "datetime.timedelta object",
                                &us_meta, meta, casting) < 0) {
                return -1;  // 返回错误
            }
            else {
                /* 在转换操作中切换回微秒 */
                us_meta.base = NPY_FR_us;  // 设置时间单位为微秒

                // 执行时间间隔到时间间隔的转换
                return cast_timedelta_to_timedelta(&us_meta, meta, td, out);
            }
        }
    }
    """
    If the casting mode is unsafe or if obj is None and casting mode is same_kind,
    set meta->base to NPY_FR_GENERIC and meta->num to 1.
    Return NPY_DATETIME_NAT and 0.
    Otherwise, if obj is a scalar integer:
        - If meta->base is NPY_FR_ERROR, set it to NPY_DATETIME_DEFAULTUNIT and meta->num to 1.
        - Convert obj to a long long integer and assign it to *out.
        - If there's an error during conversion, return -1.
        Return 0.
    If neither condition is met, raise a ValueError with a message indicating failure
    to convert object to NumPy timedelta and return -1.
    """
    if (casting == NPY_UNSAFE_CASTING or
            (obj == Py_None and casting == NPY_SAME_KIND_CASTING)):
        if (meta->base == NPY_FR_ERROR):
            meta->base = NPY_FR_GENERIC;
            meta->num = 1;
        *out = NPY_DATETIME_NAT;
        return 0;
    else if (PyArray_IsScalar(obj, Integer)):
        if (meta->base == NPY_FR_ERROR):
            meta->base = NPY_DATETIME_DEFAULTUNIT;
            meta->num = 1;
        *out = PyLong_AsLongLong(obj);
        if (error_converting(*out)):
            return -1;
        return 0;
    else {
        PyErr_SetString(PyExc_ValueError,
                "Could not convert object to NumPy timedelta");
        return -1;
    }
/*
 * Converts a datetime into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For days or coarser, returns a datetime.date.
 * For microseconds or coarser, returns a datetime.datetime.
 * For units finer than microseconds, returns an integer.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_to_pyobject(npy_datetime dt, PyArray_DatetimeMetaData *meta)
{
    PyObject *ret = NULL;
    npy_datetimestruct dts;

    /*
     * Convert NaT (not-a-time) and any value with generic units
     * into None.
     */
    if (dt == NPY_DATETIME_NAT || meta->base == NPY_FR_GENERIC) {
        Py_RETURN_NONE;
    }

    /* If the type's precision is greater than microseconds, return an int */
    if (meta->base > NPY_FR_us) {
        return PyLong_FromLongLong(dt);
    }

    /* Convert to a datetimestruct */
    if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(meta, dt, &dts) < 0) {
        return NULL;
    }

    /*
     * If the year is outside the range of years supported by Python's
     * datetime, or the datetime64 falls on a leap second,
     * return a raw int.
     */
    if (dts.year < 1 || dts.year > 9999 || dts.sec == 60) {
        return PyLong_FromLongLong(dt);
    }

    /* If the type's precision is greater than days, return a datetime */
    if (meta->base > NPY_FR_D) {
        ret = PyDateTime_FromDateAndTime(dts.year, dts.month, dts.day,
                                dts.hour, dts.min, dts.sec, dts.us);
    }
    /* Otherwise return a date */
    else {
        ret = PyDate_FromDate(dts.year, dts.month, dts.day);
    }

    return ret;
}

/*
 * Converts a timedelta into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For microseconds or coarser, returns a datetime.timedelta.
 * For units finer than microseconds, returns an integer.
 */
NPY_NO_EXPORT PyObject *
convert_timedelta_to_pyobject(npy_timedelta td, PyArray_DatetimeMetaData *meta)
{
    npy_timedelta value;
    int days = 0, seconds = 0, useconds = 0;

    /*
     * Convert NaT (not-a-time) into None.
     */
    if (td == NPY_DATETIME_NAT) {
        Py_RETURN_NONE;
    }

    /*
     * If the type's precision is greater than microseconds, is
     * Y/M/B (nonlinear units), or is generic units, return an int
     */
    if (meta->base > NPY_FR_us ||
                    meta->base == NPY_FR_Y ||
                    meta->base == NPY_FR_M ||
                    meta->base == NPY_FR_GENERIC) {
        return PyLong_FromLongLong(td);
    }

    value = td;

    /* Apply the unit multiplier (TODO: overflow treatment...) */
    value *= meta->num;

    /* Convert to days/seconds/useconds */
    # 根据 meta->base 的值选择不同的时间单位转换方式
    switch (meta->base) {
        # 如果基本时间单位是周，将值转换为天数
        case NPY_FR_W:
            days = value * 7;
            break;
        # 如果基本时间单位是天，直接使用给定值作为天数
        case NPY_FR_D:
            days = value;
            break;
        # 如果基本时间单位是小时，将值转换为天数并计算剩余的秒数
        case NPY_FR_h:
            days = extract_unit_64(&value, 24ULL);
            seconds = value * 60 * 60;
            break;
        # 如果基本时间单位是分钟，将值转换为天数并计算剩余的秒数
        case NPY_FR_m:
            days = extract_unit_64(&value, 60ULL * 24);
            seconds = value * 60;
            break;
        # 如果基本时间单位是秒，将值转换为天数并计算剩余的秒数
        case NPY_FR_s:
            days = extract_unit_64(&value, 60ULL * 60 * 24);
            seconds = value;
            break;
        # 如果基本时间单位是毫秒，将值转换为天数、秒数和微秒数
        case NPY_FR_ms:
            days = extract_unit_64(&value, 1000ULL * 60 * 60 * 24);
            seconds = extract_unit_64(&value, 1000ULL);
            useconds = value * 1000;
            break;
        # 如果基本时间单位是微秒，将值转换为天数、秒数和微秒数
        case NPY_FR_us:
            days = extract_unit_64(&value, 1000ULL * 1000 * 60 * 60 * 24);
            seconds = extract_unit_64(&value, 1000ULL * 1000);
            useconds = value;
            break;
        default:
            // 不可能到达的分支，由上面的 `if` 处理
            assert(NPY_FALSE);
            break;
    }
    /*
     * 如果天数超出 datetime.timedelta 的范围，则返回一个原始整数
     */
    if (days < -999999999 || days > 999999999) {
        return PyLong_FromLongLong(td);
    }
    else {
        // 使用天数、秒数和微秒数创建一个 Python timedelta 对象
        return PyDelta_FromDSU(days, seconds, useconds);
    }
/*
 * Returns true if the datetime metadata matches
 */
NPY_NO_EXPORT npy_bool
has_equivalent_datetime_metadata(PyArray_Descr *type1, PyArray_Descr *type2)
{
    PyArray_DatetimeMetaData *meta1, *meta2;

    // 检查type1和type2是否为datetime或timedelta类型
    if ((type1->type_num != NPY_DATETIME &&
                        type1->type_num != NPY_TIMEDELTA) ||
                    (type2->type_num != NPY_DATETIME &&
                        type2->type_num != NPY_TIMEDELTA)) {
        return 0;
    }

    // 获取type1的datetime元数据
    meta1 = get_datetime_metadata_from_dtype(type1);
    if (meta1 == NULL) {
        PyErr_Clear();
        return 0;
    }

    // 获取type2的datetime元数据
    meta2 = get_datetime_metadata_from_dtype(type2);
    if (meta2 == NULL) {
        PyErr_Clear();
        return 0;
    }

    /* 对于通用单位，忽略num字段 */
    // 如果meta1和meta2都是通用单位，则认为元数据匹配
    if (meta1->base == NPY_FR_GENERIC && meta2->base == NPY_FR_GENERIC) {
        return 1;
    }

    // 返回比较基本单位和数值是否相等的结果
    return meta1->base == meta2->base &&
            meta1->num == meta2->num;
}

/*
 * Casts a single datetime from having src_meta metadata into
 * dst_meta metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
cast_datetime_to_datetime(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_datetime src_dt,
                          npy_datetime *dst_dt)
{
    npy_datetimestruct dts;

    /* 如果元数据相同，直接复制源datetime到目标datetime */
    if (src_meta->base == dst_meta->base &&
            src_meta->num == dst_meta->num) {
        *dst_dt = src_dt;
        return 0;
    }

    /* 否则，通过datetimestruct进行转换 */
    // 将源datetime转换为datetimestruct
    if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(src_meta, src_dt, &dts) < 0) {
            *dst_dt = NPY_DATETIME_NAT;
            return -1;
    }
    // 将datetimestruct转换为目标datetime
    if (NpyDatetime_ConvertDatetimeStructToDatetime64(dst_meta, &dts, dst_dt) < 0) {
        *dst_dt = NPY_DATETIME_NAT;
        return -1;
    }

    return 0;
}

/*
 * Casts a single timedelta from having src_meta metadata into
 * dst_meta metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
cast_timedelta_to_timedelta(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_timedelta src_dt,
                          npy_timedelta *dst_dt)
{
    npy_int64 num = 0, denom = 0;

    /* 如果元数据相同，直接复制源timedelta到目标timedelta */
    if (src_meta->base == dst_meta->base &&
            src_meta->num == dst_meta->num) {
        *dst_dt = src_dt;
        return 0;
    }

    /* 获取转换系数 */
    // 获取源和目标单位的转换系数
    get_datetime_conversion_factor(src_meta, dst_meta, &num, &denom);

    // 如果转换系数为0，则转换失败
    if (num == 0) {
        return -1;
    }

    /* 应用比例 */
    // 根据转换系数对源timedelta进行缩放
    if (src_dt < 0) {
        *dst_dt = (src_dt * num - (denom - 1)) / denom;
    }
    else {
        *dst_dt = src_dt * num / denom;
    }

    return 0;
}
    /* 检查传入的对象是否是任何一种 NumPy 的日期时间对象 */
    static NPY_GCC_NONNULL(1) npy_bool
    is_any_numpy_datetime(PyObject *obj)
    {
        return (PyArray_IsScalar(obj, Datetime) ||
                (PyArray_Check(obj) && (
                    PyArray_DESCR((PyArrayObject *)obj)->type_num ==
                                                            NPY_DATETIME)) ||
                PyDate_Check(obj) ||
                PyDateTime_Check(obj));
    }

    /*
     * 如果对象最适合考虑为 Timedelta，则返回 true，否则返回 false。
     */
    static npy_bool
    is_any_numpy_timedelta(PyObject *obj)
    {
        return (PyArray_IsScalar(obj, Timedelta) ||
            (PyArray_Check(obj) && (
                PyArray_DESCR((PyArrayObject *)obj)->type_num == NPY_TIMEDELTA)) ||
            PyDelta_Check(obj));
    }

    /*
     * 如果对象最适合考虑为 Datetime 或 Timedelta，则返回 true，否则返回 false。
     */
    NPY_NO_EXPORT npy_bool
    is_any_numpy_datetime_or_timedelta(PyObject *obj)
    {
        return obj != NULL &&
               (is_any_numpy_datetime(obj) ||
                is_any_numpy_timedelta(obj));
    }

    /*
     * 根据 type_nums 中的值，将 PyObject * 数组转换为日期时间和/或时间增量，
     * 基于提供的 'inout_meta' 进行所有转换。
     *
     * 如果 inout_meta->base 为 -1，则使用 GCD（最大公约数）计算元数据，并将
     * 'inout_meta' 填充为计算结果。否则，使用提供的 'inout_meta' 进行转换。
     *
     * 当 obj[i] 为 NULL 时，out_value[i] 将被设置为 NPY_DATETIME_NAT。
     *
     * 成功返回 0，失败返回 -1。
     */
    NPY_NO_EXPORT int
    convert_pyobjects_to_datetimes(int count,
                                   PyObject **objs, const int *type_nums,
                                   NPY_CASTING casting,
                                   npy_int64 *out_values,
                                   PyArray_DatetimeMetaData *inout_meta)
    {
        int i, is_out_strict;
        PyArray_DatetimeMetaData *meta;

        /* 如果没有值，认为是成功 */
        if (count == 0) {
            return 0;
        }

        /* 使用输入解析单位元数据（如果请求） */
        
    # 如果基础类型为 NPY_FR_ERROR，则执行以下操作
    if (inout_meta->base == NPY_FR_ERROR) {
        /* 分配与对象数量对应的元数据数组 */
        meta = PyArray_malloc(count * sizeof(PyArray_DatetimeMetaData));
        // 分配内存失败处理
        if (meta == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        /* 将所有对象转换为时间增量或日期时间 */
        for (i = 0; i < count; ++i) {
            // 初始化元数据
            meta[i].base = NPY_FR_ERROR;
            meta[i].num = 1;

            /* NULL -> NaT */
            // 对象为空时，设置输出值为 NPY_DATETIME_NAT，并将基础类型设置为 NPY_FR_GENERIC
            if (objs[i] == NULL) {
                out_values[i] = NPY_DATETIME_NAT;
                meta[i].base = NPY_FR_GENERIC;
            }
            // 对象为日期时间类型时
            else if (type_nums[i] == NPY_DATETIME) {
                // 尝试将 Python 对象转换为日期时间类型
                if (convert_pyobject_to_datetime(&meta[i], objs[i],
                                            casting, &out_values[i]) < 0) {
                    PyArray_free(meta);
                    return -1;
                }
            }
            // 对象为时间增量类型时
            else if (type_nums[i] == NPY_TIMEDELTA) {
                // 尝试将 Python 对象转换为时间增量类型
                if (convert_pyobject_to_timedelta(&meta[i], objs[i],
                                            casting, &out_values[i]) < 0) {
                    PyArray_free(meta);
                    return -1;
                }
            }
            // 类型不匹配时，抛出值错误异常
            else {
                PyErr_SetString(PyExc_ValueError,
                        "convert_pyobjects_to_datetimes requires that "
                        "all the type_nums provided be datetime or timedelta");
                PyArray_free(meta);
                return -1;
            }
        }

        /* 合并所有的元数据，从第一个元素开始 */
        // 将第一个元素的元数据赋值给输入输出元数据
        *inout_meta = meta[0];
        // 根据第一个元素的类型设置输出是否严格
        is_out_strict = (type_nums[0] == NPY_TIMEDELTA);

        // 从第二个元素开始，逐个合并元数据
        for (i = 1; i < count; ++i) {
            // 计算最大公约数并更新输入输出元数据
            if (compute_datetime_metadata_greatest_common_divisor(
                                    &meta[i], inout_meta, inout_meta,
                                    type_nums[i] == NPY_TIMEDELTA,
                                    is_out_strict) < 0) {
                PyArray_free(meta);
                return -1;
            }
            // 更新是否输出严格标志
            is_out_strict = is_out_strict || (type_nums[i] == NPY_TIMEDELTA);
        }

        /* 将所有值转换为解析后的单位元数据 */
        // 遍历所有值，根据类型进行相应的转换
        for (i = 0; i < count; ++i) {
            // 如果是日期时间类型，则进行日期时间到日期时间的转换
            if (type_nums[i] == NPY_DATETIME) {
                if (cast_datetime_to_datetime(&meta[i], inout_meta,
                                         out_values[i], &out_values[i]) < 0) {
                    PyArray_free(meta);
                    return -1;
                }
            }
            // 如果是时间增量类型，则进行时间增量到时间增量的转换
            else if (type_nums[i] == NPY_TIMEDELTA) {
                if (cast_timedelta_to_timedelta(&meta[i], inout_meta,
                                         out_values[i], &out_values[i]) < 0) {
                    PyArray_free(meta);
                    return -1;
                }
            }
        }

        // 释放分配的元数据内存
        PyArray_free(meta);
    }
    /* 否则，转换为提供的单位元数据 */
    else {
        /* 如果条件不满足，则执行以下代码块 */

        /* 将所有对象转换为timedelta或datetime */
        for (i = 0; i < count; ++i) {
            /* 如果对象为NULL，则将其转换为NaT（Not a Time） */
            if (objs[i] == NULL) {
                out_values[i] = NPY_DATETIME_NAT;
            }
            /* 如果类型为NPY_DATETIME，则调用函数将Python对象转换为datetime */
            else if (type_nums[i] == NPY_DATETIME) {
                if (convert_pyobject_to_datetime(inout_meta, objs[i],
                                            casting, &out_values[i]) < 0) {
                    return -1;
                }
            }
            /* 如果类型为NPY_TIMEDELTA，则调用函数将Python对象转换为timedelta */
            else if (type_nums[i] == NPY_TIMEDELTA) {
                if (convert_pyobject_to_timedelta(inout_meta, objs[i],
                                            casting, &out_values[i]) < 0) {
                    return -1;
                }
            }
            /* 如果类型既不是datetime也不是timedelta，则设置错误并返回-1 */
            else {
                PyErr_SetString(PyExc_ValueError,
                        "convert_pyobjects_to_datetimes requires that "
                        "all the type_nums provided be datetime or timedelta");
                return -1;
            }
        }
    }

    /* 成功执行转换，返回0 */
    return 0;
    /*
`
}

NPY_NO_EXPORT PyArrayObject *
datetime_arange(PyObject *start, PyObject *stop, PyObject *step,
                PyArray_Descr *dtype)
{
    /*
     * First normalize the input parameters so there is no Py_None,
     * and start is moved to stop if stop is unspecified.
     */
    
    // 如果 step 是 Py_None，则将其设为 NULL
    if (step == Py_None) {
        step = NULL;
    }
    
    // 如果 stop 为 NULL 或者 Py_None，则将 stop 设为 start，将 start 设为 NULL
    if (stop == NULL || stop == Py_None) {
        stop = start;
        start = NULL;
        /* If start was NULL or None, raise an exception */
        // 如果 stop 仍然为 NULL 或者 Py_None，则抛出数值错误异常
        if (stop == NULL || stop == Py_None) {
            PyErr_SetString(PyExc_ValueError,
                    "arange needs at least a stopping value");
            return NULL;
        }
    }
    
    // 如果 start 是 Py_None，则将其设为 NULL
    if (start == Py_None) {
        start = NULL;
    }

    /* Step must not be a Datetime */
    // 如果 step 不为空并且是任何 NumPy datetime 类型，则抛出数值错误异常
    if (step != NULL && is_any_numpy_datetime(step)) {
        PyErr_SetString(PyExc_ValueError,
                    "cannot use a datetime as a step in arange");
        return NULL;
    }

    /* Check if the units of the given dtype are generic, in which
     * case we use the code path that detects the units
     */
    
    // 定义一个类型数组和日期时间元数据结构
    int type_nums[3];
    PyArray_DatetimeMetaData meta;
    
    // 如果 dtype 不为空
    if (dtype != NULL) {
        PyArray_DatetimeMetaData *meta_tmp;

        // 获取 dtype 的类型编号
        type_nums[0] = dtype->type_num;
        
        // 如果 dtype 不是 NPY_DATETIME 也不是 NPY_TIMEDELTA，则抛出数值错误异常
        if (type_nums[0] != NPY_DATETIME && type_nums[0] != NPY_TIMEDELTA) {
            PyErr_SetString(PyExc_ValueError,
                        "datetime_arange was given a non-datetime dtype");
            return NULL;
        }

        // 获取 dtype 的日期时间元数据
        meta_tmp = get_datetime_metadata_from_dtype(dtype);
        if (meta_tmp == NULL) {
            return NULL;
        }

        /*
         * If the dtype specified is in generic units, detect the
         * units from the input parameters.
         */
        
        // 如果 dtype 指定的是通用单位，则将 dtype 设为 NULL，meta.base 设为 NPY_FR_ERROR
        if (meta_tmp->base == NPY_FR_GENERIC) {
            dtype = NULL;
            meta.base = NPY_FR_ERROR;
        }
        /* Otherwise use the provided metadata */
        // 否则使用提供的元数据
        else {
            meta = *meta_tmp;
        }
    }
    else {
        // 如果 start 不为空并且是任何 NumPy datetime 类型，或者 stop 是任何 NumPy datetime 类型，则 type_nums[0] 设
    else {
        // 如果第二个参数是 PyLong 或者是整数类型的 NumPy 标量，或者是 NumPy 时间增量类型
        // 则将第二个参数的类型设置为 NPY_TIMEDELTA
        if (PyLong_Check(objs[1]) ||
                        PyArray_IsScalar(objs[1], Integer) ||
                        is_any_numpy_timedelta(objs[1])) {
            type_nums[1] = NPY_TIMEDELTA;
        }
        else {
            // 否则将第二个参数的类型设置为 NPY_DATETIME
            type_nums[1] = NPY_DATETIME;
        }
        // 第三个参数的类型始终为 NPY_TIMEDELTA
        type_nums[2] = NPY_TIMEDELTA;
    }

    /* Convert all the arguments
     *
     * Both datetime and timedelta are stored as int64, so they can
     * share value variables.
     */
    // 定义一个 int64 类型的数组 values，用于存储转换后的参数值
    npy_int64 values[3];
    // 将 Python 对象转换为 datetime 或 timedelta，并存储在 values 数组中
    if (convert_pyobjects_to_datetimes(3, objs, type_nums,
                                NPY_SAME_KIND_CASTING, values, &meta) < 0) {
        return NULL;
    }
    /* If no start was provided, default to 0 */
    // 如果未提供起始值 start，则默认为 0
    if (start == NULL) {
        // 确保第一个参数的类型是 NPY_TIMEDELTA
        assert(type_nums[0] == NPY_TIMEDELTA);
        values[0] = 0;
    }

    /* If no step was provided, default to 1 */
    // 如果未提供步长值 step，则默认为 1
    if (step == NULL) {
        values[2] = 1;
    }

    /*
     * In the case of arange(datetime, timedelta), convert
     * the timedelta into a datetime by adding the start datetime.
     */
    // 如果参数类型为 arange(datetime, timedelta)，则将 timedelta 转换为 datetime，通过添加起始时间来实现
    if (type_nums[0] == NPY_DATETIME && type_nums[1] == NPY_TIMEDELTA) {
        values[1] += values[0];
    }

    /* Now start, stop, and step have their values and matching metadata */
    // 现在 start、stop 和 step 都已经具有了它们的值和匹配的元数据

    // 如果任一参数的值为 NPY_DATETIME_NAT，表示无效的 datetime 值
    if (values[0] == NPY_DATETIME_NAT ||
                    values[1] == NPY_DATETIME_NAT ||
                    values[2] == NPY_DATETIME_NAT) {
        // 抛出值错误，提示不能使用 NaT（非时间）的 datetime 值
        PyErr_SetString(PyExc_ValueError,
                    "arange: cannot use NaT (not-a-time) datetime values");
        return NULL;
    }

    /* Calculate the array length */
    // 计算数组的长度
    npy_intp length;
    if (values[2] > 0 && values[1] > values[0]) {
        length = (values[1] - values[0] + (values[2] - 1)) / values[2];
    }
    else if (values[2] < 0 && values[1] < values[0]) {
        length = (values[1] - values[0] + (values[2] + 1)) / values[2];
    }
    else if (values[2] != 0) {
        length = 0;
    }
    else {
        // 如果步长为零，则抛出值错误，提示步长不能为零
        PyErr_SetString(PyExc_ValueError,
                    "arange: step cannot be zero");
        return NULL;
    }

    /* Create the dtype of the result */
    // 创建结果数组的数据类型
    if (dtype != NULL) {
        Py_INCREF(dtype);
    }
    else {
        // 如果未提供数据类型 dtype，则根据 type_nums[0] 和 meta 创建 datetime 的数据类型
        dtype = create_datetime_dtype(type_nums[0], &meta);
        if (dtype == NULL) {
            return NULL;
        }
    }

    /* Create the result array */
    // 创建结果数组 ret
    PyArrayObject *ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, dtype, 1, &length, NULL,
            NULL, 0, NULL);

    if (ret == NULL) {
        return NULL;
    }

    if (length > 0) {
        /* Extract the data pointer */
        // 提取结果数组的数据指针 ret_data
        npy_int64 *ret_data = (npy_int64 *)PyArray_DATA(ret);

        /* Create the timedeltas or datetimes */
        // 根据 values 数组的值创建 timedeltas 或 datetimes
        for (npy_intp i = 0; i < length; ++i) {
            *ret_data = values[0];
            values[0] += values[2];
            ret_data++;
        }
    }

    return ret;
/*
 * Examines all the strings in the given string array, and parses them
 * to find the right metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
find_string_array_datetime64_type(PyArrayObject *arr,
                        PyArray_DatetimeMetaData *meta)
{
    NpyIter* iter;                      // 迭代器对象
    NpyIter_IterNextFunc *iternext;     // 迭代器下一步函数指针
    char **dataptr;                     // 数据指针数组
    npy_intp *strideptr, *innersizeptr; // 内部步幅和内部循环大小指针
    PyArray_Descr *string_dtype;        // 字符串类型描述符指针
    int maxlen;                         // 最大长度
    char *tmp_buffer = NULL;            // 临时缓冲区指针

    npy_datetimestruct dts;             // 数据时间结构体
    PyArray_DatetimeMetaData tmp_meta;  // 临时元数据对象

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(arr) == 0) {       // 处理数组大小为零的情况
        return 0;                       // 返回成功
    }

    string_dtype = PyArray_DescrFromType(NPY_STRING);  // 获取字符串类型的描述符
    if (string_dtype == NULL) {         // 如果描述符获取失败
        return -1;                      // 返回失败
    }

    /* Use unsafe casting to allow unicode -> ascii string */
    iter = NpyIter_New((PyArrayObject *)arr,
                            NPY_ITER_READONLY|
                            NPY_ITER_EXTERNAL_LOOP|
                            NPY_ITER_BUFFERED,
                        NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                        string_dtype);  // 创建迭代器对象，以读取数组内容为字符串
    Py_DECREF(string_dtype);            // 释放字符串类型描述符的引用
    if (iter == NULL) {                 // 如果迭代器创建失败
        return -1;                      // 返回失败
    }

    iternext = NpyIter_GetIterNext(iter, NULL);  // 获取迭代器的下一步函数
    if (iternext == NULL) {             // 如果获取下一步函数失败
        NpyIter_Deallocate(iter);       // 释放迭代器对象
        return -1;                      // 返回失败
    }
    dataptr = NpyIter_GetDataPtrArray(iter);      // 获取数据指针数组
    strideptr = NpyIter_GetInnerStrideArray(iter); // 获取内部步幅数组
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter); // 获取内部循环大小指针

    /* Get the resulting string length */
    maxlen = NpyIter_GetDescrArray(iter)[0]->elsize; // 获取字符串的最大长度

    /* Allocate a buffer for strings which fill the buffer completely */
    tmp_buffer = PyArray_malloc(maxlen+1);   // 为填充完整缓冲区的字符串分配缓冲区空间
    if (tmp_buffer == NULL) {               // 如果分配缓冲区空间失败
        PyErr_NoMemory();                   // 抛出内存错误异常
        NpyIter_Deallocate(iter);           // 释放迭代器对象
        return -1;                          // 返回失败
    }

    /* The iteration loop */
    # 循环处理迭代器中的数据，直到迭代器结束
    do {
        # 获取内部循环的数据指针、步长和计数值
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;
        char *tmp;
    
        # 内部循环处理逻辑
        while (count--) {
            # 使用 memchr 替代 strnlen，因为 Mac OS X 不支持 strnlen
            tmp = memchr(data, '\0', maxlen);
    
            # 如果字符串全满，使用缓冲区
            if (tmp == NULL) {
                # 将数据复制到临时缓冲区，并添加结尾 '\0'
                memcpy(tmp_buffer, data, maxlen);
                tmp_buffer[maxlen] = '\0';
    
                # 解析 ISO8601 格式日期时间数据
                tmp_meta.base = NPY_FR_ERROR;
                if (NpyDatetime_ParseISO8601Datetime(
                        tmp_buffer, maxlen, -1,
                        NPY_UNSAFE_CASTING, &dts,
                        &tmp_meta.base, NULL) < 0) {
                    goto fail;
                }
            }
            # 否则直接在原地解析数据
            else {
                # 解析 ISO8601 格式日期时间数据
                tmp_meta.base = NPY_FR_ERROR;
                if (NpyDatetime_ParseISO8601Datetime(
                        data, tmp - data, -1,
                        NPY_UNSAFE_CASTING, &dts,
                        &tmp_meta.base, NULL) < 0) {
                    goto fail;
                }
            }
    
            # 设置元数据的数量为 1
            tmp_meta.num = 1;
            
            # 将临时元数据与主元数据进行最大公约数计算合并
            if (compute_datetime_metadata_greatest_common_divisor(meta,
                            &tmp_meta, meta, 0, 0) < 0) {
                goto fail;
            }
    
            # 更新数据指针
            data += stride;
        }
    } while(iternext(iter));  # 继续迭代处理直到迭代器结束
    
    # 释放临时缓冲区内存
    PyArray_free(tmp_buffer);
    
    # 释放迭代器资源
    NpyIter_Deallocate(iter);
    
    # 返回成功状态码
    return 0;
fail:
    # 释放临时缓冲区内存
    PyArray_free(tmp_buffer);
    # 释放 NpyIter 迭代器资源
    NpyIter_Deallocate(iter);

    # 返回错误码
    return -1;
}


/*
 * 递归确定 NPY_DATETIME 数据类型的元数据。
 *
 * 成功返回 0，失败返回 -1。
 */
static int
find_object_datetime64_meta(PyObject *obj, PyArray_DatetimeMetaData *meta)
{
    // 如果 obj 是 Datetime 标量对象
    if (PyArray_IsScalar(obj, Datetime)) {
        // 强制转换为 PyDatetimeScalarObject 对象
        PyDatetimeScalarObject *dts = (PyDatetimeScalarObject *)obj;

        /* 将其与 'meta' 合并 */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &dts->obmeta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }
    // 如果 obj 是字符串，需要解析以获取信息
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        npy_datetime tmp = 0;
        PyArray_DatetimeMetaData tmp_meta;

        // 初始化临时元数据
        tmp_meta.base = NPY_FR_ERROR;
        tmp_meta.num = 1;

        // 尝试将 Python 对象转换为 datetime
        if (convert_pyobject_to_datetime(&tmp_meta, obj,
                                        NPY_UNSAFE_CASTING, &tmp) < 0) {
            /* 如果是值错误，则清除错误 */
            if (PyErr_Occurred() &&
                    PyErr_GivenExceptionMatches(PyErr_Occurred(),
                                    PyExc_ValueError)) {
                PyErr_Clear();
                return 0;
            }
            /* 否则传播错误 */
            else {
                return -1;
            }
        }

        /* 将其与 'meta' 合并 */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &tmp_meta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }
    // 如果 obj 是 Python datetime 对象
    else if (PyDateTime_Check(obj)) {
        PyArray_DatetimeMetaData tmp_meta;

        // 设置临时元数据
        tmp_meta.base = NPY_FR_us;
        tmp_meta.num = 1;

        /* 将其与 'meta' 合并 */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &tmp_meta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }
    // 如果 obj 是 Python date 对象
    else if (PyDate_Check(obj)) {
        PyArray_DatetimeMetaData tmp_meta;

        // 设置临时元数据
        tmp_meta.base = NPY_FR_D;
        tmp_meta.num = 1;

        /* 将其与 'meta' 合并 */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &tmp_meta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }
    // 否则忽略此对象
    else {
        return 0;
    }
}

/*
 * 处理 PyDelta 值的处理函数，这些值可能也在 0 维 NumPy 数组中
 */
static int
delta_checker(PyArray_DatetimeMetaData *meta)
{
    PyArray_DatetimeMetaData tmp_meta;

    // 设置临时元数据
    tmp_meta.base = NPY_FR_us;
    tmp_meta.num = 1;

    /* 将其与 'meta' 合并 */
    if (compute_datetime_metadata_greatest_common_divisor(
            meta, &tmp_meta, meta, 0, 0) < 0) {
        return -1;
    }
    return 0;
}
/*
 * Recursively determines the metadata for an NPY_TIMEDELTA dtype.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
find_object_timedelta64_meta(PyObject *obj, PyArray_DatetimeMetaData *meta)
{
    /* Datetime scalar -> use its metadata */
    if (PyArray_IsScalar(obj, Timedelta)) {
        PyTimedeltaScalarObject *dts = (PyTimedeltaScalarObject *)obj;

        /* Combine it with 'meta' */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &dts->obmeta, meta, 1, 1) < 0) {
            return -1;
        }

        return 0;
    }
    /* String -> parse it to find out */
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /* No timedelta parser yet */
        return 0;
    }
    /* Python timedelta object -> 'us' */
    else if (PyDelta_Check(obj)) {
        return delta_checker(meta);
    }
    /* Otherwise ignore it */
    else {
        return 0;
    }
}

/*
 * Examines all the objects in the given Python object by
 * recursively descending the sequence structure. Returns a
 * datetime or timedelta type with metadata based on the data.
 */
NPY_NO_EXPORT PyArray_Descr *
find_object_datetime_type(PyObject *obj, int type_num)
{
    PyArray_DatetimeMetaData meta;

    meta.base = NPY_FR_GENERIC;
    meta.num = 1;

    if (type_num == NPY_DATETIME) {
        if (find_object_datetime64_meta(obj, &meta) < 0) {
            return NULL;
        }
        else {
            return create_datetime_dtype(type_num, &meta);
        }
    }
    else if (type_num == NPY_TIMEDELTA) {
        if (find_object_timedelta64_meta(obj, &meta) < 0) {
            return NULL;
        }
        else {
            return create_datetime_dtype(type_num, &meta);
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                    "find_object_datetime_type needs a datetime or "
                    "timedelta type number");
        return NULL;
    }
}


/*
 * Describes casting within datetimes or timedelta
 */
static NPY_CASTING
time_to_time_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    /* This is a within-dtype cast, which currently must handle byteswapping */
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    int is_timedelta = given_descrs[0]->type_num == NPY_TIMEDELTA;

    if (given_descrs[0] == given_descrs[1]) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }

    npy_bool byteorder_may_allow_view = (
            PyDataType_ISNOTSWAPPED(loop_descrs[0])
            == PyDataType_ISNOTSWAPPED(loop_descrs[1]));

    PyArray_DatetimeMetaData *meta1, *meta2;

    /* Placeholder for further datetime metadata operations */
    // 获取第一个循环描述符的日期时间元数据
    meta1 = get_datetime_metadata_from_dtype(loop_descrs[0]);
    // 断言第一个元数据非空
    assert(meta1 != NULL);
    
    // 获取第二个循环描述符的日期时间元数据
    meta2 = get_datetime_metadata_from_dtype(loop_descrs[1]);
    // 断言第二个元数据非空
    assert(meta2 != NULL);

    // 检查是否为相同的日期时间基准和数量，或者进行常见的度量单位前缀转换
    if ((meta1->base == meta2->base && meta1->num == meta2->num) ||
            // 处理一些常见的度量单位前缀转换，例如 1000 倍转换
            ((meta2->base >= 7) && (meta1->base - meta2->base == 1)
              && ((meta1->num / meta2->num) == 1000)) ||
            // 10^6 倍转换
            ((meta2->base >= 7) && (meta1->base - meta2->base == 2)
              && ((meta1->num / meta2->num) == 1000000)) ||
            // 10^9 倍转换
            ((meta2->base >= 7) && (meta1->base - meta2->base == 3)
              && ((meta1->num / meta2->num) == 1000000000))) {
        // 如果允许按字节顺序查看，则设置视图偏移为0，并返回无需类型转换
        if (byteorder_may_allow_view) {
            *view_offset = 0;
            return NPY_NO_CASTING;
        }
        // 否则返回等效类型转换
        return NPY_EQUIV_CASTING;
    }
    else if (meta1->base == NPY_FR_GENERIC) {
        // 如果允许按字节顺序查看，则设置视图偏移为0
        if (byteorder_may_allow_view) {
            *view_offset = 0;
        }
        // 返回安全类型转换
        return NPY_SAFE_CASTING;
    }
    else if (meta2->base == NPY_FR_GENERIC) {
        // 如果第二个元数据基准为通用日期时间类型，则返回不安全类型转换
        /* TODO: 这实际上是一个无效的类型转换（将会导致错误） */
        return NPY_UNSAFE_CASTING;
    }
    else if (is_timedelta && (
            /* 在时间增量的情况下，跳转日期和时间单位之间是不安全的 */
            (meta1->base <= NPY_FR_M && meta2->base > NPY_FR_M) ||
            (meta1->base > NPY_FR_M && meta2->base <= NPY_FR_M))) {
        // 如果是时间增量，并且跳转日期时间单位之间是不安全的，则返回不安全类型转换
        return NPY_UNSAFE_CASTING;
    }
    else if (meta1->base <= meta2->base) {
        // 如果第一个元数据的基准小于或等于第二个元数据的基准
        /* 当前认为向更精确的单位转换是安全的 */
        if (datetime_metadata_divides(meta1, meta2, is_timedelta)) {
            // 如果可以整除，则认为是安全的类型转换
            return NPY_SAFE_CASTING;
        }
        else {
            // 否则返回相同种类的类型转换
            return NPY_SAME_KIND_CASTING;
        }
    }
    // 默认情况下返回相同种类的类型转换
    return NPY_SAME_KIND_CASTING;
}

static int
time_to_time_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    int requires_wrap = 0;  // 初始化一个变量，用于标记是否需要进行包装
    int inner_aligned = aligned;  // 内部对齐标志初始化为输入的对齐标志
    PyArray_Descr *const *descrs = context->descriptors;  // 从上下文中获取描述符数组的常量指针
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;  // 设置方法标志，禁止浮点错误处理

    PyArray_DatetimeMetaData *meta1 = get_datetime_metadata_from_dtype(descrs[0]);  // 获取第一个描述符的日期时间元数据
    PyArray_DatetimeMetaData *meta2 = get_datetime_metadata_from_dtype(descrs[1]);  // 获取第二个描述符的日期时间元数据

    if (meta1->base == meta2->base && meta1->num == meta2->num) {
        /*
         * 如果元数据匹配，则使用低级别的复制或复制交换函数。
         * （如果它们不匹配但需要交换，将递归命中此路径。）
         */
        if (PyDataType_ISNOTSWAPPED(descrs[0]) ==
                    PyDataType_ISNOTSWAPPED(descrs[1])) {
            *out_loop = PyArray_GetStridedCopyFn(
                    aligned, strides[0], strides[1], NPY_SIZEOF_DATETIME);  // 获取适用于日期时间的步进复制函数
        }
        else {
            *out_loop = PyArray_GetStridedCopySwapFn(
                    aligned, strides[0], strides[1], NPY_SIZEOF_DATETIME);  // 获取适用于日期时间的步进复制交换函数
        }
        return 0;  // 返回成功状态
    }

    if (!PyDataType_ISNOTSWAPPED(descrs[0]) ||
            !PyDataType_ISNOTSWAPPED(descrs[1])) {
        inner_aligned = 1;  // 如果任一描述符需要交换，则设置内部对齐标志
        requires_wrap = 1;  // 标记需要包装
    }
    if (get_nbo_cast_datetime_transfer_function(
            inner_aligned, descrs[0], descrs[1],
            out_loop, out_transferdata) == NPY_FAIL) {
        return -1;  // 获取日期时间传输函数失败，返回错误状态
    }

    if (!requires_wrap) {
        return 0;  // 如果不需要包装，则直接返回成功状态
    }

    PyArray_Descr *src_wrapped_dtype = NPY_DT_CALL_ensure_canonical(descrs[0]);  // 获取第一个描述符的规范包装数据类型
    PyArray_Descr *dst_wrapped_dtype = NPY_DT_CALL_ensure_canonical(descrs[1]);  // 获取第二个描述符的规范包装数据类型

    int needs_api = 0;  // 标志是否需要 API
    int res = wrap_aligned_transferfunction(
            aligned, 0,
            strides[0], strides[1],
            descrs[0], descrs[1],
            src_wrapped_dtype, dst_wrapped_dtype,
            out_loop, out_transferdata, &needs_api);  // 包装对齐传输函数
    Py_DECREF(src_wrapped_dtype);  // 减少第一个包装数据类型的引用计数
    Py_DECREF(dst_wrapped_dtype);  // 减少第二个包装数据类型的引用计数

    assert(needs_api == 0);  // 断言不需要 API
    return res;  // 返回操作结果
}


/* 处理日期时间<->时间增量类型的解析（双向） */
static NPY_CASTING
datetime_to_timedelta_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);  // 获取第一个给定描述符的规范数据类型
    if (loop_descrs[0] == NULL) {
        return -1;  // 如果无法获取，则返回错误状态
    }
    if (given_descrs[1] == NULL) {
        PyArray_DatetimeMetaData *meta = get_datetime_metadata_from_dtype(given_descrs[0]);  // 从第一个给定描述符中获取日期时间元数据
        assert(meta != NULL);  // 断言元数据不为空
        loop_descrs[1] = create_datetime_dtype(dtypes[1]->type_num, meta);  // 根据第二个数据类型的类型编号和元数据创建日期时间数据类型
    }
    else {
        // 如果不满足条件，将给定的描述符转换为规范形式
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    }
    if (loop_descrs[1] == NULL) {
        // 如果转换后的描述符为空，则释放之前申请的资源并返回错误标识
        Py_DECREF(loop_descrs[0]);
        return -1;
    }
    /*
     * 大多数情况下，NPY_UNSAFE_CASTING 不为真，则类型转换可能会失败。
     * TODO: 一旦ufunc开始使用特定于dtype的推广规则，
     *       这种检查可能不再必要。
     */
    // 返回 NPY_UNSAFE_CASTING 标识
    return NPY_UNSAFE_CASTING;
/* In the current setup both strings and unicode casts support all outputs */
static NPY_CASTING
time_to_string_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr **given_descrs,
        PyArray_Descr **loop_descrs,
        npy_intp *NPY_UNUSED(view_offset))
{
    /* Check if the second descriptor is provided and the first dtype is datetime */
    if (given_descrs[1] != NULL && dtypes[0]->type_num == NPY_DATETIME) {
        /*
         * At the time of writing, NumPy does not check the length here,
         * but will error if filling fails.
         */
        Py_INCREF(given_descrs[1]);
        /* Assign the given descriptor to the loop descriptor */
        loop_descrs[1] = given_descrs[1];
    }
    else {
        /* Find the correct string length, possibly based on the unit */
        int size;
        if (given_descrs[0]->type_num == NPY_DATETIME) {
            /* Retrieve datetime metadata and calculate the required string length */
            PyArray_DatetimeMetaData *meta = get_datetime_metadata_from_dtype(given_descrs[0]);
            assert(meta != NULL);
            size = NpyDatetime_GetDatetimeISO8601StrLen(0, meta->base);
        }
        else {
            /*
             * This is arguably missing space for the unit, e.g. for:
             * `np.timedelta64(1231234342124, 'ms')`
             */
            size = 21;
        }
        /* Adjust size for Unicode strings */
        if (dtypes[1]->type_num == NPY_UNICODE) {
            size *= 4;
        }
        /* Create a new descriptor with the determined size */
        loop_descrs[1] = PyArray_DescrNewFromType(dtypes[1]->type_num);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
        loop_descrs[1]->elsize = size;
    }

    /* Ensure canonical form of the first descriptor */
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        Py_DECREF(loop_descrs[1]);
        return -1;
    }

    return NPY_UNSAFE_CASTING;
}

static int
datetime_to_string_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArray_Descr *const *descrs = context->descriptors;
    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;

    /* Determine the appropriate transfer function based on output descriptor type */
    if (descrs[1]->type_num == NPY_STRING) {
        /* Get transfer function for converting datetime to string */
        if (get_nbo_datetime_to_string_transfer_function(
                descrs[0], descrs[1],
                out_loop, out_transferdata) == NPY_FAIL) {
            return -1;
        }
    }
    else {
        assert(descrs[1]->type_num == NPY_UNICODE);
        int out_needs_api;
        /* Get transfer function for converting datetime to unicode */
        if (get_datetime_to_unicode_transfer_function(
                aligned, strides[0], strides[1], descrs[0], descrs[1],
                out_loop, out_transferdata, &out_needs_api) == NPY_FAIL) {
            return -1;
        }
    }
    return 0;
}

static NPY_CASTING
string_to_datetime_cast_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    # 如果给定描述符列表中第二个元素为 NULL
    if (given_descrs[1] == NULL) {
        # 使用默认描述符创建循环描述符
        /* NOTE: This doesn't actually work, and will error during the cast */
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
        # 如果创建的循环描述符为空，则返回错误码 -1
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }
    else {
        # 使用给定的描述符确保其规范化
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
        # 如果规范化后的描述符为空，则返回错误码 -1
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }

    # 我们目前支持字节交换，因此任何（Unicode）字符串都是可以的
    # 增加给定描述符列表中第一个元素的引用计数
    Py_INCREF(given_descrs[0]);
    # 将循环描述符设置为给定描述符列表中第一个元素
    loop_descrs[0] = given_descrs[0];

    # 返回不安全类型转换标志
    return NPY_UNSAFE_CASTING;
/*
 * This function handles the conversion from string or unicode to datetime objects.
 * It determines the appropriate conversion function based on the input descriptor type.
 * If the input descriptor is of type NPY_STRING, it sets up the conversion using 
 * get_nbo_string_to_datetime_transfer_function(); if it's of type NPY_UNICODE, it 
 * sets up the conversion using get_unicode_to_datetime_transfer_function().
 * It returns -1 on failure and 0 on success.
 */
static int
string_to_datetime_cast_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArray_Descr *const *descrs = context->descriptors;
    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;

    if (descrs[0]->type_num == NPY_STRING) {
        if (get_nbo_string_to_datetime_transfer_function(
                descrs[0], descrs[1], out_loop, out_transferdata) == NPY_FAIL) {
            return -1;
        }
    }
    else {
        assert(descrs[0]->type_num == NPY_UNICODE);
        int out_needs_api;
        if (get_unicode_to_datetime_transfer_function(
                aligned, strides[0], strides[1], descrs[0], descrs[1],
                out_loop, out_transferdata, &out_needs_api) == NPY_FAIL) {
            return -1;
        }
    }
    return 0;
}

/*
 * This function initializes casting implementations for datetime-related casts.
 * It sets up casting specifications for datetime to datetime and timedelta to
 * timedelta conversions, adding them to the casting registry.
 * It also sets up legacy casting loops for timedelta to datetime conversions.
 * Returns -1 on failure, otherwise returns 0.
 */
NPY_NO_EXPORT int
PyArray_InitializeDatetimeCasts()
{
    int result = -1;

    PyType_Slot slots[3];
    PyArray_DTypeMeta *dtypes[2];
    PyArrayMethod_Spec spec = {
        .name = "datetime_casts",
        .nin = 1,
        .nout = 1,
        .casting = NPY_UNSAFE_CASTING,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = dtypes,
        .slots = slots,
    };
    
    // Setting up slots for resolving descriptors and getting loop functions
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &time_to_time_resolve_descriptors;
    slots[1].slot = NPY_METH_get_loop;
    slots[1].pfunc = &time_to_time_get_loop;
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    // Setting up dtype metadata for datetime and timedelta
    PyArray_DTypeMeta *datetime = &PyArray_DatetimeDType;
    PyArray_DTypeMeta *timedelta = &PyArray_TimedeltaDType;
    PyArray_DTypeMeta *string = &PyArray_BytesDType;
    PyArray_DTypeMeta *unicode = &PyArray_UnicodeDType;
    PyArray_DTypeMeta *tmp = NULL;

    // Registering datetime to datetime casting implementation
    dtypes[0] = datetime;
    dtypes[1] = datetime;
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto fail;
    }

    // Registering timedelta to timedelta casting implementation
    dtypes[0] = timedelta;
    dtypes[1] = timedelta;
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto fail;
    }

    // Setting up legacy casting loops for timedelta to datetime conversions
    spec.flags = NPY_METH_REQUIRES_PYAPI;
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &datetime_to_timedelta_resolve_descriptors;
    slots[1].slot = NPY_METH_get_loop;
    slots[1].pfunc = &legacy_cast_get_strided_loop;
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    // Registering casting implementation for timedelta to datetime
    spec.name = "timedelta_and_datetime_cast";
    dtypes[0] = timedelta;
    dtypes[1] = datetime;
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto fail;
    }

    // Registering casting implementation for datetime to timedelta
    spec.name = "datetime_to_timedelta_cast";
    dtypes[0] = datetime;
    dtypes[1] = timedelta;
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto fail;
    }

fail:
    return result;
}
    // 尝试向类型转换规范中添加一个从spec到datetime的转换实现
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        // 如果添加失败，则跳转到错误处理标签fail
        goto fail;
    }

    /*
     * 从数值类型向时间类型进行转换。这些转换使用存储在数据类型上的转换函数，
     * 这些函数应该在某个时候被替换掉。某些转换可能失败（例如转换为无单位的日期时间），
     * 但这些属于特例。
     */
    for (int num = 0; num < NPY_NTYPES_LEGACY; num++) {
        // 如果不是数值类型且不是布尔类型，则继续下一次循环
        if (!PyTypeNum_ISNUMBER(num) && num != NPY_BOOL) {
            continue;
        }

        // 设置tmp为num对应的数据类型对象
        Py_XSETREF(tmp, PyArray_DTypeFromTypeNum(num));

        // 尝试添加从tmp到datetime的不安全转换实现
        if (PyArray_AddLegacyWrapping_CastingImpl(
                tmp, datetime, NPY_UNSAFE_CASTING) < 0) {
            goto fail;
        }
        // 尝试添加从datetime到tmp的不安全转换实现
        if (PyArray_AddLegacyWrapping_CastingImpl(
                datetime, tmp, NPY_UNSAFE_CASTING) < 0) {
            goto fail;
        }

        // 初始化默认的到timedelta的转换级别为不安全
        NPY_CASTING to_timedelta_casting = NPY_UNSAFE_CASTING;
        // 如果是整数类型或者布尔类型
        if (PyTypeNum_ISINTEGER(num) || num == NPY_BOOL) {
            /* timedelta目前与int64相同... */
            // 如果是无符号整数类型并且tmp的大小为8字节
            if (PyTypeNum_ISUNSIGNED(num) && tmp->singleton->elsize == 8) {
                to_timedelta_casting = NPY_SAME_KIND_CASTING;
            }
            else {
                to_timedelta_casting = NPY_SAFE_CASTING;
            }
        }
        // 尝试添加从tmp到timedelta的转换实现，级别为to_timedelta_casting
        if (PyArray_AddLegacyWrapping_CastingImpl(
                tmp, timedelta, to_timedelta_casting) < 0) {
            goto fail;
        }
        // 尝试添加从timedelta到tmp的不安全转换实现
        if (PyArray_AddLegacyWrapping_CastingImpl(
                timedelta, tmp, NPY_UNSAFE_CASTING) < 0) {
            goto fail;
        }
    }

    /*
     * 时间类型向字符串和Unicode的转换
     */
    // 设置spec的转换级别为不安全
    spec.casting = NPY_UNSAFE_CASTING;
    /*
     * 转换可能会出错，需要API支持（例如Unicode需要用于从字符串到Unicode的转换）。
     * 当前通过传统的转换实现Unicode处理。Datetime向字符串有自己的快速转换，而timedelta向字符串则使用传统的后备方式。
     */
    // 第一个槽位用于解析描述符
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &time_to_string_resolve_descriptors;
    // 第二个槽位用于获取循环（与两者不同）
    slots[1].slot = NPY_METH_get_loop;
    // 第三个槽位初始化为0，函数指针为空
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    // 设置第一个数据类型为datetime
    dtypes[0] = datetime;
    // 循环处理从NPY_DATETIME到NPY_TIMEDELTA的所有类型
    for (int num = NPY_DATETIME; num <= NPY_TIMEDELTA; num++) {
        if (num == NPY_DATETIME) {
            dtypes[0] = datetime;
            // 设置spec的标志为支持非对齐访问和需要Python API
            spec.flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
            // 第二个槽位的函数指针设置为datetime向字符串的获取循环函数
            slots[1].pfunc = &datetime_to_string_get_loop;
        }
        else {
            dtypes[0] = timedelta;
            // 设置spec的标志为需要Python API
            spec.flags = NPY_METH_REQUIRES_PYAPI;
            // 第二个槽位的函数指针设置为传统转换获取循环函数
            slots[1].pfunc = &legacy_cast_get_strided_loop;
        }

        // 处理从NPY_STRING到NPY_UNICODE的所有类型
        for (int str = NPY_STRING; str <= NPY_UNICODE; str++) {
            // 设置第二个数据类型为当前str对应的数据类型对象
            dtypes[1] = PyArray_DTypeFromTypeNum(str);

            // 尝试从spec添加一个转换实现，第一参数的数量为1
            int res = PyArray_AddCastingImplementation_FromSpec(&spec, 1);
            // 清除第二个数据类型的引用
            Py_SETREF(dtypes[1], NULL);
            // 如果添加失败，则跳转到错误处理标签fail
            if (res < 0) {
                goto fail;
            }
        }
    }
    /*
     * Cast strings to timedelta are currently only legacy casts
     */
    如果将字符串转换为时间增量（timedelta），目前只支持旧版转换方式
    if (PyArray_AddLegacyWrapping_CastingImpl(
            string, timedelta, NPY_UNSAFE_CASTING) < 0) {
        // 如果添加转换实现失败，则跳转到失败标签
        goto fail;
    }
    如果将Unicode字符串转换为时间增量（timedelta），同样采用旧版转换方式
    if (PyArray_AddLegacyWrapping_CastingImpl(
            unicode, timedelta, NPY_UNSAFE_CASTING) < 0) {
        // 如果添加转换实现失败，则跳转到失败标签
        goto fail;
    }

    /*
     * Cast strings to datetime
     */
    将字符串转换为日期时间类型（datetime）
    dtypes[1] = datetime;
    spec.casting = NPY_UNSAFE_CASTING;

    /* 默认的类型解析应该可以正常工作。*/
    第一个插槽使用解析描述符方法
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &string_to_datetime_cast_resolve_descriptors;
    第二个插槽使用获取循环方法
    slots[1].slot = NPY_METH_get_loop;
    slots[1].pfunc = &string_to_datetime_cast_get_loop;
    第三个插槽为空
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    使用字符串类型进行转换
    dtypes[0] = string;
    设置规范的标志位
    spec.flags = NPY_METH_SUPPORTS_UNALIGNED;
    如果添加来自规范的转换实现失败，则跳转到失败标签
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto fail;
    }

    使用Unicode类型进行转换
    dtypes[0] = unicode;
    /*
     * Unicode handling is currently implemented via a legacy cast, which
     * requires the Python API.
     */
    Unicode的处理目前通过旧版转换实现，因此需要使用Python API。
    spec.flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    如果添加来自规范的转换实现失败，则跳转到失败标签
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto fail;
    }

    操作成功，返回0
    result = 0;
  fail:
    Py_XDECREF(tmp);  // 释放临时对象
    返回result变量作为函数返回值
    return result;
}


这段代码表示一个代码块的结束。在许多编程语言中，大括号 `{}` 用于表示代码块的开始和结束，这里的 `}` 表示的是一个块的结束。在此例中，它结束了一个函数或是一个类的定义，具体取决于上下文。
```