# `.\numpy\numpy\_core\src\multiarray\datetime_strings.c`

```py
/*
 * This file implements string parsing and creation for NumPy datetime.
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

#include "npy_config.h"


#include "numpy/arrayscalars.h"
#include "convert_datatype.h"
#include "_datetime.h"
#include "datetime_strings.h"

#include <time.h>

/*
 * Platform-specific time_t typedef. Some platforms use 32 bit, some use 64 bit
 * and we just use the default with the exception of mingw, where we must use
 * 64 bit because MSVCRT version 9 does not have the (32 bit) localtime()
 * symbol, so we need to use the 64 bit version [1].
 *
 * [1] http://thread.gmane.org/gmane.comp.gnu.mingw.user/27011
 */
#if defined(NPY_MINGW_USE_CUSTOM_MSVCR)
 typedef __time64_t NPY_TIME_T;
#else
 typedef time_t NPY_TIME_T;
#endif

/*
 * Wraps `localtime` functionality for multiple platforms. This
 * converts a time value to a time structure in the local timezone.
 * If size(NPY_TIME_T) == 4, then years must be between 1970 and 2038. If
 * size(NPY_TIME_T) == 8, then years must be later than 1970. If the years are
 * not in this range, then get_localtime() will fail on some platforms.
 *
 * Returns 0 on success, -1 on failure.
 *
 * Notes:
 * 1) If NPY_TIME_T is 32 bit (i.e. sizeof(NPY_TIME_T) == 4), then the
 *    maximum year it can represent is 2038 (see [1] for more details). Trying
 *    to use a higher date like 2041 in the 32 bit "ts" variable below will
 *    typically result in "ts" being a negative number (corresponding roughly
 *    to a year ~ 1905). If NPY_TIME_T is 64 bit, then there is no such
 *    problem in practice.
 * 2) If the "ts" argument to localtime() is negative, it represents
 *    years < 1970 both for 32 and 64 bits (for 32 bits the earliest year it can
 *    represent is 1901, while 64 bits can represent much earlier years).
 * 3) On Linux, localtime() works for negative "ts". On Windows and in Wine,
 *    localtime() as well as the localtime_s() and _localtime64_s() functions
 *    will fail for any negative "ts" and return a nonzero exit number
 *    (localtime_s, _localtime64_s) or NULL (localtime). This behavior is the
 *    same for both 32 and 64 bits.
 *
 * From this it follows that get_localtime() is only guaranteed to work
 * correctly on all platforms for years between 1970 and 2038 for 32bit
 * NPY_TIME_T and years higher than 1970 for 64bit NPY_TIME_T. For
 * multiplatform code, get_localtime() should never be used outside of this
 * range.
 *
 * [1] https://en.wikipedia.org/wiki/Year_2038_problem
 */
static int
get_localtime(NPY_TIME_T *ts, struct tm *tms)
{
    char *func_name = "<unknown>";
#if defined(_WIN32)
 #if defined(_MSC_VER) && (_MSC_VER >= 1400)
    // Use localtime_s() on Windows platforms with MSVC 2005 or newer
    if (localtime_s(tms, ts) != 0) {
        func_name = "localtime_s";
        goto fail;

    // Use localtime_s() on Windows platforms with MSVC 2005 or newer
    if (localtime_s(tms, ts) != 0) {
        func_name = "localtime_s";
        // If localtime_s() fails, set func_name and jump to fail label
        goto fail;
    }
#else
    // Use localtime() for other Windows platforms and non-Windows platforms
    struct tm *tm_result = localtime(ts);
    if (tm_result == NULL) {
        func_name = "localtime";
        // If localtime() fails, set func_name and jump to fail label
        goto fail;
    }
    // Copy the result from tm_result to tms
    *tms = *tm_result;
#endif

    // Return success (0) if localtime function succeeded
    return 0;

fail:
    // Print an error message indicating the function that failed
    PyErr_Format(PyExc_ValueError, "%s failed", func_name);
    // Return failure (-1) if localtime function failed
    return -1;
}
    }
 #elif defined(NPY_MINGW_USE_CUSTOM_MSVCR)
    // 如果定义了 NPY_MINGW_USE_CUSTOM_MSVCR，则使用 _localtime64_s 函数获取本地时间
    if (_localtime64_s(tms, ts) != 0) {
        // 如果 _localtime64_s 函数返回非零值，表示调用失败，记录函数名并跳转到失败处理标签
        func_name = "_localtime64_s";
        goto fail;
    }
 #else
    // 否则，使用标准库函数 localtime 获取本地时间
    struct tm *tms_tmp;
    tms_tmp = localtime(ts);
    // 如果 localtime 返回空指针，则表示调用失败，记录函数名并跳转到失败处理标签
    if (tms_tmp == NULL) {
        func_name = "localtime";
        goto fail;
    }
    // 复制获取到的时间结构体到指定的 tms 结构体中
    memcpy(tms, tms_tmp, sizeof(struct tm));
 #endif
#else
    // 如果调用 localtime_r(ts, tms) 失败，则设置 func_name 并跳转到 fail 标签处
    if (localtime_r(ts, tms) == NULL) {
        func_name = "localtime_r";
        goto fail;
    }
#endif

    // 成功返回 0
    return 0;

fail:
    // 失败时格式化错误信息，并设置异常类型为 PyExc_OSError，返回 -1
    PyErr_Format(PyExc_OSError, "Failed to use '%s' to convert "
                                "to a local time", func_name);
    return -1;
}

/*
 * Converts a datetimestruct in UTC to a datetimestruct in local time,
 * also returning the timezone offset applied. This function works for any year
 * > 1970 on all platforms and both 32 and 64 bits. If the year < 1970, then it
 * will fail on some platforms.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
convert_datetimestruct_utc_to_local(npy_datetimestruct *out_dts_local,
                const npy_datetimestruct *dts_utc, int *out_timezone_offset)
{
    NPY_TIME_T rawtime = 0, localrawtime;
    struct tm tm_;
    npy_int64 year_correction = 0;

    /* Make a copy of the input 'dts' to modify */
    // 复制输入的 dts 到 out_dts_local 进行修改
    *out_dts_local = *dts_utc;

    /*
     * For 32 bit NPY_TIME_T, the get_localtime() function does not work for
     * years later than 2038, see the comments above get_localtime(). So if the
     * year >= 2038, we instead call get_localtime() for the year 2036 or 2037
     * (depending on the leap year) which must work and at the end we add the
     * 'year_correction' back.
     */
    // 对于 32 位 NPY_TIME_T，get_localtime() 函数无法处理 2038 年之后的年份
    // 如果年份大于等于 2038，则调整年份和 year_correction
    if (sizeof(NPY_TIME_T) == 4 && out_dts_local->year >= 2038) {
        if (is_leapyear(out_dts_local->year)) {
            /* 2036 is a leap year */
            year_correction = out_dts_local->year - 2036;
            out_dts_local->year -= year_correction; /* = 2036 */
        }
        else {
            /* 2037 is not a leap year */
            year_correction = out_dts_local->year - 2037;
            out_dts_local->year -= year_correction; /* = 2037 */
        }
    }

    /*
     * Convert everything in 'dts' to a time_t, to minutes precision.
     * This is POSIX time, which skips leap-seconds, but because
     * we drop the seconds value from the npy_datetimestruct, everything
     * is ok for this operation.
     */
    // 将 dts 中的所有内容转换为时间戳 rawtime，精确到分钟
    rawtime = (NPY_TIME_T)get_datetimestruct_days(out_dts_local) * 24 * 60 * 60;
    rawtime += dts_utc->hour * 60 * 60;
    rawtime += dts_utc->min * 60;

    /* localtime converts a 'time_t' into a local 'struct tm' */
    // 使用 localtime 将 time_t 转换为本地时间结构体 tm_
    if (get_localtime(&rawtime, &tm_) < 0) {
        /* This should only fail if year < 1970 on some platforms. */
        // 如果失败，通常是因为年份小于 1970 在某些平台上
        return -1;
    }

    /* Copy back all the values except seconds */
    // 复制除了秒数之外的所有值回 out_dts_local
    out_dts_local->min = tm_.tm_min;
    out_dts_local->hour = tm_.tm_hour;
    out_dts_local->day = tm_.tm_mday;
    out_dts_local->month = tm_.tm_mon + 1;
    out_dts_local->year = tm_.tm_year + 1900;

    /* Extract the timezone offset that was applied */
    // 计算应用的时区偏移量
    rawtime /= 60;
    localrawtime = (NPY_TIME_T)get_datetimestruct_days(out_dts_local) * 24 * 60;
    localrawtime += out_dts_local->hour * 60;
    localrawtime += out_dts_local->min;

    *out_timezone_offset = localrawtime - rawtime;

    /* Reapply the year 2038 year correction */
    // 重新应用年份修正值 2038
    # 将 out_dts_local 指针所指向的结构体的年份字段增加 year_correction 的值
    out_dts_local->year += year_correction;
    
    # 返回整数值 0，表示函数执行成功
    return 0;
/*NUMPY_API
 *
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
 * 'unit' should contain -1 if the unit is unknown, or the unit
 *      which will be used if it is.
 * 'casting' controls how the detected unit from the string is allowed
 *           to be cast to the 'unit' parameter.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_bestunit' gives a suggested unit based on the amount of
 *      resolution provided in the string, or -1 for NaT.
 * 'out_special' gets set to 1 if the parsed time was 'today',
 *      'now', or ''/'NaT'. For 'today', the unit recommended is
 *      'D', for 'now', the unit recommended is 's', and for 'NaT'
 *      the unit recommended is 'Y'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
NpyDatetime_ParseISO8601Datetime(
        char const *str, Py_ssize_t len,
        NPY_DATETIMEUNIT unit,
        NPY_CASTING casting,
        npy_datetimestruct *out,
        NPY_DATETIMEUNIT *out_bestunit,
        npy_bool *out_special)
{
    int year_leap = 0;
    int i, numdigits;
    char const *substr;
    Py_ssize_t sublen;
    NPY_DATETIMEUNIT bestunit;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;

    /*
     * Convert the empty string and case-variants of "NaT" to not-a-time.
     * Tried to use PyOS_stricmp, but that function appears to be broken,
     * not even matching the strcmp function signature as it should.
     */
    if (len <= 0 || (len == 3 &&
                        tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'a' &&
                        tolower(str[2]) == 't')) {
        // 将输出年份设置为不是时间（NPY_DATETIME_NAT）
        out->year = NPY_DATETIME_NAT;

        /*
         * Indicate that this was a special value, and
         * recommend generic units.
         */
        if (out_bestunit != NULL) {
            // 建议使用通用单位
            *out_bestunit = NPY_FR_GENERIC;
        }
        if (out_special != NULL) {
            // 标记为特殊值
            *out_special = 1;
        }

        // 返回成功
        return 0;
    }
    # 如果时间单位是通用单位（NPY_FR_GENERIC）
    if (unit == NPY_FR_GENERIC) {
        # 设置一个错误字符串，指示无法创建除 NaT 外的 NumPy datetime 对象
        PyErr_SetString(PyExc_ValueError,
                    "Cannot create a NumPy datetime other than NaT "
                    "with generic units");
        # 返回错误代码
        return -1;
    }

    /*
     * 字符串 "today" 表示获取当前本地时间的日期，并将其转换为日期表示。
     * 如果强制将此日期表示转换为时间单位，它将是 UTC 时间的午夜。
     * 这可能有点奇怪，但是这样做是为了确保 'datetime64[D]' 类型产生您期望的日期，
     * 而不是根据当前时间和时区切换到相邻的日期。
     */
    # 如果字符串长度为5且全小写，且依次是 'today'
    if (len == 5 && tolower(str[0]) == 't' &&
                    tolower(str[1]) == 'o' &&
                    tolower(str[2]) == 'd' &&
                    tolower(str[3]) == 'a' &&
                    tolower(str[4]) == 'y') {
        # 初始化变量 rawtime 和 tm_
        NPY_TIME_T rawtime = 0;
        struct tm tm_;

        # 获取当前时间
        time(&rawtime);
        # 获取本地时间
        if (get_localtime(&rawtime, &tm_) < 0) {
            return -1;
        }
        # 将日期分量分配给输出结构体的年、月、日字段
        out->year = tm_.tm_year + 1900;
        out->month = tm_.tm_mon + 1;
        out->day = tm_.tm_mday;

        # 将时间单位设为 'D'（天）
        bestunit = NPY_FR_D;

        /*
         * 表明这是一个特殊值，并且是一个日期（单位为 'D'）。
         */
        # 如果 out_bestunit 不为空，则将 bestunit 赋值给 out_bestunit
        if (out_bestunit != NULL) {
            *out_bestunit = bestunit;
        }
        # 如果 out_special 不为空，则将其设为1，表示是特殊值
        if (out_special != NULL) {
            *out_special = 1;
        }

        /* 检查类型转换规则 */
        # 如果单位不是错误状态，并且无法按照指定的转换规则将 bestunit 转换为 unit
        if (unit != NPY_FR_ERROR &&
                !can_cast_datetime64_units(bestunit, unit, casting)) {
            # 抛出类型错误，说明无法将字符串 str 解析为单位 'unit' 的日期
            PyErr_Format(PyExc_TypeError, "Cannot parse \"%s\" as unit "
                         "'%s' using casting rule %s",
                         str, _datetime_strings[unit],
                         npy_casting_to_string(casting));
            return -1;
        }

        # 返回成功代码
        return 0;
    }

    /* 字符串 "now" 表示当前的 UTC 时间 */
    // 检查字符串长度是否为3，并且前三个字符转换为小写后是否为 "now"
    if (len == 3 && tolower(str[0]) == 'n' &&
                    tolower(str[1]) == 'o' &&
                    tolower(str[2]) == 'w') {
        NPY_TIME_T rawtime = 0; // 初始化 rawtime 为 0
        PyArray_DatetimeMetaData meta; // 创建 PyArray_DatetimeMetaData 结构体变量 meta

        time(&rawtime); // 获取当前时间戳，并存储在 rawtime 中

        /* Set up a dummy metadata for the conversion */
        meta.base = NPY_FR_s; // 设置 meta 结构体的基础时间单位为秒
        meta.num = 1; // 设置 meta 结构体的数量为1

        bestunit = NPY_FR_s; // 设置最佳单位为秒

        /*
         * Indicate that this was a special value, and
         * use 's' because the time() function has resolution
         * seconds.
         */
        // 如果 out_bestunit 不为空，将 bestunit 赋值给它
        if (out_bestunit != NULL) {
            *out_bestunit = bestunit;
        }
        // 如果 out_special 不为空，将 1 赋值给它，表示这是一个特殊值
        if (out_special != NULL) {
            *out_special = 1;
        }

        /* Check the casting rule */
        // 检查转换规则是否允许将 bestunit 转换为 unit
        if (unit != NPY_FR_ERROR &&
                !can_cast_datetime64_units(bestunit, unit, casting)) {
            // 如果不允许，则设置错误并返回 -1
            PyErr_Format(PyExc_TypeError, "Cannot parse \"%s\" as unit "
                         "'%s' using casting rule %s",
                         str, _datetime_strings[unit],
                         npy_casting_to_string(casting));
            return -1;
        }

        // 调用函数将时间数据转换为结构体形式，并返回结果
        return NpyDatetime_ConvertDatetime64ToDatetimeStruct(&meta, rawtime, out);
    }

    /* Anything else isn't a special value */
    // 如果不是 "now"，则不是特殊值，将 out_special 设为 0
    if (out_special != NULL) {
        *out_special = 0;
    }

    substr = str; // 设置 substr 指向字符串起始处
    sublen = len; // 设置 sublen 为字符串长度

    /* Skip leading whitespace */
    // 跳过开头的空白字符
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    /* Leading '-' sign for negative year */
    // 处理负数年份的前导 '-' 符号
    if (*substr == '-' || *substr == '+') {
        ++substr;
        --sublen;
    }

    // 如果长度为 0，则跳到解析错误处
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE YEAR (digits until the '-' character) */
    // 解析年份（直到遇到 '-' 字符）
    out->year = 0;
    while (sublen > 0 && isdigit(*substr)) {
        out->year = 10 * out->year + (*substr - '0');
        ++substr;
        --sublen;
    }

    /* Negate the year if necessary */
    // 如果是负数年份，将年份取反
    if (str[0] == '-') {
        out->year = -out->year;
    }
    /* Check whether it's a leap-year */
    // 检查是否为闰年
    year_leap = is_leapyear(out->year);

    /* Next character must be a '-' or the end of the string */
    // 下一个字符必须是 '-' 或者字符串的结尾
    if (sublen == 0) {
        bestunit = NPY_FR_Y;
        goto finish;
    }
    else if (*substr == '-') {
        ++substr;
        --sublen;
    }
    else {
        goto parse_error;
    }

    /* Can't have a trailing '-' */
    // 不能以 '-' 结尾
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE MONTH (2 digits) */
    // 解析月份（两位数字）
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->month = 10 * (substr[0] - '0') + (substr[1] - '0');

        // 检查月份是否在合法范围内
        if (out->month < 1 || out->month > 12) {
            PyErr_Format(PyExc_ValueError,
                        "Month out of range in datetime string \"%s\"", str);
            goto error;
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a '-' or the end of the string */
    /* 如果子串长度为0 */
    if (sublen == 0) {
        /* 将最佳单位设置为 NPY_FR_M（月份） */
        bestunit = NPY_FR_M;
        /* 跳转到结束标签 */
        goto finish;
    }
    /* 如果子串的第一个字符是 '-' */
    else if (*substr == '-') {
        /* 移动指针到下一个字符 */
        ++substr;
        /* 减少子串长度 */
        --sublen;
    }
    /* 如果不满足上述两个条件，则说明解析错误 */
    else {
        /* 跳转到解析错误标签 */
        goto parse_error;
    }

    /* 不能以 '-' 结尾 */
    if (sublen == 0) {
        /* 跳转到解析错误标签 */
        goto parse_error;
    }

    /* 解析天数（2位数字） */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        /* 计算天数 */
        out->day = 10 * (substr[0] - '0') + (substr[1] - '0');

        /* 检查天数范围是否有效 */
        if (out->day < 1 ||
                    out->day > _days_per_month_table[year_leap][out->month-1]) {
            /* 抛出异常，天数超出范围 */
            PyErr_Format(PyExc_ValueError,
                        "Day out of range in datetime string \"%s\"", str);
            /* 跳转到错误处理标签 */
            goto error;
        }
        /* 移动指针和减少子串长度 */
        substr += 2;
        sublen -= 2;
    }
    else {
        /* 跳转到解析错误标签 */
        goto parse_error;
    }

    /* 下一个字符必须是 'T', ' ', 或者字符串结束 */
    if (sublen == 0) {
        /* 将最佳单位设置为 NPY_FR_D（日） */
        bestunit = NPY_FR_D;
        /* 跳转到结束标签 */
        goto finish;
    }
    else if (*substr != 'T' && *substr != ' ') {
        /* 跳转到解析错误标签 */
        goto parse_error;
    }
    else {
        /* 移动指针到下一个字符 */
        ++substr;
        /* 减少子串长度 */
        --sublen;
    }

    /* 解析小时数（2位数字） */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        /* 计算小时数 */
        out->hour = 10 * (substr[0] - '0') + (substr[1] - '0');

        /* 检查小时数范围是否有效 */
        if (out->hour >= 24) {
            /* 抛出异常，小时数超出范围 */
            PyErr_Format(PyExc_ValueError,
                        "Hours out of range in datetime string \"%s\"", str);
            /* 跳转到错误处理标签 */
            goto error;
        }
        /* 移动指针和减少子串长度 */
        substr += 2;
        sublen -= 2;
    }
    else {
        /* 跳转到解析错误标签 */
        goto parse_error;
    }

    /* 下一个字符必须是 ':' 或者字符串结束 */
    if (sublen > 0 && *substr == ':') {
        /* 移动指针到下一个字符 */
        ++substr;
        /* 减少子串长度 */
        --sublen;
    }
    else {
        /* 将最佳单位设置为 NPY_FR_h（小时） */
        bestunit = NPY_FR_h;
        /* 跳转到时区解析标签 */
        goto parse_timezone;
    }

    /* 不能以 ':' 结尾 */
    if (sublen == 0) {
        /* 跳转到解析错误标签 */
        goto parse_error;
    }

    /* 解析分钟数（2位数字） */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        /* 计算分钟数 */
        out->min = 10 * (substr[0] - '0') + (substr[1] - '0');

        /* 检查分钟数范围是否有效 */
        if (out->min >= 60) {
            /* 抛出异常，分钟数超出范围 */
            PyErr_Format(PyExc_ValueError,
                        "Minutes out of range in datetime string \"%s\"", str);
            /* 跳转到错误处理标签 */
            goto error;
        }
        /* 移动指针和减少子串长度 */
        substr += 2;
        sublen -= 2;
    }
    else {
        /* 跳转到解析错误标签 */
        goto parse_error;
    }

    /* 下一个字符必须是 ':' 或者字符串结束 */
    if (sublen > 0 && *substr == ':') {
        /* 移动指针到下一个字符 */
        ++substr;
        /* 减少子串长度 */
        --sublen;
    }
    else {
        /* 将最佳单位设置为 NPY_FR_m（分钟） */
        bestunit = NPY_FR_m;
        /* 跳转到时区解析标签 */
        goto parse_timezone;
    }

    /* 不能以 ':' 结尾 */
    if (sublen == 0) {
        /* 跳转到解析错误标签 */
        goto parse_error;
    }

    /* 解析秒数（2位数字） */
    // 检查子字符串长度大于等于2且前两个字符均为数字
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        // 将前两个字符解析为秒数
        out->sec = 10 * (substr[0] - '0') + (substr[1] - '0');

        // 如果秒数超过60，则抛出异常并跳转到错误处理
        if (out->sec >= 60) {
            PyErr_Format(PyExc_ValueError,
                        "Seconds out of range in datetime string \"%s\"", str);
            goto error;
        }
        // 更新子字符串和长度以排除已解析的部分
        substr += 2;
        sublen -= 2;
    }
    else {
        // 如果前两个字符不是数字，则跳转到解析错误处理
        goto parse_error;
    }

    /* 下一个字符可能是'.'，表示有小数秒 */
    if (sublen > 0 && *substr == '.') {
        ++substr;
        --sublen;
    }
    else {
        // 如果没有小数秒部分，则设定时间单位为秒，并跳转到解析时区部分
        bestunit = NPY_FR_s;
        goto parse_timezone;
    }

    /* 解析微秒部分（0到6位数字） */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        // 微秒数乘以10，加上下一个字符的数字值（如果存在）
        out->us *= 10;
        if (sublen > 0  && isdigit(*substr)) {
            out->us += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    // 如果没有剩余字符或下一个字符不是数字
    if (sublen == 0 || !isdigit(*substr)) {
        // 根据解析的数字位数设定最适合的时间单位
        if (numdigits > 3) {
            bestunit = NPY_FR_us;
        }
        else {
            bestunit = NPY_FR_ms;
        }
        // 跳转到解析时区部分
        goto parse_timezone;
    }

    /* 解析皮秒部分（0到6位数字） */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        // 皮秒数乘以10，加上下一个字符的数字值（如果存在）
        out->ps *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->ps += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    // 如果没有剩余字符或下一个字符不是数字
    if (sublen == 0 || !isdigit(*substr)) {
        // 根据解析的数字位数设定最适合的时间单位
        if (numdigits > 3) {
            bestunit = NPY_FR_ps;
        }
        else {
            bestunit = NPY_FR_ns;
        }
        // 跳转到解析时区部分
        goto parse_timezone;
    }

    /* 解析阿托秒部分（0到6位数字） */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        // 阿托秒数乘以10，加上下一个字符的数字值（如果存在）
        out->as *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->as += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    // 根据解析的数字位数设定最适合的时间单位
    if (numdigits > 3) {
        bestunit = NPY_FR_as;
    }
    else {
        bestunit = NPY_FR_fs;
    }
    // 跳转到解析时区部分
    ```
// 如果子串长度为0，跳转到结束标记
if (sublen == 0) {
    goto finish;
}
else {
    // 清除之前的错误状态
    PyErr_Clear();
    // 发出警告：对于 np.datetime64，没有显式的时区表示
    if (PyErr_WarnEx(PyExc_UserWarning,
        "no explicit representation of timezones available for np.datetime64",
        1) < 0) {
            return -1;
        }
}

// UTC 时间标识符处理
if (*substr == 'Z') {
    // 如果子串只有一个字符 'Z'，跳转到结束标记
    if (sublen == 1) {
        goto finish;
    }
    else {
        // 移动子串指针和减少子串长度
        ++substr;
        --sublen;
    }
}
// 处理时区偏移量
else if (*substr == '-' || *substr == '+') {
    int offset_neg = 0, offset_hour = 0, offset_minute = 0;

    // 判断偏移量正负
    if (*substr == '-') {
        offset_neg = 1;
    }
    ++substr;
    --sublen;

    // 处理小时偏移量
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        offset_hour = 10 * (substr[0] - '0') + (substr[1] - '0');
        substr += 2;
        sublen -= 2;
        // 如果小时偏移量超出范围，报错
        if (offset_hour >= 24) {
            PyErr_Format(PyExc_ValueError,
                        "Timezone hours offset out of range "
                        "in datetime string \"%s\"", str);
            goto error;
        }
    }
    else {
        goto parse_error;
    }

    // 处理分钟偏移量（可选部分）
    if (sublen > 0) {
        // 可选的 ':' 分隔符
        if (*substr == ':') {
            ++substr;
            --sublen;
        }

        // 处理分钟偏移量（在字符串末尾）
        if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
            offset_minute = 10 * (substr[0] - '0') + (substr[1] - '0');
            substr += 2;
            sublen -= 2;
            // 如果分钟偏移量超出范围，报错
            if (offset_minute >= 60) {
                PyErr_Format(PyExc_ValueError,
                            "Timezone minutes offset out of range "
                            "in datetime string \"%s\"", str);
                goto error;
            }
        }
        else {
            goto parse_error;
        }
    }

    // 应用时间偏移量
    if (offset_neg) {
        offset_hour = -offset_hour;
        offset_minute = -offset_minute;
    }
    // 将偏移量分钟数添加到日期时间结构
    add_minutes_to_datetimestruct(out, -60 * offset_hour - offset_minute);
}

// 跳过尾部空白字符
while (sublen > 0 && isspace(*substr)) {
    ++substr;
    --sublen;
}

// 如果子串长度不为0，报错
if (sublen != 0) {
    goto parse_error;
}

// 达到正常解析结束标记
finish:
// 如果 out_bestunit 不为 NULL，则将 bestunit 赋给 *out_bestunit
if (out_bestunit != NULL) {
    *out_bestunit = bestunit;
}

// 检查转换规则
    # 检查单元是否不是错误标志，并且无法将 datetime64 的最佳单位转换为指定的单位
    if (unit != NPY_FR_ERROR &&
            !can_cast_datetime64_units(bestunit, unit, casting)) {
        # 抛出类型错误异常，指示无法使用指定的类型转换规则将字符串解析为指定单位的日期时间单位
        PyErr_Format(PyExc_TypeError, "Cannot parse \"%s\" as unit "
                     "'%s' using casting rule %s",
                     str, _datetime_strings[unit],
                     npy_casting_to_string(casting));
        # 返回 -1 表示失败
        return -1;
    }

    # 如果能够成功解析，返回 0 表示成功
    return 0;
parse_error:
    PyErr_Format(PyExc_ValueError,
            "Error parsing datetime string \"%s\" at position %zd",
            str, substr - str);
    return -1;


// 在解析日期时间字符串时发生错误，格式化错误信息并返回 -1
parse_error:
    PyErr_Format(PyExc_ValueError,
            "Error parsing datetime string \"%s\" at position %zd",
            str, substr - str);
    return -1;



error:
    return -1;
}


// 返回 -1 表示发生错误
error:
    return -1;
}



/*NUMPY_API
 *
 * Provides a string length to use for converting datetime
 * objects with the given local and unit settings.
 */
NPY_NO_EXPORT int
NpyDatetime_GetDatetimeISO8601StrLen(int local, NPY_DATETIMEUNIT base)
{
    int len = 0;

    switch (base) {
        case NPY_FR_ERROR:
            /* If no unit is provided, return the maximum length */
            return NPY_DATETIME_MAX_ISO8601_STRLEN;
        case NPY_FR_GENERIC:
            /* Generic units can only be used to represent NaT */
            return 4;
        case NPY_FR_as:
            len += 3;  /* "###" */
        case NPY_FR_fs:
            len += 3;  /* "###" */
        case NPY_FR_ps:
            len += 3;  /* "###" */
        case NPY_FR_ns:
            len += 3;  /* "###" */
        case NPY_FR_us:
            len += 3;  /* "###" */
        case NPY_FR_ms:
            len += 4;  /* ".###" */
        case NPY_FR_s:
            len += 3;  /* ":##" */
        case NPY_FR_m:
            len += 3;  /* ":##" */
        case NPY_FR_h:
            len += 3;  /* "T##" */
        case NPY_FR_D:
        case NPY_FR_W:
            len += 3;  /* "-##" */
        case NPY_FR_M:
            len += 3;  /* "-##" */
        case NPY_FR_Y:
            len += 21; /* 64-bit year */
            break;
    }

    if (base >= NPY_FR_h) {
        if (local) {
            len += 5;  /* "+####" or "-####" */
        }
        else {
            len += 1;  /* "Z" */
        }
    }

    len += 1; /* NULL terminator */

    return len;
}


/*NUMPY_API
 *
 * 提供用于转换日期时间对象的字符串长度，根据给定的本地和单位设置。
 */
NPY_NO_EXPORT int
NpyDatetime_GetDatetimeISO8601StrLen(int local, NPY_DATETIMEUNIT base)
{
    int len = 0;

    switch (base) {
        case NPY_FR_ERROR:
            /* 如果没有提供单位，则返回最大长度 */
            return NPY_DATETIME_MAX_ISO8601_STRLEN;
        case NPY_FR_GENERIC:
            /* 通用单位只能用于表示 NaT */
            return 4;
        case NPY_FR_as:
            len += 3;  /* "###" */
        case NPY_FR_fs:
            len += 3;  /* "###" */
        case NPY_FR_ps:
            len += 3;  /* "###" */
        case NPY_FR_ns:
            len += 3;  /* "###" */
        case NPY_FR_us:
            len += 3;  /* "###" */
        case NPY_FR_ms:
            len += 4;  /* ".###" */
        case NPY_FR_s:
            len += 3;  /* ":##" */
        case NPY_FR_m:
            len += 3;  /* ":##" */
        case NPY_FR_h:
            len += 3;  /* "T##" */
        case NPY_FR_D:
        case NPY_FR_W:
            len += 3;  /* "-##" */
        case NPY_FR_M:
            len += 3;  /* "-##" */
        case NPY_FR_Y:
            len += 21; /* 64-bit year */
            break;
    }

    if (base >= NPY_FR_h) {
        if (local) {
            len += 5;  /* "+####" or "-####" */
        }
        else {
            len += 1;  /* "Z" */
        }
    }

    len += 1; /* 空字符终止符 */

    return len;
}



/*
 * Finds the largest unit whose value is nonzero, and for which
 * the remainder for the rest of the units is zero.
 */
static NPY_DATETIMEUNIT
lossless_unit_from_datetimestruct(npy_datetimestruct *dts)
{
    if (dts->as % 1000 != 0) {
        return NPY_FR_as;
    }
    else if (dts->as != 0) {
        return NPY_FR_fs;
    }
    else if (dts->ps % 1000 != 0) {
        return NPY_FR_ps;
    }
    else if (dts->ps != 0) {
        return NPY_FR_ns;
    }
    else if (dts->us % 1000 != 0) {
        return NPY_FR_us;
    }
    else if (dts->us != 0) {
        return NPY_FR_ms;
    }
    else if (dts->sec != 0) {
        return NPY_FR_s;
    }
    else if (dts->min != 0) {
        return NPY_FR_m;
    }
    else if (dts->hour != 0) {
        return NPY_FR_h;
    }
    else if (dts->day != 1) {
        return NPY_FR_D;
    }
    else if (dts->month != 1) {
        return NPY_FR_M;
    }
    else {
        return NPY_FR_Y;
    }
}


/*
 * 找到数值非零的最大单位，并且其余单位的余数为零。
 */
static NPY_DATETIMEUNIT
lossless_unit_from_datetimestruct(npy_datetimestruct *dts)
{
    if (dts->as % 1000 != 0) {
        return NPY_FR_as;
    }
    else if (dts->as != 0) {
        return NPY_FR_fs;
    }
    else if (dts->ps % 1000 != 0) {
        return NPY_FR_ps;
    }
    else if (dts->ps != 0) {
        return NPY_FR_ns;
    }
    else if (dts->us % 1000 != 0) {
        return NPY_FR_us;
    }
    else if (dts->us != 0) {
        return NPY_FR_ms;
    }
    else if (dts->sec != 0) {
        return NPY_FR_s;
    }
    else if (dts->min != 0) {
        return NPY_FR_m;
    }
    else if (dts->hour != 0) {
        return NPY_FR_h;
    }
    else if (dts->day != 1) {
        return NPY_FR_D;
    }
    else if (dts->month != 1) {
        return NPY_FR_M;
    }
    else {
        return NPY_FR_Y;
    }
}
/*
 * Converts an npy_datetimestruct to an (almost) ISO 8601
 * NULL-terminated string. If the string fits in the space exactly,
 * it leaves out the NULL terminator and returns success.
 *
 * The differences from ISO 8601 are the 'NaT' string, and
 * the number of year digits is >= 4 instead of strictly 4.
 *
 * If 'local' is non-zero, it produces a string in local time with
 * a +-#### timezone offset. If 'local' is zero and 'utc' is non-zero,
 * produce a string ending with 'Z' to denote UTC. By default, no time
 * zone information is attached.
 *
 * 'base' restricts the output to that unit. Set 'base' to
 * -1 to auto-detect a base after which all the values are zero.
 *
 * 'tzoffset' is used if 'local' is enabled, and 'tzoffset' is
 * set to a value other than -1. This is a manual override for
 * the local time zone to use, as an offset in minutes.
 *
 * 'casting' controls whether data loss is allowed by truncating
 * the data to a coarser unit. This interacts with 'local', slightly,
 * in order to form a date unit string as a local time, the casting
 * must be unsafe.
 *
 * Returns 0 on success, -1 on failure (for example if the output
 * string was too short).
 */
NPY_NO_EXPORT int
NpyDatetime_MakeISO8601Datetime(
        npy_datetimestruct *dts, char *outstr, npy_intp outlen,
        int local, int utc, NPY_DATETIMEUNIT base, int tzoffset,
        NPY_CASTING casting)
{
    npy_datetimestruct dts_local;
    int timezone_offset = 0;

    char *substr = outstr; // 初始化字符串指针
    npy_intp sublen = outlen; // 初始化字符串长度

    /* Handle NaT, and treat a datetime with generic units as NaT */
    if (dts->year == NPY_DATETIME_NAT || base == NPY_FR_GENERIC) {
        if (outlen < 3) { // 如果输出空间不足
            goto string_too_short; // 转到字符串太短的错误处理
        }
        outstr[0] = 'N'; // 设置输出字符串为 NaT
        outstr[1] = 'a';
        outstr[2] = 'T';
        if (outlen > 3) {
            outstr[3] = '\0'; // 添加字符串结束符
        }

        return 0; // 返回成功
    }

    /*
     * Only do local time within a reasonable year range. The years
     * earlier than 1970 are not made local, because the Windows API
     * raises an error when they are attempted (see the comments above the
     * get_localtime() function). For consistency, this
     * restriction is applied to all platforms.
     *
     * Note that this only affects how the datetime becomes a string.
     * The result is still completely unambiguous, it only means
     * that datetimes outside this range will not include a time zone
     * when they are printed.
     */
    if ((dts->year < 1970 || dts->year >= 10000) && tzoffset == -1) {
        local = 0; // 如果年份早于1970或者大于等于10000且没有指定时区偏移，则不使用本地时间
    }

    /* Automatically detect a good unit */
    # 如果基础单位为错误，根据日期时间结构生成最适合的单位
    if (base == NPY_FR_ERROR) {
        base = lossless_unit_from_datetimestruct(dts);
        
        /*
         * 如果有时区信息，则至少使用分钟精度，
         * 默认情况下不会将小时和分钟分开
         */
        if ((base < NPY_FR_m && local) || base == NPY_FR_h) {
            base = NPY_FR_m;
        }
        /* 默认情况下不会分开日期 */
        else if (base < NPY_FR_D) {
            base = NPY_FR_D;
        }
    }
    
    /*
     * 以与天相同的精度打印周。
     *
     * TODO: 如果周的起始日期是星期一，可以使用 YYYY-Www 格式打印周
     */
    else if (base == NPY_FR_W) {
        base = NPY_FR_D;
    }

    /* 使用 C API 将时间从 UTC 转换为本地时间 */
    if (local && tzoffset == -1) {
        if (convert_datetimestruct_utc_to_local(&dts_local, dts,
                                                &timezone_offset) < 0) {
            return -1;
        }

        /* 将 dts 指向本地时间，而不是 UTC 时间 */
        dts = &dts_local;
    }
    /* 使用手动提供的时区偏移量 */
    else if (local) {
        /* 创建 dts 的副本，以便进行修改 */
        dts_local = *dts;
        dts = &dts_local;

        /* 设置并应用所需的时区偏移量 */
        timezone_offset = tzoffset;
        add_minutes_to_datetimestruct(dts, timezone_offset);
    }

    /*
     * 现在 datetimestruct 数据已经是最终形式，用于字符串表示，
     * 因此确保根据转换规则进行类型转换。
     */
    if (casting != NPY_UNSAFE_CASTING) {
        /* 生成本地时间的日期作为字符串总是 'unsafe' */
        if (base <= NPY_FR_D && local) {
            PyErr_SetString(PyExc_TypeError, "Cannot create a local "
                        "timezone-based date string from a NumPy "
                        "datetime without forcing 'unsafe' casting");
            return -1;
        }
        /* 只有 'unsafe' 和 'same_kind' 允许数据丢失 */
        else {
            NPY_DATETIMEUNIT unitprec;

            unitprec = lossless_unit_from_datetimestruct(dts);
            if (casting != NPY_SAME_KIND_CASTING && unitprec > base) {
                PyErr_Format(PyExc_TypeError, "Cannot create a "
                            "string with unit precision '%s' "
                            "from the NumPy datetime, which has data at "
                            "unit precision '%s', "
                            "requires 'unsafe' or 'same_kind' casting",
                             _datetime_strings[base],
                             _datetime_strings[unitprec]);
                return -1;
            }
        }
    }

    /* 年 */
    /*
     * 不能使用 PyOS_snprintf，因为它总是在末尾生成 '\0' 字符，
     * 而 NumPy 字符串类型允许数据一直到缓冲区的末尾。
     */
#ifdef _WIN32
    // 在 Windows 平台下使用 _snprintf 函数格式化年份到 substr 中
    tmplen = _snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#else
    // 在非 Windows 平台下使用 snprintf 函数格式化年份到 substr 中
    tmplen = snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#endif

    /* 如果字符串空间不足或者没有空间放置 NULL 终止符 */
    if (tmplen < 0 || tmplen > sublen) {
        // 如果出现空间不足的情况，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }

    // 更新 substr 指针位置，减少剩余空间长度
    substr += tmplen;
    sublen -= tmplen;

    /* 如果时间单位为年，则直接结束 */
    if (base == NPY_FR_Y) {
        // 如果剩余空间足够，添加字符串结尾符 '\0'
        if (sublen > 0) {
            *substr = '\0';
        }
        // 返回成功状态
        return 0;
    }

    /* 月份 */
    if (sublen < 1 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加 '-' 字符
    substr[0] = '-';
    if (sublen < 2 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加月份的十位数字
    substr[1] = (char)((dts->month / 10) + '0');
    if (sublen < 3 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加月份的个位数字
    substr[2] = (char)((dts->month % 10) + '0');
    // 更新 substr 指针位置，减少剩余空间长度
    substr += 3;
    sublen -= 3;

    /* 如果时间单位为月，则直接结束 */
    if (base == NPY_FR_M) {
        // 如果剩余空间足够，添加字符串结尾符 '\0'
        if (sublen > 0) {
            *substr = '\0';
        }
        // 返回成功状态
        return 0;
    }

    /* 天数 */
    if (sublen < 1 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加 '-' 字符
    substr[0] = '-';
    if (sublen < 2 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加天数的十位数字
    substr[1] = (char)((dts->day / 10) + '0');
    if (sublen < 3 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加天数的个位数字
    substr[2] = (char)((dts->day % 10) + '0');
    // 更新 substr 指针位置，减少剩余空间长度
    substr += 3;
    sublen -= 3;

    /* 如果时间单位为天，则直接结束 */
    if (base == NPY_FR_D) {
        // 如果剩余空间足够，添加字符串结尾符 '\0'
        if (sublen > 0) {
            *substr = '\0';
        }
        // 返回成功状态
        return 0;
    }

    /* 小时 */
    if (sublen < 1 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加 'T' 字符，表示时间的开始
    substr[0] = 'T';
    if (sublen < 2 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加小时的十位数字
    substr[1] = (char)((dts->hour / 10) + '0');
    if (sublen < 3 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加小时的个位数字
    substr[2] = (char)((dts->hour % 10) + '0');
    // 更新 substr 指针位置，减少剩余空间长度
    substr += 3;
    sublen -= 3;

    /* 如果时间单位为小时，则跳转到标签 add_time_zone 处处理 */
    if (base == NPY_FR_h) {
        goto add_time_zone;
    }

    /* 分钟 */
    if (sublen < 1 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加 ':' 字符，表示小时和分钟的分隔符
    substr[0] = ':';
    if (sublen < 2 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加分钟的十位数字
    substr[1] = (char)((dts->min / 10) + '0');
    if (sublen < 3 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加分钟的个位数字
    substr[2] = (char)((dts->min % 10) + '0');
    // 更新 substr 指针位置，减少剩余空间长度
    substr += 3;
    sublen -= 3;

    /* 如果时间单位为分钟，则跳转到标签 add_time_zone 处处理 */
    if (base == NPY_FR_m) {
        goto add_time_zone;
    }

    /* 秒数 */
    if (sublen < 1 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加 ':' 字符，表示分钟和秒数的分隔符
    substr[0] = ':';
    if (sublen < 2 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加秒数的十位数字
    substr[1] = (char)((dts->sec / 10) + '0');
    if (sublen < 3 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加秒数的个位数字
    substr[2] = (char)((dts->sec % 10) + '0');
    // 更新 substr 指针位置，减少剩余空间长度
    substr += 3;
    sublen -= 3;

    /* 如果时间单位为秒，则跳转到标签 add_time_zone 处处理 */
    if (base == NPY_FR_s) {
        goto add_time_zone;
    }

    /* 毫秒 */
    if (sublen < 1 ) {
        // 如果剩余空间不足，则跳转到标签 string_too_short 处处理
        goto string_too_short;
    }
    // 在 substr 中添加 '.' 字符，表示秒数和毫秒的分隔符
    substr[0] = '.';
    // 检查子字符串长度是否小于2，如果是则跳转到标签 string_too_short
    if (sublen < 2 ) {
        goto string_too_short;
    }
    // 将微秒部分的百万位数转换为字符存入 substr[1]
    substr[1] = (char)((dts->us / 100000) % 10 + '0');
    // 检查子字符串长度是否小于3，如果是则跳转到标签 string_too_short
    if (sublen < 3 ) {
        goto string_too_short;
    }
    // 将微秒部分的十万位数转换为字符存入 substr[2]
    substr[2] = (char)((dts->us / 10000) % 10 + '0');
    // 检查子字符串长度是否小于4，如果是则跳转到标签 string_too_short
    if (sublen < 4 ) {
        goto string_too_short;
    }
    // 将微秒部分的万位数到个位数转换为字符存入 substr[3]，并更新 substr 和 sublen
    substr[3] = (char)((dts->us / 1000) % 10 + '0');
    substr += 4;
    sublen -= 4;

    /* 如果时间单位是毫秒，则跳转到标签 add_time_zone */
    if (base == NPY_FR_ms) {
        goto add_time_zone;
    }

    /* 微秒部分 */
    // 检查子字符串长度是否小于1，如果是则跳转到标签 string_too_short
    if (sublen < 1 ) {
        goto string_too_short;
    }
    // 将微秒部分的百位数转换为字符存入 substr[0]
    substr[0] = (char)((dts->us / 100) % 10 + '0');
    // 检查子字符串长度是否小于2，如果是则跳转到标签 string_too_short
    if (sublen < 2 ) {
        goto string_too_short;
    }
    // 将微秒部分的十位数转换为字符存入 substr[1]
    substr[1] = (char)((dts->us / 10) % 10 + '0');
    // 检查子字符串长度是否小于3，如果是则跳转到标签 string_too_short
    if (sublen < 3 ) {
        goto string_too_short;
    }
    // 将微秒部分的个位数转换为字符存入 substr[2]，并更新 substr 和 sublen
    substr[2] = (char)(dts->us % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* 如果时间单位是微秒，则跳转到标签 add_time_zone */
    if (base == NPY_FR_us) {
        goto add_time_zone;
    }

    /* 纳秒部分 */
    // 检查子字符串长度是否小于1，如果是则跳转到标签 string_too_short
    if (sublen < 1 ) {
        goto string_too_short;
    }
    // 将纳秒部分的十亿位数转换为字符存入 substr[0]
    substr[0] = (char)((dts->ps / 100000) % 10 + '0');
    // 检查子字符串长度是否小于2，如果是则跳转到标签 string_too_short
    if (sublen < 2 ) {
        goto string_too_short;
    }
    // 将纳秒部分的百万位数转换为字符存入 substr[1]
    substr[1] = (char)((dts->ps / 10000) % 10 + '0');
    // 检查子字符串长度是否小于3，如果是则跳转到标签 string_too_short
    if (sublen < 3 ) {
        goto string_too_short;
    }
    // 将纳秒部分的千位数转换为字符存入 substr[2]，并更新 substr 和 sublen
    substr[2] = (char)((dts->ps / 1000) % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* 如果时间单位是纳秒，则跳转到标签 add_time_zone */
    if (base == NPY_FR_ns) {
        goto add_time_zone;
    }

    /* 皮秒部分 */
    // 检查子字符串长度是否小于1，如果是则跳转到标签 string_too_short
    if (sublen < 1 ) {
        goto string_too_short;
    }
    // 将皮秒部分的十万亿位数转换为字符存入 substr[0]
    substr[0] = (char)((dts->ps / 100) % 10 + '0');
    // 检查子字符串长度是否小于2，如果是则跳转到标签 string_too_short
    if (sublen < 2 ) {
        goto string_too_short;
    }
    // 将皮秒部分的十亿位数转换为字符存入 substr[1]
    substr[1] = (char)((dts->ps / 10) % 10 + '0');
    // 检查子字符串长度是否小于3，如果是则跳转到标签 string_too_short
    if (sublen < 3 ) {
        goto string_too_short;
    }
    // 将皮秒部分的个位数转换为字符存入 substr[2]，并更新 substr 和 sublen
    substr[2] = (char)(dts->ps % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* 如果时间单位是皮秒，则跳转到标签 add_time_zone */
    if (base == NPY_FR_ps) {
        goto add_time_zone;
    }

    /* 飞秒部分 */
    // 检查子字符串长度是否小于1，如果是则跳转到标签 string_too_short
    if (sublen < 1 ) {
        goto string_too_short;
    }
    // 将飞秒部分的百万亿亿位数转换为字符存入 substr[0]
    substr[0] = (char)((dts->as / 100000) % 10 + '0');
    // 检查子字符串长度是否小于2，如果是则跳转到标签 string_too_short
    if (sublen < 2 ) {
        goto string_too_short;
    }
    // 将飞秒部分的十亿亿位数转换为字符存入 substr[1]
    substr[1] = (char)((dts->as / 10000) % 10 + '0');
    // 检查子字符串长度是否小于3，如果是则跳转到标签 string_too_short
    if (sublen < 3 ) {
        goto string_too_short;
    }
    // 将飞秒部分的亿位数转换为字符存入 substr[2]，并更新 substr 和 sublen
    substr[2] = (char)((dts->as / 1000) % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* 如果时间单位是飞秒，则跳转到标签 add_time_zone */
    if (base == NPY_FR_fs) {
        goto add_time_zone;
    }

    /* 阿托秒部分 */
    // 检查子字符串长度是否小于1，如果是则跳转到标签 string_too_short
    if (sublen < 1 ) {
        goto string_too_short;
    }
    // 将阿托秒部分的百亿亿位数转换为字符存入 substr[0]
    substr[0] = (char)((dts->as / 100) % 10 + '0');
    // 检查子字符串长度是否小于2，如果是则跳转到标签 string_too_short
    if (sublen < 2 ) {
        goto string_too_short;
    }
    // 将阿托秒部分的十亿亿位数转换为字符存入 substr[1]
    substr[1] = (char)((dts->as / 10) % 10 + '0');
    // 检查子字符串长度是否小于3，如果是则跳转到标签 string_too_short
    if (sublen < 3 ) {
        goto string_too_short;
    }
    // 将阿托秒部分的亿位数转换为字符存入 substr[2]，并更新 substr
add_time_zone:
    if (local) {
        /* 添加正负号 */
        if (sublen < 1) {
            goto string_too_short;
        }
        /* 如果时区偏移小于0，添加负号 */
        if (timezone_offset < 0) {
            substr[0] = '-';
            timezone_offset = -timezone_offset;
        }
        else {
            /* 否则添加正号 */
            substr[0] = '+';
        }
        substr += 1;  // 指向下一个字符位置
        sublen -= 1;  // 剩余字符长度减1

        /* 添加时区偏移 */
        if (sublen < 1 ) {
            goto string_too_short;
        }
        /* 添加小时位 */
        substr[0] = (char)((timezone_offset / (10*60)) % 10 + '0');
        if (sublen < 2 ) {
            goto string_too_short;
        }
        /* 添加十位分钟 */
        substr[1] = (char)((timezone_offset / 60) % 10 + '0');
        if (sublen < 3 ) {
            goto string_too_short;
        }
        /* 添加个位分钟 */
        substr[2] = (char)(((timezone_offset % 60) / 10) % 10 + '0');
        if (sublen < 4 ) {
            goto string_too_short;
        }
        /* 添加个位秒 */
        substr[3] = (char)((timezone_offset % 60) % 10 + '0');
        substr += 4;  // 指向下一个字符位置
        sublen -= 4;  // 剩余字符长度减4
    }
    /* UTC "Zulu" 时间 */
    else if (utc) {
        if (sublen < 1) {
            goto string_too_short;
        }
        substr[0] = 'Z';  // 添加UTC标识符
        substr += 1;  // 指向下一个字符位置
        sublen -= 1;  // 剩余字符长度减1
    }

    /* 添加空字符终止符，并返回 */
    if (sublen > 0) {
        substr[0] = '\0';  // 添加空字符终止符
    }

    return 0;

string_too_short:
    PyErr_Format(PyExc_RuntimeError,
                "The string provided for NumPy ISO datetime formatting "
                "was too short, with length %"NPY_INTP_FMT,
                outlen);
    return -1;
}

/*
 * 这是 Python 中公开的 datetime_as_string 函数。
 */
NPY_NO_EXPORT PyObject *
array_datetime_as_string(PyObject *NPY_UNUSED(self), PyObject *args,
                                PyObject *kwds)
{
    PyObject *arr_in = NULL, *unit_in = NULL, *timezone_obj = NULL;
    NPY_DATETIMEUNIT unit;
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;

    int local = 0;  // 本地时间标志
    int utc = 0;    // UTC时间标志
    PyArray_DatetimeMetaData *meta;
    int strsize;

    PyArrayObject *ret = NULL;

    NpyIter *iter = NULL;
    PyArrayObject *op[2] = {NULL, NULL};
    PyArray_Descr *op_dtypes[2] = {NULL, NULL};
    npy_uint32 flags, op_flags[2];

    static char *kwlist[] = {"arr", "unit", "timezone", "casting", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds,
                                "O|OOO&:datetime_as_string", kwlist,
                                &arr_in,
                                &unit_in,
                                &timezone_obj,
                                &PyArray_CastingConverter, &casting)) {
        return NULL;
    }

    /* 保留时区的引用供后续使用 */
    Py_XINCREF(timezone_obj);

    op[0] = (PyArrayObject *)PyArray_FROM_O(arr_in);
    if (op[0] == NULL) {
        goto fail;
    }
    /* 确保输入的类型为 NumPy datetime */
    if (PyArray_DESCR(op[0])->type_num != NPY_DATETIME) {
        PyErr_SetString(PyExc_TypeError,
                    "input must have type NumPy datetime");
        goto fail;
    }

    /* 获取 datetime 元数据 */
    meta = get_datetime_metadata_from_dtype(PyArray_DESCR(op[0]));
    # 从 op[0] 的数据类型描述符中获取日期时间元数据
    if (meta == NULL) {
        goto fail;
    }

    /* Use the metadata's unit for printing by default */
    # 默认情况下使用元数据的单位进行打印
    unit = meta->base;

    /* Parse the input unit if provided */
    # 如果提供了输入单位，则解析它
    if (unit_in != NULL && unit_in != Py_None) {
        PyObject *strobj;

        if (PyBytes_Check(unit_in)) {
            /* accept bytes input */
            # 接受字节输入
            PyObject *obj_str = PyUnicode_FromEncodedObject(unit_in, NULL, NULL);
            if (obj_str == NULL) {
                return 0;
            }
            strobj = obj_str;
        }
        else {
            Py_INCREF(unit_in);
            strobj = unit_in;
        }

        Py_ssize_t len;
        char const *str = PyUnicode_AsUTF8AndSize(strobj, &len);
        if (str == NULL) {
            Py_DECREF(strobj);
            goto fail;
        }

        /*
         * unit == NPY_FR_ERROR means to autodetect the unit
         * from the datetime data
         * */
        # 如果单位为 NPY_FR_ERROR，则从日期时间数据中自动检测单位
        if (strcmp(str, "auto") == 0) {
            unit = NPY_FR_ERROR;
        }
        else {
            unit = parse_datetime_unit_from_string(str, len, NULL);
            if (unit == NPY_FR_ERROR) {
                Py_DECREF(strobj);
                goto fail;
            }
        }
        Py_DECREF(strobj);

        if (unit != NPY_FR_ERROR &&
                !can_cast_datetime64_units(meta->base, unit, casting)) {
            PyErr_Format(PyExc_TypeError, "Cannot create a datetime "
                        "string as units '%s' from a NumPy datetime "
                        "with units '%s' according to the rule %s",
                        _datetime_strings[unit],
                        _datetime_strings[meta->base],
                         npy_casting_to_string(casting));
            goto fail;
        }
    }

    /* Get the input time zone */
    # 获取输入的时区
    // 检查时区对象是否非空
    if (timezone_obj != NULL) {
        PyObject *strobj;
        // 如果时区对象是字节对象
        if (PyBytes_Check(timezone_obj)) {
            /* accept bytes input */
            // 将字节对象转换为Unicode字符串对象
            PyObject *obj_str = PyUnicode_FromEncodedObject(timezone_obj, NULL, NULL);
            if (obj_str == NULL) {
                // 转换失败时跳转到错误处理
                goto fail;
            }
            strobj = obj_str;
        }
        else {
            // 增加时区对象的引用计数
            Py_INCREF(timezone_obj);
            strobj = timezone_obj;
        }

        // 将strobj赋值给timezone_obj，并释放之前的引用
        Py_SETREF(timezone_obj, strobj);

        /* Check for the supported string inputs */
        // 检查时区对象是否是Unicode字符串
        if (PyUnicode_Check(timezone_obj)) {
            Py_ssize_t len;
            // 获取Unicode字符串的UTF-8表示及其长度
            char const *str = PyUnicode_AsUTF8AndSize(timezone_obj, &len);
            if (str == NULL) {
                // 获取失败时跳转到错误处理
                goto fail;
            }

            // 根据字符串内容设置local和utc标志位
            if (strcmp(str, "local") == 0) {
                local = 1;
                utc = 0;
                Py_DECREF(timezone_obj);
                timezone_obj = NULL;
            }
            else if (strcmp(str, "UTC") == 0) {
                local = 0;
                utc = 1;
                Py_DECREF(timezone_obj);
                timezone_obj = NULL;
            }
            else if (strcmp(str, "naive") == 0) {
                local = 0;
                utc = 0;
                Py_DECREF(timezone_obj);
                timezone_obj = NULL;
            }
            else {
                // 抛出值错误异常，指示不支持的时区字符串输入
                PyErr_Format(PyExc_ValueError, "Unsupported timezone "
                            "input string \"%s\"", str);
                goto fail;
            }
        }
        /* Otherwise assume it's a Python TZInfo, or acts like one */
        else {
            // 否则认为是Python的TZInfo对象或类似对象，设置local标志位
            local = 1;
        }
    }

    /* Get a string size long enough for any datetimes we're given */
    // 获取一个足够长的字符串大小以适应任何给定的日期时间
    strsize = NpyDatetime_GetDatetimeISO8601StrLen(local, unit);
    /*
     * For Python3, allocate the output array as a UNICODE array, so
     * that it will behave as strings properly
     */
    // 对于Python3，分配输出数组作为UNICODE数组，以便它正确地作为字符串处理
    op_dtypes[1] = PyArray_DescrNewFromType(NPY_UNICODE);
    if (op_dtypes[1] == NULL) {
        // 分配失败时跳转到错误处理
        goto fail;
    }
    op_dtypes[1]->elsize = strsize * 4;
    /* This steals the UNICODE dtype reference in op_dtypes[1] */
    // 这里窃取op_dtypes[1]中的UNICODE dtype引用
    op[1] = (PyArrayObject *)PyArray_NewLikeArray(op[0],
                                        NPY_KEEPORDER, op_dtypes[1], 1);
    if (op[1] == NULL) {
        op_dtypes[1] = NULL;
        goto fail;
    }
    /* Create the iteration string data type (always ASCII string) */
    // 创建迭代字符串数据类型（始终为ASCII字符串）
    op_dtypes[1] = PyArray_DescrNewFromType(NPY_STRING);
    if (op_dtypes[1] == NULL) {
        // 创建失败时跳转到错误处理
        goto fail;
    }
    op_dtypes[1]->elsize = strsize;

    // 设置迭代器标志和操作标志
    flags = NPY_ITER_ZEROSIZE_OK|
            NPY_ITER_BUFFERED;
    op_flags[0] = NPY_ITER_READONLY|
                  NPY_ITER_ALIGNED;
    op_flags[1] = NPY_ITER_WRITEONLY|
                  NPY_ITER_ALLOCATE;

    // 创建多重迭代器对象
    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        // 创建失败时跳转到错误处理
        goto fail;
    }
    if (NpyIter_GetIterSize(iter) != 0) {
        // 获取迭代器的大小，如果不为0，则执行以下操作
        NpyIter_IterNextFunc *iternext;
        // 声明迭代器的下一个函数指针
        char **dataptr;
        // 声明数据指针的数组
        npy_datetime dt;
        // 声明numpy的日期时间变量
        npy_datetimestruct dts;
        // 声明numpy的日期时间结构体变量

        iternext = NpyIter_GetIterNext(iter, NULL);
        // 获取迭代器的下一个函数指针，传入NULL表示不需要传播错误
        if (iternext == NULL) {
            // 如果获取迭代器的下一个函数指针失败，则跳转到错误处理部分
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        // 获取迭代器的数据指针数组

        do {
            int tzoffset = -1;
            // 声明时区偏移量变量并初始化为-1

            /* Get the datetime */
            // 获取日期时间
            dt = *(npy_datetime *)dataptr[0];

            /* Convert it to a struct */
            // 将其转换为结构体
            if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(meta, dt, &dts) < 0) {
                // 如果转换日期时间失败，则跳转到错误处理部分
                goto fail;
            }

            /* Get the tzoffset from the timezone if provided */
            // 如果提供了本地化和时区对象，则获取时区偏移量
            if (local && timezone_obj != NULL) {
                tzoffset = get_tzoffset_from_pytzinfo(timezone_obj, &dts);
                // 调用函数获取时区偏移量
                if (tzoffset == -1) {
                    // 如果获取时区偏移量失败，则跳转到错误处理部分
                    goto fail;
                }
            }

            /* Zero the destination string completely */
            // 将目标字符串完全置零
            memset(dataptr[1], 0, strsize);

            /* Convert that into a string */
            // 将其转换为字符串
            if (NpyDatetime_MakeISO8601Datetime(&dts, (char *)dataptr[1], strsize,
                                local, utc, unit, tzoffset, casting) < 0) {
                // 如果转换为ISO8601格式日期时间字符串失败，则跳转到错误处理部分
                goto fail;
            }
        } while(iternext(iter));
        // 使用迭代器的下一个函数处理所有数据

    }

    ret = NpyIter_GetOperandArray(iter)[1];
    // 获取迭代器的操作数数组中的第二个操作数
    Py_INCREF(ret);
    // 增加返回数组的引用计数

    Py_XDECREF(timezone_obj);
    // 释放时区对象的引用
    Py_XDECREF(op[0]);
    // 释放操作数数组中的第一个操作数的引用
    Py_XDECREF(op[1]);
    // 释放操作数数组中的第二个操作数的引用
    Py_XDECREF(op_dtypes[0]);
    // 释放操作数据类型数组中的第一个元素的引用
    Py_XDECREF(op_dtypes[1]);
    // 释放操作数据类型数组中的第二个元素的引用
    if (iter != NULL) {
        // 如果迭代器不为NULL，则释放迭代器
        NpyIter_Deallocate(iter);
    }

    return PyArray_Return(ret);
    // 返回增加了引用计数的返回数组
fail:
    // 释放引用计数并清理内存：释放时区对象的引用
    Py_XDECREF(timezone_obj);
    // 释放引用计数并清理内存：释放操作数数组中的第一个操作数对象
    Py_XDECREF(op[0]);
    // 释放引用计数并清理内存：释放操作数数组中的第二个操作数对象
    Py_XDECREF(op[1]);
    // 释放引用计数并清理内存：释放操作数数据类型数组中的第一个数据类型对象
    Py_XDECREF(op_dtypes[0]);
    // 释放引用计数并清理内存：释放操作数数据类型数组中的第二个数据类型对象
    Py_XDECREF(op_dtypes[1]);
    // 如果迭代器对象不为空，则释放迭代器对象的内存
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    // 函数执行失败，返回空指针（NULL）
    return NULL;
}
```