# `D:\src\scipysrc\pandas\pandas\_libs\src\vendored\numpy\datetime\np_datetime_strings.c`

```
/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Written by Mark Wiebe (mwwiebe@gmail.com)
Copyright (c) 2011 by Enthought, Inc.

Copyright (c) 2005-2011, NumPy Developers
All rights reserved.

See NUMPY_LICENSE.txt for the license.

This file implements string parsing and creation for NumPy datetime.
*/

// LICENSES/NUMPY_LICENSE

// 定义预处理指令，清除 PY_SSIZE_T 的定义
#define PY_SSIZE_T_CLEAN
// 定义预处理指令，禁止导入
#define NO_IMPORT

// 如果未定义 NPY_NO_DEPRECATED_API，则定义为 NPY_1_7_API_VERSION
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif // NPY_NO_DEPRECATED_API

// 引入 Python 头文件
#include <Python.h>

// 引入时间相关的头文件
#include <time.h>

// 引入 NumPy 相关的头文件
#include <numpy/ndarraytypes.h>
#include <numpy/npy_common.h>

// 引入 pandas 的可移植头文件
#include "pandas/portable.h"
// 引入 pandas vendored 目录下 NumPy datetime 相关头文件
#include "pandas/vendored/numpy/datetime/np_datetime.h"
#include "pandas/vendored/numpy/datetime/np_datetime_strings.h"

/*
 * Parses (almost) standard ISO 8601 date strings. The differences are:
 *
 * + Only seconds may have a decimal point, with up to 18 digits after it
 *   (maximum attoseconds precision).
 * + Either a 'T' as in ISO 8601 or a ' ' may be used to separate
 *   the date and the time. Both are treated equivalently.
 * + Doesn't (yet) handle the "YYYY-DDD" or "YYYY-Www" formats.
 * + Doesn't handle leap seconds (seconds value has 60 in these cases).
 * + Doesn't handle 24:00:00 as synonym for midnight (00:00:00) tomorrow
 * + Accepts special values "NaT" (not a time), "Today", (current
 *   day according to local time) and "Now" (current time in UTC).
 * + ':' separator between hours, minutes, and seconds is optional. When
 *   omitted, each component must be 2 digits if it appears. (GH-10041)
 *
 * 'str' must be a NULL-terminated string, and 'len' must be its length.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_local' gets set to 1 if the parsed time contains timezone,
 *      to 0 otherwise.
 * 'out_tzoffset' gets set to timezone offset by minutes
 *      if the parsed time was in local time,
 *      to 0 otherwise. The values 'now' and 'today' don't get counted
 *      as local, and neither do UTC +/-#### timezone offsets, because
 *      they aren't using the computer's local timezone offset.
 *
 * Returns 0 on success, -1 on failure.
 */

// 定义枚举类型，用于标识日期时间部分解析结果
typedef enum {
  COMPARISON_SUCCESS,
  COMPLETED_PARTIAL_MATCH,
  COMPARISON_ERROR
} DatetimePartParseResult;

// 此函数用于比较格式，根据需求更新 format 指针和 characters_remaining
// 如果 `format_requirement` 为 PARTIAL_MATCH，并且 `format` 字符串已经耗尽，则返回 COMPLETED_PARTIAL_MATCH
static DatetimePartParseResult
compare_format(const char **format, int *characters_remaining,
               const char *compare_to, int n,
               const FormatRequirement format_requirement) {
  // 实现比较逻辑，根据不同的需求更新 format 指针和 characters_remaining
  if (format_requirement == INFER_FORMAT) {
    // 省略具体的比较细节
  }

  // 返回比较结果枚举值
  return COMPARISON_ERROR; // 默认返回比较错误
}
    // 如果比较成功，返回比较成功的常量
    return COMPARISON_SUCCESS;
      }
      // 如果剩余字符数小于零，返回比较错误的常量
      if (*characters_remaining < 0) {
        return COMPARISON_ERROR;
      }
      // 如果要求部分匹配且剩余字符数为零，返回部分匹配完成的常量
      if (format_requirement == PARTIAL_MATCH && *characters_remaining == 0) {
        return COMPLETED_PARTIAL_MATCH;
      }
      // 如果剩余字符数小于 n，返回比较错误的常量，并记录错误信息（TODO: pandas-dev）
      if (*characters_remaining < n) {
        // TODO(pandas-dev): PyErr to differentiate what went wrong
        return COMPARISON_ERROR;
      } else {
        // 如果前 n 个字符不匹配，返回比较错误的常量，并记录错误信息（TODO: pandas-dev）
        if (strncmp(*format, compare_to, n)) {
          // TODO(pandas-dev): PyErr to differentiate what went wrong
          return COMPARISON_ERROR;
        } else {
          // 如果前 n 个字符匹配，更新 format 指针和剩余字符数，并返回比较成功的常量
          *format += n;
          *characters_remaining -= n;
          return COMPARISON_SUCCESS;
        }
      }
      // 默认情况下返回比较成功的常量
      return COMPARISON_SUCCESS;
}

int parse_iso_8601_datetime(const char *str, int len, int want_exc,
                            npy_datetimestruct *out,
                            NPY_DATETIMEUNIT *out_bestunit, int *out_local,
                            int *out_tzoffset, const char *format,
                            int format_len,
                            FormatRequirement format_requirement) {
  // 检查输入长度是否为负数或格式长度是否为负数，如果是则跳转到解析错误
  if (len < 0 || format_len < 0)
    goto parse_error;

  int year_leap = 0;  // 初始化闰年标志为非闰年
  int i, numdigits;   // 声明循环计数器和数字位数变量
  const char *substr; // 用于指向字符串的当前子串指针
  int sublen;         // 当前子串的长度
  NPY_DATETIMEUNIT bestunit = NPY_FR_GENERIC;  // 初始化最佳时间单位为通用时间单位
  DatetimePartParseResult comparison;  // 时间部分解析结果比较枚举

  /* 如果年-月-日由有效分隔符分隔，
   * 则会解析没有前导零的月份/日期（虽然不符合ISO 8601标准）。
   * 如果组件没有分隔，则期望为4（YYYY）或8（YYYYMMDD）位数字。
   * 这里禁止6位数字（在其他地方解析为YYMMDD）。
   */
  int has_ymd_sep = 0;  // 检测年月日是否由分隔符分隔的标志
  char ymd_sep = '\0';  // 有效的年月日分隔符
  char valid_ymd_sep[] = {'-', '.', '/', '\\', ' '};  // 有效的年月日分隔符数组
  int valid_ymd_sep_len = sizeof(valid_ymd_sep);  // 有效的年月日分隔符数组长度

  /* 小时-分钟-秒可能由':'分隔，如果不是，则每个组件必须为2位数字。 */
  int has_hms_sep = 0;  // 检测小时分钟秒是否由分隔符分隔的标志
  int hour_was_2_digits = 0;  // 检测小时是否为两位数字的标志

  /* 初始化输出为全部零 */
  memset(out, 0, sizeof(npy_datetimestruct));
  out->month = 1;  // 设置月份为1
  out->day = 1;    // 设置日期为1

  substr = str;  // 设置当前子串指针指向输入字符串
  sublen = len;  // 设置当前子串长度为输入长度

  /* 跳过前导空格 */
  while (sublen > 0 && isspace(*substr)) {
    ++substr;
    --sublen;
    comparison =
        compare_format(&format, &format_len, " ", 1, format_requirement);
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      goto finish;
    }
  }

  /* 处理负号开头的负年 */
  if (*substr == '-') {
    ++substr;
    --sublen;
  }

  if (sublen == 0) {
    goto parse_error;
  }

  /* 解析年份（4位数字） */
  comparison =
      compare_format(&format, &format_len, "%Y", 2, format_requirement);
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    goto finish;
  }

  out->year = 0;  // 初始化年份为0
  if (sublen >= 4 && isdigit(substr[0]) && isdigit(substr[1]) &&
      isdigit(substr[2]) && isdigit(substr[3])) {
    out->year = 1000 * (substr[0] - '0') + 100 * (substr[1] - '0') +
                10 * (substr[2] - '0') + (substr[3] - '0');

    substr += 4;  // 移动子串指针到年份后
    sublen -= 4;  // 减去已解析的年份部分长度
  }

  /* 如果年份以负号开头，则取反年份 */
  if (str[0] == '-') {
    out->year = -out->year;
  }
  /* 检查是否是闰年 */
  year_leap = is_leapyear(out->year);

  /* 下一个字符必须是分隔符、月份的开始，或者字符串的结束 */
  if (sublen == 0) {
    if (out_local != NULL) {
      *out_local = 0;
    }
    if (format_len) {
      goto parse_error;
    }
    bestunit = NPY_FR_Y;
    goto finish;
  }

  if (!isdigit(*substr)) {
    for (i = 0; i < valid_ymd_sep_len; ++i) {
      if (*substr == valid_ymd_sep[i]) {
        break;
      }
    }
    // 如果当前索引 i 等于有效分隔符长度，跳转到解析错误处理
    if (i == valid_ymd_sep_len) {
      goto parse_error;
    }
    // 设置已找到年月日分隔符的标志为真，并记录找到的分隔符
    has_ymd_sep = 1;
    ymd_sep = valid_ymd_sep[i];
    // 移动指针到下一个字符，并减少剩余字符串长度
    ++substr;
    --sublen;

    // 比较当前分隔符与格式要求，并处理比较结果
    comparison =
        compare_format(&format, &format_len, &ymd_sep, 1, format_requirement);
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      goto finish;
    }
    /* 不能有尾随的分隔符 */
    // 如果剩余字符串长度为零或下一个字符不是数字，跳转到解析错误处理
    if (sublen == 0 || !isdigit(*substr)) {
      goto parse_error;
    }
  }

  /* 解析月份 */
  // 比较当前字符串是否匹配 "%m" 月份格式，并处理比较结果
  comparison =
      compare_format(&format, &format_len, "%m", 2, format_requirement);
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    goto finish;
  }
  /* 第一位数字是必需的 */
  // 将当前字符解析为月份的第一位数字，并移动指针
  out->month = (*substr - '0');
  ++substr;
  --sublen;
  /* 如果有分隔符，则第二位数字是可选的 */
  // 如果下一个字符是数字，则将其解析为月份的第二位数字，并移动指针
  if (isdigit(*substr)) {
    out->month = 10 * out->month + (*substr - '0');
    ++substr;
    --sublen;
  } else if (!has_ymd_sep) {
    goto parse_error;
  }
  // 如果解析得到的月份不在有效范围内（1到12），则根据需求抛出异常或跳转到错误处理
  if (out->month < 1 || out->month > 12) {
    if (want_exc) {
      PyErr_Format(PyExc_ValueError,
                   "Month out of range in datetime string \"%s\"", str);
    }
    goto error;
  }

  /* 下一个字符必须是分隔符、日期开始字符或字符串结尾 */
  // 如果剩余字符串长度为零，则认为是最佳单位为 NPY_FR_M
  if (sublen == 0) {
    bestunit = NPY_FR_M;
    /* 禁止 YYYYMM，将其解析为 YYMMDD 由其他函数处理 */
    if (!has_ymd_sep) {
      goto parse_error;
    }
    // 如果还有剩余格式长度，则跳转到解析错误处理
    if (format_len) {
      goto parse_error;
    }
    // 如果存在本地时间信息指针，则清零
    if (out_local != NULL) {
      *out_local = 0;
    }
    goto finish;
  }

  // 如果存在年月日分隔符
  if (has_ymd_sep) {
    /* 必须有分隔符，但不能是尾随的 */
    // 如果当前字符不是年月日分隔符，或者剩余字符串长度为1，则跳转到解析错误处理
    if (*substr != ymd_sep || sublen == 1) {
      goto parse_error;
    }
    ++substr;
    --sublen;
    // 比较当前字符与年月日分隔符，并处理比较结果
    comparison =
        compare_format(&format, &format_len, &ymd_sep, 1, format_requirement);
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      goto finish;
    }
  }

  /* 解析日期 */
  // 比较当前字符串是否匹配 "%d" 日期格式，并处理比较结果
  comparison =
      compare_format(&format, &format_len, "%d", 2, format_requirement);
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    goto finish;
  }
  /* 第一位数字是必需的 */
  // 如果当前字符不是数字，则跳转到解析错误处理
  if (!isdigit(*substr)) {
    goto parse_error;
  }
  // 将当前字符解析为日期的第一位数字，并移动指针
  out->day = (*substr - '0');
  ++substr;
  --sublen;
  /* 如果有分隔符，则第二位数字是可选的 */
  // 如果下一个字符是数字，则将其解析为日期的第二位数字，并移动指针
  if (isdigit(*substr)) {
    out->day = 10 * out->day + (*substr - '0');
    ++substr;
    --sublen;
  } else if (!has_ymd_sep) {
    goto parse_error;
  }
  // 如果解析得到的日期不在有效范围内，则根据需求抛出异常或跳转到错误处理
  if (out->day < 1 ||
      out->day > days_per_month_table[year_leap][out->month - 1]) {
    if (want_exc) {
      PyErr_Format(PyExc_ValueError,
                   "Day out of range in datetime string \"%s\"", str);
    }
    goto error;
  }

  /* 下一个字符必须是 'T'、空格或字符串结尾 */
  // 如果剩余字符串长度为零，则认为最佳单位为 NPY_FR_M
  if (sublen == 0) {

    bestunit = NPY_FR_M;
    /* 禁止 YYYYMM，将其解析为 YYMMDD 由其他函数处理 */
    if (!has_ymd_sep) {
      goto parse_error;
    }
    // 如果还有剩余格式长度，则跳转到解析错误处理
    if (format_len) {
      goto parse_error;
    }
    // 如果存在本地时间信息指针，则清零
    if (out_local != NULL) {
      *out_local = 0;
    }
    goto finish;
  }

  // 如果存在年月日分隔符
  if (has_ymd_sep) {
    /* 必须有分隔符，但不能是尾随的 */
    // 如果当前字符不是年月日分隔符，或者剩余字符串长度为1，则跳转到解析错误处理
    if (*substr != ymd_sep || sublen == 1) {
      goto parse_error;
    }
    ++substr;
    --sublen;
    // 比较当前字符与年月日分隔符，并处理比较结果
    comparison =
        compare_format(&format, &format_len, &ymd_sep, 1, format_requirement);
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      goto finish;
    }
  }

  /* 解析日期 */
  // 比较当前字符串是否匹配 "%d" 日期格式，并处理比较结果
  comparison =
      compare_format(&format, &format_len, "%d", 2, format_requirement);
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    goto finish;
  }
  /* 第一位数字是必需的 */
  // 如果当前字符不是数字，则跳转到解析错误处理
  if (!isdigit(*substr)) {
    goto parse_error;
  }
  // 将当前字符解析为日期的第一位数字，并移动指针
  out->day = (*substr - '0');
  ++substr;
  --sublen;
  /* 如果有分隔符，则第二位数字是可选的 */
  // 如果下一个字符是数字，则将其解析为日期的第二位数字，并移动指针
  if (isdigit(*substr)) {
    out->day = 10 * out->day + (*substr - '0');
    ++substr;
    --sublen;
  } else if (!has_ymd_sep) {
    goto parse_error;
  }
  // 如果解析得到的日期不在有效范围内，则根据需求抛出异常或跳转到错误处理
  if (out->day < 1 ||
      out->day > days_per_month_table[year_leap][out->month - 1]) {
    if (want_exc) {
      PyErr_Format(PyExc_ValueError,
                   "Day out of range in datetime string \"%s\"", str);
    }
    goto error;
  }

  /* 下一个字符必须是 'T'、空格或字符串结尾 */
  // 如果剩余字符串长度为零，则认为最佳单位为 NPY_FR_M
  if (sublen == 0) {


注释完毕。
    // 如果 out_local 不为 NULL，则将其值设为 0
    if (out_local != NULL) {
      *out_local = 0;
    }
    // 如果 format_len 不为 0，则跳转到 parse_error 标签处
    if (format_len) {
      goto parse_error;
    }
    // 将 bestunit 设为 NPY_FR_D，并跳转到 finish 标签处
    bestunit = NPY_FR_D;
    goto finish;
  }

  // 如果 *substr 不是 'T' 或 ' '，或者 sublen 等于 1，则跳转到 parse_error 标签处
  if ((*substr != 'T' && *substr != ' ') || sublen == 1) {
    goto parse_error;
  }
  // 调用 compare_format 函数比较 format 和 substr 的格式，根据返回值决定后续动作
  comparison =
      compare_format(&format, &format_len, substr, 1, format_requirement);
  // 如果比较结果为 COMPARISON_ERROR，则跳转到 parse_error 标签处
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    // 如果比较结果为 COMPLETED_PARTIAL_MATCH，则跳转到 finish 标签处
    goto finish;
  }
  // substr 指针向后移动一位，sublen 减一

  /* PARSE THE HOURS */
  // 调用 compare_format 函数比较 format 和 "%H" 的格式，根据返回值决定后续动作
  comparison =
      compare_format(&format, &format_len, "%H", 2, format_requirement);
  // 如果比较结果为 COMPARISON_ERROR，则跳转到 parse_error 标签处
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    // 如果比较结果为 COMPLETED_PARTIAL_MATCH，则跳转到 finish 标签处
    goto finish;
  }
  /* First digit required */
  // 如果 *substr 不是数字，则跳转到 parse_error 标签处
  if (!isdigit(*substr)) {
    goto parse_error;
  }
  // 将 out->hour 设为 *substr 对应的数字
  out->hour = (*substr - '0');
  // 将 bestunit 设为 NPY_FR_h
  bestunit = NPY_FR_h;
  // substr 指针向后移动一位，sublen 减一
  ++substr;
  --sublen;
  /* Second digit optional */
  // 如果 *substr 是数字，则将 out->hour 更新为两位数的小时值，并进行相应的移动和更新操作
  if (isdigit(*substr)) {
    hour_was_2_digits = 1;
    out->hour = 10 * out->hour + (*substr - '0');
    ++substr;
    --sublen;
    // 如果小时值超过 23，则根据 want_exc 标志抛出异常或跳转到 error 标签处
    if (out->hour >= 24) {
      if (want_exc) {
        PyErr_Format(PyExc_ValueError,
                     "Hours out of range in datetime string \"%s\"", str);
      }
      goto error;
    }
  }

  /* Next character must be a ':' or the end of the string */
  // 如果 sublen 为 0，则检查是否小时值是两位数，以及 format_len 是否为 0，根据情况跳转到相应标签处
  if (sublen == 0) {
    if (!hour_was_2_digits) {
      goto parse_error;
    }
    if (format_len) {
      goto parse_error;
    }
    bestunit = NPY_FR_h;
    goto finish;
  }

  // 如果 *substr 是 ':'，则标记存在时分秒分隔符，进行相应的移动和更新操作
  if (*substr == ':') {
    has_hms_sep = 1;
    ++substr;
    --sublen;
    // 如果 sublen 为 0 或 *substr 不是数字，则跳转到 parse_error 标签处
    if (sublen == 0 || !isdigit(*substr)) {
      goto parse_error;
    }
    // 调用 compare_format 函数比较 format 和 ":" 的格式，根据返回值决定后续动作
    comparison =
        compare_format(&format, &format_len, ":", 1, format_requirement);
    // 如果比较结果为 COMPARISON_ERROR，则跳转到 parse_error 标签处
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      // 如果比较结果为 COMPLETED_PARTIAL_MATCH，则跳转到 finish 标签处
      goto finish;
    }
  } else if (!isdigit(*substr)) {
    // 如果 *substr 不是数字，并且小时值不是两位数，则跳转到 parse_timezone 标签处
    if (!hour_was_2_digits) {
      goto parse_error;
    }
    goto parse_timezone;
  }

  /* PARSE THE MINUTES */
  // 调用 compare_format 函数比较 format 和 "%M" 的格式，根据返回值决定后续动作
  comparison =
      compare_format(&format, &format_len, "%M", 2, format_requirement);
  // 如果比较结果为 COMPARISON_ERROR，则跳转到 parse_error 标签处
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    // 如果比较结果为 COMPLETED_PARTIAL_MATCH，则跳转到 finish 标签处
    goto finish;
  }
  /* First digit required */
  // 将 out->min 设为 *substr 对应的数字
  out->min = (*substr - '0');
  // 将 bestunit 设为 NPY_FR_m
  bestunit = NPY_FR_m;
  // substr 指针向后移动一位，sublen 减一
  ++substr;
  --sublen;
  /* Second digit optional if there was a separator */
  // 如果 *substr 是数字，并且存在时分秒分隔符，则将 out->min 更新为两位数的分钟值，并进行相应的移动和更新操作
  if (isdigit(*substr)) {
    out->min = 10 * out->min + (*substr - '0');
    ++substr;
    --sublen;
    // 如果分钟值超过 59，则根据 want_exc 标志抛出异常或跳转到 error 标签处
    if (out->min >= 60) {
      if (want_exc) {
        PyErr_Format(PyExc_ValueError,
                     "Minutes out of range in datetime string \"%s\"", str);
      }
      goto error;
    }
  } else if (!has_hms_sep) {
    // 如果不存在时分秒分隔符，并且 *substr 不是数字，则跳转到 parse_error 标签处
    goto parse_error;
  }

  // 如果 sublen 为 0，则将 bestunit 设为 NPY_FR_m，并根据 format_len 判断是否跳转到 parse_error 标签处
  if (sublen == 0) {
    bestunit = NPY_FR_m;
    if (format_len) {
      goto parse_error;
    }
    // 正常结束，跳转到 finish 标签处
    goto finish;
  }
    // 跳转到结束标签，表示当前处理结束
    goto finish;
  }

  /* 如果程序执行到这个条件块，那么下一个字符是一个数字。*/
  if (has_hms_sep && *substr == ':') {
    // 比较当前格式与":"的匹配情况
    comparison =
        compare_format(&format, &format_len, ":", 1, format_requirement);
    // 如果比较出错，则跳转到解析错误处理
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      // 如果部分匹配完成，则跳转到结束标签
      goto finish;
    }
    // 移动到下一个字符并调整剩余长度
    ++substr;
    --sublen;
    /* 不能有尾随的':' */
    // 如果剩余长度为0或者下一个字符不是数字，则跳转到解析错误处理
    if (sublen == 0 || !isdigit(*substr)) {
      goto parse_error;
    }
  } else if (!has_hms_sep && isdigit(*substr)) {
  } else {
    // 否则跳转到解析时区处理
    goto parse_timezone;
  }

  /* 解析秒 */
  // 比较当前格式与"%S"的匹配情况
  comparison =
      compare_format(&format, &format_len, "%S", 2, format_requirement);
  // 如果比较出错，则跳转到解析错误处理
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    // 如果部分匹配完成，则跳转到结束标签
    goto finish;
  }
  /* 第一个数字是必须的 */
  // 将第一个秒数字符转换为数字赋给输出结构体的秒字段
  out->sec = (*substr - '0');
  // 移动到下一个字符并调整剩余长度
  ++substr;
  --sublen;
  /* 如果有分隔符，则第二个数字是可选的 */
  // 如果下一个字符是数字，则将其加入到秒字段中
  if (isdigit(*substr)) {
    out->sec = 10 * out->sec + (*substr - '0');
    // 移动到下一个字符并调整剩余长度
    ++substr;
    --sublen;
    // 如果秒数超过了60，则报错并跳转到错误处理
    if (out->sec >= 60) {
      // 如果需要异常处理，则抛出秒数超出范围的错误信息
      if (want_exc) {
        PyErr_Format(PyExc_ValueError,
                     "Seconds out of range in datetime string \"%s\"", str);
      }
      // 跳转到错误处理
      goto error;
    }
  } else if (!has_hms_sep) {
    // 如果没有分隔符但下一个字符不是数字，则跳转到解析错误处理
    goto parse_error;
  }

  /* 下一个字符可能是'.'，表示小数秒部分 */
  // 如果剩余长度大于0且下一个字符是'.'，则移动到小数秒部分处理
  if (sublen > 0 && *substr == '.') {
    ++substr;
    --sublen;
    // 比较当前格式与"."的匹配情况
    comparison =
        compare_format(&format, &format_len, ".", 1, format_requirement);
    // 如果比较出错，则跳转到解析错误处理
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      // 如果部分匹配完成，则跳转到结束标签
      goto finish;
    }
  } else {
    // 否则将时间单位设为秒，并跳转到解析时区处理
    bestunit = NPY_FR_s;
    goto parse_timezone;
  }

  /* 解析微秒（0到6位数字） */
  // 比较当前格式与"%f"的匹配情况
  comparison =
      compare_format(&format, &format_len, "%f", 2, format_requirement);
  // 如果比较出错，则跳转到解析错误处理
  if (comparison == COMPARISON_ERROR) {
    goto parse_error;
  } else if (comparison == COMPLETED_PARTIAL_MATCH) {
    // 如果部分匹配完成，则跳转到结束标签
    goto finish;
  }
  // 初始化数字位数为0
  numdigits = 0;
  // 循环处理微秒部分，最多6位数字
  for (i = 0; i < 6; ++i) {
    // 微秒数乘以10
    out->us *= 10;
    // 如果剩余长度大于0且下一个字符是数字，则加入到微秒字段中
    if (sublen > 0 && isdigit(*substr)) {
      out->us += (*substr - '0');
      // 移动到下一个字符并调整剩余长度
      ++substr;
      --sublen;
      // 增加数字位数计数
      ++numdigits;
    }
  }

  // 如果剩余长度为0或者下一个字符不是数字，则处理最佳时间单位并跳转到解析时区处理
  if (sublen == 0 || !isdigit(*substr)) {
    if (numdigits > 3) {
      bestunit = NPY_FR_us;
    } else {
      bestunit = NPY_FR_ms;
    }
    goto parse_timezone;
  }

  /* 解析皮秒（0到6位数字） */
  // 初始化数字位数为0
  numdigits = 0;
  // 循环处理皮秒部分，最多6位数字
  for (i = 0; i < 6; ++i) {
    // 皮秒数乘以10
    out->ps *= 10;
    // 如果剩余长度大于0且下一个字符是数字，则加入到皮秒字段中
    if (sublen > 0 && isdigit(*substr)) {
      out->ps += (*substr - '0');
      // 移动到下一个字符并调整剩余长度
      ++substr;
      --sublen;
      // 增加数字位数计数
      ++numdigits;
    }
  }

  // 如果剩余长度为0或者下一个字符不是数字，则处理最佳时间单位并跳转到解析时区处理
  if (sublen == 0 || !isdigit(*substr)) {
    if (numdigits > 3) {
      bestunit = NPY_FR_ps;
    } else {
      bestunit = NPY_FR_ns;
    }
    goto parse_timezone;
  }

  /* 解析阿托秒（0到6位数字） */
  // 初始化数字位数为0
  numdigits = 0;
  // 循环处理阿托秒部分，最多6位数字
  for (i = 0; i < 6; ++i) {
    // 阿托秒数乘以10
    out->as *= 10;
    // 检查子字符串长度大于零且第一个字符是数字时执行以下操作
    if (sublen > 0 && isdigit(*substr)) {
      // 将字符转换为数字并加到输出的as字段中
      out->as += (*substr - '0');
      // 移动到下一个字符，同时减少子字符串长度和增加数字字符计数
      ++substr;
      --sublen;
      ++numdigits;
    }
  }

  // 如果数字字符计数超过3个，选择最佳单位为NPY_FR_as，否则选择NPY_FR_fs
  if (numdigits > 3) {
    bestunit = NPY_FR_as;
  } else {
    bestunit = NPY_FR_fs;
  }
parse_timezone:
  /* trim any whitespace between time/timezone */
  // 在时间/时区之间去除任何空白字符
  while (sublen > 0 && isspace(*substr)) {
    // 移动指针以跳过空白字符
    ++substr;
    // 减少子字符串长度，以反映已经移动的字符数量
    --sublen;
    // 比较当前格式的空格符号，以确保匹配
    comparison =
        compare_format(&format, &format_len, " ", 1, format_requirement);
    // 如果比较出错，则跳转到解析错误处理
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      // 如果完成部分匹配，则跳转到完成处理
      goto finish;
    }
  }

  // 如果子字符串长度为零
  if (sublen == 0) {
    // 不同于 NumPy，将无时区视为本地时间
    if (format_len > 0) {
      // 如果格式长度大于零，则跳转到解析错误处理
      goto parse_error;
    }
    // 跳转到完成处理
    goto finish;
  }

  /* UTC specifier */
  // UTC 指定符号
  if (*substr == 'Z') {
    // 比较格式中的 "%z"，以确保匹配
    comparison =
        compare_format(&format, &format_len, "%z", 2, format_requirement);
    // 如果比较出错，则跳转到解析错误处理
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      // 如果完成部分匹配，则跳转到完成处理
      goto finish;
    }
    // "Z" 等同于时区偏移 "+00:00"
    if (out_local != NULL) {
      // 如果输出本地时间不为空，则设置为本地时间
      *out_local = 1;
    }

    if (out_tzoffset != NULL) {
      // 如果输出时区偏移不为空，则设置为 0
      *out_tzoffset = 0;
    }

    // 如果子字符串长度为 1
    if (sublen == 1) {
      // 如果格式长度大于零，则跳转到解析错误处理
      if (format_len > 0) {
        goto parse_error;
      }
      // 跳转到完成处理
      goto finish;
    } else {
      // 移动指针以跳过字符 'Z'
      ++substr;
      // 减少子字符串长度，以反映已经移动的字符数量
      --sublen;
    }
  } else if (*substr == '-' || *substr == '+') {
    // 比较格式中的 "%z"，以确保匹配
    comparison =
        compare_format(&format, &format_len, "%z", 2, format_requirement);
    // 如果比较出错，则跳转到解析错误处理
    if (comparison == COMPARISON_ERROR) {
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      // 如果完成部分匹配，则跳转到完成处理
      goto finish;
    }
    // 时区偏移
    int offset_neg = 0, offset_hour = 0, offset_minute = 0;

    /*
     * Since "local" means local with respect to the current
     * machine, we say this is non-local.
     */
    // 因为 "local" 意味着相对于当前机器的本地时间，我们说这是非本地的。

    // 如果子字符串以 '-' 开头
    if (*substr == '-') {
      // 设置偏移为负数
      offset_neg = 1;
    }
    // 移动指针以跳过偏移符号
    ++substr;
    // 减少子字符串长度，以反映已经移动的字符数量
    --sublen;

    /* The hours offset */
    // 小时偏移
    // 如果子字符串长度大于等于 2，且第一个字符是数字且第二个字符是数字
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
      // 计算小时偏移
      offset_hour = 10 * (substr[0] - '0') + (substr[1] - '0');
      // 移动指针以跳过小时部分
      substr += 2;
      // 减少子字符串长度，以反映已经移动的字符数量
      sublen -= 2;
      // 如果小时偏移大于等于 24
      if (offset_hour >= 24) {
        // 如果需要异常处理，则引发值错误异常
        if (want_exc) {
          PyErr_Format(PyExc_ValueError,
                       "Timezone hours offset out of range "
                       "in datetime string \"%s\"",
                       str);
        }
        // 跳转到错误处理
        goto error;
      }
    } else if (sublen >= 1 && isdigit(substr[0])) {
      // 否则，如果子字符串长度大于等于 1，且第一个字符是数字
      // 设置小时偏移
      offset_hour = substr[0] - '0';
      // 移动指针以跳过小时部分
      ++substr;
      // 减少子字符串长度，以反映已经移动的字符数量
      --sublen;
    } else {
      // 其它情况，跳转到解析错误处理
      goto parse_error;
    }

    /* The minutes offset is optional */
    // 分钟偏移是可选的
    # 如果剩余子串长度大于0，则执行以下逻辑
    if (sublen > 0) {
      /* Optional ':' */
      # 如果子串以':'开头，移除该字符，更新子串长度
      if (*substr == ':') {
        ++substr;
        --sublen;
      }

      /* The minutes offset (at the end of the string) */
      # 如果剩余子串长度至少为2且前两个字符是数字，则解析分钟偏移量
      if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        # 计算分钟偏移量
        offset_minute = 10 * (substr[0] - '0') + (substr[1] - '0');
        substr += 2;
        sublen -= 2;
        # 检查分钟偏移量是否超出范围，超出则根据需求抛出异常或跳转至错误处理
        if (offset_minute >= 60) {
          if (want_exc) {
            PyErr_Format(PyExc_ValueError,
                         "Timezone minutes offset out of range "
                         "in datetime string \"%s\"",
                         str);
          }
          goto error;
        }
      } else if (sublen >= 1 && isdigit(substr[0])) {
        # 如果剩余子串长度至少为1且第一个字符是数字，则解析分钟偏移量
        offset_minute = substr[0] - '0';
        ++substr;
        --sublen;
      } else {
        # 若无法解析分钟偏移量，则跳转至解析错误处理
        goto parse_error;
      }
    }

    /* Apply the time zone offset */
    # 如果偏移标志为负，则将小时和分钟偏移量取负值
    if (offset_neg) {
      offset_hour = -offset_hour;
      offset_minute = -offset_minute;
    }
    # 如果输出本地时间的指针不为空，则设置输出本地时间标志，并计算总分钟偏移量
    if (out_local != NULL) {
      *out_local = 1;
      // 与NumPy不同，不改变内部值为本地时间
      *out_tzoffset = 60 * offset_hour + offset_minute;
    }
  }

  /* Skip trailing whitespace */
  # 跳过末尾的空白字符
  while (sublen > 0 && isspace(*substr)) {
    ++substr;
    --sublen;
    # 比较格式，并根据比较结果执行相应操作
    comparison =
        compare_format(&format, &format_len, " ", 1, format_requirement);
    if (comparison == COMPARISON_ERROR) {
      # 若比较出错，则跳转至解析错误处理
      goto parse_error;
    } else if (comparison == COMPLETED_PARTIAL_MATCH) {
      # 若比较完成部分匹配，则跳转至结束处理
      goto finish;
    }
  }

  # 如果剩余子串长度不为0，或格式长度不为0，则跳转至解析错误处理
  if ((sublen != 0) || (format_len != 0)) {
    goto parse_error;
  }
finish:
  // 如果 out_bestunit 不是 NULL，则将 bestunit 的值赋给 *out_bestunit
  if (out_bestunit != NULL) {
    *out_bestunit = bestunit;
  }
  // 返回 0 表示成功完成
  return 0;

parse_error:
  // 如果 want_exc 为真，则抛出 ValueError 异常，格式化错误信息
  if (want_exc) {
    PyErr_Format(PyExc_ValueError,
                 "Error parsing datetime string \"%s\" at position %d", str,
                 (int)(substr - str));
  }
  // 返回 -1 表示解析出错
  return -1;

error:
  // 返回 -1 表示发生错误
  return -1;
}

/*
 * 提供一个字符串长度，用于转换具有给定本地和单位设置的日期时间对象。
 */
int get_datetime_iso_8601_strlen(int local, NPY_DATETIMEUNIT base) {
  int len = 0;

  switch (base) {
  /* 通用单位只能用于表示 NaT */
  /*    return 4;*/
  case NPY_FR_as:
    len += 3; /* "###" */
    PD_FALLTHROUGH;
  case NPY_FR_fs:
    len += 3; /* "###" */
    PD_FALLTHROUGH;
  case NPY_FR_ps:
    len += 3; /* "###" */
    PD_FALLTHROUGH;
  case NPY_FR_ns:
    len += 3; /* "###" */
    PD_FALLTHROUGH;
  case NPY_FR_us:
    len += 3; /* "###" */
    PD_FALLTHROUGH;
  case NPY_FR_ms:
    len += 4; /* ".###" */
    PD_FALLTHROUGH;
  case NPY_FR_s:
    len += 3; /* ":##" */
    PD_FALLTHROUGH;
  case NPY_FR_m:
    len += 3; /* ":##" */
    PD_FALLTHROUGH;
  case NPY_FR_h:
    len += 3; /* "T##" */
    PD_FALLTHROUGH;
  case NPY_FR_D:
  case NPY_FR_W:
    len += 3; /* "-##" */
    PD_FALLTHROUGH;
  case NPY_FR_M:
    len += 3; /* "-##" */
    PD_FALLTHROUGH;
  case NPY_FR_Y:
    len += 21; /* 64-bit year */
    break;
  default:
    len += 3; /* 处理已废弃的 NPY_FR_B */
    break;
  }

  if (base >= NPY_FR_h) {
    if (local) {
      len += 5; /* "+####" or "-####" */
    } else {
      len += 1; /* "Z" */
    }
  }

  len += 1; /* NULL 结尾符 */

  // 返回计算出的字符串长度
  return len;
}

/*
 * 将 npy_datetimestruct 转换为 (几乎) ISO 8601 格式的 NULL 结尾字符串，
 * 使用时区 Z (UTC)。如果字符串恰好适合空间，则省略 NULL 结尾符并返回成功。
 *
 * 与 ISO 8601 的不同之处在于 'NaT' 字符串，以及年份数字 >= 4 而不是严格为 4。
 *
 * 'base' 限制输出到该单位。将 'base' 设置为 -1 自动检测一个基准，
 * 其后所有值为零。
 *
 * 返回 0 表示成功，返回 -1 表示失败（例如如果输出字符串太短）。
 */
int make_iso_8601_datetime(npy_datetimestruct *dts, char *outstr, size_t outlen,
                           int utc, NPY_DATETIMEUNIT base) {
  char *substr = outstr;
  size_t sublen = outlen;
  int tmplen;

  /*
   * 以与天相同精度打印周。
   *
   * TODO: 如果周的起始日是星期一，则可以使用 YYYY-Www 格式打印周。
   */
  if (base == NPY_FR_W) {
    base = NPY_FR_D;
  }

/* YEAR */
/*
 * 不能使用 PyOS_snprintf，因为它总是在末尾产生一个 '\0' 字符，
 * 而 NumPy 字符串类型允许数据一直到缓冲区的末尾。
 */
#ifdef _WIN32
  tmplen = _snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#else
  tmplen = snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#endif
#endif // _WIN32
  /* 如果剩余空间不足或者没有空间存放NULL终止符 */
  if (tmplen < 0 || (size_t)tmplen > sublen) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 更新子字符串指针和剩余长度
  substr += tmplen;
  sublen -= tmplen;

  /* 如果单位是年，直接返回 */
  if (base == NPY_FR_Y) {
    // 如果剩余空间大于0，则在当前位置添加NULL终止符
    if (sublen > 0) {
      *substr = '\0';
    }
    return 0;
  }

  /* 月份 */
  if (sublen < 1) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加'-'字符
  substr[0] = '-';
  if (sublen < 2) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加十位上的月份数字字符
  substr[1] = (char)((dts->month / 10) + '0');
  if (sublen < 3) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加个位上的月份数字字符
  substr[2] = (char)((dts->month % 10) + '0');
  // 更新子字符串指针和剩余长度
  substr += 3;
  sublen -= 3;

  /* 如果单位是月，直接返回 */
  if (base == NPY_FR_M) {
    // 如果剩余空间大于0，则在当前位置添加NULL终止符
    if (sublen > 0) {
      *substr = '\0';
    }
    return 0;
  }

  /* 天数 */
  if (sublen < 1) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加'-'字符
  substr[0] = '-';
  if (sublen < 2) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加十位上的天数数字字符
  substr[1] = (char)((dts->day / 10) + '0');
  if (sublen < 3) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加个位上的天数数字字符
  substr[2] = (char)((dts->day % 10) + '0');
  // 更新子字符串指针和剩余长度
  substr += 3;
  sublen -= 3;

  /* 如果单位是天，直接返回 */
  if (base == NPY_FR_D) {
    // 如果剩余空间大于0，则在当前位置添加NULL终止符
    if (sublen > 0) {
      *substr = '\0';
    }
    return 0;
  }

  /* 小时 */
  if (sublen < 1) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加'T'字符
  substr[0] = 'T';
  if (sublen < 2) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加十位上的小时数字字符
  substr[1] = (char)((dts->hour / 10) + '0');
  if (sublen < 3) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加个位上的小时数字字符
  substr[2] = (char)((dts->hour % 10) + '0');
  // 更新子字符串指针和剩余长度
  substr += 3;
  sublen -= 3;

  /* 如果单位是小时，跳转到添加时区的处理 */
  if (base == NPY_FR_h) {
    goto add_time_zone;
  }

  /* 分钟 */
  if (sublen < 1) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加':'字符
  substr[0] = ':';
  if (sublen < 2) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加十位上的分钟数字字符
  substr[1] = (char)((dts->min / 10) + '0');
  if (sublen < 3) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加个位上的分钟数字字符
  substr[2] = (char)((dts->min % 10) + '0');
  // 更新子字符串指针和剩余长度
  substr += 3;
  sublen -= 3;

  /* 如果单位是分钟，跳转到添加时区的处理 */
  if (base == NPY_FR_m) {
    goto add_time_zone;
  }

  /* 秒数 */
  if (sublen < 1) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加':'字符
  substr[0] = ':';
  if (sublen < 2) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加十位上的秒数数字字符
  substr[1] = (char)((dts->sec / 10) + '0');
  if (sublen < 3) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加个位上的秒数数字字符
  substr[2] = (char)((dts->sec % 10) + '0');
  // 更新子字符串指针和剩余长度
  substr += 3;
  sublen -= 3;

  /* 如果单位是秒，跳转到添加时区的处理 */
  if (base == NPY_FR_s) {
    goto add_time_zone;
  }

  /* 毫秒 */
  if (sublen < 1) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加'.'字符
  substr[0] = '.';
  if (sublen < 2) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加百位上的毫秒数数字字符
  substr[1] = (char)((dts->us / 100000) % 10 + '0');
  if (sublen < 3) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加十位上的毫秒数数字字符
  substr[2] = (char)((dts->us / 10000) % 10 + '0');
  if (sublen < 4) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
  // 在当前位置添加个位上的毫秒数数字字符
  substr[3] = (char)((dts->us / 1000) % 10 + '0');
  // 更新子字符串指针和剩余长度
  substr += 4;
  sublen -= 4;

  /* 如果单位是毫秒，跳转到添加时区的处理 */
  if (base == NPY_FR_ms) {
    goto add_time_zone;
  }

  /* 微秒 */
  if (sublen < 1) {
    // 转到字符串过短的错误处理标签
    goto string_too_short;
  }
    // 如果字符串长度不足，则跳转到标签 string_too_short 处
    goto string_too_short;
  }
  // 将微秒部分的百位数转换为字符并存储到 substr 数组中
  substr[0] = (char)((dts->us / 100) % 10 + '0');
  // 如果剩余的长度小于 2，则跳转到标签 string_too_short 处
  if (sublen < 2) {
    goto string_too_short;
  }
  // 将微秒部分的十位数转换为字符并存储到 substr 数组中
  substr[1] = (char)((dts->us / 10) % 10 + '0');
  // 如果剩余的长度小于 3，则跳转到标签 string_too_short 处
  if (sublen < 3) {
    goto string_too_short;
  }
  // 将微秒部分的个位数转换为字符并存储到 substr 数组中
  substr[2] = (char)(dts->us % 10 + '0');
  // substr 指针向后移动 3 个位置
  substr += 3;
  // sublen 减去 3
  sublen -= 3;

  /* 如果单位是微秒，则跳转到标签 add_time_zone 处 */
  if (base == NPY_FR_us) {
    goto add_time_zone;
  }

  /* 纳秒部分 */
  // 如果剩余的长度小于 1，则跳转到标签 string_too_short 处
  if (sublen < 1) {
    goto string_too_short;
  }
  // 将纳秒部分的十万位数转换为字符并存储到 substr 数组中
  substr[0] = (char)((dts->ps / 100000) % 10 + '0');
  // 如果剩余的长度小于 2，则跳转到标签 string_too_short 处
  if (sublen < 2) {
    goto string_too_short;
  }
  // 将纳秒部分的万位数转换为字符并存储到 substr 数组中
  substr[1] = (char)((dts->ps / 10000) % 10 + '0');
  // 如果剩余的长度小于 3，则跳转到标签 string_too_short 处
  if (sublen < 3) {
    goto string_too_short;
  }
  // 将纳秒部分的千位数转换为字符并存储到 substr 数组中
  substr[2] = (char)((dts->ps / 1000) % 10 + '0');
  // substr 指针向后移动 3 个位置
  substr += 3;
  // sublen 减去 3
  sublen -= 3;

  /* 如果单位是纳秒，则跳转到标签 add_time_zone 处 */
  if (base == NPY_FR_ns) {
    goto add_time_zone;
  }

  /* 皮秒部分 */
  // 如果剩余的长度小于 1，则跳转到标签 string_too_short 处
  if (sublen < 1) {
    goto string_too_short;
  }
  // 将皮秒部分的百位数转换为字符并存储到 substr 数组中
  substr[0] = (char)((dts->ps / 100) % 10 + '0');
  // 如果剩余的长度小于 2，则跳转到标签 string_too_short 处
  if (sublen < 2) {
    goto string_too_short;
  }
  // 将皮秒部分的十位数转换为字符并存储到 substr 数组中
  substr[1] = (char)((dts->ps / 10) % 10 + '0');
  // 如果剩余的长度小于 3，则跳转到标签 string_too_short 处
  if (sublen < 3) {
    goto string_too_short;
  }
  // 将皮秒部分的个位数转换为字符并存储到 substr 数组中
  substr[2] = (char)(dts->ps % 10 + '0');
  // substr 指针向后移动 3 个位置
  substr += 3;
  // sublen 减去 3
  sublen -= 3;

  /* 如果单位是皮秒，则跳转到标签 add_time_zone 处 */
  if (base == NPY_FR_ps) {
    goto add_time_zone;
  }

  /* 飞秒部分 */
  // 如果剩余的长度小于 1，则跳转到标签 string_too_short 处
  if (sublen < 1) {
    goto string_too_short;
  }
  // 将飞秒部分的十万位数转换为字符并存储到 substr 数组中
  substr[0] = (char)((dts->as / 100000) % 10 + '0');
  // 如果剩余的长度小于 2，则跳转到标签 string_too_short 处
  if (sublen < 2) {
    goto string_too_short;
  }
  // 将飞秒部分的万位数转换为字符并存储到 substr 数组中
  substr[1] = (char)((dts->as / 10000) % 10 + '0');
  // 如果剩余的长度小于 3，则跳转到标签 string_too_short 处
  if (sublen < 3) {
    goto string_too_short;
  }
  // 将飞秒部分的千位数转换为字符并存储到 substr 数组中
  substr[2] = (char)((dts->as / 1000) % 10 + '0');
  // substr 指针向后移动 3 个位置
  substr += 3;
  // sublen 减去 3
  sublen -= 3;

  /* 如果单位是飞秒，则跳转到标签 add_time_zone 处 */
  if (base == NPY_FR_fs) {
    goto add_time_zone;
  }

  /* 阿托秒部分 */
  // 如果剩余的长度小于 1，则跳转到标签 string_too_short 处
  if (sublen < 1) {
    goto string_too_short;
  }
  // 将阿托秒部分的百位数转换为字符并存储到 substr 数组中
  substr[0] = (char)((dts->as / 100) % 10 + '0');
  // 如果剩余的长度小于 2，则跳转到标签 string_too_short 处
  if (sublen < 2) {
    goto string_too_short;
  }
  // 将阿托秒部分的十位数转换为字符并存储到 substr 数组中
  substr[1] = (char)((dts->as / 10) % 10 + '0');
  // 如果剩余的长度小于 3，则跳转到标签 string_too_short 处
  if (sublen < 3) {
    goto string_too_short;
  }
  // 将阿托秒部分的个位数转换为字符并存储到 substr 数组中
  substr[2] = (char)(dts->as % 10 + '0');
  // substr 指针向后移动 3 个位置
  substr += 3;
  // sublen 减去 3
  sublen -= 3;
/* 添加时区信息 */
add_time_zone:
  /* UTC "Zulu" time */
  if (utc) {
    /* 如果是 UTC 时间，检查字符串长度是否小于1 */
    if (sublen < 1) {
      /* 如果字符串太短，跳转到错误处理标签 */
      goto string_too_short;
    }
    /* 将 'Z' 字符添加到子字符串开头 */
    substr[0] = 'Z';
    substr += 1;
    sublen -= 1;
  }
  /* 添加空字符终止符，并返回 */
  if (sublen > 0) {
    /* 如果子字符串还有剩余长度，添加空字符终止符 */
    substr[0] = '\0';
  }

  /* 返回成功状态 */
  return 0;

/* 处理字符串过短的错误情况 */
string_too_short:
  /* 抛出运行时错误，指明字符串的长度 */
  PyErr_Format(PyExc_RuntimeError,
               "The string provided for NumPy ISO datetime formatting "
               "was too short, with length %d",
               outlen);
  /* 返回错误状态 */
  return -1;
}

/* 创建 ISO 8601 格式的时间间隔字符串 */
int make_iso_8601_timedelta(pandas_timedeltastruct *tds, char *outstr,
                            size_t *outlen) {
  /* 初始化输出长度 */
  *outlen = 0;
  /* 格式化输出 ISO 8601 时间间隔的字符串 */
  *outlen += snprintf(outstr, 60, // NOLINT
                      "P%" NPY_INT64_FMT "DT%" NPY_INT32_FMT "H%" NPY_INT32_FMT
                      "M%" NPY_INT32_FMT,
                      tds->days, tds->hrs, tds->min, tds->sec);
  /* 移动输出字符串指针到已写入部分的结尾 */
  outstr += *outlen;

  /* 如果有纳秒部分 */
  if (tds->ns != 0) {
    /* 添加毫秒、微秒和纳秒部分 */
    *outlen += snprintf(outstr, 12, // NOLINT
                        ".%03" NPY_INT32_FMT "%03" NPY_INT32_FMT
                        "%03" NPY_INT32_FMT "S",
                        tds->ms, tds->us, tds->ns);
  } else if (tds->us != 0) {
    /* 如果没有纳秒但有微秒，添加毫秒和微秒部分 */
    *outlen += snprintf(outstr, 9, // NOLINT
                        ".%03" NPY_INT32_FMT "%03" NPY_INT32_FMT "S", tds->ms,
                        tds->us);
  } else if (tds->ms != 0) {
    /* 如果没有微秒但有毫秒，添加毫秒部分 */
    *outlen += snprintf(outstr, 6, // NOLINT
                        ".%03" NPY_INT32_FMT "S", tds->ms);
  } else {
    /* 如果仅有秒数，添加秒数部分 */
    *outlen += snprintf(outstr, 2, // NOLINT
                        "%s", "S");
  }

  /* 返回成功状态 */
  return 0;
}
```