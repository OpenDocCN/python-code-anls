# `D:\src\scipysrc\pandas\pandas\_libs\src\vendored\numpy\datetime\np_datetime.c`

```
/*
 * 版权所有（c）2016年，PyData开发团队
 * 保留所有权利。
 * 
 * 根据BSD简化许可证分发。
 * 
 * 完整许可证位于随附此软件的LICENSE文件中。
 * 
 * 版权所有（c）2005-2011年，NumPy开发者
 * 保留所有权利。
 * 
 * 此文件源自NumPy 1.7。请参阅NUMPY_LICENSE.txt
 */

// LICENSES/NUMPY_LICENSE中的许可证

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif // NPY_NO_DEPRECATED_API

#include <Python.h>

#include "pandas/vendored/numpy/datetime/np_datetime.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PANDAS_DATETIME_NUMPY
#include <numpy/ndarrayobject.h>
#include <numpy/npy_common.h>

#if defined(_WIN32)
#ifndef ENABLE_INTSAFE_SIGNED_FUNCTIONS
#define ENABLE_INTSAFE_SIGNED_FUNCTIONS
#endif
#include <intsafe.h>
#define checked_int64_add(a, b, res) LongLongAdd(a, b, res)
#define checked_int64_sub(a, b, res) LongLongSub(a, b, res)
#define checked_int64_mul(a, b, res) LongLongMult(a, b, res)
#else
#if defined __has_builtin
#if __has_builtin(__builtin_add_overflow)
#define checked_int64_add(a, b, res) __builtin_add_overflow(a, b, res)
#define checked_int64_sub(a, b, res) __builtin_sub_overflow(a, b, res)
#define checked_int64_mul(a, b, res) __builtin_mul_overflow(a, b, res)
#else
_Static_assert(0,
               "Overflow checking not detected; please try a newer compiler");
#endif
// __has_builtin was added in gcc 10, but our muslinux_1_1 build environment
// only has gcc-9.3, so fall back to __GNUC__ macro as long as we have that
#elif __GNUC__ > 7
#define checked_int64_add(a, b, res) __builtin_add_overflow(a, b, res)
#define checked_int64_sub(a, b, res) __builtin_sub_overflow(a, b, res)
#define checked_int64_mul(a, b, res) __builtin_mul_overflow(a, b, res)
#else
_Static_assert(0, "__has_builtin not detected; please try a newer compiler");
#endif
#endif

#define PD_CHECK_OVERFLOW(FUNC)                                                \
  do {                                                                         \
    if ((FUNC) != 0) {                                                         \
      PyGILState_STATE gstate = PyGILState_Ensure();                           \
      PyErr_SetString(PyExc_OverflowError,                                     \
                      "Overflow occurred in npy_datetimestruct_to_datetime");  \
      PyGILState_Release(gstate);                                              \
      return -1;                                                               \
    }                                                                          \
  } while (0)

const int days_per_month_table[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},  // 平年每月天数
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}   // 闰年每月天数
};

/*
 * 如果给定年份是闰年则返回1，否则返回0。
 */
int is_leapyear(npy_int64 year) {
  return (year & 0x3) == 0 && /* year % 4 == 0 */
         ((year % 100) != 0 || (year % 400) == 0);
}
/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.
 */
void add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes) {
  int isleap;

  /* MINUTES */
  // 增加分钟数到日期时间结构体中
  dts->min += minutes;
  // 处理分钟数小于零的情况，向前借位小时，直到分钟数非负
  while (dts->min < 0) {
    dts->min += 60;
    dts->hour--;
  }
  // 处理分钟数超过59的情况，进位到小时
  while (dts->min >= 60) {
    dts->min -= 60;
    dts->hour++;
  }

  /* HOURS */
  // 处理小时数小于零的情况，向前借位天数，直到小时数非负
  while (dts->hour < 0) {
    dts->hour += 24;
    dts->day--;
  }
  // 处理小时数超过23的情况，进位到天数
  while (dts->hour >= 24) {
    dts->hour -= 24;
    dts->day++;
  }

  /* DAYS */
  // 处理日期数小于1的情况，向前借位月份和年份
  if (dts->day < 1) {
    dts->month--;
    if (dts->month < 1) {
      dts->year--;
      dts->month = 12;
    }
    // 判断是否是闰年，根据不同的月份修正日期
    isleap = is_leapyear(dts->year);
    dts->day += days_per_month_table[isleap][dts->month - 1];
  } else if (dts->day > 28) {
    // 判断是否是闰年，根据不同的月份修正日期
    isleap = is_leapyear(dts->year);
    if (dts->day > days_per_month_table[isleap][dts->month - 1]) {
      dts->day -= days_per_month_table[isleap][dts->month - 1];
      dts->month++;
      if (dts->month > 12) {
        dts->year++;
        dts->month = 1;
      }
    }
  }
}

/*
 * Calculates the days offset from the 1970 epoch.
 */
npy_int64 get_datetimestruct_days(const npy_datetimestruct *dts) {
  int i, month;
  npy_int64 year, days = 0;
  const int *month_lengths;

  year = dts->year - 1970;
  days = year * 365;

  /* Adjust for leap years */
  // 调整天数以考虑闰年
  if (days >= 0) {
    /*
     * 1968 is the closest leap year before 1970.
     * Exclude the current year, so add 1.
     */
    year += 1;
    /* Add one day for each 4 years */
    days += year / 4;
    /* 1900 is the closest previous year divisible by 100 */
    year += 68;
    /* Subtract one day for each 100 years */
    days -= year / 100;
    /* 1600 is the closest previous year divisible by 400 */
    year += 300;
    /* Add one day for each 400 years */
    days += year / 400;
  } else {
    /*
     * 1972 is the closest later year after 1970.
     * Include the current year, so subtract 2.
     */
    year -= 2;
    /* Subtract one day for each 4 years */
    days += year / 4;
    /* 2000 is the closest later year divisible by 100 */
    year -= 28;
    /* Add one day for each 100 years */
    days -= year / 100;
    /* 2000 is also the closest later year divisible by 400 */
    /* Subtract one day for each 400 years */
    days += year / 400;
  }

  month_lengths = days_per_month_table[is_leapyear(dts->year)];
  month = dts->month - 1;

  /* Add the months */
  // 添加月份对应的天数
  for (i = 0; i < month; ++i) {
    days += month_lengths[i];
  }

  /* Add the days */
  // 添加当月的天数
  days += dts->day - 1;

  return days;
}

/*
 * Modifies '*days_' to be the day offset within the year,
 * and returns the year.
 */
static npy_int64 days_to_yearsdays(npy_int64 *days_) {
  const npy_int64 days_per_400years = (400 * 365 + 100 - 4 + 1);
  /* Adjust so it's relative to the year 2000 (divisible by 400) */
  npy_int64 days = (*days_) - (365 * 30 + 7);
  npy_int64 year;

  /* Break down the 400 year cycle to get the year and day within the year */
  // 根据400年的周期计算年份和在年内的天数偏移
  if (days >= 0) {
    # 如果天数大于等于每400年的天数
    year = 400 * (days / days_per_400years);
    # 取天数除以每400年的天数的余数
    days = days % days_per_400years;
  } else {
    # 计算年份（考虑400年周期内的情况）
    year = 400 * ((days - (days_per_400years - 1)) / days_per_400years);
    # 取天数除以每400年的天数的余数
    days = days % days_per_400years;
    # 如果天数为负数，调整为正数
    if (days < 0) {
      days += days_per_400years;
    }
  }

  /* 计算在400年周期内的年份和天数 */
  # 如果天数大于等于366，计算年份和余下天数
  if (days >= 366) {
    year += 100 * ((days - 1) / (100 * 365 + 25 - 1));
    # 取余下天数除以（100*365 + 25 - 1）的余数
    days = (days - 1) % (100 * 365 + 25 - 1);
    # 如果余下天数大于等于365，计算年份和余下天数
    if (days >= 365) {
      year += 4 * ((days + 1) / (4 * 365 + 1));
      # 取余下天数除以（4*365 + 1）的余数
      days = (days + 1) % (4 * 365 + 1);
      # 如果余下天数大于等于366，计算年份和余下天数
      if (days >= 366) {
        year += (days - 1) / 365;
        # 取余下天数除以365的余数
        days = (days - 1) % 365;
      }
    }
  }

  # 将计算得到的天数赋值给days_指向的变量
  *days_ = days;
  # 返回计算得到的年份加上2000
  return year + 2000;
/*
 * Fills in the year, month, day in 'dts' based on the days
 * offset from 1970.
 */
static void set_datetimestruct_days(npy_int64 days, npy_datetimestruct *dts) {
  const int *month_lengths;
  int i;

  // 计算年数并将剩余的天数赋给 days
  dts->year = days_to_yearsdays(&days);
  // 根据年份是否是闰年确定每个月的天数表
  month_lengths = days_per_month_table[is_leapyear(dts->year)];

  for (i = 0; i < 12; ++i) {
    // 找到当前月份并设置 dts 结构体中的月份和天数
    if (days < month_lengths[i]) {
      dts->month = i + 1;
      dts->day = (npy_int32)days + 1;
      return;
    } else {
      days -= month_lengths[i];
    }
  }
}

/*
 * Compares two npy_datetimestruct objects chronologically
 */
int cmp_npy_datetimestruct(const npy_datetimestruct *a,
                           const npy_datetimestruct *b) {
  // 逐字段比较 npy_datetimestruct 结构体，返回比较结果
  if (a->year > b->year) {
    return 1;
  } else if (a->year < b->year) {
    return -1;
  }

  if (a->month > b->month) {
    return 1;
  } else if (a->month < b->month) {
    return -1;
  }

  if (a->day > b->day) {
    return 1;
  } else if (a->day < b->day) {
    return -1;
  }

  if (a->hour > b->hour) {
    return 1;
  } else if (a->hour < b->hour) {
    return -1;
  }

  if (a->min > b->min) {
    return 1;
  } else if (a->min < b->min) {
    return -1;
  }

  if (a->sec > b->sec) {
    return 1;
  } else if (a->sec < b->sec) {
    return -1;
  }

  if (a->us > b->us) {
    return 1;
  } else if (a->us < b->us) {
    return -1;
  }

  if (a->ps > b->ps) {
    return 1;
  } else if (a->ps < b->ps) {
    return -1;
  }

  if (a->as > b->as) {
    return 1;
  } else if (a->as < b->as) {
    return -1;
  }

  // 如果所有字段都相等，返回 0 表示相等
  return 0;
}

/*
 * Returns the offset from utc of the timezone as a timedelta.
 * The caller is responsible for ensuring that the tzinfo
 * attribute exists on the datetime object.
 *
 * If the passed object is timezone naive, Py_None is returned.
 * If extraction of the offset fails, NULL is returned.
 *
 * NOTE: This function is not vendored from numpy.
 */
PyObject *extract_utc_offset(PyObject *obj) {
  // 获取对象的 tzinfo 属性
  PyObject *tmp = PyObject_GetAttrString(obj, "tzinfo");
  if (tmp == NULL) {
    return NULL;
  }
  // 如果 tzinfo 不是 Py_None，则调用其 utcoffset 方法获取 UTC 偏移量
  if (tmp != Py_None) {
    PyObject *offset = PyObject_CallMethod(tmp, "utcoffset", "O", obj);
    if (offset == NULL) {
      Py_DECREF(tmp);
      return NULL;
    }
    return offset;
  }
  // 如果 tzinfo 是 Py_None，返回 Py_None
  return tmp;
}

static inline int scaleYearToEpoch(int64_t year, int64_t *result) {
  // 将年份转换为从 1970 年到该年的秒数
  return checked_int64_sub(year, 1970, result);
}

static inline int scaleYearsToMonths(int64_t years, int64_t *result) {
  // 将年数转换为月数
  return checked_int64_mul(years, 12, result);
}

static inline int scaleDaysToWeeks(int64_t days, int64_t *result) {
  if (days >= 0) {
    // 将天数转换为周数
    *result = days / 7;
    return 0;
  } else {
    int res;
    int64_t checked_days;
    // 处理负数天数的情况，确保结果为整数周数
    if ((res = checked_int64_sub(days, 6, &checked_days))) {
      return res;
    }

    *result = checked_days / 7;
    return 0;
  }
}

static inline int scaleDaysToHours(int64_t days, int64_t *result) {
  // 将天数转换为小时数
  return checked_int64_mul(days, 24, result);
}
/*
 * 将小时转换为分钟，并通过 result 返回结果。
 * 返回值：操作成功返回 0，否则返回错误代码。
 */
static inline int scaleHoursToMinutes(int64_t hours, int64_t *result) {
    return checked_int64_mul(hours, 60, result);
}

/*
 * 将分钟转换为秒，并通过 result 返回结果。
 * 返回值：操作成功返回 0，否则返回错误代码。
 */
static inline int scaleMinutesToSeconds(int64_t minutes, int64_t *result) {
    return checked_int64_mul(minutes, 60, result);
}

/*
 * 将秒转换为毫秒，并通过 result 返回结果。
 * 返回值：操作成功返回 0，否则返回错误代码。
 */
static inline int scaleSecondsToMilliseconds(int64_t seconds, int64_t *result) {
    return checked_int64_mul(seconds, 1000, result);
}

/*
 * 将秒转换为微秒，并通过 result 返回结果。
 * 返回值：操作成功返回 0，否则返回错误代码。
 */
static inline int scaleSecondsToMicroseconds(int64_t seconds, int64_t *result) {
    return checked_int64_mul(seconds, 1000000, result);
}

/*
 * 将微秒转换为纳秒，并通过 result 返回结果。
 * 返回值：操作成功返回 0，否则返回错误代码。
 */
static inline int scaleMicrosecondsToNanoseconds(int64_t microseconds,
                                                 int64_t *result) {
    return checked_int64_mul(microseconds, 1000, result);
}

/*
 * 将微秒转换为皮秒，并通过 result 返回结果。
 * 返回值：操作成功返回 0，否则返回错误代码。
 */
static inline int scaleMicrosecondsToPicoseconds(int64_t microseconds,
                                                 int64_t *result) {
    return checked_int64_mul(microseconds, 1000000, result);
}

/*
 * 将皮秒转换为飞秒，并通过 result 返回结果。
 * 返回值：操作成功返回结果值，否则返回错误代码。
 */
static inline int64_t scalePicosecondsToFemtoseconds(int64_t picoseconds,
                                                     int64_t *result) {
    return checked_int64_mul(picoseconds, 1000, result);
}

/*
 * 将皮秒转换为阿秒，并通过 result 返回结果。
 * 返回值：操作成功返回结果值，否则返回错误代码。
 */
static inline int64_t scalePicosecondsToAttoseconds(int64_t picoseconds,
                                                    int64_t *result) {
    return checked_int64_mul(picoseconds, 1000000, result);
}

/*
 * 将 npy_datetimestruct 结构的日期时间转换为基于给定单位 base 的 datetime 值。
 * 返回值：转换成功返回相应的日期时间值，否则返回 -1 并设置异常。
 */
npy_datetime npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT base,
                                            const npy_datetimestruct *dts) {
    if ((base == NPY_FR_Y) || (base == NPY_FR_M)) {
        int64_t years;
        PD_CHECK_OVERFLOW(scaleYearToEpoch(dts->year, &years)); // 将年份转换为纪元时间

        if (base == NPY_FR_Y) {
            return years; // 返回年份
        }

        int64_t months;
        PD_CHECK_OVERFLOW(scaleYearsToMonths(years, &months)); // 将年份转换为月数

        int64_t months_adder;
        PD_CHECK_OVERFLOW(checked_int64_sub(dts->month, 1, &months_adder)); // 计算月份偏移
        PD_CHECK_OVERFLOW(checked_int64_add(months, months_adder, &months)); // 添加月份偏移

        if (base == NPY_FR_M) {
            return months; // 返回月份
        }
    }

    const int64_t days = get_datetimestruct_days(dts); // 获取日期时间结构的天数
    if (base == NPY_FR_D) {
        return days; // 返回天数
    }

    if (base == NPY_FR_W) {
        int64_t weeks;
        PD_CHECK_OVERFLOW(scaleDaysToWeeks(days, &weeks)); // 将天数转换为周数
        return weeks; // 返回周数
    }

    int64_t hours;
    PD_CHECK_OVERFLOW(scaleDaysToHours(days, &hours)); // 将天数转换为小时数
    PD_CHECK_OVERFLOW(checked_int64_add(hours, dts->hour, &hours)); // 添加小时偏移

    if (base == NPY_FR_h) {
        return hours; // 返回小时数
    }

    int64_t minutes;
    PD_CHECK_OVERFLOW(scaleHoursToMinutes(hours, &minutes)); // 将小时数转换为分钟数
    PD_CHECK_OVERFLOW(checked_int64_add(minutes, dts->min, &minutes)); // 添加分钟偏移

    if (base == NPY_FR_m) {
        return minutes; // 返回分钟数
    }

    int64_t seconds;
    PD_CHECK_OVERFLOW(scaleMinutesToSeconds(minutes, &seconds)); // 将分钟数转换为秒数
    PD_CHECK_OVERFLOW(checked_int64_add(seconds, dts->sec, &seconds)); // 添加秒偏移

    if (base == NPY_FR_s) {
        return seconds; // 返回秒数
    }

    if (base == NPY_FR_ms) {
        int64_t milliseconds;
        PD_CHECK_OVERFLOW(scaleSecondsToMilliseconds(seconds, &milliseconds)); // 将秒数转换为毫秒数
        PD_CHECK_OVERFLOW(checked_int64_add(milliseconds, dts->us / 1000, &milliseconds)); // 添加微秒转换的毫秒偏移
        return milliseconds; // 返回毫秒数
    }
}


这些注释为给定的 C 语言代码添加了详细的解释，每个函数和代码段的目的和作用都有清晰的说明。
    // 检查并处理秒数到毫秒的转换，检测是否发生溢出
    PD_CHECK_OVERFLOW(scaleSecondsToMilliseconds(seconds, &milliseconds));
    // 将毫秒数与微秒数（除以1000得到的整数部分）相加，检测是否发生溢出
    PD_CHECK_OVERFLOW(
        checked_int64_add(milliseconds, dts->us / 1000, &milliseconds));

    // 返回计算后的毫秒数
    return milliseconds;
  }

  int64_t microseconds;
  // 检查并处理秒数到微秒的转换，检测是否发生溢出
  PD_CHECK_OVERFLOW(scaleSecondsToMicroseconds(seconds, &microseconds));
  // 将微秒数与微秒数相加，检测是否发生溢出
  PD_CHECK_OVERFLOW(checked_int64_add(microseconds, dts->us, &microseconds));

  if (base == NPY_FR_us) {
    // 如果时间基准是微秒，则直接返回微秒数
    return microseconds;
  }

  if (base == NPY_FR_ns) {
    int64_t nanoseconds;

    // 最小有效时间戳（以纳秒表示）：1677-09-21 00:12:43.145224193
    const int64_t min_nanoseconds = NPY_MIN_INT64 + 1;
    if (microseconds == min_nanoseconds / 1000 - 1) {
      // 对于接近最小纳秒时间戳的值，使用其作为基准，并加上微秒级别的偏移量，以避免在缩放期间溢出
      PD_CHECK_OVERFLOW(checked_int64_add(
          min_nanoseconds, (dts->ps - _NS_MIN_DTS.ps) / 1000, &nanoseconds));
    } else {
      // 将微秒数转换为纳秒，检测是否发生溢出
      PD_CHECK_OVERFLOW(
          scaleMicrosecondsToNanoseconds(microseconds, &nanoseconds));
      // 将纳秒数与皮秒数（除以1000得到的整数部分）相加，检测是否发生溢出
      PD_CHECK_OVERFLOW(
          checked_int64_add(nanoseconds, dts->ps / 1000, &nanoseconds));
    }

    // 返回计算后的纳秒数
    return nanoseconds;
  }

  int64_t picoseconds;
  // 将微秒数转换为皮秒，检测是否发生溢出
  PD_CHECK_OVERFLOW(scaleMicrosecondsToPicoseconds(microseconds, &picoseconds));
  // 将皮秒数与皮秒数相加，检测是否发生溢出
  PD_CHECK_OVERFLOW(checked_int64_add(picoseconds, dts->ps, &picoseconds));

  if (base == NPY_FR_ps) {
    // 如果时间基准是皮秒，则直接返回皮秒数
    return picoseconds;
  }

  if (base == NPY_FR_fs) {
    int64_t femtoseconds;
    // 将皮秒数转换为飞秒，检测是否发生溢出
    PD_CHECK_OVERFLOW(
        scalePicosecondsToFemtoseconds(picoseconds, &femtoseconds));
    // 将飞秒数与斯图夫阿美秒（除以1000得到的整数部分）相加，检测是否发生溢出
    PD_CHECK_OVERFLOW(
        checked_int64_add(femtoseconds, dts->as / 1000, &femtoseconds));
    // 返回计算后的飞秒数
    return femtoseconds;
  }

  if (base == NPY_FR_as) {
    int64_t attoseconds;
    // 将皮秒数转换为阿秒，检测是否发生溢出
    PD_CHECK_OVERFLOW(scalePicosecondsToAttoseconds(picoseconds, &attoseconds));
    // 将阿秒数与阿秒数相加，检测是否发生溢出
    PD_CHECK_OVERFLOW(checked_int64_add(attoseconds, dts->as, &attoseconds));
    // 返回计算后的阿秒数
    return attoseconds;
  }

  /* Something got corrupted */
  // 如果时间基准未知或损坏，则发出错误并返回-1
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyErr_SetString(PyExc_ValueError,
                  "NumPy datetime metadata with corrupt unit value");
  PyGILState_Release(gstate);

  // 返回-1表示错误
  return -1;
/*
 * Port numpy#13188 https://github.com/numpy/numpy/pull/13188/
 *
 * Computes the python `ret, d = divmod(d, unit)`.
 *
 * Note that GCC is smart enough at -O2 to eliminate the `if(*d < 0)` branch
 * for subsequent calls to this command - it is able to deduce that `*d >= 0`.
 */
npy_int64 extract_unit(npy_datetime *d, npy_datetime unit) {
  assert(unit > 0);  // 确保单位大于0，即非负
  npy_int64 div = *d / unit;  // 计算整除部分
  npy_int64 mod = *d % unit;  // 计算余数部分
  if (mod < 0) {  // 如果余数为负数
    mod += unit;  // 加上单位，使余数非负
    div -= 1;  // 对应的整除部分减去1
  }
  assert(mod >= 0);  // 确保余数非负
  *d = mod;  // 更新传入的日期时间变量为余数部分
  return div;  // 返回整除部分
}

/*
 * Converts a datetime based on the given metadata into a datetimestruct
 */
void pandas_datetime_to_datetimestruct(npy_datetime dt, NPY_DATETIMEUNIT base,
                                       npy_datetimestruct *out) {
  npy_int64 perday;

  /* Initialize the output to all zeros */
  memset(out, 0, sizeof(npy_datetimestruct));  // 将输出结构体初始化为全零

  out->year = 1970;  // 年份从1970年开始
  out->month = 1;    // 月份从1月开始
  out->day = 1;      // 日从1日开始

  /*
   * Note that care must be taken with the / and % operators
   * for negative values.
   */
  switch (base) {
  case NPY_FR_Y:
    out->year = 1970 + dt;  // 年份从1970年开始，加上传入的年份偏移量
    break;

  case NPY_FR_M:
    out->year = 1970 + extract_unit(&dt, 12);  // 年份从1970年开始，加上传入的月份偏移量
    out->month = (npy_int32)dt + 1;  // 月份从1月开始，加上传入的月份偏移量
    break;

  case NPY_FR_W:
    /* A week is 7 days */
    set_datetimestruct_days(dt * 7, out);  // 设置日期时间结构体的天数部分，乘以7是因为一周有7天
    break;

  case NPY_FR_D:
    set_datetimestruct_days(dt, out);  // 设置日期时间结构体的天数部分
    break;

  case NPY_FR_h:
    perday = 24LL;  // 每天24小时

    set_datetimestruct_days(extract_unit(&dt, perday), out);  // 设置日期时间结构体的天数部分，按小时计算
    out->hour = (npy_int32)dt;  // 设置小时部分
    break;

  case NPY_FR_m:
    perday = 24LL * 60;  // 每天的分钟数

    set_datetimestruct_days(extract_unit(&dt, perday), out);  // 设置日期时间结构体的天数部分，按分钟计算
    out->hour = (npy_int32)extract_unit(&dt, 60);  // 设置小时部分
    out->min = (npy_int32)dt;  // 设置分钟部分
    break;

  case NPY_FR_s:
    perday = 24LL * 60 * 60;  // 每天的秒数

    set_datetimestruct_days(extract_unit(&dt, perday), out);  // 设置日期时间结构体的天数部分，按秒计算
    out->hour = (npy_int32)extract_unit(&dt, 60 * 60);  // 设置小时部分
    out->min = (npy_int32)extract_unit(&dt, 60);  // 设置分钟部分
    out->sec = (npy_int32)dt;  // 设置秒部分
    break;

  case NPY_FR_ms:
    perday = 24LL * 60 * 60 * 1000;  // 每天的毫秒数

    set_datetimestruct_days(extract_unit(&dt, perday), out);  // 设置日期时间结构体的天数部分，按毫秒计算
    out->hour = (npy_int32)extract_unit(&dt, 1000LL * 60 * 60);  // 设置小时部分
    out->min = (npy_int32)extract_unit(&dt, 1000LL * 60);  // 设置分钟部分
    out->sec = (npy_int32)extract_unit(&dt, 1000LL);  // 设置秒部分
    out->us = (npy_int32)(dt * 1000);  // 设置微秒部分
    break;

  case NPY_FR_us:
    perday = 24LL * 60LL * 60LL * 1000LL * 1000LL;  // 每天的微秒数

    set_datetimestruct_days(extract_unit(&dt, perday), out);  // 设置日期时间结构体的天数部分，按微秒计算
    out->hour = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 60 * 60);  // 设置小时部分
    out->min = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 60);  // 设置分钟部分
    out->sec = (npy_int32)extract_unit(&dt, 1000LL * 1000);  // 设置秒部分
    out->us = (npy_int32)dt;  // 设置微秒部分
    break;

  case NPY_FR_ns:
    perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;  // 每天的纳秒数

    set_datetimestruct_days(extract_unit(&dt, perday), out);  // 设置日期时间结构体的天数部分，按纳秒计算
    out->hour = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 60 * 60);  // 设置小时部分
    out->min = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 60);  // 设置分钟部分
    out->sec = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000);  // 设置秒部分
    out->ns = (npy_int32)dt;  // 设置纳秒部分
    break;
  }
}
    // 设置秒数字段为从时间单位中提取的值（以纳秒为单位）
    out->sec = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000);
    // 设置微秒字段为从时间单位中提取的值（以微秒为单位）
    out->us = (npy_int32)extract_unit(&dt, 1000LL);
    // 设置皮秒字段为时间单位乘以1000得到的值（即皮秒为纳秒的一千倍）
    out->ps = (npy_int32)(dt * 1000);
    break;

  case NPY_FR_ps:
    // 计算一天的纳秒数
    perday = 24LL * 60 * 60 * 1000 * 1000 * 1000 * 1000;
    // 设置日期部分为从时间单位中提取的值，设置小时和分钟字段
    set_datetimestruct_days(extract_unit(&dt, perday), out);
    out->hour = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 60 * 60);
    out->min = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 60);
    // 设置秒、微秒和皮秒字段为从时间单位中提取的值
    out->sec = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000);
    out->us = (npy_int32)extract_unit(&dt, 1000LL);
    out->ps = (npy_int32)(dt * 1000);
    break;

  case NPY_FR_fs:
    /* entire range is only +- 2.6 hours */
    // 设置小时字段为从时间单位中提取的值（以飞秒为单位）
    out->hour = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000 *
                                                 1000 * 60 * 60);
    // 如果小时字段为负数，则设置日期为1969年12月31日，并修正小时字段
    if (out->hour < 0) {
      out->year = 1969;
      out->month = 12;
      out->day = 31;
      out->hour += 24;
      // 断言确保小时字段非负
      assert(out->hour >= 0);
    }
    // 设置分钟、秒、微秒和皮秒字段为从时间单位中提取的值
    out->min =
        (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000 * 1000 * 60);
    out->sec = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000 * 1000);
    out->us = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000);
    out->ps = (npy_int32)extract_unit(&dt, 1000LL * 1000);
    // 设置飞秒字段为时间单位的值
    out->as = (npy_int32)(dt * 1000);
    break;

  case NPY_FR_as:
    /* entire range is only +- 9.2 seconds */
    // 设置秒字段为从时间单位中提取的值（以太秒为单位）
    out->sec =
        (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000 * 1000 * 1000);
    // 如果秒字段为负数，则设置日期为1969年12月31日，并修正时间字段
    if (out->sec < 0) {
      out->year = 1969;
      out->month = 12;
      out->day = 31;
      out->hour = 23;
      out->min = 59;
      out->sec += 60;
      // 断言确保秒字段非负
      assert(out->sec >= 0);
    }
    // 设置微秒和皮秒字段为从时间单位中提取的值
    out->us = (npy_int32)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000);
    out->ps = (npy_int32)extract_unit(&dt, 1000LL * 1000);
    // 设置太秒字段为时间单位的值
    out->as = (npy_int32)dt;
    break;

  default:
    // 抛出异常，指示NumPy日期时间元数据损坏，带有无效的基本单位
    PyErr_SetString(PyExc_RuntimeError,
                    "NumPy datetime metadata is corrupted with invalid "
                    "base unit");
/*
 * Converts a timedelta from a timedeltastruct to a timedelta based
 * on a metadata unit. The timedelta is assumed to be valid.
 *
 * Returns 0 on success, -1 on failure.
 */
void pandas_timedelta_to_timedeltastruct(npy_timedelta td,
                                         NPY_DATETIMEUNIT base,
                                         pandas_timedeltastruct *out) {
  /* Initialize the output to all zeros */
  memset(out, 0, sizeof(pandas_timedeltastruct));

  // 定义每小时的秒数和每分钟的秒数
  const npy_int64 sec_per_hour = 3600;
  const npy_int64 sec_per_min = 60;

  switch (base) {
  case NPY_FR_W:
    // 基于周的时间增量转换为天数
    out->days = 7 * td;
    break;
  case NPY_FR_D:
    // 基于天的时间增量直接赋值给天数
    out->days = td;
    break;
  case NPY_FR_h:
    // 基于小时的时间增量转换为天数和小时数
    out->days = td / 24LL;
    td -= out->days * 24LL; // 从时间增量中减去整天的小时数
    out->hrs = (npy_int32)td; // 剩余的小时数赋值给小时字段
    break;
  case NPY_FR_m:
    // 基于分钟的时间增量转换为天数、小时数和分钟数
    out->days = td / 1440LL; // 计算整天数
    td -= out->days * 1440LL; // 减去整天的分钟数
    out->hrs = (npy_int32)(td / 60LL); // 计算剩余的小时数
    td -= out->hrs * 60LL; // 减去整小时的分钟数
    out->min = (npy_int32)td; // 剩余的分钟数赋值给分钟字段
    break;
  case NPY_FR_s:
  case NPY_FR_ms:
  case NPY_FR_us:
  case NPY_FR_ns: {
    // 定义每天的秒数
    const npy_int64 sec_per_day = 86400;
    npy_int64 per_sec;

    // 根据基础单位选择每秒的纳秒数、微秒数、毫秒数或秒数
    if (base == NPY_FR_s) {
      per_sec = 1;
    } else if (base == NPY_FR_ms) {
      per_sec = 1000;
    } else if (base == NPY_FR_us) {
      per_sec = 1000000;
    } else {
      per_sec = 1000000000;
    }

    // 计算每天的纳秒数
    const npy_int64 per_day = sec_per_day * per_sec;
    npy_int64 frac;

    // 将时间增量转换为秒数
    if (td < 0 && td % per_sec != 0)
      frac = td / per_sec - 1;
    else
      frac = td / per_sec;

    const int sign = frac < 0 ? -1 : 1;

    if (frac < 0) {
      // 处理负时间增量的情况
      if ((-frac % sec_per_day) != 0) {
        out->days = -frac / sec_per_day + 1;
        frac += sec_per_day * out->days;
      } else {
        frac = -frac;
      }
    }

    // 计算剩余的天数
    if (frac >= sec_per_day) {
      out->days += frac / sec_per_day;
      frac -= out->days * sec_per_day;
    }

    // 计算剩余的小时数
    if (frac >= sec_per_hour) {
      out->hrs = (npy_int32)(frac / sec_per_hour);
      frac -= out->hrs * sec_per_hour;
    }

    // 计算剩余的分钟数
    if (frac >= sec_per_min) {
      out->min = (npy_int32)(frac / sec_per_min);
      frac -= out->min * sec_per_min;
    }

    // 剩余的秒数赋值给秒字段
    if (frac >= 0) {
      out->sec = (npy_int32)frac;
      frac -= out->sec;
    }

    // 处理负数的情况，调整天数
    if (sign < 0)
      out->days = -out->days;

    // 根据基础单位进一步处理
    if (base > NPY_FR_s) {
      const npy_int64 sfrac =
          (out->hrs * sec_per_hour + out->min * sec_per_min + out->sec) *
          per_sec;

      npy_int64 ifrac = td - (out->days * per_day + sfrac);

      // 根据基础单位设置毫秒数、微秒数或纳秒数
      if (base == NPY_FR_ms) {
        out->ms = (npy_int32)ifrac;
      } else if (base == NPY_FR_us) {
        out->ms = (npy_int32)(ifrac / 1000LL);
        ifrac = ifrac % 1000LL;
        out->us = (npy_int32)ifrac;
      } else if (base == NPY_FR_ns) {
        out->ms = (npy_int32)(ifrac / (1000LL * 1000LL));
        ifrac = ifrac % (1000LL * 1000LL);
        out->us = (npy_int32)(ifrac / 1000LL);
        ifrac = ifrac % 1000LL;
        out->ns = (npy_int32)ifrac;
      }
    }
    break;
  }
}
  }

  } break;

这两行代码结尾。


  default:

处理默认情况。


    PyErr_SetString(PyExc_RuntimeError,
                    "NumPy timedelta metadata is corrupted with "
                    "invalid base unit");

设置运行时错误，指示NumPy时间增量元数据受损，基础单位无效。


    break;
  }

跳出switch语句。


  out->seconds =
      (npy_int32)(out->hrs * sec_per_hour + out->min * sec_per_min + out->sec);

计算输出结构体`out`的秒数，基于小时、分钟和秒数的总和。


  out->microseconds = out->ms * 1000 + out->us;

计算输出结构体`out`的微秒数，基于毫秒和微秒的总和。


  out->nanoseconds = out->ns;

将输出结构体`out`的纳秒数设置为变量`ns`的值。
/*
 * This function returns a pointer to the DateTimeMetaData
 * contained within the provided datetime dtype.
 *
 * Copied near-verbatim from numpy/core/src/multiarray/datetime.c
 */
PyArray_DatetimeMetaData
get_datetime_metadata_from_dtype(PyArray_Descr *dtype) {
#if NPY_ABI_VERSION < 0x02000000
#define PyDataType_C_METADATA(dtype) ((dtype)->c_metadata)
#endif
  // 使用提供的 datetime 数据类型获取其中的 DateTimeMetaData 指针
  return ((PyArray_DatetimeDTypeMetaData *)PyDataType_C_METADATA(dtype))->meta;
}
```