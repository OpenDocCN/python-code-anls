# `.\numpy\numpy\_core\src\multiarray\datetime_busday.c`

```py
/*
 * This file implements business day functionality for NumPy datetime.
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

#include <numpy/arrayobject.h>

#include "npy_config.h"


#include "numpy/arrayscalars.h"
#include "lowlevel_strided_loops.h"
#include "_datetime.h"
#include "datetime_busday.h"
#include "datetime_busdaycal.h"

/* Gets the day of the week for a datetime64[D] value */
static int
get_day_of_week(npy_datetime date)
{
    int day_of_week;

    /* Get the day of the week for 'date' (1970-01-05 is Monday) */
    day_of_week = (int)((date - 4) % 7);
    if (day_of_week < 0) {
        day_of_week += 7;
    }

    return day_of_week;
}

/*
 * Returns 1 if the date is a holiday (contained in the sorted
 * list of dates), 0 otherwise.
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
static int
is_holiday(npy_datetime date,
            npy_datetime *holidays_begin, const npy_datetime *holidays_end)
{
    npy_datetime *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            holidays_end = trial;
        }
        else if (date > *trial) {
            holidays_begin = trial + 1;
        }
        else {
            return 1;  // Found the holiday
        }
    }

    /* Not found */
    return 0;
}

/*
 * Finds the earliest holiday which is on or after 'date'. If 'date' does not
 * appear within the holiday range, returns 'holidays_begin' if 'date'
 * is before all holidays, or 'holidays_end' if 'date' is after all
 * holidays.
 *
 * To remove all the holidays before 'date' from a holiday range, do:
 *
 *      holidays_begin = find_holiday_earliest_on_or_after(date,
 *                                          holidays_begin, holidays_end);
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
static npy_datetime *
find_earliest_holiday_on_or_after(npy_datetime date,
            npy_datetime *holidays_begin, const npy_datetime *holidays_end)
{
    npy_datetime *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            holidays_end = trial;  // Adjust upper bound of search
        }
        else if (date > *trial) {
            holidays_begin = trial + 1;  // Adjust lower bound of search
        }
        else {
            return trial;  // Found exact match for the holiday
        }
    }

    return holidays_begin;  // Return nearest holiday date after 'date'
}
/*
 * Finds the earliest holiday which is after 'date'. If 'date' does not
 * appear within the holiday range, returns 'holidays_begin' if 'date'
 * is before all holidays, or 'holidays_end' if 'date' is after all
 * holidays.
 *
 * To remove all the holidays after 'date' from a holiday range, do:
 *
 *      holidays_end = find_holiday_earliest_after(date,
 *                                          holidays_begin, holidays_end);
 *
 * The holidays list should be normalized, which means any NaT (not-a-time)
 * values, duplicates, and dates already excluded by the weekmask should
 * be removed, and the list should be sorted.
 */
static npy_datetime *
find_earliest_holiday_after(npy_datetime date,
            npy_datetime *holidays_begin, const npy_datetime *holidays_end)
{
    npy_datetime *trial;

    /* Simple binary search */
    while (holidays_begin < holidays_end) {
        // Calculate the middle point of the holidays range
        trial = holidays_begin + (holidays_end - holidays_begin) / 2;

        if (date < *trial) {
            // Adjust the holidays_end pointer to the middle point
            holidays_end = trial;
        }
        else if (date > *trial) {
            // Move holidays_begin past the middle point
            holidays_begin = trial + 1;
        }
        else {
            // 'date' matches a holiday, return the next holiday
            return trial + 1;
        }
    }

    // Return holidays_begin if 'date' is not found
    return holidays_begin;
}

/*
 * Applies the 'roll' strategy to 'date', placing the result in 'out'
 * and setting 'out_day_of_week' to the day of the week that results.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
apply_business_day_roll(npy_datetime date, npy_datetime *out,
                    int *out_day_of_week,
                    NPY_BUSDAY_ROLL roll,
                    const npy_bool *weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    int day_of_week;

    /* Deal with NaT input */
    if (date == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        if (roll == NPY_BUSDAY_RAISE) {
            // Raise an error for NaT input if roll strategy demands it
            PyErr_SetString(PyExc_ValueError,
                    "NaT input in busday_offset");
            return -1;
        }
        else {
            // Return success for NaT input if no raise strategy
            return 0;
        }
    }

    /* Get the day of the week for 'date' */
    day_of_week = get_day_of_week(date);

    /* Apply the 'roll' if it's not a business day */
    /*
     * This part of the function applies the specified business day roll
     * strategy to adjust the given date according to business day rules,
     * considering the provided weekmask and holidays range.
     */
    # 如果给定日期是非工作日或者是节假日，需要进行偏移计算
    if (weekmask[day_of_week] == 0 ||
                        is_holiday(date, holidays_begin, holidays_end)) {
        # 记录开始日期和星期几
        npy_datetime start_date = date;
        int start_day_of_week = day_of_week;

        # 根据不同的偏移规则进行处理
        switch (roll) {
            case NPY_BUSDAY_FOLLOWING:
            case NPY_BUSDAY_MODIFIEDFOLLOWING: {
                # 向后偏移直到找到下一个工作日
                do {
                    ++date;
                    if (++day_of_week == 7) {
                        day_of_week = 0;
                    }
                } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));

                # 如果是修改后向后偏移规则，且跨月了，执行前向偏移
                if (roll == NPY_BUSDAY_MODIFIEDFOLLOWING) {
                    /* 如果跨月了，则执行前向偏移 */
                    if (days_to_month_number(start_date) !=
                                days_to_month_number(date)) {
                        date = start_date;
                        day_of_week = start_day_of_week;

                        do {
                            --date;
                            if (--day_of_week == -1) {
                                day_of_week = 6;
                            }
                        } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));
                    }
                }
                break;
            }
            case NPY_BUSDAY_PRECEDING:
            case NPY_BUSDAY_MODIFIEDPRECEDING: {
                # 向前偏移直到找到上一个工作日
                do {
                    --date;
                    if (--day_of_week == -1) {
                        day_of_week = 6;
                    }
                } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));

                # 如果是修改后向前偏移规则，且跨月了，执行后向偏移
                if (roll == NPY_BUSDAY_MODIFIEDPRECEDING) {
                    /* 如果跨月了，则执行后向偏移 */
                    if (days_to_month_number(start_date) !=
                                days_to_month_number(date)) {
                        date = start_date;
                        day_of_week = start_day_of_week;

                        do {
                            ++date;
                            if (++day_of_week == 7) {
                                day_of_week = 0;
                            }
                        } while (weekmask[day_of_week] == 0 ||
                            is_holiday(date, holidays_begin, holidays_end));
                    }
                }
                break;
            }
            case NPY_BUSDAY_NAT: {
                # 返回 NaT（Not a Time）表示非工作日
                date = NPY_DATETIME_NAT;
                break;
            }
            case NPY_BUSDAY_RAISE: {
                # 抛出值错误异常，表示在 busday_offset 中遇到非工作日
                *out = NPY_DATETIME_NAT;
                PyErr_SetString(PyExc_ValueError,
                        "Non-business day date in busday_offset");
                return -1;
            }
        }
    }

    # 将计算后的日期和星期几返回给输出参数
    *out = date;
    *out_day_of_week = day_of_week;

    # 返回成功
    return 0;
/*
 * Applies a single business day offset. See the function
 * business_day_offset for the meaning of all the parameters.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
apply_business_day_offset(npy_datetime date, npy_int64 offset,
                    npy_datetime *out,
                    NPY_BUSDAY_ROLL roll,
                    npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    int day_of_week = 0;
    npy_datetime *holidays_temp;

    /* Roll the date to a business day */
    // 尝试将日期调整为工作日
    if (apply_business_day_roll(date, &date, &day_of_week,
                                roll,
                                weekmask,
                                holidays_begin, holidays_end) < 0) {
        return -1;
    }

    /* If we get a NaT, just return it */
    // 如果得到 NaT（不合法日期），直接返回它
    if (date == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        return 0;
    }

    /* Now we're on a valid business day */
    // 现在日期在一个有效的工作日上

    if (offset > 0) {
        /* Remove any earlier holidays */
        // 移除早于当前日期的所有节假日
        holidays_begin = find_earliest_holiday_on_or_after(date,
                                            holidays_begin, holidays_end);

        /* Jump by as many weeks as we can */
        // 跳跃尽可能多的整周
        date += (offset / busdays_in_weekmask) * 7;
        offset = offset % busdays_in_weekmask;

        /* Adjust based on the number of holidays we crossed */
        // 根据经过的节假日数量调整偏移量
        holidays_temp = find_earliest_holiday_after(date,
                                            holidays_begin, holidays_end);
        offset += holidays_temp - holidays_begin;
        holidays_begin = holidays_temp;

        /* Step until we use up the rest of the offset */
        // 逐步增加日期，直到偏移量用尽
        while (offset > 0) {
            ++date;
            if (++day_of_week == 7) {
                day_of_week = 0;
            }
            // 如果当前日期是工作日并且不是节假日，则减少偏移量
            if (weekmask[day_of_week] && !is_holiday(date,
                                            holidays_begin, holidays_end)) {
                offset--;
            }
        }
    }
    else if (offset < 0) {
        /* 如果偏移量小于零，表示要向前调整日期 */

        /* 找到日期之后最早的假日 */
        holidays_end = find_earliest_holiday_after(date,
                                            holidays_begin, holidays_end);

        /* 尽可能地向前跳过整数个星期 */
        date += (offset / busdays_in_weekmask) * 7;
        offset = offset % busdays_in_weekmask;

        /* 根据跨越的假期数量进行调整 */
        holidays_temp = find_earliest_holiday_on_or_after(date,
                                            holidays_begin, holidays_end);
        offset -= holidays_end - holidays_temp;
        holidays_end = holidays_temp;

        /* 逐步向前调整，直到偏移量用尽 */
        while (offset < 0) {
            --date;
            if (--day_of_week == -1) {
                day_of_week = 6;
            }
            if (weekmask[day_of_week] && !is_holiday(date,
                                            holidays_begin, holidays_end)) {
                offset++;
            }
        }
    }

    /* 将计算出的日期写入输出指针指向的位置 */
    *out = date;
    /* 返回执行成功的标志 */
    return 0;
/*
 * 对日期范围应用单个工作日计数操作。参见函数 business_day_count 的参数含义。
 *
 * 在成功时返回 0，在失败时返回 -1。
 */
static int
apply_business_day_count(npy_datetime date_begin, npy_datetime date_end,
                    npy_int64 *out,
                    const npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    npy_int64 count, whole_weeks;

    int day_of_week = 0;
    int swapped = 0;

    /* 如果日期为 NaT，则引发错误 */
    if (date_begin == NPY_DATETIME_NAT || date_end == NPY_DATETIME_NAT) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot compute a business day count with a NaT (not-a-time) "
                "date");
        return -1;
    }

    /* 空日期范围 */
    if (date_begin == date_end) {
        *out = 0;
        return 0;
    }
    else if (date_begin > date_end) {
        /* 交换日期顺序以确保 date_begin <= date_end */
        npy_datetime tmp = date_begin;
        date_begin = date_end;
        date_end = tmp;
        swapped = 1;
        // 由于交换了 date_begin 和 date_end，需要对原始 date_end 进行修正。gh-23197
        date_begin++;
        date_end++;
    }

    /* 删除范围内较早的节假日 */
    holidays_begin = find_earliest_holiday_on_or_after(date_begin,
                                        holidays_begin, holidays_end);
    /* 删除范围内较晚的节假日 */
    holidays_end = find_earliest_holiday_on_or_after(date_end,
                                        holidays_begin, holidays_end);

    /* 以负数开始计数，表示范围内的节假日数量 */
    count = -(holidays_end - holidays_begin);

    /* 添加 date_begin 和 date_end 之间的整周工作日数量 */
    whole_weeks = (date_end - date_begin) / 7;
    count += whole_weeks * busdays_in_weekmask;
    date_begin += whole_weeks * 7;

    if (date_begin < date_end) {
        /* 获取 date_begin 的星期几 */
        day_of_week = get_day_of_week(date_begin);

        /* 逐日计算剩余天数 */
        while (date_begin < date_end) {
            if (weekmask[day_of_week]) {
                count++;
            }
            ++date_begin;
            if (++day_of_week == 7) {
                day_of_week = 0;
            }
        }
    }

    /* 如果交换了日期顺序，则结果取负 */
    if (swapped) {
        count = -count;
    }

    *out = count;
    return 0;
}
/*
 * Applies the given offsets in business days to the dates provided.
 * This is the low-level function which requires already cleaned input
 * data.
 *
 * dates:    An array of dates with 'datetime64[D]' data type.
 * offsets:  An array safely convertible into type int64.
 * out:      Either NULL, or an array with 'datetime64[D]' data type
 *              in which to place the resulting dates.
 * roll:     A rule for how to treat non-business day dates.
 * weekmask: A 7-element boolean mask, 1 for possible business days and 0
 *              for non-business days.
 * busdays_in_weekmask: A count of how many 1's there are in weekmask.
 * holidays_begin/holidays_end: A sorted list of dates matching '[D]'
 *           unit metadata, with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
 *
 * For each (date, offset) in the broadcasted pair of (dates, offsets),
 * does the following:
 *  + Applies the 'roll' rule to the date to either produce NaT, raise
 *    an exception, or land on a valid business day.
 *  + Adds 'offset' business days to the valid business day found.
 *  + Sets the value in 'out' if provided, or the allocated output array
 *    otherwise.
 */
NPY_NO_EXPORT PyArrayObject *
business_day_offset(PyArrayObject *dates, PyArrayObject *offsets,
                    PyArrayObject *out,
                    NPY_BUSDAY_ROLL roll,
                    npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    // 临时的日期时间元数据
    PyArray_DatetimeMetaData temp_meta;
    // 数据类型数组，用于存放日期时间和偏移量的数据类型描述符
    PyArray_Descr *dtypes[3] = {NULL, NULL, NULL};

    // 迭代器指针和操作对象数组
    NpyIter *iter = NULL;
    PyArrayObject *op[3] = {NULL, NULL, NULL};
    // 操作标志数组和迭代器标志
    npy_uint32 op_flags[3], flags;

    // 返回的结果数组
    PyArrayObject *ret = NULL;

    // 如果周掩码中的有效工作日数为0，则抛出值错误异常并返回NULL
    if (busdays_in_weekmask == 0) {
        PyErr_SetString(PyExc_ValueError,
                "the business day weekmask must have at least one "
                "valid business day");
        return NULL;
    }

    /* First create the data types for dates and offsets */
    // 设置日期时间的基础类型和单位数量
    temp_meta.base = NPY_FR_D;
    temp_meta.num = 1;
    // 创建日期时间数据类型
    dtypes[0] = create_datetime_dtype(NPY_DATETIME, &temp_meta);
    if (dtypes[0] == NULL) {
        goto fail;
    }
    // 创建int64类型的偏移量数据类型
    dtypes[1] = PyArray_DescrFromType(NPY_INT64);
    if (dtypes[1] == NULL) {
        goto fail;
    }
    // 第三个数据类型和第一个相同
    dtypes[2] = dtypes[0];
    Py_INCREF(dtypes[2]);

    /* Set up the iterator parameters */
    // 设置迭代器标志
    flags = NPY_ITER_EXTERNAL_LOOP|
            NPY_ITER_BUFFERED|
            NPY_ITER_ZEROSIZE_OK;
    // 设置操作数组和操作标志
    op[0] = dates;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[1] = offsets;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[2] = out;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    /* Allocate the iterator */
    // 分配迭代器
    iter = NpyIter_MultiNew(3, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto fail;
    }
    # 如果迭代器的大小大于0，则进入循环处理所有元素
    if (NpyIter_GetIterSize(iter) > 0) {
        # 获取迭代器的下一个函数
        NpyIter_IterNextFunc *iternext;
        # 数据指针数组，步长数组和内部循环大小指针
        char **dataptr;
        npy_intp *strideptr, *innersizeptr;

        # 获取迭代器的下一个函数，若获取失败则跳转到失败标签
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        # 获取数据指针数组
        dataptr = NpyIter_GetDataPtrArray(iter);
        # 获取步长数组
        strideptr = NpyIter_GetInnerStrideArray(iter);
        # 获取内部循环大小指针
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        # 开始迭代处理每一个元素
        do {
            # 获取当前迭代器中各个数据指针
            char *data_dates = dataptr[0];
            char *data_offsets = dataptr[1];
            char *data_out = dataptr[2];
            # 获取当前迭代器中各个数据的步长
            npy_intp stride_dates = strideptr[0];
            npy_intp stride_offsets = strideptr[1];
            npy_intp stride_out = strideptr[2];
            # 获取当前内部循环的大小
            npy_intp count = *innersizeptr;

            # 在当前元素计数大于0时循环处理
            while (count--) {
                # 调用函数处理工作日偏移，并检查是否小于0，若是则跳转到失败标签
                if (apply_business_day_offset(*(npy_int64 *)data_dates,
                                       *(npy_int64 *)data_offsets,
                                       (npy_int64 *)data_out,
                                       roll,
                                       weekmask, busdays_in_weekmask,
                                       holidays_begin, holidays_end) < 0) {
                    goto fail;
                }

                # 更新数据指针位置
                data_dates += stride_dates;
                data_offsets += stride_offsets;
                data_out += stride_out;
            }
        } while (iternext(iter));  # 继续迭代直至所有元素处理完毕
    }

    # 从迭代器中获取返回对象
    ret = NpyIter_GetOperandArray(iter)[2];
    # 增加返回对象的引用计数
    Py_INCREF(ret);

    # 跳转到完成标签
    goto finish;
fail:
    // 释放之前分配的返回值对象，以避免内存泄漏
    Py_XDECREF(ret);
    // 将返回值对象设为NULL，以确保失败状态被传播
    ret = NULL;

finish:
    // 释放第一个日期数据类型对象
    Py_XDECREF(dtypes[0]);
    // 释放第二个日期数据类型对象
    Py_XDECREF(dtypes[1]);
    // 释放第三个整数数据类型对象
    Py_XDECREF(dtypes[2]);
    // 如果迭代器不为空，则进行迭代器的释放和清理
    if (iter != NULL) {
        // 调用 NpyIter_Deallocate 函数释放迭代器资源，检查是否成功
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            // 如果释放失败，再次释放返回值对象并将其设为NULL
            Py_XDECREF(ret);
            ret = NULL;
        }
    }
    // 返回函数的结果对象，可能是正确结果或NULL（失败情况）
    return ret;
}

/*
 * Counts the number of business days between two dates, not including
 * the end date. This is the low-level function which requires already
 * cleaned input data.
 *
 * If dates_begin is before dates_end, the result is positive.  If
 * dates_begin is after dates_end, it is negative.
 *
 * dates_begin:  An array of dates with 'datetime64[D]' data type.
 * dates_end:    An array of dates with 'datetime64[D]' data type.
 * out:      Either NULL, or an array with 'int64' data type
 *              in which to place the resulting dates.
 * weekmask: A 7-element boolean mask, 1 for possible business days and 0
 *              for non-business days.
 * busdays_in_weekmask: A count of how many 1's there are in weekmask.
 * holidays_begin/holidays_end: A sorted list of dates matching '[D]'
 *           unit metadata, with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
 */
NPY_NO_EXPORT PyArrayObject *
business_day_count(PyArrayObject *dates_begin, PyArrayObject *dates_end,
                    PyArrayObject *out,
                    npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    PyArray_DatetimeMetaData temp_meta;
    PyArray_Descr *dtypes[3] = {NULL, NULL, NULL};

    NpyIter *iter = NULL;
    PyArrayObject *op[3] = {NULL, NULL, NULL};
    npy_uint32 op_flags[3], flags;

    PyArrayObject *ret = NULL;

    // 检查是否有足够的有效工作日来填充输出数组
    if (busdays_in_weekmask == 0) {
        PyErr_SetString(PyExc_ValueError,
                "the business day weekmask must have at least one "
                "valid business day");
        return NULL;
    }

    /* First create the data types for the dates and the int64 output */
    // 初始化日期数据类型为 NPY_FR_D
    temp_meta.base = NPY_FR_D;
    temp_meta.num = 1;
    // 创建日期数据类型并检查创建是否成功
    dtypes[0] = create_datetime_dtype(NPY_DATETIME, &temp_meta);
    if (dtypes[0] == NULL) {
        goto fail;
    }
    // 递增引用第一个日期数据类型对象，以便在释放时不重复释放
    Py_INCREF(dtypes[0]);
    // 第二个数据类型对象与第一个相同，增加引用计数
    dtypes[1] = dtypes[0];
    Py_INCREF(dtypes[1]);
    // 创建 int64 类型数据类型对象并检查创建是否成功
    dtypes[2] = PyArray_DescrFromType(NPY_INT64);
    if (dtypes[2] == NULL) {
        goto fail;
    }

    /* Set up the iterator parameters */
    // 设置迭代器标志参数
    flags = NPY_ITER_EXTERNAL_LOOP |
            NPY_ITER_BUFFERED |
            NPY_ITER_ZEROSIZE_OK;
    // 设置迭代器操作数及其标志
    op[0] = dates_begin;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[1] = dates_end;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[2] = out;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    /* Allocate the iterator */
    // 分配多重迭代器并检查是否成功
    iter = NpyIter_MultiNew(3, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto fail;
    }
    /* 遍历所有元素 */
    if (NpyIter_GetIterSize(iter) > 0) {
        // 获取迭代器的下一个函数
        NpyIter_IterNextFunc *iternext;
        // 数据指针数组，用于访问迭代器中的数据
        char **dataptr;
        // 内部步长数组，指示每个数据块的步长
        npy_intp *strideptr, *innersizeptr;

        // 获取迭代器的下一个函数，若获取失败则跳转到失败标签
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        // 获取数据指针数组，用于访问迭代器中的数据
        dataptr = NpyIter_GetDataPtrArray(iter);
        // 获取内部步长数组，指示每个数据块的步长
        strideptr = NpyIter_GetInnerStrideArray(iter);
        // 获取内部循环大小指针，指示每个数据块的大小
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        // 迭代处理每个数据块
        do {
            // 获取日期开始和结束数据的指针
            char *data_dates_begin = dataptr[0];
            char *data_dates_end = dataptr[1];
            // 获取输出数据的指针
            char *data_out = dataptr[2];
            // 获取日期开始和结束数据的步长
            npy_intp stride_dates_begin = strideptr[0];
            npy_intp stride_dates_end = strideptr[1];
            // 获取输出数据的步长
            npy_intp stride_out = strideptr[2];
            // 获取当前数据块的大小
            npy_intp count = *innersizeptr;

            // 遍历当前数据块中的每个元素
            while (count--) {
                // 调用业务日计算函数，如果返回小于0，则跳转到失败标签
                if (apply_business_day_count(*(npy_int64 *)data_dates_begin,
                                       *(npy_int64 *)data_dates_end,
                                       (npy_int64 *)data_out,
                                       weekmask, busdays_in_weekmask,
                                       holidays_begin, holidays_end) < 0) {
                    goto fail;
                }

                // 更新数据指针，移动到下一个元素
                data_dates_begin += stride_dates_begin;
                data_dates_end += stride_dates_end;
                data_out += stride_out;
            }
        } while (iternext(iter));  // 继续处理下一个数据块，直到迭代结束
    }

    /* 从迭代器获取返回对象 */
    ret = NpyIter_GetOperandArray(iter)[2];
    // 增加返回对象的引用计数
    Py_INCREF(ret);

    // 跳转到完成标签
    goto finish;
fail:
    // 释放之前分配的 ret 对象，防止内存泄漏
    Py_XDECREF(ret);
    // 将 ret 设为 NULL，以确保清理完成
    ret = NULL;

finish:
    // 释放之前分配的 dtypes 数组中的对象
    Py_XDECREF(dtypes[0]);
    Py_XDECREF(dtypes[1]);
    Py_XDECREF(dtypes[2]);
    // 如果 iter 不为 NULL，释放 NpyIter 对象
    if (iter != NULL) {
        // 如果释放过程中出错，再次释放 ret 并置为 NULL
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_XDECREF(ret);
            ret = NULL;
        }
    }
    // 返回 ret 对象，可能为 NULL 或有效的 PyArrayObject 指针
    return ret;
}

/*
 * Returns a boolean array with True for input dates which are valid
 * business days, and False for dates which are not. This is the
 * low-level function which requires already cleaned input data.
 *
 * dates:  An array of dates with 'datetime64[D]' data type.
 * out:    Either NULL, or an array with 'bool' data type
 *         in which to place the resulting dates.
 * weekmask: A 7-element boolean mask, 1 for possible business days and 0
 *           for non-business days.
 * busdays_in_weekmask: A count of how many 1's there are in weekmask.
 * holidays_begin/holidays_end: A sorted list of dates matching '[D]'
 *                              unit metadata, with any dates falling on a day of the
 *                              week without weekmask[i] == 1 already filtered out.
 */
NPY_NO_EXPORT PyArrayObject *
is_business_day(PyArrayObject *dates, PyArrayObject *out,
                    const npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    PyArray_DatetimeMetaData temp_meta;
    PyArray_Descr *dtypes[2] = {NULL, NULL};

    NpyIter *iter = NULL;
    PyArrayObject *op[2] = {NULL, NULL};
    npy_uint32 op_flags[2], flags;

    PyArrayObject *ret = NULL;

    // 检查是否至少有一个有效的工作日
    if (busdays_in_weekmask == 0) {
        PyErr_SetString(PyExc_ValueError,
                "the business day weekmask must have at least one "
                "valid business day");
        return NULL;
    }

    /* First create the data types for the dates and the bool output */
    // 创建日期和布尔输出的数据类型
    temp_meta.base = NPY_FR_D;
    temp_meta.num = 1;
    dtypes[0] = create_datetime_dtype(NPY_DATETIME, &temp_meta);
    // 如果创建失败，则跳转到 fail 标签
    if (dtypes[0] == NULL) {
        goto fail;
    }
    dtypes[1] = PyArray_DescrFromType(NPY_BOOL);
    // 如果创建失败，则跳转到 fail 标签
    if (dtypes[1] == NULL) {
        goto fail;
    }

    /* Set up the iterator parameters */
    // 设置迭代器参数
    flags = NPY_ITER_EXTERNAL_LOOP|
            NPY_ITER_BUFFERED|
            NPY_ITER_ZEROSIZE_OK;
    op[0] = dates;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[1] = out;
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    /* Allocate the iterator */
    // 分配迭代器
    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    // 如果分配失败，则跳转到 fail 标签
    if (iter == NULL) {
        goto fail;
    }

    /* Loop over all elements */
    // 循环处理所有元素
    // 检查迭代器的大小是否大于零
    if (NpyIter_GetIterSize(iter) > 0) {
        // 获取迭代器的下一个迭代函数
        NpyIter_IterNextFunc *iternext;
        // 数据指针数组，用于访问迭代器中的数据
        char **dataptr;
        // 内部步长数组，用于访问迭代器中数据的步长
        npy_intp *strideptr, *innersizeptr;

        // 获取迭代器的下一个迭代函数，并检查是否为 NULL
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            // 如果获取迭代函数失败，则跳转到失败处理标签
            goto fail;
        }
        // 获取数据指针数组，用于访问迭代器中的数据
        dataptr = NpyIter_GetDataPtrArray(iter);
        // 获取内部步长数组，用于访问迭代器中数据的步长
        strideptr = NpyIter_GetInnerStrideArray(iter);
        // 获取内部循环大小指针，指向当前迭代器内部循环的大小
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        // 迭代处理每个数据元素
        do {
            // 获取日期数据指针和输出数据指针
            char *data_dates = dataptr[0];
            char *data_out = dataptr[1];
            // 获取日期数据的步长和输出数据的步长
            npy_intp stride_dates = strideptr[0];
            npy_intp stride_out = strideptr[1];
            // 获取当前内部循环的计数值
            npy_intp count = *innersizeptr;

            // 定义日期和星期几变量
            npy_datetime date;
            int day_of_week;

            // 遍历当前内部循环中的每个数据元素
            while (count--) {
                /* 检查日期是否是工作日 */
                // 获取当前日期
                date = *(npy_datetime *)data_dates;
                // 获取日期对应的星期几
                day_of_week = get_day_of_week(date);
                // 判断当前日期是否是工作日，根据给定的工作日掩码和节假日范围进行判断
                *(npy_bool *)data_out = weekmask[day_of_week] &&
                                        !is_holiday(date,
                                            holidays_begin, holidays_end) &&
                                        date != NPY_DATETIME_NAT;

                // 更新日期数据指针和输出数据指针
                data_dates += stride_dates;
                data_out += stride_out;
            }
        } while (iternext(iter));  // 继续迭代下一个内部循环

    }

    /* 从迭代器中获取返回对象 */
    // 获取迭代器操作数数组中的第二个操作数作为返回对象
    ret = NpyIter_GetOperandArray(iter)[1];
    // 增加返回对象的引用计数
    Py_INCREF(ret);

    // 跳转到结束标签，完成处理
    goto finish;
fail:
    // 释放之前分配的 Python 对象并将 ret 设为 NULL
    Py_XDECREF(ret);
    ret = NULL;

finish:
    // 释放第一个和第二个数据类型对象的引用
    Py_XDECREF(dtypes[0]);
    Py_XDECREF(dtypes[1]);
    // 如果 iter 不为 NULL，则进行迭代器的释放
    if (iter != NULL) {
        // 如果迭代器释放失败，则再次释放 ret 并将其设为 NULL
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_XDECREF(ret);
            ret = NULL;
        }
    }
    // 返回最终的 ret 值
    return ret;
}

static int
PyArray_BusDayRollConverter(PyObject *roll_in, NPY_BUSDAY_ROLL *roll)
{
    // 初始化 obj 为 roll_in
    PyObject *obj = roll_in;

    /* Make obj into an UTF8 string */
    // 如果 obj 是字节对象，则将其转换为 UTF-8 编码的字符串
    if (PyBytes_Check(obj)) {
        /* accept bytes input */
        PyObject *obj_str = PyUnicode_FromEncodedObject(obj, NULL, NULL);
        // 如果转换失败，则返回 0
        if (obj_str == NULL) {
            return 0;
        }
        // 将 obj_str 赋给 obj
        obj = obj_str;
    }
    else {
        // 增加 obj 的引用计数
        Py_INCREF(obj);
    }

    Py_ssize_t len;
    // 将 obj 转换为 UTF-8 编码的字符串，并获取其长度
    char const *str = PyUnicode_AsUTF8AndSize(obj, &len);
    // 如果转换失败，则释放 obj 并返回 0
    if (str == NULL) {
        Py_DECREF(obj);
        return 0;
    }

    /* Use switch statements to quickly isolate the right enum value */
    // 使用 switch 语句快速匹配正确的枚举值
    switch (str[0]) {
        case 'b':
            if (strcmp(str, "backward") == 0) {
                *roll = NPY_BUSDAY_BACKWARD;
                // 转到 finish 标签，完成转换
                goto finish;
            }
            break;
        case 'f':
            if (len > 2) switch (str[2]) {
                case 'r':
                    if (strcmp(str, "forward") == 0) {
                        *roll = NPY_BUSDAY_FORWARD;
                        // 转到 finish 标签，完成转换
                        goto finish;
                    }
                    break;
                case 'l':
                    if (strcmp(str, "following") == 0) {
                        *roll = NPY_BUSDAY_FOLLOWING;
                        // 转到 finish 标签，完成转换
                        goto finish;
                    }
                    break;
            }
            break;
        case 'm':
            if (len > 8) switch (str[8]) {
                case 'f':
                    if (strcmp(str, "modifiedfollowing") == 0) {
                        *roll = NPY_BUSDAY_MODIFIEDFOLLOWING;
                        // 转到 finish 标签，完成转换
                        goto finish;
                    }
                    break;
                case 'p':
                    if (strcmp(str, "modifiedpreceding") == 0) {
                        *roll = NPY_BUSDAY_MODIFIEDPRECEDING;
                        // 转到 finish 标签，完成转换
                        goto finish;
                    }
                    break;
            }
            break;
        case 'n':
            if (strcmp(str, "nat") == 0) {
                *roll = NPY_BUSDAY_NAT;
                // 转到 finish 标签，完成转换
                goto finish;
            }
            break;
        case 'p':
            if (strcmp(str, "preceding") == 0) {
                *roll = NPY_BUSDAY_PRECEDING;
                // 转到 finish 标签，完成转换
                goto finish;
            }
            break;
        case 'r':
            if (strcmp(str, "raise") == 0) {
                *roll = NPY_BUSDAY_RAISE;
                // 转到 finish 标签，完成转换
                goto finish;
            }
            break;
    }

    // 如果没有匹配的枚举值，则抛出格式化的值错误异常
    PyErr_Format(PyExc_ValueError,
            "Invalid business day roll parameter \"%s\"",
            str);
    // 释放 obj 并返回 0，表示转换失败
    Py_DECREF(obj);
    return 0;

finish:
    // 完成转换后释放 obj，并返回 1，表示转换成功
    Py_DECREF(obj);
    return 1;
}
/*
 * This is the 'busday_offset' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_offset(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"dates", "offsets", "roll",
                             "weekmask", "holidays", "busdaycal", "out", NULL};

    PyObject *dates_in = NULL, *offsets_in = NULL, *out_in = NULL;

    PyArrayObject *dates = NULL, *offsets = NULL, *out = NULL, *ret;
    NPY_BUSDAY_ROLL roll = NPY_BUSDAY_RAISE;
    npy_bool weekmask[7] = {2, 1, 1, 1, 1, 0, 0};
    NpyBusDayCalendar *busdaycal = NULL;
    int i, busdays_in_weekmask;
    npy_holidayslist holidays = {NULL, NULL};
    int allocated_holidays = 1;

    /*
     * Parse Python arguments into C variables using specified keywords.
     * 'dates_in', 'offsets_in', 'roll', 'weekmask', 'holidays',
     * 'busdaycal', 'out' are extracted from args and kwds.
     */
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                    "OO|O&O&O&O!O:busday_offset", kwlist,
                                    &dates_in,
                                    &offsets_in,
                                    &PyArray_BusDayRollConverter, &roll,
                                    &PyArray_WeekMaskConverter, &weekmask[0],
                                    &PyArray_HolidaysConverter, &holidays,
                                    &NpyBusDayCalendar_Type, &busdaycal,
                                    &out_in)) {
        goto fail;
    }

    /* Make sure only one of the weekmask/holidays and busdaycal is supplied */
    if (busdaycal != NULL) {
        /*
         * Check if both weekmask and holidays are provided, which is invalid.
         * Set an error and jump to failure handling.
         */
        if (weekmask[0] != 2 || holidays.begin != NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Cannot supply both the weekmask/holidays and the "
                    "busdaycal parameters to busday_offset()");
            goto fail;
        }

        /* Indicate that the holidays weren't allocated by us */
        allocated_holidays = 0;

        /*
         * Copy the holidays data from busdaycal to local holidays,
         * and also copy the weekmask.
         */
        holidays = busdaycal->holidays;
        busdays_in_weekmask = busdaycal->busdays_in_weekmask;
        memcpy(weekmask, busdaycal->weekmask, 7);
    }
    else {
        /*
         * If weekmask[0] is 2, fix it to 1, indicating a valid weekmask.
         */
        if (weekmask[0] == 2) {
            weekmask[0] = 1;
        }

        /* Count the number of business days in a week from weekmask */
        busdays_in_weekmask = 0;
        for (i = 0; i < 7; ++i) {
            busdays_in_weekmask += weekmask[i];
        }

        /*
         * Normalize the holidays list according to the weekmask.
         * This ensures holidays are correctly represented for calculations.
         */
        normalize_holidays_list(&holidays, weekmask);
    }

    /* Make 'dates' into an array if it's not already */
    if (PyArray_Check(dates_in)) {
        dates = (PyArrayObject *)dates_in;
        Py_INCREF(dates);
    }
    else {
        PyArray_Descr *datetime_dtype;

        /* 使用 datetime 类型的描述符，通用单位填充 */
        datetime_dtype = PyArray_DescrFromType(NPY_DATETIME);
        if (datetime_dtype == NULL) {
            goto fail;
        }

        /* 这里获取 datetime_dtype 的引用 */
        dates = (PyArrayObject *)PyArray_FromAny(dates_in, datetime_dtype,
                                                0, 0, 0, NULL);
        if (dates == NULL) {
            goto fail;
        }
    }

    /* 将 'offsets' 转换为数组 */
    offsets = (PyArrayObject *)PyArray_FromAny(offsets_in,
                            PyArray_DescrFromType(NPY_INT64),
                            0, 0, 0, NULL);
    if (offsets == NULL) {
        goto fail;
    }

    /* 如果提供了 'out'，确保它是一个数组 */
    if (out_in != NULL) {
        if (!PyArray_Check(out_in)) {
            PyErr_SetString(PyExc_ValueError,
                    "busday_offset: must provide a NumPy array for 'out'");
            goto fail;
        }
        out = (PyArrayObject *)out_in;
    }

    /* 调用 business_day_offset 函数计算偏移量 */
    ret = business_day_offset(dates, offsets, out, roll,
                    weekmask, busdays_in_weekmask,
                    holidays.begin, holidays.end);

    Py_DECREF(dates);
    Py_DECREF(offsets);
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    /* 如果 out 为空，则返回 ret，否则返回 out */
    return out == NULL ? PyArray_Return(ret) : (PyObject *)ret;
fail:
    # 释放 'dates' 引用计数
    Py_XDECREF(dates);
    # 释放 'offsets' 引用计数
    Py_XDECREF(offsets);
    # 如果分配了节假日并且节假日开始日期不为空，则释放节假日内存
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    # 返回 NULL 表示函数执行失败
    return NULL;
}

/*
 * This is the 'busday_count' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_count(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"begindates", "enddates",
                             "weekmask", "holidays", "busdaycal", "out", NULL};

    PyObject *dates_begin_in = NULL, *dates_end_in = NULL, *out_in = NULL;

    PyArrayObject *dates_begin = NULL, *dates_end = NULL, *out = NULL, *ret;
    npy_bool weekmask[7] = {2, 1, 1, 1, 1, 0, 0};
    NpyBusDayCalendar *busdaycal = NULL;
    int i, busdays_in_weekmask;
    npy_holidayslist holidays = {NULL, NULL};
    int allocated_holidays = 1;

    # 解析 Python 函数参数并检查正确性
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                    "OO|O&O&O!O:busday_count", kwlist,
                                    &dates_begin_in,
                                    &dates_end_in,
                                    &PyArray_WeekMaskConverter, &weekmask[0],
                                    &PyArray_HolidaysConverter, &holidays,
                                    &NpyBusDayCalendar_Type, &busdaycal,
                                    &out_in)) {
        # 处理解析失败，跳转到 fail 标签处
        goto fail;
    }

    /* Make sure only one of the weekmask/holidays and busdaycal is supplied */
    # 确保只提供了 weekmask/holidays 或者 busdaycal 中的一个
    if (busdaycal != NULL) {
        if (weekmask[0] != 2 || holidays.begin != NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Cannot supply both the weekmask/holidays and the "
                    "busdaycal parameters to busday_count()");
            # 处理错误情况，跳转到 fail 标签处
            goto fail;
        }

        /* Indicate that the holidays weren't allocated by us */
        # 表明节假日列表不是由我们分配的
        allocated_holidays = 0;

        /* Copy the private normalized weekmask/holidays data */
        # 复制私有的标准化 weekmask/holidays 数据
        holidays = busdaycal->holidays;
        busdays_in_weekmask = busdaycal->busdays_in_weekmask;
        memcpy(weekmask, busdaycal->weekmask, 7);
    }
    else {
        /*
         * Fix up the weekmask from the uninitialized
         * signal value to a proper default.
         */
        # 修正未初始化的 weekmask 值为适当的默认值
        if (weekmask[0] == 2) {
            weekmask[0] = 1;
        }

        /* Count the number of business days in a week */
        # 计算一周中的工作日数量
        busdays_in_weekmask = 0;
        for (i = 0; i < 7; ++i) {
            busdays_in_weekmask += weekmask[i];
        }

        /* The holidays list must be normalized before using it */
        # 在使用节假日列表之前必须对其进行标准化处理
        normalize_holidays_list(&holidays, weekmask);
    }

    /* Make 'dates_begin' into an array */
    # 将 'dates_begin_in' 转换为数组对象
    if (PyArray_Check(dates_begin_in)) {
        dates_begin = (PyArrayObject *)dates_begin_in;
        Py_INCREF(dates_begin);
    }
    else {
        PyArray_Descr *datetime_dtype;

        /* 使用 datetime 类型来填充通用单位，以便初始化 datetime_dtype */
        datetime_dtype = PyArray_DescrFromType(NPY_DATETIME);
        if (datetime_dtype == NULL) {
            goto fail;
        }

        /* 这里会接管 datetime_dtype 的引用 */
        dates_begin = (PyArrayObject *)PyArray_FromAny(dates_begin_in,
                                                datetime_dtype,
                                                0, 0, 0, NULL);
        if (dates_begin == NULL) {
            goto fail;
        }
    }

    /* 将 'dates_end_in' 转换为数组 */
    if (PyArray_Check(dates_end_in)) {
        dates_end = (PyArrayObject *)dates_end_in;
        Py_INCREF(dates_end);
    }
    else {
        PyArray_Descr *datetime_dtype;

        /* 使用 datetime 类型和通用单位来初始化 datetime_dtype */
        datetime_dtype = PyArray_DescrFromType(NPY_DATETIME);
        if (datetime_dtype == NULL) {
            goto fail;
        }

        /* 这里会接管 datetime_dtype 的引用 */
        dates_end = (PyArrayObject *)PyArray_FromAny(dates_end_in,
                                                datetime_dtype,
                                                0, 0, 0, NULL);
        if (dates_end == NULL) {
            goto fail;
        }
    }

    /* 如果提供了 'out_in'，确保它是一个数组 */
    if (out_in != NULL) {
        if (!PyArray_Check(out_in)) {
            PyErr_SetString(PyExc_ValueError,
                    "busday_offset: must provide a NumPy array for 'out'");
            goto fail;
        }
        out = (PyArrayObject *)out_in;
    }

    /* 调用 business_day_count 函数计算工作日数量 */
    ret = business_day_count(dates_begin, dates_end, out,
                    weekmask, busdays_in_weekmask,
                    holidays.begin, holidays.end);

    /* 减少 'dates_begin' 和 'dates_end' 的引用计数 */
    Py_DECREF(dates_begin);
    Py_DECREF(dates_end);

    /* 如果 holidays.begin 被分配了且不为 NULL，则释放其内存 */
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    /* 如果 'out' 为 NULL，则返回 'ret' 的 NumPy 表示形式，否则返回 'ret' 的 PyObject 表示形式 */
    return out == NULL ? PyArray_Return(ret) : (PyObject *)ret;
fail:
    // 释放 dates_begin 指向的 Python 对象的引用计数
    Py_XDECREF(dates_begin);
    // 释放 dates_end 指向的 Python 对象的引用计数
    Py_XDECREF(dates_end);
    // 如果已分配 holidays 且 holidays.begin 不为 NULL，则释放其内存
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    // 返回空指针，表示函数执行失败
    return NULL;
}

/*
 * This is the 'is_busday' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_is_busday(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds)
{
    // 定义关键字列表
    static char *kwlist[] = {"dates",
                             "weekmask", "holidays", "busdaycal", "out", NULL};

    PyObject *dates_in = NULL, *out_in = NULL;

    // 定义变量
    PyArrayObject *dates = NULL,*out = NULL, *ret;
    npy_bool weekmask[7] = {2, 1, 1, 1, 1, 0, 0};
    NpyBusDayCalendar *busdaycal = NULL;
    int i, busdays_in_weekmask;
    npy_holidayslist holidays = {NULL, NULL};
    int allocated_holidays = 1;

    // 解析传入的参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                    "O|O&O&O!O:is_busday", kwlist,
                                    &dates_in,
                                    &PyArray_WeekMaskConverter, &weekmask[0],
                                    &PyArray_HolidaysConverter, &holidays,
                                    &NpyBusDayCalendar_Type, &busdaycal,
                                    &out_in)) {
        // 如果解析失败，跳转到失败处理标签
        goto fail;
    }

    /* Make sure only one of the weekmask/holidays and busdaycal is supplied */
    // 确保只有 weekmask/holidays 或 busdaycal 中的一个被提供
    if (busdaycal != NULL) {
        // 如果同时提供了 weekmask/holidays 和 busdaycal 参数，则报错并跳转到失败处理标签
        if (weekmask[0] != 2 || holidays.begin != NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Cannot supply both the weekmask/holidays and the "
                    "busdaycal parameters to is_busday()");
            goto fail;
        }

        /* Indicate that the holidays weren't allocated by us */
        // 表示 holidays 不是由我们分配的
        allocated_holidays = 0;

        /* Copy the private normalized weekmask/holidays data */
        // 复制私有的标准化 weekmask/holidays 数据
        holidays = busdaycal->holidays;
        busdays_in_weekmask = busdaycal->busdays_in_weekmask;
        memcpy(weekmask, busdaycal->weekmask, 7);
    }
    else {
        /*
         * Fix up the weekmask from the uninitialized
         * signal value to a proper default.
         */
        // 将未初始化的 weekmask 信号值修复为合适的默认值
        if (weekmask[0] == 2) {
            weekmask[0] = 1;
        }

        /* Count the number of business days in a week */
        // 计算一周中的工作日数量
        busdays_in_weekmask = 0;
        for (i = 0; i < 7; ++i) {
            busdays_in_weekmask += weekmask[i];
        }

        /* The holidays list must be normalized before using it */
        // 使用前需标准化 holidays 列表
        normalize_holidays_list(&holidays, weekmask);
    }

    /* Make 'dates' into an array */
    // 将 'dates' 转换为数组
    if (PyArray_Check(dates_in)) {
        dates = (PyArrayObject *)dates_in;
        Py_INCREF(dates);
    }
    else {
        PyArray_Descr *datetime_dtype;

        /* 使用 datetime dtype 并且使用通用单位，以便它填充 */
        datetime_dtype = PyArray_DescrFromType(NPY_DATETIME);
        // 检查是否成功创建 datetime_dtype 对象
        if (datetime_dtype == NULL) {
            // 如果创建失败，跳转到错误处理标签
            goto fail;
        }

        /* 这里会窃取 datetime_dtype 的引用 */
        // 将输入的日期数组 dates_in 转换为 PyArrayObject 类型，使用 datetime_dtype 描述符
        dates = (PyArrayObject *)PyArray_FromAny(dates_in,
                                                datetime_dtype,
                                                0, 0, 0, NULL);
        // 检查是否成功创建 dates 对象
        if (dates == NULL) {
            // 如果创建失败，跳转到错误处理标签
            goto fail;
        }
    }

    /* 如果提供了 'out'，确保它是一个数组 */
    if (out_in != NULL) {
        // 检查 out_in 是否为 NumPy 数组
        if (!PyArray_Check(out_in)) {
            // 如果不是数组，设置错误信息并跳转到错误处理标签
            PyErr_SetString(PyExc_ValueError,
                    "busday_offset: must provide a NumPy array for 'out'");
            goto fail;
        }
        // 将 out_in 强制转换为 PyArrayObject 类型，赋值给 out
        out = (PyArrayObject *)out_in;
    }

    // 调用 is_business_day 函数计算工作日
    ret = is_business_day(dates, out,
                    weekmask, busdays_in_weekmask,
                    holidays.begin, holidays.end);

    // 减少 dates 的引用计数
    Py_DECREF(dates);
    // 如果 allocated_holidays 为真且 holidays.begin 不为空，则释放 holidays.begin 内存
    if (allocated_holidays && holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    // 如果 out 为空，则返回 ret 的 NumPy 对象；否则返回 ret 的 PyObject 对象
    return out == NULL ? PyArray_Return(ret) : (PyObject *)ret;
fail:
    // 释放 Python 对象 dates 的引用
    Py_XDECREF(dates);
    // 检查是否分配了节假日数组并且数组的起始地址不为空
    if (allocated_holidays && holidays.begin != NULL) {
        // 释放节假日数组的内存
        PyArray_free(holidays.begin);
    }

    // 返回空指针，表示函数执行失败
    return NULL;
}
```