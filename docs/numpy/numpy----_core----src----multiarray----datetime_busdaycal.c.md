# `.\numpy\numpy\_core\src\multiarray\datetime_busdaycal.c`

```
/*
 * This file implements an object encapsulating a business day
 * calendar object for accelerating NumPy datetime business day functions.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

// 设置以 NPY_API_VERSION 版本为准，禁用已弃用的 NumPy API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义 _MULTIARRAYMODULE，可能用于多维数组模块的标志

// 清除 PY_SSIZE_T_CLEAN 宏，确保 Py_ssize_t 被正确定义
#define PY_SSIZE_T_CLEAN

// 引入 Python 标准库头文件
#include <Python.h>

// 引入 NumPy 数组对象的头文件
#include "numpy/arrayobject.h"

// 引入 NumPy 数组标量的头文件
#include "numpy/arrayscalars.h"

// 引入 NumPy 配置文件
#include "npy_config.h"

// 引入共享的常用功能头文件
#include "common.h"

// 引入低级分块循环的头文件
#include "lowlevel_strided_loops.h"

// 引入日期时间扩展的头文件
#include "_datetime.h"

// 引入工作日计算相关的头文件
#include "datetime_busday.h"

// 引入工作日历算法的头文件
#include "datetime_busdaycal.h"

// 非导出的函数，用于转换周掩码对象为布尔数组
NPY_NO_EXPORT int
PyArray_WeekMaskConverter(PyObject *weekmask_in, npy_bool *weekmask)
{
    // weekmask_in 赋给局部变量 obj
    PyObject *obj = weekmask_in;

    // 如果 obj 是字节对象
    if (PyBytes_Check(obj)) {
        /* accept bytes input */
        // 将字节对象转换为 UTF-8 字符串
        PyObject *obj_str = PyUnicode_FromEncodedObject(obj, NULL, NULL);
        // 如果转换失败，返回 0
        if (obj_str == NULL) {
            return 0;
        }
        // 更新 obj 为转换后的字符串对象
        obj = obj_str;
    }
    else {
        // 增加 obj 的引用计数
        Py_INCREF(obj);
    }

    // 如果 obj 是 Unicode 对象
    if (PyUnicode_Check(obj)) {
        // 获取 Unicode 对象的长度和 UTF-8 编码字符串
        Py_ssize_t len;
        char const *str = PyUnicode_AsUTF8AndSize(obj, &len);
        // 如果获取失败，释放 obj 并返回 0
        if (str == NULL) {
            Py_DECREF(obj);
            return 0;
        }

        // 如果字符串长度为 7，处理类似 "1111100" 的字符串
        if (len == 7) {
            // 遍历字符串的每个字符
            for (int i = 0; i < 7; ++i) {
                // 根据字符设置对应的 weekmask 值
                switch(str[i]) {
                    case '0':
                        weekmask[i] = 0;
                        break;
                    case '1':
                        weekmask[i] = 1;
                        break;
                    default:
                        // 如果遇到非预期字符，跳转到 general_weekmask_string 处理
                        goto general_weekmask_string;
                }
            }

            // 跳转到完成处理的标签
            goto finish;
        }

        // 如果长度不为 7，跳转到处理通用 weekmask 字符串的标签
        general_weekmask_string:
        ;
    }

    // 完成处理，释放 obj 对象
    finish:
    Py_DECREF(obj);
    return 1;
}
general_weekmask_string:
    /* 一般的工作日掩码字符串，例如 "SatSun" 或 "Mon Tue Wed" */
    memset(weekmask, 0, 7);  // 将 weekmask 数组初始化为零，长度为 7
    for (Py_ssize_t i = 0; i < len; i += 3) {  // 循环处理字符串，每次增加 3 个字符长度
        while (isspace(str[i]))  // 跳过空白字符
            ++i;

        if (i == len) {  // 如果已经处理完字符串，跳转到结束标签
            goto finish;
        }
        else if (i + 2 >= len) {  // 如果剩余字符不足 3 个，说明字符串无效，跳转到无效字符串标签
            goto invalid_weekmask_string;
        }

        switch (str[i]) {  // 根据当前字符判断工作日掩码
            case 'M':  // 处理 "Mon" 开头的情况
                if (str[i+1] == 'o' && str[i+2] == 'n') {
                    weekmask[0] = 1;  // 星期一设为工作日
                }
                else {
                    goto invalid_weekmask_string;  // 字符串无效，跳转到无效字符串标签
                }
                break;
            case 'T':  // 处理 "Tue" 和 "Thu" 开头的情况
                if (str[i+1] == 'u' && str[i+2] == 'e') {
                    weekmask[1] = 1;  // 星期二设为工作日
                }
                else if (str[i+1] == 'h' && str[i+2] == 'u') {
                    weekmask[3] = 1;  // 星期四设为工作日
                }
                else {
                    goto invalid_weekmask_string;  // 字符串无效，跳转到无效字符串标签
                }
                break;
            case 'W':  // 处理 "Wed" 开头的情况
                if (str[i+1] == 'e' && str[i+2] == 'd') {
                    weekmask[2] = 1;  // 星期三设为工作日
                }
                else {
                    goto invalid_weekmask_string;  // 字符串无效，跳转到无效字符串标签
                }
                break;
            case 'F':  // 处理 "Fri" 开头的情况
                if (str[i+1] == 'r' && str[i+2] == 'i') {
                    weekmask[4] = 1;  // 星期五设为工作日
                }
                else {
                    goto invalid_weekmask_string;  // 字符串无效，跳转到无效字符串标签
                }
                break;
            case 'S':  // 处理 "Sat" 和 "Sun" 开头的情况
                if (str[i+1] == 'a' && str[i+2] == 't') {
                    weekmask[5] = 1;  // 星期六设为工作日
                }
                else if (str[i+1] == 'u' && str[i+2] == 'n') {
                    weekmask[6] = 1;  // 星期日设为工作日
                }
                else {
                    goto invalid_weekmask_string;  // 字符串无效，跳转到无效字符串标签
                }
                break;
            default:
                goto invalid_weekmask_string;  // 字符串无效，跳转到无效字符串标签
        }
    }

finish:  // 成功处理字符串，跳转到结束
    /* Something like [1,1,1,1,1,0,0] */  // 返回的 weekmask 数组表示的是工作日掩码
    # 如果对象是一个序列对象
    else if (PySequence_Check(obj)) {
        # 检查序列对象的长度是否为7，或者如果是数组对象，则检查其维度是否为1
        if (PySequence_Size(obj) != 7 ||
                        (PyArray_Check(obj) &&
                         PyArray_NDIM((PyArrayObject *)obj) != 1)) {
            # 设置异常信息，表示业务日期的工作日掩码数组必须长度为7
            PyErr_SetString(PyExc_ValueError,
                "A business day weekmask array must have length 7");
            # 减少对象的引用计数
            Py_DECREF(obj);
            # 返回0，表示操作失败
            return 0;
        }
        else {
            # 循环遍历7次，处理每个元素
            int i;

            for (i = 0; i < 7; ++i) {
                long val;
                # 获取序列对象中的第i个元素
                PyObject *f = PySequence_GetItem(obj, i);
                # 如果获取失败，则减少对象的引用计数并返回0
                if (f == NULL) {
                    Py_DECREF(obj);
                    return 0;
                }

                # 将获取的元素转换为长整型
                val = PyLong_AsLong(f);
                # 如果转换过程中出错，则减少元素的引用计数和对象的引用计数，并返回0
                if (error_converting(val)) {
                    Py_DECREF(f);
                    Py_DECREF(obj);
                    return 0;
                }
                # 根据转换后的值设置周掩码数组的对应值
                if (val == 0) {
                    weekmask[i] = 0;
                }
                else if (val == 1) {
                    weekmask[i] = 1;
                }
                else {
                    # 如果值不是0或1，则设置异常信息，表示业务日期的工作日掩码数组必须全部是1或0
                    PyErr_SetString(PyExc_ValueError,
                        "A business day weekmask array must have all "
                        "1's and 0's");
                    # 减少元素的引用计数和对象的引用计数，并返回0
                    Py_DECREF(f);
                    Py_DECREF(obj);
                    return 0;
                }
                # 减少元素的引用计数
                Py_DECREF(f);
            }

            # 跳转到完成处理的标签
            goto finish;
        }
    }

    # 设置异常信息，表示无法将对象转换为业务日期的工作日掩码数组
    PyErr_SetString(PyExc_ValueError,
            "Couldn't convert object into a business day weekmask");
    # 减少对象的引用计数
    Py_DECREF(obj);
    # 返回0，表示操作失败
    return 0;
finish:
    Py_DECREF(obj);
    return 1;
}

static int
qsort_datetime_compare(const void *elem1, const void *elem2)
{
    npy_datetime e1 = *(const npy_datetime *)elem1;
    npy_datetime e2 = *(const npy_datetime *)elem2;

    return (e1 < e2) ? -1 : (e1 == e2) ? 0 : 1;
}

/*
 * Sorts the array of dates provided in place and removes
 * NaT, duplicates and any date which is already excluded on account
 * of the weekmask.
 *
 * Returns the number of dates left after removing weekmask-excluded
 * dates.
 */
NPY_NO_EXPORT void
normalize_holidays_list(npy_holidayslist *holidays, npy_bool *weekmask)
{
    npy_datetime *dates = holidays->begin;
    npy_intp count = holidays->end - dates;

    npy_datetime lastdate = NPY_DATETIME_NAT;
    npy_intp trimcount, i;
    int day_of_week;

    /* Sort the dates */
    qsort(dates, count, sizeof(npy_datetime), &qsort_datetime_compare);

    /* Sweep through the array, eliminating unnecessary values */
    trimcount = 0;
    for (i = 0; i < count; ++i) {
        npy_datetime date = dates[i];

        /* Skip any NaT or duplicate */
        if (date != NPY_DATETIME_NAT && date != lastdate) {
            /* Get the day of the week (1970-01-05 is Monday) */
            day_of_week = (int)((date - 4) % 7);
            if (day_of_week < 0) {
                day_of_week += 7;
            }

            /*
             * If the holiday falls on a possible business day,
             * then keep it.
             */
            if (weekmask[day_of_week] == 1) {
                dates[trimcount++] = date;
                lastdate = date;
            }
        }
    }

    /* Adjust the end of the holidays array */
    holidays->end = dates + trimcount;
}

/*
 * Converts a Python input into a non-normalized list of holidays.
 *
 * IMPORTANT: This function can't do the normalization, because it doesn't
 *            know the weekmask. You must call 'normalize_holiday_list'
 *            on the result before using it.
 */
NPY_NO_EXPORT int
PyArray_HolidaysConverter(PyObject *dates_in, npy_holidayslist *holidays)
{
    PyArrayObject *dates = NULL;
    PyArray_Descr *date_dtype = NULL;
    npy_intp count;

    /* Make 'dates' into an array */
    if (PyArray_Check(dates_in)) {
        // 如果 dates_in 是已经是一个数组，则直接使用它
        dates = (PyArrayObject *)dates_in;
        Py_INCREF(dates);
    }
    else {
        PyArray_Descr *datetime_dtype;

        /* Use the datetime dtype with generic units so it fills it in */
        // 使用通用单位的 datetime 数据类型来填充日期数据
        datetime_dtype = PyArray_DescrFromType(NPY_DATETIME);
        if (datetime_dtype == NULL) {
            goto fail;
        }

        /* This steals the datetime_dtype reference */
        // 从任意输入数据创建一个新的数组
        dates = (PyArrayObject *)PyArray_FromAny(dates_in, datetime_dtype,
                                                0, 0, 0, NULL);
        if (dates == NULL) {
            goto fail;
        }
    }

    // 创建一个带有指定单位的 datetime 数据类型
    date_dtype = create_datetime_dtype_with_unit(NPY_DATETIME, NPY_FR_D);
    if (date_dtype == NULL) {
        goto fail;
    }
    # 检查是否可以安全地将 dates 数组转换为 date_dtype 类型的数组
    if (!PyArray_CanCastTypeTo(PyArray_DESCR(dates),
                                    date_dtype, NPY_SAFE_CASTING)) {
        # 设置异常信息，指示无法安全地转换提供的假日输入为日期数组
        PyErr_SetString(PyExc_ValueError, "Cannot safely convert "
                        "provided holidays input into an array of dates");
        # 转到失败处理部分
        goto fail;
    }
    
    # 检查 dates 数组的维度是否为 1
    if (PyArray_NDIM(dates) != 1) {
        # 设置异常信息，指示假日必须作为一维数组提供
        PyErr_SetString(PyExc_ValueError, "holidays must be a provided "
                        "as a one-dimensional array");
        # 转到失败处理部分
        goto fail;
    }

    /* 分配用于存储日期的内存空间 */
    count = PyArray_DIM(dates, 0);
    holidays->begin = PyArray_malloc(sizeof(npy_datetime) * count);
    if (holidays->begin == NULL) {
        # 分配内存失败，设置内存不足的异常
        PyErr_NoMemory();
        # 转到失败处理部分
        goto fail;
    }
    holidays->end = holidays->begin + count;

    /* 将数据强制转换为原始日期数组 */
    if (PyArray_CastRawArrays(count,
                            PyArray_BYTES(dates), (char *)holidays->begin,
                            PyArray_STRIDE(dates, 0), sizeof(npy_datetime),
                            PyArray_DESCR(dates), date_dtype,
                            0) != NPY_SUCCEED) {
        # 转换失败，转到失败处理部分
        goto fail;
    }

    // 减少 dates 和 date_dtype 的引用计数
    Py_DECREF(dates);
    Py_DECREF(date_dtype);

    // 返回成功标志
    return 1;
fail:
    // 释放 dates 和 date_dtype 对象的引用，防止内存泄漏
    Py_XDECREF(dates);
    Py_XDECREF(date_dtype);
    // 返回 0，表示函数执行失败
    return 0;
}

static PyObject *
busdaycalendar_new(PyTypeObject *subtype,
                    PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    NpyBusDayCalendar *self;

    // 分配内存给新的 NpyBusDayCalendar 对象
    self = (NpyBusDayCalendar *)subtype->tp_alloc(subtype, 0);
    if (self != NULL) {
        /* Start with an empty holidays list */
        // 初始化 holidays 为一个空列表
        self->holidays.begin = NULL;
        self->holidays.end = NULL;

        /* Set the weekmask to the default */
        // 设置 weekmask 的默认值为工作日掩码
        self->busdays_in_weekmask = 5;
        self->weekmask[0] = 1;
        self->weekmask[1] = 1;
        self->weekmask[2] = 1;
        self->weekmask[3] = 1;
        self->weekmask[4] = 1;
        self->weekmask[5] = 0;
        self->weekmask[6] = 0;
    }

    // 返回新创建的对象
    return (PyObject *)self;
}

static int
busdaycalendar_init(NpyBusDayCalendar *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"weekmask", "holidays", NULL};
    int i, busdays_in_weekmask;

    /* Clear the holidays if necessary */
    // 如果 holidays 列表不为空，则清空它
    if (self->holidays.begin != NULL) {
        PyArray_free(self->holidays.begin);
        self->holidays.begin = NULL;
        self->holidays.end = NULL;
    }

    /* Reset the weekmask to the default */
    // 重置 weekmask 为默认值
    self->busdays_in_weekmask = 5;
    self->weekmask[0] = 1;
    self->weekmask[1] = 1;
    self->weekmask[2] = 1;
    self->weekmask[3] = 1;
    self->weekmask[4] = 1;
    self->weekmask[5] = 0;
    self->weekmask[6] = 0;

    /* Parse the parameters */
    // 解析参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                        "|O&O&:busdaycal", kwlist,
                        &PyArray_WeekMaskConverter, &self->weekmask[0],
                        &PyArray_HolidaysConverter, &self->holidays)) {
        return -1;
    }

    /* Count the number of business days in a week */
    // 统计一周中的工作日数量
    busdays_in_weekmask = 0;
    for (i = 0; i < 7; ++i) {
        busdays_in_weekmask += self->weekmask[i];
    }
    self->busdays_in_weekmask = busdays_in_weekmask;

    /* Normalize the holidays list */
    // 标准化 holidays 列表
    normalize_holidays_list(&self->holidays, self->weekmask);

    // 检查 weekmask 是否全为零，若是则抛出 ValueError 异常
    if (self->busdays_in_weekmask == 0) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot construct a numpy.busdaycal with a weekmask of "
                "all zeros");
        return -1;
    }

    // 返回 0，表示初始化成功
    return 0;
}

static void
busdaycalendar_dealloc(NpyBusDayCalendar *self)
{
    /* Clear the holidays */
    // 清空 holidays 列表
    if (self->holidays.begin != NULL) {
        PyArray_free(self->holidays.begin);
        self->holidays.begin = NULL;
        self->holidays.end = NULL;
    }

    // 释放对象的内存
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
busdaycalendar_weekmask_get(NpyBusDayCalendar *self, void *NPY_UNUSED(ignored))
{
    PyArrayObject *ret;
    npy_intp size = 7;

    /* Allocate a 7-element boolean array */
    // 分配一个包含 7 个元素的布尔数组
    ret = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_BOOL);
    if (ret == NULL) {
        return NULL;
    }

    /* Copy the weekmask data */
    // 复制 weekmask 的数据到返回的数组中
    memcpy(PyArray_DATA(ret), self->weekmask, 7);
    // 将 ret 转换为 PyObject 指针类型并返回
    return (PyObject *)ret;
}

static PyObject *
busdaycalendar_holidays_get(NpyBusDayCalendar *self, void *NPY_UNUSED(ignored))
{
    PyArrayObject *ret;
    PyArray_Descr *date_dtype;
    npy_intp size = self->holidays.end - self->holidays.begin;

    /* Create a date dtype */
    // 创建一个日期的数据类型
    date_dtype = create_datetime_dtype_with_unit(NPY_DATETIME, NPY_FR_D);
    if (date_dtype == NULL) {
        return NULL;
    }

    /* Allocate a date array (this steals the date_dtype reference) */
    // 分配一个日期数组（这里会接管 date_dtype 的引用）
    ret = (PyArrayObject *)PyArray_SimpleNewFromDescr(1, &size, date_dtype);
    if (ret == NULL) {
        return NULL;
    }

    /* Copy the holidays */
    // 复制节假日数据
    if (size > 0) {
        memcpy(PyArray_DATA(ret), self->holidays.begin,
                    size * sizeof(npy_datetime));
    }

    return (PyObject *)ret;
}

static PyGetSetDef busdaycalendar_getsets[] = {
    {"weekmask",
        (getter)busdaycalendar_weekmask_get,
        NULL, NULL, NULL},
    {"holidays",
        (getter)busdaycalendar_holidays_get,
        NULL, NULL, NULL},

    {NULL, NULL, NULL, NULL, NULL}
};

NPY_NO_EXPORT PyTypeObject NpyBusDayCalendar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.busdaycalendar",
    .tp_basicsize = sizeof(NpyBusDayCalendar),
    .tp_dealloc = (destructor)busdaycalendar_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = busdaycalendar_getsets,
    .tp_init = (initproc)busdaycalendar_init,
    .tp_new = busdaycalendar_new,
};
```