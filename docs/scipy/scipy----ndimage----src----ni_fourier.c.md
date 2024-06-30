# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_fourier.c`

```
/*
 * 该部分是版权声明和许可协议的文本，授权使用、复制和分发的条件和限制。
 * 版权归 Peter J. Verveer 所有。
 */

#include "ni_support.h"
#include "ni_fourier.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <numpy/npy_math.h>
#include "npy_2_complexcompat.h"

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

#define _NI_GAUSSIAN 0
#define _NI_UNIFORM 1
#define _NI_ELLIPSOID 2

/*
 * 函数：polevl
 * 参数：
 *   - x: 输入的 double 类型数值
 *   - coef: double 数组，系数列表
 *   - N: 整数，系数列表长度
 * 返回值：double 类型，计算的多项式求值结果
 * 功能：使用 Horner 方法计算多项式的值
 */
static double polevl(double x, const double coef[], int N)
{
    double ans;
    const double *p = coef;
    int i = N;

    ans = *p++;
    do
        ans = ans * x + *p++;
    while(--i);

    return ans ;
}

/*
 * 函数：p1evl
 * 参数：
 *   - x: 输入的 double 类型数值
 *   - coef: double 数组，系数列表
 *   - N: 整数，系数列表长度
 * 返回值：double 类型，计算的多项式求值结果
 * 功能：使用 Horner 方法计算多项式的值，与 polevl 的差别在于初始值和计算次数
 */
double p1evl(double x, const double coef[], int N)
{
    double ans;
    const double *p = coef;
    int i = N - 1;

    ans = x + *p++;
    do
        ans = ans * x + *p++;
    while(--i);

    return ans;
}

/*
 * 常量定义
 */
#define THPIO4 2.35619449019234492885
#define SQ2OPI .79788456080286535588
#define Z1 1.46819706421238932572E1
#define Z2 4.92184563216946036703E1

/*
 * 函数：_bessel_j1
 * 参数：
 *   - x: 输入的 double 类型数值
 * 返回值：double 类型，贝塞尔函数 J1(x) 的近似值
 * 功能：计算贝塞尔函数 J1(x) 的近似值
 */
static double _bessel_j1(double x)
{
    double w, z, p, q, xn;
    const double RP[4] = {
        -8.99971225705559398224E8,
        4.52228297998194034323E11,
        -7.27494245221818276015E13,
        3.68295732863852883286E15,
    };
    const double RQ[8] = {
        6.20836478118054335476E2,
        2.56987256757748830383E5,
        8.35146791431949253037E7,
        2.21511595479792499675E10,
        4.74914122079991414898E12,
        7.84369607876235854894E14,
        8.95222336184627338078E16,
        5.32278620332680085395E18,
    };
    
    // 这里进行具体的贝塞尔函数近似计算，使用了预定义的系数数组 RP 和 RQ

        w = x * x;
        z = 8.0 * x;
        xn = x * polevl(w, RP, 3) / p1evl(w, RQ, 8);
        xn = xn - z;
        p = w - Z1;
        p = p - Z2;
        q = w + Z1;
        q = q + Z2;
        xn = xn * (w - THPIO4) * (w - M_PI) / w;
        z = 1.0 - SQ2OPI * xn;
        return z;
}


注释：
    // 系数数组 PP，用于近似计算函数的多项式分子
    const double PP[7] = {
        7.62125616208173112003E-4,
        7.31397056940917570436E-2,
        1.12719608129684925192E0,
        5.11207951146807644818E0,
        8.42404590141772420927E0,
        5.21451598682361504063E0,
        1.00000000000000000254E0,
    };
    
    // 系数数组 PQ，用于近似计算函数的多项式分母
    const double PQ[7] = {
        5.71323128072548699714E-4,
        6.88455908754495404082E-2,
        1.10514232634061696926E0,
        5.07386386128601488557E0,
        8.39985554327604159757E0,
        5.20982848682361821619E0,
        9.99999999999999997461E-1,
    };
    
    // 系数数组 QP，用于近似计算函数的第二个多项式分子
    const double QP[8] = {
        5.10862594750176621635E-2,
        4.98213872951233449420E0,
        7.58238284132545283818E1,
        3.66779609360150777800E2,
        7.10856304998926107277E2,
        5.97489612400613639965E2,
        2.11688757100572135698E2,
        2.52070205858023719784E1,
    };
    
    // 系数数组 QQ，用于近似计算函数的第二个多项式分母
    const double QQ[7] = {
        7.42373277035675149943E1,
        1.05644886038262816351E3,
        4.98641058337653607651E3,
        9.56231892404756170795E3,
        7.99704160447350683650E3,
        2.82619278517639096600E3,
        3.36093607810698293419E2,
    };

    // 变量 w 初始化为 x
    w = x;
    
    // 如果 x 小于 0，将 w 设置为 -x
    if (x < 0)
        w = -x;

    // 如果 w 小于等于 5.0，使用多项式近似计算特定范围内的函数值
    if (w <= 5.0) {
        // 计算 z = x^2
        z = x * x;
        // 使用 polevl 函数计算 RP 多项式的值，再除以 p1evl 函数计算 RQ 多项式的值
        w = polevl(z, RP, 3) / p1evl(z, RQ, 8);
        // 计算函数值并返回
        w = w * x * (z - Z1) * (z - Z2);
        return w;
    }

    // 如果 w 大于 5.0，使用另一组多项式近似计算函数值
    w = 5.0 / x;
    z = w * w;
    // 使用 polevl 函数计算 PP 多项式的值，再除以 polevl 函数计算 PQ 多项式的值
    p = polevl(z, PP, 6) / polevl(z, PQ, 6);
    // 使用 polevl 函数计算 QP 多项式的值，再除以 p1evl 函数计算 QQ 多项式的值
    q = polevl(z, QP, 7) / p1evl(z, QQ, 7);
    // 计算函数值中的余弦和正弦项
    xn = x - THPIO4;
    p = p * cos(xn) - w * q * sin(xn);
    // 返回最终计算结果
    return p * SQ2OPI / sqrt(x);
# 定义宏，用于复傅里叶输出的实数到实数赋值
#define CASE_FOURIER_OUT_RR(_TYPE, _type, _po, _tmp) \
case _TYPE:                                          \
    *(_type *)_po = _tmp;                            \
    break

# 定义宏，用于复傅里叶输出的实数到复数赋值
#define CASE_FOURIER_OUT_RC(_TYPE, _type, _T, _po, _tmp) \
case _TYPE:                                          \
    NPY_CSETREAL##_T((_type *)_po, tmp);                      \
    NPY_CSETIMAG##_T((_type *)_po, 0.0);                      \
    break

# 定义宏，用于复傅里叶输出的复数到复数赋值
#define CASE_FOURIER_OUT_CC(_TYPE, _type, _T, _po, _tmp_r, _tmp_i) \
case _TYPE:                                                    \
    NPY_CSETREAL##_T((_type *)_po, _tmp_r);                             \
    NPY_CSETIMAG##_T((_type *)_po, _tmp_i);                             \
    break

# 定义宏，用于复傅里叶滤波的实数到复数赋值
#define CASE_FOURIER_FILTER_RC(_TYPE, _type, _t, _pi, _tmp, _tmp_r, _tmp_i) \
case _TYPE:                                                             \
    _tmp_r = npy_creal##_t(*((_type *)_pi)) * _tmp;                               \
    _tmp_i = npy_cimag##_t(*((_type *)_pi)) * _tmp;                               \
    break

# 定义宏，用于复傅里叶滤波的实数到实数赋值
#define CASE_FOURIER_FILTER_RR(_TYPE, _type, _pi, _tmp) \
case _TYPE:                                             \
    _tmp *= *(_type *)_pi;                              \
    break

# 定义傅里叶滤波函数
int NI_FourierFilter(PyArrayObject *input, PyArrayObject* parameter_array,
                     npy_intp n, int axis, PyArrayObject* output,
                     int filter_type)
{
    NI_Iterator ii, io;
    char *pi, *po;
    double *parameters = NULL, **params = NULL;
    npy_intp kk, hh, size;
    npy_double *iparameters = (void *)PyArray_DATA(parameter_array);
    NPY_BEGIN_THREADS_DEF;

    /* 预先计算参数：*/
    parameters = malloc(PyArray_NDIM(input) * sizeof(double));
    if (!parameters) {
        PyErr_NoMemory();
        goto exit;
    }
    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
        /* 在实数变换的方向上，必须使用给定的该维度长度，除非假定复数变换 (n < 0)： */
        int shape = kk == axis ?
                (n < 0 ? PyArray_DIM(input, kk) : n) : PyArray_DIM(input, kk);
        switch (filter_type) {
            case _NI_GAUSSIAN:
                parameters[kk] = *iparameters++ * M_PI / (double)shape;
                parameters[kk] = -2.0 * parameters[kk] * parameters[kk];
                break;
            case _NI_ELLIPSOID:
            case _NI_UNIFORM:
                parameters[kk] = *iparameters++;
                break;
        }
    }
    /* 为参数表分配内存：*/
    params = malloc(PyArray_NDIM(input) * sizeof(double*));
    if (!params) {
        PyErr_NoMemory();
        goto exit;
    }
    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
        params[kk] = NULL;
    }
    // 遍历输入数组的每个维度
    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
        // 如果当前维度的大小大于1或者滤波器类型为椭球体
        if (PyArray_DIM(input, kk) > 1 || filter_type == _NI_ELLIPSOID) {
            // 分配当前维度大小的双精度浮点数内存空间给params[kk]
            params[kk] = malloc(PyArray_DIM(input, kk) * sizeof(double));
            // 如果内存分配失败
            if (!params[kk]) {
                // 设置内存错误异常
                PyErr_NoMemory();
                // 跳转到exit标签，执行清理工作
                goto exit;
            }
        }
    }

    // 启动多线程环境
    NPY_BEGIN_THREADS;

    // 初始化输入数组元素迭代器
    if (!NI_InitPointIterator(input, &ii))
        // 如果初始化失败，跳转到exit标签，执行清理工作
        goto exit;
    // 初始化输出数组元素迭代器
    if (!NI_InitPointIterator(output, &io))
        // 如果初始化失败，跳转到exit标签，执行清理工作
        goto exit;
    // 获取输入数组的数据指针
    pi = (void *)PyArray_DATA(input);
    // 获取输出数组的数据指针
    po = (void *)PyArray_DATA(output);
    // 获取输入数组的总大小
    size = PyArray_SIZE(input);
    // 迭代处理每个元素
    // 注意：缺少循环起始的大括号可能是代码中的错误或者格式化问题

exit:
    // 结束多线程环境
    NPY_END_THREADS;
    // 释放参数数组的内存
    free(parameters);
    // 如果params数组不为空
    if (params) {
        // 释放params数组中每个维度的内存
        for (kk = 0; kk < PyArray_NDIM(input); kk++) {
            free(params[kk]);
        }
        // 释放params数组本身的内存
        free(params);
    }
    // 如果发生了异常，返回0；否则返回1
    return PyErr_Occurred() ? 0 : 1;
}

#define CASE_FOURIER_SHIFT_R(_TYPE, _type, _pi, _r, _i, _cost, _sint) \
case _TYPE:                                                          \
    _r = *(_type *)_pi * _cost;                                       \
    _i = *(_type *)_pi * _sint;                                       \
    break

#define CASE_FOURIER_SHIFT_C(_TYPE, _type, _t, _pi, _r, _i, _cost, _sint) \
case _TYPE:                                                           \
    _r = npy_creal##_t(*((_type *)_pi)) * _cost - npy_cimag##_t(*((_type *)_pi)) * _sint; \
    _i = npy_creal##_t(*((_type *)_pi)) * _sint + npy_cimag##_t(*((_type *)_pi)) * _cost; \
    break

int NI_FourierShift(PyArrayObject *input, PyArrayObject* shift_array,
            npy_intp n, int axis, PyArrayObject* output)
{
    NI_Iterator ii, io;
    char *pi, *po;
    double *shifts = NULL, **params = NULL;
    npy_intp kk, hh, size;
    npy_double *ishifts = (void *)PyArray_DATA(shift_array);
    NPY_BEGIN_THREADS_DEF;

    /* precalculate the shifts: */
    shifts = malloc(PyArray_NDIM(input) * sizeof(double));
    if (!shifts) {
        PyErr_NoMemory();
        goto exit;
    }
    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
        /* along the direction of the real transform we must use the given
             length of that dimensions, unless a complex transform is assumed
             (n < 0): */
        int shape = kk == axis ?
                (n < 0 ? PyArray_DIM(input, kk) : n) : PyArray_DIM(input, kk);
        shifts[kk] = -2.0 * M_PI * *ishifts++ / (double)shape;
    }
    /* allocate memory for tables: */
    params = malloc(PyArray_NDIM(input) * sizeof(double*));
    if (!params) {
        PyErr_NoMemory();
        goto exit;
    }
    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
        params[kk] = NULL;
    }
    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
        if (PyArray_DIM(input, kk) > 1) {
            params[kk] = malloc(PyArray_DIM(input, kk) * sizeof(double));
            if (!params[kk]) {
                PyErr_NoMemory();
                goto exit;
            }
        }
    }

    NPY_BEGIN_THREADS;

    for (hh = 0; hh < PyArray_NDIM(input); hh++) {
        if (params[hh]) {
            if (hh == axis && n >= 0) {
                for (kk = 0; kk < PyArray_DIM(input, hh); kk++) {
                    params[hh][kk] = shifts[hh] * kk;
                }
            }
            else {
                int jj = 0;
                for (kk = 0; kk < (PyArray_DIM(input, hh) + 1) / 2; kk++) {
                    params[hh][jj++] = shifts[hh] * kk;
                }
                for (kk = -(PyArray_DIM(input, hh) / 2); kk < 0; kk++) {
                    params[hh][jj++] = shifts[hh] * kk;
                }
            }
        }
    }
    /* initialize input element iterator: */
    if (!NI_InitPointIterator(input, &ii))
        goto exit;
    /* 初始化输出元素迭代器： */
    // 如果无法初始化输出点的迭代器，跳转到 exit 标签
    if (!NI_InitPointIterator(output, &io))
        goto exit;
    // 获取输入数组的数据指针
    pi = (void *)PyArray_DATA(input);
    // 获取输出数组的数据指针
    po = (void *)PyArray_DATA(output);
    // 获取输入数组的元素总数
    size = PyArray_SIZE(input);
    /* 遍历元素： */
    for(hh = 0; hh < size; hh++) {
        // 初始化临时变量
        double tmp = 0.0, sint, cost, r = 0.0, i = 0.0;
        // 遍历输入数组的维度
        for (kk = 0; kk < PyArray_NDIM(input); kk++) {
            // 如果参数数组存在，则累加对应位置的值到 tmp
            if (params[kk])
                tmp += params[kk][ii.coordinates[kk]];
        }
        // 计算 tmp 的正弦值和余弦值
        sint = sin(tmp);
        cost = cos(tmp);
        // 根据输入数组的数据类型执行不同的操作
        switch (PyArray_TYPE(input)) {
            // 定义 Fourier 变换的实部操作宏
            CASE_FOURIER_SHIFT_R(NPY_BOOL, npy_bool,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_UBYTE, npy_ubyte,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_USHORT, npy_ushort,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_UINT, npy_uint,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_ULONG, npy_ulong,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_ULONGLONG, npy_ulonglong,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_BYTE, npy_byte,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_SHORT, npy_short,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_INT, npy_int,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_LONG, npy_long,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_LONGLONG, npy_longlong,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_FLOAT, npy_float,
                                 pi, tmp, r, i, cost, sint);
            CASE_FOURIER_SHIFT_R(NPY_DOUBLE, npy_double,
                                 pi, tmp, r, i, cost, sint);
            // 定义 Fourier 变换的复数实部操作宏
            CASE_FOURIER_SHIFT_C(NPY_CFLOAT, npy_cfloat, f,
                                 pi, r, i, cost, sint);
            CASE_FOURIER_SHIFT_C(NPY_CDOUBLE, npy_cdouble,,
                                 pi, r, i, cost, sint);
        default:
            // 若数据类型不支持，结束线程，并设置运行时错误信息，跳转到 exit 标签
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
        }
        // 根据输出数组的数据类型执行不同的操作
        switch (PyArray_TYPE(output)) {
            // 定义 Fourier 变换的复数输出操作宏
            CASE_FOURIER_OUT_CC(NPY_CFLOAT, npy_cfloat, F, po, r, i);
            CASE_FOURIER_OUT_CC(NPY_CDOUBLE, npy_cdouble,, po, r, i);
        default:
            // 若数据类型不支持，结束线程，并设置运行时错误信息，跳转到 exit 标签
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
        }
        // 迭代器指向下一个元素
        NI_ITERATOR_NEXT2(ii, io, pi, po);
    }

 exit:
    // 结束线程处理
    NPY_END_THREADS;
    // 释放 shifts 数组内存
    free(shifts);
    // 检查 params 是否非空，即是否有分配内存
    if (params) {
        // 遍历 params 数组中的每个元素，释放其指向的内存空间
        for (kk = 0; kk < PyArray_NDIM(input); kk++) {
            free(params[kk]);
        }
        // 释放 params 数组本身所占用的内存空间
        free(params);
    }
    // 如果发生了异常（PyErr_Occurred() 返回非零值），返回 0；否则返回 1
    return PyErr_Occurred() ? 0 : 1;
}


注释：


# 这行代码表示代码块的结束，关闭了一个函数、循环、条件语句或类定义等。在这里可能表示函数的结束。
```