# `D:\src\scipysrc\scipy\scipy\spatial\src\distance_wrap.c`

```
/**
 * Author: Damian Eads
 * Date:   September 22, 2007 (moved to new file on June 8, 2008)
 * Adapted for incorporation into Scipy, April 9, 2008.
 *
 * Copyright (c) 2007, Damian Eads. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   - Redistributions of source code must retain the above
 *     copyright notice, this list of conditions and the
 *     following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer
 *     in the documentation and/or other materials provided with the
 *     distribution.
 *   - Neither the name of the author nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if !defined(__clang__) && defined(__GNUC__) && defined(__GNUC_MINOR__)
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)
/* enable auto-vectorizer */
#pragma GCC optimize("tree-vectorize")
/* float associativity required to vectorize reductions */
#pragma GCC optimize("unsafe-math-optimizations")
/* maybe 5% gain, manual unrolling with more accumulators would be better */
#pragma GCC optimize("unroll-loops")
#endif
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "distance_impl.h"

/* 定义一个宏，用于生成特定名称的函数包装器 */
#define DEFINE_WRAP_CDIST(name, type)                                   \
    static PyObject *                                                   \
    /* 定义一个名为 cdist_name_type_wrap 的静态函数，返回 PyObject 指针，参数为 self 和 args */ \
    cdist_ ## name ## _ ## type ## _wrap(PyObject *self, PyObject *args)\
    {
        \                                                                   \
            PyArrayObject *XA_, *XB_, *dm_;                                 \  // 声明三个PyArrayObject类型的指针变量XA_, XB_, dm_
            Py_ssize_t mA, mB, n;                                           \  // 声明三个Py_ssize_t类型的变量mA, mB, n
            double *dm;                                                     \  // 声明一个double类型的指针变量dm
            const type *XA, *XB;                                            \  // 声明两个const type类型的指针变量XA, XB
            if (!PyArg_ParseTuple(args, "O!O!O!",                           \  // 使用PyArg_ParseTuple函数解析传入的参数args，期望参数类型分别是PyArray_Type, PyArray_Type, PyArray_Type
                                  &PyArray_Type, &XA_, &PyArray_Type, &XB_, \
                                  &PyArray_Type, &dm_)) {                   \  // 如果解析失败，则返回NULL
                return NULL;                                                \
            }                                                               \
            else {                                                          \  // 如果解析成功，则执行以下代码块
                NPY_BEGIN_ALLOW_THREADS;                                    \  // 开始允许多线程操作
                XA = (const type *)PyArray_DATA(XA_);                       \  // 获取XA_指向的PyArrayObject的数据，并强制转换为const type类型的指针赋值给XA
                XB = (const type *)PyArray_DATA(XB_);                       \  // 获取XB_指向的PyArrayObject的数据，并强制转换为const type类型的指针赋值给XB
                dm = (double *)PyArray_DATA(dm_);                           \  // 获取dm_指向的PyArrayObject的数据，并强制转换为double类型的指针赋值给dm
                mA = PyArray_DIMS(XA_)[0];                                    \  // 获取XA_指向的PyArrayObject的第一个维度大小，赋值给mA
                mB = PyArray_DIMS(XB_)[0];                                    \  // 获取XB_指向的PyArrayObject的第一个维度大小，赋值给mB
                n = PyArray_DIMS(XA_)[1];                                     \  // 获取XA_指向的PyArrayObject的第二个维度大小，赋值给n
                cdist_ ## name ## _ ## type(XA, XB, dm, mA, mB, n);         \  // 调用名为cdist_name_type的函数，传入参数XA, XB, dm, mA, mB, n
                NPY_END_ALLOW_THREADS;                                      \  // 结束允许多线程操作
            }                                                               \
            return Py_BuildValue("d", 0.);                                  \  // 返回一个包含double类型值0的Python对象
        }
DEFINE_WRAP_CDIST(bray_curtis, double)
// 定义一个包装函数，用于计算 Bray-Curtis 距离（double 类型）

DEFINE_WRAP_CDIST(canberra, double)
// 定义一个包装函数，用于计算 Canberra 距离（double 类型）

DEFINE_WRAP_CDIST(chebyshev, double)
// 定义一个包装函数，用于计算 Chebyshev 距离（double 类型）

DEFINE_WRAP_CDIST(city_block, double)
// 定义一个包装函数，用于计算 City Block 距离（double 类型）

DEFINE_WRAP_CDIST(euclidean, double)
// 定义一个包装函数，用于计算 Euclidean 距离（double 类型）

DEFINE_WRAP_CDIST(jaccard, double)
// 定义一个包装函数，用于计算 Jaccard 距离（double 类型）

DEFINE_WRAP_CDIST(jensenshannon, double)
// 定义一个包装函数，用于计算 Jensen-Shannon 距离（double 类型）

DEFINE_WRAP_CDIST(sqeuclidean, double)
// 定义一个包装函数，用于计算平方 Euclidean 距离（double 类型）

DEFINE_WRAP_CDIST(dice, char)
// 定义一个包装函数，用于计算 Dice 距离（char 类型）

DEFINE_WRAP_CDIST(jaccard, char)
// 定义一个包装函数，用于计算 Jaccard 距离（char 类型）

DEFINE_WRAP_CDIST(kulczynski1, char)
// 定义一个包装函数，用于计算 Kulczynski1 距离（char 类型）

DEFINE_WRAP_CDIST(rogerstanimoto, char)
// 定义一个包装函数，用于计算 Rogerstanimoto 距离（char 类型）

DEFINE_WRAP_CDIST(russellrao, char)
// 定义一个包装函数，用于计算 Russell-Rao 距离（char 类型）

DEFINE_WRAP_CDIST(sokalmichener, char)
// 定义一个包装函数，用于计算 Sokal-Michener 距离（char 类型）

DEFINE_WRAP_CDIST(sokalsneath, char)
// 定义一个包装函数，用于计算 Sokal-Sneath 距离（char 类型）

DEFINE_WRAP_CDIST(yule, char)
// 定义一个包装函数，用于计算 Yule 距离（char 类型）

static PyObject *cdist_hamming_double_wrap(
                            PyObject *self, PyObject *args, PyObject *kwargs) 
{
    PyArrayObject *XA_, *XB_, *dm_, *w_;
    int mA, mB, n;
    double *dm;
    const double *XA, *XB, *w;
    static char *kwlist[] = {"XA", "XB", "dm", "w", NULL};
    // 解析传入的参数，包括两个数据数组 XA 和 XB，一个距离矩阵 dm，一个权重数组 w
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!O!:cdist_hamming_double_wrap", kwlist,
            &PyArray_Type, &XA_, &PyArray_Type, &XB_, 
            &PyArray_Type, &dm_,
            &PyArray_Type, &w_)) {
        return 0;
    }
    else {
        NPY_BEGIN_ALLOW_THREADS;
        // 获取数组的数据指针
        XA = (const double*)PyArray_DATA(XA_);
        XB = (const double*)PyArray_DATA(XB_);
        w = (const double*)PyArray_DATA(w_);
        dm = (double*)PyArray_DATA(dm_);
        // 获取数组的维度信息
        mA = PyArray_DIMS(XA_)[0];
        mB = PyArray_DIMS(XB_)[0];
        n = PyArray_DIMS(XA_)[1];
        // 调用 C 函数计算 Hamming 距离
        cdist_hamming_double(XA, XB, dm, mA, mB, n, w);
        NPY_END_ALLOW_THREADS;
    }
    // 返回一个 Python 对象，此处返回固定的浮点数 0.0
    return Py_BuildValue("d", 0.0);
}

static PyObject *cdist_hamming_char_wrap(
                            PyObject *self, PyObject *args, PyObject *kwargs) 
{
    PyArrayObject *XA_, *XB_, *dm_, *w_;
    int mA, mB, n;
    double *dm;
    const char *XA, *XB;
    const double *w;
    static char *kwlist[] = {"XA", "XB", "dm", "w", NULL};
    // 解析传入的参数，包括两个数据数组 XA 和 XB，一个距离矩阵 dm，一个权重数组 w
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!O!:cdist_hamming_char_wrap", kwlist,
            &PyArray_Type, &XA_, &PyArray_Type, &XB_, 
            &PyArray_Type, &dm_,
            &PyArray_Type, &w_)) {
        return 0;
    }
    else {
        NPY_BEGIN_ALLOW_THREADS;
        // 获取数组的数据指针
        XA = (const char*)PyArray_DATA(XA_);
        XB = (const char*)PyArray_DATA(XB_);
        w = (const double*)PyArray_DATA(w_);
        dm = (double*)PyArray_DATA(dm_);
        // 获取数组的维度信息
        mA = PyArray_DIMS(XA_)[0];
        mB = PyArray_DIMS(XB_)[0];
        n = PyArray_DIMS(XA_)[1];
        // 调用 C 函数计算 Hamming 距离
        cdist_hamming_char(XA, XB, dm, mA, mB, n, w);
        NPY_END_ALLOW_THREADS;
    }
    // 返回一个 Python 对象，此处返回固定的浮点数 0.0
    return Py_BuildValue("d", 0.0);
}

static PyObject *cdist_cosine_double_wrap(PyObject *self, PyObject *args, 
                                               PyObject *kwargs) {
    PyArrayObject *XA_, *XB_, *dm_;
    int mA, mB, n, status;
    double *dm;
    const double *XA, *XB;
    static char *kwlist[] = {"XA", "XB", "dm", NULL};
    // 解析传入的参数，包括两个数据数组 XA 和 XB，一个距离矩阵 dm
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!:cdist_cosine_double_wrap", kwlist,
            &PyArray_Type, &XA_, &PyArray_Type, &XB_, 
            &PyArray_Type, &dm_)) 
    {
        return 0;
    }
    else {
        // 获取数组的数据指针
        XA = (const double*)PyArray_DATA(XA_);
        XB = (const double*)PyArray_DATA(XB_);
        dm = (double*)PyArray_DATA(dm_);
        // 获取数组的维度信息
        mA = PyArray_DIMS(XA_)[0];
        mB = PyArray_DIMS(XB_)[0];
        n = PyArray_DIMS(XA_)[1];
        // 调用 C 函数计算 Cosine 距离
        cdist_cosine_double(XA, XB, dm, mA, mB, n);
    }
    // 返回一个 Python 对象，此处返回固定的浮点数 0.0
    return Py_BuildValue("d", 0.0);
}
    # 在多线程环境中开始执行 NumPy 操作
    NPY_BEGIN_THREADS_DEF;
    # 在多线程环境中开始执行更严格的 NumPy 操作
    NPY_BEGIN_THREADS;
    # 获取输入数组 XA_ 的数据，并将其转换为双精度浮点数的常量指针
    XA = (const double*)PyArray_DATA(XA_);
    # 获取输入数组 XB_ 的数据，并将其转换为双精度浮点数的常量指针
    XB = (const double*)PyArray_DATA(XB_);
    # 获取输出数组 dm_ 的数据，并将其转换为双精度浮点数的指针
    dm = (double*)PyArray_DATA(dm_);
    # 获取输入数组 XA_ 的第一个维度大小（行数）
    mA = PyArray_DIMS(XA_)[0];
    # 获取输入数组 XB_ 的第一个维度大小（行数）
    mB = PyArray_DIMS(XB_)[0];
    # 获取输入数组 XA_ 的第二个维度大小（列数）
    n = PyArray_DIMS(XA_)[1];
    
    # 调用 cdist_cosine 函数计算余弦距离，将结果保存在 dm 中
    status = cdist_cosine(XA, XB, dm, mA, mB, n);
    # 结束多线程环境下的 NumPy 操作
    NPY_END_THREADS;
    # 如果 cdist_cosine 返回负值，表示内存分配失败，返回相应错误
    if(status < 0)
        return PyErr_NoMemory();
    # 返回一个表示成功的 PyFloat 类型对象
    return Py_BuildValue("d", 0.0);
static PyObject *cdist_mahalanobis_double_wrap(PyObject *self, PyObject *args, 
                                               PyObject *kwargs) {
  PyArrayObject *XA_, *XB_, *covinv_, *dm_;
  int mA, mB, n, status;
  double *dm;
  const double *XA, *XB;
  const double *covinv;
  static char *kwlist[] = {"XA", "XB", "dm", "VI", NULL};

  // 解析 Python 函数参数，要求输入为四个数组对象
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!O!:cdist_mahalanobis_double_wrap", kwlist,
            &PyArray_Type, &XA_, &PyArray_Type, &XB_, 
            &PyArray_Type, &dm_, &PyArray_Type, &covinv_)) 
  {
    // 解析失败时返回 0
    return 0;
  }
  else {
    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    // 获取数组数据指针和维度信息
    XA = (const double*)PyArray_DATA(XA_);
    XB = (const double*)PyArray_DATA(XB_);
    covinv = (const double*)PyArray_DATA(covinv_);
    dm = (double*)PyArray_DATA(dm_);
    mA = PyArray_DIMS(XA_)[0];
    mB = PyArray_DIMS(XB_)[0];
    n = PyArray_DIMS(XA_)[1];

    // 调用 C 函数进行 Mahalanobis 距离计算
    status = cdist_mahalanobis(XA, XB, dm, mA, mB, n, covinv);
    NPY_END_THREADS;

    // 若执行出错，返回内存分配错误异常
    if(status < 0)
        return PyErr_NoMemory();
  }

  // 返回 Python 浮点数对象
  return Py_BuildValue("d", 0.0);
}

static PyObject *cdist_minkowski_double_wrap(PyObject *self, PyObject *args, 
                                             PyObject *kwargs) 
{
  PyArrayObject *XA_, *XB_, *dm_;
  int mA, mB, n;
  double *dm;
  const double *XA, *XB;
  double p;
  static char *kwlist[] = {"XA", "XB", "dm", "p", NULL};

  // 解析 Python 函数参数，要求输入为三个数组对象和一个 double 参数
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!d:cdist_minkowski_double_wrap", kwlist,
            &PyArray_Type, &XA_, &PyArray_Type, &XB_, 
            &PyArray_Type, &dm_,
            &p)) {
    // 解析失败时返回 0
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;

    // 获取数组数据指针和维度信息
    XA = (const double*)PyArray_DATA(XA_);
    XB = (const double*)PyArray_DATA(XB_);
    dm = (double*)PyArray_DATA(dm_);
    mA = PyArray_DIMS(XA_)[0];
    mB = PyArray_DIMS(XB_)[0];
    n = PyArray_DIMS(XA_)[1];

    // 调用 C 函数进行 Minkowski 距离计算
    cdist_minkowski(XA, XB, dm, mA, mB, n, p);
    NPY_END_ALLOW_THREADS;
  }

  // 返回 Python 浮点数对象
  return Py_BuildValue("d", 0.0);
}

static PyObject *cdist_seuclidean_double_wrap(PyObject *self, PyObject *args, 
                                              PyObject *kwargs) 
{
  PyArrayObject *XA_, *XB_, *dm_, *var_;
  int mA, mB, n;
  double *dm;
  const double *XA, *XB, *var;
  static char *kwlist[] = {"XA", "XB", "dm", "V", NULL};

  // 解析 Python 函数参数，要求输入为四个数组对象
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!O!:cdist_seuclidean_double_wrap", kwlist,
            &PyArray_Type, &XA_, &PyArray_Type, &XB_, 
            &PyArray_Type, &dm_, &PyArray_Type, &var_)) {
    // 解析失败时返回 0
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;

    // 获取数组数据指针和维度信息
    XA = (const double*)PyArray_DATA(XA_);
    XB = (const double*)PyArray_DATA(XB_);
    dm = (double*)PyArray_DATA(dm_);
    var = (double*)PyArray_DATA(var_);
    mA = PyArray_DIMS(XA_)[0];
    mB = PyArray_DIMS(XB_)[0];
    n = PyArray_DIMS(XA_)[1];

    // 调用 C 函数进行标准化欧氏距离计算
    cdist_seuclidean(XA, XB, var, dm, mA, mB, n);
    NPY_END_ALLOW_THREADS;
  }

  // 返回 Python 浮点数对象
  return Py_BuildValue("d", 0.0);
}
// 定义一个名为 pdist_name_type_wrap 的静态函数，用于包装特定类型和名称的 pdist 函数
#define DEFINE_WRAP_PDIST(name, type)                                   \
    static PyObject *                                                   \
    pdist_ ## name ## _ ## type ## _wrap(PyObject *self, PyObject *args)\
    {
        // 声明 PyArrayObject 类型的指针变量 X_ 和 dm_
        PyArrayObject *X_, *dm_;
        // 声明 Py_ssize_t 类型的变量 m 和 n
        Py_ssize_t m, n;
        // 声明指向 double 类型的指针变量 dm
        double *dm;
        // 声明指向 const type 类型的指针变量 X
        const type *X;
        
        // 使用 PyArg_ParseTuple 解析传入的参数 args，期望参数类型分别为 PyArray_Type 和 PyArray_Type，将解析结果赋值给 X_ 和 dm_
        if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X_,
                                        &PyArray_Type, &dm_)) {
            // 如果解析失败，返回 NULL
            return NULL;
        }
        else {
            // 进入线程安全区域
            NPY_BEGIN_ALLOW_THREADS;
            
            // 将 X_ 的数据指针转换为 const type 类型的指针，并赋值给 X
            X = (const type *)PyArray_DATA(X_);
            // 将 dm_ 的数据指针转换为 double 类型的指针，并赋值给 dm
            dm = (double *)PyArray_DATA(dm_);
            // 获取 X_ 的第一个维度大小，赋值给 m
            m = PyArray_DIMS(X_)[0];
            // 获取 X_ 的第二个维度大小，赋值给 n
            n = PyArray_DIMS(X_)[1];
            
            // 调用以 pdist_ ## name ## _ ## type 命名的函数，传入参数 X, dm, m, n
            pdist_ ## name ## _ ## type(X, dm, m, n);
            
            // 退出线程安全区域
            NPY_END_ALLOW_THREADS;
        }
        
        // 返回一个包含数字 0.0 的 Python 对象，作为函数的返回值
        return Py_BuildValue("d", 0.);
    }
DEFINE_WRAP_PDIST(bray_curtis, double)
// 定义对双精度数据进行布雷-柯蒂斯距离计算的函数包装器

DEFINE_WRAP_PDIST(canberra, double)
// 定义对双精度数据进行坎贝拉距离计算的函数包装器

DEFINE_WRAP_PDIST(chebyshev, double)
// 定义对双精度数据进行切比雪夫距离计算的函数包装器

DEFINE_WRAP_PDIST(city_block, double)
// 定义对双精度数据进行城市街区距离计算的函数包装器

DEFINE_WRAP_PDIST(euclidean, double)
// 定义对双精度数据进行欧几里得距离计算的函数包装器

DEFINE_WRAP_PDIST(jaccard, double)
// 定义对双精度数据进行杰卡德相似系数计算的函数包装器

DEFINE_WRAP_PDIST(jensenshannon, double)
// 定义对双精度数据进行Jensen-Shannon距离计算的函数包装器

DEFINE_WRAP_PDIST(sqeuclidean, double)
// 定义对双精度数据进行平方欧几里得距离计算的函数包装器

DEFINE_WRAP_PDIST(dice, char)
// 定义对字符数据进行Dice距离计算的函数包装器

DEFINE_WRAP_PDIST(kulczynski1, char)
// 定义对字符数据进行Kulczynski相似度计算的函数包装器

DEFINE_WRAP_PDIST(jaccard, char)
// 定义对字符数据进行Jaccard相似度计算的函数包装器

DEFINE_WRAP_PDIST(rogerstanimoto, char)
// 定义对字符数据进行Roger-Tanimoto距离计算的函数包装器

DEFINE_WRAP_PDIST(russellrao, char)
// 定义对字符数据进行Russell-Rao相似度计算的函数包装器

DEFINE_WRAP_PDIST(sokalmichener, char)
// 定义对字符数据进行Sokal-Michener距离计算的函数包装器

DEFINE_WRAP_PDIST(sokalsneath, char)
// 定义对字符数据进行Sokal-Sneath距离计算的函数包装器

DEFINE_WRAP_PDIST(yule, char)
// 定义对字符数据进行Yule相似度计算的函数包装器

static PyObject *pdist_hamming_double_wrap(
                            PyObject *self, PyObject *args, PyObject *kwargs) 
{
    PyArrayObject *X_, *dm_, *w_;
    int m, n;
    double *dm;
    const double *X, *w;
    static char *kwlist[] = {"X", "dm", "w", NULL};
    // 解析函数参数，接收三个数组对象 X_, dm_, w_
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
              "O!O!O!:pdist_hamming_double_wrap", kwlist,
              &PyArray_Type, &X_,
              &PyArray_Type, &dm_,
              &PyArray_Type, &w_)) {
        return 0;
    }
    else {
        NPY_BEGIN_ALLOW_THREADS;
        // 获取数组数据指针及维度信息
        X = (const double*)PyArray_DATA(X_);
        dm = (double*)PyArray_DATA(dm_);
        w = (const double*)PyArray_DATA(w_);
        m = PyArray_DIMS(X_)[0];
        n = PyArray_DIMS(X_)[1];

        // 调用函数计算双精度数据的汉明距离
        pdist_hamming_double(X, dm, m, n, w);
        NPY_END_ALLOW_THREADS;
    }
    // 返回双精度浮点数对象
    return Py_BuildValue("d", 0.0);
}

static PyObject *pdist_hamming_char_wrap(
                            PyObject *self, PyObject *args, PyObject *kwargs) 
{
    PyArrayObject *X_, *dm_, *w_;
    int m, n;
    const char *X;
    const double *w;
    double *dm;
    static char *kwlist[] = {"X", "dm", "w", NULL};
    // 解析函数参数，接收三个数组对象 X_, dm_, w_
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
              "O!O!O!:pdist_hamming_char_wrap", kwlist,
              &PyArray_Type, &X_,
              &PyArray_Type, &dm_,
              &PyArray_Type, &w_)) {
        return 0;
    }
    else {
        NPY_BEGIN_ALLOW_THREADS;
        // 获取数组数据指针及维度信息
        X = (const char*)PyArray_DATA(X_);
        dm = (double*)PyArray_DATA(dm_);
        w = (const double*)PyArray_DATA(w_);
        m = PyArray_DIMS(X_)[0];
        n = PyArray_DIMS(X_)[1];

        // 调用函数计算字符数据的汉明距离
        pdist_hamming_char(X, dm, m, n, w);
        NPY_END_ALLOW_THREADS;
    }
    // 返回双精度浮点数对象
    return Py_BuildValue("d", 0.0);
}

static PyObject *pdist_cosine_double_wrap(PyObject *self, PyObject *args, 
                                          PyObject *kwargs) 
{
    PyArrayObject *X_, *dm_;
    int m, n, status;
    double *dm;
    const double *X;
    static char *kwlist[] = {"X", "dm", NULL};
    // 解析函数参数，接收两个数组对象 X_, dm_
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
              "O!O!:pdist_cosine_double_wrap", kwlist,
              &PyArray_Type, &X_,
              &PyArray_Type, &dm_)) {
        return 0;
    }
    else {
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS;
        // 获取数组数据指针及维度信息
        X = (const double*)PyArray_DATA(X_);
        dm = (double*)PyArray_DATA(dm_);
        m = PyArray_DIMS(X_)[0];
        n = PyArray_DIMS(X_)[1];
        
        // 调用函数计算双精度数据的余弦距离
        status = pdist_cosine(X, dm, m, n);
        NPY_END_THREADS;
    # 如果 status 小于 0，则表示内存分配失败，返回内存分配失败的异常
    if(status < 0)
        return PyErr_NoMemory();
  }
  # 如果没有出现内存分配失败，返回一个 Python 浮点类型的值 0.0
  return Py_BuildValue("d", 0.0);
# 定义一个静态函数pdish_mahalanobis_double_wrap，用于计算马哈拉诺比斯距离
static PyObject *pdist_mahalanobis_double_wrap(PyObject *self, PyObject *args, 
                                               PyObject *kwargs) {
  PyArrayObject *X_, *covinv_, *dm_;
  int m, n, status;
  double *dm;
  const double *X;
  const double *covinv;
  // 定义关键字列表，用于解析参数
  static char *kwlist[] = {"X", "dm", "VI", NULL};
  // 解析参数并检查是否符合指定格式
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!:pdist_mahalanobis_double_wrap", kwlist,
            &PyArray_Type, &X_,
            &PyArray_Type, &dm_, 
            &PyArray_Type, &covinv_)) {
    // 解析失败，返回0表示出错
    return 0;
  }
  else {
    // 开始多线程操作
    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;
    // 获取数组数据指针
    X = (const double*)PyArray_DATA(X_);
    covinv = (const double*)PyArray_DATA(covinv_);
    dm = (double*)PyArray_DATA(dm_);
    // 获取数组的维度信息
    m = PyArray_DIMS(X_)[0];
    n = PyArray_DIMS(X_)[1];

    // 调用C函数pdist_mahalanobis计算马哈拉诺比斯距离
    status = pdist_mahalanobis(X, dm, m, n, covinv);
    // 结束多线程操作
    NPY_END_THREADS;
    // 如果计算失败，返回内存错误信息
    if(status < 0)
        return PyErr_NoMemory();
  }
  // 返回一个Python浮点数对象
  return Py_BuildValue("d", 0.0);
}

# 定义一个静态函数pdish_minkowski_double_wrap，用于计算闵可夫斯基距离
static PyObject *pdist_minkowski_double_wrap(PyObject *self, PyObject *args, 
                                             PyObject *kwargs) 
{
  PyArrayObject *X_, *dm_;
  int m, n;
  double *dm, *X;
  double p;
  // 定义关键字列表，用于解析参数
  static char *kwlist[] = {"X", "dm", "p", NULL};
  // 解析参数并检查是否符合指定格式
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!d:pdist_minkowski_double_wrap", kwlist,
            &PyArray_Type, &X_,
            &PyArray_Type, &dm_,
            &p)) {
    // 解析失败，返回0表示出错
    return 0;
  }
  else {
    // 开始多线程操作
    NPY_BEGIN_ALLOW_THREADS;
    // 获取数组数据指针
    X = (double*)PyArray_DATA(X_);
    dm = (double*)PyArray_DATA(dm_);
    // 获取数组的维度信息
    m = PyArray_DIMS(X_)[0];
    n = PyArray_DIMS(X_)[1];

    // 调用C函数pdist_minkowski计算闵可夫斯基距离
    pdist_minkowski(X, dm, m, n, p);
    // 结束多线程操作
    NPY_END_THREADS;
  }
  // 返回一个Python浮点数对象
  return Py_BuildValue("d", 0.0);
}

# 定义一个静态函数pdish_seuclidean_double_wrap，用于计算标准化欧几里得距离
static PyObject *pdist_seuclidean_double_wrap(PyObject *self, PyObject *args, 
                                              PyObject *kwargs) 
{
  PyArrayObject *X_, *dm_, *var_;
  int m, n;
  double *dm;
  const double *X, *var;
  // 定义关键字列表，用于解析参数
  static char *kwlist[] = {"X", "dm", "V", NULL};
  // 解析参数并检查是否符合指定格式
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, 
            "O!O!O!:pdist_seuclidean_double_wrap", kwlist,
            &PyArray_Type, &X_,
            &PyArray_Type, &dm_,
            &PyArray_Type, &var_)) {
    // 解析失败，返回0表示出错
    return 0;
  }
  else {
    // 开始多线程操作
    NPY_BEGIN_ALLOW_THREADS;
    // 获取数组数据指针
    X = (double*)PyArray_DATA(X_);
    dm = (double*)PyArray_DATA(dm_);
    var = (double*)PyArray_DATA(var_);
    // 获取数组的维度信息
    m = PyArray_DIMS(X_)[0];
    n = PyArray_DIMS(X_)[1];

    // 调用C函数pdist_seuclidean计算标准化欧几里得距离
    pdist_seuclidean(X, var, dm, m, n);
    // 结束多线程操作
    NPY_END_THREADS;
  }
  // 返回一个Python浮点数对象
  return Py_BuildValue("d", 0.0);
}

# 定义一个静态函数pdish_weighted_chebyshev_double_wrap，用于计算加权切比雪夫距离
static PyObject *pdist_weighted_chebyshev_double_wrap(
  PyObject *self, PyObject *args, PyObject *kwargs)
{
  // 定义 NumPy 数组对象指针 X_, dm_, w_
  PyArrayObject *X_, *dm_, *w_;
  // 定义整型变量 m, n
  int m, n;
  // 定义双精度浮点型指针 dm, X, w
  double *dm, *X, *w;
  // 定义静态字符指针数组 kwlist，用于参数解析
  static char *kwlist[] = {"X", "dm", "w", NULL};
  // 解析输入参数 args 和 kwargs，要求参数格式为 "O!O!O!:pdist_weighted_minkowski_double_wrap"
  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                   "O!O!O!:pdist_weighted_minkowski_double_wrap", kwlist,
                                   &PyArray_Type, &X_,  // 解析 X 参数
                                   &PyArray_Type, &dm_, // 解析 dm 参数
                                   &PyArray_Type, &w_)) { // 解析 w 参数
    return 0; // 解析失败，返回 0
  }
  else {
    // 允许线程
    NPY_BEGIN_ALLOW_THREADS;
    // 获取 X_, dm_, w_ 的数据指针并转换为双精度浮点数组 X, dm, w
    X = (double*)PyArray_DATA(X_);
    dm = (double*)PyArray_DATA(dm_);
    w = (double*)PyArray_DATA(w_);
    // 获取 X_ 的维度信息，m 为行数，n 为列数
    m = PyArray_DIMS(X_)[0];
    n = PyArray_DIMS(X_)[1];

    // 调用 C 函数 pdist_weighted_chebyshev 计算加权切比雪夫距离
    pdist_weighted_chebyshev(X, dm, m, n, w);
    // 结束线程
    NPY_END_ALLOW_THREADS;
  }
  // 返回一个双精度浮点数值对象
  return Py_BuildValue("d", 0.0);
}

// 定义 Python 函数 pdist_weighted_minkowski_double_wrap
static PyObject *pdist_weighted_minkowski_double_wrap(
                            PyObject *self, PyObject *args, PyObject *kwargs)
{
  // 定义 NumPy 数组对象指针 X_, dm_, w_
  PyArrayObject *X_, *dm_, *w_;
  // 定义整型变量 m, n
  int m, n;
  // 定义双精度浮点型指针 dm, X, w
  double *dm, *X, *w;
  // 定义双精度浮点数 p
  double p;
  // 定义静态字符指针数组 kwlist，用于参数解析
  static char *kwlist[] = {"X", "dm", "p", "w", NULL};
  // 解析输入参数 args 和 kwargs，要求参数格式为 "O!O!dO!:pdist_weighted_minkowski_double_wrap"
  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O!O!dO!:pdist_weighted_minkowski_double_wrap", kwlist,
            &PyArray_Type, &X_,   // 解析 X 参数
            &PyArray_Type, &dm_,  // 解析 dm 参数
            &p,                   // 解析 p 参数
            &PyArray_Type, &w_)) { // 解析 w 参数
    return 0; // 解析失败，返回 0
  }
  else {
    // 允许线程
    NPY_BEGIN_ALLOW_THREADS;
    // 获取 X_, dm_, w_ 的数据指针并转换为双精度浮点数组 X, dm, w
    X = (double*)PyArray_DATA(X_);
    dm = (double*)PyArray_DATA(dm_);
    w = (double*)PyArray_DATA(w_);
    // 获取 X_ 的维度信息，m 为行数，n 为列数
    m = PyArray_DIMS(X_)[0];
    n = PyArray_DIMS(X_)[1];

    // 调用 C 函数 pdist_weighted_minkowski 计算加权闵可夫斯基距离
    pdist_weighted_minkowski(X, dm, m, n, p, w);
    // 结束线程
    NPY_END_ALLOW_THREADS;
  }
  // 返回一个双精度浮点数值对象
  return Py_BuildValue("d", 0.0);
}

// 定义 Python 函数 to_squareform_from_vector_wrap
static PyObject *to_squareform_from_vector_wrap(PyObject *self, PyObject *args) 
{
  // 定义 NumPy 数组对象指针 M_, v_
  PyArrayObject *M_, *v_;
  // 定义整型变量 n
  int n;
  // 定义 numpy 的整型，用于存储数组项的大小
  npy_intp elsize;
  // 解析输入参数 args，要求参数格式为 "O!O!"
  if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &M_,  // 解析 M 参数
            &PyArray_Type, &v_)) { // 解析 v 参数
    return 0; // 解析失败，返回 0
  }
  // 允许线程
  NPY_BEGIN_ALLOW_THREADS;
  // 获取 M_ 的维度信息，n 为数组的大小
  n = PyArray_DIMS(M_)[0];
  // 获取 M_ 的数组项大小
  elsize = PyArray_ITEMSIZE(M_);
  // 如果数组项大小为 8 字节
  if (elsize == 8) {
    // 调用 C 函数 dist_to_squareform_from_vector_double
    dist_to_squareform_from_vector_double(
        (double*)PyArray_DATA(M_), (const double*)PyArray_DATA(v_), n);
  } else {
    // 调用通用函数 dist_to_squareform_from_vector_generic
    dist_to_squareform_from_vector_generic(
        (char*)PyArray_DATA(M_), (const char*)PyArray_DATA(v_), n, elsize);
  }
  // 结束线程
  NPY_END_ALLOW_THREADS;
  // 返回一个空字符串对象
  return Py_BuildValue("");
}

// 定义 Python 函数 to_vector_from_squareform_wrap
static PyObject *to_vector_from_squareform_wrap(PyObject *self, PyObject *args) 
{
  // 定义 NumPy 数组对象指针 M_, v_
  PyArrayObject *M_, *v_;
  // 定义整型变量 n
  int n;
  // 定义 numpy 的整型，用于存储数组项的大小
  npy_intp s;
  // 定义字符型指针 v, M
  char *v;
  const char *M;
  // 解析输入参数 args，要求参数格式为 "O!O!"
  if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &M_,  // 解析 M 参数
            &PyArray_Type, &v_)) { // 解析 v 参数
    return 0; // 解析失败，返回 0
  }
  else {
    // 允许线程
    NPY_BEGIN_ALLOW_THREADS;
    // 获取 M_ 的数据指针并转换为字符型数组 M
    M = (char*)PyArray_DATA(M_);
    // 获取 v_ 的数据指针并转换为字符型数组 v
    v = (char*)PyArray_DATA(v_);
    // 获取 M_ 的维度信息，n 为数组的大小
    n = PyArray_DIMS(M_)[0];
    // 获取 M_ 的数组项大小
    s = PyArray_ITEMSIZE(M_);
    // 调用 C 函数 dist_to_vector_from_squareform
    dist_to_vector_from_squareform(M, v, n, s);
    // 结束线程
    NPY_END_ALLOW_THREADS;
  }
  // 返回一个空字符串对象
  return Py_BuildValue("");
}
    NULL,
    NULL



    # 这里是一个示例占位符，表示代码中的空行或未定义的变量/值
    # 在实际代码中，应根据具体情况替换为实际的代码或变量名
    # 如果这里使用了 NULL，可能意味着在某些语言或库中表示空值或未初始化值的占位符
    # 在实际应用中，应根据语言和上下文来理解具体含义和用途
    NULL,
    NULL


这段代码是一个示例，展示了在程序中使用的占位符或未定义的变量/值。
};

PyMODINIT_FUNC
PyInit__distance_wrap(void)
{
    // 导入 NumPy C API，使得在模块中可以使用 NumPy 数组
    import_array();
    // 使用 moduledef 创建 Python 模块并返回
    return PyModule_Create(&moduledef);
}
```