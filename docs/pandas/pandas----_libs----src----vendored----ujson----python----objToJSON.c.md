# `D:\src\scipysrc\pandas\pandas\_libs\src\vendored\ujson\python\objToJSON.c`

```
/*
Copyright (c) 2011-2013, ESN Social Software AB and Jonas Tarnstrom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the ESN Social Software AB nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ESN SOCIAL SOFTWARE AB OR JONAS TARNSTROM BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Portions of code from MODP_ASCII - Ascii transformations (upper/lower, etc)
https://github.com/client9/stringencoders
Copyright (c) 2007  Nick Galbreath -- nickg [at] modp [dot] com. All rights
reserved.

Numeric decoder derived from TCL library
https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
* Copyright (c) 1988-1993 The Regents of the University of California.
* Copyright (c) 1994 Sun Microsystems, Inc.
*/

// Licence at LICENSES/ULTRAJSON_LICENSE

// 定义用于清除 PY_SSIZE_T 的宏，防止导入冲突
#define PY_SSIZE_T_CLEAN
// 导入 Python.h 头文件，提供 Python C API 功能
#include <Python.h>

// 定义不导入数组功能的宏，用于避免冲突
#define NO_IMPORT_ARRAY
// 定义用于 numpy 的符号，防止命名冲突
#define PY_ARRAY_UNIQUE_SYMBOL UJSON_NUMPY
// 导入日期时间相关头文件
#include "datetime.h"
// 导入 Pandas 的日期时间相关头文件
#include "pandas/datetime/pd_datetime.h"
// 导入 UltraJSON 库的头文件
#include "pandas/vendored/ujson/lib/ultrajson.h"
// 导入 NumPy 的数组对象头文件
#include <numpy/arrayobject.h>
// 导入 NumPy 的标量数组头文件
#include <numpy/arrayscalars.h>
// 导入 NumPy 的多维数组类型头文件
#include <numpy/ndarraytypes.h>
// 导入 NumPy 的数学函数头文件
#include <numpy/npy_math.h>

// 返回 NPY_MIN_INT64，即 NumPy 中的最小整数
npy_int64 get_nat(void) { return NPY_MIN_INT64; }

// 定义 PFN_PyTypeToUTF8 类型，用于将 Python 对象转换为 UTF-8 字符串
typedef char *(*PFN_PyTypeToUTF8)(JSOBJ obj, JSONTypeContext *ti,
                                  size_t *_outLen);

// 检查对象是否为 decimal 类型
int object_is_decimal_type(PyObject *obj);
// 检查对象是否为 dataframe 类型
int object_is_dataframe_type(PyObject *obj);
// 检查对象是否为 series 类型
int object_is_series_type(PyObject *obj);
// 检查对象是否为 index 类型
int object_is_index_type(PyObject *obj);
// 检查对象是否为 nat 类型
int object_is_nat_type(PyObject *obj);
// 检查对象是否为 na 类型
int object_is_na_type(PyObject *obj);
typedef struct __NpyArrContext {
  PyObject *array;         // 指向一个 PyObject 对象，表示一个数组
  char *dataptr;           // 指向数组中当前元素的数据指针
  npy_intp curdim;         // 当前数组的维度顺序
  npy_intp stridedim;      // 当前正在跨越的维度
  int inc;                 // 步长维度的增量（+/- 1）
  npy_intp dim;            // 维度
  npy_intp stride;         // 步长
  npy_intp ndim;           // 数组的维数
  npy_intp index[NPY_MAXDIMS];  // 每个维度的索引
  int type_num;            // 类型编号

  char **rowLabels;        // 行标签数组
  char **columnLabels;     // 列标签数组
} NpyArrContext;

typedef struct __PdBlockContext {
  Py_ssize_t colIdx;       // 列索引
  Py_ssize_t ncols;        // 列数
  int transpose;           // 是否进行转置操作

  NpyArrContext **npyCtxts;  // 每列对应的 NpyArrContext
} PdBlockContext;

typedef struct __TypeContext {
  JSPFN_ITERBEGIN iterBegin;      // 迭代开始函数指针
  JSPFN_ITEREND iterEnd;          // 迭代结束函数指针
  JSPFN_ITERNEXT iterNext;        // 迭代下一个函数指针
  JSPFN_ITERGETNAME iterGetName;  // 获取迭代名称函数指针
  JSPFN_ITERGETVALUE iterGetValue;  // 获取迭代值函数指针
  PFN_PyTypeToUTF8 PyTypeToUTF8;  // Python 类型转 UTF-8 字符串函数指针
  PyObject *newObj;               // 新对象
  PyObject *dictObj;              // 字典对象
  Py_ssize_t index;               // 索引
  Py_ssize_t size;                // 大小
  PyObject *itemValue;            // 项值
  PyObject *itemName;             // 项名
  PyObject *attrList;             // 属性列表
  PyObject *iterator;             // 迭代器

  double doubleValue;             // 双精度浮点数值
  JSINT64 longValue;              // 64 位整数值

  char *cStr;                     // C 字符串
  NpyArrContext *npyarr;          // NpyArrContext 对象
  PdBlockContext *pdblock;        // PdBlockContext 对象
  int transpose;                  // 是否进行转置操作
  char **rowLabels;               // 行标签数组
  char **columnLabels;            // 列标签数组
  npy_intp rowLabelsLen;          // 行标签数组长度
  npy_intp columnLabelsLen;       // 列标签数组长度
} TypeContext;

typedef struct __PyObjectEncoder {
  JSONObjectEncoder enc;          // JSON 对象编码器

  // 编码多维数组时通过 NpyArrContext 传递
  NpyArrContext *npyCtxtPassthru;

  // 编码块时通过 PdBlockContext 传递
  PdBlockContext *blkCtxtPassthru;

  // 直接编码 numpy 数据的传递
  int npyType;                    // numpy 数据类型
  void *npyValue;                 // numpy 数据值

  int datetimeIso;                // ISO 日期时间标志
  NPY_DATETIMEUNIT datetimeUnit;  // 日期时间单位
  NPY_DATETIMEUNIT valueUnit;     // 值单位

  // pandas 数据类型的输出格式风格
  int outputFormat;               // 输出格式
  int originalOutputFormat;       // 原始输出格式

  PyObject *defaultHandler;       // 默认处理器对象
} PyObjectEncoder;

// 定义枚举 PANDAS_FORMAT，表示 pandas 数据的输出格式
#define GET_TC(__ptrtc) ((TypeContext *)((__ptrtc)->prv))

enum PANDAS_FORMAT { SPLIT, RECORDS, INDEX, COLUMNS, VALUES };

// 定义静态函数 PdBlock_iterNext，用于迭代 PdBlock 的下一个元素
static int PdBlock_iterNext(JSOBJ, JSONTypeContext *);

// 创建并初始化 TypeContext 结构体的函数
static TypeContext *createTypeContext(void) {
  TypeContext *pc = PyObject_Malloc(sizeof(TypeContext));  // 分配内存空间
  if (!pc) {  // 检查内存是否分配成功
    PyErr_NoMemory();  // 内存分配失败，抛出内存错误异常
    return NULL;  // 返回空指针
  }
  // 初始化 TypeContext 结构体的各个成员
  pc->newObj = NULL;
  pc->dictObj = NULL;
  pc->itemValue = NULL;
  pc->itemName = NULL;
  pc->attrList = NULL;
  pc->index = 0;
  pc->size = 0;
  pc->longValue = 0;
  pc->doubleValue = 0.0;
  pc->cStr = NULL;
  pc->npyarr = NULL;
  pc->pdblock = NULL;
  pc->rowLabels = NULL;
  pc->columnLabels = NULL;
  pc->transpose = 0;
  pc->rowLabelsLen = 0;
  pc->columnLabelsLen = 0;

  return pc;  // 返回初始化后的 TypeContext 结构体指针
}

// 获取对象的值的函数，返回 PyObject 指针
static PyObject *get_values(PyObject *obj) {
  PyObject *values = NULL;  // 定义并初始化 PyObject 指针

  // 检查对象是否是索引类型或者序列类型
  if (object_is_index_type(obj) || object_is_series_type(obj)) {
    // 特殊情况是 dt64tz 和 category[dt64tz]，返回 UTC 本地化的 datetime64 ndarray
    // 而不是 Timestamps 对象数组
    // 检查对象是否具有属性 "tz"
    if (PyObject_HasAttrString(obj, "tz")) {
      // 获取对象的 "tz" 属性
      PyObject *tz = PyObject_GetAttrString(obj, "tz");
      // 如果 tz 不是 None，则继续处理
      if (tz != Py_None) {
        Py_DECREF(tz);
        // 调用对象的 "__array__" 方法，返回处理后的数据
        values = PyObject_CallMethod(obj, "__array__", NULL);
        return values;
      }
      // 释放 tz 对象的引用
      Py_DECREF(tz);
    }
    // 获取对象的 "values" 属性
    values = PyObject_GetAttrString(obj, "values");
    // 如果获取失败，则清除当前的异常状态以便后续尝试
    if (values == NULL) {
      PyErr_Clear();
    } else if (PyObject_HasAttrString(values, "__array__")) {
      // 如果 "values" 具有 "__array__" 属性，则调用该方法
      PyObject *array_values = PyObject_CallMethod(values, "__array__", NULL);
      Py_DECREF(values);
      // 将处理后的数组值赋给 values
      values = array_values;
    } else if (!PyArray_CheckExact(values)) {
      // 如果 values 不是精确的 numpy 数组类型，则置为 NULL，表示尝试下一方法
      Py_DECREF(values);
      values = NULL;
    }

  }

  // 如果 values 仍然为 NULL，则处理类型错误异常并返回 NULL
  if (values == NULL) {
    // 获取对象类型的字符串表示
    PyObject *typeRepr = PyObject_Repr((PyObject *)Py_TYPE(obj));
    PyObject *repr;
    // 如果对象具有 "dtype" 属性，则获取其表示字符串
    if (PyObject_HasAttrString(obj, "dtype")) {
      PyObject *dtype = PyObject_GetAttrString(obj, "dtype");
      repr = PyObject_Repr(dtype);
      Py_DECREF(dtype);
    } else {
      // 否则使用默认的未知类型字符串表示
      repr = PyUnicode_FromString("<unknown dtype>");
    }

    // 格式化错误消息并设置异常
    PyErr_Format(PyExc_ValueError, "%R or %R are not JSON serializable yet",
                 repr, typeRepr);
    Py_DECREF(repr);
    Py_DECREF(typeRepr);

    return NULL;
  }

  // 返回处理后的 values 数据
  return values;
/* 
   获取对象的指定属性的子属性，并返回一个新的 PyObject 对象。
   如果获取过程中出现错误或找不到属性，则返回空指针。
*/
static PyObject *get_sub_attr(PyObject *obj, char *attr, char *subAttr) {
  // 获取 obj 对象的属性 attr 对应的子属性 tmp
  PyObject *tmp = PyObject_GetAttrString(obj, attr);
  // 如果获取失败，返回空指针
  if (tmp == 0) {
    return 0;
  }
  // 获取 tmp 对象的属性 subAttr 对应的值 ret
  PyObject *ret = PyObject_GetAttrString(tmp, subAttr);
  // 释放临时对象 tmp 的引用计数
  Py_DECREF(tmp);

  return ret;
}

/* 
   获取对象的指定属性的长度，并返回其作为 Py_ssize_t 类型的值。
   如果获取过程中出现错误或找不到属性，则返回 0。
*/
static Py_ssize_t get_attr_length(PyObject *obj, char *attr) {
  // 获取 obj 对象的属性 attr 对应的值 tmp
  PyObject *tmp = PyObject_GetAttrString(obj, attr);
  // 如果获取失败，返回 0
  if (tmp == 0) {
    return 0;
  }
  // 获取 tmp 对象的长度作为 ret
  Py_ssize_t ret = PyObject_Length(tmp);
  // 释放临时对象 tmp 的引用计数
  Py_DECREF(tmp);

  // 如果获取长度失败，返回 0
  if (ret == -1) {
    return 0;
  }

  return ret;
}

/* 
   获取对象的指定属性的长整型值。
   如果对象类型是 Timedelta、Timestamp 或 NaT，则假设其类型。
*/
static npy_int64 get_long_attr(PyObject *o, const char *attr) {
  // 获取对象 o 的属性 attr 对应的值 value
  PyObject *value = PyObject_GetAttrString(o, attr);
  // 根据值的类型，转换为长整型 long_val
  const npy_int64 long_val =
      (PyLong_Check(value) ? PyLong_AsLongLong(value) : PyLong_AsLong(value));
  // 释放值对象的引用计数
  Py_DECREF(value);

  // 如果对象是 NaT 类型，返回 NPY_MIN_INT64
  if (object_is_nat_type(o)) {
    return long_val;
  }

  // 确保单位是纳秒，类似于 Timestamp._as_creso 或 _as_unit
  PyObject *reso = PyObject_GetAttrString(o, "_creso");
  // 如果单位不是长整型，则返回 -1
  if (!PyLong_Check(reso)) {
    Py_DECREF(reso);
    return -1;
  }

  // 获取单位的长整型值 cReso
  long cReso = PyLong_AsLong(reso);
  Py_DECREF(reso);
  // 如果获取单位值失败，返回 -1
  if (cReso == -1 && PyErr_Occurred()) {
    return -1;
  }

  // 根据单位转换 long_val 的值为纳秒
  if (cReso == NPY_FR_us) {
    return long_val * 1000L;
  } else if (cReso == NPY_FR_ms) {
    return long_val * 1000000L;
  } else if (cReso == NPY_FR_s) {
    return long_val * 1000000000L;
  }

  return long_val;
}

/* 
   调用对象的 total_seconds 方法，并返回其作为双精度浮点数的值。
*/
static npy_float64 total_seconds(PyObject *td) {
  // 调用对象 td 的 total_seconds 方法，获取返回值 value
  PyObject *value = PyObject_CallMethod(td, "total_seconds", NULL);
  // 将返回值转换为双精度浮点数 double_val
  const npy_float64 double_val = PyFloat_AS_DOUBLE(value);
  // 释放返回值对象的引用计数
  Py_DECREF(value);
  return double_val;
}

/* 
   将 PyBytes 对象转换为 UTF-8 编码的字符串，并返回其指针。
*/
static char *PyBytesToUTF8(JSOBJ _obj, JSONTypeContext *Py_UNUSED(tc),
                           size_t *_outLen) {
  // 将 _obj 转换为 PyObject 对象
  PyObject *obj = (PyObject *)_obj;
  // 获取对象的字节长度并赋值给 *_outLen
  *_outLen = PyBytes_GET_SIZE(obj);
  // 返回对象的 UTF-8 字符串指针
  return PyBytes_AS_STRING(obj);
}

/* 
   将 PyUnicode 对象转换为 UTF-8 编码的字符串，并返回其指针。
   如果转换失败，则设置错误消息并返回 NULL。
*/
static char *PyUnicodeToUTF8(JSOBJ _obj, JSONTypeContext *tc, size_t *_outLen) {
  // 将 _obj 转换为 PyUnicode 对象，并获取其 UTF-8 编码字符串及长度
  char *encoded = (char *)PyUnicode_AsUTF8AndSize(_obj, (Py_ssize_t *)_outLen);
  // 如果转换失败，设置错误消息并返回 NULL
  if (encoded == NULL) {
    JSONObjectEncoder *enc = (JSONObjectEncoder *)tc->encoder;
    enc->errorMsg = "Encoding failed.";
  }
  // 返回 UTF-8 编码的字符串指针
  return encoded;
}

/* 
   将 Numpy 的日期时间对象转换为 ISO 格式的字符串回调函数。
   返回 ISO 格式的字符串指针，并更新 len 指向的长度。
*/
static char *NpyDateTimeToIsoCallback(JSOBJ Py_UNUSED(unused),
                                      JSONTypeContext *tc, size_t *len) {
  // 获取 datetimeUnit 和 valueUnit
  NPY_DATETIMEUNIT base = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
  NPY_DATETIMEUNIT valueUnit = ((PyObjectEncoder *)tc->encoder)->valueUnit;
  // 调用 int64ToIso 函数转换为 ISO 格式的字符串，并更新 len 指向的长度
  GET_TC(tc)->cStr = int64ToIso(GET_TC(tc)->longValue, valueUnit, base, len);
  // 返回 ISO 格式的字符串指针
  return GET_TC(tc)->cStr;
}
/* 将 Numpy 时间差转换为 ISO 格式字符串的回调函数 */
static char *NpyTimeDeltaToIsoCallback(JSOBJ Py_UNUSED(unused),
                                       JSONTypeContext *tc, size_t *len) {
  // 调用 int64ToIsoDuration 函数将长整型时间差转换为 ISO 格式字符串，并存储在 tc->cStr 中
  GET_TC(tc)->cStr = int64ToIsoDuration(GET_TC(tc)->longValue, len);
  // 返回 tc->cStr 指向的字符串
  return GET_TC(tc)->cStr;
}

/* 将 Python 的日期时间对象转换为 ISO 格式字符串的回调函数 */
static char *PyDateTimeToIsoCallback(JSOBJ obj, JSONTypeContext *tc,
                                     size_t *len) {
  // 检查 obj 是否为日期或日期时间对象，若不是则设置错误并返回 NULL
  if (!PyDate_Check(obj) && !PyDateTime_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "Expected date or datetime object");
    // 清空错误消息
    ((JSONObjectEncoder *)tc->encoder)->errorMsg = "";
    return NULL;
  }

  // 获取日期时间单位
  NPY_DATETIMEUNIT base = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
  // 调用 PyDateTimeToIso 函数将日期时间对象转换为 ISO 格式字符串，并返回结果
  return PyDateTimeToIso(obj, base, len);
}

/* 将 Python 的时间对象转换为 ISO 格式字符串的函数 */
static char *PyTimeToJSON(JSOBJ _obj, JSONTypeContext *tc, size_t *outLen) {
  // 将输入的时间对象转换为 PyObject 类型
  PyObject *obj = (PyObject *)_obj;
  // 调用时间对象的 isoformat 方法获取 ISO 格式的字符串表示
  PyObject *str = PyObject_CallMethod(obj, "isoformat", NULL);
  // 如果获取失败，设置错误并返回 NULL
  if (str == NULL) {
    *outLen = 0;
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, "Failed to convert time");
    }
    // 清空错误消息
    ((JSONObjectEncoder *)tc->encoder)->errorMsg = "";
    return NULL;
  }
  // 如果返回的是 Unicode 字符串，则转换为 UTF-8 字符串
  if (PyUnicode_Check(str)) {
    PyObject *tmp = str;
    str = PyUnicode_AsUTF8String(str);
    Py_DECREF(tmp);
  }

  // 将转换后的字符串存储在 tc->newObj 中
  GET_TC(tc)->newObj = str;

  // 获取字符串的长度并返回其指针
  *outLen = PyBytes_GET_SIZE(str);
  char *outValue = PyBytes_AS_STRING(str);
  return outValue;
}

//=============================================================================
// Numpy 数组迭代函数
//=============================================================================

/* 释放 Numpy 数组元素值的函数 */
static void NpyArr_freeItemValue(JSOBJ Py_UNUSED(_obj), JSONTypeContext *tc) {
  // 如果当前上下文中的 numpy 数组存在且元素值不是数组本身，则释放元素值
  if (GET_TC(tc)->npyarr &&
      GET_TC(tc)->itemValue != GET_TC(tc)->npyarr->array) {
    Py_XDECREF(GET_TC(tc)->itemValue);
    GET_TC(tc)->itemValue = NULL;
  }
}

/* Numpy 数组迭代器的下一个函数 */
static int NpyArr_iterNextNone(JSOBJ Py_UNUSED(_obj),
                               JSONTypeContext *Py_UNUSED(tc)) {
  // 总是返回 0，表示没有下一个元素
  return 0;
}

/* 初始化 Numpy 数组迭代器的函数 */
static void NpyArr_iterBegin(JSOBJ _obj, JSONTypeContext *tc) {
  // 将输入对象转换为 PyArrayObject 类型
  PyArrayObject *obj =
      (PyArrayObject *)(GET_TC(tc)->newObj ? GET_TC(tc)->newObj : _obj);

  // 分配 NpyArrContext 结构的内存空间
  NpyArrContext *npyarr = PyObject_Malloc(sizeof(NpyArrContext));
  GET_TC(tc)->npyarr = npyarr;

  // 如果内存分配失败，设置错误并返回
  if (!npyarr) {
    PyErr_NoMemory();
    GET_TC(tc)->iterNext = NpyArr_iterNextNone;
    return;
  }

  // 初始化 NpyArrContext 结构的各个字段
  npyarr->array = (PyObject *)obj;
  npyarr->dataptr = PyArray_DATA(obj);
  npyarr->ndim = PyArray_NDIM(obj) - 1;
  npyarr->curdim = 0;
  npyarr->type_num = PyArray_DESCR(obj)->type_num;

  // 根据是否需要转置来设置维度、步幅等信息
  if (GET_TC(tc)->transpose) {
    npyarr->dim = PyArray_DIM(obj, (int)npyarr->ndim);
    npyarr->stride = PyArray_STRIDE(obj, (int)npyarr->ndim);
    npyarr->stridedim = npyarr->ndim;
    npyarr->index[npyarr->ndim] = 0;
    npyarr->inc = -1;
  } else {
    npyarr->dim = PyArray_DIM(obj, 0);
    npyarr->stride = PyArray_STRIDE(obj, 0);
    npyarr->stridedim = 0;
    npyarr->index[0] = 0;
    npyarr->inc = 1;
  }

  // 将列标签和行标签存储在 NpyArrContext 结构中
  npyarr->columnLabels = GET_TC(tc)->columnLabels;
  npyarr->rowLabels = GET_TC(tc)->rowLabels;
}
// 结束 NpyArr 迭代器的操作，释放相关资源
static void NpyArr_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    // 获取 JSON 上下文中的 NpyArrContext 结构体指针
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;

    // 如果 NpyArrContext 存在，释放其所持有的数据项值并释放内存
    if (npyarr) {
        NpyArr_freeItemValue(obj, tc); // 释放当前数据项的值
        PyObject_Free(npyarr); // 释放 NpyArrContext 结构体内存
    }
}

// 开始 NpyArrPassThru 迭代器操作，不执行任何操作
static void NpyArrPassThru_iterBegin(JSOBJ Py_UNUSED(obj),
                                     JSONTypeContext *Py_UNUSED(tc)) {}

// 结束 NpyArrPassThru 迭代器的操作
static void NpyArrPassThru_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    // 获取 JSON 上下文中的 NpyArrContext 结构体指针
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;
    
    // 完成当前维度的迭代，重置数据指针
    npyarr->curdim--;
    npyarr->dataptr -= npyarr->stride * npyarr->index[npyarr->stridedim];
    npyarr->stridedim -= npyarr->inc;

    // 如果 NpyArrContext 对应的数组不是 PyArrayObject 类型，则设置类型错误并返回
    if (!PyArray_Check(npyarr->array)) {
        PyErr_SetString(PyExc_TypeError,
                        "NpyArrayPassThru_iterEnd received a non-array object");
        return;
    }
    // 将数组对象转换为 PyArrayObject 类型
    const PyArrayObject *arrayobj = (const PyArrayObject *)npyarr->array;
    // 更新当前维度的尺寸和步幅
    npyarr->dim = PyArray_DIM(arrayobj, (int)npyarr->stridedim);
    npyarr->stride = PyArray_STRIDE(arrayobj, (int)npyarr->stridedim);
    npyarr->dataptr += npyarr->stride;

    // 释放当前数据项的值
    NpyArr_freeItemValue(obj, tc);
}

// 获取 NpyArr 迭代器中的下一个数据项
static int NpyArr_iterNextItem(JSOBJ obj, JSONTypeContext *tc) {
    // 获取 JSON 上下文中的 NpyArrContext 结构体指针
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;

    // 如果出现 Python 异常，返回 0 表示迭代结束
    if (PyErr_Occurred()) {
        return 0;
    }

    // 如果当前索引超出了当前维度的尺寸，返回 0 表示迭代结束
    if (npyarr->index[npyarr->stridedim] >= npyarr->dim) {
        return 0;
    }

    // 释放当前数据项的值
    NpyArr_freeItemValue(obj, tc);

    // 如果 NpyArrContext 对应的数组不是 PyArrayObject 类型，则设置类型错误并返回
    if (!PyArray_Check(npyarr->array)) {
        PyErr_SetString(PyExc_TypeError,
                        "NpyArr_iterNextItem received a non-array object");
        return 0;
    }
    // 将数组对象转换为 PyArrayObject 类型
    PyArrayObject *arrayobj = (PyArrayObject *)npyarr->array;

    // 如果数组对象是日期时间类型，则更新相应的编码器信息
    if (PyArray_ISDATETIME(arrayobj)) {
        GET_TC(tc)->itemValue = obj;
        Py_INCREF(obj);
        ((PyObjectEncoder *)tc->encoder)->npyType = PyArray_TYPE(arrayobj);
        // 同时写入 ndarray 的分辨率（单位）
        PyArray_Descr *dtype = PyArray_DESCR(arrayobj);
        ((PyObjectEncoder *)tc->encoder)->valueUnit =
            get_datetime_metadata_from_dtype(dtype).base;
        ((PyObjectEncoder *)tc->encoder)->npyValue = npyarr->dataptr;
        ((PyObjectEncoder *)tc->encoder)->npyCtxtPassthru = npyarr;
    } else {
        // 否则获取并设置当前数据项的值
        GET_TC(tc)->itemValue = PyArray_GETITEM(arrayobj, npyarr->dataptr);
    }

    // 更新数据指针并增加当前索引
    npyarr->dataptr += npyarr->stride;
    npyarr->index[npyarr->stridedim]++;
    return 1;
}

// 获取 NpyArr 迭代器的下一个数据项
static int NpyArr_iterNext(JSOBJ _obj, JSONTypeContext *tc) {
    // 获取 JSON 上下文中的 NpyArrContext 结构体指针
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;

    // 如果出现 Python 异常，返回 0 表示迭代结束
    if (PyErr_Occurred()) {
        return 0;
    }

    // 如果当前维度已经超过了总维度数或者当前索引超出了当前维度的尺寸，返回 0 表示迭代结束
    if (npyarr->curdim >= npyarr->ndim ||
        npyarr->index[npyarr->stridedim] >= npyarr->dim) {
        // 内部维度最深层，开始获取数据项的值
        GET_TC(tc)->iterNext = NpyArr_iterNextItem;
        return NpyArr_iterNextItem(_obj, tc);
    }

    // 深入下一维度
    npyarr->index[npyarr->stridedim]++;

    npyarr->curdim++;
    npyarr->stridedim += npyarr->inc;

    // 如果 NpyArrContext 对应的数组不是 PyArrayObject 类型，则设置类型错误并返回
    if (!PyArray_Check(npyarr->array)) {
        PyErr_SetString(PyExc_TypeError,
                        "NpyArr_iterNext received a non-array object");
        return 0;
    }
    // 返回整数0，表示函数执行失败
    return 0;
  }
  // 将void指针npyarr转换为PyArrayObject类型，并赋值给arrayobj
  const PyArrayObject *arrayobj = (const PyArrayObject *)npyarr->array;

  // 获取arrayobj在指定维度（npyarr->stridedim）上的维度大小，赋值给npyarr的dim成员变量
  npyarr->dim = PyArray_DIM(arrayobj, (int)npyarr->stridedim);
  // 获取arrayobj在指定维度（npyarr->stridedim）上的步长（stride），赋值给npyarr的stride成员变量
  npyarr->stride = PyArray_STRIDE(arrayobj, (int)npyarr->stridedim);
  // 将npyarr的索引数组中在npyarr->stridedim维度上的元素置为0
  npyarr->index[npyarr->stridedim] = 0;

  // 将npyarr赋值给tc->encoder强制转换为PyObjectEncoder类型的指针，并将npyarr赋值给npyCtxtPassthru成员变量
  ((PyObjectEncoder *)tc->encoder)->npyCtxtPassthru = npyarr;
  // 获取tc的itemValue成员变量，并赋值为npyarr的array成员变量
  GET_TC(tc)->itemValue = npyarr->array;
  // 返回整数1，表示函数执行成功
  return 1;
//=============================================================================
// Pandas block iteration functions
//
// Serialises a DataFrame column by column to avoid unnecessary data copies and
// more representative serialisation when dealing with mixed dtypes.
//
// Uses a dedicated NpyArrContext for each column.
//=============================================================================

// 结束块迭代时的回调函数，更新列索引以及释放资源
static void PdBlockPassThru_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    // 获取与 JSONTypeContext 相关联的 PdBlockContext 结构体
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;
    
    // 如果标记为转置，则增加列索引；否则重置为0
    if (blkCtxt->transpose) {
        blkCtxt->colIdx++;
    } else {
        blkCtxt->colIdx = 0;
    }
    
    // 释放当前项的资源
    NpyArr_freeItemValue(obj, tc);
}

// 获取下一个数据项的函数，利用块上下文中的 NpyArrContext 进行迭代
static int PdBlock_iterNextItem(JSOBJ obj, JSONTypeContext *tc) {
    // 获取与 JSONTypeContext 相关联的 PdBlockContext 结构体
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;
    
    // 如果列索引超出列数，返回0表示迭代结束
    if (blkCtxt->colIdx >= blkCtxt->ncols) {
        return 0;
    }
    
    // 获取当前列索引对应的 NpyArrContext，并将列索引增加
    GET_TC(tc)->npyarr = blkCtxt->npyCtxts[blkCtxt->colIdx];
    blkCtxt->colIdx++;
    
    // 调用 NpyArr_iterNextItem 继续迭代
    return NpyArr_iterNextItem(obj, tc);
}

// 获取当前迭代项的名称字符串，根据迭代策略选择列标签或行标签
static char *PdBlock_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                                 size_t *outLen) {
    // 获取与 JSONTypeContext 相关联的 PdBlockContext 结构体
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;
    // 获取当前列对应的 NpyArrContext
    NpyArrContext *npyarr = blkCtxt->npyCtxts[0];
    char *cStr;
    
    // 根据当前迭代策略确定返回的标签数组元素
    if (GET_TC(tc)->iterNext == PdBlock_iterNextItem) {
        // 如果是按项迭代，则使用列标签
        const npy_intp idx = blkCtxt->colIdx - 1;
        cStr = npyarr->columnLabels[idx];
    } else {
        // 否则根据迭代方式决定使用行标签的索引
        const npy_intp idx =
            GET_TC(tc)->iterNext != PdBlock_iterNext
                ? npyarr->index[npyarr->stridedim - npyarr->inc] - 1
                : npyarr->index[npyarr->stridedim];
        
        cStr = npyarr->rowLabels[idx];
    }
    
    // 设置返回字符串的长度
    *outLen = strlen(cStr);
    return cStr;
}

// 获取当前迭代项的名称字符串，在转置模式下使用不同的策略
static char *PdBlock_iterGetName_Transpose(JSOBJ Py_UNUSED(obj),
                                           JSONTypeContext *tc,
                                           size_t *outLen) {
    // 获取与 JSONTypeContext 相关联的 PdBlockContext 结构体
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;
    // 获取当前列对应的 NpyArrContext
    NpyArrContext *npyarr = blkCtxt->npyCtxts[blkCtxt->colIdx];
    char *cStr;
    
    // 根据当前迭代策略确定返回的标签数组元素
    if (GET_TC(tc)->iterNext == NpyArr_iterNextItem) {
        // 如果是按项迭代，则使用列标签
        const npy_intp idx = npyarr->index[npyarr->stridedim] - 1;
        cStr = npyarr->columnLabels[idx];
    } else {
        // 否则使用当前列索引作为行标签的索引
        const npy_intp idx = blkCtxt->colIdx;
        cStr = npyarr->rowLabels[idx];
    }
    
    // 设置返回字符串的长度
    *outLen = strlen(cStr);
    return cStr;
}
static int PdBlock_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    // 获取当前 JSON 对象的 PdBlock 上下文
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;

    // 检查是否有 Python 异常发生或者编码器中的错误消息已设置
    if (PyErr_Occurred() || ((JSONObjectEncoder *)tc->encoder)->errorMsg) {
        return 0;  // 如果有异常或错误消息，返回 0 表示迭代结束
    }

    // 如果 PdBlock 上下文中需要转置
    if (blkCtxt->transpose) {
        // 如果当前列索引超过了列数，返回 0 表示迭代结束
        if (blkCtxt->colIdx >= blkCtxt->ncols) {
            return 0;
        }
    } else {
        // 否则，获取第一个 NpyArrContext 上下文
        const NpyArrContext *npyarr = blkCtxt->npyCtxts[0];
        // 如果当前维度的索引超过了维度数，返回 0 表示迭代结束
        if (npyarr->index[npyarr->stridedim] >= npyarr->dim) {
            return 0;
        }
    }

    // 将当前 PdBlock 上下文传递给编码器的 blkCtxtPassthru 属性
    ((PyObjectEncoder *)tc->encoder)->blkCtxtPassthru = blkCtxt;
    // 将当前对象设置为 itemValue
    GET_TC(tc)->itemValue = obj;

    return 1;  // 返回 1 表示迭代可以继续
}

static void PdBlockPassThru_iterBegin(JSOBJ Py_UNUSED(obj),
                                      JSONTypeContext *tc) {
    // 获取当前 JSON 对象的 PdBlock 上下文
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;

    // 如果 PdBlock 上下文中需要转置
    if (blkCtxt->transpose) {
        // 如果转置，设置 iterNext 和 iterGetName 为相应的函数
        GET_TC(tc)->iterNext = NpyArr_iterNextItem;
        GET_TC(tc)->iterGetName = PdBlock_iterGetName_Transpose;
        // 设置 npyarr 为当前列的 NpyArrContext 上下文
        GET_TC(tc)->npyarr = blkCtxt->npyCtxts[blkCtxt->colIdx];
    }
}

static void PdBlock_iterBegin(JSOBJ _obj, JSONTypeContext *tc) {
    // 将 JSOBJ 转换为 PyObject 类型
    PyObject *obj = (PyObject *)_obj;

    // 根据转置属性设置 iterGetName 为相应的获取名称函数
    GET_TC(tc)->iterGetName = GET_TC(tc)->transpose
                                ? PdBlock_iterGetName_Transpose
                                : PdBlock_iterGetName;

    // 分配内存以存储 PdBlock 上下文
    PdBlockContext *blkCtxt = PyObject_Malloc(sizeof(PdBlockContext));
    if (!blkCtxt) {
        PyErr_NoMemory();
        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }
    GET_TC(tc)->pdblock = blkCtxt;

    // 初始化 PdBlock 上下文的属性
    blkCtxt->colIdx = 0;
    blkCtxt->transpose = GET_TC(tc)->transpose;
    blkCtxt->ncols = get_attr_length(obj, "columns");

    // 如果列数为 0，则设置为无效的迭代状态并返回
    if (blkCtxt->ncols == 0) {
        blkCtxt->npyCtxts = NULL;
        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }

    // 分配内存以存储 NpyArrContext 上下文数组
    blkCtxt->npyCtxts = PyObject_Malloc(sizeof(NpyArrContext *) * blkCtxt->ncols);
    if (!blkCtxt->npyCtxts) {
        PyErr_NoMemory();
        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }

    // 获取列数组，并检查是否成功获取
    PyObject *arrays = get_sub_attr(obj, "_mgr", "column_arrays");
    if (!arrays) {
        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }

    // 遍历列数组
    for (Py_ssize_t i = 0; i < PyObject_Length(arrays); i++) {
        PyObject *array = PyList_GET_ITEM(arrays, i);
        if (!array) {
            GET_TC(tc)->iterNext = NpyArr_iterNextNone;
            goto ARR_RET;
        }

        // 确保获得 numpy 数组（使用 np.asarray）
        PyObject *values = PyObject_CallMethod(array, "__array__", NULL);
        if ((!values) || (!PyArray_CheckExact(values))) {
            // 如果未获得 numpy 数组，则设置错误消息并返回无效的迭代状态
            ((JSONObjectEncoder *)tc->encoder)->errorMsg = "";
            GET_TC(tc)->iterNext = NpyArr_iterNextNone;
            goto ARR_RET;
        }

        GET_TC(tc)->newObj = values;

        // 初始化此列的专用上下文
        NpyArr_iterBegin(obj, tc);

        // 将当前 itemValue 和 npyCtxtPassthru 设置为 NULL
        GET_TC(tc)->itemValue = NULL;
        ((PyObjectEncoder *)tc->encoder)->npyCtxtPassthru = NULL;

        // 将当前 NpyArrContext 上下文存储到数组中
        blkCtxt->npyCtxts[i] = GET_TC(tc)->npyarr;
        GET_TC(tc)->newObj = NULL;
    }
    GET_TC(tc)->npyarr = blkCtxt->npyCtxts[0];
    goto ARR_RET;

ARR_RET:
    // 处理列数组结束时的清理工作
    ;
}
//=============================================================================
// 结束处理数组返回的函数。释放所有数组对象并清理上下文。
//=============================================================================
ARR_RET:
  Py_DECREF(arrays);
}

//=============================================================================
// 结束处理 Pandas 数据块的迭代。释放与迭代相关的资源和上下文。
//=============================================================================
static void PdBlock_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
  // 清空当前项值
  GET_TC(tc)->itemValue = NULL;
  
  // 获取 NpyArrContext 和 PdBlockContext 上下文
  NpyArrContext *npyarr = GET_TC(tc)->npyarr;
  PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;

  // 如果存在 Pandas 数据块上下文
  if (blkCtxt) {
    // 释放每个列对应的 NpyArrContext 资源
    for (int i = 0; i < blkCtxt->ncols; i++) {
      npyarr = blkCtxt->npyCtxts[i];
      if (npyarr) {
        if (npyarr->array) {
          Py_DECREF(npyarr->array); // 释放数组对象
          npyarr->array = NULL;
        }
        
        GET_TC(tc)->npyarr = npyarr;
        NpyArr_iterEnd(obj, tc); // 调用 NpyArr_iterEnd 清理迭代资源

        blkCtxt->npyCtxts[i] = NULL; // 清空当前列的 NpyArrContext 引用
      }
    }

    // 释放 Pandas 数据块的 NpyArrContext 数组
    if (blkCtxt->npyCtxts) {
      PyObject_Free(blkCtxt->npyCtxts);
    }

    // 释放 Pandas 数据块上下文
    PyObject_Free(blkCtxt);
  }
}

//=============================================================================
// 元组迭代函数
// itemValue 是借用引用，不进行引用计数
//=============================================================================
static void Tuple_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
  // 初始化元组迭代器状态
  GET_TC(tc)->index = 0;
  GET_TC(tc)->size = PyTuple_GET_SIZE((PyObject *)obj);
  GET_TC(tc)->itemValue = NULL;
}

static int Tuple_iterNext(JSOBJ obj, JSONTypeContext *tc) {
  // 检查是否超出元组大小
  if (GET_TC(tc)->index >= GET_TC(tc)->size) {
    return 0; // 迭代结束
  }

  // 获取当前元组的项
  PyObject *item = PyTuple_GET_ITEM(obj, GET_TC(tc)->index);

  // 更新迭代器状态
  GET_TC(tc)->itemValue = item;
  GET_TC(tc)->index++;
  return 1; // 迭代未结束
}

static void Tuple_iterEnd(JSOBJ Py_UNUSED(obj),
                          JSONTypeContext *Py_UNUSED(tc)) {
  // 空实现，元组迭代结束时无需清理资源
}

static JSOBJ Tuple_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 返回当前元组迭代的项值
  return GET_TC(tc)->itemValue;
}

static char *Tuple_iterGetName(JSOBJ Py_UNUSED(obj),
                               JSONTypeContext *Py_UNUSED(tc),
                               size_t *Py_UNUSED(outLen)) {
  // 返回空，元组没有命名项
  return NULL;
}

//=============================================================================
// 集合迭代函数
// itemValue 是借用引用，不进行引用计数
//=============================================================================
static void Set_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
  // 初始化集合迭代器状态
  GET_TC(tc)->itemValue = NULL;
  GET_TC(tc)->iterator = PyObject_GetIter(obj); // 获取集合的迭代器
}

static int Set_iterNext(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 清理上一次迭代的项值
  if (GET_TC(tc)->itemValue) {
    Py_DECREF(GET_TC(tc)->itemValue);
    GET_TC(tc)->itemValue = NULL;
  }

  // 获取集合的下一项
  PyObject *item = PyIter_Next(GET_TC(tc)->iterator);

  // 检查是否迭代结束
  if (item == NULL) {
    return 0; // 迭代结束
  }

  // 更新迭代器状态
  GET_TC(tc)->itemValue = item;
  return 1; // 迭代未结束
}

static void Set_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 清理最后一个迭代的项值
  if (GET_TC(tc)->itemValue) {
    Py_DECREF(GET_TC(tc)->itemValue);
    GET_TC(tc)->itemValue = NULL;
  }

  // 清理集合迭代器
  if (GET_TC(tc)->iterator) {
    Py_DECREF(GET_TC(tc)->iterator);
    GET_TC(tc)->iterator = NULL;
  }
}

static JSOBJ Set_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 返回当前集合迭代的项值
  return GET_TC(tc)->itemValue;
}
//=============================================================================
// Dir iteration functions
// itemName ref is borrowed from PyObject_Dir (attrList). No refcount
// itemValue ref is from PyObject_GetAttr. Ref counted
//=============================================================================

// 开始目录迭代，初始化迭代器状态
static void Dir_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
    // 获取对象的属性列表
    GET_TC(tc)->attrList = PyObject_Dir(obj);
    // 初始化迭代器的索引和大小
    GET_TC(tc)->index = 0;
    GET_TC(tc)->size = PyList_GET_SIZE(GET_TC(tc)->attrList);
}

// 结束目录迭代，清理迭代器状态
static void Dir_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    // 清理项目值的引用计数
    if (GET_TC(tc)->itemValue) {
        Py_DECREF(GET_TC(tc)->itemValue);
        GET_TC(tc)->itemValue = NULL;
    }
    // 清理项目名称的引用计数
    if (GET_TC(tc)->itemName) {
        Py_DECREF(GET_TC(tc)->itemName);
        GET_TC(tc)->itemName = NULL;
    }
    // 清理属性列表的引用计数
    Py_DECREF((PyObject *)GET_TC(tc)->attrList);
}

// 获取目录迭代的下一个项目
static int Dir_iterNext(JSOBJ _obj, JSONTypeContext *tc) {
    // 将 JSOBJ 类型转换为 PyObject 类型
    PyObject *obj = (PyObject *)_obj;
    PyObject *itemValue = GET_TC(tc)->itemValue;
    PyObject *itemName = GET_TC(tc)->itemName;

    // 检查是否有异常或错误信息
    if (PyErr_Occurred() || ((JSONObjectEncoder *)tc->encoder)->errorMsg) {
        return 0;
    }

    // 清理上一个项目值的引用计数
    if (itemValue) {
        Py_DECREF(GET_TC(tc)->itemValue);
        GET_TC(tc)->itemValue = itemValue = NULL;
    }

    // 清理上一个项目名称的引用计数
    if (itemName) {
        Py_DECREF(GET_TC(tc)->itemName);
        GET_TC(tc)->itemName = itemName = NULL;
    }

    // 遍历属性列表，获取下一个有效的属性名和其对应的值
    for (; GET_TC(tc)->index < GET_TC(tc)->size; GET_TC(tc)->index++) {
        PyObject *attrName = PyList_GET_ITEM(GET_TC(tc)->attrList, GET_TC(tc)->index);
        PyObject *attr = PyUnicode_AsUTF8String(attrName);
        const char *attrStr = PyBytes_AS_STRING(attr);

        // 忽略以 '_' 开头的属性
        if (attrStr[0] == '_') {
            Py_DECREF(attr);
            continue;
        }

        // 获取对象的属性值
        itemValue = PyObject_GetAttr(obj, attrName);
        if (itemValue == NULL) {
            PyErr_Clear();
            Py_DECREF(attr);
            continue;
        }

        // 如果属性值是可调用的对象，清理其引用计数
        if (PyCallable_Check(itemValue)) {
            Py_DECREF(itemValue);
            Py_DECREF(attr);
            continue;
        }

        // 保存当前属性名称和值到迭代器上下文中
        GET_TC(tc)->itemName = itemName;
        GET_TC(tc)->itemValue = itemValue;

        // 更新当前属性名称，用于返回
        itemName = attr;
        break;
    }

    // 如果未找到有效的属性名，重置迭代状态并返回结束标志
    if (itemName == NULL) {
        GET_TC(tc)->index = GET_TC(tc)->size;
        GET_TC(tc)->itemValue = NULL;
        return 0;
    }

    // 更新迭代器上下文中的项目名称和项目值
    GET_TC(tc)->itemName = itemName;
    GET_TC(tc)->itemValue = itemValue;
    GET_TC(tc)->index++;

    return 1;
}

// 获取目录迭代当前项目的值
static JSOBJ Dir_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

// 获取目录迭代当前项目的名称
static char *Dir_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                             size_t *outLen) {
    // 返回项目名称的字节表示和长度
    *outLen = PyBytes_GET_SIZE(GET_TC(tc)->itemName);
    return PyBytes_AS_STRING(GET_TC(tc)->itemName);
}

//=============================================================================
// List iteration functions
// itemValue is borrowed from object (which is list). No refcounting
//=============================================================================
// 初始化列表迭代器，设置迭代器上下文的起始索引和列表长度
static void List_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
  GET_TC(tc)->index = 0; // 将迭代器上下文中的索引设置为0
  GET_TC(tc)->size = PyList_GET_SIZE((PyObject *)obj); // 获取列表对象的长度并存储在迭代器上下文中
}

// 获取列表的下一个元素，并更新迭代器上下文中的索引
static int List_iterNext(JSOBJ obj, JSONTypeContext *tc) {
  if (GET_TC(tc)->index >= GET_TC(tc)->size) { // 如果索引超过列表长度，返回0表示迭代结束
    return 0;
  }

  GET_TC(tc)->itemValue = PyList_GET_ITEM(obj, GET_TC(tc)->index); // 获取列表中索引位置的元素，并存储在迭代器上下文中
  GET_TC(tc)->index++; // 更新迭代器上下文中的索引
  return 1; // 返回1表示还有更多元素需要迭代
}

// 结束列表迭代，无需执行任何操作
static void List_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *Py_UNUSED(tc)) {
}

// 获取当前列表迭代器的值
static JSOBJ List_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  return GET_TC(tc)->itemValue; // 返回迭代器上下文中存储的当前元素值
}

// 获取当前列表迭代器的名称，返回空指针表示不需要元素名称
static char *List_iterGetName(JSOBJ Py_UNUSED(obj),
                              JSONTypeContext *Py_UNUSED(tc),
                              size_t *Py_UNUSED(outLen)) {
  return NULL; // 返回空指针表示不支持获取元素名称
}

//=============================================================================
// pandas Index 迭代函数
//=============================================================================
// 初始化 Index 对象的迭代器，设置起始索引并分配存储名称的内存空间
static void Index_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  GET_TC(tc)->index = 0; // 将迭代器上下文中的索引设置为0
  GET_TC(tc)->cStr = PyObject_Malloc(20 * sizeof(char)); // 分配20字节的内存空间用于存储名称
  if (!GET_TC(tc)->cStr) { // 如果内存分配失败
    PyErr_NoMemory(); // 抛出内存错误异常
  }
}

// 获取 Index 对象的下一个元素，并更新迭代器上下文中的索引
static int Index_iterNext(JSOBJ obj, JSONTypeContext *tc) {
  if (!GET_TC(tc)->cStr) { // 如果名称存储空间未成功分配，返回0表示迭代结束
    return 0;
  }

  const Py_ssize_t index = GET_TC(tc)->index; // 获取当前索引
  Py_XDECREF(GET_TC(tc)->itemValue); // 释放之前的元素值引用
  if (index == 0) { // 对于索引为0的情况
    memcpy(GET_TC(tc)->cStr, "name", sizeof(char) * 5); // 将字符串"name"复制到名称存储空间中
    GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "name"); // 获取对象的"name"属性并存储在迭代器上下文中
  } else if (index == 1) { // 对于索引为1的情况
    memcpy(GET_TC(tc)->cStr, "data", sizeof(char) * 5); // 将字符串"data"复制到名称存储空间中
    GET_TC(tc)->itemValue = get_values(obj); // 调用函数获取对象的值并存储在迭代器上下文中
    if (!GET_TC(tc)->itemValue) { // 如果获取值失败
      return 0; // 返回0表示迭代结束
    }
  } else { // 对于其他索引值
    return 0; // 返回0表示迭代结束
  }

  GET_TC(tc)->index++; // 更新迭代器上下文中的索引
  return 1; // 返回1表示还有更多元素需要迭代
}

// 结束 Index 迭代，无需执行任何操作
static void Index_iterEnd(JSOBJ Py_UNUSED(obj),
                          JSONTypeContext *Py_UNUSED(tc)) {}

// 获取当前 Index 迭代器的值
static JSOBJ Index_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  return GET_TC(tc)->itemValue; // 返回迭代器上下文中存储的当前元素值
}

// 获取当前 Index 迭代器的名称，并设置名称的长度
static char *Index_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                               size_t *outLen) {
  *outLen = strlen(GET_TC(tc)->cStr); // 获取名称字符串的长度
  return GET_TC(tc)->cStr; // 返回存储名称的内存空间
}

//=============================================================================
// pandas Series 迭代函数
//=============================================================================
// 初始化 Series 对象的迭代器，设置起始索引并分配存储名称的内存空间
static void Series_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder; // 获取编码器对象
  GET_TC(tc)->index = 0; // 将迭代器上下文中的索引设置为0
  GET_TC(tc)->cStr = PyObject_Malloc(20 * sizeof(char)); // 分配20字节的内存空间用于存储名称
  enc->outputFormat = VALUES; // 设置输出格式为VALUES，适用于包含的 Series 对象
  if (!GET_TC(tc)->cStr) { // 如果内存分配失败
    PyErr_NoMemory(); // 抛出内存错误异常
  }
}

// 获取 Series 对象的下一个元素，并更新迭代器上下文中的索引
static int Series_iterNext(JSOBJ obj, JSONTypeContext *tc) {
  if (!GET_TC(tc)->cStr) { // 如果名称存储空间未成功分配，返回0表示迭代结束
    return 0;
  }

  const Py_ssize_t index = GET_TC(tc)->index; // 获取当前索引
  Py_XDECREF(GET_TC(tc)->itemValue); // 释放之前的元素值引用
  if (index == 0) { // 对于索引为0的情况
    // 将字符串"name"复制到名称存储空间中
    memcpy(GET_TC(tc)->cStr, "name", sizeof(char) * 5);
    // 获取对象的"name"属性并存储在迭代器上下文中
    GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "name");
  } else { // 对于其他索引值
    return 0; // 返回0表示迭代结束
  }

  GET_TC(tc)->index++; // 更新迭代器上下文中的索引
  return 1; // 返回1表示还有更多元素需要迭代
}
    # 根据 index 的不同值执行不同的操作
    if (index == 0) {
        # 将字符串 "name" 复制到 GET_TC(tc)->cStr 中，长度为 5 个字符
        memcpy(GET_TC(tc)->cStr, "name", sizeof(char) * 5);
        # 获取 Python 对象 obj 的属性 "name" 并赋给 GET_TC(tc)->itemValue
        GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "name");
    } else if (index == 1) {
        # 将字符串 "index" 复制到 GET_TC(tc)->cStr 中，长度为 6 个字符
        memcpy(GET_TC(tc)->cStr, "index", sizeof(char) * 6);
        # 获取 Python 对象 obj 的属性 "index" 并赋给 GET_TC(tc)->itemValue
        GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "index");
    } else if (index == 2) {
        # 将字符串 "data" 复制到 GET_TC(tc)->cStr 中，长度为 5 个字符
        memcpy(GET_TC(tc)->cStr, "data", sizeof(char) * 5);
        # 调用函数 get_values(obj) 获取数据并赋给 GET_TC(tc)->itemValue
        GET_TC(tc)->itemValue = get_values(obj);
        # 如果获取数据失败，则返回 0
        if (!GET_TC(tc)->itemValue) {
            return 0;
        }
    } else {
        # 如果 index 不为 0、1、2，则直接返回 0
        return 0;
    }
    
    # 增加 GET_TC(tc)->index 的值
    GET_TC(tc)->index++;
    # 操作成功，返回 1
    return 1;
//=============================================================================
// Series iteration functions
//=============================================================================

static void Series_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 获取 PyObjectEncoder 结构体中的 encoder 成员，并将其类型转换为 PyObjectEncoder 指针
  PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder;
  // 将 outputFormat 属性恢复为 originalOutputFormat 的值
  enc->outputFormat = enc->originalOutputFormat;
}

static JSOBJ Series_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 返回 JSONTypeContext 结构体中的 itemValue 成员
  return GET_TC(tc)->itemValue;
}

static char *Series_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                                size_t *outLen) {
  // 设置 outLen 指针所指向的值为 GET_TC(tc)->cStr 字符串的长度
  *outLen = strlen(GET_TC(tc)->cStr);
  // 返回 JSONTypeContext 结构体中的 cStr 成员
  return GET_TC(tc)->cStr;
}

//=============================================================================
// pandas DataFrame iteration functions
//=============================================================================
static void DataFrame_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 获取 PyObjectEncoder 结构体中的 encoder 成员，并将其类型转换为 PyObjectEncoder 指针
  PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder;
  // 初始化 JSONTypeContext 结构体中的 index 成员为 0
  GET_TC(tc)->index = 0;
  // 分配内存并将其赋值给 JSONTypeContext 结构体中的 cStr 成员，长度为 20 个字符
  GET_TC(tc)->cStr = PyObject_Malloc(20 * sizeof(char));
  // 将 encoder 结构体中的 outputFormat 属性设置为 VALUES，用于包含的系列和索引
  enc->outputFormat = VALUES;
  // 如果内存分配失败，则抛出内存错误异常
  if (!GET_TC(tc)->cStr) {
    PyErr_NoMemory();
  }
}

static int DataFrame_iterNext(JSOBJ obj, JSONTypeContext *tc) {
  // 如果 cStr 成员为空指针，则返回 0
  if (!GET_TC(tc)->cStr) {
    return 0;
  }

  // 获取 JSONTypeContext 结构体中的 index 成员
  const Py_ssize_t index = GET_TC(tc)->index;
  // 释放 JSONTypeContext 结构体中的 itemValue 成员的引用计数
  Py_XDECREF(GET_TC(tc)->itemValue);

  // 根据 index 的值执行相应的操作
  if (index == 0) {
    // 将字符串 "columns" 复制到 JSONTypeContext 结构体中的 cStr 成员
    memcpy(GET_TC(tc)->cStr, "columns", sizeof(char) * 8);
    // 获取对象 obj 中名为 "columns" 的属性，并将其赋值给 JSONTypeContext 结构体中的 itemValue 成员
    GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "columns");
  } else if (index == 1) {
    // 将字符串 "index" 复制到 JSONTypeContext 结构体中的 cStr 成员
    memcpy(GET_TC(tc)->cStr, "index", sizeof(char) * 6);
    // 获取对象 obj 中名为 "index" 的属性，并将其赋值给 JSONTypeContext 结构体中的 itemValue 成员
    GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "index");
  } else if (index == 2) {
    // 将字符串 "data" 复制到 JSONTypeContext 结构体中的 cStr 成员
    memcpy(GET_TC(tc)->cStr, "data", sizeof(char) * 5);
    // 增加对象 obj 的引用计数，并将其赋值给 JSONTypeContext 结构体中的 itemValue 成员
    Py_INCREF(obj);
    GET_TC(tc)->itemValue = obj;
  } else {
    // 如果 index 不是 0、1 或 2，则返回 0
    return 0;
  }

  // 将 JSONTypeContext 结构体中的 index 成员加一
  GET_TC(tc)->index++;
  // 返回 1 表示迭代成功
  return 1;
}

static void DataFrame_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 获取 PyObjectEncoder 结构体中的 encoder 成员，并将其类型转换为 PyObjectEncoder 指针
  PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder;
  // 将 outputFormat 属性恢复为 originalOutputFormat 的值
  enc->outputFormat = enc->originalOutputFormat;
}

static JSOBJ DataFrame_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 返回 JSONTypeContext 结构体中的 itemValue 成员
  return GET_TC(tc)->itemValue;
}

static char *DataFrame_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                                   size_t *outLen) {
  // 设置 outLen 指针所指向的值为 GET_TC(tc)->cStr 字符串的长度
  *outLen = strlen(GET_TC(tc)->cStr);
  // 返回 JSONTypeContext 结构体中的 cStr 成员
  return GET_TC(tc)->cStr;
}

//=============================================================================
// Dict iteration functions
// itemName might converted to string (Python_Str). Do refCounting
// itemValue is borrowed from object (which is dict). No refCounting
//=============================================================================
static void Dict_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 初始化 JSONTypeContext 结构体中的 index 成员为 0
  GET_TC(tc)->index = 0;
}

static int Dict_iterNext(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 如果 itemName 成员不为空，则减少其引用计数并将其置为空指针
  if (GET_TC(tc)->itemName) {
    Py_DECREF(GET_TC(tc)->itemName);
    GET_TC(tc)->itemName = NULL;
  }

  // 使用 PyDict_Next 函数获取字典对象 GET_TC(tc)->dictObj 中的下一个键值对，并将结果存储在相应的成员中
  if (!PyDict_Next((PyObject *)GET_TC(tc)->dictObj, &GET_TC(tc)->index,
                   &GET_TC(tc)->itemName, &GET_TC(tc)->itemValue)) {
    // 返回整数 0，表示结束函数执行
    return 0;
  }

  // 检查 itemName 是否为 PyUnicode 对象
  if (PyUnicode_Check(GET_TC(tc)->itemName)) {
    // 将 PyUnicode 对象转换为 UTF-8 字符串对象，并更新 itemName
    GET_TC(tc)->itemName = PyUnicode_AsUTF8String(GET_TC(tc)->itemName);
  } else if (!PyBytes_Check(GET_TC(tc)->itemName)) {
    // 如果 itemName 不是 PyBytes 对象，将其转换为字符串对象
    GET_TC(tc)->itemName = PyObject_Str(GET_TC(tc)->itemName);
    // 临时保存当前 itemName 的引用
    PyObject *itemNameTmp = GET_TC(tc)->itemName;
    // 将转换后的字符串对象再次转换为 UTF-8 字符串对象，并更新 itemName
    GET_TC(tc)->itemName = PyUnicode_AsUTF8String(GET_TC(tc)->itemName);
    // 释放临时保存的 itemName 引用
    Py_DECREF(itemNameTmp);
  } else {
    // 如果 itemName 是 PyBytes 对象，增加其引用计数
    Py_INCREF(GET_TC(tc)->itemName);
  }
  // 返回整数 1，表示函数执行成功
  return 1;
}

static void Dict_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 如果存在当前项名称，释放其引用并置空
  if (GET_TC(tc)->itemName) {
    Py_DECREF(GET_TC(tc)->itemName);
    GET_TC(tc)->itemName = NULL;
  }
  // 释放字典对象的引用
  Py_DECREF(GET_TC(tc)->dictObj);
}

static JSOBJ Dict_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
  // 返回迭代器当前项的值
  return GET_TC(tc)->itemValue;
}

static char *Dict_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                              size_t *outLen) {
  // 获取迭代器当前项的名称，并设置名称长度
  *outLen = PyBytes_GET_SIZE(GET_TC(tc)->itemName);
  return PyBytes_AS_STRING(GET_TC(tc)->itemName);
}

static void NpyArr_freeLabels(char **labels, npy_intp len) {
  // 如果标签数组存在，释放每个标签字符串及数组本身的内存
  if (labels) {
    for (npy_intp i = 0; i < len; i++) {
      PyObject_Free(labels[i]);
    }
    PyObject_Free(labels);
  }
}

/*
 * Function: NpyArr_encodeLabels
 * -----------------------------
 *
 * Builds an array of "encoded" labels.
 *
 * labels: PyArrayObject pointer for labels to be "encoded"
 * num : number of labels
 *
 * "encode" is quoted above because we aren't really doing encoding
 * For historical reasons this function would actually encode the entire
 * array into a separate buffer with a separate call to JSON_Encode
 * and would leave it to complex pointer manipulation from there to
 * unpack values as needed. To make things simpler and more idiomatic
 * this has instead just stringified any input save for datetime values,
 * which may need to be represented in various formats.
 */
static char **NpyArr_encodeLabels(PyArrayObject *labels, PyObjectEncoder *enc,
                                  npy_intp num) {
  // 注意：此函数会偷取 labels 的引用。

  PyObject *item = NULL;
  const NPY_DATETIMEUNIT base = enc->datetimeUnit;

  // 如果 labels 为空，返回空指针
  if (!labels) {
    return 0;
  }

  // 如果 labels 的大小小于 num，抛出值错误并释放 labels 的引用
  if (PyArray_SIZE(labels) < num) {
    PyErr_SetString(PyExc_ValueError,
                    "Label array sizes do not match corresponding data shape");
    Py_DECREF(labels);
    return 0;
  }

  // 分配存储指针的数组内存空间
  char **ret = PyObject_Malloc(sizeof(char *) * num);
  if (!ret) {
    PyErr_NoMemory();
    Py_DECREF(labels);
    return 0;
  }

  // 初始化返回数组的每个元素为空指针
  for (npy_intp i = 0; i < num; i++) {
    ret[i] = NULL;
  }

  // 获取 labels 的步幅和数据指针
  const npy_intp stride = PyArray_STRIDE(labels, 0);
  char *dataptr = PyArray_DATA(labels);
  const int type_num = PyArray_TYPE(labels);
  PyArray_Descr *dtype = PyArray_DESCR(labels);

  // 遍历 labels 的每个标签
  for (npy_intp i = 0; i < num; i++) {
    // 获取当前标签的对象
    item = PyArray_GETITEM(labels, dataptr);
    if (!item) {
      // 如果获取失败，释放 ret 数组的内存并置为 0
      NpyArr_freeLabels(ret, num);
      ret = 0;
      break;
    }

    // 检查是否是日期时间类型
    int is_datetimelike = 0;
    int64_t i8date;
    NPY_DATETIMEUNIT dateUnit = NPY_FR_ns;
    if (PyTypeNum_ISDATETIME(type_num)) {
      is_datetimelike = 1;
      i8date = *(int64_t *)dataptr;
      dateUnit = get_datetime_metadata_from_dtype(dtype).base;
      // 省略部分代码注释，具体功能见原文档
    } else if (PyDate_Check(item) || PyDelta_Check(item)) {
      // 检查 item 是否为日期或时间间隔对象
      is_datetimelike = 1;
      if (PyObject_HasAttrString(item, "_value")) {
        // 如果 item 是 pd.Timestamp 对象或 pd.NaT（不可用时间），从 _value 属性获取长整型日期
        i8date = get_long_attr(item, "_value");
      } else {
        if (PyDelta_Check(item)) {
          // 如果 item 是时间间隔对象，则计算总秒数并转换为纳秒
          // 注意：如果 total_seconds 返回值超出纳秒数的有效位数，会丢失精度，也可能溢出
          i8date = (int64_t)(total_seconds(item) *
                             1000000000LL); // 每秒的纳秒数
        } else {
          // 对于 datetime.* 对象，使用 PyDateTimeToEpoch 将其转换为纳秒级的 Epoch 时间
          i8date = PyDateTimeToEpoch(item, NPY_FR_ns);
        }
      }
    }

    size_t len;
    char *cLabel;
    if (is_datetimelike) {
      if (i8date == get_nat()) {
        // 如果 i8date 等于 NaT（不可用时间）的整数表示，将 cLabel 设置为 "null"
        len = 4;
        cLabel = PyObject_Malloc(len + 1);
        strncpy(cLabel, "null", len + 1);
      } else {
        if (enc->datetimeIso) {
          if ((type_num == NPY_TIMEDELTA) || (PyDelta_Check(item))) {
            // 如果 enc 中指定了 datetimeIso 并且 item 是时间间隔对象，则将 i8date 转换为 ISO 8601 格式的时间间隔
            // TODO(username): 是否支持非纳秒级时间间隔？
            cLabel = int64ToIsoDuration(i8date, &len);
          } else {
            if (type_num == NPY_DATETIME) {
              // 如果 type_num 指定为 NPY_DATETIME，则将 i8date 转换为指定单位的 ISO 8601 格式时间
              cLabel = int64ToIso(i8date, dateUnit, base, &len);
            } else {
              // 否则，将 item 转换为 ISO 8601 格式的字符串
              cLabel = PyDateTimeToIso(item, base, &len);
            }
          }
          if (cLabel == NULL) {
            // 如果转换失败，释放资源并设置 ret 为 0
            Py_DECREF(item);
            NpyArr_freeLabels(ret, num);
            ret = 0;
            break;
          }
        } else {
          // 如果不使用 datetimeIso 标志，将 i8date 格式化为整数字符串表示
          int size_of_cLabel = 21; // 21 个字符用于 int64
          cLabel = PyObject_Malloc(size_of_cLabel);
          if (scaleNanosecToUnit(&i8date, base) == -1) {
            // 如果转换出错，释放资源并设置 ret 为 0
            NpyArr_freeLabels(ret, num);
            ret = 0;
            break;
          }
          snprintf(cLabel, size_of_cLabel, "%" PRId64, i8date);
          len = strlen(cLabel);
        }
      }
    } else { // Fallback to string representation
      // 否则，使用字符串表示 item 并替换 item 以保持其存活状态
      Py_SETREF(item, PyObject_Str(item));
      if (item == NULL) {
        // 如果转换失败，释放资源并设置 ret 为 0
        NpyArr_freeLabels(ret, num);
        ret = 0;
        break;
      }

      // 获取 item 的 UTF-8 字符串表示并计算长度
      cLabel = (char *)PyUnicode_AsUTF8(item);
      len = strlen(cLabel);
    }

    // 分配足够空间存储 cLabel 的长度加上 NULL 终止符
    ret[i] = PyObject_Malloc(len + 1);
    memcpy(ret[i], cLabel, len + 1);
    Py_DECREF(item);

    if (is_datetimelike) {
      // 如果是日期时间对象，释放 cLabel 分配的内存空间
      PyObject_Free(cLabel);
    }

    if (PyErr_Occurred()) {
      // 如果发生错误，释放资源并设置 ret 为 0
      NpyArr_freeLabels(ret, num);
      ret = 0;
      break;
    }

    if (!ret[i]) {
      // 如果分配内存失败，设置错误状态并设置 ret 为 0
      PyErr_NoMemory();
      ret = 0;
      break;
    }

    // 更新 dataptr 指针位置，继续处理下一个元素
    dataptr += stride;
  }

  // 释放 labels 的引用计数，并返回 ret 数组
  Py_DECREF(labels);
  return ret;
static void Object_invokeDefaultHandler(PyObject *obj, PyObjectEncoder *enc) {
    // 声明并初始化临时对象指针为NULL
    PyObject *tmpObj = NULL;
    // 调用enc对象的defaultHandler函数，传入obj作为参数，返回结果给tmpObj
    tmpObj = PyObject_CallFunctionObjArgs(enc->defaultHandler, obj, NULL);
    // 检查是否有异常发生
    if (!PyErr_Occurred()) {
        // 如果tmpObj为NULL，设置类型错误异常字符串
        if (tmpObj == NULL) {
            PyErr_SetString(PyExc_TypeError, "Failed to execute default handler");
        } else {
            // 否则，将tmpObj编码为JSON对象，使用enc对象
            encode(tmpObj, (JSONObjectEncoder *)enc, NULL, 0);
        }
    }
    // 释放临时对象的引用
    Py_XDECREF(tmpObj);
    return;
}

static void Object_beginTypeContext(JSOBJ _obj, JSONTypeContext *tc) {
    // 初始化TypeContext结构体的prv成员为NULL
    tc->prv = NULL;

    // 如果_obj为空
    if (!_obj) {
        // 将type设置为无效类型，并返回
        tc->type = JT_INVALID;
        return;
    }

    // 将_jsobj类型转换为PyObject类型的obj
    PyObject *obj = (PyObject *)_obj;
    // 将tc->encoder强制转换为PyObjectEncoder类型的enc
    PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder;

    // 如果obj是Python中的布尔类型
    if (PyBool_Check(obj)) {
        // 根据obj的值设置type为JT_TRUE或JT_FALSE
        tc->type = (obj == Py_True) ? JT_TRUE : JT_FALSE;
        return;
    } else if (obj == Py_None) {
        // 如果obj是Python中的None对象，设置type为JT_NULL，并返回
        tc->type = JT_NULL;
        return;
    }

    // 创建一个新的TypeContext结构体pc
    TypeContext *pc = createTypeContext();
    // 如果创建失败，设置type为JT_INVALID，并返回
    if (!pc) {
        tc->type = JT_INVALID;
        return;
    }
    // 将pc赋值给tc的prv成员，表示它的前一个上下文
    tc->prv = pc;

    // 如果enc的npyType是日期时间类型
    if (PyTypeNum_ISDATETIME(enc->npyType)) {
        // 获取enc的npyValue所指向的npy_int64类型的值
        int64_t longVal = *(npy_int64 *)enc->npyValue;
        // 如果longVal等于get_nat()函数的返回值
        if (longVal == get_nat()) {
            // 设置type为JT_NULL
            tc->type = JT_NULL;
        } else {
            // 如果enc的datetimeIso标志为真
            if (enc->datetimeIso) {
                // 如果npyType是NPY_TIMEDELTA类型
                if (enc->npyType == NPY_TIMEDELTA) {
                    // 将pc的PyTypeToUTF8函数指针设置为NpyTimeDeltaToIsoCallback
                    pc->PyTypeToUTF8 = NpyTimeDeltaToIsoCallback;
                } else {
                    // 否则，将pc的PyTypeToUTF8函数指针设置为NpyDateTimeToIsoCallback
                    pc->PyTypeToUTF8 = NpyDateTimeToIsoCallback;
                }
                // 设置pc的longValue为longVal，表示ISO格式的时间值
                pc->longValue = longVal;
                // 设置type为JT_UTF8
                tc->type = JT_UTF8;
            } else {
                // 否则，获取enc的datetimeUnit值作为时间单位
                NPY_DATETIMEUNIT base = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
                // 将longVal转换为指定时间单位的基本单位
                if (scaleNanosecToUnit(&longVal, base) == -1) {
                    // 转换失败则跳转到INVALID标签处理
                    goto INVALID;
                }
                // 设置pc的longValue为longVal，表示长整型时间值
                pc->longValue = longVal;
                // 设置type为JT_LONG
                tc->type = JT_LONG;
            }
        }

        // 防止与混合类型DataFrame导致的无限循环，将npyCtxtPassthru和npyType重置
        enc->npyCtxtPassthru = NULL;
        enc->npyType = -1;
        return;
    }

    // 如果obj是迭代器或者数组，并且不是标量数组
    if (PyIter_Check(obj) || (PyArray_Check(obj) && !PyArray_CheckScalar(obj))) {
        // 跳转到ISITERABLE标签处理
        goto ISITERABLE;
    }

    // 如果obj是长整型
    if (PyLong_Check(obj)) {
        // 设置type为JT_LONG
        tc->type = JT_LONG;
        // 定义溢出标志和错误标志
        int overflow = 0;
        // 将obj转换为长整型，并检查是否溢出
        pc->longValue = PyLong_AsLongLongAndOverflow(obj, &overflow);
        int err;
        // 如果转换结果为-1且有异常发生
        err = (pc->longValue == -1) && PyErr_Occurred();

        // 如果发生了溢出
        if (overflow) {
            // 设置type为JT_BIGNUM
            tc->type = JT_BIGNUM;
        } else if (err) {
            // 如果转换过程中发生了错误，跳转到INVALID标签处理
            goto INVALID;
        }

        return;
    } else if (PyFloat_Check(obj)) {
        // 如果obj是浮点数类型
        const double val = PyFloat_AS_DOUBLE(obj);
        // 如果val是NaN或者无穷大
        if (npy_isnan(val) || npy_isinf(val)) {
            // 设置type为JT_NULL
            tc->type = JT_NULL;
        } else {
            // 设置pc的doubleValue为val，表示双精度浮点数值
            pc->doubleValue = val;
            // 设置type为JT_DOUBLE
            tc->type = JT_DOUBLE;
        }
        return;
    } else if (PyBytes_Check(obj)) {
        // 如果obj是字节串类型
        // 设置pc的PyTypeToUTF8函数指针为PyBytesToUTF8
        pc->PyTypeToUTF8 = PyBytesToUTF8;
        // 设置type为JT_UTF8
        tc->type = JT_UTF8;
        return;
    } else if (PyUnicode_Check(obj)) {
        // 如果obj是Unicode字符串类型
        // 设置pc的PyTypeToUTF8函数指针为PyUnicodeToUTF8
        pc->PyTypeToUTF8 = PyUnicodeToUTF8;
        // 设置type为JT_UTF8
        tc->type = JT_UTF8;
        return;
    } else if (object_is_decimal_type(obj)) {
        // 如果obj是十进制类型
        // 获取obj的双精度浮点数值，并赋给pc的doubleValue
        pc->doubleValue = PyFloat_AsDouble(obj);
        // 设置type为JT_DOUBLE
        tc->type = JT_DOUBLE;
        return;
    } else if (PyDateTime_Check(obj) || PyDate_Check(obj)) {
        // 如果obj是日期时间或日期对象
        // 标签处理部分略，待补充
        // ...
    }

    // ISITERABLE标签处理部分
ISITERABLE:
    // 标签处理部分略，待补充
    // ...

    return;

INVALID:
    // INVALID标签处理部分略，待补充
    // ...
}
    // 检查对象是否为自然类型
    if (object_is_nat_type(obj)) {
      // 如果是自然类型，则设置类型为Null并返回
      tc->type = JT_NULL;
      return;
    }

    // 如果启用了datetimeIso选项
    if (enc->datetimeIso) {
      // 设置字符转换回调函数为PyDateTimeToIsoCallback
      pc->PyTypeToUTF8 = PyDateTimeToIsoCallback;
      // 设置类型为UTF8
      tc->type = JT_UTF8;
    } else {
      // 获取datetimeUnit
      NPY_DATETIMEUNIT base = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
      // 将PyDateTime对象转换为Epoch时间并存储在longValue中
      pc->longValue = PyDateTimeToEpoch(obj, base);
      // 设置类型为LONG
      tc->type = JT_LONG;
    }
    // 返回
    return;
  } else if (PyTime_Check(obj)) {
    // 如果对象是PyTime对象
    pc->PyTypeToUTF8 = PyTimeToJSON;
    // 设置类型为UTF8
    tc->type = JT_UTF8;
    // 返回
    return;
  } else if (PyArray_IsScalar(obj, Datetime)) {
    // 如果对象是NumPy的datetime标量
    npy_int64 longVal;
    // 如果是NaT，则设置类型为Null并返回
    if (((PyDatetimeScalarObject *)obj)->obval == get_nat()) {
      tc->type = JT_NULL;
      return;
    }
    // 获取dtype
    PyArray_Descr *dtype = PyArray_DescrFromScalar(obj);
    // 如果dtype不是datetime类型，抛出错误
    if (!PyTypeNum_ISDATETIME(dtype->type_num)) {
      PyErr_Format(PyExc_ValueError, "Could not get resolution of datetime");
      return;
    }

    // 将标量对象转换为NPY_INT64类型并存储在longVal中
    PyArray_Descr *outcode = PyArray_DescrFromType(NPY_INT64);
    PyArray_CastScalarToCtype(obj, &longVal, outcode);
    Py_DECREF(outcode);

    // 如果启用了datetimeIso选项
    if (enc->datetimeIso) {
      // 存储longVal到GET_TC(tc)->longValue
      GET_TC(tc)->longValue = longVal;
      // 设置字符转换回调函数为NpyDateTimeToIsoCallback
      pc->PyTypeToUTF8 = NpyDateTimeToIsoCallback;
      // 从dtype获取datetime的元数据并存储在valueUnit中
      enc->valueUnit = get_datetime_metadata_from_dtype(dtype).base;
      // 设置类型为UTF8
      tc->type = JT_UTF8;
    } else {
      // 获取datetimeUnit
      NPY_DATETIMEUNIT base = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
      // 将PyDateTime对象转换为Epoch时间并存储在longValue中
      pc->longValue = PyDateTimeToEpoch(obj, base);
      // 设置类型为LONG
      tc->type = JT_LONG;
    }
    // 返回
    return;
  } else if (PyDelta_Check(obj)) {
    // 如果对象是PyDelta对象
    // pd.Timedelta对象或pd.NaT应该返回true
    // 对于其他对象，回退到每秒的纳秒数
    // TODO(anyone): 如果total_seconds返回值超过了有效位数能表示的范围，下面的转换会失去精度并可能溢出

    // 检查是否存在'_value'属性，如果存在则获取其长整型值，否则计算total_seconds乘以10^9得到纳秒数
    int64_t value = PyObject_HasAttrString(obj, "_value")
                        ? get_long_attr(obj, "_value")
                        : (int64_t)(total_seconds(obj) * 1000000000LL);

    // 如果值等于NaT，则设置类型为Null并返回
    if (value == get_nat()) {
      tc->type = JT_NULL;
      return;
    } else if (enc->datetimeIso) {
      // 如果启用了datetimeIso选项，设置字符转换回调函数为NpyTimeDeltaToIsoCallback
      pc->PyTypeToUTF8 = NpyTimeDeltaToIsoCallback;
      // 设置类型为UTF8
      tc->type = JT_UTF8;
    } else {
      // 否则，获取单位并进行适当的缩放
      const int unit = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
      if (scaleNanosecToUnit(&value, unit) != 0) {
        // TODO(username): 在这里添加某种形式的错误处理
      }

      // 如果发生溢出错误，跳转到INVALID标签处
      if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_OverflowError)) {
        goto INVALID;
      }

      // 设置类型为LONG，并存储值在longValue中
      tc->type = JT_LONG;
    }
    pc->longValue = value;
    // 返回
    return;
  } else if (PyArray_IsScalar(obj, Integer)) {
    // 如果对象是NumPy的整数标量
    // 设置类型为LONG
    tc->type = JT_LONG;
    // 将标量对象转换为NPY_INT64类型并存储在pc->longValue中
    PyArray_CastScalarToCtype(obj, &(pc->longValue),
                              PyArray_DescrFromType(NPY_INT64));

    // 如果发生溢出错误，跳转到INVALID标签处
    if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_OverflowError)) {
      goto INVALID;
    }

    // 返回
    return;
  } else if (PyArray_IsScalar(obj, Bool)) {
    // 如果对象是NumPy的布尔标量
    // 将标量对象转换为NPY_BOOL类型并存储在pc->longValue中
    PyArray_CastScalarToCtype(obj, &(pc->longValue),
                              PyArray_DescrFromType(NPY_BOOL));
    # 如果 pc->longValue 的值为真，则将 tc->type 设置为 JT_TRUE，否则设置为 JT_FALSE
    tc->type = (pc->longValue) ? JT_TRUE : JT_FALSE;
    # 函数执行完毕，返回
    return;
  } else if (PyArray_IsScalar(obj, Float) || PyArray_IsScalar(obj, Double)) {
    # 如果 obj 是浮点数或双精度浮点数标量，则将其转换为 C 类型，并存储到 pc->doubleValue 中
    PyArray_CastScalarToCtype(obj, &(pc->doubleValue),
                              PyArray_DescrFromType(NPY_DOUBLE));
    # 设置 tc->type 为 JT_DOUBLE
    tc->type = JT_DOUBLE;
    # 函数执行完毕，返回
    return;
  } else if (PyArray_CheckScalar(obj)) {
    # 如果 obj 是 NumPy 标量，则抛出类型错误异常，提醒无法在当前情况下对其进行 JSON 序列化
    PyErr_Format(PyExc_TypeError,
                 "%R (numpy-scalar) is not JSON serializable at the moment",
                 obj);
    # 转到标签 INVALID，处理异常情况
    goto INVALID;
  } else if (object_is_na_type(obj)) {
    # 如果 obj 是 NA 类型（特定类型的对象），则将 tc->type 设置为 JT_NULL
    tc->type = JT_NULL;
    # 函数执行完毕，返回
    return;
  }
# 检查对象是否是索引类型
if (object_is_index_type(obj)) {
    # 如果输出格式为SPLIT，设置类型为JT_OBJECT，并配置索引类型的迭代函数
    if (enc->outputFormat == SPLIT) {
        tc->type = JT_OBJECT;
        pc->iterBegin = Index_iterBegin;
        pc->iterEnd = Index_iterEnd;
        pc->iterNext = Index_iterNext;
        pc->iterGetValue = Index_iterGetValue;
        pc->iterGetName = Index_iterGetName;
        return;
    }

    # 否则获取对象的值，设置类型为JT_ARRAY，并配置相应的迭代函数
    pc->newObj = get_values(obj);
    if (pc->newObj) {
        tc->type = JT_ARRAY;
        pc->iterBegin = NpyArr_iterBegin;
        pc->iterEnd = NpyArr_iterEnd;
        pc->iterNext = NpyArr_iterNext;
        pc->iterGetValue = NpyArr_iterGetValue;
        pc->iterGetName = NpyArr_iterGetName;
    } else {
        # 获取值失败，跳转到INVALID标签
        goto INVALID;
    }

    return;
}

# 检查对象是否是序列类型
else if (object_is_series_type(obj)) {
    # 如果输出格式为SPLIT，设置类型为JT_OBJECT，并配置序列类型的迭代函数
    if (enc->outputFormat == SPLIT) {
        tc->type = JT_OBJECT;
        pc->iterBegin = Series_iterBegin;
        pc->iterEnd = Series_iterEnd;
        pc->iterNext = Series_iterNext;
        pc->iterGetValue = Series_iterGetValue;
        pc->iterGetName = Series_iterGetName;
        return;
    }

    # 否则获取对象的值
    pc->newObj = get_values(obj);
    if (!pc->newObj) {
        # 获取值失败，跳转到INVALID标签
        goto INVALID;
    }

    # 如果输出格式为INDEX或COLUMNS，处理对象索引及其值
    if (enc->outputFormat == INDEX || enc->outputFormat == COLUMNS) {
        tc->type = JT_OBJECT;
        # 获取对象的"index"属性
        PyObject *tmpObj = PyObject_GetAttrString(obj, "index");
        if (!tmpObj) {
            # 获取属性失败，跳转到INVALID标签
            goto INVALID;
        }
        # 获取属性的值
        PyObject *values = get_values(tmpObj);
        Py_DECREF(tmpObj);
        if (!values) {
            # 获取值失败，跳转到INVALID标签
            goto INVALID;
        }

        # 确保newObj是一个PyArrayObject对象
        if (!PyArray_Check(pc->newObj)) {
            PyErr_SetString(PyExc_TypeError,
                            "Object_beginTypeContext received a non-array object");
            goto INVALID;
        }
        const PyArrayObject *arrayobj = (const PyArrayObject *)pc->newObj;
        # 获取数组的维度
        pc->columnLabelsLen = PyArray_DIM(arrayobj, 0);
        # 编码标签值
        pc->columnLabels = NpyArr_encodeLabels((PyArrayObject *)values, enc,
                                               pc->columnLabelsLen);
        if (!pc->columnLabels) {
            # 编码标签失败，跳转到INVALID标签
            goto INVALID;
        }
    } else {
        # 输出格式不是INDEX或COLUMNS，设置类型为JT_ARRAY
        tc->type = JT_ARRAY;
    }
    # 配置序列类型的迭代函数
    pc->iterBegin = NpyArr_iterBegin;
    pc->iterEnd = NpyArr_iterEnd;
    pc->iterNext = NpyArr_iterNext;
    pc->iterGetValue = NpyArr_iterGetValue;
    pc->iterGetName = NpyArr_iterGetName;
    return;
}

# 检查对象是否是NumPy数组
else if (PyArray_Check(obj)) {
    # 如果npyCtxtPassthru不为空，直接传递NumPy数组上下文
    if (enc->npyCtxtPassthru) {
        pc->npyarr = enc->npyCtxtPassthru;
        tc->type = (pc->npyarr->columnLabels ? JT_OBJECT : JT_ARRAY);

        pc->iterBegin = NpyArrPassThru_iterBegin;
        pc->iterNext = NpyArr_iterNext;
        pc->iterEnd = NpyArrPassThru_iterEnd;
        pc->iterGetValue = NpyArr_iterGetValue;
        pc->iterGetName = NpyArr_iterGetName;

        enc->npyCtxtPassthru = NULL;
        return;
    }

    # 设置类型为JT_ARRAY，并配置NumPy数组的迭代函数
    tc->type = JT_ARRAY;
    pc->iterBegin = NpyArr_iterBegin;
    pc->iterEnd = NpyArr_iterEnd;
    pc->iterNext = NpyArr_iterNext;
    pc->iterGetValue = NpyArr_iterGetValue;
    pc->iterGetName = NpyArr_iterGetName;
    return;
}

# 检查对象是否是DataFrame类型
else if (object_is_dataframe_type(obj)) {
    # 如果存在块上下文透传，设置块上下文指针为透传的块上下文
    if (enc->blkCtxtPassthru) {
      pc->pdblock = enc->blkCtxtPassthru;
      # 根据透传块上下文中第一个 npyCtxts 的列标签情况，设置输出类型为对象或数组
      tc->type =
          (pc->pdblock->npyCtxts[0]->columnLabels ? JT_OBJECT : JT_ARRAY);

      # 设置迭代器开始、结束和下一个元素函数为透传块的对应函数
      pc->iterBegin = PdBlockPassThru_iterBegin;
      pc->iterEnd = PdBlockPassThru_iterEnd;
      pc->iterNext = PdBlock_iterNextItem;
      pc->iterGetName = PdBlock_iterGetName;
      pc->iterGetValue = NpyArr_iterGetValue;

      # 清空 enc 结构体中的块上下文透传指针，并返回
      enc->blkCtxtPassthru = NULL;
      return;
    }

    # 如果输出格式为 SPLIT，设置类型为对象，并设置相应的迭代函数
    if (enc->outputFormat == SPLIT) {
      tc->type = JT_OBJECT;
      pc->iterBegin = DataFrame_iterBegin;
      pc->iterEnd = DataFrame_iterEnd;
      pc->iterNext = DataFrame_iterNext;
      pc->iterGetValue = DataFrame_iterGetValue;
      pc->iterGetName = DataFrame_iterGetName;
      return;
    }

    # 设置默认的迭代函数为 PdBlock 的相应函数
    pc->iterBegin = PdBlock_iterBegin;
    pc->iterEnd = PdBlock_iterEnd;
    pc->iterNext = PdBlock_iterNext;
    pc->iterGetName = PdBlock_iterGetName;
    pc->iterGetValue = NpyArr_iterGetValue;

    # 根据输出格式设置类型为数组
    if (enc->outputFormat == VALUES) {
      tc->type = JT_ARRAY;
    } else if (enc->outputFormat == RECORDS) {
      tc->type = JT_ARRAY;
      # 获取对象 obj 的 "columns" 属性，并检查是否成功获取
      PyObject *tmpObj = PyObject_GetAttrString(obj, "columns");
      if (!tmpObj) {
        # 失败则跳转到标记为 INVALID 的处理
        goto INVALID;
      }
      # 获取 "columns" 属性的值，并检查是否成功获取
      PyObject *values = get_values(tmpObj);
      if (!values) {
        Py_DECREF(tmpObj);
        # 失败则跳转到标记为 INVALID 的处理
        goto INVALID;
      }
      # 设置 pc 结构体中的列标签长度为获取的值的大小，并对列标签进行编码
      pc->columnLabelsLen = PyObject_Size(tmpObj);
      pc->columnLabels = NpyArr_encodeLabels((PyArrayObject *)values, enc,
                                             pc->columnLabelsLen);
      Py_DECREF(tmpObj);
      if (!pc->columnLabels) {
        # 失败则跳转到标记为 INVALID 的处理
        goto INVALID;
      }
    // 如果输出格式是 INDEX 或者 COLUMNS，则执行以下操作
    } else if (enc->outputFormat == INDEX || enc->outputFormat == COLUMNS) {
      // 设置类型为 JT_OBJECT
      tc->type = JT_OBJECT;
      // 根据输出格式选择获取 "index" 或 "columns" 属性
      PyObject *tmpObj =
          (enc->outputFormat == INDEX ? PyObject_GetAttrString(obj, "index")
                                      : PyObject_GetAttrString(obj, "columns"));
      // 如果未能获取到属性对象，则跳转到 INVALID 标签
      if (!tmpObj) {
        goto INVALID;
      }
      // 获取属性对象的值
      PyObject *values = get_values(tmpObj);
      // 如果获取值失败，则释放临时对象并跳转到 INVALID 标签
      if (!values) {
        Py_DECREF(tmpObj);
        goto INVALID;
      }
      // 获取属性对象的大小作为行标签的长度
      pc->rowLabelsLen = PyObject_Size(tmpObj);
      // 编码行标签并存储到 pc->rowLabels
      pc->rowLabels =
          NpyArr_encodeLabels((PyArrayObject *)values, enc, pc->rowLabelsLen);
      // 释放临时对象
      Py_DECREF(tmpObj);
      // 根据输出格式再次获取 "columns" 或 "index" 属性
      tmpObj =
          (enc->outputFormat == INDEX ? PyObject_GetAttrString(obj, "columns")
                                      : PyObject_GetAttrString(obj, "index"));
      // 如果未能获取到属性对象，则释放已分配的行标签并跳转到 INVALID 标签
      if (!tmpObj) {
        NpyArr_freeLabels(pc->rowLabels, pc->rowLabelsLen);
        pc->rowLabels = NULL;
        goto INVALID;
      }
      // 获取属性对象的值
      values = get_values(tmpObj);
      // 如果获取值失败，则释放临时对象和已分配的行标签并跳转到 INVALID 标签
      if (!values) {
        Py_DECREF(tmpObj);
        NpyArr_freeLabels(pc->rowLabels, pc->rowLabelsLen);
        pc->rowLabels = NULL;
        goto INVALID;
      }
      // 获取属性对象的大小作为列标签的长度
      pc->columnLabelsLen = PyObject_Size(tmpObj);
      // 编码列标签并存储到 pc->columnLabels
      pc->columnLabels = NpyArr_encodeLabels((PyArrayObject *)values, enc,
                                             pc->columnLabelsLen);
      // 释放临时对象
      Py_DECREF(tmpObj);
      // 如果列标签编码失败，则释放已分配的行标签并跳转到 INVALID 标签
      if (!pc->columnLabels) {
        NpyArr_freeLabels(pc->rowLabels, pc->rowLabelsLen);
        pc->rowLabels = NULL;
        goto INVALID;
      }

      // 如果输出格式为 COLUMNS，则设置转置标志
      if (enc->outputFormat == COLUMNS) {
        pc->transpose = 1;
      }
    } else {
      // 如果输出格式不是 INDEX 或 COLUMNS，则跳转到 INVALID 标签
      goto INVALID;
    }
    // 返回
    return;
  } else if (PyDict_Check(obj)) {
    // 如果对象是字典类型，则执行以下操作
    tc->type = JT_OBJECT;
    // 设置字典迭代函数指针
    pc->iterBegin = Dict_iterBegin;
    pc->iterEnd = Dict_iterEnd;
    pc->iterNext = Dict_iterNext;
    pc->iterGetValue = Dict_iterGetValue;
    pc->iterGetName = Dict_iterGetName;
    // 存储字典对象到 pc->dictObj 并增加引用计数
    pc->dictObj = obj;
    Py_INCREF(obj);

    // 返回
    return;
  } else if (PyList_Check(obj)) {
    // 如果对象是列表类型，则执行以下操作
    tc->type = JT_ARRAY;
    // 设置列表迭代函数指针
    pc->iterBegin = List_iterBegin;
    pc->iterEnd = List_iterEnd;
    pc->iterNext = List_iterNext;
    pc->iterGetValue = List_iterGetValue;
    pc->iterGetName = List_iterGetName;
    // 返回
    return;
  } else if (PyTuple_Check(obj)) {
    // 如果对象是元组类型，则执行以下操作
    tc->type = JT_ARRAY;
    // 设置元组迭代函数指针
    pc->iterBegin = Tuple_iterBegin;
    pc->iterEnd = Tuple_iterEnd;
    pc->iterNext = Tuple_iterNext;
    pc->iterGetValue = Tuple_iterGetValue;
    pc->iterGetName = Tuple_iterGetName;
    // 返回
    return;
  } else if (PyAnySet_Check(obj)) {
    // 如果对象是集合类型，则执行以下操作
    tc->type = JT_ARRAY;
    // 设置集合迭代函数指针
    pc->iterBegin = Set_iterBegin;
    pc->iterEnd = Set_iterEnd;
    pc->iterNext = Set_iterNext;
    pc->iterGetValue = Set_iterGetValue;
    pc->iterGetName = Set_iterGetName;
    // 返回
    return;
  }

  // 尝试获取对象的 "toDict" 方法
  PyObject *toDictFunc = PyObject_GetAttrString(obj, "toDict");

  // 如果成功获取方法对象
  if (toDictFunc) {
    // 创建空元组
    PyObject *tuple = PyTuple_New(0);
    // 调用对象的 "toDict" 方法并获取结果
    PyObject *toDictResult = PyObject_Call(toDictFunc, tuple, NULL);
    // 释放元组对象
    Py_DECREF(tuple);
    // 释放方法对象
    Py_DECREF(toDictFunc);
    //```cpp
    // 如果调用成功，则执行以下操作
    if (toDictResult) {
      // 返回结果为字典类型，设置类型为 JT_OBJECT
      tc->type = JT_OBJECT;
      // 设置字典迭代函数指针
      pc->iterBegin = Dict_iterBegin;
      pc->iterEnd = Dict_iterEnd;
      pc->iterNext = Dict_iterNext;
      pc->iterGetValue = Dict_iterGetValue;
      pc->iterGetName = Dict_iterGetName;
      // 存储字典对象到 pc->dictObj 并增加引用计数
      pc->dictObj = toDictResult;
      Py_INCREF(toDictResult);
      // 返回
      return;
    }
  }

  // 如果对象不属于以上任何一种类型，则跳转到 INVALID 标签
  goto INVALID;


这段代码是一个条件语句块，根据对象的类型和输出格式设置不同的迭代器和标签，以及执行一些对象方法。
    # 如果 toDictResult 是 NULL（空指针），清除当前的 Python 异常状态
    if (toDictResult == NULL) {
      PyErr_Clear();
      # 设置 tc 对象的类型为 JT_NULL
      tc->type = JT_NULL;
      return;
    }

    # 如果 toDictResult 不是 Python 字典对象，释放其引用并返回
    if (!PyDict_Check(toDictResult)) {
      Py_DECREF(toDictResult);
      # 设置 tc 对象的类型为 JT_NULL
      tc->type = JT_NULL;
      return;
    }

    # 设置 tc 对象的类型为 JT_OBJECT
    tc->type = JT_OBJECT;
    # 设置 pc 对象的迭代器函数指针为字典对象的迭代器函数
    pc->iterBegin = Dict_iterBegin;
    pc->iterEnd = Dict_iterEnd;
    pc->iterNext = Dict_iterNext;
    pc->iterGetValue = Dict_iterGetValue;
    pc->iterGetName = Dict_iterGetName;
    # 将字典对象的引用赋值给 pc 对象的 dictObj 属性
    pc->dictObj = toDictResult;
    return;
  }

  # 清除当前的 Python 异常状态
  PyErr_Clear();

  # 如果编码器（enc）有默认的处理程序，调用对象的默认处理函数，并跳转到标签 INVALID 处
  if (enc->defaultHandler) {
    Object_invokeDefaultHandler(obj, enc);
    goto INVALID;
  }

  # 设置 tc 对象的类型为 JT_OBJECT
  tc->type = JT_OBJECT;
  # 设置 pc 对象的迭代器函数指针为目录对象的迭代器函数
  pc->iterBegin = Dir_iterBegin;
  pc->iterEnd = Dir_iterEnd;
  pc->iterNext = Dir_iterNext;
  pc->iterGetValue = Dir_iterGetValue;
  pc->iterGetName = Dir_iterGetName;
  return;
static const char *Object_getStringValue(JSOBJ obj, JSONTypeContext *tc,
                                         size_t *_outLen) {
    // 调用 JSONTypeContext 结构体中的 PyTypeToUTF8 函数，将 obj 转换为 UTF-8 编码的字符串，并返回其指针
    return GET_TC(tc)->PyTypeToUTF8(obj, tc, _outLen);
}

static JSINT64 Object_getLongValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    // 返回 JSONTypeContext 结构体中的 longValue 值
    return GET_TC(tc)->longValue;
}

static double Object_getDoubleValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    // 返回 JSONTypeContext 结构体中的 doubleValue 值
    return GET_TC(tc)->doubleValue;
}

static const char *Object_getBigNumStringValue(JSOBJ obj, JSONTypeContext *tc,
                                               size_t *_outLen) {
    // 将 obj 转换为字符串表示形式，并以 UTF-8 编码返回其指针
    PyObject *repr = PyObject_Str(obj);
    const char *str = PyUnicode_AsUTF8AndSize(repr, (Py_ssize_t *)_outLen);
    // 分配内存来保存字符串，并复制字符串内容
    char *bytes = PyObject_Malloc(*_outLen + 1);
    memcpy(bytes, str, *_outLen + 1);
    GET_TC(tc)->cStr = bytes;

    Py_DECREF(repr);

    // 返回保存在 JSONTypeContext 结构体中的 cStr 指针
    return GET_TC(tc)->cStr;
}

static void Object_releaseObject(JSOBJ _obj) {
    // 减少对象的引用计数，用于释放 Python 对象
    Py_DECREF((PyObject *)_obj);
}

static void Object_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
    // 调用 JSONTypeContext 结构体中的 iterBegin 函数，准备开始迭代对象 obj
    GET_TC(tc)->iterBegin(obj, tc);
}

static int Object_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    // 调用 JSONTypeContext 结构体中的 iterNext 函数，获取下一个迭代对象
    return GET_TC(tc)->iterNext(obj, tc);
}

static void Object_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    // 调用 JSONTypeContext 结构体中的 iterEnd 函数，结束对象的迭代
    GET_TC(tc)->iterEnd(obj, tc);
}

static JSOBJ Object_iterGetValue(JSOBJ obj, JSONTypeContext *tc) {
    // 调用 JSONTypeContext 结构体中的 iterGetValue 函数，获取当前迭代对象的值
    return GET_TC(tc)->iterGetValue(obj, tc);
}

static char *Object_iterGetName(JSOBJ obj, JSONTypeContext *tc,
                                size_t *outLen) {
    // 调用 JSONTypeContext 结构体中的 iterGetName 函数，获取当前迭代对象的名称
    return GET_TC(tc)->iterGetName(obj, tc, outLen);
}
    return NULL;
  }



  static char *kwlist[] = {"obj",
                           "ensure_ascii",
                           "double_precision",
                           "encode_html_chars",
                           "orient",
                           "date_unit",
                           "iso_dates",
                           "default_handler",
                           "indent",
                           NULL};



  PyObject *oinput = NULL;
  PyObject *oensureAscii = NULL;
  int idoublePrecision = 10; // 默认的双精度设置为10
  PyObject *oencodeHTMLChars = NULL;
  char *sOrient = NULL;
  char *sdateFormat = NULL;
  PyObject *oisoDates = 0;
  PyObject *odefHandler = 0;
  int indent = 0;



  PyObjectEncoder pyEncoder = {
      {
          .beginTypeContext = Object_beginTypeContext,
          .endTypeContext = Object_endTypeContext,
          .getStringValue = Object_getStringValue,
          .getLongValue = Object_getLongValue,
          .getIntValue = NULL,
          .getDoubleValue = Object_getDoubleValue,
          .getBigNumStringValue = Object_getBigNumStringValue,
          .iterBegin = Object_iterBegin,
          .iterNext = Object_iterNext,
          .iterEnd = Object_iterEnd,
          .iterGetValue = Object_iterGetValue,
          .iterGetName = Object_iterGetName,
          .releaseObject = Object_releaseObject,
          .malloc = PyObject_Malloc,
          .realloc = PyObject_Realloc,
          .free = PyObject_Free,
          .recursionMax = -1,
          .doublePrecision = idoublePrecision,
          .forceASCII = 1,
          .encodeHTMLChars = 0,
          .indent = indent,
          .errorMsg = NULL,
      },
      .npyCtxtPassthru = NULL,
      .blkCtxtPassthru = NULL,
      .npyType = -1,
      .npyValue = NULL,
      .datetimeIso = 0,
      .datetimeUnit = NPY_FR_ms,
      .outputFormat = COLUMNS,
      .defaultHandler = NULL,
  };
  JSONObjectEncoder *encoder = (JSONObjectEncoder *)&pyEncoder;



  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiOssOOi", kwlist, &oinput,
                                   &oensureAscii, &idoublePrecision,
                                   &oencodeHTMLChars, &sOrient, &sdateFormat,
                                   &oisoDates, &odefHandler, &indent)) {
    return NULL;
  }



  if (oensureAscii != NULL && !PyObject_IsTrue(oensureAscii)) {
    encoder->forceASCII = 0;
  }



  if (oencodeHTMLChars != NULL && PyObject_IsTrue(oencodeHTMLChars)) {
    encoder->encodeHTMLChars = 1;
  }



  if (idoublePrecision > JSON_DOUBLE_MAX_DECIMALS || idoublePrecision < 0) {
    PyErr_Format(
        PyExc_ValueError,
        "Invalid value '%d' for option 'double_precision', max is '%u'",
        idoublePrecision, JSON_DOUBLE_MAX_DECIMALS);
    return NULL;
  }
  encoder->doublePrecision = idoublePrecision;



  if (sOrient != NULL) {
    if (strcmp(sOrient, "records") == 0) {
      pyEncoder.outputFormat = RECORDS;
  } else if (strcmp(sOrient, "index") == 0) {
    // 如果指定的输出格式为 "index"，则设置编码器的输出格式为 INDEX
    pyEncoder.outputFormat = INDEX;
  } else if (strcmp(sOrient, "split") == 0) {
    // 如果指定的输出格式为 "split"，则设置编码器的输出格式为 SPLIT
    pyEncoder.outputFormat = SPLIT;
  } else if (strcmp(sOrient, "values") == 0) {
    // 如果指定的输出格式为 "values"，则设置编码器的输出格式为 VALUES
    pyEncoder.outputFormat = VALUES;
  } else if (strcmp(sOrient, "columns") != 0) {
    // 如果指定的输出格式不是 "columns"，则抛出一个值错误异常，指明无效的选项 'orient' 的值
    PyErr_Format(PyExc_ValueError, "Invalid value '%s' for option 'orient'", sOrient);
    return NULL;
  }
}

if (sdateFormat != NULL) {
  if (strcmp(sdateFormat, "s") == 0) {
    // 如果日期格式为 "s"，则设置日期时间单位为秒（NPY_FR_s）
    pyEncoder.datetimeUnit = NPY_FR_s;
  } else if (strcmp(sdateFormat, "ms") == 0) {
    // 如果日期格式为 "ms"，则设置日期时间单位为毫秒（NPY_FR_ms）
    pyEncoder.datetimeUnit = NPY_FR_ms;
  } else if (strcmp(sdateFormat, "us") == 0) {
    // 如果日期格式为 "us"，则设置日期时间单位为微秒（NPY_FR_us）
    pyEncoder.datetimeUnit = NPY_FR_us;
  } else if (strcmp(sdateFormat, "ns") == 0) {
    // 如果日期格式为 "ns"，则设置日期时间单位为纳秒（NPY_FR_ns）
    pyEncoder.datetimeUnit = NPY_FR_ns;
  } else {
    // 如果日期格式不在支持范围内，则抛出值错误异常，指明无效的选项 'date_unit' 的值
    PyErr_Format(PyExc_ValueError, "Invalid value '%s' for option 'date_unit'", sdateFormat);
    return NULL;
  }
}

if (oisoDates != NULL && PyObject_IsTrue(oisoDates)) {
  // 如果ISO日期标志 oisoDates 存在且为真，则设置编码器的 ISO 日期标志为真
  pyEncoder.datetimeIso = 1;
}

if (odefHandler != NULL && odefHandler != Py_None) {
  // 如果默认处理程序 odefHandler 存在且不是 Py_None
  if (!PyCallable_Check(odefHandler)) {
    // 如果 odefHandler 不是可调用的对象，则抛出类型错误异常，指明默认处理程序不可调用
    PyErr_SetString(PyExc_TypeError, "Default handler is not callable");
    return NULL;
  }
  // 否则，设置编码器的默认处理程序为 odefHandler
  pyEncoder.defaultHandler = odefHandler;
}

encoder->indent = indent;

// 保存编码器原始的输出格式
pyEncoder.originalOutputFormat = pyEncoder.outputFormat;

char buffer[65536];
// 调用 JSON_EncodeObject 函数将输入对象 oinput 编码为 JSON 格式，结果保存在 buffer 中
char *ret = JSON_EncodeObject(oinput, encoder, buffer, sizeof(buffer));
if (PyErr_Occurred()) {
  // 如果在编码过程中出现 Python 异常，则返回 NULL
  return NULL;
}

if (encoder->errorMsg) {
  // 如果编码器报告了错误消息
  if (ret != buffer) {
    // 如果 ret 不等于 buffer，则释放 ret 所指向的内存
    encoder->free(ret);
  }
  // 抛出溢出错误异常，消息为编码器报告的错误消息
  PyErr_Format(PyExc_OverflowError, "%s", encoder->errorMsg);
  return NULL;
}

// 根据 ret 创建一个新的 PyUnicode 对象
PyObject *newobj = PyUnicode_FromString(ret);

if (ret != buffer) {
  // 如果 ret 不等于 buffer，则释放 ret 所指向的内存
  encoder->free(ret);
}

// 返回新创建的 PyUnicode 对象
return newobj;
}



# 这行代码是一个单独的右大括号 '}'，用于结束一个代码块或者字典的定义。
```