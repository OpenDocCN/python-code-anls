# `D:\src\scipysrc\pandas\pandas\_libs\src\vendored\ujson\python\JSONtoObj.c`

```
/*
版权声明部分
Copyright (c) 2011-2013, ESN Social Software AB and Jonas Tarnstrom
保留所有权利。

在源代码和二进制形式中重新发布和使用，无论是否经过修改，
只要符合以下条件：
    * 必须保留上述版权声明、此条件列表和以下免责声明。
    * 在提供分发的文档和/或其他材料中，必须重复上述版权声明、此条件列表和以下免责声明。
    * 未经特定事先书面许可，不得使用 ESN Social Software AB 或其贡献者的名称来认可或推广从本软件衍生的产品。

本软件由版权持有人和贡献者按 "原样" 提供，任何明示或默示的保证，
包括但不限于适销性和特定用途适用性的默示保证，都被拒绝。在任何情况下，
无论是因为合同、严格责任还是侵权（包括疏忽或其他原因），都不应由 ESN Social Software AB 或 Jonas Tarnstrom 负责任何直接、间接、附带、特殊、示范性或后果性的损害赔偿。
即使已经被告知发生此类损害的可能性。
*/

// 代码片段部分来自 MODP_ASCII - ASCII 转换（大写/小写等）
// https://github.com/client9/stringencoders
// 版权所有 2007 年 Nick Galbreath -- nickg [at] modp [dot] com. 保留所有权利。

// 数字解码器源自 TCL 库
// https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
// 版权所有 1988-1993 年加利福尼亚大学理事会。
// 版权所有 1994 年 Sun Microsystems, Inc.

// 许可证位于 LICENSES/ULTRAJSON_LICENSE

#include "pandas/vendored/ujson/lib/ultrajson.h"
#include <Python.h>

// 定义 PY_SSIZE_T_CLEAN 以避免使用旧的整数类型 API

// 定义 Object_objectAddKey 函数
static int Object_objectAddKey(void *Py_UNUSED(prv), JSOBJ obj, JSOBJ name,
                               JSOBJ value) {
  // 将 name-value 对添加到 obj（假定为 Python 字典）
  int ret = PyDict_SetItem(obj, name, value);
  // 释放 name 和 value 的引用计数
  Py_DECREF((PyObject *)name);
  Py_DECREF((PyObject *)value);
  return ret == 0 ? 1 : 0;  // 返回操作是否成功的标志
}

// 定义 Object_arrayAddItem 函数
static int Object_arrayAddItem(void *Py_UNUSED(prv), JSOBJ obj, JSOBJ value) {
  // 将 value 添加到 obj（假定为 Python 列表）
  int ret = PyList_Append(obj, value);
  // 释放 value 的引用计数
  Py_DECREF((PyObject *)value);
  return ret == 0 ? 1 : 0;  // 返回操作是否成功的标志
}

// 定义 Object_newString 函数
static JSOBJ Object_newString(void *Py_UNUSED(prv), wchar_t *start,
                              wchar_t *end) {
  // 根据宽字符数组创建 Python Unicode 字符串对象
  return PyUnicode_FromWideChar(start, (end - start));
}

// 定义 Object_newTrue 函数
static JSOBJ Object_newTrue(void *Py_UNUSED(prv)) {
  // 返回 Python 中的 True 对象
  Py_RETURN_TRUE;
}

// 定义 Object_newFalse 函数
static JSOBJ Object_newFalse(void *Py_UNUSED(prv)) {
  // 返回 Python 中的 False 对象
  Py_RETURN_FALSE;
}

// 定义 Object_newNull 函数
static JSOBJ Object_newNull(void *Py_UNUSED(prv)) {
  // 返回 Python 中的 None 对象
  Py_RETURN_NONE;
}
static JSOBJ Object_newPosInf(void *Py_UNUSED(prv)) {
  // 返回一个表示正无穷大的 Python 浮点数对象
  return PyFloat_FromDouble(Py_HUGE_VAL);
}

static JSOBJ Object_newNegInf(void *Py_UNUSED(prv)) {
  // 返回一个表示负无穷大的 Python 浮点数对象
  return PyFloat_FromDouble(-Py_HUGE_VAL);
}

static JSOBJ Object_newObject(void *Py_UNUSED(prv), void *Py_UNUSED(decoder)) {
  // 返回一个新的空字典对象
  return PyDict_New();
}

static JSOBJ Object_endObject(void *Py_UNUSED(prv), JSOBJ obj) {
  // 直接返回传入的对象，用于结束一个 JSON 对象的解析
  return obj;
}

static JSOBJ Object_newArray(void *Py_UNUSED(prv), void *Py_UNUSED(decoder)) {
  // 返回一个新的空列表对象
  return PyList_New(0);
}

static JSOBJ Object_endArray(void *Py_UNUSED(prv), JSOBJ obj) {
  // 直接返回传入的对象，用于结束一个 JSON 数组的解析
  return obj;
}

static JSOBJ Object_newInteger(void *Py_UNUSED(prv), JSINT32 value) {
  // 将传入的 JSINT32 整数值转换为 Python 长整型对象并返回
  return PyLong_FromLong(value);
}

static JSOBJ Object_newLong(void *Py_UNUSED(prv), JSINT64 value) {
  // 将传入的 JSINT64 长整数值转换为 Python 长整型对象并返回
  return PyLong_FromLongLong(value);
}

static JSOBJ Object_newUnsignedLong(void *Py_UNUSED(prv), JSUINT64 value) {
  // 将传入的 JSUINT64 无符号长整数值转换为 Python 无符号长整型对象并返回
  return PyLong_FromUnsignedLongLong(value);
}

static JSOBJ Object_newDouble(void *Py_UNUSED(prv), double value) {
  // 将传入的双精度浮点数值转换为 Python 浮点数对象并返回
  return PyFloat_FromDouble(value);
}

static void Object_releaseObject(void *Py_UNUSED(prv), JSOBJ obj,
                                 void *Py_UNUSED(decoder)) {
  // 释放 Python 对象的引用计数，用于释放由 JSON 解析器创建的对象
  Py_XDECREF(((PyObject *)obj));
}

PyObject *JSONToObj(PyObject *Py_UNUSED(self), PyObject *args,
                    PyObject *kwargs) {
  // 定义一个 JSON 对象解码器，配置其中各个处理函数和内存管理函数
  JSONObjectDecoder dec = {.newString = Object_newString,
                           .objectAddKey = Object_objectAddKey,
                           .arrayAddItem = Object_arrayAddItem,
                           .newTrue = Object_newTrue,
                           .newFalse = Object_newFalse,
                           .newNull = Object_newNull,
                           .newPosInf = Object_newPosInf,
                           .newNegInf = Object_newNegInf,
                           .newObject = Object_newObject,
                           .endObject = Object_endObject,
                           .newArray = Object_newArray,
                           .endArray = Object_endArray,
                           .newInt = Object_newInteger,
                           .newLong = Object_newLong,
                           .newUnsignedLong = Object_newUnsignedLong,
                           .newDouble = Object_newDouble,
                           .releaseObject = Object_releaseObject,
                           .malloc = PyObject_Malloc,
                           .free = PyObject_Free,
                           .realloc = PyObject_Realloc,
                           .errorStr = NULL,
                           .errorOffset = NULL,
                           .preciseFloat = 0,
                           .prv = NULL};

  char *kwlist[] = {"obj", "precise_float", NULL};
  char *buf;
  Py_ssize_t len;
  
  // 解析 Python 函数的参数，获取 JSON 字符串和可选的精确浮点数标志位
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s#|b", kwlist, &buf, &len,
                                   &dec.preciseFloat)) {
    return NULL;
  }

  // 使用解码器对 JSON 字符串进行解析，并返回解析后的 Python 对象
  PyObject *ret = JSON_DecodeObject(&dec, buf, len);

  // 检查是否有 Python 异常发生，如果有则释放返回的对象
  if (PyErr_Occurred()) {
    if (ret) {
      Py_DECREF((PyObject *)ret);
    }
    return NULL;
  }

  // 返回解析后的 Python 对象
  return ret;
}
    return NULL;
  }


    // 如果解析器返回 NULL，直接返回 NULL
    return NULL;
  }

  if (dec.errorStr) {
    /*
    FIXME: 可以在这里使用更好的错误信息，包括输入中失败的元素等
    */

    // 使用解析器返回的错误字符串格式化 Python 的 ValueError 异常
    PyErr_Format(PyExc_ValueError, "%s", dec.errorStr);

    // 如果 ret 不为空，释放其引用
    if (ret) {
      Py_DECREF((PyObject *)ret);
    }

    // 返回 NULL 表示错误
    return NULL;
  }

  // 返回解析结果
  return ret;
}



# 这行代码关闭了一个代码块，结束了一个函数或者其他代码结构的定义。
```