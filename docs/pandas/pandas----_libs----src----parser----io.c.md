# `D:\src\scipysrc\pandas\pandas\_libs\src\parser\io.c`

```
/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.
*/

#include "pandas/parser/io.h"

/*
  On-disk FILE, uncompressed
*/

// 创建一个新的读取数据源的函数，接受一个 Python 对象作为参数
void *new_rd_source(PyObject *obj) {
  // 分配内存以存储 rd_source 结构体，并进行错误检查
  rd_source *rds = (rd_source *)malloc(sizeof(rd_source));
  
  if (rds == NULL) {
    // 如果内存分配失败，引发内存错误异常并返回空指针
    PyErr_NoMemory();
    return NULL;
  }
  
  // 增加 Python 对象的引用计数，以防止其在生存期内被销毁
  Py_INCREF(obj);
  // 将 Python 对象保存到 rd_source 结构体中
  rds->obj = obj;
  // 初始化 buffer 为 NULL
  rds->buffer = NULL;
  // 初始化 position 为 0
  rds->position = 0;

  return (void *)rds;
}

/*

  Cleanup callbacks

 */

// 删除读取数据源的函数，释放相关资源
void del_rd_source(void *rds) {
  // 释放 rd_source 结构体中存储的 Python 对象的引用
  Py_XDECREF(RDS(rds)->obj);
  // 释放 rd_source 结构体中存储的 buffer 的引用
  Py_XDECREF(RDS(rds)->buffer);
  // 释放 rd_source 结构体的内存
  free(rds);
}

/*

  IO callbacks

 */

// 缓冲区读取字节的函数，从 Python 对象中读取数据
char *buffer_rd_bytes(void *source, size_t nbytes, size_t *bytes_read,
                      int *status, const char *encoding_errors) {
  // 将 void 指针转换为 rd_source 结构体指针
  rd_source *src = RDS(source);
  // 保存 Python 全局解释器锁的状态
  PyGILState_STATE state = PyGILState_Ensure();

  // 释放旧的 buffer 对象
  Py_XDECREF(src->buffer);
  src->buffer = NULL;
  // 构建调用 read 方法的参数
  PyObject *args = Py_BuildValue("(i)", nbytes);

  // 获取 Python 对象的 read 方法
  PyObject *func = PyObject_GetAttrString(src->obj, "read");

  /* Note: PyObject_CallObject requires the GIL */
  // 调用 Python 对象的 read 方法
  PyObject *result = PyObject_CallObject(func, args);
  Py_XDECREF(args);
  Py_XDECREF(func);

  if (result == NULL) {
    // 如果调用失败，释放全局解释器锁并返回错误状态
    PyGILState_Release(state);
    *bytes_read = 0;
    *status = CALLING_READ_FAILED;
    return NULL;
  } else if (!PyBytes_Check(result)) {
    // 如果返回的不是字节对象，将其转换为 UTF-8 编码的字符串
    PyObject *tmp = PyUnicode_AsEncodedString(result, "utf-8", encoding_errors);
    Py_DECREF(result);
    if (tmp == NULL) {
      // 转换失败时释放全局解释器锁并返回 NULL
      PyGILState_Release(state);
      return NULL;
    }
    result = tmp;
  }

  // 获取结果对象的长度
  const size_t length = PySequence_Length(result);

  if (length == 0)
    // 如果长度为 0，表示已到达文件末尾
    *status = REACHED_EOF;
  else
    // 否则状态为正常
    *status = 0;

  // 将结果对象保存到 buffer 中
  src->buffer = result;
  // 获取结果对象的字符串指针
  char *retval = PyBytes_AsString(result);

  // 释放全局解释器锁
  PyGILState_Release(state);

  // TODO: 更多错误处理
  // 返回读取的字节长度
  *bytes_read = length;

  return retval;
}
```