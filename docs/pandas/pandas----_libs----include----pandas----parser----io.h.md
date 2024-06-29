# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\parser\io.h`

```
/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.
*/

#pragma once

// 清除 PY_SSIZE_T_CLEAN 宏定义，确保使用更安全的 ssize_t 类型
#define PY_SSIZE_T_CLEAN

// 包含 C 语言头文件 "tokenizer.h"
#include "tokenizer.h"

// 包含 Python.h 头文件，提供 Python C API 的支持
#include <Python.h>

// 定义宏 FS，将 void* 类型转换为 file_source* 类型
#define FS(source) ((file_source *)source)

// 定义结构体 _rd_source，表示读取源的对象
typedef struct _rd_source {
    PyObject *obj;      // Python 对象指针
    PyObject *buffer;   // 缓冲区 Python 对象指针
    size_t position;    // 当前位置的字节偏移量
} rd_source;

// 定义宏 RDS，将 void* 类型转换为 rd_source* 类型
#define RDS(source) ((rd_source *)source)

// 声明函数 new_rd_source，用于创建新的 rd_source 对象
void *new_rd_source(PyObject *obj);

// 声明函数 del_rd_source，用于删除 rd_source 对象
void del_rd_source(void *src);

// 声明函数 buffer_rd_bytes，用于从源中读取指定数量的字节到缓冲区
char *buffer_rd_bytes(void *source, size_t nbytes, size_t *bytes_read,
                      int *status, const char *encoding_errors);
```