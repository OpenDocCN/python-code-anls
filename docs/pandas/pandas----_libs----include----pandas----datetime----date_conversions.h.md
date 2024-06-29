# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\datetime\date_conversions.h`

```
/*
Copyright (c) 2020, PyData Development Team
All rights reserved.
Distributed under the terms of the BSD Simplified License.
The full license is in the LICENSE file, distributed with this software.
*/

// 声明使用一次的预处理器指令，确保 PyObject_HEAD 为单次声明
#pragma once

// 定义 PY_SSIZE_T_CLEAN，表示只包含“size_t”类型的宏
#define PY_SSIZE_T_CLEAN

// 包含 Python.h 头文件，提供 Python C API 的接口和定义
#include <Python.h>

// 包含 numpy/ndarraytypes.h 头文件，提供 NumPy 数组的数据类型和宏定义
#include <numpy/ndarraytypes.h>

// 将 nanosecond 分辨率的值缩放到指定单位分辨率上
// value: 待缩放的值
// unit: 目标分辨率单位
// 返回值: 操作是否成功的整数状态码
int scaleNanosecToUnit(int64_t *value, NPY_DATETIMEUNIT unit);

// 将表示日期的 int64 对象转换为 ISO 格式的字符串
// value: 待转换的 int64 值
// valueUnit: 输入值的单位
// base: ISO 格式的基准单位（例如："s" 表示秒，"ns" 表示纳秒）
// len: 返回的字符串长度
// 返回值: ISO 格式的日期时间字符串
char *int64ToIso(int64_t value, NPY_DATETIMEUNIT valueUnit,
                 NPY_DATETIMEUNIT base, size_t *len);

// 将表示持续时间的 int64 对象转换为 ISO 格式的持续时间字符串
// value: 待转换的 int64 值
// len: 返回的字符串长度
// 返回值: ISO 格式的持续时间字符串
char *int64ToIsoDuration(int64_t value, size_t *len);
```