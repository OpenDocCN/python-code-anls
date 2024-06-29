# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\parser\pd_parser.h`

```
/*
Copyright (c) 2023, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.
*/
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 定义清理 Py_ssize_t 的宏，确保与 Python 对象的兼容性
#define PY_SSIZE_T_CLEAN
#include "pandas/parser/tokenizer.h"  // 引入 pandas 解析器的头文件
#include <Python.h>  // 引入 Python.h 头文件

// 定义 PandasParser_CAPI 结构体，声明一系列函数指针作为 pandas 解析器的 API
typedef struct {
  int (*to_double)(char *, double *, char, char, int *);  // 将字符串转换为双精度浮点数
  int (*floatify)(PyObject *, double *, int *);  // 将 Python 对象转换为双精度浮点数
  void *(*new_rd_source)(PyObject *);  // 创建读取数据源的对象
  void (*del_rd_source)(void *);  // 删除读取数据源的对象
  char *(*buffer_rd_bytes)(void *, size_t, size_t *, int *, const char *);  // 从数据源中读取指定字节数的数据到缓冲区
  void (*uint_state_init)(uint_state *);  // 初始化无符号整数状态结构
  int (*uint64_conflict)(uint_state *);  // 检查无符号整数状态结构中的冲突
  void (*coliter_setup)(coliter_t *, parser_t *, int64_t, int64_t);  // 设置列迭代器
  parser_t *(*parser_new)(void);  // 创建解析器对象
  int (*parser_init)(parser_t *);  // 初始化解析器对象
  void (*parser_free)(parser_t *);  // 释放解析器对象
  void (*parser_del)(parser_t *);  // 删除解析器对象
  int (*parser_add_skiprow)(parser_t *, int64_t);  // 添加要跳过的行数
  void (*parser_set_skipfirstnrows)(parser_t *, int64_t);  // 设置要跳过的前几行
  void (*parser_set_default_options)(parser_t *);  // 设置解析器的默认选项
  int (*parser_consume_rows)(parser_t *, size_t);  // 消费指定数量的行数
  int (*parser_trim_buffers)(parser_t *);  // 修剪解析器的缓冲区
  int (*tokenize_all_rows)(parser_t *, const char *);  // 对所有行进行标记化处理
  int (*tokenize_nrows)(parser_t *, size_t, const char *);  // 对指定数量的行进行标记化处理
  int64_t (*str_to_int64)(const char *, int64_t, int64_t, int *, char);  // 将字符串转换为 int64_t
  uint64_t (*str_to_uint64)(uint_state *, const char *, int64_t, uint64_t, int *, char);  // 将字符串转换为 uint64_t
  double (*xstrtod)(const char *, char **, char, char, char, int, int *, int *);  // 将字符串转换为双精度浮点数
  double (*precise_xstrtod)(const char *, char **, char, char, char, int, int *, int *);  // 精确地将字符串转换为双精度浮点数
  double (*round_trip)(const char *, char **, char, char, char, int, int *, int *);  // 对字符串进行往返转换，返回双精度浮点数
  int (*to_boolean)(const char *, uint8_t *);  // 将字符串转换为布尔值
} PandasParser_CAPI;

// 定义 pandas 解析器的 API 封装对象名
#define PandasParser_CAPSULE_NAME "pandas._pandas_parser_CAPI"

// 如果未定义 _PANDAS_PARSER_IMPL，则声明并定义 PandasParserAPI 对象为 NULL
#ifndef _PANDAS_PARSER_IMPL
static PandasParser_CAPI *PandasParserAPI = NULL;

// 定义 PandasParserAPI 的导入宏，使用 PyCapsule_Import 导入 PandasParser_CAPSULE_NAME 对应的对象
#define PandasParser_IMPORT                                                    \
  PandasParserAPI =                                                            \
      (PandasParser_CAPI *)PyCapsule_Import(PandasParser_CAPSULE_NAME, 0)

// 定义各种函数宏，用于通过 PandasParserAPI 调用对应函数
#define to_double(item, p_value, sci, decimal, maybe_int)                      \
  PandasParserAPI->to_double((item), (p_value), (sci), (decimal), (maybe_int))
#define floatify(str, result, maybe_int)                                       \
  PandasParserAPI->floatify((str), (result), (maybe_int))
#define new_rd_source(obj) PandasParserAPI->new_rd_source((obj))
#define del_rd_source(src) PandasParserAPI->del_rd_source((src))
#define buffer_rd_bytes(source, nbytes, bytes_read, status, encoding_errors)   \
  PandasParserAPI->buffer_rd_bytes((source), (nbytes), (bytes_read), (status), \
                                   (encoding_errors))
#define uint_state_init(self) PandasParserAPI->uint_state_init((self))
#define uint64_conflict(self) PandasParserAPI->uint64_conflict((self))
#endif

#ifdef __cplusplus
}
#endif
#define coliter_setup(self, parser, i, start)                                  \
  PandasParserAPI->coliter_setup((self), (parser), (i), (start))
#define parser_new PandasParserAPI->parser_new
#define parser_init(self) PandasParserAPI->parser_init((self))
#define parser_free(self) PandasParserAPI->parser_free((self))
#define parser_del(self) PandasParserAPI->parser_del((self))
#define parser_add_skiprow(self, row)                                          \
  PandasParserAPI->parser_add_skiprow((self), (row))
#define parser_set_skipfirstnrows(self, nrows)                                 \
  PandasParserAPI->parser_set_skipfirstnrows((self), (nrows))
#define parser_set_default_options(self)                                       \
  PandasParserAPI->parser_set_default_options((self))
#define parser_consume_rows(self, nrows)                                       \
  PandasParserAPI->parser_consume_rows((self), (nrows))
#define parser_trim_buffers(self) PandasParserAPI->parser_trim_buffers((self))
#define tokenize_all_rows(self, encoding_errors)                               \
  PandasParserAPI->tokenize_all_rows((self), (encoding_errors))
#define tokenize_nrows(self, nrows, encoding_errors)                           \
  PandasParserAPI->tokenize_nrows((self), (nrows), (encoding_errors))
#define str_to_int64(p_item, int_min, int_max, error, t_sep)                   \
  PandasParserAPI->str_to_int64((p_item), (int_min), (int_max), (error),       \
                                (t_sep))
#define str_to_uint64(state, p_item, int_max, uint_max, error, t_sep)          \
  PandasParserAPI->str_to_uint64((state), (p_item), (int_max), (uint_max),     \
                                 (error), (t_sep))
#define xstrtod(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)     \
  PandasParserAPI->xstrtod((p), (q), (decimal), (sci), (tsep),                 \
                           (skip_trailing), (error), (maybe_int))
#define precise_xstrtod(p, q, decimal, sci, tsep, skip_trailing, error,        \
                        maybe_int)                                             \
  PandasParserAPI->precise_xstrtod((p), (q), (decimal), (sci), (tsep),         \
                                   (skip_trailing), (error), (maybe_int))
#define round_trip(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)  \
  PandasParserAPI->round_trip((p), (q), (decimal), (sci), (tsep),              \
                              (skip_trailing), (error), (maybe_int))
#define to_boolean(item, val) PandasParserAPI->to_boolean((item), (val))

# 如果没有定义_PANDAS_PARSER_IMPL，则定义以下宏

#ifdef __cplusplus
}
#endif
```