# `.\numpy\numpy\_core\src\multiarray\textreading\tokenize.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_TOKENIZE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_TOKENIZE_H_

#include <Python.h>
#include "numpy/ndarraytypes.h"

#include "textreading/stream.h"
#include "textreading/parser_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// 枚举定义不同的解析状态
typedef enum {
    /* Initialization of fields */
    TOKENIZE_INIT,                  // 初始化字段
    TOKENIZE_CHECK_QUOTED,          // 检查引号状态
    /* Main field parsing states */
    TOKENIZE_UNQUOTED,              // 未引用字段解析状态
    TOKENIZE_UNQUOTED_WHITESPACE,
```