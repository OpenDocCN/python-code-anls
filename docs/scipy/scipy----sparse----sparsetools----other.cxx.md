# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\other.cxx`

```
// 定义宏，禁止导入数组
#define NO_IMPORT_ARRAY
// 定义 Python 数组 API 的唯一符号
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_sparsetools_ARRAY_API

// 包含稀疏工具头文件
#include "sparsetools.h"
// 包含 DIA 存储格式头文件
#include "dia.h"
// 包含图论算法头文件
#include "csgraph.h"
// 包含 COO 存储格式头文件
#include "coo.h"

// 使用 C 语言编写的其他实现的头文件声明
extern "C" {
#include "other_impl.h"
}
```