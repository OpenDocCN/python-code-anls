# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\csc.cxx`

```
# 定义宏以防止数组导入
#define NO_IMPORT_ARRAY
# 定义用于SciPy稀疏工具库的数组唯一符号
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_sparsetools_ARRAY_API

# 包含稀疏工具库的头文件sparsetools.h
#include "sparsetools.h"
# 包含压缩列格式（CSC）的头文件csc.h

# 使用C语言接口引入CSC的实现文件csc_impl.h
extern "C" {
#include "csc_impl.h"
}
```