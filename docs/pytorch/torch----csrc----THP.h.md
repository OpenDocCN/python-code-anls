# `.\pytorch\torch\csrc\THP.h`

```
// 如果 THP_H 未定义，则定义 THP_H 宏，避免多次包含
#ifndef THP_H
#define THP_H

// 包含 Torch 库的导出宏定义
#include <torch/csrc/Export.h>
// 包含 Python 头文件
#include <torch/csrc/python_headers.h>

// 用于向后兼容的宏定义，感谢 http://cx-oracle.sourceforge.net/
// 为 Python 3.x 定义 PyInt_* 宏。注意：必须先包含 Python.h，否则会错误地认为 PyInt_Check 未定义！
#ifndef PyInt_Check
#define PyInt_Check PyLong_Check   // 如果未定义 PyInt_Check，则定义为 PyLong_Check
#define PyInt_FromLong PyLong_FromLong   // 如果未定义 PyInt_FromLong，则定义为 PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong   // 如果未定义 PyInt_AsLong，则定义为 PyLong_AsLong
#define PyInt_Type PyLong_Type   // 如果未定义 PyInt_Type，则定义为 PyLong_Type
#endif

// 引入 Torch 异常处理相关的头文件
#include <torch/csrc/Exceptions.h>
// 引入 Torch 随机数生成器相关的头文件
#include <torch/csrc/Generator.h>
// 引入 Torch 模块相关的头文件
#include <torch/csrc/Module.h>
// 引入 Torch 大小相关的头文件
#include <torch/csrc/Size.h>
// 引入 Torch 存储相关的头文件
#include <torch/csrc/Storage.h>
// 引入 Torch 类型相关的头文件
#include <torch/csrc/Types.h>
// 引入 Torch 实用工具相关的头文件，此处需要定义了 Storage 和 Tensor 类型
#include <torch/csrc/utils.h>
// 引入 Torch 字节顺序相关的头文件
#include <torch/csrc/utils/byte_order.h>

// 引入 Torch 序列化相关的头文件
#include <torch/csrc/serialization.h>

// 引入 Torch 自动求导 Python 接口相关的头文件
#include <torch/csrc/autograd/python_autograd.h>

// 结束 THP_H 的宏定义
#endif
```