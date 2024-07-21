# `.\pytorch\torch\csrc\lazy\ts_backend\ops\generic.cpp`

```
#include <torch/csrc/lazy/ts_backend/ops/generic.h>

namespace torch {
namespace lazy {

// 定义 Generic 类的构造函数，继承自 TsNode 类
Generic::Generic(
    OpKind op,                      // 操作类型
    OpList operands,                // 操作数列表
    Shape shape,                    // 形状
    size_t num_outputs,             // 输出数量
    hash_t hash_seed)               // 哈希种子
    : TsNode(op, operands, {std::move(shape)}, num_outputs, hash_seed),  // 调用 TsNode 的构造函数进行初始化
      hash_seed_(hash_seed) {}      // 初始化 hash_seed 成员变量

// 定义 Generic 类的构造函数，接受一个函数对象作为形状的参数
Generic::Generic(
    OpKind op,                                      // 操作类型
    OpList operands,                                // 操作数列表
    const std::function<Shape()>& shape_fn,          // 返回形状的函数对象
    size_t num_outputs,                             // 输出数量
    hash_t hash_seed)                               // 哈希种子
    : TsNode(op, operands, shape_fn, num_outputs, hash_seed),  // 调用 TsNode 的构造函数进行初始化
      hash_seed_(hash_seed) {}                      // 初始化 hash_seed 成员变量

// 定义 Generic 类的构造函数，仅接受输出数量和哈希种子作为参数
Generic::Generic(
    OpKind op,                    // 操作类型
    OpList operands,              // 操作数列表
    size_t num_outputs,           // 输出数量
    hash_t hash_seed)             // 哈希种子
    : TsNode(op, operands, num_outputs, hash_seed),  // 调用 TsNode 的构造函数进行初始化
      hash_seed_(hash_seed) {}    // 初始化 hash_seed 成员变量

// 定义 Generic 类的构造函数，仅接受操作类型、形状、输出数量和哈希种子作为参数
Generic::Generic(
    OpKind op,                    // 操作类型
    Shape shape,                  // 形状
    size_t num_outputs,           // 输出数量
    hash_t hash_seed)             // 哈希种子
    : TsNode(op, std::move(shape), num_outputs, hash_seed),  // 调用 TsNode 的构造函数进行初始化
      hash_seed_(hash_seed) {}    // 初始化 hash_seed 成员变量

} // namespace lazy
} // namespace torch
```