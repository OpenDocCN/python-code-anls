# `.\pytorch\torch\csrc\jit\mobile\prim_ops_registery.h`

```
#pragma once

# 定义预处理指令，确保本文件仅被编译一次


#include <ATen/core/ivalue.h>
#include <functional>
#include <vector>

# 包含头文件，用于支持 ATen 库中的 IValue 类、函数对象和向量容器


namespace torch {
namespace jit {
namespace mobile {

# 命名空间 torch::jit::mobile，用于封装相关功能


using Stack = std::vector<c10::IValue>;

# 定义别名 Stack 为存储 c10::IValue 对象的向量类型


void registerPrimOpsFunction(
    const std::string& name,
    const std::function<void(Stack&)>& fn);

# 声明函数 registerPrimOpsFunction，用于注册原始操作函数，接受函数名和函数对象作为参数


bool hasPrimOpsFn(const std::string& name);

# 声明函数 hasPrimOpsFn，用于检查是否存在特定名称的原始操作函数


std::function<void(Stack&)>& getPrimOpsFn(const std::string& name);

# 声明函数 getPrimOpsFn，用于获取特定名称的原始操作函数对象的引用


class prim_op_fn_register {
 public:
  prim_op_fn_register(
      const std::string& name,
      const std::function<void(Stack&)>& fn) {
    registerPrimOpsFunction(name, fn);
  }
};

# 定义类 prim_op_fn_register，用于在构造函数中注册原始操作函数，以名称和函数对象作为参数


} // namespace mobile
} // namespace jit
} // namespace torch

# 命名空间结束符，分别结束 torch、jit 和 mobile 命名空间
```