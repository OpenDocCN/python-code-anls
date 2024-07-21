# `.\pytorch\torch\csrc\jit\mobile\prim_ops_registery.cpp`

```
# 包含 Torch 框架中移动端原语操作注册相关的头文件
#include <torch/csrc/jit/mobile/prim_ops_registery.h>

namespace torch {
namespace jit {
namespace mobile {

# 定义静态函数，返回一个映射表，将字符串映射到处理堆栈操作的函数
static std::unordered_map<std::string, std::function<void(Stack&)>>&
primOpsFnTable() {
  # 静态局部变量，存储原语操作函数名到处理堆栈操作的函数的映射表
  static std::unordered_map<std::string, std::function<void(Stack&)>>
      prim_ops_fn;
  return prim_ops_fn;
}

# 注册一个原语操作函数，将函数映射到给定名称
void registerPrimOpsFunction(
    const std::string& name,
    const std::function<void(Stack&)>& fn) {
  # 将给定名称的原语操作函数注册到映射表中
  primOpsFnTable()[name] = fn;
}

# 检查给定名称的原语操作函数是否存在
bool hasPrimOpsFn(const std::string& name) {
  return primOpsFnTable().count(name);
}

# 获取给定名称的原语操作函数
std::function<void(Stack&)>& getPrimOpsFn(const std::string& name) {
  # 检查给定名称的原语操作函数是否存在，若不存在则抛出错误
  TORCH_CHECK(
      hasPrimOpsFn(name),
      "Prim Ops Function for ",
      name,
      " is not promoted yet.");
  return primOpsFnTable()[name];
}

} // namespace mobile
} // namespace jit
} // namespace torch
```