# `.\pytorch\torch\csrc\lazy\backend\lowering_context.cpp`

```
// 引入 Torch 库中的 LoweringContext 头文件
#include <torch/csrc/lazy/backend/lowering_context.h>

// 声明 torch 命名空间
namespace torch {
// 声明 lazy 子命名空间
namespace lazy {

// LoweringContext 类的构造函数的实现，接受名称和后端设备作为参数
LoweringContext::LoweringContext(const std::string& name, BackendDevice device)
    : device_(std::move(device)) {}

// LoweringContext 类的构造函数的实现，接受名称、后端设备、节点后序数组和发射状态映射作为参数
LoweringContext::LoweringContext(
    const std::string& name,
    BackendDevice device,
    c10::ArrayRef<const torch::lazy::Node*> post_order,
    Util::EmissionMap emit_status)
    : device_(std::move(device)), emit_status_(std::move(emit_status)) {}

// 获取参数数据的函数，返回后端数据指针向量的引用
const std::vector<BackendDataPtr>& LoweringContext::GetParametersData() const {
  return parameters_;
}

} // namespace lazy
} // namespace torch
```