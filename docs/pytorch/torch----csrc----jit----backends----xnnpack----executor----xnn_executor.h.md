# `.\pytorch\torch\csrc\jit\backends\xnnpack\executor\xnn_executor.h`

```py
#pragma once
// 引入 XNNPACK 头文件，用于 XNNExecutor 类的实现
#include <xnnpack.h>
// 引入内存管理和向量容器的头文件
#include <memory>
#include <vector>

// Torch 的命名空间
namespace torch {
// JIT 的命名空间
namespace jit {
// XNNPACK 的委托命名空间
namespace xnnpack {
// XNNPACK 委托的具体实现命名空间
namespace delegate {

// XNNExecutor 类定义
class XNNExecutor {
 private:
  // 使用智能指针管理 XNN 运行时对象，并指定删除器
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr,
      &xnn_delete_runtime};
  // 输入张量的 ID 列表
  std::vector<uint32_t> input_ids_;
  // 输出张量的 ID 列表
  std::vector<uint32_t> output_ids_;
  // 外部值的向量，用于设置输入和输出张量
  std::vector<xnn_external_value> externals_;

 public:
  // 默认构造函数
  XNNExecutor() = default;

  // 设置输入张量和输出张量的模板方法
  template <typename T>
  bool set_inputs(std::vector<T*>& inputs, std::vector<T*>& outputs) {
    // 清空外部值向量
    externals_.clear();

    // 如果输入张量的数量与输入 ID 的数量不匹配，则返回失败
    if (inputs.size() != input_ids_.size()) {
      return false;
    }

    // 遍历输入张量，将其添加到外部值向量中
    for (int i = 0; i < inputs.size(); i++) {
      externals_.emplace_back(xnn_external_value{input_ids_[i], inputs[i]});
    }

    // 如果输出张量的数量与输出 ID 的数量不匹配，则返回失败
    if (outputs.size() != output_ids_.size()) {
      return false;
    }

    // 遍历输出张量，将其添加到外部值向量中
    for (int i = 0; i < outputs.size(); i++) {
      externals_.emplace_back(xnn_external_value{output_ids_[i], outputs[i]});
    }

    // 设置成功
    return true;
  }

  // 执行前向传播操作
  bool forward() {
    // 设置运行时的外部值
    xnn_status status =
        xnn_setup_runtime(runtime_.get(), externals_.size(), externals_.data());

    // 如果设置运行时失败，则返回失败
    if (status != xnn_status_success) {
      return false;
    }

    // 调用运行时执行前向传播操作
    status = xnn_invoke_runtime(runtime_.get());

    // 如果执行前向传播操作失败，则返回失败
    if (status != xnn_status_success) {
      return false;
    }

    // 前向传播成功
    return true;
  }

  // 声明 XNNCompiler 类为友元类，可以访问 XNNExecutor 的私有成员
  friend class XNNCompiler;
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
```