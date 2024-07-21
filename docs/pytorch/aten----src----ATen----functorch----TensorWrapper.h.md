# `.\pytorch\aten\src\ATen\functorch\TensorWrapper.h`

```
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/functorch/Macros.h>
#include <ATen/Tensor.h>
#include <ATen/functorch/Interpreter.h>

namespace at::functorch {

// NOTE: [functorch's TensorWrapper]
//
// Taking better suggestions for a name. TensorWrapper is the wrapper Tensor
// Subclass for functorch's grad-based transforms (grad, vjp, jvp). It is
// analogous to how vmap uses BatchedTensor as the wrapper Tensor subclass.
//
// If you're familiar with the Tensor-Variable merge, TensorWrapper is effectively
// another Variable.
//
// Consider grad(grad(torch.sin))(x). This wraps `x` as TensorWrapper(TensorWrapper(x)).
// The reason why is so that each TensorWrapper can hold its own AutogradMeta and
// participate in a **separate** autograd graph.
//
// There are alternative designs we could have chosen (e.g. each grad transform
// stores a weak map of Tensor -> AutogradMeta); the benefit of the TensorWrapper
// design is that we can re-use existing VariableType kernels (i.e. Autograd kernels)
// without much modification. Since a TensorWrapper looks like a regular Tensor,
// the VariableType kernel can pull out the AutogradMeta struct from where it
// expects and extend the autograd graph

// TensorWrapper 结构体定义，继承自 c10::TensorImpl
struct TORCH_API TensorWrapper : public c10::TensorImpl {
  // 构造函数，接收多个参数来初始化 TensorWrapper
  explicit TensorWrapper(
      c10::DispatchKeySet key_set,    // 分发键集合
      Tensor value,                   // 包装的 Tensor 对象
      int64_t level,                  // 级别
      std::shared_ptr<bool> is_alive, // 是否存活的共享指针
      bool is_immutable = false,      // 是否不可变
      bool use_value_sizes_strides = true); // 是否使用 value 的大小和步长

  // 刷新元数据的方法
  void refreshMetadata();

  // 返回包装的 Tensor 对象的引用
  const Tensor& value() const {
    return value_;
  }
  
  // 返回级别的可选值
  optional<int64_t> level() const {
    if (is_alive()) {
      return level_;
    }
    return {};
  }
  
  // 返回是否不可变
  bool is_immutable() const {
    return is_immutable_;
  }
  
  // 返回是否存活
  bool is_alive() const;

  // 用于自动求导的必要重写
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    // 重写父类方法，允许修改张量的元数据
    c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
        c10::VariableVersion&& version_counter,
        bool allow_tensor_metadata_change) const override;
    
    // 从另一个张量实现对象浅复制并分离数据，允许修改张量元数据
    void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;
    
    private:
    // 返回张量实现对象的类型名称
    const char* tensorimpl_type_name() const override;
    
    // 值的张量对象
    Tensor value_;
    
    // 张量对象的级别
    int64_t level_;
    
    // 指示张量对象是否不可变的布尔值
    bool is_immutable_;
    
    // TensorWrapper 接收一个布尔标志，用于指示创建它的 Grad Interpreter 是否仍然存活。
    // 如果 Grad Interpreter 不再存活，则它尝试表现得像一个普通的张量。
    //
    // 当我们退出该级别时，此包装器可能被标记为 "not alive"。
    // 不再存活的包装器：
    // 1）可能仍然具有自动求导的元数据
    // 2）将前向分发到底层值(value())
    std::shared_ptr<bool> is_alive_;
// 结束 at::functorch 命名空间

// 定义了两个重载版本的 makeTensorWrapper 函数：
// 一个接受一个整数 level 和一个可选的布尔值 is_immutable，默认为 false；
// 另一个接受一个 Interpreter 对象和一个可选的布尔值 is_immutable，默认为 false。
// 第一个版本尝试从 DynamicLayerStack 上的解释器自动获取生命周期句柄。
// 使用时需要注意：如果解释器不在 DynamicLayerStack 上，将无法找到生命周期句柄。
// 实际上，这通常不会成为问题：在 Python 中构造 TensorWrapper 时，相应的解释器通常位于堆栈上。
TORCH_API Tensor makeTensorWrapper(const Tensor& tensor, int64_t level, bool is_immutable=false);

// 第二个版本的 makeTensorWrapper 函数接受一个 Tensor 对象和一个 Interpreter 对象，
// 以及一个可选的布尔值 is_immutable，默认为 false。
TORCH_API Tensor makeTensorWrapper(const Tensor& tensor, const Interpreter& interpreter, bool is_immutable=false);

// 返回与给定 Tensor 对象关联的 TensorWrapper 指针，如果没有找到对应的 TensorWrapper，则返回空指针。
TORCH_API TensorWrapper* maybeGetTensorWrapper(const Tensor& tensor);

// 将 Tensor 对象的内容转储到给定的输出流 ss 中。
TORCH_API void dumpTensor(std::ostream & ss, const Tensor& tensor);

// 将 Tensor 对象的内容转储到标准输出流（cout）中。
TORCH_API void dumpTensorCout(const Tensor& tensor);

// 结束命名空间 at::functorch
} // namespace at::functorch
```