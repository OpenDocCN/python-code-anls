# `.\pytorch\aten\src\ATen\FuncTorchTLS.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <c10/macros/Macros.h>
// 包含 C10 库中的宏定义

#include <memory>
// 包含内存管理相关的标准库头文件

namespace at::functorch {
// 进入 at::functorch 命名空间

// NOTE [functorch TLS in pytorch/pytorch]
// 注意事项：在 pytorch/pytorch 中处理 functorch 的线程本地存储（TLS）问题

//
// functorch lives out-of-tree. However, it has some TLS that needs to be
// propagated. The solution for that is we store a pointer to the TLS
// inside pytorch/pytorch and extend FuncTorchTLSBase inside functorch to
// include whatever functorch needs.
//
// functorch 位于树外。然而，它具有一些需要传播的 TLS。
// 解决方案是我们在 pytorch/pytorch 中存储 TLS 的指针，并在 functorch 中扩展 FuncTorchTLSBase，
// 以包括 functorch 需要的任何内容。

// We need to store a pointer due to the indirection:
// inside functorch, we will create a subclass of FunctorchTLSBase called
// FuncTorchTLSImpl that actually contains metadata, like the DynamicLayerStack.
// FuncTorchTLSBase doesn't have any metadata because it hasn't been defined
// yet.
//
// 由于间接性，我们需要存储一个指针：
// 在 functorch 中，我们将创建一个称为 FuncTorchTLSImpl 的 FunctorchTLSBase 子类，
// 它实际上包含元数据，如 DynamicLayerStack。
// FuncTorchTLSBase 没有任何元数据，因为它尚未定义。

struct TORCH_API FuncTorchTLSBase {
  // functorch 的线程本地存储基类

  virtual ~FuncTorchTLSBase() = default;
  // 虚析构函数

  virtual std::unique_ptr<FuncTorchTLSBase> deepcopy() const = 0;
  // 虚函数：执行深拷贝并返回 functorch 的 TLS 副本

  virtual int64_t checkSupportsSingleLevelAutogradFunction() const = 0;
  // 虚函数：检查是否支持单级自动求导函数

  virtual void checkSupportsCppAutogradFunction() const = 0;
  // 虚函数：检查是否支持 C++ 自动求导函数

  virtual void checkSupportsInplaceRequiresGrad() const = 0;
  // 虚函数：检查是否支持原位操作需要梯度

  virtual void checkSupportsRetainGrad() const = 0;
  // 虚函数：检查是否支持保留梯度
};

// returns deepcopy of the functorch tls
// 返回 functorch 的 TLS 的深拷贝
TORCH_API std::unique_ptr<FuncTorchTLSBase> getCopyOfFuncTorchTLS();

// sets the functorch tls. always does a deep copy.
// 设置 functorch 的 TLS，总是执行深拷贝
TORCH_API void setFuncTorchTLS(
    const std::shared_ptr<const FuncTorchTLSBase>& state);

// get a mutable reference to the functorch tls
// 获取 functorch 的 TLS 的可变引用
TORCH_API std::unique_ptr<FuncTorchTLSBase>& functorchTLSAccessor();

} // namespace at::functorch
// 结束 at::functorch 命名空间
```