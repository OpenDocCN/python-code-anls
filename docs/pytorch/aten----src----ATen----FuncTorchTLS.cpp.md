# `.\pytorch\aten\src\ATen\FuncTorchTLS.cpp`

```
// 包含 FuncTorchTLS.h 文件，提供了 ATen 库中的 FuncTorchTLS 相关功能
#include <ATen/FuncTorchTLS.h>

// 定义了 at::functorch 命名空间，用于封装 functorch 相关功能
namespace at::functorch {

// 匿名命名空间，用于定义线程局部变量 kFuncTorchTLS
namespace {

// 线程局部变量，保存了 FuncTorchTLSBase 类的唯一指针，默认为空指针
thread_local std::unique_ptr<FuncTorchTLSBase> kFuncTorchTLS = nullptr;

}

// 返回当前 kFuncTorchTLS 指针指向对象的深拷贝
std::unique_ptr<FuncTorchTLSBase> getCopyOfFuncTorchTLS() {
  // 如果 kFuncTorchTLS 指针为空，返回空指针
  if (kFuncTorchTLS == nullptr) {
    return nullptr;
  }
  // 返回 kFuncTorchTLS 指针指向对象的深拷贝
  return kFuncTorchTLS->deepcopy();
}

// 设置 kFuncTorchTLS 指针指向的对象为给定状态对象的深拷贝
void setFuncTorchTLS(const std::shared_ptr<const FuncTorchTLSBase>& state) {
  // 如果状态对象为空指针，直接将 kFuncTorchTLS 置为空指针并返回
  if (state == nullptr) {
    kFuncTorchTLS = nullptr;
    return;
  }
  // 将 kFuncTorchTLS 指针指向给定状态对象的深拷贝
  kFuncTorchTLS = state->deepcopy();
}

// 返回 kFuncTorchTLS 指针的引用
std::unique_ptr<FuncTorchTLSBase>& functorchTLSAccessor() {
  return kFuncTorchTLS;
}

} // namespace at::functorch
```