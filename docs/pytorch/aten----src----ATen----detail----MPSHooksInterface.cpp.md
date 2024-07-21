# `.\pytorch\aten\src\ATen\detail\MPSHooksInterface.cpp`

```py
//  Copyright © 2022 Apple Inc.
// 包含 ATen 库的 MPSHooksInterface 头文件
#include <ATen/detail/MPSHooksInterface.h>
// 包含 C10 库的 CallOnce 头文件
#include <c10/util/CallOnce.h>

// 命名空间 at 开始
namespace at {
// 命名空间 detail 开始
namespace detail {

// 返回 MPSHooksInterface 类型的引用，用于获取 MPS 相关的钩子函数
const MPSHooksInterface& getMPSHooks() {
  // 静态变量，存储 MPSHooksInterface 对象的唯一指针
  static std::unique_ptr<MPSHooksInterface> mps_hooks;
  // 如果不是在 C10_MOBILE 环境下
  #if !defined C10_MOBILE
  // 静态变量，确保 call_once 只调用一次的标志
  static c10::once_flag once;
  // 调用 call_once，初始化 MPSHooksInterface 对象
  c10::call_once(once, [] {
    // 通过 MPSHooksRegistry 创建 MPSHooksInterface 对象，传入名称 "MPSHooks" 和参数 MPSHooksArgs{}
    mps_hooks = MPSHooksRegistry()->Create("MPSHooks", MPSHooksArgs{});
    // 如果未成功创建对象，则创建一个新的空的 MPSHooksInterface 对象
    if (!mps_hooks) {
      mps_hooks = std::make_unique<MPSHooksInterface>();
    }
  });
  // 如果在 C10_MOBILE 环境下，且 mps_hooks 为空，则创建一个新的 MPSHooksInterface 对象
  #else
  if (mps_hooks == nullptr) {
    mps_hooks = std::make_unique<MPSHooksInterface>();
  }
  #endif
  // 返回 MPSHooksInterface 对象的引用
  return *mps_hooks;
}

} // namespace detail

// 定义 MPSHooksRegistry 注册表，注册 MPSHooksInterface 类型和 MPSHooksArgs 参数
C10_DEFINE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs)

} // namespace at
// 命名空间 at 结束
```