# `.\pytorch\aten\src\ATen\detail\PrivateUse1HooksInterface.cpp`

```
// 包含 ATen 库的私有使用接口头文件
#include <ATen/detail/PrivateUse1HooksInterface.h>

// ATen 命名空间
namespace at {

// 私有使用接口钩子对象指针，初始为 nullptr
static PrivateUse1HooksInterface* privateuse1_hooks = nullptr;

// 用于保护私有使用接口钩子的互斥锁对象
static std::mutex _hooks_mutex_lock;

// 注册私有使用接口钩子的全局函数，只能注册一次
TORCH_API void RegisterPrivateUse1HooksInterface(at::PrivateUse1HooksInterface* hook_) {
  // 获取互斥锁，保证线程安全
  std::lock_guard<std::mutex> lock(_hooks_mutex_lock);
  // 检查是否已经注册过私有使用接口钩子
  TORCH_CHECK(privateuse1_hooks == nullptr, "PrivateUse1HooksInterface only could be registered once.");
  // 将传入的钩子对象赋值给全局钩子指针
  privateuse1_hooks = hook_;
}

// 获取当前注册的私有使用接口钩子的全局函数
TORCH_API at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface() {
  // 检查私有使用接口钩子是否已经注册，否则抛出异常提示
  TORCH_CHECK(
      privateuse1_hooks != nullptr,
      "Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.");
  // 返回当前注册的私有使用接口钩子对象指针
  return privateuse1_hooks;
}

// 检查是否已经注册私有使用接口钩子的函数
TORCH_API bool isPrivateUse1HooksRegistered() {
  // 返回私有使用接口钩子是否不为 nullptr，即是否已经注册
  return privateuse1_hooks != nullptr;
}

// ATen 命名空间中的 detail 命名空间
namespace detail {

// 获取当前注册的私有使用接口钩子的引用
TORCH_API const at::PrivateUse1HooksInterface& getPrivateUse1Hooks() {
  // 检查私有使用接口钩子是否已经注册，否则抛出异常提示
  TORCH_CHECK(
      privateuse1_hooks != nullptr,
      "Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.");
  // 返回当前注册的私有使用接口钩子对象的引用
  return *privateuse1_hooks;
}

} // namespace detail

} // namespace at
```