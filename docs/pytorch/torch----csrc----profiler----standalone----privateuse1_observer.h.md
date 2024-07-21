# `.\pytorch\torch\csrc\profiler\standalone\privateuse1_observer.h`

```
#pragma once
// 预处理指令：确保头文件只被包含一次

#include <torch/csrc/profiler/api.h>
// 包含 Torch Profiler API 头文件

namespace torch::profiler::impl {
// 进入 torch::profiler::impl 命名空间

using CallBackFnPtr = void (*)(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes);
// 定义回调函数指针类型 CallBackFnPtr，用于指向特定函数签名的函数指针

struct PushPRIVATEUSE1CallbacksStub {
  // 定义结构体 PushPRIVATEUSE1CallbacksStub

  PushPRIVATEUSE1CallbacksStub() = default;
  // 默认构造函数

  PushPRIVATEUSE1CallbacksStub(const PushPRIVATEUSE1CallbacksStub&) = delete;
  // 删除拷贝构造函数

  PushPRIVATEUSE1CallbacksStub& operator=(const PushPRIVATEUSE1CallbacksStub&) =
      delete;
  // 删除赋值运算符

  template <typename... ArgTypes>
  void operator()(ArgTypes&&... args) {
    // 模板函数：调用操作符重载，调用 push_privateuse1_callbacks_fn 指向的函数
    return (*push_privateuse1_callbacks_fn)(std::forward<ArgTypes>(args)...);
  }

  void set_privateuse1_dispatch_ptr(CallBackFnPtr fn_ptr) {
    // 设置私有使用1的回调函数指针
    push_privateuse1_callbacks_fn = fn_ptr;
  }

 private:
  CallBackFnPtr push_privateuse1_callbacks_fn = nullptr;
  // 私有成员变量：私有使用1的回调函数指针，默认为空指针
};

extern TORCH_API struct PushPRIVATEUSE1CallbacksStub
    pushPRIVATEUSE1CallbacksStub;
// 声明外部全局变量：PushPRIVATEUSE1CallbacksStub 结构体的实例 pushPRIVATEUSE1CallbacksStub

struct RegisterPRIVATEUSE1Observer {
  // 定义结构体 RegisterPRIVATEUSE1Observer

  RegisterPRIVATEUSE1Observer(
      PushPRIVATEUSE1CallbacksStub& stub,
      CallBackFnPtr value) {
    // 构造函数：注册私有使用1的观察者
    stub.set_privateuse1_dispatch_ptr(value);
    // 调用 PushPRIVATEUSE1CallbacksStub 的设置函数，设置私有使用1的回调函数指针
  }
};

#define REGISTER_PRIVATEUSE1_OBSERVER(name, fn) \
  static RegisterPRIVATEUSE1Observer name##__register(name, fn);
// 宏定义：注册私有使用1的观察者，创建静态 RegisterPRIVATEUSE1Observer 的实例 name##__register

} // namespace torch::profiler::impl
// 结束 torch::profiler::impl 命名空间
```