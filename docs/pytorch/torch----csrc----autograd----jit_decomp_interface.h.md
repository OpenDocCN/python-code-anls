# `.\pytorch\torch\csrc\autograd\jit_decomp_interface.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <c10/macros/Export.h>

// NOTE: [Jit Decomposition Interface]
// 注释: JIT分解接口
//
// For some context of why we need this at all, see NOTE: [forward-mode AD
// decompositions mechanism]
// 为什么我们需要这个接口的背景，请参见NOTE: [forward-mode AD decompositions mechanism]

// Introducing that mechanism from the NOTE is problematic because:
// - it relies on TorchScript, so now VariableTypeX.cpp depends on TorchScript.
// - there exist internal builds like lite_trainer, which depend on VariableType
//   but do not depend on TorchScript.
//
// For internal builds like lite_trainer builds to pass, and for OSS builds that
// do depend on TorchScript to still support the forward AD decomp mechanism, we
// implement a PImpl pattern to avoid a static dependency in favor of a dynamic
// one
// - during static initialization time, if the library is built with TorchScript
//   setJitDecompImpl is called in decomposition_registry.cpp setting a global
//   ptr to the impl
// - when the program is run,if getJitDecompImpl returns a non null ptr, we can
//   carry on normally, otherwise we gracefully error out
//
// For extra context, see VariableHooksInterface.h, where a similar technique
// is used
//
// 引入上述机制的问题在于：
// - 它依赖于TorchScript，所以现在VariableTypeX.cpp依赖于TorchScript。
// - 存在像lite_trainer这样的内部构建，它们依赖于VariableType但不依赖于TorchScript。
//
// 为了让像lite_trainer这样的内部构建能够通过，并且对于那些依赖于TorchScript但仍需要支持前向AD分解机制的OSS构建，
// 我们实现了一个PImpl模式，避免静态依赖，转而采用动态依赖：
// - 在静态初始化期间，如果库使用TorchScript构建，则在decomposition_registry.cpp中调用setJitDecompImpl，
//   设置一个指向实现的全局指针
// - 当程序运行时，如果getJitDecompImpl返回一个非空指针，我们可以正常进行，否则我们将优雅地报错
//
// 更多背景信息，请参阅VariableHooksInterface.h，在那里使用了类似的技术

namespace torch::autograd::impl {

// 定义JitDecompInterface结构体，这是一个抽象基类，用于JIT分解接口
struct TORCH_API JitDecompInterface {
  virtual ~JitDecompInterface() = default;
  virtual bool has_jit_decomposition(
      const c10::FunctionSchema& schema) const = 0;
  virtual void run_jit_decomposition(
      const c10::OperatorHandle& op,
      jit::Stack* stack) const = 0;
};

// 设置JitDecompImpl的函数声明
TORCH_API void setJitDecompImpl(JitDecompInterface* impl);

// 获取JitDecompImpl的函数声明，返回JitDecompInterface指针
TORCH_API JitDecompInterface* getJitDecompImpl();

// 定义JitDecompRegisterer结构体，用于注册JIT分解接口实现
struct TORCH_API JitDecompRegisterer {
  explicit JitDecompRegisterer(JitDecompInterface* impl) {
    setJitDecompImpl(impl);
  }
};

} // namespace torch::autograd::impl
// 命名空间结束
```