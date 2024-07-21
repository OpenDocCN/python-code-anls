# `.\pytorch\torch\csrc\distributed\c10d\default_comm_hooks.hpp`

```
#pragma once
// 引入 Torch 的分布式通信相关头文件
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/comm.hpp>

// 定义 c10d 命名空间
namespace c10d {

// 枚举类型，用于表示内置通信钩子的类型
enum class BuiltinCommHookType : uint8_t {
  ALLREDUCE = 1,       // 全局归约通信钩子
  FP16_COMPRESS = 2,   // FP16 压缩通信钩子
};

// AllReduceCommHook 类，继承自 CppCommHookInterface，用于处理全局归约通信钩子
class AllReduceCommHook
    : public CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>> {
 public:
  // 显式构造函数，接受 ProcessGroup 对象作为参数
  explicit AllReduceCommHook(const c10::intrusive_ptr<ProcessGroup>& state)
      : CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>>(state) {}

  // 虚析构函数，用于释放资源
  ~AllReduceCommHook() override = default;

  // 重写父类的虚函数，运行全局归约通信钩子，返回 Future 对象
  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

// FP16CompressCommHook 类，继承自 CppCommHookInterface，用于处理 FP16 压缩通信钩子
class FP16CompressCommHook
    : public CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>> {
 public:
  // 显式构造函数，接受 ProcessGroup 对象作为参数
  explicit FP16CompressCommHook(const c10::intrusive_ptr<ProcessGroup>& state)
      : CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>>(state) {}

  // 虚析构函数，用于释放资源
  ~FP16CompressCommHook() override = default;

  // 重写父类的虚函数，运行 FP16 压缩通信钩子，返回 Future 对象
  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

// _AllReduceBySumCommHook 类，继承自 CppCommHookInterface，用于内部使用的优化全局归约通信钩子
// 与 AllReduceCommHook 几乎相同，但在钩子内部没有除法操作
class _AllReduceBySumCommHook
    : public CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>> {
 public:
  // 显式构造函数，接受 ProcessGroup 对象作为参数
  explicit _AllReduceBySumCommHook(
      const c10::intrusive_ptr<ProcessGroup>& state)
      : CppCommHookInterface<c10::intrusive_ptr<ProcessGroup>>(state) {}

  // 虚析构函数，用于释放资源
  ~_AllReduceBySumCommHook() override = default;

  // 重写父类的虚函数，运行优化的全局归约通信钩子，返回 Future 对象
  c10::intrusive_ptr<c10::ivalue::Future> runHook(GradBucket& bucket) override;
};

} // namespace c10d
```