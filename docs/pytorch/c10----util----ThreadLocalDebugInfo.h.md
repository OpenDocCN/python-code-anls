# `.\pytorch\c10\util\ThreadLocalDebugInfo.h`

```
#pragma once


// 使用 pragma once 指令确保头文件只被编译一次，防止多重包含



#include <c10/macros/Export.h>


// 包含 Export.h 头文件，该文件可能包含一些宏定义和导出声明



#include <cstdint>
#include <memory>


// 包含标准 C++ 头文件 <cstdint> 和 <memory>
// <cstdint> 提供了固定大小的整数类型，<memory> 提供了智能指针等内存管理工具



namespace c10 {


// 进入 c10 命名空间，所有的类和函数定义都在该命名空间内



enum class C10_API_ENUM DebugInfoKind : uint8_t {
  PRODUCER_INFO = 0,
  MOBILE_RUNTIME_INFO,
  PROFILER_STATE,
  INFERENCE_CONTEXT, // for inference usage
  PARAM_COMMS_INFO,

  TEST_INFO, // used only in tests
  TEST_INFO_2, // used only in tests
};


// 定义枚举类型 DebugInfoKind，表示调试信息的不同种类
// 各种调试信息的标识符和注释说明在枚举常量后面



class C10_API DebugInfoBase {
 public:
  DebugInfoBase() = default;
  virtual ~DebugInfoBase() = default;
};


// 定义调试信息基类 DebugInfoBase
// 公共接口包括默认构造函数和虚析构函数



class C10_API ThreadLocalDebugInfo {
 public:
  static DebugInfoBase* get(DebugInfoKind kind);

  // 获取当前的 ThreadLocalDebugInfo
  static std::shared_ptr<ThreadLocalDebugInfo> current();

  // 内部使用，通过 DebugInfoGuard/ThreadLocalStateGuard 设置 DebugInfo
  static void _forceCurrentDebugInfo(
      std::shared_ptr<ThreadLocalDebugInfo> info);

  // 推入指定类型的调试信息结构体
  static void _push(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

  // 弹出调试信息，如果最后推入的调试信息不是指定类型，则抛出异常
  static std::shared_ptr<DebugInfoBase> _pop(DebugInfoKind kind);

  // 查看调试信息，如果最后推入的调试信息不是指定类型，则抛出异常
  static std::shared_ptr<DebugInfoBase> _peek(DebugInfoKind kind);

 private:
  std::shared_ptr<DebugInfoBase> info_;
  DebugInfoKind kind_;
  std::shared_ptr<ThreadLocalDebugInfo> parent_info_;

  friend class DebugInfoGuard;
};


// 定义 ThreadLocalDebugInfo 类
// 提供静态方法用于管理线程局部的调试信息
// 包括获取当前的调试信息、强制设置当前调试信息、推入、弹出和查看特定类型的调试信息
// 其中包括一个内部共享指针 info_、调试信息类型 kind_ 和父线程调试信息的共享指针 parent_info_
// DebugInfoGuard 类是其友元类，可以访问其私有成员



class C10_API DebugInfoGuard {
 public:
  // 构造函数，设置指定类型的调试信息
  DebugInfoGuard(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

  // 构造函数，设置 ThreadLocalDebugInfo 的调试信息
  explicit DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfo> info);

  // 析构函数，退出作用域时恢复原调试信息
  ~DebugInfoGuard();

  // 禁止拷贝构造函数和移动构造函数
  DebugInfoGuard(const DebugInfoGuard&) = delete;
  DebugInfoGuard(DebugInfoGuard&&) = delete;

 private:
  bool active_ = false;
  std::shared_ptr<ThreadLocalDebugInfo> prev_info_ = nullptr;
};


// 定义 DebugInfoGuard 类
// 用于设置调试信息的作用域保护对象
// 可以通过构造函数设置指定类型的调试信息，也可以设置 ThreadLocalDebugInfo 的调试信息
// 在析构函数中恢复之前的调试信息，保证调试信息的语义上的不可变性
// 禁止了拷贝构造函数和移动构造函数



} // namespace c10


// 结束 c10 命名空间
```