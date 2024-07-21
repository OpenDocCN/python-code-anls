# `.\pytorch\aten\src\ATen\core\rref_interface.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <c10/util/intrusive_ptr.h>
// 包含 c10 库中的 intrusive_ptr 头文件
#include <ATen/core/jit_type_base.h>
// 包含 ATen 库中的 jit_type_base 头文件

namespace c10 {

struct Type;
// 声明 Type 结构体

using worker_id_t = int16_t;
// 定义 worker_id_t 类型为 int16_t

// 这个抽象类只包含用户可见的 API，并且将被 JIT 和分布式共享，以实现 TorchScript 支持。
class C10_EXPORT RRefInterface : public c10::intrusive_ptr_target {
 public:
  RRefInterface() = default;
  // 默认构造函数

  // 禁止 RRefInterface 的复制和移动，以防止引用计数混乱。
  RRefInterface(const RRefInterface& other) = delete;
  RRefInterface(RRefInterface&& other) = delete;
  RRefInterface& operator=(RRefInterface&& other) = delete;

  // 虚析构函数
  ~RRefInterface() override = default;

  // 返回所有者的 worker id
  virtual worker_id_t owner() const = 0;

  // 返回所有者的 worker 名称
  virtual std::string ownerName() const = 0;

  // 如果是 OwnerRRef，则返回 true
  virtual bool isOwner() const = 0;

  // 如果是 OwnerRRef，或者被其所有者确认的 UserRRef，则返回 true
  virtual bool confirmedByOwner() const = 0;

  // 返回 TypePtr 对象，表示对象的类型信息
  virtual const TypePtr type() const = 0;
};

}
// 命名空间 c10 结束
```