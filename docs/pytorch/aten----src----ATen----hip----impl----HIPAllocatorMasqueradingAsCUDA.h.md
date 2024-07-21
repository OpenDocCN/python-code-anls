# `.\pytorch\aten\src\ATen\hip\impl\HIPAllocatorMasqueradingAsCUDA.h`

```
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

// 使用 c10::hip 命名空间使 hipification 更容易，因为无需额外修复命名空间。抱歉！
namespace c10 { namespace hip {

// 接受一个有效的 HIPAllocator（任意类型）并将其转换为假装是 CUDA 分配器的分配器。
// 参见注释 [Masquerading as CUDA]
class HIPAllocatorMasqueradingAsCUDA final : public Allocator {
  Allocator* allocator_; // 指向底层实际分配器的指针

public:
  explicit HIPAllocatorMasqueradingAsCUDA(Allocator* allocator)
    : allocator_(allocator) {} // 构造函数，接受一个 Allocator 指针作为参数，并存储在成员变量中

  // 重写 Allocator 类的 allocate 方法
  DataPtr allocate(size_t size) override {
    // 调用底层实际分配器的 allocate 方法，得到数据指针 DataPtr
    DataPtr r = allocator_->allocate(size);
    // 将 DataPtr 的设备类型设置为 CUDA，并保持其设备索引不变
    r.unsafe_set_device(Device(c10::DeviceType::CUDA, r.device().index()));
    // 返回修改后的 DataPtr
    return r;
  }

  // 重写 Allocator 类的 raw_deleter 方法
  DeleterFnPtr raw_deleter() const override {
    // 返回底层实际分配器的 raw_deleter 方法
    return allocator_->raw_deleter();
  }

  // 实现 Allocator 类的 copy_data 方法
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    // 调用底层实际分配器的 copy_data 方法，执行数据拷贝操作
    allocator_->copy_data(dest, src, count);
  }
};

}} // namespace c10::hip
```