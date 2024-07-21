# `.\pytorch\aten\src\ATen\test\reportMemoryUsage.h`

```
#pragma once
// 使用预处理指令#pragma once，确保头文件只被包含一次

#include <ATen/ATen.h>
// 包含 ATen 库的头文件

#include <c10/core/Allocator.h>
// 包含 c10 核心模块中的 Allocator 头文件

#include <c10/util/ThreadLocalDebugInfo.h>
// 包含 c10 工具模块中的 ThreadLocalDebugInfo 头文件

class TestMemoryReportingInfo : public c10::MemoryReportingInfoBase {
// 定义一个类 TestMemoryReportingInfo，继承自 c10::MemoryReportingInfoBase

 public:
  struct Record {
    void* ptr;                      // 记录指针地址
    int64_t alloc_size;             // 记录分配大小
    size_t total_allocated;         // 记录总分配量
    size_t total_reserved;          // 记录总保留量
    c10::Device device;             // 记录设备信息
  };

  std::vector<Record> records;      // 存储 Record 结构体的向量

  TestMemoryReportingInfo() = default;        // 默认构造函数
  ~TestMemoryReportingInfo() override = default;  // 虚析构函数

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      size_t total_allocated,
      size_t total_reserved,
      c10::Device device) override {
    // 实现基类纯虚函数 reportMemoryUsage，将参数记录为新的 Record 并加入 records
    records.emplace_back(
        Record{ptr, alloc_size, total_allocated, total_reserved, device});
  }

  bool memoryProfilingEnabled() const override {
    // 实现基类纯虚函数 memoryProfilingEnabled，始终返回 true 表示内存分析已启用
    return true;
  }

  Record getLatestRecord() {
    // 返回最新记录的 Record 对象
    return records.back();
  }
};
```