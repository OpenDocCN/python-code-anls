# `.\pytorch\aten\src\ATen\test\reportMemoryUsage_test.cpp`

```py
#include <ATen/test/reportMemoryUsage.h>  // 包含用于内存使用报告的头文件

#include <gtest/gtest.h>  // 包含 Google Test 的头文件

#include <c10/core/CPUAllocator.h>  // 包含 C10 库中 CPU 分配器的头文件

TEST(DefaultCPUAllocator, check_reporter) {  // 定义名为 DefaultCPUAllocator 的测试用例
  auto reporter = std::make_shared<TestMemoryReportingInfo>();  // 创建共享指针 reporter，指向 TestMemoryReportingInfo 的实例
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PROFILER_STATE, reporter);  // 设置调试信息保护，用于性能分析状态，关联 reporter

  auto allocator = c10::GetCPUAllocator();  // 获取 CPU 分配器的实例

  auto alloc1 = allocator->allocate(42);  // 分配 42 字节内存，alloc1 是分配的内存的指针包装器
  auto r = reporter->getLatestRecord();  // 获取最新的内存使用记录
  EXPECT_EQ(alloc1.get(), r.ptr);  // 检查分配的指针与记录中的指针是否一致
  EXPECT_EQ(42, r.alloc_size);  // 检查分配大小是否为 42 字节
  EXPECT_EQ(42, r.total_allocated);  // 检查总分配量是否为 42 字节
  EXPECT_EQ(0, r.total_reserved);  // 检查总保留量是否为 0
  EXPECT_TRUE(r.device.is_cpu());  // 检查设备是否为 CPU

  auto alloc2 = allocator->allocate(1038);  // 分配 1038 字节内存，alloc2 是分配的内存的指针包装器
  r = reporter->getLatestRecord();  // 获取最新的内存使用记录
  EXPECT_EQ(alloc2.get(), r.ptr);  // 检查分配的指针与记录中的指针是否一致
  EXPECT_EQ(1038, r.alloc_size);  // 检查分配大小是否为 1038 字节
  EXPECT_EQ(1080, r.total_allocated);  // 检查总分配量是否为 1080 字节
  EXPECT_EQ(0, r.total_reserved);  // 检查总保留量是否为 0
  EXPECT_TRUE(r.device.is_cpu());  // 检查设备是否为 CPU

  auto alloc1_ptr = alloc1.get();  // 获取 alloc1 的原始指针
  alloc1.clear();  // 清空 alloc1 所管理的内存
  r = reporter->getLatestRecord();  // 获取最新的内存使用记录
  EXPECT_EQ(alloc1_ptr, r.ptr);  // 检查释放的指针与记录中的指针是否一致
  EXPECT_EQ(-42, r.alloc_size);  // 检查释放大小是否为 -42 字节（负数表示释放）
  EXPECT_EQ(1038, r.total_allocated);  // 检查总分配量是否为 1038 字节
  EXPECT_EQ(0, r.total_reserved);  // 检查总保留量是否为 0
  EXPECT_TRUE(r.device.is_cpu());  // 检查设备是否为 CPU

  auto alloc2_ptr = alloc2.get();  // 获取 alloc2 的原始指针
  alloc2.clear();  // 清空 alloc2 所管理的内存
  r = reporter->getLatestRecord();  // 获取最新的内存使用记录
  EXPECT_EQ(alloc2_ptr, r.ptr);  // 检查释放的指针与记录中的指针是否一致
  EXPECT_EQ(-1038, r.alloc_size);  // 检查释放大小是否为 -1038 字节（负数表示释放）
  EXPECT_EQ(0, r.total_allocated);  // 检查总分配量是否为 0
  EXPECT_EQ(0, r.total_reserved);  // 检查总保留量是否为 0
  EXPECT_TRUE(r.device.is_cpu());  // 检查设备是否为 CPU
}
```