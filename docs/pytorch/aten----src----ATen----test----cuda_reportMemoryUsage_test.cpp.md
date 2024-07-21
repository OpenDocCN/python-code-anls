# `.\pytorch\aten\src\ATen\test\cuda_reportMemoryUsage_test.cpp`

```
#include <ATen/test/reportMemoryUsage.h>  // 包含内存使用报告相关的头文件

#include <gtest/gtest.h>  // 包含 Google 测试框架的头文件

#include <c10/cuda/CUDACachingAllocator.h>  // 包含 CUDA 缓存分配器的头文件

TEST(DeviceCachingAllocator, check_reporter) {  // 定义测试案例 DeviceCachingAllocator.check_reporter
  auto reporter = std::make_shared<TestMemoryReportingInfo>();  // 创建共享指针 reporter，指向 TestMemoryReportingInfo 对象
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PROFILER_STATE, reporter);  // 使用 DebugInfoGuard 监视 PROFILER_STATE 信息，关联 reporter

  auto _200kb = 200 * 1024;  // 定义变量 _200kb，值为 200 KB
  auto _500mb = 500 * 1024 * 1024;  // 定义变量 _500mb，值为 500 MB

  auto allocator = c10::cuda::CUDACachingAllocator::get();  // 获取 CUDA 缓存分配器的实例

  auto alloc1 = allocator->allocate(_200kb);  // 分配 200 KB 内存
  auto r = reporter->getLatestRecord();  // 获取最新的内存使用记录
  EXPECT_EQ(alloc1.get(), r.ptr);  // 断言分配的内存地址与记录中的地址相等
  EXPECT_LE(_200kb, r.alloc_size);  // 断言分配的内存大小不小于记录中的分配大小
  EXPECT_LE(_200kb, r.total_allocated);  // 断言总分配的内存不小于记录中的总分配大小
  EXPECT_LE(_200kb, r.total_reserved);  // 断言总保留的内存不小于记录中的总保留大小
  EXPECT_TRUE(r.device.is_cuda());  // 断言记录中的设备为 CUDA 设备

  auto alloc1_true_ptr = r.ptr;  // 记录分配的真实指针地址
  auto alloc1_true_alloc_size = r.alloc_size;  // 记录分配的真实分配大小

  // I bet pytorch will not waste that much memory
  EXPECT_LT(r.total_allocated, 2 * _200kb);  // 断言总分配的内存不超过两倍的 _200kb
  // I bet pytorch will not reserve that much memory
  EXPECT_LT(r.total_reserved, _500mb);  // 断言总保留的内存不超过 _500mb

  auto alloc2 = allocator->allocate(_500mb);  // 分配 500 MB 内存
  r = reporter->getLatestRecord();  // 获取最新的内存使用记录
  EXPECT_EQ(alloc2.get(), r.ptr);  // 断言分配的内存地址与记录中的地址相等
  EXPECT_LE(_500mb, r.alloc_size);  // 断言分配的内存大小不小于记录中的分配大小
  EXPECT_LE(_200kb + _500mb, r.total_allocated);  // 断言总分配的内存不小于之前的总分配大小加上当前分配的大小
  EXPECT_LE(_200kb + _500mb, r.total_reserved);  // 断言总保留的内存不小于之前的总保留大小加上当前分配的大小
  EXPECT_TRUE(r.device.is_cuda());  // 断言记录中的设备为 CUDA 设备
  auto alloc2_true_ptr = r.ptr;  // 记录分配的真实指针地址
  auto alloc2_true_alloc_size = r.alloc_size;  // 记录分配的真实分配大小

  auto max_reserved = r.total_reserved;  // 记录当前最大的总保留大小

  alloc1.clear();  // 释放 alloc1 分配的内存
  r = reporter->getLatestRecord();  // 获取最新的内存使用记录
  EXPECT_EQ(alloc1_true_ptr, r.ptr);  // 断言释放的内存地址与记录中的地址相等
  EXPECT_EQ(-alloc1_true_alloc_size, r.alloc_size);  // 断言释放的内存大小为分配大小的负数
  EXPECT_EQ(alloc2_true_alloc_size, r.total_allocated);  // 断言总分配的内存等于 alloc2 的分配大小
  // alloc2 保留不变，这是一个释放内存的操作，因此不应该增加总保留内存
  EXPECT_TRUE(
      alloc2_true_alloc_size <= static_cast<int64_t>(r.total_reserved) &&
      r.total_reserved <= max_reserved);  // 断言总保留的内存在之前最大保留的范围内
  EXPECT_TRUE(r.device.is_cuda());  // 断言记录中的设备为 CUDA 设备

  alloc2.clear();  // 释放 alloc2 分配的内存
  r = reporter->getLatestRecord();  // 获取最新的内存使用记录
  EXPECT_EQ(alloc2_true_ptr, r.ptr);  // 断言释放的内存地址与记录中的地址相等
  EXPECT_EQ(-alloc2_true_alloc_size, r.alloc_size);  // 断言释放的内存大小为分配大小的负数
  EXPECT_EQ(0, r.total_allocated);  // 断言总分配的内存为 0
  EXPECT_TRUE(r.total_reserved <= max_reserved);  // 断言总保留的内存不超过之前的最大保留值
  EXPECT_TRUE(r.device.is_cuda());  // 断言记录中的设备为 CUDA 设备
}

int main(int argc, char* argv[]) {  // 程序入口函数，接受命令行参数
  ::testing::InitGoogleTest(&argc, argv);  // 初始化 Google 测试框架
  c10::cuda::CUDACachingAllocator::init(1);  // 初始化 CUDA 缓存分配器，使用设备编号 1
  return RUN_ALL_TESTS();  // 运行所有的测试案例并返回结果
}
```