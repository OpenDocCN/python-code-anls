# `.\pytorch\aten\src\ATen\test\cuda_allocatorTraceTracker_test.cpp`

```py
#include <c10/cuda/CUDACachingAllocator.h>  // 包含 CUDA 缓存分配器的头文件

#include <gtest/gtest.h>  // 包含 Google 测试框架的头文件

static int segmentAllocCalled = 0;  // 静态变量，记录分配段的调用次数
static int segmentFreeCalled = 0;   // 静态变量，记录释放段的调用次数

static void SegmentAllocTraceTracker(  // 定义跟踪分配段的回调函数
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  if (te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    segmentAllocCalled++;  // 如果跟踪条目是分配段动作，则增加分配段调用计数
  }
}

static void SegmentFreeTraceTracker(  // 定义跟踪释放段的回调函数
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  if (te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    segmentFreeCalled++;  // 如果跟踪条目是释放段动作，则增加释放段调用计数
  }
}

static void allocateLargeBuffer() {
  const auto _500mb = 500 * 1024 * 1024;  // 定义一个常量，表示500MB的大小
  auto* allocator = c10::cuda::CUDACachingAllocator::get();  // 获取 CUDA 缓存分配器的实例
  auto buffer = allocator->allocate(_500mb);  // 分配一个500MB大小的缓冲区
}

TEST(AllocatorTraceTracker, TrackMallocFree) {
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &SegmentAllocTraceTracker);  // 注册跟踪分配段的回调函数
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &SegmentFreeTraceTracker);   // 注册跟踪释放段的回调函数

  // 期望触发大缓冲区的段分配，并且预期在从 allocateLargeBuffer 返回时将缓冲区标记为非活动状态，
  // 在调用 emptyCache 时将其释放
  allocateLargeBuffer();
  ASSERT_EQ(segmentAllocCalled, 1);  // 断言分配段的调用次数为1

  // 期望已释放分配的缓冲区返回给分配器，因此 emptyCache 将触发段释放
  c10::cuda::CUDACachingAllocator::emptyCache();
  ASSERT_EQ(segmentFreeCalled, 1);  // 断言释放段的调用次数为1
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);  // 初始化 Google 测试框架
  c10::cuda::CUDACachingAllocator::init(1);  // 初始化 CUDA 缓存分配器
  return RUN_ALL_TESTS();  // 运行所有的测试用例
}
```