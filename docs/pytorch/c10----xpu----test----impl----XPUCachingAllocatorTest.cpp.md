# `.\pytorch\c10\xpu\test\impl\XPUCachingAllocatorTest.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <c10/util/irange.h>  // 引入 C10 库中的 irange 头文件，用于迭代范围
#include <c10/xpu/XPUCachingAllocator.h>  // 引入 C10 库中的 XPUCachingAllocator 头文件

bool has_xpu() {
  return c10::xpu::device_count() > 0;  // 检查是否存在 XPU 设备
}

TEST(XPUCachingAllocatorTest, GetXPUAllocator) {
  auto* allocator = c10::xpu::XPUCachingAllocator::get();  // 获取 XPUCachingAllocator 实例

  auto _500mb = 500 * 1024 * 1024;  // 定义 500MB 的内存大小
  auto buffer = allocator->allocate(_500mb);  // 分配 500MB 内存
  EXPECT_TRUE(buffer.get());  // 断言内存分配成功

  auto* xpu_allocator = c10::GetAllocator(buffer.device().type());  // 获取与 buffer 设备类型匹配的分配器
  EXPECT_EQ(allocator, xpu_allocator);  // 断言分配器一致
}

TEST(XPUCachingAllocatorTest, DeviceCachingAllocate) {
  c10::xpu::XPUCachingAllocator::emptyCache();  // 清空缓存
  auto* allocator = c10::xpu::XPUCachingAllocator::get();  // 获取 XPUCachingAllocator 实例

  {
    auto _500mb = 500 * 1024 * 1024;  // 定义 500MB 的内存大小
    auto cache = allocator->allocate(_500mb);  // 分配 500MB 内存
  }

  auto _10mb = 10 * 1024 * 1024;  // 定义 10MB 的内存大小
  auto buffer = allocator->allocate(_10mb);  // 分配 10MB 内存
  void* ptr0 = buffer.get();  // 获取第一个分配的内存块的指针

  // 通过设备缓存分配器分配 tmp 指针，其它方式分配的指针不使用缓存
  void* tmp = sycl::aligned_alloc_device(
      512, _10mb, c10::xpu::get_raw_device(0), c10::xpu::get_device_context());
  void* ptr1 = c10::xpu::XPUCachingAllocator::raw_alloc(_10mb);

  // 断言 ptr0 和 ptr1 在内存中是连续的
  auto diff = static_cast<char*>(ptr1) - static_cast<char*>(ptr0);
  EXPECT_EQ(diff, _10mb);

  c10::xpu::XPUCachingAllocator::raw_delete(ptr1);  // 删除 ptr1 指向的内存块
  sycl::free(tmp, c10::xpu::get_device_context());  // 释放 tmp 指向的内存块
  c10::xpu::XPUCachingAllocator::emptyCache();  // 清空缓存
}

TEST(XPUCachingAllocatorTest, AllocateMemory) {
  c10::xpu::XPUCachingAllocator::emptyCache();  // 清空缓存
  auto* allocator = c10::xpu::XPUCachingAllocator::get();  // 获取 XPUCachingAllocator 实例

  auto _10mb = 10 * 1024 * 1024;  // 定义 10MB 的内存大小
  auto buffer = allocator->allocate(_10mb);  // 分配 10MB 内存
  auto* deviceData = static_cast<int*>(buffer.get());  // 获取设备数据指针

  constexpr int numel = 1024;  // 定义元素数量
  int hostData[numel];  // 在主机上分配数组

  // 填充主机数据数组
  for (const auto i : c10::irange(numel)) {
    hostData[i] = i;
  }

  auto stream = c10::xpu::getStreamFromPool();  // 获取流对象
  // 主机到设备的数据传输
  stream.queue().memcpy(deviceData, hostData, sizeof(int) * numel);
  c10::xpu::syncStreamsOnDevice();  // 等待设备操作完成

  // 清空主机数据数组
  for (const auto i : c10::irange(numel)) {
    hostData[i] = 0;
  }

  // 设备到主机的数据传输
  stream.queue().memcpy(hostData, deviceData, sizeof(int) * numel);
  c10::xpu::syncStreamsOnDevice();  // 等待设备操作完成

  // 验证数据传输正确性
  for (const auto i : c10::irange(numel)) {
    EXPECT_EQ(hostData[i], i);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);  // 初始化 Google Test 框架

  auto device = c10::xpu::device_count();  // 获取设备数量
  if (device <= 0) {
    return 0;  // 如果没有可用设备，直接返回
  }

  c10::xpu::XPUCachingAllocator::init(device);  // 初始化 XPUCachingAllocator
  return RUN_ALL_TESTS();  // 执行所有测试用例
}
```