# `.\pytorch\c10\xpu\test\impl\XPUGuardTest.cpp`

```
// 引入 Google Test 框架的头文件，用于单元测试
#include <gtest/gtest.h>

// 引入 C10 库的相关头文件
#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/test/impl/XPUTest.h>

// 检查当前系统是否有 XPU 设备
bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

// 定义 XPUGuardTest 单元测试，测试设备保护的行为
TEST(XPUGuardTest, GuardBehavior) {
  // 如果系统没有 XPU 设备，直接返回
  if (!has_xpu()) {
    return;
  }

  {
    // 创建一个 XPU 设备对象
    auto device = c10::Device(c10::kXPU);
    // 使用 DeviceGuard 保护当前线程中的 XPU 设备
    const c10::DeviceGuard device_guard(device);
    // 断言当前 XPU 设备索引为 0
    EXPECT_EQ(c10::xpu::current_device(), 0);
  }

  // 创建一个包含两个 XPU 流的向量
  std::vector<c10::xpu::XPUStream> streams0 = {
      c10::xpu::getStreamFromPool(), c10::xpu::getStreamFromPool(true)};
  // 断言第一个流的设备索引为 0
  EXPECT_EQ(streams0[0].device_index(), 0);
  // 断言第二个流的设备索引为 0
  EXPECT_EQ(streams0[1].device_index(), 0);
  // 设置当前 XPU 流为 streams0 中的第一个流
  c10::xpu::setCurrentXPUStream(streams0[0]);
  // 断言当前 XPU 流为 streams0 中的第一个流
  EXPECT_EQ(c10::xpu::getCurrentXPUStream(), streams0[0]);

  // 如果系统中的 XPU 设备数量不超过 1，直接返回
  if (c10::xpu::device_count() <= 1) {
    return;
  }

  // 测试第二个 XPU 设备的 DeviceGuard
  std::vector<c10::xpu::XPUStream> streams1;
  {
    // 创建一个 XPU 设备对象，设备索引为 1
    auto device = c10::Device(c10::kXPU, 1);
    // 使用 DeviceGuard 保护当前线程中的 XPU 设备
    const c10::DeviceGuard device_guard(device);
    // 将一个新的 XPU 流加入到 streams1 向量中
    streams1.push_back(c10::xpu::getStreamFromPool());
    // 将另一个新的 XPU 流加入到 streams1 向量中
    streams1.push_back(c10::xpu::getStreamFromPool());
  }

  // 断言 streams1 中第一个流的设备索引为 1
  EXPECT_EQ(streams1[0].device_index(), 1);
  // 断言 streams1 中第二个流的设备索引为 1
  EXPECT_EQ(streams1[1].device_index(), 1);
  // 断言当前 XPU 设备索引为 0
  EXPECT_EQ(c10::xpu::current_device(), 0);
}

// 定义 XPUGuardTest 单元测试，测试事件的行为
TEST(XPUGuardTest, EventBehavior) {
  // 如果系统没有 XPU 设备，直接返回
  if (!has_xpu()) {

    return;
  }

  // 在这里继续添加事件行为的测试代码
}
    // 如果遇到这里就直接返回，结束函数执行
    return;
  }

  // 创建一个 XPU 类型的设备对象，使用当前的 XPU 设备
  auto device = c10::Device(c10::kXPU, c10::xpu::current_device());
  // 创建一个虚拟的守卫实现对象，使用指定设备类型
  c10::impl::VirtualGuardImpl impl(device.type());
  // 获取该设备上的两个流对象，stream1 和 stream2
  c10::Stream stream1 = impl.getStream(device);
  c10::Stream stream2 = impl.getStream(device);
  // 创建一个事件对象 event1，使用设备的类型
  c10::Event event1(device.type());
  // 如果事件尚未被创建，则返回 false
  // 此处期望返回 false，因为 event1 是延迟创建的
  EXPECT_FALSE(event1.eventId());

  // 定义一个常量 numel，值为 1024
  constexpr int numel = 1024;
  // 定义一个长度为 numel 的整型数组 hostData1，并初始化其内容
  int hostData1[numel];
  initHostData(hostData1, numel);
  // 定义一个长度为 numel 的整型数组 hostData2，并清空其内容
  int hostData2[numel];
  clearHostData(hostData2, numel);

  // 创建一个 XPUStream 对象 xpu_stream1，使用 stream1
  auto xpu_stream1 = c10::xpu::XPUStream(stream1);
  // 在设备上分配一个长度为 numel 的整型数组 deviceData1，使用 xpu_stream1
  int* deviceData1 = sycl::malloc_device<int>(numel, xpu_stream1);

  // 将 hostData1 的数据通过 stream1 复制到 deviceData1，然后将 deviceData1 通过 stream2 复制到 hostData2
  xpu_stream1.queue().memcpy(deviceData1, hostData1, sizeof(int) * numel);
  // 记录事件 event1 在 stream1 上的完成状态
  event1.record(stream1);
  // 在 stream2 上阻塞等待 stream1 完成
  event1.block(stream2);
  // 创建一个 XPUStream 对象 xpu_stream2，使用 stream2
  auto xpu_stream2 = c10::xpu::XPUStream(stream2);
  // 将 deviceData1 的数据通过 xpu_stream2 复制到 hostData2
  xpu_stream2.queue().memcpy(hostData2, deviceData1, sizeof(int) * numel);
  // 同步 xpu_stream2 上的操作
  xpu_stream2.synchronize();

  // 检查事件 event1 的查询状态是否为 true
  EXPECT_TRUE(event1.query());
  // 验证 hostData2 的数据是否有效
  validateHostData(hostData2, numel);
  // 记录事件 event1 在 stream2 上的完成状态
  event1.record(stream2);
  // 同步事件 event1
  event1.synchronize();
  // 再次检查事件 event1 的查询状态是否为 true
  EXPECT_TRUE(event1.query());

  // 清空 hostData2 的数据
  clearHostData(hostData2, numel);
  // 将 hostData1 的数据通过 stream1 复制到 deviceData1
  xpu_stream1.queue().memcpy(deviceData1, hostData1, sizeof(int) * numel);
  // 在 stream2 上阻塞等待 stream1 完成
  event1.record(stream1);
  event1.block(stream2);
  // 事件 event1 将覆盖先前捕获的状态
  event1.record(stream2);
  // 将 deviceData1 的数据通过 xpu_stream2 复制到 hostData2
  xpu_stream2.queue().memcpy(hostData2, deviceData1, sizeof(int) * numel);
  // 同步 xpu_stream2 上的操作
  xpu_stream2.synchronize();
  // 检查事件 event1 的查询状态是否为 true
  EXPECT_TRUE(event1.query());
  // 验证 hostData2 的数据是否有效
  validateHostData(hostData2, numel);

  // 清空 hostData2 的数据
  clearHostData(hostData2, numel);
  // 确保 deviceData1 和 deviceData2 是不同的缓冲区
  int* deviceData2 = sycl::malloc_device<int>(numel, xpu_stream1);
  // 释放 deviceData1
  sycl::free(deviceData1, c10::xpu::get_device_context());
  // 创建一个事件对象 event2，使用设备的类型
  c10::Event event2(device.type());

  // 将 hostData1 的数据通过 stream1 复制到 deviceData2
  xpu_stream1.queue().memcpy(deviceData2, hostData1, sizeof(int) * numel);
  // 记录事件 event2 在 xpu_stream1 上的完成状态
  event2.record(xpu_stream1);
  // 同步事件 event2
  event2.synchronize();
  // 检查事件 event2 的查询状态是否为 true
  EXPECT_TRUE(event2.query());
  // 清空 hostData1 的数据
  clearHostData(hostData1, numel);
  // 将 deviceData2 的数据通过 stream1 复制到 hostData1
  xpu_stream1.queue().memcpy(hostData1, deviceData2, sizeof(int) * numel);
  // 记录事件 event2 在 xpu_stream1 上的完成状态
  event2.record(xpu_stream1);
  // 同步事件 event2
  event2.synchronize();
  // 检查事件 event2 的查询状态是否为 true
  EXPECT_TRUE(event2.query());
  // 检查事件 event1 和 event2 的事件 ID 是否不相同
  EXPECT_NE(event1.eventId(), event2.eventId());
  // 断言捕获两个事件之间的时间差
  ASSERT_THROW(event1.elapsedTime(event2), c10::Error);
  // 释放 deviceData2
  sycl::free(deviceData2, c10::xpu::get_device_context());
}


注释：


# 这行代码表示一个代码块的结束，对应于一个开放的代码块的结尾。
```