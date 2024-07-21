# `.\pytorch\aten\src\ATen\test\xpu_event_test.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/xpu/XPUEvent.h>  // 包含 XPUEvent 头文件
#include <c10/util/irange.h>    // 包含 irange 工具函数
#include <c10/xpu/test/impl/XPUTest.h>  // 包含 XPUTest 实现

// XPUEventTest 测试套件，测试 XPUEvent 行为
TEST(XpuEventTest, testXPUEventBehavior) {
  // 如果当前 XPU 不可用，则直接返回
  if (!at::xpu::is_available()) {
    return;
  }
  
  // 从 XPU 的流池中获取流对象
  auto stream = c10::xpu::getStreamFromPool();
  // 创建 XPUEvent 对象
  at::xpu::XPUEvent event;

  // 断言事件查询状态为真
  EXPECT_TRUE(event.query());
  // 断言事件未被创建
  EXPECT_TRUE(!event.isCreated());

  // 记录事件的发生一次，并将其关联到指定的流上
  event.recordOnce(stream);
  // 断言事件已经被创建
  EXPECT_TRUE(event.isCreated());

  // 从流池中获取两个等待流
  auto wait_stream0 = c10::xpu::getStreamFromPool();
  auto wait_stream1 = c10::xpu::getStreamFromPool();

  // 阻塞事件，使其等待两个不同的流
  event.block(wait_stream0);
  event.block(wait_stream1);

  // 同步等待流0完成操作
  wait_stream0.synchronize();
  // 断言事件查询状态为真
  EXPECT_TRUE(event.query());
}

// XPUEventTest 测试套件，测试 XPUEvent 跨设备行为
TEST(XpuEventTest, testXPUEventCrossDevice) {
  // 如果设备数小于等于1，则直接返回
  if (at::xpu::device_count() <= 1) {
    return;
  }

  // 从流池中获取两个流对象
  const auto stream0 = at::xpu::getStreamFromPool();
  at::xpu::XPUEvent event0;

  // 从流池中获取第二个流对象，并指定设备编号为1
  const auto stream1 = at::xpu::getStreamFromPool(false, 1);
  at::xpu::XPUEvent event1;

  // 记录事件0关联到流0
  event0.record(stream0);
  // 记录事件1关联到流1
  event1.record(stream1);

  // 使用移动语义将事件1的状态转移到事件0
  event0 = std::move(event1);

  // 断言事件0所在设备为 XPU 设备，编号为1
  EXPECT_EQ(event0.device(), at::Device(at::kXPU, 1));

  // 阻塞事件0，使其等待流0完成操作
  event0.block(stream0);

  // 同步等待流0完成操作
  stream0.synchronize();
  // 断言事件0查询状态为真
  ASSERT_TRUE(event0.query());
}

// 将 sycl::event 等待函数包装为 XPUEvent 的同步函数
void eventSync(sycl::event& event) {
  event.wait();
}

// XPUEventTest 测试套件，测试 XPUEvent 功能函数
TEST(XpuEventTest, testXPUEventFunction) {
  // 如果当前 XPU 不可用，则直接返回
  if (!at::xpu::is_available()) {
    return;
  }

  // 定义常量，表示需要操作的元素数量
  constexpr int numel = 1024;
  // 在主机上初始化数据数组
  int hostData[numel];
  initHostData(hostData, numel);

  // 从流池中获取流对象
  auto stream = c10::xpu::getStreamFromPool();
  // 在设备上分配内存，用于存储数据
  int* deviceData = sycl::malloc_device<int>(numel, stream);

  // 主机到设备的数据拷贝操作
  stream.queue().memcpy(deviceData, hostData, sizeof(int) * numel);
  // 创建 XPUEvent 对象，并记录在指定流上
  at::xpu::XPUEvent event;
  event.record(stream);
  // 将 XPUEvent 隐式转换为 sycl::event，并等待其完成
  eventSync(event);
  // 断言事件查询状态为真
  EXPECT_TRUE(event.query());

  // 清理主机上的数据
  clearHostData(hostData, numel);

  // 设备到主机的数据拷贝操作
  stream.queue().memcpy(hostData, deviceData, sizeof(int) * numel);
  // 记录事件在指定流上，并等待其完成
  event.record(stream);
  event.synchronize();

  // 验证主机上的数据与设备上的数据一致性
  validateHostData(hostData, numel);

  // 再次清理主机上的数据
  clearHostData(hostData, numel);
  // 设备到主机的数据拷贝操作
  stream.queue().memcpy(hostData, deviceData, sizeof(int) * numel);
  // 由于事件已经被创建，此处不会记录流的操作
  event.recordOnce(stream);
  // 断言事件查询状态为真
  EXPECT_TRUE(event.query());

  // 同步等待流对象完成操作
  stream.synchronize();
  // 释放设备上分配的内存
  sycl::free(deviceData, c10::xpu::get_device_context());

  // 如果设备数小于等于1，则直接返回
  if (at::xpu::device_count() <= 1) {
    return;
  }
  
  // 设置当前设备为设备1
  c10::xpu::set_device(1);
  // 从流池中获取流对象
  auto stream1 = c10::xpu::getStreamFromPool();
  // 断言事件记录流对象时抛出 c10::Error 异常
  ASSERT_THROW(event.record(stream1), c10::Error);
}
```