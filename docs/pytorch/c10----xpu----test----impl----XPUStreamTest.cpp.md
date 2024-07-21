# `.\pytorch\c10\xpu\test\impl\XPUStreamTest.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <c10/util/Optional.h>  // 包含 Optional 类的头文件
#include <c10/util/irange.h>   // 包含 irange 工具函数的头文件
#include <c10/xpu/XPUStream.h> // 包含 XPUStream 类的头文件
#include <c10/xpu/test/impl/XPUTest.h>  // 包含 XPU 测试相关的头文件

#include <thread>  // 包含线程操作相关的头文件
#include <unordered_set>  // 包含无序集合相关的头文件

bool has_xpu() {
  return c10::xpu::device_count() > 0;  // 检查是否存在 XPU 设备
}

TEST(XPUStreamTest, CopyAndMoveTest) {  // 测试 XPUStream 的复制和移动操作
  if (!has_xpu()) {  // 如果没有 XPU 设备，则返回
    return;
  }

  int32_t device = -1;  // 初始化设备索引为 -1
  sycl::queue queue;  // 声明一个 SYCL 队列对象
  c10::xpu::XPUStream copyStream = c10::xpu::getStreamFromPool();  // 从流池中获取一个流对象
  {
    auto s = c10::xpu::getStreamFromPool();  // 从流池中获取一个流对象
    device = s.device_index();  // 获取流对象的设备索引
    queue = s.queue();  // 获取流对象的队列

    copyStream = s;  // 将流对象 s 赋值给 copyStream

    EXPECT_EQ(copyStream.device_index(), device);  // 检查复制后的流对象的设备索引是否正确
    EXPECT_EQ(copyStream.queue(), queue);  // 检查复制后的流对象的队列是否正确
  }

  EXPECT_EQ(copyStream.device_index(), device);  // 再次检查复制后的流对象的设备索引是否正确
  EXPECT_EQ(copyStream.queue(), queue);  // 再次检查复制后的流对象的队列是否正确

  // Tests that moving works as expected and preserves the stream
  c10::xpu::XPUStream moveStream = c10::xpu::getStreamFromPool();  // 从流池中获取一个流对象
  {
    auto s = c10::xpu::getStreamFromPool();  // 从流池中获取一个流对象
    device = s.device_index();  // 获取流对象的设备索引
    queue = s.queue();  // 获取流对象的队列

    moveStream = std::move(s);  // 移动流对象 s 到 moveStream

    EXPECT_EQ(moveStream.device_index(), device);  // 检查移动后的流对象的设备索引是否正确
    EXPECT_EQ(moveStream.queue(), queue);  // 检查移动后的流对象的队列是否正确
  }

  EXPECT_EQ(moveStream.device_index(), device);  // 再次检查移动后的流对象的设备索引是否正确
  EXPECT_EQ(moveStream.queue(), queue);  // 再次检查移动后的流对象的队列是否正确
}

TEST(XPUStreamTest, StreamBehavior) {  // 测试 XPUStream 的行为
  if (!has_xpu()) {  // 如果没有 XPU 设备，则返回
    return;
  }

  c10::xpu::XPUStream stream = c10::xpu::getStreamFromPool();  // 从流池中获取一个流对象
  EXPECT_EQ(stream.device_type(), c10::kXPU);  // 检查流对象的设备类型是否为 XPU
  c10::xpu::setCurrentXPUStream(stream);  // 设置当前的 XPU 流
  c10::xpu::XPUStream cur_stream = c10::xpu::getCurrentXPUStream();  // 获取当前的 XPU 流对象

  EXPECT_EQ(cur_stream, stream);  // 检查当前的 XPU 流对象是否与获取的流对象相同
  EXPECT_EQ(stream.priority(), 0);  // 检查流对象的优先级是否为默认值 0

  auto [least_priority, greatest_priority] = c10::xpu::XPUStream::priority_range();  // 获取优先级范围
  EXPECT_EQ(least_priority, 0);  // 检查最小优先级是否为 0
  EXPECT_TRUE(greatest_priority < 0);  // 检查最大优先级是否小于 0

  stream = c10::xpu::getStreamFromPool(/* isHighPriority */ true);  // 从流池中获取一个高优先级流对象
  EXPECT_TRUE(stream.priority() < 0);  // 检查高优先级流对象的优先级是否小于 0

  if (c10::xpu::device_count() <= 1) {  // 如果设备数量小于等于 1，则返回
    return;
  }

  c10::xpu::set_device(0);  // 设置当前设备为索引 0
  stream = c10::xpu::getStreamFromPool(false, 1);  // 从流池中获取一个指定设备索引为 1 的流对象
  EXPECT_EQ(stream.device_index(), 1);  // 检查流对象的设备索引是否为 1
  EXPECT_NE(stream.device_index(), c10::xpu::current_device());  // 检查流对象的设备索引是否不等于当前设备索引
}

void thread_fun(std::optional<c10::xpu::XPUStream>& cur_thread_stream) {  // 线程函数，接受一个 XPU 流对象的可选引用
  auto new_stream = c10::xpu::getStreamFromPool();  // 从流池中获取一个新的 XPU 流对象
  c10::xpu::setCurrentXPUStream(new_stream);  // 设置当前线程的 XPU 流
  cur_thread_stream = {c10::xpu::getCurrentXPUStream()};  // 将当前 XPU 流对象存储到可选引用中
  EXPECT_EQ(*cur_thread_stream, new_stream);  // 检查存储的 XPU 流对象是否与获取的新流对象相同
}

// Ensures streams are thread local
TEST(XPUStreamTest, MultithreadStreamBehavior) {  // 测试多线程环境下 XPU 流对象的行为
  if (!has_xpu()) {  // 如果没有 XPU 设备，则返回
    return;
  }
  std::optional<c10::xpu::XPUStream> s0, s1;  // 声明两个可选的 XPU 流对象

  std::thread t0{thread_fun, std::ref(s0)};  // 创建线程 t0，调用 thread_fun 函数，并传递 s0 的引用
  std::thread t1{thread_fun, std::ref(s1)};  // 创建线程 t1，调用 thread_fun 函数，并传递 s1 的引用
  t0.join();  // 等待线程 t0 完成
  t1.join();  // 等待线程 t1 完成

  c10::xpu::XPUStream cur_stream = c10::xpu::getCurrentXPUStream();  // 获取当前的 XPU 流对象

  EXPECT_NE(cur_stream, *s0);  // 检查当前的 XPU 流对象是否不等于线程 t0 存储的流对象
  EXPECT_NE(cur_stream, *s1);  // 检查当前的 XPU 流对象是否不等于线程 t1 存储的流对象
  EXPECT_NE(s0, s1);  // 检查线程 t0 和线程 t1 存储的流对象是否不相同
}

// Ensure queue pool round-robin fashion
TEST(XPUStreamTest, StreamPoolRoundRobinTest) {  // 测试流池中流对象的循环使用
  if (!has_xpu()) {  // 如果没有 XPU 设备，则返回
    return;
  }



  std::vector<c10::xpu::XPUStream> streams{};
  // 创建一个空的向量 streams，用于存储 c10::xpu::XPUStream 对象
  for (C10_UNUSED const auto _ : c10::irange(200)) {
    // 循环 200 次，每次从 c10::irange 生成的占位符 _ 中获取一个 XPUStream 对象，并添加到 streams 向量中
    streams.emplace_back(c10::xpu::getStreamFromPool());
    // 将新创建的 XPUStream 对象添加到 streams 向量的末尾
  }



  std::unordered_set<sycl::queue> queue_set{};
  // 创建一个空的无序集合 queue_set，用于存储 sycl::queue 对象，确保其中元素唯一
  bool hasDuplicates = false;
  // 初始化一个布尔变量 hasDuplicates，用于记录是否存在重复的队列
  for (const auto i : c10::irange(streams.size())) {
    // 对于 streams 向量中的每个索引 i，进行迭代
    auto& queue = streams[i].queue();
    // 获取 streams 向量中索引 i 处的 XPUStream 对象的队列引用，存储到 queue 变量中
    auto result_pair = queue_set.insert(queue);
    // 将 queue 插入到 queue_set 中，并返回插入结果的 pair 对象 result_pair
    if (!result_pair.second) { // already existed
      // 如果插入操作返回的 second 成员为 false，表示队列已经存在于 queue_set 中
      hasDuplicates = true;
      // 将 hasDuplicates 置为 true，表示存在重复的队列
    } else { // newly inserted
      // 如果插入操作返回的 second 成员为 true，表示队列是新插入的
      EXPECT_TRUE(!hasDuplicates);
      // 使用断言验证此时 hasDuplicates 应为 false，即新插入队列时不应存在重复
    }
  }



  EXPECT_TRUE(hasDuplicates);
  // 使用断言验证最终 hasDuplicates 应为 true，即确保在遍历结束后存在重复的队列

  auto stream = c10::xpu::getStreamFromPool(/* isHighPriority */ true);
  // 调用 c10::xpu::getStreamFromPool 函数获取一个高优先级的 XPUStream 对象，存储到 stream 变量中
  auto result_pair = queue_set.insert(stream.queue());
  // 将 stream 的队列插入到 queue_set 中，并返回插入结果的 pair 对象 result_pair
  EXPECT_TRUE(result_pair.second);
  // 使用断言验证插入操作返回的 second 成员应为 true，即确保高优先级队列成功插入到 queue_set 中
}

void asyncMemCopy(sycl::queue& queue, int* dst, int* src, size_t numBytes) {
  // 使用 SYCL 队列执行异步内存拷贝，将 src 指向的数据拷贝到 dst 指向的位置，拷贝的字节数为 numBytes
  queue.memcpy(dst, src, numBytes);
}

TEST(XPUStreamTest, StreamFunction) {
  // 如果没有 XPU 设备可用，退出测试
  if (!has_xpu()) {
    return;
  }

  // 定义要处理的数据元素数量
  constexpr int numel = 1024;
  // 在主机端初始化数据数组
  int hostData[numel];
  initHostData(hostData, numel);

  // 从 XPU 流池中获取一个流对象
  auto stream = c10::xpu::getStreamFromPool();
  // 检查流对象是否有效
  EXPECT_TRUE(stream.query());
  // 在设备端分配一个整型数组，并返回指向该数组的指针
  int* deviceData = sycl::malloc_device<int>(numel, stream);

  // 主机到设备的数据传输
  asyncMemCopy(stream, deviceData, hostData, sizeof(int) * numel);
  // 同步设备上的所有流
  c10::xpu::syncStreamsOnDevice();
  // 再次检查流对象是否有效
  EXPECT_TRUE(stream.query());

  // 清空主机端的数据数组
  clearHostData(hostData, numel);

  // 设备到主机的数据传输
  asyncMemCopy(stream, hostData, deviceData, sizeof(int) * numel);
  // 再次同步设备上的所有流
  c10::xpu::syncStreamsOnDevice();

  // 验证主机端数据的正确性
  validateHostData(hostData, numel);

  // 从流池中获取另一个流对象（传入 -1 表示获取默认流）
  stream = c10::xpu::getStreamFromPool(-1);

  // 再次清空主机端的数据数组
  clearHostData(hostData, numel);

  // 设备到主机的数据传输
  asyncMemCopy(stream, hostData, deviceData, sizeof(int) * numel);
  // 再次同步设备上的所有流
  c10::xpu::syncStreamsOnDevice();

  // 再次验证主机端数据的正确性
  validateHostData(hostData, numel);

  // 释放在设备上分配的内存
  sycl::free(deviceData, c10::xpu::get_device_context());
}
```