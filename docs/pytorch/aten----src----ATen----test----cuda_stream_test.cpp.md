# `.\pytorch\aten\src\ATen\test\cuda_stream_test.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 ATen 的 CUDA 相关头文件
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Event.h>
#include <c10/core/impl/InlineEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <c10/util/irange.h>

// 包含 CUDA 运行时的头文件
#include <cuda_runtime.h>

// 包含 C++ 标准库的头文件
#include <functional>
#include <future>
#include <thread>
#include <unordered_set>

// 定义一个宏，用于比较相等条件下的断言
#define ASSERT_EQ_CUDA(X, Y) \
  {                          \
    bool isTRUE = X == Y;    \
    ASSERT_TRUE(isTRUE);     \
  }

// 定义一个宏，用于比较不相等条件下的断言
#define ASSERT_NE_CUDA(X, Y) \
  {                          \
    bool isFALSE = X == Y;   \
    ASSERT_FALSE(isFALSE);   \
  }

/*
   Tests related to ATen streams.
   */
// 验证通过复制和移动操作流的正确性
TEST(TestStream, CopyAndMoveTest) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) return;

  // 初始化变量
  int32_t device = -1;
  cudaStream_t cuda_stream;

  // 从 CUDA 流池中获取流对象
  at::cuda::CUDAStream copyStream = at::cuda::getStreamFromPool();

  {
    // 获取另一个流对象
    auto s = at::cuda::getStreamFromPool();
    // 获取流所在的设备索引和流句柄
    device = s.device_index();
    cuda_stream = s.stream();

    // 将流对象复制给 copyStream
    copyStream = s;

    // 断言复制后的流对象与原始流对象的设备索引和流句柄相等
    ASSERT_EQ_CUDA(copyStream.device_index(), device);
    ASSERT_EQ_CUDA(copyStream.stream(), cuda_stream);
  }

  // 再次断言复制后的流对象与原始流对象的设备索引和流句柄相等
  ASSERT_EQ_CUDA(copyStream.device_index(), device);
  ASSERT_EQ_CUDA(copyStream.stream(), cuda_stream);

  // 初始化变量
  at::cuda::CUDAStream moveStream = at::cuda::getStreamFromPool();

  {
    // 获取另一个流对象
    auto s = at::cuda::getStreamFromPool();
    // 获取流所在的设备索引和流句柄
    device = s.device_index();
    cuda_stream = s.stream();

    // 将流对象移动给 moveStream
    moveStream = std::move(s);

    // 断言移动后的流对象与原始流对象的设备索引和流句柄相等
    ASSERT_EQ_CUDA(moveStream.device_index(), device);
    ASSERT_EQ_CUDA(moveStream.stream(), cuda_stream);
  }

  // 再次断言移动后的流对象与原始流对象的设备索引和流句柄相等
  ASSERT_EQ_CUDA(moveStream.device_index(), device);
  ASSERT_EQ_CUDA(moveStream.stream(), cuda_stream);
}

// 验证流对象的设置和获取
TEST(TestStream, GetAndSetTest) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) return;

  // 从 CUDA 流池中获取一个流对象
  at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();

  // 将 myStream 设置为当前的 CUDA 流
  at::cuda::setCurrentCUDAStream(myStream);
  // 获取当前的 CUDA 流
  at::cuda::CUDAStream curStream = at::cuda::getCurrentCUDAStream();

  // 断言设置后的流对象与获取的流对象相等
  ASSERT_EQ_CUDA(myStream, curStream);

  // 获取默认的 CUDA 流对象
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  // 将默认的 CUDA 流对象设置为当前流
  at::cuda::setCurrentCUDAStream(defaultStream);
  // 再次获取当前的 CUDA 流对象
  curStream = at::cuda::getCurrentCUDAStream();

  // 断言默认流对象与 myStream 流对象不相等
  ASSERT_NE_CUDA(defaultStream, myStream);
  // 断言当前流对象与默认流对象相等
  ASSERT_EQ_CUDA(curStream, defaultStream);
}

// 线程函数，用于验证流对象在多线程情况下的局部性
void thread_fun(at::optional<at::cuda::CUDAStream>& cur_thread_stream) {
  // 从 CUDA 流池中获取一个新的流对象
  auto new_stream = at::cuda::getStreamFromPool();
  // 将新的流对象设置为当前的 CUDA 流
  at::cuda::setCurrentCUDAStream(new_stream);
  // 将当前线程的流对象存入 cur_thread_stream
  cur_thread_stream = {at::cuda::getCurrentCUDAStream()};
  // 断言当前线程的流对象与新创建的流对象相等
  ASSERT_EQ_CUDA(*cur_thread_stream, new_stream);
}
TEST(TestStream, MultithreadGetAndSetTest) {
  // 检查CUDA是否可用，不可用则退出测试
  if (!at::cuda::is_available()) return;

  // 定义两个可选的CUDA流对象
  at::optional<at::cuda::CUDAStream> s0, s1;

  // 创建两个线程，分别调用thread_fun函数，并传入s0和s1的引用
  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  // 等待线程执行完毕
  t0.join();
  t1.join();

  // 获取当前CUDA流对象和默认CUDA流对象
  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAStream default_stream = at::cuda::getDefaultCUDAStream();

  // 断言当前CUDA流与默认CUDA流相等
  ASSERT_EQ_CUDA(cur_stream, default_stream);
  // 断言当前CUDA流与s0不相等
  ASSERT_NE_CUDA(cur_stream, *s0);
  // 断言当前CUDA流与s1不相等
  ASSERT_NE_CUDA(cur_stream, *s1);
  // 断言s0与s1不相等
  ASSERT_NE_CUDA(s0, s1);
}

// CUDA Guard
TEST(TestStream, CUDAGuardTest) {
  // 检查CUDA是否可用，不可用则退出测试
  if (!at::cuda::is_available()) return;
  // 如果GPU数量小于2，则退出测试
  if (at::cuda::getNumGPUs() < 2) {
    return;
  }

  // -- begin setup

  // 断言当前设备为第一个GPU
  ASSERT_EQ_CUDA(at::cuda::current_device(), 0);
  // 创建一个包含默认CUDA流和从流池中获取的CUDA流的向量
  std::vector<at::cuda::CUDAStream> streams0 = {
      at::cuda::getDefaultCUDAStream(), at::cuda::getStreamFromPool()};
  // 断言第一个CUDA流的设备索引为0
  ASSERT_EQ_CUDA(streams0[0].device_index(), 0);
  // 断言第二个CUDA流的设备索引为0
  ASSERT_EQ_CUDA(streams0[1].device_index(), 0);
  // 设置当前CUDA流为第一个CUDA流
  at::cuda::setCurrentCUDAStream(streams0[0]);

  // 创建一个新的CUDA流向量streams1
  std::vector<at::cuda::CUDAStream> streams1;
  {
    // 切换当前设备为第二个GPU
    at::cuda::CUDAGuard device_guard(1);
    // 向streams1中添加默认CUDA流和从流池中获取的CUDA流
    streams1.push_back(at::cuda::getDefaultCUDAStream());
    streams1.push_back(at::cuda::getStreamFromPool());
  }
  // 断言第一个CUDA流的设备索引为1
  ASSERT_EQ_CUDA(streams1[0].device_index(), 1);
  // 断言第二个CUDA流的设备索引为1
  ASSERT_EQ_CUDA(streams1[1].device_index(), 1);
  // 设置当前CUDA流为streams1中的第一个CUDA流
  at::cuda::setCurrentCUDAStream(streams1[0]);

  // 断言当前设备为第一个GPU
  ASSERT_EQ_CUDA(at::cuda::current_device(), 0);

  // -- end setup

  // 改变当前设备和设备上的流
  {
    // 使用CUDA流保护器设置当前CUDA流为streams1中的第二个CUDA流
    at::cuda::CUDAStreamGuard guard(streams1[1]);
    // 断言保护器当前设备为第一个GPU
    ASSERT_EQ_CUDA(guard.current_device(), at::Device(at::kCUDA, 1));
    // 断言当前设备为第二个GPU
    ASSERT_EQ_CUDA(at::cuda::current_device(), 1);
    // 断言第二个GPU上的当前CUDA流为streams1中的第二个CUDA流
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(1), streams1[1]);
  }

  // 重置设备和CUDA流
  ASSERT_EQ_CUDA(at::cuda::current_device(), 0);
  // 断言第一个GPU上的当前CUDA流为streams1中的第一个CUDA流
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(1), streams1[0]);

  // 仅设置设备，不改变CUDA流
  {
    // 使用CUDA保护器设置当前设备为第一个GPU
    at::cuda::CUDAGuard guard(/*device=*/1);
    // 断言保护器当前设备为第一个GPU
    ASSERT_EQ_CUDA(guard.current_device(), at::Device(at::kCUDA, 1));
    // 断言当前设备为第一个GPU
    ASSERT_EQ_CUDA(at::cuda::current_device(), 1);
    // 断言第一个GPU上的当前CUDA流仍为streams1中的第一个CUDA流
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(1), streams1[0]);
  }

  // 断言当前设备为第一个GPU
  ASSERT_EQ_CUDA(at::cuda::current_device(), 0);
  // 断言当前设备上的当前CUDA流为streams0中的第一个CUDA流
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(0), streams0[0]);
}

// Streampool Round Robin
TEST(TestStream, StreamPoolTest) {
  // 检查CUDA是否可用，不可用则退出测试
  if (!at::cuda::is_available()) return;
  
  // 创建一个包含从流池获取的多个CUDA流的向量
  std::vector<at::cuda::CUDAStream> streams{};
  for (const auto i : c10::irange(200)) {
    (void)i;
    // 向streams中添加从流池获取的CUDA流
    streams.emplace_back(at::cuda::getStreamFromPool());
  }

  // 创建一个用于存储CUDA流的无序集合
  std::unordered_set<cudaStream_t> stream_set{};
  // 检查CUDA流是否有重复，并标记是否有重复流
  bool hasDuplicates = false;
  for (const auto i: c10::irange(streams.size())) {
    cudaStream_t cuda_stream = streams[i];
    auto result_pair = stream_set.insert(cuda_stream);
    if (!result_pair.second)
      hasDuplicates = true;
  }

  // 断言流中存在重复的CUDA流
  ASSERT_TRUE(hasDuplicates);
}
// 在 TestStream 测试套件中的 MultiGPUTest 测试用例
TEST(TestStream, MultiGPUTest) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) return;
  // 如果可用的 GPU 数量小于 2，也退出测试
  if (at::cuda::getNumGPUs() < 2)
    return;

  // 从 CUDA 流池中获取两个 CUDA 流
  at::cuda::CUDAStream s0 = at::cuda::getStreamFromPool(true, 0);
  at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool(false, 1);

  // 设置当前 CUDA 流为 s0
  at::cuda::setCurrentCUDAStream(s0);
  // 设置当前 CUDA 流为 s1，实际上会覆盖之前设置的 s0
  at::cuda::setCurrentCUDAStream(s1);

  // 断言当前 CUDA 流为 s0
  ASSERT_EQ_CUDA(s0, at::cuda::getCurrentCUDAStream());

  // 切换到设备 1 的 CUDA 流
  at::cuda::CUDAGuard device_guard{1};
  // 断言当前 CUDA 流为 s1
  ASSERT_EQ_CUDA(s1, at::cuda::getCurrentCUDAStream());
}

// CUDAEvent Syncs 测试套件
TEST(TestStream, CUDAEventSyncTest) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) return;
  // 从 CUDA 流池中获取流
  const auto stream = at::cuda::getStreamFromPool();
  // 创建 CUDAEvent 对象
  at::cuda::CUDAEvent event;

  // 查询事件状态应为真
  ASSERT_TRUE(event.query());

  // 记录事件的发生到特定流
  event.recordOnce(stream);

  // 从 CUDA 流池中获取两个等待流
  const auto wait_stream0 = at::cuda::getStreamFromPool();
  const auto wait_stream1 = at::cuda::getStreamFromPool();

  // 阻塞事件，等待在两个不同的流上
  event.block(wait_stream0);
  event.block(wait_stream1);

  // 同步等待流 0
  cudaStreamSynchronize(wait_stream0);
  // 断言事件查询状态为真
  ASSERT_TRUE(event.query());
}

// Cross-Device Events 测试套件
TEST(TestStream, CrossDeviceTest) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) return;
  // 如果可用的 GPU 数量小于 2，也退出测试
  if (at::cuda::getNumGPUs() < 2)
    return;

  // 从 CUDA 流池中获取流 stream0
  const auto stream0 = at::cuda::getStreamFromPool();
  // 创建 CUDAEvent 对象 event0
  at::cuda::CUDAEvent event0;

  // 切换到设备 1
  at::cuda::set_device(1);
  // 从 CUDA 流池中获取流 stream1
  const auto stream1 = at::cuda::getStreamFromPool();
  // 创建 CUDAEvent 对象 event1
  at::cuda::CUDAEvent event1;

  // 记录事件0在 stream0 上的发生
  event0.record(stream0);
  // 记录事件1在 stream1 上的发生
  event1.record(stream1);

  // 将 event1 移动给 event0
  event0 = std::move(event1);

  // 断言 event0 所属设备为 CUDA 设备 1
  ASSERT_EQ_CUDA(event0.device(), at::Device(at::kCUDA, 1));

  // 在 stream0 上阻塞 event0
  event0.block(stream0);

  // 同步等待 stream0
  cudaStreamSynchronize(stream0);
  // 断言 event0 查询状态为真
  ASSERT_TRUE(event0.query());
}

// Generic Events 测试套件
TEST(TestStream, GenericInlineCUDAEventTest) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) return;

  // 创建内联 CUDA 事件对象 event
  c10::impl::InlineEvent<c10::cuda::impl::CUDAGuardImpl> event{c10::DeviceType::CUDA};
  // 从 CUDA 流池中获取流 stream
  c10::Stream stream = at::cuda::getStreamFromPool();

  // 记录事件发生到指定流
  event.record(stream);

  // 从 CUDA 流池中获取两个等待流
  const c10::Stream wait_stream0 = at::cuda::getStreamFromPool();
  const c10::Stream wait_stream1 = at::cuda::getStreamFromPool();

  // 阻塞事件，等待在两个不同的流上
  event.block(wait_stream0);
  event.block(wait_stream1);

  // 将 c10::Stream 转换为 at::cuda::CUDAStream 类型
  const at::cuda::CUDAStream cuda_stream{wait_stream0};
  // 同步等待 CUDA 流
  cudaStreamSynchronize(cuda_stream);

  // 断言事件查询状态为真
  ASSERT_TRUE(event.query());
}

// Generic Virtual CUDA Event 测试套件
TEST(TestStream, GenericVirtualCUDAEventTest) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) return;

  // 创建 CUDA 事件对象 event
  c10::Event event{c10::DeviceType::CUDA};
  // 从 CUDA 流池中获取流 stream
  c10::Stream stream = at::cuda::getStreamFromPool();

  // 记录事件发生一次到指定流
  event.recordOnce(stream);

  // 从 CUDA 流池中获取两个等待流
  const c10::Stream wait_stream0 = at::cuda::getStreamFromPool();
  const c10::Stream wait_stream1 = at::cuda::getStreamFromPool();

  // 在两个流上等待事件
  wait_stream0.wait(event);
  wait_stream1.wait(event);

  // 将 c10::Stream 转换为 at::cuda::CUDAStream 类型
  const at::cuda::CUDAStream cuda_stream{wait_stream0};
  // 同步等待 CUDA 流
  cudaStreamSynchronize(cuda_stream);

  // 断言事件查询状态为真
  ASSERT_TRUE(event.query());
  // 断言事件标志为默认标志
  ASSERT_TRUE(event.flag() == c10::EventFlag::PYTORCH_DEFAULT);
}

// 验证可以创建和使用外部流的测试套件
TEST(TestStream, ExternalTest) {
  // 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available())
    return;


// 空的 return 语句，用于立即结束当前函数的执行
    at::cuda::CUDAGuard device_guard(0);


// 创建一个 at::cuda::CUDAGuard 对象，将当前 CUDA 设备设置为编号 0


  cudaStream_t cuda_stream;
  cudaStreamCreateWithPriority(&cuda_stream, cudaStreamNonBlocking, -1);


// 声明一个 CUDA 流变量 cuda_stream，并使用非阻塞方式及最高优先级(-1)创建 CUDA 流


  at::cuda::CUDAStream myStream =
      at::cuda::getStreamFromExternal(cuda_stream, 0);


// 通过给定的 CUDA 流 cuda_stream 创建一个 at::cuda::CUDAStream 对象 myStream


  at::cuda::setCurrentCUDAStream(myStream);


// 设置当前线程的 CUDA 流为 myStream


  at::cuda::CUDAStream curStream = at::cuda::getCurrentCUDAStream();


// 获取当前线程的 CUDA 流，并将其保存在 curStream 变量中


  ASSERT_EQ_CUDA(curStream, myStream);


// 使用 ASSERT_EQ_CUDA 宏断言当前 CUDA 流 curStream 和之前创建的 myStream 相等


  ASSERT_EQ_CUDA(curStream.stream(), cuda_stream);


// 使用 ASSERT_EQ_CUDA 宏断言当前 CUDA 流 curStream 的底层 CUDA 流与之前创建的 cuda_stream 相等


  cudaStreamDestroy(cuda_stream);


// 销毁之前创建的 CUDA 流 cuda_stream
// 验证不同的外部流可以同时用于不同设备
TEST(TestStream, ExternalMultiDeviceTest) {
  // 如果CUDA不可用，则退出测试
  if (!at::cuda::is_available())
    return;
  // 如果系统中的GPU数量少于2个，则退出测试
  if (at::cuda::getNumGPUs() < 2)
    return;

  // 声明两个CUDA流对象
  cudaStream_t cuda_stream_0;
  cudaStream_t cuda_stream_1;

  {
    // 设置当前设备为第一个GPU（设备编号0）
    at::cuda::CUDAGuard device_guard(0);
    // 创建具有优先级的非阻塞CUDA流
    cudaStreamCreateWithPriority(&cuda_stream_0, cudaStreamNonBlocking, -1);
  }

  {
    // 设置当前设备为第二个GPU（设备编号1）
    at::cuda::CUDAGuard device_guard(1);
    // 创建具有优先级的非阻塞CUDA流
    cudaStreamCreateWithPriority(&cuda_stream_1, cudaStreamNonBlocking, -1);
  }

  // 通过外部流创建CUDAStream对象，关联到不同的GPU设备上
  at::cuda::CUDAStream myStream0 =
      at::cuda::getStreamFromExternal(cuda_stream_0, 0);
  at::cuda::CUDAStream myStream1 =
      at::cuda::getStreamFromExternal(cuda_stream_1, 1);

  // 设置当前线程的CUDA流为myStream0
  at::cuda::setCurrentCUDAStream(myStream0);
  // 断言当前设备上的CUDA流是否与myStream0相同
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(0), myStream0);

  // 设置当前线程的CUDA流为myStream1
  at::cuda::setCurrentCUDAStream(myStream1);
  // 断言第一个设备上的CUDA流是否与myStream0相同（因为在不同设备上，所以应该不同）
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(0), myStream0);
  // 断言第二个设备上的CUDA流是否与myStream1相同
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(1), myStream1);

  // 销毁CUDA流对象
  cudaStreamDestroy(cuda_stream_0);
  cudaStreamDestroy(cuda_stream_1);
}

// 验证外部流与Guard一起使用的功能，即使是嵌套的情况
TEST(TestStream, ExternalGuardTest) {
  // 如果CUDA不可用，则退出测试
  if (!at::cuda::is_available())
    return;
  
  // 设置当前设备为第一个GPU（设备编号0）
  at::cuda::CUDAGuard device_guard(0);

  // 声明两个CUDA流对象
  cudaStream_t a_cuda_stream;
  cudaStream_t another_cuda_stream;
  // 创建具有优先级的非阻塞CUDA流
  cudaStreamCreateWithPriority(&a_cuda_stream, cudaStreamNonBlocking, -1);
  cudaStreamCreateWithPriority(&another_cuda_stream, cudaStreamNonBlocking, -1);

  // 通过外部流创建CUDAStream对象，关联到设备0上
  at::cuda::CUDAStream myFirstStream =
      at::cuda::getStreamFromExternal(a_cuda_stream, 0);
  // 通过外部流创建CUDAStream对象，关联到设备0上
  at::cuda::CUDAStream mySecondStream =
      at::cuda::getStreamFromExternal(another_cuda_stream, 0);

  // 获取当前线程的CUDA流对象
  at::cuda::CUDAStream originalStream = at::cuda::getCurrentCUDAStream();
  
  {
    // 使用外部流对象创建CUDAStreamGuard，关联到myFirstStream
    at::cuda::CUDAStreamGuard outerGuard(myFirstStream);
    // 断言当前CUDA流对象与原始流对象是否相同
    ASSERT_EQ_CUDA(outerGuard.original_stream(), originalStream);
    // 断言当前CUDA流对象与myFirstStream是否相同
    ASSERT_EQ_CUDA(outerGuard.current_stream(), myFirstStream);
    // 断言当前设备上的CUDA流是否与myFirstStream相同
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), myFirstStream);
    
    {
      // 使用外部流对象创建CUDAStreamGuard，关联到mySecondStream
      at::cuda::CUDAStreamGuard innerGuard(mySecondStream);
      // 断言内部Guard的原始CUDA流对象与外部Guard的当前CUDA流对象相同
      ASSERT_EQ_CUDA(innerGuard.original_stream(), myFirstStream);
      // 断言内部Guard的当前CUDA流对象与mySecondStream相同
      ASSERT_EQ_CUDA(innerGuard.current_stream(), mySecondStream);
      // 断言当前设备上的CUDA流是否与mySecondStream相同
      ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), mySecondStream);
    }

    // 断言外部Guard的原始CUDA流对象与原始流对象相同
    ASSERT_EQ_CUDA(outerGuard.original_stream(), originalStream);
    // 断言外部Guard的当前CUDA流对象与myFirstStream相同
    ASSERT_EQ_CUDA(outerGuard.current_stream(), myFirstStream);
    // 断言当前设备上的CUDA流是否与myFirstStream相同
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), myFirstStream);

    // 重置外部Guard的CUDA流对象为mySecondStream
    outerGuard.reset_stream(mySecondStream);
    // 断言外部Guard的原始CUDA流对象与原始流对象相同
    ASSERT_EQ_CUDA(outerGuard.original_stream(), originalStream);
    // 断言外部Guard的当前CUDA流对象与mySecondStream相同
    ASSERT_EQ_CUDA(outerGuard.current_stream(), mySecondStream);
    // 断言当前设备上的CUDA流是否与mySecondStream相同
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), mySecondStream);
  }

  // 断言当前设备上的CUDA流是否与原始流对象相同
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), originalStream);

  // 销毁CUDA流对象
  cudaStreamDestroy(a_cuda_stream);
  cudaStreamDestroy(another_cuda_stream);
}
// 定义一个测试用例 TEST，命名为 TestStream，测试外部多线程情况
TEST(TestStream, ExternalMultiThreadTest) {
  // 如果 CUDA 可用，则执行测试，否则返回
  if (!at::cuda::is_available())
    return;
  
  // 设置当前 CUDA 设备为设备 0
  at::cuda::CUDAGuard device_guard(0);

  // 声明两个 CUDA 流变量
  cudaStream_t cuda_stream_a;
  cudaStream_t cuda_stream_b;

  // 创建具有最高优先级的非阻塞 CUDA 流 cuda_stream_a 和 cuda_stream_b
  cudaStreamCreateWithPriority(&cuda_stream_a, cudaStreamNonBlocking, -1);
  cudaStreamCreateWithPriority(&cuda_stream_b, cudaStreamNonBlocking, -1);

  // 将 cuda_stream_a 和 cuda_stream_b 转换为 PyTorch 的 CUDAStream 对象
  at::cuda::CUDAStream myStreamA =
      at::cuda::getStreamFromExternal(cuda_stream_a, 0);
  at::cuda::CUDAStream myStreamB =
      at::cuda::getStreamFromExternal(cuda_stream_b, 0);

  // 创建两个 std::promise 对象，用于线程间的同步
  std::promise<void> aToBProm;
  std::promise<void> bToAProm;

  // 声明一个 std::optional 变量，用于存储找到的 CUDA 流对象
  std::optional<at::cuda::CUDAStream> foundStream;

  // 创建线程 threadA，实现如下功能
  std::thread threadA([&]() {
    // 设置当前 CUDA 设备为设备 0
    at::cuda::CUDAGuard device_guard(0);

    // 设置当前 CUDA 流为 myStreamA
    at::cuda::setCurrentCUDAStream(myStreamA);

    // 设置 aToBProm 的值为已完成状态
    aToBProm.set_value();

    // 等待 bToAProm 的未来状态完成
    bToAProm.get_future().wait();

    // 获取当前 CUDA 流，并存储到 foundStream 中
    foundStream = at::cuda::getCurrentCUDAStream();
  });

  // 创建线程 threadB，实现如下功能
  std::thread threadB([&]() {
    // 设置当前 CUDA 设备为设备 0
    at::cuda::CUDAGuard device_guard(0);

    // 等待 aToBProm 的未来状态完成
    aToBProm.get_future().wait();

    // 设置当前 CUDA 流为 myStreamB
    at::cuda::setCurrentCUDAStream(myStreamB);

    // 设置 bToAProm 的值为已完成状态
    bToAProm.set_value();
  });

  // 等待线程 threadA 和 threadB 执行完成
  threadA.join();
  threadB.join();

  // 使用 ASSERT_EQ_CUDA 断言，验证 foundStream 是否等于 myStreamA
  ASSERT_EQ_CUDA(*foundStream, myStreamA);

  // 销毁 cuda_stream_a 和 cuda_stream_b
  cudaStreamDestroy(cuda_stream_a);
  cudaStreamDestroy(cuda_stream_b);
}
```