# `.\pytorch\test\cpp\c10d\ProcessGroupGlooTest.cpp`

```py
#ifndef _WIN32
// 如果不是在 Windows 下编译，则包含以下头文件
#include <signal.h>             // POSIX 信号处理相关函数
#include <sys/wait.h>           // 进程等待相关函数
#include <unistd.h>             // POSIX 标准函数和常量定义

#endif

#include <sys/types.h>          // UNIX 系统基本数据类型定义

#include <condition_variable>   // 条件变量，用于线程同步
#include <iostream>             // 标准输入输出流
#include <mutex>                // 互斥量，用于线程同步
#include <sstream>              // 字符串流
#include <thread>               // 线程

#include <gtest/gtest.h>        // Google 测试框架
#include <torch/csrc/autograd/profiler.h>   // PyTorch 自动求导分析器
#include <torch/cuda.h>         // PyTorch CUDA 模块

#include <c10/util/irange.h>    // C10 实用工具，提供迭代范围
#include <torch/csrc/distributed/c10d/FileStore.hpp>   // PyTorch 分布式文件存储
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>   // PyTorch Gloo 进程组
#include "TestUtils.hpp"        // 自定义测试工具

using namespace c10d::test;     // 使用 c10d::test 命名空间
using namespace torch::autograd::profiler;   // 使用 torch::autograd::profiler 命名空间

constexpr auto kSendDelay = std::chrono::milliseconds(100);    // 定义发送延迟为 100 毫秒
constexpr auto kWaitTimeout = std::chrono::milliseconds(1);    // 定义等待超时时间为 1 毫秒

#ifndef _WIN32
// SignalTest 类，用于测试信号处理
class SignalTest {
 public:
  SignalTest(const std::string& path) : path_(path) {}   // 构造函数，接收路径参数

  ~SignalTest() {   // 析构函数，用于清理资源
    if (arm_.joinable()) {    // 如果线程可加入（joinable）
      arm_.join();    // 等待线程结束
    }
  }

  // arm 方法，用于向指定 PID 发送信号
  // 当信号量解锁时（即第一个集合完成成功时），触发信号发送
  void arm(int pid, int signal) {
    arm_ = std::thread([=] {
      sem_.wait();    // 等待信号量解锁
      kill(pid, signal);    // 向指定 PID 发送信号
    });
  }

  // run 方法，运行测试并返回 Work 对象
  c10::intrusive_ptr<::c10d::Work> run(int rank, int size) {
    auto store = c10::make_intrusive<::c10d::FileStore>(path_, size);   // 创建文件存储对象

    auto options = ::c10d::ProcessGroupGloo::Options::create();    // 创建 Gloo 进程组选项对象
    // 设置小超时时间以加速测试运行，确保在 ProcessGroupGloo 构造函数中不会超时
    options->timeout = std::chrono::milliseconds(1000);
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));    // 添加设备地址

    ::c10d::ProcessGroupGloo pg(store, rank, size, options);    // 创建 Gloo 进程组对象

    // 初始化张量列表
    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),    // 创建大小为 16x16 的张量，填充为 1
    };

    // 循环直到发生异常
    c10::intrusive_ptr<::c10d::Work> work;
    while (true) {
      work = pg.allreduce(tensors);   // 执行所有张量的全局归约操作
      try {
        work->wait();   // 等待工作完成
      } catch (const std::exception& e) {
        break;    // 捕获异常后退出循环
      }
      sem_.post();    // 发送信号量解锁信号
    }

    return work;    // 返回工作对象
  }

 protected:
  std::string path_;    // 文件路径
  std::thread arm_;     // 线程对象
  Semaphore sem_;       // 信号量对象
};

// testSignal 函数，用于测试信号处理
c10::intrusive_ptr<::c10d::Work> testSignal(
    const std::string& path,
    int signal) {
  Fork fork;    // 创建 Fork 对象
  if (fork.isChild()) {    // 如果是子进程
    SignalTest test(path);    // 创建 SignalTest 对象
    test.run(1, 2);    // 运行测试
    exit(1);    // 退出子进程
  }

  SignalTest test(path);    // 创建 SignalTest 对象
  test.arm(fork.pid, signal);    // 向 Fork 对象的 PID 发送信号
  return test.run(0, 2);    // 运行测试并返回工作对象
}
#endif

// ProcessGroupGlooDelayed 类，继承自 ProcessGroupGloo 类
class ProcessGroupGlooDelayed : public ::c10d::ProcessGroupGloo {
 public:
  // 构造函数，接收存储对象、排名、大小和选项参数
  ProcessGroupGlooDelayed(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options)
      : ProcessGroupGloo(store, rank, size, options) {}

  // send 方法重载，延迟发送方法
  c10::intrusive_ptr<::c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    std::this_thread::sleep_for(kSendDelay);    // 等待发送延迟时间
    return ::c10d::ProcessGroupGloo::send(tensors, dstRank, tag);    // 调用父类的发送方法
  }
};
// 定义 CollectiveTest 类
class CollectiveTest {
 public:
  // 静态方法，初始化指定数量的 CollectiveTest 实例，并返回向量
  static std::vector<CollectiveTest> initialize(
      const std::string& path,
      int num,
      bool delayed = false) {
    // 创建空的 CollectiveTest 实例向量
    std::vector<CollectiveTest> tests;
    // 循环创建 num 个 CollectiveTest 实例并添加到向量中
    for (C10_UNUSED const auto i : c10::irange(num)) {
      tests.emplace_back(CollectiveTest(path));
    }

    // 创建线程向量
    std::vector<std::thread> threads;
    // 对每个测试实例创建一个线程
    for (const auto i : c10::irange(num)) {
      threads.emplace_back(std::thread(
          // 使用 lambda 表达式启动测试实例的 start 方法
          [i, &tests, delayed] { tests[i].start(i, tests.size(), delayed); }));
    }
    // 等待所有线程执行完毕
    for (auto& thread : threads) {
      thread.join();
    }

    // 返回初始化后的测试实例向量
    return tests;
  }

  // 构造函数，初始化 CollectiveTest 实例
  CollectiveTest(std::string path) : path_(std::move(path)) {}

  // 移动构造函数，从另一个 CollectiveTest 实例中移动数据
  CollectiveTest(CollectiveTest&& other) {
    path_ = std::move(other.path_);
    pg_ = std::move(other.pg_);
  }

  // 获取 ProcessGroupGloo 的引用
  ::c10d::ProcessGroupGloo& getProcessGroup() {
    return *pg_;
  }

  // 启动测试，创建 ProcessGroupGloo 实例并存储在 pg_ 中
  void start(int rank, int size, bool delayed) {
    // 创建 FileStore 实例
    auto store = c10::make_intrusive<::c10d::FileStore>(path_, size);

    // 设置 ProcessGroupGloo 的选项
    auto options = ::c10d::ProcessGroupGloo::Options::create();
    options->timeout = std::chrono::milliseconds(1000);
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));

    // 根据 delayed 参数选择不同的 ProcessGroupGloo 类型
    if (!delayed) {
      pg_ = std::unique_ptr<::c10d::ProcessGroupGloo>(
          new ::c10d::ProcessGroupGloo(store, rank, size, options));
    } else {
      pg_ = std::unique_ptr<ProcessGroupGlooDelayed>(
          new ProcessGroupGlooDelayed(store, rank, size, options));
    }
  }

 protected:
  std::string path_;  // 存储路径
  std::unique_ptr<::c10d::ProcessGroupGloo> pg_;  // 进程组对象的指针
};

// 复制张量的函数
std::vector<std::vector<at::Tensor>> copyTensors(
    const std::vector<std::vector<at::Tensor>>& inputs) {
  // 创建输出张量的向量
  std::vector<std::vector<at::Tensor>> outputs(inputs.size());
  // 遍历输入张量向量
  for (const auto i : c10::irange(inputs.size())) {
    const auto& input = inputs[i];
    // 创建当前输入张量的输出向量
    std::vector<at::Tensor> output(input.size());
    // 复制每个张量到 CPU
    for (const auto j : c10::irange(input.size())) {
      output[j] = input[j].cpu();
    }
    // 将当前输出向量存储到输出张量向量中
    outputs[i] = output;
  }
  // 返回复制后的输出张量向量
  return outputs;
}

// 等待工作的函数
std::vector<std::vector<at::Tensor>> waitWork(
    std::vector<c10::intrusive_ptr<c10d::Work>> works) {
  // 存储输出张量的向量
  std::vector<std::vector<at::Tensor>> outputTensors;
  // 遍历工作指针向量
  for (auto& work : works) {
    try {
      // 等待工作完成
      work->wait();
    } catch (const std::exception& ex) {
      // 捕获并记录异常
      LOG(ERROR) << "Exception received: " << ex.what() << std::endl;
    }
    // 将工作结果添加到输出张量向量中
    outputTensors.emplace_back(work->result());
  }
  // 复制并返回输出张量向量
  return copyTensors(outputTensors);
}

// 等待未来的函数
std::vector<std::vector<at::Tensor>> waitFuture(
    std::vector<c10::intrusive_ptr<c10d::Work>> works) {
  // 存储输出张量的向量
  std::vector<std::vector<at::Tensor>> outputTensors;
  // 遍历工作指针向量
  for (auto& work : works) {
    auto fut = work->getFuture();
    try {
      // 等待未来完成
      fut->wait();
    } catch (const std::exception& ex) {
      // 捕获并记录异常
      LOG(ERROR) << "Exception received: " << ex.what() << std::endl;
    }
    // 将工作结果添加到输出张量向量中
    outputTensors.emplace_back(work->result());
  }
  // 复制并返回输出张量向量
  return copyTensors(outputTensors);
}
    # 获取异步操作的结果值
    auto result = fut->value();
    # 检查结果是否为空
    if (result.isNone()) {
      # 如果结果为空，向输出张量列表添加一个空张量
      outputTensors.emplace_back();
    } else if (result.isTensorList()) {
      # 如果结果是张量列表，将其转换为标准张量并添加到输出张量列表中
      outputTensors.emplace_back(result.toTensorVector());
    } else {
      # 如果结果既不是空也不是张量列表，则抛出错误信息
      TORCH_CHECK(false, "future result should be tensor list or none");
    }
  }
  # 返回复制后的输出张量列表
  return copyTensors(outputTensors);
// 检查事件列表中是否存在符合预期的分析事件
void checkProfiledEvents(
    const thread_event_lists& event_lists,  // 线程事件列表
    const char* expected_profile_str,       // 预期的分析事件名称
    int expected_count,                     // 预期的事件数量
    std::vector<std::vector<int64_t>> expected_shapes,  // 预期的形状向量
    bool verify_shapes = true) {            // 是否验证形状，默认为真
  if (verify_shapes) {
    EXPECT_EQ(expected_count, expected_shapes.size());  // 检查预期的形状数量是否与实际一致
  }
  std::vector<bool> matched_shapes(expected_count);    // 创建一个布尔向量来标记匹配的形状
  for (const auto& li : event_lists) {                 // 遍历事件列表
    for (const auto& evt : li) {                       // 遍历事件列表中的每个事件
      auto match = !strcmp(evt.name(), expected_profile_str);  // 检查事件名称是否与预期一致
      if (verify_shapes && match) {                    // 如果需要验证形状且匹配成功
        auto shapesVec = evt.shapes();                 // 获取事件的形状
        for (const auto i : c10::irange(expected_count)) {
          // 假设：没有两个预期形状是相同的
          if (shapesVec[0] == expected_shapes[i]) {    // 检查事件的第一个形状是否与预期形状匹配
            matched_shapes[i] = true;                  // 标记该形状为匹配
          }
        }
      }
    }
  }
  if (verify_shapes) {
    for (bool match : matched_shapes) {
      EXPECT_TRUE(match);   // 验证所有预期形状是否都已匹配成功
    }
  }
}

// 测试全局归约操作
void testAllreduce(
    const std::string& path,              // 路径
    const at::DeviceType b,               // 设备类型
    const at::ScalarType dtype = at::kFloat) {  // 标量类型，默认为浮点型
  const auto size = 4;                    // 操作的大小
  auto tests = CollectiveTest::initialize(path, size);  // 初始化测试

  // 生成输入数据
  std::vector<std::vector<at::Tensor>> inputs(size);  // 输入张量的向量
  std::vector<std::vector<int64_t>> allShapes;        // 所有形状的向量
  std::vector<int64_t> shapes = {16, 16};              // 张量的形状
  for (const auto i : c10::irange(size)) {
    auto tensor = at::ones(shapes, at::dtype(dtype).device(b)) * i;  // 创建指定形状的张量
    std::vector<int64_t> shapesVec = shapes;           // 复制形状到形状向量
    allShapes.emplace_back(std::move(shapesVec));      // 添加形状向量到所有形状
    inputs[i] = std::vector<at::Tensor>({tensor});     // 将张量添加到输入向量
  }

  // 启动工作
  std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);  // 工作的指针向量
  const char* GLOO_ALLREDUCE_STR = "gloo:all_reduce";        // GLOO 全局归约字符串
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));  // 启用旧版分析器
  for (const auto i : c10::irange(size)) {
    work[i] = tests[i].getProcessGroup().allreduce(inputs[i]);  // 对输入进行全局归约操作
  }
  // 等待工作完成
  auto outputs = waitFuture(work);  // 等待工作的完成

  auto event_lists = disableProfilerLegacy();  // 禁用旧版分析器，获取事件列表
  checkProfiledEvents(
      std::move(event_lists), GLOO_ALLREDUCE_STR, size, allShapes);  // 检查分析事件

  // 验证输出结果
  const auto expected = (size * (size - 1)) / 2;  // 期望的输出值
  for (const auto i : c10::irange(size)) {
    auto tensor = outputs[i][0].to(at::kFloat);  // 将输出张量转换为浮点型
    auto data = tensor.data_ptr<float>();        // 获取张量的数据指针
    for (const auto j : c10::irange(tensor.numel())) {
      EXPECT_EQ(data[j], expected);              // 检查张量每个元素是否等于期望值
    }
  }
}

// 测试使用工作 API 的全局归约操作
void testAllreduceUsingWorkAPI(
    const std::string& path,              // 路径
    const at::DeviceType b,               // 设备类型
    const at::ScalarType dtype = at::kFloat) {  // 标量类型，默认为浮点型
  const auto size = 4;                    // 操作的大小
  auto tests = CollectiveTest::initialize(path, size);  // 初始化测试

  // 生成输入数据
  std::vector<std::vector<at::Tensor>> inputs(size);  // 输入张量的向量
  std::vector<std::vector<int64_t>> allShapes;        // 所有形状的向量
  std::vector<int64_t> shapes = {16, 16};              // 张量的形状
  for (const auto i : c10::irange(size)) {
    // 使用给定的形状创建一个元素全为1的张量，并乘以当前迭代索引 i
    auto tensor = at::ones(shapes, at::dtype(dtype).device(b)) * i;
    // 将张量的形状转换为 std::vector<int64_t> 类型
    std::vector<int64_t> shapesVec = shapes;
    // 将 shapesVec 移动到 allShapes 的末尾
    allShapes.emplace_back(std::move(shapesVec));
    // 将当前张量作为单元素向量存入 inputs[i]
    inputs[i] = std::vector<at::Tensor>({tensor});
  }

  // 启动工作
  // 创建一个 c10d::Work 对象的 vector，大小为 size
  std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);
  // 定义 GLOO_ALLREDUCE_STR 常量为 "gloo:all_reduce"
  const char* GLOO_ALLREDUCE_STR = "gloo:all_reduce";
  // 启用旧版分析器，配置为 CPU 分析，报告输入形状，但不报告内存使用情况
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  // 对于每个索引 i 在 [0, size) 范围内
  for (const auto i : c10::irange(size)) {
    // 调用 tests[i] 的 getProcessGroup() 方法执行 allreduce 操作，并将结果存入 work[i]
    work[i] = tests[i].getProcessGroup().allreduce(inputs[i]);
  }
  // 等待工作完成
  auto outputs = waitWork(work);

  // 禁用旧版分析器，并获取事件列表
  auto event_lists = disableProfilerLegacy();
  // 检查分析器记录的事件列表，以确保正确性
  checkProfiledEvents(
      std::move(event_lists), GLOO_ALLREDUCE_STR, size, allShapes);

  // 验证输出结果
  // 计算期望的输出值
  const auto expected = (size * (size - 1)) / 2;
  // 对于每个索引 i 在 [0, size) 范围内
  for (const auto i : c10::irange(size)) {
    // 将 outputs[i][0] 转换为 float 类型的张量
    auto tensor = outputs[i][0].to(at::kFloat);
    // 获取张量的数据指针
    auto data = tensor.data_ptr<float>();
    // 对张量中的每个元素进行验证，期望值为 expected
    for (const auto j : c10::irange(tensor.numel())) {
      EXPECT_EQ(data[j], expected);
    }
  }
}

// 测试广播操作的函数
void testBroadcast(
    const std::string& path,               // 输入参数：路径
    const at::DeviceType b,                // 输入参数：设备类型
    const at::ScalarType dtype = at::kFloat // 输入参数：张量数据类型，默认为浮点型
) {
  const auto size = 2;  // 常量：测试组数量
  const auto stride = 2;  // 常量：张量步长
  auto tests = CollectiveTest::initialize(path, size);  // 初始化测试集合

  std::vector<std::vector<at::Tensor>> inputs(size);  // 输入张量的二维向量
  std::vector<int64_t> shapes = {16, 16};  // 张量形状
  // 尝试每种根排名和根张量的排列组合
  for (const auto i : c10::irange(size)) {
    for (const auto j : c10::irange(stride)) {
      std::vector<std::vector<int64_t>> allShapes;  // 所有张量形状的向量
      // 初始化输入张量
      for (const auto k : c10::irange(size)) {
        std::vector<int64_t> shapesVec = shapes;  // 复制张量形状
        allShapes.emplace_back(std::move(shapesVec));  // 添加到所有张量形状中
        inputs[k].resize(stride);  // 调整输入张量大小
        // 如果支持稀疏 CUDA，则不适用
        at::OptionalDeviceGuard deviceGuard;
        for (const auto l : c10::irange(stride)) {
          if (b == at::DeviceType::CUDA) {
            deviceGuard.reset_device(at::Device(at::kCUDA, l));  // 重置设备
          }
          inputs[k][l] =  // 创建张量并初始化
              at::ones(shapes, at::dtype(dtype).device(b)) * (k * stride + l);
        }
      }

      ::c10d::BroadcastOptions options;  // 广播选项
      options.rootRank = i;  // 根排名
      options.rootTensor = j;  // 根张量

      // 开始执行工作
      const char* GLOO_BROADCAST_STR = "gloo:broadcast";
      enableProfilerLegacy(ProfilerConfig(
          ProfilerState::CPU, /* report_input_shapes */ true, false));  // 启用旧版性能分析器
      std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);  // 工作单元的向量

      for (const auto i : c10::irange(size)) {
        work[i] = tests[i].getProcessGroup().broadcast(inputs[i], options);  // 广播输入张量
      }

      // 等待工作完成
      auto outputs = waitFuture(work);  // 等待工作的完成

      auto event_lists = disableProfilerLegacy();  // 禁用旧版性能分析器
      checkProfiledEvents(
          std::move(event_lists), GLOO_BROADCAST_STR, size, allShapes);  // 检查分析事件

      // 验证输出
      const auto expected = (i * stride + j);  // 预期值
      for (const auto k : c10::irange(size)) {
        for (const auto l : c10::irange(stride)) {
          auto tensor = outputs[k][l].to(at::kFloat);  // 转换为浮点张量
          auto data = tensor.data_ptr<float>();  // 获取数据指针
          for (const auto n : c10::irange(tensor.numel())) {
            EXPECT_EQ(data[n], expected);  // 断言每个数据点的值是否符合预期
          }
        }
      }
    }
  }
}

// 测试Alltoall操作的函数
void testAlltoall(const std::string& path, const at::DeviceType b) {
  const auto size = 4;  // 常量：测试组数量
  auto tests = CollectiveTest::initialize(path, size);  // 初始化测试集合

  // 生成输入张量
  std::vector<at::Tensor> inputs(size);  // 输入张量的向量
  std::vector<std::vector<int32_t>> blobs = {  // 数据块的二维向量
      {0, 1, 2, 3, 4, 5},
      {10, 11, 12, 13, 14, 15, 16, 17, 18},
      {20, 21, 22, 23, 24},
      {30, 31, 32, 33, 34, 35, 36},
  };
  for (const auto rank : c10::irange(size)) {
    const std::vector<int32_t>& blob = blobs[rank];  // 获取当前数据块
    inputs[rank] = at::from_blob((int32_t*)(blob.data()), blob.size()).to(b);  // 从数据块创建张量
  }

  // 分配输出张量
  std::vector<at::Tensor> outputs(size);  // 输出张量的向量
  std::vector<int> outputLengths = {9, 7, 6, 5};  // 输出长度的向量
  for (const auto rank : c10::irange(size)) {
    outputs[rank] =
        at::empty(outputLengths[rank], c10::TensorOptions(at::kInt).device(b));
  }

  // Generate splits
  // 定义输入和输出数据的分割模式
  std::vector<std::vector<int64_t>> inputSplits = {
      {2, 2, 1, 1},  // 第一个输入分割模式
      {3, 2, 2, 2},  // 第二个输入分割模式
      {2, 1, 1, 1},  // 第三个输入分割模式
      {2, 2, 2, 1},  // 第四个输入分割模式
  };
  std::vector<std::vector<int64_t>> outputSplits = {
      {2, 3, 2, 2},  // 第一个输出分割模式
      {2, 2, 1, 2},  // 第二个输出分割模式
      {1, 2, 1, 2},  // 第三个输出分割模式
      {1, 2, 1, 1},  // 第四个输出分割模式
  };

  // Kick off work
  // 初始化工作数组，用于保存并行任务
  std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);
  const char* GLOO_A2A_STR = "gloo:all_to_all";
  std::vector<std::vector<int64_t>> allShapes;
  for (const auto& vec : inputSplits) {
    // 由于张量的串联，形状实际上将是它们长度的总和
    int64_t sum = 0;
    for (const auto& s : vec) {
      sum += s;
    }
    // 将计算出的形状总和添加到 allShapes 中
    allShapes.push_back({sum});
  }
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  // 对每个进程执行 all-to-all 操作，并将结果存储在 work 数组中
  for (const auto rank : c10::irange(size)) {
    work[rank] = tests[rank].getProcessGroup().alltoall_base(
        outputs[rank], inputs[rank], outputSplits[rank], inputSplits[rank]);
  }

  // Wait for work to complete
  // 等待所有工作完成
  for (const auto i : c10::irange(size)) {
    work[i]->wait();
  }

  auto event_lists = disableProfilerLegacy();
  // 检查分析器记录的事件列表，以验证 all-to-all 操作
  checkProfiledEvents(std::move(event_lists), GLOO_A2A_STR, size, allShapes);
  // Verify outputs
  // 验证输出结果是否符合预期
  std::vector<std::vector<int32_t>> expected = {
      {0, 1, 10, 11, 12, 20, 21, 30, 31},  // 第一个进程的预期输出
      {2, 3, 13, 14, 22, 32, 33},          // 第二个进程的预期输出
      {4, 15, 16, 23, 34, 35},             // 第三个进程的预期输出
      {5, 17, 18, 24, 36},                 // 第四个进程的预期输出
  };
  // 遍历每个进程的输出结果，并逐个验证其值是否符合预期
  for (const auto rank : c10::irange(size)) {
    at::Tensor tensor = outputs[rank].cpu();
    EXPECT_EQ(tensor.numel(), expected[rank].size());
    auto data = tensor.data_ptr<int32_t>();
    for (const auto j : c10::irange(tensor.numel())) {
      EXPECT_EQ(data[j], expected[rank][j]);
    }
  }
}

void testBarrier(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);

  // Kick off work
  // 启动性能分析器，配置为记录 CPU 数据，报告输入形状，不报告输出形状
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  // 创建一个工作数组，包含两个工作项
  std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);
  // 对每个进程组执行屏障操作，并将结果保存在工作数组中
  for (const auto i : c10::irange(size)) {
    work[i] = tests[i].getProcessGroup().barrier();
  }

  // Wait for work to complete
  // 等待工作完成
  waitFuture(work);

  // 关闭性能分析器，返回记录的事件列表
  auto event_lists = disableProfilerLegacy();
  const char* GLOO_STR = "gloo:barrier";
  // 创建一个空的所有形状数组
  std::vector<std::vector<int64_t>> allShapes;
  // 检查记录的事件，跳过形状验证，因为屏障操作不涉及张量
  checkProfiledEvents(
      std::move(event_lists),
      GLOO_STR,
      size,
      allShapes,
      /* verify_shapes */ false);
}

void testMonitoredBarrier(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);
  // 非失败情况：所有进程通过监控屏障
  auto runMonitoredBarrier = [&](int i) {
    tests[i].getProcessGroup().monitoredBarrier();
  };
  std::vector<std::thread> threads;
  threads.reserve(size);
  // 启动多个线程，每个线程调用监控屏障函数
  for (const auto r : c10::irange(size)) {
    threads.emplace_back(std::thread([=]() { runMonitoredBarrier(r); }));
  }
  // 等待所有线程结束
  for (auto& t : threads) {
    t.join();
  }
  // 失败情况：只有排名为 0 的进程调用监控屏障，应该引发错误
  auto runMonitoredBarrierWithException = [&](int i) {
    if (i != 0) {
      return;
    }

    try {
      // 仅排名为 0 的进程调用监控屏障，预期会抛出异常
      tests[i].getProcessGroup().monitoredBarrier();
      FAIL() << "Exception should have been thrown.";
    } catch (const std::exception& e) {
      // 检查异常消息中是否包含 "Rank 1"
      auto pos = std::string(e.what()).find("Rank 1");
      EXPECT_TRUE(pos != std::string::npos);
    }
  };
  threads.clear();
  // 启动多个线程，每个线程调用具有异常检查的监控屏障函数
  for (const auto r : c10::irange(size)) {
    threads.emplace_back(
        std::thread([=]() { runMonitoredBarrierWithException(r); }));
  }
  // 等待所有线程结束
  for (auto& t : threads) {
    t.join();
  }
}

void testSequenceNumInit(const std::string& path) {
  const auto size = 4;
  auto tests = CollectiveTest::initialize(path, size);
  // 为每个进程组设置序列号
  for (const auto i : c10::irange(size)) {
    tests[i].getProcessGroup().setSequenceNumberForGroup();
  }

  // 创建一个无序集合来存储序列号
  std::unordered_set<uint64_t> nums;
  // 获取每个进程组的序列号，并将其插入到无序集合中
  for (const auto i : c10::irange(size)) {
    auto seqNum = tests[i].getProcessGroup().getSequenceNumberForGroup();
    nums.insert(seqNum);
  }
  // 检查无序集合中序列号的唯一性
  EXPECT_EQ(nums.size(), 1);
}

void testWaitDelay(const std::string& path) {
  const auto size = 2;
  // 使用延迟模式初始化集体测试
  auto tests = CollectiveTest::initialize(path, size, /* delay */ true);

  constexpr uint64_t tag = 0x1337;
  // 测试等待工作发送能够成功中止
  auto selfRank = 0;
  auto dstRank = 1;
  // 创建一个包含张量的向量
  std::vector<at::Tensor> tensors = {
      at::ones({16, 16}),
  };
  // 获取当前进程组，并发送张量到目标进程
  auto& pg = tests[selfRank].getProcessGroup();
  auto sendWork = pg.send(tensors, dstRank, tag);
  // 预期发送工作等待超时时抛出异常
  EXPECT_THROW(sendWork->wait(kWaitTimeout), std::exception);
}
void testSend(const std::string& path) {
  // 初始化测试环境，使用指定路径和大小初始化 CollectiveTest 实例
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);

  // 定义消息标签
  constexpr uint64_t tag = 0x1337;
  
  // 测试等待发送完成可以成功中止的情况
  auto selfRank = 0;
  auto dstRank = 1;
  
  // 定义张量的形状和数据
  std::vector<int64_t> shapes{16, 16};
  std::vector<std::vector<int64_t>> allShapes;
  allShapes.push_back(shapes);
  std::vector<at::Tensor> tensors = {
      at::ones(shapes),
  };
  
  // 获取当前进程组
  auto& pg = tests[selfRank].getProcessGroup();
  
  // 定义 GLOO 发送字符串常量
  const char* GLOO_SEND_STR = "gloo:send";
  
  // 启用旧版分析器配置，报告输入形状，不报告详细信息
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  
  // 发送工作任务
  auto sendWork = pg.send(tensors, dstRank, tag);
  bool sendCompleted;
  
  // 创建等待发送任务中止的线程
  std::thread waitSendThreadAbort([&]() { sendCompleted = sendWork->wait(); });
  
  // 中止发送任务
  sendWork->abort();
  
  // 等待发送任务中止完成
  waitSendThreadAbort.join();
  
  // 检查发送是否未完成
  EXPECT_FALSE(sendCompleted);
  
  // 禁用旧版分析器，返回事件列表
  auto event_lists = disableProfilerLegacy();
  
  // 检查分析事件是否符合预期
  checkProfiledEvents(std::move(event_lists), GLOO_SEND_STR, 1, allShapes);

  // 创建一个单独的发送线程，以确保未来的等待发送可以成功完成。

  // 辅助接收者线程，模拟真实的接收/发送对
  std::thread recvThread([&]() {
    auto selfRank = 1;
    auto srcRank = 0;
    auto& pg = tests[selfRank].getProcessGroup();
    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };

    auto recvWork = pg.recv(tensors, srcRank, tag);
    recvWork->wait();
  });

  // 发送者线程
  std::thread sendThread([&]() { sendCompleted = sendWork->wait(); });
  sendThread.join();
  recvThread.join();
  
  // 检查发送是否已完成
  EXPECT_TRUE(sendCompleted);
}

void testRecv(const std::string& path) {
  // 初始化测试环境，使用指定路径和大小初始化 CollectiveTest 实例
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);
  
  // 定义消息标签
  constexpr uint64_t tag = 0x1337;
  
  // 测试等待接收完成可以成功中止的情况
  auto selfRank = 0;
  auto srcRank = 1;
  
  // 定义张量的形状和数据
  std::vector<int64_t> shapes = {16, 16};
  std::vector<std::vector<int64_t>> allShapes;
  allShapes.push_back(shapes);
  std::vector<at::Tensor> tensors = {
      at::ones(shapes),
  };
  
  // 定义 GLOO 接收字符串常量
  const char* GLOO_RECV_STR = "gloo:recv";
  
  // 获取当前进程组
  auto& pg = tests[selfRank].getProcessGroup();
  
  // 启用旧版分析器配置，报告输入形状，不报告详细信息
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  
  // 接收工作任务
  auto recvWork = pg.recv(tensors, srcRank, tag);
  bool recvCompleted;
  
  // 创建等待接收任务中止的线程
  std::thread waitRecvThreadAbort([&]() { recvCompleted = recvWork->wait(); });
  
  // 中止接收任务
  recvWork->abort();
  
  // 等待接收任务中止完成
  waitRecvThreadAbort.join();
  
  // 检查接收是否未完成
  EXPECT_FALSE(recvCompleted);
  
  // 禁用旧版分析器，返回事件列表
  auto event_lists = disableProfilerLegacy();
  
  // 检查分析事件是否符合预期
  checkProfiledEvents(std::move(event_lists), GLOO_RECV_STR, 1, allShapes);

  // 创建一个单独的接收者线程，以确保未来的等待接收可以成功完成。

  // 辅助发送者线程，模拟真实的接收/发送对
  std::thread senderThread([&]() {
    auto selfRank = 1;
    auto destRank = 0;
    auto& pg = tests[selfRank].getProcessGroup();
    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };

    auto sendWork = pg.send(tensors, destRank, tag);
    sendWork->wait();
  });

  // 接收者线程
  std::thread recvThread([&]() { recvCompleted = recvWork->wait(); });
  recvThread.join();
  senderThread.join();
  
  // 检查接收是否已完成
  EXPECT_TRUE(recvCompleted);
}
    // 获取当前进程的测试对象中指定排名的进程组
    auto& pg = tests[selfRank].getProcessGroup();

    // 创建包含一个 16x16 的全 1 张量的向量
    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };

    // 向目标排名发送张量数据，使用指定的标签
    auto sendWork = pg.send(tensors, destRank, tag);

    // 等待发送操作完成
    sendWork->wait();

    // 发送线程。
    std::thread senderThread([&]() {
        // 接收线程的完成状态
        recvCompleted = recvWork->wait();
    });

    // 接收线程。
    std::thread receiverThread([&]() {
        // 等待接收操作完成并将完成状态赋给 recvCompleted
        recvCompleted = recvWork->wait();
    });

    // 等待发送线程结束
    senderThread.join();

    // 等待接收线程结束
    receiverThread.join();

    // 断言接收操作已完成
    EXPECT_TRUE(recvCompleted);
}

// 测试存储设置和获取功能
void testStoreSetGet(const std::string& path) {
  // 初始化一个具有指定大小的 CollectiveTest 对象集合
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);

  // 测试 get() 方法是否能获取与 set() 方法设置的相同值
  std::vector<uint8_t> testVector = {1, 1, 1, 1};

  // 将第一个进程的存储对象转换为 GlooStore 类型，以测试特定的 GlooStore API
  auto rank_0_glooStore = static_cast<c10d::ProcessGroupGloo::GlooStore*>(
      tests[0].getProcessGroup()._getStore().get());

  // 将第二个进程的存储对象转换为 GlooStore 类型，以测试特定的 GlooStore API
  auto rank_1_glooStore = static_cast<c10d::ProcessGroupGloo::GlooStore*>(
      tests[1].getProcessGroup()._getStore().get());

  // 在第一个进程的 GlooStore 中设置一个无符号整数
  rank_0_glooStore->setUint("testKey", testVector);

  // 从第二个进程的 GlooStore 中获取之前设置的值
  auto value = rank_1_glooStore->getUint("testKey");

  // 断言获取的值与设置的值相等
  EXPECT_TRUE(value == testVector);
}

#ifndef _WIN32
// 测试 SIGSTOP 异常处理
TEST(ProcessGroupGlooTest, testSIGSTOPException) {
  // 测试 SIGSTOP 信号
  // Fork() 和 TSAN 不兼容，如果正在使用 TSAN 测试，则跳过该测试
  if (isTSANEnabled()) {
    LOG(INFO) << "Skipping test since Fork() + TSAN is broken";
    return;
  }

  // 创建临时文件对象
  TemporaryFile file;

  // 调用 testSignal 函数，测试 SIGSTOP 信号处理
  auto work = testSignal(file.path, SIGSTOP);

  // 断言测试结果为失败
  EXPECT_FALSE(work->isSuccess());

  // 断言抛出 std::exception 异常
  EXPECT_THROW(std::rethrow_exception(work->exception()), std::exception);
}

// 测试 SIGKILL 异常处理
TEST(ProcessGroupGlooTest, testSIGKILLException) {
  // 测试 SIGKILL 信号
  // Fork() 和 TSAN 不兼容，如果正在使用 TSAN 测试，则跳过该测试
  if (isTSANEnabled()) {
    LOG(INFO) << "Skipping test since Fork() + TSAN is broken";
    return;
  }

  // 创建临时文件对象
  TemporaryFile file;

  // 调用 testSignal 函数，测试 SIGKILL 信号处理
  auto work = testSignal(file.path, SIGKILL);

  // 断言测试结果为失败
  EXPECT_FALSE(work->isSuccess());

  // 断言抛出 std::exception 异常
  EXPECT_THROW(std::rethrow_exception(work->exception()), std::exception);
}
#endif

// 测试使用 CPU 进行全局归约操作
TEST(ProcessGroupGlooTest, testAllReduceCPU) {
  {
    // 创建临时文件对象
    TemporaryFile file;

    // 调用 testAllreduce 函数，使用 CPU 进行全局归约操作
    testAllreduce(file.path, at::DeviceType::CPU);

    // 调用 testAllreduceUsingWorkAPI 函数，使用 CPU 进行全局归约操作
    testAllreduceUsingWorkAPI(file.path, at::DeviceType::CPU);
  }
}

// 测试使用 CPU 进行 BFloat16 类型的全局归约操作
TEST(ProcessGroupGlooTest, testAllReduceBfloatCPU) {
  {
    // 创建临时文件对象
    TemporaryFile file;

    // 调用 testAllreduce 函数，使用 CPU 进行 BFloat16 类型的全局归约操作
    testAllreduce(file.path, at::DeviceType::CPU, at::kBFloat16);

    // 调用 testAllreduceUsingWorkAPI 函数，使用 CPU 进行全局归约操作
    testAllreduceUsingWorkAPI(file.path, at::DeviceType::CPU);
  }
}

// 测试使用 CPU 进行广播操作
TEST(ProcessGroupGlooTest, testBroadcastCPU) {
  {
    // 创建临时文件对象
    TemporaryFile file;

    // 调用 testBroadcast 函数，使用 CPU 进行广播操作
    testBroadcast(file.path, at::DeviceType::CPU);
  }
}

// 测试使用 CPU 进行 BFloat16 类型的广播操作
TEST(ProcessGroupGlooTest, testBroadcastBfloatCPU) {
  {
    // 创建临时文件对象
    TemporaryFile file;

    // 调用 testBroadcast 函数，使用 CPU 进行 BFloat16 类型的广播操作
    testBroadcast(file.path, at::DeviceType::CPU, at::kBFloat16);
  }
}

// 测试使用 CPU 进行全对全通信操作
TEST(ProcessGroupGlooTest, testAllToAllCPU) {
  {
    // 创建临时文件对象
    TemporaryFile file;

    // 调用 testAlltoall 函数，使用 CPU 进行全对全通信操作
    testAlltoall(file.path, at::DeviceType::CPU);
  }
}

// 测试屏障操作
TEST(ProcessGroupGlooTest, testBarrier) {
  {
    // 创建临时文件对象
    TemporaryFile file;

    // 调用 testBarrier 函数，测试屏障操作
    testBarrier(file.path);
  }
}

// 测试监控屏障操作
TEST(ProcessGroupGlooTest, testMonitoredBarrier) {
  // 创建临时文件对象
  TemporaryFile file;

  // 调用 testMonitoredBarrier 函数，测试监控屏障操作
  testMonitoredBarrier(file.path);
}

// 测试序列号初始化操作
TEST(ProcessGroupGlooTest, testSequenceNumInit) {
  // 创建临时文件对象
  TemporaryFile file;

  // 调用 testSequenceNumInit 函数，测试序列号初始化操作
  testSequenceNumInit(file.path);
}

// 测试发送操作
TEST(ProcessGroupGlooTest, testSend) {
  {
    // 创建临时文件对象
    TemporaryFile file;

    // 调用 testSend 函数，测试发送操作
    testSend(file.path);
  }
}

// 测试接收操作
TEST(ProcessGroupGlooTest, testRecv) {
  {
    // 创建临时文件对象
    TemporaryFile file;

    // 调用 testRecv 函数，测试接收操作
    testRecv(file.path);
  }
}
}

TEST(ProcessGroupGlooTest, testStoreSetGet) {
  // 创建临时文件对象
  TemporaryFile file;
  // 调用 testStoreSetGet 函数，传入临时文件路径进行测试
  testStoreSetGet(file.path);
}

TEST(ProcessGroupGlooTest, testWaitDelay) {
  {
    // 创建临时文件对象
    TemporaryFile file;
    // 调用 testWaitDelay 函数，传入临时文件路径进行测试
    testWaitDelay(file.path);
  }
}

#ifdef USE_CUDA
// 仅 CUDA 测试
TEST(ProcessGroupGlooTest, testAllReduceCUDA) {
  // 检查是否可用 CUDA
  if (!torch::cuda::is_available()) {
    // 输出日志信息并跳过测试 - 需要 CUDA
    LOG(INFO) << "Skipping test - requires CUDA";
    return;
  }
  {
    // 创建临时文件对象
    TemporaryFile file;
    // 调用 testAllreduce 函数，传入临时文件路径和 CUDA 设备类型进行测试
    testAllreduce(file.path, at::DeviceType::CUDA);
    // 调用 testAllreduceUsingWorkAPI 函数，传入临时文件路径和 CUDA 设备类型进行测试
    testAllreduceUsingWorkAPI(file.path, at::DeviceType::CUDA);
  }
}

TEST(ProcessGroupGlooTest, testBroadcastCUDA) {
  // 检查 CUDA 设备数量是否不少于 2
  if (torch::cuda::device_count() <= 1) {
    // 输出日志信息并跳过测试 - 需要多个 CUDA 设备
    LOG(INFO) << "Skipping test - requires multiple CUDA devices";
    return;
  }
  {
    // 创建临时文件对象
    TemporaryFile file;
    // 调用 testBroadcast 函数，传入临时文件路径和 CUDA 设备类型进行测试
    testBroadcast(file.path, at::DeviceType::CUDA);
  }
}

TEST(ProcessGroupGlooTest, testAlltoallCUDA) {
  // 检查是否可用 CUDA
  if (!torch::cuda::is_available()) {
    // 输出日志信息并跳过测试 - 需要 CUDA
    LOG(INFO) << "Skipping test - requires CUDA";
    return;
  }
  {
    // 创建临时文件对象
    TemporaryFile file;
    // 调用 testAlltoall 函数，传入临时文件路径和 CUDA 设备类型进行测试
    testAlltoall(file.path, at::DeviceType::CUDA);
  }
}

TEST(ProcessGroupGlooTest, testBackendName) {
  {
    // 创建临时文件对象
    TemporaryFile file;
    // 初始化 CollectiveTest 对象，传入临时文件路径和固定大小
    const auto size = 2;
    auto tests = CollectiveTest::initialize(file.path, size);

    // 遍历测试对象列表，验证每个对象的后端名称是否为 GLOO_BACKEND_NAME
    for (const auto i : c10::irange(size)) {
      EXPECT_EQ(
          tests[i].getProcessGroup().getBackendName(),
          std::string(c10d::GLOO_BACKEND_NAME));
    }
  }
}

#endif
```