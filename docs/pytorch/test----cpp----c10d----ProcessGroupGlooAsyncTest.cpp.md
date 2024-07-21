# `.\pytorch\test\cpp\c10d\ProcessGroupGlooAsyncTest.cpp`

```
// 包含 CUDA 相关的头文件
#include <c10/cuda/CUDAGuard.h>
// 包含用于循环的工具函数
#include <c10/util/irange.h>

// 包含与 CUDA 相关的 ATen 和 GTest 头文件
#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>

// 包含与分布式通信相关的头文件
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

// 包含自定义的测试辅助类的头文件
#include "CUDATest.hpp"
#include "TestUtils.hpp"

// 使用 c10d::test 命名空间
using namespace c10d::test;

// 使用 ATen 中 CUDAStream 类
using at::cuda::CUDAStream;

// 初始化函数模板，接收路径、数量 N 和可变参数 Args
template <typename T, typename... Args>
std::vector<T> initialize(const std::string& path, int N, Args&&... args) {
  std::vector<T> tests;
  // 使用 irange(N) 循环 N 次
  for (C10_UNUSED const auto i : c10::irange(N)) {
    // 将 T 类型的对象通过移动语义推送到 tests 容器中
    tests.push_back(std::move(T(path, std::forward<Args>(args)...)));
  }

  std::vector<std::thread> threads;
  // 再次使用 irange(N) 循环 N 次
  for (C10_UNUSED const auto i : c10::irange(N)) {
    // 创建 N 个线程，每个线程启动相应索引的 tests 中的对象的 start 方法
    threads.push_back(std::thread([i, N, &tests] { tests[i].start(i, N); }));
  }

  // 等待所有线程执行完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 返回初始化后的 tests 容器
  return tests;
}

// AsyncTest 类定义
class AsyncTest {
 public:
  // 构造函数，接收一个路径参数
  AsyncTest(std::string path) : path_(std::move(path)) {}

  // 移动构造函数，用于移动构造 AsyncTest 对象
  AsyncTest(AsyncTest&& other) {
    path_ = std::move(other.path_);
    pg_ = std::move(other.pg_);
  }

  // 获取 ProcessGroupGloo 对象的方法
  ::c10d::ProcessGroupGloo& getProcessGroup() {
    return *pg_;
  }

  // 启动方法，接收 rank 和 size 参数
  void start(int rank, int size) {
    // 创建 FileStore 对象，用于存储数据
    auto store = c10::make_intrusive<::c10d::FileStore>(path_, size);

    // 创建 ProcessGroupGloo 的选项对象，设置超时时间为 50 毫秒
    auto options = ::c10d::ProcessGroupGloo::Options::create();
    options->timeout = std::chrono::milliseconds(50);
    // 将本地主机作为设备添加到选项的设备列表中
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));

    // 创建 ProcessGroupGloo 对象，并存储到成员变量 pg_ 中
    pg_ = std::unique_ptr<::c10d::ProcessGroupGloo>(
        new ::c10d::ProcessGroupGloo(store, rank, size, options));
  }

 protected:
  std::string path_;  // 存储路径的成员变量
  std::unique_ptr<::c10d::ProcessGroupGloo> pg_;  // 存储 ProcessGroupGloo 对象的智能指针
};

// AsyncInputIsOutputTest 类继承自 AsyncTest 类
class AsyncInputIsOutputTest : public AsyncTest {
 public:
  // 构造函数，接收路径和 numTensors 参数
  AsyncInputIsOutputTest(const std::string& path, int numTensors)
      : AsyncTest(path),
        numTensors_(numTensors),
        numDevices_(cudaNumDevices()) {
    // 在可用设备上分配输入张量，使用 round robin 策略
    ::at::globalContext().lazyInitCUDA();
    inputs_.resize(numTensors_);
    // 使用 irange(numTensors_) 循环
    for (const auto i : c10::irange(numTensors_)) {
      // 在指定设备上分配空张量
      inputs_[i] = at::empty(
          {16, 16},
          at::device(
              {at::kCUDA, static_cast<c10::DeviceIndex>(i % numDevices_)}));
    }

    // 每个设备分配一个流
    at::cuda::OptionalCUDAGuard deviceGuard;
    streams_.reserve(numDevices_);
    // 使用 irange(numDevices_) 循环
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(i);
      // 从 CUDA 流池中获取流并存储在 streams_ 容器中
      streams_.push_back(at::cuda::getStreamFromPool());
    }
  }

  // 等待异步工作完成的方法，接收一个 c10d::Work 对象的智能指针参数
  void wait(c10::intrusive_ptr<c10d::Work>& work) {
    // 使用 CUDA 多流守护对象，将流设置为多个设备的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);


这段代码还未完成，但上面已经添加了所有必要的注释。
    work->wait();
  }

  // 获取 CPU 张量
  std::vector<at::Tensor> getCpuTensors(
      const std::vector<at::Tensor>& gpu_tensors) {
    // 创建一个与 gpu_tensors 相同大小的输出张量向量
    std::vector<at::Tensor> outputs(gpu_tensors.size());

    // 在此函数的执行期间，使 THC 使用我们的流
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 将输入复制到输出
    for (unsigned i = 0; i < gpu_tensors.size(); i++) {
      // 将每个 GPU 张量复制到 CPU
      outputs[i] = gpu_tensors[i].cpu();
    }

    return outputs;
  }

  // 获取张量
  std::vector<at::Tensor> getTensors() {
    // 返回从输入张量转换得到的 CPU 张量
    return getCpuTensors(inputs_);
  }

 protected:
  const int numTensors_;  // 张量数量
  const int numDevices_;  // 设备数量
  std::vector<at::Tensor> inputs_;  // 输入张量向量
  std::vector<CUDAStream> streams_;  // CUDA 流向量
};

// 异步全局归约测试类，继承自异步输入输出测试类
class AsyncAllreduceTest : public AsyncInputIsOutputTest {
 public:
  // 构造函数，接受路径和张量数量作为参数
  AsyncAllreduceTest(const std::string& path, int numTensors)
      : AsyncInputIsOutputTest(path, numTensors) {}

  // 运行测试的方法，返回一个指向工作对象的智能指针
  c10::intrusive_ptr<c10d::Work> run() {
    // 在函数执行期间，使用我们的流来设置 THC
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 在每个流上启动睡眠操作
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 10 * 1000 * 1000);
    }

    // 为每个张量启动数值初始化
    for (const auto i : c10::irange(numTensors_)) {
      deviceGuard.set_index(i % numDevices_);
      inputs_[i].fill_(pg_->getRank() * numTensors_ + i);
    }

    // 执行全局归约操作并返回结果
    return pg_->allreduce(inputs_);
  }
};

// 异步广播测试类，继承自异步输入输出测试类
class AsyncBroadcastTest : public AsyncInputIsOutputTest {
 public:
  // 构造函数，接受路径、张量数量作为参数
  AsyncBroadcastTest(const std::string& path, int numTensors)
      : AsyncInputIsOutputTest(path, numTensors) {}

  // 运行测试的方法，接受根排名和根张量作为参数，返回一个指向工作对象的智能指针
  c10::intrusive_ptr<c10d::Work> run(int rootRank, int rootTensor) {
    // 在函数执行期间，使用我们的流来设置 THC
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // 在每个流上启动睡眠操作
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 10 * 1000 * 1000);
    }

    // 为每个张量启动数值初始化
    for (const auto i : c10::irange(numTensors_)) {
      deviceGuard.set_index(i % numDevices_);
      inputs_[i].fill_(pg_->getRank() * numTensors_ + i);
    }

    // 设置广播选项并执行广播操作，返回结果
    ::c10d::BroadcastOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->broadcast(inputs_, options);
  }
};

// 运行异步全局归约测试的函数，接受路径、进程数量和张量数量作为参数
void runAsyncAllreduceTest(
    const std::string& path,
    size_t numProcesses = 4,
    size_t numTensors = 2) {
  // 初始化异步全局归约测试对象
  auto tests = initialize<AsyncAllreduceTest>(path, numProcesses, numTensors);
  std::vector<c10::intrusive_ptr<c10d::Work>> work(numProcesses);

  // 执行每个测试，并将返回的工作对象存储在向量中
  for (const auto i : c10::irange(numProcesses)) {
    work[i] = tests[i].run();
  }

  // 等待所有工作完成
  for (const auto i : c10::irange(numProcesses)) {
    tests[i].wait(work[i]);
  }

  // 检查结果
  for (const auto i : c10::irange(numProcesses)) {
    const auto size = numProcesses * numTensors;
    const auto expected = (size * (size - 1)) / 2;
    auto tensors = tests[i].getTensors();
    auto results = tests[i].getCpuTensors(work[i]->result());
    EXPECT_EQ(tensors.size(), results.size());
    // 遍历 tensors 数组的索引范围，每次迭代用 j 表示当前索引
    for (const auto j : c10::irange(tensors.size())) {
        // 获取当前索引 j 处的 tensor 引用
        auto& tensor = tensors[j];
        // 获取 tensor 中数据的指针，假设数据类型为 float
        auto data = tensor.data_ptr<float>();

        // 获取 results 数组中与当前 tensor 对应的结果 tensor 的引用
        auto& result_tensor = results[j];
        // 获取结果 tensor 中数据的指针，假设数据类型为 float
        auto result_data = result_tensor.data_ptr<float>();

        // 断言 tensor 和结果 tensor 的元素个数相等
        EXPECT_EQ(tensor.numel(), result_tensor.numel());

        // 遍历当前 tensor 的所有元素的索引范围，每次迭代用 k 表示当前索引
        for (const auto k : c10::irange(tensor.numel())) {
            // 断言当前 tensor 的第 k 个元素的值等于预期值 expected
            EXPECT_EQ(data[k], expected);
            // 断言当前结果 tensor 的第 k 个元素的值等于预期值 expected
            EXPECT_EQ(result_data[k], expected);
        }
    }
}

// 异步广播测试的运行函数
void runAsyncBroadcastTest(
    const std::string& path,  // 测试数据文件路径
    size_t numProcesses = 4,  // 进程数，默认为4
    size_t numTensors = 1) {  // 张量数，默认为1
  // 初始化异步广播测试，返回一个测试对象数组
  auto tests = initialize<AsyncBroadcastTest>(path, numProcesses, numTensors);

  // 尝试每种根排名和根张量的排列组合
  for (const auto rootRank : c10::irange(numProcesses)) {
    for (const auto rootTensor : c10::irange(numTensors)) {
      // 创建一个工作对象数组，大小为进程数
      std::vector<c10::intrusive_ptr<c10d::Work>> work(numProcesses);
      for (const auto i : c10::irange(numProcesses)) {
        // 对每个测试对象运行异步广播操作，得到一个工作对象
        work[i] = tests[i].run(rootRank, rootTensor);
      }

      // 等待所有工作完成
      for (const auto i : c10::irange(numProcesses)) {
        tests[i].wait(work[i]);
      }

      // 检查结果
      const auto expected = (rootRank * numTensors + rootTensor);
      for (const auto i : c10::irange(numProcesses)) {
        // 获取测试对象中的张量数组
        auto tensors = tests[i].getTensors();
        for (const auto& tensor : tensors) {
          // 获取张量的浮点数据指针
          const auto* const data = tensor.const_data_ptr<float>();
          // 检查每个元素是否与预期值相等
          for (const auto k : c10::irange(tensor.numel())) {
            EXPECT_EQ(data[k], expected);
          }
        }
      }
    }
  }
}

#ifdef USE_CUDA
// CUDA 下的异步全局归约测试
TEST(ProcessGroupGlooAsyncTest, testAsyncAllreduce) {
  // 如果 CUDA 不可用，则跳过测试
  if (!at::cuda::is_available()) {
    LOG(INFO) << "CUDA not available, skipping testAsyncAllreduce";
    return;
  }
  // 创建临时文件
  TemporaryFile file;
  // 运行异步全局归约测试，使用临时文件的路径
  runAsyncAllreduceTest(file.path);
}

// CUDA 下的异步广播测试
TEST(ProcessGroupGlooAsyncTest, testAsyncBroadcast) {
  // 如果 CUDA 不可用，则跳过测试
  if (!at::cuda::is_available()) {
    LOG(INFO) << "CUDA not available, skipping testAsyncBroadcast";
    return;
  }
  // 创建临时文件
  TemporaryFile file;
  // 运行异步广播测试，使用临时文件的路径
  runAsyncBroadcastTest(file.path);
}
#endif


这些注释详细解释了每行代码的作用，包括函数的参数说明、循环的目的、数据操作以及测试的跳过条件。
```