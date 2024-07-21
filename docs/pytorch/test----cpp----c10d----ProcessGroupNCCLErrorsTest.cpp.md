# `.\pytorch\test\cpp\c10d\ProcessGroupNCCLErrorsTest.cpp`

```py
#include <chrono>  // 包含用于时间和时间点的标准库头文件
#include <filesystem>  // 包含用于文件系统操作的标准库头文件
#include <fstream>  // 包含用于文件输入输出的标准库头文件
#include <thread>  // 包含用于多线程操作的标准库头文件

#include <c10/util/irange.h>  // 包含 Caffe2 和 PyTorch 的 C10 库中的工具和范围处理
#include <torch/csrc/cuda/nccl.h>  // 包含用于 CUDA 并行计算的 NCCL 库头文件
#include <torch/csrc/distributed/c10d/FileStore.hpp>  // 包含用于分布式存储的文件存储类定义
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>  // 包含用于 NCCL 库的实用工具函数
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>  // 包含用于基于 NCCL 的进程组通信类定义
#include "CUDATest.hpp"  // 包含 CUDA 测试的自定义头文件
#include "TestUtils.hpp"  // 包含测试工具的自定义头文件

#include <gtest/gtest.h>  // 包含 Google 测试框架的头文件

using namespace c10d::test;  // 使用 c10d::test 命名空间

constexpr int kNcclErrorHandlingVersion = 2400;  // 定义 NCCL 错误处理版本号常量

// 定义一个模拟错误的 WorkNCCL 类，继承自 ProcessGroupNCCL::WorkNCCL
class WorkNCCLSimulateErrors : public c10d::ProcessGroupNCCL::WorkNCCL {
 public:
  WorkNCCLSimulateErrors(
      at::Device& device,
      bool simulate_error,
      int rank,
      c10d::OpType opType,
      uint64_t seq)
      : WorkNCCL(device, rank, opType, seq), simulateError_(simulate_error) {}

  // 检查是否有 NCCL 错误的方法重载
  std::exception_ptr checkForNCCLErrors() override {
    if (simulateError_) {
      // 如果需要模拟错误，则返回一个运行时异常指针
      return std::make_exception_ptr(std::runtime_error("Error"));
    }
    // 否则调用基类的方法检查 NCCL 错误
    return c10d::ProcessGroupNCCL::WorkNCCL::checkForNCCLErrors();
  }

 private:
  bool simulateError_;  // 是否模拟错误的标志
};

// 定义一个模拟错误的 ProcessGroupNCCL 类，继承自 c10d::ProcessGroupNCCL
class ProcessGroupNCCLSimulateErrors : public c10d::ProcessGroupNCCL {
 public:
  ProcessGroupNCCLSimulateErrors(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts)
      : ProcessGroupNCCL(store, rank, size, opts), simulateError_(false) {}

  // 检查是否有 NCCL 错误的方法重载，带有 NCCL 通信对象参数
  std::exception_ptr checkForNCCLErrors(
      std::shared_ptr<c10d::NCCLComm>& ncclComm) override {
    if (simulateError_) {
      // 如果需要模拟错误，则返回一个运行时异常指针
      return std::make_exception_ptr(std::runtime_error("Error"));
    }
    // 否则调用基类的方法检查 NCCL 错误
    return c10d::ProcessGroupNCCL::checkForNCCLErrors(ncclComm);
  }

  // 获取监视线程休眠间隔时间的方法
  std::chrono::duration<int64_t, std::milli> getWatchdogSleepInterval() {
    return std::chrono::milliseconds(
        ProcessGroupNCCLSimulateErrors::kWatchdogThreadSleepMillis);
  }

  // 初始化 WorkNCCL 对象的方法，创建并返回一个具有模拟错误功能的实例
  c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> initWork(
      at::Device& device,
      int rank,
      c10d::OpType opType,
      const char* profilingTitle,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false) override {
    return c10::make_intrusive<WorkNCCLSimulateErrors>(
        device, simulateError_, rank, opType, seqCollective_);
  }

  // 获取 NCCL 通信缓存大小的方法
  size_t getNCCLCommCacheSize() {
    return devNCCLCommMap_.size();
  }

  // 设置模拟错误标志的方法
  void simulateError() {
    simulateError_ = true;
  }

  // 重置模拟错误标志的方法
  void resetError() {
    simulateError_ = false;
  }

 private:
  bool simulateError_;  // 是否模拟错误的标志
};

// 定义一个模拟超时错误的 WorkNCCL 类，继承自 ProcessGroupNCCL::WorkNCCL
class WorkNCCLTimedoutErrors : public c10d::ProcessGroupNCCL::WorkNCCL {
 public:
  WorkNCCLTimedoutErrors(
      at::Device& device,
      bool set_timedout_error,
      int rank,
      c10d::OpType opType,
      uint64_t seq)
      : WorkNCCL(device, rank, opType, seq),
        setTimedoutError_(set_timedout_error) {}

 private:
  // 检查是否已完成工作的方法重载
  bool isCompleted() override {
    if (setTimedoutError_) {
      // 如果设置了超时错误，则返回未完成状态
      return false;
    }
    // 否则调用基类的方法检查工作是否完成
    return c10d::ProcessGroupNCCL::WorkNCCL::isCompleted();
  }

 private:
  bool setTimedoutError_;  // 是否设置超时错误的标志
};
class ProcessGroupNCCLTimedOutErrors : public ProcessGroupNCCLSimulateErrors {
 public:
  ProcessGroupNCCLTimedOutErrors(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts)
      : ProcessGroupNCCLSimulateErrors(store, rank, size, opts),
        watchDogDebugInfoFinished_(false),
        setTimedoutError_(false) {}
        // 构造函数：初始化基类，设置成员变量

  c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> initWork(
      at::Device& device,
      int rank,
      c10d::OpType opType,
      const char* profilingTitle,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false) override {
    return c10::make_intrusive<WorkNCCLTimedoutErrors>(
        device, setTimedoutError_, rank, opType, seqCollective_);
        // 初始化工作：创建一个特定类型的工作对象并返回
  }

  void setTimedoutError() {
    setTimedoutError_ = true;
    // 设置超时错误标志
  }

  void resetTimedoutError() {
    setTimedoutError_ = false;
    // 重置超时错误标志
  }

  bool getWatchDogDebugInfoFinishedFlag() {
    return watchDogDebugInfoFinished_;
    // 返回监视狗调试信息完成标志
  }

  // In the constructor of ProcessGroupNCCL. We don't allow the watchdog thread
  // to run any handling or desync report when the main thread is block wait.
  // Even if users set handling and turn on desyncDebug flag, they will get
  // reset. For the ease of unit test, we want the main thread to be block wait,
  // so we have this hack to manually set the desync debug flag after PG
  // creation.
  void forceSetDesyncDebugFlag() {
    desyncDebug_ = true;
    // 强制设置异步调试标志，以便单元测试时主线程被阻塞等待
  }

 protected:
  std::string getNCCLWatchdogDebugInfo() override {
    LOG(INFO) << "overridden getNCCLWatchdogDebugInfo called";
    watchDogDebugInfoFinished_ = true;
    return "";
    // 重写获取 NCCL 监视狗调试信息函数，设置调试信息完成标志并返回空字符串
  }
  bool watchDogDebugInfoFinished_;

 private:
  bool setTimedoutError_;
};

class ProcessGroupNCCLNoHeartbeatCaught
    : public ProcessGroupNCCLTimedOutErrors {
 public:
  ProcessGroupNCCLNoHeartbeatCaught(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts)
      : ProcessGroupNCCLTimedOutErrors(store, rank, size, opts),
        hasMonitorThreadCaughtError_(false) {}
        // 构造函数：调用基类构造函数，初始化成员变量

  std::mutex& getWatchdogMutex() {
    return workMetaListMutex_;
    // 返回监视狗互斥量的引用
  }

  bool getErrorCaughtFlag() {
    return hasMonitorThreadCaughtError_;
    // 返回错误捕获标志
  }

  void forceTryWriteDebugInfo() {
    std::future<bool> asyncDebugDump = std::async(
        std::launch::async, [this]() { return this->dumpDebuggingInfo(); });
    asyncDebugDump.wait();
    // 强制尝试写调试信息：异步调用 dumpDebuggingInfo 函数，并等待完成
  }

 protected:
  // Override the heartbeat monitor function to make sure that we capture
  // the exception in the monitor thread because we cannot try-catch it in
  // the main thread and we set a flag for the main thread to check.
  void heartbeatMonitor() override {
    try {
      c10d::ProcessGroupNCCL::heartbeatMonitor();
      // 重写心跳监视函数：捕获监视线程中的异常，并设置主线程检查标志
    } catch (std::runtime_error& e) {
      hasMonitorThreadCaughtError_ = true;
      // 捕获运行时错误，并设置捕获错误标志


这里是给定代码的注释，每个函数和重写函数都有详细的解释，说明了它们的作用和功能。
  }
}

// 重写了 terminateProcess 方法，用于终止进程并抛出运行时异常
// 原因是 std::abort 很难进行单元测试，所以选择重写而非直接调用
void terminateProcess(std::string errMsg) override {
  throw std::runtime_error(errMsg);
}

// 标记监控线程是否捕获到错误的布尔变量
bool hasMonitorThreadCaughtError_;
};

// ProcessGroupNCCLDebugInfoStuck 类继承自 ProcessGroupNCCLNoHeartbeatCaught 类
class ProcessGroupNCCLDebugInfoStuck
    : public ProcessGroupNCCLNoHeartbeatCaught {
 public:
  // 构造函数，初始化 ProcessGroupNCCLNoHeartbeatCaught 基类
  ProcessGroupNCCLDebugInfoStuck(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts)
      : ProcessGroupNCCLNoHeartbeatCaught(store, rank, size, opts) {}

 protected:
  // 重写心跳监控函数，设置长时间超时以模拟获取调试信息时的阻塞
  std::string getNCCLWatchdogDebugInfo() override {
    // 线程休眠，模拟长时间超时
    std::this_thread::sleep_for(
        std::chrono::seconds(heartbeatTimeoutInSec_ * 20));
    watchDogDebugInfoFinished_ = true;
    return "";
  }
};

// ProcessGroupNCCLErrorsTest 类继承自 ::testing::Test
class ProcessGroupNCCLErrorsTest : public ::testing::Test {
 protected:
  // 检查是否跳过测试
  bool skipTest() {
    // 如果没有可用的 CUDA 设备，跳过测试
    if (cudaNumDevices() == 0) {
      LOG(INFO) << "Skipping test since CUDA is not available";
      return true;
    }
    // 如果使用的 NCCL 版本过旧，跳过测试
#ifdef USE_C10D_NCCL
    if (torch::cuda::nccl::version() < kNcclErrorHandlingVersion) {
      LOG(INFO) << "Skipping test since NCCL version is too old";
      return true;
    }
#endif
    return false;
  }

  void SetUp() override {
    // 初始化日志记录
    c10::initLogging();
    // 在 SetUp 中检查是否需要跳过测试，确保只在有 GPU 的情况下运行测试及初始化
    if (skipTest()) {
      GTEST_SKIP() << "Skipping ProcessGroupNCCLErrorsTest because system "
                   << "requirement is not met (no CUDA or GPU).";
    }

    size_t numDevices = 1; // 每个进程（线程）一个设备
    TemporaryFile file;
    // 创建文件存储对象
    store_ = c10::make_intrusive<::c10d::FileStore>(file.path, 1);

    // 初始化张量向量，此处仅包含一个 CUDA 张量
    tensors_.resize(numDevices);
    tensors_[0] = at::empty({3, 3}, at::kCUDA);
  }

  void TearDown() override {
    // 设置环境变量以禁用 NCCL 阻塞等待
    ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "0", 1) == 0);
  }

  // 用于存储张量的向量及文件存储对象
  std::vector<at::Tensor> tensors_;
  c10::intrusive_ptr<::c10d::FileStore> store_;
};

// 测试用例，继承自 ProcessGroupNCCLErrorsTest
TEST_F(ProcessGroupNCCLErrorsTest, testNCCLErrorsBlocking) {
  // 设置环境变量以启用 NCCL 阻塞等待
  ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  // 创建 NCCL 过程组对象，模拟错误
  auto options = c10d::ProcessGroupNCCL::Options::create();
  options->timeout = std::chrono::milliseconds(1000);
  ProcessGroupNCCLSimulateErrors pg(store_, 0, 1, options);

  // 执行 allreduce 操作，并等待完成
  auto work = pg.allreduce(tensors_);
  work->wait();
  // 断言 NCCL 通信缓存的大小为 1
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());

  // 再次运行 allreduce，模拟错误情况
  pg.simulateError();
  work = pg.allreduce(tensors_);
  // 预期抛出运行时错误异常
  EXPECT_THROW(work->wait(), std::runtime_error);

  // 验证工作项失败
  EXPECT_TRUE(work->isCompleted());
  // 再次预期抛出运行时错误异常
  EXPECT_THROW(work->wait(), std::runtime_error);

  // 在此处可能会中止通信器，进一步操作将失败
}
// 使用 Google 测试框架的宏定义一个测试案例，用于测试 NCCL 错误处理中的超时情况
TEST_F(ProcessGroupNCCLErrorsTest, testNCCLTimedoutErrorsBlocking) {
  // 设置环境变量以启用阻塞等待模式
  ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  // 创建 ProcessGroupNCCL 的选项对象，并设置超时时间为 3000 毫秒
  auto options = c10d::ProcessGroupNCCL::Options::create();
  options->timeout = std::chrono::milliseconds(3000);
  // 创建 ProcessGroupNCCLTimedOutErrors 实例，使用给定的选项
  ProcessGroupNCCLTimedOutErrors pg(store_, 0, 1, options);

  // 执行 allreduce 操作并等待其完成
  auto work = pg.allreduce(tensors_);
  work->wait();
  // 断言 NCCL 通信缓存的大小为 1
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());

  // 设置超时错误标志，并再次执行 allreduce 操作，预期会抛出 c10::DistBackendError 异常
  pg.setTimedoutError();
  work = pg.allreduce(tensors_);
  EXPECT_THROW(work->wait(), c10::DistBackendError);

  // 在此处可能会中止通信器，进一步的操作可能会失败
}

// 使用 Google 测试框架的宏定义另一个测试案例，用于测试 NCCL 错误处理中的非阻塞情况
TEST_F(ProcessGroupNCCLErrorsTest, testNCCLErrorsNonBlocking) {
  // 创建 ProcessGroupNCCL 的选项对象，并设置超时时间为 3000 毫秒
  auto options = c10d::ProcessGroupNCCL::Options::create();
  options->timeout = std::chrono::milliseconds(3000);
  // 创建 ProcessGroupNCCLSimulateErrors 实例，使用给定的选项
  ProcessGroupNCCLSimulateErrors pg(store_, 0, 1, options);

  // 执行 allreduce 操作并等待其完成
  auto work = pg.allreduce(tensors_);
  pg.barrier()->wait();
  // 断言 NCCL 通信缓存的大小为 1
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());

  // 模拟发生错误，并再次执行 allreduce 操作，预期不会抛出异常
  pg.simulateError();
  work = pg.allreduce(tensors_);

  // 等待操作完成
  work->wait();
  pg.barrier()->wait();

  // 断言工作已完成
  EXPECT_TRUE(work->isCompleted());
  // 在此处可能会中止通信器，进一步的操作可能会失败
}

// 用于从本地磁盘读取验证用数据的函数
std::string readTraceFromFile(const std::string& filename, size_t size) {
  std::ifstream file(filename, std::ios::binary);
  // 从文件中读取字符串数据
  if (file) { // 只要文件流处于良好状态
    std::string str(size, '\0');
    file.read(&str[0], size);
    if (file) {
      return str;
    }
  }
  return "";
}

// 将嵌套类扩展到父类之外的测试调试信息写入器
class TestDebugInfoWriter : public c10d::DebugInfoWriter {
 public:
  // 构造函数，初始化基类 DebugInfoWriter，并设定名称前缀
  TestDebugInfoWriter(std::string namePrefix)
      : DebugInfoWriter(namePrefix, 0) {}

  // 覆盖基类的写入方法，将 ncclTrace 写入 traces_ 成员变量
  void write(const std::string& ncclTrace) override {
    traces_.assign(ncclTrace.begin(), ncclTrace.end());
    c10d::DebugInfoWriter::write(ncclTrace);
  }

  // 返回 traces_ 成员变量的引用
  std::vector<uint8_t>& getTraces() {
    return traces_;
  }

 private:
  std::vector<uint8_t> traces_; // 存储调试信息的字节流
};
// 定义一个测试用例，测试在没有心跳时的NCCL错误情况
TEST_F(ProcessGroupNCCLErrorsTest, testNCCLErrorsNoHeartbeat) {
  // 设置心跳间隔为2秒
  int heartBeatIntervalInSec = 2;
  // 将心跳间隔转换为字符串
  std::string timeInterval = std::to_string(heartBeatIntervalInSec);
  // 设置环境变量TORCH_NCCL_BLOCKING_WAIT[0]为"1"
  ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  // 设置环境变量TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC[0]为timeInterval对应的值
  ASSERT_TRUE(
      setenv(
          c10d::TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC[0].c_str(),
          timeInterval.c_str(),
          1) == 0);
  // 设置环境变量TORCH_NCCL_ENABLE_MONITORING[0]为"1"
  ASSERT_TRUE(
      setenv(c10d::TORCH_NCCL_ENABLE_MONITORING[0].c_str(), "1", 1) == 0);
  // 设置环境变量TORCH_NCCL_DEBUG_INFO_TEMP_FILE为tempFilename对应的值
  auto tempFilename = c10::str(
      std::filesystem::temp_directory_path().string(), "/nccl_trace_rank_");
  ASSERT_TRUE(
      setenv("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", tempFilename.c_str(), 1) == 0);
  // 启用nccl飞行记录器
  ASSERT_TRUE(setenv("TORCH_NCCL_TRACE_BUFFER_SIZE", "10", 1) == 0);
  // 设置环境变量TORCH_NCCL_DUMP_ON_TIMEOUT[0]为"1"
  ASSERT_TRUE(setenv(c10d::TORCH_NCCL_DUMP_ON_TIMEOUT[0].c_str(), "1", 1) == 0);
  // 创建NCCL进程组的选项对象，并设置超时时间为30秒
  auto options = c10d::ProcessGroupNCCL::Options::create();
  options->timeout = std::chrono::milliseconds(30000);
  // 创建一个处理无心跳异常的NCCL进程组对象
  ProcessGroupNCCLNoHeartbeatCaught pg(store_, 0, 1, options);
  // 创建文件名前缀，用于获取nccl调试信息文件的路径
  std::string fileNamePrefix = c10d::getCvarString(
      {"TORCH_NCCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/nccl_trace_rank_");
  // 创建一个用于测试的调试信息写入器对象
  std::unique_ptr<TestDebugInfoWriter> wrterForTestPtr =
      std::make_unique<TestDebugInfoWriter>(fileNamePrefix);
  // 获取调试信息写入器对象的跟踪数据
  std::vector<uint8_t>& traces = wrterForTestPtr->getTraces();
  // 注册调试信息写入器对象
  c10d::DebugInfoWriter::registerWriter(std::move(wrterForTestPtr));

  // 执行正常的集体通信操作
  auto work = pg.allreduce(tensors_);
  work->wait();

  // 再次执行集体通信操作
  work = pg.allreduce(tensors_);
  {
    // 开始运行带错误的全reduce操作
    // 获取监视狗互斥锁
    std::lock_guard<std::mutex> lock(pg.getWatchdogMutex());
    LOG(INFO) << "Lock watchdog thread."; // 记录日志信息，锁定监视狗线程
    // 等待足够长的时间，以便监视线程抛出异常
    std::this_thread::sleep_for(
        std::chrono::seconds(heartBeatIntervalInSec * 3));
    // 检查监视线程是否已启动并抛出异常
    EXPECT_TRUE(pg.getErrorCaughtFlag());
  }
  // 等待操作完成
  work->wait();
  // 断言跟踪数据的长度大于0
  EXPECT_TRUE(traces.size() > 0);
  // 构造完整的文件名
  auto filename = c10::str(tempFilename, 0);
  // 从文件中读取跟踪数据
  auto traceFromStorage = readTraceFromFile(filename, traces.size());
  // 检查从存储中读取的跟踪数据是否与原始nccl跟踪数据匹配
  EXPECT_TRUE(traceFromStorage == std::string(traces.begin(), traces.end()));
  // 删除文件
  std::filesystem::remove(filename);
}

// 定义一个测试类，用于测试NCCL监视狗超时情况
class ProcessGroupNCCLWatchdogTimeoutTest : public ProcessGroupNCCLErrorsTest {
 protected:
  void SetUp() override {
    // TODO (kwen2501)
    // 跳过ProcessGroupNCCLWatchdogTimeoutTest下的测试，待重构工作队列后重新编写测试
    GTEST_SKIP() << "Skipping tests under ProcessGroupNCCLWatchdogTimeoutTest; "
                 << "will rewrite them after refactoring Work queues.";
    // 调用父类的SetUp方法
    ProcessGroupNCCLErrorsTest::SetUp();
    // 将心跳间隔转换为字符串形式
    std::string timeInterval = std::to_string(heartBeatIntervalInSec);
    // 设置环境变量 TORCH_NCCL_BLOCKING_WAIT，值为 "1"
    ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
    // 设置环境变量 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC，值为 timeInterval 的 C 字符串表示
    ASSERT_TRUE(
        setenv(
            c10d::TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC[0].c_str(),
            timeInterval.c_str(),
            1) == 0);
    // 设置环境变量 TORCH_NCCL_ENABLE_MONITORING，值为 "1"
    ASSERT_TRUE(
        setenv(c10d::TORCH_NCCL_ENABLE_MONITORING[0].c_str(), "1", 1) == 0);
    // 设置环境变量 TORCH_NCCL_DESYNC_DEBUG，值为 "1"
    ASSERT_TRUE(setenv(c10d::TORCH_NCCL_DESYNC_DEBUG[0].c_str(), "1", 1) == 0);
    // 禁用异步错误处理，设置环境变量 TORCH_NCCL_ASYNC_ERROR_HANDLING，值为 "0"
    ASSERT_TRUE(
        setenv(c10d::TORCH_NCCL_ASYNC_ERROR_HANDLING[0].c_str(), "0", 1) == 0);
    // 创建 ProcessGroupNCCL 的选项对象
    options_ = c10d::ProcessGroupNCCL::Options::create();
    // 设置选项对象的超时时间为 100 毫秒
    options_->timeout = std::chrono::milliseconds(100);
  }

  // Watchdog 超时测试的公共方法
  void watchdogTimeoutTestCommon(
      ProcessGroupNCCLNoHeartbeatCaught& pg,
      int multiplier) {
    // 强制设置异步调试标志
    pg.forceSetDesyncDebugFlag();
    // 设置超时错误
    pg.setTimedoutError();
    // 执行 allreduce 操作
    auto work = pg.allreduce(tensors_);
    // 等待指定的倍数心跳间隔的时间
    std::this_thread::sleep_for(
        std::chrono::seconds(heartBeatIntervalInSec * multiplier));
    // 预期抛出 c10::DistBackendError 异常
    EXPECT_THROW(work->wait(), c10::DistBackendError);
  }

  // 心跳间隔时间，单位秒
  const int heartBeatIntervalInSec = 2;
  // ProcessGroupNCCL 的选项对象指针
  c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> options_;
};

// 定义测试用例 `ProcessGroupNCCLWatchdogTimeoutTest` 中的测试函数 `testNCCLTimedoutDebugInfoFinished`
TEST_F(ProcessGroupNCCLWatchdogTimeoutTest, testNCCLTimedoutDebugInfoFinished) {
  // 创建一个 `ProcessGroupNCCLNoHeartbeatCaught` 对象，使用存储、0、1 和选项初始化
  ProcessGroupNCCLNoHeartbeatCaught pg(store_, 0, 1, options_);
  // 执行调用 `forceTryWriteDebugInfo` 方法，生成调试信息，会导致看门狗线程等待 30 秒
  // 由于难以覆盖此行为，因此在之前直接调用它。否则，需要设置较长的心跳超时时间，这会使测试变慢。
  pg.forceTryWriteDebugInfo();
  // 执行 `watchdogTimeoutTestCommon` 函数，传入 `pg` 和 2 作为参数，执行看门狗超时测试的共同操作
  watchdogTimeoutTestCommon(pg, 2);

  // 断言：若 `getWatchDogDebugInfoFinishedFlag` 返回 true，则说明心跳监视线程在获取调试信息（如解同步调试信息）时没有杀死看门狗线程
  EXPECT_TRUE(pg.getWatchDogDebugInfoFinishedFlag());
  // 断言：若 `getErrorCaughtFlag` 返回 false，则说明心跳监视线程在获取调试信息时未触发进程中止，且销毁进程组速度较快
  EXPECT_FALSE(pg.getErrorCaughtFlag());

  // 在此处可能会中止通信器，进一步操作可能会失败。
}

// 定义测试用例 `ProcessGroupNCCLWatchdogTimeoutTest` 中的测试函数 `testNCCLTimedoutDebugInfoStuck`
TEST_F(ProcessGroupNCCLWatchdogTimeoutTest, testNCCLTimedoutDebugInfoStuck) {
  // 创建一个 `ProcessGroupNCCLDebugInfoStuck` 对象，使用存储、0、1 和选项初始化
  ProcessGroupNCCLDebugInfoStuck pg(store_, 0, 1, options_);
  // 需要让主线程休眠更长时间，以便让心跳监视线程完成额外的等待并翻转标志。
  watchdogTimeoutTestCommon(pg, 4);
  // 断言：若 `getWatchDogDebugInfoFinishedFlag` 返回 false，则说明看门狗线程在获取调试信息（如解同步调试信息）时被卡住了
  EXPECT_FALSE(pg.getWatchDogDebugInfoFinishedFlag());
  // 断言：若 `getErrorCaughtFlag` 返回 true，则说明心跳监视线程在获取调试信息时触发了进程中止
  EXPECT_TRUE(pg.getErrorCaughtFlag());

  // 在此处可能会中止通信器，进一步操作可能会失败。
}
```