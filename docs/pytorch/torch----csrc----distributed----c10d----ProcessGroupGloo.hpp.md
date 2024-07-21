# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupGloo.hpp`

```py
#pragma once

#ifdef USE_C10D_GLOO

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <gloo/algorithm.h>
#include <gloo/common/error.h>
#include <gloo/context.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/device.h>

#include <c10/util/hash.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

namespace c10d {

// 定义常量，指定后端名为"gloo"
constexpr const char* GLOO_BACKEND_NAME = "gloo";

// ProcessGroupGloo 实现了对 c10d 的 Gloo 绑定。
//
// 预期该类的所有函数在进程组中的所有进程中以相同顺序调用。这是确保跨进程匹配相同调用的唯一方法。
// 对于多线程使用进程组，可以考虑使用多个进程组实例。
//
// 该类调用的 Gloo 算法通过它们的签名进行缓存（参见上述 AlgorithmKey 的描述）。
// 缓存工作方式如下：每个函数调用都会实例化一个 AlgorithmKey，并查找缓存中的现有条目。
// 如果有条目，则将其从缓存中移除并返回给调用者。如果没有条目，则创建一个新条目并返回。
// 如果条目之前已创建，但仍在使用中，则调用将阻塞并等待直到条目返回到缓存中。
//
// 未来，我们希望扩展此功能，允许每个键有多个条目，以实现单个键的并行性。
// 每个键的条目数量必须对所有进程保持一致。这个最大数量可以自动调整，
// 但前提是让单个进程负责，并让其广播限制。
//
class TORCH_API ProcessGroupGloo : public Backend {
 public:
  // AsyncWork 是 Gloo 特定的异步工作项的超类。
  // 我们可以将异步工作分为 3 个阶段：
  // 1) 检查和准备输入（例如 memcpy）
  // 2) 在后台线程上运行操作
  // 3) 在前台线程中与完成同步
  //
  // 这三个阶段之间有状态需要共享，所有这些状态都包含在 AsyncWork 类及其派生类中。
  //
  // 注意：在将操作转换为使用新式集合操作时，我们将使用现有的缓存方法和使用新的 AsyncWork 基类之间存在分离。
  // 随着时间的推移，我们将转移所有操作并执行必要的清理工作。
  //
  // FIXME: 这可能应该称为 WorkGloo，因为工作由后台线程以同步模式执行。
  class TORCH_API AsyncWork : public Work {
   public:
    // AsyncWork 类的构造函数，接收输出张量、操作类型、序列号以及可选的性能分析标题和输入张量
    explicit AsyncWork(
        std::vector<std::vector<at::Tensor>> outputTensors,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            c10::nullopt);

    // AsyncWork 类的析构函数，默认实现
    ~AsyncWork() override = default;

    // 执行异步工作的静态方法，接收 AsyncWork 类的智能指针作为参数
    static void execute(const c10::intrusive_ptr<AsyncWork>& work);

    // 纯虚函数，派生类需要实现的运行方法
    virtual void run() = 0;

    // 返回异步工作的结果张量
    std::vector<at::Tensor> result() override;

    // 获取异步工作的 future 对象
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
    
    // 获取序列号的方法
    uint64_t getSequencenumber() const override;

   protected:
    // ProcessGroupGloo 类是 AsyncWork 的友元类
    friend class ProcessGroupGloo;

   private:
    // 私有方法，完成 Gloo 工作的清理工作
    void finishWorkGloo();

    // 私有方法，处理 Gloo 工作错误的清理工作，接收异常指针作为参数
    void finishWorkGlooError(const std::exception_ptr& eptr);

    // 内联方法，记录异步工作的性能分析信息，接收性能分析标题和输入张量作为参数
    inline void recordAsyncWorkProfilingInfo(
        const char* profilingTitle,
        const std::optional<std::vector<at::Tensor>>& inputTensors);

    // 输出张量的二维向量，保存 AsyncWork 对象的输出结果
    const std::vector<std::vector<at::Tensor>> outputTensors_;

    // 异步工作的 future 对象，保存工作的异步结果
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // 记录函数前回调的函数对象
    std::function<void()> recordFunctionBeforeCallback_;

    // 工作的序列号
    const uint64_t seq_;
  };

  // 将 c10d 存储包装为 Gloo 存储的类
  class TORCH_API GlooStore : public ::gloo::rendezvous::Store {
   public:
    // 构造函数，接收 c10d 存储的智能指针作为参数
    GlooStore(const c10::intrusive_ptr<::c10d::Store>& store) : store_(store) {}

    // 设置无符号整数值的方法，将键和值存储到底层的 c10d 存储中
    void setUint(const std::string& key, const std::vector<uint8_t>& value) {
      store_->set(key, value);
    }

    // 设置字符向量值的方法，将键和值存储到底层的 c10d 存储中
    void set(const std::string& key, const std::vector<char>& value) override {
      std::vector<uint8_t> tmp(value.begin(), value.end());
      store_->set(key, tmp);
    }

    // 获取无符号整数值的方法，从底层的 c10d 存储中获取指定键的值
    std::vector<uint8_t> getUint(const std::string& key) {
      auto value = store_->get(key);
      return value;
    }

    // 获取字符向量值的方法，从底层的 c10d 存储中获取指定键的值
    std::vector<char> get(const std::string& key) override {
      auto value = store_->get(key);
      return std::vector<char>(value.begin(), value.end());
    }

    // 等待指定键的方法，调用底层 c10d 存储的默认超时时间
    void wait(const std::vector<std::string>& keys) override {
      store_->wait(keys, ::c10d::Store::kDefaultTimeout);
    }

    // 等待指定键的方法，调用底层 c10d 存储并指定超时时间
    void wait(
        const std::vector<std::string>& keys,
        const std::chrono::milliseconds& timeout) override {
      store_->wait(keys, timeout);
    }
#ifdef GLOO_STORE_HAS_STORE_V2
    // 如果 GLOO_STORE_HAS_STORE_V2 宏被定义，则以下代码块被编译

    // 检查是否支持 v2 扩展 API，通过调用 store_ 对象的 hasExtendedApi 方法
    bool has_v2_support() override {
      return store_->hasExtendedApi();
    }

    // 批量获取操作，接受键的向量 keys，并返回对应值的向量
    std::vector<std::vector<char>> multi_get(
        const std::vector<std::string>& keys) override {
      std::vector<std::vector<char>> res;
      // 对每个键调用 store_ 的 multiGet 方法，并将结果转换为 char 向量后加入 res
      for (auto& value : store_->multiGet(keys)) {
        res.emplace_back(value.begin(), value.end());
      }
      return res;
    }

    // 批量设置操作，接受键的向量 keys 和对应值的二维向量 values
    void multi_set(
        const std::vector<std::string>& keys,
        const std::vector<std::vector<char>>& values) override {
      std::vector<std::vector<uint8_t>> u_values;
      u_values.reserve(values.size());
      // 将每个值转换为 uint8_t 向量后加入 u_values，然后调用 store_ 的 multiSet 方法
      for (auto& value : values) {
        u_values.emplace_back(value.begin(), value.end());
      }
      store_->multiSet(keys, u_values);
    }

    // 追加操作，接受键 key 和值的 char 向量 value
    void append(const std::string& key, const std::vector<char>& value)
        override {
      // 将值转换为 uint8_t 向量 tmp，然后调用 store_ 的 append 方法
      std::vector<uint8_t> tmp(value.begin(), value.end());
      return store_->append(key, tmp);
    }

    // 加法操作，接受键 key 和 int64_t 类型的值 value
    int64_t add(const std::string& key, int64_t value) override {
      // 调用 store_ 的 add 方法，返回加法操作后的结果
      return store_->add(key, value);
    }
#endif

   // 受保护的成员变量，表示一个 c10d::Store 的智能指针
   protected:
    c10::intrusive_ptr<::c10d::Store> store_;
  };

  // 对于发送和接收操作，不需要将它们传递给线程池，因为它们完全由设备线程完成。
  // 这个工作对象用于同步发送或接收操作的完成。它保持对其操作的张量的引用，
  // 以防止在操作仍在进行时被释放。
  class TORCH_API SendWork : public Work {
   public:
    // 构造函数，接受张量 tensor、非绑定缓冲区指针 buffer 和序列号 seq
    explicit SendWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
        uint64_t seq);

    // 等待操作完成，可选超时参数 timeout
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    // 中止操作
    void abort() override;

    // 返回序列号 seq
    uint64_t getSequencenumber() const override;

   protected:
    // 操作的张量
    at::Tensor tensor_;
    // 非绑定缓冲区指针
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
    // 操作的序列号
    const uint64_t seq_;
  };

  // 接收工作类，用于接收操作
  class TORCH_API RecvWork : public Work {
   public:
    // 构造函数，接受张量 tensor、非绑定缓冲区指针 buffer、操作类型 opType、
    // 序列号 seq 和用于性能分析的标题 profilingTitle
    explicit RecvWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr);

    // 返回发送方的排名
    int sourceRank() const override;

    // 等待操作完成，可选超时参数 timeout
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    // 中止操作
    void abort() override;

    // 返回序列号 seq
    uint64_t getSequencenumber() const override;

   protected:
    // 操作的张量
    at::Tensor tensor_;
    // 非绑定缓冲区指针
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
    // 发送方的排名
    int srcRank_;
    // 操作的序列号
    const uint64_t seq_;
  };

  // 选项结构体，继承自 Backend::Options
  struct TORCH_API Options : public Backend::Options {
    // 构造函数，接受超时参数 timeout
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    // 创建 Options 对象的静态方法，返回 intrusive_ptr 指针
    static c10::intrusive_ptr<Options> create(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }
    // 保存设备的共享指针的向量
    std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
    // 线程数
    int threads;
  };

  // 返回后端名称的常量字符串
  const std::string getBackendName() const override {
    return std::string(GLOO_BACKEND_NAME);
  }

  // 用于创建新设备对象的辅助函数。
  // 这些函数是此类的静态函数，以使它们在逻辑上与代码库的其余部分（如 torch/csrc/distributed）分开。
  
  // 为特定接口创建新设备实例。
  static std::shared_ptr<::gloo::transport::Device> createDeviceForInterface(
      const std::string& interface);

  // 为特定主机名或地址创建新设备实例。
  static std::shared_ptr<::gloo::transport::Device> createDeviceForHostname(
      const std::string& hostname);

  // 创建新设备实例。
  // 它尝试解析此机器的主机名并绑定到该地址。
  // 如果失败（即主机名没有解析为地址），则退回绑定到回环地址。
  static std::shared_ptr<::gloo::transport::Device> createDefaultDevice();

  // 创建 ProcessGroupGloo 实例。
  static c10::intrusive_ptr<ProcessGroupGloo> createProcessGroupGloo(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      std::chrono::milliseconds timeout);

  // 显式构造函数，初始化 ProcessGroupGloo 实例。
  explicit ProcessGroupGloo(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  // 析构函数，清理 ProcessGroupGloo 实例。
  ~ProcessGroupGloo() override;

  // 返回选项的共享指针。
  c10::intrusive_ptr<Options> getOptions() {
    return store_;
  }

  // 与 barrier() 类似，但阻塞排名为 0，直到所有其他排名已确认其存活（通过从排名 0 发送/接收）。
  // 如果 waitAllRanks = true，则排名 0 能够报告所有失败的排名，否则报告它检测到的第一个失败的排名。
  void monitoredBarrier(
      const BarrierOptions& opts = BarrierOptions(),
      bool waitAllRanks = false) override;

  // 通过使排名 0 创建并使用存储器将其广播到其他排名，为整个组协商初始序列号。
  void setSequenceNumberForGroup() override;

  // 检索整个组的当前序列号，应保持同步。
  // 如果返回的数字在整个组中不一致，则可能表明存在某种集体失同步。
  uint64_t getSequenceNumberForGroup() override;

  // 返回线程数
  int getNumThreads() {
    return options_->threads;
  }



  // 返回当前线程数选项
  // 这里直接返回存储在 options_ 指针中的 threads 成员变量
  return options_->threads;
}

protected:
std::unique_ptr<::gloo::rendezvous::Store> store_;
const c10::intrusive_ptr<Options> options_;

// 每个 Gloo 上下文表示与其对等体的一组连接。
// 为了使用多个设备（或允许在单个设备上并行处理），您需要多个上下文。
std::vector<std::shared_ptr<::gloo::Context>> contexts_;
std::vector<std::thread> threads_;
bool stop_;

// 每次发起集体操作时递增。
// 该值用作集体操作的标签。集体操作按照相同顺序在进程之间发起。
// 因此，标签可以用来在并发执行期间匹配操作。
uint32_t collectiveCounter_;

// 返回下一个要使用的集体标签（使用 collectiveCounter_）。
uint32_t nextTag();

// 返回用于指定标签的上下文。
// 使用 `nextTag` 返回一个递增的数字，这应该导致上下文以轮换方式使用。
std::shared_ptr<::gloo::Context> getContext(uint32_t tag);

// 工作线程的入口点。
void runLoop(int workerIndex);

// 将工作排队以在工作线程上运行。
void enqueue(c10::intrusive_ptr<AsyncWork> work);

// 保持待处理工作队列和进行中工作向量。
// 在持有队列锁时只能对这两者进行变异。
// 我们保持这两者而不仅仅是队列，这样我们在执行障碍时可以获取到所有进行中和待处理工作的弱指针。
// 在执行障碍时，我们需要确保在完成自身之前完成所有先前的工作。
std::deque<c10::intrusive_ptr<AsyncWork>> workQueue_;
std::vector<c10::intrusive_ptr<AsyncWork>> workInProgress_;
std::mutex workMutex_;
std::condition_variable workProduceCV_;
std::condition_variable workConsumeCV_;
uint64_t seq_{0};
};

} // namespace c10d

#endif // USE_C10D_GLOO


注释：


// 结束 c10d 命名空间的定义

#endif // 使用 C10D_GLOO


这段代码片段是 C++ 中的预处理器指令和命名空间结束标记。具体解释如下：

- `};`: 用于结束一个类或函数的定义。
- `} // namespace c10d`: 结束命名空间 `c10d` 的定义。
- `#endif // USE_C10D_GLOO`: 预处理器指令 `#endif` 表示条件编译的结束，`USE_C10D_GLOO` 是条件编译的条件。当编译器遇到 `#ifdef USE_C10D_GLOO` 或 `#ifndef USE_C10D_GLOO` 时，会根据 `USE_C10D_GLOO` 是否被定义来决定是否编译后续代码。

这段代码的作用是在条件满足时定义 `c10d` 命名空间的结构和功能，并在不满足条件时跳过这部分代码。
```