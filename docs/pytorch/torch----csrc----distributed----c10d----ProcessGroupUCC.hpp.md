# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupUCC.hpp`

```py
#pragma once
// 只包含一次这个头文件的标准预处理命令

#ifdef USE_C10D_UCC
// 如果定义了 USE_C10D_UCC 宏，则包含 UCCUtils.hpp 头文件

#include <torch/csrc/distributed/c10d/UCCUtils.hpp>
// 包含 UCCUtils.hpp 文件，提供与 UCC 相关的实用函数和类

#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
// 包含标准 C++ 头文件，用于异常处理、内存管理、线程、容器等

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
// 包含与分布式相关的 C10d 头文件，用于后端、存储、类型和实用函数

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif
// 如果定义了 USE_CUDA 宏，则包含 CUDA 相关的头文件，用于 CUDA 事件和流管理

namespace c10d {

#define TORCH_UCC_DEVICE_NOT_SET -2
// 定义一个常量，表示未设置 UCC 设备

#ifdef USE_CUDA
#define SAVE_TENSORS(_TENSORS, _DATA)                       \
  do {                                                      \
    if ((_TENSORS)[0].device().is_cuda()) {                 \
      for (const auto i : c10::irange((_TENSORS).size())) { \
        c10::cuda::CUDACachingAllocator::recordStream(      \
            (_TENSORS)[i].storage().data_ptr(), (*stream)); \
      }                                                     \
    } else {                                                \
      (_DATA) = (_TENSORS);                                 \
    }                                                       \
  } while (0)
// 定义一个宏 SAVE_TENSORS，根据是否使用 CUDA，保存张量数据或记录 CUDA 流

#else
#define SAVE_TENSORS(_TENSORS, _DATA) (_DATA) = (_TENSORS);
// 定义一个备用宏 SAVE_TENSORS，如果不使用 CUDA，则直接保存张量数据
#endif

constexpr const char* UCC_BACKEND_NAME = "ucc";
// 定义一个常量指针，表示 UCC 后端的名称为 "ucc"

struct event_pool_t {
#ifdef USE_CUDA
  std::queue<std::unique_ptr<at::cuda::CUDAEvent>> event_pool;
#endif
  std::mutex event_pool_mutex;
};
// 定义一个结构体 event_pool_t，包含一个事件队列和一个互斥锁

class Comm;

// UCC does not support multiple CUDA devices per process.
// UCC 不支持进程内多个 CUDA 设备。

class TORCH_API ProcessGroupUCC : public Backend {
 private:
  void set_timeout(ucc_coll_args_t& args);
  // 设置超时时间的私有方法，接受 UCC 收集参数的引用

 public:
  class WorkData {
   public:
    std::vector<at::Tensor> src;
    std::vector<at::Tensor> dst;
    std::vector<at::Tensor> flat;
    WorkData() {}
    virtual ~WorkData() = default;
  };
  // 定义一个 WorkData 内部类，包含源、目标和扁平化张量向量

  class AlltoallWorkData : public WorkData {
   public:
    AlltoallWorkData(int size)
        : send_lengths(size),
          send_offsets(size),
          recv_lengths(size),
          recv_offsets(size) {}
    std::vector<uint64_t> send_lengths;
    std::vector<uint64_t> send_offsets;
    std::vector<uint64_t> recv_lengths;
    std::vector<uint64_t> recv_offsets;
  };
  // 定义一个 AlltoallWorkData 内部类，继承自 WorkData，包含发送和接收长度、偏移量向量

  class AllgathervWorkData : public WorkData {
   public:
    AllgathervWorkData(int size) : recv_lengths(size), recv_offsets(size) {}
    std::vector<uint64_t> recv_lengths;
    std::vector<uint64_t> recv_offsets;
  };
  // 定义一个 AllgathervWorkData 内部类，继承自 WorkData，包含接收长度和偏移量向量

  class ScattervWorkData : public WorkData {
   public:
    ScattervWorkData(int size) : send_lengths(size), send_offsets(size) {}
    std::vector<uint64_t> send_lengths;
    std::vector<uint64_t> send_offsets;
  };
  // 定义一个 ScattervWorkData 内部类，继承自 WorkData，包含发送长度和偏移量向量

  class ProgressEntry {
    friend class ProcessGroupUCC;
    friend class Comm;

   public:
    ProgressEntry(CommBase* comm, ucc_coll_req_h request)
        : status_(UCC_INPROGRESS), comm_(comm), request_(request) {}
    // 构造函数，初始化状态为进行中，设置通信对象和 UCC 请求句柄

    // Finalizes UCC status or exception of collective request.
    void finalize(std::exception_ptr eptr = nullptr);
    // 完成 UCC 状态或集体请求的异常处理函数

    // Members of ProgressEntry
    ucc_status_t status_;
    CommBase* comm_;
    ucc_coll_req_h request_;
  };
  // 定义一个 ProgressEntry 内部类，包含 UCC 状态、通信对象和请求句柄
    // UCC 状态变量
    ucc_status_t status_;
    // 通信基础类指针
    CommBase* comm_;
    // UCC 集体通信请求句柄
    ucc_coll_req_h request_;
    // 独占指针，管理工作数据
    std::unique_ptr<WorkData> data;
    // 异步任务的 Future 对象
    c10::intrusive_ptr<c10::ivalue::Future> future_;
    // 异常指针，用于存储异常信息
    std::exception_ptr eptr_;
  };

  // UCC 工作类，继承自基类 Work
  class WorkUCC : public Work {
    friend class ProcessGroupUCC;
    friend class Comm;

   public:
    // 构造函数，初始化 UCC 工作
    WorkUCC(
        OpType opType,  // 操作类型
        uint64_t seq,   // 序列号
        const char* prof_title,  // 分析标题
        const std::optional<std::vector<at::Tensor>>& inputs,  // 可选输入张量向量
        const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger)  // UCC 进程组日志记录器
        : Work(-1, opType, prof_title, inputs), logger_(logger), seq_(seq) {}
    // 析构函数
    ~WorkUCC();
    // 设置异常
    void setException();
    // 设置并抛出异常
    void setAndThrowException();
    // 判断任务是否完成
    bool isCompleted() override;
    // 判断任务是否成功
    bool isSuccess() const override;
    // 等待任务完成，可指定超时时间
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    // 获取 Future 对象
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
    // 获取结果张量向量
    std::vector<at::Tensor> result() override;
    // 获取来源排名
    int sourceRank() const override;
#ifdef USE_CUDA
    // 声明一个用于 CUDA 事件的唯一指针，初始化为空指针
    std::unique_ptr<at::cuda::CUDAEvent> fence = nullptr;
    // 事件池指针，初始化为空指针
    event_pool_t* ep = nullptr;
#endif
    // 源端的排名
    int sourceRank_;

   protected:
    // 共享指针，指向进度条目的实例
    std::shared_ptr<ProgressEntry> entry_;
    // 指向 ProcessGroupUCCLogger 实例的内部指针
    c10::intrusive_ptr<ProcessGroupUCCLogger> logger_;
    // 序列号
    uint64_t seq_;

   private:
    // getFuture 返回的 Future 对象
    c10::intrusive_ptr<at::ivalue::Future> future_;
    // 收集输出结果的共享指针，用于结果
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
  };

  // 显式构造函数，初始化 ProcessGroupUCC 实例
  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1,
      std::chrono::duration<float> timeout = kBackendDefaultTimeout);

  // 初始化通信方法，指定设备
  void initComm(c10::Device dev);

  // 析构函数，销毁 ProcessGroupUCC 实例
  ~ProcessGroupUCC() override;

  // 获取后端名称
  const std::string getBackendName() const override {
    return std::string(UCC_BACKEND_NAME);
  }

#ifdef USE_CUDA
  // 获取池化事件对象
  std::unique_ptr<at::cuda::CUDAEvent> getPooledEvent();
#ifdef USE_CUDA
  // CUDA 流对象，初始化为空指针
  std::unique_ptr<at::cuda::CUDAStream> stream = nullptr;
  // 用于点对点通信的 CUDA 流数组，初始化为空指针
  std::unique_ptr<at::cuda::CUDAStream> stream_p2p[2] = {nullptr, nullptr};
  // 事件池对象
  event_pool_t ep;
#endif
  // 进程组 UCC 日志记录器
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
};

class Comm {
  // UCC 日志记录器
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
  // Out-of-band 收集信息的共享指针
  std::shared_ptr<torch_ucc_oob_coll_info_t> oob;
  // UCC 通信对象
  CommUCC ucc_comm;
  // 互斥锁
  std::mutex mutex;
  // 进度线程
  std::thread progress_thread;
  // 队列生产的条件变量
  std::condition_variable queue_produce_cv;
  // 队列消费的条件变量
  std::condition_variable queue_consume_cv;
  // 进度条目的双端队列
  std::deque<std::shared_ptr<ProcessGroupUCC::ProgressEntry>> progress_queue;
  // 停止进度循环的标志
  bool stop_progress_loop;
  // 集体操作是否正在进行的标志
  bool collective_inprogress;
  // 结束阶段的 UCC 阶段
  torch_ucc_phase_t finalize_phase;

 public:
  // CUDA 设备索引
  c10::DeviceIndex cuda_device_index;
  // 构造函数，初始化 Comm 实例
  Comm(
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
      c10::Device dev,
      bool is_health_check);

  // 析构函数，销毁 Comm 实例
  ~Comm();

  // 创建 UCC 团队
  void ucc_create_team(
      ucc_team_h& team,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob);

  // 销毁 UCC 团队
  void ucc_destroy_team(ucc_team_h& team);

  // 将点对点通信操作加入队列
  c10::intrusive_ptr<Work> enqueue_p2p(
      OpType opType,
      ucc_coll_req_h request,
      const char* prof_title);

#ifdef USE_CUDA
  // 将 CUDA 集体操作加入队列
  void enqueue_cuda_collective(
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
      ucc_coll_args_t& coll,
      ucc_team_h team,
      ucc_ee_h ee);
#endif

  // 将集体操作加入队列
  void enqueue_collective(
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
      ucc_coll_args_t& coll,
      ucc_team_h team);

  // 获取 Comm 实例
  static std::shared_ptr<Comm> get_comm(
      uint32_t& id,
      c10::Device dev,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger,
      bool is_health_check = false);

  // 进度循环
  void progress_loop();
};

} // namespace c10d

#endif // USE_C10D_UCC
```