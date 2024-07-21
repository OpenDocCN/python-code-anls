# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroup.hpp`

```
// 仅在编译器支持时生效，确保头文件只被包含一次
#pragma once

// 引入必要的头文件和库
#include <torch/csrc/distributed/c10d/Backend.hpp> // 包含分布式后端相关定义
#include <memory>                                   // 引入内存管理
#include <unordered_map>                            // 引入无序映射容器
#include <utility>                                  // 引入实用工具
#include <vector>                                   // 引入向量容器

#include <ATen/ATen.h>                              // 引入 PyTorch 的 ATen 库
#include <ATen/core/dispatch/Dispatcher.h>          // 引入分发器
#include <c10/macros/Macros.h>                      // 引入 C10 宏定义

#include <torch/csrc/distributed/c10d/Work.hpp>    // 引入分布式工作相关定义

// *************************************************************************
// PROCESS GROUP collective communication API IS BEING CHANGED BETWEEN
// versions 1.7 and 1.8.
// PLEASE DO NOT ADD ANY DEPENDENCIES.
// SEE RFC: https://github.com/pytorch/pytorch/issues/39662
// *************************************************************************

// 定义默认超时时间为30分钟
constexpr auto kProcessGroupDefaultTimeout =
    std::chrono::milliseconds(30 * 60 * 1000);

namespace c10d {

// ProcessGroup 是一个基类，用于处理固定一组进程之间的集体和点对点通信。
//
// 下面列出的函数描述了 API，具体的实现在子类中提供。
//
// 所有执行 I/O 操作的函数都由 ProcessGroup 拥有的线程池异步执行。
// 它们返回一个对象，该对象可用于等待完成或错误。
//
// ProcessGroup 可以使用少于或等于成员数量的子组进行实例化。
// 实现必须注意可以并行使用多个进程组并进行适当的同步。
//
// ProcessGroup 假设一组固定的进程。如果集合发生变化，现有实例必须被销毁，
// 并且必须重新开始实例化和初始化。为了使进程组的成员找到彼此（从此处起称为
// 会合）
//
class TORCH_API ProcessGroup : public torch::CustomClassHolder {
 public:
  // ProcessGroup Options 是一个基本结构体，用于在构造 ProcessGroup 时
  // 定义基本选项。每个 ProcessGroup 子类应扩展此结构并定义其选项，如果它
  // 想为最终用户提供更多配置选项（超出此处定义的基本选项）。
  struct TORCH_API Options : torch::CustomClassHolder {
    // 构造函数，指定后端和超时时间，默认为30分钟
    explicit Options(
        std::string backend,
        std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout)
        : timeout(timeout), backend(std::move(backend)) {}
    ~Options() override = default;

    std::chrono::milliseconds timeout; // 超时时间

    // 后端名称
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::string backend;
  };

  // 后端类型枚举
  enum BackendType : uint8_t {
    UNDEFINED = 0,
    GLOO = 1,
    NCCL = 2,
    UCC = 3,
    MPI = 4,
    CUSTOM = 5,
  };

  // 仅用于向后兼容性设置，Ops.cpp 中仅用于 TypeDef
  explicit ProcessGroup(int rank, int size);

  // 构造函数，接受存储、排名、大小和选项
  explicit ProcessGroup(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options);
  ~ProcessGroup() override;

  // 获取当前进程的排名
  int getRank() const {
  // 返回 rank_ 的值
  return rank_;
}

int getSize() const {
  // 返回 size_ 的值
  return size_;
}

// 返回此进程组对象的唯一不透明 ID
int64_t getID() const {
  // 将指针地址转换为 int64_t 类型作为唯一 ID
  return reinterpret_cast<std::intptr_t>(this);
}

// 返回特定后端类型的后端 ID，用于与此进程组的集体操作相关联
int64_t getBackendID(BackendType backend_type) const {
  // 获取特定后端类型的后端对象，并将其指针地址转换为 int64_t 类型作为唯一 ID
  return reinterpret_cast<std::intptr_t>(getBackend(backend_type).get());
}

virtual const std::string getBackendName() const {
  // 返回后端名称，从选项中获取
  return options_->backend;
};

BackendType getBackendType() const {
  // 返回后端类型
  return backendType_;
};

virtual void startCoalescing(c10::DeviceType deviceType) {
  // 只有 nccl 后端实现了 startCoalescing，因此只对 nccl 后端执行
  auto backend = getBackend(deviceType);
  backend->startCoalescing();
}

virtual c10::intrusive_ptr<Work> endCoalescing(c10::DeviceType deviceType) {
  // 只有 nccl 后端实现了 endCoalescing，因此只对 nccl 后端执行
  auto backend = getBackend(deviceType);
  auto work = backend->endCoalescing();
  return work;
}

virtual c10::intrusive_ptr<Work> broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts = BroadcastOptions()) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::broadcast_", "")
          .typed<
              std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                  at::TensorList,
                  const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                  int64_t,
                  int64_t,
                  bool,
                  int64_t)>();
  // 不太方便在此处取消封装 opts 并在自定义 C++ 操作中再次封装。但是将 opts 保持为当前状态也很复杂。
  return std::get<1>(op.call(
      tensors,
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
      opts.rootRank,
      opts.rootTensor,
      opts.asyncOp,
      opts.timeout.count()));
}

virtual c10::intrusive_ptr<Work> allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts = AllreduceOptions()) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::allreduce_", "")
          .typed<
              std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                  at::TensorList,
                  const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                  const c10::intrusive_ptr<::c10d::ReduceOp>&,
                  const std::optional<at::Tensor>& sparse_indices,
                  int64_t)>();
    // 调用操作对象的call方法，执行allreduce_coalesced操作，返回第二个元素的值
    return std::get<1>(op.call(
        // 传递张量列表作为参数
        tensors,
        // 将当前对象的非所有权的ProcessGroup转换为具有所有权的智能指针
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        // 创建指定reduce操作的智能指针
        c10::make_intrusive<ReduceOp>(opts.reduceOp),
        // 使用稀疏索引执行操作的选项
        opts.sparseIndices,
        // 转换超时时长为整数
        opts.timeout.count()));
  }

  virtual c10::intrusive_ptr<Work> allreduce_coalesced(
      // 传递张量列表和可选参数进行coalesced allreduce操作
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions()) {
    // 获取c10d::allreduce_coalesced_的操作架构，并指定类型化函数签名
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::allreduce_coalesced_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             const c10::intrusive_ptr<::c10d::ReduceOp>&,
                             int64_t)>();
    // 调用操作对象的call方法，执行allreduce_coalesced操作
    return op.call(
        // 传递张量列表作为参数
        tensors,
        // 将当前对象的非所有权的ProcessGroup转换为具有所有权的智能指针
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        // 创建指定reduce操作的智能指针
        c10::make_intrusive<ReduceOp>(opts.reduceOp),
        // 转换超时时长为整数
        opts.timeout.count());
  }

  virtual c10::intrusive_ptr<Work> reduce(
      // 传递张量列表和可选参数进行reduce操作
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) {
    // 获取c10d::reduce_的操作架构，并指定类型化函数签名
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::reduce_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             const c10::intrusive_ptr<::c10d::ReduceOp>&,
                             int64_t,
                             int64_t,
                             int64_t)>();
    // 调用操作对象的call方法，执行reduce操作
    return op.call(
        // 传递张量列表作为参数
        tensors,
        // 将当前对象的非所有权的ProcessGroup转换为具有所有权的智能指针
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        // 创建指定reduce操作的智能指针
        c10::make_intrusive<ReduceOp>(opts.reduceOp),
        // 指定根张量的排名
        opts.rootRank,
        // 指定根张量的张量索引
        opts.rootTensor,
        // 转换超时时长为整数
        opts.timeout.count());
  }

  virtual c10::intrusive_ptr<Work> allgather(
      // 传递输入张量列表、输出张量列表和可选参数进行allgather操作
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) {
    // 获取c10d::allgather_的操作架构，并指定类型化函数签名
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::allgather_", "")
                         .typed<std::tuple<
                             std::vector<std::vector<at::Tensor>>,
                             c10::intrusive_ptr<Work>>(
                             const std::vector<std::vector<at::Tensor>>&,
                             at::TensorList,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             int64_t)>();
  // 调用名为 `_allgather_base_` 的分发函数，执行 allgather 操作，返回操作结果
  return std::get<1>(op.call(
      outputTensors,                  // 输出张量列表
      inputTensors,                   // 输入张量列表
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),  // 当前进程组的非所有权型指针
      opts.timeout.count()));        // 超时时间的毫秒数
}

// 将单个张量 inputBuffer 聚集到一个被解释为大小为 inputBuffer * WORLD_SIZE 的连续缓冲区 outputBuffer 中
// 仅供 ProcessGroup API 的实现者和高级用户使用
// 注意：此函数将在不久的将来被弃用
virtual c10::intrusive_ptr<Work> _allgather_base(
    at::Tensor& outputBuffer,         // 输出缓冲区张量
    at::Tensor& inputBuffer,          // 输入缓冲区张量
    const AllgatherOptions& opts = AllgatherOptions()) {  // allgather 操作的选项

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::_allgather_base_", "")  // 查找名为 `c10d::_allgather_base_` 的分发模式
          .typed<std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(  // 类型化为返回值为元组的函数类型
              at::Tensor&,                // 输出缓冲区张量
              at::Tensor&,                // 输入缓冲区张量
              const c10::intrusive_ptr<::c10d::ProcessGroup>&,  // 当前进程组的非所有权型指针
              bool,                       // 异步操作标志
              int64_t)>();                // 超时时间的毫秒数

  // 调用操作函数，执行 allgather 操作，返回操作的工作指针
  return std::get<1>(op.call(
      outputBuffer,                   // 输出缓冲区张量
      inputBuffer,                    // 输入缓冲区张量
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),  // 当前进程组的非所有权型指针
      opts.asyncOp,                   // 是否异步操作的标志
      opts.timeout.count()));         // 超时时间的毫秒数
}

// 此函数已被弃用，并将被移出 ProcessGroup 到 comms 模块：
// * 不要依赖此函数，
// * 不要在自己的 ProcessGroup 中实现此函数，应该实现 _allgather_base 函数
virtual c10::intrusive_ptr<Work> allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,  // 输出张量列表的向量
    std::vector<at::Tensor>& inputTensors,                   // 输入张量列表
    const AllgatherOptions& opts = AllgatherOptions()) {      // allgather 操作的选项

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::allgather_coalesced_", "")  // 查找名为 `c10d::allgather_coalesced_` 的分发模式
          .typed<c10::intrusive_ptr<Work>(                      // 类型化为返回值为工作指针的函数类型
              const std::vector<std::vector<at::Tensor>>&,      // 输入为输出张量列表的向量的常量引用
              const at::TensorList&,                            // 输入为输入张量列表的常量引用
              const c10::intrusive_ptr<::c10d::ProcessGroup>&)>();  // 当前进程组的非所有权型指针

  // 调用操作函数，执行 coalesced allgather 操作，返回操作的工作指针
  return op.call(
      outputTensorLists,              // 输出张量列表的向量
      inputTensors,                  // 输入张量列表
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this));  // 当前进程组的非所有权型指针
}

// 此函数是 `allgather_into_tensor` 的聚合版本（目前仍命名为 `_allgather_base`）。
// 向量中的每个张量对应一个 `allgather_into_tensor` 操作的输入/输出
virtual c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,                   // 输出张量列表
    std::vector<at::Tensor>& inputTensors,                    // 输入张量列表
    const AllgatherOptions& opts = AllgatherOptions()) {      // allgather 操作的选项

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::allgather_into_tensor_coalesced_", "")  // 查找名为 `c10d::allgather_into_tensor_coalesced_` 的分发模式
          .typed<c10::intrusive_ptr<Work>(                      // 类型化为返回值为工作指针的函数类型
              const at::TensorList,                            // 输入为输出张量列表的常量引用
              const at::TensorList,                            // 输入为输入张量列表的常量引用
              const c10::intrusive_ptr<::c10d::ProcessGroup>&)>();  // 当前进程组的非所有权型指针
  virtual c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) {
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::gather_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             const std::vector<std::vector<at::Tensor>>&,
                             const at::TensorList&,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             int64_t,
                             int64_t)>();
    // 调用分发器找到或抛出 c10d::gather_ 的模式，返回处理工作的指针
    return op.call(
        outputTensors,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.rootRank,
        opts.timeout.count());
    // 调用操作，传递输出张量、输入张量、进程组指针、根排名和超时时长
  }

  virtual c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::scatter_", "")
            .typed<
                std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                    const at::TensorList&,
                    const std::vector<std::vector<at::Tensor>>&,
                    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                    int64_t,
                    bool,
                    int64_t)>();
    // 调用分发器找到或抛出 c10d::scatter_ 的模式，返回输出张量和处理工作的元组
    return std::get<1>(op.call(
        outputTensors,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.rootRank,
        opts.asyncOp,
        opts.timeout.count()));
    // 调用操作，传递输出张量、输入张量、进程组指针、根排名、异步操作标志和超时时长，并返回处理工作的指针
  }

  virtual c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::reduce_scatter_", "")
            .typed<
                std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                    const at::TensorList&,
                    const std::vector<std::vector<at::Tensor>>&,
                    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                    const c10::intrusive_ptr<::c10d::ReduceOp>&,
                    int64_t)>();
    // 调用分发器找到或抛出 c10d::reduce_scatter_ 的模式，返回输出张量和处理工作的元组
  // 调用 op 对象的 call 方法，进行 reduce 操作
  return std::get<1>(op.call(
      // 输出张量列表作为输出
      outputTensors,
      // 输入张量列表作为输入
      inputTensors,
      // 使用当前对象的非所有权的 ProcessGroup 指针
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
      // 创建给定 opts.reduceOp 的 ReduceOp 指针
      c10::make_intrusive<::c10d::ReduceOp>(opts.reduceOp),
      // 将 opts.timeout 转换为整数表示的超时时间
      opts.timeout.count()));
}

// 基础的 reduce_scatter 操作实现
virtual c10::intrusive_ptr<Work> _reduce_scatter_base(
    // 输出缓冲区张量
    at::Tensor& outputBuffer,
    // 输入缓冲区张量
    at::Tensor& inputBuffer,
    // reduce scatter 操作的选项
    const ReduceScatterOptions& opts = ReduceScatterOptions()) {
  // 查找并获取 _reduce_scatter_base_ 方法的调度器对象
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::_reduce_scatter_base_", "")
          // 指定返回类型为 tuple<at::Tensor, c10::intrusive_ptr<Work>>
          .typed<std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(
              // 输出缓冲区张量
              at::Tensor&,
              // 输入缓冲区张量
              at::Tensor&,
              // ProcessGroup 对象的非所有权指针
              const c10::intrusive_ptr<::c10d::ProcessGroup>&,
              // ReduceOp 操作对象的指针
              const c10::intrusive_ptr<::c10d::ReduceOp>&,
              // 异步操作标志
              bool,
              // 超时时间
              int64_t)>();
  // 调用 op 对象的 call 方法进行 reduce scatter 操作
  return std::get<1>(op.call(
      // 输出缓冲区张量
      outputBuffer,
      // 输入缓冲区张量
      inputBuffer,
      // 使用当前对象的非所有权的 ProcessGroup 指针
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
      // 创建给定 opts.reduceOp 的 ReduceOp 指针
      c10::make_intrusive<::c10d::ReduceOp>(opts.reduceOp),
      // opts.asyncOp 表示是否异步操作
      opts.asyncOp,
      // 将 opts.timeout 转换为整数表示的超时时间
      opts.timeout.count()));
}

// 此函数是 reduce_scatter_tensor 的集合版本，即 `_reduce_scatter_base` 的高效实现
virtual c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
    // 输出张量列表
    std::vector<at::Tensor>& outputTensors,
    // 输入张量列表
    std::vector<at::Tensor>& inputTensors,
    // reduce scatter 操作的选项
    const ReduceScatterOptions& opts = ReduceScatterOptions()) {
  // 查找并获取 reduce_scatter_tensor_coalesced_ 方法的调度器对象
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::reduce_scatter_tensor_coalesced_", "")
          // 指定返回类型为 c10::intrusive_ptr<Work>
          .typed<c10::intrusive_ptr<Work>(
              // 输出张量列表的常量引用
              const at::TensorList,
              // 输入张量列表的常量引用
              const at::TensorList,
              // ProcessGroup 对象的非所有权指针
              const c10::intrusive_ptr<::c10d::ProcessGroup>&,
              // ReduceOp 操作对象的指针
              const c10::intrusive_ptr<::c10d::ReduceOp>&,
              // 超时时间
              int64_t)>();

  // 调用 op 对象的 call 方法进行 reduce scatter 操作
  return op.call(
      // 输出张量列表
      outputTensors,
      // 输入张量列表
      inputTensors,
      // 使用当前对象的非所有权的 ProcessGroup 指针
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
      // 创建给定 opts.reduceOp 的 ReduceOp 指针
      c10::make_intrusive<::c10d::ReduceOp>(opts.reduceOp),
      // 将 opts.timeout 转换为整数表示的超时时间
      opts.timeout.count());
}
    // 使用静态变量op调度器，查找并获取指定函数的调度模式，这里是"c10d::alltoall_base_"
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::alltoall_base_", "")
                         .typed<c10::intrusive_ptr<::c10d::Work>(
                             at::Tensor&,
                             at::Tensor&,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             std::vector<int64_t>,
                             std::vector<int64_t>,
                             int64_t)>();
    // 调用上述函数并返回结果
    return op.call(
        outputBuffer,
        inputBuffer,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        outputSplitSizes,
        inputSplitSizes,
        opts.timeout.count());
  }

  // 实现alltoall操作，将输出和输入张量列表按选项opts进行通信
  virtual c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) {
    // 使用静态变量op调度器，查找并获取指定函数的调度模式，这里是"c10d::alltoall_"
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::alltoall_", "")
            .typed<
                std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                    const at::TensorList&,
                    const at::TensorList&,
                    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                    int64_t)>();
    // 调用上述函数并返回结果的第二个元素，即Work对象
    return std::get<1>(op.call(
        outputTensors,
        inputTensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.timeout.count()));
  }

  // 实现监视式的障碍通信操作
  virtual void monitoredBarrier(
      const BarrierOptions& opts,
      bool wait_all_ranks = false) {
    // 使用静态变量op调度器，查找并获取指定函数的调度模式，这里是"c10d::monitored_barrier_"
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("c10d::monitored_barrier_", "")
                         .typed<void(
                             at::Tensor,
                             const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                             const std::vector<int64_t>&,
                             int64_t,
                             bool)>();
    // 创建一个空的CPU张量，作为参数传递给障碍通信函数
    at::Tensor tensor = at::empty({0}, at::TensorOptions().device(at::kCPU));
    // 调用上述函数，执行监视式的障碍通信操作
    op.call(
        tensor,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        opts.device_ids,
        opts.timeout.count(),
        wait_all_ranks);
  }

  // 为整个进程组设定一个初始序列号，由rank 0创建并通过存储广播给其他rank
  virtual void setSequenceNumberForGroup() {
    auto backendType = getBackendType();
    // 如果后端类型是GLOO、NCCL或UCC，则执行以下操作
    if (backendType == ProcessGroup::BackendType::GLOO ||
        backendType == ProcessGroup::BackendType::NCCL ||
        backendType == ProcessGroup::BackendType::UCC) {
      // 获取默认后端并调用其方法，设置整个组的序列号
      getDefaultBackend()->setSequenceNumberForGroup();
  // 如果不是支持的后端类型，抛出错误，指出该进程组不支持序列号
  } else {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not yet support sequence numbers."));
  }
}

// 获取整个组的当前序列号，这个序列号应该是同步的。如果返回的序列号在整个组内不一致，
// 可能表明存在某种集体失同步的情况。
virtual uint64_t getSequenceNumberForGroup() {
  auto backendType = getBackendType();

  // TODO: HACK for backend name to get sequence number for that backend.
  // 根据后端类型获取相应后端的序列号
  if (backendType == ProcessGroup::BackendType::GLOO ||
      backendType == ProcessGroup::BackendType::NCCL ||
      backendType == ProcessGroup::BackendType::UCC) {
    return getDefaultBackend()->getSequenceNumberForGroup();
  } else {
    // 如果不是支持的后端类型，抛出错误，指出该进程组不支持序列号
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not yet support sequence numbers."));
  }
}

// 发送张量到指定的目标排名（rank）和标签（tag），返回表示发送操作的工作对象指针
virtual c10::intrusive_ptr<Work> send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  // 查找并调用"c10d::send"的分发函数，返回发送操作的工作对象指针
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::send", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t)>();
  return op.call(
      tensors,
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
      dstRank,
      tag);
}

// 接收张量从指定的源排名（rank）和标签（tag），返回表示接收操作的工作对象指针
virtual c10::intrusive_ptr<Work> recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  // 查找并调用"c10d::recv_"的分发函数，返回接收操作的工作对象指针
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::recv_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t,
                           int64_t)>();
  return op.call(
      tensors,
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
      srcRank,
      tag);
}

// 接收张量从任意源的操作，根据标签（tag），返回表示接收操作的工作对象指针
virtual c10::intrusive_ptr<Work> recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  // 查找并调用"c10d::recv_any_source_"的分发函数，返回接收操作的工作对象指针
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("c10d::recv_any_source_", "")
                       .typed<c10::intrusive_ptr<::c10d::Work>(
                           at::TensorList,
                           const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                           int64_t)>();
    // 调用 op 的 call 方法，执行 barrier 操作
    return op.call(
        // 使用 tensor 作为 barrier 操作的输入张量
        tensor,
        // 获取当前对象的非所有权的 ProcessGroup 指针，并传递给 barrier 方法
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        // 使用 opts 的 device_ids 作为 barrier 方法的参数
        opts.device_ids,
        // 将 opts 的 timeout 转换为整数，并作为 barrier 方法的参数
        opts.timeout.count());
  }


这段代码是 C++ 中的一个成员函数 `barrier` 的实现。注释详细解释了每个参数和操作的作用，确保读者理解代码的每个细节。
  // 返回存储在 backendTypeToBackend_ 中与 backendType_ 对应的 Backend 对象
  return backendTypeToBackend_.at(backendType_);
}

c10::intrusive_ptr<Backend> getBackend(c10::DeviceType deviceType);

// 根据 backendType 查找并返回对应的 Backend 对象，如果找不到则抛出错误
c10::intrusive_ptr<Backend> getBackend(BackendType backendType) const {
  TORCH_CHECK(
      backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end(),
      "Could not find backend type ",
      backendType,
      ".");
  return backendTypeToBackend_.at(backendType);
}

// 返回此 ProcessGroup 支持的设备类型列表
// 注意：返回类型为 `Device` 而不是 `DeviceType`，方便在 Python 层面进行比较。
// 每个返回的 `Device` 对象具有默认索引 (-1)。
std::vector<c10::Device> getDeviceTypes() const {
  std::vector<c10::Device> devices;
  devices.reserve(deviceTypes_.size());
  for (auto& dt : deviceTypes_) {
    devices.emplace_back(dt);
  }
  return devices;
}

// 注册一个完成后的钩子函数，用于处理工作完成时的动作
void registerOnCompletionHook(
    std::function<void(std::shared_ptr<WorkInfo>)>&& hook) {
  getDefaultBackend()->registerOnCompletionHook(std::move(hook));
}

// 等待所有挂起的工作完成
void waitForPendingWorks() {
  getDefaultBackend()->waitForPendingWorks();
}

// 检查是否存在钩子函数
bool hasHooks() const {
  return getDefaultBackend()->hasHooks();
}

// 获取 ProcessGroup 的名称
const std::string& getGroupName() const;
// 设置 ProcessGroup 的名称
void setGroupName(const std::string& name);
// 获取 ProcessGroup 的描述信息
const std::string& getGroupDesc() const;
// 设置 ProcessGroup 的描述信息
void setGroupDesc(const std::string& name);
// 启用集体操作的计时
void enableCollectivesTiming();

void release_resources() override;

// ProcessGroups 可选地可以与特定设备 "绑定"。
// 目前仅对 nccl 有效，允许一些选择性的优化，如自动使用 ncclCommSplit。
// 设备是在 `init_process_group` 中指定的，最终会传递到此处，然后传递到实际的后端实例中。
std::optional<at::Device> getBoundDeviceId() const {
  return bound_device_id_;
}

// 设置绑定的设备 ID
void setBoundDeviceId(std::optional<at::Device> device) {
  // 如果设备存在，则检查其是否有有效的索引
  if (device) {
    TORCH_CHECK(device->has_index(), "setBoundDeviceId must have an index");
  }
}
    // 将设备ID绑定到类成员变量中
    bound_device_id_ = device;
  }

 protected:
  // 实现该接口的类需要调用此方法进行日志设置等初始化操作
  void init();

  c10::intrusive_ptr<c10d::Store> store_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int rank_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int size_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const c10::intrusive_ptr<Options> options_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const BackendType backendType_;
  std::string pg_desc_;

  // 调试级别设置。在构造ProcessGroup时解析一次，使用过程中保持不变。
  DebugLevel dist_debug_level_{DebugLevel::Off};

  // 该ProcessGroup的后端类集合
  std::unordered_set<c10::DeviceType> deviceTypes_;
  // 将设备类型映射到后端类型的哈希表
  std::unordered_map<c10::DeviceType, BackendType> deviceTypeToBackendType_;
  // 将设备类型映射到后端对象的哈希表
  std::unordered_map<c10::DeviceType, c10::intrusive_ptr<Backend>>
      deviceTypeToBackend_;
  // 将后端类型映射到后端对象的哈希表
  std::unordered_map<BackendType, c10::intrusive_ptr<Backend>>
      backendTypeToBackend_;

  // 可选的绑定设备ID
  std::optional<at::Device> bound_device_id_;
};

} // namespace c10d
```