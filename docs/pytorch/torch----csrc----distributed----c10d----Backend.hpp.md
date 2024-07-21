# `.\pytorch\torch\csrc\distributed\c10d\Backend.hpp`

```
#pragma once

#include <memory> // 包含标准库头文件memory，用于智能指针等功能
#include <utility> // 包含标准库头文件utility，用于一些通用工具函数
#include <vector> // 包含标准库头文件vector，用于定义和操作向量（动态数组）

#include <ATen/ATen.h> // 包含ATen张量库的头文件
#include <c10/macros/Macros.h> // 包含c10宏定义的头文件

#include <torch/csrc/distributed/c10d/Types.hpp> // 包含分布式库c10d的类型定义头文件
#include <torch/csrc/distributed/c10d/Utils.hpp> // 包含分布式库c10d的工具函数头文件
#include <torch/csrc/distributed/c10d/Work.hpp> // 包含分布式库c10d的工作项定义头文件
#include <torch/csrc/distributed/c10d/debug.h> // 包含分布式库c10d的调试信息头文件

constexpr auto kBackendDefaultTimeout = // 定义constexpr变量kBackendDefaultTimeout，表示默认超时时间为30分钟
    std::chrono::milliseconds(30 * 60 * 1000);

namespace c10d {

class TORCH_API Backend : public torch::CustomClassHolder { // 定义Backend类，继承自torch的自定义类持有者
 public:
  // Backend Options is a base struct that defines the basic options
  // when constructing a Backend. Each Backend subclass should
  // extend this struct and define its options if it wants to provide more
  // config options (beyond basic ones defined here) to end user.
  struct TORCH_API Options : torch::CustomClassHolder { // 定义Options结构体，继承自torch的自定义类持有者
    explicit Options(
        std::string backend,
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) // Options结构体构造函数，设置backend名称和超时时间
        : timeout(timeout), backend(std::move(backend)) {}
    ~Options() override = default; // 虚析构函数，默认实现

    std::chrono::milliseconds timeout; // 超时时间

    // backend name
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::string backend; // backend名称，常量成员
  };

  explicit Backend(int rank, int size); // Backend类构造函数声明，接受rank和size参数
  ~Backend() override = 0; // 纯虚析构函数声明

  int getRank() const { // 返回rank成员函数声明
    return rank_;
  }

  int getSize() const { // 返回size成员函数声明
    return size_;
  }

  // Returns an unique opaque ID of this backend that can be used to correlate
  // with its collectives.
  int64_t getID() const { // 返回Backend对象的唯一ID函数声明
    return reinterpret_cast<std::intptr_t>(this);
  }

  virtual bool supportsSplitting() const { // 虚函数，判断是否支持分割操作，默认返回false
    return false;
  }

  virtual void startCoalescing() { // 虚函数，开始合并操作，如果子类未实现则抛出错误信息
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not implement startCoalescing"));
  }

  virtual c10::intrusive_ptr<Work> endCoalescing() { // 虚函数，结束合并操作，如果子类未实现则抛出错误信息
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not implement endCoalescing"));
  }

  // Subclasses must override this method to return the backend name
  virtual const std::string getBackendName() const { // 虚函数，子类必须实现以返回backend名称，未实现则抛出错误信息
    TORCH_INTERNAL_ASSERT(false, "getBackendName is not implemented.");
  };

  virtual c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& /* tensors */, // 虚函数，广播操作，如果子类未实现则抛出错误信息
      const BroadcastOptions& /* opts */ = BroadcastOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support broadcast"));
  }

  virtual c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& /* tensors */, // 虚函数，全局归约操作，如果子类未实现则抛出错误信息
      const AllreduceOptions& /* opts */ = AllreduceOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support allreduce"));
  }

  virtual c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& /* tensors */, // 虚函数，稀疏全局归约操作，如果子类未实现则抛出错误信息
      const AllreduceOptions& /* opts */ = AllreduceOptions()) {
  // 如果后端不支持稀疏全局规约，则抛出错误信息
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          getBackendName(),
          " does not support allreduce sparse"));
}

// 执行聚合稀疏张量的工作单元，对于实现 ProcessGroup API 和高级用户使用
// 如果后端不支持聚合稀疏张量，则抛出错误信息
virtual c10::intrusive_ptr<Work> allreduce_coalesced(
    std::vector<at::Tensor>& /* tensors */,
    const AllreduceCoalescedOptions& /* opts */ =
        AllreduceCoalescedOptions()) {
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          getBackendName(),
          " does not support allreduce_coalesced"));
}

// 执行规约操作的工作单元
// 如果后端不支持规约操作，则抛出错误信息
virtual c10::intrusive_ptr<Work> reduce(
    std::vector<at::Tensor>& /* tensors */,
    const ReduceOptions& /* opts */ = ReduceOptions()) {
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support reduce"));
}

// 执行全局聚合操作的工作单元
// 如果后端不支持全局聚合操作，则抛出错误信息
virtual c10::intrusive_ptr<Work> allgather(
    std::vector<std::vector<at::Tensor>>& /* outputTensors */,
    std::vector<at::Tensor>& /* inputTensors */,
    const AllgatherOptions& /* opts */ = AllgatherOptions()) {
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support allgather"));
}

// 将单个输入缓冲区 inputBuffer 聚合成大小为 inputBuffer * WORLD_SIZE 的连续缓冲区 outputBuffer
// 仅供 ProcessGroup API 实现者和高级用户使用
// 注意：此函数将在不久的将来被弃用
virtual c10::intrusive_ptr<Work> _allgather_base(
    at::Tensor& /* outputBuffer */,
    at::Tensor& /* inputBuffer */,
    const AllgatherOptions& /* opts */ = AllgatherOptions()) {
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ", getBackendName(), " does not support _allgather_base"));
}

// 此函数已弃用，并将从 Backend 移动到 comms 中：
// * 不要依赖此函数，
// * 不要在您的 Backend 中实现它，而应实现 _allgather_base
virtual c10::intrusive_ptr<Work> allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* outputTensorLists */,
    std::vector<at::Tensor>& /* inputTensors */,
    const AllgatherOptions& /* opts */ = AllgatherOptions()) {
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          getBackendName(),
          " does not support allgather_coalesced"));
}

// 这是 allgather_into_tensor 的聚合版本（当前仍然命名为 _allgather_base）
// 向量中的每个张量对应一个 allgather_into_tensor 操作的输入/输出
virtual c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& /* outputs */,
    std::vector<at::Tensor>& /* inputs */,
    const AllgatherOptions& /* opts */ = AllgatherOptions()) {
  // 检查条件为 false，如果不满足，则抛出错误信息，指示不支持该操作
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          getBackendName(),
          " does not support allgather_into_tensor_coalesced"));
}

// 实现虚拟函数 gather，但其实际不执行任何操作，仅抛出错误信息
virtual c10::intrusive_ptr<Work> gather(
    std::vector<std::vector<at::Tensor>>& /* outputTensors */,
    std::vector<at::Tensor>& /* inputTensors */,
    const GatherOptions& /* opts */ = GatherOptions()) {
  // 检查条件为 false，如果不满足，则抛出错误信息，指示不支持该操作
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support gather"));
}

// 实现虚拟函数 scatter，但其实际不执行任何操作，仅抛出错误信息
virtual c10::intrusive_ptr<Work> scatter(
    std::vector<at::Tensor>& /* outputTensors */,
    std::vector<std::vector<at::Tensor>>& /* inputTensors */,
    const ScatterOptions& /* opts */ = ScatterOptions()) {
  // 检查条件为 false，如果不满足，则抛出错误信息，指示不支持该操作
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support scatter"));
}

// 实现虚拟函数 reduce_scatter，但其实际不执行任何操作，仅抛出错误信息
virtual c10::intrusive_ptr<Work> reduce_scatter(
    std::vector<at::Tensor>& /* outputTensors */,
    std::vector<std::vector<at::Tensor>>& /* inputTensors */,
    const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
  // 检查条件为 false，如果不满足，则抛出错误信息，指示不支持该操作
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ", getBackendName(), " does not support reduce_scatter"));
}

// 实现虚拟函数 _reduce_scatter_base，但其实际不执行任何操作，仅抛出错误信息
virtual c10::intrusive_ptr<Work> _reduce_scatter_base(
    at::Tensor& /* outputBuffer */,
    at::Tensor& /* inputBuffer */,
    const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
  // 检查条件为 false，如果不满足，则抛出错误信息，指示不支持该操作
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          getBackendName(),
          " does not support _reduce_scatter_base"));
}

// 这是 `reduce_scatter_tensor` 的聚合版本（目前仍然命名为 `_reduce_scatter_base`）。
// 向量中的每个张量对应一个 `reduce_scatter_tensor` 操作的输入/输出。
virtual c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& /* outputs */,
    std::vector<at::Tensor>& /* inputs */,
    const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
  // 检查条件为 false，如果不满足，则抛出错误信息，指示不支持该操作
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          getBackendName(),
          " does not support reduce_scatter_tensor_coalesced"));
}

// 实现虚拟函数 alltoall_base，但其实际不执行任何操作，仅抛出错误信息
virtual c10::intrusive_ptr<Work> alltoall_base(
    at::Tensor& /* outputBuffer */,
    at::Tensor& /* inputBuffer */,
    std::vector<int64_t>& /* outputSplitSizes */,
    std::vector<int64_t>& /* inputSplitSizes */,
    const AllToAllOptions& /* opts */ = AllToAllOptions()) {
  // 检查条件为 false，如果不满足，则抛出错误信息，指示不支持该操作
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ", getBackendName(), " does not support alltoall_base"));
}

// 实现虚拟函数 alltoall，但其实际不执行任何操作，仅抛出错误信息
virtual c10::intrusive_ptr<Work> alltoall(
    std::vector<at::Tensor>& /* outputTensors */,
    std::vector<at::Tensor>& /* inputTensors */,
    const AllToAllOptions& opts = AllToAllOptions()) {
  // 检查当前后端是否支持 alltoall 操作，如果不支持则报错
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support alltoall"));
}

// 监控屏障操作，当前未实现，仅 GLOO 后端支持监控屏障
virtual void monitoredBarrier(
    const BarrierOptions& /* unused */,
    bool /* unused */ = false) {
  auto backendName = getBackendName();
  // 检查当前后端是否支持监控屏障，如果不支持则报错
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          backendName,
          " does not support monitoredBarrier, only GLOO supports monitored barrier."));
}

// 为整个组设置初始序列号，仅支持 GLOO 和 NCCL 后端
virtual void setSequenceNumberForGroup() {
  auto backendName = getBackendName();
  // 检查当前后端是否支持序列号设置，如果不支持则报错
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          backendName,
          " does not yet support sequence numbers."));
}

// 获取整个组的当前序列号，应该保持同步，仅支持 GLOO 和 NCCL 后端
virtual uint64_t getSequenceNumberForGroup() {
  auto backendName = getBackendName();
  // 检查当前后端是否支持序列号获取，如果不支持则报错
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ",
          backendName,
          " does not yet support sequence numbers."));
}

// 发送操作，当前后端不支持发送操作
virtual c10::intrusive_ptr<Work> send(
    std::vector<at::Tensor>& /* tensors */,
    int /* dstRank */,
    int /* tag */) {
  // 检查当前后端是否支持发送操作，如果不支持则报错
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support send"));
}

// 接收操作，当前后端不支持接收操作
virtual c10::intrusive_ptr<Work> recv(
    std::vector<at::Tensor>& /* tensors */,
    int /* srcRank */,
    int /* tag */) {
  // 检查当前后端是否支持接收操作，如果不支持则报错
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support recv"));
}

// 接收来自任意源的操作，当前后端不支持此操作
virtual c10::intrusive_ptr<Work> recvAnysource(
    std::vector<at::Tensor>& /* tensors */,
    int /* tag */) {
  // 检查当前后端是否支持接收来自任意源的操作，如果不支持则报错
  TORCH_CHECK(
      false,
      c10::str(
          "Backend ", getBackendName(), " does not support recvAnysource"));
}

// 屏障操作，当前后端不支持屏障操作
virtual c10::intrusive_ptr<Work> barrier(
    const BarrierOptions& /* opts */ = BarrierOptions()) {
  // 检查当前后端是否支持屏障操作，如果不支持则报错
  TORCH_CHECK(
      false,
      c10::str("Backend ", getBackendName(), " does not support barrier"));
}

// 注册完成回调钩子，仅支持 ProcessGrouppNCCL 后端
virtual void registerOnCompletionHook(
    std::function<void(std::shared_ptr<WorkInfo>)>&& hook) {
  // 检查当前后端是否支持注册完成回调钩子，如果不支持则报错
  TORCH_CHECK(
      false,
      "Only ProcessGrouppNCCL supports onCompletion hook, but got ",
      getBackendName(),
      " backend.");
}

// 等待未完成的任务完成，仅支持 ProcessGrouppNCCL 后端
virtual void waitForPendingWorks() {
  // 检查当前后端是否支持等待未完成的任务完成，如果不支持则报错
  TORCH_CHECK(
      false,
      "Only ProcessGrouppNCCL supports waitForPendingWorks, but got ",
      getBackendName(),
      " backend.");
}

// 启用集体操作计时，当前未实现具体功能
virtual void enableCollectivesTiming() {
  // 使用 TORCH_CHECK 确保条件为 false，如果为 true 则抛出错误信息
  TORCH_CHECK(
      false,
      "Backend ",
      getBackendName(),
      " is missing implementation of enableCollectivesTiming.");
}

bool hasHooks() const {
  // 检查是否存在完成钩子函数
  return onCompletionHook_ != nullptr;
}

// 不要直接调用此函数，使用 ProcessGroup::setGroupName 代替
void setGroupName(const std::string& name) {
  // 设置进程组的名称
  pg_name_ = name;
}

const std::string& getGroupName() const {
  // 返回进程组的名称
  return pg_name_;
}

void setGroupDesc(const std::string& desc) {
  // 设置进程组的描述
  pg_desc_ = desc;
}

const std::string& getGroupDesc() const {
  // 返回进程组的描述
  return pg_desc_;
}

// 参见 ProcessGroup.hpp 中类似的函数以了解上下文
std::optional<at::Device> getBoundDeviceId() const {
  // 返回绑定的设备 ID（如果有）
  return bound_device_id_;
}

// 如果后端支持，则对指定设备进行即时连接
virtual void eagerConnectSingleDevice(at::Device device) {
  // 默认情况下无操作；某些后端可能会执行此优化
}

void setBoundDeviceId(std::optional<at::Device> device) {
  if (device) {
    // 如果设备存在，则使用 TORCH_CHECK 确保设备有索引
    TORCH_CHECK(device->has_index(), "setBoundDeviceId must have an index");
  }
  // 设置绑定的设备 ID
  bound_device_id_ = device;
}

protected:
// 实现此接口的类需要调用此函数进行日志设置等初始化操作
void init();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
const int rank_;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
const int size_;
// 调试级别设置。在构造 ProcessGroup 时解析一次，使用过程中保持不变
DebugLevel dist_debug_level_;
std::string pg_name_;
std::string pg_desc_;

std::function<void(std::shared_ptr<WorkInfo>)> onCompletionHook_;

std::optional<at::Device> bound_device_id_;
};

// 结束 c10d 命名空间
} // namespace c10d
```