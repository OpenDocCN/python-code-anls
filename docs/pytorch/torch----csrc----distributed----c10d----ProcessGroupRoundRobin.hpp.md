# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupRoundRobin.hpp`

```
#pragma once

#include <vector>  // 包含向量容器的头文件

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>  // 包含 Torch 分布式进程组的头文件

namespace c10d {

constexpr const char* ROUND_ROBIN_BACKEND_NAME = "round_robin";  // 定义常量表示轮询后端的名称

// ProcessGroupRoundRobin implements simple load balancing.
//
// It is constructed with multiple processes groups. Each call is dispatched to
// one of the specified process groups in a round robin fashion. Each process
// group instance must have the same rank and size.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group. This is the only way that we
// can guarantee to match up the same calls among all processes.
//
class TORCH_API ProcessGroupRoundRobin final : public ProcessGroup {
 public:
  explicit ProcessGroupRoundRobin(
      int rank,
      int size,
      std::vector<c10::intrusive_ptr<ProcessGroup>> processGroups);  // 构造函数，接受进程组的排名、大小和一组进程组的指针

  ~ProcessGroupRoundRobin() override;  // 析构函数

  const std::string getBackendName() const override {  // 获取后端名称的方法重写
  return std::string(ROUND_ROBIN_BACKEND_NAME);



  // 返回一个 std::string 对象，其内容是 ROUND_ROBIN_BACKEND_NAME 的值
  return std::string(ROUND_ROBIN_BACKEND_NAME);



  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;



  // 实现广播操作，将指定的张量广播到所有进程中
  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;



  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;



  // 实现全局归约操作，将所有进程中的张量进行归约（例如求和或求平均）
  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;



  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;



  // 实现合并全局归约操作，将所有进程中的张量进行合并后归约
  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;



  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;



  // 实现全局减少（归约）操作，将所有进程中的张量进行减少
  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;



  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;



  // 实现全收集（allgather）操作，将所有进程中的输入张量收集到每个进程的输出张量中
  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;



  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;



  // 实现基础的全收集（allgather）操作，将输入缓冲区中的数据收集到输出缓冲区中
  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;



  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;



  // 实现合并全收集（allgather）操作，将所有进程中的输入张量列表收集到每个进程的输出张量列表中
  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;



  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;



  // 实现收集（gather）操作，将所有进程中的输入张量收集到指定进程的输出张量中
  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;



  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;



  // 实现分发（scatter）操作，将指定进程的输入张量列表分发到所有进程的输出张量中
  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;



  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;



  // 实现归约分发（reduce_scatter）操作，将所有进程中的输入张量列表进行归约后分发到所有进程的输出张量中
  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;



  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;



  // 实现基础的全对全（alltoall）操作，将输入张量中的数据按指定大小分发给输出张量中的所有进程
  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;



  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;



  // 实现发送（send）操作，将指定进程的张量发送给目标进程
  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;



  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;



  // 实现接收（recv）操作，从指定进程接收张量数据
  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;



  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;



  // 实现来自任意源的接收（recvAnysource）操作，从任意进程接收张量数据
  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;



  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;



  // 实现屏障（barrier）操作，等待所有进程完成当前步骤的执行
  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;



 private:
  std::vector<c10::intrusive_ptr<ProcessGroup>> processGroups_;
  std::vector<c10::intrusive_ptr<ProcessGroup>>::const_iterator iterator_;

  // Returns the next ProcessGroup to use.
  const c10::intrusive_ptr
};
```