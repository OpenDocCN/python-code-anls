# `.\pytorch\torch\csrc\distributed\c10d\FakeProcessGroup.hpp`

```
#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace c10d {

// 定义一个虚拟的工作类 FakeWork，继承自 Work
class FakeWork : public Work {
 public:
  // 覆盖父类的 wait 方法，总是返回 true
  bool wait(std::chrono::milliseconds timeout) override {
    return true;
  }

  // 覆盖父类的 getFuture 方法，返回一个标记为已完成的 Future 对象
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    fut->markCompleted();
    return fut;
  }
};

// 定义一个虚拟的进程组类 FakeProcessGroup，继承自 Backend
class FakeProcessGroup : public Backend {
 public:
  // 构造函数，初始化 Backend 基类
  FakeProcessGroup(int rank, int size) : Backend(rank, size) {}

  // 覆盖父类的 broadcast 方法，返回一个 FakeWork 对象
  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& /* tensors */,
      const BroadcastOptions& /* opts */ = BroadcastOptions()) override {
    return c10::make_intrusive<FakeWork>();
  }

  // 覆盖父类的 allreduce 方法，返回一个 FakeWork 对象
  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) override {
    return c10::make_intrusive<FakeWork>();
  }

  // 覆盖父类的 allreduce_sparse 方法，返回一个 FakeWork 对象
  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) override {
    return c10::make_intrusive<FakeWork>();
  }

  // 覆盖父类的 allreduce_coalesced 方法，返回一个 FakeWork 对象
  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceCoalescedOptions& /* opts */ =
          AllreduceCoalescedOptions()) override {
    return c10::make_intrusive<FakeWork>();
  }

  // 覆盖父类的 reduce 方法，返回一个 FakeWork 对象
  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& /* tensors */,
      const ReduceOptions& /* opts */ = ReduceOptions()) override {
    return c10::make_intrusive<FakeWork>();
  }

  // NOTE [allgather on FakeProcessGroup]
  // 模拟实现 allgather 方法，在虚拟的环境中，将输入张量复制到所有输出张量中
  // 虽然不是真正的 allgather，但是通过这种复制逻辑可以进行简单的验证
  // 这里假设所有排名的输入张量相同，仅简单地复制到输出张量中
  // 注意：通常不建议将 FakeProcessGroup 用于真实数据，但是这里为了 DeviceMesh 的初始化代码进行数据验证而做出的权衡
  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    for (auto& tensor : outputTensors[0]) {
      tensor.copy_(inputTensors[0]);
    }
    return c10::make_intrusive<FakeWork>();
  }

  // 覆盖父类的 _allgather_base 方法，使用块切片操作将输入缓冲区的内容复制到输出缓冲区的每个块中
  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {
    auto chunks = outputBuffer.chunk(size_);
    for (auto& tensor : chunks) {
      tensor.copy_(inputBuffer);
    }
  return c10::make_intrusive<FakeWork>();


  // 创建一个指向 FakeWork 对象的智能指针并返回
  return c10::make_intrusive<FakeWork>();



  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& /* outputTensorLists */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {


  // 实现 allgather_coalesced 方法，用于收集和合并张量数据，返回一个指向 FakeWork 对象的智能指针
  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& /* outputTensorLists */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {



    return c10::make_intrusive<FakeWork>();
  }


    // 创建一个指向 FakeWork 对象的智能指针并返回
    return c10::make_intrusive<FakeWork>();
  }



  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {


  // 实现 allgather_into_tensor_coalesced 方法，用于将输入张量分块复制到输出张量中，并返回一个指向 FakeWork 对象的智能指针
  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override {



    for (size_t i = 0; i < outputs.size(); ++i) {
      auto chunks = outputs[i].chunk(size_);
      for (auto& chunk : chunks) {
        chunk.copy_(inputs[i]);
      }
    }
    return c10::make_intrusive<FakeWork>();
  }


    // 遍历输出张量的每个元素，对每个元素进行分块操作，将对应输入张量的数据复制到分块中，最后返回一个指向 FakeWork 对象的智能指针
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto chunks = outputs[i].chunk(size_);
      for (auto& chunk : chunks) {
        chunk.copy_(inputs[i]);
      }
    }
    return c10::make_intrusive<FakeWork>();
  }



  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const GatherOptions& /* opts */ = GatherOptions()) override {


  // 实现 gather 方法，用于收集张量数据，返回一个指向 FakeWork 对象的智能指针
  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const GatherOptions& /* opts */ = GatherOptions()) override {



    return c10::make_intrusive<FakeWork>();
  }


    // 创建一个指向 FakeWork 对象的智能指针并返回
    return c10::make_intrusive<FakeWork>();
  }



  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ScatterOptions& /* opts */ = ScatterOptions()) override {


  // 实现 scatter 方法，用于分发张量数据，返回一个指向 FakeWork 对象的智能指针
  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ScatterOptions& /* opts */ = ScatterOptions()) override {



    return c10::make_intrusive<FakeWork>();
  }


    // 创建一个指向 FakeWork 对象的智能指针并返回
    return c10::make_intrusive<FakeWork>();
  }



  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {


  // 实现 reduce_scatter 方法，用于执行张量的分散约简操作，返回一个指向 FakeWork 对象的智能指针
  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {



    return c10::make_intrusive<FakeWork>();
  }


    // 创建一个指向 FakeWork 对象的智能指针并返回
    return c10::make_intrusive<FakeWork>();
  }



  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {


  // 实现 _reduce_scatter_base 方法，用于执行基本的张量分散约简操作，返回一个指向 FakeWork 对象的智能指针
  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {



    return c10::make_intrusive<FakeWork>();
  }


    // 创建一个指向 FakeWork 对象的智能指针并返回
    return c10::make_intrusive<FakeWork>();
  }



  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& /* outputs */,
      std::vector<at::Tensor>& /* inputs */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {


  // 实现 reduce_scatter_tensor_coalesced 方法，用于执行张量的分散约简操作，返回一个指向 FakeWork 对象的智能指针
  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& /* outputs */,
      std::vector<at::Tensor>& /* inputs */,
      const ReduceScatterOptions& /* opts */ =
          ReduceScatterOptions()) override {



    return c10::make_intrusive<FakeWork>();
  }


    // 创建一个指向 FakeWork 对象的智能指针并返回
    return c10::make_intrusive<FakeWork>();
  }



  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      std::vector<int64_t>& /* outputSplitSizes */,
      std::vector<int64_t>& /* inputSplitSizes */,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) override {


  // 实现 alltoall_base 方法，用于执行张量的全部到全部操作，返回一个指向 FakeWork 对象的智能指针
  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      std::vector<int64_t>& /* outputSplitSizes */,
      std::vector<int64_t>& /* inputSplitSizes */,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) override {



    return c10::make_intrusive<FakeWork>();
  }


    // 创建一个指向 FakeWork 对象的智能指针并返回
    return c10::make_intrusive<FakeWork>();
  }



  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
  return c10::make_intrusive<FakeWork>();



  // 返回一个指向 FakeWork 对象的智能指针
  return c10::make_intrusive<FakeWork>();



  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& /* tensors */,
      int /* srcRank */,
      int /* tag */) override {



  // 接收来自指定源和标签的消息，此处未使用输入张量参数
  return c10::make_intrusive<FakeWork>();



  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& /* tensors */,
      int /* tag */) override {



  // 接收来自任意源和指定标签的消息，此处未使用输入张量参数
  return c10::make_intrusive<FakeWork>();



  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& /* opts */ = BarrierOptions()) override {



  // 执行一个屏障操作，此处未使用任何选项参数
  return c10::make_intrusive<FakeWork>();
};

} // namespace c10d
```