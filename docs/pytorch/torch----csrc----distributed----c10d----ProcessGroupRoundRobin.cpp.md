# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupRoundRobin.cpp`

```py
// 包含 Torch 分布式通信模块的头文件 ProcessGroupRoundRobin.hpp

namespace c10d {

// 定义 ProcessGroupRoundRobin 类，继承自 ProcessGroup
ProcessGroupRoundRobin::ProcessGroupRoundRobin(
    int rank,
    int size,
    std::vector<c10::intrusive_ptr<ProcessGroup>> processGroups)
    : ProcessGroup(rank, size), processGroups_(std::move(processGroups)) {
  // 发出警告信息，说明 ProcessGroupRoundRobin 类即将被弃用，并且在当前版本（1.13）之后将被移除
  TORCH_WARN(
      "ProcessGroupRoundRobin is deprecated and scheduled to be removed after this current release (1.13). ",
      "Please file an issue on https://github.com/pytorch/pytorch/issues if there are any concerns or issues with this deprecation.");
  
  // 检查 processGroups_ 向量是否为空
  TORCH_CHECK(!processGroups_.empty());
  
  // 遍历 processGroups_ 向量中的每个 ProcessGroup 对象，检查其 rank 和 size 是否与当前对象一致
  for (const auto& processGroup : processGroups_) {
    TORCH_CHECK(processGroup->getRank() == rank_);
    TORCH_CHECK(processGroup->getSize() == size_);
  }
  
  // 将 iterator_ 初始化为 processGroups_ 的开始迭代器
  iterator_ = processGroups_.begin();
}

// 定义 ProcessGroupRoundRobin 类的析构函数
ProcessGroupRoundRobin::~ProcessGroupRoundRobin() = default;

// 实现广播操作，调用下一个 ProcessGroup 的 broadcast 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  return next()->broadcast(tensors, opts);
}

// 实现全reduce操作，调用下一个 ProcessGroup 的 allreduce 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  return next()->allreduce(tensors, opts);
}

// 实现合并全reduce操作，调用下一个 ProcessGroup 的 allreduce_coalesced 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  return next()->allreduce_coalesced(tensors, opts);
}

// 实现reduce操作，调用下一个 ProcessGroup 的 reduce 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  return next()->reduce(tensors, opts);
}

// 实现全gather操作，调用下一个 ProcessGroup 的 allgather 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  return next()->allgather(outputs, inputs, opts);
};

// 实现合并全gather操作，调用下一个 ProcessGroup 的 allgather_coalesced 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  return next()->allgather(outputTensorLists, inputTensors, opts);
}

// 实现gather操作，调用下一个 ProcessGroup 的 gather 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& opts) {
  return next()->gather(outputs, inputs, opts);
};

// 实现scatter操作，调用下一个 ProcessGroup 的 scatter 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts) {
  return next()->scatter(outputs, inputs, opts);
};

// 实现reduce_scatter操作，调用下一个 ProcessGroup 的 reduce_scatter 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts) {
  return next()->reduce_scatter(outputs, inputs, opts);
};

// 实现基础的alltoall操作，调用下一个 ProcessGroup 的 alltoall_base 函数
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    // 调用下一个节点的 alltoall_base 方法，进行 all-to-all 操作
    return next()->alltoall_base(
        // 输出张量
        outputTensor,
        // 输入张量
        inputTensor,
        // 输出分片大小的向量
        outputSplitSizes,
        // 输入分片大小的向量
        inputSplitSizes,
        // AllToAll 操作的选项参数
        opts);
};

c10::intrusive_ptr<Work> ProcessGroupRoundRobin::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  TORCH_CHECK(false, "ProcessGroupRoundRobin does not support send");
};



// 定义 ProcessGroupRoundRobin 类的 send 方法，用于发送数据
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::send(
    std::vector<at::Tensor>& /* unused */,  // 参数未使用
    int /* unused */,                       // 参数未使用
    int /* unused */) {                     // 参数未使用
  // 发送失败时抛出错误信息
  TORCH_CHECK(false, "ProcessGroupRoundRobin does not support send");
};




c10::intrusive_ptr<Work> ProcessGroupRoundRobin::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  TORCH_CHECK(false, "ProcessGroupRoundRobin does not support recv");
};


// 定义 ProcessGroupRoundRobin 类的 recv 方法，用于接收数据
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::recv(
    std::vector<at::Tensor>& /* unused */,  // 参数未使用
    int /* unused */,                       // 参数未使用
    int /* unused */) {                     // 参数未使用
  // 接收失败时抛出错误信息
  TORCH_CHECK(false, "ProcessGroupRoundRobin does not support recv");
};



c10::intrusive_ptr<Work> ProcessGroupRoundRobin::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  TORCH_CHECK(false, "ProcessGroupRoundRobin does not support recv");
};


// 定义 ProcessGroupRoundRobin 类的 recvAnysource 方法，用于从任意源接收数据
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::recvAnysource(
    std::vector<at::Tensor>& /* unused */,  // 参数未使用
    int /* unused */) {                      // 参数未使用
  // 从任意源接收数据失败时抛出错误信息
  TORCH_CHECK(false, "ProcessGroupRoundRobin does not support recv");
};



c10::intrusive_ptr<Work> ProcessGroupRoundRobin::barrier(
    const BarrierOptions& /* unused */) {
  TORCH_CHECK(false, "ProcessGroupRoundRobin does not support barrier");
};


// 定义 ProcessGroupRoundRobin 类的 barrier 方法，用于同步
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::barrier(
    const BarrierOptions& /* unused */) {  // 参数未使用
  // 不支持 barrier 操作时抛出错误信息
  TORCH_CHECK(false, "ProcessGroupRoundRobin does not support barrier");
};



const c10::intrusive_ptr<ProcessGroup>& ProcessGroupRoundRobin::next() {
  auto& processGroup = *iterator_;
  iterator_++;
  if (iterator_ == processGroups_.end()) {
    iterator_ = processGroups_.begin();
  }
  return processGroup;
}


// 定义 ProcessGroupRoundRobin 类的 next 方法，用于获取下一个进程组
const c10::intrusive_ptr<ProcessGroup>& ProcessGroupRoundRobin::next() {
  // 获取当前迭代器指向的进程组
  auto& processGroup = *iterator_;
  iterator_++;  // 迭代器向前移动
  // 如果迭代器已经到达结尾，则重新指向进程组列表的开头
  if (iterator_ == processGroups_.end()) {
    iterator_ = processGroups_.begin();
  }
  return processGroup;  // 返回当前进程组
}



c10::intrusive_ptr<Work> ProcessGroupRoundRobin::_allgather_base(
    at::Tensor& /*unused */,
    at::Tensor& /*unused */,
    const AllgatherOptions& /*unused */) {
  TORCH_CHECK(
      false, "no support for _allgather_base in RoundRobin process group");
}


// 定义 ProcessGroupRoundRobin 类的 _allgather_base 方法，用于进行全收集操作
c10::intrusive_ptr<Work> ProcessGroupRoundRobin::_allgather_base(
    at::Tensor& /*unused */,                // 参数未使用
    at::Tensor& /*unused */,                // 参数未使用
    const AllgatherOptions& /*unused */) {  // 参数未使用
  // 不支持 RoundRobin 进程组的全收集操作时抛出错误信息
  TORCH_CHECK(
      false, "no support for _allgather_base in RoundRobin process group");
}




} // namespace c10d


// 结束 c10d 命名空间
} // namespace c10d
```