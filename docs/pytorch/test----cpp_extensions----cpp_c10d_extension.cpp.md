# `.\pytorch\test\cpp_extensions\cpp_c10d_extension.cpp`

```
#include "cpp_c10d_extension.hpp"

#include <map>

namespace c10d {

// ProcessGroupTest 的工作对象的析构函数
ProcessGroupTest::WorkTest::~WorkTest() {}

// 检查工作是否已完成，始终返回 true
bool ProcessGroupTest::WorkTest::isCompleted() {
  return true;
}

// 检查工作是否成功，始终返回 true
bool ProcessGroupTest::WorkTest::isSuccess() const {
  return true;
}

// 等待指定时间（未使用），始终返回 true
bool ProcessGroupTest::WorkTest::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

// ProcessGroupTest 构造函数，初始化 ProcessGroup 基类
ProcessGroupTest::ProcessGroupTest(int rank, int size)
    : ProcessGroup(rank, size) {}

// ProcessGroupTest 析构函数
ProcessGroupTest::~ProcessGroupTest() {}

// 执行广播操作，返回一个指向 WorkTest 实例的指针
c10::intrusive_ptr<Work> ProcessGroupTest::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  return c10::make_intrusive<ProcessGroupTest::WorkTest>();
}

// 执行全局归约操作，返回一个指向 WorkTest 实例的指针
c10::intrusive_ptr<Work> ProcessGroupTest::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  return c10::make_intrusive<ProcessGroupTest::WorkTest>();
}

// 抛出异常，不支持聚合全局归约操作
c10::intrusive_ptr<Work> ProcessGroupTest::allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support allreduce_coalesced");
}

// 抛出异常，不支持归约操作
c10::intrusive_ptr<Work> ProcessGroupTest::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support reduce");
}

// 抛出异常，不支持全员收集操作
c10::intrusive_ptr<Work> ProcessGroupTest::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support allgather");
}

// 抛出异常，不支持基础全员收集操作
c10::intrusive_ptr<Work> ProcessGroupTest::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support _allgather_base");
}

// 执行屏障同步操作，返回一个指向 WorkTest 实例的指针
c10::intrusive_ptr<Work> ProcessGroupTest::barrier(
    const BarrierOptions& opts) {
  return c10::make_intrusive<ProcessGroupTest::WorkTest>();
}

// 抛出异常，不支持聚合操作
c10::intrusive_ptr<Work> ProcessGroupTest::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support gather");
}

// 抛出异常，不支持分散操作
c10::intrusive_ptr<Work> ProcessGroupTest::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support scatter");
}

// 抛出异常，不支持归约分散操作
c10::intrusive_ptr<Work> ProcessGroupTest::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupTest does not support reduce_scatter");
}

// 抛出异常，不支持发送操作
c10::intrusive_ptr<Work> ProcessGroupTest::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  throw std::runtime_error("ProcessGroupTest does not support send");
}

// 接收操作的声明，未完整实现
c10::intrusive_ptr<Work> ProcessGroupTest::recv(
    // 定义一个函数，名称为 recv，接受以下参数：
    // - tensors: 存放 at::Tensor 的向量引用
    // - srcRank: 表示源端进程的排名
    // - tag: 表示消息标签
    // 抛出一个 std::runtime_error 异常，说明 ProcessGroupTest 不支持接收操作
    void recv(
        std::vector<at::Tensor>& tensors,
        int srcRank,
        int tag) {
      throw std::runtime_error("ProcessGroupTest does not support recv");
    }
}

# 异常情况：ProcessGroupTest 类不支持 recvAnysource 操作，抛出运行时错误
c10::intrusive_ptr<Work> ProcessGroupTest::recvAnysource(
    std::vector<at::Tensor>& tensor,
    int tag) {
  throw std::runtime_error("ProcessGroupTest does not support recvAnysource");
}

# 创建一个 ProcessGroupTest 对象作为测试用的进程组
c10::intrusive_ptr<ProcessGroup> ProcessGroupTest::createProcessGroupTest(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  # 使用 make_intrusive 创建一个 ProcessGroupTest 对象并返回
  return c10::make_intrusive<ProcessGroupTest>(rank, size);
}

# Python 绑定模块，定义了一个函数 createProcessGroupTest，可以在 Python 中调用
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupTest", &ProcessGroupTest::createProcessGroupTest);
}

}  # namespace c10d
```