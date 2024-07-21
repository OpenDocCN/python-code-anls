# `.\pytorch\torch\csrc\distributed\c10d\UCCTracing.hpp`

```
#pragma once

#ifdef USE_C10D_UCC

#include <torch/csrc/distributed/c10d/UCCUtils.hpp>

namespace c10d {

// 定义一个宏，用于记录通信追踪信息
#define RECORD_COMMS_TRACE(                                                    \
    _comms_tracer, _work, _opType, _rank, _comm_size, _inTensors, _outTensors) \
  do {                                                                         \
    // 如果启用了通信日志记录，则调用_comms_tracer对象的recordComms方法记录信息
    if (torch_ucc_config.enable_comms_logger) {                                \
      _comms_tracer->recordComms(                                              \
          opTypeToString(_opType),                                             \
          (uintptr_t)_work.get(),                                              \
          _rank,                                                               \
          _comm_size,                                                          \
          _inTensors,                                                          \
          _outTensors);                                                        \
    }                                                                          \
  } while (0)

// 定义一个用于记录通信追踪的类，继承自torch::CustomClassHolder
class TORCH_API CommTraceLogger : public torch::CustomClassHolder {
 private:
  std::vector<std::string> comms_trace_;  // 存储通信追踪信息的字符串向量
  std::vector<std::string> curBlocks_; /* unused */  // 当前块的名称列表，未使用
  std::vector<int64_t> curOutSplitSizes_;  // 当前输出分割大小的整数向量
  std::vector<int64_t> curInSplitSizes_;  // 当前输入分割大小的整数向量
  int curRoot_ = -1;  // 当前根的索引，默认为-1
  unsigned long seqnum = 0;  // 顺序号，用于标识记录的顺序

 public:
  void setCurBlock(const std::string& name); /* unused */  // 设置当前块的名称，未使用
  void popBlock(); /* unused */  // 弹出当前块，未使用
  // 记录可选的根信息，例如广播、聚集、散射
  void recordOptionalInfo(int root = -1);
  // 记录Alltoallv的输入/输出分割
  void recordOptionalInfo(
      const std::vector<int64_t>& outputSplitSizes = {},
      const std::vector<int64_t>& inputSplitSizes = {});
  // 记录关键的通信信息
  void recordComms(
      const std::string& collName,
      const uintptr_t workReq = 0,
      const int rank = -1,
      const int world_size = -1,
      const std::vector<at::Tensor>& inputTensors = {},
      const std::vector<at::Tensor>& outputTensor = {});
  // 返回收集到的通信追踪信息
  std::vector<std::string>& getCommsTrace() {
    return comms_trace_;
  }
};

} // namespace c10d

#endif // USE_C10D_UCC
```