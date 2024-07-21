# `.\pytorch\torch\csrc\distributed\autograd\autograd.cpp`

```py
// 包含 ATen 库中的记录函数和 Torch 分布式自动求导模块
#include <ATen/record_function.h>
#include <torch/csrc/distributed/autograd/autograd.h>

// 定义 Torch 命名空间下的分布式自动求导模块
namespace torch {
namespace distributed {
namespace autograd {

// 定义用于性能记录的键值，用于反向传播过程的性能分析
constexpr auto kDistAutogradBackwardProfilingKey =
    "torch::distributed::autograd::backward";

// 定义分布式自动求导的后向传播函数
void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph) {
  // 记录 API 使用情况，用于性能分析
  C10_LOG_API_USAGE_ONCE("torch.distributed.autograd.backward");
  // 记录函数调用，用于性能分析，传入空的 IValue 列表
  RECORD_FUNCTION(
      kDistAutogradBackwardProfilingKey, std::vector<c10::IValue>());
  // 执行分布式引擎中的后向传播计算
  try {
    DistEngine::getInstance().execute(context_id, roots, retain_graph);
  } catch (std::exception& e) {
    // 异常捕获：如果异常类型不是 RuntimeError，会导致程序崩溃
    TORCH_CHECK(false, e.what());
  }
}

} // namespace autograd
} // namespace distributed
} // namespace torch
```