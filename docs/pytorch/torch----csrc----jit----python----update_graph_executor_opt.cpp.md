# `.\pytorch\torch\csrc\jit\python\update_graph_executor_opt.cpp`

```
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
// 包含头文件 <torch/csrc/jit/python/update_graph_executor_opt.h>，用于引入相关库函数和声明

namespace torch::jit {
// 进入 torch::jit 命名空间

// 定义线程局部变量 kOptimize，并初始化为 true
thread_local bool kOptimize = true;

// 定义函数 setGraphExecutorOptimize，用于设置 kOptimize 的值
void setGraphExecutorOptimize(bool o) {
  kOptimize = o; // 将传入的参数 o 赋值给 kOptimize
}

// 定义函数 getGraphExecutorOptimize，用于获取当前 kOptimize 的值
bool getGraphExecutorOptimize() {
  return kOptimize; // 返回当前 kOptimize 的值
}

} // namespace torch::jit
// 退出 torch::jit 命名空间
```