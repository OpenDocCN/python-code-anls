# `.\pytorch\aten\src\ATen\PTThreadPool.h`

```py
#pragma once

# 使用预处理指令 `#pragma once`，确保头文件只被编译一次，避免重复包含。


#include <ATen/Parallel.h>
#include <c10/core/thread_pool.h>

# 包含头文件 `<ATen/Parallel.h>` 和 `<c10/core/thread_pool.h>`，用于并行计算和线程池的支持。


namespace at {

# 进入命名空间 `at`。


class TORCH_API PTThreadPool : public c10::ThreadPool {
 public:

# 定义一个名为 `PTThreadPool` 的类，继承自 `c10::ThreadPool`。


  explicit PTThreadPool(int pool_size, int numa_node_id = -1)
      : c10::ThreadPool(pool_size, numa_node_id, []() {
          c10::setThreadName("PTThreadPool");
          at::init_num_threads();
        }) {}

# `PTThreadPool` 类的构造函数，接受线程池大小 `pool_size` 和 NUMA 节点 ID `numa_node_id` 参数，并调用基类 `c10::ThreadPool` 的构造函数初始化线程池。Lambda 函数设置线程名为 "PTThreadPool" 并初始化线程数。


};  // class PTThreadPool

# 结束 `PTThreadPool` 类定义。


} // namespace at

# 结束命名空间 `at`。
```