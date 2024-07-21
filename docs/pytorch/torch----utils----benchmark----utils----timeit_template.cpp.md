# `.\pytorch\torch\utils\benchmark\utils\timeit_template.cpp`

```py
/* C++ template for Timer.timeit

This template will be consumed by `cpp_jit.py`, and will replace:
    `GLOBAL_SETUP_TEMPLATE_LOCATION`,
    `SETUP_TEMPLATE_LOCATION`
      and
    `STMT_TEMPLATE_LOCATION`
sections with user provided statements.
*/
#include <chrono>  // 引入时间处理库
#include <c10/util/irange.h>  // 引入c10库中的irange工具
#include <torch/csrc/utils/pybind.h>  // 引入torch库的pybind工具
#include <pybind11/pybind11.h>  // 引入pybind11库
#include <torch/extension.h>  // 引入torch扩展库

// Global setup. (e.g. #includes)
// GLOBAL_SETUP_TEMPLATE_LOCATION

double timeit(int n) {
  pybind11::gil_scoped_release no_gil;  // 释放全局解释器锁，允许多线程调用Python API

  // Setup
  // SETUP_TEMPLATE_LOCATION

  {
    // Warmup
    // STMT_TEMPLATE_LOCATION
  }

  // Main loop
  auto start_time = std::chrono::high_resolution_clock::now();  // 记录开始时间
  for (const auto loop_idx : c10::irange(n)) {
    (void)loop_idx;  // 遍历n次，每次执行下面的语句
    // STMT_TEMPLATE_LOCATION
  }
  auto end_time = std::chrono::high_resolution_clock::now();  // 记录结束时间
  return std::chrono::duration<double>(end_time - start_time).count();  // 返回执行时间（秒）
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("timeit", &timeit);  // 使用pybind11将timeit函数绑定为Python可调用的扩展函数
}
```