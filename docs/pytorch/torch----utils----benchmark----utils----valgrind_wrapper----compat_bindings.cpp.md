# `.\pytorch\torch\utils\benchmark\utils\valgrind_wrapper\compat_bindings.cpp`

```
/* Used to collect profiles of old versions of PyTorch. */
#include <callgrind.h>  // 包含 Callgrind 工具库的头文件
#include <pybind11/pybind11.h>  // 包含 Pybind11 库的头文件

bool _valgrind_supported_platform() {
#if defined(NVALGRIND)
  return false;  // 如果 NVALGRIND 宏定义了，返回 false，表示不支持 Valgrind
#else
  return true;   // 否则返回 true，表示支持 Valgrind
#endif
}

void _valgrind_toggle() {
#if defined(NVALGRIND)
  TORCH_CHECK(false, "Valgrind is not supported.");  // 如果 NVALGRIND 宏定义了，抛出异常，表示不支持 Valgrind
#else
  CALLGRIND_TOGGLE_COLLECT;  // 否则调用 Callgrind 开始或停止收集数据
#endif
}

void _valgrind_toggle_and_dump_stats() {
#if defined(NVALGRIND)
  TORCH_CHECK(false, "Valgrind is not supported.");  // 如果 NVALGRIND 宏定义了，抛出异常，表示不支持 Valgrind
#else
  // NB: See note in Module.cpp
  CALLGRIND_TOGGLE_COLLECT;  // 调用 Callgrind 开始或停止收集数据
  CALLGRIND_DUMP_STATS;      // 将 Callgrind 的统计信息输出到文件
#endif
}

PYBIND11_MODULE(callgrind_bindings, m) {
  m.def("_valgrind_supported_platform", &_valgrind_supported_platform);  // 将 _valgrind_supported_platform 函数绑定到 Python 模块中
  m.def("_valgrind_toggle", &_valgrind_toggle);  // 将 _valgrind_toggle 函数绑定到 Python 模块中
  m.def("_valgrind_toggle_and_dump_stats", &_valgrind_toggle_and_dump_stats);  // 将 _valgrind_toggle_and_dump_stats 函数绑定到 Python 模块中
}
```