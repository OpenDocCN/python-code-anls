# `.\pytorch\torch\csrc\distributed\c10d\Backend.cpp`

```
#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace c10d {

// 定义 Backend 类的构造函数，接受排名和大小作为参数
Backend::Backend(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level()) {
  // 记录一次 API 使用情况，标记为 "c10d.backend"
  C10_LOG_API_USAGE_ONCE("c10d.backend");
}

// Backend 类的析构函数，采用默认实现
Backend::~Backend() = default;

// 初始化函数，记录一次 API 使用情况，标记为 "c10d.backend_{backendName}"
void Backend::init() {
  C10_LOG_API_USAGE_ONCE(fmt::format("c10d.backend_{}", getBackendName()));
}

} // namespace c10d
```