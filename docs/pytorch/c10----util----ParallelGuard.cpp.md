# `.\pytorch\c10\util\ParallelGuard.cpp`

```
#include <c10/util/ParallelGuard.h>
// 包含 c10 库的 ParallelGuard 头文件

namespace c10 {
// 进入 c10 命名空间

thread_local bool in_at_parallel = false;
// 定义一个线程局部变量，初始值为 false，用于表示当前是否处于并行状态

bool ParallelGuard::is_enabled() {
    // 返回当前线程的并行状态
    return in_at_parallel;
}

ParallelGuard::ParallelGuard(bool state) : previous_state_(is_enabled()) {
    // 在构造函数中保存当前的并行状态，设置新的并行状态
    in_at_parallel = state;
}

ParallelGuard::~ParallelGuard() {
    // 在析构函数中恢复之前保存的并行状态
    in_at_parallel = previous_state_;
}

} // namespace c10
// 结束 c10 命名空间
```