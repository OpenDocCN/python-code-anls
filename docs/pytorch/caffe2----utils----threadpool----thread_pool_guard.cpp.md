# `.\pytorch\caffe2\utils\threadpool\thread_pool_guard.cpp`

```
#include <caffe2/utils/threadpool/thread_pool_guard.h>
// 包含头文件，引入了线程池相关的工具类

namespace caffe2 {
// 进入 caffe2 命名空间

thread_local bool _NoPThreadPoolGuard_enabled = false;
// 定义一个线程局部变量，用于标识线程池保护是否启用，默认为 false

bool _NoPThreadPoolGuard::is_enabled() {
  // 返回当前线程池保护是否启用的状态
  return _NoPThreadPoolGuard_enabled;
}

void _NoPThreadPoolGuard::set_enabled(bool enabled) {
  // 设置当前线程池保护的启用状态
  _NoPThreadPoolGuard_enabled = enabled;
}

} // namespace at
// 退出 at 命名空间
```