# `.\pytorch\caffe2\utils\threadpool\thread_pool_guard.h`

```py
// 防止头文件被多次包含，仅在第一次包含时有效
#pragma once

// 包含Caffe2库的宏定义
#include <c10/macros/Macros.h>

// 定义Caffe2命名空间
namespace caffe2 {

// 一个RAII（资源获取即初始化）的线程本地保护器，用于在构造时启用或禁用梯度模式，
// 并在析构时恢复到原始值。
struct TORCH_API _NoPThreadPoolGuard {
  // 检查当前是否启用了梯度模式
  static bool is_enabled();
  // 设置是否启用梯度模式
  static void set_enabled(bool enabled);

  // 构造函数：在构造时保存当前梯度模式状态，并强制启用梯度模式
  _NoPThreadPoolGuard(): prev_mode_(_NoPThreadPoolGuard::is_enabled()) {
      _NoPThreadPoolGuard::set_enabled(true);
  }

  // 析构函数：在析构时恢复到先前保存的梯度模式状态
  ~_NoPThreadPoolGuard() {
      _NoPThreadPoolGuard::set_enabled(prev_mode_);
  }

  private:
    bool prev_mode_;  // 保存先前的梯度模式状态的私有成员变量
};

}  // Caffe2命名空间结束
```