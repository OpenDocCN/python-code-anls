# `.\pytorch\torch\csrc\jit\mobile\observer.cpp`

```
#include <torch/csrc/jit/mobile/observer.h>

// 引入 torch 库中的移动端观察器头文件

namespace torch {

// 命名空间 torch 内部定义

MobileObserverConfig& observerConfig() {
  // 定义静态的移动端观察器配置实例
  static MobileObserverConfig instance;
  // 返回移动端观察器配置实例的引用
  return instance;
}

} // namespace torch

// 结束命名空间 torch
```