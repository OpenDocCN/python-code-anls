# `.\pytorch\aten\src\ATen\test\test_install\main.cpp`

```
#include <ATen/ATen.h>  // 包含 ATen 库的头文件

int main() {
  std::cout << at::ones({3,4}, at::CPU(at::kFloat)) << "\n";
  // 使用 ATen 库中的 at::ones 函数创建一个大小为 3x4 的全为 1 的张量，
  // 并指定其在 CPU 上以 float 类型存储，并输出到标准输出流
}
```