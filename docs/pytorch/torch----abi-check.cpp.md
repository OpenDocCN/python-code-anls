# `.\pytorch\torch\abi-check.cpp`

```py
#include <iostream>

int main() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  // 如果定义了宏 _GLIBCXX_USE_CXX11_ABI，则输出该宏的值
  std::cout << _GLIBCXX_USE_CXX11_ABI;
#else
  // 如果未定义宏 _GLIBCXX_USE_CXX11_ABI，则输出 0
  std::cout << 0;
#endif
}
```