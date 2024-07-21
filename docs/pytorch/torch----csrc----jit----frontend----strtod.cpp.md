# `.\pytorch\torch\csrc\jit\frontend\strtod.cpp`

```
// 包含标准库头文件，这些头文件提供了必要的函数和类型声明
#include <c10/macros/Macros.h>
#include <clocale>
#include <cstdlib>

// 如果目标平台是苹果 macOS 或者 FreeBSD，需要包含特定的头文件
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <xlocale.h>
#endif

// 下面的代码派生自 Python 函数 _PyOS_ascii_strtod
// 参见 http://hg.python.org/cpython/file/default/Python/pystrtod.c
//
// 版权归 Python Software Foundation 所有，保留所有权利
//
// 下面的修改已经应用：
// - 忽略前导空格
// - 支持解析十六进制浮点数
// - 替换 Python 的 tolower、isdigit 和 malloc 函数为对应的 C 标准库函数

// 包含 C 标准库中提供的函数和类型声明
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <locale>

// Torch JIT 的命名空间
namespace torch::jit {

// 如果是在 Microsoft Visual Studio 编译器下
#ifdef _MSC_VER
// 定义 strtod_c 函数，用于将 C 字符串转换为双精度浮点数
double strtod_c(const char* nptr, char** endptr) {
  // 静态变量 loc 是用于处理字符串转换的本地化信息对象
  static _locale_t loc = _create_locale(LC_ALL, "C");
  // 调用 C 标准库中特定于本地化的字符串转换函数
  return _strtod_l(nptr, endptr, loc);
}
// 否则（非 Microsoft Visual Studio 编译器）
#else
// 定义 strtod_c 函数，用于将 C 字符串转换为双精度浮点数
double strtod_c(const char* nptr, char** endptr) {
  // 静态变量 loc 是用于处理字符串转换的本地化信息对象
  /// NOLINTNEXTLINE(hicpp-signed-bitwise)
  static locale_t loc = newlocale(LC_ALL_MASK, "C", nullptr);
  // 调用 C 标准库中特定于本地化的字符串转换函数
  return strtod_l(nptr, endptr, loc);
}
#endif

// 定义 strtof_c 函数，用于将 C 字符串转换为单精度浮点数
float strtof_c(const char* nptr, char** endptr) {
  // 调用双精度浮点数转换函数 strtod_c，并将结果转换为单精度浮点数返回
  return (float)strtod_c(nptr, endptr);
}

} // namespace torch::jit
```