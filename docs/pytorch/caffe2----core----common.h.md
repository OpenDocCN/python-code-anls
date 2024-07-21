# `.\pytorch\caffe2\core\common.h`

```py
#ifndef CAFFE2_CORE_COMMON_H_
#define CAFFE2_CORE_COMMON_H_

// 包含一些标准库头文件，用于基本的数据结构和算法操作
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// 根据操作系统选择合适的头文件
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#if defined(_MSC_VER)
#include <io.h>
#else
#include <unistd.h>
#endif

// 包含由 CMake 脚本生成的宏定义和配置信息
// 这些宏定义在构建过程中自动生成，用于配置当前的 Caffe2 实例
#include "caffe2/core/macros.h"

// 包含 C10 库中定义的宏
#include <c10/macros/Macros.h>

namespace caffe2 {

// 在 caffe2 命名空间中使用常见的类的 using 声明，以避免全局命名空间的污染
using std::set;
using std::string;
using std::unique_ptr;
using std::vector;

// 定义一个跨平台的对齐宏
#if (defined _MSC_VER && !defined NOMINMAX)
#define NOMINMAX
#endif

// 使用 std::make_unique，这是 C++14 引入的标准库函数
using std::make_unique;

// 根据平台选择合适的 round 函数
#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
using ::round;
#else
using std::round;
#endif // defined(__ANDROID__) && !defined(__NDK_MAJOR__)

// 返回 Caffe2 配置和构建时使用的设置选项的映射
TORCH_API const std::map<string, string>& GetBuildOptions();

} // namespace caffe2

#endif // CAFFE2_CORE_COMMON_H_
```