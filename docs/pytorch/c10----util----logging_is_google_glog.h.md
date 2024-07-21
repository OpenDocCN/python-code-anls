# `.\pytorch\c10\util\logging_is_google_glog.h`

```
#ifndef C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
#define C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_

#include <map>
#include <set>
#include <vector>

#include <iomanip> // because some of the caffe2 code uses e.g. std::setw
// 使用 google glog。对于 glog 0.3.2 版本，需要在 logging.h 之前包含 stl_logging.h，
// 以确保使用 stl_logging。这是由于模板机制的特殊需求。
// 在 .cu 文件中不进行 stl 日志记录，因为 nvcc 不支持。某些移动平台也不支持 stl_logging，
// 因此我们在这种情况下添加一个重载。

#ifdef __CUDACC__
#include <cuda.h>
#endif

#if !defined(__CUDACC__) && !defined(C10_USE_MINIMAL_GLOG)
#include <glog/stl_logging.h>

// 旧版本的 glog 没有声明这个 using 声明，因此我们在这里补充声明。幸运的是，
// C++ 不会因为多次声明相同的 using 声明而报错。
namespace std {
using ::operator<<;
}

#else // !defined(__CUDACC__) && !defined(C10_USE_MINIMAL_GLOG)

// 在 cudacc 编译器的情况下，我们简单地忽略容器的打印功能。
// 基本上，我们需要为 vector/string 注册一个虚假的重载版本 - 在这里，我们只是在日志中忽略这些条目。

namespace std {
#define INSTANTIATE_FOR_CONTAINER(container)                      \
  template <class... Types>                                       \
  ostream& operator<<(ostream& out, const container<Types...>&) { \
    return out;                                                   \
  }

INSTANTIATE_FOR_CONTAINER(vector)
INSTANTIATE_FOR_CONTAINER(map)
INSTANTIATE_FOR_CONTAINER(set)
#undef INSTANTIATE_FOR_CONTAINER
} // namespace std

#endif

#include <glog/logging.h>

// glog 的附加宏定义
#define TORCH_CHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define TORCH_CHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define TORCH_CHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define TORCH_CHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define TORCH_CHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define TORCH_CHECK_GT(val1, val2) CHECK_GT(val1, val2)

#ifndef NDEBUG
#define TORCH_DCHECK_EQ(val1, val2) DCHECK_EQ(val1, val2)
#define TORCH_DCHECK_NE(val1, val2) DCHECK_NE(val1, val2)
#define TORCH_DCHECK_LE(val1, val2) DCHECK_LE(val1, val2)
#define TORCH_DCHECK_LT(val1, val2) DCHECK_LT(val1, val2)
#define TORCH_DCHECK_GE(val1, val2) DCHECK_GE(val1, val2)
#define TORCH_DCHECK_GT(val1, val2) DCHECK_GT(val1, val2)
#else // !NDEBUG
// 这些版本在优化模式下不生成代码。
#define TORCH_DCHECK_EQ(val1, val2) \
  while (false)                     \
  DCHECK_EQ(val1, val2)
#define TORCH_DCHECK_NE(val1, val2) \
  while (false)                     \
  DCHECK_NE(val1, val2)
#define TORCH_DCHECK_LE(val1, val2) \
  while (false)                     \
  DCHECK_LE(val1, val2)
#define TORCH_DCHECK_LT(val1, val2) \
  while (false)                     \
  DCHECK_LT(val1, val2)
#define TORCH_DCHECK_GE(val1, val2) \
  while (false)                     \
  DCHECK_GE(val1, val2)
#define TORCH_DCHECK_GT(val1, val2) \
  while (false)                     \
  DCHECK_GT(val1, val2)

#endif // !NDEBUG

#endif // C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
// 定义宏 TORCH_DCHECK_GT，用于调试时检查 val1 是否大于 val2，如果条件不满足则触发断言失败
#define TORCH_DCHECK_GT(val1, val2) \
  while (false)                     \  // 使用 while(false) 确保只执行一次 DCHECK_GT(val1, val2)
  DCHECK_GT(val1, val2)             // 调用 DCHECK_GT 宏进行实际的比较
#endif // NDEBUG                      // 如果未定义 NDEBUG，则结束宏定义

// 检查指针是否非空的宏定义
#define TORCH_CHECK_NOTNULL(val) CHECK_NOTNULL(val)

#ifndef NDEBUG
// 调试版本的 TORCH_CHECK_NOTNULL 宏定义
#define TORCH_DCHECK_NOTNULL(val) DCHECK_NOTNULL(val)
#else // !NDEBUG
// 非调试版本的 TORCH_DCHECK_NOTNULL 宏定义，优化后不生成实际代码
#define TORCH_DCHECK_NOTNULL(val) \
  while (false)                   \  // 使用 while(false) 确保只执行一次 DCHECK_NOTNULL(val)
  DCHECK_NOTNULL(val)             // 调用 DCHECK_NOTNULL 宏（实际上这里不执行任何实际操作）
#endif // NDEBUG                    // 如果定义了 NDEBUG，则结束宏定义

// 带有源文件和行号信息的日志记录宏定义，用于通用警告/错误处理函数的实现
//
// 注意，在此处我们简化了对 GOOGLE_STRIP_LOG 的处理
#define LOG_AT_FILE_LINE(n, file, line) \
  ::google::LogMessage(file, line, ::google::GLOG_##n).stream()

#endif // C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
```