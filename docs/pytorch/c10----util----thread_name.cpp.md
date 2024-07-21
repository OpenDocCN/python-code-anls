# `.\pytorch\c10\util\thread_name.cpp`

```
// 包含 c10/util/thread_name.h 头文件，提供了与线程命名相关的功能

#include <c10/util/thread_name.h>

// 包含算法和数组标准库头文件
#include <algorithm>
#include <array>

// 定义宏 __GLIBC_PREREQ，用于检查 GLIBC 版本兼容性
#ifndef __GLIBC_PREREQ
#define __GLIBC_PREREQ(x, y) 0
#endif

// 如果定义了 __GLIBC__，且 GLIBC 版本大于等于 2.12，并且不是在苹果和安卓平台上
#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 12) && !defined(__APPLE__) && \
    !defined(__ANDROID__)
// 定义宏 C10_HAS_PTHREAD_SETNAME_NP 表示支持 pthread_setname_np 函数
#define C10_HAS_PTHREAD_SETNAME_NP
#endif

// 如果定义了 C10_HAS_PTHREAD_SETNAME_NP，包含 pthread.h 头文件
#ifdef C10_HAS_PTHREAD_SETNAME_NP
#include <pthread.h>
#endif

// 命名空间 c10 开始
namespace c10 {

// 如果定义了 C10_HAS_PTHREAD_SETNAME_NP，定义匿名命名空间，限制线程名最大长度为 15
#ifdef C10_HAS_PTHREAD_SETNAME_NP
namespace {
constexpr size_t kMaxThreadName = 15; // pthread 线程名的最大长度，包括空终止符
} // namespace
#endif

// 函数 setThreadName，设置当前线程的名称
void setThreadName(std::string name) {
#ifdef C10_HAS_PTHREAD_SETNAME_NP
  // 将线程名字串截断为最大长度 kMaxThreadName
  name.resize(std::min(name.size(), kMaxThreadName));

  // 调用 pthread_setname_np 设置当前线程的名称
  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

// 函数 getThreadName，获取当前线程的名称
std::string getThreadName() {
#ifdef C10_HAS_PTHREAD_SETNAME_NP
  std::array<char, kMaxThreadName + 1> name{}; // 创建 char 数组存储线程名
  // 调用 pthread_getname_np 获取当前线程的名称，并存储在 name 数组中
  pthread_getname_np(pthread_self(), name.data(), name.size());
  // 返回获取到的线程名称
  return name.data();
#else
  // 如果不支持 pthread_setname_np，返回空字符串
  return "";
#endif
}

} // namespace c10
// 命名空间 c10 结束
```