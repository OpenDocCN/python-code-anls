# `.\pytorch\torch\csrc\jit\mobile\model_tracer\KernelDTypeTracer.h`

```py
#pragma once

#include <ATen/record_function.h>
#include <c10/util/Synchronized.h>
#include <map>
#include <set>
#include <string>

namespace torch {
namespace jit {
namespace mobile {
/* The KernelDTypeTracer class handles the attachment and removal of a recording
 * callback that traces the invocation of code that handles specific dtypes in
 * kernel function implementations that are tagged with specific tags.
 *
 * You can get the set of kernel tags and the dtypes using
 * getCalledKernelTags().
 *
 * Note: This class is not thread safe or re-entrant, and should not be used
 * across multiple threads of execution.
 *
 */
struct KernelDTypeTracer final {
  // 用于存储回调函数的句柄，用于跟踪特定数据类型的内核函数调用
  at::CallbackHandle handle_;
  /* The key of the map below (std::string) is the kernel tag name (constant
   * character string) which shows up in code. The value part of type
   * std::set<std::string> is the collection of dtypes for which we need to
   * generate code for the said kernel tag.
   */
  // 定义了一个映射，将内核标签名称与需要为该内核标签生成代码的数据类型集合关联起来
  typedef std::map<std::string, std::set<std::string>> kernel_tags_type;

  // 构造函数
  KernelDTypeTracer();
  // 获取被调用的内核标签和数据类型的同步映射
  static c10::Synchronized<kernel_tags_type>& getCalledKernelTags();

  // 析构函数，用于移除回调函数
  ~KernelDTypeTracer() {
    at::removeCallback(handle_);
  }
};
} // namespace mobile
} // namespace jit
} // namespace torch
```