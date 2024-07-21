# `.\pytorch\torch\csrc\jit\mobile\model_tracer\BuildFeatureTracer.h`

```py
#pragma once

#include <ATen/record_function.h>  // 引入 ATen 记录函数的头文件
#include <c10/util/Synchronized.h>  // 引入 c10 工具中的同步机制头文件
#include <map>  // 引入 map 头文件
#include <set>  // 引入 set 头文件
#include <string>  // 引入 string 头文件

namespace torch {
namespace jit {
namespace mobile {

/* BuildFeatureTracer 类处理附加和移除跟踪代码调用的记录回调，这些代码处理执行通用构建特性。
 *
 * 您可以使用 getBuildFeatures() 获取已使用的构建特性集合。
 *
 * 注意：该类不是线程安全的，也不支持重入，在多线程执行环境下不应该使用。
 */
struct BuildFeatureTracer final {
  at::CallbackHandle handle_;  // 回调句柄，用于管理回调函数的注册和注销
  /* 这些是显示在代码中的自定义类名（常量字符字符串）。 */
  typedef std::set<std::string> build_feature_type;  // 定义了一个字符串集合类型 build_feature_type

  BuildFeatureTracer();  // 构造函数声明
  static c10::Synchronized<build_feature_type>& getBuildFeatures();  // 获取构建特性集合的静态方法声明

  ~BuildFeatureTracer() {  // 析构函数定义，用于移除回调函数
    at::removeCallback(handle_);  // 移除之前注册的回调函数
  }
};

} // namespace mobile
} // namespace jit
} // namespace torch
```