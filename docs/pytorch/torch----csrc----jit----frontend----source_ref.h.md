# `.\pytorch\torch\csrc\jit\frontend\source_ref.h`

```py
#pragma once

#include <functional>  // 包含功能函数的头文件
#include <memory>      // 包含智能指针的头文件

#include <ATen/core/ivalue.h>         // 包含 ATen 库中 IValue 类的头文件
#include <c10/macros/Export.h>        // 包含 c10 库中导出宏定义的头文件
#include <torch/csrc/jit/frontend/source_range.h>  // 包含 Torch 前端源码范围定义的头文件

namespace torch {
namespace jit {

/**
 * SourceRef does two things:
 *   1. Owns a Source object.
 *   2. Serves as lookup key to the owned Source in associative containers, for
 *      runtime data aggregation.
 * We don't want to use std::shared_ptr<Source> directly because we want to
 * support heteogeneous lookup, and also shared_ptr is an implementation detail
 * which should be encapsulated.
 */
class TORCH_API SourceRef : public CustomClassHolder {  // 定义 SourceRef 类，继承自 CustomClassHolder
 public:
  explicit SourceRef(std::shared_ptr<Source> source_view)  // 构造函数，接受一个 shared_ptr<Source> 参数
      : source_view_(std::move(source_view)) {}  // 初始化 source_view_ 成员变量

  bool operator==(const SourceRef& other) const {  // 重载 == 运算符，比较两个 SourceRef 对象是否相等
    return source_view_ == other.source_view_;
  }

  bool operator<(const Source& other) const {  // 重载 < 运算符，比较当前对象与另一个 Source 的地址大小
    return source_view_.get() < &other;
  }

  friend bool operator<(const Source& other, const SourceRef& self) {  // 友元函数，比较一个 Source 与一个 SourceRef 的地址大小
    return &other < self.source_view_.get();
  }

  bool operator<(const SourceRef& other) const {  // 重载 < 运算符，比较当前对象与另一个 SourceRef 的内容大小
    return *this < *other.source_view_.get();
  }

  const Source* operator->() const {  // 重载箭头运算符，返回所持有的 Source 对象指针
    return source_view_.get();
  }

 private:
  std::shared_ptr<Source> source_view_;  // 私有成员变量，持有一个 Source 对象的智能指针
};

} // namespace jit
} // namespace torch
```