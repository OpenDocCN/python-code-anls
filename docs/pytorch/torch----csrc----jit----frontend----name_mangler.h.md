# `.\pytorch\torch\csrc\jit\frontend\name_mangler.h`

```py
#pragma once

#include <ATen/core/qualified_name.h>
#include <torch/csrc/Export.h>

namespace torch {
namespace jit {

/**
 * class NameMangler
 *
 * Utility to mangle qualified names in order to make them unique. We use this
 * in various places where we to de-duplicate qualified names.
 */
class TORCH_API NameMangler {
 public:
  // Given a qualified name, return a mangled version that is guaranteed to be
  // unique with respect to previous/future calls of `mangled()` on this name
  // mangler instance.
  // 对给定的限定名进行名称修饰，返回一个经过修饰的版本，保证在当前名称修饰器实例的所有调用中是唯一的
  c10::QualifiedName mangle(const c10::QualifiedName& name);

 private:
  size_t mangleIndex_ = 0;  // 记录名称修饰的索引，用于确保生成唯一的修饰名
};

} // namespace jit
} // namespace torch
```