# `.\pytorch\torch\csrc\jit\mobile\parse_operators.h`

```
#pragma once
// 预处理命令，确保头文件只被包含一次

#include <torch/csrc/jit/mobile/function.h>
// 包含 Torch 移动端功能的头文件

namespace torch {
namespace jit {
using c10::IValue;
// 使用 c10 命名空间中的 IValue 类型

enum MobileModuleLoadOptions {
  OPERATOR_CHECK = 1,
  // 模块加载选项，用于检查运算符

  // PARSE_ALL_EXTRA_FILE_MAPS is used to gate for ExtraFileMaps to pull all
  // files automatically without explicit entries mapping. Refer to PR for a
  // detail: https://github.com/pytorch/pytorch/pull/99747
  PARSE_ALL_EXTRA_FILE_MAPS = 2,
  // PARSE_ALL_EXTRA_FILE_MAPS 用于控制是否拉取所有附加文件，而不需要显式的文件映射条目。
  // 参考 PR 了解详细信息：https://github.com/pytorch/pytorch/pull/99747
};

const uint64_t kDefaultMobileLoadOptions =
    MobileModuleLoadOptions::OPERATOR_CHECK;
// 默认的移动模块加载选项，默认为 OPERATOR_CHECK

namespace mobile {

TORCH_API void parseOperators(
    c10::ivalue::TupleElements&& ops_list,
    const uint64_t& module_load_options,
    mobile::Function* function);
// 解析运算符函数声明，接收操作列表、模块加载选项和移动端函数对象指针作为参数

} // namespace mobile
} // namespace jit
} // namespace torch
```