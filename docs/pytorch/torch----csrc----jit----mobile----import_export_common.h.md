# `.\pytorch\torch\csrc\jit\mobile\import_export_common.h`

```py
#pragma once
/**
 * @file
 * Declarations shared between import_data.cpp and export_data.cpp
 */

namespace torch {
namespace jit {
namespace mobile {

namespace internal {
/**
 * The name of the mobile::Module attribute which contains saved parameters, as
 * a Dict of names to Tensors. Only used for Flatbuffer serialization.
 */
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char kSavedParametersAttributeName[] = "data";
// 定义了保存参数的属性名，作为 mobile::Module 的一个字典，存储名称到张量的映射关系，仅用于 Flatbuffer 序列化。
} // namespace internal

} // namespace mobile
} // namespace jit
} // namespace torch
```