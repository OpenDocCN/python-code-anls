# `.\pytorch\c10\core\ScalarTypeToTypeMeta.h`

```py
#pragma once

#include <c10/core/ScalarType.h> // 包含标量类型相关的头文件
#include <c10/util/Optional.h>  // 包含可选类型相关的头文件
#include <c10/util/typeid.h>    // 包含类型标识相关的头文件

// 这些函数仅仅是在 c10 中暴露了 TypeMeta/ScalarType 的桥接功能
// 当 TypeMeta 等类型从 caffe2 移动到 c10 后，应该移动到 typeid.h 中（或进行代码转换）

namespace c10 {

/**
 * 将 ScalarType 枚举值转换为 TypeMeta 句柄
 */
inline caffe2::TypeMeta scalarTypeToTypeMeta(ScalarType scalar_type) {
  return caffe2::TypeMeta::fromScalarType(scalar_type); // 使用 ScalarType 创建对应的 TypeMeta
}

/**
 * 将 TypeMeta 句柄转换为 ScalarType 枚举值
 */
inline ScalarType typeMetaToScalarType(caffe2::TypeMeta dtype) {
  return dtype.toScalarType(); // 从 TypeMeta 获取对应的 ScalarType
}

/**
 * 将 typeMetaToScalarType() 提升为可选类型
 */
inline optional<at::ScalarType> optTypeMetaToScalarType(
    optional<caffe2::TypeMeta> type_meta) {
  if (!type_meta.has_value()) { // 如果可选的 TypeMeta 为空
    return c10::nullopt; // 返回空的可选类型
  }
  return type_meta->toScalarType(); // 否则从 TypeMeta 获取对应的 ScalarType
}

/**
 * 方便起见：在 TypeMeta/ScalarType 转换中的相等性比较
 */
inline bool operator==(ScalarType t, caffe2::TypeMeta m) {
  return m.isScalarType(t); // 判断 TypeMeta 是否表示给定的 ScalarType
}

inline bool operator==(caffe2::TypeMeta m, ScalarType t) {
  return t == m; // TypeMeta 和 ScalarType 的相等性比较
}

inline bool operator!=(ScalarType t, caffe2::TypeMeta m) {
  return !(t == m); // TypeMeta 和 ScalarType 的不等性比较
}

inline bool operator!=(caffe2::TypeMeta m, ScalarType t) {
  return !(t == m); // TypeMeta 和 ScalarType 的不等性比较
}

} // namespace c10
```