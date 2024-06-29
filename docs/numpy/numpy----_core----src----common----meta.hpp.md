# `D:\src\scipysrc\numpy\numpy\_core\src\common\meta.hpp`

```py
#ifndef NUMPY_CORE_SRC_COMMON_META_HPP
#define NUMPY_CORE_SRC_COMMON_META_HPP

#include "npstd.hpp"

namespace np { namespace meta {
/// @addtogroup cpp_core_meta
/// @{

namespace details {
// 模板结构体：根据指定大小和无符号类型，选择合适的整数类型
template<int size, bool unsig>
struct IntBySize;

// 特化模板结构体：根据 uint8_t 的大小和无符号类型选择 int8_t 或 uint8_t 类型
template<bool unsig>
struct IntBySize<sizeof(uint8_t), unsig> {
    using Type = typename std::conditional<
        unsig, uint8_t, int8_t>::type;
};

// 特化模板结构体：根据 uint16_t 的大小和无符号类型选择 int16_t 或 uint16_t 类型
template<bool unsig>
struct IntBySize<sizeof(uint16_t), unsig> {
    using Type = typename std::conditional<
        unsig, uint16_t, int16_t>::type;
};

// 特化模板结构体：根据 uint32_t 的大小和无符号类型选择 int32_t 或 uint32_t 类型
template<bool unsig>
struct IntBySize<sizeof(uint32_t), unsig> {
    using Type = typename std::conditional<
        unsig, uint32_t, int32_t>::type;
};

// 特化模板结构体：根据 uint64_t 的大小和无符号类型选择 int64_t 或 uint64_t 类型
template<bool unsig>
struct IntBySize<sizeof(uint64_t), unsig> {
    using Type = typename std::conditional<
        unsig, uint64_t, int64_t>::type;
};
} // namespace details

/// 提供任意整数类型的安全转换为固定宽度整数类型。
template<typename T>
struct FixedWidth {
    // 获取 T 类型的大小和无符号信息，选择对应的固定宽度整数类型 TF_
    using TF_ = typename details::IntBySize<
        sizeof(T), std::is_unsigned<T>::value
    >::Type;

    // 如果 T 是整数类型，则 Type 是 TF_；否则 Type 是 T 自身。
    using Type = typename std::conditional<
        std::is_integral<T>::value, TF_, T
    >::type;
};

/// @} cpp_core_meta

}} // namespace np::meta

#endif // NUMPY_CORE_SRC_COMMON_META_HPP
```