# `.\pytorch\c10\core\DynamicCast.h`

```py
#pragma once

#pragma once 是预处理指令，确保头文件在编译单元中只包含一次。


#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Load.h>
#include <c10/util/TypeCast.h>

这些是包含其他头文件的预处理指令，引入所需的依赖项。


namespace c10 {

进入 c10 命名空间，用于组织代码，防止命名冲突。


#ifdef C10_HOST_DEVICE
#define ERROR_UNSUPPORTED_CAST CUDA_KERNEL_ASSERT(false);
#else
#define ERROR_UNSUPPORTED_CAST TORCH_CHECK(false, "Unexpected scalar type");
#endif

根据 C10_HOST_DEVICE 宏定义，定义了 ERROR_UNSUPPORTED_CAST 宏。在 CUDA 设备上，通过 CUDA_KERNEL_ASSERT(false) 断言失败；在其他情况下，通过 TORCH_CHECK(false, "Unexpected scalar type") 进行断言失败。


// Fetch a value with dynamic type src_type from ptr, and cast it to static type
// dest_t.

fetch_and_cast 函数的注释说明，从指针 ptr 中获取具有动态类型 src_type 的值，并将其转换为静态类型 dest_t。


#define FETCH_AND_CAST_CASE(type, scalartype) \
  case ScalarType::scalartype:                \
    return c10::convert<dest_t>(c10::load<type>(ptr));

FETCH_AND_CAST_CASE 宏定义了一个 switch 分支，根据 ScalarType::scalartype 的值执行不同的类型转换操作。


template <typename dest_t>
C10_HOST_DEVICE inline dest_t fetch_and_cast(
    const ScalarType src_type,
    const void* ptr) {
  switch (src_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(FETCH_AND_CAST_CASE)
    FETCH_AND_CAST_CASE(uint16_t, UInt16)
    FETCH_AND_CAST_CASE(uint32_t, UInt32)
    FETCH_AND_CAST_CASE(uint64_t, UInt64)
    default:
      ERROR_UNSUPPORTED_CAST
  }
  return dest_t(0); // just to avoid compiler warning
}

fetch_and_cast 函数模板定义，根据 src_type 参数的值调用对应的 FETCH_AND_CAST_CASE 宏，将从 ptr 指针中获取的数据转换为 dest_t 类型。如果 src_type 不在预期的范围内，则调用 ERROR_UNSUPPORTED_CAST 宏进行错误处理。


// Cast a value with static type src_t into dynamic dest_type, and store it to
// ptr.

cast_and_store 函数的注释说明，将具有静态类型 src_t 的值转换为动态类型 dest_type，并将其存储到 ptr 指针中。
#define CAST_AND_STORE_CASE(type, scalartype) \  # 定义宏，用于根据目标类型存储数据的情况
  case ScalarType::scalartype:                \  # 当目标类型是指定的标量类型时
    *(type*)ptr = c10::convert<type>(value);  \  # 将值转换为指定类型并存储在指针所指向的位置
    return;                                   \  # 返回

template <typename src_t>
C10_HOST_DEVICE inline void cast_and_store(
    const ScalarType dest_type,    # 目标类型
    void* ptr,                    # 存储数据的指针
    src_t value) {                 # 待存储的值
  switch (dest_type) {             # 根据目标类型进行分支选择
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(CAST_AND_STORE_CASE)  # 对所有标量类型执行宏定义的存储操作
    CAST_AND_STORE_CASE(uint16_t, UInt16)   # 存储操作：将 UInt16 转换为 uint16_t 类型
    CAST_AND_STORE_CASE(uint32_t, UInt32)   # 存储操作：将 UInt32 转换为 uint32_t 类型
    CAST_AND_STORE_CASE(uint64_t, UInt64)   # 存储操作：将 UInt64 转换为 uint64_t 类型
    default:;                                # 默认情况，未支持的转换类型
  }
  ERROR_UNSUPPORTED_CAST   # 如果转换类型不受支持，则抛出错误
}

#define DEFINE_UNCASTABLE(T, scalartype_)                     \  # 定义宏，用于处理不可转换的类型
  template <>                                                 \  # 模板特化：处理特定类型 T
  C10_HOST_DEVICE inline T fetch_and_cast<T>(                 \  # 从指定类型中获取数据并转换为 T 类型
      const ScalarType src_type, const void* ptr) {           \  # 源类型和数据指针
    CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == src_type);  \  # 断言：源类型必须与指定的标量类型匹配
    return c10::load<T>(ptr);                                 \  # 加载数据并转换为 T 类型返回
  }                                                           \
  template <>                                                 \  # 模板特化：处理特定类型 T
  C10_HOST_DEVICE inline void cast_and_store<T>(              \  # 将 T 类型的值存储到指定位置
      const ScalarType dest_type, void* ptr, T value) {       \  # 目标类型、存储位置和待存储的值
    CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == dest_type); \  # 断言：目标类型必须与指定的标量类型匹配
    *(T*)ptr = value;                                         \  # 将值存储到指定位置
  }

AT_FORALL_QINT_TYPES(DEFINE_UNCASTABLE)   # 对所有 qint 类型执行宏定义的处理

#undef FETCH_AND_CAST_CASE    # 取消宏定义 FETCH_AND_CAST_CASE
#undef CAST_AND_STORE_CASE    # 取消宏定义 CAST_AND_STORE_CASE
#undef DEFINE_UNCASTABLE      # 取消宏定义 DEFINE_UNCASTABLE
#undef ERROR_UNSUPPORTED_CAST # 取消宏定义 ERROR_UNSUPPORTED_CAST

} // namespace c10  # 结束命名空间 c10
```