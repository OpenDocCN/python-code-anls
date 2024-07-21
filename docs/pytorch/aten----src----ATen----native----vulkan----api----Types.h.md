# `.\pytorch\aten\src\ATen\native\vulkan\api\Types.h`

```py
#pragma once
// 使用 pragma once 来确保头文件只被编译一次

// @lint-ignore-every CLANGTIDY bugprone-branch-clone
// 忽略 lint 工具中 bugprone-branch-clone 类型的警告

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下代码块，否则跳过

#include <cstddef>
#include <cstdint>

#include <ATen/native/vulkan/api/vk_api.h>
// 包含 Vulkan API 头文件

#include <ATen/native/vulkan/api/Exception.h>
// 包含 Vulkan 异常处理头文件

#ifdef USE_VULKAN_FP16_INFERENCE
#define VK_FORMAT_FLOAT4 VK_FORMAT_R16G16B16A16_SFLOAT
// 如果定义了 USE_VULKAN_FP16_INFERENCE 宏，则定义 VK_FORMAT_FLOAT4 为 VK_FORMAT_R16G16B16A16_SFLOAT
#else
#define VK_FORMAT_FLOAT4 VK_FORMAT_R32G32B32A32_SFLOAT
// 否则定义 VK_FORMAT_FLOAT4 为 VK_FORMAT_R32G32B32A32_SFLOAT
#endif /* USE_VULKAN_FP16_INFERENCE */

#define VK_FORALL_SCALAR_TYPES(_)               \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, Byte)     \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, Char)      \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, Int)  \
  _(bool, VK_FORMAT_R8G8B8A8_SINT, Bool)        \
  _(float, VK_FORMAT_R16G16B16A16_SFLOAT, Half) \
  _(float, VK_FORMAT_FLOAT4, Float)             \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, QInt8)     \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, QUInt8)   \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, QInt32)
// 定义了一个宏 VK_FORALL_SCALAR_TYPES，展开后包含多个条目，每个条目包括 C 类型、Vulkan 格式和对应的标签名称

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// Scalar Types
//

enum class ScalarType : int8_t {
#define DEFINE_ENUM_VAL_(ctype, vkformat, name) name,
  VK_FORALL_SCALAR_TYPES(DEFINE_ENUM_VAL_)
#undef DEFINE_ENUM_VAL_
      Undefined,
  NumOptions
};
// 定义枚举 ScalarType，包含了 VK_FORALL_SCALAR_TYPES 宏中定义的各种标签名称

#define DEFINE_CONSTANT(ctype, vkformat, name) \
  constexpr ScalarType k##name = ScalarType::name;
// 定义常量 k##name，使用 ScalarType::name 作为其值

VK_FORALL_SCALAR_TYPES(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT
// 展开 VK_FORALL_SCALAR_TYPES 宏，为每种标签名称定义对应的常量

/*
 * Given a `ScalarType`, return the corresponding `VkFormat` that should be used
 * for image texture storage. The `ScalarType` to `VkFormat` mapping is dictated
 * by the `VK_FORALL_SCALAR_TYPE` macro in `api/Types.h`
 */
inline VkFormat to_vkformat(const ScalarType t) {
#define CASE_VK_FORMAT(ctype, vkformat, name) \
  case ScalarType::name:                      \
    return vkformat;
// 定义一个函数 to_vkformat，根据输入的 ScalarType 返回相应的 VkFormat

  switch (t) {
    VK_FORALL_SCALAR_TYPES(CASE_VK_FORMAT)
    default:
      VK_THROW("Unknown ScalarType: ", t);
  }
#undef CASE_VK_FORMAT
}
// 展开 VK_FORALL_SCALAR_TYPES 宏，根据不同的 ScalarType 返回对应的 VkFormat，如果未知则抛出异常

/*
 * Given a `VkFormat`, return the `ScalarType` that best represents the data
 * type of invidivual elements in an image texture of the `VkFormat`. Note that
 * this mapping is different from the `to_vkformat()` function, since different
 * `ScalarType`s may use the same `VkFormat`.
 */
inline ScalarType element_scalartype(const VkFormat vkformat) {
  switch (vkformat) {
    case VK_FORMAT_R8G8B8A8_SINT:
      return kChar;
    case VK_FORMAT_R8G8B8A8_UINT:
      return kByte;
    case VK_FORMAT_R32G32B32A32_SINT:
      return kInt;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return kFloat;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return kHalf;
    default:
      VK_THROW("No corresponding scalar type for unknown VkFormat: ", vkformat);
  }
}
// 根据输入的 VkFormat 返回对应的 ScalarType，如果未知则抛出异常

/*
 * Given a ScalarType, return `sizeof(ctype)` where ctype is the C type
 * corresponding to the ScalarType. The C type to ScalarType mapping is dictated
 * by the VK_FORALL_SCALAR_TYPE macro in api/Types.h
 */
inline size_t element_size(const ScalarType t) {
/**
 * 定义一个宏，根据 ScalarType 枚举值返回相应 C++ 类型的大小。
 * 用法类似于 switch 语句，针对每种 ScalarType 执行不同的 sizeof 操作。
 */
#define CASE_ELEMENTSIZE_CASE(ctype, vkformat, name) \
  case ScalarType::name:                             \
    return sizeof(ctype);

/**
 * 根据给定的 ScalarType 返回其对应的字符串表示。
 * 使用了 switch 语句，遍历所有可能的 ScalarType 枚举值，并返回对应的字符串。
 * 如果遇到未知的 ScalarType，返回 "UNKNOWN_SCALAR_TYPE"。
 */
inline const char* to_string(const ScalarType t) {
#define CASE_TO_STRING(ctype, vkformat, name) \
  case ScalarType::name:                      \
    return #name;

  switch (t) {
    VK_FORALL_SCALAR_TYPES(CASE_TO_STRING)
    default:
      return "UNKNOWN_SCALAR_TYPE";
  }
#undef CASE_TO_STRING
}

/**
 * 重载流输出操作符，用于将 ScalarType 类型转换为其字符串表示输出到流中。
 * 调用了 to_string 函数来获取 ScalarType 的字符串表示。
 */
inline std::ostream& operator<<(std::ostream& os, const ScalarType dtype) {
  return os << to_string(dtype);
}

/**
 * 用于将 ScalarType 枚举值映射到相应的 C++ 类型。
 * 使用了模板特化，每个 ScalarType 枚举值对应一个具体的 C++ 类型。
 * 实现时用到了宏 VK_FORALL_SCALAR_TYPES，为每种 ScalarType 生成特化的结构体定义。
 */
template <ScalarType N>
struct ScalarTypeToCType;

#define SPECIALIZE_ScalarTypeToCType(ctype, vkformat, scalar_type) \
  template <>                                                      \
  struct ScalarTypeToCType<                                        \
      ::at::native::vulkan::api::ScalarType::scalar_type> {        \
    using type = ctype;                                            \
  };

VK_FORALL_SCALAR_TYPES(SPECIALIZE_ScalarTypeToCType)

#undef SPECIALIZE_ScalarTypeToCPPType

/**
 * GPU 存储选项的枚举类型，描述了存储张量数据时使用的 GPU 内存类型。
 * 包括 BUFFER（使用 SSBO）、TEXTURE_3D（使用 3D 图像纹理）、TEXTURE_2D（使用 2D 图像纹理）和 UNKNOWN（未知类型）。
 */
enum class StorageType {
  BUFFER,       ///< 使用 SSBO 存储数据
  TEXTURE_3D,   ///< 使用 3D 图像纹理存储数据
  TEXTURE_2D,   ///< 使用 2D 图像纹理存储数据
  UNKNOWN,      ///< 未知存储类型
};

/**
 * GPU 内存布局的枚举类型，描述了张量数据在 GPU 存储时的布局方式。
 * 枚举值名称表明哪个维度是紧密打包的，如 TENSOR_WIDTH_PACKED 表示宽度维度是紧密打包的。
 * 用于计算着色器中逻辑张量坐标与物理像素坐标之间的转换。
 */
enum class GPUMemoryLayout : uint32_t {
  TENSOR_WIDTH_PACKED = 0u,     ///< 宽度维度紧密打包
  TENSOR_HEIGHT_PACKED = 1u,    ///< 高度维度紧密打包
  TENSOR_CHANNELS_PACKED = 2u,  ///< 通道维度紧密打包
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```