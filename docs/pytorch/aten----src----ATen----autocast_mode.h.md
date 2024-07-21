# `.\pytorch\aten\src\ATen\autocast_mode.h`

```
#pragma once
// 只允许本头文件在编译单元中包含一次，避免重复定义错误

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <torch/library.h>
// 包含必要的头文件，提供对 PyTorch 和 ATen 的访问

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>
// 包含 C10 库的核心功能和智能指针的实现

namespace at::autocast {

TORCH_API bool is_autocast_enabled(at::DeviceType device_type);
// 声明函数，用于检查给定设备类型是否启用自动转换

TORCH_API void set_autocast_enabled(at::DeviceType device_type, bool enabled);
// 声明函数，用于设置给定设备类型的自动转换是否启用

TORCH_API at::ScalarType get_autocast_dtype(at::DeviceType device_type);
// 声明函数，用于获取给定设备类型的自动转换数据类型

TORCH_API void set_autocast_dtype(
    at::DeviceType device_type,
    at::ScalarType dtype);
// 声明函数，用于设置给定设备类型的自动转换数据类型

TORCH_API void clear_cache();
// 声明函数，用于清除缓存

TORCH_API int increment_nesting();
// 声明函数，用于增加自动转换嵌套层级计数器

TORCH_API int decrement_nesting();
// 声明函数，用于减少自动转换嵌套层级计数器

TORCH_API bool is_autocast_cache_enabled();
// 声明函数，用于检查自动转换缓存是否启用

TORCH_API void set_autocast_cache_enabled(bool enabled);
// 声明函数，用于设置自动转换缓存是否启用

// deprecated CUDA-specific autocast APIs
// 下面是针对 CUDA 的特定自动转换 API，已废弃

C10_DEPRECATED_MESSAGE(
    "at::autocast::is_enabled() is deprecated. Please use at::autocast::is_autocast_enabled(at::kCUDA) instead.")
TORCH_API inline bool is_enabled() {
  TORCH_WARN_DEPRECATION(
      "at::autocast::",
      __func__,
      "() is deprecated. Please use at::autocast::is_autocast_enabled(at::kCUDA) instead.")
  return is_autocast_enabled(at::kCUDA);
}
// 已废弃函数：检查 CUDA 自动转换是否启用

C10_DEPRECATED_MESSAGE(
    "at::autocast::set_enabled(enabled) is deprecated. Please use at::autocast::set_autocast_enabled(at::kCUDA, enabled) instead.")
TORCH_API inline void set_enabled(bool enabled) {
  TORCH_WARN_DEPRECATION(
      "at::autocast::",
      __func__,
      "(enabled) is deprecated. Please use at::autocast::set_autocast_enabled(at::kCUDA, enabled) instead.")
  set_autocast_enabled(at::kCUDA, enabled);
}
// 已废弃函数：设置 CUDA 自动转换是否启用

C10_DEPRECATED_MESSAGE(
    "at::autocast::get_autocast_gpu_dtype() is deprecated. Please use at::autocast::get_autocast_dtype(at::kCUDA) instead.")
TORCH_API inline at::ScalarType get_autocast_gpu_dtype() {
  TORCH_WARN_DEPRECATION(
      "at::autocast::",
      __func__,
      "() is deprecated. Please use at::autocast::get_autocast_dtype(at::kCUDA) instead.")
  return get_autocast_dtype(at::kCUDA);
}
// 已废弃函数：获取 CUDA 自动转换数据类型

C10_DEPRECATED_MESSAGE(
    "at::autocast::set_autocast_gpu_dtype(dtype) is deprecated. Please use at::autocast::set_autocast_dtype(at::kCUDA, dtype) instead.")
TORCH_API inline void set_autocast_gpu_dtype(at::ScalarType dtype) {
  TORCH_WARN_DEPRECATION(
      "at::autocast::",
      __func__,
      "(dtype) is deprecated. Please use at::autocast::set_autocast_dtype(at::kCUDA, dtype) instead.")
  set_autocast_dtype(at::kCUDA, dtype);
}
// 已废弃函数：设置 CUDA 自动转换数据类型
#define DECLARE_DEPRECATED_AUTOCAST_APIS(name, device_type)                                          \
  # 宏定义：声明已过时的自动转换API，接受名称和设备类型参数
  C10_DEPRECATED_MESSAGE(                                                                            \
      "at::autocast::is_" #name                                                                      \
      "_enabled() is deprecated. Please use at::autocast::is_autocast_enabled(" #device_type         \
      ") instead.")                                                                                  \
  # 定义已过时消息：警告用户该函数已过时，建议使用新的函数代替
  TORCH_API inline bool is_##name##_enabled() {                                                      \
    # 内联函数：返回当前是否启用指定名称的自动转换
    TORCH_WARN_DEPRECATION(                                                                          \
        "at::autocast::",                                                                            \
        __func__,                                                                                    \
        "() is deprecated. Please use at::autocast::is_autocast_enabled(" #device_type               \
        ") instead.")                                                                                \
    return is_autocast_enabled(device_type);                                                         \
    # 调用底层函数 is_autocast_enabled，返回其结果
  }                                                                                                  \
                                                                                                     \
  C10_DEPRECATED_MESSAGE(                                                                            \
      "at::autocast::set_" #name                                                                     \
      "_enabled(enabled) is deprecated. Please use at::autocast::set_autocast_enabled(" #device_type \
      ", enabled) instead.")                                                                         \
  # 定义已过时消息：警告用户该设置函数已过时，建议使用新的函数代替
  TORCH_API inline void set_##name##_enabled(bool enabled) {                                         \
    # 内联函数：设置是否启用指定名称的自动转换
    TORCH_WARN_DEPRECATION(                                                                          \
        "at::autocast::",                                                                            \
        __func__,                                                                                    \
        "(enabled) is deprecated. Please use at::autocast::set_autocast_enabled(" #device_type       \
        ", enabled) instead.")                                                                       \
    # 调用底层函数 set_autocast_enabled，传递指定的 enabled 参数

        ", enabled) instead.")                                                                       \
    # 调用底层函数 set_autocast_enabled，传递指定的 enabled 参数
    // 定义一个宏，用于在指定设备类型上启用或禁用自动类型转换
    #define set_autocast_enabled(device_type, enabled);                                                      \
      }                                                                                                  \
                                                                                                         \
      // 声明一个已弃用的消息，提醒用户使用新的函数
      C10_DEPRECATED_MESSAGE(                                                                            \
          "at::autocast::get_autocast_" #name                                                            \
          "_dtype() is deprecated. Please use at::autocast::get_autocast_dtype(" #device_type            \
          ") instead.")                                                                                  \
      // 定义一个内联函数，获取指定设备上自动类型转换的数据类型
      TORCH_API inline at::ScalarType get_autocast_##name##_dtype() {                                    \
        // 发出已弃用警告信息
        TORCH_WARN_DEPRECATION(                                                                          \
            "at::autocast::",                                                                            \
            __func__,                                                                                    \
            "() is deprecated. Please at::autocast::get_autocast_dtype(" #device_type                    \
            ") instead.")                                                                                \
        // 返回获取的自动类型转换数据类型
        return get_autocast_dtype(device_type);                                                          \
      }                                                                                                  \
                                                                                                         \
      // 声明一个已弃用的消息，提醒用户使用新的函数
      C10_DEPRECATED_MESSAGE(                                                                            \
          "at::autocast::set_autocast_" #name                                                            \
          "_dtype(dtype) is deprecated. Please use at::autocast::set_autocast_dtype(" #device_type       \
          ", dtype) instead.")                                                                           \
      // 定义一个内联函数，设置指定设备上的自动类型转换数据类型
      TORCH_API inline void set_autocast_##name##_dtype(at::ScalarType dtype) {                          \
        // 发出已弃用警告信息
        TORCH_WARN_DEPRECATION(                                                                          \
            "at::autocast::",                                                                            \
            __func__,                                                                                    \
            "(dtype) is deprecated. Please use at::autocast::set_autocast_dtype(" #device_type           \
            ", dtype) instead.")                                                                         \
        // 设置指定设备上的自动类型转换数据类型
        set_autocast_dtype(device_type, dtype);                                                          \
      }
// 定义一个宏，用于声明所有已废弃的特定后端自动类型转换 API
#define AT_FORALL_DEPRECATED_AUTOCAST_BAKCNEDS(_) \
  _(cpu, at::kCPU)                                \  // 声明 CPU 自动类型转换 API
  _(xpu, at::kXPU)                                \  // 声明 XPU 自动类型转换 API
  _(xla, at::kXLA)                                \  // 声明 XLA 自动类型转换 API
  _(hpu, at::kHPU)                                \  // 声明 HPU 自动类型转换 API
  _(ipu, at::kIPU)                                \  // 声明 IPU 自动类型转换 API
  _(privateuseone, at::kPrivateUse1)              // 声明 PrivateUse1 自动类型转换 API

// 在匿名命名空间中定义一个内联函数，用于判断张量是否符合自动类型转换条件
namespace {
inline bool is_autocast_eligible(
    const Tensor& tensor,                        // 输入的张量
    c10::DeviceType device_type) {               // 设备类型
  switch (device_type) {
    case c10::DeviceType::CUDA:                  // 如果设备类型是 CUDA
      return (tensor.is_cuda() || tensor.is_xla()) && tensor.is_floating_point();
    case c10::DeviceType::CPU:                   // 如果设备类型是 CPU
      return (tensor.is_cpu() || tensor.is_mkldnn()) && tensor.is_floating_point();
    case c10::DeviceType::XPU:                   // 如果设备类型是 XPU
      return tensor.is_xpu() && tensor.is_floating_point();
    case c10::DeviceType::IPU:                   // 如果设备类型是 IPU
      return tensor.is_ipu() && tensor.is_floating_point();
    case c10::DeviceType::HPU:                   // 如果设备类型是 HPU
      return tensor.is_hpu() && tensor.is_floating_point();
    case c10::DeviceType::XLA:                   // 如果设备类型是 XLA
      return tensor.is_xla() && tensor.is_floating_point();
    case c10::DeviceType::PrivateUse1:           // 如果设备类型是 PrivateUse1
      return tensor.is_privateuseone() && tensor.is_floating_point();
    default:
      return false;                              // 默认情况下不符合自动类型转换条件
  }
}
} // namespace

// 内联函数，根据设备类型获取自动类型转换的分发键值
inline DispatchKey get_autocast_dispatch_key_from_device_type(
    c10::DeviceType device_type) {               // 输入的设备类型
  switch (device_type) {
    case c10::DeviceType::CUDA:                  // 如果设备类型是 CUDA
      return DispatchKey::Autocast;              // 返回 Autocast 分发键
    case c10::DeviceType::CPU:                   // 如果设备类型是 CPU
      return DispatchKey::AutocastCPU;           // 返回 AutocastCPU 分发键
    case c10::DeviceType::XPU:                   // 如果设备类型是 XPU
      return DispatchKey::AutocastXPU;           // 返回 AutocastXPU 分发键
    case c10::DeviceType::IPU:                   // 如果设备类型是 IPU
      return DispatchKey::AutocastIPU;           // 返回 AutocastIPU 分发键
    case c10::DeviceType::HPU:                   // 如果设备类型是 HPU
      return DispatchKey::AutocastHPU;           // 返回 AutocastHPU 分发键
    case c10::DeviceType::XLA:                   // 如果设备类型是 XLA
      return DispatchKey::AutocastXLA;           // 返回 AutocastXLA 分发键
    case c10::DeviceType::PrivateUse1:           // 如果设备类型是 PrivateUse1
      return DispatchKey::AutocastPrivateUse1;   // 返回 AutocastPrivateUse1 分发键
    default:
      throw std::runtime_error(
          "unknown device type for autocast in get_autocast_dispatch_key_from_device_type");
  }
}

// 内联函数，判断给定设备类型是否支持自动类型转换
inline bool is_autocast_available(c10::DeviceType device_type) {
  if (device_type == at::kCPU || device_type == at::kCUDA ||
      device_type == at::kXPU || device_type == at::kIPU ||
      device_type == at::kHPU || device_type == at::kXLA ||
      device_type == at::kPrivateUse1) {
    return true;                                // 如果设备类型在支持列表中，返回 true
  } else {
    return false;                               // 否则返回 false
  }
}

// 内联函数，根据设备类型获取较低精度的浮点类型
inline at::ScalarType get_lower_precision_fp_from_device_type(
    c10::DeviceType device_type) {               // 输入的设备类型
  if (is_autocast_available(device_type)) {      // 如果设备类型支持自动类型转换
    return get_autocast_dtype(device_type);      // 调用函数获取自动类型转换的浮点数类型
  } else {
    throw std::runtime_error(
        "unknown device type for autocast in get_lower_precision_fp_from_device_type");
  }
}

/********************************************************************
Logic to extract the promote type from any Tensor or TensorList args.
/********************************************************************/

// Overload to catch Tensor args.
// If nextArg is floating-point, compare its scalar_type with our
// current best guess for the promote type, and update if necessary.
inline at::ScalarType prioritize(
    at::ScalarType current,
    const Tensor& nextArg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  if (current == at::kDouble) {
    AT_ERROR("promote type is double in at::autocast::prioritize");
    return current;
  }
  // Retrieve the lower precision floating-point type based on device type
  at::ScalarType lower_precision_fp =
      get_lower_precision_fp_from_device_type(device_type);
  // Check if the next argument is eligible for autocasting
  if (is_autocast_eligible(nextArg, device_type)) {
    auto next = nextArg.scalar_type();
    if (next == at::kDouble) {
      return current; // ignores double tensors
    } else if (current == at::kFloat || next == at::kFloat) {
      return at::kFloat; // prioritizes float over lower_precision_fp
    } else if (current == lower_precision_fp && next == lower_precision_fp) {
      return lower_precision_fp;
    } else {
      AT_ERROR("Unexpected floating ScalarType in at::autocast::prioritize");
      return current;
    }
  } else {
    return current;
  }
}

// Overload to catch TensorList args (for e.g. cat, stack).
// Reuses the overload above to process each Tensor in the list.
inline at::ScalarType prioritize(
    at::ScalarType current,
    const TensorList& list,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor, device_type);
  }
  return current;
}

inline at::ScalarType prioritize(
    at::ScalarType current,
    const ITensorListRef& list,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor, device_type);
  }
  return current;
}

// Template to catch non-Tensor args (no-op that returns current best guess)
template <typename T>
inline at::ScalarType prioritize(
    at::ScalarType current,
    T nextArg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  return current;
}

// Overload for the tail case.
inline at::ScalarType promote_type(
    at::ScalarType current,
    c10::DeviceType device_type) {
  return current;
}

// Unpack args and determine if incoming lower_precision_fp tensors need to be
// promoted to float32. Non-Tensor arguments are ignored.
template <typename Arg0, typename... Args>
inline at::ScalarType promote_type(
    at::ScalarType current,
    c10::DeviceType device_type,
    Arg0 arg0,
    Args... args) {
  // Prioritize the current type based on the first argument
  auto new_current = prioritize(current, arg0, device_type);
  // Recursively call promote_type for the remaining arguments
  return promote_type(new_current, device_type, args...);
}

/****************************************************
Logic to apply cached casting to any Tensor argument.
****************************************************/
inline bool is_eligible(
    const Tensor& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {

# 定义一个名为 `device_type` 的变量，类型为 `c10::DeviceType`，并初始化为 `c10::DeviceType::CUDA`
  return (

      arg.defined() && is_autocast_eligible(arg, device_type) &&

# 返回一个布尔表达式的结果，该表达式要求：
# - `arg` 必须是已定义的对象
# - 调用 `is_autocast_eligible` 函数，传递 `arg` 和 `device_type` 作为参数，用于检查是否符合自动转换条件
# - `arg` 的标量类型不等于 `at::kDouble`

      (arg.scalar_type() != at::kDouble));

# 返回一个布尔表达式的结果，该表达式检查 `arg` 的标量类型是否不等于 `at::kDouble`
// Overload to catch Tensor args
TORCH_API Tensor cached_cast(
    at::ScalarType to_type,
    const Tensor& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA);

// Overload to process optional<Tensor>
// 如果传入的参数是 std::optional<Tensor>，则进行类型转换并返回结果；如果没有值，则返回 std::nullopt。
inline std::optional<Tensor> cached_cast(
    at::ScalarType to_type,
    const std::optional<Tensor>& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  if (arg.has_value()) {
    return cached_cast(to_type, *arg, device_type);
  } else {
    return c10::nullopt;
  }
}

// Overload to process TensorLists
// 如果传入的参数是 TensorList，则对列表中的每个 Tensor 进行类型转换并返回结果向量。
inline std::vector<Tensor> cached_cast(
    at::ScalarType to_type,
    const TensorList& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.emplace_back(cached_cast(to_type, t, device_type));
  }
  return vec;
}

// Overload to process ITensorListRef
// 如果传入的参数是 ITensorListRef，则对列表中的每个 Tensor 进行类型转换并返回结果向量。
inline std::vector<Tensor> cached_cast(
    at::ScalarType to_type,
    const ITensorListRef& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.emplace_back(cached_cast(to_type, t, device_type));
  }
  return vec;
}

// Template to catch non-Tensor args.
// 如果传入的参数不是 Tensor 类型，则直接返回原始参数。
template <typename T>
inline T cached_cast(
    at::ScalarType to_type,
    T arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  return arg;
}

/*******************************************************
Logic to flip an output dtype flag.
Keep it simple for now by assuming only one such flag is
present in the argument list.  If I ever need a function
with more than flag I'll figure out something else.
The policy is:
If the user has explicity specified a dtype, respect it.
Otherwise, set it to the autocast type.
********************************************************/

// Overload to catch dtype flags
// 如果传入的参数是 std::optional<ScalarType>，则返回用户明确指定的 dtype，否则返回默认的 to_type。
std::optional<ScalarType> inline set_opt_dtype(
    at::ScalarType to_type,
    const std::optional<ScalarType>& dtype) {
  return dtype.has_value() ? dtype : to_type;
}

// Template to catch other args
// 如果传入的参数不是 std::optional<ScalarType>，则直接返回原始参数。
template <typename T>
inline T set_opt_dtype(at::ScalarType to_type, T arg) {
  return arg;
}

template <typename... Args>
inline bool firstarg_is_eligible(
    c10::DeviceType device_type,
    const Tensor& arg,
    Args... args) {
  return is_eligible(arg, device_type);
}

template <typename... Args>
inline at::ScalarType type_from_firstarg(
    c10::DeviceType device_type,
    at::ScalarType to_type,
    const Tensor& arg,
    Args... args) {
  // 根据第一个参数的类型判断是否可以进行类型转换，如果可以则返回 to_type，否则返回参数的当前标量类型。
  return (is_eligible(arg, device_type) ? to_type : arg.scalar_type());
}

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
// 枚举类型 CastPolicy，用于定义类型转换的策略，基础类型为 uint8_t
enum class CastPolicy : uint8_t {
  lower_precision_fp = 0, // 将所有输入转换为 lower_precision_fp 后再执行操作。
                          // 对于 AutocastCUDA，lower_precision_fp 当前是 fp16；
                          // 对于 AutocastCPU 或其他设备，用户可以定义其默认值为 bf16。
  fp32, // 将所有输入转换为 at::kFloat 后再执行操作。
  fp32_set_opt_dtype, // 处理带有 std::optional<ScalarType> 参数（如 softmax）的函数，
                      // 我们希望以 fp32 运行，并且可控制输出类型。
                      // fp32_set_opt_dtype 的策略是：如果输出类型已经设置，则不修改；
                      // 否则，将输出类型设置为 at::kFloat。
  fp32_append_dtype, // 处理带有某些重载接受输出类型的函数（如 norm），
                     // 我们希望以 fp32 运行。
                     // fp32_append_dtype 包装了不带输出数据类型重载的函数。
                     // 包装策略是：将 at::kFloat 附加到参数中，并重新分派到有类型意识的重载。
  promote, // 在多个参数中选择最宽的数据类型来运行。
};

/********************************************************************************************************
Templates to provide wrapper functions

I'm copying the pattern used in core/boxing/impl/WrapFunctionIntoFunctor.h to
extract args and return type. (see also
https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer)

This strategy uses an exterior "WrapFunction" that extracts arguments on behalf
of (in my case several specializations of) an interior "WrapFunction_".
Interior WrapFunction_ specializations are defined for each CastPolicy.
********************************************************************************************************/

// WrapFunction_ 的基本模板，专门为每个 CastPolicy 进行特化，每个特化包含一个 "call" 方法
template <
    CastPolicy policy,
    c10::DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class ArgList>
struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType 的特化
template <
    c10::DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::lower_precision_fp,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  // call 方法：执行特化策略 CastPolicy::lower_precision_fp 的函数调用
  static Ret call(Args... args) {
    // 禁止自动转换派遣键的保护，根据设备类型获取自动转换的派遣键
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    // 使用缓存的类型转换器 cached_cast，将参数 args 转换为 lower_precision_fp 类型，并调用函数 F
    return (*F)(cached_cast(
        get_lower_precision_fp_from_device_type(device_type),
        args,
        device_type)...);
  }
};
// 模板定义：WrapFunction_ 模板特化，处理 CastPolicy::fp32 和 General_DeviceType 的情况
template <
    c10::DeviceType device_type,               // 设备类型参数
    class Redispatch,                         // 重分派类型
    Redispatch* F,                            // 指向重分派函数的指针
    class Ret,                                // 返回类型
    class... Args>                            // 参数包
struct WrapFunction_<
    CastPolicy::fp32,                          // 使用 fp32 转换策略
    device_type,                               // 设备类型参数
    Redispatch,                                // 重分派类型
    F,                                         // 指向重分派函数的指针
    Ret,                                       // 返回类型
    guts::typelist::typelist<Args...>> {        // 参数列表的类型列表
  static Ret call(Args... args) {               // 静态成员函数 call，接受参数包 Args...
    c10::impl::ExcludeDispatchKeyGuard no_autocast(  // 创建 ExcludeDispatchKeyGuard 对象，用于禁用自动转换
        get_autocast_dispatch_key_from_device_type(device_type));  // 获取设备类型对应的自动转换分发键
    return (*F)(cached_cast(at::kFloat, args, device_type)...);  // 调用 F 函数指针，将参数列表 args 转换为 float 类型并返回结果
  }
};

// 模板定义：WrapFunction_ 模板特化，处理 CastPolicy::fp32_set_opt_dtype 和 General_DeviceType 的情况
template <
    c10::DeviceType device_type,               // 设备类型参数
    class Redispatch,                         // 重分派类型
    Redispatch* F,                            // 指向重分派函数的指针
    class Ret,                                // 返回类型
    class... Args>                            // 参数包
struct WrapFunction_<
    CastPolicy::fp32_set_opt_dtype,            // 使用 fp32_set_opt_dtype 转换策略
    device_type,                               // 设备类型参数
    Redispatch,                                // 重分派类型
    F,                                         // 指向重分派函数的指针
    Ret,                                       // 返回类型
    guts::typelist::typelist<Args...>> {        // 参数列表的类型列表
  static Ret call(Args... args) {               // 静态成员函数 call，接受参数包 Args...
    c10::impl::ExcludeDispatchKeyGuard no_autocast(  // 创建 ExcludeDispatchKeyGuard 对象，用于禁用自动转换
        get_autocast_dispatch_key_from_device_type(device_type));  // 获取设备类型对应的自动转换分发键
    if (firstarg_is_eligible(device_type, args...)) {  // 如果第一个参数符合条件
      return (*F)(set_opt_dtype(at::kFloat, args)...);  // 调用 F 函数指针，将第一个参数设置为 float 类型并返回结果
    } else {
      // 如果不符合条件，原样调用 F 函数指针，不设置优化数据类型，
      // 因为显式设置优化数据类型可能会干扰内部的隐式提升决策。
      return (*F)(args...);
    }
  }
};

// 模板定义：WrapFunction_ 模板特化，处理 CastPolicy::fp32_append_dtype 和 General_DeviceType 的情况
template <
    c10::DeviceType device_type,               // 设备类型参数
    class Redispatch,                         // 重分派类型
    Redispatch* F,                            // 指向重分派函数的指针
    class Ret,                                // 返回类型
    class... Args>                            // 参数包
struct WrapFunction_<
    CastPolicy::fp32_append_dtype,             // 使用 fp32_append_dtype 转换策略
    device_type,                               // 设备类型参数
    Redispatch,                                // 重分派类型
    F,                                         // 指向重分派函数的指针
    Ret,                                       // 返回类型
    guts::typelist::typelist<Args...>> {        // 参数列表的类型列表
  static Ret call(Args... args) {               // 静态成员函数 call，接受参数包 Args...
    c10::impl::ExcludeDispatchKeyGuard no_autocast(  // 创建 ExcludeDispatchKeyGuard 对象，用于禁用自动转换
        get_autocast_dispatch_key_from_device_type(device_type));  // 获取设备类型对应的自动转换分发键
    at::ScalarType out_type =                   // 根据第一个参数推断输出类型
        type_from_firstarg(device_type, at::kFloat, args...);
    return (*F)(args..., out_type);             // 调用 F 函数指针，将参数列表 args 和推断的输出类型 out_type 返回结果
  }
};

// 模板定义：WrapFunction_ 模板特化，处理 CastPolicy::promote 和 General_DeviceType 的情况
template <
    c10::DeviceType device_type,               // 设备类型参数
    class Redispatch,                         // 重分派类型
    Redispatch* F,                            // 指向重分派函数的指针
    class Ret,                                // 返回类型
    class... Args>                            // 参数包
struct WrapFunction_<
    CastPolicy::promote,                       // 使用 promote 转换策略
    device_type,                               // 设备类型参数
    Redispatch,                                // 重分派类型
    F,                                         // 指向重分派函数的指针
    Ret,                                       // 返回类型
    guts::typelist::typelist<Args...>> {        // 参数列表的类型列表
  static Ret call(Args... args) {               // 静态成员函数 call，接受参数包 Args...
    c10::impl::ExcludeDispatchKeyGuard no_autocast(  // 创建 ExcludeDispatchKeyGuard 对象，用于禁用自动转换
        get_autocast_dispatch_key_from_device_type(device_type));  // 获取设备类型对应的自动转换分发键
    auto to_type = promote_type(               // 根据设备类型和参数推断提升类型
        get_lower_precision_fp_from_device_type(device_type),
        device_type,
        args...);
    return (*F)(cached_cast(to_type, args, device_type)...);  // 调用 F 函数指针，将参数列表 args 转换为推断的类型 to_type 并返回结果
  }
};

// 模板定义：WrapFunction_ 模板，用于推断 WrapFunction_ 的返回类型和参数类型
// 模拟 core/boxing/impl/WrapFunctionIntoFunctor.h
template <
    CastPolicy policy,                         // 转换策略参数
    c10::DeviceType device_type,
    # 定义一个模板类 WrapFunction_
    template <
        class Registered, // 我们要注册的函数签名。调度器的调用代码将使用与 Registered 匹配的参数调用我们注册的函数，
                          // 因此我们使用匹配签名的 WrapFunction_::call 方法注册这些函数，以便正确处理这些参数。
        // guts::function_traits 从 Registered 中提取 return_type 和 parameter_types，
        // 上面的 WrapFunction_ 模板使用这些信息声明它们的 call 方法。
        class Redispatch, // 我们要重新分派到的函数的签名。在大多数情况下，这与 Registered 相同，
                          // 但对于某些操作（例如，对 dtype 追加的操作），重新分派到具有不同签名的函数很有用。
        Redispatch* F> // 我们实际要重新分派的函数。
/*****************************************************************************************************************
这部分执行自动转换包装函数的加载时注册。

有争议的是应该在哪个级别进行补丁。我们希望强制类型转换能够在自动求导曝光之前，并且在自动求导历史记录之前进行，以便对于低精度浮点运算，
输入张量在反向传播时保存为低精度浮点数（而不是fp32）。将输入保存为低精度浮点数可以显著减少模型的内存占用。

选项1（草案）：仅在调用 cudnn/cublas 的显式调用层面进行补丁（如 cudnn_convolution 等），因为这些代码路径保证使用了张量核心，
因此它们将最大程度地从低精度浮点数中受益。潜在的问题：卷积（以及其他操作）被包装在多层 at::* 调用中。如果其中一层记录了自动求导历史，
那么我们就失去了在低精度浮点数中保存输入的机会。

选项2：对 Python 暴露的调用表面进行补丁，确保自动求导历史记录不能在自动转换之前 sneak in。这与 Apex 最接近。

我认为选项2是所有操作的正确答案，而不仅仅是卷积操作。这里实现的就是选项2。
*****************************************************************************************************************/

/********************************************************************************************************************
显式注册用于非就地操作

以下的内容可能会通过代码生成器生成。Ed 说过
> 你最终还是要写出函数定义，我不会试图变得聪明。因此，目前，这些都是从 VariableTypeEverything.cpp 中复制粘贴，并进行了适当的替换。
********************************************************************************************************************/
// 定义宏 KERNEL1，用于注册自动混合精度函数，接受三个参数：DISPATCHKEY、OP、POLICY
#define KERNEL1(DISPATCHKEY, OP, POLICY)      \
  m.impl(                                     \
      TORCH_SELECTIVE_NAME("aten::" #OP),     \
      &::at::autocast::WrapFunction<          \
          ::at::autocast::CastPolicy::POLICY, \
          DISPATCHKEY,                        \
          decltype(ATEN_FN(OP)),              \
          decltype(ATEN_FN(OP)),              \
          &ATEN_FN(OP)>::type::call);

// 定义宏 KERNEL2，用于注册自动混合精度函数，接受四个参数：DISPATCHKEY、OP、OVERLOAD、POLICY
#define KERNEL2(DISPATCHKEY, OP, OVERLOAD, POLICY)      \
  m.impl(                                               \
      TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
      &::at::autocast::WrapFunction<                    \
          ::at::autocast::CastPolicy::POLICY,           \
          DISPATCHKEY,                                  \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          &ATEN_FN2(OP, OVERLOAD)>::type::call);

// 定义宏 _KERNEL_DISPATCH，用于根据参数个数调用对应的 KERNEL 宏
#define _KERNEL_DISPATCH(DISPATCHKEY, NARG, ...) \
  C10_CONCATENATE(KERNEL, NARG)(DISPATCHKEY, __VA_ARGS__)

// 定义宏 _KERNEL_IMPL，根据参数个数调用 _KERNEL_DISPATCH 宏
#define _KERNEL_IMPL(DISPATCHKEY, ...) \
  _KERNEL_DISPATCH(DISPATCHKEY, _KERNEL_OVERLOAD_NARG(__VA_ARGS__), __VA_ARGS__)

// 定义宏 KERNEL，根据参数个数和类型调用不同的 KERNEL 宏
#define KERNEL(DISPATCHKEY, ...) _KERNEL_IMPL(DISPATCHKEY, __VA_ARGS__)

// 定义宏 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE，用于注册带有新签名的函数
#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(      \
    DISPATCHKEY,                                    \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  m.impl(                                           \
      TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
      &::at::autocast::WrapFunction<                \
          ::at::autocast::CastPolicy::POLICY,       \
          DISPATCHKEY,                              \
          REGISTER_SIGNATURE,                       \
          REDISPATCH_SIGNATURE,                     \
          &REDISPATCH_FUNC>::type::call);

// 宏 KERNEL_CPU，用于注册在 CPU 上执行的自动混合精度函数
#define KERNEL_CPU(...) KERNEL(c10::DeviceType::CPU, __VA_ARGS__)

// 宏 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CPU，用于注册在 CPU 上执行的带有新签名的函数
#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CPU( \
    REDISPATCH_FUNC,                               \
    REGISTER_NAME,                                 \
    REGISTER_SIGNATURE,                            \
    REDISPATCH_SIGNATURE,                          \
    POLICY)                                        \
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(           \
    c10::DeviceType::CPU,                          \
    REDISPATCH_FUNC,                               \
    REGISTER_NAME,                                 \
    REGISTER_SIGNATURE,                            \
    REDISPATCH_SIGNATURE,                          \
    POLICY)
    POLICY)                                        \  # 在宏定义中，对应的宏结束括号
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(           \  # 调用宏 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE，该宏跨越多行
      c10::DeviceType::CPU,                        \  # 宏参数：指定设备类型为 CPU
      REDISPATCH_FUNC,                             \  # 宏参数：重新调度的函数名
      REGISTER_NAME,                               \  # 宏参数：注册名称
      REGISTER_SIGNATURE,                          \  # 宏参数：注册签名
      REDISPATCH_SIGNATURE,                        \  # 宏参数：重新调度的签名
      POLICY)                                      # 宏参数：策略名称
// 定义宏 KERNEL_CUDA，用于在注册 AutocastCUDA 时简化操作
#define KERNEL_CUDA(...) KERNEL(c10::DeviceType::CUDA, __VA_ARGS__)

// 定义宏 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA，用于注册 AutocastCUDA 的不同签名函数
#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA( \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  // 调用 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE 宏，设置设备类型为 CUDA
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(            \
      c10::DeviceType::CUDA,                        \
      REDISPATCH_FUNC,                              \
      REGISTER_NAME,                                \
      REGISTER_SIGNATURE,                           \
      REDISPATCH_SIGNATURE,                         \
      POLICY)

// 定义宏 KERNEL_XPU，用于在注册 AutocastXPU 时简化操作
#define KERNEL_XPU(...) KERNEL(c10::DeviceType::XPU, __VA_ARGS__)

// 定义宏 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_XPU，用于注册 AutocastXPU 的不同签名函数
#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_XPU( \
    REDISPATCH_FUNC,                               \
    REGISTER_NAME,                                 \
    REGISTER_SIGNATURE,                            \
    REDISPATCH_SIGNATURE,                          \
    POLICY)                                        \
  // 调用 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE 宏，设置设备类型为 XPU
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(           \
      c10::DeviceType::XPU,                        \
      REDISPATCH_FUNC,                             \
      REGISTER_NAME,                               \
      REGISTER_SIGNATURE,                          \
      REDISPATCH_SIGNATURE,                        \
      POLICY)

// 定义宏 KERNEL_PRIVATEUSEONE，用于在注册 AutocastPrivateUse1 时简化操作
#define KERNEL_PRIVATEUSEONE(...) \
  KERNEL(c10::DeviceType::PrivateUse1, __VA_ARGS__)

// 定义宏 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE，用于注册 AutocastPrivateUse1 的不同签名函数
#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE( \
    REDISPATCH_FUNC,                                         \
    REGISTER_NAME,                                           \
    REGISTER_SIGNATURE,                                      \
    REDISPATCH_SIGNATURE,                                    \
    POLICY)                                                  \
  // 调用 KERNEL_DIFFERENT_REDISPATCH_SIGNATURE 宏，设置设备类型为 PrivateUse1
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(                     \
      c10::DeviceType::PrivateUse1,                          \
      REDISPATCH_FUNC,                                       \
      REGISTER_NAME,                                         \
      REGISTER_SIGNATURE,                                    \
      REDISPATCH_SIGNATURE,                                  \
      POLICY)
# 定义一个宏，用于批量生成具有特定前缀的函数名
#define AT_FORALL_LOWER_PRECISION_FP(_)  \
  _(convolution, deprecated)            \  # 带有 "deprecated" 标记的 convolution 函数
  _(convolution)                        \  # 普通的 convolution 函数
  _(conv1d)                              \  # 一维卷积函数
  _(conv2d)                              \  # 二维卷积函数
  _(conv3d)                              \  # 三维卷积函数
  _(conv_tbc)                            \  # 时间步卷积函数
  _(conv_transpose1d)                    \  # 一维转置卷积函数
  _(conv_transpose2d, input)             \  # 二维输入转置卷积函数
  _(conv_transpose3d, input)             \  # 三维输入转置卷积函数
  _(convolution)                         \  # 卷积函数
  _(prelu)                               \  # PReLU 激活函数
  _(addmm)                               \  # 矩阵相加乘函数
  _(addmv)                               \  # 矩阵-向量相加乘函数
  _(addr)                                \  # 向量外积相加函数
  _(matmul)                              \  # 矩阵乘法函数
  _(einsum)                              \  # Einstein Summation Notation 函数
  _(mm)                                  \  # 矩阵乘法函数
  _(mv)                                  \  # 矩阵-向量乘法函数
  _(linalg_vecdot)                       \  # 线性代数向量点积函数
  _(linear)                              \  # 线性变换函数
  _(addbmm)                              \  # 批量矩阵相加乘函数
  _(baddbmm)                             \  # 批量加法及矩阵相加乘函数
  _(bmm)                                 \  # 批量矩阵乘法函数
  _(chain_matmul)                        \  # 链式矩阵乘法函数
  _(linalg_multi_dot)                    \  # 线性代数多点乘函数
  _(_thnn_fused_lstm_cell)               \  # 融合 LSTM 单元函数
  _(_thnn_fused_gru_cell)                \  # 融合 GRU 单元函数
  _(lstm_cell)                           \  # LSTM 单元函数
  _(gru_cell)                            \  # GRU 单元函数
  _(rnn_tanh_cell)                       \  # RNN tanh 单元函数
  _(rnn_relu_cell)                       \  # RNN relu 单元函数
  _(_scaled_dot_product_flash_attention) \  # 缩放点积闪存注意力函数
  _(scaled_dot_product_attention)        # 缩放点积注意力函数
# 定义宏 AT_FORALL_FP32，用于展开一系列函数及其参数的列表
#define AT_FORALL_FP32(_)             \
  _(acos)                             \  # 宏展开：acos
  _(asin)                             \  # 宏展开：asin
  _(cosh)                             \  # 宏展开：cosh
  _(erfinv)                           \  # 宏展开：erfinv
  _(exp)                              \  # 宏展开：exp
  _(expm1)                            \  # 宏展开：expm1
  _(log)                              \  # 宏展开：log
  _(log10)                            \  # 宏展开：log10
  _(log2)                             \  # 宏展开：log2
  _(log1p)                            \  # 宏展开：log1p
  _(reciprocal)                       \  # 宏展开：reciprocal
  _(rsqrt)                            \  # 宏展开：rsqrt
  _(sinh)                             \  # 宏展开：sinh
  _(tan)                              \  # 宏展开：tan
  _(pow, Tensor_Scalar)               \  # 宏展开：pow, 接受一个张量和一个标量作为参数
  _(pow, Tensor_Tensor)               \  # 宏展开：pow, 接受两个张量作为参数
  _(pow, Scalar)                      \  # 宏展开：pow, 接受一个标量作为参数
  _(softplus)                         \  # 宏展开：softplus
  _(layer_norm)                       \  # 宏展开：layer_norm
  _(native_layer_norm)                \  # 宏展开：native_layer_norm
  _(group_norm)                       \  # 宏展开：group_norm
  _(frobenius_norm, dim)              \  # 宏展开：frobenius_norm, 接受一个维度参数
  _(nuclear_norm)                     \  # 宏展开：nuclear_norm
  _(nuclear_norm, dim)                \  # 宏展开：nuclear_norm, 接受一个维度参数
  _(cosine_similarity)                \  # 宏展开：cosine_similarity
  _(poisson_nll_loss)                 \  # 宏展开：poisson_nll_loss
  _(cosine_embedding_loss)            \  # 宏展开：cosine_embedding_loss
  _(nll_loss)                         \  # 宏展开：nll_loss
  _(nll_loss2d)                       \  # 宏展开：nll_loss2d
  _(hinge_embedding_loss)             \  # 宏展开：hinge_embedding_loss
  _(kl_div)                           \  # 宏展开：kl_div
  _(l1_loss)                          \  # 宏展开：l1_loss
  _(smooth_l1_loss)                   \  # 宏展开：smooth_l1_loss
  _(huber_loss)                       \  # 宏展开：huber_loss
  _(mse_loss)                         \  # 宏展开：mse_loss
  _(margin_ranking_loss)              \  # 宏展开：margin_ranking_loss
  _(multilabel_margin_loss)           \  # 宏展开：multilabel_margin_loss
  _(soft_margin_loss)                 \  # 宏展开：soft_margin_loss
  _(triplet_margin_loss)              \  # 宏展开：triplet_margin_loss
  _(multi_margin_loss)                \  # 宏展开：multi_margin_loss
  _(binary_cross_entropy_with_logits) \  # 宏展开：binary_cross_entropy_with_logits
  _(dist)                             \  # 宏展开：dist
  _(pdist)                            \  # 宏展开：pdist
  _(cdist)                            \  # 宏展开：cdist
  _(renorm)                           \  # 宏展开：renorm
  _(logsumexp)                        \  # 宏展开：logsumexp
  _(upsample_nearest1d)               \  # 宏展开：upsample_nearest1d
  _(_upsample_nearest_exact1d)        \  # 宏展开：_upsample_nearest_exact1d
  _(upsample_nearest2d)               \  # 宏展开：upsample_nearest2d
  _(_upsample_nearest_exact2d)        \  # 宏展开：_upsample_nearest_exact2d
  _(upsample_nearest3d)               \  # 宏展开：upsample_nearest3d
  _(_upsample_nearest_exact3d)        \  # 宏展开：_upsample_nearest_exact3d
  _(upsample_linear1d)                \  # 宏展开：upsample_linear1d
  _(upsample_bilinear2d)              \  # 宏展开：upsample_bilinear2d
  _(_upsample_bilinear2d_aa)          \  # 宏展开：_upsample_bilinear2d_aa
  _(upsample_trilinear3d)             \  # 宏展开：upsample_trilinear3d
  _(upsample_bicubic2d)               \  # 宏展开：upsample_bicubic2d
  _(_upsample_bicubic2d_aa)           \  # 宏展开：_upsample_bicubic2d_aa
#define AT_FORALL_FP32_SET_OPT_DTYPE(_) \
  _(prod)                               \  // 定义针对浮点数的操作，使用默认数据类型
  _(prod, dim_int)                      \  // 定义针对浮点数的操作，支持整数维度
  _(prod, dim_Dimname)                  \  // 定义针对浮点数的操作，支持维度名称
  _(softmax, int)                       \  // 定义 softmax 操作，接受整数参数
  _(softmax, Dimname)                   \  // 定义 softmax 操作，接受维度名称参数
  _(log_softmax, int)                   \  // 定义 log_softmax 操作，接受整数参数
  _(log_softmax, Dimname)               \  // 定义 log_softmax 操作，接受维度名称参数
  _(cumprod)                            \  // 定义累积乘积操作
  _(cumprod, dimname)                   \  // 定义带有维度名称的累积乘积操作
  _(cumsum)                             \  // 定义累积求和操作
  _(cumsum, dimname)                    \  // 定义带有维度名称的累积求和操作
  _(linalg_vector_norm)                 \  // 定义向量范数计算操作
  _(linalg_matrix_norm)                 \  // 定义矩阵范数计算操作
  _(linalg_matrix_norm, str_ord)        \  // 定义带有字符串排序参数的矩阵范数计算操作
  _(sum)                                \  // 定义求和操作
  _(sum, dim_IntList)                   \  // 定义带有整数列表维度参数的求和操作
  _(sum, dim_DimnameList)               // 定义带有维度名称列表参数的求和操作

#define AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(_)                         \
  _(ADD_NS(norm),                                                           \  // 定义不同的重新分发签名
    "norm.Scalar",                                                          \  // norm 操作，接受标量参数
    Tensor(const Tensor&, const Scalar&),                                   \  // 签名：Tensor(const Tensor&, const Scalar&)
    Tensor(const Tensor&, const std::optional<Scalar>&, ScalarType),        \  // 签名：Tensor(const Tensor&, const std::optional<Scalar>&, ScalarType)
    fp32_append_dtype)                                                      // 追加浮点数据类型
  _(ADD_NS(norm),                                                           \  // 定义不同的重新分发签名
    "norm.ScalarOpt_dim",                                                   \  // norm 操作，接受标量和可选维度参数
    Tensor(const Tensor&, const std::optional<Scalar>&, IntArrayRef, bool), \  // 签名：Tensor(const Tensor&, const std::optional<Scalar>&, IntArrayRef, bool)
    Tensor(                                                                 \  // 签名：Tensor(const Tensor&, const std::optional<Scalar>&, IntArrayRef, bool, ScalarType)
        const Tensor&,                                                      \
        const std::optional<Scalar>&,                                       \
        IntArrayRef,                                                        \
        bool,                                                               \
        ScalarType),                                                        \
    fp32_append_dtype)                                                      // 追加浮点数据类型
  _(ADD_NS(norm),                                                           \  // 定义不同的重新分发签名
    "norm.names_ScalarOpt_dim",                                             \  // norm 操作，接受标量、维度名称列表和布尔值参数
    Tensor(const Tensor&, const std::optional<Scalar>&, DimnameList, bool), \  // 签名：Tensor(const Tensor&, const std::optional<Scalar>&, DimnameList, bool)
    Tensor(                                                                 \  // 签名：Tensor(const Tensor&, const std::optional<Scalar>&, DimnameList, bool, ScalarType)
        const Tensor&,                                                      \
        const std::optional<Scalar>&,                                       \
        DimnameList,                                                        \
        bool,                                                               \
        ScalarType),                                                        \
    fp32_append_dtype)                                                      // 追加浮点数据类型
// 定义一个宏，用于展开多个函数名，每个函数名以下划线开头
#define AT_FORALL_PROMOTE(_) \
  _(addcdiv)                 \
  _(addcmul)                 \
  _(atan2)                   \
  _(bilinear)                \
  _(cross)                   \
  _(dot)                     \
  _(vdot)                    \
  _(grid_sampler)            \
  _(index_put)               \
  _(tensordot)               \
  _(scatter_add)
```