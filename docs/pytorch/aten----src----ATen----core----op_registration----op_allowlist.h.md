# `.\pytorch\aten\src\ATen\core\op_registration\op_allowlist.h`

```py
#pragma once
// TODO: 统一为 C10_MOBILE。理论上此头文件可用于 OSS。

#ifdef TEMPLATE_SELECTIVE_BUILD
#include <ATen/selected_mobile_ops.h>
#endif

/**
 * 此头文件实现了仅包含特定运算符（及其依赖项）构建 PyTorch 的功能。
 *
 * - 使用 -DTORCH_OPERATOR_WHITELIST="aten::add;aten::sub" 进行构建，将只包含这两个操作符及其依赖项。
 *   允许列表仅记录操作符，不包括重载；例如，包括 aten::add 将会包含所有的 aten::add 重载。
 *
 * 内部实现通过在编译时移除操作符注册调用来实现，链接器将会删除所有未注册的操作符函数。
 * 更多详情请参见注释 [Selective build]。
 *
 * 警告：允许列表机制并不适用于所有可能的操作符注册方式。如果调度键 / 操作符名称在编译时不够明确，
 * 则允许列表机制将失败（并且操作符将被包含在二进制文件中）。
 */

#include <c10/util/string_view.h>
#include <c10/core/DispatchKey.h>
#include <c10/macros/Macros.h>

#if defined(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE)
#include <ATen/record_function.h>
#endif

namespace c10 {

namespace impl {

// 声明 constexpr 函数 allowlist_contains，用于检查允许列表中是否包含特定的条目
constexpr bool allowlist_contains(string_view allowlist, string_view item);  // Forward Declare

/**
 * 在选择性构建模式下，根据构建特性的可用性返回 true/false。
 *
 * 在仪器化模式（跟踪模式）下，始终返回 true，并且不会触发任何副作用。
 */
constexpr bool is_build_feature_available(const char* name) {
#if !defined(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE)
  // 选择性构建模式。
#if !defined(TORCH_BUILD_FEATURE_ALLOWLIST)
  (void)name;
  return true;
#else
  return allowlist_contains(
    C10_STRINGIZE(TORCH_BUILD_FEATURE_ALLOWLIST),
    name);
#endif

#else
  // 仪器化模式。
  (void)name;
  return true;
#endif
}

/**
 * 在用户代码中使用 BUILD_FEATURE_REQUIRED 宏。
 *
 * 在选择性构建模式下，如果构建特性可用，则成为一个空操作。如果不可用，则抛出异常（c10::Error）。
 * 如果构建特性不可用，编译器能够对此方法后面的代码进行死代码消除。
 *
 * 在仪器化模式（跟踪模式）下，注册（作为副作用）此特定构建特性被触发的情况。
 */
#if !defined(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE)  // 选择性构建模式

#if defined(TORCH_BUILD_FEATURE_ALLOWLIST)
#define BUILD_FEATURE_REQUIRED(NAME)                                 \
  if (!c10::impl::is_build_feature_available(NAME)) {                \
    ::c10::impl::build_feature_required_feature_not_available(NAME); \
  }
#else  // 一切都被简单地选择了
// 定义宏 BUILD_FEATURE_REQUIRED，不带参数时为空
#define BUILD_FEATURE_REQUIRED(NAME)

#endif

// 如果不处于 trace 模式，定义宏 BUILD_FEATURE_REQUIRED(NAME)，展开为一个记录函数调用的宏
#else  // trace mode
#define BUILD_FEATURE_REQUIRED(NAME)  \
  RECORD_FUNCTION_WITH_SCOPE(         \
      at::RecordScope::BUILD_FEATURE, \
      std::string(NAME),              \
      {});
#endif

// 使用此宏来检查构建特性是否可用，而不是直接调用 is_build_feature_available
// NAME 为要检查的特性名
#define BUILD_FEATURE_AVAILABLE(NAME) ::c10::impl::is_build_feature_available(NAME)

// 返回 true 如果 allowlist 包含指定的 item
// 示例：allowlist_contains("a;bc;d", "bc") == true
constexpr bool allowlist_contains(string_view allowlist, string_view item) {
    // 选择一个非常大的值，以便如果出现问题，这段代码将以一种可检测的方式失败。
    size_t next = std::numeric_limits<size_t>::max();
    for (size_t cur = 0; cur <= allowlist.size(); cur = next) {
      // 查找分号的位置，以分割 allowlist 字符串
      next = allowlist.find(';', cur);
      if (next != string_view::npos) {
        // 如果找到了分号，比较当前子字符串是否等于 item
        if (allowlist.substr(cur, next - cur).compare(item) == 0) {
          return true;
        }
        next++;
      } else {
        // 如果未找到分号，直接比较剩余的子字符串是否等于 item
        if (allowlist.substr(cur).compare(item) == 0) {
          return true;
        }
        break;
      }
    }
    return false;
}

// 检查操作名是否在 allowlist 中，并应注册
constexpr bool op_allowlist_check(string_view op_name) {
  // 断言操作名中包含 "::"，确保是符合预期的操作名格式
  assert(op_name.find("::") != string_view::npos);
  // 由于 gcc 的一个 bug，使用 assert() 而不是 throw()，参考链接详细信息
  // https://stackoverflow.com/questions/34280729/throw-in-constexpr-function
  // https://github.com/fmtlib/fmt/issues/682
  assert(op_name.find("(") == string_view::npos);
  
  // 如果未定义 TORCH_OPERATOR_WHITELIST 参数，则所有操作都应注册
#if !defined(TORCH_OPERATOR_WHITELIST)
  return true;
#else
  // 否则，检查操作名是否在 TORCH_OPERATOR_WHITELIST 中
  return allowlist_contains(
    C10_STRINGIZE(TORCH_OPERATOR_WHITELIST),
    op_name);
#endif
}

// 检查模式字符串是否在 allowlist 中，并应注册
constexpr bool schema_allowlist_check(string_view schema) {
  // 如果定义了 TORCH_FORCE_SCHEMA_REGISTRATION 参数，则返回 true
#if defined(TORCH_FORCE_SCHEMA_REGISTRATION)
  return true;
#else
  // 否则，检查操作名是否在 allowlist 中
  return op_allowlist_check(schema.substr(0, schema.find("(")));
#endif
}

// 检查自定义类名是否在 allowlist 中，并应注册
constexpr bool custom_class_allowlist_check(string_view custom_class_name) {
  // 如果未定义 TORCH_CUSTOM_CLASS_ALLOWLIST 参数，则所有自定义类都应注册
#if !defined(TORCH_CUSTOM_CLASS_ALLOWLIST)
  // 使用 (void)custom_class_name 来避免未使用参数的警告
  (void)custom_class_name;
  return true;
#else
  // 否则，检查自定义类名是否在 TORCH_CUSTOM_CLASS_ALLOWLIST 中
  return allowlist_contains(
    C10_STRINGIZE(TORCH_CUSTOM_CLASS_ALLOWLIST),
    custom_class_name);
#endif
}
// 检查 allowlist 是否包含 schema 中的名称，schema 格式通常为函数名加参数列表，需要去掉参数部分进行比较
constexpr bool op_allowlist_contains_name_in_schema(string_view allowlist, string_view schema) {
  return allowlist_contains(allowlist, schema.substr(0, schema.find("(")));
}

// 检查给定的调度键是否在 allowlist 中，确定是否应该注册。在移动端情况下，这个列表是硬编码的，
// 但需要确保为该平台正确设置了调度键的集合。
constexpr bool dispatch_key_allowlist_check(DispatchKey /*k*/) {
#ifdef C10_MOBILE
  return true;
  // 暂时禁用：稍后启用！
  // return k == DispatchKey::CPU || k == DispatchKey::Vulkan || k == DispatchKey::QuantizedCPU || k == DispatchKey::BackendSelect || k == DispatchKey::CatchAll;
#else
  return true;
#endif
}

} // namespace impl
} // namespace c10
```