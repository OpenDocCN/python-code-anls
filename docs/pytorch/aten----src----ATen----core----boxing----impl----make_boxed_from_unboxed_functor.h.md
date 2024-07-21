# `.\pytorch\aten\src\ATen\core\boxing\impl\make_boxed_from_unboxed_functor.h`

```
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <ATen/core/boxing/OperatorKernel.h>
// 包含 OperatorKernel 头文件，用于定义操作核心

#include <ATen/core/ivalue.h>
// 包含 IValue 头文件，提供对 Torch 脚本 IValue 类型的支持

#include <ATen/core/stack.h>
// 包含 stack 头文件，定义了与 Torch 脚本的栈交互的相关函数和类

#include <c10/util/TypeList.h>
// 包含 TypeList 头文件，提供类型列表的支持

#include <ATen/core/IListRef.h>
// 包含 IListRef 头文件，定义了对列表引用的支持

#include <c10/util/intrusive_ptr.h>
// 包含 intrusive_ptr 头文件，提供了对侵入式指针的支持

#include <c10/util/Metaprogramming.h>
// 包含 Metaprogramming 头文件，提供了元编程支持

#include <utility>
// 包含 utility 头文件，提供了各种实用工具函数的支持

namespace c10 {

using Stack = torch::jit::Stack;
// 定义别名 Stack，使 torch::jit::Stack 可以通过 c10::Stack 访问

class OperatorHandle;
// 声明 OperatorHandle 类，但未提供具体定义

namespace impl {

// supported_primitive_arg_types 定义了我们在核函数中允许作为参数或返回值的原始类型列表。
using supported_primitive_arg_types = guts::typelist::typelist<
  int64_t,
  double,
  bool,
  c10::string_view,
  at::Tensor,
  at::Scalar,
  c10::QScheme,
  c10::ScalarType,
  c10::Device,
  c10::DeviceIndex,
  c10::Layout,
  c10::MemoryFormat,
  at::Dimname
>;

// We have an unboxed functor in hand that takes C++ arguments, and
// we're building a boxed functor wrapper for it that takes IValues.
// So "outside" is boxed and "inside" is unboxed.
//
// So a valid input type is one that our boxed functor wrapper can
// unbox from an IValue into a C++ value.
//
// Whereas a valid output type is one that our wrapper can recieve
// as a C++ value from the unboxed functor, and box into an IValue.

//
// assert_is_valid_input_type
// checks that T can be unboxed from an IValue into a C++ value.
//

// assert_is_valid_input_type 模板结构体用于验证类型 T 是否可以从 IValue 解包成 C++ 值。
template<class T, bool AllowDeprecatedTypes, class Enable = void>
struct assert_is_valid_input_type {
  assert_is_valid_input_type() {
    // 如果 T 包含在 supported_primitive_arg_types 中，认为一切正常，因为它是一个原始类型。
    if constexpr (guts::typelist::contains<supported_primitive_arg_types, T>::value) {
      /* everything is ok, this is a primitive type */
    } else {
      /* otherwise this must be an instance of a valid custom class, since it can only
         have been created via IValue(x), which ensures this. */
      // 否则，T 必须是一个有效自定义类的实例，因为它只能通过 IValue(x) 创建。
    }
  }
};

// 对于 std::optional<T>，验证 T 是否是有效的输入类型。
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_input_type<std::optional<T>, AllowDeprecatedTypes>
: assert_is_valid_input_type<T, AllowDeprecatedTypes> {};

// TypeCheckHelper 模板结构体辅助于检查多个类型参数是否都是有效的输入类型。
template <bool AllowDeprecatedTypes, class... Args>
struct TypeCheckHelper;

// TypeCheckHelper 的偏特化版本，当没有类型参数时，不做任何操作。
template <bool AllowDeprecatedTypes>
struct TypeCheckHelper<AllowDeprecatedTypes> {};

// TypeCheckHelper 的偏特化版本，处理第一个类型参数 Head，并递归处理其余类型参数 Rest。
template <bool AllowDeprecatedTypes, class Head, class... Rest>
struct TypeCheckHelper<AllowDeprecatedTypes, Head, Rest...>
: TypeCheckHelper<AllowDeprecatedTypes, Rest...> {
  assert_is_valid_input_type<Head, AllowDeprecatedTypes> check;
};

// 对于 std::tuple<Contained...>，验证其中的每个元素是否是有效的输入类型。
template<class... Contained, bool AllowDeprecatedTypes>
struct assert_is_valid_input_type<std::tuple<Contained...>, AllowDeprecatedTypes>
: TypeCheckHelper<AllowDeprecatedTypes, Contained...> {};

// 对于 Dict<Key, Value>，验证 Value 是否是有效的输入类型。
template<class Key, class Value, bool AllowDeprecatedTypes>
struct assert_is_valid_input_type<Dict<Key, Value>, AllowDeprecatedTypes>
: assert_is_valid_input_type<Value, AllowDeprecatedTypes> {
    // 使用 static_assert 确保 Key 类型在 impl::valid_dict_key_types 中定义，否则给出错误信息
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
      "You tried to register a kernel with an unsupported input type: Dict<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
    };
    
    // 模板特化，验证 std::unordered_map<Key, Value> 的有效性，若 AllowDeprecatedTypes 为 false，则给出错误信息
    template<class Key, class Value, bool AllowDeprecatedTypes>
    struct assert_is_valid_input_type<std::unordered_map<Key, Value>, AllowDeprecatedTypes>
      : assert_is_valid_input_type<Value, AllowDeprecatedTypes> {
      static_assert(AllowDeprecatedTypes,
        "You tried to register a kernel with an unsupported input type: std::unordered_map<Key, Value>. Please use Dict<Key, Value> instead.");
      // 使用 static_assert 确保 Key 类型在 impl::valid_dict_key_types 中定义，否则给出错误信息
      static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
        "You tried to register a kernel with an unsupported input type: std::unordered_map<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
    };
    
    // 模板特化，验证 List<T> 类型的有效性，确保 T 不是 at::Scalar 类型，否则给出错误信息
    template<class T, bool AllowDeprecatedTypes>
    struct assert_is_valid_input_type<List<T>, AllowDeprecatedTypes>
      : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
      static_assert(!std::is_same<T, at::Scalar>::value,
        "You tried to register a kernel with an unsupported input type: List<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    };
    
    // 模板特化，验证 c10::ArrayRef<T> 类型的有效性，确保 T 不是 at::Scalar 类型，否则给出错误信息
    template<class T, bool AllowDeprecatedTypes>
    struct assert_is_valid_input_type<c10::ArrayRef<T>, AllowDeprecatedTypes>
      : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
      static_assert(!std::is_same<T, at::Scalar>::value,
        "You tried to register a kernel with an unsupported input type: ArrayRef<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    };
    
    // 模板特化，验证 c10::OptionalArrayRef<T> 类型的有效性，确保 T 不是 at::Scalar 类型，否则给出错误信息
    template<class T, bool AllowDeprecatedTypes>
    struct assert_is_valid_input_type<c10::OptionalArrayRef<T>, AllowDeprecatedTypes>
      : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
      static_assert(!std::is_same<T, at::Scalar>::value,
        "You tried to register a kernel with an unsupported input type: OptionalArrayRef<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    };
    
    // 模板特化，验证 std::array<T, N> 类型的有效性，确保 T 不是 at::Scalar 类型，否则给出错误信息
    template<class T, size_t N, bool AllowDeprecatedTypes>
    struct assert_is_valid_input_type<std::array<T, N>, AllowDeprecatedTypes>
      : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
      static_assert(!std::is_same<T, at::Scalar>::value,
        "You tried to register a kernel with an unsupported input type: std::array<Scalar, N>. Please use std::array<int64_t, N> instead.");
    };
    
    // 模板特化，当 T 是 float 类型时，给出警告信息，建议使用 double 以保持 API 简洁
    template<class T, bool AllowDeprecatedTypes>
    struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<float, T>::value>> {
      // There is no reason to support float when we have double. Keep the API lean.
    };
  // 使用 static_assert 断言，如果 T 不是支持的输入类型，编译器会报错，显示相应错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel with an unsupported input type: float. Please use double instead; you should use `double` in the C++ function signature and `float` in the schema string.");
};

// 模板结构体，验证输入类型是否合法，特化版本针对 const char* 类型的处理
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<const char*, T>::value>> {
  // 使用 static_assert 断言，如果 T 是 const char* 类型，编译器会报错，显示相应错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel with an unsupported input type: const char*. Please use c10::string_view instead.");
};

// 模板结构体，验证输入类型是否合法，特化版本针对 std::vector<bool> 类型的处理
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<std::vector<bool>, T>::value>> {
  // 使用 static_assert 断言，如果 T 是 std::vector<bool> 类型，编译器会报错，显示相应错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel with an unsupported input type: vector<bool>. Please use List<bool> instead.");
};

// 模板结构体，验证输入类型是否合法，特化版本针对整数类型且不在支持列表中的处理
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
  // 使用 static_assert 断言，如果 T 是不支持的整数类型，编译器会报错，显示相应错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel with an unsupported integral input type. Please use int64_t instead; you should use `int64_t` in the C++ function signature and `int` in the schema string.");
};

// 模板结构体，验证输入类型是否合法，特化版本针对 const c10::SymInt& 类型的处理
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<const c10::SymInt&, T>::value>> {
  // 使用 static_assert 断言，如果 T 是 const c10::SymInt& 类型，编译器会报错，显示相应错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel taking c10::SymInt by reference. Please accept it by value instead.");
};

// TODO: 可能需要更严格地使用显式列表处理所有情况

//
// assert_is_valid_output_type
//

// 模板结构体，验证输出类型是否合法，默认版本处理非特化情况
template<class T, bool AllowDeprecatedTypes, class Enable = void>
struct assert_is_valid_output_type {
  // 构造函数，根据条件编译，如果 T 是支持的原始类型，则什么都不做；否则提示 T 应该是已注册的自定义类
  assert_is_valid_output_type() {
    if constexpr(guts::typelist::contains<supported_primitive_arg_types, T>::value) {
      /* everything is ok, this is a primitive type */
    } else {
      /* otherwise T is verified to be a registered custom class in the IValue
        constructor, so no benefit in double-checking here */
    }
  }
};

// 模板结构体，验证输出类型是否合法，特化版本处理 std::optional<T> 类型
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_output_type<std::optional<T>, AllowDeprecatedTypes>
: assert_is_valid_output_type<T, AllowDeprecatedTypes> {};

// 模板结构体，验证输出类型是否合法，特化版本处理 c10::OptionalArrayRef<T> 类型
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_output_type<c10::OptionalArrayRef<T>, AllowDeprecatedTypes>
: assert_is_valid_output_type<T, AllowDeprecatedTypes> {};

// 模板结构体，验证输出类型是否合法，特化版本处理字典类型 Dict<Key, Value>
template<class Key, class Value, bool AllowDeprecatedTypes>
struct assert_is_valid_output_type<Dict<Key, Value>, AllowDeprecatedTypes>
: assert_is_valid_output_type<Value, AllowDeprecatedTypes> {
    // 使用 static_assert 确保 Key 类型在 valid_dict_key_types 列表中，否则会触发编译时错误信息
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
      "You tried to register a kernel with an unsupported output type: Dict<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
    // 使用 static_assert 确保 Value 类型不是 at::Scalar 类型，否则会触发编译时错误信息
    static_assert(!std::is_same<Value, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: Dict<Key, Scalar>. Please use Dict<Key, int64_t> or Dict<Key, double>.");
  };

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<std::unordered_map<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<Value, AllowDeprecatedTypes> {
    // 使用 static_assert 确保 AllowDeprecatedTypes 为真，否则会触发编译时错误信息
    static_assert(AllowDeprecatedTypes,
      "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Value>. Please use Dict<Key, Value> instead.");
    // 使用 static_assert 确保 Key 类型在 valid_dict_key_types 列表中，否则会触发编译时错误信息
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
      "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
    // 使用 static_assert 确保 Value 类型不是 at::Scalar 类型，否则会触发编译时错误信息
    static_assert(!std::is_same<Value, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: std::unordered_map<Key, Scalar>. Please use Dict<Key, int64_t> or Dict<Key, double>.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<List<T>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {
    // 使用 static_assert 确保 T 类型不是 at::Scalar 类型，否则会触发编译时错误信息
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: List<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<std::vector<T>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {
    // 使用 static_assert 确保 T 类型不是 at::Scalar 类型，否则会触发编译时错误信息
    static_assert(!std::is_same<T, at::Scalar>::value,
      "You tried to register a kernel with an unsupported output type: std::vector<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    // TODO 使用 static_assert 确保 AllowDeprecatedTypes 为真，否则会触发编译时错误信息
    // TODO static_assert(AllowDeprecatedTypes, "You tried to register a kernel with an unsupported output type: std::vector<T>. Please use List<T> instead.");
  };

  template<class T, size_t N, bool AllowDeprecatedTypes>
  struct assert_is_valid_output_type<std::array<T, N>, AllowDeprecatedTypes>
  : assert_is_valid_output_type<T, AllowDeprecatedTypes> {
  // 确保模板参数 T 不是 at::Scalar 类型，否则静态断言失败并显示错误信息
  static_assert(!std::is_same<T, at::Scalar>::value,
    "You tried to register a kernel with an unsupported output type: std::array<Scalar, N>. Please use std::array<int64_t, N> instead.");
};

// 以下的 assert_is_valid_output_type 的特化版本在技术上并非必需，因为如果它们不存在，
// 我们会命中基础情况并显示错误消息；但在某些常见错误情况下，我们可以显示更好的错误消息。

// 当 T 为 float 类型时的特化版本
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<float, T>::value>> {
  // 当尝试注册一个不支持的输出类型 float 时，静态断言失败并显示错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel with an unsupported output type: float. Please use double instead; you should use `double` in the C++ function signature and `float` in the schema string.");
};

// 当 T 为 const char* 类型时的特化版本
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<const char*, T>::value>> {
  // 当尝试注册一个不支持的输出类型 const char* 时，静态断言失败并显示错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel with an unsupported output type: const char*. Please use c10::string_view instead.");
};

// 当 T 为 std::vector<bool> 类型时的特化版本
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<std::vector<bool>, T>::value>> {
  // 当尝试注册一个不支持的输出类型 std::vector<bool> 时，静态断言失败并显示错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel with an unsupported output type: vector<bool>. Please use List<bool> instead.");
};

// 当 T 为整数类型且不在 supported_primitive_arg_types 中时的特化版本
template<class T, bool AllowDeprecatedTypes>
struct assert_is_valid_output_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
  // 当尝试注册一个不支持的整数输出类型时，静态断言失败并显示错误信息
  static_assert(guts::false_t<T>::value,
    "You tried to register a kernel with an unsupported integral output type. Please use int64_t instead; you should use `int64_t` in the C++ function signature and `int` in the schema string.");
};

// ivalue_to_arg

// 对于非 Tensor 类型的 T，将其类型解开到 std::decay_t<T>
template<class T>
struct decay_if_not_tensor final {
  using type = std::decay_t<T>;
};

// 对于 at::Tensor& 类型的特化版本，保持类型为 at::Tensor&
template<>
struct decay_if_not_tensor<at::Tensor&> final {
  using type = at::Tensor&;
};

// 对于 const at::Tensor& 类型的特化版本，保持类型为 const at::Tensor&
template<>
struct decay_if_not_tensor<const at::Tensor&> final {
  using type = const at::Tensor&;
};

// ivalue_to_arg 的主模板
template<class T, bool AllowDeprecatedTypes>
struct ivalue_to_arg final {
  // 调用函数，确保输入类型 T 有效，并将 IValue 类型 v 转换为 T 类型并返回结果
  static decltype(auto) call(IValue& v) {
    assert_is_valid_input_type<T, AllowDeprecatedTypes>();
    return std::move(v).to<T>();
  }
};

// 以下两个特化版本利用在 IValue 上的特殊化的 `toTensor()` 重载来避免复制。
template<bool AllowDeprecatedTypes>
struct ivalue_to_arg<at::Tensor&, AllowDeprecatedTypes> final {
  // 如果请求了一个 at::Tensor& 类型的参数，我们不能使用默认实现，必须使用

// 以上注释已经涵盖了给定代码的所有部分。
    // 对于 ivalue_to_arg<const at::Tensor&, AllowDeprecatedTypes> 结构模板的特化
    struct ivalue_to_arg<const at::Tensor&, AllowDeprecatedTypes> final {
      // 如果是 const at::Tensor& 类型的参数，直接返回 v 转换为 Tensor 的结果
      static const at::Tensor& call(IValue& v) {
        // 由于返回的是引用，因此不需要进行断言
        return v.toTensor();
      }
    };

    // 对于 ivalue_to_arg<at::ITensorListRef, AllowDeprecatedTypes> 结构模板的特化
    struct ivalue_to_arg<at::ITensorListRef, AllowDeprecatedTypes> final {
      // 如果是 at::ITensorListRef 类型的参数，返回 v 转换为 Tensor 列表的结果
      static List<at::Tensor> call(IValue& v) {
        return v.toTensorList();
      }
    };

    // 对于 ivalue_to_arg<ArrayRef<T>, AllowDeprecatedTypes> 结构模板的特化
    template<class T, bool AllowDeprecatedTypes>
    struct ivalue_to_arg<ArrayRef<T>, AllowDeprecatedTypes> final {
      // 如果是 ArrayRef<T> 类型的参数，将 IValue 转换为 std::vector<T> 并传递给操作符
      static std::vector<T> call(IValue& v) {
        return ivalue_to_arg<std::vector<T>, AllowDeprecatedTypes>::call(v);
      }
    };

    // 对于 ivalue_to_arg<c10::SymIntArrayRef, AllowDeprecatedTypes> 结构模板的特化
    template<bool AllowDeprecatedTypes>
    struct ivalue_to_arg<c10::SymIntArrayRef, AllowDeprecatedTypes> final {
      // 如果是 c10::SymIntArrayRef 类型的参数
      static std::vector<c10::SymInt> call(IValue& v) {
        // 如果 IValue 是 Int 列表类型，则将其转换为 c10::SymInt 类型的列表
        if (v.isIntList()) {
          std::vector<c10::SymInt> r;
          auto src = v.toIntList();
          std::transform(src.begin(), src.end(), std::back_inserter(r), [](int64_t i) { return c10::SymInt(i); });
          return r;
        } else {
          // 否则，将 IValue 转换为 std::vector<c10::SymInt> 类型并返回
          return ivalue_to_arg<std::vector<c10::SymInt>, AllowDeprecatedTypes>::call(v);
        }
      }
    };

    // 对于 ivalue_to_arg<c10::OptionalArray<c10::SymInt>, AllowDeprecatedTypes> 结构模板的特化
    template<bool AllowDeprecatedTypes>
    struct ivalue_to_arg<c10::OptionalArray<c10::SymInt>, AllowDeprecatedTypes> final {
      // 如果是 c10::OptionalArray<c10::SymInt> 类型的参数
      static OptionalArray<c10::SymInt> call(IValue& v) {
        // 如果 IValue 是 Int 列表类型，则将其转换为 c10::SymInt 类型的列表，并构造为 OptionalArray
        if (v.isIntList()) {
          std::vector<c10::SymInt> r;
          auto src = v.toIntList();
          std::transform(src.begin(), src.end(), std::back_inserter(r), [](int64_t i) { return c10::SymInt(i); });
          return OptionalArray<c10::SymInt>(std::move(r));
        } else {
          // 否则，将 IValue 转换为 c10::OptionalArray<c10::SymInt> 类型并返回
          return std::move(v).to<c10::OptionalArray<c10::SymInt>>();
        }
      }
    };

    // 对于 ivalue_to_arg<optional<ArrayRef<T>>, AllowDeprecatedTypes> 结构模板的特化
    template<class T, bool AllowDeprecatedTypes>
    struct ivalue_to_arg<optional<ArrayRef<T>>, AllowDeprecatedTypes> final {
      // 如果是 optional<ArrayRef<T>> 类型的参数
      // 将 IValue 转换为 optional<std::vector<T>> 并传递给操作符
      static OptionalArray<T> call(IValue& v) {
        return ivalue_to_arg<OptionalArray<T>, AllowDeprecatedTypes>::call(v);
      }
    };

    // 对于 ivalue_to_arg<OptionalArrayRef<T>, AllowDeprecatedTypes> 结构模板的特化
    template<class T, bool AllowDeprecatedTypes>
    struct ivalue_to_arg<OptionalArrayRef<T>, AllowDeprecatedTypes> final {
    // 如果参数是 OptionalArrayRef<T>，则将 IValue 转换为 optional<std::vector<T>>，
    // 并将其传递给操作符。OptionalArray<T> 实际上是 optional<std::vector<T>>，
    // 但可以隐式转换为 OptionalArrayRef<T>。
    static OptionalArray<T> call(IValue& v) {
      return ivalue_to_arg<OptionalArray<T>, AllowDeprecatedTypes>::call(v);
    }
  };

  // return_to_ivalue
  // 返回到 IValue 的转换器
  template<class T, bool AllowDeprecatedTypes, class Enable = void>
  struct return_to_ivalue final {};

  // 针对非 Tensor& 类型的特化
  template<class T, bool AllowDeprecatedTypes>
  struct return_to_ivalue<T, AllowDeprecatedTypes, std::enable_if_t<!std::is_same<at::Tensor&, T>::value>> final {
    // 将对象移动到 IValue
    static IValue call(T&& v) {
      assert_is_valid_output_type<T, AllowDeprecatedTypes>();
      return c10::ivalue::from(std::move(v));
    }
    // 复制对象到 IValue
    static IValue copy(const T& v) {
      assert_is_valid_output_type<T, AllowDeprecatedTypes>();
      return IValue(v);
    }
  };

  // 允许内核返回 Tensor& 的特殊情况
  // TODO 一旦内核不再返回 Tensor&，应删除此部分
  template<bool AllowDeprecatedTypes>
  struct return_to_ivalue<at::Tensor&, AllowDeprecatedTypes, void> final {
    // 将 Tensor& 转换为 IValue
    static IValue call(at::Tensor& v) {
      return c10::ivalue::from(v);
    }
    // 复制 Tensor& 到 IValue
    static IValue copy(at::Tensor& v) {
      return IValue(v);
    }
  };

  // wrap_kernel_functor_unboxed_

  // 封装未打包的内核函数对象

  template<class KernelFunctor, class OpSignature>
  struct wrap_kernel_functor_unboxed_ final {};

  // 这个特化是为了没有 DispatchKeySet 类型作为第一个参数的内核函数
  // 这包括没有参数的内核函数。
  template<class KernelFunctor, class ReturnType, class... ParameterTypes>
  struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
      "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<ParameterTypes...>, typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
      "Parameter types mismatch");

    // 为什么 ParameterTypes 不使用 && 可以参见 [Note: Argument forwarding in the dispatcher]
    static ReturnType call(OperatorKernel* functor, DispatchKeySet, ParameterTypes... args) {
      // 将传入的 functor 转换为 KernelFunctor 类型
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      
      // 注意 [通过调度程序传递键 2]
      // 详见注意 [通过调度程序传递键] 的背景说明。
      // 这个函数显式地接收一个 dispatchKeySet 参数，并丢弃它 - 它不会将其转发给注册的内核函数。
      //
      // 这是因为调度程序内部的调用约定，它期望所有注册的内核函数的第一个参数是 DispatchKeySet 类型。
      // 然而，手动编写的大多数内核函数并非如此 - 这个函数用于分离调度程序的调用约定和手动编写内核函数的调用约定。
      
      // 调用实际的 KernelFunctor 对象，将参数 args 以完美转发方式传递给它
      return (*functor_)(std::forward<ParameterTypes>(args)...);
    }
  };

  // 这个特化用于具有 DispatchKeySet 类型作为第一个参数的内核函数
  template<class KernelFunctor, class ReturnType, class... ParameterTypes>
  struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(DispatchKeySet, ParameterTypes...)> final {
    static_assert(std::is_same<ReturnType, typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
      "Return type mismatch");
    static_assert(std::is_same<guts::typelist::typelist<DispatchKeySet, ParameterTypes...>, typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
      "Parameter types mismatch");

    // 请参见 [调度程序中的参数转发注释]，了解为什么 ParameterTypes 没有使用 &&
    static ReturnType call(OperatorKernel* functor, DispatchKeySet dispatchKeySet, ParameterTypes... args) {
      // 将传入的 functor 转换为 KernelFunctor 类型
      KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
      
      // 我们显式地接收一个 dispatchKeySet 参数，并将其转发给注册的内核函数。
      // 详见注意 [通过调度程序传递键 2]。
      
      // 调用实际的 KernelFunctor 对象，将 dispatchKeySet 和参数 args 以完美转发方式传递给它
      return (*functor_)(dispatchKeySet, std::forward<ParameterTypes>(args)...);
    }
  };

  template<class KernelFunctor>
  using wrap_kernel_functor_unboxed = wrap_kernel_functor_unboxed_<KernelFunctor, typename guts::infer_function_traits_t<KernelFunctor>::func_type>;

  // call_functor_with_args_from_stack

  template<class Functor, bool AllowDeprecatedTypes, size_t... ivalue_arg_indices,  typename... ArgTypes>
  std::decay_t<typename guts::infer_function_traits_t<Functor>::return_type>
  call_functor_with_args_from_stack_(OperatorKernel* functor, DispatchKeySet dispatchKeySet, Stack* stack, std::index_sequence<ivalue_arg_indices...>, guts::typelist::typelist<ArgTypes...>*) {
    (void)(stack); // 当 sizeof...(ivalue_arg_indices) == 0 时，这个参数将不被使用，我们必须消除编译器警告。
    
    // 我们显式地过滤掉参数列表中的 DispatchKeySet。
    // 有些内核函数将 DispatchKeySet 作为它们的第一个参数，以便通过调度程序传递键。
    // 返回一个封装了通过 wrap_kernel_functor_unboxed 调用 functor 后返回值的函数模板实例
    template<class Functor, bool AllowDeprecatedTypes>
    std::decay_t<typename guts::infer_function_traits_t<Functor>::return_type>
    call_functor_with_args_from_stack(OperatorKernel* functor, DispatchKeySet dispatchKeySet, Stack* stack) {
      // 使用模板类型推导获取 Functor 的参数类型列表，排除 DispatchKeySet
      using ArgTypes = typename c10::remove_DispatchKeySet_arg_from_func<Functor>::parameter_types;
      constexpr size_t num_ivalue_args = guts::typelist::size<ArgTypes>::value;
      // 调用具体的实现函数 call_functor_with_args_from_stack_，传递模板参数序列作为索引
      return call_functor_with_args_from_stack_<Functor, AllowDeprecatedTypes>(functor, dispatchKeySet, stack, std::make_index_sequence<num_ivalue_args>(), static_cast<ArgTypes*>(nullptr));
    }
    
    // push_outputs
    
    // 定义一个结构模板 push_outputs，用于推送输出到栈
    template<class OutputType, bool AllowDeprecatedTypes>
    struct push_outputs final {
      // 对于单个输出类型 OutputType，调用函数 push，将其转换为 IValue 后推送到栈上
      static void call(OutputType&& output, Stack* stack) {
        torch::jit::push(*stack, return_to_ivalue<OutputType, AllowDeprecatedTypes>::call(std::forward<OutputType>(output)));
      }
      // 对于单个输出类型 OutputType，调用函数 copy，将其复制为 IValue 后推送到栈上
      static void copy(const OutputType& output, Stack* stack) {
        torch::jit::push(*stack, return_to_ivalue<OutputType, AllowDeprecatedTypes>::copy(output));
      }
    };
    
    // 对于多个输出类型组成的元组 std::tuple<OutputTypes...>，特化 push_outputs 结构模板
    template<class... OutputTypes, bool AllowDeprecatedTypes>
    struct push_outputs<std::tuple<OutputTypes...>, AllowDeprecatedTypes> final {
      // 调用函数 call_，将 std::tuple<OutputTypes...> 的每个元素转换为 IValue 后推送到栈上
      static void call(std::tuple<OutputTypes...>&& output, Stack* stack) {
        call_(std::move(output), stack, std::make_index_sequence<sizeof...(OutputTypes)>());
      }
      // 调用函数 copy_，将 std::tuple<OutputTypes...> 的每个元素复制为 IValue 后推送到栈上
      static void copy(const std::tuple<OutputTypes...>& output, Stack* stack) {
        copy_(output, stack, std::make_index_sequence<sizeof...(OutputTypes)>());
      }
    
    private:
      // 辅助函数 call_，将 std::tuple<OutputTypes...> 的每个元素转换为 IValue 后推送到栈上
      template<size_t... indices>
      static void call_(std::tuple<OutputTypes...>&& output, Stack* stack, std::index_sequence<indices...>) {
        // 展开元组并逐个转换为 IValue，然后推送到栈上
        torch::jit::push(*stack, return_to_ivalue<OutputTypes, AllowDeprecatedTypes>::call(std::forward<OutputTypes>(std::get<indices>(output)))...);
      }
    };
    template<size_t... indices>
    // 定义静态函数 copy_，接受一个 output 元组、一个指向堆栈的指针 stack 和一个索引序列 indices
    static void copy_(const std::tuple<OutputTypes...>& output, Stack* stack, std::index_sequence<indices...>) {
      // 将 return_to_ivalue 模板函数应用于 output 元组的每个元素，并将结果推送到堆栈
      torch::jit::push(*stack, return_to_ivalue<OutputTypes, AllowDeprecatedTypes>::copy(std::get<indices>(output))...);
    }
  };
  template<bool AllowDeprecatedTypes>
  // 定义结构体模板 push_outputs，无返回值的特化版本
  struct push_outputs<void, AllowDeprecatedTypes> final {
    // 静态函数 call，参数为 int 和 Stack*，仅作为占位符
    static void call(int /*dummy*/, Stack* /*stack*/) {
    }
    // 静态函数 copy，参数为 int 和 Stack*，仅作为占位符
    static void copy(int /*dummy*/, Stack* /*stack*/) {
    }
  };

  // make_boxed_from_unboxed_functor

  // 定义结构体模板 make_boxed_from_unboxed_functor，接受一个 KernelFunctor 和一个布尔值 AllowDeprecatedTypes
  template<class KernelFunctor, bool AllowDeprecatedTypes>
  struct make_boxed_from_unboxed_functor final {
    // 断言 KernelFunctor 必须是 OperatorKernel 的派生类
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value,
      "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    // 静态函数 call，接受一个 OperatorKernel 指针 functor、一个 OperatorHandle 对象、一个 DispatchKeySet 对象和一个指向堆栈的指针 stack
    static void call(OperatorKernel* functor, const OperatorHandle&, DispatchKeySet dispatchKeySet, Stack* stack) {
      // 推断 KernelFunctor 的返回类型
      using ReturnType = typename guts::infer_function_traits_t<KernelFunctor>::return_type;
      // 移除 KernelFunctor 的参数类型中的 DispatchKeySet 类型，获得 ArgTypes
      using ArgTypes = typename c10::remove_DispatchKeySet_arg_from_func<KernelFunctor>::parameter_types;
      // 判断 KernelFunctor 是否有输出
      constexpr bool has_outputs = !std::is_same<void, ReturnType>::value;
      // 获取 ArgTypes 的参数数量
      constexpr size_t num_inputs = guts::typelist::size<ArgTypes>::value;
      
      // 如果有输出
      if constexpr (has_outputs) {
        // 将 ReturnType 衰减为 ReturnType_，以便如果返回了引用，则实际按值存储它，避免悬挂引用
        using ReturnType_ = ::std::decay_t<ReturnType>;
        // 调用 call_functor_with_args_from_stack 函数，将结果存储在 output 中
        ReturnType_ output = call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, dispatchKeySet, stack);
        // 丢弃堆栈上的 num_inputs 个元素
        torch::jit::drop(*stack, num_inputs);
        // 调用 push_outputs 结构体的 call 函数，将 output 移动到堆栈中
        push_outputs<ReturnType_, AllowDeprecatedTypes>::call(::std::move(output), stack);
      } else {
        // 如果没有输出，直接调用 call_functor_with_args_from_stack 函数，并丢弃堆栈上的 num_inputs 个元素
        call_functor_with_args_from_stack<KernelFunctor, AllowDeprecatedTypes>(functor, dispatchKeySet, stack);
        torch::jit::drop(*stack, num_inputs);
      }
    }
  };
} // 结束 c10 命名空间的定义

} // 结束 impl 命名空间的定义

// 开始 torch 命名空间的定义，并引入 c10::OperatorKernel 别名
namespace torch {
  using OperatorKernel = c10::OperatorKernel;
}
```