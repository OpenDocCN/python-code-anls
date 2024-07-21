# `.\pytorch\torch\custom_class_detail.h`

```py
#pragma once

#include <ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h>
#include <ATen/core/function.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/irange.h>

// 声明命名空间 torch
namespace torch {

// 声明命名空间 detail，用于包含内部实现细节
namespace detail {

/**
 * 如果在构建追踪器二进制文件时传入 -c pt.enable_record_kernel_dtype=1，
 * 则启用此宏，在 Facebook 内部构建中生效。
 */
#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
// 声明记录自定义类被加载的函数
TORCH_API void record_custom_class(std::string name);

/**
 * 记录加载的自定义类实例
 * 从限定名称中获取最后一个 '.' 后的字符串部分
 * 与用户命名自定义类的方式相一致
 * 例如：__torch__.torch.classes.xnnpack.Conv2dOpContext
 */
#define RECORD_CUSTOM_CLASS(NAME) \
  auto name = std::string(NAME);  \
  detail::record_custom_class(name.substr(name.find_last_of(".") + 1));
#else
// 如果未启用记录自定义类功能，定义空的宏体
#define RECORD_CUSTOM_CLASS(NAME)
#endif

} // namespace detail

/// 用于表示自定义类方法注册时的参数默认值的结构体
struct arg {
  // 静态方法，表示默认值为 None。用法如下：
  //     torch::arg("name") = torch::arg::none
  // 等同于：
  //     torch::arg("name") = IValue()
  static c10::IValue none() {
    return c10::IValue();
  }

  // 显式构造函数，初始化参数名
  explicit arg(std::string name)
      : name_(std::move(name)), value_(c10::nullopt) {}

  // 赋值运算符重载，支持类似 pybind 的语法：torch::arg("name") = value
  arg& operator=(const c10::IValue& rhs) {
    value_ = rhs;
    return *this;
  }

  // 参数名，将被复制到模式中；无法从 C++ 声明中提取参数名
  std::string name_;
  // IValue 的默认构造函数将其初始化为 None，无法区分实际用户提供的 None 默认值
  // 和一个真实的用户提供的默认值。此布尔值帮助区分这两种情况。
  std::optional<c10::IValue> value_;
};

namespace detail {

// 参数类型工具
template <class R, class...>
struct types {
  using type = types;
};

// 封装方法模板，用于包装类方法成员函数
template <typename Method>
struct WrapMethod;

// 封装非常量成员函数的方法模板特化
template <typename R, typename CurrClass, typename... Args>
struct WrapMethod<R (CurrClass::*)(Args...)> {
  // 构造函数，接受成员函数指针作为参数
  WrapMethod(R (CurrClass::*m)(Args...)) : m(std::move(m)) {}

  // 操作符重载，调用成员函数
  R operator()(c10::intrusive_ptr<CurrClass> cur, Args... args) {
    return c10::guts::invoke(m, *cur, args...);
  }

  // 成员函数指针
  R (CurrClass::*m)(Args...);
};

// 封装常量成员函数的方法模板特化
template <typename R, typename CurrClass, typename... Args>
struct WrapMethod<R (CurrClass::*)(Args...) const> {
  // 构造函数，接受常量成员函数指针作为参数
  WrapMethod(R (CurrClass::*m)(Args...) const) : m(std::move(m)) {}

  // 操作符重载，调用常量成员函数
  R operator()(c10::intrusive_ptr<CurrClass> cur, Args... args) {
    // 使用 invoke 函数调用成员函数
    return c10::guts::invoke(m, *cur, args...);
  }

  // 常量成员函数指针
  R (CurrClass::*m)(Args...) const;
};

} // namespace detail

} // namespace torch
    // 调用 c10::guts::invoke 函数，传入成员函数指针 m 和参数列表 args...
    return c10::guts::invoke(m, *cur, args...);
  }

  // 定义一个成员函数指针 m，其返回类型为 R，接受参数列表 Args...
  R (CurrClass::*m)(Args...) const;
};

// 适配不同可调用类型的适配器
template <
    typename CurClass,
    typename Func,
    // 如果 Func 是成员函数指针，则启用此模板函数
    std::enable_if_t<
        std::is_member_function_pointer<std::decay_t<Func>>::value,
        bool> = false>
WrapMethod<Func> wrap_func(Func f) {
  return WrapMethod<Func>(std::move(f));
}

template <
    typename CurClass,
    typename Func,
    // 如果 Func 不是成员函数指针，则启用此模板函数
    std::enable_if_t<
        !std::is_member_function_pointer<std::decay_t<Func>>::value,
        bool> = false>
Func wrap_func(Func f) {
  return f;
}

template <
    class Functor,
    bool AllowDeprecatedTypes,
    size_t... ivalue_arg_indices>
// 从堆栈调用 TorchBind 方法
typename c10::guts::infer_function_traits_t<Functor>::return_type
call_torchbind_method_from_stack(
    Functor& functor,
    jit::Stack& stack,
    std::index_sequence<ivalue_arg_indices...>) {
  (void)(stack); // 当 sizeof...(ivalue_arg_indices) == 0 时，此参数将未使用，需要消除编译器警告。

  constexpr size_t num_ivalue_args = sizeof...(ivalue_arg_indices);

  using IValueArgTypes =
      typename c10::guts::infer_function_traits_t<Functor>::parameter_types;
  // TODO 这里不应直接使用 c10::impl，应使用 KernelFunction API。
  return (functor)(c10::impl::ivalue_to_arg<
                   typename c10::impl::decay_if_not_tensor<
                       c10::guts::typelist::
                           element_t<ivalue_arg_indices, IValueArgTypes>>::type,
                   AllowDeprecatedTypes>::
                       call(torch::jit::peek(
                           stack, ivalue_arg_indices, num_ivalue_args))...);
}

template <class Functor, bool AllowDeprecatedTypes>
// 从堆栈调用 TorchBind 方法的重载版本
typename c10::guts::infer_function_traits_t<Functor>::return_type
call_torchbind_method_from_stack(Functor& functor, jit::Stack& stack) {
  constexpr size_t num_ivalue_args =
      c10::guts::infer_function_traits_t<Functor>::number_of_parameters;
  return call_torchbind_method_from_stack<Functor, AllowDeprecatedTypes>(
      functor, stack, std::make_index_sequence<num_ivalue_args>());
}

template <class RetType, class Func>
struct BoxedProxy;

template <class RetType, class Func>
struct BoxedProxy {
  // BoxedProxy 的操作符重载，调用 TorchBind 方法并处理返回值
  void operator()(jit::Stack& stack, Func& func) {
    auto retval = call_torchbind_method_from_stack<Func, false>(func, stack);
    constexpr size_t num_ivalue_args =
        c10::guts::infer_function_traits_t<Func>::number_of_parameters;
    torch::jit::drop(stack, num_ivalue_args); // 丢弃堆栈中的参数
    stack.emplace_back(c10::ivalue::from(std::move(retval))); // 将返回值作为 IValue 放入堆栈
  }
};

template <class Func>
struct BoxedProxy<void, Func> {
  // 对于返回类型为 void 的 BoxedProxy 特化版本，调用 TorchBind 方法
  void operator()(jit::Stack& stack, Func& func) {
    call_torchbind_method_from_stack<Func, false>(func, stack);
    constexpr size_t num_ivalue_args =
        c10::guts::infer_function_traits_t<Func>::number_of_parameters;
    torch::jit::drop(stack, num_ivalue_args); // 丢弃堆栈中的参数
    stack.emplace_back(); // 在堆栈中放入一个空的 IValue
  }
};
// 检查字符是否是有效的标识符字符，包括字母、下划线或数字（但数字不能作为首字符）
inline bool validIdent(size_t i, char n) {
  return isalpha(n) || n == '_' || (i > 0 && isdigit(n));
}

// 检查字符串是否是有效的标识符，并在发现非法字符时抛出异常
inline void checkValidIdent(const std::string& str, const char* type) {
  // 遍历字符串中的每个字符
  for (const auto i : c10::irange(str.size())) {
    TORCH_CHECK(
        validIdent(i, str[i]),  // 调用validIdent函数检查字符是否合法
        type,
        " must be a valid Python/C++ identifier."
        " Character '",
        str[i],
        "' at index ",
        i,
        " is illegal.");  // 如果字符非法，抛出异常指示出错的字符和位置
  }
}

// 定义一个基类class_base，用于管理自定义类的命名空间和类名等信息
class TORCH_API class_base {
 protected:
  // 构造函数，接受命名空间名称、类名、文档字符串、内部指针类型信息和标记胶囊类类型信息
  explicit class_base(
      const std::string& namespaceName,
      const std::string& className,
      std::string doc_string,
      const std::type_info& intrusivePtrClassTypeid,
      const std::type_info& taggedCapsuleClass);

  // 静态方法，接受现有函数模式和初始化参数列表，返回具有新参数的函数模式
  static c10::FunctionSchema withNewArguments(
      const c10::FunctionSchema& schema,
      std::initializer_list<arg> default_args);

  // 类限定的类名字符串
  std::string qualClassName;
  // 类型指针，指向at::ClassType类型
  at::ClassTypePtr classTypePtr;
};

// namespace detail结束

// 用于注册自定义类的函数声明
TORCH_API void registerCustomClass(at::ClassTypePtr class_type);

// 用于注册自定义类方法的函数声明
TORCH_API void registerCustomClassMethod(std::unique_ptr<jit::Function> method);

// 给定限定名称（例如__torch__.torch.classes.Foo），返回描述该自定义类的ClassType指针，
// 如果找不到相应的类，则返回nullptr。
TORCH_API at::ClassTypePtr getCustomClass(const std::string& name);

// 给定一个IValue，如果其包含的对象是自定义的C++类，则返回true，否则返回false。
TORCH_API bool isCustomClass(const c10::IValue& v);

// 此API仅用于测试目的。在任何负载承载的代码中不应使用。
TORCH_API std::vector<c10::FunctionSchema> customClassSchemasForBCCheck();

// jit命名空间，使用torch的registerCustomClass和registerCustomClassMethod函数
namespace jit {
using ::torch::registerCustomClass;
using ::torch::registerCustomClassMethod;
} // namespace jit

// torch命名空间结束
```