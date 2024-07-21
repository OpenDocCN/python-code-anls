# `.\pytorch\torch\custom_class.h`

```py
#pragma once
// 防止头文件被多次包含

#include <ATen/core/builtin_function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/class_type.h>
#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/stack.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/util/TypeTraits.h>
#include <torch/custom_class_detail.h>
#include <torch/library.h>
#include <sstream>

namespace torch {

/// 命名空间 torch 中定义的一些功能和类

/// This function is used in conjunction with `class_::def()` to register
/// a constructor for a given C++ class type. For example,
/// `torch::init<int, std::string>()` would register a two-argument constructor
/// taking an `int` and a `std::string` as argument.
template <class... Types>
detail::types<void, Types...> init() {
  // 注册一个构造函数，使得可以使用不同类型的参数进行初始化

  return detail::types<void, Types...>{};
}

template <typename Func, typename... ParameterTypeList>
struct InitLambda {
  Func f;
};

template <typename Func>
decltype(auto) init(Func&& f) {
  // 推断函数参数类型，初始化一个 lambda 函数对象

  using InitTraits = c10::guts::infer_function_traits_t<std::decay_t<Func>>;
  using ParameterTypeList = typename InitTraits::parameter_types;

  InitLambda<Func, ParameterTypeList> init{std::forward<Func>(f)};
  return init;
}

/// Entry point for custom C++ class registration. To register a C++ class
/// in PyTorch, instantiate `torch::class_` with the desired class as the
/// template parameter. Typically, this instantiation should be done in
/// the initialization of a global variable, so that the class will be
/// made available on dynamic library loading without any additional API
/// calls needed. For example, to register a class named Foo, you might
/// create a global variable like so:
///
///     static auto register_foo = torch::class_<Foo>("myclasses", "Foo")
///       .def("myMethod", &Foo::myMethod)
///       .def("lambdaMethod", [](const c10::intrusive_ptr<Foo>& self) {
///         // Do something with `self`
///       });
///
/// In addition to registering the class, this registration also chains
/// `def()` calls to register methods. `myMethod()` is registered with
/// a pointer to the Foo class's `myMethod()` method. `lambdaMethod()`
/// is registered with a C++ lambda expression.
template <class CurClass>
// 模板，用于注册自定义的 C++ 类到 PyTorch 中
class class_ : public ::torch::detail::class_base {
  // 静态断言，确保CurClass派生自CustomClassHolder
  static_assert(
      std::is_base_of<CustomClassHolder, CurClass>::value,
      "torch::class_<T> requires T to inherit from CustomClassHolder");

 public:
  /// This constructor actually registers the class type.
  /// String argument `namespaceName` is an identifier for the
  /// namespace you would like this class to appear in.
  /// String argument `className` is the name you would like to
  /// see this class exposed as in Python and TorchScript. For example, if
  /// you pass `foo` as the namespace name and `Bar` as the className, the
  /// class will appear as `torch.classes.foo.Bar` in Python and TorchScript
  explicit class_(
      const std::string& namespaceName,
      const std::string& className,
      std::string doc_string = "")
      : class_base(
            namespaceName,
            className,
            std::move(doc_string),
            typeid(c10::intrusive_ptr<CurClass>),
            typeid(c10::tagged_capsule<CurClass>)) {}

  /// def() can be used in conjunction with `torch::init()` to register
  /// a constructor for a given C++ class type. For example, passing
  /// `torch::init<int, std::string>()` would register a two-argument
  /// constructor taking an `int` and a `std::string` as argument.
  template <typename... Types>
  // 用于与torch::init<...>()结合使用，注册给定C++类类型的构造函数
  class_& def(
      torch::detail::types<void, Types...>,
      std::string doc_string = "",
      std::initializer_list<arg> default_args =
          {}) { // Used in combination with
    // torch::init<...>()
    // lambda函数，创建类对象并设置到self的第一个槽位
    auto func = [](c10::tagged_capsule<CurClass> self, Types... args) {
      auto classObj = c10::make_intrusive<CurClass>(args...);
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(std::move(classObj)));
    };

    // 定义方法"__init__"
    defineMethod(
        "__init__",
        std::move(func),
        std::move(doc_string),
        default_args);
    return *this;
  }

  // Used in combination with torch::init([]lambda(){......})
  template <typename Func, typename... ParameterTypes>
  // 用于与torch::init([]lambda(){......})结合使用
  class_& def(
      InitLambda<Func, c10::guts::typelist::typelist<ParameterTypes...>> init,
      std::string doc_string = "",
      std::initializer_list<arg> default_args = {}) {
    // lambda包装器，调用init.f并将结果设置到self的第一个槽位
    auto init_lambda_wrapper = [func = std::move(init.f)](
                                   c10::tagged_capsule<CurClass> self,
                                   ParameterTypes... arg) {
      c10::intrusive_ptr<CurClass> classObj =
          at::guts::invoke(func, std::forward<ParameterTypes>(arg)...);
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(classObj));
    };

    // 定义方法"__init__"
    defineMethod(
        "__init__",
        std::move(init_lambda_wrapper),
        std::move(doc_string),
        default_args);
    return *this;
  }

  /// 返回当前对象的引用，以支持链式调用
  /// This is the normal method registration API. `name` is the name that
  /// the method will be made accessible by in Python and TorchScript.
  /// `f` is a callable object that defines the method. Typically `f`
  /// will either be a pointer to a method on `CurClass`, or a lambda
  /// expression that takes a `c10::intrusive_ptr<CurClass>` as the first
  /// argument (emulating a `this` argument in a C++ method.)
  ///
  /// Examples:
  ///
  ///     // Exposes method `foo` on C++ class `Foo` as `call_foo()` in
  ///     // Python and TorchScript
  ///     .def("call_foo", &Foo::foo)
  ///
  ///     // Exposes the given lambda expression as method `call_lambda()`
  ///     // in Python and TorchScript.
  ///     .def("call_lambda", [](const c10::intrusive_ptr<Foo>& self) {
  ///       // do something
  ///     })
  template <typename Func>
  /// 定义一个成员方法，使得在 Python 和 TorchScript 中可以注册该方法
  class_& def(
      std::string name,
      Func f,
      std::string doc_string = "",
      std::initializer_list<arg> default_args = {}) {
    auto wrapped_f = detail::wrap_func<CurClass, Func>(std::move(f));
    /// 将传入的函数对象 `f` 包装为一个能够被注册的函数对象 `wrapped_f`
    defineMethod(
        std::move(name),
        std::move(wrapped_f),
        std::move(doc_string),
        default_args);
    return *this;
  }

  /// Method registration API for static methods.
  template <typename Func>
  /// 用于注册静态方法的方法注册API
  class_& def_static(std::string name, Func func, std::string doc_string = "") {
    auto qualMethodName = qualClassName + "." + name;
    auto schema =
        c10::inferFunctionSchemaSingleReturn<Func>(std::move(name), "");

    auto wrapped_func =
        [func = std::move(func)](jit::Stack& stack) mutable -> void {
      using RetType =
          typename c10::guts::infer_function_traits_t<Func>::return_type;
      detail::BoxedProxy<RetType, Func>()(stack, func);
    };
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        std::move(qualMethodName),
        std::move(schema),
        std::move(wrapped_func),
        std::move(doc_string));

    classTypePtr->addStaticMethod(method.get());
    /// 将静态方法注册到类类型指针 `classTypePtr` 所指向的类中
    registerCustomClassMethod(std::move(method));
    return *this;
  }

  /// Property registration API for properties with both getter and setter
  /// functions.
  template <typename GetterFunc, typename SetterFunc>
  /// 用于注册同时具有 getter 和 setter 函数的属性的属性注册API
  class_& def_property(
      const std::string& name,
      GetterFunc getter_func,
      SetterFunc setter_func,
      std::string doc_string = "") {
    torch::jit::Function* getter{};
    torch::jit::Function* setter{};

    auto wrapped_getter =
        detail::wrap_func<CurClass, GetterFunc>(std::move(getter_func));
    getter = defineMethod(name + "_getter", wrapped_getter, doc_string);

    auto wrapped_setter =
        detail::wrap_func<CurClass, SetterFunc>(std::move(setter_func));
    setter = defineMethod(name + "_setter", wrapped_setter, doc_string);

    classTypePtr->addProperty(name, getter, setter);
    /// 将属性注册到类类型指针 `classTypePtr` 所指向的类中
  // 返回当前对象的引用，用于链式调用
  return *this;
}

/// 用于只有 getter 函数的属性注册的 API
template <typename GetterFunc>
class_& def_property(
    const std::string& name,
    GetterFunc getter_func,
    std::string doc_string = "") {
  torch::jit::Function* getter{};

  // 包装 getter 函数，生成一个新的函数对象
  auto wrapped_getter =
      detail::wrap_func<CurClass, GetterFunc>(std::move(getter_func));
  // 定义 getter 方法并获取其函数对象
  getter = defineMethod(name + "_getter", wrapped_getter, doc_string);

  // 将属性添加到类类型中，使用给定的 getter 方法和空的 setter 方法
  classTypePtr->addProperty(name, getter, nullptr);
  // 返回当前对象的引用，用于链式调用
  return *this;
}

/// 用于具有读写访问权限的属性注册的 API
template <typename T>
class_& def_readwrite(const std::string& name, T CurClass::*field) {
  // 定义 getter 函数，返回属性的当前值
  auto getter_func = [field =
                          field](const c10::intrusive_ptr<CurClass>& self) {
    return self.get()->*field;
  };

  // 定义 setter 函数，设置属性的新值
  auto setter_func = [field = field](
                         const c10::intrusive_ptr<CurClass>& self, T value) {
    self.get()->*field = value;
  };

  // 调用 def_property 方法注册读写属性，使用定义的 getter 和 setter 函数
  return def_property(name, getter_func, setter_func);
}

/// 用于具有只读访问权限的属性注册的 API
template <typename T>
class_& def_readonly(const std::string& name, T CurClass::*field) {
  // 定义 getter 函数，返回属性的当前值
  auto getter_func =
      [field = std::move(field)](const c10::intrusive_ptr<CurClass>& self) {
        return self.get()->*field;
      };

  // 调用 def_property 方法注册只读属性，只使用定义的 getter 函数
  return def_property(name, getter_func);
}

/// 该方法用于注册自定义 JIT 后端支持的自定义 C++ 类的方法，属于不安全的 API，
/// 不适用于一般用途。
class_& _def_unboxed(
    const std::string& name,
    std::function<void(jit::Stack&)> func,
    c10::FunctionSchema schema,
    std::string doc_string = "") {
  // 创建一个内置操作函数对象，用于表示自定义方法
  auto method = std::make_unique<jit::BuiltinOpFunction>(
      qualClassName + "." + name,
      std::move(schema),
      std::move(func),
      std::move(doc_string));
  // 将方法添加到类类型中
  classTypePtr->addMethod(method.get());
  // 注册自定义类方法
  registerCustomClassMethod(std::move(method));

  // 返回当前对象的引用，用于链式调用
  return *this;
}
  // 返回当前对象的引用
  return *this;
}

/// def_pickle() is used to define exactly what state gets serialized
/// or deserialized for a given instance of a custom C++ class in
/// Python or TorchScript. This protocol is equivalent to the Pickle
/// concept of `__getstate__` and `__setstate__` from Python
/// (https://docs.python.org/2/library/pickle.html#object.__getstate__)
///
/// Currently, both the `get_state` and `set_state` callables must be
/// C++ lambda expressions. They should have the following signatures,
/// where `CurClass` is the class you're registering and `T1` is some object
/// that encapsulates the state of the object.
///
///     __getstate__(intrusive_ptr<CurClass>) -> T1
///     __setstate__(T2) -> intrusive_ptr<CurClass>
///
/// `T1` must be an object that is convertable to IValue by the same rules
/// for custom op/method registration.
///
/// For the common case, T1 == T2. T1 can also be a subtype of T2. An
/// example where it makes sense for T1 and T2 to differ is if __setstate__
/// handles legacy formats in a backwards compatible way.
///
/// Example:
///
///     .def_pickle(
///         // __getstate__
///         [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
///           return self->stack_;
///         },
///         [](std::vector<std::string> state) { // __setstate__
///            return c10::make_intrusive<MyStackClass<std::string>>(
///               std::vector<std::string>{"i", "was", "deserialized"});
///         })
template <typename GetStateFn, typename SetStateFn>
class_& def_pickle(GetStateFn&& get_state, SetStateFn&& set_state) {
  static_assert(
      c10::guts::is_stateless_lambda<std::decay_t<GetStateFn>>::value &&
          c10::guts::is_stateless_lambda<std::decay_t<SetStateFn>>::value,
      "def_pickle() currently only supports lambdas as "
      "__getstate__ and __setstate__ arguments.");
  // 将 __getstate__ 注册为 get_state 的lambda函数
  def("__getstate__", std::forward<GetStateFn>(get_state));

  // __setstate__ 需要进行一些定制处理注册：
  // 我们需要包装用户提供的函数调用，以获取返回值（即 c10::intrusive_ptr<CurrClass>）
  // 并将其赋给 `capsule` 属性。
  using SetStateTraits =
      c10::guts::infer_function_traits_t<std::decay_t<SetStateFn>>;
  using SetStateArg = typename c10::guts::typelist::head_t<
      typename SetStateTraits::parameter_types>;
  auto setstate_wrapper = [set_state = std::forward<SetStateFn>(set_state)](
                              c10::tagged_capsule<CurClass> self,
                              SetStateArg&& arg) {
    // 调用 set_state 函数并获取返回的类对象指针
    c10::intrusive_ptr<CurClass> classObj =
        at::guts::invoke(set_state, std::forward<SetStateArg>(arg));
    auto object = self.ivalue.toObject();
    // 将类对象指针作为 capsule 放入对象的第一个槽位
    object->setSlot(0, c10::IValue::make_capsule(classObj));
  };
    // 定义方法 "__setstate__"，使用 wrap_func 包装 setstate_wrapper 函数并传递给当前类
    defineMethod(
        "__setstate__",
        detail::wrap_func<CurClass, decltype(setstate_wrapper)>(
            std::move(setstate_wrapper)));

    // 类型验证
    // 获取 __getstate__ 方法的 schema
    auto getstate_schema = classTypePtr->getMethod("__getstate__").getSchema();
    
    // 格式化输出 getstate_schema 的字符串表示
    auto format_getstate_schema = [&getstate_schema]() {
      std::stringstream ss;
      ss << getstate_schema;
      return ss.str();
    };
    
    // 检查 __getstate__ 方法的参数个数是否为1
    TORCH_CHECK(
        getstate_schema.arguments().size() == 1,
        "__getstate__ should take exactly one argument: self. Got: ",
        format_getstate_schema());
    
    // 获取 __getstate__ 方法的第一个参数类型
    auto first_arg_type = getstate_schema.arguments().at(0).type();
    
    // 检查 __getstate__ 方法的第一个参数类型是否与 classTypePtr 相符
    TORCH_CHECK(
        *first_arg_type == *classTypePtr,
        "self argument of __getstate__ must be the custom class type. Got ",
        first_arg_type->repr_str());
    
    // 检查 __getstate__ 方法的返回值个数是否为1
    TORCH_CHECK(
        getstate_schema.returns().size() == 1,
        "__getstate__ should return exactly one value for serialization. Got: ",
        format_getstate_schema());

    // 获取 __getstate__ 方法返回值的类型
    auto ser_type = getstate_schema.returns().at(0).type();
    
    // 获取 __setstate__ 方法的 schema
    auto setstate_schema = classTypePtr->getMethod("__setstate__").getSchema();
    
    // 获取 __setstate__ 方法的第二个参数类型
    auto arg_type = setstate_schema.arguments().at(1).type();
    
    // 检查 __getstate__ 方法返回值类型是否是 __setstate__ 方法第二个参数类型的子类型
    TORCH_CHECK(
        ser_type->isSubtypeOf(*arg_type),
        "__getstate__'s return type should be a subtype of "
        "input argument of __setstate__. Got ",
        ser_type->repr_str(),
        " but expected ",
        arg_type->repr_str());

    // 返回当前对象的引用
    return *this;
  }
    // 创建一个 lambda 函数 wrapped_func，捕获并移动 func 到闭包中
    auto wrapped_func =
        [func = std::move(func)](jit::Stack& stack) mutable -> void {
      // TODO: 我们需要找出如何对调用这种自定义函数进行性能分析！
      // 目前无法做到，因为性能分析工具在 libtorch 中而不是 ATen 中
      // 推断 func 的返回类型
      using RetType =
          typename c10::guts::infer_function_traits_t<Func>::return_type;
      // 创建一个 BoxedProxy 对象，调用闭包中的 func
      detail::BoxedProxy<RetType, Func>()(stack, func);
    };
    // 创建一个 jit::BuiltinOpFunction 对象，并使用移动语义将参数传递给构造函数
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        qualMethodName,
        std::move(schema),
        std::move(wrapped_func),
        std::move(doc_string));

    // 在此注册方法以保持 Method 对象的生命周期
    // ClassTypes 通常不持有它们的方法的所有权（通常由 CompilationUnit 持有），
    // 因此我们在这里需要一个代理来处理这种行为
    auto method_val = method.get();
    // 将 method_val 添加到 classTypePtr 指向的 ClassType 对象中
    classTypePtr->addMethod(method_val);
    // 注册自定义类方法，将 method 对象移动到注册函数中
    registerCustomClassMethod(std::move(method));
    // 返回指向注册的方法的指针，以便在需要时使用
    return method_val;
  }
};

/// make_custom_class() is a convenient way to create an instance of a
/// registered custom class and wrap it in an IValue, for example when you want
/// to pass the object to TorchScript. Its syntax is equivalent to APIs like
/// `std::make_shared<>` or `c10::make_intrusive<>`.
///
/// For example, if you have a custom C++ class that can be constructed from an
/// `int` and `std::string`, you might use this API like so:
///
///     IValue custom_class_iv = torch::make_custom_class<MyClass>(3,
///     "foobarbaz");
template <typename CurClass, typename... CtorArgs>
c10::IValue make_custom_class(CtorArgs&&... args) {
  // 创建一个用户自定义类的实例，并将其封装在 IValue 中返回，用于 TorchScript
  auto userClassInstance =
      c10::make_intrusive<CurClass>(std::forward<CtorArgs>(args)...);
  return c10::IValue(std::move(userClassInstance));
}

// Alternative api for creating a torchbind class over torch::class_ this api is
// preffered to prevent size regressions on Edge usecases. Must be used in
// conjunction with TORCH_SELECTIVE_CLASS macro aka
// selective_class<foo>("foo_namespace", TORCH_SELECTIVE_CLASS("foo"))
template <class CurClass>
inline class_<CurClass> selective_class_(
    const std::string& namespace_name,
    detail::SelectiveStr<true> className) {
  // 返回一个 torch::class_<CurClass> 实例，用于创建 Torch 绑定类
  auto class_name = std::string(className.operator const char*());
  return torch::class_<CurClass>(namespace_name, class_name);
}

template <class CurClass>
inline detail::ClassNotSelected selective_class_(
    const std::string&,
    detail::SelectiveStr<false>) {
  return detail::ClassNotSelected();
}

// jit namespace for backward-compatibility
// We previously defined everything in torch::jit but moved it out to
// better reflect that these features are not limited only to TorchScript
namespace jit {

using ::torch::class_;
using ::torch::getCustomClass;
using ::torch::init;
using ::torch::isCustomClass;

} // namespace jit

template <class CurClass>
inline class_<CurClass> Library::class_(const std::string& className) {
  // 在 Library 类中定义一个名为 className 的类，用于 TorchScript 的类定义
  TORCH_CHECK(
      kind_ == DEF || kind_ == FRAGMENT,
      "class_(\"",
      className,
      "\"): Cannot define a class inside of a TORCH_LIBRARY_IMPL block.  "
      "All class_()s should be placed in the (unique) TORCH_LIBRARY block for their namespace.  "
      "(Error occurred at ",
      file_,
      ":",
      line_,
      ")");
  TORCH_INTERNAL_ASSERT(ns_.has_value(), file_, ":", line_);
  return torch::class_<CurClass>(*ns_, className);
}

const std::unordered_set<std::string> getAllCustomClassesNames();

template <class CurClass>
// 定义内联函数 class_，其返回类型为 inline class_<CurClass>
inline class_<CurClass> Library::class_(detail::SelectiveStr<true> className) {
  // 将 className 转换为 std::string 类型
  auto class_name = std::string(className.operator const char*());
  // 检查当前的 Library 对象的 kind_ 属性，确保其为 DEF 或 FRAGMENT 类型，否则抛出错误信息
  TORCH_CHECK(
      kind_ == DEF || kind_ == FRAGMENT,
      "class_(\"",
      class_name,
      "\"): Cannot define a class inside of a TORCH_LIBRARY_IMPL block.  "
      "All class_()s should be placed in the (unique) TORCH_LIBRARY block for their namespace.  "
      "(Error occurred at ",
      file_,
      ":",
      line_,
      ")");
  // 断言当前命名空间 ns_ 必须有值，否则输出错误信息并终止程序
  TORCH_INTERNAL_ASSERT(ns_.has_value(), file_, ":", line_);
  // 返回一个 torch::class_<CurClass> 对象，该对象绑定到当前的命名空间和给定的类名
  return torch::class_<CurClass>(*ns_, class_name);
}

// 模板特化版本，当 SelectiveStr<false> 时调用，返回 detail::ClassNotSelected 对象
template <class CurClass>
inline detail::ClassNotSelected Library::class_(detail::SelectiveStr<false>) {
  return detail::ClassNotSelected();
}

} // namespace torch
```