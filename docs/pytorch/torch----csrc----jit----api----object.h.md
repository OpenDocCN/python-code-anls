# `.\pytorch\torch\csrc\jit\api\object.h`

```
#pragma once

#include <ATen/core/functional.h>  // 引入 ATen 核心功能库的函数定义
#include <ATen/core/ivalue.h>  // 引入 ATen 核心库中 IValue 的定义
#include <c10/util/Optional.h>  // 引入 c10 中的 Optional 类
#include <torch/csrc/jit/api/method.h>  // 引入 Torch JIT API 中方法的定义

#include <utility>  // 引入实用工具

namespace torch::jit {

struct Resolver;
using ResolverPtr = std::shared_ptr<Resolver>;  // 使用 Resolver 的共享指针作为 ResolverPtr

using ObjectPtr = c10::intrusive_ptr<c10::ivalue::Object>;  // 使用 c10::ivalue::Object 的内部指针作为 ObjectPtr

// 在 C++ 环境中，如果 `attr` 失败，抛出此异常。这会被 Python 绑定代码转换为 AttributeError
class ObjectAttributeError : public std::runtime_error {
 public:
  ObjectAttributeError(const std::string& what) : std::runtime_error(what) {}
};

// Object 结构体，代表 Torch JIT 中的对象
struct TORCH_API Object {
  Object() = default;  // 默认构造函数
  Object(const Object&) = default;  // 拷贝构造函数
  Object& operator=(const Object&) = default;  // 拷贝赋值运算符
  Object(Object&&) noexcept = default;  // 移动构造函数
  Object& operator=(Object&&) noexcept = default;  // 移动赋值运算符
  Object(ObjectPtr _ivalue) : _ivalue_(std::move(_ivalue)) {}  // 基于给定的 ObjectPtr 构造 Object 对象
  Object(std::shared_ptr<CompilationUnit> cu, const c10::ClassTypePtr& type);  // 构造函数，初始化 CompilationUnit 和 ClassTypePtr
  Object(
      c10::QualifiedName,
      std::shared_ptr<CompilationUnit> cu,
      bool shouldMangle = false);  // 构造函数，初始化 QualifiedName、CompilationUnit 和 shouldMangle 标志

  ObjectPtr _ivalue() const {  // 返回存储的 _ivalue_ 成员变量
    TORCH_INTERNAL_ASSERT(_ivalue_);  // 内部断言，确保 _ivalue_ 不为空
    return _ivalue_;
  }

  c10::ClassTypePtr type() const {  // 返回 _ivalue_ 的类型
    return _ivalue()->type();
  }

  // 嵌套结构体 Property，包含属性名称、getter 方法和可选的 setter 方法
  struct Property {
    std::string name;
    Method getter_func;
    std::optional<Method> setter_func;
  };

  void setattr(const std::string& name, c10::IValue v) {  // 设置对象的属性值
    if (_ivalue()->type()->hasConstant(name)) {  // 如果属性是常量
      TORCH_CHECK(
          false,
          "Can't set constant '",
          name,
          "' which has value:",
          _ivalue()->type()->getConstant(name));  // 抛出错误，不能设置常量属性
    } else if (auto slot = _ivalue()->type()->findAttributeSlot(name)) {  // 如果属性是可变属性
      const c10::TypePtr& expected = _ivalue()->type()->getAttribute(*slot);  // 获取属性的预期类型
      TORCH_CHECK(
          v.type()->isSubtypeOf(*expected),  // 检查值的类型是否符合预期
          "Expected a value of type '",
          expected->repr_str(),
          "' for field '",
          name,
          "', but found '",
          v.type()->repr_str(),
          "'");  // 如果类型不符合预期，抛出错误
      _ivalue()->setSlot(*slot, std::move(v));  // 设置属性值
    } else {
      TORCH_CHECK(false, "Module has no attribute '", name, "'");  // 如果模块没有该属性，抛出错误
    }
  }

  c10::IValue attr(const std::string& name) const {  // 获取对象的属性值
    if (auto r = _ivalue()->type()->findAttributeSlot(name)) {  // 查找可变属性槽位
      return _ivalue()->getSlot(*r);  // 返回属性值
    }
    if (auto r = _ivalue()->type()->findConstantSlot(name)) {  // 查找常量属性槽位
      return _ivalue()->type()->getConstant(*r);  // 返回常量属性值
    }
    std::stringstream err;  // 创建错误消息流
    err << _ivalue()->type()->repr_str() << " does not have a field with name '"
        << name.c_str() << "'";  // 构造错误消息，指明属性不存在
    throw ObjectAttributeError(err.str());  // 抛出属性错误异常
  }

  c10::IValue attr(const std::string& name, c10::IValue or_else) const {  // 获取对象的属性值，如果属性不存在则返回默认值
    if (auto r = _ivalue()->type()->findAttributeSlot(name)) {  // 查找可变属性槽位
      return _ivalue()->getSlot(*r);  // 返回属性值
    }
    if (auto r = _ivalue()->type()->findConstantSlot(name)) {  // 查找常量属性槽位
      return _ivalue()->type()->getConstant(*r);  // 返回常量属性值
    }
    return or_else;  // 属性不存在，返回默认值
  }

  bool hasattr(const std::string& name) const {  // 检查对象是否具有指定名称的属性
  // 返回当前对象的类型，然后检查是否具有指定名称的属性或常量
  return _ivalue()->type()->hasAttribute(name) ||
      _ivalue()->type()->hasConstant(name);
}

// 每个对象拥有自己的方法。这里返回的引用保证在模块销毁之前始终有效
Method get_method(const std::string& name) const {
  // 查找指定名称的方法，如果找到则返回该方法
  if (auto method = find_method(name)) {
    return *method;
  }
  // 抛出错误，指定名称的方法未定义
  AT_ERROR("Method '", name, "' is not defined.");
}

// 返回当前对象的所有方法，使用lambda表达式转换函数指针为Method对象
const std::vector<Method> get_methods() const {
  return c10::fmap(type()->methods(), [&](Function* func) {
    return Method(_ivalue(), func);
  });
}

// 检查当前对象是否具有指定名称的属性
bool has_property(const std::string& name) const {
  // 遍历当前对象的所有属性，如果找到指定名称的属性则返回true
  for (const auto& prop : type()->properties()) {
    if (prop.name == name) {
      return true;
    }
  }
  // 没有找到指定名称的属性，返回false
  return false;
}

// 获取当前对象的指定名称的属性，包括其getter和setter方法
const Property get_property(const std::string& name) const {
  // 遍历当前对象的所有属性，查找指定名称的属性
  for (const auto& prop : type()->properties()) {
    if (prop.name == name) {
      // 如果属性具有setter方法，则创建对应的Method对象
      std::optional<Method> setter = c10::nullopt;
      if (prop.setter) {
        setter = Method(_ivalue(), prop.setter);
      }
      // 返回包含属性名称、getter方法和setter方法的Property对象
      return Property{
          prop.name, Method(_ivalue(), prop.getter), std::move(setter)};
    }
  }
  // 没有找到指定名称的属性，抛出错误
  AT_ERROR("Property '", name, "' is not defined.");
}

// 返回当前对象的所有属性，使用lambda表达式转换为Property对象
const std::vector<Property> get_properties() const {
  return c10::fmap(type()->properties(), [&](ClassType::Property prop) {
    // 如果属性具有setter方法，则创建对应的Method对象
    std::optional<Method> setter = c10::nullopt;
    if (prop.setter) {
      setter = Method(_ivalue(), prop.setter);
    }
    // 返回包含属性名称、getter方法和setter方法的Property对象
    return Property{
        std::move(prop.name),
        Method(_ivalue(), prop.getter),
        std::move(setter)};
  });
}

// 声明一个方法的查找函数，实际定义未提供在此处
std::optional<Method> find_method(const std::string& basename) const;

/// 运行此模块中的一个方法。
///
/// 例如:
/// @code
///   IValue output = module->run("relu_script", a, b);
/// @endcode
///
/// 若要从源字符串编译模块，请参见 torch::jit::compile
///
/// @param method_name 要运行的方法的名称
/// @param args 要传递给方法的参数
/// @return 包含方法返回值（如果是元组则是多个返回值）的IValue对象
template <typename... Types>
IValue run_method(const std::string& method_name, Types&&... args) {
  // 调用get_method获取方法对象，然后执行该方法并返回其结果
  return get_method(method_name)({IValue(std::forward<Types>(args))...});
}

// 允许C++用户轻松添加方法的方法
void define(const std::string& src, const ResolverPtr& resolver = nullptr);

// 返回当前对象的slot数量，即_ivalue对象的slots的大小
size_t num_slots() const {
  return _ivalue()->slots().size();
}

// 浅拷贝当前对象，返回拷贝后的新对象
Object copy() const;

// 递归地复制对象的所有属性，包括Tensor的深拷贝，返回复制后的新对象
Object deepcopy() const;

private:
// 惰性初始化模块对象的可变指针_ivalue_
mutable ObjectPtr _ivalue_;
};

namespace script {
// 声明一个名为 `script` 的命名空间

// 我们曾经有一个被删除的 `script::` 命名空间。这个类型别名是为了向后兼容
// 公共 API；新代码不应该使用这个类型别名。
using Object = ::torch::jit::Object;
// 定义一个类型别名 `Object`，表示为 `torch::jit::Object`

} // namespace script
// 结束 `script` 命名空间

} // namespace torch::jit
// 结束 `torch::jit` 命名空间
```