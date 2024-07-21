# `.\pytorch\aten\src\ATen\core\custom_class.cpp`

```py
#include <ATen/core/function_schema.h>
#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/type_factory.h>
#include <ATen/record_function.h>
#include <c10/util/flat_hash_map.h>
#include <torch/custom_class.h>
#include <torch/custom_class_detail.h>

#include <unordered_map>

namespace c10 {

// 获取自定义类类型映射表的静态函数，返回一个 ska::flat_hash_map，映射std::type_index到c10::ClassTypePtr
static ska::flat_hash_map<std::type_index, c10::ClassTypePtr>&
getCustomClassTypeMap() {
  static ska::flat_hash_map<std::type_index, c10::ClassTypePtr> tmap;
  return tmap;
}

// 根据给定的std::type_index获取对应的自定义类类型 c10::ClassTypePtr
c10::ClassTypePtr getCustomClassTypeImpl(const std::type_index& tindex) {
  auto& tmap = c10::getCustomClassTypeMap();
  auto res = tmap.find(tindex);
  if (C10_UNLIKELY(res == tmap.end())) {
    // 如果未找到对应的类型索引，通过慢速路径迭代所有已注册的类型，比较它们的名称以寻找匹配的类类型
    auto class_name = std::string(tindex.name());
    for (const auto& it : tmap) {
      if (class_name == it.first.name()) {
        // 在此模板中不修改现有类型映射，应该只有在getCustomClassTypeImpl()函数中调用
        return it.second;
      }
    }
    // 如果未找到匹配的类类型，抛出错误消息
    TORCH_CHECK(
        false,
        "Can't find class id in custom class type map for ",
        tindex.name());
  }
  return res->second;
}

} // namespace c10

namespace torch {

namespace detail {

#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
// 记录自定义类的函数，用于性能分析和调试，接受类名作为参数
void record_custom_class(std::string name) {
  RECORD_FUNCTION_WITH_SCOPE(
      at::RecordScope::CUSTOM_CLASS,
      std::move(name),
      c10::ArrayRef<const c10::IValue>{});
}
#endif

} // namespace detail

// 返回自定义类注册表的静态函数，映射类名到at::ClassTypePtr
static std::unordered_map<std::string, at::ClassTypePtr>& customClasses() {
  static std::unordered_map<std::string, at::ClassTypePtr> customClasses;
  return customClasses;
}

// 注册自定义类类型到全局注册表，确保每个类只注册一次
void registerCustomClass(at::ClassTypePtr class_type) {
  TORCH_INTERNAL_ASSERT(class_type->name());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto name = class_type->name()->qualifiedName();
  TORCH_CHECK(
      !customClasses().count(name),
      "Custom class with name ",
      name,
      " is already registered. Ensure that registration with torch::class_ is only called once.");
  customClasses()[name] = std::move(class_type);
}

// 根据类名获取已注册的自定义类类型，如果找到则记录该类名的自定义类
at::ClassTypePtr getCustomClass(const std::string& class_name) {
  auto ret =
      customClasses().count(class_name) ? customClasses()[class_name] : nullptr;
  if (ret) {
    RECORD_CUSTOM_CLASS(class_name);
  }
  return ret;
}

} // namespace torch
const std::unordered_set<std::string> getAllCustomClassesNames() {
  // 创建一个空的无序集合用于存储自定义类的名称
  std::unordered_set<std::string> ret;
  // 遍历所有已注册的自定义类，并将类名插入到集合中
  for (const auto& kv : customClasses()) {
    ret.insert(kv.first);
  }
  // 返回包含所有自定义类名称的集合
  return ret;
}

bool isCustomClass(const c10::IValue& v) {
  // 检查给定的值是否是一个对象，并且其类型名称不为空
  return v.isObject() && v.toObject()->type()->name() &&
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      // 获取对象的类型名称，并检查是否是自定义类
      getCustomClass(v.toObject()->type()->name()->qualifiedName());
}

static std::vector<std::unique_ptr<jit::Function>>& customClassMethods() {
  // 静态变量，存储自定义类的方法函数的指针向量
  static std::vector<std::unique_ptr<jit::Function>> customClassMethods;
  // 返回存储自定义类方法函数的静态向量的引用
  return customClassMethods;
}

void registerCustomClassMethod(std::unique_ptr<jit::Function> fn) {
  // 向自定义类方法函数的静态向量中注册新的方法函数
  customClassMethods().emplace_back(std::move(fn));
}

std::vector<c10::FunctionSchema> customClassSchemasForBCCheck() {
  // 获取自定义类方法函数的静态向量
  auto& methods = customClassMethods();
  // 映射方法函数向量以获取函数模式的向量，并返回结果
  return c10::fmap(methods, [](const std::unique_ptr<jit::Function>& fn) {
    return fn->getSchema();
  });
}

namespace detail {
class_base::class_base(
    const std::string& namespaceName,
    const std::string& className,
    std::string doc_string,
    const std::type_info& intrusivePtrClassTypeid,
    const std::type_info& taggedCapsuleClassTypeid)
    : qualClassName(
          "__torch__.torch.classes." + namespaceName + '.' + className),
      // 使用类的完全限定名称创建类类型指针
      classTypePtr(at::ClassType::create(
          c10::QualifiedName(qualClassName),
          std::weak_ptr<jit::CompilationUnit>(),
          /*is_module=*/false,
          std::move(doc_string))) {
  // 检查命名空间名称和类名的有效性
  detail::checkValidIdent(namespaceName, "Namespace name");
  detail::checkValidIdent(className, "Class name");
  // 添加一个名为 "capsule" 的属性到类类型指针
  classTypePtr->addAttribute(
      "capsule", c10::TypeFactory::get<c10::CapsuleType>());
  // 将类类型指针与其对应的类类型标识插入自定义类类型映射中
  c10::getCustomClassTypeMap().insert(
      {std::type_index(intrusivePtrClassTypeid), classTypePtr});
  c10::getCustomClassTypeMap().insert(
      {std::type_index(taggedCapsuleClassTypeid), classTypePtr});

  // 注册该自定义类类型
  registerCustomClass(classTypePtr);
}

c10::FunctionSchema class_base::withNewArguments(
    const c10::FunctionSchema& schema,
    std::initializer_list<arg> default_args) {
  // 获取函数模式的参数列表
  const auto& old_args = schema.arguments();
  std::vector<c10::Argument> new_args;
  new_args.reserve(old_args.size());

  // 将原始参数列表中的第一个参数（通常是 self）添加到新参数列表中
  new_args.emplace_back(old_args[0]);
  // 跳过 self 参数，从 default_args 中添加默认参数到新参数列表中
  size_t argIdx = 1;
  for (const auto& default_arg : default_args) {
    auto& old_arg = old_args[argIdx++];
    new_args.emplace_back(
        default_arg.name_,
        old_arg.type(),
        old_arg.real_type(),
        old_arg.N(),
        default_arg.value_);
  }
  // 使用新的参数列表克隆原始函数模式，并返回结果
  return schema.cloneWithArguments(std::move(new_args));
}

} // namespace detail
} // namespace torch
```