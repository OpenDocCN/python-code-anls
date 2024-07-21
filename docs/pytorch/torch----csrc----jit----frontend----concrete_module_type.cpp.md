# `.\pytorch\torch\csrc\jit\frontend\concrete_module_type.cpp`

```
#include <torch/csrc/jit/frontend/concrete_module_type.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <iostream>

namespace torch::jit {

// ConcreteModuleTypeBuilder 类的成员函数，用于根据当前对象创建一个新的 ClassTypePtr 类型
ClassTypePtr ConcreteModuleTypeBuilder::createTypeFromThis() const {
  // 获取当前 Python 编译单元
  auto cu = get_python_cu();
  // 从 torch._jit_internal 模块导入并获取当前类的限定名称
  py::object pyQualName = py::module::import("torch._jit_internal")
                              .attr("_qualified_name")(pyClass_);

  // 将 Python 字符串形式的限定类名转换为 C++ 的 QualifiedName 对象
  auto className = c10::QualifiedName(py::cast<std::string>(pyQualName));
  // 如果类名没有前缀，则添加默认前缀 "__torch__"
  if (className.prefix().empty()) {
    className = c10::QualifiedName("__torch__", className.name());
  }
  // 如果当前编译单元中已存在同名的类，则进行名称修饰
  if (cu->get_class(className) != nullptr) {
    className = cu->mangle(className);
  }
  // 创建一个新的 ClassType 对象，表示一个模块
  auto cls = ClassType::create(std::move(className), cu, /*is_module=*/true);
  // 在当前编译单元中注册这个新创建的类
  cu->register_type(cls);

  // 使用 ConcreteModuleTypeBuilder 对象中的属性信息填充新创建的类
  // 添加类的属性信息
  for (const auto& pr : attributes_) {
    const auto& name = pr.key();
    const auto& type = pr.value().type_;
    const auto& isParameter = pr.value().isParam_;
    const auto& isBuffer = pr.value().isBuffer_;
    cls->addAttribute(name, type, isParameter, isBuffer);
  }

  // 添加类的常量信息
  for (const auto& pr : constants_) {
    cls->addConstant(pr.first, pr.second);
  }

  // 添加类的模块信息
  for (const auto& moduleInfo : modules_) {
    cls->addAttribute(
        moduleInfo.name_,
        moduleInfo.meta_->getJitType(),
        /*is_parameter=*/false);
  }

  // 返回创建的类类型对象
  return cls;
}

// 从 JIT 类型 TypePtr 创建 ConcreteModuleType 的共享指针
std::shared_ptr<ConcreteModuleType> ConcreteModuleType::fromJitType(
    TypePtr type) {
  ConcreteModuleTypeBuilder builder;
  builder.setPoisoned();

  // `type` 应该是一个模块接口或者类类型
  if (auto interface = type->cast<InterfaceType>()) {
    TORCH_INTERNAL_ASSERT(interface->is_module());
  } else {
    const auto classType = type->expect<ClassType>();

    // 从 JIT 类型中填充 builder 的元数据，确保从 Python 创建和直接从 JIT 类型创建的 ConcreteModuleTypes 行为一致
    for (const auto i : c10::irange(classType->numAttributes())) {
      const auto& attrName = classType->getAttributeName(i);
      const auto& attrType = classType->getAttribute(i);
      if (attrType->is_module()) {
        builder.addModule(attrName, ConcreteModuleType::fromJitType(attrType));
      } else {
        builder.addAttribute(
            attrName,
            attrType,
            classType->is_parameter(i),
            classType->is_buffer(i));
      }
    }

    // 添加 JIT 类型中的常量信息到 builder
    for (const auto i : c10::irange(classType->numConstants())) {
      builder.addConstant(
          classType->getConstantName(i), classType->getConstant(i));
    }
  }

  // 使用 new ConcreteModuleType() 构造 ConcreteModuleType 对象，不使用 make_shared 是因为构造函数是私有的
  auto ret = std::shared_ptr<ConcreteModuleType>(new ConcreteModuleType());
  // 设置 ret 的 jitType_ 成员为传入的类型 type
  ret->jitType_ = std::move(type);
  // 设置 ret 的 data_ 成员为 builder
  ret->data_ = builder;

  // 返回构造的 ConcreteModuleType 对象
  return ret;
}

// ConcreteModuleType 的构造函数，接受一个 ConcreteModuleTypeBuilder 对象作为参数
ConcreteModuleType::ConcreteModuleType(ConcreteModuleTypeBuilder data)
    // 使用移动语义初始化成员变量 data_，减少数据的拷贝开销
    : data_(std::move(data)) {
        // 根据成员变量 data_ 创建一个 JIT 类型，并将其赋值给成员变量 jitType_
        jitType_ = data_.createTypeFromThis();
}

// 定义重载的等于运算符，比较两个 ModuleInfo 对象是否相等
bool operator==(
    const ConcreteModuleTypeBuilder::ModuleInfo& lhs,
    const ConcreteModuleTypeBuilder::ModuleInfo& rhs) {
  // 比较两个 ModuleInfo 对象的 name_ 和 meta_ 是否相等
  return lhs.name_ == rhs.name_ && lhs.meta_->equals(*rhs.meta_);
}

// 比较当前 ConcreteModuleTypeBuilder 对象与另一个对象是否相等
bool ConcreteModuleTypeBuilder::equals(
    const ConcreteModuleTypeBuilder& other) const {
  // 如果任一对象为毒药状态，则返回不相等
  if (isPoisoned_ || other.isPoisoned_) {
    return false;
  }

  // clang-format off
  // 下面的比较按照大致顺序排列，以便先进行便宜且判别性强的检查
  bool equal =
    pyClass_.is(other.pyClass_) &&                            // 检查 pyClass_ 是否相等
    iterableModuleKind_ == other.iterableModuleKind_ &&        // 检查 iterableModuleKind_ 是否相等
    ignoredAttributes_ == other.ignoredAttributes_ &&          // 检查 ignoredAttributes_ 是否相等
    constants_ == other.constants_ &&                          // 检查 constants_ 是否相等
    attributes_ == other.attributes_ &&                        // 检查 attributes_ 是否相等
    overloads_ == other.overloads_ &&                          // 检查 overloads_ 是否相等
    functionAttributes_ == other.functionAttributes_ &&        // 检查 functionAttributes_ 是否相等
    builtinFunctions_ == other.builtinFunctions_ &&            // 检查 builtinFunctions_ 是否相等
    forwardHooks_ == other.forwardHooks_ &&                    // 检查 forwardHooks_ 是否相等
    forwardPreHooks_ == other.forwardPreHooks_;                // 检查 forwardPreHooks_ 是否相等
  // clang-format on

  // 如果以上比较结果为不相等，则返回 false
  if (!equal) {
    return false;
  }

  // 将 modules_ 按名称排序，以便进行比较（插入顺序不影响相等性）
  auto thisSorted = modules_;
  std::sort(
      thisSorted.begin(),
      thisSorted.end(),
      [](const ModuleInfo& a, const ModuleInfo& b) {
        return a.name_ < b.name_;
      });

  auto otherSorted = other.modules_;
  std::sort(
      otherSorted.begin(),
      otherSorted.end(),
      [](const ModuleInfo& a, const ModuleInfo& b) {
        return a.name_ < b.name_;
      });

  // 比较排序后的 modules_ 是否相等
  return thisSorted == otherSorted;
}

// 返回当前 ConcreteModuleType 对象的 jitType_
TypePtr ConcreteModuleType::getJitType() const {
  return jitType_;
}

// 返回当前 ConcreteModuleType 对象的 pyClass_，如果不存在则返回空 optional
std::optional<py::object> ConcreteModuleType::getPyClass() const {
  if (!data_.pyClass_) {
    return c10::nullopt;
  }
  return data_.pyClass_;
}

// 根据名称查找当前 ConcreteModuleType 对象的函数重载列表，如果不存在则返回空 optional
std::optional<std::vector<std::string>> ConcreteModuleType::findOverloads(
    const std::string& name) const {
  const auto it = data_.overloads_.find(name);
  if (it != data_.overloads_.end()) {
    return it->second;
  }
  return c10::nullopt;
}

// 根据名称查找当前 ConcreteModuleType 对象的函数属性，如果不存在则返回空 optional
std::optional<Function*> ConcreteModuleType::findFunctionAttribute(
    const std::string& name) const {
  const auto it = data_.functionAttributes_.find(name);
  if (it != data_.functionAttributes_.end()) {
    return it->second.function_->function();
  }
  return c10::nullopt;
}

// 根据名称查找当前 ConcreteModuleType 对象的内置函数，如果不存在则返回空 optional
std::optional<c10::Symbol> ConcreteModuleType::findBuiltinFunction(
    const std::string& name) const {
  const auto it = data_.builtinFunctions_.find(name);
  if (it != data_.builtinFunctions_.end()) {
    return it->second;
  }
  return c10::nullopt;
}

// 根据名称查找当前 ConcreteModuleType 对象的失败属性，如果不存在则返回空 optional
std::optional<std::string> ConcreteModuleType::findFailedAttribute(
    const std::string& name) const {
  const auto it = data_.failedAttributes_.find(name);
  if (it != data_.failedAttributes_.end()) {
    return it->second;
  }
  return c10::nullopt;
}
// 检查给定名称是否在 ignoredAttributes_ 集合中，返回布尔值
bool ConcreteModuleType::isIgnoredAttribute(const std::string& name) const {
  return data_.ignoredAttributes_.count(name) > 0;
}

// 在 modules_ 列表中查找指定名称的子模块的具体类型，并返回其 shared_ptr
std::shared_ptr<ConcreteModuleType> ConcreteModuleType::
    findSubmoduleConcreteType(const std::string& name) const {
  // 使用 lambda 表达式查找符合条件的 ModuleInfo 对象
  const auto it = std::find_if(
      data_.modules_.cbegin(),
      data_.modules_.cend(),
      [&](const ConcreteModuleTypeBuilder::ModuleInfo& info) {
        return info.name_ == name;
      });
  // 断言确保找到了对应的模块信息
  TORCH_INTERNAL_ASSERT(it != data_.modules_.end());
  // 返回找到的子模块的 meta_ 成员变量，即其具体类型的 shared_ptr
  return it->meta_;
}

// 设置当前模块类型的 iterableModuleKind_ 成员变量
void ConcreteModuleTypeBuilder::setIterableModuleKind(IterableModuleKind kind) {
  iterableModuleKind_ = kind;
}

// 获取当前模块类型的 iterableModuleKind_ 成员变量的值
IterableModuleKind ConcreteModuleType::getIterableModuleKind() const {
  return data_.iterableModuleKind_;
}

// 将当前模块类型标记为 poisoned，表示其状态异常
void ConcreteModuleTypeBuilder::setPoisoned() {
  isPoisoned_ = true;
}

// 添加一个常量到 constants_ 集合，根据给定的 py::object 推断其类型
void ConcreteModuleTypeBuilder::addConstant(
    std::string name,
    py::object value) {
  auto match = tryToInferType(value);
  // 如果推断类型失败，则断言失败并输出错误信息
  if (!match.success()) {
    TORCH_INTERNAL_ASSERT(
        false,
        "We need to infer the type of constant to convert the python value to IValue,"
        " but failed to infer type of ",
        py::str(value),
        "\n:",
        match.reason());
  }
  // 将推断出的类型和值插入到 constants_ 中
  constants_.emplace(std::move(name), toIValue(std::move(value), match.type()));
}

// 直接添加一个已知类型的常量到 constants_ 中
void ConcreteModuleTypeBuilder::addConstant(std::string name, IValue value) {
  constants_.emplace(std::move(name), std::move(value));
}

// 添加一个属性到 attributes_ 集合，包括其类型信息及是否为参数或缓冲
void ConcreteModuleTypeBuilder::addAttribute(
    std::string name,
    const TypePtr& type,
    bool isParameter,
    bool isBuffer) {
  TORCH_INTERNAL_ASSERT(type);
  // 断言确保函数属性需要单独处理
  TORCH_INTERNAL_ASSERT(type->cast<FunctionType>() == nullptr);
  // 插入新的属性信息到 attributes_ 中
  attributes_.insert(
      std::move(name),
      ConcreteModuleTypeBuilder::Attribute(
          unshapedType(type), isParameter, isBuffer));
}

// 添加一个函数属性到 functionAttributes_ 集合，包括其类型信息和 Python 函数对象
void ConcreteModuleTypeBuilder::addFunctionAttribute(
    std::string name,
    const TypePtr& type,
    py::object pyFunction) {
  TORCH_INTERNAL_ASSERT(type);
  // 将函数属性添加到 functionAttributes_ 中
  functionAttributes_.emplace(
      std::move(name),
      ConcreteModuleTypeBuilder::FunctionAttribute{
          type->expect<FunctionType>(), std::move(pyFunction)});
}

// 添加一个内置函数到 builtinFunctions_ 集合，使用其名称和符号名
void ConcreteModuleTypeBuilder::addBuiltinFunction(
    std::string name,
    const std::string& symbol_name) {
  builtinFunctions_.emplace(
      std::move(name), c10::Symbol::fromQualString(symbol_name));
}

// 添加一个子模块到 modules_ 列表，使用给定的名称和元信息的 shared_ptr
void ConcreteModuleTypeBuilder::addModule(
    std::string name,
    std::shared_ptr<ConcreteModuleType> meta) {
  modules_.emplace_back(std::move(name), std::move(meta));
}

// 添加一个前向钩子函数到 forwardHooks_ 列表
void ConcreteModuleTypeBuilder::addForwardHook(py::object hook) {
  forwardHooks_.emplace_back(std::move(hook));
}

// 添加一个前向预处理钩子函数到 forwardPreHooks_ 列表
void ConcreteModuleTypeBuilder::addForwardPreHook(py::object pre_hook) {
  forwardPreHooks_.emplace_back(std::move(pre_hook));
}

// 添加一个方法重载到 overloads_ 列表，使用给定的方法名称和信息
void ConcreteModuleTypeBuilder::addOverload(
    std::string methodName,
    // 将 methodName 和其重载方法的名称列表插入到 overloads_ 中
    overloads_.emplace(std::move(methodName), std::move(overloadedMethodNames));
} // 结束 namespace torch::jit

void ConcreteModuleTypeBuilder::addFailedAttribute(
    std::string name,
    std::string failureReason) {
  // 向 failedAttributes_ 中添加一个失败的属性，使用移动语义优化参数传递
  failedAttributes_.emplace(std::move(name), std::move(failureReason));
}

void ConcreteModuleTypeBuilder::addIgnoredAttribute(std::string name) {
  // 向 ignoredAttributes_ 中添加一个被忽略的属性，使用移动语义优化参数传递
  ignoredAttributes_.emplace(std::move(name));
}

void ConcreteModuleType::dump() const {
  // 打印 ConcreteModuleType 对象的相关信息
  std::cout << "ConcreteModuleType for: "
            << py::getattr(data_.pyClass_, "__name__") << "\n";
  std::cout << "Constants: \n";
  // 打印对象的常量信息
  for (const auto& pr : data_.constants_) {
    std::cout << "\t" << pr.first << ": " << pr.second << "\n";
  }
  std::cout << "\nAttributes: \n";
  // 打印对象的属性信息
  for (const auto& pr : data_.attributes_) {
    std::cout << "\t" << pr.key() << ": " << pr.value().type_->annotation_str()
              << "\n";
  }
  std::cout << "\nSubmodules: \n";
  // 打印对象的子模块信息
  for (const auto& info : data_.modules_) {
    std::cout << "\t" << info.name_ << ": "
              << info.meta_->getJitType()->annotation_str() << "\n";
  }
  std::cout << "\nForward Pre-Hooks: \n";
  // 打印对象的前向预处理钩子信息
  for (const auto& pre_hook_id : data_.forwardPreHooks_) {
    std::cout << "\t"
              << "pre_hook id: " << pre_hook_id << "\n";
  }
  std::cout << "\nForward Hooks: \n";
  // 打印对象的前向钩子信息
  for (const auto& hook_id : data_.forwardHooks_) {
    std::cout << "\t"
              << "hook id: " << hook_id << "\n";
  }
  std::cout << "\nOverloads: \n";
  // 打印对象的重载信息
  for (const auto& pr : data_.overloads_) {
    std::cout << "\t" << pr.first << ": " << pr.second << "\n";
  }
  std::string isPoisoned = data_.isPoisoned_ ? "true" : "false";
  std::cout << "isPoisoned: " << isPoisoned << "\n";
  if (jitType_) {
    std::cout << "jit type: " << jitType_->annotation_str() << "\n";
  }
}

std::unordered_map<std::string, py::object> ConcreteModuleType::getConstantsPy()
    const {
  // 转换成更适合 pybind 的常量表示，并返回一个 unordered_map
  std::unordered_map<std::string, py::object> ret;
  for (const auto& pr : data_.constants_) {
    ret.emplace(pr.first, toPyObject(pr.second));
  }
  return ret;
}

std::unordered_map<std::string, std::pair<TypePtr, bool>> ConcreteModuleType::
    getAttributesPy() const {
  // 转换成更适合 pybind 的属性表示，并返回一个 unordered_map
  std::unordered_map<std::string, std::pair<TypePtr, bool>> ret;
  for (auto& pr : data_.attributes_) {
    ret.emplace(
        pr.key(),
        std::pair<TypePtr, bool>(pr.value().type_, pr.value().isParam_));
  }
  return ret;
}

std::vector<std::pair<std::string, std::shared_ptr<ConcreteModuleType>>>
ConcreteModuleType::getModulesPy() const {
  // 返回对象的子模块信息，以 vector 形式存储
  std::vector<std::pair<std::string, std::shared_ptr<ConcreteModuleType>>> ret;

  ret.reserve(data_.modules_.size());
  for (const auto& info : data_.modules_) {
    ret.emplace_back(info.name_, info.meta_);
  }
  return ret;
}

} // 结束 namespace torch::jit
```