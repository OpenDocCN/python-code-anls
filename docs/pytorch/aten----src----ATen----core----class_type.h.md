# `.\pytorch\aten\src\ATen\core\class_type.h`

```py
// 预处理指令，用于确保头文件只被包含一次
#pragma once

// 包含必要的头文件
#include <memory>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type_base.h>
#include <c10/util/Optional.h>

// 命名空间 c10，包含了 TorchScript 的核心功能
namespace torch::jit {
    // 前向声明，表示 TorchScript 的编译单元和函数
    struct CompilationUnit;
    struct Function;
} // namespace torch::jit

// 命名空间 c10，包含了核心的 C10 库功能
namespace c10 {

// 前向声明，表示函数模式的结构体
struct FunctionSchema;

// 枚举类型，表示属性的种类：缓冲区、参数或常规属性
enum class AttributeKind {
  BUFFER,            // 缓冲区属性
  PARAMETER,         // 参数属性
  REGULAR_ATTRIBUTE  // 常规属性
};

// 结构体，表示类属性的概念性实体：名称、种类（参见：AttributeKind）和类型（参见：TypePtr）
// 注意：该结构体不表示属性的值
struct TORCH_API ClassAttribute {
  public:
  // 构造函数，初始化类属性的种类、类型和名称
  ClassAttribute(AttributeKind kind,
                 TypePtr attributeType,
                 std::string attributeName) :
    kind_(kind),
    attributeType_(std::move(attributeType)),
    attributeName_(std::move(attributeName)) {}

  // 获取属性的种类
  AttributeKind getKind() const {
    return kind_;
  }

  // 获取属性的类型
  const TypePtr& getType() const {
    return attributeType_;
  }

  // 获取属性的名称
  const std::string& getName() const {
    return attributeName_;
  }

  private:
  AttributeKind kind_;      // 属性的种类
  TypePtr attributeType_;   // 属性的类型
  std::string attributeName_;  // 属性的名称
};

/**
 * User Defined Types
 */

// 前向声明，表示类类型的共享指针
struct ClassType;
using ClassTypePtr = std::shared_ptr<ClassType>;
using ::torch::jit::CompilationUnit;

// 结构体，表示 TorchScript 中的类类型
struct TORCH_API ClassType : public NamedType {
  // 结构体，表示类的属性；包括属性名称、获取函数和（可选的）设置函数
  struct Property {
    std::string name;               // 属性名称
    torch::jit::Function* getter;   // 获取函数指针
    torch::jit::Function* setter;   // 设置函数指针
  };

  // 创建一个具有给定名称和存储在 cu 中方法的类类型
  static ClassTypePtr create(
      std::optional<QualifiedName> qualifiedName,
      std::weak_ptr<CompilationUnit> cu,
      bool is_module = false,
      std::string doc_string = "",
      std::vector<std::string> unresolved_class_attributes = {});

  // 比较类类型是否相等的方法
  bool equals(const Type& rhs) const override {
    if (this == &rhs) {
      return true;
    }
    if (auto user_rhs = rhs.castRaw<ClassType>()) {
      const auto& lhs_name = name().value();
      const auto& rhs_name = user_rhs->name().value();

      return lhs_name == rhs_name &&
          this->compilation_unit() == user_rhs->compilation_unit();
    }
    return false;
  }

  // 返回类类型的字符串表示
  std::string str() const override {
     return annotation_str();
  }

  // 返回类类型的详细字符串表示，包括编译单元的信息
  std::string repr_str() const override {
    std::stringstream ss;
    ss << str()
       << " (of Python compilation unit at: " << compilation_unit().get() << ")";
    return ss.str();
  }

  // 获取类类型的所有方法
  const std::vector<torch::jit::Function*>& methods() const;

  // 根据名称查找属性的类型
  TypePtr findAttribute(const std::string& name) const {
    size_t pos = 0;
    for (const auto& attr : attributes_) {
      if (name == attr.getName()) {
        break;
      }
      ++pos;
    }
  // 如果索引超出属性数组的范围，返回空指针
  if (pos >= attributes_.size()) {
    return nullptr;
  }
  // 返回指定位置的属性类型指针
  return attributes_[pos].getType();
}

// 根据属性名获取属性类型指针
const TypePtr& getAttribute(const std::string& name) const {
  // 查找属性名对应的槽位索引
  auto slot = findAttributeSlot(name);
  // 检查是否找到了对应的槽位索引，否则抛出错误
  TORCH_CHECK(
      slot,
      repr_str(),
      " does not have an attribute with name '",
      name,
      "'");
  // 返回属性类型指针
  return attributes_[*slot].getType();
}

// 返回对象的属性数量
size_t numAttributes() const {
  return attributes_.size();
}

// 根据槽位索引获取属性类型指针
const TypePtr& getAttribute(size_t slot) const {
  // 断言槽位索引在属性数组的范围内
  AT_ASSERT(slot < attributes_.size());
  // 返回指定槽位的属性类型指针
  return attributes_.at(slot).getType();
}

// 根据槽位索引获取属性名
const std::string getAttributeName(size_t slot) const {
  // 断言槽位索引在属性数组的范围内
  AT_ASSERT(slot < attributes_.size());
  // 返回指定槽位的属性名
  return attributes_[slot].getName();
}

// 检查对象是否不存在指定名字的属性，用于错误处理
void checkNotExist(const std::string& name, const std::string& what) const;

// 在运行时，属性按照特定的槽位存储，以提高效率
// 在发出指令时，通过指定槽位进行常量时间的属性访问
std::optional<size_t> findAttributeSlot(const std::string& name) const {
  size_t slot = 0;
  // 遍历属性数组，查找指定名字的属性，并返回其槽位索引
  for (const auto& attr : attributes_) {
    if (name == attr.getName()) {
      return slot;
    }
    slot++;
  }
  // 如果未找到对应名字的属性，返回空值
  return c10::nullopt;
}

// 根据属性名获取其对应的槽位索引
size_t getAttributeSlot(const std::string& name) const {
  // 获取属性名对应的槽位索引
  if (auto r = findAttributeSlot(name)) {
    return *r;
  }
  // 如果未找到对应名字的属性，抛出错误
  TORCH_CHECK(
      false,
      repr_str(),
      " does not have an attribute with name '",
      name,
      "'");
}

// 检查对象是否具有指定名字的属性
bool hasAttribute(const std::string& name) const {
  // 使用 std::find_if 查找是否存在指定名字的属性
  return std::find_if(
             attributes_.cbegin(),
             attributes_.cend(),
             [&](const ClassAttribute& attr) { return attr.getName() == name; }) !=
      attributes_.cend();
}

// 判断指定名字的属性是否为未解析的类属性
bool isUnresolvedClassAttribute(const std::string& name) const;

// 实现基类方法，返回包含的类型数组引用
at::ArrayRef<TypePtr> containedTypes() const override {
  // 返回 attributeTypes_ 字段的值，即属性类型的集合
  return attributeTypes_;
}

// 添加一个属性到 ClassType 中
// 如果 is_parameter 为 true，则表示这个属性是一个参数
// 如果 is_buffer 为 true，则表示这个属性是一个缓冲区
size_t addAttribute(
    const std::string& name,
    TypePtr type,
    bool is_parameter = false,
    bool is_buffer = false);

// [仅内部使用] 从 ClassType 中移除一个属性
// 调用者需要确保这个修改是安全的：
// 不再有这个对象的现有分配，任何操作该属性的代码都无效
void unsafeRemoveAttribute(const std::string& name);

// [仅内部使用] 修改 ClassType 中属性的类型
// 调用者需要确保这个修改是安全的：
// 不再使用旧类型的属性，任何操作该属性的代码都无效
void unsafeChangeAttributeType(const std::string& name, const TypePtr& new_ty);

// 如果属性 \p NAME 不存在，则添加它；否则验证它是否具有兼容的类型
size_t addOrCheckAttribute(
    const std::string& name,
    TypePtr ty,
    bool is_parameter = false,
    bool is_buffer = false) {
  auto slot_idx = findAttributeSlot(name);
  if (!slot_idx) {
    return addAttribute(name, std::move(ty), is_parameter, is_buffer);
  }

  TORCH_CHECK(
      is_parameter == this->is_parameter(*slot_idx),
      "Parameter field mismatch for the field '",
      name,
      "'");
  const TypePtr& atype = getAttribute(*slot_idx);
  TORCH_CHECK(
    ty->isSubtypeOf(*atype),
    ty->repr_str(),
    " is not compatible with the type ",
    atype->repr_str(),
    " for the field '",
    name,
    "'");
  return *slot_idx;
}

// 获取具有给定名称的类的属性，如果存在的话
std::optional<ClassType::Property> getProperty(const std::string& name);

// 添加一个名为 \p name 的属性，其 getter 和 setter 分别为 \p getter 和 \p setter
void addProperty(const std::string& name, torch::jit::Function* getter, torch::jit::Function* setter);

// 获取所有属性的列表
const std::vector<Property>& properties() const {
  return properties_;
}

// 检查是否存在具有给定名称的常量
bool hasConstant(const std::string& name) const {
  return std::find_if(
             constantNames_.cbegin(),
             constantNames_.cend(),
             [&](const std::string& constant) { return constant == name; }) !=
      constantNames_.cend();
}

// 添加一个具有给定名称和值的常量
size_t addConstant(const std::string& name, const IValue& value);

// 查找具有给定名称的常量的槽位索引，如果不存在则返回空
std::optional<size_t> findConstantSlot(const std::string& name) const;

// 获取具有给定名称的常量的槽位索引，如果不存在则抛出错误
size_t getConstantSlot(const std::string& name) const {
  if (auto r = findConstantSlot(name)) {
    return *r;
  }
  TORCH_CHECK(
      false,
      repr_str(),
      " does not have constant field with the name '",
      name,
      "'");
}

// 获取具有给定槽位索引的常量的名称
const std::string& getConstantName(size_t slot) const;

// 获取类的文档字符串
const std::string& doc_string() const {
  // 返回对象的文档字符串
  return doc_string_;
}

// 根据名称获取常量的值
IValue getConstant(const std::string& name) const;

// 根据索引获取常量的值
IValue getConstant(size_t slot) const;

// 查找指定名称的常量，如果找到返回其值，否则返回空optional
std::optional<IValue> findConstant(const std::string& name) const;

// 返回常量的数量
size_t numConstants() const;

// 返回常量名数组的引用
at::ArrayRef<std::string> constantNames() const {
  return constantNames_;
}

// [仅内部使用] 从类类型中移除指定名称的常量
// 调用者需确保修改是安全的：
// 现有分配的对象不能再存在，对属性进行操作的代码将失效。
// 只有新创建的代码才会重新有效。
void unsafeRemoveConstant(const std::string& name);

// 创建一个包含指定类型的新类类型对象
TypePtr createWithContained(std::vector<TypePtr> contained_types) const override {
  auto ptr = ClassType::create(name(), compilation_unit_, is_module());
  AT_ASSERT(numAttributes() == contained_types.size());
  for(size_t i = 0; i < attributes_.size(); ++i) {
    AT_ASSERT(attributes_[i].getType()->isSubtypeOf(*contained_types[i]));
    ptr->addAttribute(attributes_[i].getName(), std::move(contained_types[i]));
  }
  // 将方法复制到新对象中
  for (const auto& method : methods()) {
    ptr->addMethod(method);
  }
  return ptr;
}

// 返回对象是否为模块类型
bool is_module() const override {
  return isModule_;
}

// 返回类类型的属性列表的引用
const std::vector<ClassAttribute>& getAttributes() const {
  return attributes_;
}

// 检查指定索引的属性是否为参数类型
bool is_parameter(size_t slot) const {
  TORCH_INTERNAL_ASSERT(
      is_module(), "asking for parameterSlots of non-Module");
  return attributes_.at(slot).getKind() == AttributeKind::PARAMETER;
}

// 检查指定索引的属性是否为缓冲区类型
bool is_buffer(size_t slot) const {
  TORCH_INTERNAL_ASSERT(
      is_module(), "asking for bufferWrittenSlots of non-Module");
    // 返回指定槽位的属性类型是否为 BUFFER 类型
    return attributes_.at(slot).getKind() == AttributeKind::BUFFER;
  }

  // 添加一个前向预处理钩子函数
  void addForwardPreHook(torch::jit::Function* pre_hook_ptr);
  // 添加一个前向钩子函数
  void addForwardHook(torch::jit::Function* hook_ptr);
  // 查找指定名称的前向预处理钩子函数
  torch::jit::Function* findForwardPreHook(const std::string& name) const;
  // 查找指定名称的前向钩子函数
  torch::jit::Function* findForwardHook(const std::string& name) const;
  // 返回所有前向钩子函数的向量引用
  const std::vector<torch::jit::Function*>& getForwardHooks() const;
  // 返回所有前向预处理钩子函数的向量引用
  const std::vector<torch::jit::Function*>& getForwardPreHooks() const;

  // 检查指定前向预处理钩子函数的函数模式
  void checkForwardPreHookSchema(
      int pre_hook_idx,
      const FunctionSchema& pre_hook_schema) const;
  // 检查指定前向钩子函数的函数模式
  void checkForwardHookSchema(
      int hook_idx,
      const FunctionSchema& hook_schema) const;

  // 添加一个方法函数
  void addMethod(torch::jit::Function* method);
  // 查找指定名称的方法函数
  torch::jit::Function* findMethod(const std::string& name) const;
  // 获取指定名称的方法函数（要求必须存在）
  torch::jit::Function& getMethod(const std::string& name) const;
  // 查找指定名称的钩子函数
  torch::jit::Function* findHook(const std::string& name) const;
  // 获取指定名称的钩子函数（要求必须存在）
  torch::jit::Function& getHook(const std::string& name) const;
  // 检查是否存在指定名称的方法函数
  bool hasMethod(const std::string& name) const;

  // 查找指定名称的静态方法函数
  torch::jit::Function* findStaticMethod(const std::string& name) const;
  // 添加一个静态方法函数
  void addStaticMethod(torch::jit::Function* method);

  // [Internal Only] 从 ClassType 中移除指定名称的方法函数
  // 调用者需确保修改是安全的：
  // 删除对象的现有分配是不安全的，任何操作该属性的代码现在都是无效的，
  // 只有新创建的代码才是有效的。
  // 注意：此方法仅供冻结使用。
  void unsafeRemoveMethod(const std::string& name);

  // 返回与该类相关的编译单元的共享指针
  std::shared_ptr<CompilationUnit> compilation_unit();

  // 返回与该类相关的编译单元的常量共享指针
  std::shared_ptr<const CompilationUnit> compilation_unit() const;

  // 生成该类的精炼版本。
  // 其名称相同，但槽位类型是原始槽位的子类型。
  // 仅在已知没有对对象槽位进行可能使精炼失效的赋值的上下文中，
  // 才能有效精炼类类型。
  // 这些变体未注册在全局类表中。
  ClassTypePtr refine(at::ArrayRef<TypePtr> refined_slots) const;

  // 检查该类型是否是 rhs 的子类型，并将原因写入 why_not（如果提供）
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // 类型的种类为 ClassType
  static const TypeKind Kind = TypeKind::ClassType;

 private:
  // 类型的构造函数，包括名称、编译单元、是否为模块、文档字符串和未解析类属性列表等参数
  ClassType(
      std::optional<QualifiedName> name,
      std::weak_ptr<CompilationUnit> cu,
      bool is_module = false,
      std::string doc_string = "",
      std::vector<std::string> unresolved_class_attributes = {});

  // 实现注解字符串的方法（未使用）
  std::string annotation_str_impl(C10_UNUSED const TypePrinter& printer = nullptr) const override {
    // 返回名称的值的引用
    const auto& n = name().value();
    // 返回属性的限定名称
    return n.qualifiedName();
  }

  // 添加一个类属性
  void addAttribute(ClassAttribute classAttribute);
  // 获取前置钩子错误信息
  std::string getForwardPreHookErrorMessage(int pre_hook_idx) const;
  // 获取钩子错误信息
  std::string getForwardHookErrorMessage(int hook_idx) const;

  // 属性名称到其类型的映射。
  // 注意：这里不包含方法，方法存储在模块中
  // TODO: 一旦模块支持任意 ivalue 属性，我们就不再需要这个了
  // TODO: 这最好表示为 OrderedDict，但目前 c10 不支持
  std::vector<std::string> constantNames_;
  std::vector<IValue> constantValues_;
  // 持有编译单元的弱引用
  std::weak_ptr<CompilationUnit> compilation_unit_;

  // 持有所有属性，属性详情在 ClassAttribute 中找到
  std::vector<ClassAttribute> attributes_;
  // 构建与 attributes_ 对应的类型列表，仅因 `containedTypes()` 方法返回 ArrayRef 而存在
  // 请勿直接填充此列表，应使用相应的 provideNewClassAttribute 方法
  std::vector<TypePtr> attributeTypes_;

  // 与此类相关的方法列表
  std::vector<torch::jit::Function*> methods_;
  std::vector<torch::jit::Function*> staticmethods_;

  // 在 forward 方法之前/之后运行的钩子列表
  std::vector<torch::jit::Function*> forward_hooks_;
  std::vector<torch::jit::Function*> forward_pre_hooks_;

  // 此类公开的属性列表
  std::vector<Property> properties_;

  // 类是否为模块
  bool isModule_ = false;

  // 类的文档字符串
  std::string doc_string_ = "";

  // 用于错误报告访问类级别属性时未解析的名称列表
  std::vector<std::string> unresolved_class_attributes_;
};

}


注释：


# 这部分代码片段看起来像是 JavaScript 或类似语言的结尾标记和函数结尾。
# 在某些编程语言中，'}' 表示代码块的结束，例如函数定义或条件语句的结束。
# 空行可能用于分隔不同的代码块或函数定义。
```