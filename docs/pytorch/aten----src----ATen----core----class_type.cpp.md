# `.\pytorch\aten\src\ATen\core\class_type.cpp`

```
// 包含 ATen 库的头文件

#include <ATen/core/class_type.h>

// 包含 ATen 核心功能的头文件
#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <ATen/core/grad_mode.h>
#include <ATen/core/function.h>

// 进入 ATen 命名空间
namespace c10 {

// 向类别中添加方法的函数定义
void ClassType::addMethod(torch::jit::Function* method) {
  // 检查是否已定义同名方法，防止重定义
  TORCH_CHECK(
      findMethod(method->name()) == nullptr,
      "Can't redefine method: ",
      method->name(),
      " on class: ",
      repr_str());
  // 将方法添加到方法列表中
  methods_.push_back(method);
}

// 获取前向钩子列表的方法定义
const std::vector<torch::jit::Function*>& ClassType::getForwardHooks() const {
    return forward_hooks_;
}

// 获取前向预钩子列表的方法定义
const std::vector<torch::jit::Function*>& ClassType::getForwardPreHooks() const {
    return forward_pre_hooks_;
}

// 向前向预钩子列表中添加钩子的方法定义
void ClassType::addForwardPreHook(torch::jit::Function* pre_hook_ptr) {
    forward_pre_hooks_.emplace_back(pre_hook_ptr);
}

// 向前向钩子列表中添加钩子的方法定义
void ClassType::addForwardHook(torch::jit::Function* hook_ptr) {
    forward_hooks_.emplace_back(hook_ptr);
}

// 根据名称查找前向预钩子的方法定义
torch::jit::Function* ClassType::findForwardPreHook(const std::string& name) const {
  for (const auto& pre_hook : forward_pre_hooks_) {
    if (name == pre_hook->name()) {
      return pre_hook;
    }
  }
  return nullptr;
}

// 根据名称查找前向钩子的方法定义
torch::jit::Function* ClassType::findForwardHook(const std::string& name) const {
  for (const auto& hook : forward_hooks_) {
    if (name == hook->name()) {
      return hook;
    }
  }
  return nullptr;
}

// 获取函数模式中输入类型的字符串表示
static std::string getSchemaInputTypesString(const FunctionSchema& schema) {
  std::stringstream input_types;
  const std::vector<Argument>& forward_args = schema.arguments();
  // 遍历参数列表并构建类型字符串
  for (const auto i : c10::irange(1, forward_args.size())) {
    input_types << forward_args[i].type()->annotation_str();
    if (forward_args.size() - 1 != i) {
      input_types << ", ";
    }
  }
  // 如果参数列表为空，添加空括号
  if (forward_args.size() == 1) {
    input_types << "()";
  }
  return input_types.str();
}

// 获取前向预钩子错误消息的方法定义
std::string ClassType::getForwardPreHookErrorMessage(int pre_hook_idx) const {
  // 获取前向预钩子的名称和前向方法的函数模式
  const std::string& pre_hook_name = forward_pre_hooks_[pre_hook_idx]->name();
  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  std::string input_types = getSchemaInputTypesString(forward_schema);
  const std::vector<Argument>& forward_args = forward_schema.arguments();

  std::string single_output = "";
  // 如果输出类型是单个元组，需要将其包装在外部元组中以匹配 eager 模式的行为
    # 创建一个字符串，表示函数签名中的类型注解信息，包括返回类型
    single_output = ", '" + forward_args[1].type()->annotation_str() + "',";
  }
  # 构建预处理钩子函数的签名字符串，包括函数名和输入参数的类型注解
  std::string pre_hook_schema =
      pre_hook_name + "(self, input: Tuple[" + input_types + "])";
  # 构建描述错误的返回字符串，包括错误的原因和预期的函数签名格式
  std::string return_string =
      "This error occurred while scripting the forward pre-hook '" +
      # NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      pre_hook_name + "' on module '" + name()->name() +
      "'. If you did not want to script this pre-hook remove it from the "
      "original NN module before scripting. Pre-hooks for module '" +
    # NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      name()->name() + "' are expected to have the following signature: "
      + pre_hook_schema + " with a return type of either 'None'"
      # 将单个输出类型信息添加到错误描述字符串中
      + single_output + " or 'Tuple[" + input_types + "]'.";
  # 返回构建好的错误描述字符串
  return return_string;
// 返回给定索引的前向钩子的错误消息
std::string ClassType::getForwardHookErrorMessage(int hook_idx) const {
  // 获取前向钩子的名称
  const std::string& hook_name = forward_hooks_[hook_idx]->name();
  // 获取前向方法的函数模式
  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  // 创建前向方法输入类型的字符串表示
  std::string input_types = getSchemaInputTypesString(forward_schema);

  // 创建预期输出类型的字符串表示
  const Argument& pre_output =
      (hook_idx == 0)
          ? forward_schema.returns()[0]
          : forward_hooks_[hook_idx - 1]->getSchema().returns()[0];
  std::string output_types = pre_output.type()->annotation_str();
  // 创建错误消息
  std::string hook_schema = hook_name + "(self, input: Tuple[" +
                            input_types + "], output: " + output_types + ")";
  std::string return_string =
      "This error occurred while scripting the forward hook '" +
      hook_name + "' on module " + name()->name() +
      ". If you did not want to script this hook remove it from" +
      " the original NN module before scripting. This hook was" +
      " expected to have the following signature: " + hook_schema +
      ". The type of the output arg is the returned type from" +
      " either the forward method or the previous hook if it exists. " +
      "Note that hooks can return anything, but if the hook is " +
      "on a submodule the outer module is expecting" +
      " the same return type as the submodule's forward.";
  return return_string;
}

// 检查是否存在未解析的类属性
bool ClassType::isUnresolvedClassAttribute(const std::string& name) const {
  return std::find(
      unresolved_class_attributes_.begin(),
      unresolved_class_attributes_.end(),
      name) != unresolved_class_attributes_.end();
}

// 检查前向钩子的输入参数是否符合预期
static void checkForwardHookInputArguments(
    const FunctionSchema& forward_schema,
    const FunctionSchema& hook_schema,
    const std::string& hook_id,
    const std::string& hook_err_msg) {
  // 检查输入参数是否为元组类型
  const std::vector<Argument>& forward_args = forward_schema.arguments();
  const Argument input_arg = hook_schema.arguments()[1];
  TORCH_CHECK(
      input_arg.type()->cast<TupleType>() != nullptr,
      hook_id,
      "expected the input argument to be typed as a Tuple but found type: '",
      input_arg.type()->annotation_str(),
      "' instead.\n",
      hook_err_msg
   );

  const at::ArrayRef<TypePtr> input_tuple_types = input_arg.type()->castRaw<TupleType>()->elements();
  if (forward_args.size() == 1) {
    // 检查空输入的情况
    TORCH_CHECK(
        input_tuple_types.empty(),
        hook_id,
        "was expecting Tuple[()] as the input type. Received type: '",
        input_arg.type()->annotation_str(),
        "'.\n",
        hook_err_msg
      );
  } else {
    // 检查输入元组的大小和包含的类型是否正确
    // 检查输入参数元组的类型数量是否与前向参数数量相符
    TORCH_CHECK(
        input_tuple_types.size() == forward_args.size() - 1,
        hook_id,
        "has the wrong number of contained types for the",
        " input argument's Tuple. Received type: '",
        input_arg.type()->annotation_str(),
        "'.\n",
        hook_err_msg
    );

    // 遍历除第一个外的所有前向参数
    for (const auto i : c10::irange(1, forward_args.size())) {
        // 检查当前前向参数的类型是否与对应的输入参数元组类型匹配
        if (*forward_args[i].type() != *input_tuple_types[i - 1]) {
            // 如果类型不匹配，抛出错误信息
            TORCH_CHECK(
                false,
                hook_id,
                "has the wrong inner types for the input tuple argument. Received type: '",
                input_arg.type()->annotation_str(),
                "'.\n",
                hook_err_msg
            );
        }
    }
}
}

void ClassType::checkForwardPreHookSchema(
    int pre_hook_idx,
    const FunctionSchema& pre_hook_schema) const {
  // 获取前置钩子函数的指针
  const torch::jit::Function* pre_hook = forward_pre_hooks_[pre_hook_idx];
  // 构建前置钩子的标识字符串，包括钩子函数名称和模块名称
  std::string hook_id =
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      "Pre-hook '" + pre_hook->name() + "' on module '" + name()->name() + "' ";
  // 获取前置钩子的错误信息字符串
  std::string pre_hook_err_msg = getForwardPreHookErrorMessage(pre_hook_idx) + "\n";

  // 前置钩子期望有两个输入：self 和一个包含传递给 Forward 函数的非 self 参数的元组
  TORCH_CHECK(
      pre_hook_schema.arguments().size() == 2,
      hook_id,
      "was expected to only have exactly 2 inputs but it had ",
      pre_hook_schema.arguments().size(),
      " inputs. ",
      pre_hook_err_msg
   );

  // 获取 forward 方法的函数模式
  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  // 获取 forward 方法的参数列表
  const std::vector<Argument>& forward_args = forward_schema.arguments();
  // 检查前置钩子的输入参数是否符合预期
  checkForwardHookInputArguments(forward_schema, pre_hook_schema, hook_id, pre_hook_err_msg);

  // 检查返回类型，期望是 None、与输入相同的类型，或者如果输入是包含单个类型的元组，则返回单个类型
  TORCH_CHECK(
            !pre_hook_schema.returns().empty(),
            hook_id,
            "is missing a return annotation. Return annotations are required, please add one.\n",
            pre_hook_err_msg
  );
  // 获取返回类型的参数
  const Argument return_arg = pre_hook_schema.returns()[0];
  // 构建错误消息，指出返回类型不符合预期
  std::string wrong_type_returned_err_msg = hook_id +
      "returned the wrong type of: '" +
      return_arg.type()->annotation_str() + "'.";

  // 如果返回类型为 None，则直接返回
  if (return_arg.type()->kind() == NoneType::get()->kind()) {
    return;
  }
  // 如果 forward 方法有两个参数且前置钩子返回的类型与第二个参数类型匹配，则进行以下检查
  if (forward_args.size() == 2 && *forward_args[1].type() == *return_arg.type()) {
    // 下面的 TORCH_CHECK 是为了处理特殊情况：当 forward 的输入是一个元组，而前置钩子返回一个匹配的元组时。
    // Eager 模式不支持这种情况，元组类型的返回值应该是 forward 输入元组的包装。
    TORCH_CHECK(
        return_arg.type()->cast<TupleType>() == nullptr,
        wrong_type_returned_err_msg,
        " When forward has a single tuple input argument, the return needs",
        " to be 'None' or a nested tuple containing forward's input tuple",
        " argument as in: 'Tuple[",
        forward_args[1].type()->annotation_str(),
        "]'.\n",
        pre_hook_err_msg
    );
    return;
  }
  // 现在返回值只能是嵌套类型的元组
  // 检查返回值是否是元组类型
  TORCH_CHECK(
      return_arg.type()->cast<TupleType>() != nullptr,
      wrong_type_returned_err_msg,
      pre_hook_err_msg
  );
  // 获取返回值元组中的类型列表
  const at::ArrayRef<TypePtr> return_tuple_types =
      return_arg.type()->castRaw<TupleType>()->elements();
  // 处理 forward 没有参数的边缘情况，即 forward_args.size() == 1
  if (forward_args.size() == 1) {
    // 检查返回类型的元组是否为空，应为空以满足预期条件，否则抛出错误信息
    TORCH_CHECK(
        return_tuple_types.empty(),
        wrong_type_returned_err_msg,
        " Was expecting either 'None' or 'Tuple[()]' since forward had ",
        "no arguments.\n",
        pre_hook_err_msg
    );
    // 如果不为空，则直接返回，表示发现了错误
    return;
  }

  // 检查返回的元组类型数量是否与前向传播参数数量匹配
  TORCH_CHECK(
      return_tuple_types.size() == forward_args.size() - 1,
      wrong_type_returned_err_msg,
      " The returned tuple contains the wrong number of contained types.\n",
      pre_hook_err_msg
  );
  // 检查返回的元组内部类型是否与前向传播参数的类型匹配
  for (const auto i : c10::irange(1, forward_args.size())) {
    if (*forward_args[i].type() != *return_tuple_types[i - 1]) {
      // 如果发现类型不匹配，则抛出错误信息
      TORCH_CHECK(
          false,
          wrong_type_returned_err_msg,
          " The returned tuple contains the wrong inner types.\n",
          pre_hook_err_msg);
    }
  }
}

void ClassType::checkForwardHookSchema(
      int hook_idx,
      const FunctionSchema& hook_schema) const {
  const torch::jit::Function* hook = forward_hooks_[hook_idx];
  // 构造钩子标识字符串，指定模块和钩子的名称
  std::string hook_id =
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      "Hook '" + hook->name() + "' on module '" + name()->name() + "' ";
  // 获取前向钩子的错误信息，并添加换行符
  std::string hook_err_msg = getForwardHookErrorMessage(hook_idx) + "\n";
  // 检查钩子期望的输入参数数量是否为3
  // 抛出异常消息，指出实际输入参数数量与预期不符
  TORCH_CHECK(
      hook_schema.arguments().size() == 3,
      hook_id,
      "was expected to only have exactly 3 inputs but it had ",
      hook_schema.arguments().size(),
      " inputs. ",
      hook_err_msg
  );

  // 获取前向方法的函数模式
  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  // 检查前向钩子的输入参数是否匹配前向方法的参数
  checkForwardHookInputArguments(forward_schema, hook_schema, hook_id, hook_err_msg);

  // 检查输出元组
  const Argument& prev_output = (hook_idx == 0)
            ? forward_schema.returns()[0]
            : forward_hooks_[hook_idx - 1]->getSchema().returns()[0];
  const Argument return_arg = hook_schema.arguments()[2];

  // 输出元组的类型必须与前一个输出的类型完全匹配
  // 抛出异常消息，指出实际输出类型与预期不符
  TORCH_CHECK(
      *prev_output.type() == *return_arg.type(),
      hook_id,
      "has the wrong type for the output argument. Received type: '",
      return_arg.type()->annotation_str(),
      "'. Expected type: '",
      prev_output.type()->annotation_str(),
      "'.\n",
      hook_err_msg
  );
}

torch::jit::Function* ClassType::findMethod(const std::string& name) const {
  // 遍历类中的方法列表，寻找指定名称的方法
  for (auto method : methods_) {
    if (name == method->name()) {
      return method;
    }
  }
  return nullptr;
}
torch::jit::Function& ClassType::getMethod(const std::string& name) const {
  // 获取指定名称的方法
  auto method = findMethod(name);
  // 如果找不到方法，抛出异常消息
  TORCH_CHECK(
      method != nullptr,
      "Couldn't find method: '",
      name,
      "' on class: '",
      repr_str(),
      "'");
  return *method;
}

torch::jit::Function* ClassType::findHook(const std::string& name) const {
  // 查找指定名称的前向钩子
  auto hook = findForwardHook(name);
  // 如果前向钩子未找到，尝试查找前向预处理钩子
  if (hook == nullptr) {
    hook = findForwardPreHook(name);
  }
  return hook;
}

torch::jit::Function& ClassType::getHook(const std::string& name) const {
  // 获取指定名称的钩子函数
  torch::jit::Function* function = findHook(name);
  // 如果找不到钩子函数，抛出异常消息
  TORCH_CHECK(
      function != nullptr,
      "Couldn't find: '",
      name,
      "' on class: '",
      repr_str(),
      "'as forward hook or forward pre_hook.");
  return *function;
}

bool ClassType::hasMethod(const std::string& name) const {
  // 检查类中是否存在指定名称的方法
  return findMethod(name) != nullptr;
}

void ClassType::addStaticMethod(torch::jit::Function* method) {
  // 检查静态方法是否已经存在，如果存在则抛出异常消息
  TORCH_CHECK(
      findStaticMethod(method->name()) == nullptr &&
          findMethod(method->name()) == nullptr, "Can't redefine method: ",
      method->name(),
      " on class: ",
      repr_str());
  // 将静态方法添加到静态方法列表中
  staticmethods_.emplace_back(method);
}
// 查找指定名称的静态方法并返回其指针，如果找不到则返回空指针
torch::jit::Function* ClassType::findStaticMethod(const std::string& name) const {
  // 遍历静态方法列表
  for (auto method : staticmethods_) {
    // 如果找到名称匹配的方法，则返回该方法的指针
    if (name == method->name()) {
      return method;
    }
  }
  // 如果未找到匹配的方法，则返回空指针
  return nullptr;
}

// 不安全地移除指定名称的方法
void ClassType::unsafeRemoveMethod(const std::string& name) {
  // 遍历方法列表
  size_t slot = 0;
  for (auto method : methods_) {
    // 如果找到名称匹配的方法，则从列表中移除该方法
    if (method->name() == name) {
      methods_.erase(methods_.begin() + static_cast<std::ptrdiff_t>(slot));
      return;
    }
    slot++;
  }
  // 如果未找到匹配的方法，则抛出异常
  TORCH_CHECK(
      false,
      "Can't delete undefined method ",
      name,
      " on class: ",
      repr_str());
}

// 创建一个新的 ClassType 实例，细化其属性类型
ClassTypePtr ClassType::refine(at::ArrayRef<TypePtr> refined_slots) const {
  // 创建一个与当前 ClassType 相同属性的新实例
  auto ptr = ClassType::create(name(), compilation_unit_, is_module());
  // 断言属性数量与细化类型数组的大小相同
  AT_ASSERT(numAttributes() == refined_slots.size());
  // 遍历属性列表，将每个属性细化为新类型并添加到新实例中
  for (size_t i = 0; i < attributes_.size(); ++i) {
    AT_ASSERT(refined_slots[i]->isSubtypeOf(*attributes_[i].getType()));
    ptr->addAttribute(attributes_[i].getName(), refined_slots[i], (attributes_[i].getKind() == AttributeKind::PARAMETER),
    (attributes_[i].getKind() == AttributeKind::BUFFER));
  }
  // 复制当前实例的方法到新实例中
  for (const auto& method : methods()) {
    ptr->addMethod(method);
  }
  // 返回细化后的新实例
  return ptr;
}

// 判断当前 ClassType 是否是 rhs 类型的子类型，并可选地输出原因到 why_not 流
bool ClassType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  // 如果 rhs 是 AnyClassType 类型，则当前 ClassType 是其子类型
  if (rhs.castRaw<AnyClassType>()) {
    return true;
  }
  // 如果 rhs 是 InterfaceType 类型
  if (auto iface = rhs.cast<InterfaceType>()) {
    // 如果当前 ClassType 不是模块类型而 rhs 是模块接口类型，则不匹配
    if (!is_module() && iface->is_module()) {
      if (why_not) {
        *why_not << "Class '" << repr_str() << "' is not a subtype of "
                 << "the module interface '" << rhs.repr_str()
                 << "' , only ScriptModule class can be subtype of module"
                 << " interface.\n";
      }
      return false;
    }
    // 遍历接口定义的方法列表
    for (const FunctionSchema& schema : iface->methods()) {
      // 查找当前 ClassType 中是否存在对应的方法
      auto self_method = findMethod(schema.name());
      if (!self_method) {
        // 如果找不到对应方法，则不匹配
        if (why_not) {
          *why_not << "Class '" << repr_str() << "' does not have method '"
                   << schema.name() << "' but '" << rhs.repr_str()
                   << "' does.\n";
        }
        return false;
      }
      // 检查当前方法是否兼容接口定义的方法
      if (!self_method->getSchema().isSubtypeOf(
              schema, /*is_method=*/true, why_not)) {
        // 如果不兼容，则不匹配
        if (why_not) {
          *why_not << "Method on class '" << repr_str()
                   << "' (1) is not compatible with interface '"
                   << rhs.repr_str() << "' (2)\n"
                   << "  (1) " << self_method->getSchema() << "\n"
                   << "  (2) " << schema << "\n";
        }
        return false;
      }
    }
    // 所有接口方法都匹配，则当前 ClassType 是 rhs 的子类型
    return true;
  }
  // 调用父类的类型判断方法
  return Type::isSubtypeOfExt(rhs, why_not);
}

// 创建一个新的 ClassType 实例，并返回其指针
ClassTypePtr ClassType::create(
    // 创建一个新的 ClassType 对象并返回其指针
    return ClassTypePtr(new ClassType(
        // 移动传入的 qualifiedName 到新的对象中
        std::move(qualifiedName),
        // 移动传入的 cu 弱指针到新的对象中
        std::move(cu),
        // 将传入的布尔值 is_module 设置为新对象的模块标记
        is_module,
        // 移动传入的 doc_string 到新的对象中作为文档字符串
        std::move(doc_string),
        // 移动传入的 unresolved_class_attributes 到新对象中作为未解析的类属性列表
        std::move(unresolved_class_attributes)));
}

ClassType::ClassType(
    std::optional<QualifiedName> name,  // 可选的类名
    std::weak_ptr<CompilationUnit> cu,  // 弱引用指向编译单元
    bool is_module,  // 是否为模块
    std::string doc_string,  // 文档字符串
    std::vector<std::string> unresolved_class_attributes)  // 未解析的类属性列表
    : NamedType(TypeKind::ClassType, std::move(name)),  // 调用基类构造函数初始化类名类型和名称
      compilation_unit_(std::move(cu)),  // 初始化编译单元
      isModule_(is_module),  // 初始化模块标志
      doc_string_(std::move(doc_string)),  // 初始化文档字符串
      unresolved_class_attributes_(std::move(unresolved_class_attributes)) {}  // 初始化未解析的类属性列表

const std::vector<torch::jit::Function*>& ClassType::methods() const {
  return methods_;  // 返回类的方法列表
}

void ClassType::checkNotExist(const std::string& name, const std::string& what) const {
  // 检查与已有常量名没有重叠
  for (size_t i = 0; i < constantNames_.size(); ++i) {
    TORCH_CHECK(
        name != constantNames_[i],  // 检查名称是否与已有常量名不同
        "attempting to add ",
        what,
        " '",
        name,
        "' to ",
        repr_str(),
        " but a constant field of the same name already exists with value ",
        constantValues_[i]);  // 如果重叠，抛出错误信息
  }

  // 检查与已有属性名没有重叠
  for (const auto & attribute : attributes_) {
    TORCH_CHECK(
        name != attribute.getName(),  // 检查名称是否与已有属性名不同
        "attempting to add ",
        what,
        " '",
        name,
        "' to ",
        repr_str(),
        " but an attribute field of the same name already exists with type ",
        attribute.getType()->repr_str());  // 如果重叠，抛出错误信息
  }
}

void ClassType::addAttribute(ClassAttribute classAttribute) {
    AT_ASSERT(attributes_.size() == attributeTypes_.size());  // 断言属性和属性类型的大小相等
    attributeTypes_.emplace_back(classAttribute.getType());  // 添加属性类型
    attributes_.emplace_back(std::move(classAttribute));  // 添加属性
}

size_t ClassType::addAttribute(
    const std::string& name,  // 属性名称
    TypePtr type,  // 属性类型
    bool is_parameter,  // 是否为参数
    bool is_buffer) {  // 是否为缓冲区
  if (is_parameter && is_buffer){
    TORCH_INTERNAL_ASSERT(false, "Attribute cannot be both a parameter and a buffer!");  // 如果同时是参数和缓冲区，抛出错误信息
  }

  std::string what = is_parameter ? "parameter" : "attribute";  // 属性类型描述：参数或者普通属性
  what += (is_buffer ? "buffer" : "not buffer");  // 如果是缓冲区，添加缓冲区描述
  checkNotExist(name, what);  // 检查是否存在重复的属性名

  size_t slot = attributes_.size();  // 当前属性的插槽位置

  AttributeKind kind = AttributeKind::REGULAR_ATTRIBUTE;  // 默认属性种类为普通属性
  if (is_parameter) {
    kind = AttributeKind::PARAMETER;  // 如果是参数，设置属性种类为参数
  } else if (is_buffer) {
    kind = AttributeKind::BUFFER;  // 如果是缓冲区，设置属性种类为缓冲区
  }

  if (is_parameter || is_buffer) {
    TORCH_INTERNAL_ASSERT(is_module(), "adding a parameter or buffer to a non module");  // 如果添加参数或者缓冲区到非模块，抛出错误信息
    TORCH_CHECK(
        (type->kind() == TensorType::Kind) ||  // 检查类型是否为张量类型或者可选的张量类型
            (type->kind() == OptionalType::Kind &&
             type->expectRef<OptionalType>().getElementType()->kind() ==
                 TensorType::Kind) ||
            (type->kind() == UnionType::Kind &&
             TensorType::get()->isSubtypeOf(type->expectRef<UnionType>())) ||
            (type->kind() == NoneType::Kind),
        "Expecting parameter or buffer to have either None, Tensor or Optional[Tensor] type, but got: ",
        toString(type));  // 检查参数或者缓冲区类型是否符合要求
  }

  addAttribute(ClassAttribute(kind, std::move(type), name));  // 添加属性

  return slot;  // 返回属性的插槽位置
}
// 移除类属性的不安全操作，根据属性名获取其槽位，然后从属性和属性类型列表中删除对应的条目
void ClassType::unsafeRemoveAttribute(const std::string& name) {
  auto slot = getAttributeSlot(name);  // 获取属性名对应的槽位
  attributes_.erase(attributes_.begin() + static_cast<std::ptrdiff_t>(slot));  // 移除属性列表中的对应条目
  attributeTypes_.erase(attributeTypes_.begin() + static_cast<std::ptrdiff_t>(slot));  // 移除属性类型列表中的对应条目
  AT_ASSERT(attributes_.size() == attributeTypes_.size());  // 断言确保属性列表和属性类型列表大小一致
}

// 更改类属性类型的不安全操作，根据属性名获取其槽位，然后用新类型替换原有类型
void ClassType::unsafeChangeAttributeType(const std::string& name, const TypePtr& new_ty) {
  auto slot = getAttributeSlot(name);  // 获取属性名对应的槽位
  auto old_attr_info = attributes_[slot];  // 获取原有属性信息
  AT_ASSERT(old_attr_info.getKind() == AttributeKind::REGULAR_ATTRIBUTE);  // 断言确保属性为常规属性类型
  attributes_[slot] = ClassAttribute(old_attr_info.getKind(), new_ty, old_attr_info.getName());  // 替换属性类型为新类型
  attributeTypes_[slot] = new_ty;  // 更新属性类型列表中的对应条目为新类型
}

// 添加常量到类中，检查是否已存在同名常量，返回添加的常量在列表中的槽位索引
size_t ClassType::addConstant(const std::string& name, const IValue& value) {
  checkNotExist(name, "constant");  // 检查是否已存在同名常量
  size_t slot = constantNames_.size();  // 获取新常量应该插入的槽位索引
  constantNames_.push_back(name);  // 将常量名添加到常量名列表末尾
  constantValues_.push_back(value);  // 将常量值添加到常量值列表末尾
  return slot;  // 返回新常量的槽位索引
}

// 根据常量名获取常量值，如果不存在，则抛出异常
IValue ClassType::getConstant(const std::string& name) const {
  const auto& v = findConstant(name);  // 根据常量名查找常量值
  TORCH_CHECK(
      v.has_value(),
      repr_str(),
      " does not have a constant field with name '",
      name,
      "'");  // 检查常量是否存在，否则抛出异常
  return *v;  // 返回找到的常量值
}

// 根据槽位索引获取常量值，如果索引超出范围，则抛出异常
IValue ClassType::getConstant(size_t slot) const {
  TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());  // 内部断言，确保常量名和常量值列表大小一致
  TORCH_CHECK(
      slot < constantValues_.size(),
      repr_str(),
      " does not have a constant slot of index ",
      slot);  // 检查槽位索引是否有效，否则抛出异常
  return constantValues_[slot];  // 返回指定槽位的常量值
}

// 根据常量名查找常量值，如果不存在则返回空的 std::optional
std::optional<IValue> ClassType::findConstant(const std::string& name) const {
  TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());  // 内部断言，确保常量名和常量值列表大小一致
  size_t pos = 0;  // 初始化位置索引
  for (const auto& c : constantNames_) {  // 遍历常量名列表
    if (name == c) {  // 如果找到匹配的常量名
      break;  // 结束循环
    }
    ++pos;  // 否则增加位置索引
  }

  if (pos >= constantNames_.size()) {  // 如果位置索引超出了常量名列表大小
    return c10::nullopt;  // 返回空的 std::optional
  }
  return constantValues_[pos];  // 返回找到的常量值
}

// 移除类常量的不安全操作，根据常量名获取其槽位，然后从常量名和常量值列表中删除对应的条目
void ClassType::unsafeRemoveConstant(const std::string& name) {
  auto slot = getConstantSlot(name);  // 获取常量名对应的槽位
  constantNames_.erase(constantNames_.begin() + static_cast<std::ptrdiff_t>(slot));  // 移除常量名列表中的对应条目
  constantValues_.erase(constantValues_.begin() + static_cast<std::ptrdiff_t>(slot));  // 移除常量值列表中的对应条目
}

// 获取编译单元的共享指针，用于访问类所属的编译单元
std::shared_ptr<CompilationUnit> ClassType::compilation_unit() {
  auto cu = compilation_unit_.lock();  // 获取编译单元的弱引用指针
  return cu;  // 返回编译单元的共享指针
}

// 获取编译单元的常量共享指针，用于只读访问类所属的编译单元
std::shared_ptr<const CompilationUnit> ClassType::compilation_unit() const {
  auto cu = compilation_unit_.lock();  // 获取编译单元的弱引用指针
  return cu;  // 返回编译单元的常量共享指针
}

// 根据属性名获取属性对象，如果不存在则返回空的 std::optional
std::optional<ClassType::Property> ClassType::getProperty(const std::string& name) {
  for (auto& prop : properties_) {  // 遍历属性列表
    if (name == prop.name) {  // 如果找到匹配的属性名
      return prop;  // 返回属性对象
    }
  }

  return c10::nullopt;  // 否则返回空的 std::optional
}

// 添加属性到类中，包括属性名、getter 函数和 setter 函数，如果属性名已存在则抛出异常
void ClassType::addProperty(const std::string& name, torch::jit::Function* getter, torch::jit::Function* setter) {
  TORCH_INTERNAL_ASSERT(!getProperty(name), "Property named ", name, " already exists!");  // 断言确保属性名不存在
  properties_.push_back({name, getter, setter});  // 添加新属性到属性列表末尾
}
// 在类类型 ClassType 中查找常量名称对应的插槽索引，如果找到则返回其索引值，否则返回空的 std::optional 对象
std::optional<size_t> ClassType::findConstantSlot(const std::string& name) const {
  // 检查常量名称和常量值的数组大小是否一致，以确保数据完整性
  TORCH_CHECK(constantNames_.size() == constantValues_.size());
  // 初始化插槽索引为 0
  size_t slot = 0;
  // 遍历常量名称数组
  for (const auto& constant : constantNames_) {
    // 如果找到与指定名称相匹配的常量名，则返回当前插槽索引值
    if (name == constant) {
      return slot;
    }
    // 否则递增插槽索引
    slot++;
  }
  // 如果未找到匹配的常量名称，则返回空的 std::optional 对象
  return c10::nullopt;
}

// 获取给定插槽索引处的常量名称引用
const std::string& ClassType::getConstantName(size_t slot) const {
  // 再次检查常量名称和常量值的数组大小是否一致，以确保数据完整性
  TORCH_CHECK(constantNames_.size() == constantValues_.size());
  // 检查插槽索引是否有效，即不超出常量名称数组的范围
  TORCH_CHECK(slot < constantNames_.size());
  // 返回指定插槽索引处的常量名称引用
  return constantNames_[slot];
}

// 返回类类型中存储的常量名称数量
size_t ClassType::numConstants() const {
  // 使用内部断言检查常量名称和常量值的数组大小是否一致
  TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
  // 返回常量名称数组的大小，即常量数量
  return constantNames_.size();
}

// 返回类类型中存储的常量值数组的引用
at::ArrayRef<IValue> ClassType::constantValues() const {
  // 直接返回常量值数组的引用
  return constantValues_;
}

// 命名空间 c10 结束
} // namespace c10
```