# `.\pytorch\torch\csrc\jit\mobile\module.cpp`

```
#include <torch/csrc/jit/mobile/module.h>

#include <torch/csrc/jit/backends/backend_exception.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <exception>

#include <ATen/record_function.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/irange.h>

namespace torch {
namespace jit {
// 声明流输出运算符重载，用于打印指令
std::ostream& operator<<(std::ostream& out, Instruction inst);

namespace mobile {

// 将函数注册到编译单元中
void CompilationUnit::register_function(std::unique_ptr<Function> fn) {
  methods_.emplace_back(std::move(fn));
}

// 根据限定名查找函数，并返回其指针，若不存在则返回 nullptr
const Function* CompilationUnit::find_function(
    const c10::QualifiedName& qn) const {
  for (auto& fn : methods_) {
    if (fn->qualname() == qn) {
      return fn.get();
    }
  }
  return nullptr;
}

// 重载的非常量版本，调用常量版本查找函数
Function* CompilationUnit::find_function(const c10::QualifiedName& qn) {
  // NOLINTNEXTLINE
  return const_cast<Function*>(
      static_cast<const CompilationUnit*>(this)->find_function(qn));
}

// 根据方法名获取方法对象，若找不到则抛出错误
Method Module::get_method(const std::string& name) const {
  if (auto method = find_method(name)) {
    return *method;
  }
  AT_ERROR("Method '", name, "' is not defined.");
}

// 比较两个方法的模式（schema），若相同返回 true，否则返回 false
bool Module::compareMethodSchemas(
    const std::string& name_1,
    const std::string& name_2) {
  std::optional<c10::FunctionSchema> schema_1, schema_2;
  for (const auto& fn : cu_->methods()) {
    if (fn->name() == name_1) {
      schema_1 = fn->getSchema();
    }
    if (fn->name() == name_2) {
      schema_2 = fn->getSchema();
    }
  }
  if (schema_1.has_value() && schema_2.has_value()) {
    return (schema_1 == schema_2);
  }
  return false;
}

// 从模块中不安全地移除指定的方法
void Module::unsafeRemoveMethod(const std::string& basename) {
  int64_t i = 0;
  for (; i < static_cast<int64_t>(cu_->methods().size()); ++i) {
    if ((cu_->methods()[i])->name() == basename) {
      break;
    }
  }
  object_->type()->unsafeRemoveMethod(basename);
  cu_->unsafeRemoveFunction(i);
}

// 不安全地复制指定的方法到新的方法名下
void Module::unsafeCopyMethod(
    const std::string& new_method_name,
    const Function& to_be_copied) {
  TORCH_CHECK(
      !find_method(new_method_name).has_value(),
      "Trying to replace existing method.");
  const c10::QualifiedName& tobe_copied_name = to_be_copied.qualname();
  c10::QualifiedName qualified_method_name(
      tobe_copied_name.prefix(), new_method_name);
  std::unique_ptr<Function> new_fn = std::make_unique<Function>(
      qualified_method_name, to_be_copied.get_code(), to_be_copied.getSchema());
  object_->type()->addMethod(new_fn.get());
  cu_->register_function(std::move(new_fn));
}

// 根据基础名称查找方法，若找到则返回方法对象，否则返回空值
std::optional<Method> Module::find_method(const std::string& basename) const {
  for (const auto& fn : cu_->methods()) {
    if (fn->name() == basename) {
      return c10::make_optional<Method>(Method(this, fn.get()));
    }
  }
  return c10::nullopt;
}

namespace {
// JIT 中用于迭代获取所有模块的私有函数
// 定义一个递归函数，用于设置模块对象及其子对象的'training'属性。
// 如果模块对象包含名为'training'的属性，则设置其值为指定的布尔值。
// 如果模块对象不包含'training'属性，则抛出断言错误。
void set_train_recurse(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj,
    bool on) {
  if (auto slot = obj->type()->findAttributeSlot("training")) {
    obj->setSlot(*slot, on); // 设置'training'属性的值为'on'
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "'training' attribute not found. Did you accidentally "
        "call .eval() before saving your model?");
  }
  for (const auto& slot : obj->slots()) {
    // 遍历模块对象的所有槽位（slots），继续设置'training'属性，
    // 只有当槽位为对象且为模块时才进行递归设置。
    if (slot.isObject() && slot.toObjectRef().type()->is_module()) {
      set_train_recurse(slot.toObject(), on);
    }
  }
}

// 递归函数，收集模块对象及其子对象中的所有张量参数。
void slot_params_recurse(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj,
    std::vector<at::Tensor>* params) {
  for (const auto& slot : obj->slots()) {
    if (slot.isTensor()) {
      params->emplace_back(slot.toTensor()); // 将槽位的张量参数添加到params中
    } else if (slot.isObject()) {
      slot_params_recurse(slot.toObject(), params); // 递归处理子对象
    }
  }
}

// 递归函数，收集模块对象及其子对象中所有具有'requires_grad=True'属性的命名张量参数。
void slot_named_params_recurse(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj,
    std::map<std::string, at::Tensor>* params,
    const std::string& parent_name) {
  auto slots = obj->slots();
  size_t nslots = slots.size();
  for (const auto i : c10::irange(nslots)) {
    auto slot = slots[i];
    std::string name = parent_name.empty() ? parent_name : parent_name + ".";
    name += obj->type()->getAttributeName(i); // 获取属性名作为参数名称的一部分

    // TODO: Fix this filter. Requires_grad is not the appropriate
    // filter of a parameter, but is a temporary hack to help probable
    // users of this api. The correct behavior is to filter by the
    // obj->type->is_parameter() but this currently always returns
    // false on mobile.
    // 检查槽位是否为张量，并且其'requires_grad=True'，将其添加到params中
    if (slot.isTensor() && slot.toTensor().requires_grad()) {
      (*params)[name] = slot.toTensor();
    } else if (slot.isObject()) {
      slot_named_params_recurse(slot.toObject(), params, name); // 递归处理子对象
    }
  }
}

#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
// 获取模块对象的顶级模块类型名称。
std::string getTopModuleTypeName(const Module& m) {
  std::string name;
  if (m._ivalue()->type() && m._ivalue()->type()->name()) {
    name = m._ivalue()->type()->name().value().name();
  }
  return name;
}
#endif

} // namespace

// 返回模块对象中所有参数张量的向量。
const std::vector<at::Tensor> Module::parameters() const {
  std::vector<at::Tensor> params;
  slot_params_recurse(object_, &params); // 收集模块对象中的所有参数张量
  return params;
}

// 返回模块对象中具有'requires_grad=True'属性的所有属性的映射。
// 该行为与完整的torch脚本模块不同。这是一个bug，但目前在加载移动模块时无法正确标记参数。
// TODO
// 返回模块中命名的参数的映射，每个参数表示为字符串到张量的映射
const std::map<std::string, at::Tensor> Module::named_parameters() const {
  // 创建一个空的参数映射
  std::map<std::string, at::Tensor> params;
  // 定义一个空的名称字符串
  const std::string name = "";
  // 递归地收集对象中的命名参数到params映射中
  slot_named_params_recurse(object_, &params, name);
  // 返回收集到的参数映射
  return params;
}

// 返回调试句柄关联的模块层次结构信息字符串
std::string Module::getModuleHierarchy(const int64_t debug_handle) const {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
  // 使用调试表获取给定调试句柄关联的模块层次结构信息
  return getDebugTable().getModuleHierarchyInfo(
      debug_handle, getTopModuleTypeName(*this));
#else
  // 如果未定义SYMBOLICATE_MOBILE_DEBUG_HANDLE，则返回空字符串
  return "";
#endif
}

// 返回调试句柄关联的调用堆栈信息字符串
std::string Module::getCallStack(const int64_t debug_handle) const {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
  // 使用调试表获取给定调试句柄关联的调用堆栈信息
  return getDebugTable().getSourceDebugString(
      debug_handle, getTopModuleTypeName(*this));
#else
  // 如果未定义SYMBOLICATE_MOBILE_DEBUG_HANDLE，则返回空字符串
  return "";
#endif
}

// 返回调试句柄关联的前向方法调试信息字符串
// 用于支持性能分析，当前版本需要继续支持此API
std::string Module::get_forward_method_debug_info(int64_t debug_handle) const {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
  // 使用调试表获取给定调试句柄关联的模块层次结构信息，作为前向方法的调试信息
  return getDebugTable().getModuleHierarchyInfo(
      debug_handle, getTopModuleTypeName(*this));
#else
  // 如果未定义SYMBOLICATE_MOBILE_DEBUG_HANDLE，则返回空字符串
  return "";
#endif
}

// 设置模块的训练状态（开启或关闭）
void Module::train(bool on) {
  // 递归设置对象及其子对象的训练状态
  set_train_recurse(object_, on);
}

// 返回模块当前是否处于训练状态
bool Module::is_training() const {
  // 如果对象具有"training"属性，则返回其布尔值表示的训练状态；否则默认为true
  if (auto slot = object_->type()->findAttributeSlot("training")) {
    return object_->getSlot(*slot).toBool();
  }
  return true;
}

// 返回模块的所有方法列表
const std::vector<Method> Module::get_methods() const {
  // 创建方法列表
  std::vector<Method> methods;
  // 遍历模块的方法列表，并构建Method对象加入到methods中
  for (std::unique_ptr<Function>& fn : cu_->methods()) {
    methods.emplace_back(this, fn.get());
  }
  // 返回方法列表
  return methods;
}

// Method类的构造函数，初始化所有者和函数指针
Method::Method(const Module* owner, Function* function)
    : owner_(owner), function_(function) {}

// 运行方法的核心函数，使用Stack作为输入输出
void Method::run(Stack& stack) const {
  // 获取模块观察器对象
  auto observer = torch::observerConfig().getModuleObserver();
  // 生成随机实例键
  auto instance_key = std::rand();
  // 复制所有者模块的元数据字典
  std::unordered_map<std::string, std::string> copied_metadata =
      owner_->getMetadata();

  // 如果存在观察器对象，调用进入方法运行的回调函数
  if (observer) {
    observer->onEnterRunMethod(instance_key);
  }

  // 创建共享的移动调试信息对象
  auto debug_info = std::make_shared<MobileDebugInfo>();
  // 从复制的元数据中获取模型名称，并设置到调试信息中
  std::string name = copied_metadata["model_name"];
  debug_info->setModelName(name);
  // 设置方法名称到调试信息中
  debug_info->setMethodName(function_->name());
  // 使用调试信息对象设置调试信息的作用域为移动运行时信息
  at::DebugInfoGuard guard(at::DebugInfoKind::MOBILE_RUNTIME_INFO, debug_info);

  // 定义错误消息字符串
  std::string error_message;
  // 定义失败守卫，用于处理异常情况
  auto failure_guard = c10::make_scope_exit([&]() {
    // 如果没有观察器对象，直接返回
    if (!observer) {
      return;
    }

    // 如果定义了SYMBOLICATE_MOBILE_DEBUG_HANDLE且错误消息为空
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
    if (error_message.empty()) {
      // 获取异常调试句柄关联的源调试信息字符串
      error_message = owner_->getDebugTable().getSourceDebugString(
          function_->getExceptionDebugHandles(), getTopModuleTypeName(*owner_));
    }
#endif
  observer->onFailRunMethod(
      copied_metadata,                           // 调用观察器的方法，通知运行失败的事件，传入复制的元数据
      function_->name(),                         // 获取当前函数的名称并传入观察器方法
      instance_key,                              // 传入实例键到观察器方法
      error_message.empty() ? "Unknown exception" : error_message.c_str());  // 判断错误消息是否为空，若为空则传入默认消息，否则传入错误消息的C风格字符串

});

try {
  stack.insert(stack.begin(), owner_->_ivalue()); // 将owner_的值（作为self）插入栈的开头
  function_->run(stack);                         // 运行当前函数，传入栈作为参数
  if (observer) {
    observer->onExitRunMethod(
        copied_metadata,                         // 调用观察器的方法，通知成功运行函数的事件，传入复制的元数据
        function_->name(),                       // 获取当前函数的名称并传入观察器方法
        instance_key);                           // 传入实例键到观察器方法
  }
  failure_guard.release();                       // 释放失败守卫
  // This exception must be caught first as it derived from c10::Error
} catch (c10::BackendRuntimeException& e) {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
    // 如果定义了 SYMBOLICATE_MOBILE_DEBUG_HANDLE 宏，则执行以下代码块
    for (auto handle : function_->getExceptionDebugHandles()) {
      // 遍历函数的异常调试句柄列表，将每个句柄添加到异常对象 e 中
      e.pushDebugHandle(handle);
    }
    // 对所有句柄进行符号化处理
    auto debug_string = owner_->getDebugTable().getSourceDebugString(
        e.getDebugHandles(), getTopModuleTypeName(*owner_));
    // 将符号化的调试信息添加到异常对象 e 的上下文中
    e.add_context(debug_string);
#endif
    // 将异常对象 e 的异常信息保存到 error_message 中
    error_message = e.what();
    // 抛出异常 e
    TORCH_RETHROW(e);
  } catch (c10::Error& error) {
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
    // 如果定义了 SYMBOLICATE_MOBILE_DEBUG_HANDLE 宏，则执行以下代码块
    auto debug_string = owner_->getDebugTable().getSourceDebugString(
        function_->getExceptionDebugHandles(), getTopModuleTypeName(*owner_));
    // 将符号化的调试信息添加到异常对象 error 的上下文中
    error.add_context(debug_string);
#endif
    // 将异常对象 error 的异常信息保存到 error_message 中
    error_message = error.what();
    // 抛出异常 error
    TORCH_RETHROW(error);
  }
}

// Method 类的运算符重载，接收一个包含 c10::IValue 的向量，运行后返回第一个元素
c10::IValue Method::operator()(std::vector<c10::IValue> stack) const {
  // 调用 run 方法，传入栈中的数据
  run(stack);
  // 断言栈不为空
  TORCH_INTERNAL_ASSERT(!stack.empty());
  // 返回栈中的第一个元素
  return stack.front();
}

// 静态函数 print_type，打印类型信息，返回类型的可选字符串
static std::optional<std::string> print_type(const c10::Type& t) {
  // 尝试将类型 t 转换为 NamedType
  auto namedType = t.cast<c10::NamedType>();
  // 如果成功，并且具有名称
  if (namedType && namedType->name()) {
    // 返回命名类型的限定名称
    return namedType->name().value().qualifiedName();
  }
  // 否则，如果是 DynamicType
  if (auto dyn = t.castRaw<c10::DynamicType>()) {
    // 返回动态类型的注释字符串
    return dyn->fallback()->annotation_str();
  }
  // 其他情况返回空
  return c10::nullopt;
}

// 获取 mobile::Module 的模块信息的函数
TORCH_API ModuleInfo get_module_info(const mobile::Module& module) {
  // 创建 ModuleInfo 对象 minfo
  ModuleInfo minfo;
  // 设置操作器版本和字节码版本
  minfo.operator_version = module.min_operator_version();
  minfo.bytecode_version = module.bytecode_version();
  // 创建类型名称列表
  std::vector<std::string> type_name_list;
  // 遍历模块的所有方法
  for (const auto& func_ptr : module.compilation_unit().methods()) {
    const auto& function = *func_ptr;
    // 遍历方法的操作名称列表
    for (const auto i : c10::irange(function.get_code().op_names_.size())) {
      const auto& op = function.get_code().op_names_[i];
      // 将操作名称和对应的输入大小映射到 minfo 的 opname_to_num_args 中
      minfo.opname_to_num_args[mobile::operator_str(op)] =
          function.get_code().operator_input_sizes_[i];
    }
    // 遍历方法的类型列表
    for (const c10::TypePtr& tp : function.get_code().types_) {
      // 将类型的注释字符串添加到类型名称列表中
      type_name_list.push_back(tp->annotation_str(print_type));
    }
    // 将方法的限定名称添加到 minfo 的 function_names 中
    minfo.function_names.insert(function.qualname().qualifiedName());
  }
  // 使用类型名称列表创建 TypeParser 对象 parser
  c10::TypeParser parser(type_name_list);
  // 解析类型列表
  parser.parseList();
  // 将解析后的类型集合保存到 minfo 的 type_names 中
  minfo.type_names = parser.getContainedTypes();
  // 返回填充好信息的 minfo
  return minfo;
}
```