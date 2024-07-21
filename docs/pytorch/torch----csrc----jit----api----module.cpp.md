# `.\pytorch\torch\csrc\jit\api\module.cpp`

```py
// 引入 ATen 库的符号定义
#include <ATen/core/symbol.h>
// 引入 ATen 库中的记录函数
#include <ATen/record_function.h>
// 引入 C10 库的异常处理工具
#include <c10/util/Exception.h>
// 引入 C10 库的字符串工具
#include <c10/util/StringUtil.h>
// 引入 C10 库的整数范围工具
#include <c10/util/irange.h>
// 引入 Torch 自动求导模块中的变量工厂
#include <torch/csrc/autograd/generated/variable_factories.h>
// 引入 Torch JIT API 中的函数实现
#include <torch/csrc/jit/api/function_impl.h>
// 引入 Torch JIT API 中的模块
#include <torch/csrc/jit/api/module.h>
// 引入 Torch JIT 前端错误报告
#include <torch/csrc/jit/frontend/error_report.h>
// 引入 Torch JIT 前端 IR 发射器
#include <torch/csrc/jit/frontend/ir_emitter.h>
// 引入 Torch JIT 前端模式匹配工具
#include <torch/csrc/jit/frontend/schema_matching.h>
// 引入 Torch JIT 日志记录工具
#include <torch/csrc/jit/jit_log.h>
// 引入 Torch JIT 优化器的死代码消除功能
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 引入 Torch JIT 模块冻结功能
#include <torch/csrc/jit/passes/freeze_module.h>
// 引入 Torch JIT 冻结卷积加ReLU融合功能
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
// 引入 Torch JIT 冻结图优化功能
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
// 引入 Torch JIT 冻结线性转置功能
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
// 引入 Torch JIT 冻结操作到MKLDNN功能
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
// 引入 Torch JIT 内联函数
#include <torch/csrc/jit/passes/inliner.h>
// 引入 Torch JIT 运行时操作符
#include <torch/csrc/jit/runtime/operator.h>

// 引入标准输入输出流库
#include <iostream>

// Torch JIT 命名空间
namespace torch::jit {

// 匿名命名空间，用于定义内部函数或变量，避免全局污染
namespace {

// 获取节点输入的调试名称
std::string getInputDebugName(const Node& n, const int idx) {
  return n.inputs().at(idx)->debugName();
}

// 断言忽略的方法未被调用
void assert_ignored_methods_not_called(
    torch::jit::Function& fn,
    const std::unordered_set<std::string>& ignored_methods) {
  if (ignored_methods.empty()) {
    return;
  }

  // 是否递归查找
  const bool recurse = true;
  // 查找函数图中所有的 CallMethod 节点
  std::vector<Node*> all_nodes = findAllNodes(
      *toGraphFunction(fn).graph(), c10::prim::CallMethod, recurse);

  // 提取这些节点中的方法名称
  std::unordered_set<std::string> encountered_ignored_methods;

  for (Node* n : all_nodes) {
    // 如果方法名称在忽略列表中，并且第一个输入是 "self"
    if (ignored_methods.count(n->s(attr::name)) > 0 &&
        getInputDebugName(*n, 0) == "self") {
      // 记录方法调用信息
      encountered_ignored_methods.insert(
          getInputDebugName(*n, 0) + "." + n->s(attr::name));
    }
  }
  if (encountered_ignored_methods.empty()) {
    return;
  }

  // 构造方法调用信息字符串
  const std::string encountered_ignored_methods_str =
      c10::Join(", ", encountered_ignored_methods);

  // 断言失败，输出错误信息
  TORCH_CHECK(
      false,
      "Preserved method '",
      fn.name(),
      "' references ignored method(s) '",
      encountered_ignored_methods_str,
      "'. This is not permitted.");
}

// 断言忽略的属性未被引用
void assert_ignored_attributes_not_referenced(
    torch::jit::Function& fn,
    const std::unordered_set<std::string>& ignored_attributes) {
  if (ignored_attributes.empty()) {
    return;
  }

  // 是否递归查找
  const bool recurse = true;
  // 查找函数图中所有的 GetAttr 节点
  std::vector<Node*> all_nodes =
      findAllNodes(*toGraphFunction(fn).graph(), c10::prim::GetAttr, recurse);

  // 提取这些节点中的属性名称
  std::unordered_set<std::string> encountered_ignored_attributes;

  for (Node* n : all_nodes) {
    // 如果属性名称在忽略列表中，并且第一个输入是 "self"
    if (ignored_attributes.count(n->s(attr::name)) > 0 &&
        getInputDebugName(*n, 0) == "self") {
      // 记录属性引用信息
      encountered_ignored_attributes.insert(
          getInputDebugName(*n, 0) + "." + n->s(attr::name));
    }
  }
  if (encountered_ignored_attributes.empty()) {
    return;
  }


    // 如果执行到这里，直接返回，结束函数执行
    return;
  }



  const std::string encountered_ignored_attributes_str =
      c10::Join(", ", encountered_ignored_attributes);


  // 将 encountered_ignored_attributes 中的元素用逗号连接成一个字符串
  const std::string encountered_ignored_attributes_str =
      c10::Join(", ", encountered_ignored_attributes);



  TORCH_CHECK(
      false,
      "Preserved method '",
      fn.name(),
      "' references ignored attribute(s) '",
      encountered_ignored_attributes_str,
      "'. This is not permitted.");


  // 检查条件为 false，输出错误信息并抛出异常
  TORCH_CHECK(
      false,
      "Preserved method '",
      fn.name(),
      "' references ignored attribute(s) '",
      encountered_ignored_attributes_str,
      "'. This is not permitted.");
} // namespace

// 创建模块对象的静态函数，返回一个对象指针
static ObjectPtr create_module_object(
    c10::QualifiedName class_name,  // 类名的完全限定名称
    std::shared_ptr<CompilationUnit> cu,  // 共享的编译单元指针
    bool shouldMangle = false) {  // 是否应该进行名称混淆，默认为false

  // 如果类名未限定，添加一个 '__torch__' 前缀，类似于Python中顶层代码的 '__main__'
  if (class_name.prefix().empty()) {
    class_name = c10::QualifiedName("__torch__", class_name.name());
  }

  // 如果应该进行名称混淆并且已经存在该类名对应的类，则进行名称混淆
  if (shouldMangle && cu->get_class(class_name) != nullptr) {
    class_name = cu->mangle(class_name);
  }

  // 创建一个类类型对象，标记为模块类型
  auto cls = ClassType::create(std::move(class_name), cu, /*is_module=*/true);

  // 在编译单元中注册该类型
  cu->register_type(cls);

  // 返回一个值对象指针，其类型为强类型指针，关联着给定的编译单元和类类型
  return c10::ivalue::Object::create(
      c10::StrongTypePtr(std::move(cu), std::move(cls)), 0);
}

// 使用给定的类名创建模块对象的构造函数
Module::Module(c10::QualifiedName class_name)
    : Object(create_module_object(
          std::move(class_name),
          std::make_shared<CompilationUnit>())) {}

// 使用给定的编译单元和类类型创建模块对象的构造函数
Module::Module(
    std::shared_ptr<CompilationUnit> cu,
    const c10::ClassTypePtr& type)
    : Object(c10::ivalue::Object::create(
          c10::StrongTypePtr(std::move(cu), type),
          type->numAttributes())) {}

// 使用给定的类名、编译单元和是否应该进行名称混淆标志创建模块对象的构造函数
Module::Module(
    c10::QualifiedName class_name,
    std::shared_ptr<CompilationUnit> cu,
    bool shouldMangle)
    : Object(create_module_object(
          std::move(class_name),
          std::move(cu),
          shouldMangle)) {}

// first class mode runs models as first class objects,
// and does not force inlining everywhere. This is experimental
// as we bring up the system since it will degrade performance
// and may introduce bugs. test_jit.py provides context managers
// that enable it for specific tests.
// 线程局部变量，控制是否在所有地方进行内联优化的模式
thread_local bool inline_everything = false;

// 获取内联优化模式的引用
bool& getInlineEverythingMode() {
  return inline_everything;
}

// 将模块对象转移到指定设备上的实现函数
void Module::to(at::Device device, at::ScalarType dtype, bool non_blocking) {
  to_impl(device, dtype, non_blocking);
}

// 将模块对象转移到指定数据类型上的实现函数
void Module::to(at::ScalarType dtype, bool non_blocking) {
  to_impl(/*device=*/c10::nullopt, dtype, non_blocking);
}

// 将模块对象转移到指定设备上的实现函数
void Module::to(at::Device device, bool non_blocking) {
  to_impl(device, /*dtype=*/c10::nullopt, non_blocking);
}

// 将模块状态转移到指定设备和数据类型的静态函数
static void module_state_to(
    const autograd::Variable& variable,  // 自动求导变量的引用
    const std::optional<at::Device>& device,  // 可选的设备
    const std::optional<at::ScalarType>& dtype,  // 可选的数据类型
    bool non_blocking) {  // 非阻塞标志

  // 需要将 `at::Tensor` 作为 `Variable` 访问
  // 如果未提供，使用数据的原始设备或数据类型
  auto new_data = variable.to(
      device.value_or(variable.device()),
      dtype.value_or(variable.scalar_type()),
      non_blocking);

  // 设置变量的数据
  variable.set_data(new_data);
}

// 将模块对象转移到指定设备和数据类型的实现函数
void Module::to_impl(
    const std::optional<at::Device>& device,  // 可选的设备
    const std::optional<at::ScalarType>& dtype,  // 可选的数据类型
    bool non_blocking) {  // 非阻塞标志

  // 遍历模块的参数，将其状态转移到指定设备和数据类型
  for (at::Tensor e : parameters()) {
    module_state_to(e, device, dtype, non_blocking);
  }

  // 遍历模块的缓冲区，将其状态转移到指定设备和数据类型
  for (at::Tensor e : buffers()) {
    module_state_to(e, device, dtype, non_blocking);
  }
}

// 方法类的构造函数，接受模块指针和函数指针作为参数
Method::Method(ModulePtr owner, Function* function)
    : owner_(std::move(owner)), function_(function) {}
// 返回当前模块的所有者模块对象
Module Method::owner() const {
  return Module(owner_);
}

// 返回当前方法的原始所有者对象指针
ObjectPtr Method::raw_owner() const {
  return owner_;
}

// 执行当前方法，将所有者对象作为第一个参数插入堆栈，记录 TorchScript 函数调用，然后运行该函数
void Method::run(Stack& stack) {
  stack.insert(stack.begin(), owner()._ivalue()); // self
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);
  function_->run(stack);
}

// 调用当前方法，将所有者对象作为第一个参数插入堆栈，记录 TorchScript 函数调用，然后返回函数调用结果
IValue Method::operator()(std::vector<IValue> stack, const Kwargs& kwargs)
    const {
  stack.insert(stack.begin(), owner()._ivalue()); // self
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);
  return (*function_)(std::move(stack), kwargs);
}

// 异步运行当前方法，将所有者对象作为第一个参数插入堆栈，记录 TorchScript 函数调用，
// 检查和标准化输入参数后，异步运行函数并返回 Future 对象
c10::intrusive_ptr<c10::ivalue::Future> Method::run_async(
    std::vector<IValue> stack,
    const Kwargs& kwargs,
    TaskLauncher taskLauncher) {
  stack.insert(stack.begin(), owner()._ivalue());
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);

  function_->getSchema().checkAndNormalizeInputs(stack, kwargs);
  return function_->runAsync(stack, std::move(taskLauncher));
}

// 设置当前方法的参数名，排除 "self" 参数后，将其余参数名存储到 argumentNamesOut 向量中
void Method::setArgumentNames(
    std::vector<std::string>& argumentNamesOut) const {
  TORCH_INTERNAL_ASSERT(function_);
  auto& arguments = function_->getSchema().arguments();
  argumentNamesOut.reserve(arguments.size());
  for (auto& argument : arguments) {
    if (argument.name() == "self") {
      continue;
    }
    argumentNamesOut.push_back(argument.name());
  }
}

// 调用当前模块对象，执行 forward 阶段，包括调用前向预处理钩子、前向方法以及前向钩子
IValue Module::operator()(std::vector<IValue> inputs) {
  const auto& pre_forward_hooks = type()->getForwardPreHooks();
  const auto& forward_hooks = type()->getForwardHooks();

  // 调用前向预处理钩子
  for (const auto& pre_hook : pre_forward_hooks) {
    auto tuple_input = c10::ivalue::Tuple::create(inputs);
    IValue result = Method(_ivalue(), pre_hook)({tuple_input});
    if (!result.isNone()) {
      if (result.isTuple()) {
        inputs = result.toTupleRef().elements().vec();
      } else {
        inputs = {result};
      }
    }
  }

  // 调用前向方法
  auto outputs = forward(inputs);

  // 调用前向钩子
  for (const auto& hook : forward_hooks) {
    auto tuple_input = c10::ivalue::Tuple::create(inputs);
    auto hook_result = Method(_ivalue(), hook)({tuple_input, outputs});
    if (!hook_result.isNone()) {
      outputs = hook_result;
    }
  }
  return outputs;
}

// 克隆给定模块的特定方法到当前模块中
void Module::clone_method(
    const Module& orig,
    const Function& method,
    const std::string& name) {
    // 定义函数，将一个方法的实现从一个模块单例复制到另一个时，需要更新self参数的类型以匹配新模块。
    // XXX - 这仅处理作为变量出现的模块，而不是出现在聚合类型中的模块。目前这样做是因为我们在降级步骤中限制了模块的使用方式。最终，
    // 我们需要决定什么是“复制”一个模块的含义。例如，我们可以仅复制状态（参数，属性），但共享代码。或者我们可以复制代码本身。
    // 如果选择复制代码，那么对包含模块的聚合类型应该怎么处理呢？
    auto type_remap_fn = [&](TypePtr in) {
      // 查找类型重映射表中是否存在输入类型的映射
      auto it = type_remap.find(in);
      // 如果找不到映射，返回原始输入类型
      if (it == type_remap.end())
        return in;
      // 返回找到的映射后的类型
      return it->second;
    };
    // 将方法转换为图函数，并复制其图形
    auto graph = toGraphFunction(method).graph()->copy();
    // 使用类型重映射函数重新映射图形中的类型
    graph->remapTypes(type_remap_fn);
    // 克隆方法的模式，并使用类型重映射函数进行类型的重新映射
    auto schema = method.getSchema().cloneWithRemappedTypes(type_remap_fn);
    // 获取当前方法的名称
    const auto this_method_name = getNameForMethod(method.name());
    // 在当前对象的编译单元中创建一个新函数，名称为当前方法的名称，使用复制后的图形
    auto copied =
        _ivalue()->compilation_unit()->create_function(this_method_name, graph);
    // 将复制后的方法添加到当前对象的类型中
    type()->addMethod(copied);
    // 设置复制方法的模式
    copied->setSchema(std::move(schema));
// 定义 Module 类的 clone_method 方法，用于从 orig Module 克隆方法到当前 Module 实例，并重命名为 name
void Module::clone_method(const Module& orig, const std::string& name) {
  // 创建类型重映射的无序映射表
  std::unordered_map<TypePtr, TypePtr> type_remap;
  // 准备要扫描的 Module 对象对，初始包含 orig 和当前实例的指针
  std::vector<std::pair<Module, Module>> to_scan = {{orig, *this}};
  // 当待扫描列表非空时循环处理
  while (!to_scan.empty()) {
    // 取出待处理的最后一个条目
    auto entry = to_scan.back();
    to_scan.pop_back();
    // 将 entry.first 的类型映射到 entry.second 的类型
    type_remap[entry.first._ivalue()->type()] = entry.second._ivalue()->type();
    // 遍历 entry.first 的命名子模块
    for (const NameModule& s : entry.first.named_children()) {
      // 将每个子模块添加到扫描列表中，使用当前 Module 对象的相应属性初始化新的 Module 实例
      to_scan.emplace_back(
          s.value, Module(entry.second.attr(s.name).toObject()));
    }
  }
  // 调用另一个重载的 clone_method 方法来克隆指定名称的方法
  return clone_method(orig, orig.get_method(name).function(), type_remap);
}

// 返回当前 Module 的副本，复制其内部 IValue 对象
Module Module::copy() const {
  return Module(_ivalue()->copy());
}

// 返回当前 Module 的深层副本，可以指定设备
Module Module::deepcopy(std::optional<at::Device> device) const {
  return Module(_ivalue()->deepcopy(device));
}

// 返回当前 Module 的克隆，可以选择是否原地克隆
Module Module::clone(bool inplace) const {
  // 初始化类型映射表、IValue memo、忽略方法集合和忽略属性集合
  std::unordered_map<TypePtr, TypePtr> type_remap;
  IValue::HashIdentityIValueMap memo;
  const std::unordered_set<std::string> ignored_methods;
  const std::unordered_set<std::string> ignored_attributes;
  // 调用 clone_impl 方法执行克隆操作
  return clone_impl(
      type_remap, inplace, memo, ignored_methods, ignored_attributes);
}

// 返回当前 Module 的克隆，可以选择是否原地克隆，并指定要忽略的方法集合和属性集合
Module Module::clone(
    bool inplace,
    const std::unordered_set<std::string>& ignored_methods,
    const std::unordered_set<std::string>& ignored_attributes) const {
  // 初始化类型映射表、IValue memo
  std::unordered_map<TypePtr, TypePtr> type_remap;
  IValue::HashIdentityIValueMap memo;
  // 调用 clone_impl 方法执行克隆操作，传递指定的忽略方法集合和属性集合
  return clone_impl(
      type_remap, inplace, memo, ignored_methods, ignored_attributes);
}

// 实际执行 Module 克隆操作的核心方法，接收类型映射表、是否原地克隆标志、IValue memo、忽略方法集合和属性集合
Module Module::clone_impl(
    std::unordered_map<TypePtr, TypePtr>& type_remap,
    bool inplace,
    IValue::HashIdentityIValueMap memo,
    const std::unordered_set<std::string>& ignored_methods,
    const std::unordered_set<std::string>& ignored_attributes) const {
  // 检查当前 Module 的类型是否已经被克隆过
  bool type_already_cloned = type_remap.find(type()) != type_remap.end();
  Module r;
  // 如果类型已经被克隆过，则使用已有的 ClassType 创建新的 Module
  if (type_already_cloned) {
    // 如果之前已经克隆过类类型，则重用它
    Module new_module(
        _ivalue()->compilation_unit(), type_remap[type()]->cast<ClassType>());
    r = new_module;
  } else {
    // 如果类类型还未被克隆过，则创建新的 Module 和新的 ClassType
    Module new_module(*type()->name(), _ivalue()->compilation_unit(), true);
    r = new_module;
    // 记录类型映射关系
    type_remap[type()] = r.type();
  }

  // 复制槽位。如果一个槽位是一个 Module，则递归克隆它。
  size_t N = type()->numAttributes();
  for (const auto i : c10::irange(N)) {
    // 获取当前槽位的 IValue
    IValue s = _ivalue()->getSlot(i);
    // 获取当前槽位对应的属性名
    std::string attr_name = type()->getAttributeName(i);

    // 如果该属性名在被忽略的属性集合中，则跳过不进行克隆
    if (ignored_attributes.count(attr_name) != 0) {
      continue;
    }
    // 获取当前对象的第i个属性类型指针
    TypePtr attr_type = type()->getAttribute(i);
    // 如果属性类型是模块类型
    if (attr_type->is_module()) {
      // 从Python对象s中创建一个Module对象orig
      const Module& orig = Module(s.toObject());
      // 创建一个空的无序字符串集合
      const std::unordered_set<std::string> empty_set;
      // 使用clone_impl方法克隆orig Module对象，返回一个cloned Module对象
      Module cloned =
          orig.clone_impl(type_remap, inplace, memo, empty_set, empty_set);
      // 将原始Module对象的类型映射到克隆Module对象的类型
      type_remap[orig.type()] = cloned.type();
      // 将属性添加到当前对象的类型中，根据属性类型是否为ClassType来决定使用cloned的类型或者原始类型
      r.type()->addOrCheckAttribute(
          attr_name, attr_type->cast<ClassType>() ? cloned.type() : attr_type);
      // 在当前对象的IValue中设置属性attr_name为cloned的IValue
      r._ivalue()->setAttr(attr_name, cloned._ivalue());
    } else {
      // 如果属性类型不是模块类型，则注册新的属性到当前对象
      // 如果非原地操作（inplace为false），则深拷贝s对象，否则直接使用s
      r.register_attribute(
          type()->getAttributeName(i),
          attr_type,
          inplace ? s : s.deepcopy(memo),
          type()->is_parameter(i),
          type()->is_buffer(i));
    }
  }

  // 如果当前类型尚未被克隆过
  if (!type_already_cloned) {
    // 克隆常量
    for (size_t i = 0; i < type()->numConstants(); ++i) {
      // 将当前类型的常量和其对应的名称添加到r对象中
      r.type()->addConstant(type()->getConstantName(i), type()->getConstant(i));
    }
    // 克隆方法，并根据type_remap重映射类型
    for (auto& fn : type()->methods()) {
      // 如果该方法不在被忽略方法列表中，则克隆该方法
      if (ignored_methods.count(fn->name()) == 0) {
        // 断言被忽略的方法没有被调用
        assert_ignored_methods_not_called(*fn, ignored_methods);
        // 断言被忽略的属性没有被引用
        assert_ignored_attributes_not_referenced(*fn, ignored_attributes);
        // 克隆方法fn到r对象中，并使用type_remap重映射类型
        r.clone_method(*this, *fn, type_remap);
      }
    }

    // 执行__setstate__(__getstate__())方法来初始化自定义类成员
    if (auto setstate_method = r.find_method("__setstate__")) {
      // 查找并获取__getstate__方法
      auto getstate_method = r.find_method("__getstate__");
      // 断言__getstate__方法存在
      TORCH_INTERNAL_ASSERT(getstate_method, "expect __getstate__");
      // 调用getstate_method获取状态信息
      auto state = (*getstate_method)(Stack{});
      // 调用setstate_method设置状态信息state
      (*setstate_method)(Stack{state});
    }
  }
  // 返回克隆后的对象r
  return r;
}

void Module::train(bool on) {
  // 遍历当前模块的所有子模块
  for (Module m : modules()) {
    // 查找名为 "training" 的属性槽位
    if (auto slot = m._ivalue()->type()->findAttributeSlot("training")) {
      // 设置 "training" 属性的值为 on
      m._ivalue()->setSlot(*slot, on);
    } else {
      // FIXME[T110620981]: 此断言曾经失效（从未断言过），修复后触发测试失败。请修复我！
      /* TORCH_INTERNAL_ASSERT(false, "'training' attribute not found"); */
    }
  }
}

IValue Module::create_class(const c10::QualifiedName& name, Stack stack) const {
  // 查找指定名称的类
  const auto classType =
      _ivalue()->compilation_unit()->get_class(c10::QualifiedName(name));
  // 如果未找到类，抛出错误
  if (!classType) {
    AT_ERROR(
        "Could not find class with name: '",
        name.qualifiedName(),
        "' in module.");
  }

  // 创建一个空对象，具有正确数量的槽位
  const size_t numAttrs = classType->numAttributes();
  auto obj = c10::ivalue::Object::create(
      c10::StrongTypePtr(_ivalue()->compilation_unit(), classType), numAttrs);

  // 调用类的 `__init__()` 方法，并传递提供的参数
  Stack stackWithSelf = {obj};
  for (auto& arg : stack) {
    stackWithSelf.push_back(std::move(arg));
  }
  // 注意：与 Python 类似，`__init__()` 方法会原地修改其第一个参数，并且不返回任何内容
  classType->getMethod("__init__").operator()(std::move(stackWithSelf));

  return obj;
}

Module freeze(
    const Module& module,
    const std::optional<std::vector<std::string>>& preserved_attrs,
    bool optimize_numerics) {
  // 检查模块不处于训练模式，否则抛出错误
  TORCH_CHECK(
      !module.hasattr("training") || !module.is_training(),
      "Freezing is currently only implemented for modules in eval mode. Please call .eval() before freezing");

  // 冻结模块并返回
  Module out_mod = freeze_module(
      module, preserved_attrs.value_or(std::vector<std::string>({})));
  auto graph = out_mod.get_method("forward").graph();
  // 对冻结后的计算图进行优化
  OptimizeFrozenGraph(graph, optimize_numerics);
  return out_mod;
}

namespace {
void optimize_for_inference(std::shared_ptr<Graph> graph) {
  // 对推断过程中的计算图进行优化
  FuseFrozenConvAddRelu(graph);
  ConvertFrozenOpsToMKLDNN(graph);
  FrozenLinearTranspose(graph);
}
} // namespace

Module optimize_for_inference(
    Module& module,
    const std::vector<std::string>& other_methods) {
  // 如果模块尚未冻结
  Module frozen_mod;
  if (module._ivalue()->type()->hasAttribute("training")) {
    // 对模块进行冻结处理
    frozen_mod = freeze(module, {}, true);
  } else {
    frozen_mod = module;
  }
  // 如果存在名为 "forward" 的方法，则对其计算图进行推断优化
  if (auto method = frozen_mod.find_method("forward")) {
    optimize_for_inference(frozen_mod.get_method("forward").graph());
  }
  // 对其他指定方法的计算图进行推断优化
  for (const auto& method : other_methods) {
    optimize_for_inference(frozen_mod.get_method(method).graph());
  }
  return frozen_mod;
}

buffer_list Module::buffers(bool recurse) const {
  // 返回模块的缓冲区列表，可选择递归包含子模块的缓冲区
  return buffer_list(*this, recurse, /*return_module=*/false);
}
named_buffer_list Module::named_buffers(bool recurse) const {
  // 返回模块的命名缓冲区列表，可选择递归包含子模块的命名缓冲区
  return named_buffer_list(*this, recurse, /*return_module=*/false);
}
module_list Module::children() const {
  // 返回当前模块的子模块列表，不递归获取，并且不返回模块本身
  return module_list(*this, /*recurse=*/false, /*return_module=*/false);
}

named_module_list Module::named_children() const {
  // 返回当前模块的命名子模块列表，不递归获取，并且不返回模块本身
  return named_module_list(*this, /*recurse=*/false, /*return_module=*/false);
}

module_list Module::modules() const {
  // 返回当前模块及其所有子模块的列表，递归获取，并且包含模块本身
  return module_list(*this, /*recurse=*/true, /*return_module=*/true);
}

named_module_list Module::named_modules() const {
  // 返回当前模块及其所有命名子模块的列表，递归获取，并且包含模块本身
  return named_module_list(*this, /*recurse=*/true, /*return_module=*/true);
}

parameter_list Module::parameters(bool recurse) const {
  // 返回当前模块的参数列表，可以选择是否递归获取，并且不返回模块本身
  return parameter_list(*this, recurse, /*return_module=*/false);
}

named_parameter_list Module::named_parameters(bool recurse) const {
  // 返回当前模块的命名参数列表，可以选择是否递归获取，并且不返回模块本身
  return named_parameter_list(*this, recurse, /*return_module=*/false);
}

attribute_list Module::attributes(bool recurse) const {
  // 返回当前模块的属性列表，可以选择是否递归获取，并且不返回模块本身
  return attribute_list(*this, recurse, /*return_module=*/false);
}

named_attribute_list Module::named_attributes(bool recurse) const {
  // 返回当前模块的命名属性列表，可以选择是否递归获取，并且不返回模块本身
  return named_attribute_list(*this, recurse, /*return_module=*/false);
}

void Module::apply(const std::function<void(Module&)>& fn) {
  // 对当前模块的每个子模块应用指定的函数
  for (Module s : modules()) {
    fn(s);
  }
}

std::string Module::dump_to_str(
    bool print_method_bodies,
    bool print_attr_values,
    bool print_param_values) const {
  std::stringstream ss;
  std::stringstream parameters_ss;
  std::stringstream attributes_ss;
  std::stringstream methods_ss;
  std::stringstream submodules_ss;

  // 打印当前模块的参数列表到字符串流中
  for (const NameTensor& p : named_parameters(/*recurse=*/false)) {
    parameters_ss << p.name << " = ";
    if (print_param_values) {
      parameters_ss << p.value << std::endl;
    } else {
      parameters_ss << "..." << std::endl;
    }
  }

  // 打印当前模块的属性列表到字符串流中
  for (const NameValue& p : named_attributes(/*recurse=*/false)) {
    attributes_ss << p.name << " = ";
    if (!p.value.isTensor() || print_attr_values) {
      attributes_ss << p.value << std::endl;
    } else {
      attributes_ss << "..." << std::endl;
    }
  }

  // 获取当前模块的方法列表，并将其打印到字符串流中
  for (const Method& method : get_methods()) {
    methods_ss << "  method " << method.name() << " {" << std::endl;
    if (print_method_bodies) {
      methods_ss << torch::jit::jit_log_prefix(
                        "    ", method.graph()->toString())
                 << std::endl;
    }
    methods_ss << "  }" << std::endl;
  }

  // 构建模块的字符串表示，包括类型名和其包含的参数、属性、方法及子模块的信息
  ss << "module " << type()->name()->qualifiedName() << " {" << std::endl;
  ss << "  parameters {" << std::endl;
  ss << torch::jit::jit_log_prefix("    ", parameters_ss.str());
  ss << "  }" << std::endl;
  ss << "  attributes {" << std::endl;
  ss << torch::jit::jit_log_prefix("    ", attributes_ss.str());
  ss << "  }" << std::endl;
  ss << "  methods {" << std::endl;
  ss << torch::jit::jit_log_prefix("  ", methods_ss.str());
  ss << "  }" << std::endl;
  ss << "  submodules {" << std::endl;
  // 遍历当前模块的命名子模块列表，并将每个子模块的信息打印到字符串流中
  for (const NameModule& s : named_children()) {
    // 缩进4个空格，其中一个来自于'submodules'作用域，另一个来自于当前子模块的特定打印
    // 注释中的说明涉及代码打印的缩进层次的详细信息

    // We do 4 spaces here, because one level of indentation comes from
    // 'submodules' scope and the other one goes from a specific submodule we're
    // printing.
    submodules_ss << "    " << s.name << " {" << std::endl;
    submodules_ss << s.value.dump_to_str(print_method_bodies, print_attr_values, print_param_values);
    submodules_ss << "    }" << std::endl;
  }

  ss << torch::jit::jit_log_prefix("    ", submodules_ss.str());
  ss << "  }" << std::endl;
  ss << "}" << std::endl;

  // 返回模块的完整字符串表示
  return ss.str();
}
    # 将字符串流对象 `ss` 附加带有指定前缀的日志信息
    ss << torch::jit::jit_log_prefix(
        "    ",
        # 将给定的结构体 `s.value` 转储为字符串形式，包括方法体、属性值和参数值，根据参数决定是否打印
        s.value.dump_to_str(
            print_method_bodies, print_attr_values, print_param_values));
    # 在字符串流对象 `ss` 中添加表示结构体结束的字符串
    ss << "  }" << std::endl;
    # 在字符串流对象 `ss` 中添加表示函数结束的字符串
    ss << "}" << std::endl;
    
    # 返回字符串流对象 `ss` 转换为字符串后的结果
    return ss.str();
}

// 定义 Module 类的 dump 方法，打印模块信息到标准输出流
void Module::dump(
    bool print_method_bodies = true,      // 是否打印方法体
    bool print_attr_values = true,        // 是否打印属性值
    bool print_param_values = true) const {// 是否打印参数值
  // 调用 dump_to_str 方法将模块信息转换为字符串并输出到标准输出流
  std::cout << dump_to_str(
                   print_method_bodies, print_attr_values, print_param_values)
            << std::endl;
}

} // namespace torch::jit

// 进入 c10 命名空间

namespace c10 {

// 实现 IValue 类的 toModule 方法，将 IValue 对象转换为 torch::jit::Module
torch::jit::Module IValue::toModule() const {
  return torch::jit::Module(toObject()); // 调用 toObject 方法并返回对应的 Module
}

// 实现 IValue 类的 isModule 方法，判断 IValue 是否为 Module 类型
bool IValue::isModule() const {
  return isObject() && toObjectRef().type()->is_module(); // 判断是否为对象并且对象的类型是模块
}

} // namespace c10
```