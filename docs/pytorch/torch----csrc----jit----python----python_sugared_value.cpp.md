# `.\pytorch\torch\csrc\jit\python\python_sugared_value.cpp`

```py
// 引入 Torch JIT 的 Python Sugared Value 头文件
#include <torch/csrc/jit/python/python_sugared_value.h>

// 引入 ATen 库的核心头文件
#include <ATen/core/interned_strings.h>
#include <c10/core/ScalarType.h>

// 引入 pybind11 库的类型处理头文件
#include <pybind11/pytypes.h>

// 引入 Torch 的数据类型、布局和内存格式定义头文件
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>

// 引入 Torch JIT 前端的模式匹配头文件
#include <torch/csrc/jit/frontend/schema_matching.h>

// 引入 Torch JIT Python 模块与 Python 交互的头文件
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/utils/pybind.h>

// 引入 C++ 标准库头文件
#include <climits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// 引入 Python.h 头文件
#include <Python.h>

// Torch JIT 的命名空间声明
namespace torch::jit {

// 定义一个函数 typeString，返回 py::handle 对象的类型名字符串
std::string typeString(py::handle h) {
  return py::str(h.get_type().attr("__name__"));
}

// 尝试将给定的 py::object 转换为 StrongFunctionPtr，如果不是，则返回 nullopt
std::optional<StrongFunctionPtr> as_function(const py::object& obj) {
  if (py::isinstance<StrongFunctionPtr>(obj)) {
    return py::cast<StrongFunctionPtr>(obj);
  }
  return c10::nullopt;
}

// 获取 PythonValue 对象的函数模式，根据参数数量、绑定器数量和源代码范围
FunctionSchema PythonValue::getSchema(
    const size_t n_args,
    const size_t n_binders,
    const SourceRange& loc) {
  // 导入 torch.jit.annotations 模块
  auto annotations = py::module::import("torch.jit.annotations");

  // 获取函数的可调用对象，如果有模块自身则获取其原始函数
  const auto callable = moduleSelf_ ? py::getattr(self, "original_fn") : self;

  // 确保函数不是类实例化（例如 `Exception()`）
  annotations.attr("check_fn")(callable, loc);

  // 检查函数是否有可变参数
  auto is_vararg = py::cast<bool>(annotations.attr("is_vararg")(callable));

  // 获取函数的签名
  auto signature = annotations.attr("get_signature")(
      callable, rcb ? *rcb : py::none(), loc, bool(moduleSelf_));
  
  // 初始化参数和返回值的向量
  std::vector<Argument> args, rets;

  // 获取函数参数的名称列表
  auto py_param_names = annotations.attr("get_param_names")(callable, n_args);
  auto param_names = py::cast<std::vector<std::string>>(py_param_names);
  auto names_it = param_names.begin();

  // 如果有模块自身，则添加 `self` 参数到参数列表中
  if (moduleSelf_) {
    if (param_names.empty()) {
      throw ErrorReport(loc)
          << "Non-static method does not have a self argument";
    }
    args.emplace_back(Argument(*names_it, moduleSelf_->type(), {}, {}, false));
    ++names_it;
  }

  // 如果没有提供函数签名，则使用默认签名，所有参数类型为 Tensor
  if (signature.is_none()) {
    for (; names_it != param_names.end(); ++names_it) {
      args.emplace_back(
          /*name=*/*names_it,
          /*type=*/TensorType::get(),
          /*N=*/c10::nullopt,
          /*default_value=*/c10::nullopt,
          /*kwarg_only=*/false);
    }

    // 根据绑定器数量设置返回值类型，如果为 0 则为 NoneType，大于 1 则为 TupleType
    TypePtr ret_type = TensorType::get();
    if (n_binders == 0) {
      ret_type = NoneType::get();
    } else if (n_binders > 1) {
      std::vector<TypePtr> tuple_values(n_binders, ret_type);
      ret_type = TupleType::create(std::move(tuple_values));
    }
    rets.emplace_back(Argument("0", ret_type, {}, {}, false));
  } else {
    // 使用提供的函数签名
    auto [arg_types, ret_type] =
        py::cast<std::pair<std::vector<TypePtr>, TypePtr>>(signature);
    // 这里应该还有代码，但截断了，无法完全显示
  }
}
    // 确保参数类型的数量与参数名列表中的数量匹配，如果存在模块自身参数则需要调整
    TORCH_INTERNAL_ASSERT(
        arg_types.size() == param_names.size() - (moduleSelf_ ? 1 : 0));

    // 创建迭代器以遍历参数类型和参数名
    auto types_it = arg_types.begin();
    for (; types_it != arg_types.end(); ++types_it, ++names_it) {
      // 将参数添加到参数列表中，包括参数名、类型、N、默认值和是否仅限关键字参数
      args.emplace_back(
          /*name=*/*names_it,
          /*type=*/std::move(*types_it),
          /*N=*/c10::nullopt,
          /*default_value=*/c10::nullopt,
          /*kwarg_only=*/false);
    }
    // 将返回值类型添加到返回值列表中
    rets.push_back(Argument("0", std::move(ret_type), {}, {}, false));
  }

  // 初始化函数名称字符串
  std::string name;
  // 如果 Python 对象具有 "__qualname__" 属性，则使用其完全限定名称
  if (py::hasattr(self, "__qualname__")) {
    name = py::str(py::getattr(self, "__qualname__"));
  } else if (py::hasattr(self, "__name__")) {
    // 否则，使用 "__name__" 属性作为函数名称
    name = py::str(py::getattr(self, "__name__"));
  }
  // 返回函数的函数模式对象，包括函数名称、参数列表、返回值列表和是否为可变参数
  return FunctionSchema(name, "", std::move(args), std::move(rets), is_vararg);
}

std::shared_ptr<SugaredValue> PythonValue::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  // 创建一个新的参数向量，将 self 参数添加到开头（如果存在）
  std::vector<NamedValue> argsWithSelf;
  if (moduleSelf_) {
    argsWithSelf.emplace_back("self", moduleSelf_);
  }
  argsWithSelf.insert(argsWithSelf.end(), args.begin(), args.end());

  // 获取当前函数的调用模式（schema）
  auto schema = getSchema(argsWithSelf.size(), n_binders, loc);
  
  // 将输入参数转换为 graph 的值
  auto inputs = toValues(*m.graph(), argsWithSelf);

  // 匹配函数调用的参数和关键字参数，并返回匹配的模式
  MatchedSchema matched_schema =
      matchSchema(schema, loc, *m.graph(), argsWithSelf, kwargs);

  // 如果 Python 函数标记为应该被忽略，抛出异常
  if (py::cast<bool>(py::module::import("torch._jit_internal")
                         .attr("should_drop")(self))) {
    auto g = m.graph();
    auto err_msg = insertConstant(
        *g,
        IValue(
            "This Python function is annotated to be ignored and cannot be run"));
    g->insert(prim::RaiseException, {err_msg}, {}, loc);
    // 返回一个包含未初始化节点输出的 SimpleValue 智能指针
    return std::make_shared<SimpleValue>(
        g->insertNode(g->createUninitialized(matched_schema.return_types.at(0)))
            ->output());
  }

  // 释放函数对象，以便将其包装在 PythonOp 中
  py::object func = self;
  std::string cconv(inputs.size(), 'd');
  // 在图中插入一个 PythonOp 节点，代表 Python 函数调用
  Node* new_node = m.graph()->insertNode(
      m.graph()->createPythonOp(THPObjectPtr(func.release().ptr()), cconv, {}));

  // 设置新节点的源范围，并添加匹配模式的输入
  new_node->setSourceRange(loc);
  for (auto& i : matched_schema.inputs)
    new_node->addInput(i);

  // 添加一个输出值，并设置其类型为匹配模式的返回类型之一
  Value* output =
      new_node->addOutput()->setType(matched_schema.return_types.at(0));
  
  // 返回一个包含输出值的 SimpleValue 智能指针
  return std::make_shared<SimpleValue>(output);
}

// 返回 PythonValue 类型的对象的字符串表示形式
std::string PythonValue::kind() const {
  std::stringstream ss;
  ss << "python value of type '" << typeString(self) << "'";
  return ss.str();
}

// 将 PythonValue 对象转换为元组时抛出异常
std::vector<std::shared_ptr<SugaredValue>> PythonValue::asTuple(
    const SourceRange& loc,
    GraphFunction& m,
    const std::optional<size_t>& size_hint) {
  const std::string type_str = typeString(self);
  std::stringstream ss;
  ss << kind() << " cannot be used as a tuple";
  checkForAddToConstantsError(ss);
  throw ErrorReport(loc) << ss.str();
}

// 获取 PythonValue 对象的属性时抛出异常
std::shared_ptr<SugaredValue> PythonValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  const std::string type_str = typeString(self);
  std::stringstream ss;
  ss << "attribute lookup is not defined on " << kind();
  checkForAddToConstantsError(ss);
  throw ErrorReport(loc) << ss.str();
}

// 获取 Python 对象的指定属性，若属性不存在则抛出异常
py::object PythonValue::getattr(
    const SourceRange& loc,
    const std::string& name) {
  try {
    return py::getattr(self, name.c_str());
  } catch (py::error_already_set& e) {
    throw ErrorReport(loc) << "object has no attribute " << name;
  }
}
// 检查是否需要将当前对象添加到常量列表中，以避免错误
void PythonValue::checkForAddToConstantsError(std::stringstream& ss) {
  // 导入 torch.nn 模块
  auto nn = py::module::import("torch.nn");
  // 检查当前对象是否是 nn.ModuleList 或 nn.Sequential 的实例
  if (py::isinstance(self, nn.attr("ModuleList")) ||
      py::isinstance(self, nn.attr("Sequential"))) {
    // 如果是，则向错误消息流中添加提示信息
    ss << ". Did you forget to add it to __constants__? ";
  }
}

// 获取 Python 模块的属性值，并转换为 SugaredValue
std::shared_ptr<SugaredValue> PythonModuleValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // 获取指定字段的 Python 对象
  py::object member = getattr(loc, field);
  // 注意：因为我们认为模块上的全局属性（如 math.pi 或 torch.float）是常量，
  // 即使有人可能会修改它们，我们仍将其视为常量
  return toSugaredValue(member, m, loc, /*is_constant=*/true);
}

// 获取 CUDA Python 模块的属性值，并转换为 SugaredValue
std::shared_ptr<SugaredValue> CUDAPythonModuleValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // 支持在 JIT 中的所有 CUDA 操作符的列表
  const std::unordered_set<std::string> cuda_ops = {
      "current_stream",
      "default_stream",
      "current_device",
      "_exchange_device",
      "_maybe_exchange_device",
      "set_device",
      "device_index",
      "device_count",
      "set_stream",
      "synchronize"};

  // 如果请求的字段在 CUDA 操作符列表中
  if (cuda_ops.find(field) != cuda_ops.end()) {
    // 对于 current_device 和 set_device API，由于它们属于 c10::cuda 命名空间，
    // 为了解决 JIT 中的冲突，我们在它们之前添加下划线 _
    if (field == "current_device" || field == "set_device") {
      return std::make_shared<BuiltinFunction>(
          Symbol::cuda("_" + field), c10::nullopt);
    } else {
      return std::make_shared<BuiltinFunction>(
          Symbol::cuda(field), c10::nullopt);
    }
  }

  // 如果请求的字段是 "Stream" 或 "Event"
  if (field == "Stream" || field == "Event") {
    // 获取自定义类类型
    auto class_type = getCustomClass("__torch__.torch.classes.cuda." + field);
    return std::make_shared<ClassValue>(class_type);
  }

  // 获取指定字段的 Python 对象
  py::object member = getattr(loc, field);
  // 注意：因为我们认为模块上的全局属性（如 math.pi 或 torch.float）是常量，
  // 即使有人可能会修改它们，我们仍将其视为常量
  return toSugaredValue(member, m, loc, /*is_constant=*/true);
}

// 将 ModuleValue 转换为 Value 类型
Value* ModuleValue::asValue(const SourceRange& loc, GraphFunction& m) {
  return self_;
}

// 将 ModuleValue 转换为 TupleValue 类型
SugaredValuePtr ModuleValue::asTupleValue(
    const SourceRange& loc,
    GraphFunction& m) {
  // 如果具体类型是列表类型的迭代模块
  if (concreteType_->getIterableModuleKind() == IterableModuleKind::LIST) {
    // 获取 SugaredDict，并返回其中的模块列表
    auto dict = getSugaredDict(loc, m);
    auto mods = dict->getModules();
    return mods;
  }
  // 抛出错误，仅允许 ModuleList 或 Sequential 模块作为元组使用
  throw ErrorReport(loc)
      << "Only ModuleList or Sequential modules can be used as tuple";
}

// 检查所有子模块是否都是给定类型的子类型
bool ModuleValue::areAllSubmodulesSubtypeOf(
    const TypePtr& ty,
    std::ostream* why_not) const {
  // 获取具体类型的 ClassType
  const auto& self_type = concreteType_->getJitType()->expect<ClassType>();
  // 遍历所有属性
  for (size_t i = 0; i < self_type->numAttributes(); ++i) {
    const auto& attr_type = self_type->getAttribute(i);
    if (attr_type->is_module()) {
      // 检查属性类型是否为模块类型
      std::stringstream ss;
      // 创建一个字符串流，用于存储错误信息
      if (!attr_type->isSubtypeOfExt(ty, &ss)) {
        // 如果属性类型不是指定类型的子类型，则执行以下操作
        if (why_not) {
          // 如果提供了错误信息输出流
          *why_not << "Attribute " << self_type->getAttributeName(i)
                   << " is not of annotated type " << ty->annotation_str()
                   << ": " << ss.str();
          // 输出属性名称、期望类型注解和具体错误信息到错误信息流
        }

        return false;
        // 返回 false 表示验证失败
      }
    }
  }

  return true;
  // 如果所有属性都通过验证，则返回 true
    }
    // 根据索引获取模块值
    SugaredValuePtr ModuleValue::getitem(
        const SourceRange& loc,
        GraphFunction& m,
        Value* idx,
        TypePtr type_hint) {
      // 检查具体类型是否为列表
      if (concreteType_->getIterableModuleKind() == IterableModuleKind::LIST) {
        // 如果存在类型提示
        if (type_hint) {
          // 检查所有子模块是否符合类型提示
          std::stringstream ss;
          if (!areAllSubmodulesSubtypeOf(type_hint, &ss)) {
            // 抛出错误报告，显示类型不符合的信息
            throw ErrorReport(loc) << ss.str();
          }

          // 在图中插入 prim::ModuleContainerIndex 操作符，用于获取列表中的元素
          auto graph = m.graph();
          auto* getitem_node = graph->insertNode(
              graph->create(prim::ModuleContainerIndex, {self_, idx}));
          getitem_node->output(0)->setType(type_hint);
          // 返回简单值的共享指针
          return std::make_shared<SimpleValue>(getitem_node->output(0));
        } else {
          // 如果没有类型提示，则调用 getSugaredDict 获取字典并获取索引处的模块值
          return getSugaredDict(loc, m)->getModules()->getitem(
              loc, m, idx, type_hint);
        }
      } else if (
          concreteType_->getIterableModuleKind() == IterableModuleKind::PARAMLIST) {
        // 如果具体类型为参数列表，则调用 getSugaredNamedParameterList 获取参数列表并获取索引处的模块值
        return getSugaredNamedParameterList(loc, m)->getModules()->getitem(
            loc, m, idx, type_hint);
      } else if (
          concreteType_->getIterableModuleKind() == IterableModuleKind::DICT ||
          concreteType_->getIterableModuleKind() == IterableModuleKind::PARAMDICT) {
        // 如果具体类型为字典或参数字典
        if (auto ivalue = toIValue(idx)) {
          std::shared_ptr<SugaredDict> sd;
          // 根据具体类型选择合适的 getSugaredDict 函数
          if (concreteType_->getIterableModuleKind() == IterableModuleKind::DICT) {
            sd = getSugaredDict(loc, m);
          } else if (
              concreteType_->getIterableModuleKind() ==
              IterableModuleKind::PARAMDICT) {
            sd = getSugaredNamedParameterDict(loc, m);
          }
          auto idx_str = ivalue->toStringRef();
          auto keys_iter = sd->keys_;
          auto module_values_iter = sd->modules_;
          // 遍历字典的键和模块值，寻找匹配的键
          for (size_t i = 0; i < keys_iter->tup_.size(); ++i) {
            auto key = keys_iter->tup_.at(i);
            auto key_str = toIValue(key->asValue(loc, m))->toStringRef();
            if (key_str == idx_str) {
              // 返回匹配的模块值
              return module_values_iter->tup_.at(i);
            }
          }
          // 若未找到匹配的键，则抛出错误报告
          throw ErrorReport(loc) << "Key Error, " << idx_str;
    } else if (type_hint) {
      // 如果存在类型提示，则检查所有子模块是否符合类型提示要求。
      std::stringstream ss;
      // 如果有子模块不符合类型提示要求，将错误信息收集到字符串流中。
      if (!areAllSubmodulesSubtypeOf(type_hint, &ss)) {
        // 如果存在不符合类型提示的子模块，抛出错误报告并附上错误信息。
        throw ErrorReport(loc) << ss.str();
      }

      // 发出一个 prim::ModuleContainerIndex 操作。这是因为在图中构造 dict
      // 表示的 ModuleDict 并使用 aten::__getitem__ 操作进行索引很困难，
      // 因为任何对 ModuleDict.setAttr 的调用都会使发出的 dict 失效。
      auto graph = m.graph();
      // 插入一个 prim::ModuleContainerIndex 节点来进行操作。
      auto* getitem_node = graph->insertNode(
          graph->create(prim::ModuleContainerIndex, {self_, idx}));
      // 设置输出节点的类型为 type_hint。
      getitem_node->output(0)->setType(type_hint);
      // 返回一个包含 getitem_node 输出的 SimpleValue 共享指针。
      return std::make_shared<SimpleValue>(getitem_node->output(0));
    }
    // 如果没有匹配的条件分支，则抛出错误报告，指出无法提取字符串文字索引的原因。
    throw ErrorReport(loc)
        << "Unable to extract string literal index. "
        << "ModuleDict indexing is only supported with string literals. "
        << "For example, 'i = \"a\"; self.layers[i](x)' will fail because i is not a literal. "
        << "Enumeration of ModuleDict is supported, e.g. 'for k, v in self.items(): out = v(inp)'";
  }
  // 如果没有匹配的条件分支，则抛出错误报告，说明只有特定类型的模块支持下标操作。
  throw ErrorReport(loc)
      << "Only ModuleList, Sequential, ModuleDict, "
      << "ParameterList, and ParameterDict modules are subscriptable";
void checkInterface(
    const SourceRange& loc,                          // 函数参数：源代码范围
    GraphFunction& m,                               // 函数参数：图函数引用
    const std::shared_ptr<ModuleValue>& self,        // 函数参数：模块值的共享指针
    const std::string& field) {                      // 函数参数：字段名称
  // 检查模块值是否为接口类型，如果是则抛出错误报告
  if (self->asValue(loc, m)->type()->cast<InterfaceType>()) {
    throw ErrorReport(loc)
        << "Could not compile " << field
        << "() because module is an interface type. Please file issue.";
  }
}

void recurseThroughNestedModules(
    const SourceRange& loc,                          // 函数参数：源代码范围
    GraphFunction& m,                               // 函数参数：图函数引用
    std::vector<SugaredValuePtr>& keys,              // 引用参数：键的糖值指针向量
    std::vector<SugaredValuePtr>& values,            // 引用参数：值的糖值指针向量
    std::shared_ptr<ModuleValue>& self,              // 函数参数：模块值的共享指针引用
    const std::string& prefix,                       // 函数参数：前缀字符串
    const std::string& field) {                      // 函数参数：字段名称
  // 创建一个表示前缀的简单值的共享指针
  auto prefix_value =
      std::make_shared<SimpleValue>(insertConstant(*m.graph(), prefix));

  // 将前缀值和当前模块值添加到键和值的向量中
  keys.push_back(prefix_value);
  values.push_back(self);

  // 检查模块值是否为接口类型
  checkInterface(loc, m, self, field);

  // 获取当前模块值的糖化字典
  auto module_dict = self->getSugaredDict(loc, m);
  auto keys_iter = module_dict->keys_;
  auto module_values_iter = module_dict->modules_;

  // 遍历模块字典中的键和模块值，递归处理嵌套模块
  for (size_t i = 0; i < keys_iter->tup_.size(); ++i) {
    // 获取模块值的糖化值，并尝试将其转换为模块值
    std::shared_ptr<SugaredValue> module_sugared_value =
        module_values_iter->tup_.at(i);
    auto module_value =
        std::dynamic_pointer_cast<ModuleValue>(module_sugared_value);

    // 获取当前键值的字面字符串表示
    auto keys_value = keys_iter->tup_.at(i);
    auto key_string = toIValue(keys_value->asValue(loc, m))->toStringRef();

    // 构建子模块的前缀字符串
    std::string submodule_prefix = prefix;
    if (!prefix.empty()) {
      submodule_prefix = prefix + ".";
    }
    submodule_prefix += key_string;

    // 递归调用处理嵌套模块
    recurseThroughNestedModules(
        loc, m, keys, values, module_value, submodule_prefix, field);
  }
}

std::shared_ptr<SugaredDict> ModuleValue::getSugaredNamedBufferDict(
    const SourceRange& loc,                          // 函数参数：源代码范围
    GraphFunction& m) {                              // 函数参数：图函数引用
  // 存储模块参数名称和糖值指针向量
  std::vector<std::string> paramNames;
  std::vector<SugaredValuePtr> values;

  // 获取具体类型的类类型，遍历其属性
  const auto& selfType = concreteType_->getJitType()->expect<ClassType>();
  for (size_t i = 0; i < selfType->numAttributes(); ++i) {
    // 如果属性是缓冲区，则将其名称添加到参数名称向量中
    if (selfType->is_buffer(i)) {
      paramNames.push_back(selfType->getAttributeName(i));
    }
  }

  // 存储参数名称的简单值和尝试获取属性的值
  std::vector<SugaredValuePtr> keys;
  for (const auto& name : paramNames) {
    auto name_v =
        std::make_shared<SimpleValue>(insertConstant(*m.graph(), name));
    m.graph()->insertGetAttr(self_, name);
    values.push_back(tryGetAttr(loc, m, name));
    keys.push_back(name_v);
  }

  // 返回一个新的糖化字典，包含模块值、键和值的糖化元组值
  return std::make_shared<SugaredDict>(
      std::make_shared<ModuleValue>(self_, concreteType_),
      std::make_shared<SugaredTupleValue>(keys),
      std::make_shared<SugaredTupleValue>(values));
}

std::shared_ptr<SugaredDict> ModuleValue::getSugaredNamedParameterList(
    const SourceRange& loc,                          // 函数参数：源代码范围
    GraphFunction& m) {                              // 函数参数：图函数引用
  // 存储模块参数名称和糖值指针向量
  std::vector<std::string> paramNames;
  std::vector<SugaredValuePtr> values;

  // 获取具体类型的类类型，遍历其属性
  const auto& selfType = concreteType_->getJitType()->expect<ClassType>();
  for (size_t i = 0; i < selfType->numAttributes(); ++i) {
    // 如果属性是参数，则将其名称添加到参数名称向量中
    if (selfType->is_parameter(i)) {
      paramNames.push_back(selfType->getAttributeName(i));
    }
  }

  // 存储参数名称的简单值和尝试获取属性的值
  std::vector<SugaredValuePtr> keys;
  for (const auto& name : paramNames) {
    auto name_v =
        std::make_shared<SimpleValue>(insertConstant(*m.graph(), name));
    m.graph()->insertGetAttr(self_, name);
    values.push_back(tryGetAttr(loc, m, name));
    keys.push_back(name_v);
  }

  // 返回一个新的糖化字典，包含模块值、键和值的糖化元组值
  return std::make_shared<SugaredDict>(
      std::make_shared<ModuleValue>(self_, concreteType_),
      std::make_shared<SugaredTupleValue>(keys),
      std::make_shared<SugaredTupleValue>(values));
}
  }
}



// 结束了一对大括号的作用域，可能是某个代码块的结尾
std::vector<SugaredValuePtr> keys;
// 创建一个空的 vector，用于存储参数名对应的 SugaredValue 指针
for (const auto& name : paramNames) {
  // 遍历参数名列表，对每个参数名创建一个 SimpleValue，并插入到 keys 中
  auto name_v = std::make_shared<SimpleValue>(insertConstant(*m.graph(), name));
  // 在当前图中获取 self_ 对象的属性名为 name 的值
  m.graph()->insertGetAttr(self_, name);
  // 尝试获取当前对象 self_ 的属性名为 name 的值，并将其添加到 values 中
  values.push_back(tryGetAttr(loc, m, name));
  // 将当前参数名对应的 SimpleValue 指针 name_v 添加到 keys 中
  keys.push_back(name_v);
}

// 返回一个 SugaredDict 对象，其中包含 self_ 对象所属模块的 ModuleValue、keys 的 SugaredTupleValue、values 的 SugaredTupleValue
return std::make_shared<SugaredDict>(
    std::make_shared<ModuleValue>(self_, concreteType_),
    std::make_shared<SugaredTupleValue>(keys),
    std::make_shared<SugaredTupleValue>(values));
}

std::shared_ptr<SugaredDict> ModuleValue::getSugaredDict(
    const SourceRange& loc,
    GraphFunction& m) {
  // 初始化空的子模块名称列表
  std::vector<std::string> submoduleNames;
  // 获取模块的具体类型
  const auto& selfType = concreteType_->getJitType()->expect<ClassType>();
  // 遍历模块的所有属性
  for (size_t i = 0; i < selfType->numAttributes(); ++i) {
    // 获取属性类型
    const auto& attrType = selfType->getAttribute(i);
    // 如果属性类型是模块，则将属性名称添加到子模块名称列表中
    if (attrType->is_module()) {
      submoduleNames.push_back(selfType->getAttributeName(i));
    }
  }

  // 初始化键值对列表
  std::vector<SugaredValuePtr> keys;
  std::vector<SugaredValuePtr> values;
  // 遍历子模块名称列表
  for (const auto& name : submoduleNames) {
    // 创建表示子模块名称的简单值
    auto name_v =
        std::make_shared<SimpleValue>(insertConstant(*m.graph(), name));
    // 在图中插入获取属性操作，获取子模块的值
    Value* module_v = m.graph()->insertGetAttr(self_, name);
    // 创建表示子模块的模块值
    auto mod_v = std::make_shared<ModuleValue>(
        module_v, concreteType_->findSubmoduleConcreteType(name));

    // 将键和值添加到相应的列表中
    keys.push_back(name_v);
    values.push_back(mod_v);
  }

  // 创建并返回表示字典的模块值
  return std::make_shared<SugaredDict>(
      std::make_shared<ModuleValue>(self_, concreteType_),
      std::make_shared<SugaredTupleValue>(keys),
      std::make_shared<SugaredTupleValue>(values));
}

std::shared_ptr<SugaredDict> ModuleValue::getSugaredNamedParameterDict(
    const SourceRange& loc,
    GraphFunction& m) {
  // 初始化空的参数名称列表
  std::vector<std::string> paramNames;
  // 获取模块的具体类型
  const auto& selfType = concreteType_->getJitType()->expect<ClassType>();
  // 遍历模块的所有属性
  for (size_t i = 0; i < selfType->numAttributes(); ++i) {
    // 如果属性是参数，则将属性名称添加到参数名称列表中
    if (selfType->is_parameter(i)) {
      paramNames.push_back(selfType->getAttributeName(i));
    }
  }

  // 初始化键值对列表
  std::vector<SugaredValuePtr> keys;
  std::vector<SugaredValuePtr> values;
  // 遍历参数名称列表
  for (const auto& name : paramNames) {
    // 创建表示参数名称的简单值
    auto name_v =
        std::make_shared<SimpleValue>(insertConstant(*m.graph(), name));
    // 在图中插入获取属性操作，获取参数的值
    m.graph()->insertGetAttr(self_, name);
    // 尝试获取参数的属性值
    auto val = tryGetAttr(loc, m, name);
    // 断言确保成功获取参数的属性值
    TORCH_INTERNAL_ASSERT(val != nullptr, "Could not find attribute ", name);
    // 将参数的属性值添加到值列表中
    values.push_back(val);
    // 将参数名称的简单值添加到键列表中
    keys.push_back(name_v);
  }

  // 创建并返回表示命名参数字典的模块值
  return std::make_shared<SugaredDict>(
      std::make_shared<ModuleValue>(self_, concreteType_),
      std::make_shared<SugaredTupleValue>(keys),
      std::make_shared<SugaredTupleValue>(values));
}

std::shared_ptr<SugaredValue> SugaredDict::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // 递归编译不会保持模块别名，因此不会在 "children"/"named_children"/"modules"/"named_modules" 上添加唯一性检查
  // 检查接口，确保字段的有效性
  checkInterface(loc, m, self_, field);
  // 根据字段名称返回对应的方法或值
  if (field == "keys") {
    return std::make_shared<ModuleDictMethod>(keys_, "keys");
  } else if (field == "values" || field == "children") {
    return std::make_shared<ModuleDictMethod>(modules_, field);
  } else if (
      field == "items" || field == "named_children" ||
      field == "named_buffers") {
    // 创建可迭代树并返回对应的方法
    auto iterator = std::make_shared<IterableTree>();
    iterator->addChild(loc, m, keys_);
    iterator->addChild(loc, m, modules_);
    return std::make_shared<ModuleDictMethod>(iterator, field);
  }
  // 如果字段为 "named_modules" 或 "modules"，则进行以下操作
  } else if (field == "named_modules" || field == "modules") {
    // 创建空的键和值向量
    std::vector<SugaredValuePtr> keys;
    std::vector<SugaredValuePtr> values;
    // 递归遍历嵌套模块，并填充键和值向量
    recurseThroughNestedModules(loc, m, keys, values, self_, "", field);
    
    // 如果字段为 "modules"
    if (field == "modules") {
      // 返回模块字典方法，接受值向量的元组作为参数
      return std::make_shared<ModuleDictMethod>(
          std::make_shared<SugaredTupleValue>(values), field);
    } else {
      // 否则，创建可迭代树对象
      auto iterator = std::make_shared<IterableTree>();
      // 将键和值向量作为子节点添加到可迭代树对象中
      iterator->addChild(loc, m, std::make_shared<SugaredTupleValue>(keys));
      iterator->addChild(loc, m, std::make_shared<SugaredTupleValue>(values));
      // 返回模块字典方法，接受可迭代树对象和字段作为参数
      return std::make_shared<ModuleDictMethod>(iterator, field);
    }
  }
  // 如果以上条件都不满足，则抛出内部断言错误
  TORCH_INTERNAL_ASSERT(false);
}

// 从给定的 Python 对象创建一个 SugaredEnumClass 对象
std::shared_ptr<SugaredEnumClass> createSugaredEnumClassFromObj(
    const py::object& obj,
    GraphFunction& m,
    const SourceRange& loc) {
  // 导入 torch.jit.annotations 模块并调用 try_ann_to_type 函数，获取注解类型
  auto annotation_type = py::module::import("torch.jit.annotations")
                             .attr("try_ann_to_type")(obj, loc);
  // 断言注解类型不为空
  TORCH_INTERNAL_ASSERT(!annotation_type.is_none());
  // 将注解类型转换为 TypePtr，并期望其为 EnumType 类型
  auto type = py::cast<TypePtr>(annotation_type);
  auto enum_type = type->expect<EnumType>();
  // 使用 EnumType 创建并返回 SugaredEnumClass 对象
  return std::make_shared<SugaredEnumClass>(enum_type);
}

// 从 IValue 创建 SugaredValue 的辅助函数
std::shared_ptr<SugaredValue> toSugaredValue(
    const IValue& v,
    GraphFunction& m,
    const SourceRange& loc) {
  // 如果 IValue 是元组，则处理每个元素并返回对应的 SugaredValue
  if (v.isTuple()) {
    auto tp = v.toTuple();
    std::vector<Value*> values;
    values.reserve(tp->elements().size());
    // 遍历元组的每个元素，将其转换为 SugaredValue
    for (const auto& e : tp->elements()) {
      values.push_back(toSugaredValue(e, m, loc)->asValue(loc, m));
    }
    // 创建表示元组的 SimpleValue 并返回其值
    return toSimple(
        m.graph()->insertNode(m.graph()->createTuple(values))->output());
  } else {
    // 将 IValue 直接插入图中并返回对应的 SimpleValue
    return toSimple(m.graph()->insertConstant(v, loc));
  }
}

// 控制在 ScriptModules 上进行属性查找时的解糖方法
std::shared_ptr<SugaredValue> ModuleValue::tryGetAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // 1. 在 Module 对象中查找字段
  const auto& selfType_ = concreteType_->getJitType();
  // 如果 selfType 是 InterfaceType，则作为 SimpleValue 处理并返回其属性
  if (selfType_->cast<InterfaceType>()) {
    return std::make_shared<SimpleValue>(self_)->attr(loc, m, field);
  }

  const auto& selfType = selfType_->expect<ClassType>();

  // 如果 selfType 中有指定字段且为模块类型，返回新的 ModuleValue
  if (selfType->hasAttribute(field) &&
      selfType->getAttribute(field)->is_module()) {
    if (const auto submoduleConcreteType =
            concreteType_->findSubmoduleConcreteType(field)) {
      return std::make_shared<ModuleValue>(
          m.graph()->insertGetAttr(self_, field), submoduleConcreteType);
    }

    return std::make_shared<ModuleValue>(
        m.graph()->insertGetAttr(self_, field),
        ConcreteModuleType::fromJitType(selfType->getAttribute(field)));
  } else if (selfType->hasAttribute(field) || selfType->findMethod(field)) {
    // 否则，如果字段是方法、参数、属性或缓冲区，作为 SimpleValue 处理返回
    return std::make_shared<SimpleValue>(self_)->attr(loc, m, field);
  } else if (selfType->hasConstant(field)) {
    // 如果字段是常量，则转换为 SugaredValue 返回
    auto v = selfType->getConstant(field);
    return toSugaredValue(v, m, loc);
  }

  // 2. 特殊情况：处理 Module 字典的 items()、keys()、values() 方法调用
  if (concreteType_->getIterableModuleKind() == IterableModuleKind::DICT) {
    if (field == "items" || field == "keys" || field == "values") {
      return getSugaredDict(loc, m)->attr(loc, m, field);
    }
  }

  // 对于 named_modules、modules、children、named_children 方法，暂未完全处理的情况

if (field == "named_modules" || field == "modules" || field == "children" ||
    field == "named_children") {



    // 返回空的 SugaredValue，表示该情况下的解糖操作尚未完全处理
    return nullptr;
}
    return getSugaredDict(loc, m)->attr(loc, m, field);
  }

  if (field == "named_buffers") {
    return getSugaredNamedBufferDict(loc, m)->attr(loc, m, field);
  }

  // 3. Check if this is the name of an overloaded method.

  // 如果 field 是一个重载方法的名称

  // This can also be a call to a non-script module, or a plain
  // python method. If so return this as a python value.
  
  // 这也可能是对非脚本模块的调用，或者是普通的 Python 方法调用。如果是，将其作为 Python 值返回。
  
  if (const auto overloads = concreteType_->findOverloads(field)) {
    return std::make_shared<MethodValue>(self_, *overloads);
  }

  // 4. Check if it's a function attribute.
  
  // 检查是否为函数属性。
  
  if (const auto fnAttr = concreteType_->findFunctionAttribute(field)) {
    return std::make_shared<FunctionValue>(*fnAttr);
  } else if (const auto builtin = concreteType_->findBuiltinFunction(field)) {
    return std::make_shared<BuiltinFunction>(*builtin, /*self=*/c10::nullopt);
  }

  // 5. Check if it's an attribute of the original Python class that this
  // ScriptModule was derived from. The only class attributes we handle are
  // methods.
  
  // 检查是否为此 ScriptModule 派生自的原始 Python 类的属性。我们处理的唯一类属性是方法。
  
  const auto maybePyClass = concreteType_->getPyClass();
  if (!maybePyClass) {
    // ConcreteType doesn't always have an originating Python class, e.g. if it
    // was derived from a serialized ScriptModule. In this case, we've exhausted
    // our options for attr lookup.
    
    // ConcreteType 并非始终具有源 Python 类，例如如果它是从序列化的 ScriptModule 派生而来。在这种情况下，我们已经耗尽了属性查找的选项。
    
    return nullptr;
  }
  py::object unboundMethod = py::getattr(
      *maybePyClass, field.c_str(), pybind11::cast<pybind11::none>(Py_None));

  if (py::isinstance<py::function>(unboundMethod)) {
    bool isStaticFn =
        py::cast<bool>(py::module::import("torch._jit_internal")
                           .attr("is_static_fn")(*maybePyClass, field.c_str()));
    if (isStaticFn) {
      // Functions within the module annotated with @staticmethod do not need
      // binding.
      
      // 使用 @staticmethod 注解的模块内的函数不需要绑定。
      
      py::object staticFn =
          py::module::import("torch._jit_internal")
              .attr("get_static_fn")(*maybePyClass, field.c_str());
      return toSugaredValue(staticFn, m, loc);
    }
    // For Python methods that we're trying to call directly, we need to bind
    // the method to a self. (see the documentation for lazy_bind in Python for
    // more info).
    
    // 对于我们尝试直接调用的 Python 方法，我们需要将方法绑定到一个 self 上。（有关 lazy_bind 的更多信息，请参见 Python 文档。）
    
    bool isIgnoredFn =
        py::cast<bool>(py::module::import("torch._jit_internal")
                           .attr("is_ignored_fn")(unboundMethod));
    if (isIgnoredFn) {
      // Create a generated ScriptModule type with module_ set as cpp_module
      auto boundMethod = py::module::import("torch.jit._recursive")
                             .attr("lazy_bind")(concreteType_, unboundMethod);
      TORCH_CHECK(py::isinstance<py::function>(boundMethod));
      auto rcb =
          py::module::import("torch._jit_internal")
              .attr("createResolutionCallbackFromClosure")(unboundMethod);
      return std::make_shared<PythonValue>(boundMethod, rcb, self_);
    }

    // If we reach here, it's because this is a "normal" method that just hasn't
    // been compiled yet (directly exported methods would have been returned by
    // step 1). Just compile it.
    
    // 如果到达这里，那是因为这是一个尚未编译的“普通”方法（直接导出的方法将在步骤 1 中返回）。只需编译它。
    
    // 导入名为 torch.jit._recursive 的 Python 模块，并调用其中的 compile_unbound_method 方法
    auto stub =
        py::module::import("torch.jit._recursive")
            .attr("compile_unbound_method")(concreteType_, unboundMethod);
    // 断言 stub 不为空
    TORCH_INTERNAL_ASSERT(!stub.is_none());
    // 再次查找属性，此时它作为已编译方法可用
    // 返回属性的值
    return attr(loc, m, field);
  }

  // 如果未找到匹配的属性，返回空指针
  return nullptr;
}

// 检查模块值是否具有指定属性
bool ModuleValue::hasAttr(
    const SourceRange& loc,         // 源码范围
    GraphFunction& m,               // 图函数引用
    const std::string& field) {     // 字段名称
  // 尝试获取属性并检查是否不为空
  return tryGetAttr(loc, m, field) != nullptr;
}

// 调用模块值的方法
std::shared_ptr<SugaredValue> ModuleValue::call(
    const SourceRange& loc,         // 源码范围
    GraphFunction& caller,          // 调用图函数引用
    at::ArrayRef<NamedValue> args,  // 参数列表
    at::ArrayRef<NamedValue> kwargs,// 关键字参数列表
    size_t n_binders) {             // 绑定器数量
  // 获取具体类类型指针
  c10::ClassTypePtr class_type = concreteType_->getJitType()->cast<ClassType>();
  // 检查是否有前置钩子
  bool have_pre_hooks = class_type && !class_type->getForwardPreHooks().empty();
  // 检查是否有钩子
  bool have_hooks = class_type && !class_type->getForwardHooks().empty();

  // 存储参数值的向量
  std::vector<Value*> arg_values;
  // 存储前置钩子结果的命名值向量
  std::vector<NamedValue> pre_hook_result;
  // 前向输入值指针
  Value* forward_input = nullptr;
  // 获取调用图
  std::shared_ptr<Graph> calling_graph = caller.graph();

  // 如果有前置钩子或钩子
  if (have_pre_hooks || have_hooks) {
    // 将前向参数转换为元组以供前向钩子使用（急切钩子的输入始终是元组）
    for (const auto& sv : args) {
      arg_values.push_back(sv.value(*calling_graph));
    }
    forward_input =
        calling_graph->insertNode(calling_graph->createTuple(arg_values))
            ->output();
  }

  // 调用前置钩子
  if (have_pre_hooks) {
    for (const auto& hook : class_type->getForwardPreHooks()) {
      // 断言前向输入不为空
      TORCH_INTERNAL_ASSERT(forward_input != nullptr);
      // 调用前置钩子函数并将其转换为值
      Value* pre_hook_output =
          FunctionValue(hook)
              .call(
                  loc,
                  caller,
                  {NamedValue(self_), NamedValue(forward_input)},
                  kwargs,
                  n_binders)
              ->asValue(loc, caller);
      // 如果前置钩子输出的类型不是None，则将其转换为元组类型
      if (pre_hook_output->type() != NoneType::get()) {
        if (pre_hook_output->type()->kind() != TypeKind::TupleType) {
          pre_hook_output =
              calling_graph
                  ->insertNode(calling_graph->createTuple({pre_hook_output}))
                  ->output();
        }
        forward_input = pre_hook_output;
      }
    }
    // 解包前置钩子输出以供前向使用
    at::ArrayRef<Value*> output_nodes =
        calling_graph
            ->insertNode(calling_graph->createTupleUnpack(forward_input))
            ->outputs();
    // 将解包后的输出节点添加到前置钩子结果中
    for (auto& output_node : output_nodes) {
      pre_hook_result.emplace_back(output_node);
    }
    // 如果参数列表不为空，则用前置钩子结果替换
    if (!args.empty()) { // 只有存在输入时才替换
      args = pre_hook_result;
    }
  }

  // 调用前向方法
  std::shared_ptr<SugaredValue> forwardSV =
      attr(loc, caller, "forward")->call(loc, caller, args, kwargs, n_binders);
  // 获取前向输出的值
  Value* forward_output = forwardSV->asValue(loc, caller);

  // 调用钩子
  if (have_hooks) {
    // 遍历类类型的前向钩子列表
    for (const auto& hook : class_type->getForwardHooks()) {
      // 调用函数值对象，传入参数并获取返回值
      Value* forward_hook_output = FunctionValue(hook)
                                       .call(
                                           loc,
                                           caller,
                                           {NamedValue(self_),
                                            NamedValue(forward_input),
                                            NamedValue(forward_output)},
                                           kwargs,
                                           n_binders)
                                       ->asValue(loc, caller);
      // 如果前向钩子的输出类型不是 NoneType，则更新 forward_output
      if (forward_hook_output->type() != NoneType::get()) {
        forward_output = forward_hook_output;
      }
    }
  }

  // 返回一个指向 forward_output 的简单值的 shared_ptr
  return std::make_shared<SimpleValue>(forward_output);
}

// This method controls how we desugar attribute lookups on ScriptModules.
// 控制如何在脚本模块上解析属性查找的方法。
std::shared_ptr<SugaredValue> ModuleValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // 尝试获取属性值，如果成功则直接返回
  if (auto attr = tryGetAttr(loc, m, field)) {
    return attr;
  }

  // 如果属性不是在当前类定义中找到的，尝试查找是否为属性
  auto prop =
      concreteType_->getJitType()->expectRef<ClassType>().getProperty(field);
  if (prop) {
    // 如果是属性，则调用其 getter 方法并返回结果
    return MethodValue(self_, prop->getter->name())
        .call(loc, m, {}, {}, /*n_binders=*/1);
  }

  // 如果属性既不是类定义中的，也不是属性，报告错误
  std::string hint;
  if (auto failureReason = concreteType_->findFailedAttribute(field)) {
    hint = *failureReason;
  } else if (concreteType_->isIgnoredAttribute(field)) {
    hint = "attribute was ignored during compilation";
  }

  throw ErrorReport(loc)
      << "Module '"
      << concreteType_->getJitType()->expectRef<ClassType>().name()->name()
      << "'"
      << " has no attribute '" << field << "' " << hint;
}

// 返回迭代器的方法，根据具体的迭代器类型进行分支处理
SugaredValuePtr ModuleValue::iter(const SourceRange& loc, GraphFunction& m) {
  const auto iterableModuleKind = concreteType_->getIterableModuleKind();
  if (iterableModuleKind == IterableModuleKind::NONE) {
    throw ErrorReport(loc)
        << "Only constant Sequential, ModuleList, ModuleDict, or "
        << "ParameterList can be used as an iterable";
  }

  if (iterableModuleKind == IterableModuleKind::DICT) {
    auto module_dict = getSugaredDict(loc, m);
    return module_dict->keys_;
  } else if (iterableModuleKind == IterableModuleKind::LIST) {
    auto module_dict = getSugaredDict(loc, m);
    return module_dict->modules_;
  } else if (iterableModuleKind == IterableModuleKind::PARAMLIST) {
    auto module_dict = getSugaredNamedParameterList(loc, m);
    return module_dict->modules_;
  } else {
    TORCH_INTERNAL_ASSERT(false);  // 如果不是上述类型的迭代器，引发断言错误
  }
}

// 控制如何解析 Python 类的属性查找的方法
std::shared_ptr<SugaredValue> PythonClassValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // 首先尝试从 Python 对象中解析静态方法
  if (auto* fn = type_->findStaticMethod(field)) {
    return std::make_shared<FunctionValue>(fn);
  }
  // 如果在 Python 对象中找到对应属性，则将其转换为 SugaredValue 并返回
  auto py_attr = py::getattr(py_type_, field.c_str(), py::none());
  if (!py_attr.is_none()) {
    return toSugaredValue(py_attr, m, loc);
  }

  // 否则委托给父类 ClassValue 处理
  return ClassValue::attr(loc, m, field);
}

// 检查 Python 类是否具有指定属性的方法
bool PythonClassValue::hasAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  try {
    py::getattr(py_type_, field.c_str());
    return true;
  } catch (py::error_already_set& e) {
    return false;
  }
}

// 设置属性值的方法，委托给 SimpleValue::setAttr 处理
void ModuleValue::setAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field,
    Value* newValue) {
  SimpleValue simple(self_);
  simple.setAttr(loc, m, field, newValue);
}

// 控制如何调用布尔分派值的方法
std::shared_ptr<SugaredValue> BooleanDispatchValue::call(
    const SourceRange& loc,
    ...
  GraphFunction& caller,
  at::ArrayRef<NamedValue> args,
  at::ArrayRef<NamedValue> kwargs,
  size_t n_binders) {
  // 声明一个可选布尔类型的变量
  std::optional<bool> result;
  // 从调用者对象中获取图对象的引用
  Graph& graph = *(caller.graph());

  // 从Python对象中获取索引和参数名
  auto index = py::cast<size_t>(dispatched_fn_["index"]);
  auto arg_name = py::str(dispatched_fn_["arg_name"]);

  // 错误报告对象，用于记录和报告错误信息
  ErrorReport error(loc);
  if (index < args.size()) {
    // 如果索引小于参数列表的大小，从参数列表中获取布尔类型的常量值
    result = constant_as<bool>(args.at(index).value(graph));
    // 记录错误信息，指明未能获得常量值的位置
    error << "Argument for boolean dispatch at position " << index
          << " was not constant";
  } else if (auto i = findInputWithName(arg_name, kwargs)) {
    // 如果在关键字参数中找到了指定的参数名，从关键字参数中获取布尔类型的常量值
    result = constant_as<bool>(kwargs[*i].value(graph));
    // 记录错误信息，指明未能获得常量值的参数名
    error << "Keyword argument '" << arg_name
          << "' for boolean dispatch at position was not constant";
  } else {
    // 如果未找到分派标志，则使用默认值
    result = py::cast<bool>(dispatched_fn_["default"]);
    // 内部断言确认默认值有效
    TORCH_INTERNAL_ASSERT(result);
  }

  // 如果未能获取有效的布尔值，抛出错误报告
  if (!result.has_value()) {
    throw error;
  }

  // 根据布尔值选择分发的函数，将其转换为SugaredValue对象
  std::shared_ptr<SugaredValue> value;
  if (*result) {
    value = toSugaredValue(dispatched_fn_["if_true"], caller, loc);
  } else {
    value = toSugaredValue(dispatched_fn_["if_false"], caller, loc);
  }
  // 调用SugaredValue对象的call方法，返回调用结果
  return value->call(loc, caller, args, kwargs, n_binders);
}

std::shared_ptr<SugaredValue> PythonExceptionValue::call(
    const SourceRange& loc,
    GraphFunction& caller,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t /*n_binders*/) {
  Value* error_message = nullptr;  // 声明一个指向值的指针，用于存储错误信息
  if (args.empty()) {
    error_message = insertConstant(*caller.graph(), "", loc);  // 如果参数为空，则插入空字符串常量作为错误信息
  } else if (args.size() == 1) {
    error_message = args.at(0).value(*caller.graph());  // 如果只有一个参数，则使用该参数作为错误信息
  } else {
    std::vector<Value*> message_values;
    message_values.reserve(args.size() + kwargs.size());

    for (const auto& inp : args) {
      message_values.push_back(inp.value(*caller.graph()));  // 将参数列表中的值转换为图中的值对象
    }
    for (const auto& kwarg_inp : kwargs) {
      message_values.push_back(kwarg_inp.value(*caller.graph()));  // 将关键字参数列表中的值转换为图中的值对象
    }
    error_message =
        caller.graph()
            ->insertNode(caller.graph()->createTuple(message_values))  // 创建一个包含所有值的元组节点，并获取其输出值
            ->output();
  }
  Value* qualified_class_name =
      insertConstant(*caller.graph(), exception_class_qualified_name_, loc);  // 插入异常类的限定名称作为常量值

  return std::make_shared<ExceptionMessageValue>(
      error_message, qualified_class_name);  // 返回一个指向 ExceptionMessageValue 的共享指针，传递错误信息和异常类名称
}

bool isNamedTupleClass(const py::object& obj) {
  auto tuple_type = reinterpret_cast<PyObject*>(&PyTuple_Type);  // 获取 Python 中元组类型的指针
  int is_tuple_class = PyObject_IsSubclass(obj.ptr(), tuple_type);  // 检查给定对象是否为元组类的子类
  if (is_tuple_class == -1) {  // 如果 PyObject_IsSubclass 返回 -1 表示出错
    PyErr_Clear();  // 清除 Python 错误状态
    return false;  // 返回 false，表示不是命名元组类
  }
  return is_tuple_class == 1 && py::hasattr(obj, "_fields");  // 返回检查结果：是元组类且具有 "_fields" 属性
}

TypePtr registerNamedTuple(
    const py::object& obj,
    const SourceRange& loc,
    const ResolutionCallback& rcb) {
  TORCH_INTERNAL_ASSERT(isNamedTupleClass(obj));  // 断言给定对象是命名元组类

  auto qualifiedName = c10::QualifiedName(py::cast<std::string>(
      py::module::import("torch._jit_internal").attr("_qualified_name")(obj)));  // 获取对象的限定名称

  // 注意：我们需要传递 rcb 来解析 ForwardRef 注解。参见 [Note: ForwardRef annotations in NamedTuple attributes]
  py::object props =
      py::module::import("torch._jit_internal")
          .attr("_get_named_tuple_properties")(obj, loc, py::cpp_function(rcb));  // 获取命名元组属性

  auto [unqualName, field_names, field_types, objects] = py::cast<std::tuple<
      std::string,
      std::vector<std::string>,
      std::vector<TypePtr>,
      std::vector<py::object>>>(props);  // 解包命名元组属性

  std::vector<IValue> field_defaults;
  auto min_default_idx = field_names.size() - objects.size();
  for (size_t i = min_default_idx, j = 0; i < field_names.size(); ++i, ++j) {
    py::object o = objects[j];
    auto type = tryToInferType(objects[j]);  // 尝试推断对象的类型
    IValue ival = toIValue(objects[j], type.type());  // 将对象转换为 IValue
    TORCH_CHECK(
        ival.tagKind() != "Tensor",
        "Tensors are"
        " not supported as default NamedTuple fields. Their "
        "mutability could lead to potential memory aliasing "
        "problems");  // 检查默认字段是否为张量类型，如果是则抛出错误
    field_defaults.emplace_back(ival);  // 将转换后的值添加到字段默认值列表中
  }

  auto tt = TupleType::createNamed(
      qualifiedName, field_names, field_types, field_defaults);  // 创建命名元组类型

  if (auto type = get_python_cu()->get_type(qualifiedName)) {
    // 如果已经存在相同限定名称的类型，则返回该类型

    // 如果已经存在相同限定名称的类型，则返回该类型
    return type;
    // 使用 TORCH_CHECK 宏来检查 type 是否是 tt 的子类型，否则输出错误信息和 tt 的字符串表示
    TORCH_CHECK(
        type->isSubtypeOf(tt), "Can't redefine NamedTuple: ", tt->repr_str());
    // 如果检查通过，返回 type
    return type;
  }
  // 将 tt 类型注册到 Python 的运行时环境中
  get_python_cu()->register_type(tt);
  // 返回注册后的 tt 类型
  return tt;
}

// 判断给定的 Python 对象是否为枚举类
bool isEnumClass(py::object obj) {
  // 导入并获取 Python 中的 Enum 类型对象
  auto enum_type_obj =
      py::cast<py::object>(py::module::import("enum").attr("Enum"));
  // 检查 obj 是否是 enum_type_obj 的子类
  int ret = PyObject_IsSubclass(obj.ptr(), enum_type_obj.ptr());
  if (ret == -1) {
    PyErr_Clear();
    return false;
  }
  return ret == 1;
}

// 创建简单的枚举值的 SugaredValue
std::shared_ptr<SugaredValue> createSimpleEnumValue(
    const py::object& obj,
    GraphFunction& m,
    const SourceRange& loc) {
  // 获取 obj 的 __class__ 属性，即其类对象
  auto enum_class = obj.attr("__class__");
  // 尝试将 enum_class 转换为 TypePtr
  auto enum_type =
      py::cast<TypePtr>(py::module::import("torch.jit.annotations")
                            .attr("try_ann_to_type")(enum_class, loc));
  // 将 obj 转换为 IValue
  auto enum_ivalue = toIValue(obj, enum_type);
  // 将 IValue 转换为简单的 SugaredValue
  return toSimple(m.graph()->insertConstant(enum_ivalue, loc));
}

// 实现 PythonSliceClass 的 call 方法
std::shared_ptr<SugaredValue> PythonSliceClass::call(
    const SourceRange& loc,
    GraphFunction& caller,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t /*n_binders*/) {
  if (!kwargs.empty()) {
    throw ErrorReport(loc) << "Slice does not accept any keyword arguments";
  }

  // 定义默认的 start、stop、step 值
  static constexpr int64_t default_start = 0;
  static constexpr int64_t default_stop = std::numeric_limits<int64_t>::max();
  static constexpr int64_t default_step = 1;
  Graph& graph = *(caller.graph());

  // 定义辅助函数 ValOr，用于处理参数的默认值
  auto ValOr = [&](Value* given, int64_t default_val) {
    if (!given || given->type()->isSubtypeOf(*NoneType::get())) {
      return graph.insertConstant(default_val, loc);
    }
    return given;
  };

  // 初始化 start、stop、step 为 nullptr
  Value* start = nullptr;
  Value* stop = nullptr;
  Value* step = nullptr;
  size_t n = args.size();
  
  // 根据参数个数确定 Slice 的构造方式
  if (n == 1) {
    // 只有 stop 被指定的情况
    start = ValOr(nullptr, default_start);
    stop = ValOr(args[0].value(graph), default_stop);
    step = ValOr(nullptr, default_step);
  } else if (n == 2) {
    // 指定了 start 和 stop 的情况
    start = ValOr(args[0].value(graph), default_start);
    stop = ValOr(args[1].value(graph), default_stop);
    step = ValOr(nullptr, default_step);
  } else if (n == 3) {
    // start、stop 和 step 都被指定的情况
    start = ValOr(args[0].value(graph), default_start);
    stop = ValOr(args[1].value(graph), default_stop);
    step = ValOr(args[2].value(graph), default_step);
  } else {
    // 参数个数错误，抛出异常
    throw ErrorReport(loc) << "slice accepts exactly 1, 2 or 3 arguments, got: "
                           << n;
  }

  // 返回 SliceValue 对象
  return std::make_shared<SliceValue>(start, stop, step);
}

// 将 Python 对象转换为 SugaredValue
std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    GraphFunction& m,
    const SourceRange& loc,
    bool is_constant) {
  // 直接创建 SimpleValues 当可能时，因为它们是一等公民并且可以重新赋值。
  // 否则，这种情况是无效的：
  // f = python_constant
  // while ...
  //   f = f + 1
  auto& g = *m.graph();
  if (is_constant) {
    // 如果 obj 是 bool 类型，则直接转换为 SimpleValue
    if (py::isinstance<py::bool_>(obj)) {
      return toSimple(g.insertConstant(py::cast<bool>(obj), loc));
  } else if (py::isinstance<py::int_>(obj)) {
    // 如果 obj 是 Python 中的整数类型，则将其转换为 int64_t 并插入到图中，返回其简化形式
    return toSimple(g.insertConstant(py::cast<int64_t>(obj), loc));
  } else if (py::isinstance<py::float_>(obj)) {
    // 如果 obj 是 Python 中的浮点数类型，则将其转换为 double 并插入到图中，返回其简化形式
    return toSimple(g.insertConstant(py::cast<double>(obj), loc));
  } else if (PyComplex_CheckExact(obj.ptr())) {
    // 如果 obj 是 Python 中的复数类型，则将其转换为 std::complex<double>，插入到图中，返回其简化形式
    auto c_obj = py::cast<std::complex<double>>(obj.ptr());
    return toSimple(
        g.insertConstant(static_cast<c10::complex<double>>(c_obj), loc));
  } else if (py::isinstance<py::str>(obj)) {
    // 如果 obj 是 Python 中的字符串类型，则将其转换为 std::string 并插入到图中，返回其简化形式
    return toSimple(g.insertConstant(py::cast<std::string>(obj), loc));
  } else if (obj.is_none()) {
    // 如果 obj 是 Python 中的 None 对象，则插入空值 IValue 到图中，返回其简化形式
    return toSimple(g.insertConstant(IValue(), loc));
  } else if (THPDevice_Check(obj.ptr())) {
    // 如果 obj 是 PyTorch 的 THPDevice 类型，则获取其设备并插入到图中，返回其简化形式
    auto device = reinterpret_cast<THPDevice*>(obj.ptr());
    return toSimple(g.insertConstant(device->device));
  } else if (THPLayout_Check(obj.ptr())) {
    // 如果 obj 是 PyTorch 的 THPLayout 类型，则获取其布局值并插入到图中，返回其简化形式
    auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
    const auto v = static_cast<int64_t>(layout->layout);
    return toSimple(g.insertConstant(v, loc));
  } else if (THPMemoryFormat_Check(obj.ptr())) {
    // 如果 obj 是 PyTorch 的 THPMemoryFormat 类型，则获取其内存格式值并插入到图中，返回其简化形式
    auto memory_format = reinterpret_cast<THPMemoryFormat*>(obj.ptr());
    const auto v = static_cast<int64_t>(memory_format->memory_format);
    return toSimple(g.insertConstant(v, loc));
  } else if (THPDtype_Check(obj.ptr())) {
    // 如果 obj 是 PyTorch 的 THPDtype 类型，则获取其标量类型值并插入到图中，返回其简化形式
    auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
    const auto v = static_cast<int64_t>(dtype->scalar_type);
    return toSimple(g.insertConstant(v, loc));
  } else if (THPQScheme_Check(obj.ptr())) {
    // 如果 obj 是 PyTorch 的 THPQScheme 类型，则获取其量化方案值并插入到图中，返回其简化形式
    auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
    const auto v = static_cast<uint8_t>(qscheme->qscheme);
    return toSimple(g.insertConstant(v, loc));
  } else if (py::isinstance<py::tuple>(obj)) {
    // 如果 obj 是 Python 中的元组类型，则逐个处理其中的元素，并将处理后的值插入到图中形成元组节点，返回元组的输出值
    py::tuple tup = obj;
    std::vector<Value*> values;
    values.reserve(tup.size());
    for (py::handle t : tup) {
      // 将 Python 元组中的每个对象转换为 SugaredValue，并将其作为 Value 插入到图中
      py::object obj = py::reinterpret_borrow<py::object>(t);
      values.push_back(toSugaredValue(obj, m, loc, true)->asValue(loc, m));
    }
    return toSimple(
        m.graph()->insertNode(m.graph()->createTuple(values))->output());
  }
}

auto opoverloadpacket_type =
    py::module::import("torch").attr("_ops").attr("OpOverloadPacket");
py::bool_ is_overloadpacket = py::isinstance(obj, opoverloadpacket_type);
if (is_overloadpacket) {
  // 如果 obj 是 OpOverloadPacket 类型的对象，则获取其 "op" 属性，并将其赋值给 obj
  obj = py::getattr(obj, "op");
}
#ifdef USE_RPC
  // 检查是否启用了 RPC，如果启用，则获取 RPC 可用性状态
  bool isRpcAvailable = py::cast<bool>(
      py::module::import("torch.distributed.rpc").attr("is_available")());
#endif

// 如果对象可以转换为函数，则返回其函数值的共享指针
if (auto callee = as_function(obj)) {
    return std::make_shared<FunctionValue>(callee->function_);
} else if (py::isinstance<py::module>(obj)) {
    // 如果对象是 Python 模块，则获取模块的名称
    std::string obj_name = py::cast<py::str>(py::getattr(obj, "__name__"));
    if (obj_name == "torch.cuda") {
        // 如果模块是 torch.cuda，则返回 CUDA Python 模块值的共享指针
        return std::make_shared<CUDAPythonModuleValue>(obj);
    }
    // 否则返回 Python 模块值的共享指针
    return std::make_shared<PythonModuleValue>(obj);
} else if (
    obj.ptr() == py::module::import("torch.jit").attr("_fork").ptr() ||
    obj.ptr() == py::module::import("torch.jit").attr("fork").ptr()) {
    // 如果对象是 torch.jit._fork 或 torch.jit.fork，则返回特殊形式值 prim::fork 的创建
    return SpecialFormValue::create(prim::fork);
} else if (
    obj.ptr() == py::module::import("torch.jit").attr("_awaitable").ptr()) {
    // 如果对象是 torch.jit._awaitable，则返回特殊形式值 prim::awaitable 的创建
    return SpecialFormValue::create(prim::awaitable);
} else if (
    obj.ptr() == py::module::import("torch.jit").attr("annotate").ptr()) {
    // 如果对象是 torch.jit.annotate，则返回特殊形式值 prim::annotate 的创建
    return SpecialFormValue::create(prim::annotate);
} else if (
    obj.ptr() == py::module::import("torch.jit").attr("isinstance").ptr()) {
    // 如果对象是 torch.jit.isinstance，则返回特殊形式值 prim::isinstance 的创建
    return SpecialFormValue::create(prim::isinstance);
#ifdef USE_RPC
  // RPC 模块仅在启用构建标志 "USE_DISTRIBUTED" 时才可用
} else if (
    isRpcAvailable &&
    obj.ptr() ==
        py::module::import("torch.distributed.rpc").attr("rpc_async").ptr()) {
    // 如果 RPC 可用且对象是 torch.distributed.rpc.rpc_async，则返回特殊形式值 prim::rpc_async 的创建
    return SpecialFormValue::create(prim::rpc_async);
} else if (
    isRpcAvailable &&
    obj.ptr() ==
        py::module::import("torch.distributed.rpc").attr("rpc_sync").ptr()) {
    // 如果 RPC 可用且对象是 torch.distributed.rpc.rpc_sync，则返回特殊形式值 prim::rpc_sync 的创建
    return SpecialFormValue::create(prim::rpc_sync);
} else if (
    isRpcAvailable &&
    // RPC 模块仅在启用构建标志 "USE_DISTRIBUTED" 时才可用
    obj.ptr() ==
        py::module::import("torch.distributed.rpc").attr("remote").ptr()) {
    // 如果 RPC 可用且对象是 torch.distributed.rpc.remote，则返回特殊形式值 prim::rpc_remote 的创建
    return SpecialFormValue::create(prim::rpc_remote);
#endif
} else if (auto callee = as_module(obj)) {
    // 如果对象可以转换为模块，则抛出错误报告
    throw ErrorReport(loc) << "Cannot call a ScriptModule that is not"
                           << " a submodule of the caller";
}

// 定义一组张量名称及其对应的标量类型，用于检查对象是否为特定类型的张量
std::vector<std::pair<const char*, at::ScalarType>> tensor_names = {
    {"BoolTensor", at::ScalarType::Bool},
    {"LongTensor", at::ScalarType::Long},
    {"ByteTensor", at::ScalarType::Byte},
    {"CharTensor", at::ScalarType::Char},
    {"DoubleTensor", at::ScalarType::Double},
    {"FloatTensor", at::ScalarType::Float},
    {"IntTensor", at::ScalarType::Int},
    {"ShortTensor", at::ScalarType::Short},
    {"HalfTensor", at::ScalarType::Half},
};
for (const auto& name : tensor_names) {
    if (obj.ptr() == py::module::import("torch").attr(name.first).ptr()) {
        // 如果对象是 torch.LongTensor 或其他相关函数创建的张量类型，
        // 返回基于 CPU 的遗留张量构造函数创建的特定类型张量
        // TODO: 添加对于 torch.cuda.LongTensor 等 GPU 张量的支持
        return LegacyTensorConstructor::create(
            prim::LegacyTypedConstructor, name.second, at::kCPU);
  }
}

// 导入 torch.jit._builtins 模块并调用 _find_builtin 函数查找内置函数
py::object builtin_name =
    py::module::import("torch.jit._builtins").attr("_find_builtin")(obj);
// 如果找到了内置函数，则创建 BuiltinFunction 对象并返回
if (!builtin_name.is_none()) {
  return std::make_shared<BuiltinFunction>(
      Symbol::fromQualString(py::str(builtin_name)), c10::nullopt);
}

// 检查 obj 是否表示异常对象，如果是则创建 PythonExceptionValue 对象并返回
if (py::cast<bool>(py::module::import("torch._jit_internal")
                       .attr("_is_exception")(obj))) {
  return std::make_shared<PythonExceptionValue>(obj);
}

// 检查 obj 是否是 py::function 类型，并且是内置函数或方法
if (py::isinstance<py::function>(obj)) {
  if (typeString(obj) == "builtin_function_or_method") {
    throw ErrorReport(loc) << "Python builtin " << py::str(obj)
                           << " is currently not supported in Torchscript";
  }
}

// 尝试从 torch._jit_internal 模块中获取分派的函数，如果成功则创建 BooleanDispatchValue 对象并返回
py::object dispatched_fn = py::module::import("torch._jit_internal")
                               .attr("_try_get_dispatched_fn")(obj);
if (!dispatched_fn.is_none()) {
  return std::make_shared<BooleanDispatchValue>(std::move(dispatched_fn));
}

// 检查 obj 是否是 ScriptClass 类型，如果是则创建 PythonClassValue 对象并返回
if (py::isinstance<ScriptClass>(obj)) {
  auto script_class = py::cast<ScriptClass>(obj);
  return std::make_shared<PythonClassValue>(
      script_class.class_type_.type_->expect<ClassType>(), obj);
}

// 检查 obj 是否是 NamedTupleClass 类型，如果是则注册 NamedTuple 并返回 NamedTupleConstructor 对象
if (isNamedTupleClass(obj)) {
  // 使用 _fake_rcb 避免推断类型时正确解析 NamedTuple 属性上的 ForwardRef 注释
  auto fakeRcb =
      py::module::import("torch.jit.annotations").attr("_fake_rcb");
  auto tuple_type =
      registerNamedTuple(obj, loc, fakeRcb)->expect<TupleType>();
  return std::make_shared<NamedTupleConstructor>(tuple_type);
}

// 检查 obj 是否是 EnumClass 类型，如果是则创建 SugaredEnumClass 对象并返回
if (isEnumClass(obj)) {
  return createSugaredEnumClassFromObj(obj, m, loc);
}

// 检查 obj 是否是 enum.Enum 类型的实例，如果是则创建 SimpleEnumValue 对象并返回
auto enum_type = py::module::import("enum").attr("Enum");
py::bool_ is_enum_value = py::isinstance(obj, enum_type);
if (py::cast<bool>(is_enum_value)) {
  return createSimpleEnumValue(obj, m, loc);
}

// 检查 obj 是否是类对象，如果是则获取其限定名并创建 PythonClassValue 对象返回
py::bool_ is_class = py::module::import("inspect").attr("isclass")(obj);
if (py::cast<bool>(is_class)) {
  py::str qualifiedName =
      py::module::import("torch._jit_internal").attr("_qualified_name")(obj);
  auto pyCu = get_python_cu();
  auto qualname = c10::QualifiedName(qualifiedName);

  // 获取 PythonClass 并创建 PythonClassValue 对象返回
  if (auto classType = pyCu->get_class(qualname)) {
    return std::make_shared<PythonClassValue>(classType, obj);
  } else {
    // 如果无法获取类型的源代码，则它是用 C 实现的，可能是标准库的一部分，因此放弃并将其视为 Python 的调用
    bool can_compile_class =
        py::cast<bool>(py::module::import("torch._jit_internal")
                           .attr("can_compile_class")(obj));
    if (can_compile_class) {
      // 注册类
      auto rcb = py::module::import("torch._jit_internal")
                     .attr("createResolutionCallbackForClassMethods")(obj);
      // 递归编译类
      py::module::import("torch.jit._script")
          .attr("_recursive_compile_class")(obj, loc);

      // 返回类
      auto newClassType = pyCu->get_class(qualname);
      AT_ASSERT(
          newClassType,
          "Class '",
          qualifiedName,
          "' should have been compiled but was not");
      return std::make_shared<PythonClassValue>(newClassType, obj);
    }
  }

  py::bool_ isFunction = py::module::import("inspect").attr("isfunction")(obj);
  if (py::cast<bool>(isFunction)) {
    // 获取函数的重载列表
    auto overloads =
        py::module::import("torch.jit._script").attr("_get_overloads")(obj);
    if (!overloads.is_none()) {
      // 如果存在编译过的函数，返回包含这些函数的 FunctionValue
      auto compiled_fns = py::cast<std::vector<StrongFunctionPtr>>(overloads);
      return std::make_shared<FunctionValue>(std::move(compiled_fns));
    }

    // 尝试编译函数
    auto compiled_fn = py::module::import("torch.jit._recursive")
                           .attr("try_compile_fn")(obj, loc);
    if (auto callee = as_function(compiled_fn)) {
      // 如果成功编译，返回编译后的函数的 FunctionValue
      return std::make_shared<FunctionValue>(*callee);
    }
  }

  if (obj.ptr() == py::module::import("math").attr("inf").ptr()) {
    // 如果对象是正无穷大，返回对应的常量值
    return toSimple(
        g.insertConstant(std::numeric_limits<double>::infinity(), loc));
  }

  py::bool_ isMethod = py::module::import("inspect").attr("ismethod")(obj);
  // 方法明确注释为不编译，因此不进行重载和编译检查，与函数不同
  if (isFunction || isMethod) {
    // 从闭包创建解析回调
    auto rcb = py::module::import("torch._jit_internal")
                   .attr("createResolutionCallbackFromClosure")(obj);
    return std::make_shared<PythonValue>(obj, rcb);
  }

  if (obj.is(py::module::import("builtins").attr("slice"))) {
    // 如果对象是内置的 slice 类型，返回对应的 PythonSliceClass 实例
    return std::make_shared<PythonSliceClass>();
  }

  // 对于其他情况，返回通用的 PythonValue
  return std::make_shared<PythonValue>(obj);
}
} // namespace torch::jit
```