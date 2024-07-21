# `.\pytorch\torch\csrc\jit\frontend\tracer.cpp`

```py
// 引入 Torch 的头文件，用于 JIT 前端的追踪功能
#include <torch/csrc/jit/frontend/tracer.h>

// 引入 ATen 库的头文件
#include <ATen/Backtrace.h>
#include <ATen/ScalarOps.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Dict.h>
#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 引入 Torch 自动微分引擎的相关头文件
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

// 引入 Torch JIT 模块相关头文件
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/utils/variadic.h>

// 引入 Torch 自定义类相关头文件
#include <torch/custom_class.h>

// 引入标准库头文件
#include <memory>
#include <sstream>
#include <string>

// Torch JIT 追踪器命名空间
namespace torch::jit::tracer {

////////////////////////////////////////////////////////////////////////////////
// 追踪记录相关
////////////////////////////////////////////////////////////////////////////////

// 追踪器内部细节命名空间
namespace detail {

// 向节点添加常量输入的通用函数模板
template <typename T>
void genericAddInput(Node* n, T value) {
  // 在当前图中插入常量节点，并返回对应的值
  Value* v = n->owningGraph()->insertConstant(value);
  // 记录节点对应的源代码位置
  recordSourceLocation(v->node());
  // 将常量值作为节点的输入
  n->addInput(v);
}

// 向节点添加可选输入的通用函数模板
template <typename T>
void genericAddOptionalInput(
    Node* n,
    const char* name,
    const std::optional<T>& value) {
  // 如果值存在
  if (value) {
    // 使用追踪器 API 添加输入
    jit::tracer::addInputs(n, name, *value);
  } else {
    // 否则，在当前图中创建一个 None 节点，作为输入
    Graph* g = n->owningGraph();
    Value* none = g->insertNode(g->createNone())->output();
    n->addInput(none);
  }
}

// 处理不支持的参数类型异常的函数模板
template <typename T>
void badArgType(const T& v) {
  // 抛出异常，说明 JIT 追踪器中发现了不支持的参数类型
  AT_ERROR(
      "Found an unsupported argument type in the JIT tracer: ",
      c10::demangle_type<T>(),
      ". File a bug report.");
}

// 线程局部变量，用于存储追踪状态
thread_local std::shared_ptr<TracingState> tracing_state;

} // namespace detail

// 追踪器状态警告模式的原子布尔变量
static std::atomic<bool> tracer_state_warn_mode{true};

// 获取追踪器状态警告模式的全局函数
std::atomic<bool>& getTracerStateWarnMode() {
  return tracer_state_warn_mode;
}

// 暂停追踪功能的函数，返回恢复追踪状态的闭包
std::function<void()> pauseTracing() {
  // 获取当前的追踪状态
  std::shared_ptr<tracer::TracingState> state = getTracingState();
  // 设置当前追踪状态为 null
  tracer::setTracingState(nullptr);

  // 返回一个闭包函数，用于恢复之前保存的追踪状态
  return [state]() { tracer::setTracingState(state); };
}

// 删除给定 IValue 变量的追踪记录
void delValueTrace(const IValue& var) {
  // 从追踪状态中删除指定的值
  getTracingState()->delValue(var);
}

// TracingState 类的方法，用于删除指定 IValue 变量的追踪记录
void TracingState::delValue(const IValue& var) {
  // 遍历环境堆栈中的每一层
  for (const auto i : c10::irange(env_stack.size())) {
    auto& value_map = env_stack.at(env_stack.size() - 1 - i);
    auto it = value_map.find(var);
    // 如果找到指定的变量
    if (it == value_map.end()) {
      continue;
    }
    // 删除该变量的记录
    value_map.erase(it);
  }
}

// 给定一个 IValue 'var'，返回计算该变量值的节点 'node'
// 这里将未追踪的变量视为嵌入在图中的常量，用于处理一些情况
// 例如从 torch.autograd.variable 移动到 C++ 的代码
// 定义一个名为 `getValueTrace` 的函数，接收一个 `IValue` 类型的参数 `var`，返回一个 `Value*` 类型的指针
Value* getValueTrace(const IValue& var) {
  // 调用 `getTracingState()` 获取当前的追踪状态，然后调用 `getValue(var)` 返回与 `var` 对应的值
  return getTracingState()->getValue(var);
}

// 定义一个静态函数 `getOptTensorValueTrace`，接收一个 `std::optional<at::Tensor>` 类型的可选参数 `var`，返回一个 `Value*` 类型的指针
static Value* getOptTensorValueTrace(const std::optional<at::Tensor>& var) {
  // 将 `var` 转换为 `IValue` 类型，并调用 `getValueTrace` 返回与 `var` 对应的值
  return getValueTrace(IValue(var));
}

// 在 `TracingState` 类中定义 `getValue` 函数，接收一个 `IValue` 类型的参数 `var`，返回一个 `Value*` 类型的指针
Value* TracingState::getValue(const IValue& var) {
  // 如果 `var` 是一个 Tensor 列表，则创建一个对应的 Graph 节点，并返回该节点的输出值
  if (var.isTensorList()) {
    return graph
        ->insertNode(graph->createList(
            TensorType::get(),
            fmap(
                var.toTensorVector(),
                [&](const IValue& val) { return getValue(val); })))
        ->output();
  } else if (var.isTuple()) { // 如果 `var` 是一个元组，则创建一个对应的 Graph 节点，并返回该节点的输出值
    return graph
        ->insertNode(graph->createTuple(fmap(
            var.toTupleRef().elements(),
            [&](const IValue& val) { return getValue(val); })))
        ->output();
  } else if (var.isGenericDict()) { // 如果 `var` 是一个通用字典，则创建一个对应的 Graph 节点，并返回该节点的输出值
    auto dict = var.toGenericDict();
    TypePtr key_type = dict.keyType();
    TypePtr value_type = dict.valueType();
    std::vector<Value*> keys;
    std::vector<Value*> values;
    // 遍历字典中的每一项，将键和值分别转换为 Graph 节点的值，并保存到对应的 vector 中
    for (const auto& entry : dict) {
      keys.emplace_back(getValue(entry.key()));
      values.emplace_back(getValue(entry.value()));
    }
    // 创建一个字典类型的 Graph 节点，并返回该节点的输出值
    auto dict_node = graph->createDict(key_type, value_type, keys, values);
    return graph->insertNode(dict_node)->output();
  }
  // 如果 `var` 是一个 Tensor，则尝试获取其对应的 Graph 节点，如果不存在，则创建一个 None 类型的节点并返回其输出值
  if (var.isTensor()) {
    auto& ten = var.toTensor();
    if (!ten.defined()) {
      Node* n = graph->createNone();
      return graph->insertNode(n)->output();
    }
    // 从环境栈中查找 `var` 对应的节点，并返回该节点
    for (const auto i : c10::irange(env_stack.size())) {
      auto& value_map = env_stack.at(env_stack.size() - 1 - i);
      auto it = value_map.find(var);
      if (it == value_map.end()) {
        continue;
      }
      // 如果节点没有调试名称，则为其设置一个唯一的调试名称
      if (!it->second->hasDebugName()) {
        auto unique_name = getTracingState()->lookup_var_name_fn(ten);
        if (!unique_name.empty()) {
          it->second->setDebugName(unique_name);
        }
      }
      return it->second;
    }

    // 如果在环境栈中未找到 `var` 对应的节点，则将其视为常量插入到 Graph 中
    if (ten.requires_grad()) {
      // 如果 Tensor 需要梯度，则抛出运行时异常
      pauseTracing();
      std::ostringstream oss;
      oss << "Cannot insert a Tensor that requires grad as a constant. "
          << "Consider making it a parameter or input, or detaching the gradient\n"
          << "Tensor:\n"
          << ten;
      throw std::runtime_error(oss.str());
    }

    // 将 Tensor 插入为常量到 Graph 中，并返回该常量的值
    Value* constant = graph->insertConstant(ten);
    recordSourceLocation(constant->node());
    constant->inferTypeFrom(ten);
    // 在当前环境栈的最后一个映射中插入 `var` 和其对应的常量节点
    auto it = env_stack.back().emplace(var, constant);
    return it.first->second;
  } else if (var.isFuture() || var.isObject()) {
    // 如果变量是 Future 或者 Object 类型，则从环境栈中查找最近的定义
    for (const auto i : c10::irange(env_stack.size())) {
      auto& future_map = env_stack.at(env_stack.size() - 1 - i);
      auto it = future_map.find(var);
      if (it == future_map.end()) {
        continue;
      }
      // 返回找到的变量值
      return it->second;
    }

    // 查找 Torchbind 自定义类
    if (isCustomClass(var)) {
      auto obj = Object(var.toObject());
      auto qualname = obj.type()->name();
      auto custom_class_type = getCustomClass(qualname->qualifiedName());
      if (custom_class_type) {
        auto capsule = var.toObject()->getAttr("capsule");
        // 从环境栈中查找最近的定义
        for (const auto i : c10::irange(env_stack.size())) {
          auto& value_map = env_stack.at(env_stack.size() - 1 - i);
          auto it = value_map.find(capsule);
          if (it == value_map.end()) {
            continue;
          }
          // 返回找到的变量值
          return it->second;
        }
      }
    }

    // 如果变量是 Future 类型，抛出错误信息
    std::ostringstream oss;
    if (var.isFuture()) {
      oss << "Tried to trace Future or Object that the tracer was not aware of.";
    } else {
      // 如果变量不在活跃追踪中，抛出详细错误信息
      oss << "Tried to trace " << var
          << " but it is not part of the active trace. Modules that are called during a trace"
          << " must be registered as submodules of the thing being traced.";
    }
    throw std::runtime_error(oss.str());
  } else {
    // 如果变量不是 Tensor 类型，尝试创建常量并将其插入追踪图中
    auto constant = tryInsertConstant(*graph, var);
    if (constant) {
      // 记录常量节点的源位置并返回常量值
      recordSourceLocation(constant.value()->node());
      return *constant;
    }
    // 如果无法创建常量，则抛出详细错误信息
    std::ostringstream os;
    os << "Tracer cannot get value trace for type " << var.tagKind() << ". "
       << "The below value could not be materialized as a constant:\n"
       << var;
    throw std::runtime_error(os.str());
  }
}
// 检查环境堆栈中是否存在指定的变量值
bool TracingState::hasValue(const IValue& var) const {
  // 遍历环境堆栈中的每一个帧
  for (const auto& frame : env_stack) {
    // 如果当前帧中包含指定的变量
    if (frame.count(var)) {
      return true;  // 返回true表示找到了变量值
    }
  }
  return false;  // 如果未找到变量值则返回false
}

// 获取输出值对应的图节点
Value* TracingState::getOutput(const IValue& iv, size_t i) {
  // 获取当前追踪状态的严格模式
  bool tracing_mode_strict = getTracingState()->strict;

  // 如果输入值是张量
  if (iv.isTensor()) {
    const at::Tensor& var = iv.toTensor();
    // 如果张量未定义
    if (!var.defined()) {
      // 创建一个表示None的节点
      Node* n = graph->createNone();
      // 将节点插入到图中并返回其输出
      return graph->insertNode(n)->output();
    }

    // 获取当前环境堆栈的最后一个值映射
    auto& value_map = getTracingState()->env_stack.back();
    auto it = value_map.find(iv);
    // 如果在映射中未找到对应的值
    if (it == value_map.end()) {
      // 抛出运行时异常，说明跟踪区域的输出与跟踪输入没有可观察的数据依赖关系
      std::ostringstream os;
      os << "output " << i << " (" << var
         << ") of traced region did not have observable "
         << "data dependence with trace inputs; this probably indicates your "
            "program "
         << "cannot be understood by the tracer.";
      throw std::runtime_error(os.str());
    }
    // 返回找到的值对应的图节点
    return it->second;
  }
  // 如果输入值是张量列表
  else if (iv.isTensorList()) {
    // 如果处于追踪模式严格模式，发出警告信息
    if (tracing_mode_strict) {
      tracer::warn(
          "Encountering a list at the output of the tracer", STRICT_TRACER_MSG);
    }
    // 创建一个张量类型的列表节点，并将其插入图中
    return graph
        ->insertNode(graph->createList(
            TensorType::get(),
            fmap(
                iv.toTensorVector(),
                [&](const IValue& ival) { return getOutput(ival, i); })))
        ->output();
  }
  // 如果输入值是元组
  else if (iv.isTuple()) {
    const auto& tuple = iv.toTupleRef().elements();
    // 创建一个元组节点，并将其插入到图中
    auto tuple_node = graph->createTuple(
        fmap(tuple, [&](const IValue& ival) { return getOutput(ival, i); }));
    graph->insertNode(tuple_node);
    // 返回元组节点的输出
    return tuple_node->output();
  }
  // 如果输入值是通用字典
  else if (iv.isGenericDict()) {
    // 如果处于追踪模式严格模式，抛出异常信息
    if (tracing_mode_strict) {
      throw std::runtime_error(
          "Encountering a dict at the output of the tracer" +
          std::string(STRICT_TRACER_MSG));
    }
    // 将输入值转换为通用字典
    auto dict = iv.toGenericDict();
    TypePtr key_type = dict.keyType();
    TypePtr value_type = dict.valueType();

    // 检查键类型和值类型是否有效
    bool key_type_valid = key_type->isSubtypeOf(*StringType::get()) ||
        key_type->isSubtypeOf(*TensorType::get());
    bool value_type_valid = value_type->isSubtypeOf(*TensorType::get());

    // 支持只包含张量的元组值
    if (value_type->isSubtypeOf(*AnyTupleType::get())) {
      value_type_valid = true;
      for (const auto& type : value_type->containedTypes()) {
        if (!type->isSubtypeOf(*TensorType::get())) {
          value_type_valid = false;
          break;
        }
      }
    }

    // 如果键类型或值类型无效，抛出异常信息
    if (!key_type_valid || !value_type_valid) {
      std::ostringstream os;
      os << "output " << i << " (" << dict << ") of traced region "
         << "cannot be understood by the tracer, only outputs matching"
         << "dict[Union[str, Tensor], Union[Tensor, Tuple[Tensor, ...]]] "
         << "can be a dictionary output of a traced function";
      throw std::runtime_error(os.str());
    }
    // 创建键和值的空向量
    std::vector<Value*> keys;
    std::vector<Value*> values;
    // 遍历给定的字典 `dict` 中的每一个条目
    for (const auto& entry : dict) {
      // 获取当前条目的键，并添加到 `keys` 向量中
      keys.emplace_back(getValue(entry.key()));
      // 获取当前条目的值，并根据索引 `i` 转换成输出，并添加到 `values` 向量中
      values.emplace_back(getOutput(entry.value(), i));
    }
    // 使用 `graph` 对象创建一个新的字典节点，使用 `key_type` 和 `value_type` 分别作为键和值的类型，
    // 并将 `keys` 和 `values` 向量作为其键和值的内容
    auto dict_node = graph->createDict(key_type, value_type, keys, values);
    // 将新创建的字典节点插入到 `graph` 中
    graph->insertNode(dict_node);
    // 返回新创建的字典节点的输出
    return dict_node->output();
  } else {
    // 如果无法处理给定的类型，则抛出错误信息
    AT_ERROR(
        "Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions");
  }
}

// 创建一个新的节点并添加到当前图中，使用给定的操作名和输出数量
Node* TracingState::createNode(c10::Symbol op_name, size_t num_outputs) {
  return graph->create(op_name, num_outputs);
}

// 将一个节点插入到当前图中
void TracingState::insertNode(Node* node) {
  graph->insertNode(node);
}

// XXX: 此函数会修改输入参数
// 将输入参数作为值添加到状态中，并设置对应的类型
static IValue addInput(
    const std::shared_ptr<TracingState>& state,
    const IValue& input,
    const TypePtr& type,
    Value* value) {
  value->setType(type);

  // 如果类型是 TensorType
  if (type->isSubtypeOf(*TensorType::get())) {
    auto input_tensor = input.toTensor();
    auto name = Variable(input_tensor).name();

    // 如果状态中已经存在该输入，则使用视图而不是拷贝
    if (state->hasValue(input)) {
      input_tensor = input_tensor.view(input_tensor.sizes());
    }

    // 如果值没有调试名称，则设置调试名称为变量名
    if (!value->hasDebugName()) {
      value->setDebugName(name);
    }

    // 在状态中设置值与对应的 Value
    state->setValue(input_tensor, value);
    return input_tensor;

  // 如果类型是 TupleType
  } else if (auto tuple_type = type->cast<TupleType>()) {
    auto unpack_node =
        state->graph->insertNode(state->graph->createTupleUnpack(value));
    auto elem_values = unpack_node->outputs();
    auto elem_types = tuple_type->elements();
    auto tuple = input.toTuple();
    const auto& elems = tuple->elements();
    size_t num_elems = elems.size();
    AT_ASSERT(
        elem_values.size() == num_elems && elem_types.size() == num_elems);

    // 逐个添加元组的元素到状态中，并递归调用 addInput
    for (const auto i : c10::irange(num_elems)) {
      tuple->unsafeSetElement(
          i, addInput(state, elems.at(i), elem_types[i], elem_values[i]));
    }
    return tuple;

  // 如果类型是 DictType
  } else if (auto dict_type = type->cast<DictType>()) {
    auto dict = input.toGenericDict();

    // 静态地解包字典的值
    for (const auto& entry : dict) {
      const IValue& key = entry.key();
      auto static_key = state->graph->insertConstant(key);
      auto static_value =
          state->graph->insert(aten::__getitem__, {value, static_key});

      // 记录静态值的源位置，并递归调用 addInput
      recordSourceLocation(static_value->node());
      dict.insert_or_assign(
          entry.key(),
          addInput(
              state, entry.value(), dict_type->getValueType(), static_value));
    }
    return dict;

  // 如果类型是 ListType
  } else if (auto list_type = type->cast<ListType>()) {
    size_t num_elems = input.isList() ? input.toListRef().size()
                                      : input.toTensorVector().size();
    auto list_unpack = state->graph->insertNode(
        state->graph->createListUnpack(value, num_elems));
    auto unpack_outputs = list_unpack->outputs();

    // 如果是 TensorList，则逐个处理列表中的元素，并递归调用 addInput
    if (input.isTensorList()) {
      auto elems = input.toTensorList();
      for (const auto i : c10::irange(num_elems)) {
        elems[i] = addInput(
                       state,
                       elems.get(i),
                       list_type->getElementType(),
                       unpack_outputs[i])
                       .toTensor();
      }
      return elems;
      // 返回处理后的列表
    } else {
      // 如果输入不是张量或者张量的嵌套字典或元组，则抛出错误
      auto elems = input.toList();
      // 遍历元素数量范围
      for (const auto i : c10::irange(num_elems)) {
        // 对每个元素调用 addInput 函数，将其加入到 elems 中
        elems[i] = addInput(
            state,  // 状态对象
            elems.get(i),  // 获取当前元素
            list_type->getElementType(),  // 获取列表元素的类型
            unpack_outputs[i]);  // 解包输出中的第 i 个元素
      }
      // 返回处理后的 elems 列表
      return elems;
    }
  } else {
    // 如果输入类型不符合要求，则抛出错误信息
    AT_ERROR(
        "Only tensors or (possibly nested) dict or tuples of tensors can be "
        "inputs to traced functions. Got ",
        type->repr_str());  // 打印错误信息和输入类型的字符串表示形式
  }
}

static void gatherParametersAndBuffers(
    const std::shared_ptr<TracingState>& state,
    Value* self_value,
    const Module& self,
    const std::string& prefix) {
  // 获取 self_value 对应的图形对象
  Graph& g = *self_value->owningGraph();

  // 将 self 对象和其对应的 self_value 存入 TracingState 中
  state->setValue(self._ivalue(), self_value);

  // 获取 self 对象的类型信息
  auto self_ty = self.type();
  // 遍历 self 对象的所有命名属性
  for (const NameValue& s : self.named_attributes(/*recurse=*/false)) {
    // 构建属性的全名，包括前缀
    auto qualname = prefix + "." + s.name;
    // 在图中插入 TracedAttr 节点，表示跟踪该属性的访问
    Value* trace_get_attr = g.insertNode(g.create(prim::TracedAttr))
                                ->s_(attr::scope, qualname)
                                ->output()
                                ->setType(s.value.type());
    // 如果属性类型是 TensorType，则添加其作为输入到 state 中
    if (s.value.type()->isSubtypeOf(*TensorType::get())) {
      addInput(state, s.value, s.value.type(), trace_get_attr);
    }
    // 如果属性是自定义类，使用 tracer::setValueTrace 进行跟踪
    if (isCustomClass(s.value)) {
      tracer::setValueTrace(s.value, trace_get_attr);
    }

    // 获取属性的类型信息
    auto attr_type = self_ty->getAttribute(s.name);
    // 如果属性类型是 Module，并且不是 InterfaceType，递归处理其参数和缓冲区
    // InterfaceType 不能暴露任何属性，这些属性不应在 InterfaceType 的模块外部使用
    if (attr_type->is_module() &&
        attr_type->kind() != TypeKind::InterfaceType) {
      gatherParametersAndBuffers(
          state, trace_get_attr, Module(s.value.toObject()), qualname);
    }
  }
}

// 开始跟踪函数的执行
std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self,
    const std::vector<std::string>& argument_names) {
  try {
    // 如果已经在进行跟踪，则抛出错误，因为跟踪不能嵌套
    if (isTracing()) {
      AT_ERROR("Tracing can't be nested");
    }
    // 创建 TracingState 对象并设置为当前跟踪状态
    auto state = std::make_shared<TracingState>();
    setTracingState(state);

    // 如果有 self 对象，则将其作为输入，并收集其参数和缓冲区信息
    if (self) {
      Value* self_value = state->graph->insertInput(0, "self")->setType(
          self->_ivalue()->type());
      gatherParametersAndBuffers(state, self_value, *self, {"__module"});
    }

    // 当提供了足够的参数名提示时，将其用作跟踪函数/模块的调试名称
    // argument_names 的长度可以大于 inputs 的长度，因为某些参数可能有默认值，不需要示例输入
    if (argument_names.size() >= inputs.size()) {
      for (size_t i = 0, e = inputs.size(); i < e; ++i) {
        // 将输入作为图的输入，并添加到 state 中
        IValue& input = inputs[i];
        input = addInput(
            state,
            input,
            input.type(),
            state->graph->addInput(argument_names[i]));
      }
      ```
    // 如果不处于 tracing 状态，将每个输入添加到状态的图中作为输入
    } else {
      for (IValue& input : inputs) {
        input = addInput(state, input, input.type(), state->graph->addInput());
      }
    }

    // 获取状态对象中的图对象
    auto graph = state->graph;

    // 设置追踪状态的变量名查找函数
    getTracingState()->lookup_var_name_fn = std::move(var_name_lookup_fn);
    // 设置追踪状态的 strict 模式
    getTracingState()->strict = strict;
    // 设置追踪状态的强制 outplace 模式
    getTracingState()->force_outplace = force_outplace;

    // 调用追踪后的函数，获取输出的栈
    auto out_stack = traced_fn(inputs);

    // 结束追踪，将 'out_stack' 视为追踪的输出，这些变量在后续调用中将计算其值
    size_t i = 0;
    for (auto& output : out_stack) {
      // 注意：栈是按照 "反向" 排序的，因此在传递诊断号时，基于大小需要进行翻转
      state->graph->registerOutput(
          state->getOutput(output, out_stack.size() - i));
      i++;
    }
    // 设置追踪状态为空
    setTracingState(nullptr);

    // 如果处于内联模式，内联化图形中的所有操作
    if (getInlineEverythingMode()) {
      Inline(*graph);
    }
    // 修正追踪范围块
    FixupTraceScopeBlocks(graph, self);
    // 标准化图形中的操作
    NormalizeOps(graph);

    // 返回状态对象和输出栈
    return {state, out_stack};
  } catch (...) {
    // 发生异常时放弃追踪
    tracer::abandon();
    // 重新抛出异常
    throw;
  }
}

// 中止跟踪。用于在出现错误时重置状态。
void abandon() {
  // 调用函数设置跟踪状态为nullptr，即无状态
  setTracingState(nullptr);
}

// 设置值的跟踪，将值v映射到指定的Value指针value
void setValueTrace(const IValue& v, Value* value) {
  // 调用跟踪状态对象的setValue方法，将值v和对应的Value指针value关联起来
  return getTracingState()->setValue(v, value);
}
void TracingState::setValue(const IValue& v, Value* value) {
  // 如果值v是张量
  if (v.isTensor()) {
    auto& var = v.toTensor();
    // 断言张量已定义
    AT_ASSERT(var.defined());
    // 将值value与环境栈的当前环境中的值v关联起来
    env_stack.back()[v] = value;

    // 如果值来自CallFunction或CallMethod，则可能没有形状信息。为了调试性，通过将具体值的类型赋给jit::Value来增强类型信息。
    if (auto tensor_type = value->type()->cast<TensorType>()) {
      // 如果张量类型不完整，则从var推断类型
      if (!tensor_type->isComplete()) {
        value->inferTypeFrom(var);
      }
    }
  } else if (v.isTensorList()) {
    auto outputs = v.toTensorList();
    // 在图中插入节点，进行列表解包操作
    Node* unpack_node =
        graph->insertNode(graph->createListUnpack(value, outputs.size()));
    // 遍历张量列表中的每个张量，递归调用setValue
    for (const auto i : c10::irange(outputs.size())) {
      setValue(outputs.get(i), unpack_node->outputs()[i]);
    }
  } else if (v.isTuple()) {
    // 获取元组中的输出元素
    const auto& outputs = v.toTupleRef().elements();
    // 在图中插入节点，进行元组解包操作
    Node* unpack_node = graph->insertNode(graph->createTupleUnpack(value));
    // 遍历元组中的每个元素，递归调用setValue
    for (const auto i : c10::irange(outputs.size())) {
      setValue(outputs[i], unpack_node->outputs()[i]);
    }
  } else if (v.isList()) {
    // 获取列表中的元素
    auto elements = v.toListRef();
    // 在图中插入节点，进行列表解包操作
    Node* unpack_node =
        graph->insertNode(graph->createListUnpack(value, elements.size()));
    // 遍历列表中的每个元素，递归调用setValue
    for (const auto i : c10::irange(elements.size())) {
      setValue(elements[i], unpack_node->outputs()[i]);
    }
  } else if (isCustomClass(v)) {
    // 如果是自定义类对象，获取其capsule属性并与值value关联
    auto capsule = v.toObject()->getAttr("capsule");
    env_stack.back()[capsule] = value;
  } else if (v.isFuture() || v.isObject()) {
    // 如果是Future或对象类型，将其与值value关联
    env_stack.back()[v] = value;
  } else if (v.isGenericDict()) {
    // 如果是通用字典类型
    auto dict = v.toGenericDict();
    TypePtr key_type = dict.keyType();
    TypePtr value_type = dict.valueType();
    // 遍历字典中的每个条目，将键值对插入图中
    for (const auto& entry : dict) {
      auto static_key = graph->insertConstant(entry.key());
      auto static_value = graph->insert(aten::__getitem__, {value, static_key});
      setValue(entry.value(), static_value);
    }
  } else {
    // 如果不支持的类型，抛出运行时错误
    std::ostringstream os;
    os << "Tracer cannot set value trace for type " << v.tagKind() << ". "
       << "Supported types are tensor, tensor list, and tuple of tensors.";
    throw std::runtime_error(os.str());
  }
}

// 向节点n添加输入，使用名称name和整数值value
void addInputs(Node* n, const char* name, int64_t value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  // 如果存储器中有名称为name的值，弹出并添加到节点n的输入中
  if (ArgumentStash::hasValue(name)) {
    Value* v = ArgumentStash::popValue(name);
    n->addInput(v);
  } else {
    // 否则，将整数值作为输入添加到节点n中
    detail::genericAddInput(n, value);
  }
}

// 向节点n添加输入，使用名称name和SymInt类型的值value
void addInputs(Node* n, const char* name, c10::SymInt value) {
  // 调用前一个addInputs函数，将SymInt值转换为整数值后添加
  addInputs(n, name, value.guard_int(__FILE__, __LINE__));
}

// 向节点n添加输入，使用名称name和可选的整数值value
void addInputs(Node* n, const char* name, std::optional<int64_t> value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  // 如果存储器中有名称为name的值，弹出并添加到节点n的输入中
  if (ArgumentStash::hasValue(name)) {

    // 使用ArgumentStash从存储中获取名称为name的值
    Value* v = ArgumentStash::popValue(name);
    // 将获取的值作为输入添加到节点n中
    n->addInput(v);
  } else {
    // 如果没有值，则调用detail命名空间中的genericAddInput函数，将可选的整数值作为输入添加到节点n中
    detail::genericAddInput(n, *value);
  }
}
    # 从 ArgumentStash 中弹出名称为 `name` 的值，并将其赋给变量 v
    Value* v = ArgumentStash::popValue(name);
    
    # 将变量 v 添加为节点 n 的输入值
    n->addInput(v);
  } else if (value) {
    
    # 如果 value 存在，则调用 detail 命名空间中的 genericAddInput 函数，
    # 将节点 n 和 value 作为参数传递给函数
    detail::genericAddInput(n, *value);
  } else {
    
    # 如果 value 不存在，则获取节点 n 所属的图对象
    Graph* g = n->owningGraph();
    
    # 在图对象 g 中插入一个代表 None 的节点，并获取其输出值
    Value* none = g->insertNode(g->createNone())->output();
    
    # 将代表 None 的值添加为节点 n 的输入值
    n->addInput(none);
  }
}
// 向节点添加布尔类型输入值的函数
void addInputs(Node* n, const char* name, bool value) {
  // 调用通用函数，向节点添加布尔类型输入值
  detail::genericAddInput(n, value);
}

// 向节点添加可选布尔类型输入值的函数
void addInputs(Node* n, const char* name, const std::optional<bool>& value) {
  // 调用通用函数，向节点添加可选布尔类型输入值
  detail::genericAddOptionalInput(n, name, value);
}

// 向节点添加双精度浮点数类型输入值的函数
void addInputs(Node* n, const char* name, double value) {
  // 调用通用函数，向节点添加双精度浮点数类型输入值
  detail::genericAddInput(n, value);
}

// 向节点添加可选双精度浮点数类型输入值的函数
void addInputs(Node* n, const char* name, const std::optional<double>& value) {
  // 调用通用函数，向节点添加可选双精度浮点数类型输入值
  detail::genericAddOptionalInput(n, name, value);
}

// 向节点添加标量类型输入值的函数
void addInputs(Node* n, const char* name, const at::Scalar& value) {
  // 使用 tracer::ArgumentStash 存储中介来判断是否已有该值，并据此向节点添加输入
  using ArgumentStash = jit::tracer::ArgumentStash;
  if (ArgumentStash::hasValue(name)) {
    // 若有则弹出该值，并向节点添加输入
    Value* v = ArgumentStash::popValue(name);
    n->addInput(v);
  } else {
    // 否则调用通用函数，向节点添加标量类型输入值
    detail::genericAddInput(n, value);
  }
}

// 向节点添加字符串视图类型输入值的函数
void addInputs(Node* n, const char* name, const c10::string_view value) {
  // 调用通用函数，将字符串视图类型转换为标准字符串并向节点添加输入
  detail::genericAddInput(n, std::string(value));
}

// 向节点添加张量类型输入值的函数
void addInputs(Node* n, const char* name, const at::Tensor& value) {
  // 获取张量的值追踪并将其作为输入添加到节点
  n->addInput(getValueTrace(value));
}

// 向节点添加可选张量类型输入值的函数
void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Tensor>& value) {
  // 调用通用函数，向节点添加可选张量类型输入值
  detail::genericAddOptionalInput(n, name, value);
}

// 向节点添加生成器类型输入值的函数
void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Generator>& value) {
  // 获取所属图形对象
  Graph* g = n->owningGraph();

  // 如果值存在且已定义，则向节点添加输入
  if (value.has_value() && value->defined()) {
    detail::genericAddInput(n, *value);
  } else {
    // 否则创建未定义的生成器对象并将其作为输入添加到节点
    Value* undef_gen = g->insertNode(g->createNone())->output();
    n->addInput(undef_gen);
  }
}

// 向节点添加设备类型输入值的函数
void addInputs(Node* n, const char* name, at::Device value) {
  // 调用通用函数，向节点添加设备类型输入值
  detail::genericAddInput(n, value);
}

// 向节点添加流类型输入值的函数
void addInputs(Node* n, const char* name, c10::Stream stream) {
  // 调用通用函数，将流类型转换为 IValue 类型并向节点添加输入
  detail::genericAddInput(n, c10::IValue(stream));
}

// 向节点添加布局类型输入值的函数
void addInputs(Node* n, const char* name, at::Layout value) {
  // 调用通用函数，将布局类型转换为整数类型并向节点添加输入
  detail::genericAddInput(n, static_cast<int64_t>(value));
}

// 向节点添加标量类型输入值的函数
void addInputs(Node* n, const char* name, at::ScalarType value) {
  // 调用通用函数，将标量类型转换为整数类型并向节点添加输入
  detail::genericAddInput(n, static_cast<int64_t>(value));
}

// 向节点添加内存格式类型输入值的函数
void addInputs(Node* n, const char* name, at::MemoryFormat value) {
  // 调用通用函数，将内存格式类型转换为整数类型并向节点添加输入
  detail::genericAddInput(n, static_cast<int64_t>(value));
}

// 向节点添加可选内存格式类型输入值的函数
void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::MemoryFormat>& value) {
  // 调用通用函数，向节点添加可选内存格式类型输入值
  detail::genericAddOptionalInput(n, name, value);
}

// 向节点添加可选布局类型输入值的函数
void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Layout>& value) {
  // 调用通用函数，向节点添加可选布局类型输入值
  detail::genericAddOptionalInput(n, name, value);
}

// 向节点添加可选设备类型输入值的函数
void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Device>& value) {
  // 调用通用函数，向节点添加可选设备类型输入值
  detail::genericAddOptionalInput(n, name, value);
}

// 向节点添加维度名称列表类型输入值的函数（未实现）
void addInputs(
    Node* n,
    const char* name,
    std::optional<at::DimnameList> value) {
  // 抛出错误，因为不支持使用追踪器的命名张量
  TORCH_CHECK(false, "NYI: Named tensors are not supported with the tracer");
}
void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::ScalarType>& value) {
  // 调用通用函数，添加可选输入
  detail::genericAddOptionalInput(n, name, value);
}

void addInputs(
    Node* n,
    const char* name,
    at::ArrayRef<at::Tensor> value,
    bool allow_undefined) {
  // 调用重载函数，将数组引用转换为 ITensorListRef 并添加输入
  addInputs(n, name, at::ITensorListRef(value), allow_undefined);
}

void addInputs(
    Node* n,
    const char* name,
    std::vector<at::Tensor> value,
    bool allow_undefined) {
  // 调用重载函数，将张量向量转换为 ITensorListRef 并添加输入
  addInputs(n, name, at::ITensorListRef(value), allow_undefined);
}

void addInputs(
    Node* n,
    const char* name,
    at::ITensorListRef value,
    bool allow_undefined) {
  // 获取当前节点所属的图
  Graph* g = n->owningGraph();
  Node* list_node = nullptr;
  if (allow_undefined) {
    // 如果允许未定义，创建一个包含可选张量的列表节点
    list_node = g->insertNode(
        g->createList(OptionalType::ofTensor(), fmap(value, getValueTrace)));
  } else {
    // 否则，创建一个包含张量的列表节点
    list_node = g->insertNode(
        g->createList(TensorType::get(), fmap(value, getValueTrace)));
  }
  // 将列表节点的输出作为当前节点的输入
  n->addInput(list_node->output());
}

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const List<std::optional<at::Tensor>>& value) {
  // 获取当前节点所属的图
  Graph* g = n->owningGraph();
  // 创建一个包含可选张量的列表节点
  Node* list_node = g->insertNode(g->createList(
      OptionalType::ofTensor(), fmap(value, getOptTensorValueTrace)));
  // 将列表节点的输出作为当前节点的输入
  n->addInput(list_node->output());
}

void addInputs(
    Node* n,
    const char* name,
    ArrayRef<c10::intrusive_ptr<c10::ivalue::Object>> value,
    const ClassTypePtr& class_type) {
  // 获取当前节点所属的图
  Graph* g = n->owningGraph();
  // 创建一个包含指定类型的对象列表节点
  Node* list_node =
      g->insertNode(g->createList(class_type, fmap(value, getValueTrace)));
  // 将列表节点的输出作为当前节点的输入
  n->addInput(list_node->output());
}

void addInputs(Node* n, const char* name, at::IntArrayRef value) {
  using ArgumentStash = jit::tracer::ArgumentStash;
  // 根据名称检索或创建整数数组参考的信息
  std::vector<Value*> info = ArgumentStash::hasIntArrayRef(name)
      ? ArgumentStash::popIntArrayRef(name)
      : ArgumentStash::IntArrayRefTrace(value.size());

  auto& g = getTracingState()->graph;
  // 遍历整数数组的每个元素
  for (const auto i : c10::irange(info.size())) {
    // 如果信息已经存在，则跳过
    if (info[i] != nullptr)
      continue;
    // 否则，将整数数组的元素插入为常量节点
    info[i] = g->insertConstant(value[i]);
    // 记录节点的源位置信息
    recordSourceLocation(info[i]->node());
  }
  // 检查所有值是否为整数类型
  for (jit::Value* v : info) {
    if (*v->type() != *jit::IntType::get()) {
      throw std::runtime_error(
          "Type mismatch in setposattr for IntArrayRef. Check that your program "
          "is valid without tracing, and please file a bug report if it is.");
    }
  }
  // 创建一个整数类型的列表节点，并将其输出作为当前节点的输入
  n->addInput(
      g->insertNode(g->createList(jit::IntType::get(), info))->output());
}

void addInputs(Node* n, const char* name, c10::SymIntArrayRef value) {
  // 将符号整数数组参考转换为整数数组参考后添加输入
  addInputs(n, name, C10_AS_INTARRAYREF_SLOW(value));
}

void addInputs(Node* n, const char* name, std::optional<c10::SymInt> value) {
  // 如果值存在，则转换为 c10::SymInt，然后添加输入
  addInputs(
      n,
      name,
      value.has_value()
          ? c10::make_optional(value->guard_int(__FILE__, __LINE__))
          : c10::nullopt);
}
    // 定义函数 genericAddOptionalInput 的参数和返回类型，使用了 std::optional 包装的 IntArrayRef 引用
    const std::optional<at::IntArrayRef>& opt_value) {
    // 调用 detail 命名空间下的 genericAddOptionalInput 函数，传入参数 n, name 和 opt_value
    detail::genericAddOptionalInput(n, name, opt_value);
}

void addInputs(
    Node* n,
    const char* name,
    const at::OptionalIntArrayRef& opt_value) {
  // 如果传入的可选数组引用有值
  if (opt_value.has_value()) {
    // 使用跟踪器向节点 n 添加输入，传递名称和实际值
    jit::tracer::addInputs(n, name, *opt_value);
  } else {
    // 否则，获取节点 n 所属的图
    Graph* g = n->owningGraph();
    // 创建一个 None 值的节点，并获取其输出
    Value* none = g->insertNode(g->createNone())->output();
    // 将 None 值作为输入添加到节点 n 中
    n->addInput(none);
  }
}

void addInputs(
    Node* n,
    const char* name,
    const at::OptionalSymIntArrayRef& opt_value) {
  // 如果传入的可选符号整数数组引用有值
  if (opt_value.has_value()) {
    // 使用跟踪器向节点 n 添加输入，传递名称和实际值
    jit::tracer::addInputs(n, name, *opt_value);
  } else {
    // 否则，获取节点 n 所属的图
    Graph* g = n->owningGraph();
    // 创建一个 None 值的节点，并获取其输出
    Value* none = g->insertNode(g->createNone())->output();
    // 将 None 值作为输入添加到节点 n 中
    n->addInput(none);
  }
}

void addInputs(Node* n, const char* name, ArrayRef<double> value) {
  // 创建一个空的值信息向量
  std::vector<Value*> info;
  // 获取当前跟踪状态的图
  auto& g = getTracingState()->graph;
  // 遍历双精度浮点数数组中的每个元素
  for (double elt : value) {
    // 插入一个表示该元素的常量节点，并将其添加到信息向量中
    info.push_back(g->insertConstant(elt));
    // 记录该节点的源位置信息
    recordSourceLocation(info.back()->node());
  }
  // 创建一个浮点类型的列表节点，将 info 中的值作为输入，并获取其输出
  n->addInput(
      g->insertNode(g->createList(jit::FloatType::get(), info))->output());
}

void addInputs(
    Node* n,
    const char* name,
    const std::optional<c10::ArrayRef<double>>& opt_value) {
  // 使用细节函数添加可选输入节点到节点 n 中
  detail::genericAddOptionalInput(n, name, opt_value);
}

void addInputs(
    Node* n,
    const char* name,
    const c10::intrusive_ptr<c10::ivalue::Object>& obj) {
  // 获取对象的值追踪
  Value* v = getValueTrace(obj);
  // 将该值作为输入添加到节点 n 中
  n->addInput(v);
}

void addOutput(Node* node, const at::Tensor& output) {
  // 设置节点的输出值为给定的张量输出
  setOutput(node->addOutput(), output);
}

void setOutput(Value* value, const at::Tensor& output) {
  // 如果输出张量已定义
  if (output.defined()) {
    // 推断值的类型与输出张量相同
    value->inferTypeFrom(output);
    // 设置输出张量的值追踪
    setValueTrace(output, value);
  }
}

void addOutput(Node* node, const std::vector<at::Tensor>& outputs) {
  // 将节点的输出类型设置为张量列表类型
  Value* value = node->addOutput()->setType(ListType::ofTensors());
  // 获取节点所属的图
  Graph* graph = node->owningGraph();
  // 创建一个列表解包节点，并将其插入到图中
  Node* unpack_node = graph->insertNode(
      graph->create(prim::ListUnpack, {value}, outputs.size()));
  // 遍历输出张量列表的每个元素
  for (const auto i : c10::irange(outputs.size())) {
    // 获取解包节点的输出值
    Value* output_val = unpack_node->outputs()[i];
    // 推断该值的类型与对应的输出张量相同
    output_val->inferTypeFrom(outputs[i]);
    // 设置输出张量的值追踪
    setValueTrace(outputs[i], output_val);
  }
}

void addOutput(Node* node, const c10::List<at::Tensor>& outputs) {
  // 调用上面定义的重载函数，将 c10::List 转换为 std::vector 并添加输出
  return addOutput(node, outputs.vec());
}

void addOutput(
    Node* node,
    const c10::intrusive_ptr<c10::ivalue::Object>& output) {
  // 添加对象类型的输出到节点 n 中
  Value* output_val = node->addOutput();
  // 推断该值的类型与输出对象相同
  output_val->inferTypeFrom(output);
  // 设置输出对象的值追踪
  setValueTrace(output, output_val);
}

const std::shared_ptr<TracingState>& getTracingState() {
  // 返回当前跟踪状态的共享指针
  return detail::tracing_state;
}

void setTracingState(std::shared_ptr<TracingState> state) {
  // 根据给定的状态设置调度是否启用
  at::tracer::impl::set_dispatch_enabled(state != nullptr);
  // 设置跟踪状态的共享指针为给定的状态
  detail::tracing_state = std::move(state);
}

TracingState::TracingState() : graph(new Graph()), env_stack{Frame()} {}

TracingState::~TracingState() = default;

autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim) {
  // 获取当前跟踪状态和图
  auto& tracing_state = getTracingState();
  auto& graph = tracing_state->graph;

  // 创建一个变量大小的变量
  Variable size_var;
  {
    // 确保该标量到张量不被追踪！
    // 这里可以添加更多详细说明...
  }
    // 创建一个自动调度保护，确保后续操作在AD内或视图下进行
    at::AutoDispatchBelowADInplaceOrView guard;
    // 将变量在指定维度上的大小转换为张量并赋给size_var
    size_var = scalar_to_tensor(at::Scalar(var.size(dim)));
  }
  // 获取变量的值的跟踪信息
  auto* value = getValueTrace(var);
  // 在计算图中插入一个常量节点，表示维度dim
  auto dim_val = graph->insertConstant(dim);
  // 记录节点的源位置信息
  recordSourceLocation(dim_val->node());
  // 在计算图中插入一个aten::size节点，计算变量在指定维度上的大小
  auto* node = graph->insertNode(graph->create(aten::size, {value, dim_val}));
  // 记录节点的源位置信息
  recordSourceLocation(node);
  // 设置节点的输出类型为整数类型
  node->output()->setType(jit::IntType::get());

  // 将节点的输出插入一个新的节点，将其转换为张量形式，并获取其输出
  auto ten =
      graph->insertNode(graph->createNumToTensor(node->output()))->output();
  // 设置size_var的值的跟踪信息，表示变量在指定维度上的大小
  setValueTrace(size_var, ten);
  // 返回计算得到的size_var
  return size_var;
}

// 获取变量的元素数量作为 Variable 对象返回
autograd::Variable getNumelOf(const autograd::Variable& var) {
    // 获取当前追踪状态和图
    auto& tracing_state = getTracingState();
    auto& graph = tracing_state->graph;

    // 创建一个空的 Variable 对象
    Variable numel_var;
    {
        // 确保这个标量到张量的转换不会被追踪
        at::AutoDispatchBelowADInplaceOrView guard;
        // 将标量转换为张量
        numel_var = scalar_to_tensor(at::Scalar(var.numel()));
    }
    // 获取变量的值的追踪信息
    auto* value = getValueTrace(var);
    // 插入一个新的节点到图中，表示 numel 操作
    auto* node = graph->insertNode(graph->create(Symbol::aten("numel"), {value}));
    // 记录源码位置
    recordSourceLocation(node);
    // 设置节点输出类型为整数类型
    node->output()->setType(jit::IntType::get());

    // 将节点的输出插入新的节点中，并返回输出
    auto ten = graph->insertNode(graph->createNumToTensor(node->output()))->output();
    // 设置变量的值的追踪信息
    setValueTrace(numel_var, ten);
    // 返回元素数量的 Variable 对象
    return numel_var;
}

// 如果是非原地操作，确保张量是唯一的
void ensureUniqueIfOutOfPlaced(const char* name, const at::Tensor& tensor) {
    auto& state = getTracingState();
    // 如果追踪状态存在且不强制使用原地操作，则直接返回
    if (state && state->force_outplace == false) {
        // 如果不将原地操作转换为非原地操作，这个检查是不必要的
        return;
    }
    // 获取张量的引用数
    auto aliases = tensor.storage().use_count();
    // 如果正在追踪并且引用数大于1，则发出警告
    if (isTracing() && aliases > 1) {
        std::stringstream ss;
        ss << "There are " << aliases
           << " live references to the data region being modified when tracing in-place operator "
           << name
           << ". This might cause the trace to be incorrect, because all other views "
           << "that also reference this data will not reflect this change in the trace! "
           << "On the other hand, if all other views use the same memory chunk, but are disjoint (e.g. "
           << "are outputs of torch.split), this might still be safe.";
        // 发出警告信息
        warn(ss.str().c_str());
    }
}

// 如果是非原地操作，确保张量是唯一的（重载版本，处理可选的张量）
void ensureUniqueIfOutOfPlaced(
    const char* name,
    const std::optional<at::Tensor>& tensor) {
    // 如果张量存在，则调用上面的函数处理，否则传入一个空张量
    ensureUniqueIfOutOfPlaced(name, tensor.has_value() ? *tensor : at::Tensor());
}

////////////////////////////////////////////////////////////////////////////////
// 参数存储
////////////////////////////////////////////////////////////////////////////////

// 参数存储的线程局部对象
thread_local ArgumentStash ArgumentStash::stash;

// 存储整数数组的元素
void ArgumentStash::stashIntArrayRefElem(
    const std::string& arg_name,
    size_t size,
    size_t idx,
    const Variable& var) {
    // TODO: 检查类型？
    // 如果不在追踪状态，则直接返回
    if (!isTracing())
        return;
    // 获取整数数组追踪信息的引用
    IntArrayRefTrace& list_trace =
        stash.intlists.emplace(arg_name, size).first->second;
    // 断言数组大小和索引的有效性
    AT_ASSERT(size == list_trace.size());
    AT_ASSERT(idx < list_trace.size());
    AT_ASSERT(list_trace[idx] == nullptr);

    // 获取变量的值的追踪信息
    Value* ten = getValueTrace(var);
    // 获取变量所属的图
    auto& g = *ten->owningGraph();
    // 设置插入点为当前节点的下一个节点
    WithInsertPoint guard(ten->node()->next());
    // 在图中插入整数操作节点，并将张量作为参数
    auto prim = g.insert(aten::Int, {ten});
    // 将整数操作节点存储到整数数组追踪信息中的指定位置
    list_trace[idx] = prim;
}

// 存储值的追踪信息
void ArgumentStash::stashValue(
    const std::string& arg_name,
    size_t idx,
    const Variable& var,
    const TypePtr& type) {
    // 如果不在追踪状态，则直接返回
    if (!isTracing())
        return;

    // 获取变量的值的追踪信息
    Value* ten = getValueTrace(var);
    // 设置插入点为当前节点的下一个节点
    WithInsertPoint guard(ten->node()->next());
    // 获取变量所属的图
    auto& g = *ten->owningGraph();

    // 根据类型插入不同的操作节点
    if (type == IntType::get()) {
        ten = g.insert(aten::Int, {ten});
    } else if (type == FloatType::get()) {
    # 如果类型是浮点数类型
    ten = g.insert(aten::Float, {ten});
  else:
    # 否则，假设类型是NumberType
    ten = g.insert(aten::ScalarImplicit, {ten});

  # 将处理后的值存入stash的values中，以参数名arg_name作为键
  stash.values.emplace(arg_name, ten);
} // namespace torch::jit::tracer
////////////////////////////////////////////////////////////////////////////////
// Stack trace recording
////////////////////////////////////////////////////////////////////////////////
// 如果没有 Python 环境，则不记录源代码信息
// 默认函数：记录节点的源代码位置
static void defaultRecordSourceLocation(Node* n) {}

// 原子操作：记录源代码位置的函数指针
std::atomic<decltype(&defaultRecordSourceLocation)> record_source_location(
    defaultRecordSourceLocation);

// 设置记录节点源代码位置的函数
void recordSourceLocation(Node* n) {
  return record_source_location.load()(n);
}

// 设置记录源代码位置的函数
void setRecordSourceLocation(void (*v)(Node*)) {
  record_source_location.store(v);
}

// 默认函数：返回空的 Python 调用堆栈
static std::vector<StackEntry> defaultPythonCallstack() {
  return std::vector<StackEntry>();
}

// 原子操作：Python 调用堆栈函数指针
std::atomic<decltype(&defaultPythonCallstack)> python_callstack_fn(
    defaultPythonCallstack);

// 返回当前 Python 调用堆栈
std::vector<StackEntry> pythonCallstack() {
  return python_callstack_fn.load()();
}

// 设置 Python 调用堆栈函数
void setPythonCallstack(std::vector<StackEntry> (*v)()) {
  python_callstack_fn.store(v);
}

// 默认警告函数：使用 TORCH_WARN 输出警告信息
static void defaultWarn(const std::string& str) {
  TORCH_WARN(str);
}

// 原子操作：警告回调函数指针
std::atomic<warn_fn_type> warn_callback{defaultWarn};

// 警告信息：Python 数据流跟踪可能不准确的提示
const char* WARN_PYTHON_DATAFLOW =
    " might cause the trace to be incorrect. We can't record the data flow of "
    "Python values, so this value will be treated as a constant in the future. "
    "This means that the trace might not generalize to other inputs!";

// 警告信息：构造函数结果在跟踪中被注册为常数的提示
const char* WARN_CONSTRUCTOR =
    " results are registered as constants in the trace. You can safely ignore this "
    "warning if you use this function to create tensors out of constant variables "
    "that would be the same every time you call this function. In any other case, "
    "this might cause the trace to be incorrect.";

// 警告信息：无法在 JIT 中表示调整大小的提示
const char* WARN_RESIZE =
    " can't be represented in the JIT at the moment, so we won't connect any uses of "
    "this value with its current trace. If you happen to use it again, it will show "
    "up as a constant in the graph. Consider using `view` or `reshape` to make "
    "it traceable.";

// 严格追踪器消息：容器结构根据模块输入是否改变的提示
const char* STRICT_TRACER_MSG =
    " might cause the trace to be incorrect, this is only valid if the container "
    "structure does not change based on the module's inputs. Consider using a constant "
    "container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a "
    "`NamedTuple` instead). If you absolutely need this and know the side effects, pass "
    "strict=False to trace() to allow this behavior.";

// XXX: _kind 可能为空指针
// 执行警告处理：组合原因和类型并调用警告回调
void _do_warn(const char* _reason, const char* _kind) {
  std::string reason{_reason};
  std::string kind{_kind ? _kind : ""};
  std::ostringstream s;
  s << reason << kind;
  warn_callback.load()(s.str());
}

// 设置警告处理函数
void setWarn(warn_fn_type fn) {
  warn_callback.store(fn);
}
```