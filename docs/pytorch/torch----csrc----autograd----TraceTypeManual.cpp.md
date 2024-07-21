# `.\pytorch\torch\csrc\autograd\TraceTypeManual.cpp`

```py
#include <ATen/TracerMode.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/library.h>

using namespace at;  // 使用 at 命名空间

namespace torch {  // 定义 torch 命名空间
namespace TraceType {  // 定义 TraceType 命名空间

namespace {  // 匿名命名空间，限定符内部变量的作用域

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  jit::Value* output = nullptr;  // 定义指向 JIT 值的指针，初始化为 nullptr
  if (torch::jit::tracer::isTracing()) {  // 如果正在进行追踪
    const jit::tracer::TracingState& state = *jit::tracer::getTracingState();  // 获取追踪状态
    auto& graph = state.graph;  // 获取追踪图
    if (state.force_outplace && self.storage().use_count() <= 1) {
      // 如果强制使用非就地操作，并且 self 没有其他视图，就地复制等同于将 src 扩展到与 self 相同的大小
      jit::Node* node = graph->create(jit::aten::expand_as, /*num_outputs=*/1);  // 创建扩展节点
      jit::tracer::addInputs(node, "src", src);  // 添加输入节点 src
      jit::tracer::addInputs(node, "self", self);  // 添加输入节点 self
      graph->insertNode(node);  // 插入节点到图中
      output = node->output();  // 设置输出为节点的输出
    } else {
      output = graph->insert(  // 否则插入复制节点
          jit::aten::copy_,
          {jit::tracer::getValueTrace(self), jit::tracer::getValueTrace(src)});
      jit::tracer::recordSourceLocation(output->node());  // 记录节点的源位置
    }
    jit::tracer::ensureUniqueIfOutOfPlaced(
        "copy_ (possibly due to an assignment)", self);  // 确保操作是唯一的，以防止就地操作可能引起的问题
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;  // 使用无追踪模式保护
    self.copy_(src, non_blocking);  // 执行 Tensor 的复制操作
  }

  if (torch::jit::tracer::isTracing()) {
    jit::tracer::setOutput(output, self);  // 设置追踪输出
  }
  return self;  // 返回自身 Tensor
}

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  if (torch::jit::tracer::isTracing()) {  // 如果正在进行追踪
    if (jit::tracer::ArgumentStash::hasIntArrayRef("size")) {
      jit::tracer::ArgumentStash::popIntArrayRef("size");  // 弹出 IntArrayRef "size"
    }
    jit::tracer::warn("resize_", jit::tracer::WARN_RESIZE);  // 发出调整大小的警告
    jit::tracer::delValueTrace(self);  // 删除 Tensor 的追踪值
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;  // 使用无追踪模式保护
    self.resize_(size, optional_memory_format);  // 执行 Tensor 的调整大小操作
  }
  return self;  // 返回自身 Tensor
}

const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    std::optional<MemoryFormat> optional_memory_format) {
  if (torch::jit::tracer::isTracing()) {  // 如果正在进行追踪
    jit::tracer::warn("resize_as_", jit::tracer::WARN_RESIZE);  // 发出调整大小的警告
    jit::tracer::delValueTrace(self);  // 删除 Tensor 的追踪值
  }

  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;  // 使用无追踪模式保护
    self.resize_as_(the_template, optional_memory_format);  // 执行 Tensor 的按照模板调整大小操作
  }
  return self;  // 返回自身 Tensor
}

Tensor detach(const Tensor& self) {
  torch::jit::Node* node = nullptr;  // 定义 JIT 节点指针，初始化为 nullptr
  if (jit::tracer::isTracing()) {  // 如果正在进行追踪
    auto& graph = jit::tracer::getTracingState()->graph;  // 获取追踪状态的图
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);  // 创建 detach 节点
    jit::tracer::recordSourceLocation(node);  // 记录节点的源位置
    jit::tracer::addInputs(node, "self", self);  // 添加输入节点 self
    graph->insertNode(node);  // 插入节点到图中
  }

  auto result = [&]() {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;  // 使用无追踪模式保护
    // 执行 Tensor 的 detach 操作，并返回结果
    return self.detach();
  }();

  return result;  // 返回 detach 后的 Tensor
}
    return self.detach();
  }();

  # 如果当前处于追踪模式，则将节点和结果添加到追踪器中
  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, result);
  }
  # 返回计算结果
  return result;
}

// 函数：将 Tensor 的追踪分离
Tensor& detach_(Tensor& self) {
  // 初始化一个用于追踪的节点指针
  torch::jit::Node* node = nullptr;
  
  // 如果当前正在进行追踪
  if (jit::tracer::isTracing()) {
    // 获取追踪状态下的计算图
    auto& graph = jit::tracer::getTracingState()->graph;
    // 创建一个 detach 节点，该节点无输出
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    // 记录源代码位置到节点
    jit::tracer::recordSourceLocation(node);
    // 向节点添加输入参数 "self"，即当前的 Tensor
    jit::tracer::addInputs(node, "self", self);
    // 将节点插入计算图中
    graph->insertNode(node);
    // 确保如果未分离，则唯一性分离 "detach_"
    jit::tracer::ensureUniqueIfOutOfPlaced("detach_", self);
  }

  // 禁用追踪模式，执行 Tensor 的分离操作
  {
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    self.detach_();
  }

  // 如果当前正在进行追踪
  if (jit::tracer::isTracing()) {
    // 向节点添加输出结果，即分离后的 Tensor
    jit::tracer::addOutput(node, self);
  }
  
  // 返回分离后的 Tensor
  return self;
}

// 不变条件：
// - 在下面的 DispatchKey::Tracer 中注册的操作必须包含在
//   `MANUAL_TRACER` 中，位于 tools/autograd/gen_variable_type.py 中
TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  // 注册 resize_ 函数到追踪器
  m.impl("resize_", resize_);
  // 注册 resize_as_ 函数到追踪器
  m.impl("resize_as_", resize_as_);
  // 注册 detach 函数到追踪器
  m.impl("detach", TORCH_FN(detach));
  // 注册 detach_ 函数到追踪器
  m.impl("detach_", detach_);
  // 注册 copy_ 函数到追踪器
  m.impl("copy_", copy_);

  // 对以下操作跳过追踪，通过显式注册 fallthrough 内核
  m.impl("_backward", CppFunction::makeFallthrough());
  m.impl("set_data", CppFunction::makeFallthrough());
  m.impl("data", CppFunction::makeFallthrough());
  m.impl("is_leaf", CppFunction::makeFallthrough());
  m.impl("output_nr", CppFunction::makeFallthrough());
  m.impl("_version", CppFunction::makeFallthrough());
  m.impl("requires_grad_", CppFunction::makeFallthrough());
  m.impl("retain_grad", CppFunction::makeFallthrough());
  m.impl("_fw_primal", CppFunction::makeFallthrough());
  m.impl("_make_dual", CppFunction::makeFallthrough());
}

} // namespace

} // namespace TraceType
} // namespace torch

// 命名空间：torch 中的 jit 模块
namespace torch {
// 命名空间：torch 中的 jit 模块下的子命名空间
namespace jit {
// 静态函数：通用追踪函数
static void general_trace_function(
    const c10::OperatorHandle& op,
    Stack* stack) {
  // 获取操作的输入和输出参数大小
  const auto input_size = op.schema().arguments().size();
  const auto output_size = op.schema().returns().size();

  // 初始化一个节点指针和追踪状态的共享指针
  Node* node = nullptr;
  std::shared_ptr<tracer::TracingState> tracer_state;

  // 在展开之前追踪输入，否则可能丢失输入信息
  if (tracer::isTracing()) {
    // 获取追踪状态
    tracer_state = tracer::getTracingState();
    // 从操作的模式名称中创建符号
    auto symbol = Symbol::fromQualString(op.schema().name());
    // 获取追踪状态下的计算图
    const auto& graph = tracer::getTracingState()->graph;
    // 创建一个符号节点，该节点无输出
    node = graph->create(symbol, 0);
    // 记录源代码位置到节点
    tracer::recordSourceLocation(node);
    // 获取操作的参数
    const auto& args = op.schema().arguments();
    int i = 0;
    // 将节点插入计算图中
    }
    graph->insertNode(node);

    // 禁用追踪状态
    tracer::setTracingState(nullptr);
  }

  // 调用操作的函数，执行栈上的操作
  op.callBoxed(stack);

  // 如果存在追踪状态
  if (tracer_state) {
    // 恢复追踪状态
    tracer::setTracingState(std::move(tracer_state));
    int i = 0;
    // 遍历从堆栈中取出的输出值，使用迭代器进行遍历
    for (auto iter = stack->end() - static_cast<std::ptrdiff_t>(output_size);
         iter != stack->end();
         ++iter, ++i) {
      // 获取当前输出值的类型
      const auto& type = op.schema().returns()[i].type();
      // 如果类型是 TensorType 的子类型
      if (type->isSubtypeOf(*TensorType::get())) {
        // 断言当前值是一个 Tensor
        AT_ASSERT(iter->isTensor());
        // 将 Tensor 添加到追踪器中
        tracer::addOutput(node, iter->toTensor());
      } else if (type->kind() == TypeKind::ListType) {
        // 如果类型是 ListType
        const auto& elem_type = type->expectRef<ListType>().getElementType();
        // 如果列表元素类型是 TensorType 的子类型
        if (elem_type->isSubtypeOf(*TensorType::get())) {
          // 断言当前值是一个 Tensor 列表
          AT_ASSERT(iter->isTensorList());
          // 将 Tensor 列表添加到追踪器中
          tracer::addOutput(node, iter->toTensorList());
        } else {
          // 抛出异常，表示不支持的输出列表类型
          throw std::runtime_error(
              "unsupported ouptut list type: " + elem_type->str());
        }
      } else if (type->kind() == TypeKind::ClassType) {
        // 如果类型是 ClassType
        AT_ASSERT(iter->isObject());
        // 将对象添加到追踪器中
        tracer::addOutput(node, iter->toObject());
      } else {
        // 抛出异常，表示不支持的输出类型
        throw std::runtime_error(
            "unsupported output type: " + type->str() +
            ", from operator: " + toString(op.operator_name()));
      }
    }
}
TORCH_LIBRARY_IMPL(_, Tracer, m) {
  // 实现 Torch 库的 Tracer 模块，将通用追踪函数注册为回退函数
  m.fallback(CppFunction::makeFromBoxedFunction<&general_trace_function>());
}

} // namespace jit
} // namespace torch
```