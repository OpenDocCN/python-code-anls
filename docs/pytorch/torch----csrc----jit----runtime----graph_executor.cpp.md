# `.\pytorch\torch\csrc\jit\runtime\graph_executor.cpp`

```py
// 引入 Torch JIT 的相关头文件，用于 JIT 编译和执行图的功能
#include <torch/csrc/jit/runtime/graph_executor.h>

// 引入 ATen 库中的 IValue 类型定义，用于表示任意类型的值
#include <ATen/core/ivalue.h>

// 引入 C10 库中的异常处理类定义
#include <c10/util/Exception.h>

// 引入 C10 库中的范围迭代器，用于迭代范围内的整数
#include <c10/util/irange.h>

// 引入 Torch 自动求导模块的头文件，用于管理自动求导功能的开关
#include <torch/csrc/autograd/grad_mode.h>

// 引入 Torch JIT 的前端追踪器，用于追踪程序执行过程中的图结构
#include <torch/csrc/jit/frontend/tracer.h>

// 引入 Torch JIT 的 IR 定义，用于描述 JIT 编译后的中间表示
#include <torch/csrc/jit/ir/ir.h>

// 引入 Torch JIT 的日志记录功能
#include <torch/csrc/jit/jit_log.h>

// 引入 Torch JIT 的批量矩阵乘法优化功能
#include <torch/csrc/jit/passes/batch_mm.h>

// 引入 Torch JIT 的图优化操作，规范化图中融合的操作
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>

// 引入 Torch JIT 的公共子表达式消除优化功能
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

// 引入 Torch JIT 的常量池优化功能
#include <torch/csrc/jit/passes/constant_pooling.h>

// 引入 Torch JIT 的常量传播优化功能
#include <torch/csrc/jit/passes/constant_propagation.h>

// 引入 Torch JIT 的自动微分子图创建功能
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>

// 引入 Torch JIT 的死代码消除优化功能
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 引入 Torch JIT 的操作分解优化功能
#include <torch/csrc/jit/passes/decompose_ops.h>

// 引入 Torch JIT 的图融合优化功能
#include <torch/csrc/jit/passes/graph_fuser.h>

// 引入 Torch JIT 的自动微分子图内联功能
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>

// 引入 Torch JIT 的内联函数优化功能
#include <torch/csrc/jit/passes/inliner.h>

// 引入 Torch JIT 的原位操作检查功能
#include <torch/csrc/jit/passes/inplace_check.h>

// 引入 Torch JIT 的循环展开优化功能
#include <torch/csrc/jit/passes/loop_unrolling.h>

// 引入 Torch JIT 的梯度下降算法的梯度降低操作优化功能
#include <torch/csrc/jit/passes/lower_grad_of.h>

// 引入 Torch JIT 的元组降低优化功能
#include <torch/csrc/jit/passes/lower_tuples.h>

// 引入 Torch JIT 的通用优化流程管理器
#include <torch/csrc/jit/passes/pass_manager.h>

// 引入 Torch JIT 的微操作优化功能
#include <torch/csrc/jit/passes/peephole.h>

// 引入 Torch JIT 的扩展操作移除功能
#include <torch/csrc/jit/passes/remove_expands.h>

// 引入 Torch JIT 的变异操作移除功能
#include <torch/csrc/jit/passes/remove_mutation.h>

// 引入 Torch JIT 的梯度需求分析功能
#include <torch/csrc/jit/passes/requires_grad_analysis.h>

// 引入 Torch JIT 的形状分析功能
#include <torch/csrc/jit/passes/shape_analysis.h>

// 引入 Torch JIT 的特化自动梯度为零操作功能
#include <torch/csrc/jit/passes/specialize_autogradzero.h>

// 引入 Torch JIT 的 Tensor 表达式融合优化功能
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

// 引入 Torch JIT 的资源守卫功能，用于管理资源生命周期
#include <torch/csrc/jit/resource_guard.h>

// 引入 Torch JIT 的运行时参数规格定义
#include <torch/csrc/jit/runtime/argument_spec.h>

// 引入 Torch JIT 的自动微分功能
#include <torch/csrc/jit/runtime/autodiff.h>

// 引入 Torch JIT 的自定义操作运行时支持
#include <torch/csrc/jit/runtime/custom_operator.h>

// 引入 Torch JIT 的图执行器实现
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

// 引入 Torch JIT 的解释器实现
#include <torch/csrc/jit/runtime/interpreter.h>

// 引入 Torch JIT 的性能分析图执行器实现
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>

// 引入 Torch JIT 的性能记录功能
#include <torch/csrc/jit/runtime/profiling_record.h>

// 引入 Torch JIT 的简单图执行器实现
#include <torch/csrc/jit/runtime/simple_graph_executor_impl.h>

// 引入 Torch 自动求导的边和函数定义
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>

// 引入 Torch JIT 的图执行器优化更新功能
#include <torch/csrc/jit/python/update_graph_executor_opt.h>

// 引入 Torch JIT 的运行时日志记录功能
#include <torch/csrc/jit/runtime/logging.h>

// 引入标准库头文件
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

// 定义 C10 库中的一个布尔类型标记，用于控制 JIT 执行计划是否重用代码图
C10_DEFINE_bool(
    torch_jit_execution_plan_reuse_code_graph,
    false,
    "Directly reuse the preprocessed graph in the CodeImpl to reduce the memory consumption. This is aggressive memory saving, and please be cautious!");

// Torch JIT 命名空间
namespace torch::jit {

// 启用性能分析守卫类的构造函数实现
EnableProfilingGuard::EnableProfilingGuard() {
  // 获取当前的执行模式并保存旧的执行模式
  auto& executor_mode = getExecutorMode();
  old_executor_mode = executor_mode;
  // 设置执行模式为启用状态
  executor_mode = true;
  // 保存旧的图执行器优化设置，并设置图执行器优化为启用状态
  old_get_optimize = getGraphExecutorOptimize();
  setGraphExecutorOptimize(true);
}

// 启用性能分析守卫类的析构函数实现
EnableProfilingGuard::~EnableProfilingGuard() {
  // 恢复保存的旧的执行模式
  getExecutorMode() = old_executor_mode;
  // 恢复保存的旧的图执行器优化设置
  setGraphExecutorOptimize(old_get_optimize);
}

// Torch JIT 命名空间的私有匿名命名空间
namespace {
c10::AliasAnalysisKind aliasAnalysisInternalSpecialCase() {
  // 返回内部特殊情况的别名分析类型
  return AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}
} // namespace

// 用于调试，有助于强制创建自动微分子图，
// 以便在子图太小以至于不划算时检查其正确性。
thread_local bool autodiff_subgraph_inlining = true;
void debugSetAutodiffSubgraphInlining(bool state) {
  // 设置自动微分子图内联的状态
  autodiff_subgraph_inlining = state;
}

bool getAutodiffSubgraphInlining() {
  // 获取自动微分子图内联的状态
  return autodiff_subgraph_inlining;
}

// 用于调试，有助于强制创建融合组
static std::atomic<bool> fusion_group_inlining(true);
void debugSetFusionGroupInlining(bool state) {
  // 设置融合组内联的状态
  fusion_group_inlining = state;
}

bool getFusionGroupInlining() {
  // 获取融合组内联的状态
  return fusion_group_inlining;
}

// 上下文线程本地变量，用于存储上次执行优化图的弱引用
thread_local std::weak_ptr<Graph> last_executed_optimized_graph;
std::shared_ptr<Graph> lastExecutedOptimizedGraph() {
  // 返回上次执行优化图的弱引用的共享指针
  return last_executed_optimized_graph.lock();
}
namespace {

using tensor_list = std::vector<at::Tensor>;
using Variable = autograd::Variable;
using autograd::variable_list;

struct CaptureList {
  CaptureList(size_t capture_size) {
    // 初始化捕获列表，预留空间
    capture_types_.reserve(capture_size);
    var_captures_.reserve(capture_size); // var_captures_.size() might be
                                         // greater than capture_size
    ivalue_captures_.reserve(capture_size);
  }

  void captureTensor(const at::Tensor& tensor, bool is_output) {
    // 捕获张量，并标记是否为输出
    var_captures_.emplace_back(Variable(tensor), is_output);
  }

  void capture(const IValue& val, bool is_output) {
    if (val.isTensor()) {
      // 对于张量，将其捕获类型设置为 CAPTURE_TENSOR
      capture_types_.emplace_back(CAPTURE_TENSOR);
      captureTensor(val.toTensor(), is_output);
    } else if (val.isTensorList()) {
      // 对于张量列表，将其捕获类型设置为 CAPTURE_LIST
      // 在保存期间将其扁平化为张量，在反向应用中恢复为张量列表
      // 这是为了避免在前向和后向之间发生对张量列表的隐式变异
      capture_types_.emplace_back(CAPTURE_LIST);
      auto tensors = val.toTensorList();
      sizes_.push_back(tensors.size());

      for (const auto& tensor : tensors) {
        captureTensor(tensor, is_output);
      }
    } else {
      // 对于其他类型的值，将其捕获类型设置为 CAPTURE_IVALUE
      capture_types_.emplace_back(CAPTURE_IVALUE);
      ivalue_captures_.push_back(val);
    }
  }

  size_t size() const {
    // 返回捕获列表的大小
    return capture_types_.size();
  }

  void unpack(Stack& stack, const std::shared_ptr<autograd::Node>& saved_for) {
    auto var_capture_it = var_captures_.begin();
    auto ivalue_capture_it = ivalue_captures_.begin();
    auto size_it = sizes_.begin();
    for (Capture capture_type : capture_types_) {
      // 遍历捕获类型列表
      switch (capture_type) {
        case CAPTURE_TENSOR: {
          // 如果是捕获张量类型
          stack.emplace_back(var_capture_it->unpack(saved_for));
          // 将解包后的变量压入堆栈，并移动迭代器到下一个变量
          ++var_capture_it;
        } break;
        case CAPTURE_LIST: {
          // 如果是捕获列表类型
          c10::List<at::Tensor> lst;
          auto size = *size_it++;
          // 获取列表的大小，并移动大小迭代器到下一个大小
          for (const auto i : c10::irange(size)) {
            (void)i; // 确保编译器不会发出未使用变量的警告
            lst.emplace_back(var_capture_it->unpack(saved_for));
            // 将解包后的变量加入列表，并移动变量迭代器到下一个变量
            var_capture_it++;
          }
          stack.emplace_back(std::move(lst));
          // 将填充好的列表压入堆栈
        } break;
        case CAPTURE_IVALUE: {
          // 如果是捕获 IValue 类型
          stack.push_back(*ivalue_capture_it++);
          // 将 IValue 值压入堆栈，并移动 IValue 迭代器到下一个值
        } break;
      }
    }
  }

  void release_variables() {
    for (auto& var_capture_ : var_captures_) {
      var_capture_.reset_data();
      // 释放变量捕获的数据
    }
  }

 private:
  enum Capture : uint8_t {
    CAPTURE_TENSOR,
    CAPTURE_LIST,
    CAPTURE_IVALUE,
  };

  std::vector<Capture> capture_types_;
  // 捕获类型列表
  std::vector<autograd::SavedVariable> var_captures_;
  // 保存的变量列表
  std::vector<IValue> ivalue_captures_;
  // IValue 类型的捕获列表
  std::vector<size_t> sizes_;
  // 大小列表
};

// 结构体 UnpackInstructions：用于将扁平化的张量列表转换回 DifferentiableGraphBackward 预期的 IValues 指令集
struct UnpackInstructions {
  // 构造函数，初始化指令集大小
  UnpackInstructions(size_t num_inputs) {
    insts_.reserve(num_inputs);
  }
  // 添加推送张量的指令
  void pushTensor() {
    insts_.emplace_back(PUSH_TENSOR);
  }
  // 添加推送空值的指令
  void pushNone() {
    insts_.emplace_back(PUSH_NONE);
  }
  // 添加推送张量列表的指令，同时记录列表大小
  void pushTensorList(size_t size) {
    insts_.emplace_back(PUSH_LIST);
    sizes_.push_back(size);
  }
  // 解包函数，根据指令集将输入解包到堆栈中
  void unpack(variable_list&& inputs, Stack& stack) {
    auto input_it = std::make_move_iterator(inputs.begin());
    auto sizes_it = sizes_.begin();
    // 遍历指令集
    for (Inst inst : insts_) {
      switch (inst) {
        case PUSH_TENSOR: {
          // 推送张量到堆栈
          at::Tensor t = *input_it++;
          stack.emplace_back(std::move(t));
        } break;
        case PUSH_LIST: {
          // 推送张量列表到堆栈
          std::vector<at::Tensor> lst(input_it, input_it + *sizes_it++);
          stack.emplace_back(lst);
        } break;
        case PUSH_NONE: {
          // 推送空值到堆栈
          stack.emplace_back();
        }
      }
    }
  }

 private:
  // 指令类型枚举
  enum Inst : uint8_t {
    PUSH_TENSOR,  // 推送张量
    PUSH_LIST,    // 推送列表（消耗一个大小）
    PUSH_NONE,    // 推送空值
  };
  std::vector<Inst> insts_;    // 指令集合
  std::vector<size_t> sizes_;  // 大小列表
};

// 解包由 `packReturnValuesIntoTuple` 打包的返回元组
static void unpackReturnTuple(Stack& stack) {
  auto tuple = pop(stack).toTuple();  // 弹出堆栈中的元组
  stack.insert(stack.end(), tuple->elements().begin(), tuple->elements().end());  // 将元组元素插入堆栈末尾
}

// 结构体 DifferentiableGraphBackward：继承自 autograd::Node，用于反向自动微分的图形处理
struct DifferentiableGraphBackward : public autograd::Node {
  DifferentiableGraphBackward(
      GraphExecutor executor,
      size_t input_size,
      size_t capture_size)
      : executor(std::move(executor)),
        captures_(capture_size),
        input_instructions_(input_size) {}

  // 应用函数，执行反向自动微分的操作
  variable_list apply(variable_list&& inputs) override {
    Stack stack;  // 堆栈，用于存储执行过程中的数据
    stack.reserve(captures_.size() + inputs.size());  // 预留堆栈空间

    input_instructions_.unpack(std::move(inputs), stack);  // 使用指令集解包输入到堆栈中
    captures_.unpack(stack, shared_from_this());  // 解包捕获数据到堆栈中
    GRAPH_DEBUG("Running DifferentiableGraphBackward for ", &executor);  // 调试信息，显示执行反向微分的图形
    executor.run(stack);  // 执行图形处理
    unpackReturnTuple(stack);  // 解包返回的元组到堆栈中

    // 注意事项：堆栈大小不总是等于 num_outputs()
    // 特别是在添加 TensorList 支持后
    // 示例：aten::stack(Tensor[] tensors, int)，其中
    // tensors = [x, x]
    // 这里 stack.size() 为 1，有一个 TensorList IValue 的后向图输出。
    // 然而 num_outputs() 为 2，这是 grad_fn（autograd::Node）的输出数，它们是关于张量/变量 `x` 的梯度，但不是图形输入的 TensorList [x, x]。
    // 这两个梯度将稍后使用 autograd::InputBuffer 累积到 x.grad 中。
    variable_list outputs;  // 输出列表
    outputs.reserve(num_outputs());  // 预留输出列表空间
    size_t output_index = 0;  // 输出索引初始化为 0
    // 遍历栈中的每个值
    for (IValue& v : stack) {
      // 如果当前值是一个张量列表
      if (v.isTensorList()) {
        // 遍历张量列表中的每个张量，并将其输出
        for (at::Tensor tensor : v.toTensorList()) {
          produceOutput(output_index++, std::move(tensor), outputs);
        }
      } else if (v.isTensor()) {
        // 如果当前值是一个单个张量
        if (!v.toTensor().defined()) {
          // 如果张量未定义，则可能对应于一个张量列表
          if (input_tensor_lists_.count(output_index) != 0) {
            // 获取对应输出索引处的张量列表大小
            size_t list_size = input_tensor_lists_[output_index];
            // 为每个列表项生成一个空输出
            for (size_t i = 0; i < list_size; i++) {
              produceOutput(output_index++, {}, outputs);
            }
          } else {
            // 否则生成一个空输出
            produceOutput(output_index++, {}, outputs);
          }
        } else {
          // 如果张量已定义，则将其输出
          produceOutput(output_index++, std::move(v).toTensor(), outputs);
        }
      } else {
        // 如果当前值为空值，则进行断言和处理
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(v.isNone());
        output_index++;
        // 输入梯度也可以是空的，即使需要梯度
        // 例如：在 expand_as(self, other) 中的 `other`
        outputs.emplace_back();
      }
    }
    // 断言检查生成的输出数量与预期的数量是否一致
    TORCH_INTERNAL_ASSERT(
        num_outputs() == outputs.size(),
        "DifferentiableGraphBackward: expected ",
        num_outputs(),
        " outputs but found ",
        outputs.size());
    // 返回所有生成的输出
    return outputs;
  }

  // 捕获给定的值，并记录是否作为输出
  void capture(const IValue& val, bool is_output) {
    captures_.capture(val, is_output);
  }

  // 为给定的张量添加输出
  void addOutputForTensor(const at::Tensor& tensor) {
    // 将张量封装为变量，并添加相应的边
    auto v = Variable(tensor);
    add_next_edge(
        v.defined() ? torch::autograd::impl::gradient_edge(v)
                    : autograd::Edge{});
  }

  // 为给定的 IValue 添加输出
  void addOutputForIValue(const IValue& value) {
    if (value.isTensorList()) {
      // 如果值是张量列表，记录其大小，并为每个张量调用添加输出函数
      input_tensor_lists_.insert({index_, value.toTensorList().size()});
      for (const at::Tensor& tensor : value.toTensorList()) {
        addOutputForTensor(tensor);
        index_++;
      }
    } else if (value.isTensor()) {
      // 如果值是单个张量，直接调用添加输出函数
      addOutputForTensor(value.toTensor());
      index_++;
    } else {
      // 否则，可能是通过 `Optional[Tensor]` 传递了 None
      add_next_edge(autograd::Edge{});
      index_++;
    }
  }

  // 为给定的变量添加输入
  void addInputVariable(Variable output) {
    // 注意：由于我们的 requires_grad 设置只是一个启发式方法，
    // 可能希望通过积分张量进行微分，这在 autograd 中通常是一个困难的错误。
    if (at::isFloatingType(output.scalar_type()) ||
        at::isComplexType(output.scalar_type())) {
      // 如果是浮点类型或复数类型，创建相应的梯度边，并设置 requires_grad 为 true
      autograd::create_gradient_edge(output, shared_from_this());
      output.set_requires_grad(true);
    } else {
      // 否则，添加输入元数据为 undefined_input
      add_input_metadata(autograd::Node::undefined_input{});
    }
  }

  // 为给定的 IValue 添加输入
  void addInputIValue(const IValue& v) {
    if (v.isTensorList()) {
      // 如果值是张量列表，记录其长度，并为每个张量添加输入变量
      auto tensors = v.toTensorList();
      input_instructions_.pushTensorList(tensors.size());
      for (const at::Tensor& tensor : tensors) {
        addInputVariable(tensor);
      }
    } else if (v.isTensor()) {
      // 如果值是单个张量，添加相应的输入指令并添加输入变量
      input_instructions_.pushTensor();
      addInputVariable(v.toTensor());
      index_++;
    } else {
      // 否则，可能是通过 `Optional[Tensor]` 传递了 None
      add_input_metadata(autograd::Node::undefined_input{});
    }
  }
  // 如果变量 v 是空的，执行以下操作
  } else if (v.isNone()) {
    // 将 None 值推入输入指令
    input_instructions_.pushNone();
    // 向输入变量列表添加一个空的 Variable 对象
    addInputVariable(Variable{});
  }
}

// 重写父类的 release_variables 方法
void release_variables() override {
  // 释放捕获的变量
  captures_.release_variables();
}

private:
// 生成输出的私有方法，参数包括索引 i、输出张量 output 和输出变量列表 outputs 的引用
void produceOutput(size_t i, at::Tensor output, variable_list& outputs) {
  // 如果任务需要计算第 i 个输出
  if (task_should_compute_output(i)) {
    // 获取下一个边缘的引用
    const auto& edge = next_edge(i);
    // 如果输出张量已定义
    if (output.defined()) {
      // 将输出张量移动到输出变量列表中
      outputs.emplace_back(std::move(output));
    } else if (edge.is_valid()) {  // 如果边缘有效
      // 根据边缘的函数和输入号生成一个与其形状相同的零张量，并添加到输出变量列表
      outputs.emplace_back(
          edge.function->input_metadata(edge.input_nr).zeros_like());
    } else {  // 否则
      // 添加一个未定义的张量到输出变量列表
      outputs.emplace_back();
    }
  } else {  // 如果不需要计算输出
    // 添加一个未定义的张量到输出变量列表
    outputs.emplace_back();
  }
}

// 声明 ExecutionPlan 结构体为友元
friend struct ExecutionPlan;
// GraphExecutor 对象，用于执行图
GraphExecutor executor;
// 捕获列表对象
CaptureList captures_;
// 解包指令列表对象，用于处理输入
UnpackInstructions input_instructions_;
// 需要跟踪输入列表到前向图的转换，因为在反向图中这些将变成未定义张量，如果梯度为零
// 我们需要将未定义的张量转换回列表
// TODO: 切换到使用 UnpackInstructions
size_t index_ = 0;  // 索引初始化为 0
std::map<size_t, size_t> input_tensor_lists_;  // 用于跟踪输入列表的映射
};

// 结构体定义，用于优化在张量而非变量上直接执行的子图
// 这将解开变量，运行计划，并重新封装它们。
// 如果有必要，还可以连接到输出变量的梯度。
struct DifferentiableGraphOp {
  // 构造函数，初始化不同属性
  DifferentiableGraphOp(Gradient grad)
      : f_ptr(std::make_shared<GraphExecutor>(grad.f, "<forward op>")),  // 使用 grad.f 创建 GraphExecutor 对象
        legacy_f(grad.f, "<forward op>"),  // 使用 grad.f 创建 legacy_f
        grad(std::move(grad)),  // 移动 grad 到成员变量 grad
        grad_executor(this->grad.df, "<backward op>"),  // 使用 this->grad.df 创建 grad_executor
        num_inputs(this->grad.f->inputs().size()),  // 初始化 num_inputs 为 grad.f 的输入数量
        num_outputs(this->grad.f->outputs().size()) {}  // 初始化 num_outputs 为 grad.f 的输出数量

  // XXX: 注意堆栈可能比我们需要的输入更大！
  // 操作符重载，执行操作
  void operator()(Stack& stack) const {
    auto grad_fn = std::make_shared<DifferentiableGraphBackward>(
        grad_executor,
        grad.df_input_vjps.size(),
        grad.df_input_captured_inputs.size() +
            grad.df_input_captured_outputs.size());  // 创建 DifferentiableGraphBackward 对象 grad_fn

    {
      auto inputs = last(stack, num_inputs);  // 获取堆栈中最后 num_inputs 个元素作为输入
      // 将 df 的输出连接到需要梯度的输入的梯度函数
      for (auto idx : grad.df_output_vjps) {
        grad_fn->addOutputForIValue(inputs[idx]);
      }
      captureInputs(*grad_fn, inputs);  // 捕获输入
    }

    detachVariables(stack);  // 分离变量
    if (IsNewExecutorEnabled()) {
      const ExecutionPlan& plan = f_ptr->getPlanFor(stack);  // 如果新执行器已启用，则获取 stack 的执行计划
      InterpreterState(plan.code).run(stack);  // 运行解释器状态的代码
    } else {
      InterpreterState(legacy_f).run(stack);  // 否则运行 legacy_f 的解释器状态
    }

    {
      auto outputs = last(stack, num_outputs);  // 获取堆栈中最后 num_outputs 个元素作为输出
      // 将需要梯度的输出张量的梯度连接到梯度函数 df 的输入
      // TODO - XXX - 如果任何输出多次是相同张量，则必须在此处设置视图
      // 目前故意不在此处执行，以便在引入正确性开销之前获得性能的概念
      for (auto idx : grad.df_input_vjps) {
        grad_fn->addInputIValue(outputs[idx]);
      }
      captureOutputs(*grad_fn, outputs);  // 捕获输出
      // 删除临时输出，以便我们返回与不计算梯度时相同数量的输出
      const size_t num_temporary_outputs = num_outputs - grad.f_real_outputs;
      stack.erase(stack.end() - num_temporary_outputs, stack.end());
    }
  }

 private:
  friend GraphExecutor* detail::getGradExecutor(Operation& op);
  friend GraphExecutor* detail::getDifferentiableGraphOpExecutor(Operation& op);

  // 分离张量的方法
  at::Tensor detach(at::Tensor t) const {
    if (!t.defined()) {
      return t;  // 如果未定义，直接返回
    }
    return t.detach();  // 否则分离张量并返回
  }

  // 分离 IValue 的方法
  void detach(IValue& v) const {
    if (v.isTensor()) {
      v = IValue(detach(std::move(v).toTensor()));  // 如果是张量类型，则分离该张量
    }
    // 如果不是张量类型，则不做处理
  }
  } else if (v.isTensorList()) {
    // 如果变量 v 是张量列表，则将其转换为 std::vector<at::Tensor>
    std::vector<at::Tensor> lst = v.toTensorVector();
    // 遍历列表中的每个张量，并对其执行 detach 操作
    for (auto& tensor : lst) {
      tensor = detach(tensor);
    }
    // 将处理后的张量列表通过移动语义赋值给 v，原 lst 不再使用
    // NOLINTNEXTLINE(performance-move-const-arg)
    v = std::move(lst);
  }

  void detachVariables(Stack& stack) const {
    // 这里本来希望使用 ArrayRef，但是它只能返回 const 引用，因此我们需要手动索引
    const int64_t stack_size = stack.size();
    const int64_t stack_offset = stack_size - num_inputs;
    // 对栈中的变量进行 detach 操作，范围是从 stack_offset 到 stack_size 的索引
    for (const auto i : c10::irange(stack_offset, stack_size)) {
      detach(stack[i]);
    }
  }
  
  // 捕获需要保存的输入以便后续进行反向传播
  void captureInputs(
      DifferentiableGraphBackward& grad_fn,
      at::ArrayRef<IValue> inputs) const {
    // 对于在 grad.df_input_captured_inputs 中指定的偏移量，将对应的输入捕获到 grad_fn 中
    for (size_t offset : grad.df_input_captured_inputs) {
      grad_fn.capture(inputs[offset], /*is_output*/ false);
    }
  }
  
  // 捕获需要保存的输出以便后续进行反向传播
  void captureOutputs(
      DifferentiableGraphBackward& grad_fn,
      at::ArrayRef<IValue> outputs) const {
    // 对于在 grad.df_input_captured_outputs 中指定的偏移量，将对应的输出捕获到 grad_fn 中
    for (size_t offset : grad.df_input_captured_outputs) {
      grad_fn.capture(outputs[offset], /*is_output*/ true);
    }
  }

  std::shared_ptr<GraphExecutor> f_ptr;
  Code legacy_f;
  Gradient grad;
  GraphExecutor grad_executor;

  const size_t num_inputs;
  const size_t num_outputs;
};

// 获取与给定节点关联的梯度信息并返回
Gradient getGradient(const Node* n) {
  // 断言节点的类型为 DifferentiableGraph
  AT_ASSERT(n->kind() == prim::DifferentiableGraph);
  // 创建 Gradient 结构体实例
  Gradient grad;
  // 将不同属性的子图关联到 Gradient 的成员变量
  grad.f = n->g(attr::Subgraph);
  grad.df = n->g(attr::ReverseSubgraph);
  // 将整数属性值关联到 Gradient 的成员变量
  grad.f_real_outputs = n->i(attr::f_real_outputs);
  // 将布尔属性值列表转换为 size_t 的映射，关联到 Gradient 的成员变量
  grad.df_input_vjps = fmap<size_t>(n->is(attr::df_input_vjps));
  grad.df_input_captured_inputs =
      fmap<size_t>(n->is(attr::df_input_captured_inputs));
  grad.df_input_captured_outputs =
      fmap<size_t>(n->is(attr::df_input_captured_outputs));
  grad.df_output_vjps = fmap<size_t>(n->is(attr::df_output_vjps));
  // 返回构建好的 Gradient 实例
  return grad;
}
} // anonymous namespace

// 注册不同iableGraph 操作符的执行函数
RegisterOperators reg_graph_executor_ops({Operator(
    prim::DifferentiableGraph,
    [](const Node* n) -> Operation {
      // 返回一个执行 DifferentiableGraphOp 操作的函数
      return DifferentiableGraphOp(getGradient(n));
    },
    aliasAnalysisInternalSpecialCase())});

namespace detail {

// 获取操作的梯度执行器
GraphExecutor* getGradExecutor(Operation& op) {
  if (auto diff_op = op.target<DifferentiableGraphOp>()) {
    // 如果操作是 DifferentiableGraphOp，则返回其梯度执行器
    return &diff_op->grad_executor;
  }
  // 否则返回空指针
  return nullptr;
}

// 获取可微图操作的执行器
GraphExecutor* getDifferentiableGraphOpExecutor(Operation& op) {
  TORCH_INTERNAL_ASSERT(
      IsNewExecutorEnabled(),
      __FUNCTION__,
      " is only accessible under profiling executor\n");
  if (auto diff_op = op.target<DifferentiableGraphOp>()) {
    // 如果操作是 DifferentiableGraphOp，则返回其执行器
    return diff_op->f_ptr.get();
  }
  // 否则返回空指针
  return nullptr;
}
} // namespace detail

void GraphExecutorImplBase::run(Stack& stack) {
  // 检查堆栈大小是否至少为 num_inputs
  TORCH_CHECK(
      stack.size() >= num_inputs,
      "expected ",
      num_inputs,
      " inputs, but got only ",
      stack.size());

  // 记录 API 使用情况
  C10_LOG_API_USAGE_ONCE("torch.graph_executor.run");
  // 增加运行计数器的值
  logging::getLogger()->addStatValue(
      logging::runtime_counters::GRAPH_EXECUTOR_INVOCATIONS, 1.0);

  // 获取执行计划并运行解释器状态
  const ExecutionPlan& plan = getPlanFor(stack);
  InterpreterState(plan.code).run(stack);
  // 更新最后执行的优化图
  last_executed_optimized_graph = plan.graph;
}

// 异步运行图执行器
c10::intrusive_ptr<Future> GraphExecutorImplBase::runAsync(
    Stack& stack,
    TaskLauncher taskLauncher) {
  // 检查堆栈大小是否至少为 num_inputs
  TORCH_CHECK(
      stack.size() >= num_inputs,
      "expected ",
      num_inputs,
      " inputs, but got only ",
      stack.size());

  // 记录 API 使用情况
  C10_LOG_API_USAGE_ONCE("torch.graph_executor.runAsync");
  // 增加运行计数器的值
  logging::getLogger()->addStatValue(
      logging::runtime_counters::GRAPH_EXECUTOR_INVOCATIONS, 1.0);

  // 定义执行计划和状态帧结构体
  struct Frame {
    explicit Frame(ExecutionPlan eplan, TaskLauncher taskLauncher)
        : plan(std::move(eplan)), state(plan.code, std::move(taskLauncher)) {}
    ExecutionPlan plan;
    InterpreterState state;
  };
  // 创建帧的共享指针，保存执行计划和任务启动器
  auto frame =
      std::make_shared<Frame>(getPlanFor(stack), std::move(taskLauncher));
  // 异步运行状态并获取 Future 对象
  auto res = frame->state.runAsync(stack);
  // 更新最后执行的优化图
  last_executed_optimized_graph = frame->plan.graph;
  // 如果结果未完成，则添加回调函数以持久化帧
  if (!res->completed()) {
    res->addCallback([frame](Future& /* unused */) {});
  }
  // 返回异步结果的 Future 指针
  return res;
}

// GraphExecutor 可以通过跟踪或基于语言的前端创建图
// GraphExecutor 运行它。它可以在许多不同大小的图上运行相同的图
// GraphExecutorImpl 结构体实现，继承自 GraphExecutorImplBase
struct GraphExecutorImpl : public GraphExecutorImplBase {
  // 构造函数，初始化 GraphExecutorImplBase 和 arg_spec_creator_
  GraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name)
      : GraphExecutorImplBase(graph, std::move(function_name)),
        arg_spec_creator_(*graph) {
    // 记录构造函数调用次数到日志系统中
    logging::getLogger()->addStatValue(
        logging::runtime_counters::GRAPH_EXECUTORS_CONSTRUCTED, 1.0);
  }

  // 根据栈和剩余回退深度获取执行计划
  const ExecutionPlan& getPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth) override {
    // 根据优化开关选择性地获取或编译执行计划
    return getGraphExecutorOptimize() ? getOrCompile(stack)
                                      : getOrCompileFallback();
  }

  // 获取调试状态信息
  GraphExecutorState getDebugState() override {
    GraphExecutorState state;
    state.graph = graph.get();
    if (fallback) {
      state.fallback = fallback;
    }
    for (auto& entry : plan_cache) {
      state.execution_plans.emplace(entry.first, entry.second);
    }
    return state;
  }

 protected:
  friend struct GraphExecutor;

  // 获取或编译回退执行计划
  const ExecutionPlan& getOrCompileFallback() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    if (!fallback) {
      auto graph_ = graph->copy();
      runRequiredPasses(graph_);
      fallback = ExecutionPlan(graph_, function_name_);
    }
    return fallback;
  }

  // 根据栈和参数规范获取执行计划
  const ExecutionPlan& getOrCompile(const Stack& stack) {
    // 在锁外部计算参数规范的哈希值，以减少锁的持有时间
    ArgumentSpec spec =
        arg_spec_creator_.create(autograd::GradMode::is_enabled(), stack);
    {
      std::lock_guard<std::mutex> lock(compile_mutex);
      auto it = plan_cache.find(spec);
      if (it != plan_cache.end()) {
        // 如果在计划缓存中找到了匹配的执行计划，增加命中缓存的统计量
        logging::getLogger()->addStatValue(
            logging::runtime_counters::EXECUTION_PLAN_CACHE_HIT, 1.0);
        return it->second;
      }
      // 否则，根据参数规范编译新的执行计划，并将其加入计划缓存
      auto plan = compileSpec(spec);
      auto r = plan_cache.emplace(std::move(spec), std::move(plan));
      // 增加未命中缓存的统计量
      logging::getLogger()->addStatValue(
          logging::runtime_counters::EXECUTION_PLAN_CACHE_MISS, 1.0);
      return r.first->second;
    }
  }

  // 根据参数规范编译执行计划
  ExecutionPlan compileSpec(const ArgumentSpec& spec) {
    auto opt_graph = graph->copy();
    GRAPH_DUMP("Optimizing the following function:", opt_graph);
    arg_spec_creator_.specializeTypes(*opt_graph, spec);

    // 阶段0：内联函数，清理可能阻碍优化的遗留物
    Inline(*opt_graph);
    GRAPH_DEBUG("After Inline, before LowerGradOf\n", *opt_graph);
    LowerGradOf(*opt_graph);
    GRAPH_DEBUG(
        "After LowerGradOf, before specializeAutogradZero\n", *opt_graph);
    specializeAutogradZero(opt_graph);
    GRAPH_DEBUG(
        "After specializeAutogradZero, before LowerSimpleTuples\n", *opt_graph);
    // 返回优化后的执行计划
    return ExecutionPlan(opt_graph, function_name_);
  }
    # 使用 LowerSimpleTuples 函数处理优化图，简化图中的简单元组结构
    LowerSimpleTuples(opt_graph);
    
    # 输出调试信息，显示 LowerSimpleTuples 处理后的图状态，准备进行常量池优化
    GRAPH_DEBUG(
        "After LowerSimpleTuples, before ConstantPooling\n", *opt_graph);
    
    # 对优化图进行常量池优化，合并图中相同的常量节点
    ConstantPooling(opt_graph);
    
    # 输出调试信息，显示 ConstantPooling 处理后的图状态，准备运行必要的优化 passes
    GRAPH_DEBUG(
        "After ConstantPooling, before runRequiredPasses\n", *opt_graph);

    // Phase 1. Specialize to input definedness (this is very important for
    //          gradient graphs), and run required passes to bring the graph
    //          to an executable form.
    # 第一阶段。根据输入定义性进行专门化（对于梯度图非常重要），并运行必要的 passes 将图形带入可执行形式。
    runRequiredPasses(opt_graph);

    # 输出调试信息，显示 runRequiredPasses 运行后的图状态，准备进行常量传播
    GRAPH_DEBUG(
        "After runRequiredPasses, before ConstantPropagation\n", *opt_graph);

    // Phase 2. Propagate detailed information about the spec through the
    //          graph (enabled more specializations in later passes).
    //          Shape propagation sometimes depends on certain arguments being
    //          constants, and constant propagation doesn't need shape
    //          information anyway, so it's better to run it first.
    # 第二阶段。通过图传播关于规范的详细信息（在后续 passes 中启用更多专门化）。
    # 形状传播有时依赖于某些参数是常量，而常量传播本身不需要形状信息，因此最好先运行它。
    ConstantPropagation(opt_graph);

    # 输出调试信息，显示 ConstantPropagation 处理后的图状态，准备进行输入形状传播
    GRAPH_DEBUG(
        "After ConstantPropagation, before PropagateInputShapes\n", *opt_graph);

    # 对优化图进行输入形状传播，根据输入的形状信息优化图中的节点
    PropagateInputShapes(opt_graph);

    # 输出调试信息，显示 PropagateInputShapes 处理后的图状态，准备进行梯度需求传播
    GRAPH_DEBUG(
        "After PropagateInputShapes, before PropagateRequiresGrad\n",
        *opt_graph);

    # 对优化图进行梯度需求传播，确定哪些节点需要计算梯度
    PropagateRequiresGrad(opt_graph);

    # 输出调试信息，显示 PropagateRequiresGrad 处理后的图状态，准备运行优化
    GRAPH_DEBUG(
        "After PropagateRequiresGrad, before runOptimization\n", *opt_graph);

    # 第三阶段。运行不同iable优化（即可以使用autograd执行的简单图重写）
    runOptimization(opt_graph);

    // Phase 4. If this graph will be differentiated, we need to slice out the
    //          symbolically differentiable subgraphs for further optimizations.
    // Phase 5. Apply non-differentiable optimizations to the graphs we've found
    //          (or the whole graph if we know we won't need its derivative).
    # 第四阶段。如果这个图将被微分，我们需要切出符号可微的子图进行进一步的优化。
    # 第五阶段。对找到的图应用非可微的优化（或者整个图如果我们知道不需要其导数）。
    // 检查是否需要计算梯度
    if (needsGradient(opt_graph)) {
      // 创建自动微分子图
      auto diff_nodes = CreateAutodiffSubgraphs(
          opt_graph,
          autodiff_subgraph_inlining ? autodiffSubgraphNodeThreshold : 1);
      // 输出调试信息，显示创建自动微分子图后的图结构
      GRAPH_DEBUG("After CreateAutodiffSubgraphs\n", *opt_graph);
      // 初始化索引
      size_t idx = 0;
      // 遍历不同的微分节点
      for (Node* dnode : diff_nodes) {
        // 输出调试信息，显示正在优化的微分节点
        GRAPH_DEBUG("Optimizing diff node ", idx);
        // 提取微分子图
        auto diff_graph = std::move(dnode->g(attr::Subgraph));
        // 进行微分操作，生成正向和反向传播图
        Gradient gradient = differentiate(diff_graph);
        // 输出调试信息，显示正向传播图
        GRAPH_DEBUG("Forward graph:\n", *(gradient.f));
        // 输出调试信息，显示反向传播图
        GRAPH_DEBUG("Backward graph:\n", *(gradient.df));
        // 运行后微分优化，自动微分会用新图替换一些图的部分，通常包含控制流和节点上的形状信息缺失，因此运行形状传播和可微优化来确保图被优化
        PropagateInputShapes(gradient.f);
        // 输出调试信息，显示形状传播后的图
        GRAPH_DEBUG("After PropagateInputShapes\n", *(gradient.f));
        // 运行优化器，对正向传播图进行优化
        runOptimization(gradient.f);
        // 运行非微分优化，对正向传播图进行优化
        runNondiffOptimization(gradient.f);
        // 打包梯度信息到微分节点中
        packGradient(gradient, dnode);
        // 输出调试信息，显示完成优化的微分节点
        GRAPH_DEBUG("Finished optimizing diff node ", idx++);
      }
      // 内联自动微分子图
      InlineAutodiffSubgraphs(
          opt_graph,
          autodiff_subgraph_inlining ? autodiffSubgraphInlineThreshold : 1);
      // 输出调试信息，显示内联自动微分子图后的图结构
      GRAPH_DEBUG("After InlineAutodiffSubgraphs\n", *opt_graph);
    } else {
      // 运行非微分优化，对整个图进行优化
      runNondiffOptimization(opt_graph);
    }
    // 清除图中的死代码
    EliminateDeadCode(opt_graph);
    // 输出调试信息，显示编译后的优化结果
    GRAPH_DUMP("After compileSpec optimizations:", opt_graph);
    // 返回执行计划
    return ExecutionPlan(opt_graph, function_name_);
  }

  ~GraphExecutorImpl() override = default;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // 参数规范创建器
  ArgumentSpecCreator arg_spec_creator_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // 当 optimize 为 false 时填充，图的编译版本
  ExecutionPlan fallback;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // 参数配置到优化后图版本的映射
  std::unordered_map<ArgumentSpec, ExecutionPlan> plan_cache;
};

// GraphExecutor 类的构造函数，接受一个图的共享指针和函数名作为参数
GraphExecutor::GraphExecutor(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : pImpl(
          // 根据条件选择合适的图执行器实现类作为私有实现对象
          IsNewExecutorEnabled()
              ? (getProfilingMode() ?
                    // 如果启用了性能分析模式，则选择 ProfilingGraphExecutorImpl
                    dynamic_cast<GraphExecutorImplBase*>(
                        new ProfilingGraphExecutorImpl(
                            graph,
                            std::move(function_name)))
                    // 否则选择 SimpleGraphExecutorImpl
                    : dynamic_cast<GraphExecutorImplBase*>(
                        new SimpleGraphExecutorImpl(
                            graph,
                            std::move(function_name))))
              // 如果未启用新执行器，则选择 GraphExecutorImpl
              : dynamic_cast<GraphExecutorImplBase*>(
                    new GraphExecutorImpl(graph, std::move(function_name)))) {}

// GraphExecutor 类的构造函数，接受一个图的共享指针、函数名和执行模式作为参数
GraphExecutor::GraphExecutor(
    const std::shared_ptr<Graph>& graph,
    std::string function_name,
    ExecutorExecutionMode executor_mode)
    : pImpl(
          // 根据执行模式选择合适的图执行器实现类作为私有实现对象
          executor_mode == ExecutorExecutionMode::SIMPLE
              ? dynamic_cast<GraphExecutorImplBase*>(
                    new SimpleGraphExecutorImpl(
                        graph,
                        std::move(function_name)))
              : dynamic_cast<GraphExecutorImplBase*>(
                    new ProfilingGraphExecutorImpl(
                        graph,
                        std::move(function_name)))) {}

// 执行图的方法，委托给私有实现对象的相应方法
void GraphExecutor::run(Stack& inputs) {
  return pImpl->run(inputs);
}

// 异步执行图的方法，委托给私有实现对象的相应方法
c10::intrusive_ptr<Future> GraphExecutor::runAsync(
    Stack& stack,
    TaskLauncher taskLauncher) {
  return pImpl->runAsync(stack, std::move(taskLauncher));
}

// 获取特定输入下的执行计划，委托给私有实现对象的相应方法
const ExecutionPlan& GraphExecutor::getPlanFor(
    Stack& inputs,
    std::optional<size_t> remaining_bailout_depth) {
  return pImpl->getPlanFor(inputs, remaining_bailout_depth);
}

// 获取调试状态，委托给私有实现对象的相应方法
GraphExecutorState GraphExecutor::getDebugState() {
  return pImpl->getDebugState();
}

// 调试时刷新编译缓存，仅对 ProfilingGraphExecutorImpl 有效
void GraphExecutor::debugFlushCompilationCache() {
  if (auto ppImpl =
          std::dynamic_pointer_cast<ProfilingGraphExecutorImpl>(pImpl)) {
    ppImpl->debugFlushCompilationCache();
  } else {
    // 对于旧版本执行器，抛出断言错误
    TORCH_INTERNAL_ASSERT(false, "Not Implemented for Legacy Executor");
  }
}

// 检查执行器是否已优化，委托给私有实现对象的相应方法
bool GraphExecutor::isOptimized() const {
  return pImpl && pImpl->isOptimized();
}

// 检查是否启用了新的执行器
TORCH_API bool IsNewExecutorEnabled() {
  static const auto disable_new_executor =
      std::getenv("TORCH_JIT_DISABLE_NEW_EXECUTOR");
  return getExecutorMode() && FLAGS_torch_jit_enable_new_executor &&
      !disable_new_executor;
}

// 运行必要的图传递，清除不稳定的扩展节点并进行死代码消除等优化
void runRequiredPasses(const std::shared_ptr<Graph>& g) {
  // 删除隐式插入的不稳定扩展节点
  RemoveExpands(g);
  // 规范化操作节点
  CanonicalizeOps(g);
  // 消除死代码
  EliminateDeadCode(g);
}
void packGradient(const Gradient& gradient, Node* dnode) {
    // 确保节点类型为不同iableGraph
    AT_ASSERT(dnode->kind() == prim::DifferentiableGraph);
    // 设置梯度信息到节点的子图属性
    dnode->g_(attr::Subgraph, gradient.f)
        ->g_(attr::ReverseSubgraph, gradient.df)
        ->i_(attr::f_real_outputs, gradient.f_real_outputs)
        ->is_(attr::df_input_vjps, fmap<int64_t>(gradient.df_input_vjps))
        ->is_(
            attr::df_input_captured_inputs,
            fmap<int64_t>(gradient.df_input_captured_inputs))
        ->is_(
            attr::df_input_captured_outputs,
            fmap<int64_t>(gradient.df_input_captured_outputs))
        ->is_(attr::df_output_vjps, fmap<int64_t>(gradient.df_output_vjps));
}

static bool mayIntroduceGradient(const Block* b) {
    // 检查块中是否可能引入梯度
    for (const Node* n : b->nodes()) {
        if (n->kind() == prim::PythonOp)
            return true;
        for (const Block* bb : n->blocks()) {
            if (mayIntroduceGradient(bb))
                return true;
        }
    }
    return false;
}

bool needsGradient(const std::shared_ptr<const Graph>& graph) {
    // 检查图是否需要计算梯度
    if (!autograd::GradMode::is_enabled()) {
        return false;
    }

    if (mayIntroduceGradient(graph->block())) {
        return true;
    }

    for (const Value* input : graph->inputs()) {
        if (input->type()->requires_grad()) {
            return true;
        }
    }

    return false;
}

void runNondiffOptimization(
    std::shared_ptr<Graph>& graph,
    bool strict_fuser_check) {
    GRAPH_DEBUG(
        "Before customPrePasses (beginning of runNondiffOptimization)\n", *graph);
    // 运行不同的后处理优化
    for (const auto& passPair : getCustomPrePasses()) {
        passPair.first(graph);
    }
    GRAPH_DEBUG("After customPrePasses\n", *graph);

    // 分解特定的操作
    DecomposeOps(graph);
    GRAPH_DEBUG("After DecomposeOps\n", *graph);

    // 移除TupleConstruct / TupleUnpack以进行融合
    LowerSimpleTuples(graph);
    GRAPH_DEBUG("After LowerSimpleTuples, before BatchMM\n", *graph);

    // 重写包含多个矩阵乘的子图
    BatchMM(graph);

    GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);
    if (getExecutorMode()) {
        if (tensorExprFuserEnabled()) {
            auto min_size = getFusionGroupInlining() ? 2 : 1;
            auto dyn_shapes = tensorExprDynamicShapeFusionEnabled();
            // 融合张量表达式
            FuseTensorExprs(graph, min_size, /*composed_op*/ false, dyn_shapes);
        }
    } else {
        // 默认融合图操作
        FuseGraph(graph, strict_fuser_check);
    }
    GRAPH_DEBUG("After Fusion\n", *graph);

    // 运行自定义的融合后优化
    for (const auto& passPair : getCustomPostPasses()) {
        passPair.first(graph);
    }
    GRAPH_DEBUG(
        "After customPostPasses (end of runNondiffOptimization)\n", *graph);
}

void runOptimization(
    std::shared_ptr<Graph>& graph,
    bool unroll_non_constant_loops,
    ...
  // 执行基本的图形预处理，消除噪声。
  GRAPH_DEBUG(
      "Before EliminateDeadCode (beginning of runOptimization)\n", *graph);
  // 调用消除死代码的优化函数，清除无用的计算节点。
  EliminateDeadCode(graph);

  GRAPH_DEBUG(
      "After EliminateDeadCode, before EliminateCommonSubexpression\n", *graph);
  // 调用消除公共子表达式的优化函数，在图中寻找和消除重复计算的表达式。
  EliminateCommonSubexpression(graph);

  GRAPH_DEBUG(
      "After EliminateCommonSubexpression , before PeepholeOptimize\n", *graph);
  // 调用Peephole优化，对图中的节点进行局部优化。
  PeepholeOptimize(graph);

  GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
  // 如果设置了常量传播用户类，则执行常量传播优化。
  // 否则，以保留用户类的形式执行常量传播。
  if (const_prop_user_classes) {
    ConstantPropagation(graph);
  } else {
    ConstantPropagation(graph, true);
  }

  GRAPH_DEBUG("After ConstantPropagation, before ConstantPooling\n", *graph);
  // 执行常量池优化，合并和复用常量。
  ConstantPooling(graph);

  GRAPH_DEBUG("After ConstantPooling\n", *graph);

  // 展开小循环，并消除每次迭代中相同的表达式。
  bool unroll_success = false;
  if (unroll_non_constant_loops) {
    // 尝试展开非常量循环，并在此之后执行列表变异的移除。
    unroll_success = UnrollLoops(graph);
    GRAPH_DEBUG("After UnrollLoops, before RemoveListMutation\n", *graph);
  } else {
    // 尝试展开常量循环，并在此之后执行列表变异的移除。
    unroll_success = UnrollConstantLoops(graph);
    GRAPH_DEBUG(
        "After UnrollConstantLoops, before RemoveListMutation\n", *graph);
  }

  if (unroll_success) {
    // 如果成功展开了循环，则再次执行Peephole优化、常量传播。
    RemoveListMutation(graph);
    GRAPH_DEBUG("After RemoveListMutation, before PeepholeOptimize\n", *graph);
    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG("After ConstantPropagation\n", *graph);
  }

  // 再次执行消除公共子表达式的优化，以进一步优化图。
  EliminateCommonSubexpression(graph);
  GRAPH_DEBUG(
      "After EliminateCommonSubexpression, before CheckInplace\n", *graph);
  // 执行检查就地操作的优化，确保就地操作的正确性。
  CheckInplace(graph);

  GRAPH_DEBUG("After CheckInplace (end of runOptimization)\n", *graph);
} // 结束 torch::jit 命名空间

Node* replaceBlockWithFallbackGraph(Block* b, ArrayRef<Value*> inputs) {
  // 创建一个新的计算图对象
  auto graph = std::make_shared<Graph>();

  // 如果块 b 的拥有节点不为空，说明块位于 If 或 prim::Loop 中，需要复制块内部
  // 否则，复制整个图形，需要区分这两种情况，因为 cloneFrom 方法在复制图形的块时会自动添加输入，否则需要来自用户的输入
  if (b->owningNode() != nullptr) {
    // 创建输入映射表
    std::unordered_map<Value*, Value*> input_mapping;
    auto value_map = [&input_mapping](Value* v) { return input_mapping[v]; };
    // 添加输入到新图形的块中
    for (auto inp : inputs) {
      input_mapping[inp] = graph->block()->addInput();
    }
    // 从块 b 复制到新图形的块中，并使用输入映射表
    graph->block()->cloneFrom(b, value_map);
  } else {
    // 直接从块 b 复制到新图形的块中，不需要输入映射
    auto value_map = [](Value* v) { return v; };
    graph->block()->cloneFrom(b, value_map);
  }

  // 创建一个回退图节点，并将新图形设置为其子图
  auto fallback = b->owningGraph()->create(
      prim::FallbackGraph, inputs, b->outputs().size());
  fallback->g_(attr::Subgraph, graph);
  // 将回退图节点添加到块 b 的开头
  b->prependNode(fallback);

  // 设置新图形的输入的类型和元数据
  for (const auto i : c10::irange(inputs.size())) {
    graph->inputs()[i]->setType(inputs[i]->type());
    graph->inputs()[i]->copyMetadata(inputs[i]);
  }

  // 设置回退图节点的输出的类型和元数据，并替换块 b 的输出
  for (const auto i : c10::irange(b->outputs().size())) {
    fallback->output(i)->setType(b->outputs()[i]->type());
    fallback->output(i)->copyMetadata(b->outputs()[i]);
    b->replaceOutput(i, fallback->output(i));
  }

  // 移除图形块中的性能分析节点
  ProfilingRecord::removeProfilingNodes(graph->block());

  // 逆向遍历块 b 的节点并移除，直到回退图节点之前
  for (auto it = b->nodes().rbegin(); it != fallback->iterator(); it++) {
    it.destroyCurrent();
  }

  // 返回创建的回退图节点
  return fallback;
}

} // 结束 torch::jit 命名空间
```