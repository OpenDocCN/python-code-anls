# `.\pytorch\torch\csrc\jit\passes\decompose_ops.cpp`

```
#include <torch/csrc/jit/passes/decompose_ops.h>

#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/core/symbol.h>

namespace torch {
namespace jit {

namespace {
// 返回来自模式的别名分析类型
c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}
} // namespace

// 帮助函数，用于确定传入的可选张量参数/值是否静态定义
// 返回是、否或者无法确定（如果无法确定）
static std::optional<bool> isDefined(Value* tensor) {
  if (tensor->type()->isSubtypeOf(*TensorType::get())) {
    return true;
  }
  if (tensor->node()->mustBeNone()) {
    return false;
  }
  return {};
}

// 检查是否可以分解归一化操作
static bool isDecomposableNorm(Node* normalize_op) {
  // 支持分解的归一化操作集合
  static const OperatorSet decomposable_normalization_ops = {
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, bool cudnn_enable) -> Tensor",
  };
  // 获取输入值
  Value* input = normalize_op->namedInput(attr::input);
  // 如果输入不是张量类型，则无法分解
  if (!input->type()->isSubtypeOf(*TensorType::get())) {
    return false;
  }
  // 获取输入张量的设备信息
  auto device = input->type()->expectRef<TensorType>().device();
  // 目前仅在 GPU 设备上支持批归一化和层归一化的分解
  if (!device || !(*device).is_cuda()) {
    return false;
  }

  // 如果归一化操作属于可分解的归一化操作集合
  if (normalize_op->isMemberOf(decomposable_normalization_ops)) {
    // 如果无法静态确定权重和偏置是否已定义，则无法有效地分解归一化操作
    return isDefined(normalize_op->namedInput(attr::weight)).has_value() &&
        isDefined(normalize_op->namedInput(attr::bias)).has_value();
  }
  return false;
}

// 注册操作符
RegisterOperators reg_ops(
    {Operator(
         "aten::_ncf_unsqueeze(Tensor(a) self, int ndim) -> Tensor(a)",
         [](Stack& stack) {
           // 弹出堆栈中的整数参数 ndim
           const int64_t ndim = pop(stack).toInt();
           // 弹出堆栈中的张量参数 self
           auto self = pop(stack).toTensor();
           // 创建一个包含 ndim 个元素的 SmallVector，每个元素初始化为 1
           c10::SmallVector<int64_t, 8> sizes(ndim, 1);
           // 断言张量 self 的维度为 1
           AT_ASSERT(self.dim() == 1);
           // 将张量 self 重塑为指定大小的张量，并将结果推入堆栈
           sizes.at(1) = self.size(0);
           push(stack, self.reshape(sizes));
         },
         aliasAnalysisFromSchema()),

     Operator(
         "aten::_ncf_view(Tensor(a) self, int[] input_shape, int normalized_ndim) -> Tensor(a)",
         [](Stack& stack) {
           // 弹出堆栈中的整数参数 normalized_ndim
           const int64_t normalized_ndim = pop(stack).toInt();
           // 弹出堆栈中的整数列表参数 input_shape
           auto input_shape = pop(stack).toIntList();
           // 弹出堆栈中的张量参数 self
           auto self = pop(stack).toTensor();
           // 计算输入形状列表的长度
           const int64_t input_ndim = input_shape.size();
           // 创建一个包含 input_ndim 个元素的 SmallVector，每个元素初始化为 1
           c10::SmallVector<int64_t, 8> sizes(input_ndim, 1);
           // 根据输入形状列表和 normalized_ndim 更新 sizes 向量
           for (int i = 0; i < input_ndim - normalized_ndim; ++i) {
             sizes.at(i) = input_shape.get(i);
           }
           // 将张量 self 重塑为指定大小的张量，并将结果推入堆栈
           push(stack, self.reshape(sizes));
         },
         aliasAnalysisFromSchema())});
// 对给定的代码块进行操作分解，返回是否有进行了分解的标志
static bool DecomposeOps(Block* block, CompilationUnit& decompose_funcs) {
  bool decomposed = false; // 初始化分解标志为false
  // 遍历当前块中的所有节点
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    // 对当前节点中的每个子块进行递归操作分解
    for (auto sub : it->blocks()) {
      DecomposeOps(sub, decompose_funcs);
    }

    // 检查当前节点是否匹配指定的aten::addmm操作
    if (it->matches(
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor",
            /*const_inputs=*/{attr::beta, attr::alpha})) {
      // 对于alpha和beta都为1.0的addmm情况，进行分解为mm和add操作
      if (it->get<at::Scalar>(attr::alpha)->toComplexDouble() != 1.0 ||
          it->get<at::Scalar>(attr::beta)->toComplexDouble() != 1.0) {
        continue; // 如果alpha和beta不都为1.0，继续下一个节点
      }

      decomposed = true; // 标记已进行分解
      WithInsertPoint guard(*it); // 设置插入点保护当前节点
      // 获取addmm函数对应的图形表示
      std::shared_ptr<Graph> d_graph =
          toGraphFunction(decompose_funcs.get_function("addmm")).graph();
      // 插入分解后图形的计算，并取得新的输出值
      Value* new_output =
          insertGraph(*it->owningGraph(), *d_graph, it->inputs()).at(0);
      // 设置分解后图形的输出类型与原始操作的输出类型一致，以保持规范化图形的正确性
      new_output->setType(it->output()->type());
      it->output()->replaceAllUsesWith(new_output); // 替换当前节点的所有使用为新的输出
      it.destroyCurrent(); // 删除当前节点
    } else if (
        it->matches(
            "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor")) {
      // 检查是否可以对批归一化操作进行分解
      if (!isDecomposableNorm(*it)) {
        continue;
      }
      // 设置标志表示已经分解过
      decomposed = true;
      // 设置插入点为当前操作
      WithInsertPoint insert_guard{*it};
      // 获取当前操作所属的图
      Graph* graph = it->owningGraph();
      // 获取输入值
      Value* input = it->namedInput(attr::input);
      // 计算输入张量的维度
      Value* input_dim = graph->insert(aten::dim, {input});
      // 构建输入向量
      std::vector<Value*> inputs{
          input,
          it->namedInput(attr::running_mean),
          it->namedInput(attr::running_var),
          it->namedInput(attr::training),
          it->namedInput(attr::momentum),
          it->namedInput(attr::eps)};

      // 内联编译分解后的批归一化操作
      std::shared_ptr<Graph> d_graph =
          toGraphFunction(decompose_funcs.get_function("batch_norm")).graph();
      // 插入新图并获取输出值
      Value* new_output = insertGraph(*graph, *d_graph, inputs).at(0);

      // 后处理图
      Value* weight = it->namedInput(attr::weight);
      Value* bias = it->namedInput(attr::bias);
      // 如果权重已定义，进行加权乘法操作
      if (isDefined(weight).value()) {
        Value* expanded_weight =
            graph->insert(aten::_ncf_unsqueeze, {weight, input_dim});
        new_output = graph->insert(aten::mul, {new_output, expanded_weight});
      }
      // 如果偏置已定义，进行加法操作
      if (isDefined(bias).value()) {
        Value* expanded_bias =
            graph->insert(aten::_ncf_unsqueeze, {bias, input_dim});
        new_output = graph->insert(aten::add, {new_output, expanded_bias});
      }
      // 替换当前操作的输出值并销毁当前操作
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    } else if (
        it->matches(
            "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, bool cudnn_enable) -> Tensor")) {
      // 检查是否可以对层归一化操作进行分解
      if (!isDecomposableNorm(*it)) {
        continue;
      }
      // 设置标志表示已经分解过
      decomposed = true;
      // 设置插入点为当前操作
      WithInsertPoint insert_guard{*it};
      // 获取当前操作所属的图
      Graph* graph = it->owningGraph();
      // 构建输入向量
      std::vector<Value*> inputs{
          it->namedInput(attr::input),
          it->namedInput(attr::normalized_shape),
          it->namedInput(attr::eps),
          it->namedInput(attr::cudnn_enable)};

      // 内联编译分解后的层归一化操作
      std::shared_ptr<Graph> d_graph =
          toGraphFunction(decompose_funcs.get_function("layer_norm")).graph();
      // 插入新图并获取输出值
      Value* new_output = insertGraph(*graph, *d_graph, inputs).at(0);

      // 后处理图
      Value* weight = it->namedInput(attr::weight);
      Value* bias = it->namedInput(attr::bias);
      // 如果权重已定义，进行加权乘法操作
      if (isDefined(weight).value()) {
        new_output = graph->insert(aten::mul, {new_output, weight});
      }
      // 如果偏置已定义，进行加法操作
      if (isDefined(bias).value()) {
        new_output = graph->insert(aten::add, {new_output, bias});
      }
      // 替换当前操作的输出值并销毁当前操作
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    }
  }
  // 返回标志，表示是否进行了任何分解操作
  return decomposed;
}

// DecomposeOps 函数：对图进行操作分解
void DecomposeOps(std::shared_ptr<Graph>& graph) {
  // 静态编译单元，包含了一些用于操作分解的函数定义
  static CompilationUnit decompose_funcs(R"SCRIPT(
      // addmm 函数：执行矩阵相乘并加法运算
      def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: number = 1.0, alpha: number = 1.0):
          return self + mat1.mm(mat2)

      // batch_norm 函数：执行批量归一化操作
      def batch_norm(input : Tensor, running_mean : Optional[Tensor], running_var : Optional[Tensor], training : bool, momentum : float, eps : float) -> Tensor:
          if training:
              // 计算归一化均值和方差
              norm_mean, norm_var = torch.batch_norm_update_stats(input, running_mean, running_var, momentum)
          else:
              // 使用给定的运行时均值和方差
              norm_mean = torch._unwrap_optional(running_mean)
              norm_var = torch._unwrap_optional(running_var)
          // 将均值和方差扩展为与输入数据相同维度
          norm_mean = torch._ncf_unsqueeze(norm_mean, input.dim())
          norm_var = torch._ncf_unsqueeze(norm_var, input.dim())
          // 计算归一化的标准差的倒数
          norm_invstd = 1 / (torch.sqrt(norm_var + eps))
          return ((input - norm_mean) * norm_invstd)

      // layer_norm 函数：执行层归一化操作
      def layer_norm(input : Tensor, normalized_shape : List[int], eps : float, cudnn_enable : bool) -> Tensor:
          // 计算输入数据的维度和归一化维度
          input_ndim = input.dim()
          normalized_ndim = len(normalized_shape)
          n = 1
          // 计算未归一化维度的乘积
          for i in range(input_ndim - normalized_ndim):
              n *= input.size(i)
          // 将输入数据重塑为需要的形状
          input_reshape = input.contiguous().view(1, n, -1)
          // 计算均值和倒数标准差
          mean, invstd = torch.batch_norm_stats(input_reshape, eps)
          input_shape = input.size()
          // 调整均值和倒数标准差的形状以匹配输入数据
          mean = torch._ncf_view(mean, input_shape, normalized_ndim)
          invstd = torch._ncf_view(invstd, input_shape, normalized_ndim)

          return (input - mean) * invstd
      )SCRIPT");

  // 对图进行操作分解，返回是否成功
  bool is_decomposed = DecomposeOps(graph->block(), decompose_funcs);

  // 如果图成功分解，则重新运行以下传递
  if (is_decomposed) {
    // 传播输入形状
    PropagateInputShapes(graph);
    // 常量传播
    ConstantPropagation(graph);
    // 消除死代码
    EliminateDeadCode(graph);
  }
}

} // namespace jit
} // namespace torch
```