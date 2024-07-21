# `.\pytorch\torch\csrc\jit\passes\frozen_ops_to_mkldnn.cpp`

```
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 ATen 库的配置信息
#include <ATen/Config.h>
// 包含 ATen 库的原生函数
#include <ATen/NativeFunctions.h>
// 包含 ATen 库的实用工具
#include <ATen/Utils.h>
// 包含 ATen 核心的符号定义
#include <ATen/core/symbol.h>
// 包含 ATen 库的原生层归一化功能
#include <ATen/native/layer_norm.h>
// 包含 c10 库的标量类型定义
#include <c10/core/ScalarType.h>
// 包含 c10 库的异常处理
#include <c10/util/Exception.h>
// 包含 c10 库的范围迭代器
#include <c10/util/irange.h>

// 包含 Torch JIT 的别名分析头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
// 包含 Torch JIT 的常量定义头文件
#include <torch/csrc/jit/ir/constants.h>
// 包含 Torch JIT 的 IR 定义头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 的日志功能头文件
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch JIT 的公共子表达式消除头文件
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
// 包含 Torch JIT 的常量传播头文件
#include <torch/csrc/jit/passes/constant_propagation.h>
// 包含 Torch JIT 的死代码消除头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 包含 Torch JIT 的卷积融合头文件
#include <torch/csrc/jit/passes/fold_conv_bn.h>
// 包含 Torch JIT 的冻结卷积折叠头文件
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
// 包含 Torch JIT 的冻结操作转换到 MKLDNN 头文件
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
// 包含 Torch JIT 的图重写辅助功能头文件
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
// 包含 Torch JIT 的窥孔优化头文件
#include <torch/csrc/jit/passes/peephole.h>
// 包含 Torch JIT 的移除突变头文件
#include <torch/csrc/jit/passes/remove_mutation.h>
// 包含 Torch JIT 的子图工具头文件
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
// 包含 Torch JIT 的自定义运算符运行时头文件
#include <torch/csrc/jit/runtime/custom_operator.h>
// 包含 Torch JIT 的运算符选项头文件
#include <torch/csrc/jit/runtime/operator_options.h>
// 包含 Torch JIT 的张量表达类型头文件
#include <torch/csrc/jit/tensorexpr/types.h>

// 禁用 clang 格式化，避免导入循环
// clang-format off
// 包含 ATen 原生卷积工具，避免循环依赖
#include <ATen/native/ConvUtils.h>
// clang-format on

// 包含标准库的算法
#include <algorithm>
// 包含内存管理的智能指针
#include <memory>
// 包含 ATen 核心的堆栈管理
#include <ATen/core/stack.h>
// 包含 c10 核心的布局定义
#include <c10/core/Layout.h>
// 包含 c10 核心的字符串工具
#include <c10/util/StringUtil.h>

// 如果支持 MKLDNN，则包含相应的头文件
#if AT_MKLDNN_ENABLED()
#include <ATen/CPUFunctions.h>
#include <dnnl_types.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ideep.hpp>
#endif

namespace torch {
namespace jit {

// 如果支持 MKLDNN，则使用 at::Tensor 别名定义为 Tensor
#if AT_MKLDNN_ENABLED()
using Tensor = at::Tensor;

// 匿名命名空间，用于局部定义
namespace {

// 从模式中获取别名分析种类
c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return AliasAnalysisKind::FROM_SCHEMA;
}

// 定义值集合类型为无序集合的指针
using ValueSet = std::unordered_set<Value*>;
using ValueSetPtr = std::shared_ptr<std::unordered_set<Value*>>;

// 获取值的最后使用节点
Node* getLastUse(Value* v) {
  auto last_use_node = v->node();
  for (const auto& use : v->uses()) {
    if (use.user->isAfter(last_use_node)) {
      last_use_node = use.user;
    }
  }
  return last_use_node;
}

// 合并集合
void merge_sets(
    std::unordered_map<Value*, ValueSetPtr>& alias_mapping,
    Value* existing,
    Value* new_v) {
  if (alias_mapping[existing] == alias_mapping[new_v]) {
    return;
  }
  auto existing_set = alias_mapping[existing];
  auto set_to_remove = alias_mapping[new_v];
  for (auto it = set_to_remove->begin(); it != set_to_remove->end(); it++) {
    existing_set->insert(*it);
    alias_mapping[*it] = existing_set;
  }
}

// 断言非张量类型不包含张量
void assertNonTensorTypeDoesNotContainTensors(TypePtr type) {
  if (type->cast<TensorType>()) {
    return;
  }
  for (const auto& t : type->containedTypes()) {
    TORCH_INTERNAL_ASSERT(!t->cast<TensorType>());
  }
}
  auto add_to_inplace_set = [&](Node* node) {
    // 将节点添加到待就地替换集合中，延迟替换操作以避免旧节点输出值失效
    nodes_to_inplace.push_back(node);
    // 断言节点的输出值只有一个
    TORCH_INTERNAL_ASSERT(node->outputs().size() == 1);
    // 获取输出值的生存周期结束节点
    auto output_liveness_end =
        set_liveness[alias_mapping[node->outputs().at(0)]];
    // 合并输入和输出的别名集合
    merge_sets(alias_mapping, node->inputs().at(0), node->output());



    // 节点成功就地替换后，延长输入和输出别名集合的生存周期
    for (Value* output : node->outputs()) {
      auto output_set = alias_mapping[output];
      auto input_set = alias_mapping[node->inputs().at(0)];
      extend_liveness(alias_mapping, output_set, input_set, output_liveness_end);
    }
  };

  for (Node* n : graph->nodes()) {
    // 对于每个节点，如果其具有等效的就地替换节点，并且其输入的别名集合在该节点后已死亡，
    // 则执行就地替换
    for (Value* output : n->outputs()) {
      auto output_set = alias_mapping[output];
      if (output_set->size() == 1 && n->has_inplace_version()) {
        Node* last = set_liveness[output_set];
        if (last && last->isAfter(n)) {
          add_to_inplace_set(n);
          break;
        }
      }
    }
  }
}



// 执行就地替换的辅助函数，用于合并输入和输出的别名集合
void merge_sets(std::unordered_map<Value*, ValueSetPtr>& alias_mapping,
                Value* from,
                Value* to) {
  // 合并输入和输出的别名集合
  auto& from_set = alias_mapping[from];
  auto& to_set = alias_mapping[to];
  from_set->insert(to_set->begin(), to_set->end());
  to_set = from_set;
}



// 延长别名集合的生存周期，以适应成功的就地替换操作
void extend_liveness(std::unordered_map<Value*, ValueSetPtr>& alias_mapping,
                     ValueSetPtr input_set,
                     ValueSetPtr output_set,
                     Node* liveness_end) {
  // 如果生存周期结束节点存在，则将其扩展到输入和输出别名集合
  if (liveness_end) {
    auto& input_liveness = set_liveness[input_set];
    if (!input_liveness || liveness_end->isAfter(input_liveness)) {
      input_liveness = liveness_end;
    }
    auto& output_liveness = set_liveness[output_set];
    if (!output_liveness || liveness_end->isAfter(output_liveness)) {
      output_liveness = liveness_end;
    }
  }
}
    // 将节点的输出映射到其别名，更新其生存期结束点
    set_liveness[alias_mapping[node->output()]] = output_liveness_end;
  };

  // 遍历计算图中的每个节点
  for (Node* node : graph->nodes()) {
    auto k = node->kind();
    // 检查节点类型，确定是否可进行原位操作
    if (k == aten::relu || k == aten::sigmoid || k == aten::dropout ||
        k == prim::MKLDNNHardSwish || k == prim::MKLDNNHardSigmoid ||
        k == prim::MKLDNNHardTanh || k == aten::tanh ||
        k == prim::MKLDNNClamp || k == Symbol::prim("MKLDNNScalarMul") ||
        k == Symbol::prim("MKLDNNLayerNorm")) {
      // 如果节点的第一个输入的生存期结束点在当前节点之前，则跳过
      if (set_liveness[alias_mapping[node->inputs().at(0)]]->isAfter(node)) {
        continue;
      }
      // 将节点添加到可以原位操作的集合中
      add_to_inplace_set(node);
    } else if (k == aten::mul || k == aten::add) {
      // 对于二元运算符（加法/乘法），由于其可交换且仅接受张量输入，因此可以选择原位操作第一个或第二个输入
      int64_t reusable_value_index = -1;
      for (const auto i : c10::irange(2)) {
        // 断言输入是张量类型
        TORCH_INTERNAL_ASSERT(node->inputs().at(i)->type()->cast<TensorType>());
        // 如果输入的生存期结束点不在当前节点之后，则可以重复使用该输入
        if (!set_liveness[alias_mapping[node->inputs().at(i)]]->isAfter(node)) {
          reusable_value_index = i;
          break;
        }
      }

      // 如果找不到可重复使用的输入，则跳过当前节点
      if (reusable_value_index == -1) {
        continue;
      }

      // 如果可重复使用的输入是第二个，则将其移动到第一个位置并移除第三个输入
      if (reusable_value_index == 1) {
        node->insertInput(0, node->inputs().at(1));
        node->removeInput(2);
      }
      // 将节点添加到可以原位操作的集合中
      add_to_inplace_set(node);
    }
  }

  // 替换所有需要原位操作的节点为新的符号，并销毁原节点
  for (Node* node : nodes_to_inplace) {
    node->replaceWithNewSymbol(
        Symbol::fromQualString(node->schema().name() + "_"));
    node->destroy();
  }
// 这是一个工厂函数，创建一个操作，该操作将MKLDNN张量解包成我们可以在aten操作中运行的1D连续张量。
// 使用这个函数的先决条件是aten操作在零输入时应该是一个恒等操作。换句话说，应该满足：`aten_op(0) = 0`。
// 这个先决条件与MKLDNN使用的阻塞格式有关，它将通道维度分割为8/16的块，使其成为最内层维度。
// 每当通道维度不能被8/16整除时，最内层维度就会用0填充。
// 条件`aten_op(0) == 0`允许我们避免对填充元素进行任何特殊处理。

Operation createUnaryOp(
    std::function<void(at::Tensor output, at::Tensor input)> aten_op,
    bool inplace = false) {
  return [aten_op, inplace](Stack& stack) {
    // 从堆栈中弹出张量 `a`
    auto a = pop(stack).toTensor();
    // 临时排除分派键以确保在此操作期间没有自动微分
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);

    // 将 `a` 强制转换为 `ideep::tensor`，以便获取其描述符
    // 然后使用该描述符设置具有与 `a` 相同属性的 `out` 张量
    auto a_it = at::native::itensor_from_mkldnn(a);
    auto mkldnn_raw_data = a_it.get_data_handle();
    auto a_options_with_strided = a.options().layout(c10::kStrided);

    // 张量的物理大小可能大于逻辑大小
    // `a_it.get_desc().get_size()` 返回实际的物理大小（字节）
    // 我们使用它来计算 `aten` 操作的 `nelem`
    auto nelem = static_cast<int64_t>(
        a_it.get_desc().get_size() / elementSize(a.scalar_type()));

    // 将 `a` 的存储包装成一个aten张量
    auto in_aten =
        at::from_blob(mkldnn_raw_data, {nelem}, a_options_with_strided);

    auto out_raw_data = mkldnn_raw_data;
    auto out = a;
    if (!inplace) {
      // `a_it.get_desc()` 将分配一个具有正确物理大小的张量
      auto it_empty = ideep::tensor(a_it.get_desc());
      TORCH_INTERNAL_ASSERT(it_empty.get_desc() == a_it.get_desc());

      // 创建一个新的MKLDNN张量并转换为aten张量
      out = at::native::new_with_itensor_mkldnn(
          std::move(it_empty),
          c10::optTypeMetaToScalarType(a.options().dtype_opt()),
          a.options().device_opt());

      out_raw_data = at::native::itensor_from_mkldnn(out).get_data_handle();
    }

    // 检查确保 `a_it.get_desc().get_size()` 能整除 `elementSize(a.scalar_type())`
    TORCH_INTERNAL_ASSERT(
        a_it.get_desc().get_size() % elementSize(a.scalar_type()) == 0);

    // 从原始数据创建aten张量 `out_aten`
    auto out_aten = at::from_blob(
        out_raw_data, {static_cast<int64_t>(nelem)}, a_options_with_strided);

    // 执行aten操作 `aten_op`，将结果存入 `out`
    aten_op(out_aten, in_aten);

    // 将 `out` 推入堆栈
    push(stack, out);
  };
}
void MKLDNNLayerNormOp(Stack& stack, bool inplace) {
  // 进入排除自动求导的分发键保护范围
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);

  // 从堆栈中弹出并丢弃 enable_cudnn 参数
  pop(stack);
  // 弹出并获取 eps 参数
  auto eps = pop(stack).toDouble();

  // 初始化 bias 和 weight 张量
  Tensor bias{};
  Tensor weight{};
  // 从堆栈中弹出 bias 张量
  auto bias_ival = pop(stack);
  // 断言 bias_ival 是张量类型
  TORCH_INTERNAL_ASSERT(bias_ival.isTensor());
  bias = bias_ival.toTensor();

  // 从堆栈中弹出 weight 张量
  auto weight_ival = pop(stack);
  // 断言 weight_ival 是张量类型
  TORCH_INTERNAL_ASSERT(weight_ival.isTensor());
  weight = weight_ival.toTensor();

  // 从堆栈中弹出并获取 shape 参数作为维度向量
  auto shape = pop(stack).toDimVector();
  // 从堆栈中弹出 input 张量
  auto input = pop(stack).toTensor();

  // 调用 MKLDNN 版本的 layer normalization 函数，返回结果 dst、mean、rstd
  auto [dst, mean, rstd] =
      at::native::mkldnn_layer_norm_last_index_weight_bias_f32(
          input, shape, weight, bias, eps, inplace);
  // 将 dst 压入堆栈作为输出
  push(stack, dst);
};

Operation BroadOp(const Node* node) {
  // 返回一个 Lambda 函数作为 Operation
  return [](Stack& stack) {
    // 弹出并获取 b 张量
    auto b = pop(stack).toTensor();
    // 弹出并获取 a 张量
    auto a = pop(stack).toTensor();
    // 获取 b 和 a 的大小
    auto b_size = b.sizes();
    auto a_size = a.sizes();
    // 如果 a 和 b 的大小相等
    if (a_size.equals(b_size)) {
      // TODO: 与 MKLDNN 跟进，找出最佳的处理方式来处理性能不兼容的格式
      // 将 a 和 b 压回堆栈，保持不变
      push(stack, a, b);
      return;
    } else {
      auto out_size = at::infer_size(a_size, b_size);
      // 计算输出张量的元素总数
      int64_t out_numel = out_size[0];
      for (size_t i = 1, end = out_size.size(); i < end; ++i) {
        out_numel = out_numel * out_size[i];
      }

      auto exp_a = a;
      auto exp_b = b;
      int stacked = 0;
      // MKLDNN 张量仅支持 reshape，不支持 expand 或 view 操作符
      if (a_size.equals(out_size)) {
        // 如果输入张量 a 的尺寸与输出尺寸相同，将 a 推送到栈中
        push(stack, a);
        ++stacked;
      } else if (out_numel == a.numel()) {
        // 如果输出尺寸与 a 的元素总数相同，使用 reshape 将 a 转换为输出尺寸
        exp_a = a.reshape(out_size);
      } else {
        // 否则，将 a 转换为稠密张量，扩展到输出尺寸，然后转换为 MKLDNN 格式
        exp_a = a.to_dense().expand(out_size).to_mkldnn();
      }

      if (b_size.equals(out_size)) {
        // 如果输入张量 b 的尺寸与输出尺寸相同，将 b 推送到栈中
        push(stack, b);
        ++stacked;
      } else if (out_numel == b.numel()) {
        // 如果输出尺寸与 b 的元素总数相同，使用 reshape 将 b 转换为输出尺寸
        exp_b = b.reshape(out_size);
      } else {
        // 否则，将 b 转换为稠密张量，扩展到输出尺寸，然后转换为 MKLDNN 格式
        exp_b = b.to_dense().expand(out_size).to_mkldnn();
      }

      if (stacked < 2) {
        if (stacked == 1) {
          // 如果之前只有一个张量被推送到栈中，将其弹出
          pop(stack);
        }
        // 如果其中一个输入张量被扩展并转换为 nchw 或 nhwc 格式
        // 而第二个输入张量处于 blocked 格式，可能导致性能降低
        // 这种情况下，MKLDNN 使用其参考实现进行后续广播的二进制操作，
        // 这可能慢上 ~100 倍。
        // 我们使用一个简单的启发式方法，将一个 nchw 格式的参数转换为另一个参数的 blocked 格式。
        c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
        auto a_it = at::native::itensor_from_mkldnn(exp_a);
        auto b_it = at::native::itensor_from_mkldnn(exp_b);

        // `is_public_format` 意味着张量的物理布局不是 MKLDNN 的 blocked 格式，例如 nchw 或 nhwc，但不是 nChw8c
        if (!a_it.is_public_format()) {
          if (b_it.is_public_format()) {
            b_it = b_it.reorder_if_differ_in(a_it.get_desc());
          }
        } else if (!b_it.is_public_format()) {
          if (a_it.is_public_format()) {
            a_it = a_it.reorder_if_differ_in(b_it.get_desc());
          }
        }

        auto a_options = exp_a.options();
        // 创建新的 MKLDNN 张量 a_out，并将其推送到栈中
        auto a_out = at::native::new_with_itensor_mkldnn(
            std::move(a_it),
            c10::optTypeMetaToScalarType(a_options.dtype_opt()),
            a_options.device_opt());
        push(stack, a_out);
        
        auto b_options = exp_b.options();
        // 创建新的 MKLDNN 张量 b_out，并将其推送到栈中
        auto b_out = at::native::new_with_itensor_mkldnn(
            std::move(b_it),
            c10::optTypeMetaToScalarType(b_options.dtype_opt()),
            b_options.device_opt());
        push(stack, b_out);
      };
    }
  };
}

// 定义静态函数 hardtanh_helper，用于创建 hardtanh 操作的函数对象
static std::function<void(at::Tensor output, at::Tensor input)> hardtanh_helper(
    const Node* n) {
  // 从节点 n 中获取最小值和最大值
  auto min_val = n->f(attr::min_val);
  auto max_val = n->f(attr::max_val);
  // 返回一个 lambda 函数，执行 hardtanh 操作
  return [min_val, max_val](at::Tensor output, at::Tensor input) {
    at::cpu::hardtanh_out(output, input, min_val, max_val);
  };
}

// 定义静态函数 clamp_helper，用于创建 clamp 操作的函数对象
static std::function<void(at::Tensor output, at::Tensor input)> clamp_helper(
    const Node* n) {
  // 从节点 n 中获取最小值和最大值
  auto min_val = n->f(attr::min_val);
  auto max_val = n->f(attr::max_val);
  // 返回一个 lambda 函数，执行 clamp 操作
  return [min_val, max_val](at::Tensor output, at::Tensor input) {
    at::cpu::clamp_out(output, input, min_val, max_val);
  };
}

// 注册 MKLDNNHardSwishOpReg 操作符，添加到注册表中
// 该操作要求前置条件为 `aten_op(0) == 0`
const RegisterOperators MKLDNNHardSwishOpReg({
    // 注册 MKLDNNHardSwish_ 操作符，执行 hardswish 操作
    torch::jit::Operator(
        "prim::MKLDNNHardSwish_(Tensor(a!) self) -> Tensor(a!)",
        createUnaryOp(
            [](at::Tensor output, at::Tensor input) {
              at::cpu::hardswish_out(output, input);
            },
            true),
        AliasAnalysisKind::FROM_SCHEMA),
    // 注册 MKLDNNHardSigmoid_ 操作符，执行 hardsigmoid 操作
    torch::jit::Operator(
        "prim::MKLDNNHardSigmoid_(Tensor(a!) self) -> Tensor(a!)",
        createUnaryOp(
            [](at::Tensor output, at::Tensor input) {
              at::cpu::hardsigmoid_out(output, input);
            },
            true),
        AliasAnalysisKind::FROM_SCHEMA),
    // 注册 MKLDNNHardTanh_ 操作符，执行 hardtanh 操作
    torch::jit::Operator(
        "prim::MKLDNNHardTanh_(Tensor(a!) self) -> Tensor(a!)",
        [](const Node* n) -> Operation {
          return createUnaryOp(hardtanh_helper(n), true);
        },
        AliasAnalysisKind::FROM_SCHEMA),
    // 注册 MKLDNNClamp_ 操作符，执行 clamp 操作
    torch::jit::Operator(
        "prim::MKLDNNClamp_(Tensor(a!) self) -> Tensor(a!)",
        [](const Node* n) -> Operation {
          return createUnaryOp(clamp_helper(n), true);
        },
        AliasAnalysisKind::FROM_SCHEMA),
    // 注册 MKLDNNHardSwish 操作符，执行 hardswish 操作
    torch::jit::Operator(
        "prim::MKLDNNHardSwish(Tensor a) -> Tensor",
        createUnaryOp(
            [](at::Tensor output, at::Tensor input) {
              at::cpu::hardswish_out(output, input);
            },
            false),
        AliasAnalysisKind::FROM_SCHEMA),
    // 注册 MKLDNNHardSigmoid 操作符，执行 hardsigmoid 操作
    torch::jit::Operator(
        "prim::MKLDNNHardSigmoid(Tensor a) -> Tensor",
        createUnaryOp(
            [](at::Tensor output, at::Tensor input) {
              at::cpu::hardsigmoid_out(output, input);
            },
            false),
        AliasAnalysisKind::FROM_SCHEMA),
    // 注册 MKLDNNHardTanh 操作符，执行 hardtanh 操作
    torch::jit::Operator(
        "prim::MKLDNNHardTanh(Tensor self) -> Tensor",
        [](const Node* n) -> Operation {
          return createUnaryOp(hardtanh_helper(n), false);
        },
        AliasAnalysisKind::FROM_SCHEMA),
    // 注册 MKLDNNClamp 操作符，执行 clamp 操作
    torch::jit::Operator(
        "prim::MKLDNNClamp(Tensor self) -> Tensor",
        [](const Node* n) -> Operation {
          return createUnaryOp(clamp_helper(n), false);
        },
        AliasAnalysisKind::FROM_SCHEMA),
});

// 注册 BroadOpReg 操作符，添加到注册表中
const RegisterOperators BroadOpReg({
    # 创建一个 Torch JIT 操作符，用于处理 MKLDNN 张量的广播
    torch::jit::Operator(
        prim::BroadcastMKLDNNTensors,   // 操作符的名称，用于广播 MKLDNN 张量
        BroadOp,                        // 操作符的实现函数或对象
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),  // 别名分析的种类，这里是内部特殊情况
// 注册 MKLDNNLayerNormOpReg 操作符，用于注册 MKLDNNLayerNorm 的两个版本：带下划线的版本支持原地操作
const RegisterOperators MKLDNNLayerNormOpReg({
    // 注册 torch::jit::Operator，实现了 "prim::MKLDNNLayerNorm" 操作，调用 MKLDNNLayerNormOp 函数
    torch::jit::Operator(
        "prim::MKLDNNLayerNorm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor",
        [](Stack& stack) { MKLDNNLayerNormOp(stack, false); },
        AliasAnalysisKind::FROM_SCHEMA),
    // 注册 torch::jit::Operator，实现了 "prim::MKLDNNLayerNorm_" 操作，调用 MKLDNNLayerNormOp 函数（支持原地操作）
    torch::jit::Operator(
        "prim::MKLDNNLayerNorm_(Tensor(a!) input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor(a!)",
        [](Stack& stack) { MKLDNNLayerNormOp(stack, true); },
        AliasAnalysisKind::FROM_SCHEMA),
});

// 实现 ConstantMKLDNNTensorOp 函数，返回一个操作，将节点中的常量张量 t 推送到栈顶
Operation ConstantMKLDNNTensorOp(const Node* node) {
  // 从节点属性中获取张量 t
  const auto& t = node->t(attr::value);
  return [t](Stack& stack) {
    // 将张量 t 推送到栈顶
    push(stack, t);
    return 0;
  };
}

// 实现 mkldnn_tensor_scalar_mul 函数，执行 MKLDNN 张量和标量的乘法操作
Tensor mkldnn_tensor_scalar_mul(Tensor& tensor, Tensor& out, float scalar) {
  // 将 PyTorch 张量转换为 ideep::tensor
  ideep::tensor& x = at::native::itensor_from_mkldnn(tensor);
  ideep::tensor& z = at::native::itensor_from_mkldnn(out);
  // 执行 eltwise_linear 算法的 eltwise_forward 计算，对 x 应用标量乘法
  ideep::eltwise_forward::compute(
      x,
      z,
      ideep::algorithm::eltwise_linear,
      ideep::prop_kind::forward_inference,
      /*alpha*/ scalar);
  // 返回输出张量 out
  return out;
}

// 注释关于 aten::convolution 的说明，注册 mkldnn_convolution 操作以避免预计算和调度开销
// 该自定义操作能够直接调用 mkldnn_convolution 而避免其他操作符的调度开销，如 relu、add 等
jit::RegisterOperators reg_fut_ops({
    jit::Operator(
        // 定义名为 prim::mkldnn_convolution 的自定义运算符，接受输入张量、权重张量、可选偏置张量以及步幅、填充、扩展和分组信息，返回张量
        "prim::mkldnn_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
        [](jit::Stack& stack) {
          // 弹出堆栈顶部的元素作为分组数，转换为 int64_t 类型
          int64_t groups = pop(stack).toInt();
          // 弹出堆栈顶部的元素作为扩展数组，转换为 int 型向量
          auto dilation = pop(stack).toIntVector();
          // 弹出堆栈顶部的元素作为填充数组，转换为 int 型向量
          auto padding = pop(stack).toIntVector();
          // 弹出堆栈顶部的元素作为步幅数组，转换为 int 型向量
          auto stride = pop(stack).toIntVector();

          // 定义一个张量变量 bias
          Tensor bias;
          // 弹出堆栈顶部的元素作为偏置张量，如果不为空，则将其转换为 Tensor 类型并赋给 bias
          IValue bias_ival = pop(stack);
          if (!bias_ival.isNone()) {
            bias = bias_ival.toTensor();
          }
          // 弹出堆栈顶部的元素作为权重张量，转换为 Tensor 类型并赋给 weight
          Tensor weight = pop(stack).toTensor();
          // 弹出堆栈顶部的元素作为输入张量，转换为 Tensor 类型并赋给 input
          Tensor input = pop(stack).toTensor();

          // 自动选择低于自动微分的调度模式
          at::AutoDispatchBelowAutograd mode;
          // 如果输入张量的第一个维度为 0，调用 conv_output_size 计算输出大小，并返回空的 MKLDNN 张量
          if (input.size(0) == 0) {
            std::vector<int64_t> o = at::native::conv_output_size(
                input.sizes(), weight.sizes(), padding, stride, dilation);
            push(
                stack,
                at::native::empty_mkldnn(
                    o,
                    c10::optTypeMetaToScalarType(input.options().dtype_opt()),
                    input.options().layout_opt(),
                    input.options().device_opt(),
                    input.options().pinned_memory_opt()));
            return;
          }
          // 检查输入张量与权重张量的数据类型是否匹配
          TORCH_CHECK(
              input.options().type_equal(weight.options()),
              "Input type (",
              input.toString(),
              ") and weight type (",
              weight.toString(),
              ") should be the same");

          // 调用 mkldnn_convolution 函数进行 MKLDNN 卷积操作，将结果推送回堆栈
          push(
              stack,
              at::native::mkldnn_convolution(
                  input, weight, bias, padding, stride, dilation, groups));
        },
        aliasAnalysisFromSchema()),
    jit::Operator(
        // 定义名为 prim::MKLDNNScalarMul 的自定义运算符，接受输入张量和标量，返回张量
        "prim::MKLDNNScalarMul(Tensor self, Scalar other) -> Tensor",
        [](jit::Stack& stack) {
          // 排除自动微分键集的调度键保护
          c10::impl::ExcludeDispatchKeyGuard edkg(
              c10::autograd_dispatch_keyset);
          // 弹出堆栈顶部的元素作为标量，转换为 float 类型
          float other = pop(stack).toScalar().toFloat();
          // 弹出堆栈顶部的元素作为输入张量，转换为 Tensor 类型并赋给 self
          Tensor self = pop(stack).toTensor();
          // 调用 empty_mkldnn 创建一个与输入张量相同大小的空 MKLDNN 张量
          auto out = at::native::empty_mkldnn(
              self.sizes(),
              c10::optTypeMetaToScalarType(self.options().dtype_opt()),
              self.options().layout_opt(),
              self.options().device_opt(),
              self.options().pinned_memory_opt());

          // 调用 mkldnn_tensor_scalar_mul 函数对输入张量进行标量乘法操作，并将结果推送回堆栈
          mkldnn_tensor_scalar_mul(self, out, other);
          push(stack, out);
        },
        aliasAnalysisFromSchema()),
    // 定义了一个名为 "prim::MKLDNNScalarMul_" 的运算符，接受一个 Tensor 类型的 self 和一个 Scalar 类型的 other，返回一个修改后的 Tensor self
    jit::Operator(
        "prim::MKLDNNScalarMul_(Tensor(a!) self, Scalar other) -> Tensor(a!)",
        [](jit::Stack& stack) {
          // 排除自动求导分发键集，确保不被自动求导系统拦截
          c10::impl::ExcludeDispatchKeyGuard edkg(
              c10::autograd_dispatch_keyset);
          // 从栈中弹出并转换成浮点数，作为运算中的乘数 other
          float other = pop(stack).toScalar().toFloat();
          // 从栈中弹出并转换成 Tensor 类型，作为运算中的被乘数 self
          Tensor self = pop(stack).toTensor();
          // 调用 mkldnn_tensor_scalar_mul 函数，使用 MKL-DNN 加速执行 self *= other 的操作
          mkldnn_tensor_scalar_mul(self, self, other);
          // 将计算后的 Tensor 结果推送回栈顶
          push(stack, self);
        },
        // 根据运算符的模式生成别名分析，用于优化运算符的使用
        aliasAnalysisFromSchema()),
});

// 以下注册了一个名为 prim::ConstantMKLDNNTensor 的运算符，其实现为 ConstantMKLDNNTensorOp，
// 并且指定了其别名分析的类型为 INTERNAL_SPECIAL_CASE
const RegisterOperators MKLDNNConstantOp({
    torch::jit::Operator(
        prim::ConstantMKLDNNTensor,
        ConstantMKLDNNTensorOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

// 创建并返回一个代表常量 MKLDNN 张量的节点 op
Node* createConstantMKLDNNTensorOp(Graph* g, const Tensor& mkldnn_tensor) {
  TORCH_INTERNAL_ASSERT(mkldnn_tensor.is_mkldnn());
  auto op = g->create(prim::ConstantMKLDNNTensor);  // 创建一个常量 MKLDNN 张量的操作节点
  op->t_(attr::value, mkldnn_tensor);  // 设置操作节点的 value 属性为给定的 MKLDNN 张量
  return op;  // 返回创建的操作节点
}

// 判断给定的权重张量是否是支持的 MKLDNN 权重
bool supportedMKLDNNWeight(const Tensor& weight) {
  return weight.device().is_cpu() && weight.dtype() == c10::ScalarType::Float &&
      weight.ndimension() != 0;  // 条件判断权重张量是否在 CPU 上，数据类型为浮点型，且维度不为零
}

// 将节点 n 的输入值替换为 MKLDNN 张量
void replaceInputWithMKLDNNTensor(Node* n, size_t index) {
  Value* input = n->inputs().at(index);  // 获取节点 n 的第 index 个输入值
  auto mkldnn_tensor = constant_as<Tensor>(input)->to_mkldnn();  // 将输入值转换为 MKLDNN 张量
  auto mkldnn_tensor_value =
      createConstantMKLDNNTensorOp(n->owningGraph(), mkldnn_tensor)  // 创建代表 MKLDNN 张量的节点 op
          ->insertBefore(n)  // 将 op 插入到节点 n 之前
          ->output();  // 获取 op 的输出值
  mkldnn_tensor_value->setDebugName(input->debugName() + "_mkldnn");  // 设置输出值的调试名称
  n->replaceInputWith(input, mkldnn_tensor_value);  // 将节点 n 的输入值 input 替换为 MKLDNN 张量的输出值
}

// 将节点 n 的指定名称的输入值替换为 MKLDNN 张量
void replaceInputWithMKLDNNTensor(
    Node* n,
    const std::string& name,
    const at::Tensor& mkldnn_tensor) {
  Value* input = n->namedInput(name);  // 获取节点 n 的指定名称的输入值
  auto mkldnn_tensor_value =
      createConstantMKLDNNTensorOp(n->owningGraph(), mkldnn_tensor)  // 创建代表 MKLDNN 张量的节点 op
          ->insertBefore(n)  // 将 op 插入到节点 n 之前
          ->output();  // 获取 op 的输出值
  mkldnn_tensor_value->setDebugName(input->debugName() + "_mkldnn");  // 设置输出值的调试名称
  n->replaceInputWith(input, mkldnn_tensor_value);  // 将节点 n 的输入值 input 替换为 MKLDNN 张量的输出值
}

// 将节点 n 的指定名称的输入值替换为 MKLDNN 张量
void replaceInputWithMKLDNNTensor(Node* n, const std::string& name) {
  Value* input = n->namedInput(name);  // 获取节点 n 的指定名称的输入值
  auto mkldnn_tensor = constant_as<Tensor>(input)->to_mkldnn();  // 将输入值转换为 MKLDNN 张量
  replaceInputWithMKLDNNTensor(n, name, mkldnn_tensor);  // 调用前一个函数进行替换
}

// 将卷积节点 conv 的权重张量移至 MKLDNN 张量
void moveConvWeightsToMKLDNN(Node* conv) {
  auto conv_w_mkldnn =
      constant_as<Tensor>(conv->namedInput("weight")).value().to_mkldnn();  // 获取并转换卷积节点的权重张量为 MKLDNN 张量
  std::vector<int64_t> padding =
      toIValue(conv->namedInput("padding"))->toIntVector();  // 获取卷积节点的填充参数
  std::vector<int64_t> stride =
      toIValue(conv->namedInput("stride"))->toIntVector();  // 获取卷积节点的步幅参数
  std::vector<int64_t> dilation =
      toIValue(conv->namedInput("dilation"))->toIntVector();  // 获取卷积节点的膨胀参数
  auto groups = constant_as<int64_t>(conv->namedInput("groups")).value();  // 获取卷积节点的组参数

  // 根据卷积节点的类型重新排序权重张量为 MKLDNN 张量
  if (conv->kind() == aten::conv2d) {
    conv_w_mkldnn = mkldnn_reorder_conv2d_weight(
        conv_w_mkldnn, padding, stride, dilation, groups);
  } else if (conv->kind() == aten::conv3d) {
    conv_w_mkldnn = mkldnn_reorder_conv3d_weight(
        conv_w_mkldnn, padding, stride, dilation, groups);
  } else {
    TORCH_INTERNAL_ASSERT(false);  // 如果卷积节点的类型不是 2D 或 3D 卷积，断言失败
  }
  replaceInputWithMKLDNNTensor(conv, "weight", conv_w_mkldnn);  // 将卷积节点的权重输入替换为 MKLDNN 张量

  // 如果卷积节点有偏置项，则将其输入替换为 MKLDNN 张量
  if (conv->namedInput("bias")->type() != NoneType::get()) {
    replaceInputWithMKLDNNTensor(conv, "bias");
  }
}
void moveWeightsToMKLDNN(Node* n) {
  // 如果节点是 conv2d 或者 conv3d 类型，则调用 moveConvWeightsToMKLDNN 函数
  // 这是为了将卷积操作的权重移动到 MKLDNN 中处理的特殊路径
  if (n->kind() == aten::conv2d || n->kind() == aten::conv3d) {
    moveConvWeightsToMKLDNN(n);
  } else {
    // 遍历节点的输入
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      // 如果输入不是 Tensor 类型或者不是常量节点，则跳过
      if (!n->input(i)->type()->cast<TensorType>() ||
          n->input(i)->node()->kind() != prim::Constant) {
        continue;
      }
      // 替换输入为 MKLDNN Tensor
      replaceInputWithMKLDNNTensor(n, i);
    }
  }
}

static void clamp_node_creator(
    Node* body_node,
    c10::Symbol kind,
    double min_val,
    double max_val) {
  // 设置插入点为 body_node，创建一个新节点
  WithInsertPoint insert_guard{body_node};
  auto out_node =
      body_node->owningGraph()->create({kind}, {body_node->input(0)}, 1);
  // 注意：不能使用 `insert` 方法，因为它会调用 `getOperation`（通过 `emitBuiltinCall`）
  // 而这些方法使用了尚未设置的 `min_val` 和 `max_val` 属性。
  body_node->owningGraph()->insertNode(out_node);
  auto out_val = out_node->output();
  // 设置新节点的 min_val 和 max_val 属性
  out_node->f_(attr::min_val, min_val);
  out_node->f_(attr::max_val, max_val);
  // 复制输出的元数据到新节点的输出
  out_val->copyMetadata(body_node->output());
  // 用新节点的输出替换原节点的输出
  body_node->output()->replaceAllUsesWith(out_val);
  // 销毁原节点
  body_node->destroy();
}

void ComputeSubgraphInMKLDNN(Node* subgraph_node) {
  auto graph = subgraph_node->owningGraph();
  Value* none_value = nullptr;
  {
    // 设置插入点为 subgraph_node，插入一个空的常量节点
    WithInsertPoint guard(subgraph_node);
    none_value = graph->insertConstant(IValue());
  }
  // 遍历子图节点的输入
  for (size_t i = 0; i < subgraph_node->inputs().size(); ++i) {
    Value* v = subgraph_node->inputs().at(i);
    // 如果输入不是 Tensor 类型，则跳过
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    // 创建一个 to_mkldnn 节点并插入到 subgraph_node 前面，将当前输入作为其输入
    auto to_mkldnn =
        graph->create(c10::Symbol::fromQualString("aten::to_mkldnn"), 1)
            ->insertBefore(subgraph_node);
    to_mkldnn->addInput(v);
    to_mkldnn->addInput(none_value);
    // 替换子图节点的输入为 to_mkldnn 节点的输出
    subgraph_node->replaceInput(i, to_mkldnn->output());
  }

  // 遍历子图节点的输出
  for (size_t i = 0; i < subgraph_node->outputs().size(); ++i) {
    Value* v = subgraph_node->outputs().at(i);
    // 如果输出不是 Tensor 类型，则跳过
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    // 创建一个 from_mkldnn 节点并插入到 subgraph_node 后面，将当前输出作为其输入
    auto from_mkldnn = graph
                           ->create(
                               c10::Symbol::fromQualString("aten::to_dense"),
                               {v, none_value, none_value})
                           ->insertAfter(subgraph_node);
    // 替换 v 的所有使用为 from_mkldnn 节点的输出
    v->replaceAllUsesAfterNodeWith(from_mkldnn, from_mkldnn->output());
  }

  // 获取子图节点的子图对象
  auto subgraph = SubgraphUtils::getSubgraph(subgraph_node);
  // 遍历子图中的所有节点
  for (auto it = subgraph->block()->nodes().begin();
       it != subgraph->block()->nodes().end();) {
    Node* body_node = *it;
    it++;

    // 调用 moveWeightsToMKLDNN 函数处理节点
    moveWeightsToMKLDNN(body_node);
    // 检查当前节点是否为加法或乘法，并且第二个输入是张量类型
    if (body_node->kind() == aten::add ||
        (body_node->kind() == aten::mul &&
         body_node->input(1)->type()->cast<TensorType>())) {
      // 创建一个新的节点来执行 MKLDNN 张量的广播操作
      auto node = body_node->owningGraph()->create(
          Symbol::prim("BroadcastMKLDNNTensors"),
          {body_node->inputs().at(0), body_node->inputs().at(1)},
          2);
      // 在当前节点之前插入新创建的节点
      node->insertBefore(body_node);
      // 替换当前节点的输入为新节点的输出
      body_node->replaceInput(0, node->outputs().at(0));
      body_node->replaceInput(1, node->outputs().at(1));
    }

    // 检查当前节点是否为乘法，并且第二个输入是数值类型
    if (body_node->kind() == aten::mul &&
        body_node->input(1)->type()->isSubtypeOf(*NumberType::get())) {
      // 将当前节点替换为 MKLDNN 标量乘法的新符号
      body_node->replaceWithNewSymbol(Symbol::prim("MKLDNNScalarMul"));
      // 销毁当前节点
      body_node->destroy();
      // 继续处理下一个节点
      continue;
    }

    // 检查当前节点是否匹配指定的 layer_norm 操作
    if (body_node->matches(
            "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor")) {
      // 将当前节点替换为 MKLDNN 的 layer_norm 操作符
      body_node->replaceWithNewSymbol(Symbol::prim("MKLDNNLayerNorm"));
      // 销毁当前节点
      body_node->destroy();
      // 继续处理下一个节点
      continue;
    }

    // 检查当前节点是否为 hardswish 操作
    if (body_node->kind() == aten::hardswish) {
      // 将当前节点替换为 MKLDNN 的 hardswish 操作符
      body_node->replaceWithNewSymbol(prim::MKLDNNHardSwish);
      // 销毁当前节点
      body_node->destroy();
      // 继续处理下一个节点
      continue;
    }

    // 检查当前节点是否为 hardsigmoid 操作
    if (body_node->kind() == aten::hardsigmoid) {
      // 将当前节点替换为 MKLDNN 的 hardsigmoid 操作符
      body_node->replaceWithNewSymbol(prim::MKLDNNHardSigmoid);
      // 销毁当前节点
      body_node->destroy();
      // 继续处理下一个节点
      continue;
    }

    // 检查当前节点是否为 relu6 操作
    if (body_node->kind() == aten::relu6) {
      // 使用 clamp_node_creator 函数创建一个 MKLDNN 的 hardtanh 操作符，并设置范围为 [0, 6]
      clamp_node_creator(body_node, prim::MKLDNNHardTanh, 0., 6.);
      // 继续处理下一个节点
      continue;
    }

    // 检查当前节点是否为 hardtanh 操作
    if (body_node->kind() == aten::hardtanh) {
      // 从节点的命名输入中获取 min_val 和 max_val 的常数值
      auto min_val =
          constant_as<double>(body_node->namedInput("min_val")).value();
      auto max_val =
          constant_as<double>(body_node->namedInput("max_val")).value();
      // 使用 clamp_node_creator 函数创建一个 MKLDNN 的 hardtanh 操作符，并设置范围为 [min_val, max_val]
      clamp_node_creator(body_node, prim::MKLDNNHardTanh, min_val, max_val);
      // 继续处理下一个节点
      continue;
    }

    // 检查当前节点是否为 clamp 操作
    if (body_node->kind() == aten::clamp) {
      // 从节点的命名输入中获取 min_val 和 max_val 的常数值
      auto min_val = constant_as<double>(body_node->namedInput("min")).value();
      auto max_val = constant_as<double>(body_node->namedInput("max")).value();
      // 使用 clamp_node_creator 函数创建一个 MKLDNN 的 clamp 操作符，并设置范围为 [min_val, max_val]
      clamp_node_creator(body_node, prim::MKLDNNClamp, min_val, max_val);
      // 继续处理下一个节点
      continue;
    }

    // 检查当前节点是否为 conv2d 或 conv3d 操作
    if (body_node->kind() == aten::conv2d ||
        body_node->kind() == aten::conv3d) {
      // 检查节点的 padding 输入是否不是字符串类型
      if (!body_node->namedInput("padding")->type()->cast<StringType>()) {
        // 将当前节点替换为 MKLDNN 的卷积操作符
        body_node->replaceWithNewSymbol(Symbol::prim("mkldnn_convolution"));
        // 销毁当前节点
        body_node->destroy();
        // 继续处理下一个节点
        continue;
      }
    }
}

// 检查节点的非常量输入参数是否存在
bool nonConstantParameters(Node* n) {
  // 遍历节点的输入，从第二个输入开始检查
  for (size_t i = 1; i < n->inputs().size(); i++) {
    // 如果节点的某个输入不是常量，则返回 true
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  // 如果所有输入都是常量，则返回 false
  return false;
}

// 检查是否可以将给定节点的权重转换为 MKLDNN 支持的格式，并且节点是线性操作
bool frozenMkldnnCompatibleLinearNode(Node* n) {
  // 如果节点有非常量输入参数，则不支持
  if (nonConstantParameters(n)) {
    return false;
  }

  // 如果节点不是线性操作，则不支持
  if (n->kind() != aten::linear) {
    return false;
  }

  // 获取节点的权重，并检查其是否支持 MKLDNN 格式
  auto weight = constant_as<Tensor>(n->namedInput("weight")).value();
  return supportedMKLDNNWeight(weight);
}

// 检查是否可以将给定节点的权重转换为 MKLDNN 支持的格式，并且节点是卷积操作
bool frozenMkldnnCompatibleConvNode(Node* n) {
  // 如果节点有非常量输入参数，则不支持
  if (nonConstantParameters(n)) {
    return false;
  }

  // 对于 conv1d 操作，MKLDNN 不提供支持
  // _convolution 在此优化执行前已被重写
  if (n->kind() != aten::conv2d && n->kind() != aten::conv3d) {
    return false;
  }

  // 获取节点的权重，并检查其是否支持 MKLDNN 格式
  auto weight = constant_as<Tensor>(n->namedInput("weight")).value();
  return supportedMKLDNNWeight(weight);
}

// MKLDNN 子图切片器类
class MKLDNNSubgraphSlicer {
 public:
  // 构造函数，初始化成员变量
  MKLDNNSubgraphSlicer(
      Block* block,
      std::shared_ptr<Graph> graph,
      AliasDb& aliasDb)
      : block_(block), graph_(std::move(graph)), aliasDb_(aliasDb) {}

  // 运行 MKLDNN 子图切片器
  void run() {
    // 在构建自动微分子图时保持别名数据库的正确性，但在取消内联自动微分子图时保持正确性比较困难
    buildupSubgraphs();
    computeSubgraphsInMKLDNN();
    // 运行全局的 CSE，以消除内联子图可能导致的重复计算
    // CSE：公共子表达式消除
    runGlobalCSE();
  }

 private:
  // 成员变量，用于保存基础块、图和别名数据库
  Block* block_;
  std::shared_ptr<Graph> graph_;
  AliasDb& aliasDb_;

  // 构建子图，递归构建所有子图并将其合并到图中
  void buildupSubgraphs() {
    // 实现详细略
  }

  // 计算 MKLDNN 中的子图
  void computeSubgraphsInMKLDNN() {
    // 实现详细略
  }

  // 运行全局的 CSE，消除可能在内联子图中产生的重复计算
  void runGlobalCSE() {
    // 实现详细略
  }
};
    EliminateCommonSubexpression(graph_);
  }

  void buildupSubgraphs() {
    // 我们需要多次运行切片器以获取所有合并机会。
    // 这是因为 moveBeforeTopologicalValid 可能会重新排列节点，使其在当前迭代点之后。
    // 为了正确考虑这些节点是否可以合并，我们需要运行该 pass 直到不再有更改为止。
    //
    // 示例:
    //   c = f(a, b)
    //   d = f(c)
    //   e = f(d)  <- 迭代点在此，向上移动
    // 在 c.moveBeforeTopologicallyValid(e) 之后，我们有:
    //   c = f(a, b)
    //   e = f(d)  <- 迭代点仍然在此
    //   d = f(c)  <- 这是在另一侧移动的节点。

    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block_->nodes().begin(); it != block_->nodes().end();) {
        bool changed = false;
        // 扫描节点并尝试执行可能的重组
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    // 递归构建子图
    for (Node* n : block_->nodes()) {
      for (auto subBlock : n->blocks()) {
        // 为子块构建子图
        MKLDNNSubgraphSlicer(subBlock, graph_, aliasDb_).buildupSubgraphs();
      }
    }
  }

  static bool MKLDNNGroupStart(Node* node) {
    // 如果节点已经在合并过程中
    if (node->kind() == prim::MKLDNNGroup) {
      return true;
    }
    // 查看 MKLDNN 性能策略
    return frozenMkldnnCompatibleConvNode(node);
  }

 private:
  // MKLDNN 仅支持维度大于 0 的浮点数，因此我们只支持已知类型或先前验证可在 MKLDNN 组中使用的张量
  bool tensorInputIsMKLDNNSupported(Value* v, Node* v_use) {
    auto const_tensor = constant_as<Tensor>(v);
    if (const_tensor) {
      return supportedMKLDNNWeight(*const_tensor);
    }
    auto k = v->node()->kind();
    if (k == prim::MKLDNNGroup || k == prim::ConstantMKLDNNTensor ||
        k == aten::to_mkldnn) {
      return true;
    }
    for (const auto& use : v->uses()) {
      if (use.user->kind() == aten::to_mkldnn &&
          v_use->owningBlock() == use.user->owningBlock()) {
        return true;
      }
    }
    return false;
  }

  // 我们在此处包括大致与 aten 中的 MKLDNN 性能等效的操作，其输入和输出为 float32
  bool computableInMKLDNN(Node* n) {
    for (Value* v : n->inputs()) {
      if (v->type()->cast<TensorType>() &&
          !(tensorInputIsMKLDNNSupported(v, n))) {
        return false;
      }
    }
    // 可在 MKLDNN 中计算
    return true;
  }
    // 检查节点类型是否为 layer_norm，且 weight 和 bias 非空
    if (n->matches(
            "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor") &&
        n->namedInput("weight")->type() != NoneType::get() &&
        n->namedInput("bias")->type() != NoneType::get()) {
      auto norm_shape =
          constant_as<std::vector<int64_t>>(n->namedInput("normalized_shape"));
      // 检查 normalized_shape 是否存在且长度为 1
      return norm_shape.has_value() && norm_shape->size() == 1;
    }

    // 对于一元操作，只需确保输入支持 mkldnn
    switch (n->kind()) {
      case aten::relu:
      case aten::relu6:
      case aten::gelu:
      case aten::prelu:
      case aten::sigmoid:
      case aten::hardsigmoid:
      case aten::hardswish:
      case aten::tanh:
      case aten::batch_norm:
      case aten::max_pool2d:
      case aten::max_pool3d:
      case aten::avg_pool2d:
      case aten::adaptive_avg_pool2d:
      case aten::avg_pool3d:
        // 对于这些操作，返回 true
        return true;
    }

    // 对于 hardtanh 和 clamp 操作，检查非常量参数，确保 min_val <= 0 且 max_val >= 0
    if ((n->kind() == aten::hardtanh || n->kind() == aten::clamp) &&
        !nonConstantParameters(n)) {
      const size_t MIN_INDEX = 1, MAX_INDEX = 2;
      auto min_val = constant_as<double>(n->input(MIN_INDEX)).value();
      auto max_val = constant_as<double>(n->input(MAX_INDEX)).value();
      // 保持不变式 `pointwise_func(0) == 0`
      if (min_val <= 0. && max_val >= 0.) {
        return true;
      }
    }

    // 对于 add 操作，检查是否是 Tensor-Scalar 相加
    if (n->kind() == aten::add) {
      for (const auto i : c10::irange(2)) {
        // 检查前两个输入是否都是 TensorType
        if (!n->inputs().at(i)->type()->cast<TensorType>()) {
          return false;
        }
      }
      return true;
    }

    // 对于 mul 操作，检查第一个输入是 TensorType，第二个输入可以是 TensorType 或者 NumberType
    if (n->kind() == aten::mul) {
      return n->input(0)->type()->cast<TensorType>() &&
          (n->input(1)->type()->cast<TensorType>() ||
           n->input(1)->type()->isSubtypeOf(*NumberType::get()));
    }

    // 对于 dropout 操作，检查 train 是否为 false
    if (n->kind() == aten::dropout) {
      auto train = constant_as<bool>(n->namedInput("train")).value();
      return train == false;
    }

    // 默认返回 false
    return false;
  }

  // 在当前 block 中查找 MKLDNNGroup 节点，并对其进行处理
  void computeSubgraphsInMKLDNN() {
    auto curNode = *block_->nodes().begin();
    // 遍历 block 中的每个节点
    while (curNode != *block_->nodes().end()) {
      auto nextNode = curNode->next();
      // 如果当前节点是 MKLDNNGroup 类型，进行相关操作
      if (curNode->kind() == prim::MKLDNNGroup) {
        ComputeSubgraphInMKLDNN(curNode);
        InplaceMKLDNNSubgraph(SubgraphUtils::getSubgraph(curNode));
        SubgraphUtils::unmergeSubgraph(curNode);
      }
      curNode = nextNode;
    }
    // 对 block 中的每个节点的子块递归调用 computeSubgraphsInMKLDNN
    for (Node* n : block_->nodes()) {
      for (Block* b : n->blocks()) {
        MKLDNNSubgraphSlicer(b, graph_, aliasDb_).computeSubgraphsInMKLDNN();
      }
    }
  }
  }
  // 结束了一个类或命名空间的定义

  bool shouldConsiderForMerge(Node* node) {
    // 判断是否应该考虑将节点合并到MKLDNN组中
    // 如果节点已经是MKLDNN组，则返回true
    if (node->kind() == prim::MKLDNNGroup) {
      return true;
    }
    // 否则根据一定的条件判断节点是否适合与MKLDNN组中的节点合并
    return frozenMkldnnCompatibleLinearNode(node) ||
        frozenMkldnnCompatibleConvNode(node) || computableInMKLDNN(node);
  }

  std::pair<graph_node_list::iterator, bool> scanNode(Node* producer) {
    // 扫描生产者节点及其输出节点，尝试进行节点合并操作
    if (MKLDNNGroupStart(producer)) {
      // 如果生产者节点是MKLDNN组的起始节点
      if (producer->kind() != prim::MKLDNNGroup) {
        // 如果生产者节点不是MKLDNN组，则创建单例子图并更新别名信息
        producer = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
            producer, prim::MKLDNNGroup, aliasDb_);
      }
      // 收集生产者节点所有输出节点
      std::vector<Node*> output_nodes;
      for (Value* v : producer->outputs()) {
        for (const auto& use : v->uses()) {
          output_nodes.push_back(use.user);
        }
      }
      // 根据节点顺序排序
      std::sort(
          output_nodes.begin(), output_nodes.end(), [&](Node* a, Node* b) {
            return a->isBefore(b);
          });
      // 尝试将每个输出节点与生产者节点合并
      for (auto output_node : output_nodes) {
        if (auto group = tryMerge(producer, output_node)) {
          // 如果成功合并，则可能会改变新组的输出，需要重新扫描以寻找更多合并机会
          return std::make_pair(group.value()->iterator()++, true);
        }
      }
    }

    // 返回下一个生产者节点的迭代器及合并是否成功的标志
    return std::make_pair(++producer->iterator(), false);
  }

  // 尝试将消费者节点合并到生产者节点中，如果成功则销毁消费者节点并返回生产者组
  std::optional<Node*> tryMerge(Node* producer, Node* consumer) {
    // 断言生产者节点是MKLDNN组
    AT_ASSERT(producer->kind() == prim::MKLDNNGroup);
    // 判断是否可以将消费者节点合并到生产者节点中，并满足拓扑顺序要求
    bool canMerge = shouldConsiderForMerge(consumer) &&
        aliasDb_.moveAfterTopologicallyValid(consumer, producer);

    if (!canMerge) {
      return c10::nullopt;
    }

    // 将消费者节点合并到生产者子图中，并更新别名信息
    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        consumer, producer, aliasDb_);

    // 返回合并后的生产者组
    return producer;
  }

  Block* block_;
  std::shared_ptr<Graph> graph_;
  AliasDb& aliasDb_;
} // namespace

// 将冻结的运算转换为 MKLDNN 格式
void ConvertFrozenOpsToMKLDNN(std::shared_ptr<Graph>& graph) {
  // 输出转换前的图结构
  GRAPH_DUMP("Before convert frozen ops to mkldnn", graph);

  // 替换卷积操作为 Aten Conv
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  // 如果图中包含 MKLDNN 兼容的分组
  if (containsMKLDNNGroup(graph->block())) {
    // 只有当我们确信使用 MKLDNN 可以加速时才移除张量突变
    // 仅支持功能型操作简化了此转换，因为在 MKLDNN 中运行操作会消除之前存在的输入和输出之间的别名关系
    RemoveTensorMutation(graph, [](Node* node_to_functionalize) {
      // MKLDNN 支持的操作集合
      static std::unordered_set<Symbol> mkldnn_ops = {
          aten::add_,
          aten::mul_,
          aten::relu_,
          aten::relu6_,
          aten::gelu_,
          aten::hardswish_,
          aten::dropout_,
          aten::sigmoid_,
          aten::hardsigmoid_,
          aten::hardtanh_,
          aten::tanh_,
          aten::clamp_,
      };
      // 判断节点是否属于 MKLDNN 支持的操作
      return mkldnn_ops.count(node_to_functionalize->kind()) != 0;
    });

    // 构建别名数据库
    AliasDb db(graph);
    // 执行 MKLDNN 子图切割
    MKLDNNSubgraphSlicer(graph->block(), graph, db).run();
    // 消除死代码
    EliminateDeadCode(graph);
    // 输出转换后的图结构
    GRAPH_DUMP("After convert frozen ops to mkldnn", graph);
  } else {
    // 如果图中没有 MKLDNN 兼容的冻结节点
    GRAPH_DUMP("No mkldnn compatible frozen nodes", graph);
  }
}

#else

// 当 MKLDNN 未启用时的处理函数
void ConvertFrozenOpsToMKLDNN(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("MKLDNN Not enabled", graph);
}

#endif

} // namespace jit
} // namespace torch
```