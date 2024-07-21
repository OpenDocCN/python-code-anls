# `.\pytorch\torch\csrc\jit\runtime\profiling_graph_executor_impl.cpp`

```
// 引入Torch JIT运行时的头文件，实现了性能分析相关的图执行器功能
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>

// 引入C10库中的可选类型以及整数范围的工具
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

// 引入Torch JIT的日志记录功能
#include <torch/csrc/jit/jit_log.h>

// 引入Torch JIT的一系列优化和转换 passes
#include <torch/csrc/jit/passes/add_if_then_else.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/check_strict_fusion.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/inplace_check.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 引入标准库的时间和互斥量支持
#include <chrono>
#include <mutex>

// 定义布尔类型的配置标志，用于控制TorchScript新执行器的启用
C10_DEFINE_bool(
    torch_jit_enable_new_executor,
    true,
    "If this flag is set to false TorchScript will be using the legacy/original executor");

// 定义布尔类型的配置标志，用于控制是否禁用TorchScript图中的警告打印
C10_DEFINE_bool(
    torch_jit_disable_warning_prints,
    false,
    "Disables warning.warn prints in TorchScript graph");

// 定义布尔类型的配置标志，控制静态编译和动态编译的融合策略
C10_DEFINE_bool(
    torch_jit_static_then_dynamic,
    false,
    "fuse on two static compilations then 10 dynamic");

// 定义布尔类型的配置标志，控制总是动态编译的情况
C10_DEFINE_bool(
    torch_jit_always_dynamic,
    false,
    "fuse on 12 dynamic compilations");

// 定义布尔类型的配置标志，控制是否在优化后释放性能分析图的记录以减少内存使用
C10_DEFINE_bool(
    torch_jit_release_profiling_graph_after_optimization,
    false,
    "After getOptimizedPlanFor release the optimization record for reduction of memory in inference. This is aggressive memory saving, and please be cautious!");

// 定义整数类型的配置标志，控制在优化完成后等待释放性能分析图的延迟时间
C10_DEFINE_int32(
    torch_jit_release_profiling_graph_delay_in_seconds,
    60,
    "How long to wait before releasing the profiling graph after optimizaiton is done. Only used if torch_jit_release_profiling_graph_after_optimization is set to true.");

// 默认的性能分析运行次数
constexpr size_t kDefaultNumProfiledRuns = 1;

// 默认的回退深度
constexpr size_t kDefaultBailoutDepth = 20;
// 定义一个 int64 类型的命令行标志，用于指定 Torch JIT 的分析运行次数，默认为 kDefaultNumProfiledRuns
C10_DEFINE_int64(
    torch_jit_num_profiled_runs,
    kDefaultNumProfiledRuns,
    "Number of profiling runs");

// 定义一个 int64 类型的命令行标志，用于指定 Torch JIT 的退出深度，默认为 kDefaultBailoutDepth
C10_DEFINE_int64(
    torch_jit_bailout_depth,
    kDefaultBailoutDepth,
    "Number of re-specializations");

// Torch JIT 的命名空间
namespace torch::jit {

// 匿名命名空间，包含私有函数和变量
namespace {
// 获取当前时间的秒数
int32_t getNowInSecs() {
  auto currentTimePoint = std::chrono::system_clock::now();
  // 计算当前时间距离纪元的秒数
  auto durationSinceEpoch = std::chrono::duration_cast<std::chrono::seconds>(
      currentTimePoint.time_since_epoch());
  return static_cast<int32_t>(durationSinceEpoch.count());
}
} // namespace

// 如果定义了 C10_MOBILE 宏，则为真，否则为假
#if defined(C10_MOBILE)
// 执行器模式的原子布尔值，初始为 true
static std::atomic<bool> executor_mode{true};
// 分析模式的原子布尔值，初始为 false
static std::atomic<bool> profiling_mode{false};
#else
// 执行器模式的原子布尔值，初始为 true
static std::atomic<bool> executor_mode{true};
// 分析模式的原子布尔值，初始为 true
static std::atomic<bool> profiling_mode{true};
#endif

// 用于控制融合策略的互斥锁
static std::mutex fusion_strategy_lock;

// 获取初始的融合策略
static FusionStrategy getInitialStrategy() {
  if (FLAGS_torch_jit_always_dynamic) {
    // 如果指定了始终动态融合的标志，则返回动态融合策略
    return {{FusionBehavior::DYNAMIC, 12}};
  }
  // 否则返回混合的融合策略，静态和动态各占比例
  FusionStrategy mixed = {
      {FusionBehavior::STATIC, 2}, {FusionBehavior::DYNAMIC, 10}};
  if (FLAGS_torch_jit_static_then_dynamic) {
    return mixed;
  }
  // 根据平台定义返回静态融合策略
#ifdef FBCODE_CAFFE2
  return {{FusionBehavior::STATIC, 20}};
#endif
  return mixed;
}

// 延迟初始化的融合策略，以便可以加载 gflags 中的值
static std::optional<FusionStrategy> fusion_strategy = c10::nullopt;

// 获取当前的融合策略
FusionStrategy getFusionStrategy() {
  std::lock_guard<std::mutex> guard(fusion_strategy_lock);
  if (fusion_strategy == c10::nullopt) {
    fusion_strategy = getInitialStrategy();
  }
  return *fusion_strategy;
}

// 设置融合策略
FusionStrategy setFusionStrategy(FusionStrategy& strategy) {
  std::lock_guard<std::mutex> guard(fusion_strategy_lock);
  if (fusion_strategy == c10::nullopt) {
    fusion_strategy = getInitialStrategy();
  }
  // 获取旧的融合策略并更新为新的策略
  FusionStrategy old_strategy = *fusion_strategy;
  fusion_strategy = strategy;
  return old_strategy;
}

// 记录分析运行次数的原子变量，初始为 kDefaultNumProfiledRuns
static std::atomic<size_t> num_profiled_runs{kDefaultNumProfiledRuns};

// 返回分析模式的原子布尔值的引用
std::atomic<bool>& getProfilingMode() {
  return profiling_mode;
}

// 返回执行器模式的原子布尔值的引用
std::atomic<bool>& getExecutorMode() {
  return executor_mode;
}

// 返回分析运行次数的原子变量的引用
std::atomic<size_t>& getNumProfiledRuns() {
  // 从命令行标志初始化 num_profiled_runs
  static const size_t init = []() {
    return num_profiled_runs = FLAGS_torch_jit_num_profiled_runs;
  }();
  (void)init; // 防止编译器警告
  return num_profiled_runs;
}

// 获取退出深度
size_t getBailoutDepth() {
  // 从融合策略中获取退出深度，其值为所有融合行为的权重之和
  size_t depth = 0;
  for (const auto& pair : getFusionStrategy()) {
    depth += pair.second;
  }
  return depth;
}

// 在分析模式下是否需要梯度的判断函数
static bool needsGradientInProfilingMode(Block* b) {
  for (auto n : b->nodes()) {
    if (n->kind() == prim::BailOut) {
      auto ptt = n->output()->type()->expect<TensorType>();
      // 如果节点要求梯度且支持梯度，则返回 true
      if (ptt->requiresGrad() && *ptt->requiresGrad()) {
        return true;
      }
    }
    // 如果节点不是 BailOut，则继续检查下一个节点
  }
  // 所有节点都不要求梯度，则返回 false
  return false;
}
    # 检查节点 n 是否为 profile 类型
    if (n->kind() == prim::profile) {
      # 获取节点 n 的 profiled_type 属性，期望其为 TensorType 类型
      auto type = n->ty(attr::profiled_type)->expect<TensorType>();
      # 如果 profiled_type 要求梯度且需要梯度，则返回 true
      if (type->requiresGrad() && *type->requiresGrad()) {
        return true;
      }
    }

    # 遍历节点 n 的每个子块 ib
    for (auto ib : n->blocks()) {
      # 如果子块 ib 在 profiling 模式下需要梯度，则返回 true
      if (needsGradientInProfilingMode(ib)) {
        return true;
      }
    }
  }
  # 如果以上条件都不满足，则返回 false
  return false;
// `prim::RequiresGradCheck`保证输入张量的requires_grad属性将与profiled匹配，否则将触发回退路径。
// 这使我们能够在不需要梯度的输入中修剪反向图中的梯度。我们将requires_grad属性从输入传输到differentiable图上的输入。
// Autodiff将检查这些属性并修剪不需要的梯度
// `dnode->outputs()`中的`requires_grad`属性也将被传输
static C10_UNUSED void setRequiresGradOnDiffGraph(Node* dnode) {
  // 获取子图的输入
  auto gi = dnode->g(attr::Subgraph)->inputs();
  // 遍历节点的输入
  for (size_t i = 0; i < dnode->inputs().size(); i++) {
    // 如果输入是TensorType
    if (auto ty = dnode->input(i)->type()->cast<TensorType>()) {
      // 获取子图输入的TensorType
      auto gi_ty = gi[i]->type()->expect<TensorType>();
      // 设置子图输入的requires_grad属性
      gi[i]->setType(gi_ty->withRequiresGrad(ty->requires_grad()));
      GRAPH_DEBUG(
          "Setting ",
          *gi_ty->withRequiresGrad(ty->requires_grad()),
          " on ",
          gi[i],
          " ",
          gi[i]->debugName());
    }
  }

  // 我们还需要在子图中设置输出的requires_grad，以便autodiff可以设置df_input_vjps，DifferentiableGraphOp可以正确设置`requires_grad=`
  auto go = dnode->g(attr::Subgraph)->outputs();
  auto set_requires_grad = [](const TensorTypePtr& t, Value* val) -> bool {
    if (t && t->requiresGrad().has_value()) {
      GRAPH_DEBUG("setting type ", *t);
      val->setType(t);
      return true;
    }
    return false;
  };

  // 遍历子图的输出
  for (const auto i : c10::irange(go.size())) {
    auto ty = go[i]->type()->cast<TensorType>();
    if (ty) {
      auto n = go[i]->node();
      auto dno = dnode->outputs().at(i);
      // 遍历输出的使用
      for (auto dno_use : dno->uses()) {
        GRAPH_DEBUG("found user of ", i, " as ", *dno_use.user);
        if (n->kind() == prim::profile) {
          if (set_requires_grad(
                  n->ty(attr::profiled_type)->expect<TensorType>(), go[i])) {
            break;
          }
        } else if (dno_use.user->kind() == prim::profile) {
          if (set_requires_grad(
                  dno_use.user->ty(attr::profiled_type)->expect<TensorType>(),
                  go[i])) {
            break;
          }
        } else if (dno_use.user->kind() == prim::DifferentiableGraph) {
          Value* o =
              dno_use.user->g(attr::Subgraph)->inputs().at(dno_use.offset);
          // 是否安全不检查其他使用，因为我们在DifferentiableGraph内部？
          auto nn = o->uses().at(0).user;
          if (nn->kind() == prim::profile) {
            if (set_requires_grad(
                    nn->ty(attr::profiled_type)->expect<TensorType>(), go[i])) {
              break;
            }
          }
        }
      }
    }
  }
}
// 确保不同可微图的保护方法
static bool guardDifferentiableGraph(Node* dnode) {
  // 获取子图的输入
  auto gi = dnode->g(attr::Subgraph)->inputs();
  // 初始认为所有输入都已经处理过
  bool all_inputs_seen = true;
  // 遍历所有输入
  for (const auto i : c10::irange(gi.size())) {
    // 检查输入是否为张量类型
    auto ty = gi[i]->type()->cast<TensorType>();
    if (ty) {
      // 找到使用该输入的第一个节点
      auto n = gi[i]->uses().at(0).user;
      // 获取对应的输入节点
      auto dni = dnode->inputs().at(i);
      // 调试信息：找到了第一个使用者
      GRAPH_DEBUG("found first user of ", i, " as ", *n);
      if (n->kind() == prim::profile) {
        // 如果使用者是 profile 节点，设置输入类型
        GRAPH_DEBUG(
            "setting input ", i, " to type ", *n->ty(attr::profiled_type));
        dni->setType(n->ty(attr::profiled_type));
      } else if (dni->node()->kind() == prim::DifferentiableGraph) {
        // 如果输入节点是不同iableGraph，则处理特殊情况
        // 这里处理了在前面的不同iable图中吸收了 profile 节点的情况
        // 见 TestAutodiffSubgraphSlicing.test_does_not_create_cycles。
        // 替代方案可能是在自动微分之前专门化类型或者为自动微分输出复制 profile 节点，
        // 但应在创建子图时进行，并可能会产生混乱。
        // XXX TODO: 重新审视替代方案
        Value* o = dni->node()->g(attr::Subgraph)->outputs().at(dni->offset());
        if (o->node()->kind() == prim::profile) {
          dni->setType(o->node()->ty(attr::profiled_type));
        }
      }

      // 将 requires_grad 属性传播到输入
      // 在 insertTypeGuard 中会添加 RequiresGrad 检查，以确保输入的 requires_grad 保持一致；
      // 但其他属性不能保证一致
      auto requires_grad = dni->type()->expectRef<TensorType>().requiresGrad();
      gi[i]->setType(ty->withRequiresGrad(requires_grad));

      // 检查可选值是否已定义
      all_inputs_seen &= (dni->type()->cast<TensorType>() != TensorType::get());
    }
  }
  // 如果所有输入都已处理，则返回 true
  if (all_inputs_seen) {
    // 在此处添加一个 RequiresGrad 检查，如果存在 "alternating patterns" 的梯度，
    // 这可能会带来麻烦；一种替代方法是查看在 profiling 记录中单独看到的 requires_grad
    insertTypeGuard(
        dnode,
        [](const TensorTypePtr& t) {
          return TensorType::get()->withRequiresGrad(
              t->requiresGrad().value_or(true));
        },
        prim::RequiresGradCheck);
    return true;
  } else {
    // 作为回退，将不同iable图内联
    // 理想情况下，我们应该设置这个以进行重新 profiling
    UpdateDifferentiableGraphRequiresGrad(
        dnode->g(attr::Subgraph), c10::nullopt);
    SubgraphUtils::unmergeSubgraph(dnode);
    return false;
  }
}
// 执行无优化通道的函数，操作包括内联、梯度下降降低、移除扩展操作、规范化操作、消除死代码
void runNooptPassPipeline(std::shared_ptr<Graph>& graph) {
  // 输出调试信息：内联操作前图的状态
  GRAPH_DEBUG("Before Inliner (beginning of runNooptPassPipeline)\n", *graph);
  // 执行内联操作
  Inline(*graph);
  // 输出调试信息：内联操作后、梯度下降降低操作前图的状态
  GRAPH_DEBUG("After Inline, Before NoGrad\n", *graph);
  // 执行梯度下降降低操作
  LowerGradOf(*graph);
  // 输出调试信息：梯度下降降低操作后、移除扩展操作前图的状态
  GRAPH_DEBUG("After LowerGradOf, before RemoveExpands\n", *graph);
  // 执行移除扩展操作
  RemoveExpands(graph);
  // 输出调试信息：移除扩展操作后、规范化操作前图的状态
  GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
  // 执行规范化操作
  CanonicalizeOps(graph);
  // 输出调试信息：规范化操作后、消除死代码操作前图的状态
  GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
  // 执行消除死代码操作
  EliminateDeadCode(graph);
  // 输出调试信息：消除死代码操作后图的状态，并标记为无优化通道结束
  GRAPH_DEBUG(
      "After EliminateDeadCode (end of runNooptPassPipeline)\n", *graph);
}

// 执行自动微分前的预处理通道的函数，操作包括梯度下降降低、特化自动微分零、移除扩展操作、规范化操作、消除死代码、单点优化、常量传播、常量池、循环展开、列表突变移除、常量池、常量传播、消除公共子表达式、检查原地操作
static void runPreAutodiffPassPipeline(std::shared_ptr<Graph>& graph) {
  // 输出调试信息：插入守卫操作前图的状态
  GRAPH_DEBUG(
      "Before InsertGuards (beginning of runPreAutodiffPassPipeline)\n",
      *graph);

  // 执行梯度下降降低操作
  LowerGradOf(*graph);
  // 输出调试信息：梯度下降降低操作后、特化自动微分零操作前图的状态
  GRAPH_DEBUG("After LowerGradOf, before specializeAutogradZero\n", *graph);

  // 执行特化自动微分零操作
  specializeAutogradZero(graph);
  // 输出调试信息：特化自动微分零操作后图的状态
  GRAPH_DEBUG("After specializeAutogradZero\n", *graph);

  // 运行必需的通道操作
  {
    // 执行移除扩展操作
    RemoveExpands(graph);
    // 输出调试信息：移除扩展操作后、规范化操作前图的状态
    GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
    // 执行规范化操作
    CanonicalizeOps(graph);
    // 输出调试信息：规范化操作后、消除死代码操作前图的状态
    GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
    // 执行消除死代码操作
    EliminateDeadCode(graph);
    // 输出调试信息：消除死代码操作后图的状态
    GRAPH_DEBUG("After EliminateDeadCode", *graph);
  }

  // 执行单点优化操作
  PeepholeOptimize(graph);
  // 输出调试信息：单点优化操作后、常量传播操作前图的状态
  GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
  // 执行常量传播操作
  ConstantPropagation(graph);

  // 运行优化操作
  {
    // 再次执行消除死代码操作
    EliminateDeadCode(graph);
    // 输出调试信息：消除死代码操作后、消除公共子表达式操作前图的状态
    GRAPH_DEBUG(
        "After EliminateDeadCode, before EliminateCommonSubexpression\n",
        *graph);
    // 执行消除公共子表达式操作
    EliminateCommonSubexpression(graph);
    // 输出调试信息：消除公共子表达式操作后、单点优化操作前图的状态
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before PeepholeOptimize\n",
        *graph);

    // 再次执行单点优化操作
    PeepholeOptimize(graph);
    // 输出调试信息：单点优化操作后、常量传播操作前图的状态
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    // 执行常量传播操作
    ConstantPropagation(graph);
    // 输出调试信息：常量传播操作后、常量池操作前图的状态
    GRAPH_DEBUG("After ConstantPropagation, before ConstantPooling\n", *graph);
    // 执行常量池操作
    ConstantPooling(graph);
    // 输出调试信息：常量池操作后、循环展开操作前图的状态
    GRAPH_DEBUG("After ConstantPooling, before UnrollLoops\n", *graph);

    // 执行循环展开操作
    UnrollLoops(graph);
    // 输出调试信息：循环展开操作后、列表突变移除操作前图的状态
    GRAPH_DEBUG("After UnrollLoops, before RemoveListMutation\n", *graph);
    // 再次执行列表突变移除操作
    RemoveListMutation(graph);
    // 输出调试信息：列表突变移除操作后、单点优化操作前图的状态
    GRAPH_DEBUG("After RemoveListMutation, before PeepholeOptimize\n", *graph);
    // 再次执行单点优化操作
    PeepholeOptimize(graph);
    // 输出调试信息：单点优化操作后、常量传播操作前图的状态
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    // 再次执行常量传播操作
    ConstantPropagation(graph);
    // 输出调试信息：常量传播操作后、消除公共子表达式操作前图的状态
    GRAPH_DEBUG(
        "After ConstantPropagation, before EliminateCommonSubexpression\n",
        *graph);

    // 再次执行消除公共子表达式操作
    EliminateCommonSubexpression(graph);
    // 输出调试信息：消除公共子表达式操作后、检查原地操作前图的状态
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before CheckInplace\n", *graph);
    // 执行检查原地操作
    CheckInplace(graph);
  }

  // 输出调试信息：检查原地操作结束后的图状态，并标记为自动微分前的预处理通道结束
  GRAPH_DEBUG(
      "After CheckInplace (end of runPreAutodiffPassPipeline)\n", *graph);
}

// 获取当前行为的融合行为函数的实现
FusionBehavior ProfilingGraphExecutorImpl::getCurrentBehavior(
    size_t remaining_depth) {
  // 初始化当前深度为 0
  size_t curr_depth = 0;
  // 从最后一个策略开始向前遍历
  for (int i = static_cast<int>(fusion_strategy_.size()) - 1; i >= 0; i--) {
    // 将当前深度增加上当前策略的深度
    curr_depth += fusion_strategy_[i].second;
    // 如果剩余深度小于等于当前累积深度，则返回当前策略的类型
    if (remaining_depth <= curr_depth) {
      return fusion_strategy_[i].first;
    }
  }
  // 如果程序执行到这里，表示存在未处理的情况，输出警告信息
  TORCH_WARN("Strategy changed mid-invocation, NYI");
  // 默认返回静态融合行为
  return FusionBehavior::STATIC;
}
}

void ProfilingGraphExecutorImpl::runNoGradOptimizations(
    std::shared_ptr<Graph>& graph,
    size_t remaining_bailout_depth) {
  GRAPH_DEBUG(
      "After customPostPasses (beginning of runNoGradOptimizations)\n", *graph);
  // runNondiffOptimization
  {
    // 运行注册的不同后端可以注册的自定义优化 passes
    for (const auto& passPair : getCustomPrePasses()) {
      passPair.first(graph);
    }
    GRAPH_DEBUG("After customPrePasses, before LowerSimpleTuples\n", *graph);

    // 在此时可能仍然存在 TupleConstruct / TupleUnpack 对，需要移除以进行后续融合
    LowerSimpleTuples(graph);
    GRAPH_DEBUG("After LowerSimpleTuples\n", *graph);

    if (tensorExprFuserEnabled()) {
      // 移除 prim::profile 节点并将 profile 信息直接嵌入到值类型的 IR 中
      // 这些优化试图合并/融合图中的节点（如 BatchMM 和 GraphFuser）在存在间歇性 prim::profile 节点时效果较差。
      // 依赖类型信息的优化也负责插入适当的类型检查。
      // 优化完成后，将从 IR 中删除张量类型信息，以防意外被其他 passes 使用。
      RemoveProfileNodesAndSpecializeTypes(graph);
      GRAPH_DEBUG(
          "After RemoveProfileNodesAndSpecializeTypes, before BatchMM\n",
          *graph);
      // 将包含多个 MM 的子图重写为批量处理它们的表达式
      BatchMM(graph);
      GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);
      auto min_size = getFusionGroupInlining() ? 2 : 1;
      bool dyn_shapes = getCurrentBehavior(remaining_bailout_depth) ==
          FusionBehavior::DYNAMIC;
      FuseTensorExprs(graph, min_size, /* composed op*/ false, dyn_shapes);
      GRAPH_DEBUG("After Fusion, before customPostPasses\n", *graph);
    } else {
      // 将包含多个 MM 的子图重写为批量处理它们的表达式
      BatchMM(graph);
      GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);

      FuseGraph(graph, true);
      GRAPH_DEBUG("After Fusion, before customPostPasses\n", *graph);
    }

    // 运行自定义的融合后 passes
    for (const auto& passPair : getCustomPostPasses()) {
      passPair.first(graph);
    }
    GRAPH_DEBUG(
        "After customPostPasses, before RemoveTensorTypeSpecializations \n",
        *graph);
    // 移除张量类型的特殊化
    RemoveTensorTypeSpecializations(graph);
    GRAPH_DEBUG("After RemoveTensorTypeSpecializations\n", *graph);
  }
  GRAPH_DEBUG("End of runNoGradOptimizations\n");
}

void ProfilingGraphExecutorImpl::runProfilingOptimizations(
    std::shared_ptr<Graph>& copy,
    size_t remaining_bailout_depth) {
  GRAPH_DEBUG("Before runProfilingOptimizations:\n", *copy);
  if (!getGraphExecutorOptimize()) {
    runNooptPassPipeline(copy);
    return;
  }

  // 运行预自动微分通道流水线
  runPreAutodiffPassPipeline(copy);

  // 如果在分析模式下需要梯度，则创建自动微分子图
  if (needsGradientInProfilingMode(copy->block())) {
    // 创建自动微分子图
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    // 调试输出：创建自动微分子图后的图结构
    GRAPH_DEBUG("After CreateAutodiffSubgraphs\n", *copy);
    size_t idx = 0;
    // 遍历每个自动微分子图节点
    for (Node* dnode : diff_nodes) {
      // 调试输出：优化当前自动微分节点在图中的位置
      GRAPH_DEBUG("Optimizing diff node ", idx, " in ", *copy);
      // 如果无法保护不可微分图（因为输入缺乏分析信息），则重新内联子图并删除可微分节点
      if (!guardDifferentiableGraph(dnode)) {
        GRAPH_DEBUG("Could not guardDifferentiableGraph ", idx, " in ", *copy);
        idx++;
        continue;
      }
      // 调试输出：保护不可微分图后的图结构
      GRAPH_DEBUG("After guardDifferentiableGraph:\n", *copy);
      // 提取自动微分子图并进行微分
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      // 移除梯度函数中的特定张量类型的优化
      RemoveTensorTypeSpecializations(gradient.f);
      // 移除梯度函数块中的分析节点
      ProfilingRecord::removeProfilingNodes(gradient.f->block());
      // 调试输出：前向图和后向图的结构
      GRAPH_DEBUG("Forward graph:\n", *(gradient.f));
      GRAPH_DEBUG("Backward graph:\n", *(gradient.df));
      // 更新不同iable图需要梯度为false，类似于在torch.no_grad上下文中执行的操作
      UpdateDifferentiableGraphRequiresGrad(gradient.f, false);
      // 调试输出：更新不同iable图需要梯度后的图结构
      GRAPH_DEBUG("After UpdateDifferentiableGraphRequiresGrad ", *gradient.f);
      // 替换由TE Fuser插入的回退图
      replaceFallbackGraphWithFallbackFunction(gradient.f->block());
      // 打包梯度信息到当前节点
      packGradient(gradient, dnode);
      // 调试输出：完成优化当前自动微分节点后的信息
      GRAPH_DEBUG("Finished optimizing diff node ", idx++);
    }
    // 内联自动微分子图
    InlineAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    // 替换由TE Fuser插入的回退图
    replaceFallbackGraphWithFallbackFunction(copy->block());
    // 移除分析节点
    ProfilingRecord::removeProfilingNodes(copy->block());
    // 调试输出：内联自动微分子图后和移除分析节点后的图结构
    GRAPH_DEBUG(
        "After InlineAutodiffSubgraphs and Removing Profiling Nodes\n", *copy);
  } else {
    // 运行无梯度优化
    runNoGradOptimizations(copy, remaining_bailout_depth);
  }
  // 消除死代码
  EliminateDeadCode(copy);
  // 调试输出：运行分析优化后的图结构
  GRAPH_DEBUG("After runProfilingOptimizations:\n", *copy);
}

// 运行与性能分析无关的优化步骤
void ProfilingGraphExecutorImpl::runProfilingInsensitiveOptimizations(
    std::shared_ptr<Graph>& graph) {
  // 在内联之前打印调试信息，显示执行优化前的图结构
  GRAPH_DEBUG(
      "Before inlining (beginning of runProfilingInsensitiveOptimizations)\n",
      *graph);
  // 如果启用了图执行优化，进行内联操作
  if (getGraphExecutorOptimize()) {
    Inline(*graph);
  }
  // 内联后，在清除性能分析信息之前打印调试信息
  GRAPH_DEBUG("After inlining, before ClearProfilingInformation\n", *graph);
  // 清除性能分析信息
  ClearProfilingInformation(graph);
  // 清除性能分析信息后，在降低梯度之前打印调试信息
  GRAPH_DEBUG("After ClearProfilingInformation, before LowerGradOf\n", *graph);
  // 降低梯度操作
  LowerGradOf(*graph);
  // 降低梯度后，在清除未定义性之前打印调试信息
  GRAPH_DEBUG("After LowerGradOf, before ClearUndefinedness\n", *graph);
  // 清除任何残留的未定义性，因为双向梯度图的输入可能带有来自分析后的反向图的未定义性
  ClearUndefinedness(graph);
  // 运行必需的优化步骤
  {
    // 在清除未定义性后，但在移除展开操作之前打印调试信息
    GRAPH_DEBUG("After ClearUndefinedness, before RemoveExpands\n", *graph);
    // 移除展开操作
    RemoveExpands(graph);
    // 移除展开操作后，在规范化操作之前打印调试信息
    GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
    // 规范化操作
    CanonicalizeOps(graph);
    // 规范化操作后，在消除死代码之前打印调试信息
    GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
    // 消除死代码
    EliminateDeadCode(graph);
  }
  // 如果未启用图执行优化，直接在消除死代码后打印调试信息并返回
  if (!getGraphExecutorOptimize()) {
    GRAPH_DEBUG(
        "After EliminateDeadCode (end of runProfilingInsensitiveOptimizations)\n",
        *graph);
    return;
  }

  // 在消除死代码后，在分解操作之前打印调试信息
  GRAPH_DEBUG("After EliminateDeadCode, before DecomposeOps\n", *graph);
  // 分解操作
  DecomposeOps(graph);
  // 分解操作后，在常量传播之前打印调试信息
  GRAPH_DEBUG("After DecomposeOps, before ConstantPropagation\n", *graph);
  // 常量传播操作
  ConstantPropagation(graph);
  // 常量传播操作后，在消除死代码之前打印调试信息
  GRAPH_DEBUG("After ConstantPropagation, before EliminateDeadCode\n", *graph);
  // 再次消除死代码
  EliminateDeadCode(graph);
  // 在消除死代码后，在消除公共子表达式之前打印调试信息
  GRAPH_DEBUG(
      "After EliminateDeadCode, before EliminateCommonSubexpression\n", *graph);
  // 消除公共子表达式操作
  EliminateCommonSubexpression(graph);
  // 在消除公共子表达式后，在常量池之前打印调试信息
  GRAPH_DEBUG(
      "After EliminateCommonSubexpression, before ConstantPooling\n", *graph);
  // 常量池操作
  ConstantPooling(graph);
  // 在常量池操作后，在眼簧优化之前打印调试信息
  GRAPH_DEBUG("After ConstantPooling, before PeepholeOptimize\n", *graph);
  // 眼簧优化操作
  PeepholeOptimize(graph);
  // 眼簧优化操作后，在消除死代码之前打印调试信息
  GRAPH_DEBUG("After PeepholeOptimize, before EliminateDeadCode\n", *graph);
  // 最后一次消除死代码
  EliminateDeadCode(graph);
  // 在最后一次消除死代码后，在简化简单元组之前打印调试信息
  GRAPH_DEBUG("After EliminateDeadCode, before LowerSimpleTuples\n", *graph);
  // 简化简单元组操作
  LowerSimpleTuples(graph);
  // 简化简单元组操作后，在检查就地操作之前打印调试信息
  GRAPH_DEBUG("After LowerSimpleTuples, before CheckInplace\n", *graph);
  // 检查就地操作
  CheckInplace(graph);
  // 在检查就地操作后打印调试信息，标志着运行终止
  GRAPH_DEBUG(
      "After CheckInplace (end of runProfilingInsensitiveOptimizations)\n",
      *graph);
}

// 构造函数：基于图和函数名称构造性能分析图执行器实现
ProfilingGraphExecutorImpl::ProfilingGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : GraphExecutorImplBase(graph, std::move(function_name)) {
  // 获取融合策略并初始化
  fusion_strategy_ = getFusionStrategy();
}

// 获取实例化逃逸深度，基于融合策略的深度总和
size_t ProfilingGraphExecutorImpl::getInstantiatedBailoutDepth() {
  // 从融合策略的值对中计算逃逸深度总和
  size_t depth = 0;
  for (const auto& pair : fusion_strategy_) {
    depth += pair.second;
  }
  return depth;
}

// 获取给定栈的优化计划
const ExecutionPlan& ProfilingGraphExecutorImpl::getOptimizedPlanFor(
    Stack& stack,
    std::optional<size_t> remaining_bailout_depth) {
  GRAPH_DEBUG("Running ProfilingGraphExecutorImpl ", this);

  // 如果优化模式或者性能分析模式任一关闭，则执行无优化模式
  if (!getGraphExecutorOptimize() || !getProfilingMode()) {
    // 如果没有备用计划，则复制图形对象，并执行梯度降低操作
    if (!fallback_plan_) {
      auto copy = graph->copy();
      GRAPH_DEBUG(
          "Before LowerGradOf (beginning of runNooptPassPipeline)\n", *graph);
      LowerGradOf(*copy); // 执行梯度降低操作
      GRAPH_DEBUG("After LowerGradOf, before RemoveExpands\n", *graph);
      RemoveExpands(copy); // 移除扩展操作
      fallback_plan_ = ExecutionPlan(copy, function_name_); // 创建执行计划
      GRAPH_DUMP("NoOpt Graph: ", copy); // 输出无优化图形
    }
    return *fallback_plan_; // 返回备用计划
  }

  // 如果启用了TensorExpr融合器，第一次调用时需要持久化remaining_bailout_depth_
  // 以便正确更新在ProfilingGraphExecutorImpl中的回退函数
  else if (!remaining_bailout_depth_.has_value() || !tensorExprFuserEnabled()) {
    // 如果传入了remaining_bailout_depth值，则使用该值更新remaining_bailout_depth_
    if (remaining_bailout_depth.has_value()) {
      remaining_bailout_depth_ = *remaining_bailout_depth;
    } else {
      // 否则使用getInstantiatedBailoutDepth()的返回值更新remaining_bailout_depth_
      remaining_bailout_depth_ = getInstantiatedBailoutDepth();
    }
  }

  // 如果remaining_bailout_depth_为0，则执行简单执行器模式
  if (*remaining_bailout_depth_ == 0) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy); // 运行性能分析不敏感的优化
    GRAPH_DUMP("Optimized SimpleExecutor Graph: ", copy); // 输出优化后的简单执行器图形
    optimized_plan_ = ExecutionPlan(copy, function_name_); // 创建执行计划
    time_optimized_plan_created_ = getNowInSecs(); // 记录优化计划创建时间
    return *optimized_plan_; // 返回优化计划
  }

  bool profiling_record_created_in_this_call = false;
  // 如果尚未创建性能分析图形记录
  if (!pr_) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy); // 运行性能分析不敏感的优化
    pr_ = ProfilingRecord::instrumentGraph(copy); // 创建性能分析记录
    profiling_record_created_in_this_call = true;
    // `InsertProfileNodesForSpecializeAutogradZero` 用于特定Autograd零值的优化配置
    // 它需要在任何可能插入`prim::iprofile_value`节点之前运行，以保证正确性
    InsertProfileNodesForSpecializeAutogradZero(pr_.get());
    GRAPH_DUMP("Profiled Graph: ", pr_->graph()); // 输出性能分析后的图形
    profiling_plan_ = ExecutionPlan(pr_->graph(), function_name_); // 创建执行计划
    // fall-through
  }

  // 如果性能分析记录尚未准备好
  if (!pr_->ready()) {
  // 返回当前的 profiling_plan_ 对象
  return *profiling_plan_;
}

// 复制当前图形以进行优化操作
auto copy = pr_->graph()->copy();
// 移除副本中的 profile 计数器
ProfilingRecord::removeProfileCounter(copy->block());
// 运行优化操作，根据剩余的 bailout 深度
runProfilingOptimizations(copy, *remaining_bailout_depth_);
// 如果存在的话，替换由 specialize_autogradzero 插入的回退图形
replaceFallbackGraphWithFallbackFunction(copy->block());
// 运行最终的优化操作
runFinalOptimizations(copy);
// 对优化后的图形进行严格融合检查
CheckStrictFusion(copy);
// 输出优化后的图形信息，带有调试信息前缀
GRAPH_DUMP("Optimized Graph: ", copy);
// 根据优化后的图形和函数名创建执行计划
optimized_plan_ = ExecutionPlan(copy, function_name_);
// 记录优化计划创建的时间戳（秒数）
time_optimized_plan_created_ = getNowInSecs();
// 如果在当前调用中创建了 profiled 图形，则可以释放它
if (FLAGS_torch_jit_release_profiling_graph_after_optimization &&
    profiling_record_created_in_this_call) {
  // 清除图编译的中间图形数据
  clearTheGraphCompilationIntermediateGraphs();
}
// 返回优化后的执行计划对象
return *optimized_plan_;
}

const ExecutionPlan& ProfilingGraphExecutorImpl::getPlanFor(
    Stack& stack,
    std::optional<size_t> remaining_bailout_depth) {
  // 使用互斥锁保护并发访问
  std::lock_guard<std::mutex> lock(compile_mutex);

  // IMPORTANT: This is a hot path of calling a torchscript function. Try not to
  // add any code above this.
  // 如果已经优化过执行计划，则根据标志和延迟条件释放额外的内存
  if (optimized_plan_) {
    if (FLAGS_torch_jit_release_profiling_graph_after_optimization &&
        !is_graph_extra_memory_released_) {
      int32_t now = getNowInSecs();
      // 检查是否超过了延迟时间，超过则清除编译过程中的中间图
      if ((now - time_optimized_plan_created_) >
          FLAGS_torch_jit_release_profiling_graph_delay_in_seconds) {
        clearTheGraphCompilationIntermediateGraphs();
      }
    }
    // 返回优化后的执行计划
    return *optimized_plan_;
  }

  // 如果深度未设置，则获取优化后的执行计划
  return getOptimizedPlanFor(stack, remaining_bailout_depth);
}

GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  // 创建执行状态对象
  GraphExecutorState state;
  // 断言优化后的执行计划已经存在
  TORCH_INTERNAL_ASSERT(optimized_plan_);
  auto opt_plan = *optimized_plan_;
  // 将优化后的执行计划添加到执行计划状态中
  state.execution_plans.emplace(ArgumentSpec{0, 0}, opt_plan);
  return state;
}

static Node* insertFallbackFunctionCall(
    Graph* graph,
    GraphFunction* func,
    ArrayRef<Value*> inputs) {
  // 获取函数返回值类型
  auto tuple_type = func->graph()->return_node()->input(0)->type();
  // 插入常量节点代表函数调用
  Value* fn_constant = graph->insertNode(graph->create(prim::Constant))
                           ->s_(attr::name, func->name())
                           ->i_(Symbol::attr("fallback"), 1)
                           ->output()
                           ->setType(FunctionType::create(func));
  // 构建函数调用的输入参数列表
  std::vector<Value*> func_call_inputs = {fn_constant};
  func_call_inputs.insert(func_call_inputs.end(), inputs.begin(), inputs.end());
  // 插入函数调用节点并设置返回值类型
  Value* result =
      graph->insertNode(graph->create(prim::CallFunction, func_call_inputs))
          ->output()
          ->setType(tuple_type);

  // 插入元组解包节点并返回
  auto fun_unpack_tuple = graph->insertNode(graph->createTupleUnpack(result));
  return fun_unpack_tuple;
}

static GraphFunction* createFallbackPathFunction(
    Block* b,
    const std::string& function_name) {
  // 映射值函数，复制块到新图中
  auto value_map = [](Value* v) { return v; };
  auto graph = std::make_shared<Graph>();
  graph->block()->cloneFrom(b, value_map);

  // 收集返回节点的输出类型
  auto otypes = c10::fmap(
      graph->return_node()->inputs(), [](Value* v) { return v->type(); });
  // 创建返回值类型为元组的函数
  auto tuple_type = TupleType::create(otypes);
  auto return_tuple = graph->createTuple(graph->return_node()->inputs());
  graph->appendNode(return_tuple);
  // 清除原图的所有输出
  for (int i = static_cast<int>(graph->outputs().size()) - 1; i >= 0; i--) {
    graph->eraseOutput(i);
  }
  // 注册返回值为元组
  graph->registerOutput(return_tuple->output());
  // 创建新的图函数对象
  return new GraphFunction(function_name, graph, nullptr);
}

void ProfilingGraphExecutorImpl::replaceFallbackGraphWithFallbackFunction(
    Block* b) {
  // 创建堆栈对象
  Stack s;
  // 遍历块中的所有节点
  for (auto it = b->nodes().begin(); it != b->nodes().end();) {
    // 检查迭代器指向的节点是否为 prim::FallbackGraph 类型
    if (it->kind() == prim::FallbackGraph) {
      // 创建一个名为 "fallback_function" 的回退路径函数，基于子图的内容
      auto fallback_func = createFallbackPathFunction(
          it->g(attr::Subgraph)->block(), "fallback_function");
      // 内部断言，确保剩余回退深度大于零
      TORCH_INTERNAL_ASSERT(*remaining_bailout_depth_ > 0);
      // 输出调试信息，指示为哪个节点获取计划，以及剩余的回退深度
      GRAPH_DEBUG(
          "getPlanFor for", getHeader(*it), " ", *remaining_bailout_depth_);
      // 获取回退函数的执行器，并为当前状态获取其计划，减少剩余的回退深度
      fallback_func->get_executor().getPlanFor(
          s, *remaining_bailout_depth_ - 1);
      // 将创建的回退函数对象添加到回退函数列表中
      fallback_functions_.emplace_back(fallback_func);
      // 将插入点设置在当前迭代器指向的节点位置
      WithInsertPoint wip{*it};
      // 在当前节点的拥有图中插入调用回退函数的操作，使用当前节点的输入作为参数
      auto function_call = insertFallbackFunctionCall(
          b->owningGraph(), fallback_func, it->inputs());
      // 替换当前节点输出的使用，用回退函数调用的输出替换
      for (const auto i : c10::irange(function_call->outputs().size())) {
        it->output(i)->replaceAllUsesWith(function_call->output(i));
      }
      // 销毁当前迭代器指向的节点
      it.destroyCurrent();
    } else {
      // 如果当前节点不是 prim::FallbackGraph 类型，则遍历其包含的所有子块
      for (Block* ib : it->blocks()) {
        // 递归替换子块中的回退图为回退函数
        replaceFallbackGraphWithFallbackFunction(ib);
      }
      // 迭代器指向下一个节点
      it++;
    }
}

// 结束命名空间 torch::jit

void ProfilingGraphExecutorImpl::runFinalOptimizations(
    std::shared_ptr<Graph>& graph) {
  // 调用 AddIfThenElseOp 函数进行最终优化
  AddIfThenElseOp(graph);
}

void ProfilingGraphExecutorImpl::debugFlushCompilationCache() {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> lock(compile_mutex);
  // 重置所有智能指针，释放资源
  pr_.reset();
  fallback_plan_.reset();
  profiling_plan_.reset();
  optimized_plan_.reset();
  // 清空回退函数容器，防止内存泄漏
  fallback_functions_.clear();
  // 重置剩余的回退深度
  remaining_bailout_depth_.reset();
  // 获取融合策略并赋值给 fusion_strategy_
  fusion_strategy_ = getFusionStrategy();
  // 重置优化计划创建时间
  time_optimized_plan_created_ = 0;
  // 标记额外内存已释放
  is_graph_extra_memory_released_ = false;
}

void ProfilingGraphExecutorImpl::clearTheGraphCompilationIntermediateGraphs() {
  // 标记额外内存已释放
  is_graph_extra_memory_released_ = true;
  // 重置 profiling_plan_ 智能指针，释放资源
  profiling_plan_.reset();
  // 重置 fallback_plan_ 智能指针，释放资源
  fallback_plan_.reset();
  // 重置 graph 智能指针，释放资源
  graph.reset();
  // 重置 pr_ 智能指针，释放资源
  pr_.reset();
}

} // namespace torch::jit
```