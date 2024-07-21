# `.\pytorch\torch\_functorch\config.py`

```py
"""
Global flags for aot autograd
"""

import os  # 导入操作系统相关的模块
import sys  # 导入系统相关的模块
from typing import TYPE_CHECKING  # 导入类型检查相关的模块

# Converts torch rng ops to their functional philox rng equivalents. Note that
# we functionalize only CUDA rng ops today.
# 将 torch 的随机数操作转换为 functional philox 随机数等效操作。注意，目前只针对 CUDA 的随机数操作进行功能化。
functionalize_rng_ops = False

# can be useful for debugging if we are incorrectly creating meta fake tensors
# 如果我们错误地创建了元数据假张量，这个标志可能对调试很有用
fake_tensor_allow_meta = os.environ.get("FAKE_ALLOW_META", True)

# Enables optional asserts in hotpath code to check for errors.  If
# you are seeing weird accuracy problems, try turning this on.
# This is currently off by default as it will harm tracing time,
# but it is on by default for aot_eager.
# 在热路径代码中启用可选断言以检查错误。如果出现奇怪的精度问题，请尝试打开此选项。
# 默认情况下此选项关闭，因为它会影响跟踪时间，但对于 aot_eager 默认是开启的。
debug_assert = False

debug_partitioner = os.environ.get("AOT_PARTITIONER_DEBUG", False)

# Today, if you are in a situation where there is "false aliasing"
# (e.g. you have a bunch of model parameters that all alias the same underlying buffer),
# our checks for this situation are very slow if these inputs have dynamic shapes.
# This config is set to ensure that there aren't too many aliased inputs in this situation,
# so that we error loudly instead of compiling forever.
# Eventually, we should make these checks faster.
# For now, however, you can simply turn off dynamic shapes by marking your inputs static
# when you run into this situation.
# 当前，如果出现“虚假别名”的情况（例如，您有一堆模型参数都别名同一底层缓冲区），
# 我们对这种情况的检查速度非常慢，特别是这些输入具有动态形状时。
# 此配置用于确保在这种情况下不会有太多的别名输入，以便我们能够及时报错，而不是永远编译。
# 最终，我们应该加快这些检查的速度。
# 但目前，您可以通过将输入标记为静态来简单地关闭动态形状，以应对此情况。
_max_aliased_inputs_with_dynamic_shapes_enabled = 5

static_weight_shapes = True

# Applies CSE to the graph before partitioning
# 在分区之前对图应用公共子表达式消除（CSE）
cse = True

# When AOTAutograd regenerates aliased graph outputs,
# attempte to use functionalization's view-replay logic
# before falling back to the autograd engine's view replay or as_strided.
# This can have some perf implications
# (although for many models this will not matter).
# (1) If you have many view ops chained together, replaying all of them
#     at runtime can have more overhead compared to a single as_strided call
# (2) If you are doing training, AsStridedBackward is quite slow,
#     and the individual view op backward formulas will likely be faster.
# (3) Some backends like XLA do not support as_strided
# 当 AOTAutograd 重新生成别名图输出时，
# 尝试使用 functionalization 的视图重播逻辑，然后再回退到自动求导引擎的视图重播或者 as_strided。
# 这可能会对性能产生影响（尽管对于许多模型这并不重要）。
# (1) 如果有许多视图操作链接在一起，在运行时重播它们可能比单个 as_strided 调用更耗费资源。
# (2) 如果在进行训练，AsStridedBackward 是相当慢的，
#     而单个视图操作的反向公式可能会更快。
# (3) 一些后端如 XLA 不支持 as_strided
from torch._inductor.config import is_fbcode

# Temporary hack: disable this flag for internal
# (needed to fix an internal issue while avoiding bumping XLA pin)
# eventually: either default this config to false completely
# once XLA pin update works,
# or default config to true and fix relevant bugs
# 临时 hack：对内部禁用此标志
# （在避免升级 XLA 时修复内部问题所需）
# 最终：一旦 XLA 更新完成，要么完全将此配置默认为 false，
# 要么默认配置为 true 并修复相关 bug
view_replay_for_aliased_outputs = not is_fbcode()

# Restricts the amount of computation AOTAutograd can do.
# NB: We have essentially disabled this heuristic now. However, this is kept
# here for now in case it's useful. Setting it low can artificially reduce the
# amount of recomputation AOTAutograd performs, although not in any kind of
# principled way.
# 限制 AOTAutograd 可以执行的计算量。
# 注意：我们现在基本上禁用了这个启发式方法。不过，目前暂时保留它以备可能有用。
# 将其设置得较低可以在某种程度上减少 AOTAutograd 执行的重新计算量，虽然这不是一种原则性的方法。
max_dist_from_bw = 1000
# 禁止在后向传播中重新计算远离当前节点的节点
ban_recompute_used_far_apart = True

# 当操作链过长时，断开可融合操作的长链，以防止在反向传播中出现任意长的重新计算链。
ban_recompute_long_fusible_chains = True

# 禁止在后向传播中重新计算需要在非融合节点中材料化的节点。
ban_recompute_materialized_backward = True

# 根据白名单选择禁止基于白名单的节点重新计算。将其设置为 False 则使用黑名单，对如排序/池化等开销不便宜但又不是那么昂贵的操作有影响。
ban_recompute_not_in_allowlist = True

# 禁止重新计算减少操作。通常是一个好主意，因为减少操作的结果通常非常小，但在融合中重新计算可能很昂贵。
ban_recompute_reductions = True

# 阻止分区器保存视图（即总是重新计算它们）。通常是个好主意，因为视图是免费重新计算的。
recompute_views = False

# 默认情况下，分区器仅试图优化运行时（尽管应该比急切模式使用更少的内存）
# 此旋钮控制分区器为您进行这种权衡，选择比内存预算节省更少激活的最快选项。
# 具体来说，0.0 对应于应用激活检查点到整个编译区域的激活内存，1.0 对应于默认的运行时优化策略的激活内存。因此，0.4 将导致一个策略，与默认策略相比节省 40% 的激活。
# 它解决了一个 0-1 背包问题，找到在激活内存预算下保持下来所需的最小重新计算。
# 注意：这*不能*被视为......
activation_memory_budget = 1.0

# 这控制我们在决定重新计算最便宜的操作符时如何估计运行时。有三个选项："flops"：基于 torch.utils.flop_counter 提供的 FLOP 计数；"profile"：对每个操作符进行基准测试以得出运行时；"testing"：对所有操作符返回 1。
activation_memory_budget_runtime_estimator = "flops"

# 这控制用于 0-1 背包问题的求解器。默认情况下，我们使用量化的动态规划解决方案 ("dp")。其他方法包括贪婪 ("greedy") 和整数线性规划 ("ilp")（依赖于 scipy）。
activation_memory_budget_solver = "dp"

# 这会生成一个 PNG 可视化图，展示了在激活内存预算值从 0 到 1、间隔为 0.5 的情况下，预期运行时与激活内存的权衡。参考示例：
# https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
visualize_memory_budget_pareto = (
    os.environ.get("PARTITIONER_MEMORY_BUDGET_PARETO", "0") == "1"
)

# 将所有 ban_recompute 启发式方法设置为 False，除了 ban_recompute_reductions。
# 是否启用激进的重计算模式，通常会节省内存但牺牲一定的性能
aggressive_recomputation = False

# 是否允许 FakeTensor.data_ptr() 报错
# 注意，这个选项与 AOTAutograd 和 torch.compile 是独立的，但我们的策略
# 是在 torch.compile 运行时关闭它。
fake_tensor_allow_unsafe_data_ptr_access = True

# 是否取消输入/输出中的效果令牌，改为在跟踪图中插入 make_token/sink_token 调用来创建和
# 消耗令牌。注意，这意味着图不再是函数式的，可能会导致静默错误，除非后端知道如何处理令牌。
unlift_effect_tokens = False

# 是否在模式中保留真实张量的跟踪，同时进行真实的计算。
# 尽管看起来这消除了使用虚假张量的意义，但有两个明显的用例：
#
#   1. 当用户调用 item()/其他依赖于数据的操作时，如果我们传播真实张量，
#      我们能够确定真实值并继续进行。
#
#   2. 在测试时可能很有用，当您希望查看虚假张量和真实张量是否一致时。
#      （注意，目前已知在克隆真实张量方面存在不准确性，这对于这种情况可能需要进一步调整。）
#
# 注意，通常认为虚假张量的存储成本较低廉，因此我们倾向于将其保存的时间比真实张量更长。
# 因此，我们还支持显式释放与虚假张量关联的真实张量，此时我们将停止传播真实张量。
#
# 还有一件事：当您提供一个真实张量给 fakeify 时，我们会克隆它，这样我们可以安全地对其进行突变（如果需要）。
# 这将增加实时内存使用量。这可能通过使用 COW 进行优化。目前我们也未完全保留真实张量的自动求导元数据；
# 这没问题，因为 AOTAutograd 只会使用虚假张量来确定张量的叶子状态等。
fake_tensor_propagate_real_tensors = False

# 控制 draw_graph 使用的默认图形输出格式
# 支持的格式在这里定义：https://graphviz.org/docs/outputs/
torch_compile_graph_format = os.environ.get("TORCH_COMPILE_GRAPH_FORMAT", "svg")

# 是否启用自动求导缓存
enable_autograd_cache = os.environ.get("ENABLE_AOT_AUTOGRAD_CACHE", "0") == "1"

# 在 BypassAOTAutogradCache 时是否严格报错而不仅仅是警告
# 用于测试
strict_autograd_cache = False

# 如果是类型检查，从 torch.utils._config_typing 导入相关内容
# （注意：F401, F403 禁止未使用和禁止导入）
if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

# 安装配置模块，包括补丁、保存配置、无效配置检查等
from torch.utils._config_module import install_config_module
install_config_module(sys.modules[__name__])
```