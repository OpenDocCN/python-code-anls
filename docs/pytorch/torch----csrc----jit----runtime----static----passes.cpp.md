# `.\pytorch\torch\csrc\jit\runtime\static\passes.cpp`

```
#include <torch/csrc/jit/runtime/static/passes.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/static/ops.h>

// 定义一个标志位，控制是否启用融合操作
C10_DEFINE_bool(
    enable_clip_ranges_gather_fusions,
    true,
    "If on, static runtime or optimize_sparse_nn_model will fuse clip ranges gather ops.");

namespace torch::jit {

// 检查图中是否存在指定操作节点
bool graphHasOp(std::shared_ptr<Graph>& graph, const char* op_name) {
  DepthFirstGraphNodeIterator graph_it(graph);
  for (auto node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    const char* node_qual_string = node->kind().toQualString();
    if (strcmp(node_qual_string, op_name) == 0) {
      return true;
    }
  }
  return false;
}

// 检查 forward 方法中是否存在指定操作节点
bool forwardHasOp(
    const torch::jit::script::Module& module,
    const char* op_name) {
  using Method = ::torch::jit::Method;
  Method method = module.get_method("forward");
  auto graph = method.graph();
  return graphHasOp(graph, op_name);
}

// 匿名命名空间下的函数，用于替换特定模式的子图
C10_UNUSED
void ConcatAddMulReplaceNaNClip(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  // 定义多个模式字符串，用于匹配和替换特定的计算模式
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num(%y2, %f, %g, %h)
        %res = aten::clamp(%y3, %i, %j)
        return (%res))IR";
  std::string pattern2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num_(%y2, %f, %g, %h)
        %res = aten::clamp(%y3, %i, %j)
        return (%res))IR";
  std::string pattern3 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num_(%y2, %f, %g, %h)
        %res = aten::clamp_(%y3, %i, %j)
        return (%res))IR";
  std::string pattern4 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num(%y2, %f, %g, %h)
        %res = aten::clamp_(%y3, %i, %j)
        return (%res))IR";
  std::string fused_pattern = R"IR(
    // 待融合操作的模式定义将在此插入
  )IR";

  // 在这里执行模式匹配和替换逻辑，以实现操作的融合
}
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %res = fb::concat_add_mul_replacenan_clip(%c, %e, %a, %i, %j, %b)
        return (%res))IR";


  // 定义了一个名为 graph 的函数，接受十个参数 %a 到 %j，用于执行特定的图操作

  SubgraphRewriter fuse;
  // 创建了一个名为 fuse 的 SubgraphRewriter 对象，用于重写子图的操作

  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 向 fuse 对象注册重写模式 pattern，替换为 fused_pattern

  fuse.runOnGraph(graph);
  // 在给定的图 graph 上运行 fuse 对象，执行注册的重写模式

  fuse.RegisterRewritePattern(pattern2, fused_pattern);
  // 向 fuse 对象注册另一个重写模式 pattern2，替换为 fused_pattern

  fuse.runOnGraph(graph);
  // 在同一个图 graph 上再次运行 fuse 对象，执行第二个注册的重写模式

  fuse.RegisterRewritePattern(pattern3, fused_pattern);
  // 向 fuse 对象注册第三个重写模式 pattern3，替换为 fused_pattern

  fuse.runOnGraph(graph);
  // 在同一个图 graph 上再次运行 fuse 对象，执行第三个注册的重写模式

  fuse.RegisterRewritePattern(pattern4, fused_pattern);
  // 向 fuse 对象注册第四个重写模式 pattern4，替换为 fused_pattern

  fuse.runOnGraph(graph);
  // 在同一个图 graph 上再次运行 fuse 对象，执行第四个注册的重写模式
// 定义一个函数，用于将两个子图模式融合成一个，以提高执行效率
void CastedBatchOneHotLengths(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: 检查输入的限制条件；输出在其他地方未使用
  // 定义第一个子图模式，使用 FB 库的批次单热长度操作
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %y0 : Tensor = aten::to(%a, %b, %c, %c, %d)
        %y1 : Tensor = fb::batch_one_hot_lengths(%y0, %e, %f)
        %res : Tensor = aten::to(%y1, %g, %c, %c, %d)
        return (%res))IR";
  // 定义融合后的第一个子图模式，直接调用 FB 库的优化版本
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %res : Tensor = fb::casted_batch_one_hot_lengths(%a, %e, %f)
        return (%res))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);

  // 定义第二个子图模式，使用不同的输入参数调用 FB 库的批次单热长度操作
  std::string pattern2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %y0 : Tensor = aten::to(%a, %b, %c, %c)
        %y1 : Tensor = fb::batch_one_hot_lengths(%y0, %d, %e)
        %res : Tensor = aten::to(%y1, %f, %c, %c)
        return (%res))IR";
  // 定义融合后的第二个子图模式，调用 FB 库的优化版本
  std::string fused_pattern2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %res : Tensor = fb::casted_batch_one_hot_lengths(%a, %d, %e)
        return (%res))IR";
  fuse.RegisterRewritePattern(pattern2, fused_pattern2);
  fuse.runOnGraph(graph);
}

// 定义一个函数，将多个图模式融合成一个以提高执行效率
void ConcatBatchMatMulBatchGather(std::shared_ptr<torch::jit::Graph>& graph) {
  // 定义第一个图模式，执行批量矩阵乘法和批量索引选择操作
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %y0 : Tensor = aten::stack(%a, %b)
        %y1 : Tensor = aten::transpose(%y0, %b, %c)
        %y2 : Tensor = aten::bmm(%y0, %y1)
        %y3 : Tensor = aten::flatten(%y2, %d, %e)
        %res : Tensor = aten::index_select(%y3, %b, %f)
        return (%res))IR";
  // 定义融合后的第一个图模式，调用 FB 库的优化版本
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %res : Tensor = fb::concat_batch_matmul_batch_gather(%f, %a)
        return (%res))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);

  // 定义第二个图模式，执行广播堆叠、矩阵乘法和批量索引选择操作
  std::string pattern_broadcast = R"IR(
    graph(%a, %b, %c, %d, %e, %indices):
        %y0 : Tensor = fb::broadcast_stack(%a, %b)
        %y1 : Tensor = aten::transpose(%y0, %b, %c)
        %y2 : Tensor = aten::matmul(%y0, %y1)
        %y3 : Tensor = aten::flatten(%y2, %b, %e)
        %y4 : Tensor = aten::flatten(%y3, %d, %d)
        %res : Tensor = aten::index_select(%y4, %b, %indices)
        return (%res))IR";
  // 定义融合后的第二个图模式，调用 FB 库的优化版本
  std::string fused_pattern_broadcast = R"IR(
    graph(%a, %b, %c, %d, %e, %indices):
        %res : Tensor = fb::broadcast_concat_batch_matmul_batch_gather(%indices, %a)
        return (%res))IR";
  fuse.RegisterRewritePattern(pattern_broadcast, fused_pattern_broadcast);

  // 运行图模式融合操作
  fuse.runOnGraph(graph);
}
    # 定义一个名为 graph 的图形函数，接受参数 %a, %b, %c, %d, %indices
    graph(%a, %b, %c, %d, %indices):
        # 使用 fb::broadcast_stack 函数将 %a 和 %b 进行广播叠加，得到结果 %y0
        %y0 : Tensor = fb::broadcast_stack(%a, %b)
        # 使用 aten::transpose 函数对 %y0 进行转置操作，将维度 %b 和 %c 进行交换，得到结果 %y1
        %y1 : Tensor = aten::transpose(%y0, %b, %c)
        # 使用 aten::matmul 函数对 %y0 和 %y1 进行矩阵乘法运算，得到结果 %y2
        %y2 : Tensor = aten::matmul(%y0, %y1)
        # 使用 aten::flatten 函数对 %y2 进行扁平化操作，在维度 %b 和 %d 上进行扁平化，得到结果 %y3
        %y3 : Tensor = aten::flatten(%y2, %b, %d)
        # 使用 aten::index_select 函数根据给定的 %indices 在维度 %b 上选择 %y3 中的元素，得到结果 %res
        %res : Tensor = aten::index_select(%y3, %b, %indices)
        # 返回 %res 作为函数的结果
        return (%res)
    
    
    
    // 创建一个名为 fuse 的对象，用于注册重写模式
    std::string fused_pattern_broadcast2 = R"IR(
        graph(%a, %b, %c, %d, %indices):
            // 调用 fb::broadcast_concat_batch_matmul_batch_gather 函数，将 %indices 和 %a 作为参数，得到结果 %res
            %res : Tensor = fb::broadcast_concat_batch_matmul_batch_gather(%indices, %a)
            // 将 %res 作为函数的返回值
            return (%res)
        )IR";
    // 使用 RegisterRewritePattern 函数将 pattern_broadcast2 与 fused_pattern_broadcast2 绑定注册到 fuse 对象中
    fuse.RegisterRewritePattern(pattern_broadcast2, fused_pattern_broadcast2);
    // 运行 fuse 对象上的 runOnGraph 方法，对 graph 进行融合操作
    fuse.runOnGraph(graph);
}

// 将输入的图表作为参数，执行剪切范围到偏移量的融合操作
C10_UNUSED void ClipRangesGatherRangesLengthsToOffsets(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: 检查输入的限制条件；输出在其他地方未使用
  // 定义用于匹配的模式字符串
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d):
        %y0 : Tensor = fb::clip_ranges(%b, %c)
        %y1 : Tensor, %y2 : Tensor = fb::gather_ranges(%a, %y0)
        %y3 : Tensor = fb::lengths_to_offsets(%y2, %d)
        return (%y3, %y1))IR";
  // 定义融合后的模式字符串
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather_lengths_to_offsets(%a, %b, %c, %d)
        return (%y1, %y0))IR";
  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册重写模式
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在输入图表上运行子图重写器
  fuse.runOnGraph(graph);
}

// 将输入的图表作为参数，执行剪切范围到融合操作
C10_UNUSED void ClipRangesGather(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: 检查输入的限制条件；输出在其他地方未使用
  // 不包含长度到偏移量的融合
  // 定义用于匹配的模式字符串
  std::string pattern = R"IR(
    graph(%a, %b, %c):
        %y0 : Tensor = fb::clip_ranges(%b, %c)
        %y1 : Tensor, %y2 : Tensor = fb::gather_ranges(%a, %y0)
        return (%y2, %y1))IR";
  // 定义融合后的模式字符串
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather(%a, %b, %c)
        return (%y1, %y0))IR";
  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册重写模式
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在输入图表上运行子图重写器
  fuse.runOnGraph(graph);
}

// 将输入的图表作为参数，执行预计算多重移位用于 Sigrid 哈希的融合操作
C10_UNUSED void PrecomputeMultiplierShiftForSigridHash(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // 定义用于匹配的模式字符串
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e):
        %y0 : Tensor = fb::sigrid_hash(%a, %b, %c, %d, %e)
        return (%y0)
  )IR";
  // 定义融合后的模式字符串
  std::string split_pattern = R"IR(
    graph(%a, %b, %c, %d, %e):
        %y0 : Tensor = fb::sigrid_hash_compute_multipler_shift(%c)
        %y2 : Tensor = fb::sigrid_hash_precompute(%a, %b, %c, %y0, %d, %e)
        return (%y2)
  )IR";
  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册重写模式
  fuse.RegisterRewritePattern(pattern, split_pattern);
  // 在输入图表上运行子图重写器
  fuse.runOnGraph(graph);
}

// 将输入的图表作为参数，执行剪切范围到偏移量的融合操作
C10_UNUSED void ClipRangesToGatherToOffsets(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // 定义用于匹配的模式字符串
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %to0_in0, %to0_in1, %to0_in2):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather(%a, %b, %c)
        %y2 : Tensor = aten::to(%y1, %to0_in0, %to0_in1, %to0_in1, %to0_in2)
        %y3 : Tensor = fb::lengths_to_offsets(%y2, %d)
        return (%y3, %y0))IR";
  // 定义融合后的模式字符串
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %to0_in0, %to0_in1, %to0_in2):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather_to_offsets(%a, %b, %c, %d, %to0_in0)
        return (%y1, %y0))IR";
  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册重写模式
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在输入图表上运行子图重写器
  fuse.runOnGraph(graph);

  // 继续定义下一个模式字符串
  std::string pattern2 = R"IR(
    # 定义一个名为 `graph` 的函数，接受多个参数 `%a, %b, %c, %d, %to0_in0, %to0_in1`
    graph(%a, %b, %c, %d, %to0_in0, %to0_in1):
        # 调用 `fb::clip_ranges_gather` 函数，传入参数 `%a, %b, %c`，返回两个 Tensor `%y0, %y1`
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather(%a, %b, %c)
        # 将 `%y1` Tensor 转换为指定类型 `%to0_in0`，并按照 `%to0_in1` 的格式处理
        %y2 : Tensor = aten::to(%y1, %to0_in0, %to0_in1, %to0_in1)
        # 调用 `fb::lengths_to_offsets` 函数，将 `%y2` Tensor 转换为偏移量 Tensor `%y3`，使用 `%d` 参数
        %y3 : Tensor = fb::lengths_to_offsets(%y2, %d)
        # 返回两个 Tensor `%y3, %y0`
        return (%y3, %y0))IR";
  # 定义字符串 `fused_pattern2`，存储另一个函数图的 IR 表示
  std::string fused_pattern2 = R"IR(
    graph(%a, %b, %c, %d, %to0_in0, %to0_in1):
        # 调用 `fb::clip_ranges_gather_to_offsets` 函数，传入参数 `%a, %b, %c, %d, %to0_in0`，返回两个 Tensor `%y0, %y1`
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather_to_offsets(%a, %b, %c, %d, %to0_in0)
        # 返回两个 Tensor `%y1, %y0`
        return (%y1, %y0))IR";
  # 注册重写模式，将 `pattern2` 替换为 `fused_pattern2`
  fuse.RegisterRewritePattern(pattern2, fused_pattern2);
  # 在图 `graph` 上运行融合操作
  fuse.runOnGraph(graph);
C10_UNUSED void ToLengthsToOffsets(std::shared_ptr<torch::jit::Graph>& graph) {
  // 定义字符串模式，表示第一个重写模式
  std::string pattern = R"IR(
    graph(%a, %includelastoffset, %dtype, %nonblocking, %copy, %memoryformat):
        %y0 : Tensor = aten::to(%a, %dtype, %nonblocking, %copy, %memoryformat)
        %y1 : Tensor = fb::lengths_to_offsets(%y0, %includelastoffset)
        return (%y1))IR";
  // 定义融合后的字符串模式，替代第一个模式的重写结果
  std::string fused_pattern = R"IR(
    graph(%a, %includelastoffset, %dtype, %nonblocking, %copy, %memoryformat):
        %y0 : Tensor = fb::to_lengths_to_offsets(%a, %includelastoffset, %dtype)
        return (%y0))IR";
  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册第一个重写模式
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在给定图上运行子图重写器
  fuse.runOnGraph(graph);

  // 定义字符串模式，表示第二个重写模式
  std::string pattern2 = R"IR(
    graph(%a, %includelastoffset, %dtype, %nonblocking, %copy):
        %y0 : Tensor = aten::to(%a, %dtype, %nonblocking, %copy)
        %y1 : Tensor = fb::lengths_to_offsets(%y0, %includelastoffset)
        return (%y1))IR";
  // 定义融合后的字符串模式，替代第二个模式的重写结果
  std::string fused_pattern2 = R"IR(
    graph(%a, %includelastoffset, %dtype, %nonblocking, %copy):
        %y0 : Tensor = fb::to_lengths_to_offsets(%a, %includelastoffset, %dtype)
        return (%y0))IR";
  // 创建另一个子图重写器对象
  SubgraphRewriter fuse;
  // 注册第二个重写模式
  fuse.RegisterRewritePattern(pattern2, fused_pattern2);
  // 在给定图上运行子图重写器
  fuse.runOnGraph(graph);
}

C10_UNUSED
void ClipRangesGatherSigridHash(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: 检查输入的限制；输出在其他地方未使用
  // 定义字符串模式，表示重写模式
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather_lengths_to_offsets(%a, %b, %c, %d)
        %y2 : Tensor = fb::sigrid_hash_precompute(%y0, %e, %f, %g, %h)
        return (%y2, %y1))IR";
  // 定义融合后的字符串模式，替代重写模式的结果
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h):
        %off : Tensor, %out : Tensor = fb::clip_ranges_gather_sigrid_hash_precompute_offsets(%b, %a, %c, %e, %f, %g, %h, %d)
        return (%out, %off))IR";
  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册重写模式
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在给定图上运行子图重写器
  fuse.runOnGraph(graph);
}

C10_UNUSED void ClipRangesGatherRangesSigridHash(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // 定义字符串模式，表示重写模式
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %y0 : Tensor = fb::clip_ranges(%b, %c)
        %y1 : Tensor, %y2 : Tensor = fb::gather_ranges(%a, %y0)
        %y3 : Tensor = fb::sigrid_hash_precompute(%y1, %d, %e, %f, %g)
        return (%y3, %y2))IR";
  // 定义融合后的字符串模式，替代重写模式的结果
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %off : Tensor, %out : Tensor = fb::clip_ranges_gather_sigrid_hash_precompute_v3(%b, %a, %c, %d, %e, %f, %g)
        return (%out, %off))IR";

  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册重写模式
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在给定图上运行子图重写器
  fuse.runOnGraph(graph);
}

C10_UNUSED void ClipRangesGatherRangesX2SigridHashPrecompute(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // Placeholder is a dummy op used to capture the first subgraph
  // 定义字符串模式，表示重写模式，用于捕获第一个子图
  std::string pattern = R"IR(
    // 定义一个名为 `graph` 的函数，接受多个输入参数
    graph(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32):
        // 调用 fb::clip_ranges 函数，对输入的范围进行裁剪，并将结果赋给 %clipped 张量
        %clipped : Tensor = fb::clip_ranges(%ranges, %max_length)
        // 调用 fb::gather_ranges 函数，根据裁剪后的范围从 %values 中收集数据，并将结果分别赋给 %output 和 %unused 张量
        %output : Tensor, %unused : Tensor = fb::gather_ranges(%values, %clipped)
        // 调用 fb::sigrid_hash_precompute 函数，对 %output 数据进行预处理，生成 %sigrid_hash_out 张量
        %sigrid_hash_out : Tensor = fb::sigrid_hash_precompute(%output, %salt, %max_value, %mul_shift, %hash_into_int32)
        // 返回两个张量 %sigrid_hash_out 和 %clipped
        return (%sigrid_hash_out, %clipped))IR";
    
    // 定义一个包含模板的字符串 fused_pattern，用于表示另一个图模式，与上述 graph 函数类似
    std::string fused_pattern = R"IR(
      graph(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32):
          // 使用 fb::placeholder 函数创建 %sigrid_hash_out 和 %clipped 张量
          %sigrid_hash_out : Tensor, %clipped : Tensor = fb::placeholder(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32)
          // 返回 %sigrid_hash_out 和 %clipped 张量
          return (%sigrid_hash_out, %clipped))IR";
    
    // 定义一个包含模板的字符串 pattern2，描述另一种图模式
    std::string pattern2 = R"IR(
      graph(%gather2_values, %ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32):
          // 使用 fb::placeholder 函数创建 %sigrid_hash_out 和 %clipped 张量
          %sigrid_hash_out : Tensor, %clipped : Tensor = fb::placeholder(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32)
          // 使用 fb::gather_ranges 函数，根据裁剪后的范围从 %gather2_values 中收集数据，并将结果分别赋给 %unused 和 %lengths 张量
          %unused : Tensor, %lengths : Tensor = fb::gather_ranges(%gather2_values, %clipped)
          // 返回 %lengths 和 %sigrid_hash_out 张量
          return (%lengths, %sigrid_hash_out))IR";
    
    // 定义一个包含模板的字符串 fused_pattern2，表示第二种图模式的融合结果
    std::string fused_pattern2 = R"IR(
      graph(%gather2_values, %ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32):
          // 使用 fb::clip_ranges_gather_sigrid_hash_precompute_v3 函数进行裁剪、收集和预处理操作，生成 %lengths 和 %sigrid_hash_out 张量
          %lengths : Tensor, %sigrid_hash_out : Tensor = fb::clip_ranges_gather_sigrid_hash_precompute_v3(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32)
          // 返回 %lengths 和 %sigrid_hash_out 张量
          return (%lengths, %sigrid_hash_out))IR";
    
    // 创建一个 SubgraphRewriter 对象 fuse，用于注册和执行图模式的重写操作
    SubgraphRewriter fuse;
    // 将 pattern 与 fused_pattern 注册到 fuse 中，以便进行图模式的重写
    fuse.RegisterRewritePattern(pattern, fused_pattern);
    // 在输入的图 graph 上执行注册的重写操作
    fuse.runOnGraph(graph);
    
    // 将 pattern2 与 fused_pattern2 注册到 fuse 中，以便进行第二种图模式的重写
    fuse.RegisterRewritePattern(pattern2, fused_pattern2);
    // 再次在输入的图 graph 上执行注册的重写操作
    fuse.runOnGraph(graph);
    
    // 反转第一步中融合的操作，将 fused_pattern 转换回 pattern
    fuse.RegisterRewritePattern(fused_pattern, pattern);
    // 再次在输入的图 graph 上执行注册的重写操作
    fuse.runOnGraph(graph);
namespace {

C10_UNUSED void SplitOutPrecomputeOpsForSparseNN(
    std::shared_ptr<torch::jit::Graph>& graph) {
#ifdef FBCODE_CAFFE2
  // 在稀疏神经网络图中分离出预计算操作
  PrecomputeMultiplierShiftForSigridHash(graph);
  // 对图执行常量传播优化
  ConstantPropagation(graph);
  // 在图中执行常量池化优化
  ConstantPooling(graph);
#endif
}

} // namespace

void FuseInferenceOpsForSparseNN(std::shared_ptr<torch::jit::Graph>& graph) {
#ifdef FBCODE_CAFFE2
  // 分离出稀疏神经网络推断操作的预计算步骤
  SplitOutPrecomputeOpsForSparseNN(graph);

  // 融合推断操作的各个子操作
  ConcatAddMulReplaceNaNClip(graph);
  CastedBatchOneHotLengths(graph);
  ConcatBatchMatMulBatchGather(graph);

  if (FLAGS_enable_clip_ranges_gather_fusions) {
    // 如果启用了裁剪范围和聚集范围融合，则执行相关优化
    ClipRangesGatherRangesLengthsToOffsets(graph);
  }
  // 对裁剪范围和聚集范围执行Sigrid Hash优化
  ClipRangesGatherSigridHash(graph);
  // 对裁剪范围和聚集范围执行Sigrid Hash优化
  ClipRangesGatherRangesSigridHash(graph);

  // 对裁剪范围和聚集范围执行2倍Sigrid Hash预计算
  ClipRangesGatherRangesX2SigridHashPrecompute(graph);

  if (FLAGS_enable_clip_ranges_gather_fusions) {
    // 优先执行裁剪范围和聚集融合，而不是仅执行裁剪范围和聚集
    ClipRangesGather(graph);

    // 将裁剪范围转换为聚集到偏移量
    ClipRangesToGatherToOffsets(graph);
  }

  // 将长度转换为偏移量
  ToLengthsToOffsets(graph);
#endif
}

}

void FuseSignLog1P(std::shared_ptr<torch::jit::Graph>& graph) {
  // 定义原始模式用于符号和Log1P的融合操作
  std::string pattern = R"IR(
    graph(%input):
        %0 : Tensor = aten::sign(%input)
        %1 : Tensor = aten::abs(%input)
        %2 : Tensor = aten::log1p(%1)
        %res : Tensor = aten::mul(%0, %2)
        return (%res)
  )IR";

  // 定义融合后的模式用于符号和Log1P的融合操作
  std::string fused_pattern = R"IR(
    graph(%input):
        %res : Tensor = static_runtime::signed_log1p(%input)
        return (%res)
    )IR";

  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册重写模式
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在图上运行子图重写器
  fuse.runOnGraph(graph);
}

namespace {

using TupleUnpackBlock = std::vector<Node*>;

// 收集可变元组解包融合候选项
std::vector<TupleUnpackBlock> CollectVariadicTupleUnpackFusionCandidates(
    const std::shared_ptr<Graph>& graph) {
  std::vector<TupleUnpackBlock> candidates;
  auto nodes = graph->nodes();
  std::vector<Node*> block;
  for (Node* cur_node : nodes) {
    if (cur_node->kind() == prim::TupleUnpack) {
      block.push_back(cur_node);
      continue;
    }
    if (block.size() > 1) {
      candidates.emplace_back(std::move(block));
    }
    block.clear();
  }
  // 检查最后一个块应为空
  TORCH_CHECK(block.empty());
  return candidates;
}

// 执行元组解包块的融合
void FuseTupleUnpackBlock(const TupleUnpackBlock& nodes) {
  TORCH_CHECK(!nodes.empty());
  auto graph = nodes[0]->owningGraph();
  // 创建静态运行时的变量元组解包节点
  auto var_unpack = graph->create(
      fromQualString("static_runtime::VarTupleUnpack"),
      /* num_outputs */ 0);
  var_unpack->insertAfter(nodes[nodes.size() - 1]);
  for (Node* node : nodes) {
    TORCH_CHECK(
        node->kind() == prim::TupleUnpack && node->inputs().size() == 1);
    var_unpack->addInput(node->input());

    for (Value* output : node->outputs()) {
      auto new_output = var_unpack->addOutput();
      new_output->copyMetadata(output);
      output->replaceAllUsesWith(new_output);
    }
    node->destroy();
  }
}

} // namespace

// 使用可变元组解包融合
void UseVariadicTupleUnpack(const std::shared_ptr<Graph>& graph) {
  // 对每个收集到的融合候选项执行操作
  for (auto& c : CollectVariadicTupleUnpackFusionCandidates(graph)) {
    # 调用函数 FuseTupleUnpackBlock，并传入参数 c
    FuseTupleUnpackBlock(c);
  }
// 定义一个宏，用于创建 c10::Symbol 到 c10::Symbol 映射的便捷方式
#define OP_PAIR(first, second) \
  { fromQualString(first), fromQualString(second) }

// 对于不能在内存规划中参与的 ops 的 out 变种，如果其输出与输入存在别名关系。
// 对于那些直接返回其输入或者对其进行复制（尤其是 aten::to）的 ops，
// 我们采用以下策略，而不是直接将它们制作成 out 变种，以便它们仍然可以参与内存规划。
// 假设 `a` 是 op 的输入 Tensor。
//
// 1) 将 `a`（以及其他操作符的输入）传递给特殊的 `static_runtime::$OP_maybe_copy_out` 变种。
//    此 op 返回一个正常的输出 Tensor（称为 `b_out`），以及一个 `did_copy` 标志，
//    表示是否应使用输出。如果 `did_copy` 为 false，则 `b_out` 的值是未指定的。
//    注意，此操作符是一个普通的 out 变种，非常适合内存规划。
//
// 2) 将 `a`、`b_out` 和 `did_copy` 传递给特殊的 `static_runtime::select_tensor` op，
//    如果 `did_copy` 为 true，则返回 `b_out`，否则返回 `a`。
//    注意，此操作符不需要参与内存规划，因为其输出总是与其输入之一存在别名关系。
//
// 下面是一个示意图：
//
//                        |
// |----------------------+ a
// |                      v
// |    +------------------------------------+
// |    |                                    |
// |    | static_runtime::$OP_maybe_copy_out |
// |    |                                    |
// |    +------------------+--------+--------+
// |                       |        |
// +--------------+        | b_out  | did_copy
//                | a      |        |
//                v        v        v
//      +------------------------------------+
//      |                                    |
//      |    static_runtime::select_tensor   |
//      |                                    |
//      +------------------+-----------------+
//                         |
//                         |
//                         | 要么是 a，要么是 b_out
//                         |
//                         v

// 用于替换图中某些操作的函数，将操作符可能的复制替代方式引入到图中
void ReplaceWithMaybeCopy(
    std::shared_ptr<Graph>& graph,
    bool outputs_are_immutable) {
  // 创建一个用于分析图结构的别名数据库
  AliasDb db(graph);
  // 为具有多个重载的操作匹配函数模式
  static const std::array<std::pair<c10::FunctionSchema, c10::Symbol>, 3> supported_schema =
      {{{torch::schema(
             "aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
         fromQualString("static_runtime::to_maybe_copy_out")},
        {torch::schema(
             "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"),
         fromQualString("static_runtime::to_maybe_copy_out")},
        {torch::schema(
             "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"),
         fromQualString("static_runtime::to_maybe_copy_out")}}};

  // 匹配节点的函数模式，并返回匹配的符号
  auto match_schema = [](const Node* node, c10::Symbol& out_matched_symbol) {
    for (auto& schema : supported_schema) {
      if (node->matches(schema.first)) {
        out_matched_symbol = schema.second;
        return true;
      }
    }
    return false;
  };

  // 用于存储替换操作的旧节点、新节点和选择张量节点的向量
  std::vector<std::tuple<Node*, Node*, Node*>> replacement;
  // 深度优先遍历图结构的节点迭代器
  DepthFirstGraphNodeIterator graph_it(graph);
  // 迭代遍历每个节点
  for (auto n = graph_it.next(); n != nullptr; n = graph_it.next()) {
    c10::Symbol new_symbol;
    // 如果节点不匹配任何支持的模式，则继续下一个节点
    if (!match_schema(n, new_symbol)) {
      continue;
    }
    // 检查节点输出是否只有一个
    TORCH_CHECK(n->outputs().size() == 1);

    // 如果节点有输入写入器，则跳过此节点
    if (db.hasInputWriters(n)) {
      continue;
    }

    auto* out = n->output();
    // 如果输出不是不可变的并且可能包含别名，则跳过此节点
    if (!outputs_are_immutable && db.mayContainAlias(out, graph->outputs())) {
      continue;
    }

    // 创建新节点并添加 did_copy 标志到输出
    auto* new_node = graph->create(new_symbol, n->outputs().size() + 1);
    for (auto* input : n->inputs()) {
      new_node->addInput(input);
    }
    new_node->outputs().at(1)->setType(c10::BoolType::get());

    // 创建选择张量节点并连接新节点的输出
    static const auto select_tensor_symbol =
        fromQualString("static_runtime::select_tensor");
    auto* select_tensor_node = graph->create(select_tensor_symbol, 1);
    TORCH_DCHECK_EQ(new_node->outputs().size(), 2);
    select_tensor_node->addInput(n->input(0));
    for (auto* output : new_node->outputs()) {
      select_tensor_node->addInput(output);
    }
    replacement.emplace_back(n, new_node, select_tensor_node);
  }

  // 遍历替换操作，将新节点和选择张量节点插入到旧节点之前，并复制元数据
  for (const auto& tup : replacement) {
    auto* const old_node = std::get<0>(tup);
    auto* const new_node = std::get<1>(tup);
    auto* const select_tensor_node = std::get<2>(tup);

    new_node->insertBefore(old_node);
    select_tensor_node->insertBefore(old_node);
    new_node->outputs()[0]->copyMetadata(old_node->output());
    select_tensor_node->output()->copyMetadata(old_node->output());
    old_node->replaceAllUsesWith(select_tensor_node);
    old_node->destroy();
  }
#ifndef NDEBUG
  // 如果处于调试模式，执行图的静态分析
  graph->lint();
  // 创建一个图的别名数据库，用于分析图的别名关系
  AliasDb db2(graph);
  // 执行 JIT 引擎的静态分析，传入别名数据库
  torch::jit::Lint(&db2);
#endif
}

static void ReplaceWithCopyImpl(
    std::shared_ptr<Graph>& graph,
    const c10::FastMap<c10::Symbol, c10::Symbol>& supported,
    const std::vector<std::pair<c10::FunctionSchema, c10::Symbol>>&
        supported_schema,
    const std::function<bool(Node*)>& f_extra_checks,
    bool outputs_are_immutable) {
  // 创建图的别名数据库，用于分析图中节点的别名关系
  AliasDb db(graph);

  // 匹配节点的函数，用于确定节点是否匹配给定的支持模式，并返回对应的符号
  auto match_schema = [&supported_schema](
                          const Node* node, c10::Symbol& out_matched_symbol) {
    for (auto& schema : supported_schema) {
      if (node->matches(schema.first)) {
        out_matched_symbol = schema.second;
        return true;
      }
    }
    return false;
  };

  // 存储需要替换的节点对
  std::vector<std::pair<Node*, Node*>> replacement;
  // 创建深度优先遍历的图节点迭代器
  DepthFirstGraphNodeIterator graph_it(graph);
  // 遍历图中的每个节点
  for (auto n = graph_it.next(); n != nullptr; n = graph_it.next()) {
    c10::Symbol new_symbol;
    // 如果节点的操作在支持的操作列表中，并且其操作已经注册
    if (supported.count(n->kind()) && opIsRegistered(supported.at(n->kind()))) {
      new_symbol = supported.at(n->kind());
    } else if (!match_schema(n, new_symbol)) {
      continue;
    }
    // 检查节点输出的数量是否为1
    TORCH_CHECK(n->outputs().size() == 1);

    // 如果节点的输入有写入者（即存在可能更新输入的节点），则不进行替换
    if (db.hasInputWriters(n)) {
      continue;
    }

    auto* out = n->output();
    // 如果节点的输出可能与图的输出存在别名，则不进行替换
    if (!outputs_are_immutable && db.mayContainAlias(out, graph->outputs())) {
      continue;
    }
    // 额外的自定义检查，如果不通过，则不进行替换
    if (!f_extra_checks(n)) {
      continue;
    }
    // 创建新的节点，用指定的符号和输出数量
    auto* new_node = graph->create(new_symbol, n->outputs().size());
    // 将原节点的输入复制到新节点中
    for (auto* input : n->inputs()) {
      new_node->addInput(input);
    }
    // 将原节点和新节点组成替换对，准备后续替换操作
    replacement.emplace_back(n, new_node);
  }

  // 对替换对进行实际替换操作
  for (const auto& p : replacement) {
    auto* old_node = p.first;
    auto* new_node = p.second;
    // 将新节点插入到旧节点之前
    new_node->insertBefore(old_node);
    // 复制新节点的元数据信息到旧节点的输出
    new_node->output()->copyMetadata(old_node->output());
    // 用新节点替换所有使用旧节点输出的节点
    old_node->replaceAllUsesWith(new_node);
    // 销毁旧节点
    old_node->destroy();
  }

#ifndef NDEBUG
  // 如果处于调试模式，再次执行图的静态分析
  graph->lint();
  // 创建另一个图的别名数据库，用于第二次分析
  AliasDb db2(graph);
  // 执行 JIT 引擎的静态分析，传入第二个别名数据库
  torch::jit::Lint(&db2);
#endif
}

// 只有在 ReplaceWithCopy 关闭时，才将 aten::permute 替换为其复制版本，当它后面紧跟着 reshape/flatten 时生效。
// 使用 ReplacePermuteWithCopy 函数替换图中的 aten::permute 节点为 static_runtime::permute_copy，如果输出不可变
void ReplacePermuteWithCopy(
    std::shared_ptr<Graph>& graph,
    bool outputs_are_immutable) {
  // 创建图的别名数据库
  AliasDb db(graph);
  // 定义支持替换的操作映射表
  const c10::FastMap<c10::Symbol, c10::Symbol> supported = {
#ifdef FBCODE_CAFFE2
      OP_PAIR("aten::permute", "static_runtime::permute_copy"),
#endif
  };
  // 额外的检查函数，检查节点的输出是否可以替换为复制操作
  auto f_extra_checks = [](Node* n) {
    // 获取节点的输出值和其下一个使用节点
    Value* out = n->output();
    Node* next_node = out->uses()[0].user;
    // 如果下一个节点的类型既不是 aten::reshape 也不是 aten::flatten，则返回 true
    if (next_node->kind() != aten::reshape ||
        next_node->kind() != aten::flatten) {
      return true;
    }
    // 否则返回 false
    return false;
  };
  // 调用替换函数，将符合条件的节点替换为复制操作
  ReplaceWithCopyImpl(
      graph, supported, {}, f_extra_checks, outputs_are_immutable);
}

// 使用 ReplaceWithCopyImpl 函数替换图中的指定节点为相应的复制操作，支持多种操作替换
void ReplaceWithCopy(
    std::shared_ptr<Graph>& graph,
    bool outputs_are_immutable) {
  // 创建图的别名数据库
  AliasDb db(graph);
  // 定义支持替换的操作映射表和函数原型映射表
  const c10::FastMap<c10::Symbol, c10::Symbol> supported = {
#ifdef FBCODE_CAFFE2
      OP_PAIR("aten::permute", "static_runtime::permute_copy"),
      OP_PAIR("fb::expand_dims", "static_runtime::expand_dims_copy"),
#endif
      OP_PAIR("aten::narrow", "aten::narrow_copy"),
      OP_PAIR("aten::reshape", "static_runtime::reshape_copy"),
      OP_PAIR("aten::flatten", "static_runtime::flatten_copy")};

  // 定义支持替换的函数原型和替换后的操作映射表
  static const std::vector<std::pair<c10::FunctionSchema, c10::Symbol>>
      supported_schema = {
          {{torch::schema("aten::dequantize.self(Tensor self) -> Tensor"),
            fromQualString("static_runtime::dequantize_copy")}}};

  // 调用替换函数，将符合条件的节点替换为复制操作
  ReplaceWithCopyImpl(
      graph,
      supported,
      supported_schema,
      [](Node* n) { return true; },
      outputs_are_immutable);
}

// 从图中消除 fb::equally_split 节点及其相关的 prim::ListUnpack 节点
void EliminateTrivialEquallySplit(std::shared_ptr<torch::jit::Graph>& graph) {
  // 获取 fb::equally_split 符号
  const auto equally_split = fromQualString("fb::equally_split");
  // 存储待移除的节点列表
  std::vector<Node*> to_remove;
  // 使用深度优先遍历迭代器遍历图中的每个节点
  DepthFirstGraphNodeIterator graph_it(graph);
  for (auto node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    // 如果当前节点不是 fb::equally_split，继续下一个节点
    if (node->kind() != equally_split) {
      continue;
    }

    // 获取节点的输出值
    const Value* value_out = node->outputs()[0];
    // 如果输出值有多于一个使用节点，继续下一个节点
    if (value_out->uses().size() != 1) {
      continue;
    }

    // 获取与输出值关联的 prim::ListUnpack 节点
    Node* list_unpack_node = value_out->uses()[0].user;
    // 如果关联节点不是 prim::ListUnpack，继续下一个节点
    if (list_unpack_node->kind() != prim::ListUnpack) {
      continue;
    }

    // 替换 prim::ListUnpack 节点的输出为 fb::equally_split 节点的输入
    list_unpack_node->output()->replaceAllUsesWith(node->input(0));
    // 将 prim::ListUnpack 节点和 fb::equally_split 节点加入待移除列表
    to_remove.push_back(list_unpack_node);
    to_remove.push_back(node);
  }

  // 遍历待移除节点列表，销毁这些节点
  for (Node* node : to_remove) {
    node->destroy();
  }
}

// 私有命名空间，用于检查节点是否不应进行特殊处理
bool shouldNotFuseListUnpackSpecialCase(const Node* node) {
  // 静态存储的符号数组，包含不应特殊处理的节点类型
  const static std::array<c10::Symbol, 3> sigrid_transforms_symbols{
      c10::Symbol::fromQualString("fb::variadic_sigrid_transforms_torch_bind"),
      c10::Symbol::fromQualString("fb::sigrid_transforms_torch_bind"),
      c10::Symbol::fromQualString("fb::sigrid_transforms")};

  // 如果节点的类型不在上述数组中，则返回 true，表示不应特殊处理
  if (std::find(
          sigrid_transforms_symbols.begin(),
          sigrid_transforms_symbols.end(),
          node->kind()) == sigrid_transforms_symbols.end()) {
    // 返回 false，表示函数执行失败
    return false;
  }

  // 为了与 Sigrid 转换融合，我们必须能够静态确定 `instance` 和 `use_offsets`，
  // 这两者共同帮助我们静态确定输出的类型。理由是：在没有静态类型信息的情况下，
  // 编写融合的 Sigrid 转换非常麻烦，而且每个模型中这两个参数都是静态已知的。
  // 如果尝试在没有静态类型信息的情况下融合输出，因为如果其中一个输出没有被管理，
  // 每次迭代都需要重置为正确类型的空张量。因此，如果我们不能提前收集类型信息，
  // 就必须在第一次迭代时惰性地执行，这可能会在时间和内存上浪费 - 每个线程可能会
  // 有自己的输出类型集，或者我们需要锁定以防止数据竞争。
  const auto num_inputs = node->inputs().size();
  // 如果输入的第一个元素无法转换为 IValue，或者最后一个元素无法转换为 IValue，
  // 则返回 true；否则返回 false。
  return !toIValue(node->input(0)).has_value() ||
      !toIValue(node->input(num_inputs - 1)).has_value();
} // namespace

// 定义函数 FuseListUnpack，用于将图中的特定节点替换为新的节点
void FuseListUnpack(std::shared_ptr<torch::jit::Graph>& graph) {
  // 定义映射 unfused_to_fused，将特定操作替换为新操作
  const c10::FastMap<c10::Symbol, c10::Symbol> unfused_to_fused = {
      OP_PAIR(
          "torcharrow::inference_wrapper_run_flat",
          "static_runtime::fused_inference_wrapper_run_flat"),
      OP_PAIR(
          "torcharrow::variadic_inference_wrapper_run_flat",
          "static_runtime::fused_variadic_inference_wrapper_run_flat"),
      OP_PAIR("fb::equally_split", "static_runtime::fused_equally_split"),
      OP_PAIR(
          "fb::sigrid_transforms", "static_runtime::fused_sigrid_transforms"),
      OP_PAIR(
          "static_runtime::variadic_grouped_accessor_op_v2",
          "static_runtime::fused_variadic_grouped_accessor_op_v2"),
      OP_PAIR(
          "fb::sigrid_transforms_torch_bind",
          "static_runtime::fused_sigrid_transforms_torch_bind"),
      OP_PAIR(
          "fb::variadic_sigrid_transforms_torch_bind",
          "static_runtime::fused_variadic_sigrid_transforms_torch_bind"),
      OP_PAIR(
          "fb::gather_ranges_to_dense",
          "static_runtime::fused_gather_ranges_to_dense"),
      OP_PAIR(
          "fb::gather_ranges_to_dense_v2",
          "static_runtime::fused_gather_ranges_to_dense_v2"),
      OP_PAIR(
          "fb::split_and_squeeze",
          "static_runtime::fused_split_and_squeeze_copy")};

  // 创建替换列表
  std::vector<std::tuple<Node*, Node*, Node*>> replacement;
  // 遍历图中的节点
  DepthFirstGraphNodeIterator graph_it(graph);
  for (auto node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    // 查找节点是否在映射中
    auto unfused_to_fused_it = unfused_to_fused.find(node->kind());
    if (unfused_to_fused_it == unfused_to_fused.end()) {
      continue;
    }

    // 获取节点的输出值
    const Value* value_out = node->outputs()[0];
    if (value_out->uses().size() != 1) {
      continue;
    }

    // 获取使用节点的列表解包节点
    Node* list_unpack_node = value_out->uses()[0].user;
    if (list_unpack_node->kind() != prim::ListUnpack) {
      continue;
    }

    // 获取列表解包节点的输出
    auto list_unpack_outputs = list_unpack_node->outputs();
    if (list_unpack_outputs.empty()) {
      continue;
    }

    // 检查是否应跳过特殊情况
    if (shouldNotFuseListUnpackSpecialCase(node)) {
      continue;
    }

    // 获取新操作的符号
    const auto& new_sym = unfused_to_fused_it->second;
    auto* new_node = graph->create(new_sym, 0);

    // 将节点的输入添加到新节点中
    for (Value* in : node->inputs()) {
      new_node->addInput(in);
    }

    // 将列表解包节点的输出替换为新节点的输出
    for (Value* out : list_unpack_outputs) {
      Value* new_out = new_node->addOutput();
      new_out->copyMetadata(out);
      out->replaceAllUsesWith(new_out);
    }
    // 将替换的节点信息添加到替换列表中
    replacement.emplace_back(node, new_node, list_unpack_node);
  }

  // 对替换列表中的节点进行替换操作
  for (const auto& nodes : replacement) {
    auto* old_node = std::get<0>(nodes);
    auto* new_node = std::get<1>(nodes);
    auto* list_unpack_node = std::get<2>(nodes);

    // 在旧节点后插入新节点，销毁列表解包节点和旧节点
    new_node->insertAfter(old_node);
    list_unpack_node->destroy();
    old_node->destroy();
  }
} // namespace jit

// 定义函数 RemoveImmutableInputDictLookups，用于移除不可变输入字典的查找操作
    // 对给定的 TorchScript 图执行字典优化，将不可变字典中常量键的获取操作替换为静态运行时的字典解包操作
    
    std::shared_ptr<torch::jit::Graph>& graph) {
      // 获取图中所有节点
      auto nodes = graph->nodes();
      // 创建别名分析数据库，用于分析节点之间的数据流关系
      AliasDb db(graph);
      
      // 存储字典到获取元素操作节点的映射关系
      std::unordered_map<Value*, std::vector<Node*>> dict_to_getitems;
      // 存储所有常量键的节点集合
      std::unordered_set<Node*> keys;
      
      // 遍历图中的每个节点
      for (Node* node : nodes) {
        // 如果节点不是 aten::__getitem__ 操作，跳过
        if (node->kind() != aten::__getitem__) {
          continue;
        }
        Node* getitem_node = node;
        // 获取获取元素操作的字典输入
        Value* dict = getitem_node->input(0);
        
        // 如果字典有写入操作，视为可变字典，跳过优化
        if (db.hasWriters(dict)) {
          continue;
        }
        
        // 如果字典类型不是字典类型或者字典节点不是图的参数节点，跳过
        if (dict->type()->kind() != TypeKind::DictType ||
            dict->node() != graph->param_node()) {
          continue;
        }
        
        // 确保获取元素操作有且仅有两个输入
        DCHECK(getitem_node->inputs().size() == 2);
        
        // 获取获取元素操作的键
        Node* key = getitem_node->input(1)->node();
        
        // 如果键不是常量类型，跳过
        if (key->kind() != prim::Constant) {
          continue;
        }
        
        // 将常量键节点加入到键集合中
        keys.insert(key);
        
        // 查找当前字典是否已经存在于映射关系中
        auto iter = dict_to_getitems.find(dict);
        if (iter == dict_to_getitems.end()) {
          // 如果不存在，则添加新的映射关系
          dict_to_getitems.emplace(dict, std::vector<Node*>{getitem_node});
          continue;
        }
        
        // 如果存在，则将当前获取元素操作节点添加到对应的列表中
        iter->second.push_back(getitem_node);
      }
      
      // 如果没有找到任何常量键，直接返回
      if (keys.empty()) {
        return;
      }
      
      // 在图的开头插入一个标记节点，并将所有常量键移动到该标记节点之前
      auto* marker = graph->create(prim::Constant);
      graph->prependNode(marker);
      graph->setInsertPoint(marker);
      for (Node* key : keys) {
        DCHECK(key->inputs().empty());
        key->moveBefore(marker);
      }
      
      // 创建静态运行时的字典解包操作，并替换相应的获取元素操作
      const c10::Symbol static_runtime_dict_unpack_symbol =
          fromQualString("static_runtime::dict_unpack");
      for (auto& it : dict_to_getitems) {
        Value* dict = it.first;
        std::vector<Node*>& getitems = it.second;
        DCHECK(!getitems.empty());
        auto* dict_unpack =
            graph->create(static_runtime_dict_unpack_symbol, getitems.size());
        graph->insertNode(dict_unpack);
        dict_unpack->addInput(getitems[0]->input(0));
        for (size_t i = 0; i < getitems.size(); ++i) {
          Node* getitem_node = getitems[i];
          DCHECK(getitem_node->input(0) == dict);
          dict_unpack->addInput(getitem_node->input(1));
          dict_unpack->output(i)->copyMetadata(getitem_node->output());
          getitem_node->output(0)->replaceAllUsesWith(dict_unpack->output(i));
          getitem_node->destroy();
        }
      }
      
      // 将插入点移回到图的起始块，并销毁标记节点
      graph->setInsertPoint(graph->block());
      marker->destroy();
    }
}

// 定义一个函数 UseVariadicGroupedAccessor，接受一个 shared_ptr 指向 Graph 对象作为参数
void UseVariadicGroupedAccessor(const std::shared_ptr<Graph>& graph) {
  // 调用 UseVariadicOp 函数，处理 grouped_accessor::grouped_accessor_op_v2 操作符
  UseVariadicOp(
      graph,
      fromQualString("grouped_accessor::grouped_accessor_op_v2"),
      fromQualString("static_runtime::variadic_grouped_accessor_op_v2"));
  // 调用 UseVariadicOp 函数，处理 fb::grouped_accessor_op_async 操作符
  UseVariadicOp(
      graph,
      fromQualString("fb::grouped_accessor_op_async"),
      fromQualString("static_runtime::variadic_grouped_accessor_op_async"));
}

// 匿名命名空间，定义一个帮助函数 CreateOwnedRefsForSpecialValuesHelper
namespace {

// 定义一个递归函数 CreateOwnedRefsForSpecialValuesHelper，接受 Graph 和 Block 指针作为参数
void CreateOwnedRefsForSpecialValuesHelper(Graph& graph, Block* block) {
  // 遍历当前 block 中的每个 node
  for (auto* node : block->nodes()) {
    // 遍历当前 node 中的每个子 block
    for (auto* sub_block : node->blocks()) {
      // 递归调用 CreateOwnedRefsForSpecialValuesHelper 处理子 block
      CreateOwnedRefsForSpecialValuesHelper(graph, sub_block);
    }
  }

  // 获取 block 的输出值
  auto outputs = block->outputs();
  // 创建一个快速查找输入值的集合
  c10::FastSet<Value*> inputs = {
      block->inputs().begin(), block->inputs().end()};

  // 遍历 block 的输出值
  for (const auto i : c10::irange(outputs.size())) {
    auto* output = outputs[i];

    // 如果输出值类型为 NoneType，则跳过
    if (output->type()->kind() == c10::TypeKind::NoneType) {
      continue;
    }

    // 如果输出值是输入值的一部分，或者可以转换为 IValue，或者不属于当前 block，则创建一个 owned ref
    if ((inputs.find(output) != inputs.end()) || toIValue(output).has_value() ||
        output->node()->owningBlock() != block) {
      // 创建一个创建 owned ref 的节点
      auto* create_owned_ref_node =
          graph.create(fromQualString("static_runtime::create_owned_ref"));
      // 将当前输出值作为输入
      create_owned_ref_node->addInput(output);
      // 复制输出值的元数据到新的输出
      create_owned_ref_node->output()->copyMetadata(output);

      // 将创建的节点附加到 block 中
      block->appendNode(create_owned_ref_node);
      // 替换原始输出值为新的创建的输出
      block->replaceOutput(i, create_owned_ref_node->output());
    }
  }
}

// 定义一个递归函数 ForceNonEmptyOutputsHelper，接受一个 None 值指针和 Block 指针作为参数
void ForceNonEmptyOutputsHelper(Value* none_value, Block* block) {
  // 遍历 block 中的每个 node
  for (auto* node : block->nodes()) {
    bool needs_output = false;
    // 遍历当前 node 中的每个子 block
    for (auto* sub_block : node->blocks()) {
      // 如果子 block 的输出为空，则注册一个 None 值作为输出，并标记需要输出
      if (sub_block->outputs().empty()) {
        sub_block->registerOutput(none_value);
        needs_output = true;
      }
      // 递归调用 ForceNonEmptyOutputsHelper 处理子 block
      ForceNonEmptyOutputsHelper(none_value, sub_block);
    }

    // 如果需要输出，则为当前 node 添加一个输出值
    if (needs_output) {
      DCHECK(node->kind() == prim::If);
      auto* output = node->addOutput();
      output->setType(c10::NoneType::get());
    }
  }
}

// 定义一个函数 findOrCreateNoneConstant，接受 Graph 指针作为参数
Node* findOrCreateNoneConstant(Graph& graph) {
  // 只在顶层 block 中搜索节点
  for (auto* node : graph.nodes()) {
    // 如果节点类型为 prim::Constant
    if (node->kind() != prim::Constant) {
      continue;
    }
    // 尝试获取节点输出的 IValue 值
    const auto ival_opt = toIValue(node->output());
    // 确保 IValue 有值，并且是 None 类型
    DCHECK(ival_opt.has_value());
    if (ival_opt->isNone()) {
      return node;
    }
  }

  // 如果找不到 None 类型的常量节点，则创建一个新的
  auto* none_node = graph.create(prim::Constant);
  none_node->output()->setType(c10::NoneType::get());
  // 将新节点插入到图的开头
  graph.prependNode(none_node);
  return none_node;
}

} // namespace

// 定义一个函数 CreateOwnedRefsForSpecialValues，接受 Graph 指针作为参数
void CreateOwnedRefsForSpecialValues(Graph& graph) {
  // 调用 CreateOwnedRefsForSpecialValuesHelper 函数处理 graph 的顶层 block
  CreateOwnedRefsForSpecialValuesHelper(graph, graph.block());
}
// 强制确保图中的输出非空
void ForceNonEmptyOutputs(Graph& graph) {
  // 查找或创建一个表示空值的常量节点
  auto* none_node = findOrCreateNoneConstant(graph);
  // 递归处理该常量节点的输出，以确保所有输出非空
  ForceNonEmptyOutputsHelper(none_node->output(), graph.block());
  // 如果该常量节点没有被使用，则销毁它
  if (!none_node->hasUses()) {
    none_node->destroy();
  }
}

namespace {

// 检查节点输入是否为预期的常量列表
bool inputIsConstantList(
    Node* node,
    size_t input_idx,
    const c10::List<int64_t>& expected) {
  // 尝试将节点输入转换为对应的值
  auto input_opt = toIValue(node->input(input_idx));
  // 如果转换失败或者转换后的值不是整数列表，则返回false
  if (!input_opt.has_value() || !input_opt->isIntList()) {
    return false;
  }
  // 检查转换后的整数列表是否与预期值相同
  return input_opt->toIntList() == expected;
}

// 检查节点输入是否为预期的整数常量
bool inputIsConstantInt(Node* node, size_t input_idx, int64_t expected) {
  // 尝试将节点输入转换为对应的值
  auto input_opt = toIValue(node->input(input_idx));
  // 如果转换失败或者转换后的值不是整数，则返回false
  if (!input_opt.has_value() || !input_opt->isInt()) {
    return false;
  }
  // 检查转换后的整数值是否与预期值相同
  return input_opt->toInt() == expected;
}

// 消除特定模式下的permute操作和sum操作的模式
void eliminatePermuteOpsSumPattern(std::shared_ptr<Graph>& graph) {
  // 子图重写器不能在常量上进行模式匹配，因此使用额外的过滤器确保`dim`参数的值正确
  auto dims_are_valid_constants =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        // 获取模板图中节点在实际图中的对应节点
        const auto& node_map = match.nodes_map;
        auto* sum_node = node_map.at(vmap.at("c")->node());
        auto* permute_node = node_map.at(vmap.at("b")->node());
        // 检查sum节点的第二个输入是否为[-1]的常量列表
        // 检查permute节点的第二个输入是否为[0, 2, 1]的常量列表
        return inputIsConstantList(sum_node, 1, c10::List<int64_t>{-1}) &&
            inputIsConstantList(permute_node, 1, c10::List<int64_t>{0, 2, 1});
      };

  // 定义要匹配的模式和替换后的模式
  const auto pattern = R"IR(
    graph(%a, %sum_dim, %permute_dim, %keepdim, %dtype):
        %b = aten::permute(%a, %permute_dim)
        %c = aten::sum(%b, %sum_dim, %keepdim, %dtype)
        return (%c))IR";

  const auto fused_pattern = R"IR(
    graph(%a, %sum_dim, %permute_dim, %keepdim, %dtype):
        %new_sum_dim: int[] = prim::Constant[value=[1]]()
        %d = aten::sum(%a, %new_sum_dim, %keepdim, %dtype)
        return (%d))IR";

  // 创建子图重写器实例并注册要重写的模式
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在图上运行子图重写器，并使用前面定义的过滤器进行匹配
  fuse.runOnGraph(graph, dims_are_valid_constants);
}

// 消除特定模式下的permute操作和softmax操作的模式
void eliminatePermuteOpsSoftmaxPattern(std::shared_ptr<Graph>& graph) {
  // 定义要匹配的模式
  const auto pattern = R"IR(
    graph(%a, %permute_dim_1, %permute_dim_2, %softmax_dim, %softmax_dtype):
        %b = aten::permute(%a, %permute_dim_1)
        %c = aten::softmax(%b, %softmax_dim, %softmax_dtype)
        %d = aten::permute(%c, %permute_dim_2)
        return (%d)
  )IR";

  // 定义替换后的模式（未完全提供）
  const auto fused_pattern = R"IR(
    // 定义一个名为 graph 的函数，接受五个参数：%a、%permute_dim_1、%permute_dim_2、%softmax_dim、%softmax_dtype
    graph(%a, %permute_dim_1, %permute_dim_2, %softmax_dim, %softmax_dtype):
        // 创建一个常量节点 %new_softmax_dim，其值为整数 1
        %new_softmax_dim: int = prim::Constant[value=1]()
        // 调用 PyTorch 的 softmax 操作，对输入 %a 在维度 %new_softmax_dim 上进行 softmax 计算，结果保存到 %e 中
        %e = aten::softmax(%a, %new_softmax_dim, %softmax_dtype)
        // 返回计算得到的结果 %e
        return (%e)
    )IR";
    
    // 定义一个 lambda 函数 dims_are_valid_constants，用于验证 permute_dim 是否为 (0, 2, 1)，softmax_dim 是否为 2
    auto dims_are_valid_constants =
        [](const Match& match,
           const std::unordered_map<std::string, Value*>& vmap) {
        // 获取匹配结果中的节点映射
        const auto& node_map = match.nodes_map;
        // 获取 vmap 中的节点，分别对应 permute_dim_1、permute_dim_2、softmax_dim
        auto* permute_node_1 = node_map.at(vmap.at("b")->node());
        auto* permute_node_2 = node_map.at(vmap.at("d")->node());
        auto* softmax_node = node_map.at(vmap.at("c")->node());
        // 检查 softmax_node 是否为常量且值为 2，同时检查 permute_node_1 和 permute_node_2 是否为常量列表 [0, 2, 1]
        return inputIsConstantInt(softmax_node, 1, 2) &&
            inputIsConstantList(
                   permute_node_1, 1, c10::List<int64_t>{0, 2, 1}) &&
            inputIsConstantList(permute_node_2, 1, c10::List<int64_t>{0, 2, 1});
    };
    
    // 创建 SubgraphRewriter 对象 fuse
    SubgraphRewriter fuse;
    // 注册重写模式，将 pattern 模式重写为 fused_pattern
    fuse.RegisterRewritePattern(pattern, fused_pattern);
    // 在输入的图 graph 上运行 SubgraphRewriter fuse，使用 dims_are_valid_constants 函数验证条件
    fuse.runOnGraph(graph, dims_are_valid_constants);
}

} // namespace

// 移除图中的多余 Permute 操作符
void EliminateExtraPermuteOps(std::shared_ptr<Graph>& graph) {
  // 调用函数消除 Permute 操作符与 Sum 模式
  eliminatePermuteOpsSumPattern(graph);
  // 调用函数消除 Permute 操作符与 Softmax 模式
  eliminatePermuteOpsSoftmaxPattern(graph);
}

namespace {

// 根据给定的值和类型标识符，返回可能的用户节点
Node* maybeUserWithKind(Value* value, c10::Symbol kind) {
  auto& uses = value->uses();
  // 如果使用此值的节点数不为 1，则返回空指针
  if (uses.size() != 1) {
    return nullptr;
  }
  auto* user = uses[0].user;
  // 如果使用此值的节点类型不匹配给定的类型标识符，则返回空指针
  if (user->kind() != kind) {
    return nullptr;
  }
  return user;
}

} // namespace

// 使用 Split 和 Squeeze 优化图中的操作
void UseSplitAndSqueeze(std::shared_ptr<Graph>& graph) {
  // 用于存储需要删除的节点列表
  std::vector<Node*> to_erase;
  // 遍历图中的每个节点
  for (auto* node : graph->nodes()) {
    // 如果节点类型不是 aten::split，则继续下一个节点
    if (node->kind() != aten::split) {
      continue;
    }
    // 获取第三个输入的轴信息
    auto axis_opt = toIValue(node->input(2));
    // 如果无法获取轴信息，则继续下一个节点
    if (!axis_opt) {
      continue;
    }
    auto axis = *axis_opt;
    auto* split_node_output = node->output();
    // 获取可能的 ListUnpack 用户节点
    auto* list_unpack_node =
        maybeUserWithKind(split_node_output, prim::ListUnpack);
    // 如果找不到 ListUnpack 用户节点，则继续下一个节点
    if (list_unpack_node == nullptr) {
      continue;
    }
    // 用于存储 Squeeze 节点的列表
    std::vector<Node*> squeeze_nodes;
    squeeze_nodes.reserve(list_unpack_node->outputs().size());
    // 遍历 ListUnpack 节点的每个输出
    for (auto* output : list_unpack_node->outputs()) {
      // 获取可能的 Squeeze 用户节点
      auto* squeeze_node = maybeUserWithKind(output, aten::squeeze);
      // 如果找不到 Squeeze 用户节点，则中断循环
      if (squeeze_node == nullptr) {
        break;
      }
      // 获取 Squeeze 节点的维度信息
      auto dim_opt = toIValue(squeeze_node->input(1));
      // 如果无法获取维度信息或维度不匹配，则中断循环
      if (!dim_opt || *dim_opt != axis) {
        break;
      }
      squeeze_nodes.push_back(squeeze_node);
    }
    auto num_outputs = list_unpack_node->outputs().size();
    // 如果找到的 Squeeze 节点数量与 ListUnpack 节点的输出数量不匹配，则继续下一个节点
    if (squeeze_nodes.size() != num_outputs) {
      continue;
    }
    // 创建一个新的节点来替代 Split 和 Squeeze 操作
    auto* split_and_squeeze_node = graph->create(
        c10::Symbol::fromQualString(
            "static_runtime::fused_split_and_squeeze_copy"),
        num_outputs);
    split_and_squeeze_node->addInput(node->input(0));
    split_and_squeeze_node->addInput(node->input(1));
    split_and_squeeze_node->addInput(node->input(2));
    split_and_squeeze_node->insertBefore(node);
    // 更新输出节点的元数据并替换使用节点
    for (const auto i : c10::irange(num_outputs)) {
      auto* squeeze_node = squeeze_nodes[i];
      split_and_squeeze_node->output(i)->copyMetadata(squeeze_node->output());
      squeeze_node->output()->replaceAllUsesWith(
          split_and_squeeze_node->output(i));
    }
    // 将需要删除的节点添加到删除列表中
    to_erase.insert(to_erase.end(), squeeze_nodes.begin(), squeeze_nodes.end());
    to_erase.push_back(list_unpack_node);
    to_erase.push_back(node);
  }
  // 删除所有标记为需要删除的节点
  for (auto* node : to_erase) {
    node->destroy();
  }
}

// 未使用的函数，用于移除不必要的输出
C10_UNUSED void RemoveUnnecessaryOutputs(
    std::shared_ptr<torch::jit::Graph>& graph) {
  RemoveUnnecessaryEmbeddingBagOutputs(graph);
}

// 未使用的函数，用于移除不必要的嵌入袋子输出
C10_UNUSED void RemoveUnnecessaryEmbeddingBagOutputs(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // 删除不必要的嵌入袋子输出
  std::string pattern = R"IR(
  graph(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset):
      %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
      return (%y2, %y1, %y0))IR";

# 定义一个名为 `graph` 的图函数，接收多个输入参数，执行 PyTorch 的 `aten::embedding_bag` 操作，返回四个张量 `%y0`、`%y1`、`%y2`、`%y3`。


std::string transformed_pattern = R"IR(
  graph(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset):
      %y0 : Tensor, %y1 : Tensor, %y2 : Tensor = static_runtime::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
      return (%y2, %y1, %y0))IR";

# 定义一个名为 `transformed_pattern` 的字符串，用于存储一个经过静态运行时优化后的 IR（Intermediate Representation）图模式，其中执行了 `static_runtime::embedding_bag` 操作，返回三个张量 `%y0`、`%y1`、`%y2`。


SubgraphRewriter fuse;

# 创建名为 `fuse` 的 `SubgraphRewriter` 对象，用于管理和执行图模式的重写。


fuse.RegisterRewritePattern(pattern, transformed_pattern);

# 将之前定义的 `pattern` 和 `transformed_pattern` 注册到 `fuse` 中，以便在图中找到匹配的模式并进行重写。


fuse.runOnGraph(graph);

# 在输入的 `graph` 上运行 `fuse`，对其中匹配的模式进行重写操作。


std::string pattern2 = R"IR(
  graph(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset, %padding_idx):
      %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset, %padding_idx)
      return (%y2, %y1, %y0))IR";

# 定义另一个名为 `pattern2` 的 IR 图模式字符串，它包含一个额外的参数 `%padding_idx`，执行 PyTorch 的 `aten::embedding_bag` 操作，并返回四个张量 `%y0`、`%y1`、`%y2`、`%y3`。


std::string transformed_pattern2 = R"IR(
  graph(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset, %padding_idx):
      %y0 : Tensor, %y1 : Tensor, %y2 : Tensor = static_runtime::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset, %padding_idx)
      return (%y2, %y1, %y0))IR";

# 定义另一个名为 `transformed_pattern2` 的字符串，存储经过静态运行时优化后的 IR 图模式，执行 `static_runtime::embedding_bag` 操作，返回三个张量 `%y0`、`%y1`、`%y2`。


fuse.RegisterRewritePattern(pattern2, transformed_pattern2);

# 将 `pattern2` 和 `transformed_pattern2` 注册到 `fuse` 中，准备对输入的图进行匹配和重写。


fuse.runOnGraph(graph);

# 在输入的 `graph` 上再次运行 `fuse`，对匹配的模式进行重写操作。
} // 关闭匿名命名空间

namespace {
    // 判断节点是否为无操作切片节点
    bool isNoOpSlice(Node* node) {
        // 使用断言确保节点类型为 aten::slice
        DCHECK(node->kind() == aten::slice);
        // 获取第四个输入作为步长
        auto step = toIValue(node->input(3));
        // 如果步长不存在或者不为1，则返回 false
        if (!step.has_value() || step->toInt() != 1) {
            return false;
        }
        // 获取第二个输入作为起始位置
        auto start = toIValue(node->input(1));
        // 如果起始位置不存在，或者是整数且不为0，则返回 false
        if (!start.has_value() || (start->isInt() && start->toInt() != 0)) {
            return false;
        }
        // 获取第三个输入作为结束位置
        auto end = toIValue(node->input(2));
        // 可以查看列表长度，但是大多数具有此模式的模型只是进行列表[0:]，因此目前不需要
        // 如果结束位置存在且为 None，则返回 true，否则返回 false
        return end.has_value() && end->isNone();
    }
} // 命名空间结束

// 从图中消除无操作切片
void EliminateNoOpSlice(std::shared_ptr<Graph>& graph) {
    // 深度优先遍历图的节点迭代器
    DepthFirstGraphNodeIterator it(graph);
    // 定义 ATen slice 操作的模式
    auto schema = torch::schema(
        "aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> t[]",
        /*allow_typevars*/ true);
    // 初始化节点指针
    Node* node = nullptr;
    // 初始化要删除的节点列表
    std::vector<Node*> to_delete;
    // 遍历图中的每个节点
    while ((node = it.next()) != nullptr) {
        // 如果节点不匹配模式或者不是无操作切片，则继续下一个节点
        if (!node->matches(schema) || !isNoOpSlice(node)) {
            continue;
        }
        // 用第一个输入替换节点的输出
        node->output()->replaceAllUsesWith(node->input(0));
        // 将节点添加到待删除列表
        to_delete.push_back(node);
    }
    // 遍历待删除列表，销毁节点
    for (auto* node : to_delete) {
        node->destroy();
    }
}

// 更新图中的 in-place 操作，从可选输入中获取真实输入 V2
void UseInPlaceGetRealInputsFromOptionalInputsV2(std::shared_ptr<Graph>& graph) {
#ifdef FBCODE_CAFFE2
    // 定义原始模式和新模式的 IR 表示
    const std::string original_pattern = R"IR(
        graph(%optional_input: (Tensor, Tensor?, Tensor?)?[], %include_last_offsets: bool[]):
            %x : (Tensor, Tensor?, Tensor?)[] = remote_collection::get_real_inputs_from_optional_inputs_v2(%optional_input, %include_last_offsets)
            return (%x))IR";

    const std::string new_pattern = R"IR(
        graph(%optional_input: (Tensor, Tensor?, Tensor?)?[], %include_last_offsets: bool[]):
            %x : (Tensor, Tensor?, Tensor?)[] = static_runtime::get_real_inputs_from_optional_inputs_v2_inplace(%optional_input, %include_last_offsets)
            return (%x))IR";

    // 判断值是否仅被单个使用
    auto isSingleUse = [](Value* value) { return value->uses().size() == 1; };

    // 筛选器函数，检查是否满足单一使用的条件
    auto filter = [&isSingleUse](
                      const Match& match,
                      const std::unordered_map<std::string, Value*>& vmap) {
        auto* real_node = match.nodes_map.at(vmap.at("x")->node());
        return isSingleUse(real_node->input(0));
    };

    // 子图重写器对象
    SubgraphRewriter fuse;
    // 注册重写模式
    fuse.RegisterRewritePattern(original_pattern, new_pattern);
    // 在图上运行重写器
    fuse.runOnGraph(graph, filter);
#endif
}

// 将 NaN 值截断到指定数值之间
void FuseClampNaNToNum(std::shared_ptr<Graph>& graph) {
#ifdef FBCODE_CAFFE2
    // 定义截断 NaN 到指定数值之间的模式
    std::string pattern = R"IR(
        graph(%input, %clamp_min: Scalar?, %clamp_max: Scalar?, %nan, %posinf, %neginf):
            %x : Tensor = aten::clamp(%input, %clamp_min, %clamp_max)
            %y : Tensor = aten::nan_to_num(%x, %nan, %posinf, %neginf)
            return (%y))IR";

    std::string fused_pattern = R"IR(
        graph(%input, %clamp_min: Scalar?, %clamp_max: Scalar?, %nan, %posinf, %neginf):
            %x : Tensor = static_runtime::clamp_nan_to_num(%input, %clamp_min, %clamp_max, %nan, %posinf, %neginf)
            return (%x))IR";

    // 判断值是否为常量且不为 None
    auto isConstantAndNotNone = [](Value* value) {
    // 将 value 转换为 IValue 类型的可选值
    auto ival_opt = toIValue(value);
    // 如果转换结果不包含值，返回 false
    if (!ival_opt.has_value()) {
      return false;
    }
    // 将 IValue 转换为 at::Scalar 类型的可选值
    auto scalar_opt = ival_opt->toOptional<at::Scalar>();
    // 返回是否存在有效的标量值
    return scalar_opt.has_value();
  };

  // 检查 clamp 操作的输入是否为常量且不为 None
  auto clampValuesAreConstant =
      [&isConstantAndNotNone](
          const Match& match,
          const std::unordered_map<std::string, Value*>& vmap) {
        // 从模板图中的节点映射获取真实图中的节点
        const auto& node_map = match.nodes_map;
        // 获取 clamp 节点
        auto* clamp_node = node_map.at(vmap.at("x")->node());
        // 检查 clamp 操作的第二个和第三个输入是否都是常量且不为 None
        return isConstantAndNotNone(clamp_node->input(1)) &&
            isConstantAndNotNone(clamp_node->input(2));
      };

  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册模式匹配和重写的模式
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  // 在图上运行子图重写，使用 clampValuesAreConstant 函数确定是否应用重写
  fuse.runOnGraph(graph, clampValuesAreConstant);
#endif
}

// 预处理权重函数，将图中指定模式的计算替换为量化线性运算的优化版本
void PrepackWeights(std::shared_ptr<Graph>& graph) {
  // 定义量化线性计算的模式字符串
  const auto pattern = R"IR(
    graph(%input: Tensor, %weight: Tensor, %bias: Tensor?, %scale: Tensor, %zero_point: Tensor):
        %result: Tensor = fb::quantized_linear_unpacked_weight_v2(%input, %weight, %bias, %scale, %zero_point)
        return (%result)
  )IR";

  // 定义线性层预打包的模式字符串
  const auto split_pattern = R"IR(
    graph(%input: Tensor, %weight: Tensor, %bias: Tensor?, %scale: Tensor, %zero_point: Tensor):
        %packed_params = quantized::linear_prepack(%weight, %bias)
        %scale_float: float = aten::item(%scale)
        %zero_point_int: int = aten::item(%zero_point)
        %result: Tensor = quantized::linear(%input, %packed_params, %scale_float, %zero_point_int)
        return (%result)
  )IR";

  // 创建子图重写器对象
  SubgraphRewriter fuse;
  // 注册将 pattern 替换为 split_pattern 的重写模式
  fuse.RegisterRewritePattern(pattern, split_pattern);
  // 在图上运行重写器，实施模式替换
  fuse.runOnGraph(graph);
  // 在此之后应调用常量传播和其他优化方法
}

// 命名空间结束声明，关闭 torch::jit 命名空间
} // namespace torch::jit
```