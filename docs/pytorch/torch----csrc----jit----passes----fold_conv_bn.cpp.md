# `.\pytorch\torch\csrc\jit\passes\fold_conv_bn.cpp`

```
#include <torch/csrc/jit/passes/fold_conv_bn.h>  
// 包含了实现卷积和批归一化融合的头文件

#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
// 包含了用于图重写和量化辅助的头文件

#include <ATen/TensorOperators.h>
// 包含了 ATen 的张量操作头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/zeros_like.h>
#endif
// 根据预处理器宏选择性地包含 ATen 的操作头文件

#include <stack>
#include <utility>
// 包含了标准库的堆栈和实用工具

namespace torch {
namespace jit {

std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
    const ConvBNParameters& p) {
  // 计算更新后的卷积权重和偏置
  at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);
  // 计算批归一化的标准差的倒数

  const int64_t ndim = p.conv_w.dim();
  // 获取卷积权重张量的维度数
  at::DimVector sizes(ndim, 1);
  sizes.at(0) = -1;

  auto conv_w_dtype = p.conv_w.dtype();
  auto conv_b_dtype = p.conv_b.dtype();
  // 获取卷积权重和偏置的数据类型

  at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape(sizes);
  // 计算更新后的卷积权重
  at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w + p.bn_b;
  // 计算更新后的卷积偏置
  return std::make_tuple(new_w.to(conv_w_dtype), new_b.to(conv_b_dtype));
  // 返回更新后的卷积权重和偏置的元组
}

namespace {
using graph_rewrite_helper::PatternInfo;

static bool hastensor(Module& m, const char* name) {
  // 检查模块中是否存在指定名称的张量属性
  return m.hasattr(name) && m.attr(name).isTensor();
}

void replaceConvBiasWithGetAttr(Module& module) {
  // 替换模块中所有方法的卷积偏置为获取属性操作

  for (const auto& method : module.get_methods()) {
    auto graph = method.graph();
    // 获取方法对应的计算图

    // 只查找 _convolution 模式，假设跟踪已经处理了 aten::conv2d 或 aten::conv3d
    // 如果没有处理，批归一化融合将失败。
    const PatternInfo& pattern_convolution = PatternInfo::parse_from_str(R"(
        graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
            %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
            %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
          %conv_out = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
              %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled, %allow_tf32)
          return (%conv_out) )");
    // 定义匹配卷积操作的模式信息

    const PatternInfo& pattern_convolution_deprecated =
        PatternInfo::parse_from_str(R"(
        graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
            %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
            %deterministic:bool, %cudnn_enabled:bool):
          %conv_out = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
              %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled)
          return (%conv_out) )");
    // 定义匹配过时卷积操作的模式信息
    # 定义一个lambda函数replace_pattern，用于替换模式中的特定模式信息
    auto replace_pattern = [&](const PatternInfo& pattern_convolution) {
      # 从pattern_convolution中获取模式图和映射信息
      const Graph& pattern_convolution_graph =
          *pattern_convolution.pattern_graph;
      const auto& convolution_vmap = pattern_convolution.vmap;

      # 在图中查找与给定模式图匹配的所有匹配项
      const auto& matches =
          findPatternMatches(pattern_convolution_graph, *graph);
      # 遍历每个匹配项
      for (const auto& match : matches) {
        // 只有当模块中不存在偏置项时才会进入此处
        // 在这种情况下，对应的图将不会有getAttr("bias")
        // 在图中插入偏置项
        // 并修改_convolution以使用新值
        auto conv_node =
            match.values_map.at(convolution_vmap.at("conv_out"))->node();
        # 将插入点设置在conv_node处
        WithInsertPoint ins(conv_node);
        # 插入一个获取输入图的偏置属性的操作，并设置其类型为TensorType
        Value* bias_attr_val = graph->insertGetAttr(graph->inputs()[0], "bias")
                                   ->setType(TensorType::get());
        constexpr size_t conv_bias_index = 2;
        # 替换conv_node的第conv_bias_index个输入为bias_attr_val
        conv_node->replaceInput(conv_bias_index, bias_attr_val);
      }
    };
    # 使用replace_pattern函数分别替换两个模式：pattern_convolution和pattern_convolution_deprecated
    replace_pattern(pattern_convolution);
    replace_pattern(pattern_convolution_deprecated);
}

// 为给定模块添加偏置项，如果不存在的话
void addBiasForConvIfNone(Module& module, const std::string& pattern_name) {
  // 获取模块的类型信息，并期望其为ClassType类型
  auto t = module.type()->expect<ClassType>();

  // 获取模块的实际类型名称，并解除Torch的名称修饰
  const std::string real_typename = t->name()->qualifiedName();
  const std::string demangled_typename = removeTorchMangle(real_typename);

  // 检查模块是否为浮点数卷积层(Conv1d, Conv2d, Conv3d)
  bool is_floating_point_conv =
      ((demangled_typename == "__torch__.torch.nn.modules.conv.Conv1d") ||
       (demangled_typename == "__torch__.torch.nn.modules.conv.Conv2d") ||
       (demangled_typename == "__torch__.torch.nn.modules.conv.Conv3d"));

  // 如果模块是浮点数卷积层并且没有偏置项，则添加偏置项
  if (is_floating_point_conv) {
    if (!t->hasAttribute("bias")) {
      auto optional_tensor_type = OptionalType::create(TensorType::get());
      t->addAttribute("bias", std::move(optional_tensor_type), true);
      auto optional_tensor = std::optional<at::Tensor>();
      module.setattr("bias", std::move(optional_tensor));
      replaceConvBiasWithGetAttr(module);
    }
  }

  // 递归处理模块的子模块
  for (Module m : module.children()) {
    addBiasForConvIfNone(m, pattern_name);
  }
}

// 辅助类，用于卷积-批归一化模式的折叠
class FoldConvBatchNormHelper {
 public:
  /**
   * 在此步骤中，我们查找图中所有Conv - BatchNorm模式，并提取这两个模块的相应参数，
   * 记录修改图的信息，但实际上不执行这些修改。
   */
  void analyze(Module& module, const PatternInfo& pattern);
  
  /**
   * 在此步骤中，我们执行所有的修改，包括设置conv模块的属性，重写值和删除图中的节点。
   */
  void transform();

 private:
  bool tryExtractingConvBNParameters(
      Module& conv,
      Module& bn,
      ConvBNParameters& r);

  // 保存卷积模块及其参数的映射
  std::unordered_map<ModulePtr, std::tuple<at::Tensor, at::Tensor>>
      conv_module_and_params_;

  // 图到匹配的Conv-BN模块路径列表的映射
  // 例如，如果图g包含以下代码
  // x = self.sub.conv1(..)
  // x = self.sub.bn1(..)
  // x = self.sub.conv2(..)
  // x = self.sub.bn2(..)
  // 则图g在该映射中的值将是：
  // [(['sub', 'conv1'], ['sub', 'bn1']), (['sub', 'conv2'], ['sub', 'bn2'])]
  // 列表的第一个条目是第一个Conv-BN匹配的路径，
  // 第二个条目是第二个匹配的路径
  std::unordered_map<
      Graph*,
      std::vector<
          std::tuple<std::vector<std::string>, std::vector<std::string>>>>
      conv_bn_paths_;

  // 重写映射，用于重写值
  std::unordered_map<Value*, Value*> rewrite_map_;
  
  // 需要重写的值的列表
  std::vector<Value*> values_to_rewrite_;

  // 需要删除的节点集合
  std::unordered_set<Node*> nodes_to_delete_;
};
bool extractOptionalBNParams(const script::Module& bn, ConvBNParameters& r) {
  // 获取模块中的 forward 方法
  auto bn_forward = bn.get_method("forward");
  // 获取方法的计算图
  auto graph = bn_forward.graph();
  // 定义用于匹配的模式信息
  const PatternInfo& pattern_bn = PatternInfo::parse_from_str(R"(
      graph(%a, %weight, %bias, %running_mean, %running_var,
          %training, %momentum, %eps, %cudnn_enabled):
        %bn_out = aten::batch_norm(%a, %weight, %bias, %running_mean,
            %running_var, %training, %momentum, %eps, %cudnn_enabled)
        return (%bn_out) )");
  // 获取模式的计算图
  const Graph& pattern_bn_graph = *pattern_bn.pattern_graph;
  // 获取模式中的值映射
  const auto& bn_vmap = pattern_bn.vmap;

  // 在模块的计算图中查找模式的匹配
  const auto& matches = findPatternMatches(pattern_bn_graph, *graph);

  // 如果匹配超过一个，则返回 false
  if (matches.size() > 1) {
    return false;
  }

  // 如果模块有 eps 属性，则将其值赋给 r.bn_eps
  if (bn.hasattr("eps")) {
    r.bn_eps = bn.attr("eps").toDouble();
  } else {
    // 否则从模式匹配中获取 eps 属性的值
    auto optional_eps = toIValue(matches[0].values_map.at(bn_vmap.at("eps")));
    if (!optional_eps) {
      return false;
    }
    r.bn_eps = optional_eps.value().toDouble();
  }

  // 初始化 r.bn_w 为与 bn.running_mean 维度相同的全 1 张量
  r.bn_w = at::ones_like(bn.attr("running_mean").toTensor());

  // 如果模块有 weight 属性，并且其为 Tensor，则将其值赋给 r.bn_w
  if (bn.hasattr("weight")) {
    if (bn.attr("weight").isTensor()) {
      r.bn_w = bn.attr("weight").toTensor();
    }
  } else {
    // 否则从模式匹配中获取 weight 属性的值
    auto optional_bn_weight =
        toIValue(matches[0].values_map.at(bn_vmap.at("weight")));
    if (!optional_bn_weight) {
      return false;
    }
    // 如果值为 Tensor，则赋给 r.bn_w
    if (optional_bn_weight.value().isTensor()) {
      r.bn_w = optional_bn_weight.value().toTensor();
    }
  }

  // 初始化 r.bn_b 为与 bn.running_mean 维度相同的全 0 张量
  r.bn_b = at::zeros_like(bn.attr("running_mean").toTensor());

  // 如果模块有 bias 属性，并且其为 Tensor，则将其值赋给 r.bn_b
  if (bn.hasattr("bias")) {
    if (bn.attr("bias").isTensor()) {
      r.bn_b = bn.attr("bias").toTensor();
    }
  } else {
    // 否则从模式匹配中获取 bias 属性的值
    auto optional_bn_bias =
        toIValue(matches[0].values_map.at(bn_vmap.at("bias")));
    if (!optional_bn_bias) {
      return false;
    }
    // 如果值为 Tensor，则赋给 r.bn_b
    if (optional_bn_bias.value().isTensor()) {
      r.bn_b = optional_bn_bias.value().toTensor();
    }
  }

  // 返回 true 表示成功提取参数
  return true;
}

bool FoldConvBatchNormHelper::tryExtractingConvBNParameters(
    Module& conv,
    Module& bn,
    ConvBNParameters& r) {
  // 检查 conv 和 bn 模块是否包含所需的属性和张量
  if (!hastensor(conv, "weight") || !conv.hasattr("bias") ||
      !hastensor(bn, "running_mean") || !hastensor(bn, "running_var")) {
    return false;
  }

  // 获取 bn 模块的 running_mean 和 running_var 属性值，并赋给 r.bn_rm 和 r.bn_rv
  r.bn_rm = bn.attr("running_mean").toTensor();
  r.bn_rv = bn.attr("running_var").toTensor();

  // 提取 bn 模块的可选参数到 r 中
  if (!extractOptionalBNParams(bn, r)) {
    return false;
  }

  // 获取 conv 模块的 weight 属性值，并赋给 r.conv_w
  r.conv_w = conv.attr("weight").toTensor();
  // 初始化 r.conv_b 为与 bn.running_mean 维度相同的全 0 张量
  r.conv_b = at::zeros_like(r.bn_rm);

  // 如果 conv 模块有 bias 属性，则将其值赋给 r.conv_b
  auto bias_opt = conv.attr("bias").toOptional<at::Tensor>();
  if (bias_opt) {
    r.conv_b = *bias_opt;
  }

  // 返回 true 表示成功提取参数
  return true;
}

void FoldConvBatchNormHelper::analyze(
    Module& module,
    const PatternInfo& pattern) {
  // 从输入参数中获取模式图和映射表
  const Graph& pattern_graph = *pattern.pattern_graph;
  const auto& vmap = pattern.vmap;
  // 从映射表中获取模式匹配的节点值
  Value* pattern_conv_out = vmap.at("conv_out");
  Value* pattern_bn_out = vmap.at("bn_out");
  Value* pattern_bn_submodule = vmap.at("batchnorm");
  // 从节点值获取对应的节点
  Node* pattern_conv = pattern_conv_out->node();
  Node* pattern_bn = pattern_bn_out->node();

  // 将当前模块加入工作列表，准备处理
  // 使用堆栈来处理工作列表，开始时只包含顶层模块
  std::stack<Module> worklist({module});
  while (!worklist.empty()) {
    // 取出堆栈顶部的模块作为当前处理的模块
    Module current = worklist.top();
    worklist.pop();

    // 将当前模块的子模块加入工作列表，以便后续处理
    for (const Module& submodule : current.children()) {
      worklist.push(submodule);
    }

    // 处理当前模块的所有方法
    } // methods
  } // while
}

void FoldConvBatchNormHelper::transform() {
  // 对于每个卷积模块及其参数执行以下操作
  for (const auto& item : conv_module_and_params_) {
    // 创建一个 Module 对象，并设置其权重和偏置
    Module conv(item.first);
    auto w_b = item.second;
    conv.setattr("weight", std::get<0>(w_b));
    conv.setattr("bias", std::get<1>(w_b));
  }

  // 执行计划中的重写操作
  for (auto v : values_to_rewrite_) {
    // 用重写映射中的值替换所有使用 v 的地方
    v->replaceAllUsesWith(rewrite_map_.at(v));
  }

  // 执行计划中的删除操作
  for (auto n : nodes_to_delete_) {
    // 移除节点 n 的所有输入连接
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete_) {
    // 销毁节点 n
    n->destroy();
  }
}

} // namespace

Module FoldConvBatchNorm(const Module& module) {
  // 克隆输入模块
  Module m = module.clone();

  // 如果模块中的 Conv2d 模块没有偏置，则添加偏置
  addBiasForConvIfNone(m, "Conv2d");
  // 如果模块中的 Conv3d 模块没有偏置，则添加偏置
  addBiasForConvIfNone(m, "Conv3d");

  // 定义匹配模式，用于查找 Conv2d + BatchNorm2d 模式
  const PatternInfo pattern2d = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %batchnorm):
    %conv_out = prim::CallMethod[name="forward"](%conv, %input)
    %bn_out = prim::CallMethod[name="forward"](%batchnorm, %conv_out)
    return (%bn_out))",
      {is_conv2d_module, is_batchnorm2d_module});
  
  // 定义匹配模式，用于查找 Conv3d + BatchNorm3d 模式
  const PatternInfo pattern3d = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %batchnorm):
    %conv_out = prim::CallMethod[name="forward"](%conv, %input)
    %bn_out = prim::CallMethod[name="forward"](%batchnorm, %conv_out)
    return (%bn_out))",
      {is_conv3d_module, is_batchnorm3d_module});

  // 将匹配模式存入向量中
  const std::vector<std::reference_wrapper<const PatternInfo>> patterns = {
      pattern2d, pattern3d};

  // 对每个模式执行折叠卷积与批归一化的帮助函数
  for (const auto& pattern : patterns) {
    FoldConvBatchNormHelper h;
    h.analyze(m, pattern);
    h.transform();
  }
  // 返回修改后的模块
  return m;
}

} // namespace jit
} // namespace torch
```