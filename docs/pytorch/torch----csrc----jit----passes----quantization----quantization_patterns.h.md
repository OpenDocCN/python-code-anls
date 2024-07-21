# `.\pytorch\torch\csrc\jit\passes\quantization\quantization_patterns.h`

```
#pragma once

#include <c10/util/irange.h>  // 包含用于迭代范围的头文件
#include <torch/csrc/jit/ir/ir.h>  // 包含 JIT IR 相关头文件
#include <torch/csrc/jit/ir/subgraph_matcher.h>  // 包含子图匹配器的头文件
#include <torch/csrc/jit/jit_log.h>  // 包含 JIT 日志相关头文件
#include <torch/csrc/jit/passes/quantization/helper.h>  // 包含量化辅助函数头文件
#include <torch/csrc/jit/passes/subgraph_rewrite.h>  // 包含子图重写相关头文件
#include <string>  // 包含处理字符串的头文件
#include <unordered_map>  // 包含无序映射的头文件
#include <utility>  // 包含实用工具的头文件

namespace torch {
namespace jit {

// 量化融合信息的结构体
struct QuantFusionInfo {
  std::string quantized_op_name;  // 量化操作的名称
  std::string pattern;  // 模式字符串
  std::string replacement;  // 替换字符串
  std::vector<MatchFilter> filters = {};  // 匹配过滤器的向量
};

namespace {

// 获取额外参数列表的字符串表示
std::string getExtraArgList(std::vector<std::string> extra_args) {
  return std::accumulate(
      extra_args.begin(),
      extra_args.end(),
      std::string(),
      [](std::string acc, const std::string& arg) { return acc + ", " + arg; });
}

// 获取要替换匹配项的模式字符串
std::string getAtenOpPattern(
    const std::string& graph_header,
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args,
    bool scalar_args = false) {
  std::vector<std::string> _extra_op_args = extra_op_args;
  std::string aten_op_pattern = graph_header;
  if (scalar_args) {
    for (const auto& extra_arg : _extra_op_args) {
      aten_op_pattern
          .append(R"(
          )")
          .append(extra_arg)
          .append("_scalar = aten::item(")
          .append(extra_arg)
          .append(")");  // 生成标量参数的 aten::item 操作
    }

    for (auto& _extra_op_arg : _extra_op_args) {
      _extra_op_arg.append("_scalar");  // 将参数名后添加 "_scalar"
    }
  }
  const auto& extra_op_arg_list = getExtraArgList(std::move(_extra_op_args));
  aten_op_pattern += R"(
          %r = )";  // 设置模式的结果变量
  aten_op_pattern += op_name + "(" + "%a_quant" + extra_op_arg_list + ")";  // 构建完整的操作模式字符串
  aten_op_pattern += R"(
          return (%r) )";  // 添加模式的返回结果
  return aten_op_pattern;
}

// 为标量值生成量化模式的操作
std::string getQuantizeForScalar(const std::string& value) {
  // 6 是 `torch.float` ScalarType，从标量值创建一个 float 标量张量
  std::string quantize_pattern = R"(
          )" +
      value + "_float_scalar_type : int = prim::Constant[value=6]()";  // 创建常量的量化模式
  quantize_pattern += R"(
          )" +
      value + "_none : None = prim::Constant()";  // 创建空值常量
  quantize_pattern += R"(
          )" +
      value + "_tensor : Tensor = aten::scalar_tensor(" + value + ", " + value +
      "_float_scalar_type";  // 创建标量张量
  for (const auto i : c10::irange(3)) {
    (void)i; // 抑制未使用变量警告
    quantize_pattern += ", " + value + "_none";  // 添加额外参数
  }
  quantize_pattern += ")";
  quantize_pattern +=
      R"(
          )" +
      value + "_quant = aten::quantize_per_tensor(" + value + "_tensor" +
      getExtraArgList(
          {value + "_scale", value + "_zero_point", value + "_dtype"}) +
      ")";  // 创建张量的量化模式
  return quantize_pattern;
}

// 获取反量化操作的字符串表示
std::string getDequantize(const std::string& value) {
  return R"(
          )" +
      value + "_dequant = aten::dequantize(" + value + "_quant)";  // 创建反量化操作的量化模式
}
// 返回一个字符串，包含特定值的标量项，用于生成代码
std::string getItem(const std::string& value) {
  return R"(
          )" +
      value + "_scalar : float = aten::item(" + value + "_dequant)";
}

// 生成用于输入张量参数化操作模式的字符串模式
std::string getInputTensorQParamOpPattern(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args) {
  // 获取额外操作参数列表的字符串表示
  const auto& extra_op_arg_list = getExtraArgList(extra_op_args);
  // 构建操作模式字符串，包括图模式和操作名称
  std::string op_pattern = "graph(%a_quant" + extra_op_arg_list + "):" + R"(
          %a_dequant = aten::dequantize(%a_quant)
          %r = )" +
      op_name + "(" + "%a_dequant" + extra_op_arg_list + ")" + R"(
          %r_scale : float = aten::q_scale(%a_quant)
          %r_zero_point : int = aten::q_zero_point(%a_quant)
          %r_dtype : int = prim::dtype(%a_quant)
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";
  return op_pattern;
}

// 为从输入继承参数的操作生成 QuantFusionInfo
QuantFusionInfo getInputTensorQParamOpFusionInfo(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args) {
  // 获取操作模式的字符串表示
  std::string op_pattern =
      getInputTensorQParamOpPattern(op_name, extra_op_args);
  // 获取额外操作参数列表的字符串表示
  const auto& extra_op_arg_list = getExtraArgList(extra_op_args);
  // 构建图头部字符串
  std::string graph_header = "graph(%a_quant" + extra_op_arg_list + "):";
  // 获取 ATen 操作模式的字符串表示
  std::string op_replacement =
      getAtenOpPattern(graph_header, op_name, extra_op_args);

  return {op_name, std::move(op_pattern), std::move(op_replacement)};
}

// 为像 `quantized::add_scalar`, `quantized::mul_scalar` 这样的二元操作生成 QuantFusionInfo
QuantFusionInfo getBinaryOpScalarFusionInfo(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args,
    const std::string& quantized_op_name,
    const std::vector<std::string>& extra_quantized_op_args,
    const std::vector<MatchFilter>& filters = {}) {
  // 获取输入张量参数化操作模式的字符串表示
  std::string op_pattern =
      getInputTensorQParamOpPattern(op_name, extra_op_args);

  const auto& extra_op_arg_list = getExtraArgList(extra_op_args);
  // 构建图头部字符串
  std::string graph_header = "graph(%a_quant" + extra_op_arg_list + "):";
  // 获取量化操作模式的字符串表示
  std::string op_replacement = getAtenOpPattern(
      graph_header, quantized_op_name, extra_quantized_op_args);

  return {op_name, std::move(op_pattern), std::move(op_replacement), filters};
}

// 为 `clamp` 操作生成 QuantFusionInfo
QuantFusionInfo getClampOpFusionInfo(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args) {
  // 创建额外操作参数的头部参数列表
  std::vector<std::string> header_args = extra_op_args;
  // 定义输入量化参数的后缀
  std::vector<std::string> input_qparams = {"_scale", "_zero_point", "_dtype"};
  // 对于每个额外参数，添加量化参数后缀
  for (const auto& arg : extra_op_args) {
    for (const auto& qparam : input_qparams) {
      header_args.push_back(arg + qparam);
    }
  }
  for (const auto& qparam : input_qparams) {
    header_args.push_back("%r" + qparam);

# 将 "%r" 与 qparam 拼接后，添加到 header_args 后面

  const auto& extra_header_arg_list = getExtraArgList(std::move(header_args));

# 调用 getExtraArgList 函数，获取处理后的 header_args 的不可变引用

  std::string graph_header = "graph(%a_quant" + extra_header_arg_list + "):";

# 创建 graph_header 字符串，以 "graph(%a_quant" 开头，后跟 extra_header_arg_list，以 ":" 结尾

  std::string op_pattern = graph_header;

# 将 graph_header 复制到 op_pattern 中，作为初始模式

  for (const auto& arg : extra_op_args) {
    op_pattern += getQuantizeForScalar(arg);
    op_pattern += getDequantize(arg);
    op_pattern += getItem(arg);
  }

# 遍历 extra_op_args 列表，为每个参数 arg 依次调用 getQuantizeForScalar、getDequantize 和 getItem 函数，并将结果追加到 op_pattern 中

  op_pattern += getDequantize("%a");

# 将 getDequantize("%a") 的结果追加到 op_pattern 中

  op_pattern += R"(
          %r = )";

# 将字符串 "          %r = )" 追加到 op_pattern 中（这是一个原始字符串字面量的开头）

  std::vector<std::string> scalar_extra_args;
  scalar_extra_args.reserve(extra_op_args.size());
  for (const auto& arg : extra_op_args) {
    scalar_extra_args.push_back(arg + "_scalar");
  }

# 创建 scalar_extra_args 向量，并为每个 extra_op_args 中的元素创建对应的 "_scalar" 后缀，将其添加到 scalar_extra_args 中

  op_pattern += op_name + "(" + "%a_dequant" +
      getExtraArgList(std::move(scalar_extra_args)) + ")";

# 构建 op_pattern 字符串，拼接 op_name、"%a_dequant" 和处理后的 scalar_extra_args 列表

  // IR pattern common to all ops that inherit qparam from input
  op_pattern += R"(
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";

# 将原始字符串字面量 "(          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)\n          return (%r_quant) )" 添加到 op_pattern 中

  std::string aten_op_pattern =
      getAtenOpPattern(graph_header, op_name, extra_op_args);

# 调用 getAtenOpPattern 函数，生成 aten_op_pattern 字符串，用于描述与 graph_header、op_name 和 extra_op_args 相关的操作模式

  return {op_name, std::move(op_pattern), std::move(aten_op_pattern)};

# 返回一个包含 op_name、op_pattern 和 aten_op_pattern 的元组
}

// 获取具有固定量化参数的操作的融合信息
QuantFusionInfo getFixedQParamOpFusionInfo(
    const std::string& op_name,  // 操作的名称
    const std::vector<std::string>& extra_op_args,  // 额外的操作参数列表
    bool is_symmetric) {  // 是否对称量化的标志

  const auto& extra_op_arg_list = getExtraArgList(extra_op_args);  // 获取额外参数列表的字符串表示
  std::string graph_header = "graph(%a_quant" + extra_op_arg_list + "):";  // 定义图头部
  std::string op_pattern = graph_header;  // 初始化操作模式字符串
  op_pattern += R"(
          %a_dequant = aten::dequantize(%a_quant)  // 对输入量化进行反量化
          %r = )";
  op_pattern += op_name + "(" + "%a_dequant" + extra_op_arg_list + ")";  // 构建操作模式字符串
  // IR模式，所有具有固定量化参数操作的共同模式，用于非对称量化
  std::string asym_fixed_qparam_op_suffix = R"(
          %r_scale : float = prim::Constant[value=0.00390625]()  // 输出的量化比例
          %r_zero_point : int = prim::Constant[value=0]()  // 输出的零点
          %r_dtype : int = prim::Constant[value=13]()  // 输出的数据类型
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)  // 对输出进行量化
          return (%r_quant) )";  // 返回量化后的结果

  std::string sym_fixed_qparam_op_suffix = R"(
          %r_scale : float = prim::Constant[value=0.0078125]()  // 输出的量化比例
          %r_zero_point : int = prim::Constant[value=128]()  // 输出的零点
          %r_dtype : int = prim::Constant[value=13]()  // 输出的数据类型
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)  // 对输出进行量化
          return (%r_quant) )";  // 返回量化后的结果
  op_pattern +=
      is_symmetric ? sym_fixed_qparam_op_suffix : asym_fixed_qparam_op_suffix;  // 根据是否对称量化选择相应的操作模式字符串

  std::string aten_op_pattern =
      getAtenOpPattern(graph_header, op_name, extra_op_args);  // 获取基于 ATen 操作的模式字符串

  return {op_name, std::move(op_pattern), std::move(aten_op_pattern)};  // 返回操作的融合信息
}

// 过滤器，检查 %b_scalar 是否是一个标量
bool input_b_is_scalar(
    const Match& match,  // 匹配对象
    const std::unordered_map<std::string, Value*>& vmap) {  // 值映射表

  const auto& match_vmap = match.values_map;  // 匹配对象的值映射
  auto b_scalar = match_vmap.at(vmap.at("b_scalar"));  // 获取标量 b_scalar
  return isScalar(b_scalar);  // 检查是否为标量并返回结果
}

// 获取需要观察输出量化参数的操作的融合信息
QuantFusionInfo getObservedQParamOpFusionInfo(
    const std::string& fp_op_name,  // 浮点操作的名称
    const std::string& q_op_name,  // 量化操作的名称
    const std::vector<std::string>& fp_extra_args,  // 浮点操作的额外参数列表
    ...
    // 定义函数`getOperationPatterns`，接受两个向量引用参数`fp_extra_args`和`q_extra_args`
    const std::vector<std::string>& q_extra_args) {
      // 调用`getExtraArgList`函数获取`fp_extra_args`的额外参数列表
      const auto& fp_extra_arg_list = getExtraArgList(fp_extra_args);
      // 调用`getExtraArgList`函数获取`q_extra_args`的额外参数列表
      const auto& q_extra_arg_list = getExtraArgList(q_extra_args);
    
      // 构建操作模式字符串`op_pattern`，用于`fp_op_name`操作
      std::string op_pattern = "graph(%a_quant" + fp_extra_arg_list +
          ", %r_scale, %r_zero_point, %r_dtype):" + R"(
              %a_dequant = aten::dequantize(%a_quant)
              %r = )" +
          fp_op_name + "(" + "%a_dequant" + fp_extra_arg_list + ")" + R"(
              %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
              return (%r_quant) )";
    
      // 构建操作模式字符串`aten_op_pattern`，用于`q_op_name`操作
      std::string aten_op_pattern = "graph(%a_quant" + fp_extra_arg_list +
          ", %r_scale, %r_zero_point, %r_dtype):" + R"(
              %r_quant = )" +
          q_op_name + "(%a_quant" + q_extra_arg_list +
          ", %r_scale, %r_zero_point)" + R"(
              return (%r_quant) )";
    
      // 返回包含操作名`q_op_name`、操作模式`op_pattern`和`aten_op_pattern`的字符串向量
      return {q_op_name, std::move(op_pattern), std::move(aten_op_pattern)};
    }
// 结束了命名空间声明

} // namespace

// 定义静态函数 quant_fusion_pattern_and_replacements，返回一个包含 QuantFusionInfo 结构的向量
static std::vector<QuantFusionInfo> quant_fusion_pattern_and_replacements() {
  // 定义字符串 conv1d，包含一个表示图形的原始字符串，描述了 aten::conv1d 操作的计算图
  std::string conv1d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // 定义字符串 conv1d_relu，包含一个表示图形的原始字符串，描述了 aten::conv1d - aten::relu 操作的计算图
  std::string conv1d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %conv_out = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r = aten::relu(%conv_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // 定义字符串 conv1d_inplace_relu，包含一个表示图形的原始字符串，描述了 aten::conv1d - aten::relu_ 操作的计算图
  std::string conv1d_inplace_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %conv_out = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r = aten::relu_(%conv_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // 定义字符串 quantized_conv1d，包含一个表示图形的原始字符串，描述了 quantized::conv1d 操作的计算图
  std::string quantized_conv1d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv1d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  // 定义字符串 quantized_conv1d_relu，包含一个表示图形的原始字符串，描述了 quantized::conv1d_relu 操作的计算图
  std::string quantized_conv1d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv1d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  // 定义字符串 conv2d，包含一个表示图形的原始字符串，描述了 aten::conv2d 操作的计算图
  std::string conv2d = R"(
// 定义了一个名为 conv2d_relu 的字符串，包含一个用于量化卷积后接 ReLU 激活函数的图形定义
std::string conv2d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)  // 对输入张量进行反量化操作
        %w_quant : Tensor, %b : Tensor? = quantized::conv2d_unpack(%packed_params)  // 解包量化的卷积参数
        %w_dequant = aten::dequantize(%w_quant)  // 对权重张量进行反量化操作
        %conv_out = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)  // 执行二维卷积操作
        %r = aten::relu(%conv_out)  // 对卷积结果应用 ReLU 激活函数
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)  // 将激活后的张量重新量化
        return (%r_quant) )";  // 返回量化后的张量

// 定义了一个名为 conv2d_inplace_relu 的字符串，包含一个用于量化卷积后接原地操作的 ReLU 激活函数的图形定义
std::string conv2d_inplace_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)  // 对输入张量进行反量化操作
        %w_quant : Tensor, %b : Tensor? = quantized::conv2d_unpack(%packed_params)  // 解包量化的卷积参数
        %w_dequant = aten::dequantize(%w_quant)  // 对权重张量进行反量化操作
        %conv_out = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)  // 执行二维卷积操作
        %r = aten::relu_(%conv_out)  // 对卷积结果原地应用 ReLU 激活函数
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)  // 将激活后的张量重新量化
        return (%r_quant) )";  // 返回量化后的张量

// 定义了一个名为 quantized_conv2d_relu 的字符串，包含一个用于量化卷积后接 ReLU 激活函数的图形定义
std::string quantized_conv2d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv2d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)  // 执行量化卷积后接 ReLU 激活函数
        return (%r_quant) )";  // 返回量化后的张量

// 定义了一个名为 conv3d 的字符串，包含一个用于量化三维卷积的图形定义
std::string conv3d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)  // 对输入张量进行反量化操作
        %w_quant : Tensor, %b : Tensor? = quantized::conv3d_unpack(%packed_params)  // 解包量化的卷积参数
        %w_dequant = aten::dequantize(%w_quant)  // 对权重张量进行反量化操作
        %r = aten::conv3d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)  // 执行三维卷积操作
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)  // 将卷积结果重新量化
        return (%r_quant) )";  // 返回量化后的张量
// 对输入张量 %a_quant 进行反量化操作，将其转换为浮点数张量 %a_dequant
%a_dequant = aten::dequantize(%a_quant)

// 调用 quantized::conv3d_unpack 函数解包量化的卷积参数 %packed_params，
// 返回量化权重张量 %w_quant 和可选的偏置张量 %b
%w_quant : Tensor, %b : Tensor? = quantized::conv3d_unpack(%packed_params)

// 对量化的权重张量 %w_quant 进行反量化操作，将其转换为浮点数张量 %w_dequant
%w_dequant = aten::dequantize(%w_quant)

// 调用 aten::conv3d 函数执行三维卷积操作，使用反量化后的输入张量 %a_dequant、
// 反量化后的权重张量 %w_dequant 和偏置张量 %b，指定卷积的步长 %stride、填充 %padding、
// 空洞卷积参数 %dilation 和分组数 %groups
%conv_out = aten::conv3d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)

// 对卷积结果 %conv_out 应用 ReLU 激活函数，得到张量 %r
%r = aten::relu(%conv_out)

// 将浮点数张量 %r 量化为 %r_quant 张量，使用指定的缩放因子 %r_scale、
// 零点 %r_zero_point 和数据类型 %r_dtype
%r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)

// 返回量化后的输出张量 %r_quant
return (%r_quant);



// 以下每个代码段的注释和示例注释类似，只需重复上述模式即可
// 定义一个名为 graph 的函数图形，接受多个输入参数 %a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %output_padding, %groups, %dilation
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %output_padding, %groups, %dilation):
        // 对输入 %a_quant 进行反量化操作，将量化的张量转换为浮点数张量 %a_dequant
        %a_dequant = aten::dequantize(%a_quant)
        // 调用 quantized::conv_transpose2d_unpack 函数解包 %packed_params，返回量化权重张量 %w_quant 和可选的偏置张量 %b
        %w_quant : Tensor, %b : Tensor? = quantized::conv_transpose2d_unpack(%packed_params)
        // 对量化的权重张量 %w_quant 进行反量化操作，将其转换为浮点数张量 %w_dequant
        %w_dequant = aten::dequantize(%w_quant)
        // 调用 aten::conv_transpose2d 函数执行转置卷积操作，接受输入 %a_dequant、权重 %w_dequant、偏置 %b 等参数
        %r = aten::conv_transpose2d(%a_dequant, %w_dequant, %b, %stride, %padding, %output_padding, %groups, %dilation)
        // 对输出 %r 进行量化操作，使用比例 %r_scale、零点 %r_zero_point 和数据类型 %r_dtype
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        // 返回量化后的结果 %r_quant
        return (%r_quant) )";

// 定义一个名为 quantized_conv_transpose2d 的函数图形，执行量化转置卷积操作
std::string quantized_conv_transpose2d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %output_padding, %groups, %dilation):
        // 调用 quantized::conv_transpose2d 函数执行量化转置卷积，返回量化输出 %r_quant
        %r_quant = quantized::conv_transpose2d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        // 返回量化结果 %r_quant
        return (%r_quant) )";

// 定义一个名为 add_relu 的函数图形，执行加法和 ReLU 激活操作
std::string add_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         // 对输入 %a_quant 进行反量化操作，将其转换为浮点数张量 %a_dequant
         %a_dequant = aten::dequantize(%a_quant)
         // 对输入 %b_quant 进行反量化操作，将其转换为浮点数张量 %b_dequant
         %b_dequant = aten::dequantize(%b_quant)
         // 执行张量加法 %a_dequant + %b_dequant，并添加缩放因子 %alpha
         %r_add = aten::add(%a_dequant, %b_dequant, %alpha)
         // 对加法结果 %r_add 执行 ReLU 激活函数
         %r_relu = aten::relu(%r_add)
         // 对 ReLU 结果 %r_relu 进行量化操作，使用比例 %scale、零点 %zero_point 和数据类型 %dtype
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         // 返回量化结果 %r
         return (%r) )";

// 定义一个名为 add_inplace_relu 的函数图形，执行原地加法和原地 ReLU 激活操作
std::string add_inplace_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         // 对输入 %a_quant 进行反量化操作，将其转换为浮点数张量 %a_dequant
         %a_dequant = aten::dequantize(%a_quant)
         // 对输入 %b_quant 进行反量化操作，将其转换为浮点数张量 %b_dequant
         %b_dequant = aten::dequantize(%b_quant)
         // 执行原地张量加法 %a_dequant += %b_dequant，并添加缩放因子 %alpha
         %r_add = aten::add(%a_dequant, %b_dequant, %alpha)
         // 对原地加法结果 %r_add 执行原地 ReLU 激活函数
         %r_relu = aten::relu_(%r_add)
         // 对原地 ReLU 结果 %r_relu 进行量化操作，使用比例 %scale、零点 %zero_point 和数据类型 %dtype
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         // 返回量化结果 %r
         return (%r) )";

// 定义一个名为 inplace_add_relu 的函数图形，执行原地加法和 ReLU 激活操作
std::string inplace_add_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         // 对输入 %a_quant 进行反量化操作，将其转换为浮点数张量 %a_dequant
         %a_dequant = aten::dequantize(%a_quant)
         // 对输入 %b_quant 进行反量化操作，将其转换为浮点数张量 %b_dequant
         %b_dequant = aten::dequantize(%b_quant)
         // 执行原地张量加法 %a_dequant += %b_dequant，并添加缩放因子 %alpha
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         // 对原地加法结果 %r_add 执行 ReLU 激活函数
         %r_relu = aten::relu(%r_add)
         // 对 ReLU 结果 %r_relu 进行量化操作，使用比例 %scale、零点 %zero_point 和数据类型 %dtype
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         // 返回量化结果 %r
         return (%r) )";

// 定义一个名为 inplace_add_inplace_relu 的函数图形，执行原地加法和原地 ReLU 激活操作
std::string inplace_add_inplace_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         // 对输入 %a_quant 进行反量化操作，将其转换为浮点数张量 %a_dequant
         %a_dequant = aten::dequantize(%a_quant)
         // 对输入 %b_quant 进行反量化操作，将其转换为浮点数张量 %b_dequant
         %b_dequant = aten::dequantize(%b_quant)
         // 执行原地张量加法 %a_dequant += %b_dequant，并添加缩放因子 %alpha
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         // 对原地加法结果 %r_add 执行原地 ReLU 激活函数
         %r_relu = aten::relu_(%r_add)
         // 对原地 ReLU 结果 %r_relu 进行量化操作，使用比例 %scale、零点 %zero_point 和数据类型 %dtype
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         // 返回量化结果 %r
         return (%r) )";

// 定义一个名为 quantized_add_relu 的函数图形，执行量化加法和 ReLU 激活操作
std::string quantized_add_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         // 调用 quantized::add_relu 函数执行量化加法和 ReLU 激活操作，返回量化结果 %r
         %r = quantized::add_relu(%a_quant, %b_quant, %scale, %zero_point)
         // 返回量化结果 %r
         return (%r) )";

// 定义一个名为 linear 的函数图形，执行全连接层
// 定义了一个名为 quantized_add 的字符串，包含一个用于量化加法的 TorchScript 图形定义
std::string quantized_add = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         // 对输入张量 %a_quant 进行反量化操作，将其转换为浮点数张量 %a_dequant
         %a_dequant = aten::dequantize(%a_quant)
         // 对输入张量 %b_quant 进行反量化操作，将其转换为浮点数张量 %b_dequant
         %b_dequant = aten::dequantize(%b_quant)
         // 执行 TorchScript 中的原生加法操作 %r_add = %a_dequant + %b_dequant * %alpha
         %r_add = aten::add(%a_dequant, %b_dequant, %alpha)
         // 对加法结果 %r_add 进行量化操作，使用给定的缩放因子 %scale、零点 %zero_point 和数据类型 %dtype
         %r = aten::quantize_per_tensor(%r_add, %scale, %zero_point, %dtype)
         // 返回量化后的张量结果 %r
         return (%r) )";
// 定义一个字符串变量 `graph`，包含了 quantized::add 的图形表示
std::string graph = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %r = quantized::add(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";

// 定义一个字符串变量 `inplace_add`，包含了 aten::add_ 的图形表示
std::string inplace_add = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         %r = aten::quantize_per_tensor(%r_add, %scale, %zero_point, %dtype)
         return (%r) )";

// 调用自定义函数 getBinaryOpScalarFusionInfo 获取关于标量加法的信息，用于 aten::add
auto add_scalar = getBinaryOpScalarFusionInfo(
      "aten::add",
      {"%b_scalar", "%alpha"},
      "quantized::add_scalar",
      {"%b_scalar"},
      {aten_add_alpha_is_one, input_b_is_scalar});

// 调用自定义函数 getBinaryOpScalarFusionInfo 获取关于标量原位加法的信息，用于 aten::add_
auto add_scalar_out = getBinaryOpScalarFusionInfo(
      "aten::add_",
      {"%b_scalar", "%alpha"},
      "quantized::add_scalar_out",
      {"%b_scalar", "%a_quant"},
      {aten_add_alpha_is_one, input_b_is_scalar});

// 定义一个字符串变量 `quantized_add_scalar_relu_pattern`，包含了 quantized::add_scalar 与 aten::relu 的融合图形表示
auto quantized_add_scalar_relu_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_add = quantized::add_scalar(%a_quant, %b_scalar)
         %r = aten::relu(%r_add)
         return (%r) )";

// 定义一个字符串变量 `quantized_add_scalar_inplace_relu_pattern`，包含了 quantized::add_scalar 与 aten::relu_ 的融合图形表示
auto quantized_add_scalar_inplace_relu_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_add = quantized::add_scalar(%a_quant, %b_scalar)
         %r = aten::relu_(%r_add)
         return (%r) )";

// 定义一个字符串变量 `quantized_add_scalar_relu_replacement`，包含了 quantized::add_scalar_relu 的图形表示
auto quantized_add_scalar_relu_replacement = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::add_scalar_relu(%a_quant, %b_scalar)
         return (%r) )";

// 定义一个字符串变量 `quantized_add_scalar_relu_out_pattern`，包含了 quantized::add_scalar_out 与 aten::relu 的融合图形表示
auto quantized_add_scalar_relu_out_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_add = quantized::add_scalar_out(%a_quant, %b_scalar, %a_quant)
         %r = aten::relu(%r_add)
         return (%r) )";

// 定义一个字符串变量 `quantized_add_scalar_inplace_relu_out_pattern`，包含了 quantized::add_scalar_out 与 aten::relu_ 的融合图形表示
auto quantized_add_scalar_inplace_relu_out_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_add = quantized::add_scalar_out(%a_quant, %b_scalar, %a_quant)
         %r = aten::relu_(%r_add)
         return (%r) )";

// 定义一个字符串变量 `quantized_add_scalar_relu_out_replacement`，包含了 quantized::add_scalar_relu_out 的图形表示
auto quantized_add_scalar_relu_out_replacement = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::add_scalar_relu_out(%a_quant, %b_scalar, %a_quant)
         return (%r) )";

// 定义一个字符串变量 `batch_norm`，包含了 quantized::batch_norm 的图形表示
std::string batch_norm = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %a_dequant = aten::dequantize(%a_quant)
         %r_bn = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         %r = aten::quantize_per_tensor(%r_bn, %scale, %zero_point, %scalar_type)
         return (%r) )";

// 定义一个字符串变量 `quantized_batch_norm`，留空，准备用于 quantized::batch_norm 的图形表示
std::string quantized_batch_norm = R"(
// 定义一个名为 batch_norm 的字符串，包含一个描述性的图形操作
std::string batch_norm = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         // 对输入量化张量进行反量化操作
         %a_dequant = aten::dequantize(%a_quant)
         // 执行批归一化操作，使用给定的权重、偏置、均值、方差等参数
         %bn_out = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         // 应用 ReLU 激活函数到归一化输出上
         %relu = aten::relu(%bn_out)
         // 将 ReLU 输出量化为张量
         %r = aten::quantize_per_tensor(%relu, %scale, %zero_point, %scalar_type)
         return (%r) )";

// 定义一个名为 batch_norm_inplace_relu 的字符串，包含一个描述性的图形操作
std::string batch_norm_inplace_relu = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         // 对输入量化张量进行反量化操作
         %a_dequant = aten::dequantize(%a_quant)
         // 执行批归一化操作，使用给定的权重、偏置、均值、方差等参数
         %bn_out = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         // 原地应用 ReLU 激活函数到归一化输出上
         %relu = aten::relu_(%bn_out)
         // 将 ReLU 输出量化为张量
         %r = aten::quantize_per_tensor(%relu, %scale, %zero_point, %scalar_type)
         return (%r) )";

// 定义一个名为 quantized_batch_norm_relu 的字符串，包含一个描述性的图形操作
std::string quantized_batch_norm_relu = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         // 使用量化的批归一化并应用 ReLU 激活函数
         %r = quantized::batch_norm_relu(%a_quant, %weight, %bias, %mean, %var, %eps, %scale, %zero_point)
         return (%r) )";

// 定义一个名为 mul 的字符串，包含一个描述性的图形操作
std::string mul = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         // 对输入量化张量 %a_quant 进行反量化操作
         %a_dequant = aten::dequantize(%a_quant)
         // 对输入量化张量 %b_quant 进行反量化操作
         %b_dequant = aten::dequantize(%b_quant)
         // 执行元素级乘法操作
         %r_mul = aten::mul(%a_dequant, %b_dequant)
         // 将乘法结果量化为张量
         %r = aten::quantize_per_tensor(%r_mul, %scale, %zero_point, %dtype)
         return (%r) )";

// 定义一个名为 inplace_mul 的字符串，包含一个描述性的图形操作
std::string inplace_mul = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         // 对输入量化张量 %a_quant 进行反量化操作
         %a_dequant = aten::dequantize(%a_quant)
         // 对输入量化张量 %b_quant 进行反量化操作
         %b_dequant = aten::dequantize(%b_quant)
         // 原地执行元素级乘法操作
         %r_mul = aten::mul_(%a_dequant, %b_dequant)
         // 将乘法结果量化为张量
         %r = aten::quantize_per_tensor(%r_mul, %scale, %zero_point, %dtype)
         return (%r) )";

// 定义一个名为 quantized_mul 的字符串，包含一个描述性的图形操作
std::string quantized_mul = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         // 使用量化的乘法操作
         %r = quantized::mul(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";

// 定义名为 mul_scalar 的变量，获取二元操作的标量融合信息
auto mul_scalar = getBinaryOpScalarFusionInfo(
      "aten::mul",
      {"%b_scalar"},
      "quantized::mul_scalar",
      {"%b_scalar"},
      {input_b_is_scalar});

// 定义名为 mul_scalar_out 的变量，获取二元操作的原地标量融合信息
auto mul_scalar_out = getBinaryOpScalarFusionInfo(
      "aten::mul_",
      {"%b_scalar"},
      "quantized::mul_scalar_out",
      {"%b_scalar", "%a_quant"},
      {input_b_is_scalar});

// 定义一个名为 mul_relu 的字符串，包含一个描述性的图形操作，未完整提供
std::string mul_relu = R"(
// 定义一个字符串，表示执行量化乘法、ReLU和量化的图模式
std::string mul_inplace_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)  // 对输入张量 %a_quant 进行去量化操作
         %b_dequant = aten::dequantize(%b_quant)  // 对输入张量 %b_quant 进行去量化操作
         %r_mul = aten::mul(%a_dequant, %b_dequant)  // 使用 ATen 的乘法操作，计算量化后的张量乘积 %r_mul
         %r_relu = aten::relu_(%r_mul)  // 对乘积结果 %r_mul 执行原地操作的 ReLU 激活函数
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)  // 对ReLU后的张量进行量化操作
         return (%r) )";  // 返回量化后的结果张量

// 定义一个字符串，表示执行量化乘法、原地ReLU和量化的图模式
std::string inplace_mul_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)  // 对输入张量 %a_quant 进行去量化操作
         %b_dequant = aten::dequantize(%b_quant)  // 对输入张量 %b_quant 进行去量化操作
         %r_mul = aten::mul_(%a_dequant, %b_dequant)  // 使用 ATen 的原地乘法操作，计算量化后的张量乘积 %r_mul
         %r_relu = aten::relu(%r_mul)  // 对乘积结果 %r_mul 执行 ReLU 激活函数
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)  // 对ReLU后的张量进行量化操作
         return (%r) )";  // 返回量化后的结果张量

// 定义一个字符串，表示执行量化乘法、原地ReLU和量化的图模式
std::string inplace_mul_inplace_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)  // 对输入张量 %a_quant 进行去量化操作
         %b_dequant = aten::dequantize(%b_quant)  // 对输入张量 %b_quant 进行去量化操作
         %r_mul = aten::mul_(%a_dequant, %b_dequant)  // 使用 ATen 的原地乘法操作，计算量化后的张量乘积 %r_mul
         %r_relu = aten::relu_(%r_mul)  // 对乘积结果 %r_mul 执行原地操作的 ReLU 激活函数
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)  // 对ReLU后的张量进行量化操作
         return (%r) )";  // 返回量化后的结果张量

// 定义一个字符串，表示执行量化乘法和ReLU的图模式
std::string quantized_mul_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %r = quantized::mul_relu(%a_quant, %b_quant, %scale, %zero_point)  // 使用量化操作进行乘法并对结果执行ReLU
         return (%r) )";  // 返回ReLU后的结果张量

// 定义一个字符串，表示执行量化乘法和标量ReLU的图模式
auto quantized_mul_scalar_relu_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_mul = quantized::mul_scalar(%a_quant, %b_scalar)  // 使用量化操作进行乘法
         %r = aten::relu(%r_mul)  // 对乘法结果执行ReLU激活函数
         return (%r) )";  // 返回ReLU后的结果张量

// 定义一个字符串，表示执行量化乘法和原地标量ReLU的图模式
auto quantized_mul_scalar_inplace_relu_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_mul = quantized::mul_scalar(%a_quant, %b_scalar)  // 使用量化操作进行乘法
         %r = aten::relu_(%r_mul)  // 对乘法结果执行原地操作的ReLU激活函数
         return (%r) )";  // 返回ReLU后的结果张量

// 定义一个字符串，表示替换量化乘法和标量ReLU操作的图模式
auto quantized_mul_scalar_relu_replacement = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::mul_scalar_relu(%a_quant, %b_scalar)  // 使用量化操作进行乘法并对结果执行ReLU
         return (%r) )";  // 返回ReLU后的结果张量

// 定义一个字符串，表示执行量化乘法和标量ReLUOut的图模式
auto quantized_mul_scalar_relu_out_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_mul = quantized::mul_scalar_out(%a_quant, %b_scalar, %a_quant)  // 使用量化操作进行乘法
         %r = aten::relu(%r_mul)  // 对乘法结果执行ReLU激活函数
         return (%r) )";  // 返回ReLU后的结果张量

// 定义一个字符串，表示执行量化乘法和原地标量ReLUOut的图模式
auto quantized_mul_scalar_inplace_relu_out_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_mul = quantized::mul_scalar_out(%a_quant, %b_scalar, %a_quant)  // 使用量化操作进行乘法
         %r = aten::relu_(%r_mul)  // 对乘法结果执行原地操作的ReLU激活函数
         return (%r) )";  // 返回ReLU后的结果张量

// 定义一个字符串，表示替换量化乘法和标量ReLUOut操作的图模式
auto quantized_mul_scalar_relu_out_replacement = R"(
// 定义一个字符串，表示量化乘法操作并应用ReLU激活函数
std::string graph(%a_quant, %b_scalar):
         %r = quantized::mul_scalar_relu_out(%a_quant, %b_scalar, %a_quant)
         return (%r) )";

// 定义一个字符串，表示量化ELU操作
std::string elu = R"(
graph(%a_quant, %alpha, %scale, %input_scale, %r_scale, %r_zero_point, %r_dtype):
         // 将输入量化数据反量化
         %a_dequant = aten::dequantize(%a_quant)
         // 应用ELU激活函数
         %r = aten::elu(%a_dequant, %alpha, %scale, %input_scale)
         // 将结果重新量化
         %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
         return (%r_quant) )";

// 定义一个字符串，表示量化ELU操作
std::string quantized_elu = R"(
graph(%a_quant, %alpha, %scale, %input_scale, %r_scale, %r_zero_point, %r_dtype):
         // 使用量化ELU操作
         %r_quant = quantized::elu(%a_quant, %r_scale, %r_zero_point, %alpha, %scale, %input_scale)
         return (%r_quant) )";

// 定义一个字符串，表示未完结的ELU操作
std::string elu_ = R"(
}

// 返回动态量化线性模式和替换信息的向量
inline std::vector<QuantFusionInfo>
dynamic_quantized_linear_pattern_and_replacements() {
  // 定义字符串，表示动态量化线性模式
  std::string linear_dynamic = R"(
graph(%packed_params, %a):
        // 解包量化线性参数
        %w_quant : Tensor, %b : Tensor? = quantized::linear_unpack(%packed_params)
        // 反量化权重
        %w_dequant = aten::dequantize(%w_quant)
        // 应用线性操作
        %r = aten::linear(%a, %w_dequant, %b)
        return (%r) )";

  // 忽略减少范围的模式
  // 设置减少范围的默认值为true，因为qnnpack后端忽略此参数
  std::string quantized_linear_dynamic = R"(
graph(%packed_params, %a):
        %reduce_range : bool = prim::Constant[value=1]()
        %r = quantized::linear_dynamic(%a, %packed_params, %reduce_range)
        return (%r) )";

  return {
      // 返回动态量化线性模式和替换信息的向量
      {"quantized::linear_dynamic",
       std::move(linear_dynamic),
       std::move(quantized_linear_dynamic)},
  };
}

// 返回动态量化融合模式和替换信息的向量
static std::vector<QuantFusionInfo>
dynamic_quant_fusion_pattern_and_replacements() {
  // 定义字符串，表示动态量化线性模式
  std::string linear_dynamic = R"(
graph(%packed_params, %a, %reduce_range, %a_dtype):
        // 选择每个张量的量化参数
        %a_scale : float, %a_zero_point : int = aten::_choose_qparams_per_tensor(%a, %reduce_range)
        // 对输入进行量化
        %a_quant = aten::quantize_per_tensor(%a, %a_scale, %a_zero_point, %a_dtype)
        // 反量化输入
        %a_dequant = aten::dequantize(%a_quant)
        // 解包量化线性参数
        %w_quant : Tensor, %b : Tensor? = quantized::linear_unpack(%packed_params)
        // 反量化权重
        %w_dequant = aten::dequantize(%w_quant)
        // 应用线性操作
        %r = aten::linear(%a_dequant, %w_dequant, %b)
        return (%r) )";

  // 定义字符串，表示动态量化线性模式
  std::string quantized_linear_dynamic = R"(
graph(%packed_params, %a, %reduce_range, %a_dtype):
        %r = quantized::linear_dynamic(%a, %packed_params, %reduce_range)
        return (%r) )";

  // 定义字符串，表示使用fp16的动态量化线性模式
  std::string linear_dynamic_fp16 = R"(
graph(%packed_params, %a):
        // 解包使用fp16的量化线性参数
        %w_unpacked : Tensor, %b : Tensor? = quantized::linear_unpack_fp16(%packed_params)
        // 应用线性操作
        %r = aten::linear(%a, %w_unpacked, %b)
        return (%r) )";

  std::string quantized_linear_dynamic_fp16 = R"(
static std::vector<QuantFusionInfo> linear_prepack_unpack_patterns() {
  // 定义用于线性层预打包和解包的图模式字符串
  std::string linear_with_quant = R"(
graph(%a_dequant, %w_quant, %b):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::linear(%a_dequant, %w_dequant, %b)
        return (%r) )";

  // 定义带量化的线性层预打包和解包的图模式字符串
  std::string linear_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b):
        %packed_params = quantized::linear_prepack(%w_quant, %b)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::linear(%a_dequant, %w_dequant, %b_unpacked)
        return (%r) )";
  // 定义带FP16转换的线性层图模式字符串
  std::string linear_fp16_with_cast = R"(
graph(%w, %a_dq, %b):
        %fp16_tensor = aten::_saturate_weight_to_fp16(%w)
        %r = aten::linear(%a_dq, %fp16_tensor, %b)
        return (%r) )";
  // 定义带FP16预打包和解包的线性层图模式字符串
  std::string linear_fp16_with_prepack = R"(
graph(%w, %a_dq, %b):
        %packed_params = quantized::linear_prepack_fp16(%w, %b)
        %w_unpacked : Tensor, %b_unpacked : Tensor? = quantized::linear_unpack_fp16(%packed_params)
        %r = aten::linear(%a_dq, %w_unpacked, %b_unpacked)
        return (%r) )";

  // 返回包含线性层预打包和解包模式信息的向量
  return {
      {"linear_prepack_unpack",
       std::move(linear_with_quant),
       std::move(linear_with_quant_prepack)},
      {"linear_fp16_prepack_unpack",
       std::move(linear_fp16_with_cast),
       std::move(linear_fp16_with_prepack)},
  };
}

static std::vector<QuantFusionInfo> conv_prepack_unpack_patterns() {
  // 定义带量化的一维卷积层图模式字符串
  std::string conv1d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  // 定义带量化的一维卷积层预打包和解包的图模式字符串
  std::string conv1d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv1d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv1d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

  // 返回包含一维卷积层预打包和解包模式信息的向量
  return {
      {"conv1d_prepack_unpack",
       std::move(conv1d_with_quant),
       std::move(conv1d_with_quant_prepack)},
  };
}
  std::string conv2d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv2d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

// 定义名为 conv2d_with_quant_prepack 的字符串变量，包含了一个描述图形操作的字符串
// 输入参数包括 %a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups
// %packed_params: 使用 quantized::conv2d_prepack 函数打包输入的量化权重和偏置
// %w_quant_unpacked, %b_unpacked: 使用 quantized::conv2d_unpack 函数解包打包的参数
// %w_dequant: 对解包后的量化权重进行去量化操作
// %r: 调用 aten::conv2d 函数执行卷积操作
// 返回卷积操作结果 %r
// 定义 graph 函数，接受多个参数 %a_dequant, %w_quant, %b, %stride, %padding, %output_padding, %groups, %dilation
graph(%a_dequant, %w_quant, %b, %stride, %padding, %output_padding, %groups, %dilation):
        // 使用 aten::dequantize 函数对权重 %w_quant 进行反量化得到 %w_dequant
        %w_dequant = aten::dequantize(%w_quant)
        // 调用 aten::conv_transpose2d 函数进行反卷积操作，使用反量化后的权重 %w_dequant 和其他参数
        %r = aten::conv_transpose2d(%a_dequant, %w_dequant, %b, %stride, %padding, %output_padding, %groups, %dilation)
        // 返回结果 %r
        return (%r) )";

  // 定义 conv_transpose2d_with_quant_prepack 字符串，包含量化卷积反卷积的预打包操作
  std::string conv_transpose2d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %output_padding, %groups, %dilation):
        // 调用 quantized::conv_transpose2d_prepack 函数对权重 %w_quant 和其他参数进行打包
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv_transpose2d_prepack(%w_quant, %b, %stride, %padding, %output_padding, %dilation, %groups)
        // 调用 quantized::conv_transpose2d_unpack 函数解包打包后的参数
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv_transpose2d_unpack(%packed_params)
        // 使用 aten::dequantize 函数对解包后的权重 %w_quant_unpacked 进行反量化得到 %w_dequant
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        // 调用 aten::conv_transpose2d 函数进行反卷积操作，使用反量化后的权重 %w_dequant 和其他参数
        %r = aten::conv_transpose2d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %output_padding, %groups, %dilation)
        // 返回结果 %r
        return (%r) )";

  // 返回一个包含不同量化卷积和反卷积函数字符串的字典
  return {
      {"conv1d_prepack_unpack",
       std::move(conv1d_with_quant),
       std::move(conv1d_with_quant_prepack)},
      {"conv2d_prepack_unpack",
       std::move(conv2d_with_quant),
       std::move(conv2d_with_quant_prepack)},
      {"conv3d_prepack_unpack",
       std::move(conv3d_with_quant),
       std::move(conv3d_with_quant_prepack)},
      {"conv_transpose1d_prepack_unpack",
       std::move(conv_transpose1d_with_quant),
       std::move(conv_transpose1d_with_quant_prepack)},
      // 将 "conv_transpose2d_prepack_unpack" 键和 conv_transpose2d_with_quant_prepack 值添加到字典中
      {"conv_transpose2d_prepack_unpack",
       std::move(conv_transpose2d_with_quant),
       std::move(conv_transpose2d_with_quant_prepack)}};
}

// 结束 namespace jit
} // 结束 namespace torch
```