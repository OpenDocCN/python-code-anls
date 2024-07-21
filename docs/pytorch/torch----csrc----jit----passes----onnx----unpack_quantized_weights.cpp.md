# `.\pytorch\torch\csrc\jit\passes\onnx\unpack_quantized_weights.cpp`

```
// 引入 Torch 库中的头文件，用于处理量化权重的解包操作
#include <torch/csrc/jit/passes/onnx/unpack_quantized_weights.h>

// 引入 ATen 库中的头文件，用于处理量化参数的打包操作
#include <ATen/native/quantized/PackedParams.h>
// 引入 C10 库中的实用工具，用于生成范围
#include <c10/util/irange.h>
// 引入 Torch 库中的 IR 相关头文件，用于常量和 IR 解析器
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

// TODO: 在 https://github.com/pytorch/pytorch/pull/68693 合并后，切换到每个运算符的头文件

// 引入 ATen 库中的功能函数
#include <ATen/Functions.h>

// 使用 torch 命名空间
using ::c10::Dispatcher;
namespace torch {
namespace jit {
namespace onnx {

// 使用 c10::onnx 命名空间

}

// 获取量化操作输入的比例尺度。此函数包括两种情况：
// 1. 对于在操作签名中指定了输出比例尺度的操作，获取其输出比例尺度。
// 2. 对于在操作签名中未指定输出比例尺度的操作（如 quantized::relu），遍历图形获取其输入的比例尺度，
// 直到找到明确指定比例尺度的节点。
double getScaleFromInput(Node* input_node) {
  // 可选的比例尺度
  std::optional<IValue> scale;
  // 获取输入节点的名称
  std::string input_name = input_node->kind().toQualString();
  // 不需要比例尺度的操作集合
  std::unordered_set<std::string> noscale_ops = {
      "quantized::max_pool2d",
      "aten::max_pool2d",
      "aten::relu",
      "prim::ListUnpack",
      "aten::split_with_sizes",
      "quantized::nchw2nhwc",
      "quantized::nhwc2nchw",
      "aten::slice",
      "aten::avg_pool2d",
      "quantized::cat",
      "prim::ListConstruct",
      "aten::upsample_nearest2d",
      "aten::sigmoid",
      "aten::reshape"};

  // 处理 quantize_per_tensor 操作的情况
  if (input_name == "aten::quantize_per_tensor") {
    TORCH_CHECK(
        input_node->inputs().size() > 1,
        "aten::quantize_per_tensor expected scale to be 2nd input");
    // 获取比例尺度的 IValue，并转换为 double 返回
    scale = toIValue(input_node->inputs()[1]);
    return scale.value().toDouble();
  } else if (input_name == "quantized::linear") {
    // 处理 quantized::linear 操作的情况
    // %r = quantized::linear(%input, %packed_weight, %w_scale, %w_zero_point)
    TORCH_CHECK(
        input_node->inputs().size() > 2,
        "quantized::linear expected scale to be 3rd input");
    // 获取比例尺度的 IValue，并转换为 double 返回
    scale = toIValue(input_node->inputs()[2]);
    return scale.value().toDouble();
  } else if (input_name == "quantized::conv2d") {
    // 处理 quantized::conv2d 操作的情况
    // %r = quantized::conv2d(%input, %packed_weight, %w_scale, %w_zero_point)
    TORCH_CHECK(
        input_node->inputs().size() > 2,
        "quantized::conv2d expected scale to be 3rd input");
    auto num_inputs = input_node->inputs().size();
    // 获取比例尺度的 IValue，并转换为 double 返回
    scale = toIValue(input_node->inputs()[num_inputs - 2]);
    return scale.value().toDouble();
  } else if (input_name == "quantized::conv2d_relu") {
    // 处理 quantized::conv2d_relu 操作的情况
    // %r = quantized::conv2d_relu(%input, %packed_weight, %w_scale, %w_zero_point)
    TORCH_CHECK(
        input_node->inputs().size() > 2,
        "quantized::conv2d_relu expected scale to be 3rd input");
    auto num_inputs = input_node->inputs().size();
    // 获取比例尺度的 IValue，并转换为 double 返回
    scale = toIValue(input_node->inputs()[num_inputs - 2]);
    return scale.value().toDouble();
  }
  // 默认情况下返回未指定比例尺度，需在其他地方处理
  return -1.0;
}
    // 返回一个 QVariant 对象中尺度值的浮点表示
    return scale.value().toDouble();
  } else if (input_name == "quantized::add") {
    // 对于 quantized::add 操作，期望第三个输入为尺度值
    TORCH_CHECK(
        input_node->inputs().size() > 2,
        "quantized::add expected scale to be 3rd input");
    // 从输入节点中获取尺度值
    scale = toIValue(input_node->inputs()[2]);
    // 返回尺度值的浮点表示
    return scale.value().toDouble();
  } else if (input_name == "aten::sigmoid") {
    // 对于 _caffe2::Int8Sigmoid 操作，输出的尺度设置为 1.0/256
    // 输出的零点设置为 0（quint8 类型）
    return 1.0L / 256;
  }
  // 对于以下操作，尺度值不是操作签名的一部分，因此我们需要遍历图来从其输入中获取尺度值（当图中定义时）
  else if (noscale_ops.find(input_name) != noscale_ops.end()) {
    // 从操作的输入节点中获取尺度值
    return getScaleFromInput(input_node->inputs()[0]->node());
  }
  // 如果到达这里，则表示无法识别的量化操作符，无法计算 q_scale
  TORCH_INTERNAL_ASSERT(
      false,
      "Unrecognized quantized operator while trying to compute q_scale for operator ",
      input_name);
}

std::vector<Node*> CreateQuantizedWeights(
    std::shared_ptr<Graph>& graph,  // 使用智能指针共享的图形对象
    const at::Tensor& weight,       // 输入的权重张量
    int8_t* data,                   // 指向量化权重数据的指针
    const std::vector<int64_t>& shapes,  // 权重张量的形状
    const std::vector<int64_t>& strides) {  // 权重张量的步幅信息
  auto qscheme = weight.qscheme();  // 获取权重张量的量化方案
  std::vector<Node*> unpacked_wt;   // 存储解包后的权重节点的向量

  // Retrieve scales and zero_points. Their formats are different depending on
  // different weight qscheme.
  // 根据权重的量化方案不同，获取其对应的标度和零点。它们的格式因方案而异。
  std::vector<float> scale_data;          // 存储标度值的向量
  std::vector<int64_t> scale_shapes;      // 标度值的形状信息
  std::vector<int64_t> zero_point_data;   // 存储零点值的向量
  std::vector<int64_t> zero_point_shapes; // 零点值的形状信息
  std::vector<int64_t> axis_data;         // 存储轴信息的向量

  switch (qscheme) {
    case c10::kPerTensorAffine: {
      // Cast to float since ONNX (De)QuantizeLinear only supports float scale.
      // 转换为浮点数，因为 ONNX 的量化/反量化操作只支持浮点数标度。
      scale_data = {static_cast<float>(weight.q_scale())};  // 设置标度值
      scale_shapes = {1};  // 设置标度值的形状
      zero_point_data = {weight.q_zero_point()};  // 设置零点值
      zero_point_shapes = {1};  // 设置零点值的形状
      break;
    }
    case c10::kPerChannelAffine:
    case c10::kPerChannelAffineFloatQParams: {
      auto q_scales = weight.q_per_channel_scales();  // 获取每通道标度值
      auto* scale_data_raw = q_scales.const_data_ptr<double>();  // 获取原始标度值数据的指针
      scale_shapes = q_scales.sizes().vec();  // 设置标度值的形状
      TORCH_INTERNAL_ASSERT(
          scale_shapes.size() == 1,
          "quantized per channel scales are expected as 1-d array.");  // 断言检查标度值的形状是否为一维数组
      scale_data.resize(scale_shapes[0]);  // 调整标度值向量大小
      // Cast to float since ONNX (De)QuantizeLinear only supports float scale.
      // 转换为浮点数标度
      std::transform(
          scale_data_raw,
          scale_data_raw + scale_shapes[0],
          scale_data.begin(),
          [](double x) { return static_cast<float>(x); });  // 使用 lambda 表达式进行转换

      auto q_zero_points = weight.q_per_channel_zero_points();  // 获取每通道零点值
      auto* zero_point_data_raw = q_zero_points.const_data_ptr<int64_t>();  // 获取原始零点值数据的指针
      zero_point_shapes = q_zero_points.sizes().vec();  // 设置零点值的形状
      TORCH_INTERNAL_ASSERT(
          zero_point_shapes.size() == 1,
          "quantized per channel zero points are expected as 1-d array.");  // 断言检查零点值的形状是否为一维数组
      zero_point_data = std::vector<int64_t>(
          zero_point_data_raw, zero_point_data_raw + zero_point_shapes[0]);  // 设置零点值的向量
      axis_data = {weight.q_per_channel_axis()};  // 设置每通道的轴信息
      break;
    }
    default:
      TORCH_CHECK(
          false, "Unsupported qscheme for weight, got ", toString(qscheme));
  }



// 处理默认情况：如果不支持的量化方案，则抛出错误信息并终止程序
TORCH_CHECK(
    false, "Unsupported qscheme for weight, got ", toString(qscheme));



  Node* data_node = graph->create(prim::Constant);
  auto data_value =
      at::from_blob(
          data, c10::IntArrayRef(shapes), c10::IntArrayRef(strides), at::kChar)
          .to(at::kCPU);
  // 需要进行克隆操作，因为 at::from_blob 不会接管 data 的所有权
  data_node->t_(Symbol::attr("value"), data_value.clone());



// 创建代表数据的节点，并将数据值存储为常量
Node* data_node = graph->create(prim::Constant);
auto data_value =
    at::from_blob(
        data, c10::IntArrayRef(shapes), c10::IntArrayRef(strides), at::kChar)
        .to(at::kCPU);
data_node->t_(Symbol::attr("value"), data_value.clone());



  Node* scale_node = graph->create(prim::Constant);
  auto scale_value =
      at::from_blob(
          scale_data.data(), c10::IntArrayRef(scale_shapes), at::kFloat)
          .to(at::kCPU);
  scale_node->t_(Symbol::attr("value"), scale_value.clone());



// 创建代表尺度值的节点，并将尺度数据值存储为常量
Node* scale_node = graph->create(prim::Constant);
auto scale_value =
    at::from_blob(
        scale_data.data(), c10::IntArrayRef(scale_shapes), at::kFloat)
        .to(at::kCPU);
scale_node->t_(Symbol::attr("value"), scale_value.clone());



  Node* zero_point_node = graph->create(prim::Constant);
  auto zero_point_value =
      at::from_blob(
          zero_point_data.data(), c10::IntArrayRef(zero_point_shapes), at::kInt)
          .to(at::kCPU);
  zero_point_node->t_(Symbol::attr("value"), zero_point_value.clone());



// 创建代表零点值的节点，并将零点数据值存储为常量
Node* zero_point_node = graph->create(prim::Constant);
auto zero_point_value =
    at::from_blob(
        zero_point_data.data(), c10::IntArrayRef(zero_point_shapes), at::kInt)
        .to(at::kCPU);
zero_point_node->t_(Symbol::attr("value"), zero_point_value.clone());



  Node* axis_node = graph->create(prim::Constant);
  if (!axis_data.empty()) {
    auto axis_value =
        at::from_blob(
            axis_data.data(), c10::IntArrayRef(axis_data.size()), at::kLong)
            .to(at::kCPU);
    axis_node->t_(attr::value, axis_value.clone());
  } else {
    axis_node->output()->setType(NoneType::get());
  }



// 创建代表轴的节点，并将轴数据值存储为常量，若数据为空则设置节点输出类型为 NoneType
Node* axis_node = graph->create(prim::Constant);
if (!axis_data.empty()) {
  auto axis_value =
      at::from_blob(
          axis_data.data(), c10::IntArrayRef(axis_data.size()), at::kLong)
          .to(at::kCPU);
  axis_node->t_(attr::value, axis_value.clone());
} else {
  axis_node->output()->setType(NoneType::get());
}



  return {data_node, scale_node, zero_point_node, axis_node};



// 返回创建的节点集合，用于表示数据、尺度、零点和轴信息的常量节点
return {data_node, scale_node, zero_point_node, axis_node};
}

// 创建一个包含量化偏置的节点，返回指向该节点的指针
Node* CreateQuantizedBias(
    std::vector<float> data,                    // 传入的量化偏置数据
    std::shared_ptr<Graph>& graph,              // 指向图的智能指针
    const std::vector<int64_t>& shapes) {       // 量化偏置的形状信息
  Node* const_node_1 = graph->create(prim::Constant);  // 创建一个常量节点
  auto const_bias =                              // 从数据创建一个张量
      at::from_blob(data.data(), c10::IntArrayRef(shapes), at::kFloat)
          .to(at::kCPU);
  auto options = c10::TensorOptions().dtype(at::kFloat).device(at::kCPU);  // 设置张量选项
  at::Tensor const_bias_copy = at::empty(c10::IntArrayRef(shapes), options);  // 创建一个空张量作为副本
  const_bias_copy.copy_(const_bias);             // 将数据复制到副本张量
  const_node_1->t_(Symbol::attr("value"), const_bias_copy);  // 将副本张量设置为节点的值属性
  return const_node_1;                          // 返回创建的常量节点指针
}

// 创建一个包含整数元组的节点，返回指向该节点的指针
Node* createIntTuple(
    const std::vector<int64_t>& is,             // 整数元组的值
    std::shared_ptr<Graph>& graph) {            // 指向图的智能指针
  Node* const_node = graph->create(Symbol::onnx("Constant"));  // 创建一个 ONNX 常量节点
  const_node->is_(Symbol::attr("value"), is);    // 将整数元组设置为节点的值属性
  return const_node;                            // 返回创建的常量节点指针
}

// 创建一个包含整数的节点，返回指向该节点的指针
Node* createInt(int64_t i, std::shared_ptr<Graph>& graph) {
  Node* const_node = graph->create(Symbol::onnx("Constant"));  // 创建一个 ONNX 常量节点
  const_node->i_(Symbol::attr("value"), i);      // 将整数设置为节点的值属性
  return const_node;                            // 返回创建的常量节点指针
}

// 将量化权重转换为图中的节点
void ConvertQuantizedWeight(
    std::shared_ptr<Graph>& graph,               // 指向图的智能指针
    Node* node,                                 // 要处理的节点
    at::Tensor& weight) {                       // 传入的量化权重张量引用
  std::vector<int64_t> wt_sizes = weight.sizes().vec();  // 获取权重张量的尺寸信息
  std::vector<int64_t> wt_strides = weight.strides().vec();  // 获取权重张量的步幅信息
  // 移除 packed_params
  node->removeInput(1);                         // 移除节点的第二个输入

  auto* wt_data =
      reinterpret_cast<int8_t*>(weight.mutable_data_ptr<c10::qint8>());  // 获取权重数据的指针

  std::vector<Node*> unpacked_wt =              // 创建解压后权重节点的列表
      CreateQuantizedWeights(graph, weight, wt_data, wt_sizes, wt_strides);  // 调用函数创建解压后的权重节点
  graph->setInsertPoint(node);                  // 设置插入点为当前节点
  Node* quant_node = graph->create(prim::TupleConstruct);  // 创建一个元组构造节点
  for (auto* n : unpacked_wt) {                 // 遍历解压后权重节点列表
    n->insertBefore(node);                      // 在当前节点之前插入解压后权重节点
    quant_node->addInput(n->output());          // 将解压后权重节点的输出作为元组构造节点的输入
  }
  quant_node->insertBefore(node);               // 在当前节点之前插入元组构造节点
  node->insertInput(1, quant_node->output());   // 将元组构造节点的输出作为当前节点的第二个输入
}

// CONV1D 需要与 CONV 不同的解压方式，因为它最初被故意打包为 CONV2D。
// 参见：https://github.com/pytorch/pytorch/pull/38248
enum class QuantizedParamsType { CONV1D, CONV, LINEAR };

// 在 ONNX 传递之前调用此函数。通过模式匹配找到相关节点并提取 packed_params。
// 使用 c10::Dispatcher 将 packed_params 传递给适当的解压函数。使用 caffe2::Int8GivenTensorFill 节点将解压后的权重和偏置插入图中。
void unpackQuantizedWeightsHelper(
    std::shared_ptr<Graph>& graph,               // 指向图的智能指针
    std::map<std::string, IValue>& paramsDict,  // 参数字典
    const std::string& pattern,                 // 模式字符串
    const std::string& unpack_fn,               // 解压函数名称
    QuantizedParamsType params_type,             // 量化参数类型
    bool expect_output_padding = false) {        // 是否期望输出填充
  Graph pattern_graph;                          // 创建模式图
  std::unordered_map<std::string, Value*> vmap; // 创建值映射
  parseIR(pattern, &pattern_graph, vmap);       // 解析模式字符串为模式图，并创建值映射
  const auto& matches = findPatternMatches(pattern_graph, *graph);  // 查找模式图在主图中的匹配项

  for (const auto& match : matches) {           // 遍历所有匹配项
    auto match_vmap = match.values_map;         // 获取匹配项的值映射
    auto qlinear_node = match_vmap.at(vmap.at("r"))->node();  // 获取量化线性节点
    std::string quantized_weight =              // 获取量化权重的调试名称
        match_vmap.at(vmap.at("r"))->node()->inputs()[1]->debugName();
    // 在 paramsDict 中查找 quantized_weight 对应的项
    auto itr = paramsDict.find(quantized_weight);
    // 如果未找到，抛出运行时错误
    if (itr == paramsDict.end()) {
      throw std::runtime_error(
          "getValues: Quantized weight value not found amongst constant parameters.");
    }
    // 创建变量用于存储解压后的权重和可选的偏置
    at::Tensor unpacked_weight;
    std::optional<at::Tensor> bias;
    // 定义索引常量，用于访问 stride, padding, output_padding, dilation, groups
    constexpr int64_t stride_idx = 2;
    constexpr int64_t padding_idx = 3;
    int64_t output_padding_idx = 0;
    int64_t dilation_idx = 0;
    int64_t groups_idx = 0;
    // 如果期望有 output_padding，则设置相关索引，否则设置其他索引
    if (expect_output_padding) {
      output_padding_idx = 4;
      dilation_idx = 5;
      groups_idx = 6;
    } else {
      dilation_idx = 4;
      groups_idx = 5;
    }
    // 定义存储 stride, padding, dilation, output_padding 的可选列表
    std::optional<torch::List<int64_t>> stride, padding, dilation,
        output_padding;
    // 定义存储 groups 和 transpose 的可选整数
    std::optional<int64_t> groups;
    std::optional<int64_t> transpose;

    // 定义存储 stride, padding, dilation, output_padding 的整数列表
    torch::List<int64_t> stride_int, padding_int, dilation_int,
        output_padding_int;
    // 初始化 groups_int 和 transpose_int，这些变量将在后续代码中使用
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t groups_int;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t transpose_int;

    // 如果 bias 存在，解压权重和偏置，并进行相关操作
    if (bias.has_value()) {
      TORCH_INTERNAL_ASSERT(itr->second.isTensor());
      at::Tensor packed_weight = itr->second.toTensor();
      auto op = Dispatcher::singleton()
                    .findSchemaOrThrow(unpack_fn.c_str(), "")
                    .typed<std::tuple<at::Tensor, std::optional<at::Tensor>>(
                        at::Tensor)>();
      std::tie(unpacked_weight, bias) = op.call(packed_weight);
    }

    // 将解压后的权重应用到量化线性节点
    ConvertQuantizedWeight(graph, qlinear_node, unpacked_weight);

    // 添加偏置项
    at::Tensor original_bias;
    if (bias.has_value()) {
      original_bias = bias.value();
      original_bias.set_requires_grad(false);
    } else {
      // 如果没有偏置，创建一个与权重大小相同的零张量
      int64_t bias_size = unpacked_weight.size(0);
      original_bias =
          at::zeros(bias_size, unpacked_weight.options().dtype(at::kFloat));
    }

    // 获取输入值并进行断言，确保其为张量类型
    auto input_val = match_vmap.at(vmap.at("r"))->node()->inputs()[0];
    TORCH_INTERNAL_ASSERT(
        input_val->type()->isSubtypeOf(*TensorType::get()),
        "Unsupported input type. Expected TensorType, got ",
        input_val->type()->str());

    // 将偏置值存储为浮点数向量
    std::vector<float> bias_values(original_bias.numel());
    auto bias_data = original_bias.const_data_ptr<float>();
    for (const auto i : c10::irange(original_bias.numel())) {
      bias_values[i] = bias_data[i];
    }
    // 创建量化偏置节点，并将其插入到量化线性节点之前
    Node* bias_node =
        CreateQuantizedBias(bias_values, graph, original_bias.sizes().vec());
    bias_node->insertBefore(qlinear_node);
    // 对于量化线性输入，顺序为输入，权重，偏置，...
    // 因此偏置位于位置 2
    qlinear_node->insertInput(2, bias_node->output());

    // 添加卷积参数：stride, padding, dilation, groups, output_padding
    // 检查是否存在步长、填充、膨胀和分组值，并且如果期望存在输出填充则也检查输出填充值是否存在
    if (stride.has_value() && padding.has_value() && dilation.has_value() &&
        groups.has_value() &&
        (!expect_output_padding || output_padding.has_value())) {
      // 创建存储卷积操作所需整数参数的可选列表
      std::vector<std::optional<torch::List<int64_t>>> conv_ints_args;
      // 将步长、填充和（如存在）输出填充值加入列表
      conv_ints_args.push_back(stride);
      conv_ints_args.push_back(padding);
      if (expect_output_padding) {
        conv_ints_args.push_back(output_padding);
      }
      conv_ints_args.push_back(dilation);
      // 跳过（输入，权重，偏置）的参数位置偏移量
      const size_t arg_offset = 3;
      // 遍历整数参数列表
      for (const auto i : c10::irange(conv_ints_args.size())) {
        // 创建整数元组节点，用于存储整数参数的值，并插入到图中
        Node* ints_node =
            createIntTuple(conv_ints_args[i].value().vec(), graph);
        ints_node->insertBefore(qlinear_node);
        // 将整数元组节点的输出作为qlinear_node的输入参数之一
        qlinear_node->insertInput(arg_offset + i, ints_node->output());
      }
      // 创建分组值节点，并插入到图中
      Node* groups_node = createInt(groups.value(), graph);
      groups_node->insertBefore(qlinear_node);
      // 将分组值节点的输出作为qlinear_node的输入参数之一
      qlinear_node->insertInput(groups_idx + 1, groups_node->output());
    }
    // 获取当前图形块
    auto b = graph->block();
    // 构建值到参数字典的映射
    auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
    // 从映射中删除未使用的值
    eraseUnusedValuesFromMap(valsToParamsMap);
  }
}

static std::
    unordered_map<c10::ScalarType, c10::ScalarType, ScalarTypeHashFunction>
        qTypeToValType = {
            {c10::ScalarType::QInt8, c10::ScalarType::Char},
            {c10::ScalarType::QUInt8, c10::ScalarType::Byte},
            {c10::ScalarType::QInt32, c10::ScalarType::Int},
            {c10::ScalarType::QUInt4x2, c10::ScalarType::Byte},
};

// Unpack quantized tensor inputs into {value, scale, zero_point},
// Then create a prim::TupleConstruct node based on these three values.
void UnpackQuantizedTensorInputs(std::shared_ptr<Graph>& graph) {
  for (size_t index = 0; index < graph->inputs().size();) {
    auto g_input = graph->inputs()[index];
    // 获取图中输入节点的类型，确保是张量类型并获取标量类型
    TensorTypePtr shape_type = g_input->type()->cast<TensorType>();
    if (!shape_type || !shape_type->scalarType().has_value()) {
      index++;
      continue;
    }
    auto scalar_type = shape_type->scalarType().value();
    // 检查标量类型是否在映射表中
    if (qTypeToValType.find(scalar_type) == qTypeToValType.end()) {
      index++;
      continue;
    }
    // 获取输入节点的调试名称
    std::string input_name = g_input->debugName();
    // 插入值节点，使用量化类型到数值类型的映射
    auto input_value =
        graph->insertInput(index, input_name + "_value")
            ->setType(shape_type->withScalarType(qTypeToValType[scalar_type]));
    // 插入比例节点，类型在 torch/include/ATen/Operators.h 中定义
    auto input_scale =
        graph->insertInput(index + 1, input_name + "_scale")
            ->setType(TensorType::create(
                at::kDouble, at::kCPU, 0, /*requires_grad=*/c10::nullopt));
    // 插入零点节点，类型为长整型，不计算梯度
    auto input_zero_point =
        graph->insertInput(index + 2, input_name + "_zero_point")
            ->setType(TensorType::create(
                at::kLong, at::kCPU, 0, /*requires_grad=*/c10::nullopt));
    // 创建值节点的向量，用于构造元组
    std::vector<Value*> converted{input_value, input_scale, input_zero_point};
    // 在图中前置节点，创建包含上述值的元组
    auto input_tuple =
        graph->prependNode(graph->createTuple(converted))->output();
    // 替换原始的量化张量输入节点使用
    g_input->replaceAllUsesWith(input_tuple);
    // 删除原始的量化张量输入节点
    graph->eraseInput(index + converted.size());
    index += 3;
  }
}

// https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export
void UnpackQuantizedWeights(
    std::shared_ptr<Graph>& graph,
}

// Caffe2 expects quantized ops to be in NHWC format while pytorch inputs are in
// NCHW. This pass inserts permutes to convert from NCHW to NHWC before each
// conv op and add another permute from NHWC to NCHW after the conv op.
void insertPermutesHelper(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict,
    const std::string& pattern) {
  // 解析给定的模式字符串，创建模式图并映射节点
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  parseIR(pattern, &pattern_graph, vmap);

  // 查找模式图在主图中的所有匹配项
  const auto& matches = findPatternMatches(pattern_graph, *graph);

  // 遍历每个匹配项
  for (const auto& match : matches) {
    auto match_vmap = match.values_map;
    // 获取操作节点和其输入节点
    auto op_node = match_vmap.at(vmap.at("r"))->node();
    auto input_node = match_vmap.at(vmap.at("r"))->node()->inputs()[0]->node();
    // 创建一个新的节点 `permute_node_before`，用于执行从 NCHW 格式到 NHWC 格式的转换，
    // 输入为 `input_node` 的输出结果
    Node* permute_node_before = graph->create(
        Symbol::fromQualString("quantized::nchw2nhwc"), {input_node->output()});
    
    // 将 `permute_node_before` 节点插入到 `op_node` 节点之前
    permute_node_before->insertBefore(op_node);
    
    // 在 `op_node` 节点中移除第一个输入
    op_node->removeInput(0);
    
    // 将 `permute_node_before` 的输出设置为 `op_node` 的新的第一个输入
    op_node->insertInput(0, permute_node_before->output());
    
    // 创建一个新的节点 `permute_node_after`，用于执行从 NHWC 格式到 NCHW 格式的转换，
    // 输入为 `op_node` 的第一个输出结果
    Node* permute_node_after = graph->create(
        Symbol::fromQualString("quantized::nhwc2nchw"),
        {op_node->outputs()[0]});
    
    // 将 `permute_node_after` 节点插入到 `op_node` 节点之后
    permute_node_after->insertAfter(op_node);
    
    // 获取 `op_node` 的第一个输出
    auto v = op_node->outputs().at(0);
    
    // 用 `permute_node_after` 的第一个输出替换 `v` 所有的使用
    v->replaceAllUsesWith(permute_node_after->outputs().at(0));
    
    // 在 `permute_node_after` 节点中移除第一个输入
    permute_node_after->removeInput(0);
    
    // 向 `permute_node_after` 节点添加新的第一个输入 `v`
    permute_node_after->addInput(v);
void insertPermutes(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict) {
  // 定义量化卷积的计算图模板
  std::string qconv = R"(
  graph(%input, %weight, %bias, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point):
        %r = quantized::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";
  // 定义带ReLU的量化卷积的计算图模板
  std::string qconv_relu = R"(
  graph(%input, %weight, %bias, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point):
        %r = quantized::conv2d_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";
  // 定义量化转置卷积的计算图模板
  std::string qconv_transpose = R"(
  graph(%input, %weight, %bias, %stride, %padding, %dilation, %output_padding, %groups, %w_scale, %w_zero_point):
        %r = quantized::conv_transpose2d(%input, %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";

  // 调用辅助函数插入各种量化卷积计算图模板到主计算图中
  insertPermutesHelper(graph, paramsDict, qconv);
  insertPermutesHelper(graph, paramsDict, qconv_relu);
  insertPermutesHelper(graph, paramsDict, qconv_transpose);
  // 打印插入量化卷积后的计算图
  GRAPH_DUMP("After insertPermutes: ", graph);
}

} // namespace jit
} // namespace torch
```