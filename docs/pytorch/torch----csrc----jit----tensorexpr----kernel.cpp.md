# `.\pytorch\torch\csrc\jit\tensorexpr\kernel.cpp`

```
// 引入 Torch 的 JIT Tensorexpr 库中的头文件
#include <torch/csrc/jit/tensorexpr/kernel.h>

// 引入 ATen 库中的其他头文件
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/TensorGeometry.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/util/irange.h>

// 引入 Torch JIT 的日志相关头文件
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/mkldnn_rewrite.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>

// 引入 Tensorexpr 库中的分析、表达式等头文件
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/loopnest_randomization.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

// 使用 Torch JIT Tensorexpr 命名空间
using namespace torch::jit::tensorexpr;

// Torch JIT Tensorexpr 的额外命名空间
namespace torch::jit::tensorexpr {

// 构建错误信息的函数，如果输入为空则返回通用错误信息
std::string buildErrorMessage(const std::string& s) {
  static const std::string generic_error_message =
      "This error occurred in the fuser. You can turn off the fuser with "
      "torch.jit.enable_fusion(False).";
  if (s.empty()) {
    return generic_error_message;
  }
  if (s.back() == '.') {
    return s + " " + generic_error_message;
  }
  return s + ". " + generic_error_message;
}

// Tensorexpr CUDA 点对点操作的循环层级，默认为 -1
static int te_cuda_pointwise_loop_levels = -1;

// Tensorexpr CUDA 点对点操作的块计数，默认为 -1
static int te_cuda_pointwise_block_count = -1;

// Tensorexpr CUDA 点对点操作的块大小，默认为 -1
static int te_cuda_pointwise_block_size = -1;

// 是否允许回退的标志，默认为 false
static bool fallback_allowed = false;

// 是否生成块级别代码的标志，默认为 false
static bool te_generate_block_code = false;

// CPU 上是否必须使用 LLVM 的标志，默认为 true
static bool te_must_use_llvm_on_cpu = true;

// 是否合并无条件语句的标志，默认为 true
static bool cat_wo_conditionals = true; // NOLINT

// 是否优化条件语句的标志，默认为 false
static bool opt_conditionals = false; // NOLINT

// 设置是否允许回退操作，返回原先的值
bool setFallbackAllowed(bool value) {
  bool old_value = fallback_allowed;
  fallback_allowed = value;
  return old_value;
}

// 返回当前是否允许回退操作的状态
bool fallbackAllowed() {
  static const char* enable_c_str = std::getenv("PYTORCH_TENSOREXPR_FALLBACK");
  if (!enable_c_str) {
    return fallback_allowed;
  }
  if (std::string(enable_c_str) == "0") {
    return false;
  }
  return true;
}

// 返回是否强制执行回退操作的状态
static bool fallbackEnforced() {
  static const char* enable_c_str = std::getenv("PYTORCH_TENSOREXPR_FALLBACK");
  if (tensorexpr::getTEGenerateBlockCode()) {
    return false;
  }
  if (!enable_c_str) {
    return fallback_allowed;
  }
  if (std::string(enable_c_str) == "2") {
    return true;
  }
  return false;
}

// 返回请求的随机变换数量，从环境变量中读取
static int64_t randomTransformsRequested() {
  const char* enable_c_str =
      std::getenv("PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED");
  if (!enable_c_str) {
    return 0;
  }
  return std::stoi(std::string(enable_c_str));
}

// 如果启用 LLVM，则返回是否禁用 LLVM 的标志
#ifdef TORCH_ENABLE_LLVM
static bool dontUseLLVMFlag() {
  static const char* enable_c_str =
      std::getenv("PYTORCH_TENSOREXPR_DONT_USE_LLVM");
  if (!enable_c_str) {
    return false;
  }
  return std::string(enable_c_str) == "1";
}
#endif

// 返回 Tensorexpr CUDA 点对点操作的循环层级引用
int& getTECudaPointwiseLoopLevels() {
  return te_cuda_pointwise_loop_levels;
}
// 返回 TE CUDA pointwise 模块的块数量的引用
int& getTECudaPointwiseBlockCount() {
  return te_cuda_pointwise_block_count;
}

// 返回 TE CUDA pointwise 模块的块大小的引用
int& getTECudaPointwiseBlockSize() {
  return te_cuda_pointwise_block_size;
}

// 返回是否生成块代码的标志的引用
// TODO: 移除这个全局变量
// 理想情况下，块代码生成应该根据张量在设备上的类型来确定。
bool& getTEGenerateBlockCode() {
  return te_generate_block_code;
}

// 返回是否在 CPU 上必须使用 LLVM 的标志的引用
bool& getTEMustUseLLVMOnCPU() {
  return te_must_use_llvm_on_cpu;
}

// 返回是否在没有条件的情况下执行融合的标志的引用
bool& getCatWoConditionals() {
  return cat_wo_conditionals;
}

// 返回是否优化条件的标志的引用
bool& getOptConditionals() {
  return opt_conditionals;
}

// 根据输入的 Torch JIT 值数组选择设备类型
static std::optional<at::Device> pickDeviceType(
    const at::ArrayRef<torch::jit::Value*>& inputs) {
  std::optional<at::Device> device = c10::nullopt;
  for (auto const& input : inputs) {
    auto tt = input->type()->cast<TensorType>();
    if (tt && tt->device()) {
      if (device && *device != *tt->device()) {
        return c10::nullopt;
      }
      device = *tt->device();
    }
  }
  return device;
}

// 根据输入的 Torch JIT 图选择设备类型
static std::optional<at::Device> pickDeviceType(
    const std::shared_ptr<Graph>& graph) {
  std::optional<at::Device> device = c10::nullopt;
  for (auto const& node : graph->nodes()) {
    for (auto const& input : node->inputs()) {
      if (auto tt = input->type()->cast<TensorType>()) {
        if (auto inputDevice = tt->device()) {
          TORCH_INTERNAL_ASSERT(
              !device || *device == *inputDevice,
              buildErrorMessage(
                  "Different devices specified for inputs to the fuser."));
          device = inputDevice;
        }
      }
    }
  }
  for (auto const& input : graph->inputs()) {
    if (auto tt = input->type()->cast<TensorType>()) {
      if (auto inputDevice = tt->device()) {
        TORCH_INTERNAL_ASSERT(
            !device || *device == *inputDevice,
            buildErrorMessage(
                "Different devices specified for inputs to the fuser."));
        device = inputDevice;
      }
    }
  }
  if (!device) {
    // 默认情况下假设设备是 CPU
    device = at::kCPU;
  }
  return device;
}

// 如果 v 是具有具体已知大小和数据类型的 Tensor，则返回它们；否则返回 nullopt
static std::optional<TensorInfo> getTensorInfoJit(torch::jit::Value* v) {
  auto const& it = v->type()->cast<TensorType>();

  c10::ScalarType dtype = c10::ScalarType::Float;

  if (!it) {
    return c10::nullopt;
  }
  if (!it->isComplete()) {
    return c10::nullopt;
  }
  if (it->scalarType()) {
    // TODO: 理想情况下，我们应该在这里严格处理，如果 JIT IR 中缺少 dtype，则返回 nullopt。
    // 目前假设默认为 Float dtype，直到实现 dtype 传播。
    dtype = *it->scalarType();
  }
  auto concrete_sizes = it->sizes().concrete_sizes();
  if (!concrete_sizes) {
    return c10::nullopt;
  }
  return TensorInfo{*concrete_sizes, dtype};
}

// 将 IValue 转换为包含两个整数的 vector
static std::vector<int64_t> _pair_int(IValue v) {
  if (v.isIntList()) {
    return v.toIntVector();
  } else {
    return {v.toInt(), v.toInt()};
  }
}
// 检查给定的张量是否是连续的
bool isContiguous(const torch::jit::Value* v, at::MemoryFormat memory_format) {
  // 将值转换为张量类型
  auto const& tt = v->type()->cast<TensorType>();
  if (!tt) {  // 如果不是张量类型，返回 false
    return false;
  }
  if (!tt->isComplete()) {  // 如果张量类型不完整，返回 false
    return false;
  }
  auto const& sizes = tt->sizes().concrete_sizes();  // 获取张量的具体尺寸
  auto const& strides = tt->strides().concrete_sizes();  // 获取张量的具体步长
  if (!sizes || !strides) {  // 如果尺寸或步长未知，返回 false
    return false;
  }

  // 首先检查张量的维度大小
  int ndims = (*sizes).size();
  // 根据内存格式检查维度是否符合特定要求
  if ((memory_format == at::MemoryFormat::ChannelsLast && ndims != 4) ||
      (memory_format == at::MemoryFormat::ChannelsLast3d && ndims != 5)) {
    return false;
  }

  // 检查张量的步长是否符合连续的步长要求
  return *strides == TensorType::contiguousStridesOf(*sizes, memory_format);
}

// 获取卷积操作的分组索引
static size_t get_conv_groups_index(const torch::jit::Node* node) {
  switch (node->kind()) {
    case aten::conv2d:
      return 6;  // conv2d 操作的分组索引为 6
    case aten::_convolution:
      return 8;  // _convolution 操作的分组索引为 8
    default:
      // 如果操作不是 conv2d 或 _convolution，则抛出错误信息
      TORCH_CHECK(
          false,
          "mkldnnPrepackedConvIsSupportedJit expects node kind to be conv2d or _convolution but got ",
          node->kind());
  }
}

// 检查是否支持 JIT 环境下的 conv2d 操作
// - 静态形状：4 维输入和过滤器，1 维偏置。
// - 常量步长/填充/扩张/分组
// - 相等的填充和步长，扩张率为 1。
// - 深度卷积 (分组数等于输入通道数等于输出通道数)
// - 3x3 卷积核
bool conv2dIsSupportedJit(const torch::jit::Node* node) {
  // 获取输入、权重、偏置、步长、填充、扩张率和分组信息
  auto const& input = getTensorInfoJit(node->input(0));
  auto const& weight = getTensorInfoJit(node->input(1));
  auto const& bias = getTensorInfoJit(node->input(2));
  auto const& stride = toIValue(node->input(3));
  auto const& pad = toIValue(node->input(4));
  auto const& dilation = toIValue(node->input(5));
  size_t groups_index = get_conv_groups_index(node);
  auto const& groups = toIValue(node->input(groups_index));

  // 所有参数必须是静态已知的
  if (!input || !weight || !bias || !stride || !pad || !dilation || !groups) {
    GRAPH_DEBUG("some params aren't static");
    return false;
  }

  // 所有输入都应该是连续的，不需要转置
  if (!isContiguous(node->input(0)) || !isContiguous(node->input(1)) ||
      !isContiguous(node->input(2))) {
    GRAPH_DEBUG("conv2dIsSupported: some inputs are not contiguous");
    return false;
  }

  // 调用底层的 conv2dIsSupported 函数检查是否支持该 conv2d 操作
  return conv2dIsSupported(
      *input,
      *weight,
      *bias,
      _pair_int(*stride),
      _pair_int(*pad),
      _pair_int(*dilation),
      groups->toInt());
}

// 检查是否支持 JIT 环境下的 mkldnnPrepackedConv 操作
bool mkldnnPrepackedConvIsSupportedJit(const torch::jit::Node* node) {
#if AT_MKLDNN_ENABLED()
  // 获取输入节点0的张量信息
  auto const& input = getTensorInfoJit(node->input(0));
  // 获取输入节点1的张量信息
  auto const& weight = getTensorInfoJit(node->input(1));
  // 获取输入节点3的步长信息
  auto const& stride = toIValue(node->input(3));
  // 获取输入节点4的填充信息
  auto const& pad = toIValue(node->input(4));
  // 获取输入节点5的膨胀信息
  auto const& dilation = toIValue(node->input(5));
  // 获取卷积组的索引
  size_t groups_index = get_conv_groups_index(node);
  // 获取卷积组的信息
  auto const& groups = toIValue(node->input(groups_index));

  // 所有参数都应为静态已知（偏置可能是NoneType = prim::Constant()）。
  if (!input || !weight || !stride || !pad || !dilation || !groups) {
    GRAPH_DEBUG("some params aren't static");
    return false;
  }

  // 当使用mkldnn后端时，权重和偏置应为常数
  if (node->input(1)->node()->kind() != prim::Constant ||
      node->input(2)->node()->kind() != prim::Constant) {
    GRAPH_DEBUG(
        "mkldnnPrepackedConvIsSupported: weight or bias is not Constant");
    return false;
  }

  // 输入和权重应为NHWC连续
  if (!(isContiguous(node->input(0), at::MemoryFormat::ChannelsLast) &&
        isContiguous(node->input(1), at::MemoryFormat::ChannelsLast))) {
    GRAPH_DEBUG(
        "mkldnnPrepackedConvIsSupported: input or weight is not ChannelsLast contiguous");
    return false;
  }

  // 调用mkldnnPrepackedConvIsSupported函数来判断是否支持预打包的MKLDNN卷积
  return mkldnnPrepackedConvIsSupported(
      *input,
      *weight,
      _pair_int(*stride),
      _pair_int(*pad),
      _pair_int(*dilation),
      groups->toInt());
#endif
  // 如果未启用MKLDNN，则返回false
  return false;
}

bool isConv2d(const Node* node) {
  // 如果节点不是卷积操作，则返回false
  if (node->kind() != aten::_convolution) {
    return false;
  }

  // 获取步长、填充、膨胀、是否转置、输出填充的信息
  auto const& stride = toIValue(node->input(3));
  auto const& pad = toIValue(node->input(4));
  auto const& dilation = toIValue(node->input(5));
  auto const& transposed = toIValue(node->input(6));
  auto const& output_padding = toIValue(node->input(7));

  // 如果有参数不是静态已知的，则返回false
  if (!stride || !pad || !dilation || !transposed || !output_padding) {
    GRAPH_DEBUG("some params aren't static");
    return false;
  }

  // 如果步长、填充、膨胀、输出填充的维度不符合2D卷积的要求，则返回false
  if (stride.value().toIntList().size() != 2 ||
      pad.value().toIntList().size() != 2 ||
      dilation.value().toIntList().size() != 2 ||
      output_padding.value().toIntList().size() != 2) {
    GRAPH_DEBUG("Conv not 2d");
    return false;
  }

  // 如果是转置卷积，则返回false
  if (transposed.value().toBool()) {
    GRAPH_DEBUG("transposed Conv");
    return false;
  }
  // 符合2D卷积的所有条件，返回true
  return true;
}

// 当前融合器仅支持2D矩阵的矩阵乘法
bool matmulIsSupported(const torch::jit::Node* node) {
  // 获取输入节点0和节点1的张量信息
  auto const& input0 = getTensorInfoJit(node->input(0));
  auto const& input1 = getTensorInfoJit(node->input(1));

  // 如果输入形状未知，则返回false
  if (!input0 || !input1) {
    GRAPH_DEBUG("matmulIsSupported: Input shapes aren't static");
    return false;
  }

  // 如果张量的维度不是2，则返回false
  if (input0->dims.size() != 2 || input1->dims.size() != 2) {
    GRAPH_DEBUG("matmulIsSupported: Unsupported input sizes");
    return false;
  }

  // 符合矩阵乘法支持的所有条件，返回true
    return false;
  }

  // 如果输入张量不是连续的，TensorEngine 将不必要地进行转置操作。
  if (!isContiguous(node->input(0)) || !isContiguous(node->input(1))) {
    // 打印调试信息，指出输入张量的形状不是连续的
    GRAPH_DEBUG("matmulIsSupported: Input shapes are not contiguous");
    return false;
  }

  // 如果以上条件都不满足，则说明矩阵乘法操作是支持的
  return true;
} // 结束命名空间 torch::jit::tensorexpr

} // namespace torch::jit::tensorexpr

static at::ScalarType tensorType(BufPtr b) {
  // 将 BufPtr 对象的数据类型转换为 at::ScalarType 类型并返回
  return static_cast<at::ScalarType>(b->dtype().scalar_type());
}

ExprHandle TensorExprKernel::constant(const torch::jit::Value* v) {
  // 如果给定值是常量节点
  if (v->node()->kind() == prim::Constant) {
    // 将其转换为 IValue 类型
    auto val = toIValue(v).value();
    // 根据值的类型创建相应的常量表达式
    if (val.isDouble()) {
      return DoubleImm::make(val.toDouble()); // 创建双精度常量表达式
    } else if (val.isInt()) {
      return LongImm::make(val.toInt());     // 创建长整型常量表达式
    } else if (val.isBool()) {
      return BoolImm::make(val.toBool());    // 创建布尔型常量表达式
    } else if (val.isNone()) {
      // 这只是一个占位符，避免抛出异常。None 的处理在操作特定的降低代码中应适当处理。
      return IntImm::make(0);                // 创建整型常量表达式 0
    } else {
      throw unsupported_dtype();             // 不支持的数据类型，抛出异常
    }
  }

  // 如果给定值不在 scalars_ 中，抛出异常
  if (!scalars_.count(v)) {
    throw malformed_input("no scalar in Constant");
  }

  // 返回 scalars_ 中给定值对应的表达式句柄
  return scalars_.at(v);
}

ArgValue TensorExprKernel::toArg(const torch::jit::Value* v) const {
  // 如果给定值在 scalars_ 中
  auto vi = scalars_.find(v);
  if (vi != scalars_.end()) {
    return VarHandle(vi->second); // 返回对应的变量句柄
  }
  
  // 如果给定值在 bufs_ 中
  auto ti = bufs_.find(v);
  if (ti != bufs_.end()) {
    return BufHandle(ti->second); // 返回对应的缓冲区句柄
  }
  
  // 如果给定值是 prim::ListConstruct 类型
  if (v->node()->kind() == prim::ListConstruct) {
    std::vector<ArgValue> vec;
    // 遍历列表构造节点的输入
    for (auto el : v->node()->inputs()) {
      vec.push_back(toArg(el)); // 递归调用 toArg 处理每个输入，并加入到 vec 中
    }
    // 如果 vec 是空的，返回 BufList 类型
    if (vec.empty()) {
      return BufList(); // 返回任意类型的向量
    } else if (std::get_if<BufHandle>(&vec[0])) {
      return convertVecArgValue<BufHandle>(vec); // 将 vec 转换为 BufHandle 类型的 ArgValue
    } else if (std::get_if<int64_t>(&vec[0])) {
      return convertVecArgValue<int64_t>(vec);  // 将 vec 转换为 int64_t 类型的 ArgValue
    }
    throw unsupported_dtype(); // 不支持的数据类型，抛出异常
  }
  
  // 如果给定值是 prim::Constant 类型
  if (v->node()->kind() == prim::Constant) {
    auto val = toIValue(v).value(); // 将其转换为 IValue 类型
    // 根据值的类型返回相应的 ArgValue
    if (val.isDouble()) {
      return val.toDouble();          // 返回双精度值
    } else if (val.isInt()) {
      return val.toInt();             // 返回整型值
    } else if (val.isBool()) {
      return val.toBool();            // 返回布尔值
    } else if (val.isNone()) {
      // 这只是一个占位符，避免抛出异常。None 的处理在操作特定的降低代码中应适当处理。
      return ArgNone();               // 返回空值
    } else if (val.isIntList()) {
      return val.toIntVector();       // 返回整型向量
    } else if (val.isDoubleList()) {
      return val.toDoubleVector();    // 返回双精度向量
    } else if (val.isString()) {
      return val.toStringRef();       // 返回字符串引用
    } else {
      throw unsupported_dtype(val.type()->str()); // 不支持的数据类型，抛出异常
    }
  }
  
  // 如果给定值不在 scalars_ 中，抛出异常
  if (!scalars_.count(v)) {
    throw malformed_input("no scalar in Constant");
  }
  
  // 返回 scalars_ 中给定值对应的 ArgValue
  return scalars_.at(v);
}

ExprHandle TensorExprKernel::getVarForShape(const c10::ShapeSymbol& ss) {
  // 如果 ShapeSymbol 是静态的
  if (ss.is_static()) {
    return LongImm::make(ss.static_size()); // 返回静态大小的长整型常量表达式
  }
  
  // 否则，根据 ShapeSymbol 的值查找或创建相应的变量句柄
  auto value = ss.value();
  auto it = shapeSymbolToVar_.find(value);
  if (it == shapeSymbolToVar_.end()) {
    // 创建新的 ShapeSymbol 对应的变量句柄，并将其存储在 shapeSymbolToVar_ 中
    VarHandle var("ss" + std::to_string(-value), kLong);
    shapeSymbolToVar_.emplace(value, var);
    return std::move(var); // 返回新创建的变量句柄
  }
  
  // 返回已经存在的 ShapeSymbol 对应的变量句柄
  return it->second;
}
// 从符号形状中提取尺寸信息并返回表达式句柄向量
std::vector<ExprHandle> TensorExprKernel::sizesFromSymbolicShape(
    const c10::SymbolicShape& shape) {
  // 创建一个空的维度向量
  std::vector<ExprHandle> dims;
  // 获取符号形状的可能的秩（rank）
  auto maybe_rank = shape.rank();
  // 断言秩的存在性
  TORCH_INTERNAL_ASSERT(maybe_rank);
  // 解引用秩值
  auto rank = *maybe_rank;
  // 遍历秩范围内的索引
  for (const auto i : c10::irange(rank)) {
    // 获取当前形状索引对应的变量并添加到维度向量中
    dims.push_back(getVarForShape(shape[i]));
  }
  // 返回构建好的维度向量
  return dims;
}

// 根据值的类型推断其尺寸信息并返回表达式句柄向量
std::vector<ExprHandle> TensorExprKernel::sizesForValue(
    const torch::jit::Value* v) {
  // 如果已知该值的尺寸信息，则直接返回已知的尺寸向量
  if (known_sizes_.count(v)) {
    return known_sizes_.at(v);
  }

  // 如果值的类型是张量类型，则从类型信息中提取符号尺寸信息
  if (v->type()->kind() == TypeKind::TensorType) {
    auto tt = v->type()->cast<TensorType>();
    return sizesFromSymbolicShape(tt->symbolic_sizes());
  }

  // 如果值的类型是浮点型、布尔型或整型，则返回空向量
  if (v->type()->isSubtypeOf(*FloatType::get()) ||
      v->type()->isSubtypeOf(*BoolType::get()) ||
      v->type()->isSubtypeOf(*IntType::get())) {
    return {};
  }
  
  // 如果值的类型是None类型，则返回空向量
  if (v->type()->isSubtypeOf(*NoneType::get())) {
    return {};
  }
  
  // 输出调试信息，表示未知节点的尺寸
  GRAPH_DEBUG("Unknown sizes for the node: ", *v->node());
  // 输出完整的融合组图形信息
  GRAPH_DEBUG("Full fusion group graph:\n", *v->node()->owningGraph());
  // 抛出异常，表示输入有误
  std::string msg = std::string("Unhandled node kind (in sizesForValue): ") +
      v->node()->kind().toQualString();
  throw malformed_input(msg);
}

// 查找值的数据类型并返回可选的标量类型
static std::optional<ScalarType> findDtypeForValue(const torch::jit::Value* v) {
  // 如果值的类型是张量类型，则尝试获取其标量类型
  if (v->type()->kind() == TypeKind::TensorType) {
    auto tt = v->type()->cast<TensorType>();
    if (tt->scalarType()) {
      return static_cast<ScalarType>(*tt->scalarType());
    }
  }
  // 从JIT类型中尝试获取标量类型
  return tryScalarTypeFromJitType(*v->type());
}

// 将常量零维张量作为标量参数处理
static bool constZeroDimTensorAsScalarArg(
    const Value* v,
    std::vector<ArgValue>& args) {
  // 如果值的节点类型不是常量或者不是张量类型，则返回false
  if (v->node()->kind() != prim::Constant || !v->type()->cast<TensorType>()) {
    return false;
  }

  // 将值转换为IValue并获取其张量表示
  const auto t = toIValue(v)->toTensor();
  // 如果张量的尺寸不为空，则返回false
  if (!t.sizes().empty()) {
    return false;
  }

  // 将张量的数据类型转换为C10的标量类型
  c10::ScalarType dtype = c10::typeMetaToScalarType(t.dtype());
  // 根据数据类型执行不同的操作
  switch (dtype) {
    // 如果是浮点型，将其作为浮点数添加到参数向量中
    case ScalarType::Float:
      args.emplace_back(t.item().toFloat());
      return true;
    // 如果是长整型，将其作为长整数添加到参数向量中
    case ScalarType::Long:
      args.emplace_back(t.item().toLong());
      return true;
    // 否则，抛出不支持的数据类型异常
    default:
      std::stringstream ss;
      ss << "Unsupported tensor dtype:" << dtype
         << " for converting constant 0-dim Tensor to scalar" << std::endl;
      throw unsupported_dtype(ss.str());
  }
}

// 计算给定值的张量表达式
Tensor TensorExprKernel::computeValue(const torch::jit::Value* v) {
  // 获取节点的输入和操作类型
  auto inputs = v->node()->inputs();
  auto op = v->node()->kind();

  // 如果操作是aten::rand_like，则设置标志表示存在随机数
  if (op == aten::rand_like) {
    hasRandom_ = true;
  }

  // 查找值的数据类型
  auto outputType = findDtypeForValue(v);
  // 获取值的输出形状
  std::vector<ExprHandle> outputShape = sizesForValue(v);
  // 初始化输出步长为空向量
  std::vector<ExprHandle> outputStrides = {};

  // 如果内存布局策略是kChannelsLastNdContiguous，则生成通道最后布局的步长
  if (memory_layout_policy_ == MemoryLayoutPolicy::kChannelsLastNdContiguous) {
    outputStrides =
        c10::fmap<ExprHandle>(make_channels_last_strides(outputShape));
  } else {
    // 默认情况下，不执行任何特殊的步长生成
  // 创建空的输出步长向量
  outputStrides = c10::fmap<ExprHandle>(make_contiguous_strides(outputShape));
}

// 创建空的输入参数向量
std::vector<ArgValue> argInputs;

// 根据操作类型填充输入参数向量
if (op == prim::ConstantChunk) {
  auto const& n = v->node();
  argInputs.emplace_back(toArg(inputs[0]));
  argInputs.emplace_back(static_cast<int64_t>(v->offset()));
  argInputs.emplace_back(n->i(attr::dim));
  argInputs.emplace_back(n->i(attr::chunks));
} else if (op == aten::to) {
  argInputs.emplace_back(toArg(inputs[0]));
} else if (op == aten::quantize_per_tensor) {
  argInputs.emplace_back(toArg(inputs[0]));
  // 如果输入为非零维张量，则作为标量参数加入参数向量
  if (!constZeroDimTensorAsScalarArg(inputs[1], argInputs)) {
    argInputs.emplace_back(toArg(inputs[1]));
  }
  if (!constZeroDimTensorAsScalarArg(inputs[2], argInputs)) {
    argInputs.emplace_back(toArg(inputs[2]));
  }
  argInputs.emplace_back(toArg(inputs[3]));
} else if (op == aten::conv2d) {
  // 将所有输入作为参数加入参数向量
  for (auto inp : inputs) {
    argInputs.emplace_back(toArg(inp));
  }
  // 处理可选的偏置项
  if (std::get_if<ArgNone>(&argInputs[2])) {
    // 获取输出类型或者默认为浮点型
    Dtype dtype = outputType ? Dtype(*outputType) : kFloat;
    std::vector<ExprHandle> biasShape;
    biasShape.push_back(outputShape[1]);
    // 创建零填充的偏置张量并加入常量张量列表
    auto bias_tensor = at::zeros({outputShape[1].AsNode<LongImm>()->value()});
    unpacked_constant_tensors_.push_back(bias_tensor);
    // 分配缓冲区以存储偏置张量，并将其加入常量列表
    BufPtr buf = alloc<Buf>(
        "conv2d_bias_opt_" + sanitizeName(v->debugName()),
        ExprHandleVectorToExprVector(biasShape),
        dtype);
    constants_.push_back({buf, bias_tensor.data_ptr()});
    argInputs[2] = BufHandle(buf);
  }
} else {
  // 将所有输入作为参数加入参数向量
  for (auto inp : inputs) {
    argInputs.emplace_back(toArg(inp));
  }
}

// 获取自定义降低函数并调用，返回结果
if (NNCLoweringFunction custom_lowering = getCustomLoweringFor(op)) {
  return custom_lowering(
      argInputs, outputShape, outputStrides, outputType, device_);
}

// 如果节点具有架构，则获取标准降低函数并调用，返回结果
if (v->node()->maybeSchema()) {
  if (NNCLoweringFunction lowering =
          getStandardLoweringFor(c10::toString(v->node()->schema()))) {
    return lowering(
        argInputs, outputShape, outputStrides, outputType, device_);
  }
}

// 抛出异常，指示未处理的节点类型
std::string msg = std::string("Unhandled node kind (in computeValue): ") +
    op.toQualString();
if (v->node()->maybeSchema()) {
  msg += std::string("\nSchema: ") + c10::toString(v->node()->schema());
}
throw malformed_input(msg);
// 结束静态函数 loopBoundsAllEqual 的实现

// 检查是否所有循环的边界相等，如果循环数量小于等于1，则返回 true
static bool loopBoundsAllEqual(const std::vector<ForPtr>& loops) {
  if (loops.size() <= 1) {
    return true;
  }
  // 获取第一个循环的起始和结束表达式
  const auto& start = loops.front()->start();
  const auto& stop = loops.front()->stop();
  // 遍历其余循环，比较它们的起始和结束表达式是否与第一个循环相同
  for (size_t i = 1; i < loops.size(); ++i) {
    const auto& curr_start = loops[i]->start();
    const auto& curr_stop = loops[i]->stop();
    // 如果有不相等的边界表达式，则返回 false
    if (!exprEquals(start, curr_start) || !exprEquals(stop, curr_stop)) {
      return false;
    }
  }
  // 如果所有循环的边界表达式都相等，则返回 true
  return true;
}

// 结束静态函数 fuseAllLoops 的实现

// 递归地融合具有相同边界的所有循环在 `st` 中，如果包含非循环或边界不匹配的层级，则停止融合
// 边界匹配的限制存在是为了避免在不需要的情况下插入循环索引条件，这将显著复杂化向量化过程
static void fuseAllLoops(StmtPtr st) {
  auto block = to<tensorexpr::Block>(st);
  if (block == nullptr) {
    return;
  }

  // 存储所有外层循环的向量的向量
  std::vector<std::vector<ForPtr>> all_outer_loops;
  // 存储外层循环的向量
  std::vector<ForPtr> outer_loops;
  // 遍历块中的每个语句
  for (const auto& stmt : *block) {
    auto loop = to<For>(stmt);
    auto hasReduction = !NodeFinder<ReduceOp>::find(stmt).empty();
    // 如果当前语句不是循环或含有减少操作，则将当前外层循环向量存入 all_outer_loops 并清空 outer_loops
    if (!loop || hasReduction) {
      all_outer_loops.push_back(outer_loops);
      outer_loops.clear();
    } else {
      // 将当前循环存入外层循环向量
      outer_loops.push_back(loop);
    }
  }
  // 将最后剩余的外层循环向量存入 all_outer_loops
  all_outer_loops.push_back(outer_loops);

  // 遍历 all_outer_loops 中的每个外层循环向量
  for (const auto& outer_loops : all_outer_loops) {
    if (outer_loops.empty()) {
      continue;
    }

    // 如果外层循环的边界不相等，则继续下一个外层循环向量
    if (!loopBoundsAllEqual(outer_loops)) {
      continue;
    }

    // 尝试融合外层循环向量中的循环，并将融合后的循环的主体递归传入 fuseAllLoops
    ForPtr fusedLoop;
    if (!LoopNest::fuseLoops(outer_loops, &fusedLoop)) {
      continue;
    }

    fuseAllLoops(fusedLoop->body());
  }
}

// 结束静态函数 tripCount 的实现

// 计算循环的迭代次数（trip count），如果是常数则返回其值，否则返回空 optional
static std::optional<int64_t> tripCount(ForPtr loop) {
  // 简化循环的停止和开始表达式，然后转换为 int64_t 类型
  auto tc = IRSimplifier::simplify(
      cast<int64_t>(ExprHandle(loop->stop()) - ExprHandle(loop->start())));
  // 如果简化后的结果是 LongImm 类型，则返回其值作为迭代次数
  if (auto val = to<LongImm>(tc.node())) {
    return val->value();
  }
  // 否则返回空 optional
  return c10::nullopt;
}

// 结束静态函数 pruneByGrainSize 的实现

// 通过最小粒度大小修剪最内层循环，直到迭代次数满足最小粒度大小为止
static void pruneByGrainSize(std::vector<ForPtr>& loops) {
  constexpr int64_t minGrainSize = 32768;
  int64_t grainSize = 1;
  // 从最内层循环开始向外遍历
  for (int64_t i = loops.size(); i > 0; i--) {
    // 计算当前循环的迭代次数
    auto tc = tripCount(loops[i - 1]);
    if (!tc) {
      break;
    }
    // 将当前循环的迭代次数乘以粒度大小
    grainSize *= *tc;
    // 如果粒度大小小于最小粒度大小，则移除当前循环
    if (grainSize < minGrainSize) {
      loops.pop_back();
    }
  }
}

// 结束静态函数 pruneByThreadCount 的实现

// 保留足够的最外层循环以填充线程数
static void pruneByThreadCount(std::vector<ForPtr>& loops) {
  int64_t trips = 1;
  auto threads = at::get_num_threads();
  auto it = loops.begin();
  // 遍历循环，直到迭代次数满足线程数或循环结束
  for (; it != loops.end(); it++) {
    if (trips >= threads) {
      break;
    }
    // 计算当前循环的迭代次数
    auto tc = tripCount(*it);
    if (!tc) {
      break;
    }
    // 更新总的迭代次数
    trips *= *tc;
  }
  // 移除剩余的循环，使得总的迭代次数不超过线程数
  loops.erase(it, loops.end());
}
// 平铺并行化外层循环，受内层循环最小元素数量和外层循环线程级并行度的限制。
template <typename Bufs>
static void parallelizeOuterLoops(LoopNest& l, Bufs&& bufs) {
  // 遍历每个缓冲区的循环语句
  for (auto const& buf : bufs) {
    auto loops = l.getLoopStmtsFor(buf);
    // 根据最小粒度大小修剪循环
    pruneByGrainSize(loops);
    // 根据线程数量修剪循环
    pruneByThreadCount(loops);

    // 如果没有可并行化的循环，跳过当前缓冲区
    if (loops.size() == 0) {
      continue;
    }
    // 如果循环包含归约操作，跳过当前缓冲区
    auto reductions = NodeFinder<ReduceOp>::find(loops[0]);
    if (reductions.size() > 0) {
      continue;
    }
    // 如果循环存在循环依赖，跳过当前缓冲区
    if (LoopNest::hasLoopCarriedDependence(loops[0])) {
      continue;
    }
    // 尝试平铺外层循环并进行并行化
    ForPtr flattened = nullptr;
    if (loops.size() == 1) {
      flattened = loops[0];
    } else {
      LoopNest::flatten(loops, &flattened);
    }
    // 如果成功平铺并行化，则设置其为并行循环
    if (flattened) {
      flattened->set_parallel();
    }
  }
}

// 在 TensorExprKernel 类中转换循环结构，根据后端类型和语句进行转换
StmtPtr TensorExprKernel::transformLoops(BackendType backendType, StmtPtr st) {
  // 使用给定的语句和输出缓冲区创建循环嵌套对象
  torch::jit::tensorexpr::LoopNest l(st, bufOutputs_);
  // 规范化循环中的名称
  LoopNest::sanitizeNames(l.root_stmt());
  // 调试输出原始语句内容
  GRAPH_DEBUG("Original Stmt:\n", std::to_string(l.root_stmt()), "\n");
  // 生成随机转换请求的种子值
  int64_t random_tr_seed = randomTransformsRequested();
  // 如果有随机转换请求，则根据情况设置种子值
  if (random_tr_seed) {
    if (random_tr_seed == -1)
      random_tr_seed = std::time(nullptr);
    loopnestRandomization(random_tr_seed, l);
    // 调试输出随机转换后的语句内容
    GRAPH_DEBUG(
        "After random transform:\n", std::to_string(l.root_stmt()), "\n");
  }

  // 检查循环中是否包含归约操作
  bool hasReduction = !NodeFinder<ReduceOp>::find(l.root_stmt()).empty();

  // 对于块代码生成，创建多维缓冲区信息映射
  auto block_analysis = std::make_unique<CreateBufferMap>();
  if (backendType == kBlockCodeGen) {
    // 运行块分析以获取多维缓冲区信息
    auto root_stmt = l.root_stmt();
    // 接受块分析器分析的根语句
    root_stmt->accept(block_analysis.get());
  }
  // 简化语句列表
  l.simplify();
  // 输出调试信息，显示简化后的语句列表根语句
  GRAPH_DEBUG("after simplify", *l.root_stmt());

  // 内联输出和中间缓冲可能会导致计算重复
  // 如果未通过某种方式改善，重复计算会降低程序性能
  // 在实践中发现：
  // - 在 CPU 上，LLVM 的公共子表达式消除（CSE）在水平融合输出循环时效果良好
  // - 在 GPU 上，有足够的计算资源隐藏额外的工作量，并且内联避免了内核之间的同步
  l.inlineIntermediateBufs(/*allow_duplicated_work=*/true);
  // 输出调试信息，显示内联中间缓冲后的语句列表根语句
  GRAPH_DEBUG("after inline", *l.root_stmt());

  // 在内联之后执行条件优化是必要的，因为一旦循环被分割，内联就无法正常工作
  // 同时，在循环融合之前执行条件优化是必要的，因为循环融合会引入多个条件在同一循环中的情况
  // 这种优化尚不能处理这些情况
  if (getOptConditionals()) {
    // 优化条件语句
    l.optimizeConditionals();
    // 输出调试信息，显示优化条件语句后的语句列表根语句
    GRAPH_DEBUG("after optimizing conditionals: ", *l.root_stmt());
  }

  // 水平融合循环，允许将写入不同输出缓冲区的循环结合起来，只要它们具有相同的边界
  if (backendType == kLLVMCodeGen) {
    // 对语句列表根语句执行循环融合
    fuseAllLoops(l.root_stmt());
    // 输出调试信息，显示循环融合后的语句列表根语句
    GRAPH_DEBUG("after fuse", *l.root_stmt());
    // 并行化外层循环
    parallelizeOuterLoops(l, bufsToBeParallelized_);
    // 输出调试信息，显示并行化外层循环后的语句列表根语句
    GRAPH_DEBUG("after parallelize", *l.root_stmt());
  }

  // 如果后端类型为 CUDA 代码生成
  if (backendType == kCudaCodeGen) {
    for (const auto& buf : bufOutputs_) {
      // 获取给定缓冲区的循环语句列表
      std::vector<ForPtr> loops = l.getLoopStmtsFor(buf);
      // 如果循环语句列表为空，说明缓冲区是零维的情况，跳过处理
      if (loops.empty()) {
        continue;
      }
      // 初始化用于扁平化后的循环语句指针
      ForPtr flattened = nullptr;
      // 将循环语句列表扁平化
      LoopNest::flatten(loops, &flattened);
      assert(flattened);

      // 获取 CUDA 点对点操作的循环级别
      int loopLevels = getTECudaPointwiseLoopLevels();
      const int kDefaultLoopLevels = 2;
      // 如果未指定循环级别，使用默认的循环级别
      loopLevels = (loopLevels > 0) ? loopLevels : kDefaultLoopLevels;
      // 获取 CUDA 点对点操作的块数和块大小
      int blockCount = getTECudaPointwiseBlockCount();
      int blockSize = getTECudaPointwiseBlockSize();

      // 如果循环级别为 2
      if (loopLevels == 2) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ForPtr inner;
        const int kDefaultBlockSize = 512;
        // 如果未指定块大小，使用默认的块大小
        if (blockSize < 0) {
          blockSize = kDefaultBlockSize;
        }
        // 将扁平化后的循环语句分割成指定大小的块
        LoopNest::splitWithMask(flattened, blockSize, &inner);
        // 设置 GPU 块索引为 0
        flattened->set_gpu_block_index(0);
        // 设置 GPU 线程索引为 0
        inner->set_gpu_thread_index(0);
      } else if (loopLevels == 3) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ForPtr inner;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ForPtr inner1;
        // TODO: 更改微处理器的数量
        const int kDefaultBlockCount = 1280;
        const int kDefaultBlockSize = 256;
        // 如果未指定块数和块大小，使用默认的块数和块大小
        blockCount = (blockCount > 0) ? blockCount : kDefaultBlockCount;
        blockSize = (blockSize > 0) ? blockSize : kDefaultBlockSize;
        // 将扁平化后的循环语句分割成指定大小的块
        LoopNest::splitWithMask(flattened, blockCount * blockSize, &inner);
        LoopNest::splitWithMask(inner, blockSize, &inner1);
        // 设置 GPU 块索引为 0
        inner->set_gpu_block_index(0);
        // 设置 GPU 线程索引为 0
        inner1->set_gpu_thread_index(0);
      } else {
        // 抛出异常，因为循环级别无效
        throw std::runtime_error(
            "Invalid loop-level: " + std::to_string(loopLevels));
      }
    }
  }

  // 如果后端类型为块代码生成
  if (backendType == kBlockCodeGen) {
    // 遍历输出缓冲区列表
    for (const auto& buf : bufOutputs_) {
      // 默认的块大小常量
      const int default_fp16_blocksize = 16;
      const int default_uint8_blocksize = 32;
      int blockSize = default_fp16_blocksize;
      // 只处理循环级别为 2 的情况
      if (buf->dtype().scalar_type() == ScalarType::Byte) {
        // 如果缓冲区数据类型是字节，则使用默认的 uint8 块大小
        blockSize = default_uint8_blocksize;
      }
      // 获取给定缓冲区的循环语句列表
      std::vector<ForPtr> loops = l.getLoopStmtsFor(buf);
      // 断言循环语句列表不为空
      TORCH_INTERNAL_ASSERT(
          !loops.empty(),
          buildErrorMessage(
              "No loops found for the buffer " + buf->name_hint() +
              " in the fuser."));
      // 初始化用于扁平化后的循环语句指针
      ForPtr flattened = nullptr;
      // 将循环语句列表扁平化
      LoopNest::flatten(loops, &flattened);
      assert(flattened);

      ForPtr inner = nullptr;
      // 将扁平化后的循环语句分割成指定大小的块
      LoopNest::splitWithMask(flattened, blockSize, &inner);
      // 设置 GPU 块索引为 0
      flattened->set_gpu_block_index(0);
      // 设置 GPU 线程索引为 0
      inner->set_gpu_thread_index(0);
      // 设置扁平化后的循环语句的缓冲区映射
      flattened->set_buffer_map(block_analysis->getBufferMap());
    }
  }

  // 如果预分配标志为真
  if (pre_alloc_) {
    // 获取中间缓冲区列表
    auto interm_bufs = l.getIntermediateBufs();
  // 调用函数 preAllocIntermediateBufs，传入中间缓冲区 interm_bufs
  preAllocIntermediateBufs(interm_bufs);
}

// 调用 LoopNest 的 prepareForCodegen 方法，为代码生成做准备
l.prepareForCodegen();

// 输出调试信息，显示经过 prepareForCodegen 后的语句根节点
GRAPH_DEBUG("after prepareForCodegen", *l.root_stmt());

// 对语句进行简化处理
l.simplify();

// 输出调试信息，显示经过简化后的语句根节点
GRAPH_DEBUG("after simplification", *l.root_stmt());

// 如果后端类型是 kLLVMCodeGen 并且没有进行缩减操作，则进行内部循环向量化
if (backendType == kLLVMCodeGen && !hasReduction) {
  l.vectorizeInnerLoops();
  // 输出调试信息，显示经过向量化后的语句根节点
  GRAPH_DEBUG("after vectorization", *l.root_stmt());
}

// 获取最终的语句根节点
StmtPtr stmt = l.root_stmt();

// 对语句进行算术简化
stmt = IRSimplifier::simplify(stmt);

// 输出调试信息，显示最终简化后的语句内容
GRAPH_DEBUG("Final Stmt:\n", std::to_string(stmt), "\n");

// 返回最终简化后的语句根节点
return stmt;
}

// 获取特定后端类型的代码生成名称
std::string TensorExprKernel::getCodeGenName(BackendType backendType) {
  switch (backendType) {
    case kCudaCodeGen:
      return "cuda_codegen";
    case kLLVMCodeGen:
      return "llvm_codegen";
    case kSimpleIREval:
      return "simple_ir_eval";
    case kBlockCodeGen:
      return "block_codegen";
    default:
      // 抛出异常，显示无效的后端类型及其整数表示
      throw std::runtime_error(
          "invalid backend type: " +
          std::to_string(static_cast<int>(backendType)));
  }
}

// 检查可选参数与给定值的有效性
template <typename T>
static bool isValidPrimProperty(const std::optional<T>& a, T b) {
  return !a.has_value() || *a == b;
}

// 推断给定设备对应的后端类型
TensorExprKernel::BackendType TensorExprKernel::inferBackendTypeFromDevice(
    at::Device device) {
  BackendType backendType = BackendType::kUninitialized;
  if (device.type() == at::kCUDA) {
    backendType = kCudaCodeGen;
  } else if (device.type() == at::kCPU && getTEGenerateBlockCode()) {
    backendType = kBlockCodeGen;
  } else if (device.type() == at::kCPU) {
#ifdef TORCH_ENABLE_LLVM
    // 根据 LLVM 的可用性选择后端类型
    backendType = dontUseLLVMFlag() ? kSimpleIREval : kLLVMCodeGen;
#else
    backendType = kSimpleIREval;
#endif
    // 如果强制要求在 CPU 上使用 LLVM 但当前选择了 simple_ir_eval，则抛出异常
    if (getTEMustUseLLVMOnCPU() && backendType == kSimpleIREval) {
      throw std::runtime_error("LLVM Backend not found");
    }
  } else {
    // 无效的设备类型，抛出异常
    throw std::runtime_error("Invalid device type");
  }
  return backendType;
}

// 生成输入的调试名称，确保不含无效字符
void TensorExprKernel::genInputDebugNames() {
  std::unordered_map<std::string, const torch::jit::Value*> name_to_value;
  std::unordered_set<std::string> name_set;
  std::unordered_map<const torch::jit::Value*, std::string> value_to_name;
  for (const torch::jit::Value* input : graph_->inputs()) {
    std::string sanitized_name = sanitizeName(input->debugName());
    // 处理潜在的名称冲突
    while (name_set.count(sanitized_name)) {
      sanitized_name.append("_");
    }
    value_to_name[input] = sanitized_name;
    name_set.insert(sanitized_name);
  }
  input_name_map_ = std::move(value_to_name);
}

// 将大小向量转换为表达式处理器向量
template <typename T>
static std::vector<ExprHandle> toExprHandles(const std::vector<T>& sizes) {
  std::vector<ExprHandle> dims;
  dims.reserve(sizes.size());
  for (auto const& size : sizes) {
    dims.emplace_back(size);
  }
  return dims;
}

// 获取张量输入的特定步长参数
ExprHandle TensorExprKernel::getStrideArg(
    size_t tensor_input_index,
    size_t stride_index) {
  auto it = strideArgToVar_.find(
      std::pair<size_t, size_t>(tensor_input_index, stride_index));
  if (it == strideArgToVar_.end()) {
    VarHandle var(
        // 构造步长参数的变量名
        "stride_arg" + std::to_string(tensor_input_index) + "_" +
            std::to_string(stride_index),
        kLong);
    strideArgToVar_[std::pair<size_t, size_t>(
        tensor_input_index, stride_index)] = var;
    return std::move(var);
  }
  return it->second;
}

// 获取符号步长描述的引用向量
std::vector<torch::jit::StrideInput>& TensorExprKernel::getSymbolicStrideDesc(
    // 使用 TORCH_INTERNAL_ASSERT 宏断言 symbolic_strides_ 中存在 value 对应的键
    TORCH_INTERNAL_ASSERT(symbolic_strides_.count(value));
    // 返回 symbolic_strides_ 中 value 对应的值
    return symbolic_strides_[value];
// 返回输入张量的步长向量，根据输入张量的维度信息和步长描述来计算
std::vector<ExprHandle> TensorExprKernel::getInputStrides(
    const torch::jit::Value* input,
    const std::vector<ExprHandle>& inputTensorDims) {
  // 初始化存储输入张量步长的向量
  std::vector<ExprHandle> inputTensorStrides;

  // 检查输入是否为完整张量
  if (input->isCompleteTensor()) {
    // 如果是完整张量，获取其步长信息并转换为具体大小的向量
    auto const strides =
        input->type()->expect<TensorType>()->strides().concrete_sizes();
    std::vector<ExprHandle> inputTensorStrides;
    // 遍历步长大小，将其转换为 LongImm 类型的表达式并存储到 inputTensorStrides 中
    for (size_t stride : *strides) {
      inputTensorStrides.push_back(LongImm::make(stride));
    }
    return inputTensorStrides;
  }

  // 获取输入张量的维度数量
  size_t rank = inputTensorDims.size();

  // 获取符号步长描述信息
  std::vector<StrideInput>& stride_input = getSymbolicStrideDesc(input);

  // 处理仅有一个符号步长且为连续布局的情况
  if (stride_input.size() == 1 &&
      (stride_input[0] == StrideInput::TENSOR_CONT_CHANNELS_LAST ||
       stride_input[0] == StrideInput::TENSOR_CONT)) {
    // 根据布局类型生成连续步长
    auto strides = stride_input[0] == StrideInput::TENSOR_CONT
        ? make_contiguous_strides(inputTensorDims)
        : make_channels_last_strides(inputTensorDims);
    // 使用 fmap 函数将步长转换为 ExprHandle 类型并返回
    return fmap(strides, [&](ExprPtr stride) { return ExprHandle(stride); });
  }

  // 初始化 inputTensorStrides 为给定维度数
  inputTensorStrides.resize(rank);
  
  // 初始化步长设置向量，标记每个维度的步长是否已生成
  std::vector<bool> stride_set;
  for (size_t i = 0; i < rank; ++i) {
    stride_set.push_back(false);
  }

  // 生成非依赖值的步长
  size_t generated_strides = 0;
  for (const auto i : c10::irange(rank)) {
    if (stride_input[i] == torch::jit::StrideInput::S_ONE) {
      // 对于步长为 1 的情况，使用 LongImm 类型的 1 存储步长
      inputTensorStrides[i] = LongImm::make(1);
      stride_set[i] = true;
      generated_strides++;
    } else if (stride_input[i] == torch::jit::StrideInput::S_AS_ARG) {
      // 对于作为参数的步长，调用 getStrideArg 函数获取步长值
      size_t input_index = input->offset();
      inputTensorStrides[i] = getStrideArg(input_index, i);
      stride_set[i] = true;
      generated_strides++;
    }
  }

  // 处理连续布局和转置连续布局依赖于相邻值的情况
  while (generated_strides != rank) {
    // 从最后一个维度向前遍历处理连续布局
    for (int i = static_cast<int>(rank) - 1; i >= 0; i--) {
      if (stride_input[i] == torch::jit::StrideInput::S_CONT &&
          stride_set[i + 1]) {
        // 如果当前维度步长依赖于下一个维度，则计算步长并存储
        inputTensorStrides[i] =
            inputTensorStrides[i + 1] * inputTensorDims[i + 1];
        stride_set[i] = true;
        generated_strides++;
      }
    }
    // 从第一个维度向后遍历处理转置连续布局
    for (int i = 0; i < static_cast<int>(rank); i++) {
      if (stride_input[i] == torch::jit::StrideInput::S_TRAN_CONT &&
          stride_set[i - 1]) {
        // 如果当前维度步长依赖于前一个维度，则计算步长并存储
        inputTensorStrides[i] =
            inputTensorStrides[i - 1] * inputTensorDims[i - 1];
        stride_set[i] = true;
        generated_strides++;
      }
    }
  }
  
  // 返回计算得到的输入张量步长向量
  return inputTensorStrides;
}

// 将输入绑定到张量表示
Tensor TensorExprKernel::bindInput(const torch::jit::Value* input) {
  auto const& t = input->type();
  auto const& outputs = input->owningGraph()->outputs();
  // 创建输出值的无序集合
  std::unordered_set<const Value*> outputs_set(outputs.begin(), outputs.end());

  // 定义用于检查是否具有具体连续布局的函数
  auto is_concrete_cont = [](const torch::jit::Value* input,
                             const MemoryLayoutPolicy& mem_layout_policy) {
  // 检查输入是否为完整的张量
  if (input->isCompleteTensor()) {
    // 根据内存布局策略选择内存格式，如果是连续的则选择 Contiguous，否则选择 ChannelsLast
    auto mem_layout = (mem_layout_policy == MemoryLayoutPolicy::kContiguous)
        ? at::MemoryFormat::Contiguous
        : at::MemoryFormat::ChannelsLast;
    // 检查输入张量是否符合指定的内存布局
    return isContiguous(input, mem_layout);
  } else {
    // 如果输入张量不完整，则返回 false
    return false;
  }
};

// 检查符号描述的张量是否符合符号连续性要求
auto is_symbolic_cont = [](std::vector<torch::jit::StrideInput> desc,
                           const MemoryLayoutPolicy& mem_layout_policy) {
  // 如果描述只有一个元素
  if (desc.size() == 1) {
    // 根据内存布局策略选择符号描述的内存布局
    auto mem_layout = (mem_layout_policy == MemoryLayoutPolicy::kContiguous)
        ? torch::jit::StrideInput::TENSOR_CONT
        : torch::jit::StrideInput::TENSOR_CONT_CHANNELS_LAST;
    // 检查描述是否等于选择的内存布局
    return desc[0] == mem_layout;
  } else {
    // 如果描述元素个数不为1，则返回 false
    return false;
  }
};

// 初始化一个空的张量 result
Tensor result(nullptr, nullptr);

// 根据张量的类型进行 switch 分支
switch (t->kind()) {
    case TypeKind::TensorType: {
      // 如果输入类型是张量类型
      auto tt = input->type()->cast<TensorType>();
      // 检查是否为具体的连续张量
      bool contiguous_concrete_tensor =
          is_concrete_cont(input, memory_layout_policy_);
      // 初始化符号形状连续性为假
      bool contiguous_symbolic_tensor = false;
      // 如果存在符号形状
      if (has_symbolic_shapes_) {
        // 获取符号步幅描述
        auto desc = getSymbolicStrideDesc(input);
        // 检查是否为符号形状连续
        contiguous_symbolic_tensor =
            is_symbolic_cont(desc, memory_layout_policy_);
      }

      // 获取输入的尺寸和步幅
      auto size_handles = sizesFromSymbolicShape(tt->symbolic_sizes());
      auto inputTensorStrides = getInputStrides(input, size_handles);

      // 如果不需要复制输入的条件：
      //  1) 它不是输出，并且
      //  2) 它是连续的
      bool contiguous =
          contiguous_concrete_tensor || contiguous_symbolic_tensor;
      if (!outputs_set.count(input) && contiguous) {
        // 创建输入缓冲区
        BufHandle inBuffer(
            "t" + input_name_map_[input],
            sizesFromSymbolicShape(tt->symbolic_sizes()),
            inputTensorStrides,
            ToDtype(static_cast<ScalarType>(*tt->scalarType())));
        // 断言输入缓冲区是连续的或符合特定的内存格式
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            inBuffer.node()->is_contiguous() ||
            inBuffer.node()->is_channels_last_1d_contiguous() ||
            inBuffer.node()->is_contiguous(at::MemoryFormat::ChannelsLast) ||
            inBuffer.node()->is_contiguous(at::MemoryFormat::ChannelsLast3d));
        // 将输入缓冲区加入 bufs_ 中
        bufs_.emplace(input, inBuffer.node());
        // 将输入缓冲区添加到 bufferArgs_ 中
        bufferArgs_.emplace_back(inBuffer);
        break;
      }

      // 如果输入不是连续的或者是输出，
      // 将带步幅的输入写入连续的缓冲区，
      // 然后在所有进一步的计算中使用
      ExprHandle flat_size = 1;
      for (size_t i = 0; i < size_handles.size(); ++i) {
        auto size = size_handles[i];
        if (size.AsNode<LongImm>() && immediateAs<int64_t>(size.node()) == 0) {
          flat_size = 0;
          break;
        }
        flat_size = flat_size + (size - 1) * inputTensorStrides[i];
      }
      flat_size = IRSimplifier::simplify(flat_size);
      // 创建输入缓冲区
      BufHandle inBuffer(
          "t" + input_name_map_[input],
          {flat_size},
          ToDtype(static_cast<ScalarType>(*tt->scalarType())));

      // 创建计算表达式
      result = Compute(
          "input" + std::to_string(bufs_.size() + 1),
          size_handles,
          [&](const std::vector<VarHandle>& axes) {
            ExprHandle idx = 0;
            for (size_t i = 0; i < axes.size(); i++) {
              idx = idx + axes[i] * inputTensorStrides[i];
            }
            return inBuffer.load(idx);
          });
      // 将输入缓冲区加入 bufs_ 中
      bufs_.emplace(input, result.buf());
      // 将输入缓冲区添加到 bufferArgs_ 中
      bufferArgs_.emplace_back(inBuffer);
      break;
    }
    case TypeKind::FloatType: {
      // 如果输入类型是浮点数类型
      VarHandle v("v" + input_name_map_[input], kDouble);
      // 将变量 v 添加到 bufferArgs_ 中
      bufferArgs_.emplace_back(v);
      // 将变量 v 添加到 scalars_ 中
      scalars_.emplace(input, v);
      break;
    }
    case TypeKind::BoolType: {
      // 对于布尔类型，创建一个名为 v{input_name} 的变量句柄，类型为 kBool
      VarHandle v("v" + input_name_map_[input], kBool);
      // 将该变量句柄添加到 bufferArgs_ 向量中
      bufferArgs_.emplace_back(v);
      // 将该变量句柄添加到 scalars_ 映射中，键为 input
      scalars_.emplace(input, v);
      // 跳出 switch 语句块
      break;
    }
    case TypeKind::IntType: {
      // 对于整数类型，创建一个名为 v{input_name} 的变量句柄，类型为 kLong
      VarHandle v("v" + input_name_map_[input], kLong);
      // 将该变量句柄添加到 bufferArgs_ 向量中
      bufferArgs_.emplace_back(v);
      // 将该变量句柄添加到 scalars_ 映射中，键为 input
      scalars_.emplace(input, v);
      // 跳出 switch 语句块
      break;
    }
    default: {
      // 如果出现不支持的数据类型，抛出异常，异常信息为类型 t 的字符串表示
      throw unsupported_dtype(t->repr_str());
      // 跳出 switch 语句块
      break;
    }
  }
  // 返回 result 变量
  return result;
}

NNCLoweringFunction TensorExprKernel::getCustomLoweringFor(
    c10::Symbol op) const {
  // 检查是否存在自定义降级操作，如果存在则返回其降级函数，否则返回空指针
  if (custom_lowerings_.count(op))
    return custom_lowerings_.at(op);
  // 如果没有找到对应操作的自定义降级函数，返回空指针
  return nullptr;
}

template <typename T>
std::vector<size_t> reverse_sort_indices(const std::vector<T>& v) {
  // 初始化索引数组，存储原始的索引位置
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // 根据向量 v 中的值对索引数组 idx 进行降序排序
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
    return v[i1] > v[i2];
  });
  // 返回降序排序后的索引数组
  return idx;
}

static bool denseAndNonOverlapping(
    at::ArrayRef<int64_t> sizes,
    at::ArrayRef<int64_t> strides) {
  // 检查给定的步长数组 strides 是否与给定的大小数组 sizes 推断的密集步长相匹配
  return (strides == at::infer_dense_strides(sizes, strides));
}

Tensor TensorExprKernel::convertSymbolicOutputToCorrectStrides(
    const std::vector<ExprHandle>& sizes,
    const std::vector<size_t>& sorted_stride_indices_descending,
    const std::vector<ExprPtr>& strides,
    // 将输出张量重新排列，使得其数值按照输出步长正确排列。
    // 对于大小为(2, 3)的连续张量，数值为0-5，布局如下：
    // [0] [1] [2] [3] [4] [5]
    // 步长为(1, 2)的相同数值张量的布局如下：
    // [0] [3] [1] [4] [2] [5]
    // 在我们重新排序数值到输出张量时，我们按照输入的每个元素进行迭代，
    // 并且我们固定在输出张量的[i, j]位置索引为val。
    // 这里的`val`等于输出张量中与输出位置相同的索引。
    // 位置等于 stride[i] * index[i] 的总和，我们可以通过迭代计算最大步长的索引，
    // 来计算等效于输出张量步长的索引。
    std::vector<ExprPtr> default_strides = make_contiguous_strides(sizes);
    // 创建一个值为0的长整数常量
    auto zero = LongImm::make(0);
    // 定义一个名为"output_1"的计算操作，使用给定的大小和轴
    return Compute(
        "output_1", sizes, [&](const std::vector<VarHandle>& axes_input) {
          // 将输入轴转换为表达式处理器
          std::vector<ExprHandle> axes(axes_input.begin(), axes_input.end());
          // 初始化绝对位置为0
          auto absolute_position = ExprHandle(immLike(axes[0], 0));
          // 对于每个轴，计算绝对位置
          for (size_t i = 0; i < axes.size(); ++i) {
            ExprHandle stride(default_strides[i]);
            ExprHandle axis = axes[i];
            absolute_position = absolute_position + (stride * axis);
          }
          // 创建一个新的轴向量，大小为按降序排列的步长索引
          std::vector<ExprHandle> new_axes(
              sorted_stride_indices_descending.size());
          // 对于按步长索引降序的每个步长索引
          for (size_t stride_index : sorted_stride_indices_descending) {
            // 获取对应的步长
            const auto& stride = strides[stride_index];
            // 计算索引
            auto index = absolute_position / ExprHandle(stride);
            // 在符号输出排序中，我们不需要常规输出排序中的任意步长排序，只需最后的通道，
            // 因此，即使在 size == 1 的情况下，我们也可以正确生成输出。
            absolute_position = absolute_position % ExprHandle(stride);
            new_axes[stride_index] = index;
          }
          // 返回从缓冲区加载的新轴向量
          return BufHandle(buf).load(new_axes);
        });
// 结束函数 convertSymbolicOutputToCorrectStrides 的定义
Tensor TensorExprKernel::convertSymbolicOutputToCorrectStrides(
    torch::jit::Value* v) {
  // 获取值 v 的张量类型指针
  const TensorTypePtr& tt = v->type()->expect<TensorType>();
  // 断言确保 bufs_ 中存在值 v 对应的缓冲区
  TORCH_INTERNAL_ASSERT(
      bufs_.count(v),
      buildErrorMessage(
          "Output tensor has no corresponding bufs in the fuser."));
  // 获取值 v 对应的缓冲区指针
  BufPtr buf = bufs_.at(v);
  // 断言确保 buf 不为空指针
  TORCH_INTERNAL_ASSERT(buf != nullptr);
  // 断言确保 tt 不为空指针
  TORCH_INTERNAL_ASSERT(tt != nullptr);
  // 断言确保张量类型的符号尺寸不为 nullptr
  TORCH_INTERNAL_ASSERT(tt->symbolic_sizes().rank() != c10::nullopt);

  // 获取符号步长描述
  auto stride_desc = getSymbolicStrideDesc(v);
  // 断言确保步长描述长度为1
  TORCH_INTERNAL_ASSERT(stride_desc.size() == 1);
  // 根据步长描述确定内存格式，如果是 TENSOR_CONT，则为连续内存格式；否则为 ChannelsLast
  auto memory_format = (stride_desc[0] == torch::jit::StrideInput::TENSOR_CONT)
      ? at::MemoryFormat::Contiguous
      : at::MemoryFormat::ChannelsLast;
  // 如果缓冲区已经是指定的内存格式的连续的，则直接返回对应的 Tensor 对象
  if (buf->is_contiguous(memory_format)) {
    return Tensor(buf, nullptr);
  }

  // 断言确保步长描述为 TENSOR_CONT_CHANNELS_LAST
  TORCH_INTERNAL_ASSERT(
      stride_desc[0] == torch::jit::StrideInput::TENSOR_CONT_CHANNELS_LAST);
  // 根据符号尺寸获取大小
  auto sizes = sizesFromSymbolicShape(tt->symbolic_sizes());
  // 根据大小生成 ChannelsLast 内存格式的步长
  auto strides = make_channels_last_strides(sizes);
  // 对于维度为 N C H W 的张量，ChannelsLast 格式为 N H W C，按最大到最小的顺序排列为 N, H, W, C
  std::vector<size_t> sorted_stride_indices = {0, 2, 3, 1};
  // 创建默认的连续步长
  auto zero = LongImm::make(0);
  std::vector<ExprPtr> default_strides = make_contiguous_strides(sizes);
  // 调用另一个函数 convertSymbolicOutputToCorrectStrides 进行处理
  // 参见 convertOutputToCorrectStrides 的解释
  return convertSymbolicOutputToCorrectStrides(
      sizes, sorted_stride_indices, strides, buf);
}

// 开始函数 convertStaticShapeOutputToCorrectStrides 的定义
Tensor TensorExprKernel::convertStaticShapeOutputToCorrectStrides(
    torch::jit::Value* v) {
  // 获取值 v 的张量类型指针
  const TensorTypePtr& tt = v->type()->expect<TensorType>();
  // 断言确保 bufs_ 中存在值 v 对应的缓冲区
  TORCH_INTERNAL_ASSERT(
      bufs_.count(v),
      buildErrorMessage(
          "Output tensor has no corresponding bufs in the fuser."));
  // 获取值 v 对应的缓冲区指针
  BufPtr buf = bufs_.at(v);

  // 如果图中不存在形状信息
  if (!tt->sizes().concrete_sizes()) {
    // 抛出异常，指示输出 '%v 的形状未知
    std::string msg =
        std::string("Shapes for output '%") + v->debugName() + "' are unknown";
    throw malformed_input(msg);
  }

  // 断言确保张量类型的具体尺寸不为空
  TORCH_INTERNAL_ASSERT(
      tt->sizes().concrete_sizes(),
      buildErrorMessage("Output shapes are unknown."));
  // 获取具体尺寸
  auto sizes = *tt->sizes().concrete_sizes();
  // 根据内存布局策略确定内存格式，默认为 Contiguous 或 ChannelsLast
  at::MemoryFormat memory_format =
      (memory_layout_policy_ == MemoryLayoutPolicy::kContiguous)
      ? c10::MemoryFormat::Contiguous
      : c10::MemoryFormat::ChannelsLast;
  // 获取默认的连续步长
  std::vector<int64_t> default_strides =
      TensorType::contiguousStridesOf(sizes, memory_format);
  // 如果具体步长为空
  if (!tt->strides().concrete_sizes()) {
    // 抛出异常，指示输出形状未知
    throw malformed_input("Output shapes are unknown.");
  }
  return Tensor(buf, nullptr);


// 返回一个 Tensor 对象，使用给定的 buf 和空指针作为构造参数
return Tensor(buf, nullptr);



TORCH_INTERNAL_ASSERT(
    tt->strides().concrete_sizes(),
    buildErrorMessage("Output strides are unknown."));


// 使用内部断言确保 tt 的 strides() 方法返回具体的大小，否则抛出错误消息
TORCH_INTERNAL_ASSERT(
    tt->strides().concrete_sizes(),
    buildErrorMessage("Output strides are unknown."));



const std::vector<int64_t> strides = *tt->strides().concrete_sizes();


// 获取 tt 的具体步长，并将其存储在 strides 中
const std::vector<int64_t> strides = *tt->strides().concrete_sizes();



// All Tensors in NNC are layed out in default, contiguous layout.
// If the output is also default contiguous we don't need to do anything
if (strides == default_strides) {
  return Tensor(buf, nullptr);
}


// 检查是否输出 Tensor 采用默认的连续布局，若是，则直接返回 Tensor 对象
if (strides == default_strides) {
  return Tensor(buf, nullptr);
}



// If the tensor is not dense or overlaps, we have
// no way of matching the profiled striding
if (!denseAndNonOverlapping(sizes, strides)) {
  return Tensor(buf, nullptr);
}


// 如果 Tensor 不是密集的或者存在重叠，无法匹配已配置的步长，直接返回 Tensor 对象
if (!denseAndNonOverlapping(sizes, strides)) {
  return Tensor(buf, nullptr);
}



auto dims = sizesForValue(v);
auto zero = LongImm::make(0);
std::vector<size_t> sorted_stride_indices = reverse_sort_indices(strides);


// 获取值 v 对应的大小，以及创建一个零的长整型常量
auto dims = sizesForValue(v);
auto zero = LongImm::make(0);
// 根据步长 strides 对应的排序索引，进行逆序排列
std::vector<size_t> sorted_stride_indices = reverse_sort_indices(strides);



// TODO: call into `convertOutputToCorrectStrides`. Currently this causes a
// bug in IRSimplifier to occur. See explanation in
// `convertOutputToCorrectStrides`
return Compute(
    "output_1", dims, [&](const std::vector<VarHandle>& axes_input) {
      std::vector<ExprHandle> axes(axes_input.begin(), axes_input.end());
      auto absolute_position = ExprHandle(immLike(axes[0], 0));
      for (size_t i = 0; i < axes.size(); ++i) {
        absolute_position = absolute_position +
            (ExprHandle(immLike(axes[i], default_strides[i])) * axes[i]);
      }

      std::vector<ExprHandle> new_axes(sorted_stride_indices.size());
      for (size_t stride_index : sorted_stride_indices) {
        auto size = sizes[stride_index];
        auto index = zero;
        if (size != 1) {
          auto stride = strides[stride_index];
          index = absolute_position /
              ExprHandle(immLike(absolute_position, stride));
          absolute_position = absolute_position %
              ExprHandle(immLike(absolute_position, stride));
        }
        new_axes[stride_index] = index;
      }
      return BufHandle(buf).load(new_axes);
    });


// 调用 Compute 函数生成一个新的计算表达式
// 此处计算表达式命名为 "output_1"，传入维度 dims，并定义 lambda 函数处理 axes_input
return Compute(
    "output_1", dims, [&](const std::vector<VarHandle>& axes_input) {
      // 将 axes_input 转换为 ExprHandle 类型的 axes
      std::vector<ExprHandle> axes(axes_input.begin(), axes_input.end());
      // 初始化 absolute_position 为 axes[0] 的常量表达式
      auto absolute_position = ExprHandle(immLike(axes[0], 0));
      // 遍历所有 axes，计算绝对位置 absolute_position
      for (size_t i = 0; i < axes.size(); ++i) {
        absolute_position = absolute_position +
            (ExprHandle(immLike(axes[i], default_strides[i])) * axes[i]);
      }

      // 创建一个新的轴向量 new_axes，根据步长排序索引 sorted_stride_indices 的大小
      std::vector<ExprHandle> new_axes(sorted_stride_indices.size());
      // 根据步长排序索引计算新轴的值
      for (size_t stride_index : sorted_stride_indices) {
        auto size = sizes[stride_index];
        auto index = zero;
        if (size != 1) {
          auto stride = strides[stride_index];
          index = absolute_position /
              ExprHandle(immLike(absolute_position, stride));
          absolute_position = absolute_position %
              ExprHandle(immLike(absolute_position, stride));
        }
        new_axes[stride_index] = index;
      }
      // 返回从 buf 加载新轴的 BufHandle 对象
      return BufHandle(buf).load(new_axes);
    });
}

void TensorExprKernel::bindConstant(const torch::jit::Value* v) {
  auto val = toIValue(v).value();  // 获取值 v 的 IValue
  if (torch::isCustomClass(val)) {  // 检查值是否为自定义类
    auto name_hint = "const_" + sanitizeName(v->debugName());  // 创建常量名称提示
    auto dtype = Dtype(ScalarType::Float);  // 设置数据类型为 Float
    std::vector<ExprPtr> dims;  // 创建维度向量
    BufPtr buf = alloc<Buf>(name_hint, dims, dtype);  // 分配一个缓冲区
    auto dataPtr = val.toObjectRef().getSlot(0).toCapsule().get();  // 获取对象的数据指针
    // NOLINTNEXTLINE
    constants_.push_back({buf, dataPtr, const_cast<Node*>(v->node())});  // 将常量信息加入列表
    bufs_[v] = buf;  // 将缓冲区与值关联
    return;  // 返回
  }
  if (!v->type()->cast<TensorType>()) {
    // 只有张量常量需要绑定，标量常量将在 TE IR 中转换为立即数
    return;  // 返回
  }
  auto const_tensor = toIValue(v)->toTensor();  // 获取张量常量
  auto scalar_type = c10::typeMetaToScalarType(const_tensor.options().dtype());  // 转换标量类型
  auto sizes = const_tensor.sizes();  // 获取张量大小
  std::vector<ExprHandle> te_sizes;  // 创建表达式大小向量
  te_sizes.reserve(sizes.size());
  for (auto s : sizes) {
    te_sizes.emplace_back(s);  // 将大小添加到表达式大小向量中
  }
  BufPtr buf = alloc<Buf>(
      "const_" + sanitizeName(v->debugName()),  // 创建常量缓冲区名称
      ExprHandleVectorToExprVector(te_sizes),  // 转换表达式大小向量
      ToDtype(scalar_type));  // 转换数据类型

  if (!const_tensor.is_contiguous()) {
    const_tensor = const_tensor.clone().contiguous();  // 克隆并确保张量是连续的
    unpacked_constant_tensors_.push_back(const_tensor);  // 将非连续的张量添加到列表中
  }

  constants_.push_back({buf, const_tensor.data_ptr()});  // 将常量信息加入列表
  bufs_[v] = buf;  // 将缓冲区与值关联
}

std::vector<BufPtr> TensorExprKernel::preAllocIntermediateBufs(
    const std::vector<BufPtr>& interm_bufs) {
  std::vector<BufPtr> remaining_interm_bufs;  // 创建剩余中间缓冲区向量
  for (const auto& buf : interm_bufs) {
    // 检查缓冲区形状是否静态，并计算其静态大小
    bool is_static = true;
    size_t size =
        elementSize(buf->dtype().scalar_type()) * buf->dtype().lanes();
    for (auto& d : buf->dims()) {
      if (!d->isConstant()) {
        is_static = false;
        break;
      }
      size = size * (*intValue(d));  // 计算维度大小
    }
    // 只为静态缓冲区分配内存
    if (!is_static) {
      remaining_interm_bufs.push_back(buf);  // 将非静态缓冲区添加到剩余列表中
      continue;
    }
    auto bp = (void*)malloc(size);  // 分配内存
    if (!bp) {
      remaining_interm_bufs.push_back(buf);  // 内存分配失败时将缓冲区添加到剩余列表中
      continue;
    }
    constants_.push_back({buf, bp});  // 将常量信息加入列表
  }
  return remaining_interm_bufs;  // 返回剩余的中间缓冲区列表
}

BlockPtr TensorExprKernel::bindAllInputs() {
  std::vector<CodeGen::BufferArg> symbolic_shape_args;  // 创建符号形状参数向量
  std::vector<CodeGen::BufferArg> symbolic_stride_args;  // 创建符号步长参数向量

  auto symbolic_shape_inputs_start_pos =
      nInputs_ - symbolic_shape_inputs_.size();
  if (has_symbolic_shapes_) {
    // 图形应具有表示符号维度的输入参数，这些参数位于输入列表的末尾。
    // `symbolic_shape_inputs_` 向量的大小定义了这些符号输入参数的数量。
    //
    // TODO: 检查具有符号形状的张量是否是连续的。
    TORCH_CHECK(
        nInputs_ > static_cast<int64_t>(symbolic_shape_inputs_.size()),
        "Symbolic dims not provided as inputs to the graph");
    // 首先，处理符号输入参数，并为每个参数创建一个新变量。
    // 注意：这必须在处理张量输入之前完成，因为它们的符号大小需要与我们为符号输入参数创建的变量关联。
    symbolic_shape_args.reserve(symbolic_shape_inputs_.size());

    for (size_t i = symbolic_shape_inputs_start_pos;
         i < static_cast<size_t>(nInputs_);
         ++i) {
      auto input = graph_->inputs()[i];
      // 检查输入的类型是否为整数类型
      if (input->type()->kind() != TypeKind::IntType) {
        throw std::runtime_error(
            "Expected integer type input to graph for symbolic dims.");
      }
      // 创建一个名为 "v" + input_name_map_[input] 的变量，类型为长整型
      VarHandle v("v" + input_name_map_[input], kLong);
      // 将变量添加到符号形状参数列表中
      symbolic_shape_args.emplace_back(v);
      // 将变量与输入关联起来，存储到scalars_映射中
      scalars_.emplace(input, v);
      // 将符号输入的位置映射到shapeSymbolInputPos_中
      shapeSymbolInputPos_[scalars_[input].node()] = i;
    }

    // 对于每个形状符号，存储到相应变量的映射关系
    for (size_t i = 0; i < symbolic_shape_inputs_.size(); ++i) {
      shapeSymbolToVar_[symbolic_shape_inputs_[i]] =
          scalars_[graph_->inputs()[symbolic_shape_inputs_start_pos + i]];
    }

    // 接下来，处理符号步长输入参数并为符号创建参数
    for (size_t i = 0; i < symbolic_shape_inputs_start_pos; ++i) {
      auto input = graph_->inputs()[i];
      auto tt = input->type()->cast<TensorType>();
      // 如果不是张量类型，则跳过
      if (!tt) {
        continue;
      }
      // 获取符号步长的描述
      auto symbolic_stride = getSymbolicStrideDesc(input);
      for (size_t j = 0; j < symbolic_stride.size(); ++j) {
        // 如果步长被标记为torch::jit::StrideInput::S_AS_ARG，则创建一个变量
        if (symbolic_stride[j] == torch::jit::StrideInput::S_AS_ARG) {
          VarHandle v("v" + input_name_map_[input], kLong);
          // 将变量添加到符号步长参数列表中
          symbolic_stride_args.emplace_back(v);
          // 将输入和步长的位置映射到strideArgToVar_中
          strideArgToVar_[{i, j}] = v;
          // 将输入和步长的位置对添加到input_stride_args_中
          input_stride_args_.emplace_back(i, j);
        }
      }
    }
  }

  // 创建一个块，用于收集所有张量对应的语句
  auto block = alloc<Block>(std::vector<StmtPtr>({}));

  // 处理符号形状参数之前的输入
  for (const auto i : c10::irange(symbolic_shape_inputs_start_pos)) {
    auto input = graph_->inputs()[i];
    // 将输入绑定到张量
    Tensor t = bindInput(input);
    // 如果张量有语句，则将其添加到块中
    if (t.stmt()) {
      block->append_stmt(t.stmt());
    }
  }

  // 现在，将所有符号形状参数对应的变量添加到bufferArgs_中
  bufferArgs_.insert(
      bufferArgs_.end(),
      symbolic_shape_args.begin(),
      symbolic_shape_args.end());

  // 现在，将所有符号步长输入对应的变量添加到bufferArgs_中
  bufferArgs_.insert(
      bufferArgs_.end(),
      symbolic_stride_args.begin(),
      symbolic_stride_args.end());

  // 返回块，其中包含所有的语句
  return block;
}

// 定义 TensorExprKernel 类的 deduceMemoryLayoutPolicy 方法
void TensorExprKernel::deduceMemoryLayoutPolicy() {
  // 如果张量是 channels-last 连续的，则首选内存布局传播策略是使用 channels-last。
  // 否则，首选策略是使用 contiguous。
  auto _prefer_symbolic_mem =
      [](const torch::jit::Value* val,
         const std::vector<torch::jit::StrideInput>& stride_desc_vec) {
        // 断言确保 stride_desc_vec 不为空
        TORCH_INTERNAL_ASSERT(!stride_desc_vec.empty());
        // 检查是否有符号化的步幅信息
        auto cur_stride_desc = stride_desc_vec[0];
        // 根据步幅信息选择内存布局策略
        return (cur_stride_desc ==
                torch::jit::StrideInput::TENSOR_CONT_CHANNELS_LAST)
            ? MemoryLayoutPolicy::kChannelsLastNdContiguous
            : MemoryLayoutPolicy::kContiguous;
      };

  auto _prefer_static_mem = [](const torch::jit::Value* val) {
    // 图中没有形状信息
    TORCH_INTERNAL_ASSERT(
        val->isCompleteTensor(),
        buildErrorMessage(val->debugName() + " is not a complete tensor."));
    const auto& tt = val->type()->expect<TensorType>();
    const auto sizes = *tt->sizes().concrete_sizes();
    const auto strides = *tt->strides().concrete_sizes();
    // 检查是否是 channels-last 的 2D 步幅
    return (c10::is_channels_last_strides_2d(sizes, strides))
        ? MemoryLayoutPolicy::kChannelsLastNdContiguous
        : MemoryLayoutPolicy::kContiguous;
  };

  // 从图的输入和输出中筛选出张量，以推断内存布局传播策略
  auto _is_tensor = [](const jit::Value* el) {
    return el->type()->kind() == TypeKind::TensorType;
  };
  std::vector<torch::jit::Value*> graph_io_tensors;
  std::copy_if(
      graph_->inputs().begin(),
      graph_->inputs().end(),
      std::back_inserter(graph_io_tensors),
      _is_tensor);
  std::copy_if(
      graph_->outputs().begin(),
      graph_->outputs().end(),
      std::back_inserter(graph_io_tensors),
      _is_tensor);
  // std::all_of 返回 true 如果范围为空。但是我们倾向于保留原始的内存布局传播策略。因此我们检查范围是否为空。
  auto prefer_channels_last = (!graph_io_tensors.empty());
  for (auto el : graph_io_tensors) {
    auto is_complete = el->isCompleteTensor();
    auto is_symbolic = symbolic_strides_.count(el);

    // 根据是否完整和是否有符号步幅选择首选的内存布局策略
    auto preferred_mem_layout = is_complete
        ? _prefer_static_mem(el)
        : (is_symbolic ? _prefer_symbolic_mem(el, symbolic_strides_[el])
                       : MemoryLayoutPolicy::kContiguous);
    // 如果首选的内存布局不是 channels-last，则设定 prefer_channels_last 为 false 并结束循环
    if (preferred_mem_layout != MemoryLayoutPolicy::kChannelsLastNdContiguous) {
      prefer_channels_last = false;
      break;
  }
}

// 如果所有输入和输出的内存布局都是 channels-last 连续的，
// 则传播的内存布局应为 channels-last。
// 否则，传播的内存布局是连续的，与当前情况相同。
memory_layout_policy_ = prefer_channels_last
    ? MemoryLayoutPolicy::kChannelsLastNdContiguous
    : MemoryLayoutPolicy::kContiguous;
// 输出当前张量表达式图以便调试，显示优化前的图形状态
GRAPH_DUMP("TensorExprKernel graph (Before graph optimization):", graph_);

// 由于在图形操作中可能会修改输出指针，因此在图形操作之前保存原始的输出列表，用于符号步长信息的同步
auto _orignal_graph_outputs = graph_->outputs().vec();

// 首先获取图的设备信息，图的优化可能与设备相关
device_ = *pickDeviceType(graph_);

// 推断内存布局策略
deduceMemoryLayoutPolicy();

// 将 Conv 与 Eltwise 操作融合
graph_rewrite_helper::replaceConvolutionWithAtenConv(graph_);
FuseConvWithEltwise(graph_);

// 优化连接操作
OptimizeCat(graph_);

// 同步符号步长信息
auto graph_outputs = graph_->outputs();
TORCH_INTERNAL_ASSERT(graph_outputs.size() == _orignal_graph_outputs.size());
for (int i : c10::irange(graph_outputs.size())) {
  auto el_orig = _orignal_graph_outputs.at(i);
  auto el_new = graph_outputs.at(i);
  // 如果原始张量存在符号步长信息并且原始张量与新张量不同，则将符号步长信息同步到新张量，然后删除原始张量的步长信息
  if (symbolic_strides_.count(el_orig) && (el_orig != el_new)) {
    symbolic_strides_[el_new] = symbolic_strides_[el_orig];
    symbolic_strides_.erase(el_orig);
  }
}

// 输出优化后的张量表达式图以便调试，显示优化后的图形状态
GRAPH_DUMP("TensorExprKernel graph (After graph optimization):", graph_);
  } else {
    // 遍历节点的输出
    for (auto const& output : n->outputs()) {
      // 如果输出被使用
      if (output->hasUses()) {
        // 计算输出的值
        Tensor t = computeValue(output);

        // 如果在 ExternalCall 之前有如下结构的 for 循环:
        //   stmt1: for:
        //   stmt2    for:
        //   stmt3: ExternalCall
        // 则前面的 for 循环不能并行化。因此我们标记 ExternalCall 的 buf args，
        // 以确保其前面的循环仍然可以并行化。
        if (to<ExternalCall>(t.stmt())) {
          auto _external_call = to<ExternalCall>(t.stmt());
          // 将 ExternalCall 的 buf args 加入到 bufsToBeParallelized_ 集合中
          for (const auto& _buf : _external_call->buf_args()) {
            bufsToBeParallelized_.insert(_buf);
          }
        }

        // 如果输出是 Tensor 类型
        if (output->type()->cast<TensorType>()) {
          // 将 Tensor 对应的 buf 加入 bufs_ 中
          if (t.buf()) {
            bufs_.emplace(output, t.buf());
          }
          // 将计算结果的语句追加到 block 中
          block->append_stmt(t.stmt());
        } else {
          // 如果输出是标量

          // 在 TE 中，我们用一对语句表示标量计算:
          //   Let val = <compute_expression>
          //   Store(buf_for_scalar[0], val)
          //
          // 后续的计算将使用 val，当需要将计算值作为内核的输出时，将使用 buffer。
          // 如果这不是一个输出，Store 将会在 DCE (Dead Code Elimination) 阶段被移除。
          //
          // 注意: NNC 的 lowering 函数返回 Tensor，即一对 <Buf, Stmt>，
          // 但是在这里我们还需要 Var。如何同时获取 Var、Buf 和 Stmt 呢？
          // 我们使用以下技巧: lowering 函数创建 Let-stmt 和一个 "虚假" 缓冲区，
          // 其唯一目的是保存 Var。然后在 lowering 函数之外（也就是在这里），
          // 我们生成 Store 和实际的缓冲区。
          VarPtr v = t.buf()->base_handle();
          // 将标量输出添加到 scalars_ 中
          scalars_[output] = VarHandle(v);
          // 将计算结果的语句追加到 block 中
          block->append_stmt(t.stmt());
          // 创建标量的缓冲区
          std::vector<ExprPtr> dims;
          BufHandle buf(
              "scalar_" + sanitizeName(output->debugName()), {}, v->dtype());
          // 生成 Store 语句
          StmtPtr store = Store::make(buf, {}, ExprHandle(v));
          // 将 Store 语句追加到 block 中
          block->append_stmt(store);
          // 将输出和对应的缓冲区添加到 bufs_ 中
          bufs_.emplace(output, buf.node());
        }
      }
    }
  }
  // 如果同时存在随机数和广播操作
  if (hasRandom_ && hasBroadcast_) {
    // 抛出运行时错误
    throw std::runtime_error(
        "Cannot support broadcast and random within one kernel");
  }
}

// 将输出操作数从 bufs_ 移动到 bufOutputs_
for (auto i : c10::irange(graph_->outputs().size())) {
  auto& output = graph_->outputs().at(i);
  // 如果 bufs_ 中不存在输出 Tensor
  if (!bufs_.count(output)) {
    // 抛出格式错误异常
    throw malformed_input("cannot find output Tensor");
  }
    if (!output->type()->cast<TensorType>()) {
      // 如果输出不是张量类型，表示是标量输出，将其表示为0维缓冲区。
      bufOutputs_.insert(bufs_.at(output));  // 将缓冲区插入输出缓冲区集合
      bufsToBeParallelized_.insert(bufs_.at(output));  // 将缓冲区插入待并行化的缓冲区集合
      bufferArgs_.emplace_back(BufHandle(bufs_.at(output)));  // 将缓冲区句柄添加到缓冲区参数列表
      tensorOutputTensorOptions_.emplace_back(
          c10::TensorOptions(tensorType(bufs_.at(output))).device(device_));  // 添加张量选项到张量输出选项列表
      tensorOutputSizes_.emplace_back();  // 添加一个空的尺寸列表作为标量输出的大小
      tensorOutputStrides_.emplace_back();  // 添加一个空的步幅列表作为标量输出的步幅
      isOutputScalar_.push_back(true);  // 标记输出为标量
      bufs_.erase(output);  // 从缓冲区映射中移除输出
      continue;  // 继续处理下一个输出
    }

    const auto& tt = output->type()->expect<TensorType>();
    if (has_symbolic_shapes_) {
      // 如果存在符号形状，从符号形状推导尺寸，并添加到符号尺寸列表
      auto sizes = sizesFromSymbolicShape(tt->symbolic_sizes());
      tensorOutputSymbolicSizes_.push_back(sizes);
      TORCH_INTERNAL_ASSERT(symbolic_strides_.count(output));  // 断言确保符号步幅已计算
      auto stride_desc_vec = symbolic_strides_[output];
      TORCH_INTERNAL_ASSERT(stride_desc_vec.size() == 1);  // 断言确保只有一个步幅描述
      auto stride_desc = stride_desc_vec[0];
      tensorOutputStrideDesc_.push_back(stride_desc);  // 添加符号步幅描述到步幅描述列表
      // 将符号输出转换为正确的步幅张量
      Tensor properly_strided_output =
          convertSymbolicOutputToCorrectStrides(output);
      if (properly_strided_output.stmt()) {
        block->append_stmt(properly_strided_output.stmt());  // 将转换后的语句附加到代码块中
      }
      bufs_[output] = properly_strided_output.buf();  // 更新缓冲区映射
    } else {
      // 如果不存在符号形状，将静态形状输出转换为正确的步幅张量
      Tensor properly_strided_output =
          convertStaticShapeOutputToCorrectStrides(output);
      if (properly_strided_output.stmt()) {
        block->append_stmt(properly_strided_output.stmt());  // 将转换后的语句附加到代码块中
      }
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      bufs_[output] = properly_strided_output.buf();  // 更新缓冲区映射
      auto sizes = *tt->sizes().concrete_sizes();
      tensorOutputSizes_.push_back(sizes);  // 添加静态尺寸到尺寸列表
      auto strides = tt->strides().concrete_sizes();

      // 如果张量不是密集或有重叠，无法匹配配置文件中的步幅
      if (strides && denseAndNonOverlapping(sizes, *strides)) {
        tensorOutputStrides_.push_back(*strides);  // 添加配置文件中的步幅到步幅列表
      } else {
        tensorOutputStrides_.push_back(TensorType::contiguousStridesOf(sizes));  // 添加连续步幅到步幅列表
      }
    }

    bufOutputs_.insert(bufs_.at(output));  // 将缓冲区插入输出缓冲区集合
    bufsToBeParallelized_.insert(bufs_.at(output));  // 将缓冲区插入待并行化的缓冲区集合
    bufferArgs_.emplace_back(BufHandle(bufs_.at(output)));  // 将缓冲区句柄添加到缓冲区参数列表
    tensorOutputTensorOptions_.emplace_back(
        c10::TensorOptions(tensorType(bufs_.at(output))).device(device_));  // 添加张量选项到张量输出选项列表
    isOutputScalar_.push_back(false);  // 标记输出不是标量
    bufs_.erase(output);  // 从缓冲区映射中移除输出
  }

  BackendType backendType = inferBackendTypeFromDevice(device_);
  stmt_ = transformLoops(backendType, block);  // 使用推断的后端类型和代码块进行循环变换

  for (const auto& c : constants_) {
    bufferArgs_.emplace_back(BufHandle(c.buf));  // 将常量缓冲区句柄添加到缓冲区参数列表
  }

  if (has_symbolic_shapes_) {
    tensorOutputSizes_.resize(bufOutputs_.size());  // 调整符号形状输出尺寸列表大小为输出缓冲区集合大小
    tensorOutputStrides_.resize(bufOutputs_.size());
  }

  // 生成代码。
  // 使用给定的后端类型生成代码生成器对象，传入语句、缓冲区参数、设备和内核函数名称
  codegen_ = CreateCodeGen(
      getCodeGenName(backendType),
      stmt_,
      bufferArgs_,
      device_,
      kernel_func_name_);
}

void TensorExprKernel::recompile() {
  // 创建一个新的代码生成器实例，用于生成 LLVM 代码
  codegen_ = CreateCodeGen(
      "llvm_codegen", stmt_, bufferArgs_, device_, kernel_func_name_);
}

TensorExprKernel::TensorExprKernel(
    const std::shared_ptr<Graph>& subgraph,
    const std::string& kernel_func_name,
    std::unordered_map<c10::Symbol, NNCLoweringFunction> custom_lowerings,
    std::vector<int64_t> symbolic_shape_inputs,
    bool pre_alloc /*= false*/,
    std::unordered_map<
        const torch::jit::Value*,
        std::vector<torch::jit::StrideInput>> symbolic_strides)
    : graph_(subgraph),
      code_(subgraph, ""),
      symbolic_shape_inputs_(std::move(symbolic_shape_inputs)),
      custom_lowerings_(std::move(custom_lowerings)),
      pre_alloc_(pre_alloc),
      kernel_func_name_(kernel_func_name),
      symbolic_strides_(std::move(symbolic_strides)) {
  // 优化包含的图形
  optimizeOwningGraph();

  // 检查是否允许使用后备机制
  allow_fallback_ = fallbackAllowed();

  // 如果不允许后备机制，则直接编译张量表达式内核
  if (!allow_fallback_) {
    compile();
    return;
  }

  // 强制使用后备机制
  use_fallback_ = fallbackEnforced();
  if (use_fallback_) {
    return;
  }

  // 尝试编译张量表达式内核，如果失败则启用后备机制
  try {
    compile();
  } catch (...) {
    use_fallback_ = true;
  }
}

void TensorExprKernel::run(Stack& stack) const {
  // 如果不使用和不允许后备机制，则直接运行内核
  if (!use_fallback_ && !allow_fallback_) {
    runKernel(stack);
  } 
  // 如果不使用后备机制但允许后备机制，则尝试运行内核，失败时启用后备机制
  else if (!use_fallback_ && allow_fallback_) {
    try {
      runKernel(stack);
    } catch (...) {
      fallback(stack);
    }
  } 
  // 启用后备机制
  else {
    fallback(stack);
  }
}

void TensorExprKernel::getStaticOutputSizesAndStrides(
    const at::ArrayRef<IValue>& inputs,
    std::vector<std::vector<int64_t>>* sizes,
    std::vector<std::vector<int64_t>>* strides) const {
  // 确保存在符号形状
  TORCH_INTERNAL_ASSERT(has_symbolic_shapes_);
  // 如果存在符号形状，则输出张量大小无法在编译时计算，需要在这里使用传入的符号形状输入参数计算
  TORCH_INTERNAL_ASSERT(
      tensorOutputSymbolicSizes_.size() == bufOutputs_.size());

  // 确保 sizes 和 strides 非空
  TORCH_INTERNAL_ASSERT(sizes);
  TORCH_INTERNAL_ASSERT(strides);
  // 将静态大小和步幅赋值给 sizes 和 strides
  *sizes = tensorOutputSizes_;
  *strides = tensorOutputStrides_;
  auto& static_sizes = *sizes;
  auto& static_strides = *strides;
  // 对每个输出张量进行处理
  for (size_t i = 0, e = bufOutputs_.size(); i < e; ++i) {
    static_sizes[i].clear();
    // 遍历符号形状，根据情况获取值
    for (auto t : tensorOutputSymbolicSizes_[i]) {
      if (t.AsNode<LongImm>()) {
        static_sizes[i].emplace_back(immediateAs<int64_t>(t.node()));
      } else {
        auto input_pos = shapeSymbolInputPos_.at(t.node());
        // 确保输入位置在有效范围内
        TORCH_INTERNAL_ASSERT(input_pos < inputs.size());
        // 确保输入是整数类型
        TORCH_INTERNAL_ASSERT(inputs[input_pos].isInt());
        static_sizes[i].emplace_back(inputs[input_pos].toInt());
      }
    }

    // 如果张量步幅描述为连续的，则使用连续的步幅计算
    if (tensorOutputStrideDesc_[i] == torch::jit::StrideInput::TENSOR_CONT) {
      static_strides[i] = TensorType::contiguousStridesOf(static_sizes[i]);
    } else if (
        tensorOutputStrideDesc_[i] ==
        torch::jit::StrideInput::TENSOR_CONT_CHANNELS_LAST) {
      // 如果张量的输出步幅描述为“通道最后”，则调用特定函数获取通道最后格式的二维步幅
      static_strides[i] = at::get_channels_last_strides_2d(static_sizes[i]);

    } else {
      // 否则，将张量输出步幅描述转换为字符串
      std::string output_desc = toString(tensorOutputStrideDesc_[i]);
      // 断言，如果不符合预期（连续或通道最后），输出错误信息和当前描述
      TORCH_INTERNAL_ASSERT(
          false, "Expected contiguous or channels last, got ", output_desc);
    }
// 准备运行参数 `runArgs`，该函数根据输入和输出参数构建调用所需的参数列表
std::vector<CodeGen::CallArg> TensorExprKernel::prepareRunArgs(
    const at::ArrayRef<IValue>& inputs, // 输入参数列表的引用
    std::vector<at::Tensor>& outputs) const { // 输出参数列表的引用

  // 预留足够空间以存放所有运行参数，包括输入、输入步长参数和输出缓冲区
  std::vector<CodeGen::CallArg> runArgs;
  runArgs.reserve(
      inputs.size() + input_stride_args_.size() + bufOutputs_.size());

  // 遍历输入参数列表，根据参数类型将其添加到 `runArgs`
  for (auto& input : inputs) {
    if (input.isInt()) {
      runArgs.emplace_back(input.toInt()); // 将整数转换为调用参数并添加到 `runArgs`
    } else if (input.isBool()) {
      runArgs.emplace_back(input.toBool()); // 将布尔值转换为调用参数并添加到 `runArgs`
    } else if (input.isDouble()) {
      runArgs.emplace_back(input.toDouble()); // 将双精度浮点数转换为调用参数并添加到 `runArgs`
    } else if (input.isTensor()) {
      runArgs.emplace_back(input.toTensor().data_ptr()); // 将张量的数据指针添加到 `runArgs`
    }
  }

  // 如果存在符号形状，则需要获取静态输出大小和步长
  if (has_symbolic_shapes_) {
    std::vector<std::vector<int64_t>> static_sizes;
    std::vector<std::vector<int64_t>> static_strides;
    getStaticOutputSizesAndStrides(inputs, &static_sizes, &static_strides);

    // 添加输入步长参数到 `runArgs`
    for (const auto& input_stride_arg : input_stride_args_) {
      runArgs.emplace_back(
          inputs[input_stride_arg.first].toTensor().strides().at(
              input_stride_arg.second)); // 将输入步长参数添加到 `runArgs`
    }

    // 根据静态大小和步长创建输出张量，并将其数据指针添加到 `runArgs`
    for (size_t i = 0, e = bufOutputs_.size(); i < e; ++i) {
      auto const& opts = tensorOutputTensorOptions_[i];
      outputs.emplace_back(codegen_->empty_strided(
          static_sizes[i],
          static_strides[i],
          opts.dtype,
          opts.layout,
          opts.device,
          opts.pinned_memory));
      runArgs.emplace_back(outputs.back().data_ptr()); // 将输出张量的数据指针添加到 `runArgs`
    }
  } else {
    // 根据预定义的张量大小和步长创建输出张量，并将其数据指针添加到 `runArgs`
    for (size_t i = 0, e = bufOutputs_.size(); i < e; ++i) {
      auto const& opts = tensorOutputTensorOptions_[i];
      outputs.emplace_back(codegen_->empty_strided(
          tensorOutputSizes_[i],
          tensorOutputStrides_[i],
          opts.dtype,
          opts.layout,
          opts.device,
          opts.pinned_memory));
      runArgs.emplace_back(outputs.back().data_ptr()); // 将输出张量的数据指针添加到 `runArgs`
    }
  }

  // 将常量参数的指针添加到 `runArgs`
  for (const auto& c : constants_) {
    runArgs.emplace_back(c.ptr);
  }

  return runArgs; // 返回构建好的调用参数列表
}

// 获取代码生成器的语句并返回
StmtPtr TensorExprKernel::getCodeGenStmt() {
  return codegen_->stmt(); // 返回代码生成器生成的语句
}

// 运行核函数，更新输入栈中的值并将输出添加到栈中
void TensorExprKernel::runKernel(Stack& stack) const {
  // 获取输入参数
  auto inputs = last(stack, nInputs_);
  std::vector<at::Tensor> outputs;

  // 准备运行参数
  std::vector<CodeGen::CallArg> runArgs = prepareRunArgs(inputs, outputs);

  // 调用核函数
  codegen_->call(runArgs);

  // 更新输入栈
  drop(stack, nInputs_);

  // 遍历输出张量，将标量值或张量添加到栈中
  int64_t idx = 0;
  for (auto& o : outputs) {
    if (isOutputScalar_[idx++]) {
      // 标量输出被作为0维张量返回，需要从中提取标量值
      push_one(stack, o.item());
    } else {
      push_one(stack, std::move(o));
    }
  }
}

void TensorExprKernel::runFast(
    const std::vector<void*>& inputs,
    //
    const std::vector<void*>& outputs) const {
```  
// 定义一个成员函数，接受一个常量引用的 vectors 类型 inputs，使用该类型创建 args，然后根据输入的 outputs 对象大小来扩展 args 的容量，并插入 outputs 的所有元素。


  std::vector<void*> args(inputs);
  args.reserve(inputs.size() + outputs.size() + constants_.size());
  args.insert(args.end(), outputs.begin(), outputs.end());
```  
// 创建一个名为 args 的 vectors，其元素类型为 void 指针，该 vectors 以 inputs vectors 为初始输入创建，预留容量以容纳 inputs，outputs 和 constants_ vectors 的元素，将 outputs vectors 的元素插入到 args vectors 的末尾。


  // TODO: we can consider preallocating and pre-filling the args vector.
  for (const auto& c : constants_) {
    args.push_back(c.ptr);
  }
```  
// 遍历 constants_ vectors 中的每个元素 c，将其指针 c.ptr 添加到 args vectors 的末尾。


  // Call the kernel.
  codegen_->call_raw(args);
```  
// 调用 codegen_ 对象的 call_raw 方法，传入 args vectors 作为参数。
}

// 在分配输出后运行张量表达式内核
void TensorExprKernel::runWithAllocatedOutputs(Stack& stack) const {
  // 断言设备为 CPU，只支持预先分配输出张量在 CPU 上的情况
  TORCH_INTERNAL_ASSERT(
      device_ == at::kCPU,
      "Pre-allocated output tensors are supported only on CPUs.");

  // 用于存储函数参数的向量
  std::vector<void*> args;
  args.reserve(nInputs_ + nOutputs_ + constants_.size());

  // stack 中输入在顶部，输出在其下方
  auto stack_ivals = last(stack, nOutputs_ + nInputs_);
  auto stack_outputs = stack_ivals.slice(0, nOutputs_);
  auto stack_inputs = stack_ivals.slice(nOutputs_);

  // 用于存储输入张量的整数形式
  std::vector<int64_t> int_inputs(nInputs_);
  for (auto i : c10::irange(nInputs_)) {
    auto inp = stack_inputs[i];
    if (inp.isInt()) {
      int_inputs[i] = inp.toInt();
      args.emplace_back(&int_inputs[i]);
    } else if (inp.isTensor()) {
      args.emplace_back(inp.toTensor().data_ptr());
    } else {
      // 处理未处理的输入类型异常
      TORCH_INTERNAL_ASSERT(
          false, "Unhandled input type while calling TensorExprKernel");
    }
  }

  // 用于存储输入张量的步幅值
  std::vector<int64_t> stride_values(input_stride_args_.size());
  if (has_symbolic_shapes_) {
    std::vector<std::vector<int64_t>> static_sizes;
    std::vector<std::vector<int64_t>> static_strides;
    // 获取静态输出大小和步幅
    getStaticOutputSizesAndStrides(
        stack_inputs, &static_sizes, &static_strides);

    // 添加步幅参数
    for (auto idx : c10::irange(input_stride_args_.size())) {
      const auto& input_stride_arg = input_stride_args_[idx];
      stride_values[idx] =
          stack_inputs[input_stride_arg.first].toTensor().strides().at(
              input_stride_arg.second);
      args.emplace_back(&stride_values[idx]);
    }

    // 断言输出数量与缓冲区大小相符
    TORCH_INTERNAL_ASSERT(
        nOutputs_ == static_cast<int64_t>(bufOutputs_.size()));
    for (size_t i = 0, e = bufOutputs_.size(); i < e; ++i) {
      auto& out = stack_outputs[i].toTensor();
      // 在 CPU 上进行大小调整
      // TODO: 在 GPU 上进行测试
      out.resize_(static_sizes[i]);
      args.emplace_back(out.data_ptr());
    }
  } else {
    // 对于非符号形状的情况，直接添加输出张量指针
    for (auto i : c10::irange(nOutputs_)) {
      args.emplace_back(stack_outputs[i].toTensor().data_ptr());
    }
  }

  // 添加常量参数
  for (const auto& c : constants_) {
    args.emplace_back(c.ptr);
  }

  // 调用内核函数
  codegen_->call_raw(args);

  // 从栈中移除输入，输出已经位于输入之下
  drop(stack, nInputs_);
}
```