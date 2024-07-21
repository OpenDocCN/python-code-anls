# `.\pytorch\torch\csrc\jit\tensorexpr\operators\quantization.cpp`

```
// 包含标量类型定义头文件
#include <c10/core/ScalarType.h>
// 包含张量表达式简化器头文件
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
// 包含张量表达式中的各种操作头文件
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/pointwise.h>
#include <torch/csrc/jit/tensorexpr/operators/quantization.h>

// 使用torch::jit::tensorexpr命名空间
using namespace torch::jit::tensorexpr;

// torch::jit::tensorexpr命名空间下的匿名命名空间
namespace torch {
namespace jit {
namespace tensorexpr {
namespace {

// 将ArgValue转换为包含两个int64_t的vector
std::vector<int64_t> _pair_int(ArgValue v) {
  if (auto t = std::get_if<IntList>(&v)) {
    return {(*t)[0], (*t)[1]};
  }
  auto i = std::get<int64_t>(v);
  return {i, i};
}

} // namespace

// 返回BufHandle中存储的缩放量的值（假定存在）
double immQScale(const BufHandle& qx) {
  TORCH_INTERNAL_ASSERT(
      qx.node()->qscale(), buildErrorMessage("Expects BufHandle with qscale"));
  // 简化缩放量表达式并返回其双精度浮点值
  return to<DoubleImm>(IRSimplifier::simplify(qx.node()->qscale()))->value();
}

// 返回BufHandle中存储的零点偏移值的整数值（假定存在）
int64_t immQZero(const BufHandle& qx) {
  TORCH_INTERNAL_ASSERT(
      qx.node()->qzero(), buildErrorMessage("Expects BufHandle with qzero"));
  // 简化零点偏移表达式并返回其整数值
  return to<LongImm>(IRSimplifier::simplify(qx.node()->qzero()))->value();
}

// 返回BufHandle中存储的数据类型
ScalarType immQDType(const BufHandle& qx) {
  return qx.dtype().scalar_type();
}

// 检查BufHandle是否具有量化参数
bool isQuantized(const BufHandle& qx) {
  return qx.node()->qscale() && qx.node()->qzero();
}

// 创建一个以通道为最后维度的量化BufHandle
static BufHandle makeQBufHandleChannelsLast(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const ExprPtr qscale,
    const ExprPtr qzero) {
  // 创建一个BufHandle对象
  BufHandle ResultBuf(name, dims, dtype);
  // 设置BufHandle对象的量化缩放量和零点偏移
  ResultBuf.node()->set_qscale(qscale);
  ResultBuf.node()->set_qzero(qzero);
  // 设置BufHandle对象的步长以支持通道为最后维度的布局
  ResultBuf.node()->set_strides(make_channels_last_strides(dims));
  // 返回创建的BufHandle对象
  return ResultBuf;
}

// 创建一个以通道为最后维度的量化BufHandle（重载，直接传入双精度浮点数和整数）
static BufHandle makeQBufHandleChannelsLast(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const double qscale,
    const int64_t qzero) {
  // 调用前面定义的重载函数，转换双精度浮点数和整数为表达式，并创建BufHandle对象
  return makeQBufHandleChannelsLast(
      name,
      dims,
      dtype,
      DoubleImm::make(qscale).node(),
      LongImm::make(qzero).node());
}

// 创建一个连续存储的量化BufHandle
static BufHandle makeQBufHandleContiguous(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const ExprPtr qscale,
    const ExprPtr qzero) {
  // 创建一个BufHandle对象
  BufHandle ResultBuf(name, dims, dtype);
  // 设置BufHandle对象的量化缩放量和零点偏移
  ResultBuf.node()->set_qscale(qscale);
  ResultBuf.node()->set_qzero(qzero);
  // 设置BufHandle对象的步长以支持连续存储的布局
  ResultBuf.node()->set_strides(make_contiguous_strides(dims));
  // 返回创建的BufHandle对象
  return ResultBuf;
}

// 创建一个连续存储的量化BufHandle（重载，直接传入双精度浮点数和整数）
static BufHandle makeQBufHandleContiguous(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const double qscale,
    const int64_t qzero) {
  // 调用前面定义的重载函数，转换双精度浮点数和整数为表达式，并创建BufHandle对象
  return makeQBufHandleContiguous(
      name,
      dims,
      dtype,
      DoubleImm::make(qscale).node(),
      LongImm::make(qzero).node());
}

// 检查BufHandle是否以通道为最后维度的布局
static bool isChannelsLast(const BufHandle& buf) {
  const auto& strides = buf.node()->strides();
  const auto& dims = buf.node()->dims();
  const auto rank = dims.size();
  // 如果张量维度小于3，则无法判断通道为最后维度的情况
  if (rank < 3) {
   `
  # 返回 false，表示某种条件不满足（该行代码后面可能缺少相关逻辑）
  return false;
  
  # 将 dims[1] 的值简化，并转换为 LongImm 类型，获取简化后的值
  auto dimsC = to<LongImm>(IRSimplifier::simplify(dims[1]))->value();
  
  # 将 strides[1] 的值简化，并转换为 LongImm 类型，获取简化后的值
  auto stridesC = to<LongImm>(IRSimplifier::simplify(strides[1]))->value();
  
  # 将 strides[rank - 1] 的值简化，并转换为 LongImm 类型，获取简化后的值
  auto stridesLast =
      to<LongImm>(IRSimplifier::simplify(strides[rank - 1]))->value();

  # 检查 stridesLast 是否等于 dimsC 并且 stridesC 是否等于 1，满足条件则返回 true，否则返回 false
  return ((stridesLast == dimsC) && (stridesC == 1));
}

// 对输入进行量化操作，返回量化后的表达式
static ExprHandle quant(
    ExprHandle x,
    Dtype out_dtype,
    ExprHandle qscale,
    ExprHandle qzero) {
  // 将 qscale 提升为与 x 相同的数据类型
  auto promoted_qscale =
      promoteToDtype(std::move(qscale), x.dtype().scalar_type());
  // 将 qzero 提升为与 x 相同的数据类型
  auto promoted_qzero =
      promoteToDtype(std::move(qzero), x.dtype().scalar_type());
  // 返回量化后的表达式
  return promoteToDtype(
      x / promoted_qscale + promoted_qzero + FloatImm::make(0.5f),
      out_dtype.scalar_type());
}

// 对输入进行反量化操作，返回反量化后的表达式
static ExprHandle dequant(
    ExprHandle qx,
    Dtype out_dtype,
    ExprHandle qscale,
    ExprHandle qzero) {
  // 将 qx 提升为与 out_dtype 相同的数据类型
  auto qx_promoted = promoteToDtype(std::move(qx), out_dtype.scalar_type());
  // 将 qscale 提升为与 out_dtype 相同的数据类型
  auto qscale_promoted =
      promoteToDtype(std::move(qscale), out_dtype.scalar_type());
  // 将 qzero 提升为与 out_dtype 相同的数据类型
  auto qzero_promoted =
      promoteToDtype(std::move(qzero), out_dtype.scalar_type());
  // 返回反量化后的表达式
  return promoteToDtype(
      (qx_promoted - qzero_promoted) * qscale_promoted,
      out_dtype.scalar_type());
}

// 计算每个张量的量化操作
Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>&,
    at::Device) {
  // 创建变量和索引
  std::vector<VarPtr> vars;
  std::vector<ExprHandle> indices;
  for (const auto& os : outputShape) {
    auto var = alloc<Var>("", os.node()->dtype());
    vars.push_back(var);
    indices.push_back(VarHandle(var));
  }

  // 获取量化参数 qscale 和 qzero
  ExprHandle qscale = constant(inputs[1]);
  ExprHandle qzero = constant(inputs[2]);

  // 确定输出数据类型
  const auto dtype = [](auto qdtype) {
    if (static_cast<int64_t>(ScalarType::QInt8) == qdtype) {
      return Dtype(ScalarType::QInt8);
    } else if (static_cast<int64_t>(ScalarType::QUInt8) == qdtype) {
      return Dtype(ScalarType::QUInt8);
    }
    throw malformed_input("Expected quantized dtype");
  }(std::get<int64_t>(inputs[3]));

  // 进行量化操作
  ExprHandle e =
      quant(tensorOrConstant(inputs[0], indices), dtype, qscale, qzero);

  // 创建缓冲区并返回张量
  BufPtr buf = alloc<Buf>(
      "quantize_per_tensor",
      ExprHandleVectorToExprVector(outputShape),
      dtype,
      nullptr,
      c10::nullopt,
      qscale.node(),
      qzero.node());
  return Tensor(buf, vars, e.node());
}

// 计算量化加法操作
Tensor computeQuantizedAdd(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    // 检查设备类型为 AT::Device
    if (::Device) {
      // 从输入中获取 QA 缓冲区句柄
      const BufHandle& QA = std::get<BufHandle>(inputs[0]);
      // 从输入中获取 QB 缓冲区句柄
      const BufHandle& QB = std::get<BufHandle>(inputs[1]);
      // 获取 QA 的量化比例因子
      auto qa_scale = ExprHandle(QA.node()->qscale());
      // 获取 QA 的量化零点
      auto qa_zero = ExprHandle(QA.node()->qzero());
      // 获取 QB 的量化比例因子
      auto qb_scale = ExprHandle(QB.node()->qscale());
      // 获取 QB 的量化零点
      auto qb_zero = ExprHandle(QB.node()->qzero());
      // 创建输出的量化比例因子表达式
      ExprHandle out_qscale = DoubleImm::make(std::get<double>(inputs[2]));
      // 创建输出的量化零点表达式
      ExprHandle out_qzero = LongImm::make(std::get<int64_t>(inputs[3]));
      // 设置反量化后的数据类型为 kFloat
      Dtype dequant_dtype = kFloat;
      // 如果指定了输出类型，则使用该类型；否则使用 QA 的数据类型作为输出类型
      Dtype out_dtype = outputType ? Dtype(*outputType) : QA.dtype();
      // 创建变量和索引列表以便生成输出的形状
      std::vector<VarPtr> vars;
      std::vector<ExprHandle> indices;
      // 遍历输出形状的每个维度
      for (const auto& os : outputShape) {
        // 为当前维度创建一个变量
        auto var = alloc<Var>("", os.node()->dtype());
        vars.push_back(var);
        // 将变量的句柄添加到索引列表中
        indices.push_back(VarHandle(var));
      }
      // 将输入 QA 按索引张量化或常数化
      auto lhs = tensorOrConstant(inputs[0], indices);
      // 将输入 QB 按索引张量化或常数化
      auto rhs = tensorOrConstant(inputs[1], indices);
      // 构建量化加法表达式
      ExprHandle exprHandle = quant(
          dequant(lhs, dequant_dtype, qa_scale, qa_zero) +
              dequant(rhs, dequant_dtype, qb_scale, qb_zero),
          out_dtype,
          out_qscale,
          out_qzero);
      // 创建输出缓冲区指针
      BufPtr buf = alloc<Buf>(
          "quantized_add",
          ExprHandleVectorToExprVector(outputShape),
          out_dtype,
          nullptr,
          isChannelsLast(QA) ? make_channels_last_strides(outputShape)
                             : make_contiguous_strides(outputShape),
          out_qscale.node(),
          out_qzero.node());
      // 返回表示张量的 Tensor 对象
      return Tensor(buf, vars, exprHandle.node());
    }
}

Tensor computeQuantizePerTensorExternalCall(
    const std::vector<ArgValue>& inputs,  // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出张量形状
    const std::vector<ExprHandle>& outputStrides,  // 输出张量步长
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device) {  // 设备类型

  const BufHandle& x = std::get<BufHandle>(inputs[0]);  // 获取输入缓存句柄
  const auto qscale = std::get<double>(inputs[1]);  // 获取量化比例
  const auto qzero = std::get<int64_t>(inputs[2]);  // 获取零点偏移
  const auto qdtype = std::get<int64_t>(inputs[3]);  // 获取量化数据类型

  const auto dtype = [](auto qdtype) {  // 根据量化数据类型确定数据类型
    if (static_cast<int64_t>(ScalarType::QInt8) == qdtype) {
      return Dtype(ScalarType::QInt8);
    } else if (static_cast<int64_t>(ScalarType::QUInt8) == qdtype) {
      return Dtype(ScalarType::QUInt8);
    }
    throw malformed_input("Expected quantized dtype");
  }(qdtype);

  auto ResultBuf = [&]() {  // 根据输入缓存是否通道为最后维度，创建量化缓存句柄
    if (isChannelsLast(x)) {
      return makeQBufHandleChannelsLast(
          "quantize_per_tensor", outputShape, dtype, qscale, qzero);
    }
    return makeQBufHandleContiguous(
        "quantize_per_tensor", outputShape, dtype, qscale, qzero);
  }();

  StmtPtr s = ExternalCall::make(  // 创建外部调用语句
      ResultBuf, "nnc_aten_quantize_per_tensor", {x}, {qscale, qzero, qdtype});

  return Tensor(ResultBuf.node(), s);  // 返回量化后的张量
}

Tensor computeDequantizeExternalCall(
    const std::vector<ArgValue>& inputs,  // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出张量形状
    const std::vector<ExprHandle>& outputStrides,  // 输出张量步长
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device) {  // 设备类型

  Dtype dtype = kFloat;  // 默认输出数据类型为浮点数
  if (outputType) {  // 如果有指定输出数据类型，则使用指定的类型
    dtype = Dtype(*outputType);
  }

  const BufHandle& qx = std::get<BufHandle>(inputs[0]);  // 获取输入缓存句柄
  const int64_t qdtype = (int64_t)immQDType(qx);  // 获取输入缓存的量化数据类型

  BufHandle ResultBuf("dequantize", outputShape, dtype);  // 创建反量化结果缓存句柄

  StmtPtr s = ExternalCall::make(  // 创建外部调用语句
      ResultBuf,
      "nnc_aten_dequantize",
      {qx},
      {ExprHandle(IRSimplifier::simplify(qx.node()->qscale())),
       ExprHandle(IRSimplifier::simplify(qx.node()->qzero())),
       qdtype});

  return Tensor(ResultBuf.node(), s);  // 返回反量化后的张量
}

Tensor computeQuantizedConv2dPrepack(
    const std::vector<ArgValue>& inputs,  // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出张量形状
    const std::vector<ExprHandle>& outputStrides,  // 输出张量步长
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device) {  // 设备类型

  Dtype dtype = kFloat;  // 默认输出数据类型为浮点数
  if (outputType) {  // 如果有指定输出数据类型，则使用指定的类型

    dtype = Dtype(*outputType);
  }

  const BufHandle& x = std::get<BufHandle>(inputs[0]);  // 获取输入缓存句柄

  // 省略部分代码，根据上下文推断

  return Tensor(ResultBuf.node(), s);  // 返回预打包的量化卷积结果张量
}
    # 使用 Dtype 类创建一个 dtype 对象，使用 outputType 中的参数
    dtype = Dtype(*outputType);
  }

  # 创建一个名为 ResultBuf 的 BufHandle 对象，用于保存 quantized_conv2d_prepack 的结果
  BufHandle ResultBuf("quantized_conv2d_prepack", outputShape, dtype);
  # 从 inputs 中获取第一个输入的 BufHandle，命名为 qw
  const BufHandle& qw = std::get<BufHandle>(inputs[0]);
  # 从 inputs 中获取第二个输入的 BufHandle，命名为 b
  const BufHandle& b = std::get<BufHandle>(inputs[1]);
  # 将第三个输入转换为包含两个整数的 strides 对象
  auto strides = _pair_int(inputs[2]);
  # 将第四个输入转换为包含两个整数的 padding 对象
  auto padding = _pair_int(inputs[3]);
  # 将第五个输入转换为包含两个整数的 dilation 对象
  auto dilation = _pair_int(inputs[4]);
  # 从第六个输入中获取一个 int64_t 类型的 groups 参数
  int groups = std::get<int64_t>(inputs[5]);
  
  # 检查 qw 的节点是否有 qscale 属性，如果没有抛出错误信息
  TORCH_INTERNAL_ASSERT(
      qw.node()->qscale(),
      buildErrorMessage(
          "quantized_conv2d_prepack: Expects quantized weights, qscale is missing"));
  # 检查 qw 的节点是否有 qzero 属性，如果没有抛出错误信息
  TORCH_INTERNAL_ASSERT(
      qw.node()->qzero(),
      buildErrorMessage(
          "quantized_conv2d_prepack: Expects quantized weights, qzero is missing"));
  
  # 创建一个 ExternalCall 的 StmtPtr 对象，调用 nnc_aten_quantized_conv2d_prepack 函数
  # 传递 qw、b 作为参数，以及 strides、padding、dilation、groups、qw 的 qscale、qzero、以及 immQDType(qw) 的整数值作为参数
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_conv2d_prepack",
      {qw, b},
      // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
      {strides[0],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       strides[1],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       padding[0],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       padding[1],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       dilation[0],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       dilation[1],
       groups,
       immQScale(qw),
       immQZero(qw),
       (int64_t)immQDType(qw)});
  
  # 返回一个 Tensor 对象，使用 ResultBuf.node() 作为节点，s 作为语句
  return Tensor(ResultBuf.node(), s);
# 定义计算量化卷积的函数，接收输入、输出形状、输出步长、输出类型（可选）、设备信息作为参数
Tensor computeQuantizedConv1d(
    const std::vector<ArgValue>& inputs,  // 输入参数列表，包含输入缓冲区和量化参数
    const std::vector<ExprHandle>& outputShape,  // 输出张量的形状表达式
    const std::vector<ExprHandle>& outputStrides,  // 输出张量的步长表达式
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,  // 输出张量的数据类型（可选）
    // NOLINTNEXTLINE
    at::Device device) {  // 目标设备
  const BufHandle& qx = std::get<BufHandle>(inputs[0]);  // 获取输入缓冲区 qx
  const BufHandle& prepacked = std::get<BufHandle>(inputs[1]);  // 获取预打包的缓冲区 prepacked
  const auto out_qscale = std::get<double>(inputs[2]);  // 获取输出量化比例尺度
  const auto out_qzero = std::get<int64_t>(inputs[3]);  // 获取输出量化零点
  // 根据输入缓冲区 qx 推断输出量化数据类型
  const auto out_qdtype = immQDType(qx);
  // 创建输出缓冲区 ResultBuf，采用通道为最后维度的形式
  auto ResultBuf = makeQBufHandleChannelsLast(
      "quantized_conv1d",
      outputShape,
      Dtype(out_qdtype),
      out_qscale,
      out_qzero);
  // 创建外部调用语句 s，调用名为 "nnc_aten_quantized_conv1d" 的函数
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_conv1d",
      {qx, prepacked},  // 输入参数为 qx 和 prepacked
      {immQScale(qx),    // 调用的附加参数包括输入的量化比例尺度、零点和数据类型，以及输出的比例尺度和零点
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  // 返回 Tensor 对象，其包含 ResultBuf 的节点和外部调用语句 s
  return Tensor(ResultBuf.node(), s);
}

# 定义计算量化卷积加ReLU激活的函数，参数和功能与 computeQuantizedConv1d 类似
Tensor computeQuantizedConv2dRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qx = std::get<BufHandle>(inputs[0]);  // 获取输入缓冲区 qx
  const BufHandle& prepacked = std::get<BufHandle>(inputs[1]);  // 获取预打包的缓冲区 prepacked
  const auto out_qscale = std::get<double>(inputs[2]);  // 获取输出量化比例尺度
  const auto out_qzero = std::get<int64_t>(inputs[3]);  // 获取输出量化零点
  // 根据输入缓冲区 qx 推断输出量化数据类型
  const auto out_qdtype = immQDType(qx);
  // 创建输出缓冲区 ResultBuf，采用通道为最后维度的形式
  auto ResultBuf = makeQBufHandleChannelsLast(
      "quantized_conv2d",
      outputShape,
      Dtype(out_qdtype),
      out_qscale,
      out_qzero);
  // 创建外部调用语句 s，调用名为 "nnc_aten_quantized_conv2d" 的函数
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_conv2d",
      {qx, prepacked},  // 输入参数为 qx 和 prepacked
      {immQScale(qx),    // 调用的附加参数包括输入的量化比例尺度、零点和数据类型，以及输出的比例尺度和零点
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  // 返回 Tensor 对象，其包含 ResultBuf 的节点和外部调用语句 s
  return Tensor(ResultBuf.node(), s);
}
    at::Device device) {
  // 从输入中获取量化输入缓冲区qx
  const BufHandle& qx = std::get<BufHandle>(inputs[0]);
  // 从输入中获取预打包数据的缓冲区prepacked
  const BufHandle& prepacked = std::get<BufHandle>(inputs[1]);
  // 从输入中获取输出量化参数的比例因子out_qscale
  const auto out_qscale = std::get<double>(inputs[2]);
  // 从输入中获取输出量化参数的零点out_qzero
  const auto out_qzero = std::get<int64_t>(inputs[3]);
  // 根据输入量化缓冲区qx确定输出量化数据类型out_qdtype
  const auto out_qdtype = immQDType(qx);
  // 创建名为"quantized_conv2d_relu"的量化缓冲区，采用通道末尾布局
  auto ResultBuf = makeQBufHandleChannelsLast(
      "quantized_conv2d_relu",
      outputShape,
      Dtype(out_qdtype),
      out_qscale,
      out_qzero);
  // 创建外部调用语句，调用量化卷积ReLU函数
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_conv2d_relu",
      {qx, prepacked},
      // 传递调用所需的参数：输入量化参数、输出量化参数
      {immQScale(qx),
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  // 返回输出Tensor，其底层缓冲区为ResultBuf，操作语句为s
  return Tensor(ResultBuf.node(), s);
}
}

// 计算量化线性操作的函数，返回一个 Tensor 对象
Tensor computeQuantizedLinear(
    const std::vector<ArgValue>& inputs,            // 输入参数列表
    const std::vector<ExprHandle>& outputShape,     // 输出张量的形状表达式列表
    const std::vector<ExprHandle>& outputStrides,   // 输出张量的步长表达式列表
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,    // 可选的输出类型
    // NOLINTNEXTLINE
    at::Device device) {                            // 设备参数

  const BufHandle& qx = std::get<BufHandle>(inputs[0]);     // 获取输入中的缓冲区句柄 qx
  const BufHandle& prepacked = std::get<BufHandle>(inputs[1]);  // 获取输入中的预打包数据缓冲区句柄 prepacked
  const auto out_qscale = std::get<double>(inputs[2]);    // 获取输入中的输出量化比例 out_qscale
  const auto out_qzero = std::get<int64_t>(inputs[3]);    // 获取输入中的输出量化零点 out_qzero
  // 根据输入 qx 确定输出量化数据类型
  const auto out_qdtype = immQDType(qx);

  // 创建一个连续的量化缓冲区句柄 ResultBuf
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_linear",         // 缓冲区名称
      outputShape,                // 输出形状
      Dtype(out_qdtype),          // 输出数据类型
      out_qscale,                 // 输出量化比例
      out_qzero);                 // 输出量化零点

  // 创建一个外部调用语句 s，调用量化线性函数
  StmtPtr s = ExternalCall::make(
      ResultBuf,                                  // 结果缓冲区句柄
      "nnc_aten_quantized_linear",                // 外部函数名称
      {qx, prepacked},                            // 输入参数列表
      {immQScale(qx),                             // 输入量化比例
       immQZero(qx),                              // 输入量化零点
       (int64_t)immQDType(qx),                    // 输入量化数据类型
       out_qscale,                                // 输出量化比例
       out_qzero});                               // 输出量化零点

  // 返回一个 Tensor 对象，包含结果缓冲区和外部调用语句 s
  return Tensor(ResultBuf.node(), s);
}

// 计算带 ReLU 的量化线性操作的函数，返回一个 Tensor 对象
Tensor computeQuantizedLinearRelu(
    const std::vector<ArgValue>& inputs,            // 输入参数列表
    const std::vector<ExprHandle>& outputShape,     // 输出张量的形状表达式列表
    const std::vector<ExprHandle>& outputStrides,   // 输出张量的步长表达式列表
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,    // 可选的输出类型
    // NOLINTNEXTLINE
    at::Device device) {                            // 设备参数

  const BufHandle& qx = std::get<BufHandle>(inputs[0]);     // 获取输入中的缓冲区句柄 qx
  const BufHandle& prepacked = std::get<BufHandle>(inputs[1]);  // 获取输入中的预打包数据缓冲区句柄 prepacked
  const auto out_qscale = std::get<double>(inputs[2]);    // 获取输入中的输出量化比例 out_qscale
  const auto out_qzero = std::get<int64_t>(inputs[3]);    // 获取输入中的输出量化零点 out_qzero
  // 根据输入 qx 确定输出量化数据类型
  const auto out_qdtype = immQDType(qx);

  // 创建一个连续的量化缓冲区句柄 ResultBuf
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_linear_relu",     // 缓冲区名称
      outputShape,                // 输出形状
      Dtype(out_qdtype),          // 输出数据类型
      out_qscale,                 // 输出量化比例
      out_qzero);                 // 输出量化零点

  // 创建一个外部调用语句 s，调用带 ReLU 的量化线性函数
  StmtPtr s = ExternalCall::make(
      ResultBuf,                                  // 结果缓冲区句柄
      "nnc_aten_quantized_linear_relu",           // 外部函数名称
      {qx, prepacked},                            // 输入参数列表
      {immQScale(qx),                             // 输入量化比例
       immQZero(qx),                              // 输入量化零点
       (int64_t)immQDType(qx),                    // 输入量化数据类型
       out_qscale,                                // 输出量化比例
       out_qzero});                               // 输出量化零点

  // 返回一个 Tensor 对象，包含结果缓冲区和外部调用语句 s
  return Tensor(ResultBuf.node(), s);
}

// 计算带量化加法的外部调用的函数，返回一个 Tensor 对象
Tensor computeQuantizedAddExternalCall(
    const std::vector<ArgValue>& inputs,            // 输入参数列表
    const std::vector<ExprHandle>& outputShape,     // 输出张量的形状表达式列表
    const std::vector<ExprHandle>& outputStrides,   // 输出张量的步长表达式列表
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,    // 可选的输出类型
    // NOLINTNEXTLINE
    at::Device device) {                            // 设备参数
    // 传入设备参数 device
    at::Device device) {
      // 获取输入的第一个缓冲区对象 qa
      const BufHandle& qa = std::get<BufHandle>(inputs[0]);
      // 获取输入的第二个缓冲区对象 qb
      const BufHandle& qb = std::get<BufHandle>(inputs[1]);
      // 获取输出的量化比例尺度 out_qscale
      const auto out_qscale = std::get<double>(inputs[2]);
      // 获取输出的量化零点 out_qzero
      const auto out_qzero = std::get<int64_t>(inputs[3]);
      // 根据输入缓冲区对象 qa 推断输出的量化数据类型 out_qdtype
      const auto out_qdtype = immQDType(qa);
      // 检查输入缓冲区对象 qa 是否是以通道为最后维度存储的
      const bool isQAChannelsLast = isChannelsLast(qa);
      // 检查输入缓冲区对象 qb 是否是以通道为最后维度存储的
      const bool isQBChannelsLast = isChannelsLast(qb);
      // 根据输入缓冲区对象的存储方式，选择创建输出缓冲区的方式：
      // 如果任意一个输入缓冲区以通道为最后维度存储，则创建通道为最后维度的缓冲区
      auto ResultBuf = (isQAChannelsLast || isQBChannelsLast)
          ? makeQBufHandleChannelsLast(
                "quantized_add",
                outputShape,
                Dtype(out_qdtype),
                out_qscale,
                out_qzero)
          // 否则创建连续存储的缓冲区
          : makeQBufHandleContiguous(
                "quantized_add",
                outputShape,
                Dtype(out_qdtype),
                out_qscale,
                out_qzero);
      // 创建外部调用语句，调用 quantized_add 函数
      StmtPtr s = ExternalCall::make(
          ResultBuf,
          "nnc_aten_quantized_add",
          {qa, qb},  // 输入参数为 qa 和 qb
          {immQScale(qa),  // qa 的量化比例尺度
           immQZero(qa),   // qa 的量化零点
           (int64_t)immQDType(qa),  // qa 的量化数据类型
           immQScale(qb),  // qb 的量化比例尺度
           immQZero(qb),   // qb 的量化零点
           (int64_t)immQDType(qb),  // qb 的量化数据类型
           out_qscale,     // 输出的量化比例尺度
           out_qzero});    // 输出的量化零点
      // 返回一个 Tensor 对象，该对象使用 ResultBuf 作为数据节点，s 作为外部调用语句
      return Tensor(ResultBuf.node(), s);
    }
}

Tensor computeQuantizedMul(
    const std::vector<ArgValue>& inputs,  // 输入参数：包含输入数据和标量
    const std::vector<ExprHandle>& outputShape,  // 输出形状的表达式列表
    const std::vector<ExprHandle>& outputStrides,  // 输出步长的表达式列表
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,  // 可选的输出类型，当前未实现类型传播
    // NOLINTNEXTLINE
    at::Device device) {  // 设备参数

  const BufHandle& qa = std::get<BufHandle>(inputs[0]);  // 获取输入缓冲区句柄 qa
  const BufHandle& qb = std::get<BufHandle>(inputs[1]);  // 获取输入缓冲区句柄 qb
  const auto out_qscale = std::get<double>(inputs[2]);  // 获取输出量化比例尺度
  const auto out_qzero = std::get<int64_t>(inputs[3]);  // 获取输出量化零点
  // 根据输入缓冲区 qa 推断输出量化数据类型
  const auto out_qdtype = immQDType(qa);
  // 创建连续的量化缓冲区句柄 ResultBuf
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_mul", outputShape, Dtype(out_qdtype), out_qscale, out_qzero);
  // 构造外部调用语句 s，调用量化乘法函数
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_mul",
      {qa, qb},
      {immQScale(qa),
       immQZero(qa),
       (int64_t)immQDType(qa),
       immQScale(qb),
       immQZero(qb),
       (int64_t)immQDType(qb),
       out_qscale,
       out_qzero});
  // 返回结果张量
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedMulScalar(
    const std::vector<ArgValue>& inputs,  // 输入参数：包含输入数据和标量
    const std::vector<ExprHandle>& outputShape,  // 输出形状的表达式列表
    const std::vector<ExprHandle>& outputStrides,  // 输出步长的表达式列表
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,  // 可选的输出类型，当前未实现类型传播
    // NOLINTNEXTLINE
    at::Device device) {  // 设备参数

  const BufHandle& qa = std::get<BufHandle>(inputs[0]);  // 获取输入缓冲区句柄 qa
  const auto scalar = std::get<double>(inputs[1]);  // 获取标量值
  // 根据输入缓冲区 qa 推断输出量化数据类型
  const auto out_qdtype = immQDType(qa);
  double scale1 = immQScale(qa);
  // 创建连续的量化缓冲区句柄 ResultBuf
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_mul_scalar",
      outputShape,
      Dtype(out_qdtype),
      scale1 * scalar,
      immQZero(qa));
  // 构造外部调用语句 s，调用量化乘法标量函数
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_mul_scalar",
      {qa},
      {scale1, immQZero(qa), (int64_t)immQDType(qa), scalar});
  // 返回结果张量
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedRelu(
    const std::vector<ArgValue>& inputs,  // 输入参数：包含输入数据和标量
    const std::vector<ExprHandle>& outputShape,  // 输出形状的表达式列表
    const std::vector<ExprHandle>& outputStrides,  // 输出步长的表达式列表
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,  // 可选的输出类型，当前未实现类型传播
    // NOLINTNEXTLINE
    at::Device device) {  // 设备参数
    at::Device device) {
  // 获取第一个输入的缓冲区句柄
  const BufHandle& qa = std::get<BufHandle>(inputs[0]);
  // 确定输出的量化数据类型
  const auto out_qdtype = immQDType(qa);
  // 检查缓冲区是否按通道存储
  const bool isQAChannelsLast = isChannelsLast(qa);
  // 根据是否按通道存储选择创建量化缓冲区句柄的方式
  auto ResultBuf = isQAChannelsLast ? makeQBufHandleChannelsLast(
                                          "quantized_relu",
                                          outputShape,
                                          Dtype(out_qdtype),
                                          immQScale(qa),
                                          immQZero(qa))
                                    : makeQBufHandleContiguous(
                                          "quantized_relu",
                                          outputShape,
                                          Dtype(out_qdtype),
                                          immQScale(qa),
                                          immQZero(qa));
  // 创建外部调用语句，调用量化 ReLU 操作
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_relu",
      {qa},  // 传入量化缓冲区句柄作为参数
      {immQScale(qa), immQZero(qa), (int64_t)immQDType(qa)});  // 传入量化参数
  // 返回一个 Tensor 对象，使用 ResultBuf 和生成的外部调用语句 s
  return Tensor(ResultBuf.node(), s);
}
}

Tensor computeQuantizedCat(
    const std::vector<ArgValue>& inputs,                     // 接收输入参数列表
    const std::vector<ExprHandle>& outputShape,              // 输出张量的形状表达式列表
    const std::vector<ExprHandle>& outputStrides,            // 输出张量的步长表达式列表
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,             // 可选的输出张量数据类型
    // NOLINTNEXTLINE
    at::Device device) {                                     // 设备类型参数
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto inputList = std::get<BufList>(inputs[0]);             // 获取输入参数的缓冲区列表
  auto argDim = std::get<int64_t>(inputs[1]);                // 获取输入参数的整数维度
  auto n = inputList.size();                                 // 获取输入列表的大小（数量）
  // TODO: handle optional out_qscale, out_qzero
  const auto out_qscale = std::get<double>(inputs[2]);       // 获取输出量化比例
  const auto out_qzero = std::get<int64_t>(inputs[3]);       // 获取输出量化零点

  std::vector<BufHandle> args;                               // 创建缓冲区句柄列表
  std::vector<ExprHandle> extra_args;                        // 创建额外的表达式句柄列表
  for (const auto i : c10::irange(n)) {                      // 对于每个输入列表中的元素
    const BufHandle& bh = inputList[i];                      // 获取当前缓冲区句柄
    args.emplace_back(bh);                                   // 将当前缓冲区句柄添加到参数列表中
    extra_args.emplace_back(immQScale(bh));                  // 添加当前缓冲区的量化比例
    extra_args.emplace_back(immQZero(bh));                   // 添加当前缓冲区的量化零点
    extra_args.emplace_back((int64_t)immQDType(bh));         // 添加当前缓冲区的量化数据类型（整数表示）
  }
  extra_args.emplace_back(argDim);                           // 添加参数维度
  extra_args.emplace_back(out_qscale);                       // 添加输出量化比例
  extra_args.emplace_back(out_qzero);                        // 添加输出量化零点
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_cat",                                       // 创建一个连续的量化缓冲区句柄
      outputShape,                                            // 使用输出形状
      Dtype(immQDType(inputList[0])),                         // 使用第一个输入的量化数据类型
      out_qscale,                                             // 使用输出的量化比例
      out_qzero);                                             // 使用输出的量化零点
  StmtPtr s =
      ExternalCall::make(ResultBuf, "nnc_aten_quantized_cat",  // 创建外部调用语句
                         args, extra_args);                   // 使用参数和额外参数
  return Tensor(ResultBuf.node(), s);                         // 返回张量对象
}

Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,                      // 接收输入参数列表
    const std::vector<ExprHandle>& outputShape,               // 输出张量的形状表达式列表
    const std::vector<ExprHandle>& outputStrides,             // 输出张量的步长表达式列表
    const std::optional<ScalarType>& outputType,              // 可选的输出张量数据类型
    at::Device) {                                             // 设备类型参数
  Dtype dtype = kFloat;                                       // 默认数据类型为浮点数
  if (outputType) {                                           // 如果输出类型已定义
    dtype = Dtype(*outputType);                               // 使用指定的输出数据类型
  }
  auto qx = std::get<BufHandle>(inputs[0]);                   // 获取输入的缓冲区句柄
  TORCH_INTERNAL_ASSERT(
      qx.node()->qscale(),                                   // 确保输入缓冲区具有量化比例
      buildErrorMessage("Missing quantized scale for dequantize"));  // 如果没有量化比例，则报错
  TORCH_INTERNAL_ASSERT(
      qx.node()->qzero(),                                    // 确保输入缓冲区具有量化零点
      buildErrorMessage("Missing quantized zero point for dequantize"));  // 如果没有量化零点，则报错
  auto qscale = ExprHandle(qx.node()->qscale());              // 获取量化比例的表达式句柄
  auto qzero = ExprHandle(qx.node()->qzero());                // 获取量化零点的表达式句柄
  std::vector<VarPtr> vars;                                  // 创建变量指针列表
  std::vector<ExprHandle> indices;                           // 创建表达式句柄列表
  for (const auto& os : outputShape) {                       // 对于每个输出形状
    auto var = alloc<Var>("", os.node()->dtype());           // 分配一个新变量
    vars.push_back(var);                                     // 将变量添加到变量列表中
    indices.push_back(VarHandle(var));                       // 将变量的句柄添加到索引列表中
  }
  auto y = dequant(tensorOrConstant(inputs[0], indices),      // 执行反量化操作
                   dtype, qscale, qzero);
  BufPtr buf = alloc<Buf>(                                   // 分配一个新的缓冲区
      "dequantize", ExprHandleVectorToExprVector(outputShape),// 使用输出形状表达式列表
      dtype);                                                // 使用指定的数据类型
  return Tensor(buf, vars, y.node());                        // 返回张量对象
}

Tensor computeUpsampleNearest2d(
    const std::vector<ArgValue>& inputs,                      // 接收输入参数列表
    const std::vector<ExprHandle>& outputShape,               // 输出张量的形状表达式列表
    const std::vector<ExprHandle>& outputStrides,             // 输出张量的步长表达式列表
    const std::optional<ScalarType>& outputType,              // 可选的输出张量数据类型
    at::Device) {
  // 从输入中获取第一个缓冲区处理器对象
  auto A = std::get<BufHandle>(inputs[0]);
  // 获取输出张量的高度和宽度
  const auto& output_height = outputShape[2];
  const auto& output_width = outputShape[3];
  // 获取输入张量的高度和宽度
  auto input_height = ExprHandle(A.dim(2));
  auto input_width = ExprHandle(A.dim(3));

  // 创建一个包含索引变量的向量，用于迭代输出形状
  std::vector<VarHandle> args = create_index_vars(outputShape);
  // 当比例已指定时，单独处理？例如在 'scalar_t compute_scales_value' 中的 UpSample.h 中
  // 计算高度和宽度的缩放比例
  auto scale_h =
      promoteToDtype(input_height, ScalarType::Double) / output_height;
  auto scale_w = promoteToDtype(input_width, ScalarType::Double) / output_width;
  // TODO: 如果在索引计算中重复使用 if 语句，是否应将其移出循环？
  // 定义计算最近索引的 Lambda 函数
  auto compute_nearest_idx = [](ExprHandle scale,
                                const ExprHandle& dst_index,
                                const ExprHandle& input_size) {
    return Min::make(
        promoteToDtype(floor(dst_index * scale), ScalarType::Long),
        input_size - 1,
        true);
  };
  // 定义循环体函数，根据缩放比例和输入大小更新轴
  auto body_func = [&](std::vector<VarHandle> axes) {
    std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
    newAxes[2] = compute_nearest_idx(scale_h, axes[2], input_height);
    newAxes[3] = compute_nearest_idx(scale_w, axes[3], input_width);
    return A.load(newAxes);
  };
  // 执行循环体函数生成表达式
  auto e = body_func(args);
  // 根据输入是否通道为最后一维，选择生成相应步长的函数
  auto strides = isChannelsLast(A) ? make_channels_last_strides(outputShape)
                                   : make_contiguous_strides(outputShape);
  // 创建新的缓冲区对象，包括名称、输出形状、数据类型和其他参数
  BufHandle buf = Buf::make(
      "upsample_nearest2d",
      outputShape,
      Dtype(*outputType),
      c10::nullopt, // 初始化器
      fmap(strides, [&](ExprPtr stride) { return ExprHandle(stride); }),
      ExprHandle(A.node()->qscale()), // 量化缩放因子
      ExprHandle(A.node()->qzero())); // 量化零点

  // 返回一个新的张量对象，包含新的缓冲区、参数和表达式
  return Tensor(buf, args, e);
}
// 结束函数体

Tensor computeUpsampleNearest2dExternalCall(
    const std::vector<ArgValue>& inputs,
    // 输入参数列表
    const std::vector<ExprHandle>& outputShape,
    // 输出形状的表达式列表
    const std::vector<ExprHandle>& outputStrides,
    // 输出步长的表达式列表
    const std::optional<ScalarType>& outputType,
    // 可选的输出类型
    at::Device) {
  // 定义变量dtype，默认为kFloat
  Dtype dtype = kFloat;
  // 如果提供了输出类型，则使用该类型
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  // 初始化输出尺寸的高度和宽度为-1
  int64_t output_size_h = -1;
  int64_t output_size_w = -1;
  // 如果输入参数中包含输出尺寸信息，则提取出来
  if (auto output_sizes = std::get_if<IntList>(&inputs[1])) {
    output_size_h = (*output_sizes)[0];
    output_size_w = (*output_sizes)[1];
  }

  // 初始化高度和宽度的缩放因子为-1.0
  double scale_factor_h = -1.f;
  double scale_factor_w = -1.f;
  // 如果输入参数中包含缩放因子信息，则提取出来
  if (auto scale_factors = std::get_if<DoubleList>(&inputs[2])) {
    scale_factor_h = (*scale_factors)[0];
    scale_factor_w = (*scale_factors)[1];
  }
  // 提取输入参数中的缓冲区句柄x
  const BufHandle& x = std::get<BufHandle>(inputs[0]);
  // 初始化量化缩放因子、零点和数据类型为-1
  double qx_qscale = -1.f;
  int64_t qx_qzero = -1l;
  int64_t qx_qdtype = -1l;
  // 如果输入缓冲区是量化的，则获取其量化参数
  if (isQuantized(x)) {
    qx_qscale = immQScale(x);
    qx_qzero = immQZero(x);
    qx_qdtype = (int64_t)immQDType(x);
  }

  // 定义结果缓冲区句柄ResultBuf
  BufHandle ResultBuf = [&]() {
    // 如果输入缓冲区是量化的，则创建一个量化缓冲区句柄
    if (isQuantized(x)) {
      return makeQBufHandleChannelsLast(
          "upsample_nearest2d",
          outputShape,
          Dtype(immQDType(x)),
          qx_qscale,
          qx_qzero);
    }
    // 否则，创建一个普通缓冲区句柄
    return BufHandle("upsample_nearest2d", outputShape, dtype);
  }();

  // 创建外部调用语句s，用于执行最近邻插值上采样操作
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_upsample_nearest2d",
      {x},
      {qx_qscale,
       qx_qzero,
       qx_qdtype,
       output_size_h,
       output_size_w,
       scale_factor_h,
       scale_factor_w});
  // 返回一个Tensor对象，包含结果缓冲区和外部调用语句
  return Tensor(ResultBuf.node(), s);
}

// 下面是另一个函数computeQuantizedSigmoidExternalCall的定义，由于注释长度限制，未完全展示
Tensor computeQuantizedSigmoidExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const std::optional<ScalarType>& outputType,
    // 继续提供computeQuantizedSigmoidExternalCall函数的注释
  // 获取第一个输入的缓冲区句柄 qx
  const BufHandle& qx = std::get<BufHandle>(inputs[0]);

  // 确定输出的量化数据类型
  const auto out_qdtype = immQDType(qx);
  
  // 设置输出的量化比例尺
  const double out_qscale = 1.0f / 256.0f;
  
  // 根据输出的量化数据类型选择输出的量化零点
  const int64_t out_qzero = (out_qdtype == ScalarType::QInt8) ? -128 : 0;

  // 根据输入数据是否是按通道最后一维排列，创建不同的量化缓冲区句柄
  auto ResultBuf = isChannelsLast(qx) ? makeQBufHandleChannelsLast(
                                            "quantized_sigmoid",
                                            outputShape,
                                            Dtype(out_qdtype),
                                            out_qscale,
                                            out_qzero)
                                      : makeQBufHandleContiguous(
                                            "quantized_sigmoid",
                                            outputShape,
                                            Dtype(out_qdtype),
                                            out_qscale,
                                            out_qzero);
  
  // 创建外部调用语句，调用量化 sigmoid 函数
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_sigmoid",
      {qx},
      {immQScale(qx),
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  
  // 返回一个 Tensor 对象，其底层使用 ResultBuf 节点和外部调用语句 s
  return Tensor(ResultBuf.node(), s);
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
```