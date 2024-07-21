# `.\pytorch\torch\csrc\jit\tensorexpr\lowerings.cpp`

```py
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

#include <ATen/native/Activation.h>
#include <ATen/native/mkldnn/Common.h>

// 定义命名空间 torch::jit::tensorexpr 下的函数 getNNCLoweringRegistry()
FunctionSchemaMap<NNCLoweringFunction>& getNNCLoweringRegistry() {
  // 静态局部变量，用于保存 NNCLoweringFunction 的注册表
  static FunctionSchemaMap<NNCLoweringFunction> lowering_registry_;
  return lowering_registry_;
}

// RegisterNNCLoweringsFunction 类的构造函数实现
RegisterNNCLoweringsFunction::RegisterNNCLoweringsFunction(
    const std::vector<std::string>& schemas,
    NNCLoweringFunction fn) {
  // 遍历传入的 schemas 列表
  for (const auto& schema_str : schemas) {
    // 调用 parseSchema 函数解析 schema_str，并将解析结果与 fn 插入到 lowering_registry_ 中
    getNNCLoweringRegistry().insert(parseSchema(schema_str), fn);
  }
}

// 匿名命名空间，定义 nnc_lowerings_lazy_registration 函数
// NOLINTNEXTLINE 是一个代码静态分析工具（如 clang-tidy）的注释，用于跳过下一行的警告
int nnc_lowerings_lazy_registration() {
  // 注册 aten::dropout 操作的 NNCLoweringsFunction computeNoop
  RegisterNNCLoweringsFunction aten_dropout(
      {"aten::dropout(Tensor input, float p, bool train) -> (Tensor)"},
      computeNoop);
  // 注册 aten::contiguous 操作的 NNCLoweringsFunction computeNoop
  RegisterNNCLoweringsFunction aten_contiguous(
      {"aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> (Tensor(a))"},
      computeNoop);

#ifdef USE_XNNPACK
  // 如果定义了 USE_XNNPACK 宏，则注册 prepacked::conv2d_clamp_run 操作的 computePrepackedConv2dClampRun
  // TODO: add a test 为待添加的测试提供了提示
  RegisterNNCLoweringsFunction prepacked_conv2d_clamp_run(
      {"prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> (Tensor Y)"},
      computePrepackedConv2dClampRun);

  // TODO: add a test 为待添加的测试提供了提示
  RegisterNNCLoweringsFunction prepacked_linear_clamp_run(
      {"prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> (Tensor Y)"},
      computePrepackedLinearClampRun);
#endif

#if AT_MKLDNN_ENABLED()
  // 如果定义了 AT_MKLDNN_ENABLED 宏，则注册 mkldnn_prepacked::conv2d_run 操作的 computeMkldnnPrepackedConvRun
  RegisterNNCLoweringsFunction mkldnn_prepacked_conv2d_run(
      {"mkldnn_prepacked::conv2d_run(Tensor X, __torch__.torch.classes.mkldnn.ConvOpContext W_prepack) -> (Tensor Y)"},
      computeMkldnnPrepackedConvRun);
#endif // AT_MKLDNN_ENABLED()

RegisterNNCLoweringsFunction aten_sub(
    // 注册aten::sub.Scalar和aten::sub.Tensor操作的处理函数
    {"aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)",
     "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)"},
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const std::vector<ExprHandle>& outputStrides,
       const std::optional<ScalarType>& outputType,
       at::Device device) {
      auto sub_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        // 注意：布尔类型不支持sub操作，无需转换为整数类型
        return lhs - rhs;
      };
      // 输入参数数量必须为2或3，否则抛出错误
      TORCH_INTERNAL_ASSERT(
          inputs.size() == 2 || inputs.size() == 3,
          buildErrorMessage("Invalid number of input operands"));
      // 根据输入参数数量调用对应的计算函数，带有或不带有alpha参数
      return (inputs.size() > 2) ? computeTwoOperandWithAlpha(
                                       "aten_sub",
                                       inputs,
                                       outputShape,
                                       outputStrides,
                                       outputType,
                                       sub_lambda)
                                 : computeTwoOperand(
                                       "aten_sub",
                                       inputs,
                                       outputShape,
                                       outputStrides,
                                       outputType,
                                       sub_lambda);
    });

RegisterNNCLoweringsFunction aten_mul(
    // 注册aten::mul.Scalar和aten::mul.Tensor操作的处理函数
    {"aten::mul.Scalar(Tensor self, Scalar other) -> (Tensor)",
     "aten::mul.Tensor(Tensor self, Tensor other) -> (Tensor)"},
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const std::vector<ExprHandle>& outputStrides,
       const std::optional<ScalarType>& outputType,
       at::Device device) {
      // 执行两个操作数的乘法运算，将布尔类型转换为整数类型
      return computeTwoOperand(
          "aten_mul",
          inputs,
          outputShape,
          outputStrides,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) * boolToInteger(rhs);
          });
    });
#define DEFINE_BINARY_SCALAR_OP_LOWERING(op_name, op)                     \
  // 定义一个宏，用于注册标量操作降低函数，处理多种输入类型
  RegisterNNCLoweringsFunction aten_##op_name##_scalar(                   \
      {"aten::" #op_name ".int(int a, int b) -> (int)",                   \
       "aten::" #op_name ".int_float(int a, float b) -> (float)",         \
       "aten::" #op_name ".float_int(float a, int b) -> (float)",         \
       "aten::" #op_name ".float(float a, float b) -> (float)"},          \
      [](const std::vector<ArgValue>& inputs,                             \
         const std::vector<ExprHandle>& outputShape,                      \
         const std::vector<ExprHandle>& outputStrides,                    \
         const std::optional<ScalarType>& outputType,                     \
         at::Device device) {                                             \
        // 调用computeScalar函数来计算标量操作
        return computeScalar(                                             \
            "aten_#op_name",                                              \
            inputs,                                                       \
            outputShape,                                                  \
            outputStrides,                                                \
            outputType,                                                   \
            [](const ExprHandle& a, const ExprHandle& b) { return op; }); \
      });

// 定义标量乘法操作降低函数
DEFINE_BINARY_SCALAR_OP_LOWERING(mul, a * b)
// 定义标量加法操作降低函数
DEFINE_BINARY_SCALAR_OP_LOWERING(add, a + b)
// 定义标量减法操作降低函数
DEFINE_BINARY_SCALAR_OP_LOWERING(sub, a - b)
#undef DEFINE_BINARY_SCALAR_OP_LOWERING

// 注册标量除法操作降低函数
RegisterNNCLoweringsFunction aten_div_scalar(
    {"aten::div(Scalar a, Scalar b) -> (float)",
     "aten::div.int(int a, int b) -> (float)",
     "aten::div.int_float(int a, float b) -> (float)",
     "aten::div.float_int(float a, int b) -> (float)",
     "aten::div.float(float a, float b) -> (float)"},
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const std::vector<ExprHandle>& outputStrides,
       const std::optional<ScalarType>& outputType,
       at::Device device) {
      // 使用computeScalar函数计算除法操作，确保类型推广为默认类型
      return computeScalar(
          "aten_div",
          inputs,
          outputShape,
          outputStrides,
          outputType,
          [](const ExprHandle& a, const ExprHandle& b) {
            return promoteIntegerToDefaultType(a) /
                promoteIntegerToDefaultType(b);
          });
    });
// 定义宏 `DEFINE_COMPARISON_SCALAR_OP_LOWERING`，用于注册标量操作的降级函数
#define DEFINE_COMPARISON_SCALAR_OP_LOWERING(op_name, op)                 \
  // 注册具体操作 `aten_##op_name##_scalar` 的降级函数
  RegisterNNCLoweringsFunction aten_##op_name##_scalar(                   \
      // 定义接受的输入参数签名，以及返回值的类型签名
      {"aten::" #op_name ".bool(bool a, bool b) -> (bool)",               \
       "aten::" #op_name ".int(int a, int b) -> (bool)",                  \
       "aten::" #op_name ".int_float(int a, float b) -> (bool)",          \
       "aten::" #op_name ".float_int(float a, int b) -> (bool)",          \
       "aten::" #op_name ".float(float a, float b) -> (bool)"},           \
      // 定义 Lambda 函数，处理输入、输出形状、步幅、输出类型和设备
      [](const std::vector<ArgValue>& inputs,                             \
         const std::vector<ExprHandle>& outputShape,                      \
         const std::vector<ExprHandle>& outputStrides,                    \
         const std::optional<ScalarType>& outputType,                     \
         at::Device device) {                                             \
        // 调用 `computeScalar` 函数，执行具体的标量操作
        return computeScalar(                                             \
            // 操作的名称，例如 "aten_lt_scalar"
            "aten_#op_name",                                              \
            // 输入参数列表
            inputs,                                                       \
            // 输出形状
            outputShape,                                                  \
            // 输出步幅
            outputStrides,                                                \
            // 输出数据类型
            outputType,                                                   \
            // Lambda 表达式，实现具体的操作
            [](const ExprHandle& a, const ExprHandle& b) { return op; }); \
      });

// 使用宏 `DEFINE_COMPARISON_SCALAR_OP_LOWERING` 分别定义 `<`, `<=`, `==`, `!=`, `>`, `>=` 操作的降级函数
DEFINE_COMPARISON_SCALAR_OP_LOWERING(lt, cast<bool>(a < b))
DEFINE_COMPARISON_SCALAR_OP_LOWERING(le, cast<bool>(a <= b))
DEFINE_COMPARISON_SCALAR_OP_LOWERING(eq, cast<bool>(a == b))
DEFINE_COMPARISON_SCALAR_OP_LOWERING(ne, cast<bool>(a != b))
DEFINE_COMPARISON_SCALAR_OP_LOWERING(gt, cast<bool>(a > b))
DEFINE_COMPARISON_SCALAR_OP_LOWERING(ge, cast<bool>(a >= b))

// 取消宏 `DEFINE_COMPARISON_SCALAR_OP_LOWERING` 的定义
#undef DEFINE_COMPARISON_SCALAR_OP_LOWERING
`
#define DEFINE_BITWISE_SCALAR_OP_LOWERING(op_name, op)                    \
  #define 注册一个名为 aten_##op_name##_int_scalar 的函数到 NNCLoweringsFunction  \
  RegisterNNCLoweringsFunction aten_##op_name##_int_scalar(               \
      {"aten::" #op_name ".int(int a, int b) -> (int)"},                  \
      [](const std::vector<ArgValue>& inputs,                             \
         const std::vector<ExprHandle>& outputShape,                      \
         const std::vector<ExprHandle>& outputStrides,                    \
         const std::optional<ScalarType>& outputType,                     \
         at::Device device) {                                             \
        return computeScalar(                                             \
            "aten_#op_name",                                              \
            inputs,                                                       \
            outputShape,                                                  \
            outputStrides,                                                \
            outputType,                                                   \
            [](const ExprHandle& a, const ExprHandle& b) { return op; }); \
      });                                                                \
  #undef 注册一个名为 aten_##op_name##_int_scalar 的函数到 NNCLoweringsFunction  \
  DEFINE_BITWISE_SCALAR_OP_LOWERING(                                       \
      __and__, boolToInteger(a) & boolToInteger(b))                        \
  DEFINE_BITWISE_SCALAR_OP_LOWERING(__or__, boolToInteger(a) | boolToInteger(b)) \
  DEFINE_BITWISE_SCALAR_OP_LOWERING(                                       \
      __xor__, boolToInteger(a) ^ boolToInteger(b))                        \
  DEFINE_BITWISE_SCALAR_OP_LOWERING(__lshift__, a << b)                     \
  DEFINE_BITWISE_SCALAR_OP_LOWERING(__rshift__, a >> b)                     \
#undef DEFINE_BITWISE_SCALAR_OP_LOWERING

#define DEFINE_LOGICAL_SCALAR_OP_LOWERING(op_name, op)                    \
  #define 注册一个名为 aten_##op_name##_bool_scalar 的函数到 NNCLoweringsFunction  \
  RegisterNNCLoweringsFunction aten_##op_name##_bool_scalar(              \
      {"aten::" #op_name ".bool(bool a, bool b) -> (bool)"},              \
      [](const std::vector<ArgValue>& inputs,                             \
         const std::vector<ExprHandle>& outputShape,                      \
         const std::vector<ExprHandle>& outputStrides,                    \
         const std::optional<ScalarType>& outputType,                     \
         at::Device device) {                                             \
        return computeScalar(                                             \
            "aten_#op_name",                                              \
            inputs,                                                       \
            outputShape,                                                  \
            outputStrides,                                                \
            outputType,                                                   \
            [](const ExprHandle& a, const ExprHandle& b) { return op; }); \
      });                                                                \
  #undef 注册一个名为 aten_##op_name##_bool_scalar 的函数到 NNCLoweringsFunction  \
  DEFINE_LOGICAL_SCALAR_OP_LOWERING(__and__, a && b)                        \
  DEFINE_LOGICAL_SCALAR_OP_LOWERING(__or__, a || b)                        \
  DEFINE_LOGICAL_SCALAR_OP_LOWERING(__xor__, a != b)                       \
#define NNC_QUANTIZATION_EXPR_QUANT 1
#define NNC_QUANTIZATION_EXPR_DEQUANT 1

# 定义一个宏，表示开启了 NNCompiler 的量化表达式 dequant 的选项

RegisterNNCLoweringsFunction aten_quantize_per_tensor(
    {"aten::quantize_per_tensor(Tensor self, float scale, int zero_point, int dtype) -> (Tensor)",
     "aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int dtype) -> (Tensor)",
     "aten::quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, int dtype) -> (Tensor[])"},
#if NNC_QUANTIZATION_EXPR_QUANT == 1
    computeQuantizePerTensor
#else
    computeQuantizePerTensorExternalCall
#endif
);

# 注册 NNCompiler 降低函数，支持对张量进行量化操作，包括不同的量化表达式选项

RegisterNNCLoweringsFunction aten_dequantize(
    {"aten::dequantize.self(Tensor self) -> (Tensor)"},
#if NNC_QUANTIZATION_EXPR_DEQUANT == 1
    computeDequantize
#else
    computeDequantizeExternalCall
#endif
);

# 注册 NNCompiler 降低函数，支持对张量进行反量化操作，根据是否开启 NNCompiler 的 dequant 选项来选择相应的函数
#endif
);
// 注册 quantized_conv1d 函数，接受 quantized::conv1d 的调用，使用 computeQuantizedConv1d 函数实现
RegisterNNCLoweringsFunction quantized_conv1d(
    {"quantized::conv1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)"},
    computeQuantizedConv1d);

// 注册 quantized_conv2d 函数，接受 quantized::conv2d.new 的调用，使用 computeQuantizedConv2d 函数实现
RegisterNNCLoweringsFunction quantized_conv2d(
    {"quantized::conv2d.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)"},
    computeQuantizedConv2d);

// 注册 quantized_conv2d_relu 函数，接受 quantized::conv2d_relu.new 的调用，使用 computeQuantizedConv2dRelu 函数实现
RegisterNNCLoweringsFunction quantized_conv2d_relu(
    {"quantized::conv2d_relu.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)"},
    computeQuantizedConv2dRelu);

// 注册 quantized_linear 函数，接受 quantized::linear 的调用，使用 computeQuantizedLinear 函数实现
RegisterNNCLoweringsFunction quantized_linear(
    {"quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> (Tensor Y)"},
    computeQuantizedLinear);

// 注册 quantized_linear_relu 函数，接受 quantized::linear_relu 的调用，使用 computeQuantizedLinear 函数实现
RegisterNNCLoweringsFunction quantized_linear_relu(
    {"quantized::linear_relu(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> (Tensor Y)"},
    computeQuantizedLinear);

// 注册 quantized_add 函数，接受 quantized::add 的调用，使用 computeQuantizedAdd 函数实现
RegisterNNCLoweringsFunction quantized_add(
    {"quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> (Tensor qc)"},
    computeQuantizedAdd);

// 注册 quantized_mul 函数，接受 quantized::mul 的调用，使用 computeQuantizedMul 函数实现
RegisterNNCLoweringsFunction quantized_mul(
    {"quantized::mul(Tensor qa, Tensor qb, float scale, int zero_point) -> (Tensor qc)"},
    computeQuantizedMul);

// 注册 quantized_mul_scalar 函数，接受 quantized::mul.Scalar 的调用，使用 computeQuantizedMulScalar 函数实现
RegisterNNCLoweringsFunction quantized_mul_scalar(
    {"quantized::mul.Scalar(Tensor qa, Scalar b) -> (Tensor qc)"},
    computeQuantizedMulScalar);

// 注册 quantized_conv2d_prepack 函数，接受 quantized::conv2d_prepack 的调用，使用 computeQuantizedConv2dPrepack 函数实现
RegisterNNCLoweringsFunction quantized_conv2d_prepack(
    {"quantized::conv2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> (__torch__.torch.classes.quantized.Conv2dPackedParamsBase)"},
    computeQuantizedConv2dPrepack);

// 注册 quantized_cat 函数，接受 quantized::cat 的调用，使用 computeQuantizedCat 函数实现
RegisterNNCLoweringsFunction quantized_cat(
    {"quantized::cat(Tensor[] qx, int dim, float? scale, int? zero_point) -> (Tensor)"},
    computeQuantizedCat);

// 注册 aten_upsample_nearest2d 函数，接受 aten::upsample_nearest2d.vec 的调用，使用 computeUpsampleNearest2dExternalCall 函数实现
RegisterNNCLoweringsFunction aten_upsample_nearest2d(
    {"aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)"},
    computeUpsampleNearest2dExternalCall);

// 返回整数值 0，标志函数执行完毕
return 0;
}
} // namespace
// namespace torch::jit::tensorexpr 结束
```