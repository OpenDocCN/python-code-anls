# `.\pytorch\aten\src\ATen\native\quantized\cpu\qmul.cpp`

```py
// 定义宏，用于声明只使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入必要的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <torch/library.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/quantized/Quantizer.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含特定的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/empty_like.h>
#endif

#include <algorithm>

// 定义命名空间 at 和 native
namespace at {
namespace native {

// 定义分发 qmul_relu_stub 和 qmul_stub
DEFINE_DISPATCH(qmul_relu_stub);
DEFINE_DISPATCH(qmul_stub);

// 定义匿名命名空间，内部函数或静态变量只在当前文件可见
namespace {

// 内联函数，检查输入张量的量化方案和数据类型是否符合要求
inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine,
              "Only per tensor quantization is supported in Mul.");
  TORCH_CHECK(qa.scalar_type() == qb.scalar_type(),
              "Mul operands should have same data type.");
  TORCH_CHECK(qa.qscheme() == qb.qscheme(),
              "Both inputs to Mul must have the same quantization scheme.");
}

// 注意：假设 out 和 self、other 的大小相同
// 注意：仅当 self、other、out 具有相同的数据类型时才支持乘法
// 模板函数，实现乘法操作并返回结果张量 out
template <bool ReLUFused = false>
Tensor _mul_out(Tensor& out, const Tensor& self, const Tensor& other) {
  if (ReLUFused) {
    // 调用 qmul_relu_stub 分发函数执行乘法和 ReLU 激活
    qmul_relu_stub(self.device().type(), out, self, other);
  } else {
    // 调用 qmul_stub 分发函数执行乘法
    qmul_stub(self.device().type(), out, self, other);
  }
  return out;
}

// 如果定义了 USE_XNNPACK，则使用 XNNPACK 加速库执行乘法操作
#ifdef USE_XNNPACK
template <typename scalar_t, bool ReLUFused = false>
Tensor _mul_out_xnnpack(
    const Tensor& self,
    const Tensor& other,
    double output_scale,
    int64_t output_zero_point) {
  using underlying_t = typename scalar_t::underlying;

  const string func_name = "xnnp_mul()";
  // 检查输入张量 self 是否为空
  TORCH_CHECK(self.ndimension() > 0, func_name, ": Got empty input tensor.");
  // 检查 XNNPACK 是否可用
  TORCH_CHECK(
      at::native::xnnpack::available(), func_name, ": XNNPACK is not available")

  // 使用 self 的内存格式建议作为 other 的内存格式，以便 XNNPACK 内核展平所有维度
  auto qa_mem_format = self.suggest_memory_format();
  // 对 self 和 other 进行内存连续性处理
  Tensor self_contig = self.contiguous(qa_mem_format);
  Tensor other_contig = other.contiguous(qa_mem_format);

  // 创建一个新的量化张量 out，使用 empty_affine_quantized 函数
  Tensor out = at::native::empty_affine_quantized(
      at::infer_size_dimvector(self_contig.sizes(), other_contig.sizes()),
      self.scalar_type(),
      c10::nullopt /* layout */,
      kCPU,
      c10::nullopt /* pin_memory */,
      output_scale,
      output_zero_point,
      qa_mem_format);

  // 如果 self_contig 的大小为 0，则直接返回 out
  if (self_contig.size(0) == 0) {
  return out;
}

int64_t self_zero_point = self_contig.q_zero_point();
double self_scale = self_contig.q_scale();
int64_t other_zero_point = other_contig.q_zero_point();
double other_scale = other_contig.q_scale();

int64_t output_min = std::numeric_limits<underlying_t>::min();
int64_t output_max = std::numeric_limits<underlying_t>::max();

if(ReLUFused) {
  /*
   * FIXME: use activationLimits<T>()
   * With <T>, MSVC runs into "error C3862: identifier activationLimits not
   * found".
   */
  // 定义输出的最小和最大量化值
  constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
  constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();
  int64_t qvalue = static_cast<int64_t>(output_zero_point);
  qvalue = std::max<int64_t>(qvalue, qmin);  // 取较大值
  output_min = static_cast<underlying_t>(std::min<int64_t>(qvalue, qmax));  // 取较小值
}

xnn_operator_t xnnp_op = nullptr;
xnnpack_operator xnnp_qmul_operator;

// 创建 xnnpack 乘法运算符 ...
auto status = xnn_create_multiply_nd_qs8(
    self_zero_point,
    self_scale,
    other_zero_point,
    other_scale,
    static_cast<underlying_t>(output_zero_point),
    static_cast<float>(output_scale),
    output_min,
    output_max,
    0,
    &xnnp_op);

TORCH_CHECK(
    status == xnn_status_success,
    func_name,
    ": xnn create operator failed(",
    status,
    ")!");
xnnp_qmul_operator = xnnpack_operator(xnnp_op);

const auto self_shape = xnnp_utils::get_mem_format_aware_shape(self_contig);
const auto other_shape = xnnp_utils::get_mem_format_aware_shape(other_contig);

// 重塑运算符
status = xnn_reshape_multiply_nd_qs8(
    xnnp_qmul_operator.get(),
    self_shape.size(),
    self_shape.data(),
    other_shape.size(),
    other_shape.data(),
    caffe2::pthreadpool_());

TORCH_CHECK(
    status == xnn_status_success,
    func_name,
    ": xnn reshape operator failed(",
    status,
    ")!");

// 设置运算符
status = xnn_setup_multiply_nd_qs8(
    xnnp_qmul_operator.get(),
    reinterpret_cast<const underlying_t*>(self_contig.data_ptr<scalar_t>()),
    reinterpret_cast<const underlying_t*>(other_contig.data_ptr<scalar_t>()),
    reinterpret_cast<underlying_t*>(out.data_ptr<scalar_t>())
);

TORCH_CHECK(
    status == xnn_status_success,
    func_name,
    ": xnn setup operator failed(",
    status,
    ")!");

// 运行运算符
status = xnn_run_operator(
    xnnp_qmul_operator.get(), /* xnn_operator_t op */
    caffe2::pthreadpool_()); /* pthreadpool_t threadpool */
TORCH_CHECK(
    status == xnn_status_success,
    func_name,
    ": xnn run operator failed(",
    status,
    ")");

return out;


注释：

// 返回输出张量 out
return out;
}

// 获取自身张量的量化零点
int64_t self_zero_point = self_contig.q_zero_point();
// 获取自身张量的量化比例
double self_scale = self_contig.q_scale();
// 获取另一张量的量化零点
int64_t other_zero_point = other_contig.q_zero_point();
// 获取另一张量的量化比例
double other_scale = other_contig.q_scale();

// 定义输出张量的最小和最大值
int64_t output_min = std::numeric_limits<underlying_t>::min();
int64_t output_max = std::numeric_limits<underlying_t>::max();

// 如果启用了 ReLU 激活函数
if(ReLUFused) {
  /*
   * FIXME: use activationLimits<T>()
   * With <T>, MSVC runs into "error C3862: identifier activationLimits not
   * found".
   */
  // 定义量化的最小和最大输出值
  constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
  constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();
  int64_t qvalue = static_cast<int64_t>(output_zero_point);
  qvalue = std::max<int64_t>(qvalue, qmin);  // 取较大值
  output_min = static_cast<underlying_t>(std::min<int64_t>(qvalue, qmax));  // 取较小值
}

// 声明 xnnpack 运算符指针
xnn_operator_t xnnp_op = nullptr;
// 声明 xnnpack 乘法运算符
xnnpack_operator xnnp_qmul_operator;

// 创建 xnnpack 乘法运算符，设置量化参数
auto status = xnn_create_multiply_nd_qs8(
    self_zero_point,
    self_scale,
    other_zero_point,
    other_scale,
    static_cast<underlying_t>(output_zero_point),
    static_cast<float>(output_scale),
    output_min,
    output_max,
    0,
    &xnnp_op);

// 检查 xnnpack 运算符创建状态
TORCH_CHECK(
    status == xnn_status_success,
    func_name,
    ": xnn create operator failed(",
    status,
    ")!");
// 将 xnnpack 运算符赋值给 xnnpack 乘法运算符对象
xnnp_qmul_operator = xnnpack_operator(xnnp_op);

// 获取自身张量和另一张量的形状
const auto self_shape = xnnp_utils::get_mem_format_aware_shape(self_contig);
const auto other_shape = xnnp_utils::get_mem_format_aware_shape(other_contig);

// 重塑 xnnpack 乘法运算符，根据张量形状重新设置参数
status = xnn_reshape_multiply_nd_qs8(
    xnnp_qmul_operator.get(),
    self_shape.size(),
    self_shape.data(),
    other_shape.size(),
    other_shape.data(),
    caffe2::pthreadpool_());

// 检查 xnnpack 乘法运算符重塑状态
TORCH_CHECK(
    status == xnn_status_success,
    func_name,
    ": xnn reshape operator failed(",
    status,
    ")!");

// 设置 xnnpack 乘法运算符的输入和输出张量
status = xnn_setup_multiply_nd_qs8(
    xnnp_qmul_operator.get(),
    reinterpret_cast<const underlying_t*>(self_contig.data_ptr<scalar_t>()),
    reinterpret_cast<const underlying_t*>(other_contig.data_ptr<scalar_t>()),
    reinterpret_cast<underlying_t*>(out.data_ptr<scalar_t>())
);

// 检查 xnnpack 乘法运算符设置状态
TORCH_CHECK(
    status == xnn_status_success,
    func_name,
    ": xnn setup operator failed(",
    status,
    ")!");

// 运行 xnnpack 乘法运算符
status = xnn_run_operator(
    xnnp_qmul_operator.get(), /* xnn_operator_t op */
    caffe2::pthreadpool_()); /* pthreadpool_t threadpool */
TORCH_CHECK(
    status == xnn_status_success,
    func_name,
    ": xnn run operator failed(",
    status,
    ")");

// 返回输出张量 out
return out;
#endif // use XNNPACK
#ifdef USE_XNNPACK



template <bool ReLUFused = false>
Tensor _mul_scalar_out(Tensor& out, const Tensor& self, const Scalar& other) {
  // 获取 self 张量的量化零点值
  int64_t self_zero_point = self.q_zero_point();
  // 获取 self 张量的量化比例因子
  double self_scale = self.q_scale();
  // 将 Scalar 转换为 double 类型
  double other_val = other.toDouble();

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double scale_prime;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t zero_point_prime;

  // 使用宏 AT_DISPATCH_QINT_TYPES 根据输出张量的标量类型分发量化整数操作
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qmul_scalar", [&]() {
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    // 获取 underlying_t 类型的最小值和最大值
    int64_t q_min = std::numeric_limits<underlying_t>::min();
    int64_t q_max = std::numeric_limits<underlying_t>::max();

    // 根据 Scalar 的值进行不同的处理分支
    if (other_val > 0.0) {
      // 计算新的比例因子
      scale_prime = other_val * self_scale;
      // 使用原始的零点值
      zero_point_prime = self_zero_point;

      // 如果启用了 ReLU 融合，则调用对应的函数
      if (ReLUFused) {
        qrelu_stub(self.device().type(), self, out);
      } else {
        // 否则直接复制 self 到 out
        out.copy_(self);
      }
      // 设置输出张量的量化器
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          scale_prime, zero_point_prime, self.scalar_type()));
    } else if (other_val == 0.0) {
      // 如果 Scalar 的值为 0
      scale_prime = 1.0;
      zero_point_prime = 0;

      // 使用 TensorIterator 设置 out 张量所有元素为 0
      // Strided "memset"
      auto iter = TensorIterator::unary_op(out, self);
      cpu_kernel_vec(
          iter,
          [&](scalar_t a) -> scalar_t { return scalar_t(0); },
          [&](Vectorized<scalar_t> vec) -> Vectorized<scalar_t> {
            return Vectorized<scalar_t>(scalar_t(0));
          });
      // 设置输出张量的量化器
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          scale_prime, zero_point_prime, self.scalar_type()));
    } else /* other_val < 0.0 */ {
      // 如果 Scalar 的值小于 0
      scale_prime = std::abs(other_val) * self_scale;
      // 根据量化整数类型的范围重新计算零点值
      zero_point_prime = q_max - (self_zero_point - q_min);

      // 使用 TensorIterator 处理 out 张量的每个元素
      auto iter = TensorIterator::unary_op(out, self);
      cpu_kernel(
          iter,
          [&](scalar_t a) -> scalar_t {
            // 应用量化整数的转换公式
            a = scalar_t(underlying_t(q_max + q_min - a.val_));
            if (ReLUFused) {
              // 如果启用了 ReLU 融合，对结果进行修正
              a = scalar_t(std::max(a.val_, underlying_t(zero_point_prime)));
            }
            return a;
          });
      // 设置输出张量的量化器
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          scale_prime, zero_point_prime, self.scalar_type()));
    }
  });

  return out;
}



template <bool ReLUFused = false>
class QMul final {
 public:
  // 静态方法，用于执行量化乘法运算
  static Tensor run(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
    // 检查输入张量是否有效
    check_inputs(qa, qb);
    // 如果定义了 USE_XNNPACK 宏，并且 zero_point 小于 qint8 类型的最大值
    if (zero_point < std::numeric_limits<c10::qint8::underlying>::max() && qa.scalar_type() == kQInt8) {
      // 调用 XNNPACK 库中的乘法函数进行计算
      return _mul_out_xnnpack<c10::qint8, ReLUFused>(qa, qb, scale, zero_point);
    }
    // 根据输入张量的形状创建一个空的量化张量 qc
    auto qc = at::_empty_affine_quantized(
        infer_size_dimvector(qa.sizes(), qb.sizes()),
        at::device(kCPU).dtype(qa.scalar_type()),
        scale,
        zero_point,
        qa.suggest_memory_format());

    // 调用 _mul_out 函数进行乘法操作
    return _mul_out<ReLUFused>(qc, qa, qb);
  }
};
// 定义一个模板类 QMulOut，用于执行量化张量的乘法操作，支持可选的 ReLU 融合功能
template <bool ReLUFused = false>
class QMulOut final {
 public:
  // 静态方法 run，执行量化张量 qa 和 qb 的乘法，并将结果写入张量 out
  static Tensor run(at::Tensor qa, at::Tensor qb, Tensor out) {
    // 检查输入张量 qa 和 qb 是否符合规范
    check_inputs(qa, qb);
    // 调用内部函数 _mul_out 执行乘法操作，支持可选的 ReLU 融合功能
    return _mul_out<ReLUFused>(out, qa, qb);
  }
};

// 定义一个模板类 QMulScalar，用于执行量化张量与标量的乘法操作，支持可选的 ReLU 融合功能
template <bool ReLUFused = false>
class QMulScalar final {
 public:
  // 静态方法 run，执行量化张量 qa 与标量 b 的乘法，并返回结果张量
  static Tensor run(Tensor qa, const Scalar& b) {
    // 检查输入张量 qa 是否为每张量仿射或每张量对称量化，否则抛出错误
    TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Mul.");
    // 根据输入张量 qa 的形状创建一个相同形状的空张量 qc
    auto qc = at::empty_like(qa, qa.suggest_memory_format());
    // 调用内部函数 _mul_scalar_out 执行乘法操作，支持可选的 ReLU 融合功能
    return _mul_scalar_out<ReLUFused>(qc, qa, b);
  }
};

// 定义一个模板类 QMulScalar2，用于执行标量与量化张量的乘法操作，支持可选的 ReLU 融合功能
template <bool ReLUFused = false>
class QMulScalar2 final {
 public:
  // 静态方法 run，执行标量 b 与量化张量 qa 的乘法，并返回结果张量
  static Tensor run(const Scalar& b, Tensor qa) {
    // 检查输入张量 qa 是否为每张量仿射或每张量对称量化，否则抛出错误
    TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Mul.");
    // 根据输入张量 qa 的形状创建一个相同形状的空张量 qc
    auto qc = at::empty_like(qa, qa.suggest_memory_format());
    // 调用内部函数 _mul_scalar_out 执行乘法操作，支持可选的 ReLU 融合功能
    return _mul_scalar_out<ReLUFused>(qc, qa, b);
  }
};

// 定义一个模板类 QMulScalarOut，用于执行量化张量与标量的乘法操作，并将结果写入输出张量 out
template <bool ReLUFused = false>
class QMulScalarOut final {
 public:
  // 静态方法 run，执行量化张量 qa 与标量 b 的乘法，并将结果写入输出张量 out
  static Tensor run(Tensor qa, const Scalar& b, Tensor out) {
    // 检查输入张量 qa 和输出张量 out 是否符合规范
    check_inputs(qa, out);
    // 调用内部函数 _mul_scalar_out 执行乘法操作，支持可选的 ReLU 融合功能
    return _mul_scalar_out<ReLUFused>(out, qa, b);
  }
};

// `torch.jit.trace` 将标量 Scalar 追踪为张量 Tensor
// 在广播支持并且所有 `quantized::mul` 变体合并为 `quantized::mul` 后，此段代码可移除
// 定义一个模板类 QMulScalarTensor，用于执行量化张量与张量标量的乘法操作，支持可选的 ReLU 融合功能
template <bool ReLUFused = false>
class QMulScalarTensor final {
 public:
  // 静态方法 run，执行量化张量 qa 与张量标量 b 的乘法，并返回结果张量
  static Tensor run(Tensor qa, Tensor b) {
    // 检查输入张量 qa 是否为每张量仿射或每张量对称量化，否则抛出错误
    TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Mul.");
    // 根据输入张量 qa 的形状创建一个相同形状的空张量 qc
    auto qc = at::empty_like(qa, qa.suggest_memory_format());
    // 调用内部函数 _mul_scalar_out 执行乘法操作，支持可选的 ReLU 融合功能，将标量 b 转换为张量进行操作
    return _mul_scalar_out<ReLUFused>(qc, qa, b.item());
  }
};

// `torch.jit.trace` 将标量 Scalar 追踪为张量 Tensor
// 在广播支持并且所有 `quantized::mul` 变体合并为 `quantized::mul` 后，此段代码可移除
// 定义一个模板类 QMulScalarTensorOut，用于执行量化张量与张量标量的乘法操作，并将结果写入输出张量 out
template <bool ReLUFused = false>
class QMulScalarTensorOut final {
 public:
  // 静态方法 run，执行量化张量 qa 与张量标量 b 的乘法，并将结果写入输出张量 out
  static Tensor run(Tensor qa, Tensor b, Tensor out) {
    // 检查输入张量 qa 和输出张量 out 是否符合规范
    check_inputs(qa, out);
    // 调用内部函数 _mul_scalar_out 执行乘法操作，支持可选的 ReLU 融合功能，将标量 b 转换为张量进行操作
    return _mul_scalar_out<ReLUFused>(out, qa, b.item());
  }
};
// 定义 TORCH_LIBRARY_IMPL 宏，用于实现 quantized 模块的 QuantizedCPU 版本
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // 实现 quantized::mul 操作，调用 QMul<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul"),                 TORCH_FN(QMul</*ReLUFused=*/false>::run));
  // 实现 quantized::mul.out 操作，调用 QMulOut<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul.out"),             TORCH_FN(QMulOut</*ReLUFused=*/false>::run));
  // 实现 quantized::mul.Scalar 操作，调用 QMulScalar<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar"),          TORCH_FN(QMulScalar</*ReLUFused=*/false>::run));
  // 实现 quantized::mul.Scalar2 操作，调用 QMulScalar2<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar2"),         TORCH_FN(QMulScalar2</*ReLUFused=*/false>::run));
  // 实现 quantized::mul.Scalar_out 操作，调用 QMulScalarOut<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar_out"),      TORCH_FN(QMulScalarOut</*ReLUFused=*/false>::run));
  // 实现 quantized::mul_relu 操作，调用 QMul<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu"),            TORCH_FN(QMul</*ReLUFused=*/true>::run));
  // 实现 quantized::mul_relu.out 操作，调用 QMulOut<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.out"),        TORCH_FN(QMulOut</*ReLUFused=*/true>::run));
  // 实现 quantized::mul_relu.Scalar 操作，调用 QMulScalar<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar"),     TORCH_FN(QMulScalar</*ReLUFused=*/true>::run));
  // 实现 quantized::mul_relu.Scalar2 操作，调用 QMulScalar2<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar2"),    TORCH_FN(QMulScalar2</*ReLUFused=*/true>::run));
  // 实现 quantized::mul_relu.Scalar_out 操作，调用 QMulScalarOut<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar_out"), TORCH_FN(QMulScalarOut</*ReLUFused=*/true>::run));

  // 以下是已弃用的函数，为了向后兼容而保留
  // 实现 quantized::mul_out 操作，调用 QMulOut<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_out"),             TORCH_FN(QMulOut</*ReLUFused=*/false>::run));
  // 实现 quantized::mul_relu_out 操作，调用 QMulOut<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu_out"),        TORCH_FN(QMulOut</*ReLUFused=*/true>::run));
  // 实现 quantized::mul_scalar 操作，调用 QMulScalar<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar"),          TORCH_FN(QMulScalar</*ReLUFused=*/false>::run));
  // 实现 quantized::mul_scalar_relu 操作，调用 QMulScalar<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu"),     TORCH_FN(QMulScalar</*ReLUFused=*/true>::run));
  // 实现 quantized::mul_scalar_out 操作，调用 QMulScalarOut<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_out"),      TORCH_FN(QMulScalarOut</*ReLUFused=*/false>::run));
  // 实现 quantized::mul_scalar_relu_out 操作，调用 QMulScalarOut<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu_out"), TORCH_FN(QMulScalarOut</*ReLUFused=*/true>::run));

  // TODO: 在广播支持后移除以下函数
  // 实现 quantized::mul_scalar.Tensor 操作，调用 QMulScalarTensor<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar.Tensor"), TORCH_FN(QMulScalarTensor</*ReLUFused=*/false>::run));
  // 实现 quantized::mul_scalar_relu.Tensor 操作，调用 QMulScalarTensor<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu.Tensor"), TORCH_FN(QMulScalarTensor</*ReLUFused=*/true>::run));
  // 实现 quantized::mul_scalar_out.Tensor 操作，调用 QMulScalarTensorOut<ReLUFused=false>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_out.Tensor"), TORCH_FN(QMulScalarTensorOut</*ReLUFused=*/false>::run));
  // 实现 quantized::mul_scalar_relu_out.Tensor 操作，调用 QMulScalarTensorOut<ReLUFused=true>::run 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu_out.Tensor"), TORCH_FN(QMulScalarTensorOut</*ReLUFused=*/true>::run));
}

// 结束 quantized 命名空间
}  // namespace
// 结束 at::native 命名空间
}}  // namespace at::native
```