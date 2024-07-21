# `.\pytorch\aten\src\ATen\native\quantized\cpu\BinaryOps.cpp`

```py
// 定义宏以启用仅在方法操作符中使用的Torch断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含ATen库中所需的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/BinaryOps.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

// 根据AT_PER_OPERATOR_HEADERS宏选择包含不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/relu_native.h>
#endif

// 包含标准库头文件
#include <algorithm>
#include <utility>

// 定义命名空间at::native
namespace at {
namespace native {

// 定义一个分发函数指针qadd_relu_stub、qadd_stub、qadd_scalar_relu_stub、qadd_scalar_stub
DEFINE_DISPATCH(qadd_relu_stub);
DEFINE_DISPATCH(qadd_stub);
DEFINE_DISPATCH(qadd_scalar_relu_stub);
DEFINE_DISPATCH(qadd_scalar_stub);

// 嵌套命名空间，未命名的命名空间
namespace {

// 内联函数，检查输入张量qa和qb的属性
inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  // 断言检查：只支持每张量仿射量化
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine,
      "Only per tensor quantization is supported in Add.");
  // 断言检查：输入qa和qb必须具有相同的量化方案
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Add must have the same quantization scheme.");
  // 断言检查：加法操作数应具有相同的数据类型
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "Add operands should have same data type.");
}

// 注意：假设out与self和other具有相同的大小。
// 注意：仅当self、other和out具有相同的dtype时才支持加法。
template <bool ReLUFused = false>
Tensor _add_out(Tensor& out, const Tensor& self, const Tensor& other) {
  // 如果启用ReLU融合，则调用qadd_relu_stub进行加法操作
  if (ReLUFused) {
    qadd_relu_stub(self.device().type(), out, self, other);
  } else {
    // 否则，调用qadd_stub进行加法操作
    qadd_stub(self.device().type(), out, self, other);
  }
  return out;
}

// 注意：假设out与self和other具有相同的大小。
// 注意：仅当self、other和out具有相同的dtype时才支持加法。
template <bool ReLUFused = false>
Tensor _add_scalar_out(Tensor& out, const Tensor& self, const Scalar& other) {
  // 断言检查：仅支持每张量仿射量化
  TORCH_CHECK(
      self.qscheme() == kPerTensorAffine,
      "Only per tensor affine is supported for now!!");
  // 在量化空间中实现张量-标量加法，我们简单地根据以下规则调整量化参数：
  //
  // 让s = scale, z = zero point, c = other.toFloat(), c_q = round(c/s)
  // q_min = scalar类型的最低可表示值
  // q_max = scalar类型的最高可表示值
  //
  // 让s' = 计算得出的输出的scale
  // z' = 输出的计算得出的zero-point
  //
  // 如果q_min > z - c_q
  //   s' = [(q_max - (z - c_q)]/[q_max - q_min] * s
  //   z' = q_min
  //   Xq' = at::requantize_from_int(Xq - z + c_q, s/s', z')
  // 如果q_max < z - c_q
  //   s' = [z - c_q -q_min]/[q_max - q_min] * s
  //   z' = q_max
  //   Xq' = at::requantize_from_int(Xq - z + c_q, s/s', z')
  // 否则
  //   s' = s
  //   z' = z - c_q

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd_scalar", [&]() {
    double s = self.q_scale();
    // 获取 self 对象的零点值
    int64_t z = self.q_zero_point();
    // 将 other 转换为 double 类型
    double c = other.toDouble();
    // 获取 underlying_t 类型的最小值
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    int64_t q_min = std::numeric_limits<underlying_t>::min();
    // 获取 underlying_t 类型的最大值
    int64_t q_max = std::numeric_limits<underlying_t>::max();

    // 计算 c / s 的四舍五入值并转换为 int64_t 类型
    int64_t c_q = std::nearbyint(c / s);

    double s_prime;
    int64_t z_prime;

    // 根据不同情况设置 s_prime 和 z_prime
    if (q_min > z - c_q) {
      // 计算 s_prime
      s_prime = (((double)q_max - (z - c_q))) / ((double)q_max - q_min) * s;
      // 设置 z_prime
      z_prime = q_min;
      // 设置输出张量的量化方式为固定范围量化器
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      // 如果启用了 ReLU 合并，则调用带有 ReLU 的量化加法
      if (ReLUFused) {
        qadd_scalar_relu_stub(self.device().type(), out, self, c_q);
      } else {
        // 否则，调用普通的量化加法
        qadd_scalar_stub(self.device().type(), out, self, c_q);
      }
    } else if (q_max < z - c_q) {
      // 计算 s_prime
      s_prime = ((double)(z - c_q) - q_min) / ((double)q_max - q_min) * s;
      // 设置 z_prime
      z_prime = q_max;
      // 设置输出张量的量化方式为固定范围量化器
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      // 如果启用了 ReLU 合并，则调用带有 ReLU 的量化加法
      if (ReLUFused) {
        qadd_scalar_relu_stub(self.device().type(), out, self, c_q);
      } else {
        // 否则，调用普通的量化加法
        qadd_scalar_stub(self.device().type(), out, self, c_q);
      }
    } else {
      // 默认情况下，保持原始的 s 和 z - c_q
      s_prime = s;
      z_prime = z - c_q;
      // 将 self 复制到 out
      out.copy_(self);
      // 设置输出张量的量化方式为固定范围量化器
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      // 如果启用了 ReLU 合并，则调用带有 ReLU 的量化函数
      if (ReLUFused) {
        at::native::relu_quantized_cpu_(out);
      }
    }
  });
  // 返回输出张量 out
  return out;
#ifdef USE_PYTORCH_QNNPACK
// 定义一个模板函数 qnnpack_add，用于执行 QNNPACK 的加法操作
template <bool ReLUFused = false>
Tensor qnnpack_add(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  // 检查输入张量的维度是否大于0，确保不是空输入
  TORCH_CHECK(qa.ndimension() > 0, "qnnpack_add(): Got empty input tensor.");
  // 检查输入张量的数据类型，要求都是 c10::kQUInt8 类型
  TORCH_CHECK(qa.scalar_type() == c10::kQUInt8 && qb.scalar_type() == c10::kQUInt8,
                "qnnpack_add(): Expected both input data types to be ",
                toString(c10::kQUInt8),
                " but got ",
                toString(qa.scalar_type()),
                " and ",
                toString(qb.scalar_type()));
  // 对输入张量 qa 进行内存连续性处理，使用建议的内存格式
  Tensor qa_contig = qa.contiguous(qa.suggest_memory_format());
  // 对输入张量 qb 进行内存连续性处理，使用 qa 的内存格式
  // 这样做是因为底层的 kernel 可以展平所有维度并同时迭代两个张量
  // 大多数情况下，qa 和 qb 都使用相同的内存格式
  // 当它们的内存格式不同时，会有复制开销将其转换为 qa 的内存格式
  Tensor qb_contig = qb.contiguous(qa.suggest_memory_format());

  // 获取 qa_contig 和 qb_contig 的量化零点和量化比例
  const auto a_zero_point = qa_contig.q_zero_point();
  const auto b_zero_point = qb_contig.q_zero_point();
  const auto a_scale = qa_contig.q_scale();
  const auto b_scale = qb_contig.q_scale();

  // 创建一个空的仿射量化的输出张量 qy
  Tensor qy = at::native::empty_affine_quantized(
      qa_contig.sizes(),
      kQUInt8,
      c10::nullopt /* layout */,
      kCPU,
      c10::nullopt /* pin_memory */,
      scale,
      zero_point,
      qa.suggest_memory_format());

  // 如果输入张量 qa_contig 的大小为0，则直接返回空的输出张量 qy
  if (qa_contig.size(0) == 0) {
    return qy;
  }

  // 初始化 QNNPACK 库
  initQNNPACK();

  // 创建 QNNPACK 操作符对象
  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  // 计算输入张量元素个数
  size_t num_elems = qa_contig.numel() / qa_contig.size(0);

  // 计算输出张量的最小值和最大值，根据是否使用 ReLU 进行条件判断
  auto output_min = ReLUFused
      // 使用 ReLU 激活函数限制的输出最小值
      ? activationLimits<uint8_t>(scale, zero_point, Activation::RELU)
            .first
      // 如果未使用 ReLU，则输出范围为 uint8_t 的最小值
      : std::numeric_limits<uint8_t>::min();
  auto output_max = ReLUFused
      // 使用 ReLU 激活函数限制的输出最大值
      ? activationLimits<uint8_t>(scale, zero_point, Activation::RELU)
            .second
      // 如果未使用 ReLU，则输出范围为 uint8_t 的最大值
      : std::numeric_limits<uint8_t>::max();

  // 创建 QNNPACK 加法操作符
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_add_nc_q8(
      num_elems /* 输入大小 */,
      a_zero_point /* 张量 a 的零点 */,
      a_scale /* 张量 a 的缩放因子 */,
      b_zero_point /* 张量 b 的零点 */,
      b_scale /* 张量 b 的缩放因子 */,
      static_cast<uint8_t>(zero_point) /* 输出张量的零点 */,
      scale /* 输出张量的缩放因子 */,
      output_min /* 输出张量的最小值 */,
      output_max /* 输出张量的最大值 */,
      0 /* 标志 */,
      &qnnpack_operator);

  // 断言 QNNPACK 操作符创建成功
  TORCH_INTERNAL_ASSERT(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK Add operator");

  // 使用智能指针管理 QNNPACK 操作符的生命周期
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  // 设置 QNNPACK 加法操作符的输入和输出
  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_add_nc_q8(
      qnnpack_operator /* 加法操作符 */,
      qa_contig.size(0) /* 批次大小 */,
      (uint8_t*)qa_contig.data_ptr<c10::quint8>() /* 张量 a 数据指针 */,
      num_elems /* 张量 a 步长 */,
      (uint8_t*)qb_contig.data_ptr<c10::quint8>() /* 张量 b 数据指针 */,
      num_elems /* 张量 b 步长 */,
      (uint8_t*)qy.data_ptr<c10::quint8>() /* 输出张量数据指针 */,
      num_elems /* 输出张量步长 */);

  // 断言 QNNPACK 加法操作符设置成功
  TORCH_INTERNAL_ASSERT(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Add operator");

  // 获取线程池对象
  pthreadpool_t threadpool = caffe2::pthreadpool_();

  // 执行 QNNPACK 加法操作
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

  // 断言 QNNPACK 加法操作执行成功
  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Add operator");

  // 返回输出张量 qy
  return qy;
#ifdef USE_PYTORCH_QNNPACK
#endif // USE_PYTORCH_QNNPACK

#ifdef USE_XNNPACK
C10_ALWAYS_INLINE
// 创建一个使用指定参数的 xnnpack 加法运算符
enum xnn_status xnnp_create_add_nd(
    int8_t azp,         /* 第一个输入的零点值 */
    float ascale,       /* 第一个输入的量化比例 */
    int8_t bzp,         /* 第二个输入的零点值 */
    float bscale,       /* 第二个输入的量化比例 */
    int8_t czp,         /* 输出的零点值 */
    float cscale,       /* 输出的量化比例 */
    int8_t output_min,  /* 输出的最小值 */
    int8_t output_max,  /* 输出的最大值 */
    uint32_t flags,     /* 标志位 */
    xnn_operator_t* op) /* 返回的 xnnpack 操作符指针 */
{
  return xnn_create_add_nd_qs8(
      azp,        /* int8_t input1_zero_point   */
      ascale,     /* float input1_scale         */
      bzp,        /* int8_t input2_zero_point   */
      bscale,     /* float input2_scale         */
      czp,        /* int8_t output_zero_point   */
      cscale,     /* float output_scale         */
      output_min, /* int8_t output_min          */
      output_max, /* int8_t output_max          */
      flags,      /* uint32_t flags             */
      op);        /* xnn_operator_t* add_op_out */
}

C10_ALWAYS_INLINE
// 重塑已有的 xnnpack 加法操作符，改变输入形状
enum xnn_status xnnp_reshape_add_nd(
    xnn_operator_t op,                   /* 已有的 xnnpack 加法操作符 */
    const std::vector<size_t>& a_shape,  /* 第一个输入的形状 */
    const std::vector<size_t>& b_shape,  /* 第二个输入的形状 */
    pthreadpool_t pt_pool)               /* 线程池指针 */
{
  return xnn_reshape_add_nd_qs8(
      op,             /* xnn_operator_t add_op      */
      a_shape.size(), /* size_t num_input1_dims     */
      a_shape.data(), /* const size_t* input1_shape */
      b_shape.size(), /* size_t num_input2_dims     */
      b_shape.data(), /* const size_t* input2_shape */
      pt_pool);       /* pthreadpool_t threadpool   */
}

C10_ALWAYS_INLINE
// 设置 xnnpack 加法操作符的输入和输出
enum xnn_status xnnp_setup_add_nd(
    xnn_operator_t op,      /* 已有的 xnnpack 加法操作符 */
    const int8_t* da,       /* 第一个输入的数据指针 */
    const int8_t* db,       /* 第二个输入的数据指针 */
    int8_t* dc,             /* 输出数据指针 */
    pthreadpool_t pt_pool)  /* 线程池指针 */
{
  return xnn_setup_add_nd_qs8(
      op,             /* xnn_operator_t add_op      */
      da,             /* const int8_t* input1       */
      db,             /* const int8_t* input2       */
      dc);            /* int8_t* output             */
}

template <typename scalar_t, bool ReLUFused = false>
// 使用 xnnpack 执行加法运算，并返回量化后的结果张量
Tensor xnnp_add(Tensor qa,           /* 第一个输入张量 */
                Tensor qb,           /* 第二个输入张量 */
                double scale,        /* 输出的量化比例 */
                int64_t zero_point)  /* 输出的零点值 */
{
  using underlying_t = typename scalar_t::underlying;
  const string func_name = "xnnp_add()";
  TORCH_CHECK(qa.ndimension() > 0, func_name, ": Got empty input tensor.");
  TORCH_CHECK(at::native::xnnpack::available(), func_name, ": XNNPACK is not available")

  // 使用第一个输入张量的内存格式对第二个输入进行建议，以便 xnnpack 核心可以展平所有维度
  auto qa_mem_format = qa.suggest_memory_format();
  Tensor qa_contig = qa.contiguous(qa_mem_format);
  Tensor qb_contig = qb.contiguous(qa_mem_format);

  const auto a_zero_point = qa_contig.q_zero_point();
  const auto b_zero_point = qb_contig.q_zero_point();
  const auto a_scale = qa_contig.q_scale();
  const auto b_scale = qb_contig.q_scale();

  // 创建一个与输入形状兼容的量化输出张量
  Tensor qy = at::native::empty_affine_quantized(
      at::infer_size_dimvector(qa_contig.sizes(), qb_contig.sizes()),
      qa.scalar_type(),
      c10::nullopt /* layout */,
      kCPU,
      c10::nullopt /* pin_memory */,
      scale,
      zero_point,
      qa_mem_format);

  // 如果第一个输入张量的大小为零，则直接返回空的量化输出张量
  if (qa_contig.size(0) == 0) {
    return qy;
  }

  xnn_operator_t xnnp_op = nullptr;  // 定义一个 XNN 操作符对象，默认为 nullptr
  xnnpack_operator xnnp_add_operator;  // 定义一个 XNNPack 操作符对象

  auto output_max = std::numeric_limits<underlying_t>::max();  // 获取 underlying_t 类型的最大值
  auto output_min = std::numeric_limits<underlying_t>::min();  // 获取 underlying_t 类型的最小值
  if (ReLUFused) {
    /*
     * FIXME: use activationLimits<T>()
     * With <T>, MSVC runs into "error C3862: identifier activationLimits not found".
     */
    constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();  // 获取 underlying_t 类型的最小值
    constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();  // 获取 underlying_t 类型的最大值
    int64_t qvalue = static_cast<int64_t>(zero_point);  // 将 zero_point 转换为 int64_t 类型赋给 qvalue
    qvalue = std::max<int64_t>(qvalue, qmin);  // 取 qvalue 和 qmin 中的较大值
    output_min = static_cast<underlying_t>(std::min<int64_t>(qvalue, qmax));  // 将 qvalue 与 qmax 中的较小值转换为 underlying_t 类型，并赋给 output_min
  }

  // 创建一个加法操作的运算符
  auto status = xnnp_create_add_nd(
      a_zero_point,
      a_scale,
      b_zero_point,
      b_scale,
      static_cast<underlying_t>(zero_point),
      static_cast<float>(scale),
      output_min,
      output_max,
      0,
      &xnnp_op);
  xnnp_add_operator = xnnpack_operator(xnnp_op);  // 将创建的 XNN 操作符对象赋给 XNNPack 操作符对象
  TORCH_CHECK(
      status == xnn_status_success,
      func_name, ": xnn create operator failed(", status,")!");

  const auto qa_shape = xnnp_utils::get_mem_format_aware_shape(qa_contig);  // 获取 QA 张量的内存格式感知形状
  const auto qb_shape = xnnp_utils::get_mem_format_aware_shape(qb_contig);  // 获取 QB 张量的内存格式感知形状

  // 重塑操作符
  status = xnnp_reshape_add_nd(
      xnnp_add_operator.get(),
      qa_shape,
      qb_shape,
      caffe2::pthreadpool_());

  TORCH_CHECK(
      status == xnn_status_success,
      func_name, ": xnn reshape operator failed(", status,")!");

  // 设置操作符
  status = xnnp_setup_add_nd(
      xnnp_add_operator.get(),
      reinterpret_cast<const underlying_t*>(qa_contig.data_ptr<scalar_t>()),
      reinterpret_cast<const underlying_t*>(qb_contig.data_ptr<scalar_t>()),
      reinterpret_cast<underlying_t*>(qy.data_ptr<scalar_t>()),
      caffe2::pthreadpool_());
  TORCH_CHECK(
      status == xnn_status_success,
      func_name, ": xnn setup operator failed(", status,")!");

  // 运行操作符
  status = xnn_run_operator(
      xnnp_add_operator.get(), /* xnn_operator_t op */
      caffe2::pthreadpool_()); /* pthreadpool_t threadpool */
  TORCH_CHECK(
      status == xnn_status_success,
      func_name, ": xnn run operator failed(", status,")");
  return qy;
#endif // USE_XNNPACK

// 定义模板函数 qadd，用于在量化操作中执行加法
template <bool ReLUFused = false>
Tensor qadd(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  // 检查输入张量的有效性
  check_inputs(qa, qb);

  // 如果当前的量化引擎为 QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    // 检查输入张量的数据类型必须相同
    TORCH_CHECK(
        qa.scalar_type() == qb.scalar_type(),
        "Both inputs to qadd must have same type");

#ifdef USE_XNNPACK
    // 如果输入张量的数据类型为 kQInt8，调用 XNNPACK 加法函数
    if (qa.scalar_type() == kQInt8) {
          return xnnp_add<c10::qint8, ReLUFused>(qa, qb, scale, zero_point);
    }
#endif // USE_XNNPACK

#ifdef USE_PYTORCH_QNNPACK
    // 如果使用 PyTorch QNNPACK 并且输入张量的尺寸相同，执行 QNNPACK 加法操作
    if(qa.sizes() == qb.sizes() && /* qnnpack does not support boardcasting */
      qa.scalar_type() == kQUInt8) {
    return qnnpack_add<ReLUFused>(qa, qb, scale, zero_point);
    }
#endif // USE_PYTORCH_QNNPACK
  }
  // 创建一个新的量化张量，与输入张量相同的形状
  auto qc = at::_empty_affine_quantized(
      qa.sizes(),
      at::device(kCPU)
         .dtype(qa.scalar_type())
         .memory_format(qa.suggest_memory_format()),
      scale,
      zero_point,
      c10::nullopt);
  // 调用内部函数执行加法操作，并返回结果张量
  return _add_out<ReLUFused>(qc, qa, qb);
}

// 定义模板函数 qadd_out，在指定的输出张量上执行量化加法
template <bool ReLUFused = false>
Tensor qadd_out(Tensor qa, Tensor qb, Tensor out) {
  // 检查输入张量的有效性
  check_inputs(qa, qb);
  // 检查输入和输出张量的有效性
  check_inputs(qa, out);
  // 在输出张量上执行量化加法操作，并返回结果张量
  return _add_out<ReLUFused>(out, qa, qb);
}

// 定义模板函数 qadd_scalar，在输入张量和标量之间执行量化加法
template <bool ReLUFused = false>
Tensor qadd_scalar(Tensor qa, const Scalar& b) {
  // 检查输入张量是否支持每张量仿射量化
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Add.");
  // 创建一个新的量化张量，与输入张量相同的形状
  auto qc = at::empty_like(qa, qa.suggest_memory_format());
  // 在新创建的张量上执行量化加法操作，并返回结果张量
  return _add_scalar_out<ReLUFused>(qc, qa, b);
}

// 定义模板函数 qadd_scalar2，在输入标量和张量之间执行量化加法
template <bool ReLUFused = false>
Tensor qadd_scalar2(Scalar b, Tensor qa) {
  // 检查输入张量是否支持每张量仿射量化
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Add.");
  // 创建一个新的量化张量，与输入张量相同的形状
  auto qc = at::empty_like(qa, qa.suggest_memory_format());
  // 在新创建的张量上执行量化加法操作，并返回结果张量
  return _add_scalar_out<ReLUFused>(qc, qa, b);
}

// 定义模板函数 qadd_scalar_out，在指定的输出张量上执行量化加法
template <bool ReLUFused = false>
Tensor qadd_scalar_out(Tensor qa, const Scalar& b, Tensor out) {
  // 检查输入和输出张量的有效性
  check_inputs(qa, out);
  // 在输出张量上执行量化加法操作，并返回结果张量
  return _add_scalar_out<ReLUFused>(out, qa, b);
}

// `torch.jit.trace` 将标量视为张量
// 在支持广播操作并且所有 `quantized::add` 变体合并为 `quantized::add` 后可以移除此函数
template <bool ReLUFused = false>
Tensor qadd_scalar_tensor(Tensor qa, Tensor b) {
  // 在输入张量和标量之间执行量化加法，并返回结果张量
  return qadd_scalar(std::move(qa), b.item());
}

// `torch.jit.trace` 将标量视为张量
// 在支持广播操作并且所有 `quantized::add` 变体合并为 `quantized::add` 后可以移除此函数
template <bool ReLUFused = false>
Tensor qadd_scalar_tensor_out(Tensor qa, Tensor b, Tensor out) {
  // 在指定的输出张量上执行量化加法，并返回结果张量
  return qadd_scalar_out(std::move(qa), b.item(), std::move(out));
}
// 定义 TORCH_LIBRARY_IMPL 宏，注册 quantized 模块的 CPU 实现
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // 注册 quantized::add 方法的实现，使用 qadd 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add"),                 TORCH_FN(qadd</*ReLUFused=*/false>));
  // 注册 quantized::add.out 方法的实现，使用 qadd_out 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.out"),             TORCH_FN(qadd_out</*ReLUFused=*/false>));
  // 注册 quantized::add.Scalar 方法的实现，使用 qadd_scalar 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar"),          TORCH_FN(qadd_scalar</*ReLUFused=*/false>));
  // 注册 quantized::add.Scalar2 方法的实现，使用 qadd_scalar2 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar2"),          TORCH_FN(qadd_scalar2</*ReLUFused=*/false>));
  // 注册 quantized::add.Scalar_out 方法的实现，使用 qadd_scalar_out 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar_out"),      TORCH_FN(qadd_scalar_out</*ReLUFused=*/false>));
  // 注册 quantized::add_relu 方法的实现，使用 qadd 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu"),            TORCH_FN(qadd</*ReLUFused=*/true>));
  // 注册 quantized::add_relu.out 方法的实现，使用 qadd_out 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.out"),        TORCH_FN(qadd_out</*ReLUFused=*/true>));
  // 注册 quantized::add_relu.Scalar 方法的实现，使用 qadd_scalar 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar"),     TORCH_FN(qadd_scalar</*ReLUFused=*/true>));
  // 注册 quantized::add_relu.Scalar2 方法的实现，使用 qadd_scalar2 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar2"),     TORCH_FN(qadd_scalar2</*ReLUFused=*/true>));
  // 注册 quantized::add_relu.Scalar_out 方法的实现，使用 qadd_scalar_out 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar_out"), TORCH_FN(qadd_scalar_out</*ReLUFused=*/true>));
  // 以下为已废弃的函数，保留以确保向后兼容性
  // 注册 quantized::add_out 方法的实现，使用 qadd_out 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_out"),             TORCH_FN(qadd_out</*ReLUFused=*/false>));
  // 注册 quantized::add_relu_out 方法的实现，使用 qadd_out 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu_out"),        TORCH_FN(qadd_out</*ReLUFused=*/true>));
  // 注册 quantized::add_scalar 方法的实现，使用 qadd_scalar 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar"),          TORCH_FN(qadd_scalar</*ReLUFused=*/false>));
  // 注册 quantized::add_scalar_relu 方法的实现，使用 qadd_scalar 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu"),     TORCH_FN(qadd_scalar</*ReLUFused=*/true>));
  // 注册 quantized::add_scalar_out 方法的实现，使用 qadd_scalar_out 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_out"),      TORCH_FN(qadd_scalar_out</*ReLUFused=*/false>));
  // 注册 quantized::add_scalar_relu_out 方法的实现，使用 qadd_scalar_out 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu_out"), TORCH_FN(qadd_scalar_out</*ReLUFused=*/true>));
  // 注册 quantized::add_scalar.Tensor 方法的实现，使用 qadd_scalar_tensor 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar.Tensor"),   TORCH_FN(qadd_scalar_tensor</*ReLUFused=*/false>));
  // 注册 quantized::add_scalar_relu.Tensor 方法的实现，使用 qadd_scalar_tensor 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu.Tensor"), TORCH_FN(qadd_scalar_tensor</*ReLUFused=*/true>));
  // 注册 quantized::add_scalar_out.Tensor 方法的实现，使用 qadd_scalar_tensor_out 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_out.Tensor"), TORCH_FN(qadd_scalar_tensor_out</*ReLUFused=*/false>));
  // 注册 quantized::add_scalar_relu_out.Tensor 方法的实现，使用 qadd_scalar_tensor_out 函数，并与 ReLU 合并
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu_out.Tensor"), TORCH_FN(qadd_scalar_tensor_out</*ReLUFused=*/true>));
}

// 定义 TORCH_LIBRARY_IMPL 宏，注册 _quantized 模块的 CPU 实现
TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  // 注册 _quantized::add 方法的实现，使用 qadd 函数
  m.impl(TORCH_SELECTIVE_NAME("_quantized::add"), TORCH_FN(qadd</*ReLUFused=*/false>));
}

// 定义 quantized_add 函数，实现量化加法操作
Tensor quantized_add(Tensor qa, Tensor qb, double scale, int64_t zero_point){
  // 调用 qadd 函数执行量化加法操作，不使用 ReLU 合并
  return qadd<false>(std::move(qa), std::move(qb), scale, zero_point);
}

// 结束定义 at::native 命名空间
}}  // namespace at::native
```