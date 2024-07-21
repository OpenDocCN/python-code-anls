# `.\pytorch\aten\src\ATen\native\quantized\cpu\qsigmoid.cpp`

```py
// 定义编译标志以仅支持方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 PyTorch 的头文件和相关依赖
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来包含不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/sigmoid_native.h>
#endif

#include <algorithm>
#include <utility>

// 定义命名空间 at 和 native
namespace at {
namespace native {

// 定义 QSigmoid 的分发调度器
DEFINE_DISPATCH(qsigmoid_stub);

// 如果定义了 USE_PYTORCH_QNNPACK，则使用 QNNPACK 提供的量化 sigmoid 函数
#ifdef USE_PYTORCH_QNNPACK
static Tensor qnnpack_sigmoid(
    Tensor input, double output_scale, int64_t output_zero_point) {
  // 检查输入张量维度是否大于0
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_sigmoid(): Got empty input tensor");
  // 检查输入张量数据类型是否为 c10::kQUInt8
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
               "qnnpack_sigmoid(): Expected input data type ",
               toString(c10::kQUInt8),
               " but got ",
               toString(input.scalar_type()));

  Tensor qy;
  
  // 初始化 QNNPACK 库
  initQNNPACK();

  // 将输入张量转换为连续的内存布局
  Tensor input_contig = input.contiguous(input.suggest_memory_format());

  // 计算输入张量元素的总数
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }

  // 获取输入张量的量化零点和量化比例
  const auto zero_point = input_contig.q_zero_point();
  const auto scale = input_contig.q_scale();

  // 创建 QNNPACK sigmoid 运算符
  pytorch_qnnp_operator_t sigmoid_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_sigmoid_nc_q8(
    num_elems /* channels */,
    zero_point /* input zero point */,
    scale /* input scale */,
    output_zero_point /* output zero point */,
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    output_scale /* output scale */,
    std::numeric_limits<uint8_t>::min() /* output min */,
    std::numeric_limits<uint8_t>::max() /* output max */,
    0 /* flags */,
    &sigmoid_op);

  // 使用智能指针管理 QNNPACK sigmoid 运算符的生命周期
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(sigmoid_op);

  // 断言 QNNPACK sigmoid 运算符创建成功
  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK sigmoid operator");

  // 创建输出张量，使用 _empty_affine_quantized 函数
  qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    at::device(kCPU).dtype(input_contig.dtype()),
    output_scale,
    output_zero_point,
    input_contig.suggest_memory_format());

  // 设置 QNNPACK sigmoid 运算符
  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_sigmoid_nc_q8(
    sigmoid_op,
    input_contig.size(0) /* batch size */,
    (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
    num_elems /* input stride */,
    (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
    num_elems /* output stride */);

该行代码是一个函数调用或语句，但是在提供了一条注释之后，我们无法确定具体的函数或操作。它似乎涉及到一个输出步幅（output stride）的概念。


  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK sigmoid operator");

这里使用了一个宏 `TORCH_INTERNAL_ASSERT`，用于在条件不满足时输出一条错误信息。它检查 QNNPACK sigmoid 运算符的设置状态，确保设置成功。


  pthreadpool_t threadpool = caffe2::pthreadpool_();

这行代码声明了一个 pthread 线程池对象 `threadpool`，通过调用 `caffe2::pthreadpool_()` 函数来获取。


  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(sigmoid_op, threadpool);

这里调用了 `pytorch_qnnp_run_operator` 函数来运行 QNNPACK sigmoid 运算符，使用之前声明的线程池 `threadpool` 执行运算。


  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK sigmoid operator");

再次使用 `TORCH_INTERNAL_ASSERT` 宏来检查运算符的运行状态，确保运行成功。如果运行失败，则输出相应的错误信息。


  return qy;

函数返回变量 `qy`，这里未提供具体代码以查看 `qy` 的类型或定义，但可以推测这是函数的返回值。
#endif  // USE_PYTORCH_QNNPACK



// 如果定义了 USE_PYTORCH_QNNPACK 宏，则结束当前的条件编译块

// This ALWAYS outputs scale=1.0/256, dtype=quint8
// The zero_point is 0 for qint32 and quint8, but -128 for qint8.
Tensor sigmoid_quantized_cpu(const Tensor& qx) {
#ifdef USE_PYTORCH_QNNPACK
  // 如果使用 QNNPACK 引擎，并且输入张量的数据类型是 kQUInt8
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    // 定义输出的 scale 和 zero_point
    constexpr double output_scale = 1.0f / 256.0f;
    constexpr int64_t output_zero_point = 0;
    // 调用 qnnpack_sigmoid 函数，使用定义的 scale 和 zero_point 进行计算并返回结果
    return qnnpack_sigmoid(qx, output_scale, output_zero_point);
  }
#endif  // USE_PYTORCH_QNNPACK

  // 对于其他情况，定义一个输出张量 qy
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
    // Naive implementation: uses dequantize/execute/quantize routine
    // - Output scale is set to 1.0 / 2^(BIT_NUM)
    // - For signed types output zero point is set to 0
    // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
    // See https://stackoverflow.com/a/34448562/3606192 for potential
    // optimizations

    // 设置默认的输出 scale
    double output_scale = 0.00390625;  // 1.0 / 2^8
    int64_t output_zero_point = 0;

    // 根据输入张量的数据类型进行不同的设置
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    if (SCALAR_TYPE == at::kQInt32) {
      // 如果数据类型是 kQInt32，则设置特定的 output scale
      output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
    } else if (SCALAR_TYPE == at::kQInt8) {
      // 如果数据类型是 kQInt8，则设置特定的 output zero_point
      output_zero_point = -128;
    }

    // 调用 qsigmoid_stub 函数，使用定义的 scale 和 zero_point 进行计算
    qsigmoid_stub(qx.device().type(), qx, qy, output_scale, output_zero_point);
  });

  // 返回计算后的输出张量 qy
  return qy;
}

namespace {

class QSigmoid final {
 public:
  static Tensor run(Tensor qx, double output_scale, int64_t output_zero_point) {
#ifdef USE_PYTORCH_QNNPACK
    // 如果使用 QNNPACK 引擎，并且输入张量的数据类型是 kQUInt8
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
        qx.scalar_type() == kQUInt8) {
      // 调用 qnnpack_sigmoid 函数，使用给定的 scale 和 zero_point 进行计算并返回结果
      return qnnpack_sigmoid(std::move(qx), output_scale, output_zero_point);
    }
#endif  // USE_PYTORCH_QNNPACK

    // 否则，定义一个输出张量 qy
    Tensor qy;
    // 调用 qsigmoid_stub 函数，使用给定的 scale 和 zero_point 进行计算
    qsigmoid_stub(qx.device().type(), qx, qy, output_scale, output_zero_point);
    // 返回计算后的输出张量 qy
    return qy;
  }
};

// 注册 QSigmoid 类作为 quantized::sigmoid 的实现
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::sigmoid"), TORCH_FN(QSigmoid::run));
}
}  // namespace

}}  // namespace at::native



// 结束匿名命名空间
// 结束命名空间 at::native
```