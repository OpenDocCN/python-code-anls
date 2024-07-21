# `.\pytorch\aten\src\ATen\native\quantized\cpu\qtanh.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/tanh_native.h>
#endif

namespace at {
namespace native {

// 定义一个分发函数指针，用于调度 QNNPACK TanH 运算
DEFINE_DISPATCH(qtanh_stub);

#ifdef USE_PYTORCH_QNNPACK
// 这个函数用于执行 QNNPACK 加速的 TanH 操作，固定输出 scale=2.0/256, zp=128, dtype=quint8
static Tensor qnnpack_tanh(Tensor input) {
  // 检查输入张量维度是否大于0
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_tanh(): Got empty input tensor");
  // 检查输入张量的数据类型是否为 quint8
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
               "qnnpack_tanh(): Expected input data type ",
               toString(c10::kQUInt8),
               " but got ",
               toString(input.scalar_type()));
  
  // 定义输出的 scale 和 zero point
  Tensor qy;
  constexpr float output_scale = 2.0f / 256.0f;
  constexpr int32_t output_zero_point = 128;

  // 初始化 QNNPACK 库
  initQNNPACK();

  // 使输入张量在内存中连续，并根据建议的内存格式进行处理
  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  // 计算输入张量的总元素数
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }
  // 获取输入张量的零点和缩放因子
  const auto zero_point = input_contig.q_zero_point();
  const auto scale = input_contig.q_scale();

  // 创建 QNNPACK TanH 运算符
  pytorch_qnnp_operator_t tanh_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_tanh_nc_q8(
    num_elems /* channels */,
    zero_point /* input zero point */,
    scale /* input scale */,
    output_zero_point /* output zero point */,
    output_scale /* output scale */,
    std::numeric_limits<uint8_t>::min() /* output min */,
    std::numeric_limits<uint8_t>::max() /* output max */,
    0 /* flags */,
    &tanh_op);

  // 使用智能指针管理 QNNPACK 运算符，确保在离开作用域时正确释放资源
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(tanh_op);

  // 断言 QNNPACK TanH 运算符创建是否成功
  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK TanH operator");

  // 创建输出张量 qy，采用仿射量化方式
  qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    at::device(kCPU).dtype(input_contig.dtype()),
    output_scale,
    output_zero_point,
    input_contig.suggest_memory_format());

  // 设置 QNNPACK TanH 运算符，准备执行
  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_tanh_nc_q8(
    tanh_op,
    input_contig.size(0) /* batch size */,
    (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
    num_elems /* input stride */,
    (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
    num_elems /* output stride */);

  // 断言 QNNPACK TanH 运算符设置是否成功
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK TanH operator");

  // 获取线程池以便在多线程环境下执行 QNNPACK TanH 运算
  pthreadpool_t threadpool = caffe2::pthreadpool_();

  // 执行 QNNPACK TanH 运算符
  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(tanh_op, threadpool);

  // 断言 QNNPACK TanH 运算符执行是否成功
  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK TanH operator");

# 检查运行状态是否为 QNNPACK TanH 操作成功，否则输出错误信息并终止程序


  return qy;

# 返回变量 `qy` 的值作为函数的结果
#endif  // USE_PYTORCH_QNNPACK



Tensor tanh_quantized_cpu(const Tensor& qx) {
#ifdef USE_PYTORCH_QNNPACK
  // 如果使用 QNNPACK 引擎并且输入张量 qx 的数据类型是 kQUInt8
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    // 调用 QNNPACK 提供的 tanh 函数，并返回结果
    return qnnpack_tanh(qx);
  }
#endif  // USE_PYTORCH_QNNPACK
  // 如果未使用 QNNPACK 引擎或者输入类型不是 kQUInt8，则使用 qtanh_stub 函数处理 qx，并将结果存储在 qy 中
  Tensor qy;
  qtanh_stub(qx.device().type(), qx, qy);
  // 返回处理后的张量 qy
  return qy;
}
}}  // namespace at::native



namespace at::native



}
```