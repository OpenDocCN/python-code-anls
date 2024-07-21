# `.\pytorch\aten\src\ATen\native\quantized\cpu\qclamp.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>
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
#include <ATen/ops/clamp_native.h>
#include <ATen/ops/hardtanh_native.h>
#endif

#include <algorithm>

namespace at {
namespace native {

// 定义了 QNNPACK 的 clamp 操作的调度器
DEFINE_DISPATCH(qclamp_stub);
DEFINE_DISPATCH(qclamp_min_stub);
DEFINE_DISPATCH(qclamp_max_stub);

namespace {

#ifdef USE_PYTORCH_QNNPACK
// 使用 QNNPACK 库实现的量化 clamp 操作
Tensor qnnpack_clamp(Tensor input, const Scalar& min, const Scalar& max) {

  // 检查输入张量的维度是否大于 0
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_clamp(): Got empty input tensor");

  // 初始化 QNNPACK 库
  initQNNPACK();

  // 确保输入张量是连续的，并根据推荐的内存格式重新排列数据
  Tensor input_contig = input.contiguous(input.suggest_memory_format());

  // 计算输入张量的总元素个数
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }

  // 将最小值和最大值转换为 float 类型
  auto min_f = min.to<float>();
  auto max_f = max.to<float>();

  // 根据输入张量的量化参数，将浮点数值量化为对应的量化整数值
  uint8_t min_q =
      at::native::quantize_val<quint8>(input.q_scale(), input.q_zero_point(), min_f).val_;
  uint8_t max_q =
      at::native::quantize_val<quint8>(input.q_scale(), input.q_zero_point(), max_f).val_;

  // 创建 QNNPACK 的 clamp 运算符对象
  pytorch_qnnp_operator_t clamp_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_clamp_nc_u8(
    num_elems, // 通道数（对于 nc 模式，此处为总元素数）
    min_q,     // 最小值的量化整数表示
    max_q,     // 最大值的量化整数表示
    0,         // 标志位
    &clamp_op);

  // 使用智能指针管理 QNNPACK 运算符对象的生命周期
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(clamp_op);

  // 断言 QNNPACK Clamp 运算符创建成功
  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK Clamp operator");

  // 创建输出张量，使用和输入张量相同的大小、选项和量化参数
  Tensor qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    input_contig.options(),
    input_contig.q_scale(),
    input_contig.q_zero_point());

  // 设置 QNNPACK Clamp 运算符的输入输出及相关参数
  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_clamp_nc_u8(
    clamp_op,
    input_contig.size(0),                               // 批大小
    (uint8_t*)input_contig.data_ptr<c10::quint8>(),     // 输入数据
    num_elems,                                          // 输入步长
    (uint8_t*)qy.data_ptr<c10::quint8>(),               // 输出数据
    num_elems);                                         // 输出步长

  // 断言 QNNPACK Clamp 运算符设置成功
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK Clamp operator");

  // 获取线程池对象
  pthreadpool_t threadpool = caffe2::pthreadpool_();

  // 执行 QNNPACK Clamp 运算
  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(clamp_op, threadpool);

  // 断言 QNNPACK Clamp 运算执行成功
  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK Clamp operator");

  // 返回量化输出张量
  return qy;
}

#endif // USE_PYTORCH_QNNPACK

// 实现了量化 clamp 操作的函数
Tensor quantized_clamp_impl(
    const Tensor& qx,
    const optional<Scalar>& min,


**继续注释下面的代码段。**
    const optional<Scalar>& max) {

# 定义一个函数，接受一个常量引用的可选标量 `max` 参数
  Tensor qy;
  # 声明一个 Tensor 对象 `qy`

  if (min && max) {
  # 如果 `min` 和 `max` 都有值，执行以下操作
#ifdef USE_PYTORCH_QNNPACK
    // 如果使用 QNNPACK 引擎，并且输入张量的标量类型是 kQUInt8
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
        qx.scalar_type() == kQUInt8) {
      // 调用 qnnpack_clamp 函数进行量化的 clamp 操作
      return qnnpack_clamp(qx, *min, *max);
    }
#endif

    // 使用 qclamp_stub 函数执行张量的 clamp 操作，根据设备类型选择实现
    qclamp_stub(qx.device().type(), qx, *min, *max, qy);
  } else {
#ifdef USE_PYTORCH_QNNPACK
    // 如果使用 QNNPACK 引擎但未提供 min 和 max，抛出异常
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          false, "Both min and max should be specified for quantized clamp!");
    }
#endif
    // 根据是否提供 max 或 min，选择调用 qclamp_max_stub 或 qclamp_min_stub 函数
    if (max) {
      qclamp_max_stub(qx.device().type(), qx, *max, qy);
    } else if (min) {
      qclamp_min_stub(qx.device().type(), qx, *min, qy);
    } else {
      // 如果既未提供 min 也未提供 max，抛出异常
      TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
    }
  }
  // 返回经 clamp 操作后的张量 qy
  return qy;
}
} // namespace

// at::native functions for the native_functions.yaml
// 执行量化 clamp 操作的 CPU 实现
Tensor clamp_quantized_cpu(
    const Tensor& qx,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  Tensor qy;
  // 根据 qx 的数据类型分发到 quantized_clamp_impl 函数执行
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
    qy = quantized_clamp_impl(qx, min, max);
  });
  return qy;
}

// hardtanh is clamp with default min==-1.0f and default max==1.0f
// 执行量化 hardtanh 操作的 CPU 实现，硬切除操作，min 默认为 -1.0f，max 默认为 1.0f
Tensor hardtanh_quantized_cpu(
    const Tensor& qx,
    const Scalar& min,
    const Scalar& max) {
  Tensor qy;
  // 调用 quantized_clamp_impl 函数执行 hardtanh 操作
  qy = quantized_clamp_impl(qx, min, max);
  return qy;
}

// 在已分配的 result 张量上执行 hardtanh 操作的量化 CPU 实现
Tensor& hardtanh_out_quantized_cpu(const Tensor& qx,
    const Scalar& min,
    const Scalar& max,
    Tensor& result) {
  // 调用 quantized_clamp_impl 函数执行 hardtanh 操作
  result = quantized_clamp_impl(qx, min, max);
  return result;
}

// 在原地执行 hardtanh 操作的量化 CPU 实现
Tensor& hardtanh_quantized_cpu_(
    Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  Tensor qy;
  // 调用 quantized_clamp_impl 函数执行 hardtanh 操作
  qy = quantized_clamp_impl(self, min, max);
  // 将操作后的结果复制回 self 张量，可能会在未来的 PR 中进行优化，以提高效率
  self.copy_(qy);
  return self;
}

// 注册 quantized::clamp 函数到 quantized CPU 实现中
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::clamp"), TORCH_FN(clamp_quantized_cpu));
}

} // namespace native
} // namespace at
```