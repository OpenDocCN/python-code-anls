# `.\pytorch\aten\src\ATen\native\quantized\cpu\qhardsigmoid.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
// 包含 ATen 核心的 Tensor 头文件

#include <ATen/Context.h>
// 包含 ATen 的 Context 头文件

#include <ATen/native/quantized/cpu/QuantizedOps.h>
// 包含 ATen 量化操作的 CPU 实现头文件

#include <ATen/native/quantized/cpu/init_qnnpack.h>
// 包含 ATen 量化操作的 QNNPACK 初始化头文件

#include <ATen/native/quantized/cpu/QnnpackUtils.h>
// 包含 ATen 量化操作的 QNNPACK 实用工具头文件

#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
// 包含 caffe2 的线程池实现头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS，包含 ATen 的函数头文件

#include <ATen/NativeFunctions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS，包含 ATen 的本地函数头文件

#else
#include <ATen/ops/_empty_affine_quantized.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，包含 ATen 的空量化仿射操作头文件

#include <ATen/ops/hardsigmoid_native.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，包含 ATen 的硬 Sigmoid 实现头文件

#endif

#include <algorithm>
// 包含 STL 算法头文件

namespace at {
namespace native {

DEFINE_DISPATCH(qhardsigmoid_stub);
// 定义 qhardsigmoid_stub 的分发器函数

namespace {

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_hardsigmoid(Tensor input) {
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_hardsigmoid(): Got empty input tensor");
  // 检查输入张量维度是否大于 0

  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
                "qnnpack_hardsigmoid(): Expected input data type ",
                toString(c10::kQUInt8),
                " but got ",
                toString(input.scalar_type()));
  // 检查输入张量的数据类型是否为 c10::kQUInt8

  initQNNPACK();
  // 初始化 QNNPACK 库

  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  // 将输入张量转换为连续的内存布局

  size_t num_elems = input_contig.numel() / input_contig.size(0);
  // 计算每个通道的元素数量

  const auto i_zero_point = input_contig.q_zero_point();
  // 获取输入张量的量化零点

  const auto i_scale = input_contig.q_scale();
  // 获取输入张量的量化比例

  constexpr float o_scale = 1.0f / 256.0f;
  constexpr int32_t o_zero_point = 0;
  // 设置输出张量的量化参数

  pytorch_qnnp_operator_t hardsigmoid_op{nullptr};
  // 声明 QNNPACK 硬 Sigmoid 运算符的指针

  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_hardsigmoid_nc_q8(
    num_elems, // channels
    i_zero_point,
    i_scale,
    o_zero_point,
    o_scale,
    std::numeric_limits<uint8_t>::min(), // output min
    std::numeric_limits<uint8_t>::max(), // output max
    0, // flags
    &hardsigmoid_op);
  // 创建 QNNPACK 硬 Sigmoid 运算符

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(hardsigmoid_op);
  // 使用智能指针管理 QNNPACK 运算符的生命周期

  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK Hardsigmoid operator");
  // 内部断言，确保成功创建 QNNPACK 硬 Sigmoid 运算符

  Tensor qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    at::device(kCPU).dtype(input_contig.dtype()),
    o_scale,
    o_zero_point,
    input_contig.suggest_memory_format());
  // 创建空的仿射量化张量 qy

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_hardsigmoid_nc_q8(
    hardsigmoid_op,
    input_contig.size(0), // batch size
    (uint8_t*)input_contig.data_ptr<c10::quint8>(), // input data
    num_elems, // input stride
    (uint8_t*)qy.data_ptr<c10::quint8>(), // output data
    num_elems); // output stride
  // 设置 QNNPACK 硬 Sigmoid 运算符

  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK Hardsigmoid operator");
  // 内部断言，确保成功设置 QNNPACK 硬 Sigmoid 运算符

  pthreadpool_t threadpool = caffe2::pthreadpool_();
  // 获取 pthread 线程池

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(hardsigmoid_op, threadpool);
  // 运行 QNNPACK 硬 Sigmoid 运算符

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK Hardsigmoid operator");
  // 内部断言，确保成功运行 QNNPACK 硬 Sigmoid 运算符

  return qy;
  // 返回仿射量化后的输出张量 qy
}
#endif // USE_PYTORCH_QNNPACK

} // namespace
// 匿名命名空间结束
// 定义一个函数，用于在 CPU 上执行硬 sigmoid 函数的量化版本，输入参数为 qx 张量
Tensor hardsigmoid_quantized_cpu(const Tensor& qx) {
#ifdef USE_PYTORCH_QNNPACK
  // 检查是否启用了 QNNPACK 加速库，并且输入张量 qx 的数据类型是 kQUInt8
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    // 若条件成立，调用 QNNPACK 的硬 sigmoid 函数，并返回结果
    return qnnpack_hardsigmoid(qx);
  }
#endif  // USE_PYTORCH_QNNPACK

  // 如果不满足 QNNPACK 加速条件或者输入数据类型不是 kQUInt8，则调用 qhardsigmoid_stub 函数处理 qx，并将结果存储在新的张量 qy 中
  Tensor qy;
  qhardsigmoid_stub(qx.device().type(), qx, qy);
  // 返回处理后的结果张量 qy
  return qy;
}

// 定义一个函数，在 CPU 上执行硬 sigmoid 函数的量化版本，并将结果存储在指定的 result 张量中
Tensor& hardsigmoid_out_quantized_cpu(const Tensor& qx, Tensor& result) {
  // 注意：我们创建一个新的临时张量，因为硬 sigmoid 函数的输出通常具有不同的量化参数，而当前只支持整个张量或张量通道的量化
  // 调用 hardsigmoid_quantized_cpu 函数处理输入张量 qx，返回处理后的结果存储在 qy 中
  Tensor qy = hardsigmoid_quantized_cpu(qx);
  // 将 qy 的内容复制到预先提供的结果张量 result 中
  result.copy_(qy);
  // 返回结果张量 result 的引用
  return result;
}
```