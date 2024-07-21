# `.\pytorch\aten\src\ATen\native\quantized\cpu\qhardswish.cpp`

```py
// 定义编译时仅启用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 PyTorch 的张量和上下文相关的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>

// 引入 PyTorch 的库和量化操作相关的头文件
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>

// 引入 pthreadpool 的头文件用于多线程支持
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

// 根据编译选项选择包含标准函数或者特定操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#endif

// 引入标准库的头文件
#include <algorithm>

// 定义 at 命名空间下的 native 子命名空间
namespace at {
namespace native {

// 定义 qhardswish_stub 的调度器
DEFINE_DISPATCH(qhardswish_stub);

// 匿名命名空间，用于隐藏实现细节
namespace {

// 如果启用了 PyTorch QNNPACK，则定义 qnnpack_hardswish 函数
#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_hardswish(const Tensor& qx, Tensor& qy) {
  // 检查输入张量是否为空
  TORCH_CHECK(qx.ndimension() > 0, "qnnpack_hardswish(): Got empty input tensor");
  // 检查输入张量的数据类型是否为 c10::kQUInt8
  TORCH_CHECK(qx.scalar_type() == c10::kQUInt8,
              "qnnpack_hardswish(): Expected input data type to be ",
              toString(c10::kQUInt8),
              " but got ",
              toString(qx.scalar_type()));
  
  // 初始化 QNNPACK 库
  initQNNPACK();

  // 计算输入张量和输出张量的零点和比例
  size_t num_elems = qx.numel() / qx.size(0);
  const auto i_zero_point = qx.q_zero_point();
  const auto i_scale = qx.q_scale();
  const auto o_zero_point = qy.q_zero_point();
  const auto o_scale = qy.q_scale();

  // 创建 QNNPACK 的 Hardswish 运算符
  pytorch_qnnp_operator_t hardswish_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_hardswish_nc_q8(
    num_elems,                          // 通道数
    i_zero_point,                       // 输入的零点
    i_scale,                            // 输入的比例
    o_zero_point,                       // 输出的零点
    o_scale,                            // 输出的比例
    std::numeric_limits<uint8_t>::min(),// 输出的最小值
    std::numeric_limits<uint8_t>::max(),// 输出的最大值
    0,                                  // 标志位
    &hardswish_op                       // 创建的 Hardswish 运算符指针
  );

  // 使用智能指针管理创建的运算符，确保释放资源
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(hardswish_op);

  // 检查创建运算符的状态是否成功
  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK Hardswish operator");

  // 设置 QNNPACK Hardswish 运算符的输入和输出张量
  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_hardswish_nc_q8(
    hardswish_op,
    qx.size(0),                          // 批量大小
    (uint8_t*)qx.data_ptr<c10::quint8>(),// 输入数据
    num_elems,                          // 输入步长
    (uint8_t*)qy.data_ptr<c10::quint8>(),// 输出数据
    num_elems                            // 输出步长
  );

  // 检查设置运算符的状态是否成功
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK Hardswish operator");

  // 获取 pthread 线程池对象
  pthreadpool_t threadpool = caffe2::pthreadpool_();

  // 执行 QNNPACK Hardswish 运算符
  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(hardswish_op, threadpool);

  // 检查运行运算符的状态是否成功
  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK Hardswish operator");

  // 返回输出张量 qy
  return qy;
}
#endif // USE_PYTORCH_QNNPACK

} // namespace

// 定义静态函数 quantized_hardswish
static Tensor quantized_hardswish(const Tensor& qx, double output_scale, int64_t output_zero_point) {
  // 创建一个新的量化张量 qy，形状和 qx 相同
  Tensor qy = at::_empty_affine_quantized(
      qx.sizes(),                                 // 张量的形状
      at::device(kCPU).dtype(qx.scalar_type()),   // CPU 设备上的数据类型
      output_scale,                               // 输出的比例
      output_zero_point,                          // 输出的零点
      qx.suggest_memory_format()                  // 建议的内存格式
  );
#ifdef USE_PYTORCH_QNNPACK
  // 如果使用了 QNNPACK 引擎，并且输入张量 qx 的数据类型是 kQUInt8
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    // 使得输入张量 qx 连续化，并且尝试使用推荐的内存格式
    Tensor qx_contig = qx.contiguous(qx.suggest_memory_format());
    // 调用 QNNPACK 的硬切线激活函数，并将结果保存在 qy 中
    qnnpack_hardswish(qx_contig, qy);
    // 返回计算结果张量 qy
    return qy;
  }
#endif  // USE_PYTORCH_QNNPACK

// 如果未使用 QNNPACK 或者输入数据类型不是 kQUInt8，则调用默认的硬切线激活函数
qhardswish_stub(qx.device().type(), qx, qy);
// 返回计算结果张量 qy
return qy;
}

// 注册量化 CPU 版本的硬切线激活函数到 Torch 库中
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // 将 "quantized::hardswish" 对应的实现设置为 quantized_hardswish 函数
  m.impl(TORCH_SELECTIVE_NAME("quantized::hardswish"), TORCH_FN(quantized_hardswish));
}

}}  // namespace at::native
```