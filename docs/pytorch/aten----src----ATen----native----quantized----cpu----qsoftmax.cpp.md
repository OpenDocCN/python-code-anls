# `.\pytorch\aten\src\ATen\native\quantized\cpu\qsoftmax.cpp`

```
// 包含 PyTorch 的 ATen 头文件和库声明
#include <ATen/ATen.h>
#include <torch/library.h>

#ifdef USE_PYTORCH_QNNPACK
// 如果使用 QNNPACK，则包含相关的头文件
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <pytorch_qnnpack.h>

#include <utility>
#endif // USE_PYTORCH_QNNPACK

namespace at {
namespace native {

namespace {

#ifdef USE_PYTORCH_QNNPACK
// 在使用 QNNPACK 的情况下，定义 QNNPACK softmax 的输出规模和零点
const static float qnnpack_softmax_output_scale = 0x1.0p-8f;
const static int qnnpack_softmax_output_zero_point = 0;

// 检查是否可以使用 QNNPACK 进行计算
bool is_qnnpack_compatible(
    const Tensor& qx,
    const double output_scale,
    const int64_t output_zero_point) {
  return (
      (qx.qscheme() == kPerTensorAffine ||
       qx.qscheme() == kPerTensorSymmetric) &&
      qx.scalar_type() == c10::kQUInt8 && qx.ndimension() > 0 &&
      output_scale == qnnpack_softmax_output_scale &&
      output_zero_point == qnnpack_softmax_output_zero_point);
}

// 使用 QNNPACK 实现的 softmax 函数
Tensor qsoftmax_qnnpack(const Tensor& qx, const int64_t dim) {
  /*
    检查 qx 张量在不同维度上的连续性：
    1) 如果目标维度的步长为 1，则直接使用 qx
    2) 如果 dim 是最后一个维度但 qx 不是连续的，则使用 qx.contiguous()
    3) 其他情况，需要对 qx 进行维度置换和连续化处理
   */

  const int64_t last_dim = qx.dim() - 1;
  std::optional<std::vector<int64_t>> permuted_dims = c10::nullopt;
  std::optional<at::Tensor> qx_contig = c10::nullopt;
  const at::Tensor* qx_contig_ptr = nullptr;

  if (qx.stride(dim) == 1) {
    qx_contig_ptr = &qx;
  } else if (dim == last_dim) {
    qx_contig = qx.contiguous();
    qx_contig_ptr = &qx_contig.value();
  } else {
    permuted_dims = std::vector<int64_t>(qx.dim());
    // 创建一个包含所有维度索引的向量，用于执行维度置换
    std::iota(permuted_dims->begin(), permuted_dims->end(), 0);
    permuted_dims->at(last_dim) = dim;
    permuted_dims->at(dim) = last_dim;
    // 对 qx 执行维度置换和连续化处理
    qx_contig = qx.permute(permuted_dims.value()).contiguous();
    // 将指针 qx_contig_ptr 指向 qx_contig 的值
    qx_contig_ptr = &qx_contig.value();
  }

  // 创建一个空的仿射量化张量 qy
  at::Tensor qy = at::_empty_affine_quantized(
      qx_contig_ptr->sizes(),  // 使用 qx_contig 的尺寸创建张量
      at::device(kCPU)  // 指定张量在 CPU 上
          .dtype(qx.scalar_type())  // 指定张量的数据类型与 qx 相同
          .memory_format(qx_contig_ptr->suggest_memory_format()),  // 指定张量的内存格式
      qnnpack_softmax_output_scale,  // 设置量化 softmax 输出的比例因子
      qnnpack_softmax_output_zero_point,  // 设置量化 softmax 输出的零点
      c10::nullopt);  // 指定没有特殊选项

  // 获取 qx 张量在指定维度 dim 上的大小
  const size_t channels = qx.size(dim);
  // 获取 qx 张量的量化比例因子，并转换为 float 类型
  const float input_scale = static_cast<float>(qx.q_scale());
  // 设置标志位为 0
  const uint32_t flags = 0;
  // 计算批次大小，即 qx 张量中元素数量除以通道数
  const size_t batch_size = qx.numel() / channels;
  // 将 qx_contig_ptr 强制转换为 quint8 数据类型的指针，并赋给 input
  const uint8_t* input =
      reinterpret_cast<const uint8_t*>(qx_contig_ptr->data_ptr<c10::quint8>());
  // 设置输入的步长为通道数
  const size_t input_stride = channels;
  // 将 qy 的数据指针强制转换为 quint8 数据类型的指针，并赋给 output
  uint8_t* output = reinterpret_cast<uint8_t*>(qy.data_ptr<c10::quint8>());
  // 设置输出的步长为通道数
  const size_t output_stride = channels;

  // 初始化 QNNPACK 库
  initQNNPACK();
  // 定义 QNNPACK Softmax 运算符的指针 softargmax，并初始化为 nullptr
  pytorch_qnnp_operator_t softargmax = nullptr;

  // 创建 QNNPACK Softmax 运算符并返回状态
  pytorch_qnnp_status status = pytorch_qnnp_create_softargmax_nc_q8(
      channels,
      input_scale,
      qnnpack_softmax_output_zero_point,
      qnnpack_softmax_output_scale,
      flags,
      &softargmax);
  // 检查创建 Softmax 运算符的状态，确保成功
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,
      "failed to create QNNPACK Softmax operator");
  TORCH_CHECK_NOTNULL(softargmax);

  // 使用 std::unique_ptr 管理 softargmax，确保在退出作用域时自动释放
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter> softmax_op(
    softargmax);

  // 配置 Softmax 运算符的输入、输出数据以及相关参数
  status = pytorch_qnnp_setup_softargmax_nc_q8(
      softargmax, batch_size, input, input_stride, output, output_stride);
  // 检查配置 Softmax 运算符的状态，确保成功
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Softmax operator");

  // 获取 pthread 线程池对象
  pthreadpool_t threadpool = caffe2::pthreadpool_();
  // 运行 QNNPACK Softmax 运算符，并返回运行状态
  status = pytorch_qnnp_run_operator(softargmax, threadpool);
  // 检查运行 Softmax 运算符的状态，确保成功
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,
      "failed to run QNNPACK Softmax operator");

  // 如果 permuted_dims 有值，则对 qy 进行维度重排，否则直接返回 qy
  return permuted_dims.has_value() ? qy.permute(permuted_dims.value()) : std::move(qy);
#endif // USE_PYTORCH_QNNPACK



Tensor qsoftmax_naive(
    const Tensor& qx,
    const int64_t dim,
    const double output_scale,
    const int64_t output_zero_point) {
  // 将量化的输入张量qx反量化为浮点数张量rx
  Tensor rx = at::dequantize(qx);
  // 对rx沿指定维度dim进行softmax操作，得到ry
  Tensor ry = at::softmax(rx, dim);
  // 将ry重新量化为与qx相同的量化张量，并使用指定的缩放因子和零点偏移量
  return at::quantize_per_tensor(
      ry, output_scale, output_zero_point, qx.scalar_type());
}

Tensor qsoftmax(
    const Tensor& qx,
    const int64_t dim,
    const double output_scale,
    const int64_t output_zero_point) {
#ifdef USE_PYTORCH_QNNPACK
  // 如果使用QNNPACK引擎且qx与输出参数兼容，则调用QNNPACK的qsoftmax实现
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      is_qnnpack_compatible(qx, output_scale, output_zero_point)) {
    return qsoftmax_qnnpack(qx, dim);
  }
#endif // USE_PYTORCH_QNNPACK
  // 否则调用纯Python实现的qsoftmax_naive函数
  return qsoftmax_naive(qx, dim, output_scale, output_zero_point);
}

// 定义quantized命名空间下的softmax实现，并注册到QuantizedCPU设备
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::softmax"), TORCH_FN(qsoftmax));
}

} // namespace

} // namespace native
} // namespace at


这些注释为给定的C++代码片段提供了详细的解释，涵盖了每行代码的作用和上下文。
```