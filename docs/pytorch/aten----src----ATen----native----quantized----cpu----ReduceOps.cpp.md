# `.\pytorch\aten\src\ATen\native\quantized\cpu\ReduceOps.cpp`

```py
// 定义宏以启用仅在方法操作符中的 Torch 断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含必要的头文件以实现量化相关功能和操作
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

// 根据条件选择包含的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>         // 用于 _empty_affine_quantized
#include <ATen/ops/mean.h>                            // 用于 mean
#include <ATen/ops/mean_native.h>                     // 用于 mean_out_quantized
#include <ATen/ops/quantize_per_tensor.h>             // 用于 quantize_per_tensor
#include <ATen/ops/std.h>                             // 用于 std
#include <ATen/ops/std_native.h>                      // 用于 std_native
#include <ATen/ops/zeros_like_ops.h>                  // 用于 zeros_like
#endif

// 命名空间 at::native 开始
namespace at {
namespace native {

// 定义用于分发的函数调度器，用于量化操作中的均值和标准差计算
DEFINE_DISPATCH(qmean_inner_dim_stub);
DEFINE_DISPATCH(qstd_inner_dim_stub);

// 内联函数：检查是否在最内层维度上进行操作
inline bool is_innnermost_dim(
    const Tensor& self,
    OptionalIntArrayRef opt_dim) {
  // 如果未提供维度信息，视为操作在最内层维度上进行
  if (!opt_dim.has_value()) {
    return true;
  }
  auto dims = opt_dim.value().vec();  // 获取维度信息
  auto ndim = self.dim();             // 获取张量的维度数
  maybe_wrap_dims(dims, ndim);        // 根据情况调整维度信息
  std::sort(dims.begin(), dims.end(), std::greater<int64_t>());  // 对维度信息进行降序排序
  bool is_innermost = dims.empty() || dims[0] == ndim - 1;       // 检查是否在最内层维度上操作
  for (size_t i = 1; i < dims.size(); ++i) {
    is_innermost = is_innermost && (dims[i] == dims[i-1] - 1);   // 迭代检查每个维度是否连续
  }
  return is_innermost;  // 返回是否在最内层维度上操作的结果
}

// 内联函数：检查是否可以使用快速路径计算均值在最内层维度上
inline bool is_mean_inner_dim_fast_path(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    std::optional<ScalarType> opt_dtype) {
  // 快速路径条件：操作在最内层维度上且数据类型匹配（如果指定了数据类型）
  bool is_fast_path =
      is_innnermost_dim(self, opt_dim) &&
      (!opt_dtype.has_value() || opt_dtype.value() == self.scalar_type());
  return is_fast_path;  // 返回是否可以使用快速路径计算均值的结果
}

#ifdef USE_PYTORCH_QNNPACK
static Tensor qnnpack_mean(const Tensor& input, IntArrayRef dim, bool keepdim) {
  Tensor output;  // 定义输出张量

  TORCH_CHECK(
      input.ndimension() == 4,
      "qnnpack_global_average_pool: Expected input to be 4-dimensional: got ",
      input.ndimension());  // 检查输入张量是否为四维

  TORCH_CHECK(
      dim.size() == 2,
      "qnnpack_global_average_pool: dim size must be a tuple of two ints");  // 检查维度参数是否为长度为2的元组

  TORCH_CHECK(
      dim[0] == 2 && dim[1] == 3,
      "qnnpack_global_average_pool: Reduction dimensions must match last 2 dimensions of input tensor");  // 检查维度参数是否为(2, 3)，适用于NCHW格式

  const int64_t batch_size = input.size(0);  // 获取批次大小
  const int64_t inC = input.size(1);  // 获取输入通道数
  const int64_t inH = input.size(2);  // 获取输入高度
  const int64_t inW = input.size(3);  // 获取输入宽度

  Tensor input_contig = input.contiguous(MemoryFormat::ChannelsLast);  // 将输入张量转换为ChannelsLast格式的连续张量

  initQNNPACK();  // 初始化QNNPACK库

  const auto scale = input_contig.q_scale();  // 获取输入张量的量化比例
  const auto zero_point = input_contig.q_zero_point();  // 获取输入张量的量化零点
  const auto outC = inC;  // 输出通道数等于输入通道数

  // 创建一个新的量化输出张量
  output = at::_empty_affine_quantized(
      keepdim ? IntArrayRef{batch_size, outC, 1, 1}
              : IntArrayRef{batch_size, outC},
      at::device(kCPU).dtype(kQUInt8),
      scale,
      zero_point);

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};  // 定义QNNPACK操作符的指针

  // 创建QNNPACK全局平均池化操作符
  const pytorch_qnnp_status createStatus =
      pytorch_qnnp_create_global_average_pooling_nwc_q8(
          inC,
          zero_point,
          scale,
          zero_point,
          scale,
          std::numeric_limits<uint8_t>::min() /* output min */,
          std::numeric_limits<uint8_t>::max() /* output max */,
          0,
          &qnnpack_operator);

  CAFFE_ENFORCE(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK Global Average Pooling operator");  // 强制执行，检查QNNPACK操作符创建是否成功

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);  // 使用智能指针管理QNNPACK操作符的生命周期

  // 设置QNNPACK全局平均池化操作符的参数和输入输出数据
  const pytorch_qnnp_status setupStatus =
      pytorch_qnnp_setup_global_average_pooling_nwc_q8(
          qnnpack_operator,
          batch_size,
          inH * inW,
          (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
          inC,
          (uint8_t*)output.data_ptr<c10::quint8>() /* output data */,
          outC);

  CAFFE_ENFORCE(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Global Average Pooling operator");  // 强制执行，检查QNNPACK操作符设置是否成功

  pthreadpool_t threadpool = caffe2::pthreadpool_();  // 获取线程池对象
  // 运行QNNPACK全局平均池化操作符
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Global Average Pool operator");  // 内部断言，检查QNNPACK操作符运行是否成功

  return output;  // 返回输出张量
}
#endif

Tensor& mean_out_quantized_cpu(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    std::optional<ScalarType> opt_dtype,
    Tensor& result) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      self.scalar_type() == kQUInt8 && opt_dim.has_value()) {
    auto dim = opt_dim.value();  // 获取维度参数
    // QNNPACK当前仅支持NCHW格式和dim=(2, 3)的情况
    // 如果输入张量是四维的，并且维度参数是二维的，并且维度的大小分别为2和3
    // 则调用 qnnpack_mean 函数进行计算，这是一个特定版本的处理方式
    if (self.ndimension() == 4 && dim.size() == 2 && dim[0] == 2 && dim[1] == 3) {
      result = qnnpack_mean(self, dim, keepdim);
      return result;
    }
  }
#endif

// 在最内部维度上取平均值
if (self.is_contiguous(c10::MemoryFormat::Contiguous) &&
    is_mean_inner_dim_fast_path(self, opt_dim, opt_dtype)) {
  // 调用优化的平均值计算内核函数
  qmean_inner_dim_stub(self.device().type(), self, opt_dim, keepdim, opt_dtype, result);
  // 返回计算结果
  return result;
}

// 对输入进行去量化操作
auto self_dequantized = self.dequantize();
// 计算去量化后的平均值
auto result_dequantized = at::mean(self_dequantized, opt_dim, keepdim, opt_dtype);
// 对平均值结果进行量化操作
result = at::quantize_per_tensor(
    result_dequantized,
    self.q_scale(),
    self.q_zero_point(),
    opt_dtype.value_or(self.scalar_type()));
// 返回量化后的结果
return result;
}

// 计算量化后的平均值（CPU 版本）
Tensor mean_quantized_cpu(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  // 定义结果张量
  Tensor result;
  // 调用量化后的平均值计算函数
  mean_out_quantized_cpu(self, opt_dim, keepdim, dtype, result);
  // 返回计算结果
  return result;
}

// qstd
inline bool is_std_inner_dim_fast_path(
    const Tensor& self,
    OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction) {
  // 如果元素数量太少，不进入快速路径
  IntArrayRef dims = dim.has_value() ? dim.value() : IntArrayRef();
  // 获取所有维度
  auto all_dims = std::vector<int64_t>(self.dim());
  std::iota(all_dims.begin(), all_dims.end(), 0);
  // 如果未指定维度，则使用所有维度
  dims = dims.empty() ? all_dims : dims;
  // 检查是否需要修正
  bool has_correction = !correction.value_or(1).equal(0);
  int64_t num_ele = 1;
  // 计算指定维度下的元素总数
  for (auto d : dims) {
    num_ele *= self.size(d);
  }
  // 如果只有一个元素且需要修正，则不使用快速路径
  if (num_ele == 1 && has_correction) {
    return false;
  }
  // 检查是否是最内部的维度
  return is_innnermost_dim(self, dims);
}

// 计算量化后的标准差（CPU 版本），将结果写入已分配的结果张量
Tensor& std_out_quantized_cpu(
    const Tensor& self,
    OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result) {
  // 快速路径
  if (self.is_contiguous(c10::MemoryFormat::Contiguous) &&
      is_std_inner_dim_fast_path(self, dim, correction)) {
    // 调用优化的标准差计算内核函数
    qstd_inner_dim_stub(self.device().type(), self, dim, correction, keepdim, result);
    // 返回计算结果
    return result;
  }

  // 参考路径
  // 对输入进行去量化操作
  auto self_dequantized = self.dequantize();
  // 计算去量化后的标准差
  auto result_dequantized = at::std(self_dequantized, dim, correction, keepdim);
  // 对标准差结果进行量化操作
  result = at::quantize_per_tensor(
      result_dequantized,
      self.q_scale(),
      self.q_zero_point(),
      self.scalar_type());
  // 返回量化后的结果
  return result;
}

// 计算量化后的标准差（CPU 版本）
Tensor std_quantized_cpu(
    const Tensor& self,
    OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  // 定义结果张量
  Tensor result;
  // 调用量化后的标准差计算函数
  std_out_quantized_cpu(self, dim, correction, keepdim, result);
  // 返回计算结果
  return result;
}

} // namespace native
} // namespace at
```