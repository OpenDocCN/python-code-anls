# `.\pytorch\aten\src\ATen\native\quantized\QTensor.cpp`

```py
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

#include <c10/util/irange.h>

#include <cmath>
#include <utility>

namespace at {
namespace native {

// 对输入张量进行动态整数或无符号整数或半精度浮点数量化
Tensor quantize_per_tensor_dynamic(
    const Tensor& self,
    ScalarType dtype,
    bool reduce_range) {
  TORCH_CHECK( (dtype == ScalarType::QInt8 || dtype == ScalarType::QUInt8 || dtype == ScalarType::Half), "dtype ", dtype, "not supported");
  // 使输入张量连续存储
  auto input_contig = self.contiguous();
  // 如果 dtype 是半精度浮点数，则直接返回张量转换为半精度浮点数的结果
  if (dtype == ScalarType::Half) {
    return input_contig.to(ScalarType::Half);
  }
  // 计算输入张量的最小值和最大值
  float x_min = input_contig.min().item<float>();
  float x_max = input_contig.max().item<float>();

  // 如果需要缩小范围且使用 QNNPACK 引擎，则将 reduce_range 设为 false
  if (reduce_range && at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    reduce_range = false;
  }

  int qmin;
  int qmax;

  // 根据 dtype 设置量化的最小值和最大值
  if (dtype == ScalarType::QInt8) {
    qmin = -128;
    qmax = 127;
  } else {
    // 目前仅支持 ScalarType::QUInt8，未来支持其他 dtype 的量化时会扩展这个分支
    qmin = 0;
    qmax = 255;
  }

  // 根据输入的最小值、最大值和量化的最小最大值，选择量化参数
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/qmin,
      /*qmax=*/qmax,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  // 使用选定的量化参数对输入张量进行整数或无符号整数量化，并返回结果
  return at::native::quantize_per_tensor(self, q_params.scale, q_params.zero_point, dtype);
}

// 对输入张量进行指定比例和零点的整数量化
Tensor quantize_per_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  // 创建一个指定比例和零点的量化器
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  // 使用量化器对输入张量进行量化，并返回结果
  return quantizer->quantize(self);
}

// 对输入张量使用张量形式的比例和零点进行整数量化
Tensor quantize_per_tensor_tensor_qparams(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    ScalarType dtype) {
  // 创建一个使用张量形式的比例和零点的量化器
  auto quantizer = make_per_tensor_affine_quantizer(scale.item().toDouble(), zero_point.item().toLong(), dtype);
  // 使用量化器对输入张量进行量化，并返回结果
  return quantizer->quantize(self);
}

// 对张量列表进行批量整数量化（CPU 版本）
std::vector<Tensor> quantize_per_tensor_list_cpu(
    TensorList tensors,
    const Tensor& scales,
    const Tensor& zero_points,
    ScalarType dtype) {
  // 创建一个空的张量向量，用于存储量化后的张量
  std::vector<Tensor> quantized_tensors;
  // 遍历输入张量列表
  for (const auto i : c10::irange(tensors.size())) {
    // 将每个张量按照对应的比例和零点进行整数量化，并加入结果向量中
    quantized_tensors.push_back(at::quantize_per_tensor(
        tensors[i],
        scales[i].item<double>(),
        zero_points[i].item<int64_t>(),
        dtype));
  }
  // 返回量化后的张量向量
  return quantized_tensors;
}

// 对输入张量进行通道级别的整数量化
Tensor quantize_per_channel(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  // 创建一个使用通道级别比例和零点的量化器
  auto quantizer = make_per_channel_affine_quantizer(scales, zero_points, axis, dtype);
  // 使用量化器对输入张量进行量化，并返回结果
  return quantizer->quantize(self);
}

// 将输入张量从量化表示还原为浮点数表示（CPU 或 CUDA 版本）
Tensor dequantize_cpu_or_cuda(const Tensor& self) {
  // 将输入张量转换为 float 类型，即还原为浮点数表示
  return self.to(at::kFloat);
}

} // namespace native
} // namespace at
// 使用量化张量的量化器对象，将张量反量化并返回
Tensor dequantize_quantized(const Tensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

// 对输入的张量列表进行逐个反量化操作，返回反量化后的张量列表
std::vector<Tensor> dequantize_tensors_quantized_cpu(TensorList tensors) {
  std::vector<Tensor> dequantized_tensors;
  for (const auto & tensor : tensors) {
    dequantized_tensors.push_back(tensor.dequantize());
  }
  return dequantized_tensors;
}

// 获取张量的量化比例尺度
double q_scale_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
  return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->scale();
}

// 获取张量的量化零点
int64_t q_zero_point_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
  return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->zero_point();
}

// 获取张量每通道的量化比例尺度
Tensor q_per_channel_scales(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->scales();
}

// 获取张量每通道的量化零点
Tensor q_per_channel_zero_points(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->zero_points();
}

// 获取张量每通道量化的轴
int64_t q_per_channel_axis(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->axis();
}

// 创建一个按通道量化的张量
Tensor make_per_channel_quantized_tensor_cpu(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  Tensor dst = at::_empty_per_channel_affine_quantized(
      self.sizes(),
      scales,
      zero_points,
      axis,
      self.options().dtype(toQIntType(self.scalar_type())));
  Tensor self_contig = self.contiguous();
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "per_channel_affine_qtensor", [&]() {
        underlying_t* self_data = self_contig.data_ptr<underlying_t>();
        underlying_t* dst_data =
            reinterpret_cast<underlying_t*>(dst.data_ptr<scalar_t>());
        if (self.numel() > 0) {
          memcpy(dst_data, self_data, self.nbytes());
        }
      });
  return dst;
}

// 设置量化张量的存储
Tensor& set_storage_quantized_(
    Tensor& self,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef sizes,
    IntArrayRef strides) {
  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_storage_keep_dtype(std::move(storage));
  self_->set_storage_offset(storage_offset);
  self_->set_sizes_and_strides(sizes, strides);
  return self;
}

// 获取张量的量化方案
QScheme qscheme_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  return quantizer->qscheme();
}
// 克隆量化张量 `self`，并返回克隆的张量
Tensor quantized_clone(
    const Tensor& self,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 如果未指定内存格式，则默认为连续内存
  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);

  // 如果内存格式为 MemoryFormat::Preserve，则根据建议的内存格式重新设置
  if (memory_format == MemoryFormat::Preserve) {
    memory_format = self.suggest_memory_format();
  }

  // 创建一个新的张量 `dst`
  Tensor dst;
  // 如果 `self` 的量化方案为 PerTensorAffine
  if (self.qscheme() == at::kPerTensorAffine) {
    // 调用 _empty_affine_quantized 函数创建新的量化张量 `dst`
    dst = at::_empty_affine_quantized(
        self.sizes(),
        self.options().memory_format(memory_format),
        self.q_scale(),
        self.q_zero_point(),
        c10::nullopt);
  } 
  // 如果 `self` 的量化方案为 PerChannelAffine
  else if (self.qscheme() == at::kPerChannelAffine) {
    // 调用 _empty_per_channel_affine_quantized 函数创建新的量化张量 `dst`
    dst = at::_empty_per_channel_affine_quantized(
        self.sizes(),
        self.q_per_channel_scales(),
        self.q_per_channel_zero_points(),
        self.q_per_channel_axis(),
        self.options().memory_format(memory_format),
        c10::nullopt);
  } 
  // 如果量化方案既不是 PerTensorAffine 也不是 PerChannelAffine，抛出错误
  else {
    TORCH_CHECK(false, "clone for quantized Tensor only works for \
      PerTensorAffine and PerChannelAffine qscheme right now");
  }

  // 将 `self` 的数据拷贝到 `dst` 中
  at::native::copy_(dst, self, false);

  // 返回新的量化张量 `dst`
  return dst;
}

// 比较两个 CPU 上的量化张量 `self` 和 `other` 是否相等
bool equal_quantized_cpu(const Tensor& self, const Tensor& other) {
  // 检查张量 `self` 和 `other` 是否均为量化张量且位于 CPU 上
  TORCH_CHECK(
      self.device().type() == kCPU && other.device().type() == kCPU,
      "quantized_equal is implemented only for the QuantizedCPU backend");
  if (!self.is_quantized() || !other.is_quantized()) {
    return false;
  }

  // 委托给虚函数 `equalTo` 进行具体的比较逻辑，以确保不同的量化器可以有特定的比较方法
  auto self_quantizer = get_qtensorimpl(self)->quantizer();
  auto other_quantizer = get_qtensorimpl(other)->quantizer();
  if (!self_quantizer->equalTo(other_quantizer)) {
    return false;
  }

  // 检查张量的大小和元素类型是否相同
  if (self.sizes() != other.sizes()) {
    return false;
  }
  if (self.scalar_type() != other.scalar_type()) {
    return false;
  }

  // 比较张量的数据是否完全一致
  auto self_contig = self.contiguous();
  auto other_contig = other.contiguous();

  // 获取张量数据的指针和大小，进行比较
  void* self_data = self_contig.data_ptr();
  void* other_data = other_contig.data_ptr();
  auto data_size = self.numel() * self.element_size();

  // 对于特定的量化类型（如 kQUInt4x2 或 kQUInt2x4），每个字节可能存储多个元素，需要特殊处理数据大小
  if (self.scalar_type() == kQUInt4x2 || self.scalar_type() == kQUInt2x4) {
      TORCH_INTERNAL_ASSERT(self.element_size() == 1);
      data_size = (data_size>>1) + (data_size&1);
  }

  // 使用 memcmp 函数比较数据是否相等
  return 0 == memcmp(self_data, other_data, data_size);
}

// 计算激活张量的量化参数
std::tuple<double, int64_t> _choose_qparams_per_tensor(
    const Tensor& self,
    ```
    bool reduce_range) {
  // 声明一个Tensor变量a，用于存储计算结果
  at::Tensor a;
  // 将输入张量self进行连续化处理，并存储在input_contig中
  auto input_contig = self.contiguous();
  // 计算连续化处理后张量的最小值，并转换为float类型存储在x_min中
  float x_min = input_contig.min().item<float>();
  // 计算连续化处理后张量的最大值，并转换为float类型存储在x_max中
  float x_max = input_contig.max().item<float>();

  // 如果reduce_range为true并且当前的量化引擎为QNNPACK，则将reduce_range设为false
  if (reduce_range && at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    reduce_range = false;
  }

  // 调用quant_utils::ChooseQuantizationParams函数，选择量化参数
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/255,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  // 返回量化参数中的scale和zero_point作为tuple的元素
  return std::make_tuple(q_params.scale, q_params.zero_point);
}
}

// 计算量化损失的静态函数，输入为浮点型指针、元素数量、最小值、最大值、量化输入、位宽
static float calculate_quant_loss(
    const float* input,
    int numel,
    float xmin,
    float xmax,
    float* q_input,
    int bit_width) {
  // 将最小值强制转换为半精度浮点数
  xmin = static_cast<at::Half>(xmin);
  // 计算数据范围
  float data_range = xmax - xmin;
  // 计算量化最大值
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  float qmax = (1 << bit_width) - 1;
  // 计算比例尺度
  float scale = data_range == 0
      ? 1.0
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      : static_cast<float>(static_cast<at::Half>(data_range / qmax));
  // 计算逆比例尺度
  float inverse_scale = scale == 0 ? 1.0f : 1.0f / scale;

  // 初始化范数
  float norm = 0.0f;
  int i = 0;

  // 循环处理剩余元素
  // TODO 添加 FBGEMM 内核
  // #ifdef USE_FBGEMM
  // #endif

  // 剩余循环
  for (; i < numel; i++) {
    // 进行量化输入的计算
    q_input[i] = std::max(
        0.0f, std::min<float>(std::nearbyint((input[i] - xmin) * inverse_scale), qmax));
    // 量化结果反向映射到原始值域
    q_input[i] = q_input[i] * scale + xmin;
    // 更新范数
    norm += (input[i] - q_input[i]) * (input[i] - q_input[i]);
  }
  // 返回范数的平方根作为量化损失
  return std::sqrt(norm);
}

/*
  选择优化的量化参数的辅助函数，用于计算张量的最佳最小值和最大值
  使用贪婪方法调整最小值和最大值，并计算L2范数，试图最小化量化误差
  返回张量的优化最大值和最小值
*/
std::tuple<Tensor, Tensor> choose_qparams_optimized(
    const at::Tensor& input_tensor,
    int64_t numel,
    const int64_t n_bins,
    const double ratio,
    int64_t bit_width) {

  if (numel < 0 || numel > input_tensor.numel()) {
    // 检查元素数量是否超出输入张量的范围
    TORCH_CHECK(false, "numel is out of the bound of input tensor");
  }

  // 检查元素数量是否不超过输入张量的总元素数
  TORCH_CHECK(numel <= input_tensor.numel(), "numel ", numel,
      " greater than input_tensor.numel() ", input_tensor.numel());
  // 获取输入张量中的最小值和最大值
  const float* input_row = input_tensor.const_data_ptr<float>();
  float xmin = *std::min_element(input_row, input_row + numel);
  float xmax = *std::max_element(input_row, input_row + numel);

  // 计算步长
  float stepsize = (xmax - xmin) / n_bins;
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  int min_bins = n_bins * (1.0 - (float) ratio);
  // 确保张量是连续的
  Tensor input_tensor_contig = input_tensor.contiguous();
  // 获取连续张量的数据指针
  const float* input = input_tensor_contig.const_data_ptr<float>();
  // 初始化量化输入向量
  std::vector<float> q_input(numel);

  // 计算量化损失，并记录初始最佳损失
  float loss =
      calculate_quant_loss(input, numel, xmin, xmax, q_input.data(), bit_width);
  float best_loss = loss;

  // 初始化当前最小值和最大值以及损失
  float cur_min = xmin;
  float cur_max = xmax;
  float cur_loss = loss;

  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  float thr = min_bins * stepsize;
  // 当当前最小值加上阈值小于当前最大值时循环
  while (cur_min + thr < cur_max) {
    // 向左移动并计算损失
    float loss1 = calculate_quant_loss(
        input, numel, cur_min + stepsize, cur_max, q_input.data(), bit_width);
    // 向右移动并计算损失
    float loss2 = calculate_quant_loss(
        input, numel, cur_min, cur_max - stepsize, q_input.data(), bit_width);
    // 如果当前损失值小于两个备选损失值和最佳损失值，则进入条件判断
    if (cur_loss < loss1 && cur_loss < loss2 && cur_loss < best_loss) {
      // 找到局部最优解
      best_loss = cur_loss;  // 更新最佳损失值为当前损失值
      xmin = cur_min;        // 更新最佳 xmin 为当前最小值
      xmax = cur_max;        // 更新最佳 xmax 为当前最大值
    }
    // 比较两个备选损失值，选择较小的进行下一步优化
    if (loss1 < loss2) {
      cur_min = cur_min + stepsize;  // 更新当前最小值为当前最小值加上步长
      cur_loss = loss1;              // 更新当前损失值为第一个备选损失值
    } else {
      cur_max = cur_max - stepsize;  // 更新当前最大值为当前最大值减去步长
      cur_loss = loss2;              // 更新当前损失值为第二个备选损失值
    }
  }

  // 创建包含一个元素的空 Tensor 作为最终结果的容器
  at::Tensor xmax_tensor = at::empty({1});
  at::Tensor xmin_tensor = at::empty({1});
  // 将找到的最佳 xmax 和 xmin 分别存入 Tensor
  xmax_tensor[0] = xmax;
  xmin_tensor[0] = xmin;
  // 返回包含 xmax 和 xmin 的元组作为函数的结果
  return std::make_tuple(xmax_tensor, xmin_tensor);
}
} // namespace native
} // namespace at
```