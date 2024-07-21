# `.\pytorch\aten\src\ATen\quantized\Quantizer.cpp`

```py
// 包含 ATen 库的头文件
#include <ATen/ArrayRef.h>
#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/Dispatch.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/NativeFunctions.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/accumulate.h>

// 包含标准库头文件
#include <cmath>
#include <utility>

// ATen 命名空间
namespace at {

// 匿名命名空间，用于实现局部函数和变量的封装
namespace {

// 检查每通道量化参数的维度是否正确
void checkPerChannelParamDims(const Tensor& scales, const Tensor& zero_points) {
    TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
    TORCH_CHECK(
        zero_points.dim() == 1, "zero_points tensor must have dimension 1");
    TORCH_CHECK(
        scales.numel() == zero_points.numel(),
        "number of elements in scales and zero_points must match");
}

} // anonymous namespace

// 以下是一个注释的示例，QuantizerPtr TensorBase::quantizer() const 函数
// 并不是一个本地函数，因为 Quantizer 尚未暴露给 Python

// 获取张量的量化器指针
QuantizerPtr TensorBase::quantizer() const {
    // 这是一种模仿 VariableType 的可怕 hack
    at::AutoDispatchBelowAutograd mode;
    return get_qtensorimpl(*this)->quantizer();
}

// 创建每张量的仿射量化器
QuantizerPtr make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type) {
  return c10::make_intrusive<PerTensorAffineQuantizer>(scalar_type,
      scale, zero_point);
}

// 创建每通道的仿射量化器
QuantizerPtr make_per_channel_affine_quantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type) {
  // 检查每通道量化参数的维度是否正确
  checkPerChannelParamDims(scales, zero_points);
  TORCH_CHECK(
      isFloatingType(scales.scalar_type()),
      "scale tensor must be floating point");

  // 根据零点参数的数据类型选择合适的量化器类型
  if (isFloatingType(zero_points.scalar_type())) {
    Tensor scales_float = scales.to(kFloat).contiguous();
    Tensor zero_points_float = zero_points.to(kFloat).contiguous();
    return c10::make_intrusive<PerChannelAffineFloatQParamsQuantizer>(scalar_type,
                                                                      scales_float,
                                                                      zero_points_float,
                                                                      axis);
  }
  else {
    Tensor scales_double = scales.to(kDouble).contiguous();
    Tensor zero_points_int64 = zero_points.to(kLong).contiguous();
    return c10::make_intrusive<PerChannelAffineQuantizer>(scalar_type,
                                                          scales_double,
                                                          zero_points_int64,
                                                          axis);
  }
}

// 获取量化张量的 QTensorImpl 指针
QTensorImpl* get_qtensorimpl(const TensorBase& self) {
  TORCH_CHECK(
      !self.requires_grad(),
      "quantized tensors do not support autograd");
  TORCH_INTERNAL_ASSERT(self.is_quantized(), "get_qtensorimpl: not a quantized tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}
static int64_t get_sub_byte_tensor_size(IntArrayRef sizes, size_t dtype_itemsize, at::ScalarType t) {
  // 定义变量用于表示每字节的元素个数
  int64_t element_per_byte;
  // 根据数据类型 t 进行选择
  switch(t) {
    case at::ScalarType::QUInt4x2:
      element_per_byte = 2;
      break;
    case at::ScalarType::QUInt2x4:
      element_per_byte = 4;
      break;
    default:
      element_per_byte = 1;
  }
  // 如果 sizes 为空，返回零维张量的总字节大小
  if (sizes.empty()) {
    return c10::multiply_integers(sizes) * dtype_itemsize;
  }
  // 将最内层的维度视为列数
  int64_t cols = sizes.at(sizes.size()-1);
  // 计算每行所占字节数
  int64_t bytes_per_row = cols * dtype_itemsize;
  // 对齐量化张量的最内层维度，计算每行元素个数（向上取整）
  return c10::multiply_integers(IntArrayRef(sizes.data(), sizes.size() - 1)) * at::ceil_div(bytes_per_row, element_per_byte);
}

inline Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  // 获取内存格式，默认为连续存储
  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  // 获取设备类型
  auto device = options.device();
  at::Allocator* allocator = nullptr;
  // 如果设备是 CUDA
  if (device.is_cuda()) {
    // 获取 CUDA 设备的分配器
    allocator = at::detail::getCUDAHooks().getCUDADeviceAllocator();
  } else if (device.is_cpu()) {
    // 获取 CPU 的分配器
    allocator = at::getCPUAllocator();
  } else if (device.is_meta()) {
    // 获取 meta 设备的分配器
    allocator = GetAllocator(kMeta);
  } else if (device.is_privateuseone()) {
    // 获取私有设备一的分配器
    allocator = GetAllocator(kPrivateUse1);
  } else {
    // 报错，不识别的设备类型
    TORCH_INTERNAL_ASSERT(0, "unrecognized device for new_qtensor: ", device);
  }

#ifdef USE_PYTORCH_QNNPACK
  // 如果使用了 QNNPACK 引擎
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    // 检查是否在启用 QNNPACK 后尝试量化 CUDA 张量
    TORCH_CHECK(!device.is_cuda(), "It looks like you are trying to quantize a CUDA tensor ",
                "while QNNPACK backend is enabled. Although not expected to happen in ",
                "practice, you might have done it for testing purposes. ",
                "Please, either change the quantization engine or move the tensor to a CPU.");
    // 使用默认的移动端 CPU 分配器
    allocator = c10::GetDefaultMobileCPUAllocator();
  }
#endif

  // 计算张量的调度键
  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  // 检查张量大小是否非负
  native::check_size_nonnegative(sizes);
  // 获取张量的数据类型
  auto dtype = options.dtype();
  // 检查是否是整数量化类型
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      "ScalarType ",
      typeMetaToScalarType(dtype),
      " is not supported in new_qtensor.");
  auto scalar_type = typeMetaToScalarType(dtype);
  // 计算子字节张量的总字节大小
  int64_t size_bytes = get_sub_byte_tensor_size(sizes, dtype.itemsize(), scalar_type);

  // 创建存储实现
  auto storage = make_storage_impl(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true,
      device);
  // 创建量化张量
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);
  // 设置量化张量的大小为连续
  get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  // 重新排列空张量
  get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);
  // 返回创建的张量
  return tensor;
}
// 实现 PerTensorAffineQuantizer 类的 quantize 方法，用于对输入张量进行量化
Tensor PerTensorAffineQuantizer::quantize(const Tensor& rtensor) {
  // 检查输入张量是否为 Float 类型，如果不是则抛出错误信息
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat,
      "Quantize only works on Float Tensor, got ", rtensor.scalar_type());
  // 创建一个新的量化后的张量 qtensor，其形状与 rtensor 相同，使用当前量化器的类型和内存布局
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options()
          .dtype(scalar_type_)
          .memory_format(rtensor.suggest_memory_format()),
      intrusive_from_this());

  // 将 rtensor 转换为连续内存布局的张量 rtensor_contig
  auto rtensor_contig = rtensor.expect_contiguous(rtensor.suggest_memory_format());
  // 调用本地方法 quantize_tensor_per_tensor_affine 进行张量的逐元素量化
  native::quantize_tensor_per_tensor_affine(
      *rtensor_contig, qtensor, scale_, zero_point_);
  // 返回量化后的张量 qtensor
  return qtensor;
}

// 静态函数，实现了对张量的逐元素反量化操作
static void per_tensor_affine_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const double scale,
    const int64_t zero_point) {
  // 检查 qtensor 是否为连续内存布局的张量，并获取其期望的内存布局
  const auto qtensor_contig =
    qtensor.expect_contiguous(qtensor.suggest_memory_format());
  // 调用本地方法 dequantize_tensor_per_tensor_affine 对 qtensor 进行反量化操作
  native::dequantize_tensor_per_tensor_affine(
      *qtensor_contig, rtensor, scale, zero_point);
}

// 实现 PerTensorAffineQuantizer 类的 dequantize_out 方法，用于对 qtensor 进行反量化并写入 rtensor
Tensor& PerTensorAffineQuantizer::dequantize_out(
    Tensor& rtensor, const Tensor& qtensor) {
  // 调整 rtensor 的形状与 qtensor 相同
  rtensor.resize_(qtensor.sizes());
  // 检查 rtensor 是否为连续内存布局的 Float 张量，并与 qtensor 的内存布局一致
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
      rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  // 调用 per_tensor_affine_dequantize_impl 方法对 qtensor 进行反量化操作，并写入 rtensor
  per_tensor_affine_dequantize_impl(rtensor, qtensor, scale_, zero_point_);
  // 返回反量化后的 rtensor
  return rtensor;
}

// 实现 PerTensorAffineQuantizer 类的 dequantize 方法，对 qtensor 进行反量化操作
Tensor PerTensorAffineQuantizer::dequantize(const Tensor& qtensor) {
  // 创建一个新的 Float 张量 rtensor，形状与 qtensor 相同，内存布局与 qtensor 一致
  Tensor rtensor = at::empty(
      qtensor.sizes(),
      qtensor.options()
          .dtype(at::kFloat)
          .memory_format(qtensor.suggest_memory_format()));
  // 调用 per_tensor_affine_dequantize_impl 方法对 qtensor 进行反量化操作，并写入 rtensor
  per_tensor_affine_dequantize_impl(rtensor, qtensor, scale_, zero_point_);
  // 返回反量化后的 rtensor
  return rtensor;
}

// 实现 PerChannelAffineQuantizer 类的 quantize 方法，用于对输入张量进行按通道量化
Tensor PerChannelAffineQuantizer::quantize(const Tensor& rtensor) {
  // 创建一个新的量化后的张量 qtensor，其形状与 rtensor 相同，使用当前量化器的类型和内存布局
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options()
          .dtype(scalar_type_)
          .memory_format(rtensor.suggest_memory_format()),
      intrusive_from_this());
  // 将 rtensor 转换为连续内存布局的张量 rtensor_contig
  auto rtensor_contig = rtensor.expect_contiguous(rtensor.suggest_memory_format());
  // 调用本地方法 quantize_tensor_per_channel_affine 进行张量的按通道量化
  native::quantize_tensor_per_channel_affine(
      *rtensor_contig, qtensor, scales_, zero_points_, axis_);
  // 返回量化后的张量 qtensor
  return qtensor;
}

// 静态函数，实现了对张量的按通道反量化操作
static void per_channel_affine_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scale,
    const Tensor& zero_point,
    const int64_t axis) {
  // 检查 qtensor 是否为连续内存布局的张量，并获取其期望的内存布局
  const auto qtensor_contig =
    qtensor.expect_contiguous(qtensor.suggest_memory_format());
  // 调用本地方法 dequantize_tensor_per_channel_affine 对 qtensor 进行按通道反量化操作
  native::dequantize_tensor_per_channel_affine(
      *qtensor_contig, rtensor, scale, zero_point, axis);
}
Tensor PerChannelAffineQuantizer::dequantize(const Tensor& qtensor) {
  // 创建一个和 qtensor 相同大小的空张量 rtensor，使用 Float 类型
  Tensor rtensor = at::empty(
      qtensor.sizes(),
      qtensor.options()
          .dtype(at::kFloat)
          .memory_format(qtensor.suggest_memory_format()));
  // 调用 per_channel_affine_dequantize_impl 函数，将 qtensor 解量化到 rtensor
  per_channel_affine_dequantize_impl(rtensor, qtensor, scales_, zero_points_, axis_);
  // 返回解量化后的 rtensor
  return rtensor;
}

Tensor& PerChannelAffineQuantizer::dequantize_out(
    Tensor& rtensor, const Tensor& qtensor) {
  // 调整 rtensor 的大小和 qtensor 相同
  rtensor.resize_(qtensor.sizes());
  // 检查 rtensor 是否是连续的 Float 张量，和数据格式是否和 qtensor 的建议格式匹配
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
      rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  // 调用 per_channel_affine_dequantize_impl 函数，将 qtensor 解量化到 rtensor
  per_channel_affine_dequantize_impl(rtensor, qtensor, scales_, zero_points_, axis_);
  // 返回解量化后的 rtensor
  return rtensor;
}

Tensor PerChannelAffineFloatQParamsQuantizer::quantize(const Tensor& rtensor) {
  // 检查 rtensor 是否是 Float 张量
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat,
      "Quantize only works on Float Tensor, got ", rtensor.scalar_type());
  // 创建一个新的量化后的 qtensor，使用当前量化器的标量类型
  Tensor qtensor = new_qtensor(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());
  // 获取 rtensor 的连续版本
  auto rtensor_contig = rtensor.expect_contiguous();
  // 调用 native::quantize_tensor_per_channel_float_qparams 函数，对 rtensor 进行量化，存储到 qtensor 中
  native::quantize_tensor_per_channel_float_qparams(
      *rtensor_contig, qtensor, scales_, zero_points_, axis_);
  // 返回量化后的 qtensor
  return qtensor;
}

static void per_channel_affine_float_q_params_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scale,
    const Tensor& zero_point,
    const int64_t axis) {
  // 获取 qtensor 的连续版本
  const auto qtensor_contig =
    qtensor.expect_contiguous(qtensor.suggest_memory_format());
  // 调用 native::dequantize_tensor_per_channel_float_qparams 函数，将 qtensor 解量化到 rtensor
  native::dequantize_tensor_per_channel_float_qparams(
      *qtensor_contig, rtensor, scale, zero_point, axis);
}

Tensor PerChannelAffineFloatQParamsQuantizer::dequantize(const Tensor& qtensor) {
  // 创建一个和 qtensor 相同大小的空张量 rtensor，使用 Float 类型
  Tensor rtensor = at::empty(qtensor.sizes(), qtensor.options().dtype(at::kFloat));
  // 调用 per_channel_affine_float_q_params_dequantize_impl 函数，将 qtensor 解量化到 rtensor
  per_channel_affine_float_q_params_dequantize_impl(
      rtensor, qtensor, scales_, zero_points_, axis_);
  // 返回解量化后的 rtensor
  return rtensor;
}

Tensor& PerChannelAffineFloatQParamsQuantizer::dequantize_out(
    Tensor& rtensor, const Tensor& qtensor) {
  // 调整 rtensor 的大小和 qtensor 相同
  rtensor.resize_(qtensor.sizes());
  // 检查 rtensor 是否是连续的 Float 张量，和数据格式是否和 qtensor 的建议格式匹配
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
      rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  // 调用 per_channel_affine_float_q_params_dequantize_impl 函数，将 qtensor 解量化到 rtensor
  per_channel_affine_float_q_params_dequantize_impl(
      rtensor, qtensor, scales_, zero_points_, axis_);
  // 返回解量化后的 rtensor
  return rtensor;
}

Quantizer::~Quantizer() = default;

C10_EXPORT void set_quantizer_(const Tensor& self, ConstQuantizerPtr quantizer) {
  // 调用 get_qtensorimpl 函数获取 self 的量化 TensorImpl，并设置其量化器为 quantizer
  get_qtensorimpl(self)->set_quantizer_(quantizer);
}

Tensor from_blob_quantized_per_tensor_affine(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    std::function<void(void*)> deleter,  // 定义一个函数对象，用于释放内存的回调函数
    const float scale,  // 量化张量的缩放因子
    const int64_t zeroPoint,  // 量化张量的零点
    const TensorOptions& options) {  // 张量的选项参数引用

  auto dtype = typeMetaToScalarType(options.dtype());  // 获取选项中的数据类型
  TORCH_CHECK(
      isQIntType(dtype),  // 检查数据类型是否为量化整数类型
      "from_blob_quantized_per_tensor_affine expects QInt dtypes, got ", dtype);

  const std::size_t itemsize = options.dtype().itemsize();  // 获取选项中数据类型的字节大小
  std::size_t size = 1;  // 初始化张量的总元素数
  for (std::int64_t s : sizes) {  // 遍历张量的每个维度大小
    size *= static_cast<std::size_t>(s);  // 计算张量的总元素数
  }
  const std::size_t datasize = size * itemsize;  // 计算张量数据的总字节数

  DataPtr data_ptr = InefficientStdFunctionContext::makeDataPtr(
      data, deleter, options.device());  // 创建数据指针，用于存储张量的数据

  Storage storage{Storage::use_byte_size_t{}, datasize, std::move(data_ptr)};  // 创建存储器对象，指定字节大小和数据指针

  QuantizerPtr quantizer =
      make_per_tensor_affine_quantizer(scale, zeroPoint, dtype);  // 创建量化器对象，设置缩放因子、零点和数据类型

  Tensor qtensor = at::detail::make_tensor<QTensorImpl>(
      std::move(storage),  // 使用移动语义将存储器对象传递给张量实现构造函数
      at::DispatchKeySet(options.computeDispatchKey()),  // 设置张量的调度键集合
      options.dtype(),  // 设置张量的数据类型
      quantizer);  // 设置张量的量化器

  get_qtensorimpl(qtensor)->set_sizes_and_strides(sizes, strides);  // 设置张量实现的大小和步长
  return qtensor;  // 返回构造好的量化张量对象
}
}

Tensor from_blob_quantized_per_tensor_affine(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const float scale,
    const int64_t zeroPoint,
    const TensorOptions& options) {
  std::vector<int64_t> strides;
  const auto ndim = sizes.size();
  if (ndim > 0) {
    // 初始化步长向量，长度与张量维度相同
    strides.resize(ndim);
    // 设置最后一个维度的步长为1
    int32_t i = ndim - 1;
    strides[i] = 1;
    // 计算其他维度的步长
    while (--i >= 0) {
      strides[i] = sizes[i + 1] * strides[i + 1];
    }
  }
  // 调用另一个函数，从给定数据创建量化张量（每张量元素固定量化）
  return from_blob_quantized_per_tensor_affine(
      data,
      sizes,
      strides,
      std::move(deleter),
      scale,
      zeroPoint,
      options);
}

Tensor from_blob_quantized_per_channel_affine(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const Tensor& scales,
    const Tensor& zero_points,
    const int64_t axis,
    const TensorOptions& options) {
  // 检查通道量化参数的维度是否匹配
  checkPerChannelParamDims(scales, zero_points);
  // 获取沿指定轴的通道数量
  int64_t channel = sizes[axis];
  // 检查量化参数的长度是否与通道数相等
  TORCH_CHECK(
      channel == int64_t(scales.numel()),
      "length of scales must equal to channel, expected ", channel, " got, ", scales.numel());
  TORCH_CHECK(
      channel == int64_t(zero_points.numel()),
      "length of zero_points must equal to channel, expected ", channel, " got, ", zero_points.numel());

  // 确定张量数据类型
  auto dtype = typeMetaToScalarType(options.dtype());
  // 检查是否为量化整数类型
  TORCH_CHECK(
      isQIntType(dtype),
      "from_blob_quantized_per_channel_affine expects QInt dtypes, got ", dtype);

  // 计算数据项大小和总大小
  const std::size_t itemsize = options.dtype().itemsize();
  std::size_t size = 1;
  for (std::int64_t s : sizes) {
    size *= static_cast<std::size_t>(s);
  }
  const std::size_t datasize = size * itemsize;

  // 创建数据指针
  DataPtr data_ptr = InefficientStdFunctionContext::makeDataPtr(
      data, deleter, options.device());

  // 创建存储空间
  Storage storage{Storage::use_byte_size_t{}, datasize, std::move(data_ptr)};

  // 创建通道量化器
  QuantizerPtr quantizer =
      make_per_channel_affine_quantizer(scales, zero_points, axis, dtype);

  // 创建量化张量
  Tensor qtensor = at::detail::make_tensor<QTensorImpl>(
      std::move(storage),
      at::DispatchKeySet(options.computeDispatchKey()),
      options.dtype(),
      quantizer);

  // 设置量化张量的大小信息，并确保连续存储
  get_qtensorimpl(qtensor)->set_sizes_contiguous(sizes);

  return qtensor;
}

Tensor UnknownQuantizer::quantize(const Tensor& tensor) {
  // 抛出内部断言错误，因为无法在未知量化器上调用量化操作
  TORCH_INTERNAL_ASSERT(false, "cannot call quantize on UnknownQuantizer");
}
Tensor UnknownQuantizer::dequantize(const Tensor& qtensor) {
  // 抛出内部断言错误，因为无法在未知量化器上调用去量化操作
  TORCH_INTERNAL_ASSERT(false, "cannot call dequantize on UnknownQuantizer");
}
Tensor& UnknownQuantizer::dequantize_out(Tensor& rtensor, const Tensor& qtensor) {
  // 抛出内部断言错误，因为无法在未知量化器上调用输出去量化操作
  TORCH_INTERNAL_ASSERT(false, "cannot call dequantize_out on UnknownQuantizer");
}
QScheme UnknownQuantizer::qscheme() const {
  // 抛出内部断言错误，因为无法在未知量化器上获取量化方案
  TORCH_INTERNAL_ASSERT(false, "cannot call qscheme on UnknownQuantizer");
}
bool UnknownQuantizer::equalTo(QuantizerPtr other) const{
  // 抛出内部断言错误，因为无法在未知量化器上进行相等性比较
  TORCH_INTERNAL_ASSERT(false, "cannot call equalTo on UnknownQuantizer");
}
# 创建一个未知量化器的指针，并返回该指针
QuantizerPtr make_unknown_quantizer(ScalarType scalar_type) {
    # 使用 c10 命名空间中的 make_intrusive 函数创建一个 UnknownQuantizer 类型的智能指针，
    # 传入的参数是给定的标量类型 scalar_type
    return c10::make_intrusive<UnknownQuantizer>(scalar_type);
}

# 结束 at 命名空间的定义
} // namespace at
```