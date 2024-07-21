# `.\pytorch\aten\src\ATen\native\quantized\cpu\fbgemm_utils.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于在编译时指定仅包含操作符方法

#include <ATen/Context.h>
// 包含 ATen 库中的 Context 头文件，提供了与上下文相关的函数和类

#include <ATen/Dispatch.h>
// 包含 ATen 库中的 Dispatch 头文件，实现了分发机制，用于根据参数类型调用不同的函数

#include <ATen/Utils.h>
// 包含 ATen 库中的 Utils 头文件，提供了一些实用函数和工具类

#include <ATen/core/TensorBody.h>
// 包含 ATen 库中的 TensorBody 头文件，定义了 Tensor 对象的内部结构

#include <ATen/core/ivalue.h>
// 包含 ATen 库中的 IValue 头文件，定义了用于表示运行时值的类

#include <ATen/core/jit_type_base.h>
// 包含 ATen 库中的 jit_type_base 头文件，提供了用于 JIT 类型的基本定义和操作

#include <ATen/native/quantized/PackedParams.h>
// 包含 ATen 库中的 quantized/PackedParams 头文件，提供了量化模型中打包参数的支持

#include <ATen/native/quantized/cpu/conv_serialization.h>
// 包含 ATen 库中的 quantized/cpu/conv_serialization 头文件，提供了 CPU 上卷积操作序列化的支持

#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
// 包含 ATen 库中的 quantized/cpu/EmbeddingPackedParams 头文件，提供了 CPU 上嵌入打包参数的支持

#include <ATen/native/quantized/cpu/fbgemm_utils.h>
// 包含 ATen 库中的 quantized/cpu/fbgemm_utils 头文件，提供了使用 FBGEMM 的量化操作支持

#include <ATen/native/quantized/cpu/QnnpackUtils.h>
// 包含 ATen 库中的 quantized/cpu/QnnpackUtils 头文件，提供了使用 QNNPACK 的量化操作支持

#include <ATen/native/quantized/cpu/OnednnUtils.h>
// 包含 ATen 库中的 quantized/cpu/OnednnUtils 头文件，提供了使用 OneDNN 的量化操作支持

#include <ATen/native/TensorFactories.h>
// 包含 ATen 库中的 TensorFactories 头文件，提供了创建 Tensor 的工厂函数

#include <ATen/quantized/QTensorImpl.h>
// 包含 ATen 库中的 QTensorImpl 头文件，定义了量化 Tensor 的实现类

#include <ATen/quantized/Quantizer.h>
// 包含 ATen 库中的 Quantizer 头文件，定义了量化器类

#include <c10/core/QScheme.h>
// 包含 c10 库中的 QScheme 头文件，定义了量化方案枚举

#include <c10/core/TensorOptions.h>
// 包含 c10 库中的 TensorOptions 头文件，定义了 Tensor 的选项类

#include <c10/util/accumulate.h>
// 包含 c10 库中的 accumulate 头文件，提供了对容器中元素进行累积操作的函数

#include <c10/util/irange.h>
// 包含 c10 库中的 irange 头文件，提供了生成整数范围的函数

#include <torch/custom_class.h>
// 包含 torch 库中的 custom_class 头文件，支持自定义类的注册

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
// 如果没有定义 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 库中的 Functions 头文件；否则包含 ops/cat 头文件

#include <utility>
// 包含实用工具类的头文件
#endif

// 注册嵌入参数的函数声明
int register_embedding_params();

#ifdef USE_FBGEMM
// 如果定义了 USE_FBGEMM 宏，则进入 FBGEMM 相关命名空间
namespace at {
namespace native {
namespace fbgemm_utils {

namespace {

// 检查是否为 Channels Last 的 3D Tensor
bool IsChannelsLast3d(const Tensor& tensor) {
  if (tensor.dim() != 5) {
    return false;
  }
  const int64_t C = tensor.size(1);
  const int64_t D = tensor.size(2);
  const int64_t H = tensor.size(3);
  const int64_t W = tensor.size(4);
  return tensor.stride(0) == D * H * W * C && tensor.stride(1) == 1 &&
      tensor.stride(2) == H * W * C && tensor.stride(3) == W * C &&
      tensor.stride(4) == C;
}

// 将数据复制到 Channels Last 的 3D Tensor
template <typename T>
void CopyToChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const T* src,
    T* dst) {
  const int64_t inner_size = D * H * W;
  for (const auto i : c10::irange(N)) {
    for (const auto j : c10::irange(inner_size)) {
      for (const auto k : c10::irange(C)) {
        dst[(i * inner_size + j) * C + k] = src[(i * C + k) * inner_size + j];
      }
    }
  }
}

// 将 IC First 的 3D Tensor 复制到 Channels Last 的 3D Tensor
template <typename T>
void CopyICFirst3dTensorToChannelsLast3dTensor(
    int64_t G,
    int64_t IC_G,
    int64_t OC_G,
    int64_t D,
    int64_t H,
    int64_t W,
    const T* src,
    T* dst) {
  // IC OC/G THW -> G OC/G THW IC/G
  const int64_t inner_size = D * H * W;
  for (int64_t i = 0; i < G * OC_G; ++i) {
    for (const auto j : c10::irange(inner_size)) {
      for (const auto ic : c10::irange(IC_G)) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        int g = i / OC_G;
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        int oc = i % OC_G;
        dst[(i * inner_size + j) * IC_G + ic] =
            src[((g * IC_G + ic) * OC_G + oc) * inner_size + j];
      }
    }
  }
}

} // namespace

// 创建 FbgemmConvParam 结构的模板函数，用于构造 FBGEMM 卷积的参数
template <int kSpatialDim>
fbgemm::conv_param_t<kSpatialDim> MakeFbgemmConvParam(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    // 使用 const 引用传递 pads 向量，包含每个空间维度的填充量
    // 使用 const 引用传递 dilations 向量，包含每个空间维度的膨胀率
    // 使用 const 引用传递 output_padding 向量，包含每个空间维度的输出填充
    // 传递一个布尔值 transposed，指示是否是转置卷积操作
    {
      // 声明一个固定大小的数组，存储图像形状的维度信息
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      std::array<int, kSpatialDim> image_shape_;
      // 声明一个固定大小的数组，存储卷积核大小的维度信息
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      std::array<int, kSpatialDim> kernels_;
      // 声明一个固定大小的数组，存储步长的维度信息
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      std::array<int, kSpatialDim> strides_;
      // 声明一个固定大小的数组，存储填充的维度信息（扩展为二维）
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      std::array<int, kSpatialDim * 2> pads_;
      // 声明一个固定大小的数组，存储膨胀率的维度信息
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      std::array<int, kSpatialDim> dilations_;
      // 声明一个固定大小的数组，存储输出填充的维度信息
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      std::array<int, kSpatialDim> output_padding_;
    
      // 将 image_shape 向量的数据移动到 image_shape_ 数组中
      std::move(image_shape.begin(), image_shape.begin() + image_shape.size(), image_shape_.begin());
      // 将 kernels 向量的数据移动到 kernels_ 数组中
      std::move(kernels.begin(), kernels.begin() + kernels.size(), kernels_.begin());
      // 将 strides 向量的数据移动到 strides_ 数组中
      std::move(strides.begin(), strides.begin() + strides.size(), strides_.begin());
      // 将 dilations 向量的数据移动到 dilations_ 数组中
      std::move(dilations.begin(), dilations.begin() + dilations.size(), dilations_.begin());
      // 将 output_padding 向量的数据移动到 output_padding_ 数组中
      std::move(output_padding.begin(), output_padding.begin() + output_padding.size(), output_padding_.begin());
      // 将 pads 向量的数据复制到 pads_ 数组中
      std::copy(pads.begin(), pads.begin() + pads.size(), pads_.begin());
      // 将 pads 向量的数据移动到 pads_ 数组中，从 pads_ 数组的第 pads.size() 个元素开始移动
      std::move(pads.begin(), pads.begin() + pads.size(), pads_.begin() + pads.size());
    
      // 返回使用给定参数创建的卷积参数对象
      return fbgemm::conv_param_t<kSpatialDim>(
          N, // 批量大小
          C, // 输入通道数
          M, // 输出通道数
          image_shape_, // 特征图大小
          groups, // 组数
          kernels_, // 卷积核大小
          strides_, // 步长
          pads_, // 填充
          dilations_, // 膨胀率
          output_padding_, // 转置卷积的输出填充
          transposed);
    }
Tensor MakeStridedQTensorCPU(
    const IntArrayRef& sizes,                        // 接收张量的尺寸
    const IntArrayRef& strides,                      // 接收张量的步长
    const TensorOptions& options,                   // 接收张量的选项
    QuantizerPtr quantizer) {                        // 接收量化器指针
  AT_ASSERT(options.device().is_cpu());             // 断言张量选项的设备为 CPU
  at::native::check_size_nonnegative(sizes);        // 检查张量尺寸是否非负
  auto* allocator = at::getCPUAllocator();          // 获取 CPU 分配器
  const int64_t nelements = c10::multiply_integers(sizes);  // 计算张量元素总数
  auto dtype = options.dtype();                     // 获取张量数据类型
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),      // 检查数据类型是否支持量化整数类型
      "ScalarType is not supported in new_qtensor_cpu.");
  int64_t size_bytes = nelements * dtype.itemsize(); // 计算张量总字节数
  auto storage = c10::make_intrusive<StorageImpl>(   // 创建存储实现对象
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /* resizable = */ true);
  constexpr auto quantized_cpu_ks = at::DispatchKeySet(at::DispatchKey::QuantizedCPU);  // 定义量化 CPU 分发键集合
  auto tensor = detail::make_tensor<QTensorImpl>(   // 创建量化张量对象
      storage,
      quantized_cpu_ks,
      dtype,
      quantizer);
  get_qtensorimpl(tensor)->set_sizes_and_strides(sizes, strides);  // 设置张量的尺寸和步长
  return tensor;                                    // 返回创建的张量对象
}

Tensor MakeEmptyAffineQuantizedChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const TensorOptions& options,
    double scale,
    int64_t zero_point) {
  return MakeStridedQTensorCPU(                     // 调用 MakeStridedQTensorCPU 函数创建量化张量
      {N, C, D, H, W},                              // 张量尺寸
      {D * H * W * C, 1, H * W * C, W * C, C},       // 张量步长
      options,                                      // 张量选项
      make_per_tensor_affine_quantizer(              // 创建每张量元素仿射量化器
          scale, zero_point, typeMetaToScalarType(options.dtype())));
}

Tensor MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const TensorOptions& options,
    const Tensor& scales,
    const Tensor& zero_points) {
  return MakeStridedQTensorCPU(                     // 调用 MakeStridedQTensorCPU 函数创建量化张量
      {N, C, D, H, W},                              // 张量尺寸
      {D * H * W * C, 1, H * W * C, W * C, C},       // 张量步长
      options,                                      // 张量选项
      make_per_channel_affine_quantizer(            // 创建每通道仿射量化器
          scales,
          zero_points,
          0, // axis                                // 指定量化的轴
          typeMetaToScalarType(options.dtype())));
}

Tensor ConvertToChannelsLast3dTensor(const Tensor& src) {
  TORCH_CHECK(src.dim() == 5);                      // 检查输入张量是否是五维张量
  Tensor dst;
  if (IsChannelsLast3d(src)) {                      // 如果输入张量已经是通道最后的三维张量
    dst = src;                                      // 直接复制输入张量到目标张量
  } else {
    const int64_t N = src.size(0);                  // 获取输入张量的尺寸
    const int64_t C = src.size(1);
    const int64_t D = src.size(2);
    const int64_t H = src.size(3);
    const int64_t W = src.size(4);
    dst = MakeStridedQTensorCPU(                    // 调用 MakeStridedQTensorCPU 函数创建量化张量
        {N, C, D, H, W},                            // 张量尺寸
        {D * H * W * C, 1, H * W * C, W * C, C},     // 张量步长
        src.options(),                              // 使用输入张量的选项
        src.quantizer());                           // 使用输入张量的量化器
    AT_DISPATCH_QINT_TYPES(
        src.scalar_type(),                          // 遍历量化整数类型
        "ConvertToChannelsLast3dTensor", [&]() {
          const Tensor src_contig = src.contiguous(); // 获取输入张量的连续版本
          CopyToChannelsLast3dTensor<scalar_t>(      // 调用函数将数据复制到通道最后的三维张量
              N,
              C,
              D,
              H,
              W,
              src_contig.data_ptr<scalar_t>(),
              dst.data_ptr<scalar_t>());
        });
  }
  return dst;                                       // 返回创建或复制的目标张量
}
// 定义模板函数，将具有2维卷积转置的张量转换为所需的格式
Tensor TransposeConvTensorUnpackConversion<2>(const Tensor& src, int groups) {
  // 将输出通道 OC 按组分块，得到组内通道数 IC/G 和 HW 组成的张量列表
  auto oc_g_ic_g_hw_tensors = src.chunk(groups);
  // 在通道维度上连接所有分块张量，形成融合后的张量
  auto fused_tensor = at::cat(oc_g_ic_g_hw_tensors, 1);
  // 设置融合后的张量的量化器为原始张量的量化器
  set_quantizer_(fused_tensor, src.quantizer());
  // 对融合后的张量执行维度置换，将 IC 放在前面，以符合逻辑顺序 IC OC/G HW
  return fused_tensor.permute({1, 0, 2, 3});
}

// 定义模板函数，生成适用于 Fbgemm 的卷积参数结构体，用于2维卷积
template fbgemm::conv_param_t<2> MakeFbgemmConvParam<2>(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations,
    const std::vector<int>& output_padding,
    bool transposed);

// 定义模板函数，生成适用于 Fbgemm 的卷积参数结构体，用于3维卷积
template fbgemm::conv_param_t<3> MakeFbgemmConvParam<3>(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations,
    const std::vector<int>& output_padding,
    bool transposed);

// 定义模板特化函数，将具有3维卷积转置的张量转换为所需的格式
template <>
Tensor TransposeConvTensorUnpackConversion<3>(const Tensor& src, int groups) {
  // 将输出通道 OC 按组分块，得到组内通道数 IC/G 和 DHW 组成的张量列表
  auto oc_g_ic_g_hw_tensors = src.chunk(groups);
  // 在通道维度上连接所有分块张量，形成融合后的张量
  auto fused_tensor = at::cat(oc_g_ic_g_hw_tensors, 1);
  // 设置融合后的张量的量化器为原始张量的量化器
  set_quantizer_(fused_tensor, src.quantizer());
  // 对融合后的张量执行维度置换，将 IC 放在前面，以符合逻辑顺序 IC OC/G DHW
  return fused_tensor.permute({1, 0, 2, 3, 4});
}

// 定义模板特化函数，将2维卷积的权重张量转换为通道最后的张量格式
template <>
Tensor ConvertConvWeightsToChannelLastTensor<2>(
    const at::Tensor& src,
    int groups,
    bool transpose) {
  return transpose ?
                   // 如果进行转置，则执行2维卷积转置权重变换
                   [&]() {
                     // 将输入通道 IC 按组分块
                     auto ic_g_oc_g_hw_tensors = src.chunk(groups);
                     // 对每个分块张量在第0维度（batch维度）上增加一维，使得形状变为 G IC/G OC/G KH KW
                     for (auto& tensor : ic_g_oc_g_hw_tensors) {
                       tensor = tensor.unsqueeze(0);
                     }
                     // 在第0维度（batch维度）上连接所有分块张量，形成融合后的张量
                     auto fused_tensor = at::cat(ic_g_oc_g_hw_tensors);
                     // 设置融合后的张量的量化器为原始张量的量化器
                     set_quantizer_(fused_tensor, src.quantizer());
                     // 对融合后的张量执行维度置换，将 G 放在第0维，以符合逻辑顺序 G OC/G KH KW IC/G
                     return fused_tensor.permute({0, 2, 3, 4, 1})
                         // 将融合后的张量转换为连续内存格式
                         .contiguous(c10::MemoryFormat::Contiguous);
                   }()
                   // 如果不进行转置，则保持原始张量的通道最后格式
                   : src.contiguous(c10::MemoryFormat::ChannelsLast);
}

// 定义模板特化函数，将3维卷积的权重张量转换为通道最后的张量格式
template <>
Tensor ConvertConvWeightsToChannelLastTensor<3>(
    const at::Tensor& src,
    int groups,
    bool transpose) {
  if (!transpose) {
    // 如果不进行转置，则调用函数将3维卷积的权重张量转换为通道最后的张量格式
    return ConvertToChannelsLast3dTensor(src);
  } else {
    // 如果进行转置
    // 检查输入张量的维度是否为5
    TORCH_CHECK(src.dim() == 5);
    // 声明目标张量
    Tensor dst;
    // 获取各个维度的大小
    const int64_t N = src.size(0);
    const int64_t IC_G = N / groups;
    const int64_t OC_G = src.size(1);
    const int64_t D = src.size(2);
    const int64_t H = src.size(3);
    const int64_t W = src.size(4);
    // 这里可以继续编写转换过程的详细描述，包括各种操作的目的和影响
    // 创建一个新的量化张量 dst，通过 MakeStridedQTensorCPU 函数生成
    dst = MakeStridedQTensorCPU(
        {groups * OC_G, IC_G, D, H, W},   // 设置新张量的形状为 {groups * OC_G, IC_G, D, H, W}
        {D * H * W * IC_G, 1, H * W * IC_G, W * IC_G, IC_G},  // 设置新张量的步幅
        src.options(),  // 使用原始张量 src 的选项（如设备类型等）创建 dst
        src.quantizer());  // 使用原始张量 src 的量化器创建 dst

    // 调用 AT_DISPATCH_QINT_TYPES 宏，根据 src 的数据类型执行后续操作
    AT_DISPATCH_QINT_TYPES(
        src.scalar_type(), "CopyICFirst3dTensorToChannelsLast3dTensor", [&]() {
          // 将 src 转换为连续存储的张量 src_contig
          const Tensor src_contig = src.contiguous();
          // 调用模板函数 CopyICFirst3dTensorToChannelsLast3dTensor 复制 src_contig 到 dst
          CopyICFirst3dTensorToChannelsLast3dTensor<scalar_t>(
              groups,
              IC_G,
              OC_G,
              D,
              H,
              W,
              src_contig.data_ptr<scalar_t>(),  // src_contig 的数据指针
              dst.data_ptr<scalar_t>());  // dst 的数据指针
        });

    // 返回处理后的 dst 张量
    return dst;
}
#ifndef USE_FBGEMM

// 如果未定义 USE_FBGEMM 宏，则编译器会包含以下代码块

namespace {
  // 这段代码是一个匿名命名空间，用于定义一个内部函数
  // 该函数将整数转换为对应的类名字符串，并在量化注册过程中使用
  // 如果给定的整数不在预期的范围内，则断言失败
  constexpr const char* _hack_int_to_class_name(int x) {
    switch(x) {
      case 2:
        return "Conv2dPackedParamsBase";
      case 3:
        return "Conv3dPackedParamsBase";
      default:
        assert(false);  // 如果 x 不是 2 或 3，则断言失败
        return "NotAValidDimension";
    }
  }
}

template <int kSpatialDim = 2>
TORCH_API int
register_conv_params() {
  // 定义一个静态变量，用于注册量化卷积参数
  static auto register_conv_params =
    torch::selective_class_<ConvPackedParamsBase<kSpatialDim>>(
        "quantized", TORCH_SELECTIVE_CLASS(_hack_int_to_class_name(kSpatialDim)))
    .def_pickle(
        // 定义序列化函数 __getstate__
        [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params)
        -> ConvParamsSerializationType {
          return serialize_conv<kSpatialDim>(params);
        },
        // 定义反序列化函数 __setstate__
        [](c10::IValue v)
        -> c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> {
          ConvParamsSerializationTypeV3 state = parse_conv_serialized_state<kSpatialDim>(v);
          return deserialize_conv<kSpatialDim>(state);
        })
    .def("weight", [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& self) {
                     return std::get<0>(self->unpack());
                   })
    .def("bias", [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& self) {
                     return std::get<1>(self->unpack());
                 })
    .def("unpack", &ConvPackedParamsBase<kSpatialDim>::unpack) // 定义解包函数
    .def("stride", &ConvPackedParamsBase<kSpatialDim>::stride) // 定义步长函数
    .def("padding", &ConvPackedParamsBase<kSpatialDim>::padding) // 定义填充函数
    .def("output_padding", &ConvPackedParamsBase<kSpatialDim>::output_padding) // 定义输出填充函数
    .def("dilation", &ConvPackedParamsBase<kSpatialDim>::dilation) // 定义膨胀函数
    .def("groups", &ConvPackedParamsBase<kSpatialDim>::groups) // 定义组数函数
    .def("transpose", &ConvPackedParamsBase<kSpatialDim>::transpose); // 定义转置函数
  return 0; // 返回注册成功的标志
}

template
TORCH_API int register_conv_params<2>(); // 实例化注册二维卷积参数的模板

template
TORCH_API int register_conv_params<3>(); // 实例化注册三维卷积参数的模板

TORCH_API int register_linear_params(); // 声明注册线性参数的函数
// 注册线性参数的函数
TORCH_API int register_linear_params() {
  // 定义序列化类型为包含权重张量和可选偏置张量的元组
  using SerializationType = std::tuple<at::Tensor, std::optional<at::Tensor>>;
  // 静态局部变量，注册 LinearPackedParamsBase 类型的选择性类，并命名为 "quantized"
  static auto register_linear_params =
      torch::selective_class_<LinearPackedParamsBase>(
          "quantized", TORCH_SELECTIVE_CLASS("LinearPackedParamsBase"))
          // 定义序列化和反序列化方法
          .def_pickle(
              [](const c10::intrusive_ptr<LinearPackedParamsBase>& params)
                  -> SerializationType { // __getstate__
                return params->unpack();
              },
              [](SerializationType state)
                  -> c10::intrusive_ptr<
                      LinearPackedParamsBase> { // __setstate__
                // 从序列化状态中解包出权重和偏置
                at::Tensor weight;
                std::optional<at::Tensor> bias;
                weight = std::move(std::get<0>(state));
                bias = std::move(std::get<1>(state));

#ifdef USE_FBGEMM
                // 如果使用 FBGEMM 引擎或者 X86 引擎
                if (at::globalContext().qEngine() == at::QEngine::FBGEMM ||
                    at::globalContext().qEngine() == at::QEngine::X86) {
                  // 如果权重类型为 int8
                  if (weight.scalar_type() == at::kQInt8) {
                    // 使用 PackedLinearWeight 类预打包权重和偏置
                    return PackedLinearWeight::prepack(
                        std::move(weight), std::move(bias));
                  } else if (weight.scalar_type() == at::kFloat) {
                    // 注意：fp16 权重序列化为 float 类型
                    // 使用 PackedLinearWeightFp16 类预打包权重和偏置
                    return PackedLinearWeightFp16::prepack(
                        std::move(weight), std::move(bias));
                  } else {
                    // 抛出异常，不支持的数据类型
                    TORCH_CHECK(
                        false,
                        "Unsupported data type",
                        c10::toString(weight.scalar_type()),
                        " in serialized LinearPackedParams object!");
                  }
                }
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
                // 如果使用 QNNPACK 引擎
                if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
                  // QNNPACK 只支持 int8 类型的权重
                  TORCH_CHECK(
                      weight.scalar_type() == at::kQInt8,
                      "QNNPACK only supports INT8 bit width currently. Got ",
                      c10::toString(weight.scalar_type()));
                  // 使用 PackedLinearWeightsQnnp 类预打包权重和偏置
                  return PackedLinearWeightsQnnp::prepack(
                      std::move(weight), std::move(bias));
                }
#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
                // 如果使用 ONEDNN 引擎
                if (at::globalContext().qEngine() == at::QEngine::ONEDNN) {
                  // ONEDNN 只支持 int8 类型的权重
                  TORCH_CHECK(
                      weight.scalar_type() == at::kQInt8,
                      "ONEDNN only supports INT8 bit width currently. Got ",
                      c10::toString(weight.scalar_type()));
                  // 使用 PackedLinearWeightsOnednn 类预打包权重和偏置
                  return PackedLinearWeightsOnednn::prepack(
                      std::move(weight), std::move(bias));
                }
#endif // AT_MKLDNN_ENABLED()
#ifdef #if AT_MKLDNN_ENABLED()
                TORCH_CHECK(false, "Unknown qengine");
              })
              .def("bias", [](const c10::intrusive_ptr<LinearPackedParamsBase>& self) {
                  return std::get<1>(self->unpack());
                 })
              .def("unpack", &LinearPackedParamsBase::unpack);
  // (1) 返回静态初始化器本身不容易，因为由于选择性构建可能具有不同的类型
  // (2) 不能返回 void 并且能在全局作用域中调用函数
  return 0;
}


int register_embedding_params() {
  // __getstate__/__setstate__ 序列化的类型定义
  //
  // 元素 0 是 PackedParam 结构的版本
  // 元素 1 是 Param 实例中包含的张量
  // 元素 2 是 Param 实例中包含的双精度值（如果有的话）
  // 元素 3 是 Param 实例中包含的整数值（如果有的话）
  using EmbeddingParamsSerializationType = std::tuple<
    int64_t, // 版本
    std::vector<at::Tensor>, // 张量向量
    std::vector<double>, // 双精度值向量
    std::vector<int64_t>>; // 整数值向量

  static auto register_embedding_params =
    torch::selective_class_<EmbeddingPackedParamsBase>(
      "quantized", TORCH_SELECTIVE_CLASS("EmbeddingPackedParamsBase"))
      .def_pickle(
          [](const c10::intrusive_ptr<EmbeddingPackedParamsBase>& params)
              -> EmbeddingParamsSerializationType { // __getstate__ 调用
            at::Tensor weight = params->unpack();
            std::vector<at::Tensor> tensors_to_serialize = {std::move(weight)};
            std::vector<double> doubles_to_serialize = {};
            int64_t bit_rate = params->bit_rate();
            int64_t version = params->version();
            std::vector<int64_t> longs_to_serialize = {bit_rate};
            return EmbeddingParamsSerializationType(
              version,
              std::move(tensors_to_serialize),
              std::move(doubles_to_serialize),
              std::move(longs_to_serialize));
          },
          [](EmbeddingParamsSerializationType state)
              -> c10::intrusive_ptr<EmbeddingPackedParamsBase> { // __setstate__ 调用

            auto [version, tensors, doubles, longs] = std::move(state);

            TORCH_INTERNAL_ASSERT(tensors.size() == 1, "EmbeddingPackedParams: 期望序列化的权重张量");
            TORCH_INTERNAL_ASSERT(longs.size() == 1, "EmbeddingPackedParams: 期望序列化的比特率");
            TORCH_CHECK(version == 1, "EmbeddingPackedParams: 当前仅支持版本 1.");

            at::Tensor weight = std::move(tensors[0]);
            return PackedEmbeddingBagWeight::prepack(std::move(weight));
          })
      .def("bit_rate", &EmbeddingPackedParamsBase::bit_rate) // 定义获取比特率的方法
      .def("unpack", &EmbeddingPackedParamsBase::unpack) // 定义解包的方法
      .def("version", &EmbeddingPackedParamsBase::version); // 定义获取版本号的方法

  return 0;
}

namespace {

static C10_UNUSED auto conv2d_params = register_conv_params<2>();
# 声明并初始化静态变量 conv3d_params，用于注册三维卷积参数
static C10_UNUSED auto conv3d_params = register_conv_params<3>();

# 声明并初始化静态变量 linear_params，用于注册线性层参数
static C10_UNUSED auto linear_params = register_linear_params();

# 声明并初始化静态变量 embedding_params，用于注册嵌入层参数
static C10_UNUSED auto embedding_params = register_embedding_params();

# 结束当前命名空间
} // namespace
```