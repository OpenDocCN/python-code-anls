# `.\pytorch\aten\src\ATen\native\TensorFactories.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于仅包含方法操作符的头文件
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的张量工厂头文件
#include <ATen/native/TensorFactories.h>

// 包含 ATen 核心张量类的头文件
#include <ATen/core/Tensor.h>

// 包含 ATen CPU 生成器实现的头文件
#include <ATen/CPUGeneratorImpl.h>

// 包含 ATen 分发机制的头文件
#include <ATen/Dispatch.h>

// 包含 ATen 空张量的头文件
#include <ATen/EmptyTensor.h>

// 包含 ATen 张量扩展工具的头文件
#include <ATen/ExpandUtils.h>

// 包含 ATen 并行处理的头文件
#include <ATen/Parallel.h>

// 包含 ATen 内存分配器映射的头文件
#include <ATen/MapAllocator.h>

// 包含 ATen 稀疏 CSR 张量工具的头文件
#include <ATen/SparseCsrTensorUtils.h>

// 包含 ATen 追踪模式的头文件
#include <ATen/TracerMode.h>

// 包含 ATen 张量操作符的头文件
#include <ATen/TensorOperators.h>

// 包含 ATen 命名张量工具的头文件
#include <ATen/NamedTensorUtils.h>

// 包含 ATen 一元运算的头文件
#include <ATen/native/UnaryOps.h>

// 包含 C10 标量类型的头文件
#include <c10/core/ScalarType.h>

// 包含 C10 张量选项的头文件
#include <c10/core/TensorOptions.h>

// 包含 C10 异常处理的头文件
#include <c10/util/Exception.h>

// 包含 C10 范围计算的头文件
#include <c10/util/irange.h>

// 包含 C10 数学常量的头文件
#include <c10/util/MathConstants.h>

// 根据编译宏 AT_PER_OPERATOR_HEADERS 的不同值，选择性地包含不同的 ATen 操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cast_Byte_native.h>
#include <ATen/ops/_cast_Char_native.h>
#include <ATen/ops/_cast_Double_native.h>
#include <ATen/ops/_cast_Float_native.h>
#include <ATen/ops/_cast_Half_native.h>
#include <ATen/ops/_cast_Int_native.h>
#include <ATen/ops/_cast_Long_native.h>
#include <ATen/ops/_cast_Short_native.h>
#include <ATen/ops/_dim_arange_native.h>
#include <ATen/ops/_efficientzerotensor_native.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/_sparse_compressed_tensor_with_dims_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/bartlett_window_native.h>
#include <ATen/ops/blackman_window_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/complex_native.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_permuted_native.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/eye_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/from_file_native.h>
#include <ATen/ops/full_like_native.h>
#include <ATen/ops/full_native.h>
#include <ATen/ops/hamming_window_native.h>
#include <ATen/ops/hann_window_native.h>
#include <ATen/ops/kaiser_window_native.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/logspace_native.h>
#include <ATen/ops/new_empty_native.h>
#include <ATen/ops/new_empty_strided_native.h>
#include <ATen/ops/new_full_native.h>
#include <ATen/ops/new_ones_native.h>
#include <ATen/ops/new_zeros_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like_native.h>
#include <ATen/ops/ones_native.h>
#include <ATen/ops/polar.h>
#include <ATen/ops/polar_native.h>
#include <ATen/ops/promote_types.h>
#include <ATen/ops/rand_like_native.h>
#include <ATen/ops/rand_native.h>
#include <ATen/ops/randint_like_native.h>
#include <ATen/ops/randint_native.h>
#include <ATen/ops/randn_like_native.h>
#endif
// 包含 ATen 库的相关头文件

#include <ATen/ops/randn_native.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/randperm_native.h>
#include <ATen/ops/range.h>
#include <ATen/ops/range_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/tril_indices_native.h>
#include <ATen/ops/triu_indices_native.h>
#include <ATen/ops/vander_native.h>
#include <ATen/ops/zeros_like_native.h>
#include <ATen/ops/zeros_like_ops.h>
#include <ATen/ops/zeros_native.h>
#endif

#include <c10/core/SymIntArrayRef.h>
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace at::native {

// 定义匿名命名空间，包含局部函数和常量

namespace {
    // 窗口函数参数检查函数
    void window_function_checks(
        const char* function_name,
        const TensorOptions& options,
        int64_t window_length) {
      
      // 检查稀疏布局
      TORCH_CHECK(
          options.layout() != kSparse,
          function_name,
          " is not implemented for sparse types, got: ",
          options);
      
      // 检查浮点或复数类型
      TORCH_CHECK(
          at::isFloatingType(typeMetaToScalarType(options.dtype())) || at::isComplexType(typeMetaToScalarType(options.dtype())),
          function_name,
          " expects floating point dtypes, got: ",
          options);
      
      // 检查非负窗口长度
      TORCH_CHECK(
          window_length >= 0,
          function_name,
          " requires non-negative window_length, got window_length=",
          window_length);
    }
} // namespace

// 定义分发函数的具体实现

DEFINE_DISPATCH(complex_stub);
DEFINE_DISPATCH(polar_stub);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ arange ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 创建从0到end（不包含）的序列张量
Tensor arange(const Scalar& end,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::arange(/*start=*/0, end, dtype, layout, device, pin_memory);
}

// 创建从start到end（不包含）的序列张量
Tensor arange(const Scalar& start, const Scalar& end,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::arange(
      start, end, /*step=*/1, dtype, layout, device, pin_memory);
}

// 创建从start到end（不包含），步长为step的序列张量
Tensor arange(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  
  // 创建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 如果没有指定数据类型，且输入的start、end、step都是整数，则将结果张量类型设为Long型
  bool set_to_integral_dtype = !options.has_dtype() &&
       // 布尔类型的输入被视为整数
       start.isIntegral(true) &&
       end.isIntegral(true) &&
       step.isIntegral(true);

  // 根据是否设置为整数类型，创建空张量
  Tensor result = set_to_integral_dtype
      ? at::empty({0}, options.dtype(at::ScalarType::Long))
      : at::empty({0}, options);
  
  // 调用arange_out函数生成序列张量
  return at::arange_out(result, start, end, step);
}

// 在已有张量result的基础上创建从0到end（不包含）的序列张量
Tensor& arange_out(const Scalar& end, Tensor& result) {
  return at::arange_out(result, /*start=*/0, end, /*step=*/1);
}

// 创建与指定张量like的特定维度大小相同的Long型序列张量
Tensor _dim_arange(const Tensor& like, int64_t dim) {
  return at::arange(like.size(dim), like.options().dtype(at::kLong));
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complex / polar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 检查输入张量的数据类型是否为浮点类型（float、double 或 half）
static void complex_check_floating(const Tensor& a, const Tensor& b) {
  TORCH_CHECK((a.scalar_type() == kFloat || a.scalar_type() == kDouble || a.scalar_type() == kHalf) &&
              (b.scalar_type() == kFloat || b.scalar_type() == kDouble || b.scalar_type() == kHalf),
              "Expected both inputs to be Half, Float or Double tensors but got ",
              a.scalar_type(), " and ", b.scalar_type());
}

// 检查结果张量和输入张量的数据类型是否一致
static void complex_check_dtype(
    const Tensor& result,
    const Tensor& a,
    const Tensor& b) {
  complex_check_floating(a, b);
  // 检查第一个和第二个输入张量的数据类型是否相同
  TORCH_CHECK(a.scalar_type() == b.scalar_type(),
              "Expected object of scalar type ", a.scalar_type(),
              " but got scalar type ", b.scalar_type(), " for second argument");
  // 检查结果张量的数据类型是否为与第一个输入张量相对应的复数类型
  TORCH_CHECK(result.scalar_type() == toComplexType(a.scalar_type()),
              "Expected object of scalar type ", toComplexType(a.scalar_type()),
              " but got scalar type ", result.scalar_type(),
              " for argument 'out'");
}

// 复数的输出函数，根据实部和虚部创建结果张量
Tensor& complex_out(const Tensor& real, const Tensor& imag, Tensor& result) {
  complex_check_dtype(result, real, imag);
  // 设置迭代器配置，指定输出结果张量，实部和虚部作为输入，并关闭相同数据类型的检查
  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_const_input(real)
      .add_const_input(imag)
      .check_all_same_dtype(false)
      .build();
  // 调用复数操作的 stub 函数，根据设备类型执行相应的操作
  complex_stub(iter.device_type(), iter);
  return result;
}

// 创建复数张量，根据实部和虚部张量返回复数张量
Tensor complex(const Tensor& real, const Tensor& imag) {
  complex_check_floating(real, imag);
  // 获取与实部张量相同的选项，并将数据类型设置为对应的复数类型
  c10::TensorOptions options = real.options();
  options = options.dtype(toComplexType(real.scalar_type()));
  // 创建一个空的张量，用于存储复数结果，并调用复数输出函数
  Tensor result = at::empty(0, options);
  return at::complex_out(result, real, imag);
}

// 极坐标的输出函数，根据绝对值和角度创建结果张量
Tensor& polar_out(const Tensor& abs, const Tensor& angle, Tensor& result) {
  complex_check_dtype(result, abs, angle);
  // 设置迭代器配置，指定输出结果张量，绝对值和角度作为输入，并关闭相同数据类型的检查
  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_const_input(abs)
      .add_const_input(angle)
      .check_all_same_dtype(false)
      .build();
  // 调用极坐标操作的 stub 函数，根据设备类型执行相应的操作
  polar_stub(iter.device_type(), iter);
  return result;
}

// 创建极坐标张量，根据绝对值和角度张量返回极坐标张量
Tensor polar(const Tensor& abs, const Tensor& angle) {
  complex_check_floating(abs, angle);
  // 获取与绝对值张量相同的选项，并将数据类型设置为对应的复数类型
  c10::TensorOptions options = abs.options();
  options = options.dtype(toComplexType(abs.scalar_type()));
  // 创建一个空的张量，用于存储极坐标结果，并调用极坐标输出函数
  Tensor result = at::empty(0, options);
  return at::polar_out(result, abs, angle);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 创建一个在 CPU 上的空张量，根据给定的大小和其他可选参数
Tensor empty_cpu(IntArrayRef size, std::optional<ScalarType> dtype_opt, std::optional<Layout> layout_opt,
                 std::optional<Device> device_opt, std::optional<bool> pin_memory_opt, std::optional<c10::MemoryFormat> memory_format_opt) {
  // 调用 detail::empty_cpu 函数创建空张量
  Tensor result = at::detail::empty_cpu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  // 查看是否需要启用确定性操作
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    // 调用函数 fill_empty_deterministic_，传入 result 作为参数，用于填充结果集
    fill_empty_deterministic_(result);
  }
  // 返回填充后的结果集 result
  return result;
Tensor empty_names(
    IntArrayRef size,  // 接受一个整数数组作为尺寸参数
    std::optional<DimnameList> names,  // 可选的维度名称列表
    std::optional<ScalarType> dtype,  // 可选的张量数据类型
    std::optional<Layout> layout,  // 可选的张量布局
    std::optional<Device> device,  // 可选的设备类型
    std::optional<bool> pin_memory,  // 可选的内存固定标志
    optional<MemoryFormat> optional_memory_format) {  // 可选的内存格式

  // 查看[注意：关于TensorOptions的hacky wrapper去除]
  // 使用给定的dtype、layout、device和pin_memory参数构建TensorOptions对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 如果没有提供names参数，则创建一个不带名称的张量
  if (!names.has_value()) {
    return at::empty(size, options, optional_memory_format);
  }

  // 如果使用了命名张量，确保布局为Strided
  TORCH_CHECK(options.layout() == Layout::Strided,
      "NYI: named tensors only support strided layout");

  // 检查设备类型是否为CPU、CUDA、XPU或私有使用1后端
  TORCH_CHECK(options.device().is_cpu() || options.device().is_cuda() || options.device().is_xpu() || options.device().is_privateuseone(),
      "NYI: named tensors only support CPU, CUDA, XPU or ", c10::get_privateuse1_backend(), " tensors.");

  // 使用给定的选项创建一个空张量
  auto result = at::empty(size, options, optional_memory_format);

  // 在结果张量上设置名称
  internal_set_names_inplace(result, names);

  // 返回设置了名称的张量
  return result;
}

Tensor empty_permuted_symint(
  SymIntArrayRef size,  // 接受一个符号整数数组作为尺寸参数
  IntArrayRef physical_layout,  // 物理布局数组，指示逻辑维度顺序
  std::optional<ScalarType> dtype_opt,  // 可选的张量数据类型
  std::optional<Layout> layout_opt,  // 可选的张量布局
  std::optional<Device> device_opt,  // 可选的设备类型
  std::optional<bool> pin_memory_opt  // 可选的内存固定标志
) {
  // size 是逻辑尺寸；即操作后的输出尺寸
  //
  // physical_layout 遵循 NCHW/NHWC 约定：
  // contiguous 是 [0,1,2,3]，channels last 是 [0,2,3,1]
  //
  // 这意味着如果 i 是物理索引，physical_layout[i] 是逻辑索引；
  // 例如，要找到最内层的物理维度（3），查询 NHWC[3] == 1
  // （即它是通道维度）
  int64_t dim = static_cast<int64_t>(size.size());
  SymDimVector phys_size(dim);

  // 检查物理布局长度与尺寸数组长度是否匹配
  TORCH_CHECK(static_cast<int64_t>(physical_layout.size()) == dim,
    "Number of dimensions in size does not match the "
    "length of the physical_layout; i.e. len(size) = ", dim,
    " is not equal to len(physical_layout) = ", physical_layout.size());

  std::vector<bool> seen_dims(dim);

  // 遍历物理布局数组，生成物理尺寸数组
  for (const auto i : c10::irange(dim)) {
    TORCH_CHECK(physical_layout[i] >= 0 && physical_layout[i] < dim,
      "Dimension out of range (expected to be between 0 and ", dim - 1, ", but got ",
      physical_layout[i], " at index ", i, ").  NB: negative dims "
      "not currently supported; file an issue if you want it.");
    TORCH_CHECK(!seen_dims[physical_layout[i]], "Duplicate dim not allowed");
    phys_size[i] = size[physical_layout[i]];
    seen_dims[physical_layout[i]] = true;
  }

  // 执行连续的分配
  Tensor phys_tensor = at::empty_symint(phys_size, dtype_opt, layout_opt, device_opt, pin_memory_opt, c10::nullopt);

  // 获取物理张量的符号步长
  SymIntArrayRef phys_strides = phys_tensor.sym_strides();

  // 对步长进行排列（逆排列！这就是为什么是 empty_permute*d*，而不是 empty_permute；这不是空 + 排列）
  SymDimVector strides(dim);
  for (const auto i : c10::irange(dim)) {
    // 使用物理布局数组中的索引 i 找到对应的物理布局值，并将其作为键，
    // 将 phys_strides 数组中的第 i 个元素作为值存入 strides 字典中
    strides[physical_layout[i]] = phys_strides[i];
    
    // 返回一个通过给定大小和步幅创建的新的符号化张量
    return phys_tensor.as_strided_symint(size, strides);
}

// 返回一个新的 Tensor，具有给定的大小和步长，以及可选的数据类型、布局、设备和针对内存的固定选项
Tensor empty_strided_cpu(IntArrayRef size, IntArrayRef stride, std::optional<ScalarType> dtype_opt,
                         std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt) {
  // 调用 ATen 库的具体函数来创建一个具有指定大小和步长的 Tensor
  Tensor result = at::detail::empty_strided_cpu(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  // 如果全局上下文启用了确定性算法和填充未初始化内存的确定性策略
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    // 填充 Tensor 中未初始化的内存块
    fill_empty_deterministic_(result);
  }
  // 返回创建的 Tensor
  return result;
}

// 返回给定大小的 Tensor，如果提供了内存格式参数，将会检查是否与输出 Tensor 参数兼容
Tensor& empty_out(IntArrayRef size,
    std::optional<c10::MemoryFormat> optional_memory_format,
    Tensor& result) {
  // 检查是否提供了与 'out' Tensor 参数不兼容的 'memory_format' 参数
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "'memory_format' argument is incompatible with 'out' tensor argument");
  // 检查 Tensor 大小是否为非负数
  check_size_nonnegative(size);
  // 如果输出 Tensor 是稀疏 Tensor
  if (result.is_sparse()) {
    // 调整稀疏 Tensor 的大小并清空其内容
    result.sparse_resize_and_clear_(size, size.size(), 0);
  } else {
    // 调整 Tensor 的大小
    result.resize_(size);
  }
  // 如果全局上下文启用了确定性算法和填充未初始化内存的确定性策略
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    // 填充 Tensor 中未初始化的内存块
    fill_empty_deterministic_(result);
  }
  // 返回输出的 Tensor
  return result;
}

// 定义临时的类型转换操作符，用于对 Tensor 进行类型转换
// 当我们在 IR 中没有类型支持时，需要调用这些专门的操作符来跟踪类型转换
// TODO: 当 IR 中支持类型时，可以删除这部分
#define DEFINE_CAST_OP(_1, n)                                    \
  Tensor _cast_##n(const Tensor& self, bool non_blocking) {      \
    // 如果 Tensor 的数据类型已经是 ScalarType::n，则直接返回自身
    if (self.scalar_type() == ScalarType::n)                     \
      return self;                                               \
    // 否则，将 Tensor 转换为 ScalarType::n 的数据类型，并返回结果
    return self.to(ScalarType::n, non_blocking);                 \
  }

// 某些 CAST_OP 中的标量类型可能在 PyTorch 中未使用，但我们保留它们并在此处忽略警告，直到将来验证
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wmissing-prototypes")
// 对所有标量类型（除了 Bool、Half、BFloat16）调用 DEFINE_CAST_OP 宏定义
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CAST_OP)
C10_DIAGNOSTIC_POP()

#undef DEFINE_CAST_OP

// 返回一个新的 Tensor，具有与给定 Tensor 相同的大小和类型，可以选择指定数据类型、布局、设备和针对内存的固定选项
Tensor empty_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    // 根据传入的参数创建 TensorOptions 对象，设置数据类型、布局、设备和固定内存等选项
    TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
    
    // 将当前张量的选项与新创建的选项合并，同时合并可选的内存格式选项
    TensorOptions options =
        self.options()
            .merge_in(options_)
            .merge_memory_format(optional_memory_format);
    
    // 检查是否存在布局与内存格式选项不兼容的情况，抛出异常信息
    TORCH_CHECK(
        !(options.layout() != kStrided &&
            optional_memory_format.has_value()),
        "memory format option is only supported by strided tensors");
    
    // 获取最终确定的内存格式选项，如果未指定则使用默认的保持内存格式
    auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);
    
    // 声明一个 Tensor 类型的变量 result
    Tensor result;
    
    // 根据最终确定的内存格式选项选择相应的操作路径来创建新的 Tensor
    if (memory_format == MemoryFormat::Preserve) {
      // 如果保持内存格式，则根据条件选择合适的函数来创建新的 Tensor
      if (self.is_non_overlapping_and_dense()) {
        // 如果输入张量是非重叠且密集的，使用指定的内存格式创建对称整型的空张量
        result = at::empty_strided_symint(self.sym_sizes(), self.sym_strides(), options.memory_format(c10::nullopt));
      } else if (self.unsafeGetTensorImpl()->support_as_strided() && self.layout() == kStrided) {
        // 如果输入张量不是密集且非重叠，但是是分块的，推断出一个保持输入布局排列的输出步幅
        std::vector<int64_t> strides = infer_dense_strides(self.sizes(), self.strides());
        // 使用推断出的步幅创建新的分块张量，使用指定的内存格式
        result = at::empty_strided(self.sizes(), strides, options.memory_format(c10::nullopt));
      } else {
        // 如果以上条件都不满足，使用指定的内存格式创建对称整型的空张量
        result = at::empty_symint(self.sym_sizes(), options.memory_format(self.suggest_memory_format()), c10::nullopt);
      }
    } else {
      // 如果不是保持内存格式，则根据指定的内存格式选项创建对称整型的空张量
      result = at::empty_symint(self.sym_sizes(), options.memory_format(memory_format), c10::nullopt);
    }
    
    // 如果输入张量有名称信息，将名称信息传播到新创建的结果张量中
    if (self.opt_names()) {
      namedinference::propagate_names(result, self.names());
    }
    
    // 设置新创建张量的 dispatch key 为非共轭、非负、非零张量
    result._set_conj(false);
    result._set_neg(false);
    result._set_zero(false);
    
    // 返回最终创建的结果张量
    return result;
  // 创建一个新的 TensorOptions 对象，根据传入的 dtype、layout、device、pin_memory 参数设置选项
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 检查是否同时在 TensorOptions 和 explicit argument 中设置了 memory_format，如果是则报错
  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");

  // 将传入 Tensor 的 options 与上面创建的 options_ 合并，并且考虑 optional_memory_format
  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  // 如果 TensorOptions 指定了非 strided layout，同时又有 optional_memory_format，报错
  TORCH_CHECK(
      !(options.layout() != kStrided &&
          optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");

  // 获取最终确定的 memory_format，如果是 Preserve，则推荐使用 suggest_memory_format()
  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);


  // TODO: 为了支持 MemoryFormat::Preserve 的所有特性，需要添加 _empty_affine_quantized_strided 函数，并且类似于 Tensor 的 clone() 函数使用它
  // 如果当前 Tensor 是 kPerTensorAffine 格式，则根据 suggest_memory_format() 建议的格式来设置 memory_format
  if (memory_format == MemoryFormat::Preserve) {
    memory_format = self.suggest_memory_format();
  }


  // Note [Explicit nullopt MemoryFormat argument]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 有些函数的可选 MemoryFormat 参数默认不是 nullopt。如果我们通过 TensorOptions 传递 MemoryFormat，
  // 我们必须显式地禁用此默认过程，通过在 MemoryFormat 参数处显式地传递 nullopt。当代码生成调整后，
  // 我们可以从方法签名中删除此参数，参数将完全消失。

  // 我们应该检查 dtype 是否仍然是 quantized 吗？但是这样我们应该移动/缩放 q_zero_point / q_scale 吗？
  // 检查通过 TensorOptions 指定的 dtype 是否与输入 Tensor 的 dtype 匹配，如果不匹配则报错
  TORCH_CHECK(!options.has_dtype() || options.dtype() == self.dtype(),
              "It is currently not supported to specify a dtype that doesn't match "
              "the input tensor's dtype via empty_like.  Specified: ", options.dtype(),
              " Input tensor's dtype: ", self.dtype());

  // 获取当前 Tensor 的量化方案（qscheme）
  auto qscheme = self.qscheme();
  // 如果 qscheme 是 kPerTensorAffine，则执行以下代码块
  if (qscheme == kPerTensorAffine) {
    // 如果量化方案是 kPerTensorAffine
    return at::_empty_affine_quantized(self.sizes(), options.memory_format(memory_format),
                                        self.q_scale(),
                                        self.q_zero_point(),
                                        // 查看注释 [显式 nullopt MemoryFormat 参数]
                                        c10::nullopt);
  } else if (qscheme == kPerChannelAffine) {
    // 复制具有通道的张量，以避免意外覆盖
    return at::_empty_per_channel_affine_quantized(
        self.sizes(),
        self.q_per_channel_scales().clone(at::MemoryFormat::Preserve),
        self.q_per_channel_zero_points().clone(at::MemoryFormat::Preserve),
        self.q_per_channel_axis(),
        options.memory_format(memory_format),
        // 查看注释 [显式 nullopt MemoryFormat 参数]
        c10::nullopt);
  } else {
    // 如果量化方案不支持，则抛出错误信息
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qscheme));
  }
}

// 函数：创建一个空的符号整数 Tensor
Tensor new_empty_symint(
    const Tensor& self,                          // 输入的 Tensor 引用
    SymIntArrayRef size,                         // 符号整数数组的引用，用于指定 Tensor 的大小
    std::optional<ScalarType> dtype_opt,         // 可选的数据类型标量
    std::optional<Layout> layout_opt,            // 可选的布局选项
    std::optional<Device> device_opt,            // 可选的设备选项
    std::optional<bool> pin_memory_opt           // 可选的固定内存选项
    ) {
  auto dtype = dtype_opt.has_value() ? dtype_opt : optTypeMetaToScalarType(self.options().dtype_opt());
  // 如果指定了 dtype，则使用指定的值；否则从 self 的选项中获取默认的数据类型
  auto layout = layout_opt.has_value() ? layout_opt : self.options().layout_opt();
  // 如果指定了 layout，则使用指定的值；否则从 self 的选项中获取默认的布局
  auto device = device_opt.has_value() ? device_opt : self.options().device_opt();
  // 如果指定了 device，则使用指定的值；否则从 self 的选项中获取默认的设备
  auto pin_memory = pin_memory_opt.has_value() ? pin_memory_opt : self.options().pinned_memory_opt();
  // 如果指定了 pin_memory，则使用指定的值；否则从 self 的选项中获取是否固定内存的设置
  return at::empty_symint(size, dtype, layout, device, pin_memory, c10::nullopt);
  // 调用 ATen 库的 empty_symint 函数创建一个空的符号整数 Tensor，并返回
}

// 函数：创建一个带步长的空的符号整数 Tensor
Tensor new_empty_strided_symint(
    const Tensor& self,                          // 输入的 Tensor 引用
    c10::SymIntArrayRef size,                    // 符号整数数组的引用，用于指定 Tensor 的大小
    c10::SymIntArrayRef stride,                  // 符号整数数组的引用，用于指定 Tensor 的步长
    std::optional<ScalarType> dtype,             // 可选的数据类型标量
    std::optional<Layout> layout,                // 可选的布局选项
    std::optional<Device> device,                // 可选的设备选项
    std::optional<bool> pin_memory               // 可选的固定内存选项
    ) {
  // See [Note: hacky wrapper removal for TensorOptions]
  // 根据 TensorOptions 的设置创建 options 对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  return at::empty_strided_symint(size, stride, self.options().merge_in(options));
  // 调用 ATen 库的 empty_strided_symint 函数创建一个带步长的空的符号整数 Tensor，并返回
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ eye ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 函数：创建一个单位矩阵的 Tensor，形状为 n x n
Tensor eye(int64_t n,
    std::optional<ScalarType> dtype,             // 可选的数据类型标量
    std::optional<Layout> layout,                // 可选的布局选项
    std::optional<Device> device,                // 可选的设备选项
    std::optional<bool> pin_memory) {            // 可选的固定内存选项
  // 默认情况下，m 的值等于 n
  return at::eye(n, n, dtype, layout, device, pin_memory);
  // 调用 ATen 库的 eye 函数创建一个 n x n 的单位矩阵 Tensor，并返回
}

// 函数：创建一个指定形状的单位矩阵的 Tensor
Tensor eye(int64_t n, int64_t m,
    std::optional<ScalarType> dtype,             // 可选的数据类型标量
    std::optional<Layout> layout,                // 可选的布局选项
    std::optional<Device> device,                // 可选的设备选项
    std::optional<bool> pin_memory) {            // 可选的固定内存选项
  // See [Note: hacky wrapper removal for TensorOptions]
  // 根据 TensorOptions 的设置创建 options 对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto tensor = at::empty({0}, options);         // 创建一个空的 Tensor 以便后续调整大小
  return at::eye_out(tensor, n, m);              // 调用 ATen 库的 eye_out 函数生成一个 n x m 的单位矩阵，并返回
}

// 函数：在 CPU 上创建一个 n x n 的单位矩阵的 Tensor
Tensor& eye_out_cpu(int64_t n, Tensor& result) {
  // 默认情况下，m 的值等于 n
  return native::eye_out_cpu(n, n, result);
  // 调用 native 命名空间中的 eye_out_cpu 函数创建一个 n x n 的单位矩阵的 Tensor，并返回结果的引用
}

// 函数：在 CPU 上创建一个指定形状的单位矩阵的 Tensor
Tensor& eye_out_cpu(int64_t n, int64_t m, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);   // 检查 n 的值必须大于等于 0
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);   // 检查 m 的值必须大于等于 0

  result.resize_({n, m});                       // 调整 result 的大小为 n x m

  if (result.is_meta()) return result;          // 如果 result 是元数据，则直接返回 result

  result.zero_();                               // 将 result 的所有元素置为零

  int64_t sz = std::min<int64_t>(n, m);         // 计算 n 和 m 中的较小值
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBFloat16, kHalf, kBool, result.scalar_type(), "eye", [&]() -> void {
    scalar_t* result_data = result.data_ptr<scalar_t>();    // 获取 result 的数据指针
    at::parallel_for(0, sz, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      // 使用并行方式设置对角线上的元素为 1
      for (const auto i : c10::irange(p_begin, p_end))
        result_data[i*(result.strides()[0] + result.strides()[1])] = 1;
    });
  });

  return result;                                // 返回设置好的 result 引用
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ full ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {
// 推断填充满数据时的数据类型选项
TensorOptions infer_full_options(
  const Scalar& fill_value,
  const TensorOptions& options) {

  // 如果未指定数据类型
  if (!options.has_dtype()) {
    // 如果填充值是布尔类型
    if (fill_value.isBoolean()) {
      // 返回具有布尔数据类型的选项
      return options.dtype(at::kBool);
    } else if (fill_value.isIntegral(false)) {  // 如果填充值是整数类型（不考虑布尔类型）
      // 返回具有长整型数据类型的选项
      return options.dtype(at::kLong);
    } else if (fill_value.isComplex()) {  // 如果填充值是复数类型
      // 根据默认的双精度或单精度复数数据类型选择合适的标量类型
      auto scalar_type = (get_default_dtype() == ScalarType::Double) ?
                            ScalarType::ComplexDouble :
                            ScalarType::ComplexFloat;
      // 返回具有复数数据类型的选项
      return options.dtype(scalar_type);
    } else {
      // 返回具有默认数据类型的选项
      return options.dtype(get_default_dtype());
    }
  }

  // 如果已经指定了数据类型，则直接返回原始选项
  return options;
}

} // 匿名命名空间结束

// 创建一个指定大小的全填充张量
Tensor full(IntArrayRef size, const Scalar& fill_value,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 构建张量选项，包括数据类型、布局、设备以及是否固定内存等信息
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 检查布局是否为稀疏布局，若是，则抛出异常
  TORCH_CHECK(options.layout() != kSparse,
    "full(...) is not implemented for sparse layout");

  // 创建一个未初始化的指定大小的张量，并使用推断的选项填充它
  auto result = at::empty(size, infer_full_options(fill_value, options));
  return result.fill_(fill_value);  // 使用指定的填充值填充张量并返回
}

// 在现有张量上进行全填充操作
Tensor& full_out(IntArrayRef size, const Scalar& fill_value, Tensor& result) {
  // 检查结果张量是否为稀疏张量，若是，则抛出异常
  TORCH_CHECK(!result.is_sparse(),
    "full(...) is not implemented for sparse layout");

  // 调整结果张量的大小，并使用指定的填充值填充它
  result.resize_(size);
  return result.fill_(fill_value);  // 返回填充后的结果张量的引用
}

// 创建与给定张量相同大小的全填充张量
Tensor full_like(
    const Tensor& self,
    const Scalar& fill_value,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 构建张量选项，包括数据类型、布局、设备以及是否固定内存等信息
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 使用与给定张量相同的大小和选项创建一个未初始化的张量，并使用指定的填充值填充它
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.fill_(fill_value);  // 返回填充后的结果张量
}

// 在现有张量的基础上创建一个新的全填充张量
Tensor new_full(
    const Tensor& self,
    IntArrayRef size,
    const Scalar& fill_value,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory
    ) {
  // 使用给定的数据类型、布局、设备以及固定内存信息创建一个未初始化的张量
  Tensor r = self.new_empty(size, TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory));
  // 使用指定的填充值填充新创建的张量
  r.fill_(fill_value);
  return r;  // 返回填充后的张量
}

namespace {
// 推断线性或对数空间张量的选项
TensorOptions linspace_logspace_infer_options(
    const Scalar& start,
    const Scalar& end,
    const TensorOptions& options,
    const char* fn_name) {
  // 如果起始或结束值为复数，则选择默认的复数数据类型
  if (start.isComplex() || end.isComplex()) {
    const auto default_complex_dtype = c10::get_default_complex_dtype();
    // 返回具有复数数据类型的选项

    return options.dtype(default_complex_dtype);
  }

  // 如果起始和结束值都不是复数，则返回原始的选项
  return options;
}
    # 检查选项中是否指定了数据类型（dtype）
    if (options.has_dtype()) {
      # 如果指定了数据类型，将其转换为对应的标量类型
      auto dtype = c10::typeMetaToScalarType(options.dtype());
      # 检查转换后的数据类型是否为复数类型
      TORCH_CHECK(at::isComplexType(dtype),
          fn_name, ": inferred dtype ", default_complex_dtype, " can't be safely cast to passed dtype ", dtype);
    } else {
      # 如果选项中未指定数据类型，则返回默认复数数据类型
      return options.dtype(default_complex_dtype);
    }
  }

  # 如果选项中已指定数据类型，则返回选项本身；否则将默认复数数据类型设置为选项的数据类型并返回选项
  return options.has_dtype() ? options : options.dtype(c10::get_default_dtype());
} // anonymous namespace
    // 创建一个 TensorOptions 对象，设置数据类型、布局、设备，并根据是否固定内存设置 pinned_memory
    TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

    // 使用 TORCH_CHECK 确保 steps 大于等于 0，否则抛出异常 "number of steps must be non-negative"
    TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

    // 调用 linspace_logspace_infer_options 函数推断并返回结果的 TensorOptions
    auto result_options = linspace_logspace_infer_options(start, end, options, "torch.logspace()");

    // 创建一个形状为 {steps} 的空 Tensor，并使用推断的 TensorOptions 初始化
    Tensor result = at::empty({steps}, result_options);

    // 调用 at::logspace_out 函数填充结果 Tensor，并返回填充后的 Tensor
    return at::logspace_out(result, start, end, steps, base);
}

Tensor logspace(
    const Tensor& start,
    const Tensor& end,
    int64_t steps,
    double base,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 检查起始和结束张量是否为0维，否则抛出错误
  TORCH_CHECK(start.dim() == 0 && end.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s) and end with ", end.dim()," dimension(s).");
  // 调用ATen库的logspace函数，返回对数空间张量
  return at::logspace(start.item(), end.item(), steps, base, dtype, layout, device, pin_memory);
}

Tensor logspace(
    const Tensor& start,
    const Scalar& end,
    int64_t steps,
    double base,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 检查起始张量是否为0维，否则抛出错误
  TORCH_CHECK(start.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s).");
  // 调用ATen库的logspace函数，返回对数空间张量
  return at::logspace(start.item(), end, steps, base, dtype, layout, device, pin_memory);
}

Tensor logspace(
    const Scalar& start,
    const Tensor& end,
    int64_t steps,
    double base,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 检查结束张量是否为0维，否则抛出错误
  TORCH_CHECK(end.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got end with ", end.dim()," dimension(s).");
  // 调用ATen库的logspace函数，返回对数空间张量
  return at::logspace(start, end.item(), steps, base, dtype, layout, device, pin_memory);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor ones(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用native命名空间下的full函数，返回指定形状的张量，填充值为1.0
  return native::full(size, /*fill_value=*/1., dtype, layout, device, pin_memory);
}

Tensor& ones_out(IntArrayRef size, Tensor& result) {
  // 调用native命名空间下的full_out函数，填充指定形状的张量为1.0，并将结果存入result中
  return native::full_out(size, /*fill_value=*/1., result);
}

Tensor ones_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 创建与输入张量self相同形状的空张量
  auto result = at::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
  // 将结果张量填充为1.0
  return result.fill_(1.);
}

Tensor new_ones(
    const Tensor& self,
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 创建一个新的空张量，形状与输入self和指定size相同，数据类型、布局和设备与self相同，填充为1.0
  Tensor r = self.new_empty(size, TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory));
  r.fill_(1.);
  return r;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ scalar_tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor scalar_tensor(const Scalar& s,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 根据给定的 dtype、layout、device 和 pin_memory 创建 TensorOptions 对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 如果选项指定的设备是 CPU
  if (options.device() == at::kCPU) {
    // 这是一个快速路径，用于在 CPU 上创建标量张量时跳过设备分发。
    // 详细性能差异请参见 https://github.com/pytorch/pytorch/pull/29915
    // 在将来，当我们消除设备分发的开销后，我们将乐意恢复到以下方式：
    //   auto result = at::empty({}, options);
    // 禁用追踪器分发模式
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    // 进入自动分发模式下自动梯度之下
    at::AutoDispatchBelowAutograd mode;
    // 在 CPU 上创建一个空的张量
    auto result = empty_cpu({}, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
    // 用标量 s 填充结果张量
    at::native::fill_(result, s);
    // 返回填充后的结果张量
    return result;
  }
  // 在给定选项下创建一个空张量，并用标量 s 填充
  return at::empty({}, options).fill_(s);
}
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ rand ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 生成指定大小的随机张量
Tensor rand(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 使用默认生成器调用具体实现函数
  return native::rand(size, static_cast<std::optional<Generator>>(c10::nullopt), dtype, layout, device, pin_memory);
}

// 生成指定大小的随机张量，可以指定生成器
Tensor rand(IntArrayRef size, std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 创建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 生成空的张量
  auto result = at::empty(size, options);
  // 在张量中填充均匀分布的随机数
  return result.uniform_(0, 1, std::move(generator));
}

// 将生成的随机张量存储到指定的结果张量中
Tensor& rand_out(IntArrayRef size, Tensor& result) {
  // 调用具体实现函数
  return native::rand_out(size, c10::nullopt, result);
}

// 将生成的随机张量存储到指定的结果张量中，可以指定生成器
Tensor& rand_out(IntArrayRef size, std::optional<Generator> generator, Tensor& result) {
  // 调整结果张量的大小
  result.resize_(size);
  // 在结果张量中填充均匀分布的随机数
  return result.uniform_(0, 1, std::move(generator));
}

// 生成与指定张量相同大小的随机张量
Tensor rand_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 创建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 生成与输入张量相同大小的空张量
  auto result = at::empty_like(self, options, optional_memory_format);
  // 在结果张量中填充均匀分布的随机数，使用默认生成器
  return result.uniform_(0, 1, c10::nullopt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 生成指定范围内的随机整数张量
Tensor randint(int64_t high, IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用具体实现函数，不指定生成器
  return native::randint(high, size, c10::nullopt /* generator*/, dtype, layout, device, pin_memory);
}

// 生成指定范围内的随机整数张量，可以指定生成器
Tensor randint(
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用具体实现函数，指定生成器
  return native::randint(0, high, size, std::move(generator), dtype, layout, device, pin_memory);
}

// 生成指定范围内的随机整数张量，不指定生成器
Tensor randint(
    int64_t low,
    int64_t high,
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用具体实现函数，不指定生成器
  return native::randint(low, high, size, c10::nullopt, dtype, layout, device, pin_memory);
}

// 生成指定范围内的随机整数张量，可以指定生成器
Tensor randint(
    int64_t low,
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用具体实现函数，指定生成器
  return native::randint(low, high, size, std::move(generator), dtype, layout, device, pin_memory);
}
    std::optional<bool> pin_memory) {
```  
# 接收一个可选的布尔值参数 `pin_memory`，表示是否使用固定内存。

  // See [Note: hacky wrapper removal for TensorOptions]

  // 参考注释：用于处理 TensorOptions 的不优雅包装的移除
```  
# 此注释提醒读者参考有关如何处理 `TensorOptions` 的说明，特别是移除不优雅包装的部分。

  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 创建 TensorOptions 对象，并设置数据类型、布局、设备以及是否固定内存
```  
# 创建一个 `TensorOptions` 对象，并设置其数据类型为 `dtype`，布局为 `layout`，设备为 `device`，是否固定内存由 `pin_memory` 决定。


  auto result = at::empty(size, options);
```  
# 使用指定的 `TensorOptions` 创建一个大小为 `size` 的空张量 `result`。


  return result.random_(low, high, std::move(generator));
```  
# 在张量 `result` 上调用 `random_` 方法，生成指定范围 `[low, high)` 内的随机数，并使用移动语义传递 `generator`。
}

// 返回指定范围内的随机整数张量，存储到给定的结果张量中
Tensor& randint_out(int64_t high, IntArrayRef size, Tensor& result) {
  return native::randint_out(high, size, c10::nullopt, result);
}

// 返回指定范围内的随机整数张量，存储到给定的结果张量中，并可指定随机数生成器
Tensor& randint_out(int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    Tensor& result) {
  result.resize_(size); // 调整结果张量的大小
  return result.random_(0, high, std::move(generator)); // 在指定范围内生成随机整数填充结果张量
}

// 返回指定范围内的随机整数张量，存储到给定的结果张量中
Tensor& randint_out(int64_t low, int64_t high, IntArrayRef size, Tensor& result) {
  return native::randint_out(low, high, size, c10::nullopt, result);
}

// 返回指定范围内的随机整数张量，存储到给定的结果张量中，并可指定随机数生成器
Tensor& randint_out(int64_t low,
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    Tensor& result) {
  result.resize_(size); // 调整结果张量的大小
  return result.random_(low, high, std::move(generator)); // 在指定范围内生成随机整数填充结果张量
}

// 返回与给定张量相同大小的随机整数张量
Tensor randint_like(
    const Tensor& self,
    int64_t high,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 根据给定的参数创建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty_like(self, options, optional_memory_format); // 根据self创建与其同样类型的空张量
  return result.random_(0, high, c10::nullopt); // 在指定范围内生成随机整数填充结果张量
}

// 返回与给定张量相同大小的随机整数张量
Tensor randint_like(
    const Tensor& self,
    int64_t low,
    int64_t high,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 根据给定的参数创建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty_like(self, options, optional_memory_format); // 根据self创建与其同样类型的空张量
  return result.random_(low, high, c10::nullopt); // 在指定范围内生成随机整数填充结果张量
}

// 返回指定大小的随机标准正态分布张量
Tensor randn(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randn(size, static_cast<std::optional<Generator>>(c10::nullopt), dtype, layout, device, pin_memory);
}

// 返回指定大小的随机标准正态分布张量，并可指定随机数生成器
Tensor randn(IntArrayRef size, std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 根据给定的参数创建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, options); // 创建指定大小的空张量
  return result.normal_(0, 1, std::move(generator)); // 用标准正态分布填充结果张量
}

// 返回与给定结果张量相同大小的随机标准正态分布张量
Tensor& randn_out(IntArrayRef size, Tensor& result) {
  return native::randn_out(size, c10::nullopt, result);
}

// 返回与给定结果张量相同大小的随机标准正态分布张量，并可指定随机数生成器
Tensor& randn_out(IntArrayRef size, std::optional<Generator> generator, Tensor& result) {
  result.resize_(size); // 调整结果张量的大小
  return result.normal_(0, 1, std::move(generator)); // 用标准正态分布填充结果张量
}
// 生成指定均值和标准差的正态分布随机数张量
Tensor normal(double mean, double std, IntArrayRef size,
              std::optional<Generator> generator,
              std::optional<ScalarType> dtype,
              std::optional<Layout> layout,
              std::optional<Device> device,
              std::optional<bool> pin_memory) {
  // 创建包含指定选项的张量选项对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 使用指定大小和选项创建一个未初始化的张量
  auto result = at::empty(size, options);
  // 填充张量中的数据，使用正态分布进行初始化
  return result.normal_(mean, std, std::move(generator));
}

// 在给定的张量上生成指定均值和标准差的正态分布随机数，结果存储在 result 中
Tensor& normal_out(double mean, double std,
                   IntArrayRef size, std::optional<Generator> generator, Tensor& result) {
  // 调整 result 的大小以适应新的尺寸
  result.resize_(size);
  // 填充 result 中的数据，使用正态分布进行初始化
  return result.normal_(mean, std, std::move(generator));
}

// 生成与给定张量相同大小的张量，其中每个元素都是从标准正态分布中随机抽取的
Tensor randn_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 创建包含指定选项的张量选项对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 创建一个与 self 大小相同的未初始化张量，并使用标准正态分布进行初始化
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.normal_(0, 1, c10::nullopt);
}

// 在 CPU 上生成一个长度为 n 的随机排列的张量
namespace {
template <typename scalar_t>
void randperm_cpu(Tensor& result, int64_t n, CPUGeneratorImpl* generator) {
  // 获取结果张量的数据指针
  scalar_t *r__data = result.data_ptr<scalar_t>();

  // 调整结果张量的大小为 n
  result.resize_({n});
  // 获取结果张量在第一维度上的步幅
  int64_t r__stride_0 = result.stride(0);

  // 并行地为结果张量填充随机排列的值
  at::parallel_for(0, n, internal::GRAIN_SIZE,
                  [&r__data, &r__stride_0](int64_t p_begin, int64_t p_end) {
    for (const auto i : c10::irange(p_begin, p_end)) {
      r__data[i*r__stride_0] = static_cast<scalar_t>(i);
    }
  });

  // 使用给定的生成器生成随机排列
  for(int64_t i = 0; i < n - 1; i++)
  {
    // 使用生成器获取一个随机数，用于交换位置
    int64_t z = generator->random() % (n-i);
    // 交换位置，生成随机排列
    scalar_t sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}
} // namespace

// 生成一个长度为 n 的随机排列的张量，结果张量的数据类型、布局、设备、是否锁定内存由参数指定
Tensor randperm(int64_t n,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用具体实现函数生成随机排列张量
  return native::randperm(n, c10::nullopt, dtype, layout, device, pin_memory);
}

// 生成一个长度为 n 的随机排列的张量，结果张量的数据类型、生成器、布局、设备、是否锁定内存由参数指定
Tensor randperm(int64_t n, std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 如果未指定数据类型，则默认使用 Long 类型
  if (!dtype.has_value()) {
    dtype = ScalarType::Long;
  }

  // 创建包含指定选项的张量选项对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 创建一个未初始化的张量，用于存储随机排列的结果
  auto tensor = at::empty(n, options);
  // 在给定的张量上生成长度为 n 的随机排列
  return at::randperm_out(tensor, n, std::move(generator));
}
// 返回一个随机的排列结果，存储在给定的张量 `result` 中，范围为 0 到 n-1
Tensor& randperm_out(int64_t n, Tensor& result) {
  // 调用 PyTorch 中的 randperm_out 函数生成随机排列，使用默认生成器
  return at::randperm_out(result, n, c10::nullopt);
}

// 在 CPU 上生成一个随机的排列，存储在给定的张量 `result` 中
Tensor& randperm_out_cpu(int64_t n, std::optional<Generator> generator, Tensor& result) {
  // 检查 n 是否为非负数
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  // 检查生成器是否存在，并且结果张量的设备与生成器的设备匹配
  TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()), "Expected a '", result.device(), "' generator device but found '", generator->device(), "'");
  // 检查结果张量的数据类型是否支持给定的最大整数范围
  check_supported_max_int_with_precision(n, result);
  // 调整结果张量的形状为 [n]
  result.resize_({n});
  // 获取 CPU 上的生成器
  auto gen = get_generator_or_default<CPUGeneratorImpl>(generator, detail::getDefaultCPUGenerator());
  // 获取生成器的互斥锁，确保并发安全
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // 使用 randperm_cpu 函数生成随机排列，针对不同的数据类型进行分发
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, result.scalar_type(), "randperm", [&]() -> void {
    randperm_cpu<scalar_t>(result, n, gen);
  });

  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ range ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 创建一个范围内的张量，从 start 到 end，使用指定的步长 step，以及其他选项
Tensor range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 根据提供的选项创建 TensorOptions 对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  // 创建一个空的张量 result，其形状为 {0}，使用指定的选项
  Tensor result = at::empty({0}, options);
  // 调用 at::range_out 函数生成具有指定范围和步长的张量，并将结果存储在 result 中
  return at::range_out(result, start, end, step);
}

// 创建一个范围内的张量，从 start 到 end，默认步长为 1，使用指定的选项
Tensor range(
    const Scalar& start,
    const Scalar& end,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用 native::range 函数，生成具有默认步长的范围张量
  return at::native::range(start, end, 1, dtype, layout, device, pin_memory);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 在 CPU 上生成一个上三角部分的索引张量，用于创建一个对角线偏移的三角矩阵
Tensor tril_indices_cpu(
    int64_t row, int64_t col, int64_t offset, std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt) {
  // 如果未指定数据类型，创建默认的 TensorOptions
  if (!dtype_opt.has_value()) {
  dtype_opt = ScalarType::Long;
  // 设置数据类型选项为长整型

  check_args(row, col, layout_opt);
  // 检查参数的有效性

  auto tril_size = get_tril_size(row, col, offset);
  // 计算下三角矩阵的大小

  // create an empty Tensor with correct size
  // 使用正确的大小创建一个空的 Tensor
  auto result = at::native::empty_cpu({2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  // The following three approaches result in very little performance
  // differences. Hence, the 2nd option is taken for simpler code, and to return
  // contiguous tensors. Refer to #14904 for more details.
  //
  // 1. sequential RAM access: fill row coordinates first, then columns. This
  //    results in two for-loop and more arithmetic operations.
  //
  // 2. interleaved RAM access: fill in index coordinates one by one, which
  //    jumps between the two output Tensor rows in every iteration.
  //
  // 3. sequential RAM + transpose: create an n X 2 Tensor, fill the Tensor
  //    sequentially, and then transpose it.
  //
  // 根据性能测试，选择第二种方法以获得更简单的代码和连续的张量返回。

  AT_DISPATCH_INDEX_TYPES(result.scalar_type(), "tril_indices", [&]() -> void {
    // fill the Tensor with correct values
    // 使用正确的值填充 Tensor
    index_t* result_data = result.data_ptr<index_t>();
    int64_t i = 0;

    index_t r = std::max<int64_t>(0, -offset), c = 0;
    while (i < tril_size) {
      result_data[i] = r;
      result_data[tril_size + i++] = c;

      // move to the next column and check if (r, c) is still in bound
      // 移动到下一列，并检查 (r, c) 是否仍在边界内
      c += 1;
      if (c > r + offset || c >= col) {
        r += 1;
        c = 0;
        // NOTE: not necessary to check if r is less than row here, because i
        // and tril_size provide the guarantee
        // 这里不需要检查 r 是否小于 row，因为 i 和 tril_size 已经提供了保证
      }
    }
  });

  return result;
  // 返回生成的 Tensor
// 定义一个函数 triu_indices_cpu，用于生成上三角矩阵的索引
Tensor triu_indices_cpu(
    int64_t row, int64_t col, int64_t offset, std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt) {

  // 如果未提供 dtype_opt，则设置默认值为 ScalarType::Long
  if (!dtype_opt.has_value()) {
    dtype_opt = ScalarType::Long;
  }

  // 检查传入参数的合法性，主要是行、列数和布局
  check_args(row, col, layout_opt);

  // 计算生成上三角矩阵的非零元素个数
  auto triu_size = row * col - get_tril_size(row, col, offset - 1);

  // 创建一个大小正确的空 Tensor
  auto result = at::native::empty_cpu({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  // 使用 AT_DISPATCH_INDEX_TYPES 宏根据索引类型分发操作
  AT_DISPATCH_INDEX_TYPES(result.scalar_type(), "triu_indices", [&]() -> void {
    // 获取结果 Tensor 的数据指针
    index_t* result_data = result.data_ptr<index_t>();
    int64_t i = 0;
    index_t c = std::max<int64_t>(0, offset), r = 0;

    // 填充结果 Tensor 的值
    while (i < triu_size) {
      result_data[i] = r;
      result_data[triu_size + i++] = c;

      // 移动到下一个列，并检查 (r, c) 是否仍在边界内
      c += 1;
      if (c >= col) {
        r += 1;
        // 更新列 c 的值，确保不越界
        c = std::max<int64_t>(0, r + offset);
      }
    }
  });

  // 返回生成的结果 Tensor
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ zeros ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 定义一个静态函数 zeros_sparse_compressed_symint，用于生成稀疏压缩的对称整数数组的零张量
static Tensor zeros_sparse_compressed_symint(c10::SymIntArrayRef size,
    std::optional<ScalarType> dtype,
    Layout layout,
    std::optional<Device> device,
    // 检查输入的尺寸是否非负
    check_size_nonnegative(size);
    // 确保尺寸大小至少为2，因为torch.zeros仅支持批处理稀疏压缩（非块）张量
    TORCH_CHECK(size.size() >= 2, "torch.zeros: Only batched sparse compressed (non-block) tensors are supported, but got size ", size);
    // 将size转换为数组形式
    auto size_ = C10_AS_INTARRAYREF_SLOW(size);
    // 如果使用torch.zeros创建块张量，会因为API不支持指定块大小而失败
    // 通过AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS调度非块布局的操作
    AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(layout, "zeros_sparse_compressed", [&]{});

    // 初始化非零元素计数为0
    int64_t nnz = 0;
    // 计算压缩索引的尺寸
    auto compressed_indices_size = DimVector(size_.slice(0, size.size() - 2));
    // 计算普通索引和值的尺寸
    auto plain_indices_and_values_size = DimVector(size_.slice(0, size.size() - 2));
    // 将压缩索引的最后一个维度大小设置为压缩维度大小加1
    compressed_indices_size.push_back(size_[at::sparse_csr::compressedDimension(layout, size_)] + 1);
    // 将普通索引和值的最后一个维度大小设置为0
    plain_indices_and_values_size.push_back(nnz);

    // 定义张量选项，设置张量为长整型、分布式、指定设备，并可用固定内存
    TensorOptions options = TensorOptions().dtype(ScalarType::Long).layout(Layout::Strided).device(device).pinned_memory(pin_memory);
    // 创建空的压缩索引张量并清零
    auto compressed_indices = at::empty(compressed_indices_size, options);
    compressed_indices.zero_();
    // 创建空的普通索引张量
    auto plain_indices = at::empty(plain_indices_and_values_size, options);
    // 创建空的值张量，其类型由输入的dtype决定
    auto values = at::empty(plain_indices_and_values_size, options.dtype(dtype));

    // 调用内部函数_at::_sparse_compressed_tensor_unsafe，返回稀疏压缩张量
    return at::_sparse_compressed_tensor_unsafe(compressed_indices,
                                                plain_indices,
                                                values,
                                                size_,
                                                dtype,
                                                layout,
                                                device,
                                                pin_memory);
}

// 返回一个形状为 size 的零张量，支持可选的数据类型、布局、设备和固定内存选项
Tensor zeros_symint(SymIntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 确定张量的布局，默认为 Strided
  Layout layout_ = layout.value_or(Layout::Strided);
  // 如果布局是稀疏 CSR 格式，则调用 zeros_sparse_compressed_symint 函数生成零张量
  if (at::sparse_csr::is_sparse_compressed(layout_)) {
    return zeros_sparse_compressed_symint(size, dtype, layout_, device, pin_memory);
  }
  // 创建张量选项，可以包括数据类型、布局、设备和固定内存选项
  // 注意：下面的 at::empty_symint 函数需要在其实现中考虑到这些选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  // 调用 at::empty_symint 函数创建一个空的合适形状的符号整数张量
  auto result = at::empty_symint(size, options);
  // 将张量的所有元素设为零，并返回引用
  return result.zero_();
}

// 返回一个形状为 size 的零张量，支持可选的数据类型、布局、设备和固定内存选项
Tensor _efficientzerotensor(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
    // 确定设备，默认使用默认设备
    auto device_ = device_or_default(device);
    // 创建一个特定设备的零张量分配器
    auto allocator = at::native::ZeroTensorAllocator(device_);
    // 确定数据类型，默认为浮点数类型
    auto dtype_ = dtype_or_default(dtype);
    // 定义零张量的分发键集合，包括 CPU 和 ZeroTensor
    auto zero_ks = at::DispatchKeySet(c10::DispatchKey::CPU) | at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
    // 调用 empty_generic 函数生成一个具有指定选项的零张量
    auto out = at::detail::empty_generic(size, &allocator, zero_ks, dtype_, c10::nullopt);
    return out;
}

// 返回一个形状为 size 的符号整数类型的零张量，支持可选的数据类型、布局、设备和固定内存选项
Tensor _efficientzerotensor_meta_symint(SymIntArrayRef size,
                                        std::optional<ScalarType> dtype,
                                        std::optional<Layout> layout,
                                        std::optional<Device> device,
                                        std::optional<bool> pin_memory) {
  // 确定设备，默认使用默认设备
  auto device_ = device_or_default(device);
  // 创建一个特定设备的零张量分配器
  auto allocator = at::native::ZeroTensorAllocator(device_);
  // 确定数据类型，默认为浮点数类型
  auto dtype_ = dtype_or_default(dtype);
  // 定义零张量的分发键集合，包括 Meta 和 ZeroTensor
  auto zero_ks = at::DispatchKeySet(c10::DispatchKey::Meta) | at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
  // 调用 empty_generic_symint 函数生成一个具有指定选项的符号整数类型的零张量
  auto out = at::detail::empty_generic_symint(size, &allocator, zero_ks, dtype_, c10::nullopt);
  return out;
}

// 根据给定的 size，调整 result 张量的大小，并将其设为稀疏张量并清零
Tensor& zeros_sparse_out(IntArrayRef size, Tensor& result) {
  // 如果结果张量是稀疏的，则调用 sparse_resize_and_clear_ 函数重新调整其大小并清零
  result.sparse_resize_and_clear_(size, size.size(), 0.);
  return result;
}

// 根据给定的 size，调整 result 张量的大小，并将其设为非稀疏张量，并返回对 result 的引用
Tensor& zeros_out(IntArrayRef size, Tensor& result) {
  // 如果结果张量是稀疏的，则调用 sparse_resize_and_clear_ 函数重新调整其大小并清零
  if (result.is_sparse()) {
    // 注意：这个分支目前认为是不可达的，但由于当前的稀疏内核限制，保留这个分支
    result.sparse_resize_and_clear_(size, size.size(), 0.);
    return result;
  } else {
    // 否则，直接调整结果张量的大小为指定的 size
    result.resize_(size);
  }
  // 将结果张量的所有元素设为零，并返回对 result 的引用
  return result.zero_();
}

// 返回与给定张量 self 相同形状的零张量，支持可选的数据类型、布局、设备和固定内存选项
Tensor zeros_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  // 创建另一个 TensorOptions 对象 other_options，用于包含给定的 dtype、layout、device 和 pin_memory 信息
  auto other_options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  
  // 合并其他选项（other_options）和 self 的选项（self.options()），优先使用显式传入的值，否则使用 self 中的默认值
  auto options = self.options().merge_in(other_options);

  // 如果选项的布局为稀疏（kSparse）
  if (options.layout() == kSparse) {
    // 如果 optional_memory_format 有值，则抛出错误，因为内存格式选项仅支持步进张量
    TORCH_CHECK(
        !(optional_memory_format.has_value()),
        "memory format option is only supported by strided tensors");
    
    // 创建一个空张量 res，用于稍后调整大小
    auto res = at::empty({0}, self.options().merge_in(options)); // to be resized

    // 如果 self 是稀疏张量，则调整 res 的大小并清空数据
    if (self.is_sparse()) {
      res.sparse_resize_and_clear_(
          self.sizes(), self.sparse_dim(), self.dense_dim());
    } else if (at::sparse_csr::is_sparse_compressed(self)) {
      // 如果 self 是压缩稀疏张量，则根据情况调整 res 的大小并清空数据
      res.sparse_resize_and_clear_(
          self.sizes(), self.sizes().size() - self.dense_dim(), self.dense_dim());
    } else {
      // 否则，将 res 调整为与 self 大小相同并清空数据
      res.sparse_resize_and_clear_(self.sizes(), self.sizes().size(), 0);
    }
    // 设置 res 为已合并状态
    res._coalesced_(true);

    return res; // 返回调整大小后的稀疏张量
  } else if (at::sparse_csr::is_sparse_compressed(options.layout())) {
    // 如果选项的布局表示为压缩稀疏张量
    int64_t nnz = 0;
    int64_t dense_dim = (self.layout() == kStrided ? self.dim() - 2: self.dense_dim());
    DimVector blocksize{};
    if (self.layout() == kSparseBsr || self.layout() == kSparseBsc) {
      blocksize.append(at::sparse_csr::getBlockSize(self));
    }
    ScalarType index_dtype = at::sparse_csr::getIndexDtype(self);
    
    // 创建一个带有指定维度的压缩稀疏张量 res
    auto res = at::native::sparse_compressed_tensor_with_dims(
      nnz, dense_dim, self.sizes(), blocksize, index_dtype,
      typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory());
    
    // 获取压缩和普通索引，然后将压缩索引置零
    Tensor compressed_indices, plain_indices;
    std::tie(compressed_indices, plain_indices) = at::sparse_csr::getCompressedPlainIndices(res);
    compressed_indices.zero_();
    
    return res; // 返回创建的压缩稀疏张量
  }
  
  // 如果以上条件都不满足，则创建一个与 self 类型和选项相同的空张量 result
  auto result = at::empty_like(self, options, optional_memory_format);
  
  // 将 result 中的数据置零并返回
  return result.zero_();
}
Tensor blackman_window(
    int64_t window_length,
    bool periodic,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 从 dtype_opt 中获取数据类型，如果未指定则使用默认类型
  ScalarType dtype = c10::dtype_or_default(dtype_opt);
  // 根据指定的选项创建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 执行窗口函数前的检查，确保参数合法性
  window_function_checks("blackman_window", options, window_length);
  
  // 如果窗口长度为 0，则返回一个空张量
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  
  // 如果窗口长度为 1，则返回一个包含单个元素的张量
  if (window_length == 1) {
    return native::ones({1}, dtype, layout, device, pin_memory);
  }
  
  // 如果指定为周期性窗口，则增加窗口长度
  if (periodic) {
    window_length += 1;
  }
  
  // 生成 Blackman 窗口的计算过程
  auto window = native::arange(window_length, dtype, layout, device, pin_memory)
                    .mul_(2. / static_cast<double>(window_length - 1));
  const int64_t first_half_size = ((window_length - 1) >> 1) + 1;
  window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2);
  
  // 如果是周期性窗口，则截取有效窗口部分并返回；否则直接返回生成的窗口
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}
    window_length += 1;
  // 从维基百科上的链接 https://en.wikipedia.org/wiki/Window_function#Blackman_window
  // 计算窗口函数，具体使用Blackman窗口函数
  auto window =
      native::arange(window_length, dtype, layout, device, pin_memory)
          .mul_(c10::pi<double> / static_cast<double>(window_length - 1));
  // 应用Blackman窗口函数的数学表达式
  window = window.mul(4).cos_().mul_(0.08) - window.mul(2).cos_().mul_(0.5) + 0.42;
  // 如果需要周期性窗口，则截取除最后一个元素外的所有元素；否则直接返回窗口
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hamming_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 返回一个汉明窗口函数的张量
Tensor hamming_window(int64_t window_length,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用 native 命名空间中的 hamming_window 函数，使用默认的周期性和指定的参数
  return native::hamming_window(
      window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
}

// 返回一个汉明窗口函数的张量
Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用 native 命名空间中的 hamming_window 函数，指定周期性、alpha 值，以及其它参数
  return native::hamming_window(
      window_length,
      periodic,
      /*alpha=*/0.54,
      dtype,
      layout,
      device,
      pin_memory);
}

// 返回一个汉明窗口函数的张量
Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    double alpha,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用 native 命名空间中的 hamming_window 函数，指定周期性、alpha 和 beta 值，以及其它参数
  return native::hamming_window(
      window_length, periodic, alpha, /*beta=*/0.46, dtype, layout, device, pin_memory);
}

// 返回一个汉明窗口函数的张量
Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    double alpha,
    double beta,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 获取 Tensor 的选项，根据参数设置 dtype、layout、device 和 pinned_memory
  ScalarType dtype = c10::dtype_or_default(dtype_opt);
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 检查窗口函数的参数是否有效
  window_function_checks("hamming_window", options, window_length);
  // 如果窗口长度为 0，则返回一个空张量
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  // 如果窗口长度为 1，则返回一个元素为 1 的张量
  if (window_length == 1) {
    return native::ones({1}, dtype, layout, device, pin_memory);
  }
  // 如果周期性为 true，则增加窗口长度
  if (periodic) {
    window_length += 1;
  }
  // 创建一个从 0 到 window_length-1 的张量
  auto window = native::arange(window_length, dtype, layout, device, pin_memory);
  // 对窗口张量应用汉明窗口函数的计算公式
  window.mul_(c10::pi<double> * 2. / static_cast<double>(window_length - 1)).cos_().mul_(-beta).add_(alpha);
  // 如果周期性为 true，则返回裁剪后的窗口张量；否则返回原始的窗口张量
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hann_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 返回一个汉宁窗口函数的张量
Tensor hann_window(int64_t window_length,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用 native 命名空间中的 hann_window 函数，使用默认的周期性和指定的参数
  return native::hann_window(window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
}

Tensor hann_window(
    int64_t window_length,
    bool periodic,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
    std::optional<bool> pin_memory) {
  // 创建TensorOptions对象并设置其属性：数据类型、布局、设备以及是否固定内存
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 调用window_function_checks函数，用于验证窗口函数参数的合法性
  window_function_checks("hann_window", options, window_length);
  // 调用native::hamming_window函数生成汉明窗口，并返回结果
  return native::hamming_window(
      window_length, periodic, /*alpha=*/0.5, /*beta=*/0.5, dtype, layout, device, pin_memory);
}
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ kaiser_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 创建 Kaiser 窗口函数的默认接口，返回一个张量
Tensor kaiser_window(int64_t window_length,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用底层函数 native::kaiser_window，使用默认参数 beta=12.0 和 periodic=true
  return native::kaiser_window(
      window_length,
      /*periodic=*/true,
      /*beta=*/12.0,
      dtype,
      layout,
      device,
      pin_memory);
}

// 创建 Kaiser 窗口函数的重载接口，允许指定是否周期性
Tensor kaiser_window(int64_t window_length, bool periodic,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用底层函数 native::kaiser_window，指定是否周期性和默认参数 beta=12.0
  return native::kaiser_window(window_length, periodic, /*beta=*/12.0, dtype, layout, device, pin_memory);
}

// 创建 Kaiser 窗口函数的重载接口，允许指定周期性、beta 值及其他选项
Tensor kaiser_window(
    int64_t window_length,
    bool periodic,
    double beta,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 获取数据类型，如果未指定则使用默认类型
  ScalarType dtype = c10::dtype_or_default(dtype_opt);
  // 构造张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 检查窗口函数的有效性
  window_function_checks("kaiser_window", options, window_length);
  // 如果设备类型为 kMeta，则返回一个空张量
  if (device == kMeta) {
    return at::empty({window_length}, options);
  }

  // 如果窗口长度为 0，则返回一个空张量
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  // 如果窗口长度为 1，则返回一个全为 1 的张量
  if (window_length == 1) {
    return at::ones({1}, options);
  }
  // 如果设置为周期性窗口，则增加窗口长度
  if (periodic) {
    window_length += 1;
  }
  // 创建初始序列
  auto initial = at::arange(window_length, options);
  // 创建用于存储窗口函数的张量
  auto window = at::empty(window_length, options);
  // 创建张量迭代器，并调用 stub 函数生成 Kaiser 窗口
  auto iter = TensorIterator::unary_op(window, initial);
  kaiser_window_stub(iter.device_type(), iter, window_length, beta);
  // 如果是周期性窗口，则返回除去最后一个元素的部分
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ vandermonde_matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 创建 Vandermonde 矩阵的函数
Tensor vander(const Tensor& x, std::optional<int64_t> N, bool increasing) {
  // 检查输入张量 x 是否为一维张量
  TORCH_CHECK(x.dim() == 1, "x must be a one-dimensional tensor.");

  // 获取矩阵的列数 n，如果未指定则使用 x 的长度
  int64_t n = x.size(0);
  if (N.has_value()) {
    n = *N;
    // 检查 n 是否为非负数
    TORCH_CHECK(n >= 0, "N must be non-negative.");
  }

  // 根据 x 的数据类型创建结果张量，如果 x 是整数张量，则结果为 long 类型
  auto result = at::empty({x.size(0), n}, x.options().dtype(at::promote_types(x.scalar_type(), c10::ScalarType::Long)));

  // 如果 n 大于 0，则初始化结果张量的第一列为 1
  if (n > 0) {
    result.select(1, 0).fill_(1);
  }
  // 如果 n 大于 1，则填充第二列为 x
  if (n > 1) {
    result.slice(1, 1).copy_(x.unsqueeze(1));
    // 对第二列进行累积乘积操作
    result.slice(1, 1).copy_(at::cumprod(result.slice(1, 1), 1));
  }

  // 如果不是递增顺序，则翻转结果张量
  if (!increasing) {
    return at::flip(result, {1});
  }
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 创建 CPU 端的张量，从给定的值数组和选项中
template <typename T>
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options) {
  // 调用 detail 命名空间中的 tensor_cpu 函数，创建 CPU 张量
  return at::detail::tensor_cpu(values, options);
}
// 使用模板实现，调用 detail 命名空间下的 tensor_backend 函数来创建一个 Tensor 对象
template <typename T>
Tensor tensor_backend(ArrayRef<T> values, const TensorOptions& options) {
  return at::detail::tensor_backend(values, options);
}

// 使用模板实现，调用 detail 命名空间下的 tensor_complex_cpu 函数来创建一个 Tensor 对象
template <typename T>
Tensor tensor_complex_cpu(ArrayRef<T> values, const TensorOptions& options) {
  return at::detail::tensor_complex_cpu(values, options);
}

// 使用模板实现，调用 detail 命名空间下的 tensor_complex_backend 函数来创建一个 Tensor 对象
template <typename T>
Tensor tensor_complex_backend(ArrayRef<T> values, const TensorOptions& options) {
  return at::detail::tensor_complex_backend(values, options);
}

// 根据文件名从文件中创建一个 Tensor 对象
Tensor from_file(c10::string_view filename, std::optional<bool> shared, std::optional<int64_t> size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 使用传入的参数构建 TensorOptions 对象
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

    // 检查是否有指定固定内存标志，如果有则抛出异常，因为从文件创建的张量不能固定内存
    TORCH_CHECK(!options.pinned_memory(), "tensors constructed from a file cannot be pinned");

    // 获取文件大小，如果未指定则默认为 0
    int64_t my_size = size.value_or(0);

    // 根据是否共享创建标志位
    int flags = shared.value_or(false) ? ALLOCATOR_MAPPED_SHARED : 0;

    // 获取 TensorOptions 中的数据类型
    auto my_dtype = options.dtype();

    // 计算存储空间的字节大小
    size_t size_bytes = my_size * my_dtype.itemsize();

    // 使用 MapAllocator 创建存储实现，其中包含文件名和相关标志位
    auto storage_impl = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        MapAllocator::makeDataPtr(
            std::string(filename), flags, size_bytes, nullptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    // 使用 detail 命名空间下的 make_tensor 函数创建 Tensor 对象
    auto tensor = detail::make_tensor<at::TensorImpl>(
        storage_impl, at::DispatchKey::CPU, my_dtype);

    // 设置张量的大小为 my_size
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous({my_size});

    // 返回创建的 Tensor 对象
    return tensor;
}

// 克隆给定张量 src，可选地指定内存格式
Tensor clone(const Tensor& src, std::optional<c10::MemoryFormat> optional_memory_format) {
  // 获取或指定内存格式，默认为 MemoryFormat::Preserve
  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Preserve);

  // 定义一个空的 Tensor 对象 self
  Tensor self;

  // 根据内存格式选择复制方法
  if (memory_format == MemoryFormat::Preserve) {
    // 如果保留内存格式，并且原张量是非重叠且密集的，直接复制所有步长，稍微比调用 empty_like 快一点
    if (src.is_non_overlapping_and_dense()) {
      self = at::empty_strided_symint(src.sym_sizes(), src.sym_strides(), src.options());
    } else {
      self = at::empty_like(src);
    }
  } else {
    // 如果指定了特定的内存格式，则根据该格式创建空张量
    self = at::empty_like(src, src.options(), memory_format);
  }

  // 如果原张量是零张量，则将 self 张量置零
  if (src._is_zerotensor()) {
    self.zero_();
  } else {
    // 否则复制原张量的数据到 self 张量
    self.copy_(src);
  }

  // 返回克隆后的 Tensor 对象 self
  return self;
}

// 具名张量的重载函数，暂时存在，长期计划将 DimnameList 移入 TensorOptions 以避免这些重载
Tensor full(
    IntArrayRef size,
    const Scalar& fill_value,
    optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    // 使用指定的 dtype、layout、device 和 pinned_memory 创建 TensorOptions 对象
    TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
    
    // 检查 options 的布局是否为稀疏布局，若是则抛出异常
    TORCH_CHECK(options.layout() != kSparse,
      "full(...) is not implemented for sparse layout");
    
    // 使用指定的 size、names 和填充值创建一个空的 Tensor，并根据填充值和 options 推断完整的选项
    auto result = at::empty(size, names, infer_full_options(fill_value, options));
    
    // 使用填充值对 result 进行填充，并返回填充后的结果
    return result.fill_(fill_value);
}

Tensor ones(
    IntArrayRef size,
    optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]

  // 调用底层的 `native::full` 函数，创建所有元素为 1 的张量
  return native::full(
      size, /*fill_value=*/1., names, dtype, layout, device, pin_memory);
}

Tensor zeros(
    IntArrayRef size,
    optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用底层的 `native::full` 函数，创建所有元素为 0 的张量
  return native::full(size, /*fill_value=*/0., names, dtype, layout, device, pin_memory);
}

Tensor randn(
    IntArrayRef size,
    optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用底层的 `native::randn` 函数，创建服从标准正态分布的随机张量
  return native::randn(size, c10::nullopt, names, dtype, layout, device, pin_memory);
}

Tensor randn(
    IntArrayRef size,
    std::optional<Generator> generator,
    optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  // 根据指定的选项创建张量，并使用正态分布进行初始化
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, names, options);
  return result.normal_(0, 1, std::move(generator));
}

Tensor rand(
    IntArrayRef size,
    optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 调用底层的 `native::rand` 函数，创建服从均匀分布的随机张量
  return native::rand(size, c10::nullopt, names, dtype, layout, device, pin_memory);
}

Tensor rand(
    IntArrayRef size,
    std::optional<Generator> generator,
    optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  // 根据指定的选项创建张量，并使用均匀分布进行初始化
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, names, options);
  return result.uniform_(0, 1, std::move(generator));
}


DEFINE_DISPATCH(kaiser_window_stub);

} // namespace at::native


注释：
- `ones`: 创建所有元素为 1 的张量。
- `zeros`: 创建所有元素为 0 的张量。
- `randn`: 创建服从标准正态分布的随机张量。
- `randn` (重载): 根据指定的选项创建张量，并使用正态分布进行初始化。
- `rand`: 创建服从均匀分布的随机张量。
- `rand` (重载): 根据指定的选项创建张量，并使用均匀分布进行初始化。
- `DEFINE_DISPATCH(kaiser_window_stub);`: 定义 `kaiser_window_stub` 的分发函数。
- `}`: 结束 `at::native` 命名空间。
```