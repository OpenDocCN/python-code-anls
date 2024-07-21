# `.\pytorch\aten\src\ATen\native\SegmentReduce.cpp`

```py
// 定义宏以限制只包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的 SegmentReduce.h 头文件
#include <ATen/native/SegmentReduce.h>

// 包含 ATen 核心 Tensor 类定义和调度相关头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>

// 根据条件包含不同的 ATen 头文件集
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_segment_reduce_backward_native.h>
#include <ATen/ops/all.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/segment_reduce_native.h>
#include <ATen/ops/zeros.h>
#endif

// ATen 的命名空间 at::native
namespace at::native {

// 定义分发函数指针，用于调度不同的 CPU 内核函数
DEFINE_DISPATCH(_segment_reduce_lengths_stub);
DEFINE_DISPATCH(_segment_reduce_offsets_stub);
DEFINE_DISPATCH(_segment_reduce_lengths_backward_stub);
DEFINE_DISPATCH(_segment_reduce_offsets_backward_stub);

// 匿名命名空间，定义了一个模板函数 _segment_reduce_lengths_cpu_kernel1
namespace {

// 模板函数 _segment_reduce_lengths_cpu_kernel1 的定义
template <typename T, bool is_offsets_like=false>
void _segment_reduce_lengths_cpu_kernel1(
    ReductionType reduction,
    const Tensor& data,
    const T* lengths_data,
    int64_t axis,
    const std::optional<Scalar>& initial,
    Tensor& output,
    int64_t segment_count,
    int64_t lengths_stride_axis) {
    // 函数体内容在具体实现中定义
}

} // anonymous namespace

// 定义了一个 CPU 内核函数 _segment_reduce_lengths_cpu_kernel
Tensor _segment_reduce_lengths_cpu_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  // 检查 data 张量是否是连续的
  TORCH_CHECK(data.is_contiguous(), "Expected data to be contiguous.");
  // 检查 lengths 张量是否是连续的
  TORCH_CHECK(lengths.is_contiguous(), "Expected lengths to be contiguous.");
  // 确定 reduction 轴总是长度张量的最后一个维度
  axis = lengths.dim() - 1;
  // 获取长度张量在指定轴上的大小
  int64_t segment_count = lengths.size(axis);
  // 获取长度张量在指定轴上的步长
  int64_t lengths_stride_axis = lengths.stride(axis);
  // 根据 data 张量的大小和类型创建一个空张量作为输出
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  // 使用索引类型分发长度数据指针给模板函数 _segment_reduce_lengths_cpu_kernel1
  AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "_segment_reduce_lengths_cpu_kernel1", [&]() {
    const auto* lengths_data = lengths.const_data_ptr<index_t>();
    _segment_reduce_lengths_cpu_kernel1(
        reduction, data, lengths_data, axis, initial, output, segment_count, lengths_stride_axis);
  });

  // 返回计算结果的输出张量
  return output;
}

// 定义了一个 CPU 内核函数 _segment_reduce_offsets_cpu_kernel
Tensor _segment_reduce_offsets_cpu_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial) {
    // 函数体内容在具体实现中定义
}
    const std::optional<Scalar>& initial) {
```  
// 接收一个可选的初始值参数 `initial`

  // data and lengths should be contiguous from the call to .contiguous in segment_reduce_kernel
  TORCH_CHECK(data.is_contiguous(), "Expected data to be contiguous.");
```py  
// 检查 `data` 张量是否是连续存储的，否则抛出错误信息 "Expected data to be contiguous."

  TORCH_CHECK(offsets.is_contiguous(), "Expected offsets to be contiguous.");
```  
// 检查 `offsets` 张量是否是连续存储的，否则抛出错误信息 "Expected offsets to be contiguous."

  // reduction axis should always be the last dimension of lengths
  axis = offsets.dim() - 1;
```py  
// 确定减少操作的轴是 `offsets` 张量的最后一个维度

  int64_t segment_count = offsets.size(axis) - 1;
```  
// 计算分段的数量，通过 `offsets` 张量在轴 `axis` 上的大小减去 1

  int64_t offsets_stride_axis = offsets.stride(axis);
```py  
// 计算 `offsets` 张量在轴 `axis` 上的步长（stride）

  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
```  
// 创建输出张量的形状，保持与 `data` 张量相同，但将其在轴 `axis` 上的尺寸设置为 `segment_count`

  auto output = at::empty(output_shape, data.options());
```py  
// 使用 `data` 张量的选项创建一个空的输出张量，形状为 `output_shape`

  AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_segment_reduce_offsets_cpu_kernel1", [&]() {
```  
// 使用 `offsets` 张量的索引类型分发到不同的内核函数，此处命名为 "_segment_reduce_offsets_cpu_kernel1"

    const auto* offsets_data = offsets.const_data_ptr<index_t>();
```py  
// 获取 `offsets` 张量的常量数据指针，类型为 `index_t`

    _segment_reduce_lengths_cpu_kernel1<index_t, /*is_offsets_like=*/true>(
        reduction, data, offsets_data, axis, initial, output, segment_count, offsets_stride_axis);
```  
// 调用内核函数 `_segment_reduce_lengths_cpu_kernel1` 处理数据，传入减少操作类型 `reduction`、`data` 张量、`offsets_data` 数据指针、轴 `axis`、可选的初始值 `initial`、输出张量 `output`、分段数量 `segment_count` 以及步长 `offsets_stride_axis`

  });

  return output;
```py  
// 返回计算后的输出张量 `output`
}

template <typename T, bool is_offsets_like = false>
void _segment_reduce_cpu_lengths_backward_kernel1(
    const Tensor& grad_contig,  // 梯度张量，用于存储反向传播的梯度
    const Tensor& output_contig,  // 输出张量，用于存储前向传播的结果
    const Tensor& data_contig,  // 数据张量，包含需要进行分段操作的数据
    ReductionType reduction,  // 分段操作的类型（如求和、求平均等）
    const T* lengths_data,  // 分段长度数据的指针，用于指向长度数据的起始位置
    int64_t axis,  // 分段操作的轴向，即在哪个维度上进行分段操作
    const std::optional<Scalar>& initial,  // 可选的初始值，用于一些操作的初始值设定
    Tensor& grad_input,  // 梯度输入张量，用于存储计算得到的梯度值
    int64_t segment_count,  // 分段的数量，根据长度或偏移量张量计算得到
}

Tensor _segment_reduce_cpu_lengths_backward_kernel(
    const Tensor& grad_contig,  // 梯度张量，用于存储反向传播的梯度
    const Tensor& output_contig,  // 输出张量，用于存储前向传播的结果
    const Tensor& data_contig,  // 数据张量，包含需要进行分段操作的数据
    ReductionType reduction,  // 分段操作的类型（如求和、求平均等）
    const Tensor& lengths_contig,  // 长度张量，指定每个分段的长度
    int64_t axis,  // 分段操作的轴向，即在哪个维度上进行分段操作
    const std::optional<Scalar>& initial) {  // 可选的初始值，用于一些操作的初始值设定
  axis = lengths_contig.dim() - 1;  // 计算分段操作的轴向，通常为长度张量的最后一个维度
  int64_t segment_count = lengths_contig.size(axis);  // 计算分段的数量
  int64_t lengths_stride_axis = lengths_contig.stride(axis);  // 获取长度张量在轴向上的步长
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());  // 初始化梯度输入张量为与数据张量相同大小的零张量

  AT_DISPATCH_INDEX_TYPES(
      lengths_contig.scalar_type(), "_segment_reduce_cpu_lengths_backward_kernel1", [&] {
        const auto* lengths_data = lengths_contig.const_data_ptr<index_t>();  // 获取长度数据的指针
        _segment_reduce_cpu_lengths_backward_kernel1(
            grad_contig,
            output_contig,
            data_contig,
            reduction,
            lengths_data,
            axis,
            initial,
            grad_input,
            segment_count,
            lengths_stride_axis);  // 调用分段操作的实际计算函数
      });

  return grad_input;  // 返回计算得到的梯度输入张量
}


Tensor _segment_reduce_cpu_offsets_backward_kernel(
    const Tensor& grad_contig,  // 梯度张量，用于存储反向传播的梯度
    const Tensor& output_contig,  // 输出张量，用于存储前向传播的结果
    const Tensor& data_contig,  // 数据张量，包含需要进行分段操作的数据
    ReductionType reduction,  // 分段操作的类型（如求和、求平均等）
    const Tensor& offsets_contig,  // 偏移量张量，指定每个分段的起始偏移量
    int64_t axis,  // 分段操作的轴向，即在哪个维度上进行分段操作
    const std::optional<Scalar>& initial) {  // 可选的初始值，用于一些操作的初始值设定
  axis = offsets_contig.dim() - 1;  // 计算分段操作的轴向，通常为偏移量张量的最后一个维度
  int64_t segment_count = offsets_contig.size(axis) - 1;  // 计算分段的数量，偏移量张量大小减一
  int64_t offsets_stride_axis = offsets_contig.stride(axis);  // 获取偏移量张量在轴向上的步长
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());  // 初始化梯度输入张量为与数据张量相同大小的零张量

  AT_DISPATCH_INDEX_TYPES(
      offsets_contig.scalar_type(), "_segment_reduce_cpu_offsets_backward_kernel1", [&] {
        const auto* offsets_data = offsets_contig.const_data_ptr<index_t>();  // 获取偏移量数据的指针
        _segment_reduce_cpu_lengths_backward_kernel1<index_t, /*is_offsets_like=*/true>(
            grad_contig,
            output_contig,
            data_contig,
            reduction,
            offsets_data,
            axis,
            initial,
            grad_input,
            segment_count,
            offsets_stride_axis);  // 调用分段操作的实际计算函数，特指偏移量形式的分段操作
      });

  return grad_input;  // 返回计算得到的梯度输入张量
}

} // namespace
    // 对轴进行边界检查，并确保其在数据维度范围内
    axis = maybe_wrap_dim(axis, data.ndimension());
    // 检查数据元素数量是否为非负数
    TORCH_CHECK(data.numel() >= 0);
    
    // 检查长度或偏移量是否有定义
    auto lengths_has_value = lengths.has_value();
    auto offsets_has_value = offsets.has_value();
    // 检查是否不支持基于索引的减少操作
    TORCH_CHECK(
      !indices.has_value(),
      "segment_reduce(): indices based reduction is not supported yet.");
    // 确保长度或偏移量至少有一个被定义
    TORCH_CHECK(
        lengths_has_value || offsets_has_value,
        "segment_reduce(): Either lengths or offsets must be defined.")
    
    // 获取减少操作的枚举值，并确保数据是连续的
    auto reduction = get_reduction_enum(reduce);
    const auto data_contig = data.contiguous();
    
    if (offsets_has_value) {
      const auto& offsets_value = offsets.value();
    
      // 关于偏移量的一些检查
      TORCH_CHECK(data.get_device() == offsets_value.get_device());
      TORCH_CHECK(data.dim() >= offsets_value.dim());
      // 确保轴是偏移量的最后一个维度
      TORCH_CHECK(axis == offsets_value.dim() - 1,
                  "segment_reduce(): Expected axis to be the last dimension of offsets but got ", axis, ".");
    
      // TODO: 当 !unsafe 时添加检查
    
      const auto offsets_contig = offsets_value.contiguous();
    
      // 调用偏移量相关的分段减少操作的底层函数
      return _segment_reduce_offsets_stub(
        data_contig.device().type(),
        reduction,
        data_contig,
        offsets_contig,
        axis,
        initial);
    
    } else {
      const auto& lengths_value = lengths.value();
    
      // 关于长度的一些检查
      TORCH_CHECK(data.get_device() == lengths_value.get_device());
      TORCH_CHECK(data.dim() >= lengths_value.dim());
      // 确保轴是长度的最后一个维度
      TORCH_CHECK(axis == lengths_value.dim() - 1,
                  "segment_reduce(): Expected axis to be the last dimension of lengths but got ", axis, ".");
    
      // 如果不是不安全模式，则进行进一步的检查
      if (!unsafe) {
        auto min_length = lengths_value.min().item<int64_t>();
        TORCH_CHECK((min_length >= 0), "lengths contains negative value!");
        TORCH_CHECK(all(lengths_value.sum({-1}) == data.size(axis)).item<bool>(),
                    "segment_reduce(): Expected all rows of lengths along axis ",
                    "to sum to data.size(lengths.dim()-1) when !unsafe.");
      }
    
      const auto lengths_contig = lengths_value.contiguous();
    
      // 调用长度相关的分段减少操作的底层函数
      return _segment_reduce_lengths_stub(
        data_contig.device().type(),
        reduction,
        data_contig,
        lengths_contig,
        axis,
        initial);
    }
}

// 注册分段减少长度的 CPU 核函数到默认分发器
REGISTER_ARCH_DISPATCH(
    _segment_reduce_lengths_stub,
    DEFAULT,
    &_segment_reduce_lengths_cpu_kernel);

// 注册 AVX2 指令集下的分段减少长度 CPU 核函数
REGISTER_AVX2_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel);

// 注册 AVX512 指令集下的分段减少长度 CPU 核函数
REGISTER_AVX512_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel);

// 注册 VSX 指令集下的分段减少长度 CPU 核函数
REGISTER_VSX_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel);

// 注册 ZVector 指令集下的分段减少长度 CPU 核函数
REGISTER_ZVECTOR_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel);

// offsets dispatches
// 注册分段减少偏移量的 CPU 核函数到默认分发器
REGISTER_ARCH_DISPATCH(
    _segment_reduce_offsets_stub,
    DEFAULT,
    &_segment_reduce_offsets_cpu_kernel);

// 注册 AVX2 指令集下的分段减少偏移量 CPU 核函数
REGISTER_AVX2_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel);

// 注册 AVX512 指令集下的分段减少偏移量 CPU 核函数
REGISTER_AVX512_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel);

// 注册 VSX 指令集下的分段减少偏移量 CPU 核函数
REGISTER_VSX_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel);

// 注册 ZVector 指令集下的分段减少偏移量 CPU 核函数
REGISTER_ZVECTOR_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel);

// Currently some computation is being duplicated across forward and backward.
// TODO: Cache indices in forward pass to re-use in backward
// 分段减少的反向传播核函数
Tensor _segment_reduce_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    c10::string_view reduce,
    const std::optional<Tensor>& lengths,
    const std::optional<Tensor>& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  // 检查 lengths 或 offsets 是否已定义
  // 生成导数.yaml 为 None 传递未定义的 Tensor 而不是 std::optional，所以 .has_value() 检查不像在前向传递中那样工作
  auto lengths_has_value = lengths.has_value() && lengths.value().defined();
  auto offsets_has_value = offsets.has_value() && offsets.value().defined();
  TORCH_CHECK(
      lengths_has_value ||  offsets_has_value,
      "segment_reduce(): Either lengths or offsets must be defined.");

  const auto grad_contig = grad.contiguous();
  const auto output_contig = output.contiguous();
  const auto data_contig = data.contiguous();
  auto reduction = get_reduction_enum(reduce);

  if (offsets_has_value) {
    const auto& offsets_value = offsets.value();
    const auto offsets_contig = offsets_value.contiguous();
    // 返回分段减少偏移量的反向传播核函数的结果
    return _segment_reduce_offsets_backward_stub(
      grad_contig.device().type(),
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      offsets_contig,
      axis,
      initial);
  } else {
    const auto& lengths_value = lengths.value();
    const auto lengths_contig = lengths_value.contiguous();
    // 返回分段减少长度的反向传播核函数的结果
    return _segment_reduce_lengths_backward_stub(
      grad_contig.device().type(),
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      lengths_contig,
      axis,
      initial);
  }
}

// 注册分段减少长度的 CPU 核函数到默认分发器，用于反向传播
REGISTER_ARCH_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    DEFAULT,
    &_segment_reduce_cpu_lengths_backward_kernel);
REGISTER_AVX512_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel);
# 注册 AVX-512 指令集的分发函数，将 _segment_reduce_lengths_backward_stub 映射到 _segment_reduce_cpu_lengths_backward_kernel 函数

REGISTER_AVX2_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel);
# 注册 AVX2 指令集的分发函数，将 _segment_reduce_lengths_backward_stub 映射到 _segment_reduce_cpu_lengths_backward_kernel 函数

REGISTER_VSX_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel);
# 注册 VSX 指令集的分发函数，将 _segment_reduce_lengths_backward_stub 映射到 _segment_reduce_cpu_lengths_backward_kernel 函数

REGISTER_ZVECTOR_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel);
# 注册 ZVECTOR 指令集的分发函数，将 _segment_reduce_lengths_backward_stub 映射到 _segment_reduce_cpu_lengths_backward_kernel 函数

REGISTER_ARCH_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    DEFAULT,
    &_segment_reduce_cpu_offsets_backward_kernel);
# 注册默认架构的分发函数，将 _segment_reduce_offsets_backward_stub 映射到 _segment_reduce_cpu_offsets_backward_kernel 函数

REGISTER_AVX512_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel);
# 注册 AVX-512 指令集的分发函数，将 _segment_reduce_offsets_backward_stub 映射到 _segment_reduce_cpu_offsets_backward_kernel 函数

REGISTER_AVX2_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel);
# 注册 AVX2 指令集的分发函数，将 _segment_reduce_offsets_backward_stub 映射到 _segment_reduce_cpu_offsets_backward_kernel 函数

REGISTER_VSX_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel);
# 注册 VSX 指令集的分发函数，将 _segment_reduce_offsets_backward_stub 映射到 _segment_reduce_cpu_offsets_backward_kernel 函数

REGISTER_ZVECTOR_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel);
# 注册 ZVECTOR 指令集的分发函数，将 _segment_reduce_offsets_backward_stub 映射到 _segment_reduce_cpu_offsets_backward_kernel 函数

} // namespace at::native
# 结束 at::native 命名空间
```