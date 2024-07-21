# `.\pytorch\aten\src\ATen\native\ConvolutionMM3d.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/div_rtn.h>
#include <ATen/native/ConvolutionMM3d.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/Unfold3d.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/slow_conv3d_forward.h>
#include <ATen/ops/slow_conv3d_forward_native.h>
#include <ATen/ops/slow_conv3d_native.h>
#include <ATen/ops/sum.h>
#endif

// 定义常量 CONV3D_GRAIN_SALT 的值为 20
constexpr int64_t CONV3D_GRAIN_SALT = 20;

namespace at::native {

namespace {

// 静态函数 compute_columns3d，用于计算 3D 卷积操作的列向量
static Tensor compute_columns3d(
    const Tensor& input_,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef kernel_size,
    const int64_t groups) {
  // 将输入张量进行内存连续化
  const Tensor input = input_.contiguous();
  // 获取卷积核的深度、高度、宽度
  const int64_t kernel_depth = kernel_size[0];
  const int64_t kernel_height = kernel_size[1];
  const int64_t kernel_width = kernel_size[2];
  // 获取填充值的深度、高度、宽度
  const int64_t pad_depth = padding[0];
  const int64_t pad_height = padding[1];
  const int64_t pad_width = padding[2];
  // 获取步长的深度、高度、宽度
  const int64_t stride_depth = stride[0];
  const int64_t stride_height = stride[1];
  const int64_t stride_width = stride[2];
  // 定义维度常量
  const int64_t dim_planes = 1;
  const int64_t dim_depth = 2;
  const int64_t dim_height = 3;
  const int64_t dim_width = 4;
  // 获取输入张量的相关维度信息
  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_depth = input.size(dim_depth);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  // 计算输出张量的深度、高度、宽度
  const int64_t output_depth =
      (input_depth + 2 * pad_depth - kernel_depth) / stride_depth + 1;
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
  const int64_t batch_size = input.size(0);

  Tensor columns;
  // 对于特殊情况，当卷积核和步长都为1，填充为0，并且 groups 为1时
  if ((kernel_depth == 1) && (kernel_height == 1) && (kernel_width == 1) &&
      (pad_depth == 0) && (pad_height == 0) && (pad_width == 0) &&
      (stride_depth == 1) && (stride_height == 1) && (stride_width == 1) && (groups == 1)) {
    // 对于这种特殊情况，columns 是输入张量的视图
    columns = input.view({batch_size, n_input_plane, output_height * output_width * output_depth}).detach();
  } else {
    // 对于一般情况，创建一个空的张量作为 columns，用于存储计算的列向量
    columns = at::empty({batch_size,
                        n_input_plane * kernel_depth * kernel_height * kernel_width,
                        output_depth * output_height * output_width},
                        input.options());
    # 使用宏展开，处理所有类型和 BFloat16、Half 类型的输入，调用名为 "compute_columns3d" 的分发函数
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "compute_columns3d", [&] {
      # 获取输入张量的访问器，以便访问 5 维数据（5 维张量）
      auto input_a = input.accessor<const scalar_t, 5>();
      # 获取输出列张量的访问器，以便访问 3 维数据（3 维张量）
      auto columns_a = columns.accessor<scalar_t, 3>();

      # 并行处理，使用指定的 GRAIN_SALT 大小进行 3D 卷积操作
      at::parallel_for(0, batch_size, CONV3D_GRAIN_SALT, [&](int64_t start, int64_t end) {
        # 遍历每个 batch 中的索引范围 [start, end)
        for (const auto t : c10::irange(start, end)) {
          # 获取当前时间步 t 的输入张量和输出列张量
          auto input_t = input_a[t];
          auto columns_t = columns_a[t];
          
          # 调用 CPU 上的 Unfold3dCopyCPU 函数，进行 3D 数据展开和复制操作
          Unfold3dCopyCPU(
            c10::CppTypeToScalarType<scalar_t>::value,  # 数据类型转换，获取标量类型
            input_t.data(),         # 输入张量数据的指针
            n_input_plane,          # 输入平面数
            input_depth,            # 输入深度
            input_height,           # 输入高度
            input_width,            # 输入宽度
            output_depth,           # 输出深度
            output_height,          # 输出高度
            output_width,           # 输出宽度
            kernel_depth,           # 卷积核深度
            kernel_height,          # 卷积核高度
            kernel_width,           # 卷积核宽度
            stride_depth,           # 步幅深度
            stride_height,          # 步幅高度
            stride_width,           # 步幅宽度
            pad_depth,              # 填充深度
            pad_height,             # 填充高度
            pad_width,              # 填充宽度
            columns_t.data());      # 输出列张量数据的指针
        }
      });
    });
  }

  # 返回填充好的输出列张量
  return columns;
}

static inline void slow_conv3d_shape_check(
    const Tensor& input,  // 输入张量
    const Tensor& grad_output,  // 梯度输出张量
    const Tensor& weight,  // 权重张量
    const Tensor& bias,  // 偏置张量
    int64_t kernel_depth,  // 卷积核的深度
    int64_t kernel_height,  // 卷积核的高度
    int64_t kernel_width,  // 卷积核的宽度
    int64_t stride_depth,  // 卷积步长的深度
    int64_t stride_height,  // 卷积步长的高度
    int64_t stride_width,  // 卷积步长的宽度
    int64_t pad_depth,  // 填充的深度
    int64_t pad_height,  // 填充的高度
    int64_t pad_width,  // 填充的宽度
    int64_t groups,  // 分组卷积中的组数
    bool weight_optional) {  // 是否允许权重张量为空的标志
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0 && kernel_depth > 0,
      "kernel size should be greater than zero, but got: ",
      kernel_depth,
      " x ",
      kernel_height,
      " x ",
      kernel_width,
      " (TxHxW)");  // 检查卷积核大小是否大于零
  TORCH_CHECK(
      stride_width > 0 && stride_height > 0 && stride_depth > 0,
      "stride should be greater than zero, but got: ",
      stride_depth,
      " x ",
      stride_height,
      " x ",
      stride_width,
      " (TxHxW)");  // 检查卷积步长是否大于零
  if (weight.defined()) {  // 如果权重张量已定义
    TORCH_CHECK(
        weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 5),
        "non-empty 2D or 5D weight tensor expected, but got: ",
        weight.sizes());  // 检查权重张量是否非空且为2D或5D张量
    if (bias.defined()) {  // 如果偏置张量已定义
      check_dim_size(bias, 1, 0, weight.size(0));  // 检查偏置张量的维度大小是否符合预期
    }
  } else {
  // 检查权重是否可选，如果不可选则抛出错误信息
  TORCH_CHECK(weight_optional, "weight tensor is undefined");
}

const int64_t ndim = input.dim();  // 获取输入张量的维度数
const int64_t dim_batch = 0;  // 批量维度索引
const int64_t dim_planes = 1;  // 通道数维度索引
const int64_t dim_depth = 2;  // 深度维度索引
const int64_t dim_height = 3;  // 高度维度索引
const int64_t dim_width = 4;  // 宽度维度索引

// 允许空的批量大小，但不允许其他维度为空
bool valid_empty = ndim == 5 && input.size(dim_batch) == 0 &&
    input.size(dim_planes) != 0 && input.size(dim_depth) != 0 &&
    input.size(dim_height) != 0 && input.size(dim_width) != 0;

// 检查输入张量是否符合非空的5维输入要求，否则抛出错误信息
TORCH_CHECK(
    (input.numel() > 0 || valid_empty) && ndim == 5,
    "non-empty 5D input tensor expected but got: ",
    input.sizes());

const int64_t input_depth = input.size(dim_depth);  // 输入张量的深度维度大小
const int64_t input_height = input.size(dim_height);  // 输入张量的高度维度大小
const int64_t input_width = input.size(dim_width);  // 输入张量的宽度维度大小

// 计算带填充的输入深度、高度和宽度
const int64_t exact_input_depth = input_depth + 2 * pad_depth;
const int64_t exact_input_height = input_height + 2 * pad_height;
const int64_t exact_input_width = input_width + 2 * pad_width;

// 检查计算的带填充的输入大小是否大于等于卷积核大小，否则抛出错误信息
TORCH_CHECK(
    exact_input_depth >= kernel_depth &&
        exact_input_height >= kernel_height &&
        exact_input_width >= kernel_width,
    "Calculated padded input size per channel: (",
    exact_input_depth,
    " x ",
    exact_input_height,
    " x ",
    exact_input_width,
    "). ",
    "Kernel size: (",
    kernel_depth,
    " x ",
    kernel_height,
    " x ",
    kernel_width,
    "). Kernel size can't be greater than actual input size");

// 计算输出深度、高度和宽度
const int64_t output_depth =
    div_rtn<int64_t>(exact_input_depth - kernel_depth, stride_depth) + 1;
const int64_t output_height =
    div_rtn<int64_t>(exact_input_height - kernel_height, stride_height) + 1;
const int64_t output_width =
    div_rtn<int64_t>(exact_input_width - kernel_width, stride_width) + 1;

// 检查计算的输出大小是否至少为1，否则抛出错误信息
TORCH_CHECK(
    output_depth >= 1 && output_width >= 1 && output_height >= 1,
    "Given input size per channel: (",
    input_depth,
    " x ",
    input_height,
    " x ",
    input_width,
    "). "
    "Calculated output size per channel: (",
    output_depth,
    " x ",
    output_height,
    " x ",
    output_width,
    "). Output size is too small");

if (weight.defined()) {
  int64_t n_input_plane = weight.size(1);  // 输入平面数
  if (weight.dim() == 2) {
    n_input_plane /= (kernel_height * kernel_width);  // 对于二维权重，计算输入平面数的实际值
  }
  // 支持分组卷积，需要检查输入通道数是否是权重通道数的倍数
  TORCH_CHECK(groups > 0, "none zero group size expected");
  check_dim_size(input, ndim, dim_planes, n_input_plane * groups);  // 检查输入张量的通道数是否与权重匹配
}

if (grad_output.defined()) {
  if (weight.defined()) {
    int64_t n_output_plane = weight.size(0);  // 输出平面数
    check_dim_size(grad_output, ndim, dim_planes, n_output_plane);  // 检查梯度输出张量的通道数是否与输出平面数匹配
    // 如果偏置 bias 已定义，则执行以下操作
    } else if (bias.defined()) {
      // 断言偏置 tensor 非空
      TORCH_CHECK(bias.numel() > 0, "non-empty bias tensor expected");
      // 计算输出平面数，如果偏置是标量则为1，否则为偏置的第一维大小
      const int64_t n_output_plane = bias.dim() == 0 ? 1 : bias.size(0);
      // 检查 grad_output tensor 在指定维度上的大小与输出平面数是否匹配
      check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
    }
    // 检查 grad_output tensor 在深度维度上的大小与输出深度是否匹配
    check_dim_size(grad_output, ndim, dim_depth, output_depth);
    // 检查 grad_output tensor 在高度维度上的大小与输出高度是否匹配
    check_dim_size(grad_output, ndim, dim_height, output_height);
    // 检查 grad_output tensor 在宽度维度上的大小与输出宽度是否匹配
    check_dim_size(grad_output, ndim, dim_width, output_width);
  }
}

// 定义静态函数，用于将输入的权重张量视图化为二维张量
static Tensor view_weight_2d(const Tensor& weight_) {
    // 确保权重张量是连续的
    Tensor weight = weight_.contiguous();
    // 如果权重张量的维度是5
    if (weight.dim() == 5) {
        // 获取权重张量的各维度大小
        const int64_t s1 = weight.size(0);
        const int64_t s2 = weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4);
        // 将权重张量视图化为二维张量并返回
        return weight.view({s1, s2});
    } else {
        // 如果不是5维，则直接返回原始的权重张量
        return weight;
    }
}

// 定义模板函数，用于处理三维卷积层的前向传播
template <typename scalar_t>
static void slow_conv3d_update_output_frame(
    TensorAccessor<const scalar_t, 4> input,
    TensorAccessor<scalar_t, 4> output,
    TensorAccessor<const scalar_t, 2> weight,
    bool has_bias,
    TensorAccessor<const scalar_t, 2> finput,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_depth,
    int64_t pad_height,
    int64_t pad_width,
    int64_t n_input_plane,
    int64_t groups,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t n_output_plane,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  
  // 如果有偏置，则beta为1，否则为0
  const int beta = has_bias ? 1 : 0;

  // 计算矩阵乘法 output = weight * input
  // 注意：gemm函数期望Fortran顺序，因此所有三个矩阵都需要转置
  // 通过交换参数顺序取消这种要求，因为 C == AB <=> T(C) == T(B)T(A)
  const int64_t m = output_depth * output_height * output_width;
  const int64_t n = (n_output_plane / groups);
  const int64_t k = (n_input_plane / groups) * kernel_depth * kernel_height * kernel_width;

  const int64_t lda = m;
  const int64_t ldb = k;
  const int64_t ldc = m;

  // 调用批量gemm函数执行矩阵乘法
  at::native::cpublas::gemm_batched_with_stride(
      TransposeType::NoTranspose,
      TransposeType::NoTranspose,
      groups, m, n, k,
      static_cast<scalar_t>(1),
      finput.data(), lda, finput.stride(0) * k,
      weight.data(), ldb, weight.stride(0) * n,
      static_cast<scalar_t>(beta),
      output.data(), ldc, output.stride(0) * n);
}

// 定义模板函数，用于处理三维卷积层的反向传播（更新输入梯度）
template <typename scalar_t>
void slow_conv3d_backward_update_grad_input_frame(
    TensorAccessor<scalar_t, 4> grad_input,
    TensorAccessor<const scalar_t, 4> grad_output,
    TensorAccessor<const scalar_t, 2> weight,
    TensorAccessor<scalar_t, 2> fgrad_input,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_depth,
    int64_t pad_height,
    int64_t pad_width,
    // 计算 fgrad_input = weight.T * grad_output.reshape({grad_output.shape(0), -1})
    // 注意 gemm 函数要求 Fortran 排序，因此所有三个矩阵都要进行转置。
    // 交换参数顺序可以取消这一要求，因为 C == AB <=> T(C) == T(B)T(A)
    const int64_t m = grad_output.size(1) * grad_output.size(2) * grad_output.size(3);
    const int64_t n = weight.size(1);
    const int64_t k = weight.size(0) / groups;
    
    const int64_t lda = m;  // grad_output.data() 的行跨度
    const int64_t ldb = n;  // weight.data() 的行跨度
    const int64_t ldc = m;  // fgrad_input.data() 的行跨度
    
    // 调用 gemm_batched_with_stride 函数进行批量的矩阵乘法计算
    at::native::cpublas::gemm_batched_with_stride(
        TransposeType::NoTranspose,  // grad_output.data() 不转置
        TransposeType::Transpose,    // weight.data() 转置
        groups, m, n, k,             // groups, m, n, k 参数
        static_cast<scalar_t>(1),    // alpha = 1
        grad_output.data(), lda, grad_output.stride(0) * k,  // grad_output 数据指针、行跨度、列跨度
        weight.data(), ldb, weight.stride(0) * k,            // weight 数据指针、行跨度、列跨度
        static_cast<scalar_t>(0),    // beta = 0
        fgrad_input.data(), ldc, fgrad_input.stride(0) * n);  // fgrad_input 数据指针、行跨度、列跨度
    
    // 调用 Unfold3dAccCPU 函数进行 3D 数据的展开和卷积操作
    Unfold3dAccCPU(
        c10::CppTypeToScalarType<scalar_t>::value,  // scalar_t 类型转换
        fgrad_input.data(),                        // fgrad_input 数据指针
        grad_input.size(0),                        // grad_input 第一维大小
        grad_input.size(1),                        // grad_input 第二维大小
        grad_input.size(2),                        // grad_input 第三维大小
        grad_input.size(3),                        // grad_input 第四维大小
        grad_output.size(1),                       // grad_output 第二维大小
        grad_output.size(2),                       // grad_output 第三维大小
        grad_output.size(3),                       // grad_output 第四维大小
        kernel_depth,                             // 卷积核的深度
        kernel_height,                            // 卷积核的高度
        kernel_width,                             // 卷积核的宽度
        stride_depth,                             // 卷积的深度步长
        stride_height,                            // 卷积的高度步长
        stride_width,                             // 卷积的宽度步长
        pad_depth,                                // 深度填充
        pad_height,                               // 高度填充
        pad_width,                                // 宽度填充
        grad_input.data());                       // grad_input 数据指针
}

void slow_conv3d_backward_out_cpu_template(
    Tensor& grad_input,                            # 定义 grad_input 引用，用于存储反向传播的梯度
    const Tensor& grad_output,                     # 输入的梯度张量，用于计算反向传播的梯度
    const Tensor& input,                           # 输入的原始数据张量
    const Tensor& weight,                          # 卷积核张量
    IntArrayRef kernel_size,                       # 卷积核大小的数组引用
    IntArrayRef stride,                            # 步长的数组引用
    IntArrayRef padding,                           # 填充的数组引用
    int64_t groups) {                              # 卷积分组数

  const int64_t kernel_depth = kernel_size[0];      # 提取卷积核的深度维度大小
  const int64_t kernel_height = kernel_size[1];     # 提取卷积核的高度维度大小
  const int64_t kernel_width = kernel_size[2];      # 提取卷积核的宽度维度大小
  const int64_t pad_depth = padding[0];             # 提取填充的深度维度大小
  const int64_t pad_height = padding[1];            # 提取填充的高度维度大小
  const int64_t pad_width = padding[2];             # 提取填充的宽度维度大小
  const int64_t stride_depth = stride[0];           # 提取步长的深度维度大小
  const int64_t stride_height = stride[1];          # 提取步长的高度维度大小
  const int64_t stride_width = stride[2];           # 提取步长的宽度维度大小

  slow_conv3d_shape_check(                         # 执行卷积参数的形状检查
      input,
      grad_output,
      weight,
      Tensor(),
      kernel_depth,
      kernel_height,
      kernel_width,
      stride_depth,
      stride_height,
      stride_width,
      pad_depth,
      pad_height,
      pad_width,
      groups,
      false);

  const Tensor weight2d = view_weight_2d(weight);  # 获取卷积核的二维视图张量
  const Tensor grad_output_contiguous = grad_output.contiguous();  # 获取连续内存的梯度输出张量
  grad_input.resize_as_(input);                    # 调整 grad_input 的大小与输入数据相同
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous")  # 检查 grad_input 是否是连续的

  const int64_t dim_planes = 1;                    # 定义平面维度索引
  const int64_t dim_depth = 2;                     # 定义深度维度索引
  const int64_t dim_height = 3;                    # 定义高度维度索引
  const int64_t dim_width = 4;                     # 定义宽度维度索引
  const int64_t n_input_plane = input.size(dim_planes);  # 获取输入张量的平面维度大小
  const int64_t input_depth = input.size(dim_depth);  # 获取输入张量的深度维度大小
  const int64_t input_height = input.size(dim_height);  # 获取输入张量的高度维度大小
  const int64_t input_width = input.size(dim_width);  # 获取输入张量的宽度维度大小
  const int64_t output_depth =
      (input_depth + 2 * pad_depth - kernel_depth) / stride_depth + 1;  # 计算输出的深度维度大小
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;  # 计算输出的高度维度大小
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;  # 计算输出的宽度维度大小
  const int64_t batch_size = input.size(0);        # 获取批量大小

  Tensor fgrad_input = at::empty({batch_size,
      n_input_plane * kernel_depth * kernel_height * kernel_width,
      output_depth * output_height * output_width}, input.options());  # 创建用于存储梯度输入的张量

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, input.scalar_type(), "slow_conv3d_cpu_grad_input", [&] {
    auto grad_input_a = grad_input.accessor<scalar_t, 5>();  # 获取 grad_input 的访问器
    auto grad_output_a = grad_output_contiguous.accessor<const scalar_t, 5>();  # 获取 grad_output 的访问器
    auto fgrad_input_a = fgrad_input.accessor<scalar_t, 3>();  # 获取 fgrad_input 的访问器
    auto weight_2d_a = weight2d.accessor<const scalar_t, 2>();  # 获取 weight2d 的访问器
    // 使用 ATen 库中的并行方法 `parallel_for` 进行并行处理，范围是从 0 到 batch_size，使用 CONV3D_GRAIN_SALT 作为粒度参数
    at::parallel_for(0, batch_size, CONV3D_GRAIN_SALT,
                    [&](int64_t start, int64_t end) {

        // 对于每个 t 在范围 [start, end) 内进行迭代处理
        for (const auto t : c10::irange(start, end)) {
          // 获取当前迭代索引 t 对应的梯度输入、梯度输出和权重梯度输入
          auto grad_input_t = grad_input_a[t];
          auto grad_output_t = grad_output_a[t];
          auto fgrad_input_t = fgrad_input_a[t];
          // 调用慢速 3D 卷积反向更新梯度输入帧的函数，传入相应的参数
          slow_conv3d_backward_update_grad_input_frame(
              grad_input_t,
              grad_output_t,
              weight_2d_a,
              fgrad_input_t,
              kernel_depth,
              kernel_height,
              kernel_width,
              stride_depth,
              stride_height,
              stride_width,
              pad_depth,
              pad_height,
              pad_width,
              groups);
        }
    });
  });
// 定义函数，用于计算 3D 卷积操作的权重梯度
template <typename scalar_t>
void slow_conv3d_backward_weight_frame(
    TensorAccessor<scalar_t, 2> grad_weight,  // 梯度权重的访问器，二维
    TensorAccessor<const scalar_t, 4> grad_output,  // 梯度输出的访问器，四维常量
    TensorAccessor<const scalar_t, 2> finput,  // 输入特征图的访问器，二维常量
    int64_t groups) {  // 卷积组数

  // 计算 grad_weight += grad_output.reshape({grad_output.shape(0), -1}) * finput.T
  // 注：gemm 函数期望的是 Fortran 顺序，因此所有三个矩阵都进行了转置。
  // 通过交换参数顺序可以取消这种影响，因为 C == AB <=> T(C) == T(B)T(A)
  const int64_t m = grad_weight.size(1);  // grad_weight 的第二维大小
  const int64_t n = grad_weight.size(0) / groups;  // grad_weight 的第一维大小除以组数
  const int64_t k = grad_output.size(1) * grad_output.size(2) * grad_output.size(3);  // grad_output 展开后的大小

  const int64_t lda = k;  // finput 的列数
  const int64_t ldb = k;  // grad_output 的列数
  const int64_t ldc = m;  // grad_weight 的列数

  // 调用 gemm_batched_with_stride 函数执行批量矩阵乘法
  at::native::cpublas::gemm_batched_with_stride(
      TransposeType::Transpose,  // 第一个矩阵转置
      TransposeType::NoTranspose,  // 第二个矩阵不转置
      groups, m, n, k,  // 组数，grad_weight 的行数，grad_weight 的列数，finput 的列数
      static_cast<scalar_t>(1),  // 乘法因子
      finput.data(), lda, finput.stride(0) * m,  // finput 数据指针，以及对应的步长
      grad_output.data(), ldb, grad_output.stride(0) * n,  // grad_output 数据指针，以及对应的步长
      static_cast<scalar_t>(1),  // 加法因子
      grad_weight.data(), ldc, grad_weight.stride(0) * n);  // grad_weight 数据指针，以及对应的步长
}

// 定义静态函数，用于在 CPU 上模板化反向传播计算卷积参数
static void slow_conv3d_backward_parameters_out_cpu_template(
    Tensor& grad_weight,  // 梯度权重张量
    const Tensor& input,  // 输入张量
    const Tensor& grad_output,  // 梯度输出张量
    IntArrayRef kernel_size,  // 卷积核大小
    IntArrayRef stride,  // 步长
    IntArrayRef padding,  // 填充
    int64_t groups) {  // 卷积组数

  CheckedFrom c = "slow_conv3d_backward_parameters_cpu";  // 检查来源
  auto grad_weight_arg = TensorArg(grad_weight, "grad_weight_arg", 0);  // grad_weight 的张量参数

  // 提取卷积核尺寸、填充、步长信息
  const int64_t kernel_depth = kernel_size[0];
  const int64_t kernel_height = kernel_size[1];
  const int64_t kernel_width = kernel_size[2];
  const int64_t pad_depth = padding[0];
  const int64_t pad_height = padding[1];
  const int64_t pad_width = padding[2];
  const int64_t stride_depth = stride[0];
  const int64_t stride_height = stride[1];
  const int64_t stride_width = stride[2];

  // 执行卷积形状检查，确保输入张量和梯度输出张量与卷积参数匹配
  slow_conv3d_shape_check(
      input,
      grad_output,
      grad_weight,
      {},
      kernel_depth,
      kernel_height,
      kernel_width,
      stride_depth,
      stride_height,
      stride_width,
      pad_depth,
      pad_height,
      pad_width,
      groups,
      true);

  // 将 grad_weight 视图转换为二维张量
  Tensor grad_weight_2d = view_weight_2d(grad_weight);
  // 检查 grad_weight 是否是连续的
  checkContiguous(c, grad_weight_arg);

  // 使 grad_output 连续化
  auto grad_output_contiguous = grad_output.contiguous();

  const int64_t batch_size = input.size(0);  // 批量大小
  // 计算 3D 卷积的输入特征图列
  Tensor finput = compute_columns3d(input, stride, padding, kernel_size, groups);

  // 根据输入的浮点类型分派不同的操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, input.scalar_type(), "slow_conv3d_cpu_grad_weight", [&] {
    auto grad_weight_2d_a = grad_weight_2d.accessor<scalar_t, 2>();  // 访问 grad_weight_2d 的访问器
    auto grad_output_a = grad_output_contiguous.accessor<const scalar_t, 5>();  // 访问 grad_output 的访问器
    auto finput_a = finput.accessor<const scalar_t, 3>();  // 访问 finput 的访问器
    for (const auto t : c10::irange(batch_size)) {  // 遍历批次
      auto grad_output_t = grad_output_a[t];  // 获取当前批次的 grad_output
      auto finput_t = finput_a[t];  // 获取当前批次的 finput
      // 调用 slow_conv3d_backward_weight_frame 函数计算当前批次的权重梯度
      slow_conv3d_backward_weight_frame(
          grad_weight_2d_a, grad_output_t, finput_t, groups);
    }
  });
}
} // 结束 slow_conv3d_forward_out_cpu 函数定义

} // 结束命名空间

Tensor& slow_conv3d_forward_out_cpu(const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  // 根据可选的偏置参数，创建一个 MaybeOwned<Tensor> 对象
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 解引用 MaybeOwned<Tensor> 对象，获取真实的偏置 Tensor
  const Tensor& bias = *bias_maybe_owned;

  // 提取卷积核的深度、高度、宽度以及填充值、步长等参数
  const int64_t kernel_depth = kernel_size[0];
  const int64_t kernel_height = kernel_size[1];
  const int64_t kernel_width = kernel_size[2];
  const int64_t pad_depth = padding[0];
  const int64_t pad_height = padding[1];
  const int64_t pad_width = padding[2];
  const int64_t stride_depth = stride[0];
  const int64_t stride_height = stride[1];
  const int64_t stride_width = stride[2];

  // TODO: 用于确定分组的一种简单方式
  // 假设组大小已在上游函数中检查过
  const int64_t groups = weight.size(1) > 0 ? self.size(1) / weight.size(1) : 0;

  // 进行卷积参数的完整性检查
  slow_conv3d_shape_check(
      self,
      Tensor(),
      weight,
      bias,
      kernel_depth,
      kernel_height,
      kernel_width,
      stride_depth,
      stride_height,
      stride_width,
      pad_depth,
      pad_height,
      pad_width,
      groups,
      false);

  // 获取连续存储的输入 Tensor
  const Tensor input = self.contiguous();
  // 将权重 Tensor 视图为二维形式
  const Tensor weight_2d = view_weight_2d(weight);

  // 定义输入、深度、高度、宽度维度的索引
  const int64_t dim_planes = 1;
  const int64_t dim_depth = 2;
  const int64_t dim_height = 3;
  const int64_t dim_width = 4;

  // 提取输入 Tensor 的尺寸信息
  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_depth = input.size(dim_depth);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  // 提取权重 Tensor 二维视图的输出平面数量
  const int64_t n_output_plane = weight_2d.size(0);
  // 计算输出 Tensor 的深度、高度、宽度
  const int64_t output_depth =
      (input_depth + 2 * pad_depth - kernel_depth) / stride_depth + 1;
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  // 计算输入 Tensor 的列向量，用于卷积计算
  Tensor finput = compute_columns3d(input, stride, padding, kernel_size, groups);
  // 调整输出 Tensor 的尺寸
  const int64_t batch_size = input.size(0);
  output.resize_(
      {batch_size, n_output_plane, output_depth, output_height, output_width});
  // 如果存在偏置，则将其复制到输出 Tensor 中
  if (bias.defined()) {
    output.copy_(bias.reshape({-1, 1, 1, 1}));
  }

  // 检查输出 Tensor 是否是连续存储的
  TORCH_CHECK(output.is_contiguous(), "slow_conv3d output must be contiguous");

  // 使用宏来支持所有数据类型的卷积计算，访问输入、输出、列向量和二维权重 Tensor
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "slow_conv3d_cpu", [&] {
    auto input_a = input.accessor<const scalar_t, 5>();
    auto output_a = output.accessor<scalar_t, 5>();
    auto finput_a = finput.accessor<const scalar_t, 3>();
    auto weight_2d_a = weight_2d.accessor<const scalar_t, 2>();
    # 使用 ATen 库的并行操作函数 parallel_for，将任务划分成多个线程执行
    at::parallel_for(
        0, batch_size, CONV3D_GRAIN_SALT, [&](int64_t start, int64_t end) {
          # 遍历指定范围内的索引，这里是一个线程执行的任务范围
          for (const auto t : c10::irange(start, end)) {
            # 获取当前处理的输入、输出和 finput（可能是特征输入）的引用
            auto input_t = input_a[t];
            auto output_t = output_a[t];
            auto finput_t = finput_a[t];
            # 调用慢速 3D 卷积的更新输出函数，计算输出帧
            slow_conv3d_update_output_frame(
                input_t,
                output_t,
                weight_2d_a,
                bias.defined(),  # 检查是否定义了偏置
                finput_t,
                kernel_depth,
                kernel_height,
                kernel_width,
                stride_depth,
                stride_height,
                stride_width,
                pad_depth,
                pad_height,
                pad_width,
                n_input_plane,
                groups,
                input_depth,
                input_height,
                input_width,
                n_output_plane,
                output_depth,
                output_height,
                output_width);
          }
        });
  });
  
  # 返回计算后的输出结果
  return output;
// 定义一个函数，执行 CPU 上的 3D 慢速卷积操作
Tensor slow_conv3d_forward_cpu(
    const Tensor& self,                      // 输入张量
    const Tensor& weight,                    // 卷积核张量
    IntArrayRef kernel_size,                 // 卷积核尺寸
    const std::optional<Tensor>& bias_opt,   // 可选的偏置张量
    IntArrayRef stride,                      // 步长
    IntArrayRef padding) {                   // 填充

  // 基于可选的偏置张量创建一个 MaybeOwned 类型的引用
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 获取真正的偏置张量
  const Tensor& bias = *bias_maybe_owned;

  // 创建一个空的输出张量
  auto output = at::empty({0}, self.options());
  // 调用 native 命名空间中的函数执行慢速 3D 卷积的前向传播
  at::native::slow_conv3d_forward_out_cpu(
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output);
  // 返回输出张量
  return output;
}

// 定义一个静态函数，执行 CPU 上的 3D 慢速卷积反向传播操作，返回三个张量的元组引用
static std::tuple<Tensor&, Tensor&, Tensor&> slow_conv3d_backward_out_cpu(
    const Tensor& grad_output,               // 梯度输出张量
    const Tensor& self,                      // 输入张量
    const Tensor& weight,                    // 卷积核张量
    IntArrayRef kernel_size,                 // 卷积核尺寸
    IntArrayRef stride,                      // 步长
    IntArrayRef padding,                     // 填充
    Tensor& grad_input,                      // 梯度输入张量
    Tensor& grad_weight,                     // 梯度卷积核张量
    Tensor& grad_bias) {                     // 梯度偏置张量

  // TODO: hacky way of determine the group size （确定组大小的笨拙方法）
  int64_t groups = self.size(1) / weight.size(1);

  // 如果定义了梯度输入张量，调用对应的反向传播模板函数
  if (grad_input.defined()) {
    slow_conv3d_backward_out_cpu_template(
        grad_input,
        grad_output,
        self,
        weight,
        kernel_size,
        stride,
        padding,
        groups);
  }

  // 如果定义了梯度偏置张量，使用指定的维度对梯度输出张量进行求和
  if (grad_bias.defined()) {
    at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3, 4});
  }

  // 如果定义了梯度卷积核张量，重置并清零其内容，然后调用对应的参数反向传播模板函数
  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
    slow_conv3d_backward_parameters_out_cpu_template(
        grad_weight,
        self,
        grad_output,
        kernel_size,
        stride,
        padding,
        groups);
  }

  // 返回三个张量的引用元组
  return std::tuple<Tensor&, Tensor&, Tensor&>(
      grad_input, grad_weight, grad_bias);
}

// 定义一个函数，执行 CPU 上的 3D 慢速卷积反向传播操作，返回三个张量的元组
std::tuple<Tensor, Tensor, Tensor> slow_conv3d_backward_cpu(
    const Tensor& grad_output,               // 梯度输出张量
    const Tensor& self,                      // 输入张量
    const Tensor& weight,                    // 卷积核张量
    IntArrayRef kernel_size,                 // 卷积核尺寸
    IntArrayRef stride,                      // 步长
    IntArrayRef padding,                     // 填充
    std::array<bool, 3> output_mask) {       // 输出遮罩数组

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  // 如果输出遮罩数组的第一个元素为真，创建一个空的梯度输入张量
  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  }

  // 如果输出遮罩数组的第二个元素为真，创建一个空的梯度卷积核张量
  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  }

  // 如果输出遮罩数组的第三个元素为真，创建一个空的梯度偏置张量
  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  }

  // 调用 native 命名空间中的函数执行慢速 3D 卷积的反向传播
  at::native::slow_conv3d_backward_out_cpu(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      grad_input,
      grad_weight,
      grad_bias);

  // 返回梯度输入张量、梯度卷积核张量和梯度偏置张量的元组
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// 定义一个函数，执行 CPU 上的 3D 慢速卷积操作并返回输出张量的引用
Tensor& slow_conv3d_out(const Tensor& self,      // 输入张量
    const Tensor& weight,                        // 卷积核张量
    IntArrayRef kernel_size,                     // 卷积核尺寸
    const std::optional<Tensor>& bias_opt,       // 可选的偏置张量
    IntArrayRef stride,                          // 步长
    IntArrayRef padding,                         // 填充
    // 从可选的张量中借用权重张量，处理可能存在的包装器以获取实际的张量对象
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    // 获取不可变的常量引用，确保bias变量始终指向一个有效的张量对象
    const Tensor& bias = *bias_maybe_owned;
    
    // 调用PyTorch的慢速3D卷积前向传播函数，将输出写入到指定的张量output中
    return at::slow_conv3d_forward_out(
        output,         // 输出张量的引用，用于存储卷积结果
        self,           // 调用该函数的卷积模块自身
        weight,         // 卷积核的权重张量
        kernel_size,    // 卷积核的尺寸
        bias,           // 可选的偏置张量，如果没有则为nullptr
        stride,         // 卷积的步长
        padding);       // 卷积的填充方式
}

Tensor slow_conv3d(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding) {
  // [Note: hacky wrapper removal for optional tensor]
  // 从可选的 Tensor 类型中获取可能拥有的 Tensor 对象
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 从 MaybeOwned<Tensor> 中获取实际的 bias Tensor
  const Tensor& bias = *bias_maybe_owned;

  // 调用 ATen 库中的 slow_conv3d_forward 函数进行 3D 慢速卷积操作
  return at::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
}

} // namespace at::native
```