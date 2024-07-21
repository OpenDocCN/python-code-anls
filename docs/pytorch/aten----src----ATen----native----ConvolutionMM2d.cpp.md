# `.\pytorch\aten\src\ATen\native\ConvolutionMM2d.cpp`

```
// 定义宏以仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量相关头文件
#include <ATen/core/Tensor.h>
// 包含调度相关头文件
#include <ATen/Dispatch.h>
// 包含并行处理相关头文件
#include <ATen/Parallel.h>
// 包含张量工具相关头文件
#include <ATen/TensorUtils.h>
// 包含整数除法相关头文件
#include <ATen/div_rtn.h>
// 包含卷积相关工具头文件
#include <ATen/native/ConvUtils.h>
// 包含CPU BLAS操作相关头文件
#include <ATen/native/CPUBlas.h>
// 包含2D展开操作相关头文件
#include <ATen/native/Unfold2d.h>
// 包含整数范围工具头文件
#include <c10/util/irange.h>

// 如果未定义每个操作符的头文件，则包含功能相关头文件，否则包含适用于慢速2D卷积的特定头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_slow_conv2d_backward_native.h>
#include <ATen/ops/_slow_conv2d_forward.h>
#include <ATen/ops/_slow_conv2d_forward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/thnn_conv2d_native.h>
#endif

// 命名空间：at::native
namespace at::native {

// 匿名命名空间，用于静态函数
namespace {

// 静态函数：计算2D列向量
static Tensor compute_columns2d(
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef kernel_size,
    bool is_channels_last) {
  
  // 提取核大小、填充、步幅等参数
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];
  
  // 提取输入张量的形状参数
  const int64_t batch_size = input.size(0);
  const int64_t n_input_plane = input.size(1);
  const int64_t input_height = input.size(2);
  const int64_t input_width = input.size(3);
  
  // 计算输出张量的形状参数
  const int64_t output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =  (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  // 声明一个Tensor变量columns用于存储计算结果
  Tensor columns;
  
  // 如果是1x1的特殊情况，直接将columns视作输入的一个视图
  if ((kernel_height == 1) && (stride_height == 1) && (pad_height == 0) &&
      (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
    if (is_channels_last) {
      // 如果是channels_last，将columns作为输入的重构视图
      columns = input.as_strided({batch_size, output_height * output_width, n_input_plane},
          {output_height * output_width * n_input_plane, n_input_plane, 1}).detach();
    } else {
      // 否则，将columns作为重新视图化的输入
      columns = input.view({batch_size, n_input_plane, output_height * output_width}).detach();
    }
  } else {
    // 否则，根据输入的选项创建一个空的Tensor，用于存储计算的columns
    int64_t row = is_channels_last ?
        output_height * output_width : n_input_plane * kernel_height * kernel_width;
    int64_t col = is_channels_last ?
        kernel_height * kernel_width * n_input_plane : output_height * output_width;
    columns = at::empty({batch_size, row, col}, input.options());
    // 使用宏展开，处理所有数据类型（包括 kBFloat16 和 kHalf），生成函数名为 "slow_conv2d_cpu"
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "slow_conv2d_cpu", [&]{
      // 获取输入张量的访问器，数据类型为 scalar_t，4 维张量
      auto input_a = input.accessor<const scalar_t, 4>();
      // 获取列矩阵的访问器，数据类型为 scalar_t，3 维张量
      auto columns_a = columns.accessor<scalar_t, 3>();

      // 并行处理每个批次中的数据
      at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
        // 对于每个范围内的索引 t 进行迭代处理
        for (const auto t : c10::irange(start, end)) {
          // 获取当前批次中的输入张量和对应的列矩阵
          auto input_t = input_a[t];
          auto columns_t = columns_a[t];
          // 调用 unfolded2d_copy_stub 函数，复制展开的 2D 数据块
          unfolded2d_copy_stub(
              kCPU,  // 使用 CPU 计算
              c10::CppTypeToScalarType<scalar_t>::value,  // 数据类型转换为 scalar_t 对应的标量类型
              columns_t.data(),  // 列矩阵数据指针
              input_t.data(),  // 输入张量数据指针
              kernel_height,  // 卷积核的高度
              kernel_width,   // 卷积核的宽度
              stride_height,  // 步幅的高度
              stride_width,   // 步幅的宽度
              pad_height,     // 填充的高度
              pad_width,      // 填充的宽度
              n_input_plane,  // 输入平面数
              input_height,   // 输入张量的高度
              input_width,    // 输入张量的宽度
              output_height,  // 输出张量的高度
              output_width,   // 输出张量的宽度
              is_channels_last);  // 是否通道在最后
        }
      });
    });
  }

  // 返回连续存储的列矩阵
  return columns.contiguous();
}

static inline void slow_conv2d_shape_check(
    const Tensor& input,                                     // 输入张量
    const Tensor& grad_output,                               // 梯度输出张量
    const Tensor& weight,                                    // 权重张量
    const Tensor& bias,                                      // 偏置张量
    int64_t kernel_height,                                   // 卷积核高度
    int64_t kernel_width,                                    // 卷积核宽度
    int64_t stride_height,                                   // 高度方向步长
    int64_t stride_width,                                    // 宽度方向步长
    int64_t pad_height,                                      // 高度方向填充
    int64_t pad_width,                                       // 宽度方向填充
    bool weight_optional) {                                  // 权重是否可选

  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0,                 // 检查卷积核大小是否大于零
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);

  TORCH_CHECK(
      stride_width > 0 && stride_height > 0,                 // 检查步长是否大于零
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);

  if (weight.defined()) {                                   // 如果权重张量已定义
    TORCH_CHECK(
        weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 4),  // 检查权重张量是否非空且为2D或4D
        "non-empty 2D or 4D weight tensor expected, but got: ",
        weight.sizes());
    if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(0));           // 检查偏置张量维度是否匹配
    }
  } else {
    TORCH_CHECK(weight_optional, "weight tensor is undefined");  // 如果权重张量未定义，检查是否可选
  }

  const int64_t ndim = input.dim();                         // 获取输入张量的维度数
  const int64_t dim_planes = 1;                             // 定义平面维度
  const int64_t dim_height = 2;                             // 定义高度维度
  const int64_t dim_width = 3;                              // 定义宽度维度

  // 允许批量大小和通道大小为空，但不允许其他维度为空
  TORCH_CHECK(ndim == 4, "Expected 4D input tensor, but got: ", input.sizes());
  for (const auto dim : c10::irange(2, ndim)) {
    TORCH_CHECK(input.size(dim) != 0,
                "Expected non-zero size for input dimension ", dim,
                ", but got input shape: ", input.sizes(), ". Only the batch and channel dimensions support size 0.");
  }

  const int64_t input_height = input.size(dim_height);      // 获取输入张量的高度
  const int64_t input_width = input.size(dim_width);        // 获取输入张量的宽度

  const int64_t exact_input_height = input_height + 2 * pad_height;  // 计算精确的填充后的输入高度
  const int64_t exact_input_width = input_width + 2 * pad_width;     // 计算精确的填充后的输入宽度

  TORCH_CHECK(
      exact_input_height >= kernel_height && exact_input_width >= kernel_width,
      "Calculated padded input size per channel: (",
      exact_input_height,
      " x ",
      exact_input_width,
      "). ",
      "Kernel size: (",
      kernel_height,
      " x ",
      kernel_width,
      "). Kernel size can't be greater than actual input size");

  const int64_t output_height =
      div_rtn<int64_t>(exact_input_height - kernel_height, stride_height) + 1;  // 计算输出的高度
  const int64_t output_width =
      div_rtn<int64_t>(exact_input_width - kernel_width, stride_width) + 1;      // 计算输出的宽度

  TORCH_CHECK(
      output_width >= 1 && output_height >= 1,               // 检查输出的高度和宽度是否大于等于1
      "Given input size per channel: (",
      input_height,
      " x ",
      input_width,
      "). "
      "Calculated output size per channel: (",
      output_height,
      " x ",
      output_width,
      "). Output size is too small");

  if (weight.defined()) {                                   // 如果权重张量已定义
    int64_t n_input_plane = weight.size(1);                 // 获取输入平面数
    if (weight.dim() == 2) {
      n_input_plane /= (kernel_height * kernel_width);      // 如果是2D权重，计算输入平面数
    }
    // 检查输入张量是否有第二维度并且不为零，若有则调用函数检查其维度尺寸
    if (input.size(1) != 0) {
      check_dim_size(input, ndim, dim_planes, n_input_plane);
    }
  }

  // 检查梯度输出张量是否已定义
  if (grad_output.defined()) {
    // 如果权重张量已定义，则获取其第一维度大小，并调用函数检查梯度输出张量的对应维度尺寸
    if (weight.defined()) {
      int64_t n_output_plane = weight.size(0);
      check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
    } else if (bias.defined()) {
      // 如果偏置张量已定义，则检查其元素数量是否大于零，然后获取其第一维度大小（如果为零则为1），最后调用函数检查梯度输出张量的对应维度尺寸
      TORCH_CHECK(bias.numel() > 0, "non-empty bias tensor expected");
      const int64_t n_output_plane = bias.dim() == 0 ? 1 : bias.size(0);
      check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
    }
    // 调用函数检查梯度输出张量的高度维度尺寸
    check_dim_size(grad_output, ndim, dim_height, output_height);
    // 调用函数检查梯度输出张量的宽度维度尺寸
    check_dim_size(grad_output, ndim, dim_width, output_width);
  }
}

// 内联函数，用于将权重张量按指定的内存格式进行连续化处理
static inline Tensor view_weight_2d(const Tensor& weight_,
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) {
  // 连续化权重张量并赋给新的张量变量 weight
  Tensor weight = weight_.contiguous(memory_format);
  // 如果权重张量是四维的
  if (weight.dim() == 4) {
    // 获取权重张量的各个维度大小
    const int64_t s1 = weight.size(0);
    const int64_t s2 = weight.size(1) * weight.size(2) * weight.size(3);
    // 如果指定了 ChannelsLast 内存格式
    return memory_format == at::MemoryFormat::ChannelsLast
        ? weight.as_strided({s1, s2}, {s2, 1}) // CL: 视图为 {oc, kh*kw*ic}
        : weight.view({s1, s2}); // CF: 视图为 {oc, ic*kh*kw}
  } else {
    // 如果权重张量不是四维，则直接返回原始权重张量
    return weight;
  }
}

// 模板函数，用于实现二维卷积的输出更新计算
template <typename scalar_t>
static void slow_conv2d_update_output_frame(
    TensorAccessor<const scalar_t, 3> input,
    TensorAccessor<scalar_t, 3> output,
    TensorAccessor<const scalar_t, 2> weight,
    bool has_bias,
    TensorAccessor<scalar_t, 2> finput,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t n_output_plane,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last) {
  // 根据是否有偏置项设置 beta 的值
  const int beta = has_bias ? 1 : 0;

  // 计算输出 = 权重 * 输入
  // 注意 gemm 函数要求 Fortran 排序，所以三个矩阵都需要转置
  // 通过交换参数顺序取消转置的影响，因为 C == AB <=> T(C) == T(B)T(A)
  if (is_channels_last) {
    const int64_t m = n_output_plane;
    const int64_t n = output_height * output_width;
    const int64_t k = n_input_plane * kernel_height * kernel_width;

    const int64_t lda = k;
    const int64_t ldb = k;
    const int64_t ldc = m;

    // 调用 CPU BLAS 库的 gemm 函数进行矩阵乘法计算
    at::native::cpublas::gemm(
        TransposeType::Transpose,
        TransposeType::NoTranspose,
        m, n, k,
        static_cast<scalar_t>(1),
        weight.data(), lda,
        finput.data(), ldb,
        static_cast<scalar_t>(beta),
        output.data(), ldc);
  } else {
    const int64_t m = output_height * output_width;
    const int64_t n = n_output_plane;
    const int64_t k = n_input_plane * kernel_height * kernel_width;

    const int64_t lda = m;
    const int64_t ldb = k;
    const int64_t ldc = m;

    // 调用 CPU BLAS 库的 gemm 函数进行矩阵乘法计算
    at::native::cpublas::gemm(
        TransposeType::NoTranspose,
        TransposeType::NoTranspose,
        m, n, k,
        static_cast<scalar_t>(1),
        finput.data(), lda,
        weight.data(), ldb,
        static_cast<scalar_t>(beta),
        output.data(), ldc);
  }
}

// 模板函数，用于实现二维卷积的反向传播计算，更新梯度输入帧
template <typename scalar_t>
void slow_conv2d_backward_update_grad_input_frame(
    TensorAccessor<scalar_t, 3> grad_input,
    TensorAccessor<const scalar_t, 3> grad_output,
    TensorAccessor<const scalar_t, 2> weight,
    scalar_t *fgrad_input,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    // 如果是按照 channels_last 的顺序处理
    if (is_channels_last) {
        // 获取权重矩阵的列数
        const int64_t m = weight.size(1);
        // 计算梯度输出张量的展开尺寸（宽度*高度）
        const int64_t n = grad_output.size(1) * grad_output.size(2);
        // 获取权重矩阵的行数
        const int64_t k = weight.size(0);
    
        // 定义矩阵乘法操作数的步幅
        const int64_t lda = m;
        const int64_t ldb = k;
        const int64_t ldc = m;
    
        // 调用矩阵乘法函数，计算反向梯度输入
        at::native::cpublas::gemm(
            TransposeType::NoTranspose,                // 不转置权重矩阵
            TransposeType::NoTranspose,                // 不转置梯度输出
            m, n, k,                                   // 矩阵维度
            static_cast<scalar_t>(1),                  // 系数为 1
            weight.data(), lda,                        // 权重数据和步幅
            grad_output.data(), ldb,                   // 梯度输出数据和步幅
            static_cast<scalar_t>(0),                  // 输出初始化为 0
            fgrad_input, ldc);                         // 反向梯度输入和步幅
    } else {
        // 否则，按照 channels_first 的顺序处理
    
        // 计算梯度输出张量的展开尺寸（宽度*高度）
        const int64_t m = grad_output.size(1) * grad_output.size(2);
        // 获取权重矩阵的列数
        const int64_t n = weight.size(1);
        // 获取权重矩阵的行数
        const int64_t k = weight.size(0);
    
        // 定义矩阵乘法操作数的步幅
        const int64_t lda = m;
        const int64_t ldb = n;
        const int64_t ldc = m;
    
        // 调用矩阵乘法函数，计算反向梯度输入
        at::native::cpublas::gemm(
            TransposeType::NoTranspose,                // 不转置梯度输出
            TransposeType::Transpose,                  // 转置权重矩阵
            m, n, k,                                   // 矩阵维度
            static_cast<scalar_t>(1),                  // 系数为 1
            grad_output.data(), lda,                   // 梯度输出数据和步幅
            weight.data(), ldb,                        // 权重数据和步幅
            static_cast<scalar_t>(0),                  // 输出初始化为 0
            fgrad_input, ldc);                         // 反向梯度输入和步幅
    }
    
    // 调用 unfolded2d_acc_stub 函数，计算卷积操作的反向传播
    unfolded2d_acc_stub(
        kCPU,                                           // CPU 上的计算
        c10::CppTypeToScalarType<scalar_t>::value,       // 标量类型转换
        fgrad_input,                                    // 反向梯度输入
        grad_input.data(),                              // 梯度输入数据
        kernel_height,                                  // 卷积核高度
        kernel_width,                                   // 卷积核宽度
        stride_height,                                  // 步幅高度
        stride_width,                                   // 步幅宽度
        pad_height,                                     // 高度填充
        pad_width,                                      // 宽度填充
        grad_input.size(0),                             // 输入张量的批量大小
        grad_input.size(1),                             // 输入张量的通道数
        grad_input.size(2),                             // 输入张量的高度
        grad_output.size(1),                            // 梯度输出的宽度
        grad_output.size(2),                            // 梯度输出的高度
        is_channels_last);                              // 是否按照 channels_last 排列
}

void slow_conv2d_backward_out_cpu_template(
    Tensor& grad_input,                                // 用于保存梯度的输入张量
    const Tensor& grad_output_,                        // 梯度的输出张量（前一层传来的梯度）
    const Tensor& input_,                              // 前向传播时的输入张量
    const Tensor& weight_,                             // 卷积核权重张量
    IntArrayRef kernel_size,                           // 卷积核尺寸
    IntArrayRef stride,                                // 步幅大小
    IntArrayRef padding)                               // 填充大小
{
  const int64_t kernel_height = kernel_size[0];         // 卷积核高度
  const int64_t kernel_width = kernel_size[1];          // 卷积核宽度
  const int64_t pad_height = padding[0];                // 垂直填充数
  const int64_t pad_width = padding[1];                 // 水平填充数
  const int64_t stride_height = stride[0];              // 垂直步幅大小
  const int64_t stride_width = stride[1];               // 水平步幅大小

  bool use_channels_last = thnn_conv_use_channels_last(input_, weight_);  // 检查是否使用通道优先内存布局
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;  // 根据通道优先性确定内存布局格式

  const Tensor weight = view_weight_2d(weight_, memory_format);  // 根据内存布局格式视图化权重张量
  slow_conv2d_shape_check(
      input_,
      grad_output_,
      weight,
      Tensor(),
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      false);                                              // 执行慢速卷积的形状检查

  const Tensor input = input_.contiguous(memory_format);    // 获取连续存储的输入张量

  // 计算经过列化处理的数据形状（不包括批处理维度）
  const int64_t batch_size = input.size(0);                 // 批处理大小
  const int64_t n_input_plane = input.size(1);              // 输入平面数
  const int64_t input_height = input.size(2);               // 输入高度
  const int64_t input_width = input.size(3);                // 输入宽度
  const int64_t output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;  // 输出高度
  const int64_t output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;          // 输出宽度
  const int64_t fgrad_input_size = n_input_plane * kernel_height * kernel_width * output_height * output_width;  // 梯度输入大小

  const Tensor grad_output = grad_output_.contiguous(memory_format);  // 获取连续存储的梯度输出张量
  grad_input.resize_as_(input, memory_format);             // 调整梯度输入张量的大小与输入张量相同
  grad_input.zero_();                                      // 将梯度输入张量置零
  TORCH_CHECK(grad_input.is_contiguous(memory_format), "slow_conv2d: grad_input must be contiguous");  // 检查梯度输入张量是否连续存储

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, input.scalar_type(), "slow_conv2d_cpu_grad_input", [&] {
    auto grad_output_a = grad_output.accessor<const scalar_t, 4>();      // 访问梯度输出张量的访问器
    auto grad_input_a = grad_input.accessor<scalar_t, 4>();              // 访问梯度输入张量的访问器
    auto weight_a = weight.accessor<const scalar_t, 2>();                // 访问权重张量的访问器

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {  // 并行处理每个批次的数据
      auto fgrad_input = std::make_unique<scalar_t[]>(fgrad_input_size);  // 创建梯度输入的临时缓冲区
      for (const auto t : c10::irange(start, end)) {                      // 遍历每个批次
        auto grad_input_t = grad_input_a[t];                              // 获取当前批次的梯度输入张量
        auto grad_output_t = grad_output_a[t];                            // 获取当前批次的梯度输出张量
        slow_conv2d_backward_update_grad_input_frame(
            grad_input_t,
            grad_output_t,
            weight_a,
            fgrad_input.get(),
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            pad_height,
            pad_width,
            use_channels_last);                                          // 更新当前批次的梯度输入张量
      }
    });
  });
}
  // 根据 channels_last 参数选择不同的矩阵乘法操作，计算 grad_weight += grad_output.reshape({grad_output.shape(0), -1}) * finput.T
  // 注意 gemm 函数要求 Fortran 排序，因此所有三个矩阵都被转置。
  // 通过交换参数顺序取消这一要求，因为 C == AB <=> T(C) == T(B)T(A)

  if (is_channels_last) {
    // 如果 channels_last 为 true，获取矩阵的维度信息
    const int64_t m = finput.size(1);  // finput 的第二维度
    const int64_t n = grad_output.size(0);  // grad_output 的第一维度
    const int64_t k = grad_output.size(1) * grad_output.size(2);  // grad_output 的第二维度乘以第三维度

    // 设置矩阵乘法中的参数
    const int64_t lda = m;  // finput 的列数
    const int64_t ldb = n;  // grad_output 的行数
    const int64_t ldc = m;  // grad_weight 的列数

    // 调用 ATen 的 gemm 函数进行矩阵乘法
    at::native::cpublas::gemm(
        TransposeType::NoTranspose,  // 不转置 finput
        TransposeType::Transpose,    // 转置 grad_output
        m, n, k,
        static_cast<scalar_t>(1),    // 缩放因子为 1
        finput.data(), lda,          // finput 数据和 leading dimension
        grad_output.data(), ldb,     // grad_output 数据和 leading dimension
        static_cast<scalar_t>(1),    // 缩放因子为 1
        grad_weight.data(), ldc);    // grad_weight 数据和 leading dimension
  } else {
    // 如果 channels_last 为 false，获取矩阵的维度信息
    const int64_t m = finput.size(0);  // finput 的第一维度
    const int64_t n = grad_output.size(0);  // grad_output 的第一维度
    const int64_t k = grad_output.size(1) * grad_output.size(2);  // grad_output 的第二维度乘以第三维度

    // 设置矩阵乘法中的参数
    const int64_t lda = k;  // finput 的行数
    const int64_t ldb = k;  // grad_output 的行数
    const int64_t ldc = m;  // grad_weight 的列数

    // 调用 ATen 的 gemm 函数进行矩阵乘法
    at::native::cpublas::gemm(
        TransposeType::Transpose,    // 转置 finput
        TransposeType::NoTranspose,  // 不转置 grad_output
        m, n, k,
        static_cast<scalar_t>(1),    // 缩放因子为 1
        finput.data(), lda,          // finput 数据和 leading dimension
        grad_output.data(), ldb,     // grad_output 数据和 leading dimension
        static_cast<scalar_t>(1),    // 缩放因子为 1
        grad_weight.data(), ldc);    // grad_weight 数据和 leading dimension
  }
// 定义静态函数，计算慢速二维卷积操作的反向传播，更新权重的计算结果到 grad_weight 中
static void slow_conv2d_backward_weight_out_cpu_template(
    Tensor& grad_weight,                              // 权重的梯度张量
    const Tensor& input,                              // 输入张量
    const Tensor& grad_output_,                       // 输出梯度张量
    IntArrayRef kernel_size,                          // 卷积核大小
    IntArrayRef stride,                               // 步幅
    IntArrayRef padding) {                            // 填充

  // 从 kernel_size 和 padding 中获取相关参数
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  // 检查是否使用通道在后的内存格式
  bool use_channels_last = thnn_conv_use_channels_last(input, grad_weight);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  // 检查 grad_weight 是否符合指定的内存格式要求
  TORCH_CHECK(grad_weight.is_contiguous(memory_format), "slow_conv2d: grad_weight must be contiguous");

  // 将 grad_weight 重新视图为二维张量
  Tensor grad_weight_2d = view_weight_2d(grad_weight, memory_format);

  // 执行形状检查，确保慢速卷积操作的输入和输出张量的形状符合要求
  slow_conv2d_shape_check(
      input,
      grad_output_,
      grad_weight_2d,
      {},
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      true);

  // 确保 grad_output 张量符合指定的内存格式要求
  auto grad_output = grad_output_.contiguous(memory_format);

  // 计算二维卷积的列（或行）数据，存储在 finput 张量中
  Tensor finput = compute_columns2d(input, padding, stride, kernel_size, use_channels_last);

  // 获取批量大小
  const int64_t batch_size = input.size(0);

  // 根据输入张量的数据类型进行分发处理
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, input.scalar_type(), "slow_conv2d_cpu_grad_weight", [&] {
    auto grad_output_a = grad_output.accessor<const scalar_t, 4>();
    auto grad_weight_2d_a = grad_weight_2d.accessor<scalar_t, 2>();
    auto finput_a = finput.accessor<const scalar_t, 3>();

    // 遍历每个批次中的数据进行反向传播计算
    for (const auto t : c10::irange(batch_size)) {
      auto grad_output_t = grad_output_a[t];
      auto finput_t = finput_a[t];

      // 调用函数执行单帧的权重反向传播更新
      slow_conv2d_backward_weight_frame(
          grad_weight_2d_a, grad_output_t, finput_t, use_channels_last);
    }
  });
}

// 结束命名空间声明
} // namespace
  // 可选张量的包装器移除的注释，请参见 [Note: hacky wrapper removal for optional tensor]

  // 检查内核大小是否为2维
  TORCH_CHECK(kernel_size.size() == 2, "2D kernel_size expected");
  // 检查步长是否为2维
  TORCH_CHECK(stride.size() == 2, "2D stride expected");
  // 检查填充是否为2维
  TORCH_CHECK(padding.size() == 2, "2D padding expected");

  // 从可选的张量中借用 bias，并确保其有效性
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 提取内核大小、填充和步长的各个维度
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  // 确定是否使用 channels last 内存布局
  bool use_channels_last = thnn_conv_use_channels_last(self, weight_);
  auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  // 根据内存布局视图化权重矩阵为2维
  const Tensor weight_2d = view_weight_2d(weight_, memory_format);

  // 执行慢速卷积的形状检查
  slow_conv2d_shape_check(
      self,
      Tensor(),
      weight_2d,
      bias,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      false);

  // 获取连续内存中的输入张量
  const Tensor input = self.contiguous(memory_format);
  // 提取输入张量的批处理大小及其维度
  const int64_t batch_size = input.size(0);
  const int64_t n_input_plane = input.size(1);
  const int64_t input_height = input.size(2);
  const int64_t input_width = input.size(3);
  // 提取权重矩阵的输出平面数量及输出张量的高度和宽度
  const int64_t n_output_plane = weight_2d.size(0);
  const int64_t output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  // 计算2维卷积的输入列
  Tensor finput = compute_columns2d(input, padding, stride, kernel_size, use_channels_last);
  // 调整输出张量的大小，并按照内存布局初始化
  output.resize_({batch_size, n_output_plane, output_height, output_width}, memory_format);
  // 如果存在偏置，则将其复制到输出张量中
  if (bias.defined()) {
    output.copy_(bias.reshape({-1, 1, 1}));
  }
  // 检查输出张量是否是连续的，根据内存布局
  TORCH_CHECK(output.is_contiguous(memory_format), "slow_conv2d output tensor must be contiguous");

  // 根据输入张量的数据类型分发慢速卷积的计算
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "slow_conv2d_cpu", [&]{
    auto input_a = input.accessor<const scalar_t, 4>();
    auto output_a = output.accessor<scalar_t, 4>();
    auto finput_a = finput.accessor<scalar_t, 3>();
    auto weight_2d_a = weight_2d.accessor<const scalar_t, 2>();
  
    # 使用 ATen 的 parallel_for 函数并行处理批次中的每个样本
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      # 遍历从 start 到 end 范围内的每个索引 t
      for (const auto t : c10::irange(start, end)) {
        # 获取当前样本的输入张量
        auto input_t = input_a[t];
        # 获取当前样本的输出张量
        auto output_t = output_a[t];
        # 获取当前样本的 finput 张量
        auto finput_t = finput_a[t];
        # 调用 slow_conv2d_update_output_frame 函数更新当前样本的输出帧
        slow_conv2d_update_output_frame(
            input_t,                        # 输入张量
            output_t,                       # 输出张量
            weight_2d_a,                    # 二维权重张量
            bias.defined(),                 # 是否定义了偏置
            finput_t,                       # finput 张量
            kernel_height,                  # 卷积核高度
            kernel_width,                   # 卷积核宽度
            stride_height,                  # 高度方向的步长
            stride_width,                   # 宽度方向的步长
            pad_height,                     # 高度方向的填充
            pad_width,                      # 宽度方向的填充
            n_input_plane,                  # 输入平面数
            input_height,                   # 输入图像高度
            input_width,                    # 输入图像宽度
            n_output_plane,                 # 输出平面数
            output_height,                  # 输出图像高度
            output_width,                   # 输出图像宽度
            use_channels_last               # 是否使用通道在最后的布局
        );
      }
    });
  });

  # 返回更新后的输出张量
  return output;
}

// 定义一个函数 slow_conv2d_forward_cpu，用于在 CPU 上执行卷积操作的前向传播
Tensor slow_conv2d_forward_cpu(
    const Tensor& self,  // 输入张量 self，表示输入特征图
    const Tensor& weight,  // 权重张量，表示卷积核
    IntArrayRef kernel_size,  // 卷积核大小的引用
    const std::optional<Tensor>& bias_opt,  // 可选的偏置张量
    IntArrayRef stride,  // 步长的引用
    IntArrayRef padding) {  // 填充的引用

  // 查看注释：用于处理可选张量的包装移除
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;  // 获取偏置张量

  auto output = at::empty({0}, self.options());  // 创建一个空的输出张量
  // 调用 slow_conv2d_forward_out_cpu 函数执行卷积的前向传播
  at::native::slow_conv2d_forward_out_cpu(
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output);

  return output;  // 返回计算得到的输出张量
}

// 定义一个函数 slow_conv2d_backward_out_cpu，用于在 CPU 上执行卷积操作的反向传播
std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cpu(
    const Tensor& grad_output,  // 梯度输出张量
    const Tensor& self,  // 输入张量 self，表示输入特征图
    const Tensor& weight,  // 权重张量，表示卷积核
    IntArrayRef kernel_size,  // 卷积核大小的引用
    IntArrayRef stride,  // 步长的引用
    IntArrayRef padding,  // 填充的引用
    Tensor& grad_input,  // 梯度输入张量
    Tensor& grad_weight,  // 梯度权重张量
    Tensor& grad_bias) {  // 梯度偏置张量

  if (grad_input.defined()) {
    // 如果定义了梯度输入张量，则调用 slow_conv2d_backward_out_cpu_template 函数执行反向传播
    slow_conv2d_backward_out_cpu_template(
        grad_input,
        grad_output,
        self,
        weight,
        kernel_size,
        stride,
        padding);
  }

  if (grad_bias.defined()) {
    // 如果定义了梯度偏置张量，则通过对 grad_output 沿指定维度求和得到 grad_bias
    at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3});
  }

  if (grad_weight.defined()) {
    // 如果定义了梯度权重张量，则重置 grad_weight，然后调用 slow_conv2d_backward_weight_out_cpu_template 函数计算梯度
    grad_weight.resize_(weight.sizes(), weight.suggest_memory_format());
    grad_weight.zero_();
    slow_conv2d_backward_weight_out_cpu_template(
        grad_weight,
        self,
        grad_output,
        kernel_size,
        stride,
        padding);
  }

  // 返回梯度输入、梯度权重和梯度偏置张量的引用的元组
  return std::tuple<Tensor&, Tensor&, Tensor&>(
      grad_input, grad_weight, grad_bias);
}

// 定义一个函数 slow_conv2d_backward_cpu，用于在 CPU 上执行卷积操作的反向传播
std::tuple<Tensor, Tensor, Tensor> slow_conv2d_backward_cpu(
    const Tensor& grad_output,  // 梯度输出张量
    const Tensor& self,  // 输入张量 self，表示输入特征图
    const Tensor& weight,  // 权重张量，表示卷积核
    IntArrayRef kernel_size,  // 卷积核大小的引用
    IntArrayRef stride,  // 步长的引用
    IntArrayRef padding,  // 填充的引用
    std::array<bool, 3> output_mask) {  // 输出遮罩标志数组

  Tensor grad_input;  // 梯度输入张量
  Tensor grad_weight;  // 梯度权重张量
  Tensor grad_bias;  // 梯度偏置张量

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());  // 如果 output_mask[0] 为 true，则创建空的梯度输入张量
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());  // 如果 output_mask[1] 为 true，则创建空的梯度权重张量
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());  // 如果 output_mask[2] 为 true，则创建空的梯度偏置张量
  }

  // 调用 slow_conv2d_backward_out_cpu 函数执行卷积的反向传播
  at::native::slow_conv2d_backward_out_cpu(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      grad_input,
      grad_weight,
      grad_bias);

  // 返回梯度输入、梯度权重和梯度偏置张量的元组
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// 定义一个函数 thnn_conv2d_out，用于执行 THNN 卷积的输出操作
Tensor & thnn_conv2d_out(
    const Tensor & self,  // 输入张量 self，表示输入特征图
    const Tensor & weight,  // 权重张量，表示卷积核
    IntArrayRef kernel_size,  // 卷积核大小的引用
    const std::optional<Tensor>& bias_opt,  // 可选的偏置张量
    IntArrayRef stride,  // 步长的引用
    IntArrayRef padding,  // 填充的引用
    Tensor & output) {  // 输出张量

  // 查看注释：用于处理可选张量的包装移除
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;  // 获取偏置张量

  // 调用 _slow_conv2d_forward_out 函数执行 THNN 卷积的输出操作，并返回输出张量的引用
  return at::_slow_conv2d_forward_out(output, self, weight, kernel_size, bias, stride, padding);
}
// 定义了一个函数 thnn_conv2d，接收参数 self, weight, kernel_size, bias_opt, stride, padding，返回 Tensor 类型
Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding) {
  // 创建一个 MaybeOwned<Tensor> 对象，该对象可以拥有或借用 bias_opt 中的 Tensor
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 从 MaybeOwned<Tensor> 中获取实际的 bias 引用
  const Tensor& bias = *bias_maybe_owned;

  // 调用 _slow_conv2d_forward 函数执行卷积操作，使用给定的参数 self, weight, kernel_size, bias, stride, padding
  return at::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
}

// 结束 namespace at::native
} // namespace at::native
```