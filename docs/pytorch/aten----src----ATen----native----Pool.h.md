# `.\pytorch\aten\src\ATen\native\Pool.h`

```
#pragma once


// 声明头，确保此头文件只被编译一次
namespace at::native {


// 使用 max_pool2d_fn 声明一个函数指针类型，用于最大池化操作
using max_pool2d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input,
    int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);


// 使用 max_pool2d_backward_fn 声明一个函数指针类型，用于最大池化反向传播操作
using max_pool2d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);


// 声明 max_pool2d_kernel 函数指针，用于实际执行最大池化的调度分发
DECLARE_DISPATCH(max_pool2d_fn, max_pool2d_kernel);


// 声明 max_pool2d_backward_kernel 函数指针，用于实际执行最大池化反向传播的调度分发
DECLARE_DISPATCH(max_pool2d_backward_fn, max_pool2d_backward_kernel);


// avg_pool2d_fn 与 avg_pool2d_backward_fn 具有相同的函数签名，用于平均池化正向和反向传播操作
using avg_pool2d_fn = void(*)(const Tensor& output, const Tensor& input, int64_t kW, int64_t kH,
    int64_t dW, int64_t dH, int64_t padW, int64_t padH, bool count_include_pad, std::optional<int64_t> divisor_override);


// 使用 avg_pool2d_backward_fn 声明一个函数指针类型，用于平均池化反向传播操作
using avg_pool2d_backward_fn = void(*)(const Tensor& output, const Tensor& input, int kW, int kH,
    int dW, int dH, int padW, int padH, bool count_include_pad, std::optional<int64_t> divisor_override);


// 声明 avg_pool2d_kernel 函数指针，用于实际执行平均池化的调度分发
DECLARE_DISPATCH(avg_pool2d_fn, avg_pool2d_kernel);


// 声明 avg_pool2d_backward_kernel 函数指针，用于实际执行平均池化反向传播的调度分发
DECLARE_DISPATCH(avg_pool2d_backward_fn, avg_pool2d_backward_kernel);


// avg_pool3d_fn 与 avg_pool3d_backward_fn 具有相同的函数签名，用于三维平均池化正向和反向传播操作
using avg_pool3d_fn = void(*)(const Tensor& output, const Tensor& input,
    int64_t kW, int64_t kH, int64_t kD, int64_t dW, int64_t dH, int64_t dD,
    int64_t padW, int64_t padH, int64_t padD, bool count_include_pad,
    std::optional<int64_t> divisor_override);


// 使用 avg_pool3d_backward_fn 声明一个函数指针类型，用于三维平均池化反向传播操作
using avg_pool3d_backward_fn = void(*)(const Tensor& output, const Tensor& input,
    int kW, int kH, int kD, int dW, int dH, int dD,
    int padW, int padH, int padD, bool count_include_pad,
    std::optional<int64_t> divisor_override);


// 声明 avg_pool3d_kernel 函数指针，用于实际执行三维平均池化的调度分发
DECLARE_DISPATCH(avg_pool3d_fn, avg_pool3d_kernel);


// 声明 avg_pool3d_backward_kernel 函数指针，用于实际执行三维平均池化反向传播的调度分发
DECLARE_DISPATCH(avg_pool3d_backward_fn, avg_pool3d_backward_kernel);


// max_pool3d_fn 与 max_pool3d_backward_fn 具有相同的函数签名，用于三维最大池化正向和反向传播操作
using max_pool3d_fn = void(*)(Tensor& output, Tensor& indices, const Tensor& input,
    int kW, int kH, int kD, int dW, int dH, int dD, int pW, int pH, int pD, int dilationW, int dilationH, int dilationD);


// 使用 max_pool3d_backward_fn 声明一个函数指针类型，用于三维最大池化反向传播操作
using max_pool3d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);


// 声明 max_pool3d_kernel 函数指针，用于实际执行三维最大池化的调度分发
DECLARE_DISPATCH(max_pool3d_fn, max_pool3d_kernel);


// 声明 max_pool3d_backward_kernel 函数指针，用于实际执行三维最大池化反向传播的调度分发
DECLARE_DISPATCH(max_pool3d_backward_fn, max_pool3d_backward_kernel);


namespace {


// 安全的类型转换函数，确保将 src_t 类型的值安全地转换为 dest_t 类型
template <typename dest_t, typename src_t>
inline dest_t
safe_downcast(src_t v)
{
  // 使用 TORCH_CHECK 确保转换后的值在 dest_t 类型的范围内
  TORCH_CHECK(std::numeric_limits<dest_t>::min() <= v && v <= std::numeric_limits<dest_t>::max(),
              "integer out of range");

  // 执行静态类型转换并返回结果
  return static_cast<dest_t>(v);
}


// 池化操作的输出形状计算函数，计算二维池化操作的输出形状，考虑填充和步幅
template<typename T>
inline T pooling_output_shape_pad_lr(
        T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
        bool ceil_mode) {
    // 计算池化操作后的输出尺寸
    T outputSize = div_rtn<T>(
        inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
        (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode) {
        // 如果使用 ceil 模式，则确保最后一个池化操作在图像内部开始
        // 这是为了避免在 ceil 模式下出现问题
        if ((outputSize - 1) * stride >= inputSize + pad_l) {
            // 如果最后一个池化操作超出了输入图像范围，则减少输出尺寸
            --outputSize;
        }
    }
    // 返回调整后的输出尺寸
    return outputSize;
// 模板函数，计算池化层输出的尺寸
template<typename T>
inline T pooling_output_shape(
      T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    // 检查步长是否为零，如果是则抛出异常
    TORCH_CHECK(stride != 0, "stride should not be zero");
    // 检查填充是否为非负数，如果不是则抛出异常
    TORCH_CHECK(pad >= 0,
                "pad must be non-negative, but got pad: ", pad);
    // 检查填充是否超过有效核大小的一半，如果是则抛出异常
    TORCH_CHECK(pad <= ((kernelSize - 1) * dilation + 1) / 2,
                "pad should be at most half of effective kernel size, but got pad=",
                pad, ", kernel_size=", kernelSize, " and dilation=", dilation)
    // 调用具体计算池化层输出尺寸的函数，并返回结果
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

// 模板函数，计算对称填充的左右填充量
template <typename T>
std::pair<T, T> _pooling_same_mode_padding_lr(
    T inputSize, T kernelSize, T stride, T dilation) {
  // 根据步长和核大小计算总的填充量
  auto total_padding = T(dilation) * (kernelSize - 1);

  // 如果步长大于2且总填充量是奇数，则尽量选择对称填充
  if (stride > 2 && (total_padding % 2 == 1)) {
    // 计算输出尺寸时，向下取整会留出一些余地
    auto wiggle_room = inputSize % stride - 1;
    // 如果有余地，则减少一个填充量
    if (wiggle_room > 0) {
      total_padding = total_padding - 1;
    }
  }

  // 计算左填充量
  auto left = total_padding / 2;
  // 返回左右填充量的pair
  return {left, total_padding - left};
}

// 整数版本的池化层对称填充函数，调用通用的模板函数
inline std::pair<int64_t, int64_t> pooling_same_mode_padding_lr(
    int64_t inputSize, int64_t kernelSize, int64_t stride, int64_t dilation) {
  return _pooling_same_mode_padding_lr(inputSize, kernelSize, stride, dilation);
}

// 符号整数版本的池化层对称填充函数，调用通用的模板函数
inline std::pair<c10::SymInt, c10::SymInt> pooling_same_mode_padding_lr(
    c10::SymInt inputSize, c10::SymInt kernelSize, c10::SymInt stride, c10::SymInt dilation) {
  return _pooling_same_mode_padding_lr(std::move(inputSize), std::move(kernelSize), std::move(stride), std::move(dilation));
}

// 池化层参数检查函数，用于检查输入张量和池化层参数是否符合规范
// AveragePool2d/DilatedMaxPool2d (forward)
inline void
pool2d_shape_check(
  const Tensor& input,
  int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth, MemoryFormat memory_format)
{
  // 获取输入张量的维度数和输出平面数
  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  // 检查核大小是否大于零
  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got ",
              "kH: ", kH, " kW: ", kW);
  // 检查步长是否大于零
  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got "
              "dH: ", dH, " dW: ", dW);
  // 检查扩展大小是否大于零
  TORCH_CHECK(dilationH > 0 && dilationW > 0,
              "dilation should be greater than zero, but got ",
              "dilationH: ", dilationH, " dilationW: ", dilationW);

  // 检查输入张量的维度是否有效
  bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
  // 如果内存格式为ChannelsLast，则期望张量格式为NHWC，并允许N维度为0
  if (memory_format == at::MemoryFormat::ChannelsLast){
    // Expect tensor in NHWC format and allow 0-dim only for N.
    # 如果输入张量的维度为4且满足有效维度条件，并且第4维的大小不为0，则通过检查通过
    if (ndim == 4 && valid_dims && input.size(3) != 0) {
        # 对于通道最后布局，期望输入是4维张量（批处理模式），可选的批处理大小为0维，但实际得到的是：输入张量的尺寸
        TORCH_CHECK("Expected 4D (batch mode) tensor expected for input with channels_last layout"
                    " with optional 0 dim batch size for input, but got: ", input.sizes());
    } else {
        # 对于输入张量是3维或者4维（批处理模式），并且可选的批处理大小为0维，但实际得到的是：输入张量的尺寸
        TORCH_CHECK((ndim == 3 && input.size(0) != 0 && valid_dims) ||
                    (ndim == 4 && valid_dims && input.size(3) != 0),
                    "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got:",
                    input.sizes());
    }
    
    # 检查填充大小是否小于或等于卷积核大小的一半
    TORCH_CHECK(kW/2 >= padW && kH/2 >= padH,
                "pad should be smaller than or equal to half of kernel size, but got ",
                "padW = ", padW, ", padH = ", padH, ", kW = ", kW, ", kH = ", kH);
    
    # 检查输出宽度和高度是否至少为1
    TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
                "Given input size: (",
                nInputPlane, "x", inputHeight, "x", inputWidth, "). ",
                "Calculated output size: (",
                nOutputPlane, "x", outputHeight, "x", outputWidth, "). ",
                "Output size is too small");
// DilatedMaxPool2d (backward)
inline void
max_pool2d_backward_shape_check(
  const Tensor& input,  // 输入张量
  const Tensor& gradOutput,  // 梯度输出张量
  const Tensor& indices,  // 索引张量
  int kH, int kW,  // 池化核的高度和宽度
  int dH, int dW,  // 池化步长的高度和宽度
  int padH, int padW,  // 填充的高度和宽度
  int dilationH, int dilationW,  // 膨胀率的高度和宽度
  int64_t nInputPlane,  // 输入平面数
  int64_t inputHeight, int64_t inputWidth,  // 输入张量的高度和宽度
  int64_t outputHeight, int64_t outputWidth,  // 输出张量的高度和宽度
  MemoryFormat memory_format)  // 内存格式
{
  // 检查池化的形状
  pool2d_shape_check(
    input,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, memory_format);

  const int64_t ndim = input.ndimension();  // 获取输入张量的维度数
  const int64_t nOutputPlane = nInputPlane;  // 输出平面数与输入平面数相同

  // 检查梯度输出张量的维度是否符合预期
  check_dim_size(gradOutput, ndim, ndim-3, nOutputPlane);
  check_dim_size(gradOutput, ndim, ndim-2, outputHeight);
  check_dim_size(gradOutput, ndim, ndim-1, outputWidth);

  // 检查索引张量的维度是否符合预期
  check_dim_size(indices, ndim, ndim-3, nOutputPlane);
  check_dim_size(indices, ndim, ndim-2, outputHeight);
  check_dim_size(indices, ndim, ndim-1, outputWidth);
}

// AveragePool2d (backward)
inline void
avg_pool2d_backward_shape_check(
  const Tensor& input,  // 输入张量
  const Tensor& gradOutput,  // 梯度输出张量
  int64_t /*nbatch*/,  // 批次大小（注释掉，未使用）
  int kH, int kW,  // 池化核的高度和宽度
  int dH, int dW,  // 池化步长的高度和宽度
  int padH, int padW,  // 填充的高度和宽度
  int64_t nInputPlane,  // 输入平面数
  int64_t inputHeight, int64_t inputWidth,  // 输入张量的高度和宽度
  int64_t outputHeight, int64_t outputWidth,  // 输出张量的高度和宽度
  MemoryFormat memory_format)  // 内存格式
{
  // 检查池化的形状
  pool2d_shape_check(
    input,
    kH, kW, dH, dW, padH, padW, 1, 1,
    nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    memory_format);

  const int64_t ndim = input.ndimension();  // 获取输入张量的维度数
  const int64_t nOutputPlane = nInputPlane;  // 输出平面数与输入平面数相同

  // 检查梯度输出张量的维度是否符合预期
  check_dim_size(gradOutput, ndim, ndim-3, nOutputPlane);
  check_dim_size(gradOutput, ndim, ndim-2, outputHeight);
  check_dim_size(gradOutput, ndim, ndim-1, outputWidth);
}

// AveragePool3d/DilatedMaxPool3d (forward)
inline void
pool3d_shape_check(
  const Tensor& input,  // 输入张量
  int64_t nslices,  // 输入张量的切片数
  int kT, int kH, int kW,  // 池化核的时间、高度和宽度
  int dT, int dH, int dW,  // 池化步长的时间、高度和宽度
  int pT, int pH, int pW,  // 填充的时间、高度和宽度
  int dilationT, int dilationH, int dilationW,  // 膨胀率的时间、高度和宽度
  int64_t itime, int64_t iheight, int64_t iwidth,  // 输入张量的时间、高度和宽度
  int64_t otime, int64_t oheight, int64_t owidth,  // 输出张量的时间、高度和宽度
  const char *fn_name,  // 函数名
  bool check_input_size=false)  // 是否检查输入大小
{
  const int64_t ndim = input.ndimension();  // 获取输入张量的维度数

  // 检查池化核的大小是否合法
  TORCH_CHECK(kT > 0 && kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got ",
              "kT: ", kT, " kH: ", kH, " kW: ", kW);
  // 检查池化步长是否合法
  TORCH_CHECK(dT > 0 && dW > 0 && dH > 0,
              "stride should be greater than zero, but got ",
              "dT: ", dT, " dH: ", dH, " dW: ", dW);
  // 检查膨胀率是否合法
  TORCH_CHECK(dilationT > 0 && dilationW > 0 && dilationH > 0,
              "dilation should be greater than zero, but got ",
              "dilationT: ", dilationT, " dilationH: ", dilationH, " dilationW: ", dilationW);

  // 检查输入张量的维度是否为4D或5D
  TORCH_CHECK(ndim == 4 || ndim == 5,
              fn_name, ": Expected 4D or 5D tensor for input, but got: ", input.sizes());

  // 遍历张量的每个维度
  for (const auto i : c10::irange(ndim)) {
    if (ndim == 5 && i == 0) {
      // 批次维度的大小可以为0，跳过检查
      continue;
    }
    // 检查输入张量的第 i 维度是否大于 0，如果不是则抛出错误信息
    TORCH_CHECK(
        input.size(i) > 0,
        fn_name,
        ": Expected input's non-batch dimensions to have positive length,"
        " but input has a shape of ",
        input.sizes(),
        " and non-batch dimension ",
        input.size(i),
        " has length zero!")
    }
    
    // 如果需要检查输入尺寸（针对 AveragePool3d），则检查输入图像尺寸是否大于或等于核的尺寸
    if (check_input_size) { // AveragePool3d
        TORCH_CHECK(itime >= kT && iheight >= kH && iwidth >= kW,
                    "input image ", "(T: ", itime, " H: ", iheight, " W: ", iwidth, ") smaller than ",
                    "kernel size ", "(kT: ", kT, " kH: ", kH, " kW: ", kW, ")");
    }
    
    // 检查填充值是否小于或等于核大小的一半，如果不是则抛出错误信息
    TORCH_CHECK(kT/2 >= pT && kW/2 >= pW && kH/2 >= pH,
                "pad should be smaller than or equal to half of kernel size, but got "
                "kT: ", kT, " kW: ", kW, " kH: ", kH, " padT: ", pT, " padW: ", pW, " padH: ", pH);
    
    // 检查输出时间、高度和宽度是否大于或等于 1，如果不是则抛出错误信息
    TORCH_CHECK(otime >= 1 && owidth >= 1 && oheight >= 1,
                "Given input size: (",
                nslices,"x", itime, "x", iheight, "x", iwidth, "). ",
                "Calculated output size: (",
                nslices, "x", otime, "x", oheight, "x", owidth, "). ",
                "Output size is too small");
    
    
    这段代码是对张量操作中的不同条件进行检查，并在条件不符合预期时抛出错误信息。
// 匿名命名空间结束，用于隐藏函数和变量避免全局作用域污染
} // anonymous namespace

// 结束 native 命名空间，该命名空间用于包含与本地 (native) 实现相关的函数和数据结构
} // namespace at::native
```