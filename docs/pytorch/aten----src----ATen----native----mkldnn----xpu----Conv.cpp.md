# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\Conv.cpp`

```py
#include <vector>  // 引入标准库中的向量容器

#include <ATen/core/ATen_fwd.h>  // 引入 ATen 库的前向声明
#include <ATen/core/interned_strings.h>  // 引入 ATen 库中的内部字符串处理
#include <ATen/ops/full.h>  // 引入 ATen 库中的全连接操作相关头文件
#include <ATen/ops/neg.h>  // 引入 ATen 库中的取负操作相关头文件
#include <c10/core/Scalar.h>  // 引入 C10 库中的标量定义
#include <c10/util/Exception.h>  // 引入 C10 库中的异常处理工具
#include <c10/util/Optional.h>  // 引入 C10 库中的可选值工具
#include <ATen/native/utils/ParamUtils.h>  // 引入 ATen 库中的参数处理工具
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>  // 引入 ATen 库中的 oneDNN 细节头文件
#include <torch/library.h>  // 引入 Torch 库的库声明
#include <ATen/native/ConvUtils.h>  // 引入 ATen 库中的卷积操作工具

using namespace dnnl;  // 使用 dnnl 命名空间
using namespace at::native;  // 使用 ATen 库中的 native 命名空间
using namespace at::native::onednn;  // 使用 ATen 库中 oneDNN 命名空间

namespace at::native {  // 定义 ATen 库中的 native 命名空间
namespace xpu {  // 定义 xpu 命名空间
namespace impl {  // 定义 impl 命名空间

struct ConvParams {  // 定义结构体 ConvParams
  std::vector<int64_t> stride;  // 存储卷积的步长
  std::vector<int64_t> padding;  // 存储卷积的填充大小
  std::vector<int64_t> dilation;  // 存储卷积的扩张大小
  bool transposed;  // 表示卷积是否是转置的
  std::vector<int64_t> output_padding;  // 存储卷积输出的填充大小
  int groups;  // 表示卷积的分组数
  bool benchmark;  // 表示是否使用基准测试
  bool deterministic;  // 表示是否是确定性卷积

  bool is_strided() const;  // 判断是否有步长
  bool is_dilated() const;  // 判断是否有扩张
  bool is_padded() const;  // 判断是否有填充
  bool is_output_padding_neg() const;  // 判断输出填充是否为负数
  bool is_output_padding_big() const;  // 判断输出填充是否过大
  bool is_padding_neg() const;  // 判断填充是否为负数
  bool is_stride_nonpos() const;  // 判断步长是否为非正数
  void view1d_as_2d();  // 将一维视为二维
  bool use_cpu_depthwise3x3_winograd(
      const at::Tensor& input,
      const at::Tensor& weight) const;  // 使用 CPU 下的深度可分离 3x3 Winograd 卷积
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;  // 判断是否是深度卷积
};

std::ostream& operator<<(std::ostream& out, const ConvParams& params) {  // 重载输出流操作符，用于打印 ConvParams 结构体信息
  out << "ConvParams {"
      << "  stride = " << IntArrayRef{params.stride}
      << "  padding = " << IntArrayRef{params.padding}
      << "  dilation = " << IntArrayRef{params.dilation}
      << "  transposed = " << params.transposed
      << "  output_padding = " << IntArrayRef{params.output_padding}
      << "  groups = " << params.groups << "  benchmark = " << params.benchmark
      << "  deterministic = " << params.deterministic << "}";
  return out;
}

bool ConvParams::is_strided() const {  // 判断是否有步长
  bool is_strided = false;
  for (int s : stride) {
    is_strided |= (s != 1);
  }
  return is_strided;
}

bool ConvParams::is_dilated() const {  // 判断是否有扩张
  bool is_dilated = false;
  for (int d : dilation) {
    is_dilated |= (d != 1);
  }
  return is_dilated;
}

bool ConvParams::is_padded() const {  // 判断是否有填充
  bool is_padded = false;
  for (int p : padding) {
    is_padded |= (p != 0);
  }
  return is_padded;
}

bool ConvParams::is_output_padding_neg() const {  // 判断输出填充是否为负数
  bool is_non_neg = false;
  for (int p : output_padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

bool ConvParams::is_output_padding_big() const {  // 判断输出填充是否过大
  bool is_big = false;
  for (size_t i = 0; i < output_padding.size(); i++) {
    is_big |=
        (output_padding[i] >= stride[i] || output_padding[i] >= dilation[i]);
  }
  return is_big;
}

bool ConvParams::is_padding_neg() const {  // 判断填充是否为负数
  bool is_non_neg = false;
  for (int p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

bool ConvParams::is_stride_nonpos() const {  // 判断步长是否为非正数
  bool is_nonpos = false;
  for (int s : stride) {
    is_nonpos |= (s <= 0);
  }
  return is_nonpos;
}

void ConvParams::view1d_as_2d() {  // 将一维视为二维
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    # 在列表的开头插入0，用于填充操作
    padding.insert(padding.begin(), 0);
    # 在列表的开头插入1，用于膨胀操作
    dilation.insert(dilation.begin(), 1);
    # 在列表的开头插入0，用于输出填充操作
    output_padding.insert(output_padding.begin(), 0);
  }
}

// 返回 false，表示不使用 CPU 进行深度可分离 3x3 Winograd 卷积
bool ConvParams::use_cpu_depthwise3x3_winograd(
    const at::Tensor& input,
    const at::Tensor& weight) const {
  return false;
}

// 检查是否为深度可分离卷积的条件：非转置，输入是四维，通道数等于组数且大于1，输出通道是输入通道的整数倍
bool ConvParams::is_depthwise(const at::Tensor& input, const at::Tensor& weight)
    const {
  return !transposed && input.ndimension() == 4 && input.size(1) == groups &&
      groups > 1 && // 如果只有一个组则无意义
      weight.size(0) % input.size(1) ==
      0; // 输出通道必须是输入通道的整数倍
}

// 静态函数，检查前向传播的形状
static void check_shape_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ConvParams& params,
    bool input_is_mkldnn) {
  // 获取输入和权重的维度
  int64_t k = input.ndimension();
  int64_t weight_dim = weight.ndimension();
  // 创建权重尺寸的向量
  std::vector<int64_t> weight_sizes(weight_dim);
  // 如果权重维度等于输入维度加一，并且输入是 MKLDNN，则进行特定处理
  if ((weight_dim == k + 1) && input_is_mkldnn) {
    weight_sizes[0] = weight.size(0) * weight.size(1);
    // 复制除了前两个维度外的其余维度大小到权重尺寸中
    std::copy_n(weight.sizes().cbegin() + 2, k - 1, weight_sizes.begin() + 1);
    weight_dim = k;
  } else {
    // 否则，直接复制权重的所有维度大小到权重尺寸中
    std::copy_n(weight.sizes().cbegin(), weight_dim, weight_sizes.begin());
  }
  // 获取组数、填充、输出填充、步幅、扩张、是否转置的参数
  int64_t groups = params.groups;
  auto padding = params.padding;
  auto output_padding = params.output_padding;
  auto stride = params.stride;
  auto dilation = params.dilation;
  bool transposed = params.transposed;

  // 检查填充是否为负数，不支持负填充
  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  // 检查输出填充是否为负数，不支持负输出填充
  TORCH_CHECK(
      !params.is_output_padding_neg(),
      "negative output_padding is not supported");
  // 检查步幅是否为非正数，不支持非正步幅
  TORCH_CHECK(
      !params.is_stride_nonpos(), "non-positive stride is not supported");

  // 检查权重维度是否与输入维度匹配
  TORCH_CHECK(
      weight_dim == k,
      "Expected ",
      weight_dim,
      "-dimensional input for ",
      weight_dim,
      "-dimensional weight ",
      weight_sizes,
      ", but got ",
      k,
      "-dimensional input of size ",
      input.sizes(),
      " instead");
  // 检查权重第一维度大小是否大于等于组数
  TORCH_CHECK(
      weight_sizes[0] >= groups,
      "Given groups=",
      groups,
      ", expected weight to be at least ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");
  // 检查权重第一维度大小是否能整除组数
  TORCH_CHECK(
      weight_sizes[0] % groups == 0,
      "Given groups=",
      groups,
      ", expected weight to be divisible by ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");

  // 如果非转置卷积，进一步检查输入形状是否符合预期
  if (!transposed) {
    // 创建输入形状和卷积核形状的向量
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    // 标记卷积核尺寸是否正确
    bool kernel_size_correct = true;

    // 检查输入通道数是否等于（卷积核通道数 * 组数）
    TORCH_CHECK(
        input.size(1) == (weight_sizes[1] * groups),
        "Given groups=",
        groups,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        input.sizes(),
        " to have ",
        (weight_sizes[1] * groups),
        " channels, but got ",
        input.size(1),
        " channels instead");
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
        "Given weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[0],
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");


    // 检查是否定义了偏置 (bias)，或者偏置是一个一维向量且长度与权重的第一个维度大小相同
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
        "Given weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[0],
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");



    for (int i = 2; i < k; ++i) {
      input_shape.push_back(input.size(i) + 2 * padding[i - 2]);
      kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }


    // 遍历从2到k的范围，计算每个维度的输入形状和卷积核形状，并检查卷积核大小是否正确
    for (int i = 2; i < k; ++i) {
      // 计算当前维度的输入形状，加上对应的填充值
      input_shape.push_back(input.size(i) + 2 * padding[i - 2]);
      // 计算当前维度的卷积核形状
      kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
      // 如果当前维度的输入形状小于卷积核形状，将标记 kernel_size_correct 设为 false
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }



    TORCH_CHECK(
        input_shape.size() == kernel_shape.size(),
        "Inconsistent shape between Input and Kernel");


    // 检查输入形状和卷积核形状的维度是否一致
    TORCH_CHECK(
        input_shape.size() == kernel_shape.size(),
        "Inconsistent shape between Input and Kernel");



    if (!kernel_size_correct) {
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::ostringstream output_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      TORCH_CHECK(
          0,
          "Calculated padded input size per channel: (",
          input_ss.str(),
          "). "
          "Kernel size: (",
          kernel_ss.str(),
          "). Kernel size can't be greater than actual input size");
    }


    // 如果 kernel_size_correct 标记为 false，说明存在不正确的卷积核大小
    if (!kernel_size_correct) {
      // 创建字符串流对象，用于构建描述输入形状和卷积核形状的字符串
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::ostringstream output_ss;
      std::string separator = "";

      // 遍历输入形状和卷积核形状，将它们格式化为字符串
      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      // 报告错误，说明计算出的每个通道的填充输入大小和卷积核大小不匹配
      TORCH_CHECK(
          0,
          "Calculated padded input size per channel: (",
          input_ss.str(),
          "). "
          "Kernel size: (",
          kernel_ss.str(),
          "). Kernel size can't be greater than actual input size");
    }



  } else {
    TORCH_CHECK(
        input.size(1) == weight_sizes[0],
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        input.sizes(),
        " to have ",
        weight_sizes[0],
        " channels, but got ",
        input.size(1),
        " channels instead");
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 &&
             bias.size(0) == weight_sizes[1] * groups),
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[1] * groups,
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");
  }


  // 如果不是标准模式，而是转置模式
  } else {
    // 检查输入的通道数是否等于权重的第一个维度大小
    TORCH_CHECK(
        input.size(1) == weight_sizes[0],
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        input.sizes(),
        " to have ",
        weight_sizes[0],
        " channels, but got ",
        input.size(1),
        " channels instead");
    // 检查是否定义了偏置 (bias)，或者偏置是一个一维向量且长度符合预期
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 &&
             bias.size(0) == weight_sizes[1] * groups),
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[1] * groups,
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");
  }
} // 结束 impl 命名空间

static at::Tensor view4d(const at::Tensor& tensor) {
  // 检查输入张量的维度是否为 3
  TORCH_CHECK(
      tensor.ndimension() == 3,
      "expected 3D tensor, got tensor with ",
      tensor.ndimension(),
      " dimensions instead");
  // 返回在第二维上增加一个维度的张量
  return tensor.unsqueeze(2);
}

static at::Tensor view3d(const at::Tensor& tensor) {
  // 检查输入张量的维度是否为 4
  TORCH_CHECK(
      tensor.ndimension() == 4,
      "expected 4D tensor, got tensor with ",
      tensor.ndimension(),
      " dimensions instead");
  // 压缩张量的第二维
  return tensor.squeeze(2);
}

// 获取 OneDNN 卷积求和的属性
Attr get_onednn_conv_sum_attr(
    const Tensor& input_r,
    const Tensor& weight_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    Tensor& accumu,
    double scale,
    Tensor& output,
    bool& is_fused,
    Attr attr = Attr(),
    bool force_inplace = false) {
  // 设置 is_fused 为 true
  is_fused = true;
  // 如果 scale 等于 0，则直接返回 attr
  if (scale == 0.f)
    return attr;

  // 获取输入张量的维度
  auto ndim = input_r.ndimension();
  // 计算卷积输出的大小
  auto output_size = conv_dst_size(
      ndim,
      input_r.sizes(),
      weight_r.sizes(),
      padding_,
      padding_,
      stride_,
      dilation_);
  // 内存格式设为连续
  MemoryFormat mem_fmt = at::MemoryFormat::Contiguous;
  // 建议输入张量的内存格式
  auto input_fmt = input_r.suggest_memory_format();
  // 判断输入张量是否为 ChannelsLast 或 ChannelsLast3d 格式
  auto input_is_cl = (input_fmt == at::MemoryFormat::ChannelsLast || input_fmt == at::MemoryFormat::ChannelsLast3d);
  // 建议权重张量的内存格式
  auto weight_fmt = weight_r.suggest_memory_format();
  // 判断权重张量是否为 ChannelsLast 或 ChannelsLast3d 格式
  auto weight_is_cl = (weight_fmt == at::MemoryFormat::ChannelsLast || weight_fmt == at::MemoryFormat::ChannelsLast3d);

  // 是否传播 ChannelsLast 标签
  bool propagate_channels_last = input_is_cl || weight_is_cl;
  // 若传播 ChannelsLast 标签，则设置内存格式为对应的 ChannelsLast 标签
  if (propagate_channels_last)
    mem_fmt = get_cl_tag_by_ndim(ndim);

  // 创建一个空张量作为输出
  Tensor out = at::empty(output_size, input_r.options().memory_format(mem_fmt));
  // 若无法进行 OneDNN 二进制有效性检查，则设置 is_fused 为 false，并返回 attr
  if (!onednn::binary_valid(out, accumu)) {
    is_fused = false;
    return attr;
  }

  // 如果 scale 不等于 1.f，则进行后续元素级操作的追加
  if (scale != 1.f)
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f / scale,
        /* beta */ 0.f,
        attr.kind_with_linear);

  // 如果 force_inplace 为 true，则使用后缀 sum 进行后续操作
  if (force_inplace) {
    output = accumu;
    attr.append_post_sum(/* sum_scale */ 1.f);
  } else {
    // 否则，使用后缀 binary 进行后续操作
    attr.append_post_binary(attr.kind_with_binary_add, accumu);
  }

  // 如果 scale 不等于 1.f，则再次进行后续元素级操作的追加
  if (scale != 1.f)
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ scale,
        /* beta */ 0.f,
        attr.kind_with_linear);

  // 返回最终的属性对象
  return attr;
}
    IntArrayRef pad_nd = IntArrayRef({})) {
  // 获取输入张量的维度
  auto ndim = input_r.ndimension();
  // 检查张量维度是否为3、4或5，否则抛出错误信息
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "convolution only supports 3D, 4D, 5D tensor");
  // 获取用于 Conv/TransposedConv 的计算格式建议是否使用通道为最后一维
  bool is_channels_last_suggested = use_channels_last_for_conv(input_r, weight_r, transposed_);

  // 将输入张量和权重张量赋给局部变量，方便后续处理
  Tensor input = input_r, weight = weight_r;
  // 对于三维张量，因为 PyTorch 不支持 ChannelsLast1D，需进行格式转换
  if (ndim == 3) {
    input = view4d(input_r);
    weight = view4d(weight_r);
  }
  // 确保输入/权重/偏置/输出在期望格式下连续存储
  at::MemoryFormat mfmt = is_channels_last_suggested
      ? get_cl_tag_by_ndim(input.ndimension())
      : at::MemoryFormat::Contiguous;
  // 如果偏置存在，则保证其连续存储
  auto bias = bias_r.defined() ? bias_r.contiguous() : bias_r;
  input = input.contiguous(mfmt);
  weight = weight.contiguous(mfmt);

  // 确定权重张量的维度
  auto k = weight.ndimension();
  if (k == input.ndimension() + 1) {
    k = input.ndimension();
  }
  // 计算卷积核的空间维度
  int64_t dim = k - 2;
  // 检查权重张量的维度是否大于1，否则抛出错误信息
  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  // 创建卷积参数对象
  ConvParams params;
  if (ndim == 3) {
    // 对于三维张量，因为 PyTorch 不支持 ChannelsLast1D，需进行格式转换
    params.stride = stride_.vec();
    params.padding = padding_.vec();
    params.dilation = dilation_.vec();
    params.transposed = transposed_;
    params.output_padding = output_padding_.vec();
    params.groups = groups_;
    // 将视为1D的输入转换为2D
    params.view1d_as_2d();
  } else {
    // 对于非三维张量，扩展填充参数为所需维度
    params.stride = expand_param_if_needed(stride_, "stride", dim);
    // PyTorch 默认的卷积填充应该是一个整数值或与卷积维度匹配的值列表
    // 对于 conv2d，填充值数量应为1或2
    // 对于 conv3d，填充值数量应为1或3
    // 填充值将填充到 Conv 输入的两侧（D、H、W）
    params.padding = expand_param_if_needed(padding_, "padding", dim);
    params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
    params.transposed = transposed_;
    params.output_padding =
        expand_param_if_needed(output_padding_, "output_padding", dim);
    params.groups = groups_;
  }
  // 检查前向传播的形状是否匹配要求
  check_shape_forward(input, weight, bias, params, true);

  // 创建输出张量
  Tensor output;
  if (transposed_) {
    // 如果是转置卷积，创建输出并传播内存格式
    if (!output_r.defined()) {
      auto dst_tz = deconv_dst_size(
          input.sizes(),
          weight.sizes(),
          params.padding,
          params.stride,
          params.dilation,
          params.output_padding,
          params.groups);
      output = at::empty(dst_tz, input.options(), mfmt);
    }
    // 调用 OneDNN 的反卷积操作
    onednn::deconvolution(
        output,
        input,
        weight,
        bias,
        params.stride,
        params.padding,
        params.output_padding,
        params.dilation,
        params.groups,
        attr);
  } else {
    // 如果是普通卷积，OneDNN 支持在源张量两侧填充不同的值
    // 此处继续处理普通卷积的情况
    // 定义变量，存储填充值，按照前部、顶部、左侧和后部、底部、右侧的顺序填充
    auto padding_front_top_left = params.padding;
    auto padding_back_bottom_right = params.padding;

    // PyTorch 的 constant_pad_nd 函数：
    // 可以在 Conv 输入的不同侧（宽度、高度、深度）上进行填充
    // (padding_left, padding_right,
    //  padding_top, padding_bottom,
    //  padding_front, padding_back)
    if (pad_nd.vec().size() > 0) {
      // 遍历每个维度，根据 pad_nd 中的值更新填充值
      for (int i = 0; i < dim; ++i) {
        // 更新前部、顶部、左侧的填充值
        padding_front_top_left[i] += pad_nd[2 * dim - 2 * i - 2]; // 4, 2, 0
        // 更新后部、底部、右侧的填充值
        padding_back_bottom_right[i] += pad_nd[2 * dim - 2 * i - 1]; // 5, 3, 1
      }
    }

    // 创建输出并传播内存格式
    if (!output_r.defined()) {
      // 计算卷积输出的尺寸
      auto dst_tz = conv_dst_size(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding_front_top_left,
          padding_back_bottom_right,
          params.stride,
          params.dilation);
      // 根据计算得到的尺寸创建空的输出张量
      output = at::empty(dst_tz, input.options(), mfmt);
    }

    // 调用 OneDNN 的卷积操作
    onednn::convolution(
        output,
        input,
        weight,
        bias,
        padding_front_top_left,
        padding_back_bottom_right,
        params.stride,
        params.dilation,
        params.groups,
        attr);
  }

  // 如果是三维数据，则调整输出的视图
  if (ndim == 3) {
    output = view3d(output);
  }

  // 如果输出结果已经定义且与当前输出不同，则复制到 output_r
  if (output_r.defined() && !output_r.is_same(output)) {
    output_r.copy_(output);
  } else {
    // 否则，将当前输出赋给 output_r
    output_r = output;
  }

  // 返回最终的输出结果 output_r
  return output_r;
}

// 执行卷积运算的核心函数，返回卷积结果的张量
Tensor _convolution(
    const Tensor& input_r,                // 输入张量
    const Tensor& weight_r,               // 卷积核张量
    const Tensor& bias_r,                 // 偏置张量
    IntArrayRef stride_,                  // 步幅
    IntArrayRef padding_,                 // 填充
    IntArrayRef dilation_,                // 膨胀
    bool transposed_,                     // 是否转置卷积
    IntArrayRef output_padding_,          // 输出填充
    int64_t groups_,                      // 分组数
    Attr attr) {                          // 附加属性
  Tensor output_r;                       // 定义输出张量
  return _convolution_out(
      output_r,                          // 输出张量
      input_r,                           // 输入张量
      weight_r,                          // 卷积核张量
      bias_r,                            // 偏置张量
      stride_,                           // 步幅
      padding_,                          // 填充
      dilation_,                         // 膨胀
      transposed_,                       // 是否转置卷积
      output_padding_,                   // 输出填充
      groups_,                           // 分组数
      attr);                             // 附加属性
}

// 可覆盖的卷积操作函数，返回卷积结果的张量
Tensor convolution_overrideable(
    const Tensor& input_r,                // 输入张量
    const Tensor& weight_r,               // 卷积核张量
    const std::optional<at::Tensor>& bias_r_opt,  // 可选的偏置张量
    IntArrayRef stride_,                  // 步幅
    IntArrayRef padding_,                 // 填充
    IntArrayRef dilation_,                // 膨胀
    bool transposed_,                     // 是否转置卷积
    IntArrayRef output_padding_,          // 输出填充
    int64_t groups_) {                    // 分组数
  c10::MaybeOwned<Tensor> bias_r_maybe_owned =
      at::borrow_from_optional_tensor(bias_r_opt);  // 可能拥有的偏置张量
  const Tensor& bias_r = *bias_r_maybe_owned;       // 实际的偏置张量

  auto k = weight_r.ndimension();         // 卷积核张量的维度
  at::MemoryFormat backend_memory_format = at::MemoryFormat::Contiguous;  // 后端内存格式，默认连续
  if (xpu_conv_use_channels_last(input_r, weight_r)) {
      backend_memory_format = (k == 5) ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;  // 若使用通道最后的格式，则设置对应的内存格式
  }
  Tensor input_c = input_r.contiguous(backend_memory_format);    // 将输入张量转换为指定的内存格式的连续张量
  Tensor weight_c = weight_r.contiguous(backend_memory_format);  // 将卷积核张量转换为指定的内存格式的连续张量

  return _convolution(
      input_c,                           // 输入张量
      weight_c,                          // 卷积核张量
      bias_r,                            // 偏置张量
      stride_,                           // 步幅
      padding_,                          // 填充
      dilation_,                         // 膨胀
      transposed_,                       // 是否转置卷积
      output_padding_,                   // 输出填充
      groups_,                           // 分组数
      Attr());                           // 空属性
}

// 可覆盖的卷积反向传播函数，返回梯度计算结果的张量元组
std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable(
    const Tensor& grad_output,            // 梯度输出张量
    const Tensor& input,                  // 输入张量
    const Tensor& weight,                 // 卷积核张量
    IntArrayRef stride,                   // 步幅
    IntArrayRef padding,                  // 填充
    IntArrayRef dilation,                 // 膨胀
    bool transposed,                      // 是否转置卷积
    IntArrayRef output_padding,           // 输出填充
    int64_t groups,                       // 分组数
    std::array<bool, 3> output_mask) {    // 输出掩码数组
  auto ndim = input.ndimension();         // 输入张量的维度
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,  // 断言，仅支持3D、4D、5D张量
      "convolution bwd only supports 3D, 4D, 5D tensor");
  TORCH_CHECK(
      grad_output.scalar_type() == ScalarType::Float ||    // 断言，仅支持浮点类型的梯度输出张量
          grad_output.scalar_type() == ScalarType::BFloat16 ||
          grad_output.scalar_type() == ScalarType::Double ||
          grad_output.scalar_type() == ScalarType::Half,
      "so far only support float, bfloat16, half and double convolution backward in XPU backend, your data type is ",
      grad_output.scalar_type());         // 打印不支持的数据类型信息

  bool is_channels_last_suggested = use_channels_last_for_conv(input, weight, transposed);  // 推荐是否使用通道最后的格式

  Tensor grad_output_, input_, weight_;   // 定义梯度输出、输入、卷积核张量
  IntArrayRef stride_, padding_, dilation_, output_padding_;  // 定义步幅、填充、膨胀、输出填充
  bool transposed_;                       // 是否转置卷积
  int64_t groups_;                        // 分组数
  ConvParams params;                      // 卷积参数对象
  if (3 == ndim) {                        // 若输入张量维度为3
    grad_output_ = view4d(grad_output);   // 将梯度输出张量视图化为4D
    input_ = view4d(input);               // 将输入张量视图化为4D
    weight_ = view4d(weight);             // 将卷积核张量视图化为4D
    params.stride = stride.vec();         // 设置卷积参数中的步幅
    params.padding = padding.vec();       // 设置卷积参数中的填充
    params.dilation = dilation.vec();     // 设置卷积参数中的膨胀
    params.transposed = transposed;       // 设置卷积参数中的转置标志
    params.output_padding = output_padding.vec();  // 设置卷积参数中的输出填充
    // 将参数组(groups)赋给成员变量
    params.groups = groups;
    // 将参数视为一维数据转换为二维数据
    params.view1d_as_2d();
    // 将参数的步幅赋给成员变量
    stride_ = params.stride;
    // 将参数的填充赋给成员变量
    padding_ = params.padding;
    // 将参数的扩张赋给成员变量
    dilation_ = params.dilation;
    // 将参数的转置标志赋给成员变量
    transposed_ = params.transposed;
    // 将参数的输出填充赋给成员变量
    output_padding_ = params.output_padding;
    // 将参数的组数赋给成员变量
    groups_ = params.groups;
  } else {
    // 将梯度输出(grad_output)赋给成员变量
    grad_output_ = grad_output;
    // 将输入(input)赋给成员变量
    input_ = input;
    // 将权重(weight)赋给成员变量
    weight_ = weight;
    // 将步幅(stride)赋给成员变量
    stride_ = stride;
    // 将填充(padding)赋给成员变量
    padding_ = padding;
    // 将扩张(dilation)赋给成员变量
    dilation_ = dilation;
    // 将转置标志(transposed)赋给成员变量
    transposed_ = transposed;
    // 将输出填充(output_padding)赋给成员变量
    output_padding_ = output_padding;
    // 将组数(groups)赋给成员变量
    groups_ = groups;
  }

  // 确保张量是连续的
  auto mfmt = is_channels_last_suggested ? get_cl_tag_by_ndim(input_.ndimension())
      : at::MemoryFormat::Contiguous;
  // 将梯度输出(grad_output)转换为连续的张量
  grad_output_ =  grad_output_.contiguous(mfmt);
  // 将权重(weight)转换为连续的张量
  weight_ = weight_.contiguous(mfmt);
  // 将输入(input)转换为连续的张量
  input_ = input_.contiguous(mfmt);

  // 获取梯度输出(grad_output)的选项
  auto opt = grad_output_.options();
  // 创建与输入(input)相同尺寸的空张量作为梯度输入(grad_input)
  Tensor grad_input = at::empty(input_.sizes(), opt, mfmt);
  // 创建与权重(weight)相同尺寸的空张量作为梯度权重(grad_weight)
  Tensor grad_weight = at::empty(weight_.sizes(), opt, mfmt);
  // 创建空张量作为梯度偏差(grad_bias)
  Tensor grad_bias;
  // 如果输出掩码中有第三个位置为真，则创建与梯度输出(grad_output)第二个维度大小相同的空张量作为梯度偏差(grad_bias)
  if (output_mask[2])
    grad_bias = at::empty({grad_output_.size(1)}, opt);

  // 如果输出掩码中第一个位置为真
  if (output_mask[0]) {
    // 如果输入(input)的元素数大于0
    if (input.numel() > 0) {
      // 如果是转置操作
      if (transposed_) {
        // 使用OneDNN执行反卷积数据的反向传播
        onednn::deconvolution_backward_data(
            grad_input,
            grad_output_,
            weight_,
            stride_,
            padding_,
            dilation_,
            groups_,
            output_mask[2]);
      } else {
        // 使用OneDNN执行卷积数据的反向传播
        onednn::convolution_backward_data(
            grad_input,
            grad_output_,
            weight_,
            padding_,
            padding_,
            stride_,
            dilation_,
            groups_,
            output_mask[2]);
      }
    }
  }
  // 如果输出掩码中第一个或第二个位置为真
  if (output_mask[1] || output_mask[2]) {
    // 如果输入(input)的元素数大于0
    if (input.numel() > 0) {
      // 如果是转置操作
      if (transposed_) {
        // 使用OneDNN执行反卷积权重的反向传播
        onednn::deconvolution_backward_weights(
            grad_weight,
            grad_bias,
            grad_output_,
            input_,
            stride_,
            padding_,
            dilation_,
            groups_);
      } else {
        // 使用OneDNN执行卷积权重的反向传播
        onednn::convolution_backward_weights(
            grad_weight,
            grad_bias,
            grad_output_,
            input_,
            weight_.sizes(),
            padding_,
            padding_,
            stride_,
            dilation_,
            groups_);
      }
    }
  }

  // 如果张量维度为3
  if (3 == ndim) {
    // 如果输出掩码中第一个位置为真
    if (output_mask[0])
      // 将梯度输入(grad_input)视为三维张量
      grad_input = view3d(grad_input);
    // 将梯度权重(grad_weight)视为三维张量
    grad_weight = view3d(grad_weight);
  }
  // 返回梯度输入(grad_input)、梯度权重(grad_weight)和梯度偏差(grad_bias)的元组
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

TORCH_LIBRARY_IMPL(aten, XPU, m){
  // 在模块 m 中注册 convolution_overrideable 函数的实现
  m.impl("convolution_overrideable", TORCH_FN(convolution_overrideable));
  // 在模块 m 中注册 convolution_backward_overrideable 函数的实现
  m.impl("convolution_backward_overrideable", TORCH_FN(convolution_backward_overrideable));
}

} // namespace xpu
} // namespace at::native
```