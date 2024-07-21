# `.\pytorch\aten\src\ATen\native\mkldnn\Utils.cpp`

```py
// 定义宏，仅在 Torch 断言期间启用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 MKLDNN 工具类头文件
#include <ATen/native/mkldnn/Utils.h>
// 包含原生池化操作头文件
#include <ATen/native/Pool.h>
// 包含 C10 工具库中的范围处理工具头文件
#include <c10/util/irange.h>

// 命名空间：at 原生命名空间
namespace at { namespace native {

// 函数：计算池化层输出尺寸
std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,         // 输入尺寸数组的引用
    IntArrayRef kernel_size,        // 卷积核尺寸数组的引用
    IntArrayRef stride,             // 步幅数组的引用
    IntArrayRef padding_l,          // 左填充数组的引用
    IntArrayRef padding_r,          // 右填充数组的引用
    IntArrayRef dilation,           // 膨胀率数组的引用
    bool ceil_mode) {               // 是否使用 ceil mode

  std::vector<int64_t> output_size(input_size.size());  // 输出尺寸数组
  // 复制 N 和 C
  output_size[0] = input_size[0];
  output_size[1] = input_size[1];

  // 遍历除了 N 和 C 外的其他维度
  for (const auto i : c10::irange(2, input_size.size())) {
    // 计算池化层输出形状及填充左右的尺寸
    output_size[i] = pooling_output_shape_pad_lr<int64_t>(
      input_size[i],
      kernel_size[i - 2],
      padding_l[i - 2],
      padding_r[i - 2],
      stride[i - 2],
      dilation[i - 2],
      ceil_mode
    );
  }

  return output_size;  // 返回计算得到的输出尺寸数组
}

// 函数：检查 MKLDNN 二进制融合的输入
void check_mkldnn_binary_fusion_inputs(
    const Tensor& input,    // 输入张量引用
    const Tensor& other,    // 其他张量引用
    const Tensor& weight,   // 权重张量引用
    const Tensor& bias) {   // 偏置张量引用

  // 如果权重张量不是 MKLDNN 张量
  if (!weight.is_mkldnn()) {
    // 断言：输入张量和权重张量的数据类型应相同
    TORCH_CHECK(
        input.options().type_equal(weight.options()),
        "Input type (",
        input.toString(),
        ") and weight type (",
        weight.toString(),
        ") should be the same");
  } else {
    // 断言：MKLDNN 点对点二进制融合的输入数据类型应相同
    TORCH_CHECK(
        input.scalar_type() == input.scalar_type(),
        "mkldnn pointwise binary: input dtype and weight dtype should be the same");
  }
  // 断言：输入张量和其他张量的数据类型应相同
  TORCH_CHECK(
      input.options().type_equal(other.options()),
      "Input type (",
      input.toString(),
      ") and other type (",
      other.toString(),
      ") should be the same");
  // 断言：如果定义了偏置张量，则输入张量和偏置张量的数据类型应相同
  TORCH_CHECK(
      !bias.defined() || (input.options().type_equal(bias.options())),
      "Input type (",
      input.toString(),
      ") and bias type (",
      bias.toString(),
      ") should be the same");
  // 断言：输入张量的设备类型应为 CPU
  TORCH_CHECK(
      input.device().is_cpu(),
      "mkldnn pointwise binary fusion: input's device should be CPU");
  // 断言：输入张量的数据类型应为 float、bfloat16 或 half
  TORCH_CHECK(
      input.scalar_type() == ScalarType::Float ||
          input.scalar_type() == ScalarType::BFloat16 ||
          input.scalar_type() == ScalarType::Half,
      "mkldnn pointwise binary: input's dtype should be float, bfloat16 or half");
  // 检查 MKLDNN 低精度处理
  mkldnn_check_low_precision(input.scalar_type(), "mkldnn pointwise binary");
}

// 如果 MKLDNN 已启用
#if AT_MKLDNN_ENABLED()

// 定义宏：属性函数定义
#define ATTR_FUNC(NAME)                              \
  [](torch::List<std::optional<at::Scalar>> scalars, \
     std::optional<c10::string_view> algorithm) {    \
    return ideep::attr_t::fuse_##NAME();             \
  }

// 属性函数：leaky_relu 融合属性函数
AttrFunction attr_func_leaky_relu =
    // 定义一个匿名函数，接受一个名为scalars的torch::List<std::optional<at::Scalar>>类型参数和一个名为algorithm的std::optional<c10::string_view>类型参数
    [](torch::List<std::optional<at::Scalar>> scalars,
       std::optional<c10::string_view> algorithm) {
      // 使用TORCH_CHECK宏确保scalars的长度为1，并且第一个元素是有效的标量
      TORCH_CHECK(
          scalars.size() == 1 &&
              scalars[0].get().toOptional<at::Scalar>().has_value(),
          "leaky_relu is expected to have one scalar input: negative_slope");
      // 获取第一个标量的值，并将其转换为float类型
      auto alpha_value =
          scalars[0].get().toOptional<at::Scalar>().value().to<float>();
      // 调用ideep::attr_t::fuse_relu方法，传递1.0和alpha_value作为参数，返回结果
      return ideep::attr_t::fuse_relu(1.0, alpha_value);
    };
AttrFunction attr_func_hardtanh =
    [](torch::List<std::optional<at::Scalar>> scalars,
       std::optional<c10::string_view> algorithm) {
      // 检查输入参数，确保有两个标量值且都有值
      TORCH_CHECK(
          scalars.size() == 2 &&
              scalars[0].get().toOptional<at::Scalar>().has_value() &&
              scalars[1].get().toOptional<at::Scalar>().has_value(),
          "hardtanh is expected to have two scalar input: min_val and max_val");

      // 获取下界和上界的数值
      auto lower_bound_value =
          scalars[0].get().toOptional<at::Scalar>().value().to<float>();
      auto upper_bound_value =
          scalars[1].get().toOptional<at::Scalar>().value().to<float>();
      // 返回融合的 clamp 属性
      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    };

AttrFunction attr_func_gelu = [](torch::List<std::optional<at::Scalar>> scalars,
                                 std::optional<c10::string_view> algorithm) {
  // 检查算法参数是否存在
  TORCH_CHECK(
      algorithm.has_value(),
      "gelu is expected to have one str input: algorithm");
  
  // 根据算法参数选择 GELU 类型
  dnnl::algorithm gelu_type;
  if (algorithm.value() == "none") {
    gelu_type = dnnl::algorithm::eltwise_gelu_erf;
  } else if (algorithm.value() == "tanh") {
    gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
  } else {
    // 如果算法不被支持，抛出错误
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported gelu algorithm: ", algorithm.value());
  }

  // 返回融合的 GELU 属性
  return ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type);
};

AttrFunction attr_func_hardsigmoid =
    [](torch::List<std::optional<at::Scalar>> scalars,
       std::optional<c10::string_view> algorithm) {
      // 创建属性对象和后操作对象
      ideep::attr_t attr;
      ideep::post_ops po;
      // 添加硬 sigmoid 操作到后操作对象
      po.append_eltwise(
          ideep::algorithm::eltwise_hardsigmoid, 1.0f / 6.0f, 0.5f);
      attr.set_post_ops(po);
      // 返回属性对象
      return attr;
    };

const std::map<c10::string_view, AttrFunction>& fusion_unary_attr_map() {
  // 定义并返回一元操作的属性映射表
  static const std::map<c10::string_view, AttrFunction> fusion_attr_map{
      {"relu", ATTR_FUNC(relu)},
      {"sigmoid", ATTR_FUNC(sigmoid)},
      {"tanh", ATTR_FUNC(tanh)},
      {"swish", ATTR_FUNC(swish)},
      {"hardswish", ATTR_FUNC(hardswish)},
      {"hardsigmoid", attr_func_hardsigmoid},
      {"leaky_relu", attr_func_leaky_relu},
      {"hardtanh", attr_func_hardtanh},
      {"gelu", attr_func_gelu},
  };
  return fusion_attr_map;
};

const std::map<c10::string_view, ideep::algorithm>& fusion_unary_alg_map() {
  // 定义并返回一元操作的算法映射表
  static const std::map<c10::string_view, ideep::algorithm> fusion_attr_map{
      {"relu", {ideep::algorithm::eltwise_relu}},
  };
  return fusion_attr_map;
};

const std::map<c10::string_view, ideep::algorithm>& fusion_binary_alg_map() {
  // 定义并返回二元操作的算法映射表
  static const std::map<c10::string_view, ideep::algorithm> fusion_attr_map{
      {"add", {ideep::algorithm::binary_add}},
      {"sub", {ideep::algorithm::binary_sub}},
      {"mul", {ideep::algorithm::binary_mul}},
      {"div", {ideep::algorithm::binary_div}},
  };
  return fusion_attr_map;
};

#endif // AT_MKLDNN_ENABLED()
```