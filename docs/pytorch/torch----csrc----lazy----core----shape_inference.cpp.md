# `.\pytorch\torch\csrc\lazy\core\shape_inference.cpp`

```
/**
 * This is a handwritten file that accompanies codegenerated header
 * LazyShapeDtype.h
 *
 * The purpose of these shape/dtype inference methods are to fill gaps
 * where we do not yet have structured kernels in pytorch core.  Ops
 * for which there _are_ structured kernels can use meta::op() to infer
 * shape/dtype, and codegen makes use of this.  Ops for which there are not
 * yet structured kernels can still be used with lazy_tensor codegen, but
 * require manual intervention to implement compute_shape_{op} and
 * compute_dtype_{op}.
 *
 * READ THIS!
 *
 * 1. Beware: Tech Debt!
 * ---------------------
 * These functions are tech debt.  We want to delete them all and use structured
 * kernels instead, but it's a lot faster to write these so we're decoupling the
 * two efforts to move fast for adding support for codegenned Lazy Tensor ops.
 *
 * Codegenned Lazy Tensor ops with handwritten shape formulae are still better
 * than fully handwritten Lazy Tensor ops (which also have handwritten shape
 * formulae).
 *
 * 2. Structured Kernels For The Win
 * ---------------------------------
 * Long term, more and more ops should be supported as 'structured kernels'.
 * Consider doing your part and porting an op.  As ops get ported over, the
 * codegen will automatically notice and stop generating declarations for these
 * shape formulae, so we'll need to manually clean up the unused functions in
 * this file, or somehow automate that.
 *
 * https://dev-discuss.pytorch.org/t/slides-from-structured-kernel-presentation/179
 *
 * 3. How to figure out the shape/dtype
 * ------------------------------------
 * Unfortunately there isn't a one-stop-shop for learning the output shape
 * formulae for all operators.  This is partly because some operators are not
 * part of our 'public' API, including backward operators which users don't
 * directly invoke.
 *
 * Check our opinfo registry:
 *  https://github.com/pytorch/pytorch/blob/13b859983183ea9938deb5030ac9a0747841f0a8/torch/csrc/jit/runtime/symbolic_shape_registry.cpp
 *
 * Read the manual (for ops that are 1:1 with python frontend):
 *  https://pytorch.org/docs/stable/generated/torch.trace.html
 *
 */

#include <torch/csrc/lazy/core/shape_inference.h>

#include <ATen/AccumulateType.h>
#include <ATen/CompositeExplicitAutogradFunctions.h>
#include <ATen/CompositeExplicitAutogradNonFunctionalFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorConversions.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/lazy/core/dynamic_ir.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/util.h>
#include <ostream>
#include <vector>

namespace torch {
namespace lazy {
// 从 ATen/native/utils/ParamUtils.h 复制而来，这里无法直接包含它？
static std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,         // 输入的整数数组引用
    const char* param_name,             // 参数名字符串指针
    int64_t expected_dim) {             // 预期的维度大小

  // 如果输入数组只有一个元素，将其扩展为指定维度大小的数组
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    // 如果输入数组大小不是预期的维度大小，抛出错误信息
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    // 否则返回输入数组的副本
    return list_param.vec();
  }
}

// 常见情况下不使用参数而不是使用它们，所以禁用未使用参数的警告
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

// 计算 arange 操作的输出形状的函数
TORCH_API std::vector<Shape> compute_shape_arange_out(
    const at::Scalar& start,    // 起始值的标量
    const at::Scalar& end,      // 终止值的标量
    const at::Scalar& step,     // 步长的标量
    // 计算从 start 到 end 的范围，并推断输出张量的形状
    at::Tensor& out) {
      double size_d = 0;
      // 从 RangeFactories.cpp 的 arange_out 函数中复制的形状推断代码
      // 注意：AT_DISPATCH_ALL_TYPES_AND 是一个宏，根据输出张量的数据类型定义正确的 scalar_t 类型
    
      AT_DISPATCH_ALL_TYPES_AND(
          c10::kBFloat16, out.scalar_type(), "compute_shape_arange_out", [&]() {
            // 注意：acc_type 根据 scalar_t 的类型和它是在 GPU 还是 CPU 上进一步定义累积类型。
            using accscalar_t = at::acc_type<scalar_t, false>;
            auto xstart = start.to<accscalar_t>();
            auto xend = end.to<accscalar_t>();
            auto xstep = step.to<accscalar_t>();
    
            // 我们使用双精度浮点数来计算 (start - end) / step，以确保跨设备的一致性。
            // 使用 accscalar_t 的问题在于，对于同一个 float32 的 scalar_t，在 GPU 上可能是 float32，
            // 而在 CPU 上可能是 double，这会导致精度问题，输出的大小因精度而异。
            // 我们要考虑的特殊情况是 int64_t，它比 double 具有更高的精度。
            if constexpr (std::is_same_v<scalar_t, int64_t>) {
              size_d = std::ceil(
                  static_cast<double>(
                      end.to<accscalar_t>() - start.to<accscalar_t>()) /
                  step.to<accscalar_t>());
            } else {
              size_d = std::ceil(
                  static_cast<double>(end.to<double>() - start.to<double>()) /
                  step.to<double>());
            }
    
            // 检查步长是否非零
            TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
            // 检查起始值和结束值是否有限
            TORCH_CHECK(
                std::isfinite(static_cast<double>(xstart)) &&
                    std::isfinite(static_cast<double>(xend)),
                "unsupported range: ",
                xstart,
                " -> ",
                xend);
            // 检查上界和下界是否与步长符号一致
            TORCH_CHECK(
                ((xstep > 0) && (xend >= xstart)) ||
                    ((xstep < 0) && (xend <= xstart)),
                "upper bound and larger bound inconsistent with step sign");
    
            // 检查大小是否有效，可能会溢出
            TORCH_CHECK(
                size_d >= 0 &&
                    size_d <=
                        static_cast<double>(std::numeric_limits<int64_t>::max()),
                "invalid size, possible overflow?");
          });
    
      // 将浮点数大小转换为整数大小
      int64_t size = static_cast<int64_t>(size_d);
    
      // 从 torch.arange 文档中：
      // dtype (torch.dtype, optional) – 返回张量的所需数据类型。
      // 默认情况下，如果为 None，则使用全局默认值 (参见 torch.set_default_dtype())。
      // 如果未指定 dtype，则从其他输入参数推断数据类型。
      // 如果 start、end 或 stop 中有任何浮点数，则推断的 dtype 将为默认 dtype，参见 get_default_dtype()。
      // 否则，推断的 dtype 将为 torch.int64。
      
      // 返回一个形状对象，其中包含输出张量的数据类型和大小信息
      return {Shape(out.scalar_type(), {size})};
    }
}

// 计算绝对形状的函数，根据输入张量的复数属性进行条件判断
std::vector<Shape> compute_shape_abs(const at::Tensor& self) {
  if (self.is_complex()) { // 检查张量是否为复数类型
    const auto float_type = c10::toRealValueType(self.scalar_type()); // 获取张量的实部数值类型
    return {Shape(float_type, self.sizes().vec())}; // 返回一个包含形状信息的向量
  }
  return {Shape(self.scalar_type(), self.sizes().vec())}; // 返回一个包含形状信息的向量
}

// 计算 Bernoulli 分布形状的函数，包含概率参数和随机数生成器的可选参数
std::vector<Shape> compute_shape_bernoulli(
    const at::Tensor& self,
    ::std::optional<at::Generator> generator) {
  return {Shape(self.scalar_type(), self.sizes().vec())}; // 返回一个包含形状信息的向量
}

// 重载的 Bernoulli 分布形状计算函数，不包含随机数生成器参数
std::vector<Shape> compute_shape_bernoulli(
    const at::Tensor& self,
    double p,
    ::std::optional<at::Generator> generator) {
  return compute_shape_bernoulli(self, generator); // 调用前面定义的计算 Bernoulli 分布形状的函数
}

// 计算二元交叉熵形状的函数，根据指定的减少（reduction）选项进行条件判断
std::vector<Shape> compute_shape_binary_cross_entropy(
    const at::Tensor& self,
    const at::Tensor& target,
    const ::std::optional<at::Tensor>& weight,
    int64_t reduction) {
  if (reduction == at::Reduction::None) { // 检查是否不进行减少操作
    return {Shape(self.scalar_type(), self.sizes().vec())}; // 返回一个包含形状信息的向量
  }
  return {Shape(self.scalar_type(), {})}; // 返回一个包含空形状信息的向量
}

// 计算二元交叉熵反向传播形状的函数
std::vector<Shape> compute_shape_binary_cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const ::std::optional<at::Tensor>& weight,
    int64_t reduction) {
  return {Shape(self.scalar_type(), self.sizes().vec())}; // 返回一个包含形状信息的向量
}

// 计算常数填充多维形状的函数
std::vector<Shape> compute_shape_constant_pad_nd(
    const at::Tensor& self,
    at::IntArrayRef pad,
    const at::Scalar& value) {
  // 基于具体的源码位置和函数定义进行常数填充的多维形状计算
  TORCH_CHECK(
      pad.size() % 2 == 0, // 检查填充大小是否为偶数
      "Length of pad must be even but instead it equals ",
      pad.size());

  auto input_sizes = self.sizes(); // 获取输入张量的大小
  auto l_inp = self.dim(); // 获取输入张量的维度数

  auto l_pad = pad.size() / 2; // 计算填充数量
  auto l_diff = l_inp - l_pad; // 计算差异数量
  TORCH_CHECK(
      l_inp >= (int64_t)l_pad, // 检查填充数量是否超出张量的维度
      "Length of pad should be no more than twice the number of "
      "dimensions of the input. Pad length is ",
      pad.size(),
      "while the input has ",
      l_inp,
      "dimensions.");

  std::vector<int64_t> new_shape; // 创建新的形状向量
  for (size_t i = 0; i < (size_t)l_diff; i++) { // 遍历差异数量
    new_shape.emplace_back(input_sizes[i]); // 添加每个维度的大小
  }

  for (const auto i : c10::irange((size_t)l_pad)) { // 遍历填充数量
    auto pad_idx = pad.size() - ((i + 1) * 2); // 计算填充索引
    auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1]; // 计算新的维度大小
    TORCH_CHECK(
        new_dim > 0, // 检查新的维度大小是否为正数
        "The input size ",
        input_sizes[l_diff + i],
        ", plus negative padding ",
        pad[pad_idx],
        " and ",
        pad[pad_idx + 1],
        " resulted in a negative output size, "
        "which is invalid. Check dimension ",
        l_diff + i,
        " of your input.");
    new_shape.emplace_back(new_dim); // 添加新的维度大小到形状向量中
  }
  return {Shape(self.scalar_type(), new_shape)}; // 返回一个包含形状信息的向量
}

// 计算卷积反向传播形状的函数
std::vector<Shape> compute_shape_convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::OptionalIntArrayRef bias_sizes,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    // 如果 bias_sizes 有值，则返回包含三个 Shape 对象的元组：
    // 1. 根据 input 的类型和大小创建的 Shape 对象
    // 2. 根据 weight 的类型和大小创建的 Shape 对象
    // 3. 根据 bias_sizes 的值创建的 Shape 对象
    if (bias_sizes.has_value()) {
        return {
            Shape(input.scalar_type(), input.sizes().vec()),
            Shape(weight.scalar_type(), weight.sizes().vec()),
            Shape(grad_output.scalar_type(), bias_sizes.value().vec())};
    } else {
        // 如果 bias_sizes 没有值，则返回包含两个 Shape 对象的元组：
        // 1. 根据 input 的类型和大小创建的 Shape 对象
        // 2. 根据 weight 的类型和大小创建的 Shape 对象
        // 这里存在一个待办事项，作者不确定是否应该返回两个形状，或者第三个形状是空的。
        return {
            Shape(input.scalar_type(), input.sizes().vec()),
            Shape(weight.scalar_type(), weight.sizes().vec())};
    }
}

std::vector<Shape> compute_shape_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const ::std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {
  // 计算权重张量的维度减去2，用于确定卷积的维度
  int64_t dim = weight.ndimension() - 2;
  // 检查权重张量的维度是否大于0，否则抛出错误信息
  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  // 如果不是转置卷积，则计算输出大小并返回
  auto expanded_stride = expand_param_if_needed(stride, "stride", dim);
  auto expanded_padding = expand_param_if_needed(padding, "padding", dim);
  auto expanded_dilation = expand_param_if_needed(dilation, "dilation", dim);
  if (!transposed) {
    return {Shape(
        input.scalar_type(),
        at::native::conv_output_size(
            input.sizes(),
            weight.sizes(),
            expanded_padding,
            expanded_stride,
            expanded_dilation))};
  } else {
    // 如果是转置卷积，则计算输入大小并返回
    auto expanded_output_padding =
        expand_param_if_needed(output_padding, "output_padding", dim);
    auto out_shape = at::native::conv_input_size(
        input.sizes(),
        weight.sizes(),
        expanded_padding,
        expanded_output_padding,
        expanded_stride,
        expanded_dilation,
        groups);
    return {Shape(input.scalar_type(), out_shape)};
  }
}

std::vector<Shape> compute_shape_masked_fill(
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  // 返回self张量的形状作为Shape对象
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_masked_fill(
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& value) {
  // 返回self张量的形状作为Shape对象
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_max(const at::Tensor& self) {
  // 检查self张量的元素数量是否大于0，否则抛出错误信息
  TORCH_CHECK(
      self.numel() > 0,
      "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  // 返回空维度的Shape对象
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_min(const at::Tensor& self) {
  // 检查self张量的元素数量是否大于0，否则抛出错误信息
  TORCH_CHECK(
      self.numel() > 0,
      "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  // 返回空维度的Shape对象
  return {Shape(self.scalar_type(), {})};
}

static std::vector<Shape> compute_shape_nonzero(
    const at::Tensor& t,
    bool as_tuple) {
  // 如果as_tuple为true，返回t张量每个维度的长度作为Shape对象的向量
  if (as_tuple) {
    auto res = std::vector<Shape>();
    for (auto dim_size : t.sizes()) {
      res.emplace_back(Shape(at::kLong, {dim_size}));
    }
    return res;
  }
  // 否则计算t张量的总元素数，并返回空Shape对象
  int64_t max_elements = 1;
  for (auto dim_size : t.sizes()) {
    max_elements *= dim_size;

# 将 max_elements 乘以 dim_size 的值，更新 max_elements 的结果


  }

# 结束 for 循环块


  return {Shape(at::kLong, {max_elements, (int64_t)t.sizes().size()})};

# 返回一个 Shape 对象，使用 at::kLong 类型，包含两个维度：
# - 第一个维度为 max_elements，代表元素的最大数量
# - 第二个维度为 t.sizes().size() 的值，即张量 t 的维度数量的长整型表示
It looks like the format got a bit off. Let me correct that for you:


"""
将 max_elements 乘以 dim_size 的值，更新 max_elements 的结果
"""
max_elements *= dim_size;

"""
结束 for 循环块
"""
}

"""
返回一个 Shape 对象，使用 at::kLong 类型，包含两个维度：
- 第一个维度为 max_elements，代表元素的最大数量
- 第二个维度为 t.sizes().size() 的值，即张量 t 的维度数量的长整型表示
"""
return {Shape(at::kLong, {max_elements, (int64_t)t.sizes().size()})};
}

// 计算非零元素的形状，返回形状向量
std::vector<Shape> compute_shape_nonzero(const at::Tensor& self) {
  // 调用带有默认参数的 compute_shape_nonzero 函数
  return compute_shape_nonzero(self, false);
}

// 计算嵌入层的形状
std::vector<Shape> compute_shape_embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  // 基于 aten/src/ATen/native/Embedding.cpp::embedding 实现
  std::vector<int64_t> out_sizes = indices.sizes().vec();
  out_sizes.emplace_back(weight.size(1));
  // 返回一个 Shape 对象的列表
  return {Shape(weight.scalar_type(), out_sizes)};
}

// 计算标准差的形状
std::vector<Shape> compute_shape_std(const at::Tensor& self, bool unbiased) {
  // 调用带有 ::std::nullopt 参数的 compute_shape_std 函数
  return compute_shape_std(self, ::std::nullopt, ::std::nullopt, false);
}

// 计算标准差的形状
std::vector<Shape> compute_shape_std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  // 调用带有 ::std::nullopt 和 keepdim 参数的 compute_shape_std 函数
  return compute_shape_std(self, dim, ::std::nullopt, keepdim);
}

// 计算标准差的形状
std::vector<Shape> compute_shape_std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const ::std::optional<at::Scalar>& correction,
    bool keepdim) {
  if (dim.has_value()) {
    // 使用 make_dim_mask 从 self 中创建形状，并根据 keepdim 保持维度
    auto shape = at::native::shape_from_dim_mask(
        self, at::native::make_dim_mask(dim.value(), self.dim()), keepdim);
    // 返回一个 Shape 对象的列表，表示 self 的形状
    return {Shape(
        self.scalar_type(), std::vector<int64_t>(shape.begin(), shape.end()))};
  }
  // 返回一个空的 Shape 对象的列表，表示 self 的标量类型形状
  return {Shape(self.scalar_type(), {})};
}

// 计算嵌入层稠密反向传播的形状
std::vector<Shape> compute_shape_embedding_dense_backward(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  // 基于 aten/src/ATen/native/Embedding.cpp::embedding_dense_backward_cpu 实现
  // 返回一个 Shape 对象的列表，表示 grad_output 的形状
  return {
      Shape(grad_output.scalar_type(), {num_weights, grad_output.size(-1)})};
}

// 计算扩展后的形状
std::vector<Shape> compute_shape_expand(
    const at::Tensor& self,
    at::IntArrayRef size,
    bool implicit) {
  // 检查 size 的维度是否大于等于 self 的维度
  TORCH_CHECK_GE(static_cast<int64_t>(size.size()), self.dim());
  size_t num_new_dimensions = size.size() - self.dim();
  std::vector<int64_t> padded_self(num_new_dimensions, 0);
  padded_self.insert(
      padded_self.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(size.size());
  // 根据 size 中的值扩展 self 的形状
  for (const auto idx : c10::irange(size.size())) {
    target_size[idx] = size[idx] == -1 ? padded_self[idx] : size[idx];
  }
  // 返回一个 Shape 对象的列表，表示 self 的目标形状
  return {Shape(self.scalar_type(), target_size)};
}

// 计算扩展后的形状
std::vector<Shape> compute_shape_expand(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    bool implicit) {
  // 检查 size 的维度是否大于等于 self 的维度
  TORCH_CHECK_GE(static_cast<int64_t>(size.size()), self.dim());
  std::vector<c10::SymInt> _sizes = ToVector<c10::SymInt>(size);
  size_t num_new_dimensions = _sizes.size() - self.dim();
  std::vector<int64_t> padded_self(num_new_dimensions, 0);
  padded_self.insert(
      padded_self.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(_sizes.size());
  // 根据 size 中的值扩展 self 的形状
  for (const auto idx : c10::irange(_sizes.size())) {
    // 如果 size[idx] 为 -1，则使用 padded_self[idx]，否则使用 size[idx]
    target_size[idx] = _sizes[idx] == c10::SymInt::negative_one() ? padded_self[idx] : _sizes[idx].value();
  }
  // 返回一个 Shape 对象的列表，表示 self 的目标形状
  return {Shape(self.scalar_type(), target_size)};
}
    // 如果_sizes[idx]可以转换为整数，将其赋值给ma
    if (auto ma = _sizes[idx].maybe_as_int()) {
      // 将ma的值赋给target_size[idx]
      target_size[idx] = *ma;
      // 如果ma为-1，要求对于不存在的维度不能指定为-1
      if (*ma == -1) {
        // 检查索引idx是否大于等于num_new_dimensions
        TORCH_CHECK(idx >= num_new_dimensions);
        // 将padded_self[idx]的值赋给target_size[idx]
        target_size[idx] = padded_self[idx];
      } else {
        // 否则将ma的值再次赋给target_size[idx]
        target_size[idx] = *ma;
      }
    } else {
      // 如果_sizes[idx]无法转换为整数，假设其为torch::lazy::SymNodeImpl*
      auto* lazySymNode = dynamic_cast<torch::lazy::SymNodeImpl*>(
          _sizes[idx].toSymNodeImplUnowned());
      // 内部断言lazySymNode不为空
      TORCH_INTERNAL_ASSERT(lazySymNode);
      // 获取lazySymNode的节点并赋给size_node
      auto size_node = lazySymNode->node_;
      // 获取尺寸节点的静态值
      auto static_value =
          std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node)
              ->getStaticValue();
      // 将静态值赋给target_size[idx]
      target_size[idx] = static_value;
    }
  }
  // 返回Shape对象，包含self的标量类型和目标尺寸
  return {Shape(self.scalar_type(), target_size)};
}

std::vector<Shape> compute_shape_index_select(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  // 根据 pytorch 中 torch.index_select 函数的定义，将 Rank 0 的索引张量提升为 1 * 1 的张量。
  dim = at::maybe_wrap_dim(dim, self);
  // 确定索引张量的维度，如果大于 0，则取其维度；否则为 1。
  auto index_dim = index.dim() > 0 ? index.dim() : 1;
  // 确定索引张量的大小，如果大于 0，则取其大小；否则为 1。
  auto index_size = index.dim() > 0 ? index.size(0) : 1;
  // 检查索引张量是否为 1 维。
  TORCH_CHECK(index_dim == 1);

  // 获取自身张量的大小。
  auto self_sizes = self.sizes();
  // 复制自身张量的大小到输出大小的向量中。
  std::vector<int64_t> output_sizes(self_sizes.begin(), self_sizes.end());
  // 检查输出大小向量是否为空。
  TORCH_CHECK(!output_sizes.empty(), "Empty output_sizes is not supported.");
  // 更新输出大小向量中指定维度的大小为索引张量的大小。
  output_sizes[dim] = index_size;

  // 返回形状信息的向量，包含指定的标量类型和更新后的输出大小向量。
  return {Shape(self.scalar_type(), output_sizes)};
}

std::vector<Shape> compute_shape_inverse(const at::Tensor& self) {
  // 返回包含自身张量的标量类型和大小向量的形状信息向量。
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_isnan(const at::Tensor& self) {
  // 返回包含布尔类型标量的形状信息向量，大小向量与自身张量的大小向量相同。
  return {Shape(c10::ScalarType::Bool, self.sizes().vec())};
}

std::vector<Shape> compute_shape_cat(at::TensorList tensors, int64_t dim) {
  // TODO(whc) support cat in codegen and move this to compute_*_cat functions
  // 初始化输出形状的大小向量为第一个张量的大小。
  std::vector<int64_t> out_shape(
      tensors[0].sizes().begin(), tensors[0].sizes().end());

  // 确定维度参数的有效性，用于张量的拼接操作。
  dim = at::maybe_wrap_dim(dim, tensors);
  // 计算在指定维度上扩展后的形状。
  size_t extended_dim_shape = 0;
  for (auto& tensor : tensors) {
    extended_dim_shape += tensor.sizes()[dim];
  }
  // 检查输出形状向量是否为空，不支持标量张量的拼接。
  TORCH_CHECK(!out_shape.empty(), "Scalar tensors are not supported in cat.");
  // 检查扩展后的维度形状是否超出整型限制。
  TORCH_CHECK(
      extended_dim_shape <=
          static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "Size overflow");
  // 更新输出形状向量中指定维度的大小。
  out_shape[dim] = extended_dim_shape;

  // 返回形状信息的向量，包含第一个张量的标量类型和更新后的输出形状向量。
  return {Shape(tensors[0].scalar_type(), out_shape)};
}

TORCH_API std::vector<torch::lazy::Shape> compute_shape_cholesky(
    const at::Tensor& self,
    bool upper) {
  // 返回包含自身张量的标量类型和大小向量的形状信息向量。
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_native_batch_norm(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const ::std::optional<at::Tensor>& running_mean,
    const ::std::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {
  // 初始化形状信息向量，预留空间以存储三个元素。
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  // 将输入张量的标量类型和大小向量添加到形状信息向量中。
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  // 检查输入张量的维度是否至少为 2，必须包含批处理和通道维度。
  TORCH_CHECK(
      input.sizes().size() >= 2,
      "Input tensor must have at least batch and channel dimensions!");
  // 计算输入张量的特征数。
  int64_t num_features = input.size(1);

  // 如果存在运行时均值，则将其标量类型和大小向量添加到形状信息向量中；否则创建默认类型和大小向量。
  if (running_mean.has_value()) {
    shapes.emplace_back(
        running_mean.value().scalar_type(), running_mean.value().sizes().vec());
  } else {
    shapes.emplace_back(
        at::get_default_dtype_as_scalartype(),
        std::vector<int64_t>{num_features});
  }

  // 省略部分代码，用于计算运行时方差的形状信息。
    # 将形状信息添加到容器中
    shapes.emplace_back(
        running_var.value().scalar_type(), running_var.value().sizes().vec());
    # 如果条件为假，则执行以下操作
    else:
        # 使用默认的数据类型作为标量类型，并创建包含 num_features 的整数向量
        shapes.emplace_back(
            at::get_default_dtype_as_scalartype(),
            std::vector<int64_t>{num_features});
    # 返回形状容器
    return shapes;
}

std::vector<torch::lazy::Shape> compute_shape_native_batch_norm_backward(
    const at::Tensor& grad_out,  // 输入：梯度张量 grad_out
    const at::Tensor& input,     // 输入：输入张量 input
    const ::std::optional<at::Tensor>& weight,  // 输入（可选）：权重张量 weight
    const ::std::optional<at::Tensor>& running_mean,  // 输入（可选）：运行均值张量 running_mean
    const ::std::optional<at::Tensor>& running_var,   // 输入（可选）：运行方差张量 running_var
    const ::std::optional<at::Tensor>& save_mean,     // 输入（可选）：保存的均值张量 save_mean
    const ::std::optional<at::Tensor>& save_invstd,   // 输入（可选）：保存的标准差的倒数张量 save_invstd
    bool train,        // 输入：训练标志 train
    double eps,        // 输入：epsilon 参数 eps
    ::std::array<bool, 3> output_mask) {  // 输入：输出掩码数组 output_mask
  std::vector<torch::lazy::Shape> shapes;  // 创建一个空的 lazy::Shape 向量 shapes
  shapes.reserve(3);  // 预留三个元素的空间

  // 将输入的形状作为第一个 Shape 对象存入向量
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  // 检查输入张量的维度是否至少包含批次和通道维度
  TORCH_CHECK(
      input.sizes().size() >= 2,
      "Input tensor must have at least batch and channel dimensions!");
  int64_t num_features = input.size(1);  // 获取通道数

  // 将权重和偏置的形状作为后续两个 Shape 对象存入向量，长度为通道数 C
  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{num_features});
  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{num_features});

  return shapes;  // 返回包含形状信息的向量
}

std::vector<Shape> compute_shape_native_layer_norm(
    const at::Tensor& input,       // 输入：输入张量 input
    at::IntArrayRef normalized_shape,  // 输入：规范化形状数组 normalized_shape
    const ::std::optional<at::Tensor>& weight,  // 输入（可选）：权重张量 weight
    const ::std::optional<at::Tensor>& bias,    // 输入（可选）：偏置张量 bias
    double eps) {   // 输入：epsilon 参数 eps
  // 从 aten/src/ATen/native/layer_norm.cpp::layer_norm_cpu_out 复制的实现

  auto input_shape = input.sizes().vec();  // 获取输入张量的形状
  const size_t axis = input.dim() - normalized_shape.size();  // 计算规范化的轴数

  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(axis)) {
    TORCH_CHECK(idx < input_shape.size(), "Shape mismatch");  // 检查形状匹配
    stat_shape.emplace_back(input_shape[idx]);  // 将维度添加到统计形状中
  }
  for (const auto idx : c10::irange(axis, input.dim())) {
    (void)idx; // 抑制未使用变量警告
    stat_shape.emplace_back(1);  // 将大小为 1 的维度添加到统计形状中
  }

  // 返回包含三个 Shape 对象的向量
  return {
      Shape(input.scalar_type(), input_shape),    // 输入张量的形状
      Shape(input.scalar_type(), stat_shape),     // 统计形状
      Shape(input.scalar_type(), stat_shape)};    // 统计形状
}

std::vector<Shape> compute_shape_native_layer_norm_backward(
    const at::Tensor& grad_out,       // 输入：梯度张量 grad_out
    const at::Tensor& input,          // 输入：输入张量 input
    at::IntArrayRef normalized_shape, // 输入：规范化形状数组 normalized_shape
    const at::Tensor& mean,           // 输入：均值张量 mean
    const at::Tensor& rstd,           // 输入：倒数标准差张量 rstd
    const ::std::optional<at::Tensor>& weight,  // 输入（可选）：权重张量 weight
    const ::std::optional<at::Tensor>& bias,    // 输入（可选）：偏置张量 bias
    ::std::array<bool, 3> output_mask) {   // 输入：输出掩码数组 output_mask
  std::vector<Shape> shapes;  // 创建一个空的 Shape 向量 shapes

  // 根据输出掩码选择是否包含对应的形状信息
  shapes.emplace_back(
      input.scalar_type(),
      output_mask[0] ? input.sizes().vec() : std::vector<int64_t>{});
  shapes.emplace_back(
      weight && weight->defined() ? weight->scalar_type() : input.scalar_type(),
      output_mask[1] && weight ? weight->sizes().vec()
                               : std::vector<int64_t>{});
  shapes.emplace_back(
      bias && bias->defined() ? bias->scalar_type() : input.scalar_type(),
      output_mask[2] && bias ? bias->sizes().vec() : std::vector<int64_t>{});

  return shapes;  // 返回包含形状信息的向量
}

std::vector<Shape> compute_shape_mean(
    # 如果提供了 dtype 参数值
    if (dtype.has_value()) {
        # 返回一个 Shape 对象列表，使用提供的 dtype 值和空维度
        return {Shape(dtype.value(), {})};
    }
    # 如果未提供 dtype 参数值
    return {Shape(self.scalar_type(), {})};
}

// 计算带有新空的步幅的形状
std::vector<Shape> compute_shape_new_empty_strided(
    const at::Tensor& self, // 输入张量
    at::IntArrayRef size, // 尺寸
    at::IntArrayRef stride, // 步幅
    ::std::optional<at::ScalarType> dtype, // 数据类型（可选）
    ::std::optional<at::Layout> layout, // 布局（可选）
    ::std::optional<at::Device> device, // 设备（可选）
    ::std::optional<bool> pin_memory) { // 是否固定内存（可选）
  return {Shape(dtype.has_value() ? *dtype : self.scalar_type(), size.vec())};
}

// 计算形状（适用于张量和向量）
std::vector<Shape> compute_shape_mv(
    const at::Tensor& self, // 输入张量
    const at::Tensor& vec) { // 输入向量
  return {Shape(self.scalar_type(), {self.size(0)})};
}

// 计算本地丢弃形状（输入张量、丢弃率、训练标志）
std::vector<Shape> compute_shape_native_dropout(
    const at::Tensor& input, // 输入张量
    double p, // 丢弃率
    ::std::optional<bool> train) { // 训练标志（可选）
  return {
      Shape(input.scalar_type(), input.sizes().vec()), // 输入张量的形状
      Shape(c10::ScalarType::Bool, input.sizes().vec())}; // 布尔类型形状
}

// 计算本地丢弃反向形状（梯度输出、掩码、尺度）
std::vector<Shape> compute_shape_native_dropout_backward(
    const at::Tensor& grad_output, // 梯度输出张量
    const at::Tensor& mask, // 掩码张量
    double scale) { // 尺度
  return {Shape(grad_output.scalar_type(), grad_output.sizes().vec())};
}

// 计算随机形状（输入张量、生成器）
std::vector<Shape> compute_shape_random(
    const at::Tensor& self, // 输入张量
    ::std::optional<at::Generator> generator) { // 生成器（可选）
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算随机形状（输入张量、上限、生成器）
std::vector<Shape> compute_shape_random(
    const at::Tensor& self, // 输入张量
    int64_t to, // 上限
    ::std::optional<at::Generator> generator) { // 生成器（可选）
  return compute_shape_random(self, generator);
}

// 计算随机形状（输入张量、下限、上限、生成器）
std::vector<Shape> compute_shape_random(
    const at::Tensor& self, // 输入张量
    int64_t from, // 下限
    ::std::optional<int64_t> to, // 上限（可选）
    ::std::optional<at::Generator> generator) { // 生成器（可选）
  return compute_shape_random(self, generator);
}

// 计算 ReLU 形状（输入张量）
std::vector<Shape> compute_shape_relu(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算求和形状（输入张量、数据类型）
std::vector<Shape> compute_shape_sum(
    const at::Tensor& self, // 输入张量
    ::std::optional<at::ScalarType> dtype) { // 数据类型（可选）
  if (dtype.has_value()) {
    return {Shape(dtype.value(), {})};
  }
  // 对于所有整数类型（包括布尔类型），torch::sum 默认将其提升为 int64_t
  if (isIntegralType(self.scalar_type(), /*includeBool*/ true)) {
    return {Shape(c10::ScalarType::Long, {})};
  }
  return {Shape(self.scalar_type(), {})};
  ;
}

// 计算零形状（输入张量）
std::vector<Shape> compute_shape_zero(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算采取形状（输入张量、索引张量）
TORCH_API std::vector<torch::lazy::Shape> compute_shape_take(
    const at::Tensor& self, // 输入张量
    const at::Tensor& index) { // 索引张量
  return {Shape(self.scalar_type(), index.sizes().vec())};
}

// 计算迹形状（输入张量）
std::vector<Shape> compute_shape_trace(const at::Tensor& self) {
  return {Shape(self.scalar_type(), {})};
}

// 计算排序形状（输入张量、维度、降序标志）
std::vector<Shape> compute_shape_sort(
    const at::Tensor& self, // 输入张量
    int64_t dim, // 维度
    bool descending) { // 是否降序
  return {
      Shape(self.scalar_type(), self.sizes().vec()), // 输入张量的形状
      Shape(c10::ScalarType::Long, self.sizes().vec())}; // 长整型形状
}
// 计算形状为 {*, n, n} 的输入张量的形状，返回形状为 * 的向量
std::vector<Shape> compute_shape_slogdet(const at::Tensor& self) {
  // 断言输入张量的维度至少为 2
  TORCH_INTERNAL_ASSERT(self.dim() >= 2);
  // 根据输入张量的大小，创建一个向量来存储输出形状的大小，省略最后两个维度
  std::vector<int64_t> out_sizes(self.sizes().begin(), self.sizes().end() - 2);
  // 返回两个相同形状和数据类型的 Shape 对象作为输出
  return {
      Shape(self.scalar_type(), out_sizes),
      Shape(self.scalar_type(), out_sizes)};
}

// 计算逻辑 AND 操作的输出形状
std::vector<torch::lazy::Shape> compute_shape_logical_and(
    const at::Tensor& self,
    const at::Tensor& other) {
  // 断言两个张量的大小可以扩展
  TORCH_INTERNAL_ASSERT(at::are_expandable(self.sizes(), other.sizes()));
  // 返回一个具有推断大小的布尔类型的 Shape 对象
  return {Shape(
      c10::ScalarType::Bool, at::infer_size(self.sizes(), other.sizes()))};
}

// 计算逻辑 NOT 操作的输出形状
std::vector<torch::lazy::Shape> compute_shape_logical_not(
    const at::Tensor& self) {
  // 返回一个具有与输入张量相同大小的布尔类型的 Shape 对象
  return {Shape(c10::ScalarType::Bool, self.sizes().vec())};
}

// 计算逻辑 OR 操作的输出形状
std::vector<torch::lazy::Shape> compute_shape_logical_or(
    const at::Tensor& self,
    const at::Tensor& other) {
  // 断言两个张量的大小可以扩展
  TORCH_INTERNAL_ASSERT(at::are_expandable(self.sizes(), other.sizes()));
  // 返回一个具有推断大小的布尔类型的 Shape 对象
  return {Shape(
      c10::ScalarType::Bool, at::infer_size(self.sizes(), other.sizes()))};
}

// 计算逻辑 XOR 操作的输出形状
std::vector<torch::lazy::Shape> compute_shape_logical_xor(
    const at::Tensor& self,
    const at::Tensor& other) {
  // 断言两个张量的大小可以扩展
  TORCH_INTERNAL_ASSERT(at::are_expandable(self.sizes(), other.sizes()));
  // 返回一个具有推断大小的布尔类型的 Shape 对象
  return {Shape(
      c10::ScalarType::Bool, at::infer_size(self.sizes(), other.sizes()))};
}

// 计算 Smooth L1 损失函数的反向传播的输出形状
std::vector<Shape> compute_shape_smooth_l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  // 这个函数的输入 grad_output 实际上是前向传播的输入，输出的形状应该与前向传播的输入匹配
  // 返回一个具有与输入张量相同大小的 Shape 对象
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算对数行列式的输出形状
std::vector<Shape> compute_shape_logdet(const at::Tensor& self) {
  // 断言输入张量的维度至少为 2
  TORCH_INTERNAL_ASSERT(self.dim() >= 2);
  // 根据输入张量的大小，创建一个向量来存储输出形状的大小，省略最后两个维度
  // 不检查输入数据类型，但输出数据类型要么与输入匹配，要么实际的对数行列式操作会在不支持的类型上抛出异常
  return {Shape(self.scalar_type(), self.sizes().begin(), self.sizes().end() - 2)};
}

// 计算对数 sigmoid 函数前向传播的输出形状
std::vector<Shape> compute_shape_log_sigmoid_forward(const at::Tensor& self) {
  // 基于 log_sigmoid_forward_out_cpu 函数的定义
  // 返回两个具有与输入张量相同大小的 Shape 对象作为输出
  return {
      Shape(self.scalar_type(), self.sizes().vec()),
      Shape(self.scalar_type(), self.sizes().vec())};
}
    // 基于以下函数的定义：
    // aten/src/ATen/native/Activation.cpp::log_sigmoid_backward_cpu*.
    // 返回一个由指定形状和数据类型构成的张量
    return {Shape(grad_output.scalar_type(), grad_output.sizes().vec())};
}
// 定义函数 compute_shape_nll_loss2d_forward，计算二维 NLL 损失的前向传播
std::vector<Shape> compute_shape_nll_loss2d_forward(
    const at::Tensor& self,  // 输入张量 self
    const at::Tensor& target,  // 目标张量 target
    const ::std::optional<at::Tensor>& weight,  // 权重张量（可选）
    int64_t reduction,  // 减少方式参数
    int64_t ignore_index) {  // 忽略的索引值
  // 基于 aten/src/ATen/native/LossNLL2d.cpp:nll_loss2d_forward_cpu 的定义
  auto sizes =
      (reduction == at::Reduction::Reduction::None ? target.sizes().vec()  // 根据减少方式选择大小向量
                                                   : std::vector<int64_t>{});  // 否则为空向量
  return {Shape(self.scalar_type(), sizes), Shape(self.scalar_type(), {})};  // 返回形状向量
}

// 定义函数 compute_shape_nll_loss2d_backward，计算二维 NLL 损失的反向传播
std::vector<Shape> compute_shape_nll_loss2d_backward(
    const at::Tensor& grad_output,  // 梯度输出张量
    const at::Tensor& self,  // 输入张量 self
    const at::Tensor& target,  // 目标张量 target
    const ::std::optional<at::Tensor>& weight,  // 权重张量（可选）
    int64_t reduction,  // 减少方式参数
    int64_t ignore_index,  // 忽略的索引值
    const at::Tensor& total_weight) {  // 总权重张量
  return {Shape(self.scalar_type(), self.sizes().vec())};  // 返回形状向量
}

// 定义函数 compute_shape_grid_sampler_2d，计算二维网格采样器的形状
std::vector<Shape> compute_shape_grid_sampler_2d(
    const at::Tensor& input,  // 输入张量
    const at::Tensor& grid,  // 网格张量
    int64_t interpolation_mode,  // 插值模式
    int64_t padding_mode,  // 填充模式
    bool align_corners) {  // 是否对齐角点
  // 来自 `aten/src/ATen/native/cpu/GridSamplerKernel.cpp`
  int64_t N = input.size(0);  // 批次大小
  int64_t C = input.size(1);  // 通道数
  int64_t H = grid.size(1);  // 网格高度
  int64_t W = grid.size(2);  // 网格宽度
  return {Shape(input.scalar_type(), {N, C, H, W})};  // 返回形状向量
}

// 定义函数 compute_shape_grid_sampler_2d_backward，计算二维网格采样器的反向传播形状
std::vector<Shape> compute_shape_grid_sampler_2d_backward(
    const at::Tensor& grad_output,  // 梯度输出张量
    const at::Tensor& input,  // 输入张量
    const at::Tensor& grid,  // 网格张量
    int64_t interpolation_mode,  // 插值模式
    int64_t padding_mode,  // 填充模式
    bool align_corners,  // 是否对齐角点
    ::std::array<bool, 2> output_mask) {  // 输出掩码数组
  // 来自 `aten/src/ATen/native/cpu/GridSamplerKernel.cpp`
  auto grad_input_shape = Shape(input.scalar_type(), input.sizes().vec());  // 梯度输入形状
  auto grad_grid_shape = Shape(grid.scalar_type(), grid.sizes().vec());  // 梯度网格形状
  return {grad_input_shape, grad_grid_shape};  // 返回形状向量
}

// 定义函数 compute_shape_flip，计算翻转操作的形状
std::vector<Shape> compute_shape_flip(
    const at::Tensor& self,  // 输入张量 self
    at::IntArrayRef dims) {  // 维度列表
  return {Shape(self.scalar_type(), self.sizes().vec())};  // 返回形状向量
}

// 定义函数 compute_shape__adaptive_avg_pool2d，计算自适应二维平均池化的形状
std::vector<Shape> compute_shape__adaptive_avg_pool2d(
    const at::Tensor& self,  // 输入张量 self
    at::IntArrayRef output_size) {  // 输出尺寸列表
  // 基于 `aten/src/ATen/native/AdaptiveAveragePooling.cpp` 和 `aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp` 的检查
  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");  // 检查输出尺寸长度必须为 2
  TORCH_CHECK(
      (output_size[0] >= 0 && output_size[1] >= 0),  // 检查输出尺寸元素必须大于等于 0
      "adaptive_avg_pool2d: elements of output_size must be greater than or equal to 0 ",
      "but received {",
      output_size[0],
      ", ",
      output_size[1],
      "}");  // 输出尺寸元素错误时的错误消息
  int64_t ndim = self.ndimension();  // 输入张量的维度数
  for (const auto i : c10::irange(1, ndim)) {  // 迭代输入张量的维度（从 1 到 ndim-1）
    // 检查在自适应平均池化操作中，第 i 维度的大小是否大于零，否则抛出错误信息
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_avg_pool2d(): Expected self to have non-zero size for non-batch dimensions, "
        "but Tensor has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
    
    // 检查张量的维度数是否为3或4，否则抛出错误信息
    TORCH_CHECK(
        (ndim == 3 || ndim == 4),
        "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
        self.sizes());
    
    // 获取通道数，即在倒数第三个维度的大小
    int64_t channels = self.size(-3);
    
    // 获取输出的高度和宽度
    int64_t output_height = output_size[0];
    int64_t output_width = output_size[1];
    
    // 根据张量的维度数返回相应的形状
    if (ndim == 3) {
        // 对于3维张量，返回形状为 [通道数, 输出高度, 输出宽度]
        return {Shape(self.scalar_type(), {channels, output_height, output_width})};
    } else {
        // 对于4维张量，获取批量大小，返回形状为 [批量大小, 通道数, 输出高度, 输出宽度]
        int64_t nbatch = self.size(0);
        return {Shape(
            self.scalar_type(), {nbatch, channels, output_height, output_width})};
    }
// 计算自适应平均池化反向传播的形状
std::vector<Shape> compute_shape__adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output, // 输入参数：梯度输出张量
    const at::Tensor& self) {      // 输入参数：自身张量

  // 根据 `aten/src/ATen/native/AdaptiveAveragePooling.cpp` 进行检查
  int64_t ndim = grad_output.ndimension(); // 获取梯度输出张量的维度数

  // 遍历非批处理维度，检查梯度输出是否具有非零大小
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        grad_output.size(i) > 0,
        "adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero size for non-batch dimensions, "
        "but grad_output has sizes ",
        grad_output.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  // 检查自身张量的维度是否为3D或4D
  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got ",
      self.sizes());
  // 检查梯度输出的数据类型是否与自身张量的数据类型相匹配
  TORCH_CHECK(
      self.dtype() == grad_output.dtype(),
      "expected dtype ",
      self.dtype(),
      " for `grad_output` but got dtype ",
      grad_output.dtype());

  // 返回一个形状为自身张量大小的Shape对象的向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算自适应平均池化的形状
std::vector<Shape> compute_shape__adaptive_avg_pool3d(
    const at::Tensor& self,          // 输入参数：自身张量
    at::IntArrayRef output_size) {   // 输入参数：输出大小

  // 根据 `aten/src/ATen/native/AdaptiveAveragePooling.cpp` 和 `aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp` 进行检查
  TORCH_CHECK(
      output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");
  TORCH_CHECK(
      (output_size[0] >= 0 && output_size[1] >= 0 && output_size[2] >= 0),
      "adaptive_avg_pool3d: elements of output_size must be greater than or equal to 0 "
      "but received {",
      output_size[0],
      ", ",
      output_size[1],
      ", ",
      output_size[2],
      "}");

  int64_t ndim = self.ndimension(); // 获取自身张量的维度数
  // 遍历非批处理维度，检查自身张量是否具有非零大小
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_avg_pool3d(): Expected self to have non-zero size for non-batch dimensions, "
        "but Tensor has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  // 检查自身张量的维度是否为4D或5D
  TORCH_CHECK(
      (ndim == 4 || ndim == 5),
      "adaptive_avg_pool3d(): Expected 4D or 5D tensor, but got ",
      self.sizes());

  int64_t channels = self.size(-4);   // 获取通道数
  int64_t output_depth = output_size[0];    // 获取输出深度
  int64_t output_height = output_size[1];   // 获取输出高度
  int64_t output_width = output_size[2];    // 获取输出宽度

  if (ndim == 4) {
    // 如果自身张量为4维，则返回一个形状为 {channels, output_depth, output_height, output_width} 的Shape对象的向量
    return {Shape(
        self.scalar_type(),
        {channels, output_depth, output_height, output_width})};
  } else {
    int64_t nbatch = self.size(0);   // 获取批次大小
    // 如果自身张量为5维，则返回一个形状为 {nbatch, channels, output_depth, output_height, output_width} 的Shape对象的向量
    return {Shape(
        self.scalar_type(),
        {nbatch, channels, output_depth, output_height, output_width})};
  }
}

// 计算自适应平均池化反向传播的形状
std::vector<Shape> compute_shape__adaptive_avg_pool3d_backward(
    const at::Tensor& grad_output,   // 输入参数：梯度输出张量
    const at::Tensor& self) {        // 输入参数：自身张量

  // 根据 `aten/src/ATen/native/AdaptiveAveragePooling.cpp` 进行检查
  int64_t ndim = grad_output.ndimension(); // 获取梯度输出张量的维度数

  // 遍历非批处理维度，检查梯度输出是否具有非零大小
  for (const auto i : c10::irange(1, ndim)) {
    # 检查梯度输出张量在指定维度上是否具有非零大小，否则抛出错误
    TORCH_CHECK(
        grad_output.size(i) > 0,
        "adaptive_avg_pool3d_backward(): Expected grad_output to have non-zero size for non-batch dimensions, "
        "but grad_output has sizes ",
        grad_output.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
    
    # 检查输入张量的维度是否为4D或5D，否则抛出错误
    TORCH_CHECK(
        (ndim == 4 || ndim == 5),
        "adaptive_avg_pool3d_backward(): Expected 4D or 5D tensor, but got ",
        self.sizes());
    
    # 检查输入张量的数据类型是否与梯度输出张量的数据类型相匹配，否则抛出错误
    TORCH_CHECK(
        self.dtype() == grad_output.dtype(),
        "expected dtype ",
        self.dtype(),
        " for `grad_output` but got dtype ",
        grad_output.dtype());
    
    # 返回一个Shape对象，其数据类型为self的标量类型，尺寸为self的向量化大小
    return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算 GLU 操作的反向传播形状
std::vector<Shape> compute_shape_glu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    int64_t dim) {
  // 返回包含当前张量形状的 Shape 向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算 GLU 操作的 JVP 形状
std::vector<Shape> compute_shape_glu_jvp(
    const at::Tensor& glu,
    const at::Tensor& x,
    const at::Tensor& dx,
    int64_t dim) {
  // 返回包含 glu 张量形状的 Shape 向量
  return {Shape(glu.scalar_type(), glu.sizes().vec())};
}

// 计算 clamp_min 操作的形状
std::vector<Shape> compute_shape_clamp_min(
    const at::Tensor& self,
    const at::Scalar& min) {
  // 返回包含当前张量形状的 Shape 向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算 _to_copy 操作的形状
std::vector<Shape> compute_shape__to_copy(
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    bool non_blocking,
    ::std::optional<at::MemoryFormat> memory_format) {
  if (dtype) {
    // 如果指定了 dtype，返回指定 dtype 的 Shape 向量
    return {Shape(*dtype, self.sizes().vec())};
  }
  // 否则返回当前张量形状的 Shape 向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算 clone 操作的形状
TORCH_API std::vector<Shape> compute_shape_clone(
    const at::Tensor& self,
    ::std::optional<at::MemoryFormat> memory_format) {
  // 返回包含当前张量形状的 Shape 向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算 stack 操作的形状
std::vector<Shape> compute_shape_stack(at::TensorList tensors, int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "stack expects a non-empty TensorList");
  auto wrapped_dim = at::maybe_wrap_dim(dim, tensors[0].ndimension() + 1);

  // 检查输入的张量列表是否尺寸相同
  at::IntArrayRef entry_shape = tensors[0].sizes();
  for (const auto i : c10::irange(1, tensors.size())) {
    TORCH_CHECK(
        tensors[i].sizes() == entry_shape,
        "stack expects each tensor to be equal size, but got ",
        entry_shape,
        " at entry 0 and ",
        tensors[i].sizes(),
        " at entry ",
        i);
  }

  // 计算结果张量的形状
  auto result_sizes = tensors[0].sizes().vec();
  result_sizes.insert(result_sizes.begin() + wrapped_dim, tensors.size());
  return {Shape(tensors[0].scalar_type(), result_sizes)};
}

// 计算 repeat 操作的形状
std::vector<Shape> compute_shape_repeat(
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  // 检查重复次数是否足够
  TORCH_CHECK_GE(static_cast<int64_t>(repeats.size()), self.dim());
  size_t num_new_dimensions = repeats.size() - self.dim();

  // 构造目标形状
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(repeats.size());
  for (const auto idx : c10::irange(repeats.size())) {
    target_size[idx] = padded_size[idx] * repeats[idx];
  }
  return {Shape(self.scalar_type(), target_size)};
}

// 计算 narrow_copy_symint 操作的形状
std::vector<Shape> compute_shape_narrow_copy_symint(
    const at::Tensor& self,
    int64_t dim,
    int64_t start,
    c10::SymInt length) {
  // 返回包含当前张量形状的 Shape 向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算 hardswish 操作的形状
std::vector<Shape> compute_shape_hardswish(const at::Tensor& self) {
  // 返回包含当前张量形状的 Shape 向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 计算 hardswish 操作的反向传播形状
    // 返回一个形状与输入张量相同的张量
    const at::Tensor& grad_output,
    // grad_output: 梯度输出张量的常量引用
    const at::Tensor& self) {
    // self: 输入张量的常量引用
    return {Shape(self.scalar_type(), self.sizes().vec())};
    // 返回一个张量，其形状与输入张量 self 相同
    // Shape(self.scalar_type(), self.sizes().vec()): 创建一个张量，使用输入张量 self 的数据类型和尺寸向量
    }
}

// 计算返回用于 SELU 操作的张量的形状
std::vector<Shape> compute_shape_selu(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 非原生操作的形状计算：返回一个标量的形状
std::vector<Shape> compute_shape_scalar(
    const at::Scalar& value,
    const at::ScalarType& type) {
  return {Shape(type, {})};
}

// 非原生操作的形状计算：返回扩展操作的形状
std::vector<Shape> compute_shape_expand(
    const Output& input,
    const std::vector<int64_t>& size,
    const bool& is_scalar_expand) {
  return {Shape(input.shape().scalar_type(), size)};
}

// 非原生操作的形状计算：返回视图操作的形状
std::vector<Shape> compute_shape_view(
    const Output& input,
    const std::vector<int64_t>& output_sizes) {
  const Shape& input_shape = input.shape();
  const auto complete_output_sizes =
      at::infer_size(output_sizes, input_shape.numel());
  return {Shape(input_shape.scalar_type(), complete_output_sizes)};
}

// 非原生操作的形状计算：返回类型转换后的形状
std::vector<Shape> compute_shape_cast(
    const Output& input,
    const at::ScalarType& dtype,
    const ::std::optional<at::ScalarType>& stype) {
  Shape shape = input.shape();
  shape.set_scalar_type(dtype);
  return {shape};
}

// 视图操作的形状计算：返回更新后的 strided 视图的形状
std::vector<Shape> compute_shape_as_strided_view_update(
    const Output& target,
    const Output& input,
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& stride,
    const int64_t& storage_offset) {
  return {Shape(target.shape().scalar_type(), size)};
}

// 视图操作的形状计算：返回 strided 视图的形状
std::vector<Shape> compute_shape_as_strided(
    const Output& input,
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& stride,
    const int64_t& storage_offset) {
  return {Shape(input.shape().scalar_type(), size)};
}

// 视图操作的形状计算：返回更新后的对角线视图的形状
std::vector<Shape> compute_shape_diagonal_view_update(
    const Output& target,
    const Output& input,
    const int64_t& offset,
    const int64_t& dim1,
    const int64_t& dim2) {
  return {target.shape()};
}

// 视图操作的形状计算：返回对角线视图的形状
std::vector<Shape> compute_shape_diagonal(
    const Output& input,
    const int64_t& offset,
    const int64_t& dim1,
    const int64_t& dim2) {
  return {MakeDiagonalShape(input.shape(), offset, dim1, dim2)};
}

// 视图操作的形状计算：返回更新后的窄视图的形状
std::vector<Shape> compute_shape_narrow_view_update(
    const Output& input,
    const Output& source,
    const std::vector<int64_t>& base_indices) {
  return {input.shape()};
}

// 视图操作的形状计算：返回窄视图的形状
std::vector<Shape> compute_shape_narrow(
    const Output& input,
    const std::vector<int64_t>& base_indices,
    const std::vector<int64_t>& sizes) {
  return {Shape(input.shape().scalar_type(), sizes)};
}

// 视图操作的形状计算：返回置换操作的形状
std::vector<Shape> compute_shape_permute(
    const Output& input,
    const std::vector<int64_t>& dims) {
  return {MakePermuteShape(input.shape(), dims)};
}

// 视图操作的形状计算：返回调整大小操作的形状
std::vector<Shape> compute_shape_resize(
    const Output& input,
    const std::vector<int64_t>& size) {
  return {Shape(input.shape().scalar_type(), size)};
}

// 视图操作的形状计算：返回更新后的选择视图的形状
std::vector<Shape> compute_shape_select_view_update(
    const Output& target,
    const Output& source,
    const int64_t& dim,
    const int64_t& start,
    const int64_t& end,
    const int64_t& stride) {
  return {target.shape()};
}

// 视图操作的形状计算：返回选择操作的形状
std::vector<Shape> compute_shape_select(
    const Output& input,
    const std::vector<int64_t>& indices) {
  return {Shape(input.shape().scalar_type(), indices)};
}
    # 定义一个函数，返回一个包含选择操作结果的形状的元组
    const int64_t& dim,
    # 定义一个常量引用参数 dim，表示选择操作的维度
    const int64_t& start,
    # 定义一个常量引用参数 start，表示选择操作的起始位置
    const int64_t& end,
    # 定义一个常量引用参数 end，表示选择操作的结束位置
    const int64_t& stride) {
    # 定义一个常量引用参数 stride，表示选择操作的步长
    return {MakeSelectShape(input.shape(), dim, start, end, stride)};
    # 调用 MakeSelectShape 函数，根据输入张量的形状和给定的 dim、start、end、stride 参数进行选择操作，并将结果作为包含在大括号中的元组返回
}
// 计算 squeeze 操作后的形状
std::vector<Shape> compute_shape_squeeze(const Output& input, const int& dim) {
  // 获取输入的形状
  const auto& input_shape = input.shape();
  // 返回 squeeze 操作后的形状
  return {torch::lazy::Shape(
      input_shape.scalar_type(),
      BuildSqueezedDimensions(input_shape.sizes(), dim))};
}
// 计算 unsqueeze 操作后的形状
std::vector<Shape> compute_shape_unsqueeze(
    const Output& input,
    const int& dim) {
  // 获取输入的形状
  const auto& input_shape = input.shape();
  // 返回 unsqueeze 操作后的形状
  return {torch::lazy::Shape(
      input_shape.scalar_type(),
      BuildUnsqueezedDimensions(input_shape.sizes(), dim))};
}

// 计算 select_scatter 操作后的形状
std::vector<Shape> compute_shape_select_scatter(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t dim,
    int64_t index) {
  // 创建 self 的元数据
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(),
      self.sym_strides(),
      /*dtype=*/::std::make_optional(self.scalar_type()),
      /*layout=*/::std::make_optional(self.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  // 创建 src 的元数据
  auto src_meta = at::native::empty_strided_meta_symint(
      src.sym_sizes(),
      src.sym_strides(),
      /*dtype=*/::std::make_optional(src.scalar_type()),
      /*layout=*/::std::make_optional(src.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  // 执行 select_scatter 操作
  auto out_meta = at::compositeexplicitautogradnonfunctional::select_scatter(
      self_meta, src_meta, dim, index);
  // 返回 select_scatter 操作后的形状
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

// 计算 diagonal_scatter 操作后的形状
std::vector<Shape> compute_shape_diagonal_scatter(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  // 创建 self 的元数据
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(),
      self.sym_strides(),
      /*dtype=*/::std::make_optional(self.scalar_type()),
      /*layout=*/::std::make_optional(self.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  // 创建 src 的元数据
  auto src_meta = at::native::empty_strided_meta_symint(
      src.sym_sizes(),
      src.sym_strides(),
      /*dtype=*/::std::make_optional(src.scalar_type()),
      /*layout=*/::std::make_optional(src.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  // 执行 diagonal_scatter 操作
  auto out_meta = at::compositeexplicitautogradnonfunctional::diagonal_scatter(
      self_meta, src_meta, offset, dim1, dim2);
  // 返回 diagonal_scatter 操作后的形状
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

// 计算 slice_scatter_symint 操作后的形状
std::vector<Shape> compute_shape_slice_scatter_symint(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t dim,
    ::std::optional<c10::SymInt> start,
    ::std::optional<c10::SymInt> end,
    c10::SymInt step) {
  // 创建一个包含空数据的元数据张量，使用符号化整数来表示
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(),                        // 使用 self 的符号化大小
      self.sym_strides(),                      // 使用 self 的符号化步长
      /*dtype=*/::std::make_optional(self.scalar_type()),  // 可选的数据类型，使用 self 的标量类型
      /*layout=*/::std::make_optional(self.layout()),       // 可选的布局信息，使用 self 的布局
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),  // 可选的设备信息，使用元数据设备
      /*pin_memory=*/::std::nullopt);          // 不指定内存锁定

  // 创建一个包含空数据的元数据张量，使用符号化整数来表示
  auto src_meta = at::native::empty_strided_meta_symint(
      src.sym_sizes(),                         // 使用 src 的符号化大小
      src.sym_strides(),                       // 使用 src 的符号化步长
      /*dtype=*/::std::make_optional(src.scalar_type()),   // 可选的数据类型，使用 src 的标量类型
      /*layout=*/::std::make_optional(src.layout()),        // 可选的布局信息，使用 src 的布局
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),  // 可选的设备信息，使用元数据设备
      /*pin_memory=*/::std::nullopt);          // 不指定内存锁定

  // 使用切片散开符号化整数操作，对 self_meta 和 src_meta 执行切片散开操作
  auto out_meta =
      at::compositeexplicitautogradnonfunctional::slice_scatter_symint(
          self_meta, src_meta, dim, start, end, step);

  // 返回一个形状信息对象，包含输出元数据的标量类型和大小向量
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}
}

std::vector<Shape> compute_shape_as_strided_scatter_symint(
    const at::Tensor& self,
    const at::Tensor& src,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride,
    ::std::optional<c10::SymInt> storage_offset) {
  // 创建 self 的元数据，使用 empty_strided_meta_symint 函数
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(),
      self.sym_strides(),
      /*dtype=*/::std::make_optional(self.scalar_type()),
      /*layout=*/::std::make_optional(self.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  
  // 创建 src 的元数据，使用 empty_strided_meta_symint 函数
  auto src_meta = at::native::empty_strided_meta_symint(
      src.sym_sizes(),
      src.sym_strides(),
      /*dtype=*/::std::make_optional(src.scalar_type()),
      /*layout=*/::std::make_optional(src.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  
  // 调用 as_strided_scatter_symint 函数，计算并返回输出元数据
  auto out_meta =
      at::compositeexplicitautogradnonfunctional::as_strided_scatter_symint(
          self_meta, src_meta, size, stride, storage_offset);
  
  // 构造并返回包含输出元数据形状的 Shape 向量
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<Shape> compute_shape_normal_functional(
    const at::Tensor& self,
    double mean,
    double std,
    ::std::optional<at::Generator> generator) {
  // 返回包含 self 形状的 Shape 向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_uniform(
    const at::Tensor& self,
    double from,
    double to,
    ::std::optional<at::Generator> generator) {
  // 返回包含 self 形状的 Shape 向量
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// 恢复未使用参数的警告
#pragma GCC diagnostic pop

} // namespace lazy
} // namespace torch
```