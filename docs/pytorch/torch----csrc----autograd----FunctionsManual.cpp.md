# `.\pytorch\torch\csrc\autograd\FunctionsManual.cpp`

```py
// 引入 Torch 库中的不同模块和头文件，用于自动求导和张量操作

#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/ScalarOps.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/Activation.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/SmallBuffer.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <ciso646>
#include <functional>
#include <numeric>
#include <utility>

// 自动生成代码的辅助函数
// 这些函数曾经内联到自动生成的 Functions.cpp 中

namespace torch {
namespace autograd {
namespace generated {
namespace details {

using at::areAnyTensorSubclassLike;
using at::IntArrayRef;
using at::OptionalIntArrayRef;
using at::Scalar;
using at::Tensor;
using at::TensorList;

// CuDNN 双向传播不支持的消息提示
const char* kCudnnDoubleBackwardMsg =
    "Double backwards is not supported for CuDNN RNNs due to limitations in the CuDNN API. To run double backwards, please disable the CuDNN backend temporarily while running the forward pass of your RNN. For example: \nwith torch.backends.cudnn.flags(enabled=False):\n    output = model(inputs)";

// 应用损失函数的归约操作
Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

// 检查是否定义了可选的 Tensor
static bool isDefined(const std::optional<Tensor>& t) {
  return t.has_value() && t->defined();
}

// 将可选的 Tensor 转换为非可选的 Tensor
Tensor toNonOptTensor(const std::optional<Tensor>& t) {
  return t.has_value() ? *t : Tensor();
}

// 将可选的 Tensor 转换为非可选的前向梯度 Tensor
Tensor toNonOptFwGrad(const std::optional<Tensor>& t) {
  return (t.has_value() && t->defined()) ? t->_fw_grad(/*level */ 0) : Tensor();
}

// 将可选的 Tensor 转换为非可选的前向原始 Tensor
Tensor toNonOptPrimal(const std::optional<Tensor>& t) {
  if (t.has_value() && t->defined()) {
    if (t->unsafeGetTensorImpl()->is_wrapped_number()) {
      return *t;
    }
    return t->_fw_primal(/* level */ 0);
  }
  return Tensor();
}

// 复制 Tensor 范围内的输出到变量列表中的指定位置
void copy_range(variable_list& out, IndexRange range, const Tensor& t) {
  TORCH_CHECK(range.second <= out.size());
  TORCH_CHECK(
      range.second - range.first == 1, "inconsistent range for Tensor output");
  out[range.first] = t;
}

// 结束自动生成的命名空间 details
} // namespace details
} // namespace generated
} // namespace autograd
} // namespace torch
// 拷贝指定范围内的张量到输出变量列表中
void copy_range(variable_list& out, IndexRange range, at::ArrayRef<Tensor> t) {
  // 检查范围是否合法，不超过输出变量列表的大小
  TORCH_CHECK(range.second <= out.size());
  // 检查张量列表与范围的大小是否一致
  TORCH_CHECK(
      range.second - range.first == t.size(),
      "inconsistent range for TensorList output");
  // 将张量列表 t 中的数据拷贝到输出变量列表 out 的指定范围内
  std::copy(
      t.begin(), t.end(), out.begin() + static_cast<int64_t>(range.first));
}

// 计算张量的 copysign 自反向传播梯度
Tensor copysign_tensor_self_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result) {
  // 计算结果张量 result 与自身张量 self 的比值
  auto ratio = result / self;
  // 当 self 为零时，用零填充比值张量 ratio
  ratio.masked_fill_(self == 0, 0);
  // 返回梯度 grad 乘以比值张量 ratio
  return grad * ratio;
}

// 返回未实现的通用函数的结果，用于 Tensor 类型
template <typename T>
T not_implemented_base(const char* name, const char* reason) {
  // 构建未实现的错误消息
  std::string msg =
      c10::str("the derivative for '", name, "' is not implemented.");
  // 如果存在具体的原因，将其添加到错误消息中
  if (reason[0] != '\0') {
    msg = c10::str(msg, " ", reason);
  };
  // 抛出未实现错误，包含自定义的错误消息
  TORCH_CHECK_NOT_IMPLEMENTED(false, msg);
}

// 返回未实现的 Tensor 类型的函数结果
Tensor not_implemented(const char* name, const char* reason) {
  return not_implemented_base<Tensor>(name, reason);
}

// 返回未实现的 std::vector<Tensor> 类型的函数结果
std::vector<Tensor> not_implemented_list(const char* name, const char* reason) {
  return not_implemented_base<std::vector<Tensor>>(name, reason);
}

// 根据标量值可能进行乘法操作
Tensor maybe_multiply(const Tensor& t, const Scalar& s) {
  bool is_one = false;
  // 检查标量类型，判断其是否等于 1
  if (s.isFloatingPoint()) {
    is_one = s.toSymFloat() == 1;
  } else if (s.isIntegral(true)) {
    is_one = s.toSymInt() == 1;
  }

  // 如果标量值为 1，返回原始张量 t；否则返回 t 乘以标量 s 的结果
  if (is_one) {
    return t;
  } else {
    return t * s;
  }
}

// 计算安全的尺寸大小，用于标量和整数数组
int64_t _safe_size(IntArrayRef sizes, IntArrayRef dim) {
  // 初始化尺寸为 1
  int64_t size = 1;
  // 如果尺寸数组为空，直接返回 1
  if (sizes.empty()) {
    return 1;
  }
  // 计算指定维度上的尺寸乘积
  for (auto d : dim) {
    // 调用 maybe_wrap_dim 函数确保维度合法性，然后累积尺寸乘积
    d = at::maybe_wrap_dim(d, static_cast<int64_t>(sizes.size()));
    size *= sizes[d];
  }
  return size;
}

// 计算安全的尺寸大小，用于符号整数数组和整数数组
static c10::SymInt _safe_size(c10::SymIntArrayRef sizes, c10::IntArrayRef dim) {
  // 初始化符号整数尺寸为 1
  c10::SymInt size = 1;
  // 如果尺寸数组为空，直接返回 1
  if (sizes.empty()) {
    return 1;
  }
  // 计算指定维度上的尺寸乘积
  for (auto d : dim) {
    // 调用 maybe_wrap_dim 函数确保维度合法性，然后累积尺寸乘积
    d = at::maybe_wrap_dim(d, static_cast<int64_t>(sizes.size()));
    size *= sizes[d];
  }
  return size;
}

// 处理实数到复数的梯度转换，返回处理后的梯度张量
Tensor handle_r_to_c(ScalarType self_st, Tensor gradient_result) {
  // 如果 self_st 不是复数类型，但 gradient_result 是复数类型
  if (!at::isComplexType(self_st) && gradient_result.is_complex()) {
    // 返回复数张量的实部
    return at::real(gradient_result);
  }
  // 否则直接返回原始的梯度结果张量
  return gradient_result;
}

// 处理实数到复数的梯度转换，返回处理后的梯度张量
static Tensor handle_r_to_c(const Tensor& self, Tensor gradient_result) {
  // 如果 self 不是复数类型，但 gradient_result 是复数类型
  if (!self.is_complex() && gradient_result.is_complex()) {
    // 返回复数张量的实部
    return at::real(gradient_result);
  }
  // 否则直接返回原始的梯度结果张量
  return gradient_result;
}

// 恢复被减少维度的张量的形状
Tensor restore_reduced_dims(
    const Tensor& output,
    IntArrayRef dims,
    bool keepdim) {
  // 如果需要保持维度，直接返回原始输出张量
  if (keepdim) {
    return output;
  }
  // 计算恢复后的总维度数
  auto total_dims = output.dim() + dims.size();
  // 初始化目标形状为全零向量
  std::vector<c10::SymInt> target_shape(total_dims, 0);
  // 遍历指定的维度列表
  for (int64_t i : dims) {
    // 处理负数索引，将其转换为对应的正数索引
    if (i < 0) {
      i = static_cast<int64_t>(total_dims) + i;
    }
    // 将目标形状中对应维度设为 1
    target_shape[i] = 1;
  }
  // 初始化遍历输出张量的索引
  int64_t j = 0;
  // 遍历输出张量的符号整数尺寸
  for (const c10::SymInt& i : output.sym_sizes()) {
    // 寻找目标形状中下一个为零的位置
    while (target_shape[j] > 0)
      j++;
    // 将输出张量的当前尺寸赋给目标形状的对应位置
    target_shape[j++] = i;
  }
  // 根据目标形状重塑输出张量，保留符号整数信息
  return output.reshape_symint(target_shape);
}

// 按计数缩放梯度张量
Tensor scale_grad_by_count(
    const Tensor& grad,
    const Tensor& mask,
    # 计算梯度 grad 在指定维度 dims 上的和，返回一个与 mask 形状相同的张量
    return (grad / mask.sum(dims, true)) * mask;
}

// 计算带有自动微分的 Jacobian 向量乘积 (JVP)，用于计算范数的反向传播梯度
Tensor norm_jvp(
    const Tensor& self_p,  // 输入张量 self_p
    const Tensor& self_t,  // 输入张量 self_t
    const optional<Scalar>& p_,  // 可选参数 p_
    Tensor norm,  // 输入张量 norm
    IntArrayRef dim,  // 维度数组 dim
    bool keepdim) {  // 是否保持维度标志

  // NB: 我们在输出中对 NaN 进行了零填充，但仍然做浮点除法，这导致 ASAN 报错。
  //     为了缓解 ASAN 的问题，可以在除法之前将有问题的值填充为任意值，
  //     但由于性能损失，我们决定不这么做。相反，我们只在必要时抑制 ASAN 报错。

  size_t ndim = self.dim();  // 获取输入张量 self 的维度数
  double p = p_.value_or(2.0).toDouble();  // 获取参数 p_ 的值，如果未提供，默认为 2.0
  Tensor self_scaled;  // 定义缩放后的张量 self_scaled
  Tensor scale_v;  // 定义缩放因子张量 scale_v

  if (!keepdim && self.dim() != 0) {
    grad = unsqueeze_multiple(grad, dim, ndim);  // 如果不保持维度且 self 不是零维，则扩展 grad 和 norm
    norm = unsqueeze_multiple(norm, dim, ndim);
  }

  if (p == 0.0) {
    return {};  // 如果 p 为 0，返回空张量
  } else if (p == 1.0) {
    return self.sgn() * grad;  // 如果 p 为 1，返回 self 的符号函数与 grad 的乘积
  } else if (p == 2.0) {
    return grad * (self / norm).masked_fill_(norm == 0, 0);  // 如果 p 为 2，返回 grad 乘以 (self / norm)，并在 norm 为 0 时填充 0
  } else if (std::isinf(p)) {
    // 计算 amax(abs(self), dim, keepdim) 的导数，但考虑 NaN
    // 创建 `argmax` 的掩码：如果 self.abs() == norm 或为 NaN，则为 argmax
    auto self_abs = self.abs();
    auto mask = self_abs.eq(norm).logical_or(self_abs.isnan());
    return self.sgn() * ((grad / mask.sum(dim, true)) * mask);  // 返回 self 的符号函数乘以 grad 与 mask 和的商的乘积
  } else if (p < 1.0) {
    self_scaled =
        self.sgn() * self.abs().pow_(p - 1).masked_fill_(self == 0, 0);  // 如果 p 小于 1，计算 self 的缩放后的张量 self_scaled
    return self_scaled * grad * norm.pow(1 - p);  // 返回 self_scaled、grad 和 norm 的乘积
  } else if (p < 2.0) {
    self_scaled = self.sgn() * self.abs().pow_(p - 1);  // 如果 p 小于 2，计算 self 的缩放后的张量 self_scaled
    scale_v = grad / norm.pow(p - 1);  // 计算 grad 与 norm 的 p-1 次幂的商
    scale_v.masked_fill_(norm == 0, 0);  // 在 norm 为 0 时填充 0
    return self_scaled * scale_v;  // 返回 self_scaled 和 scale_v 的乘积
  } else {
    self_scaled = self * self.abs().pow_(p - 2);  // 如果 p 大于等于 2，计算 self 的缩放后的张量 self_scaled
    scale_v = grad / norm.pow(p - 1);  // 计算 grad 与 norm 的 p-1 次幂的商
    scale_v.masked_fill_(norm == 0, 0);  // 在 norm 为 0 时填充 0
    return self_scaled * scale_v;  // 返回 self_scaled 和 scale_v 的乘积
  }
}
    // 求解向量范数的函数，支持不同的范数计算
    // 参数包括：dim 表示维度，keepdim 表示是否保持维度
    
    IntArrayRef dim,
    bool keepdim) {
      // 注意：目前 norm_jvp 也用于 dist 的 jvp（具有两个可微输入）
      //     但 self_t 仍然不能是 ZT，因为这将要求 self_t 和 other_t 都是 ZT
      TORCH_INTERNAL_ASSERT(!self_t._is_zerotensor());
      // 获取 self_p 的维度
      size_t ndim = self_p.dim(); // 复合一致性？
      // 获取范数的值，默认为 2.0
      double p = p_.value_or(2.0).toDouble();
    
      if (p == 0.0) {
        // 如果 p 等于 0，则返回与 norm 相同形状的零张量
        return at::zeros_like(norm);
      } else if (p == 1.0) {
        // 如果 p 等于 1，则计算符号函数后乘以 self_t 的共轭，并取实部，然后按 dim 维度求和
        auto result = self_p.sgn();
        result = areAnyTensorSubclassLike({self_t}) ? result.mul(self_t.conj())
                                                    : result.mul_(self_t.conj());
        result = at::real(result);
        return result.sum(dim, keepdim);
      } else if (p == 2.0) {
        // 如果 p 等于 2，则计算 self_p 乘以 self_t 的共轭的实部，按 dim 维度求和后再除以 norm，并在 norm 为 0 时填充 0
        auto result = self_p.mul(self_t.conj());
        result = at::real(result);
        result = result.sum(dim, keepdim);
        return result.div_(norm).masked_fill_(norm == 0, 0);
      } else if (std::isinf(p)) {
        // 如果 p 为无穷大，则处理特殊情况
        if (!keepdim && self_p.dim() != 0) {
          // 如果不保持维度并且 self_p 的维度不为 0，则对 norm 在指定的 dim 维度上进行扩展
          norm = unsqueeze_multiple(norm, dim, ndim);
        }
        // 检查 self_p 和 norm 是否为 NaN
        const auto self_isnan = self_p.isnan();
        const auto norm_isnan = norm.isnan();
        const auto& self_and_norm_isnan = areAnyTensorSubclassLike({norm})
            ? self_isnan.logical_and(norm_isnan)
            : self_isnan.logical_and_(norm_isnan);
        // 计算等于最大值的元素个数
        const auto is_eq_max =
            (self_p.abs() == norm).logical_or_(self_and_norm_isnan).type_as(norm);
        auto nb_max = is_eq_max.count_nonzero(dim);
        if (self_p.dim() != 0) {
          // 如果 self_p 的维度不为 0，则对 nb_max 在指定的 dim 维度上进行扩展
          nb_max = unsqueeze_multiple(nb_max, dim, ndim);
        }
        // 计算最终结果
        return (at::real(self_p.sgn() * self_t.conj()) * is_eq_max / nb_max)
            .sum(dim, keepdim);
      } else if (p < 1.0) {
        // 如果 p 小于 1，则计算 abs(self_p) 的 (p-1) 次方后乘以 self_p 的共轭的实部，按 dim 维度求和，最后乘以 norm 的 (1-p) 次方
        auto sumpow_t = (self_p.abs().pow_(p - 1).masked_fill_(self_p == 0, 0) *
                         at::real(self_p.sgn() * self_t.conj()))
                            .sum(dim, keepdim);
        return sumpow_t * norm.pow(1 - p);
      } else if (p < 2.0) {
        // 如果 p 在 1 和 2 之间，则计算 abs(self_p) 的 (p-1) 次方乘以 self_p 的共轭的实部后按 dim 维度求和，最后除以 norm 的 (p-1) 次方，并在 norm 为 0 时填充 0
        auto sumpow_t =
            (self_p.abs().pow_(p - 1) * at::real(self_p.sgn() * self_t.conj()))
                .sum(dim, keepdim);
        auto out = sumpow_t / norm.pow(p - 1);
        return out.masked_fill_(norm == 0, 0);
      } else {
        // 如果 p 大于 2，则计算 abs(self_p) 的 (p-2) 次方乘以 self_p 和 self_t 的共轭的实部后按 dim 维度求和，最后除以 norm 的 (p-1) 次方，并在 norm 为 0 时填充 0
        auto sumpow_t =
            (self_p.abs().pow_(p - 2) * at::real(self_p * self_t.conj()))
                .sum(dim, keepdim);
        auto out = sumpow_t / norm.pow(p - 1);
        return out.masked_fill_(norm == 0, 0);
      }
    }
}

// 函数：计算向量范数的Jacobian向量积
Tensor norm_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const optional<Scalar>& p_,
    Tensor norm) {
  // 调用具体实现函数，传入参数self_p, self_t, p_, norm，使用默认keepdim=true
  return norm_jvp(self_p, self_t, p_, std::move(norm), {}, true);
}

// 函数：从填充张量的反向传播中返回嵌套张量
Tensor _nested_from_padded_backward(
    const Tensor& grad,
    const Tensor& input,
    bool do_transform_0213) {
  if (do_transform_0213) {
    // 计算新的大小：第0维不变，第2维取input的第2维，第1和3维分别相乘
    auto new_sizes = {
        input.size(0), input.size(2), (input.size(1) * input.size(3))};
    // 将grad转换为填充张量，填充值为0，大小为new_sizes
    auto out = grad.to_padded_tensor(0, new_sizes);
    // 扩展最后一维的大小顺序：第0和2维度不变，第1和3维度交换
    auto expand_last_dim_size = {
        input.size(0), input.size(2), input.size(1), input.size(3)};
    // 返回按指定顺序视图的out张量
    return out.view(expand_last_dim_size).permute({0, 2, 1, 3});
  }
  // 否则，直接将grad转换为填充张量，填充值为0，大小为input的大小
  return grad.to_padded_tensor(0, input.sizes());
}

// 函数：线性操作的双向传播
std::tuple<Tensor, Tensor, Tensor> linear_double_backward(
    const variable_list& grads,
    const Tensor& self,
    const Tensor& grad_output,
    const Tensor& weight) {
  // 如果grad_output未定义，返回三个空张量
  if (!grad_output.defined()) {
    return std::make_tuple(Tensor(), Tensor(), Tensor());
  }

  Tensor grad_self, grad_grad_output, grad_weight;

  // 如果grads[1]已定义
  if (grads[1].defined()) {
    // 计算grad_self，根据grad_output的维度情况调整形状
    grad_self =
        (grad_output.dim() == 1 ? grad_output.unsqueeze(0) : grad_output)
            .matmul(grads[1]);
    // 如果grad_output是1维的，去除多余的维度
    if (grad_output.dim() == 1) {
      grad_self = grad_self.squeeze(0);
    }
  }
  // 如果grads[0]已定义
  if (grads[0].defined()) {
    // 计算grad_weight，根据grad_output和grads[0]的维度情况调整形状
    grad_weight =
        (grad_output.dim() == 1 ? grad_output.unsqueeze(1) : grad_output.mT())
            .matmul(grads[0].dim() == 1 ? grads[0].unsqueeze(0) : grads[0]);
  }

  // 如果grads[0], grads[1], grads[2]中任何一个已定义
  if (grads[0].defined() || grads[1].defined() || grads[2].defined()) {
    // 初始化grad_grad_output为和grad_output相同形状的零张量
    grad_grad_output = at::zeros_like(grad_output);
    // 如果grad_output是1维的，扩展为2维
    if (grad_output.dim() == 1) {
      grad_grad_output = grad_grad_output.unsqueeze(0);
    }
  }

  // 如果grads[0]已定义
  if (grads[0].defined()) {
    // 计算grad_grad_output，加上grads[0]与weight的矩阵乘积
    grad_grad_output = grad_grad_output +
        (grads[0].dim() == 1 ? grads[0].unsqueeze(0) : grads[0])
            .matmul(weight.mT());
  }
  // 如果grads[1]已定义
  if (grads[1].defined()) {
    // 计算grad_grad_output，加上self与grads[1]的转置矩阵乘积
    grad_grad_output = grad_grad_output +
        (self.dim() == 1 ? self.unsqueeze(0) : self).matmul(grads[1].mT());
  }
  // 如果grads[2]已定义
  if (grads[2].defined()) {
    // 计算grad_grad_output，加上grads[2]
    grad_grad_output = grad_grad_output + grads[2];
  }
  // 如果grad_grad_output已定义且grad_output是1维的，去除多余的维度
  if (grad_grad_output.defined() && grad_output.dim() == 1) {
    grad_grad_output = grad_grad_output.squeeze(0);
  }

  // 返回grad_self, grad_grad_output, grad_weight的元组
  return std::make_tuple(
      std::move(grad_self),
      std::move(grad_grad_output),
      std::move(grad_weight));
}

// 函数：线性代数中向量范数的反向传播
Tensor linalg_vector_norm_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const Scalar& scalar_ord,
    Tensor norm,
    const at::OptionalIntArrayRef& opt_dim,
    bool keepdim) {
  // 不需要处理dtype参数，因为在函数中通过广播处理
  auto dim = opt_dim.value_or(IntArrayRef({}));
  // 调用norm_jvp函数，传入参数self_p, self_t, scalar_ord, norm, dim, keepdim
  return norm_jvp(self_p, self_t, scalar_ord, std::move(norm), dim, keepdim);
}

// 函数：线性代数中向量范数的反向传播
Tensor linalg_vector_norm_backward(
    Tensor grad,
    const Tensor& self,
    const Scalar& scalar_ord,
    Tensor norm,
    const at::OptionalIntArrayRef& opt_dim,
    bool keepdim) {
  // 不需要处理 dtype 参数，因为它通过函数中的广播处理了
  // 获取可选的维度参数，如果没有提供则使用空数组
  auto dim = opt_dim.value_or(IntArrayRef({}));
  // 调用 norm_backward 函数进行梯度反向传播计算
  // 使用给定的梯度 grad、张量 self、标量 ord、规范值 norm、维度 dim 和 keepdim 参数
  return norm_backward(
      std::move(grad), self, scalar_ord, std::move(norm), dim, keepdim);
}

// 计算指数函数的反向传播梯度
Tensor pow_backward(Tensor grad, const Tensor& self, const Scalar& exponent) {
  // 如果指数为0，返回一个与self形状相同的零张量
  if (exponent.equal(0.0)) {
    return at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    // 定义一个lambda函数，根据指数exp计算梯度
    auto grad_lambda = [&](auto exp) {
      return grad * (exp * self.pow(exp - 1)).conj();
    };
    // 根据指数的实部和虚部类型，选择不同的计算方式
    Tensor out = (exponent.isComplex())
        ? grad_lambda(exponent.toComplexDouble())
        : grad_lambda(exponent.toDouble());
    // 将结果处理为复数形式并返回
    return handle_r_to_c(self, std::move(out));
  }
}

// 计算指数函数自身的反向传播梯度
Tensor pow_backward_self(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& exponent) {
  // 使用`at::where`根据指数为0的条件选择返回0或计算的结果
  auto out = at::where(
      exponent == 0.0,
      at::zeros({}, grad.options()),
      grad * (exponent * self.pow(exponent - 1)).conj());
  // 将结果处理为复数形式并返回
  return handle_r_to_c(self, std::move(out));
}

// 根据特定规则计算指数函数指数部分的反向传播梯度
Tensor pow_backward_exponent(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& exponent,
    const Tensor& result) {
  Tensor cond;
  // 根据指数是否为复数类型选择条件判断方式
  if (exponent.is_complex()) {
    auto is_real_exp =
        at::logical_and(at::imag(exponent) == 0, at::real(exponent) >= 0);
    cond = at::logical_and(self == 0, is_real_exp);
  } else {
    cond = at::logical_and(self == 0, exponent >= 0);
  }
  // 推广self和exponent的数据类型
  auto promoted_dtype = at::result_type(self, exponent);
  auto self_ = self.to(promoted_dtype);

  // 根据条件使用`at::where`选择返回0或计算的结果，并返回处理后的结果
  auto out =
      grad *
      at::where(
          cond, at::zeros({}, grad.options()), (result * self_.log()).conj());
  return handle_r_to_c(exponent, std::move(out));
}

// 根据特定规则计算指数函数指数部分的反向传播梯度（基于标量指数）
Tensor pow_backward_exponent(
    const Tensor& grad,
    const Scalar& base,
    const Tensor& exponent,
    const Tensor& result) {
  auto grad_lambda = [](const Tensor& a, const Scalar& b) {
    return (a * b.log()).conj();
  };
  // 将标量基数推广为复数形式，如果指数为复数且基数不是复数的情况下
  auto base_ = exponent.is_complex() && !base.isComplex()
      ? base.toComplexDouble()
      : base;
  // 如果基数为0，根据指数的条件选择返回0或计算的结果，并返回处理后的结果
  if (base.equal(0.0)) {
    auto cond = [](auto exp) {
      if (exp.is_complex()) {
        return at::logical_and(at::imag(exp) == 0, at::real(exp) >= 0);
      } else {
        return exp >= 0;
      }
    };
    auto out = grad *
        at::where(cond(exponent),
                  at::zeros({}, grad.options()),
                  grad_lambda(result, base_));
    return handle_r_to_c(exponent, std::move(out));
  } else {
    // 否则直接计算结果并返回处理后的结果
    auto out = grad * grad_lambda(result, base_);
    return handle_r_to_c(exponent, std::move(out));
  }
}

// 计算角度函数的反向传播梯度
Tensor angle_backward(const Tensor& grad, const Tensor& self) {
  if (self.is_complex()) {
    // 根据self是否为复数，选择返回0或计算的结果，并返回处理后的结果
    return at::where(
        self == 0.0,
        at::zeros({}, self.options()),
        grad * self / self.abs().pow(2) *
            Scalar(c10::complex<double>{0.0, 1.0}));
  } else {
    // 对于实数情况，直接返回0
    return at::zeros({}, self.options());
  }
}
    return at::zeros_like(self, at::MemoryFormat::Preserve);


    调用 PyTorch 的 at::zeros_like 函数，用于创建一个与 self 张量相同大小和数据类型的全零张量。
    at::MemoryFormat::Preserve 表示保持张量的内存格式不变。
}

Tensor mvlgamma_backward(const Tensor& grad, const Tensor& self, int64_t p) {
  // 创建一个张量，其中包含从 -p/2 + 0.5 开始，以步长 0.5 的序列
  Tensor args =
      at::arange(-static_cast<double>(p) / 2. + 0.5, 0.5, 0.5, self.options());
  // 将 self 张量在最后一维度上增加一个维度，然后加到 args 张量上
  args = args.add(self.unsqueeze(-1));
  // 返回梯度乘以 args 张量在最后一维度上的 digamma 函数的和
  return grad * args.digamma_().sum(-1);
}

Tensor sgn_backward(const Tensor& x, const Tensor& gx, const Tensor& sgn) {
  // 如果 x 是复数张量
  if (x.is_complex()) {
    // 计算 x 的绝对值
    auto abs = x.abs();
    // 返回 gx 减去 sgn * sgn 的共轭乘积，再除以 2 * abs，当 abs 为零时用 0 替换
    return ((gx - (sgn * sgn) * gx.conj()) / (2. * abs))
        .masked_fill_(abs == 0., 0.);
  } else {
    // 返回一个与 sgn 相同大小和选项的零张量
    return at::_efficientzerotensor(sgn.sizes(), sgn.options());
  }
}

Tensor masked_fill_backward(const Tensor& grad, const Tensor& mask) {
  // 如果 grad 或 mask 中有任何张量类似的子类
  // 因为 masked_select 在 functorch 中的形状依赖于数据，所以不好用
  return areAnyTensorSubclassLike({grad, mask})
      // 如果是这种情况，返回通过 mask 选择 grad 或 0 的和
      ? at::where(mask, grad, 0).sum()
      // 否则返回通过 mask 选择 grad 的和
      : grad.masked_select(mask).sum();
}

template <typename T>
Tensor mul_tensor_backward(const Tensor& grad, T other, ScalarType self_st) {
  // 计算 grad 与 other 的共轭乘积
  auto out = grad * other.conj();
  // 使用 handle_r_to_c 处理将结果转换为复数张量
  return handle_r_to_c(self_st, std::move(out));
}
template Tensor mul_tensor_backward(const Tensor&, Tensor, ScalarType);
template Tensor mul_tensor_backward(const Tensor&, Scalar, ScalarType);

template <typename T>
Tensor div_tensor_self_backward(
    const Tensor& grad,
    T other,
    ScalarType self_st,
    const std::optional<c10::string_view>& rounding_mode) {
  // 如果 rounding_mode 有值
  if (rounding_mode.has_value()) {
    // 返回一个与 grad 相同大小和 self_st 数据类型的零张量
    return at::zeros_like(grad, grad.options().dtype(self_st));
  }

  // 计算 grad 除以 other 的共轭
  auto result = grad / other.conj();
  // 使用 handle_r_to_c 处理将结果转换为复数张量
  return handle_r_to_c(self_st, std::move(result));
}
template Tensor div_tensor_self_backward(
    const Tensor&,
    Tensor,
    ScalarType,
    const std::optional<c10::string_view>&);
template Tensor div_tensor_self_backward(
    const Tensor&,
    Scalar,
    ScalarType,
    const std::optional<c10::string_view>&);

template <typename T>
Tensor div_tensor_self_backward(
    const Tensor& grad,
    T other,
    ScalarType self_st) {
  // 调用带有 rounding_mode 参数的 div_tensor_self_backward 函数
  return div_tensor_self_backward(
      grad, std::move(other), self_st, c10::nullopt);
}
template Tensor div_tensor_self_backward(const Tensor&, Tensor, ScalarType);
template Tensor div_tensor_self_backward(const Tensor&, Scalar, ScalarType);

Tensor div_tensor_other_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    const std::optional<c10::string_view>& rounding_mode) {
  // 如果 rounding_mode 有值
  if (rounding_mode.has_value()) {
    // 返回一个与 grad 相同大小和 other 数据类型的零张量
    return at::zeros_like(grad, grad.options().dtype(other.scalar_type()));
  }

  // 计算 grad 乘以 -((self / other) / other 的共轭)
  auto result = -grad * ((self / other) / other).conj();
  // 使用 handle_r_to_c 处理将结果转换为复数张量
  return handle_r_to_c(other, std::move(result));
}

Tensor div_tensor_other_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other) {
  // 调用带有 rounding_mode 参数的 div_tensor_other_backward 函数
  return div_tensor_other_backward(grad, self, other, c10::nullopt);
}

Tensor permute_backwards(const Tensor& grad, IntArrayRef fwd_dims) {
  // 反转排列顺序
  auto ndims = fwd_dims.size();
  std::vector<int64_t> dims(ndims);
  for (const auto i : c10::irange(ndims)) {
    // 将 fwd_dims 中的维度反转到 dims 中
    dims[fwd_dims[i]] = i;
  }
    # 使用 at::maybe_wrap_dim 函数将 fwd_dims[i] 转换为有效的维度索引，并存入 dims 中
    dims[at::maybe_wrap_dim(fwd_dims[i], static_cast<int64_t>(ndims))] = 
        static_cast<int64_t>(i);
  }
  # 对梯度张量 grad 按照 dims 中指定的顺序进行维度重排
  return grad.permute(dims);
}

// 计算弧度到角度的反向传播
Tensor rad2deg_backward(const Tensor& grad) {
  // 定义常量 M_180_PI，表示 180/π 的近似值
  constexpr double M_180_PI =
      57.295779513082320876798154814105170332405472466564;
  // 将梯度 grad 乘以 M_180_PI 的标量值，返回结果
  return at::mul(grad, Scalar(M_180_PI));
}

// 计算角度到弧度的反向传播
Tensor deg2rad_backward(const Tensor& grad) {
  // 定义常量 M_PI_180，表示 π/180 的近似值
  constexpr double M_PI_180 =
      0.017453292519943295769236907684886127134428718885417;
  // 将梯度 grad 乘以 M_PI_180 的标量值，返回结果
  return at::mul(grad, Scalar(M_PI_180));
}

// 对张量进行多次展开操作
Tensor unsqueeze_multiple(
    const Tensor& t,
    OptionalIntArrayRef opt_dim,
    size_t n_dims) {
  if (opt_dim.has_value()) {
    IntArrayRef dim = opt_dim.value();
    auto dim_size = dim.size();
    // 优化两种常见情况
    if (dim_size == 0) {
      // 如果维度数组为空，则直接返回输入张量 t
      return t;
    } else if (dim_size == 1) {
      // 如果维度数组只有一个元素，则在该维度上对输入张量 t 进行展开操作
      return t.unsqueeze(dim[0]);
    }
  }
  // 将 opt_dim 转换为位集，指示哪些维度需要展开
  auto dims_to_unsqueeze = at::dim_list_to_bitset(opt_dim, n_dims);
  // 初始化结果张量为输入张量 t
  Tensor res = t;
  // 遍历所有可能的维度，如果标记为需要展开，则在该维度上对结果张量 res 进行展开操作
  for (const auto i : c10::irange(n_dims)) {
    if (dims_to_unsqueeze[i]) {
      res = res.unsqueeze(static_cast<int64_t>(i));
    }
  }
  // 返回展开后的结果张量 res
  return res;
}

// 对 sum 操作的反向传播
Tensor sum_backward(
    const Tensor& grad,
    c10::SymIntArrayRef sizes,
    OptionalIntArrayRef opt_dims,
    bool keepdim) {
  if (!keepdim && !sizes.empty()) {
    if (opt_dims.has_value() && !opt_dims.value().empty()) {
      // 对梯度 grad 在指定维度上进行展开，并根据 sizes 扩展张量的尺寸
      return unsqueeze_multiple(grad, opt_dims, sizes.size())
          .expand_symint(sizes);
    }
  }
  // 根据 sizes 扩展梯度 grad 的尺寸
  return grad.expand_symint(sizes);
}

// 对 sum 操作的反向传播（使用固定维度）
Tensor sum_backward(
    const Tensor& grad,
    c10::SymIntArrayRef sizes,
    c10::IntArrayRef dims,
    bool keepdim) {
  if (!keepdim && !sizes.empty() && !dims.empty()) {
    // 如果没有实现非常实施 `keepdim=true` 的 SymInt 支持路径
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "Only the keepdim=true path is implemented to support symints in autograd");
  } else {
    // 根据 sizes 扩展梯度 grad 的尺寸
    return grad.expand_symint(sizes);
  }
}

// 对 nansum 操作的反向传播
Tensor nansum_backward(
    const Tensor& grad,
    const Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim) {
  // 使用 sum_backward 函数计算 nansum 的反向传播，同时根据 self 的状态处理 NaN 值
  return sum_backward(grad, self.sym_sizes(), dims, keepdim) *
      self.isnan().logical_not();
}

// 对 mean 操作的反向传播
Tensor mean_backward(
    const Tensor& grad,
    c10::SymIntArrayRef shape,
    OptionalIntArrayRef opt_dim,
    c10::SymInt numel,
    bool keepdim) {
  // 判断是否进行全局求和操作
  bool is_all_reduce = !opt_dim.has_value() || opt_dim.value().empty();
  // 获取总数 n，如果不是全局操作，则根据 shape 和 opt_dim 计算安全大小
  auto n =
      is_all_reduce ? std::move(numel) : _safe_size(shape, opt_dim.value());
  // 使用 sum_backward 函数计算 mean 的反向传播，再除以总数 n 得到结果
  return sum_backward(grad, shape, opt_dim, keepdim) / std::move(n);
}

// 对 SymIntArrayRef 类型的列表进行反转
std::vector<c10::SymInt> reverse_list_symint(const c10::SymIntArrayRef list) {
  auto result = std::vector<c10::SymInt>();
  result.reserve(list.size());
  // 遍历列表的逆序，将元素逐个添加到结果向量中
  for (auto iter = list.rbegin(); iter != list.rend(); iter++) {
    result.push_back(*iter);
  }
  // 返回反转后的结果向量
  return result;
}

// 对 IntArrayRef 类型的列表进行反转
std::vector<int64_t> reverse_list(const IntArrayRef list) {
  auto result = std::vector<int64_t>();
  result.reserve(list.size());
  // 遍历列表的逆序，将元素逐个添加到结果向量中
  for (auto iter = list.rbegin(); iter != list.rend(); iter++) {
    result.push_back(*iter);
  }
  // 返回反转后的结果向量
  return result;
}

// 对使用零填充的安全乘积操作的反向传播
Tensor prod_safe_zeros_backward(
    const Tensor& grad,
    const Tensor& inp,
    // 检查输入张量是否为空（元素个数为0）
    if (inp.sym_numel() == 0) {
        // 当输入张量的指定维度大小为0时，不需要计算梯度。
        // 直接将 `grad` 张量 reshape 成与 `inp` 相同的形状并返回。
        return grad.expand_as(inp);
    }

    // 检查输入张量在指定维度上是否大小为1
    if (inp.sym_size(dim) == 1) {
        // 如果指定维度大小为1，直接返回 `grad` 张量。
        return grad;
    }

    // 创建一个大小与输入张量 `inp` 相同的向量，所有元素为1，除了在指定维度上为1
    auto ones_size = inp.sym_sizes().vec();
    ones_size[dim] = 1;
    Tensor ones = at::ones_symint(ones_size, grad.options());

    // 在指定维度上进行窄化操作，从0到`inp.sym_size(dim)-1`的部分
    Tensor narrow_inp = inp.narrow_symint(dim, 0, inp.sym_size(dim) - 1);

    // 创建包含首元素为1，其余元素为`narrow_inp`的张量
    Tensor exclusive_normal_nocp = at::cat({ones, narrow_inp}, dim);
    
    // 在指定维度上对累积乘积进行计算
    Tensor exclusive_normal = exclusive_normal_nocp.cumprod(dim);

    // 在指定维度上对反向窄化和翻转操作
    Tensor narrow_reverse = inp.narrow_symint(dim, 1, inp.sym_size(dim) - 1).flip(dim);

    // 创建包含首元素为1，其余元素为`narrow_reverse`的张量
    Tensor exclusive_reverse_nocp = at::cat({std::move(ones), std::move(narrow_reverse)}, dim);

    // 在指定维度上对累积乘积进行计算，并翻转张量
    Tensor exclusive_reverse = exclusive_reverse_nocp.cumprod(dim).flip(dim);

    // 返回 `grad` 与两个累积乘积张量的共轭乘积
    return grad * (exclusive_normal * exclusive_reverse).conj();
}
// 请注意，prod 的梯度等价于：
// cumprod(exclusive, normal) * cumprod(exclusive, reverse)，例如：
// 输入：                        [    a,     b,     c]
// cumprod(exclusive, normal)：  [1    ,     a, a * b]
// cumprod(exclusive, reverse)： [b * c,     c,     1]
// 乘积：                        [b * c, a * c, a * b]
// 并且这在包含 0 的输入下是安全的。
Tensor prod_backward(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& result) {
  if (input.dim() == 0) {
    return grad;
  }
  if (input.is_meta() || isTensorSubclassLike(input)) {
    // 对于 Composite Compliance，始终选择更安全（但更慢）的路径
    return prod_safe_zeros_backward(grad, input.contiguous().view(-1), 0)
        .view_as(input);
  }
  // 找出输入中值为 0 的索引位置
  Tensor zero_idx = (input == 0).nonzero();
  if (zero_idx.sym_numel() == 0) {
    // 如果没有零值，直接计算梯度
    return grad * (result / input).conj();
  } else if (!at::GradMode::is_enabled() && zero_idx.sym_size(0) > 1) {
    // 如果不是在梯度模式下，并且存在多个零值，返回与输入形状相同的零张量
    return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    // 存在零值时，采用安全的方法计算梯度
    return prod_safe_zeros_backward(grad, input.contiguous().view(-1), 0)
        .view_as(input);
  }
}

// prod_backward 函数的重载，处理具有指定维度的输入的情况
Tensor prod_backward(
    Tensor grad,
    const Tensor& input,
    Tensor result,
    int64_t dim,
    bool keepdim) {
  if (input.dim() == 0) {
    return grad;
  }
  // 确保维度 dim 在有效范围内
  dim = at::maybe_wrap_dim(dim, static_cast<int64_t>(input.sym_sizes().size()));
  if (!keepdim) {
    // `prod` 函数会在维度 dim 上进行减少，
    // 因此在该维度上对 grad 和 result 进行 unsqueeze
    grad = grad.unsqueeze(dim);
    result = result.unsqueeze(dim);
  }
  if (input.is_meta() || isTensorSubclassLike(input)) {
    // 对于 Composite Compliance，始终选择更安全（但更慢）的路径
    return prod_safe_zeros_backward(grad, input, dim);
  }

  // 找出输入中为 0 的位置
  Tensor zero_mask = (input == 0);
  Tensor slice_zero_count = zero_mask.sum(dim, true);
  int64_t total_zeros = slice_zero_count.sum().item<int64_t>();
  if (total_zeros == 0) {
    // 如果没有零值，直接计算梯度
    return grad * (result / input).conj();
  } else {
    // 存在零值时，采用安全的方法计算梯度
    return prod_safe_zeros_backward(grad, input, dim);
  }
}

// 使用泛型 solve 函数求解 JVP（Jacobian Vector Product）
template <typename solve_f>
static Tensor generic_solve_jvp(
    solve_f solve,
    const Tensor& X,
    const Tensor& A,
    const Tensor& dA,
    const Tensor& dB) {
  auto is_vector_case = at::native::linalg_solve_is_vector_rhs(dA, dB);
  auto dA_contrib =
      is_vector_case ? dA.matmul(X.unsqueeze(-1)).squeeze(-1) : dA.matmul(X);
  // 一般来说，
  // dX = solve(A, dB - dA_contrib)，但是对于 lu_solve，行为有所不同。
  // 请参阅 lu_solve_jvp 了解更多详细信息。
  return solve(A, dB, dA_contrib);
}

// cumsum 函数的反向传播
Tensor cumsum_backward(const Tensor& grad, int64_t dim) {
  // 简单情况
  if (grad.sym_numel() <= 1 || grad.sym_size(dim) == 1) {
    return grad;
  }
  // 反向传播的具体计算过程
  return grad.flip(dim).cumsum(dim).flip(dim);
}

// logsumexp 函数的反向传播
Tensor logsumexp_backward(
    Tensor grad,
    const Tensor& self,
    Tensor result,
    IntArrayRef dim,
    bool keepdim) {
  if (!keepdim && self.dim() != 0) {
    # 使用 unsqueeze_multiple 函数对 grad 和 result 进行维度扩展，使它们具有与 self.sym_sizes().size() 相同的维度
    grad = unsqueeze_multiple(grad, dim, self.sym_sizes().size());
    result = unsqueeze_multiple(result, dim, self.sym_sizes().size());
  }
  # 计算 self 与 result 的差值，并对每个元素应用指数函数 exp()
  return grad * (self - result).exp();
}

Tensor logcumsumexp_backward(
    Tensor grad,
    const Tensor& self,
    Tensor result,
    int64_t dim) {
  if (grad.dim() == 0 || grad.sym_numel() == 0) {
    return grad;
  }

  // 引用链接: https://github.com/tensorflow/tensorflow/blob/
  // 2a5910906a0e0f3dbc186ff9db6386d81a63448c/tensorflow/python/ops/math_grad.py#L1832-L1863

  auto scalar_min = AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
      at::ScalarType::BFloat16,
      at::typeMetaToScalarType(grad.dtype()),
      "logcumsumexp_backward",
      []() { return c10::Scalar(std::numeric_limits<scalar_t>::lowest()); });

  auto reverse_logcumsumexp = [dim](auto x) {
    return at::flip(at::logcumsumexp(at::flip(x, {dim}), dim), {dim});
  };

  if (!at::is_complex(grad)) {
    auto grad_min = at::scalar_tensor(scalar_min, grad.options());
    auto log_abs_grad = grad.abs().log();
    auto log_grad_positive = at::where(grad > 0, log_abs_grad, grad_min);
    auto log_grad_negative = at::where(grad < 0, log_abs_grad, grad_min);

    auto output_pos =
        (reverse_logcumsumexp(log_grad_positive - result) + self).exp();
    auto output_neg =
        (reverse_logcumsumexp(log_grad_negative - result) + self).exp();

    return output_pos - output_neg;
  } else {
    // 无需分开正负数的技巧
    auto log_grad = grad.conj().log();
    auto output = (reverse_logcumsumexp(log_grad - result) + self).exp();
    return output.conj();
  }
}

Tensor logcumsumexp_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    int64_t dim) {
  // 大部分取自 logsumexp_jvp

  // 注意: 为了简化，我们重新计算一些在前向传播中可以重复使用的值

  auto self_p_exp = [&self_p, dim]() {
    if (!at::is_complex(self_p)) {
      return (self_p - std::get<0>(at::max(self_p, dim, true)))
          .exp(); // 使用指数标准化技巧
    } else {
      // at::max 不支持 complex128
      return self_p.exp();
    }
  }();

  auto cumsumexp_p = self_p_exp.cumsum(dim);

  TORCH_INTERNAL_ASSERT(!self_t._is_zerotensor())

  constexpr double eps = 1e-13;

  if (areAnyTensorSubclassLike({self_p, self_t})) {
    auto result = (self_p_exp * self_t).cumsum(dim);
    result /= cumsumexp_p.add_(eps);
    return result;
  } else {
    self_p_exp *= self_t;
    auto cumsumexp_t = self_p_exp.cumsum(dim);
    return cumsumexp_t /= cumsumexp_p.add_(eps);
  }
}

Tensor unbind_backward(const variable_list& grads, int64_t dim) {
  c10::SymIntArrayRef sizes;
  at::TensorOptions o;
  for (const auto& v : grads) {
    if (v.defined()) {
      sizes = v.sym_sizes();
      o = static_cast<Tensor>(v).options();
      break;
    }
  }
  auto grads_tensors = fmap(grads, [&](const Variable& v) {
    return (
        v.defined() ? static_cast<Tensor>(v)
                    : at::zeros({}, o).expand_symint(sizes));
  });
  return at::stack(grads_tensors, dim);
}

Tensor unbind_backward_nested(
    const variable_list& grads,
    const Tensor& nt_sizes,
    int64_t dim,
    // 定义一个向量，用于存储梯度张量
    std::vector<Tensor> grads_tensors;
    
    // 遍历梯度张量列表
    for (int64_t i : c10::irange(static_cast<int64_t>(grads.size()))) {
        // 检查当前梯度张量是否已定义
        if (grads[i].defined()) {
            // 如果已定义，将其转换为Tensor类型并添加到grads_tensors向量中
            grads_tensors.push_back(static_cast<Tensor>(grads[i]));
        } else {
            // 如果未定义，则需要创建一个全零张量
            // 获取当前梯度张量应有的形状大小
            const auto component_size = nt_sizes[i].contiguous();
            // 从指针中获取形状大小并创建IntArrayRef
            const c10::IntArrayRef grad_size(
                component_size.data_ptr<int64_t>(), component_size.size(0));
            // 使用给定的选项创建全零张量，并添加到grads_tensors向量中
            grads_tensors.push_back(at::zeros(grad_size, options));
        }
    }

    // 调用_nested_tensor_from_tensor_list函数，将grads_tensors中的张量列表转换为嵌套张量
    return at::_nested_tensor_from_tensor_list(grads_tensors);
Tensor unbind_backward_nested_jagged(
    const variable_list& grads,  // 传入梯度列表，类型为 variable_list
    const Tensor& self,          // 传入自身张量 self
    int64_t dim) {               // 传入维度参数 dim

  TORCH_INTERNAL_ASSERT(
      dim == 0, "unbind_backward_nested_jagged() only supports dim=0")
  // 断言 dim 必须为 0，否则抛出错误信息

  auto grad_nt = at::zeros_like(self);
  // 创建一个与 self 相同大小的全零张量 grad_nt
  auto unbound_grads = grad_nt.unbind();
  // 对 grad_nt 进行解绑操作，返回解绑后的张量列表 unbound_grads

  for (int64_t i : c10::irange(static_cast<int64_t>(grads.size()))) {
    // 遍历 grads 的索引范围
    if (grads[i].defined()) {
      // 检查 grads[i] 是否已定义
      unbound_grads[i].copy_(static_cast<Tensor>(grads[i]));
      // 将 grads[i] 的值复制到 unbound_grads[i]
    }
  }

  return grad_nt;
  // 返回创建的全零张量 grad_nt
}

Tensor unsqueeze_to(const Tensor& self, c10::SymIntArrayRef sym_sizes) {
  auto result = self;
  // 将输入张量 self 赋值给 result

  auto nDims = sym_sizes.size();
  // 获取 sym_sizes 的维度数目 nDims

  for (const auto dim : c10::irange(nDims)) {
    // 遍历维度范围
    if (sym_sizes[dim] == 1) {
      result = result.unsqueeze(static_cast<int64_t>(dim));
      // 如果 sym_sizes[dim] 等于 1，则在 result 上执行 unsqueeze 操作
    }
  }

  return result;
  // 返回操作后的结果张量 result
}

Tensor unsqueeze_to(
    const Tensor& self,
    IntArrayRef dims,
    c10::SymIntArrayRef sym_sizes) {
  const auto ndim = sym_sizes.size();
  // 获取 sym_sizes 的维度数目 ndim
  auto mask = at::dim_list_to_bitset(dims, ndim);
  // 根据 dims 创建一个位掩码 mask，用于标记要进行 unsqueeze 的维度

  Tensor result = self;
  // 将输入张量 self 赋值给 result

  for (const auto d : c10::irange(ndim)) {
    // 遍历维度范围
    if (mask.test(d) && sym_sizes[d] == 1) {
      result = result.unsqueeze(static_cast<int64_t>(d));
      // 如果 mask 中对应位为 true，并且 sym_sizes[d] 等于 1，则在 result 上执行 unsqueeze 操作
    }
  }

  return result;
  // 返回操作后的结果张量 result
}

Tensor unsqueeze_to(
    const Tensor& self,
    int64_t dim,
    c10::SymIntArrayRef sym_sizes) {
  return unsqueeze_to(self, IntArrayRef{dim}, sym_sizes);
  // 调用前一个 unsqueeze_to 函数进行处理，将 dim 转换为 IntArrayRef 类型
}

std::vector<Tensor> cat_tensors_backward(
    const Tensor& grad,
    const std::vector<std::vector<c10::SymInt>>& sizes,
    const std::vector<ScalarType>& dtypes,
    int64_t dim) {
  std::vector<Tensor> grad_inputs(sizes.size());
  // 创建一个大小为 sizes.size() 的 Tensor 类型向量 grad_inputs

  if (!grad.defined()) {
    return grad_inputs;
    // 如果 grad 未定义，则直接返回 grad_inputs
  }

  dim = at::legacy_cat_wrap_dim_symint(dim, sizes);
  // 使用 at::legacy_cat_wrap_dim_symint 处理 dim，以支持符号整数

  c10::SymInt accumulate = 0;
  // 初始化一个累加器 accumulate，类型为符号整数

  Tensor grad_;
  bool grad_is_complex = grad.is_complex();
  if (grad_is_complex) {
    grad_ = at::real(grad);
    // 如果 grad 是复数类型，则取其实部赋值给 grad_
  }

  for (const auto i : c10::irange(sizes.size())) {
    // 遍历 sizes 的索引范围
    Tensor grad_val;
    if (!at::isComplexType(dtypes[i]) && grad_is_complex) {
      // 如果 dtypes[i] 不是复数类型，并且 grad 是复数类型
      grad_val = grad_;
    } else {
      grad_val = grad;
      // 否则直接使用 grad
    }

    auto& shape = sizes[i];
    // 获取 sizes[i] 中的形状信息

    if (shape.size() == 1) {
      // 如果 shape 的大小为 1
      if (TORCH_GUARD_SIZE_OBLIVIOUS(shape[0].sym_eq(0))) {
        // 使用 TORCH_GUARD_SIZE_OBLIVIOUS 检查 shape[0] 是否为 0
        grad_inputs[i] = at::zeros({0}, grad_val.options());
        // 如果是，则将 grad_inputs[i] 初始化为大小为 0 的全零张量
        continue;
      }
    }

    const auto& size = shape[dim];
    // 获取 shape 在 dim 维度上的大小信息
    accumulate += size;
    // 累加 size 到 accumulate 中

    grad_inputs[i] = grad_val.narrow_symint(dim, accumulate - size, size);
    // 使用 narrow_symint 在 dim 维度上对 grad_val 进行裁剪，并将结果赋给 grad_inputs[i]
  }

  return grad_inputs;
  // 返回处理后的梯度输入张量向量 grad_inputs
}

std::vector<Tensor> stack_tensors_backward(
    const Tensor& grad,
    int64_t dim,
    const std::vector<ScalarType>& dtypes) {
  std::vector<Tensor> grad_inputs(dtypes.size());
  // 创建一个大小为 dtypes.size() 的 Tensor 类型向量 grad_inputs

  if (!grad.defined()) {
    return grad_inputs;
    // 如果 grad 未定义，则直接返回 grad_inputs
  }

  bool grad_is_complex = grad.is_complex();
  // 检查 grad 是否为复数类型

  for (const auto i : c10::irange(dtypes.size())) {
    // 遍历 dtypes 的索引范围
    auto gr = grad.select(dim, static_cast<int64_t>(i));
    // 从 grad 中选择第 dim 维的第 i 个张量 gr

    if (grad_is_complex && !at::isComplexType(dtypes[i])) {
      gr = at::real(gr);
      // 如果 grad 是复数类型，并且 dtypes[i] 不是复数类型，则取 gr 的实部赋值给 gr
    }

    grad_inputs[i] = gr;
    // 将处理后的张量 gr 存储到 grad_inputs[i] 中
  }

  return grad_inputs;
  // 返回处理后的梯度输入张量向量 grad_inputs
}
    # 将 gr 赋值给 grad_inputs 列表的第 i 个位置
    grad_inputs[i] = gr;
    # 返回填充了梯度输入数据的 grad_inputs 列表
    return grad_inputs;
}

std::vector<Tensor> block_diag_backward(
    const Tensor& grad,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<ScalarType>& dtypes) {
  // 初始化梯度输入向量，大小与sizes相同
  std::vector<Tensor> grad_inputs(sizes.size());
  // 如果梯度grad未定义，则返回空的grad_inputs向量
  if (!grad.defined()) {
    return grad_inputs;
  }
  // 定义实数视图的梯度
  Tensor real_view_of_grad;
  // 检查梯度是否为复数类型
  bool grad_is_complex = grad.is_complex();
  if (grad_is_complex) {
    // 获取梯度的实部视图
    real_view_of_grad = at::real(grad);
  }

  // 当前维度的起始位置
  int64_t cur_dim0 = 0;
  int64_t cur_dim1 = 0;

  // 遍历sizes中的每一个索引
  for (const auto i : c10::irange(sizes.size())) {
    // R -> C
    // 根据条件选择梯度值
    Tensor grad_val = (!at::isComplexType(dtypes[i]) && grad_is_complex)
        ? real_view_of_grad
        : grad;

    auto& shape = sizes[i];
    // 如果输入是空张量，则gradInput也应该是空张量
    if (shape.size() == 1 && shape[0] == 0) {
      grad_inputs[i] = at::zeros({0}, grad_val.options());
      continue;
    }
    // 处理0维情况
    int64_t dim0 = 1;
    int64_t dim1 = 1;
    // 处理2维情况
    if (shape.size() == 2) {
      dim0 = shape[0];
      dim1 = shape[1];
    // 处理1维情况
    } else if (shape.size() == 1) {
      dim1 = shape[0];
    }
    // 切片操作，获取当前分块的梯度
    auto slice = grad_val.slice(0, cur_dim0, cur_dim0 + dim0)
                     .slice(1, cur_dim1, cur_dim1 + dim1);
    // 如果是1维情况，压缩掉最后一个维度
    if (shape.size() == 1) {
      slice = slice.squeeze(-1);
    // 如果shape为空，压缩掉两个最后维度
    } else if (shape.empty()) {
      slice = slice.squeeze(-1).squeeze(-1);
    }
    // 将切片结果存入grad_inputs中
    grad_inputs[i] = slice;
    // 更新当前维度的起始位置
    cur_dim0 += dim0;
    cur_dim1 += dim1;
  }
  // 返回结果向量
  return grad_inputs;
}

Tensor clamp_backward(
    const Tensor& grad,
    const Tensor& self,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  // clamp操作的反向传播：在min和max未定义的情况下，返回子梯度1
  if (max && min) {
    auto zero = at::scalar_tensor(0., grad.options());
    return where((self >= *min).logical_and_(self <= *max), grad, zero);
  } else if (min) {
    auto zero = at::scalar_tensor(0., grad.options());
    return where(self >= *min, grad, zero);
  } else if (max) {
    auto zero = at::scalar_tensor(0., grad.options());
    return where(self <= *max, grad, zero);
  } else {
    // min和max均未定义时，直接返回梯度grad
    return grad;
  }
}

Tensor clamp_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& min,
    const Tensor& max) {
  // clamp操作的反向传播：在min和max未定义的情况下，返回子梯度1
  if (max.defined() && min.defined()) {
    auto zero = at::scalar_tensor(0., grad.options());
    const auto self_ge_min = self >= min;
    const auto self_le_max = self <= max;
    const auto& pred = areAnyTensorSubclassLike({self, min, max})
        ? self_ge_min.logical_and(self_le_max)
        : self_ge_min.logical_and_(self_le_max);
    return where(pred, grad, zero);
  } else if (min.defined()) {
    auto zero = at::scalar_tensor(0., grad.options());
    return where(self >= min, grad, zero);
  } else if (max.defined()) {
    auto zero = at::scalar_tensor(0., grad.options());
    return where(self <= max, grad, zero);
  } else {
    // min和max均未定义时，直接返回梯度grad
    return grad;
  }
}
    return grad;
  }


# 返回当前函数中定义的变量 `grad` 的值作为函数的返回值
return grad;
# 结束当前函数的定义
}
}

std::tuple<at::Tensor, at::Tensor> clamp_backward_min_max(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& min,
    const Tensor& max,
    const std::array<bool, 2>& grad_input_mask) {
  // 如果 grad 未定义，则返回空元组
  std::tuple<at::Tensor, at::Tensor> ret;
  if (!grad.defined()) {
    return ret;
  }

  // 创建一个标量张量 zero，用于填充条件不满足时的梯度值
  auto zero = at::scalar_tensor(0., grad.options());
  // 当 min 和 max 均已定义时执行以下代码块
  if (max.defined() && min.defined()) {
    // 如果 grad_input_mask 的第一个元素为 true
    if (grad_input_mask[0]) {
      // 创建布尔张量 self_lt_min，表示 self 是否小于 min
      const auto self_lt_min = self < min;
      // 创建布尔张量 min_lt_max，表示 min 是否小于 max
      const auto min_lt_max = min < max;
      // 根据张量类型创建逻辑表达式 pred，用于选择是否使用 in-place 操作
      const auto& pred = areAnyTensorSubclassLike({self, min, max})
          ? self_lt_min.logical_and(min_lt_max)
          : self_lt_min.logical_and_(min_lt_max);
      // 将满足条件的梯度值填充到返回的元组的第一个位置
      std::get<0>(ret) = where(pred, grad, zero);
    }
    // 如果 grad_input_mask 的第二个元素为 true
    if (grad_input_mask[1]) {
      // 创建布尔张量 self_gt_max，表示 self 是否大于 max
      const auto self_gt_max = self > max;
      // 创建布尔张量 max_lt_min，表示 max 是否小于 min
      const auto max_lt_min = max < min;
      // 根据张量类型创建逻辑表达式 pred，用于选择是否使用 in-place 操作
      const auto& pred = areAnyTensorSubclassLike({self, min, max})
          ? self_gt_max.logical_or(max_lt_min)
          : self_gt_max.logical_or_(max_lt_min);
      // 将满足条件的梯度值填充到返回的元组的第二个位置
      std::get<1>(ret) = where(pred, grad, zero);
    }
  } else if (min.defined() && grad_input_mask[0]) {
    // 当只有 min 已定义且 grad_input_mask 的第一个元素为 true 时
    std::get<0>(ret) = where(self < min, grad, zero);
  } else if (max.defined() && grad_input_mask[1]) {
    // 当只有 max 已定义且 grad_input_mask 的第二个元素为 true 时
    std::get<1>(ret) = where(self > max, grad, zero);
  }
  // 返回包含梯度的元组
  return ret;
}

// 实现 clamp 函数的 JVP（Jacobean Vector Product）版本
at::Tensor clamp_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const Tensor& min_p,
    const Tensor& min_t,
    const Tensor& max_p,
    const Tensor& max_t) {
  // 当 min_p 和 max_p 均已定义时执行以下代码块
  if (min_p.defined() && max_p.defined()) {
    // 如果 min_p 大于 max_p，则返回 max_t
    return where(
        min_p > max_p,
        max_t,
        // 否则根据 self_p 和 min_p、max_p 的关系进行选择
        where(self_p < min_p, min_t, where(self_p > max_p, max_t, self_t)));
  } else if (min_p.defined()) {
    // 当仅 min_p 已定义时，根据 self_p 和 min_p 的关系进行选择
    return where(self_p > min_p, self_t, min_t);
  } else if (max_p.defined()) {
    // 当仅 max_p 已定义时，根据 self_p 和 max_p 的关系进行选择
    return where(self_p < max_p, self_t, max_t);
  } else {
    // 如果 min_p 和 max_p 均未定义，则返回 self_t
    return self_t;
  }
}

// 实现卷积操作的 JVP（Jacobean Vector Product）版本
Tensor convolution_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    at::SymIntArrayRef stride,
    at::SymIntArrayRef padding,
    at::SymIntArrayRef dilation,
    bool transposed,
    at::SymIntArrayRef output_padding,
    const c10::SymInt& groups) {
  // 根据是否存在偏置张量 bias_t 来选择是否传递 std::optional
  auto bias_t_opt =
      bias_t.defined() ? std::optional<at::Tensor>(bias_t) : c10::nullopt;
  // 返回输入张量与权重张量的 JVP 结果的和
  return (
      at::convolution_symint(
          input_t,
          weight_p,
          c10::nullopt,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups) +
      at::convolution_symint(
          input_p,
          weight_t,
          bias_t_opt,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups));
}

// 实现带偏置的卷积操作的 JVP（Jacobean Vector Product）版本
Tensor _convolution_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    // 根据给定的参数执行符号整数（SymInt）卷积操作，返回两个操作的结果之和
    auto bias_t_opt =
        // 如果给定的偏置张量 bias_t 已定义，则创建一个包含该张量的 std::optional 对象；否则为 c10::nullopt
        bias_t.defined() ? std::optional<at::Tensor>(bias_t) : c10::nullopt;
    return (
        // 执行第一个符号整数卷积操作，传入输入张量 input_t、权重参数 weight_p、空的偏置张量、步长 stride、填充 padding、
        // 扩展 dilation、是否转置 transposed、输出填充 output_padding、分组数 groups、是否基准测试 benchmark、
        // 是否确定性 deterministic、是否启用 cuDNN cudnn_enabled、是否允许 TF32 加速 allow_tf32
        at::_convolution_symint(
            input_t,
            weight_p,
            c10::nullopt,  // 由于没有提供偏置张量，传入空的 std::optional 对象
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            benchmark,
            deterministic,
            cudnn_enabled,
            allow_tf32) +
        // 执行第二个符号整数卷积操作，传入输入张量 input_p、权重参数 weight_t、可能包含偏置张量的 std::optional 对象 bias_t_opt（如有），
        // 此处使用与第一个卷积操作相同的参数
        at::_convolution_symint(
            input_p,
            weight_t,
            bias_t_opt,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            benchmark,
            deterministic,
            cudnn_enabled,
            allow_tf32));
}



Tensor convolution_backward_jvp_grad_bias(
    const Tensor& grad_out_t,                     // 输入参数：梯度张量 grad_out_t
    const Tensor& grad_bias) {                    // 输入参数：梯度偏置 grad_bias
  if (!grad_bias.defined()) {                     // 检查 grad_bias 是否已定义
    return Tensor();                             // 如果未定义，返回空张量
  }
  int64_t dim = grad_out_t.dim() - 2;             // 计算 grad_out_t 的维度减去2
  if (dim == 1) {                                 // 如果 dim 等于1
    // 由于重载歧义无法传递初始化列表
    auto dimlist = std::vector<int64_t>{0, 2};    // 创建维度列表 {0, 2}
    return grad_out_t.sum(dimlist);               // 返回在指定维度上的张量求和
  } else if (dim == 2) {                          // 如果 dim 等于2
    return grad_out_t.sum({0, 2, 3});             // 返回在指定维度上的张量求和
  } else if (dim == 3) {                          // 如果 dim 等于3
    return grad_out_t.sum({0, 2, 3, 4});          // 返回在指定维度上的张量求和
  } else {                                        // 否则
    TORCH_INTERNAL_ASSERT(                        // 内部断言，如果条件不满足，则抛出异常
        false,
        "convolution_backward_jvp_grad_bias expected dim of grad_out_t to be 3, 4, or 5, but got: ",
        grad_out_t.dim());                        // 异常信息：grad_out_t 的维度不是3、4或5
  }
}



// This function is used by load_derivatives.py to replace tensor.strides()
// calls that appear in derivative formulas. If the tensor has requires_grad
// set, this function returns its strides or an empty array if the tensor
// is sparse. If requires_grad is not set, an empty array is returned since
// there will be no backward pass. There has one special case, if input is
// MKLDNN tensor and has requires_grad set, just return an empty array, the
// reason is that MKLDNN tensor is a opaque tensor which has not stride info.
//
// This function only supports the case where `input` is the tensor whose
// single derivative is being calculated.
//
// This function does not support `self` derivatives for inplace functions.
//
// Args:
//  input              Tensor to call .strides() on     // 输入参数：需要调用 .strides() 的张量
//  input_name         Name of `input` tensor, from derivative formula   // 输入参数：`input` 张量的名称，来自导数公式
at::SymIntArrayRef strides_or_error(
    const Tensor& input,                            // 输入参数：输入张量
    c10::string_view const& input_name) {           // 输入参数：输入张量的名称
  // TODO: Ideally, this function would never be called if requires_grad is
  // not set. Once codegen is updated to avoid the call, we can remove this
  // check.
  if (input.requires_grad()) {                      // 检查输入张量是否需要梯度
    if (input.is_mkldnn())                          // 如果输入张量是 MKLDNN 张量
      return {};                                    // 返回空数组，因为 MKLDNN 张量没有步长信息
    if (input.is_sparse() || at::sparse_csr::is_sparse_compressed(input))
      return {};                                    // 如果输入张量是稀疏张量或稀疏 CSR 压缩张量，返回空数组
    return input.sym_strides();                     // 返回输入张量的符号步长
  } else {
    return {};                                      // 如果输入张量不需要梯度，返回空数组
  }
}



Tensor mm_mat1_backward(
    const Tensor& grad,                            // 输入参数：梯度张量 grad
    const Tensor& mat2,                            // 输入参数：矩阵张量 mat2
    at::SymIntArrayRef mat1_sizes,                 // 输入参数：矩阵 mat1 的大小
    at::SymIntArrayRef mat1_strides,               // 输入参数：矩阵 mat1 的步长
    c10::Layout mat1_layout,                       // 输入参数：矩阵 mat1 的布局
    const Scalar& alpha) {                         // 输入参数：标量 alpha
  if (grad.layout() == c10::kStrided &&            // 检查梯度张量的布局是否为 strided
      mat2.layout() == c10::kStrided &&            // 检查 mat2 的布局是否为 strided
      mat1_layout == c10::kStrided) {              // 检查 mat1 的布局是否为 strided
    // if input was column-major, return grad as column-order for efficiency
    if (mat1_strides[0] == 1 && mat1_strides[1] == mat1_sizes[0]) {
      return maybe_multiply(mat2.conj().mm(grad.t()).t(), alpha.conj());  // 如果 mat1 是列主序，返回经优化的列序梯度
    }
  }

  // General fallback, should work for any layout
  return maybe_multiply(grad.mm(mat2.t().conj()), alpha.conj());  // 一般情况下的后备方案，适用于任何布局
}

Tensor mm_mat2_backward(
    const Tensor& grad,                            // 输入参数：梯度张量 grad
    const Tensor& mat1,                            // 输入参数：矩阵张量 mat1
    at::SymIntArrayRef mat2_sizes,                 // 输入参数：矩阵 mat2 的大小
    at::SymIntArrayRef mat2_strides,               // 输入参数：矩阵 mat2 的步长
    c10::Layout mat2_layout,                       // 输入参数：矩阵 mat2 的布局
    const Scalar& alpha) {                         // 输入参数：标量 alpha
    // 如果梯度(grad)、mat1和mat2都是连续内存布局（strided layout）
    if (grad.layout() == c10::kStrided && mat1.layout() == c10::kStrided &&
        mat2_layout == c10::kStrided) {
        // 如果输入是列主序（column-major），为了效率将梯度grad转换为列顺序
        if (mat2_strides[0] == 1 && mat2_strides[1] == mat2_sizes[0]) {
            // 返回转置后的梯度grad的乘积与mat1的共轭转置矩阵的乘积，乘以alpha的共轭
            return maybe_multiply(grad.t().mm(mat1.conj()).t(), alpha.conj());
        }
    }

    // 一般性回退方案，适用于任何布局情况
    // 返回mat1的共轭转置矩阵与梯度grad的乘积，乘以alpha的共轭
    return maybe_multiply(mat1.t().conj().mm(grad), alpha.conj());
static Tensor sparse_mask_like_grad(
    const Tensor& x,
    const Tensor& gx,
    bool accumulate_matches) {
  // 检查输入张量 x 和 gx 是否都是稀疏且已压缩
  if (x.is_coalesced() && gx.is_coalesced()) {
    // 如果 x 的非零元素个数大于等于 gx 的非零元素个数，优先在 x 中搜索
    if (x._nnz() >= gx._nnz()) {
      // 返回 gx 在 x 上的稀疏掩码投影
      return gx._sparse_mask_projection(x, accumulate_matches);
    } else {
      // 返回 gx 对 x 的稀疏掩码
      return gx.sparse_mask(x);
    }
  } else if (x.is_coalesced()) {
    // 如果只有 x 是稀疏且已压缩的，返回 gx 对 x 的稀疏掩码
    return gx.sparse_mask(x);
  } else if (gx.is_coalesced()) {
    // 如果只有 gx 是稀疏且已压缩的，返回 gx 在 x 上的稀疏掩码投影
    return gx._sparse_mask_projection(x, accumulate_matches);
  } else {
    // 如果 x 和 gx 都不是稀疏且已压缩的，根据非零元素个数选择更高效的方法
    if (x._nnz() >= gx._nnz()) {
      // 返回 gx 稀疏化后在 x 上的稀疏掩码投影
      return gx.coalesce()._sparse_mask_projection(x, accumulate_matches);
    } else {
      // 返回 gx 对 x 稀疏化后的稀疏掩码
      return gx.sparse_mask(x.coalesce());
    }
  }
}

std::tuple<Tensor, Tensor, Tensor> sparse_sampled_addmm_backward(
    const Tensor& grad,
    const Tensor& self,
    const std::optional<Tensor>& mat1,
    const std::optional<Tensor>& mat2,
    const Scalar& alpha,
    const Scalar& beta,
    const std::array<bool, 3>& grad_input_mask) {
  // 如果梯度 grad 未定义，返回空张量元组
  if (!grad.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }

  // 对梯度 grad 在 self 上进行稀疏掩码处理
  const auto grad_projected = grad.sparse_mask(self);
  const auto self_requires_grad = grad_input_mask[0];
  const auto mat1_requires_grad = grad_input_mask[1];
  const auto mat2_requires_grad = grad_input_mask[2];
  return std::make_tuple(
      // 如果 self 需要梯度，返回 grad 乘以 beta 的共轭
      self_requires_grad ? maybe_multiply(grad, beta.conj()) : Tensor{},
      // 如果 mat1 需要梯度，返回 grad 在 self 上的投影与 mat2 的共轭转置矩阵相乘再乘以 alpha 的共轭
      mat1_requires_grad ? maybe_multiply(grad_projected.mm(mat2->mH()), alpha.conj()) : Tensor{},
      // 如果 mat2 需要梯度，返回 mat1 的共轭转置矩阵与 grad 在 self 上的投影相乘再乘以 alpha 的共轭
      mat2_requires_grad ? maybe_multiply(mat1->mH().mm(grad_projected), alpha.conj()) : Tensor{});
}

Tensor sparse_mask_backward(
    // 定义函数，根据梯度、掩码和张量布局进行反向传播计算
    const Tensor& grad,
    const Tensor& mask,
    const c10::Layout self_layout) {
      // 注意：sparse_mask accumulates matches，因此反向步骤也必须累积。
      // 调用函数 sparse_mask_like_grad，生成带有掩码的梯度 self_grad，并指定累积匹配为 true
      const auto self_grad =
          sparse_mask_like_grad(mask, grad, /*accumulate_matches=*/true);
      // 根据 self_layout 的值判断布局类型是否为 kStrided，如果是则将 self_grad 转换为稠密张量，否则保持 self_grad 不变
      return self_layout == at::kStrided ? self_grad.to_dense() : self_grad;
    }
}

Tensor sparse_sparse_matmul_backward(
    const Tensor& grad,
    const Tensor& a,
    const Tensor& b,
    int64_t grad_order) {
  /*
  To implement the backward algorithm for sparse matrix-matrix matmul (SPMM) we
  can start from the following definition for dense tensors:

  c = a @ b
      then
  a_grad = c_grad @ b^H
  b_grad = a^H @ c_grad

  So for sparse matrices we can use the following definition:

  if grad_order == 0:
      a_grad = sparse_matrix_mask(c_grad @ b^H, mask=a)
  else:
      b_grad = sparse_matrix_mask(a^H @ c_grad, mask=b)
  */
  // 检查 grad_order 是否为 0 或 1，不在这个范围内则报错
  TORCH_CHECK(
      grad_order == 0 || grad_order == 1,
      ": grad_order not in [0, 1] at sparse_sparse_matmul_backward function");

  // 注意：_sparse_sparse_matmul 返回一个紧凑的梯度，
  // 因此不需要累积匹配。
  if (grad_order == 0) {
    // 计算 a_grad = _sparse_sparse_matmul(grad, b.conj().t())
    auto a_grad = _sparse_sparse_matmul(grad, b.conj().t());
    // 返回 a_grad 关于 a 的梯度，并应用稀疏矩阵的掩码
    return sparse_mask_like_grad(a, a_grad, /*accumulate_matches=*/false);
  }
  // 计算 b_grad = _sparse_sparse_matmul(a.conj().t(), grad)
  auto b_grad = _sparse_sparse_matmul(a.conj().t(), grad);
  // 返回 b_grad 关于 b 的梯度，并应用稀疏矩阵的掩码
  return sparse_mask_like_grad(b, b_grad, /*accumulate_matches=*/false);
}

Tensor renorm_backward(
    const Tensor& grad,
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm) {
  auto n = self.dim();
  // 包装维度以确保在有效范围内
  dim = c10::maybe_wrap_dim(dim, n);
  // 创建一个包含所有维度的向量，并从中删除指定的维度
  auto reduce_dims = at::DimVector(n);
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  // 确定积累类型
  auto acc_type =
      at::toAccumulateType(self.scalar_type(), self.device().type());
  // 计算向量的 p 范数
  auto norm = at::linalg_vector_norm(
      self, p, reduce_dims, /*keepdim=*/true, /*dtype=*/acc_type);

  // 获取实际的积累类型
  const auto real_acc_type = c10::toRealValueType(acc_type);
  // 计算梯度的输出，因为 vector_norm 输出为实数，所以 grad_output 也应为实数
  auto grad_output = (self.conj() * grad);
  if (real_acc_type != acc_type) {
    // 如果实际的积累类型与预期类型不同，则获取其实部
    grad_output = at::real(grad_output);
  }
  // 沿指定维度求和 grad_output
  grad_output =
      grad_output.sum(reduce_dims, /*keepdim=*/true, /*dtype=*/real_acc_type);
  // 计算 norm 的反向传播
  auto nb = norm_backward(
      std::move(grad_output), self, p, norm, reduce_dims, /*keepdim=*/true);

  // 计算 invnorm 和 grad_norm
  auto invnorm = (norm + 1e-7).reciprocal();
  auto grad_norm = maxnorm * invnorm * (grad - invnorm * nb);
  // 根据条件应用 grad_norm 到 grad
  return at::where(norm > maxnorm, grad_norm.to(grad.scalar_type()), grad);
}

Tensor renorm_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm) {
  auto self_sizes = self_p.sizes();
  // 包装维度以确保在有效范围内
  dim = at::maybe_wrap_dim(dim, static_cast<int64_t>(self_sizes.size()));

  // 创建一个包含所有维度的向量，并从中删除指定的维度
  at::DimVector reduce_dims(self_sizes.size());
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  // 对于 cuda half，以 float 精度计算范数，然后将规范化因子转换为 half 类型
  auto dtype = self_p.scalar_type();
  auto acc_type = at::toAccumulateType(dtype, /*is_cuda=*/true);
  // 计算向量的 p 范数
  Tensor norm = [&self_p, &p, &reduce_dims, acc_type, dtype]() {
    // 如果累积类型不等于数据类型
    if (acc_type != dtype) {
      // 返回张量 self_p 的 p 范数，指定降维维度 reduce_dims，保持维度 keepdim=true，使用累积类型 acc_type
      return at::linalg_vector_norm(
          self_p,
          p.toDouble(),
          reduce_dims,
          /*keepdim=*/true,
          /*dtype=*/acc_type);
    } else {
      // 返回张量 self_p 的 p 范数，指定降维维度 reduce_dims，保持维度 keepdim=true，不指定累积类型（默认为数据类型 dtype）
      return at::linalg_vector_norm(
          self_p,
          p.toDouble(),
          reduce_dims,
          /*keepdim=*/true);
    }
  }();

  // 将 maxnorm 转换为双精度浮点数
  auto double_maxnorm = maxnorm.toDouble();
  // 计算 norm + 1e-7 的倒数
  auto invnorm = (norm + 1e-7).reciprocal();
  // 计算因子 factor，为倒数乘以 double_maxnorm
  auto factor = invnorm * double_maxnorm;

  // 返回条件选择：如果 norm 大于 double_maxnorm，则执行计算；否则返回 self_t
  return where(
      norm > double_maxnorm,
      // 计算修正后的 self_t
      factor *
          (self_t -
           self_p * invnorm *
               // 计算 norm_jvp 的结果，传入参数为 self_p, self_t, p, norm, reduce_dims，保持维度 keepdim=true
               norm_jvp(
                   self_p, self_t, p, norm, reduce_dims, /*keepdim=*/true)),
      // 如果 norm 不大于 double_maxnorm，则返回原始 self_t
      self_t);
// 定义函数 `repeat_backward`，接受梯度 `grad`、重复次数 `repeats` 和输入形状 `input_shape` 作为参数，并返回一个张量
Tensor repeat_backward(
    Tensor grad,
    c10::SymIntArrayRef repeats,
    c10::SymIntArrayRef input_shape) {
  // 查找重复次数中是否存在0，如果存在则返回一个与输入形状相同的零张量
  auto find_iter = std::find(repeats.cbegin(), repeats.cend(), 0);
  if (find_iter != repeats.cend()) {
    return at::zeros_symint(input_shape, grad.options());
  }
  // 计算输入形状的维度数量
  const auto input_dims = input_shape.size();
  // 计算未被挤压的维度数目
  auto num_unsqueezed = grad.dim() - input_dims;
  // 对于未被挤压的每一个维度，沿着该维度对梯度进行求和
  for (const auto i : c10::irange(num_unsqueezed)) {
    (void)i; // 抑制未使用变量警告
    grad = grad.sum(0, false);
  }

  // 初始化梯度大小向量和求和维度向量
  at::SymDimVector grad_size;
  at::DimVector sum_dims;
  // 对于每一个输入维度
  for (const auto dim : c10::irange(input_dims)) {
    // 获取重复次数
    const auto& repeat = repeats[dim + num_unsqueezed];
    // 重塑梯度（当 repeat > 1）
    // 索引：      [..., dim    , ...]    [..., dim   ,  dim+1        , ...]
    // 形状：从 [..., dimsize, ...] 到 [..., repeat, dimsize/repeat, ...]
    // 在维度 `dim` 处，梯度张量被重塑为输入张量的 `repeat` 倍。然后，沿着重复的张量在 `dim` 维度上对梯度进行求和，并将形状从 `repeat * dimsize/repeat` 减少到 `dimsize/repeat`（`input_dimsize`）。例如：
    //        大小(3, 2)                                      大小(6, 2)
    //                                                      [[v1_0, v1_1],
    //                                                       [v1_2, v1_3],
    //        [[v0, v1],               重复(2, 1)          [v1_4, v1_5],
    //         [v2, v3],              ------------->         [v2_0, v2_1],
    //         [v4, v5]]                                     [v2_2, v2_3],
    //                                                       [v2_4, v2_5]]
    //
    //    输入梯度 (3, 2)      重塑 (2, 3, 2)         输出梯度 (6, 2)
    //                            [[[g1_0, g1_1],            [[g1_0, g1_1],
    //                              [g1_2, g1_3],             [g1_2, g1_3],
    // [[g1_0+g2_0, g1_1+g2_1],     [g1_4, g1_5]],            [g1_4, g1_5],
    //  [g1_2+g2_2, g1_3+g2_3],     [g2_0, g2_1],            [[g2_0, g2_1],
    //  [g1_4+g2_4, g1_5+g2_5]]     [g2_2, g2_3],             [g2_2, g2_3],
    //                              [g2_4, g2_5]]             [g2_4, g2_5]]]
    //
    // 如果梯度张量被重塑为 [..., dimsize/repeat, repeat, ...] 然后沿着 `dim+1` 求和。
    // 那么输入的梯度与输入不能正确对齐。例如：
    //  输入梯度 (3, 2)        重塑 (3, 2, 2)        输出梯度 (6, 2)
    //                           [[[g1_0, g1_1],           [[g1_0, g1_1],
    //                             [g1_2, g1_3]],           [g1_2, g1_3],
    // [[g1_0+g1_2, g1_1+g1_3],   [[g1_4, g1_5],            [g1_4, g1_5],
    //  [g1_4+g2_0, g1_5+g2_1],    [g2_0, g2_1]],           [g2_0, g2_1],
    //  [g2_2+g2_4, g2_3+g2_5]]   [[g2_2, g2_3],            [g2_2, g2_3],
    //                             [g2_4, g2_5]]]           [g2_4, g2_5]]
    // 如果 repeat 不等于 1，则将 repeat 添加到 grad_size 向量中，并将当前维度索引添加到 sum_dims 向量中
    if (repeat != 1) {
      grad_size.push_back(repeat);
      sum_dims.push_back(static_cast<int64_t>(grad_size.size() - 1));
    }
    // 当 repeat 等于 1 时，不需要将梯度重塑为 (repeat, input_shape[dim]) 的形式

    // 将当前维度的 input_shape[dim] 添加到 grad_size 向量中
    grad_size.push_back(input_shape[dim]);
  }
  // 一次性的重塑和求和操作
  // 将梯度重塑为 grad_size：
  //   1. 如果 repeat 等于 1，则在该维度上追加输入大小，
  //   2. 如果 repeat 大于 1，则在该维度上同时追加 repeat 和输入大小。
  // 在 sum_dims 指定的所有 "repeat" 维度上进行求和：
  // 示例：
  // 输入大小         (2,    3,    4,    5)
  // repeat           [4,    1,    9,    3]
  // 输出/梯度大小   (8,    3,    36,   15)
  // grad_size        [4, 2,    3, 9, 4, 3, 5]
  // sum_dims         [0,          3,    5]

  // 当在所有原始维度上重复 1 次时，空的 sum_dims 将会将整个梯度张量减少为一个标量，而不是保持原始维度。
  if (!sum_dims.empty()) {
    // 对梯度 grad 进行符号整数重塑为 grad_size
    grad = grad.reshape_symint(grad_size);
    // 对 grad 在 sum_dims 维度上进行求和
    grad = grad.sum(sum_dims);
  }
  // 返回处理后的梯度 grad
  return grad;
// 结束函数 _fused_dropout_backward
Tensor _fused_dropout_backward(
    const Tensor& grad,    // 梯度张量
    const Tensor& mask,    // 掩码张量
    double p1m) {          // 计算 p1m = 1 - p 的结果
  if (grad.requires_grad()) {  // 如果梯度需要计算梯度信息
    // 使用自动求导友好的反向传播方式（如果需要进行双向传播）
    return grad * (mask.type_as(grad) * (1. / p1m));
  } else {
    // 使用 PyTorch 的 _masked_scale 函数进行缩放
    return at::_masked_scale(grad, mask, 1. / p1m);
  }
}

// 计算比例 scale = (1 / (1 - prob))
Tensor infinitely_differentiable_native_dropout_backward(
    const Tensor& grad,    // 梯度张量
    const Tensor& mask,    // 掩码张量
    double scale) {        // 比例尺度
  // 返回按比例缩放的梯度
  return grad * (mask.type_as(grad) * scale);
}

// 双向传播函数 native_dropout_double_backward
Tensor native_dropout_double_backward(
    const Tensor& ggI,     // 梯度乘积
    const Tensor& grad,    // 梯度张量
    const Tensor& mask,    // 掩码张量
    double scale) {        // 比例尺度
  // 返回乘积的缩放梯度
  return ggI.type_as(grad) * (mask.type_as(grad) * scale);
}

// 均匀分布反向传播函数
Tensor evenly_distribute_backward(
    const Tensor& grad,    // 梯度张量
    const Tensor& input,   // 输入张量
    const Tensor& value) { // 值张量
  // 检查是否有任何张量子类或输入张量是否在 CUDA 上
  bool any_tensor_subclass_like =
      areAnyTensorSubclassLike({grad, input, value});
  if (any_tensor_subclass_like || input.is_cuda()) {
    // 计算是否输入和值存在 NaN，创建逻辑掩码
    const auto input_isnan = input.isnan();
    const auto value_isnan = value.isnan();
    const auto& input_and_value_isnan = any_tensor_subclass_like
        ? input_isnan.logical_and(value_isnan)
        : input_isnan.logical_and_(value_isnan);
    const auto mask = (input == value).logical_or_(input_and_value_isnan);
    // 返回掩码与梯度之间的元素乘积，并除以掩码的总和
    return mask * (grad / mask.sum());
  } else {
    // 创建掩码，根据情况处理 NaN 值
    auto mask = value.isnan().item<bool>() ? input.isnan() : input == value;
    // 创建与输入相同大小的零张量，并用梯度除以掩码的总和
    return grad.new_zeros(input.sizes(), input.options())
        .masked_fill_(mask, grad / mask.sum());
  }
}

// 均匀读取 JVP 的反向传播函数
Tensor evenly_read_jvp(
    const Tensor& fw_grad, // 前向梯度
    const Tensor& input,   // 输入张量
    const Tensor& value) { // 值张量
  // 创建掩码，标记输入与值相等的位置
  auto mask = (input == value);
  // 计算掩码的总和
  auto count = mask.sum();
  // 计算前向梯度的均值，返回其总和
  auto grad_output = fw_grad / count;
  return at::sum(mask * grad_output); // 返回掩码与梯度元素乘积的总和
}

// 方差反向传播函数
Tensor var_backward(
    Tensor grad,                    // 梯度张量
    const Tensor& self,             // 输入张量
    at::OptionalIntArrayRef dim_opt,// 可选的维度数组引用
    const std::optional<at::Scalar>& correction_opt, // 可选的修正标量
    bool keepdim) {                 // 保持维度标志
  const auto correction = correction_opt.value_or(1).toSymFloat();
  if (self.dim() == 0 || !dim_opt.has_value()) {
    // 计算自由度
    const auto dof = c10::SymFloat(self.sym_numel()) - correction;
    if (dof <= 0) {
      // 当自由度为负或零时，根据条件返回 NaN 或 infinity
      return grad *
          at::where(
                 self == self.mean(),
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::infinity());
    } else {
      // 计算方差的梯度
      return (c10::SymFloat(2.0) / dof) * grad * (self - self.mean());
    }
  }
  auto dim = dim_opt.value();
  if (!keepdim && self.dim() > 1) {
    // 使用 unsqueeze_multiple 函数将 grad 在指定维度 dim 上扩展，保证符号张量的大小与 self 的大小一致
    grad = unsqueeze_multiple(grad, dim, self.sym_sizes().size());
  }
  // 计算安全尺寸，确保符号张量 self 在维度 dim 上的大小
  const c10::SymFloat rnumel(_safe_size(self.sym_sizes(), dim));
  // 计算修正后的均值，确保在计算均值时不会除以零
  return (c10::SymFloat(2.0) / (rnumel - correction)) * grad *
      // 计算 self 在指定维度 dim 上的均值，保持维度不变，并与 grad 相乘
      (self - self.mean(dim, /*keepdim=*/true));
}

// 计算标准差的反向传播
Tensor std_backward(
    const Tensor& result, // 输入的标准差
    const Tensor& grad,   // 上游梯度
    const Tensor& self,   // 输入张量
    at::OptionalIntArrayRef dim, // 维度参数
    const std::optional<c10::Scalar>& correction_opt, // 修正项参数
    bool keepdim) {       // 是否保持维度

  // 计算梯度变量
  auto grad_var = (grad / (result * 2)).masked_fill_(result == 0, 0);
  
  // 调用方差的反向传播函数
  return var_backward(std::move(grad_var), self, dim, correction_opt, keepdim);
}

// 计算方差均值的反向传播
Tensor var_mean_backward(
    const Tensor& gvar,   // 方差的梯度
    const Tensor& gmean,  // 均值的梯度
    const Tensor& self,   // 输入张量
    at::OptionalIntArrayRef dim_opt, // 维度参数
    const std::optional<c10::Scalar>& correction_opt, // 修正项参数
    bool keepdim) {       // 是否保持维度

  Tensor gself;
  
  // 如果方差梯度已定义，计算方差的反向传播
  if (gvar.defined()) {
    gself = var_backward(gvar, self, dim_opt, correction_opt, keepdim);
  }

  // 如果均值梯度已定义，计算均值的反向传播
  if (gmean.defined()) {
    auto aux = mean_backward(
        gmean,
        self.sym_sizes(), // 计算符号大小
        dim_opt.value_or(IntArrayRef({})), // 默认维度
        self.sym_numel(), // 计算符号元素个数
        keepdim);
    gself = gself.defined() ? gself + aux : std::move(aux);
  }
  
  return gself;
}

// 计算标准差均值的反向传播
Tensor std_mean_backward(
    const Tensor& gstd,   // 标准差的梯度
    const Tensor& gmean,  // 均值的梯度
    const Tensor& self,   // 输入张量
    const Tensor& std,    // 标准差张量
    at::OptionalIntArrayRef dim_opt, // 维度参数
    const std::optional<c10::Scalar>& correction_opt, // 修正项参数
    bool keepdim) {       // 是否保持维度

  Tensor gself;
  
  // 如果标准差梯度已定义，计算标准差的反向传播
  if (gstd.defined()) {
    gself = std_backward(std, gstd, self, dim_opt, correction_opt, keepdim);
  }

  // 如果均值梯度已定义，计算均值的反向传播
  if (gmean.defined()) {
    auto aux = mean_backward(
        gmean,
        self.sym_sizes(), // 计算符号大小
        dim_opt.value_or(IntArrayRef({})), // 默认维度
        self.sym_numel(), // 计算符号元素个数
        keepdim);
    gself = gself.defined() ? gself + aux : std::move(aux);
  }
  
  return gself;
}

// Cholesky 分解的 Jacobian 向量积
Tensor cholesky_jvp(const Tensor& dA, const Tensor& L, bool upper) {
  at::NoTF32Guard disable_tf32;
  
  // 计算 Cholesky 分解的 Jacobian 向量积
  // 设 A = LL^H
  // dA = dLL^H + L(dL)^H
  // L^{-1}dA(L^{-H}) = L^{-1}dL + (L^{-1}dL)^H
  //               = sym(L^{-1}dL)
  // 其中 sym(X) = X + X^H
  // 短时间计算给出 sym 的逆，定义为
  // \pi(X) = X.tril() - 0.5*diag(X)
  // 所以
  // dL = L\pi(L^{-1}dA(L^{-H}))

  // 预条件：dA 是对称/Hermitian 的
  auto L_ = upper ? L.mH() : L; // 如果 upper 为真，使用 L 的共轭转置
  auto dL = at::linalg_solve_triangular(L_, dA, /*upper=*/false, /*left=*/true); // 求解三角线性方程组
  dL = at::linalg_solve_triangular(L_.mH(), dL, /*upper=*/true, /*left=*/false); // 求解三角线性方程组
  dL = dL.tril() - dL.diagonal(0, -2, -1).mul(0.5).diag_embed(); // 计算 dL
  dL = L_.matmul(dL); // 计算 L_ 与 dL 的矩阵乘积
  return upper ? dL.mH() : std::move(dL); // 如果 upper 为真，返回 dL 的共轭转置，否则返回 dL
}
// 定义函数 cholesky_backward，计算 Cholesky 分解的反向传播
Tensor cholesky_backward(const Tensor& gL, bool upper, const Tensor& L) {
  // 禁用 TF32，确保在没有 TensorFloat 32 位加速的环境下运行
  at::NoTF32Guard disable_tf32;
  
  // 从 cholesky_jvp 推导得到的反向传播公式
  // dL = L\pi(L^{-1}dA(L^-H))
  //
  // 将 gL 投影为关于 L 的下三角梯度。通过伴随操作得到 gA 的表达式
  // gA = L^{-H}\pi^*((L^HgL).tril())L^{-1}
  // 其中 \pi^*(X) = 0.5 * (X + X^H - diag(X))
  // 此处需要注意，左乘下三角矩阵 L 的伴随操作是左乘 L^H，然后再投影回下三角矩阵（因此使用 .tril() 投影）
  auto L_ = upper ? L.mH() : L;   // 如果 upper 为 true，则使用 L 的共轭转置
  auto gL_ = upper ? gL.mH() : gL; // 如果 upper 为 true，则使用 gL 的共轭转置
  
  // 不需要计算 gL_ = gL.tril()，因为
  // tril(L^H gL) = tril(L^H (triu(gL, 1) + tril(gL)))
  //              = tril(L^H tril(gL)) + tril(L^H triu(gL, 1))
  //              = tril(L^H tril(gL))
  // 因为 L^H triu(gL, 1) 是上三角矩阵，所以 tril(L^H triu(gL, 1)) = 0
  auto gA = L_.mH().matmul(gL_).tril();
  
  // 等同于 0.5 * (gA + gA^H - diag(gA))
  gA = 0.5 * (gA + gA.tril(-1).mH());
  
  // 使用三角求解方法解方程 L_.mH() X = gA，返回结果给 gA
  gA = at::linalg_solve_triangular(L_.mH(), gA, /*upper=*/true, /*left=*/true);
  gA = at::linalg_solve_triangular(L_, gA, /*upper=*/false, /*left=*/false);
  
  // 返回 gA 作为结果
  return gA;
}

// 定义函数 cholesky_inverse_backward，计算 Cholesky 逆的反向传播
Tensor cholesky_inverse_backward(
    const Tensor& grad,
    const Tensor& L,
    bool upper,
    const Tensor& inverse) {
  // 禁用 TF32，确保在没有 TensorFloat 32 位加速的环境下运行
  at::NoTF32Guard disable_tf32;
  
  Tensor grad_L;
  if (grad.defined()) {
    // 计算共同项 common_term = grad + grad.mH()
    Tensor common_term = grad + grad.mH();
    
    // 计算 common_term = inverse * (common_term * inverse)
    common_term = at::matmul(inverse, at::matmul(common_term, inverse));
    
    // 根据 upper 参数计算 grad_L
    if (upper) {
      grad_L = -at::matmul(L, common_term);
    } else {
      grad_L = -at::matmul(common_term, L);
    }
  }
  
  // 返回 grad_L 作为结果
  return grad_L;
}

// 定义函数 cholesky_inverse_jvp，计算 Cholesky 逆的 Jacobian 向量积
Tensor cholesky_inverse_jvp(
    const Tensor& F,
    const Tensor& dF,
    const Tensor& X,
    bool upper) {
  // 禁用 TF32，确保在没有 TensorFloat 32 位加速的环境下运行
  at::NoTF32Guard disable_tf32;
  
  // 根据 upper 参数选择相应的 CF 和 dCF
  const auto CF = upper ? F : F.mH();
  const auto dCF = upper ? dF.mH() : dF;
  
  // 计算 partial_dX = -X * dCF * X * CF
  const auto partial_dX = -X.matmul(dCF).matmul(X).matmul(CF);
  
  // 返回 partial_dX 加上其共轭转置作为结果
  return partial_dX + partial_dX.mH();
}

// 定义函数 forward AD，根据 Golub 和 Pereyra 的文章推导前向自动微分的公式
// Golub, Gene H., and Victor Pereyra. "The Differentiation of Pseudo-Inverses
// and Nonlinear Least Squares Problems Whose Variables Separate." SIAM Journal
// on Numerical Analysis 10(2). (1973). 413-432. doi: 10.1137/0710036
//
// 我们在此简要阐述推导过程：
// 如果 X = (L L^H)^{-1}，其中 L 是具有实正对角线的下三角矩阵，
// 则 dX = K^H + K，其中
// K =  L^{-H} dL^{-1} [dL^{-1} = -L^{-1} dL L^{-1}]
//   = -L^{-H} L^{-1} dL L^{-1} [L^{-H} L^{-1} = X]
//   = -X dL L^{-1} [X = X^H = L^{-H} L^{-1}]
//
// 如果 X = (U^H U)^{-1}，其中 U 是具有实正对角线的上三角矩阵，
// 则 K 变为
// K = -X dU^H X U
//
// 计算伪逆的雅可比向量积（Jacobian vector product，JVP）。
//
// 在这段代码中，我们实现了计算伪逆的雅可比向量积的函数 pinv_jvp。该函数用于求解逆矩阵的变化对输入矩阵的变化的影响。
//
// 首先禁用 TF32 加速，确保在运算中不会使用 TF32。
at::NoTF32Guard disable_tf32;
// 获取输入矩阵 A 的尺寸信息
auto m = A.size(-2);
auto n = A.size(-1);
// 计算输入矩阵 A 和输入的微分矩阵 dA 的共轭转置
auto dAh = dA.mH();
// 计算输入的伪逆矩阵 pinvA 的共轭转置
auto pinvAh = pinvA.mH();
// 根据矩阵尺寸大小选择最优化策略，确保产生最小维度的矩阵
if (m <= n) {
    // 计算中间矩阵 K
    auto K = pinvAh.matmul(dAh);
    // 使用中间矩阵 K 计算 pinvA 对 grad 的雅可比向量积，即 pinv_jvp 的核心计算部分
    return pinvA.matmul(K - K.mH() - K.matmul(A.matmul(pinvA))) +
        (dAh - pinvA.matmul(A.matmul(dAh))).matmul(pinvAh.matmul(pinvA));
} else {
    // 计算中间矩阵 K
    auto K = pinvA.matmul(dA);
    auto Kh = K.mH();
    // 使用中间矩阵 K 和其共轭转置 Kh 计算 pinv_jvp 的核心计算部分
    return (Kh - K - pinvA.matmul(A).matmul(Kh)).matmul(pinvA) +
        (pinvA.matmul(pinvAh)).matmul(dAh - (dAh.matmul(A)).matmul(pinvA));
}



//
// 计算伪逆的反向传播梯度。
//
// 在这段代码中，我们实现了计算伪逆的反向传播梯度的函数 pinv_backward。该函数用于求解伪逆在反向传播过程中的梯度。
//
// 首先禁用 TF32 加速，确保在运算中不会使用 TF32。
at::NoTF32Guard disable_tf32;
// 获取输入矩阵 A 的尺寸信息
auto m = A.sym_size(-2);
auto n = A.sym_size(-1);
// 计算输入的伪逆矩阵 pinvA 的共轭转置
auto pinvAh = pinvA.mH();
// 计算输入的梯度 grad 的共轭转置
auto gradh = grad.mH();
// 根据矩阵尺寸大小选择最优化策略，确保产生最小维度的矩阵
if (m <= n) {
    // 计算中间矩阵 K
    auto K = gradh.matmul(pinvA);
    auto KpinvAh = K.matmul(pinvAh);
    // 使用中间矩阵 K 和 KpinvAh 计算 pinv_backward 的核心计算部分
    return -(pinvA.matmul(K)).mH() + KpinvAh -
        (A.matmul(pinvA)).matmul(KpinvAh) +
        (pinvAh.matmul(pinvA)).matmul(gradh - K.matmul(A));
} else {
    // 计算中间矩阵 K
    auto K = pinvA.matmul(gradh);
    auto pinvAhK = pinvAh.matmul(K);
    // 使用中间矩阵 K 和 pinvAhK 计算 pinv_backward 的核心计算部分
    return -(K.matmul(pinvA)).mH() +
        (gradh - A.matmul(K)).matmul(pinvA).matmul(pinvAh) + pinvAhK -
        pinvAhK.matmul(pinvA).matmul(A);
}



//
// 变分分解梯度的反向传播。
//
// 在这段代码中，我们实现了变分分解的梯度反向传播函数 split_with_sizes_backward。
// 该函数用于计算根据给定的分割大小在指定维度上的梯度。
//
// 获取输入的梯度向量 grads、分割大小 split_sizes、维度 dim 和尺寸大小 sizes。
Tensor split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,
    c10::SymIntArrayRef split_sizes,
    int64_t dim,
    c10::SymIntArrayRef sizes,
    // 根据 sizes.size() 调整 dim 的值，确保在有效范围内
    dim = at::maybe_wrap_dim(dim, static_cast<int64_t>(sizes.size()));

    // 遍历 grads 数组，确保所有的梯度张量都已定义
    // 如果未定义，则根据 split_sizes[j] 创建一个全零张量并赋值给 grads_all_defined[j]
    std::vector<Tensor> grads_all_defined(grads.size());
    for (const auto j : c10::irange(grads.size())) {
        if (grads[j].defined()) {
            grads_all_defined[j] = grads[j];
        } else {
            // 获取当前梯度张量应有的长度信息
            const auto& length = split_sizes[j];
            // 创建与原张量相同维度的大小信息
            auto grad_size = sizes.vec();
            // 将当前梯度张量的维度dim的值设为length
            dimensions
}

// 反向传播函数，用于处理具有分割大小的张量的反向传播
Tensor _nested_split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,  // 梯度张量的向量
    c10::SymIntArrayRef split_sizes,                       // 分割大小的符号整数数组引用
    int64_t dim,                                           // 分割的维度
    const Tensor& nt_sizes,                                // 内部大小张量
    const at::TensorOptions& options) {                    // 张量选项

  // 将维度调整为考虑批次维度后的实际维度
  dim = at::maybe_wrap_dim(dim, static_cast<int64_t>(nt_sizes.size(1)) + 1);

  // 创建一个存储所有定义梯度的向量
  std::vector<Tensor> grads_all_defined;

  for (int64_t i : c10::irange(static_cast<int64_t>(grads.size()))) {
    if (grads[i].defined()) {
      grads_all_defined.push_back(static_cast<Tensor>(grads[i]));
    } else {
      const auto& length = split_sizes[i].guard_int(__FILE__, __LINE__);

      // 克隆内部大小张量
      auto nt_split_size = nt_sizes.clone();
      auto nt_split_size_ptr = nt_split_size.data_ptr<int64_t>();

      for (int64_t j : c10::irange(static_cast<int64_t>(nt_sizes.size(0)))) {
        // 调整维度，减去批次维度
        nt_split_size_ptr[j * static_cast<int64_t>(nt_sizes.size(1)) + (dim - 1)] = length;
      }

      // 创建零缓冲区
      Tensor zeros_buffer = at::zeros(
          {at::native::get_numel_from_nested_size_tensor(nt_split_size)},
          options);

      // 包装缓冲区
      auto nt_split_grad = at::native::wrap_buffer(zeros_buffer, nt_split_size);
      grads_all_defined.push_back(nt_split_grad);
    }
  }

  // 在给定维度上对所有定义的梯度张量进行拼接
  auto ret = at::cat(grads_all_defined, dim);
  return ret;
}

// 分割函数的反向传播，处理具有分割大小的张量
Tensor split_backward(
    const std::vector<torch::autograd::Variable>& grads,  // 梯度张量的向量
    const c10::SymInt& split_size,                        // 分割大小的符号整数
    int64_t dim,                                          // 分割的维度
    c10::SymIntArrayRef sym_sizes,                        // 分割大小的符号整数数组引用
    const at::TensorOptions& options) {                   // 张量选项

  // 调整维度，考虑到符号大小数组的实际维度
  dim = at::maybe_wrap_dim(dim, static_cast<int64_t>(sym_sizes.size()));

  // 获取维度的大小
  const auto& dim_size = sym_sizes[dim];

  // 确定分割的数量
  auto num_splits = grads.size();

  // 创建分割大小的符号整数向量
  std::vector<c10::SymInt> split_sizes(num_splits, split_size);

  // 调整最后一个分割大小，以便匹配维度大小
  split_sizes[num_splits - 1] =
      split_size - (split_size * num_splits - dim_size);

  // 调用带大小的分割反向传播函数
  return split_with_sizes_backward(grads, split_sizes, dim, sym_sizes, options);
}

// 最大池化双向传播的反向传播函数
Tensor max_pool_double_backward(
    const Tensor& grad,     // 梯度张量
    const Tensor& indices,  // 索引张量
    int dim) {              // 池化的维度

  // 断言索引张量的维度大于等于指定维度
  AT_ASSERT(indices.dim() >= dim);

  // 处理非空输入情况
  if (indices.sym_numel() != 0) {
    // 创建新的大小向量
    auto size = indices.sym_sizes().slice(0, indices.dim() - dim).vec();
    size.emplace_back(-1);

    // 查看索引张量，调整内存格式
    auto indices_view = indices.view_symint(size);
    const auto memory_format = indices.suggest_memory_format();

    // 连续化梯度张量，按索引查看，再调整维度
    return grad.contiguous(memory_format)
        .view_symint(size)
        .gather(-1, indices_view)
        .view_symint(indices.sym_sizes());
  }
  // 处理空输入情况
  else {
    return at::empty_like(indices, grad.options());
  }
}
Tensor error_for_max_pool2d_double_backward() { // This is mps-only.
  // 强制检查条件，如果条件不满足则抛出错误信息
  TORCH_CHECK(
      false,
      "max_pool2d with `return_indices=False` is not infinitely differentiable.",
      " If you want to calculate higher order derivatives, e.g. second order,",
      " set `return_indices=True`.");
  // 返回一个空的 Tensor 对象
  return Tensor();
}

Tensor glu_double_backward(
    const Tensor& grad,
    const Tensor& grad_output,
    const Tensor& input,
    int64_t dim) {
  // 获取梯度 grad_output 的引用
  auto& gO = grad_output;
  // 计算在指定维度上的输入尺寸的一半
  auto input_size = input.size(dim) / 2;
  // 按维度缩小输入张量，获取前半部分和后半部分
  auto first_half = input.narrow(dim, 0, input_size);
  auto second_half = input.narrow(dim, input_size, input_size);
  // 对后半部分应用 sigmoid 函数
  auto sig_second_half = second_half.sigmoid();
  // 计算 1 - sig_second_half
  auto one_sub_sig_second_half = 1 - sig_second_half;
  // 计算 sig_second_half * (1 - sig_second_half)
  auto sig_one_sub_sig = sig_second_half * one_sub_sig_second_half;

  // 缩小梯度张量到第一半和第二半部分
  auto ggI_first_half = grad.narrow(dim, 0, input_size);
  auto ggI_second_half = grad.narrow(dim, input_size, input_size);
  // 计算 ggI_second_half * first_half
  auto ggI_second_half_times_first_half = ggI_second_half * first_half;

  // 计算第一半部分的梯度
  auto gI_first_half = ggI_second_half * gO * sig_one_sub_sig;
  // 计算第二半部分的梯度
  auto second_order_sh = sig_one_sub_sig * one_sub_sig_second_half -
      sig_second_half * sig_one_sub_sig;
  auto gI_second_half =
      ggI_second_half_times_first_half * gO * second_order_sh +
      ggI_first_half * gO * sig_one_sub_sig;
  // 沿指定维度连接两个 Tensor，并返回结果
  return at::cat({std::move(gI_first_half), std::move(gI_second_half)}, dim);
}

Tensor glu_double_backward_grad_output(
    const Tensor& grad,
    const Tensor& input,
    int64_t dim) {
  // 如果维度小于零，则将其转换为非负数
  if (dim < 0)
    dim += input.dim();
  // 获取输入张量的尺寸并将指定维度的大小减半
  auto sizes = input.sizes().vec();
  sizes[dim] /= 2;
  // 使用 glu_backward 计算梯度，并与 grad 相乘
  auto tmp = grad * glu_backward(at::ones(sizes, input.options()), input, dim);
  // 按维度缩小 tmp 张量的前半部分和后半部分，然后相加
  return tmp.narrow(dim, 0, sizes[dim]) +
      tmp.narrow(dim, sizes[dim], sizes[dim]);
}

Tensor infinitely_differentiable_silu_backward(
    const Tensor& grad_output,
    const Tensor& input) {
  // 计算输入张量的 sigmoid
  const Tensor sigmoid = input.sigmoid();
  // 返回 grad_output 乘以 sigmoid 和 (1.0 + input * (1.0 - sigmoid))
  return grad_output * sigmoid * (1.0 + input * (1.0 - sigmoid));
}

Tensor infinitely_differentiable_mish_backward(
    const Tensor& grad_output,
    const Tensor& input) {
  // 计算输入张量的 sigmoid 和 softplus
  const Tensor sigmoid = input.sigmoid();
  const Tensor softplus = input.exp().log1p();
  const Tensor tanh_softplus = softplus.tanh();
  // 返回 grad_output 乘以 mish 函数的导数
  return grad_output *
      (tanh_softplus + input * sigmoid * (1.0 - tanh_softplus * tanh_softplus));
}

Tensor infinitely_differentiable_logit_backward(
    const Tensor& grad,
    const Tensor& self,
    std::optional<double> eps) {
  // 如果 eps 存在
  if (eps) {
    // 定义边界值和返回条件判断
    const double lo = eps.value();
    const double hi = 1.0 - lo;
    return at::where(
        at::logical_and(self >= lo, self <= hi),
        grad / (self * (1.0 - self)),
        at::zeros({}, self.options()));
  } else {
    // 如果 eps 不存在，返回 NaN 的张量
    return at::where(
        at::logical_and(self >= 0.0, self <= 1.0),
        grad / (self * (1.0 - self)),
        at::empty({}, self.options())
            .fill_(std::numeric_limits<double>::quiet_NaN()));
  }
}

Tensor binary_cross_entropy_target_backward(
    const Tensor& grad,
    // 定义函数，计算对输入张量 self 的逻辑斯谛函数的负梯度
    const Tensor& self,
    // 目标张量，用于计算梯度
    const Tensor& target,
    // 可选参数，权重张量
    const std::optional<Tensor>& weight,
    // 指定的减少方式，如均值、求和等
    int64_t reduction) {
  // 计算 self 的逻辑斯谛函数的负梯度
  auto grad_target = at::logit(self).neg_();

  // 检查 grad 是否不是任何张量子类，若是，则执行元素级乘法
  if (!areAnyTensorSubclassLike({grad})) {
    grad_target.mul_(grad);
  } else {
    grad_target = grad_target * grad;
  }

  // 如果定义了权重 weight
  if (isDefined(weight)) {
    // 检查 weight 是否不是张量子类，若是，则执行元素级乘法
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (!isTensorSubclassLike(weight.value())) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      grad_target.mul_(weight.value());
    } else {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      grad_target = grad_target * weight.value();
    }
  }

  // 如果指定的减少方式为均值
  if (reduction == at::Reduction::Mean) {
    // 将 grad_target 的每个元素除以 target 的总元素数目
    grad_target.div_(target.sym_numel());
  }

  // 返回计算得到的梯度
  return grad_target;
// 计算二进制交叉熵损失函数的反向传播目标
Tensor binary_cross_entropy_double_backward_target(
    const Tensor& grad,                   // 输入梯度
    const Tensor& grad_output,            // 损失函数梯度
    const Tensor& self,                   // 自身张量
    const Tensor& target,                 // 目标张量
    const std::optional<Tensor>& weight,  // 权重张量（可选）
    int64_t reduction) {                  // 缩减模式

  auto res = -grad * grad_output;         // 计算梯度乘积的负值

  if (isDefined(weight)) {                // 如果定义了权重
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    res = isTensorSubclassLike(weight.value())
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        ? res.mul(weight.value())        // 使用权重乘法
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        : res.mul_(weight.value());      // 使用原地权重乘法
  }

  auto neg_self = 1 - self;               // 计算 1 减去自身张量
  auto denom =
      isTensorSubclassLike(self) ? neg_self.mul(self) : neg_self.mul_(self);  // 计算分母，考虑张量子类

  {
    at::NoGradGuard guard;                // 禁止梯度计算上下文
    // 默认的 eps 用于所有数据类型的二进制交叉熵
    // TODO: 可能根据数据类型改变此值
    double eps = 1e-12;
    denom.clamp_min_(eps);                // 对分母进行最小值截断
  }

  res = isTensorSubclassLike(denom) ? res.div(denom) : res.div_(denom);  // 计算最终结果，考虑张量子类

  if (reduction == at::Reduction::Mean) {
    res.div_(target.sym_numel());         // 如果是均值缩减模式，计算结果除以目标张量的符号元素数量
  }

  return res;                             // 返回计算结果
}

// 计算带 logits 的二进制交叉熵损失函数的反向传播
Tensor binary_cross_entropy_with_logits_backward(
    const Tensor& grad,                   // 输入梯度
    const Tensor& input,                  // 输入 logits 张量
    const Tensor& target,                 // 目标张量
    const std::optional<Tensor>& weight,  // 权重张量（可选）
    const std::optional<Tensor>& pos_weight,  // 正权重张量（可选）
    int64_t reduction) {                  // 缩减模式

  // Trivial case
  if (grad._is_zerotensor()) {            // 如果梯度为零张量
    return at::_efficientzerotensor(input.sizes(), input.options());  // 返回高效的零张量
  }

  // -w * [ pos * y * (1 -sigmoid(x)) - (1 - y) sigmoid(x)] * grad

  // 如果存在子类张量，使用非原地版本
  Tensor grad_input;
  if (isDefined(pos_weight)) {            // 如果定义了正权重
    // pos_weight 可能需要广播，因此 mul(target) 不是原地操作
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto t = pos_weight->mul(target);     // 使用正权重乘以目标张量
    grad_input = at::areAnyTensorSubclassLike({input, target}) ||
            at::GradMode::is_enabled()
        ? t.add(1).sub(target).mul(input.sigmoid()).sub(t)  // 使用非原地操作
        : t.add(1).sub_(target).mul_(input.sigmoid()).sub_(t);  // 使用原地操作
  } else {
    grad_input = at::areAnyTensorSubclassLike({input, target}) ||
            at::GradMode::is_enabled()
        ? input.sigmoid().sub(target)    // 使用非原地操作
        : input.sigmoid().sub_(target);  // 使用原地操作
  }

  if (at::isTensorSubclassLike(grad) || at::GradMode::is_enabled()) {
    grad_input = grad_input.mul(grad);   // 如果梯度是子类张量或梯度模式启用，使用乘法操作
  } else {
    grad_input.mul_(grad);               // 否则使用原地乘法操作
  }

  if (isDefined(weight)) {                // 如果定义了权重
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (at::isTensorSubclassLike(*weight) || at::GradMode::is_enabled()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      grad_input = grad_input.mul(*weight);  // 使用权重乘法
    } else {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      grad_input.mul_(*weight);          // 使用原地权重乘法
    }
  }

  if (reduction == at::Reduction::Mean) {
    grad_input.div_(input.sym_numel());   // 如果是均值缩减模式，计算结果除以输入张量的符号元素数量
  }

  return grad_input;                      // 返回计算结果
}
    // 如果 grad_output 是零张量，则返回与目标张量形状相同的零张量
    if (grad_output._is_zerotensor()) {
        return at::_efficientzerotensor(target.sizes(), target.options());
    }

    // 初始化梯度目标张量
    Tensor grad_target;

    // 如果定义了正权重 pos_weight
    if (isDefined(pos_weight)) {
        // 检查正权重和梯度输出是否是张量子类
        if (areAnyTensorSubclassLike({*pos_weight, grad_output})) {
            // 计算梯度目标，应用对数 Sigmoid 函数和正权重
            grad_target = at::log_sigmoid(-self)
                              .sub(at::log_sigmoid(self).mul(*pos_weight))
                              .mul(grad_output);
        } else {
            // 就地计算梯度目标，应用对数 Sigmoid 函数和正权重
            grad_target = at::log_sigmoid(-self)
                              .sub_(at::log_sigmoid(self).mul_(*pos_weight))
                              .mul_(grad_output);
        }
    } else {
        // 如果未定义正权重，直接计算梯度目标
        grad_target = -self * grad_output;
    }

    // 如果定义了权重 weight
    if (isDefined(weight)) {
        // 检查权重是否是张量子类
        if (at::isTensorSubclassLike(*weight)) {
            // 使用权重乘以梯度目标张量
            grad_target = grad_target.mul(*weight);
        } else {
            // 就地使用权重乘以梯度目标张量
            grad_target.mul_(*weight);
        }
    }

    // 如果指定了减少方式为平均
    if (reduction == at::Reduction::Mean) {
        // 对梯度目标进行均值归一化
        grad_target.div_(target.sym_numel());
    }

    // 返回计算后的梯度目标张量
    return grad_target;
}

// 定义函数：计算对数sigmoid函数的双向传播梯度
Tensor log_sigmoid_double_backward(const Tensor& grad, const Tensor& input) {
  // 计算输入张量的sigmoid函数值
  auto z = input.sigmoid();
  // 返回计算得到的梯度
  return grad * (z - 1) * z;
}

// 定义函数：softmax函数的双向传播梯度
Tensor softmax_double_backward(
    const Tensor& grad,
    const Tensor& grad_output,
    int dim,
    const Tensor& output) {
  // 计算softmax函数的双向传播梯度
  return grad_output * grad - (output * grad_output).sum(dim, true) * grad -
      grad_output * (output * grad).sum(dim, true);
}

// 注释块：如何编写兼容vmap的反向传播公式
//
// 参见注释：[vmap-incompatible in-place operations]，了解哪些原地操作对vmap不兼容。
//
// 如果反向传播公式中使用的原地操作对vmap不兼容，开发者有以下选择：
//
// - 如果原地操作直接跟在通过像at::zeros(...)这样的工厂函数创建的张量之后，
//   应该用对应的grad.new_zeros(...)调用替换工厂函数。
//   grad.new_zeros(...)调用会将批次维度传播到结果张量。
//   例如：
//     Before: at::zeros(input.sizes(), grad.options()).copy_(grad)
//     After:  grad.new_zeros(input.sizes()).copy_(grad)
//
// - 如果原地操作跟在一系列操作之后，如果希望能够直接对反向传播公式进行vmap（通常适用于简单的(<15loc)反向传播公式），
//   则使用areAnyTensorSubclassLike保护该操作。例如：
//             c = a * b
//     Before: c.mul_(grad)
//     After:  c = !areAnyTensorSubclassLike({c, grad}) ? c.mul_(grad) : c * grad
//
// - 如果不希望直接对反向传播公式进行vmap（例如，如果反向传播公式过于复杂或包含大量对vmap不兼容的操作），
//   则将反向传播公式注册为一个操作，并最终为其编写一个批处理规则。

// 定义函数：二元交叉熵损失函数的双向传播梯度
Tensor binary_cross_entropy_double_backward(
    const Tensor& grad_output,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction) {
  // 定义小量eps
  auto eps = 1e-12;
  // 输入张量加上eps
  auto inp_pl_eps = input + eps;
  // 1减去输入张量加上eps
  auto one_m_inp_pl_eps = 1 - input + eps;
  // 计算梯度关于输入的部分
  auto gI = (input * input - 2 * input * target + target) /
      (inp_pl_eps.pow(2) * one_m_inp_pl_eps.pow(2));
  
  // 如果gI和grad都不是张量子类，使用原地乘法更新gI
  if (!areAnyTensorSubclassLike({gI, grad})) {
    gI *= (grad * grad_output);
  } else {
    gI = gI * (grad * grad_output);
  }

  // 如果weight已定义
  if (isDefined(weight)) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    // 如果weight不是张量子类，使用乘法更新gI
    if (!isTensorSubclassLike(*weight)) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      gI *= *weight;
    } else {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      gI = gI.mul(*weight);
    }
  }

  // 如果reduction为均值，返回均值化的gI
  if (reduction == at::Reduction::Mean) {
    return gI / input.sym_numel();
  }

  // 返回gI
  return gI;
}

// 函数定义未完整，继续下一个函数
Tensor binary_cross_entropy_double_backward_grad_output(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction) {
  auto eps = 1e-12;
  // 计算梯度相对于 grad_output
  auto ggO = (input - target) / ((input + eps) * (1 - input + eps));
  // 检查 ggO 和 grad 是否是 Tensor 的子类
  if (!areAnyTensorSubclassLike({ggO, grad})) {
    ggO *= grad;
  } else {
    ggO = ggO * grad;
  }

  // 检查权重是否定义
  if (isDefined(weight)) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    // 检查权重是否是 Tensor 的子类，如果不是，直接乘以权重
    if (!isTensorSubclassLike(*weight)) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      ggO *= *weight;
    } else {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      // 如果是 Tensor 的子类，使用乘法运算符乘以权重
      ggO = ggO.mul(*weight);
    }
  }
  // 如果 reduction 是均值，返回 ggO 除以 input 的元素个数
  if (reduction == at::Reduction::Mean) {
    return ggO / input.sym_numel();
  }
  // 否则直接返回 ggO
  return ggO;
}

// 定义了一个函数 smooth_l1_loss_double_backward，用于计算 Smooth L1 损失函数的反向传播
Tensor smooth_l1_loss_double_backward(
    const Tensor& grad,  // 输入的梯度
    const Tensor& input,  // 输入数据
    const Tensor& target,  // 目标数据
    int64_t reduction,  // 减少方式的标志
    double beta) {  // beta 参数

  // 特殊情况处理，避免除零错误
  if (beta == 0) {
    return at::zeros(grad.sizes(), grad.options());  // 返回与 grad 大小相同的零张量
  }

  auto d = (input - target).abs();  // 计算输入与目标的绝对差
  auto grad_input = grad * (d < beta).type_as(grad) / beta;  // 计算梯度输入

  if (reduction == at::Reduction::Mean) {
    grad_input /= input.sym_numel();  // 若减少方式为平均，则除以输入元素的数量
  }

  return grad_input;  // 返回计算得到的梯度输入
}

// 定义了一个函数 huber_loss_double_backward，用于计算 Huber 损失函数的反向传播
Tensor huber_loss_double_backward(
    const Tensor& grad,  // 输入的梯度
    const Tensor& input,  // 输入数据
    const Tensor& target,  // 目标数据
    int64_t reduction,  // 减少方式的标志
    double delta) {  // delta 参数

  auto d = (input - target).abs();  // 计算输入与目标的绝对差
  auto grad_input = grad * (d < delta);  // 计算梯度输入

  if (reduction == at::Reduction::Mean) {
    grad_input /= input.sym_numel();  // 若减少方式为平均，则除以输入元素的数量
  }

  return grad_input;  // 返回计算得到的梯度输入
}

// 定义了一个函数 huber_loss_double_backward_grad_output，用于计算 Huber 损失函数梯度输出的反向传播
Tensor huber_loss_double_backward_grad_output(
    const Tensor& grad,  // 输入的梯度
    const Tensor& grad_output,  // 梯度输出
    const Tensor& input,  // 输入数据
    const Tensor& target,  // 目标数据
    int64_t reduction,  // 减少方式的标志
    double delta) {  // delta 参数

  if (reduction == at::Reduction::None) {
    return huber_loss_backward(grad, input, target, reduction, delta);  // 若减少方式为无，则返回 Huber 损失函数的反向传播结果
  }

  auto r = huber_loss_backward(
      ones_like(grad_output), input, target, reduction, delta);  // 计算 Huber 损失函数的反向传播结果
  return (r * grad).sum();  // 返回结果与输入梯度的乘积的和
}

// 定义了一个函数 mse_loss_double_backward，用于计算 MSE 损失函数的反向传播
Tensor mse_loss_double_backward(
    const Tensor& grad,  // 输入的梯度
    const Tensor& input,  // 输入数据
    int64_t reduction) {  // 减少方式的标志

  auto grad_input = 2 * grad;  // 计算梯度输入

  if (reduction == at::Reduction::Mean) {
    grad_input /= input.sym_numel();  // 若减少方式为平均，则除以输入元素的数量
  }

  return grad_input;  // 返回计算得到的梯度输入
}

// 定义了一个函数 soft_margin_loss_double_backward，用于计算 Soft Margin 损失函数的反向传播
Tensor soft_margin_loss_double_backward(
    const Tensor& grad,  // 输入的梯度
    const Tensor& input,  // 输入数据
    const Tensor& target,  // 目标数据
    int64_t reduction) {  // 减少方式的标志

  auto z = (input * -target).exp();  // 计算输入与目标的乘积的负数的指数
  auto zplus1 = z + 1;  // 计算 z 加 1 的结果
  auto grad_input = grad * (target * target) * z / (zplus1 * zplus1);  // 计算梯度输入

  if (reduction == at::Reduction::Mean) {
    grad_input /= input.sym_numel();  // 若减少方式为平均，则除以输入元素的数量
  }

  return grad_input;  // 返回计算得到的梯度输入
}

// 定义了一个函数 soft_margin_loss_double_backward_grad_output，用于计算 Soft Margin 损失函数梯度输出的反向传播
Tensor soft_margin_loss_double_backward_grad_output(
    const Tensor& grad,  // 输入的梯度
    const Tensor& grad_output,  // 梯度输出
    const Tensor& input,  // 输入数据
    const Tensor& target,  // 目标数据
    int64_t reduction) {  // 减少方式的标志

  if (reduction == at::Reduction::None) {
    return soft_margin_loss_backward(grad, input, target, reduction);  // 若减少方式为无，则返回 Soft Margin 损失函数的反向传播结果
  }

  auto r = soft_margin_loss_backward(
      ones_like(grad_output), input, target, reduction);  // 计算 Soft Margin 损失函数的反向传播结果
  return (r * grad).sum();  // 返回结果与输入梯度的乘积的和
}

// 定义了一个函数 softplus_double_backward，用于计算 Softplus 激活函数的反向传播
Tensor softplus_double_backward(
    const Tensor& grad,  // 输入的梯度
    const Tensor& input,  // 输入数据
    const Scalar& beta,  // beta 参数
    const Scalar& threshold) {  // 阈值参数

  auto x = (input * beta);  // 计算输入乘以 beta 的结果
  return sigmoid_backward(grad, x.sigmoid()) * (x < threshold).type_as(grad) * beta;  // 计算梯度输入
}

// 注意事项的注释，这部分代码涉及到 as_strided 的反向传播和布局无关/有关的自动求导
//
// `storage_offset` 在这个注释中被简化忽略。如果你只是想要算法而不是解释，请直接向这个注释的底部滚动。
//
// 实现 as_strided 的反向传播是棘手的，因为你必须处理将一个内存位置映射到多个索引的映射，即，
//
// output tensor has multiple indices pointing to overlapping memory
// addresses. This can happen in all in all sorts of weird cases. For example,
//
//   x = torch.randn(15)
//   x.as_strided([3, 3], [1, 0])  // "expand" case
//   x.as_strided([3, 3], [2, 1])  // "size too large" case
//   x.as_strided([3, 2], [3, 6])  // res[2, 0] points to 2*3 + 0*6 = 6
//                                 // res[0, 1] points to 0*3 + 1*6 = 6
//
// Here is the general strategy we apply in implementing as_strided backward:
//   0. ??? (optimization step. we will talk about this later)
//   1. Create some underlying flattened tensor as if it is the base tensor
//      representing the contiguous memory storage for both input and output.
//   2. Use the output geometry to scatter (or index_add) the gradients into
//      this storage tensor.
//   3. ??? (fix for input tensor with overlapping memory. we will talk about
//           this later)
//   4. Return the as_strided view of the storage tensor using input geometry.
//
// In step (2), if the output tensor doesn't have overlapping memory, we can
// safely scatter (`storage.as_strided(output_geometry).copy_(grad)`);
// otherwise, we must use `index_add` as gradients at different indices may need
// to be summed to a single location.
//
// For example, in this case:
//
//   x = torch.randn(3)
//   y = x.as_strided([3, 3], [1, 0])  // "expand" case
//                                     // size   [ 3, 3]
//                                     // stride [ 1, 0]
//   y.backward()  // step (1): contiguous storagte tensor `s` of size 3, which
//                             is large enough to be used as underlying storage
//                             for `x` and `y`.
//                               s = [ 0, 0, 0]
//                 // step (2): since `y` has overlapping memory, index_add grad
//                             into `s` basing on `y`'s geometry, i.e.,
//                             s[i * y.stride(0) + j * y.stride(1)] += gy[i, j].
//                               s = [ 3, 3, 3]
//                 // step (4): as_strided view `s` using `x`'s geometry
//                               s = [ 3, 3, 3]
//                               grad_input = s.as_strided(x.size(), x.stride())
//                                          = s.as_strided([3], [1])
//                                          = [ 3, 3, 3]
//
// This is exactly what we would get if using `expand`. However, here the input
// tensor doesn't have overlapping memory. If it does, we must add an extra step
// before (4). Considering this case:
//
//   t = torch.randn(3)
//   x = t.expand(3, 3)            // input with overlapping memory
//                                 // size   [3, 3]
//                                 // stride [0, 1]
//   y = x.as_strided([1], [1])    // contiguous output
//                                 // size   [1]
//                                 // stride [1]
//   y.backward()  # 执行反向传播，计算梯度
//                 # 步骤（1）：创建大小为 3 的连续存储张量 `s`，作为 `x` 和 `y` 的底层存储
//                               s = [ 0, 0, 0]
//                 # 步骤（2）：根据 `y` 的几何结构将梯度散布到 `s` 中
//                               s = [ 1, 0, 0]
//                 # 步骤（4）：使用 `x` 的几何结构创建 `s` 的 as_strided 视图
//                               s = [ 1, 0, 0]
//                               grad_input = s.as_strided([3, 3], [0, 1])
//                                          = s.as_strided([3, 3], [0, 1])
//                                          = [[ 1, 0, 0],
//                                             [ 1, 0, 0],
//                                             [ 1, 0, 0]]
// 这个结果是否正确？

// 对于任意 `x`，`x.as_strided([1], [1])` 的调用显然等同于 `x[(0,) * x.dim()].view(1)`。
// 但通过第二种方式进行自动求导得到的梯度是 `[ [ 1, 0, 0], [ 0, 0, 0], [ 0, 0, 0]]`。
// 在这种特定情况下，索引 `x` 在第一列的任何索引也是等效的，会产生形状为 `[3 x 3]` 的梯度，包含八个 0 和一个 1。
// 从其他 PyTorch 操作中计算的梯度与从 as_strided 得到的梯度之间存在 `x.size(1)` 倍的差异。

// 你可能会认为从 as_strided 得到的梯度是错误的。然而，让我们首先看看它们为什么实际上是合理的。
// 考虑在 `x` 的第一列的任何位置进行 `delta` 的逐点扰动。它将导致相同内存位置的 `delta` 改变，然后 `y` 将改变 `delta`。
// 因此，可以说梯度应该在第一列正好为 1，正如我们上面的过程所示。

// 在上述数值梯度计算中，它们之所以与分析结果匹配，是因为在前向传播中考虑了步幅和内存位置，即此操作（包括前向和反向传播）是与布局相关的。

// 然而，在 PyTorch 中，大多数（可能是所有）其他操作（前向和反向）是与布局无关的。例如，

//   t = torch.randn(1)
//   x = t.expand(2)
//   y = x.sum()
//   y.backward()

// 无关布局的自动求导（目前在 PyTorch 中）将给出

//   gy = 1
//   gx = [ 1, 1]  # SumBackward:    torch.ones_like(x)
//   gt = [ 2]     # ExpandBackward: gx.sum()

// 注意 `gx = [ 1, 1]`。然而，如果你通过 `delta` 扰动 `x` 中的任何值（另一个值也会改变 `delta`），`y` 将改变 `2 * delta`。
// 因此，如果考虑到步幅，梯度应该是 2。

// 布局感知的自动求导应该给出

//   gy = 1
//   gx = [ 2, 2]  # 因为反向传播考虑到了输入 `x` 已经扩展的事实。
//   gt = [ 2]     # 扩展的反向传播只是一个切片，因为之前的反向传播应该已经处理了
//                 # strides and made sure that gradients are the same along the
//                 # expanded dimension.
//
// As shown above, these two types are not compatible. Therefore, we must either
// make as_strided layout-agnostic, or make all other ops layout-aware.
//
// It is difficult to support layout-aware autograd (at least in the current
// codebase structure), because it would mean
//   1. storing tensor geometries of every input tensor for backward
//   2. depending on input geometry, the gradient computed from backward change
//   3. ideally enforcing gradient of T to always have same strides as T
// (although these two methods only differ when it comes to overlapping memory)
//
// Therefore, we must formulate `as_strided` in a layout-agnostic way, i.e.,
// giving the same output regardless of the input layout. We consider
// `input.stride()` as a separate independent fixed argument `input_stride`.
// Then, `as_strided(input, size, stride)` can be thought of as:
//   1. "Scatter" each value of `input` into a "storage" using storage location
//      computed from the value's index in `input`, `input.size()` and
//      `input_stride`, but if N values end up in the same location, the value
//      is average of those N values (they will be the same value anyways).
//
//      Formal description:
//        Denote the set of all input indices that pointing to the same storage
//        location `storage[n]` as `S(n)`, i.e.,
//
//            S(n) = { index : <index, input_stride> == n, index is valid given
//            input.size() },
//
//        where `<x, y>` is the dot product between `x` and `y`.
//
//        Then, the process is:
//
//            storage[n] = Avg { S(n) }
//
//        Note that all values in `S(n)` are the same (they point to the same
//        memory location anyways, so this step doesn't change anything, but
//        effectively avoids having the dependency on the layout of `input`.
//        I.e., the result holds fixed regardless of the layout of `input`, as
//        long as `input_stride` is fixed.
//
//      NOTE: for forward pass, we can equivalently simply select any one of
//            `S(n)` as `storage[n]`. However, considering this as an average
//            operation makes backward easier (so all values in set
//            `{ grad_input[i] : i in S(n) }` are the same, and it can use the
//            same geometry as input).
//   2. As usual, return the as_strided view of `storage` using required output
//      `size` and `stride`.
//
// To backward through this layout-agnostic version, we simply add the following
// step:
//   .... (scatter gradients into the storage tensor using output geometry)
//   3. For all storage location n, `storage[n] /= |S(n)|`.
//   .... (return as_strided view of the storage tensor using input geometry)
//
// Finally, we note that these general operations are expensive, so we apply the
// following optimizations:
//
// ```
//   Add step (0): For all output dimension `d` with output stride 0, sum the
//                 gradients along dimension `d` (don't keepdim), and remove
//                 dimension `d` from output size and stride.
//                 (An optimization for "expand" cases so we may avoid step (3))
//  Only apply step (3) when input tensor has overlapping memory.
//
// FULL ALGORITHM:
//   0. For all output dimension `d` with output stride 0, sum the gradients
//       along dimension `d` (don't keepdim), and remove dimension `d` from
//       output size and stride.
//   1. Create some underlying flattened tensor as if it is the base tensor
//      representing the contiguous memory storage for both input and output.
//   2. Use the output geometry to scatter (or index_add) the gradients into
//      this storage tensor `storage`.
//   3. If input tensor has overlapping memory,
//      For all storage location `i`, `storage[i] /= N(i)`, where `N(i)` is the
//      number of indices in input geometry pointing to the same storage
//      location `i` (i.e., `|S(i)|` in equations above).
//   4. Return the as_strided view of the storage tensor using input geometry.
//
// See NOTE [ Detecting Memory Overlap Within A Strided Tensor ] on how to
// roughly detect overlapping memory.

// NOTE [ Detecting Memory Overlap Within A Strided Tensor ]
//
// Checking memory overlap within a strided tensor is the special case of
// detecting memory overlap of two strided tensors, where the two tensors start
// at the same memory address. The latter is HARD (see #8212).
//
// But even this special case isn't simple. This note describes a check for an
// even more constrained simple case where we can be certain that there is no
// overlap.
//
// The checking algorithm can be described as:
//   0. Return [ pass check ] if any dimension has size 0
//   1. Ignore all dimensions that have size 1
//   2. If no remaining dimensions, return [ pass check ]
//   3. Sort the remaining dimensions according to the strides decreasingly
//   4. Check that for each dimension k,
//
//           stride[k] > \sum_{ i > k } (size[i] - 1) * stride[i]
//
//      That is equivalent to, after reordering the dimensions so strides are
//      in decreasing order, checking that the stride of each dimension is larger
//      than the maximum memory offset in a slice at that dimension.
//
// Obviously, this check passes for contiguous tensors (the dimensions will be
// already sorted with LHS = stride[0] = \prod size[i] being exactly 1 larger
// than RHS). Similarly, the check passes for tensors contiguous in all but
// the last dimension, and LHS = stride[0] = stride[-1] * \prod size[i] being
// exactly stride[-1] larger than RHS. (*)
//
// We will show that these view operations, including all our view operations
// *except for* general as_strided and unfold, also preserve this invariant:
//
//  alias:      Obviously preserves
//
//  expand:     All changed dimensions are removed in step (1)
//
//  view:       将输入维度视为分组成连续的维度“块”，其中每个块中的维度是连续的。
//              只有当输出维度也可以分组成相同的连续块并保持相同顺序时，view操作才有效。
//
//              注意：
//                这意味着在每个块中，输入和输出的最后一个维度的元素数和步长是相同的。 (**)
//
//              符号表示：
//                考虑一个这样的块 B，
//                    ... B_prev[-1]], [ B[0], ..., B[i], ..., B[k] = B[-1] ], [
//                    B_next[0], ...
//                                start--^^^^                  ^^^^^^^^^^^^--end
//                每个 B[i] 表示一个维度索引，满足 B[i] = B[0] + i。
//
//              我们首先展示在一个张量（即输入）满足不变性后，经过排序后，每个块内的维度仍然是连续的。 (***)
//
//                在移除大小为1的维度后，每个块内的维度已经按降序排序。因此，对所有维度进行排序不会改变它们之间的相对顺序。
//
//                假设某个块 B 在排序后不是连续的，即在排序后的顺序中存在一个维度 d 在 B[0] 和 B[-1] 之间。
//
//                根据 (*)，我们知道
//                       stride[B[0]]
//                    =  \sum_{i > 0}   (size[B[i]] - 1) * stride[B[i]] +
//                    stride[B[-1]] <  \sum_{i > 0}   (size[B[i]] - 1) *
//                    stride[B[i]] + stride[d]
//                    <= \sum_{i > 0}   (size[B[i]] - 1) * stride[B[i]] +
//                    (size[d] - 1) * stride[d]
//                    <= \sum{j > B[0]} (size[j]    - 1) * stride[j],
//
//                第一个 <   来自于排序，
//                第二个 <= 来自于维度 d 存在于第一步后，并且因此其大小必须大于1，
//                第三个  <= 来自于求和中每一项均为非负数。
//
//                那么我们在 B[0] 处就有了一个矛盾，因为不变性在此处不应成立。因此，原命题成立。
//
//              现在我们已经证明了上述主张 (***)，我们将视图操作视为首先对维度（即块）进行排序，
//              应用原始视图（因为它只关心每个块内的维度是否连续和连续），然后撤消排序。
//
// Consider a single block B in the output,
// ... ], [ B[0], ..., B[i], ..., B[k] = B[-1] ], [ ...
// start--^^^^                  ^^^^^^^^^^^^--end
//
// By (*), we know that for all i
// stride[i] = stride[B[-1]] +
//             \sum_{j=i+1}^{k} (size[B[j]] - 1) *
//             stride[B[j]]
//
// Then the invariant is obviously satisfied at every dimension
// in this block if it is satisfied at dimension B[-1]. It only
// remains to show that it is satisfied at the last dimension in
// each block.
//
// Since the same blocks are present in both input and output
// with the same ordering, we will abuse the notation in the
// following statements.
//
// By (*), we know that the following holds for both input and
// output, for any block B:
//   \sum_{i > B[-1]} (size[i] - 1) * stride[i]
// = \sum_{block B' after B} \prod_{j in B'} size[B[j]] *
//   stride[B'[-1]] = \sum_{block B' after B} numel(B') *
//   stride[B'[-1]].
//   ^^^^^^^^^^^^^^^^^^^^^^^|^^^^^^^^^^^^^^^^^^^^^^^^^^
// By (**), we know that, this quantity in the above equation
// remains the same in input and output. So both
//   \sum_{i > B[-1]} (size[i] - 1) * stride[i]
// and
//   stride[B[-1]]
// are the same in input and output.
//
// These two quantities are exactly the LHS and RHS of the
// invariant inequality. Since by assumption the invariant is
// satisfied in input at B[-1], it is also satisfied in output at
// B[-1]. This concludes the proof.
//
// squeeze: Special case of view
//
// unsqueeze: Special case of view
//
// slice: Consider slicing dimension i with step = k >= 1.
//
// Let stride' and size' be the output strides and sizes. We have
//
// stride'[i] = k * stride[i]
// size'[i] <= floor(size[i] / k)
//
// If size'[i] = 1, invariant is obviously satisfied as we are
// just removing a dimension (after step (1)).
//
// Assume size'[i] > 1.
//
// By assumption, the invariant is satisfied at every dimension
// in input.
//
// For any dimension j, if stride[j] > stride[i], we have
//   stride'[j] = stride[j]
//              > (size[i] - 1) * stride[i]
//              = (size[i] / k * k - 1) * k * stride[i] / k
//              = (size[i] / k - 1 / k) * stride'[i]
//              >= (size'[i] - 1 / k) * stride'[i]
//              >= stride'[i].
//
// If stride[j] < stride[i], we have
// This comment block describes transformations and operations related to tensor dimensions and strides.
// It explains how various operations such as slice, narrow, select, permute, transpose, and diagonal affect tensor dimensions and strides.
//
// slice:      Slicing keeps the sorted order of dimensions unchanged.
//
// narrow:     A special case of slice operation.
//
// select:     Combines narrow and squeeze operations.
//
// permute:    Sorting dimensions renders permutation irrelevant.
//
// transpose:  Sorting makes swapping dimensions irrelevant.
//
// diagonal:   Merges dimensions i and j into a new dimension k where:
//             stride'[k] = stride[i] + stride[j]
//             size'[k]   <= min(size[i], size[j])
//             Assuming size[i] > 1 and size[j] > 1; if either is 1, it's an unsqueeze.
//             It effectively removes dimension i and replaces j with k.
//
//             For dimensions d where stride[d] > stride[j], the invariant inequality's term from dimension i is removed.
//             For dimensions d where stride[i] < stride[d] < stride[j], the term from i in the inequality decreases.
//             For dimensions d where stride[d] > stride[j], the combined term from i and j can only decrease.
//             Thus, the operation relaxes constraints and preserves the invariant.
//
// This implements steps (2)~(4) of the algorithm in
// NOTE [ Detecting Memory Overlap Within A Strided Tensor ]
// Helper for as_strided_backward
static inline bool _maybe_overlapping_memory(
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides) {
  // Check if sizes is not empty
  if (!sizes.empty()) {
    // Create a vector of indices [0, 1, ..., sizes.size()-1]
    std::vector<std::size_t> argsort(sizes.size());
    std::iota(argsort.begin(), argsort.end(), 0);
    // Sort argsort based on the strides array
    std::sort(
        argsort.begin(), argsort.end(), [&](std::size_t i, std::size_t j) {
          return strides[i] < strides[j];
        });
    # 初始化最大切片索引为0
    c10::SymInt max_index_in_slice = 0;
    # 遍历排序后的索引列表argsort中的每个索引i
    for (auto i : argsort) {
      # 获取当前索引对应的步长stride_
      const auto& stride_ = strides[i];
      # 如果当前步长小于等于最大切片索引，返回true，表示切片不是有效的
      if (stride_ <= max_index_in_slice) {
        return true;
      }
      # 更新最大切片索引，乘以当前维度sizes[i]减1后加到max_index_in_slice中
      max_index_in_slice += stride_ * (sizes[i] - 1);
    }
  }
  # 如果所有索引都遍历完毕仍未返回true，则返回false，表示切片是有效的
  return false;
}

// 返回包含给定张量大小、步长和存储偏移量的最小存储大小，作为 as_strided_backward 的辅助函数
static inline c10::SymInt _min_storage_size(
    c10::SymIntArrayRef sizes,             // 张量的大小数组
    c10::SymIntArrayRef strides,           // 张量的步长数组
    c10::SymInt storage_offset) {          // 存储偏移量
  c10::SymInt storage_size = storage_offset + 1;  // 初始化存储大小为偏移量加一
  auto dim = sizes.size();                // 张量的维度
  for (const auto i : c10::irange(dim)) {  // 遍历每一个维度
    const auto& size_i = sizes[i];        // 当前维度的大小
    if (size_i == 0) {                    // 如果当前维度的大小为0
      return storage_offset;              // 直接返回存储偏移量
    }
    storage_size += (size_i - 1) * strides[i];  // 计算存储大小的增量
  }
  return storage_size;                    // 返回计算得到的存储大小
}

// 查看 NOTE [ as_strided Backward and layout-aware/agnostic autograd ] 获取详细解释
// 对输出几何结构进行 as_strided_backward 操作，处理符号化大小、步长和可选的存储偏移量
Tensor as_strided_backward(
    Tensor grad,                          // 梯度张量
    const TensorGeometry& input_geometry, // 输入几何结构
    c10::SymIntArrayRef sym_sizes,        // 符号化大小数组
    c10::SymIntArrayRef sym_strides,      // 符号化步长数组
    const optional<c10::SymInt>& sym_storage_offset_) {  // 可选的符号化存储偏移量
  // 对于输出几何结构，
  //   检查大小为0的维度，
  //   跳过大小为1的维度，
  //   在扩展维度上减少梯度 (步长为0，大小大于1)
  // 步骤 (0)     查看 NOTE [ as_strided Backward and layout-aware/agnostic autograd ] 中的算法步骤 (0)~(1)
  //              在输出几何结构上
  auto sym_storage_offset =
      sym_storage_offset_.value_or(input_geometry.sym_storage_offset());  // 使用提供的或者输入几何结构的存储偏移量
  auto odim = grad.dim();                 // 输出张量的维度
  std::vector<c10::SymInt> out_sizes_, out_strides_;  // 输出大小和步长的容器
  out_sizes_.reserve(odim);               // 预留输出维度的空间
  out_strides_.reserve(odim);             // 预留输出维度的空间
  for (int64_t i = odim - 1; i >= 0; i--) {  // 逆序遍历输出张量的每一个维度
    const auto& size_i = sym_sizes[i];     // 当前维度的符号化大小
    const auto& stride_i = sym_strides[i]; // 当前维度的符号化步长
    if (size_i == 0) {                     // 如果当前维度的大小为0
      return at::zeros_symint(input_geometry.sym_sizes(), grad.options());  // 返回相同大小的零张量
    } else if (size_i == 1) {              // 如果当前维度的大小为1
      grad = grad.squeeze(i);              // 压缩当前维度
    } else if (stride_i == 0) {            // 如果当前维度的步长为0
      grad = grad.sum(i, false);           // 在当前维度上求和
    } else {                               // 否则
      out_sizes_.insert(out_sizes_.begin(), size_i);    // 插入输出大小数组的开始位置
      out_strides_.insert(out_strides_.begin(), stride_i);// 插入输出步长数组的开始位置
    }
  }
  // 步骤 (2)~(4) 查看 NOTE [ Detecting Memory Overlap Within A Strided Tensor ] 中的算法步骤
  //              在输出几何结构上
  auto out_maybe_overlap = _maybe_overlapping_memory(out_sizes_, out_strides_);  // 检测可能的内存重叠

  // 对于输入几何结构，
  //   检查大小为0的维度，
  //   跳过大小为1的维度，
  // 步骤 (0)~(1) 查看 NOTE [ Detecting Memory Overlap Within A Strided Tensor ] 中的算法步骤
  //              在输入几何结构上
  auto idim = input_geometry.dim();       // 输入张量的维度
  auto inp_sizes = input_geometry.sym_sizes(),  // 输入符号化大小数组
       inp_strides = input_geometry.sym_strides();  // 输入符号化步长数组
  std::vector<c10::SymInt> inp_sizes_, inp_strides_;  // 输入大小和步长的容器
  inp_sizes_.reserve(idim);               // 预留输入维度的空间
  inp_strides_.reserve(idim);             // 预留输入维度的空间
  for (int64_t i = idim - 1; i >= 0; i--) {  // 逆序遍历输入张量的每一个维度
    const auto& size_i = inp_sizes[i];     // 当前维度的符号化大小
    const auto& stride_i = inp_strides[i]; // 当前维度的符号化步长
    if (size_i == 0) {                     // 如果当前维度的大小为0
      return at::zeros_symint(input_geometry.sym_sizes(), grad.options());  // 返回相同大小的零张量
  } else if (size_i != 1) {
    // 如果 size_i 不等于 1，则在输入大小的开头插入 size_i
    inp_sizes_.insert(inp_sizes_.begin(), size_i);
    // 在输入步幅的开头插入 stride_i
    inp_strides_.insert(inp_strides_.begin(), stride_i);
  }
}
// 步骤（1）~（4）：在输入几何结构上执行内存重叠检测算法的实现
//               参考注释中的算法笔记 [检测跨步张量内存重叠]
auto inp_maybe_overlap = _maybe_overlapping_memory(inp_sizes_, inp_strides_);

// 函数的其余部分实现了
// 步骤（1）~（4）：实现注释中的算法 [as_strided 反向传播和与布局无关/有关的自动求导]
// TODO: 如果输出值在输入几何结构中不可见，则引发异常。
//       严格来说，如果将这些值视为常量，不引发异常也是正确的数学行为。
//       然而，这些值实际上包含在某个基本张量中，将它们视为常量意味着忽略了这种紧密的依赖关系。
//       因此，在此处引发异常更为合理。

// 步骤（1）：将底层张量创建为“存储”
auto shared_offset =
    // TODO: 使用 SymInt 进行符号整数化。我们是否需要 SymInt 的 min() 和 max() 函数？
    input_geometry.sym_storage_offset().min(sym_storage_offset);
auto inp_effective_offset =
    input_geometry.sym_storage_offset() - shared_offset;
auto out_effective_offset = sym_storage_offset - shared_offset;
auto base_size1 =
    _min_storage_size(inp_sizes_, inp_strides_, inp_effective_offset);
auto base_size2 =
    _min_storage_size(out_sizes_, out_strides_, out_effective_offset);
auto base_size = base_size1.max(base_size2);
// 创建一个与 base_size 相同的符号整数数组的新零张量
auto storage = grad.new_zeros_symint(c10::SymIntArrayRef(base_size));

// 如果稍后将执行 index_add_，则准备索引张量
std::optional<at::Tensor> flatten_full_indices;
if (inp_maybe_overlap || out_maybe_overlap) {
  flatten_full_indices =
      // TODO: 应该对 arange 使用 SymInt 吗？需要 SymScalar。
      at::arange(
          0,
          base_size.guard_int(__FILE__, __LINE__),
          grad.options().dtype(at::kLong));
}

// 步骤（2）：使用输出几何结构将梯度散射到存储中
if (out_maybe_overlap) {
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto out_indices = flatten_full_indices->as_strided_symint(
      out_sizes_, out_strides_, out_effective_offset);
  // 在 storage 中根据 out_indices 散射梯度，对应位置进行累加
  storage.index_add_(0, out_indices.reshape(-1), grad.reshape(-1));
} else {
  // 假设新张量的存储偏移为 0
  // 使用 out_sizes_ 和 out_strides_ 在 storage 上创建 as_strided_symint
  storage.as_strided_symint(out_sizes_, out_strides_, out_effective_offset)
      .copy_(grad);
}

// 步骤（3）：如果输入张量具有重叠的内存，则将散射到存储中的梯度除以 i 在输入几何结构中出现的次数
if (inp_maybe_overlap) {
  // 创建一个与 storage 形状相同的零张量 count
  auto count = at::zeros_like(storage, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto inp_indices =
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      flatten_full_indices
          ->as_strided_symint(inp_sizes_, inp_strides_, inp_effective_offset)
          .reshape(-1);
    # 使用索引增加操作，将输入索引处的值加上1，用于计数
    count.index_add_(
        0, inp_indices, at::ones({1}, grad.options()).expand_as(inp_indices));
    # 对存储张量进行除法操作，这可能在可见范围之外导致结果为 NaN
    storage.div_(count); // this will give nan outside visible range
  }
  // 步骤 (4): 返回存储张量的按步幅视图，使用输入的几何信息
  return storage.as_strided_symint(
      inp_sizes, inp_strides, inp_effective_offset);
}

// 定义一个函数，用于计算反向传播时的 as_strided_scatter 操作
Tensor as_strided_scatter_backward(
    const Tensor& grad,  // 输入参数：梯度张量
    const TensorGeometry& input_geometry,  // 输入参数：输入张量的几何属性
    const TensorGeometry& src_geometry,  // 输入参数：源张量的几何属性
    c10::SymIntArrayRef sizes,  // 输入参数：目标大小
    c10::SymIntArrayRef strides,  // 输入参数：目标步幅
    optional<c10::SymInt> storage_offset) {  // 输入参数：存储偏移量的可选值

  // 注释部分解释了 as_strided_scatter 在 autograd 中的支持情况
  // 由于大多数情况下只关心连续的情况，所以这里假设输入是连续的张量
  // 并且会将 grad 张量变为连续张量以确保处理的简化性
  auto grad_ = grad.contiguous();
  // 使用给定的大小、步幅和存储偏移创建 grad_ 的视图
  auto grad_slice = grad_.as_strided_symint(sizes, strides, storage_offset);
  // 创建一个与输入几何形状相同的全零张量
  auto result_buffer = grad_.new_zeros_symint(input_geometry.sym_sizes());
  // 创建 result_buffer 的视图，使用输入几何形状的大小和步幅
  auto result = result_buffer.as_strided_symint(
      input_geometry.sym_sizes(), input_geometry.sym_strides());
  // 创建 result_buffer 的视图，使用给定的大小、步幅和存储偏移
  auto result_slice = result_buffer.as_strided_symint(
      sizes, strides, std::move(storage_offset));
  // 将 grad_slice 的数据复制到 result_slice
  result_slice.copy_(grad_slice);
  // 返回 result 张量作为结果
  return result;
}

// 定义 atan2 函数的反向传播
std::tuple<Tensor, Tensor> atan2_backward(
    const Tensor& grad,  // 输入参数：梯度张量
    const Tensor& self,  // 输入参数：自变量 self 张量
    const Tensor& other,  // 输入参数：自变量 other 张量
    std::array<bool, 2> output_mask) {  // 输入参数：输出掩码数组

  // 如果梯度张量未定义，则返回两个空张量的元组
  if (!grad.defined()) {
    return std::tuple<Tensor, Tensor>{Tensor(), Tensor()};
  }
  // 计算 recip = 1 / (self^2 + other^2)
  auto recip = (self * self + other * other).reciprocal();
  // 返回梯度张量乘以相应的导数
  return std::tuple<Tensor, Tensor>{
      output_mask[0] ? grad * other * recip : Tensor(),  // 第一个输出张量
      output_mask[1] ? grad * -self * recip : Tensor()};  // 第二个输出张量
}

// 定义 gelu 函数的双向传播
Tensor gelu_double_backward(
    const Tensor& ggI,  // 输入参数：中间梯度 ggI
    const Tensor& gO,  // 输入参数：输出梯度 gO
    const Tensor& input,  // 输入参数：输入张量 input
    c10::string_view approximate) {  // 输入参数：近似方式的字符串视图

  // 如果采用 tanh 近似方式
  if (approximate == "tanh") {
    // 定义常量 kBeta 和 kKappa
    constexpr auto kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr auto kKappa = 0.044715;

    // 计算内部变量和其 tanh、sech
    auto inner = kBeta * (input + kKappa * pow(input, 3));
    auto tanh_inner = tanh(inner);
    auto sech_inner = 1 / cosh(inner);

    // 计算 f, g, h 和 f_prime_gh
    auto f = 0.5 * input;
    auto g = 1 - tanh_inner * tanh_inner;
    auto h = kBeta * (1 + 3 * kKappa * input * input);
    auto f_prime_gh = 0.5 * g * h;

    // 计算 g_prime 和 g_prime_fh
    auto g_prime = (2 * sech_inner) * (-sech_inner * tanh_inner) * h;
    auto g_prime_fh = f * h * g_prime;

    // 计算 h_prime 和 h_prime_fg
    auto h_prime = 6 * kKappa * input * kBeta;
    auto h_prime_fg = f * g * h_prime;

    // 计算 left_derivative 和 right_derivative
    // 然后计算 dgrad_dX = left_derivative + right_derivative
    auto gI = ggI * gO * (2 * f_prime_gh + g_prime_fh + h_prime_fg);
    // 返回 gI 作为结果
    return gI;
  } else {  // 如果采用其他近似方式
    // 定义常量 kBeta 和计算输入的平方
    constexpr auto kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    auto input_sq = input * input;
    // 计算正态分布概率密度函数 pdf
    auto pdf = kBeta * at::exp(-0.5 * input_sq);
    // 计算输入变量的梯度 dgrad_dInput
    auto dgrad_dInput = 2 * pdf - input_sq * pdf;
    // 返回 ggI 与 gO 相乘后乘以 dgrad_dInput 作为结果
    auto gI = ggI * gO * dgrad_dInput;
    // 返回 gI 作为结果
    return gI;
  }
}
  // 如果 `is_result` 为真，则执行以下逻辑
  if (is_result) {
    // 返回计算得到的梯度
    return grad * grad_output * input_scale *
        // 返回self_or_result是否小于0的张量
        (self_or_result < 0).type_as(grad);
  } else {
    // 否则，调用ELU的反向传播函数，计算梯度
    return at::elu_backward(
               grad * grad_output * input_scale,
               alpha,
               scale,
               input_scale,
               is_result,
               self_or_result) *
        // 返回self_or_result是否小于0的张量
        (self_or_result < 0).type_as(grad);
  }
}

// 返回切片操作的反向传播张量
Tensor slice_backward_wrapper(
    const at::Tensor& grad,
    const c10::SymIntArrayRef& input_sizes,
    int64_t dim,
    std::optional<c10::SymInt> start,
    std::optional<c10::SymInt> end,
    c10::SymInt step) {
  // 确定起始和结束位置的值
  auto start_val = start.has_value() ? start.value() : 0;
  auto end_val = end.has_value() ? end.value() : INT64_MAX;

  // 调用slice_backward_symint函数执行切片的反向传播计算
  return slice_backward_symint(
      grad,
      input_sizes,
      dim,
      std::move(start_val),
      std::move(end_val),
      std::move(step));
}

// 执行奇异值分解（SVD）的雅可比向量积（Jacobian Vector Product）运算
std::tuple<Tensor, Tensor, Tensor> linalg_svd_jvp(
    const Tensor& dA,
    const Tensor& U_,
    const Tensor& S,
    const Tensor& Vh_,
    const bool full_matrices) {
  // 禁用TF32运算
  at::NoTF32Guard disable_tf32;
  // 参考svd_backward的推导过程
  // 根据sym(X) = X + X^H，实现不同条件下的dU、dS、dV、dVh计算
  // 这些计算在奇异值分解的Jacobian Vector Product中起到关键作用

  // 检查U_和Vh_的维度是否满足要求
  TORCH_INTERNAL_ASSERT(U_.dim() >= 2 && Vh_.dim() >= 2);

  // 判断dA是否为复数张量，以及获取相关维度信息
  const auto is_complex = dA.is_complex();
  const auto m = dA.size(-2);
  const auto n = dA.size(-1);
  const auto k = S.size(-1);

  // 根据full_matrices参数选择合适的U和Vh
  const auto U = full_matrices ? U_.narrow(-1, 0, k) : U_;
  const auto Vh = full_matrices ? Vh_.narrow(-2, 0, k) : Vh_;
  const auto V = Vh.mH();

  // 计算dP = U^H * dA * V
  auto dP = m >= n ? at::matmul(U.mH(), at::matmul(dA, V))
                   : at::matmul(at::matmul(U.mH(), dA), V);

  // 计算实部对角线dS
  auto dS =
      is_complex ? at::real(dP.diagonal(0, -2, -1)) : dP.diagonal(0, -2, -1);

  // 计算dX = dP - diag(dS)
  dP = dP - dS.diag_embed();

  // 计算E矩阵
  auto E = [&S] {
    const auto S2 = S * S;
    auto ret = S2.unsqueeze(-2) - S2.unsqueeze(-1);
    // 对角线上的元素填充为1
    ret.diagonal(0, -2, -1).fill_(1);
    return ret;
  }();

  // 定义对称函数sym(X) = X + X^H
  const auto sym = [](const Tensor& X) { return X + X.mH(); };

  // 计算diag(dP) / (2S)
  auto diagdP2S = is_complex ? dP.diagonal(0, -2, -1).div(2. * S) : Tensor{};

  // 计算dU = U * (sym(dP * S) / E) + i * Im(diag(dP)) / (2S)
  auto dU = [&] {
    auto dUaux = sym(dP * S.unsqueeze(-2)) / E;
    // 返回结果
    return dUaux + (self_or_result < 0).type_as(grad);
  };
    // 如果是复数运算，将 dUaux 增加对角矩阵 diagdP2S 的对角化结果
    if (is_complex) {
      dUaux = dUaux + diagdP2S.diag_embed();
    }
    // 返回 U 和 dUaux 的矩阵乘积作为结果
    return at::matmul(U, dUaux);
  }();
  // 如果 m 大于 n
  if (m > n) {
    // 计算 dU += (I_m - UU^H) dA V S^{-1}
    const auto dAVSinv = at::matmul(dA, V / S.unsqueeze(-2));
    dU = dU + dAVSinv - at::matmul(U, at::matmul(U.mH(), dAVSinv));

    // 如果 full_matrices 为真，调整 dU 的形状以修正全矩阵情况
    if (full_matrices) {
      auto shape = dU.sizes().vec();
      shape.end()[-1] = m - n;
      dU = at::cat({dU, dU.new_zeros(shape)}, /*dim=*/-1);
    }
  }

  // 计算 dVh = -sym(S dP) / E + i Im(diag(dP)) / (2S)
  // 注意：这里对 S 取反，因为它是方程中的最小张量
  auto dVh = [&] {
    auto dVhaux = sym(dP * (-S).unsqueeze(-1)) / E;
    // 如果是复数运算，将 dVhaux 增加对角矩阵 diagdP2S 的对角化结果
    if (is_complex) {
      dVhaux = dVhaux + diagdP2S.diag_embed();
    }
    // 返回 dVhaux 与 Vh 的矩阵乘积作为结果
    return at::matmul(dVhaux, Vh);
  }();
  // 如果 m 小于 n
  if (m < n) {
    // 计算 dVh += S^{-1} U^H dA (I_n - VV^H)
    const auto UHdASinv = at::matmul(U.mH() / S.unsqueeze(-1), dA);
    dVh = dVh + UHdASinv - at::matmul(at::matmul(UHdASinv, V), Vh);

    // 如果 full_matrices 为真，调整 dVh 的形状以修正全矩阵情况
    if (full_matrices) {
      auto shape = dVh.sizes().vec();
      shape.end()[-2] = n - m;
      dVh = at::cat({dVh, dVh.new_zeros(shape)}, /*dim=*/-2);
    }
  }

  // 返回三元组 (dU, dS, dVh) 作为最终结果
  return std::make_tuple(std::move(dU), std::move(dS), std::move(dVh));
  }

Tensor svd_backward(
    const Tensor& gU,                   // 输入参数：U 的梯度
    const Tensor& gS,                   // 输入参数：S 的梯度
    const Tensor& gVh,                  // 输入参数：V^H 的梯度
    const Tensor& U,                    // 输入参数：矩阵 U
    const Tensor& S,                    // 输入参数：奇异值 S
    return {};                          // 返回空字典，占位用

  }

  const auto m = U.sym_size(-2);        // 计算 U 矩阵的倒数第二维度大小
  const auto n = Vh.sym_size(-1);       // 计算 V^H 矩阵的倒数第一维度大小

  // 优化 svdvals 函数：gA = U @ diag(gS) @ Vh
  if (!gU.defined() && !gVh.defined()) {
    return m >= n ? at::matmul(U, gS.unsqueeze(-1) * Vh)
                  : at::matmul(U * gS.unsqueeze(-2), Vh);
  }
  // 此时至少一个 gU 或 gVh 已定义

  const bool is_complex = U.is_complex();  // 检查矩阵 U 是否为复数类型
  const auto skew = [](const Tensor& A) { return A - A.mH(); };  // 计算反对称部分的函数
  const auto UhgU = gU.defined() ? skew(at::matmul(U.mH(), gU)) : Tensor{};  // 计算 U^H @ gU 的反对称部分或返回空张量
  const auto VhgV = gVh.defined() ? skew(at::matmul(Vh, gVh.mH())) : Tensor{};  // 计算 V^H @ gVh 的反对称部分或返回空张量

  // 检查损失函数的不变性，即 Im(diag(U^H gU)) + Im(diag(V^H gV)) = 0
  if (is_complex) {
    const auto imdiag_UhgU =
        gU.defined() ? at::imag(UhgU.diagonal(0, -2, -1)) : at::zeros_like(S);  // 计算 U^H gU 的对角线虚部或返回与 S 相同形状的零张量
    const auto imdiag_VhgV =
        gVh.defined() ? at::imag(VhgV.diagonal(0, -2, -1)) : at::zeros_like(S);  // 计算 V^H gV 的对角线虚部或返回与 S 相同形状的零张量
    // 使用宽松的 atol 和 rtol，以避免误报
    TORCH_CHECK(
        at::allclose(imdiag_UhgU, -imdiag_VhgV, /*rtol=*/1e-2, /*atol=*/1e-2),
        "svd_backward: The singular vectors in the complex case are specified up to multiplication "
        "by e^{i phi}. The specified loss function depends on this phase term, making "
        "it ill-defined.");
  }

  // gA = ((U^H gU) / E) S +  S (((V^H gV) / E) + I o (gS + diag(U^H gU) / (2 *
  // S))
  Tensor gA = [&] {
    // ret holds everything but the diagonal of gA
    auto ret = [&] {
      const auto E = [&S] {
        const auto S2 = S * S;
        auto ret = S2.unsqueeze(-2) - S2.unsqueeze(-1);  // 计算矩阵 S 乘积的差
        // 将对角线上的元素设为 1，以免后续除零操作报错
        ret.diagonal(0, -2, -1).fill_(1);
        return ret;
      }();

      if (gU.defined()) {
        if (gVh.defined()) {
          return (UhgU * S.unsqueeze(-2) + S.unsqueeze(-1) * VhgV) / E;  // 计算复杂情况下的 gA
        } else {
          return (UhgU / E) * S.unsqueeze(-2);  // 计算简单情况下的 gA
        }
      } else { // gVh.defined();
        return S.unsqueeze(-1) * (VhgV / E);  // 计算简单情况下的 gA
      }
    }();
    // 填充对角线元素
    if (gS.defined()) {
      ret = ret + gS.diag_embed();
    }
    if (is_complex && gU.defined() && gVh.defined()) {
      ret = ret + (UhgU.diagonal(0, -2, -1) / (2. * S)).diag_embed();
    }
    return ret;
  }();

  if (m > n && gU.defined()) {
    // gA = [UgA + (I_m - UU^H)gU S^{-1}]V^H
    gA = at::matmul(U, gA);
    const auto gUSinv = gU / S.unsqueeze(-2);
    gA = gA + gUSinv - at::matmul(U, at::matmul(U.mH(), gUSinv));
    gA = at::matmul(gA, Vh);
  } else if (m < n && gVh.defined()) {
    //   gA = U[gA V^H + S^{-1} (gV)^H (I_n - VV^H)]
    gA = at::matmul(gA, Vh);
    const auto SinvgVh = gVh / S.unsqueeze(-1);
    gA = gA + SinvgVh - at::matmul(at::matmul(SinvgVh, Vh.mH()), Vh);
  }
    // 如果 m 大于等于 n，执行 U gA V^H 的矩阵乘法，结果赋给 gA
    gA = m >= n ? at::matmul(U, at::matmul(gA, Vh))
                // 如果 m 小于 n，执行 U gA V^H 的矩阵乘法，结果赋给 gA
                : at::matmul(at::matmul(U, gA), Vh);
// 本函数用于计算特征值分解反向传播的梯度
Tensor linalg_eig_backward(
    const Tensor& gL,  // 对特征值矩阵L的梯度
    const Tensor& gV,  // 对特征向量矩阵V的梯度
    const Tensor& L,   // 特征值矩阵L
    const Tensor& V,   // 特征向量矩阵V
    const bool is_hermitian,            // 是否是埃尔米特矩阵的标志
    const bool symeig_eigenvectors) {   // 是否计算特征向量的标志

  at::NoTF32Guard disable_tf32;  // 临时禁用 TF32 加速

  // 引用论文 https://arxiv.org/pdf/1701.00392.pdf 中的公式 4.77

  // 检查是否可以触发在 torch.symeig 的反向传播中
  TORCH_CHECK(
      symeig_eigenvectors,
      "linalg_eig_backward: torch.symeig(A, eigenvectors=False) is not differentiable. ",
      "Use torch.linalg.eigvalsh(A) instead.");

  // 如果 gL 和 gV 都未定义，则返回空张量
  if (!gL.defined() && !gV.defined()) {
    return {};
  }

  // 当 gV 未定义时，简化计算路径，用于 linalg.eigvals/eigvalsh 的情况
  if (!gV.defined()) {
    if (is_hermitian) {
      // 如果是埃尔米特矩阵，直接返回 V * gL.unsqueeze(-2) * V^H
      return at::matmul(V * gL.unsqueeze(-2), V.mH());
    } else {
      // 对于一般情况，返回 V^H * (gL.unsqueeze(-1) * V.mH()) 的解
      return at::linalg_solve(V.mH(), gL.unsqueeze(-1) * V.mH());
    }
  }

  // 计算 V^H * gV
  auto VhgV = at::matmul(V.mH(), gV);
  // 计算 V^H * gV 的对角线部分
  const auto diag_VhgV = VhgV.diagonal(0, -2, -1);

  // 如果 V 是复数类型并且 diag_VhgV 不是 Tensor 的子类，则进行额外检查
  if (V.is_complex() && !at::isTensorSubclassLike(diag_VhgV)) {
    // 检查损失函数对于 V -> V * e^{i\phi} 变换的不变性
    const auto imdiag_VhgV = at::imag(diag_VhgV);
    TORCH_CHECK(
        at::allclose(
            imdiag_VhgV,
            at::zeros_like(imdiag_VhgV),
            /*rtol=*/1e-2,
            /*atol=*/1e-2),
        is_hermitian ? "linalg_eigh_backward" : "linalg_eig_backward",
        ": The eigenvectors in the complex case are specified up to multiplication ",
        "by e^{i phi}. The specified loss function depends on this quantity, so it is ill-defined.");
  }

  // 根据是否是埃尔米特矩阵，进行不同的投影
  if (is_hermitian) {
    // 在 U(n) 的单位元的切空间上投影，即斜埃尔米特矩阵
    VhgV = 0.5 * (VhgV - VhgV.mH());
  } else {
    // 在具有列范数为1的复数矩阵的 V^H V 的切空间上投影
    VhgV = VhgV - at::matmul(V.mH(), V * at::real(diag_VhgV).unsqueeze(-2));
  }

  // 定义 lambda 函数 gA，用于后续计算
  auto gA = [&, VhgV = std::move(VhgV)] {
    // 定义 Econj 矩阵
    auto Econj = [&L] {
      auto Lconj = L.conj();  // L 的共轭
      auto ret = Lconj.unsqueeze(-2) - Lconj.unsqueeze(-1);  // 计算差值
      ret.diagonal(0, -2, -1).fill_(1.);  // 对角线上填充 1
      return ret;
    }();

    // 返回 VhgV 除以 Econj
    auto ret = VhgV.div_(Econj);


这部分代码包含了函数 `linalg_eig_backward` 的详细注释，解释了每一行代码的作用和背后的数学推导或算法逻辑。
    // 如果 gL 已定义
    if (gL.defined()) {
      // 对于 CompositeCompliance，如果 gL 是子类但 ret 是常规张量，
      // 则使用 diagonal_scatter 的 out-of-place 版本进行对角线复制
      if (at::isTensorSubclassLike(gL)) {
        // 使用 gL 在维度 0 上进行 diagonal_scatter 操作，将结果赋给 ret
        ret = ret.diagonal_scatter(gL, 0, -2, -1);
      } else {
        // 在维度 0 上提取 ret 的对角线并用 gL 进行 in-place 复制
        ret.diagonal(0, -2, -1).copy_(gL);
      }
    }
    // 返回 ret

  // Conjugate by V^{-H}
  // 如果是 Hermitian 矩阵
  if (is_hermitian) {
    // 返回 V * gA * V 的共轭转置
    return at::matmul(V, at::matmul(gA, V.mH()));
  } else {
    // 返回 V.mH() 的逆乘以 gA 乘以 V.mH()
    return at::linalg_solve(V.mH(), at::matmul(gA, V.mH()));
  }
}

std::tuple<Tensor, Tensor> linalg_eig_jvp(
    const Tensor& dA,
    const Tensor& L,
    const Tensor& V,
    const bool is_hermitian) {
  at::NoTF32Guard disable_tf32;
  // 设置一个上下文管理器，禁用 TF32 模式，确保使用双精度浮点运算

  // https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
  // https://arxiv.org/pdf/1701.00392.pdf 中的公式 (4.60) 和 (4.63)
  // 注意：这些论文中的公式都不正确，因为它们未假设特征向量具有单位范数。
  // 因此，它们缺少了 dV dL = diag(dP) dV = dX - V Re(diag V^H dX)) 中的对角项，
  // 其中 dP = V^{-1} dA V, dX = V ((dP - diag(dP)) / E), E_{ij} = L_j - L_i if i != j, 1 otherwise

  // 前提条件：如果 is_hermitian == true，则 dA 是 Hermite 矩阵
  const auto to_complex = [](const Tensor& A) {
    return A.to(c10::toComplexType(A.scalar_type()));
  };

  const auto dP = is_hermitian
      ? at::matmul(at::matmul(V.mH(), dA), V)
      : at::linalg_solve(V, at::matmul(to_complex(dA), V));
  // 计算 dP = V^H dA V 或 V^-1 dA V，取决于是否是 Hermitian 矩阵
  auto dL = is_hermitian && dA.is_complex() ? at::real(dP.diagonal(0, -2, -1))
                                            : dP.diagonal(0, -2, -1);
  // 计算 dL，如果是 Hermitian 并且 dA 是复数，则为 dP 对角线的实部，否则为 dP 对角线

  auto dV = [&dP, &V, &L, is_hermitian] {
    auto dX = [&] {
      auto ret = dP / (L.unsqueeze(-2) - L.unsqueeze(-1));
      ret.diagonal(0, -2, -1).zero_();
      ret = at::matmul(V, ret);
      return ret;
    }();
    // 计算 dX = dP / (L.unsqueeze(-2) - L.unsqueeze(-1))，然后用 V 乘以结果

    if (is_hermitian) {
      return dX;
    } else {
      return dX -
          V *
          at::real(at::matmul(V.mH(), dX).diagonal(0, -2, -1)).unsqueeze(-2);
    }
    // 如果是 Hermitian，则直接返回 dX；否则，计算 dX - V Re(V^H dX) 的对角线的实部
  }();
  // 根据是否是 Hermitian，计算 dV

  return std::make_pair(std::move(dL), std::move(dV));
  // 返回 dL 和 dV 的元组
}

Tensor linalg_lstsq_jvp(
    const Tensor& A,
    const Tensor& B,
    const Tensor& dA,
    const Tensor& dB) {
  at::NoTF32Guard disable_tf32;
  // 设置一个上下文管理器，禁用 TF32 模式，确保使用双精度浮点运算

  auto pinvA = at::linalg_pinv(A);
  // 计算 A 的伪逆 pinvA
  auto dpinvA = pinv_jvp(A, pinvA, dA);
  // 计算 pinvA 的 JVP（Jacobi 乘积）

  auto dX = dpinvA.matmul(B) + pinvA.matmul(dB);
  // 计算最小二乘问题的 JVP

  return dX;
  // 返回结果 dX
}

std::tuple<Tensor, Tensor> linalg_lstsq_backward(
    const Tensor& gX_,
    const Tensor& A,
    const Tensor& B_,
    const std::array<bool, 2>& grad_input_mask) {
  at::NoTF32Guard disable_tf32;
  // 设置一个上下文管理器，禁用 TF32 模式，确保使用双精度浮点运算

  auto A_requires_grad = grad_input_mask[0];
  auto B_requires_grad = grad_input_mask[1];
  if (!gX_.defined() || (!A_requires_grad && !B_requires_grad)) {
    return {};
  }
  // 如果 gX_ 未定义或者 A 和 B 都不需要梯度，则返回空元组

  const bool vector_case = at::native::linalg_solve_is_vector_rhs(A, B_);
  // 检查是否是向量情况（右侧向量）

  const auto vector_to_matrix = [vector_case](const Tensor& X) {
    return vector_case ? X.unsqueeze(-1) : X;
  };
  // 定义一个函数，根据 vector_case 将向量转换为矩阵

  const auto matrix_to_vector = [vector_case](const Tensor& X) {
    return vector_case ? X.squeeze(-1) : X;
  };
  // 定义一个函数，根据 vector_case 将矩阵转换为向量

  auto gX = vector_to_matrix(gX_);
  auto B = vector_to_matrix(B_);
  // 将 gX_ 和 B_ 转换为矩阵形式

  Tensor pinvA = at::linalg_pinv(A);
  // 计算 A 的伪逆 pinvA
  Tensor A_grad, B_grad;
  if (A_requires_grad) {
    auto pinvA_grad = gX.matmul(B.mH());
    // 计算 A 的梯度，使用 gX 和 B 的共轭转置
    A_grad = pinv_backward(pinvA_grad, pinvA, A);
    // 计算 pinvA 的梯度
  }

  if (B_requires_grad) {
    // 如果 B 需要梯度
    // Equivalent to
    // B_grad = std::get<0>(at::linalg_lstsq(A.mH(), gX, rcond, driver));
    // but we avoid this approach as `gelsy` is non-deterministic
    // 避免使用 `gelsy` 方法，我们采用以下方法计算 B_grad
    B_grad = pinvA.matmul(dB);
    // 计算 B 的梯度
  }

  return std::make_tuple(std::move(A_grad), std::move(B_grad));
  // 返回 A_grad 和 B_grad 的元组
}
    // 计算矩阵乘积 pinvA.mH() 和向量 gX 的乘积，并将结果展开为向量
    B_grad = matrix_to_vector(pinvA.mH().matmul(gX));
  }

  // 返回包含 A_grad 和 B_grad 的元组
  return std::make_tuple(A_grad, B_grad);
}
// 结束函数 linalg_qr_jvp

std::tuple<Tensor, Tensor> linalg_qr_jvp(
    const Tensor& dA,
    const Tensor& Q,
    const Tensor& R,
    const c10::string_view mode) {
  // 计算 Jacobi 矩阵乘积的 QR 分解
  // dA = dQR + QdR
  //
  // Case m >= n
  // 当 m >= n 时
  // 可以用 dR 的函数来表示 dQ
  // dQ = dAR^{-1} - QdRR^{-1}
  // 然后我们有
  // Q^H dA R^{-1} = Q^HdQ + dRR^{-1}
  // 其中 Q^HdQ 是反对称的，dRR^{-1} 是上三角的
  // 定义 sym(X) = X + X^H
  // sym(dRR^{-1}) = sym(Q^H dA R^{-1})
  // 并定义 syminv(X) = triu(X) - 0.5 * diag(X) 作为 sym 的逆操作
  // Case m < n
  // 当 m < n 时
  // 将 dR 表示为 dQ 的函数
  // dR = Q^H dA - Q^H dQ R
  // 令 X_1 为矩阵 X \in C^{m x n} 的主要 m x m 子矩阵
  // Q^H A_1 R_1^{-1} = Q^H dQ + dR_1 R_1^{-1}
  // 定义 trilIm(X) = X.tril(-1) + i * Im diag(X)
  // trilIm(Q^H dQ) = trilIm(Q^H A_1 R_1^{-1})
  // 并定义 trilIminv(X) = X - X^H - i*Im diag(X) 作为 trilIm 的逆操作
  at::NoTF32Guard disable_tf32;

  auto [compute_q, reduced] = at::native::_parse_qr_mode(mode);

  TORCH_CHECK(
      compute_q,
      "The derivative of linalg.qr depends on Q, which is not computed when "
      "mode='r'. Please use linalg.qr(A, mode='reduced') if you are "
      "going to differentiate through linalg.qr.");
  auto m = dA.size(-2);
  auto n = dA.size(-1);

  TORCH_CHECK(
      reduced || m <= n,
      "The QR decomposition is not differentiable when "
      "mode='complete' and nrows > ncols.");
  if (m >= n) {
    const auto sym = [](const Tensor& X) { return X + X.mH(); };
    const auto syminv = [](const Tensor& X) {
      auto ret = X.triu();
      ret.diagonal(0, -2, -1).mul_(0.5);
      return ret;
    };
    auto dARinv =
        at::linalg_solve_triangular(R, dA, /*upper=*/true, /*left=*/false);
    auto dR = syminv(sym(Q.mH().matmul(dARinv)));
    auto dQ = dARinv - Q.matmul(dR);
    dR = dR.matmul(R);
    return std::make_tuple(std::move(dQ), std::move(dR));
  } else {
    const auto trilim = [](const Tensor& X) {
      if (X.is_complex()) {
        auto ret = X.tril();
        at::real(ret.diagonal(0, -2, -1)).zero_();
        return ret;
      } else {
        return X.tril(-1);
      }
    };
    const auto triliminv = [](const Tensor& X) {
      if (X.is_complex()) {
        auto ret = X - X.mH();
        ret.diagonal(0, -2, -1).mul_(0.5);
        return ret;
      } else {
        return X - X.mT();
      }
    };

    auto QHdA = Q.mH().matmul(dA);
    auto QHdA1Rinv = at::linalg_solve_triangular(
        R.narrow(-1, 0, m),
        QHdA.narrow(-1, 0, m),
        /*upper=*/true,
        /*left=*/false);
    auto dQ = triliminv(trilim(QHdA1Rinv)));
    auto dR = QHdA - dQ.matmul(R);
    dQ = Q.matmul(dQ);
    # 使用 std::make_tuple 函数创建一个包含两个元素的 tuple
    return std::make_tuple(std::move(dQ), std::move(dR));
  }
}
// 结束函数体的右花括号，表示 linalg_qr_backward 函数的结束

Tensor linalg_qr_backward(
    const Tensor& gQ,  // 输入参数：梯度 gQ
    const Tensor& gR,  // 输入参数：梯度 gR
    const Tensor& Q,   // 输入参数：矩阵 Q
    const Tensor& R,   // 输入参数：矩阵 R
    const c10::string_view mode) {  // 输入参数：QR 分解模式

  // 禁用 TF32 加速器（不在 TorchScript 上下文中）
  at::NoTF32Guard disable_tf32;

  // 解析 QR 模式，获取是否计算 Q 和是否为 reduced 模式
  auto [compute_q, reduced] = at::native::_parse_qr_mode(mode);

  // 检查是否计算 Q，因为当 mode='r' 时，不计算 Q 的话导数不可用
  TORCH_CHECK(
      compute_q,
      "The derivative of linalg.qr depends on Q, which is not computed when "
      "mode='r'. Please use linalg.qr(A, mode='reduced') if you are "
      "going to differentiate through linalg.qr.");

  // 获取矩阵 Q 和 R 的尺寸
  auto m = Q.sym_size(-2);  // Q 的行数
  auto n = R.sym_size(-1);  // R 的列数

  // 检查在完整模式下是否可导，即 nrows > ncols 时不可导
  TORCH_CHECK(
      reduced || m <= n,
      "The QR decomposition is not differentiable when "
      "mode='complete' and nrows > ncols.");

  // 如果 gQ 和 gR 都未定义，则返回空张量
  if (!gQ.defined() && !gR.defined()) {
    return {};
  }

  Tensor gA;  // 定义梯度结果张量 gA

  // 根据情况计算 gA
  if (gQ.defined()) {
    if (gR.defined()) {
      // Case m >= n：计算 gA
      gA = gR.matmul(R.mH()) - Q.mH().matmul(gQ);
    } else {
      gA = -Q.mH().matmul(gQ);
    }
  } else {
    gA = gR.matmul(R.mH());
  }

  // 根据不同的情况处理 gA
  if (m >= n) {
    // 当 m >= n 时的处理方式

    // 定义函数 syminvadj，对输入张量 X 进行对称逆伴随运算
    const auto syminvadj = [](const Tensor& X) {
      auto ret = X + X.mH();  // X 与其共轭转置的和
      at::real(ret.diagonal(0, -2, -1)).mul_(0.5);  // 实部对角线元素乘以 0.5
      return ret;
    };

    // 计算 gA 的更新
    gA = Q.matmul(syminvadj(gA.triu()));  // Q 与 syminvadj(gA.triu()) 的乘积
    if (gQ.defined()) {
      gA = gA + gQ;  // 加上 gQ
    }
    gA = at::linalg_solve_triangular(
        R.mH(), gA, /*upper*/ false, /*left*/ false);  // 使用 R.mH() 求解三角线性方程组
    return gA;
  } else {
    // 当 m < n 时的处理方式

    // 定义函数 trilImInvAdjSkew，对输入张量 X 进行下三角部分的伴随斜对称逆操作
    auto trilImInvAdjSkew = [](const Tensor& X) {
      auto ret = (X - X.mH()).tril();  // (X - X 的共轭转置) 的下三角部分
      if (X.is_complex()) {
        at::imag(ret.diagonal(0, -2, -1)).mul_(0.5);  // 如果 X 是复数类型，虚部对角线元素乘以 0.5
      }
      return ret;
    };

    // 计算 gA 的更新
    gA = Q.matmul(trilImInvAdjSkew(-gA));  // Q 与 trilImInvAdjSkew(-gA) 的乘积
    gA = at::linalg_solve_triangular(
        R.narrow_symint(-1, 0, m).mH(), gA, /*upper*/ false, /*left*/ false);  // 使用 R 的子部分求解三角线性方程组
    auto shape = R.sym_sizes().vec();  // 获取 R 的尺寸信息
    shape.end()[-1] = n - m;  // 调整 shape 的最后一个维度大小
    gA = at::cat({gA, gA.new_zeros_symint(shape)}, /*dim=*/-1);  // 在最后一个维度上拼接 gA 和新的零张量
    if (gR.defined()) {
      gA = gA + Q.matmul(gR);  // 加上 Q 与 gR 的乘积
    }
    return gA;
  }
}

// 基于：
// Mathias, Roy.
// A Chain Rule for Matrix Functions and Applications.
// SIAM J. Matrix Anal. Appl. 17 (1996): 610-620.

template <typename func_t>
// 定义一个模板函数，用于计算分析矩阵函数的微分或其伴随的微分
Tensor differential_analytic_matrix_function(
    const Tensor& self,  // 输入矩阵张量
    const Tensor& grad,  // 输入梯度张量
    const func_t& matrix_function,  // 矩阵函数
    const bool adjoint  // 是否计算伴随（true表示计算伴随）
) {
  // 根据给定的分析矩阵函数，计算微分（正向自动微分）或其伴随的微分（反向自动微分）
  auto A = adjoint ? self.transpose(-2, -1).conj() : self;
  // 计算元梯度的尺寸
  auto meta_grad_sizes = A.sym_sizes().vec();
  meta_grad_sizes[A.dim() - 2] *= 2;
  meta_grad_sizes[A.dim() - 1] *= 2;

  auto n = A.sym_size(-1);
  Tensor meta_grad;
  // 对于复合兼容性，我们不能将子类复制到常规张量中，所以使用等效输出的原地操作。
  // 注意：不能直接使用 `new_zeros`，因为 `A` 和 `grad` 可能是张量的子类，
  // 我们不想假设要选择哪一个来创建输出缓冲区。
  // 例如，如果两者都是不同级别的批量张量。
  if (areAnyTensorSubclassLike({A, grad})) {
    // 如果存在任何子类张量，则合并它们以创建元梯度
    meta_grad = at::cat(
        {at::cat({A, grad}, -1),
         at::cat({at::zeros_like(A), std::move(A)}, -1)},
        -2);
  } else {
    // 否则创建零填充的元梯度
    meta_grad = at::zeros_symint(meta_grad_sizes, grad.options());
    meta_grad.narrow_symint(-2, 0, n).narrow_symint(-1, 0, n).copy_(A);
    meta_grad.narrow_symint(-2, n, n).narrow_symint(-1, n, n).copy_(A);
    meta_grad.narrow_symint(-2, 0, n).narrow_symint(-1, n, n).copy_(grad);
  }

  // 应用矩阵函数到元梯度，然后截取对应区域返回结果
  return matrix_function(meta_grad).narrow_symint(-2, 0, n).narrow_symint(
      -1, n, n);
}

// 计算矩阵指数函数的微分
Tensor linalg_matrix_exp_differential(
    const Tensor& self,  // 输入矩阵张量
    const Tensor& grad,  // 输入梯度张量
    bool adjoint  // 是否计算伴随（true表示计算伴随）
) {
  at::NoTF32Guard disable_tf32;

  // 调用通用的分析矩阵函数计算微分
  return differential_analytic_matrix_function(
      self, grad, at::linalg_matrix_exp, /* adjoint */ adjoint);
}

template <typename F1, typename F2, typename... Ts>
// 掩码映射函数，接受掩码、两个函数和一个（可变长度的）张量列表，并根据掩码应用不同的函数创建新的张量
Tensor masked_fmap(
    const Tensor& mask,  // 掩码张量
    const F1& f1,  // 用于掩码为真的情况的函数
    const F2& f2,  // 用于掩码为假的情况的函数
    const Tensor& t,  // 第一个张量，作为形状模板
    const Ts&... ts  // 其他张量参数
) {
  // 此函数用于在有公式适用于所有非奇异输入和另一种适用于奇异输入时创建与第一个张量列表元素相同形状的新张量
  // 例如，det_backward 的应用场景

  // 预条件：确保 t 的符号元素数量不为 0
  TORCH_INTERNAL_ASSERT(t.sym_numel() != 0);
  // 根据掩码索引 t
  auto t_masked = t.index({mask});
  auto n = t_masked.sym_numel();
  if (n == t.sym_numel()) {
    // 如果掩码应用于所有元素，则使用 f1 处理
    return f1(t, ts...);
  } else if (n == 0) {
    // 如果掩码未应用于任何元素，则使用 f2 处理
    return f2(t, ts...);
  } else {
    // 否则，创建一个与 t 形状相同的空张量
    // ret = torch.empty_like(t)
    // 创建一个逻辑非掩码，将输入掩码取反，得到不满足条件的索引
    auto not_mask = mask.logical_not();
    // 返回一个与输入张量 t 相同形状的空张量，用作最终结果的容器
    return at::empty_like(t)
        // 使用掩码对应的索引位置，调用函数 f1 处理对应的张量子集 t_masked 和对应的 ts 子集
        .index_put_({mask}, f1(t_masked, ts.index({mask})...))
        // 使用取反后的掩码对应的索引位置，调用函数 f2 处理对应的张量子集 t 和对应的 ts 子集
        .index_put_(
            {not_mask}, f2(t.index({not_mask}), ts.index({not_mask})...));
    }
}

Tensor linalg_det_jvp(
    const Tensor& dA,
    const Tensor& det,
    const Tensor& LU,
    const Tensor& pivots,
    const bool use_A_T) {
  // 计算 Jacobian-Vector Product (JVP)：(d det)_A(E) = tr(A^{-1}E)*det
  // 在这里，我们利用行列式是可微的事实来近似奇异输入的梯度。
  // 由于我们从不对前向自动微分求导，因此不需要处理更深层次的梯度，就像在 grad_backward 中一样。
  
  // 获取机器精度 epsilon
  auto eps = at::native::_get_epsilon(c10::toRealValueType(LU.scalar_type()));
  // 对 LU 矩阵进行优化处理，以处理对角线元素为零的情况
  auto LU_ =
      LU + at::diag_embed(at::where(LU.diagonal(0, -2, -1) == 0., eps, 0.));
  // 计算 A^{-1}E，其中 E 是输入的变化量 dA
  auto AinvE =
      at::linalg_lu_solve(LU_, pivots, dA, /*left=*/true, /*adjoint=*/use_A_T);
  // 返回 A^{-1}E 的对角线元素之和乘以行列式 det
  return AinvE.diagonal(0, -2, -1).sum(-1) * det;
}

Tensor linalg_det_backward(
    const Tensor& grad,
    const Tensor& det,
    const Tensor& A,
    const Tensor& LU,
    const Tensor& pivots) {
  // 禁用 TF32 模式的上下文管理器
  at::NoTF32Guard disable_tf32;
  // 当输入的梯度 grad 未定义，或者 A 是奇异矩阵时，返回空张量
  if (!grad.defined() || A.sym_numel() == 0) {
    return {};
  }

  // 梯度 G 是满足 A.mH G = det(A).conj() * grad * I 的矩阵
  auto d_diag = grad * det.conj();
  // 优化：将 d 调整为 F 转置，因为 lu_solve 期望这种形式
  auto d = at::diag_embed(d_diag.unsqueeze(-1).expand_as(pivots)).mT();
  // 获取机器精度 epsilon
  auto eps = at::native::_get_epsilon(c10::toRealValueType(LU.scalar_type()));

  // 如果不需要计算更高阶梯度
  if (!at::GradMode::is_enabled()) {
    // 当 A 是可逆矩阵时，根据解方程 AX = det.conj() * det * I 的解，计算 LU_ 矩阵
    // 如果 A 不可逆，我们可以对 LU 分解施加微扰，并使用得到的矩阵作为非奇异近似
    auto LU_ =
        LU + at::diag_embed(at::where(LU.diagonal(0, -2, -1) == 0., eps, 0.));
    // 检查是否可以使用 A 转置来优化计算
    auto use_A_T = A.is_contiguous() && !A.is_complex();
    return at::linalg_lu_solve(
        LU_, pivots, d, /*left=*/true, /*adjoint=*/!use_A_T);
  } else {
    // 如果需要计算更高阶梯度，则需要重新计算 LU 分解，以便自动求导能够正确计算相对于 A 的梯度
    auto non_singular =
        [](const Tensor& A, const Tensor& d, const Tensor& /*grad*/) {
          return at::linalg_solve(A.mH(), d);
        };

    // 求导可以通过注意到行列式的导数的梯度可以用伴随矩阵来表示来显式计算。
    // 对于奇异矩阵的伴随矩阵可以参考 https://nhigham.com/2020/06/16/what-is-the-adjugate-of-a-matrix/
    // 定义一个 lambda 函数 singular，用于计算特定输入张量的梯度
    auto singular = [](const Tensor& A,
                       const Tensor& /*d*/,  // 第二个参数未使用，仅起占位作用
                       const Tensor& grad) {  // 输入张量 A 和其梯度 grad
      // 对张量 A 进行奇异值分解（SVD）
      auto [U, S, Vh] = at::linalg_svd(A);
      // 计算 alpha，它是 U 和 Vh 的行列式的共轭乘积乘以 grad
      auto alpha = (at::linalg_det(U) * at::linalg_det(Vh)).conj() * grad;
      // 调用 prod_safe_zeros_backward 函数，生成一个梯度相关的新张量 D
      auto D = prod_safe_zeros_backward(alpha.unsqueeze(-1), S, S.dim() - 1);
      // 返回 U * D.unsqueeze(-2) * Vh 的矩阵乘积结果
      return (U * D.unsqueeze(-2)).matmul(Vh);
    };

    // 如果输入张量 A, d, grad 中至少有一个是张量子类，直接调用 singular 函数
    if (areAnyTensorSubclassLike({A, d, grad})) {
      return singular(A, d, grad);
    } else {
      // 否则，根据条件 det.abs() < 100. * eps，选择调用 masked_fmap 函数的不同分支
      return masked_fmap(
          det.abs() < 100. * eps,  // 根据 det 的绝对值与阈值比较，生成掩码
          singular,  // 在掩码条件成立时调用 singular 函数
          non_singular,  // 在掩码条件不成立时调用 non_singular 函数
          A, d, grad);  // 传递参数 A, d, grad 给 masked_fmap 函数
    }
  }
}

std::tuple<Tensor, Tensor> slogdet_jvp(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& dA,
    const Tensor& sign,
    const bool use_A_T) {
  // 如果矩阵是奇异的话，无需单独处理，因为这个函数在奇异矩阵上不可导
  // 使用 LU 分解求解线性方程组 LUx = dA，其中 LU 是 LU 分解后的矩阵，pivots 是 LU 分解的置换向量
  auto trAinvE = at::linalg_lu_solve(LU, pivots, dA, /*left*/ true, use_A_T)
                     .diagonal(0, -2, -1)  // 取解的对角线元素
                     .sum(-1);  // 求和得到结果
  if (LU.is_complex()) {
    auto i = c10::complex<double>{0.0, 1.0};
    // 如果 LU 是复数类型的，返回一个复数元组 (imag(trAinvE) * (i * sign), real(trAinvE))
    return std::make_tuple(at::imag(trAinvE) * (i * sign), at::real(trAinvE));
  } else {
    // 如果 LU 是实数类型的，返回一个实数元组 (_efficientzerotensor(sign.sizes(), sign.options()), trAinvE)
    return std::make_tuple(
        at::_efficientzerotensor(sign.sizes(), sign.options()), trAinvE);
  }
}

Tensor slogdet_backward(
    const Tensor& grad_sign,
    const Tensor& grad_logabsdet,
    const Tensor& A,
    const Tensor& signdet,
    const Tensor& LU,
    const Tensor& pivots) {
  // 计算复数情况下的梯度，实数情况下的梯度可以由此推导
  // Forward AD
  // d (logabsdet)_A(E) = Re(tr(A^{-1}E))
  // d (signdet)_A(E) = sgn * Im(tr(A^{-1}E)) * i
  // So
  // d (logabsdet)*_A(g) = gA^{-H}
  // Now, to compute the adjoint of d(signdet), note that
  // Re(z * Im(w)) = Re(-Re(z)iw)
  // So, let g \in C,
  // <g, d(signdet)_A(E)> = Re(g.conj() * sgn * i * Im(A^{-1}E))
  //                      = Re(Re(g.conj() * sgn * i) * -i * A^{-1}E)
  //                      = Re(Im(g.conj() * sgn) * i * A^{-1}E)
  //                      = <Im(g.conj() * sgn) * -i * A^{-H}, E>
  // As such,
  // (d slogabs)*_A(g_sign, g_abs) = (g_abs - g_sign.conj() * sgn) * A^{-H}

  if (!grad_sign.defined() && !grad_logabsdet.defined()) {
    // 如果 grad_sign 和 grad_logabsdet 都未定义，返回空张量
    return {};
  }

  auto is_complex = A.is_complex();

  // 在实数情况下，grad_sign 总是零
  if (!is_complex && !grad_logabsdet.defined()) {
    // 如果是实数情况且 grad_logabsdet 未定义，返回空张量
    return {};
  }

  auto g = grad_logabsdet;
  if (is_complex) {
    if (grad_sign.defined()) {
      auto i = c10::complex<double>{0.0, 1.0};
      if (g.defined()) {
        // 如果 grad_sign 定义了，则更新 g = g - i * imag(grad_sign.conj() * signdet)
        g = g - i * at::imag(grad_sign.conj() * signdet);
      } else {
        // 否则 g = -i * imag(grad_sign.conj() * signdet)
        g = -i * at::imag(grad_sign.conj() * signdet);
      }
    } else {
      // 显式将 g 转换为复数类型
      g = g.to(A.scalar_type());
    }
  }

  // 不需要单独处理奇异情况（与 det 函数不同），因为这个函数在奇异矩阵上不可导
  // 优化，使其与 lu_solve 函数期望的结果形状一致
  auto d = at::diag_embed(g.unsqueeze(-1).expand_as(pivots)).mT();
  if (!at::GradMode::is_enabled()) {
    auto use_A_T = A.is_contiguous() && !A.is_complex();
    // 如果不处于梯度计算模式下，使用 LU 分解求解线性方程组 LUx = d，其中 LU 是 LU 分解后的矩阵，pivots 是 LU 分解的置换向量
    return at::linalg_lu_solve(
        LU, pivots, d, /*left=*/true, /*adjoint=*/!use_A_T);
  } else {
    // 如果想要计算进一步的梯度，需要重新计算 LU 分解，以便 autograd 根据 A 计算正确的梯度（参见 solve_backward 函数）
    return at::linalg_solve(A.mH(), d);
  }
}

// 参考文献:
// https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
//`
// Sec. 2.3.1 Matrix inverse product
std::tuple<Tensor, Tensor> triangular_solve_backward(
    const Tensor& grad_x,  // Gradient of x
    const Tensor& grad_m,  // Gradient of m
    const Tensor& b,       // Tensor b
    const Tensor& a,       // Tensor a
    const Tensor& x,       // Tensor x
    const bool upper,      // Upper triangular flag
    const bool transpose,  // Transpose flag
    const bool unitriangular,  // Unit triangular flag
    std::array<bool, 2> output_mask) {  // Output mask
  at::NoTF32Guard disable_tf32;  // Disable TF32 if supported
  Tensor grad_b, grad_a;  // Gradient tensors
  if (grad_x.defined() || grad_m.defined()) {
    if (grad_x.defined()) {
      // Compute gradient of b using triangular solve
      grad_b = std::get<0>(
          grad_x.triangular_solve(a.conj(), upper, !transpose, unitriangular));
      if (output_mask[1]) {
        // Compute gradient of a based on flags and matrix operations
        grad_a =
            transpose ? -x.conj().matmul(grad_b.mT()) : -grad_b.matmul(x.mH());
        if (upper) {
          grad_a = grad_a.triu((int)unitriangular);  // Upper triangular adjustment
        } else {
          grad_a = grad_a.tril(-((int)unitriangular));  // Lower triangular adjustment
        }
      }
    }
    if (!grad_a.defined()) {
      grad_a = at::zeros({1}, a.options()).expand_as(a);  // Initialize grad_a if undefined
    }
    if (!grad_b.defined()) {
      grad_b = at::zeros({1}, b.options()).expand_as(b);  // Initialize grad_b if undefined
    }
    if (output_mask[1] && grad_m.defined()) {
      grad_a = grad_a.add(grad_m);  // Add grad_m to grad_a if specified
    }
  }
  return std::tuple<Tensor, Tensor>{grad_b, grad_a};  // Return gradients of b and a
}

Tensor triangular_solve_jvp(
    const Tensor& X,       // Tensor X
    const Tensor& A,       // Tensor A
    const Tensor& dA,      // Tensor dA
    const Tensor& dB,      // Tensor dB
    const bool upper,      // Upper triangular flag
    const bool transpose,  // Transpose flag
    const bool unitriangular) {  // Unit triangular flag
  return generic_solve_jvp(
      [&](const Tensor& A, const Tensor& dB, const Tensor& dA_contrib) {
        // Compute JVP using triangular solve
        return std::get<0>(at::triangular_solve(
            dB - dA_contrib, A, upper, transpose, unitriangular));
      },
      X,
      A,
      dA,
      dB);
}

Tensor linalg_solve_triangular_forward_AD(
    const Tensor& A_t,          // Tensor A_t
    const Tensor& B_t,          // Tensor B_t
    const Tensor& A,            // Tensor A
    const Tensor& X,            // Tensor X
    const bool upper,           // Upper triangular flag
    const bool left,            // Left triangular flag
    const bool unitriangular) {  // Unit triangular flag
  at::NoTF32Guard disable_tf32;  // Disable TF32 if supported
  // Projected A_t based on upper and unitriangular flags
  const Tensor proj_A_t = upper ? A_t.triu(static_cast<int>(unitriangular))
                                : A_t.tril(-static_cast<int>(unitriangular));
  const Tensor X_t =
      B_t - (left ? at::matmul(proj_A_t, X) : at::matmul(X, proj_A_t));  // Compute X_t
  return at::linalg_solve_triangular(A, X_t, upper, left, unitriangular);  // Return solve_triangular result
}

std::tuple<Tensor, Tensor> linalg_solve_triangular_backward(
    const Tensor& grad,         // Gradient tensor
    const Tensor& A,            // Tensor A
    const Tensor& X,            // Tensor X
    const bool upper,           // Upper triangular flag
    const bool left,            // Left triangular flag
    const bool unitriangular,   // Unit triangular flag
    // 禁用 Torch 的 TF32 加速
    at::NoTF32Guard disable_tf32;
    // 根据输出掩码确定是否需要计算 A 和 B 的梯度
    const bool A_requires_grad = output_mask[0];
    const bool B_requires_grad = output_mask[1];
    
    // [Note: Forward / Backward AD solve_triangular]
    // 假设 left=true 简化讨论，A^{-1}B 的求解器
    //
    // Forward AD:
    // 如果 f(A) = A^{-1}，对方程 A^{-1}A = I_n 求导得到
    // (df)_A(E) = -A^{-1}EA^{-1}
    // 因此，如果 g(A,B) = A^{-1}B，
    // (dg)_(A,B)(E_A, E_B) = -A^{-1}E_AA^{-1}B + A^{-1}E_B
    //                  = A^{-1}(E_B - E_AX)
    
    // Backward AD:
    // 用 G_A, G_B 表示梯度，我们求解得到
    // G_B = A^{-H}G_X
    // G_A = -A^{-H}G_XX^H = -G_B X^H
    //
    // 注意：前向和反向传播都不需要存储 B
    //
    // 这些公式适用于线性方程组的一般求解器。
    // 现在让我们证明当 A 是三角矩阵时，G_A 是上述公式的投影，即简单地在上三角或下三角中取值。
    // 这是因为三角矩阵形成一个向量空间，在任意点的切空间就是三角矩阵的空间。该结论类似于 [Note: eigh backward] 最后的推理。
    // 对于 `unitriangular`，类似的推理也适用，只是在这种情况下切空间是下三角矩阵，对角线为零。
    
    // 如果没有定义梯度或者 A 和 B 都不需要梯度，则返回空张量元组
    if (!grad.defined() || (!A_requires_grad && !B_requires_grad)) {
        return std::make_tuple(Tensor{}, Tensor{});
    }
    
    // 计算 G_B
    const Tensor A_H = A.mH();
    const Tensor G_B = at::linalg_solve_triangular(A_H, grad, !upper, left, unitriangular);
    
    // 如果需要计算 A 的梯度
    if (A_requires_grad) {
        const Tensor X_H = X.mH();
        // 计算 G_A
        Tensor G_A = left ? -at::matmul(G_B, X_H) : -at::matmul(X_H, G_B);
        // 根据 upper 和 unitriangular 条件将 G_A 裁剪为上三角或下三角
        G_A = upper ? G_A.triu(static_cast<int>(unitriangular))
                    : G_A.tril(-static_cast<int>(unitriangular));
        // 返回 G_A 和需要计算 B 梯度时的 G_B，否则返回空张量
        return std::make_tuple(G_A, B_requires_grad ? G_B : Tensor{});
    } else {
        // 如果不需要计算 A 的梯度，则返回空张量和 G_B
        return std::make_tuple(Tensor{}, G_B);
    }
}

std::tuple<Tensor, Tensor> cholesky_solve_backward(
    const Tensor& grad_x,
    const Tensor& self,
    const Tensor& input2,
    const Tensor& result,
    const bool upper,
    std::array<bool, 2> output_mask) {
  // 禁用 TF32，确保使用的是标准的计算路径
  at::NoTF32Guard disable_tf32;
  Tensor grad_self, grad_input2;
  if (grad_x.defined()) {
    // 计算 Cholesky 解的反向传播
    grad_self = grad_x.cholesky_solve(input2, /*upper=*/upper);

    if (output_mask[1]) {
      // 计算公共项，涉及 grad_self 和 result 的转置共轭
      Tensor common_term = at::matmul(grad_self, result.mH());
      common_term = common_term + common_term.mH();

      if (upper) {
        // 计算 input2 的梯度，考虑上三角或下三角的情况
        grad_input2 = -at::matmul(input2, common_term);
      } else {
        grad_input2 = -at::matmul(common_term, input2);
      }
    }
  }
  // 返回 Cholesky 解的梯度
  return std::tuple<Tensor, Tensor>{grad_self, grad_input2};
}

Tensor cholesky_solve_jvp(
    const Tensor& X,
    const Tensor& U,
    const Tensor& dU,
    const Tensor& dB,
    const bool upper) {
  // 禁用 TF32，确保使用的是标准的计算路径
  at::NoTF32Guard disable_tf32;
  // 计算 dK
  auto dK = upper ? dU.mH().matmul(U) : dU.matmul(U.mH());
  // 计算 dA
  auto dA = dK + dK.mH();
  // 调用通用的 Jacobian-vector product 函数，并返回结果
  return generic_solve_jvp(
      [&](const Tensor& A, const Tensor& dB, const Tensor& dA_contrib) {
        return at::cholesky_solve(dB - dA_contrib, A, upper);
      },
      X,
      /*A=*/U,
      dA,
      dB);
}

Tensor fft_c2r_backward(
    const Tensor& grad,
    IntArrayRef dim,
    int64_t normalization) {
  // 正向操作为 C2R 的反向操作解释
  // 对于 onesided 的 C2R rfft 操作：
  //    1. 填充另一半以确保共轭对称性
  //    2. 进行逆 C2C ifft
  //    3. 丢弃复数维度
  // 反向操作：
  //    1. 进行 R2C 的 rfft（添加虚拟复数维度并进行离散傅里叶变换）
  //    2. 根据共轭对称性累积梯度，由于 rfft 结果遵循共轭对称性，仅需对某些条目加倍，
  //       即其反射索引也位于 onesided 范围之外的条目。详细考虑最后一个维度的索引：
  //          - 当 N 偶数时，需要加倍的索引为 1 到 N/2 - 1
  //          - 当 N 奇数时，需要加倍的索引为 1 到 (N-1)/2
  auto gI = at::_fft_r2c(grad, dim, normalization, /*onesided=*/true);

  // 计算需要加倍的长度
  auto double_length = grad.sym_size(dim.back()) - gI.sym_size(dim.back());
  if (double_length > 0) { // 同时也覆盖了信号大小为零的情况
    // 对需要加倍的部分进行乘以 2 的操作
    gI.narrow_symint(dim.back(), 1, double_length).mul_(2);
  }
  // 返回计算得到的梯度
  return gI;
}
// 反向传播的 FFT 实现，用于计算 R2C 变换的梯度
Tensor fft_r2c_backward(
    const Tensor& grad,                       // 输入的梯度张量
    at::IntArrayRef dim,                      // 变换维度
    int64_t normalization,                    // 归一化方式
    bool onesided,                            // 是否是单边频谱
    const c10::SymInt& last_dim_size) {       // 最后一个维度的大小

  // 如果不是单边频谱，则进行复数到复数的逆 FFT 变换
  if (!onesided) {
    return at::real(at::_fft_c2c(grad, dim, normalization, /*forward=*/false));
  }

  // 单边频谱的反向传播
  // 将单边 R2C 变换想象为：
  //   1. 将输入视为复数（用零填充复数维度）
  //   2. 复数到复数的 FFT
  //   3. 丢弃一半的结果
  // 因此反向传播包括：
  //   1. 用零填充另一半（使用下面的 `zero_grad_shape`）
  //      （因为 C2C ifft 只接受双边输入，所以这里需要填充）
  //   2. 反向复数到复数的逆 FFT
  //   3. 丢弃复数维度
  auto half_sizes = grad.sym_sizes();
  std::vector<c10::SymInt> new_grad_shape(half_sizes.begin(), half_sizes.end());
  const auto last_dim =
      at::maybe_wrap_dim(dim.back(), static_cast<int64_t>(half_sizes.size()));
  new_grad_shape[last_dim] = last_dim_size;

  const auto zero_length = last_dim_size - grad.sym_size(dim.back());
  auto complex_full_grad =
      zero_length > 0 ? grad.new_zeros_symint(new_grad_shape) : grad;
  if (zero_length > 0) {
    complex_full_grad.slice_symint(last_dim, 0, half_sizes[last_dim])
        .copy_(grad);
  }
  return at::real(
      at::_fft_c2c(complex_full_grad, dim, normalization, /*forward=*/false));
}

// batchnorm_double_backward 的辅助函数，计算指定维度上的和
static Tensor sum_exclude_dim1(const Tensor& to_sum, bool keepdim = true) {
  auto r = to_sum.sum(0, keepdim);  // 按第一个维度求和，保持维度不变
  int64_t start_point_exclusive = keepdim ? 1 : 0;
  for (int64_t dim = r.dim() - 1; dim > start_point_exclusive; dim--) {
    r = r.sum(dim, keepdim);  // 沿着指定维度继续求和，保持维度不变
  }
  return r;
}

// batchnorm_double_backward 的辅助函数，模拟 expand_as 操作，但不进行实际扩展，操作时保持维度为 True
static Tensor unsqueeze_dim1(const Tensor& src, const Tensor& target) {
  auto src_expanded = src;
  while (src_expanded.sizes().size() < target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(1);  // 在维度 1 处添加维度
  }
  if (src_expanded.sizes().size() == target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(0);  // 在维度 0 处添加维度
  }
  return src_expanded;
}

// batchnorm_double_backward 的辅助函数，扩展 src 张量以匹配 target 的维度，维持维度为 1 的情况
static Tensor expand_as_dim1(const Tensor& src, const Tensor& target) {
  auto src_expanded = src;
  while (src_expanded.sizes().size() < target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(1);  // 在维度 1 处添加维度
  }
  return src_expanded.expand_as(target);  // 根据 target 的形状扩展 src
}

std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    const Tensor& input,                           // 输入张量
    const std::optional<Tensor>& gamma,            // gamma 参数（可选）
    const Tensor& ggI,                             // ggI 参数
    const Tensor& ggG,                             // ggG 参数
    const Tensor& ggB,                             // ggB 参数
    const Tensor& gO,                              // gO 参数
    const std::optional<Tensor>& running_mean,     // 运行时均值（可选）
    const std::optional<Tensor>& running_var,      // 运行时方差（可选）
    bool training,                                 // 是否为训练模式
    double eps,                                    // 用于稳定性的小值
  const std::optional<Tensor>& save_mean,
  const std::optional<Tensor>& save_invstd,
  std::array<bool, 3> output_mask) {
bool affine = isDefined(gamma);
// 检查是否定义了 gamma，以确定是否需要进行仿射变换
Tensor gamma_expanded;
Tensor ggG_expanded, ggB_expanded;
if (affine) {
  // 如果需要仿射变换，根据 input 的维度扩展 gamma
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  gamma_expanded = expand_as_dim1(*gamma, input);
  if (ggG.defined()) {
    // 如果定义了 ggG，则同样根据 input 的维度扩展 ggG
    ggG_expanded = expand_as_dim1(ggG, input);
  }
  if (ggB.defined()) {
    // 如果定义了 ggB，则同样根据 input 的维度扩展 ggB
    ggB_expanded = expand_as_dim1(ggB, input);
  }
} else {
  // 如果不需要仿射变换，则设定 gamma_expanded 为与 input 相同维度的全为 1 的张量
  gamma_expanded = at::ones({}, input.options());
}

// 定义一些将要重复使用的术语
auto M = input.size(0);
for (auto s : input.sizes().slice(2)) {
  M *= s;
}
// 对于半输入（half inputs），save_mean 和 save_invstd 是 float 类型（理想情况下，我们会转换其他所有内容，但现在不这样做）
auto mu = unsqueeze_dim1(
    training ? toNonOptTensor(save_mean).to(input.scalar_type())
             : toNonOptTensor(running_mean),
    input);
auto input_sub_mu = input - mu;
auto sigma2_eps_neg_1_2 = unsqueeze_dim1(
    training ? toNonOptTensor(save_invstd).to(input.scalar_type())
             : toNonOptTensor(running_var).add(Scalar(eps)).pow(-0.5),
    input);
auto sigma2_eps_neg_1 = sigma2_eps_neg_1_2.pow(2);
auto sigma2_eps_neg_3_2 = sigma2_eps_neg_1_2.pow(3);

// 计算 gI
auto input_mu_sigma2_neg_3_2 = input_sub_mu * sigma2_eps_neg_3_2;
auto gOinmu_sum = sum_exclude_dim1(gO * input_sub_mu);
auto gO_sum = sum_exclude_dim1(gO);

Tensor gI;
if (ggI.defined() && training) {
  // 如果定义了 ggI 并且处于训练模式
  auto ggI_sum = sum_exclude_dim1(ggI);
  auto ggIinmu_sum = sum_exclude_dim1(ggI * input_sub_mu);
  auto all_sub = ((ggI_sum * gO_sum).div_(M))
                     .sub_(sum_exclude_dim1(gO * ggI))
                     .add_((sigma2_eps_neg_1 * gOinmu_sum * ggIinmu_sum)
                               .mul_(3. / static_cast<double>(M)));
  auto gI_0t = (input_mu_sigma2_neg_3_2 * all_sub).div_(M);
  auto gI_1t =
      (ggIinmu_sum * sigma2_eps_neg_3_2).div_(M) * (gO_sum.div(M) - gO);
  auto gI_2t =
      (gOinmu_sum * sigma2_eps_neg_3_2).div_(M) * (ggI_sum.div(M) - ggI);
  gI = gamma_expanded * (gI_0t.add_(gI_1t).add_(gI_2t));
}

// 添加 gamma 项对 gI 的贡献
Tensor gI_G_term;
if (affine && ggG.defined()) {
  // 如果需要仿射变换并且定义了 ggG
  if (training) {
    auto t0 = gO * sigma2_eps_neg_1_2;
    auto t1 = (sigma2_eps_neg_1_2 * gO_sum).div_(-M);
    auto t2 = (input_mu_sigma2_neg_3_2 * sum_exclude_dim1(gO * input_sub_mu))
                  .div_(-M);
    gI_G_term = ggG_expanded * (t0.add_(t1).add_(t2));
    gI = gI.defined() ? gI.add_(gI_G_term) : gI_G_term;
  } else {
    gI_G_term = ggG_expanded * sigma2_eps_neg_1_2 * gO;
    gI = gI.defined() ? gI.add_(gI_G_term) : gI_G_term;
  }
}

// 定义一个 lambda 函数，计算第一个反向传播的梯度输入
auto first_back_grad_input = [&](const Tensor& gO,
                                 const Tensor& gamma) -> Tensor {
  // 计算 h0：gamma 乘以 sigma2_eps_neg_1_2，再除以 M
  auto h0 = (gamma * sigma2_eps_neg_1_2).div_(M);
  // 计算 h1：
  auto h1 = (M * gO)
                // 减去 gO 沿着第一个维度求和的结果
                .sub_(sum_exclude_dim1(gO))
                // 减去 input_sub_mu 乘以 sigma2_eps_neg_1 乘以 gO 沿着第一个维度求和的结果
                .sub_(
                    input_sub_mu.mul(sigma2_eps_neg_1) *
                    sum_exclude_dim1(gO * input_sub_mu));
  // 返回 h0 乘以 h1 的结果
  return h0 * h1;
};

// 计算 gG
Tensor gG;
if (affine && ggI.defined()) {
  if (training) {
    // 如果在训练阶段，gG 就是去掉 gamma 项的第一个反向传播的结果（然后进行形状调整）
    gG = ggI *
        first_back_grad_input(gO, at::ones({}, sigma2_eps_neg_1_2.options()));
    // 沿着第一个维度求和 gG
    gG = sum_exclude_dim1(gG, false);
  } else {
    // 否则，计算 ggI 乘以 gO 乘以 sigma2_eps_neg_1_2 沿着第一个维度求和的结果
    gG = sum_exclude_dim1(ggI * gO * sigma2_eps_neg_1_2, false);
  }
}

// 计算 ggO
Tensor ggO;
// 计算输入项的贡献
if (ggI.defined()) {
  if (training) {
    // 如果在训练阶段，ggO 就是第一个反向传播的结果乘以 gamma_expanded
    ggO = first_back_grad_input(ggI, gamma_expanded);
  } else {
    // 否则，计算 ggI 乘以 sigma2_eps_neg_1_2 乘以 gamma_expanded 的结果
    ggO = ggI * sigma2_eps_neg_1_2 * gamma_expanded;
  }
}
// 如果 ggG 被定义了
if (ggG.defined()) {
  // 计算 ggG_expanded 乘以 input_sub_mu 乘以 sigma2_eps_neg_1_2 的结果
  auto ggO_G_term = ggG_expanded * input_sub_mu * sigma2_eps_neg_1_2;
  // 如果 ggO 已经定义，则将 ggO_G_term 加到 ggO 上，否则直接赋值给 ggO
  ggO = ggO.defined() ? ggO.add_(ggO_G_term) : ggO_G_term;
}
// 如果 ggB 被定义了
if (ggB.defined()) {
  // 将 ggB_expanded 移动到 ggO 上，如果 ggO 已经定义，则将 ggO_B_term 加到 ggO 上，否则直接赋值给 ggO
  auto ggO_B_term = std::move(ggB_expanded);
  ggO = ggO.defined() ? ggO.add_(ggO_B_term) : ggO_B_term;
}

// 如果 output_mask 的第二个元素为真，并且 gG 没有被定义
if (output_mask[1] && !gG.defined()) {
  // 断言，当需要梯度时，gamma 应该总是被定义的
  AT_ASSERTM(affine, "gamma should always be defined when it requires grad");
}

// 返回三个 Tensor 的元组：gI, gG, ggO
return std::tuple<Tensor, Tensor, Tensor>{gI, gG, ggO};
  // 定义函数 layer_norm_double_backward，接受多个参数，并返回三个张量的元组
  const auto normalized_ndim = normalized_shape.size();
  // 计算输入张量的规范化维度数量
  const auto input_shape = input_t.sizes();
  // 获取输入张量的形状
  const auto input_ndim = input_t.dim();
  // 获取输入张量的维度数
  const auto axis = input_ndim - normalized_ndim;
  // 计算规范化的起始轴
  const int64_t M =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  // 计算 M，即规范化之前维度的乘积
  const int64_t N =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());
  // 计算 N，即规范化后维度的乘积

  // 将输入张量和输出梯度张量重塑为二维张量
  auto input = input_t.reshape({M, N});
  auto gO = gO_t.reshape({M, N});
  auto save_mean = save_mean_t.reshape({M, 1});
  auto save_invstd = save_invstd_t.reshape({M, 1});

  // 检查是否定义了 gamma 张量
  bool affine = isDefined(gamma);
  Tensor gamma_expanded;
  Tensor ggG_expanded, ggB_expanded;
  if (affine) {
    // 如果定义了 gamma 张量，则将其重塑为二维张量
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    gamma_expanded = gamma->reshape({1, N});
    // 如果定义了 ggG 张量，则将其重塑为二维张量
    if (ggG.defined()) {
      ggG_expanded = ggG.reshape({1, N});
    }
    // 如果定义了 ggB 张量，则将其重塑为二维张量
    if (ggB.defined()) {
      ggB_expanded = ggB.reshape({1, N});
    }
  } else {
    // 如果未定义 gamma 张量，则创建一个全为 1 的二维张量
    gamma_expanded = at::ones({1}, input.options());
  }

  Tensor ggI_expanded;
  if (ggI.defined()) {
    // 如果定义了 ggI 张量，则将其重塑为二维张量
    ggI_expanded = ggI.reshape({M, N});
  }

  // 将 save_mean 和 save_invstd 张量转换为与输入张量相同的数据类型
  // 这一步是为了处理半精度输入的情况，确保数据类型一致性
  auto mu = save_mean.to(input.scalar_type());
  auto input_sub_mu = input - mu;
  auto sigma2_eps_neg_1_2 = save_invstd.to(input.scalar_type());
  auto sigma2_eps_neg_1 = sigma2_eps_neg_1_2.pow(2);
  auto sigma2_eps_neg_3_2 = sigma2_eps_neg_1_2.pow(3);

  Tensor gI;
  // 计算 gI 张量
  auto input_mu_sigma2_neg_3_2 = input_sub_mu * sigma2_eps_neg_3_2;

  if (ggI.defined()) {
    // 如果定义了 ggI 张量，则执行以下计算
    auto gxhat = gO * gamma_expanded;
    auto gxhat_mu_sum = (gxhat * input_sub_mu).sum(1, true);
    auto gxhat_sum = gxhat.sum(1, true);

    auto ggI_sum = ggI_expanded.sum(1, true);
    auto ggI_mu_sum = (ggI_expanded * input_sub_mu).sum(1, true);

    auto all_sub = ((ggI_sum * gxhat_sum).div_(N))
                       .sub_((ggI_expanded * gxhat).sum(1, true))
                       .add_((sigma2_eps_neg_1 * gxhat_mu_sum * ggI_mu_sum)
                                 .mul_(3. / static_cast<double>(N)));
    auto gI_0t = (input_mu_sigma2_neg_3_2 * all_sub).div_(N);
    auto gI_1t =
        (ggI_mu_sum * sigma2_eps_neg_3_2).div_(N) * (gxhat_sum.div(N) - gxhat);
    auto gI_2t = (gxhat_mu_sum * sigma2_eps_neg_3_2).div_(N) *
        (ggI_sum.div(N) - ggI_expanded);

    gI = (gI_0t.add_(gI_1t).add_(gI_2t));
  }

  // 添加 gamma 项对 gI 的贡献
  if (affine && ggG.defined()) {
    // 如果定义了 gamma 张量并且定义了 ggG 张量，则执行以下计算
    auto t0 = gO * ggG_expanded * sigma2_eps_neg_1_2;
  // 计算 t1，对应于公式中的第一项
  auto t1 = (sigma2_eps_neg_1_2 * (gO * ggG_expanded).sum(1, true)).div_(-N);

  // 计算 t2，对应于公式中的第二项
  auto t2 = (input_mu_sigma2_neg_3_2 *
             (gO * ggG_expanded * input_sub_mu).sum(1, true))
                .div_(-N);

  // 计算 gI_G_term，对应于将 t0、t1、t2 汇总得到的项
  auto gI_G_term = t0.add_(t1).add_(t2);

  // 更新 gI，如果之前已经定义过 gI，则累加 gI_G_term，否则直接赋值为 gI_G_term
  gI = gI.defined() ? gI.add_(gI_G_term) : gI_G_term;
}

// 如果 gI 已经定义，则将其重塑成与 input_t 相同的形状
if (gI.defined()) {
  gI = gI.reshape_as(input_t);
}

// 定义第一个反向传播函数的梯度输入 first_bwd_fn_grad_input
auto first_bwd_fn_grad_input = [&](const Tensor& gO_local,
                                   const Tensor& gamma_local) -> Tensor {
  // 计算 h0，对应于公式中的第一部分
  auto h0 = (gamma_local * sigma2_eps_neg_1_2).div_(N);

  // 计算 h1，对应于公式中的第二部分
  auto h1 = (N * gO_local)
                .sub_(gO_local.sum(1, true))
                .sub_(
                    input_sub_mu.mul(sigma2_eps_neg_1) *
                    (gO_local * input_sub_mu).sum(1, true));

  // 返回 h0 和 h1 的乘积作为结果
  return h0 * h1;
};

// 计算 gG
Tensor gG;
if (affine && ggI.defined()) {
  // 使用 first_bwd_fn_grad_input 计算 gG
  gG = first_bwd_fn_grad_input(
      ggI_expanded, at::ones({}, sigma2_eps_neg_1_2.options()));
  // 对 gO 和 gG 的乘积进行求和，得到 gG
  gG = (gO * gG).sum(0);
  // 将 gG 重塑成与 gamma 相同的形状
  gG = gG.reshape_as(*gamma);
}

// 计算 ggO
Tensor ggO;

// 贡献输入项的部分
if (ggI.defined()) {
  ggO = first_bwd_fn_grad_input(ggI_expanded, gamma_expanded);
}

// 如果定义了 ggG，则添加 ggO 的贡献
if (ggG.defined()) {
  auto ggO_G_term = ggG_expanded * input_sub_mu * sigma2_eps_neg_1_2;
  ggO = ggO.defined() ? ggO.add_(ggO_G_term) : ggO_G_term;
}

// 如果定义了 ggB，则添加 ggO 的贡献
if (ggB.defined()) {
  auto ggO_B_term = std::move(ggB_expanded);
  ggO = ggO.defined() ? ggO.add_(ggO_B_term) : ggO_B_term;
}

// 如果 ggO 已经定义，则将其扩展为形状为 {M, N}，然后重塑成与 input_t 相同的形状
if (ggO.defined()) {
  ggO = ggO.expand({M, N}).reshape_as(input_t);
}

// 检查是否需要计算 gG，并确保 affine 为真时 gamma 必须定义
if (output_mask[1] && !gG.defined()) {
  AT_ASSERTM(affine, "gamma should always be defined when it requires grad");
}

// 返回 gI、gG、ggO 的元组作为结果
return std::tuple<Tensor, Tensor, Tensor>{gI, gG, ggO};
}

std::tuple<Tensor, Tensor, Tensor>
infinitely_differentiable_native_group_norm_backward(
    const Tensor& dY,                          // 输入：梯度 dY
    const Tensor& dmean,                       // 输入：均值的梯度 dmean
    const Tensor& drstd,                       // 输入：标准差的逆的梯度 drstd
    const Tensor& X,                           // 输入：输入张量 X
    const Tensor& mean,                        // 输入：均值张量 mean
    const Tensor& rstd,                        // 输入：标准差的逆 rstd
    const std::optional<Tensor>& gamma,        // 输入：可选的缩放因子 gamma
    c10::SymInt N,                             // 输入：批量大小 N
    const c10::SymInt& C,                      // 输入：通道数 C
    c10::SymInt HxW,                           // 输入：高度乘宽度 HxW
    int64_t group,                             // 输入：分组数 group
    double eps,                                // 输入：epsilon 值 eps
    std::array<bool, 3> grad_input_mask) {     // 输入：梯度输入掩码 grad_input_mask
  const int64_t G = group;                     // 计算：将 group 赋值给 G
  const auto D = C / G;                        // 计算：计算分组大小 D
  c10::SymFloat s = c10::SymFloat(1.0) / c10::SymFloat(D * HxW);  // 计算：s 的值
  Tensor dX;                                   // 输出：dX 梯度张量
  Tensor dgamma;                               // 输出：dgamma 梯度张量
  Tensor dbeta;                                // 输出：dbeta 梯度张量
  const Tensor X_tensor = X.reshape_symint({N, G, D, HxW});         // 计算：将 X 重塑为张量 X_tensor
  const Tensor mean_tensor = mean.reshape_symint({N, G, 1, 1});     // 计算：将 mean 重塑为张量 mean_tensor
  const Tensor rstd_tensor = rstd.reshape_symint({N, G, 1, 1});     // 计算：将 rstd 重塑为张量 rstd_tensor
  Tensor dY_tensor;                             // 临时变量：dY 的张量形状
  Tensor ds;                                    // 临时变量：ds 张量
  Tensor db;                                    // 临时变量：db 张量
  if (dY.defined()) {                          // 条件：如果 dY 已定义
    dY_tensor = dY.reshape_symint({N, G, D, std::move(HxW)});  // 计算：将 dY 重塑为张量 dY_tensor
    ds = (dY_tensor * X_tensor).sum(3).unsqueeze_(-1);  // 计算：计算 ds 张量
    db = dY_tensor.sum(3).unsqueeze_(-1);       // 计算：计算 db 张量
  }
  if (grad_input_mask[0]) {                    // 条件：如果 grad_input_mask 的第一个元素为真
    Tensor gamma_tensor;                       // 临时变量：gamma 的张量形状
    if (isDefined(gamma)) {                    // 条件：如果 gamma 已定义
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      gamma_tensor = gamma->reshape_symint({1, G, D, 1});  // 计算：将 gamma 重塑为张量 gamma_tensor
    }
    const Tensor var =                         // 计算：计算 var 张量
        ((rstd_tensor * rstd_tensor).reciprocal_() - eps).clamp_min(0);
    const Tensor rstd_cube = rstd_tensor * rstd_tensor * rstd_tensor;  // 计算：计算 rstd_cube 张量
    Tensor dvar;                               // 临时变量：dvar 张量
    if (drstd.defined()) {                     // 条件：如果 drstd 已定义
      dvar = -0.5 * rstd_cube * drstd.view_symint({N, G, 1, 1});  // 计算：计算 dvar 张量
    }
    if (dY.defined()) {                        // 条件：如果 dY 已定义
      const Tensor a =                         // 计算：计算 a 张量
          isDefined(gamma) ? rstd_tensor * gamma_tensor : rstd_tensor;
      Tensor b = (isDefined(gamma) ? (ds * gamma_tensor).sum(2) : ds.sum(2))  // 计算：计算 b 张量
                     .unsqueeze_(-2);
      Tensor c = (isDefined(gamma) ? (db * gamma_tensor).sum(2) : db.sum(2))  // 计算：计算 c 张量
                     .unsqueeze_(-2);
      b = (c * mean_tensor - b) * rstd_cube * s;  // 计算：计算 b 张量
      c = -b * mean_tensor - c * rstd_tensor * std::move(s);  // 计算：计算 c 张量
      dX = a * dY_tensor + b * X_tensor + c;   // 计算：计算 dX 张量
      if (dmean.defined() && drstd.defined()) {
        dX += var_mean_backward(
            dvar,
            dmean.view_symint({std::move(N), G, 1, 1}),
            X_tensor,
            IntArrayRef{2, 3},
            0,
            true);                            // 计算：调用 var_mean_backward 函数
      }
      dX = dX.reshape_as(X);                   // 计算：将 dX 重塑为 X 的形状
    } else if (dmean.defined() && drstd.defined()) {
      dX = var_mean_backward(
               dvar,
               dmean.view_symint({std::move(N), G, 1, 1}),
               X_tensor,
               IntArrayRef{2, 3},
               0,
               true)
               .reshape_as(X);                // 计算：调用 var_mean_backward 函数
    }
  }
  if (grad_input_mask[1] && dY.defined()) {     // 条件：如果 grad_input_mask 的第二个元素为真，并且 dY 已定义
    dgamma = ((ds - db * mean_tensor) * rstd_tensor)  // 计算：计算 dgamma 张量
                 .sum(0)
                 .reshape_as(toNonOptTensor(gamma));
  }
  if (grad_input_mask[2] && dY.defined()) {     // 条件：如果 grad_input_mask 的第三个元素为真，并且 dY 已定义
    dbeta = db.sum(0).reshape_as(toNonOptTensor(gamma));  // 计算：计算 dbeta 张量
  }

  return std::make_tuple(dX, dgamma, dbeta);    // 返回：dX, dgamma, dbeta 的元组
}
    const std::optional<Tensor>& i1,   // 第一个输入张量的可选引用
    const std::optional<Tensor>& i2,   // 第二个输入张量的可选引用
    const std::optional<Tensor>& i3,   // 第三个输入张量的可选引用
    IntArrayRef expand1,                // 扩展维度数组1
    IntArrayRef expand2,                // 扩展维度数组2
    IntArrayRef expand3,                // 扩展维度数组3
    IntArrayRef sumdim,                 // 求和维度数组
    std::array<bool, 3> grad_mask) {    // 梯度掩码数组，指示哪些梯度需要计算

  Tensor grad_i1, grad_i2, grad_i3;     // 定义梯度张量变量

  // 如果输出梯度已定义
  if (grad_out.defined()) {
    // 如果需要计算第一个输入张量的梯度
    if (grad_mask[0])
      grad_i1 =
          // 调用 _trilinear 函数计算第一个输入张量的梯度
          at::_trilinear(grad_out, *i2, *i3, sumdim, expand2, expand3, expand1);

    // 如果需要计算第二个输入张量的梯度
    if (grad_mask[1])
      grad_i2 =
          // 调用 _trilinear 函数计算第二个输入张量的梯度
          at::_trilinear(*i1, grad_out, *i3, expand1, sumdim, expand3, expand2);

    // 如果需要计算第三个输入张量的梯度
    if (grad_mask[2])
      grad_i3 =
          // 调用 _trilinear 函数计算第三个输入张量的梯度
          at::_trilinear(*i1, *i2, grad_out, expand1, expand2, sumdim, expand3);
  }

  // 返回计算得到的三个梯度张量
  return std::tuple<Tensor, Tensor, Tensor>(grad_i1, grad_i2, grad_i3);
// 返回值是一个张量，表示 log1p 操作的反向传播结果，接受两个张量作为输入参数：grad 和 self
Tensor log1p_backward(const Tensor& grad, const Tensor& self) {
  // 如果 self 的布局是稀疏的，需要通过 to_dense 方法进行条件初始化
  Tensor self_p1_conj;
  if (self.layout() == c10::kSparse || self.layout() == c10::kSparseCsr ||
      self.layout() == c10::kSparseCsc || self.layout() == c10::kSparseBsr ||
      self.layout() == c10::kSparseBsc) {
    // 如果 self 是稀疏布局，发出警告，并将其转换为稠密张量后执行 log1p 操作
    TORCH_WARN(
        "log1p_backward: received self with sparse layout, but backward requires materialization of a dense tensor with this shape");
    self_p1_conj = (self.to_dense() + 1).conj();
  } else {
    // 否则，直接在 self 上执行 log1p 操作
    self_p1_conj = (self + 1).conj();
  }
  // 如果 grad 的布局也是稀疏的，则需要乘以 self_p1_conj 的倒数；否则直接除以 self_p1_conj
  if (grad.layout() == c10::kSparse || grad.layout() == c10::kSparseCsr ||
      grad.layout() == c10::kSparseCsc || grad.layout() == c10::kSparseBsr ||
      grad.layout() == c10::kSparseBsc) {
    // 保持 grad 的布局不变，乘以 self_p1_conj 的倒数
    return grad * self_p1_conj.reciprocal_();
  }
  // 返回 grad 除以 self_p1_conj 的结果
  return grad / self_p1_conj;
}

// 返回值是一个张量，表示 sinc 函数的反向传播结果，接受两个张量作为输入参数：grad 和 self
Tensor sinc_backward(const Tensor& grad, const Tensor& self) {
  // 计算 self 乘以 π
  auto self_pi = self * M_PI;
  // 计算 self 平方乘以 π
  auto self_squared_pi = self * self * M_PI;
  // 计算 sinc 函数的反向传播结果
  auto out = grad *
      ((self_pi * self_pi.cos() - self_pi.sin()) / self_squared_pi).conj();
  // 返回根据 self_squared_pi 是否为零的条件，选择不同的输出
  return at::where(self_squared_pi == 0.0, at::zeros({}, grad.options()), out);
}

// 返回值是一个张量，表示常数填充的多维张量反向传播结果，接受 grad 和 pad 作为输入参数
Tensor constant_pad_nd_backward(const Tensor& grad, c10::SymIntArrayRef pad) {
  // 将 pad 转换为负数形式
  auto negated_pad = pad.vec();
  std::transform(
      negated_pad.cbegin(),
      negated_pad.cend(),
      negated_pad.begin(),
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::negate<c10::SymInt>());
  // 调用 constant_pad_nd_symint 函数进行常数填充的多维张量的反向传播
  return at::constant_pad_nd_symint(grad, negated_pad, 0);
}

// 返回值是一个张量，表示嵌入层的双向反向传播结果，接受 grad、indices 和 padding_idx 作为输入参数
Tensor embedding_dense_double_backward_symint(
    const Tensor& grad,
    const Tensor& indices,
    const c10::SymInt& padding_idx) {
  // 根据 indices 选择对应的梯度
  auto gg_weight = grad.index_select(0, indices.reshape(-1));

  // 根据 indices 的形状重塑梯度
  auto size = indices.sizes().vec();
  size.push_back(-1);

  // 如果 padding_idx 大于等于 0，则用 0 替换 gg_weight 中与 padding_idx 相等的元素
  if (padding_idx >= 0) {
    gg_weight.masked_fill_((indices == padding_idx).reshape({-1, 1}), 0);
  }
  // 将 gg_weight 视图重塑为 size 的形状并返回
  return gg_weight.view(size);
}

// 返回值是一个张量，表示索引操作的反向传播结果，接受 zeros_like_self 和 indices 作为输入参数
Tensor index_backward(
    Tensor zeros_like_self,
    const torch::List<std::optional<Tensor>>& indices,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    const c10::SymInt& size) {
    // 定义一个函数，接受三个参数：zeros_like_self、indices、grad，并返回一个张量
    const Tensor& grad) {
      // 检查 zeros_like_self 和 grad 是否有任何子类与张量相似，或者检查 indices 是否有任何可选的子类与张量相似
      return (areAnyTensorSubclassLike({zeros_like_self, grad}) ||
              areAnyOptionalTensorSubclassLike(indices))
          // 如果上述条件成立，调用 zeros_like_self 的 index_put 方法，对 indices 所指定位置进行赋值操作，保留历史版本
          ? zeros_like_self.index_put(indices, grad, true)
          // 如果上述条件不成立，则调用底层的 _index_put_impl_ 函数处理 zeros_like_self、indices 和 grad 的索引赋值操作，同时保留历史版本和深度复制
          : at::_index_put_impl_(zeros_like_self, indices, grad, true, true);
    }
// 返回在给定梯度、损失、原始梯度和零无穷标志的情况下的 CTC 损失的反向传播
Tensor _cudnn_ctc_loss_backward(
    const Tensor& grad_out,                    // CTC 损失的梯度
    const Tensor& loss,                        // CTC 损失
    const Tensor& raw_grad,                    // 原始梯度
    bool zero_infinity) {                      // 零无穷标志

  if (zero_infinity) {
    // 如果 zero_infinity 为真，则根据损失是否为零来选择性地设置梯度为零
    return at::where(
        loss.unsqueeze(0).unsqueeze(2) == 0,   // 检查损失是否为零的条件
        at::zeros({}, raw_grad.options()),     // 如果损失为零则返回零梯度
        raw_grad * grad_out.unsqueeze(0).unsqueeze(2));  // 否则返回计算的梯度
  } else {
    // 如果 zero_infinity 为假，则直接返回计算的梯度
    return raw_grad * grad_out.unsqueeze(0).unsqueeze(2);
  }
}

// 检查给定变量列表中是否有任何已定义的变量
bool any_variable_defined(const variable_list& variables) {
  for (const auto& variable : variables) {
    if (variable.defined()) {
      return true;  // 如果找到已定义的变量，则返回真
    }
  }
  return false;       // 如果所有变量都未定义，则返回假
}

// householder_product.backward 方法的导数推导
//
// 给定向量序列 v_1, ..., v_n 和标量序列 tau_1, ..., tau_k，torch.linalg.householder_product
// 计算以下乘积的前 n 列：Q = (I - tau_1 v_1 v_1^H) ... (I - tau_k v_k v_k^H)。定义
//     H_i(sigma) := I - sigma v_i v_i^H，因此 Q = (H_1(sigma_1) ... H_k(sigma_k))[:, :k]；
//     H_i_minus = H_1(tau_1) ... H_{i - 1}(tau_{i - 1})，其中 H_1_minus := I；
//     H_i_plus = H_{i + 1}(tau_{i + 1}) ... H_k(tau_k)，其中 H_k_plus := I。
//
// Forward AD:
// dQ = sum_{i = 1}^k H_i_minus (-dtau_i v_i v_i^H - tau_i dv_i v_i^H - tau_i v_i dv_i^H) H_i_plus。
//
// Backward AD:
// Tr(Q_grad^H dQ) = sum_{i = 1}^k Tr(H_i_plus Q_grad^H H_i_minus (-dtau_i v_i v_i^H - tau_i dv_i v_i^H - tau_i v_i dv_i^H))。
// 定义 K_i := H_i_plus Q_grad^H H_i_minus，因此梯度为 v_i_grad = (-tau_i v_i^H K_i)^H - tau_i K_i v_i，
// tau_i_grad = Tr(-v_i^H K_i v_i).conj()。注意：算法忽略只观察 Q 的前 n 列的事实，因此不需要完全重建 Q。
//
// 注意 K_{i + 1} = H_{i + 1}^{-1} K_i H_i，因此可以通过有效更新 K_i 逐个计算 v_i_grad 和 tau_i_grad。
// 从右侧使用矩阵向量乘积进行 H_i 的乘法更新，但逆 H_{i + 1}^{-1} 的问题如何解决？
// 幸运的是，在某些假设下，H_{i + 1}^{-1} 存在并且可以表示为 H_i(sigma_i)，因此左侧更新也可以通过矩阵向量而不是矩阵-矩阵乘法来完成。
//
// 设 H(tau) := I - tau v v^H。
// H(tau) 的特征值是 1，其重数为 (m - 1)，对应于垂直于 v 的特征向量，并且有一个特征值 (1 - tau ||v||^2)，其特征向量为 v / ||v||。
// 如果 (1 - tau ||v||^2) != 0，则 H(tau) 是可逆的。如果 (1 - tau ||v||^2) != 0，则定义 sigma := tau / (||v||^2 tau - 1)，
// 可以得到 H(tau) H(sigma) = H(sigma) H(tau) = I，因此 H(sigma) 是 H(tau) 的逆。
//
// 警告：下面的算法假设所有的 H_i(tau_i) 都是可逆的，因此它期望对所有 i 都有 (1 - tau_i ||v_i||^2) != 0。
// 我们要指出，即使有 H_i(tau_i) 不可逆的情况，householder_product 仍然是可微分的！我们不会
//
// 计算简单变换后的张量，支持在原地或者非原地修改
static Tensor apply_simple_transformation(
    const c10::SymInt& m,  // 维度参数 m
    const c10::SymInt& k,  // 维度参数 k
    const Tensor& u_full,  // 输入向量 u 的完整张量
    const Tensor& v_full,  // 输入向量 v 的完整张量
    const Tensor& t,       // 标量参数 t
    Tensor& K,             // 待变换的张量 K
    bool modify_K_in_place = true,  // 是否原地修改 K，默认为 true
    bool condition_with_I = true,   // 是否与单位矩阵 I 结合，默认为 true
    bool left = true) {             // 是否从左边进行变换，默认为 true
  // 假设 u_full 是维度 (..., m, 1) 的向量，t 是维度 (..., 1) 的标量

  // TODO: 下面的矩阵向量乘积在代码中被调度到矩阵矩阵乘积。我们需要扩展 matmul 支持批处理的矩阵向量乘积，
  // 或者实现一个批处理的 mv 变体。我们可以为未批处理的输入启用 mv，但是目前没有这样做以消除代码重复。

  // 如果 left 为 true，返回 (I - t u v^H) K 或者 -t u v^H K
  if (left) {
    if (modify_K_in_place) {
      auto v = u_full.narrow_symint(-2, k, m - k);  // 从 u_full 中选择第 k 至 m-k 维的子张量 v
      auto u = v_full.narrow_symint(-2, k, m - k)   // 从 v_full 中选择第 k 至 m-k 维的子张量 u，
                   .mH()                              // 求其共轭转置
                   .matmul(K.narrow_symint(-2, k, m - k));  // 计算矩阵乘积
      K.narrow_symint(-2, k, m - k).sub_((t.unsqueeze(-1) * v) * u);  // 原地修改 K 的部分张量
      return K;  // 返回修改后的 K
    } else {
      auto transformation = (t.unsqueeze(-1) * u_full) * v_full.mH().matmul(K);  // 计算变换矩阵
      return condition_with_I ? K - transformation : -transformation;  // 根据 condition_with_I 返回 K 或者 -transformation
    }
  }
  // 如果 left 为 false，返回 K (I - t u v^H) 或者 -K t u v^H
  else {
    if (modify_K_in_place) {
      auto v = u_full.narrow_symint(-2, k, m - k);  // 从 u_full 中选择第 k 至 m-k 维的子张量 v
      auto u =
          K.narrow_symint(-1, k, m - k)  // 从 K 中选择第 k 至 m-k 维的子张量 u
              .matmul(t.unsqueeze(-1) * v_full.narrow_symint(-2, k, m - k));  // 计算矩阵乘积
      K.narrow_symint(-1, k, m - k).sub_(u * v.mH());  // 原地修改 K 的部分张量
      return K;  // 返回修改后的 K
    } else {
      auto transformation = K.matmul(t.unsqueeze(-1) * u_full) * v_full.mH();  // 计算变换矩阵
      return condition_with_I ? K - transformation : -transformation;  // 根据 condition_with_I 返回 K 或者 -transformation
    }
  }
};

// 对 Householder 变换的反向传播，返回梯度 grad 和结果 result 的元组
std::tuple<Tensor, Tensor> householder_product_backward(
    const Tensor& grad,        // 梯度张量
    const Tensor& result,      // 结果张量
    const Tensor& input_,      // 输入张量
    const Tensor& tau,         // 变换参数 tau
    const bool flip_order) {   // 是否翻转顺序的标志
  // 注意：当 flip_order 为 true 时，算法会反转主循环中的处理方向，并翻转左右 Householder 投影的应用。
  // 下面关于算法细节的注释假设 flip_order = false。
  if (!grad.defined() || input_.sym_numel() == 0 || tau.sym_numel() == 0) {
    // 返回一个空的 Tensor 元组
    return std::tuple<Tensor, Tensor>(Tensor(), Tensor());
  }
  // 获取 input_ 张量的倒数第二个维度的符号大小
  auto m = input_.sym_size(-2);
  // 由于下面的 irange 调用，guard_int 是用于保护 tau 的整数性
  auto k = tau.sym_size(-1).guard_int(__FILE__, __LINE__);

  // forward 只在假设 input 的对角线为 1 的情况下操作其下三角部分
  auto input = input_.tril(-1);
  input.diagonal(0, -2, -1).fill_(1.0);

  // 计算 sigma，满足 H(sigma_i) == H(tau_i)^{-1}。
  // 如果 householder_product 的输入来自 GEQRF，我们永远不会遇到 ||v_i||^2 tau_i == 1 的情况，
  // 因此 H(tau_i) 总是可逆的。这来自于文档 https://www.netlib.org/lapack/lug/node128.html，
  // 并且 tau 始终满足条件 |tau|^2 ||v||^2 == 2 * Re(tau)。
  auto input_first_k_cols = input.narrow(-1, 0, k);
  auto input_first_k_cols_norm_squared =
      (input_first_k_cols * input_first_k_cols.conj()).sum(-2);
  auto sigma = tau / (tau * input_first_k_cols_norm_squared - 1.0);

  auto K = result.matmul(grad.mH());

  // 算法通过左/右乘以 Householder 反射器来更新 K。
  // 如果只运行单个反向传播，我们就可以就地修改 K 并利用输入的三角性。
  // 对于高阶导数，我们无法重写 K 的存储方式，因此使用了效率较低的非就地方法。
  //
  // 如果只期望一阶导数，我们可以就地修改 K 以获得更好的性能
  bool modify_K_in_place = !at::GradMode::is_enabled();

  // 此方法利用了在第 k 次迭代时向量 v_k 只有非零元素 v_k[k:] 的事实。
  auto update_grad = [&m](
                         int64_t k,
                         const Tensor& v_full,
                         const Tensor& t,
                         const Tensor& K) -> std::tuple<Tensor, Tensor> {
    // v_full 是维度为 (..., m, 1) 的向量，t 是维度为 (..., 1) 的标量
    auto v = v_full.narrow_symint(-2, k, m - k);
    auto vHK = v.mH().matmul(K.narrow_symint(-2, k, m - k));
    auto Kv = K.narrow_symint(-1, k, m - k).matmul(v);
    auto t_unsqueezed = t.unsqueeze(-1);
    auto v_grad = (-t_unsqueezed * vHK).conj().squeeze(-2) -
        (t_unsqueezed * Kv).squeeze(-1);
    auto tau_grad = -(vHK.narrow_symint(-1, k, m - k).matmul(v)).conj();
    return std::make_tuple(v_grad.unsqueeze(-1), tau_grad.squeeze(-1));
  };

  // 应用 Householder 反射器
  auto apply_householder_reflector = [m, modify_K_in_place](
                                         int64_t k,
                                         const Tensor& v_full,
                                         const Tensor& t,
                                         Tensor& K,
                                         bool left = true) -> Tensor {
  return apply_simple_transformation(
      m,
      k,
      v_full,
      v_full,
      t,
      K,
      modify_K_in_place,
      /*condition_with_I=*/true,
      left);
};

const auto flip_i = [flip_order, k](int64_t i) -> int64_t {
  // 根据 flip_order 决定是否翻转索引 i 的顺序
  return !flip_order ? i : k - i - 1;
};
const auto next_i = [flip_order](int64_t i) -> int64_t {
  // 根据 flip_order 决定递增或递减索引 i
  return !flip_order ? ++i : --i;
};
const auto apply_left = !flip_order;

// K <- H_0^{-1} @ K
const auto zero_idx = flip_i(0);
K = apply_householder_reflector(
    zero_idx,
    input.narrow(-1, zero_idx, 1),
    sigma.narrow(-1, zero_idx, 1),
    K,
    /*left=*/apply_left);

Tensor input_grad, tau_grad;
// For Composite Compliance, we can't copy a Subclass into a Regular Tensor,
// so we use out-of-place ops with equivalent output.
// NOTE: We can't use `new_zeros` directly as `input`, 'tau' or `grad` can
// be Tensor Subclass and we don't want to make assumption about which
// one to choose for creating output buffer.
// eg. if both are BatchedTensor at different level.
if (areAnyTensorSubclassLike({input, tau, K})) {
  // k + 1 if input_grads hold a matrix of zeros for inactive parts of input.
  auto input_grads = std::vector<Tensor>(k < input.sym_size(-1) ? k + 1 : k);
  auto tau_grads = std::vector<Tensor>(k);

  for (const auto i_idx : c10::irange(k)) {
    auto i = flip_i(i_idx);
    // NOTE: narrow will unsqueeze(-1)
    auto v_i = input.narrow(-1, i, 1);
    auto t_i = tau.narrow(-1, i, 1);

    // 更新梯度，获取输入梯度和 tau 梯度
    std::tie(input_grads[i], tau_grads[i]) = update_grad(i, v_i, t_i, K);

    // K <- H_{i + 1}^{-1} @ K @ H_i
    if (i != flip_i(k - 1)) {
      auto i_next = next_i(i);
      auto v_i_next = input.narrow(-1, i_next, 1);
      auto s_i_next = sigma.narrow(-1, i_next, 1);
      // 应用 Householder 反射变换更新 K
      K = apply_householder_reflector(
          i_next, v_i_next, s_i_next, K, /*left=*/apply_left);
      K = apply_householder_reflector(i, v_i, t_i, K, /*left=*/!apply_left);
    }
  }

  // Only first k columns are active in forward.
  // zero gradients for the inactive input.
  if (k < input.sym_size(-1)) {
    auto zero_grad_shape =
        at::SymDimVector(input_.sym_sizes().slice(0, input_.dim() - 1));
    zero_grad_shape.push_back(input.sym_size(-1) - k);
    // 创建一个形状为 zero_grad_shape 的零梯度张量
    auto zero_grad = at::zeros_symint(zero_grad_shape, input_.options());
    input_grads[k] = zero_grad;
  }

  // 将所有输入梯度连接成一个张量
  input_grad = at::cat(input_grads, -1);
  // 将所有 tau 梯度连接成一个张量
  tau_grad = at::cat(tau_grads, -1);
} else {
  // 创建与 input_ 和 tau 相同形状的零张量
  input_grad = at::zeros_like(input_);
  tau_grad = at::zeros_like(tau);
    // 遍历从 0 到 k-1 的索引 i_idx
    for (const auto i_idx : c10::irange(k)) {
      // 计算翻转后的索引 i
      auto i = flip_i(i_idx);
      // 从输入张量 input 中按最后一个维度对第 i 列进行切片，获取 v_i
      auto v_i = input.narrow(-1, i, 1);
      // 从 tau 张量中按最后一个维度对第 i 列进行切片，获取 t_i
      auto t_i = tau.narrow(-1, i, 1);

      // 调用 update_grad 函数更新 v_i 和 t_i 的梯度，返回结果存储在 v_i_grad 和 tau_i_grad 中
      auto [v_i_grad, tau_i_grad] = update_grad(i, v_i, t_i, K);
      // 将 v_i_grad 压缩维度后复制到 input_grad 的第 i 列中
      input_grad.select(-1, i).copy_(v_i_grad.squeeze(-1));
      // 将 tau_i_grad 压缩维度后复制到 tau_grad 的第 i 列中
      tau_grad.select(-1, i).copy_(tau_i_grad.squeeze(-1));

      // 如果 i 不是 k-1 对应的翻转索引
      if (i != flip_i(k - 1)) {
        // 计算下一个索引 i_next
        auto i_next = next_i(i);
        // 从输入张量 input 中按最后一个维度对第 i_next 列进行切片，获取 v_i_next
        auto v_i_next = input.narrow(-1, i_next, 1);
        // 从 sigma 张量中按最后一个维度对第 i_next 列进行切片，获取 s_i_next
        auto s_i_next = sigma.narrow(-1, i_next, 1);
        
        // 在 K 上应用 Householder 反射变换，左乘还是右乘由 apply_left 决定
        K = apply_householder_reflector(
            i_next, v_i_next, s_i_next, K, /*left=*/apply_left);
        // 在 K 上应用 Householder 反射变换，左乘还是右乘由 apply_left 决定
        K = apply_householder_reflector(i, v_i, t_i, K, /*left=*/!apply_left);
      }
    }
  }

  // forward 函数仅在输入的下三角部分（不包括主对角线）进行操作，因此梯度也是下三角形式的。
  // 将 input_grad 转换为下三角形式，不包括主对角线以上的部分
  input_grad.tril_(-1);

  // 返回输入梯度 input_grad 和 tau 梯度 tau_grad 的 tuple
  return std::make_tuple(input_grad, tau_grad);
// `}` 表示前面的代码块结束

// `householder_product_jvp` 函数计算 Householder 乘积的 Jacobian-向量乘积
Tensor householder_product_jvp(
    const Tensor& dV_,  // 输入张量 dV_
    const Tensor& dtau, // 输入张量 dtau
    const Tensor& prod, // 输入张量 prod
    const Tensor& V_,   // 输入张量 V_
    const Tensor& tau)  // 输入张量 tau
{
  auto m = V_.sym_size(-2); // 提取 V_ 的倒数第二维度大小作为 m
  auto k = tau.size(-1);    // 提取 tau 的最后一个维度大小作为 k

  // forward 仅在假定输入对角线填充为 1 的情况下操作下三角部分
  auto V = V_.tril(-1);  // 提取 V_ 的下三角部分
  V.diagonal(0, -2, -1).fill_(1.0); // 将 V 的对角线填充为 1.0
  auto dV = dV_.tril(-1); // 提取 dV_ 的下三角部分

  // 计算 sigma，使得 H(sigma_i) == H(tau_i)^{-1}
  auto V_first_k_cols = V.narrow(-1, 0, k); // 提取 V 的最前面 k 列
  auto V_first_k_cols_norm_squared =
      (V_first_k_cols * V_first_k_cols.conj()).sum(-2); // 计算这些列的范数的平方和
  auto sigma = tau / (tau * V_first_k_cols_norm_squared - 1.0); // 计算 sigma

  // 定义应用 Householder 反射器的函数
  auto apply_householder_reflector = [m](const Tensor& v_full,
                                         const Tensor& t,
                                         Tensor& K,
                                         bool left = true) -> Tensor {
    return apply_simple_transformation(
        m,
        /*k=*/0,
        v_full,
        v_full,
        t,
        K,
        /*modify_K_in_place=*/false,
        /*condition_with_I=*/true,
        left);
  };

  // 定义计算简单乘积的函数
  auto apply_simple_product = [m](const Tensor& u_full,
                                  const Tensor& v_full,
                                  const Tensor& t,
                                  Tensor& K) -> Tensor {
    return apply_simple_transformation(
        m,
        /*k=*/0,
        u_full,
        v_full,
        t,
        K,
        /*modify_K_in_place=*/false,
        /*condition_with_I=*/false,
        /*left=*/true);
  };

  // 复制 prod 并分离它
  auto H_plus = prod.detach().clone();
  IntArrayRef batch_vector_shape(V.sizes().data(), V.dim() - 1); // 定义批次向量形状
  auto H_minus =
      at::diag_embed(at::ones({1}, V.options()).expand(batch_vector_shape)); // 创建对角矩阵 H_minus

  auto dprod = at::zeros_like(prod); // 创建与 prod 类型和大小相同的零张量
  for (const auto i : c10::irange(k)) {
    auto v_i = V.narrow(-1, i, 1);    // 提取 V 的第 i 列
    auto dv_i = dV.narrow(-1, i, 1);  // 提取 dV 的第 i 列
    auto tau_i = tau.narrow(-1, i, 1); // 提取 tau 的第 i 个元素
    auto dtau_i = dtau.narrow(-1, i, 1); // 提取 dtau 的第 i 个元素
    auto sigma_i = sigma.narrow(-1, i, 1); // 提取 sigma 的第 i 个元素

    H_plus = apply_householder_reflector(v_i, sigma_i, H_plus, /*left=*/true);

    // `H_minus_dH_i_H_plus` = H_1 * ... * H_{i-1} dH_i * H_{i+1} * ...
    # 计算 H_minus_dH_i_H_plus，这是一个张量乘积和的结果
    H_minus_dH_i_H_plus = H_minus.matmul(
        apply_simple_product(v_i, v_i, dtau_i, H_plus) +
        apply_simple_product(dv_i, v_i, tau_i, H_plus) +
        apply_simple_product(v_i, dv_i, tau_i, H_plus));
    
    # 如果 H_minus_dH_i_H_plus 是 Tensor-Subclass 类型，使用不改变原地的 add 方法
    if (at::isTensorSubclassLike(H_minus_dH_i_H_plus)):
        dprod = dprod.add(H_minus_dH_i_H_plus);
    else:
        # 否则在原地将 H_minus_dH_i_H_plus 添加到 dprod 中
        dprod.add_(H_minus_dH_i_H_plus);

    # 应用 Householder 变换到 H_minus 上，更新 H_minus
    H_minus = apply_householder_reflector(v_i, tau_i, H_minus, /*left=*/false);
  }

  # 返回最终的结果张量 dprod
  return dprod;
}

std::tuple<Tensor, Tensor, Tensor> ormqr_backward(
    const Tensor& grad,  // 输入参数：梯度张量，表示对输出的梯度
    const Tensor& result,  // 输入参数：结果张量，表示 ormqr 操作的输出
    const Tensor& self,  // 输入参数：自身张量，ormqr 操作中的自身张量
    const Tensor& tau,  // 输入参数：tau 张量，ormqr 操作中的 tau 参数
    const Tensor& other,  // 输入参数：其他张量，ormqr 操作中的其他张量
    bool left,  // 输入参数：布尔值，表示 ormqr 操作中的 left 参数
    bool transpose,  // 输入参数：布尔值，表示 ormqr 操作中的 transpose 参数
    std::array<bool, 3> grad_output_mask) {  // 输入参数：布尔数组，指示哪些梯度需要计算
  Tensor self_grad, tau_grad, other_grad;  // 定义自身、tau 和其他的梯度张量变量

  if (!grad.defined()) {  // 如果梯度张量未定义
    return std::make_tuple(self_grad, tau_grad, other_grad);  // 返回空的梯度元组
  }

  const auto self_requires_grad = grad_output_mask[0];  // 获取自身梯度需求的布尔值
  const auto tau_requires_grad = grad_output_mask[1];  // 获取tau梯度需求的布尔值
  const auto other_requires_grad = grad_output_mask[2];  // 获取其他梯度需求的布尔值

  if (other_requires_grad) {  // 如果需要计算其他的梯度
    other_grad = at::ormqr(self, tau, grad, left, !transpose);  // 计算其他的梯度
  }
  if (self_requires_grad || tau_requires_grad) {  // 如果需要计算自身或tau的梯度
    if (left ^ transpose) {  // 如果 left 与 transpose 不同
      // 假设 left = true, transpose = false。与只传递转置参数到householder_product_backward类似。
      // Ormqr 计算 B = H_1 * ... * H_k * A。
      // 相对于 H_i 的灵敏度由 Tr(H_i_plus B B_grad^H H_i_minus dH_i) 给出，
      // 因此，由于 householder_product_backward 遵循 `for i in range(k)`，我们可以重用它，
      // 使用 householder_product_backward.grad = grad 和 householder_product_backward.result = result。
      const auto hpb_grad = !transpose ? grad : grad.mH();  // 计算用于 householder_product_backward 的 grad
      const auto hpb_result = !transpose ? result : result.mH();  // 计算用于 householder_product_backward 的 result
      std::tie(self_grad, tau_grad) =
          householder_product_backward(hpb_grad, hpb_result, self, tau);  // 调用 householder_product_backward
    } else {
      // 假设 left = false, transpose = false。与只传递转置参数到householder_product_backward类似。
      // 在这种情况下，Ormqr 计算 B = H_1 * ... * H_k * A，相对于 H_i 的灵敏度变为 Tr(H_i_plus B_grad^H B H_i_minus dH_k)。
      // 我们可以看到 householder_product_backward 中的 `grad` 和 `result` 的角色被“交换”和“转置”，
      // 为了高效计算 H_k_grad，我们需要以相反顺序计算梯度 (`for i in range(k - 1, -1, -1)`)。
      // 因此，我们使用 householder_product_backward 重用 householder_product_backward.grad = result.mH，
      // householder_product_backward.result = grad.mH，householder_product_backward.flip_order = true。
      const auto hpb_grad = !transpose ? result.mH() : result;  // 计算用于 householder_product_backward 的 grad
      const auto hpb_result = !transpose ? grad.mH() : grad;  // 计算用于 householder_product_backward 的 result
      std::tie(self_grad, tau_grad) = householder_product_backward(
          hpb_grad, hpb_result, self, tau, /*flip_order=*/true);  // 调用 householder_product_backward
    }
  }

  return std::make_tuple(self_grad, tau_grad, other_grad);  // 返回计算得到的梯度元组
}

std::tuple<Tensor, Tensor> polar_backward(
    const Tensor& grad,  // 输入参数：梯度张量，表示对极坐标变换的输出的梯度
    const Tensor& result) {  // 输入参数：结果张量，表示极坐标变换的输出
  Tensor grad_abs, grad_angle;  // 定义极坐标变换的梯度张量变量

  if (grad.defined()) {  // 如果梯度张量已定义
    auto grad_conj = grad.conj();  // 计算梯度的共轭
    // 计算梯度的绝对值，使用共轭梯度和结果的符号函数
    grad_abs = at::real(grad_conj * at::sgn(result));
    // 计算结果乘以虚数单位j (0.0 + 1.0j)，用于后续角度梯度计算
    auto result_mul_1_j = result * Scalar(c10::complex<double>{0.0, 1.0});
    // 计算梯度的角度，使用共轭梯度和结果乘以虚数单位j后的实部
    grad_angle = at::real(grad_conj * result_mul_1_j);
  }
  // 返回计算得到的梯度的绝对值和角度
  return std::make_tuple(grad_abs, grad_angle);
// 定义了一个函数 i1_backward，用于计算某种类型的张量的反向传播梯度
Tensor i1_backward(
    const Tensor& grad,  // 输入参数：梯度
    const Tensor& self,  // 输入参数：自身张量
    const Tensor& result) {  // 输入参数：函数结果张量
  return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "i1_backward", [&]() {
    // 使用 AT_DISPATCH_FLOATING_TYPES 宏，根据张量的数据类型分派计算任务
    // 当自身张量 self 的绝对值大于 epsilon（机器 epsilon）时，标记为非微小值
    auto eps = std::numeric_limits<scalar_t>::epsilon();
    auto self_is_not_tiny = self.abs() > eps;

    // 为避免 NaN，将 self 的微小值部分替换为 eps，得到安全的 self 张量
    auto safe_self =
        at::where(self_is_not_tiny, self, at::full({}, eps, self.options()));

    // 计算 gradx，根据特定函数 i0 的定义和 safe_self，result 的乘积
    auto gradx = (safe_self.i0() - (result * safe_self.reciprocal()));

    // 返回最终的梯度，根据 self_is_not_tiny 条件选择 gradx 或者固定值 0.5
    return grad *
        at::where(self_is_not_tiny, gradx, at::full({}, 0.5, self.options()));
  });
}

// 定义了另一个函数 i1e_backward，用于计算某种类型的张量的反向传播梯度
Tensor i1e_backward(
    const Tensor& grad,  // 输入参数：梯度
    const Tensor& self,  // 输入参数：自身张量
    const Tensor& result) {  // 输入参数：函数结果张量
  return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "i1e_backward", [&]() {
    // 使用 AT_DISPATCH_FLOATING_TYPES 宏，根据张量的数据类型分派计算任务
    // 当自身张量 self 的绝对值大于 epsilon（机器 epsilon）时，标记为非微小值
    auto eps = std::numeric_limits<scalar_t>::epsilon();
    auto self_is_not_tiny = self.abs() > eps;

    // 为避免 NaN，将 self 的微小值部分替换为 eps，得到安全的 self 张量
    auto safe_self =
        at::where(self_is_not_tiny, self, at::full({}, eps, self.options()));

    // 计算 gradx，根据特定函数 i0e 的定义和 safe_self，result 的乘积
    auto gradx =
        (at::special_i0e(safe_self) -
         result * (safe_self.sgn() + safe_self.reciprocal()));

    // 返回最终的梯度，根据 self_is_not_tiny 条件选择 gradx 或者固定值 0.5
    return grad *
        at::where(self_is_not_tiny, gradx, at::full({}, 0.5, self.options()));
  });
}
// 定义函数 linalg_lu_solve_LU，用于计算 LU 分解的线性方程组的解的雅可比乘积
Tensor linalg_lu_solve_LU(
    const Tensor& gX,          // 输入参数 gX，雅可比乘积的梯度
    const Tensor& LU,          // 输入参数 LU，包含 LU 分解的张量
    const Tensor& pivots,      // 输入参数 pivots，LU 分解的置换向量
    const Tensor& X,           // 输入参数 X，LU 分解的输入张量
    const bool left,           // 输入参数 left，指示是否左乘
    const bool adjoint) {      // 输入参数 adjoint，指示是否共轭转置

  // 禁用 TF32 模式，确保计算精度
  at::NoTF32Guard disable_tf32;

  // 解包 LU 分解的结果到 P, L, U
  auto [P, L, U] = at::lu_unpack(
      LU, pivots, /*unpack_data=*/true, /*unpack_pivots=*/left == adjoint);

  // 根据 left 和 adjoint 的不同情况选择不同的路径计算 gL 和 gR
  if (left != adjoint) {
    // 计算 gR = U^{-H} op_2(-gX) op_2(X)^H
    auto gR = at::linalg_solve_triangular(
        U.mH(),
        -(left ? gX : gX.mH()).matmul(left ? X.mH() : X),
        /*upper*/ false);

    // 计算 gL = (L^{-H} gR U^H).tril(-1)
    auto gL = at::linalg_solve_triangular(
                  L.mH(),
                  gR.matmul(U.mH()),
                  /*upper*/ true,
                  /*left*/ true,
                  /*unitriangular*/ true)
                  .tril(-1);

    // 返回 gL + gR 的结果作为 LU_grad
    return gL + gR.triu();

  } else {
    // 计算 gR = -P^T op_3(X) op_1(op_2(gX)) P
    auto gR =
        -P.mT().matmul(left ? X : X.mH()).matmul(left ? gX.mH() : gX).matmul(P);

    // 计算 gL = gR.tril(-1)
    auto gL = gR.tril(-1);

    // 返回 gL + gU 的结果作为 LU_grad
    return gL + (L.matmul(gR)).triu();
  }
}
    // 使用 Cholesky 分解得到的下三角矩阵 L 的共轭转置 mH() 求解线性方程组 L^H x = gR，并返回结果至 gR
    gR = at::linalg_solve_triangular(
        L.mH(), gR, /*upper*/ true, /*left*/ false, /*unitriangular*/ true);
    
    // 计算 L 的共轭转置 mH() 与 gR 的矩阵乘积，然后使用 Cholesky 分解得到的上三角矩阵 U 的共轭转置 mH() 求解线性方程组 U^H x = (L^H gR)
    // 将结果取上三角部分
    auto gU = at::linalg_solve_triangular(
                  U.mH(), L.mH().matmul(gR), /*upper*/ false, /*left*/ false)
                  .triu();
    
    // 返回 gR 的下三角矩阵（除了主对角线以下的部分）与 gU（上三角矩阵）的和
    return gR.tril(-1) + gU;
}
// 定义函数 linalg_lu_solve_jvp，计算 LU 分解后的线性方程组的 JVP（Jacobian Vector Product）
Tensor linalg_lu_solve_jvp(
    const Tensor& X,        // 输入张量 X
    const Tensor& LU,       // LU 分解的结果
    const Tensor& pivots,   // LU 分解的置换向量
    const Tensor& dLU,      // LU 分解的导数
    const Tensor& dB,       // 输出张量 B 的导数
    const bool left,        // 指示左侧乘法
    const bool adjoint) {   // 指示是否共轭转置

  // 根据不同的 left 和 adjoint 组合，推导出不同的线性方程组求解方式
  // left = True, adjoint = True: A^H X = B
  // left = True, adjoint = False: A X = B
  // left = False, adjoint = True: A X^H = B^H
  // left = False, adjoint = False: A^H X^H = B^H

  // 计算 lu_solve(LU, pivots, dB, left, adjoint)，得到 S
  at::NoTF32Guard disable_tf32;
  auto S = at::linalg_lu_solve(LU, pivots, dB, left, adjoint);

  if (left != adjoint) {
    // 当 left != adjoint 时，使用 A^{-1}op_3(B) = op_2(X) 的替换
    // 计算 R = -U^{-1}(dUU^{-1} + L^{-1}dL L^{-1} P^T)
    auto R = at::linalg_solve_triangular(
        LU,
        dLU.tril(-1),
        /*upper*/ false,
        /*left*/ true,
        /*unitriangular*/ true);
    auto U = LU.triu();
    R = -at::linalg_solve_triangular(
        U, dLU.triu() + R.matmul(U), /*upper*/ true);

    // 返回结果 dX = op_2(R op_2(X)) + S
    return (left ? R.matmul(X) : X.matmul(R.mH())) + S;
  } else {
    // 当 left == adjoint 时，使用 op_1(A) = A^H 的替换
    // 计算 R = op_3(X)^H P(LdUU^{-1} + dL L^{-1} P^T)
    auto [P, L, U] = at::lu_unpack(LU, pivots);
    auto V = left ? X.mH() : X;
    auto R = at::linalg_solve_triangular(
                 U, L.matmul(dLU.triu()), /*upper*/ true, /*left*/ false) +
        dLU.tril(-1);
    R = at::linalg_solve_triangular(
            L,
            -V.matmul(P).matmul(R),
            /*upper*/ false,
            /*left*/ false,
            /*unitriangular*/ true)
            .matmul(P.mT());

    // 返回结果 dX = op_2(R^H) + S
    return (left ? R.mH() : std::move(R)) + S;
  }
}
    const bool use_A_T) {
```py  

// 禁用 TF32 加速
at::NoTF32Guard disable_tf32;


  // 对于 left=True 的情况（left=False 类似）
  // dX = A^{-1}(dB - dAX)


  // [NumPy 兼容] 当右手边是向量时的情况。
  // 我们使用下划线表示已经通过 `unsqueeze(-1)` 转换为矩阵的向量。
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(LU, X);
  const auto vector_to_matrix = [vector_case](const Tensor& X) {
    return vector_case ? X.unsqueeze(-1) : X;
  };
  const auto matrix_to_vector = [vector_case](const Tensor& X) {
    return vector_case ? X.squeeze(-1) : X;
  };


  // 在原始操作中禁止这种情况，因为 A.shape = (*, 1, 1)
  TORCH_INTERNAL_ASSERT(left || !vector_case);


  auto X_ = vector_to_matrix(X);
  auto dB_ = vector_to_matrix(dB);
  auto R_ = left ? dA.matmul(X_) : X_.matmul(dA);
  auto dX_ =
      at::linalg_lu_solve(LU, pivots, dB_ - R_, left, /*adjoint*/ use_A_T);
  return matrix_to_vector(dX_);
}

std::tuple<Tensor, Tensor> linalg_solve_backward(
    const Tensor& gX,
    const Tensor& X,
    const Tensor& A,
    const Tensor& LU,
    const Tensor& pivots,
    const bool left,
    const bool B_requires_grad) {
  // for X = A^{-1}B
  // gB = A^{-H}gX
  // gA = -gB X^H
  // 禁用 TF32
  at::NoTF32Guard disable_tf32;
  // 检查 A 和 B 是否需要梯度
  const auto A_requires_grad = A.requires_grad();
  // 如果 gX 未定义，或者 A 和 B 均不需要梯度，则返回空元组
  if (!gX.defined() || (!A_requires_grad && !B_requires_grad)) {
    return {};
  }

  // [NumPy compat] Case where the rhs is a vector.
  // We denote with an underscore vectors that have been converted to matrices
  // by `unsqueeze(-1)`
  // 判断是否是向量情况，将向量转换为矩阵
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(LU, X);
  const auto vector_to_matrix = [vector_case](const Tensor& X) {
    return vector_case ? X.unsqueeze(-1) : X;
  };
  const auto matrix_to_vector = [vector_case](const Tensor& X) {
    return vector_case ? X.squeeze(-1) : X;
  };

  // 如果用户需要计算高阶梯度，则需要重新计算 LU 分解和置换
  Tensor gB_;
  if (at::GradMode::is_enabled()) {
    // 计算 gB，A^H 是 A 的共轭转置
    gB_ = at::linalg_solve(A.mH(), vector_to_matrix(gX), left);
  } else {
    const auto use_A_T = A.is_contiguous() && !A.is_complex();
    // 使用 LU 分解求解 gB
    gB_ = at::linalg_lu_solve(
        LU, pivots, vector_to_matrix(gX), left, /*adjoint*/ !use_A_T);
  }

  Tensor gA_;
  if (A_requires_grad) {
    auto X_ = vector_to_matrix(X);
    // 计算 gA
    gA_ = left ? -gB_.matmul(X_.mH()) : -X_.mH().matmul(gB_);
  }
  // 返回 gA 和 gB
  return std::make_tuple(
      A_requires_grad ? std::move(gA_) : Tensor{},
      B_requires_grad ? matrix_to_vector(gB_) : Tensor{});
}

Tensor solve_jvp(
    const Tensor& X,
    const Tensor& A,
    const Tensor& dA,
    const Tensor& dB) {
  // 调用通用的 Jacobian-Vector Product 函数
  return generic_solve_jvp(
      [](const Tensor& A, const Tensor& dB, const Tensor& dA_contrib) {
        // 使用 linalg_solve 求解 A * dx = dB - dA_contrib
        return at::linalg_solve(A, dB - dA_contrib);
      },
      X,
      A,
      dA,
      dB);
}

Tensor lu_unpack_backward(
    const Tensor& L_grad,
    const Tensor& U_grad,
    const c10::SymInt& m,
    const c10::SymInt& n) {
  // 如果 L_grad 和 U_grad 均未定义，则返回空张量
  if (!L_grad.defined() && !U_grad.defined()) {
    return {};
  }
  const auto k = std::min(m, n);

  // 获取矩阵的主要部分和补充部分
  const auto get_L1 = [m, k](const Tensor& L) {
    return m == k ? L.tril(-1) : L.narrow_symint(-2, 0, k).tril(-1);
  };
  const auto get_L2 = [m, k](const Tensor& L) {
    return L.narrow_symint(-2, k, m - k);
  };
  const auto get_U1 = [n, k](const Tensor& U) {
    return n == k ? U.triu() : U.narrow_symint(-1, 0, k).triu();
  };
  const auto get_U2 = [n, k](const Tensor& U) {
    return U.narrow_symint(-1, k, n - k);
  };

  if (L_grad.defined()) {
    // 对 L 的梯度进行 LU 分解的反向传播
    # 检查是否定义了上三角梯度U_grad
    if (U_grad.defined()) {
      # 如果行数m等于列数n
      if (m == n) {
        # 返回下三角部分的梯度L_grad（除对角线外）和上三角部分的梯度U_grad（包括对角线）
        return L_grad.tril(-1) + U_grad.triu();
      } else {
        # 计算A1_grad为L_grad的第一部分梯度和U_grad的第一部分梯度
        auto A1_grad = get_L1(L_grad) + get_U1(U_grad);
        # 计算A2_grad为行数m大于列数n时的L_grad的第二部分梯度，否则为U_grad的第二部分梯度
        auto A2_grad = m > n ? get_L2(L_grad) : get_U2(U_grad);
        # 确定拼接的维度，当行数m大于列数n时为-2，否则为-1
        const auto dim = m > n ? -2 : -1;
        # 返回A1_grad和A2_grad在指定维度上的拼接结果
        return at::cat({std::move(A1_grad), std::move(A2_grad)}, /*dim=*/dim);
      }
    } else {
      # 如果未定义上三角梯度U_grad
      if (m >= n) {
        # 返回L_grad的下三角部分梯度（除对角线外）
        return L_grad.tril(-1);
      } else {
        # 计算size为L_grad的对称大小的向量，并调整其最后一个维度
        auto size = L_grad.sym_sizes().vec();
        size.end()[-1] = n - m;
        # 返回L_grad的下三角部分梯度（除对角线外）和填充零的对称张量
        return at::cat(
            {L_grad.tril(-1), at::zeros_symint(size, L_grad.options())},
            /*dim=*/-1);
      }
    }
  } else {
    # 如果未定义下三角梯度L_grad
    if (n >= m) {
      # 返回U_grad的上三角部分梯度（包括对角线）
      return U_grad.triu();
    } else {
      # 计算size为U_grad的对称大小的向量，并调整其倒数第二个维度
      auto size = U_grad.sym_sizes().vec();
      size.end()[-2] = m - n;
      # 返回U_grad的上三角部分梯度（包括对角线）和填充零的对称张量
      return at::cat(
          {U_grad.triu(), at::zeros_symint(size, U_grad.options())},
          /*dim=*/-2);
    }
  }
// 计算传递到后向梯度的张量的“cat”操作结果
Tensor cat_jvp(const at::ITensorListRef& tensors, int64_t dim) {
  // 初始化输出梯度张量
  Tensor out_fw_grad;

  // 将未实例化的张量列表转换为实例化的张量列表
  auto materialized = tensors.materialize();
  // 检查是否存在任何已定义的前向梯度
  auto any_defined = false;
  for (const Tensor& t : materialized) {
    any_defined |= isFwGradDefined(t);
  }

  // 如果存在已定义的前向梯度，执行以下操作
  if (any_defined) {
    // 初始化前向梯度张量列表
    std::vector<Tensor> fw_grads;

    // 遍历每个实例化的张量，生成对应的前向梯度张量
    for (const Tensor& t : materialized) {
      fw_grads.push_back(
          isFwGradDefined(t)
              ? t._fw_grad(/*level*/ 0)
              : at::_efficientzerotensor(t.sizes(), t.options()));
    }

    // 对生成的前向梯度张量列表进行“cat”操作，沿指定维度连接
    out_fw_grad = at::cat(fw_grads, dim);
  }

  // 返回合并后的前向梯度张量
  return out_fw_grad;
}

// 计算传递到后向梯度的张量的“block_diag”操作结果
Tensor block_diag_jvp(at::TensorList tensors) {
  // 初始化输出梯度张量
  Tensor out_fw_grad;

  // 检查是否存在任何已定义的前向梯度
  auto any_defined = false;
  for (const auto& t : tensors) {
    any_defined |= isFwGradDefined(t);
  }

  // 如果存在已定义的前向梯度，执行以下操作
  if (any_defined) {
    // 初始化前向梯度张量列表
    std::vector<Tensor> fw_grads;
    fw_grads.reserve(tensors.size());

    // 遍历每个张量，生成对应的前向梯度张量
    for (const auto& t : tensors) {
      fw_grads.push_back(
          isFwGradDefined(t)
              ? t._fw_grad(/*level*/ 0)
              : at::_efficientzerotensor(t.sizes(), t.options()));
    }

    // 对生成的前向梯度张量列表进行“block_diag”操作，生成分块对角矩阵
    out_fw_grad = at::block_diag(fw_grads);
  }

  // 返回生成的分块对角矩阵的前向梯度张量
  return out_fw_grad;
}

// 计算传递到后向梯度的张量的“stack”操作结果
Tensor stack_jvp(at::TensorList tensors, int64_t dim) {
  // 初始化输出梯度张量
  Tensor out_fw_grad;

  // 检查是否存在任何已定义的前向梯度
  auto any_defined = false;
  for (const auto& t : tensors) {
    any_defined |= isFwGradDefined(t);
  }

  // 如果存在已定义的前向梯度，执行以下操作
  if (any_defined) {
    // 初始化前向梯度张量列表
    std::vector<Tensor> fw_grads;

    // 遍历每个张量，生成对应的前向梯度张量
    for (auto& t : tensors) {
      fw_grads.push_back(
          isFwGradDefined(t)
              ? t._fw_grad(/*level*/ 0)
              : at::_efficientzerotensor(t.sizes(), t.options()));
    }

    // 对生成的前向梯度张量列表进行“stack”操作，沿指定维度堆叠张量
    out_fw_grad = at::stack(fw_grads, dim);
  }

  // 返回堆叠后的前向梯度张量
  return out_fw_grad;
}

// 计算传递到后向梯度的张量的“cumprod”操作结果
Tensor cumprod_jvp(
    const Tensor& self_t,
    const Tensor& self_p,
    const Tensor& result,
    int dim) {
  // 使用通用公式计算梯度，当没有 0 参与时
  Tensor gradient = (self_t / self_p).cumsum(dim) * result;

  // 注意我们必须使用 at::where，因为我们要移除 NaN 值

  // 如果 self_p 的维度为 0，执行以下操作
  if (self_p.dim() == 0) {
    // 使用 self_p 等于 0 的位置来掩盖梯度
    gradient.masked_fill_(self_p.eq(0), self_t);
    // 返回修正后的梯度
    return gradient;
  } else {
    // 对于输入 (a, 0, b, 0, c)，并且对应的梯度 (t0, t1, t2, t3, t4)
    // cumprod 的输出为 (a, 0, 0, 0, 0)
    // 我们希望计算的梯度为 (t0, a*t1, a*b*t1, 0, 0)
    // 我们通过以下步骤来实现：
    // 获取所有零值的掩码 (0, 1, 0, 1, 0)
    auto mask_zeros = self_p.eq(0);
    // 获取每个维度的第一个零值的掩码 (0, 1, 0, 0, 0)
    auto mask_first_zero = mask_zeros.logical_and(mask_zeros.cumsum(dim).eq(1));

    // 获取应该在任何零值发生后使用的新梯度值：
    // (X, a*t1, a*b*t1, 0, 0) = cumprod((a, t1, b, 0, c))
    auto new_grad = at::where(mask_first_zero, self_t, self_p).cumprod(dim);

    // 获取第一个零值后的所有内容的掩码：(0, 1, 1, 1, 1)
    auto mask_after_first_zero = mask_first_zero.cumsum(dim);

    // 执行最终替换操作
    # 使用 PyTorch 的 where 函数根据条件选择返回新的梯度或原始梯度
    return at::where(
        # 将 mask_after_first_zero 转换为布尔类型的标量，作为条件
        mask_after_first_zero.to(ScalarType::Bool),
        # 如果条件为真，则返回 new_grad
        new_grad,
        # 如果条件为假，则返回原始梯度 gradient
        gradient);
    # 函数结束，返回根据条件选择后的梯度结果
// Helper for {batch,layer,group}_norms below
// Computes the jvp for `1 / input.std(dims, keepdim)`
static Tensor _invstd_jvp(
    const Tensor& input_p,    // Parameter: original input tensor
    const Tensor& input_t,    // Tangent: perturbed input tensor
    const Tensor& mean_p,     // Parameter: mean of the original input tensor
    const Tensor& invstd_p,   // Parameter: inverse standard deviation of the original input tensor
    IntArrayRef dims,         // Dimensions along which to compute std and mean
    int64_t numel,            // Number of elements in the tensor
    bool keepdim) {           // Flag indicating whether to keep the dimensions
  Tensor invstd_t;           // Declare tensor to hold the result of inverse std deviation JVP
  // Check if any of the inputs are subclasses or if input_t is a zero tensor
  if (areAnyTensorSubclassLike({input_t, input_p, mean_p, invstd_p}) ||
      input_t._is_zerotensor()) {
    // Compute the Jacobian vector product (JVP) using the formula for zero tensor or subclasses
    invstd_t = -invstd_p.pow(3) * (input_t - input_t.mean(dims, true)) *
        (input_p - mean_p);
  } else {
    // Compute the JVP using the standard formula
    invstd_t = input_t - input_t.mean(dims, true);
    invstd_t *= input_p - mean_p;
    invstd_t *= -invstd_p.pow(3);
  }
  // Sum along the specified dimensions and normalize
  invstd_t = invstd_t.sum(dims, keepdim);
  invstd_t /= numel;
  return invstd_t;           // Return the computed JVP for inverse standard deviation
}

// Helper for {batch,layer,group}_norms below only
// Computes the jvp for `(input - input.mean(dims)) * input.invstd(dims)`
static Tensor _norm_jvp(
    const Tensor& input_p,    // Parameter: original input tensor
    const Tensor& input_t,    // Tangent: perturbed input tensor
    const Tensor& mean_p,     // Parameter: mean of the original input tensor
    const Tensor& invstd_p,   // Parameter: inverse standard deviation of the original input tensor
    IntArrayRef dims,         // Dimensions along which to compute mean and invstd
    int64_t numel) {          // Number of elements in the tensor
  auto invstd_t =
      _invstd_jvp(input_p, input_t, mean_p, invstd_p, dims, numel, true); // Compute invstd JVP
  Tensor result_t;           // Declare tensor to hold the result of norm JVP
  // Check if any of the inputs are subclasses or if input_t is a zero tensor
  if (areAnyTensorSubclassLike({input_t, input_p, mean_p, invstd_p}) ||
      input_t._is_zerotensor()) {
    // Compute the Jacobian vector product (JVP) using the formula for zero tensor or subclasses
    result_t = (input_t - input_t.mean(dims, true)) * invstd_p +
        (input_p - mean_p) * invstd_t;
  } else {
    // Compute the JVP using the standard formula
    result_t = input_t - input_t.mean(dims, true);
    result_t *= invstd_p;
    auto temp = input_p - mean_p;
    temp *= invstd_t;
    result_t += temp;
  }
  return result_t;           // Return the computed JVP for normalization
}

// Helper for {batch,layer,group}_norms below only
// Computes the jvp for `input * weight + bias` where weight and bias may be
// undefined Possibly modifies the input inplace
static Tensor _affine_jvp(
    const std::optional<Tensor>& input_p,  // Optional parameter: original input tensor
    Tensor& input_t,                      // Reference to perturbed input tensor
    const Tensor& weight_p,                // Parameter: weight tensor of the original input
    const Tensor& weight_t,                // Tangent: perturbed weight tensor
    const Tensor& bias_t) {                // Tangent: perturbed bias tensor
  // Assertion to check the validity of input_p and weight_p
  TORCH_INTERNAL_ASSERT(input_p.has_value() == weight_p.defined());
  // Check if any of the inputs are subclasses or if input_t and weight_t are zero tensors
  if (weight_p.defined()) {
    if (areAnyTensorSubclassLike(
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            {input_p.value(), input_t, weight_p, weight_t}) ||
        input_t._is_zerotensor() || weight_t._is_zerotensor()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      input_t = input_t * weight_p + input_p.value() * weight_t;  // Compute affine JVP for subclasses or zero tensors
    } else {
      input_t *= weight_p;    // Standard computation for input_t multiplied by weight_p
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      auto temp = input_p.value();
      temp *= weight_t;       // Multiply temp by weight_t
      input_t += temp;        // Add temp to input_t
    }
  }
  // Check if bias_t is defined and compute affine JVP for it
  if (bias_t.defined()) {
    if (areAnyTensorSubclassLike({input_t, bias_t}) ||
        input_t._is_zerotensor()) {
      input_t = input_t + bias_t;  // Compute affine JVP for subclasses or zero tensors
    } else {
      input_t += bias_t;    // Add bias_t to input_t
    }
  }
  return input_t;           // Return the computed affine JVP
}
  // 定义一个空的整数向量 dims
  auto dims = std::vector<int64_t>{};
  // 创建一个向量 view_size，包含输入张量 input_t 的尺寸
  auto view_size = input_t.sizes().vec();
  // 初始化元素总数为 1
  int64_t numel = 1;
  // 遍历 view_size 中的每一个维度
  for (const auto dim : c10::irange(view_size.size())) {
    // 如果维度不为 1
    if (dim != 1) {
      // 计算元素总数
      numel *= input_t.size(static_cast<int64_t>(dim));
      // 将当前维度设为 1
      view_size[dim] = 1;
      // 将当前维度加入 dims 中
      dims.push_back(static_cast<int64_t>(dim));
    }
  }
  // 定义三个张量 mean_p, invstd_p, result_t
  Tensor mean_p;
  Tensor invstd_p;
  Tensor result_t;
  // 如果处于训练模式
  if (train) {
    // 将 saved_mean 和 saved_invstd 按照 view_size 的尺寸进行视图重塑
    mean_p = saved_mean.view(view_size);
    invstd_p = saved_invstd.view(view_size);
    // 调用 _norm_jvp 函数计算结果 result_t
    result_t = _norm_jvp(input_p, input_t, mean_p, invstd_p, dims, numel);
  } else {
    // 断言 running_mean 和 running_var 必须有值
    TORCH_INTERNAL_ASSERT(
        running_mean.has_value() && running_var.has_value(),
        "Expect running_mean and running_var to have value when train=false");
    // 检查 running_mean 和 running_var 是否没有前向梯度定义
    TORCH_CHECK(
        !running_mean.value()._fw_grad(/*level=*/0).defined() &&
            !running_var.value()._fw_grad(/*level=*/0).defined(),
        "batch_norm is not differentiable wrt running_mean and running_var, they cannot have forward grad defined");
    // 将 running_mean 按照 view_size 的尺寸进行视图重塑
    mean_p = running_mean.value().view(view_size);
    // 计算 invstd_p，按照 view_size 的尺寸进行视图重塑
    invstd_p =
        (1 / at::sqrt(running_var.value() + at::Scalar(eps))).view(view_size);
    // 计算 result_t
    result_t = input_t * invstd_p;
  }

  // 定义一个可选的张量 result_p
  std::optional<Tensor> result_p = weight_p.defined()
      ? std::optional<Tensor>((input_p - mean_p) * invstd_p)
      : c10::nullopt;
  // 调用 _affine_jvp 函数，返回结果
  return _affine_jvp(
      result_p,
      result_t,
      // 如果 weight_p 已定义，则按照 view_size 的尺寸进行视图重塑，否则直接使用 weight_p
      weight_p.defined() ? weight_p.view(view_size) : weight_p,
      // 如果 weight_t 已定义，则按照 view_size 的尺寸进行视图重塑，否则直接使用 weight_t
      weight_t.defined() ? weight_t.view(view_size) : weight_t,
      // 如果 bias_t 已定义，则按照 view_size 的尺寸进行视图重塑，否则直接使用 bias_t
      bias_t.defined() ? bias_t.view(view_size) : bias_t);
// 定义函数 `layer_norm_jvp`，用于计算层归一化的 Jacobian 向量积
Tensor layer_norm_jvp(
    const Tensor& input_p,                // 输入的反向传播梯度
    const Tensor& input_t,                // 输入的正向传播值
    const Tensor& weight_p,               // 权重的反向传播梯度
    const Tensor& weight_t,               // 权重的正向传播值
    const Tensor& bias_p,                 // 偏置的反向传播梯度
    const Tensor& bias_t,                 // 偏置的正向传播值
    const Tensor& saved_mean,             // 保存的均值
    const Tensor& saved_invstd,           // 保存的标准差的倒数
    c10::SymIntArrayRef normalized_shape  // 归一化的形状
) {
  auto dims = std::vector<int64_t>{};     // 存储维度的向量
  auto view_size = input_t.sizes().vec(); // 正向传播值的尺寸向量
  auto view_size_affine = input_t.sizes().vec();  // 用于仿射操作的尺寸向量

  int64_t numel = 1;                      // 元素总数初始化为1
  // 遍历输入的尺寸向量
  for (const auto i : c10::irange(view_size.size())) {
    // 如果当前维度小于归一化形状的维度数量
    if (i < view_size.size() - normalized_shape.size()) {
      view_size_affine[i] = 1;            // 将仿射操作的尺寸设置为1
    } else {
      numel *= input_t.size(static_cast<int64_t>(i));  // 更新总元素数
      view_size[i] = 1;                   // 将正向传播尺寸向量中的当前维度设置为1
      dims.push_back(static_cast<int64_t>(i));  // 将当前维度加入到 dims 中
    }
  }
  auto mean_p = saved_mean.view(view_size);      // 使用正向传播尺寸重新视图保存的均值
  auto invstd_p = saved_invstd.view(view_size);  // 使用正向传播尺寸重新视图保存的标准差的倒数
  auto result_t = _norm_jvp(input_p, input_t, mean_p, invstd_p, dims, numel);  // 计算归一化操作的 Jacobian 向量积

  std::optional<Tensor> result_p = weight_p.defined()
      ? std::optional<Tensor>((input_p - mean_p) * invstd_p)  // 如果定义了权重，计算其反向传播梯度
      : c10::nullopt;                     // 否则置为空

  // 返回仿射操作的 Jacobian 向量积结果
  return _affine_jvp(
      result_p,
      result_t,
      weight_p.defined() ? weight_p.view(view_size_affine) : weight_p,  // 根据是否定义了权重，重新视图权重张量
      weight_t.defined() ? weight_t.view(view_size_affine) : weight_t,  // 根据是否定义了权重，重新视图权重张量
      bias_t.defined() ? bias_t.view(view_size_affine) : bias_t);        // 根据是否定义了偏置，重新视图偏置张量
}

// 定义函数 `group_norm_jvp`，用于计算分组归一化的 Jacobian 向量积
Tensor group_norm_jvp(
    const Tensor& input_p,                // 输入的反向传播梯度
    const Tensor& input_t,                // 输入的正向传播值
    const Tensor& weight_p,               // 权重的反向传播梯度
    const Tensor& weight_t,               // 权重的正向传播值
    const Tensor& bias_p,                 // 偏置的反向传播梯度
    const Tensor& bias_t,                 // 偏置的正向传播值
    const Tensor& saved_mean,             // 保存的均值
    const Tensor& saved_invstd,           // 保存的标准差的倒数
    int64_t groups                        // 分组数量
) {
  auto input_shape = input_p.sizes();     // 输入的形状
  int64_t N = input_p.size(0);            // 批次大小
  int64_t C = input_p.size(1);            // 通道数

  auto input_t_reshaped = input_t.view({1, N * groups, N ? -1 : 1});   // 重新形状化的正向传播值
  auto input_p_reshaped = input_p.view({1, N * groups, N ? -1 : 1});   // 重新形状化的反向传播梯度

  auto result_t = batch_norm_jvp(          // 计算批次归一化的 Jacobian 向量积结果
                      input_p_reshaped,
                      input_t_reshaped,
                      /*weight_p=*/{},    // 权重的反向传播梯度置为空
                      /*weight_t=*/{},    // 权重的正向传播值置为空
                      /*bias_p=*/{},      // 偏置的反向传播梯度置为空
                      /*bias_t=*/{},      // 偏置的正向传播值置为空
                      /*running_mean=*/{},// 运行时均值置为空
                      /*running_var=*/{}, // 运行时方差置为空
                      saved_mean,
                      saved_invstd,
                      /*train=*/true,     // 训练模式
                      /*eps=*/0)          // epsilon 参数设为0
                      .view(input_shape); // 根据输入形状视图化结果

  std::optional<Tensor> result_p = c10::nullopt;  // 初始化反向传播梯度为为空
  if (weight_p.defined()) {
    std::vector<int64_t> view_size(input_t_reshaped.dim(), 1);  // 初始化视图尺寸为当前尺寸的维度数量为1的向量
    view_size[1] = input_t_reshaped.size(1);  // 更新视图尺寸的第二个维度为重新形状化正向传播值的第二个尺寸
    result_p = ((input_p_reshaped - saved_mean.view(view_size)) *
                saved_invstd.view(view_size))
                   .view(input_shape);

# 计算归一化后的输出结果 result_p。
# input_p_reshaped 是重塑后的输入张量。
# saved_mean 和 saved_invstd 是存储的均值和标准差的张量。
# view(view_size) 用于将 saved_mean 和 saved_invstd 重塑为与 input_p_reshaped 兼容的形状。
# 最后通过 element-wise 的减法和乘法计算 result_p，并使用 view(input_shape) 重新调整形状。


  std::vector<int64_t> affine_param_shape(input_p.dim(), 1);
  affine_param_shape[1] = C;

# 创建一个整型向量 affine_param_shape，其长度与 input_p 的维度相同，初始值均为 1。
# 将第二个元素（索引为 1）设置为 C，用于描述亚流形参数的形状。


  return _affine_jvp(
      result_p,
      result_t,
      weight_p.defined() ? weight_p.view(affine_param_shape) : weight_p,
      weight_t.defined() ? weight_t.view(affine_param_shape) : weight_t,
      bias_t.defined() ? bias_t.view(affine_param_shape) : bias_t);

# 调用 _affine_jvp 函数计算仿射变换的 JVP（Jacobian Vector Product）。
# 将 result_p 和 result_t 作为输入结果和输入梯度。
# 根据权重和偏置的定义状态，对它们进行必要的视图重塑（reshape）处理，以保证与亚流形参数形状匹配。
# 返回 _affine_jvp 函数的结果作为整个函数的返回值。
}

// 定义一个函数 group_norm_mean_jvp，计算输入张量的分组归一化均值的 Jacobian 向量乘积
Tensor group_norm_mean_jvp(
    const Tensor& input_t,  // 输入张量
    const Tensor& mean_p,   // 均值张量
    int64_t groups) {       // 分组数目
  int64_t N = input_t.size(0);  // 获取输入张量的第一维大小
  std::array<int64_t, 3> view_shape = {1, N * groups, N ? -1 : 1};  // 定义重塑形状的数组
  auto input_t_reshaped = input_t.view(view_shape);  // 对输入张量进行重塑
  return input_t_reshaped.mean({2}, false).view_as(mean_p);  // 计算均值并返回与均值张量相同形状的张量
}

// 定义一个函数 group_norm_invstd_jvp，计算输入张量的分组归一化标准差的 Jacobian 向量乘积
Tensor group_norm_invstd_jvp(
    const Tensor& input_p,    // 输入张量
    const Tensor& input_t,    // 输入张量
    const Tensor& mean_p,     // 均值张量
    const Tensor& invstd_p,   // 标准差的倒数张量
    int64_t groups) {         // 分组数目
  int64_t N = input_p.size(0);  // 获取输入张量的第一维大小

  std::vector<int64_t> view_shape = {1, N * groups, N ? -1 : 1};  // 定义重塑形状的向量

  auto input_t_reshaped = input_t.view(view_shape);   // 对输入张量进行重塑
  auto input_p_reshaped = input_p.view(view_shape);   // 对输入张量进行重塑

  return _invstd_jvp(
             input_t_reshaped,    // 重塑后的输入张量
             input_p_reshaped,    // 重塑后的输入张量
             mean_p.view(view_shape),    // 重塑后的均值张量
             invstd_p.view(view_shape),  // 重塑后的标准差的倒数张量
             /*dims=*/{2},    // 维度为2
             /*numel=*/input_t_reshaped.size(2),  // 元素数目为重塑后的第二维大小
             /*keepdim=*/false)   // 不保持维度
      .view_as(invstd_p);   // 返回与标准差的倒数张量相同形状的张量
}

// 定义一个函数 gather_with_keepdimed_indices，执行带有保持维度索引的 gather 操作
Tensor gather_with_keepdimed_indices(
    const Tensor& input,    // 输入张量
    int64_t dim,            // 维度
    const Tensor& indices,  // 索引张量
    bool keepdim) {         // 是否保持维度
  auto full_indices = indices;   // 全索引等于索引张量
  if (!keepdim) {   // 如果不保持维度
    full_indices = indices.unsqueeze(dim);   // 在指定维度上增加一个维度
  }
  auto out_fw_grad = at::gather(input, dim, full_indices);   // 执行 gather 操作
  if (!keepdim) {   // 如果不保持维度
    out_fw_grad = out_fw_grad.squeeze(dim);   // 在指定维度上压缩张量
  }

  return out_fw_grad;   // 返回输出的前向梯度张量
}
// Define a function linalg_lu_backward that computes gradients for LU decomposition.
Tensor linalg_lu_backward(
    const Tensor& L_grad,            // Gradient of output with respect to L matrix
    const Tensor& U_grad,            // Gradient of output with respect to U matrix
    const Tensor& P,                 // Permutation matrix P
    const Tensor& L,                 // Lower triangular matrix L from LU decomposition
    const Tensor& U,                 // Upper triangular matrix U from LU decomposition
    const bool pivot) {              // Flag indicating whether to apply pivoting

  at::NoTF32Guard disable_tf32;     // Disable TF32 if enabled

  // Return an empty tensor if both L_grad and U_grad are not defined
  if (!L_grad.defined() && !U_grad.defined()) {
    return {};
  }

  // Retrieve sizes of dimensions
  auto m = L.sym_size(-2);          // Size of L's second-to-last dimension
  auto n = U.sym_size(-1);          // Size of U's last dimension
  auto k = std::min(m, n);          // Compute the minimum of m and n

  if (m == n) {
    // Square case: A_grad = P L^{-H} [L^H L_grad o 1_L + U_grad U^H o 1_U] U^{-H}
    auto A_grad = L_grad.defined() ? L.mH().matmul(L_grad).tril(-1) : Tensor{};
    if (U_grad.defined()) {
      A_grad = A_grad.defined() ? A_grad + U_grad.matmul(U.mH()).triu() : U_grad.matmul(U.mH()).triu();
    }
    A_grad = at::linalg_solve_triangular(
        U.mH(),
        A_grad,
        /*upper=*/false,
        /*left=*/false);
    A_grad = at::linalg_solve_triangular(
        L.mH(),
        A_grad,
        /*upper=*/true,
        /*left=*/true,
        /*unitriangular=*/true);

    // Return P * A_grad if pivot is true, otherwise return A_grad
    return pivot ? P.matmul(A_grad) : std::move(A_grad);
  } else if (m < n) {
    // Wide case: A1_grad = P L^{-H} [U1_grad + (L^H L_grad o 1_L - U_grad U^H o 1_U) U1^{-H}] U^{-H}
    //            A2_grad = P L^{-H} U2_grad
    const auto get_U1 = [n, k](const Tensor& U) {
      return n == k ? U : U.narrow_symint(-1, 0, k);
    };
    const auto get_U2 = [n, k](const Tensor& U) {
      return U.narrow_symint(-1, k, n - k);
    };

    auto A_grad = L_grad.defined() ? L.mH().matmul(L_grad) : Tensor{};
    if (U_grad.defined()) {
      A_grad = A_grad.defined() ? A_grad - U_grad.triu().matmul(U.mH()) : -U_grad.triu().matmul(U.mH());
    }
    // 解决线性方程组 A_grad = U1^H * A_grad * U1 下三角部分，其中 U1 是 U 的部分视图
    A_grad = at::linalg_solve_triangular(
        get_U1(U).mH(),                      // 左侧上三角矩阵 U1^H
        A_grad.tril(-1),                     // 右侧下三角部分 A_grad
        /*upper=*/false,                     // U1^H 是下三角矩阵
        /*left=*/false);                     // 在右侧乘法 A_grad

    // 如果定义了 U_grad，则更新 A_grad 为 [A_grad + U1_grad^H 上三角部分, U2_grad] 的拼接
    if (U_grad.defined()) {
      A_grad =
          at::cat({A_grad + get_U1(U_grad).triu(), get_U2(U_grad)}, /*dim=*/-1);
    }

    // 解决线性方程组 A_grad = L^H * A_grad，其中 L 是 L 的部分视图
    A_grad = at::linalg_solve_triangular(
        L.mH(),                              // 左侧上三角矩阵 L^H
        A_grad,                              // 右侧 A_grad
        /*upper=*/true,                      // L^H 是上三角矩阵
        /*left=*/true,                       // 在左侧乘法 A_grad
        /*unitriangular=*/true);             // L 是单位三角矩阵

    // 如果未定义 U_grad，则将 A_grad 与与 U2(U) 形状相同的零张量在最后一维拼接
    if (!U_grad.defined()) {
      A_grad = at::cat({A_grad, at::zeros_like(get_U2(U))}, /*dim=*/-1);
    }
    
    // 如果需要执行主元素置换（pivot），则用 P 矩阵左乘 A_grad
    if (pivot) {
      A_grad = P.matmul(A_grad);
    }
    return A_grad;
  } else {
    // Tall case
    // 对于高瘦情况，执行以下操作：
    // A1_grad = P [L1_grad + L^{-H} (U_grad U^H o 1_U - L^H L_grad o 1_L)] U^{-H}
    // A2_grad = P L2_grad U^{-H}

    // 定义获取 L1 和 L2 的函数，根据 m 和 k 对 L 进行裁剪
    const auto get_L1 = [m, k](const Tensor& L) {
      return m == k ? L : L.narrow_symint(-2, 0, k);
    };
    const auto get_L2 = [m, k](const Tensor& L) {
      return L.narrow_symint(-2, k, m - k);
    };

    // 初始化 A_grad，如果定义了 U_grad，则为 U_grad 乘以 U 的共轭转置
    auto A_grad = U_grad.defined() ? U_grad.matmul(U.mH()) : Tensor{};
    // 如果定义了 L_grad，则更新 A_grad 为 A_grad - L^H L_grad 的下三角部分
    if (L_grad.defined()) {
      A_grad = A_grad.defined() ? A_grad - L.mH().matmul(L_grad.tril(-1))
                                : -L.mH().matmul(L_grad.tril(-1));
    }
    // 解决线性方程组 A_grad = L1^H * A_grad，其中 L 是 L 的部分视图
    A_grad = at::linalg_solve_triangular(
        get_L1(L).mH(),                      // 左侧上三角矩阵 L1^H
        A_grad.triu(),                       // 右侧上三角部分 A_grad
        /*upper=*/true,                      // L1^H 是上三角矩阵
        /*left=*/true,                       // 在左侧乘法 A_grad
        /*unitriangular=*/true);             // L1 是单位三角矩阵

    // 如果定义了 L_grad，则更新 A_grad 为 [A_grad + L1_grad 下三角部分, L2_grad] 的拼接
    if (L_grad.defined()) {
      A_grad = at::cat(
          {A_grad + get_L1(L_grad).tril(-1), get_L2(L_grad)}, /*dim=*/-2);
    }

    // 解决线性方程组 A_grad = U^H * A_grad，其中 U 是 U 的部分视图
    A_grad = at::linalg_solve_triangular(
        U.mH(),                              // 左侧下三角矩阵 U^H
        A_grad,                              // 右侧 A_grad
        /*upper=*/false,                     // U^H 是下三角矩阵
        /*left=*/false);                     // 在右侧乘法 A_grad

    // 如果未定义 L_grad，则将 A_grad 与与 L2(L) 形状相同的零张量在倒数第二维拼接
    if (!L_grad.defined()) {
      A_grad = at::cat({A_grad, at::zeros_like(get_L2(L))}, /*dim=*/-2);
    }
    
    // 如果需要执行主元素置换（pivot），则用 P 矩阵左乘 A_grad
    if (pivot) {
      A_grad = P.matmul(A_grad);
    }
    return A_grad;
  }
// 定义函数 lu_factor_ex_backward，计算 LU 分解的反向传播梯度
Tensor lu_factor_ex_backward(
    const Tensor& grad,    // 输入梯度张量
    const Tensor& LU,      // LU 分解结果张量
    const Tensor& pivs,    // 主元位置张量
    const bool pivot) {    // 是否使用主元位置进行 LU 分解

  // 解压 LU 分解结果，获取 P、L、U 三个张量
  auto [P, L, U] =
      at::lu_unpack(LU, pivs, /*unpack_data=*/true, /*unpack_pivots=*/pivot);

  // L.shape == (..., m, k)
  // U.shape == (..., k, n)
  const auto m = LU.size(-2);  // 计算 LU 张量在倒数第二个维度的大小
  const auto n = LU.size(-1);  // 计算 LU 张量在最后一个维度的大小
  const auto k = std::min(m, n);  // 计算 m 和 n 的较小值作为 k
  const auto L_grad = grad.narrow(-1, 0, k);  // 对输入梯度 grad 在倒数第一个维度上进行裁剪
  const auto U_grad = grad.narrow(-2, 0, k);  // 对输入梯度 grad 在倒数第二个维度上进行裁剪
  // 调用 linalg_lu_backward 函数进行 LU 分解的反向传播计算
  return linalg_lu_backward(
      /*L_grad=*/L_grad,   // L 的梯度
      /*U_grad=*/U_grad,   // U 的梯度
      P,                   // 排列矩阵 P
      L,                   // 下三角矩阵 L
      U,                   // 上三角矩阵 U
      pivot);              // 是否使用主元位置进行 LU 分解
}

// 该函数基于 linalg_lu_backward 函数的正向 AD 推导
std::tuple<Tensor, Tensor> linalg_lu_jvp(
    const Tensor& dA,      // 输入张量 dA
    const Tensor& P,       // 排列矩阵 P
    const Tensor& L,       // 下三角矩阵 L
    const Tensor& U,       // 上三角矩阵 U
    const bool pivot) {    // 是否使用主元位置进行 LU 分解

  at::NoTF32Guard disable_tf32;  // 禁用 TF32 加速

  auto m = dA.size(-2);   // 获取输入张量 dA 在倒数第二个维度上的大小
  auto n = dA.size(-1);   // 获取输入张量 dA 在最后一个维度上的大小
  auto k = std::min(m, n);  // 计算 m 和 n 的较小值作为 k

  auto PdA = pivot ? P.transpose(-2, -1).matmul(dA) : dA;  // 根据 pivot 决定是否转置 P 后与 dA 相乘

  // 类似于反向传播的实现，我们也考虑块结构
  // 例如对于大小为 m x n 的矩阵 A，我们将其分解为 A = (A1 | A2)，其中 A1 大小为 m x m（如果 m <= n）
  // 或者 A = (A1^T | A2^T)^T，其中 A1 大小为 n x n（如果 m > n）
  auto PdA1 = PdA.narrow(-2, 0, k).narrow(-1, 0, k);  // 对 PdA 在倒数第二个维度和最后一个维度上进行裁剪
  auto L1 = L.narrow(-2, 0, k).narrow(-1, 0, k);       // 对 L 在倒数第二个维度和最后一个维度上进行裁剪
  auto U1 = U.narrow(-2, 0, k).narrow(-1, 0, k);       // 对 U 在倒数第二个维度和最后一个维度上进行裁剪

  // 我们通过两次三角解求得矩阵 dK，第二次是 in-place 操作
  auto dK = at::linalg_solve_triangular(
      L1, PdA1, /*upper=*/false, /*left=*/true, /*unitriangular=*/true);  // 求解 L1^{-1} PdA1

  // TODO 我们应该能够进行原地操作。目前会引发 RuntimeError：linalg_solve_triangular()：
  // out=... 参数的函数不支持自动微分，但其中一个参数需要梯度。

  // at::linalg_solve_triangular_out(dK, U1, dK, /*upper=*/true, /*left=*/false);
  dK = at::linalg_solve_triangular(U1, dK, /*upper=*/true, /*left=*/false);  // 求解 U1^{-1} dK

  auto dL1 = L1.matmul(dK.tril(-1));  // 计算 dL1
  auto dU1 = dK.triu().matmul(U1);    // 计算 dU1

  if (m == n) {
    return std::make_tuple(std::move(dL1), std::move(dU1));  // 如果 m 等于 n，则返回 dL1 和 dU1
  } else if (m < n) {
    // 我们只需更新 dU2，定义为：dU2 := L1^{-1} PdA2 - dK.tril(-1) U2
    const auto PdA2 = PdA.narrow(-1, k, n - k);  // 对 PdA 在最后一个维度上进行裁剪
    const auto U2 = U.narrow(-1, k, n - k);       // 对 U 在最后一个维度上进行裁剪
    auto dU2 =
        at::linalg_solve_triangular(
            L1, PdA2, /*upper=*/false, /*left=*/true, /*unitriangular=*/true) -  // 求解 L1^{-1} PdA2
        dK.tril(-1).matmul(U2);  // 计算 dU2
    return std::make_tuple(
        std::move(dL1), at::cat({std::move(dU1), std::move(dU2)}, /*dim=*/-1));  // 返回 dL1 和连接后的 dU1、dU2
  } else {
    // 我们只需更新 dL2，定义为：dL2 := PdA2 U^{-1} - L2 dK.triu()
    const auto PdA2 = PdA.narrow(-2, k, m - k);  // 对 PdA 在倒数第二个维度上进行裁剪
    const auto L2 = L.narrow(-2, k, m - k);       // 对 L 在倒数第二个维度上进行裁剪
    auto dL2 =
        at::linalg_solve_triangular(U1, PdA2, /*upper=*/true, /*left=*/false) -  // 求解 U1^{-1} PdA2
        L2.matmul(dK.triu());  // 计算 dL2
    return std::make_tuple(
        std::move(dL1), std::move(dL2));  // 返回 dL1 和 dL2
  }
}
    # 使用 std::make_tuple 创建一个元组，其中包含两个元素：
    # 1. 调用 at::cat 函数，将 std::move(dL1) 和 std::move(dL2) 合并为一个张量，沿着指定的维度 -2 进行拼接
    # 2. std::move(dU1)，将变量 dU1 移动到元组中的第二个位置
    return std::make_tuple(
        at::cat({std::move(dL1), std::move(dL2)}, /*dim=*/-2), std::move(dU1));
}

// 对称整数数组引用的参数推导函数，用于计算 LU 分解的 JVP（Jacobian Vector Product）
Tensor lu_factor_ex_jvp(
    const Tensor& dA,                   // 输入张量 dA
    const Tensor& LU,                   // LU 分解结果张量 LU
    const Tensor& pivs,                 // 基元排列张量 pivs
    const bool pivot) {                 // 是否进行枢轴选取的布尔值

  // 从 LU 分解中解包出 P、L、U 三个张量
  auto [P, L, U] =
      at::lu_unpack(LU, pivs, /*unpack_data=*/true, /*unpack_pivots=*/pivot);

  // 计算 LU 分解的 JVP
  auto [dL, dU] = linalg_lu_jvp(dA, P, L, U, pivot);

  auto m = dA.size(-2);                // dA 张量的倒数第二个维度大小
  auto n = dA.size(-1);                // dA 张量的最后一个维度大小

  // 根据 dA 张量的维度大小选择性地更新 dL 或 dU 张量
  if (m >= n) {
    dL.narrow(-2, 0, n).add_(dU);      // 在 dL 张量的特定维度上添加 dU 张量
    return dL;                         // 返回更新后的 dL 张量
  } else {
    dU.narrow(-1, 0, m).add_(dL);      // 在 dU 张量的特定维度上添加 dL 张量
    return dU;                         // 返回更新后的 dU 张量
  }
}

// 对数求和指数函数的 JVP（Jacobian Vector Product）
Tensor logsumexp_jvp(
    const Tensor& self_p,               // 第一个输入张量 self_p
    const Tensor& self_t,               // 第二个输入张量 self_t
    IntArrayRef dim,                    // 整数数组引用的维度
    bool keepdim) {                     // 是否保持维度的布尔值

  // 重新计算可以从前向传播中重复使用的一些值，以简化操作
  auto self_p_exp = [&self_p, &dim]() {
    if (self_p.sym_numel() > 0) {
      return (self_p - at::amax(self_p, dim, true))
          .exp();                       // 使用 exp-normalize 技巧
    } else {
      return self_p.exp();
    }
  }();

  auto sumexp_p = self_p_exp.sum(dim, keepdim);  // 对 self_p_exp 张量沿指定维度求和

  // 断言 self_t 张量不是零张量
  TORCH_INTERNAL_ASSERT(!self_t._is_zerotensor())

  // 根据输入张量的类型，计算 JVP 结果
  if (areAnyTensorSubclassLike({self_p, self_t})) {
    auto result = (self_p_exp * self_t).sum(dim, keepdim);  // 计算乘积并沿指定维度求和
    result /= sumexp_p;                 // 结果除以 sumexp_p
    return result;                      // 返回计算结果张量
  } else {
    self_p_exp *= self_t;               // 计算乘积
    auto sumexp_t = self_p_exp.sum(dim, keepdim);  // 对乘积张量沿指定维度求和
    return sumexp_t /= sumexp_p;        // 返回除以 sumexp_p 的结果张量
  }
}

// 在反向传播时发出警告的函数
Tensor warn_backwards(const Tensor& grad_output) {
  TORCH_WARN("Warn from backward");     // 发出警告信息
  return grad_output;                   // 返回输入的梯度张量
}

// cuDNN 不支持偏置梯度计算的卷积反向传播函数
std::tuple<Tensor, Tensor> _cudnn_convolution_backward(
    const at::Tensor& self,              // 输入张量 self
    const at::Tensor& grad_output,       // 梯度输出张量 grad_output
    const at::Tensor& weight,            // 权重张量 weight
    at::SymIntArrayRef padding,          // 对称整数数组引用的填充参数
    at::SymIntArrayRef output_padding,   // 对称整数数组引用的输出填充参数
    at::SymIntArrayRef stride,           // 对称整数数组引用的步长参数
    at::SymIntArrayRef dilation,         // 对称整数数组引用的扩张参数
    bool transposed,                     // 是否转置的布尔值
    c10::SymInt groups,                  // 对称整数的分组数
    ::std::array<bool, 2> output_mask) { // 布尔值数组的输出掩码

  if (!grad_output.defined()) {
    return std::tuple<Tensor, Tensor>();  // 如果梯度输出未定义，则返回空元组
  }

  // 调用通用的反向传播函数，忽略偏置梯度部分
  std::tuple<Tensor, Tensor, Tensor> grad_inputs =
      at::convolution_backward_symint(
          grad_output,
          self,
          weight,
          c10::nullopt,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          std::move(groups),
          {output_mask[0], output_mask[1], false});

  // 从完整的梯度输入元组中提取梯度输入和梯度权重张量
  std::tuple<Tensor, Tensor> result =
      std::make_tuple(std::get<0>(grad_inputs), std::get<1>(grad_inputs));

  return result;                        // 返回梯度输入和梯度权重的元组
}

// 散射减少的 JVP（Jacobian Vector Product）
Tensor scatter_reduce_jvp(
    const Tensor& self_p,                // 输入张量 self_p
    const Tensor& self_t,                      // 引用类型，表示输入张量 self_t
    int dim,                                   // 整数类型，表示维度 dim
    const Tensor& index,                       // 引用类型，表示索引张量 index
    const Tensor& src_p,                       // 引用类型，表示源张量 src_p
    const Tensor& src_t,                       // 引用类型，表示源张量 src_t
    c10::string_view reduce,                   // 字符串视图，表示减少操作类型 reduce
    bool include_self,                         // 布尔类型，表示是否包含自身 include_self
    const Tensor& result) {                    // 引用类型，表示结果张量 result

  if (reduce == "sum" || reduce == "mean") {    // 如果减少操作为 "sum" 或 "mean"
    // 函数是线性的，调用 scatter_reduce 函数进行操作
    return at::scatter_reduce(self_t, dim, index, src_t, reduce, include_self);

  } else if (reduce == "amin" || reduce == "amax") {  // 如果减少操作为 "amin" 或 "amax"
    // 使用 gather 函数根据索引从结果张量中收集数据
    auto gather_result = at::gather(result, dim, index);
    // 创建布尔掩码，用于指示 self_p 是否等于 result
    auto mask_self = self_p == result;
    // 创建布尔掩码，用于指示 src_p 是否等于 gather_result
    auto mask_src = src_p == gather_result;
    // 根据掩码选择性地应用 src_t，创建 masked_src_t
    auto masked_src_t = at::where(mask_src, src_t, 0.);
    // 计算分母，使用 scatter_reduce 函数进行操作
    auto div =
        mask_self.to(self_t.dtype())
            .scatter_reduce(
                dim, index, mask_src.to(self_t.dtype()), "sum", include_self);
    // 计算分子，使用 scatter_reduce 函数进行操作
    return at::where(mask_self, self_t, 0.)
        .scatter_reduce(dim, index, masked_src_t, "sum", include_self)
        .div(div);

  } else {
    // 如果未实现指定的减少操作类型
    // 返回空张量
    return Tensor{};
  }
}
  // FIXME: 复杂梯度暂未正确处理
  // 目前在 tools/autograd/gen_variable_type.py 中 scatter_reduce 尚未添加到白名单，因此目前不需要处理复杂梯度

  // 如果梯度未定义，则返回空的 grad_self 和 grad_src
  if (!grad.defined()) {
    return std::make_tuple(grad_self, grad_src);
  }

  // 根据 reduce 参数选择不同的处理方式
  if (reduce == "sum") {
    // 对于 sum 操作，grad_self 直接等于 grad，grad_src 为 grad 在指定维度和索引上的聚合
    grad_self = grad;
    grad_src = grad.gather(dim, index);
  } else if (reduce == "prod") {
    // 对于 prod 操作，计算 self 和 src 中为 0 的元素的 exclusive prod
    Tensor masked_self = self.masked_fill(self == 0, 1);
    Tensor masked_self_result =
        masked_self.scatter_reduce(dim, index, src, reduce, include_self);
    grad_self = grad * masked_self_result / masked_self;

    // 计算 src 中为 0 的情况
    Tensor src_zero = src == 0;
    Tensor src_num_zeros =
        zeros_like(self)
            .scatter_add(dim, index, src_zero.to(self.dtype()))
            .gather(dim, index);
    Tensor src_single_zero = bitwise_and(src_zero, src_num_zeros == 1);

    // 处理 src_single_zero 情况下的梯度计算
    Tensor masked_src = src.masked_fill(src_single_zero, 1);
    Tensor masked_src_result =
        self.scatter_reduce(dim, index, masked_src, reduce, include_self);
    Tensor grad_src1 = where(
        src_single_zero,
        (grad * masked_src_result).gather(dim, index),
        (grad * result).gather(dim, index) / src.masked_fill(src_zero, 1));

    // 如果梯度模式开启且 src 中存在多个零值，则抛出错误
    if (GradMode::is_enabled() && (src_num_zeros > 1).any().item<bool>()) {
      auto node = std::make_shared<DelayedError>(
          "scatter_reduce(): Double backward is unsupported for src when >1 zeros in src are scattered to the same position in self",
          /* num inputs */ 1);
      auto result = node->apply({std::move(grad_src1)});
      grad_src = result[0];
    } else {
      grad_src = grad_src1;
    }
  } else if (reduce == "mean") {
    // 对于 mean 操作，计算 N，并进行相应的梯度计算
    Tensor N = include_self ? ones_like(grad) : zeros_like(grad);
    N = N.scatter_add(dim, index, ones_like(src));
    N.masked_fill_(N == 0, 1);
    grad_self = grad / N;
    Tensor N_src = N.gather(dim, index);
    grad_src = grad.gather(dim, index) / N_src;
  } else if (reduce == "amax" || reduce == "amin") {
    // 对于 amax 或 amin 操作，均匀分配梯度
    Tensor value = result.gather(dim, index);
    Tensor self_is_result = (self == result).to(self.scalar_type());
    Tensor src_is_result = (src == value).to(self.scalar_type());
    Tensor N_to_distribute =
        self_is_result.scatter_add(dim, index, src_is_result);
    // 计算每个节点的梯度相对于分布数的平均值
    Tensor grad_distributed = grad / N_to_distribute;
    
    // 如果节点等于结果，则将分布梯度赋给 grad_self
    grad_self = (self == result) * grad_distributed;
    
    // 如果节点等于值，则将按索引收集的分布梯度赋给 grad_src
    grad_src = (src == value) * grad_distributed.gather(dim, index);
  } else {
    // 如果 reduce 参数不在预期的范围内，则抛出错误信息
    AT_ERROR(
        "Expected 'reduce' to be one of 'sum', 'prod', 'mean', 'amax', 'amin' but got ",
        reduce,
        ".");
  }
  
  // 如果不包括自身节点，则将 grad_self 在指定维度和索引处的散射值设为 0
  if (!include_self) {
    grad_self = grad_self.scatter(dim, index, 0);
  }
  
  // 返回包含 grad_self 和 grad_src 的元组
  return std::make_tuple(grad_self, grad_src);
}

Tensor _to_copy_backward(
    const Tensor& grad_,
    const c10::TensorOptions& self_options) {
  // 处理从复数到实数的复制操作，避免出现警告
  const auto self_type = self_options.dtype().toScalarType();
  // 使用 borrow 方式获取梯度张量的引用
  auto grad = c10::MaybeOwned<at::Tensor>::borrowed(grad_);
  // 如果目标数据类型不是复数类型，但梯度张量是复数类型，则将其实部作为新的梯度张量
  if (!c10::isComplexType(self_type) && grad->is_complex()) {
    grad = c10::MaybeOwned<at::Tensor>::owned(at::real(grad_));
  }

  // 将梯度张量转换为指定的选项和数据类型，并且不使用非阻塞方式复制
  return grad->to(self_options, /*non_blocking=*/false, /*copy=*/false);
}

std::tuple<Tensor, Tensor> index_reduce_backward(
    const Tensor& grad,
    const Tensor& self,
    int dim,
    const Tensor& index,
    const Tensor& source,
    c10::string_view reduce,
    bool include_self,
    const Tensor& result) {
  Tensor grad_self, grad_src;

  // FIXME: index_add 的反向传播公式对于 source.dim == 0 的特殊情况有一个问题
  // 可能会抛出 "IndexError: dimension specified as 0 but tensor has no dimensions" 错误
  // 需要进一步确认这种情况是否会出现，并在此处进行处理

  // 如果梯度张量未定义，则直接返回空的梯度张量对
  if (!grad.defined()) {
    return std::make_tuple(grad_self, grad_src);
  }

  // 如果采用的是 "prod" 归约方式
  if (reduce == "prod") {
    // 使用 1 替换 self 张量中为 0 的元素，以避免除以 0 的情况
    Tensor masked_self = self.masked_fill(self == 0, 1);
    // 执行索引归约操作，得到 masked_self 在指定维度上的归约结果
    Tensor masked_self_result =
        masked_self.index_reduce(dim, index, source, reduce, include_self);
    // 计算 self 张量的梯度
    grad_self = grad * masked_self_result / masked_self;
    // 在 source 张量为 0 的位置创建一个全零张量
    Tensor src_zero = source == 0;
    // 计算 self 张量中每个索引位置上零值的数量，并更新到相应位置
    Tensor src_num_zeros = zeros_like(self)
                               .index_add(dim, index, src_zero.to(self.dtype()))
                               .index_select(dim, index);
    // 在源张量中只有一个零值的位置标记为真
    Tensor src_single_zero = bitwise_and(src_zero, src_num_zeros == 1);
    // 对于具有单个零值的源位置，避免在梯度传播时将零传播出去
    Tensor masked_src = source.masked_fill(src_single_zero, 1);
    // 执行索引归约操作，得到 self 张量在指定维度上的 masked_src 归约结果
    Tensor masked_src_result =
        self.index_reduce(dim, index, masked_src, reduce, include_self);
    // 根据 src_single_zero 的位置条件不同，计算 grad_src1 的值
    Tensor grad_src1 = where(
        src_single_zero,
        (grad * masked_src_result).index_select(dim, index),
        (grad * result).index_select(dim, index) /
            source.masked_fill(src_zero, 1));
    // 如果梯度模式已启用，并且源张量中有多个零值的位置，抛出错误
    if (GradMode::is_enabled() && (src_num_zeros > 1).any().item<bool>()) {
      auto node = std::make_shared<DelayedError>(
          "index_reduce(): Double backward is unsupported for source when >1 zeros in source are scattered to the same position in self",
          /* num inputs */ 1);
      auto result = node->apply({std::move(grad_src1)});
      grad_src = result[0];
    } else {
      grad_src = grad_src1;
    }
  }
  // 如果采用的是 "mean" 归约方式
  else if (reduce == "mean") {
    // 如果 include_self 为真，则创建一个与梯度张量相同大小的全一张量 N，否则创建全零张量
    Tensor N = include_self ? ones_like(grad) : zeros_like(grad);
    // 在指定维度上，将 N 中对应位置加 1，并且将值为 0 的位置替换为 1
    N = N.index_add(dim, index, ones_like(source));
    N.masked_fill_(N == 0, 1);
    // 计算 self 张量的梯度
    grad_self = grad / N;
    // 如果 reduce 参数为 "prod"，计算 N_src 并计算梯度 grad_src
    Tensor N_src = N.index_select(dim, index);
    grad_src = grad.index_select(dim, index) / N_src;
  } else if (reduce == "amax" || reduce == "amin") {
    // 如果 reduce 参数为 "amax" 或 "amin"，处理最大或最小值的情况
    // 获取结果中指定维度和索引位置的值
    Tensor value = result.index_select(dim, index);
    // 判断 self 和 result 是否相等，并转换为当前数据类型的张量
    Tensor self_is_result = (self == result).to(self.scalar_type());
    // 判断 source 和 value 是否相等，并转换为当前数据类型的张量
    Tensor source_is_result = (source == value).to(self.scalar_type());
    // 计算需要分布的数量 N_to_distribute
    Tensor N_to_distribute =
        self_is_result.index_add(dim, index, source_is_result);
    // 计算分布后的梯度 grad_distributed
    Tensor grad_distributed = grad / N_to_distribute;
    // 计算 self 的梯度 grad_self
    grad_self = self_is_result * grad_distributed;
    // 计算 source 的梯度 grad_src
    grad_src = source_is_result * grad_distributed.index_select(dim, index);
  } else {
    // 如果 reduce 参数不是预期的 "prod", "amax", "amin", "mean" 中的一种，抛出错误
    AT_ERROR(
        "Expected 'reduce' to be one of 'prod', 'amax', 'amin' or 'mean' but got ",
        reduce,
        ".");
  }

  // 如果不包含 self 的梯度，将其对应位置的梯度 grad_self 置为 0
  if (!include_self) {
    grad_self = grad_self.index_fill(dim, index, 0);
  }

  // 返回包含 grad_self 和 grad_src 的元组
  return std::make_tuple(grad_self, grad_src);
}



Tensor take_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& indices) {
  // 创建一个和 self 张量相同形状的 grad_self 张量，并填充零值
  Tensor grad_self = at::zeros_like(self);
  // 如果 grad 和 indices 中有任何一个是张量子类的 CCT，则使用 put 的非就地变体
  if (areAnyTensorSubclassLike({grad, indices})) {
    // 使用非就地方式将 grad 中的数据按照 indices 的索引写入 grad_self
    return grad_self.put(indices, grad, true);
  }
  // 否则使用就地方式将 grad 中的数据按照 indices 的索引写入 grad_self
  return grad_self.put_(indices, grad, true);
}

Tensor to_sparse_backward(
    const Tensor& grad,
    const c10::Layout self_layout,
    const c10::OptionalArrayRef<c10::SymInt>& self_blocksize) {
  // 对于 self 布局为 kStrided 的路径
  if (self_layout == c10::kStrided) {
    // 将 grad 张量转换为稠密形式（to_dense）
    return grad.to_dense();
  } else {
    // 否则，根据给定的 self_layout 和 self_blocksize 将 grad 转换为稀疏形式
    OptionalIntArrayRef blocksize = c10::nullopt;
    if (self_blocksize.has_value()) {
      blocksize = c10::asIntArrayRefSlowOpt(*self_blocksize);
    }
    return grad.to_sparse(self_layout, blocksize);
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
mkldnn_rnn_layer_differentiable_backward(
    const Tensor& input,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& hx_,
    const Tensor& cx_tmp,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace) {
  const Tensor& grad_output_r =
      c10::value_or_else(grad_output_r_opt, [] { return Tensor(); });
  const Tensor& grad_hy_r =
      c10::value_or_else(grad_hy_r_opt, [] { return Tensor(); });
  const Tensor& grad_cy_r =
      c10::value_or_else(grad_cy_r_opt, [] { return Tensor(); });
  // 如果 grad_output_r、grad_hy_r 和 grad_cy_r 都未定义，则返回空张量元组
  if (!grad_output_r.defined() && !grad_hy_r.defined() &&
      !grad_cy_r.defined()) {
    return std::make_tuple(
        Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
  }
  // 如果 grad_output_r 已定义，则使用其内容创建连续的 grad_output 张量；否则使用 output 形状创建零填充的张量
  auto grad_output = grad_output_r.defined()
      ? grad_output_r.contiguous()
      : at::zeros_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 如果 grad_hy_r 已定义，则使用其内容创建连续的 grad_hy 张量；否则使用 hx_ 形状创建零填充的张量
  auto grad_hy = grad_hy_r.defined()
      ? grad_hy_r.contiguous()
      : at::zeros_like(hx_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 如果 cx_tmp 已定义，则使用其内容创建连续的 grad_cy 张量；否则使用 grad_cy_r 的内容创建连续的张量
  auto grad_cy = cx_tmp.defined()
      ? (grad_cy_r.defined()
             ? grad_cy_r.contiguous()
             : at::zeros_like(cx_tmp, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
      : grad_cy_r.contiguous();
  // 初始化 bias_ih 和 bias_hh 张量
  Tensor bias_ih, bias_hh;
  // 如果有 biases，则使用 weight2 和 weight3 初始化 bias_ih 和 bias_hh
  if (has_biases) {
    bias_ih = weight2;
    bias_hh = weight3;
  } else {
    // 否则，使用 weight0 的选项创建全零的 biases 张量
    bias_ih = at::zeros(
        {4 /* LSTM 的 bias 门数量 */ * hidden_size}, weight0.options());
  // 初始化 bias_hh，用来存储 LSTM 的偏置向量，大小为 4 * hidden_size
  bias_hh = at::zeros({4 /* num_bias_gates of LSTM */ * hidden_size}, weight0.options());

const auto& input_ = input;
auto hx_prev = hx_;
auto cx_prev = cx_tmp;

// 在一层中重新计算门和隐藏状态，将用于反向传播
int64_t seq_length = input.size(0);
std::vector<std::tuple<Tensor, Tensor, Tensor, Tensor>> layer_gates(seq_length);
std::vector<std::tuple<Tensor, Tensor>> layer_states(seq_length + 1);
layer_states[0] = std::make_tuple(hx_, cx_tmp);

// 遍历序列中的每一个时间步
for (int64_t seq = 1; seq < seq_length + 1; seq++) {
  auto hx = hx_prev;
  auto cx = cx_prev;
  auto x_index = reverse ? seq_length - seq : seq - 1;
  
  // 计算输入的线性变换和隐藏状态的线性变换得到的门
  auto gate = at::linear(input_[x_index], weight0, bias_ih)
                  .add_(at::linear(hx, weight1, bias_hh));
  
  // 将门分割成四个部分：输入门(i), 遗忘门(f), 单元更新(g), 输出门(o)
  auto chunked_gates = gate.unsafe_chunk(4, 1);
  auto i = chunked_gates[0].sigmoid_();
  auto f = chunked_gates[1].sigmoid_();
  auto g = chunked_gates[2].tanh_();
  auto o = chunked_gates[3].sigmoid_();
  
  // 保存每个时间步的门信息
  layer_gates[x_index] = std::make_tuple(i, f, g, o);
  
  // 计算细胞状态和隐藏状态
  auto cy = (f * cx).add(i * g);
  auto hy = o * cy.tanh();
  
  // 保存每个时间步的状态信息
  layer_states[seq] = std::make_tuple(hy, cy);
  
  // 更新前一个隐藏状态和细胞状态
  hx_prev = hy;
  cx_prev = cy;
}

Tensor dx, dWx, dWh, db, db_, dprev_h, dprev_c, dWh_, dWx_;
Tensor new_grad_hy, d1, dgp, dip, dfp, dop, do_, dg, df, di, da;
std::vector<at::Tensor> layer_dx(seq_length);

// 反向遍历序列中的每一个时间步
for (int64_t seq = seq_length - 1; seq >= 0; seq--) {
  int64_t x_index = reverse ? seq_length - seq - 1 : seq;
  
  // 从保存的门信息和状态信息中获取数据
  auto i = std::get<0>(layer_gates[x_index]);
  auto f = std::get<1>(layer_gates[x_index]);
  auto g = std::get<2>(layer_gates[x_index]);
  auto o = std::get<3>(layer_gates[x_index]);
  auto hy = std::get<0>(layer_states[seq + 1]);
  auto cy = std::get<1>(layer_states[seq + 1]);
  auto hx = std::get<0>(layer_states[seq]);
  auto cx = std::get<1>(layer_states[seq]);
  
  // 计算新的隐藏状态梯度
  new_grad_hy = grad_output[x_index].add(grad_hy);
  
  // 计算当前时间步的反向传播梯度
  d1 = grad_cy.add(new_grad_hy * o * (1 - cy.tanh() * cy.tanh()));
  dgp = d1 * i;
  dip = d1 * g;
  dprev_c = d1 * f;
  dfp = d1 * cx;
  dop = new_grad_hy * cy.tanh();
  do_ = dop * o * (1 - o);
  dg = dgp * (1 - g * g);
  df = dfp * f * (1 - f);
  di = dip * i * (1 - i);
  da = at::cat({di, df, dg, do_}, 1);
  db_ = at::sum(da, 0);
  dx = at::matmul(da, weight0);
  dx = at::unsqueeze(dx, 0);
  dprev_h = at::matmul(da, weight1);
  dWx_ = at::matmul(da.transpose(0, 1), input_[x_index]);
  dWh_ = at::matmul(da.transpose(0, 1), hx);
  
  // 累加梯度，如果是第一个时间步，初始化梯度变量
  if (seq == seq_length - 1) {
    db = db_;
    dWx = dWx_;
    dWh = dWh_;
  } else {
    db += db_;
    dWx += dWx_;
    dWh += dWh_;
  }
  
  // 保存当前时间步的输入梯度
  layer_dx[x_index] = dx;
  
  // 更新前一个隐藏状态和细胞状态的梯度
  grad_hy = dprev_h;
  grad_cy = dprev_c;
}

// 拼接所有时间步的输入梯度，并返回最终结果
auto cat_layer_dx = at::cat(layer_dx, 0);
return std::make_tuple(cat_layer_dx, dWx, dWh, db, db, dprev_h, dprev_c);
} // 关闭 autograd 命名空间
} // 关闭 torch 命名空间

// 关闭 generated 命名空间
} // 关闭 details 命名空间

// 函数实现结束，返回与梯度相关的自变量的值
Tensor values_backward(const Tensor& grad, const Tensor& self) {
    // 定义用于存储梯度的自变量的张量
    Tensor grad_self;
    
    // 检查梯度是否已定义
    if (grad.defined()) {
        // 如果自变量的布局为稀疏
        if (self.layout() == c10::kSparse) {
            // 返回一个稀疏的 COO 张量，使用不安全的符号整数运算
            return at::_sparse_coo_tensor_unsafe_symint(
                self.indices(),
                grad,
                self.sym_sizes(),
                self.options(),
                /*is_coalesced=*/true);
        } else if (at::sparse_csr::is_sparse_compressed(self)) {
            // 如果自变量为稀疏 CSR 压缩格式
            auto [compressed_indices, plain_indices] =
                at::sparse_csr::getCompressedPlainIndices(self);
            // 返回一个压缩稀疏张量，使用不安全的符号整数运算
            return at::_sparse_compressed_tensor_unsafe_symint(
                compressed_indices,
                plain_indices,
                grad,
                self.sym_sizes(),
                self.options());
        } else {
            // 若不支持当前布局类型，抛出错误
            TORCH_CHECK_NOT_IMPLEMENTED(
                false,
                "values backward with respect to self with layout ",
                self.layout());
        }
    }
    
    // 返回自变量的梯度值
    return grad_self;
}
```