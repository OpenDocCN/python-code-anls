# `.\pytorch\aten\src\ATen\native\LossMultiLabelMargin.cpp`

```
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/multilabel_margin_loss_backward_native.h>
#include <ATen/ops/multilabel_margin_loss_forward.h>
#include <ATen/ops/multilabel_margin_loss_forward_native.h>
#include <ATen/ops/multilabel_margin_loss_native.h>
#include <ATen/ops/zeros_like.h>
#endif


// 根据是否定义了 AT_PER_OPERATOR_HEADERS 宏，选择性包含不同的 ATen 操作头文件

namespace at::native {

namespace {

template <typename scalar_t>
inline scalar_t multilabel_margin_loss_forward_inner_sum_cpu(
    const scalar_t* input_data,
    const int64_t* target_data,
    scalar_t* is_target_data,
    int64_t dim) {
  using accscalar_t = at::acc_type<scalar_t, false>;
  accscalar_t sum = 0;

  // 计算每个样本的损失
  for (const auto ddt : c10::irange(dim)) {
    int64_t target_idx = target_data[ddt];
    if (target_idx < 0) {
      break;
    }
    is_target_data[target_idx] = 1; // 标记目标索引
  }

  // 计算损失值
  for (const auto dt : c10::irange(dim)) {
    int64_t target_idx = target_data[dt];
    if (target_idx < 0) {
      break;
    }

    scalar_t input_target = input_data[target_idx];
    for (const auto d : c10::irange(dim)) {
      if (!is_target_data[d]) {
        scalar_t z = 1 - input_target + input_data[d];
        if (z > 0) {
          sum += z; // 累加损失值
        }
      }
    }
  }

  return sum; // 返回总损失值
}

template <typename scalar_t>
static void multilabel_margin_loss_forward_out_frame(
    const Tensor& input_contiguous,
    const Tensor& target_contiguous,
    Tensor& output,
    Tensor& is_target,
    int64_t reduction,
    int64_t nframe,
    int64_t dim) {
  using accscalar_t = at::acc_type<scalar_t, false>;
  const scalar_t* input_data = input_contiguous.const_data_ptr<scalar_t>();
  const int64_t* target_data = target_contiguous.const_data_ptr<int64_t>();
  scalar_t* is_target_data = is_target.data_ptr<scalar_t>();

  if (reduction != Reduction::None || output.dim() == 0) {
    scalar_t* output_data = output.data_ptr<scalar_t>();

    accscalar_t sum = 0;

    // 计算总损失值
    for (C10_UNUSED const auto t : c10::irange(nframe)) {
      sum += multilabel_margin_loss_forward_inner_sum_cpu(
          input_data, target_data, is_target_data, dim);

      input_data += dim;
      target_data += dim;
      is_target_data += dim;
    }

    sum /= dim; // 求平均损失值
    if (reduction == Reduction::Mean) {
      sum /= nframe; // 如果是均值模式，则再除以样本数
    }

    *output_data = sum; // 将计算得到的损失值写入输出张量
  } else {
    auto output_acc = output.accessor<scalar_t, 1>();

    // 计算每个样本的损失值
    for (const auto t : c10::irange(nframe)) {
      scalar_t sum = multilabel_margin_loss_forward_inner_sum_cpu(
          input_data, target_data, is_target_data, dim);

      sum /= dim; // 求平均损失值
      output_acc[t] = sum; // 将损失值写入输出张量的对应位置

      input_data += dim;
      target_data += dim;
      is_target_data += dim;
    }
  }
}
static void multilabel_margin_loss_forward_out_cpu_template(
    const Tensor& input,
    const Tensor& target,
    Tensor& output,
    Tensor& is_target,
    int64_t reduction) {
  auto target_arg = TensorArg(target, "target", 2);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t nframe, dim;
  const int64_t ndims = input.dim();
  // 调用形状检查函数，获取输入数据的帧数和维度
  multilabel_margin_loss_shape_check(nframe, dim, ndims, input, target);

  // 如果需要进行汇总(reduction != Reduction::None)，或者目标张量维度小于等于1，则输出为标量
  // 即使 reduction == Reduction::None 的情况下
  if (reduction != Reduction::None || target.dim() <= 1) {
    output.resize_({});  // 调整输出张量为标量
  } else {
    output.resize_({nframe});  // 调整输出张量为 nframe 长度的向量
  }

  // 调整 is_target 张量的大小与 target 张量相同
  is_target.resize_as_(target);
  // 检查 is_target 是否是连续的张量
  TORCH_CHECK(is_target.is_contiguous(), "is_target must be contiguous");
  // 将 is_target 张量的值都设为 0
  is_target.zero_();

  // 如果输入张量的元素个数为 0，则直接返回
  if (input.numel() == 0) {
    return;
  }

  // 检查目标张量的值范围是否合法
  TORCH_CHECK(
      target.min().item<int64_t>() >= -1, target_arg, " is out of range");
  TORCH_CHECK(
      target.max().item<int64_t>() < dim, target_arg, " is out of range");

  // 将输入张量和目标张量调整为连续的张量
  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  // 根据输入张量的数据类型分发到具体的处理函数
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multilabel_margin_loss_forward_out_frame", [&] {
        // 调用具体的前向计算函数
        multilabel_margin_loss_forward_out_frame<scalar_t>(
            input_contiguous, target_contiguous, output, is_target, reduction, nframe, dim);
      });
}

template <typename scalar_t>
static void multilabel_margin_loss_backward_out_frame(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input_contiguous,
    const Tensor& target_contiguous,
    int64_t reduction,
    const Tensor& is_target_contiguous,
    int64_t nframe,
    int64_t dim) {
  auto is_target_arg = TensorArg(is_target_contiguous, "is_target", 5);

  // 检查 is_target 张量的值范围是否合法
  TORCH_CHECK(
      is_target_contiguous.min().item<scalar_t>() >= 0, is_target_arg, " is out of range");
  TORCH_CHECK(
      is_target_contiguous.max().item<scalar_t>() <= 1, is_target_arg, " is out of range");

  // 获取输入张量、目标张量和 is_target 张量的数据指针
  const scalar_t* input_data = input_contiguous.const_data_ptr<scalar_t>();
  const int64_t* target_data = target_contiguous.const_data_ptr<int64_t>();
  const scalar_t* is_target_data = is_target_contiguous.const_data_ptr<scalar_t>();
  // 计算缩放因子 g，用于计算梯度
  scalar_t g = static_cast<scalar_t>(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      reduction == Reduction::Mean ? 1. / (nframe * dim) : 1. / dim);

  // 获取梯度输入张量的可变数据指针
  scalar_t* grad_input_row_data = grad_input.mutable_data_ptr<scalar_t>();
  // 循环遍历每一帧数据的索引范围
  for (C10_UNUSED const auto t : c10::irange(nframe)) {
    // 遍历每个维度 dt
    for (const auto dt : c10::irange(dim)) {
      // 获取目标索引 target_idx
      int64_t target_idx = target_data[dt];
      // 如果目标索引小于 0，则跳出循环
      if (target_idx < 0) {
        break;
      }

      // 获取输入目标值 input_target
      scalar_t input_target = input_data[target_idx];
      // 再次遍历每个维度 d
      for (const auto d : c10::irange(dim)) {
        // 如果不是目标数据的维度
        if (!is_target_data[d]) {
          // 计算 z 值
          scalar_t z = 1 - input_target + input_data[d];
          // 如果 z 大于 0
          if (z > 0) {
            // 更新梯度数据
            grad_input_row_data[target_idx] -= g;
            grad_input_row_data[d] += g;
          }
        }
      }
    }
    // 更新数据指针位置
    input_data += dim;
    target_data += dim;
    is_target_data += dim;
    grad_input_row_data += dim;
  }

  // 获取可变梯度输入数据指针
  scalar_t* grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  // 如果不是无缩减模式或者梯度输出维度为 0
  if (reduction != Reduction::None || grad_output.dim() == 0) {
    // 断言条件：不是无缩减模式，或者梯度输出维度大于 0，或者 nframe 等于 1
    assert(
        reduction != Reduction::None || grad_output.dim() > 0 || nframe == 1);
    // 获取标量 d
    const auto d = *grad_output.const_data_ptr<scalar_t>();
    // 遍历所有的 t，更新 grad_input_data[t] *= d
    for (int64_t t = 0; t < nframe * dim; t++) {
      grad_input_data[t] *= d;
    }
  } else {
    // 检查梯度输出的维度是否为 1
    check_dim_size(grad_output, 1, 0, nframe);
    // 访问器：获取梯度输出的累加器
    auto grad_output_acc = grad_output.accessor<const scalar_t, 1>();
    // 再次遍历所有的 t 和 d，更新 grad_input_data[t * dim + d] *= grad_output_acc[t]
    for (const auto t : c10::irange(nframe)) {
      for (const auto d : c10::irange(dim)) {
        grad_input_data[t * dim + d] *= grad_output_acc[t];
      }
    }
  }
}

// 定义静态函数 multilabel_margin_loss_backward_out_cpu_template，用于计算多标签边际损失函数的梯度
static void multilabel_margin_loss_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t nframe, dim;
  CheckedFrom c = "multilabel_margin_loss_backward_cpu_template";
  auto target_arg = TensorArg(target, "target", 3);
  auto is_target_arg = TensorArg(is_target, "is_target", 5);
  const int64_t ndims = input.dim();

  // 调用 multilabel_margin_loss_shape_check 函数检查输入张量的形状
  multilabel_margin_loss_shape_check(nframe, dim, ndims, input, target);
  // 检查 target 和 is_target 张量是否具有相同的大小
  checkSameSize(c, target_arg, is_target_arg);

  // 将 grad_input 张量调整为与 input 相同的大小
  grad_input.resize_as_(input);
  // 如果 grad_input 张量为空，直接返回
  if (grad_input.numel() == 0) {
    return;
  }

  // 检查 grad_input 是否是连续的
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  // 将 grad_input 张量置零
  grad_input.zero_();

  // 检查 target 张量的最小值和最大值是否在有效范围内
  TORCH_CHECK(
      target.min().item<int64_t>() >= -1, target_arg, " is out of range");
  TORCH_CHECK(
      target.max().item<int64_t>() < dim, target_arg, " is out of range");

  // 获取 input、target 和 is_target 张量的连续版本
  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto is_target_contiguous = is_target.contiguous();

  // 使用 AT_DISPATCH_FLOATING_TYPES 宏，根据 input 的数据类型选择合适的函数实现
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multilabel_margin_loss_backward_out_frame", [&] {
        // 调用 multilabel_margin_loss_backward_out_frame 函数，计算梯度
        multilabel_margin_loss_backward_out_frame<scalar_t>(
            grad_input,
            grad_output,
            input_contiguous,
            target_contiguous,
            reduction,
            is_target_contiguous,
            nframe,
            dim);
      });
}

// 结束 multilabel_margin_loss_backward_out_cpu_template 函数定义

} // namespace

// 定义 multilabel_margin_loss_forward_out_cpu 函数，用于计算多标签边际损失函数的前向传播
std::tuple<Tensor&, Tensor&> multilabel_margin_loss_forward_out_cpu(const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  // 调用 multilabel_margin_loss_forward_out_cpu_template 函数进行前向传播计算
  multilabel_margin_loss_forward_out_cpu_template(
      self, target, output, is_target, reduction);
  // 返回计算结果的元组 output 和 is_target
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

// 定义 multilabel_margin_loss_forward_cpu 函数，用于计算多标签边际损失函数的前向传播
std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_cpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  // 创建空的输出张量 output 和 is_target
  auto output = at::empty({0}, self.options());
  auto is_target = at::empty({0}, self.options());
  // 调用 multilabel_margin_loss_forward_out_cpu 函数计算前向传播结果
  at::native::multilabel_margin_loss_forward_out_cpu(
      self, target, reduction, output, is_target);
  // 返回计算结果的元组 output 和 is_target
  return std::make_tuple(output, is_target);
}

// 定义 multilabel_margin_loss_backward_cpu_out 函数，用于计算多标签边际损失函数的反向传播
Tensor& multilabel_margin_loss_backward_cpu_out(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  // 调用 multilabel_margin_loss_backward_out_cpu_template 函数进行反向传播计算
  multilabel_margin_loss_backward_out_cpu_template(
      grad_input, grad_output, self, target, reduction, is_target);
  // 返回计算得到的 grad_input 张量
  return grad_input;
}

// 定义 multilabel_margin_loss_backward_cpu 函数，用于计算多标签边际损失函数的反向传播
Tensor multilabel_margin_loss_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  // 创建一个与输入张量 self 具有相同形状和数据类型的全零张量，使用传统的连续内存格式
  auto grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用 C++ 后端函数 multilabel_margin_loss_backward_cpu_out 处理多标签边界损失的反向传播
  // 将输出梯度 grad_output、输入张量 self、目标张量 target、减少方式 reduction、目标标记 is_target、梯度输入 grad_input 作为参数传递
  at::native::multilabel_margin_loss_backward_cpu_out(
      grad_output, self, target, reduction, is_target, grad_input);
  // 返回计算得到的梯度输入 grad_input
  return grad_input;
}

// 结束 at::native 命名空间

Tensor & multilabel_margin_loss_out(const Tensor & self, const Tensor & target, int64_t reduction, Tensor & output) {
  // 创建一个空的 Tensor is_target，使用 self 的选项
  Tensor is_target = at::empty({0}, self.options());
  // 调用 multilabel_margin_loss_forward_out 函数，将输出存储在 output 中，返回第一个元组元素
  return std::get<0>(at::multilabel_margin_loss_forward_out(output, is_target, self, target, reduction));
}

Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  // 调用 multilabel_margin_loss_forward 函数，返回其第一个元组元素
  return std::get<0>(at::multilabel_margin_loss_forward(self, target, reduction));
}

} // namespace at::native
```