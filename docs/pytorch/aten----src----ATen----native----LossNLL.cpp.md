# `.\pytorch\aten\src\ATen\native\LossNLL.cpp`

```py
// 定义宏以仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含相关的头文件以支持张量操作
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/Resize.h>
#include <c10/util/SmallBuffer.h>
#include <ATen/TensorSubclassLikeUtils.h>

// 根据宏定义条件包含不同的运算符头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cross_entropy_loss_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/nll_loss.h>
#include <ATen/ops/nll_loss2d.h>
#include <ATen/ops/nll_loss_backward_native.h>
#include <ATen/ops/nll_loss_forward.h>
#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/nll_loss_native.h>
#include <ATen/ops/nll_loss_nd.h>
#include <ATen/ops/nll_loss_nd_native.h>
#endif

#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>

#include <utility>

// 使用 AT 命名空间中的 meta 子命名空间
namespace at::meta {

// 定义 nll_loss_forward 的元函数
TORCH_META_FUNC(nll_loss_forward)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index) {
  
  // 获取权重张量的引用
  const Tensor& weight = weight_opt.getTensorRef();

  // 检查输入张量 self 应为 1D 或 2D
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  
  // 检查目标张量 target 应为 0D 或 1D
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");

  // 检查是否无批次维度或者批次大小相符
  auto no_batch_dim = self.dim() == 1 && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (self.size(0) == target.size(0)),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  // 获取类别数目
  const auto n_classes = self.size(-1);

  // 检查权重张量是否定义，或者其维度和元素数目与类别数相符
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == n_classes),
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  // 获取输入张量的维度数和批次大小
  const auto n_dims = self.dim();
  const auto batch_size = self.size(0);

  // 根据 reduction 参数设置输出张量的形状
  if (reduction == Reduction::None && n_dims == 2) {
    set_output_raw_strided(0, {batch_size}, {}, self.options());
  } else {
    // 在减少操作或输入为 1D 时，生成标量输出
    set_output_raw_strided(0, {}, {}, self.options());
  }

  // 设置第二个输出张量的形状
  set_output_raw_strided(1, {}, {}, self.options());
}

// 定义 nll_loss_backward 的元函数
TORCH_META_FUNC(nll_loss_backward)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight) {
  // 检查输入张量 self 的维度应为1D或2D
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  // 检查目标张量 target 的维度应不超过1
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");

  // 检查是否没有批次维度（self为1D且target为0D）
  auto no_batch_dim = self.dim() == 1  && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (self.size(0) == target.size(0)),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")")
  // 检查 total_weight 张量应为单元素张量
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a  single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  // 获取权重张量的引用
  const auto& weight = weight_opt.getTensorRef();

  // 检查权重张量应对所有类别定义，或者不定义
  TORCH_CHECK(
      !weight.defined() || weight.numel() == self.size(-1),
      "weight tensor should be defined either for all or no classes");

  // 获取输入张量的维度数
  const auto n_dims = self.dim();

  // 如果 reduction 为 Reduction::None 并且输入张量为2D，则执行以下操作
  if (reduction == Reduction::None && n_dims == 2) {
    // 获取批次大小
    const auto batch_size = self.size(0);
    // 检查梯度输出张量的维度为1且第0维大小为批次大小
    check_dim_size(grad_output, 1, 0, batch_size);
  } else {
    // 检查梯度输出张量维度应不超过1且为单元素张量
    TORCH_CHECK(
        grad_output.dim() <= 1 && grad_output.numel() == 1,
        "Expected a single element grad_output tensor, but got: ",
        grad_output.sizes());
  }

  // 设置输出的内存布局为 LEGACY_CONTIGUOUS_MEMORY_FORMAT 的原始步长张量
  set_output_raw_strided(0, self.sizes(), {}, self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
}
} // namespace at::meta

namespace at::native {

namespace {

// 如果源张量已定义，则返回连续的张量；否则返回未修改的未定义的源张量
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

// 返回张量的第一个元素的地址，如果张量未定义则返回 nullptr
template <typename scalar_t>
inline scalar_t* optional_data(const Tensor& source) {
  if constexpr (std::is_const<scalar_t>::value) {
    return source.defined() ? source.const_data_ptr<scalar_t>() : nullptr;
  } else {
    return source.defined() ? source.data_ptr<scalar_t>() : nullptr;
  }
}

// 定义了 nll_loss_out_frame 函数的模板，用于处理输出张量、总权重、输入张量、目标张量、权重张量、减少方式、忽略索引
template <typename scalar_t, typename target_t>
static void nll_loss_out_frame(
    const Tensor& output,
    const Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  // 获取输入张量的维度数
  const auto n_dims = input.dim();
  // 获取输入张量最后一个维度的类别数
  const auto n_classes = input.size(-1);

  // 获取总权重张量的数据指针，并将其设置为0
  scalar_t* total_weight_data = total_weight.data_ptr<scalar_t>();
  *total_weight_data = 0;

  // 获取连续的权重张量
  auto weight_contiguous = optional_contiguous(weight);
  // 获取连续权重张量的数据指针
  const scalar_t* weight_data = optional_data<const scalar_t>(weight_contiguous);

  // 如果 reduction 为 Reduction::None 并且输入张量为2D，则执行以下操作
  if (reduction == Reduction::None && n_dims == 2) {
    // 获取批次大小
    const auto batch_size = input.size(0);
    // 调整输出张量的大小为批次大小
    at::native::resize_output(output, {batch_size});

    // 获取输入张量的访问器
    auto input_acc = input.accessor<const scalar_t, 2>();
    // 获取 target 张量的访问器，类型为常量 target_t，维度为 1
    auto target_acc = target.accessor<const target_t, 1>();
    // 获取 output 张量的访问器，类型为 scalar_t，维度为 1
    auto output_acc = output.accessor<scalar_t, 1>();

    // 使用并行化方式对 batch_size 进行迭代处理
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        // 获取当前索引处的目标值
        const auto cur_target = target_acc[i];

        // 如果当前目标值等于 ignore_index，则将输出置为 0 并跳过当前循环
        if (cur_target == ignore_index) {
          output_acc[i] = 0;
          continue;
        }

        // 检查目标值是否在有效范围内
        TORCH_CHECK_INDEX(
            cur_target >= 0 && cur_target < n_classes,
            "Target ",
            cur_target,
            " is out of bounds.");

        // 获取当前目标值对应的权重
        scalar_t cur_weight = weight_data != nullptr ? weight_data[cur_target]
                                                     : static_cast<scalar_t>(1);
        // 计算输出的损失值
        output_acc[i] = -input_acc[i][cur_target] * cur_weight;
      }
    });

    // 函数返回
    return;
  }

  // 对于缩减情况，调整输出张量为标量输出
  at::native::resize_output(output, {});

  // 如果目标张量中元素个数为 0
  if (target.numel() == 0) {
    // 在目标张量和输入张量均为空的情况下
    // 对空张量进行均值缩减会产生 NaN。参见 https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if (reduction == Reduction::Mean) {
      // 如果缩减方式为均值，则将输出张量填充为 NaN
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      // 否则将输出张量置零
      output.zero_();
    }
    // 将总权重置零
    total_weight.zero_();
    // 函数返回
    return;
  }

  // 使输入张量连续
  auto input_contiguous = input.contiguous();
  // 使目标张量连续
  auto target_contiguous = target.contiguous();

  // 获取输入张量的常量数据指针
  const scalar_t* input_data = input_contiguous.const_data_ptr<scalar_t>();
  // 获取目标张量的常量数据指针
  const target_t* target_data = target_contiguous.const_data_ptr<target_t>();

  // 获取输入张量的维度数
  const int64_t ndim = input.dim();
  // 计算 batch_size，若输入张量维度为 1，则 batch_size 为 1，否则为第一个维度的大小
  const int64_t batch_size = ndim == 1 ? 1 : input.size(0);

  // 定义级联求和的层数
  constexpr int64_t cascade_sum_num_levels = 8;
  // 计算级联求和的幂级数，取 batch_size 的对数并向上取整后除以层数
  const int64_t level_power =
      std::max(int64_t(4), utils::CeilLog2(batch_size) / cascade_sum_num_levels);
  // 计算级联求和的步长
  const int64_t level_step = (1 << level_power);
  // 计算级联求和的掩码
  const int64_t level_mask = level_step - 1;

  // 记录被忽略的目标数量
  int64_t num_ignored = 0;

  // 定义级联求和的权重部分和数组
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  scalar_t weight_partial_sums[cascade_sum_num_levels] = {0};
  // 定义级联求和的损失部分和数组
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  scalar_t loss_partial_sums[cascade_sum_num_levels] = {0};
  // 对每个 batch 中的样本进行迭代处理
  for (const auto b : c10::irange(batch_size)) {
    // 获取当前样本的目标值
    const int64_t cur_target = target_data[b];
    // 如果当前目标值等于 ignore_index，则增加被忽略的计数并继续下一个样本
    if (cur_target == ignore_index) {
      ++num_ignored;
      continue;
    }

    // 检查目标值是否在有效范围内
    TORCH_CHECK_INDEX(
        cur_target >= 0 && cur_target < n_classes,
        "Target ",
        cur_target,
        " is out of bounds.");

    // 获取当前样本对应目标的输入数据
    const auto data = input_data[b * n_classes + cur_target];
    // 如果提供了权重数据，则使用权重值调整损失部分和，同时累加权重部分和
    if (weight_data) {
      const scalar_t weight_val = weight_data[cur_target];
      loss_partial_sums[0] -= data * weight_val;
      weight_partial_sums[0] += weight_val;
    } else {
      // 否则直接减去输入数据
      loss_partial_sums[0] -= data;
    }
    for (int64_t j = 0; j + 1 < cascade_sum_num_levels; ++j) {
      // 循环遍历级联求和的层级，直到倒数第二层
      const auto mask = (level_mask << (j * level_power));
      // 根据当前层级计算掩码值

      if (C10_LIKELY((b & mask) != 0)) {
        // 如果当前位与掩码值不为零，则跳出循环
        break;
      }

      weight_partial_sums[j + 1] += weight_partial_sums[j];
      // 将当前层级权重部分和加到下一层级

      loss_partial_sums[j + 1] += loss_partial_sums[j];
      // 将当前层级损失部分和加到下一层级

      weight_partial_sums[j] = 0;
      // 当前层级权重部分和清零

      loss_partial_sums[j] = 0;
      // 当前层级损失部分和清零
    }
  }

  const scalar_t total_weight_val = !weight_data ?
    static_cast<scalar_t>(batch_size - num_ignored) :
    std::accumulate(std::begin(weight_partial_sums),
                    std::end(weight_partial_sums),
                    scalar_t{0});
  // 计算总权重值，如果权重数据为空，则直接使用批次大小减去被忽略的数量作为总权重值

  scalar_t output_val = std::accumulate(std::begin(loss_partial_sums),
                                        std::end(loss_partial_sums),
                                        scalar_t{0});
  // 计算输出值，即损失部分和的总和

  if (reduction == Reduction::Mean) {
    // 如果指定的减少方式为平均
    output_val /= total_weight_val;
    // 将输出值除以总权重值，得到平均损失值
  }

  // 将结果写入输出张量
  *output.data_ptr<scalar_t>() = output_val;
  *total_weight_data = total_weight_val;
  // 将计算得到的输出值和总权重值分别写入输出张量和总权重数据
}

void nll_loss_forward_out_cpu_template(
    const Tensor& output,
    const Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  // 在给定输入和目标下执行 NLL 损失的前向传播
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      input.scalar_type(),
      "nll_loss_out_frame",
      [&] {
        // 根据目标数据类型选择适当的损失函数处理函数
        if (target.scalar_type() == kByte) {
          // 如果目标类型为 uint8_t，则调用相应的损失函数处理函数
          nll_loss_out_frame<scalar_t, uint8_t>(
              output,
              total_weight,
              input,
              target,
              weight,
              reduction,
              ignore_index);
        } else {
          // 否则假设目标类型为 int64_t，调用相应的损失函数处理函数
          nll_loss_out_frame<scalar_t, int64_t>(
              output,
              total_weight,
              input,
              target,
              weight,
              reduction,
              ignore_index);
        }
      });
}

template <typename scalar_t, typename target_t>
static void nll_loss_backward_out_frame(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  // 计算输入张量的维数和类别数
  const auto n_dims = input.dim();
  const auto n_classes = input.size(-1);

  // 处理目标张量，如果其维数为零，则增加一维
  auto target_ = target;
  if (target.dim() == 0) {
    target_ = target.unsqueeze(0);
  }
  auto target_acc = target_.accessor<const target_t, 1>();

  // 处理权重张量，确保其连续性
  auto weight_contiguous = optional_contiguous(weight);
  const scalar_t* weight_data = optional_data<const scalar_t>(weight_contiguous);

  // 如果不降维并且输入张量维数为 2，则并行处理每个样本
  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = input.size(0);
    auto grad_input_acc = grad_input.accessor<scalar_t, 2>();
    auto grad_output_acc = grad_output.accessor<const scalar_t, 1>();
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        auto cur_target = target_acc[i];
        if (cur_target == ignore_index) {
          continue;
        }
        // 计算梯度
        const scalar_t w =
            weight_data ? weight_data[cur_target] : static_cast<scalar_t>(1);
        grad_input_acc[i][cur_target] = -w * grad_output_acc[i];
      }
    });
    return;
  }

  // 计算总权重值和梯度输出值
  const scalar_t total_weight_value = *total_weight.const_data_ptr<scalar_t>();
  const scalar_t grad_output_value = *grad_output.const_data_ptr<scalar_t>();

  // 如果输入张量维数为 1，则处理单个样本的情况
  if (input.dim() == 1) {
    auto grad_input_acc = grad_input.accessor<scalar_t, 1>();

    const auto t = target_acc[0];
    if (t != ignore_index) {
      // 检查目标索引是否在有效范围内
      TORCH_CHECK_INDEX(t >= 0 && t < n_classes, "Target ", t, " is out of bounds.");
      // 计算梯度
      const auto grad = -(reduction == Reduction::Mean ? grad_output_value / total_weight_value
                                                       : grad_output_value);
      grad_input_acc[t] = weight_data != nullptr ? weight_data[t] * grad
                                                 : grad;
    }
  } else if (input.dim() == 2) {
    // 检查输入张量的维度是否为2，即二维张量
    auto grad_input_acc = grad_input.accessor<scalar_t, 2>();
    // 计算梯度值，根据减少方式（均值或总和）进行不同处理
    const auto grad = -(reduction == Reduction::Mean ? grad_output_value / total_weight_value
                                                     : grad_output_value);

    // 获取批量大小
    const auto batch_size = input.size(0);

    // 并行处理每个批次的数据
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      // 遍历每个批次中的数据索引范围
      for (const auto i : c10::irange(start, end)) {
        // 获取当前目标的索引值
        const auto t = target_acc[i];
        // 如果目标索引不等于忽略索引值
        if (t != ignore_index) {
          // 检查目标索引是否在有效范围内
          TORCH_CHECK_INDEX(t >= 0 && t < n_classes, "Target ", t, " is out of bounds.");
          // 根据权重数据是否为空，设置梯度值
          grad_input_acc[i][t] = weight_data != nullptr ? weight_data[t] * grad
                                                        : grad;
        }
      }
    });
  }


这段代码的作用是根据输入张量的维度和条件，计算并更新梯度值。
}

// 定义一个函数 nll_loss_backward_out_cpu_template，用于计算 NLL 损失的反向传播
void nll_loss_backward_out_cpu_template(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  
  // 将 grad_input 张量的所有元素清零
  grad_input.zero_();

  // 根据输入张量的数据类型进行分派，处理不同的数据类型情况
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      input.scalar_type(),
      "nll_loss_backward_out_frame",
      [&] {
        // 如果目标张量的数据类型是 kByte（uint8_t）
        if (target.scalar_type() == kByte) {
          // 调用模板函数 nll_loss_backward_out_frame 处理 uint8_t 类型的情况
          nll_loss_backward_out_frame<scalar_t, uint8_t>(
              grad_input,
              grad_output,
              input,
              target,
              weight,
              reduction,
              ignore_index,
              total_weight);
        } else {
          // 否则假设目标张量的数据类型为 int64_t
          // 调用模板函数 nll_loss_backward_out_frame 处理 int64_t 类型的情况
          nll_loss_backward_out_frame<scalar_t, int64_t>(
              grad_input,
              grad_output,
              input,
              target,
              weight,
              reduction,
              ignore_index,
              total_weight);
        }
      });
}

// 结束 nll_loss_backward_out_cpu_template 函数定义

} // namespace

// 实现函数 nll_loss_forward_out_cpu，用于计算 NLL 损失的前向传播
TORCH_IMPL_FUNC(nll_loss_forward_out_cpu)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  
  // 获取权重张量的引用
  const Tensor& weight = weight_opt.getTensorRef();
  
  // 调用模板函数 nll_loss_forward_out_cpu_template 处理前向传播
  nll_loss_forward_out_cpu_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
}

// 实现函数 nll_loss_backward_out_cpu，用于计算 NLL 损失的反向传播
TORCH_IMPL_FUNC(nll_loss_backward_out_cpu)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input
) {
  
  // 获取权重张量的引用
  const Tensor& weight = weight_opt.getTensorRef();
  
  // 调用模板函数 nll_loss_backward_out_cpu_template 处理反向传播
  nll_loss_backward_out_cpu_template(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);
}

// 定义静态函数 cross_entropy_loss_prob_target，用于计算交叉熵损失的概率目标版本
static Tensor cross_entropy_loss_prob_target(
    const Tensor& self,
    const Tensor& target_,
    const Tensor& weight,
    int64_t reduction,
    double label_smoothing) {
  
  // 确定类别维度和类别数目
  const auto class_dim = self.dim() == 1 ? 0 : 1;
  const auto n_classes = self.size(class_dim);
  
  // 检查权重张量的形状是否与类别数目匹配
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == n_classes),
      "cross_entropy: weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  // 对输入张量进行对数 softmax 处理
  auto input = at::log_softmax(self, class_dim, self.scalar_type());
  Tensor target;

  // 如果设置了 label_smoothing
  if (label_smoothing > 0.0) {
    // 检查 label_smoothing 的范围
    TORCH_CHECK(label_smoothing <= 1.0, "label_smoothing must be between 0.0 and 1.0. Got: ", label_smoothing);
    // 应用 label_smoothing 到目标张量
    target = target_ * (1 - label_smoothing) + label_smoothing / n_classes;
  } else {
    // 否则直接使用目标张量作为目标
    target = target_;
  }

  // 如果定义了权重张量
  if (weight.defined()) {
    // 扩展权重张量以便与输入张量和目标张量广播
    // 将权重赋值给临时变量 weight_
    Tensor weight_ = weight;
    // 检查输入张量的维度是否大于1
    if (input.dim() > 1) {
        // 创建用于广播权重的形状数组
        auto weight_broadcast_shape = SmallBuffer<int64_t, 5>(input.dim());
        // 将形状数组的所有元素初始化为1
        std::fill(weight_broadcast_shape.begin(), weight_broadcast_shape.end(), 1);
        // 将形状数组的第二个元素设置为权重张量的大小
        weight_broadcast_shape[1] = weight.size(0);
        // 将权重张量按照广播形状重塑为新的权重_
        weight_ = weight.view(weight_broadcast_shape);
    }

    // 根据指定的 reduction 类型执行不同的操作
    switch (reduction) {
      case Reduction::Mean:
        // 如果输入张量的元素数量为0，返回 NaN
        if (input.sym_numel() == 0){
          return -(input * target * weight_).sum().fill_(std::numeric_limits<double>::quiet_NaN());
        } else {
          // 返回按照平均值减少的交叉熵损失
          return -(input * target * weight_).sum() / (input.sym_numel() / n_classes);
        }
      case Reduction::Sum:
        // 返回按照总和减少的交叉熵损失
        return -(input * target * weight_).sum();
      case Reduction::None:
        // 返回未减少的交叉熵损失，并按照指定的类别维度进行求和
        return -(input * target * weight_).sum(class_dim);
      default:
        // 如果遇到无效的 reduction 类型，则抛出错误信息
        TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
    }
  } else {
    // 如果没有权重张量，根据指定的 reduction 类型执行不同的操作
    switch (reduction) {
      case Reduction::Mean:
        // 如果输入张量的元素数量为0，返回 NaN
        if (input.sym_numel() == 0){
          return -(input * target).sum().fill_(std::numeric_limits<double>::quiet_NaN());
        } else {
          // 返回按照平均值减少的交叉熵损失
          return -(input * target).sum() / (input.sym_numel() / n_classes);
        }
      case Reduction::Sum:
        // 返回按照总和减少的交叉熵损失
        return -(input * target).sum();
      case Reduction::None:
        // 返回未减少的交叉熵损失，并按照指定的类别维度进行求和
        return -(input * target).sum(class_dim);
      default:
        // 如果遇到无效的 reduction 类型，则抛出错误信息
        TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
    }
  }
// 计算交叉熵损失，支持标签平滑
static Tensor cross_entropy_loss_label_smoothing(
    const Tensor& self,                          // 输入张量
    const Tensor& target,                        // 目标张量
    const Tensor& weight,                        // 权重张量（可选）
    int64_t reduction,                           // 损失归约方式
    c10::SymInt ignore_index,                    // 忽略索引
    double label_smoothing                       // 标签平滑参数
) {
    auto class_dim = self.dim() == 1 ? 0 : 1;    // 确定类别维度
    auto input = at::log_softmax(self, class_dim, self.scalar_type()); // 对输入张量进行 log_softmax 操作
    auto nllloss = at::nll_loss_nd_symint(input, target, weight, reduction, ignore_index); // 计算负对数似然损失

    auto n_classes = input.sym_size(class_dim);  // 获取类别数量

    Tensor smooth_loss;                          // 平滑损失张量
    if (weight.defined()) {
        // 将权重扩展到与输入/目标广播的正确维度
        auto weight_broadcast_shape = SmallBuffer<int64_t, 5>(input.dim());
        std::fill(weight_broadcast_shape.begin(), weight_broadcast_shape.end(), 1);
        weight_broadcast_shape[class_dim] = weight.size(0);
        Tensor weight_ = weight.view(weight_broadcast_shape);

        smooth_loss = -(input * weight_).sum(class_dim); // 计算加权平滑损失
    } else {
        smooth_loss = -input.sum(class_dim);        // 计算非加权平滑损失
    }

    auto ignore_mask = target == std::move(ignore_index); // 创建忽略掩码
    smooth_loss.masked_fill_(ignore_mask, 0.0);   // 根据忽略掩码清零平滑损失中的相应元素

    Tensor ret;                                   // 返回张量
    switch (reduction) {
        case Reduction::Mean:
            if (weight.defined()) {
                if (isTensorSubclassLike(weight)) {
                    // 从始终有效的0索引收集权重，然后根据是否被忽略进行掩码
                    auto filtered_target = target.masked_fill(ignore_mask, 0);
                    auto tgt_weights = weight.gather(0, filtered_target.flatten());
                    auto weight_sum = tgt_weights.masked_fill_(ignore_mask.flatten(), 0).sum();
                    ret = smooth_loss.sum() / weight_sum; // 计算平均损失
                } else {
                    // 如果 #61309 被解决，可以移除此代码路径
                    // 损失通过权重进行归一化，以保持与 nll_loss_nd 一致
                    ret = smooth_loss.sum() / weight.gather(0, target.masked_select(~ignore_mask).flatten()).sum();
                }
            } else {
                auto true_mask = ~ignore_mask;
                ret = smooth_loss.sum() / true_mask.sum(); // 计算平均损失
            }
            break;
        case Reduction::Sum:
            ret = smooth_loss.sum();                  // 计算总和损失
            break;
        case Reduction::None:
            ret = smooth_loss;                        // 无归约，直接返回平滑损失
            break;
        default:
            TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction); // 不支持的损失归约类型
    }
    return (1 - label_smoothing) * nllloss + ret * (label_smoothing / n_classes); // 返回加权的交叉熵损失
}
    // 检查目标张量是否为浮点类型，否则抛出异常
    TORCH_CHECK(at::isFloatingType(target.scalar_type()),
        "Expected floating point type for target with class probabilities, got ", target.scalar_type());
    // 检查忽略索引是否小于0，如果是则抛出异常，浮点目标不支持忽略索引
    TORCH_CHECK(ignore_index < 0, "ignore_index is not supported for floating point target");

    // 查看 [Note: hacky wrapper removal for optional tensor] 处的说明
    // 从可选的张量中获取权重张量，确保操作不会复制数据，而是借用现有数据
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
    // 取出权重张量的引用
    const Tensor& weight_ = *weight_maybe_owned;
    // 调用带有类概率目标的交叉熵损失函数
    ret = cross_entropy_loss_prob_target(self, target, weight_, reduction, label_smoothing);
  } else if (label_smoothing > 0.0) {
    // 如果使用标签平滑，检查标签平滑参数是否在合理范围内（0.0 到 1.0）
    TORCH_CHECK(label_smoothing <= 1.0, "label_smoothing must be between 0.0 and 1.0. Got: ", label_smoothing);

    // 查看 [Note: hacky wrapper removal for optional tensor] 处的说明
    // 从可选的张量中获取权重张量，确保操作不会复制数据，而是借用现有数据
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
    // 取出权重张量的引用
    const Tensor& weight_ = *weight_maybe_owned;
    // 调用带有标签平滑的交叉熵损失函数
    ret = cross_entropy_loss_label_smoothing(self, target, weight_, reduction, std::move(ignore_index), label_smoothing);
  } else {
    // 自动确定类别维度，为单维度则使用0，否则使用1
    auto class_dim = self.dim() == 1 ? 0 : 1;
    // 计算负对数似然损失
    ret = at::nll_loss_nd_symint(
        // 对输入数据进行 log_softmax 处理，然后计算负对数似然
        at::log_softmax(self, class_dim, self.scalar_type()),
        // 目标张量
        target,
        // 权重张量
        weight,
        // 损失缩减方式
        reduction,
        // 忽略索引的移动语义
        std::move(ignore_index));
  }
  // 返回计算得到的损失值
  return ret;
}

Tensor & nll_loss_out(const Tensor & self, const Tensor & target, const std::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中借用权重数据
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 创建一个空的总权重张量
  Tensor total_weight = at::empty({0}, self.options());
  // 调用nll_loss_forward_out函数执行负对数似然损失的前向计算，并将结果存入output和total_weight中
  return std::get<0>(at::nll_loss_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss_symint(const Tensor & self, const Tensor & target, const std::optional<Tensor>& weight_opt, int64_t reduction, c10::SymInt ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中借用权重数据
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 调用nll_loss_forward_symint函数执行对称整数类型的负对数似然损失的前向计算，并返回结果张量
  return std::get<0>(at::nll_loss_forward_symint(self, target, weight, reduction, std::move(ignore_index)));
}

Tensor nll_loss_nd_symint(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    c10::SymInt ignore_index) {
  // 检查输入张量self的维度是否至少为1
  if (self.dim() < 1) {
    TORCH_CHECK_VALUE(
        false, "Expected 1 or more dimensions (got ", self.dim(), ")");
  }

  // 如果输入张量self的维度不为1且其第一个符号尺寸不等于目标张量target的第一个符号尺寸，则抛出错误
  if (self.dim() != 1 && self.sym_sizes()[0] != target.sym_sizes()[0]) {
    TORCH_CHECK_VALUE(
        false,
        "Expected input batch_size (",
        self.sym_sizes()[0],
        ") to match target batch_size (",
        target.sym_sizes()[0],
        ").");
  }

  // 初始化返回值张量和中间变量input_、target_
  Tensor ret;
  Tensor input_ = self;
  Tensor target_ = target;

  // 根据输入张量self的维度选择调用不同的nll_loss函数进行损失计算
  if (input_.dim() == 1 || input_.dim() == 2) {
    ret = at::nll_loss_symint(input_, target_, weight, reduction, std::move(ignore_index));
  } else if (input_.dim() == 4) {
    ret = at::nll_loss2d_symint(input_, target_, weight, reduction, std::move(ignore_index));
  } else {
    // 若维度为3或大于4，则处理非标准情况
    auto n = input_.sym_sizes()[0];
    auto c = input_.sym_sizes()[1];
    auto out_size = input_.sym_sizes().slice(2).vec();
    out_size.insert(out_size.begin(), n);

    // 检查目标张量target的尺寸是否符合预期
    if (target_.sym_sizes().slice(1) != input_.sym_sizes().slice(2)) {
      TORCH_CHECK(
          false,
          "Expected target size ",
          SymIntArrayRef(out_size),
          ", got ",
          target_.sym_sizes());
    }

    // 使输入张量和目标张量连续化
    input_ = input_.contiguous();
    target_ = target_.contiguous();

    // 支持空批次，参见GitHub问题#15870
    if (input_.sym_numel() > 0) {
      input_ = input_.view_symint({n, std::move(c), 1, -1});
    } else {
      input_ = input_.view_symint({n, std::move(c), 0, 0});
    }

    if (target_.sym_numel() > 0) {
      target_ = target_.view_symint({std::move(n), 1, -1});
    } else {
      target_ = target_.view_symint({std::move(n), 0, 0});
    }

    // 根据reduction参数选择调用不同的nll_loss函数进行损失计算
    if (reduction != Reduction::None) {
      ret = at::nll_loss2d_symint(input_, target_, weight, reduction, std::move(ignore_index));
    } else {
      // 如果条件不满足，则执行以下代码块
      auto out =
          at::nll_loss2d_symint(input_, target_, weight, reduction, std::move(ignore_index));
      // 调用 ATen 库的 nll_loss2d_symint 函数计算对称整数标签的负对数似然损失
      ret = out.view_symint(out_size);
      // 将 out 对象按照指定的 out_size 尺寸进行对称整数视图重塑，并赋值给 ret
    }
  }
  // 返回计算结果 ret
  return ret;
}

} // namespace at::native
```