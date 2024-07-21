# `.\pytorch\aten\src\ATen\native\LossNLL2d.cpp`

```
// 定义宏，用于在编译时仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含一组功能和原生函数的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，则包含一组具体的操作头文件
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/nll_loss2d_backward_native.h>
#include <ATen/ops/nll_loss2d_forward.h>
#include <ATen/ops/nll_loss2d_forward_native.h>
#include <ATen/ops/nll_loss2d_native.h>
#include <ATen/ops/zeros_like.h>

#include <utility>
#endif

namespace at::native {

// 命名空间内部的匿名命名空间，用于实现私有的辅助函数或变量

// 如果源张量已定义，则返回连续的张量；否则返回未定义的源张量
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

// 返回张量的第一个元素的地址，如果张量未定义则返回 nullptr
template <typename scalar_t>
inline scalar_t* optional_data(const Tensor& source) {
  // 使用 constexpr 条件编译，检查 scalar_t 是否为常量类型
  if constexpr (std::is_const<scalar_t>::value) {
    return source.defined() ? source.const_data_ptr<scalar_t>() : nullptr;
  } else {
    return source.defined() ? source.data_ptr<scalar_t>() : nullptr;
  }
}

// 检查输入张量的维度是否符合 nll_loss2d 的要求
inline void check_inputs_nll_loss2d(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight) {
  // 检查目标张量的维度是否为 3
  TORCH_CHECK(
      target.dim() == 3,
      "only batches of spatial targets supported (3D tensors)"
      " but got targets of dimension: ",
      target.dim());
  // 检查输入张量的维度是否为 4
  TORCH_CHECK(
      input.dim() == 4,
      "only batches of spatial inputs supported (4D tensors), "
      "but got input of dimension: ",
      input.dim());
  // 检查权重张量的定义情况，或者权重张量的元素数是否等于输入张量的第一维大小
  TORCH_CHECK(
      !weight.defined() || weight.numel() == input.size(1),
      "weight tensor should be defined either for all or no classes");

  // 获取各张量的尺寸信息
  const int64_t input0 = input.size(0);
  const int64_t input2 = input.size(2);
  const int64_t input3 = input.size(3);
  const int64_t target0 = target.size(0);
  const int64_t target1 = target.size(1);
  const int64_t target2 = target.size(2);
  // 检查输入和目标张量的尺寸是否匹配
  TORCH_CHECK(
      input0 == target0 && input2 == target1 && input3 == target2,
      "size mismatch (got input: ",
      input.sizes(),
      " , target: ",
      target.sizes());
}

// 检查梯度输出张量的形状是否符合 nll_loss2d 的要求
inline void check_gradout_shape_nll_loss2d(
    const Tensor& grad_output,
    # 对梯度输出（grad_output）和目标张量（target）的维度进行验证
    TORCH_CHECK(
        grad_output.dim() == 3,  # 检查梯度输出的维度是否为3
        "grad_output must have same dimension as target (3) but got dimension: ",
        grad_output.sizes()  # 打印实际得到的梯度输出的维度信息
    )
    
    # 提取各维度的大小，用于后续的维度匹配验证
    const int64_t grad_output0 = grad_output.size(0);
    const int64_t grad_output1 = grad_output.size(1);
    const int64_t grad_output2 = grad_output.size(2);
    const int64_t target0 = target.size(0);
    const int64_t target1 = target.size(1);
    const int64_t target2 = target.size(2);
    
    # 验证梯度输出和目标张量的维度是否完全匹配
    TORCH_CHECK(
        grad_output0 == target0 && grad_output1 == target1 &&
            grad_output2 == target2,
        "size mismatch (got grad_output: ",
        grad_output.sizes(),  # 打印实际得到的梯度输出的大小信息
        " target: ",
        target.sizes()  # 打印目标张量的大小信息
    );
  // 如果目标张量为空，根据 reduction 类型设置输出张量和总权重张量的值
  if (target.numel() == 0) {
    // 在空张量上进行均值约简会产生 NaN。参考：https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if (reduction == Reduction::Mean) {
      // 将输出张量填充为 NaN
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      // 将输出张量置零
      output.zero_();
    }
    // 将总权重张量置零
    total_weight.zero_();
  }
    return;
  }

  // 将输入和目标张量转换为连续内存存储
  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  // 获取输入数据和目标数据的指针
  const scalar_t* input_data = input_contiguous.const_data_ptr<scalar_t>();
  const int64_t* target_data = target_contiguous.const_data_ptr<int64_t>();

  // 获取批次大小、特征图大小、样本大小以及迭代次数
  const int64_t batch_size = input.size(0);
  const int64_t map_size = input.size(2) * input.size(3);
  const int64_t sample_size = map_size * n_classes;
  const int64_t numiter = batch_size * map_size;

  // 定义级联求和使用的级别数量
  constexpr int64_t cascade_sum_num_levels = 8;

  // 定义局部权重和损失的数组，初始化为零
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  scalar_t weight_partial_sums[cascade_sum_num_levels] = {0};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  scalar_t loss_partial_sums[cascade_sum_num_levels] = {0};

  // 计算每个级别的步长和掩码
  const int64_t level_power =
      std::max(int64_t(4), utils::CeilLog2(numiter) / cascade_sum_num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  // 记录被忽略的样本数
  int64_t num_ignored = 0;

  // 遍历每个批次和每个特征图元素
  for (const auto b : c10::irange(batch_size)) {
    for (const auto elem : c10::irange(map_size)) {
      // 获取当前目标值
      const int64_t cur_target = target_data[b * map_size + elem];
      // 如果当前目标值等于忽略索引，则增加被忽略计数并继续下一个循环
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

      // 计算输入数据的索引并获取数据值
      const auto data = input_data[b * sample_size + cur_target * map_size + elem];

      // 如果有权重数据，计算权重值并更新损失和权重部分和数组
      if (weight_data) {
        const scalar_t weight_val = weight_data[cur_target];
        loss_partial_sums[0] -= data * weight_val;
        weight_partial_sums[0] += weight_val;
      } else {
        // 否则仅更新损失部分和数组
        loss_partial_sums[0] -= data;
      }

      // 计算线性索引
      const int64_t linear_idx = b * map_size + elem;

      // 对级联求和的不同级别进行迭代
      for (int64_t j = 0; j + 1 < cascade_sum_num_levels; ++j) {
        const auto mask = (level_mask << (j * level_power));
        // 如果当前线性索引与当前级别的掩码与运算结果不为零，则退出循环
        if (C10_LIKELY((linear_idx & mask) != 0)) {
          break;
        }

        // 更新权重和损失部分和数组
        weight_partial_sums[j + 1] += weight_partial_sums[j];
        loss_partial_sums[j + 1] += loss_partial_sums[j];

        // 重置当前级别的权重和损失部分和数组
        weight_partial_sums[j] = 0;
        loss_partial_sums[j] = 0;
      }
    }
  }

  // 计算总权重值
  const scalar_t total_weight_val = !weight_data ?
    static_cast<scalar_t>(numiter - num_ignored) :
    std::accumulate(std::begin(weight_partial_sums),
                    std::end(weight_partial_sums),
                    scalar_t{0});

  // 计算最终输出值
  scalar_t output_val = std::accumulate(std::begin(loss_partial_sums),
                                        std::end(loss_partial_sums),
                                        scalar_t{0});

  // 如果指定了平均值缩减方式，则计算平均损失
  if (reduction == Reduction::Mean) {
    output_val /= total_weight_val;
  }

  // 将总权重值写入输出的权重数据指针
  *total_weight_data = total_weight_val;
  // 将输出值写入输出张量的数据指针
  *output.data_ptr<scalar_t>() = output_val;
}

void nll_loss2d_forward_out_cpu_template(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  // 检查输入的合法性
  check_inputs_nll_loss2d(input, target, weight);
  // 重新设置总权重张量的形状为空
  total_weight.resize_({});

  // 根据输入张量的数据类型进行分发计算
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      input.scalar_type(),
      "nll_loss2d_forward_out_frame",
      [&] {
        // 调用模板函数nll_loss2d_forward_out_frame，执行前向传播计算
        nll_loss2d_forward_out_frame<scalar_t>(
            output,
            total_weight,
            input,
            target,
            weight,
            reduction,
            ignore_index);
      });
}

template <typename scalar_t>
static void nll_loss2d_backward_out_frame(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  // 检查权重张量是否需要转置以便连续访问
  auto weight_contiguous = optional_contiguous(weight);
  // 获取权重数据的指针
  const scalar_t* weight_data = optional_data<const scalar_t>(weight_contiguous);

  // 如果减少(reduction)方式为非降维模式
  if (reduction == at::Reduction::None) {
    // 检查梯度输出的形状是否与目标张量匹配
    check_gradout_shape_nll_loss2d(grad_output, target);

    // 获取输入张量的批次大小、高度和宽度
    const int64_t batch_size = input.size(0);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);

    // 访问梯度输入和梯度输出的访问器
    auto grad_input_acc = grad_input.accessor<scalar_t, 4>();
    auto grad_output_acc = grad_output.accessor<const scalar_t, 3>();
    auto target_acc = target.accessor<const int64_t, 3>();

    // 使用并行处理，对每个批次、高度和宽度进行迭代
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto b : c10::irange(start, end)) {
        for (const auto h : c10::irange(H)) {
          for (const auto w : c10::irange(W)) {
            // 获取当前像素位置的目标值
            const int64_t cur_target = target_acc[b][h][w];
            // 如果当前目标值等于忽略索引，跳过此次迭代
            if (cur_target == ignore_index) {
              continue;
            }
            // 计算权重值，如果权重数据存在则使用相应的权重值，否则使用默认值1
            const scalar_t value =
                -(weight_data ? weight_data[cur_target]
                              : static_cast<scalar_t>(1));
            // 获取梯度输出值
            const scalar_t grad_output_value = grad_output_acc[b][h][w];
            // 计算并设置梯度输入值
            grad_input_acc[b][cur_target][h][w] = value * grad_output_value;
          }
        }
      }
    });
    // 函数返回，表示结束函数执行
    return;
  }

  // 获取 total_weight 引用的标量值
  const scalar_t total_weight_value = *total_weight.const_data_ptr<scalar_t>();

  // 检查 grad_output 张量是否是一维且只有一个元素
  TORCH_CHECK(
      grad_output.dim() <= 1 && grad_output.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      grad_output.sizes());

  // 获取 grad_output 张量中的标量值
  const scalar_t grad_output_value = *grad_output.const_data_ptr<scalar_t>();

  // 获取 target 张量的连续版本，并获取其数据指针
  const auto target_contiguous = target.contiguous();
  const int64_t* target_data = target_contiguous.const_data_ptr<int64_t>();

  // 获取 grad_input 的可变数据指针
  scalar_t* grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 获取输入张量的维度信息
  const int64_t batch_size = input.size(0);
  const int64_t n_classes = input.size(1);
  const int64_t map_size = input.size(2) * input.size(3);
  const int64_t sample_size = map_size * n_classes;

  // 计算梯度
  const auto grad = -(reduction == Reduction::Mean ? grad_output_value / total_weight_value
                                                   : grad_output_value);

  // 并行处理每个批次中的数据
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (const auto b : c10::irange(start, end)) {
      for (const auto elem : c10::irange(map_size)) {
        // 获取目标值索引
        const int64_t t = target_data[b * map_size + elem];

        // 如果目标值不是忽略索引，则执行以下操作
        if (t != ignore_index) {
          // 检查目标值是否在有效范围内
          TORCH_CHECK_INDEX(t >= 0 && t < n_classes, "Target ", t, " is out of bounds.");

          // 计算 grad_input_data 中的索引位置
          const int64_t index = b * sample_size + t * map_size + elem;

          // 根据是否有权重数据，更新梯度输入数据
          grad_input_data[index] = weight_data != nullptr ? weight_data[t] * grad
                                                          : grad;
        }
      }
    }
  });
} // 结束 nll_loss2d_backward_out_cpu_template 函数的定义

void nll_loss2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  // 检查输入张量的有效性
  check_inputs_nll_loss2d(input, target, weight);

  // 重新调整 grad_input 张量的大小，使其与 input 张量相同，并清零
  grad_input.resize_as_(input);
  grad_input.zero_();

  // 断言 grad_input 张量是连续的
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

  // 断言 total_weight 张量仅包含一个元素
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  // 根据 input 的数据类型分发到具体的计算函数
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      input.scalar_type(),
      "nll_loss2d_backward_out_frame",
      [&] {
        nll_loss2d_backward_out_frame<scalar_t>(
            grad_input,
            grad_output,
            input,
            target,
            weight,
            reduction,
            ignore_index,
            total_weight);
      });
}

} // 结束 namespace

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out_cpu(const Tensor& self,
    const Tensor& target, const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中借用权重
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 调用 nll_loss2d_forward_out_cpu_template 函数计算前向传播
  nll_loss2d_forward_out_cpu_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
  
  // 返回输出张量和总权重的元组引用
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_cpu(
    const Tensor& self,
    const Tensor& target, const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中借用权重
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 创建一个空的输出张量和总权重张量
  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  
  // 调用 at::native::nll_loss2d_forward_out_cpu 函数计算前向传播
  at::native::nll_loss2d_forward_out_cpu(
      self, target, weight, reduction, ignore_index, output, total_weight);
  
  // 返回输出张量和总权重的元组
  return std::make_tuple(output, total_weight);
}

Tensor& nll_loss2d_backward_out_cpu(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target, const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    // 引用传入的梯度张量 grad_output，表示损失函数的反向传播结果
    Tensor& grad_input) {
      // 查看注释：用于处理可选张量的包装器移除 [Note: hacky wrapper removal for optional tensor]
      // 从可选的张量 weight_opt 中借用权重数据，转换为 MaybeOwned<Tensor> 类型
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      // 将借用的权重数据解引用为常量引用 Tensor 类型
      const Tensor& weight = *weight_maybe_owned;
    
      // 调用 nll_loss2d_backward_out_cpu_template 模板函数，执行二维 NLL 损失函数的反向传播
      nll_loss2d_backward_out_cpu_template(
          grad_input,    // 输出的梯度张量，用于存储计算得到的输入梯度
          grad_output,   // 输入的梯度张量，即损失函数的反向传播梯度
          self,          // 自身张量，通常为损失函数的输入张量
          target,        // 目标张量，损失函数的目标值
          weight,        // 权重张量，用于加权损失函数计算
          reduction,     // 损失函数的减少方式，如求和或平均
          ignore_index,  // 忽略的索引，对应于目标张量中的忽略类别
          total_weight); // 总体权重，用于加权计算损失值
    
      // 返回输入的梯度张量 grad_input，经过损失函数反向传播计算后更新其值
      return grad_input;
    }
} // namespace at::native
```