# `.\pytorch\aten\src\ATen\native\LossMultiMargin.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义预处理指令，指定仅包含操作符方法

#include <ATen/core/Tensor.h>
// 引入 ATen 核心张量头文件

#include <ATen/AccumulateType.h>
// 引入 ATen 累加类型头文件

#include <ATen/Dispatch.h>
// 引入 ATen 分发头文件

#include <ATen/native/LossMulti.h>
// 引入 ATen 多损失头文件

#include <c10/util/irange.h>
// 引入 c10 工具中的 irange 函数头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/multi_margin_loss_backward_native.h>
#include <ATen/ops/multi_margin_loss_native.h>
#endif
// 根据 AT_PER_OPERATOR_HEADERS 的定义条件引入不同的 ATen 操作头文件

namespace at::native {

namespace {

template <typename scalar_t>
inline scalar_t multi_margin_inner_sum_cpu(
    const scalar_t* input_data,
    const scalar_t* weight_data,
    const int p,
    const scalar_t margin,
    const int64_t dim,
    const int64_t target_idx) {
  const scalar_t input_target = input_data[target_idx];
  // 获取目标索引处的输入数据值

  scalar_t sum = 0;
  // 初始化总和为零
  for (const auto d : c10::irange(dim)) {
    // 遍历维度范围
    if (d == target_idx) {
      // 如果当前维度索引等于目标索引，则跳过
      continue;
    }

    const scalar_t z = margin - input_target + input_data[d];
    // 计算损失函数中的 z 值

    if (z > 0) {
      // 如果 z 大于零
      scalar_t h = (p == 1) ? z : z * z;
      // 根据 p 的值选择计算方式

      if (weight_data != nullptr) {
        h *= weight_data[target_idx];
        // 如果权重数据不为空，则根据目标索引调整 h 的值
      }
      sum += h;
      // 累加 h 到总和
    }
  }

  sum /= dim;
  // 对总和进行维度归一化处理
  return sum;
  // 返回归一化后的总和值
}

inline int64_t target_index_checked(
    const int64_t* target_data,
    const int64_t index,
    const int64_t dim) {
  const int64_t idx = target_data[index];
  // 获取目标数据中指定索引的值

  TORCH_CHECK(idx >= 0 && idx < dim, "target out of range");
  // 使用 Torch 的检查功能确认索引值在范围内

  return idx;
  // 返回确认后的索引值
}

template <typename scalar_t>
static inline void multi_margin_loss_cpu_kernel(
    Tensor& output,
    const scalar_t* input_data,
    const int64_t* target_data,
    const int p,
    scalar_t margin,
    const scalar_t* weight_data,
    const int64_t nframe,
    const int64_t dim,
    const int64_t reduction) {
  using accscalar_t = at::acc_type<scalar_t, false>;

  // dim() != 0 check is for 1d input which produces a scalar output (that
  // cannot be handled by TensorAccessor)
  // 对于产生标量输出的一维输入进行维度检查（不能由 TensorAccessor 处理）
  if (reduction == Reduction::None && output.dim() > 0) {
    // 如果没有减少，并且输出张量的维度大于 0
    auto output_acc = output.accessor<scalar_t, 1>();
    // 使用 TensorAccessor 访问输出张量的数据

    for (const auto t : c10::irange(nframe)) {
      // 遍历帧数范围
      const auto idx = target_index_checked(target_data, t, dim);
      // 获取检查后的目标索引值

      auto sum = multi_margin_inner_sum_cpu(
          input_data, weight_data, p, margin, dim, idx);
      // 计算多边距损失函数的内部总和

      output_acc[t] = sum;
      // 将计算得到的总和值存储到输出张量对应位置
      input_data += dim;
      // 移动输入数据指针到下一个帧的起始位置
    }
  } else {
    // 否则
    accscalar_t sum = 0;
    // 使用累加类型初始化总和为零
    auto output_acc = output.data_ptr<scalar_t>();
    // 获取输出张量的数据指针

    for (const auto t : c10::irange(nframe)) {
      // 遍历帧数范围
      const auto idx = target_index_checked(target_data, t, dim);
      // 获取检查后的目标索引值

      sum += multi_margin_inner_sum_cpu(
          input_data, weight_data, p, margin, dim, idx);
      // 累加计算多边距损失函数的内部总和

      input_data += dim;
      // 移动输入数据指针到下一个帧的起始位置
    }
    if (reduction == Reduction::Mean) {
      // 如果减少方式为均值
      sum /= nframe;
      // 对总和进行均值处理
    }
    output_acc[0] = sum;
    // 存储处理后的总和到输出张量的数据指针位置
  }
}

void multi_margin_loss_out_cpu_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    // 定义函数 multi_margin_loss，并传入参数 input、target、weight 和 reduction
    int64_t reduction) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 声明变量 nframe 和 dim，用于存储帧数和维度信息
      int64_t nframe, dim;
      // 获取输入张量的维度数量
      const auto ndims = input.dim();
    
      // 检查参数 p 的取值，只支持 p 等于 1 或 2
      TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");
    
      // 进行形状检查，获取 nframe 和 dim 的值，并验证输入输出张量的合法性
      multi_margin_loss_shape_check(nframe, dim, ndims, input, target, weight);
    
      // 如果不进行汇总（reduction == Reduction::None）且目标张量的维度大于 0，则输出张量的形状为 {nframe}
      if (reduction == Reduction::None && target.dim() > 0) {
        output.resize_({nframe});
      } else {
        // 否则，输出张量的形状为空
        output.resize_({});
      }
      // 如果输入张量的元素个数为 0，则直接返回
      if (input.numel() == 0) {
        return;
      }
    
      // 将输入张量转为连续存储，确保数据连续性
      auto input_contiguous = input.contiguous();
      auto target_contiguous = target.contiguous();
      Tensor weight_contiguous;
      // 如果权重张量存在且已定义，则将其转为连续存储
      if (weight && weight->defined()) {
        weight_contiguous = weight->contiguous();
      }
    
      // 根据输入张量的数据类型调度相应的 CPU 内核函数
      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "multi_margin_loss_cpu_kernel", [&] {
            // 获取输入张量、目标张量和权重张量的数据指针
            auto input_data = input_contiguous.const_data_ptr<scalar_t>();
            auto target_data = target_contiguous.const_data_ptr<int64_t>();
            auto weight_data =
                weight_contiguous.defined() ? weight_contiguous.const_data_ptr<scalar_t>() : nullptr;
            // 调用多分类边界损失的 CPU 内核函数，计算损失并存入输出张量
            multi_margin_loss_cpu_kernel<scalar_t>(
                output,
                input_data,
                target_data,
                p,
                margin.to<scalar_t>(),
                weight_data,
                nframe,
                dim,
                reduction);
          });
// 多分类边界损失函数的CPU版本反向传播核函数
template <typename scalar_t>
static void multi_margin_loss_backward_cpu_kernel(
    scalar_t* grad_input_data,                           // 梯度输入数据的指针
    const Tensor& grad_output,                           // 梯度输出张量
    const scalar_t* input_data,                          // 输入数据的指针
    const int64_t* target_data,                          // 目标数据的指针
    int p,                                               // 范数的阶数
    scalar_t margin,                                     // 边界间隔
    scalar_t g,                                          // 损失函数的梯度比例
    const scalar_t* weight_data,                         // 权重数据的指针
    int64_t nframe,                                      // 批处理中的样本数
    int64_t dim,                                         // 输入张量的维度
    int64_t reduction) {                                 // 损失函数的减少方式
  scalar_t* grad_input_row_data = grad_input_data;       // 指向梯度输入数据的当前行数据指针
  for (const auto t : c10::irange(nframe)) {             // 遍历样本批次
    int64_t target_idx = target_index_checked(target_data, t, dim);  // 检查目标索引位置
    scalar_t input_target = input_data[target_idx];      // 目标位置的输入数据值
    scalar_t grad_input_target = 0;                      // 目标位置的梯度输入初始值为0
    for (const auto d : c10::irange(dim)) {              // 遍历输入张量的维度
      scalar_t z = margin - input_target + input_data[d];  // 计算边界间隔损失函数的差值z
      if (d == target_idx) {                             // 如果当前维度是目标索引位置，则跳过
        continue;
      }

      if (z > 0) {                                       // 如果z大于0
        scalar_t h = (p == 1) ? g : 2 * g * z;            // 根据范数阶数计算h值
        if (weight_data != nullptr) {                    // 如果存在权重数据
          h *= weight_data[target_idx];                   // 应用权重到损失梯度
        }
        grad_input_target -= h;                          // 更新目标位置的梯度输入值
        grad_input_row_data[d] = h;                      // 更新当前行的梯度输入数据
      } else {
        grad_input_row_data[d] = 0;                      // 否则当前行的梯度输入数据为0
      }
    }
    grad_input_row_data[target_idx] = grad_input_target; // 更新目标位置的梯度输入数据

    input_data += dim;                                   // 移动输入数据指针到下一个样本
    grad_input_row_data += dim;                          // 移动梯度输入数据指针到下一个样本
  }

  if (reduction != Reduction::None || grad_output.dim() == 0) {
    assert(
        reduction != Reduction::None || grad_output.dim() > 0 ||
        nframe == 1); // 检查1维标量回退情况
    const auto d = *grad_output.const_data_ptr<scalar_t>();  // 获取梯度输出的常量数据指针
    for (int64_t t = 0; t < nframe * dim; t++) {           // 遍历所有梯度输入数据
      grad_input_data[t] *= d;                            // 应用梯度输出到梯度输入数据
    }
  } else {
    auto grad_output_acc = grad_output.accessor<const scalar_t, 1>();  // 获取梯度输出的访问器
    for (const auto t : c10::irange(nframe)) {             // 遍历样本批次
      for (const auto d : c10::irange(dim)) {              // 遍历输入张量的维度
        grad_input_data[t * dim + d] *= grad_output_acc[t];  // 应用梯度输出到梯度输入数据
      }
    }
  }
}

// 多分类边界损失函数的CPU版本模板，计算梯度输入
void multi_margin_loss_backward_out_cpu_template(
    Tensor& grad_input,                                  // 梯度输入张量
    const Tensor& grad_output,                           // 梯度输出张量
    const Tensor& input,                                 // 输入张量
    const Tensor& target,                                // 目标张量
    int p,                                               // 范数的阶数
    const Scalar& margin,                                // 边界间隔
    const Tensor& weight,                                // 权重张量
    int64_t reduction) {                                 // 损失函数的减少方式
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t nframe, dim;                                   // 样本批次数和输入张量的维度
  const auto ndims = input.dim();                        // 输入张量的维度数

  TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");  // 检查范数阶数支持情况

  multi_margin_loss_shape_check(nframe, dim, ndims, input, target, weight);  // 检查输入张量的形状
  grad_input.resize_as_(input);                          // 调整梯度输入张量与输入张量相同大小
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");  // 检查梯度输入是否连续

  if (input.numel() == 0) {                              // 如果输入张量为空
    // 空语句，直接返回，结束函数执行
    return;
  }

  // 将输入张量、目标张量和权重张量转换为连续存储以提高计算效率
  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto weight_contiguous = weight.contiguous();
  
  // 根据输入张量的数据类型，调度对应的浮点类型的处理函数
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multi_margin_loss_backward_cpu_kernel", [&] {
        // 获取梯度输入张量的可变数据指针
        auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
        // 获取输入张量的常量数据指针
        auto input_data = input_contiguous.const_data_ptr<scalar_t>();
        // 获取目标张量的常量数据指针（假设目标为int64类型）
        auto target_data = target_contiguous.const_data_ptr<int64_t>();
        // 如果权重张量已定义，则获取其常量数据指针；否则设为nullptr
        auto weight_data = weight_contiguous.defined()
            ? weight_contiguous.const_data_ptr<scalar_t>()
            : nullptr;
        // 根据指定的减少类型（平均或总和）设定归一化因子g
        scalar_t g = reduction == Reduction::Mean
            ? static_cast<scalar_t>(1. / (nframe * dim))
            : static_cast<scalar_t>(1. / dim);
        
        // 调用多类别边界损失函数的反向传播 CPU 内核
        multi_margin_loss_backward_cpu_kernel<scalar_t>(
            grad_input_data,
            grad_output,
            input_data,
            target_data,
            p,
            margin.to<scalar_t>(),
            g,
            weight_data,
            nframe,
            dim,
            reduction);
      });
} // namespace
```