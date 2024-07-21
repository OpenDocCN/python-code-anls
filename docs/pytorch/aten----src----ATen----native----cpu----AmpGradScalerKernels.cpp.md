# `.\pytorch\aten\src\ATen\native\cpu\AmpGradScalerKernels.cpp`

```
// 定义宏以仅包含方法操作符

#include <ATen/native/AmpKernels.h>  // 包含AMP核心功能的头文件
#include <math.h>  // 数学函数库
#include <ATen/DeviceGuard.h>  // 包含设备守卫相关的头文件
#include <ATen/Dispatch.h>  // 包含分发相关的头文件
#include <ATen/OpMathType.h>  // 包含操作数类型相关的头文件
#include <ATen/core/Tensor.h>  // 包含张量核心功能的头文件
#include <ATen/native/ForeachUtils.h>  // 包含Foreach工具相关的头文件
#include <ATen/native/TensorIterator.h>  // 包含Tensor迭代器相关的头文件
#include <ATen/native/cpu/Loops.h>  // 包含CPU循环相关的头文件
#include <ATen/cpu/vec/vec.h>  // 包含向量化操作相关的头文件
#include <ATen/cpu/vec/functional.h>  // 包含向量功能相关的头文件

namespace at::native {

namespace {
// 模仿CUDA的实现。
// 该函数对scaled_grads中的每个张量乘以inv_scale，并进行原地操作。
// 如果scaled_grads中任何张量的任何元素为inf或NaN，则将found_inf设置为1.0。
//
// Args:
// scaled_grads: 一个TensorList，包含了被缩放梯度的张量列表，可能包含inf或NaN。
// found_inf: 一个单元素浮点张量，如果任何梯度包含inf或NaN，则将写入1.0。
//            如果合适，调用者的责任是预先将found_inf清零。
// inv_scale: 当前用于乘以scaled_grads的比例因子的倒数。
void _amp_foreach_non_finite_check_and_unscale_cpu_kernel(
    TensorList scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  if (scaled_grads.size() == 0) {
    return;  // 如果scaled_grads为空，则直接返回
  }

  TORCH_CHECK(inv_scale.is_cpu(), "inv_scale必须是CPU张量。");  // 检查inv_scale必须是CPU张量
  TORCH_CHECK(found_inf.is_cpu(), "found_inf必须是CPU张量。");  // 检查found_inf必须是CPU张量
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale必须是一个包含一个元素的张量。");  // 检查inv_scale必须是包含一个元素的张量
  TORCH_CHECK(found_inf.numel() == 1, "found_inf必须是一个包含一个元素的张量。");  // 检查found_inf必须是包含一个元素的张量
  TORCH_CHECK(
      inv_scale.scalar_type() == at::ScalarType::Float,
      "inv_scale必须是一个浮点数张量。");  // 检查inv_scale必须是浮点数张量
  TORCH_CHECK(
      found_inf.scalar_type() == at::ScalarType::Float,
      "found_inf必须是一个浮点数张量。");  // 检查found_inf必须是浮点数张量

  // 确保客户端代码（GradScaler）已通过dtype对scaled_grads进行了筛选。
  at::native::check_foreach_api_restrictions(scaled_grads);  // 检查Foreach API的限制条件
  for (const at::Tensor& t : scaled_grads) {
    TORCH_CHECK(t.is_cpu(), "scaled_grads中的一个张量不是CPU张量。");  // 检查scaled_grads中的张量必须是CPU张量
    TORCH_CHECK(
        t.layout() == at::kStrided,
        "scaled_grads中的一个张量不是分块张量。");  // 检查scaled_grads中的张量必须是分块张量
    auto iter = at::TensorIterator::unary_op(
        const_cast<at::Tensor&>(t), const_cast<at::Tensor&>(t));  // 创建Tensor迭代器用于对t进行操作
    # 检查迭代器的数据类型是否是降低浮点类型（半精度浮点数），使用 ATen 的方法
    if (at::isReducedFloatingType(iter.dtype())) {
      # 在降低浮点类型下，使用指定的 CPU 内核向量化操作
      AT_DISPATCH_REDUCED_FLOATING_TYPES(
      iter.dtype(),
      "_amp_foreach_non_finite_check_and_unscale_cpu",
      [&iter, &found_inf, &inv_scale] {
          # 获取指向 found_inf 和 inv_scale 的指针，指向 float 类型的数据
          auto* found_inf_ptr = found_inf.data_ptr<float>();
          auto* inv_scale_ptr = inv_scale.data_ptr<float>();

          # 定义操作数的数学类型为 scalar_t 对应的操作数类型
          using opmath_t = at::opmath_type<scalar_t>;

          # 调用 ATen 提供的 CPU 内核向量化函数
          at::native::cpu_kernel_vec(
              iter,
              # 对每个标量值进行操作，返回一个标量值
              [found_inf_ptr, inv_scale_ptr](scalar_t val_in) -> scalar_t {
                # 将输入值转换为 opmath_t 类型
                auto val = static_cast<opmath_t>(val_in);
                # 如果值不是有限的，则设置 found_inf_ptr 指向的值为 1.0
                if (!std::isfinite(val)) {
                  *found_inf_ptr = 1.f;
                }
                # 每个线程访问 inv_scale，但它会在缓存中命中
                const auto inv_scale_val = *inv_scale_ptr;
                # 返回经过缩放的标量值
                return static_cast<scalar_t>(
                    inv_scale_val == 1.f ? val : val * inv_scale_val);
              },
              # 对每个向量化的标量值进行操作，返回一个向量化的标量值
              [found_inf_ptr, inv_scale_ptr](Vectorized<scalar_t> val_vec) -> Vectorized<scalar_t>{
                # 将向量分解为两个 float 类型的值
                auto [val_vec0, val_vec1] = convert_to_float<scalar_t>(val_vec);
                # 如果任一向量元素包含无穷大或 NaN，则设置 found_inf_ptr 指向的值为 1.0
                if (val_vec0.has_inf_nan() || val_vec1.has_inf_nan()) {
                  *found_inf_ptr = 1.f;
                }
                # 每个线程访问 inv_scale，但它会在缓存中命中
                const auto inv_scale_val = *inv_scale_ptr;
                # 对每个向量元素应用缩放因子，并返回转换回标量值的结果
                val_vec0 = inv_scale_val == 1.f ? val_vec0 : val_vec0 * Vectorized<opmath_t>(inv_scale_val);
                val_vec1 = inv_scale_val == 1.f ? val_vec1 : val_vec1 * Vectorized<opmath_t>(inv_scale_val);
                return convert_from_float<scalar_t>(val_vec0, val_vec1);
              });
      });
    } else {
      # 对于其他浮点类型，使用指定的 CPU 内核向量化操作
      AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(),
        "_amp_foreach_non_finite_check_and_unscale_cpu",
        [&iter, &found_inf, &inv_scale] {
          # 获取指向 found_inf 和 inv_scale 的指针，指向 float 类型的数据
          auto* found_inf_ptr = found_inf.data_ptr<float>();
          auto* inv_scale_ptr = inv_scale.data_ptr<float>();
          # 调用 ATen 提供的 CPU 内核向量化函数
          at::native::cpu_kernel_vec(
              iter,
              # 对每个标量值进行操作，返回一个标量值
              [found_inf_ptr, inv_scale_ptr](scalar_t val_in) -> scalar_t {
                # 如果值不是有限的，则设置 found_inf_ptr 指向的值为 1.0
                if (!std::isfinite(val_in)) {
                  *found_inf_ptr = 1.f;
                }
                # 每个线程访问 inv_scale，但它会在缓存中命中
                const auto inv_scale_val = *inv_scale_ptr;
                # 返回经过缩放的标量值
                return static_cast<scalar_t>(
                    inv_scale_val == 1.f ? val_in : val_in * inv_scale_val);
              },
              # 对每个向量化的标量值进行操作，返回一个向量化的标量值
              [found_inf_ptr, inv_scale_ptr](Vectorized<scalar_t> val_vec) -> Vectorized<scalar_t>{
                # 如果向量中有任何元素包含无穷大或 NaN，则设置 found_inf_ptr 指向的值为 1.0
                if (val_vec.has_inf_nan()) {
                  *found_inf_ptr = 1.f;
                }
                # 每个线程访问 inv_scale，但它会在缓存中命中
                const auto inv_scale_val = *inv_scale_ptr;
                # 对整个向量应用缩放因子，并返回结果
                return inv_scale_val == 1.f ? val_vec : val_vec * Vectorized<scalar_t>(inv_scale_val);
              });
        });
    }
  }


注释：


    # 这里是两个嵌套的代码块的结尾
    }
  }


这段代码展示了两个嵌套的代码块的闭合：
- 第一个 `}` 结束了内部的代码块。
- 第二个 `}` 结束了外部的代码块。
每个 `}` 对应一个 `{`，用于标识代码块的开始和结束。
// _amp_update_scale_cpu updates the scale tensor in place.
//
// Args:
// current_scale: 一个包含尺度值的单元素浮点张量。
// growth_tracker: 一个包含最近连续未跳过步骤数的单元素整数张量。
// found_inf: 一个单元素浮点张量。如果 > 0，则表示在相关的 _amp_non_finite_check_and_unscale_cpu 调用中发现了无穷大或 NaN 值，如果没有发现，则为 0。
// growth_factor: 如果没有发现无穷大/NaN，则用作乘法因子的倍数（通常略大于 1）。
// backoff_factor: 如果发现了无穷大/NaN，则用作乘法因子的倍数（通常为 0.5）。
// growth_interval: 必须发生的连续未跳过步骤数，以便将 current_scale 乘以 growth_factor。
//
// Returns:
// 更新后的 current_scale 张量的引用。
at::Tensor& _amp_update_scale_cpu_kernel(
    at::Tensor& current_scale,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
  TORCH_CHECK(growth_tracker.is_cpu(), "growth_tracker must be a CPU tensor.");
  TORCH_CHECK(current_scale.is_cpu(), "current_scale must be a CPU tensor.");
  TORCH_CHECK(found_inf.is_cpu(), "found_inf must be a CPU tensor.");
  TORCH_CHECK(
      growth_tracker.numel() == 1,
      "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(
      current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(
      growth_tracker.scalar_type() == at::ScalarType::Int,
      "growth_tracker must be an int tensor.");
  TORCH_CHECK(
      current_scale.scalar_type() == at::ScalarType::Float,
      "current_scale must be a float tensor.");
  TORCH_CHECK(
      found_inf.scalar_type() == at::ScalarType::Float,
      "found_inf must be a float tensor.");

  // 获取 current_scale, growth_tracker 和 found_inf 的指针
  float* current_scale_ptr = current_scale.data_ptr<float>();
  int* growth_tracker_ptr = growth_tracker.data_ptr<int>();
  float* found_inf_ptr = found_inf.data_ptr<float>();

  // 如果发现了无穷大或 NaN
  if (*found_inf_ptr) {
    // 缩小 current_scale，并将 growth_tracker 置零
    *current_scale_ptr = (*current_scale_ptr) * backoff_factor;
    *growth_tracker_ptr = 0;
  } else {
    // 进入此分支意味着刚执行了一个成功的步骤，因此增加 growth_tracker 然后与 growth_interval 比较
    auto successful = (*growth_tracker_ptr) + 1;
    if (successful == growth_interval) {
      // 计算新的尺度值，乘以 growth_factor
      auto new_scale = static_cast<float>((*current_scale_ptr) * growth_factor);
      // 确保不将尺度增长到超过 fp32 的界限到达无穷大
      if (std::isfinite(new_scale)) {
        *current_scale_ptr = new_scale;
      }
      // 将 growth_tracker 置零
      *growth_tracker_ptr = 0;
    } else {
      // 更新 growth_tracker
      *growth_tracker_ptr = successful;
    }
  }

  // 返回更新后的 current_scale 张量的引用
  return current_scale;
}
REGISTER_DISPATCH(_amp_update_scale_cpu_stub, &_amp_update_scale_cpu_kernel);


// 注册分发函数，将 _amp_update_scale_cpu_stub 映射到 _amp_update_scale_cpu_kernel 函数上
REGISTER_DISPATCH(_amp_update_scale_cpu_stub, &_amp_update_scale_cpu_kernel);


} // namespace at::native


// 结束 at::native 命名空间的定义
} // namespace at::native
```