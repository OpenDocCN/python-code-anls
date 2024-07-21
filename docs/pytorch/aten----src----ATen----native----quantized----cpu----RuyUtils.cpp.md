# `.\pytorch\aten\src\ATen\native\quantized\cpu\RuyUtils.cpp`

```py
#ifdef USE_RUY_QMATMUL
// 如果定义了 USE_RUY_QMATMUL 宏，则包含以下代码块

#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/RuyUtils.h>

namespace at {
namespace native {
namespace ruy_utils {

// 定义了一个线程局部变量，用于存储 Ruy 的上下文
static thread_local ruy::Context context;

// 返回当前线程的 Ruy 上下文指针
ruy::Context* get_ruy_context() {
  return &context;
}

// 根据给定的缩放因子 scale，计算量化乘数的固定点表示和指数
// 参考 Ruy 的实现：https://github.com/google/ruy/blob/2d950b3bfa7ebfbe7a97ecb44b1cc4da5ac1d6f0/ruy/test.h#L1602
void quantize_multiplier(double scale,
                         int* multiplier_fixedpoint,
                         int* multiplier_exponent) {
  // 检查缩放因子 scale 必须大于 0
  TORCH_CHECK(scale > 0, "Quantization scale (", scale, ") must be positive.");
  // 将 scale 拆解为尾数和指数部分
  const double q = std::frexp(scale, multiplier_exponent);
  // 将尾数部分转换为固定点表示，并四舍五入到整数
  auto q_fixed = static_cast<std::int64_t>(std::round(q * (1ll << 31)));
  // 检查固定点表示不超过 2^31
  TORCH_CHECK(q_fixed <= (1ll << 31));
  // 如果固定点表示达到 2^31，则需要调整并增加指数部分
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*multiplier_exponent;
  }
  // 最终检查固定点表示不超过 std::int32_t 的最大值
  TORCH_CHECK(q_fixed <= std::numeric_limits<std::int32_t>::max());
  // 将固定点表示转换为 int32_t 类型，并赋值给 multiplier_fixedpoint
  *multiplier_fixedpoint = static_cast<std::int32_t>(q_fixed);
}

} // namespace ruy_utils
} // namespace native
} // namespace

#endif // USE_RUY_QMATMUL
// 结束 ifdef 宏 USE_RUY_QMATMUL 的条件编译
```