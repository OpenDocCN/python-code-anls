# `.\pytorch\aten\src\ATen\native\UnaryOps.h`

```py
#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/Generator.h>
#include <c10/core/Scalar.h>
#include <stdexcept>

// 声明命名空间，引入所需的类和结构体
namespace at {
class Tensor;
class TensorBase;
struct TensorIteratorBase;
}

// 声明在at::native命名空间内的函数指针类型
namespace at::native {

// 用于没有标量参数的一元函数的函数指针类型
using unary_fn = void(*)(TensorIteratorBase&);
// 用于带有标量参数的一元函数的函数指针类型
using unary_fn_with_scalar = void(*)(TensorIteratorBase&, const Scalar& a);

// 内联命名空间，定义一些CPU特定的函数
inline namespace CPU_CAPABILITY {
    void conj_kernel(TensorIteratorBase &iter);  // 计算共轭
    void neg_kernel(TensorIteratorBase &iter);   // 计算负数
    void reciprocal_kernel(TensorIteratorBase &iter);  // 计算倒数
    void rsqrt_kernel(TensorIteratorBase& iter);   // 计算倒数的平方根
    void sqrt_kernel(TensorIteratorBase& iter);    // 计算平方根
} // namespace CPU_CAPABILITY

// 声明分发函数的模板实例，传入相应的函数指针
DECLARE_DISPATCH(unary_fn, abs_stub);              // 绝对值
DECLARE_DISPATCH(unary_fn, angle_stub);            // 角度
DECLARE_DISPATCH(unary_fn, conj_physical_stub);    // 物理共轭
DECLARE_DISPATCH(unary_fn, acos_stub);             // 反余弦
DECLARE_DISPATCH(unary_fn, acosh_stub);            // 反双曲余弦
DECLARE_DISPATCH(unary_fn, asinh_stub);            // 反双曲正弦
DECLARE_DISPATCH(unary_fn, atanh_stub);            // 反双曲正切
DECLARE_DISPATCH(unary_fn, asin_stub);             // 反正弦
DECLARE_DISPATCH(unary_fn, atan_stub);             // 反正切
DECLARE_DISPATCH(unary_fn, bitwise_not_stub);      // 按位取反
DECLARE_DISPATCH(unary_fn, logical_not_stub);      // 逻辑非
DECLARE_DISPATCH(unary_fn, ceil_stub);             // 向上取整
DECLARE_DISPATCH(unary_fn, cos_stub);              // 余弦
DECLARE_DISPATCH(unary_fn, cosh_stub);             // 双曲余弦
DECLARE_DISPATCH(unary_fn, digamma_stub);          // Digamma函数
DECLARE_DISPATCH(unary_fn, special_entr_stub);     // 信息熵函数
DECLARE_DISPATCH(unary_fn, special_erfcx_stub);    // 调整误差函数
DECLARE_DISPATCH(unary_fn, erf_stub);              // 误差函数
DECLARE_DISPATCH(unary_fn, erfc_stub);             // 补误差函数
DECLARE_DISPATCH(unary_fn, erfinv_stub);           // 逆误差函数
DECLARE_DISPATCH(unary_fn, exp_stub);              // 指数函数
DECLARE_DISPATCH(unary_fn, exp2_stub);             // 2的指数函数
DECLARE_DISPATCH(unary_fn, expm1_stub);            // 指数函数减一
DECLARE_DISPATCH(unary_fn, floor_stub);            // 向下取整
DECLARE_DISPATCH(unary_fn, frac_stub);             // 获取小数部分
DECLARE_DISPATCH(unary_fn, frexp_stub);            // 获取浮点数的尾数和指数
DECLARE_DISPATCH(unary_fn, i0_stub);               // 修正的0阶贝塞尔函数
DECLARE_DISPATCH(unary_fn, special_i0e_stub);      // 修正的0阶指数贝塞尔函数
DECLARE_DISPATCH(unary_fn, special_i1_stub);       // 修正的1阶贝塞尔函数
DECLARE_DISPATCH(unary_fn, special_i1e_stub);      // 修正的1阶指数贝塞尔函数
DECLARE_DISPATCH(unary_fn, log_stub);              // 自然对数
DECLARE_DISPATCH(unary_fn, log10_stub);            // 10为底的对数
DECLARE_DISPATCH(unary_fn, log1p_stub);            // 自然对数(1+x)
DECLARE_DISPATCH(unary_fn, log2_stub);             // 2为底的对数
DECLARE_DISPATCH(unary_fn, special_ndtri_stub);    // 标准正态分布的逆
DECLARE_DISPATCH(unary_fn, special_log_ndtr_stub); // 对数正态分布的累积分布函数的对数
DECLARE_DISPATCH(unary_fn, neg_stub);              // 取负数

DECLARE_DISPATCH(unary_fn, reciprocal_stub);       // 取倒数
DECLARE_DISPATCH(unary_fn, round_stub);            // 四舍五入
DECLARE_DISPATCH(unary_fn, rsqrt_stub);            // 取倒数的平方根
DECLARE_DISPATCH(unary_fn, sigmoid_stub);          // Sigmoid函数
DECLARE_DISPATCH(unary_fn_with_scalar, logit_stub); // Logit函数
DECLARE_DISPATCH(unary_fn, sign_stub);             // 符号函数
DECLARE_DISPATCH(unary_fn, signbit_stub);          // 符号位
DECLARE_DISPATCH(unary_fn, sgn_stub);              // 符号函数
DECLARE_DISPATCH(unary_fn, sin_stub);              // 正弦
DECLARE_DISPATCH(unary_fn, sinc_stub);             // Sinc函数
DECLARE_DISPATCH(unary_fn, sinh_stub);             // 双曲正弦
DECLARE_DISPATCH(unary_fn, sqrt_stub);             // 平方根
DECLARE_DISPATCH(unary_fn, tan_stub);              // 正切
DECLARE_DISPATCH(unary_fn, tanh_stub);             // 双曲正切
DECLARE_DISPATCH(unary_fn, trigamma_stub);         // 三角gamma函数
DECLARE_DISPATCH(unary_fn, trunc_stub);            // 截断
DECLARE_DISPATCH(unary_fn, lgamma_stub);           // Gamma函数的对数
DECLARE_DISPATCH(unary_fn, special_airy_ai_stub);  // Airy函数的第一类
DECLARE_DISPATCH(unary_fn, special_bessel_j0_stub);// 贝塞尔函数J0
# 声明一系列分发函数，用于特殊的贝塞尔函数
DECLARE_DISPATCH(unary_fn, special_bessel_j1_stub);
DECLARE_DISPATCH(unary_fn, special_bessel_y0_stub);
DECLARE_DISPATCH(unary_fn, special_bessel_y1_stub);

# 声明一系列分发函数，用于特殊的修正贝塞尔函数
DECLARE_DISPATCH(unary_fn, special_modified_bessel_i0_stub);
DECLARE_DISPATCH(unary_fn, special_modified_bessel_i1_stub);
DECLARE_DISPATCH(unary_fn, special_modified_bessel_k0_stub);
DECLARE_DISPATCH(unary_fn, special_modified_bessel_k1_stub);

# 声明一系列分发函数，用于特殊的缩放修正贝塞尔函数
DECLARE_DISPATCH(unary_fn, special_scaled_modified_bessel_k0_stub);
DECLARE_DISPATCH(unary_fn, special_scaled_modified_bessel_k1_stub);

# 声明一系列分发函数，用于特殊的球形贝塞尔函数
DECLARE_DISPATCH(unary_fn, special_spherical_bessel_j0_stub);

# 注意：以下函数实际上在 Distribution 中定义
# 声明分发函数，用于生成伯努利分布的张量
DECLARE_DISPATCH(void(*)(const TensorBase&, const TensorBase&, std::optional<Generator>), bernoulli_tensor_stub);
# 声明分发函数，用于生成伯努利分布的标量
DECLARE_DISPATCH(void(*)(const TensorBase&, const double, std::optional<Generator>), bernoulli_scalar_stub);
# 声明分发函数，用于生成柯西分布
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, std::optional<Generator>), cauchy_stub);
# 声明分发函数，用于生成指数分布
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, std::optional<Generator>), exponential_stub);
# 声明分发函数，用于生成几何分布
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, std::optional<Generator>), geometric_stub);
# 声明分发函数，用于生成对数正态分布
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, const double, std::optional<Generator>), log_normal_stub);
# 声明分发函数，用于生成均匀分布
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, const double, std::optional<Generator>), uniform_stub);
# 声明分发函数，用于生成正态分布
DECLARE_DISPATCH(void(*)(const TensorBase&, const double, const double, std::optional<Generator>), normal_stub);
# 声明分发函数，用于生成指定范围内的随机整数
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const uint64_t, const int64_t, std::optional<Generator>), random_from_to_stub);
# 声明分发函数，用于生成全范围的64位随机数
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, std::optional<Generator>), random_full_64_bits_range_stub);
# 声明分发函数，用于生成随机数
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, std::optional<Generator>), random_stub);

# 声明分发函数，用于生成凯塞窗口
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const int64_t, const double), kaiser_window_stub);
# 声明分发函数，用于生成多次对数
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const int64_t), polygamma_stub);
# 声明分发函数，用于对张量执行夹紧操作
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const Scalar& a, const Scalar& b), clamp_stub);
# 声明分发函数，用于多项式分布的抽样
DECLARE_DISPATCH(void (*)(Tensor&, const Tensor&, int64_t, std::optional<Generator>), multinomial_with_replacement_stub);
# 声明分发函数，用于将 NaN 替换为数字
DECLARE_DISPATCH(void (*)(TensorIteratorBase&, std::optional<double>, std::optional<double>, std::optional<double>), nan_to_num_stub);
# 声明分发函数，用于四舍五入数字
DECLARE_DISPATCH(void (*)(TensorIteratorBase&, int64_t), round_decimals_stub);

# 命名空间结束注释
} // namespace at::native
```