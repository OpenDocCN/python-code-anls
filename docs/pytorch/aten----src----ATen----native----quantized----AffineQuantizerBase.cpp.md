# `.\pytorch\aten\src\ATen\native\quantized\AffineQuantizerBase.cpp`

```
#include <ATen/native/quantized/AffineQuantizerBase.h>
#include <c10/util/irange.h>
#include <cfenv>
#include <climits>

#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace at {
namespace native {

namespace {

// 匿名命名空间，用于定义局部函数或变量，限制其作用域在当前文件内

template <typename T>
void checkZeroPoint(const std::string& fn_name, int64_t zero_point) {
  // 检查零点是否在类型 T 的表示范围内
  TORCH_CHECK(
      zero_point <= std::numeric_limits<T>::max(),
      fn_name,
      " zero_point ",
      zero_point,
      " is out of range.");
  TORCH_CHECK(
      zero_point >= std::numeric_limits<T>::min(),
      fn_name,
      " zero_point ",
      zero_point,
      " is out of range.");
}

} // anonymous namespace

#ifdef USE_FBGEMM

// Note: quantize_val is only explicitly used in test outside of this file
// 根据指定的缩放因子、零点和值，量化浮点数值为整数类型 T
template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // 内部使用 fbgemm::Quantize 函数进行量化，使用 std::nearbyint 近似为最接近的整数
  // 这种方式通常在处理器架构如 x86 和 ARM 上的 SIMD 实现中更快速
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int32_t qvalue;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  qvalue = fbgemm::Quantize<typename T::underlying, false /*LEGACY*/>(
      value,
      static_cast<int32_t>(zero_point),
      static_cast<float>(scale),
      /*result_precision=*/CHAR_BIT * sizeof(typename T::underlying));
  return static_cast<T>(qvalue);
}

// 根据指定的缩放因子、零点和精度，量化浮点数组成的向量到整数类型 T 的向量
template <typename T, int precision>
void quantize_vec(
    double scale,
    int64_t zero_point,
    const float* src,
    T* dst,
    size_t count) {
  fbgemm::Quantize<typename T::underlying, false /*LEGACY*/>(
      src,
      (typename T::underlying*)dst,
      count,
      fbgemm::TensorQuantizationParams{
          (float)scale, (int32_t)zero_point, precision});
}

#if defined(__ARM_NEON__) || defined(__aarch64__)
// 在编译时支持 aarch64，但仍然通过 simde 支持 x86 指令集时使用
// 根据指定的缩放因子、零点和值，量化浮点数值为整数类型 T（适用于 ARM 架构）
template <typename T>
T quantize_val_arm(
    const float scale,
    const int32_t zero_point,
    const float value) {
  constexpr int32_t qmin = std::numeric_limits<T>::min();
  constexpr int32_t qmax = std::numeric_limits<T>::max();
  float inv_scale = 1.0f / scale;
  auto r = zero_point + static_cast<int32_t>(std::nearbyint(value * inv_scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<T>(r);
}

// 显式实例化模板函数 quantize_val_arm 以便在编译时生成具体的类型实现
template uint8_t quantize_val_arm<uint8_t>(
    const float scale,
    const int32_t zero_point,
    const float value);
template int8_t quantize_val_arm<int8_t>(
    const float scale,
    const int32_t zero_point,
    const float value);
#endif // defined(__ARM_NEON__) || defined(__aarch64__)

#endif // USE_FBGEMM

} // namespace native
} // namespace at
    // 定义一个名为 zero_point 的常量整数型参数，表示数据的零点
    const int32_t zero_point,
    // 定义一个名为 value 的常量浮点数型参数，表示要处理的值
    const float value);
    // 函数声明结束，接受一个整数型的零点参数和一个浮点数的值参数
#else

template <typename T>
inline float dequantize_val(double scale, int64_t zero_point, T value) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 创建一个 TensorQuantizationParams 对象 qparams，设置其 scale 和 zero_point 属性
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = static_cast<float>(scale);  // 将 double 类型的 scale 转换为 float
  qparams.zero_point = static_cast<int32_t>(zero_point);  // 将 int64_t 类型的 zero_point 转换为 int32_t
  // 使用 fbgemm 库中的 Dequantize 函数将 value 进行反量化，返回浮点数结果
  return fbgemm::Dequantize<typename T::underlying>(value.val_, qparams);
}
#else // USE_FBGEMM

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  // 在 Android 环境中定义的浮点数近似取整函数，返回最接近 x 的整数
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  // 在非 Android 环境中定义的浮点数近似取整函数，返回最接近 x 的整数
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  // 使用 std::nearbyint 近似取整函数，返回最接近 x 的整数
  return std::nearbyint(x);
}
#endif

template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // 使用 std::nearbyint 近似取整函数，根据 scale 和 zero_point 将浮点数 value 量化为整数类型 T
  int64_t qvalue;
  // 获取 T 类型的整数的最小值和最大值
  constexpr int64_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int64_t qmax = std::numeric_limits<typename T::underlying>::max();
  float inv_scale = 1.0f / static_cast<float>(scale);  // 计算 scale 的倒数
  // 计算量化后的整数值 qvalue，确保在指定的最小和最大值范围内
  qvalue = static_cast<int64_t>(zero_point + Round(value * inv_scale));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  // 将 qvalue 转换为类型 T 并返回
  return static_cast<T>(qvalue);
}

template <typename T>
T quantize_val_arm(
    const float scale,
    const int32_t zero_point,
    const float value) {
  // 在 ARM 环境下进行量化操作，与 x86 环境略有不同
  constexpr int32_t qmin = std::numeric_limits<T>::min();
  constexpr int32_t qmax = std::numeric_limits<T>::max();
  float inv_scale = 1.0f / scale;  // 计算 scale 的倒数
#ifndef _MSC_VER
  auto r = static_cast<int32_t>(Round(value * inv_scale));  // 使用 Round 函数取整
  // 检查是否发生了加法溢出，如果发生，则将 r 设置为 qmax
  if (__builtin_add_overflow(zero_point, r, &r)) {
    // 在量化过程中，zero_point 必须是介于 qmin 和 qmax 之间的非负值，因此只有溢出的可能性
    r = qmax;
  }
#else
  auto r = zero_point + static_cast<int32_t>(Round(value * inv_scale));  // 使用 Round 函数取整
#endif
  r = std::max(r, qmin);  // 确保 r 不小于 qmin
  r = std::min(r, qmax);  // 确保 r 不大于 qmax
  // 将 r 转换为类型 T 并返回
  return static_cast<T>(r);
}

template <typename T, int precision>
void quantize_vec(
    double scale,
    int64_t zero_point,
    const float* src,
    T* dst,
    size_t count) {
  // 检查 zero_point 是否合法
  checkZeroPoint<typename T::underlying>("quantize_vec", zero_point);
  // 遍历每个元素，将浮点数数组 src 中的每个元素量化为类型 T，并存储到 dst 数组中
  for (const auto i : c10::irange(count)) {
    dst[i] = quantize_val<T>(scale, zero_point, src[i]);
  }
}

template uint8_t quantize_val_arm<uint8_t>(
    const float scale,
    const int32_t zero_point,
    const float value);
template int8_t quantize_val_arm<int8_t>(
    const float scale,
    const int32_t zero_point,
    const float value);
#endif
    const float scale,
    const int32_t zero_point,
    const float value);


// 声明函数参数：scale 为浮点数，表示缩放因子
// zero_point 为 32 位整数，表示零点偏移量
// value 为浮点数，表示具体的数值
/*
 * 用于将量化值反量化的函数。
 * 根据给定的缩放因子、零点和输入值，返回反量化后的浮点数值。
 */
template <typename T>
TORCH_API float dequantize_val(double scale, int64_t zero_point, T value) {
  return static_cast<float>(scale) * (value.val_ - static_cast<int32_t>(zero_point));
}
#endif // USE_FBGEMM

/*
 * 基于以下方程对值进行量化：
 * Xq = Round(Xf * inv_scale + zero_point)
 * 这里的 zero_point 是浮点数。
 *
 * 注意：对于嵌入量化的情况，我们将设置 zero_point 为 (-Xmin/scale)，其中 Xmin 是输入张量行中的最小值。
 */
int quantize_val_float_qparams(float scale, float zero_point, float value, int qmin, int qmax) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int qvalue;

  float inv_scale = scale == 0 ? 1.0f : 1.0f / scale;
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  qvalue = lrintf(value * inv_scale + zero_point);
  qvalue = std::max(qmin, std::min(qvalue, qmax));
  return qvalue;
}

/*
 * 重新量化值的模板函数。
 * 根据源缩放因子、源零点、目标缩放因子和目标零点，返回重新量化后的目标类型值。
 */
template <typename SRC_T, typename DST_T>
DST_T requantize_val(
    double src_scale,
    int64_t src_zero_point,
    double dst_scale,
    int64_t dst_zero_point,
    SRC_T src) {
  const auto dq = dequantize_val<SRC_T>(src_scale, src_zero_point, src);
  return quantize_val<DST_T>(dst_scale, dst_zero_point, dq);
}

/*
 * 从整数重新量化到目标类型的模板函数。
 * 根据给定的乘数、零点和输入整数值，返回重新量化后的目标类型值。
 */
template <typename DST_T>
DST_T requantize_from_int(double multiplier, int64_t zero_point, int64_t src) {
  int64_t quantize_down =
      zero_point + lrintf(src * static_cast<float>(multiplier));
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  int32_t min = std::numeric_limits<typename DST_T::underlying>::min();
  int32_t max = std::numeric_limits<typename DST_T::underlying>::max();
  return static_cast<DST_T>(
      std::min<int64_t>(std::max<int64_t>(quantize_down, min), max));
}

/*
 * 以下是特化的模板实例化，用于不同的量化类型和函数的调用。
 */
template TORCH_API qint8
quantize_val<qint8>(double scale, int64_t zero_point, float value);
template TORCH_API quint8
quantize_val<quint8>(double scale, int64_t zero_point, float value);
template TORCH_API qint32
quantize_val<qint32>(double scale, int64_t zero_point, float value);
template TORCH_API void quantize_vec<c10::qint8>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::qint8* dst,
    size_t count);
template TORCH_API void quantize_vec<c10::quint8>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::quint8* dst,
    size_t count);
template TORCH_API void quantize_vec<c10::qint32, 32>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::qint32* dst,
    size_t count);

template TORCH_API float dequantize_val<qint8>(
    double scale,
    int64_t zero_point,
    qint8 value);
template TORCH_API float dequantize_val<quint8>(
    double scale,
    int64_t zero_point,
    quint8 value);
template TORCH_API float dequantize_val<qint32>(
    double scale,
    int64_t zero_point,
    qint32 value);

template TORCH_API qint8
requantize_val<qint8, qint8>(double, int64_t, double, int64_t, qint8);
template TORCH_API quint8
requantize_val<qint8, quint8>(double, int64_t, double, int64_t, qint8);
# 定义模板函数：将输入值重新量化为指定的输出整数类型 qint32
template TORCH_API qint32
requantize_val<qint8, qint32>(double scale_in, int64_t zero_point_in, double scale_out, int64_t zero_point_out, qint8 input);

# 定义模板函数：将输入值重新量化为指定的输出整数类型 qint8
template TORCH_API qint8
requantize_val<quint8, qint8>(double scale_in, int64_t zero_point_in, double scale_out, int64_t zero_point_out, quint8 input);

# 定义模板函数：将输入值重新量化为指定的输出整数类型 quint8
template TORCH_API quint8
requantize_val<quint8, quint8>(double scale_in, int64_t zero_point_in, double scale_out, int64_t zero_point_out, quint8 input);

# 定义模板函数：将输入值重新量化为指定的输出整数类型 qint32
template TORCH_API qint32
requantize_val<quint8, qint32>(double scale_in, int64_t zero_point_in, double scale_out, int64_t zero_point_out, quint8 input);

# 定义模板函数：将输入值重新量化为指定的输出整数类型 qint8
template TORCH_API qint8
requantize_val<qint32, qint8>(double scale_in, int64_t zero_point_in, double scale_out, int64_t zero_point_out, qint32 input);

# 定义模板函数：将输入值重新量化为指定的输出整数类型 quint8
template TORCH_API quint8
requantize_val<qint32, quint8>(double scale_in, int64_t zero_point_in, double scale_out, int64_t zero_point_out, qint32 input);

# 定义模板函数：将输入值重新量化为指定的输出整数类型 qint32
template TORCH_API qint32
requantize_val<qint32, qint32>(double scale_in, int64_t zero_point_in, double scale_out, int64_t zero_point_out, qint32 input);

# 定义模板函数：从整数值重新量化为 qint8 类型
template TORCH_API qint8 requantize_from_int<qint8>(double scale, int64_t zero_point, int64_t input);

# 定义模板函数：从整数值重新量化为 quint8 类型
template TORCH_API quint8
requantize_from_int<quint8>(double scale, int64_t zero_point, int64_t input);

# 定义模板函数：从整数值重新量化为 qint32 类型
template TORCH_API qint32
requantize_from_int<qint32>(double scale, int64_t zero_point, int64_t input);

} // namespace native
} // namespace at
```