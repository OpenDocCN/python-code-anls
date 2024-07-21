# `.\pytorch\aten\src\ATen\native\cpu\UpSampleKernel.cpp`

```py
// 定义预处理器宏，用于仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量类相关的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
// 包含 AVX 反锯齿插值的头文件
#include <ATen/native/cpu/UpSampleKernelAVXAntialias.h>

#ifndef AT_PER_OPERATOR_HEADERS
// 如果未定义每个操作符的头文件，则包含函数相关头文件
#include <ATen/Functions.h>
#else
// 如果定义了每个操作符的头文件，则单独包含空操作、空原生和全1操作的头文件
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/ones.h>
#endif

namespace at::native {
namespace {

// 使用 std::vector 定义 scale_t 类型，其元素为可选的 double 类型
using scale_t = std::vector<std::optional<double>>;

// TODO: 该文件可以从以下全局重命名其函数和类名，并添加更多注释。特别是：
// - 尽管它们的名称（以及文件名）如此，所有这些内核并不仅仅是上采样：它们是一般的插值，即它们也支持下采样。
// - "horizontal"、"within dims" 或 "contiguous dim" 术语指的是最后一个维度。
//   这不仅适用于2D图像，还适用于3D（和1D？？）输入。类似地，"vertical" 或 "across dims" 指的是除了最后一个维度之外的所有维度。
//   在其他内核中，这些也称为 "zero-stride" 和 "non-zero-stride"，我们应该统一所有这些术语。
// - "zero-stride" 和 "non-zero strides" 术语指的是权重和索引，而不是输入或输出的连续性。
// - 哪些内核是矢量化的，哪些不是，有时并不清楚。
// - 像 _use_vectorized_kernel_cond() 这样的函数应该重命名，并更新其描述，因为它们不是代码路径中唯一的 "分支"，
//   在那里我们可以选择矢量化内核与非矢量化内核。例如，看看 upsample_bilinear2d_kernel_impl()，我们在 _use_vectorized_kernel_cond() 之前已经进行了类似的检查。
// - 不清楚哪些代码属于 "separable interpolation" 的代码路径。
// - 一些名称需要更具体。例如 "cpu_upsample_generic_aa()" 看起来像一个超级通用的名称，但该函数实际上是相当具体的 - 我们需要更清楚地表达这一点。
// - 一些函数具有 "aa" 后缀，但并不意味着它们只支持抗锯齿。现在有些还支持 antialias=False。
// - 各种注释已经过时。例如，关于 `Interpolate` 结构体下面的注释：cpu_upsample_linear 现在已经不存在了，这些结构体用于各种模式，而不仅仅是线性插值。
// - 文档化插值如何工作以及特别指出：
//   - 沿着给定维度的权重和索引对于所有像素都是相同的（因此预先计算它们的好处）
//   - 它可以是 "separated" 的，即我们可以进行水平通道和
// Helper structs and methods for cpu_upsample_linear
//
// Interpolation methods that are used below are separable, allowing computation of interpolation
// independently per dimension in a recursive manner. For more context, refer to issue #10482.
//
// Interpolation structure to compute output value in n-dimensional case.
// - `eval` recursively computes interpolated output for each dimension
// - Relies on compiler optimizations for automatic factorization and vectorization using SSE and AVX2
template <int n, typename scalar_t, typename opmath_t, typename index_t, int interp_size>
struct Interpolate {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      // Fetch index and weight for the current dimension
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      opmath_t wts = *(scalar_t*)&data[1][i * strides[1]];
      // Recursively compute interpolated value
      opmath_t t = Interpolate<n - 1, scalar_t, opmath_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
      // Compute final output for this dimension
      opmath_t output = t * wts;
      // Loop through remaining dimensions to compute the final output
      for (const auto j : c10::irange(1, interp_size)) {
        ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
        wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
        t = Interpolate<n - 1, scalar_t, opmath_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
        output += t * wts;
      }
      return output;
  }
};

// Specialization for n = 1 (base case)
template <typename scalar_t, typename opmath_t, typename index_t, int interp_size>
struct Interpolate<1, scalar_t, opmath_t, index_t, interp_size> {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      // Fetch index and weight for the single dimension
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      opmath_t wts = *(scalar_t*)&data[1][i * strides[1]];
      // Read value from source and compute output
      opmath_t t = *(scalar_t *)&src[ids];
      opmath_t output = t * wts;
      // Loop through remaining interpolation points in the single dimension
      for (const auto j : c10::irange(1, interp_size)) {
        ids = *(index_t*)&data[2 * j + 0][i * strides[2 * j + 0]];
        wts = *(scalar_t*)&data[2 * j + 1][i * strides[2 * j + 1]];
        t = *(scalar_t *)&src[ids];
        output += t * wts;
      }
      return output;
    }
};

// Specialization for interp_size = 1 (base case of interpolation points)
template <int n, typename scalar_t, typename opmath_t, typename index_t>
struct Interpolate<n, scalar_t, opmath_t, index_t, 1> {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      // Fetch index for the current dimension and recursively compute output
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      return Interpolate<n - 1, scalar_t, opmath_t, index_t, 1>::eval(src + ids, &data[2], &strides[2], i);
  }
};

// Specialization for n = 1 and interp_size = 1 (both base cases)
template <typename scalar_t, typename opmath_t, typename index_t>
struct Interpolate<1, scalar_t, opmath_t, index_t, 1> {
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
      // Fetch index for the single dimension and directly return the interpolated value
      index_t ids = *(index_t*)&data[0][i * strides[0]];
      return *(scalar_t *)&src[ids];
    }
};
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        // 将数据数组的第一个元素的偏移量乘以步长得到索引值
        index_t ids = *(index_t*)&data[0][i * strides[0]];
        // 根据索引值在源字符数组中获取相应的标量值并转换成 opmath_t 类型返回
        return *(scalar_t *)&src[ids];
    }
};

// 以下是一个注释段落，用于说明三维线性插值的问题和解决方案
// 在 channels_first 布局下，使用 upsample_trilinear3d 会导致意外的两倍速度减慢，
// 无论是使用1个线程还是6个线程。我们需要针对这种情况进行特殊处理，如下所示：
// 一旦问题解决，我们可以保留通用实现并删除以下内容：
// struct Interpolate<n, scalar_t, opmath_t, index_t, 2> 和
// struct Interpolate<1, scalar_t, opmath_t, index_t, 2>

template <int n, typename scalar_t, typename opmath_t, typename index_t>
struct Interpolate<n, scalar_t, opmath_t, index_t, 2> {
    // 计算插值结果的静态方法，使用于特定的插值尺寸为2的情况
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        // 从数据数组中读取索引值和权重
        index_t i0 = *(index_t*)&data[0][i * strides[0]];
        index_t i1 = *(index_t*)&data[2][i * strides[2]];
        opmath_t w0 = *(scalar_t *)&data[1][i * strides[1]];
        opmath_t w1 = *(scalar_t *)&data[3][i * strides[3]];

        // 递归调用以获取两个插值点的插值结果
        opmath_t t0 = Interpolate<n - 1, scalar_t, opmath_t, index_t, 2>::eval(src + i0, &data[4], &strides[4], i);
        opmath_t t1 = Interpolate<n - 1, scalar_t, opmath_t, index_t, 2>::eval(src + i1, &data[4], &strides[4], i);

        // 返回加权后的插值结果
        return t0 * w0 + t1 * w1;
    }
};

// 对于插值尺寸为2时的 n=1 的特化版本
template <typename scalar_t, typename opmath_t, typename index_t>
struct Interpolate<1, scalar_t, opmath_t, index_t, 2> {
    // 计算插值结果的静态方法
    static inline opmath_t eval(char* src, char** data, const int64_t* strides, int64_t i) {
        // 从数据数组中读取索引值和权重
        index_t i0 = *(index_t*)&data[0][i * strides[0]];
        index_t i1 = *(index_t*)&data[2][i * strides[2]];
        opmath_t w0 = *(scalar_t *)&data[1][i * strides[1]];
        opmath_t w1 = *(scalar_t *)&data[3][i * strides[3]];

        // 直接从源数据中读取插值点的值并进行加权求和
        opmath_t t0 = *(scalar_t *)&src[i0];
        opmath_t t1 = *(scalar_t *)&src[i1];
        return t0 * w0 + t1 * w1;
    }
};

// 插值的通用函数模板，根据插值维度 n 和插值尺寸 interp_size 调用对应的 Interpolate 结构体
template <int n, typename scalar_t, typename index_t, int interp_size>
static inline scalar_t interpolate(char* src, char** data, const int64_t* strides, int64_t i) {
  using opmath_t = at::opmath_type<scalar_t>;
  return Interpolate<n, scalar_t, opmath_t, index_t, interp_size>::eval(src, data, strides, i);
}

// 零步长情况下的单维度反走样插值函数
template <typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa_single_dim_zero_strides(
    char* src,
    char** data,
    const index_t ids_stride) {
  // 从数据中读取最小索引和尺寸
  const index_t ids_min = *(index_t*)&data[0][0];
  const index_t ids_size = *(index_t*)&data[1][0];

  // 计算最小源数据指针
  char* src_min = src + ids_min;

  // 读取第一个插值点的值和权重
  scalar_t t = *(scalar_t*)&src_min[0];
  index_t wts_idx = *(index_t*)&data[4][0];
  scalar_t* wts_ptr = (scalar_t*)&data[3][wts_idx];
  scalar_t wts = wts_ptr[0];

  // 计算初始的加权插值结果
  scalar_t output = t * wts;

  // 遍历剩余的插值点，读取值并加权求和
  for (const auto j : c10::irange(1, ids_size)) {
    wts = wts_ptr[j];
    t = *(scalar_t*)&src_min[j * ids_stride];
    output += t * wts;
  }
  return output;
}

// 带有步长的单维度反走样插值函数
template <typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa_single_dim(
    char* src,
    char** data,
    const int64_t* strides,
    int64_t i,
    // 从data数组中读取索引和大小信息
    index_t ids_min = *(index_t*)&data[0][i * strides[0]];
    // 从data数组中读取ids_size
    index_t ids_size = *(index_t*)&data[1][i * strides[1]];
    
    // 计算src_min的地址，基于ids_min的偏移量
    char* src_min = src + ids_min;
    
    // 从src_min开始读取第一个scalar_t类型的数据
    scalar_t t = *(scalar_t*)&src_min[0];
    
    // 从data数组中读取第四个元素，根据wts_idx的索引位置读取wts_ptr指向的scalar_t类型的数据
    index_t wts_idx = *(index_t*)&data[4][i * strides[4]];
    scalar_t* wts_ptr = (scalar_t*)&data[3][wts_idx];
    
    // 从wts_ptr中读取第一个权重值
    scalar_t wts = wts_ptr[0];
    
    // 计算第一个输出值，即t乘以wts
    scalar_t output = t * wts;
    
    // 循环计算从1到ids_size-1的输出值
    for (const auto j : c10::irange(1, ids_size)) {
        // 从src_min中读取偏移量为j * ids_stride的scalar_t类型的数据
        t = *(scalar_t*)&src_min[j * ids_stride];
        // 从wts_ptr中读取第j个权重值
        wts = wts_ptr[j];
        // 累加t乘以wts到输出值中
        output += t * wts;
    }
    
    // 返回最终的输出值
    return output;
}

// Template function to check if the stride is all zero for a given number of dimensions `m`.
// It iterates through the strides array and checks if all strides except the first one are zero.
template<int m>
static inline bool is_zero_stride(const int64_t* strides) {
  // Initialize output with the check for the first stride being zero
  bool output = strides[0] == 0;
  // Iterate through the remaining strides (from 1 to m-1) and perform bitwise AND with output
  for (const auto i : c10::irange(1, m)) {
    output &= (strides[i] == 0);
  }
  return output;
}

// Template function to check if the strides correspond to contiguous memory access for interpolated dimensions.
// It verifies if each pair of strides matches the expected values (sizeof(index_t) and sizeof(scalar_t)).
template <typename scalar_t, typename index_t, int interp_size>
static inline bool is_contiguous_stride(const int64_t* strides) {
  // Initialize output with the check for the first two strides
  bool output = (strides[0] == sizeof(index_t)) && (strides[1] == sizeof(scalar_t));
  // Iterate through the remaining strides (from 2 to 2 * interp_size - 1) in steps of 2 and perform bitwise AND with output
  for (int i=2; i<2 * interp_size; i+=2) {
    output &= (strides[i] == sizeof(index_t)) && (strides[i + 1] == sizeof(scalar_t));
  }
  return output;
}

// Helper class to recursively check if all input strides corresponding to interpolated dimensions
// are equal zero except on a single dimension.
//
// Inputs: array of strides of size N, non_zero_stride_dim which can be -1, 0, 1, 2, ...
//   if non_zero_stride_dim, we check that all strides are equal zero, otherwise
//   4 strides corresponding to the strides for index_0, weight_0, index_1 and weight_1 for non_zero_stride_dim
//   dimension should be non zero.
//
// Unit check of the recursion is to verify whether 4 strides for one interpolated dimension are either zero,
// see method is_zero_stride, or (sizeof(index_t), sizeof(scalar_t), sizeof(index_t), sizeof(scalar_t)), see
// method is_contiguous_stride.
//
// In practice, we have the following cases:
// - for ND, float32, channel first, strides are
//         dimN-1,              dim1,           dim0
//         i0, w0, i1, w1, ..., i0, w0, i1, w1, i0, w0, i1, w1
// strides=(0,  0,  0,  0, ...,  0,  0,  0,  0,  4,  4,  4,  4)
//
// if size dim0 is 1 then its strides are 0 and dim1 strides are equal 4
//
// - for ND, float32, channel last, strides are
//         dimN-1,         dimN-2,             dim0
//         i0, w0, i1, w1, i0, w0, i1, w1, ... i0, w0, i1, w1
// strides=(0,  0,  0,  0,  0,  0,  0,  0, ..., 0,  0,  0,  0)
//
// Using these methods we can hint the compiler to factorize constant indices and weights
// in cpu_upsample_linear method
template <int N, int non_zero_stride_dim, typename scalar_t, typename index_t, int interp_size>
struct CheckAlmostAllZeroStrides {
  // Recursive function to evaluate the strides array up to dimension N
  static inline bool eval(const int64_t* strides) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool output;
    // Check if the current dimension matches non_zero_stride_dim for contiguous stride check
    if (N == non_zero_stride_dim) {
      output = is_contiguous_stride<scalar_t, index_t, interp_size>(strides);
    } else {
      output = is_zero_stride<2 * interp_size>(strides);
    }
    // Recursively call eval for the next lower dimension (N-1)
    return output &&
      CheckAlmostAllZeroStrides<N - 1, non_zero_stride_dim, scalar_t, index_t, interp_size>::eval(
        &strides[2 * interp_size]);
  }
};

// Specialization of CheckAlmostAllZeroStrides for dimension 0.
// This serves as the base case for the recursion, always returning true.
template <int non_zero_stride_dim, typename scalar_t, typename index_t, int interp_size>
struct CheckAlmostAllZeroStrides<0, non_zero_stride_dim, scalar_t, index_t, interp_size> {
  static inline bool eval(const int64_t* /*strides*/) {
    return true;
  }
};
// 检查几乎所有零步长的辅助方法，根据给定的类型和参数，调用模板方法来评估
template <int n, int s, typename scalar_t, typename index_t, int interp_size>
static inline bool check_almost_all_zero_stride(const int64_t* strides) {
  return CheckAlmostAllZeroStrides<n, s, scalar_t, index_t, interp_size>::eval(strides);
}

// 基础循环的辅助方法，用于计算最近、线性和立方插值模式
template <typename scalar_t, typename index_t, int out_ndims, int interp_size>
static inline void basic_loop(char** data, const int64_t* strides, int64_t n) {
  char* dst = data[0];  // 目标数据指针
  char* src = data[1];  // 源数据指针
  for (const auto i : c10::irange(n)) {
    // 调用插值方法，根据给定的维度数、标量类型、索引类型和插值尺寸进行插值计算
    *(scalar_t*)&dst[i * strides[0]] = interpolate<out_ndims, scalar_t, index_t, interp_size>(
        src + i * strides[1], &data[2], &strides[2], i);
  }
}

// 垂直方向的基础循环方法，特化为uint8_t类型
template <>
inline void basic_loop_aa_vertical<uint8_t>(
    char** data,
    const int64_t* strides,
    int64_t n,
    unsigned int weights_precision) {
  // "uint8_t和乘法技巧的权重计算"注解参见说明[ Weights computation for uint8_t and multiplication trick ]
  char* dst = data[0];  // 目标数据指针
  char* src = data[1];  // 源数据指针

  // 给定维度的索引步长是常量
  const int64_t ids_stride = *(int64_t*)&data[2 + 2][0];
  const int64_t ids_size = *(int64_t*)&data[2 + 1][0];
  const int64_t ids_min = *(int64_t*)&data[2 + 0][0];

  int64_t i = 0;

  for (; i < n; i++) {
    // 计算源数据的最小值指针
    char* src_min = src + i * strides[1] + ids_min;

    // 读取源数据的值，并根据权重索引进行调整
    uint8_t t = *(uint8_t*)&src_min[0];
    int64_t wts_idx = *(int64_t*)&data[2 + 4][0];
    int16_t* wts_ptr = (int16_t*)&data[2 + 3][wts_idx];
    int16_t wts = wts_ptr[0];

    // 中间计算使用整数类型
    int output = 1 << (weights_precision - 1);  // 考虑+0.5部分
    output += t * wts;
    for (const auto j : c10::irange(1, ids_size)) {
      wts = wts_ptr[j];
      t = *(uint8_t*)&src_min[j * ids_stride];
      output += t * wts;
    }
    // 对输出进行范围限制，然后存储到目标数据中
    *(uint8_t*)&dst[i * strides[0]] = (uint8_t)std::clamp(output >> weights_precision, 0, 255);
  }
}

// 水平方向的基础循环方法，特化为指定的标量类型
template <typename scalar_t>
static inline void basic_loop_aa_horizontal(
    char** data,
    const int64_t* strides,
    int64_t n,
    unsigned int weights_precision) {
  char* dst = data[0];  // 目标数据指针
  char* src = data[1];  // 源数据指针
  // 给定维度的索引步长是常量
  const int64_t ids_stride = *(int64_t*)&data[2 + 2][0];

  // 如果源数据的步长为零
  if (strides[1] == 0) {
    for (const auto i : c10::irange(n)) {
      // 调用单维度插值方法，根据标量类型进行插值计算
      *(scalar_t*)&dst[i * strides[0]] =
          interpolate_aa_single_dim<scalar_t, int64_t>(
              src, &data[2], &strides[2], i, ids_stride);
    }
  } else {
    // 使用 C++11 范围循环遍历 0 到 n-1 的整数序列
    for (const auto i : c10::irange(n)) {
      // 计算目标数组中的偏移量，并强制转换为 scalar_t 类型指针，将插值结果写入目标数组
      *(scalar_t*)&dst[i * strides[0]] =
          // 调用插值函数 interpolate_aa_single_dim，传入源数组指针、数据和步长数组的偏移指针、当前迭代索引 i、以及 ids_stride
          interpolate_aa_single_dim<scalar_t, int64_t>(
              src + i * strides[1], &data[2], &strides[2], i, ids_stride);
    }
  }
}

template <>
inline void basic_loop_aa_horizontal<uint8_t>(
    char** data,
    const int64_t* strides,
    int64_t n,
    unsigned int weights_precision) {
  // See Note [ Weights computation for uint8_t and multiplication trick ]
  // 获取输入数据和输出数据的指针
  char* dst = data[0];  // 输出数据指针
  char* src = data[1];  // 输入数据指针
  // 获取在给定维度上的索引步长（偏移量）
  const int64_t ids_stride = *(int64_t*)&data[2 + 2][0];

  int64_t i = 0;

  // Here we are implementing data interpolation within the same line (vs between the lines)
  // output[x, y] = input[xmin[x], y] * W[x] + input[xmin[x] + 1, y] * W[x + 1] + ... + input[xmin[x] + xsize, y] * W[x + xsize]

  // 循环遍历每个元素
  for (; i<n; i++) {

    // 获取最小索引和大小
    int64_t ids_min = *(int64_t*)&data[2 + 0][i * strides[2 + 0]];
    int64_t ids_size = *(int64_t*)&data[2 + 1][i * strides[2 + 1]];

    // 计算源数据的起始指针
    char* src_min = src + i * strides[1] + ids_min;

    // 从源数据中读取一个无符号8位整数
    uint8_t t = *(uint8_t*)&src_min[0];

    // 获取权重索引并从权重数据中读取权重值
    int64_t wts_idx = *(int64_t*)&data[2 + 4][i * strides[2 + 4]];
    int16_t* wts_ptr = (int16_t*)&data[2 + 3][wts_idx];
    int16_t wts = wts_ptr[0];

    // 中间计算使用整数类型
    int output = 1 << (weights_precision - 1);  // accounts for the +0.5 part
    output += t * wts;

    // 对于每个元素进行插值计算
    for (const auto j : c10::irange(1, ids_size)) {
      wts = wts_ptr[j];
      t = *(uint8_t*)&src_min[j * ids_stride];
      output += t * wts;
    }

    // 将计算结果写入目标数据，进行范围限制和类型转换
    *(uint8_t*)&dst[i * strides[0]] = (uint8_t)std::clamp(output >> weights_precision, 0, 255);
  }
}

// Generic upsampling computation method using TensorIterator for Nd case.
// Supports: nearest, linear, cubic modes with interp_size template argument: 1, 2, 4
//
// Single loop function for 1d, 2d and 3d cases and modes
// For N dimensions, output value up to Di dimension can be computed as
//
// output_i[a] = interpolate(output_{i+1}[a], w_{i+1}[a], output_{i+1}[a+1], w_{i+1}[a+1], ...)
// with
// output_DN[a] = interpolate(input_DN[a], w_DN[a], input_DN[a+1], w_DN[a+1], ...)
// and i - dimension index and a - linear index for spatial coordinates
//
// The recursive call is implemented with InterpLinear struct using template for
// the loop unrolling on compile time.
template <typename scalar_t, int out_ndims, int interp_size>
void cpu_upsample_generic(at::TensorIterator& iter)
{
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    // special-cases to let the compiler apply compile-time input-specific optimizations
    if ((strides[0] == sizeof(scalar_t) && (strides[1] == 0) &&
        // NOLINTNEXTLINE(bugprone-branch-clone)
        check_almost_all_zero_stride<out_ndims, 1, scalar_t, int64_t, interp_size>(&strides[2]))) {
      // contiguous channels-first case
      // 调用基础循环函数进行水平插值
      basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
    } else if ((strides[0] == sizeof(scalar_t) && (strides[1] == sizeof(scalar_t)) &&
               check_almost_all_zero_stride<out_ndims, -1, scalar_t, int64_t, interp_size>(&strides[2]))) {
      // 如果满足条件：第一个维度步长等于标量类型大小，并且第二个维度步长也等于标量类型大小，
      // 且除去前两个维度外的所有维度步长几乎都为零，那么进入这个分支，处理连续的通道末尾情况
      basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
    } else {
      // 否则，执行回退操作
      basic_loop<scalar_t, int64_t, out_ndims, interp_size>(data, strides, n);
    }
  };
  // 使用迭代器对每个元素执行循环处理函数 loop
  iter.for_each(loop);
}



// 结束 CPU 上采样最近邻操作的函数定义
template <typename scalar_t, typename scale_type, nearest_idx_fn_t nearest_idx_fn>
void cpu_upsample_nearest_channels_last(
    // 输出张量
    const Tensor& output_,
    // 输入张量
    const Tensor& input_,
    // 缩放参数
    const scale_type& scales) {
  // 检查输入和输出张量的数据类型是否一致
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());

  // 获取输入和输出张量的尺寸
  auto input_sizes = input_.sizes().vec();
  auto output_sizes = output_.sizes().vec();
  auto ndim = input_sizes.size();
  // 检查张量的维度是否为4或5，因为 NHWC 格式只支持这两种情况
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  // 确定内存格式为 ChannelsLast 或 ChannelsLast3d
  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  // 使输入和输出张量连续化，按照指定的通道位置内存格式
  auto input = input_.contiguous(channels_last_memory_format);
  auto output = output_.contiguous(channels_last_memory_format);

  // 获取输入和输出张量的数据指针
  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // 获取批次数、通道数以及深度、高度、宽度等维度信息
  int64_t num_batches =  input_sizes[0];
  int64_t channels =  input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];
  int64_t numel = output.numel();

  // 检查通道数是否大于0
  TORCH_CHECK(channels > 0, "expected input and output channels greater than 0 but got ", channels);

  // 定义向量化操作的类型 Vec
  using Vec = vec::Vectorized<scalar_t>;
  // 定义复制数据的函数
  auto copy = [](scalar_t* out, const scalar_t* in, int64_t size) {
    int64_t d = 0;
    // 使用向量化加载和存储数据
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec out_vec = Vec::loadu(in + d);
      out_vec.store(out + d);
    }
    // 处理剩余的数据
    for (; d < size; d++) {
      out[d] = in[d];
    }
  };

  // 定义二维循环处理函数
  auto loop2d = [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    // 初始化数据索引
    data_index_init(begin, n, num_batches, oh, output_height, ow, output_width);

    // 进行循环处理
    for (const auto i : c10::irange(begin, end)) {
      // 计算最近邻索引
      int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[0]);
      int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[1]);
      // 设置输出指针和输入指针
      scalar_t* output_ptr = output_data + i * channels;
      const scalar_t* input_ptr = input_data + n * input_height * input_width * channels +
          ih * input_width * channels + iw * channels;
      // 复制数据
      copy(output_ptr, input_ptr, channels);
      // 更新数据索引
      data_index_step(n, num_batches, oh, output_height, ow, output_width);
    }
  };

  // 定义三维循环处理函数
  auto loop3d = [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    // 初始化数据索引
    data_index_init(begin, n, num_batches, od, output_depth, oh, output_height, ow, output_width);
    // 对于输入数据的每个索引 i，执行以下操作：
    for (const auto i : c10::irange(begin, end)) {
      // 获取最近邻的索引 id、ih、iw，用于输入数据的索引计算
      int64_t id = nearest_idx_fn(od, input_depth, output_depth, scales[0]);
      int64_t ih = nearest_idx_fn(oh, input_height, output_height, scales[1]);
      int64_t iw = nearest_idx_fn(ow, input_width, output_width, scales[2]);
      
      // 计算输出指针和输入指针，用于数据复制操作
      scalar_t* output_ptr = output_data + i * channels;
      const scalar_t* input_ptr = input_data + n * input_depth * input_height * input_width * channels +
          id * input_height * input_width * channels +
          ih * input_width * channels + iw * channels;
      
      // 复制输入数据到输出数据
      copy(output_ptr, input_ptr, channels);
      
      // 更新数据索引，用于下一轮迭代
      data_index_step(n, num_batches, od, output_depth, oh, output_height, ow, output_width);
    }
  };

  // 根据数据维度进行并行处理
  if (ndim == 4) {
    // 对于二维最近邻上采样，使用并行处理
    at::parallel_for(0, numel / channels, at::internal::GRAIN_SIZE / channels, loop2d);
  } else {
    // 对于三维最近邻上采样，使用并行处理
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, numel / channels, at::internal::GRAIN_SIZE / channels, loop3d);
  }

  // 如果输出数据不是按照 channels_last_memory_format 连续存储，则执行拷贝操作
  if (!output_.is_contiguous(channels_last_memory_format)) {
    output_.copy_(output);
  }
    }



template <typename scalar_t, typename accscalar_t>
// 定义插值函数，用于从给定的数据和权重进行插值计算
inline VecType<scalar_t> interpolate(const scalar_t* t, accscalar_t w) {
  // 加载数据并乘以权重，返回结果向量
  return VecType<scalar_t>::loadu(t) * VecType<scalar_t>(w);
}



template <typename scalar_t, typename accscalar_t, typename... Args>
// 重载的插值函数，支持多个参数的插值计算
inline VecType<scalar_t> interpolate(const scalar_t* t, accscalar_t w, Args... args) {
  // 加载数据并乘以权重，加上递归调用以处理更多参数
  return VecType<scalar_t>::loadu(t) * VecType<scalar_t>(w) + interpolate(args...);
}



template <typename scalar_t, typename scale_type>
// 在CPU上进行线性上采样，使用ChannelsLast内存布局
void cpu_upsample_linear_channels_last(
    const Tensor& output_,
    const Tensor& input_,
    bool align_corners,
    const scale_type& scales) {
  // 检查输入和输出的数据类型是否一致
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());

  auto input_sizes = input_.sizes().vec(); // 获取输入张量的尺寸向量
  auto output_sizes = output_.sizes().vec(); // 获取输出张量的尺寸向量
  auto ndim = input_sizes.size(); // 获取张量的维度数
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  // 确定内存布局格式
  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  // 确保张量按照指定的内存布局格式进行连续化
  auto input = input_.contiguous(channels_last_memory_format);
  auto output = output_.contiguous(channels_last_memory_format);

  auto input_data = input.const_data_ptr<scalar_t>(); // 获取输入数据的指针
  auto output_data = output.data_ptr<scalar_t>(); // 获取输出数据的指针

  int64_t num_batches =  input_sizes[0]; // 获取批次大小
  int64_t channels =  input_sizes[1]; // 获取通道数
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1; // 获取输入深度（若存在）
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1; // 获取输出深度（若存在）
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1; // 获取输入高度
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1; // 获取输出高度
  int64_t input_width = input_sizes[ndim - 1]; // 获取输入宽度
  int64_t output_width = output_sizes[ndim - 1]; // 获取输出宽度

  // 检查通道数是否大于0
  TORCH_CHECK(channels > 0, "expected input and output channels greater than 0 but got ", channels);
  // 计算每个输出切片的大小
  int64_t output_slice_size = output_depth * output_height * output_width * channels;

  // 定义操作类型和向量化类型
  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<scalar_t>;

  // 定义2D循环函数
  auto loop2d = [&](int64_t begin, int64_t end) {
    // 计算高度和宽度的缩放比例
    const auto height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[0]);
    const auto width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[1]);

    // 定义输入索引函数
    auto input_indexr = [=](int64_t n, int64_t h, int64_t w) {
      return input_data + n * input_height * input_width * channels +
          h * input_width * channels + w * channels;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ih0, ih1, iw0, iw1;
    opmath_t h0lambda, h1lambda, w0lambda, w1lambda;



    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ih0, ih1, iw0, iw1;
    opmath_t h0lambda, h1lambda, w0lambda, w1lambda;


这些注释覆盖了每行代码的具体功能和作用，确保了代码结构和注释之间的一致性和完整性。
    for (const auto n : c10::irange(begin, end)) {
      // 对于每个 batch 中的元素 n，遍历输入数据的深度方向
      for (const auto oh : c10::irange(output_height)) {
        // 对于输出的每一行 oh，计算源索引和插值权重
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        // 对于输出的每一列 ow，计算源索引和插值权重
        for (const auto ow : c10::irange(output_width)) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

          // 计算输出张量中的位置
          scalar_t* out = output_data + n * output_slice_size +
              oh * output_width * channels + ow * channels;
          // 获取四个插值点的指针
          const scalar_t* i00 = input_indexr(n, ih0, iw0);
          const scalar_t* i01 = input_indexr(n, ih0, iw1);
          const scalar_t* i10 = input_indexr(n, ih1, iw0);
          const scalar_t* i11 = input_indexr(n, ih1, iw1);
          // 计算四个插值点的权重
          opmath_t w00 = h0lambda * w0lambda;
          opmath_t w01 = h0lambda * w1lambda;
          opmath_t w10 = h1lambda * w0lambda;
          opmath_t w11 = h1lambda * w1lambda;

          // 向量化计算的步长
          int64_t size = channels;
          int64_t d = 0;
          // 向量化计算，每次处理 Vec::size() 个元素
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            // 插值计算并存储到输出向量中
            auto out_vec = interpolate(i00 + d, w00, i01 + d, w01, i10 + d, w10, i11 + d, w11);
            out_vec.store(out + d);
          }
          // 处理剩余的元素（不足 Vec::size() 个的部分）
          for (; d < size; d++) {
            // 计算并存储剩余元素的插值结果
            out[d] = i00[d] * w00 + i01[d] * w01 + i10[d] * w10 + i11[d] * w11;
          }
        }
      }
    }
  };

  // 定义三维循环的 lambda 函数
  auto loop3d = [&](int64_t begin, int64_t end) {
    // 计算深度、高度和宽度的缩放比例
    const auto depth_scale = area_pixel_compute_scale<opmath_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const auto height_scale = area_pixel_compute_scale<opmath_t>(
        input_height, output_height, align_corners, scales[1]);
    const auto width_scale = area_pixel_compute_scale<opmath_t>(
        input_width, output_width, align_corners, scales[2]);

    // 定义输入索引函数
    auto input_indexr = [=](int64_t n, int64_t d, int64_t h, int64_t w) {
      // 返回指向输入数据中指定元素的指针
      return input_data + n * input_depth * input_height * input_width * channels +
          d * input_height * input_width * channels +
          h * input_width * channels + w * channels;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t id0, id1, ih0, ih1, iw0, iw1;
    opmath_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
  for (const auto n : c10::irange(begin, end)) {
    // 遍历批次中的每个样本
    for (const auto od : c10::irange(output_depth)) {
      // 遍历输出深度维度中的每个位置
      compute_source_index_and_lambda(
          id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
      // 计算深度维度上的源索引和插值权重
      for (const auto oh : c10::irange(output_height)) {
        // 遍历输出高度维度中的每个位置
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        // 计算高度维度上的源索引和插值权重
        for (const auto ow : c10::irange(output_width)) {
          // 遍历输出宽度维度中的每个位置
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

          // 计算输出张量中当前位置的指针
          scalar_t* out = output_data + n * output_slice_size +
              od * output_height * output_width * channels +
              oh * output_width * channels + ow * channels;

          // 计算输入张量中相邻8个点的指针
          const scalar_t* i000 = input_indexr(n, id0, ih0, iw0);
          const scalar_t* i001 = input_indexr(n, id0, ih0, iw1);
          const scalar_t* i010 = input_indexr(n, id0, ih1, iw0);
          const scalar_t* i011 = input_indexr(n, id0, ih1, iw1);
          const scalar_t* i100 = input_indexr(n, id1, ih0, iw0);
          const scalar_t* i101 = input_indexr(n, id1, ih0, iw1);
          const scalar_t* i110 = input_indexr(n, id1, ih1, iw0);
          const scalar_t* i111 = input_indexr(n, id1, ih1, iw1);

          // 计算插值权重
          opmath_t w000 = d0lambda * h0lambda * w0lambda;
          opmath_t w001 = d0lambda * h0lambda * w1lambda;
          opmath_t w010 = d0lambda * h1lambda * w0lambda;
          opmath_t w011 = d0lambda * h1lambda * w1lambda;
          opmath_t w100 = d1lambda * h0lambda * w0lambda;
          opmath_t w101 = d1lambda * h0lambda * w1lambda;
          opmath_t w110 = d1lambda * h1lambda * w0lambda;
          opmath_t w111 = d1lambda * h1lambda * w1lambda;

          // 使用向量化计算插值结果
          int64_t size = channels;
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            auto out_vec = interpolate(
                i000 + d, w000, i001 + d, w001, i010 + d, w010, i011 + d, w011,
                i100 + d, w100, i101 + d, w101, i110 + d, w110, i111 + d, w111);
            out_vec.store(out + d);
          }
          // 处理剩余的元素，不使用向量化
          for (; d < size; d++) {
            out[d] =
                i000[d] * w000 + i001[d] * w001 + i010[d] * w010 + i011[d] * w011 +
                i100[d] * w100 + i101[d] * w101 + i110[d] * w110 + i111[d] * w111;
          }
        }
      }
    }
  };

  if (ndim == 4) {
    // 如果输入张量是4维，则执行2D最近邻上采样
    at::parallel_for(0, num_batches, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // 如果输入张量是5维，则执行3D最近邻上采样
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, num_batches, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  if (!output_.is_contiguous(channels_last_memory_format)) {
    // 如果输出张量不是连续的，则进行拷贝操作
    output_.copy_(output);
  }
}

// 在 upsample_generic_Nd_kernel_impl 中使用的辅助结构体
struct HelperInterpBase {

  // 初始化索引和权重
  static inline void init_indices_weights(
    at::ScalarType output_type,
    std::vector<Tensor> & output, int64_t output_size, int64_t ndims,
    int64_t reshape_dim, int interp_size
  ) {

    // 创建新形状，将指定维度设置为输出大小，其余维度为1
    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    // 遍历插值次数，创建空的输出张量对，并将其加入输出向量
    for (const auto j C10_UNUSED : c10::irange(interp_size)) {
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
      output.emplace_back(empty(new_shape, CPU(output_type)));
    }
  }

  // 这是 _compute_index_ranges_weights 方法的辅助函数，用于计算索引范围和权重
  // 用于具有抗锯齿=true模式的插值。它返回最大权重值
  template <typename scalar_t, typename aa_filter_fn_t>
  static inline scalar_t _compute_indices_min_size_weights_aa(
    const int64_t i, const int64_t input_size, const scalar_t scale, const scalar_t support,
    scalar_t* wt_ptr, const int64_t max_interp_size, aa_filter_fn_t filter_fn,
    int64_t& xmin, int64_t& xsize
  ) {

    scalar_t center = scale * (i + 0.5);
    scalar_t total_w = 0.0;
    scalar_t invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
    
    // 计算 xmin 和 xsize
    xmin = std::max(
        static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
    xsize = std::min(
        static_cast<int64_t>(center + support + 0.5), input_size) - xmin;
    // 由于精度问题，有时 xsize 可能比 max_interp_size 大一
    // 需要对其进行修剪
    xsize = std::clamp(xsize, static_cast<int64_t>(0), max_interp_size);

    int64_t j = 0;
    for (; j < xsize; j++) {
      // 计算权重并存入 wt_ptr
      scalar_t w = filter_fn((j + xmin - center + 0.5) * invscale);
      wt_ptr[j] = w;
      total_w += w;
    }

    scalar_t wt_max = 0.0;
    if (total_w != 0.0) {
      // 归一化权重并找出最大权重值
      for (j = 0; j < xsize; j++) {
        wt_ptr[j] /= total_w;
        wt_max = std::max(wt_max, wt_ptr[j]);
      }
    }

    // 对未使用的权重位置填充零
    for (; j < max_interp_size; j++) {
      wt_ptr[j] = static_cast<scalar_t>(0.0);
    }
    return wt_max;
  }

  // 这是 _compute_index_ranges_weights 方法的辅助函数，用于计算索引范围和权重
  // 用于插值与抗锯齿=false模式。它返回最大权重值。
  // 这个函数使用 scalar_t 来表示比例和权重的类型，但仅用于 uint8 输入和抗锯齿=false 的双线性/双三次插值
  // 对于浮点输入类型，我们使用 upsample_generic_Nd_kernel_impl 和 compute_indices_weights 方法
  template <typename scalar_t, typename aa_filter_fn_t>
  static inline scalar_t _compute_indices_min_size_weights(
    const int64_t i, const int64_t input_size, const scalar_t scale,
    ```
    // 在此方法中不使用 opmath_t，因为 f16 和其他较小的浮点类型不会路由到这里
    // 典型用法是 scalar_t = double，用于计算 uint8 输入的索引和权重
    // 以下代码部分从 compute_indices_weights 方法和 _compute_indices_min_size_weights_aa 方法中适配索引和 lambda 计算以及 index_min/index_size 的使用

    // 检查是否采用立方插值
    bool cubic = max_interp_size > 2;
    
    // 计算真实的输入索引，根据像素区域计算源索引
    const auto real_input_index = area_pixel_compute_source_index<scalar_t>(
        scale, i, align_corners, /*cubic=*/cubic);
    
    scalar_t lambda;
    int64_t input_index;
    
    // 保护索引和 lambda 值
    guard_index_and_lambda(real_input_index, input_size, input_index, lambda);
    
    // 计算支持范围
    const auto support = static_cast<int64_t>(max_interp_size * 0.5);
    
    // 计算未限制的最小和最大索引
    const auto unbound_index_min = input_index - support + 1;
    const auto unbound_index_max = input_index + support + 1;
    
    // 确定实际的 index_min
    index_min = std::max(unbound_index_min, static_cast<int64_t>(0));
    
    // 确定实际的 index_size
    index_size = std::min(unbound_index_max, input_size) - index_min;
    
    // 在某些罕见情况下，由于精度问题，index_size 可能比 max_interp_size 大一
    // 因此需要进行截断处理
    index_size = std::clamp(index_size, static_cast<int64_t>(0), max_interp_size);

    // 计算权重，使用 filter_fn，并且对超出边界的索引进行累加处理
    // 例如，在双立方插值模式下，对于输出索引 i = 0，input_index = -1，
    // unbound_index_min = -2，unbound_index_max = 1
    // 对于超出边界的输入索引，计算四个非零权重值 [w0, w1, w2, w3]
    // 有效的输入索引是 [0, 1]
    // 对于超出边界的输入索引，累加值如下：[w0 + w1 + w2, w3, 0.0, 0.0]
    // 这相当于浮点路径中计算索引为 [0, 0, 0, 1]，权重为 [w0, w1, w2, s3] 的方式
    // 对于超出输入尺寸的索引，也应该进行类似的累加处理
    auto w_index = 0;
    scalar_t wt_max = 0.0;
    for (const auto j : c10::irange(max_interp_size)) {
        // 初始化权重值，将在下面进行累加
        wt_ptr[j] = 0.0;

        // 计算权重值 w
        scalar_t w = filter_fn(static_cast<scalar_t>(j + 1 - support) - lambda);
        
        // 根据索引范围确定累加的位置
        if (unbound_index_min + j <= 0) {
            w_index = 0;
        } else if (unbound_index_min + j >= input_size - 1) {
            w_index = index_size - 1;
        }
        
        // 累加权重值
        wt_ptr[w_index] += w;
        
        // 更新最大权重值
        wt_max = std::max(wt_max, wt_ptr[w_index]);
        
        // 移动累加索引
        w_index++;
    }
    return wt_max;
  }

  // Note [ Support for antialias=False as a subcase of antialias=True ]
  // This function was originally written with the hard assumption that
  // antialias=True and it was later extended to support antialias=False.
  // The only difference between aa and no-aa is in how the
  // weights and indices are computed (and their number). In aa their number is
  // variable but with no-aa, they're fixed to interp_size. The same "filters"
  // can be used otherwise. HOWEVER, support for antialias=False here may not be
  // optimally optimized: the code assumes an arbitrary number of weights and
  // indices, but this can be optimized further when aa=False since we know
  // their actual dimensions.
  template <typename scalar_t, typename aa_filter_fn_t, int weight_index_stride=sizeof(scalar_t)>
  static inline std::tuple<std::vector<Tensor>, int, scalar_t> _compute_index_ranges_weights(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, scalar_t scale,
    int interp_size, aa_filter_fn_t aa_filter_fn, bool antialias, bool align_corners
  ) {

    std::vector<Tensor> output;

    scalar_t support;
    int max_interp_size;
    if (antialias) {
        // Calculate support based on scale for antialiased interpolation
        support = (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
        // Determine max_interp_size for antialiased interpolation
        max_interp_size = (int) std::ceil(support) * 2 + 1;
    } else {
        // For non-antialiased interpolation, fixed support and max_interp_size
        support = interp_size * 0.5;
        max_interp_size = interp_size;
    }

    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    // Bounds approach as in PIL: xmin/xmax
    // Create tensors for xmin, size, and stride
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));

    {
      // Weights
      new_shape[reshape_dim] = output_size * max_interp_size;
      // Create empty tensor for weights
      auto wts = empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>()));
      auto strides = wts.strides().vec();
      strides[reshape_dim] = 0;
      new_shape[reshape_dim] = output_size;
      // Stride the weights tensor according to new_shape and strides
      wts = wts.as_strided(new_shape, strides);
      output.emplace_back(wts);
      // Weights indices
      // Create empty tensor for weights indices
      output.emplace_back(
          empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
    }

    // Obtain pointers to data in output tensors
    int64_t* idx_ptr_xmin = output[0].data_ptr<int64_t>();
    int64_t* idx_ptr_size = output[1].data_ptr<int64_t>();
    int64_t* idx_ptr_stride = output[2].data_ptr<int64_t>();
    scalar_t* wt_ptr = output[3].data_ptr<scalar_t>();
    int64_t* wt_idx_ptr = output[4].data_ptr<int64_t>();

    // Initialize maximum weight to 0.0
    scalar_t wt_max = 0.0;
    // 遍历输出尺寸的范围
    for (const auto i : c10::irange(output_size)) {
      // 定义变量：xmin（最小索引），xsize（尺寸），wt_max_i（权重的最大值）
      int64_t xmin, xsize;
      scalar_t wt_max_i;
      // 如果需要抗锯齿处理
      if (antialias) {
        // 计算带抗锯齿效果的最小索引、尺寸和权重
        wt_max_i = HelperInterpBase::_compute_indices_min_size_weights_aa(
            i,
            input_size,
            scale,
            support,
            wt_ptr + i * max_interp_size,
            max_interp_size,
            aa_filter_fn,
            xmin,
            xsize);
      } else {
        // 计算不带抗锯齿效果的最小索引、尺寸和权重
        wt_max_i = HelperInterpBase::_compute_indices_min_size_weights(
            i,
            input_size,
            scale,
            wt_ptr + i * max_interp_size,
            max_interp_size,
            aa_filter_fn,
            align_corners,
            xmin,
            xsize);
      }
      // 更新权重的最大值
      wt_max = std::max(wt_max, wt_max_i);

      // 将计算得到的xmin乘以步长，存入索引指针数组
      idx_ptr_xmin[i] = xmin * stride;
      // 存入计算得到的xsize到尺寸指针数组
      idx_ptr_size[i] = xsize;
      // 存入步长到步长指针数组
      idx_ptr_stride[i] = stride;
      // 计算权重索引的起始位置，存入权重索引指针数组
      wt_idx_ptr[i] = i * max_interp_size * weight_index_stride;
    }
    // 返回包含输出、最大插值尺寸和权重最大值的元组
    return {output, max_interp_size, wt_max};
  }

  /*
  NOTE [ Weights computation for uint8_t and multiplication trick ]
  当输入/输出的数据类型是uint8_t时，我们仍然使用double计算插值权重，但然后通过某些详细说明的转换逻辑将它们转换为int16。
  这允许我们将所有插值操作（乘法的总和）作为整数而不是浮点数来计算。结果最终会在basic_loop_aa_horizontal<uint8_t>（和vertical）中转换回uint8。

  实质上的想法是避免在浮点数（权重）和整数（像素值）之间进行乘法，而是在两个整数之间进行乘法运算：

  ```py
  COEF_PREC = 16

  def mul(a:float, b:int) -> Tuple[float, int]:
    # return a * b, round(a * b)
    actual = a * b

    assert a > 0  # I'm lazy
    int_a = floor(0.5 + a * (1 << COEF_PREC))
    with_trick = ((int_a * b) + (1 << (COEF_PREC - 1))) >> COEF_PREC

    return actual, with_trick  # round(actual) == with_trick!!
  ```

  下面是它的工作原理：
  N == COEFF_PREC
  1 << N == 2**N
  floor(0.5 + x) == round(x)

  所以操作类似于

  int_a = round(a * 2**N)  -- 简单地说就是 `a * 2**N`

  res = ((int_a * b) + (1 << (N - 1))) >> N
      = ((a * 2**N * b + 2**(N - 1)) / 2**N
      = a * b + 0.5
      = round(a * b)
      = 我们想要的结果
  */
  // 模板函数：计算int16类型权重的索引范围
  template <typename aa_filter_fn_t>
  static inline std::tuple<std::vector<Tensor>, int, unsigned int> _compute_index_ranges_int16_weights(
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, bool align_corners, const std::optional<double> opt_scale,
    int interp_size, aa_filter_fn_t aa_filter_fn, bool antialias, bool align_i32=false
  ) {
    // 计算像素级别的缩放比例
    double scale = area_pixel_compute_scale<double>(
        input_size, output_size, align_corners, opt_scale);

    // 创建存放索引权重的向量
    std::vector<Tensor> indices_weights;
    // 初始化最大权重
    double wt_max;
    // 调用 HelperInterpBase::_compute_index_ranges_weights 函数，计算索引、权重、最大权重
    std::tie(indices_weights, interp_size, wt_max) = HelperInterpBase::_compute_index_ranges_weights<double, aa_filter_fn_t, sizeof(int16_t)>(
        input_size, output_size, stride, ndims, reshape_dim, scale, interp_size, aa_filter_fn, antialias, align_corners);

    // 将浮点权重重新缩放为 int16，并计算权重精度
    auto weights_f64 = indices_weights[3];
    double * data_f64 = weights_f64.data_ptr<double>();

    unsigned int weights_precision = 0;
    for (weights_precision = 0; weights_precision < 22; ++weights_precision) {
        // 计算下一个可能的权重值，并检查是否超过 int16 的范围
        int next_value = (int) (0.5 + wt_max * (1 << (weights_precision + 1)));
        if (next_value >= (1 << 15))
            break;
    }

    // 将浮点值重新缩放为 int16
    int16_t * data_i16 = (int16_t *) data_f64;
    auto aligned_interp_size = interp_size;

    if (align_i32) {
        // 如果需要 int32 对齐，则调整 aligned_interp_size 到最近的 int32 对齐的值
        while (aligned_interp_size % sizeof(int32_t) != 0) {
            aligned_interp_size += 1;
        }
        // 断言确保不会超出边界
        TORCH_INTERNAL_ASSERT(aligned_interp_size * sizeof(int16_t) < interp_size * sizeof(double));
    }

    // 将重新缩放后的浮点权重值转换为 int16 存储在 data_i16 中
    for (const auto j : c10::irange(output_size)) {
        for (const auto k : c10::irange(interp_size)) {
            double v = data_f64[j * interp_size + k] * (1 << weights_precision);
            data_i16[j * aligned_interp_size + k] = (v < 0) ? (int) (-0.5 + v) : (int) (0.5 + v);
        }
    }

    // 返回计算得到的结果：索引、对齐后的 interp_size、权重精度
    return {indices_weights, aligned_interp_size, weights_precision};
}
};

// 结构体 HelperInterpNearest 继承自 HelperInterpBase，实现了使用过时且存在缺陷的最近邻插值计算索引的方法。
// 我们保留该结构体用于向后兼容，并将其视为已弃用。
// 可以查看 HelperInterpNearestExact 作为替代方法。

struct HelperInterpNearest : public HelperInterpBase {

  // 插值大小的常量，这里是1
  static const int interp_size = 1;

  // 初始化索引和权重
  static inline void init_indices_weights(
    at::ScalarType output_type,
    std::vector<Tensor> & output, int64_t output_size, int64_t ndims,
    int64_t reshape_dim, int interp_size
  ) {
    // 创建一个新形状的向量，所有维度都为1，但指定位置 reshape_dim 处为 output_size
    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    // 对于 interp_size 中的每一个元素，执行以下操作
    for (const auto j C10_UNUSED : c10::irange(interp_size)) {
      // 向 output 向量中添加一个形状为 new_shape 的空张量，类型为 int64_t
      output.emplace_back(empty(new_shape, CPU(c10::CppTypeToScalarType<int64_t>())));
      // 为保持一致性定义权重，但未使用
      output.emplace_back(at::ones(new_shape, CPU(output_type)));
    }
  }

  // 计算最近邻模式下每个插值维度的索引和权重
  // indices_weights = {
  //      {indices_0, 1.0, },  // 维度 -n
  //      {indices_0, 1.0, },  // 维度 -(n-1)
  //      ...
  //      {indices_0, 1.0, },  // 维度 -1
  // }
  // 索引和权重被重塑为 (1, 1, ..., N, ..., 1, 1) 以适应输入/输出张量。
  // 索引已经包含了步长以优化计算。
  static inline std::vector<Tensor> compute_indices_weights(
    at::ScalarType scalar_type,
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims,
    int64_t reshape_dim, bool align_corners, const std::optional<double> opt_scale
  ) {

    // 内部断言，确保不使用 align_corners
    TORCH_INTERNAL_ASSERT(!align_corners);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> output;
    // 使用 HelperInterpNearest::interp_size 初始化索引和权重
    HelperInterpNearest::init_indices_weights(
      scalar_type, output, output_size, ndims, reshape_dim, HelperInterpNearest::interp_size);

    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，遍历浮点类型和 scalar_type
    AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, scalar_type, "compute_indices_weights_nearest", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        // 计算像素级别的缩放比例
        opmath_t scale = area_pixel_compute_scale<opmath_t>(input_size, output_size, align_corners, opt_scale);

        // 获取输出中的输入索引指针，并初始化输入索引
        auto input_index_ptr = output[0].data_ptr<int64_t>();
        int64_t input_index;

        // 计算索引的方法如下：
        // scale = 1.0 * isize / osize
        // index_f32 = (output_index) * scale
        // input_index = floor(index_f32)
        // 与 OpenCV INTER_NEAREST 相同
        for (const auto i : c10::irange(output_size)) {
          const auto real_input_index =
              area_pixel_compute_source_index<opmath_t>(
                  scale, i, /*align_corners=*/true, /*cubic=*/false);
          input_index = static_cast<int64_t>(floorf(real_input_index));
          // 将 input_index_ptr[i] 设置为输入索引的最小值，乘以步长
          input_index_ptr[i] = static_cast<int64_t>(std::min(input_index, input_size - 1)) * stride;
        }
      }
    );
    // 返回结果向量
    return output;
  }

};
// HelperInterpNearestExact 结构体，继承自 HelperInterpNearest 类

// 为每个插值维度计算最近插值模式的索引和权重
// indices_weights 是一个包含以下结构的向量：
// {
//      {indices_0, 1.0, },  // 维度 -n
//      {indices_0, 1.0, },  // 维度 -(n-1)
//      ...
//      {indices_0, 1.0, },  // 维度 -1
// }
// 索引和权重被重塑为 (1, 1, ..., N, ..., 1, 1) 来适应输入/输出张量。
// 索引已经包含了步长以优化计算。

static inline std::vector<Tensor> compute_indices_weights(
  at::ScalarType scalar_type,        // 标量类型
  int64_t input_size,                // 输入尺寸
  int64_t output_size,               // 输出尺寸
  int64_t stride,                    // 步长
  int64_t ndims,                     // 维度数
  int64_t reshape_dim,               // 重塑维度
  bool align_corners,                // 是否对齐角点
  const std::optional<double> opt_scale  // 可选的缩放比例
) {

  TORCH_INTERNAL_ASSERT(!align_corners);  // 内部断言，确保不对齐角点
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<Tensor> output;  // 输出张量向量
  HelperInterpNearest::init_indices_weights(  // 初始化索引和权重
    scalar_type, output, output_size, ndims, reshape_dim, HelperInterpNearest::interp_size);

  // 根据浮点类型分派计算，处理浮点类型和 kBFloat16、kHalf 类型
  AT_DISPATCH_FLOATING_TYPES_AND2(
    kBFloat16, kHalf, scalar_type, "compute_indices_weights_nearest", [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_t scale = area_pixel_compute_scale<opmath_t>(input_size, output_size, align_corners, opt_scale);

      auto input_index_ptr = output[0].data_ptr<int64_t>();  // 输入索引指针
      int64_t input_index;

      // 索引计算如下：
      // scale = 1.0 * isize / osize
      // index_f32 = (output_index + 0.5) * scale - 0.5
      // input_index = round(index_f32)
      // 与 Pillow 和 Scikit-Image/Scipy ndi.zoom 相同
      for (const auto i : c10::irange(output_size)) {
        const auto real_input_index =
            area_pixel_compute_source_index<opmath_t>(
                scale, i, /*align_corners=*/align_corners, /*cubic=*/false);
        input_index = static_cast<int64_t>(floorf(real_input_index + 0.5));  // 四舍五入到最近整数
        input_index_ptr[i] = static_cast<int64_t>(std::min(input_index, input_size - 1)) * stride;  // 设置输入索引
      }
    }
  );
  return output;  // 返回结果向量
}



// HelperInterpLinear 结构体，继承自 HelperInterpBase 类

// 为每个插值维度计算索引和权重
// indices_weights 是一个包含以下结构的向量：
// {
//      {indices_0, weights_0, indices_1, weights_1},  // 维度 -n
//      {indices_0, weights_0, indices_1, weights_1},  // 维度 -(n-1)
//      ...
//      {indices_0, weights_0, indices_1, weights_1},  // 维度 -1
// }
// 索引和权重被重塑为 (1, 1, ..., N, ..., 1, 1) 来适应输入/输出张量。
// 索引已经包含了步长以优化计算。

static inline std::vector<Tensor> compute_indices_weights(
  at::ScalarType scalar_type,        // 标量类型
  int64_t input_size,                // 输入尺寸
  int64_t output_size,               // 输出尺寸
  int64_t stride,                    // 步长
  int64_t ndims,                     // 维度数
  int64_t reshape_dim,               // 重塑维度
  bool align_corners,                // 是否对齐角点
  const std::optional<double> opt_scale  // 可选的缩放比例
) {
    // 创建一个空的 Tensor 向量用于存储输出
    std::vector<Tensor> output;
    // 调用 HelperInterpLinear 类的静态方法 init_indices_weights，
    // 初始化输出 Tensor 向量，填充索引和权重
    HelperInterpLinear::init_indices_weights(
      scalar_type, output, output_size, ndims, reshape_dim, HelperInterpLinear::interp_size);
    // 根据标量类型进行分发，使用 lambda 表达式定义计算线性插值索引和权重的操作
    AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, scalar_type, "compute_indices_weights_linear", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        // 计算像素区域的比例因子
        opmath_t scale = area_pixel_compute_scale<opmath_t>(input_size, output_size, align_corners, opt_scale);

        // 获取输出向量中各个 Tensor 的指针，用于访问数据
        auto input_index0_ptr = output[0].data_ptr<int64_t>();
        auto lambda0_ptr = output[1].data_ptr<scalar_t>();
        auto input_index1_ptr = output[2].data_ptr<int64_t>();
        auto lambda1_ptr = output[3].data_ptr<scalar_t>();

        // 遍历输出大小的范围
        for (const auto i : c10::irange(output_size)) {
          // 计算源索引和权重的函数调用
          compute_source_index_and_lambda<scalar_t, opmath_t>(
            input_index0_ptr[i], input_index1_ptr[i],
            lambda0_ptr[i], lambda1_ptr[i],
            scale, i, input_size, output_size, align_corners
          );
          // 将步长乘以输入索引，将计算结果存入 indices
          // 索引值对应于输入索引 (0, 1, 2, 3, ...)，乘以输入步长，对于给定维度可以得到最大可能的值
          input_index0_ptr[i] *= stride;
          input_index1_ptr[i] *= stride;
        }
      }
    );
    // 返回填充好的输出 Tensor 向量
    return output;
  }

  // 从以下链接中获取：https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L20-L29
  // 定义一个静态内联函数，实现双线性插值的滤波器
  template<typename scalar_t>
  static inline scalar_t aa_filter(scalar_t x) {
    // 计算绝对值
    x = std::abs(x);
    // 根据 x 的大小返回滤波器系数
    if (x < 1.0) {
      return 1.0 - x;
    }
    return 0.0;
  }

  // 定义一个静态内联函数，计算索引范围和权重的 Tensor 向量
  static inline std::vector<Tensor> compute_index_ranges_weights(
    at::ScalarType scalar_type,
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const std::optional<double> opt_scale,
    bool antialias
  ) {
    // 创建一个 Tensor 向量用于存储索引和权重
    std::vector<Tensor> indices_weights;
    // 根据标量类型进行分发，使用 lambda 表达式定义计算索引范围和权重的操作
    AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "compute_index_ranges_weights", [&] {

        // 计算像素区域的比例因子
        scalar_t scale = area_pixel_compute_scale<scalar_t>(
            input_size, output_size, align_corners, opt_scale);

        // 获取插值大小
        auto interp_size = HelperInterpLinear::interp_size;

        // 调用 HelperInterpLinear 类的静态方法 _compute_index_ranges_weights，
        // 初始化索引和权重的 Tensor 向量
        indices_weights = std::get<0>(HelperInterpLinear::_compute_index_ranges_weights<scalar_t>(
            input_size,
            output_size,
            stride,
            ndims,
            reshape_dim,
            scale,
            interp_size,
            &HelperInterpLinear::aa_filter<scalar_t>,
            /*antialias=*/antialias,
            /*align_corners=*/align_corners));
      }
    );
    // 返回填充好的索引和权重的 Tensor 向量
    return indices_weights;
  }

  // 定义一个静态内联函数，计算索引范围和权重的 Tensor 向量
  static inline std::tuple<std::vector<Tensor>, int, unsigned int> compute_index_ranges_int16_weights(
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const std::optional<double> opt_scale,
    bool antialias
  ) {
    // 定义一个函数，接受多个参数：
    // - input_size: 输入大小
    // - output_size: 输出大小
    // - stride: 步长
    // - ndims: 维度数目
    // - reshape_dim: 重塑维度
    // - align_corners: 是否对齐角点
    // - opt_scale: 可选的缩放因子，可能为空
    // - interp_size: 线性插值大小，从 HelperInterpLinear::interp_size 获取
    // - fn: 双线性插值滤波器函数，从 HelperInterpLinear::aa_filter<double> 获取
    // - antialias: 是否开启抗锯齿
    // - align_i32: 是否对齐到32位，默认为false
    ) {
    
    // 获取 HelperInterpLinear 的静态成员 interp_size
    auto interp_size = HelperInterpLinear::interp_size;
    // 获取 HelperInterpLinear 的静态成员函数 aa_filter<double> 并赋值给 fn
    auto fn = HelperInterpLinear::aa_filter<double>;
    // 调用 HelperInterpLinear 类的静态方法 _compute_index_ranges_int16_weights，
    // 传入所有参数，计算并返回结果
    return HelperInterpLinear::_compute_index_ranges_int16_weights(
        input_size, output_size, stride, ndims, reshape_dim,
        align_corners, opt_scale, interp_size, fn, antialias, align_i32);
    }
};

// 辅助插值立方类，继承自基础插值辅助类 HelperInterpBase
struct HelperInterpCubic : public HelperInterpBase {

  // 插值大小常量设为 4
  static const int interp_size = 4;

  // 计算每个插值维度的索引和权重
  // indices_weights 是一个包含每个维度插值索引和权重的向量
  // 每个维度的数据格式为 {indices_0, weights_0, indices_1, weights_1, ..., indices_3, weights_3}
  // 索引和权重被重新整形为 (1, 1, ..., N, ..., 1, 1)，以适应输入/输出张量
  // 索引已包含步长，用于优化计算
  static inline std::vector<Tensor> compute_indices_weights(
    at::ScalarType scalar_type,
    int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim,
    bool align_corners, const std::optional<double> opt_scale
  ) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> output;
    // 初始化索引和权重数组
    HelperInterpCubic::init_indices_weights(
      scalar_type, output, output_size, ndims, reshape_dim, HelperInterpCubic::interp_size);

    // 根据浮点类型分派计算任务，使用 lambda 表达式
    AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, scalar_type, "compute_indices_weights_cubic", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        // 计算缩放因子，用于像素区域计算
        opmath_t scale = area_pixel_compute_scale<opmath_t>(input_size, output_size, align_corners, opt_scale);

        int64_t input_index;
        int64_t zero = static_cast<int64_t>(0);
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        opmath_t coeffs[4];

        int64_t * idx_ptr;
        scalar_t * wt_ptr;
        // 遍历输出尺寸
        for (const auto i : c10::irange(output_size)) {
          // 计算实际输入索引
          const auto real_input_index =
              area_pixel_compute_source_index<opmath_t>(
                  scale, i, align_corners, /*cubic=*/true);
          opmath_t lambda;
          // 确保索引和 lambda 在有效范围内
          guard_index_and_lambda(real_input_index, input_size, input_index, lambda);
          // 获取立方插值系数
          get_cubic_upsample_coefficients<opmath_t>(coeffs, lambda);

          // 遍历插值大小
          for (const auto j : c10::irange(interp_size)) {
            idx_ptr = output[2 * j + 0].data_ptr<int64_t>();
            // 计算索引值
            idx_ptr[i] = static_cast<int64_t>(std::max(std::min(input_index + j - 1, input_size - 1), zero)) * stride;
            wt_ptr = output[2 * j + 1].data_ptr<scalar_t>();
            // 存储权重值
            wt_ptr[i] = coeffs[j];
          }
        }
      }
    );
    return output;
  }

  // 源自于 Pillow 库，用于双三次插值滤波
  template<typename scalar_t, bool use_keys_cubic=true>
  static inline scalar_t aa_filter(scalar_t x) {
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    // Keys 提出了在数字图像处理中使用的双三次插值算法
    // 使用 a = -0.5 来实现双三次插值，antialiasing=true (与 PIL 兼容)
    // 返回双三次插值滤波后的结果
    return -0.5;
  }
    // 根据 use_keys_cubic 的值选择合适的常数 -0.5 或 -0.75
    constexpr scalar_t a = use_keys_cubic ? -0.5 : -0.75;

    // 取 x 的绝对值
    x = std::abs(x);
    // 根据 x 的值选择不同的三次插值函数进行计算和返回
    if (x < 1.0) {
        return cubic_convolution1(x, a);
    }
    if (x < 2.0) {
        return cubic_convolution2(x, a);
    }
    // 如果 x 大于等于 2.0，则返回 0.0
    return 0.0;
  }

  // 计算索引范围和权重的函数
  static inline std::vector<Tensor> compute_index_ranges_weights(
    at::ScalarType scalar_type,
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const std::optional<double> opt_scale,
    bool antialias
  ) {

    // 存储计算结果的向量
    std::vector<Tensor> indices_weights;
    // 根据数据类型分发处理浮点数类型的计算
    AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "compute_index_ranges_weights", [&] {

        // 计算缩放比例
        scalar_t scale = area_pixel_compute_scale<scalar_t>(
            input_size, output_size, align_corners, opt_scale);

        // 获取插值方法的大小
        auto interp_size = HelperInterpCubic::interp_size;

        // 调用 HelperInterpCubic 类的静态方法计算索引范围和权重
        indices_weights = std::get<0>(HelperInterpCubic::_compute_index_ranges_weights<scalar_t>(
            input_size,
            output_size,
            stride,
            ndims,
            reshape_dim,
            scale,
            interp_size,
            // 根据 antialias 的值选择不同的滤波器函数
            &HelperInterpCubic::aa_filter<scalar_t>,
            /*antialias=*/antialias,
            /*align_corners=*/align_corners));
      }
    );
    // 返回计算得到的索引范围和权重的结果向量
    return indices_weights;
  }

  // 计算 int16 类型索引范围和权重的函数
  static inline std::tuple<std::vector<Tensor>, int, unsigned int> compute_index_ranges_int16_weights(
    int64_t input_size,
    int64_t output_size,
    int64_t stride,
    int64_t ndims,
    int64_t reshape_dim,
    bool align_corners,
    const std::optional<double> opt_scale,
    bool antialias,
    bool align_i32=false
  ) {

    // 获取插值方法的大小
    auto interp_size = HelperInterpCubic::interp_size;
    // 根据 antialias 的值选择适当的滤波器函数
    auto fn = antialias ? HelperInterpCubic::aa_filter<double, true> : HelperInterpCubic::aa_filter<double, false>;
    
    // 调用 HelperInterpCubic 类的静态方法计算 int16 类型索引范围和权重
    return HelperInterpCubic::_compute_index_ranges_int16_weights(
        input_size, output_size, stride, ndims, reshape_dim,
        align_corners, opt_scale, interp_size, fn, antialias, align_i32);
  }
};

// 通用的N维上采样插值核函数。
// 假设输入格式如NCHW、NCL或NCKHW，其中插值的空间维度从末尾到批量大小N和通道数C。
//
// 内部使用TensorIterator来优化计算。
// - out_ndims是插值维度的数量：1、2或3
// - scale_type是用于缩放的模板类型，通常为std::optional<double>
// - template<typename> class F是用于计算索引和权重的结构体之一
template <int out_ndims, typename scale_type, class F>
void upsample_generic_Nd_kernel_impl(
    const Tensor& output,      // 输出张量
    const Tensor& input,       // 输入张量
    bool align_corners,        // 是否对齐角点
    const scale_type& scales) {  // 缩放因子

  // input可以是NCHW、NCL或NCKHW格式
  auto shape = input.sizes().vec();    // 获取输入张量的形状
  auto strides = input.strides().vec();  // 获取输入张量的步幅
  auto oshape = output.sizes();         // 获取输出张量的形状

  TORCH_INTERNAL_ASSERT(
    shape.size() == oshape.size() && shape.size() == 2 + out_ndims
  );  // 断言输入和输出张量形状的维度匹配
  TORCH_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);  // 断言步幅的维度正确

  for (const auto i : c10::irange(out_ndims)) {
    shape[i + 2] = oshape[i + 2];     // 更新形状中的插值维度
    strides[i + 2] = 0;               // 插值维度的步幅设为0
  }
  auto restrided_input = input.as_strided(shape, strides);  // 对输入进行重新步进化

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<std::vector<Tensor>> indices_weights;  // 存储索引和权重的向量

  constexpr int interp_size = F::interp_size;  // 插值大小
  auto input_scalar_type = input.scalar_type();  // 输入张量的数据类型
  if ((interp_size == 1 && input_scalar_type == at::ScalarType::Byte)) {
    // 如果插值大小为1且输入类型为Byte，需要使用Float类型进行计算
    input_scalar_type = at::ScalarType::Float;
  }

  for (const auto i : c10::irange(out_ndims)) {
    // NOLINTNEXTLINE(performance-inefficient-vector-operation)
    indices_weights.emplace_back(
      F::compute_indices_weights(
        input_scalar_type, input.size(i + 2), oshape[i + 2],
        input.stride(i + 2) * input.element_size(),
        input.dim(), i + 2, align_corners, scales[i]
      )
    );  // 计算每个插值维度的索引和权重
  }

  TensorIteratorConfig config;  // 张量迭代器配置
  config.check_all_same_dtype(false)  // 不检查所有张量是否具有相同的数据类型
    .declare_static_dtype_and_device(input.scalar_type(), input.device())  // 声明静态数据类型和设备
    .add_output(output)   // 添加输出张量
    .add_const_input(restrided_input);  // 添加常量输入张量

  for (auto & idx_weight: indices_weights) {
    for (auto& tensor : idx_weight) {
      config.add_const_input(tensor);  // 添加每个插值维度的索引和权重张量
    }
  }

  auto iter = config.build();  // 构建张量迭代器

  if (interp_size > 1) {
    // 如果插值大小大于1
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16, kHalf, iter.dtype(), "upsample_generic_Nd", [&] {
        // MSVC无法在此处捕捉constexpr int interp_size
        constexpr int mode = F::interp_size;
        cpu_upsample_generic<scalar_t, out_ndims, mode>(iter);  // 调用CPU上采样通用函数
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf,
        iter.dtype(), "upsample_generic_Nd", [&] {
        constexpr int mode = F::interp_size;
        cpu_upsample_generic<scalar_t, out_ndims, mode>(iter);  // 调用CPU上采样通用函数
    });
  }
}
// 定义模板函数，用于在CPU上进行通用的双线性插值或双三次插值的上采样操作
template <typename scalar_t, bool is_horizontal>
void cpu_upsample_generic_aa(at::TensorIterator& iter, unsigned int weights_precision) {

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    if (is_horizontal) {
      
      // 判断是否为水平方向上采样，根据通道顺序和步长情况进行判断和操作
      // 如果步长符合特定条件，执行基本水平双线性或双三次插值操作
      if ((strides[0] == sizeof(scalar_t)) && (strides[1] == sizeof(scalar_t)) &&
          is_zero_stride<3 + 2>(&strides[2])) {
        // 通道顺序为最后一维时，执行水平方向的插值操作
        basic_loop_aa_horizontal<scalar_t>(
            data, strides, n, weights_precision);
      } else {
        // 通道顺序为第一维时，同样执行水平方向的插值操作
        basic_loop_aa_horizontal<scalar_t>(
            data, strides, n, weights_precision);
      }
    } else {
      // 判断是否为垂直方向上采样，根据通道顺序和步长情况进行判断和操作
      // 如果步长符合特定条件，执行基本垂直双线性或双三次插值操作
      if ((strides[0] == sizeof(scalar_t)) && (strides[1] == sizeof(scalar_t)) &&
          is_zero_stride<3 + 2>(&strides[2])) {
        // 执行垂直方向的插值操作
        basic_loop_aa_vertical<scalar_t>(
            data, strides, n, weights_precision);
      } else {
        // 通道顺序为第一维时，同样执行垂直方向的插值操作
        basic_loop_aa_vertical<scalar_t>(
            data, strides, n, weights_precision);
      }
    }
  };

  // 对迭代器应用循环操作
  iter.for_each(loop);
}

// 定义模板函数，用于在N维数据中的单个维度上执行可分离的插值操作
template <int out_ndims, typename scale_type, class F, bool is_horizontal>
void _separable_upsample_generic_Nd_kernel_impl_single_dim(
    const Tensor& output,
    const Tensor& input,
    int interp_dim,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {

  // 获取输入和输出张量的形状和步长信息
  auto shape = input.sizes().vec();
  auto strides = input.strides().vec();
  auto oshape = output.sizes();

  // 确保输入张量和输出张量的维度匹配
  TORCH_INTERNAL_ASSERT(
      shape.size() == oshape.size() && shape.size() == 2 + out_ndims);
  TORCH_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);

  // 根据输出张量的形状，调整输入张量的形状
  for (const auto i : c10::irange(out_ndims)) {
    shape[i + 2] = oshape[i + 2];
  }
  
  // 在指定的插值维度上设置步长为0，实现重叠计算
  strides[interp_dim] = 0;
  auto restrided_input = input.as_strided(shape, strides);

  // 获取输入张量的标量类型
  auto input_scalar_type = input.scalar_type();

  // 初始化索引权重和精度信息
  std::vector<Tensor> indices_weights;
  unsigned int weights_precision = 0;
  int unused;

  // 如果输入张量为字节类型，支持双线性和双三次插值模式
  if (input_scalar_type == at::kByte) {
    // 对于uint8类型，特别处理以支持双线性和双三次插值
    TORCH_INTERNAL_ASSERT(F::interp_size == 2 || F::interp_size == 4);
    // 计算索引范围和权重，返回精度信息
    std::tie(indices_weights, unused, weights_precision) =
      F::compute_index_ranges_int16_weights(
        input.size(interp_dim), oshape[interp_dim],
        input.stride(interp_dim) * input.element_size(),
        input.dim(), interp_dim, align_corners, scales[interp_dim - 2],
        antialias);
    // 确保权重精度大于0
    TORCH_INTERNAL_ASSERT(weights_precision > 0);
  } else {
    // 初始化 indices_weights 变量，调用 F::compute_index_ranges_weights 函数计算索引范围权重
    indices_weights =
      F::compute_index_ranges_weights(
        input_scalar_type, input.size(interp_dim), oshape[interp_dim],
        input.stride(interp_dim) * input.element_size(),
        input.dim(), interp_dim, align_corners, scales[interp_dim - 2],
        antialias);
    }
    
    // 创建 TensorIteratorConfig 对象 config
    TensorIteratorConfig config;
    // 设置 config 对象的参数：关闭检查所有张量是否相同数据类型
    config.check_all_same_dtype(false)
        // 声明输入和输出的数据类型和设备
        .declare_static_dtype_and_device(input.scalar_type(), input.device())
        // 添加输出张量
        .add_output(output)
        // 添加常量输入 restrided_input
        .add_const_input(restrided_input);
    
    // 遍历 indices_weights 中的每个张量，将其作为 config 对象的常量输入
    for (auto& tensor : indices_weights) {
      config.add_const_input(tensor);
    }
    
    // 构建 Tensor 迭代器 iter
    auto iter = config.build();
    
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND 宏，针对浮点数类型和字节类型进行分派
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::Byte, iter.dtype(), "upsample_generic_Nd_aa", [&] {
          // 调用 cpu_upsample_generic_aa 函数进行通用多维反卷积或上采样
          cpu_upsample_generic_aa<scalar_t, is_horizontal>(iter, weights_precision);
        });
// 通用的可分离上采样插值核函数，适用于带有抗锯齿功能的N维情况。
// 当(dtype == uint8 and mode in ("bilinear", "bicubic"))时支持antialias=False，
// 在不支持AVX指令集的情况下作为这些设置的后备方案。
template <int out_ndims, typename scale_type, class F>
void separable_upsample_generic_Nd_kernel_impl(
    const Tensor& output,  // 输出张量
    const Tensor& input,   // 输入张量
    bool align_corners,    // 是否对齐角点
    const scale_type& scales,  // 缩放因子
    bool antialias) {      // 是否使用抗锯齿

  auto output_shape = output.sizes();  // 输出张量的形状
  auto input_shape = input.sizes();    // 输入张量的形状
  auto temp_oshape = input_shape.vec();  // 临时输出形状

  if (output_shape == input_shape) {  // 如果输出形状与输入形状相同
    output.copy_(input);  // 直接复制输入到输出
    return;
  }

  at::Tensor temp_output, temp_input = input;  // 临时输出和输入张量

  int interp_dim = 0;  // 初始化插值维度
  // 预计算单维度调整方法的数量，以避免将临时缓冲区复制到输出
  int num_single_dim_ops = 0;
  for (const auto i : c10::irange(out_ndims)) {
    interp_dim = 2 + out_ndims - 1 - i;
    if (output_shape[interp_dim] != input_shape[interp_dim]) {
      num_single_dim_ops += 1;
    }
  }

  // 在连续维度内进行上采样数据（水平重采样）
  interp_dim = 2 + out_ndims - 1;
  if (output_shape[interp_dim] != input_shape[interp_dim]) {

    num_single_dim_ops -= 1;
    if (num_single_dim_ops > 0) {
      temp_oshape[interp_dim] = output_shape[interp_dim];
      temp_output = at::empty(temp_oshape, input.options());  // 创建临时输出张量
    } else {
      temp_output = output;
    }

    // 调用单维度的通用可分离上采样核函数实现
    _separable_upsample_generic_Nd_kernel_impl_single_dim<
        out_ndims,
        scale_t,
        F,
        true>(
        temp_output, temp_input, interp_dim, align_corners, scales, antialias);
    temp_input = temp_output;
  }

  // 在连续维度之间进行上采样数据（垂直重采样）
  for (const auto i : c10::irange(1, out_ndims)) {
    interp_dim = 2 + out_ndims - 1 - i;
    if (output_shape[interp_dim] != input_shape[interp_dim]) {

      num_single_dim_ops -= 1;
      if (num_single_dim_ops > 0) {
        temp_oshape[interp_dim] = output_shape[interp_dim];
        temp_output = at::empty(temp_oshape, input.options());  // 创建临时输出张量
      } else {
        temp_output = output;
      }

      // 调用单维度的通用可分离上采样核函数实现
      _separable_upsample_generic_Nd_kernel_impl_single_dim<
          out_ndims,
          scale_t,
          F,
          false>(
          temp_output, temp_input, interp_dim, align_corners, scales, antialias);
      temp_input = temp_output;
    }
  }
}

// 1维最近邻上采样核函数的实现
void upsample_nearest1d_kernel_impl(
    const Tensor& output,   // 输出张量
    const Tensor& input,    // 输入张量
    std::optional<double> scales_w) {  // 缩放因子（宽度）

  // 调用通用N维上采样核函数的实现，使用最近邻插值助手
  upsample_generic_Nd_kernel_impl<1, scale_t, HelperInterpNearest>(
      output, input, false, {scales_w});
}

// 精确1维最近邻上采样核函数的实现
void _upsample_nearest_exact1d_kernel_impl(
    const Tensor& output,   // 输出张量
    const Tensor& input,    // 输入张量
    std::optional<double> scales_w) {  // 缩放因子（宽度）

  // 调用通用N维上采样核函数的实现，使用精确最近邻插值助手
  upsample_generic_Nd_kernel_impl<1, scale_t, HelperInterpNearestExact>(
    output, input, false, {scales_w});
}

// 用于判断是否使用矢量化核函数的条件，针对2维情况
    // 检查是否应调用矢量化内核，还是更一般的upsample_generic_Nd_kernel_impl()函数。
    // 目前，矢量化内核仅针对channels_last和C >= 4（形状为NCHW）进行了优化。
    // 对于许多用例（通常是图像或掩码调整大小，其中C < 4），实际上使用upsample_generic_Nd_kernel_impl()更快。
    // 此外，基准测试显示，这也取决于*输出*大小（output_H + output_W），无论是上采样还是下采样。
    // 当前的阈值128是通过基准测试确定的。
    return ((input.is_contiguous(at::MemoryFormat::ChannelsLast)) && (input.size(1) > 3)) || ((output.size(-2) + output.size(-1)) <= 128);
}

// 检查是否可以使用向量化的3D内核进行上采样
int _use_vectorized_kernel_cond_3d(
    // 类似于 _use_vectorized_kernel_cond_2d()，但用于3D重采样（例如视频）
    // 注意，与2D情况不同，这里不受输出尺寸小的额外开销的影响，因此在条件中没有128的阈值。
    const Tensor& output,
    const Tensor& input) {
      // 检查输入张量是否以ChannelsLast3d内存格式连续，并且第二个维度大小大于3
      return ((input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) && (input.size(1) > 3));
}


// 2D最近邻上采样的内核实现
void upsample_nearest2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  if (_use_vectorized_kernel_cond_2d(output, input)) {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf,
        input.scalar_type(), "upsample_nearest2d_channels_last", [&] {
      // 在ChannelsLast布局下，使用CPU实现的最近邻上采样
      cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_idx>(output, input, {scales_h, scales_w});
    });
  } else {
    // 使用通用的Nd维度上采样内核实现2D最近邻上采样
    upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpNearest>(
      output, input, false, {scales_h, scales_w});
  }
}

// 精确的2D最近邻上采样的内核实现
void _upsample_nearest_exact2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  if (_use_vectorized_kernel_cond_2d(output, input)) {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf, input.scalar_type(), "upsample_nearest2d_channels_last", [&] {
      // 在ChannelsLast布局下，使用CPU实现的精确最近邻上采样
      cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_exact_idx>(output, input, {scales_h, scales_w});
    });
  } else {
    // 使用通用的Nd维度上采样内核实现精确的2D最近邻上采样
    upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpNearestExact>(
      output, input, false, {scales_h, scales_w});
  }
}

// 3D最近邻上采样的内核实现
void upsample_nearest3d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  if (_use_vectorized_kernel_cond_3d(output, input)) {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf,
        input.scalar_type(), "upsample_nearest3d_channels_last", [&] {
      // 在ChannelsLast布局下，使用CPU实现的最近邻3D上采样
      cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_idx>(output, input, {scales_d, scales_h, scales_w});
    });
  } else {
    // 使用通用的Nd维度上采样内核实现3D最近邻上采样
    upsample_generic_Nd_kernel_impl<3, scale_t, HelperInterpNearest>(
      output, input, false, {scales_d, scales_h, scales_w});
  }
}

// 精确的3D最近邻上采样的内核实现
void _upsample_nearest_exact3d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  if (_use_vectorized_kernel_cond_3d(output, input)) {
    AT_DISPATCH_FLOATING_TYPES_AND3(kByte, kBFloat16, kHalf, input.scalar_type(), "upsample_nearest3d_channels_last", [&] {
      // 在ChannelsLast布局下，使用CPU实现的精确最近邻3D上采样
      cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_exact_idx>(output, input, {scales_d, scales_h, scales_w});
    });
  } else {
    // 使用通用的Nd维度上采样内核实现精确的3D最近邻上采样
    upsample_generic_Nd_kernel_impl<3, scale_t, HelperInterpNearestExact>(
      output, input, false, {scales_d, scales_h, scales_w});
  }
}
void upsample_linear1d_kernel_impl(
    const Tensor& output,                                // 输出张量
    const Tensor& input,                                 // 输入张量
    bool align_corners,                                  // 是否对齐角点
    std::optional<double> scales_w) {                    // 可选的水平缩放因子
  upsample_generic_Nd_kernel_impl<1, scale_t, HelperInterpLinear>(  // 调用通用的 N 维线性插值内核
    output, input, align_corners, {scales_w});           // 传递输出、输入张量及相关参数
}


void upsample_bilinear2d_kernel_impl_float(
    const Tensor& output,                                // 输出张量
    const Tensor& input,                                 // 输入张量
    bool align_corners,                                  // 是否对齐角点
    std::optional<double> scales_h,                      // 可选的垂直缩放因子
    std::optional<double> scales_w) {                    // 可选的水平缩放因子

  // 有关 _use_vectorized_kernel_cond_2d(output, input) 的注释，请参见上文。额外的条件是因为基准测试表明，
  // 当只有一个线程时，图像（C == 3）使用向量化内核比通用内核稍快。但对于掩模（C == 1），使用通用内核会更有优势。
  if ((_use_vectorized_kernel_cond_2d(output, input)) || (at::get_num_threads() == 1 && input.size(1) == 3)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "upsample_bilinear2d_channels_last", [&] {
      cpu_upsample_linear_channels_last<scalar_t, scale_t>(output, input, align_corners, {scales_h, scales_w});  // 使用 CPU 执行通道优先的双线性上采样
    });
  } else {
    upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(  // 调用通用的二维双线性插值内核
      output, input, align_corners, {scales_h, scales_w});           // 传递输出、输入张量及相关参数
  }
}

void upsample_bilinear2d_kernel_impl(
    const Tensor& output,                                // 输出张量
    const Tensor& input,                                 // 输入张量
    bool align_corners,                                  // 是否对齐角点
    std::optional<double> scales_h,                      // 可选的垂直缩放因子
    std::optional<double> scales_w) {                    // 可选的水平缩放因子

  if (input.dtype() == at::kByte){                       // 如果输入张量的数据类型是字节型
    #ifdef CPU_CAPABILITY_AVX2
      if (input.size(1) <= 4) {
        upsample_avx_bilinear_bicubic_uint8<scale_t, HelperInterpLinear>(input,
          output, align_corners, {scales_h, scales_w},
          /*antialias=*/false);                         // 使用 AVX2 指令集执行双线性或双三次上采样（无抗锯齿）
      } else {
        separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
          output, input, align_corners, {scales_h, scales_w},
          /*antialias=*/false);                         // 使用分离的 N 维通用内核执行双线性或双三次上采样（无抗锯齿）
      }
    #else  // CPU_CAPABILITY_AVX2
      separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
        output, input, align_corners, {scales_h, scales_w},
        /*antialias=*/false);                           // 使用分离的 N 维通用内核执行双线性或双三次上采样（无抗锯齿）
    #endif  // CPU_CAPABILITY_AVX2
  } else {
    upsample_bilinear2d_kernel_impl_float(output, input, align_corners, scales_h, scales_w);  // 否则使用浮点数实现的双线性插值
  }
}


void upsample_bilinear2d_aa_kernel_impl(
    const Tensor& output,                                // 输出张量
    const Tensor& input,                                 // 输入张量
    bool align_corners,                                  // 是否对齐角点
    std::optional<double> scales_h,                      // 可选的垂直缩放因子
    std::optional<double> scales_w) {                    // 可选的水平缩放因子
#ifdef CPU_CAPABILITY_AVX2
  if (input.dtype() == at::kByte && input.size(1) <= 4) {
    upsample_avx_bilinear_bicubic_uint8<scale_t, HelperInterpLinear>(
      input, output, align_corners, {scales_h, scales_w},
      /*antialias=*/true);                              // 使用 AVX2 指令集执行双线性或双三次上采样（有抗锯齿）
  } else {
    separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
        output, input, align_corners, {scales_h, scales_w},
        /*antialias=*/true);                            // 使用分离的 N 维通用内核执行双线性或双三次上采样（有抗锯齿）
  }
#else // CPU_CAPABILITY_AVX2
  // 如果不支持 AVX2 指令集，则使用通用的 Nd 分离上采样实现，针对 2 维数据，使用 HelperInterpLinear 插值方法
  separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
      output, input, align_corners, {scales_h, scales_w},
      /*antialias=*/true);
#endif // CPU_CAPABILITY_AVX2
}

void upsample_trilinear3d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  if ((_use_vectorized_kernel_cond_3d(output, input))) {
    // 如果输入和输出满足向量化条件，使用通用 Nd 上采样实现，针对 3 维数据，使用 HelperInterpLinear 插值方法
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "upsample_trilinear3d_channels_last", [&] {
      cpu_upsample_linear_channels_last<scalar_t, scale_t>(output, input, align_corners, {scales_d, scales_h, scales_w});
    });
  } else {
    // 否则使用通用 Nd 上采样实现，针对 3 维数据，使用 HelperInterpLinear 插值方法
    upsample_generic_Nd_kernel_impl<3, scale_t, HelperInterpLinear>(
      output, input, align_corners, {scales_d, scales_h, scales_w});
  }
}

void upsample_bicubic2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {

  if (input.dtype() == at::kByte){
    #ifdef CPU_CAPABILITY_AVX2
      // 如果支持 AVX2 指令集且输入张量的通道数小于等于 4，则使用 AVX2 加速的双线性和双三次插值方法
      upsample_avx_bilinear_bicubic_uint8<scale_t, HelperInterpCubic>(input,
        output, align_corners, {scales_h, scales_w},
        /*antialias=*/false);
    #else  // CPU_CAPABILITY_AVX2
      // 否则使用通用 Nd 分离上采样实现，针对 2 维数据，使用 HelperInterpCubic 插值方法
      separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
        output, input, align_corners, {scales_h, scales_w},
        /*antialias=*/false);
    #endif  // CPU_CAPABILITY_AVX2
  }
  else {
    // 如果输入张量不是字节类型，则使用通用 Nd 分离上采样实现，针对 2 维数据，使用 HelperInterpCubic 插值方法
    upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
      output, input, align_corners, {scales_h, scales_w});
  }
}

void upsample_bicubic2d_aa_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {

#ifdef CPU_CAPABILITY_AVX2
  if (input.dtype() == at::kByte && input.size(1) <= 4) {
    // 如果支持 AVX2 指令集且输入张量的通道数小于等于 4，则使用 AVX2 加速的双线性和双三次插值方法，进行抗锯齿处理
    upsample_avx_bilinear_bicubic_uint8<scale_t, HelperInterpCubic>(
      input, output, align_corners, {scales_h, scales_w},
      /*antialias=*/true);
  } else {
    // 否则使用通用 Nd 分离上采样实现，针对 2 维数据，使用 HelperInterpCubic 插值方法，进行抗锯齿处理
    separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
        output, input, align_corners, {scales_h, scales_w},
        /*antialias=*/true);
  }
#else // CPU_CAPABILITY_AVX2
  // 如果不支持 AVX2 指令集，则使用通用 Nd 分离上采样实现，针对 2 维数据，使用 HelperInterpCubic 插值方法，进行抗锯齿处理
  separable_upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
      output, input, align_corners, {scales_h, scales_w},
      /*antialias=*/true);
#endif // CPU_CAPABILITY_AVX2
}

template <
    typename scalar_t,
    typename scale_type,
    class F>
void cpu_upsample_genNd_backward_aa(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    bool align_corners,
    const scale_type& scales) {
  // 检查梯度输入和梯度输出的数据类型是否一致
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  // 将梯度输出进行连续化处理
  auto grad_output = grad_output_.contiguous();
  // 将梯度输入进行连续化处理
  auto grad_input = grad_input_.contiguous();

  // 获取梯度输出和梯度输入的数据指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  // 获取梯度输入和梯度输出的尺寸
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();
  auto ndim = input_sizes.size();

  // 将 nbatch 和 channels 视为一个维度
  int64_t channels = input_sizes[0] * input_sizes[1];
  // 根据输入尺寸的维度数确定输出的深度
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  // 确定输入的高度和输出的高度
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  // 确定输入和输出的宽度
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  // 计算输出切片的大小
  int64_t output_slice_size = output_depth * output_height * output_width;
  // 获取插值大小
  int interp_size = F::interp_size;

  // 定义二维循环处理函数
  auto loop2d = [&](int64_t begin, int64_t end) {
    // 计算高度和宽度的缩放比例
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    // 定义输入索引函数
    auto input_indexr = [=](int64_t c, int64_t h, int64_t w) {
      return grad_input_data + c * input_height * input_width +
          h * input_width + w;
    };

    // 计算支持的高度和宽度
    const scalar_t support_h = (height_scale >= 1.0)
        ? (interp_size * 0.5) * height_scale
        : interp_size * 0.5;
    const scalar_t support_w = (width_scale >= 1.0)
        ? (interp_size * 0.5) * width_scale
        : interp_size * 0.5;

    // 计算插值的高度和宽度
    const int interp_height = (int)ceilf(support_h) * 2 + 1;
    const int interp_width = (int)ceilf(support_w) * 2 + 1;

    // 创建存储权重的向量 wx 和 wy
    std::vector<scalar_t> wx(interp_width, 0.0);
    std::vector<scalar_t> wy(interp_height, 0.0);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t xmin, ymin;
    int64_t xsize, ysize;

    // 定义滤波器函数指针
    typedef scalar_t (*aa_filter_fn_t)(scalar_t);
    aa_filter_fn_t filter_fn = &F::aa_filter;
    // 遍历输出图像的每一个高度位置 oh
    for (const auto oh : c10::irange(output_height)) {
      // 调用函数计算垂直方向上的插值参数和权重
      F::_compute_indices_min_size_weights_aa(
          oh,
          input_height,
          height_scale,
          support_h,
          wy.data(),
          interp_height,
          filter_fn,
          ymin,
          ysize);

      // 遍历输出图像的每一个宽度位置 ow
      for (const auto ow : c10::irange(output_width)) {
        // 调用函数计算水平方向上的插值参数和权重
        F::_compute_indices_min_size_weights_aa(
            ow,
            input_width,
            width_scale,
            support_w,
            wx.data(),
            interp_width,
            filter_fn,
            xmin,
            xsize);

        // 遍历通道索引范围内的每一个通道 c
        for (const auto c : c10::irange(begin, end)) {
          // 计算梯度输出的值，用于后续的插值计算
          scalar_t grad_output_value =
              grad_output_data[c * output_slice_size + oh * output_width + ow];

          // 遍历 ysize 和 xsize 区域内的每一个像素 y 和 x
          for (const auto y : c10::irange(ysize)) {
            for (const auto x : c10::irange(xsize)) {
              // 将插值权重应用于输入图像的像素值，并加到输入梯度中
              *input_indexr(c, ymin + y, xmin + x) +=
                  wx[x] * wy[y] * grad_output_value;
            }
          }
        }
      }
    }
  };

  // 如果输入张量是四维的
  if (ndim == 4) {
    // 使用双线性插值方法并行处理二维数据
    at::parallel_for(
        0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // 如果张量维度不支持，抛出错误信息
    TORCH_CHECK(false, "Unsupported tensor ndim");
  }

  // 如果梯度输入张量不是连续的，进行拷贝操作使其连续
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
// 定义一个匿名命名空间，用于限定内部函数和变量的作用域
} // anonymous namespace

// 注册向量化分发函数，将最近邻插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(upsample_nearest1d_kernel, &upsample_nearest1d_kernel_impl);

// 注册向量化分发函数，将精确最近邻插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(_upsample_nearest_exact1d_kernel, &_upsample_nearest_exact1d_kernel_impl);

// 注册向量化分发函数，将二维最近邻插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(upsample_nearest2d_kernel, &upsample_nearest2d_kernel_impl);

// 注册向量化分发函数，将精确二维最近邻插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(_upsample_nearest_exact2d_kernel, &_upsample_nearest_exact2d_kernel_impl);

// 注册向量化分发函数，将三维最近邻插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(upsample_nearest3d_kernel, &upsample_nearest3d_kernel_impl);

// 注册向量化分发函数，将一维线性插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(upsample_linear1d_kernel, &upsample_linear1d_kernel_impl);

// 注册向量化分发函数，将二维双线性插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(upsample_bilinear2d_kernel, &upsample_bilinear2d_kernel_impl);

// 注册向量化分发函数，将精确二维双线性插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(_upsample_bilinear2d_aa_kernel, &upsample_bilinear2d_aa_kernel_impl);

// 注册向量化分发函数，将双线性插值的反向处理函数与其实现函数关联起来
REGISTER_DISPATCH(_upsample_bilinear2d_aa_backward_kernel, &upsample_bilinear2d_aa_backward_kernel_impl);

// 注册向量化分发函数，将三维三线性插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(upsample_trilinear3d_kernel, &upsample_trilinear3d_kernel_impl);

// 注册向量化分发函数，将二维双三次插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(upsample_bicubic2d_kernel, &upsample_bicubic2d_kernel_impl);

// 注册向量化分发函数，将精确二维双三次插值的处理函数与其实现函数关联起来
REGISTER_DISPATCH(_upsample_bicubic2d_aa_kernel, &upsample_bicubic2d_aa_kernel_impl);

// 注册向量化分发函数，将双三次插值的反向处理函数与其实现函数关联起来
REGISTER_DISPATCH(_upsample_bicubic2d_aa_backward_kernel, &upsample_bicubic2d_aa_backward_kernel_impl);

// 定义命名空间 `at::native`，包含了 PyTorch 内核的本地实现
namespace at::native {
```