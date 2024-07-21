# `.\pytorch\aten\src\ATen\native\quantized\cpu\kernels\QuantizedOpKernels.cpp`

```
// 定义宏，仅包含 Torch 断言方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入 ATen 库中的各种头文件
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TopKImpl.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/IndexKernelUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/FakeQuantAffine.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <ATen/native/cpu/utils.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含 ATen/Functions.h 头文件，否则包含特定的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#endif

#include <cmath>  // 包含数学函数库
#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>  // 如果定义了 USE_FBGEMM，则包含 FBGEMM 的量化工具头文件
#endif
#ifdef _OPENMP
#include <omp.h>  // 如果定义了 _OPENMP，则包含 OpenMP 头文件
#endif
#if defined(__ARM_NEON__) || defined(__aarch64__)
#include <ATen/quantized/Quantizer.h>
#include <arm_neon.h>  // 如果定义了 ARM NEON 或 aarch64，包含 ARM NEON 头文件
#endif

namespace at {
namespace native {
namespace {

// 检查张量内存格式是否与参考张量匹配
void check_tensor_memory_format(const Tensor& ref, const Tensor& other) {
  TORCH_CHECK(
      ref.is_contiguous(ref.suggest_memory_format()),
      "Quantized tensor should be contiguous");
  TORCH_CHECK(
      other.is_contiguous(ref.suggest_memory_format()),
      "Float tensor should be contiguous "
      "in same memory format as quantized tensor");
}

// ****************** HEY YOU! YES YOU! Read this! ********************
//
// Please read the README.md in this directory before editing this file

// qcat_nhwc_kernel 模板函数，用于在 NHWC 格式下拼接量化张量
template <bool ReLUFused = false>
Tensor qcat_nhwc_kernel(
    const MaterializedITensorListRef& qxs,  // 输入量化张量列表的引用
    int64_t dim,  // 拼接维度
    double scale,  // 量化参数：缩放因子
    int64_t zero_point) {  // 量化参数：零点

  const at::Tensor& qx0 = qxs[0];  // 获取第一个量化张量作为参考
  int64_t C_out = 0;  // 输出通道数初始化为 0
  std::vector<int64_t> Cs_in;  // 输入通道数的向量
  std::vector<int64_t> Cs_sum;  // 输入通道数的前缀和向量
  std::vector<double> scales;  // 各量化张量的缩放因子向量
  std::vector<int64_t> zero_pts;  // 各量化张量的零点向量
  std::vector<void*> data_ptrs;  // 数据指针向量
  std::vector<bool> is_fast_path;  // 快速路径标记向量

  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const at::Tensor& qx : qxs) {  // 遍历每个量化张量
    TORCH_CHECK(
        qx.dim() == qx0.dim(),  // 检查张量维度是否相同
        "Tensors must have the same number of dimensions: got ",
        qx.dim(),
        " and ",
        qx0.dim());

    // 定义宏，检查指定维度的大小是否与第一个张量相同
#define CHECK_DIM(d)                                            \
  TORCH_CHECK(                                                  \
      qx.size(d) == qx0.size(d),                                \
      "Sizes of tensors must match expect in dimension 1. Got", \
      qx.size(d),                                               \
      " and ",                                                  \
      qx0.size(d));
    CHECK_DIM(0);  // 检查第一维度
    CHECK_DIM(2);  // 检查第三维度
    CHECK_DIM(3);  // 检查第四维度
    // 检查 qx 和 qx0 的标量类型是否一致，如果不一致则抛出异常信息
    TORCH_CHECK(
        qx.scalar_type() == qx0.scalar_type(),
        "Expected object of scalar type ",
        toString(qx0.scalar_type()),
        " but got scalar type ",
        toString(qx.scalar_type()));

    // 将 qx 的第二维大小添加到 Cs_in 向量末尾
    Cs_in.push_back(qx.size(1));

    // 将 C_out 添加到 Cs_sum 向量末尾，然后更新 C_out 为当前 qx 的第二维大小
    Cs_sum.push_back(C_out);
    C_out += qx.size(1);

    // 将 qx 的量化比例因子添加到 scales 向量末尾
    scales.push_back(qx.q_scale());

    // 将 qx 的零点偏移添加到 zero_pts 向量末尾
    zero_pts.push_back(qx.q_zero_point());

    // 将 qx 的数据指针添加到 data_ptrs 向量末尾
    data_ptrs.push_back(qx.data_ptr());

    // 检查是否符合快速路径的条件，将结果添加到 is_fast_path 向量末尾
    is_fast_path.push_back(
        qx.q_scale() == scale &&
        qx.q_zero_point() == zero_point);
  }

  // 定义并计算 N、H、W 的值，分别为 qx0 的第0、2、3维的大小
  const int64_t N = qx0.size(0);
  const int64_t H = qx0.size(2);
  const int64_t W = qx0.size(3);

  // 计算 scale 的倒数，并将结果保存到 inv_scale 变量中
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float inv_scale = 1.0 / scale;

  // 创建一个与 qx0 具有相同大小和格式的空张量 output
  auto output = at::_empty_affine_quantized(
      {N, C_out, H, W},  // 输出张量的形状为 {N, C_out, H, W}
      qx0.options().memory_format(MemoryFormat::ChannelsLast),  // 使用 ChannelsLast 内存格式
      scale,  // 输出张量的量化比例因子
      zero_point,  // 输出张量的零点偏移
      c10::nullopt);  // 不使用任何其他选项

  // 在以下代码中显式捕获 N、H 和 W，因为在 GCC5 和 clang5 中有一个内部编译器错误
  // 如果不显式捕获它们，会导致内部编译器错误
  AT_DISPATCH_QINT_TYPES(output.scalar_type(), "qcat_nhwc", [&, N, H, W]() {
    using Vec = Vectorized<scalar_t>;  // 定义一个使用 scalar_t 类型的向量化操作
    });
  });

  // 返回创建的输出张量 output
  return output;
}

// horizontal sum over a range of uint8_t
int64_t hsum(const uint8_t* A, int len) {
  int64_t row_sum = 0;  // 初始化行总和为0
  int i = 0;  // 初始化循环变量 i 为0

#ifdef CPU_CAPABILITY_AVX2
  __m256i sum_v = _mm256_setzero_si256();  // 使用 AVX2，初始化一个256位整数全零向量
  __m256i one_epi16_v = _mm256_set1_epi16(1);  // 初始化一个256位整数向量，每个元素为16位，全为1
  __m256i one_epi8_v = _mm256_set1_epi8(1);  // 初始化一个256位整数向量，每个元素为8位，全为1
  // 使用向量化处理
  for (; i < len / 32 * 32; i += 32) {
    __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));  // 加载未对齐的256位整数向量
    sum_v = _mm256_add_epi32(
      sum_v,
      _mm256_madd_epi16(
        _mm256_maddubs_epi16(src_v, one_epi8_v),  // 进行16位乘法和32位加法运算
        one_epi16_v)
    );
  }

  alignas(64) int32_t temp[8];  // 以64字节对齐的临时数组
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);  // 将256位整数向量存储到临时数组中
  for (const auto k : c10::irange(8)) {  // 循环遍历临时数组
    row_sum += temp[k];  // 累加到行总和中
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_v = _mm512_setzero_si512();  // 使用 AVX512，初始化一个512位整数全零向量
  __m512i one_epi16_v = _mm512_set1_epi16(1);  // 初始化一个512位整数向量，每个元素为16位，全为1
  __m512i one_epi8_v = _mm512_set1_epi8(1);  // 初始化一个512位整数向量，每个元素为8位，全为1
  // 使用向量化处理
  for (; i < len / 64 * 64; i += 64) {
    __m512i src_v = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));  // 加载未对齐的512位整数向量
    sum_v = _mm512_add_epi32(
      sum_v,
      _mm512_madd_epi16(
        _mm512_maddubs_epi16(src_v, one_epi8_v),  // 进行16位乘法和32位加法运算
        one_epi16_v)
    );
  }

  alignas(64) int32_t temp[16];  // 以64字节对齐的临时数组
  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v);  // 将512位整数向量存储到临时数组中
  for (const auto k : c10::irange(16)) {  // 循环遍历临时数组
    row_sum += temp[k];  // 累加到行总和中
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar，标量处理剩余的部分
  for (; i < len; ++i) {
    row_sum += A[i];  // 普通循环累加到行总和中
  }

  return row_sum;  // 返回行总和
}

// horizontal sum over a range of int8_t
int64_t hsum(const int8_t* A, int len) {
  int64_t row_sum = 0;  // 初始化行总和为0
  int i = 0;  // 初始化循环变量 i 为0

#ifdef CPU_CAPABILITY_AVX2
  __m256i sum_v = _mm256_setzero_si256();  // 使用 AVX2，初始化一个256位整数全零向量
  __m256i one_epi16_v = _mm256_set1_epi16(1);  // 初始化一个256位整数向量，每个元素为16位，全为1
  __m256i one_epi8_v = _mm256_set1_epi8(1);  // 初始化一个256位整数向量，每个元素为8位，全为1
  // 使用向量化处理
  for (; i < len / 32 * 32; i += 32) {
    __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));  // 加载未对齐的256位整数向量
    sum_v = _mm256_add_epi32(
      sum_v,
      _mm256_madd_epi16(
        _mm256_maddubs_epi16(one_epi8_v, src_v),  // 进行16位乘法和32位加法运算
        one_epi16_v)
    );
  }

  alignas(64) int32_t temp[8];  // 以64字节对齐的临时数组
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);  // 将256位整数向量存储到临时数组中
  for (const auto k : c10::irange(8)) {  // 循环遍历临时数组
    row_sum += temp[k];  // 累加到行总和中
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_v = _mm512_setzero_si512();  // 使用 AVX512，初始化一个512位整数全零向量
  __m512i one_epi16_v = _mm512_set1_epi16(1);  // 初始化一个512位整数向量，每个元素为16位，全为1
  __m512i one_epi8_v = _mm512_set1_epi8(1);  // 初始化一个512位整数向量，每个元素为8位，全为1
  // 使用向量化处理
  for (; i < len / 64 * 64; i += 64) {
    __m512i src_v = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));  // 加载未对齐的512位整数向量
    sum_v = _mm512_add_epi32(
      sum_v,
      _mm512_madd_epi16(
        _mm512_maddubs_epi16(one_epi8_v, src_v),  // 进行16位乘法和32位加法运算
        one_epi16_v)
    );
  }

  alignas(64) int32_t temp[16];  // 以64字节对齐的临时数组
  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v);  // 将512位整数向量存储到临时数组中
  for (const auto k : c10::irange(16)) {  // 循环遍历临时数组
    row_sum += temp[k];  // 累加到行总和中
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar，标量处理剩余的部分
  for (; i < len; ++i) {
    row_sum += A[i];  // 普通循环累加到行总和中
  }

  return row_sum;  // 返回行总和
}
  );



// 这行代码是一个语法错误，缺少了前面代码的上下文，无法准确解释其作用。



  }

  alignas(64) int32_t temp[16];



// 以64字节对齐方式声明一个包含16个int32_t元素的数组temp
alignas(64) int32_t temp[16];



  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v);



// 使用512位的存储指令将sum_v中的数据存储到temp数组中
_mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v);



  for (const auto k : c10::irange(16)) {



// 使用范围遍历循环，遍历从0到15的整数
for (const auto k : c10::irange(16)) {



    row_sum += temp[k];
  }



// 将temp数组中索引为k的元素累加到变量row_sum中
row_sum += temp[k];



// 以上为给定代码的注释
// scalar
for (; i < len; ++i) {
  row_sum += A[i];
}
// 对数组 A 中的元素进行逐个求和，存储在 row_sum 中

return row_sum;
}

// horizontal sum over a range of int32_t
int64_t hsum(const int32_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
__m256i sum_epi64 = _mm256_setzero_si256();
// 使用 AVX2 进行向量化求和
for (; i < len / 8 * 8; i += 8) {
  __m256i src_epi32 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
  // 加载未对齐的 int32_t 数据到 AVX2 寄存器
  __m128i src_lo_epi32 = _mm256_castsi256_si128(src_epi32);
  __m128i src_hi_epi32 = _mm256_extracti128_si256(src_epi32, 1);
  __m256i src_lo_epi64 = _mm256_cvtepi32_epi64(src_lo_epi32);
  __m256i src_hi_epi64 = _mm256_cvtepi32_epi64(src_hi_epi32);
  // 扩展 int32_t 到 int64_t，然后进行加法操作
  sum_epi64 = _mm256_add_epi64(sum_epi64, src_lo_epi64);
  sum_epi64 = _mm256_add_epi64(sum_epi64, src_hi_epi64);
}

alignas(64) int64_t temp[4];
_mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_epi64);
for (const auto k : c10::irange(4)) {
  row_sum += temp[k];
}
#elif defined(CPU_CAPABILITY_AVX512)
__m512i sum_epi64 = _mm512_setzero_si512();
// 使用 AVX512 进行向量化求和
for (; i < len / 16 * 16; i += 16) {
  __m512i src_epi32 = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
  // 加载未对齐的 int32_t 数据到 AVX512 寄存器
  __m256i src_lo_epi32 = _mm512_castsi512_si256(src_epi32);
  __m256i src_hi_epi32 = _mm512_extracti32x8_epi32(src_epi32, 1);
  __m512i src_lo_epi64 = _mm512_cvtepi32_epi64(src_lo_epi32);
  __m512i src_hi_epi64 = _mm512_cvtepi32_epi64(src_hi_epi32);
  // 扩展 int32_t 到 int64_t，然后进行加法操作
  sum_epi64 = _mm512_add_epi64(sum_epi64, src_lo_epi64);
  sum_epi64 = _mm512_add_epi64(sum_epi64, src_hi_epi64);
}

alignas(64) int64_t temp[8];
_mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_epi64);
for (const auto k : c10::irange(8)) {
  row_sum += temp[k];
}
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

// scalar
for (; i < len; ++i) {
  row_sum += A[i];
}
// 对剩余的元素进行标量求和操作

return row_sum;
}

// horizontal sum of squares over a range of uint8_t
int64_t hsum_sq(const uint8_t* A, int len) {
int64_t row_sum = 0;
int i = 0;

#ifdef CPU_CAPABILITY_AVX2
// vectorized
__m256i sum_v_epu32 = _mm256_setzero_si256();
alignas(64) int32_t temp[8];
int overflow_threshold = 262144; // 2147483647(max of int32)/(256*256)*8 = 262144
int loop = len / overflow_threshold + 1;
for(int j=0; j<=loop; j++){
    // 对于每个 i，直到达到溢出阈值乘以 j 或者数据长度的 1/16 处的整数倍，每次增加 16
    for (; ((i < overflow_threshold * j) && (i < len / 16 * 16)); i += 16) {
      // 加载从 A[i] 开始的 16 个字节到 src_epu8 中作为 128 位整数
      __m128i src_epu8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(A + i));
      // 将 src_epu8 转换为 256 位整数，每个字节扩展为 16 位
      __m256i src_epu16 = _mm256_cvtepu8_epi16(src_epu8);
      // 计算 src_epu16 中每个元素的平方，结果存储在 sq_epu16 中
      __m256i sq_epu16 = _mm256_mullo_epi16(src_epu16, src_epu16);
      // 提取 sq_epu16 的低 128 位到 sq_lo_epu16 中
      __m128i sq_lo_epu16 = _mm256_castsi256_si128(sq_epu16);
      // 提取 sq_epu16 的高 128 位到 sq_hi_epu16 中
      __m128i sq_hi_epu16 = _mm256_extractf128_si256(sq_epu16, 1);
      // 将 sq_lo_epu16 和 sq_hi_epu16 扩展为 256 位整数到 sq_lo_epu32 和 sq_hi_epu32 中
      __m256i sq_lo_epu32 = _mm256_cvtepu16_epi32(sq_lo_epu16);
      __m256i sq_hi_epu32 = _mm256_cvtepu16_epi32(sq_hi_epu16);
      // 将 sq_lo_epu32 和 sq_hi_epu32 加到 sum_v_epu32 中作为运行总和
      sum_v_epu32 = _mm256_add_epi32(sum_v_epu32, sq_lo_epu32);
      sum_v_epu32 = _mm256_add_epi32(sum_v_epu32, sq_hi_epu32);
    }
    // 将 sum_v_epu32 存储到 temp 中，作为临时数组
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v_epu32);
    // 对 temp 中的每个元素求和并添加到 row_sum 中
    for (const auto k : c10::irange(8)) {
      row_sum += temp[k];
    }
    // 将 sum_v_epu32 重置为全零
    sum_v_epu32 = _mm256_setzero_si256();
  }
#elif defined(CPU_CAPABILITY_AVX512)
  // AVX-512向量化计算
  __m512i sum_v_epu32 = _mm512_setzero_si512();
  // 为临时数组temp分配64字节对齐的存储空间
  alignas(64) int32_t temp[16];
  // 溢出阈值设定为262144，用于控制循环次数
  int overflow_threshold = 262144; // 2147483647(max of int32)/(512*512)*8 = 262144
  // 计算循环次数
  int loop = len / overflow_threshold + 1;
  // 循环执行直到达到指定的循环次数
  for(int j=0; j<=loop; j++){
    // 内层循环，每次处理32个元素
    for (; ((i < overflow_threshold * j) && (i < len / 32 * 32)); i += 32) {
      // 加载输入数据，将int8_t数组A中的数据转换为__m256i类型
      __m256i src_epu8 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
      // 将__m256i类型的数据转换为__m512i类型
      __m512i src_epu16 = _mm512_cvtepu8_epi16(src_epu8);
      // 计算平方
      __m512i sq_epu16 = _mm512_mullo_epi16(src_epu16, src_epu16);
      // 将结果转换为__m256i类型的低128位
      __m256i sq_lo_epu16 = _mm512_castsi512_si256(sq_epu16);
      // 将结果转换为__m256i类型的高128位
      __m256i sq_hi_epu16 = _mm512_extracti32x8_epi32(sq_epu16, 1);
      // 将__m256i类型的数据转换为__m512i类型
      __m512i sq_lo_epu32 = _mm512_cvtepu16_epi32(sq_lo_epu16);
      __m512i sq_hi_epu32 = _mm512_cvtepu16_epi32(sq_hi_epu16);
      // 将计算结果累加到sum_v_epu32中
      sum_v_epu32 = _mm512_add_epi32(sum_v_epu32, sq_lo_epu32);
      sum_v_epu32 = _mm512_add_epi32(sum_v_epu32, sq_hi_epu32);
    }
    // 将sum_v_epu32中的数据存储到temp数组中
    _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v_epu32);
    // 将temp数组中的值累加到row_sum中
    for (const auto k : c10::irange(16)) {
      row_sum += temp[k];
    }
    // 将sum_v_epu32重置为零向量
    sum_v_epu32 = _mm512_setzero_si512();
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar标量计算
  // 处理剩余的数据（如果有）
  for (; i < len; ++i) {
    row_sum += A[i] * A[i];
  }

  // 返回结果
  return row_sum;
}

// horizontal sum of squares over a range of int8_t
// 计算int8_t类型数组A中指定范围内元素的平方和
int64_t hsum_sq(const int8_t* A, int len) {
  // 初始化行和为0
  int64_t row_sum = 0;
  // 初始化索引i为0
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  // AVX2向量化计算
  __m256i sum_v_epi32 = _mm256_setzero_si256();
  // 为临时数组temp分配64字节对齐的存储空间
  alignas(64) int32_t temp[8];
  // 溢出阈值设定为1048576，用于控制循环次数
  int overflow_threshold = 1048576; //2147483647/(128*128)*8 = 1048576
  // 计算循环次数
  int loop = len / overflow_threshold + 1;

  // 循环执行直到达到指定的循环次数
  for(int j=0; j<=loop; j++){
    // 内层循环，每次处理16个元素
    for (; ((i < overflow_threshold * j) && (i < len / 16 * 16)); i += 16) {
      // 加载输入数据，将int8_t数组A中的数据转换为__m128i类型
      __m128i src_epi8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(A + i));
      // 将__m128i类型的数据转换为__m256i类型
      __m256i src_epi16 = _mm256_cvtepi8_epi16(src_epi8);
      // 计算平方
      __m256i sq_epi16 = _mm256_mullo_epi16(src_epi16, src_epi16);
      // 将结果转换为__m128i类型的低64位
      __m128i sq_lo_epi16 = _mm256_castsi256_si128(sq_epi16);
      // 将结果转换为__m128i类型的高64位
      __m128i sq_hi_epi16 = _mm256_extractf128_si256(sq_epi16, 1);
      // 将__m128i类型的数据转换为__m256i类型
      __m256i sq_lo_epi32 = _mm256_cvtepi16_epi32(sq_lo_epi16);
      __m256i sq_hi_epi32 = _mm256_cvtepi16_epi32(sq_hi_epi16);
      // 将计算结果累加到sum_v_epi32中
      sum_v_epi32 = _mm256_add_epi32(sum_v_epi32, sq_lo_epi32);
      sum_v_epi32 = _mm256_add_epi32(sum_v_epi32, sq_hi_epi32);
    }
    // 将sum_v_epi32中的数据存储到temp数组中
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v_epi32);
    // 将temp数组中的值累加到row_sum中
    for (const auto k : c10::irange(8)) {
      row_sum += temp[k];
    }
    // 将sum_v_epi32重置为零向量
    sum_v_epi32 = _mm256_setzero_si256();
  }
#elif defined(CPU_CAPABILITY_AVX512)
  // vectorized
  __m512i sum_v_epi32 = _mm512_setzero_si512();
  alignas(64) int32_t temp[16];

  // 定义溢出阈值
  int overflow_threshold = 1048576; // 2147483647 / (256 * 256) * 8 = 1048576
  // 计算循环次数
  int loop = len / overflow_threshold + 1;

  for(int j=0; j<=loop; j++){
    // 对于每个子区间，处理长度为32的数据块
    for (; ((i < overflow_threshold * j) && (i < len / 32 * 32)); i += 32) {
      // 加载数据块中的8个字节为256位整数
      __m256i src_epi8 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
      // 将256位整数扩展为512位整数
      __m512i src_epi16 = _mm512_cvtepi8_epi16(src_epi8);
      // 计算每个16位元素的平方，并存储在512位整数中
      __m512i sq_epi16 = _mm512_mullo_epi16(src_epi16, src_epi16);
      // 将512位整数拆分为两个256位整数
      __m256i sq_lo_epi16 = _mm512_castsi512_si256(sq_epi16);
      __m256i sq_hi_epi16 = _mm512_extracti32x8_epi32(sq_epi16, 1);
      // 将16位整数扩展为32位整数
      __m512i sq_lo_epi32 = _mm512_cvtepi16_epi32(sq_lo_epi16);
      __m512i sq_hi_epi32 = _mm512_cvtepi16_epi32(sq_hi_epi16);
      // 将计算结果累加到总和中
      sum_v_epi32 = _mm512_add_epi32(sum_v_epi32, sq_lo_epi32);
      sum_v_epi32 = _mm512_add_epi32(sum_v_epi32, sq_hi_epi32);
    }
    // 将累加结果存储到临时数组中
    _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v_epi32);

    // 对临时数组中的每个元素进行水平求和
    for (const auto k : c10::irange(16)) {
      row_sum += temp[k];
    }
    // 重置累加器
    sum_v_epi32 = _mm512_setzero_si512();
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar，处理剩余部分的数据（长度不足32的数据）
  for (; i < len; ++i) {
    row_sum += A[i] * A[i];
  }

  return row_sum;
}

// 对一段int32_t类型的数据进行平方后的水平求和
float hsum_sq(const int32_t* A, int len) {
  float row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256 sum_ps = _mm256_setzero_ps();
  // 使用AVX2进行向量化计算
  for (; i < len / 8 * 8; i += 8) {
    // 加载8个整数到256位整数寄存器中
    __m256i src_epi32 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    // 将整数转换为单精度浮点数，并计算平方
    __m256 src_ps = _mm256_cvtepi32_ps(src_epi32);
    sum_ps = _mm256_add_ps(sum_ps, _mm256_mul_ps(src_ps, src_ps));
  }

  // 将256位浮点数寄存器中的结果存储到临时数组中
  alignas(64) float temp[8];
  _mm256_store_ps(temp, sum_ps);
  // 对临时数组中的每个元素进行累加
  for (const auto k : c10::irange(8)) {
    row_sum += static_cast<float>(temp[k]);
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512 sum_ps = _mm512_setzero_ps();
  // 使用AVX512进行向量化计算
  for (; i < len / 16 * 16; i += 16) {
    // 加载16个整数到512位整数寄存器中
    __m512i src_epi32 = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    // 将整数转换为单精度浮点数，并计算平方
    __m512 src_ps = _mm512_cvtepi32_ps(src_epi32);
    sum_ps = _mm512_add_ps(sum_ps, _mm512_mul_ps(src_ps, src_ps));
  }

  // 将512位浮点数寄存器中的结果存储到临时数组中
  alignas(64) float temp[16];
  _mm512_store_ps(temp, sum_ps);
  // 对临时数组中的每个元素进行累加
  for (const auto k : c10::irange(16)) {
    row_sum += static_cast<float>(temp[k]);
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar，处理剩余部分的数据（长度不足16的数据）
  for (; i < len; ++i) {
    int64_t cur = static_cast<int64_t>(A[i]);
    row_sum += static_cast<float>(cur) * static_cast<float>(cur);
  }

  return row_sum;
}

void qrelu_kernel(const Tensor& qx, Tensor& qy) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    // 创建一个空的量化张量 `qy`，其形状与输入张量 `qx` 相同
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        // 使用 `qx` 的建议内存格式，CPU 设备，标量类型 `SCALAR_TYPE`，以及 `qx` 的量化参数创建内存格式
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        c10::nullopt);
    // 定义一个名为 `Vec` 的类型别名，表示矢量化的标量类型 `scalar_t`
    using Vec = Vectorized<scalar_t>;
    // 创建一个矢量化的零点 `zero_point_vec`，初始化为 `zero_point` 的值
    auto zero_point_vec = Vec(scalar_t(zero_point));
    // 创建一个张量迭代器 `iter`，用于对 `qy` 和 `qx` 进行单目操作
    auto iter = TensorIterator::unary_op(qy, qx);
    // 在 CPU 上执行矢量化的内核操作，对每个标量执行指定的操作
    cpu_kernel_vec(
        iter,
        // 对每个标量值执行操作，返回最大值与 `zero_point` 中较大的值
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        },
        // 对每个矢量值执行 ReLU 操作，与 `zero_point_vec` 进行比较
        [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
  });
}

static void leaky_qrelu_out_kernel(Tensor& out, const Tensor& qx,
                                   const Scalar& negval_) {
  int64_t i_zp = qx.q_zero_point();
  // 获取输入张量 qx 的量化比例因子
  float i_scale = qx.q_scale();

  int64_t o_zp = out.q_zero_point();
  // 获取输出张量 out 的量化零点
  float o_scale = out.q_scale();
  // 计算输出张量的倒数量化比例因子
  float o_inv_scale = 1.0f / o_scale;

  // 获取 negval 的 float 值
  float negval = negval_.to<float>();

  // 根据输出张量的数据类型调度相应的处理逻辑
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "leaky_qrelu", [&] {
    using Vec = Vectorized<float>;  // Naive implementation uses dequant/quant loop.
    using qVec = Vectorized<scalar_t>;
    // 初始化一些向量变量
    Vec zero_vec = Vec(0.0f);
    Vec one_vec = Vec(1.0f);
    
    // 创建输入张量的量化比例因子向量
    Vec i_scale_vec = Vec((float)i_scale);
    // 创建输入张量的量化零点向量
    Vec i_zp_vec = Vec((float)i_zp);
    // 计算量化比例因子和量化零点的负值乘积向量
    Vec i_scale_zp_neg_premul_vec = i_scale_vec * i_zp_vec.neg();
    
    // 创建 negval 的向量
    Vec negval_vec = Vec(negval);

    // 创建张量迭代器对象
    auto iter = TensorIterator::unary_op(out, qx);

    // 调用 CPU 核函数处理向量化计算
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          // 反量化输入值 value_qx
          auto value_dx = at::native::dequantize_val(i_scale, i_zp, value_qx);
          // 执行 leaky ReLU 激活函数
          auto value_dy = value_dx > 0 ? value_dx : value_dx * negval;
          // 量化输出值 value_dy
          return at::native::quantize_val<scalar_t>(o_scale, o_zp, value_dy);
        },
        [&](qVec qx_vec) -> qVec {
          /* 向量化实现创建一个乘法因子向量，对所有负值的 dx 值使用 alpha，对所有正值的 dx 值使用 ones 向量。
           * 然后将乘法因子应用于输入。
           */
          // 反量化输入张量的向量
          auto dx_vec_vec = qx_vec.dequantize(i_scale_vec, i_zp_vec,
                                              i_scale_zp_neg_premul_vec);
          for (auto & dx_vec : dx_vec_vec) {
            // 使用 blendv 函数创建一个乘法因子
            const auto multiplicand = Vec::blendv(negval_vec, one_vec,
                                                  dx_vec > zero_vec);
            dx_vec *= multiplicand;
          }
          // 量化输出向量
          return qVec::quantize(dx_vec_vec, o_scale, o_zp, o_inv_scale);
        });
  });
}

static void qprelu_out_kernel(Tensor& out,
                              const Tensor& qx,
                              const Tensor& qw) {
  int32_t i_zp = static_cast<int32_t>(qx.q_zero_point());
  float i_scale = static_cast<float>(qx.q_scale());

  int32_t w_zp = static_cast<int32_t>(qw.q_zero_point());
  float w_scale = static_cast<float>(qw.q_scale());

  int32_t o_zp = static_cast<int32_t>(out.q_zero_point());
  float o_scale = static_cast<float>(out.q_scale());
  float o_inv_scale = 1.0f / o_scale;

  // 计算输入、权重和输出的比例因子乘积
  float multiplier = i_scale * w_scale * o_inv_scale;

  // 获取输入张量的维度
  int64_t input_ndim = qx.dim();
  // 检查输入张量的维度是否大于 0
  TORCH_CHECK(input_ndim > 0, "qprelu: zero-dim input tensor is not allowed.");

  // 如果输入张量和权重张量的维度不同，则进行调整
  auto qw_nd = qw;
  if (input_ndim != qw_nd.dim()) {
    // 创建一个维度为 input_ndim 的一维向量 dim_w，每个元素初始化为 1
    DimVector dim_w(input_ndim, 1);

    // 如果 input_ndim 大于 1，则将 dim_w 的第二个元素设置为 qw.numel() 的结果
    if (input_ndim > 1) {
      dim_w[1] = qw.numel();
    }

    // 根据 dim_w 对 qw_nd 进行形状重塑
    // 在 CPU/CUDA 中，这将始终是一个视图，但某些后端（如 MKLDNN）不支持视图
    qw_nd = qw_nd.reshape(dim_w);
  }

  // 配置张量迭代器，设置输出张量为 out，输入张量为 qx 和 qw_nd
  auto iter = TensorIteratorConfig()
    .add_output(out)
    .add_input(qx)
    .add_input(qw_nd)
    .build();

  // 在量化整数类型 scalar_t 下，执行 qprelu 操作
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qprelu", [&] {
    using qVec = Vectorized<scalar_t>;

    // 创建两个向量，分别用于 i_zp 和 w_zp
    qVec i_zp_vec = qVec(static_cast<scalar_t>(i_zp));
    qVec w_zp_vec = qVec(static_cast<scalar_t>(w_zp));

    // 使用 w_scale、w_zp 和浮点数 1.0f 进行量化，得到量化后的权重 qw_one
    auto qw_one = at::native::quantize_val<scalar_t>(w_scale, w_zp, 1.0f);
    qVec vec_qw_one = qVec(qw_one);

    // 计算 vec_qw_one 减去 w_zp_vec 的宽展减法结果中的第一个元素
    auto vec_qw_one_sub_zp = vec_qw_one.widening_subtract(w_zp_vec)[0];

    // 计算 qw_one 减去 w_zp 的结果，保存在 qw_one_sub_zp 中
    int32_t qw_one_sub_zp = qw_one.val_ - w_zp;

    // 使用 CPU 内核函数处理张量迭代器 iter
    cpu_kernel_vec(
      iter,
      // 标量版本的计算函数，对每对 qx 和 qw 进行计算，返回结果标量
      [=](scalar_t val_qx, scalar_t val_qw) -> scalar_t {
        // 计算 qx 的正值和负值与 i_zp 的最大和最小值
        int32_t qx_pos = std::max(static_cast<int32_t>(val_qx.val_), i_zp);
        int32_t qx_neg = std::min(static_cast<int32_t>(val_qx.val_), i_zp);

        // 计算 qx_pos 和 qx_neg 减去 i_zp 的结果
        int32_t qx_pos_sub_zp = qx_pos - i_zp;
        int32_t qx_neg_sub_zp = qx_neg - i_zp;

        // 计算 val_qw 减去 w_zp 的结果
        int32_t qw_sub_zp = val_qw.val_ - w_zp;

        // 计算最终的 qy_sub_zp，并使用 requantize_from_int 进行从整数的重新量化
        auto qy_sub_zp = qx_pos_sub_zp * qw_one_sub_zp + qx_neg_sub_zp * qw_sub_zp;
        return at::native::requantize_from_int<scalar_t>(
            multiplier, o_zp, qy_sub_zp);
      },
      // 向量化版本的计算函数，对每对 vec_qx 和 vec_qw 进行计算，返回结果向量
      [=](qVec vec_qx, qVec vec_qw) -> qVec {
        // 获取 vec_qx 中大于 i_zp_vec 的最大值和小于 i_zp_vec 的最小值
        auto vec_qx_pos = vec_qx.maximum(i_zp_vec);
        auto vec_qx_neg = vec_qx.minimum(i_zp_vec);

        // 计算 vec_qx_pos 和 vec_qx_neg 减去 i_zp_vec 的宽展减法结果
        qVec::int_vec_return_type qx_pos_sub_zp = vec_qx_pos.widening_subtract(i_zp_vec);
        qVec::int_vec_return_type qx_neg_sub_zp = vec_qx_neg.widening_subtract(i_zp_vec);

        // 计算 vec_qw 减去 w_zp_vec 的宽展减法结果
        qVec::int_vec_return_type qw_sub_zp = vec_qw.widening_subtract(w_zp_vec);

        // 初始化结果向量 qy_sub_zp
        qVec::int_vec_return_type qy_sub_zp;

        // 遍历 qVec::int_num_vecs() 范围，计算每个向量元素的 qy_sub_zp
        for (const auto i : c10::irange(qVec::int_num_vecs())) {
          qy_sub_zp[i] = qx_pos_sub_zp[i] * vec_qw_one_sub_zp + qx_neg_sub_zp[i] * qw_sub_zp[i];
        }

        // 对 qy_sub_zp 进行从整数的重新量化，并返回结果向量
        return qVec::requantize_from_int(qy_sub_zp, multiplier, o_zp);
      });
  });
}

// 定义函数 qgelu_kernel，实现 GELU 算法的量化版本
void qgelu_kernel(const Tensor& qx, Tensor& qy, GeluType approximate) {
  // 获取输入张量的零点
  int64_t zero_point = qx.q_zero_point();
  // 获取输入张量的量化比例因子
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float scale = qx.q_scale();
  // 创建一个包含 scale 值的向量化对象
  auto scale_vec = Vectorized<float>(scale);
  // 创建一个包含 zero_point 值的向量化对象
  auto zero_point_vec = Vectorized<float>((float)zero_point);
  // 计算 scale 乘以负的 zero_point 的预乘结果向量
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();
  // 设置输出的零点与量化比例因子等于输入的零点与量化比例因子
  int64_t output_zero_point = zero_point;
  float output_scale = scale;
  // 计算输出的逆量化比例因子
  float inv_output_scale = 1.0 / output_scale;
  // 创建常量向量 kAlphaVec，kBetaVec，kKappaVec，kOneVec 和 kPointFiveVec
  const auto kAlphaVec = Vectorized<float>(M_SQRT1_2);
  const auto kBetaVec = Vectorized<float>(M_SQRT2 * M_2_SQRTPI * 0.5);
  const auto kKappaVec = Vectorized<float>(0.044715);
  const auto kOneVec = Vectorized<float>(1);
  const auto kPointFiveVec = Vectorized<float>(0.5);

  // 如果选择使用近似为 Tanh 的 GELU 算法
  if (approximate == GeluType::Tanh) {
    // 根据输入张量类型的分发，执行以下计算
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qgelu", [&]() {
      // 创建一个与 qx 大小相同的空张量 qy，用于存储结果
      qy = at::_empty_affine_quantized(
          qx.sizes(),
          // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
          at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
          output_scale,
          output_zero_point,
          c10::nullopt);
      // 创建一个迭代器，用于逐元素操作 qy 和 qx
      auto iter = TensorIterator::unary_op(qy, qx);

      // 使用标量类型 scalar_t 和 Vectorized<scalar_t> 执行向量化 CPU 计算
      using Vec = Vectorized<scalar_t>;
      cpu_kernel_vec(
          iter,
          [&](scalar_t value_qx) -> scalar_t {
            // 对输入值进行反量化
            const auto value_dx =
                at::native::dequantize_val(scale, zero_point, value_qx);

            // 定义常量 kBeta 和 kKappa，并计算 GELU 函数的值
            const auto kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
            const auto kKappa = 0.044715;
            const auto x_cube = value_dx * value_dx * value_dx;
            const auto inner = kBeta * (value_dx + kKappa * x_cube);
            const auto value_dy = 0.5 * value_dx * (1.0 + std::tanh(inner));

            // 将结果量化并返回
            return at::native::quantize_val<scalar_t>(
                output_scale, output_zero_point, value_dy);
          },
          [&](Vec value_qx) -> Vec {
            // 对向量化输入进行反量化
            auto value_dx = value_qx.dequantize(
                scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
            // 对每个元素应用 GELU 函数的向量化计算
            for (auto & value : value_dx) {
              auto value_cube = value * value * value;
              auto inner = kBetaVec * (value + kKappaVec * value_cube);
              value = kPointFiveVec * value * (kOneVec + inner.tanh());
            }
            // 将结果向量量化并返回
            return Vec::quantize(
                value_dx, output_scale, output_zero_point, inv_output_scale);
          });
    });
  } else {
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qgelu", [&]() {
      // 使用 AT_DISPATCH_QINT_TYPES 宏处理量化整数类型，执行下面的 Lambda 表达式
      qy = at::_empty_affine_quantized(
          qx.sizes(),
          // 使用 qx 的推荐内存格式和标量类型创建一个空的量化张量
          at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
          output_scale,
          output_zero_point,
          c10::nullopt);
      // 创建 TensorIterator 用于遍历 qy 和 qx
      auto iter = TensorIterator::unary_op(qy, qx);

      using Vec = Vectorized<scalar_t>;
      // 调用 CPU 核心函数 cpu_kernel_vec 处理 TensorIterator
      cpu_kernel_vec(
          iter,
          [&](scalar_t value_qx) -> scalar_t {
            // 对每个 qx 的值进行反量化操作，得到 value_dx
            const auto value_dx =
                at::native::dequantize_val(scale, zero_point, value_qx);
            // 计算 GELU 函数对应的量化输出 value_dy
            const auto value_dy =
                value_dx * 0.5 * (1 + std::erf(value_dx * M_SQRT1_2));
            // 将 value_dy 量化为量化整数类型 scalar_t 并返回
            return at::native::quantize_val<scalar_t>(
                output_scale, output_zero_point, value_dy);
          },
          [&](Vec value_qx) -> Vec {
            // 对向量化的 value_qx 进行反量化操作，得到 value_dx
            auto value_dx = value_qx.dequantize(
                scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
            // 使用向量化计算 GELU 函数对应的量化输出 value_dy
            for (auto & value : value_dx) {
              value = value * kPointFiveVec * (kOneVec + (value * kAlphaVec).erf());
            }
            // 将 value_dy 向量化量化为 Vec 类型并返回
            return Vec::quantize(
                value_dx, output_scale, output_zero_point, inv_output_scale);
          });
    });
}

// 定义了一个名为 qsigmoid_kernel 的函数，用于计算量化 Sigmoid 激活函数
void qsigmoid_kernel(
    const Tensor& qx, Tensor& qy, double output_scale, int64_t output_zero_point ) {
  // 获取输入张量 qx 的零点值
  int64_t zero_point = qx.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取输入张量 qx 的缩放因子
  float scale = qx.q_scale();
  // 创建一个缩放因子的向量化对象
  auto scale_vec = Vectorized<float>(scale);
  // 创建一个零点值的向量化对象
  auto zero_point_vec = Vectorized<float>((float)zero_point);

  // 根据输入张量的数据类型，调度不同类型的量化操作
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
    // 计算输出的缩放因子的倒数
    float inv_output_scale = 1.0 / output_scale;

    // 创建一个与 qx 大小相同的空张量 qy，用于存储量化 Sigmoid 的结果
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
        // 指定返回张量的设备、数据类型和内存格式
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point,
        c10::nullopt);
    // 创建一个张量迭代器，用于在 qx 和 qy 之间执行逐元素操作
    auto iter = TensorIterator::unary_op(qy, qx);

    // 定义一个向量化数据类型 Vec
    using Vec = Vectorized<scalar_t>;
    // 调用 CPU 内核函数处理迭代器 iter
    cpu_kernel_vec(
        iter,
        // 对于每个 qx 中的元素，执行如下操作
        [&](scalar_t value_qx) -> scalar_t {
          // 对输入张量进行反量化操作，得到浮点数值 value_dx
          const auto value_dx =
              at::native::dequantize_val(scale, zero_point, value_qx);
          // 计算 Sigmoid 函数的值
          const auto value_dy = 1.0f / (1.0 + std::exp((-value_dx)));
          // 对输出结果进行量化操作，得到量化后的值
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, value_dy);
        },
        // 对于每个 Vec 类型的 qx 中的向量，执行如下操作
        [&](Vec value_qx) -> Vec {
          // 对向量化的 qx 进行反量化操作
          auto value_dx = value_qx.dequantize(scale_vec, zero_point_vec);
          // 对每个元素执行 Sigmoid 函数的计算
          for (auto & value : value_dx) {
            value = value.neg();  // 取反
            value = value.exp();  // 指数运算
            value = Vectorized<float>(1.0f) + value;  // 加 1
            value = value.reciprocal();  // 求倒数
          }
          // 对处理后的向量化数据进行量化操作
          return Vec::quantize(
              value_dx, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

// 定义了一个名为 qhardsigmoid_kernel 的函数，用于计算量化 Hard Sigmoid 激活函数
void qhardsigmoid_kernel(const Tensor& qx, Tensor& qy) {
  // 获取输入张量 qx 的零点值
  int64_t zero_point = qx.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取输入张量 qx 的缩放因子
  float scale = qx.q_scale();
  // 创建一个缩放因子的向量化对象
  auto scale_vec = Vectorized<float>(scale);
  // 创建一个零点值的向量化对象
  auto zero_point_vec = Vectorized<float>((float)zero_point);
  // 计算缩放因子与零点值的负积的向量化对象
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

  // 根据输入张量的数据类型，调度不同类型的量化操作
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qhardsigmoid", [&]() {

    // - Output scale is set to 1.0 / 2^(BIT_NUM)
    // 设置输出的缩放因子为 1.0 / 2^8
    float output_scale = 0.00390625;  // 1.0 / 2^8
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    // 如果数据类型为 kQInt32，则重新设置输出缩放因子为 1.0 / 2^32
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
    }
    // 计算输出缩放因子的倒数
    float inv_output_scale = 1.0 / output_scale;

    // 默认情况下的输出零点值为 0。对于 kQInt8 类型，设置输出零点值为 -128，
    // 以在 [0, 1] 输出范围内最大化精度。对于 kQInt32，可以在以后的 PR 中处理。
    int64_t output_zero_point = 0;
    if (SCALAR_TYPE == at::kQInt8) {
      output_zero_point = -128;
    }

    // 创建一个与 qx 大小相同的空张量 qy，用于存储量化 Hard Sigmoid 的结果
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE),
        output_scale,
        output_zero_point,
        qx.suggest_memory_format());
    // 创建一个张量迭代器，用于一元操作，将 qx 作为输入
    auto iter = TensorIterator::unary_op(qy, qx);

    // 定义使用的向量化类型 qVec 和 fVec
    using qVec = Vectorized<scalar_t>;
    using fVec = Vectorized<float>;

    // 创建常量向量对象，分别表示 0.0f, 3.0f 和 6.0f
    fVec kZeroVec(0.0f);
    fVec kThreeVec(3.0f);
    fVec kSixVec(6.0f);

    // Naive implementation: uses dequantize/execute/quantize routine
    // 使用简单的实现方法：使用去量化/执行/量化例程
    cpu_kernel_vec(
        iter,
        // 对每个 scalar_t 类型的 qx 执行以下 lambda 函数
        [&](scalar_t qx) -> scalar_t {
          // 去量化 qx，得到 x
          auto x = at::native::dequantize_val(scale, zero_point, qx);
          // 对 x 执行截断和缩放操作，将结果保存到 y
          const auto y = std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
          // 将 y 量化为 scalar_t 类型并返回
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, y);
        },
        // 对每个 qVec 类型的 value_qx 执行以下 lambda 函数
        [&](qVec value_qx) -> qVec {
          // 将 value_qx 去量化为 value_dx
          auto value_dx = value_qx.dequantize(
              scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
          // 对 value_dx 中的每个元素执行截断和缩放操作
          for (auto & value : value_dx) {
            value =
                vec::minimum(
                    vec::maximum(value + kThreeVec, kZeroVec),
                    kSixVec) /
                kSixVec;
          }
          // 将处理后的 value_dx 量化为 qVec 类型并返回
          return qVec::quantize(
              value_dx, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

// 定义函数 qclamp_kernel，用于对输入张量进行量化上下限限制操作
void qclamp_kernel(
    const Tensor& qx,                    // 输入张量 qx
    const Scalar& min_scalar,            // 最小值标量 min_scalar
    const Scalar& max_scalar,            // 最大值标量 max_scalar
    Tensor& qy) {                        // 输出张量 qy

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    // 创建一个与 qx 相同大小的空量化张量 qy
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        c10::nullopt);

    using Vec = Vectorized<scalar_t>;
    // 创建一个迭代器 iter 用于对 qx 和 qy 进行逐元素操作
    auto iter = TensorIterator::unary_op(qy, qx);

    // 将 min_scalar 和 max_scalar 转换为 float 类型
    auto min = min_scalar.to<float>();
    auto max = max_scalar.to<float>();

    // 将 min 和 max 量化为对应数据类型的量化值
    scalar_t min_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), min);
    scalar_t max_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), max);

    auto min_vec = Vec(min_q);
    auto max_vec = Vec(max_q);

    // 调用 CPU 核函数对张量进行向量化处理
    cpu_kernel_vec(
        iter,
        // Lambda 函数，对每个标量值进行 min-max 限制
        [&](scalar_t value) -> scalar_t {
          underlying_t min_clamped =
              std::max<underlying_t>(value.val_, min_q.val_);
          return scalar_t(std::min<underlying_t>(min_clamped, max_q.val_));
        },
        // Lambda 函数，对每个向量值进行 min-max 限制
        [&](Vec val) -> Vec {
          auto min_clamped = val.maximum(min_vec);
          return min_clamped.minimum(max_vec);
        });
  });
}

// 定义函数 qclamp_min_kernel，用于对输入张量进行最小值限制操作
void qclamp_min_kernel(const Tensor& qx, const Scalar& min_scalar, Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    // 创建一个与 qx 相同大小的空量化张量 qy
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU)
            // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        c10::nullopt);

    using Vec = Vectorized<scalar_t>;
    // 创建一个迭代器 iter 用于对 qx 和 qy 进行逐元素操作
    auto iter = TensorIterator::unary_op(qy, qx);

    // 将 min_scalar 转换为 float 类型，并量化为对应数据类型的量化值
    auto min = min_scalar.to<float>();
    scalar_t min_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), min);

    auto min_vec = Vec(min_q);

    // 调用 CPU 核函数对张量进行向量化处理
    cpu_kernel_vec(
        iter,
        // Lambda 函数，对每个标量值进行最小值限制
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, min_q.val_));
        },
        // Lambda 函数，对每个向量值进行最小值限制
        [&](Vec val) -> Vec { return val.maximum(min_vec); });
  });
}

// 定义函数 qclamp_max_kernel，用于对输入张量进行最大值限制操作
void qclamp_max_kernel(const Tensor& qx, const Scalar& max_scalar, Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    // 创建一个与 qx 相同大小的空量化张量 qy
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU)
            // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        c10::nullopt);

    using Vec = Vectorized<scalar_t>;
    // 创建一个迭代器 iter 用于对 qx 和 qy 进行逐元素操作
    auto iter = TensorIterator::unary_op(qy, qx);

    // 将 max_scalar 转换为 float 类型，并量化为对应数据类型的量化值
    auto max = max_scalar.to<float>();
    scalar_t max_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), max);

    auto max_vec = Vec(max_q);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          // 使用lambda函数，对输入的value进行处理，返回value.val_和max_q.val_的最小值
          return scalar_t(std::min<underlying_t>(value.val_, max_q.val_));
        },
        [&](Vec val) -> Vec {
          // 使用lambda函数，对输入的Vec对象val进行处理，返回val和max_vec的逐元素最小值
          return val.minimum(max_vec);
        }
    );
void qthreshold_kernel(
  // TODO: For future tasks, since output quantization parameters are set equal to
  // the input ones, it might make sense to implement this completely in the
  // quantized domain.
   const Tensor& qx,  // 输入张量 qx，是一个常量引用，表示输入数据
   const Scalar& threshold_scalar,  // 阈值标量，用于定义阈值的标量值
   const Scalar& value_scalar,  // 值标量，用于定义替换值的标量值
   Tensor& qy) {  // 输出张量 qy，作为函数的输出，传入的是一个引用

  // defines input and output scales and zero_points
  int64_t input_zero_point = qx.q_zero_point();  // 获取输入张量 qx 的零点值
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float input_scale = qx.q_scale();  // 获取输入张量 qx 的量化比例尺度
  int64_t output_zero_point = qy.q_zero_point();  // 获取输出张量 qy 的零点值
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float output_scale = qy.q_scale();  // 获取输出张量 qy 的量化比例尺度
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float inv_output_scale = 1.0 / output_scale;  // 计算输出张量的量化比例尺度的倒数

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qthreshold", [&]() {
    qy = at::_empty_affine_quantized(
      qx.sizes(),  // 使用输入张量 qx 的尺寸创建一个空的仿射量化张量 qy
      // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
      at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),  // 指定设备、数据类型和内存格式
      qx.q_scale(),  // 使用输入张量 qx 的量化比例尺度
      qx.q_zero_point(),  // 使用输入张量 qx 的零点值
      c10::nullopt);  // 无额外的量化参数

    // vectorized
    using Vec = Vectorized<float>;  // 使用 Vectorized 类型处理 float 类型的向量化操作
    using qVec = Vectorized<scalar_t>;  // 使用 Vectorized 类型处理 scalar_t 类型的向量化操作
    // defines the iterator
    auto iter = TensorIterator::unary_op(qy, qx);  // 创建张量迭代器，用于对两个张量进行一元操作
    // defines the vectorized versions
    Vec input_scale_vec = Vec(input_scale);  // 创建输入比例尺度的向量化版本
    Vec input_zero_point_vec = Vec(input_zero_point);  // 创建输入零点值的向量化版本
    Vec input_scale_neg_zp_premul_vec = input_scale_vec * input_zero_point_vec.neg();  // 计算输入比例尺度与负零点值的乘积向量化版本
    // defines the floating-point versions of threshold and value
    float threshold_float = threshold_scalar.to<float>();  // 将阈值标量转换为 float 类型
    float value_float = value_scalar.to<float>();  // 将值标量转换为 float 类型
    Vec threshold_vec = Vec(threshold_float);  // 创建阈值的向量化版本
    Vec value_vec = Vec(value_float);  // 创建值的向量化版本

    // Naive implementation: uses dequantize/execute/quantize routine
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          // dequantize
          const auto x = at::native::dequantize_val(input_scale, input_zero_point, value_qx);  // 反量化输入值
          // Applies the Threshold operation
          const auto y = x > threshold_float ? x : value_float;  // 应用阈值操作
          // quantize
          return at::native::quantize_val<scalar_t>(output_scale, output_zero_point, y);  // 量化输出值
        },
        [&](qVec value_qx) -> qVec {
          // dequantize
          auto dx_vec = value_qx.dequantize(
            input_scale_vec, input_zero_point_vec, input_scale_neg_zp_premul_vec);  // 反量化输入向量
          for (auto & value : dx_vec) {
            // check if any elements are below threshold
            const auto cmp_to_threshold = value > threshold_vec;  // 检查是否有元素小于阈值
            if (cmp_to_threshold.zero_mask()) {
              // blend
              value = Vec::blendv(value_vec, value, cmp_to_threshold);  // 根据条件混合值
            }
          }
          // quantize
          return qVec::quantize(dx_vec, output_scale, output_zero_point, inv_output_scale);  // 量化输出向量
        });
  });
}
void qhardswish_kernel(const Tensor& qx, Tensor& qy) {
  // 获取输入张量 qx 的量化参数：缩放因子和零点
  const auto i_scale = qx.q_scale();
  const auto i_zero_point = qx.q_zero_point();

  // 获取输出张量 qy 的量化参数：缩放因子和零点
  const auto o_scale = qy.q_scale();
  const auto o_zero_point = qy.q_zero_point();

  // 计算输出张量的缩放因子的倒数
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const float o_inv_scale = 1.0 / o_scale;

  // 使用 Vectorized 类型进行向量化操作
  using fVec = Vectorized<float>;

  // 创建向量化的量化参数
  fVec i_scale_vec(i_scale);
  fVec i_zero_point_vec(i_zero_point);
  fVec i_scale_neg_zp_premul_vec = i_scale_vec * i_zero_point_vec.neg();
  fVec zero_vec(0.0f);
  fVec three_vec(3.0f);
  fVec six_vec(6.0f);

  // 根据输入张量的标量类型调度具体的操作
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qhardswish", [&]() {
    // 创建一个张量迭代器以对 qy 和 qx 进行逐元素操作
    auto iter = TensorIterator::unary_op(qy, qx);
    cpu_kernel_vec(
        iter,
        // 对每个标量值执行的操作
        [&](scalar_t value) -> scalar_t {
          // 将输入值反量化为浮点数
          const auto x =
              at::native::dequantize_val(i_scale, i_zero_point, value);
          // 计算 HardSwish 激活函数的输出，并重新量化为标量类型
          const auto y = x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
          return at::native::quantize_val<scalar_t>(o_scale, o_zero_point, y);
        },
        // 对向量化值执行的操作
        [&](qVec value) -> qVec {
          // 反量化向量化输入值
          auto value_dx = value.dequantize(i_scale_vec, i_zero_point_vec,
                                           i_scale_neg_zp_premul_vec);
          // 使用向量化操作计算 HardSwish 激活函数的输出，并重新量化
          for (auto & value : value_dx) {
            value = value * vec::minimum(
              vec::maximum(value + three_vec, zero_vec),
              six_vec
            ) / six_vec;
          }
          return qVec::quantize(value_dx, o_scale, o_zero_point, o_inv_scale);
        });
  });
}


void qtanh_kernel(const Tensor& qx, Tensor& qy) {
  // 获取输入张量 qx 的零点
  int64_t zero_point = qx.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取输入张量 qx 的缩放因子
  float scale = qx.q_scale();
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>((float)zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

  // 根据输入张量的标量类型调度具体的操作
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qtanh", [&]() {
    // Naive implementation: uses dequantize/execute/quantize routine
    // - Output scale is set to 2.0 / 2^(BIT_NUM)
    // - For signed types output zero point is set to 0
    // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
    // 设置输出张量的缩放因子
    float output_scale = 0.0078125;  // 2.0 / 512
    int64_t output_zero_point = 0;
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    // 如果标量类型是 kQInt32，设置不同的输出缩放因子
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 4.656612873077393e-10;  // 2.0 / 2^32
    } else if (SCALAR_TYPE == at::kQUInt8) {
      // 如果标量类型是 kQUInt8，设置不同的输出零点
      output_zero_point = 128;
    }
    // 计算输出缩放因子的倒数
    float inv_output_scale = 1.0 / output_scale;

    // 创建一个与 qx 具有相同大小和量化设置的空张量 qy
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point,
        c10::nullopt);
    // 创建一个张量迭代器以对 qy 和 qx 进行逐元素操作
    auto iter = TensorIterator::unary_op(qy, qx);

    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          // 对输入的量化值进行反量化操作，使用给定的缩放因子和零点
          const auto value_dx =
              at::native::dequantize_val(scale, zero_point, value_qx);
          // 对反量化后的值应用双曲正切函数，并量化到输出的范围内
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, std::tanh(value_dx));
        },
        [&](Vec value_qx) -> Vec {
          // 对向量形式的输入值进行反量化操作，使用向量形式的缩放因子和零点
          const auto value_dx = value_qx.dequantize(
              scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
          // 初始化返回值向量
          Vec::float_vec_return_type retvals;
          // 对每个向量元素应用双曲正切函数
          for (const auto idx : c10::irange(Vec::float_num_vecs())) {
            retvals[idx] = value_dx[idx].tanh();
          }
          // 将双曲正切后的向量值量化到输出的范围内
          return Vec::quantize(
              retvals, output_scale, output_zero_point, inv_output_scale);
        });
  // 计算输入张量 qx 的量化零点值
  int64_t i_zp = qx.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 从输入张量 qx 获取量化比例因子
  float i_scale = qx.q_scale();

  // 在未来的 PR 中，可以改进输出的零点和比例因子的选择

  // 计算输出张量 qy 的量化零点值
  int64_t o_zp = qy.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 从输出张量 qy 获取量化比例因子
  float o_scale = qy.q_scale();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 计算输出张量 qy 的倒数量化比例因子
  float inv_o_scale = 1.0 / o_scale;

  // 将 alpha 转换为 float 类型
  float alpha_float = alpha.to<float>();
  // 将 scale 转换为 float 类型
  float scale_coef = scale.to<float>();
  // 将 input_scale 转换为 float 类型
  float input_scale_coef = input_scale.to<float>();

  // 根据输入张量 qx 的标量类型分发运行 qelu_kernel
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qelu_kernel", [&] {

    // 创建一个迭代器，对输出张量 qy 和输入张量 qx 执行一元操作
    auto iter = TensorIterator::unary_op(qy, qx);

    // 使用 Vectorized 类处理 float 类型向量化运算
    using Vec = Vectorized<float>;
    // 使用 Vectorized 类处理输入张量 qx 标量类型的向量化运算
    using qVec = Vectorized<scalar_t>;

    // 创建一个全零的 Vectorized<float> 对象
    Vec zero_vec = Vec(0.0f);
    // 创建一个全一的 Vectorized<float> 对象
    Vec one_vec = Vec(1.0f);
    // 创建一个包含 alpha_float 值的 Vectorized<float> 对象
    Vec alpha_vec = Vec(alpha_float);
    // 创建一个包含 scale_coef 值的 Vectorized<float> 对象
    Vec scale_coef_vec = Vec(scale_coef);
    // 创建一个包含 input_scale_coef 值的 Vectorized<float> 对象
    Vec input_scale_coef_vec = Vec(input_scale_coef);
    // 创建一个包含 i_scale 值的 Vectorized<float> 对象
    Vec i_scale_vec = Vec(i_scale);
    // 创建一个包含 i_zp 值的 Vectorized<float> 对象
    Vec i_zero_point_vec = Vec((float)i_zp);
    // 创建一个包含 i_scale * (-i_zp) 值的 Vectorized<float> 对象
    Vec i_scale_neg_zp_premul_vec = i_scale_vec * i_zero_point_vec.neg();
    cpu_kernel_vec(
      iter,
      [&](scalar_t value_qx) -> scalar_t {
        // 对输入值进行去量化操作
        const auto x = at::native::dequantize_val(i_scale, i_zp, value_qx);
        // 应用ELU激活函数
        const auto y = x >= 0
          ? x * scale_coef
          : ((std::exp(x * input_scale_coef) - 1) * alpha_float * scale_coef);

        // 对输出值进行量化操作
        return at::native::quantize_val<scalar_t>(o_scale, o_zp, y);
      },
      [&](qVec value_qx) -> qVec {
        // 对输入向量进行去量化操作
        auto dx_vec_vec = value_qx.dequantize(i_scale_vec, i_zero_point_vec,
                                            i_scale_neg_zp_premul_vec);
        for (auto & value : dx_vec_vec) {
          // 快速检查是否有元素小于零
          const auto cmp_to_zero = value > zero_vec;

          if (cmp_to_zero.zero_mask()) {
            // 创建副本并计算ELU的负部分
            Vec dx_vec_copy_neg_elu = value * one_vec;
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu * input_scale_coef_vec;
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu.exp();
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu - one_vec;
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu * alpha_vec;
            // 混合原始值和ELU负部分
            value = Vec::blendv(dx_vec_copy_neg_elu, value,
                                        value > zero_vec);
          }

          // 应用量化和缩放系数
          value = value * scale_coef_vec;
        }
        // 对输出向量进行量化操作
        return qVec::quantize(dx_vec_vec, o_scale, o_zp, inv_o_scale);
      }
    );
// } 结束了函数模板的定义

// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self and out are of the same dtype.
// Note: other is already assumed to be in int32, i.e., it's
// round(float/self_scale)
// 模板函数定义，用于将标量 other 加到张量 self 上，并将结果写入张量 out 中
template <bool ReLUFused = false>
void qadd_scalar_kernel(Tensor& out, const Tensor& self, const Scalar& other) {
  // 获取输出张量的零点值
  int64_t zero_point = out.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取输出张量的量化比例
  float scale = out.q_scale();
  // 计算输出张量的量化比例的倒数
  float inv_scale = 1.0f / scale;
  // 获取输入张量 self 的零点值
  int64_t self_zero_point = self.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取输入张量 self 的量化比例
  float self_scale = self.q_scale();

  // 计算乘数，用于转换输入到输出的量化值
  float multiplier = self_scale * inv_scale;

  // 根据输入张量的类型，分发到相应的处理函数
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd_scalar", [&]() {
    using Vec = Vectorized<scalar_t>;
    // 创建张量迭代器，用于遍历输出张量和输入张量 self
    auto iter = TensorIterator::unary_op(out, self);
    // 将标量 other 转换为 int32 类型
    auto other_val = other.to<int32_t>();
    // 创建用于 SIMD 向量化的包装类型
    auto other_vec = Vectorized<c10::qint32>(static_cast<c10::qint32>(other_val));
    // 调用 CPU 内核向量化函数，对每个元素执行操作
    cpu_kernel_vec(
        iter,
        // 标量版本的操作，计算输出值 res
        [&](scalar_t a) -> scalar_t {
          // 计算输入值减去输入张量零点值的差
          int32_t a_sub_z = static_cast<int32_t>(a.val_) -
              static_cast<int32_t>(self_zero_point);
          // 计算最终的量化值
          int32_t c = a_sub_z + other_val;
          scalar_t res = at::native::requantize_from_int<scalar_t>(
              multiplier, zero_point, c);
          // 如果启用了 ReLU 后处理，进行额外的处理
          if (ReLUFused) {
            res.val_ = std::max<scalar_t::underlying>(res.val_, zero_point);
          }
          return res;
        },
        // 向量化版本的操作，对向量化数据执行相同的操作
        [&](Vec a) -> Vec {
          // 计算向量化数据减去输入张量零点值的差
          Vec::int_vec_return_type a_sub_z =
              a.widening_subtract(Vec(static_cast<scalar_t>(self_zero_point)));
          // 计算最终的量化值
          Vec::int_vec_return_type c;
          for (const auto i : c10::irange(Vec::int_num_vecs())) {
            c[i] = a_sub_z[i] + other_vec;
          }
          Vec rv = Vec::requantize_from_int(c, multiplier, zero_point);
          // 如果启用了 ReLU 后处理，进行额外的处理
          if (ReLUFused) {
            rv = rv.maximum(Vec(static_cast<scalar_t>(zero_point)));
          }
          return rv;
        });
  });
}
// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self, other, out are of the same dtype.
// 模板函数定义，用于将标量 other 加到张量 self 上，并将结果写入张量 out 中
template <bool ReLUFused = false>
void qadd_kernel(Tensor& out, const Tensor& self, const Tensor& other) {
  // 获取输出张量的量化零点
  int64_t zero_point = out.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取输出张量的量化比例因子
  float scale = out.q_scale();
  // 计算输出张量的量化比例因子的倒数
  float inv_scale = 1.0f / scale;
  // 获取第一个输入张量的量化零点
  int64_t self_zero_point = self.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取第一个输入张量的量化比例因子
  float self_scale = self.q_scale();
  // 获取第二个输入张量的量化零点
  int64_t other_zero_point = other.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 获取第二个输入张量的量化比例因子
  float other_scale = other.q_scale();

  // 在此处广播参数以在循环迭代中分摊成本。
  // TODO: 通过预先乘以零点和比例因子，以及在比例因子 * x_q - (比例因子 * 零点) 上执行 FMA，优化反量化过程
  // 创建第一个输入张量的比例因子和零点的向量化对象
  auto self_zero_point_vec = Vectorized<float>((float)self_zero_point);
  auto self_scale_vec = Vectorized<float>(self_scale);
  // 创建第二个输入张量的比例因子和零点的向量化对象
  auto other_zero_point_vec = Vectorized<float>((float)other_zero_point);
  auto other_scale_vec = Vectorized<float>(other_scale);

  // 计算第一个输入张量的比例因子 * 零点的负数预乘积向量
  auto self_scale_neg_zp_premul_vec = self_scale_vec * self_zero_point_vec.neg();
  // 计算第二个输入张量的比例因子 * 零点的负数预乘积向量
  auto other_scale_zp_premul_vec = other_scale_vec * other_zero_point_vec.neg();

  // 借用二进制运算迭代器，将输出张量、第一个输入张量和第二个输入张量传递给它
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);

  // 根据输出张量的标量类型分派量化整数类型的操作
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qadd", [&]() {
    // 定义向量化类型为标量类型的向量化
    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t a, scalar_t b) -> scalar_t {
          // 对输入张量元素a和b进行反量化操作，得到浮点数da和db
          const auto da =
              at::native::dequantize_val(self_scale, self_zero_point, a);
          const auto db =
              at::native::dequantize_val(other_scale, other_zero_point, b);
          // 计算浮点数之和c
          float c = da + db;
          // 如果启用了ReLU融合，则将c与0比较取最大值
          if (ReLUFused) {
            c = std::max<float>(c, 0.0);
          }
          // 将浮点数c量化为输出类型scalar_t，并返回
          return at::native::quantize_val<scalar_t>(scale, zero_point, c);
        },
        [&](Vec a, Vec b) -> Vec {
          // 对输入向量a和b进行批量反量化操作，得到浮点向量da和db
          const auto da = a.dequantize(
              self_scale_vec, self_zero_point_vec, self_scale_neg_zp_premul_vec);
          const auto db = b.dequantize(
              other_scale_vec, other_zero_point_vec, other_scale_zp_premul_vec);
          // 初始化返回值向量
          Vec::float_vec_return_type retvals;
          // 遍历所有向量元素并计算其和
          for (const auto i : c10::irange(Vec::float_num_vecs())) {
            auto c = da[i] + db[i];
            // 如果启用了ReLU融合，则将c与0向量逐元素比较取最大值
            if (ReLUFused) {
              c = vec::maximum(c, Vectorized<float>(0.0f));
            }
            // 将每个元素和量化参数进行量化操作，并存入返回向量中
            retvals[i] = c;
          }
          // TODO: fbgemm::Quantize不支持使用预广播的参数，可以在API中优化以节省计算周期
          // TODO: 特化fbgemm::Quantize用于单个向量并使其可内联化，可能有助于张量迭代器的交错实现
          // 将浮点向量retvals量化为向量rv，并返回
          auto rv = Vec::quantize(retvals, scale, zero_point, inv_scale);
          return rv;
        });
  });
}

// Note: out is assumed to be the same size as self and other.
// Note: Multiplication is only supported when self, other, out are of the same
// dtype.
// 定义一个模板函数，用于执行量化乘法操作
template <bool ReLUFused = false>
void qmul_kernel(Tensor& out, const Tensor& self, const Tensor& other) {
  // 获取输出张量的量化零点
  int64_t zero_point = out.q_zero_point();
  // 获取输出张量的量化比例
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float scale = out.q_scale();
  // 计算输出张量的倒数量化比例
  float inv_scale = 1.0f / scale;
  // 获取第一个输入张量的量化零点
  int64_t self_zero_point = self.q_zero_point();
  // 获取第一个输入张量的量化比例
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float self_scale = self.q_scale();
  // 获取第二个输入张量的量化零点
  int64_t other_zero_point = other.q_zero_point();
  // 获取第二个输入张量的量化比例
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float other_scale = other.q_scale();

  // 计算乘法操作的乘数
  float multiplier = self_scale * other_scale * inv_scale;

  // 使用张量迭代器创建一个迭代器对象，用于执行张量之间的二元操作
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);

  // 根据输出张量的数据类型执行量化乘法操作
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qmul", [&]() {
    using Vec = Vectorized<scalar_t>;
    // 使用 CPU 内核进行向量化计算
    cpu_kernel_vec(
        iter,
        [&](scalar_t a, scalar_t b) -> scalar_t {
          // 将输入张量值减去对应的量化零点，得到 a 和 b 的偏移量
          int32_t a_sub_z = static_cast<int32_t>(a.val_) -
              static_cast<int32_t>(self_zero_point);
          int32_t b_sub_z = static_cast<int32_t>(b.val_) -
              static_cast<int32_t>(other_zero_point);
          // 计算偏移量的乘积
          int32_t c = a_sub_z * b_sub_z;
          // 将乘积值重新量化为输出张量的数据类型
          scalar_t res = at::native::requantize_from_int<scalar_t>(
              multiplier, zero_point, c);
          // 如果启用了 ReLU 融合，则进行 ReLU 操作
          if (ReLUFused) {
            res.val_ = std::max<scalar_t::underlying>(res.val_, zero_point);
          }
          return res;
        },
        [&](Vec a, Vec b) -> Vec {
          // 使用向量化计算的方式处理输入张量的向量数据
          Vec::int_vec_return_type a_sub_zp =
              a.widening_subtract(Vec(static_cast<scalar_t>(self_zero_point)));
          Vec::int_vec_return_type b_sub_zp =
              b.widening_subtract(Vec(static_cast<scalar_t>(other_zero_point)));
          Vec::int_vec_return_type c;
          // 对每个向量数据执行乘法操作
          for (const auto i : c10::irange(Vec::int_num_vecs())) {
            c[i] = a_sub_zp[i] * b_sub_zp[i];
          }
          // 将乘积值重新量化为输出张量的向量数据类型
          Vec rv = Vec::requantize_from_int(c, multiplier, zero_point);
          // 如果启用了 ReLU 融合，则进行 ReLU 操作
          if (ReLUFused) {
            rv = rv.maximum(Vec(static_cast<scalar_t>(zero_point)));
          }
          return rv;
        });
  });
}

// 定义一个模板函数，用于执行二维最大池化操作（NHWC 格式）
template <typename scalar_t, typename scalar_t_underlying>
void _qmaxpool_2d_nhwc_kernel(
    const Tensor& qx,
    int64_t iC, // 输入/输出通道数
    int64_t iH,
    int64_t iW, // 输入尺寸
    int64_t oH,
    int64_t oW, // 输出尺寸
    int64_t kH,
    int64_t kW, // 核大小
    int64_t sH,
    int64_t sW, // 步长
    int64_t pH,
    int64_t pW, // 填充
    int64_t dH,
    int64_t dW, // 膨胀
    Tensor& qy) {
    // 获取输入张量和输出张量的数据指针
    scalar_t* idata = static_cast<scalar_t*>(qx.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());

    // 获取批次大小
    int64_t nBatch = qx.size(0);
    // 并行循环遍历输出张量的所有元素
    at::parallel_for(0, nBatch * oH * oW, 0, [&](int64_t begin, int64_t end) {
      // 初始化循环变量 b, row, col
      int64_t b{0}, row{0}, col{0};
      // 使用 data_index_init 函数初始化 b, row, col
      data_index_init(begin, b, nBatch, row, oH, col);
    
      // 循环处理当前线程分配的元素范围 [begin, end)
      for (const auto i : c10::irange(begin, end)) {
        // 计算输入数据和输出数据的指针
        auto* i_p = reinterpret_cast<scalar_t_underlying*>(idata + b * iW * iH * iC);
        auto* o_p = reinterpret_cast<scalar_t_underlying*>(odata + i * iC);
    
        // 计算 reduction block 的起始和结束位置
        int64_t h_start = row * sH - pH;
        int64_t w_start = col * sW - pW;
        int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
        int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);
    
        // 确保起始位置不小于 0
        while (h_start < 0)
          h_start += dH;
        while (w_start < 0)
          w_start += dW;
    
        int64_t c = 0;
    
        // 使用矢量化操作的并行循环，每次处理 4 个向量
        constexpr auto vec_width = Vectorized<scalar_t>::size();
        for (; c + 4 * vec_width <= iC; c += 4 * vec_width) {
          // 初始化累加器向量，设置为最小值
          Vectorized<scalar_t> acc{
              scalar_t(std::numeric_limits<scalar_t_underlying>::lowest())};
          // 声明一个包含 4 个累加器向量的数组
          Vectorized<scalar_t> accs[4] = {acc, acc, acc, acc};
          int64_t tcntr = 0;
          int64_t x, y;
          // 循环遍历 reduction block 中的每个像素
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              // 并行处理 4 个向量
              for (const auto i : c10::irange(4)) {
                tcntr = y * iW + x;
                // 加载当前像素位置的向量数据
                auto vals = Vectorized<scalar_t>::loadu(
                    i_p + tcntr * iC + c + Vectorized<scalar_t>::size() * i);
                // 更新累加器向量的最大值
                accs[i] = vec::maximum(accs[i], vals);
              }
            } // for x
          } // for y
          // 将更新后的累加器向量存储到输出数据中
          for (const auto i : c10::irange(4)) {
            accs[i].store(o_p + c + Vectorized<scalar_t>::size() * i);
          }
        } // for c
    
        // 处理剩余的不足 4 个向量的部分，使用单个向量处理
        for (; c + vec_width <= iC; c += vec_width) {
          // 初始化累加器向量，设置为最小值
          Vectorized<scalar_t> acc{
              scalar_t(std::numeric_limits<scalar_t_underlying>::lowest())};
          int64_t tcntr = 0;
          int64_t x, y;
          // 循环遍历 reduction block 中的每个像素
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              // 计算当前像素在数据中的位置
              tcntr = y * iW + x;
              // 加载当前位置的向量数据
              auto vals = Vectorized<scalar_t>::loadu(i_p + tcntr * iC + c);
              // 更新累加器向量的最大值
              acc = vec::maximum(acc, vals);
            } // for x
          } // for y
          // 将更新后的累加器向量存储到输出数据中
          acc.store(o_p + c);
        } // for c
    
        // 处理剩余的不足一个向量的部分，使用标量处理
        for (; c < iC; ++c) {
          // 初始化最大值为最小可能值
          auto max_val = std::numeric_limits<scalar_t_underlying>::lowest();
          int64_t tcntr = 0;
          int64_t x, y;
          // 循环遍历 reduction block 中的每个像素
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              // 计算当前像素在数据中的位置
              tcntr = y * iW + x;
              // 加载当前位置的标量数据
              auto val = *(i_p + tcntr * iC + c);
              // 更新最大值
              max_val = std::max(max_val, val);
            } // for x
          } // for y
    
          // 将最大值存储到输出数据中
          o_p[c] = max_val;
        } // for c
    
        // 更新 b, row, col，准备处理下一个元素
        data_index_step(b, nBatch, row, oH, col, oW);
      }
    });
void qmaxpool_2d_nhwc_kernel(
    const Tensor& qx,
    int64_t iC, // 输入/输出通道数
    int64_t iH,
    int64_t iW, // 输入尺寸：高度和宽度
    int64_t oH,
    int64_t oW, // 输出尺寸：高度和宽度
    int64_t kH,
    int64_t kW, // 卷积核大小：高度和宽度
    int64_t sH,
    int64_t sW, // 步幅：高度和宽度
    int64_t pH,
    int64_t pW, // 填充：高度和宽度
    int64_t dH,
    int64_t dW, // 膨胀：高度和宽度
    Tensor& qy) { // 输出张量引用
  // 检查输入张量的数据类型是否为无符号字节类型
  if (qx.scalar_type() == ScalarType::Byte) {
    // 使用宏 AT_DISPATCH_INTEGRAL_TYPES 处理积分类型，调用 _qmaxpool_2d_nhwc_kernel 函数
    AT_DISPATCH_INTEGRAL_TYPES(qx.scalar_type(), "max_pool2d_nhwc", [&]() {
      _qmaxpool_2d_nhwc_kernel<scalar_t, scalar_t>(qx, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
    });
  } else {
    // 使用宏 AT_DISPATCH_QINT_TYPES 处理量化整数类型，调用 _qmaxpool_2d_nhwc_kernel 函数
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool2d_nhwc", [&]() {
      _qmaxpool_2d_nhwc_kernel<scalar_t, scalar_t::underlying>(qx, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
    });
  }
}

void qmaxpool_3d_nthwc_kernel(
    const Tensor& qx,
    int64_t iC, // 输入/输出通道数
    int64_t iT,
    int64_t iH,
    int64_t iW, // 输入尺寸：时间、高度和宽度
    int64_t oT,
    int64_t oH,
    int64_t oW, // 输出尺寸：时间、高度和宽度
    int64_t kT,
    int64_t kH,
    int64_t kW, // 卷积核大小：时间、高度和宽度
    int64_t sT,
    int64_t sH,
    int64_t sW, // 步幅：时间、高度和宽度
    int64_t pT,
    int64_t pH,
    int64_t pW, // 填充：时间、高度和宽度
    int64_t dT,
    int64_t dH,
    int64_t dW, // 膨胀：时间、高度和宽度
    Tensor& qy) { // 输出张量引用
  // 使用宏 AT_DISPATCH_QINT_TYPES 处理量化整数类型，调用 _qmaxpool_3d_nthwc_kernel 函数
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool3d_nthwc", [&]() {
    // 获取输入张量和输出张量的数据指针
    scalar_t* idata = static_cast<scalar_t*>(qx.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());
    int64_t nBatch = qx.size(0); // 获取批次大小
    // 使用 ATen 的 parallel_for 函数进行并行操作，范围是 0 到 nBatch * oT * oH * oW
    at::parallel_for(0, nBatch * oT * oH * oW, 0, [&](int64_t begin, int64_t end) {
      // 初始化变量 b, time, row, col
      int64_t b{0}, time{0}, row{0}, col{0};

      // 调用 data_index_init 函数初始化 b, time, row, col
      data_index_init(begin, b, nBatch, time, oT, row, oH, col, oW);

      // 对于给定范围内的每个索引 i
      for (const auto i : c10::irange(begin, end)) {
        // 获取输入和输出数据的指针
        auto* i_p = reinterpret_cast<scalar_t::underlying*>(idata + b * iT * iW * iH * iC);
        auto* o_p = reinterpret_cast<scalar_t::underlying*>(odata + i * iC);

        // 循环遍历减少块
        int64_t t_start = time * sT - pT;
        int64_t h_start = row * sH - pH;
        int64_t w_start = col * sW - pW;
        int64_t t_end = std::min(t_start + (kT - 1) * dT + 1, iT);
        int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
        int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);

        // 确保起始索引非负
        while (t_start < 0)
          t_start += dT;
        while (h_start < 0)
          h_start += dH;
        while (w_start < 0)
          w_start += dW;

        int64_t c = 0;
        constexpr auto vec_width = Vectorized<scalar_t>::size();

        // 向量化循环
        for (; c + vec_width <= iC; c += vec_width) {
          // 初始化累加器为 scalar_t 类型的最小值
          Vectorized<scalar_t> acc{
              scalar_t(std::numeric_limits<scalar_t::underlying>::lowest())};
          int64_t tcntr = 0;
          int64_t t, x, y;
          // 循环遍历 t, y, x
          for (t = t_start; t < t_end; t += dT) {
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                // 计算数据的索引
                tcntr = t * iH * iW + y * iW + x;
                // 加载向量化的输入数据
                auto vals = Vectorized<scalar_t>::loadu(i_p + tcntr * iC + c);
                // 计算最大值并更新累加器
                acc = vec::maximum(acc, vals);
              } // for x
            } // for y
          } // for t
          // 存储累加器中的结果到输出数据
          acc.store(o_p + c);
        } // for c

        // 处理剩余的非向量化部分
        for (; c < iC; ++c) {
          // 初始化最大值为 scalar_t 类型的最小值
          auto max_val = std::numeric_limits<scalar_t::underlying>::lowest();
          int64_t tcntr = 0;
          int64_t t, x, y;
          // 循环遍历 t, y, x
          for (t = t_start; t < t_end; t += dT) {
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                // 计算数据的索引
                tcntr = t * iH * iW + y * iW + x;
                // 获取并更新最大值
                auto val = *(i_p + tcntr * iC + c);
                max_val = std::max(max_val, val);
              } // for x
            } // for y
          } // for t
          // 将最大值存储到输出数据中
          o_p[c] = max_val;
        } // for c

        // 更新 b, time, row, col 的索引
        data_index_step(b, nBatch, time, oT, row, oH, col, oW);
      }

    });

  });
// 定义模板函数，执行基于 AVX 指令集的平均池化操作，按照 NHWC 格式处理
template <typename T>
void do_avg_pool_nhwc_on_AVX_n(
    const typename T::underlying* i_p,  // 输入数据指针，使用模板类型 T 的基础类型
    typename T::underlying* o_p,        // 输出数据指针，使用模板类型 T 的基础类型
    int& c_start,                       // 开始处理的通道索引
    int input_zero_point_m_size,        // 输入零点乘法尺寸
    int output_zero_point,              // 输出零点
    float multiplier,                   // 乘法因子
    int dstart, int dend,               // 深度（通道）方向的起始和结束索引
    int hstart, int hend,               // 高度方向的起始和结束索引
    int wstart, int wend,               // 宽度方向的起始和结束索引
    int dsize, int hsize, int wsize,    // 深度、高度、宽度的尺寸
    int csize) {                        // 通道的尺寸
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && !defined(_MSC_VER)
  // 用于通道累加器的缓冲区，用于将通道循环置于最内层，以便输入张量数据的内存访问是连续的。
#ifdef CPU_CAPABILITY_AVX2
  constexpr int cb_size = 16;  // AVX2 情况下的通道缓冲区尺寸
#else
  constexpr int cb_size = 8;   // AVX512 情况下的通道缓冲区尺寸
#endif
  constexpr int vec_width = Vectorized<T>::size() / 4;  // 向量化宽度
  constexpr int cb_step = cb_size * vec_width;           // 通道步长
  Vectorized<int32_t> acc_buffer[cb_size];               // 整型累加缓冲区
  Vectorized<float> acc_buffer_fp[cb_size];              // 浮点型累加缓冲区

#ifdef CPU_CAPABILITY_AVX2
  if (vec_width == 8) {  // 如果向量化宽度为 8
#else
  if (vec_width == 16) { // 如果向量化宽度为 16
#endif
    for (int c = c_start; c < csize; c += cb_step) {  // 按通道步长循环处理通道
      int cend = std::min(cb_size, (csize - c) / vec_width);  // 计算当前轮次的通道结束索引
      // 初始化循环
      for (const auto ic : c10::irange(cend)) {
        acc_buffer[ic] = Vectorized<int32_t>(input_zero_point_m_size);  // 将累加缓冲区初始化为输入零点乘法尺寸
      }
      // 计算循环
      for (const auto id : c10::irange(dstart, dend)) {
        for (const auto ih : c10::irange(hstart, hend)) {
          for (const auto iw : c10::irange(wstart, wend)) {
            const int i_idx =
                (id * wsize * hsize + ih * wsize + iw) *
                    csize +
                c;
            for (const auto ic : c10::irange(cend)) {
              auto vals = vec::convert_to_int32<typename T::underlying>(
                  i_p + i_idx + ic * vec_width);  // 将输入数据转换为整型并累加到缓冲区
              acc_buffer[ic] = acc_buffer[ic] + vals;  // 更新累加缓冲区
            }
          }
        }
      }
      // 将累加的 int32 值转换为 float32
      vec::convert((int*)acc_buffer, (float*)acc_buffer_fp, cend * vec_width);

      // 根据 AVX2 或 AVX512 的能力使用 32 或 8 个通道进行量化，最后退化到单通道量化
#ifdef CPU_CAPABILITY_AVX2
      QuantizeAvx2<typename T::underlying>(
          (float*)acc_buffer_fp,
          o_p + c,
          cend * vec_width,
          multiplier,
          output_zero_point);
#else
      QuantizeAvx512<typename T::underlying>(
          (float*)acc_buffer_fp,
          o_p + c,
          cend * vec_width,
          multiplier,
          output_zero_point);
#endif
    }
    c_start = csize / vec_width * vec_width;  // 更新 c_start，使其为向量化宽度的整数倍
  }
#endif
}
    # 定义函数的参数列表开始，这里是三个 int64_t 类型的参数
    int64_t stride_D,
    int64_t stride_H,
    int64_t stride_W) {
    # 函数参数列表结束，函数体开始
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && !defined(_MSC_VER)
  // 如果编译器支持 AVX2 或 AVX512 并且不是 Microsoft 编译器
  constexpr int vec_width = Vectorized<T>::size() / 4;
#ifdef CPU_CAPABILITY_AVX2
  // 根据 AVX2 支持的向量宽度选择分支
  if (vec_width == 8) {
#else
  // 根据 AVX512 支持的向量宽度选择分支
  if (vec_width == 16) {
#endif
    // 循环处理每个通道，每次处理 vec_width 个通道
    for (; c + vec_width <= channel_size; c += vec_width) {
      int64_t tcntr = 0;

      // 初始化累加器，用于存储输入量化后的值
      Vectorized<int32_t> acc(input_zero_point_m_size);
      // 遍历输入数据的维度
      for (const auto id : c10::irange(dstart, dend)) {
        for (const auto ih : c10::irange(hstart, hend)) {
          for (const auto iw : c10::irange(wstart, wend)) {
            // 计算输入数据的索引
            tcntr = id * stride_D + ih * stride_H + iw * stride_W;
            // 将输入数据转换为 int32 类型并累加到 acc 中
            auto vals = vec::convert_to_int32<typename T::underlying>(
                i_p + tcntr * channel_multiplier + c * stride_C);
            acc = acc + vals;
          }
        }
      }

      // 存储累加器中的整数值到数组
      int32_t acc_int[vec_width];
      float acc_fp[vec_width];
      acc.store(acc_int);
      // 将整数值转换为浮点数
      vec::convert(acc_int, acc_fp, vec_width);
      // 对累加结果进行量化和写入输出
      at::native::quantize_vec<T>(
          1.0f / multiplier,
          output_zero_point,
          acc_fp,
          reinterpret_cast<T*>(o_p + c),
          vec_width);
    }
  }
#endif
}

template <typename T>
void _qadaptive_avg_pool_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t nBatch,
    int64_t sizeC,
    int64_t isizeD,  // 设置为1表示2D
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeD,  // 设置为1表示2D
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideC,
    int64_t istrideD,  // 设置为1表示2D
    int64_t istrideH,
    int64_t istrideW) {

  // 获取输入和输出数据的指针
  T* idata = static_cast<T*>(qx.data_ptr());
  T* odata = static_cast<T*>(qy.data_ptr());

  // 获取输入和输出的量化比例因子和零点
  const float input_scale = qx.q_scale();
  const float output_scale = qy.q_scale();
  const int input_zero_point = qx.q_zero_point();
  const int output_zero_point = qy.q_zero_point();

  // 并行处理每个批次
  at::parallel_for(0, nBatch, 0, [&](int64_t batch_start, int64_t batch_end) {
    }
  });
}

void qadaptive_avg_pool2d_nhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t nBatch,
    int64_t sizeC,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideC,
    int64_t istrideH,
    int64_t istrideW) {
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "adaptive_avg_pool2d_nhwc", [&]() {
        // 调用具体的池化核心函数，处理二维自适应平均池化
        _qadaptive_avg_pool_kernel<scalar_t>(
          qx,
          qy,
          nBatch,
          sizeC,
          /*isizeD=*/1,
          isizeH,
          isizeW,
          /*osizeD=*/1,
          osizeH,
          osizeW,
          istrideB,
          istrideC,
          /*istrideD=*/1,
          istrideH,
          istrideW);
      }
    );
}

void qadaptive_avg_pool3d_ndhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t nBatch,
    int64_t sizeC,
    int64_t isizeD,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeD,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideC,
    int64_t istrideD,
    int64_t istrideH,
    int64_t istrideW) {
    // 未实现
}
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "adaptive_avg_pool3d_ndhwc", [&]() {
        // 根据量化类型分发函数，执行自适应平均池化的核心计算
        _qadaptive_avg_pool_kernel<scalar_t>(
            qx,                 // 输入张量 qx，类型为 scalar_t
            qy,                 // 输出张量 qy，类型为 scalar_t
            nBatch,             // 批次大小
            sizeC,              // 输入通道数
            isizeD,             // 输入数据深度维度大小
            isizeH,             // 输入数据高度维度大小
            isizeW,             // 输入数据宽度维度大小
            osizeD,             // 输出数据深度维度大小
            osizeH,             // 输出数据高度维度大小
            osizeW,             // 输出数据宽度维度大小
            istrideB,           // 批次维度的步幅
            istrideC,           // 通道维度的步幅
            istrideD,           // 深度维度的步幅
            istrideH,           // 高度维度的步幅
            istrideW);          // 宽度维度的步幅
    });
  // 结束函数模板特化的定义
  }
  
  // 定义了一个模板函数 _qavg_pool_nhwc_kernel，用于执行 NHWC 格式的平均池化操作
  template <typename T>
  void _qavg_pool_nhwc_kernel(
      // 输入张量 qx，输出张量 qy，以及各维度的大小参数
      const Tensor& qx,
      Tensor& qy,
      int64_t nBatch,
      int64_t nInputPlane,
      int64_t inputWidth,
      int64_t inputHeight,
      int64_t inputDepth,
      int64_t outputWidth,
      int64_t outputHeight,
      int64_t outputDepth,
      int kW,
      int kH,
      int kD,
      int dW,
      int dH,
      int dD,
      int padW,
      int padH,
      int padD,
      bool count_include_pad,
      std::optional<int64_t> divisor_override) {
    // 获取输入和输出数据指针
    T* idata = static_cast<T*>(qx.data_ptr());
    T* odata = static_cast<T*>(qy.data_ptr());
    
    // 计算在 NHWC 格式下的各个维度的步长
    int strideC = 1;
    int strideW = strideC * nInputPlane;
    int istrideH = strideW * inputWidth;
    int istrideD = istrideH * inputHeight;
    int istrideB = istrideD * inputDepth;
    
    // 将这些操作移出循环以减少访问开销
    // 获取输入和输出的缩放因子和零点
    float input_scale = qx.q_scale();
    float output_scale = qy.q_scale();
    int input_zero_point = qx.q_zero_point();
    int output_zero_point = qy.q_zero_point();
    
    // 如果有指定的除数覆盖值，使用该值作为除数，否则使用默认值 0
    int64_t divisor_override_factor =
        divisor_override.has_value() ? divisor_override.value() : 0;
    
    // 使用并行计算，对每个批次中的输出元素执行操作
    at::parallel_for(0, nBatch * outputDepth * outputHeight * outputWidth, 0, [&](int64_t begin, int64_t end) {
      int64_t b{0}, od{0}, oh{0}, ow{0};
      // 初始化数据索引，确定当前处理的数据位置
      data_index_init(begin, b, nBatch, od, outputDepth, oh, outputHeight, ow, outputWidth);
    // 遍历指定范围内的索引 i
    for (const auto i : c10::irange(begin, end)) {
      // 计算输入数据的起始指针 i_p 和输出数据的起始指针 o_p
      auto* i_p = reinterpret_cast<typename T::underlying*>(idata + b * istrideB);
      auto* o_p = reinterpret_cast<typename T::underlying*>(odata + i * strideW);
      // 计算当前输出的起始位置 (dstart, hstart, wstart)
      int dstart = od * dD - padD;
      int hstart = oh * dH - padH;
      int wstart = ow * dW - padW;

      // 计算当前输出的结束位置 (dend, hend, wend)
      int dend = std::min(dstart + kD, (int)inputDepth + padD);
      int hend = std::min(hstart + kH, (int)inputHeight + padH);
      int wend = std::min(wstart + kW, (int)inputWidth + padW);
      // 计算池化窗口的大小
      int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);

      // 确保起始位置和结束位置在有效范围内
      dstart = std::max(dstart, 0);
      hstart = std::max(hstart, 0);
      wstart = std::max(wstart, 0);
      dend = std::min(dend, (int)inputDepth);
      hend = std::min(hend, (int)inputHeight);
      wend = std::min(wend, (int)inputWidth);

      // 计算有效数据区域的大小
      int size = (dend - dstart) * (hend - hstart) * (wend - wstart);
      // 根据是否包含填充区域，确定分母大小
      int divide_size = count_include_pad ? pool_size : size;
      // 确定除数因子，可以选择覆盖的因子或者动态计算的分母
      int divide_factor =
          divisor_override_factor ? divisor_override_factor : divide_size;
      // 计算乘数，用于输入和输出数据的缩放
      float multiplier = input_scale / output_scale  / divide_factor;
      // 计算输入零点乘以大小的负值
      int input_zero_point_m_size = -input_zero_point * size;

      // 设置通道起始索引
      int c_start = 0;

      // 对于 int8 量化，使用 int32 作为累加器处理
      // 否则，将会采用较慢的路径处理
      // TODO: 支持16位、32位等其他路径
      // 调用 AVX 指令集下的平均池化操作，处理当前数据块
      do_avg_pool_nhwc_on_AVX_n<T>(
          i_p,
          o_p,
          c_start,
          input_zero_point_m_size,
          output_zero_point,
          multiplier,
          dstart,
          dend,
          hstart,
          hend,
          wstart,
          wend,
          inputDepth,
          inputHeight,
          inputWidth,
          nInputPlane);

      // 1) 以下循环处理剩余通道
      // 2) 同时处理非 AVX2 路径
      for (const auto c: c10::irange(c_start, nInputPlane)) {
        // 使用 int32 作为累加器进行初始化
        int32_t acc_int32 = input_zero_point_m_size;
        // 遍历所有深度、高度和宽度的像素值，并累加到 acc_int32 中
        for (const auto id : c10::irange(dstart, dend)) {
          for (const auto ih : c10::irange(hstart, hend)) {
            for (const auto iw : c10::irange(wstart, wend)) {
              auto val =
                  *(i_p + id * istrideD + ih * istrideH + iw * strideW +
                  c * strideC);
              acc_int32 += val;
            }
          }
       }
       // 将累加器转换为浮点数
       double acc_fp = acc_int32 * 1.0;
       // 对结果进行截断处理，并存储到输出数据的对应通道中
       o_p[c] = at::native::quantize_val<T>(
           1.0f / multiplier, output_zero_point, acc_fp)
           .val_;
      } // c

      // 更新数据索引步长
      data_index_step(b, nBatch, od, outputDepth, oh, outputHeight, ow, outputWidth);
    }
  });
}

// 定义了一个处理 NHWC 格式的二维平均池化的函数，使用量化整数类型
void qavg_pool2d_nhwc_kernel(
    // 输入量化张量 qx，输出量化张量 qy，当前批次索引 b，输入平面数 nInputPlane
    const Tensor& qx,
    Tensor& qy,
    int64_t b,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t outputWidth,
    int64_t outputHeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad,
    // 可选参数：覆盖除数的整数值
    std::optional<int64_t> divisor_override) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "avg_pool2d_nhwc", [&]() {
    // 调用具体的池化内核函数 _qavg_pool_nhwc_kernel，对输入进行处理
    _qavg_pool_nhwc_kernel<scalar_t>(
      qx,
      qy,
      b,
      nInputPlane,
      inputWidth,
      inputHeight,
      1, // 输入的深度为1，适用于二维池化
      outputWidth,
      outputHeight,
      1, // 输出的深度为1，适用于二维池化
      kW,
      kH,
      1, // 二维池化的深度维度为1
      dW,
      dH,
      1, // 二维池化的深度维度为1
      padW,
      padH,
      0, // 未使用 dilation
      count_include_pad,
      divisor_override);
  });
}

// 定义了一个处理 NHWC 格式的三维平均池化的函数，使用量化整数类型
void qavg_pool3d_nhwc_kernel(
    // 输入量化张量 qx，输出量化张量 qy，当前批次索引 b，输入平面数 nInputPlane
    const Tensor& qx,
    Tensor& qy,
    int64_t b,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t inputDepth,
    int64_t outputWidth,
    int64_t outputHeight,
    int64_t outputDepth,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool count_include_pad,
    // 可选参数：覆盖除数的整数值
    std::optional<int64_t> divisor_override) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "avg_pool3d_nhwc", [&]() {
    // 调用具体的池化内核函数 _qavg_pool_nhwc_kernel，对输入进行处理
    _qavg_pool_nhwc_kernel<scalar_t>(
      qx,
      qy,
      b,
      nInputPlane,
      inputWidth,
      inputHeight,
      inputDepth,
      outputWidth,
      outputHeight,
      outputDepth,
      kW,
      kH,
      kD,
      dW,
      dH,
      dD,
      padW,
      padH,
      padD,
      count_include_pad,
      divisor_override);
  });
}

// 使用 AVX 指令集进行量化双线性插值的函数模板
template <typename T>
int64_t do_quantized_bilinear_on_AVX_n(
    // 第一个输入位置的指针，第二个输入位置的指针
    const typename T::underlying*& pos1,
    typename T::underlying*& pos2,
    // 输入高度、宽度，输出高度、宽度，通道数，输出的零点，输入的零点
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t channels,
    int32_t output_zero_point,
    int32_t input_zero_point,
    // 逆标度，四个插值参数
    float inverse_scale,
    const float h0lambda,
    const float h1lambda,
    const float w0lambda,
    const float w1lambda,
    const int64_t h1p,
    const int64_t w1p) {
  int64_t c = 0;
  // 如果支持 AVX2 或 AVX512 并且不是 MSVC 编译器
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && !defined(_MSC_VER)
  // 确定向量化宽度
  constexpr auto vec_width = Vectorized<T>::size() / 4;
#ifdef CPU_CAPABILITY_AVX2
  // 如果是 AVX2
  if (vec_width == 8) {
#else
  // 如果是 AVX512
  if (vec_width == 16) {
#endif
    for (; c + vec_width <= channels; c += vec_width) {
      // 循环处理每个通道，直到剩余通道数不足一个向量的长度
      Vectorized<float> pos1_fp_v[4];  // 创建四个浮点向量，用于存储 pos1 的浮点表示
      Vectorized<int32_t> pos1_int_v[4];  // 创建四个整型向量，用于存储 pos1 的整数表示
      pos1_int_v[0] = vec::convert_to_int32<typename T::underlying>(pos1);  // 将 pos1 转换为整型向量
      pos1_int_v[1] = vec::convert_to_int32<typename T::underlying>(
          pos1 + w1p * channels);  // 计算偏移后的 pos1，并转换为整型向量
      pos1_int_v[2] = vec::convert_to_int32<typename T::underlying>(
          pos1 + h1p * input_width * channels);  // 计算偏移后的 pos1，并转换为整型向量
      pos1_int_v[3] = vec::convert_to_int32<typename T::underlying>(
          pos1 + (h1p * input_width + w1p) * channels);  // 计算偏移后的 pos1，并转换为整型向量
      for (const auto i : c10::irange(4)) {
        int32_t pos1_int[vec_width];  // 创建整型数组，用于存储整型向量的值
        float pos1_fp[vec_width];  // 创建浮点数组，用于存储浮点向量的值
        pos1_int_v[i].store(pos1_int);  // 将整型向量存储到整型数组中
        vec::convert(pos1_int, pos1_fp, vec_width);  // 将整型数组转换为浮点数组
        pos1_fp_v[i] = Vectorized<float>::loadu(pos1_fp, 8);  // 加载浮点数组到浮点向量中
      }
      Vectorized<float> h0lambda_v(h0lambda);  // 创建浮点向量，加载 h0lambda 的值
      Vectorized<float> h1lambda_v(h1lambda);  // 创建浮点向量，加载 h1lambda 的值
      Vectorized<float> w0lambda_v(w0lambda);  // 创建浮点向量，加载 w0lambda 的值
      Vectorized<float> w1lambda_v(w1lambda);  // 创建浮点向量，加载 w1lambda 的值
      Vectorized<float> input_zero_point_v(input_zero_point);  // 创建浮点向量，加载 input_zero_point 的值
      Vectorized<float> result =
          h0lambda_v * (w0lambda_v * pos1_fp_v[0] + w1lambda_v * pos1_fp_v[1]) +  // 计算结果向量的第一部分
          h1lambda_v * (w0lambda_v * pos1_fp_v[2] + w1lambda_v * pos1_fp_v[3]) -  // 计算结果向量的第二部分
          input_zero_point_v;  // 减去 input_zero_point
      float result_fp[vec_width];  // 创建浮点数组，用于存储结果向量的值
      result.store(result_fp);  // 将结果向量存储到浮点数组中
      at::native::quantize_vec<T>(
          inverse_scale,
          output_zero_point,
          result_fp,
          reinterpret_cast<T*>(pos2),
          vec_width);  // 对结果向量进行量化并存储到 pos2 中
      pos1 += vec_width;  // 更新 pos1 的位置
      pos2 += vec_width;  // 更新 pos2 的位置
    }
#endif
  // 返回计数器变量 c，该变量可能是一个计数器或指针
  return c;
}

void qupsample_bilinear2d_nhwc_kernel(
    // 输出张量，用于存储双线性插值结果
    Tensor& output,
    // 输入张量，包含待插值的数据
    const Tensor& input,
    // 输入张量的高度
    int64_t input_height,
    // 输入张量的宽度
    int64_t input_width,
    // 输出张量的高度
    int64_t output_height,
    // 输出张量的宽度
    int64_t output_width,
    // 批量大小
    int64_t nbatch,
    // 通道数
    int64_t channels,
    // 是否对齐角点进行插值
    bool align_corners,
    // 高度缩放因子（可选）
    std::optional<double> scales_h,
    // 宽度缩放因子（可选）
    std::optional<double> scales_w) {
  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "upsample_bilinear2d_nhwc", [&]() {
    // 获取输入数据的指针，类型转换为对应的标量类型
    auto* idata = static_cast<scalar_t*>(input.data_ptr());
    // 获取输出数据的指针，类型转换为对应的标量类型
    auto* odata = static_cast<scalar_t*>(output.data_ptr());
    // 计算输出张量的比例因子，用于缩放输出量化值到输入的范围
    float inverse_scale = output.q_scale() / input.q_scale();
    // 计算高度的插值比例
    const auto rheight = area_pixel_compute_scale<float>(
        input_height, output_height, align_corners, scales_h);
    // 计算宽度的插值比例
    const auto rwidth = area_pixel_compute_scale<float>(
        input_width, output_width, align_corners, scales_w);

    // 获取输入张量的量化零点
    auto input_q_zero_point = input.q_zero_point();
    // 获取输出张量的量化零点
    auto output_q_zero_point = output.q_zero_point();
    // 并行循环，按照指定范围并行执行任务
    at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
      // 初始化循环变量和数据索引
      int64_t b{0}, h2{0}, w2{0};
      data_index_init(begin, b, nbatch, h2, output_height, w2, output_width);

      // 循环处理指定范围内的索引
      for (C10_UNUSED const auto i : c10::irange(begin, end)) {
        // 计算输入和输出数据的指针位置
        auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(
            idata + b * input_height * input_width * channels);
        auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(
            odata + b * output_height * output_width * channels);

        // 计算高度和宽度的源索引
        const auto h1r = area_pixel_compute_source_index<float>(
            rheight, h2, align_corners, /*cubic=*/false);

        const int64_t h1 = h1r;
        const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1.) - h1lambda;

        // 计算宽度的源索引
        const auto w1r = area_pixel_compute_source_index<float>(
            rwidth, w2, align_corners, /*cubic=*/false);
        const int64_t w1 = w1r;
        const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = static_cast<float>(1.) - w1lambda;

        int64_t c = 0;

        // 使用 AVX 指令集执行量化双线性插值
        const typename scalar_t::underlying* pos1 =
            i_p + (h1 * input_width + w1) * channels;
        typename scalar_t::underlying* pos2 =
            o_p + (h2 * output_width + w2) * channels;
        c = do_quantized_bilinear_on_AVX_n<scalar_t>(
            pos1,
            pos2,
            input_height,
            input_width,
            output_height,
            output_width,
            channels,
            output_q_zero_point,
            input_q_zero_point,
            inverse_scale,
            h0lambda,
            h1lambda,
            w0lambda,
            w1lambda,
            h1p,
            w1p);

        // 处理剩余的通道和非 AVX2 路径
        // 计算输出像素值并进行量化
        for (; c < channels; ++c) {
          float result = h0lambda *
                  (w0lambda * pos1[0] + w1lambda * pos1[w1p * channels]) +
              h1lambda *
                  (w0lambda * pos1[h1p * input_width * channels] +
                   w1lambda * pos1[(h1p * input_width + w1p) * channels]);
          pos2[0] = at::native::quantize_val<scalar_t>(
                        inverse_scale,
                        output_q_zero_point,
                        result - input_q_zero_point)
                        .val_;
          pos1 += 1;
          pos2 += 1;
        } // c

        // 更新数据索引，准备处理下一个像素
        data_index_step(b, nbatch, h2, output_height, w2, output_width);
      }
    });
  });
}

void qtopk_kernel(Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  auto sizes = self.sizes();  // 获取输入张量的尺寸信息
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)  // 设置迭代器配置，不检查所有张量是否具有相同的数据类型
    .resize_outputs(false)  // 禁止自动调整输出张量的大小
    .declare_static_shape(sizes, /*squash_dims=*/dim)  // 声明静态形状，可能压缩维度dim
    .add_output(values)  // 添加输出张量 values
    .add_output(indices)  // 添加输出张量 indices
    .add_input(self)  // 添加输入张量 self
    .build();  // 构建张量迭代器

  auto mode_values_stride = values.strides()[dim];  // 获取 values 张量在 dim 维度上的步幅
  auto mode_indices_stride = indices.strides()[dim];  // 获取 indices 张量在 dim 维度上的步幅
  auto tmp_values_stride = self.strides()[dim];  // 获取 self 张量在 dim 维度上的步幅
  // 如果 sizes 为空，说明张量是标量。这防止访问空数组。
  auto dim_size = sizes.empty() ? 1 : sizes[dim];  // 获取 dim 维度的尺寸，如果 sizes 为空则默认为1

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qtopk_cpu", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {  // 定义循环函数，处理 QINT 类型的数据
      using underlying_t = typename scalar_t::underlying;
      static_assert(sizeof(scalar_t) == sizeof(underlying_t), "");  // 静态断言确保 scalar_t 与其底层类型 underlying_t 的大小相同
      return topk_impl_loop<underlying_t, underlying_t>(
          mode_values_stride, mode_indices_stride, tmp_values_stride,
          k, dim_size, largest, sorted, data, strides, n);  // 调用 topk_impl_loop 函数进行实际的 top-k 计算
    };

    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);  // 计算用于并行化的粒度大小
    iter.for_each(loop, /*grain_size=*/grain_size);  // 使用迭代器并行处理数据，设置粒度大小为 grain_size
  });
}

template <typename T>
inline void do_bn_compute(
    typename T::underlying* X_ptr,
    typename T::underlying* Y_ptr,
    Vectorized<float> & fake_scale,
    Vectorized<float> & in_zp_vec,
    Vectorized<float> & scale_neg_zp_premul,
    int64_t out_zero_point,
    Vectorized<T> & out_zero_point_v,
    float*  alpha,
    float* beta,
    int64_t vec_num,
    bool ReluFused,
    int64_t kVLen
) {
  using Vec = Vectorized<T>;
  auto vals_q = Vec::loadu(X_ptr);  // 加载未对齐的输入数据到 Vec 类型的 vals_q
  // 这里使用假的比例尺为 1.0，不影响性能（替代了减法的 FMA）
  auto vals_dq = vals_q.dequantize(fake_scale, in_zp_vec, scale_neg_zp_premul);  // 对输入数据进行反量化

  for (const auto idx : c10::irange(vec_num)) {  // 遍历向量化的元素数量
    auto alpha_v = Vectorized<float>::loadu(alpha + idx * kVLen);  // 加载未对齐的 alpha 数据到 alpha_v
    auto beta_v = Vectorized<float>::loadu(beta + idx * kVLen);  // 加载未对齐的 beta 数据到 beta_v
    vals_dq[idx] = vec::fmadd(alpha_v, vals_dq[idx], beta_v);  // 使用 FMA 计算 alpha_v * vals_dq[idx] + beta_v
  }

  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto outputs_q = Vec::quantize(vals_dq, /*output_scale=*/1.0f, out_zero_point, /*inv_output_scale=*/1.0f);  // 对结果进行量化，使用假的比例尺为 1.0

  // 再次使用假的比例尺
  if (ReluFused) {
    outputs_q = outputs_q.maximum(out_zero_point_v);  // 如果启用了 ReluFused，应用 ReLU 操作
  }
  outputs_q.store(Y_ptr, vec_num * kVLen);  // 存储结果到输出指针 Y_ptr，乘以向量长度 kVLen
}

template <bool ReluFused>
void q_batch_norm_kernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t in_zero_point,
    int64_t out_zero_point,
    const Tensor& input,
    const Tensor& a,
    const Tensor& b,
    Tensor& output) {

  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qbatch_norm", [&]() {
    float* alpha = a.data_ptr<float>();  // 获取 alpha 张量的数据指针
    float* beta = b.data_ptr<float>();  // 获取 beta 张量的数据指针
    auto minimum = std::numeric_limits<scalar_t::underlying>::lowest();  // 获取底层类型的最小值
    // 计算 scalar_t::underlying 类型的最大值
    auto maximum = std::numeric_limits<scalar_t::underlying>::max();
    // 将 input 的数据指针转换为 scalar_t::underlying* 类型
    scalar_t::underlying* X = reinterpret_cast<scalar_t::underlying*>(input.data_ptr());
    // 将 output 的数据指针转换为 scalar_t::underlying* 类型
    scalar_t::underlying* Y = reinterpret_cast<scalar_t::underlying*>(output.data_ptr());

    // 定义常量 kVLen 为 Vectorized<float> 的大小
    constexpr int kVLen = Vectorized<float>::size();
    // 计算 outer_size，表示数据批次大小 N * HxW
    const int64_t outer_size = N * HxW;
    // 使用 Vec 表示 Vectorized<scalar_t> 类型
    using Vec = Vectorized<scalar_t>;
    // Hoisted variables，提前计算的变量
    auto in_zp_vec = Vectorized<float>(static_cast<float>(in_zero_point));
    auto fake_scale = Vectorized<float>(1.0f);
    auto scale_neg_zp_premul = fake_scale * in_zp_vec.neg();
    auto out_zero_point_v = Vec(scalar_t(out_zero_point));
    // 计算 lanes，表示每个向量化运算的元素数目
    const auto lanes = static_cast<int64_t>(Vec::float_num_vecs() * kVLen);

    // 使用并行处理，将任务分配给多个线程
    at::parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        // 计算当前位置的 X_ptr 和 Y_ptr
        auto* X_ptr = reinterpret_cast<typename scalar_t::underlying*>(X + i * C);
        auto* Y_ptr = reinterpret_cast<typename scalar_t::underlying*>(Y + i * C);
        // 初始化通道数为 0
        int64_t ch = 0;

        // 对于每组 lanes 个通道的数据执行批量归一化计算
        for(; ch + lanes <= C; ch += lanes) {
          do_bn_compute<scalar_t>(
            X_ptr + ch,
            Y_ptr + ch,
            fake_scale,
            in_zp_vec,
            scale_neg_zp_premul,
            out_zero_point,
            out_zero_point_v,
            alpha + ch,
            beta + ch,
            Vec::float_num_vecs(),
            ReluFused,
            kVLen
          );
        }

        // 对于通道数在 8 到 32 之间，依然使用 32 宽度进行性能优化
        // 实验表明，比每次处理 8 个通道更快
        int64_t elem_size = C - ch;
        if ((lanes == 32) && elem_size >= kVLen) {
          int64_t vec_num = elem_size / kVLen;
          // 创建缓冲区 buf_in，存储从 X_ptr + ch 处复制的数据
          std::vector<typename scalar_t::underlying> buf_in(lanes);
          // 将数据从 X_ptr + ch 处复制到 buf_in 中
          memcpy(buf_in.data(), X_ptr + ch, vec_num * kVLen * sizeof(typename scalar_t::underlying)); // 3 cycles
          // 执行批量归一化计算
          do_bn_compute<scalar_t>(
            buf_in.data(),
            Y_ptr + ch,
            fake_scale,
            in_zp_vec,
            scale_neg_zp_premul,
            out_zero_point,
            out_zero_point_v,
            alpha + ch,
            beta + ch,
            vec_num,
            ReluFused,
            kVLen
          );
          ch += vec_num * kVLen;
        }

        // 对于小于 8 个通道的情况，逐个通道进行量化计算
        for (; ch < C; ++ch) {
          // 计算量化值 quantized_down
          long quantized_down = out_zero_point +
              lrintf(alpha[ch] * (X_ptr[ch] - in_zero_point) +
                          beta[ch]);
          // 如果开启了 ReluFused，则进行 ReLU 操作
          if (ReluFused) { // static if
            quantized_down = std::max<long>(quantized_down, out_zero_point);
          }
          // 将量化结果限制在 [minimum, maximum] 范围内，并存储到 Y_ptr[ch] 中
          Y_ptr[ch] = std::min<long>(
              std::max<long>(quantized_down, minimum), maximum);
        }
      }
    });
void _fake_quantize_tensor_helper(
  Tensor& output,  // 输出张量，用于存储量化后的结果
  Tensor& mask,    // 掩码张量，标记哪些值经过了量化
  const Tensor& input,  // 输入张量，待量化的原始数据
  int fake_quant_on,     // 是否开启伪量化的标志
  float sc,              // 缩放因子，用于量化操作
  int64_t z_point,       // 零点偏移量，用于量化操作
  int64_t quant_min,     // 最小量化值
  int64_t quant_max) {   // 最大量化值

  float inv_scale = 1.0f / sc;  // 计算缩放因子的倒数

  auto iter_combined = TensorIteratorConfig()  // 创建张量迭代器配置
    .check_all_same_dtype(false)               // 不检查所有张量是否具有相同的数据类型
    .add_output(output)                       // 将输出张量添加到迭代器
    .add_output(mask)                         // 将掩码张量添加到迭代器
    .add_input(input)                         // 将输入张量添加到迭代器
    .build();                                 // 构建张量迭代器

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_type_handling", [&] {
    iter_combined.for_each([&](char** data, const int64_t* strides, int64_t n) {  // 使用迭代器并行处理每个张量元素
      for (const auto i : c10::irange(n)) {  // 遍历每个元素
        scalar_t* output_val = (scalar_t*)(data[0] + i * strides[0]);  // 输出张量当前元素的指针
        bool* mask_val = (bool*)(data[1] + i * strides[1]);            // 掩码张量当前元素的指针
        scalar_t* input_val = (scalar_t*)(data[2] + i * strides[2]);    // 输入张量当前元素的指针

        const auto qval = static_cast<int64_t>(z_point + std::nearbyint(*input_val * inv_scale));  // 计算量化后的值
        if (fake_quant_on) {
          *output_val = (std::fmin(std::fmax(qval, quant_min), quant_max) - z_point) * sc;  // 应用伪量化函数到输出张量
          *mask_val = ((quant_min <= qval) && (qval <= quant_max));  // 更新掩码张量，标记量化的有效值
        } else {
          *output_val = *input_val;  // 如果未开启伪量化，则直接将输入复制到输出张量
          *mask_val = 1;             // 并标记所有值为有效
        }
      }
    });
  });
}

void fake_quantize_tensor_cachemask_kernel(
    Tensor& output,          // 输出张量，用于存储量化后的结果
    Tensor& mask,            // 掩码张量，标记哪些值经过了量化
    const Tensor& input,     // 输入张量，待量化的原始数据
    float sc,                // 缩放因子，用于量化操作
    int64_t z_point,         // 零点偏移量，用于量化操作
    int64_t quant_min,       // 最小量化值
    int64_t quant_max) {     // 最大量化值
  _fake_quantize_tensor_helper(output, mask, input, 1, sc, z_point, quant_min, quant_max);  // 调用帮助函数进行伪量化操作
}

void fake_quantize_tensor_cachemask_tensor_qparams_kernel(
    Tensor& output,                  // 输出张量，用于存储量化后的结果
    Tensor& mask,                    // 掩码张量，标记哪些值经过了量化
    const Tensor& input,             // 输入张量，待量化的原始数据
    const Tensor& sc,                // 缩放因子张量，用于量化操作
    const Tensor& z_point,           // 零点偏移量张量，用于量化操作
    const Tensor& fake_quant_enabled,// 是否开启伪量化的张量
    int64_t quant_min,               // 最小量化值
    int64_t quant_max) {             // 最大量化值
  _fake_quantize_tensor_helper(output, mask, input, fake_quant_enabled.item().toInt(), sc.item().toFloat(), z_point.item().toInt(), quant_min, quant_max);  // 调用帮助函数进行伪量化操作，从张量中提取参数并转换为对应类型
}

void fake_quantize_learnable_tensor_grad_kernel_cpu(
    TensorIterator& iter,     // 张量迭代器，用于处理梯度计算
    float scale,              // 缩放因子
    float inv_scale,          // 缩放因子的倒数
    int64_t zero_point,       // 零点偏移量
    int64_t quant_min,        // 最小量化值
    int64_t quant_max,        // 最大量化值
    float grad_factor) {      // 梯度因子
  float dscale_small = quant_min - zero_point;  // 计算小范围的缩放因子
  float dscale_big = quant_max - zero_point;    // 计算大范围的缩放因子
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {  // 使用迭代器处理每个张量元素
    /*  当在一个 TensorIterator 上调用 for_each 时，如果其有多个输入和输出，
        它们被访问的顺序遵循它们在迭代器中构建的顺序。
        例如，如果迭代器按照以下顺序构建：
        auto iter = TensorIteratorConfig().
          .add_output(firstOutput)
          .add_output(secondOutput)
          .add_input(firstInput)
          .add_input(secondInput)
          .build()
        data 数组将按照以下顺序包含 4 个指向值的指针：
        firstOutput, secondOutput, firstInput, secondInput。
        通过适当的指针引用和解引用，以及使用步长（用于移动到不同的元素），可以访问输入并对输出进行赋值。
    */
    for (const auto i : c10::irange(n)) {
      // 计算 X 的梯度。
      float* dXOutput = (float*)(data[0] + i * strides[0]);
      float* dScaleOutput = (float*)(data[1] + i * strides[1]);
      float* dZeroPointOutput = (float*)(data[2] + i * strides[2]);
      float* XInput = (float*)(data[3] + i * strides[3]);
      float* dYInput = (float*)(data[4] + i * strides[4]);
      
      // 计算 xqi，这是 std::nearbyint(zero_point + (*XInput) * inv_scale) 的结果。
      int64_t xqi = std::nearbyint(zero_point + (*XInput) * inv_scale);
      
      // 根据 clamp 函数的梯度计算 X 的梯度。
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      *dXOutput = (*dYInput) * (xqi >= quant_min && xqi <= quant_max);
      
      // 计算 scale 和 zero point 的梯度。
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      float xfqi = static_cast<float>((std::max(std::min(xqi, quant_max), quant_min) - zero_point) * scale);
      
      // 根据 clamp 函数的梯度计算 scale 和 zero point 的梯度。
      if (xqi < quant_min || xqi > quant_max) {
        *dZeroPointOutput = (*dYInput) * (-1) * scale * grad_factor;
        *dScaleOutput = ((xqi < quant_min) ? ((*dYInput) * dscale_small) : ((*dYInput) * dscale_big)) * grad_factor;
      } else {
        *dZeroPointOutput = 0;
        *dScaleOutput = (*dYInput) * (xfqi - (*XInput)) * inv_scale * grad_factor;
      }
    }
}

template <typename SelfType>
void _fake_quant_per_channel_cachemask_cpu_helper(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    const int64_t quant_min,
    const int64_t quant_max) {

  // 获取零点的数据类型
  const auto& zero_point_dtype = iter.input_dtype(2);

  // 如果零点数据类型为浮点型
  if(at::isFloatingType(zero_point_dtype)){
    // 当零点为浮点型时，执行仿量化镜像仿射量化器方程
    // Xq = Round(Xf * inv_scale + zero_point)
    // 其中 zero_point 是浮点数

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(zero_point_dtype, "fake_quantize_channel_cachemask_cpu_zero_point_handling", [&] {
      // 写入掩码
      cpu_kernel(iter_mask, [=](SelfType self, float scale, scalar_t zero_point) -> bool {
        float inv_scale = 1.0f / scale;
        const auto qval = std::lrintf(zero_point + (self * inv_scale));
        return ((quant_min <= qval) && (qval <= quant_max));
      });

      // 写入假量化
      cpu_kernel(iter, [=](SelfType self, float scale, scalar_t zero_point) -> SelfType {
        float inv_scale = 1.0f / scale;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        return (std::fmin(
                    std::fmax(
                        std::lrintf(zero_point + self * inv_scale),
                        quant_min),
                    quant_max) -
                zero_point) *
            scale;
      });
    });

  } else {
      // 当零点数据类型不是浮点型时

      // 写入掩码
      cpu_kernel(iter_mask, [=](SelfType self, float scale, int32_t zero_point) -> bool {
        float inv_scale = 1.0f / scale;
        const auto qval = static_cast<int64_t>(zero_point + std::nearbyint(self * inv_scale));
        return ((quant_min <= qval) && (qval <= quant_max));
      });

      // 写入假量化
      cpu_kernel(iter, [=](SelfType self, float scale, int32_t zero_point) -> SelfType {
        float inv_scale = 1.0f / scale;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        return (std::fmin(
                    std::fmax(
                        static_cast<int64_t>(
                            zero_point + std::nearbyint(self * inv_scale)),
                        quant_min),
                    quant_max) -
                zero_point) *
            scale;
      });
  }

}


void fake_quant_per_channel_cachemask_cpu(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    int64_t quant_min,
    int64_t quant_max) {
  // TODO(future, optional): read once, write twice.  Not done at the moment
  //   for simplicity, as we do not expect this to be a bottleneck.

  // 分派浮点类型和半精度类型的函数处理器
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "fake_quantize_channel_cachemask_cpu_type_handling", [&] {
    // 调用具体的仿量化处理函数
    _fake_quant_per_channel_cachemask_cpu_helper<scalar_t>(iter, iter_mask, quant_min, quant_max);
  });
}


void fake_quantize_learnable_channel_grad_kernel_cpu(
    TensorIterator& iter,
    int64_t quant_min,
    int64_t quant_max,



// 结束函数 _fake_quant_per_channel_cachemask_cpu_helper

template <typename SelfType>
void _fake_quant_per_channel_cachemask_cpu_helper(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    const int64_t quant_min,
    const int64_t quant_max) {

  // 获取零点的数据类型
  const auto& zero_point_dtype = iter.input_dtype(2);

  // 如果零点数据类型为浮点型
  if(at::isFloatingType(zero_point_dtype)){
    // 当零点为浮点型时，执行仿量化镜像仿射量化器方程
    // Xq = Round(Xf * inv_scale + zero_point)
    // 其中 zero_point 是浮点数

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(zero_point_dtype, "fake_quantize_channel_cachemask_cpu_zero_point_handling", [&] {
      // 写入掩码
      cpu_kernel(iter_mask, [=](SelfType self, float scale, scalar_t zero_point) -> bool {
        float inv_scale = 1.0f / scale;
        const auto qval = std::lrintf(zero_point + (self * inv_scale));
        return ((quant_min <= qval) && (qval <= quant_max));
      });

      // 写入假量化
      cpu_kernel(iter, [=](SelfType self, float scale, scalar_t zero_point) -> SelfType {
        float inv_scale = 1.0f / scale;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        return (std::fmin(
                    std::fmax(
                        std::lrintf(zero_point + self * inv_scale),
                        quant_min),
                    quant_max) -
                zero_point) *
            scale;
      });
    });

  } else {
      // 当零点数据类型不是浮点型时

      // 写入掩码
      cpu_kernel(iter_mask, [=](SelfType self, float scale, int32_t zero_point) -> bool {
        float inv_scale = 1.0f / scale;
        const auto qval = static_cast<int64_t>(zero_point + std::nearbyint(self * inv_scale));
        return ((quant_min <= qval) && (qval <= quant_max));
      });

      // 写入假量化
      cpu_kernel(iter, [=](SelfType self, float scale, int32_t zero_point) -> SelfType {
        float inv_scale = 1.0f / scale;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        return (std::fmin(
                    std::fmax(
                        static_cast<int64_t>(
                            zero_point + std::nearbyint(self * inv_scale)),
                        quant_min),
                    quant_max) -
                zero_point) *
            scale;
      });
  }

}


// 假量化每通道带缓存掩码的 CPU 实现
void fake_quant_per_channel_cachemask_cpu(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    int64_t quant_min,
    int64_t quant_max) {
  // TODO(future, optional): read once, write twice.  Not done at the moment
  //   for simplicity, as we do not expect this to be a bottleneck.

  // 分派浮点类型和半精度类型的函数处理器
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "fake_quantize_channel_cachemask_cpu_type_handling", [&] {
    // 调用具体的仿量化处理函数
    _fake_quant_per_channel_cachemask_cpu_helper<scalar_t>(iter, iter_mask, quant_min, quant_max);
  });
}


void fake_quantize_learnable_channel_grad_kernel_cpu(
    TensorIterator& iter,
    int64_t quant_min,
    int64_t quant_max,
    // 对每个迭代器执行以下操作：
    // 1. 获取每个数据指针和步长数组，并处理 n 个元素
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
        // 遍历每个元素 i
        for (const auto i : c10::irange(n)) {
            // 指向输出的梯度数组的指针
            float* dx_output = (float*)(data[0] + i * strides[0]);
            float* dscale_output = (float*)(data[1] + i * strides[1]);
            float* dzero_point_output = (float*)(data[2] + i * strides[2]);
            // 指向输入的数据数组的指针
            float* x_input = (float*)(data[3] + i * strides[3]);
            float* dy_input = (float*)(data[4] + i * strides[4]);
            float* scale_input = (float*)(data[5] + i * strides[5]);
            float* zero_point_input = (float*)(data[6] + i * strides[6]);
    
            // 计算输入数据的缩放因子的倒数
            float inv_scale = 1.0f / (*scale_input);
            // 计算量化范围的最小值对应的梯度
            float dscale_small = quant_min - (*zero_point_input);
            // 计算量化范围的最大值对应的梯度
            float dscale_big = quant_max - (*zero_point_input);
    
            // 计算输入数据 x 的量化后的整数值
            int64_t xqi = std::nearbyint((*zero_point_input) + (*x_input) * inv_scale);
    
            // 计算输入数据 x 的梯度
            // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
            *dx_output = (*dy_input) * (xqi >= quant_min && xqi <= quant_max);
    
            // 计算缩放因子和零点的梯度
            float xfqi = static_cast<float>((std::max(std::min(xqi, quant_max), quant_min) - (*zero_point_input)) * (*scale_input));
            if (xqi < quant_min || xqi > quant_max) {
                // 如果 xqi 超出量化范围，则计算零点的梯度
                *dzero_point_output = (*dy_input) * (-1) * (*scale_input) * grad_factor;
                // 如果 xqi 超出量化范围，则计算缩放因子的梯度
                *dscale_output = ((xqi < quant_min) ? ((*dy_input) * dscale_small) : ((*dy_input) * dscale_big)) * grad_factor;
            } else {
                // 如果 xqi 在量化范围内，则设置零点的梯度为 0
                *dzero_point_output = 0;
                // 计算缩放因子的梯度
                *dscale_output = (*dy_input) * (xfqi - (*x_input)) * inv_scale * grad_factor;
            }
        }
    });
// 结束 quantized_normalize_kernel 函数定义

// 假设 X 是由 M 组 N 元素组成。对每组进行标准化，并可选择应用仿射缩放。
// 适用于 LayerNorm、GroupNorm 和 InstanceNorm。
void quantized_normalize_kernel(
    const Tensor& X, // 输入张量
    const Tensor& gamma, // 权重（可选）
    const Tensor& beta, // 偏置（可选）
    bool affine_per_channel, // 如果为 false，则元素级应用缩放；如果为 true，则按通道应用
    int num_channels, // 仅在 affine_per_channel 设置时使用
    int num_groups, // 仅在 affine_per_channel 设置时使用
    int64_t M, // 组数
    int64_t N, // 每组中的元素数
    double eps, // 用于数值稳定性的小数
    Tensor* Y) { // 输出张量指针
  AT_DISPATCH_QINT_TYPES(X.scalar_type(), "quantized_layer_norm_kernel_impl_cpu", [&]() {
    using qVec = vec::Vectorized<scalar_t>; // 定义 qVec 为 scalar_t 类型的向量化操作类型
    using fVec = vec::Vectorized<float>; // 定义 fVec 为 float 类型的向量化操作类型

    // 运行时断言，确保 X 中的元素数量为 M * N
    TORCH_INTERNAL_ASSERT(X.numel() == M * N, "Unexpected num elements in X");

    // 运行时断言，确保 gamma 未定义或其大小符合预期
    TORCH_INTERNAL_ASSERT(
        !gamma.defined() ||
        (!affine_per_channel && gamma.numel() == N) ||
        (affine_per_channel && gamma.numel() == num_channels),
        "Unexpected size of gamma");

    // 运行时断言，确保 beta 未定义或其大小符合预期
    TORCH_INTERNAL_ASSERT(
        !beta.defined() ||
        (!affine_per_channel && beta.numel() == N) ||
        (affine_per_channel && beta.numel() == num_channels),
        "Unexpected size of beta");

    // 获取 X、gamma、beta 和 Y 的数据指针
    scalar_t* X_data = X.data_ptr<scalar_t>();
    const float* gamma_data = gamma.defined() ? gamma.const_data_ptr<float>() : nullptr;
    const float* beta_data = beta.defined() ? beta.const_data_ptr<float>() : nullptr;
    scalar_t* Y_data = Y->data_ptr<scalar_t>();

    // 检查 gamma 和 beta 是否为 null
    const bool gamma_null = gamma_data == nullptr;
    const bool beta_null = beta_data == nullptr;

    // 获取 X 的量化零点和比例因子
    int64_t x_zp = X.q_zero_point();
    float x_scale = X.q_scale();

    // 创建 float 类型的向量化操作对象
    fVec x_zp_vec((float)x_zp);
    fVec one_vec(1.0f);
    fVec zero_vec(0.0f);

    // 设置仿射缩放的虚拟比例因子
    float x_fake_scale = 1.0f;
    fVec x_fake_scale_vec(x_fake_scale);
    fVec x_fake_scale_zp_neg_premul_vec = x_fake_scale_vec * x_zp_vec.neg();

    // 获取 Y 的量化零点和比例因子
    int64_t y_zp = Y->q_zero_point();
    float y_scale = Y->q_scale();
    float y_inv_scale = 1.0f / y_scale;

    // 向量长度和向量化整数向量数
    constexpr int kFloatVLen = fVec::size();
    int64_t kIntVLen = kFloatVLen * qVec::float_num_vecs();
    int64_t kNumIntVecInLayer = N / kIntVLen;
    int64_t kNonVecRemInLayer = N % kIntVLen;
    int channels_per_group = num_channels / num_groups;
    int64_t NPerChannel = N / channels_per_group;
    int64_t kNumIntVecInChannel = NPerChannel / kIntVLen;
    int64_t kNonVecRemInChannel = NPerChannel % kIntVLen;

  }); // parallel_for

} // 结束 quantized_normalize_kernel 函数定义

// qmean_inner_dim_kernel 函数的定义未提供在此片段中
    // 计算输入张量的数据类型
    ScalarType dtype = self.scalar_type();
    // 获取输入张量的维度
    auto in_dims = self.sizes().vec();
    // 复制输入张量的维度作为输出维度的初始值
    auto out_dims = in_dims;
    // 检查是否需要对所有维度进行归约操作
    bool is_all_reduce = !opt_dim.has_value() || opt_dim.value().empty();
    // 计算需要挤压（squeeze）的维度数目
    size_t num_dims_to_squeeze = is_all_reduce ? self.dim() : opt_dim.value().size();
    // 初始化分组数 M 和每组内元素数 N
    int64_t M = 1; // 每组的数量
    int64_t N = 1; // 每组中要求取平均值的元素数量
    // 计算不需要挤压的维度对应的 M
    for (size_t i = 0; i < in_dims.size() - num_dims_to_squeeze; ++i) {
      M *= in_dims[i];
    }
    // 根据需要挤压的维度计算 N 并进行挤压操作
    for (size_t i = 0; i < num_dims_to_squeeze; ++i) {
      auto idx = out_dims.size() - 1 - i;
      N *= out_dims[idx];
      out_dims[idx] = 1;
    }
    // 如果不保持维度，则移除挤压的维度
    if (!keepdim) {
      out_dims.erase(out_dims.end() - num_dims_to_squeeze, out_dims.end());
    }
    // 使用预定义的函数创建一个空张量，用于存放结果
    result = at::_empty_affine_quantized(
        out_dims,
        at::device(kCPU).dtype(dtype).memory_format(self.suggest_memory_format()),
        self.q_scale(),
        self.q_zero_point(),
        c10::nullopt);
    
    // 根据输入张量的数据类型分发量化整数计算函数
    AT_DISPATCH_QINT_TYPES(self.scalar_type(), "quantized_mean_kernel_impl_cpu", [&]() {
      // 获取输入张量和输出结果张量的数据指针
      scalar_t* X_data = self.data_ptr<scalar_t>();
      scalar_t* Y_data = result.data_ptr<scalar_t>();
    
      // 使用并行化方式对 M 组数据进行处理
      at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
        for (const auto i : c10::irange(start, end)) {
          // 定位当前组的输入和输出数据指针
          scalar_t* X_ptr = X_data + i * N;
          scalar_t* Y_ptr = Y_data + i;
          // 将输入数据指针转换为底层类型的指针，以便进行数据和处理
          scalar_t::underlying* X_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(X_ptr);
          scalar_t::underlying* Y_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(Y_ptr);
          // 计算当前组数据的和并计算平均值，将结果保存到输出指针中
          auto x_sum = hsum(X_ptr_underlying, N);
          float y_float = static_cast<float>(x_sum) / N;
          *Y_ptr_underlying = std::nearbyint(y_float);
        }
      });
    });
}

void qstd_inner_dim_kernel(
    const Tensor& self, // 输入张量
    OptionalIntArrayRef dim, // 可选的维度数组引用
    const std::optional<Scalar>& correction_opt, // 可选的修正值
    bool keepdim, // 是否保持维度
    Tensor& result) { // 结果张量引用
  ScalarType dtype = self.scalar_type(); // 获取输入张量的数据类型
  auto in_dims = self.sizes().vec(); // 获取输入张量的维度信息
  auto out_dims = in_dims; // 复制输入维度作为输出维度的初始值
  size_t num_dims_to_squeeze = dim.has_value() && !dim.value().empty() ?
                               dim.value().size() :
                               self.dim(); // 计算需要挤压的维度数目
  int64_t M = 1; // 组数初始化为1
  int64_t N = 1; // 每个组中用于计算标准差的元素数目初始化为1
  for (size_t i = 0; i < in_dims.size() - num_dims_to_squeeze; ++i) {
    M *= in_dims[i]; // 计算 M：前几维的乘积，用于并行处理
  }
  for (size_t i = 0; i < num_dims_to_squeeze; ++i) {
    auto idx = out_dims.size() - 1 - i;
    N *= out_dims[idx]; // 计算 N：需要挤压的维度对应的乘积，用于计算标准差
    out_dims[idx] = 1; // 设置输出维度对应的位置为1
  }
  if (!keepdim) {
    out_dims.erase(out_dims.end() - num_dims_to_squeeze, out_dims.end()); // 如果不保持维度，从输出维度中删除挤压的维度
  }
  const auto correction = correction_opt.value_or(1).toDouble(); // 获取修正值，如果不存在则默认为1，并转换为 double 类型
  double den = std::max(N - correction, 0.0); // 计算均值和标准差时的分母
  auto x_scale = self.q_scale(); // 获取输入张量的量化比例因子
  auto x_zp = self.q_zero_point(); // 获取输入张量的量化零点
  result = at::_empty_affine_quantized(
      out_dims,
      at::device(kCPU).dtype(dtype).memory_format(self.suggest_memory_format()), // 创建一个空的量化张量作为结果
      x_scale,
      x_zp,
      c10::nullopt);

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "quantized_std_kernel_impl_cpu", [&]() {
    scalar_t* X_data = self.data_ptr<scalar_t>(); // 获取输入张量的数据指针
    scalar_t* Y_data = result.data_ptr<scalar_t>(); // 获取结果张量的数据指针

    at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) { // 并行处理组内的元素
      for (const auto i : c10::irange(start, end)) {
        scalar_t* X_ptr = X_data + i * N; // 指向当前组的起始位置
        scalar_t* Y_ptr = Y_data + i; // 指向结果张量中当前组的位置
        scalar_t::underlying* X_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(X_ptr); // 将输入张量的数据指针转换为底层类型指针
        scalar_t::underlying* Y_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(Y_ptr); // 将结果张量的数据指针转换为底层类型指针
        auto x_sum_shifted = hsum(X_ptr_underlying, N); // 计算移位后的和
        auto x_sum_sq_shifted = hsum_sq(X_ptr_underlying, N); // 计算移位后的平方和
        // 使用 double 类型的中间变量以避免精度问题
        // 带有零点的均值
        double x_mean_shifted_div_scale_x = static_cast<double>(x_sum_shifted) / N;
        double x_mean_unbiased_shifted_div_scale_x = static_cast<double>(x_sum_shifted) / den;
        // 方差 / x_scale^2
        double x_var_div_scale_x_sq =
            std::max(static_cast<double>(x_sum_sq_shifted) / den -
                2 * x_mean_shifted_div_scale_x * x_mean_unbiased_shifted_div_scale_x +
                x_mean_shifted_div_scale_x * x_mean_shifted_div_scale_x * N / den, (double)0.0);
        double y_float = std::sqrt(x_var_div_scale_x_sq) * x_scale; // 计算标准差并乘以量化比例因子
        *Y_ptr_underlying = at::native::quantize_val<scalar_t>(
                            x_scale, x_zp, y_float)
                            .val_; // 将浮点数值量化并存储到结果张量中
      }
    });
  });
}

// 用于 channels_last 输入的组归一化
void quantized_groupnorm_nhwc_kernel(
    const Tensor& X, // 输入张量
    const Tensor& gamma, // 权重（可选）
    const Tensor& beta, // 偏置（可选）
    bool affine_per_channel, // 是否对每个通道进行仿射变换，用于组/实例归一化
    int num_channels, // 如果设置了affine_per_channel，则使用的通道数
    int num_groups, // 如果设置了affine_per_channel，则使用的组数
    int64_t M, // 组数 = Bs * G
    int64_t N, // 每组中的元素数 = C * H * W / G
    double eps, // epsilon值，用于数值稳定性
    Tensor* Y) { // 输出张量指针

  AT_DISPATCH_QINT_TYPES(X.scalar_type(), "quantized_norm_nhwc_kernel_impl_cpu", [&]() {
    // 根据输入张量X的数据类型调度分派量化整数类型的操作

    using qVec = vec::Vectorized<scalar_t>;
    using fVec = vec::Vectorized<float>;

    int64_t G = num_groups; // 设置组数G
    int64_t Bs = M / G; // 每组的大小Bs
    int64_t C = num_channels; // 通道数C

    // 检查输入张量X的元素数是否符合预期
    TORCH_INTERNAL_ASSERT(X.numel() == M * N, "Unexpected num elements in X");

    // 检查gamma张量的大小是否符合预期
    TORCH_INTERNAL_ASSERT(
        !gamma.defined() ||
        (!affine_per_channel && gamma.numel() == N) ||
        (affine_per_channel && gamma.numel() == C),
        "Unexpected size of gamma");

    // 检查beta张量的大小是否符合预期
    TORCH_INTERNAL_ASSERT(
        !beta.defined() ||
        (!affine_per_channel && beta.numel() == N) ||
        (affine_per_channel && beta.numel() == C),
        "Unexpected size of beta");

    // 获取输入张量X的数据指针
    scalar_t* X_data = X.data_ptr<scalar_t>();

    // 获取gamma张量的数据指针，如果未定义则为nullptr
    const float* gamma_data = gamma.defined() ? gamma.const_data_ptr<float>() : nullptr;

    // 获取beta张量的数据指针，如果未定义则为nullptr
    const float* beta_data = beta.defined() ? beta.const_data_ptr<float>() : nullptr;

    // 获取输出张量Y的数据指针
    scalar_t* Y_data = Y->data_ptr<scalar_t>();

    // 检查gamma_data是否为nullptr的布尔值
    const bool gamma_null = gamma_data == nullptr;

    // 检查beta_data是否为nullptr的布尔值
    const bool beta_null = beta_data == nullptr;

    // 获取输入张量X的量化零点
    int64_t x_zp = X.q_zero_point();

    // 获取输入张量X的量化比例
    float x_scale = X.q_scale();

    // 创建包含输入张量X的量化零点的浮点向量
    fVec x_zp_vec((float)x_zp);

    // 创建浮点值为1.0的浮点向量
    fVec one_vec(1.0f);

    // 创建浮点值为0.0的浮点向量
    fVec zero_vec(0.0f);

    // 设置假的缩放因子为1.0的浮点值
    float x_fake_scale = 1.0f;

    // 创建包含假缩放因子的浮点向量
    fVec x_fake_scale_vec(x_fake_scale);

    // 计算x_fake_scale_vec * x_zp_vec.neg()的值并创建其结果的浮点向量
    fVec x_fake_scale_zp_neg_premul_vec = x_fake_scale_vec * x_zp_vec.neg();

    // 获取输出张量Y的量化零点
    int64_t y_zp = Y->q_zero_point();

    // 获取输出张量Y的量化比例
    float y_scale = Y->q_scale();

    // 计算输出张量Y的量化逆比例
    float y_inv_scale = 1.0f / y_scale;

    // 定义浮点向量的长度常量
    constexpr int kFloatVLen = fVec::size();

    // 计算整数向量的长度常量
    int64_t kIntVLen = kFloatVLen * qVec::float_num_vecs();

    // 计算每组中的通道数
    int64_t channels_per_group = C / G;

    // 计算N除以每组中的通道数得到HxW
    int64_t HxW = N / channels_per_group;

    // 计算每组中的整数向量数量
    int64_t kNumIntVecInHxW = channels_per_group / kIntVLen;

    // 计算每组中的非向量剩余数
    int64_t kNonVecRemInHxW = channels_per_group % kIntVLen;

    // 计算每通道的整数向量数量
    int64_t kNumIntVecOnChannel = C / kIntVLen;

    // 计算每通道的非向量剩余数
    int64_t kNonVecRemOnChannel = C % kIntVLen;

    // 创建空的缓冲区，用于存储x和x^2
    Tensor buffer = at::empty({M, 2 * channels_per_group}, X.options().dtype(at::kFloat));
    float* buffer_data = buffer.mutable_data_ptr<float>();

    // 可以在以下两种实现中并行处理：
    //
    // impl-1：在N * G上并行。只需要一个OMP会话，但每个线程的内存访问是非连续的。
    //
    // impl-2：在N * HxW上并行。每个线程的内存访问是连续的，但需要额外的临时缓冲区大小为{T, N, 2C}。
    //
    // 一般来说，当HxW足够大时，impl-2的性能更好。这个阈值是通过测试找到的。
    // 定义一个常量，表示特征图大小的阈值为512
    constexpr int64_t feature_map_threshold = 512;
    // 结束常量定义后的条件语句，判断图像的高乘以宽是否大于阈值
    } // if HxW > feature_map_threshold

  }); // AT_DISPATCH_QINT_TYPES
#else // USE_FBGEMM

#if defined(__ARM_NEON__) || defined(__aarch64__)
// 如果未定义 USE_FBGEMM，且正在使用 ARM NEON 或者 aarch64 架构

const static int PARALLEL_THRESHOLD = 1 << 20;
// 定义并初始化并行阈值常量 PARALLEL_THRESHOLD 为 2^20
// Generic template defaults to naive quantize implementation
template <typename T>
void quantize_tensor_arm(
    const float* __restrict__ in,        // 输入数据的指针，限定为只读，包含浮点数
    T* __restrict__ out,                 // 输出数据的指针，限定为只写，类型为模板参数 T
    const int64_t N,                     // 数据元素的数量
    const float scale,                   // 缩放因子，用于量化操作
    const int32_t zero_point) {          // 零点偏移量，用于量化操作
  for (const auto i : c10::irange(N)) {  // 遍历 N 个数据元素
    out[i] = at::native::quantize_val<T>(scale, zero_point, in[i]);  // 对输入数据进行量化并存储到输出数组中
  }
}

namespace quantize_tensor_arm_intrinsics {
template <typename Tx8>
C10_ALWAYS_INLINE Tx8 vqmov(int16x8_t vraw);  // 量化操作的通用模板声明

template <>
C10_ALWAYS_INLINE uint8x8_t vqmov<uint8x8_t>(int16x8_t vraw) {  // 无符号8位整数的量化模板特化
  return vqmovun_s16(vraw);  // 使用 ARM NEON 指令量化为无符号8位整数
}

template <>
C10_ALWAYS_INLINE int8x8_t vqmov<int8x8_t>(int16x8_t vraw) {  // 有符号8位整数的量化模板特化
  return vqmovn_s16(vraw);  // 使用 ARM NEON 指令量化为有符号8位整数
}

template <typename T, typename Tx8>
C10_ALWAYS_INLINE void vst1(T* out, Tx8 vout);  // 写回操作的通用模板声明

template <>
C10_ALWAYS_INLINE void vst1<uint8_t, uint8x8_t>(uint8_t* out, uint8x8_t vout) {
  vst1_u8(out, vout);  // 将 ARM NEON 寄存器中的无符号8位整数写回到内存中
}

template <>
C10_ALWAYS_INLINE void vst1<int8_t, int8x8_t>(int8_t* out, int8x8_t vout) {
  vst1_s8(out, vout);  // 将 ARM NEON 寄存器中的有符号8位整数写回到内存中
}
} // namespace quantize_tensor_arm_intrinsics

// Specialized implementation from caffe2::Int8Quantize.
// There may be slight accuracy difference between this and implementation of
// quantize_val
// TODO Update quantize_tensor_arm implementation to follow quantize_val,
// i.e. f = Round(value/scale + zero_point)
// TODO Make quantize_tensor_arm work for int32 datatype too.
template <typename scalar_t, typename underlying_t, typename underlying_x8_t>
void quantize_tensor_arm_q8(
    const float* __restrict__ in,         // 输入数据的指针，限定为只读，包含浮点数
    scalar_t* __restrict__ out,           // 输出数据的指针，限定为只写，类型为模板参数 scalar_t
    const int64_t N,                      // 数据元素的数量
    const float scale,                    // 缩放因子，用于量化操作
    const int32_t zero_point) {           // 零点偏移量，用于量化操作
  const float inv_scale = 1.0f / scale;   // 计算缩放因子的倒数
  uint32_t i = 0;                         // 循环计数器
  underlying_t* out_underlying = reinterpret_cast<underlying_t*>(out);  // 将输出指针强制转换为 underlying_t 类型的指针
  const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);  // 使用 ARM NEON 指令创建缩放因子的向量

#if defined(__ARM_NEON__)
  // magic float and magic int to take care of rounding
  // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
  // Some detail:
  // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
  // add a small number to a large number, the result rounds to the precision of
  // the least significant bit of the large number. For IEEE-754
  // single-precision number mantissa has 23 bits, and adding 2**23 would cause
  // rounding to the nearest even integer. The we cast to int and subtract the
  // same number (0x4B400000 is the integer representation of 12582912.0f) to
  // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
  // sign for negative numbers.
  const int32x4_t voffset = vdupq_n_s32(zero_point - 0x4B400000);  // 创建偏移量的向量
  const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);  // 创建魔数的浮点向量

  // 循环处理每8个数据元素，使用 ARM NEON 指令进行量化
  for (i = 0; i + 8 <= N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in);  // 加载四个浮点数到 ARM NEON 寄存器中
    in += 4;  // 更新输入数据指针，跳到下一个四元组
    const float32x4_t vin4567 = vld1q_f32(in);  // 加载接下来的四个浮点数到 ARM NEON 寄存器中
    in += 4;  // 更新输入数据指针，跳到下一个四元组
    // 计算输入向量vin0123的缩放乘积，并添加魔数后，再加上偏移量，得到vraw0123向量
    const int32x4_t vraw0123 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));

    // 计算输入向量vin4567的缩放乘积，并添加魔数后，再加上偏移量，得到vraw4567向量
    const int32x4_t vraw4567 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));

    // 将vraw0123和vraw4567向量转换为16位整数，组成vraw01234567向量
    const int16x8_t vraw01234567 =
        vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));

    // 对vraw01234567向量进行量化，并转换为指定类型underlying_x8_t
    const underlying_x8_t vout01234567 =
        quantize_tensor_arm_intrinsics::vqmov<underlying_x8_t>(vraw01234567);

    // 将量化后的结果vout01234567存储到out_underlying指向的内存位置
    quantize_tensor_arm_intrinsics::vst1<underlying_t, underlying_x8_t>(
        out_underlying, vout01234567);

    // 更新out_underlying指针，指向下一个8个元素的位置
    out_underlying += 8;
  }
  
  // 处理剩余不足8个元素的情况，通过循环逐个量化和存储
  for (; i < N; ++i) {
    // 调用Arm平台特定函数quantize_val_arm，将单个输入元素量化为underlying_t类型，并存储到out_underlying
    (*out_underlying++) =
        at::native::quantize_val_arm<underlying_t>(scale, zero_point, (*in++));
  }
#else
  // 定义一个以零点值为基础的int16x8_t类型的向量vzero_point
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  // 循环处理数据，每次处理8个元素，直到剩余不足8个元素
  for (i = 0; i + 8 <= N; i += 8) {
    // 从内存中加载4个单精度浮点数到vin0123
    const float32x4_t vin0123 = vld1q_f32(in);
    in += 4;
    // 从内存中加载下一个4个单精度浮点数到vin4567
    const float32x4_t vin4567 = vld1q_f32(in);
    in += 4;
    // 将vin0123乘以vinv_scale，并转换为四个整型数，并做近似取整
    const int32x4_t v0123_rounded = vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
    // 将vin4567乘以vinv_scale，并转换为四个整型数，并做近似取整
    const int32x4_t v4567_rounded = vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
    // 将四个整型数转换为int16x8_t类型，并加上vzero_point，得到v01234567_packed
    const int16x8_t v01234567_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded), vzero_point);
    // 调用quantize_tensor_arm_intrinsics::vqmov函数，将v01234567_packed量化为underlying_x8_t类型的数据vout01234567
    const underlying_x8_t vout01234567 =
        quantize_tensor_arm_intrinsics::vqmov<underlying_x8_t>(
            v01234567_packed);
    // 将vout01234567存储到out_underlying指针所指向的内存位置
    quantize_tensor_arm_intrinsics::vst1<underlying_t, underlying_x8_t>(
        out_underlying, vout01234567);
    // 将指针out_underlying移动到下一个8个元素的位置
    out_underlying += 8;
  }
  // 处理剩余的不足8个元素的数据
  for (; i < N; ++i) {
    // 调用at::native::quantize_val_arm函数，将in指向的单精度浮点数量化为underlying_t类型，并存储到out_underlying指向的内存位置
    (*out_underlying++) =
        at::native::quantize_val_arm<underlying_t>(scale, zero_point, (*in++));
  }
#endif
}

// 模板特化，调用quantize_tensor_arm_q8函数，将float类型的输入量化为c10::quint8类型
template <>
void quantize_tensor_arm<c10::quint8>(
    const float* __restrict__ in,
    c10::quint8* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  quantize_tensor_arm_q8<c10::quint8, uint8_t, uint8x8_t>(
      in, out, N, scale, zero_point);
}

// 模板特化，调用quantize_tensor_arm_q8函数，将float类型的输入量化为c10::qint8类型
template <>
void quantize_tensor_arm<c10::qint8>(
    const float* __restrict__ in,
    c10::qint8* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  quantize_tensor_arm_q8<c10::qint8, int8_t, int8x8_t>(
      in, out, N, scale, zero_point);
}

// 根据是否为aarch64平台定义不同的宏，用于高位扩展指令
#if defined(__aarch64__)
#define VMOVL_HIGH_U8(x) vmovl_high_u8(x)
#define VMOVL_HIGH_S8(x) vmovl_high_s8(x)
#define VMOVL_HIGH_U16(x) vmovl_high_u16(x)
#define VMOVL_HIGH_S16(x) vmovl_high_s16(x)
#else // vmovl_high intrinsic not supported
#define VMOVL_HIGH_U8(x) vmovl_u8(vget_high_u8(x))
#define VMOVL_HIGH_S8(x) vmovl_s8(vget_high_s8(x))
#define VMOVL_HIGH_U16(x) vmovl_u16(vget_high_u16(x))
#define VMOVL_HIGH_S16(x) vmovl_s16(vget_high_s16(x))
#endif

// 通用模板，默认使用简单的反量化实现
template <typename T>
void dequantize_tensor_arm(
    const T* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  // 循环处理每个元素，将输入in中的T类型量化值反量化为float，并存储到out中
  for (int i = 0; i < N; ++i) {
    out[i] = dequantize_val<T>(scale, zero_point, in[i]);
  }
}

// 模板特化，将c10::qint8类型的输入反量化为float类型
template <>
void dequantize_tensor_arm<c10::qint8>(
    const c10::qint8* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  // 将输入in强制转换为int8_t类型的指针
  const int8_t* in_underlying = reinterpret_cast<const int8_t*>(in);

  // 创建一个四个元素的单精度浮点数向量scale_fp32x4，每个元素值都为scale
  const float32x4_t scale_fp32x4 = vdupq_n_f32(scale);
  // 创建一个包含一个元素的int8x8_t类型向量zero_point_s8x8，元素值为zero_point
  const int8x8_t zero_point_s8x8 = vget_low_s8(vdupq_n_s8(static_cast<int8_t>(zero_point)));

  int i;
  // 处理16个元素的数据，每次处理16个元素
  for (i = 0; i + 16 <= N; i += 16) {
    // 从内存中加载16个int8_t类型数据到vin_s8向量中
    const int8x16_t vin_s8 = vld1q_s8(in_underlying);
    // 将vin_s8的低位和高位分别提取为int16x8，并减去zero_point_s8x8
    // 每个输入元素和zero_point都限制在有符号8位整数的范围内，因此差值可以适应有符号16位整数
    const int16x8_t minus_zp_low_s16 = vsubl_s8(vget_low_s8(vin_s8), zero_point_s8x8); // 0 ... 7
    const int16x8_t minus_zp_high_s16 = vsubl_s8(vget_high_s8(vin_s8), zero_point_s8x8); // 8 ... 15

    // 将minus_zp_low_s16的低位扩展为int32x4，得到0 ... 3范围的元素
    const int32x4_t minus_zp_low_low = vmovl_s16(vget_low_s16(minus_zp_low_s16)); // 0 ... 3
    // 将minus_zp_low_s16的高位扩展为int32x4，得到4 ... 7范围的元素
    const int32x4_t minus_zp_low_high = vmovl_high_s16(minus_zp_low_s16); // 4 ... 7
    // 将minus_zp_high_s16的低位扩展为int32x4，得到8 ... 11范围的元素
    const int32x4_t minus_zp_high_low = vmovl_s16(vget_low_s16(minus_zp_high_s16)); // 8 ... 11
    // 将minus_zp_high_s16的高位扩展为int32x4，得到12 ... 15范围的元素
    const int32x4_t minus_zp_high_high = vmovl_high_s16(minus_zp_high_s16); // 12 ... 15

    // 将四个int32x4整数向量分别乘以scale_fp32x4并转换为float32，然后存储到out指向的内存
    vst1q_f32(out,      vmulq_f32(vcvtq_f32_s32(minus_zp_low_low), scale_fp32x4));
    vst1q_f32(out + 4,  vmulq_f32(vcvtq_f32_s32(minus_zp_low_high), scale_fp32x4));
    vst1q_f32(out + 8,  vmulq_f32(vcvtq_f32_s32(minus_zp_high_low), scale_fp32x4));
    vst1q_f32(out + 12, vmulq_f32(vcvtq_f32_s32(minus_zp_high_high), scale_fp32x4));

    // 指针移动到下一个16个元素的位置
    out += 16;
    in += 16;
    in_underlying += 16;
  }

  // 对于剩余的N个元素，使用默认的dequantize_val函数进行反量化
  for (; i < N; ++i) {
    (*out++) = dequantize_val<c10::qint8>(scale, zero_point, (*in++));
  }
// 模板特化，用于将量化整数类型 c10::quint8 的张量反量化为浮点数张量
template <>
void dequantize_tensor_arm<c10::quint8>(
    const c10::quint8* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  // 将输入指针 in 转换为 const uint8_t* 类型
  const uint8_t* in_underlying = reinterpret_cast<const uint8_t*>(in);

  // 创建一个包含四个浮点数 scale 的向量，用于乘法运算
  const float32x4_t scale_fp32x4 = vdupq_n_f32(scale);
  // 创建一个包含单个无符号8位整数 zero_point 的向量，用于量化计算
  const uint8x8_t zero_point_u8x8 = vget_low_u8(vdupq_n_u8(static_cast<uint8_t>(zero_point)));

  int i;
  // 以每次处理16个元素的方式循环处理数据，直到剩余不足16个元素
  for (i = 0; i + 16 <= N; i += 16) {
    // 从输入指针 in_underlying 中加载16个无符号8位整数，存储到 vin_u8 中
    const uint8x16_t vin_u8 = vld1q_u8(in_underlying);

    // 分别从 vin_u8 中提取低8个和高8个元素，将它们转换为有符号16位整数，然后减去 zero_point_u8x8
    const int16x8_t minus_zp_low_s16 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vin_u8), zero_point_u8x8)); // 0 ... 7
    const int16x8_t minus_zp_high_s16 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vin_u8), zero_point_u8x8)); // 8 ... 15

    // 将上述的有符号16位整数转换为32位整数
    const int32x4_t minus_zp_low_low = vmovl_s16(vget_low_s16(minus_zp_low_s16)); // 0 ... 3
    const int32x4_t minus_zp_low_high = VMOVL_HIGH_S16(minus_zp_low_s16); // 4 ... 7
    const int32x4_t minus_zp_high_low = vmovl_s16(vget_low_s16(minus_zp_high_s16)); // 8 ... 11
    const int32x4_t minus_zp_high_high = VMOVL_HIGH_S16(minus_zp_high_s16); // 12 ... 15

    // 将每个32位整数乘以 scale_fp32x4 中的浮点数，并存储为32位浮点数到输出指针 out
    vst1q_f32(out,      vmulq_f32(vcvtq_f32_s32(minus_zp_low_low), scale_fp32x4));
    vst1q_f32(out + 4,  vmulq_f32(vcvtq_f32_s32(minus_zp_low_high), scale_fp32x4));
    vst1q_f32(out + 8,  vmulq_f32(vcvtq_f32_s32(minus_zp_high_low), scale_fp32x4));
    vst1q_f32(out + 12, vmulq_f32(vcvtq_f32_s32(minus_zp_high_high), scale_fp32x4));

    // 更新输出指针、输入指针和底层输入指针，以便处理下一组16个元素
    out += 16;
    in += 16;
    in_underlying += 16;
  }

  // 处理剩余不足16个元素的情况，使用默认的反量化函数 dequantize_val
  for (; i < N; ++i) {
    (*out++) = dequantize_val<c10::quint8>(scale, zero_point, (*in++));
  }
}

#endif // defined(__ARM_NEON__) || defined(__aarch64__)

// CPU 上基于每张量的仿射量化进行张量量化
void quantize_tensor_per_tensor_affine_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  // 检查张量的内存格式是否一致
  check_tensor_memory_format(rtensor, qtensor);
  // 获取输入张量的常量数据指针
  const float* rdata = rtensor.const_data_ptr<float>();
  // 获取输入张量的元素数量
  int numel = rtensor.numel();
  
  // 如果是 ARM NEON 或者 aarch64 平台
  #if defined(__ARM_NEON__) || defined(__aarch64__)
  // 使用 AT_DISPATCH_QINT_TYPES 宏分发量化整数类型
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
        // 获取量化后的张量数据指针
        scalar_t* qdata = qtensor.data_ptr<scalar_t>();
        // 定义量化范围函数对象
        auto quantize_range = [&](int64_t begin, int64_t end) {
          // 调用 ARM 平台上的量化函数 quantize_tensor_arm 进行量化操作
          quantize_tensor_arm<scalar_t>(
            rdata + begin, qdata + begin, end - begin, scale, zero_point);
        };
        // 如果元素数量大于等于并行处理的阈值 PARALLEL_THRESHOLD，则使用并行处理
        if (numel >= PARALLEL_THRESHOLD) {
          at::parallel_for(0, numel, 1, quantize_range);
        } else {
          // 否则直接调用量化范围函数
          quantize_range(0, numel);
        }
      });
  #endif // defined(__ARM_NEON__) || defined(__aarch64__)
}
#else
  // Fallback path
  // 如果不支持 ARM NEON 指令集或者 aarch64 架构，则执行以下代码块作为后备方案
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
        // 获取量化后的张量数据指针
        scalar_t* qdata = qtensor.data_ptr<scalar_t>();
        // 遍历张量中的每个元素
        for (const auto i : c10::irange(numel)) {
          // 对每个元素进行量化操作，将结果存储在量化后的张量中
          qdata[i] = quantize_val<scalar_t>(scale, zero_point, rdata[i]);
        }
      });
#endif // defined(__ARM_NEON__) || defined(__aarch64__)
}

void dequantize_tensor_per_tensor_affine_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  // 检查张量的内存格式是否匹配
  check_tensor_memory_format(qtensor, rtensor);
  // 获取张量 rtensor 的数据指针
  float* rdata = rtensor.data_ptr<float>();
  // 获取量化后张量的元素个数
  int numel = qtensor.numel();
#if defined(__ARM_NEON__) || defined(__aarch64__)
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
        // 获取量化后张量的常量数据指针
        const scalar_t* qdata = qtensor.const_data_ptr<scalar_t>();
        // 定义范围解量化函数
        auto dequantize_range = [&](int64_t begin, int64_t end) {
          // 调用 ARM 平台上的解量化函数进行解量化操作
          dequantize_tensor_arm<scalar_t>(
            qdata + begin, rdata + begin, end - begin, scale, zero_point);
        };
        // 如果元素个数大于等于并行阈值，则使用并行方式进行解量化
        if (numel >= PARALLEL_THRESHOLD) {
          at::parallel_for(0, numel, 1, dequantize_range);
        } else {
          // 否则使用串行方式进行解量化
          dequantize_range(0, numel);
        }
      });
#else
  // Fallback path
  // 如果不支持 ARM NEON 指令集或者 aarch64 架构，则执行以下代码块作为后备方案
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
        // 获取量化后张量的常量数据指针
        const scalar_t* qdata = qtensor.const_data_ptr<scalar_t>();
        // 遍历量化后张量中的每个元素
        for (const auto i : c10::irange(numel)) {
          // 对每个元素进行解量化操作，将结果存储在解量化后的张量中
          rdata[i] = dequantize_val<scalar_t>(scale, zero_point, qdata[i]);
        }
      });
#endif // defined(__ARM_NEON__) || defined(__aarch64__)
}
#endif // USE_FBGEMM

// TODO: add fbgemm for per channel
// Generic template defaults to naive quantize implementation
// 通用模板默认使用简单的量化实现方式
template <typename T>
void quantize_tensor_per_channel_impl(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  // TODO: channels last kernel can be made faster.
  // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
  // For channels_last/3d however axis == 0 or 1.
  // Since current implementation on channels_last format does not
  // cover per channel quant with arbitrary axis value, it is better
  // to check and fail.
  // TODO: channels last 内核可以优化。
  // 对于连续张量，例如 NCHW，可以使用任意轴。
  // 但对于 channels_last/3d，轴通常为 0 或 1。
  // 由于当前的 channels_last 格式实现不涵盖使用任意轴值的通道量化，最好进行检查并失败。
  // 计算批次数
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  // 计算每个通道的元素个数
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  // 获取通道数
  int64_t channels = rtensor.size(axis);
  // 获取量化比例和零点的数据指针
  auto scales_data = scales.data_ptr<double>();
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  // 获取输入张量的数据指针
  const float* in = rtensor.const_data_ptr<float>();
  // 获取输出张量的数据指针
  auto out = qtensor.data_ptr<T>();
  // 如果 axis 为 1 并且张量为 channels_last 连续格式或 channels_last3d
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // This code handles per channel quant when axis = 1 and
    // channels_last contig.
    // 此代码处理 axis = 1 和 channels_last 连续格式时的通道量化。
    // 如果 axis = 0 并且 channels_last 是连续的，则处理 channels_first (NCHW) 的实现。
    if (axis == 0 && channels_last_contig) {
        // 遍历每个 batch
        for (const auto b : c10::irange(batches)) {
            // 遍历每个通道内的元素
            for (const auto e : c10::irange(elements_per_channel)) {
                // 遍历每个通道
                for (const auto c : c10::irange(channels)) {
                    // 计算输入张量中的索引 i
                    auto i = b * channels * elements_per_channel + e * channels + c;
                    // 对输入张量中的值进行量化，使用对应的缩放因子、零点和输入值
                    out[i] = at::native::quantize_val<T>(
                        scales_data[c], zero_points_data[c], in[i]);
                }
            }
        }
    } else {
        // 遍历每个 batch
        for (const auto b : c10::irange(batches)) {
            // 遍历每个通道
            for (const auto c : c10::irange(channels)) {
                // 遍历每个通道内的元素
                for (const auto e : c10::irange(elements_per_channel)) {
                    // 计算输入张量中的索引 i
                    auto i = b * channels * elements_per_channel +
                        c * elements_per_channel + e;
                    // 对输入张量中的值进行量化，使用对应的缩放因子、零点和输入值
                    out[i] = at::native::quantize_val<T>(
                        scales_data[c], zero_points_data[c], in[i]);
                }
            }
        }
    }
}

// 如果定义了 __ARM_NEON__ 或者 __aarch64__，使用专门的实现来量化张量到 quint8 类型
// 从 caffe2::Int8Quantize 获得。这里可能会有一些精度差异，与 quantize_val 的实现相比。
// TODO 更新 quantize_tensor_per_channel_impl 的实现，以便遵循 quantize_val 的方式，即 f = Round(value/scale + zero_point)
// TODO 使 quantize_tensor_per_channel_impl 能够适用于其他数据类型（比如 int8, int32）。
template <>
void quantize_tensor_per_channel_impl<c10::quint8>(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  // 计算批次数，即轴之前的所有维度乘积
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  // 每个通道的元素数，即轴后面的所有维度乘积
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  // 通道数，即在指定轴上的大小
  int64_t channels = rtensor.size(axis);
  // 获取 scales 和 zero_points 的数据指针
  auto scales_data = scales.data_ptr<double>();
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  // 输入张量的常量数据指针，假设为 float 类型
  const float* in = rtensor.const_data_ptr<float>();
  // 输出张量的数据指针，假设为 quint8 类型
  auto out = (uint8_t*)qtensor.data_ptr<c10::quint8>();
#if defined(__ARM_NEON__)
  // 用于处理舍入的神奇浮点数和神奇整数
  // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
  // 一些细节：
  // 12582912.0f 是 2**23 + 2**22。该技巧基于以下事实：当您向大数添加小数时，结果会舍入到大数最不重要位的精度。
  // 对于 IEEE-754 单精度数，尾数有 23 位，添加 2**23 会导致四舍五入到最接近的偶数整数。
  // 然后转换为 int 并减去相同的数字（0x4B400000 是 12582912.0f 的整数表示），以获取仅保留尾数。
  const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
  // 将每个通道的反比例尺度（double）复制到 float 数组中
  // 将 zero_points 与神奇整数（int64_t）复制到 int32_t 数组中
  std::vector<float> inv_scales(channels);
  std::vector<int32_t> zero_points_int32t(channels);
  for (const auto i : c10::irange(channels)) {
    inv_scales[i] = 1.0f / (float)scales_data[i];
    zero_points_int32t[i] = (int32_t)(uint32_t)zero_points_data[i] - 0x4B400000;
  }
  // 如果 axis == 1 并且张量是通道最后连续的（ChannelsLast 或 ChannelsLast3d 格式）
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // 当 axis = 1 且通道在最后连续时，处理每个通道的量化
    // 如果 axis = 0 且通道在最后连续，适用于通道优先（NCHW）的实现方式。
    // 对于每个批次中的每个元素通道，进行量化操作
    for (C10_UNUSED const auto b : c10::irange(batches)) {
      // 遍历每个元素通道中的元素
      for (C10_UNUSED const auto e : c10::irange(elements_per_channel)) {
        // 初始化通道计数器为零
        uint32_t c = 0;
        // 当通道数 c 加上 8 小于总通道数时，执行以下循环
        while (c + 8 < channels) {
          // 加载当前通道的零点偏移量为 int32x4_t 类型
          const int32x4_t voffset0123 = vld1q_s32(&zero_points_int32t[c]);
          // 加载当前通道的反比例尺度为 float32x4_t 类型
          const float32x4_t vinv_scale0123 = vld1q_f32(&inv_scales[c]);
          // 增加通道计数器以处理下一个四个通道
          c += 4;
          // 加载接下来四个通道的零点偏移量为 int32x4_t 类型
          const int32x4_t voffset4567 = vld1q_s32(&zero_points_int32t[c]);
          // 加载接下来四个通道的反比例尺度为 float32x4_t 类型
          const float32x4_t vinv_scale4567 = vld1q_f32(&inv_scales[c]);
          // 增加通道计数器以处理下一个四个通道
          c += 4;
          // 加载输入指针指向的四个浮点数为 float32x4_t 类型
          const float32x4_t vin0123 = vld1q_f32(in);
          // 增加输入指针以处理下一个四个浮点数
          in += 4;
          // 加载输入指针指向的接下来四个浮点数为 float32x4_t 类型
          const float32x4_t vin4567 = vld1q_f32(in);
          // 增加输入指针以处理下一个四个浮点数
          in += 4;
          // 对 vin0123 进行量化操作并加上零点偏移量，存储为 int32x4_t 类型
          const int32x4_t vraw0123 = vaddq_s32(
              voffset0123,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale0123))));
          // 对 vin4567 进行量化操作并加上零点偏移量，存储为 int32x4_t 类型
          const int32x4_t vraw4567 = vaddq_s32(
              voffset4567,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale4567))));
          // 合并 vraw0123 和 vraw4567，并转换为 int16x8_t 类型
          const int16x8_t vraw01234567 =
              vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
          // 将 vraw01234567 转换为 uint8x8_t 类型并存储到输出指针指向的位置
          const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
          // 增加输出指针以处理下一个八个字节
          vst1_u8(out, vout01234567);
          // 增加输出指针以处理下一个八个字节
          out += 8;
        }
        // 处理剩余不足八个通道的情况
        for (; c < channels; ++c) {
          // 调用 ARM 平台的量化函数 quantize_val_arm 进行单个值的量化操作
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  } else {
    // 对于每个批次中的每个通道，进行量化操作
    for (C10_UNUSED const auto b : c10::irange(batches)) {
      // 遍历每个通道
      for (const auto c : c10::irange(channels)) {
        // 初始化元素计数器为零
        uint32_t e = 0;
        // 加载当前通道的零点偏移量为 int32x4_t 类型
        const int32x4_t voffset = vdupq_n_s32(zero_points_int32t[c]);
        // 加载当前通道的反比例尺度为 float32x4_t 类型
        const float32x4_t vinv_scale = vdupq_n_f32(inv_scales[c]);
        // 当元素计数器 e 小于元素通道总数减 8 时，执行以下循环
        for (; e + 8 < elements_per_channel; e += 8) {
          // 加载输入指针指向的四个浮点数为 float32x4_t 类型
          const float32x4_t vin0123 = vld1q_f32(in);
          // 增加输入指针以处理下一个四个浮点数
          in += 4;
          // 加载输入指针指向的接下来四个浮点数为 float32x4_t 类型
          const float32x4_t vin4567 = vld1q_f32(in);
          // 增加输入指针以处理下一个四个浮点数
          in += 4;
          // 对 vin0123 进行量化操作并加上零点偏移量，存储为 int32x4_t 类型
          const int32x4_t vraw0123 = vaddq_s32(
              voffset,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
          // 对 vin4567 进行量化操作并加上零点偏移量，存储为 int32x4_t 类型
          const int32x4_t vraw4567 = vaddq_s32(
              voffset,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
          // 合并 vraw0123 和 vraw4567，并转换为 int16x8_t 类型
          const int16x8_t vraw01234567 =
              vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
          // 将 vraw01234567 转换为 uint8x8_t 类型并存储到输出指针指向的位置
          const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
          // 增加输出指针以处理下一个八个字节
          vst1_u8(out, vout01234567);
          // 增加输出指针以处理下一个八个字节
          out += 8;
        }
        // 处理剩余不足八个元素的情况
        for (; e < elements_per_channel; ++e) {
          // 调用 ARM 平台的量化函数 quantize_val_arm 进行单个值的量化操作
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  }
#else // defined(__ARM_NEON__)
  // 将缩放因子（double）复制到 float 数组
  // 将零点（int64_t）复制到 int16_t 数组
  std::vector<float> inv_scales(channels); // 创建存储缩放因子的 float 数组
  std::vector<int16_t> zero_points_int16t(channels); // 创建存储零点的 int16_t 数组
  for (const auto i : c10::irange(channels)) {
    inv_scales[i] = 1.0f / (float)scales_data[i]; // 计算每个通道的倒数缩放因子
    zero_points_int16t[i] = (int16_t)(uint16_t)zero_points_data[i]; // 将零点数据转换为 int16_t 类型
  }
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // 处理轴为 1 且通道为最后维度连续的情况
    // 如果轴为 0 且通道为最后维度连续，则使用通道为第一维度（NCHW）的实现
    for (const auto b C10_UNUSED : c10::irange(batches)) {
      for (const auto e C10_UNUSED : c10::irange(elements_per_channel)) {
        uint32_t c = 0;
        while (c + 8 < channels) {
          const int16x8_t vzero_point = vld1q_s16(&zero_points_int16t[c]); // 加载 int16_t 类型的零点数据
          const float32x4_t vinv_scale0123 = vld1q_f32(&inv_scales[c]); // 加载 float 类型的缩放因子数据
          c += 4;
          const float32x4_t vinv_scale4567 = vld1q_f32(&inv_scales[c]); // 加载接下来的四个缩放因子数据
          c += 4;
          const float32x4_t vin0123 = vld1q_f32(in); // 加载输入数据的前四个元素
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in); // 加载输入数据的后四个元素
          in += 4;
          const int32x4_t v0123_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale0123)); // 对 vin0123 应用缩放因子并进行四舍五入
          const int32x4_t v4567_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale4567)); // 对 vin4567 应用缩放因子并进行四舍五入
          const int16x8_t v01234567_packed = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded), // 将四舍五入后的结果与零点相加
              vzero_point);
          const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed); // 将结果转换为 uint8_t 类型
          vst1_u8(out, vout01234567); // 存储转换后的结果
          out += 8; // 更新输出指针
        }
        for (; c < channels; ++c) {
          (*out++) = at::native::quantize_val_arm<uint8_t>( // 调用 ARM 平台的量化函数
              scales_data[c], zero_points_data[c], (*in++)); // 使用通道的缩放因子和零点量化输入数据
        }
      }
    }
  } else {
    // 遍历批次数和通道数，分别为 b 和 c
    for (const auto b C10_UNUSED : c10::irange(batches)) {
      for (const auto c C10_UNUSED : c10::irange(channels)) {
        // 初始化元素索引 e 为 0
        uint32_t e = 0;
        // 从预设的 zero_points_int16t 数组中复制出 vzero_point，并赋给 vzero_point
        const int16x8_t vzero_point = vdupq_n_s16(zero_points_int16t[c]);
        // 从预设的 inv_scales 数组中复制出 vinv_scale，并赋给 vinv_scale
        const float32x4_t vinv_scale = vdupq_n_f32(inv_scales[c]);
        // 迭代处理每个通道的元素，每次处理 8 个元素
        for (; e + 8 < elements_per_channel; e += 8) {
          // 加载 4 个 float 型数据到 vin0123，并移动输入指针 in
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          // 加载接下来的 4 个 float 型数据到 vin4567，并移动输入指针 in
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          // 将 vin0123 中的数据乘以 vinv_scale 并转换为整数，得到 v0123_rounded
          const int32x4_t v0123_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
          // 将 vin4567 中的数据乘以 vinv_scale 并转换为整数，得到 v4567_rounded
          const int32x4_t v4567_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
          // 将 v0123_rounded 和 v4567_rounded 合并成一个 int16x8_t 型数据，并加上 vzero_point
          const int16x8_t v01234567_packed = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
              vzero_point);
          // 将 v01234567_packed 中的数据转换为 uint8x8_t 型数据，得到 vout01234567
          const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed);
          // 存储 vout01234567 中的数据到输出指针 out，并移动输出指针 out
          vst1_u8(out, vout01234567);
          out += 8;
        }
        // 处理剩余的少于 8 个元素的情况
        for (; e < elements_per_channel; ++e) {
          // 使用 ARM 架构的 quantize_val_arm 函数对单个元素进行量化，并存储到输出指针 out
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  }
#endif // defined(__ARM_NEON__)
}
#endif // defined(__ARM_NEON__) || defined(__aarch64__)

void quantize_tensor_per_channel_affine_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  // 检查是否是连续存储或者轴小于等于1
  TORCH_CHECK(
      rtensor.is_contiguous() || (axis <= 1),
      "If tensor is channels_last contig then per channel quantization "
      "is supported only for axis = 0 or 1.");
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_channel_affine_cpu", [&]() {
        // 检查张量的内存格式是否符合预期
        check_tensor_memory_format(rtensor, qtensor);
        // 调用模板函数 quantize_tensor_per_channel_impl，对张量进行逐通道量化
        quantize_tensor_per_channel_impl<scalar_t>(
            rtensor, qtensor, scales, zero_points, axis);
      });
}

template<typename T, typename N, typename Q>
void dequantize_per_channel_affine_kernel(
      const Tensor& qtensor,
      Tensor& rtensor,
      const Tensor& scales,
      const Tensor& zero_points,
      int64_t axis,
      int bit_width=8) {

  // 对于连续存储的张量（例如NCHW），可以使用任意的轴值。
  // 对于 channels_last/3d 格式，axis 只能为 0 或 1。
  // 因为当前 channels_last 实现不支持任意轴值的逐通道量化，因此最好检查并报错。
  TORCH_CHECK(rtensor.is_contiguous() || (axis <=1),
      "If tensor is channels_last contig then per channel quantization "
      "is supported only for axis = 0 or 1.");
  // 计算批次数
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  // 计算每个通道的元素个数
  int64_t elements_per_channel =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      size_from_dim_(axis + 1, rtensor.sizes());
  // 获取通道数
  int64_t channel = rtensor.size(axis);
  // 获取张量 scales 和 zero_points 的数据指针
  auto scales_data = scales.data_ptr<T>();
  auto zero_points_data = zero_points.data_ptr<N>();
  // 检查张量的内存格式是否符合预期
  check_tensor_memory_format(qtensor, rtensor);
  // 获取输入张量 qtensor 的常量数据指针和输出张量 rtensor 的数据指针
  const auto* qd = qtensor.const_data_ptr<Q>();
  float* rd = rtensor.data_ptr<float>();
  // 计算每字节的元素个数（每个字节可以存储的元素个数）
  const auto elem_per_byte = 8 / bit_width;
  // 如果 axis 等于 1 并且 rtensor 的内存格式为 ChannelsLast 或 ChannelsLast3d
  if (axis == 1 && (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
      rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // 循环遍历批次
    for (const auto b : c10::irange(batches)) {
      // 循环遍历每个通道内的元素
      for (const auto e : c10::irange(elements_per_channel)) {
        // 循环遍历每个通道
        for (const auto c : c10::irange(channel)) {
          // 计算输入和输出张量中元素的索引
          auto i = b * channel * elements_per_channel + e * channel + c;
          // 将 qint8 值转换为 float，以确保减法子表达式返回 float 类型
          auto qvalue = qd[i / elem_per_byte].val_;
          // 如果 bit_width 小于 8，则对 qvalue 进行右移和掩码操作
          if (bit_width < 8) {
            qvalue >>= (i % elem_per_byte) * bit_width;
            qvalue &= (1 << bit_width) - 1;
          }
          // 计算量化值的反量化结果，并存储到输出张量中
          rd[i] = (static_cast<float>(qvalue) - zero_points_data[c]) * scales_data[c];
        }
      }
    }
  } else {
    // 遍历批次（batch）
    for (const auto b : c10::irange(batches)) {
      // 遍历通道（channel）
      for (const auto c : c10::irange(channel)) {
        // 遍历每个通道内的元素（element）
        for (const auto e : c10::irange(elements_per_channel)) {
          // 计算当前元素在一维数组中的索引
          auto i = b * channel * elements_per_channel +
              c * elements_per_channel + e;
          
          // 获取 qint8 类型的值，并转换为 float 类型以确保减法表达式返回 float 类型结果
          auto qvalue = qd[i / elem_per_byte].val_;

          // 如果比特宽度小于 8，则进行右移和位与操作，以获取正确的 qint8 值
          if (bit_width < 8) {
            qvalue >>= (i % elem_per_byte) * bit_width;
            qvalue &= (1 << bit_width) - 1;
          }

          // 将 qint8 值转换为 float 类型，并应用零点偏移和缩放因子进行处理，并存入结果数组 rd
          rd[i] = (static_cast<float>(qvalue) - zero_points_data[c]) * scales_data[c];
        }
      }
    }
}

// 对每个通道进行仿射反量化，CPU版本
void dequantize_tensor_per_channel_affine_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_channel_affine_cpu", [&]() {
        // 调用模板函数进行每通道仿射反量化操作
        dequantize_per_channel_affine_kernel<double, int64_t, scalar_t>(qtensor, rtensor, scales, zero_points, axis);
      });
}

// 用于浮点数比例因子和零点的张量量化存根
void quantize_tensor_per_channel_float_qparams_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
}

void dequantize_tensor_per_channel_float_qparams_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  // NOLINTNEXTLINE(clang-diagnostic-unused-variable)
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_channel_float_qparams_cpu", [&]() {
        // 调用模板函数进行每通道浮点数参数仿射反量化操作
        dequantize_per_channel_affine_kernel<float, float, scalar_t>(qtensor, rtensor, scales, zero_points, axis, bit_width);
      });
}

void quantize_tensor_per_tensor_affine_sub_byte_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    float scale,
    float zero_point) {
  // TODO 使用 fbgemm 内核打包数值
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
    qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
      // 检查张量的内存格式
      check_tensor_memory_format(rtensor, qtensor);
      const float* const rdata = rtensor.const_data_ptr<float>();
      auto qdata = reinterpret_cast<underlying_t*>(qtensor.data_ptr<scalar_t>());
      auto numel = rtensor.numel();
      const auto elem_per_byte = CHAR_BIT / bit_width;
      for (const auto i : c10::irange(numel)) {
        float inv_scale = scale == 0 ? 1.0f : 1.0f / scale;
        int64_t qvalue = lrintf(std::nearbyint(rdata[i] * inv_scale) + zero_point);
        qvalue = std::max(quant_min, std::min(qvalue, quant_max));

        // 将子字节值打包并对齐到字节
        // 例如，对于4位索引，索引0打包在低4位，索引1打包在高4位
        // NOLINTNEXTLINE(clang-analyzer-core.DivideZero)
        if (i % elem_per_byte == 0) {
          qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
        } else {
          qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
        }
      } // for numel
    });
}

void dequantize_tensor_per_tensor_affine_sub_byte_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    float scale,
    float zero_point) {
  // TODO 使用 fbgemm 内核打包数值
  // NOLINTNEXTLINE(clang-diagnostic-unused-variable)
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
        // 调用模板函数进行每张量仿射子字节反量化操作
        dequantize_per_channel_affine_kernel<float, float, scalar_t>(qtensor, rtensor, scales, zero_points, axis, bit_width);
      });
}
    qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
      // 检查张量的内存格式是否匹配
      check_tensor_memory_format(rtensor, qtensor);
      // 获取结果张量的数据指针，并转换为 float 类型
      auto rdata = rtensor.data_ptr<float>();
      // 获取量化张量的数据指针，并转换为 underlying_t 类型
      const underlying_t* qdata = reinterpret_cast<const underlying_t*>(qtensor.const_data_ptr<scalar_t>());
      // 获取结果张量的元素总数
      auto numel = rtensor.numel();
      // 计算每字节的元素个数
      const auto elem_per_byte = CHAR_BIT / bit_width;

      // 遍历结果张量的所有元素
      for (const auto i : c10::irange(numel)) {
        // NOLINTNEXTLINE(clang-analyzer-core.DivideZero)
        // 计算当前元素对应的量化值
        underlying_t qvalue = qdata[i / elem_per_byte];
        // 右移以获取正确的量化值
        qvalue >>= (i % elem_per_byte) * bit_width;
        // 掩码操作以确保取得正确的量化值
        qvalue &= (1 << bit_width) - 1;
        // 将量化值反量化为 float 类型，并应用量化零点和缩放因子
        rdata[i] = (static_cast<float>(qvalue) - zero_point) * scale;
      }
  });
}

// 这个函数期望 quantized_val 输入已经被量化
template <typename scalar_t>
void cpu_masked_fill_kernel_quantized_cpu(TensorIterator& iter, scalar_t quantized_val) {
  // 定义 lambda 函数 loop，用于处理迭代器中的每个元素
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    // 获取目标张量的指针和掩码张量的指针
    char* dst = data[0];
    char* mask = data[1];
    // 遍历迭代器中的每个元素
    for (const auto i : c10::irange(n)) {
      // 解析出当前位置的掩码值
      bool mask_value = *reinterpret_cast<bool*>(mask + strides[1] * i);

      // 如果掩码为真，则将 quantized_val 写入目标张量中对应位置
      if (mask_value) {
        *(scalar_t*)(dst + strides[0] * i) = quantized_val;
      }
    }
  };
  // 使用迭代器对象调用 loop 函数，对目标张量进行操作
  iter.for_each(loop);
}

// 在 CPU 上执行量化版本的 masked_fill 操作
void masked_fill_kernel_quantized_cpu(TensorIterator& iter, const Scalar& value, double scale, int zero_point) {
  // 根据张量类型执行相应的量化操作
  AT_DISPATCH_QINT_TYPES(iter.dtype(), "masked_fill", [&] {
    // 将 Scalar 值转换为 float 类型
    float float_val = value.to<float>();
    // 根据指定的 scale 和 zero_point 进行量化
    auto quantized_val = quantize_val<scalar_t>(scale, zero_point, float_val);
    // 获取掩码张量的数据类型
    auto mask_dtype = iter.input_dtype(0);
    // 检查掩码张量的数据类型是否为布尔类型
    TORCH_CHECK(mask_dtype == ScalarType::Bool, "masked_fill only supports boolean masks, "
      "but got mask with dtype ", mask_dtype);
    // 调用量化版本的 masked_fill 内核函数
    cpu_masked_fill_kernel_quantized_cpu<scalar_t>(iter, quantized_val);
  });
}

// 当前，我们不支持对量化张量使用 accumulate=True。我们在 _index_put_impl_quantized_cpu_ 中抛出异常。
void index_put_kernel_quantized_cpu(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate, double scale, int zero_point) {
  // 注意：仅在 accumulate 为 true 时支持重复索引。
  AT_DISPATCH_QINT_TYPES(iter.dtype(), "index_put", [&] {
    // 查看注释 [Enabling Deterministic Operations]
    // 如果启用了确定性算法，由于并行的 cpu_index_kernel 结合积累操作是非确定性的，
    // 我们必须在这种情况下启用串行执行。
    const bool is_deterministic = at::globalContext().deterministicAlgorithms();
    // 调用 cpu_index_kernel 处理量化类型的张量索引更新
    at::native::cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [scale, zero_point](char* dst, char* src, int64_t offset) {
      // 将浮点数值 src 量化为 scalar_t 类型，并写入到 dst 中
      *(scalar_t*)(dst + offset) = quantize_val<scalar_t>(scale, zero_point, *(float*)src);
    }, /*serial_execution=*/is_deterministic);
  });
}

} // 匿名命名空间结束

// 一些量化测试在 Windows 上使用 AVX512 可能会出现问题。如果使用了 --continue-through-error，
// 只有一个测试会失败。但如果跳过失败的测试，另一个测试会失败。
// 如果第二个测试也被跳过，第三个测试会失败。
// 因此，在修复 AVX512 下的量化支持之前，我们会使用 AVX2 内核。参考：GH 56992。
#if defined(_WIN32)
// 注册函数指针以支持特定的量化操作
REGISTER_DISPATCH(dequantize_tensor_per_channel_affine_stub,
                  &dequantize_tensor_per_channel_affine_cpu);
REGISTER_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub,
                  &dequantize_tensor_per_channel_float_qparams_cpu);
REGISTER_DISPATCH(fake_quant_per_channel_cachemask_stub,
                  &fake_quant_per_channel_cachemask_cpu);
REGISTER_DISPATCH(qavg_pool2d_nhwc_stub, &qavg_pool2d_nhwc_kernel);
// 如果条件不满足 AVX512 和 _WIN32，则注册以下函数到 AVX512 指令集
ALSO_REGISTER_AVX512_DISPATCH(dequantize_tensor_per_channel_affine_stub,
                  &dequantize_tensor_per_channel_affine_cpu);
ALSO_REGISTER_AVX512_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub,
                  &dequantize_tensor_per_channel_float_qparams_cpu);
ALSO_REGISTER_AVX512_DISPATCH(fake_quant_per_channel_cachemask_stub,
                  &fake_quant_per_channel_cachemask_cpu);
ALSO_REGISTER_AVX512_DISPATCH(qavg_pool2d_nhwc_stub, &qavg_pool2d_nhwc_kernel);
ALSO_REGISTER_AVX512_DISPATCH(qavg_pool3d_nhwc_stub, &qavg_pool3d_nhwc_kernel);

// 如果条件满足 AVX512 和 _WIN32，则不注册以上函数

// 以下这些内核函数由于在 AVX512 下表现不佳，所以被注册到 AVX2 指令集
REGISTER_DISPATCH(dequantize_tensor_per_tensor_affine_stub,
                  &dequantize_tensor_per_tensor_affine_cpu);
REGISTER_DISPATCH(fake_quant_grad_learnable_tensor_stub,
                  &fake_quantize_learnable_tensor_grad_kernel_cpu);
REGISTER_DISPATCH(fake_quant_tensor_cachemask_stub,
                  &fake_quantize_tensor_cachemask_kernel);
REGISTER_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_stub,
                  &fake_quantize_tensor_cachemask_tensor_qparams_kernel);
REGISTER_DISPATCH(qadaptive_avg_pool2d_nhwc_stub,
                  &qadaptive_avg_pool2d_nhwc_kernel);
REGISTER_DISPATCH(qadaptive_avg_pool3d_ndhwc_stub,
                  &qadaptive_avg_pool3d_ndhwc_kernel);
REGISTER_DISPATCH(qadd_relu_stub, &qadd_kernel<true>);
REGISTER_DISPATCH(qadd_scalar_relu_stub, &qadd_scalar_kernel<true>);
REGISTER_DISPATCH(qadd_scalar_stub, &qadd_scalar_kernel<false>);
REGISTER_DISPATCH(qadd_stub, &qadd_kernel<false>);
REGISTER_DISPATCH(qbatch_norm_relu_stub, &q_batch_norm_kernel<true>);
REGISTER_DISPATCH(qbatch_norm_stub, &q_batch_norm_kernel<false>);
REGISTER_DISPATCH(qcat_nhwc_stub, &qcat_nhwc_kernel<false>);
REGISTER_DISPATCH(qcat_relu_nhwc_stub, &qcat_nhwc_kernel<true>);
REGISTER_DISPATCH(qclamp_stub, &qclamp_kernel);
REGISTER_DISPATCH(qclamp_min_stub, &qclamp_min_kernel);
REGISTER_DISPATCH(qclamp_max_stub, &qclamp_max_kernel);
REGISTER_DISPATCH(qelu_stub, &qelu_kernel);
REGISTER_DISPATCH(qhardsigmoid_stub, &qhardsigmoid_kernel);
REGISTER_DISPATCH(qhardswish_stub, &qhardswish_kernel);
REGISTER_DISPATCH(qmaxpool_2d_nhwc_stub, &qmaxpool_2d_nhwc_kernel);
REGISTER_DISPATCH(qmaxpool_3d_nthwc_stub, &qmaxpool_3d_nthwc_kernel);
REGISTER_DISPATCH(qmul_relu_stub, &qmul_kernel<true>);
REGISTER_DISPATCH(qmul_stub, &qmul_kernel<false>);
REGISTER_DISPATCH(qrelu_leaky_stub, &leaky_qrelu_out_kernel);
REGISTER_DISPATCH(qrelu_stub, &qrelu_kernel);
REGISTER_DISPATCH(qprelu_stub, &qprelu_out_kernel);
REGISTER_DISPATCH(qgelu_stub, &qgelu_kernel);
REGISTER_DISPATCH(qsigmoid_stub, &qsigmoid_kernel);
REGISTER_DISPATCH(qtanh_stub, &qtanh_kernel);
# 注册 qthreshold_stub 到 qthreshold_kernel 的调度器
REGISTER_DISPATCH(qthreshold_stub, &qthreshold_kernel);

# 注册 qtopk_stub 到 qtopk_kernel 的调度器
REGISTER_DISPATCH(qtopk_stub, &qtopk_kernel);

# 注册 fake_quant_grad_learnable_channel_stub 到 fake_quantize_learnable_channel_grad_kernel_cpu 的调度器
REGISTER_DISPATCH(fake_quant_grad_learnable_channel_stub,
                  &fake_quantize_learnable_channel_grad_kernel_cpu);

# 注册 quantize_tensor_per_tensor_affine_stub 到 quantize_tensor_per_tensor_affine_cpu 的调度器
REGISTER_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &quantize_tensor_per_tensor_affine_cpu);

# 注册 quantize_tensor_per_channel_affine_stub 到 quantize_tensor_per_channel_affine_cpu 的调度器
REGISTER_DISPATCH(
    quantize_tensor_per_channel_affine_stub,
    &quantize_tensor_per_channel_affine_cpu);

# 注册 quantize_tensor_per_channel_float_qparams_stub 到 quantize_tensor_per_channel_float_qparams_cpu 的调度器
REGISTER_DISPATCH(
    quantize_tensor_per_channel_float_qparams_stub,
    &quantize_tensor_per_channel_float_qparams_cpu);

# 注册 quantized_normalize_stub 到 quantized_normalize_kernel 的调度器
REGISTER_DISPATCH(quantized_normalize_stub, &quantized_normalize_kernel);

# 注册 quantized_groupnorm_nhwc_stub 到 quantized_groupnorm_nhwc_kernel 的调度器
REGISTER_DISPATCH(quantized_groupnorm_nhwc_stub, &quantized_groupnorm_nhwc_kernel);

# 注册 qupsample_bilinear2d_nhwc_stub 到 qupsample_bilinear2d_nhwc_kernel 的调度器
REGISTER_DISPATCH(qupsample_bilinear2d_nhwc_stub,
                  &qupsample_bilinear2d_nhwc_kernel);

# 注册 quantize_tensor_per_tensor_affine_sub_byte_stub 到 quantize_tensor_per_tensor_affine_sub_byte_cpu 的调度器
REGISTER_DISPATCH(
    quantize_tensor_per_tensor_affine_sub_byte_stub,
    &quantize_tensor_per_tensor_affine_sub_byte_cpu);

# 注册 dequantize_tensor_per_tensor_affine_sub_byte_stub 到 dequantize_tensor_per_tensor_affine_sub_byte_cpu 的调度器
REGISTER_DISPATCH(
    dequantize_tensor_per_tensor_affine_sub_byte_stub,
    &dequantize_tensor_per_tensor_affine_sub_byte_cpu);

# 注册 masked_fill_kernel_quantized_stub 到 masked_fill_kernel_quantized_cpu 的调度器
REGISTER_DISPATCH(
    masked_fill_kernel_quantized_stub,
    &masked_fill_kernel_quantized_cpu);

# 注册 index_put_kernel_quantized_stub 到 index_put_kernel_quantized_cpu 的调度器
REGISTER_DISPATCH(
    index_put_kernel_quantized_stub,
    &index_put_kernel_quantized_cpu);

# 注册 qmean_inner_dim_stub 到 qmean_inner_dim_kernel 的调度器
REGISTER_DISPATCH(qmean_inner_dim_stub, &qmean_inner_dim_kernel);

# 注册 qstd_inner_dim_stub 到 qstd_inner_dim_kernel 的调度器
REGISTER_DISPATCH(qstd_inner_dim_stub, &qstd_inner_dim_kernel);

# 关闭命名空间 native
} // namespace native

# 关闭命名空间 at
} // namespace at
```