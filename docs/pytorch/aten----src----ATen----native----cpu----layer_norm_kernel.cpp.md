# `.\pytorch\aten\src\ATen\native\cpu\layer_norm_kernel.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 layer_norm.h 文件，包含层归一化操作的相关定义和实现
#include <ATen/native/layer_norm.h>

// 引入数学函数库，例如 std::sqrt
#include <cmath>
// 引入元组支持
#include <tuple>

// 引入 Tensor 类定义
#include <ATen/core/Tensor.h>
// 引入分发机制定义
#include <ATen/Dispatch.h>
// 引入数学操作类型定义
#include <ATen/OpMathType.h>
// 引入向量化函数相关的定义
#include <ATen/cpu/vec/functional.h>
// 引入向量化支持
#include <ATen/cpu/vec/vec.h>
// 引入计算统计特征相关的实用函数
#include <ATen/native/cpu/moments_utils.h>
// 引入混合数据类型支持
#include <ATen/native/cpu/mixed_data_type.h>
// 引入范围迭代器支持
#include <c10/util/irange.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则引入 ATen 函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 否则，引入空函数头文件
#else
#include <ATen/ops/empty.h>
#endif

// 定义 at::native 命名空间
namespace at::native {

// 匿名命名空间，内部实现函数
namespace {

// 模板函数，用于实现非降维浮点数类型的层归一化操作
template <typename T,
          typename std::enable_if_t<!is_reduced_floating_point_v<T>, int> = 0>
void LayerNormKernelImplInternal(
    const Tensor& X,        // 输入张量 X
    const Tensor& gamma,    // 缩放参数 gamma
    const Tensor& beta,     // 偏移参数 beta
    int64_t M,              // 第一维大小
    int64_t N,              // 第二维大小
    T eps,                  // epsilon 参数，用于数值稳定性
    Tensor* Y,              // 输出张量 Y
    Tensor* mean,           // 输出平均值张量
    Tensor* rstd) {         // 输出标准差的倒数张量

  // 使用向量化类型 Vec
  using Vec = vec::Vectorized<T>;
  
  // 获取输入张量 X 的数据指针
  const T* X_data = X.const_data_ptr<T>();
  // 获取 gamma 的数据指针，若未定义则设为 nullptr
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  // 获取 beta 的数据指针，若未定义则设为 nullptr
  const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
  // 获取输出张量 Y 的数据指针
  T* Y_data = Y->data_ptr<T>();
  // 获取输出平均值张量的数据指针，若不存在则设为 nullptr
  T* mean_data = mean ? mean->data_ptr<T>() : nullptr;
  // 获取输出标准差的倒数张量的数据指针，若不存在则设为 nullptr
  T* rstd_data = rstd ? rstd->data_ptr<T>() : nullptr;

  // 判断是否 gamma_data 为 nullptr
  const bool gamma_null = gamma_data == nullptr;
  // 判断是否 beta_data 为 nullptr
  const bool beta_null = beta_data == nullptr;
  // 判断是否 mean_data 为 nullptr
  const bool mean_null = mean_data == nullptr;
  // 判断是否 rstd_data 为 nullptr
  const bool rstd_null = rstd_data == nullptr;

  // 并行处理每个 M 维度的数据
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    // 循环处理每一行数据
    for (const auto i : c10::irange(start, end)) {
      // 获取当前行的 X 数据指针
      const T* X_ptr = X_data + i * N;
      // 获取当前行的 Y 数据指针
      T* Y_ptr = Y_data + i * N;
      // 计算当前行的均值和标准差的倒数
      auto [mean_val, rstd_val] = RowwiseMoments(X_ptr, N);
      // 计算标准差的倒数
      rstd_val = T(1) / std::sqrt(rstd_val + eps);
      // 缩放比例设为 rstd_val
      const T scale = rstd_val;
      // 偏移量设为 -mean_val
      const T bias = - mean_val;

      // 判断 gamma_data 或 beta_data 是否为 nullptr
      if (gamma_null || beta_null) {
        // 非向量化处理，逐元素计算 Y_ptr
        for (const auto j : c10::irange(N)) {
          const T gamma_v = gamma_null ? T(1) : gamma_data[j];
          const T beta_v = beta_null ? T(0) : beta_data[j];
          Y_ptr[j] = (X_ptr[j] + bias) * rstd_val * gamma_v + beta_v;
        }
      } else {
        // 向量化处理，使用 vec::map3 函数
        vec::map3<T>(
            [scale, bias](Vec x, Vec gamma, Vec beta) {
              return (x + Vec(bias)) * Vec(scale) * gamma + beta;
            },
            Y_ptr,
            X_ptr,
            gamma_data,
            beta_data,
            N);
      }

      // 如果 mean_data 不为 nullptr，则将 mean_val 存入
      if (!mean_null) {
        mean_data[i] = mean_val;
      }
      // 如果 rstd_data 不为 nullptr，则将 rstd_val 存入
      if (!rstd_null) {
        rstd_data[i] = rstd_val;
      }
    }
  });
}

// 混合数据类型的层归一化操作函数模板
template <typename T, typename param_t,
          typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
void layer_norm_kernel_mixed_type(
    const Tensor& X,        // 输入张量 X
    const Tensor& gamma,    // 缩放参数 gamma
    const Tensor& beta,     // 偏移参数 beta
    int64_t M,              // 第一维大小
    int64_t N,              // 第二维大小
    float eps,              // epsilon 参数，用于数值稳定性
    Tensor* Y,              // 输出张量 Y
    Tensor* mean,           // 输出平均值张量
    // 使用模板参数 T 定义 Tensor 指针 rstd
    Tensor* rstd) {
  // 使用别名 bVec 和 fVec 分别代表 Vectorized<T> 和 Vectorized<float>
  using bVec = Vectorized<T>;
  using fVec = Vectorized<float>;
  // 获取输入 Tensor X 的数据指针 X_data
  const T* X_data = X.const_data_ptr<T>();
  // 如果 gamma 已定义，则获取其数据指针 gamma_data；否则设为 nullptr
  const param_t* gamma_data = gamma.defined() ? gamma.const_data_ptr<param_t>() : nullptr;
  // 如果 beta 已定义，则获取其数据指针 beta_data；否则设为 nullptr
  const param_t* beta_data = beta.defined() ? beta.const_data_ptr<param_t>() : nullptr;
  // 获取输出 Tensor Y 的数据指针 Y_data
  T* Y_data = Y->data_ptr<T>();
  // 如果 mean 已定义，则获取其数据指针 mean_data；否则设为 nullptr
  param_t* mean_data = mean ? mean->data_ptr<param_t>() : nullptr;
  // 如果 rstd 已定义，则获取其数据指针 rstd_data；否则设为 nullptr
  param_t* rstd_data = rstd ? rstd->data_ptr<param_t>() : nullptr;

  // 检查是否 gamma_data、beta_data、mean_data、rstd_data 为 nullptr
  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;
  const bool mean_null = mean_data == nullptr;
  const bool rstd_null = rstd_data == nullptr;
  
  // 并行处理 M 次迭代，每次处理一段范围内的数据
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    // 遍历从 start 到 end 的索引 i
    for (const auto i : c10::irange(start, end)) {
      // 获取 X_ptr 和 Y_ptr 指向第 i 行数据的指针
      const T* X_ptr = X_data + i * N;
      T* Y_ptr = Y_data + i * N;
      // 计算第 i 行数据的均值 mean_val 和反标准差 rstd_val
      auto [mean_val, rstd_val] = RowwiseMoments(X_ptr, N);
      // 计算 rstd_val 的倒数再开方，得到标准差的倒数，存放在 rstd_val 中
      rstd_val = float(1) / std::sqrt(rstd_val + eps);
      // 计算缩放因子 scale 和偏置 bias
      const float scale = rstd_val;
      const float bias = -rstd_val * mean_val;
      int64_t d = 0;
      // 以 bVec::size() 为步长循环处理数据
      for (; d < N - (N % bVec::size()); d += bVec::size()) {
        // 加载 X_ptr + d 处的数据为 bVec 类型的向量 x_bvec
        bVec x_bvec = bVec::loadu(X_ptr + d);
        // 将 x_bvec 转换为两个 fVec 类型的浮点向量 x_fvec0 和 x_fvec1
        auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
        // 如果 gamma_data 为 nullptr，则设置 gamma_fvec0 和 gamma_fvec1 为 1；否则加载 gamma_data + d 处的两个参数
        auto [gamma_fvec0, gamma_fvec1] = gamma_null ? std::make_tuple(fVec(1), fVec(1)) : load2f(gamma_data + d);
        // 如果 beta_data 为 nullptr，则设置 beta_fvec0 和 beta_fvec1 为 0；否则加载 beta_data + d 处的两个参数
        auto [beta_fvec0, beta_fvec1] = beta_null ? std::make_tuple(fVec(0), fVec(0)) : load2f(beta_data + d);
        // 计算 y_fvec0 和 y_fvec1，并存储到 Y_ptr + d 处
        fVec y_fvec0 = (x_fvec0 * fVec(scale) + fVec(bias)) * gamma_fvec0 + beta_fvec0;
        fVec y_fvec1 = (x_fvec1 * fVec(scale) + fVec(bias)) * gamma_fvec1 + beta_fvec1;
        bVec y_bvec = convert_from_float<T>(y_fvec0, y_fvec1);
        y_bvec.store(Y_ptr + d);
      }
      // 处理剩余不足一个 bVec::size() 的数据
      for (; d < N; d++) {
        // 计算 gamma_v 和 beta_v 的值
        const float gamma_v = gamma_null ? float(1) : float(gamma_data[d]);
        const float beta_v = beta_null ? float(0) : float(beta_data[d]);
        // 计算 Y_ptr[d] 的值，并存储到 Y_ptr[d] 处
        Y_ptr[d] = (float(X_ptr[d]) * scale + bias) * gamma_v + beta_v;
      }
      // 如果 mean_data 非空，则将 mean_val 存储到 mean_data[i] 处
      if (!mean_null) {
        mean_data[i] = mean_val;
      }
      // 如果 rstd_data 非空，则将 rstd_val 存储到 rstd_data[i] 处
      if (!rstd_null) {
        rstd_data[i] = rstd_val;
      }
    }
  });
}

template <typename T,
          typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    float eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  // 检查是否输入数据类型混合
  const bool mixed_type = is_mixed_type(X, gamma, beta);
  // 如果是混合类型，则调用相应的混合类型核函数
  if (mixed_type) {
    layer_norm_kernel_mixed_type<T, float>(X, gamma, beta, M, N, eps, Y, mean, rstd);
  } else {
    // 否则调用相同类型的核函数
    layer_norm_kernel_mixed_type<T, T>(X, gamma, beta, M, N, eps, Y, mean, rstd);
  }
}

void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  // 断言输入张量X的元素数量等于M*N
  TORCH_DCHECK_EQ(X.numel(), M * N);
  // 断言gamma未定义或者其元素数量等于N
  DCHECK(!gamma.defined() || gamma.numel() == N);
  // 断言beta未定义或者其元素数量等于N
  DCHECK(!beta.defined() || beta.numel() == N);
  // 根据X的数据类型分派到具体的浮点类型处理函数，并命名为"LayerNormKernelImpl"
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, X.scalar_type(),
      "LayerNormKernelImpl", [&]() {
    LayerNormKernelImplInternal<scalar_t>(
        X, gamma, beta, M, N, eps, Y, mean, rstd);
  });
}

template <typename T, typename T2, typename opmath_t>
void layer_norm_backward_frame(
    const T* dY_data,
    const T* X_data,
    const T2* mean_data,
    const T2* rstd_data,
    const T2* gamma_data,
    T* dX_data,
    T* dgamma_buffer_ptr,
    T* dbeta_buffer_ptr,
    const opmath_t scale,
    const bool gamma_null,
    const bool dX_null,
    const bool dgamma_null,
    const bool dbeta_null,
    int64_t N,
    int64_t i) {
  // 使用Vectorized类型，处理opmath_t类型的向量化操作
  using Vec = vec::Vectorized<opmath_t>;
  // 计算指针偏移量，以访问对应元素
  const T* dY_ptr = dY_data + i * N;
  const T* X_ptr = X_data + i * N;
  // 如果dgamma不为null，则进行向量化操作
  if (!dgamma_null) {
    const opmath_t a = rstd_data[i];
    const opmath_t b = -a * mean_data[i];
    // 使用向量化函数，计算dgamma的更新值
    vec::map3<T>(
        [a, b](Vec dgamma, Vec dy, Vec x) {
          return dgamma + dy * (Vec(a) * x + Vec(b));
        },
        dgamma_buffer_ptr,
        dgamma_buffer_ptr,
        dY_ptr,
        X_ptr,
        N);
  }
  // 如果dbeta不为null，则进行向量化操作
  if (!dbeta_null) {
    // 使用向量化函数，计算dbeta的更新值
    vec::map2<T>(
        [](Vec dbeta, Vec dy) { return dbeta + dy; },
        dbeta_buffer_ptr,
        dbeta_buffer_ptr,
        dY_ptr,
        N);
  }
  // 如果dX不为null，则进行标量计算
  if (!dX_null) {
    T* dX_ptr = dX_data + i * N;
    opmath_t ds = opmath_t(0);
    opmath_t db = opmath_t(0);
    // 使用标量循环，计算dX的更新值
    // Scalar math:
    // for (const auto j : c10::irange(N)) {
    //   const T gamma_v = gamma_null ? T(1) : gamma_data[j];
    //   ds += dY_ptr[j] * X_ptr[j] * gamma_v;
    //   db += dY_ptr[j] * gamma_v;
    // }
    // 如果 gamma_null 为真，则执行以下代码块
    if (gamma_null) {
      // 计算 ds，对每对 dY_ptr 和 X_ptr 执行向量乘法，并对结果向量进行求和
      ds = vec::map2_reduce_all<T>(
          [](Vec x, Vec y) { return x * y; },
          [](Vec x, Vec y) { return x + y; },
          dY_ptr,
          X_ptr,
          N);
      // 计算 db，对 dY_ptr 中的所有向量进行求和
      db = vec::reduce_all<T>(
          [](Vec& x, Vec& y) { return x + y; }, dY_ptr, N);
    } else {
      // 计算 ds，对每对 dY_ptr、X_ptr 和 gamma_data 执行向量乘法，并对结果向量进行求和
      ds = vec::map3_reduce_all<T>(
          [](Vec x, Vec y, Vec z) { return x * y * z; },
          [](Vec x, Vec y) { return x + y; },
          dY_ptr,
          X_ptr,
          gamma_data,
          N);
      // 计算 db，对 dY_ptr 和 gamma_data 中的每对向量进行乘法，然后对结果向量进行求和
      db = vec::map2_reduce_all<T>(
          [](Vec x, Vec y) { return x * y; },
          [](Vec x, Vec y) { return x + y; },
          dY_ptr,
          gamma_data,
          N);
    }
    // 获取 rstd_data 数组中的值赋给常量 a
    const opmath_t a = rstd_data[i];
    // 计算常量 b，包括 db、mean_data[i]、ds、a、scale 等的数学运算
    const opmath_t b = (db * opmath_t(mean_data[i]) - ds) * a * a * a * scale;
    // 计算常量 c，包括 b、mean_data[i]、db、a、scale 等的数学运算
    const opmath_t c = -b * opmath_t(mean_data[i]) - db * a * scale;
    // 标量数学：
    // 对于 c10::irange(N) 中的每个索引 j，执行以下操作
    // const T gamma_v = gamma_null ? T(1) : gamma_data[j];
    // dX_ptr[j] = a * dY_ptr[j] * gamma_v + b * X_ptr[j] + c;
    // 如果 gamma_null 为真，则执行以下代码块
    if (gamma_null) {
      // 使用 vec::map2 函数对 dX_ptr、dY_ptr 和 X_ptr 中的向量执行映射操作，应用函数返回向量
      vec::map2<T>(
          [a, b, c](Vec dy, Vec x) {
            return Vec(a) * dy + Vec(b) * x + Vec(c);
          },
          dX_ptr,
          dY_ptr,
          X_ptr,
          N);
    } else {
      // 否则，使用 vec::map3 函数对 dX_ptr、dY_ptr、gamma_data 和 X_ptr 中的向量执行映射操作，应用函数返回向量
      vec::map3<T>(
          [a, b, c](Vec dy, Vec gamma, Vec x) {
            return Vec(a) * dy * gamma + Vec(b) * x + Vec(c);
          },
          dX_ptr,
          dY_ptr,
          gamma_data,
          X_ptr,
          N);
    }
  }
  
  // 如果 dgamma 不为空，则更新 dgamma_buffer_ptr
  if (!dgamma_null) {
    // 获取归一化标准差和均值的值
    const float a = rstd_data[i];
    const float b = -a * mean_data[i];
    // 使用向量化函数更新 dgamma_buffer_ptr
    vec::map3<T>(
        [a, b](fVec dgamma, fVec dy, fVec x) {
          return dgamma + dy * (fVec(a) * x + fVec(b));
        },
        dgamma_buffer_ptr,
        dgamma_buffer_ptr,
        dY_ptr,
        X_ptr,
        N);
  }
  
  // 如果 dbeta 不为空，则更新 dbeta_buffer_ptr
  if (!dbeta_null) {
    // 使用向量化函数更新 dbeta_buffer_ptr
    vec::map2<T>(
        [](fVec dbeta, fVec dy) { return dbeta + dy; },
        dbeta_buffer_ptr,
        dbeta_buffer_ptr,
        dY_ptr,
        N);
  }
  
  // 如果 dX 不为空，则更新 dX_ptr
  if (!dX_null) {
    // 计算 ds 和 db 的初始值
    float ds = float(0);
    float db = float(0);
    // 如果 gamma_null 为真，则使用向量化函数计算 ds 和 db 的累积值
    if (gamma_null) {
      ds = vec::map2_reduce_all<T>(
          [](fVec x, fVec y) { return x * y; },
          [](fVec x, fVec y) { return x + y; },
          dY_ptr,
          X_ptr,
          N);
      db = vec::reduce_all<T>(
          [](fVec& x, fVec& y) { return x + y; }, dY_ptr, N);
    }
    // 计算中间值 a, b, c
    const float a = rstd_data[i];
    const float b = (db * mean_data[i] - ds) * a * a * a * scale;
    const float c = -b * mean_data[i] - db * a * scale;
    // 使用向量化函数更新 dX_ptr
    if (gamma_null) {
      vec::map2<T>(
          [a, b, c](fVec dy, fVec x) {
            return fVec(a) * dy + fVec(b) * x + fVec(c);
          },
          dX_ptr,
          dY_ptr,
          X_ptr,
          N);
    }
  }
}


注释：

  }
  // 如果 dgamma 不为空，则更新 dgamma_buffer_ptr
  if (!dgamma_null) {
    // 获取归一化标准差和均值的值
    const float a = rstd_data[i];
    const float b = -a * mean_data[i];
    // 使用向量化函数更新 dgamma_buffer_ptr
    vec::map3<T>(
        [a, b](fVec dgamma, fVec dy, fVec x) {
          return dgamma + dy * (fVec(a) * x + fVec(b));
        },
        dgamma_buffer_ptr,
        dgamma_buffer_ptr,
        dY_ptr,
        X_ptr,
        N);
  }
  
  // 如果 dbeta 不为空，则更新 dbeta_buffer_ptr
  if (!dbeta_null) {
    // 使用向量化函数更新 dbeta_buffer_ptr
    vec::map2<T>(
        [](fVec dbeta, fVec dy) { return dbeta + dy; },
        dbeta_buffer_ptr,
        dbeta_buffer_ptr,
        dY_ptr,
        N);
  }
  
  // 如果 dX 不为空，则更新 dX_ptr
  if (!dX_null) {
    // 计算 ds 和 db 的初始值
    float ds = float(0);
    float db = float(0);
    // 如果 gamma_null 为真，则使用向量化函数计算 ds 和 db 的累积值
    if (gamma_null) {
      ds = vec::map2_reduce_all<T>(
          [](fVec x, fVec y) { return x * y; },
          [](fVec x, fVec y) { return x + y; },
          dY_ptr,
          X_ptr,
          N);
      db = vec::reduce_all<T>(
          [](fVec& x, fVec& y) { return x + y; }, dY_ptr, N);
    }
    // 计算中间值 a, b, c
    const float a = rstd_data[i];
    const float b = (db * mean_data[i] - ds) * a * a * a * scale;
    const float c = -b * mean_data[i] - db * a * scale;
    // 使用向量化函数更新 dX_ptr
    if (gamma_null) {
      vec::map2<T>(
          [a, b, c](fVec dy, fVec x) {
            return fVec(a) * dy + fVec(b) * x + fVec(c);
          },
          dX_ptr,
          dY_ptr,
          X_ptr,
          N);
    }
  }
}
    } else {
      // 如果不满足条件，执行以下代码块

      // 初始化 d 为 0
      int64_t d = 0;
      
      // 循环直到 d 达到 N - (N % bVec::size())
      for (; d < N - (N % bVec::size()); d += bVec::size()) {
        // 从 X_ptr 和 dY_ptr 处加载 bVec::size() 个元素到 x_bvec 和 dy_bvec
        bVec x_bvec = bVec::loadu(X_ptr + d);
        bVec dy_bvec = bVec::loadu(dY_ptr + d);
        
        // 将 x_bvec 和 dy_bvec 转换为 float 类型并分别存储到 x_fvec0, x_fvec1 和 dy_fvec0, dy_fvec1
        auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
        auto [dy_fvec0, dy_fvec1] = convert_to_float<T>(dy_bvec);
        
        // 从 gamma_data + d 处加载两个 float 值到 gamma_fvec0 和 gamma_fvec1
        auto [gamma_fvec0, gamma_fvec1] = load2f(gamma_data + d);
        
        // 计算结果向量 r_fvec0 和 r_fvec1
        fVec r_fvec0 = fVec(a) * dy_fvec0 * gamma_fvec0 + fVec(b) * x_fvec0 + fVec(c);
        fVec r_fvec1 = fVec(a) * dy_fvec1 * gamma_fvec1 + fVec(b) * x_fvec1 + fVec(c);
        
        // 将 r_fvec0 和 r_fvec1 转换为 bVec，并存储到 dX_ptr + d 处
        bVec r_bvec = convert_from_float<T>(r_fvec0, r_fvec1);
        r_bvec.store(dX_ptr + d);
      }
      
      // 处理剩余的不足 bVec::size() 的部分
      if (N - d > 0) {
        // 加载剩余部分的 x_bvec 和 dy_bvec
        bVec x_bvec = bVec::loadu(X_ptr + d, N - d);
        bVec dy_bvec = bVec::loadu(dY_ptr + d, N - d);
        
        // 转换为 float 类型并存储到 x_fvec0, x_fvec1 和 dy_fvec0, dy_fvec1
        auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
        auto [dy_fvec0, dy_fvec1] = convert_to_float<T>(dy_bvec);
        
        // 加载剩余部分的 gamma 值到 gamma_fvec0 和 gamma_fvec1
        auto [gamma_fvec0, gamma_fvec1] = load2f(gamma_data + d, N - d);
        
        // 计算结果向量 r_fvec0 和 r_fvec1
        fVec r_fvec0 = fVec(a) * dy_fvec0 * gamma_fvec0 + fVec(b) * x_fvec0 + fVec(c);
        fVec r_fvec1 = fVec(a) * dy_fvec1 * gamma_fvec1 + fVec(b) * x_fvec1 + fVec(c);
        
        // 将 r_fvec0 和 r_fvec1 转换为 bVec，并存储到 dX_ptr + d 处
        bVec r_bvec = convert_from_float<T>(r_fvec0, r_fvec1);
        r_bvec.store(dX_ptr + d, N - d);
      }
    }
  }
  // }
  // 结束 LayerNormBackwardKernelImplInternal 函数的实现

  // template <typename T, typename T2>
  // LayerNormBackwardKernelImplInternal 函数的模板定义，接受多个模板参数 T 和 T2

  // void LayerNormBackwardKernelImplInternal(
  //    LayerNorm 层的反向传播内核实现函数，计算 LayerNorm 操作的反向传播

  // const Tensor& dY,
  //    输入张量 dY，表示损失函数关于输出的梯度

  // const Tensor& X,
  //    输入张量 X，表示 LayerNorm 操作的输入

  // const Tensor& mean,
  //    输入张量 mean，表示均值

  // const Tensor& rstd,
  //    输入张量 rstd，表示标准差的倒数

  // const Tensor& gamma,
  //    输入张量 gamma，表示缩放系数

  // int64_t M,
  //    整数 M，表示输入张量的第一个维度大小

  // int64_t N,
  //    整数 N，表示输入张量的第二个维度大小

  // Tensor* dX,
  //    输出参数，指向存储 dX 的张量指针

  // Tensor* dgamma,
  //    输出参数，指向存储 dgamma 的张量指针

  // Tensor* dbeta) {
  //    输出参数，指向存储 dbeta 的张量指针

  // using opmath_t = at::opmath_type<T>;
  //    定义模板别名 opmath_t，用于执行 T 类型的操作

  // TORCH_DCHECK_EQ(dY.numel(), M * N);
  //    检查 dY 的元素数是否等于 M * N，确保维度匹配

  // TORCH_DCHECK_EQ(X.numel(), M * N);
  //    检查 X 的元素数是否等于 M * N，确保维度匹配

  // TORCH_DCHECK_EQ(mean.numel(), M);
  //    检查 mean 的元素数是否等于 M，确保维度匹配

  // TORCH_DCHECK_EQ(rstd.numel(), M);
  //    检查 rstd 的元素数是否等于 M，确保维度匹配

  // DCHECK(!gamma.defined() || gamma.numel() == N);
  //    使用 DCHECK 检查 gamma 是否未定义或者其元素数是否等于 N，确保维度匹配

  // const T* dY_data = dY.template const_data_ptr<T>();
  //    获取 dY 的数据指针，类型为 T*

  // const T* X_data = X.template const_data_ptr<T>();
  //    获取 X 的数据指针，类型为 T*

  // const T2* mean_data = mean.template const_data_ptr<T2>();
  //    获取 mean 的数据指针，类型为 T2*

  // const T2* rstd_data = rstd.template const_data_ptr<T2>();
  //    获取 rstd 的数据指针，类型为 T2*

  // const T2* gamma_data =
  //    gamma.defined() ? gamma.template const_data_ptr<T2>() : nullptr;
  //    获取 gamma 的数据指针，类型为 T2*，若未定义则设置为 nullptr

  // T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
  //    获取 dX 的数据指针，类型为 T*，若未定义则设置为 nullptr

  // T2* dgamma_data = dgamma->defined() ? dgamma->template data_ptr<T2>() : nullptr;
  //    获取 dgamma 的数据指针，类型为 T2*，若未定义则设置为 nullptr

  // T2* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T2>() : nullptr;
  //    获取 dbeta 的数据指针，类型为 T2*，若未定义则设置为 nullptr

  // const opmath_t scale = opmath_t(1) / static_cast<opmath_t>(N);
  //    定义缩放因子 scale，类型为 opmath_t，用于除以 N

  // const bool gamma_null = gamma_data == nullptr;
  //    检查 gamma_data 是否为 nullptr，用于条件判断

  // const bool dX_null = dX_data == nullptr;
  //    检查 dX_data 是否为 nullptr，用于条件判断

  // const bool dgamma_null = dgamma_data == nullptr;
  //    检查 dgamma_data 是否为 nullptr，用于条件判断

  // const bool dbeta_null = dbeta_data == nullptr;
  //    检查 dbeta_data 是否为 nullptr，用于条件判断

  // 1. Use two path parallel reduction for dgamma and dbeta:
  //    First path: allocate an immediate buffer of size {2, max_threads, N},
  //        dgamma_buffer = buffer[0], dbeta_buffer = buffer[1]
  //    Parallel along dim0 and reduce dY and X along dim0 to buffer.
  //    Second path: parallel along dim1 and reduce buffer to dgamma and dbeta.
  //
  // 2. Fuse first path of dgamma/dbeta with dX to reuse X[i] and dY[i] in L1
  // cache.
  //    使用两条路径并行减少 dgamma 和 dbeta：
  //    第一路径：分配大小为 {2, max_threads, N} 的立即缓冲区，
  //        dgamma_buffer = buffer[0], dbeta_buffer = buffer[1]
  //    沿 dim0 并行，将 dY 和 X 沿 dim0 减少到缓冲区。
  //    第二路径：沿 dim1 并行，将缓冲区减少到 dgamma 和 dbeta。
  //    
  //    将 dgamma/dbeta 的第一路径与 dX 融合，以重用 X[i] 和 dY[i] 在 L1 缓存中。

  // int num_threads = at::get_num_threads();
  //    获取当前线程数，用于并行计算

  // Tensor buffer = at::empty({0}, X.options());
  //    创建一个空张量 buffer，与 X 具有相同的选项

  // T* buffer_data = nullptr;
  //    初始化 buffer_data 指针为 nullptr

  // if (!dgamma_null || !dbeta_null) {
  //    如果 dgamma 或 dbeta 不为 nullptr，则执行以下操作：

  //    buffer.resize_({2, num_threads, N}).zero_();
  //        调整 buffer 的大小为 {2, num_threads, N}，并将其元素置零

  //    buffer_data = buffer.template data_ptr<T>();
  //        获取 buffer 的数据指针，类型为 T*
  // }

  // First path of dgamma/dbeta and dX
  //    dgamma/dbeta 和 dX 的第一路径计算

  // at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
  //    使用并行循环计算从 start 到 end 的索引范围

  //    int tid = at::get_thread_num();
  //        获取当前线程的线程 ID

  //    TORCH_CHECK(
  //        tid < num_threads,
  //        "expect thread id smaller than ",
  //        num_threads,
  //        ", got thread id ",
  //        tid);
  //        检查线程 ID 是否小于当前线程数，否则抛出错误消息

  //    T* dgamma_buffer_ptr = dgamma_null ? nullptr : buffer_data + tid * N;
  //        初始化 dgamma_buffer_ptr，如果 dgamma_null 为真则设置为 nullptr，
  //        否则设置为 buffer_data 加上 tid 乘以 N

  //    T* dbeta_buffer_ptr =
  //        dbeta_null ? nullptr : buffer_data + num_threads * N + tid * N;
  //        初始化 dbeta_buffer_ptr，如果 dbeta_null 为真则设置为 nullptr，
  //        否则设置为 buffer_data 加上 num_threads 乘以 N 加上 tid 乘以 N

  //    for (const auto i : c10::irange(start, end)) {
  //        循环遍历 start 到 end 范围内的索引 i

  //        layer_norm_backward_frame<T, T2, opmath_t>(
  //            dY_data, X_data, mean_data, rstd_data, gamma_data,
  //            dX_data, dgamma_buffer_ptr
    // 使用并行循环执行以下操作：从0到N，步长为1
    parallel_for(0, N, 1, [&](int64_t start, int64_t end) {
      // 对于每个范围内的j值进行迭代
      for (const auto j : c10::irange(start, end)) {
        // 初始化 dgamma_v 和 dbeta_v 变量为零
        opmath_t dgamma_v = opmath_t(0);
        opmath_t dbeta_v = opmath_t(0);
        // 对于每个线程的范围进行迭代
        for (const auto i : c10::irange(num_threads)) {
          // 累加 buffer_data 中的值到 dgamma_v 和 dbeta_v
          dgamma_v += buffer_data[i * N + j];
          dbeta_v += buffer_data[num_threads * N + i * N + j];
        }
        // 如果 dgamma_null 为假，则将 dgamma_v 存入 dgamma_data[j] 中
        if (!dgamma_null) {
          // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
          dgamma_data[j] = dgamma_v;
        }
        // 如果 dbeta_null 为假，则将 dbeta_v 存入 dbeta_data[j] 中
        if (!dbeta_null) {
          // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
          dbeta_data[j] = dbeta_v;
        }
      }
    });
  }
}

void LayerNormBackwardKernelImpl(
    const Tensor& dY,                    // 输入：梯度 dY
    const Tensor& X,                     // 输入：正向传播时的输入 X
    const Tensor& mean,                  // 输入：均值 mean
    const Tensor& rstd,                  // 输入：标准差的倒数 rstd
    const Tensor& gamma,                 // 输入：缩放参数 gamma
    int64_t M,                           // 输入：维度 M
    int64_t N,                           // 输入：维度 N
    Tensor* dX,                          // 输出：对 X 的梯度 dX
    Tensor* dgamma,                      // 输出：对 gamma 的梯度 dgamma
    Tensor* dbeta) {                     // 输出：对 beta 的梯度 dbeta
  if (at::isReducedFloatingType(X.scalar_type())) {  // 如果 X 是降维浮点类型
    AT_DISPATCH_REDUCED_FLOATING_TYPES(X.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
      if (gamma.scalar_type() == at::kFloat) {  // 如果 gamma 是 float 类型
        // 调用内部函数，计算梯度
        LayerNormBackwardKernelImplInternal<scalar_t, float>(
            dY.contiguous(), X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
      } else {
        // 调用内部函数，计算梯度
        LayerNormBackwardKernelImplInternal<scalar_t, scalar_t>(
            dY.contiguous(), X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
      }
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
      // 调用内部函数，计算梯度
      LayerNormBackwardKernelImplInternal<scalar_t, scalar_t>(
          dY.contiguous(), X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
    });
  }
}

} // namespace

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);  // 注册正向传播的分发函数

REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl);  // 注册反向传播的分发函数

} // namespace at::native  // 结束 at::native 命名空间
```