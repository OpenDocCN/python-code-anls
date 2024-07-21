# `.\pytorch\aten\src\ATen\native\cpu\WeightNormKernel.cpp`

```py
// 定义宏，禁用Torch的运算符断言功能
#define TORCH_ASSERT_NO_OPERATORS
// 包含ATen库中的TensorBase头文件
#include <ATen/core/TensorBase.h>

// 包含ATen库中的调度功能
#include <ATen/Dispatch.h>
// 包含ATen库中的空张量处理功能
#include <ATen/EmptyTensor.h>
// 包含ATen库中的并行处理功能
#include <ATen/Parallel.h>
// 包含ATen库中的数学运算类型定义
#include <ATen/OpMathType.h>
// 包含ATen库中CPU上的权重归一化核心功能
#include <ATen/native/cpu/WeightNormKernel.h>
// 包含ATen库中CPU上的向量功能
#include <ATen/cpu/vec/functional.h>
// 包含ATen库中CPU上的向量功能和矢量化
#include <ATen/cpu/vec/vec.h>
// 包含C10库中的范围迭代器功能
#include <c10/util/irange.h>

// ATen命名空间
namespace at::native {

// 匿名命名空间开始
namespace {

// 模板函数：在权重归一化处理中，针对第一维度的核心计算
template <typename scalar_t, typename accscalar_t>
void weight_norm_first_dim_kernel(
    TensorBase& w,                 // 权重张量
    TensorBase& norm,              // 归一化结果张量
    const TensorBase& v,           // 输入张量v
    const TensorBase& g,           // 缩放因子张量g
    int64_t M, int64_t N) {        // 维度参数M和N
  const auto v_data = v.data_ptr<scalar_t>();     // 获取输入张量v的数据指针
  const auto g_data = g.data_ptr<scalar_t>();     // 获取缩放因子张量g的数据指针
  auto w_data = w.data_ptr<scalar_t>();           // 获取权重张量w的数据指针
  auto norm_data = norm.data_ptr<accscalar_t>();  // 获取归一化结果张量norm的数据指针

  using Vec = vec::Vectorized<accscalar_t>;       // 使用Vectorized类型进行矢量化操作
  // 并行处理：对M范围内的索引执行以下操作
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {  // 对每个索引i执行以下操作
      // 计算第i行的归一化值norm_val：对v_data中第i行的N个元素进行平方和操作
      accscalar_t norm_val = vec::map_reduce_all<scalar_t>(
          [](Vec x) { return x * x; },               // 映射函数：计算平方
          [](Vec x, Vec y) { return x + y; },        // 归约函数：计算和
          v_data + i * N,                            // 输入数据起始地址
          N);                                        // 元素个数

      norm_val = std::sqrt(norm_val);                // 对归一化值取平方根
      norm_data[i] = norm_val;                       // 将归一化结果存入norm_data中

      accscalar_t a = g_data[i] / norm_val;          // 计算缩放因子a
      // 对第i行的权重w_data进行归一化处理：w_data = w_data * a
      vec::map(
          [a](Vec x) { return x * Vec(a); },         // 映射函数：乘以缩放因子a
          w_data + i * N,                            // 输出数据起始地址
          v_data + i * N,                            // 输入数据起始地址
          N);                                        // 元素个数
    }
  });
}

// 函数模板：计算每行的范数和
template <typename scalar_t>
inline void sum_norm_per_row(
    scalar_t* out_ptr,              // 输出指针
    const scalar_t* v_ptr,          // 输入指针
    int64_t size) {                 // 数据大小
  using Vec = vec::Vectorized<scalar_t>;  // 使用Vectorized类型进行矢量化操作
  // 对每对数据执行以下操作：out_ptr += v_ptr * v_ptr
  vec::map2(
      [](Vec out, Vec v) { return out + v * v; },  // 映射函数：计算平方和
      out_ptr,                                     // 输出数据起始地址
      out_ptr,                                     // 输入输出数据起始地址（in-place）
      v_ptr,                                       // 输入数据起始地址
      size);                                       // 元素个数
}

// 函数：计算每行的范数和，特化版本（针对BFloat16输入）
inline void sum_norm_per_row(
    float* out_ptr,                 // 输出指针
    const BFloat16* v_ptr,          // BFloat16输入指针
    int64_t size) {                 // 数据大小
  using bVec = vec::Vectorized<BFloat16>;  // 使用BFloat16的Vectorized类型进行矢量化操作
  using fVec = vec::Vectorized<float>;     // 使用float的Vectorized类型进行矢量化操作
  int64_t d = 0;                          // 初始化计数器d
  // 对数据执行以下操作，每次处理bVec::size()个元素
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec v_bvec = bVec::loadu(v_ptr + d);           // 加载未对齐的BFloat16数据
    auto [v_fvec0, v_fvec1] = convert_bfloat16_float(v_bvec);  // 转换为float类型数据

    fVec out_fvec0 = fVec::loadu(out_ptr + d) + v_fvec0 * v_fvec0;  // 计算平方和并加载数据
    fVec out_fvec1 = fVec::loadu(out_ptr + d + fVec::size()) + v_fvec1 * v_fvec1;  // 计算平方和并加载数据
    out_fvec0.store(out_ptr + d);         // 存储计算结果
    out_fvec1.store(out_ptr + d + fVec::size());  // 存储计算结果
  }
  // 处理剩余的数据，逐个计算平方和
  for(; d < size; ++d) {
    float v_val = float(v_ptr[d]);        // 转换为float类型
    out_ptr[d] += v_val * v_val;          // 计算平方和并累加
  }
}

// 函数模板：对每行应用归一化
template <typename scalar_t>
inline void apply_norm_per_row(
    scalar_t* w_ptr,                 // 权重指针
    const scalar_t* v_ptr,           // 输入指针
    const scalar_t* a_ptr,           // 缩放因子指针
    int64_t size) {                  // 数据大小
  using Vec = vec::Vectorized<scalar_t>;  // 使用Vectorized类型进行矢量化操作
  // 对每对数据执行以下操作：w_ptr = v_ptr * a_ptr
  vec::map2(
      [](Vec v, Vec a) { return v * a; },  // 映射函数：乘以缩放因子a
      w_ptr,                               // 输出数据起始地址
      v_ptr,                               // 输入数据起始地址
      a_ptr,                               // 缩放因子起始地址
      size);                               // 元素个数
}

// 函数：对每行应用归一化，特化版本（针对BFloat16输入）
inline void apply_norm_per_row(
    BFloat16* w_ptr,                 // 权重指针
    const BFloat16* v_ptr,           // 输入指针
    const float* a_ptr,              // 缩放因子指针
    int64_t size) {                  // 数据大小
  using bVec = vec::Vectorized<BFloat16>;  // 使用BFloat16的Vectorized类型进行矢量化操作
  using fVec = vec::Vectorized<float>;     // 使用float的Vectorized类型进行矢量化操作
  int64_t d = 0;                          // 初始化计数器d
  // 对数据执行以下操作，每次处理bVec::size()
    auto [v_fvec0, v_fvec1] = convert_bfloat16_float(v_bvec);
    # 使用 convert_bfloat16_float 函数将 bfloat16 向量 v_bvec 转换为两个 float 向量 v_fvec0 和 v_fvec1

    fVec w_fvec0 = fVec::loadu(a_ptr + d) * v_fvec0;
    # 从内存地址 a_ptr + d 处加载一个未对齐的 fVec 类型向量，并与 v_fvec0 相乘，得到结果存入 w_fvec0

    fVec w_fvec1 = fVec::loadu(a_ptr + d + fVec::size()) * v_fvec1;
    # 从内存地址 a_ptr + d + fVec::size() 处加载一个未对齐的 fVec 类型向量，并与 v_fvec1 相乘，得到结果存入 w_fvec1

    bVec w_bvec = convert_float_bfloat16(w_fvec0, w_fvec1);
    # 使用 convert_float_bfloat16 函数将 float 向量 w_fvec0 和 w_fvec1 转换为 bfloat16 向量 w_bvec

    w_bvec.store(w_ptr + d);
    # 将 bfloat16 向量 w_bvec 存储到内存地址 w_ptr + d 处
  }
  for(; d < size; ++d) {
    w_ptr[d] = float(v_ptr[d]) * a_ptr[d];
    # 对于剩余的每个索引 d，计算 v_ptr[d] 的 float 值与 a_ptr[d] 的乘积，并将结果存储到 w_ptr[d] 处
  }
// 结束模板函数 weight_norm_last_dim_kernel 的声明
template <typename scalar_t, typename accscalar_t>
void weight_norm_last_dim_kernel(
    // 输入张量 w，用于存储权重数据
    TensorBase& w,
    // 输出张量 norm，用于存储权重的范数
    TensorBase& norm,
    // 输入张量 v，包含权重的原始数据
    const TensorBase& v,
    // 输入张量 g，包含权重的缩放因子
    const TensorBase& g,
    // 权重矩阵的行数 M
    int64_t M, int64_t N) {
  // 获取张量 v 的数据指针
  const auto v_data = v.data_ptr<scalar_t>();
  // 获取张量 g 的数据指针
  const auto g_data = g.data_ptr<scalar_t>();
  // 获取张量 w 的数据指针，用于存储处理后的权重数据
  auto w_data = w.data_ptr<scalar_t>();
  // 获取张量 norm 的数据指针，用于存储每行权重的范数
  auto norm_data = norm.data_ptr<accscalar_t>();

  // 获取当前线程数
  int num_threads = at::get_num_threads();
  // 创建并初始化一个与线程数和 N 大小匹配的缓冲区张量
  TensorBase buffer = at::detail::empty_cpu({num_threads, N}, norm.options()).zero_();
  // 获取缓冲区张量的数据指针
  auto buffer_data = buffer.data_ptr<accscalar_t>();

  // 垂直方向的并行归约
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    // 获取当前线程的线程 ID
    int tid = at::get_thread_num();
    // 检查线程 ID 是否小于总线程数
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    // 计算当前线程在缓冲区中的起始指针
    auto buffer_ptr = buffer_data + tid * N;
    // 遍历处理 v 的行数据，累加每行的范数到缓冲区中
    for (const auto i : c10::irange(begin, end)) {
      sum_norm_per_row(buffer_ptr, v_data + i * N, N);
    }
  });

  // 对每列进行归一化计算
  for (const auto j : c10::irange(N)) {
    accscalar_t sum = 0;
    // 对所有线程的缓冲区进行归约求和
    for (const auto t : c10::irange(num_threads)) {
      sum += buffer_data[t * N + j];
    }
    // 计算每列的平方根作为范数，并存储到 norm_data 中
    norm_data[j] = std::sqrt(sum);
  }

  // 重复使用缓冲区的第一行来存储 g / norm 的值
  vec::convert(g_data, buffer_data, N);
  using Vec = vec::Vectorized<accscalar_t>;
  // 对 g_data 和 norm_data 中的对应元素进行除法运算，结果存储在 buffer_data 中
  vec::map2(
      [](Vec g, Vec norm) { return g / norm; },
      buffer_data,
      buffer_data,
      norm_data,
      N);

  // 应用权重归一化后的结果 w = v * (g/norm)，并行处理每行
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    // 对每一行的权重进行应用归一化操作
    for (const auto i : c10::irange(begin, end)) {
      apply_norm_per_row(w_data + i * N, v_data + i * N, buffer_data, N);
    }
  });
}

// 开始模板函数 weight_norm_backward_first_dim_kernel 的声明
template <typename scalar_t, typename accscalar_t>
void weight_norm_backward_first_dim_kernel(
    // 输出张量 grad_v，用于存储 v 的梯度
    TensorBase& grad_v,
    // 输出张量 grad_g，用于存储 g 的梯度
    TensorBase& grad_g,
    // 输入张量 grad_w，包含权重 w 的梯度
    const TensorBase& grad_w,
    // 输入张量 saved_v，包含权重 v 的原始数据
    const TensorBase& saved_v,
    // 输入张量 saved_g，包含权重 g 的原始数据
    const TensorBase& saved_g,
    // 输入张量 saved_norm，包含权重的范数
    const TensorBase& saved_norm,
    // 权重矩阵的行数 M
    int64_t M, int64_t N) {
  // 获取 grad_w 的数据指针
  const auto grad_w_data = grad_w.data_ptr<scalar_t>();
  // 获取 saved_v 的数据指针
  const auto saved_v_data = saved_v.data_ptr<scalar_t>();
  // 获取 saved_g 的数据指针
  const auto saved_g_data = saved_g.data_ptr<scalar_t>();
  // 获取 saved_norm 的数据指针
  const auto saved_norm_data = saved_norm.data_ptr<accscalar_t>();
  // 获取 grad_v 的数据指针，用于存储 v 的梯度
  auto grad_v_data = grad_v.data_ptr<scalar_t>();
  // 获取 grad_g 的数据指针，用于存储 g 的梯度
  auto grad_g_data = grad_g.data_ptr<scalar_t>();

  // 定义向量化类型 Vec，用于加速计算
  using Vec = vec::Vectorized<accscalar_t>;
  // 并行处理每行数据，计算梯度
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    // 对每行数据进行处理
    for (const auto i : c10::irange(begin, end)) {
      // TODO: Implement gradient computation for each row
      // 这里应该实现对每行数据的梯度计算，但具体实现未在提供的代码中给出
    }
  });
}
    // 对于范围 [begin, end) 中的每个索引 i，执行以下操作
    for (const auto i : c10::irange(begin, end)) {
      // 计算每个维度上的和，使用 map2_reduce_all 函数
      accscalar_t per_dim_sum_val = vec::map2_reduce_all<scalar_t>(
          // 使用 lambda 函数计算 grad_w 和 saved_v 的乘积
          [](Vec grad_w, Vec saved_v) { return grad_w * saved_v; },
          // 使用 lambda 函数计算两个向量的和
          [](Vec x, Vec y) { return x + y; },
          grad_w_data + i * N,    // grad_w 数据的起始位置
          saved_v_data + i * N,   // saved_v 数据的起始位置
          N);                     // 向量的长度 N

      // 获取 saved_norm_data 中的保存的归一化值
      accscalar_t saved_norm_val = saved_norm_data[i];
      // 获取 saved_g_data 中的保存的梯度值，并转换为 accscalar_t 类型
      accscalar_t saved_g_val = accscalar_t(saved_g_data[i]);
      // 计算 grad_g_val，即 per_dim_sum_val 除以 saved_norm_val
      accscalar_t grad_g_val = per_dim_sum_val / saved_norm_val;

      // 更新 grad_g_data 中的梯度值为 grad_g_val
      grad_g_data[i] = scalar_t(grad_g_val);
      // 计算 a = saved_g_val / saved_norm_val
      accscalar_t a = saved_g_val / saved_norm_val;
      // 计算 b = a * grad_g_val / saved_norm_val
      accscalar_t b = a * grad_g_val / saved_norm_val;

      // 使用 map2 函数更新 grad_v_data
      vec::map2(
          // 使用 lambda 函数计算更新 grad_v_data
          [a, b](Vec grad_w, Vec v) { return Vec(a) * grad_w - Vec(b) * v; },
          grad_v_data + i * N,    // grad_v 数据的起始位置
          grad_w_data + i * N,    // grad_w 数据的起始位置
          saved_v_data + i * N,   // saved_v 数据的起始位置
          N);                     // 向量的长度 N
    }
// 基于模板的函数，计算每行的加权和，用于计算梯度更新
template <typename scalar_t>
inline void sum_product_per_row(
    scalar_t* out_ptr,                           // 输出数组指针，保存每行的加权和结果
    const scalar_t* grad_w_ptr,                  // 梯度权重数组指针，用于加权
    const scalar_t* v_ptr,                       // 输入向量数组指针，参与加权
    int64_t size) {                              // 数组大小

  using Vec = vec::Vectorized<scalar_t>;          // 使用标量类型的向量化类型 Vec

  // 使用向量化操作，对每个元素进行加权和的计算
  vec::map3(
      [](Vec out, Vec grad_w, Vec v) { return out + grad_w * v; },  // lambda 函数，计算加权和
      out_ptr,                          // 输出数组指针
      out_ptr,                          // 输出数组指针（被重用，保存更新后的加权和）
      grad_w_ptr,                       // 梯度权重数组指针
      v_ptr,                            // 输入向量数组指针
      size);                            // 数组大小
}

// 特化版本，处理 BFloat16 类型的输入，实现每行的加权和计算
inline void sum_product_per_row(
    float* out_ptr,                           // 输出数组指针，保存每行的加权和结果
    const BFloat16* grad_w_ptr,               // BFloat16 类型的梯度权重数组指针，用于加权
    const BFloat16* v_ptr,                    // BFloat16 类型的输入向量数组指针，参与加权
    int64_t size) {                           // 数组大小

  using bVec = vec::Vectorized<BFloat16>;     // 使用 BFloat16 类型的向量化类型 bVec
  using fVec = vec::Vectorized<float>;        // 使用 float 类型的向量化类型 fVec

  int64_t d = 0;                             // 初始化循环索引

  // 处理向量大小的整数倍的部分
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec grad_w_bvec = bVec::loadu(grad_w_ptr + d);             // 加载 BFloat16 类型的梯度权重向量
    auto [grad_w_fvec0, grad_w_fvec1] = convert_bfloat16_float(grad_w_bvec);  // 转换为 float 类型向量

    bVec v_bvec = bVec::loadu(v_ptr + d);                       // 加载 BFloat16 类型的输入向量
    auto [v_fvec0, v_fvec1] = convert_bfloat16_float(v_bvec);   // 转换为 float 类型向量

    // 计算每行的加权和
    fVec out_fvec0 = fVec::loadu(out_ptr + d) + grad_w_fvec0 * v_fvec0;
    fVec out_fvec1 = fVec::loadu(out_ptr + d + fVec::size()) + grad_w_fvec1 * v_fvec1;

    // 存储计算结果
    out_fvec0.store(out_ptr + d);
    out_fvec1.store(out_ptr + d + fVec::size());
  }

  // 处理剩余部分（非向量大小的整数倍）
  for(; d < size; ++d) {
    float grad_w_val = float(grad_w_ptr[d]);     // 将 BFloat16 类型的梯度权重转换为 float 类型
    float v_val = float(v_ptr[d]);               // 将 BFloat16 类型的输入向量转换为 float 类型
    out_ptr[d] += grad_w_val * v_val;            // 计算每行的加权和
  }
}

// 基于模板的函数，计算每行的应用反向传播结果
template <typename scalar_t>
inline void apply_per_row_backward(
    scalar_t* grad_v_ptr,                        // 输出梯度向量指针
    const scalar_t* grad_w_ptr,                  // 梯度权重向量指针
    const scalar_t* v_ptr,                       // 输入向量指针
    const scalar_t* a_ptr,                       // 辅助向量 a 指针
    const scalar_t* b_ptr,                       // 辅助向量 b 指针
    int64_t size) {                              // 数组大小

  using Vec = vec::Vectorized<scalar_t>;          // 使用标量类型的向量化类型 Vec

  // 使用向量化操作，对每个元素进行反向传播的计算
  vec::map4(
      [](Vec grad_w, Vec v, Vec a, Vec b) { return a * grad_w - b * v; },  // lambda 函数，计算反向传播结果
      grad_v_ptr,                       // 输出梯度向量指针
      grad_w_ptr,                       // 梯度权重向量指针
      v_ptr,                            // 输入向量指针
      a_ptr,                            // 辅助向量 a 指针
      b_ptr,                            // 辅助向量 b 指针
      size);                            // 数组大小
}

// 特化版本，处理 BFloat16 类型的输入，实现每行的应用反向传播计算
inline void apply_per_row_backward(
    BFloat16* grad_v_ptr,                        // 输出梯度向量指针
    const BFloat16* grad_w_ptr,                  // BFloat16 类型的梯度权重向量指针
    const BFloat16* v_ptr,                       // BFloat16 类型的输入向量指针
    const float* a_ptr,                          // float 类型的辅助向量 a 指针
    const float* b_ptr,                          // float 类型的辅助向量 b 指针
    int64_t size) {                              // 数组大小

  using bVec = vec::Vectorized<BFloat16>;        // 使用 BFloat16 类型的向量化类型 bVec
  using fVec = vec::Vectorized<float>;           // 使用 float 类型的向量化类型 fVec

  int64_t d = 0;                                // 初始化循环索引

  // 处理向量大小的整数倍的部分
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec grad_w_bvec = bVec::loadu(grad_w_ptr + d);             // 加载 BFloat16 类型的梯度权重向量
    auto [grad_w_fvec0, grad_w_fvec1] = convert_bfloat16_float(grad_w_bvec);  // 转换为 float 类型向量

    bVec v_bvec = bVec::loadu(v_ptr + d);                       // 加载 BFloat16 类型的输入向量
    auto [v_fvec0, v_fvec1] = convert_bfloat16_float(v_bvec);   // 转换为 float 类型向量

    // 计算每行的应用反向传播结果
    fVec grad_v_fvec0 = fVec::loadu(a_ptr + d) * grad_w_fvec0 - fVec::loadu(b_ptr + d) * v_fvec0;
    fVec grad_v_fvec1 = fVec::loadu(a_ptr + d + fVec::size()) * grad_w_fvec1
        - fVec::loadu(b_ptr + d + fVec::size()) * v_fvec1;

    // 将 float 类型向量转换为 BFloat16 类型向量，并存储结果
    bVec grad_v_bvec = convert_float_bfloat16(grad_v_fvec0, grad_v_fvec1);
    grad_v_bvec.store(grad_v_ptr + d);
  }

  // 处理剩余部分（非向量大小的整数倍）
  for(; d < size; ++d) {
    grad_v_ptr[d] = float(grad_w_ptr[d]) * a_ptr[d] - float(v_ptr[d]) * b_ptr[d];  // 计算每行的应用反向传播结果
  }
}
    const TensorBase& saved_g,
    const TensorBase& saved_norm,
    int64_t M, int64_t N) {
  const auto grad_w_data = grad_w.data_ptr<scalar_t>();
  const auto saved_v_data = saved_v.data_ptr<scalar_t>();
  const auto saved_g_data = saved_g.data_ptr<scalar_t>();
  const auto saved_norm_data = saved_norm.data_ptr<accscalar_t>();
  auto grad_v_data = grad_v.data_ptr<scalar_t>();

  // 创建指向梯度张量数据的指针
  auto grad_g_data = grad_g.data_ptr<scalar_t>();

  // 准备临时缓冲区将被两次使用：
  // 1. 垂直从 [M, N] 到 [T, N] 的归约
  // 2. 存储 `sum`、`a` 和 `b` 的中间数据，
  //    因此需要确保至少有 3 行
  //
  int num_threads = at::get_num_threads();
  int K = std::max(3, num_threads);
  // 创建并初始化缓冲区张量，用于存储中间数据
  TensorBase buffer = at::detail::empty_cpu({K, N}, saved_norm.options()).zero_();
  auto buffer_data = buffer.data_ptr<accscalar_t>();

  // 垂直并行归约
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    // 检查线程 ID 是否小于总线程数
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    auto buffer_ptr = buffer_data + tid * N;
    for (const auto i : c10::irange(begin, end)) {
      // 对每一行执行行内积的归约操作
      sum_product_per_row(buffer_ptr, grad_w_data + i * N, saved_v_data + i * N, N);
    }
  });

  // 将结果存储在缓冲区的第一行
  for (const auto j : c10::irange(N)) {
    accscalar_t sum = 0;
    for (const auto t : c10::irange(num_threads)) {
      // 对所有线程的结果进行求和
      sum += buffer_data[t * N + j];
    }
    buffer_data[j] = sum;
  }

  // 重新使用缓冲区的第一行存储总和、系数 a 和 b
  accscalar_t* per_dim_sum = buffer_data;
  accscalar_t* a = buffer_data + N;
  accscalar_t* b = buffer_data + 2 * N;

  // 计算系数 a 和 b
  for (const auto j : c10::irange(N)) {
    accscalar_t saved_norm_val = saved_norm_data[j];
    accscalar_t saved_g_val = accscalar_t(saved_g_data[j]);
    // 计算梯度 g 对应的标准化值
    accscalar_t grad_g_val = per_dim_sum[j] / saved_norm_val;
    grad_g_data[j] = scalar_t(grad_g_val);

    // 计算系数 a
    a[j] = saved_g_val / saved_norm_val;
    // 计算系数 b
    b[j] = a[j] * grad_g_val / saved_norm_val;
  }

  // 应用 grad_v = a * grad_w - b * v 的反向计算
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      // 对每一行应用反向计算
      apply_per_row_backward(
          grad_v_data + i * N,
          grad_w_data + i * N,
          saved_v_data + i * N,
          a,
          b,
          N);
    }
  });
} // anonymous namespace



// 结束匿名命名空间，该命名空间用于限定代码块的作用域，避免命名冲突
} // anonymous namespace

// 注册调度函数 `weight_norm_stub`，将 `weight_norm_kernel` 作为实现
REGISTER_DISPATCH(weight_norm_stub, &weight_norm_kernel);

// 注册调度函数 `weight_norm_backward_stub`，将 `weight_norm_backward_kernel` 作为实现
REGISTER_DISPATCH(weight_norm_backward_stub, &weight_norm_backward_kernel);

// 结束 `at::native` 命名空间，这里的 `at::native` 命名空间包含了所有相关的函数定义和注册
} // at::native
```