# `.\pytorch\aten\src\ATen\native\cpu\batch_norm_kernel.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含批归一化的相关头文件
#include <ATen/native/batch_norm.h>

// 包含基础的张量操作和类型定义
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>
#include <ATen/OpMathType.h>

// 根据条件包含不同的操作函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace {

using namespace vec;

// 函数模板，用于在 CPU 上收集线性和常数项
template<typename param_t, typename opmath_t>
void batch_norm_cpu_collect_linear_and_constant_terms(
    opmath_t* alpha, opmath_t* beta, int64_t n_channel,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {

  // 获取权重和偏置数据的指针，如果未定义则置为 nullptr
  const param_t* weight_data = weight.defined() ? weight.const_data_ptr<param_t>() : nullptr;
  const param_t* bias_data = bias.defined() ? bias.const_data_ptr<param_t>() : nullptr;

  // 定义用于访问保存均值、逆标准差、运行均值、运行方差的访问器
  auto save_mean_a = conditional_accessor_1d<const param_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<const param_t>(save_invstd);
  auto running_mean_a = conditional_accessor_1d<const param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<const param_t>(running_var);

  /// Collect the linear and constant terms regarding the input.
  /// output(n, c, h, w)
  ///     = (input(n, c, h, w) - mean(c)) / sqrt(var(c) + eps) * weight(c)
  ///         + bias(c)
  ///     = input(n, c, h, w) * inv_var(c) * weight(c)
  ///         - mean(c) * inv_var(c) * weight(c) + bias(c),
  /// where inv_var(c) = 1 / sqrt(var(c) + eps).
  /// So the linear term, alpha(c) = inv_var(c) * weight(c),
  ///   the constant term beta(c) = bias(c) - mean(c) * inv_var(c) * weight(c)
  /// Note that this is only a good idea if (input_size >> c), in degenerate
  /// cases where image_size == 1 && batch_size == 1, it is slow.
  // 遍历通道数，计算线性和常数项
  for (const auto c : c10::irange(n_channel)) {
    opmath_t mean, invstd;
    // 根据训练状态选择使用保存的均值和逆标准差，或者运行时的均值和逆标准差
    if (train) {
      mean = save_mean_a[c];
      invstd = save_invstd_a[c];
    } else {
      mean = running_mean_a[c];
      invstd = 1 / std::sqrt(running_var_a[c] + static_cast<opmath_t>(eps));
    }
    // 获取权重和偏置值，如果未定义则使用默认值
    param_t weight_v = weight_data ? weight_data[c] : param_t(1);
    param_t bias_v = bias_data ? bias_data[c] : param_t(0);
    // 计算 alpha 和 beta
    alpha[c] = invstd * weight_v;
    beta[c] = bias_v - mean * alpha[c];
  }
}

/// A fast path for CPU inference and training forward when all tensors are contiguous.
// 当所有张量都是连续时，用于 CPU 推断和训练前向传播的快速路径
template<typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
batch_norm_cpu_contiguous_impl(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {


// 定义函数，参数包括权重、偏置、保存的均值、保存的标准差倒数、运行时均值、运行时方差、训练标志、epsilon值
// 函数开始

  using Vec = Vectorized<scalar_t>;
  // 使用别名Vec表示Vectorized<scalar_t>类型

  int64_t n_batch = input.size(0);
  // 获取输入张量的批次数
  int64_t n_channel = input.size(1);
  // 获取输入张量的通道数
  int64_t image_size = input.numel() / n_batch / n_channel;
  // 计算图像尺寸，即输入张量的元素总数除以批次数和通道数

  Tensor alpha = at::empty({n_channel}, input.options());
  // 创建一个形状为{n_channel}的空张量alpha，与input张量使用相同的选项
  Tensor beta = at::empty({n_channel}, input.options());
  // 创建一个形状为{n_channel}的空张量beta，与input张量使用相同的选项

  scalar_t* alpha_data = alpha.mutable_data_ptr<scalar_t>();
  // 获取alpha张量的可变数据指针
  scalar_t* beta_data = beta.data_ptr<scalar_t>();
  // 获取beta张量的数据指针

  batch_norm_cpu_collect_linear_and_constant_terms<scalar_t, scalar_t>(
     alpha_data, beta_data, n_channel, weight, bias,
     save_mean, save_invstd, running_mean, running_var, train, eps);
  // 调用batch_norm_cpu_collect_linear_and_constant_terms函数，收集线性和常数项的参数

  scalar_t* output_data = output.data_ptr<scalar_t>();
  // 获取输出张量output的数据指针
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  // 获取输入张量input的常量数据指针

  // Apply the linear terms to the input,
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  // 对输入应用线性项，
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)

  const int64_t loop_size = image_size - (image_size % Vec::size());
  // 计算循环的大小，确保是Vec::size()的整数倍

  at::parallel_for(0, n_batch * n_channel, 1, [&](int64_t begin, int64_t end) {
    // 并行循环，从begin到end，步长为1
    int64_t n = 0;
    int64_t c = 0;
    data_index_init(begin, n, n_batch, c, n_channel);
    // 初始化数据索引，确定n和c的初始值

    for (const auto i : c10::irange(begin, end)) {
      // 遍历索引范围内的元素
      const Vec alpha_vec(alpha_data[c]);
      // 获取alpha_data[c]并转换为Vec类型的向量alpha_vec
      const Vec beta_vec(beta_data[c]);
      // 获取beta_data[c]并转换为Vec类型的向量beta_vec
      int64_t offset = i * image_size;
      // 计算偏移量，基于当前索引i和图像尺寸
      int64_t d = 0;
      // 初始化循环变量d

      for (; d < loop_size; d += Vec::size()) {
        // 循环处理向量化大小的数据块
        Vec data_vec = Vec::loadu(input_data + offset + d);
        // 加载未对齐的数据向量
        Vec output_vec = data_vec * alpha_vec + beta_vec;
        // 计算输出向量
        output_vec.store(output_data + offset + d);
        // 存储输出向量到output_data中的指定位置
      }

      if (image_size - d > 0) {
        // 处理剩余的不足一个向量化块大小的数据
        Vec data_vec = Vec::loadu(input_data + offset + d, image_size - d);
        // 加载未对齐的数据向量
        Vec output_vec = data_vec * alpha_vec + beta_vec;
        // 计算输出向量
        output_vec.store(output_data + offset + d, image_size - d);
        // 存储输出向量到output_data中的指定位置
      }

      // move on to next index
      // 移动到下一个索引位置
      data_index_step(n, n_batch, c, n_channel);
      // 更新数据索引，移动到下一个(n, c)位置
    }
  });
// 结束前一个函数模板的定义

template <typename scalar_t>
// 如果标量类型为 at::opmath_type<scalar_t>，则启用此函数模板；否则，此函数不存在
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
// 执行 CPU 上 channels-last 排布的批量归一化计算，输出到指定的 Tensor
batch_norm_cpu_channels_last_impl(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {

  using Vec = Vectorized<scalar_t>;
  // 获取输入张量的批量数、通道数和图像尺寸
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  // 创建空的 alpha 和 beta 张量，与输入张量的选项相匹配
  Tensor alpha = at::empty({n_channel}, input.options());
  Tensor beta = at::empty({n_channel}, input.options());
  // 获取 alpha 和 beta 的数据指针
  scalar_t* alpha_data = alpha.mutable_data_ptr<scalar_t>();
  scalar_t* beta_data = beta.data_ptr<scalar_t>();

  // 调用 batch_norm_cpu_collect_linear_and_constant_terms 函数，
  // 收集线性和常数项到 alpha_data 和 beta_data 中
  batch_norm_cpu_collect_linear_and_constant_terms<scalar_t, scalar_t>(
      alpha_data, beta_data, n_channel, weight, bias,
      save_mean, save_invstd, running_mean, running_var, train, eps);

  // 获取输出和输入数据的指针
  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();

  // 对输入数据应用线性项，
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  const int64_t loop_size = n_channel - (n_channel % Vec::size());
  // 并行循环处理每个 batch 中的图像像素数据
  at::parallel_for(0, n_batch * image_size, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t offset = i * n_channel;
      int64_t d = 0;
      // 在通道维度上进行向量化，对于正常的批量归一化输入大小，
      // alpha/beta 应适合于 L1 缓存，否则需要考虑分块处理。
      for (; d < loop_size; d += Vec::size()) {
        Vec alpha_vec = Vec::loadu(alpha_data + d);
        Vec beta_vec = Vec::loadu(beta_data + d);
        Vec data_vec = Vec::loadu(input_data + offset + d);
        Vec output_vec = data_vec * alpha_vec + beta_vec;
        output_vec.store(output_data + offset + d);
      }
      // 处理剩余的通道数据（不能整除 Vec::size() 的情况）
      if (n_channel - d > 0) {
        Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
        Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
        Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
        Vec output_vec = data_vec * alpha_vec + beta_vec;
        output_vec.store(output_data + offset + d, n_channel - d);
      }
    }
  });
}

// 开始下一个函数模板的定义
template <typename scalar_t>
// 如果标量类型为 at::opmath_type<scalar_t>，则启用此函数模板；否则，此函数不存在
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
// 执行在连续内存布局的情况下收集批量归一化的统计信息
batch_norm_cpu_collect_stats_contiguous_impl(
  // 保持 acc_type 为 opmath_type，当 scalar_t 为 float 时使用 float 类型，而 acc_type 使用 double 类型用于 float。
  using accscalar_t = at::acc_type<scalar_t, false>;

  // 计算输入张量的批次数、通道数和图像大小
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  // 获取输入张量的常量数据指针
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();

  // 获取均值和方差总和张量的数据指针
  scalar_t* mean_data = mean.data_ptr<scalar_t>();
  scalar_t* var_sum_data = var_sum.data_ptr<scalar_t>();

  // 在 'channel' 维度上并行进行降维操作
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      // 计算每个通道 c 上的均值
      accscalar_t sum = 0;
      for (const auto n : c10::irange(n_batch)) {
        for (const auto i : c10::irange(image_size)) {
          auto offset = n * n_channel * image_size + c * image_size + i;
          sum += input_data[offset];
        }
      }
      scalar_t mean = sum / N;
      mean_data[c] = mean;

      // 计算每个通道 c 上的方差总和
      accscalar_t _var_sum = 0;
      for (const auto n : c10::irange(n_batch)) {
        for (const auto i : c10::irange(image_size)) {
          auto offset = n * n_channel * image_size + c * image_size + i;
          auto x = input_data[offset];
          _var_sum += (x - mean) * (x - mean);
        }
      }
      var_sum_data[c] = _var_sum;
    }
  });
}

template <typename scalar_t>
// 当标量类型为 at::opmath_type<scalar_t> 时，启用此函数用于计算批标准化的均值和方差
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
batch_norm_cpu_collect_stats_channels_last_impl(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {

  using Vec = Vectorized<scalar_t>;
  // 使用 opmath_type 定义的类型作为 accscalar_t，当 scalar_t 为 float 时，accscalar_t 为 double 类型
  using accscalar_t = at::acc_type<scalar_t, false>;
  // 获取输入张量中的通道数
  int64_t n_channel = input.size(1);
  // 计算输入张量的元素总数除以通道数，得到 N 的值
  int64_t N = input.numel() / n_channel;

  // 获取输入张量的数据指针，均值张量和方差和张量的数据指针
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  scalar_t* mean_data = mean.data_ptr<scalar_t>();
  scalar_t* var_sum_data = var_sum.data_ptr<scalar_t>();

  // 典型的垂直约简，从形状 {NHW, C} 到 {C}
  // 当 NHW > max_threads 时，应用两个路径的并行约简：
  // 第一路径：分配一个大小为 {max_threads, C} 的临时缓冲区，在 dim0 上并行，
  //    {NHW, C} => {max_threads, C}
  //
  // 第二路径：在临时缓冲区的 dim1 上并行，
  //    {max_threads, C} => {C}
  //
  // C 的正常大小应适合于 L1 缓存，否则考虑在 C 上进行分块。
  //
  int num_threads = at::get_num_threads();

  if (N > num_threads) {
    // 创建一个与输入张量相同选项的大小为 {num_threads, n_channel} 的零张量作为缓冲区
    Tensor buffer = at::zeros({num_threads, n_channel}, input.options());
    scalar_t* buffer_data = buffer.data_ptr<scalar_t>();

    // 计算每个输入的均值
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      int tid = at::get_thread_num();
      TORCH_CHECK(tid < num_threads,
                  "expect thread id smaller than ", num_threads, ", got thread id ", tid);
      scalar_t* buffer_ptr = buffer_data + tid * n_channel;
      for (const auto i : c10::irange(begin, end)) {
        const scalar_t* x_ptr = input_data + i * n_channel;
        // 使用 SIMD 向量化操作 map2 对每个通道进行求和
        vec::map2<scalar_t>(
            [](Vec x, Vec y) { return x + y; },
            buffer_ptr,
            x_ptr,
            buffer_ptr,
            n_channel);
      }
    });

    // 计算每个通道的均值，利用上一步的缓冲区
    at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
      for (const auto c : c10::irange(begin, end)) {
        accscalar_t sum = 0;
        for (const auto t : c10::irange(num_threads)) {
          sum += buffer_data[t * n_channel + c];
        }
        // 计算均值
        scalar_t mean = sum / N;
        mean_data[c] = mean;
      }
    });

    // 计算每个输入的方差，重用上述缓冲区
    buffer.zero_();
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      int tid = at::get_thread_num();
      TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
      scalar_t* buffer_ptr = buffer_data + tid * n_channel;
      for (const auto i : c10::irange(begin, end)) {
        const scalar_t* x_ptr = input_data + i * n_channel;
        // 使用 SIMD 向量化操作 map3 对每个通道进行方差计算
        vec::map3<scalar_t>(
            [](Vec x, Vec y, Vec mean) { return y + (x - mean) * (x - mean); },
            buffer_ptr,
            x_ptr,
            buffer_ptr,
            mean_data,
            n_channel);
      }
    });
    // 使用 ATen 的并行函数 parallel_for，对于给定的通道数 n_channel，使用 lambda 函数进行并行操作
    at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
      // 遍历每个通道 c，范围是 [begin, end)
      for (const auto c : c10::irange(begin, end)) {
        // 初始化累加器 _var_sum
        accscalar_t _var_sum = 0;
        // 遍历 num_threads 数量的线程
        for (const auto t : c10::irange(num_threads)) {
          // 累加每个线程对应通道 c 上的 buffer 数据
          _var_sum += buffer_data[t * n_channel + c];
        }
        // 将 _var_sum 的值保存到 var_sum_data 数组中对应的通道 c 处
        var_sum_data[c] = _var_sum;
      }
    });
  } else {
    // 当 NHW <= max_threads 时，进行垂直归约操作，从 {NHW, C} 形状到 {C} 形状
    // 使用两种方法，Method 1 和 Method 2

    // Method 1: 当 TILE_SIZE < C <= THRESHOLD 时，在 C 上进行并行操作
    //    {NHW, C} => {C}

    // Method 2: 当 C <= TILE_SIZE 或者 C > THRESHOLD 时，对 C 进行切片并在每个切片上进行矢量化
    //    C 被切片为：{TILE_SIZE, TILE_SIZE, ..., Remainder}
    // 在切片上并行操作，对每个切片进行矢量化垂直归约
    //    {NHW, TILE_SIZE} => {TILE_SIZE}

    // THRESHOLD 是经验性确定的最佳切片阈值
    // 当 C > THRESHOLD 时，C 足够大，切片和矢量化带来的好处超过同步开销
    // 当 C <= TILE_SIZE 且 NHW <= max_threads 时，问题规模足够小，此时使用单线程启动矢量化要优于不使用矢量化的 C 线程启动

    // 当 num_threads == 1 时，始终使用 Method 2，因为没有同步开销

    int64_t TILE_SIZE = 16;  // 设定切片大小为 16
    int64_t THRESHOLD = 2048;  // 设定阈值为 2048
    // 如果线程数为1，或者通道数小于等于TILE_SIZE或者大于THRESHOLD，则执行以下代码块
    if (num_threads == 1 || (n_channel <= TILE_SIZE || n_channel > THRESHOLD)) {
      // 计算每个输入的均值
      mean.zero_();
      // 使用并行方式对每个块进行迭代，块大小为TILE_SIZE，范围为[0, (n_channel + TILE_SIZE - 1) / TILE_SIZE)
      at::parallel_for(0, (n_channel + TILE_SIZE - 1) / TILE_SIZE, 1, [&](int64_t tile_idx_begin, int64_t tile_idx_end) {
        // 对于每个块的索引tile_idx，进行以下循环
        for (int64_t tile_idx = tile_idx_begin; tile_idx < tile_idx_end; tile_idx++) {
          // 计算当前块的起始和结束索引
          int64_t jj_begin = tile_idx * TILE_SIZE;
          int64_t jj_end = std::min(jj_begin + TILE_SIZE, n_channel);
          // 指向均值数据的指针
          scalar_t* mean_ptr = mean_data + jj_begin;
          // 对于每个输入i，指向输入数据的指针
          for (const auto i : c10::irange(N)) {
            const scalar_t* x_ptr = input_data + (i * n_channel + jj_begin);
            // 对当前块进行向量化操作，计算累加和
            vec::map2<scalar_t>(
              [](Vec x, Vec y) { return x + y; },
              mean_ptr,
              x_ptr,
              mean_ptr,
              jj_end - jj_begin);
          }
          // 对当前块的均值进行归一化操作
          vec::map<scalar_t>(
            [N](Vec x) { return x / Vec(N); },
            mean_ptr,
            mean_ptr,
            jj_end - jj_begin);
        }
      });

      // 计算每个输入的方差
      var_sum.zero_();
      // 使用并行方式对每个块进行迭代，块大小为TILE_SIZE，范围为[0, (n_channel + TILE_SIZE - 1) / TILE_SIZE)
      at::parallel_for(0, (n_channel + TILE_SIZE - 1) / TILE_SIZE, 1, [&](int64_t tile_idx_begin, int64_t tile_idx_end) {
        // 对于每个块的索引tile_idx，进行以下循环
        for (int64_t tile_idx = tile_idx_begin; tile_idx < tile_idx_end; tile_idx++) {
          // 计算当前块的起始和结束索引
          int64_t jj_begin = tile_idx * TILE_SIZE;
          int64_t jj_end = std::min(jj_begin + TILE_SIZE, n_channel);
          // 指向方差和数据的指针
          scalar_t* var_sum_ptr = var_sum_data + jj_begin;
          // 指向均值数据的指针
          scalar_t* mean_ptr = mean_data + jj_begin;
          // 对于每个输入i，指向输入数据的指针
          for (const auto i : c10::irange(N)) {
            const scalar_t* x_ptr = input_data + (i * n_channel + jj_begin);
            // 对当前块进行向量化操作，计算方差和
            vec::map3<scalar_t>(
              [](Vec x, Vec y, Vec mean) { return y + (x - mean) * (x - mean); },
              var_sum_ptr,
              x_ptr,
              var_sum_ptr,
              mean_ptr,
              jj_end - jj_begin);
          }
        }
      });
    }
    // 方法1：在C上进行并行操作，垂直归约
    else {
      // 计算每个输入的均值
      at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
        // 对于范围[begin, end)内的每个通道c进行以下循环
        for (const auto c : c10::irange(begin, end)) {
          // 初始化累加和为0
          accscalar_t sum = 0;
          // 对于每个时间步t进行以下循环
          for (const auto t : c10::irange(N)) {
            // 累加当前通道c上的输入数据
            sum += input_data[t * n_channel + c];
          }
          // 计算均值并存储到mean_data中
          scalar_t mean = sum / N;
          mean_data[c] = mean;
        }
      });

      // 计算每个输入的方差
      at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
        // 对于范围[begin, end)内的每个通道c进行以下循环
        for (const auto c : c10::irange(begin, end)) {
          // 初始化方差和为0
          accscalar_t _var_sum = 0;
          // 对于每个时间步t进行以下循环
          for (const auto t : c10::irange(N)) {
            // 计算当前输入数据与均值之差的平方，累加到_var_sum中
            _var_sum += (input_data[t * n_channel + c] - mean_data[c]) * (input_data[t * n_channel + c] - mean_data[c]);
          }
          // 存储方差和到var_sum_data中
          var_sum_data[c] = _var_sum;
        }
      });
    }
  }
}
// 结束模板函数的定义

template <typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
batch_norm_cpu_backward_contiguous_impl(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {

  using Vec = Vectorized<scalar_t>;
  // 使用 opmath_type 获取 scalar_t 的数学类型，即 float 时返回 float
  // 而 acc_type 在 scalar_t==float 时返回 double
  using accscalar_t = at::acc_type<scalar_t, false>;
  // 计算 batch 的数量
  int64_t n_batch = input.size(0);
  // 计算 channel 的数量
  int64_t n_channel = input.size(1);
  // 计算每张图像的像素总数
  int64_t image_size = input.numel() / n_batch / n_channel;
  // 计算输入张量的总元素数，除以通道数得到 N
  int64_t N = input.numel() / n_channel;

  // 获取梯度输出的数据指针
  const scalar_t* grad_output_data = grad_output.const_data_ptr<scalar_t>();
  // 获取输入数据的常量数据指针
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();

  // 获取梯度输入数据的可变数据指针，如果未定义 grad_input 则为 nullptr
  scalar_t* grad_input_data = grad_input.defined() ? grad_input.mutable_data_ptr<scalar_t>() : nullptr;
  // 获取梯度权重数据的可变数据指针，如果未定义 grad_weight 则为 nullptr
  scalar_t* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<scalar_t>() : nullptr;
  // 获取梯度偏置数据的可变数据指针，如果未定义 grad_bias 则为 nullptr
  scalar_t* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<scalar_t>() : nullptr;
  // 检查 grad_input 是否为 nullptr
  const bool grad_input_null = grad_input_data == nullptr;
  // 检查 grad_weight 是否为 nullptr
  const bool grad_weight_null = grad_weight_data == nullptr;
  // 检查 grad_bias 是否为 nullptr
  const bool grad_bias_null = grad_bias_data == nullptr;

  // 条件性访问器，用于访问 weight 的一维数据
  auto weight_a = conditional_accessor_1d<const scalar_t>(weight);
  // 条件性访问器，用于访问 save_mean 的一维数据
  auto save_mean_a = conditional_accessor_1d<const scalar_t>(save_mean);
  // 条件性访问器，用于访问 save_invstd 的一维数据
  auto save_invstd_a = conditional_accessor_1d<const scalar_t>(save_invstd);
  // 条件性访问器，用于访问 running_mean 的一维数据
  auto running_mean_a = conditional_accessor_1d<const scalar_t>(running_mean);
  // 条件性访问器，用于访问 running_var 的一维数据
  auto running_var_a = conditional_accessor_1d<const scalar_t>(running_var);

  // 在 'channel' 维度上并行减少
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    // 并行处理 'channel' 维度的内容
    }
  });
}

template <typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
batch_norm_cpu_backward_channels_last_impl(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {

// 函数签名，声明一个函数，接受布尔类型train和双精度浮点数eps作为参数


  using Vec = Vectorized<scalar_t>;
  // keep acc_type as opmath_type will use float type when scalar_t==float
  // while acc_type uses double for float.
  using accscalar_t = at::acc_type<scalar_t, false>;

// 定义类型别名Vec为Vectorized<scalar_t>，accscalar_t为at::acc_type<scalar_t, false>
// 该注释解释了为什么保持acc_type作为opmath_type，以及在scalar_t为float时使用float类型而在acc_type中使用double类型。


  int64_t n_channel = input.size(1);
  int64_t N = input.numel() / n_channel;

// 获取输入张量input的第一个维度大小作为n_channel，计算总元素数除以通道数得到N。


  const scalar_t* grad_output_data = grad_output.const_data_ptr<scalar_t>();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();

// 获取grad_output和input的数据指针，分别存储在grad_output_data和input_data中。


  scalar_t* grad_input_data = grad_input.defined() ? grad_input.mutable_data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<scalar_t>() : nullptr;
  scalar_t* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<scalar_t>() : nullptr;

// 如果grad_input、grad_weight、grad_bias已定义，则获取它们的可变数据指针，否则为nullptr。


  const scalar_t* save_mean_data = conditional_data_ptr<const scalar_t>(save_mean);
  scalar_t* save_invstd_data = conditional_data_ptr<scalar_t>(save_invstd);
  const scalar_t* running_mean_data = conditional_data_ptr<const scalar_t>(running_mean);
  const scalar_t* running_var_data = conditional_data_ptr<const scalar_t>(running_var);

// 根据条件获取save_mean、save_invstd、running_mean、running_var的数据指针，并存储在相应的指针变量中。


  Tensor weight_ = weight.defined() ? weight : at::ones({n_channel}, input.options());
  const scalar_t* weight_data = weight_.const_data_ptr<scalar_t>();

// 如果weight已定义，则使用它；否则创建一个全为1的张量weight_，并获取其数据指针存储在weight_data中。


  const scalar_t* mean_ptr = nullptr;
  scalar_t* invstd_ptr = nullptr;
  Tensor invstd = at::empty({0}, input.options());
  if (train) {
    mean_ptr = save_mean_data;
    invstd_ptr = save_invstd_data;
  } else {
    mean_ptr = running_mean_data;

    invstd.resize_({n_channel});
    invstd_ptr = invstd.data_ptr<scalar_t>();
    for (const auto c : c10::irange(n_channel)) {
      invstd_ptr[c] = 1 / std::sqrt(running_var_data[c] + eps);
    }
  }

// 根据train的值选择mean_ptr和invstd_ptr的来源，如果train为true则使用save_mean_data和save_invstd_data，否则使用running_mean_data和计算得到的invstd数据。


  // Typical vertical reduce from shape of {NHW, C} to {C}.
  // Apply two path parallel reduction:
  // First path: allocate an immediate buffer of size {2, max_threads, C}, parallel along dim0,
  //    sum = buffer[0], dotp = buffer[2]
  //
  // Second path: parallel along dim1 of the immediate buffer.
  //
  int num_threads = at::get_num_threads();
  Tensor buffer = at::zeros({2, num_threads, n_channel}, input.options());
  scalar_t* sum_data = buffer.data_ptr<scalar_t>();
  scalar_t* dotp_data = sum_data + num_threads * n_channel;

// 描述典型的从形状{NHW, C}到{C}的垂直缩减过程，同时应用两种路径的并行缩减。分配一个大小为{2, max_threads, C}的立即缓冲区，并行于dim0。


  // compute sum and dotp per feature plain,
  // fuse into a single loop to reuse grad_output in L1.
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    scalar_t* sum_ptr = sum_data + tid * n_channel;
    scalar_t* dotp_ptr = dotp_data + tid * n_channel;

// 并行计算每个特征平面的sum和dotp，融合成一个单独的循环以在L1中重用grad_output。
    // 使用并行循环遍历范围 [begin, end) 中的索引 i
    for (const auto i : c10::irange(begin, end)) {
      // 计算当前位置在输入数据中的偏移量
      const scalar_t* x_ptr = input_data + i * n_channel;
      // 计算当前位置在梯度输出数据中的偏移量
      const scalar_t* dy_ptr = grad_output_data + i * n_channel;

      // 对长度为 n_channel 的向量进行逐元素映射求和
      vec::map2<scalar_t>(
          [](Vec sum, Vec dy) { return sum + dy; },
          sum_ptr,
          sum_ptr,
          dy_ptr,
          n_channel);

      // 对长度为 n_channel 的向量进行逐元素映射计算点积
      vec::map4<scalar_t>(
          [](Vec dotp, Vec x, Vec mean, Vec dy) { return dotp + (x - mean) * dy; },
          dotp_ptr,
          dotp_ptr,
          x_ptr,
          mean_ptr,
          dy_ptr,
          n_channel);
    }
  });

  // 使用并行循环遍历范围 [0, n_channel) 中的索引 c
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      // 将每列的和与点积结果存储在临时缓冲区的第一个位置，避免额外的缓冲区分配
      accscalar_t _sum = 0;
      // 计算每个线程对应列 c 的 sum_data 的和
      for (const auto t : c10::irange(num_threads)) {
        _sum += sum_data[t * n_channel + c];
      }
      // 将计算得到的总和存储在 sum_data 的第 c 列
      sum_data[/* 0 * n_channel + */c] = _sum;

      accscalar_t _dotp = 0;
      // 计算每个线程对应列 c 的 dotp_data 的和
      for (const auto t : c10::irange(num_threads)) {
        _dotp += dotp_data[t * n_channel + c];
      }
      // 将计算得到的点积结果存储在 dotp_data 的第 c 列
      dotp_data[/* 0 * n_channel + */c] = _dotp;
    }
  });

  // 计算 grad_input
  // 计算循环的大小，使其是 Vec::size() 的整数倍
  const int64_t loop_size = n_channel - (n_channel % Vec::size());
  if (grad_input.defined()) {
    // 使用并行方式对范围 [0, N) 进行迭代，每个线程处理一部分数据
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      // 遍历每个数据点的索引范围 [begin, end)
      for (const auto i : c10::irange(begin, end)) {
        // 获取当前数据点在梯度输入、输入、梯度输出中的起始位置指针
        scalar_t* dx_ptr = grad_input_data + i * n_channel;
        const scalar_t* x_ptr = input_data + i * n_channel;
        const scalar_t* dy_ptr = grad_output_data + i * n_channel;
        
        // 如果是训练模式
        if (train) {
          int64_t d = 0;
          // 使用向量化方式处理循环，每次处理 Vec::size() 大小的数据
          for (; d < loop_size; d += Vec::size()) {
            // 加载输入、均值、点积、反标准差的向量数据
            Vec x = Vec::loadu(x_ptr + d);
            Vec mean = Vec::loadu(mean_ptr + d);
            Vec dotp = Vec::loadu(dotp_data + d);
            Vec invstd = Vec::loadu(invstd_ptr + d);
            
            // 计算归一化系数 k
            Vec k = dotp * invstd * invstd / Vec(N);
            // 计算输入的梯度 dx
            Vec dx = (x - mean) * k;
            // 加载梯度输出的向量数据 dy
            Vec dy = Vec::loadu(dy_ptr + d);
            // 加载均值梯度的向量数据 grad_mean
            Vec grad_mean = Vec::loadu(sum_data + d) / Vec(N);
            // 加载权重向量数据 w
            Vec w = Vec::loadu(weight_data + d);
            // 计算最终的梯度 dx
            dx = (dy - grad_mean - dx) * invstd * w;
            // 将结果存储到梯度输入的指定位置
            dx.store(dx_ptr + d);
          }
          
          // 处理剩余的数据，不足一个向量大小的部分
          if (n_channel - d > 0) {
            Vec x = Vec::loadu(x_ptr + d, n_channel - d);
            Vec mean = Vec::loadu(mean_ptr + d, n_channel - d);
            Vec dotp = Vec::loadu(dotp_data + d, n_channel - d);
            Vec invstd = Vec::loadu(invstd_ptr + d, n_channel - d);
            Vec k = dotp * invstd * invstd / Vec(N);
            Vec dx = (x - mean) * k;
            Vec dy = Vec::loadu(dy_ptr + d, n_channel - d);
            Vec grad_mean = Vec::loadu(sum_data + d, n_channel - d) / Vec(N);
            Vec w = Vec::loadu(weight_data + d, n_channel - d);
            dx = (dy - grad_mean - dx) * invstd * w;
            dx.store(dx_ptr + d, n_channel - d);
          }
        } else { // 评估模式
          int64_t d = 0;
          // 使用向量化方式处理循环，每次处理 Vec::size() 大小的数据
          for (; d < loop_size; d += Vec::size()) {
            // 加载梯度输出的向量数据 dy
            Vec dy = Vec::loadu(dy_ptr + d);
            // 加载反标准差的向量数据 invstd
            Vec invstd = Vec::loadu(invstd_ptr + d);
            // 加载权重向量数据 w
            Vec w = Vec::loadu(weight_data + d);
            // 计算最终的梯度 dx
            Vec dx = dy * invstd * w;
            // 将结果存储到梯度输入的指定位置
            dx.store(dx_ptr + d);
          }
          
          // 处理剩余的数据，不足一个向量大小的部分
          if (n_channel - d > 0) {
            Vec dy = Vec::loadu(dy_ptr + d, n_channel - d);
            Vec invstd = Vec::loadu(invstd_ptr + d, n_channel - d);
            Vec w = Vec::loadu(weight_data + d, n_channel - d);
            Vec dx = dy * invstd * w;
            dx.store(dx_ptr + d, n_channel - d);
          }
        }
      }
    });
  }

  // 如果定义了梯度权重，则计算梯度权重
  if (grad_weight.defined()) {
    // 使用向量化操作计算梯度权重，dotp * invstd
    vec::map2<scalar_t>(
        [](Vec dotp, Vec invstd) { return dotp * invstd; },
        grad_weight_data,
        dotp_data,
        invstd_ptr,
        n_channel);
  }

  // 如果定义了梯度偏置，则计算梯度偏置，即简单地拷贝求和数据到梯度偏置
  if (grad_bias.defined()) {
    vec::map<scalar_t>(
        [](Vec sum) { return sum; },
        grad_bias_data,
        sum_data,
        n_channel);
  }
/// bfloat16/Half kernels
template<typename scalar_t>
/// 如果 scalar_t 不是 opmath_type<scalar_t> 的别名，则启用该模板函数
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
/// 实现在 CPU 上对连续存储的输入进行批量归一化操作，生成输出张量
batch_norm_cpu_contiguous_impl(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
  /// 定义 opmath_t 作为 scalar_t 的 opmath_type 类型别名
  using opmath_t = at::opmath_type<scalar_t>;
  /// 定义 bVec 和 fVec 作为 scalar_t 和 opmath_t 的向量化类型
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<opmath_t>;
  /// 获取输入张量的批次数、通道数和图像尺寸
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  // use float as acc type
  /// 创建 alpha 和 beta 张量，作为浮点数类型的累加器
  Tensor alpha = at::empty({n_channel}, input.options().dtype(kFloat));
  Tensor beta = at::empty({n_channel}, input.options().dtype(kFloat));
  /// 获取 alpha 和 beta 张量的数据指针
  opmath_t* alpha_data = alpha.mutable_data_ptr<opmath_t>();
  opmath_t* beta_data = beta.data_ptr<opmath_t>();

  /// 检查是否存在混合类型输入
  const bool mixed_type = is_mixed_type(input, weight, bias, save_mean, save_invstd, running_mean, running_var);
  /// 根据混合类型调用不同类型的 batch_norm_cpu_collect_linear_and_constant_terms 函数
  if (mixed_type) {
    batch_norm_cpu_collect_linear_and_constant_terms<opmath_t, opmath_t>(
        alpha_data, beta_data, n_channel, weight, bias,
        save_mean, save_invstd, running_mean, running_var, train, eps);
  } else {
    batch_norm_cpu_collect_linear_and_constant_terms<scalar_t, opmath_t>(
        alpha_data, beta_data, n_channel, weight, bias,
        save_mean, save_invstd, running_mean, running_var, train, eps);
  }

  /// 获取输出张量和输入张量的数据指针
  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();

  /// 计算向量化循环的大小
  const int64_t loop_size = image_size - (image_size % bVec::size());
  /// 并行处理每个批次和通道的元素
  at::parallel_for(0, n_batch * n_channel, 1, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t c = 0;
    /// 初始化数据索引
    data_index_init(begin, n, n_batch, c, n_channel);

    for (const auto i : c10::irange(begin, end)) {
      /// 获取当前输入和输出指针
      const scalar_t* input_ptr = input_data + i * image_size;
      scalar_t* output_ptr = output_data + i * image_size;
      /// 获取当前通道的 alpha 和 beta 值
      const opmath_t alpha_val = alpha_data[c];
      const opmath_t beta_val = beta_data[c];
      /// 将 alpha 和 beta 值转换为向量化形式
      const fVec alpha_fvec(alpha_val);
      const fVec beta_fvec(beta_val);
      int64_t d = 0;
      /// 执行向量化计算循环
      for (; d < loop_size; d += bVec::size()) {
        bVec data_bvec = bVec::loadu(input_ptr + d);
        auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);

        fVec out_fvec0 = data_fvec0 * alpha_fvec + beta_fvec;
        fVec out_fvec1 = data_fvec1 * alpha_fvec + beta_fvec;
        bVec out_bvec = convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        out_bvec.store(output_ptr + d);
      }
      /// 处理剩余的单个元素
      for (; d < image_size; d++) {
        output_ptr[d] = scalar_t(opmath_t(input_ptr[d]) * alpha_val + beta_val);
      }
      /// 移动到下一个索引
      data_index_step(n, n_batch, c, n_channel);
    }
  });
}
# 批量归一化 CPU 实现函数，处理通道最后维度的张量
batch_norm_cpu_channels_last_impl(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
  # 定义 opmath_t 为标量类型的操作数
  using opmath_t = at::opmath_type<scalar_t>;
  # 定义 bVec 为 scalar_t 类型的向量化类型
  using bVec = Vectorized<scalar_t>;
  # 定义 fVec 为 opmath_t 类型的向量化类型
  using fVec = Vectorized<opmath_t>;
  # 计算输入张量的批次数
  int64_t n_batch = input.size(0);
  # 计算输入张量的通道数
  int64_t n_channel = input.size(1);
  # 计算输入张量的图像大小
  int64_t image_size = input.numel() / n_batch / n_channel;

  # 创建空的 alpha 张量，形状为 (n_channel,)，使用与输入相同的浮点数类型
  Tensor alpha = at::empty({n_channel}, input.options().dtype(kFloat));
  # 创建空的 beta 张量，形状为 (n_channel,)，使用与输入相同的浮点数类型
  Tensor beta = at::empty({n_channel}, input.options().dtype(kFloat));
  # 获取 alpha 张量的可变数据指针，类型为 opmath_t
  opmath_t* alpha_data = alpha.mutable_data_ptr<opmath_t>();
  # 获取 beta 张量的数据指针，类型为 opmath_t
  opmath_t* beta_data = beta.data_ptr<opmath_t>();

  # 检查输入和权重等张量是否混合类型
  const bool mixed_type = is_mixed_type(input, weight, bias, save_mean, save_invstd, running_mean, running_var);
  # 根据混合类型选择适当的模板参数化批量归一化 CPU 收集线性和常数项
  if (mixed_type) {
    batch_norm_cpu_collect_linear_and_constant_terms<opmath_t, opmath_t>(
        alpha_data, beta_data, n_channel, weight, bias,
        save_mean, save_invstd, running_mean, running_var, train, eps);
  } else {
    batch_norm_cpu_collect_linear_and_constant_terms<scalar_t, opmath_t>(
        alpha_data, beta_data, n_channel, weight, bias,
        save_mean, save_invstd, running_mean, running_var, train, eps);
  }

  # 获取输出张量的数据指针，类型为 scalar_t
  scalar_t* output_data = output.data_ptr<scalar_t>();
  # 获取输入张量的常量数据指针，类型为 scalar_t
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();

  # 计算循环大小，以便使用向量化类型进行并行计算
  const int64_t loop_size = n_channel - (n_channel % bVec::size());
  # 使用并行化的方式对每个批次和图像大小范围内的数据进行操作
  at::parallel_for(0, n_batch * image_size, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      # 计算当前批次和图像中的输入数据指针
      const scalar_t* input_ptr = input_data + i * n_channel;
      # 计算当前批次和图像中的输出数据指针
      scalar_t* output_ptr = output_data + i * n_channel;
      int64_t d = 0;
      # 循环处理向量化大小范围内的数据
      for (; d < loop_size; d += bVec::size()) {
        # 加载 alpha 数据到 fVec 向量
        fVec alpha_fvec0 = fVec::loadu(alpha_data + d);
        fVec alpha_fvec1 = fVec::loadu(alpha_data + d + fVec::size());
        # 加载 beta 数据到 fVec 向量
        fVec beta_fvec0 = fVec::loadu(beta_data + d);
        fVec beta_fvec1 = fVec::loadu(beta_data + d + fVec::size());
        # 加载输入数据到 bVec 向量
        bVec data_bvec = bVec::loadu(input_ptr + d);
        # 将 bVec 向量转换为两个 opmath_t 类型的 fVec 向量
        auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);

        # 计算输出 fVec 向量
        fVec out_fvec0 = data_fvec0 * alpha_fvec0 + beta_fvec0;
        fVec out_fvec1 = data_fvec1 * alpha_fvec1 + beta_fvec1;
        # 将 fVec 向量转换为 bVec 向量，并存储到输出指针中
        bVec out_bvec = convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        out_bvec.store(output_ptr + d);
      }
      # 处理剩余的未能整除向量化大小的数据
      for (; d < n_channel; d++) {
        output_ptr[d] = scalar_t(opmath_t(input_ptr[d]) * alpha_data[d] + beta_data[d]);
      }
    }
  });
}
    Tensor& mean, Tensor& var_sum, const Tensor& input) {
  using opmath_t = at::opmath_type<scalar_t>;  # 定义操作数的数学类型，根据输入张量的标量类型确定
  using bVec = Vectorized<scalar_t>;  # 定义矢量化类型 bVec，基于输入张量的标量类型
  using fVec = Vectorized<opmath_t>;  # 定义数学操作类型的矢量化类型 fVec

  int64_t n_batch = input.size(0);  # 计算输入张量的批次大小
  int64_t n_channel = input.size(1);  # 计算输入张量的通道数
  int64_t image_size = input.numel() / n_batch / n_channel;  # 计算每个图像的像素数
  int64_t N = input.numel() / n_channel;  # 计算总像素数

  const scalar_t* input_data = input.const_data_ptr<scalar_t>();  # 获取输入张量的常量数据指针
  param_t* mean_data = mean.data_ptr<param_t>();  # 获取均值张量的数据指针
  param_t* var_sum_data = var_sum.data_ptr<param_t>();  # 获取方差总和张量的数据指针

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {  # 并行循环，对每个通道进行处理
    for (const auto c : c10::irange(begin, end)) {  # 遍历每个通道
      opmath_t sum_val = opmath_t(0);  # 初始化总和值为零
      fVec sum_fvec = fVec(opmath_t(0));  # 初始化总和的矢量化版本为零
      for (int64_t n = 0; n < n_batch; n++) {  # 遍历每个批次
        const scalar_t* input_ptr = input_data + n * n_channel * image_size + c * image_size;  # 获取当前通道和批次的输入数据指针
        int64_t d = 0;
        for (; d < image_size - (image_size % bVec::size()); d += bVec::size()) {  # 使用矢量化处理输入数据
          bVec data_bvec = bVec::loadu(input_ptr + d);  # 加载未对齐的输入数据到矢量化数据结构
          auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);  # 转换为浮点数矢量
          sum_fvec += data_fvec0;  # 更新矢量化总和
          sum_fvec += data_fvec1;  # 更新矢量化总和
        }
        for (; d < image_size; d++) {  # 处理剩余的未矢量化数据
          sum_val += opmath_t(input_ptr[d]);  # 更新总和值
        }
      }
      // TODO: use fast version  # 待优化：使用快速版本
      sum_val += vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, sum_fvec, fVec::size());  # 对矢量化总和应用快速规约

      opmath_t mean_val = sum_val / N;  # 计算均值
      mean_data[c] = param_t(mean_val);  # 将均值存储到均值张量中

      opmath_t var_val = opmath_t(0);  # 初始化方差值为零
      fVec var_fvec = fVec(opmath_t(0));  # 初始化方差的矢量化版本为零
      fVec mean_fvec = fVec(mean_val);  # 将均值转换为矢量化类型
      for (int64_t n = 0; n < n_batch; n++) {  # 遍历每个批次
        const scalar_t* input_ptr = input_data + n * n_channel * image_size + c * image_size;  # 获取当前通道和批次的输入数据指针
        int64_t d = 0;
        for (; d < image_size - (image_size % bVec::size()); d += bVec::size()) {  # 使用矢量化处理输入数据
          bVec data_bvec = bVec::loadu(input_ptr + d);  # 加载未对齐的输入数据到矢量化数据结构
          auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);  # 转换为浮点数矢量
          var_fvec += (data_fvec0 - mean_fvec) * (data_fvec0 - mean_fvec);  # 更新方差的矢量化版本
          var_fvec += (data_fvec1 - mean_fvec) * (data_fvec1 - mean_fvec);  # 更新方差的矢量化版本
        }
        for (; d < image_size; d++) {  # 处理剩余的未矢量化数据
          opmath_t data_val = input_ptr[d];
          var_val += (data_val - mean_val) * (data_val - mean_val);  # 更新方差值
        }
      }
      // TODO: use fast version  # 待优化：使用快速版本
      var_val += vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, var_fvec, fVec::size());  # 对矢量化方差应用快速规约

      var_sum_data[c] = param_t(var_val);  # 将总方差存储到方差总和张量中
    }
  });
}

template <typename scalar_t>
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
batch_norm_cpu_collect_stats_contiguous_impl(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {
  // 检查是否为混合类型
  const bool mixed_type = is_mixed_type(input, mean, var_sum);
  // 如果是混合类型，调用具体实现函数，使用 at::opmath_type<scalar_t> 作为参数类型
  if (mixed_type) {
    batch_norm_cpu_collect_stats_contiguous_internal<scalar_t, at::opmath_type<scalar_t>>(mean, var_sum, input);
  } else {
    // 否则，调用具体实现函数，使用 scalar_t 作为参数类型
    batch_norm_cpu_collect_stats_contiguous_internal<scalar_t, scalar_t>(mean, var_sum, input);
  }
}

template <typename scalar_t, typename param_t>
inline void batch_norm_cpu_collect_stats_channels_last_internal(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {
  // 定义使用的数学操作类型
  using opmath_t = at::opmath_type<scalar_t>;
  // 定义向量化类型
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<opmath_t>;
  // 获取输入张量的通道数和元素数
  int64_t n_channel = input.size(1);
  int64_t N = input.numel() / n_channel;

  // 获取输入数据指针和均值、方差数据指针
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  param_t* mean_data = mean.data_ptr<param_t>();
  param_t* var_sum_data = var_sum.data_ptr<param_t>();

  // 获取当前可用线程数
  int num_threads = at::get_num_threads();
  // 创建缓冲区张量，用于存储部分计算结果
  Tensor buffer = at::zeros({num_threads, n_channel}, input.options().dtype(kFloat));
  opmath_t* buffer_data = buffer.data_ptr<opmath_t>();

  // 并行处理每个元素
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    // 检查线程 ID 是否小于线程总数
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    // 计算当前线程在缓冲区中的起始位置
    opmath_t* buffer_ptr = buffer_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* input_ptr = input_data + i * n_channel;
      int64_t d = 0;
      // 对每个通道进行向量化计算
      for (; d < n_channel - (n_channel % bVec::size()); d += bVec::size()) {
        bVec data_bvec = bVec::loadu(input_ptr + d);
        auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
        fVec sum_fvec0 = fVec::loadu(buffer_ptr + d) + data_fvec0;
        fVec sum_fvec1 = fVec::loadu(buffer_ptr + d + fVec::size()) + data_fvec1;
        sum_fvec0.store(buffer_ptr + d);
        sum_fvec1.store(buffer_ptr + d + fVec::size());
      }
      // 处理未对齐的剩余通道
      for (; d < n_channel; d++) {
        buffer_ptr[d] += input_ptr[d];
      }
    }
  });

  // 计算每个通道的均值
  for (const auto c : c10::irange(n_channel)) {
    opmath_t sum = 0;
    for (const auto t : c10::irange(num_threads)) {
      sum += buffer_data[t * n_channel + c];
    }
    mean_data[c] = param_t(sum / N);
  }

  // 清空缓冲区
  buffer.zero_();
  // 再次并行处理每个元素，重复上述操作
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    opmath_t* buffer_ptr = buffer_data + tid * n_channel;
    // 遍历指定范围内的整数序列，范围为 [begin, end)
    for (const auto i : c10::irange(begin, end)) {
      // 计算当前输入数据的指针位置
      const scalar_t* input_ptr = input_data + i * n_channel;
      // 初始化变量 d 为 0，用于遍历通道数，每次递增 bVec::size() 个元素
      int64_t d = 0;
      // 使用 SIMD 向量化操作处理数据，每次处理 bVec::size() 个元素
      for (; d < n_channel - (n_channel % bVec::size()); d += bVec::size()) {
        // 加载输入数据到 SIMD 向量 bVec
        bVec data_bvec = bVec::loadu(input_ptr + d);
        // 将加载的数据转换为 float 类型向量
        auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
        // 加载均值数据到 float 向量
        auto [mean_fvec0, mean_fvec1] = load2f(mean_data + d);
        // 加载缓冲区中的方差向量
        fVec var_fvec0 = fVec::loadu(buffer_ptr + d);
        fVec var_fvec1 = fVec::loadu(buffer_ptr + d + fVec::size());
        // 更新方差向量的值，使用 SIMD 操作
        var_fvec0 += (data_fvec0 - mean_fvec0) * (data_fvec0 - mean_fvec0);
        var_fvec1 += (data_fvec1 - mean_fvec1) * (data_fvec1 - mean_fvec1);
        // 存储更新后的方差向量回缓冲区
        var_fvec0.store(buffer_ptr + d);
        var_fvec1.store(buffer_ptr + d + fVec::size());
      }
      // 处理剩余不足一个 SIMD 向量大小的数据元素
      for (; d < n_channel; d++) {
        // 加载单个数据元素到 opmath_t 类型
        opmath_t data_val = opmath_t(input_ptr[d]);
        // 加载对应位置的均值数据
        opmath_t mean_val = opmath_t(mean_data[d]);
        // 更新缓冲区中的方差数据
        buffer_ptr[d] += (data_val - mean_val) * (data_val - mean_val);
      }
    }
  });

  // 计算每个通道上的方差总和
  for (const auto c : c10::irange(n_channel)) {
    // 初始化 _var_sum 为 0，用于累加每个线程的缓冲区数据
    opmath_t _var_sum = 0;
    // 遍历所有线程的缓冲区数据，累加同一通道上的数据
    for (const auto t : c10::irange(num_threads)) {
      _var_sum += buffer_data[t * n_channel + c];
    }
    // 将累加的方差总和存储到 var_sum_data 数组中
    var_sum_data[c] = param_t(_var_sum);
  }
}

template <typename scalar_t>
// 如果 scalar_t 不是 opmath_type<scalar_t> 类型，则启用该函数，返回类型为 void
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
// 执行批量归一化（BN）的通道优先统计信息收集实现
batch_norm_cpu_collect_stats_channels_last_impl(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {
  // 检查是否存在混合类型的输入
  const bool mixed_type = is_mixed_type(input, mean, var_sum);
  // 如果存在混合类型
  if (mixed_type) {
    // 调用内部函数，使用 scalar_t 和 opmath_type<scalar_t> 类型进行统计信息收集
    batch_norm_cpu_collect_stats_channels_last_internal<scalar_t, at::opmath_type<scalar_t>>(mean, var_sum, input);
  } else {
    // 调用内部函数，使用 scalar_t 类型进行统计信息收集
    batch_norm_cpu_collect_stats_channels_last_internal<scalar_t, scalar_t>(mean, var_sum, input);
  }
}

template <typename scalar_t, typename param_t>
// 执行连续数据的 CPU 后向传播的批量归一化内部实现
void batch_norm_cpu_backward_contiguous_internal(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {
  // 定义 opmath_t 为 scalar_t 的 opmath_type 类型
  using opmath_t = at::opmath_type<scalar_t>;
  // 定义 bVec 和 fVec 为 scalar_t 和 opmath_t 的向量化类型
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<opmath_t>;
  // 计算批次数、通道数和图像尺寸
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;
  int64_t N = input.numel() / n_channel;

  // 获取梯度输出和输入数据的指针
  const scalar_t* grad_output_data = grad_output.const_data_ptr<scalar_t>();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();

  // 获取梯度输入、梯度权重和梯度偏置的可变数据指针，如果未定义则为 nullptr
  scalar_t* grad_input_data = grad_input.defined() ? grad_input.mutable_data_ptr<scalar_t>() : nullptr;
  param_t* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<param_t>() : nullptr;
  param_t* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<param_t>() : nullptr;

  // 检查梯度输入、梯度权重和梯度偏置是否为空
  const bool grad_input_null = grad_input_data == nullptr;
  const bool grad_weight_null = grad_weight_data == nullptr;
  const bool grad_bias_null = grad_bias_data == nullptr;

  // 定义对权重、保存均值、保存标准差、运行均值和运行方差的条件访问器
  auto weight_a = conditional_accessor_1d<const param_t>(weight);
  auto save_mean_a = conditional_accessor_1d<const param_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<const param_t>(save_invstd);
  auto running_mean_a = conditional_accessor_1d<const param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<const param_t>(running_var);

  // 并行处理通道维度上的归约操作
  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    // 并行处理代码段
  });
}

template <typename scalar_t>
// 如果 scalar_t 不是 opmath_type<scalar_t> 类型，则启用该函数，返回类型为 void
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
// 执行连续数据的 CPU 后向传播的批量归一化实现
batch_norm_cpu_backward_contiguous_impl(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {
  // 检查是否存在混合类型的输入
  const bool mixed_type = is_mixed_type(input, weight, running_mean, running_var, save_mean, save_invstd);
  // 如果存在混合类型
  if (mixed_type) {
    // 调用内部函数，使用 scalar_t 和 opmath_type<scalar_t> 类型进行批量归一化的后向传播
    # 如果当前数据类型为 opmath_type<scalar_t>，则调用带有 opmath_type<scalar_t> 模板参数的批量归一化 CPU 反向传播函数
    batch_norm_cpu_backward_contiguous_internal<scalar_t, at::opmath_type<scalar_t>>(grad_input, grad_weight, grad_bias,
        grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
    # 否则，调用带有 scalar_t 数据类型的批量归一化 CPU 反向传播函数
    batch_norm_cpu_backward_contiguous_internal<scalar_t, scalar_t>(grad_input, grad_weight, grad_bias,
        grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
  }



// 批量归一化反向传播的内部实现，处理通道最后的数据布局
void batch_norm_cpu_backward_channels_last_internal(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {
  // 定义标量类型和参数类型
  using opmath_t = at::opmath_type<scalar_t>;
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<opmath_t>;
  
  // 计算输入数据的通道数和总元素数
  int64_t n_channel = input.size(1);
  int64_t N = input.numel() / n_channel;

  // 获取梯度输出、输入数据的指针
  const scalar_t* grad_output_data = grad_output.const_data_ptr<scalar_t>();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();

  // 定义梯度输入、权重梯度、偏置梯度的指针
  scalar_t* grad_input_data = grad_input.defined() ? grad_input.mutable_data_ptr<scalar_t>() : nullptr;
  param_t* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<param_t>() : nullptr;
  param_t* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<param_t>() : nullptr;

  // 条件访问权重和保存的均值、逆标准差
  auto weight_a = conditional_accessor_1d<const param_t>(weight);
  auto save_mean_a = conditional_accessor_1d<const param_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<const param_t>(save_invstd);
  auto running_mean_a = conditional_accessor_1d<const param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<const param_t>(running_var);

  // 使用浮点数作为累加类型
  bool weight_defined = weight.defined();
  Tensor weight_f = at::empty({n_channel}, input.options().dtype(kFloat));
  Tensor mean = at::empty({n_channel}, input.options().dtype(kFloat));
  Tensor invstd = at::empty({n_channel}, input.options().dtype(kFloat));
  opmath_t* weight_data = weight_f.data_ptr<opmath_t>();
  opmath_t* mean_data = mean.data_ptr<opmath_t>();
  opmath_t* invstd_data = invstd.data_ptr<opmath_t>();

  // 遍历每个通道，设置权重、均值和逆标准差
  for (const auto c : c10::irange(n_channel)) {
    weight_data[c] = weight_defined ? opmath_t(weight_a[c]) : 1;

    if (train) {
      mean_data[c] = save_mean_a[c];
      invstd_data[c] = save_invstd_a[c];
    } else {
      mean_data[c] = running_mean_a[c];
      invstd_data[c] = 1 / std::sqrt(running_var_a[c] + eps);
    }
  }

  // 获取线程数并创建缓冲区
  int num_threads = at::get_num_threads();
  Tensor buffer = at::zeros({2, num_threads, n_channel}, input.options().dtype(kFloat));
  opmath_t* sum_data = buffer.data_ptr<opmath_t>();
  opmath_t* dotp_data = sum_data + num_threads * n_channel;

  // 并行处理每个数据片段
  at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    opmath_t* sum_ptr = sum_data + tid * n_channel;
    opmath_t* dotp_ptr = dotp_data + tid * n_channel;
    for (const auto i : c10::irange(begin, end)) {
      // 计算当前迭代索引对应的输入和梯度数据指针
      const scalar_t* x_ptr = input_data + i * n_channel;
      const scalar_t* dy_ptr = grad_output_data + i * n_channel;

      int64_t d = 0;
      // 对于每个通道，使用向量化处理
      for(; d < n_channel - (n_channel % bVec::size()); d += bVec::size()) {
        // 加载梯度数据并转换为浮点向量
        bVec dy_bvec = bVec::loadu(dy_ptr + d);
        auto [dy_fvec0, dy_fvec1] = convert_to_float<scalar_t>(dy_bvec);
        // 计算和并存储到结果向量中
        fVec sum_fvec0 = dy_fvec0 + fVec::loadu(sum_ptr + d);
        fVec sum_fvec1 = dy_fvec1 + fVec::loadu(sum_ptr + d + fVec::size());
        sum_fvec0.store(sum_ptr + d);
        sum_fvec1.store(sum_ptr + d + fVec::size());

        // 加载输入数据并转换为浮点向量
        bVec x_bvec = bVec::loadu(x_ptr + d);
        auto [x_fvec0, x_fvec1] = convert_to_float<scalar_t>(x_bvec);
        // 加载均值、点积数据并执行点积计算
        fVec mean_fvec0 = fVec::loadu(mean_data + d);
        fVec mean_fvec1 = fVec::loadu(mean_data + d + fVec::size());
        fVec dotp_fvec0 = fVec::loadu(dotp_ptr + d);
        fVec dotp_fvec1 = fVec::loadu(dotp_ptr + d + fVec::size());
        dotp_fvec0 += (x_fvec0 - mean_fvec0) * dy_fvec0;
        dotp_fvec1 += (x_fvec1 - mean_fvec1) * dy_fvec1;
        dotp_fvec0.store(dotp_ptr + d);
        dotp_fvec1.store(dotp_ptr + d + fVec::size());
      }
      // 处理剩余的通道，非向量化
      for (; d < n_channel; d++) {
        // 获取当前通道的输入、梯度、均值数据
        opmath_t dy_val = dy_ptr[d];
        opmath_t x_val = x_ptr[d];
        opmath_t mean_val = mean_data[d];
        // 更新和与点积
        sum_ptr[d] += dy_val;
        dotp_ptr[d] += (x_val - mean_val) * dy_val;
      }
    }
  });

  at::parallel_for(0, n_channel, 1, [&](int64_t begin, int64_t end) {
    // 并行计算每个通道的最终和与点积结果
    for (const auto c : c10::irange(begin, end)) {
      // 计算所有线程中该通道的和
      opmath_t _sum = 0;
      for (const auto t : c10::irange(num_threads)) {
        _sum += sum_data[t * n_channel + c];
      }
      // 将计算结果存储到结果数组中
      sum_data[c] = _sum;

      // 计算所有线程中该通道的点积
      opmath_t _dotp = 0;
      for (const auto t : c10::irange(num_threads)) {
        _dotp += dotp_data[t * n_channel + c];
      }
      // 将点积结果存储到结果数组中
      dotp_data[c] = _dotp;
    }
  });

  // 计算梯度输入
  if (grad_input.defined()) {
    // 这里可能有一些梯度输入的计算代码，此处省略
    });
  }

  // 计算梯度权重
  if (grad_weight.defined()) {
    // 对于每个通道计算梯度权重
    for (const auto c : c10::irange(n_channel)) {
      grad_weight_data[c] = param_t(dotp_data[c] * invstd_data[c]);
    }
  }

  // 计算梯度偏置
  if (grad_bias.defined()) {
    // 对于每个通道计算梯度偏置
    for (const auto c : c10::irange(n_channel)) {
      grad_bias_data[c] = param_t(sum_data[c]);
    }
  }
}

template <typename scalar_t>
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
batch_norm_cpu_backward_channels_last_impl(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps) {
  const bool mixed_type = is_mixed_type(input, weight, running_mean, running_var, save_mean, save_invstd);
  // 检查输入张量是否包含不同数据类型
  if (mixed_type) {
    // 调用内部函数处理混合数据类型情况
    batch_norm_cpu_backward_channels_last_internal<scalar_t, at::opmath_type<scalar_t>>(grad_input, grad_weight, grad_bias,
        grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
  } else {
    // 调用内部函数处理相同数据类型情况
    batch_norm_cpu_backward_channels_last_internal<scalar_t, scalar_t>(grad_input, grad_weight, grad_bias,
        grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
  }
}

void batch_norm_cpu_kernel(Tensor& output, const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& save_mean,  const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
  // 计算每个图像数据的大小
  int64_t image_size = input.numel() / input.size(0) / input.size(1);
  // 检查输入张量是否是连续的
  if (input.is_contiguous()) { // NC11 is also channels last
    // 分发不同的浮点数类型，处理连续存储的情况
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "batch_norm_cpu_contiguous", [&] {
      // 如果图像大小为1，调用处理通道末尾布局的实现函数
      if (image_size == 1) {
        batch_norm_cpu_channels_last_impl<scalar_t>(output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
      } else {
        // 否则调用处理连续布局的实现函数
        batch_norm_cpu_contiguous_impl<scalar_t>(output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
      }
    });
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast) || input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    // 分发不同的浮点数类型，处理通道末尾布局的情况
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "batch_norm_cpu_channels_last", [&] {
      batch_norm_cpu_channels_last_impl<scalar_t>(output, input, weight, bias,
          save_mean, save_invstd, running_mean, running_var, train, eps);
    });
  } else {
    // 如果输入张量既不是连续的，也不是通道末尾布局，抛出错误
    TORCH_CHECK(false, "batch_norm_cpu_kernel: expecting input to be contiguous.");
  }
}

void batch_norm_cpu_collect_stats_kernel(
    Tensor& mean, Tensor& var_sum, const Tensor& input) {
  // 计算每个图像数据的大小
  int64_t image_size = input.numel() / input.size(0) / input.size(1);
  // 检查输入张量是否是连续的
  if (input.is_contiguous()) {
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏展开，处理浮点数类型以及 BFloat16 和 Half 类型
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "batch_norm_cpu_collect_stats_contiguous", [&] {
      // 检查图像尺寸是否为 1，对于 NC11 格式，也是通道在最后的情况
      if (image_size == 1) {
        // 调用通道在最后的实现函数，传入平均值、方差和输入数据
        batch_norm_cpu_collect_stats_channels_last_impl<scalar_t>(mean, var_sum, input);
      } else {
        // 调用连续存储的实现函数，传入平均值、方差和输入数据
        batch_norm_cpu_collect_stats_contiguous_impl<scalar_t>(mean, var_sum, input);
      }
    });
  } else if (input.is_contiguous(at::MemoryFormat::ChannelsLast) || input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    // 如果输入数据以 ChannelsLast 或 ChannelsLast3d 内存格式连续
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "batch_norm_cpu_collect_stats_channels_last", [&] {
      // 调用通道在最后的实现函数，传入平均值、方差和输入数据
      batch_norm_cpu_collect_stats_channels_last_impl<scalar_t>(mean, var_sum, input);
    });
  } else {
    // 如果输入数据不符合预期的内存格式
    TORCH_CHECK(false, "batch_norm_cpu_collect_stats_kernel: expecting input to be contiguous.");
  }
}// anonymous namespace

// 注册 batch_norm_cpu_kernel 函数为 batch_norm_cpu_stub 的分发函数
REGISTER_DISPATCH(batch_norm_cpu_stub, &batch_norm_cpu_kernel);

// 注册 batch_norm_cpu_collect_stats_kernel 函数为 batch_norm_cpu_collect_stats_stub 的分发函数
REGISTER_DISPATCH(batch_norm_cpu_collect_stats_stub, &batch_norm_cpu_collect_stats_kernel);

// 注册 batch_norm_cpu_backward_kernel 函数为 batch_norm_cpu_backward_stub 的分发函数
REGISTER_DISPATCH(batch_norm_cpu_backward_stub, &batch_norm_cpu_backward_kernel);

// 结束命名空间 at::native
} // namespace at::native
```