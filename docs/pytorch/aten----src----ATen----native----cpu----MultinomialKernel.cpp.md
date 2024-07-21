# `.\pytorch\aten\src\ATen\native\cpu\MultinomialKernel.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的 Tensor 类定义
#include <ATen/core/Tensor.h>

// 包含 ATen 库中的调度机制和分发函数
#include <ATen/Dispatch.h>
// 包含 ATen 库中的分布帮助函数
#include <ATen/core/DistributionsHelper.h>
// 包含 ATen 库中的复制操作函数
#include <ATen/native/Copy.h>
// 包含 ATen 库中的 Tensor 迭代器
#include <ATen/native/TensorIterator.h>
// 包含 ATen 库中的一元操作函数
#include <ATen/native/UnaryOps.h>
// 包含 ATen 库中 CPU 环路函数
#include <ATen/native/cpu/Loops.h>
// 包含 C10 实用工具中的整数范围迭代器
#include <c10/util/irange.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 ATen 库中的整体功能函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 否则，包含 ATen 库中的 empty 操作函数
#else
#include <ATen/ops/empty.h>
#endif

// 在 ATen 库的 native 命名空间中定义一个匿名命名空间
namespace at::native {
namespace {

// 模板函数，用于对非降低浮点数类型的 scalar_t 进行多项式分布取样
template <typename scalar_t>
typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, void>
multinomial_with_replacement_apply(
    Tensor& result,  // 输出结果张量
    const Tensor& self,  // 输入的概率分布张量
    const int64_t n_sample,  // 取样数量
    std::optional<Generator> generator) {  // 可选的随机数生成器

  // 获取默认的 CPU 生成器或者传入的生成器
  auto gen = get_generator_or_default<CPUGeneratorImpl>(
      generator, detail::getDefaultCPUGenerator());
  // 通过互斥锁保护生成器的使用，避免多线程竞争
  std::lock_guard<std::mutex> lock(gen->mutex_);

  // 获取概率分布张量的最后一个维度大小（类别数）
  int64_t n_categories = self.size(-1);
  // 获取概率分布张量的行数（分布数）
  int64_t n_dist = self.dim() > 1 ? self.size(-2) : 1;

  /* cumulative probability distribution vector */
  // 创建与类别数大小相同的空张量，用于累积概率分布
  Tensor cum_dist = at::empty({n_categories}, self.options());

  // 获取输入张量和累积概率分布张量的数据指针
  const scalar_t* const self_ptr = self.const_data_ptr<scalar_t>();
  scalar_t* const cum_dist_ptr = cum_dist.data_ptr<scalar_t>();
  int64_t* const result_ptr = result.data_ptr<int64_t>();

  // 获取输入张量的步长参数
  auto self_stride_0 = self.dim() > 1 ? self.stride(-2) : 0;
  auto self_stride_1 = self.stride(-1);

  // 获取累积概率分布张量的步长参数
  auto cum_dist_stride_0 = cum_dist.stride(0);

  // 获取输出结果张量的步长参数
  auto result_dist_stride_0 = result.dim() > 1 ? result.stride(-2) : 0;
  auto result_dist_stride_1 = result.stride(-1);

  // 循环遍历每个分布
  for (const auto i : c10::irange(n_dist)) {
    /* Get normalized cumulative distribution from prob distribution */
    // 初始化累积和为 0
    scalar_t sum = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    scalar_t val;
    // 遍历每个类别，计算累积概率分布
    for (const auto j : c10::irange(n_categories)) {
      // 获取当前类别的概率值
      val = self_ptr[i * self_stride_0 + j * self_stride_1];
      // 检查概率值是否大于等于 0
      TORCH_CHECK(
          val >= 0,
          "invalid multinomial distribution (encountering probability entry < 0)");
      // 根据不同的标准库版本，检查概率值是否为有限值
      // 对于 libc++，需要将概率值转换为 double 类型再进行检查
#if defined(_LIBCPP_VERSION)
      TORCH_CHECK(
          std::isfinite(static_cast<double>(val)),
          "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#else
      TORCH_CHECK(
          std::isfinite(val),
          "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#endif

      // 更新累积和
      sum += val;
      // 将累积和存入累积概率分布张量
      cum_dist_ptr[j * cum_dist_stride_0] = sum;
    }

    // 检查累积和是否大于 0
    TORCH_CHECK(
        sum > 0,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    /* normalize cumulative probability distribution so that last val is 1
    i.e. doesn't assume original self row sums to one */
    // 归一化累积概率分布，确保最后一个值为 1
    // 即不假设原始概率分布行之和为一
    // 如果 sum 大于 0 或者 sum 在 (0, 1) 区间内，表示累积概率分布需要调整
    if ((sum > 0) || ((sum < 1.00001) && (sum > 0.99999))) {
      // 遍历每个类别
      for (const auto j : c10::irange(n_categories)) {
        // 将累积分布中每个类别的概率除以 sum，进行归一化
        cum_dist_ptr[j * cum_dist_stride_0] /= sum;
      }
    }

    // 遍历每个采样点
    for (const auto j : c10::irange(n_sample)) {
      /* 从均匀分布中采样一个概率质量 */
      at::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(gen);
      /* 通过二分查找确定 uniform_sample 所对应的概率分布区间
      即找到使得 cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] 的槽位 */
      int left_pointer = 0;
      int right_pointer = n_categories;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int mid_pointer;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      scalar_t cum_prob;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int sample_idx;
      /* 确保最后一个累积分布槽位的和为 1 */
      cum_dist_ptr[(n_categories - 1) * cum_dist_stride_0] = 1;

      // 二分查找过程
      while (right_pointer - left_pointer > 0) {
        mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
        cum_prob = cum_dist_ptr[mid_pointer * cum_dist_stride_0];
        if (cum_prob < uniform_sample) {
          left_pointer = mid_pointer + 1;
        } else {
          right_pointer = mid_pointer;
        }
      }
      sample_idx = left_pointer;

      /* 将采样结果存储在结果张量中（Lua 兼容性需要增加对应的值） */
      result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] =
          sample_idx;
    }
  }
  /* cumulative probability distribution vector */
  // 创建累积概率分布向量

  Tensor cum_dist = at::empty({n_categories}, self.options().dtype(kFloat));
  // 使用与 self 相同的选项创建一个空的张量 cum_dist，用于存储累积分布

  const scalar_t* const self_ptr = self.const_data_ptr<scalar_t>();
  // 获取 self 张量的常量数据指针，类型为 scalar_t

  float* const cum_dist_ptr = cum_dist.data_ptr<float>();
  // 获取 cum_dist 张量的数据指针，类型为 float

  int64_t* const result_ptr = result.data_ptr<int64_t>();
  // 获取 result 张量的数据指针，类型为 int64_t

  auto self_stride_0 = self.dim() > 1 ? self.stride(-2) : 0;
  // 计算 self 张量在第一个维度上的步长

  auto self_stride_1 = self.stride(-1);
  // 计算 self 张量在最后一个维度上的步长

  auto cum_dist_stride_0 = cum_dist.stride(0);
  // 计算 cum_dist 张量在第一个维度上的步长

  auto result_dist_stride_0 = result.dim() > 1 ? result.stride(-2) : 0;
  // 计算 result 张量在第一个维度上的步长，如果维度大于 1

  auto result_dist_stride_1 = result.stride(-1);
  // 计算 result 张量在最后一个维度上的步长
    // 对于每个样本进行循环采样
    for (const auto j : c10::irange(n_sample)) {
      /* 从均匀分布中采样一个概率质量 */
      at::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(gen);
      /* 通过二分查找确定概率所属的槽位
      即 cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
      int left_pointer = 0;
      int right_pointer = n_categories;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int mid_pointer;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      float cum_prob;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int sample_idx;
      /* 确保最后一个累积分布桶的总和为1 */
      cum_dist_ptr[(n_categories - 1) * cum_dist_stride_0] = 1;

      // 执行二分查找
      while (right_pointer - left_pointer > 0) {
        mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
        cum_prob = cum_dist_ptr[mid_pointer * cum_dist_stride_0];
        if (cum_prob < uniform_sample) {
          left_pointer = mid_pointer + 1;
        } else {
          right_pointer = mid_pointer;
        }
      }
      sample_idx = left_pointer;

      /* 将采样结果存储在结果张量中（由包装器负责增量更新以实现与 Lua 的兼容） */
      result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] =
          sample_idx;
    }
  }
}

static void multinomial_with_replacement_kernel_impl(
    Tensor& result,                        // 用于存储采样结果的张量
    const Tensor& self,                    // 输入张量，从中进行采样
    const int64_t n_sample,                // 采样次数
    std::optional<Generator> gen) {        // 可选的随机数生成器
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "multinomial", [&] {
        // 调用具体的采样实现函数，根据不同的浮点类型进行分发
        multinomial_with_replacement_apply<scalar_t>(
            result, self, n_sample, gen);
      });
}
} // namespace

REGISTER_DISPATCH(
    multinomial_with_replacement_stub,     // 注册分发函数，用于分发到具体的实现函数
    &multinomial_with_replacement_kernel_impl);
} // namespace at::native
```