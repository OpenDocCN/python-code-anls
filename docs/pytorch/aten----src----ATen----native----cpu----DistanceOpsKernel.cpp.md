# `.\pytorch\aten\src\ATen\native\cpu\DistanceOpsKernel.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于特定的方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含距离计算的头文件
#include <ATen/native/Distance.h>

// 包含标准库的算法头文件
#include <algorithm>

// 包含 ATen 库的张量头文件
#include <ATen/core/Tensor.h>
// 包含 ATen 库的调度头文件
#include <ATen/Dispatch.h>
// 包含 ATen 库的并行处理头文件
#include <ATen/Parallel.h>
// 包含 ATen 库的张量迭代器头文件
#include <ATen/TensorIterator.h>
// 包含 ATen 库的 CPU 向量功能头文件
#include <ATen/cpu/vec/functional.h>
// 包含 C10 库的工具头文件，用于生成范围
#include <c10/util/irange.h>

// ATen 库的 native 命名空间
namespace at::native {

// 匿名命名空间，用于局部结构和函数的定义
namespace {

// 模板结构 Dist，用于计算距离
template<typename scalar_t>
struct Dist {
  using Vec = vec::Vectorized<scalar_t>;

  // 根据 p 范数的值，选择特定的实现方式以提高效率
  // 具体实现方式有多个静态函数组成，用于内部循环的不同逻辑处理
  // 这些函数分别是：
  //   map :      说明如何修改 (a - b) 以形成要求求和的组件。
  //   red :      说明如何对 map 的结果进行求和。这是独立的，因为无穷范数使用 max 而不是 sum。
  //   finish :   说明如何使用汇总值来计算范数。通常这是 val ^ (1 / p) 的结果。
  //   backward : 说明该范数的梯度计算。参数相当自明。
  // 有几种情况下不使用这些函数。0 范数没有 backward，因为它总是 0，所以会在前面被短路。
  // 当 p 小于 2 时，通常的梯度计算有特殊实现，因此有一个仅包含此情况下 backward 计算的结构。

  // TODO 这是一种计算符号的低效方式，可以使用本地 SSE 指令，这应该添加到 Vectorized 中以提高性能。
  // 计算给定向量的符号函数
  static inline Vec sign(Vec val) {
    return vec::minimum(vec::maximum(Vec(0), val.ceil()), Vec(1)) +
      vec::minimum(vec::maximum(Vec(-1), val.floor()), Vec(0));
  }

  // 计算给定向量的绝对值函数
  static inline Vec abs(Vec val) {
    return val.abs();
  }

  // 计算给定标量的绝对值函数
  static inline scalar_t abs(scalar_t val) {
    return std::abs(val);
  }

  // 计算给定向量的向上取整函数
  static inline Vec ceil(Vec val) {
    return val.ceil();
  }

  // 计算给定标量的向上取整函数
  static inline scalar_t ceil(scalar_t val) {
    return std::ceil(val);
  }

  // 计算给定向量和标量的最小值函数
  static inline Vec min(Vec val, scalar_t other) {
    return vec::minimum(val, Vec(other));
  }

  // 计算给定两个标量的最小值函数
  static inline scalar_t min(scalar_t val, scalar_t other) {
    return std::min(val, other);
  }

  // 计算给定向量的最大值函数
  static inline Vec max(Vec val, Vec other) {
    return vec::maximum(val, other);
  }

  // 计算给定两个标量的最大值函数
  static inline scalar_t max(scalar_t val, scalar_t other) {
    return std::max(val, other);
  }

  // 计算给定向量的幂函数
  static inline Vec pow(Vec val, Vec p) {
    return val.pow(p);
  }

  // 计算给定标量的幂函数
  static inline scalar_t pow(scalar_t val, scalar_t p) {
    // 返回 val 的 p 次幂的结果
    return std::pow(val, p);
  }

  // Zero norm
  template<typename data_t>
  struct zdist_calc {
    // 将 diff 映射为 0 或 1，并返回结果
    static inline data_t map(const data_t& diff, const data_t& p) { return min(ceil(abs(diff)), 1); }
    // 对聚合值 agg 和更新值 up 进行简单的累加操作
    static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
    // 返回累加值 agg 作为最终结果
    static inline scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
  };

  // One norm
  template<typename data_t>
  struct odist_calc {
    // 直接返回 diff 的值作为映射结果
    static inline data_t map(const data_t& diff, const data_t& p) { return diff; }
    // 对聚合值 agg 和更新值 up 进行简单的累加操作
    static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
    // 返回累加值 agg 作为最终结果
    static inline scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
    // 根据梯度 grad、距离 dist 和向量 p，计算并返回反向传播结果
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t /*dist*/, const Vec& /*p*/) { return Vec(grad) * sign(diff); }
  };

  // Special general pnorm derivative if p is less than two
  struct lttdist_calc {
    // 根据 diff、grad、dist 和 p 计算反向传播的结果向量
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) {
      // 计算结果向量 result，考虑特殊情况 dist == 0
      Vec result = (dist == 0.0) ? Vec(0) : (sign(diff) * diff.abs().pow(p - Vec(1)) * Vec(grad) / Vec(dist).pow(p - Vec(1)));
      // 在 diff == 0 且 p < 1 的情况下，使用 blendv 将 result 中对应位置置为 0
      result = Vec::blendv(result, Vec(0), (diff == Vec(0)) & (p < Vec(1)));
      return result;
    }
  };

  // Two norm
  template<typename data_t>
  struct tdist_calc {
    // TODO This can probably use fused add multiply to get better perf
    // 将 diff 映射为其平方，并返回结果
    static inline data_t map(const data_t& diff, const data_t& p) { return diff * diff; }
    // 对聚合值 agg 和更新值 up 进行简单的累加操作
    static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
    // 返回聚合值 agg 的平方根作为最终结果
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return std::sqrt(agg); }
    // 根据梯度 grad、距离 dist 和向量 p，计算并返回反向传播结果向量
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return dist == 0.0 ? Vec(0) : Vec(grad) * diff / Vec(dist); }
  };

  // General p norm
  template<typename data_t>
  struct pdist_calc {
    // 将 diff 映射为 diff 的 p 次方，并返回结果
    static inline data_t map(const data_t& diff, const data_t& p) { return pow(diff, p); }
    // 对聚合值 agg 和更新值 up 进行简单的累加操作
    static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
    // 返回聚合值 agg 的 p 次根作为最终结果
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return std::pow(agg, 1.0 / p); }
    // 根据梯度 grad、距离 dist 和向量 p，计算并返回反向传播结果向量
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return dist == 0.0 ? Vec(0) : diff * diff.abs().pow(p - Vec(2)) * Vec(grad) / Vec(dist).pow(p - Vec(1)); }
  };

  // Inf norm
  template<typename data_t>
  struct idist_calc {
    // 直接返回 diff 的值作为映射结果
    static inline data_t map(const data_t& diff, const data_t& p) { return diff; }
    // 对聚合值 agg 和更新值 up 进行最大值比较操作
    static inline data_t red(const data_t& agg, const data_t& up) { return max(agg, up); }
    // 返回聚合值 agg 作为最终结果
    static inline scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    // TODO This backward pass uses a very complext expression to compute (diff
    // == dist) that could be much faster if using SSE instructions.
    // 根据 diff、grad、dist 和向量 p，计算并返回反向传播结果向量；有潜在的优化空间
    static inline Vec backward(const Vec& diff, const scalar_t grad, const scalar_t dist, const Vec& p) { return Vec(grad) * sign(diff) * (Vec(1) - vec::minimum(Vec(1), (diff.abs() - Vec(dist)).abs().ceil())); }
  };

  template <typename F>
  static void run_parallel_pdist(Tensor& result, const Tensor& self, const scalar_t p) {
    const scalar_t * const self_start = self.const_data_ptr<scalar_t>();  // 获取 self 张量的常量数据指针
    const scalar_t * const self_end = self_start + self.numel();  // 计算 self 张量数据结束位置的指针
    int64_t n = self.size(0);  // 获取 self 张量的第一维大小
    int64_t m = self.size(1);  // 获取 self 张量的第二维大小

    scalar_t * const res_start = result.data_ptr<scalar_t>();  // 获取结果张量的数据指针
    int64_t combs = result.numel(); // n * (n - 1) / 2，计算结果张量元素个数

    // We conceptually iterate over tuples of (i, j, k) where i is the first
    // vector from the input, j is the second, and k is the result index. This
    // parallelizes over the range of k and infers what i and j are from the
    // value of k.
    parallel_for(0, combs, internal::GRAIN_SIZE / (16 * m), [p, self_start, self_end, n, m, res_start](int64_t k, int64_t end) {
      const Vec pvec(p);  // 创建 Vec 对象 pvec，用于表示标量 p 的向量形式
      double n2 = n - .5;  // 计算 n - 0.5
      // The -1 accounts for floating point truncation issues
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      int64_t i = static_cast<int64_t>((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));  // 根据 k 计算第一个向量索引 i
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;  // 根据 k 计算第二个向量索引 j

      const scalar_t * self_i = self_start + i * m;  // 获取第一个向量 self_i 的起始地址
      const scalar_t * self_j = self_start + j * m;  // 获取第二个向量 self_j 的起始地址
      scalar_t * res = res_start + k;  // 获取结果张量的当前位置的指针
      const scalar_t * const res_end = res_start + end;  // 获取结果张量的结束位置的指针

      while (res != res_end) {
        *res = F::finish(vec::map2_reduce_all<scalar_t>(
          [&pvec](Vec a, Vec b) { return F::map((a - b).abs(), pvec); },
          F::red, self_i, self_j, m), p);  // 计算两个向量之间的距离，并将结果存入结果张量

        res += 1;  // 移动到结果张量的下一个位置
        self_j += m;  // 移动到第二个向量的下一个位置
        if (self_j == self_end) {  // 如果第二个向量已经到达末尾
          self_i += m;  // 移动到第一个向量的下一个位置
          self_j = self_i + m;  // 重新设置第二个向量的起始位置
        }
      }
    });
  }

  // Assumes self is nonempty, contiguous, and 2D
  static void apply_pdist(Tensor& result, const Tensor& self, const scalar_t p) {
    if (p == 0.0) {
      run_parallel_pdist<zdist_calc<Vec>>(result, self, p);  // 使用 zdist_calc 计算零范数距离
    } else if (p == 1.0) {
      run_parallel_pdist<odist_calc<Vec>>(result, self, p);  // 使用 odist_calc 计算一范数距离
    } else if (p == 2.0) {
      run_parallel_pdist<tdist_calc<Vec>>(result, self, p);  // 使用 tdist_calc 计算二范数距离
    } else if (std::isinf(p)) {
      run_parallel_pdist<idist_calc<Vec>>(result, self, p);  // 使用 idist_calc 计算无穷范数距离
    } else {
      run_parallel_pdist<pdist_calc<Vec>>(result, self, p);  // 使用 pdist_calc 计算 p 范数距离
    }
  }

  template <typename F>
  static void run_parallel_cdist(Tensor& result, const Tensor& t1, const Tensor& t2, const scalar_t p) {
    const scalar_t * const t1_start = t1.const_data_ptr<scalar_t>();  // 获取 t1 张量的常量数据指针
    const scalar_t * const t2_start = t2.const_data_ptr<scalar_t>();  // 获取 t2 张量的常量数据指针
    int64_t d = t1.size(0);  // 获取 t1 张量的第一维大小
    int64_t r1 = t1.size(-2);  // 获取 t1 张量的倒数第二维大小
    int64_t r2 = t2.size(-2);  // 获取 t2 张量的倒数第二维大小
    int64_t m = t1.size(-1);  // 获取 t1 张量的最后一维大小

    scalar_t * const res_start = result.data_ptr<scalar_t>();  // 获取结果张量的数据指针
    int64_t combs = r1 * r2;  // 计算结果张量的元素个数
    int64_t size1 = r1 * m;
    // 计算第二个维度的大小
    int64_t size2 = r2 * m;

    // 并行循环，对每对数据进行计算，使用一定的GRAIN_SIZE和内部函数
    parallel_for(0, combs * d, internal::GRAIN_SIZE / (16 * m), [=](int64_t start, int64_t end) {
      // 指向结果的指针，从指定的起始位置开始
      scalar_t * res = res_start + start;
      // 指向结果的结束位置
      const scalar_t * const res_end = res_start + end;
      // 计算在矩阵中的位置索引
      int64_t l = start / combs;
      int64_t k = start % combs;
      int64_t i = k / r2;
      int64_t j = k % r2;
      // 调整索引以匹配数据存储
      i = i * m;
      j = j * m;

      // 遍历处理每对数据直到结束
      while (res != res_end) {
        // 指向第一个输入张量的起始位置
        const scalar_t * self_i = t1_start + size1 * l + i;
        // 指向第二个输入张量的起始位置
        const scalar_t * self_j = t2_start + size2 * l + j;

        // 初始化聚合值
        scalar_t agg = 0;
        // 对每一维度进行迭代，计算聚合结果
        for (const auto x : c10::irange(m)) {
          scalar_t a = *(self_i + x);
          scalar_t b = *(self_j + x);
          // 使用给定函数进行聚合操作
          agg = F::red(agg, F::map(std::abs(a-b), p));
        }
        // 完成聚合操作，将结果写入指定位置
        *res = F::finish(agg, p);

        // 移动到下一个结果位置
        res += 1;
        // 调整第二个输入张量的起始位置索引
        j += m;
        // 如果第二个索引达到其最大值，则调整第一个索引
        if (j == size2) {
          j = 0;
          i += m;
          // 如果第一个索引达到其最大值，则调整批处理索引
          if (i == size1) {
            i = 0;
            l += 1;
          }
        }
      }
    });
  }

  // 应用指定的距离函数到输入张量对，并使用并行处理
  static void apply_cdist(Tensor& result, const Tensor& x1, const Tensor& x2, const scalar_t p) {
    // 根据距离函数的参数选择适当的计算函数并执行
    if (p == 0.0) {
      run_parallel_cdist<zdist_calc<scalar_t>>(result, x1, x2, p);
    } else if (p == 1.0) {
      run_parallel_cdist<odist_calc<scalar_t>>(result, x1, x2, p);
    } else if (p == 2.0) {
      run_parallel_cdist<tdist_calc<scalar_t>>(result, x1, x2, p);
    } else if (std::isinf(p)) {
      run_parallel_cdist<idist_calc<scalar_t>>(result, x1, x2, p);
    } else {
      run_parallel_cdist<pdist_calc<scalar_t>>(result, x1, x2, p);
    }
  }

  // 执行向后传递算法以处理输入向量的列
  // F 是用于向后传递的函数对象
  template <typename F>
  inline static void backward_down_column_pdist(const scalar_t * self_i, scalar_t * res_i, const scalar_t * grad_k, const scalar_t * dist_k, const Vec& pvec, int64_t n, int64_t m, int64_t gs, int64_t count = Vec::size()) {
    // 遍历处理每一列的向后传递
    for (const scalar_t * const self_end = self_i + m * n; self_i != self_end - m; self_i += m, res_i += m) {

      // 加载当前向量数据到向量对象
      const Vec self_vec_i = Vec::loadu(self_i, count);
      Vec res_vec_i = Vec::loadu(res_i, count);

      // 初始化第二个向量的指针位置
      const scalar_t * self_j = self_i + m;
      scalar_t * res_j = res_i + m;
      // 遍历处理每一列的向后传递
      for (; self_j != self_end; self_j += m, res_j += m, grad_k += gs, dist_k += 1) {
        // 加载第二个向量的数据到向量对象
        const Vec self_vec_j = Vec::loadu(self_j, count);
        Vec res_vec_j = Vec::loadu(res_j, count);

        // 使用指定的函数对象执行向后传递操作
        Vec res = F::backward(self_vec_i - self_vec_j, *grad_k, *dist_k, pvec);
        // 更新第一和第二向量的结果
        res_vec_i = res_vec_i + res;
        res_vec_j = res_vec_j - res;

        // 将更新后的第二向量结果存储回数组
        res_vec_j.store(res_j, count);
      }

      // 将更新后的第一向量结果存储回数组
      res_vec_i.store(res_i, count);
    }
  }

  // 执行并行向后传递算法以处理输入张量
  // F 是用于向后传递的函数对象
  template <typename F>
  static void run_backward_parallel_pdist(Tensor& result, const Tensor & grad, const Tensor & self, const scalar_t p, const Tensor& dist) {
    // 获取输入张量的维度信息
    const int64_t n = self.size(0);
    const int64_t m = self.size(1);
    const int64_t gs = grad.stride(0);

    // 获取梯度张量的起始指针
    const scalar_t * const grad_start = grad.const_data_ptr<scalar_t>();
    // 定义指向 dist 张量数据起始位置的常量指针，类型为 scalar_t
    const scalar_t * const dist_start = dist.const_data_ptr<scalar_t>();
    // 定义指向 self 张量数据起始位置的常量指针，类型为 scalar_t
    const scalar_t * const self_start = self.const_data_ptr<scalar_t>();
    // 定义指向 result 张量数据起始位置的指针，类型为 scalar_t
    scalar_t * const res_start = result.data_ptr<scalar_t>();

    // 并行化的唯一方式是避免锁定，需要并行化输入的列，即独立计算每个向量的第一部分的梯度，
    // 不受第二部分等影响
    at::parallel_for(0, m / Vec::size(), internal::GRAIN_SIZE / (8 * n * n), [p, n, m, gs, grad_start, dist_start, self_start, res_start](int64_t l, int64_t end) {
      // 创建 Vec 类型的对象 pvec
      const Vec pvec(p);

      // 定义指向 self 张量第 l 组数据起始位置的指针
      const scalar_t * self_l = self_start + l * Vec::size();
      // 定义指向 result 张量第 l 组数据起始位置的指针
      scalar_t * res_l = res_start + l * Vec::size();

      // 遍历处理每个 res_l 到 res_end 的数据段
      for (const scalar_t * const res_end = res_start + end * Vec::size(); res_l != res_end; self_l += Vec::size(), res_l += Vec::size()) {
        // 调用 backward_down_column_pdist 函数处理 self_l 和 res_l 数据段
        backward_down_column_pdist<F>(self_l, res_l, grad_start, dist_start, pvec, n, m, gs);
      }
    });

    // 计算 m 对 Vec::size() 的余数
    const int64_t remainder = m % Vec::size();
    // 如果余数不为零，处理余下的数据段
    if (remainder) {
      // 调用 backward_down_column_pdist 处理余下的 self_start 和 res_start 的数据段
      backward_down_column_pdist<F>(self_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, Vec(p), n, m, gs, remainder);
    }
  }

  // 假设 self 非空、连续且二维，dist 也是连续的，应用于向后推导 pdist
  static void apply_backward_pdist(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
    // 用零填充 result 张量
    result.fill_(0);
    // 根据 p 的不同值，调用不同的 run_backward_parallel_pdist 函数
    if (p == 0.0) {
    } else if (p == 1.0) {
      run_backward_parallel_pdist<odist_calc<Vec>>(result, grad, self, p, dist);
    } else if (p < 2.0) {
      run_backward_parallel_pdist<lttdist_calc>(result, grad, self, p, dist);
    } else if (p == 2.0) {
      run_backward_parallel_pdist<tdist_calc<Vec>>(result, grad, self, p, dist);
    } else if (std::isinf(p)) {
      run_backward_parallel_pdist<idist_calc<Vec>>(result, grad, self, p, dist);
    } else {
      run_backward_parallel_pdist<pdist_calc<Vec>>(result, grad, self, p, dist);
    }
  }

  // 用于向后推导 cdist
  static void apply_backward_cdist(Tensor& result, const Tensor& grad, const Tensor& x1, const Tensor& x2, const double p, const Tensor& dist) {
    // 用零填充 result 张量
    result.fill_(0);
    // 根据 p 的不同值，调用不同的 run_backward_parallel_cdist 函数
    if (p == 0.0) {
    } else if (p == 1.0) {
      run_backward_parallel_cdist<odist_calc<Vec>>(result, grad, x1, x2, p, dist);
    } else if (p < 2.0) {
      run_backward_parallel_cdist<lttdist_calc>(result, grad, x1, x2, p, dist);
    } else if (p == 2.0) {
      run_backward_parallel_cdist<tdist_calc<Vec>>(result, grad, x1, x2, p, dist);
    } else if (std::isinf(p)) {
      run_backward_parallel_cdist<idist_calc<Vec>>(result, grad, x1, x2, p, dist);
    } else {
      run_backward_parallel_cdist<pdist_calc<Vec>>(result, grad, x1, x2, p, dist);
    }
  }

  // 使用模板函数 F，运行并行的 cdist 向后推导
  template <typename F>
  static void run_backward_parallel_cdist(Tensor& result, const Tensor & grad, const Tensor & t1, const Tensor & t2, const scalar_t p, const Tensor& dist) {
    // 获取 t1 和 t2 的维度大小
    const int64_t r1 = t1.size(-2);
    const int64_t r2 = t2.size(-2);
    // 获取张量 t1 的最后一个维度大小作为 m
    const int64_t m = t1.size(-1);
    // 获取结果张量 result 的第一个维度大小作为 d
    const int64_t d = result.size(0);
    // 计算 r1 乘以 m 的结果作为 l1_size
    const int64_t l1_size = r1 * m;
    // 计算 r2 乘以 m 的结果作为 l2_size
    const int64_t l2_size = r2 * m;
    
    // 当前实现仅支持可以折叠到一维的张量。为了避免检查梯度是否满足此假设，
    // 我们在执行 backward 之前调用 .contiguous() 方法，确保张量的步长为 1。
    // 不使用 grad.stride(-1)，因为如果最后一个维度为 1，步长可能是错误的。
    const int64_t gs = 1;

    // 获取 grad 张量的指针
    const scalar_t * const grad_start = grad.const_data_ptr<scalar_t>();
    // 获取 dist 张量的指针
    const scalar_t * const dist_start = dist.const_data_ptr<scalar_t>();
    // 获取 t1 张量的指针
    const scalar_t * const t1_start = t1.const_data_ptr<scalar_t>();
    // 获取 t2 张量的指针
    const scalar_t * const t2_start = t2.const_data_ptr<scalar_t>();
    // 获取 result 张量的指针
    scalar_t * const res_start = result.data_ptr<scalar_t>();

    // 并行处理主循环，将工作分配给多个线程
    at::parallel_for(0, m / Vec::size(), internal::GRAIN_SIZE / (16 * r1), [=](int64_t l, int64_t end) {
      // 创建一个 Vec 对象 pvec，用于向量化操作
      const Vec pvec(p);

      // 设置指针 i 指向 t1 的起始位置加上偏移量 l * Vec::size()
      const scalar_t * i = t1_start + l * Vec::size();
      // 设置指针 j 指向 t2 的起始位置加上偏移量 l * Vec::size()
      const scalar_t * j = t2_start + l * Vec::size();
      // 设置指针 res_l 指向 result 的起始位置加上偏移量 l * Vec::size()
      scalar_t * res_l = res_start + l * Vec::size();

      // 对每个 Vec::size() 大小的向量执行循环，直到达到 end * Vec::size() 大小
      for (const scalar_t * const res_end = res_start + end * Vec::size(); res_l != res_end; i += Vec::size(), j += Vec::size(), res_l += Vec::size()) {
        // 调用 backward_down_column_cdist 函数处理当前块的向量计算
        backward_down_column_cdist<F>(i, j, res_l, grad_start, dist_start, pvec, r1, r2, m, d, gs, l1_size, l2_size);
      }
    });

    // 处理剩余的部分，如果 m 不能整除 Vec::size()
    const int64_t remainder = m % Vec::size();
    if (remainder) {
      // 调用 backward_down_column_cdist 处理剩余部分
      backward_down_column_cdist<F>(t1_start + (m - remainder), t2_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, Vec(p), r1, r2, m, d, gs, l1_size, l2_size, remainder);
    }
  }

  // 后向传播计算距离的列之间的负梯度
  template <typename F>
  inline static void backward_down_column_cdist(const scalar_t * t1, const scalar_t * t2, scalar_t * res, const scalar_t * grad_k, const scalar_t * dist_k, const Vec& pvec, int64_t r1, int64_t r2, int64_t m, int64_t d, int64_t gs, int64_t l1_size, int64_t l2_size, int64_t count = Vec::size()) {
    // 设置指针 t1_end 指向 t1 加上 l1_size 的位置
    const scalar_t * t1_end = t1 + l1_size;
    // 设置指针 t2_end 指向 t2 加上 l2_size 的位置
    const scalar_t * t2_end = t2 + l2_size;

    // 对每个 d 进行迭代
    for (const auto l C10_UNUSED : c10::irange(d)) {
      // 对每个 t1 的块执行迭代，直到 t1_end
      for (; t1 != t1_end; t1 += m, res += m) {
        // 加载 t1 的 Vec::size() 大小的向量到 vec_t1
        const Vec vec_t1 = Vec::loadu(t1, count);
        // 加载 res 的 Vec::size() 大小的向量到 res_vec
        Vec res_vec = Vec::loadu(res, count);

        // 对每个 t2 进行迭代，直到 t2_end
        for (const scalar_t * t2_curr = t2; t2_curr != t2_end; t2_curr += m, grad_k += gs, dist_k += 1) {
          // 加载 t2_curr 的 Vec::size() 大小的向量到 vec_t2
          const Vec vec_t2 = Vec::loadu(t2_curr, count);
          // 计算 F::backward 函数的结果，并将其加到 res_vec 中
          Vec res = F::backward(vec_t1 - vec_t2, *grad_k, *dist_k, pvec);
          res_vec = res_vec + res;
        }

        // 将 res_vec 的值存储回 res
        res_vec.store(res, count);
      }
      // 更新 t1_end 和 t2_end 的位置
      t1_end += l1_size;
      t2_end += l2_size;
      t2 += l2_size;
    }
  }
}  // anonymous namespace



// 匿名命名空间结束，定义了一组不具名的实体，其作用域限定在当前文件中

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
// 注册分发函数，将 pdist_forward_stub 与 pdist_forward_kernel_impl 函数关联起来，用于分发特定类型的计算任务

REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl);
// 注册分发函数，将 pdist_backward_stub 与 pdist_backward_kernel_impl 函数关联起来，用于分发特定类型的反向计算任务

REGISTER_DISPATCH(cdist_stub, &cdist_kernel_impl);
// 注册分发函数，将 cdist_stub 与 cdist_kernel_impl 函数关联起来，用于分发特定类型的计算任务

REGISTER_DISPATCH(cdist_backward_stub, &cdist_backward_kernel_impl);
// 注册分发函数，将 cdist_backward_stub 与 cdist_backward_kernel_impl 函数关联起来，用于分发特定类型的反向计算任务

}  // namespace at::native



// 结束 at::native 命名空间的定义，此命名空间可能用于封装一些本地操作或功能


这段代码片段主要涉及注册分发函数以及命名空间的使用，用于在 PyTorch 框架中注册并分发计算任务和反向计算任务。
```