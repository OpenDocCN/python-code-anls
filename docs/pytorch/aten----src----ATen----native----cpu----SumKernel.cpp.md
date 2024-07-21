# `.\pytorch\aten\src\ATen\native\cpu\SumKernel.cpp`

```
/* 禁用 Torch 运算符断言，包含必要的头文件和命名空间 */

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <ATen/cpu/vec/functional.h>
#include <algorithm>

namespace at::native {
namespace {

// 从较小类型（元素较多）加载向量到较大类型（元素较少），通过邻近元素的缩减使其适应向量大小。
template <typename acc_t, typename scalar_t, typename F>
Vectorized<acc_t> load_reduce_vec(const scalar_t* data, F reduce, acc_t ident) {
  using vec_t = Vectorized<scalar_t>;
  using vacc_t = Vectorized<acc_t>;
  static_assert(vacc_t::size() <= vec_t::size(), "");
  // 加载未对齐的向量数据到 val
  const auto val = vec_t::loadu(data);
  alignas(64) std::array<scalar_t, vec_t::size()> values;
  // 将 val 存储到 values 数组中
  val.store(values.data());

  constexpr int vstride = vec_t::size() / vacc_t::size();
  alignas(64) std::array<acc_t, vacc_t::size()> acc;
  // 使用 ident 初始化 acc 数组
  acc.fill(ident);
  // 对 vstride 和 vacc_t::size() 进行循环，以减少相邻元素的值，返回 vacc_t 的加载
  for (const auto k : c10::irange(vstride)) {
    for (const auto i : c10::irange(vacc_t::size())) {
      acc[i] = reduce(acc[i], values[i * vstride + k]);
    }
  }

  return vacc_t::loadu(acc.data());
}

template <typename scalar_t>
struct LoadPolicy {
  // 返回标量类型所占用的内存大小
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  // 加载指定索引处的数据，使用步长来计算数据地址
  static scalar_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = reinterpret_cast<const scalar_t*>(data + index * stride);
    return *ptr;
  }
};

template <typename scalar_t>
struct LoadPolicy<Vectorized<scalar_t>> {
  // 返回向量类型所占用的内存大小
  static constexpr int64_t memsize() {
    return sizeof(scalar_t) * Vectorized<scalar_t>::size();
  }

  // 加载指定索引处的向量数据，使用步长来计算数据地址
  static Vectorized<scalar_t> load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = data + index * stride;
    return Vectorized<scalar_t>::loadu(ptr);
  }
};

/* 当对 float16 或 BFloat16 求和时，由于硬件只支持 float，必须执行加法运算。
 * 这些 CastLoad 策略确保整个求和循环在 float 类型下执行，从而提高性能和准确性。
 */

template <typename scalar_t, typename acc_t>
struct CastLoadPolicy {
  // 返回标量类型所占用的内存大小
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  // 加载指定索引处的数据并转换为 acc_t 类型
  static acc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    const auto val = LoadPolicy<scalar_t>::load(data, stride, index);
    return acc_t(val);
  }
};

template <typename scalar_t>
struct CastLoadPolicy<scalar_t, scalar_t>:
    LoadPolicy<scalar_t> {
};

// 对于内部求和，加载完整的 vec_t 然后将部分求和到 vacc_t 大小
template <typename vec_t, typename vacc_t, typename = void>
struct InnerSumCastLoadPolicy;


注释：以上是对 C++ 代码中一部分定义和模板的注释，涵盖了数据加载和类型转换策略。
// 定义一个模板结构体 `InnerSumCastLoadPolicy`，接受类型参数 `vec_t`、`vacc_t`，
// 并使用 `std::enable_if_t` 条件来限制类型，当 `vec_t` 不是简化的浮点类型并且不与 `vacc_t` 相同时生效
template <vec_t, vacc_t,
  std::enable_if_t<(!is_reduced_floating_point_v<vechold_type<vec_t>>) &&
                    !std::is_same_v<vec_t, vacc_t>>> {
  // 定义类型别名 `scalar_t` 作为 `vec_t` 的元素类型
  using scalar_t = vechold_type<vec_t>;
  // 定义类型别名 `acc_t` 作为 `vacc_t` 的元素类型
  using acc_t = vechold_type<vacc_t>;

  // 返回 `LoadPolicy<vec_t>` 的内存大小作为常量表达式
  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  // 加载函数 `load`，接受指向数据的 `data` 指针、步长 `stride` 和索引 `index`
  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    // 将 `data` 指针解释为 `scalar_t` 类型指针
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    // 使用 `load_reduce_vec` 函数加载 `ptr` 指向的数据，并将其归约为 `acc_t` 类型
    return load_reduce_vec<acc_t>(ptr, [](acc_t a, scalar_t b) {
      return a + b;
    }, acc_t(0));
  }
};

// 当 `vec_t` 和 `vacc_t` 相同时的偏特化，继承自 `LoadPolicy<scalar_t>`
template <typename scalar_t>
struct InnerSumCastLoadPolicy<scalar_t, scalar_t, void>:
    LoadPolicy<scalar_t> {
};

// 定义一个模板结构体 `InnerSumCastLoadPolicy` 的另一个特化版本，
// 限制条件为 `vec_t` 是简化的浮点类型时生效
template <vec_t, vacc_t,
  std::enable_if_t<is_reduced_floating_point_v<vechold_type<vec_t>>>> {
  // 定义类型别名 `scalar_t` 作为 `vec_t` 的元素类型
  using scalar_t = vechold_type<vec_t>;

  // 返回 `LoadPolicy<vec_t>` 的内存大小作为常量表达式
  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  // 加载函数 `load`，接受指向数据的 `data` 指针、步长 `stride` 和索引 `index`
  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    // 将 `data` 指针解释为 `scalar_t` 类型指针
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    // 声明 `vacc_t` 类型的变量 `first` 和 `second`
    vacc_t first, second;
    // 使用 `load_to_float` 函数加载 `ptr` 指向的数据，并存储到 `first` 和 `second`
    vec::load_to_float<scalar_t>(ptr, first, second);
    // 返回 `first` 和 `second` 相加的结果
    return first + second;
  }
};

// 定义模板结构体 `OuterSumCastLoadPolicy`，接受类型参数 `vec_t`、`vacc_t` 和一个默认的 `void` 参数
// 用 `std::enable_if_t` 限制条件来选择特定的实现，当 `vec_t` 不是简化的浮点类型并且不与 `vacc_t` 相同时生效
template <typename vec_t, typename vacc_t>
struct OuterSumCastLoadPolicy <vec_t, vacc_t,
  std::enable_if_t<(!is_reduced_floating_point_v<vechold_type<vec_t>>) &&
                    !std::is_same_v<vec_t, vacc_t>>> {

  // 定义类型别名 `scalar_t` 作为 `vec_t` 的元素类型
  using scalar_t = vechold_type<vec_t>;
  // 定义类型别名 `acc_t` 作为 `vacc_t` 的元素类型
  using acc_t = vechold_type<vacc_t>;

  // 返回 `sizeof(scalar_t) * vacc_t::size()` 的结果作为内存大小的常量表达式
  static constexpr int64_t memsize() {
    return sizeof(scalar_t) * vacc_t::size();
  }

  // 加载函数 `load`，接受指向数据的 `data` 指针、步长 `stride` 和索引 `index`
  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    // 静态断言确保 `vacc_t::size()` 不大于 `vec_t::size()`
    static_assert(vacc_t::size() <= vec_t::size(), "");
    // 使用 `vec_t::loadu` 加载指定大小的数据到 `val`
    const auto val = vec_t::loadu(data + stride * index, vacc_t::size());
    // 声明 `scalar_t` 类型数组 `values`，并将 `val` 的数据存储到其中
    alignas(64) scalar_t values[vec_t::size()];
    val.store(values);

    // 声明 `acc_t` 类型数组 `acc`，并用 `values` 中的数据初始化
    alignas(64) acc_t acc[vacc_t::size()];
    for (const auto i : c10::irange(vacc_t::size())) {
      acc[i] = values[i];
    }

    // 使用 `vacc_t::loadu` 加载 `acc` 中的数据并返回结果
    return vacc_t::loadu(acc);
  }
};

// 定义模板结构体 `OuterSumCastLoadPolicy` 的另一个特化版本，
// 当 `vec_t` 是简化的浮点类型时生效
template <typename vec_t, typename vacc_t>
struct OuterSumCastLoadPolicy <vec_t, vacc_t, std::enable_if_t<is_reduced_floating_point_v<vechold_type<vec_t>>>> {
  // 定义类型别名 `scalar_t` 作为 `vec_t` 的元素类型
  using scalar_t = vechold_type<vec_t>;

  // 返回 `sizeof(scalar_t) * vacc_t::size()` 的结果作为内存大小的常量表达式
  static constexpr int64_t memsize() {
    return sizeof(scalar_t) * vacc_t::size();
  }

  // 加载函数 `load`，接受指向数据的 `data` 指针、步长 `stride` 和索引 `index`
  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    // 将 `data` 指针解释为 `scalar_t` 类型指针
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    // 声明 `vacc_t` 类型的变量 `values`
    vacc_t values;
    // 使用 `load_to_float` 函数加载 `ptr` 指向的数据，并存储到 `values`
    vec::load_to_float<scalar_t>(ptr, values);
    // 返回加载的 `values`
    return values;
  }
};

// 当 `vec_t` 和 `vacc_t` 相同时的偏特化，继承自 `LoadPolicy<scalar_t>`
template <typename scalar_t>
struct OuterSumCastLoadPolicy<scalar_t, scalar_t, void>:
    LoadPolicy<scalar_t> {
};
/* 实现 nansum 时，扩展加载操作以在进入常规求和循环之前屏蔽 NaN 值 */

template <typename scalar_t>
struct NanSumLoadPolicy {
  // 返回标量类型的内存大小
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  // 加载数据，如果是 NaN 则返回 0，否则返回值本身
  static scalar_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto val = LoadPolicy<scalar_t>::load(data, stride, index);
    return at::_isnan(val) ? scalar_t(0) : val;
  }
};

template <typename scalar_t>
struct NanSumLoadPolicy<Vectorized<scalar_t>> {
  using vec_t = Vectorized<scalar_t>;

  // 返回矢量类型的内存大小
  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  // 加载矢量数据，使用混合运算（blendv）屏蔽 NaN 值，替换为 0
  static vec_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto val = LoadPolicy<vec_t>::load(data, stride, index);
    return vec_t::blendv(val, vec_t(0), val.isnan());
  }
};

template <typename scalar_t, typename acc_t>
struct NanSumCastLoadPolicy {
  // 返回标量类型到累加器类型的内存大小
  static constexpr int64_t memsize() {
    return sizeof(scalar_t);
  }

  // 加载数据，如果是 NaN 则返回 0，否则返回值本身
  static acc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto val = CastLoadPolicy<scalar_t, acc_t>::load(data, stride, index);
    return at::_isnan(val) ? acc_t(0) : val;
  }
};

template <typename vec_t, typename vacc_t, typename = void>
struct InnerNanSumCastLoadPolicy;

template <typename vec_t, typename vacc_t>
struct InnerNanSumCastLoadPolicy <vec_t, vacc_t,
  std::enable_if_t<(!is_reduced_floating_point_v<vechold_type<vec_t>>) &&
                    !std::is_same_v<vec_t, vacc_t>>> {
  using scalar_t = vechold_type<vec_t>;
  using acc_t = vechold_type<vacc_t>;

  // 返回矢量类型到累加器类型的内存大小
  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  // 加载矢量数据并进行归约操作，屏蔽 NaN 值后求和
  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    return load_reduce_vec<acc_t>(ptr, [](acc_t a, scalar_t b) {
      return at::_isnan(b) ? a : a + b;
    }, acc_t(0));
  }
};

template <typename scalar_t>
struct InnerNanSumCastLoadPolicy<scalar_t, scalar_t, void>:
    NanSumLoadPolicy<scalar_t> {
};

template <typename vec_t, typename vacc_t>
struct InnerNanSumCastLoadPolicy <vec_t, vacc_t, std::enable_if_t<is_reduced_floating_point_v<vechold_type<vec_t>>>> {
  using scalar_t = vechold_type<vec_t>;

  // 返回矢量类型到累加器类型的内存大小
  static constexpr int64_t memsize() {
    return LoadPolicy<vec_t>::memsize();
  }

  // 加载矢量数据并进行浮点数到累加器类型的转换，屏蔽 NaN 值后求和
  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto ptr = reinterpret_cast<const scalar_t*>(data + stride * index);
    vacc_t first, second;
    vec::load_to_float<scalar_t>(ptr, first, second);
    const vacc_t zero(0);
    return (vacc_t::blendv(first, zero, first.isnan()) +
            vacc_t::blendv(second, zero, second.isnan()));
  }
};

template <typename vec_t, typename vacc_t>
struct OuterNanSumCastLoadPolicy {
  // 返回矢量类型到累加器类型的内存大小
  static constexpr int64_t memsize() {
    // 调用 OuterSumCastLoadPolicy 模板类的静态成员函数 memsize()，返回其计算出的内存大小
    return OuterSumCastLoadPolicy<vec_t, vacc_t>::memsize();
  }

  // 静态成员函数 load，接收指向数据的指针、步幅和索引，加载数据并处理 NaN 值
  static vacc_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    // 调用 OuterSumCastLoadPolicy 模板类的静态成员函数 load，加载数据并返回处理后的结果
    auto val = OuterSumCastLoadPolicy<vec_t, vacc_t>::load(data, stride, index);
    // 使用 vacc_t 类的 blendv() 方法，将 NaN 值替换为 0，并返回处理后的结果
    return vacc_t::blendv(val, vacc_t(0), val.isnan());
  }


这段代码是 C++ 中的静态成员函数的定义和调用示例。
};

// 定义一个结构体模板 CastStoreAccumulate，用于将值累加到指定地址
template <typename scalar_t, typename acc_t>
struct CastStoreAccumulate {
  // 静态函数 store，将累加值存储到指定地址
  static void store(char * C10_RESTRICT data, int64_t stride, int64_t index, acc_t value) {
    // 将 data 强制转换为 scalar_t* 类型的指针，然后在偏移 index * stride 处累加 value
    auto * ptr = reinterpret_cast<scalar_t*>(data + index * stride);
    *ptr += value;
  }
};

// 静态函数 store，根据给定的存储策略 StorePolicy 存储值到指定地址
template <typename StorePolicy, typename scalar_t>
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index, scalar_t value) {
  StorePolicy::store(data, stride, index, value);
}

// 静态函数 store，根据给定的存储策略 StorePolicy，将数组 values 的值存储到指定地址
template <typename StorePolicy, typename scalar_t, size_t numel>
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index,
                  const std::array<scalar_t, numel> &values) {
  auto *base_ptr = data + stride * index;
  // 遍历数组 values，根据存储策略 StorePolicy 将每个值存储到指定位置
  for (const auto k : c10::irange(numel)) {
    auto val = values[k];
    StorePolicy::store(base_ptr, stride, k, val);
  }
}

// 静态函数 store，根据给定的存储策略 StorePolicy，将向量化数据 values 存储到指定地址
template <typename StorePolicy, typename scalar_t>
static void store(char * C10_RESTRICT data, int64_t stride, int64_t index,
                  const Vectorized<scalar_t> &values) {
  using vec_t = Vectorized<scalar_t>;
  alignas(64) std::array<scalar_t, vec_t::size()> array_values;
  // 将向量化数据 values 存储到 array_values 数组中
  values.store(array_values.data());
  // 调用前面定义的 store 函数，根据存储策略 StorePolicy 将 array_values 的值存储到指定地址
  store<StorePolicy>(data, stride, index, array_values);
}

/** Simultaneously sum over n rows at once

This algorithm calculates the sum without loss of precision over large axes. It
does this by chunking the sum into groups of 16 or more elements. The sums of
these chunks are also summed in chunks and so on until there is just a single sum
value remaining. This means only numbers of a similar order of magnitude are
added together, thus minimising rounding errors.

This is done in a single linear pass over the data and with O(1) extra storage.
A simplified recursive implementation would look like this:

  scalar_t row_sum(const scalar_t * data, int64_t n) {
    // Note, in practice the chunk size can increase with n
    // This allows the recursion depth to be limited to O(1).
    constexpr int64_t min_chunk_size = 16;

    scalar_t sum = 0;
    if (n <= min_chunk_size) {
      // Recursive base case, calculate a simple running sum
      for (const auto i : c10::irange(n)) {
        sum += data[i];
      }
      return sum;
    }

    // Recursively sum larger chunks of elements
    const int64_t chunk_size = std::max(divup(n, min_chunk_size), min_chunk_size);
    for (int64_t i = 0; i < n; i += chunk_size) {
      sum += row_sum(data + i, std::min(chunk_size, n - i));
    }
    return sum;
  }
*/
// 定义一个模板函数 multi_row_sum，用于同时对 n 行进行求和
template <typename scalar_t, int64_t nrows, typename LoadPolicy>
std::array<scalar_t, nrows> multi_row_sum(
    const char * C10_RESTRICT in_data,
    const int64_t row_stride,
    const int64_t col_stride,
    // 定义函数，将输入数据流按指定行步进进行累加并返回累加结果数组
    const int64_t size) {
      // 定义层级数为4
      constexpr int64_t num_levels = 4;
    
      // 计算每个层级的步长
      const int64_t level_power =
          std::max(int64_t(4), utils::CeilLog2(size) / num_levels);
      const int64_t level_step = (1 << level_power);
      const int64_t level_mask = level_step - 1;
    
      // 声明一个二维数组用于累加，每个层级有 nrows 个元素
      // 并将其所有元素初始化为零
      scalar_t acc[num_levels][nrows];
      std::fill_n(&acc[0][0], num_levels * nrows, scalar_t(0));
    
      // 初始化循环变量 i 为 0，开始累加过程
      int64_t i = 0;
      for (; i + level_step <= size;) {
        // 按层级步长累加输入数据流的内容
        for (int64_t j = 0; j < level_step; ++j, ++i) {
          // 计算当前数据位置的指针
          const char * sum_base = in_data + i * row_stride;
    
          // 对每一行进行累加，使用 LoadPolicy::load 方法加载数据
          #if !defined(COMPILING_FOR_MIN_SIZE)
          # pragma unroll
          #endif
          for (const auto k : c10::irange(nrows)) {
            acc[0][k] += LoadPolicy::load(sum_base, col_stride, k);
          }
        }
    
        // 对每个层级进行累加
        for (const auto j : c10::irange(1, num_levels)) {
          #if !defined(COMPILING_FOR_MIN_SIZE)
          # pragma unroll
          #endif
          for (const auto k : c10::irange(nrows)) {
            acc[j][k] += acc[j-1][k];  // 当前层级累加上一层级的结果
            acc[j-1][k] = scalar_t(0); // 清零上一层级的累加结果
          }
    
          // 根据层级掩码检查是否需要跳出当前累加层级的循环
          const auto mask = (level_mask << (j * level_power));
          if ((i & mask) != 0) {
            break;
          }
        }
      }
    
      // 处理剩余的未完全累加的数据
      for (; i < size; ++i) {
        const char * sum_base = in_data + i * row_stride;
        #if !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (const auto k : c10::irange(nrows)) {
          acc[0][k] += LoadPolicy::load(sum_base, col_stride, k);
        }
      }
    
      // 最后将所有层级的累加结果累加到第一层级，并返回结果
      for (const auto j : c10::irange(1, num_levels)) {
        #if !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (const auto k : c10::irange(nrows)) {
          acc[0][k] += acc[j][k];
        }
      }
    
      // 声明返回结果数组，并将累加结果复制到返回数组中
      std::array<scalar_t, nrows> ret;
      for (const auto k : c10::irange(nrows)) {
        ret[k] = acc[0][k];
      }
      return ret;
    }
// 结束之前的 C++ 函数定义

template <typename scalar_t, typename LoadPolicy>
scalar_t row_sum(const char * C10_RESTRICT in_data,
                 const int64_t in_stride, const int64_t size) {
  // 定义 ILP 因子为 4
  constexpr int64_t ilp_factor = 4;

  // 将行视为 (-1, ilp_factor) 形状的数组以找到部分和
  const int64_t size_ilp = size / ilp_factor;
  // 调用 multi_row_sum 函数获取部分和
  auto partial_sums = multi_row_sum<scalar_t, ilp_factor, LoadPolicy>(
      in_data, in_stride * ilp_factor, in_stride, size_ilp);

  // 处理剩余的元素，计算最终的部分和
  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_sums[0] += LoadPolicy::load(in_data, in_stride, i);
  }

  // 将 ILP 因子中的部分和相加得到最终的行和
  for (const auto k : c10::irange(1, ilp_factor)) {
    partial_sums[0] += partial_sums[k];
  }

  // 返回最终的行和
  return partial_sums[0];
}

template <typename acc_t, typename VecLoadPolicy, typename ScalarLoadPolicy, typename StorePolicy>
void vectorized_inner_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t outer_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  // 使用 Vectorized 类型的 acc_t
  using vacc_t = Vectorized<acc_t>;
  // 确定向量化加载策略和标量加载策略的步长
  constexpr int64_t vec_stride = VecLoadPolicy::memsize();
  constexpr int64_t scalar_stride = ScalarLoadPolicy::memsize();
  // 计算向量的元素个数
  constexpr int64_t vec_numel = vec_stride / scalar_stride;
  // 计算可以向量化处理的大小
  const int64_t vec_size = size0 / vec_numel;

  // 对第一个（较小的）维度上的输入进行连续处理
  for (const auto j : c10::irange(size1)) {
    // 获取当前行的输入数据指针
    const auto *row_in = data[1] + j * outer_stride;
    // 计算当前行的向量化累加和
    auto vec_acc = row_sum<vacc_t, VecLoadPolicy>(row_in, vec_stride, vec_size);

    // 初始化最终的累加和为零
    acc_t final_acc = 0;
    // 处理剩余的元素，计算最终的累加和
    for (int64_t k = vec_size * vec_numel; k < size0; ++k) {
      final_acc += ScalarLoadPolicy::load(row_in, scalar_stride, k);
    }

    // 对向量化累加和进行存储
    alignas(64) std::array<acc_t, vacc_t::size()> partials{};
    vec_acc.store(partials.data());
    // 将向量化累加和中的部分和相加，得到最终的累加和
    for (const auto k : c10::irange(partials.size())) {
      final_acc += partials[k];
    }
    // 使用存储策略将最终的累加和存储到输出数据中
    store<StorePolicy>(data[0], out_stride, j, final_acc);
  }
}

template <typename acc_t, typename LoadPolicy, typename StorePolicy>
void scalar_inner_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {
  // 对第二个维度进行循环处理
  for (const auto j : c10::irange(size1)) {
    // 获取当前行的输入数据指针
    const auto *row_in = data[1] + j * in_strides[1];
    // 计算当前行的标量累加和
    auto ans = row_sum<acc_t, LoadPolicy>(row_in, in_strides[0], size0);
    // 使用存储策略将累加和存储到输出数据中
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

template <typename acc_t, typename VecLoadPolicy, typename ScalarLoadPolicy, typename StorePolicy>
void vectorized_outer_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t inner_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
    int64_t size0, int64_t size1) {
  using vacc_t = Vectorized<acc_t>;
  constexpr int64_t scalar_stride = ScalarLoadPolicy::memsize();
  constexpr int64_t vec_stride = VecLoadPolicy::memsize();
  constexpr int64_t nrows = 4;

  // Input is contiguous over the second (non-reduced) dimension
  // 初始化变量 j 为 0，用于迭代处理数据的第二维度
  int64_t j = 0;
  // 对于每个 nrows 行的数据块，处理数据直到不足 nrows 行
  for (; j + nrows * vacc_t::size() <= size1; j += nrows * vacc_t::size()) {
    // 获取指向当前处理数据行起始的指针
    const auto *row_in = data[1] + j * scalar_stride;
    // 计算多行的和并存储在 sums 中
    auto sums = multi_row_sum<vacc_t, nrows, VecLoadPolicy>(
        row_in, inner_stride, vec_stride, size0);

    // 遍历每行的结果 sums，将其存储到输出数据中
    for (const auto i : c10::irange(nrows)) {
      // 计算基础索引，用于存储当前行的结果
      const int64_t base_idx = j + i * vacc_t::size();
      // 将 sums[i] 存储到输出数据中的指定位置
      store<StorePolicy>(data[0], out_stride, base_idx, sums[i]);
    }
  }

  // 处理剩余不足 nrows 行但至少能处理一个向量的数据块
  for (; j + vacc_t::size() <= size1; j += vacc_t::size()) {
    // 获取指向当前处理数据行起始的指针
    const auto *row_in = data[1] + j * scalar_stride;
    // 计算单行的和并存储在 sums 中
    const vacc_t sums = row_sum<vacc_t, VecLoadPolicy>(
        row_in, inner_stride, size0);

    // 将 sums 存储到输出数据中的指定位置
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    store<StorePolicy>(data[0], out_stride, j, sums);
  }

  // 处理剩余的数据，每次处理一行
  for (; j < size1; ++j) {
    // 获取指向当前处理数据行起始的指针
    const auto *row_in = data[1] + j * scalar_stride;
    // 计算单行的和并存储在 ans 中
    auto ans = row_sum<acc_t, ScalarLoadPolicy>(row_in, inner_stride, size0);
    // 将 ans 存储到输出数据中的指定位置
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}
}  // 结束匿名命名空间

template <typename acc_t, typename LoadPolicy, typename StorePolicy>
void scalar_outer_sum(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {
  // 定义常量，表示每次处理的行数
  constexpr int64_t nrows = 4;
  // 初始化列索引
  int64_t j = 0;
  // 循环处理每一列，每次处理 nrows 行
  for (; j + (nrows - 1) < size1; j += nrows) {
    // 获取当前行的输入数据指针
    const auto *row_in = data[1] + j * in_strides[1];
    // 调用 multi_row_sum 函数计算多行的和，根据 LoadPolicy 指定加载策略
    auto sums = multi_row_sum<acc_t, nrows, LoadPolicy>(
        row_in, in_strides[0], in_strides[1], size0);
    // 调用 store 函数根据 StorePolicy 将计算结果存储到 data[0] 中
    store<StorePolicy>(data[0], out_stride, j, sums);
  }

  // 处理剩余的列，每次处理一行
  for (; j < size1; ++j) {
    // 获取当前行的输入数据指针
    const auto *row_in = data[1] + j * in_strides[1];
    // 调用 row_sum 函数计算单行的和，根据 LoadPolicy 指定加载策略
    auto ans = row_sum<acc_t, LoadPolicy>(
        row_in, in_strides[0], size0);
    // 调用 store 函数根据 StorePolicy 将计算结果存储到 data[0] 中
    store<StorePolicy>(data[0], out_stride, j, ans);
  }
}

// 自定义浮点数求和，以提高精度
template <bool ignore_nan, typename scalar_t>
void cascade_sum(TensorIterator &iter) {
  // 将输出张量的基础数据填充为 0
  iter.output_base().fill_(scalar_t(0));
  // 并行化计算过程
  iter.parallel_reduce(
    });
}

void sum_kernel_impl(TensorIterator &iter) {
  // 如果数据类型是整数类型（包括布尔类型）
  if (isIntegralType(iter.dtype(), /*includeBool=*/ true)) {
    // 调度整数类型的求和计算
    AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, iter.dtype(), "sum_cpu",
      [&] {
        // 调用 binary_kernel_reduce_vec 函数执行向量化的二元操作
        binary_kernel_reduce_vec(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
            [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a + b; });
      });
    return;
  }

  // 对于浮点数和复数类型（不包括 Float16 和 Half）
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "sum_cpu", [&] {
    // 调用 cascade_sum 函数进行浮点数求和计算，ignore_nan=false
    cascade_sum</*ignore_nan=*/false, scalar_t>(iter);
  });
}

void nansum_kernel_impl(TensorIterator &iter) {
  // 对于浮点数类型（不包括 Float16 和 Half）
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "nansum_cpu", [&] {
    // 调用 cascade_sum 函数进行浮点数求和计算，ignore_nan=true
    cascade_sum</*ignore_nan=*/true, scalar_t>(iter);
  });
}

}  // namespace (anonymous)

// 对于 Float16 类型，AVX2 下精度较差，AVX512 更甚。在修复之前，不使用 AVX512 调度。参见 GH 59415。
// 此外，这些内核在 AVX512 下比 AVX2 更慢。
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_impl);
REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);

}  // namespace at::native
```