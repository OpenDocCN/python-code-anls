# `.\pytorch\aten\src\ATen\native\cpu\TensorCompareKernel.cpp`

```py
// 包含C++头文件，用于张量操作和计算
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorCompare.h>

#include <numeric>    // 数值计算库
#include <iterator>   // 迭代器库
#include <algorithm>  // 算法库
#include <utility>    // 实用工具库
#include <vector>     // 向量容器库

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/Loops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/result_type.h>
#endif

namespace at::native { namespace {

// 模板函数，执行张量比较的核心操作，处理非空向量和维度重塑
template <typename scalar_t, typename scalar_t_2 = int64_t, typename loop1d_t>
static inline void compare_base_kernel_core(
    const Tensor& result1,
    const Tensor& result2,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const loop1d_t& loop) {
  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());  // 获取非空向量的张量尺寸
  self_sizes[dim] = 1;  // 将指定维度的大小设置为1，用于保持维度的情况

  // 如果不保持维度，则将 result1 和 result2 重新调整维度以匹配 self
  if (!keepdim) {
    if (result1.ndimension() >= dim) {
      result1.unsqueeze_(dim);
    }
    if (result2.ndimension() >= dim) {
      result2.unsqueeze_(dim);
    }
  }

  at::native::resize_output(result1, self_sizes);  // 调整 result1 的输出大小以匹配 self_sizes
  at::native::resize_output(result2, self_sizes);  // 调整 result2 的输出大小以匹配 self_sizes

  // 创建张量迭代器，配置静态形状和输出
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)  // 不检查所有张量是否具有相同的数据类型
    .resize_outputs(false)        // 不调整输出大小
    .declare_static_shape(self.sizes(), /*squash_dims=*/dim)  // 声明静态形状，并且在指定维度上压缩
    .add_output(result1)  // 添加输出 result1
    .add_output(result2)  // 添加输出 result2
    .add_const_input(self)  // 添加常量输入 self
    .build();  // 构建迭代器

  iter.for_each(loop, /* grain_size */ 1);  // 使用指定的循环函数和粒度大小执行迭代

  // 如果不保持维度，则将 result1 和 result2 在指定维度上挤压
  if (!keepdim) {
    result1.squeeze_(dim);
    result2.squeeze_(dim);
  }
}

// 模板函数，执行张量比较的核心操作，处理标量和功能函数
template <typename scalar_t, typename scalar_t_2=int64_t, typename func_t>
static inline void compare_base_kernel(const Tensor& result1, const Tensor& result2,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const func_t& f) {

  auto self_dim_stride = ensure_nonempty_stride(self, dim);  // 确保非空维度的步长

  // 定义循环函数，处理数据指针和步长，执行功能函数
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result1_data_bytes = data[0];  // result1 数据指针
    auto* result2_data_bytes = data[1];  // result2 数据指针
    const auto* self_data_bytes = data[2];  // self 数据指针
    for (const auto i C10_UNUSED : c10::irange(n)) {  // 对于范围 n 执行循环
      f((scalar_t*)result1_data_bytes,
        (scalar_t_2*)result2_data_bytes,
        (scalar_t*)self_data_bytes,
        self_dim_stride);  // 调用功能函数 f 处理数据
      result1_data_bytes += strides[0];  // 更新 result1 数据指针
      result2_data_bytes += strides[1];  // 更新 result2 数据指针
      self_data_bytes += strides[2];     // 更新 self 数据指针
    }
  };

  compare_base_kernel_core<scalar_t, scalar_t_2>(
      result1, result2, self, dim, keepdim, loop);  // 执行核心比较操作
}

// 实现最小化函数的内核，处理结果、指数和自身张量，指定维度
static void min_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  // 确保维度非空，获取 self 在指定维度上的大小
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  // 使用 AT_DISPATCH_ALL_TYPES_AND3 宏处理所有类型（包括半精度、BF16 和布尔类型），执行 min_cpu 操作
  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "min_cpu", [&] {
    // 定义比较基础的内核函数，针对指定的数据类型进行操作
    compare_base_kernel<scalar_t>(result, indice, self, dim, keepdim, [&] (
      scalar_t* result_data, int64_t* indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        // 定义值类型，根据 scalar_t 类型获取实际的数值类型
        using value_t = typename c10::scalar_value_type<scalar_t>::type;
        // 定义 zabs_ 函数指针，用于获取值的绝对值
        value_t (*zabs_)(scalar_t) = zabs<scalar_t, value_t>;
        // 初始化最小数值为 self_data 的第一个元素
        scalar_t min_number = c10::load(self_data);
        // 初始化索引为 0
        int64_t index = 0;
        // 遍历指定维度上的所有元素
        for (const auto i : c10::irange(self_dim_size)) {
          // 获取当前元素的值
          scalar_t value = self_data[i * self_dim_stride];
          // 如果当前值的绝对值不大于最小数值的绝对值，则更新最小数值和对应的索引
          if (!(zabs_(value) >= zabs_(min_number))) {
            min_number = value;
            index = i;
            // 如果当前值是 NaN，跳出循环
            if (_isnan<scalar_t>(value)) {
              break;
            }
          }
        }
        // 将找到的最小数值和对应的索引存入结果中
        *result_data = min_number;
        *indice_data = index;
      }
    );
  });
}

// 实现了在指定维度上计算最大值的函数
static void max_kernel_impl(
    // 存储最大值结果的张量
    const Tensor& result,
    // 存储最大值索引的张量
    const Tensor& indice,
    // 输入的张量
    const Tensor& self,
    // 操作的维度
    int64_t dim,
    // 是否保持维度
    bool keepdim) {
  // 确定指定维度上张量的大小
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  // 根据张量的数据类型分发计算最大值的内核函数
  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "max_cpu", [&] {
    // 调用比较基础内核函数，用于计算最大值和索引
    compare_base_kernel<scalar_t>(result, indice, self, dim, keepdim, [&] (
      // 存储结果数据的指针
      scalar_t* result_data, 
      // 存储索引数据的指针
      int64_t* indice_data,
      // 输入张量数据的指针
      const scalar_t* self_data, 
      // 自身维度步长的自动推断类型
      auto self_dim_stride) {
        // 定义标量类型
        using value_t = typename c10::scalar_value_type<scalar_t>::type;
        // 函数指针，用于获取标量的绝对值
        value_t (*zabs_)(scalar_t) = zabs<scalar_t, value_t>;
        // 初始化最大数为输入数据的第一个元素
        scalar_t max_number = c10::load(self_data);
        // 初始化索引为0
        int64_t index = 0;
        // 遍历指定维度上的所有元素
        for (const auto i : c10::irange(self_dim_size)) {
          // 获取当前元素的值
          scalar_t value = c10::load(&self_data[i * self_dim_stride]);
          // 如果当前值的绝对值大于最大值的绝对值，则更新最大值和索引
          if (!(zabs_(value) <= zabs_(max_number))) {
            max_number = value;
            index = i;
            // 如果当前值是 NaN，则终止循环
            if (_isnan<scalar_t>(value)) {
              break;
            }
          }
        }
        // 将计算得到的最大值和索引存入结果中
        *result_data = max_number;
        *indice_data = index;
      }
    );
  });
}

// 实现了在指定维度上计算最小值和最大值的函数
static void aminmax_kernel(
    // 输入的张量
    const Tensor& self,
    // 操作的维度
    int64_t dim,
    // 是否保持维度
    bool keepdim,
    // 存储最小值结果的张量
    Tensor& min_result,
    // 存储最大值结果的张量
    Tensor& max_result) {
  // 可能对维度进行包装以适应张量的大小
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  // 确定指定维度上张量的大小
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  // 检查结果张量的数据类型是否与输入张量的数据类型相同
  TORCH_CHECK(min_result.scalar_type() == self.scalar_type() && max_result.scalar_type() == self.scalar_type(),
    "Expect min and max dtype ", self.scalar_type(),
    " but got ", min_result.scalar_type(), " and ", max_result.scalar_type());

  // 如果张量是标量且维度为0，则特殊处理
  if (self.numel() == 1 && self.ndimension() == 0) {
    // 检查复数类型是否支持 aminmax 计算
    TORCH_CHECK(!self.is_complex(), "aminmax not implemented for ", self.scalar_type());
    // 调整结果张量的大小为标量
    min_result.resize_({});
    max_result.resize_({});
    // 将结果张量填充为输入的标量值
    min_result.fill_(self);
    max_result.fill_(self);
    return;
  }

  // 根据张量的数据类型分发计算最小值和最大值的内核函数
  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "aminmax_cpu", [&] {
    // 调用比较基础内核函数，用于计算最小值和最大值
    compare_base_kernel<scalar_t, scalar_t>(min_result, max_result, self, wrap_dim, keepdim, [&] (
      // 存储最小值结果的指针
      scalar_t* min_result_data, 
      // 存储最大值结果的指针
      scalar_t* max_result_data,
      // 输入张量数据的指针
      const scalar_t* self_data, 
      // 自身维度步长的自动推断类型
      auto self_dim_stride) {
        // 初始化最小数为输入数据的第一个元素
        scalar_t min_number = c10::load(self_data);
        // 初始化最大数为最小数
        scalar_t max_number = min_number;
        // 遍历指定维度上的所有元素
        for (const auto i : c10::irange(self_dim_size)) {
          // 获取当前元素的值
          scalar_t value = c10::load(&self_data[i * self_dim_stride]);
          // 如果当前值小于最小值，则更新最小值
          if (!(value >= min_number)) {
            min_number = value;
            // 如果当前值是 NaN，则同时将最大值更新为当前值并终止循环
            if (_isnan<scalar_t>(value)) {
              max_number = value;
              break;
            }
          // 否则，如果当前值大于最大值，则更新最大值
          } else if (!(value <= max_number)) {
            max_number = value;
          }
        }
        // 将计算得到的最小值和最大值存入结果中
        *min_result_data = min_number;
        *max_result_data = max_number;
      }
    );
  });
}
// 实现了在TensorIterator上执行的where操作，根据条件值选择self_val或other_val
static void where_kernel_impl(TensorIterator &iter) {
  // 根据TensorIterator的数据类型分发CPU操作，处理所有标量类型和复数类型以及特定类型Half、BFloat16和Bool
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBFloat16, kBool,
    iter.dtype(), "where_cpu", [&] {
      // 在CPU上执行核心操作，根据条件值cond_val返回self_val或other_val的值
      cpu_kernel(
        iter,
        [=](bool cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
  });
}

// 实现了在TensorIteratorBase上执行的isposinf操作，检查浮点数是否正无穷大
static void isposinf_kernel_impl(TensorIteratorBase& iter) {
  // 根据TensorIteratorBase的输入数据类型分发CPU操作，处理所有浮点数类型和Half、BFloat16类型
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cpu", [&]() {
    // 在CPU上执行核心操作，判断浮点数a是否等于正无穷大
    cpu_kernel(iter, [](scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); });
  });
}

// 实现了在TensorIteratorBase上执行的isneginf操作，检查浮点数是否负无穷大
static void isneginf_kernel_impl(TensorIteratorBase& iter) {
  // 根据TensorIteratorBase的输入数据类型分发CPU操作，处理所有浮点数类型和Half、BFloat16类型
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cpu", [&]() {
    // 在CPU上执行核心操作，判断浮点数a是否等于负无穷大
    cpu_kernel(iter, [](scalar_t a) -> bool { return a == -std::numeric_limits<scalar_t>::infinity(); });
  });
}

// 实现了在values和indices张量上执行的mode操作，计算指定维度上的众数
static void mode_kernel_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    // 确保在指定维度上的张量尺寸不为空，返回尺寸大小
    auto self_dim_size = ensure_nonempty_size(self, dim);
    // 确保在指定维度上的张量步长不为空，返回步长大小
    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    
    // 根据张量的数据类型调度处理器，对张量进行模式计算
    AT_DISPATCH_ALL_TYPES_AND3(
        kHalf, kBFloat16, kBool, self.scalar_type(), "mode_cpu", [&] {
          
          // 定义循环操作 lambda 函数，处理每个张量的数据和索引
          auto loop = [&](char** data, const int64_t* strides, int64_t n) {
            // 分别获取值、索引和自身数据的字节指针
            auto* values_data_bytes = data[0];
            auto* indices_data_bytes = data[1];
            const auto* self_data_bytes = data[2];
    
            // 创建存储元素值和对应索引的向量，大小为 self_dim_size
            std::vector<std::pair<scalar_t, int64_t>> elements(self_dim_size);
    
            // 遍历张量的每个元素，将元素值与索引存入 elements 向量中
            for (const auto k C10_UNUSED : c10::irange(n)) {
              scalar_t* values_data = (scalar_t*)values_data_bytes;
              int64_t* indices_data = (int64_t*)indices_data_bytes;
              const scalar_t* self_data = (scalar_t*)self_data_bytes;
    
              scalar_t mode = 0;
              int64_t modei = 0;
              int64_t temp_freq = 0;
              int64_t max_freq = 0;
    
              // 将张量中的元素值与对应的索引存入 elements 向量
              for (const auto i : c10::irange(self_dim_size)) {
                elements[i] = std::make_pair(c10::load(&self_data[i * self_dim_stride]), i);
              }
    
              // 使用 std::sort 对 elements 向量进行排序，按元素值升序排列
              std::sort(
                  elements.begin(),
                  elements.end(),
                  [=](const auto& i, const auto& j) {
                    return i.first < j.first;
                  });
    
              // 计算元素的众数及其出现次数
              for (const auto i : c10::irange(self_dim_size)) {
                temp_freq++;
                if ((i == self_dim_size - 1) ||
                    (elements[i].first != elements[i + 1].first)) {
                  if (temp_freq > max_freq) {
                    mode = elements[i].first;
                    modei = elements[i].second;
                    max_freq = temp_freq;
                  }
                  temp_freq = 0;
                }
              }
    
              // 将计算得到的众数和对应的索引存入指定地址中
              *values_data = mode;
              *indices_data = modei;
    
              // 更新字节指针，指向下一个元素的数据
              values_data_bytes += strides[0];
              indices_data_bytes += strides[1];
              self_data_bytes += strides[2];
            }
          };
    
          // 调用核心比较函数，传递参数进行处理
          compare_base_kernel_core<scalar_t>(
              values, indices, self, dim, keepdim, loop);
        });
// 默认的暴力实现isin()函数，用于测试元素数量较小时使用。
// 遍历每个元素并与每个测试元素进行比较。
static void isin_default_kernel_cpu(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out) {
  // 因为test_elements不是TensorIterator的输入，需要手动进行类型提升。
  ScalarType common_type = at::result_type(elements, test_elements);
  // 将elements提升为公共类型。
  Tensor promoted_elements = elements.to(common_type);
  // 将test_elements扁平化为一维，并提升为公共类型。
  Tensor test_elements_flat = test_elements.to(common_type).view(-1);
  auto test_elements_stride = test_elements_flat.stride(0);

  // 创建TensorIteratorConfig对象，配置输出、输入等参数。
  auto iter = TensorIteratorConfig()
    .add_output(out)
    .add_const_input(promoted_elements)
    .check_all_same_dtype(false)
    .build();
  // 根据提升后的类型分发处理。
  AT_DISPATCH_ALL_TYPES(iter.dtype(1), "isin_default_cpu", [&]() {
    cpu_kernel(iter, [&](scalar_t element_val) -> bool {
      // 获取test_elements_flat的常量数据指针。
      const auto* test_element_data = test_elements_flat.const_data_ptr<scalar_t>();
      // 遍历test_elements_flat中的元素。
      for (const auto j : c10::irange(test_elements_flat.numel())) {
        // 如果element_val等于test_elements_flat中的某个元素，则根据invert返回结果。
        if (element_val == *(test_element_data + test_elements_stride * j)) {
          return !invert;
        }
      }
      // 如果未找到匹配元素，则根据invert返回结果。
      return invert;
    });
  });
}

// clamp操作的实现函数，处理TensorIteratorBase对象。
static void clamp_kernel_impl(TensorIteratorBase& iter) {
  // 根据iter的公共类型分发处理。
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_cpu", [&]() {
    cpu_kernel_vec(iter,
      // 标量操作，对每个元素执行clamp。
      [](scalar_t a, scalar_t min, scalar_t max) -> scalar_t {
        // 如果min或max为NaN，则返回quiet NaN。
        if (min != min || max != max) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        } else {
            // 否则返回a在[min, max]范围内的值。
            return std::min(std::max(a, min), max);
        }
      },
      // 向量化操作，对每个向量执行clamp。
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> min, Vectorized<scalar_t> max) {
        return vec::minimum(vec::maximum(a, min), max);
      });
  });
}

// clamp_scalar操作的实现函数，处理TensorIteratorBase对象和标量min和max。
static void clamp_scalar_kernel_impl(TensorIteratorBase& iter, const Scalar& min_, const Scalar& max_) {
  // 根据iter的公共类型分发处理。
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_scalar_cpu", [&]() {
    const auto min = min_.to<scalar_t>();
    const auto max = max_.to<scalar_t>();
    const Vectorized<scalar_t> min_vec(min);
    const Vectorized<scalar_t> max_vec(max);
    cpu_kernel_vec(iter,
      // 标量操作，对每个元素执行clamp。
      [=](scalar_t a) -> scalar_t {
        return std::min(std::max(a, min), max);
      },
      // 向量化操作，对每个向量执行clamp。
      [=](Vectorized<scalar_t> a) {
        return vec::clamp(a, min_vec, max_vec);
      });
  });
}

// clamp_max_scalar操作的实现函数，处理TensorIteratorBase对象和标量max。
static void clamp_max_scalar_kernel_impl(TensorIteratorBase& iter, Scalar max_) {
  // 根据iter的公共类型分发处理。
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_max_scalar_cpu", [&]() {
    const auto max = max_.to<scalar_t>();
    const Vectorized<scalar_t> max_vec(max);
    cpu_kernel_vec(iter,
      // 标量操作，对每个元素执行clamp。
      [=](scalar_t a) -> scalar_t {
        return std::min(a, max);
      },
      // 向量化操作，对每个向量执行clamp。
      [=](Vectorized<scalar_t> a) {
        return vec::clamp_max(a, max_vec);
      });
  });
}
// 实现 clamp_min_scalar_kernel_impl 函数，对 TensorIteratorBase 迭代器执行最小值截断操作
static void clamp_min_scalar_kernel_impl(TensorIteratorBase& iter, Scalar min_) {
  // 使用宏 AT_DISPATCH_ALL_TYPES_AND2 遍历所有数据类型和两个特定类型（kBFloat16 和 kHalf）
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_min_scalar_cpu", [&]() {
    // 将 Scalar 类型的 min_ 转换为当前数据类型 scalar_t
    const auto min = min_.to<scalar_t>();
    // 使用 Vectorized 类型封装 min 的向量化版本
    const Vectorized<scalar_t> min_vec(min);
    // 调用 cpu_kernel_vec 函数对 iter 迭代器执行向量化计算
    cpu_kernel_vec(iter,
        // 对每个元素执行 a 和 min 的最大值操作
        [=](scalar_t a) -> scalar_t {
          return std::max(a, min);
        },
        // 对 Vectorized<scalar_t> 类型执行向量化的最小值截断操作
        [=](Vectorized<scalar_t> a) {
          return vec::clamp_min(a, min_vec);
        });
  });
}

} // 匿名命名空间结束

// 注册以下函数调用的分发函数指针
REGISTER_DISPATCH(max_stub, &max_kernel_impl);
REGISTER_DISPATCH(min_stub, &min_kernel_impl);
REGISTER_DISPATCH(aminmax_stub, &aminmax_kernel);
REGISTER_DISPATCH(where_kernel, &where_kernel_impl);
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl);
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl);
REGISTER_DISPATCH(mode_stub, &mode_kernel_impl);
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl);
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_impl);
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl);
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_impl);
REGISTER_DISPATCH(isin_default_stub, &isin_default_kernel_cpu);

} // namespace at::native
```