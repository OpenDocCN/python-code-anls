# `.\pytorch\aten\src\ATen\native\cpu\ScatterGatherKernel.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入非空工具函数头文件
#include <ATen/native/NonEmptyUtils.h>
// 引入分发存根头文件
#include <ATen/native/DispatchStub.h>
// 引入张量迭代器头文件
#include <ATen/native/TensorIterator.h>
// 引入张量高级索引头文件
#include <ATen/native/TensorAdvancedIndexing.h>
// 引入张量类定义头文件
#include <ATen/core/Tensor.h>
// 引入 ATen 配置头文件
#include <ATen/Config.h>
// 引入分发机制头文件
#include <ATen/Dispatch.h>
// 引入数值工具函数头文件
#include <ATen/NumericUtils.h>
// 引入并行计算工具头文件
#include <ATen/Parallel.h>
// 引入 CPU 平台下的归约工具函数头文件
#include <ATen/native/cpu/ReduceUtils.h>
// 引入 CPU 平台下的向量化函数头文件
#include <ATen/cpu/vec/functional.h>
// 引入 CPU 平台下的向量化工具头文件
#include <ATen/cpu/vec/vec.h>
// 引入循环工具头文件
#include <c10/util/irange.h>
// 根据是否使用 FBGEMM 引入 FBGEMM 工具头文件
#ifdef USE_FBGEMM
#include <fbgemm/Utils.h>
#endif
// 引入操作数数学类型头文件
#include <ATen/OpMathType.h>

// 根据宏定义引入不同版本的 ATen 操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

// 进入 ATen 的 native 命名空间
namespace at::native {

// 进入匿名命名空间，定义多个归约操作类作为函数对象
namespace {

// 实现为函数对象，因为 lambda 表达式不会被优化。
// 乘法归约操作类
class ReduceMultiply {
public:
  // 模板函数，用于计算乘法归约
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data *= opmath_t(*src_data);
  }

  // 特化模板函数，用于布尔类型的逻辑与归约
  constexpr void operator() (bool * self_data, bool * src_data) const {
    *self_data = *self_data && *src_data;
  }
};
// 静态实例化乘法归约操作类对象
static ReduceMultiply reduce_multiply;

// 加法归约操作类
class ReduceAdd {
public:
  // 模板函数，用于计算加法归约
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data += opmath_t(*src_data);
  }
};
// 静态实例化加法归约操作类对象
static ReduceAdd reduce_add;

// 平均值归约操作类
class ReduceMean {
public:
  // 模板函数，用于计算平均值归约
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data += opmath_t(*src_data);
  }
};
// 静态实例化平均值归约操作类对象
static ReduceMean reduce_mean;

// 最大值归约操作类
class ReduceMaximum {
public:
  // 模板函数，用于计算最大值归约
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data = at::_isnan<scalar_t>(*src_data) ? opmath_t(*src_data) : std::max(*self_data, opmath_t(*src_data));
  }
};
// 静态实例化最大值归约操作类对象
static ReduceMaximum reduce_maximum;

// 最小值归约操作类
class ReduceMinimum {
public:
  // 模板函数，用于计算最小值归约
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data = at::_isnan<scalar_t>(*src_data) ? opmath_t(*src_data) : std::min(*self_data, opmath_t(*src_data));
  }
};
// 静态实例化最小值归约操作类对象
static ReduceMinimum reduce_minimum;

// 张量赋值操作类
class TensorAssign {
public:
  // 模板函数，用于张量元素赋值
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data = opmath_t(*src_data);
  }
};
// 静态实例化张量赋值操作类对象
static TensorAssign tensor_assign;

// 定义模板结构体，用于 CPU 平台的 scatter-gather 操作
template <bool is_scatter_like = true>
struct _cpu_scatter_gather_dim_loop {
  // 模板函数，用于 scatter-gather 操作
  template <typename scalar_t, typename func_t>
  void operator()(
  // 对传入的数据进行操作，根据索引数据和指定的维度参数进行处理
  for (const auto i : c10::irange(index_dim_size)) {
    // 获取当前索引在指定维度上的值
    int64_t idx_dim = index_data[i * index_dim_stride];
    // 检查索引是否在有效范围内，如果不在则抛出错误信息
    // 详细说明索引越界的具体信息：索引值、维度、维度的尺寸
    TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
      "index ", index_data[i * index_dim_stride],
      " is out of bounds for dimension ", dim,
      " with size ", index_upper_bound
    );

    // 调用传入的函数对象 f 处理 self_data 和 src_data 的特定位置数据
    f(
      self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
      src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
    );
  }



  // 对传入的数据进行操作，根据索引数据和指定的维度参数进行处理
  for (const auto i : c10::irange(index_dim_size)) {
    // 获取当前索引在指定维度上的值
    int64_t idx_dim = index_data[i * index_dim_stride];
    // 检查索引是否在有效范围内，如果不在则抛出错误信息
    // 详细说明索引越界的具体信息：索引值、维度、维度的尺寸
    TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
      "index ", index_data[i * index_dim_stride],
      " is out of bounds for dimension ", dim,
      " with size ", index_upper_bound
    );
    // 将标量值 value 转换为与 self_data 类型相同的标量类型
    auto temp = value.to<scalar_t>();
    // 调用传入的函数对象 f 处理 self_data 和临时变量 temp 的特定位置数据
    f(
      self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp
    );
  }
    };

    // 在给定条件下创建累加缓冲区，用于操作张量的函数
    inline void create_acc_buffer(Tensor& buffer, const Tensor& self, bool need_acc) {
      // 如果需要累加操作
      if (need_acc) {
        // 将张量转换为相应的操作数类型
        auto acc_type = at::toOpMathType(self.scalar_type());
        // 根据原张量的大小和选项创建空的张量作为累加缓冲区，并复制原张量的数据
        buffer = at::empty(self.sizes(), self.options().dtype(acc_type));
        buffer.copy_(self);
      } else {
        // 如果不需要累加操作，直接使用原张量作为缓冲区
        buffer = self;
      }
    }

    // 定义 CPU 端的 scatter-gather 基础内核结构体模板
    template <bool is_scatter_like = true>
    struct cpu_scatter_gather_base_kernel {
      // 重载函数调用操作符，执行 scatter-gather 操作
      template <typename func_t>
      void operator()(const Tensor& self, int64_t dim,
        const Tensor& index, const Scalar& value,
        const std::string& method_name, func_t& kernel_func) {

        // 定义缓冲区张量
        Tensor buffer;
        // 判断是否需要累加操作
        bool need_acc = isReducedFloatingType(self.scalar_type());
        // 创建累加缓冲区
        create_acc_buffer(buffer, self, need_acc);

        // 获取索引张量的尺寸和步长信息，并确保它们非空
        auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
        auto index_strides = ensure_nonempty_vec(index.strides().vec());

        // 对于在内核中遍历的维度 `dim`，
        // index.stride(dim) = 0 和 index.size(dim) = 1。
        // 同时，index.size(dim) = 1 确保 TensorIterator.DimCounter 的形式为：
        // (i_1, ..., i_{dim-1}, 0, i_{dim+1}, ..., i_n)。
        index_sizes[dim] = 1;
        index_strides[dim] = 0;

        // 配置张量迭代器以支持 scatter-gather 操作
        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .resize_outputs(false)
          // NOLINTNEXTLINE(bugprone-argument-comment)
          .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
          .add_output(buffer)
          .add_const_input(index)
          .build();

        // 确保在维度 `dim` 上的自身张量步长和大小非空
        auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
        auto self_dim_size = ensure_nonempty_size(buffer, dim);

        // 确保在维度 `dim` 上的索引张量步长和大小非空
        auto index_dim_stride = ensure_nonempty_stride(index, dim);
        auto index_dim_size = ensure_nonempty_size(index, dim);

        // 计算索引的上界，即自身张量在维度 `dim` 上的大小
        auto index_upper_bound = self_dim_size;

        // 由于索引维度被压缩，需要根据索引大小调整粒度，以保持并行性的均匀性
        int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, self.scalar_type(),
      "scatter_gather_scalar_cpu", [&] {
        // 定义两个常量，用于指定在数据指针数组中的位置
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 1;
        // 使用当前标量类型的操作数类型
        using opmath_t = at::opmath_type<scalar_t>;
        // 创建 CPU 上的 scatter_gather_dim_loop 函数对象
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        // 定义循环函数
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          // 如果 dim 是最后一个维度
          if (dim == buffer.dim() - 1) {
            // 遍历 n 次，每次执行 loop_func 操作
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // 调用 loop_func 处理当前维度的数据
              loop_func.template operator()<scalar_t, func_t>(
                (opmath_t*)self_data_bytes, self_dim_stride,
                (int64_t*)index_data_bytes, index_dim_stride,
                value, dim, index_dim_size, index_upper_bound,
                kernel_func);

              // 更新数据指针
              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
            }
          }
          else {
            // 如果 dim 不是最后一个维度
            // 遍历 index_dim_size 次
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              // 遍历 n 次
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                // 从索引数据中获取索引值
                int64_t idx_dim = *(int64_t*)index_data;
                // 检查索引是否在有效范围内
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                // 将 value 转换为当前标量类型的临时变量
                auto temp = value.to<scalar_t>();
                // 调用 kernel_func 处理数据
                kernel_func((opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp);

                // 更新数据指针
                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
              }
            }
          }
        };
        // 针对每个迭代器元素应用循环函数
        iter.for_each(loop, grain_size);
      }
    );
    // 如果需要累加，则将 buffer 复制到 self
    if (need_acc) {
      self.copy_(buffer);
    }
  }

  // 定义模板函数 operator()，处理输入的 Tensor 数据
  template <typename func_t>
  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, func_t& kernel_func) {

    // 创建一个 buffer Tensor
    Tensor buffer;
    // 判断是否需要累加操作
    bool need_acc = isReducedFloatingType(self.scalar_type());
    // 创建累加缓冲区
    create_acc_buffer(buffer, self, need_acc);
    // 创建一个 TensorIteratorConfig 对象，并设置以下属性：
    // - 不检查所有张量的数据类型是否相同
    // - 不调整输出大小
    // - 使用给定的索引大小和维度来声明静态形状，并压缩指定维度
    // - 将 buffer 添加为输出
    // - 将 src 和 index 添加为常量输入
    // - 构建 Tensor 迭代器对象
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // 禁止 NOLINT 下一个行的 LINT 检查指定为 bugprone-argument-comment
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(buffer)
      .add_const_input(src)
      .add_const_input(index)
      .build();
    
    // 确保 buffer 在给定维度 dim 上有非空的步长，并分配给 self_dim_stride
    auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
    // 确保 buffer 在给定维度 dim 上有非空的大小，并分配给 self_dim_size
    auto self_dim_size = ensure_nonempty_size(buffer, dim);
    
    // 确保 index 在给定维度 dim 上有非空的步长，并分配给 index_dim_stride
    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    // 确保 index 在给定维度 dim 上有非空的大小，并分配给 index_dim_size
    auto index_dim_size = ensure_nonempty_size(index, dim);
    
    // 确保 src 在给定维度 dim 上有非空的步长，并分配给 src_dim_stride
    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    // 确保 src 在给定维度 dim 上有非空的大小，并分配给 src_dim_size
    auto src_dim_size = ensure_nonempty_size(src, dim);
    
    // 如果 is_scatter_like 为真，则将 self_dim_size 赋给 index_upper_bound，否则将 src_dim_size 赋给 index_upper_bound
    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;
    
    // 计算 grain_size，为 std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size)
    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);
    // 使用宏 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3，生成代码模板，覆盖三种标量类型：Bool、Half、BFloat16
    // 并调用 scatter_gather_tensor_cpu 函数处理迭代器 iter 的数据
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(1),
      "scatter_gather_tensor_cpu", [&] {
        // 定义迭代过程中的常量索引值
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        // 使用标量类型 scalar_t 确定操作数的数学运算类型
        using opmath_t = at::opmath_type<scalar_t>;
        // 创建 _cpu_scatter_gather_dim_loop 对象实例
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        // 定义循环处理函数 loop，处理 TensorIterator 的数据
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          // 获取自身数据、索引数据、源数据的字节指针
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // 若 dim 是最后一个维度，执行此分支
          if (dim == buffer.dim() - 1) {
            // 遍历 n 个元素，执行 scatter_gather_dim_loop 操作
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // 使用循环函数对象执行模板化的 scatter_gather 操作
              loop_func.template operator()<scalar_t, func_t>(
                 (opmath_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );
    
              // 更新指针以便处理下一个元素
              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            // 若 dim 不是最后一个维度，执行此分支
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              // 遍历 n 个元素，处理 scatter_gather 操作
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                // 从 index_data 中提取索引值 idx_dim，并检查其合法性
                int64_t idx_dim = *(int64_t*)index_data;
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);
    
                // 调用 kernel_func 处理 self_data 和 src_data
                kernel_func(
                  (opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);
    
                // 更新指针以便处理下一个元素
                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        // 使用 TensorIterator 的 for_each 方法，以 grain_size 为粒度调用 loop 函数
        iter.for_each(loop, grain_size);
      }
    );
    // 如果需要进行累积操作，将 buffer 的内容复制到 self 中
    if (need_acc) {
      self.copy_(buffer);
    }
    }
  }

  // 重载函数调用操作符，用于执行降维操作
  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, ReduceMean& kernel_func) {

    // 声明一个缓冲区张量
    Tensor buffer;
    // 检查是否需要累积缓冲区，对于浮点数类型进行判断
    bool need_acc = isReducedFloatingType(self.scalar_type());
    // 创建累积缓冲区
    create_acc_buffer(buffer, self, need_acc);

    // 配置张量迭代器，设置不检查所有张量的数据类型一致性，
    // 不调整输出大小，声明静态形状并指定压缩的维度
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(buffer)
      .add_const_input(src)
      .add_const_input(index)
      .build();

    // 确保缓冲区在指定维度上有非空步长
    auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
    // 确保缓冲区在指定维度上有非空大小
    auto self_dim_size = ensure_nonempty_size(buffer, dim);

    // 确保索引张量在指定维度上有非空步长
    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    // 确保索引张量在指定维度上有非空大小
    auto index_dim_size = ensure_nonempty_size(index, dim);

    // 确保源张量在指定维度上有非空步长
    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    // 确保源张量在指定维度上有非空大小
    auto src_dim_size = ensure_nonempty_size(src, dim);

    // 根据是否类似散布操作，确定索引的上界
    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    // 计算谷粒大小，以内部定义的GRAIN_SIZE和索引维度大小的较大值为准
    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(1),
      "scatter_gather_tensor_cpu_reduce_mean", [&] {
        // 定义迭代器步长索引
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        // 使用操作类型宏定义
        using opmath_t = at::opmath_type<scalar_t>;
        // 定义循环处理函数对象
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        // 定义循环执行的lambda函数
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          // 获取自身数据、索引数据和源数据的字节指针
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // 如果dim是最后一个维度
          if (dim == buffer.dim() - 1) {
            // 对于每个元素执行循环
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // 使用loop_func处理当前维度的数据
              loop_func.template operator()<scalar_t, ReduceMean>(
                 (opmath_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );
              // 更新指针位置
              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            // 如果dim不是最后一个维度
            for (const auto i : c10::irange(index_dim_size)) {
              // 计算当前索引数据的位置
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              // 对于每个元素执行循环
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                // 获取索引维度值并进行边界检查
                int64_t idx_dim = *(int64_t*)index_data;
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);
                // 调用kernel_func处理数据
                kernel_func(
                  (opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                // 更新指针位置
                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        // 使用迭代器对每个元素执行loop函数
        iter.for_each(loop, grain_size);
      }
    );
    // 如果需要累加，则将buffer的内容复制到self中
    if (need_acc) {
      self.copy_(buffer);
    }
  }
  ```

  // 结束匿名命名空间

  }

  // 结束类定义

  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, ReduceMaximum& kernel_func) {
    // 重载函数调用运算符，接受参数 self 张量、维度 dim、索引张量 index、源张量 src、方法名称 method_name 和核函数 kernel_func

    Tensor buffer;
    // 定义临时张量 buffer

    bool need_acc = isReducedFloatingType(self.scalar_type());
    // 根据 self 张量的数据类型判断是否需要累加器

    create_acc_buffer(buffer, self, need_acc);
    // 创建累加器缓冲区 buffer，根据 need_acc 决定是否需要累加

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      // 配置张量迭代器：不检查所有张量是否具有相同的数据类型，不调整输出大小，
      // 使用 index 张量的尺寸声明静态形状，将维度 dim 压缩
      .add_output(buffer)
      // 将 buffer 添加为输出
      .add_const_input(src)
      // 将 src 添加为常量输入
      .add_const_input(index)
      // 将 index 添加为常量输入
      .build();
      // 构建张量迭代器

    auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
    // 确保 buffer 在 dim 维度上的步长非空，保存到 self_dim_stride

    auto self_dim_size = ensure_nonempty_size(buffer, dim);
    // 确保 buffer 在 dim 维度上的尺寸非空，保存到 self_dim_size

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    // 确保 index 在 dim 维度上的步长非空，保存到 index_dim_stride

    auto index_dim_size = ensure_nonempty_size(index, dim);
    // 确保 index 在 dim 维度上的尺寸非空，保存到 index_dim_size

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    // 确保 src 在 dim 维度上的步长非空，保存到 src_dim_stride

    auto src_dim_size = ensure_nonempty_size(src, dim);
    // 确保 src 在 dim 维度上的尺寸非空，保存到 src_dim_size

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;
    // 根据 is_scatter_like 的布尔值确定 index 的上界，若为 true 使用 self_dim_size，否则使用 src_dim_size

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);
    // 计算 grain_size，作为 at::internal::GRAIN_SIZE 与 index_dim_size 的商与 1 的较大值
    AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(1),
      "scatter_gather_tensor_cpu_reduce_amax", [&] {
        // 定义索引常量
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        // 定义操作数的数学类型
        using opmath_t = at::opmath_type<scalar_t>;
        // 定义循环功能对象
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        // 定义循环函数
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          // 获取自身数据的字节指针
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          // 获取索引数据的字节指针
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          // 获取源数据的字节指针
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // 如果维度是最后一个维度
          if (dim == buffer.dim() - 1) {
            // 对于每个元素执行循环
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // 调用循环函数处理自身数据、索引数据和源数据
              loop_func.template operator()<scalar_t, ReduceMaximum>(
                 (opmath_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );
              // 更新数据指针
              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            // 对于每个索引维度大小执行循环
            for (const auto i : c10::irange(index_dim_size)) {
              // 获取自身数据、索引数据和源数据的指针
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              // 对于每个元素执行循环
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                // 获取索引维度的值
                int64_t idx_dim = *(int64_t*)index_data;
                // 检查索引维度是否在有效范围内
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                // 调用 kernel_func 处理自身数据和源数据
                kernel_func(
                  (opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                // 更新数据指针
                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        // 对迭代器应用循环函数
        iter.for_each(loop, grain_size);
      }
    );
    // 如果需要累加结果，则将 buffer 复制给 self
    if (need_acc) {
      self.copy_(buffer);
    }
  }
}

void operator()(const Tensor& self, int64_t dim,
  const Tensor& index, const Tensor& src,
  const std::string& method_name, ReduceMinimum& kernel_func) {

  // 创建一个缓冲区用于存储归约操作的中间结果
  Tensor buffer;
  // 检查是否需要累加，针对浮点类型的归约
  bool need_acc = isReducedFloatingType(self.scalar_type());
  // 根据需要创建累加缓冲区
  create_acc_buffer(buffer, self, need_acc);

  // 配置张量迭代器，用于并行计算
  auto iter = TensorIteratorConfig()
    // 禁用类型一致性检查
    .check_all_same_dtype(false)
    // 禁用输出大小调整
    .resize_outputs(false)
    // 设置静态形状，并在指定维度上压缩
    // NOLINTNEXTLINE(bugprone-argument-comment)
    .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
    // 添加输出缓冲区
    .add_output(buffer)
    // 添加常量输入张量 src
    .add_const_input(src)
    // 添加常量输入张量 index
    .add_const_input(index)
    // 构建迭代器
    .build();

  // 确保在指定维度上缓冲区非空的步长和大小
  auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
  auto self_dim_size = ensure_nonempty_size(buffer, dim);

  // 确保在指定维度上索引张量非空的步长和大小
  auto index_dim_stride = ensure_nonempty_stride(index, dim);
  auto index_dim_size = ensure_nonempty_size(index, dim);

  // 确保在指定维度上源张量非空的步长和大小
  auto src_dim_stride = ensure_nonempty_stride(src, dim);
  auto src_dim_size = ensure_nonempty_size(src, dim);

  // 计算索引的上界，根据是否为类似 scatter 的操作选择不同的上界
  auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

  // 计算粒度大小，用于调整工作负载的大小
  int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);
    AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(1),
      "scatter_gather_tensor_cpu_reduce_amin", [&] {
        // 定义几个常量来表示在数据块中不同数据的索引
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        // 定义操作数的数学类型
        using opmath_t = at::opmath_type<scalar_t>;
        // 使用 _cpu_scatter_gather_dim_loop<is_scatter_like> 初始化循环函数对象
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        // 定义循环函数
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          // 获取自身数据块、索引数据块和源数据块的指针
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // 如果当前维度是最后一个维度
          if (dim == buffer.dim() - 1) {
            // 对于每个元素执行循环
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // 调用循环函数处理当前维度的数据
              loop_func.template operator()<scalar_t, ReduceMinimum>(
                 (opmath_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );
               // 更新数据指针
              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            // 对于每个索引维度的循环
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              // 计算索引数据的偏移量
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              // 对于每个元素执行循环
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                // 读取索引维度的索引值
                int64_t idx_dim = *(int64_t*)index_data;
                // 检查索引是否在有效范围内
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);
                // 调用核函数处理数据
                kernel_func(
                  (opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                // 更新数据指针
                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        // 使用 TensorIterator 执行循环函数
        iter.for_each(loop, grain_size);
      }
    );
    // 如果需要进行累加，则将 buffer 的内容复制到 self 中
    if (need_acc) {
      self.copy_(buffer);
    }
// 结束了 FBGEMM 命名空间
};

// 如果未定义 USE_FBGEMM，则进入 fbgemm 命名空间
#ifndef USE_FBGEMM
namespace fbgemm {

// 定义模板函数 radix_sort_parallel，用于并行基数排序
template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(
    K* const inp_key_buf,           // 输入键缓冲区
    V* const inp_value_buf,         // 输入值缓冲区
    K* const tmp_key_buf,           // 临时键缓冲区
    V* const tmp_value_buf,         // 临时值缓冲区
    const int64_t elements_count,   // 元素数量
    const int64_t max_value) {      // 最大值
  TORCH_INTERNAL_ASSERT(false, "radix_sort_parallel: ATen not compiled with FBGEMM support");
  return std::make_pair(nullptr, nullptr);  // 返回空指针对
}

}
#endif

// 注释以下内容涉及散射归约优化的注意事项
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// 1. 优化“scatter_reduce”在经典 PyG 情况下的使用：
//    “scatter_reduce” 在聚合信息时广泛用于“消息传递”中。
//
//    通常情况下，self 将是一个二维张量，index 是一个一维的扩展/广播张量，
//    这意味着聚合是在行上进行的，我们可以在内部维度上进行向量化。
//
// 2. 实现：将“scatter_reduce”映射到形状为 `[M, N]` 的 `spmm` 归约，
//    其中：
//
//    M: self 的维度大小
//    nnz: index 的维度大小
//    K: index.numel() / nnz;
//
//    步骤1：将输入的 index 转换为 CSR 格式（使用基数排序来解决对 self 张量的写入地址冲突）
//
//    步骤2：spmm 归约，在 M 上并行，在 K 上向量化
//

// 定义模板函数 cpu_scatter_reduce_expanded_index，处理 CPU 上扩展索引的散射归约
template <typename scalar_t, ReductionType reduce>
void cpu_scatter_reduce_expanded_index(const Tensor& self, const Tensor& index, const Tensor& src, bool include_self) {
  const int64_t* index_data = index.const_data_ptr<int64_t>();  // 获取 index 的数据指针
  scalar_t* self_data = self.data_ptr<scalar_t>();               // 获取 self 的数据指针
  const scalar_t* src_data = src.const_data_ptr<scalar_t>();     // 获取 src 的数据指针

  const int64_t M = ensure_nonempty_size(self, 0);               // 确定 self 的非空大小作为 M
  const int64_t nnz = ensure_nonempty_size(index, 0);            // 确定 index 的非空大小作为 nnz
  const int64_t K = index.numel() / nnz;                         // 计算 K

  const int64_t index_upper_bound = M;                           // index 的上界为 M

  // 创建长度为 nnz 的键和值数组
  auto keys = std::make_unique<int64_t[]>(nnz);
  auto values = std::make_unique<int64_t[]>(nnz);
  auto keys_tmp = std::make_unique<int64_t[]>(nnz);
  auto values_tmp = std::make_unique<int64_t[]>(nnz);

  // 并行处理 index 数据填充 keys 和 values
  at::parallel_for(0, nnz, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t index = index_data[i];
      TORCH_CHECK(index >= 0 && index < index_upper_bound,
                  "index ", index,
                  " is out of bounds for dimension ", 0,
                  " with size ", index_upper_bound);
      keys[i] = index;
      values[i] = i;
    }
  });

  int64_t* sorted_col_index_keys = nullptr;
  int64_t* sorted_col_index_values = nullptr;

  // 调用 fbgemm 的基数排序函数，返回排序后的键和值
  std::tie(sorted_col_index_keys, sorted_col_index_values) = fbgemm::radix_sort_parallel(
      keys.get(),
      values.get(),
      keys_tmp.get(),
      values_tmp.get(),
      nnz,
      M);

  int num_threads = at::get_num_threads();  // 获取线程数
  // 创建长度为 num_threads 的 num_uniq 数组
  std::vector<int64_t> num_uniq(num_threads, 0);

  // 并行处理 nnz 的数据，填充 num_uniq 数组
  at::parallel_for(1, nnz, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
  for(const auto i : c10::irange(begin, end)) {
    // 遍历指定范围内的索引 i
    if (sorted_col_index_keys[i] != sorted_col_index_keys[i - 1]) {
      // 检查当前索引与前一个索引是否相同，如果不同则增加 num_uniq[tid]
      num_uniq[tid]++;
    }
  });
  // 对于第一个线程，增加 num_uniq[0]，处理特殊情况
  num_uniq[0]++;
  // 计算各个线程的 num_uniq 数组的累计和
  for (const auto n : c10::irange(1, num_threads)) {
    num_uniq[n] += num_uniq[n - 1];
  }

  // 如果有些行没有写入，num_nonzero_rows 将小于 M
  int64_t num_nonzero_rows = num_uniq[num_threads - 1];
  // 创建临时的行索引和行偏移数组
  auto row_index_tmp = std::make_unique<int64_t[]>(num_nonzero_rows);
  auto row_index_offset_tmp = std::make_unique<int64_t[]>(num_nonzero_rows + 1);
  int64_t* row_index = row_index_tmp.get();
  int64_t* row_index_offset = row_index_offset_tmp.get();
  // 初始化第一个行索引和偏移量
  row_index[0] = sorted_col_index_keys[0];
  row_index_offset[0] = 0;
  // 设置最后一个偏移量为 nnz
  row_index_offset[num_nonzero_rows] = nnz;

  // 并行处理每个线程的行索引和偏移量
  at::parallel_for(1, nnz, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    // 计算当前线程的行索引和偏移数组的起始位置
    int64_t* t_index = row_index + ((tid == 0) ? 1 : num_uniq[tid - 1]);
    int64_t* t_index_offset = row_index_offset + ((tid == 0) ? 1 : num_uniq[tid - 1]);
    // 遍历指定范围内的索引 i
    for (const auto i : c10::irange(begin, end)) {
      // 检查当前索引与前一个索引是否相同，如果不同则更新行索引和偏移量
      if (sorted_col_index_keys[i] != sorted_col_index_keys[i - 1]) {
        *t_index = sorted_col_index_keys[i];
        *t_index_offset = i;
        t_index++;
        t_index_offset++;
      }
    }
  });

  using opmath_t = at::opmath_type<scalar_t>;
  Tensor buffer;
  opmath_t* buffer_data = nullptr;
  static constexpr bool need_acc = is_reduced_floating_point_v<scalar_t>;
  // 如果需要累加操作，则创建一个缓冲区
  if constexpr (need_acc) {
    auto acc_type = at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true);
    buffer = at::zeros({num_threads, K}, self.options().dtype(acc_type));
    buffer_data = buffer.data_ptr<opmath_t>();
  }

  // TODO: 对列维度进行分块以减少写入带宽
  // 并行处理每个线程对应的操作
  at::parallel_for(0, num_nonzero_rows, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    // 获取当前线程的缓冲区指针
    opmath_t* buffer_ptr = nullptr;
    // 对于范围 [begin, end) 中的每个索引 m，执行循环
    for (const auto m : c10::irange(begin, end)) {
      // 获取当前索引 m 对应的行号
      int64_t row = row_index[m];
      // 获取当前索引 m 对应的行偏移起始位置
      int64_t off_start = row_index_offset[m];
      // 获取当前索引 m+1 对应的行偏移结束位置
      int64_t off_end = row_index_offset[m + 1];
      // 计算出当前行在 self_data 中的起始指针
      scalar_t* self_ptr = self_data + row * K;

      // 根据需要初始化 self_ptr 指向的行数据，使用 buffer_ptr 作为中间缓冲区
      if constexpr (need_acc) {
        // 如果需要累加操作，则使用 buffer_data 作为缓冲区
        buffer_ptr = buffer_data + tid * K;
      } else {
        // 否则，将 self_ptr 强制转换为 opmath_t* 类型，作为 buffer_ptr
        buffer_ptr = reinterpret_cast<opmath_t*>(self_ptr);
      }

      // 步骤 1：根据需要重新初始化 self_ptr 指向的行数据
      _init<scalar_t, reduce>(self_ptr, buffer_ptr, K, include_self);

      // 步骤 2：执行归约操作
      // 遍历 [off_start, off_end) 区间内的每个索引 n
      for (const auto n : c10::irange(off_start, off_end)) {
        // 获取当前索引 n 对应的列号
        int64_t col = sorted_col_index_values[n];
        // 更新 buffer_ptr 和 src_data[col * K] 之间的数据，执行归约操作
        update<scalar_t, reduce>(buffer_ptr, src_data + col * K, K);
      }

      // 如果需要累加操作，则执行向量化转换，将 buffer_ptr 转换为 self_ptr
      if constexpr (need_acc) {
        vec::convert(buffer_ptr, self_ptr, K);
      }

      // 步骤 3：完成最终操作
      // 初始化 count 为 include_self 是否为真的值（1 或 0）
      int64_t count = include_self ? 1 : 0;
      // 累加 off_end - off_start 到 count 中，作为最终的行写入操作的计数
      count += off_end - off_start;
      // 将 self_ptr 指向的行数据进行最终的写入操作
      write<scalar_t, reduce>(self_ptr, count, K);
    }
  });
}

template <typename scalar_t>
void cpu_gather_expanded_index_kernel(const Tensor& result, const Tensor& index, const Tensor& self) {
  // 获取索引数据的指针，类型为 int64_t
  const int64_t* index_data = index.const_data_ptr<int64_t>();
  // 获取结果数据的指针，类型为 scalar_t
  scalar_t* result_data = result.data_ptr<scalar_t>();
  // 获取源数据的指针，类型为 scalar_t
  const scalar_t* self_data = self.const_data_ptr<scalar_t>();

  // 确定结果张量的大小 M
  const int64_t M = ensure_nonempty_size(result, 0);
  // 确定源张量的大小 N
  const int64_t N = ensure_nonempty_size(self, 0);
  // 确定索引张量元素个数除以 M，得到 K
  const int64_t K = index.numel() / M;

  // 确定索引的上界，即 N
  const int64_t index_upper_bound = N;

  // 使用 Vectorized 类型定义 Vec，用于向量化操作
  using Vec = vec::Vectorized<scalar_t>;
  // 确定粒度大小，至少为 1 或者 at::internal::GRAIN_SIZE / K，用于并行化
  int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / K);
  // 并行处理主循环，范围从 0 到 M，以 grain_size 为步长
  at::parallel_for(0, M, grain_size, [&](int64_t begin, int64_t end) {
    for (const auto m : c10::irange(begin, end)) {
      // 获取当前结果指针的位置
      scalar_t* result_ptr = result_data + m * K;
      // 获取当前索引值
      int64_t index = index_data[m];
      // 检查索引是否在有效范围内，输出错误信息
      TORCH_CHECK(index >= 0 && index < index_upper_bound,
                  "index ", index,
                  " is out of bounds for dimension ", 0,
                  " with size ", index_upper_bound);
      // 获取源数据中对应索引的指针位置
      const scalar_t* self_ptr = self_data + index * K;
      // 初始化循环变量 d
      int64_t d = 0;
      // 对于每个向量化长度 Vec::size() 大小的部分，加载并存储向量化数据
      for (; d < K - (K % Vec::size()); d += Vec::size()) {
        Vec out_vec = Vec::loadu(self_ptr + d);
        out_vec.store(result_ptr + d);
      }
      // 如果剩余的部分无法向量化，逐个处理
      #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (; d < K; d++) {
        result_ptr[d] = self_ptr[d];
      }
    }
  });
}

void scatter_add_expanded_index_kernel(const Tensor& self, const Tensor& index, const Tensor& src) {
  // 根据张量类型分发到对应的浮点类型处理函数，执行求和操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
    ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "scatter_add_expanded_index", [&] {
      cpu_scatter_reduce_expanded_index<scalar_t, ReductionType::SUM>(self, index, src, /*include_self*/true);
  });
}

void scatter_reduce_expanded_index_kernel(
    const Tensor& self, const Tensor& index, const Tensor& src,
    const ReductionType& reduction, bool include_self) {
  // 根据张量类型分发到对应的浮点类型和规约类型处理函数，执行规约操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
    ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "scatter_reduce_expanded_index", [&] {
    AT_DISPATCH_REDUCTION_TYPES(reduction, [&]() {
      cpu_scatter_reduce_expanded_index<scalar_t, reduce>(self, index, src, include_self);
    });
  });
}

void gather_expanded_index_kernel(const Tensor& result, const Tensor& self, const Tensor& index) {
  // 根据张量类型分发到对应的浮点类型处理函数，执行 gather 操作
  AT_DISPATCH_FLOATING_TYPES_AND(
    ScalarType::BFloat16, self.scalar_type(), "gather_expanded_index", [&] {
      cpu_gather_expanded_index_kernel<scalar_t>(result, index, self);
  });
}

void gather_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  // 调用 scatter_gather_base_kernel 函数处理 gather 操作
  cpu_scatter_gather_base_kernel</*is_scatter_like=*/false>()(
    result, dim, index, self,
    "gather_out_cpu", tensor_assign);
}

void scatter_cpu_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  // 调用 scatter_gather_base_kernel 函数处理 scatter 操作
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src, "scatter_cpu_", tensor_assign);
}
void scatter_fill_cpu_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
  // 调用通用的 CPU 散布-聚集基础内核函数，执行填充操作
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, value, "scatter_fill_cpu_", tensor_assign);
}

void scatter_add_cpu_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  // 调用通用的 CPU 散布-聚集基础内核函数，执行加法散布操作
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src,
    "scatter_add_", reduce_add);
}

void scatter_reduce_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                               const Tensor& src, const ReductionType& reduce) {
  switch (reduce) {
  case ReductionType::SUM :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行按和聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_add_", reduce_add);
    break;
  case ReductionType::PROD :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行按积聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_multiply_", reduce_multiply);
    break;
  default :
    break;
  }
}

void scatter_reduce_two_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                                   const Tensor& src, const ReductionType& reduce) {
  switch (reduce) {
  case ReductionType::SUM :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行两个张量按和聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_sum_", reduce_add);
    break;
  case ReductionType::PROD :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行两个张量按积聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_prod_", reduce_multiply);
    break;
  case ReductionType::MAX :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行两个张量按最大值聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_amax_", reduce_maximum);
    break;
  case ReductionType::MIN :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行两个张量按最小值聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_amin_", reduce_minimum);
    break;
  case ReductionType::MEAN :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行两个张量按均值聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_mean_", reduce_mean);
    break;
  }
}

void scatter_scalar_reduce_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                                      const Scalar& value, const ReductionType& reduce) {
  switch (reduce) {
  case ReductionType::SUM :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行标量和张量按和聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                       "scatter_scalar_reduce_add_", reduce_add);
    break;
  case ReductionType::PROD :
    // 调用通用的 CPU 散布-聚集基础内核函数，执行标量和张量按积聚集操作
    cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                       "scatter_scalar_reduce_multiply_", reduce_multiply);
    break;
  default:
    break;
  }
}

} // 匿名命名空间

// 注册不同操作的分发函数到相应的 CPU 内核
REGISTER_DISPATCH(gather_stub, &gather_cpu_kernel);
REGISTER_DISPATCH(scatter_stub, &scatter_cpu_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cpu_kernel);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cpu_kernel);
// 注册 scatter_reduce_stub 函数指针，用于 scatter_reduce_cpu_kernel 函数的调度
REGISTER_DISPATCH(scatter_reduce_stub, &scatter_reduce_cpu_kernel);

// 注册 scatter_scalar_reduce_stub 函数指针，用于 scatter_scalar_reduce_cpu_kernel 函数的调度
REGISTER_DISPATCH(scatter_scalar_reduce_stub, &scatter_scalar_reduce_cpu_kernel);

// 注册 scatter_reduce_two_stub 函数指针，用于 scatter_reduce_two_cpu_kernel 函数的调度
REGISTER_DISPATCH(scatter_reduce_two_stub, &scatter_reduce_two_cpu_kernel);

// 在命名空间 at::native 内，注册 scatter_add_expanded_index_stub 函数指针，
// 用于 scatter_add_expanded_index_kernel 函数的调度，提供了 GNN 使用的快速路径
REGISTER_DISPATCH(scatter_add_expanded_index_stub, &scatter_add_expanded_index_kernel);

// 在命名空间 at::native 内，注册 scatter_reduce_expanded_index_stub 函数指针，
// 用于 scatter_reduce_expanded_index_kernel 函数的调度，提供了 GNN 使用的快速路径
REGISTER_DISPATCH(scatter_reduce_expanded_index_stub, &scatter_reduce_expanded_index_kernel);

// 在命名空间 at::native 内，注册 gather_expanded_index_stub 函数指针，
// 用于 gather_expanded_index_kernel 函数的调度，提供了 GNN 使用的快速路径
REGISTER_DISPATCH(gather_expanded_index_stub, &gather_expanded_index_kernel);
```