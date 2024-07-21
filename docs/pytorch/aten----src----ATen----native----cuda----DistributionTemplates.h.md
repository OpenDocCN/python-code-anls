# `.\pytorch\aten\src\ATen\native\cuda\DistributionTemplates.h`

```
#pragma once

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandBase.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/util/Half.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/core/DistributionsHelper.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <cstdint>
#include <limits>
#include <utility>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace at {
namespace native {
namespace {

// launch bounds used for kernels utilizing TensorIterator
const uint32_t block_size_bound = 256;  // 定义一个常量，表示 CUDA 核函数的块大小上限
const uint32_t grid_size_bound = 4;     // 定义一个常量，表示 CUDA 核函数的网格大小上限

// number of randoms given by distributions like curand_uniform4, curand_uniform2_double
// used in calculating philox offset.
const uint32_t curand4_engine_calls = 4;  // 定义一个常量，表示使用 curand_uniform4 等分布时生成随机数的数量

// utility function that calculates proper philox_offset
// for distributions utilizing TensorIterator. For distributions using
// TensorIterator, we are using a grid-stride loop with each
// thread yielding one element per thread. For the edge of the grid-stride
// loop, if the tensor size is large, the unroll loop will kick in and the float4
// from curand4 will start getting utilized (for common tensor sizes, we end up
// using rand.x from each thread). Hence, the philox_offset is
// (number of elements per thread * number of engine calls), which makes
// sure that philox offset increment is not less than the number of randoms used
// in each thread.
std::tuple<uint64_t, dim3, dim3> calc_execution_policy(int64_t total_elements) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);  // 计算总元素数并转换为 uint64_t 类型
  const uint32_t block_size = block_size_bound;  // 使用之前定义的 CUDA 核函数块大小上限
  const uint32_t unroll = curand4_engine_calls;  // 使用之前定义的 curand4_engine_calls
  dim3 dim_block(block_size);  // 定义 CUDA 核函数的块维度
  dim3 grid((numel + block_size - 1) / block_size);  // 计算 CUDA 核函数的网格维度
  uint32_t blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm,
      grid.x);
  // 计算每个线程生成随机数的次数，用于偏移 Philox 计数器
  uint64_t counter_offset = ((numel - 1) / (block_size * grid.x * unroll) + 1)
                                * curand4_engine_calls;
  return std::make_tuple(counter_offset, grid, dim_block);  // 返回计算结果作为元组
}

// grid stride loop kernel for distributions
template<typename accscalar_t, int unroll_factor, typename dist_t, typename transform_t>
C10_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)  // 定义 CUDA 核函数的启动参数，包括块大小和网格大小
/**
 * distribution_elementwise_grid_stride_kernel 是一个 CUDA 全局函数，用于并行处理张量中的元素。
 * 它使用了 grid-stride loop 策略来优化计算效率，并通过 PhiloxCudaState 状态生成随机数。
 *
 * @param numel 张量中的元素总数
 * @param philox_args 包含 Philox 算法所需参数的结构体
 * @param dist_func 分布函数，用于生成随机数
 * @param transform_func 变换函数，用于对生成的随机数进行变换
 */
__global__ void distribution_elementwise_grid_stride_kernel(int numel,
                                                            PhiloxCudaState philox_args,
                                                            const dist_t dist_func,
                                                            const transform_t transform_func) {
  auto seeds = at::cuda::philox::unpack(philox_args);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);

  int rounded_size = ((numel - 1)/(blockDim.x * gridDim.x * unroll_factor)+1) *
      blockDim.x * gridDim.x * unroll_factor;
  for(int linear_index = idx; linear_index < rounded_size; linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_func(&state);
    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        transform_func(li, static_cast<accscalar_t>((&rand.x)[ii]));
      }
    }
    __syncthreads();
  }
}

/**
 * distribution_nullary_kernel 类似于 ATen/native/cuda/Loops.cuh 中的 gpu_kernel。
 * 它使用 TensorIterator 来启动一个内核，专门用于分布计算。
 * 区别在于：
 *   - 它启动一个基于 grid-stride loop 的内核，不像 Loops.cuh 中的 elementwise_kernel 那样通用，
 *     而是为这里的分布内核专门设计的。
 *   - 对于大尺寸张量，我们可以递归地启动多个内核（例如，如果 (!iter.can_use_32bit_indexing())），
 *     因此，这个函数中进行了 Philox 偏移的计算。
 *
 * FIXME: 我们是否可以专门为 Loops.cuh 中的 elementwise_kernel 和 launch_kernel 添加
 * grid-stride loop 内核，并使用它们来启动我们的分布内核？需要注意的是，我们发现通过测试，
 * grid-stride loop 内核可以达到最佳的有效带宽。
 *
 * @tparam scalar_t 标量类型
 * @tparam accscalar_t 累加标量类型
 * @tparam unroll_factor 循环展开因子，必须 >= 1
 * @tparam RNG 随机数生成器类型
 * @tparam dist_t 分布函数类型
 * @tparam transform_t 变换函数类型
 * @param iter TensorIteratorBase 类型的迭代器，用于描述要操作的张量
 * @param gen 随机数生成器
 * @param dist_func 分布函数，生成随机数
 * @param transform_func 变换函数，对生成的随机数进行变换
 */
template<typename scalar_t,
         typename accscalar_t,
         int unroll_factor,
         typename RNG,
         typename dist_t,
         typename transform_t>
void distribution_nullary_kernel(at::TensorIteratorBase& iter,
                                 RNG gen,
                                 const dist_t& dist_func,
                                 const transform_t transform_func) {
  static_assert(unroll_factor >= 1, "unroll_factor must be >= 1.");
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  auto execution_policy = calc_execution_policy(numel);
  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  if (!iter.can_use_32bit_indexing()) {
    // 遍历迭代器，使用32位索引进行迭代
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      // 调用 nullary 分布内核函数，处理子迭代器的数据
      distribution_nullary_kernel<scalar_t, accscalar_t, unroll_factor>(sub_iter,
        gen, dist_func, transform_func);
    }
    // 函数结束，返回
    return;
  }

  // 获取迭代器的数据指针，并将其转换为 char* 类型
  char* out_data = (char*)iter.data_ptr(0);

  // 获取当前 CUDA 流
  auto stream = at::cuda::getCurrentCUDAStream();
  
  // 如果迭代器是简单的一维情况
  if (iter.is_trivial_1d()) {
    // 获取内部步长
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    // 调用元素级别的 grid-stride 内核函数，使用 grid 和 block 尺寸
    // 该函数在 GPU 上执行，处理每个元素
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      // Lambda 函数，以设备端方式捕获，处理索引和随机数
      [=]__device__(int idx, accscalar_t rand) {
        scalar_t* out = (scalar_t*)&out_data[stride0 * idx];
        *out = transform_func(rand);
      }
    );
    // 检查 CUDA 核函数的启动情况
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // 创建偏移计算器，用于非简单一维情况
    auto offset_calc = make_offset_calculator<1>(iter);
    // 调用元素级别的 grid-stride 内核函数，使用 grid 和 block 尺寸
    // 该函数在 GPU 上执行，处理每个元素
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      // Lambda 函数，以设备端方式捕获，处理索引和随机数
      [=]__device__(int idx, accscalar_t rand) {
        auto offsets = offset_calc.get(idx);
        scalar_t* out = (scalar_t*)&out_data[offsets[0]];
        *out = transform_func(rand);
      }
    );
    // 检查 CUDA 核函数的启动情况
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// 二进制核心函数模板
template <typename func_t, typename inp_offset_calc_t, typename out_offset_calc_t>
__global__ void distribution_binary_elementwise_kernel(
    // 元素总数
    int numel,
    // 函数对象
    func_t f,
    // Philox CUDA 状态
    PhiloxCudaState philox_args,
    // 输出数据指针
    typename function_traits<func_t>::result_type *output_data,
    // 输入数据1指针
    const typename function_traits<func_t>::template arg<1>::type *input_data_1,
    // 输入数据2指针
    const typename function_traits<func_t>::template arg<2>::type *input_data_2,
    // 输入偏移计算器
    inp_offset_calc_t inp_calc,
    // 输出偏移计算器
    out_offset_calc_t out_calc) {
  
  // 从 Philox 参数中解包种子
  auto seeds = at::cuda::philox::unpack(philox_args);

  // 定义输入数据类型
  using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
  using input_t_2 = typename function_traits<func_t>::template arg<2>::type;

  // 定义输入数据1和数据2的数组
  input_t_1 inputs_1[thread_work_size()];
  input_t_2 inputs_2[thread_work_size()];

  // 计算当前块的基础索引和剩余数量
  int base_index = block_work_size() * blockIdx.x;
  int remaining = std::min<int>(numel - base_index, block_work_size());

  // 初始化随机状态
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              blockIdx.x * blockDim.x + threadIdx.x,
              std::get<1>(seeds),
              &state);

  // 将数据加载到寄存器中
  int thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int input_idx = thread_idx + base_index;
    auto offsets = inp_calc.get(input_idx);
    inputs_1[i] = input_data_1[offsets[0]];
    inputs_2[i] = input_data_2[offsets[1]];

    thread_idx += num_threads();
  }

  // 计算并存储结果
  thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int input_idx = thread_idx + base_index;
    auto offsets = out_calc.get(input_idx);
    output_data[offsets[0]] = f(state, inputs_1[i], inputs_2[i]);
    thread_idx += num_threads();
  }
}

// 分发二进制核心函数模板
template <typename func_t>
void distribution_binary_kernel(TensorIteratorBase &iter, PhiloxCudaState philox_args, const func_t &f) {
  // 确保函数对象的第一个参数是 curandStatePhilox4_32_10_t 的类型
  static_assert(std::is_same<typename function_traits<func_t>::template arg<0>::type, curandStatePhilox4_32_10_t&>::value, "the first argument of functor must be curandStatePhilox4_32_10_t");

  // 定义输入数据1、数据2类型和输出数据类型
  using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
  using input_t_2 = typename function_traits<func_t>::template arg<2>::type;
  using output_t = typename function_traits<func_t>::result_type;

  // 如果迭代器不支持32位索引，使用32位索引进行迭代
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_binary_kernel(sub_iter, philox_args, f);
    }
    return;
  }

  // 断言迭代器可以使用32位索引
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(iter.can_use_32bit_indexing());

  // 获取元素总数
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  output_t *output_data = static_cast<output_t *>(iter.data_ptr(0));
  const input_t_1 *input_data_1 = static_cast<const input_t_1 *>(iter.data_ptr(1));
  const input_t_2 *input_data_2 = static_cast<const input_t_2 *>(iter.data_ptr(2));

  // 计算执行的网格数，确保处理所有元素
  int64_t grid = (numel + block_work_size() - 1) / block_work_size();
  // 获取当前 CUDA 流
  auto stream = at::cuda::getCurrentCUDAStream();

  // 检查迭代器是否是连续的
  if (iter.is_contiguous()) {
    // 如果连续，使用分发二进制逐元素操作的 CUDA 内核
    distribution_binary_elementwise_kernel<<<grid,num_threads(), 0, stream>>>(
        numel, f, philox_args, output_data, input_data_1, input_data_2,
        TrivialOffsetCalculator<2>(), TrivialOffsetCalculator<1>());
    // 检查 CUDA 内核启动是否成功
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // 如果不连续，使用自定义的输入输出偏移计算器创建分发二进制逐元素操作的 CUDA 内核
    distribution_binary_elementwise_kernel<<<grid, num_threads(), 0, stream>>>(
        numel, f, philox_args, output_data, input_data_1, input_data_2,
        make_input_offset_calculator<2>(iter), make_output_offset_calculator(iter));
    // 检查 CUDA 内核启动是否成功
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
// 定义了一个命名空间 cuda，在此命名空间下定义了一个名为 templates 的子命名空间，其内部包含一个名为 cuda 的子命名空间
namespace at {
namespace native {
namespace templates {
namespace cuda {

// ==================================================== Random ========================================================

// 定义了一个模板函数 random_from_to_kernel，接受一个 RNG 类型的参数 gen 和一个 TensorIteratorBase 类型的引用 iter
template<typename RNG>
void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, RNG gen) {
  // 根据 tensor 迭代器的数据类型调度对应的 CUDA 函数，函数名为 "random_from_to_kernel_cuda"
  AT_DISPATCH_V2(iter.dtype(), "random_from_to_kernel_cuda", AT_WRAP([&] {
    // 检查数据类型是否为 int64_t、double、float 或者 at::BFloat16，且 range 大于等于 2^32
    if ((std::is_same<scalar_t, int64_t>::value ||
         std::is_same<scalar_t, double>::value ||
         std::is_same<scalar_t, float>::value ||
         std::is_same<scalar_t, at::BFloat16>::value) && range >= 1ULL << 32) {
      // 定义一个 lambda 表达式 random_func，在设备端执行，用于生成在 [base, base+range) 范围内均匀分布的随机数
      auto random_func = [range, base] __device__ (uint64_t rand) {
        return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
      };
      // 调用 distribution_nullary_kernel 函数处理数据分发任务，使用 uint64_t 类型的 rand 作为输入
      distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
        gen,
        // 定义一个 lambda 表达式，生成 curand4 引擎的随机状态
        [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
          ulonglong2 ret;
          uint4 rand_val = curand4(state);
          ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
          ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
          return ret;
        },
        random_func);
    } else {
      // 定义一个 lambda 表达式 random_func，在设备端执行，用于生成在 [base, base+range) 范围内均匀分布的随机数
      auto random_func = [range, base] __device__ (uint32_t rand) {
        return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
      };
      // 调用 distribution_nullary_kernel 函数处理数据分发任务，使用 uint32_t 类型的 rand 作为输入
      distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
        gen,
        // 定义一个 lambda 表达式，生成 curand4 引擎的随机状态
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand4(state);
        },
        random_func);
    }
   }), AT_EXPAND(AT_ALL_TYPES), kBool, kHalf, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
// 定义了一个模板函数 random_full_64_bits_range_kernel，接受一个 RNG 类型的参数 gen 和一个 TensorIteratorBase 类型的引用 iter
template<typename RNG>
void random_full_64_bits_range_kernel(TensorIteratorBase& iter, RNG gen) {
  // 根据 tensor 迭代器的数据类型调度对应的 CUDA 函数，函数名为 "random_full_64_bits_range_kernel_cuda"
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cuda", [&] {
    // 检查模板参数是否为 int64_t, double, float 或 at::BFloat16 中的一种
    if (std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) {
      // 定义一个 lambda 函数 random_func，用于生成随机数
      auto random_func = [] __device__ (uint64_t rand) {
        return transformation::uniform_int_full_range<scalar_t>(rand);
      };
      // 调用 distribution_nullary_kernel 函数进行随机数分布计算
      distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
        gen,
        // 定义一个 lambda 函数，用于从 curandStatePhilox4_32_10_t 状态生成两个 64 位整数
        [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
          ulonglong2 ret;
          // 调用 curand4 函数生成一个 uint4 类型的随机数
          uint4 rand_val = curand4(state);
          // 将生成的四个 32 位随机数合并成两个 64 位随机数
          ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
          ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
          return ret;
        },
        random_func);
    } else {
      // 如果模板参数不是指定的类型之一，则抛出错误信息
      TORCH_CHECK(false, "random_full_64_bits_range_kernel_cuda handles only int64, double, float and bfloat16");
    }
// 结构体模板 RandomFromToKernel，用于生成介于指定范围内随机数的操作
template<typename RNG>
struct RandomFromToKernel {
  // 如果传入了范围和基数，则调用此函数，使用指定的随机数生成器生成随机数
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen) {
    // 调用 random_from_to_kernel 函数进行随机数生成
    random_from_to_kernel(iter, range, base, check_generator<RNG>(gen));
  }
  // 如果未传入范围和基数，则调用此函数，使用指定的随机数生成器生成满 64 位范围内的随机数
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    // 调用 random_full_64_bits_range_kernel 函数进行随机数生成
    random_full_64_bits_range_kernel(iter, check_generator<RNG>(gen));
  }
};

// 函数模板 random_kernel，调度和分发不同数据类型的随机数生成
template<typename RNG>
void random_kernel(TensorIteratorBase& iter, RNG gen) {
  // 使用 AT_DISPATCH_ALL_TYPES_AND3 宏展开的 Lambda 表达式，处理所有数据类型和三种特殊类型
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "random_kernel_cuda", [&] {
    // 如果数据类型是 double 或 int64_t
    if (std::is_same<scalar_t, double>::value || std::is_same<scalar_t, int64_t>::value) {
      // 定义在 GPU 上执行的 Lambda 函数 random_func，用于处理随机数变换
      auto random_func = [] __device__ (uint64_t rand) {
        return transformation::uniform_int<scalar_t>(rand);
      };
      // 调用 distribution_nullary_kernel 函数，使用 curand4_engine_calls/2 引擎调用数
      distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter, gen,
        // 定义在 GPU 上执行的 Lambda 函数，用于从 curandStatePhilox4_32_10_t 状态生成随机数
        [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
          ulonglong2 ret;
          uint4 rand_val = curand4(state);
          ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
          ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
          return ret;
        },
        random_func);
    } else {
      // 如果数据类型不是 double 或 int64_t
      // 定义在 GPU 上执行的 Lambda 函数 random_func，用于处理随机数变换
      auto random_func = [] __device__ (uint32_t rand) {
        return transformation::uniform_int<scalar_t>(rand);
      };
      // 调用 distribution_nullary_kernel 函数，使用 curand4_engine_calls 引擎调用数
      distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
        gen,
        // 定义在 GPU 上执行的 Lambda 函数，用于从 curandStatePhilox4_32_10_t 状态生成随机数
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand4(state);
        },
        random_func);
    }
  });
}

// 结构体模板 RandomKernel，用于统一调用 random_kernel 函数
template<typename RNG>
struct RandomKernel {
  // 调用 random_kernel 函数，对迭代器 iter 执行随机数生成操作
  void operator()(TensorIteratorBase& iter, RNG gen) {
    random_kernel(iter, gen);
  }
};

// ====================================================================================================================

// 函数模板 uniform_and_transform，生成均匀分布随机数并进行变换
template<typename scalar_t, typename accscalar_t, size_t curand4_engine_calls, typename RNG, typename transform_t>
void uniform_and_transform(TensorIteratorBase& iter, RNG gen, transform_t transform) {
  // 如果数据类型是 double
  if (std::is_same<scalar_t, double>::value) {
    // 调用 distribution_nullary_kernel 函数，使用 curand4_engine_calls/2 引擎调用数和双精度浮点数生成器
    distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
      gen,
      // 定义在 GPU 上执行的 Lambda 函数，用于从 curandStatePhilox4_32_10_t 状态生成双精度浮点数均匀分布随机数
      [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
      transform);
  } else {
    // 如果数据类型不是 double
    // 调用 distribution_nullary_kernel 函数，使用 curand4_engine_calls 引擎调用数和指定类型生成器
    distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
      gen,
      // 定义在 GPU 上执行的 Lambda 函数，用于从 curandStatePhilox4_32_10_t 状态生成随机数
      [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
      transform);
  }
}

// 函数模板 normal_and_transform，生成正态分布随机数并进行变换
template<typename scalar_t, typename accscalar_t, size_t curand4_engine_calls, typename RNG, typename transform_t>
void normal_and_transform(TensorIteratorBase& iter, RNG gen, transform_t transform) {
  // 如果数据类型是 double
  if (std::is_same<scalar_t, double>::value) {
    // 调用 distribution_nullary_kernel 函数，使用 curand4_engine_calls/2 引擎调用数和双精度浮点数生成器
    distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
      gen,
      // 定义在 GPU 上执行的 Lambda 函数，用于从 curandStatePhilox4_32_10_t 状态生成双精度浮点数正态分布随机数
      [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
      transform);
  } else {
    // 如果数据类型不是 double
    // 调用 distribution_nullary_kernel 函数，使用 curand4_engine_calls 引擎调用数和指定类型生成器
    distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
      gen,
      // 定义在 GPU 上执行的 Lambda 函数，用于从 curandStatePhilox4_32_10_t 状态生成随机数
      [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
      transform);
  }
}
    // 如果条件成立，执行以下代码块
    if (curand_version == 2) {
      // 调用模板函数 distribution_nullary_kernel，传入参数 iter, gen, 以及一个 lambda 表达式作为参数
      // lambda 表达式在设备上执行，使用 curand_normal2_double 函数从 state 中生成双精度正态分布随机数
      // curand_normal2_double 是 curand 库中的函数，用于生成双精度正态分布的随机数
      // transform 是一个参数，可能用于进一步处理生成的随机数
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
        transform);
    } else {
      // 如果条件不成立，执行以下代码块
      // 调用模板函数 distribution_nullary_kernel，传入参数 iter, gen, 以及一个 lambda 表达式作为参数
      // lambda 表达式在设备上执行，使用 curand_normal4 函数从 state 中生成四个正态分布随机数
      // curand_normal4 是 curand 库中的函数，用于生成四个正态分布的随机数
      // transform 是一个参数，可能用于进一步处理生成的随机数
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
        transform);
    }
}

// ==================================================== Normal ========================================================

// 定义模板函数，用于在 CUDA 设备上执行正态分布生成操作
template<typename RNG>
void normal_kernel(const TensorBase &self, double mean_, double std_, RNG gen) {
  // 借用空操作创建张量迭代器
  auto iter = TensorIterator::borrowing_nullary_op(self);
  // 针对所有浮点数类型（包括半精度和 BF16），生成 CUDA 核心的正态分布操作
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "normal_kernel_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // 定义 lambda 函数，用于乘以标准差并加上均值
    auto normal_func = [mean, std] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::normal<accscalar_t>(rand, mean, std));
    };
    // 执行正态分布生成和转换操作
    normal_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, normal_func);
   });
}

// 定义模板结构体，包装正态分布生成操作
template<typename RNG>
struct NormalKernel {
  // 函数调用运算符，调用正态分布生成操作
  void operator()(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
    normal_kernel(self, mean, std, check_generator<RNG>(gen));
  }
};

// ==================================================== Uniform ========================================================

// 定义模板函数，用于在 CUDA 设备上执行均匀分布生成操作
template<typename RNG>
void uniform_kernel(TensorIteratorBase& iter, double from_, double to_, RNG gen) {
  // 针对所有浮点数类型（包括半精度和 BF16），生成 CUDA 核心的均匀分布操作
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "uniform_kernel_cuda", [&] {
    auto from = static_cast<scalar_t>(from_);
    auto to = static_cast<scalar_t>(to_);
    using opmath_t = at::opmath_type<scalar_t>;
    auto range = static_cast<opmath_t>(to-from);
    // 定义 lambda 函数，用于反转边界，乘以范围并加上起始点
    auto uniform_func = [range, from, to] __device__ (opmath_t rand) {
      // 计算反转边界前的输出值
      // 在修改此代码之前，请阅读：https://github.com/pytorch/pytorch/issues/96947
      auto value = static_cast<scalar_t>(rand * range + from);
      // 反转 curand4 的边界从 (0, 1] 到 [0, 1)
      // 注意：此方法来自传统的 THCTensorRandom，并且可能会产生更多的 0 值，因为得到 1 的概率高于得到 0
      // 通过反转边界，我们改变了得到 1 和 0 的概率
      // 在修改此代码之前，请阅读：https://github.com/pytorch/pytorch/issues/16706
      auto reverse_bound_value = value == to ? from : value;
      return reverse_bound_value;
    };
    // 执行均匀分布生成和转换操作
    uniform_and_transform<scalar_t, opmath_t, curand4_engine_calls>(iter, gen, uniform_func);
   });
}

// 定义模板结构体，包装均匀分布生成操作
template<typename RNG>
struct UniformKernel {
  // 函数调用运算符，调用均匀分布生成操作
  void operator()(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
    uniform_kernel(iter, from, to, check_generator<RNG>(gen));
  }
};

// ================================================== LogNormal =======================================================
void log_normal_kernel(TensorIteratorBase& iter, double mean_, double std_, RNG gen) {
  // 使用宏AT_DISPATCH_FLOATING_TYPES_AND2分派到不同浮点类型的处理函数，函数名为"log_normal_cuda"
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_cuda", [&] {
    // 定义accscalar_t为scalar_t的精度推断类型，带有溢出检查
    using accscalar_t = at::acc_type<scalar_t, true>;
    // 将mean_和std_转换为accscalar_t类型
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // 定义log_normal_func lambda函数，用于对数正态分布变换
    auto log_normal_func = [mean, std] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::log_normal<accscalar_t>(transformation::normal<accscalar_t>(rand, mean, std)));
    };
    // 调用normal_and_transform函数，使用log_normal_func进行处理
    normal_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, log_normal_func);
   });
}

template<typename RNG>
struct LogNormalKernel {
  // LogNormalKernel结构体的operator()函数，调用log_normal_kernel函数进行处理
  void operator()(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
    log_normal_kernel(iter, mean, std, check_generator<RNG>(gen));
  }
};

// =================================================== Geometric ======================================================

template<typename RNG>
void geometric_kernel(TensorIteratorBase& iter, double p, RNG gen) {
  // 使用宏AT_DISPATCH_ALL_TYPES_AND2分派到所有类型的处理函数，函数名为"geometric_cuda"
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "geometric_cuda", [&] {
    // 定义accscalar_t为scalar_t的离散分布类型
    using accscalar_t = at::DiscreteDistributionType<scalar_t>::type;
    // 定义geometric_func lambda函数，用于几何分布变换
    auto geometric_func = [p] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::geometric<accscalar_t>(rand, p));
    };
    // 调用uniform_and_transform函数，使用geometric_func进行处理
    uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, geometric_func);
  });
}

template<typename RNG>
struct GeometricKernel {
  // GeometricKernel结构体的operator()函数，调用geometric_kernel函数进行处理
  void operator()(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
    geometric_kernel(iter, p, check_generator<RNG>(gen));
  }
};

// ================================================== Exponential =====================================================

template<typename RNG>
void exponential_kernel(TensorIteratorBase& iter, double lambda_, RNG gen) {
  // 检查iter的dtype是否为浮点类型，否则抛出错误信息
  TORCH_CHECK(isFloatingType(iter.dtype()), "Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified ", iter.dtype());
  // 使用宏AT_DISPATCH_FLOATING_TYPES_AND2分派到不同浮点类型的处理函数，函数名为"exponential_cuda"
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "exponential_cuda", [&] {
    // 定义accscalar_t为scalar_t的精度推断类型，带有溢出检查
    using accscalar_t = at::acc_type<scalar_t, true>;
    // 将lambda_转换为accscalar_t类型
    auto lambda = static_cast<accscalar_t>(lambda_);
    // 定义exponential_func lambda函数，用于指数分布变换
    auto exponential_func = [lambda] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::exponential<accscalar_t>(rand, lambda));
    };
    // 调用uniform_and_transform函数，使用exponential_func进行处理
    uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, exponential_func);
   });
}

template<typename RNG>
struct ExponentialKernel {
  // ExponentialKernel结构体的operator()函数，调用exponential_kernel函数进行处理
  void operator()(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
    exponential_kernel(iter, lambda, check_generator<RNG>(gen));
  }
};
    exponential_kernel(iter, lambda, check_generator<RNG>(gen));



// 调用 exponential_kernel 函数，传入参数 iter、lambda 和通过 check_generator 函数生成的 RNG 对象 gen
exponential_kernel(iter, lambda, check_generator<RNG>(gen));


这行代码调用了 `exponential_kernel` 函数，并传入了三个参数：`iter`、`lambda` 和通过 `check_generator<RNG>(gen)` 生成的 `RNG` 对象 `gen`。
};

// ==================================================== Cauchy ========================================================

template<typename RNG>
void cauchy_kernel(TensorIteratorBase& iter, double median_, double sigma_, RNG gen) {
  // 根据浮点类型分发处理函数，处理数据类型为浮点类型的情况以及Half和BFloat16类型
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "cauchy_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    // 将中位数和标准差转换为精度更高的数据类型
    auto median = static_cast<accscalar_t>(median_);
    auto sigma = static_cast<accscalar_t>(sigma_);
    // 定义柯西分布的变换函数
    auto cauchy_func = [median, sigma] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::cauchy<accscalar_t>(rand, median, sigma));
    };
    // 调用uniform_and_transform函数，将均匀分布的随机数变换为柯西分布的随机数
    uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, cauchy_func);
   });
}

template<typename RNG>
struct CauchyKernel {
  // CauchyKernel结构体的操作符重载函数，调用cauchy_kernel函数处理迭代器，中位数，标准差和随机数生成器
  void operator()(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
    cauchy_kernel(iter, median, sigma, check_generator<RNG>(gen));
  }
};

// ==================================================== Bernoulli =====================================================

template<typename scalar_t, typename prob_t>
void bernoulli_tensor_cuda_kernel(
    const TensorBase &ret, const at::TensorBase &p,
    PhiloxCudaState philox_args) {
  auto functor = [philox_args] __device__(
          int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4,
          const prob_t& p1, const prob_t& p2, const prob_t& p3, const prob_t& p4) {
        // 解包PhiloxCudaState结构体，获取种子值
        auto seeds = at::cuda::philox::unpack(philox_args);
        // 初始化curand状态
        curandStatePhilox4_32_10_t state;
        curand_init(std::get<0>(seeds),
                    blockIdx.x * blockDim.x + threadIdx.x,
                    std::get<1>(seeds),
                    &state);

        // 查看[curand调用的寄存器溢出问题，适用于CUDA < 10]注释
        float4 rand = curand_uniform4(&state);
        switch (n) {
          // 对每个元素进行伯努利采样
          case 4: {
            CUDA_KERNEL_ASSERT(0 <= p4 && p4 <= 1);
            v4 = static_cast<scalar_t>(rand.w <= p4);
            // fallthrough
          }
          case 3: {
            CUDA_KERNEL_ASSERT(0 <= p3 && p3 <= 1);
            v3 = static_cast<scalar_t>(rand.z <= p3);
            // fallthrough
          }
          case 2: {
            CUDA_KERNEL_ASSERT(0 <= p2 && p2 <= 1);
            v2 = static_cast<scalar_t>(rand.y <= p2);
            // fallthrough
          }
          case 1: {
            CUDA_KERNEL_ASSERT(0 <= p1 && p1 <= 1);
            v1 = static_cast<scalar_t>(rand.x <= p1);
          }
        }
      };
  // 使用CUDA_tensor_apply2模板函数，每次处理4个元素，对ret和p应用functor函数
  at::cuda::CUDA_tensor_apply2<scalar_t, const prob_t, 4, decltype(functor),
                               /*max_threads_per_block=*/512,
                               /*min_blocks_per_sm==*/2>(ret, p, functor);
}
template<typename RNG>
void bernoulli_kernel(const TensorBase &self, const TensorBase &p_, RNG gen) {
  // 创建一个用于CUDA状态的随机数引擎输入
  PhiloxCudaState rng_engine_inputs;
  {
    // 见注释 [使用随机生成器时获取锁]
    // 使用互斥锁保护，确保在使用随机生成器时获取锁
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(10);
  }
  // 检查概率张量是否为浮点类型
  TORCH_CHECK(at::isFloatingType(p_.scalar_type()), "expected probabilities tensor to have floating type, got ", p_.scalar_type());
  // 将概率张量强制转换为双精度（double）类型，如果self张量为双精度，则将其转换为双精度；否则转换为单精度（float）
  const auto p_type = self.dtype() == at::kDouble ? at::kDouble : at::kFloat;
  auto p_cuda = p_.to(TensorOptions().device(self.device()).dtype(p_type));
  auto p = expand_inplace(self, p_cuda);
  // 对self张量的所有数据类型进行分派，并调用对应的CUDA核函数
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, self.scalar_type(), "bernoulli_tensor_cuda_self_", [&] {
      if (std::is_same<scalar_t, double>::value) {
        // 如果scalar_t为双精度（double），则调用双精度版本的CUDA核函数
        return bernoulli_tensor_cuda_kernel<double, double>(self, *p, rng_engine_inputs);
      } else {
        // 否则调用与scalar_t对应的单精度（float）版本的CUDA核函数
        return bernoulli_tensor_cuda_kernel<scalar_t, float>(self, *p, rng_engine_inputs);
      }
   });
}

template<typename RNG>
void bernoulli_kernel(TensorIteratorBase& iter, double p, RNG gen) {
  // 对迭代器进行分派，并调用对应的CUDA标量核函数
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "bernoulli_scalar_cuda_", [&] {
      // 使用accscalar_t定义函数bernoulli_func，实现伯努利变换
      auto bernoulli_func = [p] __device__ (accscalar_t rand) {
        return static_cast<scalar_t>(transformation::bernoulli<accscalar_t>(rand, p));
      };
      // 调用uniform_and_transform函数，对iter进行均匀分布和变换操作
      uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, bernoulli_func);
   });
}

template<typename RNG>
struct BernoulliKernel {
  // 重载操作符()，调用TensorIteratorBase版本的伯努利核函数
  void operator()(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
    bernoulli_kernel(iter, p, check_generator<RNG>(gen));
  }
  // 重载操作符()，调用TensorBase版本的伯努利核函数
  void operator()(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen) {
    bernoulli_kernel(self, p_, check_generator<RNG>(gen));
  }
};
```