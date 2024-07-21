# `.\pytorch\aten\src\ATen\native\cpu\SoftMaxKernel.cpp`

```py
// 包含必要的头文件和宏定义
#include <memory>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cpu/SoftmaxKernel.h>

#include <algorithm>  // 包含标准库算法头文件
#include <iterator>   // 包含迭代器头文件
#include <numeric>    // 包含数值计算头文件

#include <ATen/Dispatch.h>        // ATen 分发机制头文件
#include <ATen/Parallel.h>        // ATen 并行计算头文件
#include <ATen/TensorIterator.h>  // ATen 张量迭代器头文件
#include <ATen/OpMathType.h>      // ATen 操作数数学类型头文件
#include <ATen/core/Tensor.h>     // ATen 张量核心头文件
#include <ATen/cpu/vec/functional.h>  // ATen CPU 向量功能头文件
#include <ATen/cpu/vec/vec.h>          // ATen CPU 向量头文件
#include <c10/util/Optional.h>         // c10 可选值头文件
#include <c10/util/irange.h>           // c10 迭代范围头文件
#include <ATen/OpMathType.h>           // ATen 操作数数学类型头文件

// [注意 AVX-SSE 转换] 通常我们在使用 AVX/AVX2 编译的代码中避免调用 cmath，
// 这是因为 SSE 到 AVX 的转换以及 Glibc2.23 中的一个 bug。
// 参见 https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280
//
// 关于 grainsize: grainsize 被选择为每个任务大约处理 GRAIN_SIZE 个计算。
// 每个任务在 dim_size 元素上工作。通过简单计算 (*, +, -) 计数为 1，exp 或 log 计数为 4，
// 16 应该是对 dim_size 元素每个计算的非常粗略的估计。
//
// 我们选择了一个能够适应 L1D 缓存的块大小。

namespace at::native {

namespace {
template <typename scalar_t>
// 在最后一个维度上计算向量化的 log softmax
inline void _vec_log_softmax_lastdim(
    const scalar_t* input_data_base,  // 输入数据的基地址
    scalar_t* output_data_base,       // 输出数据的基地址
    int64_t outer_size,               // 外部尺寸
    int64_t dim_size) {               // 维度尺寸
  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  // 现在偶然地，at::internal::GRAIN_SIZE 是 32768，这与许多处理器上 L1D 缓存的大小相同。
  // 现在一些处理器现在有 48 KB 的 L1D 缓存，因此也许在未来我们可以利用机器 L1D 缓存大小的知识。
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(
      1,
      at::internal::GRAIN_SIZE / (sizeof(scalar_t) * dim_size));
  int64_t CHUNK_SIZE = std::min<int64_t>(MAX_CHUNK_SIZE, outer_size);
  // 注意: grain_size 值为 0
  // 我们不改变 OpenMP 线程池中的线程数量，因此一些线程会执行有用的工作，而其他线程则不会。
  // 我们可以简单地使用 grain_size 为 0，并依赖 invoke_parallel 在线程之间均匀分配工作。
  // 我们计算 CHUNK_SIZE 来确保每个线程的计算是高效的。
  parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {
    // MSVC 要求这样声明动态数组
    // 来源: https://stackoverflow.com/a/33423538
    auto tmp_sum_scalar = std::make_unique<scalar_t[]>(CHUNK_SIZE);
    auto max_input_arr = std::make_unique<scalar_t[]>(CHUNK_SIZE);
    // 循环处理数据区间 [begin, end)，每次处理 CHUNK_SIZE 大小的数据块
    for (int64_t ii = begin; ii < end; ii += CHUNK_SIZE) {
      // 计算当前数据块的实际大小 loop_end
      int64_t loop_end = CHUNK_SIZE;
      if (ii + CHUNK_SIZE > end)
        loop_end = end - ii;
      
      // 第一轮循环：计算每个元素的最大值并存储在 max_input_arr 中
      for (const auto j : c10::irange(loop_end)) {
        int64_t i = ii + j;
        // 计算当前数据元素在输入数据数组中的偏移量
        const scalar_t* input_data = input_data_base + i * dim_size;
        // 使用向量化指令计算当前数据块的最大值
        max_input_arr[j] = vec::reduce_all<scalar_t>(
            [](Vec& x, Vec& y) { return vec::maximum(x, y); },
            input_data,
            dim_size);
      }
      
      // 第二轮循环：计算每个元素的临时和并存储在 tmp_sum_scalar 中
      for (const auto j : c10::irange(loop_end)) {
        int64_t i = ii + j;
        // 计算当前数据元素在输入数据数组中的偏移量
        const scalar_t* input_data = input_data_base + i * dim_size;
        // 获取前一步计算得到的最大值
        scalar_t max_input = max_input_arr[j];
        // 使用向量化指令计算当前数据块的临时和
        tmp_sum_scalar[j] = vec::map_reduce_all<scalar_t>(
            [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
            [](Vec x, Vec y) { return x + y; },
            input_data,
            dim_size);
      }
      
      // 对 tmp_sum_scalar 中的每个元素执行对数操作，使用向量化指令
      // 详见文档 [Note AVX-SSE transitions]，这里选择调用向量化版本以提高性能
      vec::map(
          [](Vec x) { return x.log(); },
          tmp_sum_scalar.get(),
          tmp_sum_scalar.get(),
          loop_end);
      
      // 第三轮循环：计算并更新输出数据
      for (const auto j : c10::irange(loop_end)) {
        int64_t i = ii + j;
        // 计算当前数据元素在输入数据数组中的偏移量和输出数据数组中的偏移量
        const scalar_t* input_data = input_data_base + i * dim_size;
        scalar_t* output_data = output_data_base + i * dim_size;
        // 获取前面计算得到的临时和和最大值
        scalar_t tmp_sum = tmp_sum_scalar[j];
        scalar_t max_input = max_input_arr[j];

        // 保持下面操作的顺序非常重要
        // 在处理大数和小差异时，如果先计算 `max_input + tmp_sum`，可能会出现数值问题
        // 详情参考 https://github.com/pytorch/pytorch/issues/11752#issuecomment-422883379
        // 使用向量化指令更新输出数据，确保计算顺序正确
        vec::map(
            [tmp_sum, max_input](Vec x) {
              return x - Vec(max_input) - Vec(tmp_sum);
            },
            output_data,
            input_data,
            dim_size);
      }
    }
}

template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax_lastdim(
    const scalar_t* input_data_base,  // 输入数据的基址（指针），假设为标量类型的指针
    scalar_t* output_data_base,       // 输出数据的基址（指针），假设为标量类型的指针
    int64_t outer_size,               // 外部尺寸，表示迭代的次数
    int64_t dim_size) {               // 维度大小，表示每个迭代中处理的元素数
  using Vec = vec::Vectorized<scalar_t>;  // 使用 Vectorized 类型来处理标量类型 scalar_t
  // 查看注释：grain_size 值为 0
  parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {  // 并行循环，从 begin 到 end
    for (const auto i : c10::irange(begin, end)) {  // 遍历每个并行块的元素范围
      const scalar_t* input_data = input_data_base + i * dim_size;  // 当前迭代的输入数据起始地址
      scalar_t* output_data = output_data_base + i * dim_size;      // 当前迭代的输出数据起始地址
      scalar_t max_input = vec::reduce_all<scalar_t>(  // 计算当前输入数据块的最大值
          [](Vec& x, Vec& y) { return vec::maximum(x, y); },
          input_data,
          dim_size);
      vec::map(
          [max_input](Vec x) { return (x - Vec(max_input)).exp(); },  // 对每个输入数据进行 softmax 操作
          output_data,
          input_data,
          dim_size);
      scalar_t tmp_sum = vec::reduce_all<scalar_t>(  // 计算 softmax 结果的总和
          [](Vec x, Vec y) { return x + y; }, output_data, dim_size);
      tmp_sum = 1 / tmp_sum;  // 计算归一化因子
      vec::map(
          [tmp_sum](Vec x) { return x * Vec(tmp_sum); },  // 将 softmax 结果归一化
          output_data,
          output_data,
          dim_size);
    }
  });
}

template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax_lastdim(
    const scalar_t* input_data_base,  // 输入数据的基址（指针），假设为非标量类型的指针
    scalar_t* output_data_base,       // 输出数据的基址（指针），假设为非标量类型的指针
    int64_t outer_size,               // 外部尺寸，表示迭代的次数
    int64_t dim_size) {               // 维度大小，表示每个迭代中处理的元素数
  using Vec = vec::Vectorized<scalar_t>;    // 使用 Vectorized 类型来处理标量类型 scalar_t
  using fVec = vec::Vectorized<float>;      // 使用 Vectorized 类型来处理 float 类型
  // 查看注释：grain_size 值为 0
  parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {  // 并行循环，从 begin 到 end
    // 线程本地临时缓冲区。
    auto buffer = std::make_unique<float []>(dim_size);  // 创建维度大小的唯一指针数组缓冲区
    float* buffer_data = buffer.get();  // 获取缓冲区的数据指针
    // 对于指定范围内的每个索引 i 进行循环处理
    for (const auto i : c10::irange(begin, end)) {
      // 计算当前输入数据和输出数据的基址
      const scalar_t* input_data = input_data_base + i * dim_size;
      scalar_t* output_data = output_data_base + i * dim_size;
      
      // 初始化用于存储最大值的向量，初始值为负无穷大
      fVec max_fvec = fVec(-std::numeric_limits<float>::infinity());
      int64_t d0 = 0;
      
      // 以向量化方式处理数据，每次处理 Vec::size() 个数据
      for (; d0 < dim_size - (dim_size % Vec::size()); d0 += Vec::size()) {
        // 加载并转换数据向量为浮点数向量
        Vec data_vec = Vec::loadu(input_data + d0);
        auto [data_fvec0, data_fvec1] = vec::convert_to_float<scalar_t>(data_vec);
        
        // 更新最大值向量
        max_fvec = vec::maximum(max_fvec, data_fvec0);
        max_fvec = vec::maximum(max_fvec, data_fvec1);
        
        // 存储浮点数数据到缓冲区
        data_fvec0.store(buffer_data + d0);
        data_fvec1.store(buffer_data + d0 + fVec::size());
      }
      
      // 处理剩余不足 Vec::size() 的数据
      float max_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return vec::maximum(x, y); }, max_fvec);
      for (; d0 < dim_size; d0++) {
        float data_val = input_data[d0];
        max_val = std::max(max_val, data_val);
        buffer_data[d0] = data_val;
      }

      // 初始化用于存储总和的浮点数向量
      fVec sum_fvec = fVec(float(0));
      int64_t d1 = 0;
      
      // 对缓冲区中的数据进行向量化处理，每次处理 fVec::size() 个数据
      for (; d1 < dim_size - (dim_size % fVec::size()); d1 += fVec::size()) {
        // 计算 (x - max).exp() 并累加到总和向量
        fVec data_fvec = (fVec::loadu(buffer_data + d1) - fVec(max_val)).exp();
        sum_fvec += data_fvec;
        
        // 存储处理后的数据回缓冲区
        data_fvec.store(buffer_data + d1);
      }
      
      // 处理剩余不足 fVec::size() 的数据
      float sum_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, sum_fvec);
      for (; d1 < dim_size; d1++) {
        float data_val = std::exp(buffer_data[d1] - max_val);
        sum_val += data_val;
        buffer_data[d1] = data_val;
      }

      // 计算总和的倒数
      sum_val = 1 / sum_val;
      int64_t d2 = 0;
      
      // 对输出数据进行向量化处理，每次处理 Vec::size() 个数据
      for (; d2 < dim_size - (dim_size % Vec::size()); d2 += Vec::size()) {
        // 加载数据并乘以总和的倒数，然后转换为输出数据类型
        fVec out_fvec0 = fVec::loadu(buffer_data + d2) * fVec(sum_val);
        fVec out_fvec1 = fVec::loadu(buffer_data + d2 + fVec::size()) * fVec(sum_val);
        Vec out_vec = vec::convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        
        // 存储处理后的数据到输出数据中
        out_vec.store(output_data + d2);
      }
      
      // 处理剩余不足 Vec::size() 的数据
      for (; d2 < dim_size; d2++) {
        output_data[d2] = scalar_t(buffer_data[d2] * sum_val);
      }
    }
// 结束函数模板的定义
}

template <typename scalar_t, bool log_softmax>
inline void _vec_host_softmax_backward_lastdim(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<at::opmath_type<scalar_t>>;
  // 定义并行操作的粒度为0，即由系统自动确定
  parallel_for(
      0,
      outer_size,
      0,
      [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
          // 计算当前迭代的梯度输入、梯度数据和输出数据的指针位置
          scalar_t* grad_input_data = grad_input_data_base + i * dim_size;
          const scalar_t* grad_data = grad_data_base + i * dim_size;
          const scalar_t* output_data = output_data_base + i * dim_size;
          // 声明并初始化变量 sum
          scalar_t sum;
          // 如果 log_softmax 为 true，则执行求和操作
          if (log_softmax) {
            sum = vec::reduce_all<scalar_t>(
                [](Vec& x, Vec& y) { return x + y; }, grad_data, dim_size);
          } else {
            // 否则执行 map2_reduce_all 操作，结合 grad_data 和 output_data
            sum = vec::map2_reduce_all<scalar_t>(
                [](Vec x, Vec y) { return x * y; },
                [](Vec x, Vec y) { return x + y; },
                grad_data,
                output_data,
                dim_size);
          }
          // 根据 log_softmax 的值执行不同的 vec::map2 操作
          if (log_softmax) {
            vec::map2(
                [sum](Vec x, Vec y) { return x - ((y.exp()) * Vec(sum)); },
                grad_input_data,
                grad_data,
                output_data,
                dim_size);
          } else {
            vec::map2(
                [sum](Vec x, Vec y) { return (x - Vec(sum)) * y; },
                grad_input_data,
                grad_data,
                output_data,
                dim_size);
          }
        }
      });
}

template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax_backward(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_output_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
}

template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax_backward(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_output_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
}

template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_log_softmax_backward(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_output_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
}

template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_log_softmax_backward(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_output_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
}
// 定义模板结构体 `vec_host_softmax_lastdim`，用于在最后一个维度上对张量进行 softmax 操作
template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_lastdim {
  // 静态方法 apply，用于在输出张量 `output` 上应用 softmax 操作，输入张量为 `input`
  static void apply(const Tensor& output, const Tensor& input) {
    // 计算张量的外部尺寸和最后一个维度的尺寸
    int64_t outer_size = 1;
    int64_t dim_size = input.size(input.ndimension() - 1);
    for (int64_t i = 0; i < input.ndimension() - 1; ++i)
      outer_size *= input.size(i);

    // 获取输入数据和输出数据的指针
    const scalar_t* input_data_base = input.const_data_ptr<scalar_t>();
    scalar_t* output_data_base = output.data_ptr<scalar_t>();

    // 根据 LogSoftMax 参数选择调用不同的底层函数进行 softmax 或 log_softmax 操作
    if (LogSoftMax) {
      _vec_log_softmax_lastdim(
          input_data_base, output_data_base, outer_size, dim_size);
    } else {
      _vec_softmax_lastdim(
          input_data_base, output_data_base, outer_size, dim_size);
    }
  }
};

// 当 scalar_t 不是 opmath_type<scalar_t> 时，实现 _vec_softmax 函数
template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
}

// 当 scalar_t 是 opmath_type<scalar_t> 时，实现 _vec_softmax 函数
template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
}

// 快速的 log_softmax 内核，当维度 dim != -1 时使用
// 输入张量形状被标准化为 {outer_size, dim_size, inner_size}
//
// 该算法要求加载输入张量 3 次，以增加并行性和缓存命中率，inner_size 被分块为：
//   inner_size: {CHUNK_SIZE, CHUNK_SIZE, ..., Remainder}
//
// 在 {outer_size, num_chunks} 上并行执行，对每个 {dim_size, CHUNK_SIZE} 的块进行垂直约简，
// 块大小 (128KB) 被选为 L2 命中。
//
template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_logsoftmax(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t BLOCK_SIZE = 128 * 1024;
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(BLOCK_SIZE / dim_size / sizeof(scalar_t), Vec::size());
  MAX_CHUNK_SIZE = MAX_CHUNK_SIZE / Vec::size() * Vec::size();
  int64_t CHUNK_SIZE = std::min<int64_t>(MAX_CHUNK_SIZE, inner_size);
  int64_t num_chunks = divup(inner_size, CHUNK_SIZE);

  // 注意: grain_size 值为 0
  // 使用并行库 ATen 的 parallel_for 函数，对每个 chunk 进行并行处理
  at::parallel_for(0, outer_size * num_chunks, 0, [&](int64_t begin, int64_t end) {
    // 线程局部的临时缓冲区，用于保存垂直约简结果：最大值和总和
    auto buffer = std::make_unique<scalar_t []>(CHUNK_SIZE * 2);
    scalar_t* input_max_data = buffer.get();
    scalar_t* tmp_sum_data = buffer.get() + CHUNK_SIZE;

    // 在这里实现并行操作的具体代码，处理每个 chunk 的数据
    // ...
    }
  });
}

// 当 scalar_t 不是 opmath_type<scalar_t> 时，实现 _vec_logsoftmax 函数
template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_logsoftmax(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
  // 定义函数参数，声明一个名为 dim_size 的 int64_t 类型的参数
  int64_t dim_size) {
  // 使用 Vec 和 fVec 别名分别表示 scalar_t 和 float 类型的向量化数据结构
  using Vec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  // 定义 BLOCK_SIZE 常量，表示块大小为 128KB
  int64_t BLOCK_SIZE = 128 * 1024;
  // 计算 MAX_CHUNK_SIZE，限制块大小，确保每块内存不超过 BLOCK_SIZE / dim_size / sizeof(scalar_t) 和 Vec::size() 中较小的值
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(BLOCK_SIZE / dim_size / sizeof(scalar_t), Vec::size());
  // 将 MAX_CHUNK_SIZE 调整为 Vec 对齐的最大值
  MAX_CHUNK_SIZE = MAX_CHUNK_SIZE / Vec::size() * Vec::size();
  // 计算 CHUNK_SIZE，限制块大小不超过 MAX_CHUNK_SIZE 和 inner_size 的较小值
  int64_t CHUNK_SIZE = std::min<int64_t>(MAX_CHUNK_SIZE, inner_size);
  // 计算 num_chunks，表示总块数，使用 divup 函数计算 inner_size 除以 CHUNK_SIZE 的上取整
  int64_t num_chunks = divup(inner_size, CHUNK_SIZE);

  // See Note: grain_size value of 0
  // 使用 at::parallel_for 函数并行处理任务范围为 0 到 outer_size * num_chunks
  at::parallel_for(0, outer_size * num_chunks, 0, [&](int64_t begin, int64_t end) {
    // 创建大小为 CHUNK_SIZE * 2 的 float 数组，buffer 指向这段内存
    auto buffer = std::make_unique<float []>(CHUNK_SIZE * 2);
    // input_max_data 指向 buffer 的起始位置
    float* input_max_data = buffer.get();
    // tmp_sum_data 指向 buffer 的后半段位置
    float* tmp_sum_data = buffer.get() + CHUNK_SIZE;

    // thread local buffer that holds input data in float32 to save next 2 dtype conversion
    // 创建大小为 dim_size * CHUNK_SIZE 的 float 数组，input_buffer 指向这段内存
    auto input_buffer = std::make_unique<float []>(dim_size * CHUNK_SIZE);
    // input_buffer_data 指向 input_buffer 的起始位置
    float* input_buffer_data = input_buffer.get();

    // init
    // 初始化任务处理逻辑
    }
  });
// 定义一个结构模板 vec_softmax，用于执行向量的 Softmax 或 LogSoftmax 操作
template <typename scalar_t, bool LogSoftMax>
struct vec_softmax {
  // 静态成员函数 apply，接受输出张量 output、输入张量 input 和维度 dim
  static void apply(const Tensor& output, const Tensor& input, int64_t dim) {
    // 初始化 outer_size 为 1，表示外部尺寸大小
    int64_t outer_size = 1;
    // 获取输入张量在指定维度 dim 上的大小
    int64_t dim_size = input.size(dim);
    // 初始化 inner_size 为 1，表示内部尺寸大小
    int64_t inner_size = 1;
    // 计算除了 dim 维度外的所有维度乘积，得到 outer_size
    for (const auto i : c10::irange(dim))
      outer_size *= input.size(i);
    // 计算从 dim+1 维度到最后一维的维度乘积，得到 inner_size
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    // 获取输入张量的数据指针
    const scalar_t* input_data_base = input.const_data_ptr<scalar_t>();
    // 获取输出张量的数据指针
    scalar_t* output_data_base = output.data_ptr<scalar_t>();
    // 如果 LogSoftMax 为真，则调用 _vec_logsoftmax 执行 LogSoftmax 操作
    if (LogSoftMax) {
      _vec_logsoftmax(
          input_data_base, output_data_base, outer_size, inner_size, dim_size);
    } else {
      // 否则调用 _vec_softmax 执行 Softmax 操作
      _vec_softmax(
          input_data_base, output_data_base, outer_size, inner_size, dim_size);
    }
  }
};

// 定义一个结构模板 vec_host_softmax_backward_lastdim，用于执行 Host 端最后一维 Softmax 的反向传播
template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_backward_lastdim {
  // 静态成员函数 apply，接受梯度输入张量 grad_input、梯度张量 grad 和输出张量 output
  static void
  apply(const Tensor& grad_input, const Tensor& grad, const Tensor& output) {
    // 初始化 outer_size 为 1，表示外部尺寸大小
    int64_t outer_size = 1;
    // 获取 grad 张量在最后一维的大小
    int64_t dim_size = grad.size(grad.ndimension() - 1);
    // 计算除了最后一维外的所有维度乘积，得到 outer_size
    for (int64_t i = 0; i < grad.ndimension() - 1; ++i)
      outer_size *= grad.size(i);
    // 获取 grad_input 张量的可变数据指针
    scalar_t* grad_input_data_base = grad_input.mutable_data_ptr<scalar_t>();
    // 获取 grad 张量的数据指针
    const scalar_t* grad_data_base = grad.const_data_ptr<scalar_t>();
    // 获取 output 张量的数据指针
    const scalar_t* output_data_base = output.const_data_ptr<scalar_t>();
    // 调用 _vec_host_softmax_backward_lastdim 执行最后一维 Softmax 的反向传播
    _vec_host_softmax_backward_lastdim<scalar_t, LogSoftMax>(
        grad_input_data_base,
        grad_data_base,
        output_data_base,
        outer_size,
        dim_size);
  }
};

// 定义一个结构模板 vec_host_softmax_backward，用于执行 Host 端 Softmax 的反向传播
template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_backward {
  // 静态成员函数 apply，接受梯度输入张量 grad_input、梯度张量 grad、输出张量 output 和维度 dim
  static void apply(
      const Tensor& grad_input,
      const Tensor& grad,
      const Tensor& output,
      int64_t dim) {
    // 初始化 outer_size 为 1，表示外部尺寸大小
    int64_t outer_size = 1;
    // 获取 grad 张量在指定维度 dim 上的大小
    int64_t dim_size = grad.size(dim);
    // 初始化 inner_size 为 1，表示内部尺寸大小
    int64_t inner_size = 1;
    // 计算除了 dim 维度外的所有维度乘积，得到 outer_size
    for (const auto i : c10::irange(dim)) {
      outer_size *= grad.size(i);
    }
    // 计算从 dim+1 维度到最后一维的维度乘积，得到 inner_size
    for (int64_t i = dim + 1; i < grad.dim(); ++i) {
      inner_size *= grad.size(i);
    }
    // 获取 grad_input 张量的可变数据指针
    scalar_t* grad_input_data_base = grad_input.mutable_data_ptr<scalar_t>();
    // 获取 grad 张量的数据指针
    const scalar_t* grad_output_data_base = grad.const_data_ptr<scalar_t>();
    // 获取 output 张量的数据指针
    const scalar_t* output_data_base = output.const_data_ptr<scalar_t>();
    // 如果 LogSoftMax 为真，则调用 _vec_log_softmax_backward 执行 LogSoftmax 的反向传播
    if (LogSoftMax) {
      _vec_log_softmax_backward<scalar_t>(
          grad_input_data_base,
          grad_output_data_base,
          output_data_base,
          outer_size,
          inner_size,
          dim_size);
    } else {
      // 否则调用 _vec_softmax_backward 执行 Softmax 的反向传播
      _vec_softmax_backward<scalar_t>(
          grad_input_data_base,
          grad_output_data_base,
          output_data_base,
          outer_size,
          inner_size,
          dim_size);
    }
  }
};

// 定义一个静态函数 softmax_lastdim_kernel_impl，实现最后一维 Softmax 的 CUDA 内核实现
static void softmax_lastdim_kernel_impl(
    const Tensor& result,
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，遍历 self 张量的浮点数类型（包括 BFloat16 和 Half 类型）
    AT_DISPATCH_FLOATING_TYPES_AND2(
        // self 张量的标量类型，用于生成适当的模板实例化
        at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
        // 调用的函数名称字符串，用于错误消息和调试目的
        "softmax_lastdim_kernel_impl",
        // lambda 函数开始，实际执行特定类型的 softmax 操作
        [&] {
            // 使用 vec_host_softmax_lastdim<scalar_t, false>::apply 函数对 self 进行 softmax 操作，并将结果存储在 result 中
            vec_host_softmax_lastdim<scalar_t, false>::apply(result, self);
        }
    );
static void softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
  // 分发浮点类型和半精度类型，应用 softmax 操作
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
    "softmax_kernel_impl",
    [&] { vec_softmax<scalar_t, false>::apply(result, self, dim); });
}

static void log_softmax_lastdim_kernel_impl(
    const Tensor& result,
    const Tensor& self) {
  // 分发浮点类型和半精度类型，应用在最后一个维度上的 log softmax 操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
      "log_softmax_lastdim_kernel_impl",
      [&] { vec_host_softmax_lastdim<scalar_t, true>::apply(result, self); });
}

static void log_softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
  // 分发浮点类型和半精度类型，应用 log softmax 操作
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
    "softmax_kernel_impl",
    [&] { vec_softmax<scalar_t, true>::apply(result, self, dim); });
}

static void softmax_backward_lastdim_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  // 分发梯度类型为浮点类型和半精度类型，应用在最后一个维度上的 softmax 反向传播操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, grad.scalar_type(),
      "softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, false>::apply(
            grad_input, grad, output);
      });
}

static void log_softmax_backward_lastdim_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  // 分发梯度类型为浮点类型和半精度类型，应用在最后一个维度上的 log softmax 反向传播操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, grad.scalar_type(),
      "log_softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, true>::apply(
            grad_input, grad, output);
      });
}

static void softmax_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output,
    int64_t dim) {
  // 分发梯度类型为浮点类型和半精度类型，应用 softmax 在指定维度上的反向传播操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      grad.scalar_type(),
      "softmax_backward_kernel_impl",
      [&] {
        vec_host_softmax_backward<scalar_t, false>::apply(
            grad_input, grad, output, dim);
      });
}

static void log_softmax_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output,
    int64_t dim) {
  // 分发梯度类型为浮点类型和半精度类型，应用 log softmax 在指定维度上的反向传播操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      grad.scalar_type(),
      "log_softmax_backward_kernel_impl",
      [&] {
        vec_host_softmax_backward<scalar_t, true>::apply(
            grad_input, grad, output, dim);
      });
}

} // 匿名命名空间结束

// 使用 AVX512 指令集注册 softmax 在最后一个维度上的优化实现
ALSO_REGISTER_AVX512_DISPATCH(softmax_lastdim_kernel, &softmax_lastdim_kernel_impl);

// 使用 AVX512 指令集注册 log softmax 在最后一个维度上的优化实现
ALSO_REGISTER_AVX512_DISPATCH(log_softmax_lastdim_kernel, &log_softmax_lastdim_kernel_impl);

// 使用 AVX512 指令集注册在最后一个维度上的 softmax 反向传播的优化实现
ALSO_REGISTER_AVX512_DISPATCH(
    softmax_backward_lastdim_kernel,
    &softmax_backward_lastdim_kernel_impl);

// 使用 AVX512 指令集注册在最后一个维度上的 log softmax 反向传播的优化实现
ALSO_REGISTER_AVX512_DISPATCH(
    log_softmax_backward_lastdim_kernel,
    &log_softmax_backward_lastdim_kernel_impl);
    &log_softmax_backward_lastdim_kernel_impl);
ALSO_REGISTER_AVX512_DISPATCH(softmax_kernel, &softmax_kernel_impl);
# 使用宏注册 AVX512 指令集的 softmax 核函数及其实现函数

ALSO_REGISTER_AVX512_DISPATCH(log_softmax_kernel, &log_softmax_kernel_impl);
# 使用宏注册 AVX512 指令集的 log_softmax 核函数及其实现函数

ALSO_REGISTER_AVX512_DISPATCH(softmax_backward_kernel, &softmax_backward_kernel_impl);
# 使用宏注册 AVX512 指令集的 softmax 反向传播核函数及其实现函数

ALSO_REGISTER_AVX512_DISPATCH(
    log_softmax_backward_kernel,
    &log_softmax_backward_kernel_impl);
# 使用宏注册 AVX512 指令集的 log_softmax 反向传播核函数及其实现函数

} // namespace at::native
# 结束命名空间 at::native
```