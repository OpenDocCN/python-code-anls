# `.\pytorch\aten\src\ATen\native\cpu\FlashAttentionKernel.cpp`

```
// 定义宏，指定仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量头文件
#include <ATen/core/Tensor.h>

// 包含调度、并行处理、向量化等相关头文件
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
// 包含CPU端BLAS操作的头文件
#include <ATen/native/CPUBlas.h>
// 包含CPU端工具函数的头文件
#include <ATen/native/cpu/utils.h>
// 包含注意力机制相关的头文件
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
// 包含C++工具的头文件，提供了一些实用功能，例如范围迭代器
#include <c10/util/irange.h>

// 根据AT_PER_OPERATOR_HEADERS宏条件编译，选择性地包含功能头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

// 定义在at::native命名空间中的匿名命名空间
namespace at::native {

// 匿名命名空间下的内部函数定义开始

// out = val * a + b
// is_b_stride_zero: 如果b的步长为0（掩码广播情况），将b视为标量指针。
template <bool is_b_stride_zero, typename T1, typename T2>
inline void _scale_attn_mask_fusion_kernel(
    T1* a,
    T2* b,
    const int& size,
    T1* out,
    T1& val) {
  // 获取向量化类型的大小
  const auto vec_size1 = at::vec::Vectorized<T1>::size();
  const auto vec_size2 = at::vec::Vectorized<T2>::size();
  // 根据条件设置T1和T2的向量化因子
  constexpr int64_t T1_n =
      (vec_size2 == vec_size1 * 2 && is_reduced_floating_point_v<T2>) ? 2 : 1;
  constexpr int64_t T2_n = 1;
  // 创建用于向量化的比例值
  auto vec_scale = at::vec::VectorizedN<T1, T1_n>(val);
  int64_t i = 0;
  // 对于整个向量化范围执行循环
  for (; i < size - (size % vec_size2); i += vec_size2) {
    // 加载并向量化a中的数据
    auto a_n = at::vec::VectorizedN<T1, T1_n>::loadu(a + i);
    // 加载并向量化b中的数据
    at::vec::VectorizedN<T2, T2_n> b_n;
    // 根据是否b的步长为零，决定如何加载b的数据
    if constexpr(is_b_stride_zero) {
      b_n = at::vec::VectorizedN<T2, T2_n>((T1)b[0]);
    } else {
      b_n = at::vec::VectorizedN<T2, T2_n>::loadu(b + i);
    }
    // 转换b_n到与a_n相同的类型，进行乘法和加法操作，结果存储到out中
    auto b_n_convert = at::vec::convert<T1, T1_n, T2, T2_n, true>(b_n);
    auto res = a_n * vec_scale + b_n_convert;
    res.store(out + i); // 存储结果到out中
  }
  // 处理剩余的非向量化范围
  for (; i < size; i++) {
    auto tmp0 = a[i];
    T1 tmp1;
    // 根据是否b的步长为零，决定如何处理b的数据
    if constexpr(is_b_stride_zero) {
      tmp1 = (T1)b[0];
    } else {
      tmp1 = (T1)b[i];
    }
    // 计算乘法和加法操作，结果存储到out中
    out[i] = tmp0 * val + tmp1;
  }
}

// out = exp(a - val)
// val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  // 获取向量化类型的大小和初始值
  auto vec_size = vec::Vectorized<T1>::size();
  auto vec_max = vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = vec::Vectorized<T1>(tmp_sum);
  // 对于整个向量化范围执行循环
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    // 加载并向量化a中的数据，并执行指数运算
    auto tmp0 = vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2; // 累加到临时和中
    _store(out + i, tmp2); // 存储结果到out中
  }
  // 对于剩余的非向量化范围执行循环
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2; // 累加到临时和中
    out[i] = tmp2; // 存储结果到out中
  }
  val = tmp_sum; // 更新val为临时和的最终值
}

// out = a * scale
// max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
```  
# 定义一个函数，接受一个数组a、一个缩放因子scale、一个输出数组out，以及一个用于存储最大值的引用max。

  auto vec_size = vec::Vectorized<scalar_t>::size();
```  
# 使用vec::Vectorized<scalar_t>::size()函数获取标量类型scalar_t的向量化大小，并赋值给vec_size。

  auto vec_scale = vec::Vectorized<scalar_t>(scale);
```  
# 使用给定的scale创建一个标量类型scalar_t的向量vec_scale。

  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
```  
# 使用标量类型scalar_t的负无穷大值初始化一个临时变量tmp_max。

  auto vec_tmp_max = vec::Vectorized<scalar_t>(tmp_max);
```  
# 使用临时变量tmp_max创建一个标量类型scalar_t的向量vec_tmp_max。

  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
```  
# 从0到向量大小乘以整除size/vec_size之间的long型i进行迭代。

    auto tmp0 = vec::Vectorized<scalar_t>::loadu(a + i);
```  
# 加载a + i中的向量化标量类型scalar_t。

    auto tmp1 = tmp0 * vec_scale;
```  
# vec_scale中的 tmp0 * vec_scale。

 in
}



template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  // 检查 ptr2 是否为 nullptr，如果不是则抛出错误
  TORCH_CHECK(ptr2 == nullptr);
  // 返回 ptr
  return ptr;
}



template <typename scalar_t,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  // 如果 scalar_t 是降级浮点数类型，则返回 ptr2
  return ptr2;
}



template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  // 创建一个用 val 填充的向量
  Vec data_vec = Vec(val);
  int64_t d = 0;
  // 循环填充向量化处理的数据部分
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  // 对于剩余的数据，使用标量方式填充
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}



void reshape_attn_mask_to_4d(
    Tensor& attn_mask,
    int64_t batchSize,
    int64_t num_head,
    int64_t qSize,
    int64_t kvSize) {
  // 支持的注意力掩码形状：
  // 2维: ({Q_seq_len, 1}  x {KV_seq_len, 1})
  // 4维: ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})
  // 在 check_attn_mask_shape 中已经保证
  int64_t attn_mask_size_0 = 1;
  int64_t attn_mask_size_1 = 1;
  if (attn_mask.dim() == 4) {
    // 如果维度匹配 batchSize，则设置 attn_mask_size_0
    if (attn_mask.size(0) == batchSize) {
      attn_mask_size_0 = batchSize;
    }
    // 如果维度匹配 num_head，则设置 attn_mask_size_1
    if (attn_mask.size(1) == num_head) {
      attn_mask_size_1 = num_head;
    }
  }
  // 重新形状化注意力掩码为4维，并扩展至给定的 qSize 和 kvSize
  attn_mask = attn_mask
                .view({attn_mask_size_0, attn_mask_size_1, attn_mask.size(-2), attn_mask.size(-1)})
                .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
}



template <typename scalar_t, typename mask_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention(
    const Tensor& output,
    const Tensor& logsumexp,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    std::optional<Tensor> attn_mask,


注：以上为 C++ 代码的详细注释。
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // 将输入张量按照维度重新排列，以便后续的注意力计算
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  // 检查是否为降低精度的类型
  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  // 使用 opmath_type 计算累积类型
  using accum_t = at::opmath_type<scalar_t>;
  // 使用 Vectorized 类别别名 Vec
  using Vec = vec::Vectorized<accum_t>;
  // 计算缩放因子
  accum_t scaling_factor =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  // 检查输入张量的维度是否符合要求
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
        "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
  // 获取批量大小及各维度的大小信息
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  // 检查是否存在注意力掩码并且其元素个数不为零
  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();
  if (has_attn_mask) {
    // 调用函数将注意力掩码重塑为四维张量，参数分别为注意力掩码的值、批量大小、注意头数、查询尺寸和键值尺寸
    reshape_attn_mask_to_4d(attn_mask.value(), batchSize, num_head, qSize, kvSize);
  }

  // Strides
  // 获取查询张量的步幅信息
  int64_t qStrideB = query.stride(0);  // 批量维度步幅
  int64_t qStrideM = query.stride(1);  // 查询尺寸维度步幅
  int64_t qStrideH = query.stride(2);  // 注意头维度步幅
  // 获取键张量的步幅信息
  int64_t kStrideB = key.stride(0);    // 批量维度步幅
  int64_t kStrideN = key.stride(1);    // 键尺寸维度步幅
  int64_t kStrideH = key.stride(2);    // 注意头维度步幅
  // 获取值张量的步幅信息
  int64_t vStrideB = value.stride(0);  // 批量维度步幅
  int64_t vStrideN = value.stride(1);  // 值尺寸维度步幅
  int64_t vStrideH = value.stride(2);  // 注意头维度步幅
  // 获取输出张量的步幅信息
  int64_t oStrideB = output.stride(0); // 批量维度步幅
  int64_t oStrideM = output.stride(1); // 查询尺寸维度步幅
  int64_t oStrideH = output.stride(2); // 注意头维度步幅
  // 获取 logsumexp 张量的步幅信息
  int64_t lStrideB = logsumexp.stride(0); // 批量维度步幅
  int64_t lStrideM = logsumexp.stride(1); // 查询尺寸维度步幅
  int64_t lStrideH = logsumexp.stride(2); // 注意头维度步幅
  // 如果有注意力掩码且其尺寸大于1，则获取其步幅信息；否则置为0
  int64_t mStrideB =
      (has_attn_mask && attn_mask.value().size(0) > 1)
      ? attn_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (has_attn_mask && attn_mask.value().size(1) > 1)
      ? attn_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (has_attn_mask && attn_mask.value().size(2) > 1)
      ? attn_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (has_attn_mask && attn_mask.value().size(3) > 1)
      ? attn_mask.value().stride(3)
      : 0;

  // 计算拆分后的查询和键值的尺寸
  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  // 计算查询切片的数量
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  // 获取线程数
  int64_t num_thread = at::get_num_threads();

  // 获取查询张量的数据类型
  const auto dtype = query.scalar_type();
  // 将数据类型转换为累积类型
  const auto accumulate_dtype = toOpMathType(dtype);

  // 为每个线程分配临时缓冲区（累积类型）
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;
  // 创建一个空的张量 buf，用于存储临时数据，形状为 [num_thread, size_per_thread]
  at::Tensor buf = at::empty({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  // 创建一个空的张量 buf_reduced，用于存储临时数据，形状为 [num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0]
  at::Tensor buf_reduced = at::empty({num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0}, query.options());

  // 数据指针
  // 查询张量的常量数据指针
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  // 键张量的常量数据指针
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  // 值张量的常量数据指针
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  // 如果有注意力掩码，获取其数据指针；否则置为nullptr
  mask_t* mask_data = has_attn_mask
      ? attn_mask.value().data_ptr<mask_t>()
      : nullptr;
  // 输出张量的数据指针
  scalar_t* out_data = output.data_ptr<scalar_t>();
  // logsumexp 张量的数据指针
  accum_t* lse_data = logsumexp.data_ptr<accum_t>();
  // buf 张量的数据指针
  accum_t* buf_data = buf.data_ptr<accum_t>();
  // 如果是降维类型，buf_reduced 张量的数据指针；否则置为nullptr
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  // 并行循环，按照指定的范围并行执行
  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    // 初始化循环索引
    int64_t i = 0, j = 0, k = 0;
    // 调用函数初始化索引值
    data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    // 获取当前线程的索引
    int ompIdx = at::get_thread_num();
    // 获取当前线程的缓冲区指针
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    // 设置缓冲区中各个部分的数据指针
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize;
    accum_t* dst_data = qk_sum_data + qSplitSize;
    // 根据条件判断是否使用缩减类型的数据，若是，则指向缓冲区中对应位置的数据块，否则指向空指针
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize : nullptr;
    // 使用 OpenMP 多线程并行执行以下代码块
  });
// 定义了一个模板函数，用于执行反向传播的注意力机制操作
template <typename scalar_t, typename mask_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention_backward(
    const at::Tensor& grad_q,  // Q 的梯度张量
    const at::Tensor& grad_k,  // K 的梯度张量
    const at::Tensor& grad_v,  // V 的梯度张量
    const at::Tensor& grad_out,  // 输出的梯度张量
    const at::Tensor& query,  // 查询张量 Q
    const at::Tensor& key,  // 键张量 K
    const at::Tensor& value,  // 值张量 V
    const at::Tensor& out,  // 输出张量
    const at::Tensor& logsumexp,  // 对数总和指数
    double dropout_p,  // dropout 概率
    bool is_causal,  // 是否是因果注意力
    std::optional<Tensor> attn_mask,  // 可选的注意力掩码
    std::optional<double> scale) {  // 可选的缩放因子
  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;  // 累加类型
  using Vec = vec::Vectorized<accum_t>;  // 向量化类型
  accum_t scaling_factor = sdp::calculate_scale(query, scale).as_float_unchecked();  // 计算缩放因子

  // 检查 Q/K/V 的头部大小是否相同
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
        "scaled_dot_product_attention_flash_attention_backward: Q/K/V should have the same head size");
  
  // 计算张量的尺寸
  // Query (Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // Key   (Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value (Batch x KV_seq_len x Num_heads x Dim_per_head)
  int64_t batchSize = query.size(0);  // 批量大小
  int64_t qSize = query.size(1);  // 查询序列长度
  int64_t kvSize = value.size(1);  // 键值序列长度
  int64_t num_head = query.size(2);  // 注意力头的数量
  int64_t headSize = query.size(3);  // 每个注意力头的维度大小

  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();  // 是否存在注意力掩码
  if (has_attn_mask) {
    int64_t i = 0, j = 0;  // 初始化数据索引
    data_index_init(begin, i, batchSize, j, num_head);  // 初始化数据索引
    int ompIdx = at::get_thread_num();  // 获取 OpenMP 线程索引
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;  // 缓冲区指针
    accum_t* attn_data = buf_ptr;  // 注意力数据指针
    accum_t* grad_attn_data = attn_data + qSplitSize * kvSplitSize;  // 注意力梯度数据指针
    scalar_t* buf_reduced_ptr = is_reduced_type ? buf_reduced_data + ompIdx * size_per_thread_reduced : nullptr;  // 缩减类型的缓冲区指针
    scalar_t* attn_reduced_data = is_reduced_type ? buf_reduced_ptr : nullptr;  // 注意力缩减数据指针
    scalar_t* grad_attn_reduced_data = is_reduced_type ? attn_reduced_data + qSplitSize * kvSplitSize : nullptr;  // 注意力梯度缩减数据指针

    at::Tensor dsum = at::empty({qSplitSize}, query.options().dtype(accumulate_dtype));  // 创建累加张量
    accum_t* dsum_data = dsum.data_ptr<accum_t>();  // 累加数据指针
  }
});

// 定义宏，用于处理不同类型的掩码
#define AT_DISPATCH_MASK_TYPES(TYPE, NAME, ...)            \
  AT_DISPATCH_SWITCH(                                      \
      TYPE,                                                \
      NAME,                                                \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Bool, mask_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Float, mask_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Double, mask_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::BFloat16, mask_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Half, mask_t, __VA_ARGS__))
// 定义了一个用于执行闪存注意力机制的函数，接受多个张量作为输入参数
void flash_attention_kernel_impl(
    const Tensor& output,                    // 输出张量，存储计算结果
    const Tensor& logsumexp,                 // logsumexp 张量，包含 log(sum(exp(attn_scores))) 的结果
    const at::Tensor& query,                 // 查询张量，用于计算注意力分数
    const at::Tensor& key,                   // 键张量，用于计算注意力分数
    const at::Tensor& value,                 // 值张量，根据注意力分数计算加权和
    double dropout_p,                        // dropout 概率，用于随机丢弃注意力分数
    bool is_causal,                          // 布尔值，表示是否使用因果注意力机制
    std::optional<Tensor> attn_mask,         // 可选的注意力掩码张量，用于屏蔽无效位置的注意力
    std::optional<double> scale) {           // 可选的缩放因子，用于调整注意力分数的大小

  // 获取查询张量的序列长度
  auto q_seq_len = query.size(2);

  // 根据查询序列长度选择不同的计算方式
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, query.scalar_type(), "flash_attention", [&] {
    if (!attn_mask.has_value()) {
      // 如果没有指定注意力掩码
      if (q_seq_len >= 768) {
        // 根据序列长度选择不同的闪存注意力机制实现
        cpu_flash_attention<scalar_t, scalar_t, 256, 512>(
          output, logsumexp, query, key, value,
          dropout_p, is_causal, attn_mask, scale);
      } else if (q_seq_len >= 192) {
        cpu_flash_attention<scalar_t, scalar_t, 64, 512>(
          output, logsumexp, query, key, value,
          dropout_p, is_causal, attn_mask, scale);
      } else {
        cpu_flash_attention<scalar_t, scalar_t, 32, 512>(
          output, logsumexp, query, key, value,
          dropout_p, is_causal, attn_mask, scale);
      }
    } else {
      // 如果指定了注意力掩码，则根据掩码类型选择不同的计算方式
      AT_DISPATCH_MASK_TYPES(attn_mask.value().scalar_type(), "flash_attention_mask", [&]() {
        if (q_seq_len >= 768) {
          cpu_flash_attention<scalar_t, mask_t, 256, 512>(
            output, logsumexp, query, key, value,
            dropout_p, is_causal, attn_mask, scale);
        } else if (q_seq_len >= 192) {
          cpu_flash_attention<scalar_t, mask_t, 64, 512>(
            output, logsumexp, query, key, value,
            dropout_p, is_causal, attn_mask, scale);
        } else {
          cpu_flash_attention<scalar_t, mask_t, 32, 512>(
            output, logsumexp, query, key, value,
            dropout_p, is_causal, attn_mask, scale);
        }
      });
    }
  });
}

// 定义了用于闪存注意力机制反向传播的函数
void flash_attention_backward_kernel_impl(
    const at::Tensor& grad_q,                // 查询张量的梯度
    const at::Tensor& grad_k,                // 键张量的梯度
    const at::Tensor& grad_v,                // 值张量的梯度
    const at::Tensor& grad_out,              // 输出张量的梯度
    const at::Tensor& query,                 // 查询张量，用于计算梯度
    const at::Tensor& key,                   // 键张量，用于计算梯度
    const at::Tensor& value,                 // 值张量，用于计算梯度
    const at::Tensor& out,                   // 输出张量，用于计算梯度
    const at::Tensor& logsumexp,             // logsumexp 张量，用于计算梯度
    double dropout_p,                        // dropout 概率，用于计算梯度
    bool is_causal,                          // 布尔值，表示是否使用因果注意力机制
    std::optional<Tensor> attn_mask,         // 可选的注意力掩码张量，用于计算梯度
    std::optional<double> scale) {           // 可选的缩放因子，用于计算梯度

  // 确保 grad_out 是连续的，以避免在调用 gemm 函数时出现零步幅的情况
  auto grad_out_contig = grad_out.contiguous();
  // 获取查询张量的序列长度
  auto q_seq_len = query.size(1);

  // 根据查询序列长度选择不同的计算方式
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, query.scalar_type(), "flash_attention_backward", [&] {
    // 如果注意力掩码不存在或未定义
    if (!attn_mask.has_value() || !attn_mask.value().defined()) {
      // 定义积累类型为标量类型的操作数
      using accum_t = at::opmath_type<scalar_t>;
      // 根据查询序列长度选择合适的参数，调用 CPU 版本的 Flash Attention 反向传播
      if (q_seq_len >= 768) {
        cpu_flash_attention_backward<scalar_t, accum_t, 256, 512>(
          grad_q, grad_k, grad_v, grad_out_contig,
          query, key, value, out, logsumexp,
          dropout_p, is_causal, attn_mask, scale);
      } else if (q_seq_len >= 192) {
        cpu_flash_attention_backward<scalar_t, accum_t, 64, 512>(
          grad_q, grad_k, grad_v, grad_out_contig,
          query, key, value, out, logsumexp,
          dropout_p, is_causal, attn_mask, scale);
      } else {
        cpu_flash_attention_backward<scalar_t, accum_t, 32, 512>(
          grad_q, grad_k, grad_v, grad_out_contig,
          query, key, value, out, logsumexp,
          dropout_p, is_causal, attn_mask, scale);
      }
    } else {
      // 使用 AT_DISPATCH_MASK_TYPES 宏处理注意力掩码的类型
      AT_DISPATCH_MASK_TYPES(attn_mask.value().scalar_type(), "flash_attention_mask_backward", [&]() {
        // 根据查询序列长度选择合适的参数，调用 CPU 版本的 Flash Attention 反向传播
        if (q_seq_len >= 768) {
          cpu_flash_attention_backward<scalar_t, mask_t, 256, 512>(
            grad_q, grad_k, grad_v, grad_out_contig,
            query, key, value, out, logsumexp,
            dropout_p, is_causal, attn_mask, scale);
        } else if (q_seq_len >= 192) {
          cpu_flash_attention_backward<scalar_t, mask_t, 64, 512>(
            grad_q, grad_k, grad_v, grad_out_contig,
            query, key, value, out, logsumexp,
            dropout_p, is_causal, attn_mask, scale);
        } else {
          cpu_flash_attention_backward<scalar_t, mask_t, 32, 512>(
            grad_q, grad_k, grad_v, grad_out_contig,
            query, key, value, out, logsumexp,
            dropout_p, is_causal, attn_mask, scale);
        }
      });
    }
} // 结束匿名命名空间

} // 结束外部命名空间

// 注册 AVX512 分发函数，将 flash_attention_kernel_impl 注册为 flash_attention_kernel 的实现
ALSO_REGISTER_AVX512_DISPATCH(flash_attention_kernel, &flash_attention_kernel_impl);

// 注册 AVX512 分发函数，将 flash_attention_backward_kernel_impl 注册为 flash_attention_backward_kernel 的实现
ALSO_REGISTER_AVX512_DISPATCH(flash_attention_backward_kernel, &flash_attention_backward_kernel_impl);

} // 结束 at::native 命名空间
```