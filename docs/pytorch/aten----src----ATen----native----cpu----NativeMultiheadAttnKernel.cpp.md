# `.\pytorch\aten\src\ATen\native\cpu\NativeMultiheadAttnKernel.cpp`

```
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/transformers/attention.h>
#include <c10/util/irange.h>

namespace at::native {

// 匿名命名空间，用于隐藏实现细节
namespace {

// 模板函数，用于在 CPU 上变换偏置并重新缩放 QKV 数据
template <typename scalar_t>
void cpu_transform_bias_rescale_qkv(
    scalar_t* q_k_v_data,
    const scalar_t* qkv_data,
    const scalar_t* qkv_bias_data,
    int64_t B,
    int64_t T,
    int64_t D,
    int64_t num_head) {

  // 计算每个头的维度
  int64_t dim_per_head = D / num_head;

  // shapes and strides:
  //   qkv      : {B, T, 3, num_head, dim_per_head}
  //   qkv_bias : {3, num_head, dim_per_head}
  //   q_k_v    : {3, B, num_head, T, dim_per_head}
  //
  // 计算输入和输出的步幅
  int64_t i_strideB = T * 3 * D;
  int64_t i_strideT = 3 * D;
  int64_t o_stride = B * num_head * T * dim_per_head;

  // 定义累加类型 acc_t 为 opmath_type<scalar_t>
  using acc_t = at::opmath_type<scalar_t>;
  using Vec =  vec::Vectorized<acc_t>;
  // 计算倒数的平方根作为标量 s
  const acc_t s = 1.0 / std::sqrt(static_cast<acc_t>(dim_per_head));

  // 并行执行在 {B, num_head, T} 上
  // 使用 grain_size 控制并行粒度
  int64_t grain_size = std::max(at::internal::GRAIN_SIZE / (3 * dim_per_head), (int64_t)1);
  at::parallel_for(0, B * num_head * T, grain_size, [&](int64_t begin, int64_t end) {
    int64_t b{0}, nh{0}, t{0};
    // 初始化数据索引
    data_index_init(begin, b, B, nh, num_head, t, T);

    // 遍历每个索引
    for (const auto i : c10::irange(begin, end)) {
      // 获取输入指针
      const scalar_t* q_in_ptr = qkv_data + b * i_strideB + t * i_strideT + 0 * D + nh * dim_per_head;
      const scalar_t* k_in_ptr = qkv_data + b * i_strideB + t * i_strideT + 1 * D + nh * dim_per_head;
      const scalar_t* v_in_ptr = qkv_data + b * i_strideB + t * i_strideT + 2 * D + nh * dim_per_head;

      // 获取偏置指针
      const scalar_t* q_bias_ptr = qkv_bias_data + 0 * D + nh * dim_per_head;
      const scalar_t* k_bias_ptr = qkv_bias_data + 1 * D + nh * dim_per_head;
      const scalar_t* v_bias_ptr = qkv_bias_data + 2 * D + nh * dim_per_head;

      // 计算输出指针
      scalar_t* q_out_ptr = q_k_v_data + 0 * o_stride + i * dim_per_head;
      scalar_t* k_out_ptr = q_k_v_data + 1 * o_stride + i * dim_per_head;
      scalar_t* v_out_ptr = q_k_v_data + 2 * o_stride + i * dim_per_head;

      // q = (q + bias) * inv_sqrt_dim_per_head
      // 使用 SIMD 向量化操作来计算
      vec::map2<scalar_t>(
          [s](Vec q, Vec q_bias) { return (q + q_bias) * Vec(s); },
          q_out_ptr, q_in_ptr, q_bias_ptr, dim_per_head);

      // k = k + bias
      vec::map2<scalar_t>([](Vec k, Vec k_bias) { return k + k_bias; },
          k_out_ptr, k_in_ptr, k_bias_ptr, dim_per_head);

      // v = v + bias
      vec::map2<scalar_t>([](Vec v, Vec v_bias) { return v + v_bias; },
          v_out_ptr, v_in_ptr, v_bias_ptr, dim_per_head);

      // 移动到下一个索引
      data_index_step(b, B, nh, num_head, t, T);
    }
  });
}

// 变换偏置并重新缩放 QKV 数据的核心实现函数
void transform_bias_rescale_qkv_kernel_impl(
    // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，根据给定的标量类型 type 进行模板分发
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, type, "transform_bias_rescale_qkv", [&] {
        // 将 _q_k_v 强制转换为 scalar_t 类型指针，以便进行后续操作
        scalar_t* q_k_v = static_cast<scalar_t*>(_q_k_v);
        // 将 _qkv 强制转换为 const scalar_t 类型指针，作为输入数据
        const scalar_t* qkv = static_cast<const scalar_t*>(_qkv);
        // 将 _qkv_bias 强制转换为 const scalar_t 类型指针，作为偏置数据
        const scalar_t* qkv_bias = static_cast<const scalar_t*>(_qkv_bias);
        // 调用 cpu_transform_bias_rescale_qkv 函数，对 q_k_v 进行偏置和重新缩放操作
        cpu_transform_bias_rescale_qkv<scalar_t>(
            q_k_v,
            qkv,
            qkv_bias,
            B,
            T,
            D,
            num_head);
    });
}

} // anonymous namespace

REGISTER_DISPATCH(transform_bias_rescale_qkv_stub, &transform_bias_rescale_qkv_kernel_impl);
// 注册一个分发函数，将 transform_bias_rescale_qkv_stub 映射到 transform_bias_rescale_qkv_kernel_impl 函数上

} // at::native
```