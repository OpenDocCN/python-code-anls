# `.\pytorch\aten\src\ATen\native\cpu\CrossKernel.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Cross.h>

#include <numeric>
#include <iterator>
#include <algorithm>
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>

namespace at::native {
namespace {

// 定义模板函数，用于计算向量的叉积
template<typename scalar_t>
static void apply_cross(const Tensor& result, const Tensor& a, const Tensor& b, const int64_t dim) {
  // 计算总元素数量，每个向量有三个分量
  int64_t total = a.numel() / 3;
  // 获取在指定维度上的步长
  int64_t a_stride = a.stride(dim);
  int64_t b_stride = b.stride(dim);
  int64_t r_stride = result.stride(dim);

  // 获取各个张量的数据指针
  const scalar_t *a_ptr = a.const_data_ptr<scalar_t>();
  const scalar_t *b_ptr = b.const_data_ptr<scalar_t>();
  scalar_t *r_ptr = result.data_ptr<scalar_t>();

  // 并行处理每个向量的计算
  parallel_for(0, total, internal::GRAIN_SIZE, [&](int64_t s, int64_t e) {
    const int64_t a_dim = a.dim();
    // 存储当前维度的位置信息
    std::vector<int64_t> position_in_dims(a_dim);
    int64_t index_in_curr_dim = s;
    int64_t a_start = 0;
    int64_t b_start = 0;
    int64_t r_start = 0;
    // 遍历除指定维度外的所有维度
    for (const auto i : c10::irange(a.dim())) {
      if (i == dim) continue;
      // 计算在当前维度上的起始位置
      position_in_dims[i] = index_in_curr_dim % a.size(i);
      a_start += (index_in_curr_dim % a.size(i)) * a.stride(i);
      b_start += (index_in_curr_dim % b.size(i)) * b.stride(i);
      r_start += (index_in_curr_dim % result.size(i)) * result.stride(i);
      index_in_curr_dim = index_in_curr_dim / a.size(i);
    }

    // 对每个向量执行叉积计算
    while (s < e) {
      r_ptr[r_start+0*r_stride] = a_ptr[a_start+1*a_stride]*b_ptr[b_start+2*b_stride] - a_ptr[a_start+2*a_stride]*b_ptr[b_start+1*b_stride];
      r_ptr[r_start+1*r_stride] = a_ptr[a_start+2*a_stride]*b_ptr[b_start+0*b_stride] - a_ptr[a_start+0*a_stride]*b_ptr[b_start+2*b_stride];
      r_ptr[r_start+2*r_stride] = a_ptr[a_start+0*a_stride]*b_ptr[b_start+1*b_stride] - a_ptr[a_start+1*a_stride]*b_ptr[b_start+0*b_stride];
      s++;

      // 更新每个维度上的位置信息
      for (const auto i : c10::irange(a.dim())) {
        if (i == dim) {
          continue;
        }
        position_in_dims[i]++;
        a_start += a.stride(i);
        b_start += b.stride(i);
        r_start += result.stride(i);
        // 如果当前维度达到边界并且不是最后一个维度，则重置起始位置和位置信息
        if (position_in_dims[i] == a.size(i) && i != a.dim()-1) {
            a_start -= position_in_dims[i] * a.stride(i);
            b_start -= position_in_dims[i] * b.stride(i);
            r_start -= position_in_dims[i] * result.stride(i);
            position_in_dims[i] = 0;
        } else {
          break;
        }
      }
    }
  });
}

// 实现跨张量叉积的核心函数
static void cross_kernel_impl(const Tensor& result, const Tensor& a, const Tensor& b, const int64_t dim) {
  // 分派不同类型的叉积计算
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, result.scalar_type(), "cross", [&]() {
    apply_cross<scalar_t>(result, a, b, dim);
  });
}

} // anonymous namespace

// 注册叉积操作的分派函数
REGISTER_DISPATCH(cross_stub, &cross_kernel_impl);

} // namespace at::native
```