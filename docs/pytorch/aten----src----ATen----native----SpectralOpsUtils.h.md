# `.\pytorch\aten\src\ATen\native\SpectralOpsUtils.h`

```py
#pragma once
// 防止头文件重复包含的预处理指令

#include <string>
// 包含标准字符串库

#include <stdexcept>
// 包含标准异常库

#include <sstream>
// 包含字符串流库

#include <c10/core/ScalarType.h>
// 包含PyTorch中张量数据类型相关的头文件

#include <c10/util/ArrayRef.h>
// 包含PyTorch中数组引用相关的头文件

#include <c10/util/Exception.h>
// 包含PyTorch中异常处理相关的头文件

#include <ATen/native/DispatchStub.h>
// 包含PyTorch中调度函数存根相关的头文件

#include <ATen/core/TensorBase.h>
// 包含PyTorch中张量基类相关的头文件

namespace at::native {
// 进入PyTorch的native命名空间

// _fft_with_size函数中使用的正规化类型
enum class fft_norm_mode {
  none,       // 无正规化
  by_root_n,  // 除以sqrt(signal_size)
  by_n,       // 除以signal_size
};

// NOTE [ Fourier Transform Conjugate Symmetry ]
//
// 实到复傅立叶变换满足共轭对称性。即，
// 假设X是变换后的K维信号，则有
//
//     X[i_1, ..., i_K] = X[j_i, ..., j_K]*,
//
// 其中j_k = (N_k - i_k) mod N_k，N_k为第k维信号的大小，
// *为共轭操作符。
//
// 因此，在这种情况下，FFT库仅返回大约一半的值以避免冗余：
//
//     X[:, :, ..., :floor(N / 2) + 1]
//
// 这也是cuFFT和MKL的假设。在ATen SpectralOps中，默认也会返回这种减半的信号（设置onesided=True）。
// 下面的infer_ft_real_to_complex_onesided_size函数计算从双边大小推导出单边大小。
//
// 注意，这会丢失关于最后一维信号大小的一些信息。例如，11和10都映射到6。因此，
// infer_ft_complex_to_real_onesided_size函数接受可选参数，从给定的单边大小推导出双边大小。
//
// cuFFT文档：http://docs.nvidia.com/cuda/cufft/index.html#multi-dimensional
// MKL文档：https://software.intel.com/en-us/mkl-developer-reference-c-dfti-complex-storage-dfti-real-storage-dfti-conjugate-even-storage#CONJUGATE_EVEN_STORAGE

// 推导实到复傅立叶变换单边大小的函数
inline int64_t infer_ft_real_to_complex_onesided_size(int64_t real_size) {
  return (real_size / 2) + 1;
}

// 推导复到实傅立叶变换单边大小的函数
inline int64_t infer_ft_complex_to_real_onesided_size(int64_t complex_size,
                                                      int64_t expected_size=-1) {
  int64_t base = (complex_size - 1) * 2;
  if (expected_size < 0) {
    return base + 1;
  } else if (base == expected_size) {
    return base;
  } else if (base + 1 == expected_size) {
    return base + 1;
  } else {
    std::ostringstream ss;
    ss << "expected real signal size " << expected_size << " is incompatible "
       << "with onesided complex frequency size " << complex_size;
    AT_ERROR(ss.str());
  }
}

// fft_fill_with_conjugate_symmetry_fn函数指针类型的别名
using fft_fill_with_conjugate_symmetry_fn =
    void (*)(ScalarType dtype, IntArrayRef mirror_dims, IntArrayRef half_sizes,
             IntArrayRef in_strides, const void* in_data,
             IntArrayRef out_strides, void* out_data);

// 声明fft_fill_with_conjugate_symmetry_stub调度函数存根
DECLARE_DISPATCH(fft_fill_with_conjugate_symmetry_fn, fft_fill_with_conjugate_symmetry_stub);

// 在实到复变换中，cuFFT和MKL只填充一半的值，因为共轭对称性。该函数填充完整的另一半
// 使用 Hermitian 对称性填充信号的 FFT。
// self 应该是完整信号的形状，dims.back() 应该是单边维度。
// 参见 NOTE [ Fourier Transform Conjugate Symmetry ]
TORCH_API void _fft_fill_with_conjugate_symmetry_(const Tensor& self, IntArrayRef dims);

} // namespace at::native
```