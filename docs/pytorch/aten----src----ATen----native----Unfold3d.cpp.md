# `.\pytorch\aten\src\ATen\native\Unfold3d.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/Unfold3d.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif // AT_MKL_ENABLED()

namespace at::native {

namespace {

// 检查 a 是否大于等于 0 且小于 b
bool IsAGeZeroAndALtB(int64_t a, int64_t b) {
  return static_cast<uint64_t>(a) < static_cast<uint64_t>(b);
}

// 复制矩阵 A 到 B，每次复制 N 个元素
template <typename T>
void MatCopy(int64_t M, int64_t N, int64_t lda, int64_t ldb, const T* A, T* B) {
  for (const auto i : c10::irange(M)) {
    std::memcpy(B + i * ldb, A + i * lda, N * sizeof(T));
  }
}

// 复制矩阵 A 到 B，每次复制 N 个元素，带有步长
template <typename T>
void MatCopy(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t stridea,
    int64_t ldb,
    int64_t strideb,
    const T* A,
    T* B) {
  for (const auto i : c10::irange(M)) {
    const T* A_ptr = A + i * lda;
    T* B_ptr = B + i * ldb;
    for (const auto j : c10::irange(N)) {
      B_ptr[j * strideb] = A_ptr[j * stridea];
    }
  }
}

// 矩阵加法 Y += X
template <typename T>
void MatAdd(int64_t M, int64_t N, int64_t ldx, int64_t ldy, const T* X, T* Y) {
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      Y[i * ldy + j] += X[i * ldx + j];
    }
  }
}

// 矩阵加法 Y += X，带有步长
template <typename T>
void MatAdd(
    int64_t M,
    int64_t N,
    int64_t ldx,
    int64_t stridex,
    int64_t ldy,
    int64_t stridey,
    const T* X,
    T* Y) {
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      Y[i * ldy + j * stridey] += X[i * ldx + j * stridex];
    }
  }
}

#if AT_MKL_ENABLED()

// 使用 MKL 库优化的单精度矩阵复制
template <>
void MatCopy<float>(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t ldb,
    const float* A,
    float* B) {
  mkl_somatcopy('R', 'N', M, N, 1.0f, A, lda, B, ldb);
}

// 使用 MKL 库优化的双精度矩阵复制
template <>
void MatCopy<double>(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t ldb,
    const double* A,
    double* B) {
  mkl_domatcopy('R', 'N', M, N, 1.0, A, lda, B, ldb);
}

// 使用 MKL 库优化的单精度矩阵复制，带有步长
template <>
void MatCopy<float>(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t stridea,
    int64_t ldb,
    int64_t strideb,
    const float* A,
    float* B) {
  mkl_somatcopy2('R', 'N', M, N, 1.0f, A, lda, stridea, B, ldb, strideb);
}

// 使用 MKL 库优化的双精度矩阵复制，带有步长
template <>
void MatCopy<double>(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t stridea,
    int64_t ldb,
    int64_t strideb,
    const double* A,
    double* B) {
  mkl_domatcopy2('R', 'N', M, N, 1.0, A, lda, stridea, B, ldb, strideb);
}

// 使用 MKL 库优化的单精度矩阵加法
template <>
void MatAdd<float>(
    int64_t M,
    int64_t N,
    int64_t ldx,
    int64_t ldy,
    const float* X,
    float* Y) {
  mkl_somatadd('R', 'N', 'N', M, N, 1.0f, X, ldx, 1.0f, Y, ldy, Y, ldy);
}

// 使用 MKL 库优化的双精度矩阵加法
template <>
void MatAdd<double>(
    int64_t M,
    int64_t N,
    int64_t ldx,
    int64_t ldy,
    const double* X,
    double* Y) {
  mkl_domatadd('R', 'N', 'N', M, N, 1.0, X, ldx, 1.0, Y, ldy, Y, ldy);
}

// 使用 MKL 库优化的矩阵加法，带有步长
template <>
void MatAdd(
    int64_t M,
    int64_t N,
    int64_t ldx,
    // 循环遍历 M 次，其中 M 为迭代次数，i 为当前迭代的索引
    for (const auto i : c10::irange(M)) {
        // 使用 cblas_saxpy 函数执行向量运算：Y[i * ldy] += X[i * ldx] * 1.0f
        // 其中，X 和 Y 是输入和输出向量指针，ldx 和 ldy 是 X 和 Y 的跨步大小
        // stridex 和 stridey 分别是 X 和 Y 的跨步大小
        cblas_saxpy(N, 1.0f, X + i * ldx, stridex, Y + i * ldy, stridey);
    }
#else // AT_MKL_ENABLED()

# 如果未启用 MKL，则编译以下代码段


template <typename T>
void Unfold3dZeroPaddingCopyKernelImpl(
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    const T* src,
    T* dst) {
  // 计算要处理的元素总数
  const int64_t n = C * kernel_d * kernel_h * kernel_w;
  // 计算输入张量的大小
  const int64_t X_size = X_D * X_H * X_W;
  // 计算输出张量的大小
  const int64_t Y_size = Y_D * Y_H * Y_W;
  // 并行处理每个元素
  at::parallel_for(0, n, 0, [=](int64_t begin, int64_t end) {
    for (const auto p : c10::irange(begin, end)) {
      int64_t c = p;
      // 计算当前元素在 kernel_w 维度上的偏移
      const int64_t kw = c % kernel_w;
      c /= kernel_w;
      // 计算当前元素在 kernel_h 维度上的偏移
      const int64_t kh = c % kernel_h;
      c /= kernel_h;
      // 计算当前元素在 kernel_d 维度上的偏移
      const int64_t kd = c % kernel_d;
      c /= kernel_d;
      // 遍历输出张量的深度维度
      for (const auto yd : c10::irange(Y_D)) {
        // 计算输入张量的深度维度上的偏移
        const int64_t xd = yd * stride_d + kd;
        // 计算输入张量中当前元素的起始位置
        const T* src_ptr = src + c * X_size + xd * X_H * X_W + kh * X_W + kw;
        // 计算输出张量中当前元素的起始位置
        T* dst_ptr = dst + p * Y_size + yd * Y_H * Y_W;
        // 根据条件选择复制函数
        if (stride_w == 1) {
          // 执行矩阵复制操作
          MatCopy<T>(Y_H, Y_W, stride_h * X_W, Y_W, src_ptr, dst_ptr);
        } else {
          // 执行矩阵复制操作，带有步幅参数
          MatCopy<T>(
              Y_H, Y_W, stride_h * X_W, stride_w, Y_W, 1, src_ptr, dst_ptr);
        }
      }
    }
  });
}
    for (const auto p : c10::irange(begin, end)) {
      // 遍历范围 [begin, end) 中的整数 p
      int64_t c = p;
      // 将 p 赋值给 c
      const int64_t kw = c % kernel_w;
      // 计算 kw，即 c 对 kernel_w 取模的结果
      c /= kernel_w;
      // 将 c 除以 kernel_w
      const int64_t kh = c % kernel_h;
      // 计算 kh，即 c 对 kernel_h 取模的结果
      c /= kernel_h;
      // 将 c 除以 kernel_h
      const int64_t kd = c % kernel_d;
      // 计算 kd，即 c 对 kernel_d 取模的结果
      c /= kernel_d;
      // 将 c 除以 kernel_d
      const T* src_ptr = src + c * X_size;
      // 计算 src_ptr，指向输入数组 src 中的特定位置
      T* dst_ptr = dst + p * Y_size;
      // 计算 dst_ptr，指向输出数组 dst 中的特定位置

      // 遍历输出张量的深度维度 Y_D
      for (const auto yd : c10::irange(Y_D)) {
        // 计算当前深度维度的偏移量 xd
        const int64_t xd = yd * stride_d - pad_d + kd;
        // 如果 xd 不在合法范围内，将 dst_ptr 中对应位置置零并继续下一次迭代
        if (!IsAGeZeroAndALtB(xd, X_D)) {
          std::memset(dst_ptr + yd * Y_H * Y_W, 0, Y_H * Y_W * sizeof(T));
          continue;
        }

        // 遍历输出张量的高度维度 Y_H
        for (const auto yh : c10::irange(Y_H)) {
          // 计算当前高度维度的偏移量 xh
          const int64_t xh = yh * stride_h - pad_h + kh;
          // 如果 xh 不在合法范围内，将 dst_ptr 中对应位置置零并继续下一次迭代
          if (!IsAGeZeroAndALtB(xh, X_H)) {
            std::memset(
                dst_ptr + yd * Y_H * Y_W + yh * Y_W, 0, Y_W * sizeof(T));
            continue;
          }

          // 遍历输出张量的宽度维度 Y_W
          for (const auto yw : c10::irange(Y_W)) {
            // 计算当前宽度维度的偏移量 xw
            const int64_t xw = yw * stride_w - pad_w + kw;
            // 将输入张量 src 中的对应位置的值复制到输出张量 dst 的对应位置
            dst_ptr[yd * Y_H * Y_W + yh * Y_W + yw] = IsAGeZeroAndALtB(xw, X_W)
                ? src_ptr[xd * X_H * X_W + xh * X_W + xw]
                : T(0);
          }
        }
      }
    }
}

template <typename T>
void Unfold3dZeroPaddingAccKernelImpl(
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    const T* src,
    T* dst) {
  // 计算输入和输出的总大小
  const int64_t X_size = X_D * X_H * X_W;
  const int64_t Y_size = Y_D * Y_H * Y_W;
  const int64_t kernel_size = kernel_d * kernel_h * kernel_w;
  // 并行处理C个通道
  at::parallel_for(0, C, 0, [=](int64_t begin, int64_t end) {
    // 初始化目标区域为0
    std::memset(dst + begin * X_size, 0, (end - begin) * X_size * sizeof(T));
    // 遍历每个通道
    for (const auto c : c10::irange(begin, end)) {
      // 遍历卷积核的深度维度
      for (const auto kd : c10::irange(kernel_d)) {
        // 遍历卷积核的高度维度
        for (const auto kh : c10::irange(kernel_h)) {
          // 遍历卷积核的宽度维度
          for (const auto kw : c10::irange(kernel_w)) {
            // 计算当前卷积核元素在源数据中的偏移量
            const int64_t p =
                c * kernel_size + kd * kernel_h * kernel_w + kh * kernel_w + kw;
            // 遍历输出的深度维度
            for (const auto yd : c10::irange(Y_D)) {
              // 计算输入数据中的深度维度偏移量
              const int64_t xd = yd * stride_d + kd;
              // 获取源数据中的指针
              const T* src_ptr = src + p * Y_size + yd * Y_H * Y_W;
              // 获取目标数据中的指针
              T* dst_ptr = dst + c * X_size + xd * X_H * X_W + kh * X_W + kw;
              // 根据stride_w值选择合适的矩阵加法操作
              if (stride_w == 1) {
                MatAdd<T>(Y_H, Y_W, Y_W, stride_h * X_W, src_ptr, dst_ptr);
              } else {
                MatAdd<T>(
                    Y_H,
                    Y_W,
                    Y_W,
                    1,
                    stride_h * X_W,
                    stride_w,
                    src_ptr,
                    dst_ptr);
              }
            }
          }
        }
      }
    }
  });
}

template <typename T>
void Unfold3dAccKernelImpl(
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    const T* src,
    T* dst) {
  // 如果没有填充，则直接调用无填充版本的实现
  if (pad_d == 0 && pad_h == 0 && pad_w == 0) {
    Unfold3dZeroPaddingAccKernelImpl<T>(
        C,
        X_D,
        X_H,
        X_W,
        Y_D,
        Y_H,
        Y_W,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        src,
        dst);
    return;
  }
  // 计算输入和输出的总大小
  const int64_t X_size = X_D * X_H * X_W;
  const int64_t Y_size = Y_D * Y_H * Y_W;
  const int64_t kernel_size = kernel_d * kernel_h * kernel_w;
  // 并行处理C个通道
  at::parallel_for(0, C, 0, [=](int64_t begin, int64_t end) {
    // 初始化目标区域为0
    std::memset(dst + begin * X_size, 0, (end - begin) * X_size * sizeof(T));
    // 对于指定范围内的每个索引 c，执行以下操作
    for (const auto c : c10::irange(begin, end)) {
      // 计算目标数组的起始地址，偏移量为 c * X_size
      T* dst_ptr = dst + c * X_size;
      // 对于每个核心深度 kd
      for (const auto kd : c10::irange(kernel_d)) {
        // 对于每个核心高度 kh
        for (const auto kh : c10::irange(kernel_h)) {
          // 对于每个核心宽度 kw
          for (const auto kw : c10::irange(kernel_w)) {
            // 计算源数据的索引 p，基于 c, kd, kh, kw 和 kernel_size
            const int64_t p =
                c * kernel_size + kd * kernel_h * kernel_w + kh * kernel_w + kw;
            // 计算源数据的起始地址，偏移量为 p * Y_size
            const T* src_ptr = src + p * Y_size;
            // 对于每个输出深度 yd
            for (const auto yd : c10::irange(Y_D)) {
              // 计算在深度方向的输出坐标 xd
              const int64_t xd = yd * stride_d - pad_d + kd;
              // 如果 xd 不在合法范围内，跳过当前循环
              if (!IsAGeZeroAndALtB(xd, X_D)) {
                continue;
              }
              // 对于每个输出高度 yh
              for (const auto yh : c10::irange(Y_H)) {
                // 计算在高度方向的输出坐标 xh
                const int64_t xh = yh * stride_h - pad_h + kh;
                // 如果 xh 不在合法范围内，跳过当前循环
                if (!IsAGeZeroAndALtB(xh, X_H)) {
                  continue;
                }
                // 对于每个输出宽度 yw
                for (const auto yw : c10::irange(Y_W)) {
                  // 计算在宽度方向的输出坐标 xw
                  const int64_t xw = yw * stride_w - pad_w + kw;
                  // 如果 xw 在合法范围内
                  if (IsAGeZeroAndALtB(xw, X_W)) {
                    // 将 src_ptr 中的数据加到 dst_ptr 对应位置上
                    dst_ptr[xd * X_H * X_W + xh * X_W + xw] +=
                        src_ptr[yd * Y_H * Y_W + yh * Y_W + yw];
                  }
                }
              }
            }
          }
        }
      }
    }
  });
} // namespace

// 定义函数 Unfold3dCopyCPU，用于在 CPU 上执行三维展开复制操作
void Unfold3dCopyCPU(
    ScalarType dtype,             // 数据类型
    const void *src,              // 源数据指针
    int64_t C,                    // 通道数
    int64_t X_D,                  // 输入数据的深度维度
    int64_t X_H,                  // 输入数据的高度维度
    int64_t X_W,                  // 输入数据的宽度维度
    int64_t Y_D,                  // 输出数据的深度维度
    int64_t Y_H,                  // 输出数据的高度维度
    int64_t Y_W,                  // 输出数据的宽度维度
    int64_t kernel_d,             // 深度方向的核大小
    int64_t kernel_h,             // 高度方向的核大小
    int64_t kernel_w,             // 宽度方向的核大小
    int64_t stride_d,             // 深度方向的步幅
    int64_t stride_h,             // 高度方向的步幅
    int64_t stride_w,             // 宽度方向的步幅
    int64_t pad_d,                // 深度方向的填充
    int64_t pad_h,                // 高度方向的填充
    int64_t pad_w,                // 宽度方向的填充
    void* dst) {                  // 目标数据指针
  // 使用 AT_DISPATCH_ALL_TYPES_AND2 宏分发到指定的数据类型，调用 Unfold3dCopyKernelImpl 函数实现具体操作
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      dtype,
      "Unfold3dCopyCPU",
      [=, &src]() {
        Unfold3dCopyKernelImpl<scalar_t>( // 调用模板函数 Unfold3dCopyKernelImpl，处理具体数据类型 scalar_t
            C,
            X_D,
            X_H,
            X_W,
            Y_D,
            Y_H,
            Y_W,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            static_cast<const scalar_t*>(src), // 将 src 强制转换为 scalar_t 类型的指针
            static_cast<scalar_t*>(dst));      // 将 dst 强制转换为 scalar_t 类型的指针
      });
}

// 定义函数 Unfold3dAccCPU，用于在 CPU 上执行三维展开累加操作
void Unfold3dAccCPU(
    ScalarType dtype,             // 数据类型
    const void *src,              // 源数据指针
    int64_t C,                    // 通道数
    int64_t X_D,                  // 输入数据的深度维度
    int64_t X_H,                  // 输入数据的高度维度
    int64_t X_W,                  // 输入数据的宽度维度
    int64_t Y_D,                  // 输出数据的深度维度
    int64_t Y_H,                  // 输出数据的高度维度
    int64_t Y_W,                  // 输出数据的宽度维度
    int64_t kernel_d,             // 深度方向的核大小
    int64_t kernel_h,             // 高度方向的核大小
    int64_t kernel_w,             // 宽度方向的核大小
    int64_t stride_d,             // 深度方向的步幅
    int64_t stride_h,             // 高度方向的步幅
    int64_t stride_w,             // 宽度方向的步幅
    int64_t pad_d,                // 深度方向的填充
    int64_t pad_h,                // 高度方向的填充
    int64_t pad_w,                // 宽度方向的填充
    void* dst) {                  // 目标数据指针
  // 使用 AT_DISPATCH_ALL_TYPES_AND2 宏分发到指定的数据类型，调用 Unfold3dAccKernelImpl 函数实现具体操作
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      dtype,
      "Unfold3dAccCPU",
      [=, &src]() {
        Unfold3dAccKernelImpl<scalar_t>( // 调用模板函数 Unfold3dAccKernelImpl，处理具体数据类型 scalar_t
            C,
            X_D,
            X_H,
            X_W,
            Y_D,
            Y_H,
            Y_W,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            static_cast<const scalar_t*>(src), // 将 src 强制转换为 scalar_t 类型的指针
            static_cast<scalar_t*>(dst));      // 将 dst 强制转换为 scalar_t 类型的指针
      });
}

} // namespace at::native
```