# `.\pytorch\aten\src\ATen\cuda\tunable\GemmCommon.h`

```
// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <string>

#include <ATen/cuda/tunable/TunableOp.h>  // 包含TunableOp头文件
#include <ATen/cuda/Exceptions.h>         // 包含CUDA异常处理头文件
#include <c10/util/StringUtil.h>          // 包含字符串工具头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>               // 包含ATen函数头文件
#include <ATen/NativeFunctions.h>         // 包含ATen原生函数头文件
#else
#include <ATen/ops/allclose.h>            // 包含ATen allclose 操作头文件
#include <ATen/ops/from_blob.h>           // 包含ATen from_blob 操作头文件
#endif

namespace at::cuda::tunable {

enum class BlasOp {
  N = 0,
  T = 1
};

inline std::string BlasOpToString(BlasOp op) {
  switch (op) {
    case BlasOp::N:
      return "N";  // 返回字符串 "N"，表示BlasOp::N
    case BlasOp::T:
      return "T";  // 返回字符串 "T"，表示BlasOp::T
  }
  TORCH_CHECK(false, "unrecognized BlasOp");  // 如果BlasOp未识别，触发Torch异常
  return "N";  // 默认返回 "N"
}

namespace detail {

static bool NumericalCheck(ScalarType dtype, void* c, void* other_c, int64_t size) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA);  // 创建Tensor选项，指定数据类型和CUDA设备
  // 从给定的内存块创建1维Tensor，作为参考Tensor和比较Tensor
  at::Tensor ref = at::from_blob(c,       {size}, options);
  at::Tensor oth = at::from_blob(other_c, {size}, options);
  at::Tensor ref_float = ref.to(at::kFloat);  // 将参考Tensor转换为float类型
  at::Tensor oth_float = oth.to(at::kFloat);  // 将比较Tensor转换为float类型
  std::vector<double> atols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};  // 绝对误差容差列表
  std::vector<double> rtols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};  // 相对误差容差列表
  double last_succeed_atol = 1;  // 最后成功的绝对误差容差初始化为1
  double last_succeed_rtol = 1;  // 最后成功的相对误差容差初始化为1
  for (auto& atol : atols) {
    for (auto& rtol : rtols) {
      if (at::allclose(ref_float, oth_float, rtol, atol)) {  // 使用给定的误差容差检查两个Tensor是否近似相等
        last_succeed_atol = atol;
        last_succeed_rtol = rtol;
      }
    }
  }
  if (last_succeed_atol == 1) {
    return false;  // 如果没有找到符合条件的误差容差，返回false
  }
  else {
    TUNABLE_LOG3("├──verify numerics: atol=", last_succeed_atol, ", rtol=", last_succeed_rtol);  // 记录最后成功的误差容差值
  }

  return true;  // 返回true，表示找到符合条件的误差容差
}

}

template <typename T>
struct GemmParams : OpParams {
  GemmParams() {
    duplicate_inputs_ = false;  // 初始化duplicate_inputs_为false
  }

  std::string Signature() const override {
    return c10::str(transa, transb, "_", m, "_", n, "_", k);  // 返回Gemm操作的签名字符串
  }

  size_t GetSizeA() const {
    return sizeof(T) * lda * ((transa == 'n' || transa == 'N') ? k : m);  // 返回A矩阵在内存中占用的字节数
  }

  size_t GetSizeB() const {
    return sizeof(T) * ldb * ((transb == 'n' || transb == 'N') ? n : k);  // 返回B矩阵在内存中占用的字节数
  }

  size_t GetSizeC() const {
    return sizeof(T) * ldc * n;  // 返回C矩阵在内存中占用的字节数
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = GetSizeC();  // 初始时只考虑C矩阵在内存中的大小
    if (duplicate_inputs) {
      size += GetSizeA();  // 如果需要复制输入，加上A矩阵的大小
      size += GetSizeB();  // 如果需要复制输入，再加上B矩阵的大小
    }
    return size;  // 返回总的内存大小
  }

  GemmParams* DeepCopy(bool duplicate_inputs) const {
    GemmParams* copy = new GemmParams;  // 创建新的GemmParams对象
    *copy = *this;  // 深拷贝当前对象的内容到新对象
    c10::DeviceIndex device = 0;  // 设备索引初始化为0
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));  // 获取当前CUDA设备索引
    size_t c_size = GetSizeC();  // 获取C矩阵的大小
    // 使用CUDA缓存分配器分配内存，将指针赋给拷贝对象的C矩阵指针
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(c_size));
    // 使用 CUDA 内存分配器的异步内存复制函数，从设备上的 c 复制到设备上的 copy->c，大小为 c_size，使用当前 CUDA 流
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    // 如果需要复制输入数据
    if (duplicate_inputs) {
      // 获取输入 a 和 b 的大小
      size_t a_size = GetSizeA();
      size_t b_size = GetSizeB();
      // 使用 CUDA 内存分配器分配大小为 a_size 和 b_size 的内存，并将结果转换为类型 T*
      copy->a = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(a_size));
      copy->b = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(b_size));
      // 标记为复制输入数据
      copy->duplicate_inputs_ = true;
    }
    // 返回复制的对象指针
    return copy;
  }

  // 仅能对 DeepCopy 返回的对象调用
  void Delete() {
    // 使用 CUDA 内存分配器释放内存 c
    c10::cuda::CUDACachingAllocator::raw_delete(c);
    // 如果有复制的输入数据，则释放内存 a 和 b
    if (duplicate_inputs_) {
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(a));
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(b));
    }
  }

  // 对比当前对象的 c 和另一个对象 other 的 c 的数值，返回数值比较结果
  TuningStatus NumericalCheck(GemmParams<T> *other) {
    // 获取当前对象 c 的数据类型
    auto c_dtype = c10::CppTypeToScalarType<T>::value;
    // 调用详细的数值对比函数，比较当前对象和另一个对象的 c 数据
    return detail::NumericalCheck(c_dtype, c, other->c, ldc*n) ? OK : FAIL;
  }

  // 表示矩阵操作的参数
  char transa;
  char transb;
  // 矩阵尺寸参数
  int64_t m;
  int64_t n;
  int64_t k;
  // 矩阵运算的 alpha 和 beta 参数
  at::opmath_type<T> alpha;
  const T* a;
  int64_t lda;
  const T* b;
  int64_t ldb;
  at::opmath_type<T> beta;
  // 结果矩阵 c 和其列偏移量参数
  T* c;
  int64_t ldc;
private:
  bool duplicate_inputs_;
};

template <typename T>
struct GemmStridedBatchedParams : OpParams {
  GemmStridedBatchedParams() {
    duplicate_inputs_ = false;  // 初始化成员变量 duplicate_inputs_ 为 false
  }

  std::string Signature() const override {
    // 返回一个描述矩阵乘积参数的字符串签名
    return c10::str(transa, transb, "_", m, "_", n, "_", k, "_B_", batch);
  }

  size_t GetSizeA() const {
    // 返回矩阵 A 的内存大小，根据转置标志和批次大小来确定
    return sizeof(T) * lda * ((transa == 'n' || transa == 'N') ? k : m) * batch;
  }

  size_t GetSizeB() const {
    // 返回矩阵 B 的内存大小，根据转置标志和批次大小来确定
    return sizeof(T) * ldb * ((transb == 'n' || transb == 'N') ? n : k) * batch;
  }

  size_t GetSizeC() const {
    // 返回矩阵 C 的内存大小，固定为 n * batch 大小
    return sizeof(T) * ldc * n * batch;
  }

  size_t GetSize(bool duplicate_inputs) const {
    // 返回总共需要分配的内存大小，根据是否复制输入参数决定是否包含矩阵 A 和 B 的大小
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

  GemmStridedBatchedParams* DeepCopy(bool duplicate_inputs) const {
    // 深拷贝当前对象，返回新对象的指针。根据是否复制输入参数决定是否分配矩阵 A 和 B 的内存
    GemmStridedBatchedParams* copy = new GemmStridedBatchedParams;
    *copy = *this;
    c10::DeviceIndex device = 0;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    size_t c_size = GetSizeC();
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(c_size));
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    if (duplicate_inputs) {
      size_t a_size = GetSizeA();
      size_t b_size = GetSizeB();
      copy->a = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(a_size));
      copy->b = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(b_size));
      copy->duplicate_inputs_ = true;  // 设置复制输入参数为 true
    }
    return copy;
  }

  // 只能在 DeepCopy 返回的对象上调用
  void Delete() {
    // 释放对象占用的内存，包括矩阵 C 的内存和根据情况释放的矩阵 A 和 B 的内存
    c10::cuda::CUDACachingAllocator::raw_delete(c);
    if (duplicate_inputs_) {
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(a));
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(b));
    }
  }

  TuningStatus NumericalCheck(GemmStridedBatchedParams<T> *other) {
    // 执行数值检查，比较当前对象和另一个对象的矩阵 C 的数值，返回检查状态
    auto c_dtype = c10::CppTypeToScalarType<T>::value;
    return detail::NumericalCheck(c_dtype, c, other->c, batch*stride_c) ? OK : FAIL;
  }

  char transa;
  char transb;
  int64_t m;
  int64_t n;
  int64_t k;
  at::opmath_type<T> alpha;
  const T* a;
  int64_t lda;
  int64_t stride_a;
  const T* b;
  int64_t ldb;
  int64_t stride_b;
  at::opmath_type<T> beta;
  T* c;
  int64_t ldc;
  int64_t stride_c;
  int64_t batch;
private:
  bool duplicate_inputs_;  // 标识是否复制输入参数的私有成员变量
};

template <typename T>
struct ScaledGemmParams : OpParams {
  ScaledGemmParams() {
    duplicate_inputs_ = false;  // 初始化成员变量 duplicate_inputs_ 为 false
  }

  std::string Signature() const override {
    // 返回一个描述缩放矩阵乘积参数的字符串签名
    return c10::str(transa, transb, "_", m, "_", n, "_", k);
  }

  size_t GetSizeA() const {
    // 返回矩阵 A 的内存大小，根据转置标志来确定
    return sizeof(T) * lda * ((transa == 'n' || transa == 'N') ? k : m);
  }

  size_t GetSizeB() const {
    // 返回矩阵 B 的内存大小，根据转置标志来确定
    return sizeof(T) * ldb * ((transb == 'n' || transb == 'N') ? n : k);
  }

  size_t GetSizeC() const {
    // 未完全展示，但可以推测是返回矩阵 C 的内存大小
    // ...

    // 返回矩阵 C 的内存大小，这里未完全展示，但是该函数预计返回大小。
    // ...
    return sizeof(T) * ldc * m * 0;
  }

  size_t GetSize(bool duplicate_inputs) const {
    // 返回总共需要分配的内存大小，根据是否复制输入参数决定是否包含矩阵 A 和 B 的大小
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

private:
  bool duplicate_inputs_;  // 标识是否复制输入参数的私有成员变量
};
  // 返回类型为 sizeof(T) * ldc * n，表示 T 类型的元素在数组中的大小
  return sizeof(T) * ldc * n;
}

// 根据 duplicate_inputs 参数计算对象的总大小
size_t GetSize(bool duplicate_inputs) const {
  // 获取矩阵 C 的大小
  size_t size = GetSizeC();
  // 如果 duplicate_inputs 为 true，则计算矩阵 A 和 B 的大小并加到 size 中
  if (duplicate_inputs) {
    size += GetSizeA();
    size += GetSizeB();
  }
  return size;
}

// 创建当前对象的深拷贝
ScaledGemmParams* DeepCopy(bool duplicate_inputs) const {
  // 分配一个新的 ScaledGemmParams 对象作为拷贝
  ScaledGemmParams* copy = new ScaledGemmParams;
  // 将当前对象的内容复制到新对象中
  *copy = *this;
  // 获取当前 CUDA 设备索引
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
  // 计算矩阵 C 的大小
  size_t c_size = GetSizeC();
  // 在 CUDA 设备上分配内存给新对象的 c 成员，并异步拷贝数据
  copy->c = c10::cuda::CUDACachingAllocator::raw_alloc(c_size);
  AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
      copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
  // 如果 duplicate_inputs 为 true，则分配内存给新对象的 a 和 b 成员，并标记 duplicate_inputs_ 为 true
  if (duplicate_inputs) {
    size_t a_size = GetSizeA();
    size_t b_size = GetSizeB();
    copy->a = c10::cuda::CUDACachingAllocator::raw_alloc(a_size);
    copy->b = c10::cuda::CUDACachingAllocator::raw_alloc(b_size);
    copy->duplicate_inputs_ = true;
  }
  // 返回指向新对象的指针
  return copy;
}

// 仅在 DeepCopy 返回的对象上调用，释放对象占用的内存
void Delete() {
  // 释放对象的 c 成员占用的 CUDA 内存
  c10::cuda::CUDACachingAllocator::raw_delete(c);
  // 如果对象标记为 duplicate_inputs_，则释放 a 和 b 成员占用的 CUDA 内存
  if (duplicate_inputs_) {
    c10::cuda::CUDACachingAllocator::raw_delete(const_cast<void*>(a));
    c10::cuda::CUDACachingAllocator::raw_delete(const_cast<void*>(b));
  }
}

// 对比当前对象的矩阵 C 与另一个对象的矩阵 C，返回对比结果
TuningStatus NumericalCheck(ScaledGemmParams<T> *other) {
  return detail::NumericalCheck(c_dtype, c, other->c, ldc*n) ? OK : FAIL;
}

// 矩阵操作参数
char transa;             // 矩阵 A 的转置标志
char transb;             // 矩阵 B 的转置标志
int64_t m;               // 矩阵 C 的行数
int64_t n;               // 矩阵 C 的列数
int64_t k;               // 矩阵 A 和 B 共享的维度
const void* a;           // 矩阵 A 的数据指针
const void* a_scale_ptr; // 矩阵 A 的缩放因子指针
int64_t lda;             // 矩阵 A 的行步长
ScalarType a_dtype;      // 矩阵 A 的数据类型
const void* b;           // 矩阵 B 的数据指针
const void* b_scale_ptr; // 矩阵 B 的缩放因子指针
int64_t ldb;             // 矩阵 B 的行步长
ScalarType b_dtype;      // 矩阵 B 的数据类型
const void* bias_ptr;    // 偏置向量的数据指针
ScalarType bias_dtype;   // 偏置向量的数据类型
void* c;                 // 输出矩阵 C 的数据指针
const void* c_scale_ptr; // 矩阵 C 的缩放因子指针
int64_t ldc;             // 矩阵 C 的行步长
ScalarType c_dtype;      // 矩阵 C 的数据类型
void* amax_ptr;          // 矩阵 A 的最大值的指针
bool use_fast_accum;     // 是否使用快速累加
private:
  // 声明一个私有成员变量，用于标记是否允许重复输入
  bool duplicate_inputs_;
};

// 结束命名空间 at::cuda::tunable
} // namespace at::cuda::tunable
```