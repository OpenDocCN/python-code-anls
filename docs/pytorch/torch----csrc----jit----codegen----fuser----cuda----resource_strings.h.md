# `.\pytorch\torch\csrc\jit\codegen\fuser\cuda\resource_strings.h`

```py
// 预处理指令，确保头文件只被包含一次
#pragma once

// 包含 ATen 库中的代码模板头文件
#include <ATen/code_template.h>
// 包含 Torch 库中的导出头文件
#include <torch/csrc/Export.h>

// Torch 的命名空间：jit -> fuser -> cuda
namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/* 
   当 type_as 不检查其输入类型时，融合组可以有非 fp32 张量作为输入。
   生成正确的代码，但 nvrtc 无法处理 int*_t 整数类型，因此使用 typedef 帮助处理这些情况。
*/
#if defined(USE_ROCM)
// ROCm 平台下的类型声明模板字符串
static auto type_declarations_template = at::jit::CodeTemplate(R"(
${HalfHeader}
${BFloat16Header}
${RandHeader}

// 定义特定浮点数常量
#define NAN __int_as_float(0x7fffffff)
#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)

// 定义 IndexType 类型
typedef ${IndexType} IndexType;

// 模板：TensorInfo，存储张量信息
template<typename T, size_t N>
struct TensorInfo {
  T* data;              // 数据指针
  IndexType sizes[N];   // 大小数组
  IndexType strides[N]; // 步长数组
};

// 特化模板：TensorInfo<T, 0>
template<typename T>
struct TensorInfo<T, 0> {
  T * data;  // 数据指针（零维张量）
};
)")
#else
// 非 ROCm 平台下的类型声明模板字符串，包含各种整数类型的 typedef
static auto type_declarations_template = at::jit::CodeTemplate(R"(
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int  int16_t;
typedef long long int int64_t;
typedef unsigned long long int uint64_t;
${HalfHeader}
${BFloat16Header}
${RandHeader}

// 定义特定浮点数常量
#define NAN __int_as_float(0x7fffffff)
#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)

// 定义 IndexType 类型
typedef ${IndexType} IndexType;

// 模板：TensorInfo，存储张量信息
template<typename T, size_t N>
struct TensorInfo {
  T* data;              // 数据指针
  IndexType sizes[N];   // 大小数组
  IndexType strides[N]; // 步长数组
};

// 特化模板：TensorInfo<T, 0>
template<typename T>
struct TensorInfo<T, 0> {
  T * data;  // 数据指针（零维张量）
};
)")
#endif

// 用于处理 Philox 随机数生成器的代码重写，因为 nvrtc 无法正确解析 curand 头文件
constexpr auto rand_support_literal = R"(

  class Philox {
  public:
    // 构造函数，初始化 Philox 随机数生成器
    __device__ inline Philox(unsigned long long seed,
                             unsigned long long subsequence,
                             unsigned long long offset) {
      key.x = (unsigned int)seed;
      key.y = (unsigned int)(seed >> 32);
      counter = make_uint4(0, 0, 0, 0);
      counter.z = (unsigned int)(subsequence);
      counter.w = (unsigned int)(subsequence >> 32);
      STATE = 0;
      incr_n(offset / 4);
    }

    // 重载操作符，生成随机数
    __device__ inline unsigned long operator()() {
      if(STATE == 0) {
        uint4 counter_ = counter;
        uint2 key_ = key;
        for(int i = 0; i < 9; i++) {
          counter_ = single_round(counter_, key_);
          key_.x += (kPhilox10A); key_.y += (kPhilox10B);
        }
        output = single_round(counter_, key_);
        incr();
      }
      unsigned long ret;
      switch(STATE) {
        case 0: ret = output.x; break;
        case 1: ret = output.y; break;
        case 2: ret = output.z; break;
        case 3: ret = output.w; break;
      }
      STATE = (STATE + 1) % 4;
      return ret;
    }

  private:
    uint4 counter;   // 计数器
    uint4 output;    // 输出
    uint2 key;       // 密钥
    unsigned int STATE;  // 状态
    # 定义一个内联函数，用于增加一个64位无符号整数到计数器中
    __device__ inline void incr_n(unsigned long long n) {
      # 将64位整数拆分为低32位和高32位
      unsigned int nlo = (unsigned int)(n);
      unsigned int nhi = (unsigned int)(n >> 32);
      # 将低32位加到计数器的x分量
      counter.x += nlo;
      # 如果x分量溢出，增加高32位
      if (counter.x < nlo)
        nhi++;
      # 将高32位加到计数器的y分量
      counter.y += nhi;
      # 如果y分量未溢出，返回
      if (nhi <= counter.y)
        return;
      # 如果z分量未溢出，返回
      if (++counter.z)
        return;
      # 否则增加w分量
      ++counter.w;
    }
    
    # 定义一个内联函数，用于增加计数器的值
    __device__ inline void incr() {
      # 如果x分量未溢出，返回
      if (++counter.x)
        return;
      # 如果y分量未溢出，返回
      if (++counter.y)
        return;
      # 如果z分量未溢出，返回
      if (++counter.z)
        return;
      # 否则增加w分量
      ++counter.w;
    }
    
    # 定义一个设备函数，用于计算32位整数的乘积和高32位乘积
    __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                      unsigned int *result_high) {
      # 计算a和b的高32位乘积，将结果存储到result_high中
      *result_high = __umulhi(a, b);
      # 返回a和b的32位乘积
      return a * b;
    }
    
    # 定义一个内联函数，实现Philox随机数生成算法的单轮运算
    __device__ inline uint4 single_round(uint4 ctr, uint2 key) {
      # 定义变量存储两次乘法的高32位
      unsigned int hi0;
      unsigned int hi1;
      # 计算ctr.x与kPhiloxSA的高32位乘积，并存储结果到hi0中
      unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
      # 计算ctr.z与kPhiloxSB的高32位乘积，并存储结果到hi1中
      unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    
      # 构造返回的uint4结构体，其中四个元素分别由上述计算得到
      uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
      # 返回结果
      return ret;
    }
    
    # 定义常量，用于Philox随机数生成算法中的常量初始化
    static const unsigned long kPhilox10A = 0x9E3779B9;
    static const unsigned long kPhilox10B = 0xBB67AE85;
    static const unsigned long kPhiloxSA = 0xD2511F53;
    static const unsigned long kPhiloxSB = 0xCD9E8D57;
    
    // 2^32的倒数
    #define M_RAN_INVM32 2.3283064e-10f
    
    # 定义一个内联设备函数，将输入的32位无符号整数映射到一个范围在0到1之间的浮点数
    __device__ __inline__ float uniform(unsigned int x) {
      # 返回输入整数乘以2^(-32)的结果
      return x * M_RAN_INVM32;
    }
// 定义了一个常量字符串，用于声明 CUDA 内核函数的参数列表
constexpr auto rand_param =
    ",unsigned long long seed, unsigned long long offset";

// 定义了一个常量字符串，初始化 CUDA 内核函数中的随机数生成器
constexpr auto rand_init = R"(
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Philox rnd(seed, idx, offset);
)";

// 定义了一个模板，用于生成 CUDA 编译单元的代码
static auto cuda_compilation_unit_template = at::jit::CodeTemplate(R"(
${type_declarations}

extern "C" __global__
void ${kernelName}(IndexType totalElements, ${formals} ${RandParam}) {
  ${RandInit}
  // 检查是否进行向量化加载/存储并分配缓冲区
  bool flag_vec4 = true;
  ${tensorChecks}
  if (flag_vec4) {
    for (IndexType linearIndex = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
         linearIndex < totalElements;
         linearIndex += 4 * gridDim.x * blockDim.x) {
      // 将 `linearIndex` 转换为张量的偏移量:
      ${tensorOffsets}
      // 一次加载4个元素
      ${kernelLoad}
      #pragma unroll 4
      for (int i=0; i<4; i++) {
        // 计算结果
        ${kernelBody_vec4}
      }
      // 一次存储4个元素
      ${kernelStore}
    }
  } else {
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < totalElements;
         linearIndex += gridDim.x * blockDim.x) {
      // 将 `linearIndex` 转换为张量的偏移量:
      ${tensorOffsets}
      // 计算结果
      ${kernelBody}
    }
  }
}
)");

// 此段代码启用了 jit 中的半精度支持。按照规范，对于归约操作，fp16 输入数据会被立即转换为 float，
// 使用 __half2float() 函数。所有的数学操作都在 float 值上进行，如果需要，中间的 float 表示会
// 在写入到半精度张量时用 __float2half() 转换为半精度。
#if defined(USE_ROCM)
// ROCm 平台下的半精度支持类型为 __half
constexpr auto half_support_literal =
    R"(
typedef __half half;
)";
#else
// CUDA 平台下的半精度支持定义
constexpr auto half_support_literal =
    R"(
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#if defined(__cplusplus)
  struct __align__(2) __half {
    __host__ __device__ __half() { }

  protected:
    unsigned short __x;
  };

  /* All intrinsic functions are only available to nvcc compilers */
  #if defined(__CUDACC__)
    /* Definitions of intrinsics */
    __device__ __half __float2half(const float f) {
      __half val;
      asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
      return val;
    }

    __device__ float __half2float(const __half h) {
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
      return val;
    }
    // 使用字符串粘贴来将 " 和 #endif 分隔开，以解决这个问题
    R"(
    #endif /* defined(__CUDACC__) */
#if defined(__cplusplus)
// 如果正在使用 C++ 编译环境，则定义以下内容

// 将宏 __BFLOAT16_TO_US 定义为取变量的无符号短整型指针，并解引用
#define __BFLOAT16_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
// 将宏 __BFLOAT16_TO_CUS 定义为取变量的常量无符号短整型指针，并解引用
#define __BFLOAT16_TO_CUS(var) \
  *(reinterpret_cast<const unsigned short*>(&(var)))

// 定义一个 2 字节对齐的结构 __nv_bfloat16_raw，包含一个无符号短整型成员 x
typedef struct __align__(2) {
  unsigned short x;
}
__nv_bfloat16_raw;

// 如果定义了 __cplusplus，表示是 C++ 编译环境
#if defined(__cplusplus)
// 如果同时定义了 __CUDACC__，表示是 CUDA 编译环境
#if defined(__CUDACC__)
// 在 CUDA 编译环境中，定义以下内容

// 将浮点数转换为 bfloat16 类型的内部函数声明，返回无符号短整型
__device__ unsigned short __internal_float2bfloat16(
    const float f,           // 输入的浮点数
    unsigned int& sign,      // 返回的符号位
    unsigned int& remainder  // 返回的余数
);

// 定义浮点数转换为 bfloat16 类型的函数声明，返回 __nv_bfloat16 类型
__device__ __nv_bfloat16 __float2bfloat16(const float a);

#endif  // defined(__CUDACC__)

// 如果不是 CUDA 编译环境，则定义以下内容

// 定义一个 2 字节对齐的结构 __nv_bfloat16，包含一个无符号短整型成员 __x
struct __align__(2) __nv_bfloat16 {
  __host__ __device__ __nv_bfloat16() {}  // 默认构造函数

  // 赋值运算符重载，从 __nv_bfloat16_raw 类型赋值给当前对象
  __host__ __device__ __nv_bfloat16& operator=(const __nv_bfloat16_raw& hr) {
    __x = hr.x;
    return *this;
  }

 protected:
  unsigned short __x;  // 保护类型的无符号短整型成员 __x
};

#if defined(__CUDACC__)
// 在 CUDA 编译环境中，定义以下内容

// 浮点数转换为 bfloat16 类型的内部函数实现
__device__ unsigned short __internal_float2bfloat16(
    const float f,           // 输入的浮点数
    unsigned int& sign,      // 返回的符号位
    unsigned int& remainder  // 返回的余数
) {
  unsigned int x;

  x = __float_as_uint(f);

  // 如果浮点数的指数部分超出范围
  if ((x & 0x7fffffffU) > 0x7f800000U) {
    sign = 0U;
    remainder = 0U;
    return static_cast<unsigned short>(0x7fffU);  // 返回 bfloat16 类型的最大值
  }
  sign = x >> 31;          // 符号位
  remainder = x << 16;     // 余数
  return static_cast<unsigned short>(x >> 16);  // 返回 bfloat16 类型的值
}

// 浮点数转换为 bfloat16 类型的函数实现
__device__ __nv_bfloat16 __float2bfloat16(const float a) {
  __nv_bfloat16 val;
#if __CUDA_ARCH__ >= 800
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
#else
  __nv_bfloat16_raw r;
  unsigned int sign;
  unsigned int remainder;
  r.x = __internal_float2bfloat16(a, sign, remainder);
  // 对 bfloat16 进行四舍五入
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
    r.x++;
  }
  val = r;  // 赋值给返回值
#endif
  return val;  // 返回转换后的 bfloat16 值
}

#endif  // defined(__CUDACC__)

// bfloat16 的反向转换函数声明，在 C++ 和 CUDA 环境中均可用
__device__ float __bfloat162float(const __nv_bfloat16 a);

#endif  // defined(__cplusplus)
#else
// 如果不是 C++ 编译环境，则定义以下内容

// 如果没有定义 __cplusplus，则定义以下内容

// 定义一个 2 字节对齐的结构 __nv_bfloat16_raw，包含一个无符号短整型成员 x
typedef struct __align__(2) {
  unsigned short x;
}
__nv_bfloat16_raw;

// 如果定义了 __cplusplus，表示是 C++ 编译环境
#if defined(__cplusplus)
// 如果同时定义了 __CUDACC__，表示是 CUDA 编译环境
#if defined(__CUDACC__)
// 在 CUDA 编译环境中，定义以下内容

// 浮点数转换为 bfloat16 类型的内部函数声明
__device__ unsigned short __internal_float2bfloat16(
    const float f,           // 输入的浮点数
    unsigned int& sign,      // 返回的符号位
    unsigned int& remainder  // 返回的余数
);

// 定义浮点数转换为 bfloat16 类型的函数声明
__device__ __nv_bfloat16 __float2bfloat16(const float a);

#endif  // defined(__CUDACC__)

// 如果不是 CUDA 编译环境，则定义以下内容

// 定义一个 2 字节对齐的结构 __nv_bfloat16，包含一个无符号短整型成员 __x
struct __align__(2) __nv_bfloat16 {
  __host__ __device__ __nv_bfloat16() {}  // 默认构造函数

  // 赋值运算符重载，从 __nv_bfloat16_raw 类型赋值给当前对象
  __host__ __device__ __nv_bfloat16& operator=(const __nv_bfloat16_raw& hr) {
    __x = hr.x;
    return *this;
  }

 protected:
  unsigned short __x;  // 保护类型的无符号短整型成员 __x
};

// 如果定义了 __CUDACC__，表示是 CUDA 编译环境
#if defined(__CUDACC__)
// 在 CUDA 编译环境中，定义以下内容

// 浮点数转换为 bfloat16 类型的内部函数实现
__device__ unsigned short __internal_float2bfloat16(
    const float f,           // 输入的浮点数
    unsigned int& sign,      // 返回的符号位
    unsigned int& remainder  // 返回的余数
) {
  unsigned int x;

  x = __float_as_uint(f);

  // 如果浮点数的指数部分超出范围
  if ((x & 0x7fffffffU) > 0x7f800000U) {
    sign = 0U;
    remainder = 0U;
    return static_cast<unsigned short>(0x7fffU);  // 返回 bfloat16 类型的最大值
  }
  sign = x >> 31;          // 符号位
  remainder = x << 16;     // 余数
  return static_cast<unsigned short>(x >> 16);  // 返回 bfloat16 类型的值
}

// 浮点数转换为 bfloat16 类型的函数实现
__device__ __nv_bfloat16 __float2bfloat16(const float a) {
  __nv_bfloat16 val;
  __nv_bfloat16_raw r;
  unsigned int sign;
  unsigned int remainder;
  r.x = __internal_float2bfloat16(a, sign, remainder);
  // 对 bfloat16 进行四舍五入
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
    r.x++;
  }
  val = r;  // 赋值给返回值
#endif

#endif  // defined(__CUDACC__)
#endif  // defined(__cplusplus
#endif
  return val;
}

__device__ float __bfloat162float(const __nv_bfloat16 a) {
  float val;
  // 使用 CUDA 内联汇编指令将 __nv_bfloat16 转换为 float
  asm("{ mov.b32 %0, {0,%1};}\n" : "=f"(val) : "h"(__BFLOAT16_TO_CUS(a)));
  return val;
}
#endif /* defined(__CUDACC__) */
#endif /* defined(__cplusplus__) */

#undef __BFLOAT16_TO_US
#undef __BFLOAT16_TO_CUS

// 结束 CUDA 命名空间
)";
#endif

// 结束 fuser 命名空间
} // namespace fuser

// 结束 jit 命名空间
} // namespace jit

// 结束 torch 命名空间
} // namespace torch
```