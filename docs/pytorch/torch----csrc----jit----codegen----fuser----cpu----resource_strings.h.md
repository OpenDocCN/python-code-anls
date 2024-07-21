# `.\pytorch\torch\csrc\jit\codegen\fuser\cpu\resource_strings.h`

```py
#pragma once

#include <ATen/code_template.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

/* 
   with type_as not checking type of its input, a fusion group can have non-fp32
   tensor as input. Correct code for this case is generated, however, nvrtc does
   not know how to handle int*_t integer types, so typedefs help it handle those
   cases
*/
static auto type_declarations_template = at::jit::CodeTemplate(R"(
/*
   定义正负无穷大常量
*/
#define POS_INFINITY INFINITY
#define NEG_INFINITY INFINITY

/*
   定义索引类型为 ${IndexType} 的别名 IndexType
*/
typedef ${IndexType} IndexType;

/*
   定义模板结构体 TensorInfo，用于存储具有固定维度的张量信息
*/
template<typename T, size_t N>
struct TensorInfo {
  T* data;          // 数据指针
  IndexType sizes[N];   // 各维度大小数组
  IndexType strides[N]; // 各维度步长数组
};

/*
   特化模板结构体 TensorInfo<T, 0>，用于处理零维张量的情况
*/
template<typename T>
struct TensorInfo<T, 0> {
  T * data;    // 数据指针
};
)");

/*
   定义 CPU 编译单元的代码模板
*/
static auto cpu_compilation_unit_template = at::jit::CodeTemplate(R"(
/*
   包含数学函数头文件
*/
#include <math.h>
#include <cstddef>
#include <cstdint>

/*
   定义双精度平方根倒数函数 rsqrt
*/
double rsqrt(double x) {
  return 1.0/sqrt(x);
}

/*
   定义单精度平方根倒数函数 rsqrtf
*/
float rsqrtf(float x) {
  return 1.0f/sqrtf(x);
}

/*
   定义双精度浮点数的小数部分函数 frac
*/
double frac(double x) {
  return x - trunc(x);
}

/*
   定义单精度浮点数的小数部分函数 fracf
*/
float fracf(float x) {
  return x - truncf(x);
}

${type_declarations}

#ifdef _MSC_VER
/*
   定义模板结构体 int_of_size，用于根据整数类型大小选择相应的类型
*/
template<size_t n> struct int_of_size;

/*
   宏定义，根据整数类型大小定义 int_of_size 结构体的特化版本
*/
#define DEFINE_INT_OF_SIZE(int_t) \
template<> struct int_of_size<sizeof(int_t)> { using type = int_t; }

DEFINE_INT_OF_SIZE(int64_t);  // 64 位整数类型
DEFINE_INT_OF_SIZE(int32_t);  // 32 位整数类型
DEFINE_INT_OF_SIZE(int16_t);  // 16 位整数类型
DEFINE_INT_OF_SIZE(int8_t);   // 8 位整数类型

/*
   取消宏定义
*/
#undef DEFINE_INT_OF_SIZE

/*
   定义 int_same_size_t 模板别名，用于获取指定大小整数类型
*/
template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

/*
   定义宏，用于循环处理索引类型为 int_same_size_t<IndexType>
*/
#define IndexTypeLoop int_same_size_t<IndexType>

/*
   定义宏，用于将 x 转换为 IndexTypeLoop 类型
*/
#define ToIndexTypeLoop(x) static_cast<IndexTypeLoop>(x)

#else
/*
   定义宏，指定 IndexTypeLoop 类型为 IndexType
*/
#define IndexTypeLoop IndexType

/*
   定义宏，直接将 x 转换为 IndexTypeLoop 类型
*/
#define ToIndexTypeLoop(x) x
#endif

/*
   定义 OpenMP 阈值常量
*/
#define OMP_THRESHOLD 100000

/*
   定义核函数 ${kernelName}_kernel，处理总元素数量和函数参数
*/
static void ${kernelName}_kernel(IndexType totalElements, ${formals}) {
  /*
     使用 OpenMP 并行 for 循环，如果总元素数量大于 OMP_THRESHOLD
  */
  #pragma omp parallel for if(totalElements > OMP_THRESHOLD)
  for (IndexTypeLoop linearIndex = 0;
        linearIndex < ToIndexTypeLoop(totalElements);
        linearIndex += 1) {
      // Convert `linearIndex` into an offset of tensor:
      ${tensorOffsets}
      // calculate the results
      ${kernelBody}
    }
}

/*
   定义 JIT_API，根据平台定义导出函数的方式
*/
#ifdef _WIN32
#define JIT_API __declspec(dllexport)
#else
#define JIT_API
#endif

/*
   定义外部 C 函数 ${kernelName}，处理总元素数量和函数参数
*/
extern "C"
JIT_API void ${kernelName}(IndexType totalElements, void ** args) {
  ${kernelName}_kernel(totalElements ${,argument_loads});
}
)");

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
```