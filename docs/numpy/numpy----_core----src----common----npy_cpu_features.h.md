# `.\numpy\numpy\_core\src\common\npy_cpu_features.h`

```py
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPU_FEATURES_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CPU_FEATURES_H_

#include <Python.h> // 包含 Python.h 头文件，用于 PyObject
#include "numpy/numpyconfig.h" // 包含 numpyconfig.h 头文件，用于 NPY_VISIBILITY_HIDDEN 宏

#ifdef __cplusplus
extern "C" {
#endif

enum npy_cpu_features
{
    NPY_CPU_FEATURE_NONE = 0, // 定义枚举常量 NPY_CPU_FEATURE_NONE，值为 0，表示无 CPU 特性
    // X86
    NPY_CPU_FEATURE_MMX               = 1, // 定义 MMX 特性的枚举值为 1
    NPY_CPU_FEATURE_SSE               = 2, // 定义 SSE 特性的枚举值为 2
    NPY_CPU_FEATURE_SSE2              = 3, // 定义 SSE2 特性的枚举值为 3
    NPY_CPU_FEATURE_SSE3              = 4, // 定义 SSE3 特性的枚举值为 4
    NPY_CPU_FEATURE_SSSE3             = 5, // 定义 SSSE3 特性的枚举值为 5
    NPY_CPU_FEATURE_SSE41             = 6, // 定义 SSE4.1 特性的枚举值为 6
    NPY_CPU_FEATURE_POPCNT            = 7, // 定义 POPCNT 特性的枚举值为 7
    NPY_CPU_FEATURE_SSE42             = 8, // 定义 SSE4.2 特性的枚举值为 8
    NPY_CPU_FEATURE_AVX               = 9, // 定义 AVX 特性的枚举值为 9
    NPY_CPU_FEATURE_F16C              = 10, // 定义 F16C 特性的枚举值为 10
    NPY_CPU_FEATURE_XOP               = 11, // 定义 XOP 特性的枚举值为 11
    NPY_CPU_FEATURE_FMA4              = 12, // 定义 FMA4 特性的枚举值为 12
    NPY_CPU_FEATURE_FMA3              = 13, // 定义 FMA3 特性的枚举值为 13
    NPY_CPU_FEATURE_AVX2              = 14, // 定义 AVX2 特性的枚举值为 14
    NPY_CPU_FEATURE_FMA               = 15, // AVX2 和 FMA3，提供向后兼容性，枚举值为 15

    NPY_CPU_FEATURE_AVX512F           = 30, // 定义 AVX-512F 特性的枚举值为 30
    NPY_CPU_FEATURE_AVX512CD          = 31, // 定义 AVX-512CD 特性的枚举值为 31
    NPY_CPU_FEATURE_AVX512ER          = 32, // 定义 AVX-512ER 特性的枚举值为 32
    NPY_CPU_FEATURE_AVX512PF          = 33, // 定义 AVX-512PF 特性的枚举值为 33
    NPY_CPU_FEATURE_AVX5124FMAPS      = 34, // 定义 AVX-5124FMAPS 特性的枚举值为 34
    NPY_CPU_FEATURE_AVX5124VNNIW      = 35, // 定义 AVX-5124VNNIW 特性的枚举值为 35
    NPY_CPU_FEATURE_AVX512VPOPCNTDQ   = 36, // 定义 AVX-512VPOPCNTDQ 特性的枚举值为 36
    NPY_CPU_FEATURE_AVX512BW          = 37, // 定义 AVX-512BW 特性的枚举值为 37
    NPY_CPU_FEATURE_AVX512DQ          = 38, // 定义 AVX-512DQ 特性的枚举值为 38
    NPY_CPU_FEATURE_AVX512VL          = 39, // 定义 AVX-512VL 特性的枚举值为 39
    NPY_CPU_FEATURE_AVX512IFMA        = 40, // 定义 AVX-512IFMA 特性的枚举值为 40
    NPY_CPU_FEATURE_AVX512VBMI        = 41, // 定义 AVX-512VBMI 特性的枚举值为 41
    NPY_CPU_FEATURE_AVX512VNNI        = 42, // 定义 AVX-512VNNI 特性的枚举值为 42
    NPY_CPU_FEATURE_AVX512VBMI2       = 43, // 定义 AVX-512VBMI2 特性的枚举值为 43
    NPY_CPU_FEATURE_AVX512BITALG      = 44, // 定义 AVX-512BITALG 特性的枚举值为 44
    NPY_CPU_FEATURE_AVX512FP16        = 45, // 定义 AVX-512FP16 特性的枚举值为 45

    // X86 CPU Groups
    // Knights Landing (F,CD,ER,PF)
    NPY_CPU_FEATURE_AVX512_KNL        = 101, // 定义 Knights Landing 特性组的枚举值为 101
    // Knights Mill    (F,CD,ER,PF,4FMAPS,4VNNIW,VPOPCNTDQ)
    NPY_CPU_FEATURE_AVX512_KNM        = 102, // 定义 Knights Mill 特性组的枚举值为 102
    // Skylake-X       (F,CD,BW,DQ,VL)
    NPY_CPU_FEATURE_AVX512_SKX        = 103, // 定义 Skylake-X 特性组的枚举值为 103
    // Cascade Lake    (F,CD,BW,DQ,VL,VNNI)
    NPY_CPU_FEATURE_AVX512_CLX        = 104, // 定义 Cascade Lake 特性组的枚举值为 104
    // Cannon Lake     (F,CD,BW,DQ,VL,IFMA,VBMI)
    NPY_CPU_FEATURE_AVX512_CNL        = 105, // 定义 Cannon Lake 特性组的枚举值为 105
    // Ice Lake        (F,CD,BW,DQ,VL,IFMA,VBMI,VNNI,VBMI2,BITALG,VPOPCNTDQ)
    NPY_CPU_FEATURE_AVX512_ICL        = 106, // 定义 Ice Lake 特性组的枚举值为 106
    // Sapphire Rapids (Ice Lake, AVX-512FP16)
    NPY_CPU_FEATURE_AVX512_SPR        = 107, // 定义 Sapphire Rapids 特性组的枚举值为 107

    // IBM/POWER VSX
    // POWER7
    NPY_CPU_FEATURE_VSX               = 200, // 定义 POWER7 的 VSX 特性的枚举值为 200
    // POWER8
    NPY_CPU_FEATURE_VSX2              = 201, // 定义 POWER8 的 VSX2 特性的枚举值为 201
    // POWER9
    NPY_CPU_FEATURE_VSX3              = 202, // 定义 POWER9 的 VSX3 特性的枚举值为 202
    // POWER10
    NPY_CPU_FEATURE_VSX4              = 203, // 定义 POWER10 的 VSX4 特性的枚举值为 203

    // ARM
    NPY_CPU_FEATURE_NEON              = 300, // 定义 ARM 的 NEON 特性的枚举值为 300
    NPY_CPU_FEATURE_NEON_FP16         = 301, // 定义 ARM 的 NEON FP16 特性的枚举值为 301
    // FMA
    NPY_CPU_FEATURE_NEON_VFPV4        = 302, // 定义 ARM 的 NEON VFPV4 特性的枚举值为 302
    // Advanced SIMD
    NPY_CPU_FEATURE_ASIMD             = 303, // 定义 ARM 的 Advanced SIMD 特性的枚举值为 303
    // ARMv8.2 half-precision
    NPY_CPU_FEATURE_FPHP              = 304, // 定义 ARMv8.2 的 FPHP 特性的枚举值为 304
    // ARMv8.2 half-precision vector arithm
    NPY_CPU_FEATURE_ASIMDHP           = 305,
    // ARMv8.2 dot product
    NPY_CPU_FEATURE_ASIMDDP           = 306,
    // ARMv8.2 single&half-precision multiply
    NPY_CPU_FEATURE_ASIMDFHM          = 307,
    // Scalable Vector Extensions (SVE)
    NPY_CPU_FEATURE_SVE               = 308,

    // IBM/ZARCH
    NPY_CPU_FEATURE_VX                = 350,
    // Vector-Enhancements Facility 1
    NPY_CPU_FEATURE_VXE               = 351,
    // Vector-Enhancements Facility 2
    NPY_CPU_FEATURE_VXE2              = 352,

    // RISC-V
    NPY_CPU_FEATURE_RVV               = 400,

    // 定义的最大 CPU 特性值，用于边界检查或遍历
    NPY_CPU_FEATURE_MAX
};

/*
 * 初始化 CPU 特性
 *
 * 这个函数
 *  - 检测运行时的 CPU 特性
 *  - 检查基准 CPU 特性是否存在
 *  - 使用 'NPY_DISABLE_CPU_FEATURES' 来禁用可调度的特性
 *  - 使用 'NPY_ENABLE_CPU_FEATURES' 来启用可调度的特性
 *
 * 当以下情况发生时会设置 RuntimeError：
 *  - 构建时的 CPU 基准特性在运行时不受支持
 *  - 'NPY_DISABLE_CPU_FEATURES' 尝试禁用一个基准特性
 *  - 同时设置了 'NPY_DISABLE_CPU_FEATURES' 和 'NPY_ENABLE_CPU_FEATURES'
 *  - 'NPY_ENABLE_CPU_FEATURES' 尝试启用一个不被机器或构建支持的特性
 *  - 项目在没有任何特性优化支持的情况下尝试启用特性
 *  
 * 当以下情况发生时会设置 ImportWarning：
 *  - 'NPY_DISABLE_CPU_FEATURES' 尝试禁用一个不被机器或构建支持的特性
 *  - 在项目没有任何特性优化支持的情况下，'NPY_DISABLE_CPU_FEATURES' 或 'NPY_ENABLE_CPU_FEATURES'
 *    尝试禁用/启用一个特性
 * 
 * 成功时返回 0，否则返回 -1
 */
NPY_VISIBILITY_HIDDEN int
npy_cpu_init(void);

/*
 * 如果 CPU 特性不可用，则返回 0
 * 注意：必须先调用 `npy_cpu_init`，否则将始终返回 0
 */
NPY_VISIBILITY_HIDDEN int
npy_cpu_have(int feature_id);

#define NPY_CPU_HAVE(FEATURE_NAME) \
npy_cpu_have(NPY_CPU_FEATURE_##FEATURE_NAME)

/*
 * 返回一个新的字典，包含 CPU 特性名称及其运行时可用性
 * 与 `npy_cpu_have` 类似，必须先调用 `npy_cpu_init`
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_features_dict(void);

/*
 * 返回一个新的 Python 列表，包含根据指定 '--cpu-baseline' 参数值
 * 在编译器和平台支持的最小必需优化集合
 *
 * 此函数主要用于实现 umath 的 '__cpu_baseline__' 属性，
 * 并且项目按照从最低到最高兴趣的顺序对项目进行排序
 *
 * 例如，根据默认的构建配置，并假设编译器支持所有相关优化，则返回的列表应该等效于：
 *
 * 在 x86 上：['SSE', 'SSE2']
 * 在 x64 上：['SSE', 'SSE2', 'SSE3']
 * 在 armhf 上：[]
 * 在 aarch64 上：['NEON', 'NEON_FP16', 'NEON_VPFV4', 'ASIMD']
 * 在 ppc64 上：[]
 * 在 ppc64le 上：['VSX', 'VSX2']
 * 在 s390x 上：[]
 * 在其他架构或如果禁用了优化时：[]
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_baseline_list(void);
/*
 * Return a new a Python list contains the dispatched set of additional optimizations
 * that supported by the compiler and platform according to the specified
 * values to command argument '--cpu-dispatch'.
 *
 * This function is mainly used to implement umath's attribute '__cpu_dispatch__',
 * and the items are sorted from the lowest to highest interest.
 *
 * For example, according to the default build configuration and by assuming the compiler
 * support all the involved optimizations then the returned list should equivalent to:
 *
 * On x86: ['SSE3', 'SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C', 'FMA3', 'AVX2', 'AVX512F', ...]
 * On x64: ['SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C', 'FMA3', 'AVX2', 'AVX512F', ...]
 * On armhf: ['NEON', 'NEON_FP16', 'NEON_VPFV4', 'ASIMD', 'ASIMDHP', 'ASIMDDP', 'ASIMDFHM']
 * On aarch64: ['ASIMDHP', 'ASIMDDP', 'ASIMDFHM']
 * On ppc64:  ['VSX', 'VSX2', 'VSX3', 'VSX4']
 * On ppc64le: ['VSX3', 'VSX4']
 * On s390x: ['VX', 'VXE', VXE2]
 * On any other arch or if the optimization is disabled: []
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_dispatch_list(void);
```