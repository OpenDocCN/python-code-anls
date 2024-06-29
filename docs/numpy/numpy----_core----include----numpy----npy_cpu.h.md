# `.\numpy\numpy\_core\include\numpy\npy_cpu.h`

```
/*
 * This block defines CPU-specific macros based on the detected CPU architecture.
 * The macros determine the target CPU and are used for conditional compilation.
 * The possible values include:
 *     - NPY_CPU_X86
 *     - NPY_CPU_AMD64
 *     - NPY_CPU_PPC
 *     - NPY_CPU_PPC64
 *     - NPY_CPU_PPC64LE
 *     - NPY_CPU_SPARC
 *     - NPY_CPU_S390
 *     - NPY_CPU_IA64
 *     - NPY_CPU_HPPA
 *     - NPY_CPU_ALPHA
 *     - NPY_CPU_ARMEL
 *     - NPY_CPU_ARMEB
 *     - NPY_CPU_SH_LE
 *     - NPY_CPU_SH_BE
 *     - NPY_CPU_ARCEL
 *     - NPY_CPU_ARCEB
 *     - NPY_CPU_RISCV64
 *     - NPY_CPU_LOONGARCH
 *     - NPY_CPU_WASM
 */
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_CPU_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_CPU_H_

#include "numpyconfig.h"

#if defined( __i386__ ) || defined(i386) || defined(_M_IX86)
    /*
     * __i386__ is defined by gcc and Intel compiler on Linux,
     * _M_IX86 by VS compiler,
     * i386 by Sun compilers on opensolaris at least
     */
    #define NPY_CPU_X86
#elif defined(__x86_64__) || defined(__amd64__) || defined(__x86_64) || defined(_M_AMD64)
    /*
     * both __x86_64__ and __amd64__ are defined by gcc
     * __x86_64 defined by sun compiler on opensolaris at least
     * _M_AMD64 defined by MS compiler
     */
    #define NPY_CPU_AMD64
#elif defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
    #define NPY_CPU_PPC64LE
#elif defined(__powerpc64__) && defined(__BIG_ENDIAN__)
    #define NPY_CPU_PPC64
#elif defined(__ppc__) || defined(__powerpc__) || defined(_ARCH_PPC)
    /*
     * __ppc__ is defined by gcc, I remember having seen __powerpc__ once,
     * but can't find it ATM
     * _ARCH_PPC is used by at least gcc on AIX
     * As __powerpc__ and _ARCH_PPC are also defined by PPC64 check
     * for those specifically first before defaulting to ppc
     */
    #define NPY_CPU_PPC
#elif defined(__sparc__) || defined(__sparc)
    /* __sparc__ is defined by gcc and Forte (e.g. Sun) compilers */
    #define NPY_CPU_SPARC
#elif defined(__s390__)
    #define NPY_CPU_S390
#elif defined(__ia64)
    #define NPY_CPU_IA64
#elif defined(__hppa)
    #define NPY_CPU_HPPA
#elif defined(__alpha__)
    #define NPY_CPU_ALPHA
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
    /* _M_ARM64 is defined in MSVC for ARM64 compilation on Windows */
    #if defined(__ARMEB__) || defined(__AARCH64EB__)
        #if defined(__ARM_32BIT_STATE)
            #define NPY_CPU_ARMEB_AARCH32
        #elif defined(__ARM_64BIT_STATE)
            #define NPY_CPU_ARMEB_AARCH64
        #else
            #define NPY_CPU_ARMEB
        #endif
    #elif defined(__ARMEL__) || defined(__AARCH64EL__) || defined(_M_ARM64)
        #if defined(__ARM_32BIT_STATE)
            #define NPY_CPU_ARMEL_AARCH32
        #elif defined(__ARM_64BIT_STATE) || defined(_M_ARM64) || defined(__AARCH64EL__)
            #define NPY_CPU_ARMEL_AARCH64
        #else
            #define NPY_CPU_ARMEL
        #endif
            /*
             * Detect ARM specific configurations:
             * - Check for little-endian (__ARMEL__) or big-endian (__ARMEB__) ARM architectures
             * - Distinguish between ARM32 (__ARM_32BIT_STATE) and ARM64 (__ARM_64BIT_STATE)
             * - Define appropriate macros based on detected conditions
             */
            #define NPY_CPU_ARMEB_AARCH32
        #elif defined(__ARM_64BIT_STATE)
            /*
             * Define ARM64 little-endian specific macros when __ARM_64BIT_STATE is detected:
             * - __ARM_64BIT_STATE is used to distinguish ARM64 architecture
             * - Also include checks for _M_ARM64 and __AARCH64EL__ for MSVC and AARCH64EL compatibility
             */
            #define NPY_CPU_ARMEB_AARCH64
        #else
            /*
             * Default to ARM architecture in little-endian mode when no specific condition is met:
             * - Define NPY_CPU_ARMEL for general ARM little-endian architectures
             */
            #define NPY_CPU_ARMEB
        #endif
    #elif defined(__ARMEL__) || defined(__AARCH64EL__) || defined(_M_ARM64)
        #if defined(__ARM_32BIT_STATE)
            /*
             * Define ARM32 little-endian specific macros when __ARM_32BIT_STATE is detected:
             * - __ARM_32BIT_STATE is used to distinguish ARM32 architecture
             * - Include _M_ARM64 and __AARCH64EL__ checks for MSVC and AARCH64EL compatibility
             */
            #define NPY_CPU_ARMEL_AARCH32
        #elif defined(__ARM_64BIT_STATE) || defined(_M_ARM64) || defined(__AARCH64EL__)
            /*
             * Define ARM64 little-endian specific macros when __ARM_64BIT_STATE or equivalent is detected:
             * - __ARM_64BIT_STATE is used to distinguish ARM64 architecture
             * - Also include _M_ARM64 and __AARCH64EL__ checks for MSVC and AARCH64EL compatibility
             */
            #define NPY_CPU_ARMEL_AARCH64
        #else
            /*
             * Default to ARM architecture in little-endian mode when no specific condition is met:
             * - Define NPY_CPU_ARMEL for general ARM little-endian architectures
             */
            #define NPY_CPU_ARMEL
        #endif

            /*
             * Define ARM architecture-specific macros:
             * - __ARM__ and __aarch64__ are defined by compilers supporting ARM architectures
             * - _M_ARM64 is defined by MSVC for ARM64 compilation on Windows
             * - This section distinguishes between little-endian (__ARMEB__) and big-endian (__AARCH64EB__) modes
             * - Further checks differentiate between ARM32 and ARM64 states (__ARM_32BIT_STATE and __ARM_64BIT_STATE)
             */
            #if defined(__ARMEB__) || defined(__AARCH64EB__)
                #if defined(__ARM_32BIT_STATE)
                    /*
                     * Define ARM32 big-endian specific macros when __ARM_32BIT_STATE is detected:
                     * - __ARM_32BIT_STATE is used to distinguish ARM32 architecture
                     */
                    #define NPY_CPU_ARMEB_AARCH32
                #elif defined(__ARM_64BIT_STATE)
                    /*
                     * Define ARM64 big-endian specific macros when __ARM_64BIT_STATE is detected:
                     * - __ARM_64BIT_STATE is used to distinguish ARM64 architecture
                     */
                    #define NPY_CPU_ARMEB_AARCH64
                #else
                    /*
                     * Default to big-endian ARM architecture when no specific condition is met:
                     * - Define NPY_CPU_ARMEB for general big-endian ARM architectures
                     */
                    #define NPY_CPU_ARMEB
                #endif
            #elif defined(__ARMEL__) || defined(__AARCH64EL__) || defined(_M_ARM64)
                #if defined(__ARM_32BIT_STATE)
                    /*
                     * Define ARM32 little-endian specific macros when __ARM_32BIT_STATE is detected:
                     * - __ARM_32BIT_STATE is used to distinguish ARM32 architecture
                     */
                    #define NPY_CPU_ARMEL_AARCH32
                #elif defined(__ARM_64BIT_STATE) || defined(_M_ARM64) || defined(__AARCH64EL__)
                    /*
                     * Define ARM64 little-endian specific macros when __ARM_64BIT_STATE or equivalent is detected:
                     * - __ARM_64BIT_STATE is used to distinguish ARM64 architecture
                     * - Also include _M_ARM64 and __AARCH64EL__ checks for MSVC and AARCH64EL compatibility
                     */
                    #define NPY_CPU_ARMEL_AARCH64
                #else
                    /*
                     * Default to little-endian ARM architecture when no specific condition is met:
                     * - Define NPY_CPU_ARMEL for general little-endian ARM architectures
                     */
                    #define NPY_CPU_ARMEL
                #endif
    # 如果条件不满足，则执行以下代码
    # 向用户报告错误：未知的 ARM CPU，请提供平台信息（操作系统、CPU和编译器）给 numpy 维护人员
    # 结束条件判断
#elif defined(__sh__) && defined(__LITTLE_ENDIAN__)
    #define NPY_CPU_SH_LE
#elif defined(__sh__) && defined(__BIG_ENDIAN__)
    #define NPY_CPU_SH_BE
#elif defined(__MIPSEL__)
    #define NPY_CPU_MIPSEL
#elif defined(__MIPSEB__)
    #define NPY_CPU_MIPSEB
#elif defined(__or1k__)
    #define NPY_CPU_OR1K
#elif defined(__mc68000__)
    #define NPY_CPU_M68K
#elif defined(__arc__) && defined(__LITTLE_ENDIAN__)
    #define NPY_CPU_ARCEL
#elif defined(__arc__) && defined(__BIG_ENDIAN__)
    #define NPY_CPU_ARCEB
#elif defined(__riscv) && defined(__riscv_xlen) && __riscv_xlen == 64
    #define NPY_CPU_RISCV64
#elif defined(__loongarch__)
    #define NPY_CPU_LOONGARCH
#elif defined(__EMSCRIPTEN__)
    /* __EMSCRIPTEN__ is defined by emscripten: an LLVM-to-Web compiler */
    #define NPY_CPU_WASM
#else
    #error Unknown CPU, please report this to numpy maintainers with \
    information about your platform (OS, CPU and compiler)
#endif

/*
 * Except for the following architectures, memory access is limited to the natural
 * alignment of data types otherwise it may lead to bus error or performance regression.
 * For more details about unaligned access, see https://www.kernel.org/doc/Documentation/unaligned-memory-access.txt.
*/
#if defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64) || defined(__aarch64__) || defined(__powerpc64__)
    #define NPY_ALIGNMENT_REQUIRED 0
#endif
#ifndef NPY_ALIGNMENT_REQUIRED
    #define NPY_ALIGNMENT_REQUIRED 1
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_CPU_H_ */



#elif defined(__sh__) && defined(__LITTLE_ENDIAN__)
    #define NPY_CPU_SH_LE  // Define for SH architecture with little endian byte order
#elif defined(__sh__) && defined(__BIG_ENDIAN__)
    #define NPY_CPU_SH_BE  // Define for SH architecture with big endian byte order
#elif defined(__MIPSEL__)
    #define NPY_CPU_MIPSEL  // Define for MIPS architecture with little endian byte order
#elif defined(__MIPSEB__)
    #define NPY_CPU_MIPSEB  // Define for MIPS architecture with big endian byte order
#elif defined(__or1k__)
    #define NPY_CPU_OR1K  // Define for OpenRISC architecture
#elif defined(__mc68000__)
    #define NPY_CPU_M68K  // Define for Motorola 68000 architecture
#elif defined(__arc__) && defined(__LITTLE_ENDIAN__)
    #define NPY_CPU_ARCEL  // Define for ARC architecture with little endian byte order
#elif defined(__arc__) && defined(__BIG_ENDIAN__)
    #define NPY_CPU_ARCEB  // Define for ARC architecture with big endian byte order
#elif defined(__riscv) && defined(__riscv_xlen) && __riscv_xlen == 64
    #define NPY_CPU_RISCV64  // Define for RISC-V 64-bit architecture
#elif defined(__loongarch__)
    #define NPY_CPU_LOONGARCH  // Define for LoongArch architecture
#elif defined(__EMSCRIPTEN__)
    /* __EMSCRIPTEN__ is defined by emscripten: an LLVM-to-Web compiler */
    #define NPY_CPU_WASM  // Define for WebAssembly (emscripten) platform
#else
    #error Unknown CPU, please report this to numpy maintainers with \
    information about your platform (OS, CPU and compiler)
#endif

/*
 * Except for the following architectures, memory access is limited to the natural
 * alignment of data types otherwise it may lead to bus error or performance regression.
 * For more details about unaligned access, see https://www.kernel.org/doc/Documentation/unaligned-memory-access.txt.
*/
#if defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64) || defined(__aarch64__) || defined(__powerpc64__)
    #define NPY_ALIGNMENT_REQUIRED 0  // Disable alignment requirement for these architectures
#endif
#ifndef NPY_ALIGNMENT_REQUIRED
    #define NPY_ALIGNMENT_REQUIRED 1  // Enable alignment requirement by default
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_CPU_H_ */
```