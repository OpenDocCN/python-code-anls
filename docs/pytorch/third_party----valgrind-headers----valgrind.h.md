# `.\pytorch\third_party\valgrind-headers\valgrind.h`

```
/*
   -*- c -*-
   ----------------------------------------------------------------

   Notice that the following BSD-style license applies to this one
   file (valgrind.h) only.  The rest of Valgrind is licensed under the
   terms of the GNU General Public License, version 2, unless
   otherwise indicated.  See the COPYING file in the source
   distribution for details.

   ----------------------------------------------------------------

   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2000-2017 Julian Seward.  All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. The origin of this software must not be misrepresented; you must 
      not claim that you wrote the original software.  If you use this 
      software in a product, an acknowledgment in the product 
      documentation would be appreciated but is not required.

   3. Altered source versions must be plainly marked as such, and must
      not be misrepresented as being the original software.

   4. The name of the author may not be used to endorse or promote 
      products derived from this software without specific prior written 
      permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
   OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ----------------------------------------------------------------

   Notice that the above BSD-style license applies to this one file
   (valgrind.h) only.  The entire rest of Valgrind is licensed under
   the terms of the GNU General Public License, version 2.  See the
   COPYING file in the source distribution for details.

   ---------------------------------------------------------------- 
*/
/* This file is for inclusion into client (your!) code.
   
   You can use these macros to manipulate and query Valgrind's 
   execution inside your own programs.
   
   The resulting executables will still run without Valgrind, just a
   little bit more slowly than they otherwise would, but otherwise
   unchanged.  When not running on valgrind, each client request
   consumes very few (eg. 7) instructions, so the resulting performance
   loss is negligible unless you plan to execute client requests
   millions of times per second.  Nevertheless, if that is still a
   problem, you can compile with the NVALGRIND symbol defined (gcc
   -DNVALGRIND) so that client requests are not even compiled in.  */

#ifndef __VALGRIND_H
#define __VALGRIND_H


/* ------------------------------------------------------------------ */
/* VERSION NUMBER OF VALGRIND                                         */
/* ------------------------------------------------------------------ */

/* Specify Valgrind's version number, so that user code can
   conditionally compile based on our version number.  Note that these
   were introduced at version 3.6 and so do not exist in version 3.5
   or earlier.  The recommended way to use them to check for "version
   X.Y or later" is (eg)

#if defined(__VALGRIND_MAJOR__) && defined(__VALGRIND_MINOR__)   \
    && (__VALGRIND_MAJOR__ > 3                                   \
        || (__VALGRIND_MAJOR__ == 3 && __VALGRIND_MINOR__ >= 6))
*/
#define __VALGRIND_MAJOR__    3
#define __VALGRIND_MINOR__    17


#include <stdarg.h>

/* Nb: this file might be included in a file compiled with -ansi.  So
   we can't use C++ style "//" comments nor the "asm" keyword (instead
   use "__asm__"). */

/* Derive some tags indicating what the target platform is.  Note
   that in this file we're using the compiler's CPP symbols for
   identifying architectures, which are different to the ones we use
   within the rest of Valgrind.  Note, __powerpc__ is active for both
   32 and 64-bit PPC, whereas __powerpc64__ is only active for the
   latter (on Linux, that is).

   Misc note: how to find out what's predefined in gcc by default:
   gcc -Wp,-dM somefile.c
*/
#undef PLAT_x86_darwin
#undef PLAT_amd64_darwin
#undef PLAT_x86_win32
#undef PLAT_amd64_win64
#undef PLAT_x86_linux
#undef PLAT_amd64_linux
#undef PLAT_ppc32_linux
#undef PLAT_ppc64be_linux
#undef PLAT_ppc64le_linux
#undef PLAT_arm_linux
#undef PLAT_arm64_linux
#undef PLAT_s390x_linux
#undef PLAT_mips32_linux
#undef PLAT_mips64_linux
#undef PLAT_nanomips_linux
#undef PLAT_x86_solaris
#undef PLAT_amd64_solaris


#if defined(__APPLE__) && defined(__i386__)
#  define PLAT_x86_darwin 1
#elif defined(__APPLE__) && defined(__x86_64__)
#  define PLAT_amd64_darwin 1
#elif (defined(__MINGW32__) && defined(__i386__)) \
      || defined(__CYGWIN32__) \
      || (defined(_WIN32) && defined(_M_IX86))
#  define PLAT_x86_win32 1
#elif (defined(__MINGW32__) && defined(__x86_64__)) \
      || (defined(_WIN32) && defined(_M_X64))
/* __MINGW32__ and _WIN32 are defined in 64 bit mode as well. */
#  define PLAT_amd64_win64 1
#elif defined(__linux__) && defined(__i386__)
#  define PLAT_x86_linux 1
#elif defined(__linux__) && defined(__x86_64__) && !defined(__ILP32__)
#  define PLAT_amd64_linux 1
#elif defined(__linux__) && defined(__powerpc__) && !defined(__powerpc64__)
#  define PLAT_ppc32_linux 1
#elif defined(__linux__) && defined(__powerpc__) && defined(__powerpc64__) && _CALL_ELF != 2
/* Big Endian uses ELF version 1 */
#  define PLAT_ppc64be_linux 1
#elif defined(__linux__) && defined(__powerpc__) && defined(__powerpc64__) && _CALL_ELF == 2
/* Little Endian uses ELF version 2 */
#  define PLAT_ppc64le_linux 1
#elif defined(__linux__) && defined(__arm__) && !defined(__aarch64__)
#  define PLAT_arm_linux 1
#elif defined(__linux__) && defined(__aarch64__) && !defined(__arm__)
#  define PLAT_arm64_linux 1
#elif defined(__linux__) && defined(__s390__) && defined(__s390x__)
#  define PLAT_s390x_linux 1
#elif defined(__linux__) && defined(__mips__) && (__mips==64)
#  define PLAT_mips64_linux 1
#elif defined(__linux__) && defined(__mips__) && (__mips==32)
#  define PLAT_mips32_linux 1
#elif defined(__linux__) && defined(__nanomips__)
#  define PLAT_nanomips_linux 1
#elif defined(__sun) && defined(__i386__)
#  define PLAT_x86_solaris 1
#elif defined(__sun) && defined(__x86_64__)
#  define PLAT_amd64_solaris 1
#else
/* If we're not compiling for our target platform, don't generate
   any inline asms.  */
#  if !defined(NVALGRIND)
#    define NVALGRIND 1
#  endif
#endif



/* ------------------------------------------------------------------ */
/* ARCHITECTURE SPECIFICS for SPECIAL INSTRUCTIONS.  There is nothing */
/* in here of use to end-users -- skip to the next section.           */
/* ------------------------------------------------------------------ */

/*
 * VALGRIND_DO_CLIENT_REQUEST(): a statement that invokes a Valgrind client
 * request. Accepts both pointers and integers as arguments.
 *
 * VALGRIND_DO_CLIENT_REQUEST_STMT(): a statement that invokes a Valgrind
 * client request that does not return a value.

 * VALGRIND_DO_CLIENT_REQUEST_EXPR(): a C expression that invokes a Valgrind
 * client request and whose value equals the client request result.  Accepts
 * both pointers and integers as arguments.  Note that such calls are not
 * necessarily pure functions -- they may have side effects.
 */

#define VALGRIND_DO_CLIENT_REQUEST(_zzq_rlval, _zzq_default,            \
                                   _zzq_request, _zzq_arg1, _zzq_arg2,  \
                                   _zzq_arg3, _zzq_arg4, _zzq_arg5)     \
  do { (_zzq_rlval) = VALGRIND_DO_CLIENT_REQUEST_EXPR((_zzq_default),   \
                        (_zzq_request), (_zzq_arg1), (_zzq_arg2),       \
                        (_zzq_arg3), (_zzq_arg4), (_zzq_arg5)); } while (0)
/* 定义 VALGRIND_DO_CLIENT_REQUEST_STMT 宏，用于执行 Valgrind 客户端请求 */
#define VALGRIND_DO_CLIENT_REQUEST_STMT(_zzq_request, _zzq_arg1,        \
                           _zzq_arg2,  _zzq_arg3, _zzq_arg4, _zzq_arg5) \
  do { (void) VALGRIND_DO_CLIENT_REQUEST_EXPR(0,                        \
                    (_zzq_request), (_zzq_arg1), (_zzq_arg2),           \
                    (_zzq_arg3), (_zzq_arg4), (_zzq_arg5)); } while (0)

/* 如果定义了 NVALGRIND，则将 VALGRIND_DO_CLIENT_REQUEST_EXPR 定义为返回默认值，不执行 Valgrind 魔术序列 */
#if defined(NVALGRIND)

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
      (_zzq_default)

#else  /* ! NVALGRIND */

/* 如果未定义 NVALGRIND，则定义 VALGRIND_DO_CLIENT_REQUEST_EXPR 为执行 Valgrind 魔术序列 */
/* 该宏用于 Valgrind 客户端请求，确保默认值被放在返回位置，以便在非 Valgrind 环境下正常工作 */
/* 参数以内存块形式传递，目前支持最多五个参数 */
#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
      /* 这里省略了具体的实现细节，因为这些魔术序列由 JIT 编译器特殊处理 */

#endif  /* ! NVALGRIND */

/* 定义 x86 平台上的特殊指令序列前导部分 */
/* 这些指令用于 Valgrind 魔术序列的处理，在不同的平台上具有不同的实现 */
#if defined(PLAT_x86_linux)  ||  defined(PLAT_x86_darwin)  \
    ||  (defined(PLAT_x86_win32) && defined(__GNUC__)) \
    ||  defined(PLAT_x86_solaris)

/* 定义用于包装函数调用的 OrigFn 结构体 */
typedef
   struct { 
      unsigned int nraddr; /* 这里说明了 nraddr 的作用 */
   }
   OrigFn;

/* 定义用于 x86 平台的特殊指令序列前导部分 */
#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     "roll $3,  %%edi ; roll $13, %%edi\n\t"      \
                     "roll $29, %%edi ; roll $19, %%edi\n\t"

#endif  /* PLAT_x86_linux || PLAT_x86_darwin || (PLAT_x86_win32 && __GNUC__) || PLAT_x86_solaris */
#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
  __extension__                                                   \
  ({volatile unsigned int _zzq_args[6];                           \
    volatile unsigned int _zzq_result;                            \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %EDX = client_request ( %EAX ) */         \
                     "xchgl %%ebx,%%ebx"                          \
                     : "=d" (_zzq_result)                         \
                     : "a" (&_zzq_args[0]), "0" (_zzq_default)    \
                     : "cc", "memory"                             \
                    );                                            \
    _zzq_result;                                                  \
  })

注释：

# 定义一个宏 VALGRIND_DO_CLIENT_REQUEST_EXPR，用于执行客户端请求表达式
__extension__ \
({volatile unsigned int _zzq_args[6]; \
  volatile unsigned int _zzq_result; \
  _zzq_args[0] = (unsigned int)(_zzq_request); \
  _zzq_args[1] = (unsigned int)(_zzq_arg1); \
  _zzq_args[2] = (unsigned int)(_zzq_arg2); \
  _zzq_args[3] = (unsigned int)(_zzq_arg3); \
  _zzq_args[4] = (unsigned int)(_zzq_arg4); \
  _zzq_args[5] = (unsigned int)(_zzq_arg5); \
  __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE \
                   /* %EDX = client_request ( %EAX ) */ \
                   "xchgl %%ebx,%%ebx" \
                   : "=d" (_zzq_result) \
                   : "a" (&_zzq_args[0]), "0" (_zzq_default) \
                   : "cc", "memory" \
                  ); \
  _zzq_result; \
})



#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    volatile unsigned int __addr;                                 \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %EAX = guest_NRADDR */                    \
                     "xchgl %%ecx,%%ecx"                          \
                     : "=a" (__addr)                              \
                     :                                            \
                     : "cc", "memory"                             \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }

注释：

# 定义一个宏 VALGRIND_GET_NR_CONTEXT，用于获取上下文号
{ volatile OrigFn* _zzq_orig = &(_zzq_rlval); \
  volatile unsigned int __addr; \
  __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE \
                   /* %EAX = guest_NRADDR */ \
                   "xchgl %%ecx,%%ecx" \
                   : "=a" (__addr) \
                   : \
                   : "cc", "memory" \
                  ); \
  _zzq_orig->nraddr = __addr; \
}



#define VALGRIND_CALL_NOREDIR_EAX                                 \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* call-noredir *%EAX */                     \
                     "xchgl %%edx,%%edx\n\t"

注释：

# 定义一个宏 VALGRIND_CALL_NOREDIR_EAX，用于调用不重定向 EAX 寄存器
__SPECIAL_INSTRUCTION_PREAMBLE \
/* call-noredir *%EAX */ \
"xchgl %%edx,%%edx\n\t"



#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "xchgl %%edi,%%edi\n\t"                     \
                     : : : "cc", "memory"                        \
                    );                                           \
 } while (0)

注释：

# 定义一个宏 VALGRIND_VEX_INJECT_IR，用于注入 IR
do { \
   __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE \
                    "xchgl %%edi,%%edi\n\t" \
                    : : : "cc", "memory" \
                   ); \
} while (0)



#endif /* PLAT_x86_linux || PLAT_x86_darwin || (PLAT_x86_win32 && __GNUC__)
          || PLAT_x86_solaris */

注释：

# 结束 x86_linux、x86_darwin、(x86_win32 并且 __GNUC__)、x86_solaris 平台的条件编译
#endif /* PLAT_x86_linux || PLAT_x86_darwin || (PLAT_x86_win32 && __GNUC__) || PLAT_x86_solaris */



/* ------------------------- x86-Win32 ------------------------- */

注释：

# x86-Win32 平台相关的代码开始
/* ------------------------- x86-Win32 ------------------------- */
#if defined(PLAT_x86_win32) && !defined(__GNUC__)
// 如果定义了 PLAT_x86_win32 并且未定义 __GNUC__

typedef
   struct { 
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;
// 定义结构体 OrigFn，包含一个 unsigned int 类型的成员 nraddr

#if defined(_MSC_VER)
// 如果定义了 _MSC_VER，即使用 Microsoft 编译器

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     __asm rol edi, 3  __asm rol edi, 13          \
                     __asm rol edi, 29 __asm rol edi, 19
// 定义一个特殊指令前导宏 __SPECIAL_INSTRUCTION_PREAMBLE，包含一系列汇编指令

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
    valgrind_do_client_request_expr((uintptr_t)(_zzq_default),    \
        (uintptr_t)(_zzq_request), (uintptr_t)(_zzq_arg1),        \
        (uintptr_t)(_zzq_arg2), (uintptr_t)(_zzq_arg3),           \
        (uintptr_t)(_zzq_arg4), (uintptr_t)(_zzq_arg5))
// 定义一个宏 VALGRIND_DO_CLIENT_REQUEST_EXPR，用于调用 valgrind_do_client_request_expr 函数

static __inline uintptr_t
valgrind_do_client_request_expr(uintptr_t _zzq_default, uintptr_t _zzq_request,
                                uintptr_t _zzq_arg1, uintptr_t _zzq_arg2,
                                uintptr_t _zzq_arg3, uintptr_t _zzq_arg4,
                                uintptr_t _zzq_arg5)
{
    volatile uintptr_t _zzq_args[6];
    volatile unsigned int _zzq_result;
    _zzq_args[0] = (uintptr_t)(_zzq_request);
    _zzq_args[1] = (uintptr_t)(_zzq_arg1);
    _zzq_args[2] = (uintptr_t)(_zzq_arg2);
    _zzq_args[3] = (uintptr_t)(_zzq_arg3);
    _zzq_args[4] = (uintptr_t)(_zzq_arg4);
    _zzq_args[5] = (uintptr_t)(_zzq_arg5);
    __asm { __asm lea eax, _zzq_args __asm mov edx, _zzq_default
            __SPECIAL_INSTRUCTION_PREAMBLE
            /* %EDX = client_request ( %EAX ) */
            __asm xchg ebx,ebx
            __asm mov _zzq_result, edx
    }
    return _zzq_result;
}
// 定义 valgrind_do_client_request_expr 函数，使用汇编语句执行特定操作，返回一个 uintptr_t 类型的结果

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    volatile unsigned int __addr;                                 \
    __asm { __SPECIAL_INSTRUCTION_PREAMBLE                        \
            /* %EAX = guest_NRADDR */                             \
            __asm xchg ecx,ecx                                    \
            __asm mov __addr, eax                                 \
    }                                                             \
    _zzq_orig->nraddr = __addr;                                   \
  }
// 定义一个宏 VALGRIND_GET_NR_CONTEXT，用于获取 nraddr 的上下文信息并存储到 _zzq_rlval 中

#define VALGRIND_CALL_NOREDIR_EAX ERROR
// 定义一个宏 VALGRIND_CALL_NOREDIR_EAX，设置为 ERROR

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm { __SPECIAL_INSTRUCTION_PREAMBLE                       \
            __asm xchg edi,edi                                   \
    }                                                            \
 } while (0)
// 定义一个宏 VALGRIND_VEX_INJECT_IR，用于注入 IR（中间代码）

#else
#error Unsupported compiler.
// 如果不是使用 Microsoft 编译器，输出错误信息：不支持的编译器
#endif

#endif /* PLAT_x86_win32 */

/* ----------------- amd64-{linux,darwin,solaris} --------------- */

#if defined(PLAT_amd64_linux)  ||  defined(PLAT_amd64_darwin) \
    ||  defined(PLAT_amd64_solaris) \
// 如果定义了 PLAT_amd64_linux 或者 PLAT_amd64_darwin 或者 PLAT_amd64_solaris
    ||  (defined(PLAT_amd64_win64) && defined(__GNUC__))
typedef
   struct { 
      unsigned long int nraddr; /* where's the code? */
   }
   OrigFn;

// 定义一个特殊指令的前缀字符串，包含一系列的指令操作
#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     "rolq $3,  %%rdi ; rolq $13, %%rdi\n\t"      \
                     "rolq $61, %%rdi ; rolq $51, %%rdi\n\t"

// 定义一个宏，用于执行Valgrind客户端请求，包括传递的参数和返回结果
#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
    __extension__                                                 \
    ({ volatile unsigned long int _zzq_args[6];                   \
    volatile unsigned long int _zzq_result;                       \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    // 使用内联汇编执行Valgrind请求，返回结果存储在_zzq_result中
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %RDX = client_request ( %RAX ) */         \
                     "xchgq %%rbx,%%rbx"                          \
                     : "=d" (_zzq_result)                         \
                     : "a" (&_zzq_args[0]), "0" (_zzq_default)    \
                     : "cc", "memory"                             \
                    );                                            \
    _zzq_result;                                                  \
    })

// 定义一个宏，用于获取Valgrind的上下文编号，将结果存储到指定结构体OrigFn的nraddr字段中
#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    volatile unsigned long int __addr;                            \
    // 使用内联汇编执行Valgrind请求，将结果存储在__addr中，然后写入到_zzq_orig->nraddr
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %RAX = guest_NRADDR */                    \
                     "xchgq %%rcx,%%rcx"                          \
                     : "=a" (__addr)                              \
                     :                                            \
                     : "cc", "memory"                             \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }

// 定义一个宏，用于执行Valgrind调用，不重定向RAX寄存器
#define VALGRIND_CALL_NOREDIR_RAX                                 \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* call-noredir *%RAX */                     \
                     "xchgq %%rdx,%%rdx\n\t"

// 定义一个宏，用于注入VEX IR（Valgrind内部表示），但未提供具体的实现部分

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    # 使用内联汇编语言编写的特殊指令，包含一个预处理器宏来插入汇编指令的前导部分
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "xchgq %%rdi,%%rdi\n\t"                     \
                     : : : "cc", "memory"                        \
                    );
    # 使用 do-while 结构确保上述代码块被视为单一语句，并终止 do-while 循环
    } while (0)
/* ------------------------- amd64-Win64 ------------------------- */
/* 检查平台定义是否为 amd64_win64，并且不是使用 GCC 编译器 */
#if defined(PLAT_amd64_win64) && !defined(__GNUC__)
/* 如果是不支持的编译器，则抛出错误 */
#error Unsupported compiler.
#endif /* PLAT_amd64_win64 */

/* ------------------------ ppc32-linux ------------------------ */

/* 定义结构体 OrigFn，包含一个无符号整数成员 nraddr，表示代码的地址 */
typedef
   struct { 
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;

/* 定义宏 __SPECIAL_INSTRUCTION_PREAMBLE，包含一系列特殊指令 */
#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                    "rlwinm 0,0,3,0,31  ; rlwinm 0,0,13,0,31\n\t" \
                    "rlwinm 0,0,29,0,31 ; rlwinm 0,0,19,0,31\n\t"

/* 定义宏 VALGRIND_DO_CLIENT_REQUEST_EXPR，执行 Valgrind 客户端请求的表达式 */
#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
    __extension__                                                 \
  ({         unsigned int  _zzq_args[6];                          \
             unsigned int  _zzq_result;                           \
             unsigned int* _zzq_ptr;                              \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
    _zzq_ptr = _zzq_args;                                         \
    /* 内联汇编执行 Valgrind 请求 */                              \
    __asm__ volatile("mr 3,%1\n\t" /*default*/                    \
                     "mr 4,%2\n\t" /*ptr*/                        \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = client_request ( %R4 ) */           \
                     "or 1,1,1\n\t"                               \
                     "mr %0,3"     /*result*/                     \
                     : "=b" (_zzq_result)                         \
                     : "b" (_zzq_default), "b" (_zzq_ptr)         \
                     : "cc", "memory", "r3", "r4");               \
    _zzq_result;                                                  \
    })

/* 定义宏 VALGRIND_GET_NR_CONTEXT，获取 Valgrind 上下文编号 */
#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned int __addr;                                          \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = guest_NRADDR */                     \
                     "or 2,2,2\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );                                            \

这段代码使用了内联汇编语言。下面是对每一行的解释：

1. `__asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \`
   - 这行开始定义一个内联汇编块，`__asm__`表示后续是汇编代码，`volatile`表示编译器不应优化此处的代码。
   - `__SPECIAL_INSTRUCTION_PREAMBLE`可能是一个宏，用于插入特殊的汇编指令前缀。

2. `/* %R3 = guest_NRADDR */                     \`
   - 注释，说明接下来的汇编指令的作用是将 `%R3` 设置为 `guest_NRADDR`。

3. `"or 2,2,2\n\t"                               \`
   - 汇编指令，执行按位或运算。具体操作为将寄存器 `2`（通常是通用寄存器）与自身进行按位或运算，结果仍存入寄存器 `2`。

4. `"mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );                                            \`
   - 汇编指令，将寄存器 `3` 的值移动到输出操作数 `%0` 所指示的位置（通常是变量 `__addr`）。
   - `: "=b" (__addr)` 表示输出操作数约束，将 `__addr` 关联到寄存器 `3` 的值。
   - 约束 `: "cc", "memory", "r3"` 指示编译器保留的寄存器和内存约束。

5. `);                                            \`
   - 结束内联汇编块。

6. `_zzq_orig->nraddr = __addr;                                   \`
   - 将变量 `__addr` 的值赋给 `_zzq_orig` 结构体或对象中的 `nraddr` 成员变量。

7. `}`                                                      
   - 结束代码块。
#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R11 */       \
                     "or 3,3,3\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or 5,5,5\n\t"                              \
                    );                                           \
 } while (0)
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = guest_NRADDR */                     \
                     "or 2,2,2\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \  // 将寄存器3的值赋给__addr变量，并且将__addr的值作为输出操作数（b寄存器）
                     :                                            \  // 没有输入操作数
                     : "cc", "memory", "r3"                       \  // 指定被修改的标志寄存器（cc），内存（memory）和寄存器3（r3）
                    );                                            \  // 结束汇编指令

    _zzq_orig->nraddr = __addr;                                   \  // 将__addr变量的值赋给_zzq_orig结构体的nraddr成员

    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = guest_NRADDR_GPR2 */                \
                     "or 4,4,4\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \  // 将寄存器3的值赋给__addr变量，并且将__addr的值作为输出操作数（b寄存器）
                     :                                            \  // 没有输入操作数
                     : "cc", "memory", "r3"                       \  // 指定被修改的标志寄存器（cc），内存（memory）和寄存器3（r3）
                    );                                            \  // 结束汇编指令

    _zzq_orig->r2 = __addr;                                       \  // 将__addr变量的值赋给_zzq_orig结构体的r2成员
  }
#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R11 */       \
                     "or 3,3,3\n\t"



#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     // 执行特殊指令前导部分 \
                     /* branch-and-link-to-noredir *%R11 */       \
                     "or 3,3,3\n\t"



#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or 5,5,5\n\t"                              \
                    );                                           \
 } while (0)



#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    // 执行特殊指令前导部分 \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or 5,5,5\n\t"                              \
                    );                                           \
 } while (0)



#endif /* PLAT_ppc64be_linux */



#endif /* PLAT_ppc64be_linux */



#if defined(PLAT_ppc64le_linux)



#if defined(PLAT_ppc64le_linux)



typedef
   struct {
      unsigned long int nraddr; /* where's the code? */
      unsigned long int r2;     /* what tocptr do we need? */
   }
   OrigFn;



typedef
   struct {
      unsigned long int nraddr; // 代码所在地址
      unsigned long int r2;     // 需要的 tocptr 指针
   }
   OrigFn;



#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     "rotldi 0,0,3  ; rotldi 0,0,13\n\t"          \
                     "rotldi 0,0,61 ; rotldi 0,0,51\n\t"



#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     // 执行特殊指令前导部分 \
                     "rotldi 0,0,3  ; rotldi 0,0,13\n\t"          \
                     "rotldi 0,0,61 ; rotldi 0,0,51\n\t"



#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({         unsigned long int  _zzq_args[6];                     \
             unsigned long int  _zzq_result;                      \
             unsigned long int* _zzq_ptr;                         \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    _zzq_ptr = _zzq_args;                                         \
    __asm__ volatile("mr 3,%1\n\t" /*default*/                    \
                     "mr 4,%2\n\t" /*ptr*/                        \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = client_request ( %R4 ) */           \
                     "or 1,1,1\n\t"                               \
                     "mr %0,3"     /*result*/                     \
                     : "=b" (_zzq_result)                         \
                     : "b" (_zzq_default), "b" (_zzq_ptr)         \
                     : "cc", "memory", "r3", "r4");               \
    _zzq_result;                                                  \
  })



#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({         unsigned long int  _zzq_args[6];                     \
             unsigned long int  _zzq_result;                      \
             unsigned long int* _zzq_ptr;                         \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    _zzq_ptr = _zzq_args;                                         \
    // 执行特殊指令前导部分 \
    __asm__ volatile("mr 3,%1\n\t" /*default*/                    \
                     "mr 4,%2\n\t" /*ptr*/                        \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = client_request ( %R4 ) */           \
                     "or 1,1,1\n\t"                               \
                     "mr %0,3"     /*result*/                     \
                     : "=b" (_zzq_result)                         \
                     : "b" (_zzq_default), "b" (_zzq_ptr)         \
                     : "cc", "memory", "r3", "r4");               \
    _zzq_result;                                                  \
  })



#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned long int __addr;                                     \



#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned long int __addr;                                     \



#endif



#endif
    // 使用内联汇编嵌入特殊指令和参数预处理部分
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     // 设置寄存器 %R3 = guest_NRADDR
                     "or 2,2,2\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );
    // 将获取的地址值赋给 _zzq_orig 结构体的 nraddr 成员变量
    _zzq_orig->nraddr = __addr;
    // 使用内联汇编嵌入特殊指令和参数预处理部分
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     // 设置寄存器 %R3 = guest_NRADDR_GPR2
                     "or 4,4,4\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );
    // 将获取的地址值赋给 _zzq_orig 结构体的 r2 成员变量
    _zzq_orig->r2 = __addr;
  }
#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R12 */       \
                     "or 3,3,3\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or 5,5,5\n\t"                              \
                    );                                           \
 } while (0)

#endif /* PLAT_ppc64le_linux */

/* ------------------------- arm-linux ------------------------- */

#if defined(PLAT_arm_linux)

typedef
   struct { 
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
            "mov r12, r12, ror #3  ; mov r12, r12, ror #13 \n\t"  \
            "mov r12, r12, ror #29 ; mov r12, r12, ror #19 \n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({volatile unsigned int  _zzq_args[6];                          \
    volatile unsigned int  _zzq_result;                           \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
    __asm__ volatile("mov r3, %1\n\t" /*default*/                 \
                     "mov r4, %2\n\t" /*ptr*/                     \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* R3 = client_request ( R4 ) */             \
                     "orr r10, r10, r10\n\t"                      \
                     "mov %0, r3"     /*result*/                  \
                     : "=r" (_zzq_result)                         \
                     : "r" (_zzq_default), "r" (&_zzq_args[0])    \
                     : "cc","memory", "r3", "r4");                \
    _zzq_result;                                                  \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned int __addr;                                          \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* R3 = guest_NRADDR */                      \
                     "orr r11, r11, r11\n\t"                      \
                     "mov %0, r3"                                 \
                     : "=r" (__addr)                              \  # 将寄存器 R3 的值存入变量 __addr 中
                     :                                            \  # 输入寄存器列表为空，无需输入寄存器
                     : "cc", "memory", "r3"                       \  # clobber 列表包含条件码寄存器、内存和寄存器 R3
                    );                                            \  # 内联汇编结束
    _zzq_orig->nraddr = __addr;                                   \  # 将变量 __addr 的值赋给 _zzq_orig 结构体中的 nraddr 成员
  }
#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                    \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R4 */        \
                     "orr r12, r12, r12\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "orr r9, r9, r9\n\t"                        \
                     : : : "cc", "memory"                        \
                    );                                           \
 } while (0)

#endif /* PLAT_arm_linux */

/* ------------------------ arm64-linux ------------------------- */

#if defined(PLAT_arm64_linux)

typedef
   struct { 
      unsigned long int nraddr; /* where's the code? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
            "ror x12, x12, #3  ;  ror x12, x12, #13 \n\t"         \
            "ror x12, x12, #51 ;  ror x12, x12, #61 \n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({volatile unsigned long int  _zzq_args[6];                     \
    volatile unsigned long int  _zzq_result;                      \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    __asm__ volatile("mov x3, %1\n\t" /*default*/                 \
                     "mov x4, %2\n\t" /*ptr*/                     \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* X3 = client_request ( X4 ) */             \
                     "orr x10, x10, x10\n\t"                      \
                     "mov %0, x3"     /*result*/                  \
                     : "=r" (_zzq_result)                         \
                     : "r" ((unsigned long int)(_zzq_default)),   \
                       "r" (&_zzq_args[0])                        \
                     : "cc","memory", "x3", "x4");                \
    _zzq_result;                                                  \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned long int __addr;                                     \


注释：


#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                    \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R4 */        \
                     "orr r12, r12, r12\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "orr r9, r9, r9\n\t"                        \
                     : : : "cc", "memory"                        \
                    );                                           \
 } while (0)

这段代码定义了两个宏，用于在特定硬件架构上执行特殊指令。第一个宏`VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4`执行了一个与硬件相关的操作，注释说明它是将寄存器R4设为分支链接到无重定向模式。第二个宏`VALGRIND_VEX_INJECT_IR()`利用内联汇编来注入一个特定的指令，注释说明这是为了在硬件上注入IR（指令记录）。


#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
            "ror x12, x12, #3  ;  ror x12, x12, #13 \n\t"         \
            "ror x12, x12, #51 ;  ror x12, x12, #61 \n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({volatile unsigned long int  _zzq_args[6];                     \
    volatile unsigned long int  _zzq_result;                      \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    __asm__ volatile("mov x3, %1\n\t" /*default*/                 \
                     "mov x4, %2\n\t" /*ptr*/                     \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* X3 = client_request ( X4 ) */             \
                     "orr x10, x10, x10\n\t"                      \
                     "mov %0, x3"     /*result*/                  \
                     : "=r" (_zzq_result)                         \
                     : "r" ((unsigned long int)(_zzq_default)),   \
                       "r" (&_zzq_args[0])                        \
                     : "cc","memory", "x3", "x4");                \
    _zzq_result;                                                  \
  })

这部分代码定义了一些在arm64-linux平台上使用的宏和内联汇编函数，用于执行客户端请求。`__SPECIAL_INSTRUCTION_PREAMBLE`宏展开了一系列的移位指令，用于预处理。`VALGRIND_DO_CLIENT_REQUEST_EXPR`宏利用内联汇编实现了对客户端请求的处理，注释解释了其参数和返回值的含义。


#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned long int __addr;                                     \

最后这段代码定义了一个宏`VALGRIND_GET_NR_CONTEXT`，它声明了一个结构体指针和一个无符号长整型变量，用于获取上下文。
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* X3 = guest_NRADDR */                      \
                     "orr x11, x11, x11\n\t"                      \  # 执行汇编指令：逻辑或运算，将寄存器 x11 的值与自身进行或操作
                     "mov %0, x3"                                 \  # 执行汇编指令：将寄存器 x3 的值移动到占位符 %0 所代表的变量 __addr 中
                     : "=r" (__addr)                              \  # 声明输出操作数 "=r"，将结果写入 __addr 变量
                     :                                            \  # 声明输入操作数列表为空
                     : "cc", "memory", "x3"                       \  # 声明影响条件码寄存器 ("cc")、内存 ("memory") 和寄存器 x3 的部分
                    );                                            \  # 结束汇编代码块
    _zzq_orig->nraddr = __addr;                                   \  # 将 __addr 的值赋给 _zzq_orig 结构体的成员变量 nraddr
  }
#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                    \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir X8 */          \
                     "orr x12, x12, x12\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "orr x9, x9, x9\n\t"                        \
                     : : : "cc", "memory"                        \
                    );                                           \
 } while (0)
/* 定义一个宏 VALGRIND_VEX_INJECT_IR()，用于插入 Valgrind 特定的 IR 指令。
 * 这段代码使用内联汇编语法，执行特定于 Valgrind 的指令序列。
 * 该序列包括对__SPECIAL_INSTRUCTION_PREAMBLE的调用，然后执行"orr x9, x9, x9"指令。
 * 最后使用 "cc" 和 "memory" 作为输入、输出操作数。
 */

#endif /* PLAT_arm64_linux */

/* ------------------------ s390x-linux ------------------------ */

#if defined(PLAT_s390x_linux)

typedef
  struct {
     unsigned long int nraddr; /* where's the code? */
  }
  OrigFn;

/* __SPECIAL_INSTRUCTION_PREAMBLE will be used to identify Valgrind specific
 * code. This detection is implemented in platform specific toIR.c
 * (e.g. VEX/priv/guest_s390_decoder.c).
 */
#define __SPECIAL_INSTRUCTION_PREAMBLE                           \
                     "lr 15,15\n\t"                              \
                     "lr 1,1\n\t"                                \
                     "lr 2,2\n\t"                                \
                     "lr 3,3\n\t"
/* 定义一个宏 __SPECIAL_INSTRUCTION_PREAMBLE，用于包含 Valgrind 特定的指令前导。
 * 这些指令在 s390x-linux 平台上执行特定的寄存器加载操作，用于识别 Valgrind 特定的代码。
 * 这些指令用于设置寄存器 15, 1, 2 和 3 的值。
 */

#define __CLIENT_REQUEST_CODE "lr 2,2\n\t"
/* 定义一个宏 __CLIENT_REQUEST_CODE，用于将寄存器 2 设置为 2 的值。 */

#define __GET_NR_CONTEXT_CODE "lr 3,3\n\t"
/* 定义一个宏 __GET_NR_CONTEXT_CODE，用于将寄存器 3 设置为 3 的值。 */

#define __CALL_NO_REDIR_CODE  "lr 4,4\n\t"
/* 定义一个宏 __CALL_NO_REDIR_CODE，用于将寄存器 4 设置为 4 的值。 */

#define __VEX_INJECT_IR_CODE  "lr 5,5\n\t"
/* 定义一个宏 __VEX_INJECT_IR_CODE，用于将寄存器 5 设置为 5 的值。 */
#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                         \
       _zzq_default, _zzq_request,                               \
       _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
  __extension__                                                  \
 ({volatile unsigned long int _zzq_args[6];                      \
   volatile unsigned long int _zzq_result;                       \
   _zzq_args[0] = (unsigned long int)(_zzq_request);             \
   _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
   _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
   _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
   _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
   _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
   __asm__ volatile(/* r2 = args */                              \
                    "lgr 2,%1\n\t"                               \
                    /* r3 = default */                           \
                    "lgr 3,%2\n\t"                               \
                    __SPECIAL_INSTRUCTION_PREAMBLE               \
                    __CLIENT_REQUEST_CODE                        \
                    /* results = r3 */                           \
                    "lgr %0, 3\n\t"                              \
                    : "=d" (_zzq_result)                         \
                    : "a" (&_zzq_args[0]), "0" (_zzq_default)    \
                    : "cc", "2", "3", "memory"                   \
                   );                                            \
   _zzq_result;                                                  \
 })

/*
  Macro VALGRIND_DO_CLIENT_REQUEST_EXPR:

  This macro performs a client request operation using inline assembly on architectures
  where direct manipulation of registers and special instructions are necessary.

  It takes several parameters:
  - _zzq_default: Default value to use if no specific request is given
  - _zzq_request: The specific request to execute
  - _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5: Arguments for the request

  The macro sets up an array of volatile unsigned long integers (_zzq_args) to hold
  the arguments and a volatile unsigned long integer (_zzq_result) for the result.
  It then uses inline assembly to perform the client request, setting registers
  according to the arguments and executing the request code (__CLIENT_REQUEST_CODE).
  Finally, it returns the result (_zzq_result).
*/

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                      \
 { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
   volatile unsigned long int __addr;                            \
   __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                    __GET_NR_CONTEXT_CODE                        \
                    "lgr %0, 3\n\t"                              \
                    : "=a" (__addr)                              \
                    :                                            \
                    : "cc", "3", "memory"                        \
                   );                                            \
   _zzq_orig->nraddr = __addr;                                   \
 }

/*
  Macro VALGRIND_GET_NR_CONTEXT:

  This macro retrieves the number of the current execution context using inline assembly.
  It sets up a volatile OrigFn pointer (_zzq_orig) to store the result address.
  The address itself is obtained through inline assembly (__GET_NR_CONTEXT_CODE),
  which fetches it into __addr using register r3.
  Finally, the macro assigns __addr to _zzq_orig->nraddr.
*/

#define VALGRIND_CALL_NOREDIR_R1                                 \
                    __SPECIAL_INSTRUCTION_PREAMBLE               \
                    __CALL_NO_REDIR_CODE

/*
  Macro VALGRIND_CALL_NOREDIR_R1:

  This macro executes a special instruction (__CALL_NO_REDIR_CODE) without redirection,
  using inline assembly (__SPECIAL_INSTRUCTION_PREAMBLE).
  It is designed to perform certain operations without redirecting them.
*/

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     __VEX_INJECT_IR_CODE);                      \
 } while (0)

/*
  Macro VALGRIND_VEX_INJECT_IR:

  This macro injects intermediate representation (IR) into the VEX translation
  engine using inline assembly (__SPECIAL_INSTRUCTION_PREAMBLE and __VEX_INJECT_IR_CODE).
  It is typically used for dynamic binary instrumentation and analysis purposes.
*/

#endif /* PLAT_s390x_linux */

/* ------------------------- mips32-linux ---------------- */

/*
  The end of the file with a section marker for mips32-linux.
  This typically denotes the end of a platform-specific section
  or the start of another platform's specific section.
*/
#if defined(PLAT_mips32_linux)
// 如果定义了 PLAT_mips32_linux，则编译以下代码段

typedef
   struct { 
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;
// 定义了一个结构体 OrigFn，包含一个无符号整数成员 nraddr，用于存储地址信息

/* .word  0x342
 * .word  0x742
 * .word  0xC2
 * .word  0x4C2*/
#define __SPECIAL_INSTRUCTION_PREAMBLE          \
                     "srl $0, $0, 13\n\t"       \
                     "srl $0, $0, 29\n\t"       \
                     "srl $0, $0, 3\n\t"        \
                     "srl $0, $0, 19\n\t"
// 定义了一个宏 __SPECIAL_INSTRUCTION_PREAMBLE，展开时包含四条 MIPS 汇编指令

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
       _zzq_default, _zzq_request,                                \
       _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)     \
  __extension__                                                   \
  ({ volatile unsigned int _zzq_args[6];                          \
    volatile unsigned int _zzq_result;                            \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
    // 执行一段汇编代码，使用输入参数进行客户端请求，并返回结果
    __asm__ volatile("move $11, %1\n\t" /*default*/               \
                     "move $12, %2\n\t" /*ptr*/                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* T3 = client_request ( T4 ) */             \
                     "or $13, $13, $13\n\t"                       \
                     "move %0, $11\n\t"     /*result*/            \
                     : "=r" (_zzq_result)                         \
                     : "r" (_zzq_default), "r" (&_zzq_args[0])    \
                     : "$11", "$12", "memory");                   \
    _zzq_result;                                                  \
  })
// 定义了一个宏 VALGRIND_DO_CLIENT_REQUEST_EXPR，用于执行客户端请求，并返回执行结果

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    volatile unsigned int __addr;                                 \
    // 获取当前地址信息，并存储在 _zzq_orig->nraddr 中
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %t9 = guest_NRADDR */                     \
                     "or $14, $14, $14\n\t"                       \
                     "move %0, $11"     /*result*/                \
                     : "=r" (__addr)                              \
                     :                                            \
                     : "$11"                                      \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }
// 定义了一个宏 VALGRIND_GET_NR_CONTEXT，用于获取当前上下文的地址信息，并存储在指定结构体的 nraddr 成员中

#endif // 结束条件编译指令，结束 PLAT_mips32_linux 的条件编译
#define VALGRIND_CALL_NOREDIR_T9                                 \
                     __SPECIAL_INSTRUCTION_PREAMBLE              \
                     /* call-noredir *%t9 */                     \
                     "or $15, $15, $15\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or $11, $11, $11\n\t"                      \
                    );                                           \
 } while (0)


#endif /* PLAT_mips32_linux */

/* ------------------------- mips64-linux ---------------- */

#if defined(PLAT_mips64_linux)

typedef
   struct {
      unsigned long nraddr; /* where's the code? */
   }
   OrigFn;

/* dsll $0,$0, 3
 * dsll $0,$0, 13
 * dsll $0,$0, 29
 * dsll $0,$0, 19*/
#define __SPECIAL_INSTRUCTION_PREAMBLE                              \
                     "dsll $0,$0, 3 ; dsll $0,$0,13\n\t"            \
                     "dsll $0,$0,29 ; dsll $0,$0,19\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                            \
       _zzq_default, _zzq_request,                                  \
       _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)       \
  __extension__                                                     \
  ({ volatile unsigned long int _zzq_args[6];                       \
    volatile unsigned long int _zzq_result;                         \
    _zzq_args[0] = (unsigned long int)(_zzq_request);               \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                  \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                  \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                  \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                  \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                  \
        __asm__ volatile("move $11, %1\n\t" /*default*/             \
                         "move $12, %2\n\t" /*ptr*/                 \
                         __SPECIAL_INSTRUCTION_PREAMBLE             \
                         /* $11 = client_request ( $12 ) */         \
                         "or $13, $13, $13\n\t"                     \
                         "move %0, $11\n\t"     /*result*/          \
                         : "=r" (_zzq_result)                       \
                         : "r" (_zzq_default), "r" (&_zzq_args[0])  \
                         : "$11", "$12", "memory");                 \
    _zzq_result;                                                    \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                         \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                     \
    volatile unsigned long int __addr;                              \


注释：


#define VALGRIND_CALL_NOREDIR_T9                                 \
                     __SPECIAL_INSTRUCTION_PREAMBLE              \
                     /* call-noredir *%t9 */                     \
                     "or $15, $15, $15\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    // 插入特殊指令前导，将 $11 寄存器设置为 $11 的值
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or $11, $11, $11\n\t"                      \
                    );                                           \
 } while (0)


#endif /* PLAT_mips32_linux */

/* ------------------------- mips64-linux ---------------- */

#if defined(PLAT_mips64_linux)

typedef
   struct {
      unsigned long nraddr; /* where's the code? */
   }
   OrigFn;

/* dsll $0,$0, 3
 * dsll $0,$0, 13
 * dsll $0,$0, 29
 * dsll $0,$0, 19*/
// 定义特殊指令前导，通过多次左移 $0 寄存器来设置前导内容
#define __SPECIAL_INSTRUCTION_PREAMBLE                              \
                     "dsll $0,$0, 3 ; dsll $0,$0,13\n\t"            \
                     "dsll $0,$0,29 ; dsll $0,$0,19\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                            \
       _zzq_default, _zzq_request,                                  \
       _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)       \
  __extension__                                                     \
  ({ volatile unsigned long int _zzq_args[6];                       \
    volatile unsigned long int _zzq_result;                         \
    _zzq_args[0] = (unsigned long int)(_zzq_request);               \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                  \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                  \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                  \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                  \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                  \
    // 使用汇编指令执行客户端请求，并将结果存入 _zzq_result
        __asm__ volatile("move $11, %1\n\t" /*default*/             \
                         "move $12, %2\n\t" /*ptr*/                 \
                         __SPECIAL_INSTRUCTION_PREAMBLE             \
                         /* $11 = client_request ( $12 ) */         \
                         "or $13, $13, $13\n\t"                     \
                         "move %0, $11\n\t"     /*result*/          \
                         : "=r" (_zzq_result)                       \
                         : "r" (_zzq_default), "r" (&_zzq_args[0])  \
                         : "$11", "$12", "memory");                 \
    _zzq_result;                                                    \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                         \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                     \
    volatile unsigned long int __addr;                              \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE                 \
                     /* $11 = guest_NRADDR */                       \
                     "or $14, $14, $14\n\t"                         \
                     "move %0, $11"     /*result*/                  \
                     : "=r" (__addr)                                \
                     :                                              \
                     : "$11");                                      \
    _zzq_orig->nraddr = __addr;                                     \
  }



// 使用内联汇编嵌入特殊指令，执行以下操作：
// - 设置特殊指令前导
// - 将寄存器 $11（guest_NRADDR）的值赋给 __addr
// - 执行逻辑或操作，将 $14 寄存器与自身进行或运算
// - 将 $11 的值移动到 %0 寄存器（结果寄存器）
// - 输出限定符 "=r" 表示将结果写入 __addr 变量
// - 输入限定符 "$11" 表示内联汇编使用 $11 寄存器
// - 没有使用的寄存器列表为空

// 将 __addr 的值赋给 _zzq_orig 结构体的 nraddr 成员变量
_zzq_orig->nraddr = __addr;


这段代码是一段使用内联汇编的 C 代码片段，其中使用了特殊的汇编语法来操作寄存器和变量。
#define VALGRIND_CALL_NOREDIR_T9                                    \
                     __SPECIAL_INSTRUCTION_PREAMBLE                 \
                     /* call-noredir $25 */                         \
                     "or $15, $15, $15\n\t"

#define VALGRIND_VEX_INJECT_IR()                                    \
 do {                                                               \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE                 \
                     "or $11, $11, $11\n\t"                         \
                    );                                              \
 } while (0)

#endif /* PLAT_mips64_linux */

#if defined(PLAT_nanomips_linux)

typedef
   struct {
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;
/*
   8000 c04d  srl  zero, zero, 13
   8000 c05d  srl  zero, zero, 29
   8000 c043  srl  zero, zero,  3
   8000 c053  srl  zero, zero, 19
*/

// 定义特殊指令序言字符串，包含多个移位指令
#define __SPECIAL_INSTRUCTION_PREAMBLE "srl[32] $zero, $zero, 13 \n\t" \
                                       "srl[32] $zero, $zero, 29 \n\t" \
                                       "srl[32] $zero, $zero, 3  \n\t" \
                                       "srl[32] $zero, $zero, 19 \n\t"

// 执行Valgrind的VEX插入IR指令的宏定义
#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
       _zzq_default, _zzq_request,                                \
       _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)     \
  __extension__                                                   \
  ({ volatile unsigned int _zzq_args[6];                          \
    volatile unsigned int _zzq_result;                            \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
    // 内联汇编调用Valgrind客户端请求函数，并获取返回结果
    __asm__ volatile("move $a7, %1\n\t" /* default */             \
                     "move $t0, %2\n\t" /* ptr */                 \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* $a7 = client_request( $t0 ) */            \
                     "or[32] $t0, $t0, $t0\n\t"                   \
                     "move %0, $a7\n\t"     /* result */          \
                     : "=r" (_zzq_result)                         \
                     : "r" (_zzq_default), "r" (&_zzq_args[0])    \
                     : "$a7", "$t0", "memory");                   \
    _zzq_result;                                                  \
  })

// 获取当前上下文的宏定义，存储在指定的变量中
#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                         \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                     \
    volatile unsigned long int __addr;                              \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE                 \
                     /* $a7 = guest_NRADDR */                       \
                     "or[32] $t1, $t1, $t1\n\t"                     \
                     "move %0, $a7"     /*result*/                  \
                     : "=r" (__addr)                                \
                     :                                              \
                     : "$a7");                                      \

这段代码使用了内联汇编（inline assembly），用于执行特定的汇编指令序列。以下是对每行的注释：

- `__asm__ volatile(`：开始定义内联汇编块，`volatile` 表示编译器不应该对这段代码进行优化或重排。
- `__SPECIAL_INSTRUCTION_PREAMBLE`：这是一个宏或标识符，用于表示特殊的汇编指令前导部分。

- `/* $a7 = guest_NRADDR */`：注释说明，此行将 `$a7` 寄存器设置为 `guest_NRADDR` 的值。

- `"or[32] $t1, $t1, $t1\n\t"`：汇编指令，使用逻辑或操作符将 `$t1` 寄存器与自身进行逻辑或操作。

- `"move %0, $a7"`：将 `$a7` 寄存器的值移动到输出操作数 `%0` 中，作为结果。

- `: "=r" (__addr)`：约束说明，指定了如何将结果（即 `$a7` 寄存器的值）输出给变量 `__addr`。

- `: `：输入操作数列表为空，表示这段内联汇编没有输入操作数。

- `: "$a7");`：指定了使用了哪些寄存器，这里表示使用了 `$a7` 寄存器。


    _zzq_orig->nraddr = __addr;                                     \

- ` _zzq_orig->nraddr = __addr;`：将 `__addr` 变量的值赋给 `_zzq_orig` 结构体或对象的 `nraddr` 成员。


  }

- `}`：结束内联汇编块。
/* 定义 VALGRIND_CALL_NOREDIR_T9 宏，用于生成特殊指令前导 */
#define VALGRIND_CALL_NOREDIR_T9                                    \
                     __SPECIAL_INSTRUCTION_PREAMBLE                 \
                     /* call-noredir $25 */                         \
                     "or[32] $t2, $t2, $t2\n\t"

/* 定义 VALGRIND_VEX_INJECT_IR 宏，用于在插入汇编代码时注入 IR */
#define VALGRIND_VEX_INJECT_IR()                                    \
 do {                                                               \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE                 \
                     "or[32] $t3, $t3, $t3\n\t"                     \
                    );                                              \
 } while (0)

/* 结束条件：不再定义 NVALGRIND */
#endif

/* 在其他平台插入汇编代码的部分... */

/* 结束条件：不再定义 NVALGRIND */
#endif /* NVALGRIND */

/* ------------------------------------------------------------------ */
/* 函数包装的平台特定实现。这些代码非常丑陋，但这是我能想到的最好的折衷方案。 */
/* ------------------------------------------------------------------ */

/* 定义用于函数包装的特殊宏，确保没有重定向。目的是从函数包装器到它们包装的函数之间构建标准调用序列，使用特殊的无重定向调用伪指令，JIT 理解并特殊处理。这一节很长和重复，我无法想到更短的方法。 */

/* 命名方案如下：

   CALL_FN_{W,v}_{v,W,WW,WWW,WWWW,5W,6W,7W,etc}

   'W' 表示 "word"，'v' 表示 "void"。因此，不同的宏用于调用 arity 0、1、2、3、4 等函数，每个函数可能返回一个字型结果，也可能没有结果。
*/

/* 使用这些宏来编写您的包装器的名称。注意：在 pub_tool_redir.h 中与 VG_WRAP_FUNCTION_Z{U,Z} 重复。还注意：将默认行为等价类标签 "0000" 插入名称中。详见 pub_tool_redir.h -- 通常情况下不需要考虑这一点。 */

/* 使用额外的宏级别确保 soname/fnname 参数在粘贴在一起之前已完全宏展开。 */
#define VG_CONCAT4(_aa,_bb,_cc,_dd) _aa##_bb##_cc##_dd

/* 定义用于生成函数包装器名称的宏，插入 "ZU" 到名称中 */
#define I_WRAP_SONAME_FNNAME_ZU(soname,fnname)                    \
   VG_CONCAT4(_vgw00000ZU_,soname,_,fnname)

/* 定义用于生成函数包装器名称的宏，插入 "ZZ" 到名称中 */
#define I_WRAP_SONAME_FNNAME_ZZ(soname,fnname)                    \
   VG_CONCAT4(_vgw00000ZZ_,soname,_,fnname)

/* 在包装器函数内部使用此宏来收集原始函数的上下文（地址和可能的其他信息）。一旦获取了这些信息，就可以在 CALL_FN_ 宏中使用它。参数 _lval 的类型是 OrigFn。 */
#define VALGRIND_GET_ORIG_FN(_lval)  VALGRIND_GET_NR_CONTEXT(_lval)
/* 
   提供终端用户替换函数的便捷方式，而不是简单包装。替换函数与包装函数的区别在于，它无法获取被调用的原始函数，因此无法继续调用它。
   在替换函数中，VALGRIND_GET_ORIG_FN 总是返回零。
*/

/*
   宏定义：根据库名（soname）和函数名（fnname）生成一个唯一的标识符。
   这些宏允许动态生成函数名，并用于在Valgrind工具中进行替换函数的实现。
*/

#define I_REPLACE_SONAME_FNNAME_ZU(soname,fnname)                 \
   VG_CONCAT4(_vgr00000ZU_,soname,_,fnname)

#define I_REPLACE_SONAME_FNNAME_ZZ(soname,fnname)                 \
   VG_CONCAT4(_vgr00000ZZ_,soname,_,fnname)

/*
   下面是用于调用返回void的函数的宏定义。
   这些宏定义了多个版本的函数调用宏，每个版本中的参数个数不同（从一个到七个）。
   这些宏使用了一个“volatile unsigned long _junk”变量，以确保编译器不会优化掉这些宏调用。
*/

#define CALL_FN_v_v(fnptr)                                        \
   do { volatile unsigned long _junk;                             \
        CALL_FN_W_v(_junk,fnptr); } while (0)

#define CALL_FN_v_W(fnptr, arg1)                                  \
   do { volatile unsigned long _junk;                             \
        CALL_FN_W_W(_junk,fnptr,arg1); } while (0)

#define CALL_FN_v_WW(fnptr, arg1,arg2)                            \
   do { volatile unsigned long _junk;                             \
        CALL_FN_W_WW(_junk,fnptr,arg1,arg2); } while (0)

#define CALL_FN_v_WWW(fnptr, arg1,arg2,arg3)                      \
   do { volatile unsigned long _junk;                             \
        CALL_FN_W_WWW(_junk,fnptr,arg1,arg2,arg3); } while (0)

#define CALL_FN_v_WWWW(fnptr, arg1,arg2,arg3,arg4)                \
   do { volatile unsigned long _junk;                             \
        CALL_FN_W_WWWW(_junk,fnptr,arg1,arg2,arg3,arg4); } while (0)

#define CALL_FN_v_5W(fnptr, arg1,arg2,arg3,arg4,arg5)             \
   do { volatile unsigned long _junk;                             \
        CALL_FN_W_5W(_junk,fnptr,arg1,arg2,arg3,arg4,arg5); } while (0)

#define CALL_FN_v_6W(fnptr, arg1,arg2,arg3,arg4,arg5,arg6)        \
   do { volatile unsigned long _junk;                             \
        CALL_FN_W_6W(_junk,fnptr,arg1,arg2,arg3,arg4,arg5,arg6); } while (0)

#define CALL_FN_v_7W(fnptr, arg1,arg2,arg3,arg4,arg5,arg6,arg7)   \
   do { volatile unsigned long _junk;                             \
        CALL_FN_W_7W(_junk,fnptr,arg1,arg2,arg3,arg4,arg5,arg6,arg7); } while (0)

/*
   x86架构（linux、darwin、solaris）下的特定宏定义。
   这些宏定义了被隐藏调用时会被破坏的寄存器列表。其中，由于gcc已经知道eax寄存器的用途，故而不需要在列表中显式声明。
*/

/*
   宏定义：保存和对齐堆栈，以便在进行函数调用之前和之后进行堆栈恢复。
   这些宏的作用是确保在调用其他函数时，堆栈指针得到正确的对齐，因为gcc在不知道函数调用时，可能不会保持堆栈的正确对齐。
*/
#define VALGRIND_ALIGN_STACK               \
      "movl %%esp,%%edi\n\t"               \
      "andl $0xfffffff0,%%esp\n\t"
#define VALGRIND_RESTORE_STACK             \
      "movl %%edi,%%esp\n\t"

/* These CALL_FN_ macros assume that on x86-linux, sizeof(unsigned
   long) == 4. */

// 定义宏 CALL_FN_W_v，用于调用无返回值函数
#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[1];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

// 定义宏 CALL_FN_W_W，用于调用有一个参数且有返回值的函数
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[2];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $12, %%esp\n\t"                                    \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_WW(lval, orig, arg1,arg2)                       \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个指向 OrigFn 的变量 _orig，将传入的 orig 赋值给它
      volatile unsigned long _argvec[3];                          \
      // 声明一个包含3个 unsigned long 类型元素的数组 _argvec
      volatile unsigned long _res;                                \
      // 声明一个 volatile unsigned long 类型的变量 _res，用于保存函数调用结果
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将 _orig.nraddr 转换为 unsigned long 类型后存入 _argvec[0]
      _argvec[1] = (unsigned long)(arg1);                         \
      // 将 arg1 转换为 unsigned long 类型后存入 _argvec[1]
      _argvec[2] = (unsigned long)(arg2);                         \
      // 将 arg2 转换为 unsigned long 类型后存入 _argvec[2]
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $8, %%esp\n\t"                                     \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      // 使用内联汇编进行函数调用，具体操作包括栈对齐、参数推入栈、寄存器设置等
      lval = (__typeof__(lval)) _res;                             \
      // 将函数调用的返回值转换为 lval 的类型，并赋值给 lval
   } while (0)
   // do-while 循环，整体为一个宏定义的结构，确保宏定义在使用时像一个单独的语句使用
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      // 声明一个 volatile 指针 _orig 指向传入的函数指针 orig
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个 volatile 数组 _argvec，用于存储函数调用时的参数
      volatile unsigned long _argvec[4];                          \
      // 声明一个 volatile 无符号长整型变量 _res，用于存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数指针的地址和参数依次存入 _argvec 数组
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      // 使用内联汇编调用函数
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $4, %%esp\n\t"                                     \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      // 将函数调用的返回值转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)

   lval: 函数调用返回值的目标变量
   orig: 原始函数指针
   arg1, arg2, arg3, arg4: 函数调用的参数

   以下是宏的具体实现，使用内联汇编来调用函数，适用于特定调试环境（如Valgrind）：

   1. 声明一个指向原始函数的指针 _orig，并使用传入的 orig 初始化。
   2. 声明一个长度为 5 的无符号长整型数组 _argvec，用于存储函数参数和函数指针地址。
   3. 声明一个无符号长整型变量 _res，用于存储函数调用的返回值。
   4. 将原始函数指针地址存入 _argvec[0]。
   5. 将传入的参数依次存入 _argvec[1] 到 _argvec[4]。
   6. 使用内联汇编执行函数调用：
      - 将 _argvec[4] 到 _argvec[1] 压入栈中，以便传递给函数。
      - 将 _argvec[0] 中的地址加载到 %eax 寄存器中，作为函数调用的目标地址。
      - 执行函数调用，Valgrind 相关的宏定义用于调整堆栈和处理。
      - 输出约束为 "=a" (_res)，将函数返回值存入 _res。
      - 输入约束为 "a" (&_argvec[0])，将 _argvec 数组的地址作为函数参数传递。
      - 破坏约束包括 "cc"（条件码寄存器）、"memory"（内存）、__CALLER_SAVED_REGS（调用者保存的寄存器）、"edi"（特定寄存器）。

   7. 将 _res 转换为 lval 的类型，并赋值给 lval。

   注：这段代码假定了宏定义中的一些外部约定和环境（如 Valgrind 调试工具），可能不适用于所有环境。
*/
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)           \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[5];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   使用宏定义实现调用具有5个参数的函数。

   @param lval 用于接收函数返回值的变量
   @param orig 原始函数指针
   @param arg1 第一个函数参数
   @param arg2 第二个函数参数
   @param arg3 第三个函数参数
   @param arg4 第四个函数参数
   @param arg5 第五个函数参数

   注：
   - 使用 volatile 修饰 _orig, _argvec 数组和 _res 变量，确保编译器不会优化掉这些变量的存取操作。
   - 将 _orig.nraddr 赋给 _argvec[0]，将参数依次赋给 _argvec 数组的后续元素。
   - 使用内联汇编嵌入 VALGRIND_ALIGN_STACK, pushl 指令将参数依次压入堆栈，然后将目标函数地址加载到 %eax 寄存器。
   - 调用目标函数，结果存储在 _res 变量中。
   - 将 _res 赋给 lval，并使用 __typeof__ 确保类型匹配。
   - 使用 do { ... } while (0) 结构确保宏定义可以安全地用作语句。
*/
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[6];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $12, %%esp\n\t"                                    \
         "pushl 20(%%eax)\n\t"                                    \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_6W

   参数：
   lval     - 返回值的左值
   orig     - 原始函数指针
   arg1-arg6- 函数调用的六个参数

   功能：
   这个宏用于调用一个具有六个参数的函数，并获取其返回值。

   细节说明：
   - 使用 volatile 声明了 _orig、_argvec 和 _res，确保编译器不会对它们进行优化。
   - 将原始函数指针 _orig 赋给 _argvec 数组的第一个元素。
   - 将六个参数依次存入 _argvec 数组的后续位置。
   - 使用内联汇编语句执行函数调用：
     - 执行堆栈对齐操作。
     - 将参数依次压入堆栈。
     - 将原始函数指针加载到 %eax 寄存器。
     - 执行函数调用并保存返回值到 _res 中。
     - 恢复堆栈。
   - 使用约束（constraints）指定了输入输出的寄存器和内存位置，以及需要保存的寄存器列表。

   注意事项：
   - 使用了 do {...} while (0) 结构，确保宏在展开时能够正常工作。
   - 宏中的注释和代码行都符合 C 语言的语法和语义要求。
*/
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[7];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $8, %%esp\n\t"                                     \
         "pushl 24(%%eax)\n\t"                                    \
         "pushl 20(%%eax)\n\t"                                    \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义: CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,arg7)
   参数说明:
   - lval: 函数调用结果的返回值
   - orig: 原始函数指针
   - arg1...arg7: 函数调用的7个参数

   功能：
   - 将原始函数指针 _orig 和参数列表 arg1 到 arg7 组装成一个函数调用的参数数组 _argvec
   - 使用内联汇编执行函数调用，将参数依次压入堆栈，设置 %eax 寄存器为目标函数地址并执行
   - 恢复堆栈状态，并将调用结果赋给 lval 变量

   注意：
   - 使用了 volatile 修饰符来防止编译器优化相关变量的读写操作
   - VALGRIND_ALIGN_STACK 和 VALGRIND_CALL_NOREDIR_EAX 是预定义的宏，用于在调试时对齐堆栈和调用
   - 操作涉及底层寄存器和内存，需要小心处理以避免潜在的副作用
*/
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7)                            \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[8];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $4, %%esp\n\t"                                     \
         "pushl 28(%%eax)\n\t"                                    \
         "pushl 24(%%eax)\n\t"                                    \
         "pushl 20(%%eax)\n\t"                                    \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   定义一个宏，用于调用一个具有8个参数的函数，并将结果存储在给定的左值变量中
   使用宏的注意事项：
   - _orig: 原始函数指针，用于保存函数地址
   - _argvec: 一个包含9个元素的无符号长整型数组，用于存储函数调用的参数
   - _res: 用于存储函数调用的返回值
   - 汇编部分的说明：
     - 将参数依次推入栈中，这些参数从寄存器 %eax 中取出
     - 将目标函数地址加载到寄存器 %eax 中
     - 进行函数调用并保存返回值到 _res 中
   - 输出（Output）部分：
     - "=a" (_res): 将 %eax 中的值作为输出结果存储在 _res 中
   - 输入（Input）部分：
     - "a" (&_argvec[0]): 将 _argvec 数组的地址作为输入传递给汇编代码
   - 破坏（Clobber）部分：
     - "cc", "memory", __CALLER_SAVED_REGS, "edi": 告知编译器这些寄存器在宏执行过程中可能会被修改
*/
#define CALL_FN_W_8W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8)                       \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[9];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "pushl 32(%%eax)\n\t"                                    \
         "pushl 28(%%eax)\n\t"                                    \
         "pushl 24(%%eax)\n\t"                                    \
         "pushl 20(%%eax)\n\t"                                    \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
// 定义一个宏，用于调用具有9个参数的函数
#define CALL_FN_W_9W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8,arg9)                  \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[10];                         \
      volatile unsigned long _res;                                \
      // 将函数指针存储在 _orig 中
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 存储每个参数的值到 _argvec 数组中
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      _argvec[9] = (unsigned long)(arg9);                         \
      // 内联汇编调用，将参数加载到寄存器并调用函数
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $12, %%esp\n\t"                                    \
         "pushl 36(%%eax)\n\t"                                    \
         "pushl 32(%%eax)\n\t"                                    \
         "pushl 28(%%eax)\n\t"                                    \
         "pushl 24(%%eax)\n\t"                                    \
         "pushl 20(%%eax)\n\t"                                    \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      // 将函数返回值转换为目标类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 定义一个宏，用于以指定的方式调用一个函数，传递十个参数
#define CALL_FN_W_10W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,  \
                                  arg7,arg8,arg9,arg10)           \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[11];                         \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      _argvec[9] = (unsigned long)(arg9);                         \
      _argvec[10] = (unsigned long)(arg10);                       \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $8, %%esp\n\t"                                     \
         "pushl 40(%%eax)\n\t"                                    \
         "pushl 36(%%eax)\n\t"                                    \
         "pushl 32(%%eax)\n\t"                                    \
         "pushl 28(%%eax)\n\t"                                    \
         "pushl 24(%%eax)\n\t"                                    \
         "pushl 20(%%eax)\n\t"                                    \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_11W

   参数：
      lval: 函数调用结果存储位置的变量
      orig: 原始函数指针
      arg1-arg11: 函数调用的11个参数

   功能：
      - 声明 _orig 变量，存储 orig 参数
      - 声明 _argvec 数组，存储函数调用参数
      - 声明 _res 变量，存储函数调用结果
      - 将 orig.nraddr 存入 _argvec[0]
      - 将 arg1-arg11 依次存入 _argvec[1] 到 _argvec[11]
      - 使用内联汇编执行以下操作：
        1. 对栈进行对齐操作
        2. 将 _argvec 中的参数依次压入栈中
        3. 将 _orig.nraddr（目标函数地址）放入 %eax 寄存器
        4. 使用 VALGRIND 相关的宏进行调用
        5. 恢复栈的状态
      - 输出结果存入 _res 中，并赋值给 lval

   注意：
      - __asm__ volatile 是 GCC 中的内联汇编语法
      - "cc", "memory", __CALLER_SAVED_REGS, "edi" 是被修改或使用的寄存器和内存列表
*/
#define CALL_FN_W_11W(lval, orig, arg1,arg2,arg3,arg4,arg5,       \
                                  arg6,arg7,arg8,arg9,arg10,      \
                                  arg11)                          \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[12];                         \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      _argvec[9] = (unsigned long)(arg9);                         \
      _argvec[10] = (unsigned long)(arg10);                       \
      _argvec[11] = (unsigned long)(arg11);                       \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "subl $4, %%esp\n\t"                                     \
         "pushl 44(%%eax)\n\t"                                    \
         "pushl 40(%%eax)\n\t"                                    \
         "pushl 36(%%eax)\n\t"                                    \
         "pushl 32(%%eax)\n\t"                                    \
         "pushl 28(%%eax)\n\t"                                    \
         "pushl 24(%%eax)\n\t"                                    \
         "pushl 20(%%eax)\n\t"                                    \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_12W(lval, orig, arg1,arg2,arg3,arg4,arg5,       \
                                  arg6,arg7,arg8,arg9,arg10,      \
                                  arg11,arg12)                    \
   do {                                                           \
      // 声明一个指向原始函数的指针，类型为 OrigFn
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个包含13个无符号长整型元素的数组 _argvec，用于存储参数
      volatile unsigned long _argvec[13];                         \
      // 声明一个无符号长整型变量 _res，用于存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将原始函数的地址存入参数数组 _argvec 的第一个元素
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将每个参数转换为无符号长整型，并依次存入 _argvec 数组
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      _argvec[9] = (unsigned long)(arg9);                         \
      _argvec[10] = (unsigned long)(arg10);                       \
      _argvec[11] = (unsigned long)(arg11);                       \
      _argvec[12] = (unsigned long)(arg12);                       \
      // 使用内联汇编语句进行函数调用
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "pushl 48(%%eax)\n\t"                                    \
         "pushl 44(%%eax)\n\t"                                    \
         "pushl 40(%%eax)\n\t"                                    \
         "pushl 36(%%eax)\n\t"                                    \
         "pushl 32(%%eax)\n\t"                                    \
         "pushl 28(%%eax)\n\t"                                    \
         "pushl 24(%%eax)\n\t"                                    \
         "pushl 20(%%eax)\n\t"                                    \
         "pushl 16(%%eax)\n\t"                                    \
         "pushl 12(%%eax)\n\t"                                    \
         "pushl 8(%%eax)\n\t"                                     \
         "pushl 4(%%eax)\n\t"                                     \
         "movl (%%eax), %%eax\n\t"  /* target->%eax */            \
         VALGRIND_CALL_NOREDIR_EAX                                \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=a" (_res)                                  \
         : /*in*/    "a" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "edi"   \
      );                                                          \
      // 将函数调用的返回值转换为目标类型 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

#endif /* PLAT_x86_linux || PLAT_x86_darwin || PLAT_x86_solaris */


注释已按要求添加到每行代码中。
/* ---------------- amd64-{linux,darwin,solaris} --------------- */

#if defined(PLAT_amd64_linux)  ||  defined(PLAT_amd64_darwin) \
    ||  defined(PLAT_amd64_solaris)
/* 定义平台为amd64架构下的Linux、Darwin或Solaris操作系统 */

/* ARGREGS: rdi rsi rdx rcx r8 r9 (the rest on stack in R-to-L order) */
/* 定义寄存器参数顺序，按照从右到左的顺序分别是rdi、rsi、rdx、rcx、r8、r9 */

/* These regs are trashed by the hidden call. */
/* 下列寄存器在隐式调用中会被破坏 */
#define __CALLER_SAVED_REGS /*"rax",*/ "rcx", "rdx", "rsi",       \
                            "rdi", "r8", "r9", "r10", "r11"
/* 定义被调用者保存的寄存器，其中注释掉的"rax"寄存器被省略 */
#endif
/*
   这段代码注释解释了一系列宏的复杂性和其在确保堆栈展开工作可靠性方面的作用。详细讨论了一个bug，编号为243270。
   下面的所有宏都涉及在%rsp上添加或减去128。如果gcc认为CFA在%rsp上，那么展开可能会失败，因为CFA处的内容与gcc在实例化宏时构造CFI时的“预期”不符。

   但我们不能只是添加一个CFI注释来增加CFA偏移量，以匹配对%rsp的128的减法，因为我们不知道gcc是否已经选择了%rsp作为CFA，或者是否选择了其他寄存器（例如%rbp）。在后一种情况下，添加一个CFI注释来更改CFA偏移量是错误的。

   因此，解决方案是使用__builtin_dwarf_cfa()获取CFA，将其放入已知的寄存器中，并添加一个CFI注释来说明寄存器是什么。这里选择%rbp寄存器，因为：

   (1) %rbp已经在展开中使用。如果选择了一个新的寄存器，那么展开器将不得不在所有堆栈跟踪中展开它，这是昂贵的。

   (2) %rbp已经在JIT中用于精确的异常更新。如果选择了一个新的寄存器，那么我们也必须为其实现精确的异常处理，这会降低生成代码的性能。

   然而，这里还有一个额外的复杂性。我们不能简单地将__builtin_dwarf_cfa()的结果放入%rbp中，然后将%rbp添加到内联汇编片段末尾的破坏寄存器列表中；gcc不允许%rbp出现在那个列表中。因此，我们需要在asm的持续时间内将%rbp存储在%r15中，并声明%r15已被破坏。gcc似乎对这样做感到满意。

   这段代码需要条件化处理，以便在不支持__builtin_dwarf_cfa的旧gcc下不改变。此外，由于这个头文件是独立的，不依赖于config.h，因此以下条件化处理不能依赖于配置时检查。

   虽然从'defined(__GNUC__) && defined(__GCC_HAVE_DWARF2_CFI_ASM)'来看，这个表达式排除了Darwin。
   Darwin中的汇编中的.cfi指令似乎完全不同，我还没有调查它们的工作原理。

   更有趣的是，我们必须使用完全未记录的__builtin_dwarf_cfa()，它似乎真正计算CFA，而__builtin_frame_address(0)声称这样做，但实际上并不是。参见https://bugs.kde.org/show_bug.cgi?id=243270#c47
*/
#if defined(__GNUC__) && defined(__GCC_HAVE_DWARF2_CFI_ASM)
#  define __FRAME_POINTER                                         \
      ,"r"(__builtin_dwarf_cfa())
/* 定义 VALGRIND_CFI_PROLOGUE 宏，用于设置函数调用前的栈帧信息：
   - 将当前的 rbp 寄存器保存到 r15 寄存器中
   - 将传入的第三个参数设置为当前的 rbp 寄存器的值
   - 使用 .cfi_remember_state 指令保存当前的调用帧状态
   - 使用 .cfi_def_cfa 指令将 rbp 寄存器设为栈指针，偏移为 0 */
#define VALGRIND_CFI_PROLOGUE                                   \
      "movq %%rbp, %%r15\n\t"                                     \
      "movq %2, %%rbp\n\t"                                        \
      ".cfi_remember_state\n\t"                                   \
      ".cfi_def_cfa rbp, 0\n\t"

/* 定义 VALGRIND_CFI_EPILOGUE 宏，用于恢复函数调用后的栈帧信息：
   - 将之前保存的 r15 寄存器的值恢复到 rbp 寄存器中
   - 使用 .cfi_restore_state 恢复之前保存的调用帧状态 */
#define VALGRIND_CFI_EPILOGUE                                   \
      "movq %%r15, %%rbp\n\t"                                     \
      ".cfi_restore_state\n\t"

/* 如果不是在 Valgrind 环境下，则定义空的宏 */
#else
/* 定义空的 __FRAME_POINTER 宏 */
#define __FRAME_POINTER
/* 定义空的 VALGRIND_CFI_PROLOGUE 和 VALGRIND_CFI_EPILOGUE 宏 */
#define VALGRIND_CFI_PROLOGUE
#define VALGRIND_CFI_EPILOGUE
#endif

/* 定义 VALGRIND_ALIGN_STACK 宏，用于调整栈的对齐：
   - 将当前的栈指针 rsp 寄存器的值保存到 r14 寄存器中
   - 使用 andq 指令将 rsp 寄存器的值按照 16 字节对齐 */
#define VALGRIND_ALIGN_STACK               \
      "movq %%rsp,%%r14\n\t"               \
      "andq $0xfffffffffffffff0,%%rsp\n\t"

/* 定义 VALGRIND_RESTORE_STACK 宏，用于恢复栈的状态：
   - 将之前保存的 r14 寄存器中的值恢复到 rsp 寄存器中 */
#define VALGRIND_RESTORE_STACK             \
      "movq %%r14,%%rsp\n\t"

/* 这些 CALL_FN_ 宏假设在 amd64-linux 下，sizeof(unsigned long) == 8。 */

/* 注意：在所有的 CALL_FN_ 宏中，存在一个不好的应急解决方案。为了避免破坏栈的保护区域，
   我们需要在隐藏调用之前将 %rsp 减少 128，然后在调用之后恢复。
   这个应急解决方案的问题在于，仅仅靠运气，栈在隐藏调用期间仍然可以展开，因此可能与 CFI 数据的行为不匹配。
   为什么这很重要？想象一个包装器有一个在栈上分配的局部变量，并将其指针传递给隐藏调用。
   因为 gcc 不知道隐藏调用，它可能会将该局部变量分配在保护区域内。不幸的是，隐藏调用可能会在使用之前破坏它。
   因此，我们必须在隐藏调用的持续时间内清除保护区域，以确保安全。
   可能同样的问题也会影响其他保护区域风格的 ABI（如 ppc64-linux），但是对于这些 ABI，栈是自描述的（没有这种 CFI 的麻烦），
   所以至少 messing with the stack pointer 不会导致栈无法展开的危险。 */
#define CALL_FN_W_v(lval, orig)                                        \
   do {                                                                \
      // 声明一个 volatile 类型的指针 _orig，指向传入的 orig
      volatile OrigFn        _orig = (orig);                           \
      // 声明一个 volatile 类型的 unsigned long 数组 _argvec，长度为 1
      volatile unsigned long _argvec[1];                               \
      // 声明一个 volatile 类型的 unsigned long 变量 _res，用于存放函数调用结果
      volatile unsigned long _res;                                     \
      // 将 _orig.nraddr 转换为 unsigned long 存入 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      // 使用内联汇编语句执行函数调用
      __asm__ volatile(                                                \
         // 插入 Valgrind 的 CFI prologue 指令
         VALGRIND_CFI_PROLOGUE                                         \
         // 插入 Valgrind 的栈对齐指令
         VALGRIND_ALIGN_STACK                                          \
         // 分配额外的栈空间
         "subq $128,%%rsp\n\t"                                         \
         // 将目标地址中的内容加载到 %rax 寄存器（target->%rax）
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         // 插入 Valgrind 的无重定向 RAX 调用指令
         VALGRIND_CALL_NOREDIR_RAX                                     \
         // 恢复栈空间
         VALGRIND_RESTORE_STACK                                        \
         // 插入 Valgrind 的 CFI epilogue 指令
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \
      );                                                               \
      // 将 _res 强制类型转换为 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) _res;                                  \
   } while (0)

#define CALL_FN_W_W(lval, orig, arg1)                                  \
   do {                                                                \
      // 声明一个 volatile 类型的指针 _orig，指向传入的 orig
      volatile OrigFn        _orig = (orig);                           \
      // 声明一个 volatile 类型的 unsigned long 数组 _argvec，长度为 2
      volatile unsigned long _argvec[2];                               \
      // 声明一个 volatile 类型的 unsigned long 变量 _res，用于存放函数调用结果
      volatile unsigned long _res;                                     \
      // 将 _orig.nraddr 转换为 unsigned long 存入 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      // 将 arg1 转换为 unsigned long 存入 _argvec[1]
      _argvec[1] = (unsigned long)(arg1);                              \
      // 使用内联汇编语句执行函数调用
      __asm__ volatile(                                                \
         // 插入 Valgrind 的 CFI prologue 指令
         VALGRIND_CFI_PROLOGUE                                         \
         // 插入 Valgrind 的栈对齐指令
         VALGRIND_ALIGN_STACK                                          \
         // 分配额外的栈空间
         "subq $128,%%rsp\n\t"                                         \
         // 将目标地址中偏移 8 字节的内容加载到 %rdi 寄存器
         "movq 8(%%rax), %%rdi\n\t"                                    \
         // 将目标地址中的内容加载到 %rax 寄存器（target->%rax）
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         // 插入 Valgrind 的无重定向 RAX 调用指令
         VALGRIND_CALL_NOREDIR_RAX                                     \
         // 恢复栈空间
         VALGRIND_RESTORE_STACK                                        \
         // 插入 Valgrind 的 CFI epilogue 指令
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \
      );                                                               \
      // 将 _res 强制类型转换为 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) _res;                                  \
   } while (0)
/*
   宏定义：CALL_FN_W_WW(lval, orig, arg1, arg2)
   参数说明：
     - lval: 返回值变量
     - orig: 原始函数指针
     - arg1: 第一个参数
     - arg2: 第二个参数

   实现说明：
     - 使用 volatile 关键字声明 _orig, _argvec[3], _res 变量，确保编译器不会优化它们的读写操作。
     - 将 _orig 初始化为 orig 参数（即原始函数指针）。
     - 将 _argvec 数组的前三个元素分别设置为 _orig.nraddr、arg1、arg2 的值。
     - 使用内联汇编 __asm__ volatile 插入汇编代码块，实现函数调用。
       具体汇编内容包括：
       - 执行 VALGRIND_CFI_PROLOGUE 和 VALGRIND_ALIGN_STACK 操作。
       - 分别将 _orig.nraddr、arg1、arg2 的值存入寄存器 %rdi、%rsi、%rax。
       - 使用 VALGRIND_CALL_NOREDIR_RAX 进行函数调用，目标函数地址存放在 %rax 寄存器。
       - 执行 VALGRIND_RESTORE_STACK 和 VALGRIND_CFI_EPILOGUE 操作。
     - 指定汇编代码中的输入、输出、以及可能被修改的寄存器和内存位置。

   注意事项：
     - 宏定义的目的是在不同的编译环境中调用原始函数。
     - 汇编代码通过操作寄存器和栈来传递参数和执行函数调用，需要确保操作的正确性和安全性。
     - 使用 volatile 声明变量可以避免编译器优化对这些变量的操作。

   参考：类似的宏定义常见于需要跨平台、跨编译器调用函数的场景。
*/
#define CALL_FN_W_WW(lval, orig, arg1,arg2)                            \
   do {                                                                \
      volatile OrigFn        _orig = (orig);                           \
      volatile unsigned long _argvec[3];                               \
      volatile unsigned long _res;                                     \
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         "subq $128,%%rsp\n\t"                                         \
         "movq 16(%%rax), %%rsi\n\t"                                   \
         "movq 8(%%rax), %%rdi\n\t"                                    \
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         VALGRIND_CALL_NOREDIR_RAX                                     \
         VALGRIND_RESTORE_STACK                                        \
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \
      );                                                               \
      lval = (__typeof__(lval)) _res;                                  \
   } while (0)
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                      \
   do {                                                                \
      volatile OrigFn        _orig = (orig);                           \
      volatile unsigned long _argvec[4];                               \
      volatile unsigned long _res;                                     \
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      _argvec[3] = (unsigned long)(arg3);                              \
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         "subq $128,%%rsp\n\t"                                         \
         "movq 24(%%rax), %%rdx\n\t"                                   \
         "movq 16(%%rax), %%rsi\n\t"                                   \
         "movq 8(%%rax), %%rdi\n\t"                                    \
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         VALGRIND_CALL_NOREDIR_RAX                                     \
         VALGRIND_RESTORE_STACK                                        \
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \  // 输出寄存器 %rax 作为 _res
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \  // 输入寄存器 %rax 作为 _argvec[0]，使用帧指针
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \  // 声明不使用寄存器 cc，内存和指定的寄存器列表
      );                                                               \
      lval = (__typeof__(lval)) _res;                                  \  // 将 _res 赋给 lval，根据 lval 的类型进行转换
   } while (0)
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)                \
   do {                                                                \
      volatile OrigFn        _orig = (orig);                           \
      volatile unsigned long _argvec[5];                               \
      volatile unsigned long _res;                                     \
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      _argvec[3] = (unsigned long)(arg3);                              \
      _argvec[4] = (unsigned long)(arg4);                              \
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         "subq $128,%%rsp\n\t"                                         \
         "movq 32(%%rax), %%rcx\n\t"                                   \   // 将地址偏移为32字节的值加载到寄存器 %rcx
         "movq 24(%%rax), %%rdx\n\t"                                   \   // 将地址偏移为24字节的值加载到寄存器 %rdx
         "movq 16(%%rax), %%rsi\n\t"                                   \   // 将地址偏移为16字节的值加载到寄存器 %rsi
         "movq 8(%%rax), %%rdi\n\t"                                    \   // 将地址偏移为8字节的值加载到寄存器 %rdi
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \   // 将地址偏移为0字节的值加载到寄存器 %rax，目标为 %rax
         VALGRIND_CALL_NOREDIR_RAX                                     \   // 使用 %rax 寄存器中的地址调用函数，不重定向
         VALGRIND_RESTORE_STACK                                        \   // 恢复栈指针
         VALGRIND_CFI_EPILOGUE                                         \   // CFI (Call Frame Information) 函数退出时的指令序列
         : /*out*/   "=a" (_res)                                       \   // 输出约束：返回值放在 %rax 中，并赋给 _res
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \   // 输入约束：参数数组的地址放在 %rax 中，并且使用 __FRAME_POINTER
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \   // 被破坏的寄存器和内存，以及其他约束
      );                                                               \
      lval = (__typeof__(lval)) _res;                                  \   // 将 _res 转换为 lval 的类型并赋值给 lval
   } while (0)
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)             \
   do {                                                                \
      volatile OrigFn        _orig = (orig);                           \
      volatile unsigned long _argvec[6];                               \
      volatile unsigned long _res;                                     \
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      _argvec[3] = (unsigned long)(arg3);                              \
      _argvec[4] = (unsigned long)(arg4);                              \
      _argvec[5] = (unsigned long)(arg5);                              \
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         "subq $128,%%rsp\n\t"                                         \
         "movq 40(%%rax), %%r8\n\t"                                    \  // 将 _argvec 中的第五个参数加载到寄存器 %r8
         "movq 32(%%rax), %%rcx\n\t"                                   \  // 将 _argvec 中的第四个参数加载到寄存器 %rcx
         "movq 24(%%rax), %%rdx\n\t"                                   \  // 将 _argvec 中的第三个参数加载到寄存器 %rdx
         "movq 16(%%rax), %%rsi\n\t"                                   \  // 将 _argvec 中的第二个参数加载到寄存器 %rsi
         "movq 8(%%rax), %%rdi\n\t"                                    \  // 将 _argvec 中的第一个参数加载到寄存器 %rdi
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \  // 将 _argvec 中的函数地址加载到寄存器 %rax，这是函数调用的目标地址
         VALGRIND_CALL_NOREDIR_RAX                                     \  // 使用 %rax 寄存器进行函数调用
         VALGRIND_RESTORE_STACK                                        \  // 恢复栈指针
         VALGRIND_CFI_EPILOGUE                                         \  // Valgrind CFI 结束
         : /*out*/   "=a" (_res)                                       \  // 输出操作数：将函数返回值放入 _res
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \  // 输入操作数：使用 &_argvec[0] 作为参数地址，启用帧指针
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \  // 被修改的寄存器和内存，以及不需要保存的寄存器
      );                                                               \
      lval = (__typeof__(lval)) _res;                                  \  // 将 _res 转换为 lval 的类型并赋值给 lval
   } while (0)
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)        \
   do {                                                                \
      // 定义一个指向原始函数的变量 _orig，并将传入的 orig 参数赋值给它
      volatile OrigFn        _orig = (orig);                           \
      // 定义一个长度为 7 的无符号长整型数组 _argvec，用于存储函数调用的参数
      volatile unsigned long _argvec[7];                               \
      // 定义一个无符号长整型变量 _res，用于存储函数调用的返回值
      volatile unsigned long _res;                                     \
      // 将函数指针的地址存入 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      // 将参数 arg1 到 arg6 依次存入 _argvec 数组中
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      _argvec[3] = (unsigned long)(arg3);                              \
      _argvec[4] = (unsigned long)(arg4);                              \
      _argvec[5] = (unsigned long)(arg5);                              \
      _argvec[6] = (unsigned long)(arg6);                              \
      // 使用内联汇编进行函数调用，传递参数并获取返回值
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         // 在栈上分配空间，以便进行函数调用
         "subq $128,%%rsp\n\t"                                         \
         // 将参数从 _argvec 数组加载到寄存器中
         "movq 48(%%rax), %%r9\n\t"                                    \
         "movq 40(%%rax), %%r8\n\t"                                    \
         "movq 32(%%rax), %%rcx\n\t"                                   \
         "movq 24(%%rax), %%rdx\n\t"                                   \
         "movq 16(%%rax), %%rsi\n\t"                                   \
         "movq 8(%%rax), %%rdi\n\t"                                    \
         // 将函数地址加载到 %rax 寄存器中，进行函数调用
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         VALGRIND_CALL_NOREDIR_RAX                                     \
         // 恢复栈的原始状态
         VALGRIND_RESTORE_STACK                                        \
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \
      );                                                               \
      // 将返回值转换为目标类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                                  \
   } while (0)
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,        \
                                 arg7)                                 \
   do {                                                                \
      // 声明 _orig 为 volatile 类型的 OrigFn，用于保存原始函数指针
      volatile OrigFn        _orig = (orig);                           \
      // 声明 _argvec 数组，用于保存函数调用的参数
      volatile unsigned long _argvec[8];                               \
      // 声明 _res 用于保存函数调用的返回值
      volatile unsigned long _res;                                     \
      // 将原始函数指针的地址存入 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      // 将参数 arg1 至 arg7 存入 _argvec 数组对应位置
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      _argvec[3] = (unsigned long)(arg3);                              \
      _argvec[4] = (unsigned long)(arg4);                              \
      _argvec[5] = (unsigned long)(arg5);                              \
      _argvec[6] = (unsigned long)(arg6);                              \
      _argvec[7] = (unsigned long)(arg7);                              \
      // 内联汇编，调用函数体
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         "subq $136,%%rsp\n\t"                                         \
         "pushq 56(%%rax)\n\t"                                         \
         "movq 48(%%rax), %%r9\n\t"                                    \
         "movq 40(%%rax), %%r8\n\t"                                    \
         "movq 32(%%rax), %%rcx\n\t"                                   \
         "movq 24(%%rax), %%rdx\n\t"                                   \
         "movq 16(%%rax), %%rsi\n\t"                                   \
         "movq 8(%%rax), %%rdi\n\t"                                    \
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         VALGRIND_CALL_NOREDIR_RAX                                     \
         VALGRIND_RESTORE_STACK                                        \
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \
      );                                                               \
      // 将函数调用返回值赋给 lval
      lval = (__typeof__(lval)) _res;                                  \
   } while (0)
/*
   宏定义：CALL_FN_W_8W

   该宏用于调用一个指定函数指针 _orig，并传递多达八个参数 arg1 到 arg8。
   参数：
     - lval: 用于接收函数调用结果的变量
     - orig: 原始函数指针
     - arg1 到 arg8: 函数调用的实际参数

   实现细节：
     - 声明 _orig 作为 volatile 类型的原始函数指针，指向参数 orig
     - 声明 _argvec[9] 作为 volatile 类型的无符号长整型数组，用于存储函数调用参数
     - 声明 _res 作为 volatile 类型的无符号长整型，用于存储函数调用结果
     - 将函数指针和参数值分别存入 _argvec 数组中
     - 使用内联汇编调用指定的函数指针 _orig，并传递参数
     - 使用 VALGRIND 相关宏处理栈和寄存器状态，确保调用过程中的正确性和调试信息

   注意事项：
     - 宏内部使用了 __asm__ volatile 来定义内联汇编代码块
     - 涉及 VALGRIND 相关宏来支持 Valgrind 工具的内存检查和调试信息记录
     - 函数调用结果通过 _res 返回给调用者，存储在 lval 中
     - 宏定义中的 do { ... } while (0) 结构用于确保宏在使用时正确展开，即使在条件语句或循环中也能正常工作
*/
#define CALL_FN_W_8W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,        \
                                 arg7,arg8)                            \
   do {                                                                \
      volatile OrigFn        _orig = (orig);                           \
      volatile unsigned long _argvec[9];                               \
      volatile unsigned long _res;                                     \
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      _argvec[3] = (unsigned long)(arg3);                              \
      _argvec[4] = (unsigned long)(arg4);                              \
      _argvec[5] = (unsigned long)(arg5);                              \
      _argvec[6] = (unsigned long)(arg6);                              \
      _argvec[7] = (unsigned long)(arg7);                              \
      _argvec[8] = (unsigned long)(arg8);                              \
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         "subq $128,%%rsp\n\t"                                         \
         "pushq 64(%%rax)\n\t"                                         \
         "pushq 56(%%rax)\n\t"                                         \
         "movq 48(%%rax), %%r9\n\t"                                    \
         "movq 40(%%rax), %%r8\n\t"                                    \
         "movq 32(%%rax), %%rcx\n\t"                                   \
         "movq 24(%%rax), %%rdx\n\t"                                   \
         "movq 16(%%rax), %%rsi\n\t"                                   \
         "movq 8(%%rax), %%rdi\n\t"                                    \
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         VALGRIND_CALL_NOREDIR_RAX                                     \
         VALGRIND_RESTORE_STACK                                        \
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \
      );                                                               \
      lval = (__typeof__(lval)) _res;                                  \
   } while (0)
/*
   宏定义：CALL_FN_W_9W

   参数说明：
   lval: 接收函数调用结果的变量
   orig: 原始函数指针

   arg1-arg9: 函数调用的9个参数

   功能：
   使用汇编内联代码调用一个函数，并将结果存储在lval中。

   详细说明：
   - _orig: 将orig参数强制转换为OrigFn类型，并将其存储在_orig变量中。
   - _argvec: 一个长为10的无符号长整型数组，用于存储函数参数的地址。
   - _res: 无符号长整型变量，用于存储函数调用的返回值。
   - 使用汇编内联代码进行函数调用，具体步骤包括：
     - 设置栈帧以进行函数调用的准备。
     - 将寄存器中的参数值压入堆栈。
     - 设置目标函数的参数寄存器。
     - 调用目标函数，返回值存储在_res中。
     - 恢复堆栈状态。
   - 使用Valgrind工具的CFI保护、对齐堆栈等指令进行调用的前后处理。

   注意事项：
   - _orig.nraddr代表原始函数的地址。
   - __asm__ volatile用于告知编译器内联汇编代码的使用。
   - 代码中使用了Valgrind的指令进行内存和寄存器的保护和恢复。
   - lval最终将存储函数调用的返回值，类型为orig函数指针的返回类型。
*/
#define CALL_FN_W_9W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,        \
                                 arg7,arg8,arg9)                       \
   do {                                                                \
      volatile OrigFn        _orig = (orig);                           \
      volatile unsigned long _argvec[10];                              \
      volatile unsigned long _res;                                     \
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      _argvec[3] = (unsigned long)(arg3);                              \
      _argvec[4] = (unsigned long)(arg4);                              \
      _argvec[5] = (unsigned long)(arg5);                              \
      _argvec[6] = (unsigned long)(arg6);                              \
      _argvec[7] = (unsigned long)(arg7);                              \
      _argvec[8] = (unsigned long)(arg8);                              \
      _argvec[9] = (unsigned long)(arg9);                              \
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         "subq $136,%%rsp\n\t"                                         \
         "pushq 72(%%rax)\n\t"                                         \
         "pushq 64(%%rax)\n\t"                                         \
         "pushq 56(%%rax)\n\t"                                         \
         "movq 48(%%rax), %%r9\n\t"                                    \
         "movq 40(%%rax), %%r8\n\t"                                    \
         "movq 32(%%rax), %%rcx\n\t"                                   \
         "movq 24(%%rax), %%rdx\n\t"                                   \
         "movq 16(%%rax), %%rsi\n\t"                                   \
         "movq 8(%%rax), %%rdi\n\t"                                    \
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         VALGRIND_CALL_NOREDIR_RAX                                     \
         VALGRIND_RESTORE_STACK                                        \
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \
      );                                                               \
      lval = (__typeof__(lval)) _res;                                  \
   } while (0)
#define CALL_FN_W_10W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,       \
                                  arg7,arg8,arg9,arg10)                \
   do {                                                                \
      // 定义一个指向原始函数的指针，使用 volatile 修饰以确保编译器不会优化掉它
      volatile OrigFn        _orig = (orig);                           \
      // 定义一个长度为 11 的无符号长整型数组，用于存储函数调用的参数
      volatile unsigned long _argvec[11];                              \
      // 定义一个无符号长整型变量用于存储函数调用的返回值
      volatile unsigned long _res;                                     \
      // 将原始函数的地址存入参数数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                        \
      // 将传入的第 1 到第 10 个参数分别存入参数数组的对应位置
      _argvec[1] = (unsigned long)(arg1);                              \
      _argvec[2] = (unsigned long)(arg2);                              \
      _argvec[3] = (unsigned long)(arg3);                              \
      _argvec[4] = (unsigned long)(arg4);                              \
      _argvec[5] = (unsigned long)(arg5);                              \
      _argvec[6] = (unsigned long)(arg6);                              \
      _argvec[7] = (unsigned long)(arg7);                              \
      _argvec[8] = (unsigned long)(arg8);                              \
      _argvec[9] = (unsigned long)(arg9);                              \
      _argvec[10] = (unsigned long)(arg10);                            \
      // 使用内联汇编调用函数，使用 VALGRIND 相关宏来确保正确的内存访问
      __asm__ volatile(                                                \
         VALGRIND_CFI_PROLOGUE                                         \
         VALGRIND_ALIGN_STACK                                          \
         "subq $128,%%rsp\n\t"                                         \
         "pushq 80(%%rax)\n\t"                                         \
         "pushq 72(%%rax)\n\t"                                         \
         "pushq 64(%%rax)\n\t"                                         \
         "pushq 56(%%rax)\n\t"                                         \
         "movq 48(%%rax), %%r9\n\t"                                    \
         "movq 40(%%rax), %%r8\n\t"                                    \
         "movq 32(%%rax), %%rcx\n\t"                                   \
         "movq 24(%%rax), %%rdx\n\t"                                   \
         "movq 16(%%rax), %%rsi\n\t"                                   \
         "movq 8(%%rax), %%rdi\n\t"                                    \
         "movq (%%rax), %%rax\n\t"  /* target->%rax */                 \
         VALGRIND_CALL_NOREDIR_RAX                                     \
         VALGRIND_RESTORE_STACK                                        \
         VALGRIND_CFI_EPILOGUE                                         \
         : /*out*/   "=a" (_res)                                       \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER                 \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r14", "r15" \
      );                                                               \
      // 将函数返回值转换为目标类型并存入 lval 变量中
      lval = (__typeof__(lval)) _res;                                  \
   } while (0)
/* ------------------------ ppc32-linux ------------------------ */

#if defined(PLAT_ppc32_linux)

/* This is useful for finding out about the on-stack stuff:

   extern int f9  ( int,int,int,int,int,int,int,int,int );
   extern int f10 ( int,int,int,int,int,int,int,int,int,int );
   extern int f11 ( int,int,int,int,int,int,int,int,int,int,int );
   extern int f12 ( int,int,int,int,int,int,int,int,int,int,int,int );

   int g9 ( void ) {
      return f9(11,22,33,44,55,66,77,88,99);
   }
   int g10 ( void ) {
      return f10(11,22,33,44,55,66,77,88,99,110);
   }
   int g11 ( void ) {
      return f11(11,22,33,44,55,66,77,88,99,110,121);
   }
   int g12 ( void ) {
      return f12(11,22,33,44,55,66,77,88,99,110,121,132);
   }
*/

/* ARGREGS: r3 r4 r5 r6 r7 r8 r9 r10 (the rest on stack somewhere) */

/* These regs are trashed by the hidden call. */
#define __CALLER_SAVED_REGS                                       \
   "lr", "ctr", "xer",                                            \
   "cr0", "cr1", "cr2", "cr3", "cr4", "cr5", "cr6", "cr7",        \
   "r0", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10",   \
   "r11", "r12", "r13"

/* Macros to save and align the stack before making a function
   call and restore it afterwards as gcc may not keep the stack
   pointer aligned if it doesn't realise calls are being made
   to other functions. */

#define VALGRIND_ALIGN_STACK               \
      "mr 28,1\n\t"                        \
      "rlwinm 1,1,0,0,27\n\t"
#define VALGRIND_RESTORE_STACK             \
      "mr 1,28\n\t"

/* These CALL_FN_ macros assume that on ppc32-linux, 
   sizeof(unsigned long) == 4. */

#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[1];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 定义一个宏，用于调用带有两个参数的函数并返回结果
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      // 声明一个 volatile 类型的变量 _orig，并将传入的 orig 参数赋值给它
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个数组 _argvec，用于存放两个参数的地址
      volatile unsigned long _argvec[2];                          \
      // 声明一个变量 _res，用于存放函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数指针的地址赋给 _argvec 数组的第一个元素
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将第一个参数的值赋给 _argvec 数组的第二个元素
      _argvec[1] = (unsigned long)arg1;                           \
      // 使用内联汇编语句调用函数
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将函数返回值转换为宏定义的 lval 变量的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

# 定义一个宏，用于调用带有三个参数的函数并返回结果
#define CALL_FN_W_WW(lval, orig, arg1, arg2)                      \
   do {                                                           \
      // 声明一个 volatile 类型的变量 _orig，并将传入的 orig 参数赋值给它
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个数组 _argvec，用于存放三个参数的地址
      volatile unsigned long _argvec[3];                          \
      // 声明一个变量 _res，用于存放函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数指针的地址赋给 _argvec 数组的第一个元素
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将第一个参数的值赋给 _argvec 数组的第二个元素
      _argvec[1] = (unsigned long)arg1;                           \
      // 将第二个参数的值赋给 _argvec 数组的第三个元素
      _argvec[2] = (unsigned long)arg2;                           \
      // 使用内联汇编语句调用函数
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         "lwz 4,8(11)\n\t"   /* arg2->r4 */                       \
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将函数返回值转换为宏定义的 lval 变量的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[4];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)arg1;                           \
      _argvec[2] = (unsigned long)arg2;                           \
      _argvec[3] = (unsigned long)arg3;                           \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         "lwz 4,8(11)\n\t"   /* arg2->r4 */                       \
         "lwz 5,12(11)\n\t"  /* arg3->r5 */                       \
         "lwz 11,0(11)\n\t"  /* original function address->r11 */ \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：


# 宏定义：用于调用一个函数，函数原型和三个参数传入
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      // 定义一个 volatile 类型的 _orig 变量，存储传入的 orig 参数
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个 4 元素的 _argvec 数组，用于存储参数
      volatile unsigned long _argvec[4];                          \
      // 定义一个 volatile 类型的 _res 变量，存储函数调用返回值
      volatile unsigned long _res;                                \
      // 将函数指针 _orig 的地址存入 _argvec[0]，传递给汇编代码使用
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数 arg1 转换成 unsigned long 存入 _argvec[1]
      _argvec[1] = (unsigned long)arg1;                           \
      // 将参数 arg2 转换成 unsigned long 存入 _argvec[2]
      _argvec[2] = (unsigned long)arg2;                           \
      // 将参数 arg3 转换成 unsigned long 存入 _argvec[3]
      _argvec[3] = (unsigned long)arg3;                           \
      // 使用内联汇编语句执行以下操作：
      __asm__ volatile(                                           \
         // 对齐栈以符合要求
         VALGRIND_ALIGN_STACK                                     \
         // 将 _argvec[1] 存入寄存器 11
         "mr 11,%1\n\t"                                           \
         // 将 11(寄存器 11) 偏移 4 字节处的内容加载到寄存器 3 中，作为 arg1 的值
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         // 将 11(寄存器 11) 偏移 8 字节处的内容加载到寄存器 4 中，作为 arg2 的值
         "lwz 4,8(11)\n\t"   /* arg2->r4 */                       \
         // 将 11(寄存器 11) 偏移 12 字节处的内容加载到寄存器 5 中，作为 arg3 的值
         "lwz 5,12(11)\n\t"  /* arg3->r5 */                       \
         // 将 11(寄存器 11) 偏移 0 字节处的内容加载到寄存器 11 中，作为原始函数地址
         "lwz 11,0(11)\n\t"  /* original function address->r11 */ \
         // 根据 Valgrind 的宏定义，跳转并链接到不重定向的 r11
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         // 恢复栈以符合要求
         VALGRIND_RESTORE_STACK                                   \
         // 将寄存器 3 的值存入 _res，作为函数调用的返回值
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将 _res 转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)

   该宏用于调用一个函数，并且函数接受四个参数和一个原始函数指针。

   参数说明：
   - lval: 调用结果存储的变量
   - orig: 原始函数指针
   - arg1, arg2, arg3, arg4: 函数的四个参数

   实现细节：
   - 定义一个名为_orig的volatile变量，存储原始函数指针(orig)
   - 定义一个名为_argvec的volatile unsigned long数组，长度为5，存储参数和函数指针的地址
   - 定义一个名为_res的volatile unsigned long变量，存储函数调用的结果
   - 使用内联汇编(__asm__ volatile)来执行函数调用：
     * 将参数加载到寄存器中
     * 执行函数调用
     * 将函数返回值保存到_res变量中
   - 将_res的值转换为lval的类型，并赋给lval

   注：该宏使用了一些特定的宏定义，如VALGRIND_ALIGN_STACK、VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11等，
       这些宏定义可能用于特定的调试或者优化目的，但具体实现不在这段代码的注释范围内。
*/
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)           \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[5];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)arg1;                           \
      _argvec[2] = (unsigned long)arg2;                           \
      _argvec[3] = (unsigned long)arg3;                           \
      _argvec[4] = (unsigned long)arg4;                           \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         "lwz 4,8(11)\n\t"                                        \
         "lwz 5,12(11)\n\t"                                       \
         "lwz 6,16(11)\n\t"  /* arg4->r6 */                       \
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   定义一个宏，用于调用一个带有五个参数的函数，通过直接调用汇编语言实现。
   这个宏会将函数指针、参数值等数据封装成一个参数向量数组，并使用汇编语言进行函数调用和参数传递。

   参数说明：
   lval: 函数调用后的返回值存储位置
   orig: 原始函数指针
   arg1: 第一个参数
   arg2: 第二个参数
   arg3: 第三个参数
   arg4: 第四个参数
   arg5: 第五个参数
*/
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[6];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)arg1;                           \
      _argvec[2] = (unsigned long)arg2;                           \
      _argvec[3] = (unsigned long)arg3;                           \
      _argvec[4] = (unsigned long)arg4;                           \
      _argvec[5] = (unsigned long)arg5;                           \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         "lwz 4,8(11)\n\t"   /* arg2->r4 */                       \
         "lwz 5,12(11)\n\t"  /* arg3->r5 */                       \
         "lwz 6,16(11)\n\t"  /* arg4->r6 */                       \
         "lwz 7,20(11)\n\t"  /* arg5->r7 */                       \
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      // 定义一个指向原始函数的变量 _orig，并将传入的 orig 参数赋给它
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个长度为 7 的无符号长整型数组 _argvec，用于保存函数调用时的参数
      volatile unsigned long _argvec[7];                          \
      // 定义一个无符号长整型变量 _res，用于存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将 _orig.nraddr 转换为无符号长整型并存入 _argvec 数组的第一个元素
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将 arg1 转换为无符号长整型并存入 _argvec 数组的第二个元素
      _argvec[1] = (unsigned long)arg1;                           \
      // 将 arg2 转换为无符号长整型并存入 _argvec 数组的第三个元素
      _argvec[2] = (unsigned long)arg2;                           \
      // 将 arg3 转换为无符号长整型并存入 _argvec 数组的第四个元素
      _argvec[3] = (unsigned long)arg3;                           \
      // 将 arg4 转换为无符号长整型并存入 _argvec 数组的第五个元素
      _argvec[4] = (unsigned long)arg4;                           \
      // 将 arg5 转换为无符号长整型并存入 _argvec 数组的第六个元素
      _argvec[5] = (unsigned long)arg5;                           \
      // 将 arg6 转换为无符号长整型并存入 _argvec 数组的第七个元素
      _argvec[6] = (unsigned long)arg6;                           \
      // 使用内联汇编语句，执行函数调用操作
      __asm__ volatile(                                           \
         // 对齐栈
         VALGRIND_ALIGN_STACK                                     \
         // 将 _argvec 数组的第二个元素（arg1）加载到寄存器 3 中作为参数
         "mr 11,%1\n\t"                                           \
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         // 将 _argvec 数组的第三个元素（arg2）加载到寄存器 4 中作为参数
         "lwz 4,8(11)\n\t"                                        \
         // 将 _argvec 数组的第四个元素（arg3）加载到寄存器 5 中作为参数
         "lwz 5,12(11)\n\t"                                       \
         // 将 _argvec 数组的第五个元素（arg4）加载到寄存器 6 中作为参数
         "lwz 6,16(11)\n\t"  /* arg4->r6 */                       \
         // 将 _argvec 数组的第六个元素（arg5）加载到寄存器 7 中作为参数
         "lwz 7,20(11)\n\t"                                       \
         // 将 _argvec 数组的第七个元素（arg6）加载到寄存器 8 中作为参数
         "lwz 8,24(11)\n\t"                                       \
         // 将 _argvec 数组的第一个元素（_orig.nraddr）加载到寄存器 11 中作为目标地址
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         // 调用指令，跳转并链接到不重定向 R11 的目标地址
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         // 恢复栈
         VALGRIND_RESTORE_STACK                                   \
         // 将寄存器 3 的值（函数返回值）存入 _res 变量
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将 _res 强制转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义 CALL_FN_W_7W 是一个用于调用带有7个参数的函数的宏。
   lval: 接收函数调用返回值的变量
   orig: 原始函数指针
   arg1-arg7: 函数参数，分别对应第1到第7个参数

   使用 volatile 修饰 _orig, _argvec, _res 变量，以确保编译器不会优化它们。

   将 _orig.nraddr 赋值给 _argvec[0]，将 arg1-arg7 分别赋值给 _argvec[1] 到 _argvec[7]。

   使用内联汇编嵌入到代码中，通过 VALGRIND_ALIGN_STACK 和 VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11 等宏来处理栈对齐和分支跳转。
   具体步骤如下：
   - 将 _argvec[0] 的值加载到寄存器 11 (r11) 中
   - 分别将 _argvec[1] 到 _argvec[7] 的值加载到寄存器 3 到 9 中
   - 将目标地址加载到寄存器 11 中
   - 执行带有嵌入式汇编的函数调用
   - 将返回值保存到 _res 中，并赋给 lval

   在内联汇编部分，使用以下限定符和宏：
   - "=r" (_res): 输出约束，表示将返回值放在寄存器中 (_res)
   - "r" (&_argvec[0]): 输入约束，表示使用 _argvec 数组的地址作为输入数据
   - "cc", "memory", __CALLER_SAVED_REGS, "r28": 限定符，指示内联汇编所需的条件码、内存和寄存器保存

   最后使用 do { ... } while (0) 结构，确保宏展开后形成一个完整的语句块。
*/
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7)                            \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[8];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)arg1;                           \
      _argvec[2] = (unsigned long)arg2;                           \
      _argvec[3] = (unsigned long)arg3;                           \
      _argvec[4] = (unsigned long)arg4;                           \
      _argvec[5] = (unsigned long)arg5;                           \
      _argvec[6] = (unsigned long)arg6;                           \
      _argvec[7] = (unsigned long)arg7;                           \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         "lwz 4,8(11)\n\t"                                        \
         "lwz 5,12(11)\n\t"                                       \
         "lwz 6,16(11)\n\t"  /* arg4->r6 */                       \
         "lwz 7,20(11)\n\t"                                       \
         "lwz 8,24(11)\n\t"                                       \
         "lwz 9,28(11)\n\t"                                       \
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义 CALL_FN_W_8W 实现了一个用于调用函数的宏，支持八个参数。
   这段代码使用了内联汇编，通过向寄存器传递参数并调用目标函数来完成操作。
*/
#define CALL_FN_W_8W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8)                       \
   do {                                                           \
      // 声明并初始化 _orig 变量，存储函数指针 orig
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个数组 _argvec 来存储九个参数
      volatile unsigned long _argvec[9];                          \
      // 声明一个变量 _res 来存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数指针的地址存入参数数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数 arg1 至 arg8 的值依次存入参数数组
      _argvec[1] = (unsigned long)arg1;                           \
      _argvec[2] = (unsigned long)arg2;                           \
      _argvec[3] = (unsigned long)arg3;                           \
      _argvec[4] = (unsigned long)arg4;                           \
      _argvec[5] = (unsigned long)arg5;                           \
      _argvec[6] = (unsigned long)arg6;                           \
      _argvec[7] = (unsigned long)arg7;                           \
      _argvec[8] = (unsigned long)arg8;                           \
      // 使用内联汇编，实现具体的函数调用过程
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         // 将参数数组的第一个元素存入寄存器 11
         "mr 11,%1\n\t"                                           \
         // 将参数数组中的各个参数从寄存器 11 中读取到相应寄存器中
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         "lwz 4,8(11)\n\t"                                        \
         "lwz 5,12(11)\n\t"                                       \
         "lwz 6,16(11)\n\t"  /* arg4->r6 */                       \
         "lwz 7,20(11)\n\t"                                       \
         "lwz 8,24(11)\n\t"                                       \
         "lwz 9,28(11)\n\t"                                       \
         "lwz 10,32(11)\n\t" /* arg8->r10 */                      \
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         // 调用目标函数并将返回值存入 _res
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         // 将返回值 _res 存入 lval
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将 _res 转换为 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/* 定义一个宏，用于在ppc64 Linux平台上调用具有9个参数的函数 */
#define CALL_FN_W_9W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8,arg9)                  \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[10];                         \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)arg1;                           \
      _argvec[2] = (unsigned long)arg2;                           \
      _argvec[3] = (unsigned long)arg3;                           \
      _argvec[4] = (unsigned long)arg4;                           \
      _argvec[5] = (unsigned long)arg5;                           \
      _argvec[6] = (unsigned long)arg6;                           \
      _argvec[7] = (unsigned long)arg7;                           \
      _argvec[8] = (unsigned long)arg8;                           \
      _argvec[9] = (unsigned long)arg9;                           \
      /* 使用内联汇编调用指定参数的函数 */                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "addi 1,1,-16\n\t"                                       \
         /* arg9 */                                               \
         "lwz 3,36(11)\n\t"                                       \
         "stw 3,8(1)\n\t"                                         \
         /* args1-8 */                                            \
         "lwz 3,4(11)\n\t"   /* arg1->r3 */                       \
         "lwz 4,8(11)\n\t"                                        \
         "lwz 5,12(11)\n\t"                                       \
         "lwz 6,16(11)\n\t"  /* arg4->r6 */                       \
         "lwz 7,20(11)\n\t"                                       \
         "lwz 8,24(11)\n\t"                                       \
         "lwz 9,28(11)\n\t"                                       \
         "lwz 10,32(11)\n\t" /* arg8->r10 */                      \
         "lwz 11,0(11)\n\t"  /* target->r11 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         VALGRIND_RESTORE_STACK                                   \
         "mr %0,3"                                                \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

#endif /* PLAT_ppc32_linux */

/* ------------------------ ppc64-linux ------------------------ */

#if defined(PLAT_ppc64be_linux)
/* ARGREGS: r3 r4 r5 r6 r7 r8 r9 r10 (the rest on stack somewhere) */
/* 定义用于函数调用的寄存器列表，其中 r3 到 r10 是参数寄存器，其余可能存放在堆栈中 */

/* These regs are trashed by the hidden call. */
/* 下面这些寄存器会被隐式调用破坏 */
#define __CALLER_SAVED_REGS                                       \
   "lr", "ctr", "xer",                                            \
   "cr0", "cr1", "cr2", "cr3", "cr4", "cr5", "cr6", "cr7",        \
   "r0", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10",         \
   "r11", "r12", "r13"

/* Macros to save and align the stack before making a function
   call and restore it afterwards as gcc may not keep the stack
   pointer aligned if it doesn't realise calls are being made
   to other functions. */
/* 宏用于在进行函数调用前保存并对齐堆栈，以及在调用后恢复堆栈，因为 gcc 可能不会意识到调用其他函数会影响堆栈指针对齐 */

#define VALGRIND_ALIGN_STACK               \
      "mr 28,1\n\t"                        \
      "rldicr 1,1,0,59\n\t"
#define VALGRIND_RESTORE_STACK             \
      "mr 1,28\n\t"

/* These CALL_FN_ macros assume that on ppc64-linux, sizeof(unsigned
   long) == 8. */
/* 下面的 CALL_FN_ 宏假定在 ppc64-linux 上，sizeof(unsigned long) == 8 */

#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+0];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1] = (unsigned long)_orig.r2;                       \
      _argvec[2] = (unsigned long)_orig.nraddr;                   \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义 CALL_FN_W_W(lval, orig, arg1)，用于调用原始函数 orig，并传递参数 arg1。
   lval: 用于存储函数调用结果的变量
   orig: 原始函数的结构体，包含 r2 和 nraddr 字段
   arg1: 作为参数传递给原始函数的参数

   以下是宏的具体实现：
*/
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+1];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   解释：
   - _orig: 将传入的 orig 结构体赋给一个 volatile 的本地变量，确保不会被优化掉
   - _argvec: 用于存储传递给汇编代码的参数的数组
   - _res: 存储汇编代码返回的结果

   汇编部分的解释：
   - VALGRIND_ALIGN_STACK: 对齐栈以支持 Valgrind
   - "mr 11,%1\n\t": 将参数 1 (r2) 的值复制到寄存器 11
   - "std 2,-16(11)\n\t": 存储当前 tocptr（线程本地存储指针）到寄存器 2 中的值
   - "ld   2,-8(11)\n\t": 使用 nraddr 的 tocptr
   - "ld   3, 8(11)\n\t": 将参数 arg1 的值加载到寄存器 3
   - "ld  11, 0(11)\n\t": 将目标函数的地址加载到寄存器 11
   - VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11: 分支并跳转到没有重定向的 r11 寄存器
   - "mr %0,3\n\t": 将寄存器 3 的值（函数返回值）赋给 _res
   - "ld 2,-16(11)\n\t": 恢复 tocptr
   - VALGRIND_RESTORE_STACK: 恢复栈以支持 Valgrind

   约束部分的解释：
   - "cc": 告知编译器可能会改变条件代码（Condition Code）寄存器的值
   - "memory": 告知编译器汇编代码可能会影响内存
   - __CALLER_SAVED_REGS: 告知编译器会影响调用者保存的寄存器
   - "r28": 指示寄存器 28 可能会被修改
*/
/*
   宏定义：CALL_FN_W_WW(lval, orig, arg1, arg2)

   lval:       接收函数调用结果的变量
   orig:       原始函数指针结构体
   arg1, arg2: 函数调用的参数

   该宏用于调用一个函数，并处理函数调用的底层细节，包括参数传递和结果接收。

   实际操作步骤：
   1. 定义 _orig 变量，存储传入的 orig 结构体，这里假定 OrigFn 是一个结构体类型。
   2. 定义 _argvec 数组，用于存放函数调用过程中的参数。
   3. 定义 _res 变量，用于存储函数调用的返回结果。
   4. 使用内联汇编指令来执行实际的函数调用：
      a. 将 _orig.r2 存入 _argvec[1]，保持当前的 r2 寄存器值。
      b. 将 _orig.nraddr 存入 _argvec[2]，保存函数的地址。
      c. 将 arg1 和 arg2 存入 _argvec[3] 和 _argvec[4]。
      d. 执行汇编代码块，包括保存和恢复寄存器、调用目标函数、处理返回值等操作。
   5. 将调用结果转换成 lval 变量的类型，并赋给 lval。

   注意：
   - 使用 volatile 修饰变量，避免编译器优化对宏的影响。
   - 使用内联汇编嵌入了具体的 CPU 指令，用于处理函数调用的底层实现细节。
   - 宏定义使用 do-while 结构，保证宏在展开时不会产生意外的行为。
*/
#define CALL_FN_W_WW(lval, orig, arg1,arg2)                       \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+2];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(11)\n\t" /* arg2->r4 */                      \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+3];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(11)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(11)\n\t" /* arg3->r5 */                      \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：

# 定义宏CALL_FN_W_WWW，用于调用带有6个参数的函数
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      # 使用volatile修饰，确保编译器不优化相关变量
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+3];                        \
      volatile unsigned long _res;                                \
      # _argvec[0] 保存当前r2寄存器的值
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      # 内嵌汇编代码，执行具体的函数调用
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"  # 将寄存器11设置为_r2值 \
         "std 2,-16(11)\n\t"  # 存储tocptr \
         "ld   2,-8(11)\n\t"  # 使用nraddr的tocptr \
         "ld   3, 8(11)\n\t"  # 将arg1加载到r3寄存器 \
         "ld   4, 16(11)\n\t"  # 将arg2加载到r4寄存器 \
         "ld   5, 24(11)\n\t"  # 将arg3加载到r5寄存器 \
         "ld  11, 0(11)\n\t"  # 将target加载到r11寄存器 \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"  # 设置寄存器11为_r2值 \
         "mr %0,3\n\t"  # 将结果存储到寄存器0 \
         "ld 2,-16(11)\n\t"  # 恢复tocptr \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      # 将结果转换为lval的类型，并赋值给lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义 CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)
   用于调用一个函数并处理参数和返回值。

   lval: 函数调用的返回值
   orig: 包含函数原始信息的结构体
   arg1, arg2, arg3, arg4: 函数的四个参数

   注意事项：
   - volatile 用于确保编译器不会优化这些变量
   - asm volatile 声明内联汇编代码
   - 保存和恢复寄存器状态确保函数调用前后寄存器值的正确性
   - VALGRIND_ALIGN_STACK、VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11、VALGRIND_RESTORE_STACK 是特定的 Valgrind 指令，用于调试和性能分析

   注：该宏可能依赖于特定的编译器和体系结构。
*/
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)           \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+4];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(11)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(11)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(11)\n\t" /* arg4->r6 */                      \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_5W

   参数说明：
   lval: 调用后的返回值
   orig: 原始函数的描述结构体
   arg1, arg2, arg3, arg4, arg5: 函数调用的参数

   功能：
   使用给定的参数调用原始函数，并将返回值赋给lval

   具体步骤：
   1. 定义volatile类型的_orig，存储传入的orig参数（原始函数描述结构体）
   2. 定义volatile类型的unsigned long数组 _argvec，用于存储函数调用中的参数
   3. 定义volatile类型的unsigned long变量 _res，存储函数调用的返回值
   4. 将当前函数的r2寄存器值存入_argvec[1]，nraddr的地址存入_argvec[2]
   5. 将传入的参数arg1到arg5分别存入_argvec数组中适当的位置
   6. 使用内联汇编语句调用函数，具体包括寄存器的保存和恢复，以及对参数的传递和返回值的处理

   注意事项：
   - 使用VALGRIND_ALIGN_STACK、VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11、VALGRIND_RESTORE_STACK来处理栈和跳转
   - 汇编部分的寄存器使用和内存访问需要特别小心，以确保正确传递参数和获取返回值
   - 汇编语句中涉及到的寄存器和内存区域需要避免被修改，以及处理器状态的安全性

   返回值：
   函数调用结果存储在lval中
*/
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+5];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      _argvec[2+5] = (unsigned long)arg5;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(11)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(11)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(11)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(11)\n\t" /* arg5->r7 */                      \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+6];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      _argvec[2+5] = (unsigned long)arg5;                         \
      _argvec[2+6] = (unsigned long)arg6;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(11)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(11)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(11)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(11)\n\t" /* arg5->r7 */                      \
         "ld   8, 48(11)\n\t" /* arg6->r8 */                      \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：


#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      // 定义一个 volatile 类型的 OrigFn 结构体 _orig，并将参数 (orig) 赋给它
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个数组 _argvec，大小为 3+6，用来存储函数调用的参数
      volatile unsigned long _argvec[3+6];                        \
      // 定义一个 volatile 类型的 unsigned long 变量 _res，用来存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将当前函数的 r2 寄存器值保存到 _argvec[1] 中，以便在函数调用过程中使用
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      // 将 _orig 的 nraddr 字段保存到 _argvec[2] 中，也是为了函数调用过程中使用
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      // 将参数 arg1 至 arg6 分别保存到 _argvec 数组中，以便传递给被调用的函数
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      _argvec[2+5] = (unsigned long)arg5;                         \
      _argvec[2+6] = (unsigned long)arg6;                         \
      // 使用内联汇编语句进行函数调用，具体步骤包括保存和恢复寄存器状态，以及函数调用本身
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(11)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(11)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(11)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(11)\n\t" /* arg5->r7 */                      \
         "ld   8, 48(11)\n\t" /* arg6->r8 */                      \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将函数调用的返回值转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7)                            \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+7];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      _argvec[2+5] = (unsigned long)arg5;                         \
      _argvec[2+6] = (unsigned long)arg6;                         \
      _argvec[2+7] = (unsigned long)arg7;                         \
      /* Inline assembly code to prepare for and execute function call */ \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(11)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(11)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(11)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(11)\n\t" /* arg5->r7 */                      \
         "ld   8, 48(11)\n\t" /* arg6->r8 */                      \
         "ld   9, 56(11)\n\t" /* arg7->r9 */                      \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：这段代码是一个宏定义，用于调用带有七个参数的函数，并使用内联汇编来实现函数调用。使用了 `volatile` 修饰符来确保编译器不会优化掉这些变量的存取操作。
#define CALL_FN_W_8W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8)                       \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+8];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      _argvec[2+5] = (unsigned long)arg5;                         \
      _argvec[2+6] = (unsigned long)arg6;                         \
      _argvec[2+7] = (unsigned long)arg7;                         \
      _argvec[2+8] = (unsigned long)arg8;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 11,%1\n\t"                                           \
         "std 2,-16(11)\n\t"  /* save tocptr */                   \
         "ld   2,-8(11)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(11)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(11)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(11)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(11)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(11)\n\t" /* arg5->r7 */                      \
         "ld   8, 48(11)\n\t" /* arg6->r8 */                      \
         "ld   9, 56(11)\n\t" /* arg7->r9 */                      \
         "ld  10, 64(11)\n\t" /* arg8->r10 */                     \
         "ld  11, 0(11)\n\t"  /* target->r11 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                  \
         "mr 11,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(11)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：

// 定义宏 CALL_FN_W_8W，用于调用带有 8 个参数的函数，并保存相关寄存器状态
// lval: 用于接收函数返回值的变量
// orig: 原始函数指针，用于获取函数地址及相关寄存器状态
// arg1,...,arg8: 函数调用的参数
// _orig: 使用 volatile 声明的原始函数指针变量，用于保存 orig 参数
// _argvec: 用于存储函数调用时的参数和寄存器状态的数组
// _res: 用于保存函数调用的返回值
// __asm__ volatile: 内联汇编代码块，用于进行函数调用
// VALGRIND_ALIGN_STACK, VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11, VALGRIND_RESTORE_STACK: 汇编宏，用于处理栈对齐和寄存器状态
// 注释详细解释了每行汇编代码的作用和影响
/* ARGREGS: r3 r4 r5 r6 r7 r8 r9 r10 (the rest on stack somewhere) */
/* 定义了一组寄存器，这些寄存器在隐式调用中会被破坏 */

/* These regs are trashed by the hidden call. */
/* 下面的寄存器在隐式调用中会被破坏 */
#define __CALLER_SAVED_REGS                                       \
   "lr", "ctr", "xer",                                            \
   "cr0", "cr1", "cr2", "cr3", "cr4", "cr5", "cr6", "cr7",        \
   "r0", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10",         \
   "r11", "r12", "r13"

/* Macros to save and align the stack before making a function
   call and restore it afterwards as gcc may not keep the stack
   pointer aligned if it doesn't realise calls are being made
   to other functions. */
/* 宏用于在进行函数调用之前保存和对齐栈，并在调用后恢复，
   因为 gcc 可能不会意识到调用了其他函数时需要保持栈指针对齐 */

#define VALGRIND_ALIGN_STACK               \
      "mr 28,1\n\t"                        \
      "rldicr 1,1,0,59\n\t"
#define VALGRIND_RESTORE_STACK             \
      "mr 1,28\n\t"

/* These CALL_FN_ macros assume that on ppc64-linux, sizeof(unsigned
   long) == 8. */
/* 这些 CALL_FN_ 宏假定在 ppc64-linux 上，sizeof(unsigned long) == 8 */

#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+0];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1] = (unsigned long)_orig.r2;                       \
      _argvec[2] = (unsigned long)_orig.nraddr;                   \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 定义一个宏，用于调用带两个参数的函数，并将结果存储在指定的左值变量中
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+1];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"  // 将当前函数的 r2 寄存器值保存到寄存器 12
         "std 2,-16(12)\n\t"  // 保存 tocptr 到栈
         "ld   2,-8(12)\n\t"  // 使用 nraddr 的 tocptr
         "ld   3, 8(12)\n\t"  // 将参数 arg1 的 r3 寄存器值加载到寄存器 3
         "ld  12, 0(12)\n\t"  // 加载目标函数的 r12 寄存器值到寄存器 12
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"  // 将当前函数的 r2 寄存器值再次保存到寄存器 12
         "mr %0,3\n\t"   // 将寄存器 3 的值存储为结果
         "ld 2,-16(12)\n\t"  // 恢复 tocptr
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_WW(lval, orig, arg1,arg2)                       \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+2];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：


#define CALL_FN_W_WW(lval, orig, arg1,arg2)                       \
   do {                                                           \
      // 声明一个 volatile 的 _orig 变量，用于存放 orig 参数
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个包含 3+2 个元素的 volatile unsigned long 数组 _argvec
      volatile unsigned long _argvec[3+2];                        \
      // 声明一个 volatile unsigned long 变量 _res，用于存放函数调用的结果
      volatile unsigned long _res;                                \
      // _argvec[0] 用于在函数调用过程中保持当前的 r2 寄存器的值
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      // 内联汇编代码段，执行函数调用
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* 保存 tocptr */                   \
         "ld   2,-8(12)\n\t"  /* 使用 nraddr 的 tocptr */          \
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         "ld  12, 0(12)\n\t"  /* 目标地址->r12 */                  \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* 恢复 tocptr */                     \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将函数调用的结果转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+3];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(12)\n\t" /* arg3->r5 */                      \
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：


// 定义一个宏，用于调用函数并传递三个参数，返回值赋给 lval
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      // 声明一个 volatile 变量 _orig，并将 orig 赋值给它
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个 volatile 数组 _argvec，大小为 6，用于存储函数调用的参数
      volatile unsigned long _argvec[3+3];                        \
      // 声明一个 volatile unsigned long 类型的变量 _res，用于存储函数调用的返回值
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      // 将 _orig.r2 的值存入 _argvec[1]，作为函数调用过程中的 r2 寄存器的值
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      // 将 _orig.nraddr 的值存入 _argvec[2]，作为函数调用过程中的 nraddr 寄存器的值
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      // 将 arg1 的值存入 _argvec[3]，作为第一个函数参数
      _argvec[2+1] = (unsigned long)arg1;                         \
      // 将 arg2 的值存入 _argvec[4]，作为第二个函数参数
      _argvec[2+2] = (unsigned long)arg2;                         \
      // 将 arg3 的值存入 _argvec[5]，作为第三个函数参数
      _argvec[2+3] = (unsigned long)arg3;                         \
      // 使用内联汇编语法执行以下操作
      __asm__ volatile(                                           \
         // 插入 valgrind 相关的堆栈对齐指令
         VALGRIND_ALIGN_STACK                                     \
         // 将 %1 的值（_argvec[2]）存入寄存器 r12
         "mr 12,%1\n\t"                                           \
         // 将 tocptr 保存到内存中（在寄存器 2 上）
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         // 从内存中加载 nraddr 的 tocptr 到寄存器 2
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         // 将 arg1 的值加载到寄存器 3（作为第一个参数）
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         // 将 arg2 的值加载到寄存器 4（作为第二个参数）
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         // 将 arg3 的值加载到寄存器 5（作为第三个参数）
         "ld   5, 24(12)\n\t" /* arg3->r5 */                      \
         // 将目标地址加载到寄存器 12
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         // 执行 valgrind 分支和链接到不重定向到 r12 的指令
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         // 将寄存器 %1 的值（_argvec[2]）存入寄存器 r12
         "mr 12,%1\n\t"                                           \
         // 将寄存器 3 的值存入 %0（_res 变量），作为函数调用的返回值
         "mr %0,3\n\t"                                            \
         // 从内存中加载 tocptr 到寄存器 2，恢复 tocptr
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         // 执行 valgrind 恢复堆栈指令
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将 _res 转换为 lval 的类型，并将其赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


这段代码是一个宏，用于调用函数并传递三个参数，使用了内联汇编来实现。
/*
   宏定义：CALL_FN_W_WWWW(lval, orig, arg1, arg2, arg3, arg4)

   参数说明：
   lval: 调用后返回的值
   orig: 原始函数指针结构体
   arg1, arg2, arg3, arg4: 函数调用时的参数

   功能：
   这个宏用于调用一个函数，并且处理参数和返回值的传递。它通过内联汇编语句来完成函数调用。

   详细注释：
   - _orig: 声明一个 volatile 的 OrigFn 结构体指针，用于存储传入的原始函数指针
   - _argvec: 声明一个 volatile 的无符号长整型数组，用于存储函数调用时的参数
     _argvec[0]: 保留当前 r2 寄存器的值，用于跨函数调用时的传递
     _argvec[1]: 存储 _orig.r2 的值
     _argvec[2]: 存储 _orig.nraddr 的值
     _argvec[3]: 存储 arg1 的值
     _argvec[4]: 存储 arg2 的值
     _argvec[5]: 存储 arg3 的值
     _argvec[6]: 存储 arg4 的值
   - 内联汇编语句:
     - VALGRIND_ALIGN_STACK: 对齐栈
     - "mr 12,%1\n\t": 将 _orig.nraddr 的值加载到寄存器 r12 中
     - "std 2,-16(12)\n\t": 保存 tocptr 到栈中
     - "ld   2,-8(12)\n\t": 使用 nraddr 的 tocptr
     - "ld   3, 8(12)\n\t": 将 arg1->r3
     - "ld   4, 16(12)\n\t": 将 arg2->r4
     - "ld   5, 24(12)\n\t": 将 arg3->r5
     - "ld   6, 32(12)\n\t": 将 arg4->r6
     - "ld  12, 0(12)\n\t": 将 target->r12
     - VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12: 分支并链接到非重定向的 r12
     - "mr 12,%1\n\t": 将 _orig.nraddr 的值加载到寄存器 r12 中
     - "mr %0,3\n\t": 将返回值加载到 _res 中
     - "ld 2,-16(12)\n\t": 从栈中恢复 tocptr
     - VALGRIND_RESTORE_STACK: 恢复栈
   - 输出约束:
     - "=r" (_res): 将返回值存储到 _res 中
   - 输入约束:
     - "r" (&_argvec[2]): 将参数数组的地址传递给内联汇编
   - 破坏约束:
     - "cc", "memory", __CALLER_SAVED_REGS, "r28": 列出可能被修改的寄存器和内存位置

   注意事项：
   - 使用 do { ... } while (0) 结构，确保宏能在不引起语法错误的情况下用作单个语句
*/
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)           \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+4];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(12)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(12)\n\t" /* arg4->r6 */                      \
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+5];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      _argvec[2+5] = (unsigned long)arg5;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(12)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(12)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(12)\n\t" /* arg5->r7 */                      \
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：


#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \  // 定义宏，带有5个参数和返回值，通过内联汇编调用函数
   do {                                                           \  // 宏定义开始
      volatile OrigFn        _orig = (orig);                      \  // 声明并初始化_orig变量为类型为OrigFn的volatile变量，值为(orig)
      volatile unsigned long _argvec[3+5];                        \  // 声明一个包含8个元素的volatile unsigned long数组
      volatile unsigned long _res;                                \  // 声明一个volatile unsigned long变量_res，用于保存函数调用的返回值
      /* _argvec[0] holds current r2 across the call */           \  // 注释：_argvec[0]用于在调用过程中保持当前的r2寄存器的值
      _argvec[1]   = (unsigned long)_orig.r2;                     \  // 将_orig的r2成员的值转换为unsigned long类型并赋给_argvec[1]
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \  // 将_orig的nraddr成员的值转换为unsigned long类型并赋给_argvec[2]
      _argvec[2+1] = (unsigned long)arg1;                         \  // 将arg1的值转换为unsigned long类型并赋给_argvec[3]
      _argvec[2+2] = (unsigned long)arg2;                         \  // 将arg2的值转换为unsigned long类型并赋给_argvec[4]
      _argvec[2+3] = (unsigned long)arg3;                         \  // 将arg3的值转换为unsigned long类型并赋给_argvec[5]
      _argvec[2+4] = (unsigned long)arg4;                         \  // 将arg4的值转换为unsigned long类型并赋给_argvec[6]
      _argvec[2+5] = (unsigned long)arg5;                         \  // 将arg5的值转换为unsigned long类型并赋给_argvec[7]
      __asm__ volatile(                                           \  // 使用内联汇编
         VALGRIND_ALIGN_STACK                                     \  // 对齐栈，用于调试工具Valgrind
         "mr 12,%1\n\t"                                           \  // 将_argvec[1]的值移到寄存器12中
         "std 2,-16(12)\n\t"  /* save tocptr */                   \  // 保存tocptr到寄存器2的偏移位置-16处
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \  // 使用nraddr的tocptr加载寄存器2的偏移位置-8处
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \  // 将arg1的值加载到寄存器3中
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \  // 将arg2的值加载到寄存器4中
         "ld   5, 24(12)\n\t" /* arg3->r5 */                      \  // 将arg3的值加载到寄存器5中
         "ld   6, 32(12)\n\t" /* arg4->r6 */                      \  // 将arg4的值加载到寄存器6中
         "ld   7, 40(12)\n\t" /* arg5->r7 */                      \  // 将arg5的值加载到寄存器7中
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \  // 将目标的值加载到寄存器12中
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \  // Valgrind跟踪分支并链接到不重新定向r12
         "mr 12,%1\n\t"                                           \  // 将_argvec[1]的值移到寄存器12中
         "mr %0,3\n\t"                                            \  // 将寄存器3的值移到%0中
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \  // 恢复tocptr到寄存器2的偏移位置-16处
         VALGRIND_RESTORE_STACK                                   \  // 恢复栈，用于调试工具Valgrind
         : /*out*/   "=r" (_res)                                  \  // 输出：将_res赋值给寄存器
         : /*in*/    "r" (&_argvec[2])                            \  // 输入：将&_argvec[2]赋值给寄存器
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \  // 垃圾：指令集为cc、memory、__CALLER_SAVED_REGS、r28
      );                                                          \  // 内联汇编结束
      lval = (__typeof__(lval)) _res;                             \  // 将类型为__typeof__(lval)的_res值赋给lval
   } while (0)                                                     \  // 结束宏定义


这段代码是一个使用内联汇编实现的宏，用于调用带有5个参数的函数，并返回一个值给lval。
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个保存原始函数指针的变量 _orig
      volatile unsigned long _argvec[3+6];                        \
      // 定义一个包含参数的数组 _argvec，数组大小为参数个数加上3
      volatile unsigned long _res;                                \
      // 定义一个保存函数调用结果的变量 _res
      /* _argvec[0] holds current r2 across the call */           \
      // _argvec[0] 用于在函数调用期间保存当前的 r2 寄存器值
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      // 将 _orig 结构体中的 r2 字段赋值给 _argvec[1]
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      // 将 _orig 结构体中的 nraddr 字段赋值给 _argvec[2]
      _argvec[2+1] = (unsigned long)arg1;                         \
      // 将函数参数 arg1 赋值给 _argvec[3]
      _argvec[2+2] = (unsigned long)arg2;                         \
      // 将函数参数 arg2 赋值给 _argvec[4]
      _argvec[2+3] = (unsigned long)arg3;                         \
      // 将函数参数 arg3 赋值给 _argvec[5]
      _argvec[2+4] = (unsigned long)arg4;                         \
      // 将函数参数 arg4 赋值给 _argvec[6]
      _argvec[2+5] = (unsigned long)arg5;                         \
      // 将函数参数 arg5 赋值给 _argvec[7]
      _argvec[2+6] = (unsigned long)arg6;                         \
      // 将函数参数 arg6 赋值给 _argvec[8]
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(12)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(12)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(12)\n\t" /* arg5->r7 */                      \
         "ld   8, 48(12)\n\t" /* arg6->r8 */                      \
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 使用内联汇编实现函数调用，使用 _argvec 数组传递参数
      lval = (__typeof__(lval)) _res;                             \
      // 将函数调用的返回值转换为指定类型，并赋给 lval
   } while (0)
// 结束宏定义
/*
   定义一个宏函数 CALL_FN_W_7W，用于调用一个带有7个参数的原始函数。
   参数：
   lval: 接收函数调用结果的变量
   orig: 原始函数的结构体
   arg1-arg7: 原始函数的参数
*/
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7)                            \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[3+7];                        \
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      _argvec[2+1] = (unsigned long)arg1;                         \
      _argvec[2+2] = (unsigned long)arg2;                         \
      _argvec[2+3] = (unsigned long)arg3;                         \
      _argvec[2+4] = (unsigned long)arg4;                         \
      _argvec[2+5] = (unsigned long)arg5;                         \
      _argvec[2+6] = (unsigned long)arg6;                         \
      _argvec[2+7] = (unsigned long)arg7;                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(12)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(12)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(12)\n\t" /* arg5->r7 */                      \
         "ld   8, 48(12)\n\t" /* arg6->r8 */                      \
         "ld   9, 56(12)\n\t" /* arg7->r9 */                      \
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_8W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8)                       \
   do {                                                           \
      // 定义一个指向原始函数的指针_orig，并将传入的orig赋值给它
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个包含3+8个元素的无符号长整型数组_argvec
      volatile unsigned long _argvec[3+8];                        \
      // 定义一个无符号长整型变量_res
      volatile unsigned long _res;                                \
      /* _argvec[0] holds current r2 across the call */           \
      // 将当前r2的值存储在_argvec[1]中
      _argvec[1]   = (unsigned long)_orig.r2;                     \
      // 将nraddr的值存储在_argvec[2]中
      _argvec[2]   = (unsigned long)_orig.nraddr;                 \
      // 将arg1的值存储在_argvec[2+1]中
      _argvec[2+1] = (unsigned long)arg1;                         \
      // 将arg2的值存储在_argvec[2+2]中
      _argvec[2+2] = (unsigned long)arg2;                         \
      // 将arg3的值存储在_argvec[2+3]中
      _argvec[2+3] = (unsigned long)arg3;                         \
      // 将arg4的值存储在_argvec[2+4]中
      _argvec[2+4] = (unsigned long)arg4;                         \
      // 将arg5的值存储在_argvec[2+5]中
      _argvec[2+5] = (unsigned long)arg5;                         \
      // 将arg6的值存储在_argvec[2+6]中
      _argvec[2+6] = (unsigned long)arg6;                         \
      // 将arg7的值存储在_argvec[2+7]中
      _argvec[2+7] = (unsigned long)arg7;                         \
      // 将arg8的值存储在_argvec[2+8]中
      _argvec[2+8] = (unsigned long)arg8;                         \
      // 内联汇编代码块，执行函数调用
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "mr 12,%1\n\t"                                           \
         "std 2,-16(12)\n\t"  /* save tocptr */                   \
         "ld   2,-8(12)\n\t"  /* use nraddr's tocptr */           \
         "ld   3, 8(12)\n\t"  /* arg1->r3 */                      \
         "ld   4, 16(12)\n\t" /* arg2->r4 */                      \
         "ld   5, 24(12)\n\t" /* arg3->r5 */                      \
         "ld   6, 32(12)\n\t" /* arg4->r6 */                      \
         "ld   7, 40(12)\n\t" /* arg5->r7 */                      \
         "ld   8, 48(12)\n\t" /* arg6->r8 */                      \
         "ld   9, 56(12)\n\t" /* arg7->r9 */                      \
         "ld  10, 64(12)\n\t" /* arg8->r10 */                     \
         "ld  12, 0(12)\n\t"  /* target->r12 */                   \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                  \
         "mr 12,%1\n\t"                                           \
         "mr %0,3\n\t"                                            \
         "ld 2,-16(12)\n\t" /* restore tocptr */                  \
         VALGRIND_RESTORE_STACK                                   \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[2])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r28"   \
      );                                                          \
      // 将_res转换为lval的类型，并赋值给lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

#endif /* PLAT_ppc64le_linux */

/* ------------------------- arm-linux ------------------------- */

#if defined(PLAT_arm_linux)

/* These regs are trashed by the hidden call. */
/* 定义调用者保存的寄存器列表 */
#define __CALLER_SAVED_REGS "r0", "r1", "r2", "r3","r4", "r12", "r14"

/* 宏用于在进行函数调用前保存和对齐堆栈，并在调用后恢复堆栈，因为 gcc 可能不会保持堆栈指针对齐，如果它没有意识到正在调用其他函数。 */

/* 这有点棘手。我们将原始堆栈指针存储在 r10 中，因为它是被调用者保存的。由于某种原因，gcc 不允许使用 r11。此外，在 thumb 模式下不能直接“bic”堆栈指针，因为在该上下文中 r13 不是允许的寄存器编号。因此，使用 r4 作为临时寄存器，因为它在使用此宏后会立即被破坏。副作用是我们需要非常小心任何未来的更改，因为 VALGRIND_ALIGN_STACK 简单地假定 r4 是可用的。 */
#define VALGRIND_ALIGN_STACK               \
      "mov r10, sp\n\t"                    \
      "mov r4,  sp\n\t"                    \
      "bic r4,  r4, #7\n\t"                \
      "mov sp,  r4\n\t"
#define VALGRIND_RESTORE_STACK             \
      "mov sp,  r10\n\t"

/* 这些 CALL_FN_ 宏假设在 arm-linux 上，sizeof(unsigned long) == 4。 */

#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[1];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0\n"                                           \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      // 定义一个指向原始函数的指针_orig，并将传入的orig赋给它
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个包含两个元素的无符号长整型数组_argvec，用于存放参数
      volatile unsigned long _argvec[2];                          \
      // 定义一个无符号长整型变量_res，用于存放函数调用的返回值
      volatile unsigned long _res;                                \
      // 将_orig.nraddr转换为无符号长整型后存入_argvec的第一个元素
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将arg1转换为无符号长整型后存入_argvec的第二个元素
      _argvec[1] = (unsigned long)(arg1);                         \
      // 内联汇编语句，调用函数并将返回值保存到_res中
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0\n"                                           \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      // 将_res强制转换为lval的类型并赋值给lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

#define CALL_FN_W_WW(lval, orig, arg1,arg2)                       \
   do {                                                           \
      // 定义一个指向原始函数的指针_orig，并将传入的orig赋给它
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个包含三个元素的无符号长整型数组_argvec，用于存放参数
      volatile unsigned long _argvec[3];                          \
      // 定义一个无符号长整型变量_res，用于存放函数调用的返回值
      volatile unsigned long _res;                                \
      // 将_orig.nraddr转换为无符号长整型后存入_argvec的第一个元素
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将arg1和arg2转换为无符号长整型后分别存入_argvec的第二和第三个元素
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      // 内联汇编语句，调用函数并将返回值保存到_res中
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r1, [%1, #8] \n\t"                                  \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0\n"                                           \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      // 将_res强制转换为lval的类型并赋值给lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 定义一个宏，用于调用带有三个参数的函数指针并返回结果给指定变量
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      // 声明原始函数指针_orig，并将传入的参数 orig 赋值给它
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个存储参数的数组 _argvec，包括函数指针和三个参数
      volatile unsigned long _argvec[4];                          \
      // 声明一个变量 _res 用于存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数指针的地址、三个参数依次存入 _argvec 数组
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      // 使用内联汇编执行具体的函数调用和返回值处理
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r1, [%1, #8] \n\t"                                  \
         "ldr r2, [%1, #12] \n\t"                                 \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0\n"                                           \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      // 将函数调用返回的结果转换为目标变量 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 定义一个宏，用于调用指定的函数并传递五个参数
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)           \
   do {                                                           \
      // 声明一个 volatile 的函数指针 _orig，指向传入的函数指针 (orig)
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个 volatile 的 unsigned long 数组 _argvec，存放函数调用的参数
      volatile unsigned long _argvec[5];                          \
      // 声明一个 volatile 的 unsigned long 变量 _res，用于存放函数调用的返回值
      volatile unsigned long _res;                                \
      // 将原始函数指针的地址存入参数数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将传入的参数 arg1 存入参数数组的第二个位置
      _argvec[1] = (unsigned long)(arg1);                         \
      // 将传入的参数 arg2 存入参数数组的第三个位置
      _argvec[2] = (unsigned long)(arg2);                         \
      // 将传入的参数 arg3 存入参数数组的第四个位置
      _argvec[3] = (unsigned long)(arg3);                         \
      // 将传入的参数 arg4 存入参数数组的第五个位置
      _argvec[4] = (unsigned long)(arg4);                         \
      // 使用内联汇编，加载参数并调用函数
      __asm__ volatile(                                           \
         // 栈对齐（如果有必要）
         VALGRIND_ALIGN_STACK                                     \
         // 加载参数到寄存器 r0-r4
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r1, [%1, #8] \n\t"                                  \
         "ldr r2, [%1, #12] \n\t"                                 \
         "ldr r3, [%1, #16] \n\t"                                 \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         // 分支并链接到目标函数的地址，使用 r4 寄存器作为跳转目标
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         // 恢复栈状态
         VALGRIND_RESTORE_STACK                                   \
         // 将函数返回值移动到 _res 变量
         "mov %0, r0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      // 将函数返回值转换为宏定义时指定的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[6];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "sub sp, sp, #4 \n\t"                                    \
         "ldr r0, [%1, #20] \n\t"                                 \
         "push {r0} \n\t"                                         \
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r1, [%1, #8] \n\t"                                  \
         "ldr r2, [%1, #12] \n\t"                                 \
         "ldr r3, [%1, #16] \n\t"                                 \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)


注释：


# 定义一个宏，用于调用带有5个参数的函数
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      // 声明原始函数指针_orig，并将参数orig转换为volatile类型
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个包含6个unsigned long类型元素的参数向量_argvec
      volatile unsigned long _argvec[6];                          \
      // 声明一个unsigned long类型变量用于存储函数调用的结果_res
      volatile unsigned long _res;                                \
      // 将原始函数指针的地址存入参数向量的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数arg1至arg5依次存入参数向量的后续位置
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      // 使用内联汇编嵌入指令来调用函数，具体实现依赖于平台和编译器
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "sub sp, sp, #4 \n\t"                                    \
         "ldr r0, [%1, #20] \n\t"                                 \
         "push {r0} \n\t"                                         \
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r1, [%1, #8] \n\t"                                  \
         "ldr r2, [%1, #12] \n\t"                                 \
         "ldr r3, [%1, #16] \n\t"                                 \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      // 将函数调用的返回值_res转换为宏参数lval指定的类型，并赋给lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   使用宏定义封装的函数调用宏，用于调用指定原始函数 `_orig`，传递六个参数 (arg1, arg2, arg3, arg4, arg5, arg6)。
   函数参数：
      lval: 调用结果的左值引用
      orig: 原始函数指针
      arg1-arg6: 六个函数调用的参数

   实现细节：
      - 声明并初始化 `_orig` 为原始函数的地址
      - 定义 `_argvec` 数组用于存储参数
      - 声明 `_res` 用于存储函数调用的返回值
      - 将原始函数地址和各个参数存入 `_argvec`
      - 使用内联汇编 (`__asm__ volatile`) 执行以下操作：
        - 加载 `_argvec` 中的地址作为参数传递给汇编语句
        - 使用汇编指令加载寄存器 r0-r4 的值，并将它们推入栈中
        - 执行汇编指令将 r0-r4 中的值加载到相应的寄存器中
        - 调用 `_orig` 函数，并将返回值存入 r0 中
        - 将 r0 中的返回值移动到 `_res` 中
      - 最后将 `_res` 转换为 `lval` 的类型，并赋值给 `lval`

   注意事项：
      - 使用 volatile 关键字确保变量不会被编译器优化，以保证汇编代码的正确性
      - 汇编部分需要注意的寄存器保存和恢复，以及内存和条件码的使用
*/
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[7];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr r0, [%1, #20] \n\t"                                 \
         "ldr r1, [%1, #24] \n\t"                                 \
         "push {r0, r1} \n\t"                                     \
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r1, [%1, #8] \n\t"                                  \
         "ldr r2, [%1, #12] \n\t"                                 \
         "ldr r3, [%1, #16] \n\t"                                 \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
// 定义宏 CALL_FN_W_7W，用于调用指定原始函数 _orig，并传入七个参数 arg1 到 arg7
#define CALL_FN_W_7W(lval, orig, arg1, arg2, arg3, arg4, arg5, arg6, arg7) \
   do {                                                                    \
      // 声明 volatile 类型的变量 _orig，存储传入的原始函数指针 (orig)
      volatile OrigFn _orig = (orig);                                      \
      // 声明 volatile 类型的数组 _argvec，用于存储传递给原始函数的参数
      volatile unsigned long _argvec[8];                                   \
      // 声明 volatile 类型的变量 _res，用于存储原始函数调用的返回值
      volatile unsigned long _res;                                         \
      // 将原始函数指针的地址存入 _argvec 数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                            \
      // 将参数 arg1 到 arg7 分别存入 _argvec 数组的后续位置
      _argvec[1] = (unsigned long)(arg1);                                  \
      _argvec[2] = (unsigned long)(arg2);                                  \
      _argvec[3] = (unsigned long)(arg3);                                  \
      _argvec[4] = (unsigned long)(arg4);                                  \
      _argvec[5] = (unsigned long)(arg5);                                  \
      _argvec[6] = (unsigned long)(arg6);                                  \
      _argvec[7] = (unsigned long)(arg7);                                  \
      // 使用内联汇编调用原始函数，传递参数并获取返回值
      __asm__ volatile(                                                    \
         // 在调用之前对栈进行调整，以满足特定对齐要求
         VALGRIND_ALIGN_STACK                                              \
         // 依次将参数加载到寄存器 r0 到 r4，并将 r0 到 r2 入栈
         "sub sp, sp, #4 \n\t"                                             \
         "ldr r0, [%1, #20] \n\t"                                          \
         "ldr r1, [%1, #24] \n\t"                                          \
         "ldr r2, [%1, #28] \n\t"                                          \
         "push {r0, r1, r2} \n\t"                                          \
         "ldr r0, [%1, #4] \n\t"                                           \
         "ldr r1, [%1, #8] \n\t"                                           \
         "ldr r2, [%1, #12] \n\t"                                          \
         "ldr r3, [%1, #16] \n\t"                                          \
         "ldr r4, [%1] \n\t"  /* target->r4 */                             \
         // 调用原始函数，并保存返回值到 r0 中
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                            \
         // 恢复栈的状态
         VALGRIND_RESTORE_STACK                                            \
         // 将返回值 r0 赋给 _res
         "mov %0, r0"                                                      \
         : /*out*/   "=r" (_res)                                           \
         : /*in*/    "0" (&_argvec[0])                                     \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"            \
      );                                                                   \
      // 将 _res 转换为 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) _res;                                       \
   } while (0)
/*
   宏定义：用于调用一个带有8个参数的函数，并将结果赋给lval。
   注意事项：
   - 宏参数包括：lval（结果存储变量）、orig（函数指针）、arg1到arg8（函数的8个参数）。
   - 使用volatile关键字声明_orig、_argvec、_res，确保编译器不对它们进行优化。
   - 将函数指针_orig的地址和参数arg1到arg8的值存储在数组_argvec中。
   - 使用内联汇编（__asm__ volatile）来执行具体的函数调用操作。
   - 汇编代码加载参数到寄存器r0到r4，将r0到r3压入堆栈，执行函数调用，然后将结果保存到_res中。
   - 最后将_res强制类型转换为lval的类型，并将其赋值给lval。
*/
#define CALL_FN_W_8W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8)                       \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[9];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr r0, [%1, #20] \n\t"                                 \
         "ldr r1, [%1, #24] \n\t"                                 \
         "ldr r2, [%1, #28] \n\t"                                 \
         "ldr r3, [%1, #32] \n\t"                                 \
         "push {r0, r1, r2, r3} \n\t"                             \
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r1, [%1, #8] \n\t"                                  \
         "ldr r2, [%1, #12] \n\t"                                 \
         "ldr r3, [%1, #16] \n\t"                                 \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：用于调用具有9个参数的函数，支持异常处理与寄存器管理

   #define CALL_FN_W_9W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,
                                    arg7,arg8,arg9)                  \
      do {                                                           \
         volatile OrigFn        _orig = (orig);                      \ 定义并初始化函数指针_orig，指向传入的orig函数
         volatile unsigned long _argvec[10];                         \ 声明并定义长度为10的无符号长整型数组_argvec
         volatile unsigned long _res;                                \ 声明结果变量_res为无符号长整型

         // 将函数指针_orig的地址存入_argvec数组的第一个位置
         _argvec[0] = (unsigned long)_orig.nraddr;                   \

         // 将传入的9个参数分别存入_argvec数组的后续位置
         _argvec[1] = (unsigned long)(arg1);                         \
         _argvec[2] = (unsigned long)(arg2);                         \
         _argvec[3] = (unsigned long)(arg3);                         \
         _argvec[4] = (unsigned long)(arg4);                         \
         _argvec[5] = (unsigned long)(arg5);                         \
         _argvec[6] = (unsigned long)(arg6);                         \
         _argvec[7] = (unsigned long)(arg7);                         \
         _argvec[8] = (unsigned long)(arg8);                         \
         _argvec[9] = (unsigned long)(arg9);                         \

         // 内联汇编代码块开始，使用volatile关键字确保编译器不会优化这些代码
         __asm__ volatile(                                           \
            VALGRIND_ALIGN_STACK                                     \ 对齐栈，可能是一个宏定义，作用在调试时确保栈正确
            "sub sp, sp, #4 \n\t"                                    \ 减小栈指针sp，分配4字节空间
            "ldr r0, [%1, #20] \n\t"                                 \ 从参数指针数组中加载第1个参数到寄存器r0
            "ldr r1, [%1, #24] \n\t"                                 \ 从参数指针数组中加载第2个参数到寄存器r1
            "ldr r2, [%1, #28] \n\t"                                 \ 从参数指针数组中加载第3个参数到寄存器r2
            "ldr r3, [%1, #32] \n\t"                                 \ 从参数指针数组中加载第4个参数到寄存器r3
            "ldr r4, [%1, #36] \n\t"                                 \ 从参数指针数组中加载第5个参数到寄存器r4
            "push {r0, r1, r2, r3, r4} \n\t"                         \ 将寄存器r0到r4的值依次压入栈中
            "ldr r0, [%1, #4] \n\t"                                  \ 从参数指针数组中加载第6个参数到寄存器r0
            "ldr r1, [%1, #8] \n\t"                                  \ 从参数指针数组中加载第7个参数到寄存器r1
            "ldr r2, [%1, #12] \n\t"                                 \ 从参数指针数组中加载第8个参数到寄存器r2
            "ldr r3, [%1, #16] \n\t"                                 \ 从参数指针数组中加载第9个参数到寄存器r3
            "ldr r4, [%1] \n\t"  /* target->r4 */                    \ 加载目标寄存器r4
            VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \ 调用函数并链接，不重定向r4寄存器
            VALGRIND_RESTORE_STACK                                   \ 恢复栈指针
            "mov %0, r0"                                             \ 将寄存器r0的值移动到结果变量_res中
            : /*out*/   "=r" (_res)                                  \ 输出约束：告诉编译器将r0寄存器的内容输出到_res变量
            : /*in*/    "0" (&_argvec[0])                            \ 输入约束：告诉编译器使用_argvec数组的地址作为输入参数
            : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \ 临时寄存器列表，告诉编译器这些寄存器的值可能被修改
         );                                                          \
         // 将结果转换为目标类型并赋值给lval
         lval = (__typeof__(lval)) _res;                             \
      } while (0)
*/
/* 定义一个宏，用于调用带有10个参数的函数 */
#define CALL_FN_W_10W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,  \
                                  arg7,arg8,arg9,arg10)           \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[11];                         \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      _argvec[9] = (unsigned long)(arg9);                         \
      _argvec[10] = (unsigned long)(arg10);                       \
      /* 使用内联汇编调用函数，将参数和函数指针传递给目标 */     \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr r0, [%1, #40] \n\t"                                 \
         "push {r0} \n\t"                                         \
         "ldr r0, [%1, #20] \n\t"                                 \
         "ldr r1, [%1, #24] \n\t"                                 \
         "ldr r2, [%1, #28] \n\t"                                 \
         "ldr r3, [%1, #32] \n\t"                                 \
         "ldr r4, [%1, #36] \n\t"                                 \
         "push {r0, r1, r2, r3, r4} \n\t"                         \
         "ldr r0, [%1, #4] \n\t"                                  \
         "ldr r1, [%1, #8] \n\t"                                  \
         "ldr r2, [%1, #12] \n\t"                                 \
         "ldr r3, [%1, #16] \n\t"                                 \
         "ldr r4, [%1] \n\t"  /* target->r4 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, r0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "r10"   \
      );                                                          \
      /* 将返回值转换为目标类型并赋给左值 */                      \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#endif /* PLAT_arm_linux */

/* ------------------------ arm64-linux ------------------------ */

#if defined(PLAT_arm64_linux)

/* These regs are trashed by the hidden call. */
/* 定义了 __CALLER_SAVED_REGS 宏，包含了所有需要在函数调用前保存的寄存器，包括通用寄存器和向量寄存器 */
#define __CALLER_SAVED_REGS \
     "x0", "x1", "x2", "x3","x4", "x5", "x6", "x7", "x8", "x9",   \
     "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17",      \
     "x18", "x19", "x20", "x30",                                  \
     "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",  \
     "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",      \
     "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",      \
     "v26", "v27", "v28", "v29", "v30", "v31"

/* 定义了 VALGRIND_ALIGN_STACK 宏，用于调整栈的对齐，将栈指针 sp 对齐到 16 字节边界 */
#define VALGRIND_ALIGN_STACK               \
      "mov x21, sp\n\t"                    \
      "bic sp, x21, #15\n\t"

/* 定义了 VALGRIND_RESTORE_STACK 宏，用于恢复栈指针 sp 的原始值 */
#define VALGRIND_RESTORE_STACK             \
      "mov sp,  x21\n\t"

/* 定义了 CALL_FN_W_v 宏，用于调用一个函数，其中 lval 是返回值，orig 是函数指针 */
#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[1];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, x0\n"                                           \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 定义一个宏，用于调用带有一个参数的函数指针并返回一个值
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      // 声明一个指向函数原型的变量 _orig，并将传入的 orig 赋值给它
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个包含两个元素的无符号长整型数组 _argvec，用于存储参数和结果
      volatile unsigned long _argvec[2];                          \
      // 声明一个无符号长整型变量 _res，用于存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数原型 _orig 的地址存入 _argvec 数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数 arg1 的值存入 _argvec 数组的第二个位置
      _argvec[1] = (unsigned long)(arg1);                         \
      // 使用内联汇编语法调用函数，将函数返回值存入 _res，使用 x0 寄存器传递第一个参数
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr x0, [%1, #8] \n\t"                                  \
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, x0\n"                                           \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      // 将 _res 强制转换为 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

# 定义一个宏，用于调用带有两个参数的函数指针并返回一个值
#define CALL_FN_W_WW(lval, orig, arg1,arg2)                       \
   do {                                                           \
      // 声明一个指向函数原型的变量 _orig，并将传入的 orig 赋值给它
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个包含三个元素的无符号长整型数组 _argvec，用于存储参数和结果
      volatile unsigned long _argvec[3];                          \
      // 声明一个无符号长整型变量 _res，用于存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数原型 _orig 的地址存入 _argvec 数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数 arg1 的值存入 _argvec 数组的第二个位置
      _argvec[1] = (unsigned long)(arg1);                         \
      // 将参数 arg2 的值存入 _argvec 数组的第三个位置
      _argvec[2] = (unsigned long)(arg2);                         \
      // 使用内联汇编语法调用函数，将函数返回值存入 _res，使用 x0 和 x1 寄存器传递前两个参数
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr x0, [%1, #8] \n\t"                                  \
         "ldr x1, [%1, #16] \n\t"                                 \
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, x0\n"                                           \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      // 将 _res 强制转换为 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 宏定义，用于调用指定签名的函数，返回结果给 lval，传入参数为 orig, arg1, arg2, arg3
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      // 定义一个 volatile 变量 _orig，用于保存原始函数指针
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个 volatile 数组 _argvec，存储函数调用的参数
      volatile unsigned long _argvec[4];                          \
      // 定义一个 volatile 变量 _res，用于保存函数调用的结果
      volatile unsigned long _res;                                \
      // 将参数 orig 的地址作为函数指针保存到 _argvec[0] 中
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数 arg1 转换为无符号长整型后存入 _argvec[1]
      _argvec[1] = (unsigned long)(arg1);                         \
      // 将参数 arg2 转换为无符号长整型后存入 _argvec[2]
      _argvec[2] = (unsigned long)(arg2);                         \
      // 将参数 arg3 转换为无符号长整型后存入 _argvec[3]
      _argvec[3] = (unsigned long)(arg3);                         \
      // 使用内联汇编，加载参数寄存器 x0、x1、x2、x8，调用函数并返回结果
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr x0, [%1, #8] \n\t"                                  \
         "ldr x1, [%1, #16] \n\t"                                 \
         "ldr x2, [%1, #24] \n\t"                                 \
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, x0\n"                                           \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      // 将函数调用的结果 _res 赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义 CALL_FN_W_WWWW 用于调用函数并传递五个参数
*/
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)           \
   do {                                                           \
      // 声明 volatile 原始函数指针 _orig，并初始化为 orig
      volatile OrigFn        _orig = (orig);                      \
      // 声明 volatile 无符号长整型数组 _argvec，存放参数
      volatile unsigned long _argvec[5];                          \
      // 声明 volatile 无符号长整型 _res，存放函数调用结果
      volatile unsigned long _res;                                \
      // 将原始函数地址存入参数数组
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数1、参数2、参数3、参数4存入参数数组
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      // 使用内联汇编执行以下操作
      __asm__ volatile(                                           \
         // 对齐栈
         VALGRIND_ALIGN_STACK                                     \
         // 加载参数到寄存器 x0-x3
         "ldr x0, [%1, #8] \n\t"                                  \
         "ldr x1, [%1, #16] \n\t"                                 \
         "ldr x2, [%1, #24] \n\t"                                 \
         "ldr x3, [%1, #32] \n\t"                                 \
         // 加载目标函数的地址到寄存器 x8
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         // 分支并链接到无重定向的 x8
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         // 恢复栈
         VALGRIND_RESTORE_STACK                                   \
         // 将 x0 寄存器的值移动到 _res
         "mov %0, x0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      // 将 _res 转换为 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      // 定义一个指向原始函数的变量 _orig，并强制转换为 volatile 类型
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个长度为 6 的 unsigned long 类型数组 _argvec，用于存放函数参数
      volatile unsigned long _argvec[6];                          \
      // 定义一个 unsigned long 类型的变量 _res，用于存放函数调用的返回值
      volatile unsigned long _res;                                \
      // 将原始函数的地址存入 _argvec 数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将函数的参数依次存入 _argvec 数组的后续位置
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      // 使用内联汇编执行以下操作：
      // 加载 _argvec 数组中的第二到第六个元素到寄存器 x0 到 x4
      // 加载 _argvec 数组中的第一个元素的值作为目标寄存器 x8 的值
      // 执行汇编代码中的栈对齐、跳转和恢复栈等操作
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr x0, [%1, #8] \n\t"                                  \
         "ldr x1, [%1, #16] \n\t"                                 \
         "ldr x2, [%1, #24] \n\t"                                 \
         "ldr x3, [%1, #32] \n\t"                                 \
         "ldr x4, [%1, #40] \n\t"                                 \
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, x0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      // 将 _res 的值转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
// 定义一个宏，用于调用具有6个参数的函数，将结果存储在 lval 中，使用 _orig 作为函数指针
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[7];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr x0, [%1, #8] \n\t"                                  \
         "ldr x1, [%1, #16] \n\t"                                 \
         "ldr x2, [%1, #24] \n\t"                                 \
         "ldr x3, [%1, #32] \n\t"                                 \
         "ldr x4, [%1, #40] \n\t"                                 \
         "ldr x5, [%1, #48] \n\t"                                 \
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, x0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   定义一个宏，用于调用带7个参数的函数，并将结果存储在指定的左值变量中
   @param lval: 左值变量，用于接收函数调用的结果
   @param orig: 原始函数指针或地址
   @param arg1-arg7: 函数调用的7个参数
*/
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7)                            \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[8];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "ldr x0, [%1, #8] \n\t"                                  \
         "ldr x1, [%1, #16] \n\t"                                 \
         "ldr x2, [%1, #24] \n\t"                                 \
         "ldr x3, [%1, #32] \n\t"                                 \
         "ldr x4, [%1, #40] \n\t"                                 \
         "ldr x5, [%1, #48] \n\t"                                 \
         "ldr x6, [%1, #56] \n\t"                                 \
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, x0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
// 宏定义：CALL_FN_W_8W，用于调用具有8个参数的原始函数
#define CALL_FN_W_8W(lval, orig, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) \
   do {                                                                           \
      // 声明并初始化原始函数指针_orig，并标记为volatile
      volatile OrigFn _orig = (orig);                                             \
      // 声明并初始化参数数组 _argvec，包含9个元素
      volatile unsigned long _argvec[9];                                          \
      // 声明并初始化结果变量 _res
      volatile unsigned long _res;                                                \
      // 将原始函数地址存入参数数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                                   \
      // 将参数1至参数8分别存入参数数组的后续位置
      _argvec[1] = (unsigned long)(arg1);                                         \
      _argvec[2] = (unsigned long)(arg2);                                         \
      _argvec[3] = (unsigned long)(arg3);                                         \
      _argvec[4] = (unsigned long)(arg4);                                         \
      _argvec[5] = (unsigned long)(arg5);                                         \
      _argvec[6] = (unsigned long)(arg6);                                         \
      _argvec[7] = (unsigned long)(arg7);                                         \
      _argvec[8] = (unsigned long)(arg8);                                         \
      // 内联汇编：加载参数数组的值到寄存器x0至x8，并执行函数调用
      __asm__ volatile(                                                           \
         VALGRIND_ALIGN_STACK                                                     \
         "ldr x0, [%1, #8] \n\t"                                                  \
         "ldr x1, [%1, #16] \n\t"                                                 \
         "ldr x2, [%1, #24] \n\t"                                                 \
         "ldr x3, [%1, #32] \n\t"                                                 \
         "ldr x4, [%1, #40] \n\t"                                                 \
         "ldr x5, [%1, #48] \n\t"                                                 \
         "ldr x6, [%1, #56] \n\t"                                                 \
         "ldr x7, [%1, #64] \n\t"                                                 \
         "ldr x8, [%1] \n\t"  /* target->x8 */                                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                                   \
         VALGRIND_RESTORE_STACK                                                   \
         "mov %0, x0"                                                             \
         : /*out*/   "=r" (_res)                                                  \
         : /*in*/    "0" (&_argvec[0])                                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"                   \
      );                                                                          \
      // 将返回值赋给左值变量 lval，并强制类型转换为 lval 的类型
      lval = (__typeof__(lval)) _res;                                             \
   } while (0)
/*
   宏定义：CALL_FN_W_9W

   该宏用于调用具有9个参数的函数，并通过汇编内联代码实现。下面是具体的步骤和解释：

   - volatile OrigFn _orig = (orig);
     声明一个 volatile 类型的变量 _orig，用于存储函数指针 orig 的值。

   - volatile unsigned long _argvec[10];
     声明一个 volatile 类型的无符号长整型数组 _argvec，用于存储函数调用时的参数，包括函数指针和其余9个参数。

   - volatile unsigned long _res;
     声明一个 volatile 类型的无符号长整型变量 _res，用于存储函数调用的返回值。

   - _argvec[0] = (unsigned long)_orig.nraddr;
     将函数指针 _orig 的 nraddr 成员赋值给 _argvec[0]，即函数地址。

   - _argvec[1] = (unsigned long)(arg1);
     将第一个函数参数 arg1 转换为无符号长整型后赋值给 _argvec[1]。

   - _argvec[2] = (unsigned long)(arg2);
     将第二个函数参数 arg2 转换为无符号长整型后赋值给 _argvec[2]。

   - _argvec[3] = (unsigned long)(arg3);
     将第三个函数参数 arg3 转换为无符号长整型后赋值给 _argvec[3]。

   - _argvec[4] = (unsigned long)(arg4);
     将第四个函数参数 arg4 转换为无符号长整型后赋值给 _argvec[4]。

   - _argvec[5] = (unsigned long)(arg5);
     将第五个函数参数 arg5 转换为无符号长整型后赋值给 _argvec[5]。

   - _argvec[6] = (unsigned long)(arg6);
     将第六个函数参数 arg6 转换为无符号长整型后赋值给 _argvec[6]。

   - _argvec[7] = (unsigned long)(arg7);
     将第七个函数参数 arg7 转换为无符号长整型后赋值给 _argvec[7]。

   - _argvec[8] = (unsigned long)(arg8);
     将第八个函数参数 arg8 转换为无符号长整型后赋值给 _argvec[8]。

   - _argvec[9] = (unsigned long)(arg9);
     将第九个函数参数 arg9 转换为无符号长整型后赋值给 _argvec[9]。

   - __asm__ volatile(
       VALGRIND_ALIGN_STACK
       "sub sp, sp, #0x20 \n\t"
       "ldr x0, [%1, #8] \n\t"
       "ldr x1, [%1, #16] \n\t"
       "ldr x2, [%1, #24] \n\t"
       "ldr x3, [%1, #32] \n\t"
       "ldr x4, [%1, #40] \n\t"
       "ldr x5, [%1, #48] \n\t"
       "ldr x6, [%1, #56] \n\t"
       "ldr x7, [%1, #64] \n\t"
       "ldr x8, [%1, #72] \n\t"
       "str x8, [sp, #0]  \n\t"
       "ldr x8, [%1] \n\t"  /* target->x8 */
       VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8
       VALGRIND_RESTORE_STACK
       "mov %0, x0"
       : /*out*/   "=r" (_res)
       : /*in*/    "0" (&_argvec[0])
       : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"
     );
     使用内联汇编来调用函数：
     - VALGRIND_ALIGN_STACK: 对齐栈
     - "sub sp, sp, #0x20 \n\t": 减小栈指针，为了栈帧的空间
     - "ldr x0, [%1, #8] \n\t" 到 "ldr x8, [%1] \n\t"：加载参数到寄存器 x0 到 x8
     - "str x8, [sp, #0]  \n\t"：将寄存器 x8 的值保存到栈中
     Returns
#define CALL_FN_W_10W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,  \
                                  arg7,arg8,arg9,arg10)           \
   do {                                                           \
      // 声明一个 volatile 原始函数指针 _orig，用于保存传入的 orig 参数
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个包含11个元素的 volatile unsigned long 数组 _argvec，用于存放参数
      volatile unsigned long _argvec[11];                         \
      // 声明一个 volatile unsigned long 变量 _res，用于保存函数调用的返回值
      volatile unsigned long _res;                                \
      // 将 orig 参数的 nraddr 成员赋给 _argvec[0]，即函数地址
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将 arg1 到 arg10 的值依次存放到 _argvec 数组的后续元素中
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      _argvec[9] = (unsigned long)(arg9);                         \
      _argvec[10] = (unsigned long)(arg10);                       \
      // 使用内联汇编调用函数，传入参数 _argvec 并接收返回值到 _res
      __asm__ volatile(                                           \
         VALGRIND_ALIGN_STACK                                     \
         "sub sp, sp, #0x20 \n\t"                                 \
         "ldr x0, [%1, #8] \n\t"                                  \
         "ldr x1, [%1, #16] \n\t"                                 \
         "ldr x2, [%1, #24] \n\t"                                 \
         "ldr x3, [%1, #32] \n\t"                                 \
         "ldr x4, [%1, #40] \n\t"                                 \
         "ldr x5, [%1, #48] \n\t"                                 \
         "ldr x6, [%1, #56] \n\t"                                 \
         "ldr x7, [%1, #64] \n\t"                                 \
         "ldr x8, [%1, #72] \n\t"                                 \
         "str x8, [sp, #0]  \n\t"                                 \
         "ldr x8, [%1, #80] \n\t"                                 \
         "str x8, [sp, #8]  \n\t"                                 \
         "ldr x8, [%1] \n\t"  /* target->x8 */                    \
         VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                   \
         VALGRIND_RESTORE_STACK                                   \
         "mov %0, x0"                                             \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS, "x21"   \
      );                                                          \
      // 将 _res 转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/* 
   Similar workaround as amd64 (see above), but we use r11 as frame
   pointer and save the old r11 in r7. r11 might be used for
   argvec, therefore we copy argvec in r1 since r1 is clobbered
   after the call anyway.
*/
#if defined(__GNUC__) && defined(__GCC_HAVE_DWARF2_CFI_ASM)
#  define __FRAME_POINTER                                         \
      ,"d"(__builtin_dwarf_cfa())
#  define VALGRIND_CFI_PROLOGUE                                   \
      ".cfi_remember_state\n\t"                                   \
      "lgr 1,%1\n\t" /* copy the argvec pointer in r1 */          \
      "lgr 7,11\n\t"                                              \
      "lgr 11,%2\n\t"                                             \
      ".cfi_def_cfa r11, 0\n\t"                                   \
      /* Save state for Call Frame Information (CFI) */            \
      /* Copy argvec pointer into r1, save r11, and define CFA */  \
      /* where r11 points to the current frame */                 
#  define VALGRIND_CFI_EPILOGUE                                   \
      "lgr 11, 7\n\t"                                             \
      ".cfi_restore_state\n\t"                                    \
      /* Restore the state for CFI and r11 */                     
#else
#  define __FRAME_POINTER
#  define VALGRIND_CFI_PROLOGUE                                   \
      "lgr 1,%1\n\t"                                              \
      /* Copy the argvec pointer into r1 */                       
#  define VALGRIND_CFI_EPILOGUE
#endif

/* 
   Nb: On s390 the stack pointer is properly aligned *at all times*
   according to the s390 GCC maintainer. (The ABI specification is not
   precise in this regard.) Therefore, VALGRIND_ALIGN_STACK and
   VALGRIND_RESTORE_STACK are not defined here.
*/

/* 
   These regs are trashed by the hidden call. Note that we overwrite
   r14 in s390_irgen_noredir (VEX/priv/guest_s390_irgen.c) to give the
   function a proper return address. All others are ABI defined call
   clobbers.
*/
#if defined(__VX__) || defined(__S390_VX__)
#define __CALLER_SAVED_REGS "0", "1", "2", "3", "4", "5", "14",   \
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",             \
      "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",       \
      "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",     \
      "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
#else
#define __CALLER_SAVED_REGS "0", "1", "2", "3", "4", "5", "14",   \
      "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"
#endif

/* 
   Nb: Although r11 is modified in the asm snippets below (inside 
   VALGRIND_CFI_PROLOGUE) it is not listed in the clobber section, for
   two reasons:
   (1) r11 is restored in VALGRIND_CFI_EPILOGUE, so effectively it is not
       modified
   (2) GCC will complain that r11 cannot appear inside a clobber section,
       when compiled with -O -fno-omit-frame-pointer
 */
#define CALL_FN_W_v(lval, orig)                                  \
   do {                                                          \
      volatile OrigFn        _orig = (orig);                     \
      volatile unsigned long  _argvec[1];                        \
      volatile unsigned long _res;                               \
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      // 设置参数向量，包含函数指针的地址
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-160\n\t"                                      \
         "lg 1, 0(1)\n\t"  /* target->r1 */                      \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,160\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "d" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"7"     \
      );                                                         \
      // 将返回值赋给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)

/* The call abi has the arguments in r2-r6 and stack */
#define CALL_FN_W_W(lval, orig, arg1)                            \
   do {                                                          \
      volatile OrigFn        _orig = (orig);                     \
      volatile unsigned long _argvec[2];                         \
      volatile unsigned long _res;                               \
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      _argvec[1] = (unsigned long)arg1;                          \
      // 设置参数向量，包含函数指针的地址和第一个参数
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-160\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,160\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"7"     \
      );                                                         \
      // 将返回值赋给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
#define CALL_FN_W_WW(lval, orig, arg1, arg2)                     \
   do {                                                          \
      // 声明 _orig 变量，并将传入的 orig 参数赋值给 _orig
      volatile OrigFn        _orig = (orig);                     \
      // 声明 _argvec 数组，用于保存参数和返回结果
      volatile unsigned long _argvec[3];                         \
      // 声明 _res 变量，用于保存函数调用的返回值
      volatile unsigned long _res;                               \
      // 将 _orig.nraddr 赋值给 _argvec[0]，将 arg1 赋值给 _argvec[1]，将 arg2 赋值给 _argvec[2]
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      // 使用内联汇编执行以下操作：
      // 设置 VALGRIND 的调用前插入代码
      // 设置地址寄存器 2 为 _argvec[1]
      // 设置地址寄存器 3 为 _argvec[2]
      // 设置地址寄存器 1 为 _argvec[0]
      // 执行 VALGRIND 的调用宏 VALGRIND_CALL_NOREDIR_R1
      // 设置 VALGRIND 的调用后插入代码
      // 将寄存器 2 的内容移动到 _res 变量中
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-160\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,160\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"7"     \
      );                                                         \
      // 将 _res 转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
/*
   宏定义 CALL_FN_W_WWW 用于调用函数，并处理参数传递和返回值。
   参数说明：
     lval: 调用函数返回的值将存储在此变量中
     orig: 原始函数指针
     arg1, arg2, arg3: 调用函数的参数

   注意事项：
     - volatile 修饰符用于确保编译器不会优化相关变量
     - 使用汇编内联指令直接进行函数调用
     - VALGRIND_CFI_PROLOGUE 和 VALGRIND_CFI_EPILOGUE 是 Valgrind 工具的调用指令序列，用于调用框架一致性检查
     - __asm__ volatile( ... ) 内联汇编语句用于执行实际的函数调用
     - 最终调用结果存储在 lval 中，其类型与函数返回类型相同
*/
#define CALL_FN_W_WWW(lval, orig, arg1, arg2, arg3)              \
   do {                                                          \
      volatile OrigFn        _orig = (orig);                     \
      volatile unsigned long _argvec[4];                         \
      volatile unsigned long _res;                               \
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-160\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,160\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"7"     \
      );                                                         \
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
/*
   定义一个宏，用于调用带有5个参数的函数，并将结果存储到指定的左值变量中

   参数说明：
   lval: 左值变量，用于存储函数调用的结果
   orig: 原始函数指针，指向要调用的函数
   arg1, arg2, arg3, arg4: 函数的四个参数

   实现细节：
   - 声明一个 volatile 类型的原始函数指针 _orig，用于保存传入的 orig 参数
   - 声明一个 volatile 类型的无符号长整型数组 _argvec[5]，用于保存函数调用时的参数
   - 声明一个 volatile 类型的无符号长整型变量 _res，用于保存函数调用的结果
   - 将函数原始地址和四个参数依次保存到 _argvec 数组中
   - 使用内联汇编 __asm__ volatile(...) 执行函数调用的汇编代码：
     - 将参数加载到寄存器中
     - 执行函数调用
     - 将返回值保存到 _res 中
   - 将 _res 强制类型转换为宏定义中指定的左值类型 lval，并赋值给 lval

   注意事项：
   - 使用 do { ... } while (0) 结构，确保宏定义的语义正确性和安全性
   - 汇编代码中使用了特定的 VALGRIND_CFI_PROLOGUE 和 VALGRIND_CFI_EPILOGUE 宏定义，可能用于性能分析或调试
   - __FRAME_POINTER 表示在编译时帧指针优化的开启状态
   - 使用了多个约束符号和修饰符以确保代码的正确性和性能

   参考资料：
   - https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
   - https://stackoverflow.com/questions/2161803/explain-volatile-variable-in-c-language
   - https://valgrind.org/docs/manual/manual-core-adv.html#manual-core-adv.clientrequests
*/
#define CALL_FN_W_WWWW(lval, orig, arg1, arg2, arg3, arg4)       \
   do {                                                          \
      volatile OrigFn        _orig = (orig);                     \
      volatile unsigned long _argvec[5];                         \
      volatile unsigned long _res;                               \
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-160\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,160\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"7"     \
      );                                                         \
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
#define CALL_FN_W_5W(lval, orig, arg1, arg2, arg3, arg4, arg5)   \
   do {                                                          \
      // 声明一个指向原始函数地址的变量 _orig，并将参数 orig 赋给它
      volatile OrigFn        _orig = (orig);                     \
      // 声明一个包含6个元素的无符号长整型数组 _argvec，用于存储函数调用的参数
      volatile unsigned long _argvec[6];                         \
      // 声明一个无符号长整型变量 _res，用于存储函数调用的结果
      volatile unsigned long _res;                               \
      // 将原始函数地址的指针存入 _argvec 数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      // 将函数调用的参数依次存入 _argvec 数组的后续位置
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      _argvec[5] = (unsigned long)arg5;                          \
      // 使用内联汇编进行函数调用
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-160\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 6,40(1)\n\t"                                        \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,160\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         // 将函数调用的结果存入 _res 变量
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"6","7" \
      );                                                         \
      // 将 _res 强制类型转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
/*
   定义一个宏，用于调用具有六个参数的函数。

   @param lval      - 用于存储函数调用结果的左值
   @param orig      - 原始函数指针或地址
   @param arg1      - 第一个函数参数
   @param arg2      - 第二个函数参数
   @param arg3      - 第三个函数参数
   @param arg4      - 第四个函数参数
   @param arg5      - 第五个函数参数
   @param arg6      - 第六个函数参数
*/
#define CALL_FN_W_6W(lval, orig, arg1, arg2, arg3, arg4, arg5,   \
                     arg6)                                       \
   do {                                                          \
      // 声明并初始化临时变量 _orig，用于保存原始函数地址
      volatile OrigFn        _orig = (orig);                     \
      // 声明并初始化存储函数参数的数组 _argvec
      volatile unsigned long _argvec[7];                         \
      // 声明并初始化用于存储函数调用结果的变量 _res
      volatile unsigned long _res;                               \
      // 将函数指针或地址存入参数数组 _argvec 的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      // 将函数的六个参数依次存入参数数组 _argvec
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      _argvec[5] = (unsigned long)arg5;                          \
      _argvec[6] = (unsigned long)arg6;                          \
      // 使用内联汇编执行函数调用，调用方式依赖于特定架构和平台的约定
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-168\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 6,40(1)\n\t"                                        \
         "mvc 160(8,15), 48(1)\n\t"                              \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,168\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"6","7" \
      );                                                         \
      // 将函数调用结果转换为目标类型，并赋给左值 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
/*
   宏定义：CALL_FN_W_7W(lval, orig, arg1, arg2, arg3, arg4, arg5, arg6, arg7)

   这个宏用于调用带有7个参数的函数指针，并将结果保存在变量 lval 中。

   参数说明：
   - lval: 保存函数调用结果的变量
   - orig: 原始函数指针
   - arg1 to arg7: 函数调用时的7个参数

   实现细节：
   - 声明 _orig 变量用于存储原始函数指针
   - 声明 _argvec 数组用于存储参数，包括原始函数指针和7个参数
   - 声明 _res 变量用于存储函数调用的返回值
   - 使用内联汇编语句来执行函数调用，具体操作如下：
     1. 设置寄存器值，准备调用指定的函数
     2. 执行函数调用，传递参数并接收返回值
     3. 处理函数返回结果，并将其赋给 lval
*/

#define CALL_FN_W_7W(lval, orig, arg1, arg2, arg3, arg4, arg5,   \
                     arg6, arg7)                                 \
   do {                                                          \
      volatile OrigFn        _orig = (orig);                     \
      volatile unsigned long _argvec[8];                         \
      volatile unsigned long _res;                               \
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      _argvec[5] = (unsigned long)arg5;                          \
      _argvec[6] = (unsigned long)arg6;                          \
      _argvec[7] = (unsigned long)arg7;                          \
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-176\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 6,40(1)\n\t"                                        \
         "mvc 160(8,15), 48(1)\n\t"                              \
         "mvc 168(8,15), 56(1)\n\t"                              \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,176\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"6","7" \
      );                                                         \
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
#define CALL_FN_W_8W(lval, orig, arg1, arg2, arg3, arg4, arg5,   \
                     arg6, arg7 ,arg8)                           \
   do {                                                          \
      // 声明 volatile 变量 _orig 并初始化为传入的函数指针 orig
      volatile OrigFn        _orig = (orig);                     \
      // 声明一个数组 _argvec，用于存储函数调用的参数，共9个元素
      volatile unsigned long _argvec[9];                         \
      // 声明一个 volatile unsigned long 类型变量 _res，用于存储函数调用的结果
      volatile unsigned long _res;                               \
      // 将函数指针 _orig 的地址存入 _argvec 数组的第一个元素
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      // 将函数调用的参数 arg1 到 arg8 依次存入 _argvec 数组的后续元素
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      _argvec[5] = (unsigned long)arg5;                          \
      _argvec[6] = (unsigned long)arg6;                          \
      _argvec[7] = (unsigned long)arg7;                          \
      _argvec[8] = (unsigned long)arg8;                          \
      // 使用内联汇编执行具体的函数调用过程
      __asm__ volatile(                                          \
         // 插入 Valgrind CFI 的函数调用前插入代码
         VALGRIND_CFI_PROLOGUE                                   \
         // 将第一个参数加载到寄存器 2
         "aghi 15,-184\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 6,40(1)\n\t"                                        \
         // 依次复制参数数据到寄存器 15 的指定内存偏移处
         "mvc 160(8,15), 48(1)\n\t"                              \
         "mvc 168(8,15), 56(1)\n\t"                              \
         "mvc 176(8,15), 64(1)\n\t"                              \
         // 将函数指针加载到寄存器 1
         "lg 1, 0(1)\n\t"                                        \
         // 插入 Valgrind 的函数调用指令
         VALGRIND_CALL_NOREDIR_R1                                \
         // 恢复寄存器 15 的偏移量
         "aghi 15,184\n\t"                                       \
         // 插入 Valgrind CFI 的函数调用后恢复代码
         VALGRIND_CFI_EPILOGUE                                   \
         // 将寄存器 2 的值传递给 _res 变量
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"6","7" \
      );                                                         \
      // 将 _res 强制转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
#define CALL_FN_W_9W(lval, orig, arg1, arg2, arg3, arg4, arg5,   \
                     arg6, arg7 ,arg8, arg9)                     \
   do {                                                          \
      // 定义一个指向原始函数的变量 _orig，并将传入的 orig 参数赋给它
      volatile OrigFn        _orig = (orig);                     \
      // 定义一个长度为 10 的无符号长整型数组 _argvec，用于存放函数调用的参数
      volatile unsigned long _argvec[10];                        \
      // 定义一个无符号长整型变量 _res，用于存放函数调用的结果
      volatile unsigned long _res;                               \
      // 将各个参数依次转换为无符号长整型，并存放在 _argvec 数组中
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      _argvec[5] = (unsigned long)arg5;                          \
      _argvec[6] = (unsigned long)arg6;                          \
      _argvec[7] = (unsigned long)arg7;                          \
      _argvec[8] = (unsigned long)arg8;                          \
      _argvec[9] = (unsigned long)arg9;                          \
      // 内嵌汇编代码块，用于调用函数，包括参数传递和返回值处理
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-192\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 6,40(1)\n\t"                                        \
         "mvc 160(8,15), 48(1)\n\t"                              \
         "mvc 168(8,15), 56(1)\n\t"                              \
         "mvc 176(8,15), 64(1)\n\t"                              \
         "mvc 184(8,15), 72(1)\n\t"                              \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,192\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"6","7" \
      );                                                         \
      // 将 _res 的值转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
#define CALL_FN_W_10W(lval, orig, arg1, arg2, arg3, arg4, arg5,  \
                     arg6, arg7 ,arg8, arg9, arg10)              \
   do {                                                          \
      // 声明 _orig 变量，并将原始函数指针赋给它
      volatile OrigFn        _orig = (orig);                     \
      // 声明 _argvec 数组，用于存储函数调用的参数
      volatile unsigned long _argvec[11];                        \
      // 声明 _res 变量，用于存储函数调用的结果
      volatile unsigned long _res;                               \
      // 将函数指针的地址作为第一个参数存入参数数组
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      // 将后续的十个参数依次存入参数数组
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      _argvec[5] = (unsigned long)arg5;                          \
      _argvec[6] = (unsigned long)arg6;                          \
      _argvec[7] = (unsigned long)arg7;                          \
      _argvec[8] = (unsigned long)arg8;                          \
      _argvec[9] = (unsigned long)arg9;                          \
      _argvec[10] = (unsigned long)arg10;                        \
      // 使用内联汇编执行函数调用
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-200\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 6,40(1)\n\t"                                        \
         "mvc 160(8,15), 48(1)\n\t"                              \
         "mvc 168(8,15), 56(1)\n\t"                              \
         "mvc 176(8,15), 64(1)\n\t"                              \
         "mvc 184(8,15), 72(1)\n\t"                              \
         "mvc 192(8,15), 80(1)\n\t"                              \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,200\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"6","7" \
      );                                                         \
      // 将函数调用结果转换为目标类型并赋给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
#define CALL_FN_W_11W(lval, orig, arg1, arg2, arg3, arg4, arg5,  \
                     arg6, arg7 ,arg8, arg9, arg10, arg11)       \
   do {                                                          \
      // 定义一个指向原始函数的指针 _orig，并将传入的 orig 参数赋给它
      volatile OrigFn        _orig = (orig);                     \
      // 定义一个长度为 12 的无符号长整型数组 _argvec，用来存放函数参数
      volatile unsigned long _argvec[12];                        \
      // 定义一个无符号长整型变量 _res，用于存放函数调用的返回值
      volatile unsigned long _res;                               \
      // 将函数指针 _orig 的地址存入 _argvec 数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      // 将参数 arg1 到 arg11 依次存入 _argvec 数组
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      _argvec[5] = (unsigned long)arg5;                          \
      _argvec[6] = (unsigned long)arg6;                          \
      _argvec[7] = (unsigned long)arg7;                          \
      _argvec[8] = (unsigned long)arg8;                          \
      _argvec[9] = (unsigned long)arg9;                          \
      _argvec[10] = (unsigned long)arg10;                        \
      _argvec[11] = (unsigned long)arg11;                        \
      // 使用内联汇编执行以下操作
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-208\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 6,40(1)\n\t"                                        \
         "mvc 160(8,15), 48(1)\n\t"                              \
         "mvc 168(8,15), 56(1)\n\t"                              \
         "mvc 176(8,15), 64(1)\n\t"                              \
         "mvc 184(8,15), 72(1)\n\t"                              \
         "mvc 192(8,15), 80(1)\n\t"                              \
         "mvc 200(8,15), 88(1)\n\t"                              \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,208\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"6","7" \
      );                                                         \
      // 将 _res 转换为目标类型并赋给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)
#define CALL_FN_W_12W(lval, orig, arg1, arg2, arg3, arg4, arg5,  \
                     arg6, arg7 ,arg8, arg9, arg10, arg11, arg12)\
   do {                                                          \
      // 定义一个指向原始函数的指针 _orig，并将 orig 参数赋给它
      volatile OrigFn        _orig = (orig);                     \
      // 定义一个包含13个元素的无符号长整型数组 _argvec，用于存放函数调用的参数
      volatile unsigned long _argvec[13];                        \
      // 定义一个无符号长整型变量 _res，用于存放函数调用的返回值
      volatile unsigned long _res;                               \
      // 将参数赋值给 _argvec 数组的各个元素，注意将指针或值强制转换为 unsigned long
      _argvec[0] = (unsigned long)_orig.nraddr;                  \
      _argvec[1] = (unsigned long)arg1;                          \
      _argvec[2] = (unsigned long)arg2;                          \
      _argvec[3] = (unsigned long)arg3;                          \
      _argvec[4] = (unsigned long)arg4;                          \
      _argvec[5] = (unsigned long)arg5;                          \
      _argvec[6] = (unsigned long)arg6;                          \
      _argvec[7] = (unsigned long)arg7;                          \
      _argvec[8] = (unsigned long)arg8;                          \
      _argvec[9] = (unsigned long)arg9;                          \
      _argvec[10] = (unsigned long)arg10;                        \
      _argvec[11] = (unsigned long)arg11;                        \
      _argvec[12] = (unsigned long)arg12;                        \
      // 使用内联汇编调用指令集，执行函数调用过程
      __asm__ volatile(                                          \
         VALGRIND_CFI_PROLOGUE                                   \
         "aghi 15,-216\n\t"                                      \
         "lg 2, 8(1)\n\t"                                        \
         "lg 3,16(1)\n\t"                                        \
         "lg 4,24(1)\n\t"                                        \
         "lg 5,32(1)\n\t"                                        \
         "lg 6,40(1)\n\t"                                        \
         "mvc 160(8,15), 48(1)\n\t"                              \
         "mvc 168(8,15), 56(1)\n\t"                              \
         "mvc 176(8,15), 64(1)\n\t"                              \
         "mvc 184(8,15), 72(1)\n\t"                              \
         "mvc 192(8,15), 80(1)\n\t"                              \
         "mvc 200(8,15), 88(1)\n\t"                              \
         "mvc 208(8,15), 96(1)\n\t"                              \
         "lg 1, 0(1)\n\t"                                        \
         VALGRIND_CALL_NOREDIR_R1                                \
         "aghi 15,216\n\t"                                       \
         VALGRIND_CFI_EPILOGUE                                   \
         "lgr %0, 2\n\t"                                         \
         : /*out*/   "=d" (_res)                                 \
         : /*in*/    "a" (&_argvec[0]) __FRAME_POINTER           \
         : /*trash*/ "cc", "memory", __CALLER_SAVED_REGS,"6","7" \
      );                                                         \
      // 将 _res 转换为 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) _res;                            \
   } while (0)

#endif /* PLAT_s390x_linux */
/* ------------------------- mips32-linux ----------------------- */

#if defined(PLAT_mips32_linux)

/* These regs are trashed by the hidden call. */
// 定义被隐藏调用破坏的寄存器列表
#define __CALLER_SAVED_REGS "$2", "$3", "$4", "$5", "$6",       \
"$7", "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "$24", \
"$25", "$31"

/* These CALL_FN_ macros assume that on mips-linux, sizeof(unsigned
   long) == 4. */
// CALL_FN_W_v 宏的定义，假设在 mips-linux 平台上，sizeof(unsigned long) == 4
#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[1];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      __asm__ volatile(                                           \
         "subu $29, $29, 8 \n\t"                                  \
         "sw $28, 0($29) \n\t"                                    \
         "sw $31, 4($29) \n\t"                                    \
         "subu $29, $29, 16 \n\t"                                 \
         "lw $25, 0(%1) \n\t"  /* target->t9 */                   \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "addu $29, $29, 16\n\t"                                  \
         "lw $28, 0($29) \n\t"                                    \
         "lw $31, 4($29) \n\t"                                    \
         "addu $29, $29, 8 \n\t"                                  \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_W(lval, orig, arg1)

   lval: 保存函数调用结果的变量
   orig: 原始函数指针
   arg1: 第一个参数

   使用内联汇编实现函数调用，并将结果存储在lval中。

   注意事项：
   - volatile关键字用于确保编译器不会对_orig、_argvec、_res进行优化。
   - 内联汇编语句使用了特定的寄存器约束和内存约束。
   - 在调用前后进行了寄存器的保存和恢复。
   - VALGRIND_CALL_NOREDIR_T9是一个宏定义，用于处理Valgrind工具的相关调用。
   - 使用了输入、输出和破坏的约束来告诉编译器内联汇编的影响。
*/
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[2];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      __asm__ volatile(                                           \
         "subu $29, $29, 8 \n\t"                                  \
         "sw $28, 0($29) \n\t"                                    \
         "sw $31, 4($29) \n\t"                                    \
         "subu $29, $29, 16 \n\t"                                 \
         "lw $4, 4(%1) \n\t"   /* arg1*/                          \
         "lw $25, 0(%1) \n\t"  /* target->t9 */                   \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "addu $29, $29, 16 \n\t"                                 \
         "lw $28, 0($29) \n\t"                                    \
         "lw $31, 4($29) \n\t"                                    \
         "addu $29, $29, 8 \n\t"                                  \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "memory",  __CALLER_SAVED_REGS               \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_WW(lval, orig, arg1,arg2)                       \
   do {                                                           \
      // 声明 _orig 变量，并赋值为传入的 orig 参数
      volatile OrigFn        _orig = (orig);                      \
      // 声明 _argvec 数组，用于存储参数和结果
      volatile unsigned long _argvec[3];                          \
      // 声明 _res 变量，用于存储函数调用的结果
      volatile unsigned long _res;                                \
      // 将函数指针 _orig.nraddr 转换为无符号长整型，存入 _argvec 数组的第一个位置
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将 arg1 转换为无符号长整型，存入 _argvec 数组的第二个位置
      _argvec[1] = (unsigned long)(arg1);                         \
      // 将 arg2 转换为无符号长整型，存入 _argvec 数组的第三个位置
      _argvec[2] = (unsigned long)(arg2);                         \
      // 使用内联汇编执行函数调用
      __asm__ volatile(                                           \
         "subu $29, $29, 8 \n\t"                                  \
         "sw $28, 0($29) \n\t"                                    \
         "sw $31, 4($29) \n\t"                                    \
         "subu $29, $29, 16 \n\t"                                 \
         "lw $4, 4(%1) \n\t"                                      \
         "lw $5, 8(%1) \n\t"                                      \
         "lw $25, 0(%1) \n\t"  /* target->t9 */                   \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "addu $29, $29, 16 \n\t"                                 \
         "lw $28, 0($29) \n\t"                                    \
         "lw $31, 4($29) \n\t"                                    \
         "addu $29, $29, 8 \n\t"                                  \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将函数调用的结果 _res 转换为 lval 变量的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      // 定义一个指向原始函数的指针 _orig，并将参数 arg1, arg2, arg3 包装成一个数组 _argvec
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[4];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      // 使用内联汇编调用原始函数
      __asm__ volatile(                                           \
         "subu $29, $29, 8 \n\t"                                  \
         "sw $28, 0($29) \n\t"                                    \
         "sw $31, 4($29) \n\t"                                    \
         "subu $29, $29, 16 \n\t"                                 \
         "lw $4, 4(%1) \n\t"                                      \
         "lw $5, 8(%1) \n\t"                                      \
         "lw $6, 12(%1) \n\t"                                     \
         "lw $25, 0(%1) \n\t"  /* target->t9 */                   \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "addu $29, $29, 16 \n\t"                                 \
         "lw $28, 0($29) \n\t"                                    \
         "lw $31, 4($29) \n\t"                                    \
         "addu $29, $29, 8 \n\t"                                  \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将结果 _res 转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)

   该宏用于调用一个函数，并将结果存储在变量 lval 中，接受多达四个参数。

   do {                                                           
      将参数 orig 转换为类型 OrigFn，并声明为 volatile 变量 _orig
      volatile OrigFn        _orig = (orig);                      
      
      声明一个包含五个 unsigned long 元素的数组 _argvec
      volatile unsigned long _argvec[5];                           
      
      声明一个 volatile unsigned long 类型的变量 _res
      volatile unsigned long _res;                                

      将函数指针 _orig.nraddr 转换为 unsigned long 存储在 _argvec[0] 中
      _argvec[0] = (unsigned long)_orig.nraddr;                   
      
      将参数 arg1,arg2,arg3,arg4 分别转换为 unsigned long 存储在 _argvec[1] 到 _argvec[4] 中
      _argvec[1] = (unsigned long)(arg1);                          
      _argvec[2] = (unsigned long)(arg2);                          
      _argvec[3] = (unsigned long)(arg3);                          
      _argvec[4] = (unsigned long)(arg4);                          
      
      内联汇编开始，执行以下操作：
      1. 分配 8 字节空间给栈帧，$29 是栈指针
      2. 保存 $28 寄存器到栈上
      3. 保存 $31 寄存器到栈上
      4. 分配 16 字节空间给栈帧
      5. 从 _argvec[1] 到 _argvec[4] 依次加载数据到 $4 到 $7 寄存器
      6. 加载 _argvec[0] 到 $25 寄存器（target->t9）
      7. VALGRIND_CALL_NOREDIR_T9 是一个宏或函数，调用不重定向 T9
      8. 恢复栈帧
      9. 恢复 $28 和 $31 寄存器
      10. 恢复栈帧
      11. 将结果 $2 寄存器中的值移动到 _res
      12. 将 _res 转换为 lval 的类型并存储在 lval 中
      : /*out*/   "=r" (_res)                                     
         输出操作数约束：输出到 _res 变量中
      : /*in*/    "0" (&_argvec[0])                                
         输入操作数约束：_argvec[0] 放在内存中（输出操作数索引 0）
      : /*trash*/ "memory", __CALLER_SAVED_REGS                    
         破坏列表：内存、调用者保存的寄存器
   );                                                              
   } while (0)
*/
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[6];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      __asm__ volatile(                                           \
         "subu $29, $29, 8 \n\t"                                  \
         "sw $28, 0($29) \n\t"                                    \
         "sw $31, 4($29) \n\t"                                    \
         "lw $4, 20(%1) \n\t"                                     \
         "subu $29, $29, 24\n\t"                                  \
         "sw $4, 16($29) \n\t"                                    \
         "lw $4, 4(%1) \n\t"                                      \
         "lw $5, 8(%1) \n\t"                                      \
         "lw $6, 12(%1) \n\t"                                     \
         "lw $7, 16(%1) \n\t"                                     \
         "lw $25, 0(%1) \n\t"  /* target->t9 */                   \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "addu $29, $29, 24 \n\t"                                 \
         "lw $28, 0($29) \n\t"                                    \
         "lw $31, 4($29) \n\t"                                    \
         "addu $29, $29, 8 \n\t"                                  \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \    // 输出操作数: 将结果 _res 放入寄存器 %0 中
         : /*in*/    "0" (&_argvec[0])                            \    // 输入操作数: 使用 &_argvec[0] 作为输入参数
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \    // 被破坏寄存器: 内存和调用者保存的寄存器
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \    // 将 _res 强制类型转换为 lval 的类型，并赋给 lval
   } while (0)
/*
   宏定义：CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)
   参数：
   - lval: 函数调用的返回值
   - orig: 原始函数指针
   - arg1 ~ arg6: 函数的六个参数

   功能：
   - 将参数和函数指针封装到一个特定的数据结构中，用于传递给汇编代码
   - 执行汇编代码块，通过寄存器和内存操作调用原始函数
   - 将函数调用的返回值赋给lval

   注意：
   - 使用了内联汇编 __asm__ volatile 来直接操作寄存器和内存
   - 使用了 volatile 修饰符以确保编译器不优化相关变量
   - 使用了 do { ... } while (0) 结构来确保宏定义可以安全地被使用在表达式中
*/
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[7];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      __asm__ volatile(                                           \
         "subu $29, $29, 8 \n\t"                                  \
         "sw $28, 0($29) \n\t"                                    \
         "sw $31, 4($29) \n\t"                                    \
         "lw $4, 20(%1) \n\t"                                     \
         "subu $29, $29, 32\n\t"                                  \
         "sw $4, 16($29) \n\t"                                    \
         "lw $4, 24(%1) \n\t"                                     \
         "nop\n\t"                                                \
         "sw $4, 20($29) \n\t"                                    \
         "lw $4, 4(%1) \n\t"                                      \
         "lw $5, 8(%1) \n\t"                                      \
         "lw $6, 12(%1) \n\t"                                     \
         "lw $7, 16(%1) \n\t"                                     \
         "lw $25, 0(%1) \n\t"  /* target->t9 */                   \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "addu $29, $29, 32 \n\t"                                 \
         "lw $28, 0($29) \n\t"                                    \
         "lw $31, 4($29) \n\t"                                    \
         "addu $29, $29, 8 \n\t"                                  \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_7W

   参数说明：
     lval: 函数调用的返回值
     orig: 原始函数指针
     arg1-arg7: 函数调用的参数，共7个

   功能说明：
     这个宏用于在MIPS32 Linux平台上调用函数，并支持7个参数。它通过内联汇编实现了函数调用，具体步骤如下：
     1. 将原始函数指针和参数封装成一个参数向量 _argvec。
     2. 保存当前的 $28 和 $31 寄存器状态。
     3. 将 _argvec 中的内容加载到对应的寄存器中，同时进行栈操作。
     4. 使用内联汇编执行函数调用：
        a. 设置目标函数地址（target->t9）
        b. 执行函数调用，并获取返回值到 $2 寄存器。
     5. 恢复保存的寄存器状态。
     6. 将函数调用的返回值转换成宏定义中 lval 的类型，并赋给 lval。

   注意事项：
     - 此宏对寄存器和内存的使用有详细规定。
     - 代码中有针对特定平台（如 MIP32 Linux）和特定寄存器的设置和恢复。
*/
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7)                            \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[8];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      __asm__ volatile(                                           \
         "subu $29, $29, 8 \n\t"                                  \
         "sw $28, 0($29) \n\t"                                    \
         "sw $31, 4($29) \n\t"                                    \
         "lw $4, 20(%1) \n\t"                                     \
         "subu $29, $29, 32\n\t"                                  \
         "sw $4, 16($29) \n\t"                                    \
         "lw $4, 24(%1) \n\t"                                     \
         "sw $4, 20($29) \n\t"                                    \
         "lw $4, 28(%1) \n\t"                                     \
         "sw $4, 24($29) \n\t"                                    \
         "lw $4, 4(%1) \n\t"                                      \
         "lw $5, 8(%1) \n\t"                                      \
         "lw $6, 12(%1) \n\t"                                     \
         "lw $7, 16(%1) \n\t"                                     \
         "lw $25, 0(%1) \n\t"  /* target->t9 */                   \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "addu $29, $29, 32 \n\t"                                 \
         "lw $28, 0($29) \n\t"                                    \
         "lw $31, 4($29) \n\t"                                    \
         "addu $29, $29, 8 \n\t"                                  \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
#endif /* PLAT_mips32_linux */

/* ------------------------- nanomips-linux -------------------- */

#if defined(PLAT_nanomips_linux)

/* These regs are trashed by the hidden call. */

/*
   平台说明：nanomips-linux

   说明：
     以下部分可能包含与 nanomips-linux 平台相关的特定寄存器和调用规则的定义或注释。
     未给出具体的代码示例，但提到了特定寄存器受隐藏调用影响，这可能需要额外的上下文来理解其含义。
*/
#endif /* PLAT_nanomips_linux */
/*
   定义调用者保存的寄存器列表，这些寄存器在函数调用期间需要保存它们的值
   "$t4", "$t5", "$a0", "$a1", "$a2", "$a3", "$a4", "$a5", "$a6", "$a7", "$t0", "$t1", "$t2", "$t3",
   "$t8", "$t9", "$at"
*/
#define __CALLER_SAVED_REGS "$t4", "$t5", "$a0", "$a1", "$a2", \
"$a3", "$a4", "$a5", "$a6", "$a7", "$t0", "$t1", "$t2", "$t3",     \
"$t8","$t9", "$at"

/*
   这些 CALL_FN_ 宏假设在 mips-linux 环境下，sizeof(unsigned long) == 4。
*/

/*
   定义一个调用函数并返回 void 的宏，lval 为函数返回值，orig 为函数指针。
   使用汇编代码调用函数，将结果存入 lval 中。
*/
#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[1];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      __asm__ volatile(                                           \
         "lw $t9, 0(%1)\n\t"                                      \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0\n"                                         \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

/*
   定义一个调用函数并返回 unsigned long 的宏，lval 为函数返回值，orig 为函数指针，arg1 为函数参数。
   使用汇编代码调用函数，将结果存入 lval 中。
*/
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[2];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      __asm__ volatile(                                           \
         "lw $t9, 0(%1)\n\t"                                      \
         "lw $a0, 4(%1)\n\t"                                      \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0\n"                                         \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
// 定义一个宏，用于调用无返回值、带两个参数的原始函数指针
#define CALL_FN_W_WW(lval, orig, arg1, arg2)                       \
   do {                                                            \
      // 声明并初始化一个指向原始函数的变量 _orig
      volatile OrigFn        _orig = (orig);                       \
      // 声明一个包含三个无符号长整型元素的数组 _argvec
      volatile unsigned long _argvec[3];                           \
      // 声明一个无符号长整型变量 _res，用于存放函数调用的结果
      volatile unsigned long _res;                                 \
      // 将原始函数的地址存入 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                    \
      // 将参数 arg1 和 arg2 转换为无符号长整型，并存入 _argvec[1] 和 _argvec[2]
      _argvec[1] = (unsigned long)(arg1);                          \
      _argvec[2] = (unsigned long)(arg2);                          \
      // 使用内联汇编调用指定的函数，并将结果存入 _res 中
      __asm__ volatile(                                            \
         "lw $t9, 0(%1)\n\t"                                       \
         "lw $a0, 4(%1)\n\t"                                       \
         "lw $a1, 8(%1)\n\t"                                       \
         VALGRIND_CALL_NOREDIR_T9                                  \
         "move %0, $a0\n"                                          \
         : /*out*/   "=r" (_res)                                   \
         : /*in*/    "r" (&_argvec[0])                             \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                 \
      );                                                           \
      // 将 _res 强制转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                              \
   } while (0)

// 定义一个宏，用于调用无返回值、带三个参数的原始函数指针
#define CALL_FN_W_WWW(lval, orig, arg1, arg2, arg3)                \
   do {                                                            \
      // 声明并初始化一个指向原始函数的变量 _orig
      volatile OrigFn        _orig = (orig);                       \
      // 声明一个包含四个无符号长整型元素的数组 _argvec
      volatile unsigned long _argvec[4];                           \
      // 声明一个无符号长整型变量 _res，用于存放函数调用的结果
      volatile unsigned long _res;                                 \
      // 将原始函数的地址存入 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                    \
      // 将参数 arg1、arg2 和 arg3 转换为无符号长整型，并存入 _argvec[1]、_argvec[2] 和 _argvec[3]
      _argvec[1] = (unsigned long)(arg1);                          \
      _argvec[2] = (unsigned long)(arg2);                          \
      _argvec[3] = (unsigned long)(arg3);                          \
      // 使用内联汇编调用指定的函数，并将结果存入 _res 中
      __asm__ volatile(                                            \
         "lw $t9, 0(%1)\n\t"                                       \
         "lw $a0, 4(%1)\n\t"                                       \
         "lw $a1, 8(%1)\n\t"                                       \
         "lw $a2, 12(%1)\n\t"                                      \
         VALGRIND_CALL_NOREDIR_T9                                  \
         "move %0, $a0\n"                                          \
         : /*out*/   "=r" (_res)                                   \
         : /*in*/    "r" (&_argvec[0])                             \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                 \
      );                                                           \
      // 将 _res 强制转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                              \
   } while (0)
# 定义宏 CALL_FN_W_WWWW，用于调用指定函数，并传递四个参数
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)           \
   do {                                                           \
      // 定义 _orig 变量，存储传入的函数指针 (orig)
      volatile OrigFn        _orig = (orig);                      \
      // 定义 _argvec 数组，存储函数调用的参数
      volatile unsigned long _argvec[5];                          \
      // 定义 _res 变量，存储函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数指针的地址存入 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数 arg1 存入 _argvec[1]
      _argvec[1] = (unsigned long)(arg1);                         \
      // 将参数 arg2 存入 _argvec[2]
      _argvec[2] = (unsigned long)(arg2);                         \
      // 将参数 arg3 存入 _argvec[3]
      _argvec[3] = (unsigned long)(arg3);                         \
      // 将参数 arg4 存入 _argvec[4]
      _argvec[4] = (unsigned long)(arg4);                         \
      // 使用内联汇编执行以下指令序列
      __asm__ volatile(                                           \
         "lw $t9, 0(%1)\n\t"                                      \
         "lw $a0, 4(%1)\n\t"                                      \
         "lw $a1, 8(%1)\n\t"                                      \
         "lw $a2,12(%1)\n\t"                                      \
         "lw $a3,16(%1)\n\t"                                      \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0\n"                                         \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将 _res 转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)

   lval: 函数调用的返回值，将被赋值给这个变量
   orig: 原始函数指针，用于调用特定函数
   arg1, arg2, arg3, arg4, arg5: 函数调用的参数

   使用内联汇编实现对指定函数的调用，通过以下步骤：
   1. 将原始函数指针 _orig 赋给 volatile 类型变量 _orig
   2. 准备参数数组 _argvec，包括函数号和五个参数的地址
   3. 使用 volatile 类型变量 _res 存储函数调用的返回值
   4. 内联汇编部分，依次将参数载入寄存器 $a0 - $a4
   5. 执行函数调用，使用 $t9 寄存器存储原始函数地址，VALGRIND_CALL_NOREDIR_T9 可能是另外一个宏的调用
   6. 将函数返回值传递给 _res，并转换为 lval 的类型

   注：该宏用于在内核或低级系统编程中调用函数，并通过汇编语言控制参数传递和函数调用过程。
*/
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[6];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      __asm__ volatile(                                           \
         "lw $t9, 0(%1)\n\t"                                      \
         "lw $a0, 4(%1)\n\t"                                      \
         "lw $a1, 8(%1)\n\t"                                      \
         "lw $a2,12(%1)\n\t"                                      \
         "lw $a3,16(%1)\n\t"                                      \
         "lw $a4,20(%1)\n\t"                                      \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0\n"                                         \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)

   该宏用于调用函数，并将结果赋给lval，使用六个参数arg1到arg6，同时保留原始函数指针orig。

   注意事项：
   - 使用volatile修饰_orig、_argvec、_res，确保编译器不会优化它们。
   - 将orig.nraddr存入_argvec[0]，将参数arg1到arg6依次存入_argvec[1]到_argvec[6]。
   - 使用内联汇编语句执行函数调用，加载各参数到寄存器。
   - VALGRIND_CALL_NOREDIR_T9是一个宏，用于在Valgrind环境中处理特定的函数调用。
   - 将函数调用的返回值保存到_res，并将其转换为lval的类型。

   注：此宏的实现依赖于特定的汇编语法和寄存器约定，用于在嵌入式环境或操作系统内核中进行函数调用。
*/
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[7];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      __asm__ volatile(                                           \
         "lw $t9, 0(%1)\n\t"                                      \
         "lw $a0, 4(%1)\n\t"                                      \
         "lw $a1, 8(%1)\n\t"                                      \
         "lw $a2,12(%1)\n\t"                                      \
         "lw $a3,16(%1)\n\t"                                      \
         "lw $a4,20(%1)\n\t"                                      \
         "lw $a5,24(%1)\n\t"                                      \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0\n"                                         \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   宏定义：CALL_FN_W_7W

   参数说明：
   lval: 返回值
   orig: 原始函数指针
   arg1-arg7: 函数参数

   功能：
   - 使用给定的原始函数指针和参数调用一个函数，将结果存储在lval中。

   实现细节：
   - 声明一个volatile变量_orig用于存储orig指针。
   - 声明一个volatile数组_argvec用于存储参数及原始函数地址。
   - 声明一个volatile变量_res用于存储函数调用的结果。
   - 将orig.nraddr存储在_argvec[0]中，将参数arg1-arg7依次存储在_argvec数组中。
   - 使用内联汇编执行函数调用：
     - 依次加载_argvec中的数据到寄存器$t9、$a0-$a6。
     - 执行VALGRIND_CALL_NOREDIR_T9宏定义（未提供详细说明）。
     - 将函数调用结果移动到_res变量中。
   - 将_res转换为lval的类型并存储在lval中。

   注意事项：
   - 使用volatile关键字确保编译器不会对变量进行优化。
   - 汇编部分直接操作寄存器进行参数传递和函数调用。
   - 常量VALGRIND_CALL_NOREDIR_T9可能在其他部分定义，用于调用时的特定处理。
*/
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7)                            \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long _argvec[8];                          \
      volatile unsigned long _res;                                \
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      __asm__ volatile(                                           \
         "lw $t9, 0(%1)\n\t"                                      \
         "lw $a0, 4(%1)\n\t"                                      \
         "lw $a1, 8(%1)\n\t"                                      \
         "lw $a2,12(%1)\n\t"                                      \
         "lw $a3,16(%1)\n\t"                                      \
         "lw $a4,20(%1)\n\t"                                      \
         "lw $a5,24(%1)\n\t"                                      \
         "lw $a6,28(%1)\n\t"                                      \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0\n"                                         \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 定义一个宏，用于调用具有8个参数的函数
#define CALL_FN_W_8W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8)                       \
   do {                                                           \
      // 声明一个 volatile 类型的指针 _orig，指向传入的函数指针 (orig)
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个长度为9的 volatile unsigned long 数组 _argvec，用于存放参数
      volatile unsigned long _argvec[9];                          \
      // 声明一个 volatile unsigned long 类型的变量 _res，用于存放函数调用的返回值
      volatile unsigned long _res;                                \
      // 将函数指针的地址存入 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将每个参数依次存入 _argvec 数组中
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      // 使用内联汇编执行以下操作：
      // 从 _argvec 数组读取参数，加载到相应的寄存器中
      // 执行 VALGRIND_CALL_NOREDIR_T9 指令（在这里没有给出具体定义）
      // 将 $a0 中的返回值移动到 _res 中
      __asm__ volatile(                                           \
         "lw $t9, 0(%1)\n\t"                                      \
         "lw $a0, 4(%1)\n\t"                                      \
         "lw $a1, 8(%1)\n\t"                                      \
         "lw $a2,12(%1)\n\t"                                      \
         "lw $a3,16(%1)\n\t"                                      \
         "lw $a4,20(%1)\n\t"                                      \
         "lw $a5,24(%1)\n\t"                                      \
         "lw $a6,28(%1)\n\t"                                      \
         "lw $a7,32(%1)\n\t"                                      \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0\n"                                         \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将 _res 的值转换为宏定义中指定类型 lval，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
# 宏定义，用于调用具有9个参数的函数
#define CALL_FN_W_9W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8,arg9)                  \
   do {                                                           \
      # 声明一个 volatile 变量 _orig，用于保存原始函数地址
      volatile OrigFn        _orig = (orig);                      \
      # 声明一个 volatile 数组 _argvec，用于保存参数的地址
      volatile unsigned long _argvec[10];                         \
      # 声明一个 volatile unsigned long 变量 _res，用于保存函数调用结果
      volatile unsigned long _res;                                \
      # 将参数和函数地址存入 _argvec 数组
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      _argvec[9] = (unsigned long)(arg9);                         \
      # 使用内联汇编，准备函数调用环境
      __asm__ volatile(                                           \
         "addiu $sp, $sp, -16  \n\t"                              \
         "lw $t9,36(%1)        \n\t"                              \
         "sw $t9, 0($sp)       \n\t"                              \
         "lw $t9, 0(%1)        \n\t"                              \
         "lw $a0, 4(%1)        \n\t"                              \
         "lw $a1, 8(%1)        \n\t"                              \
         "lw $a2,12(%1)        \n\t"                              \
         "lw $a3,16(%1)        \n\t"                              \
         "lw $a4,20(%1)        \n\t"                              \
         "lw $a5,24(%1)        \n\t"                              \
         "lw $a6,28(%1)        \n\t"                              \
         "lw $a7,32(%1)        \n\t"                              \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0         \n\t"                              \
         "addiu $sp, $sp, 16   \n\t"                              \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      # 将函数调用结果转换为指定类型 lval，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)
/*
   定义一个宏，用于调用带有十个参数的函数，并将结果赋给 lval。
   参数说明：
   lval: 用于存储函数调用结果的变量
   orig: 原始函数的地址
   arg1...arg10: 十个函数参数
*/
#define CALL_FN_W_10W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,  \
                                  arg7,arg8,arg9,arg10)           \
   do {                                                           \
      // 声明一个指向原始函数地址的变量 _orig，并设置为 volatile 类型，确保不被优化
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个长度为 11 的无符号长整型数组 _argvec，用于存储函数参数
      volatile unsigned long _argvec[11];                         \
      // 声明一个无符号长整型变量 _res，用于存储函数调用结果
      volatile unsigned long _res;                                \
      // 将原始函数地址的 nraddr 成员赋值给 _argvec[0]
      _argvec[0] = (unsigned long)_orig.nraddr;                   \
      // 将参数 arg1...arg10 依次赋值给 _argvec[1] 到 _argvec[10]
      _argvec[1] = (unsigned long)(arg1);                         \
      _argvec[2] = (unsigned long)(arg2);                         \
      _argvec[3] = (unsigned long)(arg3);                         \
      _argvec[4] = (unsigned long)(arg4);                         \
      _argvec[5] = (unsigned long)(arg5);                         \
      _argvec[6] = (unsigned long)(arg6);                         \
      _argvec[7] = (unsigned long)(arg7);                         \
      _argvec[8] = (unsigned long)(arg8);                         \
      _argvec[9] = (unsigned long)(arg9);                         \
      _argvec[10] = (unsigned long)(arg10);                       \
      // 内联汇编代码块，执行函数调用
      __asm__ volatile(                                           \
         "addiu $sp, $sp, -16  \n\t"                              \
         "lw $t9,36(%1)        \n\t"                              \
         "sw $t9, 0($sp)       \n\t"                              \
         "lw $t9,40(%1)        \n\t"                              \
         "sw $t9, 4($sp)       \n\t"                              \
         "lw $t9, 0(%1)        \n\t"                              \
         "lw $a0, 4(%1)        \n\t"                              \
         "lw $a1, 8(%1)        \n\t"                              \
         "lw $a2,12(%1)        \n\t"                              \
         "lw $a3,16(%1)        \n\t"                              \
         "lw $a4,20(%1)        \n\t"                              \
         "lw $a5,24(%1)        \n\t"                              \
         "lw $a6,28(%1)        \n\t"                              \
         "lw $a7,32(%1)        \n\t"                              \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $a0         \n\t"                              \
         "addiu $sp, $sp, 16   \n\t"                              \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将函数调用的结果转换为 lval 的类型，并赋给 lval
      lval = (__typeof__(lval)) _res;                             \
   } while (0)

#endif /* PLAT_nanomips_linux */

/* ------------------------- mips64-linux ------------------------- */

#if defined(PLAT_mips64_linux)

/* These regs are trashed by the hidden call. */
// 定义一个宏 __CALLER_SAVED_REGS，表示在隐藏调用中会被破坏的寄存器列表
#define __CALLER_SAVED_REGS "$2", "$3", "$4", "$5", "$6",
/* 定义一个宏，用于将64位MIPS架构上的长整型转换为寄存器类型 */
#define MIPS64_LONG2REG_CAST(x) ((long long)(long)x)

/* 定义一个宏，用于调用一个没有返回值的函数 */
#define CALL_FN_W_v(lval, orig)                                   \
   do {                                                           \
      // 声明一个 volatile 类型为 OrigFn 的变量 _orig，并将参数 orig 赋给它
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个 volatile 类型为 unsigned long long 的数组 _argvec，包含一个元素
      volatile unsigned long long _argvec[1];                     \
      // 声明一个 volatile 类型为 unsigned long long 的变量 _res，用于存储函数调用结果
      volatile unsigned long long _res;                           \
      // 将 _orig 的地址转换为 long long 类型，存入 _argvec 数组的第一个元素
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            \
      // 使用内联汇编调用函数
      __asm__ volatile(                                           \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "0" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将结果转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)

/* 定义一个宏，用于调用一个有一个参数且返回值为 long long 类型的函数 */
#define CALL_FN_W_W(lval, orig, arg1)                             \
   do {                                                           \
      // 声明一个 volatile 类型为 OrigFn 的变量 _orig，并将参数 orig 赋给它
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个 volatile 类型为 unsigned long long 的数组 _argvec，包含两个元素
      volatile unsigned long long _argvec[2];                     \
      // 声明一个 volatile 类型为 unsigned long long 的变量 _res，用于存储函数调用结果
      volatile unsigned long long  _res;                          \
      // 将 _orig 的地址转换为 long long 类型，存入 _argvec 数组的第一个元素
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            \
      // 将参数 arg1 转换为 long long 类型，存入 _argvec 数组的第二个元素
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      // 使用内联汇编调用函数
      __asm__ volatile(                                           \
         "ld $4, 8(%1)\n\t"   /* arg1*/                           \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将结果转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)
# 定义一个宏，用于调用一个带有两个参数的函数
#define CALL_FN_W_WW(lval, orig, arg1,arg2)                       \
   do {                                                           \
      // 声明一个 volatile 类型的 OrigFn 变量 _orig，用于存储传入的函数指针 orig
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个包含三个 unsigned long long 类型元素的数组 _argvec，用于存储参数
      volatile unsigned long long _argvec[3];                     \
      // 声明一个 unsigned long long 类型的变量 _res，用于存储函数调用的返回值
      volatile unsigned long long _res;                           \
      // 将第一个参数 orig 的函数地址存入 _argvec 数组的第一个位置
      _argvec[0] = _orig.nraddr;                                  \
      // 将第二个参数 arg1 转换成 MIPS64_LONG2REG_CAST 宏定义的寄存器表示形式，并存入 _argvec 数组的第二个位置
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      // 将第三个参数 arg2 转换成 MIPS64_LONG2REG_CAST 宏定义的寄存器表示形式，并存入 _argvec 数组的第三个位置
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    \
      // 使用内联汇编执行以下指令序列，其中 $4、$5、$25 是寄存器，VALGRIND_CALL_NOREDIR_T9 是宏定义的汇编指令
      __asm__ volatile(                                           \
         "ld $4, 8(%1)\n\t"                                       \
         "ld $5, 16(%1)\n\t"                                      \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将 _res 的值转换成 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)


# 定义一个宏，用于调用一个带有三个参数的函数
#define CALL_FN_W_WWW(lval, orig, arg1,arg2,arg3)                 \
   do {                                                           \
      // 声明一个 volatile 类型的 OrigFn 变量 _orig，用于存储传入的函数指针 orig
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个包含四个 unsigned long long 类型元素的数组 _argvec，用于存储参数
      volatile unsigned long long _argvec[4];                     \
      // 声明一个 unsigned long long 类型的变量 _res，用于存储函数调用的返回值
      volatile unsigned long long _res;                           \
      // 将第一个参数 orig 的函数地址存入 _argvec 数组的第一个位置
      _argvec[0] = _orig.nraddr;                                  \
      // 将第二个参数 arg1 转换成 MIPS64_LONG2REG_CAST 宏定义的寄存器表示形式，并存入 _argvec 数组的第二个位置
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      // 将第三个参数 arg2 转换成 MIPS64_LONG2REG_CAST 宏定义的寄存器表示形式，并存入 _argvec 数组的第三个位置
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    \
      // 将第四个参数 arg3 转换成 MIPS64_LONG2REG_CAST 宏定义的寄存器表示形式，并存入 _argvec 数组的第四个位置
      _argvec[3] = MIPS64_LONG2REG_CAST(arg3);                    \
      // 使用内联汇编执行以下指令序列，其中 $4、$5、$6、$25 是寄存器，VALGRIND_CALL_NOREDIR_T9 是宏定义的汇编指令
      __asm__ volatile(                                           \
         "ld $4, 8(%1)\n\t"                                       \
         "ld $5, 16(%1)\n\t"                                      \
         "ld $6, 24(%1)\n\t"                                      \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将 _res 的值转换成 lval 的类型并赋值给 lval
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)
/*
   宏定义：CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)

   lval:       调用函数后的返回值
   orig:       原始函数指针
   arg1-arg4:  函数调用时的参数

   使用宏展开后，执行以下操作：
   1. 声明 _orig 变量并将原始函数指针 (orig) 赋给它
   2. 声明 _argvec 数组，存放函数调用时的参数
   3. 声明 _res 变量，存放函数调用的返回值

   在内联汇编中：
   - 将 _argvec 数组中的值加载到寄存器中，用于函数调用
   - 使用 __asm__ volatile 定义内联汇编代码块，实现函数调用过程
   - 使用 t9 寄存器调用目标函数，根据 VALGRIND_CALL_NOREDIR_T9 宏可能有特殊处理
   - 将函数调用的返回值存入 _res 变量中

   最后，将 _res 强制转换为 lval 的类型，并赋给 lval。

   注意事项：
   - 代码块使用 do { ... } while (0) 结构，确保宏展开后的语法正确性
   - 内联汇编中使用了 "memory" 和 __CALLER_SAVED_REGS 来声明寄存器和内存的影响
*/
#define CALL_FN_W_WWWW(lval, orig, arg1,arg2,arg3,arg4)           \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long long _argvec[5];                     \
      volatile unsigned long long _res;                           \
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            \
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    \
      _argvec[3] = MIPS64_LONG2REG_CAST(arg3);                    \
      _argvec[4] = MIPS64_LONG2REG_CAST(arg4);                    \
      __asm__ volatile(                                           \
         "ld $4, 8(%1)\n\t"                                       \
         "ld $5, 16(%1)\n\t"                                      \
         "ld $6, 24(%1)\n\t"                                      \
         "ld $7, 32(%1)\n\t"                                      \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)
#define CALL_FN_W_5W(lval, orig, arg1,arg2,arg3,arg4,arg5)        \
   do {                                                           \
      // 声明一个 volatile 指针 _orig，指向传入的函数指针 orig
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个 volatile unsigned long long 数组 _argvec，用于存放函数调用时的参数
      volatile unsigned long long _argvec[6];                     \
      // 声明一个 volatile unsigned long long 变量 _res，用于存放函数调用的返回值
      volatile unsigned long long _res;                           \
      // 将函数指针 _orig.nraddr 转换为 unsigned long long 存入参数数组的第一个位置
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            \
      // 将参数 arg1,arg2,arg3,arg4,arg5 依次转换为 unsigned long long 存入参数数组的对应位置
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    \
      _argvec[3] = MIPS64_LONG2REG_CAST(arg3);                    \
      _argvec[4] = MIPS64_LONG2REG_CAST(arg4);                    \
      _argvec[5] = MIPS64_LONG2REG_CAST(arg5);                    \
      // 使用内联汇编，加载参数数组中的内容到寄存器，并调用目标函数指针的位置
      __asm__ volatile(                                           \
         "ld $4, 8(%1)\n\t"                                       \
         "ld $5, 16(%1)\n\t"                                      \
         "ld $6, 24(%1)\n\t"                                      \
         "ld $7, 32(%1)\n\t"                                      \
         "ld $8, 40(%1)\n\t"                                      \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将 _res 强制转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)
#define CALL_FN_W_6W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6)   \
   do {                                                           \
      // 定义一个 volatile 原始函数指针 _orig，并初始化为传入的 orig 参数
      volatile OrigFn        _orig = (orig);                      \
      // 定义一个包含7个元素的 volatile 无符号长整型数组 _argvec，用于存储函数调用的参数
      volatile unsigned long long _argvec[7];                     \
      // 定义一个 volatile 无符号长整型变量 _res，用于存储函数调用的结果
      volatile unsigned long long _res;                           \
      // 将函数指针的地址和各个参数转换为长整型，并存储到 _argvec 数组中
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            \
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    \
      _argvec[3] = MIPS64_LONG2REG_CAST(arg3);                    \
      _argvec[4] = MIPS64_LONG2REG_CAST(arg4);                    \
      _argvec[5] = MIPS64_LONG2REG_CAST(arg5);                    \
      _argvec[6] = MIPS64_LONG2REG_CAST(arg6);                    \
      // 使用内联汇编，将 _argvec 的地址传递给汇编语句，同时加载部分寄存器和目标地址
      __asm__ volatile(                                           \
         "ld $4, 8(%1)\n\t"                                       \
         "ld $5, 16(%1)\n\t"                                      \
         "ld $6, 24(%1)\n\t"                                      \
         "ld $7, 32(%1)\n\t"                                      \
         "ld $8, 40(%1)\n\t"                                      \
         "ld $9, 48(%1)\n\t"                                      \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将 _res 强制转换为 lval 的类型，并将其赋给 lval
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)
# 定义一个宏，用于调用具有7个参数的函数
#define CALL_FN_W_7W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7)                            \
   do {                                                           \
      # 声明一个指向原始函数的指针
      volatile OrigFn        _orig = (orig);                      
      # 声明一个包含8个unsigned long long类型元素的数组，用于存储参数和结果
      volatile unsigned long long _argvec[8];                     
      # 声明一个unsigned long long类型变量，用于存储结果
      volatile unsigned long long _res;                           
      # 将原始函数的地址转换为unsigned long long类型并存储到_argvec数组的第一个元素中
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            
      # 将参数1到参数7转换为unsigned long long类型并存储到_argvec数组中
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    
      _argvec[3] = MIPS64_LONG2REG_CAST(arg3);                    
      _argvec[4] = MIPS64_LONG2REG_CAST(arg4);                    
      _argvec[5] = MIPS64_LONG2REG_CAST(arg5);                    
      _argvec[6] = MIPS64_LONG2REG_CAST(arg6);                    
      _argvec[7] = MIPS64_LONG2REG_CAST(arg7);                    
      # 使用内联汇编语句执行具体的函数调用过程
      __asm__ volatile(                                           \
         "ld $4, 8(%1)\n\t"                                       \
         "ld $5, 16(%1)\n\t"                                      \
         "ld $6, 24(%1)\n\t"                                      \
         "ld $7, 32(%1)\n\t"                                      \
         "ld $8, 40(%1)\n\t"                                      \
         "ld $9, 48(%1)\n\t"                                      \
         "ld $10, 56(%1)\n\t"                                     \
         "ld $25, 0(%1) \n\t"  /* target->t9 */                   \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      # 将结果转换为目标类型并赋值给lval
      lval = (__typeof__(lval)) (long)_res;                       
   } while (0)
#define CALL_FN_W_8W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8)                       \
   do {                                                           \
      // 声明并初始化函数指针 _orig，并将传入的 orig 参数赋值给 _orig
      volatile OrigFn        _orig = (orig);                      \
      // 声明一个长度为 9 的无符号长整型数组 _argvec，用于存放函数调用的参数
      volatile unsigned long long _argvec[9];                     \
      // 声明一个无符号长整型变量 _res，用于存放函数调用的返回值
      volatile unsigned long long _res;                           \
      // 将函数指针 _orig 的地址转换为 MIPS64 架构下的寄存器形式，存入 _argvec[0]
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            \
      // 将参数 arg1 到 arg8 分别转换为 MIPS64 架构下的寄存器形式，存入 _argvec 数组中相应位置
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    \
      _argvec[3] = MIPS64_LONG2REG_CAST(arg3);                    \
      _argvec[4] = MIPS64_LONG2REG_CAST(arg4);                    \
      _argvec[5] = MIPS64_LONG2REG_CAST(arg5);                    \
      _argvec[6] = MIPS64_LONG2REG_CAST(arg6);                    \
      _argvec[7] = MIPS64_LONG2REG_CAST(arg7);                    \
      _argvec[8] = MIPS64_LONG2REG_CAST(arg8);                    \
      // 使用内联汇编进行函数调用
      __asm__ volatile(                                           \
         // 加载 _argvec[0] 到 $4 寄存器
         "ld $4, 8(%1)\n\t"                                       \
         // 加载 _argvec[1] 到 $5 寄存器
         "ld $5, 16(%1)\n\t"                                      \
         // 加载 _argvec[2] 到 $6 寄存器
         "ld $6, 24(%1)\n\t"                                      \
         // 加载 _argvec[3] 到 $7 寄存器
         "ld $7, 32(%1)\n\t"                                      \
         // 加载 _argvec[4] 到 $8 寄存器
         "ld $8, 40(%1)\n\t"                                      \
         // 加载 _argvec[5] 到 $9 寄存器
         "ld $9, 48(%1)\n\t"                                      \
         // 加载 _argvec[6] 到 $10 寄存器
         "ld $10, 56(%1)\n\t"                                     \
         // 加载 _argvec[7] 到 $11 寄存器
         "ld $11, 64(%1)\n\t"                                     \
         // 加载 _argvec[0] 到 $25 寄存器，这是目标函数地址
         "ld $25, 0(%1) \n\t"  /* target->t9 */                   \
         // 在调试模式下，将 VALGRIND_CALL_NOREDIR_T9 插入到汇编中
         VALGRIND_CALL_NOREDIR_T9                                 \
         // 将 $2 寄存器中的值移动到 _res 变量中
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      // 将 _res 强制转换为 lval 的类型，并赋值给 lval
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)
/*
   宏定义：CALL_FN_W_9W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,
                                arg7,arg8,arg9)

   参数说明：
   lval: 调用结果将被存储在此变量中
   orig: 原始函数指针
   arg1-arg9: 函数调用的9个参数

   功能：
   使用给定的原始函数指针和9个参数调用目标函数，并将结果存储在lval中。

   实现细节：
   - 将_orig转换为volatile OrigFn类型，并存储在_orig中
   - 定义一个长度为10的volatile unsigned long long数组 _argvec 用于存储函数参数
   - 将原始函数指针和每个参数转换为MIPS64_LONG2REG_CAST类型，并存储在_argvec数组中对应的位置
   - 使用内联汇编调用目标函数：
     - 分配8字节的栈空间
     - 加载原始函数指针的值到寄存器$4，并存储到栈中
     - 依次加载arg1至arg9的值到寄存器$4至$11，并存储到栈中
     - 加载目标函数指针的值到寄存器$25（target->t9）
     - 执行VALGRIND_CALL_NOREDIR_T9宏（可能是调试器相关的宏）
     - 恢复栈空间
     - 将函数调用结果存储到_res中
   - 将_res强制转换为lval的类型，存储在lval中

   注意事项：
   - volatile用于确保编译器不会优化相关变量和代码段
   - 内联汇编使用了特定的MIPS64架构指令集
   - 注释中的寄存器名称如$2是根据MIPS64架构的寄存器分配约定来命名的

   参考资料：
   - MIPS64_LONG2REG_CAST是一个宏定义，可能用于将long型转换为MIPS64架构的寄存器值
   - __CALLER_SAVED_REGS是一个可能定义的宏，用于指定在内联汇编中应该保存的寄存器列表
*/
#define CALL_FN_W_9W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,   \
                                 arg7,arg8,arg9)                  \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long long _argvec[10];                    \
      volatile unsigned long long _res;                           \
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            \
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    \
      _argvec[3] = MIPS64_LONG2REG_CAST(arg3);                    \
      _argvec[4] = MIPS64_LONG2REG_CAST(arg4);                    \
      _argvec[5] = MIPS64_LONG2REG_CAST(arg5);                    \
      _argvec[6] = MIPS64_LONG2REG_CAST(arg6);                    \
      _argvec[7] = MIPS64_LONG2REG_CAST(arg7);                    \
      _argvec[8] = MIPS64_LONG2REG_CAST(arg8);                    \
      _argvec[9] = MIPS64_LONG2REG_CAST(arg9);                    \
      __asm__ volatile(                                           \
         "dsubu $29, $29, 8\n\t"                                  \
         "ld $4, 72(%1)\n\t"                                      \
         "sd $4, 0($29)\n\t"                                      \
         "ld $4, 8(%1)\n\t"                                       \
         "ld $5, 16(%1)\n\t"                                      \
         "ld $6, 24(%1)\n\t"                                      \
         "ld $7, 32(%1)\n\t"                                      \
         "ld $8, 40(%1)\n\t"                                      \
         "ld $9, 48(%1)\n\t"                                      \
         "ld $10, 56(%1)\n\t"                                     \
         "ld $11, 64(%1)\n\t"                                     \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "daddu $29, $29, 8\n\t"                                  \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)
/* 定义一个宏，用于调用具有10个参数的函数 */
#define CALL_FN_W_10W(lval, orig, arg1,arg2,arg3,arg4,arg5,arg6,  \
                                  arg7,arg8,arg9,arg10)           \
   do {                                                           \
      volatile OrigFn        _orig = (orig);                      \
      volatile unsigned long long _argvec[11];                    \
      volatile unsigned long long _res;                           \
      /* 将函数原始地址转换为寄存器格式并存储在参数向量中 */    \
      _argvec[0] = MIPS64_LONG2REG_CAST(_orig.nraddr);            \
      _argvec[1] = MIPS64_LONG2REG_CAST(arg1);                    \
      _argvec[2] = MIPS64_LONG2REG_CAST(arg2);                    \
      _argvec[3] = MIPS64_LONG2REG_CAST(arg3);                    \
      _argvec[4] = MIPS64_LONG2REG_CAST(arg4);                    \
      _argvec[5] = MIPS64_LONG2REG_CAST(arg5);                    \
      _argvec[6] = MIPS64_LONG2REG_CAST(arg6);                    \
      _argvec[7] = MIPS64_LONG2REG_CAST(arg7);                    \
      _argvec[8] = MIPS64_LONG2REG_CAST(arg8);                    \
      _argvec[9] = MIPS64_LONG2REG_CAST(arg9);                    \
      _argvec[10] = MIPS64_LONG2REG_CAST(arg10);                  \
      /* 嵌入汇编代码块，调用函数并处理返回值 */                  \
      __asm__ volatile(                                           \
         "dsubu $29, $29, 16\n\t"                                 \
         "ld $4, 72(%1)\n\t"                                      \
         "sd $4, 0($29)\n\t"                                      \
         "ld $4, 80(%1)\n\t"                                      \
         "sd $4, 8($29)\n\t"                                      \
         "ld $4, 8(%1)\n\t"                                       \
         "ld $5, 16(%1)\n\t"                                      \
         "ld $6, 24(%1)\n\t"                                      \
         "ld $7, 32(%1)\n\t"                                      \
         "ld $8, 40(%1)\n\t"                                      \
         "ld $9, 48(%1)\n\t"                                      \
         "ld $10, 56(%1)\n\t"                                     \
         "ld $11, 64(%1)\n\t"                                     \
         "ld $25, 0(%1)\n\t"  /* target->t9 */                    \
         VALGRIND_CALL_NOREDIR_T9                                 \
         "daddu $29, $29, 16\n\t"                                 \
         "move %0, $2\n"                                          \
         : /*out*/   "=r" (_res)                                  \
         : /*in*/    "r" (&_argvec[0])                            \
         : /*trash*/ "memory", __CALLER_SAVED_REGS                \
      );                                                          \
      /* 将结果强制转换为 lval 的类型并存储在 lval 中 */         \
      lval = (__typeof__(lval)) (long)_res;                       \
   } while (0)
/* ------------------------------------------------------------------ */

/* Some request codes.  There are many more of these, but most are not
   exposed to end-user view.  These are the public ones, all of the
   form 0x1000 + small_number.

   Core ones are in the range 0x00000000--0x0000ffff.  The non-public
   ones start at 0x2000.
*/

/* These macros are used by tools -- they must be public, but don't
   embed them into other programs. */

/* 定义一个宏，用于生成工具请求的请求码。
   参数a和b分别是8位无符号整数，返回一个32位无符号整数。*/
#define VG_USERREQ_TOOL_BASE(a,b) \
   ((unsigned int)(((a)&0xff) << 24 | ((b)&0xff) << 16))

/* 定义一个宏，用于检查给定的请求码是否是工具请求。
   参数a和b是8位无符号整数，v是待检查的请求码。返回一个布尔值。*/
#define VG_IS_TOOL_USERREQ(a, b, v) \
   (VG_USERREQ_TOOL_BASE(a,b) == ((v) & 0xffff0000))

/* !! ABIWARNING !! ABIWARNING !! ABIWARNING !! ABIWARNING !! 
   This enum comprises an ABI exported by Valgrind to programs
   which use client requests.  DO NOT CHANGE THE NUMERIC VALUES OF THESE
   ENTRIES, NOR DELETE ANY -- add new ones at the end of the most
   relevant group. */

/* 如果未定义__GNUC__，则定义__extension__为空 */
#if !defined(__GNUC__)
#  define __extension__ /* */
#endif

/* Returns the number of Valgrinds this code is running under.  That
   is, 0 if running natively, 1 if running under Valgrind, 2 if
   running under Valgrind which is running under another Valgrind,
   etc. */

/* 返回当前代码运行在多少个Valgrind实例下。
   如果未运行在Valgrind下，则返回0；运行在一个Valgrind下则返回1；
   运行在一个运行另一个Valgrind下的Valgrind下则返回2，以此类推。*/
#define RUNNING_ON_VALGRIND                                           \
    (unsigned)VALGRIND_DO_CLIENT_REQUEST_EXPR(0 /* if not */,         \
                                    VG_USERREQ__RUNNING_ON_VALGRIND,  \
                                    0, 0, 0, 0, 0)

/* Discard translation of code in the range [_qzz_addr .. _qzz_addr +
   _qzz_len - 1].  Useful if you are debugging a JITter or some such,
   since it provides a way to make sure valgrind will retranslate the
   invalidated area.  Returns no value. */

/* 丢弃范围内代码的翻译，范围为[_qzz_addr .. _qzz_addr + _qzz_len - 1]。
   如果你正在调试JIT或类似工具，这将确保Valgrind重新翻译无效区域。
   没有返回值。*/
#define VALGRIND_DISCARD_TRANSLATIONS(_qzz_addr,_qzz_len)              \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__DISCARD_TRANSLATIONS,  \
                                    _qzz_addr, _qzz_len, 0, 0, 0)

/* Perform inner thread operations on the given address. */

/* 对给定地址执行内部线程操作。*/
#define VALGRIND_INNER_THREADS(_qzz_addr)                               \
   VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__INNER_THREADS,           \
                                   _qzz_addr, 0, 0, 0, 0)

/* These requests are for getting Valgrind itself to print something.
   Possibly with a backtrace.  This is a really ugly hack.  The return value
   is the number of characters printed, excluding the "**<pid>** " part at the
   start and the backtrace (if present). */

/* 下面这些请求用于让Valgrind自身打印一些内容，可能包括回溯信息。
   这是一个非常不优雅的hack。返回值是打印的字符数，不包括开头的"**<pid>** "部分和回溯信息。*/

/* 如果定义了__GNUC__或者__INTEL_COMPILER且未定义_MSC_VER，则声明VALGRIND_PRINTF为静态函数。
   使用__attribute__((format(__printf__, 1, 2), __unused__))声明printf风格的格式化字符串。
   返回值类型为int。 */
#if defined(__GNUC__) || (defined(__INTEL_COMPILER) && !defined(_MSC_VER))
static int VALGRIND_PRINTF(const char *format, ...)
   __attribute__((format(__printf__, 1, 2), __unused__));
#endif

/* 如果定义了_MSC_VER，则声明VALGRIND_PRINTF为内联函数。 */
#if defined(_MSC_VER)
__inline
#endif
/* 定义VALGRIND_PRINTF函数，用于在Valgrind中打印格式化输出。
   如果未定义NVALGRIND，则使用printf风格的格式化字符串，返回打印的字符数。
   否则，忽略format参数，返回0。*/
VALGRIND_PRINTF(const char *format, ...)
{
#if defined(NVALGRIND)
   (void)format;
   return 0;
#else /* NVALGRIND */
#if defined(_MSC_VER) || defined(__MINGW64__)
   uintptr_t _qzz_res;
/* 
   定义了一个带有参数宏VALGRIND_NON_SIMD_CALL0，该宏用于调用Valgrind的客户端请求，
   从模拟CPU切换到真实CPU执行任意函数。这里是带0个参数的版本。
*/
#define VALGRIND_NON_SIMD_CALL0(_qyy_fn)                          \
#if defined(NVALGRIND)                                             \
   (void)_qyy_fn;                                                   \
#else /* NVALGRIND */
   /*
      如果未定义NVALGRIND，则进入Valgrind环境。以下代码用于执行VALGRIND_DO_CLIENT_REQUEST_EXPR，
      以执行带有0个参数的客户端请求，从模拟CPU切换到真实CPU执行函数_qyy_fn。
   */
   _qzz_res = VALGRIND_DO_CLIENT_REQUEST_EXPR(0,                   \
                              VG_USERREQ__qyy_fn,                 \
                              0, 0, 0, 0, 0);                     \
   /*
      完成参数列表的使用
   */
#endif /* NVALGRIND */
    # 使用 Valgrind 提供的宏执行客户端请求，这里是一个表达式形式的请求
    VALGRIND_DO_CLIENT_REQUEST_EXPR(
        0 /* default return */,       \  # 设置默认的返回值
        VG_USERREQ__CLIENT_CALL0,     \  # 发起一个客户端请求，标识为 CLIENT_CALL0
        _qyy_fn,                      \  # 指定请求的函数名或标识符
        0, 0, 0, 0                     \  # 其他参数，这里没有额外参数
    )
#define VALGRIND_NON_SIMD_CALL1(_qyy_fn, _qyy_arg1)                    \
    // 执行 Valgrind 客户端请求，调用单参数函数
    VALGRIND_DO_CLIENT_REQUEST_EXPR(0 /* default return */,            \
                                    VG_USERREQ__CLIENT_CALL1,          \
                                    _qyy_fn,                           \
                                    _qyy_arg1, 0, 0, 0)

#define VALGRIND_NON_SIMD_CALL2(_qyy_fn, _qyy_arg1, _qyy_arg2)         \
    // 执行 Valgrind 客户端请求，调用双参数函数
    VALGRIND_DO_CLIENT_REQUEST_EXPR(0 /* default return */,            \
                                    VG_USERREQ__CLIENT_CALL2,          \
                                    _qyy_fn,                           \
                                    _qyy_arg1, _qyy_arg2, 0, 0)

#define VALGRIND_NON_SIMD_CALL3(_qyy_fn, _qyy_arg1, _qyy_arg2, _qyy_arg3) \
    // 执行 Valgrind 客户端请求，调用三参数函数
    VALGRIND_DO_CLIENT_REQUEST_EXPR(0 /* default return */,             \
                                    VG_USERREQ__CLIENT_CALL3,           \
                                    _qyy_fn,                            \
                                    _qyy_arg1, _qyy_arg2,               \
                                    _qyy_arg3, 0)

/* Counts the number of errors that have been recorded by a tool.  Nb:
   the tool must record the errors with VG_(maybe_record_error)() or
   VG_(unique_error)() for them to be counted. */
#define VALGRIND_COUNT_ERRORS                                     \
    // 执行 Valgrind 客户端请求，统计已记录的错误数
    (unsigned)VALGRIND_DO_CLIENT_REQUEST_EXPR(                    \
                               0 /* default return */,            \
                               VG_USERREQ__COUNT_ERRORS,          \
                               0, 0, 0, 0, 0)

#define VALGRIND_MALLOCLIKE_BLOCK(addr, sizeB, rzB, is_zeroed)          \
    // 执行 Valgrind 客户端请求，模拟类似 malloc 的内存分配
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__MALLOCLIKE_BLOCK,       \
                                    addr, sizeB, rzB, is_zeroed, 0)

/* See the comment for VALGRIND_MALLOCLIKE_BLOCK for details.
   Ignored if addr == 0.
*/
#define VALGRIND_RESIZEINPLACE_BLOCK(addr, oldSizeB, newSizeB, rzB)     \
    // 执行 Valgrind 客户端请求，调整就地重分配的内存块
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__RESIZEINPLACE_BLOCK,    \
                                    addr, oldSizeB, newSizeB, rzB, 0)

/* See the comment for VALGRIND_MALLOCLIKE_BLOCK for details.
   Ignored if addr == 0.
*/
#define VALGRIND_FREELIKE_BLOCK(addr, rzB)                              \
    // 执行 Valgrind 客户端请求，模拟类似 free 的内存释放
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__FREELIKE_BLOCK,         \
                                    addr, rzB, 0, 0, 0)

/* Create a memory pool. */
#define VALGRIND_CREATE_MEMPOOL(pool, rzB, is_zeroed)             \
    // 执行 Valgrind 客户端请求，创建内存池
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__CREATE_MEMPOOL,   \
                                    pool, rzB, is_zeroed, 0, 0)
/* 创建一个内存池，使用一些标志来指定扩展行为。
   当 flags 为零时，行为与 VALGRIND_CREATE_MEMPOOL 相同。
   
   标志 VALGRIND_MEMPOOL_METAPOOL 指定，使用 VALGRIND_MEMPOOL_ALLOC 分配的内存
   会被应用程序用作超级块，用于分配使用 VALGRIND_MALLOCLIKE_BLOCK 的类似 MALLOC 的块。
   换句话说，元池是一个“两级”池：第一级是由 VALGRIND_MEMPOOL_ALLOC 描述的块。
   第二级块由 VALGRIND_MALLOCLIKE_BLOCK 描述。
   注意，池与第二级块之间的关联是隐含的：第二级块将位于第一级块内部。
   对于这种两级池，必须使用 VALGRIND_MEMPOOL_METAPOOL 标志，否则 valgrind 将检测到
   内存块重叠，并在执行过程中终止（例如在泄漏搜索期间）。

   这样的元池也可以使用 VALGRIND_MEMPOOL_AUTO_FREE 标志标记为“自动释放”池，
   必须与 VALGRIND_MEMPOOL_METAPOOL 按位 OR 在一起使用。
   对于“自动释放”池，调用 VALGRIND_MEMPOOL_FREE 将自动释放包含在使用
   VALGRIND_MEMPOOL_FREE 释放的第一级块中的所有第二级块。
   换句话说，调用 VALGRIND_MEMPOOL_FREE 将导致对所有包含在第一级块中的第二级块
   隐式调用 VALGRIND_FREELIKE_BLOCK。
   注意：在没有 VALGRIND_MEMPOOL_METAPOOL 标志的情况下使用 VALGRIND_MEMPOOL_AUTO_FREE 标志是错误的。
*/
#define VALGRIND_MEMPOOL_AUTO_FREE  1
#define VALGRIND_MEMPOOL_METAPOOL   2

/* 创建一个扩展内存池 */
#define VALGRIND_CREATE_MEMPOOL_EXT(pool, rzB, is_zeroed, flags)        \
   VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__CREATE_MEMPOOL,          \
                                   pool, rzB, is_zeroed, flags, 0)

/* 销毁一个内存池 */
#define VALGRIND_DESTROY_MEMPOOL(pool)                            \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__DESTROY_MEMPOOL,  \
                                    pool, 0, 0, 0, 0)

/* 将一段内存关联到内存池 */
#define VALGRIND_MEMPOOL_ALLOC(pool, addr, size)                  \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__MEMPOOL_ALLOC,    \
                                    pool, addr, size, 0, 0)

/* 从内存池中解除一段内存的关联 */
#define VALGRIND_MEMPOOL_FREE(pool, addr)                         \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__MEMPOOL_FREE,     \
                                    pool, addr, 0, 0, 0)

/* 解除内存池中特定范围外的所有段的关联 */
#define VALGRIND_MEMPOOL_TRIM(pool, addr, size)                   \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__MEMPOOL_TRIM,     \
                                    pool, addr, size, 0, 0)

/* 调整和/或移动与内存池关联的一段内存 */
/* 宏定义：VALGRIND_MOVE_MEMPOOL(poolA, poolB)
   通过客户端请求移动内存池
   参数：poolA - 源内存池标识符
         poolB - 目标内存池标识符
*/
#define VALGRIND_MOVE_MEMPOOL(poolA, poolB)                       \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__MOVE_MEMPOOL,     \
                                    poolA, poolB, 0, 0, 0)

/* 宏定义：VALGRIND_MEMPOOL_CHANGE(pool, addrA, addrB, size)
   通过客户端请求改变或移动与内存池相关的内存块
   参数：pool  - 内存池标识符
         addrA - 原始地址
         addrB - 目标地址
         size  - 内存块大小
*/
#define VALGRIND_MEMPOOL_CHANGE(pool, addrA, addrB, size)         \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__MEMPOOL_CHANGE,   \
                                    pool, addrA, addrB, size, 0)

/* 宏定义：VALGRIND_MEMPOOL_EXISTS(pool)
   通过客户端请求检查内存池是否存在
   参数：pool - 内存池标识符
   返回值：1 - 存在
           0 - 不存在
*/
#define VALGRIND_MEMPOOL_EXISTS(pool)                             \
    (unsigned)VALGRIND_DO_CLIENT_REQUEST_EXPR(0,                  \
                               VG_USERREQ__MEMPOOL_EXISTS,        \
                               pool, 0, 0, 0, 0)

/* 宏定义：VALGRIND_STACK_REGISTER(start, end)
   通过客户端请求将内存区域标记为堆栈
   参数：start - 堆栈起始地址
         end   - 堆栈结束地址
   返回值：堆栈标识符
*/
#define VALGRIND_STACK_REGISTER(start, end)                       \
    (unsigned)VALGRIND_DO_CLIENT_REQUEST_EXPR(0,                  \
                               VG_USERREQ__STACK_REGISTER,        \
                               start, end, 0, 0, 0)

/* 宏定义：VALGRIND_STACK_DEREGISTER(id)
   通过客户端请求取消标记内存区域为堆栈
   参数：id - 堆栈标识符
*/
#define VALGRIND_STACK_DEREGISTER(id)                             \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__STACK_DEREGISTER, \
                                    id, 0, 0, 0, 0)

/* 宏定义：VALGRIND_STACK_CHANGE(id, start, end)
   通过客户端请求改变堆栈的起始和结束地址
   参数：id     - 堆栈标识符
         start  - 新的堆栈起始地址
         end    - 新的堆栈结束地址
*/
#define VALGRIND_STACK_CHANGE(id, start, end)                     \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__STACK_CHANGE,     \
                                    id, start, end, 0, 0)

/* 宏定义：VALGRIND_LOAD_PDB_DEBUGINFO(fd, ptr, total_size, delta)
   通过客户端请求加载 Wine PE 映像的 PDB 调试信息
   参数：fd         - 文件描述符
         ptr        - 指向调试信息的指针
         total_size - 总大小
         delta      - 偏移量
*/
#define VALGRIND_LOAD_PDB_DEBUGINFO(fd, ptr, total_size, delta)     \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__LOAD_PDB_DEBUGINFO, \
                                    fd, ptr, total_size, delta, 0)

/* 宏定义：VALGRIND_MAP_IP_TO_SRCLOC(addr, buf64)
   通过客户端请求将代码地址映射到源文件名和行号
   参数：addr  - 代码地址
         buf64 - 指向64字节缓冲区的指针，用于存放结果
   返回值：映射结果，保证以零结尾
*/
#define VALGRIND_MAP_IP_TO_SRCLOC(addr, buf64)                    \
    (unsigned)VALGRIND_DO_CLIENT_REQUEST_EXPR(0,                  \
                               VG_USERREQ__MAP_IP_TO_SRCLOC,      \
                               addr, buf64, 0, 0, 0)
/* 禁用当前线程的错误报告。工作方式类似于堆栈，因此可以安全地多次调用此宏，
   前提是调用 VALGRIND_ENABLE_ERROR_REPORTING 相同次数以重新启用报告。
   第一次调用此宏禁用报告。后续调用不会生效，除非增加了 VALGRIND_ENABLE_ERROR_REPORTING 的调用次数以重新启用报告。
   子线程不会从父线程继承此设置 -- 它们始终使用启用报告创建。
*/
#define VALGRIND_DISABLE_ERROR_REPORTING                                \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__CHANGE_ERR_DISABLEMENT, \
                                    1, 0, 0, 0, 0)

/* 根据 VALGRIND_DISABLE_ERROR_REPORTING 的注释重新启用错误报告。*/
#define VALGRIND_ENABLE_ERROR_REPORTING                                 \
    VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__CHANGE_ERR_DISABLEMENT, \
                                    -1, 0, 0, 0, 0)

/* 执行客户端程序的监视器命令。
   如果使用 GDB 打开了连接，则输出将根据设置的 vgdb 输出模式发送。
   如果未打开连接，则输出将发送到日志输出。
   如果命令未被识别，返回 1，否则返回 0。
*/
#define VALGRIND_MONITOR_COMMAND(command)                               \
   VALGRIND_DO_CLIENT_REQUEST_EXPR(0, VG_USERREQ__GDB_MONITOR_COMMAND, \
                                   command, 0, 0, 0, 0)

/* 更改动态命令行选项的值。
   注意，未知或不可动态更改的选项将导致输出警告消息。
*/
#define VALGRIND_CLO_CHANGE(option)                           \
   VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__CLO_CHANGE, \
                                   option, 0, 0, 0, 0)

/* 解除定义不同平台的宏，在此处示例中都被解除定义。 */
#undef PLAT_x86_darwin
#undef PLAT_amd64_darwin
#undef PLAT_x86_win32
#undef PLAT_amd64_win64
#undef PLAT_x86_linux
#undef PLAT_amd64_linux
#undef PLAT_ppc32_linux
#undef PLAT_ppc64be_linux
#undef PLAT_ppc64le_linux
#undef PLAT_arm_linux
#undef PLAT_s390x_linux
#undef PLAT_mips32_linux
#undef PLAT_mips64_linux
#undef PLAT_nanomips_linux
#undef PLAT_x86_solaris
#undef PLAT_amd64_solaris

#endif   /* __VALGRIND_H */
```