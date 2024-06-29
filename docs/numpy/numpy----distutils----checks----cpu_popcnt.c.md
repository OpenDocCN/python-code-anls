# `.\numpy\numpy\distutils\checks\cpu_popcnt.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且使用了Intel编译器
     * 与GCC和CLANG不同，Intel编译器会暴露所有支持的内部函数，
     * 无论是否指定了这些特征的构建选项。
     * 因此，当通过`--cpu-baseline`启用选项本机/主机或通过环境变量`CFLAGS`启用选项时，
     * 我们必须测试CPU特性的#定义，否则测试将被破坏，并导致启用所有可能的特性。

     */
    #if !defined(__SSE4_2__) && !defined(__POPCNT__)
        #error "HOST/ARCH doesn't support POPCNT"
    #endif
#endif

#ifdef _MSC_VER
    #include <nmmintrin.h>
#else
    #include <popcntintrin.h>
#endif

int main(int argc, char **argv)
{
    // 确保生成popcnt指令
    // 并对汇编代码进行测试
    unsigned long long a = *((unsigned long long*)argv[argc-1]);
    unsigned int b = *((unsigned int*)argv[argc-2]);

#if defined(_M_X64) || defined(__x86_64__)
    a = _mm_popcnt_u64(a);
#endif
    b = _mm_popcnt_u32(b);
    return (int)a + b;
}
```