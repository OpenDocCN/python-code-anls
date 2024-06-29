# `.\numpy\numpy\distutils\checks\cpu_avx512_spr.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了 DETECT_FEATURES 并且使用 Intel 编译器，
     * Intel 编译器会公开所有支持的内置函数，无论是否指定了这些特性的构建选项。
     * 因此，当通过 `--cpu-baseline` 或环境变量 `CFLAGS` 启用 native/host 选项时，
     * 我们必须测试 CPU 特性的 #定义，否则测试将失败，并且会启用所有可能的特性。
     */
    #if !defined(__AVX512FP16__)
        #error "HOST/ARCH 不支持 Sapphire Rapids AVX512FP16 特性"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    /* clang 在我们的 spr 代码上存在一个 bug，请参见 gh-23730。 */
    #if __clang__
        #error
    #endif
    // 从命令行参数中加载一个 __m512h 类型的向量 a
    __m512h a = _mm512_loadu_ph((void*)argv[argc-1]);
    // 计算 a * a + a，并将结果保存在 temp 中
    __m512h temp = _mm512_fmadd_ph(a, a, a);
    // 将 temp 的结果存回到命令行参数指向的地址中
    _mm512_storeu_ph((void*)(argv[argc-1]), temp);
    // 返回程序结束状态
    return 0;
}
```