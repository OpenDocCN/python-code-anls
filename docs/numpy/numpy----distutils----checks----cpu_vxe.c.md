# `.\numpy\numpy\distutils\checks\cpu_vxe.c`

```py
#if (__VEC__ < 10302) || (__ARCH__ < 12)
    #error VXE not supported
#endif

#include <vecintrin.h>
int main(int argc, char **argv)
{
    // 使用向量指令库中的函数，对参数进行处理
    __vector float x = vec_nabs(vec_xl(argc, (float*)argv));
    // 加载参数指定的数据到向量 x
    __vector float y = vec_load_len((float*)argv, (unsigned int)argc);
    
    // 对向量 x 和 y 中的元素进行取整运算，并将结果存入 x
    x = vec_round(vec_ceil(x) + vec_floor(y));
    // 比较向量 x 和 y 中的元素，生成布尔向量 m，指示 x 中大于等于 y 的元素位置
    __vector bool int m = vec_cmpge(x, y);
    // 根据布尔向量 m，在 x 和 y 中选择元素，将选择结果存回 x
    x = vec_sel(x, y, m);

    // 需要测试是否存在内置函数 "vflls"，因为 vec_doublee 映射到错误的内置函数 "vfll"
    // 参考 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100871
#if defined(__GNUC__) && !defined(__clang__)
    // 使用 GNU 编译器的特定函数对向量 x 进行操作，将结果存入长长整型向量 i
    __vector long long i = vec_signed(__builtin_s390_vflls(x));
#else
    // 否则使用默认的函数 vec_doublee 对向量 x 进行操作，将结果存入长长整型向量 i
    __vector long long i = vec_signed(vec_doublee(x));
#endif

    // 返回向量 i 中第一个元素的整数值
    return (int)vec_extract(i, 0);
}
```