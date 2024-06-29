# `.\numpy\numpy\distutils\checks\cpu_vx.c`

```
#if (__VEC__ < 10301) || (__ARCH__ < 11)
    #error VX not supported
#endif

#include <vecintrin.h>

int main(int argc, char **argv)
{
    // 使用 vec_xl 函数加载 argv 指向的内存中的双精度浮点数向量到 x 中
    __vector double x = vec_abs(vec_xl(argc, (double*)argv));
    
    // 使用 vec_load_len 函数加载 argv 指向的内存中的长度为 argc 的双精度浮点数向量到 y 中
    __vector double y = vec_load_len((double*)argv, (unsigned int)argc);

    // 将 x 向量取整并向上取整后与 y 向量向下取整相加，并将结果赋给 x
    x = vec_round(vec_ceil(x) + vec_floor(y));

    // 使用 vec_cmpge 函数比较 x 和 y 中的元素，将比较结果存储在 m 向量中
    __vector bool long long m = vec_cmpge(x, y);

    // 使用 vec_sel 函数根据 m 向量的值选择 x 或 y 中的元素，并将结果转换为有符号的 long long 类型，存储在 i 中
    __vector long long i = vec_signed(vec_sel(x, y, m));

    // 返回 i 向量中第一个元素的整数值作为函数的返回值
    return (int)vec_extract(i, 0);
}
```