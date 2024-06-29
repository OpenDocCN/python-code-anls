# `.\numpy\numpy\distutils\checks\cpu_vsx3.c`

```
#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

// 定义一个 4 个无符号整数向量的类型
typedef __vector unsigned int v_uint32x4;

int main(void)
{
    // 初始化一个 4 个无符号整数向量都为 0 的向量 z4
    v_uint32x4 z4 = (v_uint32x4){0, 0, 0, 0};
    // 计算 z4 的绝对差值，结果赋值给 z4
    z4 = vec_absd(z4, z4);
    // 提取 z4 向量中索引为 0 的值，并转换为整数返回
    return (int)vec_extract(z4, 0);
}
```