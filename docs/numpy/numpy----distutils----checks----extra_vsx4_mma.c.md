# `.\numpy\numpy\distutils\checks\extra_vsx4_mma.c`

```
#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

typedef __vector float fv4sf_t;  // 定义 4 个单精度浮点数向量类型
typedef __vector unsigned char vec_t;  // 定义无符号字符向量类型

int main(void)
{
    __vector_quad acc0;  // 定义一个四元素向量类型 acc0

    float a[4] = {0,1,2,3};  // 定义包含 4 个浮点数的数组 a
    float b[4] = {0,1,2,3};  // 定义包含 4 个浮点数的数组 b

    vec_t *va = (vec_t *) a;  // 将数组 a 转换为 vec_t 类型的指针 va
    vec_t *vb = (vec_t *) b;  // 将数组 b 转换为 vec_t 类型的指针 vb

    __builtin_mma_xvf32ger(&acc0, va[0], vb[0]);  // 使用 MMA 指令进行向量乘法并存储到 acc0 中

    fv4sf_t result[4];  // 定义包含 4 个 fv4sf_t 类型元素的数组 result

    __builtin_mma_disassemble_acc((void *)result, &acc0);  // 将 acc0 的内容解析为 result 数组

    fv4sf_t c0 = result[0];  // 获取 result 数组中的第一个元素赋值给 c0

    return (int)((float*)&c0)[0];  // 将 c0 转换为 float* 类型再转换为 int 类型并返回第一个元素
}
```