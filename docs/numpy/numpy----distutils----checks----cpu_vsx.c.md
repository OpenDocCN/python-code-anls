# `.\numpy\numpy\distutils\checks\cpu_vsx.c`

```py
#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

#if (defined(__GNUC__) && !defined(vec_xl)) || (defined(__clang__) && !defined(__IBMC__))
    #define vsx_ld  vec_vsx_ld
    #define vsx_st  vec_vsx_st
#else
    #define vsx_ld  vec_xl
    #define vsx_st  vec_xst
#endif

int main(void)
{
    // 定义一个无符号整数数组，长度为4
    unsigned int zout[4];
    // 定义并初始化一个包含四个0的整数数组
    unsigned int z4[] = {0, 0, 0, 0};
    // 使用VSX指令加载z4数组中的数据到一个向量寄存器中
    __vector unsigned int v_z4 = vsx_ld(0, z4);
    // 使用VSX指令将向量寄存器v_z4的数据存储到zout数组中
    vsx_st(v_z4, 0, zout);
    // 返回zout数组的第一个元素
    return zout[0];
}
```