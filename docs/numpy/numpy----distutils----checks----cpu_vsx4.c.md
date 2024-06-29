# `.\numpy\numpy\distutils\checks\cpu_vsx4.c`

```py
#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

// 定义一个 4 个无符号整数向量类型，用于操作 4 个 32 位整数
typedef __vector unsigned int v_uint32x4;

// 主函数入口
int main(void)
{
    // 初始化一个包含 2, 4, 8, 16 的向量 v1
    v_uint32x4 v1 = (v_uint32x4){2, 4, 8, 16};
    // 初始化一个包含 2, 2, 2, 2 的向量 v2
    v_uint32x4 v2 = (v_uint32x4){2, 2, 2, 2};
    // 对 v1 和 v2 执行向量模运算，将结果存入 v3
    v_uint32x4 v3 = vec_mod(v1, v2);
    // 提取 v3 的有效位并转换为整数返回
    return (int)vec_extractm(v3);
}
```