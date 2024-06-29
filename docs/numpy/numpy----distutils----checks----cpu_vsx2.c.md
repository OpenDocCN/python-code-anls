# `.\numpy\numpy\distutils\checks\cpu_vsx2.c`

```py
#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

// 定义了一个名为v_uint64x2的类型别名，代表一个包含两个无符号长整型元素的向量
typedef __vector unsigned long long v_uint64x2;

int main(void)
{
    // 创建一个v_uint64x2类型的变量z2，并初始化为(0, 0)
    v_uint64x2 z2 = (v_uint64x2){0, 0};
    // 将z2与自身进行逐元素比较，将比较结果存储回z2中
    z2 = (v_uint64x2)vec_cmpeq(z2, z2);
    // 提取z2中索引为0的元素，并转换为int类型后返回
    return (int)vec_extract(z2, 0);
}
```