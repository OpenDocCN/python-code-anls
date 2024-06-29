# `.\numpy\numpy\distutils\checks\cpu_rvv.c`

```py
#ifndef __riscv_vector
  #error RVV not supported  // 如果未定义 __riscv_vector，抛出错误提示RVV不支持
#endif

#include <riscv_vector.h>  // 包含riscv_vector头文件

int main(void)
{
    size_t vlmax = __riscv_vsetvlmax_e32m1();  // 设置向量最大长度为e32m1，并返回该长度
    vuint32m1_t a = __riscv_vmv_v_x_u32m1(0, vlmax);  // 将值0移动至vuint32m1_t类型的向量a中
    vuint32m1_t b = __riscv_vadd_vv_u32m1(a, a, vlmax);  // 将向量a和向量a相加，结果存储在向量b中
    return __riscv_vmv_x_s_u32m1_u32(b);  // 将向量b转换为u32类型并返回
}
```