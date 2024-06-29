# `.\numpy\numpy\distutils\checks\cpu_asimdhp.c`

```py
#ifdef _MSC_VER
    #include <Intrin.h>
#endif

#include <arm_neon.h>

int main(int argc, char **argv)
{
    // 将命令行参数最后一个作为 float16_t 类型数组的起始地址
    float16_t *src = (float16_t*)argv[argc-1];

    // 创建一个包含 src[0] 值的 float16x8_t 类型变量 vhp
    float16x8_t vhp  = vdupq_n_f16(src[0]);

    // 创建一个包含 src[1] 值的 float16x4_t 类型变量 vlhp
    float16x4_t vlhp = vdup_n_f16(src[1]);

    // 计算 vhp 与自身的绝对差，提取结果的第一个元素并转换为整数，赋给 ret
    int ret  =  (int)vgetq_lane_f16(vabdq_f16(vhp, vhp), 0);

    // 计算 vlhp 与自身的绝对差，提取结果的第一个元素并转换为整数，累加到 ret
    ret  += (int)vget_lane_f16(vabd_f16(vlhp, vlhp), 0);

    // 返回累加结果 ret
    return ret;
}
```