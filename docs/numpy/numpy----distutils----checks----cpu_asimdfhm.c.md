# `.\numpy\numpy\distutils\checks\cpu_asimdfhm.c`

```
#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    // 将最后一个命令行参数解释为 float16_t 类型的指针
    float16_t *src = (float16_t*)argv[argc-1];
    // 将倒数第二个命令行参数解释为 float 类型的指针
    float *src2 = (float*)argv[argc-2];
    
    // 创建一个包含 src[0] 值的 float16x8_t 类型变量
    float16x8_t vhp  = vdupq_n_f16(src[0]);
    // 创建一个包含 src[1] 值的 float16x4_t 类型变量
    float16x4_t vlhp = vdup_n_f16(src[1]);
    // 创建一个包含 src2[0] 值的 float32x4_t 类型变量
    float32x4_t vf   = vdupq_n_f32(src2[0]);
    // 创建一个包含 src2[1] 值的 float32x2_t 类型变量
    float32x2_t vlf  = vdup_n_f32(src2[1]);

    // 计算 vfmlal_low_f16(vlf, vlhp, vlhp) 的第 0 个元素并转换为整数
    int ret  = (int)vget_lane_f32(vfmlal_low_f16(vlf, vlhp, vlhp), 0);
    // 计算 vfmlslq_high_f16(vf, vhp, vhp) 的第 0 个元素并转换为整数，并加到 ret
    ret += (int)vgetq_lane_f32(vfmlslq_high_f16(vf, vhp, vhp), 0);

    // 返回 ret 作为程序的退出码
    return ret;
}
```