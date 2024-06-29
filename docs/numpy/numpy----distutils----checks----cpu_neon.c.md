# `.\numpy\numpy\distutils\checks\cpu_neon.c`

```
#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    // passing from untraced pointers to avoid optimizing out any constants
    // so we can test against the linker.
    // 将传递的指针转换为未跟踪的指针，以避免优化掉任何常量
    // 从命令行参数中获取最后一个参数作为浮点数数组的指针
    float *src = (float*)argv[argc-1];
    // 使用首个浮点数创建四个相同值的 Neon 浮点向量 v1 和 v2
    float32x4_t v1 = vdupq_n_f32(src[0]), v2 = vdupq_n_f32(src[1]);
    // 计算向量 v1 和 v2 的乘积，并取结果向量的第一个元素作为整数
    int ret = (int)vgetq_lane_f32(vmulq_f32(v1, v2), 0);
#ifdef __aarch64__
    // 如果是 ARM64 架构，处理双精度浮点数
    // 从命令行参数中获取倒数第二个参数作为双精度浮点数数组的指针
    double *src2 = (double*)argv[argc-2];
    // 使用首个双精度浮点数创建两个相同值的 Neon 双精度向量 vd1 和 vd2
    float64x2_t vd1 = vdupq_n_f64(src2[0]), vd2 = vdupq_n_f64(src2[1]);
    // 计算双精度向量 vd1 和 vd2 的乘积，并取结果向量的第一个元素作为整数
    ret += (int)vgetq_lane_f64(vmulq_f64(vd1, vd2), 0);
#endif
    // 返回计算结果
    return ret;
}
```