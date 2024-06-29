# `.\numpy\numpy\distutils\checks\cpu_asimd.c`

```py
#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    // 将最后一个命令行参数视为浮点数数组的起始地址
    float *src = (float*)argv[argc-1];
    // 创建两个包含相同值的 NEON 浮点向量
    float32x4_t v1 = vdupq_n_f32(src[0]), v2 = vdupq_n_f32(src[1]);
    
    /* MAXMIN */
    // 计算 v1 和 v2 向量的最大非数字和和最小非数字和，并取第一个元素转换为整数加到 ret 中
    int ret  = (int)vgetq_lane_f32(vmaxnmq_f32(v1, v2), 0);
    // 计算 v1 和 v2 向量的最小非数字和，并取第一个元素转换为整数加到 ret 中
    ret += (int)vgetq_lane_f32(vminnmq_f32(v1, v2), 0);
    
    /* ROUNDING */
    // 对 v1 向量进行舍入操作，并取第一个元素转换为整数加到 ret 中
    ret += (int)vgetq_lane_f32(vrndq_f32(v1), 0);
    
#ifdef __aarch64__
    {
        // 将最后一个命令行参数视为双精度浮点数数组的起始地址（仅在 ARM 64 位架构下）
        double *src2 = (double*)argv[argc-1];
        // 创建两个包含相同值的 NEON 双精度向量
        float64x2_t vd1 = vdupq_n_f64(src2[0]), vd2 = vdupq_n_f64(src2[1]);
        
        /* MAXMIN */
        // 计算 vd1 和 vd2 向量的最大非数字和和最小非数字和，并取第一个元素转换为整数加到 ret 中
        ret += (int)vgetq_lane_f64(vmaxnmq_f64(vd1, vd2), 0);
        // 计算 vd1 和 vd2 向量的最小非数字和，并取第一个元素转换为整数加到 ret 中
        ret += (int)vgetq_lane_f64(vminnmq_f64(vd1, vd2), 0);
        
        /* ROUNDING */
        // 对 vd1 向量进行舍入操作，并取第一个元素转换为整数加到 ret 中
        ret += (int)vgetq_lane_f64(vrndq_f64(vd1), 0);
    }
#endif
    // 返回累加结果作为 main 函数的返回值
    return ret;
}
```