# `.\numpy\numpy\distutils\checks\cpu_neon_vfpv4.c`

```
#ifdef _MSC_VER
    #include <Intrin.h>  // 如果是 MSC 编译器，则包含 Intrin 头文件
#endif
#include <arm_neon.h>  // 包含 ARM NEON 头文件

int main(int argc, char **argv)  // 主函数
{
    float *src = (float*)argv[argc-1];  // 获取参数中的最后一个元素，转换为 float 指针
    float32x4_t v1 = vdupq_n_f32(src[0]);  // 将 src[0] 的值复制到一个四元素的向量中
    float32x4_t v2 = vdupq_n_f32(src[1]);  // 将 src[1] 的值复制到一个四元素的向量中
    float32x4_t v3 = vdupq_n_f32(src[2]);  // 将 src[2] 的值复制到一个四元素的向量中
    int ret = (int)vgetq_lane_f32(vfmaq_f32(v1, v2, v3), 0);  // 计算 v1 * v2 + v3 的结果并将第一个值转换为整数

#ifdef __aarch64__
    double *src2 = (double*)argv[argc-2];  // 如果是 aarch64 架构，则获取倒数第二个参数，转换为 double 指针
    float64x2_t vd1 = vdupq_n_f64(src2[0]);  // 将 src2[0] 的值复制到一个双元素的向量中
    float64x2_t vd2 = vdupq_n_f64(src2[1]);  // 将 src2[1] 的值复制到一个双元素的向量中
    float64x2_t vd3 = vdupq_n_f64(src2[2]);  // 将 src2[2] 的值复制到一个双元素的向量中
    ret += (int)vgetq_lane_f64(vfmaq_f64(vd1, vd2, vd3), 0);  // 计算 vd1 * vd2 + vd3 的结果并将第一个值转换为整数
#endif

    return ret;  // 返回 ret 的值作为函数结果
}
```