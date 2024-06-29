# `.\numpy\numpy\distutils\checks\cpu_neon_fp16.c`

```py
#ifdef _MSC_VER
    // 如果编译器是 MSC，则包含 Intrinsics 头文件
    #include <Intrin.h>
#endif
// 包含 ARM NEON 头文件，用于使用 NEON 指令集
#include <arm_neon.h>

// 主函数入口，接收命令行参数
int main(int argc, char **argv)
{
    // 将最后一个命令行参数解释为 short 类型指针，并转换为 short 指针
    short *src = (short*)argv[argc-1];
    
    // 使用 NEON 指令将 short 数组转换为 float32x4_t 类型
    float32x4_t v_z4 = vcvt_f32_f16((float16x4_t)vld1_s16(src));
    
    // 返回 v_z4 中第一个元素的整数值
    return (int)vgetq_lane_f32(v_z4, 0);
}
```