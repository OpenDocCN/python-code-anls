# `.\numpy\numpy\distutils\checks\cpu_asimddp.c`

```py
#ifdef _MSC_VER
    // 如果是 Microsoft Visual C++ 编译器
    #include <Intrin.h>
#endif
// 包含 ARM NEON 指令集的头文件
#include <arm_neon.h>

// 主函数
int main(int argc, char **argv)
{
    // 获取最后一个命令行参数，强制转换为无符号字符指针
    unsigned char *src = (unsigned char*)argv[argc-1];
    // 使用 src[0] 复制成 16 个相同的字节作为 v1
    uint8x16_t v1 = vdupq_n_u8(src[0]), 
    // 使用 src[1] 复制成 16 个相同的字节作为 v2
    v2 = vdupq_n_u8(src[1]);
    // 创建包含四个相同值的向量 va
    uint32x4_t va = vdupq_n_u32(3);
    // 计算 va 和 v1 与 v2 的点积结果，取第一个元素强制转换为整数
    int ret = (int)vgetq_lane_u32(vdotq_u32(va, v1, v2), 0);
#ifdef __aarch64__
    // 如果是 AArch64 架构，再计算一次点积结果并加到 ret 上
    ret += (int)vgetq_lane_u32(vdotq_laneq_u32(va, v1, v2, 0), 0);
#endif
    // 返回计算结果
    return ret;
}
```