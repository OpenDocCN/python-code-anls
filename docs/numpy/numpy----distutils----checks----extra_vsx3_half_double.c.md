# `.\numpy\numpy\distutils\checks\extra_vsx3_half_double.c`

```py
/**
 * 主函数，程序入口点
 */
int main(void)
{
    // 定义并初始化一个无符号短整型变量 bits，赋值为十六进制数 0xFF (255)
    unsigned short bits = 0xFF;
    // 定义一个双精度浮点数变量 f
    double f;
    // 内联汇编语句：将 bits 的低位 16 位转换为双精度浮点数 f
    __asm__ __volatile__("xscvhpdp %x0,%x1" : "=wa"(f) : "wa"(bits));
    // 内联汇编语句：将双精度浮点数 f 转换为 bits 的低位 16 位
    __asm__ __volatile__ ("xscvdphp %x0,%x1" : "=wa" (bits) : "wa" (f));
    // 返回 bits 变量的值作为函数的返回值
    return bits;
}
```