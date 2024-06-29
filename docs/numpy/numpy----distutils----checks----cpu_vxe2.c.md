# `.\numpy\numpy\distutils\checks\cpu_vxe2.c`

```
#if (__VEC__ < 10303) || (__ARCH__ < 13)
    #error VXE2 not supported
#endif

#include <vecintrin.h>

int main(int argc, char **argv)
{
    // 声明一个整型变量 val
    int val;
    // 声明一个包含8个有符号短整型元素的 SIMD 向量 large，初始化为字符 'a', 'b', 'c', 'a', 'g', 'h', 'g', 'o'
    __vector signed short large = { 'a', 'b', 'c', 'a', 'g', 'h', 'g', 'o' };
    // 声明一个包含4个有符号短整型元素的 SIMD 向量 search，初始化为字符 'g', 'h', 'g', 'o'
    __vector signed short search = { 'g', 'h', 'g', 'o' };
    // 声明一个包含8个无符号字符元素的 SIMD 向量 len，初始化为全零
    __vector unsigned char len = { 0 };
    // 调用 vec_search_string_cc 函数，在 large 向量中搜索 search 向量，将结果存储到 res 向量中，val 用于存储返回值
    __vector unsigned char res = vec_search_string_cc(large, search, len, &val);
    // 调用 vec_xl 函数从 argv 数组中加载数据到 SIMD 向量 x 中
    __vector float x = vec_xl(argc, (float*)argv);
    // 调用 vec_signed 函数将 x 向量中的元素转换为有符号整型，并存储到 i 向量中
    __vector int i = vec_signed(x);

    // 对 i 向量中的元素执行向左和向右移位操作
    i = vec_srdb(vec_sldb(i, i, 2), i, 3);
    // 将 res 向量中第二个元素的值加到 val 上
    val += (int)vec_extract(res, 1);
    // 将 i 向量中第一个元素的值加到 val 上
    val += vec_extract(i, 0);
    // 返回 val 作为主函数的返回值
    return val;
}
```