# `D:\src\scipysrc\scipy\scipy\fftpack\tests\fftw_dct.c`

```
#include <stdlib.h>
#include <stdio.h>

#include <fftw3.h>

#if DCT_TEST_PRECISION == 1
typedef float float_prec;                   // 定义浮点数精度为单精度浮点数
#define PF "%.7f"                           // 定义打印格式为小数点后7位的单精度浮点数
#define FFTW_PLAN fftwf_plan                // 定义 FFTW 的计划类型为单精度版本
#define FFTW_MALLOC fftwf_malloc            // 定义 FFTW 的内存分配函数为单精度版本
#define FFTW_FREE fftwf_free                // 定义 FFTW 的内存释放函数为单精度版本
#define FFTW_PLAN_CREATE fftwf_plan_r2r_1d  // 定义 FFTW 创建实数到实数的一维计划为单精度版本
#define FFTW_EXECUTE fftwf_execute          // 定义 FFTW 执行函数为单精度版本
#define FFTW_DESTROY_PLAN fftwf_destroy_plan// 定义 FFTW 计划销毁函数为单精度版本
#define FFTW_CLEANUP fftwf_cleanup          // 定义 FFTW 清理函数为单精度版本
#elif DCT_TEST_PRECISION == 2
typedef double float_prec;                  // 定义浮点数精度为双精度浮点数
#define PF "%.18f"                          // 定义打印格式为小数点后18位的双精度浮点数
#define FFTW_PLAN fftw_plan                 // 定义 FFTW 的计划类型为双精度版本
#define FFTW_MALLOC fftw_malloc             // 定义 FFTW 的内存分配函数为双精度版本
#define FFTW_FREE fftw_free                 // 定义 FFTW 的内存释放函数为双精度版本
#define FFTW_PLAN_CREATE fftw_plan_r2r_1d   // 定义 FFTW 创建实数到实数的一维计划为双精度版本
#define FFTW_EXECUTE fftw_execute           // 定义 FFTW 执行函数为双精度版本
#define FFTW_DESTROY_PLAN fftw_destroy_plan // 定义 FFTW 计划销毁函数为双精度版本
#define FFTW_CLEANUP fftw_cleanup           // 定义 FFTW 清理函数为双精度版本
#elif DCT_TEST_PRECISION == 3
typedef long double float_prec;             // 定义浮点数精度为长双精度浮点数
#define PF "%.18Lf"                         // 定义打印格式为小数点后18位的长双精度浮点数
#define FFTW_PLAN fftwl_plan                // 定义 FFTW 的计划类型为长双精度版本
#define FFTW_MALLOC fftwl_malloc            // 定义 FFTW 的内存分配函数为长双精度版本
#define FFTW_FREE fftwl_free                // 定义 FFTW 的内存释放函数为长双精度版本
#define FFTW_PLAN_CREATE fftwl_plan_r2r_1d  // 定义 FFTW 创建实数到实数的一维计划为长双精度版本
#define FFTW_EXECUTE fftwl_execute          // 定义 FFTW 执行函数为长双精度版本
#define FFTW_DESTROY_PLAN fftwl_destroy_plan// 定义 FFTW 计划销毁函数为长双精度版本
#define FFTW_CLEANUP fftwl_cleanup          // 定义 FFTW 清理函数为长双精度版本
#else
#error DCT_TEST_PRECISION must be a number 1-3  // 如果 DCT_TEST_PRECISION 不在1到3之间，抛出错误信息
#endif


enum type {
        DCT_I = 1,      // 离散余弦变换类型I
        DCT_II = 2,     // 离散余弦变换类型II
        DCT_III = 3,    // 离散余弦变换类型III
        DCT_IV = 4,     // 离散余弦变换类型IV
        DST_I = 5,      // 离散正弦变换类型I
        DST_II = 6,     // 离散正弦变换类型II
        DST_III = 7,    // 离散正弦变换类型III
        DST_IV = 8,     // 离散正弦变换类型IV
};

int gen(int type, int sz)  // 定义函数 gen，接收两个整数参数：变换类型和大小
{
        // 声明指向浮点数的指针 a 和 b
        float_prec *a, *b;
        // 声明 FFTW 的计划 p
        FFTW_PLAN p;
        // 声明整数变量 i 和 tp
        int i, tp;

        // 分配大小为 sz 个浮点数的内存给 a
        a = FFTW_MALLOC(sizeof(*a) * sz);
        // 如果分配失败，打印错误消息并退出程序
        if (a == NULL) {
                fprintf(stderr, "failure\n");
                exit(EXIT_FAILURE);
        }
        // 分配大小为 sz 个浮点数的内存给 b
        b = FFTW_MALLOC(sizeof(*b) * sz);
        // 如果分配失败，打印错误消息并退出程序
        if (b == NULL) {
                fprintf(stderr, "failure\n");
                exit(EXIT_FAILURE);
        }

        // 根据 type 的值选择对应的 DCT 或 DST 类型
        switch(type) {
                // DCT 类型选择不同的变换类型
                case DCT_I:
                        tp = FFTW_REDFT00;
                        break;
                case DCT_II:
                        tp = FFTW_REDFT10;
                        break;
                case DCT_III:
                        tp = FFTW_REDFT01;
                        break;
                case DCT_IV:
                        tp = FFTW_REDFT11;
                        break;
                // DST 类型选择不同的变换类型
                case DST_I:
                        tp = FFTW_RODFT00;
                        break;
                case DST_II:
                        tp = FFTW_RODFT10;
                        break;
                case DST_III:
                        tp = FFTW_RODFT01;
                        break;
                case DST_IV:
                        tp = FFTW_RODFT11;
                        break;
                // 如果 type 值未知，打印错误消息并退出程序
                default:
                        fprintf(stderr, "unknown type\n");
                        exit(EXIT_FAILURE);
        }

        // 根据 type 的值初始化数组 a
        switch(type) {
            // 对于 DCT 类型和 DST 类型，将数组 a 初始化
            case DCT_I:
            case DCT_II:
            case DCT_III:
            case DCT_IV:
                for(i=0; i < sz; ++i) {
                    a[i] = i;
                }
                break;
            case DST_I:
            case DST_II:
            case DST_III:
            case DST_IV:
/*                TODO: what should we do for dst's?*/
                // 对于 DST 类型，将数组 a 初始化
                for(i=0; i < sz; ++i) {
                    a[i] = i;
                }
                break;
            // 如果 type 值未知，打印错误消息并退出程序
            default:
                fprintf(stderr, "unknown type\n");
                exit(EXIT_FAILURE);
        }

        // 创建 FFTW 的计划 p
        p = FFTW_PLAN_CREATE(sz, a, b, tp, FFTW_ESTIMATE);
        // 执行 FFT 变换
        FFTW_EXECUTE(p);
        // 销毁 FFTW 的计划 p
        FFTW_DESTROY_PLAN(p);

        // 打印数组 b 中的结果
        for(i=0; i < sz; ++i) {
                printf(PF"\n", b[i]);
        }
        // 释放数组 b 和 a 的内存
        FFTW_FREE(b);
        FFTW_FREE(a);

        // 返回成功退出码
        return 0;
}

// 主函数，接受命令行参数，执行相关的函数调用
int main(int argc, char* argv[])
{
        // 声明整数变量 n 和 tp
        int n, tp;

        // 如果命令行参数少于 3 个，打印错误消息并退出程序
        if (argc < 3) {
                fprintf(stderr, "missing argument: program type n\n");
                exit(EXIT_FAILURE);
        }
        // 将命令行参数转换为整数，分别赋值给 tp 和 n
        tp = atoi(argv[1]);
        n = atoi(argv[2]);

        // 调用 gen 函数，根据 tp 和 n 执行相应操作
        gen(tp, n);
        // 清理 FFTW 库的状态
        FFTW_CLEANUP();

        // 返回成功退出码
        return 0;
}
```