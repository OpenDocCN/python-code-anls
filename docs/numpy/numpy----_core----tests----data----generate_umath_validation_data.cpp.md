# `.\numpy\numpy\_core\tests\data\generate_umath_validation_data.cpp`

```py
/*
 * 包含标准库头文件
 */
#include <algorithm>    // 提供算法操作
#include <fstream>      // 提供文件操作
#include <iostream>     // 提供标准输入输出流
#include <math.h>       // 提供数学函数
#include <random>       // 提供随机数生成器
#include <cstdio>       // 提供C风格输入输出
#include <ctime>        // 提供时间函数
#include <vector>       // 提供动态数组

/*
 * 定义结构体ufunc，用于存储数学函数信息
 */
struct ufunc {
    std::string name;                   // 函数名称
    double (*f32func)(double);          // 双精度浮点函数指针
    long double (*f64func)(long double);// 长双精度浮点函数指针
    float f32ulp;                       // 单精度浮点误差限
    float f64ulp;                       // 双精度浮点误差限
};

/*
 * 模板函数RandomFloat，生成指定范围[a, b)内的随机浮点数
 */
template <typename T>
T
RandomFloat(T a, T b)
{
    T random = ((T)rand()) / (T)RAND_MAX;   // 生成0到1之间的随机数
    T diff = b - a;                         // 计算范围差值
    T r = random * diff;                    // 缩放随机数到指定范围
    return a + r;                           // 返回随机数
}

/*
 * 模板函数append_random_array，向向量arr中添加N个指定范围[min, max)内的随机数
 */
template <typename T>
void
append_random_array(std::vector<T> &arr, T min, T max, size_t N)
{
    for (size_t ii = 0; ii < N; ++ii)
        arr.emplace_back(RandomFloat<T>(min, max));  // 调用RandomFloat生成随机数并添加到向量中
}

/*
 * 模板函数computeTrueVal，计算输入向量in中每个元素应用数学函数mathfunc后的结果，并返回结果向量
 */
template <typename T1, typename T2>
std::vector<T1>
computeTrueVal(const std::vector<T1> &in, T2 (*mathfunc)(T2))
{
    std::vector<T1> out;            // 定义输出向量
    for (T1 elem : in) {            // 遍历输入向量中的每个元素
        T2 elem_d = (T2)elem;       // 将元素转换为T2类型
        T1 out_elem = (T1)mathfunc(elem_d);    // 计算元素经过mathfunc函数后的结果
        out.emplace_back(out_elem); // 将结果添加到输出向量中
    }
    return out;                     // 返回结果向量
}

/*
 * FP range:
 * [-inf, -maxflt, -1., -minflt, -minden, 0., minden, minflt, 1., maxflt, inf]
 */

/*
 * 定义各种特殊浮点数常量的宏
 */
#define MINDEN std::numeric_limits<T>::denorm_min()      // 最小非规范化数
#define MINFLT std::numeric_limits<T>::min()             // 最小正浮点数
#define MAXFLT std::numeric_limits<T>::max()             // 最大浮点数
#define INF std::numeric_limits<T>::infinity()           // 正无穷大
#define qNAN std::numeric_limits<T>::quiet_NaN()         // 安静NaN
#define sNAN std::numeric_limits<T>::signaling_NaN()     // 信号NaN

/*
 * 模板函数generate_input_vector，根据函数名称func生成对应的输入向量
 */
template <typename T>
std::vector<T>
generate_input_vector(std::string func)
{
    std::vector<T> input = {MINDEN,  -MINDEN, MINFLT, -MINFLT, MAXFLT,
                            -MAXFLT, INF,     -INF,   qNAN,    sNAN,
                            -1.0,    1.0,     0.0,    -0.0};   // 初始化输入向量

    // 根据函数名称func生成不同范围的随机数并添加到输入向量中
    // [-1.0, 1.0]
    if ((func == "arcsin") || (func == "arccos") || (func == "arctanh")) {
        append_random_array<T>(input, -1.0, 1.0, 700);
    }
    // (0.0, INF]
    else if ((func == "log2") || (func == "log10")) {
        append_random_array<T>(input, 0.0, 1.0, 200);
        append_random_array<T>(input, MINDEN, MINFLT, 200);
        append_random_array<T>(input, MINFLT, 1.0, 200);
        append_random_array<T>(input, 1.0, MAXFLT, 200);
    }
    // (-1.0, INF]
    else if (func == "log1p") {
        append_random_array<T>(input, -1.0, 1.0, 200);
        append_random_array<T>(input, -MINFLT, -MINDEN, 100);
        append_random_array<T>(input, -1.0, -MINFLT, 100);
        append_random_array<T>(input, MINDEN, MINFLT, 100);
        append_random_array<T>(input, MINFLT, 1.0, 100);
        append_random_array<T>(input, 1.0, MAXFLT, 100);
    }
    // [1.0, INF]
    else if (func == "arccosh") {
        append_random_array<T>(input, 1.0, 2.0, 400);
        append_random_array<T>(input, 2.0, MAXFLT, 300);
    }
    // [-INF, INF]
    // 否则分支：在 input 后追加不同范围的随机数组成分
    append_random_array<T>(input, -1.0, 1.0, 100);
    append_random_array<T>(input, MINDEN, MINFLT, 100);
    append_random_array<T>(input, -MINFLT, -MINDEN, 100);
    append_random_array<T>(input, MINFLT, 1.0, 100);
    append_random_array<T>(input, -1.0, -MINFLT, 100);
    append_random_array<T>(input, 1.0, MAXFLT, 100);
    append_random_array<T>(input, -MAXFLT, -100.0, 100);

    // 对 input 中的元素进行随机重排
    std::random_shuffle(input.begin(), input.end());
    // 返回重排后的 input 数组
    return input;
}
```