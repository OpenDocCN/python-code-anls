# `D:\src\scipysrc\scipy\scipy\special\_wright.cxx`

```
#include "_wright.h"
// 包含名为 _wright.h 的头文件，这里假设它包含了某些程序中需要的声明和定义

#include <complex>
// 包含复数操作相关的标准头文件

using namespace std;
// 使用标准命名空间，使得标准库中的符号可以直接使用

extern "C" {
// 使用 C 语言的链接约定，确保函数名在编译后不会被 C++ 的名称修饰

npy_cdouble wrightomega(npy_cdouble zp)
{
    // 将传入的 npy_cdouble 类型的复数 zp 转换为 C++ 中的复数对象
    complex<double> z(npy_creal(zp), npy_cimag(zp));

    // 调用 wright::wrightomega 函数计算 wrightomega(z) 的值
    complex<double> w = wright::wrightomega(z);

    // 将 C++ 中的复数 w 转换为 npy_cdouble 类型返回
    return npy_cpack(real(w), imag(w));
}

double wrightomega_real(double x)
{
    // 调用 wright::wrightomega_real 函数计算 wrightomega_real(x) 的值并返回
    return wright::wrightomega_real(x);
}

}  // extern "C"
// 结束 extern "C" 块，标志着 C 语言链接约定的结束
```