# `.\numpy\numpy\_core\src\npymath\arm64_exports.c`

```py
#if defined(__arm64__) && defined(__APPLE__)
#include <math.h>
/*
 * 对于 macOS arm64 平台，导出这些函数供 SciPy 使用。
 * SciPy 在 macOS arm64 上构建时，会下载 macOS x86_64 平台的 NumPy，
 * 并且链接器在无法使用 npymathlib.a 时不会报错。导入 numpy 会暴露这些外部函数。
 * 参考 https://github.com/numpy/numpy/issues/22673#issuecomment-1327520055
 *
 * 实际上，这个文件会作为主模块的一部分进行编译。
 */

// 计算反双曲正弦值
double npy_asinh(double x) {
    return asinh(x);
}

// 返回带有给定符号的 y 的值
double npy_copysign(double y, double x) {
    return copysign(y, x);
}

// 返回 log(1+x)
double npy_log1p(double x) {
    return log1p(x);
}

// 返回靠近 x 但大于 y 的浮点数
double npy_nextafter(double x, double y) {
    return nextafter(x, y);
}

#endif
```