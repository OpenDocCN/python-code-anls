# `D:\src\scipysrc\scipy\scipy\special\special\error.h`

```
#pragma once

// 如果正在使用 CUDA 编译器，则定义 SPECFUN_HOST_DEVICE 宏为 __host__ __device__，表示可以在主机和设备上运行
#ifdef __CUDACC__
#define SPECFUN_HOST_DEVICE __host__ __device__
// 否则，定义 SPECFUN_HOST_DEVICE 宏为空，仅限于主机上运行
#else
#define SPECFUN_HOST_DEVICE
#endif

// 定义枚举类型 sf_error_t，用于表示特殊函数库可能遇到的各种错误
typedef enum {
    SF_ERROR_OK = 0,    /* 没有错误 */
    SF_ERROR_SINGULAR,  /* 遇到奇异点 */
    SF_ERROR_UNDERFLOW, /* 浮点下溢 */
    SF_ERROR_OVERFLOW,  /* 浮点上溢 */
    SF_ERROR_SLOW,      /* 需要太多迭代 */
    SF_ERROR_LOSS,      /* 精度损失 */
    SF_ERROR_NO_RESULT, /* 没有获得结果 */
    SF_ERROR_DOMAIN,    /* 输入超出定义域 */
    SF_ERROR_ARG,       /* 无效的输入参数 */
    SF_ERROR_OTHER,     /* 未分类的错误 */
    SF_ERROR__LAST      /* 错误枚举的末尾标记 */
} sf_error_t;

#ifdef __cplusplus

#include <complex>

namespace special {

// 如果未定义 SP_SPECFUN_ERROR，则定义内联函数 set_error，用于设置错误信息（空函数体）
#ifndef SP_SPECFUN_ERROR
SPECFUN_HOST_DEVICE inline void set_error(const char *func_name, sf_error_t code, const char *fmt, ...) {
    // nothing
}
// 否则，声明 set_error 函数，由外部提供实现
#else
void set_error(const char *func_name, sf_error_t code, const char *fmt, ...);
#endif

// 模板函数，当遇到错误时设置错误信息并将变量设为 NaN（对于标量类型 T）
template <typename T>
void set_error_and_nan(const char *name, sf_error_t code, T &value) {
    if (code != SF_ERROR_OK) {
        // 调用 set_error 函数设置错误信息
        set_error(name, code, nullptr);

        // 如果错误是由于定义域、溢出或没有结果，则将 value 设为 NaN
        if (code == SF_ERROR_DOMAIN || code == SF_ERROR_OVERFLOW || code == SF_ERROR_NO_RESULT) {
            value = std::numeric_limits<T>::quiet_NaN();
        }
    }
}

// 模板函数，当遇到错误时设置错误信息并将复数变量的实部和虚部都设为 NaN
template <typename T>
void set_error_and_nan(const char *name, sf_error_t code, std::complex<T> &value) {
    if (code != SF_ERROR_OK) {
        // 调用 set_error 函数设置错误信息
        set_error(name, code, nullptr);

        // 如果错误是由于定义域、溢出或没有结果，则将复数的实部和虚部都设为 NaN
        if (code == SF_ERROR_DOMAIN || code == SF_ERROR_OVERFLOW || code == SF_ERROR_NO_RESULT) {
            value.real(std::numeric_limits<T>::quiet_NaN());
            value.imag(std::numeric_limits<T>::quiet_NaN());
        }
    }
}

} // namespace special

#endif
```