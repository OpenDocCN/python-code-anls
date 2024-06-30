# `D:\src\scipysrc\scipy\scipy\stats\_levy_stable\c_src\levyst.h`

```
#ifndef LEVYST_H
#define LEVYST_H

# 如果未定义 LEVYST_H 宏，则定义它，防止头文件重复包含


struct nolan_precanned
{
    double (*g)(struct nolan_precanned *, double);
    double alpha;
    double zeta;
    double xi;
    double zeta_prefactor;
    double alpha_exp;
    double alpha_xi;
    double zeta_offset;
    double two_beta_div_pi;
    double pi_div_two_beta;
    double x0_div_term;
    double c1;
    double c2;
    double c3;
};

# 定义了一个结构体 `nolan_precanned`，用于存储多个 double 类型的成员变量和一个函数指针成员 `g`


typedef double (*g_callback)(struct nolan_precanned *, double);

# 定义了一个函数指针类型 `g_callback`，该函数指针接受一个 `struct nolan_precanned*` 类型和一个 `double` 类型参数，并返回 `double` 类型值


extern struct nolan_precanned *
nolan_precan(double, double, double);

# 声明了一个函数 `nolan_precan`，该函数返回一个 `struct nolan_precanned*` 类型指针，接受三个 `double` 类型的参数


#endif

# 结束了头文件的条件编译指令
```