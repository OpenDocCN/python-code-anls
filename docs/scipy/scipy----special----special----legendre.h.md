# `D:\src\scipysrc\scipy\scipy\special\special\legendre.h`

```
// 一次性包含头文件，确保本文件只被编译一次
#pragma once

// 包含第三方库cephes中的头文件poch.h和自定义的配置文件config.h
#include "cephes/poch.h"
#include "config.h"

// 声明特定的命名空间special
namespace special {

// 2024年由SciPy开发人员翻译成C++
//
// ===============================================================
// Purpose: 计算Legendre多项式Pn(z)及其导数Pn'(z)
// Input :  x --- Pn(z)的参数
//          n --- Pn(z)的阶数 ( n = 0,1,...)
// Output:  p(n) --- Pn(z)
// ===============================================================
template <typename T, typename OutputVec>
void legendre_all(T z, OutputVec p) {
    // 计算多项式的最高阶数
    long n = p.extent(0) - 1;

    // 初始化P0
    T p0 = 1;
    p(0) = p0;

    // 如果阶数大于等于1，计算更高阶的多项式
    if (n >= 1) {
        // 初始化P1
        T p1 = z;
        p(1) = p1;

        // 计算阶数大于等于2的所有多项式
        for (long k = 2; k <= n; k++) {
            T pf = (static_cast<T>(2 * k - 1) * z * p1 - static_cast<T>(k - 1) * p0) / static_cast<T>(k);
            p(k) = pf;

            // 更新P0和P1
            p0 = p1;
            p1 = pf;
        }
    }
}

// 2024年由SciPy开发人员翻译成C++
// 原始注释在下面显示
//
// =======================================================================
// Purpose: 计算实数参数下的关联Legendre函数Pmn(x)及其导数Pmn'(x)
// Input :  x  --- Pmn(x)的参数
//          m  --- Pmn(x)的阶数, m = 0,1,2,...,n
//          n  --- Pmn(x)的次数, n = 0,1,2,...,N
//          mm --- PM和PD的物理维度
// Output:  PM(m,n) --- Pmn(x)
//          PD(m,n) --- Pmn'(x)
// =======================================================================
template <typename T, typename OutputMat>
void assoc_legendre_all(T x, OutputMat p) {
    // 计算矩阵PM的行数m和列数n
    long m = p.extent(0) - 1;
    long n = p.extent(1) - 1;

    // 初始化PM为零矩阵
    for (long i = 0; i <= m; ++i) {
        for (long j = 0; j <= n; ++j) {
            p(i, j) = 0;
        }
    }

    // 初始化P00为1
    p(0, 0) = 1;
    // 如果 n 大于 0，则进入条件判断
    if (n > 0) {
        // 如果 x 的绝对值等于 1，则执行以下代码块
        if (std::abs(x) == 1) {
            // 对于 i 从 1 到 n 的循环，计算 x 的 i 次幂，并存储在 p(0, i) 中
            for (long i = 1; i <= n; i++) {
                p(0, i) = std::pow(x, i);
            }
        } else {
            // 计算 ls 的值，根据 x 的绝对值大小决定 ls 的正负
            long ls = (std::abs(x) > 1 ? -1 : 1);
            // 计算 xq，它是 ls * sqrt(1 - x^2)，以保证与 |x| > 1 时的复数函数连接
            T xq = std::sqrt(ls * (1 - x * x));
            // 如果 x 小于 -1，则将 xq 取负，以确保与 |x| > 1 时的复数函数连接
            if (x < -1) {
                xq = -xq;
            }

            // 对于 i 从 1 到 m 的循环，计算并存储 p(i, i) 的值
            p(i, i) = -ls * (2 * i - 1) * xq * p(i - 1, i - 1);

            // 对于 i 从 0 到 min(n - 1, m) 的循环，计算并存储 p(i, i+1) 的值
            p(i, i + 1) = (2 * i + 1) * x * p(i, i);

            // 嵌套循环，计算并存储 p(i, j) 的值，其中 i 从 0 到 m，j 从 i+2 到 n
            for (long i = 0; i <= m; i++) {
                for (long j = i + 2; j <= n; j++) {
                    p(i, j) = ((2 * j - 1) * x * p(i, j - 1) - static_cast<T>(i + j - 1) * p(i, j - 2)) /
                              static_cast<T>(j - i);
                }
            }
        }
    }
template <typename T, typename OutputMat>
void assoc_legendre_all(T x, bool m_signbit, OutputMat p) {
    // 调用已定义的函数，计算关联勒让德多项式
    assoc_legendre_all(x, p);

    // 获取矩阵维度
    int m = p.extent(0) - 1;
    int n = p.extent(1) - 1;

    // 如果 m_signbit 为真，执行以下操作
    if (m_signbit) {
        // 循环遍历矩阵中的每个元素
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= m; ++i) {
                T fac = 0;
                // 如果 i <= j，计算系数 fac
                if (i <= j) {
                    fac = std::tgamma(j - i + 1) / std::tgamma(j + i + 1);
                    // 如果 abs(x) < 1，乘以 (-1)^i
                    if (std::abs(x) < 1) {
                        fac *= std::pow(-1, i);
                    }
                }

                // 将 p(i, j) 乘以 fac
                p(i, j) *= fac;
            }
        }
    }
}

template <typename T, typename InputMat, typename OutputMat>
void assoc_legendre_all_jac(T x, InputMat pm, OutputMat pd) {
    // 获取矩阵维度
    int m = pm.extent(0) - 1;
    int n = pm.extent(1) - 1;

    // 初始化输出矩阵 pd 的所有元素为 0
    for (int i = 0; i < m + 1; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            pd(i, j) = 0;
        }
    }

    // 如果 n == 0，直接返回
    if (n == 0) {
        return;
    }

    // 如果 abs(x) == 1，执行以下操作
    if (std::abs(x) == 1) {
        for (int i = 1; i <= n; i++) {
            // 计算特定公式给出的值并赋给 pd(0, i)
            pd(0, i) = i * (i + 1) * std::pow(x, i + 1) / 2;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 1) {
                    // 对于 i == 1，pd(1, j) 被设为正无穷
                    pd(1, j) = std::numeric_limits<T>::infinity();
                } else if (i == 2) {
                    // 对于 i == 2，使用特定公式计算并赋给 pd(2, j)
                    pd(2, j) = -(j + 2) * (j + 1) * j * (j - 1) * std::pow(x, j + 1) / 4;
                }
            }
        }
        return;
    }

    // 计算 ls 和 xq 的值
    int ls = (std::abs(x) > 1 ? -1 : 1);
    T xq = std::sqrt(ls * (1 - x * x));
    // 对于 x < -1，确保与复数值函数的连接
    if (x < -1) {
        xq = -xq;
    }
    T xs = ls * (1 - x * x);

    // 初始化 pd(0, 0)
    pd(0, 0) = 0;
    // 计算 pd(0, j) 的值
    for (int j = 1; j <= n; j++) {
        pd(0, j) = ls * j * (pm(0, j - 1) - x * pm(0, j)) / xs;
    }

    // 计算 pd(i, j) 的值，其中 i 和 j 从 1 到 m 和 n
    for (int i = 1; i <= m; i++) {
        for (int j = i; j <= n; j++) {
            pd(i, j) = ls * i * x * pm(i, j) / xs + (j + i) * (j - i + 1) / xq * pm(i - 1, j);
        }
    }
}

template <typename T, typename InputMat, typename OutputMat>
void assoc_legendre_all_jac(T x, bool m_signbit, InputMat p, OutputMat p_jac) {
    // 获取矩阵维度
    long m = p.extent(0) - 1;
    long n = p.extent(1) - 1;

    // 调用已定义的函数，计算关联勒让德多项式的雅可比矩阵
    assoc_legendre_all_jac(x, p, p_jac);

    // 如果 m_signbit 为真，执行以下操作
    if (m_signbit) {
        // 循环遍历矩阵中的每个元素
        for (long j = 0; j <= n; ++j) {
            for (long i = 0; i <= std::min(j, m); ++i) {
                // 计算系数 fac
                T fac = std::tgamma(j - i + 1) / std::tgamma(j + i + 1);
                // 如果 abs(x) < 1，乘以 (-1)^i
                if (std::abs(x) < 1) {
                    fac *= std::pow(-1, i);
                }

                // 将 p_jac(i, j) 乘以 fac
                p_jac(i, j) *= fac;
            }
        }
    }
}

template <typename T, typename OutMat>
void sph_legendre_all(T phi, OutMat p) {
    // 获取矩阵维度
    long n = p.extent(1) - 1;

    // 调用已定义的函数，计算球谐函数的关联勒让德多项式
    assoc_legendre_all(std::cos(phi), p);

    // 计算球谐函数的系数
    for (long j = 0; j <= n; ++j) {
        for (long i = 0; i <= j; ++i) {
            p(i, j) *= std::sqrt((2 * j + 1) * cephes::poch(j + i + 1, -2 * i) / (4 * M_PI));
        }
    }
}
// Translated into C++ by SciPy developers in 2024.
// 2024年由SciPy开发者翻译成C++
//
// =========================================================
// Purpose: Compute the associated Legendre functions Pmn(z)
//          and their derivatives Pmn'(z) for a complex
//          argument
// 目的：计算复数参数z的相关Legendre函数Pmn(z)及其导数Pmn'(z)
// Input :  x     --- Real part of z
//          y     --- Imaginary part of z
//          m     --- Order of Pmn(z),  m = 0,1,2,...,n
//          n     --- Degree of Pmn(z), n = 0,1,2,...,N
//          mm    --- Physical dimension of CPM and CPD
//          ntype --- type of cut, either 2 or 3
// 输入：x     --- z的实部
//       y     --- z的虚部
//       m     --- Pmn(z)的阶数，m = 0,1,2,...,n
//       n     --- Pmn(z)的次数，n = 0,1,2,...,N
//       mm    --- CPM和CPD的物理维度
//       ntype --- 切割类型，为2或3
// Output:  CPM(m,n) --- Pmn(z)
//          CPD(m,n) --- Pmn'(z)
// 输出：CPM(m,n) --- Pmn(z)
//       CPD(m,n) --- Pmn'(z)
// =========================================================

template <typename T, typename OutputMat1, typename OutputMat2>
void clpmn(std::complex<T> z, long ntype, OutputMat1 cpm, OutputMat2 cpd) {
    // 计算CPM和CPD矩阵的尺寸
    int m = cpm.extent(0) - 1;
    int n = cpm.extent(1) - 1;

    // 初始化CPM和CPD矩阵
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            cpm(i, j) = 0;
            cpd(i, j) = 0;
        }
    }

    // 设置初值 P_0^0(z) = 1
    cpm(0, 0) = 1;
    if (n == 0) {
        return;
    }

    // 如果 z 是实数且在单位圆上，特殊处理
    if ((std::abs(std::real(z)) == 1) && (std::imag(z) == 0)) {
        // 计算 P_0^i(z) 和 P_0^i'(z)
        for (int i = 1; i <= n; i++) {
            cpm(0, i) = std::pow(std::real(z), i);
            cpd(0, i) = i * (i + 1) * std::pow(std::real(z), i + 1) / 2;
        }
        // 对于 i = 1 和 i = 2 的特殊处理
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 1) {
                    cpd(i, j) = std::numeric_limits<T>::infinity();
                } else if (i == 2) {
                    cpd(i, j) = -(j + 2) * (j + 1) * j * (j - 1) * std::pow(std::real(z), j + 1) / 4;
                }
            }
        }
        return;
    }

    // 计算平方根 zq 和符号变量 ls
    std::complex<T> zq, zs;
    int ls;
    if (ntype == 2) {
        // 对于切割类型2，计算 sqrt(1 - z**2)
        zs = (static_cast<T>(1) - z * z);
        zq = -std::sqrt(zs);
        ls = -1;
    } else {
        // 对于切割类型3，计算 sqrt(z**2 - 1)
        zs = (z * z - static_cast<T>(1));
        zq = std::sqrt(zs);
        if (std::real(z) < 0) {
            zq = -zq;
        }
        ls = 1;
    }

    // 计算对角线上的元素 P_i^i(z)
    for (int i = 1; i <= m; i++) {
        cpm(i, i) = static_cast<T>(2 * i - 1) * zq * cpm(i - 1, i - 1);
    }

    // 计算第一次超对角线的元素 P_i^(i+1)(z)
    for (int i = 0; i <= (m > n - 1 ? n - 1 : m); i++) {
        cpm(i, i + 1) = static_cast<T>(2 * i + 1) * z * cpm(i, i);
    }

    // 计算其它元素 P_i^j(z) (i < j)
    for (int i = 0; i <= m; i++) {
        for (int j = i + 2; j <= n; j++) {
            cpm(i, j) = (static_cast<T>(2 * j - 1) * z * cpm(i, j - 1) - static_cast<T>(i + j - 1) * cpm(i, j - 2)) /
                        static_cast<T>(j - i);
        }
    }

    // 计算导数 CPD(0,j) (j >= 1)
    cpd(0, 0) = 0;
    for (int j = 1; j <= n; j++) {
        cpd(0, j) = ls * static_cast<T>(j) * (z * cpm(0, j) - cpm(0, j - 1)) / zs;
    }
    // 循环变量 i 从 1 到 m，对每个 i 进行迭代
    for (int i = 1; i <= m; i++) {
        // 循环变量 j 从 i 到 n，对每个 j 进行迭代
        for (int j = i; j <= n; j++) {
            // 使用 DLMF 14.7.11 和 DLMF 14.10.6 推导的类型 3 的导数
            // 使用 DLMF 14.7.8 和 DLMF 14.10.1 推导的类型 2 的导数
            // 计算 cpd(i, j) 的值，其中 ls 是静态类型转换后的常量，T 是模板类型
            cpd(i, j) = static_cast<T>(ls) * (-static_cast<T>(i) * z * cpm(i, j) / zs +
                                              static_cast<T>((j + i) * (j - i + 1)) / zq * cpm(i - 1, j));
        }
    }
}

// 结束函数 clpmn 的定义

template <typename T, typename OutputMat1, typename OutputMat2>
void clpmn(std::complex<T> z, long ntype, bool m_signbit, OutputMat1 cpm, OutputMat2 cpd) {
    // 调用重载的 clpmn 函数，计算 Legendre 函数
    clpmn(z, ntype, cpm, cpd);

    // 获取输出矩阵的维度
    int m = cpm.extent(0) - 1;
    int n = cpm.extent(1) - 1;

    // 如果 m_signbit 为 true，则执行下面的操作
    if (m_signbit) {
        // 遍历输出矩阵的每个元素
        for (int j = 0; j < n + 1; ++j) {
            for (int i = 0; i < m + 1; ++i) {
                T fac = 0;
                // 如果 i 小于等于 j，则计算 fac 的值
                if (i <= j) {
                    fac = std::tgamma(j - i + 1) / std::tgamma(j + i + 1);
                    // 如果 ntype 等于 2，则额外乘以 -1 的幂次方
                    if (ntype == 2) {
                        fac *= std::pow(-1, i);
                    }
                }

                // 修改 cpm 和 cpd 的对应元素值
                cpm(i, j) *= fac;
                cpd(i, j) *= fac;
            }
        }
    }
}

// ====================================================
// Purpose: Compute Legendre functions Qn(x) & Qn'(x)
// Input :  x  --- Argument of Qn(x)
//          n  --- Degree of Qn(x)  ( n = 0,1,2,…)
// Output:  QN(n) --- Qn(x)
//          QD(n) --- Qn'(x)
// ====================================================

template <typename T, typename OutputVec1, typename OutputVec2>
void lqn(T x, OutputVec1 qn, OutputVec2 qd) {
    // 获取输出向量的长度
    int n = qn.size() - 1;

    T x2, q0, q1, qf, qc1, qc2, qr, qf0, qf1, qf2;
    const T eps = 1.0e-14;

    // 如果 x 的绝对值为 1.0，则将所有元素设置为大数，并返回
    if (fabs(x) == 1.0) {
        for (int k = 0; k <= n; k++) {
            qn[k] = 1.0e300;
            qd[k] = 1.0e300;
        }
        return;
    }

    // 如果 x 小于等于 1.021，则执行以下操作
    if (x <= 1.021) {
        // 计算 q0 和 q1 的初值
        x2 = fabs((1.0 + x) / (1.0 - x));
        q0 = 0.5 * log(x2);
        q1 = x * q0 - 1.0;
        qn[0] = q0;
        qn[1] = q1;
        qd[0] = 1.0 / (1.0 - x * x);
        qd[1] = qn[0] + x * qd[0];

        // 计算 Qn(x) 和 Qn'(x) 的值
        for (int k = 2; k <= n; k++) {
            qf = ((2.0 * k - 1.0) * x * q1 - (k - 1.0) * q0) / k;
            qn[k] = qf;
            qd[k] = (qn[k - 1] - x * qf) * k / (1.0 - x * x);
            q0 = q1;
            q1 = qf;
        }
    } else {
        // 如果 x 大于 1.021，则执行以下操作
        qc1 = 0.0;
        qc2 = 1.0 / x;

        // 计算 qc2 的值
        for (int j = 1; j <= n; j++) {
            qc2 *= j / ((2.0 * j + 1.0) * x);
            if (j == n - 1)
                qc1 = qc2;
        }

        // 计算 Qn(x) 和 Qn'(x) 的值
        for (int l = 0; l <= 1; l++) {
            int nl = n + l;
            qf = 1.0;
            qr = 1.0;

            // 使用级数展开计算 Qn(x) 的近似值
            for (int k = 1; k <= 500; k++) {
                qr = qr * (0.5 * nl + k - 1.0) * (0.5 * (nl - 1) + k) / ((nl + k - 0.5) * k * x * x);
                qf += qr;
                if (fabs(qr / qf) < eps)
                    break;
            }

            if (l == 0) {
                qn[n - 1] = qf * qc1;
            } else {
                qn[n] = qf * qc2;
            }
        }

        qf2 = qn[n];
        qf1 = qn[n - 1];

        // 逆向计算 Qn(x) 的值
        for (int k = n; k >= 2; k--) {
            qf0 = ((2 * k - 1.0) * x * qf1 - k * qf2) / (k - 1.0);
            qn[k - 2] = qf0;
            qf2 = qf1;
            qf1 = qf0;
        }

        // 计算 Qn'(x) 的值
        qd[0] = 1.0 / (1.0 - x * x);

        for (int k = 1; k <= n; k++) {
            qd[k] = k * (qn[k - 1] - x * qn[k]) / (1.0 - x * x);
        }
    }
}
}

// ==================================================
// Purpose: Compute the Legendre functions Qn(z) and
//          their derivatives Qn'(z) for a complex
//          argument
// Input :  x --- Real part of z
//          y --- Imaginary part of z
//          n --- Degree of Qn(z), n = 0,1,2,...
// Output:  CQN(n) --- Qn(z)
//          CQD(n) --- Qn'(z)
// ==================================================

template <typename T, typename OutputVec1, typename OutputVec2>
void lqn(std::complex<T> z, OutputVec1 cqn, OutputVec2 cqd) {
    int n = cqn.size() - 1;  // Determine the degree of the Legendre functions

    std::complex<T> cq0, cq1, cqf0 = 0.0, cqf1, cqf2;  // Define complex variables for computations

    if (std::real(z) == 1) {  // Handle the case where the real part of z is exactly 1
        for (int k = 0; k <= n; ++k) {
            cqn(k) = 1e300;  // Set CQN(n) to a large number
            cqd(k) = 1e300;  // Set CQD(n) to a large number
        }
        return;  // Exit function early
    }
    int ls = ((std::abs(z) > 1.0) ? -1 : 1);  // Determine ls based on the magnitude of z

    // Compute initial values cq0 and cq1
    cq0 = std::log(static_cast<T>(ls) * (static_cast<T>(1) + z) / (static_cast<T>(1) - z)) / static_cast<T>(2);
    cq1 = z * cq0 - static_cast<T>(1);

    cqn(0) = cq0;  // Store Q0(z) in CQN(0)
    cqn(1) = cq1;  // Store Q1(z) in CQN(1)

    if (std::abs(z) < 1.0001) {  // Case where |z| is less than a threshold
        cqf0 = cq0;
        cqf1 = cq1;
        // Compute Legendre functions Q2(z) to Qn(z)
        for (int k = 2; k <= n; k++) {
            cqf2 = (static_cast<T>(2 * k - 1) * z * cqf1 - static_cast<T>(k - 1) * cqf0) / static_cast<T>(k);
            cqn(k) = cqf2;
            cqf0 = cqf1;
            cqf1 = cqf2;
        }
    } else {  // Case where |z| is greater than or equal to the threshold
        int km;
        if (std::abs(z) > 1.1) {
            km = 40 + n;
        } else {
            km = (int) ((40 + n) * floor(-1.0 - 1.8 * log(std::abs(z - static_cast<T>(1)))));
        }

        cqf2 = 0.0;
        cqf1 = 1.0;
        // Compute Legendre functions Q0(z) to Qn(z) using a different method
        for (int k = km; k >= 0; k--) {
            cqf0 = (static_cast<T>(2 * k + 3) * z * cqf1 - static_cast<T>(k + 2) * cqf2) / static_cast<T>(k + 1);
            if (k <= n) {
                cqn[k] = cqf0;
            }
            cqf2 = cqf1;
            cqf1 = cqf0;
        }
        // Normalize the results
        for (int k = 0; k <= n; ++k) {
            cqn[k] *= cq0 / cqf0;
        }
    }

    // Compute derivatives Qn'(z)
    cqd(0) = (cqn(1) - z * cqn(0)) / (z * z - static_cast<T>(1));

    for (int k = 1; k <= n; ++k) {
        cqd(k) = (static_cast<T>(k) * z * cqn(k) - static_cast<T>(k) * cqn(k - 1)) / (z * z - static_cast<T>(1));
    }
}

// ==========================================================
// Purpose: Compute the associated Legendre functions of the
//          second kind, Qmn(x) and Qmn'(x)
// Input :  x  --- Argument of Qmn(x)
//          m  --- Order of Qmn(x)  ( m = 0,1,2,… )
//          n  --- Degree of Qmn(x) ( n = 0,1,2,… )
//          mm --- Physical dimension of QM and QD
// Output:  QM(m,n) --- Qmn(x)
//          QD(m,n) --- Qmn'(x)
// ==========================================================

template <typename T, typename OutputMat1, typename OutputMat2>
void lqmn(T x, OutputMat1 qm, OutputMat2 qd) {
    int m = qm.extent(0) - 1;  // Determine the order of the Legendre functions
    int n = qm.extent(1) - 1;  // Determine the degree of the Legendre functions

    double q0, q1, q10, qf, qf0, qf1, qf2, xs, xq;
    int i, j, k, km, ls;
    # 如果 x 的绝对值为 1.0
    if (fabs(x) == 1.0) {
        # 初始化 qm 和 qd 数组的值为 1e300
        for (i = 0; i < (m + 1); i++) {
            for (j = 0; j < (n + 1); j++) {
                qm(i, j) = 1e300;
                qd(i, j) = 1e300;
            }
        }
        # 直接返回，不再执行后续代码
        return;
    }
    
    # ls 初始值为 1
    ls = 1;
    # 如果 x 的绝对值大于 1.0，则 ls 变为 -1
    if (fabs(x) > 1.0) {
        ls = -1;
    }
    
    # 计算 xs 和 xq
    xs = ls * (1.0 - x * x);
    xq = sqrt(xs);
    
    # 计算 q0
    q0 = 0.5 * log(fabs((x + 1.0) / (x - 1.0)));
    
    # 如果 x 的绝对值小于 1.0001
    if (fabs(x) < 1.0001) {
        # 初始化 qm 数组的前两行
        qm(0, 0) = q0;
        qm(0, 1) = x * q0 - 1.0;
        qm(1, 0) = -1.0 / xq;
        qm(1, 1) = -ls * xq * (q0 + x / (1. - x * x));
        
        # 计算 qm 的第一行后续元素
        for (i = 0; i <= 1; i++) {
            for (j = 2; j <= n; j++) {
                qm(i, j) = ((2.0 * j - 1.) * x * qm(i, j - 1) - (j + i - 1) * qm(i, j - 2)) / (j - i);
            }
        }
        
        # 计算 qm 的后续行和列
        for (i = 2; i <= m; i++) {
            for (j = 0; j <= n; j++) {
                qm(i, j) = -2.0 * (i - 1.0) * x / xq * qm(i - 1, j) - ls * (j + i - 1.0) * (j - i + 2.0) * qm(i - 2, j);
            }
        }
    } else {
        # 如果 x 的绝对值大于 1.1
        if (fabs(x) > 1.1) {
            # 设置 km 为 40 + m + n
            km = 40 + m + n;
        } else {
            # 根据 x 计算 km
            km = (40 + m + n) * ((int) (-1. - 1.8 * log(x - 1.)));
        }
        
        # 初始化 qf0, qf1, qf2
        qf2 = 0.0;
        qf1 = 1.0;
        qf0 = 0.0;
        
        # 计算 qf0 的系数
        for (k = km; k >= 0; k--) {
            qf0 = ((2.0 * k + 3.0) * x * qf1 - (k + 2.0) * qf2) / (k + 1.0);
            # 如果 k 小于等于 n，则将 qf0 赋给 qm 的第一行对应位置
            if (k <= n) {
                qm(0, k) = qf0;
            }
            qf2 = qf1;
            qf1 = qf0;
        }
        
        # 根据 qf0 更新 qm 的第一行
        for (k = 0; k <= n; k++) {
            qm(0, k) *= q0 / qf0;
        }
        
        # 重新初始化 qf0, qf1
        qf2 = 0.0;
        qf1 = 1.0;
        
        # 计算 qf0 的系数
        for (k = km; k >= 0; k--) {
            qf0 = ((2.0 * k + 3.0) * x * qf1 - (k + 1.0) * qf2) / (k + 2.0);
            # 如果 k 小于等于 n，则将 qf0 赋给 qm 的第二行对应位置
            if (k <= n) {
                qm(1, k) = qf0;
            }
            qf2 = qf1;
            qf1 = qf0;
        }
        
        # 计算 q10
        q10 = -1.0 / xq;
        # 根据 qf0 更新 qm 的第二行
        for (k = 0; k <= n; k++) {
            qm(1, k) *= q10 / qf0;
        }
        
        # 更新 qm 的后续行和列
        for (j = 0; j <= n; j++) {
            q0 = qm(0, j);
            q1 = qm(1, j);
            for (i = 0; i <= (m - 2); i++) {
                qf = -2. * (i + 1.) * x / xq * q1 + (j - i) * (j + i + 1.) * q0;
                qm(i + 2, j) = qf;
                q0 = q1;
                q1 = qf;
            }
        }
    }
    
    # 计算 qd 的第一行
    qd(0, 0) = ls / xs;
    for (j = 1; j <= n; j++) {
        qd(0, j) = ls * j * (qm(0, j - 1) - x * qm(0, j)) / xs;
    }
    
    # 计算 qd 的后续行和列
    for (i = 1; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            qd(i, j) = ls * i * x / xs * qm(i, j) + (i + j) * (j - i + 1.) / xq * qm(i - 1, j);
        }
    }
// }
// 上面的注释表示函数 lqmn 的结束位置

// =======================================================
// Purpose: 计算复参数 z 的关联Legendre函数的第二类 Qmn(z) 和 Qmn'(z)
//          其中，
//          x  --- z 的实部
//          y  --- z 的虚部
//          m  --- Qmn(z) 的阶数 ( m = 0,1,2,… )
//          n  --- Qmn(z) 的次数 ( n = 0,1,2,… )
//          mm --- CQM 和 CQD 的物理维度
// Output:  CQM(m,n) --- Qmn(z)
//          CQD(m,n) --- Qmn'(z)
// =======================================================

template <typename T, typename OutputMat1, typename OutputMat2>
void lqmn(std::complex<T> z, OutputMat1 cqm, OutputMat2 cqd) {
    int m = cqm.extent(0) - 1;  // 获取 CQM 的维度大小减1，即阶数 m
    int n = cqm.extent(1) - 1;  // 获取 CQM 的维度大小减1，即次数 n

    int i, j, k, km, ls;  // 定义循环和临时变量
    std::complex<T> cq0, cq1, cq10, cqf0 = 0, cqf, cqf1, cqf2, zq, zs;  // 定义复数变量

    // 检查 z 是否为实数且其绝对值为1，若是，则将所有 cqm 和 cqd 的元素设置为极大值 1e300，并返回
    if ((std::abs(std::real(z)) == 1) && (std::imag(z) == 0)) {
        for (i = 0; i < (m + 1); i++) {
            for (j = 0; j < (n + 1); j++) {
                cqm(i, j) = 1e300;
                cqd(i, j) = 1e300;
            }
        }

        return;
    }

    T xc = std::abs(z);  // 计算 z 的绝对值
    ls = 0;  // 初始化 ls 变量为0
    if ((std::imag(z) == 0) || (xc < 1)) {  // 如果 z 是实数或者其绝对值小于1，则 ls 设为1
        ls = 1;
    }
    if (xc > 1) {  // 如果 z 的绝对值大于1，则 ls 设为-1
        ls = -1;
    }
    zs = static_cast<T>(ls) * (static_cast<T>(1) - z * z);  // 计算 zs = ls * (1 - z^2)
    zq = std::sqrt(zs);  // 计算 zq = sqrt(zs)

    // 计算 cq0 = log(ls * (1 + z) / (1 - z)) / 2
    cq0 = std::log(static_cast<T>(ls) * (static_cast<T>(1) + z) / (static_cast<T>(1) - z)) / static_cast<T>(2);
    
    // 如果 xc 小于1.0001，则计算以下系数并存储到 cqm 中
    if (xc < 1.0001) {
        cqm(0, 0) = cq0;
        cqm(1, 0) = -static_cast<T>(1) / zq;
        cqm(0, 1) = z * cq0 - static_cast<T>(1);
        cqm(1, 1) = -zq * (cq0 + z / (static_cast<T>(1) - z * z));

        // 计算 cqm 的其他元素值，使用递推公式
        for (i = 0; i <= 1; i++) {
            for (j = 2; j <= n; j++) {
                cqm(i, j) =
                    (static_cast<T>(2 * j - 1) * z * cqm(i, j - 1) - static_cast<T>(j + i - 1) * cqm(i, j - 2)) /
                    static_cast<T>(j - i);
            }
        }

        // 计算剩余的 cqm 元素，同样使用递推公式
        for (i = 2; i <= m; i++) {
            for (j = 0; j <= n; j++) {
                cqm(i, j) = -2 * static_cast<T>(i - 1) * z / zq * cqm(i - 1, j) -
                            static_cast<T>(ls * (j + i - 1) * (j - i + 2)) * cqm(i - 2, j);
            }
        }

    }
}


注释完毕。
    } else {
        // 如果条件不成立，执行以下代码块

        if (xc > 1.1) {
            // 如果 xc 大于 1.1，计算 km 的值
            km = 40 + m + n;
        } else {
            // 如果 xc 不大于 1.1，计算 km 的值
            km = (40 + m + n) * ((int) (-1.0 - 1.8 * log(xc - 1.)));
        }

        // 初始化变量
        cqf2 = 0.0;
        cqf1 = 1.0;

        // 计算一维 Chebyshev 多项式系数 cqm(0, k)
        for (k = km; k >= 0; k--) {
            cqf0 = (static_cast<T>(2 * k + 3) * z * cqf1 - static_cast<T>(k + 2) * cqf2) / static_cast<T>(k + 1);
            if (k <= n) {
                cqm(0, k) = cqf0; // 将计算结果存入 cqm(0, k)
            }
            cqf2 = cqf1;
            cqf1 = cqf0;
        }

        // 对 cqm(0, k) 应用归一化系数 cq0 / cqf0
        for (k = 0; k <= n; k++) {
            cqm(0, k) *= cq0 / cqf0;
        }

        // 重新初始化变量
        cqf2 = 0.0;
        cqf1 = 1.0;

        // 计算二维 Chebyshev 多项式系数 cqm(1, k)
        for (k = km; k >= 0; k--) {
            cqf0 = (static_cast<T>(2 * k + 3) * z * cqf1 - static_cast<T>(k + 1) * cqf2) / static_cast<T>(k + 2);
            if (k <= n) {
                cqm(1, k) = cqf0; // 将计算结果存入 cqm(1, k)
            }
            cqf2 = cqf1;
            cqf1 = cqf0;
        }

        // 计算归一化系数 cq10 / cqf0，并应用到 cqm(1, k)
        cq10 = -static_cast<T>(1) / zq;
        for (k = 0; k <= n; k++) {
            cqm(1, k) *= cq10 / cqf0;
        }

        // 计算二维 Chebyshev 多项式系数 cqm(i, j) （i >= 2）
        for (j = 0; j <= n; j++) {
            cq0 = cqm(0, j);
            cq1 = cqm(1, j);
            for (i = 0; i <= (m - 2); i++) {
                // 计算 cqm(i+2, j) 的值
                cqf = -static_cast<T>(2 * (i + 1)) * z / zq * cq1 + static_cast<T>((j - i) * (j + i + 1)) * cq0;
                cqm(i + 2, j) = cqf; // 将计算结果存入 cqm(i+2, j)
                cq0 = cq1;
                cq1 = cqf;
            }
        }

        // 计算导数的第一行 cqd(0, j)
        cqd(0, 0) = static_cast<T>(ls) / zs;
        for (j = 1; j <= n; j++) {
            cqd(0, j) = ls * static_cast<T>(j) * (cqm(0, j - 1) - z * cqm(0, j)) / zs; // 计算 cqd(0, j)
        }

        // 计算导数的其余部分 cqd(i, j) （i >= 1）
        for (i = 1; i <= m; i++) {
            for (j = 0; j <= n; j++) {
                cqd(i, j) = static_cast<T>(ls * i) * z / zs * cqm(i, j) +
                            static_cast<T>((i + j) * (j - i + 1)) / zq * cqm(i - 1, j); // 计算 cqd(i, j)
            }
        }
    }
}

} // namespace special
```