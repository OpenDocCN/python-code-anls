# `D:\src\scipysrc\scipy\scipy\special\special_wrappers.h`

```
/*
 * This file is a collection of wrappers around the
 * Special Function Fortran library of functions
 * to be compiled with the other special functions in cephes
 *
 * Functions written by Shanjie Zhang and Jianming Jin.
 * Interface by
 * Travis E. Oliphant
 */

#pragma once

#include "Python.h"                     // 包含 Python 头文件
#include "npy_2_complexcompat.h"        // 包含复数兼容性头文件
#include "sf_error.h"                   // 包含特殊函数错误处理头文件
#include <math.h>                       // 包含数学函数头文件
#include <numpy/npy_math.h>             // 包含 NumPy 数学头文件

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

npy_cdouble clngamma_wrap(npy_cdouble z);               // 声明 Gamma 函数的复数版本包装函数
npy_cdouble chyp2f1_wrap(double a, double b, double c, npy_cdouble z);    // 声明超几何函数 2F1 的复数版本包装函数
npy_cdouble chyp1f1_wrap(double a, double b, npy_cdouble z);              // 声明超几何函数 1F1 的复数版本包装函数
double hyp1f1_wrap(double a, double b, double x);       // 声明超几何函数 1F1 的包装函数
double hypU_wrap(double a, double b, double x);         // 声明超几何函数 U 的包装函数
npy_cdouble cerf_wrap(npy_cdouble z);                   // 声明误差函数的复数版本包装函数

double special_exp1(double x);                          // 声明特殊指数函数
npy_cdouble special_cexp1(npy_cdouble z);               // 声明特殊复指数函数

double special_expi(double x);                          // 声明指数积分函数
npy_cdouble special_cexpi(npy_cdouble z);               // 声明复指数积分函数

double struve_wrap(double v, double x);                 // 声明斯特鲁夫函数的包装函数
double special_itstruve0(double x);                     // 声明修正斯特鲁夫函数的零阶
double special_it2struve0(double x);                    // 声明第二种类型修正斯特鲁夫函数的零阶

double modstruve_wrap(double v, double x);              // 声明修正斯特鲁夫函数的包装函数
double special_itmodstruve0(double x);                  // 声明修正斯特鲁夫函数的零阶

double special_ber(double x);                           // 声明贝塞尔函数第一类
double special_bei(double x);                           // 声明贝塞尔函数第二类
double special_ker(double x);                           // 声明修正贝塞尔函数第一类
double special_kei(double x);                           // 声明修正贝塞尔函数第二类
double special_berp(double x);                          // 声明贝塞尔函数第一类的导数
double special_beip(double x);                          // 声明贝塞尔函数第二类的导数
double special_kerp(double x);                          // 声明修正贝塞尔函数第一类的导数
double special_keip(double x);                          // 声明修正贝塞尔函数第二类的导数

void special_ckelvin(double x, npy_cdouble *Be, npy_cdouble *Ke, npy_cdouble *Bep, npy_cdouble *Kep);  // 声明开尔文函数

void it1j0y0_wrap(double x, double *, double *);        // 声明第一类、零阶贝塞尔函数和零阶贝塞尔函数导数的包装函数
void it2j0y0_wrap(double x, double *, double *);        // 声明第二类、零阶贝塞尔函数和零阶贝塞尔函数导数的包装函数
void it1i0k0_wrap(double x, double *, double *);        // 声明第一类、零阶修正贝塞尔函数和零阶修正贝塞尔函数导数的包装函数
void it2i0k0_wrap(double x, double *, double *);        // 声明第二类、零阶修正贝塞尔函数和零阶修正贝塞尔函数导数的包装函数

int cfresnl_wrap(npy_cdouble x, npy_cdouble *sf, npy_cdouble *cf);  // 声明复解非线性特殊函数的包装函数
double cem_cva_wrap(double m, double q);                // 声明指数积分类型函数
double sem_cva_wrap(double m, double q);                // 声明指数积分类型函数
void cem_wrap(double m, double q, double x, double *csf, double *csd);   // 声明指数积分类型函数
void sem_wrap(double m, double q, double x, double *csf, double *csd);   // 声明指数积分类型函数
void mcm1_wrap(double m, double q, double x, double *f1r, double *d1r);  // 声明 Jacobi 椭圆函数类型函数
void msm1_wrap(double m, double q, double x, double *f1r, double *d1r);  // 声明 Jacobi 椭圆函数类型函数
void mcm2_wrap(double m, double q, double x, double *f2r, double *d2r);  // 声明 Jacobi 椭圆函数类型函数
void msm2_wrap(double m, double q, double x, double *f2r, double *d2r);  // 声明 Jacobi 椭圆函数类型函数
double pmv_wrap(double, double, double);                // 声明概率函数
void pbwa_wrap(double, double, double *, double *);      // 声明概率函数
void pbdv_wrap(double, double, double *, double *);      // 声明概率函数
void pbvv_wrap(double, double, double *, double *);      // 声明概率函数

void prolate_aswfa_wrap(double, double, double, double, double, double *, double *);   // 声明 prolate 椭球体表面波类型函数
void prolate_radial1_wrap(double, double, double, double, double, double *, double *); // 声明 prolate 椭球体径向波类型函数
void prolate_radial2_wrap(double, double, double, double, double, double *, double *); // 声明 prolate 椭球体径向波类型函数
void oblate_aswfa_wrap(double, double, double, double, double, double *, double *);    // 声明 oblate 椭球体表面波类型函数
void oblate_radial1_wrap(double, double, double, double, double, double *, double *);  // 声明 oblate 椭球体径向波类型函数
void oblate_radial2_wrap(double, double, double, double, double, double *, double *);  // 声明 oblate 椭球体径向波类型函数

#ifdef __cplusplus
}
#endif /* __cplusplus */
*/
// 声明 prolate_aswfa_nocv_wrap 函数，返回类型为 double，接受五个参数和一个 double 指针参数
double prolate_aswfa_nocv_wrap(double, double, double, double, double *);

// 声明 prolate_radial1_nocv_wrap 函数，返回类型为 double，接受五个参数和一个 double 指针参数
double prolate_radial1_nocv_wrap(double, double, double, double, double *);

// 声明 prolate_radial2_nocv_wrap 函数，返回类型为 double，接受五个参数和一个 double 指针参数
double prolate_radial2_nocv_wrap(double, double, double, double, double *);

// 声明 oblate_aswfa_nocv_wrap 函数，返回类型为 double，接受五个参数和一个 double 指针参数
double oblate_aswfa_nocv_wrap(double, double, double, double, double *);

// 声明 oblate_radial1_nocv_wrap 函数，返回类型为 double，接受五个参数和一个 double 指针参数
double oblate_radial1_nocv_wrap(double, double, double, double, double *);

// 声明 oblate_radial2_nocv_wrap 函数，返回类型为 double，接受五个参数和一个 double 指针参数
double oblate_radial2_nocv_wrap(double, double, double, double, double *);

// 声明 prolate_segv_wrap 函数，返回类型为 double，接受三个 double 类型参数
double prolate_segv_wrap(double, double, double);

// 声明 oblate_segv_wrap 函数，返回类型为 double，接受三个 double 类型参数
double oblate_segv_wrap(double, double, double);

// 声明 modified_fresnel_plus_wrap 函数，返回类型为 void，接受一个 double 参数和两个复数结构体指针参数
void modified_fresnel_plus_wrap(double x, npy_cdouble *F, npy_cdouble *K);

// 声明 modified_fresnel_minus_wrap 函数，返回类型为 void，接受一个 double 参数和两个复数结构体指针参数
void modified_fresnel_minus_wrap(double x, npy_cdouble *F, npy_cdouble *K);

// 声明 special_airy 函数，返回类型为 void，接受一个 double 参数和四个 double 指针参数
void special_airy(double x, double *ai, double *aip, double *bi, double *bip);

// 声明 special_cairy 函数，返回类型为 void，接受一个复数结构体参数和四个复数结构体指针参数
void special_cairy(npy_cdouble z, npy_cdouble *ai, npy_cdouble *aip, npy_cdouble *bi, npy_cdouble *bip);

// 声明 special_airye 函数，返回类型为 void，接受一个 double 参数和四个 double 指针参数
void special_airye(double z, double *ai, double *aip, double *bi, double *bip);

// 声明 special_cairye 函数，返回类型为 void，接受一个复数结构体参数和四个复数结构体指针参数
void special_cairye(npy_cdouble z, npy_cdouble *ai, npy_cdouble *aip, npy_cdouble *bi, npy_cdouble *bip);

// 声明 special_itairy 函数，返回类型为 void，接受一个 double 参数和四个 double 指针参数
void special_itairy(double x, double *apt, double *bpt, double *ant, double *bnt);

// 声明 special_cyl_bessel_j 函数，返回类型为 double，接受两个 double 类型参数
double special_cyl_bessel_j(double v, double z);

// 声明 special_ccyl_bessel_j 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_bessel_j(double v, npy_cdouble z);

// 声明 special_cyl_bessel_je 函数，返回类型为 double，接受两个 double 类型参数
double special_cyl_bessel_je(double v, double z);

// 声明 special_ccyl_bessel_je 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_bessel_je(double v, npy_cdouble z);

// 声明 special_cyl_bessel_y 函数，返回类型为 double，接受两个 double 类型参数
double special_cyl_bessel_y(double v, double x);

// 声明 special_ccyl_bessel_y 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_bessel_y(double v, npy_cdouble z);

// 声明 special_cyl_bessel_ye 函数，返回类型为 double，接受两个 double 类型参数
double special_cyl_bessel_ye(double v, double z);

// 声明 special_ccyl_bessel_ye 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_bessel_ye(double v, npy_cdouble z);

// 声明 special_cyl_bessel_i 函数，返回类型为 double，接受两个 double 类型参数
double special_cyl_bessel_i(double v, double z);

// 声明 special_ccyl_bessel_i 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_bessel_i(double v, npy_cdouble z);

// 声明 special_cyl_bessel_ie 函数，返回类型为 double，接受两个 double 类型参数
double special_cyl_bessel_ie(double v, double z);

// 声明 special_ccyl_bessel_ie 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_bessel_ie(double v, npy_cdouble z);

// 声明 special_cyl_bessel_k_int 函数，返回类型为 double，接受一个 int 和一个 double 类型参数
double special_cyl_bessel_k_int(int n, double z);

// 声明 special_cyl_bessel_k 函数，返回类型为 double，接受两个 double 类型参数
double special_cyl_bessel_k(double v, double z);

// 声明 special_ccyl_bessel_k 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_bessel_k(double v, npy_cdouble z);

// 声明 special_cyl_bessel_ke 函数，返回类型为 double，接受两个 double 类型参数
double special_cyl_bessel_ke(double v, double z);

// 声明 special_ccyl_bessel_ke 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_bessel_ke(double v, npy_cdouble z);

// 声明 special_ccyl_hankel_1 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_hankel_1(double v, npy_cdouble z);

// 声明 special_ccyl_hankel_2 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_hankel_2(double v, npy_cdouble z);

// 声明 special_ccyl_hankel_1e 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_hankel_1e(double v, npy_cdouble z);

// 声明 special_ccyl_hankel_2e 函数，返回类型为复数结构体，接受两个 double 类型参数
npy_cdouble special_ccyl_hankel_2e(double v, npy_cdouble z);

// 声明 hyp2f1_complex_wrap 函数，返回类型为复数结构体，接受三个 double 类型参数和一个复数结构体参数
npy_cdouble hyp2f1_complex_wrap(double a, double b, double c, npy_cdouble zp);

// 声明 sin_pi 函数，返回类型为 double，接受一个 double 参数
double sin_pi(double x);

// 声明 gammaln_wrap 函数，返回类型为 double，接受一个 double 参数
double gammaln_wrap
// 声明特殊函数 special_log_expitl，参数为 npy_longdouble 类型，返回值为 npy_longdouble 类型
npy_longdouble special_log_expitl(npy_longdouble x);

// 声明特殊函数 special_logitf，参数为 float 类型，返回值为 double 类型
double special_logitf(float x);

// 声明特殊函数 special_logit，参数为 double 类型，返回值为 double 类型
double special_logit(double x);

// 声明特殊函数 special_logitl，参数为 npy_longdouble 类型，返回值为 npy_longdouble 类型
npy_longdouble special_logitl(npy_longdouble x);

// 声明特殊函数 special_loggamma，参数为 double 类型，返回值为 double 类型
double special_loggamma(double x);

// 声明特殊函数 special_cloggamma，参数为 npy_cdouble 类型，返回值为 npy_cdouble 类型
npy_cdouble special_cloggamma(npy_cdouble z);

// 声明特殊函数 special_gamma，参数为 double 类型，返回值为 double 类型
double special_gamma(double x);

// 声明特殊函数 special_cgamma，参数为 npy_cdouble 类型，返回值为 npy_cdouble 类型
npy_cdouble special_cgamma(npy_cdouble z);

// 声明特殊函数 special_hyp2f1，参数为四个 double 类型，返回值为 double 类型
double special_hyp2f1(double a, double b, double c, double z);

// 声明特殊函数 special_chyp2f1，参数为两个 double 和一个 npy_cdouble 类型，返回值为 npy_cdouble 类型
npy_cdouble special_chyp2f1(double a, double b, double c, npy_cdouble z);

// 声明特殊函数 special_lambertw，参数为一个 npy_cdouble 和两个其他类型，返回值为 npy_cdouble 类型
npy_cdouble special_lambertw(npy_cdouble z, long k, double tol);

// 声明特殊函数 special_rgamma，参数为 double 类型，返回值为 double 类型
double special_rgamma(double x);

// 声明特殊函数 special_crgamma，参数为 npy_cdouble 类型，返回值为 npy_cdouble 类型
npy_cdouble special_crgamma(npy_cdouble z);

// 声明特殊函数 special_sph_harm，参数为两个 long 和两个 double 类型，返回值为 npy_cdouble 类型
npy_cdouble special_sph_harm(long m, long n, double theta, double phi);

// 声明特殊函数 special_sph_harm_unsafe，参数为四个 double 类型，返回值为 npy_cdouble 类型
npy_cdouble special_sph_harm_unsafe(double m, double n, double theta, double phi);

// 声明特殊函数 special_ellipk，参数为 double 类型，返回值为 double 类型
double special_ellipk(double m);

// 声明特殊函数 binom_wrap，参数为两个 double 类型，返回值为 double 类型
double binom_wrap(double n, double k);

// 声明特殊函数 hyp2f1_complex_wrap，参数为三个 double 和一个 npy_cdouble 类型，返回值为 npy_cdouble 类型
npy_cdouble hyp2f1_complex_wrap(double a, double b, double c, npy_cdouble zp);

// 声明特殊函数 cephes_hyp2f1_wrap，参数为三个 double 和一个 double 类型，返回值为 double 类型
double cephes_hyp2f1_wrap(double a, double b, double c, double x);

// 声明特殊函数 cephes_airy_wrap，参数为 double 和四个 double* 类型，返回值为 double 类型
double cephes_airy_wrap(double x, double *ai, double *aip, double *bi, double *bip);

// 声明特殊函数 cephes_beta_wrap，参数为两个 double 类型，返回值为 double 类型
double cephes_beta_wrap(double a, double b);

// 声明特殊函数 cephes_lbeta_wrap，参数为两个 double 类型，返回值为 double 类型
double cephes_lbeta_wrap(double a, double b);

// 声明特殊函数 cephes_bdtr_wrap，参数为 double、int 和 double 类型，返回值为 double 类型
double cephes_bdtr_wrap(double k, int n, double p);

// 声明特殊函数 cephes_bdtri_wrap，参数为 double、int 和 double 类型，返回值为 double 类型
double cephes_bdtri_wrap(double k, int n, double y);

// 声明特殊函数 cephes_bdtrc_wrap，参数为 double、int 和 double 类型，返回值为 double 类型
double cephes_bdtrc_wrap(double k, int n, double p);

// 声明特殊函数 cephes_cosm1_wrap，参数为 double 类型，返回值为 double 类型
double cephes_cosm1_wrap(double x);

// 声明特殊函数 cephes_expm1_wrap，参数为 double 类型，返回值为 double 类型
double cephes_expm1_wrap(double x);

// 声明特殊函数 cephes_expn_wrap，参数为 int 和 double 类型，返回值为 double 类型
double cephes_expn_wrap(int n, double x);

// 声明特殊函数 cephes_log1p_wrap，参数为 double 类型，返回值为 double 类型
double cephes_log1p_wrap(double x);

// 声明特殊函数 cephes_gamma_wrap，参数为 double 类型，返回值为 double 类型
double cephes_gamma_wrap(double x);

// 声明特殊函数 cephes_gammasgn_wrap，参数为 double 类型，返回值为 double 类型
double cephes_gammasgn_wrap(double x);

// 声明特殊函数 cephes_lgam_wrap，参数为 double 类型，返回值为 double 类型
double cephes_lgam_wrap(double x);

// 声明特殊函数 cephes_iv_wrap，参数为两个 double 类型，返回值为 double 类型
double cephes_iv_wrap(double v, double x);

// 声明特殊函数 cephes_jv_wrap，参数为两个 double 类型，返回值为 double 类型
double cephes_jv_wrap(double v, double x);

// 声明特殊函数 cephes_ellpk_wrap，参数为 double 类型，返回值为 double 类型
double cephes_ellpk_wrap(double x);

// 声明特殊函数 cephes_ellpj_wrap，参数为 double、double 和四个 double* 类型，返回值为 int 类型
int cephes_ellpj_wrap(double u, double m, double *sn, double *cn, double *dn, double *ph);

// 声明特殊函数 cephes_fresnl_wrap，参数为 double 和两个 double* 类型，返回值为 int 类型
int cephes_fresnl_wrap(double xxa, double *ssa, double *cca);

// 声明特殊函数 cephes_nbdtr_wrap，参数为 int、int 和 double 类型，返回值为 double 类型
double cephes_nbdtr_wrap(int k, int n, double p);

// 声明特殊函数 cephes_nbdtrc_wrap，参数为 int、int 和 double 类型，返回值为 double 类型
double cephes_nbdtrc_wrap(int k, int n, double p);

// 声明特殊函数 cephes_nbdtri_wrap，参数为 int、int 和 double 类型，返回值为 double 类型
double cephes_nbdtri_wrap(int k, int n, double p);

// 声明特殊函数 cephes_ndtr_wrap，参数为 double 类型，返回值为 double 类型
double cephes_ndtr_wrap(double x);

// 声明特殊函数 cephes_ndtri_wrap，参数为 double 类型，返回值为 double 类型
double cephes_ndtri_wrap(double x);

// 声明特殊函数 cephes_pdtri_wrap，参数为 int 和 double 类型，返回值为 double 类型
double cephes_pdtri_wrap(int k, double y);

// 声明特殊函数 cephes_poch_wrap，参数为两个 double 类型，返回值为 double 类型
double cephes_poch_wrap(double x, double m);

// 声明特殊函数 cephes_sici_wrap，参数为 double 和两个 double* 类型，返回值为 int 类型
int cephes_sici_wrap(double x, double *si, double *ci);

//
// 计算特殊对数权贝塞尔函数
double special_log_wright_bessel(double a, double b, double x);

// 计算特殊缩放的指数函数
double special_scaled_exp1(double x);

// 计算切比雪夫贝塞尔多项式函数
double cephes_besselpoly(double a, double lambda, double nu);

// 计算特殊球贝塞尔函数 J_n(x)
double special_sph_bessel_j(long n, double x);
// 计算复数特殊球贝塞尔函数 J_n(z)
npy_cdouble special_csph_bessel_j(long n, npy_cdouble z);

// 计算特殊球贝塞尔函数 J_n(x) 的导数
double special_sph_bessel_j_jac(long n, double x);
// 计算复数特殊球贝塞尔函数 J_n(z) 的导数
npy_cdouble special_csph_bessel_j_jac(long n, npy_cdouble z);

// 计算特殊球贝塞尔函数 Y_n(x)
double special_sph_bessel_y(long n, double x);
// 计算复数特殊球贝塞尔函数 Y_n(z)
npy_cdouble special_csph_bessel_y(long n, npy_cdouble z);

// 计算特殊球贝塞尔函数 Y_n(x) 的导数
double special_sph_bessel_y_jac(long n, double x);
// 计算复数特殊球贝塞尔函数 Y_n(z) 的导数
npy_cdouble special_csph_bessel_y_jac(long n, npy_cdouble z);

// 计算特殊修正球贝塞尔函数 I_n(x)
double special_sph_bessel_i(long n, double x);
// 计算复数特殊修正球贝塞尔函数 I_n(z)
npy_cdouble special_csph_bessel_i(long n, npy_cdouble z);

// 计算特殊修正球贝塞尔函数 I_n(x) 的导数
double special_sph_bessel_i_jac(long n, double x);
// 计算复数特殊修正球贝塞尔函数 I_n(z) 的导数
npy_cdouble special_csph_bessel_i_jac(long n, npy_cdouble z);

// 计算特殊调和球贝塞尔函数 K_n(x)
double special_sph_bessel_k(long n, double x);
// 计算复数特殊调和球贝塞尔函数 K_n(z)
npy_cdouble special_csph_bessel_k(long n, npy_cdouble z);

// 计算特殊调和球贝塞尔函数 K_n(x) 的导数
double special_sph_bessel_k_jac(long n, double x);
// 计算复数特殊调和球贝塞尔函数 K_n(z) 的导数
npy_cdouble special_csph_bessel_k_jac(long n, npy_cdouble z);

// 计算切比雪夫贝塞尔函数 Beta(a, b)
double cephes_beta(double a, double b);

// 计算 Chi 分布累积分布函数
double cephes_chdtr(double df, double x);

// 计算 Chi 分布补充累积分布函数
double cephes_chdtrc(double df, double x);

// 计算 Chi 分布逆累积分布函数
double cephes_chdtri(double df, double y);

// 计算对数 Beta 函数
double cephes_lbeta(double a, double b);

// 计算正弦函数乘 Pi
double cephes_sinpi(double x);

// 计算余弦函数乘 Pi
double cephes_cospi(double x);

// 计算立方根函数
double cephes_cbrt(double x);

// 计算伽玛函数
double cephes_Gamma(double x);

// 伽玛函数符号函数
double cephes_gammasgn(double x);

// 超几何函数 2F1
double cephes_hyp2f1(double a, double b, double c, double x);

// 计算修正 Bessel 函数 I_0(x)
double cephes_i0(double x);

// 计算修正 Bessel 函数 I_0e(x)
double cephes_i0e(double x);

// 计算修正 Bessel 函数 I_1(x)
double cephes_i1(double x);

// 计算修正 Bessel 函数 I_1e(x)
double cephes_i1e(double x);

// 计算修正 Bessel 函数 Iv(v, x)
double cephes_iv(double v, double x);

// 计算 Bessel 函数 J_0(x)
double cephes_j0(double x);

// 计算 Bessel 函数 J_1(x)
double cephes_j1(double x);

// 计算 Bessel 函数 K_0(x)
double cephes_k0(double x);

// 计算修正 Bessel 函数 K_0e(x)
double cephes_k0e(double x);

// 计算 Bessel 函数 K_1(x)
double cephes_k1(double x);

// 计算修正 Bessel 函数 K_1e(x)
double cephes_k1e(double x);

// 计算 Bessel 函数 Y_0(x)
double cephes_y0(double x);

// 计算 Bessel 函数 Y_1(x)
double cephes_y1(double x);

// 计算修正 Bessel 函数 Y_n(x)
double cephes_yn(int n, double x);

// 计算不完全伽玛函数 P(a, x)
double cephes_igam(double a, double x);

// 计算不完全伽玛函数 Q(a, x)
double cephes_igamc(double a, double x);

// 计算不完全伽玛函数 P(a, p)
double cephes_igami(double a, double p);

// 计算不完全伽玛函数 Q(a, p)
double cephes_igamci(double a, double p);

// 计算伽玛函数的系数
double cephes_igam_fac(double a, double x);

// 计算兰布达函数求和项 e^(-x) * g(x)
double cephes_lanczos_sum_expg_scaled(double x);

// 计算 Kolmogorov 分布函数
double cephes_kolmogorov(double x);

// 计算 Kolmogorov 分布函数的补充
double cephes_kolmogc(double x);

// 计算 Kolmogorov 分布函数的逆
double cephes_kolmogi(double x);

// 计算 Kolmogorov 分布函数的逆的补充
double cephes_kolmogci(double x);

// 计算 Kolmogorov 分布函数的参数 p
double cephes_kolmogp(double x);

// 计算 Smirnov 分布函数
double cephes_smirnov(int n, double x);

// 计算 Smirnov 分布函数的补充
double cephes_smirnovc(int n, double x);

// 计算 Smirnov 分布函数的逆
double cephes_smirnovi(int n, double x);

// 计算 Smirnov 分布函数的逆的补充
double cephes_smirnovci(int n, double x);

// 计算 Smirnov 分布函数的参数 p
double cephes_smirnovp(int n, double x);

// 计算标准正态分布的累积分布函数
double cephes_ndtr(double x);

// 计算误差函数 erf(x)
double cephes_erf(double x);

// 计算余误差函数 erfc(x)
double cephes_erfc(double x);

// 计算 Pochhammer 符号 (x)_m
double cephes_poch(double x, double m);

// 计算逆伽玛函数
double cephes_rgamma(double x);

// 计算 Riemann Zeta 函数
double cephes_zeta(double x, double q);

// 计算 Riemann Zeta 函数的补充
double cephes_zetac(double x);

// 计算 log(1 + x)
double cephes_log1p(double x);

// 计算 log(1 + x) - x
double cephes_log1pmx(double x);

// 计算 lgamma(1 + x)
double cephes_lgam1p(double x);

// 计算 e^x - 1
double cephes_expm1(double x);
// 计算 x 的余弦的近似值减去 1
double cephes_cosm1(double x);

// 计算指数积分 En(x)
double cephes_expn(int n, double x);

// 计算椭圆积分 K 的完全椭圆积分值
double cephes_ellpe(double x);

// 计算椭圆积分 K 的完全椭圆积分 K'(m)
double cephes_ellpk(double x);

// 计算椭圆积分 E(phi, m)
double cephes_ellie(double phi, double m);

// 计算椭圆积分 K(phi, m)
double cephes_ellik(double phi, double m);

// 计算角度 x 的正弦（度数制）
double cephes_sindg(double x);

// 计算角度 x 的余弦（度数制）
double cephes_cosdg(double x);

// 计算角度 x 的正切（度数制）
double cephes_tandg(double x);

// 计算角度 x 的余切（度数制）
double cephes_cotdg(double x);

// 将度分秒转换为弧度
double cephes_radian(double d, double m, double s);

// 根据正态分布的累积分布函数的逆函数，返回概率 x 对应的 z 分数
double cephes_ndtri(double x);

// 计算二项分布累积分布函数的值
double cephes_bdtr(double k, int n, double p);

// 二项分布的累积分布函数的逆函数
double cephes_bdtri(double k, int n, double y);

// 二项分布的补充累积分布函数
double cephes_bdtrc(double k, int n, double p);

// 贝塔分布的逆累积分布函数
double cephes_btdtri(double aa, double bb, double yy0);

// 贝塔分布的累积分布函数
double cephes_btdtr(double a, double b, double x);

// 计算互补误差函数的逆函数
double cephes_erfcinv(double y);

// 计算 10 的 x 次方
double cephes_exp10(double x);

// 计算 2 的 x 次方
double cephes_exp2(double x);

// 计算 Fisher F 分布的累积分布函数
double cephes_fdtr(double a, double b, double x);

// 计算 Fisher F 分布的补充累积分布函数
double cephes_fdtrc(double a, double b, double x);

// Fisher F 分布的累积分布函数的逆函数
double cephes_fdtri(double a, double b, double y);

// 计算 Gamma 分布的累积分布函数
double cephes_gdtr(double a, double b, double x);

// Gamma 分布的补充累积分布函数
double cephes_gdtrc(double a, double b, double x);

// 计算 Owen's T 函数
double cephes_owens_t(double h, double a);

// 负二项分布的累积分布函数
double cephes_nbdtr(int k, int n, double p);

// 负二项分布的补充累积分布函数
double cephes_nbdtrc(int k, int n, double p);

// 负二项分布的累积分布函数的逆函数
double cephes_nbdtri(int k, int n, double p);

// 泊松分布的累积分布函数
double cephes_pdtr(double k, double m);

// 泊松分布的补充累积分布函数
double cephes_pdtrc(double k, double m);

// 泊松分布的累积分布函数的逆函数
double cephes_pdtri(int k, double y);

// 返回最接近 x 的整数
double cephes_round(double x);

// 计算斯宾斯函数 Si(x)
double cephes_spence(double x);

// 计算 Tukey Lambda 分布的累积分布函数
double cephes_tukeylambdacdf(double x, double lmbda);

// 计算斯特劳夫函数 H_v(x)
double cephes_struve_h(double v, double z);

// 计算斯特劳夫函数 L_v(x)
double cephes_struve_l(double v, double z);
```