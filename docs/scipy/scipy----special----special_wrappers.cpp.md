# `D:\src\scipysrc\scipy\scipy\special\special_wrappers.cpp`

```
// 包含特殊函数库中的头文件

#include "special_wrappers.h"
#include "special.h"
#include "special/airy.h"
#include "special/amos.h"
#include "special/bessel.h"
#include "special/binom.h"
#include "special/expint.h"
#include "special/fresnel.h"
#include "special/gamma.h"
#include "special/hyp2f1.h"
#include "special/kelvin.h"
#include "special/lambertw.h"
#include "special/log_exp.h"
#include "special/loggamma.h"
#include "special/mathieu.h"
#include "special/par_cyl.h"
#include "special/specfun.h"
#include "special/sph_bessel.h"
#include "special/sph_harm.h"
#include "special/sphd_wave.h"
#include "special/struve.h"
#include "special/trig.h"
#include "special/wright_bessel.h"

// 包含特殊函数库中的单个头文件
#include "special/binom.h"
#include "special/digamma.h"
#include "special/ellipk.h"
#include "special/gamma.h"
#include "special/hyp2f1.h"
#include "special/lambertw.h"
#include "special/loggamma.h"
#include "special/trig.h"
#include "special/wright_bessel.h"

// 包含特殊函数库中的cephes子目录中的头文件
#include "special/cephes/bdtr.h"
#include "special/cephes/besselpoly.h"
#include "special/cephes/beta.h"
#include "special/cephes/cbrt.h"
#include "special/cephes/chdtr.h"
#include "special/cephes/ellie.h"
#include "special/cephes/ellik.h"
#include "special/cephes/ellpe.h"
#include "special/cephes/ellpj.h"
#include "special/cephes/ellpk.h"
#include "special/cephes/erfinv.h"
#include "special/cephes/exp10.h"
#include "special/cephes/exp2.h"
#include "special/cephes/expn.h"
#include "special/cephes/fdtr.h"
#include "special/cephes/gamma.h"
#include "special/cephes/gdtr.h"
#include "special/cephes/hyp2f1.h"
#include "special/cephes/hyperg.h"
#include "special/cephes/i0.h"
#include "special/cephes/i1.h"
#include "special/cephes/igam.h"
#include "special/cephes/igami.h"
#include "special/cephes/incbet.h"
#include "special/cephes/incbi.h"
#include "special/cephes/j0.h"
#include "special/cephes/j1.h"
#include "special/cephes/jv.h"
#include "special/cephes/k0.h"
#include "special/cephes/k1.h"
#include "special/cephes/kolmogorov.h"
#include "special/cephes/lanczos.h"
#include "special/cephes/nbdtr.h"
#include "special/cephes/ndtr.h"
#include "special/cephes/ndtri.h"
#include "special/cephes/owens_t.h"
#include "special/cephes/pdtr.h"
#include "special/cephes/poch.h"
#include "special/cephes/rgamma.h"
#include "special/cephes/round.h"
#include "special/cephes/scipy_iv.h"
#include "special/cephes/sindg.h"
#include "special/cephes/spence.h"
#include "special/cephes/struve.h"
#include "special/cephes/tandg.h"
#include "special/cephes/trig.h"
#include "special/cephes/tukey.h"
#include "special/cephes/unity.h"
#include "special/cephes/yn.h"
#include "special/cephes/zeta.h"
#include "special/cephes/zetac.h"

// 包含特殊函数库中cephes子目录中的单个头文件
#include "special/cephes/airy.h"
#include "special/cephes/bdtr.h"
#include "special/cephes/beta.h"
#include "special/cephes/ellpj.h"
#include "special/cephes/ellpk.h"
#include "special/cephes/expn.h"
#include "special/cephes/fresnl.h"
#include "special/cephes/gamma.h"
#include "special/cephes/hyp2f1.h"
#include "special/cephes/jv.h"
// 引入特殊函数库中的头文件，用于特殊数学函数的计算
#include "special/cephes/kolmogorov.h"
#include "special/cephes/nbdtr.h"
#include "special/cephes/ndtr.h"
#include "special/cephes/ndtri.h"
#include "special/cephes/pdtr.h"
#include "special/cephes/shichi.h"
#include "special/cephes/sici.h"

// 使用标准命名空间，以便直接使用标准库中的对象和函数
using namespace std;

// 创建匿名命名空间，用于定义内部辅助函数，避免全局污染
namespace {

// 将复数结构体转换为复数类型
complex<double> to_complex(npy_cdouble z) {
    return {npy_creal(z), npy_cimag(z)};
}

// 将复数类型转换为复数结构体
npy_cdouble to_ccomplex(complex<double> z) {
    return {z.real(), z.imag()};
}

} // namespace

// 封装调用 chyp2f1 特殊函数的包装函数，处理复数输入和输出
npy_cdouble chyp2f1_wrap(double a, double b, double c, npy_cdouble z) {
    return to_ccomplex(special::chyp2f1(a, b, c, to_complex(z)));
}

// 封装调用 chyp1f1 特殊函数的包装函数，处理复数输入和输出
npy_cdouble chyp1f1_wrap(double a, double b, npy_cdouble z) {
    return to_ccomplex(special::chyp1f1(a, b, to_complex(z)));
}

// 封装调用 hypU 特殊函数的包装函数
double hypU_wrap(double a, double b, double x) {
    return special::hypu(a, b, x);
}

// 封装调用 hyp1f1 特殊函数的包装函数
double hyp1f1_wrap(double a, double b, double x) {
    return special::hyp1f1(a, b, x);
}

// 封装调用 itairy 特殊函数的包装函数
void special_itairy(double x, double *apt, double *bpt, double *ant, double *bnt) {
    special::itairy(x, *apt, *bpt, *ant, *bnt);
}

// 封装调用 exp1 特殊函数的包装函数
double special_exp1(double x) {
    return special::exp1(x);
}

// 封装调用 cexp1 特殊函数的包装函数，处理复数输入和输出
npy_cdouble special_cexp1(npy_cdouble z) {
    return to_ccomplex(special::exp1(to_complex(z)));
}

// 封装调用 expi 特殊函数的包装函数
double special_expi(double x) {
    return special::expi(x);
}

// 封装调用 cexpi 特殊函数的包装函数，处理复数输入和输出
npy_cdouble special_cexpi(npy_cdouble z) {
    return to_ccomplex(special::expi(to_complex(z)));
}

// 封装调用 exprel 特殊函数的包装函数
npy_double special_exprel(npy_double x) {
    return special::exprel(x);
}

// 封装调用 cerf 特殊函数的包装函数，处理复数输入和输出
npy_cdouble cerf_wrap(npy_cdouble z) {
    return to_ccomplex(special::cerf(to_complex(z)));
}

// 封装调用 itstruve0 特殊函数的包装函数
double special_itstruve0(double x) {
    return special::itstruve0(x);
}

// 封装调用 it2struve0 特殊函数的包装函数
double special_it2struve0(double x) {
    return special::it2struve0(x);
}

// 封装调用 itmodstruve0 特殊函数的包装函数
double special_itmodstruve0(double x) {
    return special::itmodstruve0(x);
}

// 封装调用 ber 特殊函数的包装函数
double special_ber(double x) {
    return special::ber(x);
}

// 封装调用 bei 特殊函数的包装函数
double special_bei(double x) {
    return special::bei(x);
}

// 封装调用 ker 特殊函数的包装函数
double special_ker(double x) {
    return special::ker(x);
}

// 封装调用 kei 特殊函数的包装函数
double special_kei(double x) {
    return special::kei(x);
}

// 封装调用 berp 特殊函数的包装函数
double special_berp(double x) {
    return special::berp(x);
}

// 封装调用 beip 特殊函数的包装函数
double special_beip(double x) {
    return special::beip(x);
}

// 封装调用 kerp 特殊函数的包装函数
double special_kerp(double x) {
    return special::kerp(x);
}

// 封装调用 keip 特殊函数的包装函数
double special_keip(double x) {
    return special::keip(x);
}

// 封装调用 kelvin 特殊函数的包装函数，处理复数输入和输出
void special_ckelvin(double x, npy_cdouble *Be, npy_cdouble *Ke, npy_cdouble *Bep, npy_cdouble *Kep) {
    special::kelvin(
        x, *reinterpret_cast<complex<double> *>(Be), *reinterpret_cast<complex<double> *>(Ke),
        *reinterpret_cast<complex<double> *>(Bep), *reinterpret_cast<complex<double> *>(Kep)
    );
}

// 封装调用 hyp2f1 特殊函数的包装函数，处理复数输入和输出
npy_cdouble hyp2f1_complex_wrap(double a, double b, double c, npy_cdouble z) {
    return to_ccomplex(special::hyp2f1(a, b, c, to_complex(z)));
}

// 封装调用 it1j0y0 特殊函数的包装函数
void it1j0y0_wrap(double x, double *j0int, double *y0int) {
    special::it1j0y0(x, *j0int, *y0int);
}

// 封装调用 it2j0y0 特殊函数的包装函数
void it2j0y0_wrap(double x, double *j0int, double *y0int) {
    special::it2j0y0(x, *j0int, *y0int);
}

// 封装调用 it1i0k0 特殊函数的包装函数
void it1i0k0_wrap(double x, double *i0int, double *k0int) {
    special::it1i0k0(x, *i0int, *k0int);
}


这段代码定义了一系列函数，用于封装调用特殊数学函数库中的各种数学函数，包括处理复数输入输出的函数和常规的数值处理函数。
void it2i0k0_wrap(double x, double *i0int, double *k0int) {
    // 调用特殊函数库中的it2i0k0函数，计算修正的Bessel函数I0和K0的积分
    special::it2i0k0(x, *i0int, *k0int);
}

int cfresnl_wrap(npy_cdouble z, npy_cdouble *zfs, npy_cdouble *zfc) {
    // 调用特殊函数库中的cfresnl函数，计算Fresnel积分S和C
    special::cfresnl(to_complex(z), reinterpret_cast<complex<double> *>(zfs), reinterpret_cast<complex<double> *>(zfc));
    return 0;
}

double cem_cva_wrap(double m, double q) {
    // 调用特殊函数库中的cem_cva函数，计算修正的完全椭圆积分第一类C(m, q)
    return special::cem_cva(m, q);
}

double sem_cva_wrap(double m, double q) {
    // 调用特殊函数库中的sem_cva函数，计算椭圆积分第二类S(m, q)
    return special::sem_cva(m, q);
}

void cem_wrap(double m, double q, double x, double *csf, double *csd) {
    // 调用特殊函数库中的cem函数，计算完全椭圆积分第一类C(m, q, x)和其导数
    special::cem(m, q, x, *csf, *csd);
}

void sem_wrap(double m, double q, double x, double *csf, double *csd) {
    // 调用特殊函数库中的sem函数，计算椭圆积分第二类S(m, q, x)和其导数
    special::sem(m, q, x, *csf, *csd);
}

void mcm1_wrap(double m, double q, double x, double *f1r, double *d1r) {
    // 调用特殊函数库中的mcm1函数，计算M1(m, q, x)和其导数
    special::mcm1(m, q, x, *f1r, *d1r);
}

void msm1_wrap(double m, double q, double x, double *f1r, double *d1r) {
    // 调用特殊函数库中的msm1函数，计算M2(m, q, x)和其导数
    special::msm1(m, q, x, *f1r, *d1r);
}

void mcm2_wrap(double m, double q, double x, double *f2r, double *d2r) {
    // 调用特殊函数库中的mcm2函数，计算M3(m, q, x)和其导数
    special::mcm2(m, q, x, *f2r, *d2r);
}

void msm2_wrap(double m, double q, double x, double *f2r, double *d2r) {
    // 调用特殊函数库中的msm2函数，计算M4(m, q, x)和其导数
    special::msm2(m, q, x, *f2r, *d2r);
}

double pmv_wrap(double m, double v, double x) {
    // 调用特殊函数库中的pmv函数，计算连带勒让德函数P(m, v, x)
    return special::pmv(m, v, x);
}

void pbwa_wrap(double a, double x, double *wf, double *wd) {
    // 调用特殊函数库中的pbwa函数，计算连带雅各比函数W(a, x)及其导数
    special::pbwa(a, x, *wf, *wd);
}

void pbdv_wrap(double v, double x, double *pdf, double *pdd) {
    // 调用特殊函数库中的pbdv函数，计算连带贝塞尔函数D(v, x)及其导数
    special::pbdv(v, x, *pdf, *pdd);
}

void pbvv_wrap(double v, double x, double *pvf, double *pvd) {
    // 调用特殊函数库中的pbvv函数，计算连带贝塞尔函数V(v, x)及其导数
    special::pbvv(v, x, *pvf, *pvd);
}

double prolate_segv_wrap(double m, double n, double c) {
    // 调用特殊函数库中的prolate_segv函数，计算长圆柱体的广义Stäckel参数
    return special::prolate_segv(m, n, c);
}

double oblate_segv_wrap(double m, double n, double c) {
    // 调用特殊函数库中的oblate_segv函数，计算短圆柱体的广义Stäckel参数
    return special::oblate_segv(m, n, c);
}

double prolate_aswfa_nocv_wrap(double m, double n, double c, double x, double *s1d) {
    double s1f;
    // 调用特殊函数库中的prolate_aswfa_nocv函数，计算无中心势场长圆柱体的角度谐波函数
    special::prolate_aswfa_nocv(m, n, c, x, s1f, *s1d);

    return s1f;
}

double oblate_aswfa_nocv_wrap(double m, double n, double c, double x, double *s1d) {
    double s1f;
    // 调用特殊函数库中的oblate_aswfa_nocv函数，计算无中心势场短圆柱体的角度谐波函数
    special::oblate_aswfa_nocv(m, n, c, x, s1f, *s1d);

    return s1f;
}

void prolate_aswfa_wrap(double m, double n, double c, double cv, double x, double *s1f, double *s1d) {
    // 调用特殊函数库中的prolate_aswfa函数，计算有中心势场长圆柱体的角度谐波函数和其导数
    special::prolate_aswfa(m, n, c, cv, x, *s1f, *s1d);
}

void oblate_aswfa_wrap(double m, double n, double c, double cv, double x, double *s1f, double *s1d) {
    // 调用特殊函数库中的oblate_aswfa函数，计算有中心势场短圆柱体的角度谐波函数和其导数
    special::oblate_aswfa(m, n, c, cv, x, *s1f, *s1d);
}

double prolate_radial1_nocv_wrap(double m, double n, double c, double x, double *r1d) {
    double r1f;
    // 调用特殊函数库中的prolate_radial1_nocv函数，计算无中心势场长圆柱体的径向波函数R1
    special::prolate_radial1_nocv(m, n, c, x, r1f, *r1d);

    return r1f;
}

double prolate_radial2_nocv_wrap(double m, double n, double c, double x, double *r2d) {
    double r2f;
    // 调用特殊函数库中的prolate_radial2_nocv函数，计算无中心势场长圆柱体的径向波函数R2
    special::prolate_radial2_nocv(m, n, c, x, r2f, *r2d);

    return r2f;
}

void prolate_radial1_wrap(double m, double n, double c, double cv, double x, double *r1f, double *r1d) {
    // 调用特殊函数库中的prolate_radial1函数，计算有中心势场长圆柱体的径向波函数R1及其导数
    special::prolate_radial1(m, n, c, cv, x, *r1f, *r1d);
}
// 调用 special 命名空间中的 prolate_radial2 函数，计算椭圆体第二类径向函数 R2 的值和导数
void prolate_radial2_wrap(double m, double n, double c, double cv, double x, double *r2f, double *r2d) {
    special::prolate_radial2(m, n, c, cv, x, *r2f, *r2d);
}

// 调用 special 命名空间中的 oblate_radial1_nocv 函数，计算旋转椭球体第一类径向函数 R1 的值和导数
double oblate_radial1_nocv_wrap(double m, double n, double c, double x, double *r1d) {
    double r1f;
    special::oblate_radial1_nocv(m, n, c, x, r1f, *r1d);

    return r1f;
}

// 调用 special 命名空间中的 oblate_radial2_nocv 函数，计算旋转椭球体第二类径向函数 R2 的值和导数
double oblate_radial2_nocv_wrap(double m, double n, double c, double x, double *r2d) {
    double r2f;
    special::oblate_radial2_nocv(m, n, c, x, r2f, *r2d);

    return r2f;
}

// 调用 special 命名空间中的 oblate_radial1 函数，计算旋转椭球体第一类径向函数 R1 的值和导数
void oblate_radial1_wrap(double m, double n, double c, double cv, double x, double *r1f, double *r1d) {
    special::oblate_radial1(m, n, c, cv, x, *r1f, *r1d);
}

// 调用 special 命名空间中的 oblate_radial2 函数，计算旋转椭球体第二类径向函数 R2 的值和导数
void oblate_radial2_wrap(double m, double n, double c, double cv, double x, double *r2f, double *r2d) {
    special::oblate_radial2(m, n, c, cv, x, *r2f, *r2d);
}

// 调用 special 命名空间中的 modified_fresnel_plus 函数，计算修改过的菲涅尔积分 F+ 和 K+
void modified_fresnel_plus_wrap(double x, npy_cdouble *Fplus, npy_cdouble *Kplus) {
    special::modified_fresnel_plus(
        x, *reinterpret_cast<complex<double> *>(Fplus), *reinterpret_cast<complex<double> *>(Kplus)
    );
}

// 调用 special 命名空间中的 modified_fresnel_minus 函数，计算修改过的菲涅尔积分 F- 和 K-
void modified_fresnel_minus_wrap(double x, npy_cdouble *Fminus, npy_cdouble *Kminus) {
    special::modified_fresnel_minus(
        x, *reinterpret_cast<complex<double> *>(Fminus), *reinterpret_cast<complex<double> *>(Kminus)
    );
}

// 调用 special 命名空间中的 special_sinpi 函数，计算正弦函数的派系乘积
double special_sinpi(double x) { return special::sinpi(x); }

// 调用 special 命名空间中的 special_csinpi 函数，计算复数正弦函数的派系乘积
npy_cdouble special_csinpi(npy_cdouble z) { return to_ccomplex(special::sinpi(to_complex(z))); }

// 调用 special 命名空间中的 special_cospi 函数，计算余弦函数的派系乘积
double special_cospi(double x) { return special::cospi(x); }

// 调用 special 命名空间中的 special_airy 函数，计算 Airy 函数的值及其导数
void special_airy(double x, double *ai, double *aip, double *bi, double *bip) {
    special::airy(x, *ai, *aip, *bi, *bip);
}

// 调用 special 命名空间中的 special_cairy 函数，计算复数 Airy 函数的值及其导数
void special_cairy(npy_cdouble z, npy_cdouble *ai, npy_cdouble *aip, npy_cdouble *bi, npy_cdouble *bip) {
    special::airy(
        to_complex(z), *reinterpret_cast<complex<double> *>(ai), *reinterpret_cast<complex<double> *>(aip),
        *reinterpret_cast<complex<double> *>(bi), *reinterpret_cast<complex<double> *>(bip)
    );
}

// 调用 special 命名空间中的 special_airye 函数，计算修正 Airy 函数的值及其导数
void special_airye(double z, double *ai, double *aip, double *bi, double *bip) {
    special::airye(z, *ai, *aip, *bi, *bip);
}

// 调用 special 命名空间中的 special_cairye 函数，计算复数修正 Airy 函数的值及其导数
void special_cairye(npy_cdouble z, npy_cdouble *ai, npy_cdouble *aip, npy_cdouble *bi, npy_cdouble *bip) {
    special::airye(
        to_complex(z), *reinterpret_cast<complex<double> *>(ai), *reinterpret_cast<complex<double> *>(aip),
        *reinterpret_cast<complex<double> *>(bi), *reinterpret_cast<complex<double> *>(bip)
    );
}

// 调用 special 命名空间中的 special_cyl_bessel_j 函数，计算复数球形贝塞尔函数 J 的值
double special_cyl_bessel_j(double v, double x) { return special::cyl_bessel_j(v, x); }

// 调用 special 命名空间中的 special_ccyl_bessel_j 函数，计算复数球形贝塞尔函数 J 的值
npy_cdouble special_ccyl_bessel_j(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_bessel_j(v, to_complex(z)));
}

// 调用 special 命名空间中的 special_cyl_bessel_je 函数，计算复数球形贝塞尔函数 Je 的值
double special_cyl_bessel_je(double v, double z) { return special::cyl_bessel_je(v, z); }

// 调用 special 命名空间中的 special_ccyl_bessel_je 函数，计算复数球形贝塞尔函数 Je 的值
npy_cdouble special_ccyl_bessel_je(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_bessel_je(v, to_complex(z)));
}

// 调用 special 命名空间中的 special_cyl_bessel_y 函数，计算复数球形贝塞尔函数 Y 的值
double special_cyl_bessel_y(double v, double x) { return special::cyl_bessel_y(v, x); }
// 计算修正的圆柱贝塞尔函数 Y_v(z)，返回复数结果
npy_cdouble special_ccyl_bessel_y(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_bessel_y(v, to_complex(z)));
}

// 计算修正的圆柱贝塞尔函数 Y_v(z)，返回实数结果
double special_cyl_bessel_ye(double v, double z) {
    return special::cyl_bessel_ye(v, z);
}

// 计算修正的圆柱贝塞尔函数 Y_v(z)，返回复数结果
npy_cdouble special_ccyl_bessel_ye(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_bessel_ye(v, to_complex(z)));
}

// 计算圆柱贝塞尔函数 I_v(z)，返回实数结果
double special_cyl_bessel_i(double v, double z) {
    return special::cyl_bessel_i(v, z);
}

// 计算圆柱贝塞尔函数 I_v(z)，返回复数结果
npy_cdouble special_ccyl_bessel_i(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_bessel_i(v, to_complex(z)));
}

// 计算修改过的圆柱贝塞尔函数 I_v(z)，返回实数结果
double special_cyl_bessel_ie(double v, double z) {
    return special::cyl_bessel_ie(v, z);
}

// 计算修改过的圆柱贝塞尔函数 I_v(z)，返回复数结果
npy_cdouble special_ccyl_bessel_ie(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_bessel_ie(v, to_complex(z)));
}

// 计算整数阶圆柱贝塞尔函数 K_n(z)，返回实数结果
double special_cyl_bessel_k_int(int n, double z) {
    return special::cyl_bessel_k(static_cast<double>(n), z);
}

// 计算圆柱贝塞尔函数 K_v(z)，返回实数结果
double special_cyl_bessel_k(double v, double z) {
    return special::cyl_bessel_k(v, z);
}

// 计算圆柱贝塞尔函数 K_v(z)，返回复数结果
npy_cdouble special_ccyl_bessel_k(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_bessel_k(v, to_complex(z)));
}

// 计算修改过的圆柱贝塞尔函数 K_v(z)，返回实数结果
double special_cyl_bessel_ke(double v, double z) {
    return special::cyl_bessel_ke(v, z);
}

// 计算修改过的圆柱贝塞尔函数 K_v(z)，返回复数结果
npy_cdouble special_ccyl_bessel_ke(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_bessel_ke(v, to_complex(z)));
}

// 计算第一类修正的汉克尔函数 H^(1)_v(z)，返回复数结果
npy_cdouble special_ccyl_hankel_1(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_hankel_1(v, to_complex(z)));
}

// 计算修改过的第一类修正的汉克尔函数 H^(1)_v(z)，返回复数结果
npy_cdouble special_ccyl_hankel_1e(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_hankel_1e(v, to_complex(z)));
}

// 计算第二类修正的汉克尔函数 H^(2)_v(z)，返回复数结果
npy_cdouble special_ccyl_hankel_2(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_hankel_2(v, to_complex(z)));
}

// 计算修改过的第二类修正的汉克尔函数 H^(2)_v(z)，返回复数结果
npy_cdouble special_ccyl_hankel_2e(double v, npy_cdouble z) {
    return to_ccomplex(special::cyl_hankel_2e(v, to_complex(z)));
}

// 计算二项式系数 C(n, k)
double binom_wrap(double n, double k) {
    return special::binom(n, k);
}

// 计算二项式系数 C(n, k)
double special_binom(double n, double k) {
    return special::binom(n, k);
}

// 计算ψ(z)，即 digamma 函数
double special_digamma(double z) {
    return special::digamma(z);
}

// 计算ψ(z)，即 digamma 函数，返回复数结果
npy_cdouble special_cdigamma(npy_cdouble z) {
    return to_ccomplex(special::digamma(to_complex(z)));
}

// 计算Γ(x)，即 gamma 函数
double special_gamma(double x) {
    return special::gamma(x);
}

// 计算Γ(z)，即 gamma 函数，返回复数结果
npy_cdouble special_cgamma(npy_cdouble z) {
    return to_ccomplex(special::gamma(to_complex(z)));
}

// 计算 Γ'(x)，即 rgamma 函数
double special_rgamma(double x) {
    return special::rgamma(x);
}

// 计算 Γ'(z)，即 rgamma 函数，返回复数结果
npy_cdouble special_crgamma(npy_cdouble z) {
    return to_ccomplex(special::rgamma(to_complex(z)));
}

// 计算 sigmoid 函数 expit(x)，返回 float 类型
float special_expitf(float x) {
    return special::expit(x);
}

// 计算 sigmoid 函数 expit(x)，返回 double 类型
double special_expit(double x) {
    return special::expit(x);
}

// 计算 sigmoid 函数 expit(x)，返回 long double 类型
npy_longdouble special_expitl(npy_longdouble x) {
    return special::expit(x);
}

// 计算 logit 函数的 sigmoid 逆函数，返回 float 类型
float special_log_expitf(float x) {
    return special::log_expit(x);
}

// 计算 logit 函数的 sigmoid 逆函数，返回 double 类型
double special_log_expit(double x) {
    return special::log_expit(x);
}

// 计算 logit 函数的 sigmoid 逆函数，返回 long double 类型
npy_longdouble special_log_expitl(npy_longdouble x) {
    return special::log_expit(x);
}

// 计算 logistic 函数的 logit 函数，返回 float 类型
float special_logitf(float x) {
    return special::logit(x);
}
// 使用 special 命名空间中的 logit 函数计算输入 x 的特殊对数
double special_logit(double x) { return special::logit(x); };

// 使用 special 命名空间中的 logit 函数计算输入 x 的长双精度特殊对数
npy_longdouble special_logitl(npy_longdouble x) { return special::logit(x); };

// 使用 special 命名空间中的 loggamma 函数计算输入 x 的特殊对数 Gamma 函数的值
double special_loggamma(double x) { return special::loggamma(x); }

// 使用 special 命名空间中的 loggamma 函数计算输入复数 z 的特殊对数 Gamma 函数的值
npy_cdouble special_cloggamma(npy_cdouble z) { return to_ccomplex(special::loggamma(to_complex(z))); }

// 使用 special 命名空间中的 hyp2f1 函数计算输入 a, b, c, z 的超几何函数 2F1 的值
double special_hyp2f1(double a, double b, double c, double z) { return special::hyp2f1(a, b, c, z); }

// 使用 special 命名空间中的 hyp2f1 函数计算输入 a, b, c, z 的复数形式的超几何函数 2F1 的值
npy_cdouble special_chyp2f1(double a, double b, double c, npy_cdouble z) {
    return to_ccomplex(special::hyp2f1(a, b, c, to_complex(z)));
}

// 使用 special 命名空间中的 lambertw 函数计算输入复数 z 的 Lambert W 函数的值
npy_cdouble special_lambertw(npy_cdouble z, long k, double tol) {
    return to_ccomplex(special::lambertw(to_complex(z), k, tol));
}

// 使用全局命名空间中的 sph_harm 函数计算输入 m, n, theta, phi 的球谐函数值
npy_cdouble special_sph_harm(long m, long n, double theta, double phi) {
    return to_ccomplex(::sph_harm(m, n, theta, phi));
}

// 使用全局命名空间中的 sph_harm 函数计算输入 m, n, theta, phi 的球谐函数值（将 m, n 强制转换为 long 类型）
npy_cdouble special_sph_harm_unsafe(double m, double n, double theta, double phi) {
    return to_ccomplex(::sph_harm(static_cast<long>(m), static_cast<long>(n), theta, phi));
}

// 使用 special::cephes 命名空间中的 hyp2f1 函数计算输入 a, b, c, x 的超几何函数 2F1 的值
double cephes_hyp2f1_wrap(double a, double b, double c, double x) { return special::cephes::hyp2f1(a, b, c, x); }

// 使用 special::cephes 命名空间中的 airy 函数计算输入 x 的 Airy 函数值，并输出 ai, aip, bi, bip
double cephes_airy_wrap(double x, double *ai, double *aip, double *bi, double *bip) {
    return special::cephes::airy(x, ai, aip, bi, bip);
}

// 使用 special::cephes 命名空间中的 beta 函数计算输入 a, b 的 Beta 函数值
double cephes_beta_wrap(double a, double b) { return special::cephes::beta(a, b); }

// 使用 special::cephes 命名空间中的 lbeta 函数计算输入 a, b 的自然对数 Beta 函数值
double cephes_lbeta_wrap(double a, double b) { return special::cephes::lbeta(a, b); }

// 使用 special::cephes 命名空间中的 bdtr 函数计算输入 k, n, p 的负二项分布值
double cephes_bdtr_wrap(double k, int n, double p) { return special::cephes::bdtr(k, n, p); }

// 使用 special::cephes 命名空间中的 bdtri 函数计算输入 k, n, y 的负二项分布分位点
double cephes_bdtri_wrap(double k, int n, double y) { return special::cephes::bdtri(k, n, y); }

// 使用 special::cephes 命名空间中的 bdtrc 函数计算输入 k, n, p 的补负二项分布值
double cephes_bdtrc_wrap(double k, int n, double p) { return special::cephes::bdtrc(k, n, p); }

// 使用 special::cephes 命名空间中的 cosm1 函数计算输入 x 的 cos(x) - 1
double cephes_cosm1_wrap(double x) { return special::cephes::cosm1(x); }

// 使用 special::cephes 命名空间中的 expm1 函数计算输入 x 的 exp(x) - 1
double cephes_expm1_wrap(double x) { return special::cephes::expm1(x); }

// 使用 special::cephes 命名空间中的 expn 函数计算输入 n, x 的指数积分函数值
double cephes_expn_wrap(int n, double x) { return special::cephes::expn(n, x); }

// 使用 special::cephes 命名空间中的 log1p 函数计算输入 x 的 log(1 + x)
double cephes_log1p_wrap(double x) { return special::cephes::log1p(x); }

// 使用 special::cephes 命名空间中的 Gamma 函数计算输入 x 的 Gamma 函数值
double cephes_gamma_wrap(double x) { return special::cephes::Gamma(x); }

// 使用 special::cephes 命名空间中的 gammasgn 函数计算输入 x 的 Gamma 函数符号值
double cephes_gammasgn_wrap(double x) { return special::cephes::gammasgn(x); }

// 使用 special::cephes 命名空间中的 lgam 函数计算输入 x 的自然对数 Gamma 函数值
double cephes_lgam_wrap(double x) { return special::cephes::lgam(x); }

// 使用 special::cephes 命名空间中的 iv 函数计算输入 v, x 的修正贝塞尔函数 Iv
double cephes_iv_wrap(double v, double x) { return special::cephes::iv(v, x); }

// 使用 special::cephes 命名空间中的 jv 函数计算输入 v, x 的贝塞尔函数 Jv
double cephes_jv_wrap(double v, double x) { return special::cephes::jv(v, x); }

// 使用 special::cephes 命名空间中的 ellpj 函数计算输入 u, m 的 Jacobian 椭圆函数值，并输出 sn, cn, dn, ph
int cephes_ellpj_wrap(double u, double m, double *sn, double *cn, double *dn, double *ph) {
    return special::cephes::ellpj(u, m, sn, cn, dn, ph);
}

// 使用 special::cephes 命名空间中的 ellpk 函数计算输入 x 的完全椭圆积分 K 的值
double cephes_ellpk_wrap(double x) { return special::cephes::ellpk(x); }

// 使用 special::cephes 命名空间中的 fresnl 函数计算输入 xxa 的 Fresnel 积分值，并输出 ssa, cca
int cephes_fresnl_wrap(double xxa, double *ssa, double *cca) { return special::cephes::fresnl(xxa, ssa, cca); }

// 使用 special::cephes 命名空间中的 nbdtr 函数计算输入 k, n, p 的负二项分布值
double cephes_nbdtr_wrap(int k, int n, double p) { return special::cephes::nbdtr(k, n, p); }

// 使用 special::cephes 命名空间中的 nbdtrc 函数计算输入 k, n, p 的补负二项分布值
double cephes_nbdtrc_wrap(int k, int n, double p) { return special::cephes::nbdtrc(k, n, p); }
// 调用 Cephes 库中的 nbdtri 函数，计算负二项分布的累积分布函数的逆函数
double cephes_nbdtri_wrap(int k, int n, double p) { return special::cephes::nbdtri(k, n, p); }

// 调用 Cephes 库中的 ndtr 函数，计算标准正态分布的累积分布函数
double cephes_ndtr_wrap(double x) { return special::cephes::ndtr(x); }

// 调用 Cephes 库中的 ndtri 函数，计算标准正态分布的累积分布函数的逆函数
double cephes_ndtri_wrap(double x) { return special::cephes::ndtri(x); }

// 调用 Cephes 库中的 pdtri 函数，计算 F 分布的累积分布函数的逆函数
double cephes_pdtri_wrap(int k, double y) { return special::cephes::pdtri(k, y); }

// 调用 Cephes 库中的 poch 函数，计算 Pochhammer 符号
double cephes_poch_wrap(double x, double m) { return special::cephes::poch(x, m); }

// 调用 Cephes 库中的 sici 函数，计算 sine and cosine integral
int cephes_sici_wrap(double x, double *si, double *ci) { return special::cephes::sici(x, si, ci); }

// 调用 Cephes 库中的 shichi 函数，计算 hyperbolic sine and cosine integrals
int cephes_shichi_wrap(double x, double *si, double *ci) { return special::cephes::shichi(x, si, ci); }

// 调用 Cephes 库中的 smirnov 函数，计算 Smirnov one-sided test statistic
double cephes_smirnov_wrap(int n, double x) { return special::cephes::smirnov(n, x); }

// 调用 Cephes 库中的 smirnovc 函数，计算 complementary Smirnov one-sided test statistic
double cephes_smirnovc_wrap(int n, double x) { return special::cephes::smirnovc(n, x); }

// 调用 Cephes 库中的 smirnovi 函数，计算 inverse Smirnov one-sided test statistic
double cephes_smirnovi_wrap(int n, double x) { return special::cephes::smirnovi(n, x); }

// 调用 Cephes 库中的 smirnovci 函数，计算 inverse complementary Smirnov one-sided test statistic
double cephes_smirnovci_wrap(int n, double x) { return special::cephes::smirnovci(n, x); }

// 调用 Cephes 库中的 smirnovp 函数，计算 Kolmogorov-Smirnov distribution 的累积分布函数
double cephes_smirnovp_wrap(int n, double x) { return special::cephes::smirnovp(n, x); }

// 调用 Cephes 库中的 detail::struve_asymp_large_z 函数，计算 Struve 函数的渐近展开式（大 z 情况）
double cephes__struve_asymp_large_z(double v, double z, int is_h, double *err) {
    return special::cephes::detail::struve_asymp_large_z(v, z, is_h, err);
}

// 调用 Cephes 库中的 detail::struve_bessel_series 函数，计算 Struve 函数的 Bessel 级数展开
double cephes__struve_bessel_series(double v, double z, int is_h, double *err) {
    return special::cephes::detail::struve_bessel_series(v, z, is_h, err);
}

// 调用 Cephes 库中的 detail::struve_power_series 函数，计算 Struve 函数的幂级数展开
double cephes__struve_power_series(double v, double z, int is_h, double *err) {
    return special::cephes::detail::struve_power_series(v, z, is_h, err);
}

// 调用 Cephes 库中的 yn 函数，计算贝塞尔函数第二类的第一类形式
double cephes_yn_wrap(int n, double x) { return special::cephes::yn(n, x); }

// 调用 Cephes 库中的 polevl 函数，计算多项式函数的值
double cephes_polevl_wrap(double x, const double coef[], int N) { return special::cephes::polevl(x, coef, N); }

// 调用 Cephes 库中的 p1evl 函数，计算多项式函数的值
double cephes_p1evl_wrap(double x, const double coef[], int N) { return special::cephes::p1evl(x, coef, N); }

// 调用特殊数学函数库中的 gammaln 函数，计算对数伽玛函数
double gammaln_wrap(double x) { return special::gammaln(x); }

// 调用特殊数学函数库中的 wright_bessel 函数，计算特殊的 Bessel 函数
double special_wright_bessel(double a, double b, double x) { return special::wright_bessel(a, b, x); }

// 调用特殊数学函数库中的 log_wright_bessel 函数，计算特殊 Bessel 函数的对数
double special_log_wright_bessel(double a, double b, double x) { return special::log_wright_bessel(a, b, x); }

// 调用特殊数学函数库中的 scaled_exp1 函数，计算 scaled exp1 函数的值
double special_scaled_exp1(double x) { return special::scaled_exp1(x); }

// 调用特殊数学函数库中的 sph_bessel_j 函数，计算球 Bessel 函数 J_n(x)
double special_sph_bessel_j(long n, double x) { return special::sph_bessel_j(n, x); }

// 调用特殊数学函数库中的 sph_bessel_j 函数，计算复数球 Bessel 函数 J_n(z)
npy_cdouble special_csph_bessel_j(long n, npy_cdouble z) {
    return to_ccomplex(special::sph_bessel_j(n, to_complex(z)));
}

// 调用特殊数学函数库中的 sph_bessel_j_jac 函数，计算球 Bessel 函数 J_n(x) 的导数
double special_sph_bessel_j_jac(long n, double x) { return special::sph_bessel_j_jac(n, x); }

// 调用特殊数学函数库中的 sph_bessel_j_jac 函数，计算复数球 Bessel 函数 J_n(z) 的导数
npy_cdouble special_csph_bessel_j_jac(long n, npy_cdouble z) {
    return to_ccomplex(special::sph_bessel_j_jac(n, to_complex(z)));
}

// 调用特殊数学函数库中的 sph_bessel_y 函数，计算球 Bessel 函数 Y_n(x)
double special_sph_bessel_y(long n, double x) { return special::sph_bessel_y(n, x); }

// 调用特殊数学函数库中的 sph_bessel_y 函数，计算复数球 Bessel 函数 Y_n(z)
npy_cdouble special_csph_bessel_y(long n, npy_cdouble z) {
    return to_ccomplex(special::sph_bessel_y(n, to_complex(z)));
}

// 调用特殊数学函数库中的 sph_bessel_y_jac 函数，计算球 Bessel 函数 Y_n(x) 的导数
double special_sph_bessel_y_jac(long n, double x) { return special::sph_bessel_y_jac(n, x); }
// 返回复数形式的特殊球面贝塞尔函数 Y_n(z)
npy_cdouble special_csph_bessel_y_jac(long n, npy_cdouble z) {
    return to_ccomplex(special::sph_bessel_y_jac(n, to_complex(z)));
}

// 返回特殊球面贝塞尔函数 I_n(x)
double special_sph_bessel_i(long n, double x) { return special::sph_bessel_i(n, x); }

// 返回复数形式的特殊球面贝塞尔函数 I_n(z)
npy_cdouble special_csph_bessel_i(long n, npy_cdouble z) {
    return to_ccomplex(special::sph_bessel_i(n, to_complex(z)));
}

// 返回特殊球面贝塞尔函数 I_n(x) 的导数
double special_sph_bessel_i_jac(long n, double x) { return special::sph_bessel_i_jac(n, x); }

// 返回复数形式的特殊球面贝塞尔函数 I_n(z) 的导数
npy_cdouble special_csph_bessel_i_jac(long n, npy_cdouble z) {
    return to_ccomplex(special::sph_bessel_i_jac(n, to_complex(z)));
}

// 返回特殊球面贝塞尔函数 K_n(x)
double special_sph_bessel_k(long n, double x) { return special::sph_bessel_k(n, x); }

// 返回复数形式的特殊球面贝塞尔函数 K_n(z)
npy_cdouble special_csph_bessel_k(long n, npy_cdouble z) {
    return to_ccomplex(special::sph_bessel_k(n, to_complex(z)));
}

// 返回特殊球面贝塞尔函数 K_n(x) 的导数
double special_sph_bessel_k_jac(long n, double x) { return special::sph_bessel_k_jac(n, x); }

// 返回复数形式的特殊球面贝塞尔函数 K_n(z) 的导数
npy_cdouble special_csph_bessel_k_jac(long n, npy_cdouble z) {
    return to_ccomplex(special::sph_bessel_k_jac(n, to_complex(z)));
}

// 返回特殊椭圆积分 K(m)
double special_ellipk(double m) { return special::ellipk(m); }

// 返回 Cephes 库中的 Bessel 多项式函数
double cephes_besselpoly(double a, double lambda, double nu) { return special::cephes::besselpoly(a, lambda, nu); }

// 返回 Cephes 库中的 Beta 函数
double cephes_beta(double a, double b) { return special::cephes::beta(a, b); }

// 返回 Cephes 库中的卡方分布的累积分布函数
double cephes_chdtr(double df, double x) { return special::cephes::chdtr(df, x); }

// 返回 Cephes 库中的卡方分布的补累积分布函数
double cephes_chdtrc(double df, double x) { return special::cephes::chdtrc(df, x); }

// 返回 Cephes 库中的卡方分布的逆累积分布函数
double cephes_chdtri(double df, double y) { return special::cephes::chdtri(df, y); }

// 返回 Cephes 库中的对数 Beta 函数
double cephes_lbeta(double a, double b) { return special::cephes::lbeta(a, b); }

// 返回 Cephes 库中的 sin(πx) 函数
double cephes_sinpi(double x) { return special::cephes::sinpi(x); }

// 返回 Cephes 库中的 cos(πx) 函数
double cephes_cospi(double x) { return special::cephes::cospi(x); }

// 返回 Cephes 库中的立方根函数
double cephes_cbrt(double x) { return special::cephes::detail::cbrt(x); }

// 返回 Cephes 库中的 Gamma 函数
double cephes_Gamma(double x) { return special::cephes::Gamma(x); }

// 返回 Cephes 库中的 Gamma 函数的符号
double cephes_gammasgn(double x) { return special::cephes::gammasgn(x); }

// 返回 Cephes 库中的超几何函数 2F1
double cephes_hyp2f1(double a, double b, double c, double x) { return special::cephes::hyp2f1(a, b, c, x); }

// 返回 Cephes 库中的修正 Bessel 函数 I_0(x)
double cephes_i0(double x) { return special::cephes::i0(x); }

// 返回 Cephes 库中的修正 Bessel 函数 I_0e(x)
double cephes_i0e(double x) { return special::cephes::i0e(x); }

// 返回 Cephes 库中的修正 Bessel 函数 I_1(x)
double cephes_i1(double x) { return special::cephes::i1(x); }

// 返回 Cephes 库中的修正 Bessel 函数 I_1e(x)
double cephes_i1e(double x) { return special::cephes::i1e(x); }

// 返回 Cephes 库中的修正 Bessel 函数 Iv(v, x)
double cephes_iv(double v, double x) { return special::cephes::iv(v, x); }

// 返回 Cephes 库中的 Bessel 函数 J_0(x)
double cephes_j0(double x) { return special::cephes::j0(x); }

// 返回 Cephes 库中的 Bessel 函数 J_1(x)
double cephes_j1(double x) { return special::cephes::j1(x); }

// 返回 Cephes 库中的修正 Bessel 函数 K_0(x)
double cephes_k0(double x) { return special::cephes::k0(x); }

// 返回 Cephes 库中的修正 Bessel 函数 K_0e(x)
double cephes_k0e(double x) { return special::cephes::k0e(x); }

// 返回 Cephes 库中的修正 Bessel 函数 K_1(x)
double cephes_k1(double x) { return special::cephes::k1(x); }

// 返回 Cephes 库中的修正 Bessel 函数 K_1e(x)
double cephes_k1e(double x) { return special::cephes::k1e(x); }

// 返回 Cephes 库中的 Bessel 函数 Y_0(x)
double cephes_y0(double x) { return special::cephes::y0(x); }

// 返回 Cephes 库中的 Bessel 函数 Y_1(x)
double cephes_y1(double x) { return special::cephes::y1(x); }
// 调用特殊数学函数库中的贝塞尔函数 Y_n(n, x)
double cephes_yn(int n, double x) { return special::cephes::yn(n, x); }

// 调用特殊数学函数库中的不完全伽玛函数 γ(a, x)
double cephes_igam(double a, double x) { return special::cephes::igam(a, x); }

// 调用特殊数学函数库中的补充不完全伽玛函数 γᶜ(a, x)
double cephes_igamc(double a, double x) { return special::cephes::igamc(a, x); }

// 调用特殊数学函数库中的不完全伽玛函数的逆函数 γ⁻¹(a, p)
double cephes_igami(double a, double p) { return special::cephes::igami(a, p); }

// 调用特殊数学函数库中的补充不完全伽玛函数的逆函数 γᶜ⁻¹(a, p)
double cephes_igamci(double a, double p) { return special::cephes::igamci(a, p); }

// 调用特殊数学函数库中的不完全伽玛函数的对数倍乘积 γ⁺ₐ(a, x)
double cephes_igam_fac(double a, double x) { return special::cephes::detail::igam_fac(a, x); }

// 调用特殊数学函数库中的兰之斯和指数函数的缩放版本
double cephes_lanczos_sum_expg_scaled(double x) { return special::cephes::lanczos_sum_expg_scaled(x); }

// 调用特殊数学函数库中的科尔莫哥洛夫分布函数 K(x)
double cephes_kolmogorov(double x) { return special::cephes::kolmogorov(x); }

// 调用特殊数学函数库中的科尔莫哥洛夫分布函数的补函数 Kₓ⁻¹(x)
double cephes_kolmogc(double x) { return special::cephes::kolmogc(x); }

// 调用特殊数学函数库中的科尔莫哥洛夫分布函数的逆函数 K⁻¹(x)
double cephes_kolmogi(double x) { return special::cephes::kolmogi(x); }

// 调用特殊数学函数库中的科尔莫哥洛夫分布函数的补逆函数 Kₓ⁻¹⁻¹(x)
double cephes_kolmogci(double x) { return special::cephes::kolmogci(x); }

// 调用特殊数学函数库中的科尔莫哥洛夫分布函数的累积分布函数 P(x)
double cephes_kolmogp(double x) { return special::cephes::kolmogp(x); }

// 调用特殊数学函数库中的斯米尔诺夫分布函数 Dₙ⁺₁(x)
double cephes_smirnov(int n, double x) { return special::cephes::smirnov(n, x); }

// 调用特殊数学函数库中的斯米尔诺夫分布函数的补函数 Dₙ⁻¹(x)
double cephes_smirnovc(int n, double x) { return special::cephes::smirnovc(n, x); }

// 调用特殊数学函数库中的斯米尔诺夫分布函数的逆函数 D⁻¹ₙ⁺₁(n, x)
double cephes_smirnovi(int n, double x) { return special::cephes::smirnovi(n, x); }

// 调用特殊数学函数库中的斯米尔诺夫分布函数的补逆函数 D⁻¹ₙ⁻¹(n, x)
double cephes_smirnovci(int n, double x) { return special::cephes::smirnovci(n, x); }

// 调用特殊数学函数库中的斯米尔诺夫分布函数的累积分布函数 Pₙ⁺₁(n, x)
double cephes_smirnovp(int n, double x) { return special::cephes::smirnovp(n, x); }

// 调用特殊数学函数库中的标准正态分布函数 Φ(x)
double cephes_ndtr(double x) { return special::cephes::ndtr(x); }

// 调用特殊数学函数库中的误差函数 erf(x)
double cephes_erf(double x) { return special::cephes::erf(x); }

// 调用特殊数学函数库中的互补误差函数 erfc(x)
double cephes_erfc(double x) { return special::cephes::erfc(x); }

// 调用特殊数学函数库中的泊松函数 P(x, m)
double cephes_poch(double x, double m) { return special::cephes::poch(x, m); }

// 调用特殊数学函数库中的伽玛函数 Γ(x)
double cephes_rgamma(double x) { return special::cephes::rgamma(x); }

// 调用特殊数学函数库中的默顿-斯隆兹塔塔函数 ζ(x, q)
double cephes_zeta(double x, double q) { return special::cephes::zeta(x, q); }

// 调用特殊数学函数库中的默顿-斯隆兹塔塔函数的互补函数 ζᶜ(x)
double cephes_zetac(double x) { return special::cephes::zetac(x); }

// 调用特殊数学函数库中的默顿-斯隆兹塔塔函数 ζ(x)
double cephes_riemann_zeta(double x) { return special::cephes::riemann_zeta(x); }

// 调用特殊数学函数库中的 log(1 + x)
double cephes_log1p(double x) { return special::cephes::log1p(x); }

// 调用特殊数学函数库中的 log(1 + x) - x
double cephes_log1pmx(double x) { return special::cephes::log1pmx(x); }

// 调用特殊数学函数库中的 log(Gamma(1 + x))
double cephes_lgam1p(double x) { return special::cephes::lgam1p(x); }

// 调用特殊数学函数库中的 exp(x) - 1
double cephes_expm1(double x) { return special::cephes::expm1(x); }

// 调用特殊数学函数库中的 cos(x) - 1
double cephes_cosm1(double x) { return special::cephes::cosm1(x); }

// 调用特殊数学函数库中的指数函数 Eₙ(x)
double cephes_expn(int n, double x) { return special::cephes::expn(n, x); }

// 调用特殊数学函数库中的椭圆积分 Eₚ(x)
double cephes_ellpe(double x) { return special::cephes::ellpe(x); }

// 调用特殊数学函数库中的椭圆积分 Kₚ(x)
double cephes_ellpk(double x) { return special::cephes::ellpk(x); }

// 调用特殊数学函数库中的完全椭圆积分 E(φ, m)
double cephes_ellie(double phi, double m) { return special::cephes::ellie(phi, m); }

// 调用特殊数学函数库中的完全椭圆积分 K(φ, m)
double cephes_ellik(double phi, double m) { return special::cephes::ellik(phi, m); }

// 调用特殊数学函数库中的正弦函数（角度制） sin(x)
double cephes_sindg(double x) { return special::cephes::sindg(x); }

// 调用特殊数学函数库中的余弦函数（角度制） cos(x)
double cephes_cos
// 调用 special::cephes::radian 函数，将度、分和秒转换为弧度
double cephes_radian(double d, double m, double s) { return special::cephes::radian(d, m, s); }

// 调用 special::cephes::ndtri 函数，计算标准正态分布的逆累积分布函数
double cephes_ndtri(double x) { return special::cephes::ndtri(x); }

// 调用 special::cephes::bdtr 函数，计算二项分布的累积分布函数
double cephes_bdtr(double k, int n, double p) { return special::cephes::bdtr(k, n, p); }

// 调用 special::cephes::bdtri 函数，计算二项分布的逆累积分布函数
double cephes_bdtri(double k, int n, double y) { return special::cephes::bdtri(k, n, y); }

// 调用 special::cephes::bdtrc 函数，计算二项分布的补累积分布函数
double cephes_bdtrc(double k, int n, double p) { return special::cephes::bdtrc(k, n, p); }

// 调用 special::cephes::incbi 函数，计算不完全贝塔函数的逆函数
double cephes_btdtri(double aa, double bb, double yy0) { return special::cephes::incbi(aa, bb, yy0); }

// 调用 special::cephes::incbet 函数，计算不完全贝塔函数
double cephes_btdtr(double a, double b, double x) { return special::cephes::incbet(a, b, x); }

// 调用 special::cephes::erfcinv 函数，计算余误差函数的逆函数
double cephes_erfcinv(double y) { return special::cephes::erfcinv(y); }

// 调用 special::cephes::exp10 函数，计算 10 的 x 次方
double cephes_exp10(double x) { return special::cephes::exp10(x); }

// 调用 special::cephes::exp2 函数，计算 2 的 x 次方
double cephes_exp2(double x) { return special::cephes::exp2(x); }

// 调用 special::cephes::fdtr 函数，计算 F 分布的累积分布函数
double cephes_fdtr(double a, double b, double x) { return special::cephes::fdtr(a, b, x); }

// 调用 special::cephes::fdtrc 函数，计算 F 分布的补累积分布函数
double cephes_fdtrc(double a, double b, double x) { return special::cephes::fdtrc(a, b, x); }

// 调用 special::cephes::fdtri 函数，计算 F 分布的逆累积分布函数
double cephes_fdtri(double a, double b, double y) { return special::cephes::fdtri(a, b, y); }

// 调用 special::cephes::gdtr 函数，计算 gamma 分布的累积分布函数
double cephes_gdtr(double a, double b, double x) { return special::cephes::gdtr(a, b, x); }

// 调用 special::cephes::gdtrc 函数，计算 gamma 分布的补累积分布函数
double cephes_gdtrc(double a, double b, double x) { return special::cephes::gdtrc(a, b, x); }

// 调用 special::cephes::owens_t 函数，计算 Owen's T 函数
double cephes_owens_t(double h, double a) { return special::cephes::owens_t(h, a); }

// 调用 special::cephes::nbdtr 函数，计算负二项分布的累积分布函数
double cephes_nbdtr(int k, int n, double p) { return special::cephes::nbdtr(k, n, p); }

// 调用 special::cephes::nbdtrc 函数，计算负二项分布的补累积分布函数
double cephes_nbdtrc(int k, int n, double p) { return special::cephes::nbdtrc(k, n, p); }

// 调用 special::cephes::nbdtri 函数，计算负二项分布的逆累积分布函数
double cephes_nbdtri(int k, int n, double p) { return special::cephes::nbdtri(k, n, p); }

// 调用 special::cephes::pdtr 函数，计算 Poisson 分布的累积分布函数
double cephes_pdtr(double k, double m) { return special::cephes::pdtr(k, m); }

// 调用 special::cephes::pdtrc 函数，计算 Poisson 分布的补累积分布函数
double cephes_pdtrc(double k, double m) { return special::cephes::pdtrc(k, m); }

// 调用 special::cephes::pdtri 函数，计算 Poisson 分布的逆累积分布函数
double cephes_pdtri(int k, double y) { return special::cephes::pdtri(k, y); }

// 调用 special::cephes::round 函数，对 x 进行四舍五入
double cephes_round(double x) { return special::cephes::round(x); }

// 调用 special::cephes::spence 函数，计算斯宾塞函数
double cephes_spence(double x) { return special::cephes::spence(x); }

// 调用 special::cephes::tukeylambdacdf 函数，计算 Tukey lambda 分布的累积分布函数
double cephes_tukeylambdacdf(double x, double lmbda) { return special::cephes::tukeylambdacdf(x, lmbda); }

// 调用 special::cephes::struve_h 函数，计算斯特鲁夫函数 H
double cephes_struve_h(double v, double z) { return special::cephes::struve_h(v, z); }

// 调用 special::cephes::struve_l 函数，计算斯特鲁夫函数 L
double cephes_struve_l(double v, double z) { return special::cephes::struve_l(v, z); }
```