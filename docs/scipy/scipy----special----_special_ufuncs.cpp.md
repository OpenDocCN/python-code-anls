# `D:\src\scipysrc\scipy\scipy\special\_special_ufuncs.cpp`

```
// 引入自定义头文件 "ufunc.h"
#include "ufunc.h"

// 引入数学库的头文件
#include <cmath>
#include <complex>

// 引入 SciPy 中特殊函数模块的错误处理头文件
#include "sf_error.h"

// 引入 SciPy 中特殊函数模块的各种特殊函数的头文件
#include "special.h"
#include "special/airy.h"
#include "special/amos.h"
#include "special/bessel.h"
#include "special/binom.h"
#include "special/digamma.h"
#include "special/expint.h"
#include "special/fresnel.h"
#include "special/gamma.h"
#include "special/hyp2f1.h"
#include "special/iv_ratio.h"
#include "special/kelvin.h"
#include "special/lambertw.h"
#include "special/legendre.h"
#include "special/log_exp.h"
#include "special/mathieu.h"
#include "special/par_cyl.h"
#include "special/specfun.h"
#include "special/sph_bessel.h"
#include "special/sph_harm.h"
#include "special/sphd_wave.h"
#include "special/struve.h"
#include "special/trig.h"
#include "special/wright_bessel.h"
#include "special/zeta.h"

// 扩展模块用于 NumPy 的 ufuncs，位于 SciPy 的 special 模块中。
// 要创建这样的 ufunc，请调用 "SpecFun_NewUFunc" 并使用核函数的列表作为参数，这些核函数将成为 ufunc 的重载。
// 下面的代码中有许多示例。每个 ufunc 的文档都保存在名为 _special_ufuncs_docs.cpp 的配套文件中。

using namespace std;

// 基于 NumPy 的 dtype 类型码和函数如 PyUFunc_dd_d 的基础上
// 以及以下修饰符
// p 表示指针
// r 表示引用
// c 表示常量
// v 表示易失性（volatile）

// 定义 float 类型的函数指针类型
using func_f_f_t = float (*)(float);
// 定义 double 类型的函数指针类型
using func_d_d_t = double (*)(double);
// 定义 complex<float> 类型的函数指针类型
using func_F_F_t = complex<float> (*)(complex<float>);
// 定义 complex<double> 类型的函数指针类型
using func_D_D_t = complex<double> (*)(complex<double>);

// 定义 float 类型的函数指针类型，接受两个 float 参数
using func_f_ff_t = void (*)(float, float &, float &);
// 定义 double 类型的函数指针类型，接受两个 double 参数
using func_d_dd_t = void (*)(double, double &, double &);
// 定义 float 类型的函数指针类型，接受一个 float 参数和一个 complex<float> 引用参数
using func_f_FF_t = void (*)(float, complex<float> &, complex<float> &);
// 定义 double 类型的函数指针类型，接受一个 double 参数和一个 complex<double> 引用参数
using func_d_DD_t = void (*)(double, complex<double> &, complex<double> &);

// 定义 float 类型的函数指针类型，接受四个 float 参数
using func_f_ffff_t = void (*)(float, float &, float &, float &, float &);
// 定义 double 类型的函数指针类型，接受四个 double 参数
using func_d_dddd_t = void (*)(double, double &, double &, double &, double &);

// 定义 float 类型的函数指针类型，接受一个 float 参数和四个 complex<float> 引用参数
using func_f_FFFF_t = void (*)(float, complex<float> &, complex<float> &, complex<float> &, complex<float> &);
// 定义 double 类型的函数指针类型，接受一个 double 参数和四个 complex<double> 引用参数
using func_d_DDDD_t = void (*)(double, complex<double> &, complex<double> &, complex<double> &, complex<double> &);
// 定义 complex<float> 类型的函数指针类型，接受一个 complex<float> 参数和四个 complex<float> 引用参数
using func_F_FFFF_t = void (*)(complex<float>, complex<float> &, complex<float> &, complex<float> &, complex<float> &);
// 定义 complex<double> 类型的函数指针类型，接受一个 complex<double> 参数和四个 complex<double> 引用参数
using func_D_DDDD_t =
    void (*)(complex<double>, complex<double> &, complex<double> &, complex<double> &, complex<double> &);

// 定义 float 类型的函数指针类型，接受两个 float 参数并返回一个 float 结果
using func_ff_f_t = float (*)(float, float);
// 定义 double 类型的函数指针类型，接受两个 double 参数并返回一个 double 结果
using func_dd_d_t = double (*)(double, double);
// 定义 complex<float> 类型的函数指针类型，接受两个 complex<float> 参数并返回一个 complex<float> 结果
using func_FF_F_t = complex<float> (*)(complex<float>, complex<float>);
// 定义 complex<double> 类型的函数指针类型，接受两个 complex<double> 参数并返回一个 complex<double> 结果
using func_DD_D_t = complex<double> (*)(complex<double>, complex<double>);
// 定义 complex<float> 类型的函数指针类型，接受一个 float 参数和一个 complex<float> 参数并返回一个 complex<float> 结果
using func_fF_F_t = complex<float> (*)(float, complex<float>);
using func_dD_D_t = complex<double> (*)(double, complex<double>);
// 定义 func_dD_D_t 类型，表示一个函数指针，指向返回类型为 complex<double>，参数为 double 和 complex<double> 的函数

using func_lf_f_t = float (*)(long, float);
// 定义 func_lf_f_t 类型，表示一个函数指针，指向返回类型为 float，参数为 long 和 float 的函数

using func_ld_d_t = double (*)(long, double);
// 定义 func_ld_d_t 类型，表示一个函数指针，指向返回类型为 double，参数为 long 和 double 的函数

using func_lF_F_t = complex<float> (*)(long, complex<float>);
// 定义 func_lF_F_t 类型，表示一个函数指针，指向返回类型为 complex<float>，参数为 long 和 complex<float> 的函数

using func_lD_D_t = complex<double> (*)(long, complex<double>);
// 定义 func_lD_D_t 类型，表示一个函数指针，指向返回类型为 complex<double>，参数为 long 和 complex<double> 的函数

using func_ff_ff_t = void (*)(float, float, float &, float &);
// 定义 func_ff_ff_t 类型，表示一个函数指针，指向返回类型为 void，参数为 float, float, float &, float & 的函数

using func_dd_dd_t = void (*)(double, double, double &, double &);
// 定义 func_dd_dd_t 类型，表示一个函数指针，指向返回类型为 void，参数为 double, double, double &, double & 的函数

using func_fff_f_t = float (*)(float, float, float);
// 定义 func_fff_f_t 类型，表示一个函数指针，指向返回类型为 float，参数为 float, float, float 的函数

using func_ddd_d_t = double (*)(double, double, double);
// 定义 func_ddd_d_t 类型，表示一个函数指针，指向返回类型为 double，参数为 double, double, double 的函数

using func_Flf_F_t = complex<float> (*)(complex<float>, long, float);
// 定义 func_Flf_F_t 类型，表示一个函数指针，指向返回类型为 complex<float>，参数为 complex<float>, long, float 的函数

using func_Dld_D_t = complex<double> (*)(complex<double>, long, double);
// 定义 func_Dld_D_t 类型，表示一个函数指针，指向返回类型为 complex<double>，参数为 complex<double>, long, double 的函数

using func_fff_ff_t = void (*)(float, float, float, float &, float &);
// 定义 func_fff_ff_t 类型，表示一个函数指针，指向返回类型为 void，参数为 float, float, float, float &, float & 的函数

using func_ddd_dd_t = void (*)(double, double, double, double &, double &);
// 定义 func_ddd_dd_t 类型，表示一个函数指针，指向返回类型为 void，参数为 double, double, double, double &, double & 的函数

using func_llff_F_t = complex<float> (*)(long, long, float, float);
// 定义 func_llff_F_t 类型，表示一个函数指针，指向返回类型为 complex<float>，参数为 long, long, float, float 的函数

using func_lldd_D_t = complex<double> (*)(long, long, double, double);
// 定义 func_lldd_D_t 类型，表示一个函数指针，指向返回类型为 complex<double>，参数为 long, long, double, double 的函数

using func_ffff_f_t = float (*)(float, float, float, float);
// 定义 func_ffff_f_t 类型，表示一个函数指针，指向返回类型为 float，参数为 float, float, float, float 的函数

using func_dddd_d_t = double (*)(double, double, double, double);
// 定义 func_dddd_d_t 类型，表示一个函数指针，指向返回类型为 double，参数为 double, double, double, double 的函数

using func_fffF_F_t = complex<float> (*)(float, float, float, complex<float>);
// 定义 func_fffF_F_t 类型，表示一个函数指针，指向返回类型为 complex<float>，参数为 float, float, float, complex<float> 的函数

using func_dddD_D_t = complex<double> (*)(double, double, double, complex<double>);
// 定义 func_dddD_D_t 类型，表示一个函数指针，指向返回类型为 complex<double>，参数为 double, double, double, complex<double> 的函数

using func_ffff_F_t = complex<float> (*)(float, float, float, float);
// 定义 func_ffff_F_t 类型，表示一个函数指针，指向返回类型为 complex<float>，参数为 float, float, float, float 的函数

using func_dddd_D_t = complex<double> (*)(double, double, double, double);
// 定义 func_dddd_D_t 类型，表示一个函数指针，指向返回类型为 complex<double>，参数为 double, double, double, double 的函数

using func_ffff_ff_t = void (*)(float, float, float, float, float &, float &);
// 定义 func_ffff_ff_t 类型，表示一个函数指针，指向返回类型为 void，参数为 float, float, float, float, float &, float & 的函数

using func_dddd_dd_t = void (*)(double, double, double, double, double &, double &);
// 定义 func_dddd_dd_t 类型，表示一个函数指针，指向返回类型为 void，参数为 double, double, double, double, double &, double & 的函数

using func_fffff_ff_t = void (*)(float, float, float, float, float, float &, float &);
// 定义 func_fffff_ff_t 类型，表示一个函数指针，指向返回类型为 void，参数为 float, float, float, float, float, float &, float & 的函数

using func_ddddd_dd_t = void (*)(double, double, double, double, double, double &, double &);
// 定义 func_ddddd_dd_t 类型，表示一个函数指针，指向返回类型为 void，参数为 double, double, double, double, double, double &, double & 的函数

#if (NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE)
using func_g_g_t = double (*)(double);
// 如果 long double 和 double 大小相等，则定义 func_g_g_t 类型为指向返回类型为 double，参数为 double 的函数指针
using func_gg_g_t = double (*)(double);
// 如果 long double 和 double 大小相等，则定义 func_gg_g_t 类型为指向返回类型为 double，参数为 double 的函数指针
#else
using func_g_g_t = long double (*)(long double);
// 如果 long double 和 double 大小不相等，则定义 func_g_g_t 类型为指向返回类型为 long double，参数为 long double 的函数指针
using func_gg_g_t = long double (*)(long double);
// 如果 long double 和 double 大小不相等，则定义 func_gg_g_t 类型为指向返回类型为 long double，参数为 long double 的函数指针
#endif

extern const char *_cospi_doc;
// 声明一个指向 const char 的外部变量 _cospi_doc

extern const char *_sinpi_doc;
// 声明一个指向 const char 的外部变量 _sinpi_doc

extern const char *airy_doc;
// 声明一个指向 const char 的外部变量 airy_doc

extern const char *airye_doc;
// 声明一个指向 const char 的外部变量 airye_doc

extern const char *bei_doc;
// 声明一个指向 const char 的外部变量 bei_doc

extern const char *beip_doc;
// 声明一个指向 const char 的外部变量 beip_doc

extern const char *ber_doc;
// 声明一个指向 const char 的外部变量 ber_doc

extern const char *berp_doc;
// 声明一个指向 const char 的外部变量 berp_doc

extern const char *binom_doc;
// 声明一个指向 const char 的外部变量 binom_doc

extern const char *exp1_doc;
// 声明一个指向 const char 的外部变量 exp1_doc

extern const char *expi_doc;
// 声明一个指向 const char 的外部变量 expi_doc

extern const char *expit_doc;
// 声明一个指向 const char 的外部变量 expit_doc

extern const char *exprel_doc;
// 声明一个指向 const char 的外部变量 exprel_doc

extern const char *gamma_doc;
// 声明一个指向 const char 的外部变量 gamma_doc

extern const char *gammaln_doc;
// 声明
// 声明外部常量指针，这些指针指向不同函数的文档字符串
extern const char *kei_doc;
extern const char *keip_doc;
extern const char *kelvin_doc;
extern const char *ker_doc;
extern const char *kerp_doc;
extern const char *kv_doc;
extern const char *kve_doc;
extern const char *lambertw_doc;
extern const char *logit_doc;
extern const char *loggamma_doc;
extern const char *log_expit_doc;
extern const char *log_wright_bessel_doc;
extern const char *mathieu_a_doc;
extern const char *mathieu_b_doc;
extern const char *mathieu_cem_doc;
extern const char *mathieu_modcem1_doc;
extern const char *mathieu_modcem2_doc;
extern const char *mathieu_modsem1_doc;
extern const char *mathieu_modsem2_doc;
extern const char *mathieu_sem_doc;
extern const char *modfresnelm_doc;
extern const char *modfresnelp_doc;
extern const char *obl_ang1_doc;
extern const char *obl_ang1_cv_doc;
extern const char *obl_cv_doc;
extern const char *obl_rad1_doc;
extern const char *obl_rad1_cv_doc;
extern const char *obl_rad2_doc;
extern const char *obl_rad2_cv_doc;
extern const char *_zeta_doc;
extern const char *pbdv_doc;
extern const char *pbvv_doc;
extern const char *pbwa_doc;
extern const char *pro_ang1_doc;
extern const char *pro_ang1_cv_doc;
extern const char *pro_cv_doc;
extern const char *pro_rad1_doc;
extern const char *pro_rad1_cv_doc;
extern const char *pro_rad2_doc;
extern const char *pro_rad2_cv_doc;
extern const char *psi_doc;
extern const char *rgamma_doc;
extern const char *scaled_exp1_doc;
extern const char *spherical_jn_doc;
extern const char *spherical_jn_d_doc;
extern const char *spherical_yn_doc;
extern const char *spherical_yn_d_doc;
extern const char *spherical_in_doc;
extern const char *spherical_in_d_doc;
extern const char *spherical_kn_doc;
extern const char *spherical_kn_d_doc;
extern const char *sph_harm_doc;
extern const char *wright_bessel_doc;
extern const char *yv_doc;
extern const char *yve_doc;

// 此函数在 Cython 的 "_ufuncs_extra_code_common.pxi" 中为 "_generate_pyx.py" 定义，用于在 PyUFunc_API 数组初始化的上下文中调用 PyUFunc_getfperr。
// 在这里，我们已经处于这样的上下文中。
extern "C" int wrap_PyUFunc_getfperr() { return PyUFunc_getfperr(); }

// 定义一个静态的 PyModuleDef 结构体 _special_ufuncs_def
static PyModuleDef _special_ufuncs_def = {
    PyModuleDef_HEAD_INIT,   // 模块定义头部初始化
    "_special_ufuncs",       // 模块名
    NULL,                    // 模块文档字符串
    -1,                      // 模块状态（不使用）
    NULL,                    // 模块方法
    NULL,                    // 模块加载器
    NULL,                    // 模块卸载器
    NULL,                    // 模块的全局状态
    NULL                     // 自定义数据
};

// PyInit__special_ufuncs 函数，模块初始化函数
PyMODINIT_FUNC PyInit__special_ufuncs() {
    import_array();   // 导入 NumPy 数组接口
    import_umath();   // 导入 NumPy 中的数学函数
    if (PyErr_Occurred()) {
        return NULL;  // 如果发生异常，返回 NULL
    }

    // 创建 _special_ufuncs 模块对象
    PyObject *_special_ufuncs = PyModule_Create(&_special_ufuncs_def);
    if (_special_ufuncs == nullptr) {
        return NULL;  // 如果创建失败，返回 NULL
    }

    // 创建 _cospi 对象，使用 SpecFun_NewUFunc 函数创建
    PyObject *_cospi = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::cospi), static_cast<func_d_d_t>(special::cospi),
         static_cast<func_F_F_t>(special::cospi), static_cast<func_D_D_t>(special::cospi)},
        "_cospi", _cospi_doc  // 使用 _cospi_doc 作为文档字符串
    );

    // 将 _cospi 对象添加到 _special_ufuncs 模块中
    PyModule_AddObjectRef(_special_ufuncs, "_cospi", _cospi);
    // 创建名为 _lambertw 的特殊函数对象，使用 lambertw 函数处理双精度和单精度输入
    PyObject *_lambertw = SpecFun_NewUFunc(
        {static_cast<func_Dld_D_t>(special::lambertw), static_cast<func_Flf_F_t>(special::lambertw)}, "_lambertw",
        lambertw_doc
    );
    // 将 _lambertw 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_lambertw", _lambertw);

    // 创建名为 _scaled_exp1 的特殊函数对象，使用 scaled_exp1 函数处理双精度和单精度输入
    PyObject *_scaled_exp1 = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::scaled_exp1), static_cast<func_f_f_t>(special::scaled_exp1)}, "_scaled_exp1",
        scaled_exp1_doc
    );
    // 将 _scaled_exp1 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_scaled_exp1", _scaled_exp1);

    // 创建名为 _sinpi 的特殊函数对象，使用 sinpi 函数处理不同精度的浮点数输入
    PyObject *_sinpi = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::sinpi), static_cast<func_d_d_t>(special::sinpi),
         static_cast<func_F_F_t>(special::sinpi), static_cast<func_D_D_t>(special::sinpi)},
        "_sinpi", _sinpi_doc
    );
    // 将 _sinpi 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_sinpi", _sinpi);

    // 创建名为 _zeta 的特殊函数对象，使用 zeta 函数处理双精度和单精度输入
    PyObject *_zeta = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::zeta), static_cast<func_dd_d_t>(special::zeta)}, "_zeta", _zeta_doc
    );
    // 将 _zeta 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_zeta", _zeta);

    // 创建名为 airy 的特殊函数对象，使用 airy 函数处理四个不同精度的输入
    PyObject *airy = SpecFun_NewUFunc(
        {static_cast<func_f_ffff_t>(special::airy), static_cast<func_d_dddd_t>(special::airy),
         static_cast<func_F_FFFF_t>(special::airy), static_cast<func_D_DDDD_t>(special::airy)},
        4, "airy", airy_doc
    );
    // 将 airy 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "airy", airy);

    // 创建名为 airye 的特殊函数对象，使用 airye 函数处理四个不同精度的输入
    PyObject *airye = SpecFun_NewUFunc(
        {static_cast<func_f_ffff_t>(special::airye), static_cast<func_d_dddd_t>(special::airye),
         static_cast<func_F_FFFF_t>(special::airye), static_cast<func_D_DDDD_t>(special::airye)},
        4, "airye", airye_doc
    );
    // 将 airye 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "airye", airye);

    // 创建名为 bei 的特殊函数对象，使用 bei 函数处理双精度和单精度输入
    PyObject *bei = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::bei), static_cast<func_d_d_t>(special::bei)}, "bei", bei_doc
    );
    // 将 bei 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "bei", bei);

    // 创建名为 beip 的特殊函数对象，使用 beip 函数处理双精度和单精度输入
    PyObject *beip = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::beip), static_cast<func_d_d_t>(special::beip)}, "beip", beip_doc
    );
    // 将 beip 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "beip", beip);

    // 创建名为 ber 的特殊函数对象，使用 ber 函数处理双精度和单精度输入
    PyObject *ber = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::ber), static_cast<func_d_d_t>(special::ber)}, "ber", ber_doc
    );
    // 将 ber 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "ber", ber);

    // 创建名为 berp 的特殊函数对象，使用 berp 函数处理双精度和单精度输入
    PyObject *berp = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::berp), static_cast<func_d_d_t>(special::berp)}, "berp", berp_doc
    );
    // 将 berp 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "berp", berp);

    // 创建名为 binom 的特殊函数对象，使用 binom 函数处理双精度和单精度输入
    PyObject *binom = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::binom), static_cast<func_dd_d_t>(special::binom)}, "binom", binom_doc
    );
    // 将 binom 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "binom", binom);
    // 创建 exp1 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *exp1 = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::exp1), static_cast<func_d_d_t>(special::exp1),
         static_cast<func_F_F_t>(special::exp1), static_cast<func_D_D_t>(special::exp1)},
        "exp1", exp1_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "exp1", exp1);

    // 创建 expi 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *expi = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::expi), static_cast<func_d_d_t>(special::expi),
         static_cast<func_F_F_t>(special::expi), static_cast<func_D_D_t>(special::expi)},
        "expi", expi_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "expi", expi);

    // 创建 expit 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *expit = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::expit), static_cast<func_f_f_t>(special::expit),
         static_cast<func_g_g_t>(special::expit)},
        "expit", expit_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "expit", expit);

    // 创建 exprel 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *exprel = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::exprel), static_cast<func_f_f_t>(special::exprel)}, "exprel", exprel_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "exprel", exprel);

    // 创建 gamma 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *gamma = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::gamma), static_cast<func_D_D_t>(special::gamma),
         static_cast<func_f_f_t>(special::gamma), static_cast<func_F_F_t>(special::gamma)},
        "gamma", gamma_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "gamma", gamma);

    // 创建 gammaln 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *gammaln = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::gammaln), static_cast<func_d_d_t>(special::gammaln)}, "gammaln", gammaln_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "gammaln", gammaln);

    // 创建 hyp2f1 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *hyp2f1 = SpecFun_NewUFunc(
        {static_cast<func_dddd_d_t>(special::hyp2f1), static_cast<func_dddD_D_t>(special::hyp2f1),
         static_cast<func_ffff_f_t>(special::hyp2f1), static_cast<func_fffF_F_t>(special::hyp2f1)},
        "hyp2f1", hyp2f1_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "hyp2f1", hyp2f1);

    // 创建 hankel1 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *hankel1 = SpecFun_NewUFunc(
        {static_cast<func_fF_F_t>(special::cyl_hankel_1), static_cast<func_dD_D_t>(special::cyl_hankel_1)}, "hankel1",
        hankel1_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "hankel1", hankel1);

    // 创建 hankel1e 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *hankel1e = SpecFun_NewUFunc(
        {static_cast<func_fF_F_t>(special::cyl_hankel_1e), static_cast<func_dD_D_t>(special::cyl_hankel_1e)},
        "hankel1e", hankel1e_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "hankel1e", hankel1e);

    // 创建 hankel2 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *hankel2 = SpecFun_NewUFunc(
        {static_cast<func_fF_F_t>(special::cyl_hankel_2), static_cast<func_dD_D_t>(special::cyl_hankel_2)}, "hankel2",
        hankel2_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "hankel2", hankel2);

    // 创建 hankel2e 函数对象，并注册到 _special_ufuncs 模块中
    PyObject *hankel2e = SpecFun_NewUFunc(
        {static_cast<func_fF_F_t>(special::cyl_hankel_2e), static_cast<func_dD_D_t>(special::cyl_hankel_2e)},
        "hankel2e", hankel2e_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "hankel2e", hankel2e);

# 向特殊模块中添加名为 "hankel2e" 的对象引用，该对象是 hankel2e 函数的指针。


    PyObject *it2i0k0 = SpecFun_NewUFunc(
        {static_cast<func_f_ff_t>(special::it2i0k0), static_cast<func_d_dd_t>(special::it2i0k0)}, 2, "it2i0k0",
        it2i0k0_doc
    );

# 创建一个新的通用函数对象 it2i0k0，使用 special 模块中的 it2i0k0 函数作为其实现，支持单精度和双精度输入，参数个数为 2，名称为 "it2i0k0"，并使用提供的文档字符串 it2i0k0_doc。


    PyModule_AddObjectRef(_special_ufuncs, "it2i0k0", it2i0k0);

# 向特殊模块中添加名为 "it2i0k0" 的对象引用，该对象指向之前创建的 it2i0k0 通用函数对象。


    PyObject *it2j0y0 = SpecFun_NewUFunc(
        {static_cast<func_f_ff_t>(special::it2j0y0), static_cast<func_d_dd_t>(special::it2j0y0)}, 2, "it2j0y0",
        it2j0y0_doc
    );

# 创建一个新的通用函数对象 it2j0y0，使用 special 模块中的 it2j0y0 函数作为其实现，支持单精度和双精度输入，参数个数为 2，名称为 "it2j0y0"，并使用提供的文档字符串 it2j0y0_doc。


    PyModule_AddObjectRef(_special_ufuncs, "it2j0y0", it2j0y0);

# 向特殊模块中添加名为 "it2j0y0" 的对象引用，该对象指向之前创建的 it2j0y0 通用函数对象。


    PyObject *it2struve0 = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::it2struve0), static_cast<func_d_d_t>(special::it2struve0)}, "it2struve0",
        it2struve0_doc
    );

# 创建一个新的通用函数对象 it2struve0，使用 special 模块中的 it2struve0 函数作为其实现，支持单精度和双精度输入，名称为 "it2struve0"，并使用提供的文档字符串 it2struve0_doc。


    PyModule_AddObjectRef(_special_ufuncs, "it2struve0", it2struve0);

# 向特殊模块中添加名为 "it2struve0" 的对象引用，该对象指向之前创建的 it2struve0 通用函数对象。


    PyObject *itairy = SpecFun_NewUFunc(
        {static_cast<func_f_ffff_t>(special::itairy), static_cast<func_d_dddd_t>(special::itairy)}, 4, "itairy",
        itairy_doc
    );

# 创建一个新的通用函数对象 itairy，使用 special 模块中的 itairy 函数作为其实现，支持单精度和双精度输入，参数个数为 4，名称为 "itairy"，并使用提供的文档字符串 itairy_doc。


    PyModule_AddObjectRef(_special_ufuncs, "itairy", itairy);

# 向特殊模块中添加名为 "itairy" 的对象引用，该对象指向之前创建的 itairy 通用函数对象。


    PyObject *iti0k0 = SpecFun_NewUFunc(
        {static_cast<func_f_ff_t>(special::it1i0k0), static_cast<func_d_dd_t>(special::it1i0k0)}, 2, "iti0k0",
        iti0k0_doc
    );

# 创建一个新的通用函数对象 iti0k0，使用 special 模块中的 it1i0k0 函数作为其实现，支持单精度和双精度输入，参数个数为 2，名称为 "iti0k0"，并使用提供的文档字符串 iti0k0_doc。


    PyModule_AddObjectRef(_special_ufuncs, "iti0k0", iti0k0);

# 向特殊模块中添加名为 "iti0k0" 的对象引用，该对象指向之前创建的 iti0k0 通用函数对象。


    PyObject *itj0y0 = SpecFun_NewUFunc(
        {static_cast<func_f_ff_t>(special::it1j0y0), static_cast<func_d_dd_t>(special::it1j0y0)}, 2, "itj0y0",
        itj0y0_doc
    );

# 创建一个新的通用函数对象 itj0y0，使用 special 模块中的 it1j0y0 函数作为其实现，支持单精度和双精度输入，参数个数为 2，名称为 "itj0y0"，并使用提供的文档字符串 itj0y0_doc。


    PyModule_AddObjectRef(_special_ufuncs, "itj0y0", itj0y0);

# 向特殊模块中添加名为 "itj0y0" 的对象引用，该对象指向之前创建的 itj0y0 通用函数对象。


    PyObject *itmodstruve0 = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::itmodstruve0), static_cast<func_d_d_t>(special::itmodstruve0)},
        "itmodstruve0", itmodstruve0_doc
    );

# 创建一个新的通用函数对象 itmodstruve0，使用 special 模块中的 itmodstruve0 函数作为其实现，支持单精度和双精度输入，名称为 "itmodstruve0"，并使用提供的文档字符串 itmodstruve0_doc。


    PyModule_AddObjectRef(_special_ufuncs, "itmodstruve0", itmodstruve0);

# 向特殊模块中添加名为 "itmodstruve0" 的对象引用，该对象指向之前创建的 itmodstruve0 通用函数对象。


    PyObject *itstruve0 = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::itstruve0), static_cast<func_d_d_t>(special::itstruve0)}, "itstruve0",
        itstruve0_doc
    );

# 创建一个新的通用函数对象 itstruve0，使用 special 模块中的 itstruve0 函数作为其实现，支持单精度和双精度输入，名称为 "itstruve0"，并使用提供的文档字符串 itstruve0_doc。


    PyModule_AddObjectRef(_special_ufuncs, "itstruve0", itstruve0);

# 向特殊模块中添加名为 "itstruve0" 的对象引用，该对象指向之前创建的 itstruve0 通用函数对象。


    PyObject *iv = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::cyl_bessel_i), static_cast<func_dd_d_t>(special::cyl_bessel_i),
         static_cast<func_fF_F_t>(special::cyl_bessel_i), static_cast<func_dD_D_t>(special::cyl_bessel_i)},
        "iv", iv_doc
    );

# 创建一个新的通用函数对象 iv，使用 special 模块中的 cyl_bessel_i 函数作为其实现，支持多种输入类型和参数个数，名称为 "iv"，并使用提供的文档字符串 iv_doc。


    PyModule_AddObjectRef(_special_ufuncs, "iv", iv);

# 向特殊模块中添加名为 "iv
    // 将自定义特殊函数对象 'ive' 添加到 Python 模块 '_special_ufuncs' 中
    PyModule_AddObjectRef(_special_ufuncs, "ive", ive);

    // 创建并注册新的特殊函数对象 'jv' 到 Python 模块 '_special_ufuncs' 中
    PyObject *jv = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::cyl_bessel_j), static_cast<func_dd_d_t>(special::cyl_bessel_j),
         static_cast<func_fF_F_t>(special::cyl_bessel_j), static_cast<func_dD_D_t>(special::cyl_bessel_j)},
        "jv", jv_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "jv", jv);

    // 创建并注册新的特殊函数对象 'jve' 到 Python 模块 '_special_ufuncs' 中
    PyObject *jve = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::cyl_bessel_je), static_cast<func_dd_d_t>(special::cyl_bessel_je),
         static_cast<func_fF_F_t>(special::cyl_bessel_je), static_cast<func_dD_D_t>(special::cyl_bessel_je)},
        "jve", jve_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "jve", jve);

    // 创建并注册新的特殊函数对象 'kei' 到 Python 模块 '_special_ufuncs' 中
    PyObject *kei = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::kei), static_cast<func_d_d_t>(special::kei)}, "kei", kei_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "kei", kei);

    // 创建并注册新的特殊函数对象 'keip' 到 Python 模块 '_special_ufuncs' 中
    PyObject *keip = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::keip), static_cast<func_d_d_t>(special::keip)}, "keip", keip_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "keip", keip);

    // 创建并注册新的特殊函数对象 'kelvin' 到 Python 模块 '_special_ufuncs' 中
    PyObject *kelvin = SpecFun_NewUFunc(
        {static_cast<func_f_FFFF_t>(special::kelvin), static_cast<func_d_DDDD_t>(special::kelvin)}, 4, "kelvin",
        kelvin_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "kelvin", kelvin);

    // 创建并注册新的特殊函数对象 'ker' 到 Python 模块 '_special_ufuncs' 中
    PyObject *ker = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::ker), static_cast<func_d_d_t>(special::ker)}, "ker", ker_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "ker", ker);

    // 创建并注册新的特殊函数对象 'kerp' 到 Python 模块 '_special_ufuncs' 中
    PyObject *kerp = SpecFun_NewUFunc(
        {static_cast<func_f_f_t>(special::kerp), static_cast<func_d_d_t>(special::kerp)}, "kerp", kerp_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "kerp", kerp);

    // 创建并注册新的特殊函数对象 'kv' 到 Python 模块 '_special_ufuncs' 中
    PyObject *kv = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::cyl_bessel_k), static_cast<func_dd_d_t>(special::cyl_bessel_k),
         static_cast<func_fF_F_t>(special::cyl_bessel_k), static_cast<func_dD_D_t>(special::cyl_bessel_k)},
        "kv", kv_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "kv", kv);

    // 创建并注册新的特殊函数对象 'kve' 到 Python 模块 '_special_ufuncs' 中
    PyObject *kve = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::cyl_bessel_ke), static_cast<func_dd_d_t>(special::cyl_bessel_ke),
         static_cast<func_fF_F_t>(special::cyl_bessel_ke), static_cast<func_dD_D_t>(special::cyl_bessel_ke)},
        "kve", kve_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "kve", kve);

    // 创建并注册新的特殊函数对象 'log_expit' 到 Python 模块 '_special_ufuncs' 中
    PyObject *log_expit = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::log_expit), static_cast<func_f_f_t>(special::log_expit),
         static_cast<func_g_g_t>(special::log_expit)},
        "log_expit", log_expit_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "log_expit", log_expit);
    // 创建一个名为 log_wright_bessel 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *log_wright_bessel = SpecFun_NewUFunc(
        {static_cast<func_ddd_d_t>(special::log_wright_bessel), static_cast<func_fff_f_t>(special::log_wright_bessel)},
        "log_wright_bessel", log_wright_bessel_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "log_wright_bessel", log_wright_bessel);

    // 创建一个名为 logit 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *logit = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::logit), static_cast<func_f_f_t>(special::logit),
         static_cast<func_g_g_t>(special::logit)},
        "logit", logit_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "logit", logit);

    // 创建一个名为 loggamma 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *loggamma = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::loggamma), static_cast<func_D_D_t>(special::loggamma),
         static_cast<func_f_f_t>(special::loggamma), static_cast<func_F_F_t>(special::loggamma)},
        "loggamma", loggamma_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "loggamma", loggamma);

    // 创建一个名为 mathieu_a 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *mathieu_a = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::cem_cva), static_cast<func_dd_d_t>(special::cem_cva)}, "mathieu_a",
        mathieu_a_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "mathieu_a", mathieu_a);

    // 创建一个名为 mathieu_b 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *mathieu_b = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::sem_cva), static_cast<func_dd_d_t>(special::sem_cva)}, "mathieu_b",
        mathieu_b_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "mathieu_b", mathieu_b);

    // 创建一个名为 mathieu_cem 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *mathieu_cem = SpecFun_NewUFunc(
        {static_cast<func_fff_ff_t>(special::cem), static_cast<func_ddd_dd_t>(special::cem)}, 2, "mathieu_cem",
        mathieu_cem_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "mathieu_cem", mathieu_cem);

    // 创建一个名为 mathieu_modcem1 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *mathieu_modcem1 = SpecFun_NewUFunc(
        {static_cast<func_fff_ff_t>(special::mcm1), static_cast<func_ddd_dd_t>(special::mcm1)}, 2, "mathieu_modcem1",
        mathieu_modcem1_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "mathieu_modcem1", mathieu_modcem1);

    // 创建一个名为 mathieu_modcem2 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *mathieu_modcem2 = SpecFun_NewUFunc(
        {static_cast<func_fff_ff_t>(special::mcm2), static_cast<func_ddd_dd_t>(special::mcm2)}, 2, "mathieu_modcem2",
        mathieu_modcem2_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "mathieu_modcem2", mathieu_modcem2);

    // 创建一个名为 mathieu_modsem1 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *mathieu_modsem1 = SpecFun_NewUFunc(
        {static_cast<func_fff_ff_t>(special::msm1), static_cast<func_ddd_dd_t>(special::msm1)}, 2, "mathieu_modsem1",
        mathieu_modsem1_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "mathieu_modsem1", mathieu_modsem1);

    // 创建一个名为 mathieu_modsem2 的新的特殊函数对象，并注册到 _special_ufuncs 模块中
    PyObject *mathieu_modsem2 = SpecFun_NewUFunc(
        {static_cast<func_fff_ff_t>(special::msm2), static_cast<func_ddd_dd_t>(special::msm2)}, 2, "mathieu_modsem2",
        mathieu_modsem2_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "mathieu_modsem2", mathieu_modsem2);
    // 创建 mathieu_sem 的特殊函数对象，使用 sem 函数，定义为双参数的 UFunc
    PyObject *mathieu_sem = SpecFun_NewUFunc(
        {static_cast<func_fff_ff_t>(special::sem), static_cast<func_ddd_dd_t>(special::sem)}, 2, "mathieu_sem",
        mathieu_sem_doc
    );
    // 将 mathieu_sem 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "mathieu_sem", mathieu_sem);

    // 创建 modfresnelm 的特殊函数对象，使用 modified_fresnel_minus 函数，定义为双参数的 UFunc
    PyObject *modfresnelm = SpecFun_NewUFunc(
        {static_cast<func_f_FF_t>(special::modified_fresnel_minus),
         static_cast<func_d_DD_t>(special::modified_fresnel_minus)},
        2, "modfresnelm", modfresnelm_doc
    );
    // 将 modfresnelm 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "modfresnelm", modfresnelm);

    // 创建 modfresnelp 的特殊函数对象，使用 modified_fresnel_plus 函数，定义为双参数的 UFunc
    PyObject *modfresnelp = SpecFun_NewUFunc(
        {static_cast<func_f_FF_t>(special::modified_fresnel_plus),
         static_cast<func_d_DD_t>(special::modified_fresnel_plus)},
        2, "modfresnelp", modfresnelp_doc
    );
    // 将 modfresnelp 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "modfresnelp", modfresnelp);

    // 创建 obl_ang1 的特殊函数对象，使用 oblate_aswfa_nocv 函数，定义为双参数的 UFunc
    PyObject *obl_ang1 = SpecFun_NewUFunc(
        {static_cast<func_ffff_ff_t>(special::oblate_aswfa_nocv),
         static_cast<func_dddd_dd_t>(special::oblate_aswfa_nocv)},
        2, "obl_ang1", obl_ang1_doc
    );
    // 将 obl_ang1 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "obl_ang1", obl_ang1);

    // 创建 obl_ang1_cv 的特殊函数对象，使用 oblate_aswfa 函数，定义为双参数的 UFunc
    PyObject *obl_ang1_cv = SpecFun_NewUFunc(
        {static_cast<func_fffff_ff_t>(special::oblate_aswfa), static_cast<func_ddddd_dd_t>(special::oblate_aswfa)}, 2,
        "obl_ang1_cv", obl_ang1_cv_doc
    );
    // 将 obl_ang1_cv 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "obl_ang1_cv", obl_ang1_cv);

    // 创建 obl_cv 的特殊函数对象，使用 oblate_segv 函数，定义为单参数的 UFunc
    PyObject *obl_cv = SpecFun_NewUFunc(
        {static_cast<func_fff_f_t>(special::oblate_segv), static_cast<func_ddd_d_t>(special::oblate_segv)}, "obl_cv",
        obl_cv_doc
    );
    // 将 obl_cv 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "obl_cv", obl_cv);

    // 创建 obl_rad1 的特殊函数对象，使用 oblate_radial1_nocv 函数，定义为双参数的 UFunc
    PyObject *obl_rad1 = SpecFun_NewUFunc(
        {static_cast<func_ffff_ff_t>(special::oblate_radial1_nocv),
         static_cast<func_dddd_dd_t>(special::oblate_radial1_nocv)},
        2, "obl_rad1", obl_rad1_doc
    );
    // 将 obl_rad1 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "obl_rad1", obl_rad1);

    // 创建 obl_rad1_cv 的特殊函数对象，使用 oblate_radial1 函数，定义为双参数的 UFunc
    PyObject *obl_rad1_cv = SpecFun_NewUFunc(
        {static_cast<func_fffff_ff_t>(special::oblate_radial1), static_cast<func_ddddd_dd_t>(special::oblate_radial1)},
        2, "obl_rad1_cv", obl_rad1_cv_doc
    );
    // 将 obl_rad1_cv 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "obl_rad1_cv", obl_rad1_cv);

    // 创建 obl_rad2 的特殊函数对象，使用 oblate_radial2_nocv 函数，定义为双参数的 UFunc
    PyObject *obl_rad2 = SpecFun_NewUFunc(
        {static_cast<func_ffff_ff_t>(special::oblate_radial2_nocv),
         static_cast<func_dddd_dd_t>(special::oblate_radial2_nocv)},
        2, "obl_rad2", obl_rad2_doc
    );
    // 将 obl_rad2 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "obl_rad2", obl_rad2);

    // 创建 obl_rad2_cv 的特殊函数对象，使用 oblate_radial2 函数，定义为双参数的 UFunc
    PyObject *obl_rad2_cv = SpecFun_NewUFunc(
        {static_cast<func_fffff_ff_t>(special::oblate_radial2), static_cast<func_ddddd_dd_t>(special::oblate_radial2)},
        2, "obl_rad2_cv", obl_rad2_cv_doc
    );
    // 将 obl_rad2_cv 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "obl_rad2_cv", obl_rad2_cv);

    // 创建 pbdv 的特殊函数对象，使用 pbdv 函数，定义为双参数的 UFunc
    PyObject *pbdv = SpecFun_NewUFunc(
        {static_cast<func_ff_ff_t>(special::pbdv), static_cast<func_dd_dd_t>(special::pbdv)}, 2, "pbdv", pbdv_doc
    );
    );
    // 向特殊函数模块中添加对象引用 "pbdv"
    PyModule_AddObjectRef(_special_ufuncs, "pbdv", pbdv);

    // 创建并添加特殊函数对象 "pbvv" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pbvv = SpecFun_NewUFunc(
        {static_cast<func_ff_ff_t>(special::pbvv), static_cast<func_dd_dd_t>(special::pbvv)}, 2, "pbvv", pbvv_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pbvv", pbvv);

    // 创建并添加特殊函数对象 "pbwa" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pbwa = SpecFun_NewUFunc(
        {static_cast<func_ff_ff_t>(special::pbwa), static_cast<func_dd_dd_t>(special::pbwa)}, 2, "pbwa", pbwa_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pbwa", pbwa);

    // 创建并添加特殊函数对象 "pro_ang1" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pro_ang1 = SpecFun_NewUFunc(
        {static_cast<func_ffff_ff_t>(special::prolate_aswfa_nocv),
         static_cast<func_dddd_dd_t>(special::prolate_aswfa_nocv)},
        2, "pro_ang1", pro_ang1_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pro_ang1", pro_ang1);

    // 创建并添加特殊函数对象 "pro_ang1_cv" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pro_ang1_cv = SpecFun_NewUFunc(
        {static_cast<func_fffff_ff_t>(special::prolate_aswfa), static_cast<func_ddddd_dd_t>(special::prolate_aswfa)}, 2,
        "pro_ang1_cv", pro_ang1_cv_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pro_ang1_cv", pro_ang1_cv);

    // 创建并添加特殊函数对象 "pro_cv" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pro_cv = SpecFun_NewUFunc(
        {static_cast<func_fff_f_t>(special::prolate_segv), static_cast<func_ddd_d_t>(special::prolate_segv)}, "obl_cv",
        pro_cv_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pro_cv", pro_cv);

    // 创建并添加特殊函数对象 "pro_rad1" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pro_rad1 = SpecFun_NewUFunc(
        {static_cast<func_ffff_ff_t>(special::prolate_radial1_nocv),
         static_cast<func_dddd_dd_t>(special::prolate_radial1_nocv)},
        2, "pro_rad1", pro_rad1_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pro_rad1", pro_rad1);

    // 创建并添加特殊函数对象 "pro_rad1_cv" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pro_rad1_cv = SpecFun_NewUFunc(
        {static_cast<func_fffff_ff_t>(special::prolate_radial1), static_cast<func_ddddd_dd_t>(special::prolate_radial1)},
        2, "pro_rad1_cv", pro_rad1_cv_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pro_rad1_cv", pro_rad1_cv);

    // 创建并添加特殊函数对象 "pro_rad2" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pro_rad2 = SpecFun_NewUFunc(
        {static_cast<func_ffff_ff_t>(special::prolate_radial2_nocv),
         static_cast<func_dddd_dd_t>(special::prolate_radial2_nocv)},
        2, "pro_rad2", pro_rad2_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pro_rad2", pro_rad2);

    // 创建并添加特殊函数对象 "pro_rad2_cv" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *pro_rad2_cv = SpecFun_NewUFunc(
        {static_cast<func_fffff_ff_t>(special::prolate_radial2), static_cast<func_ddddd_dd_t>(special::prolate_radial2)},
        2, "pro_rad2_cv", pro_rad2_cv_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "pro_rad2_cv", pro_rad2_cv);

    // 创建并添加特殊函数对象 "psi" 到特殊函数模块，使用静态转换函数和文档字符串
    PyObject *psi = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::digamma), static_cast<func_D_D_t>(special::digamma),
         static_cast<func_f_f_t>(special::digamma), static_cast<func_F_F_t>(special::digamma)},
        "psi", psi_doc
    );
    PyModule_AddObjectRef(_special_ufuncs, "psi", psi);
    // 创建名为 "rgamma" 的特殊函数对象，支持不同精度的伽玛函数计算，附带文档字符串 rgamma_doc
    PyObject *rgamma = SpecFun_NewUFunc(
        {static_cast<func_d_d_t>(special::rgamma), static_cast<func_D_D_t>(special::rgamma),
         static_cast<func_f_f_t>(special::rgamma), static_cast<func_F_F_t>(special::rgamma)},
        "rgamma", rgamma_doc
    );
    // 将 "rgamma" 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "rgamma", rgamma);

    // 创建名为 "_spherical_jn" 的特殊函数对象，支持不同精度的球面贝塞尔函数计算，附带文档字符串 spherical_jn_doc
    PyObject *_spherical_jn = SpecFun_NewUFunc(
        {static_cast<func_ld_d_t>(special::sph_bessel_j), static_cast<func_lD_D_t>(special::sph_bessel_j),
         static_cast<func_lf_f_t>(special::sph_bessel_j), static_cast<func_lF_F_t>(special::sph_bessel_j)},
        "_spherical_jn", spherical_jn_doc
    );
    // 将 "_spherical_jn" 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_spherical_jn", _spherical_jn);

    // 创建名为 "_spherical_jn_d" 的特殊函数对象，支持不同精度的球面贝塞尔函数的导数计算，附带文档字符串 spherical_jn_d_doc
    PyObject *_spherical_jn_d = SpecFun_NewUFunc(
        {static_cast<func_ld_d_t>(special::sph_bessel_j_jac), static_cast<func_lD_D_t>(special::sph_bessel_j_jac),
         static_cast<func_lf_f_t>(special::sph_bessel_j_jac), static_cast<func_lF_F_t>(special::sph_bessel_j_jac)},
        "_spherical_jn_d", spherical_jn_d_doc
    );
    // 将 "_spherical_jn_d" 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_spherical_jn_d", _spherical_jn_d);

    // 创建名为 "_spherical_yn" 的特殊函数对象，支持不同精度的第二类球面贝塞尔函数计算，附带文档字符串 spherical_yn_doc
    PyObject *_spherical_yn = SpecFun_NewUFunc(
        {static_cast<func_ld_d_t>(special::sph_bessel_y), static_cast<func_lD_D_t>(special::sph_bessel_y),
         static_cast<func_lf_f_t>(special::sph_bessel_y), static_cast<func_lF_F_t>(special::sph_bessel_y)},
        "_spherical_yn", spherical_yn_doc
    );
    // 将 "_spherical_yn" 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_spherical_yn", _spherical_yn);

    // 创建名为 "_spherical_yn_d" 的特殊函数对象，支持不同精度的第二类球面贝塞尔函数的导数计算，附带文档字符串 spherical_yn_d_doc
    PyObject *_spherical_yn_d = SpecFun_NewUFunc(
        {static_cast<func_ld_d_t>(special::sph_bessel_y_jac), static_cast<func_lD_D_t>(special::sph_bessel_y_jac),
         static_cast<func_lf_f_t>(special::sph_bessel_y_jac), static_cast<func_lF_F_t>(special::sph_bessel_y_jac)},
        "_spherical_yn_d", spherical_yn_d_doc
    );
    // 将 "_spherical_yn_d" 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_spherical_yn_d", _spherical_yn_d);

    // 创建名为 "_spherical_in" 的特殊函数对象，支持不同精度的第一类球面贝塞尔函数计算，附带文档字符串 spherical_in_doc
    PyObject *_spherical_in = SpecFun_NewUFunc(
        {static_cast<func_ld_d_t>(special::sph_bessel_i), static_cast<func_lD_D_t>(special::sph_bessel_i),
         static_cast<func_lf_f_t>(special::sph_bessel_i), static_cast<func_lF_F_t>(special::sph_bessel_i)},
        "_spherical_in", spherical_in_doc
    );
    // 将 "_spherical_in" 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_spherical_in", _spherical_in);

    // 创建名为 "_spherical_in_d" 的特殊函数对象，支持不同精度的第一类球面贝塞尔函数的导数计算，附带文档字符串 spherical_in_d_doc
    PyObject *_spherical_in_d = SpecFun_NewUFunc(
        {static_cast<func_ld_d_t>(special::sph_bessel_i_jac), static_cast<func_lD_D_t>(special::sph_bessel_i_jac),
         static_cast<func_lf_f_t>(special::sph_bessel_i_jac), static_cast<func_lF_F_t>(special::sph_bessel_i_jac)},
        "_spherical_in_d", spherical_in_d_doc
    );
    // 将 "_spherical_in_d" 对象添加到 _special_ufuncs 模块中，并增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_spherical_in_d", _spherical_in_d);
    // 创建名为 _spherical_kn 的特殊函数对象，使用 sph_bessel_k 函数处理不同类型的参数
    PyObject *_spherical_kn = SpecFun_NewUFunc(
        {static_cast<func_ld_d_t>(special::sph_bessel_k), static_cast<func_lD_D_t>(special::sph_bessel_k),
         static_cast<func_lf_f_t>(special::sph_bessel_k), static_cast<func_lF_F_t>(special::sph_bessel_k)},
        "_spherical_kn", spherical_kn_doc
    );
    // 将 _spherical_kn 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_spherical_kn", _spherical_kn);

    // 创建名为 _spherical_kn_d 的特殊函数对象，使用 sph_bessel_k_jac 函数处理不同类型的参数
    PyObject *_spherical_kn_d = SpecFun_NewUFunc(
        {static_cast<func_ld_d_t>(special::sph_bessel_k_jac), static_cast<func_lD_D_t>(special::sph_bessel_k_jac),
         static_cast<func_lf_f_t>(special::sph_bessel_k_jac), static_cast<func_lF_F_t>(special::sph_bessel_k_jac)},
        "_spherical_kn_d", spherical_kn_d_doc
    );
    // 将 _spherical_kn_d 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "_spherical_kn_d", _spherical_kn_d);

    // 创建名为 sph_harm 的特殊函数对象，使用 ::sph_harm 函数处理不同类型的参数
    PyObject *sph_harm = SpecFun_NewUFunc(
        {static_cast<func_lldd_D_t>(::sph_harm), static_cast<func_dddd_D_t>(::sph_harm),
         static_cast<func_llff_F_t>(::sph_harm), static_cast<func_ffff_F_t>(::sph_harm)},
        "sph_harm", sph_harm_doc
    );
    // 将 sph_harm 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "sph_harm", sph_harm);

    // 创建名为 wright_bessel 的特殊函数对象，使用 wright_bessel 函数处理不同类型的参数
    PyObject *wright_bessel = SpecFun_NewUFunc(
        {static_cast<func_ddd_d_t>(special::wright_bessel), static_cast<func_fff_f_t>(special::wright_bessel)},
        "wright_bessel", wright_bessel_doc
    );
    // 将 wright_bessel 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "wright_bessel", wright_bessel);

    // 创建名为 yv 的特殊函数对象，使用 cyl_bessel_y 函数处理不同类型的参数
    PyObject *yv = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::cyl_bessel_y), static_cast<func_dd_d_t>(special::cyl_bessel_y),
         static_cast<func_fF_F_t>(special::cyl_bessel_y), static_cast<func_dD_D_t>(special::cyl_bessel_y)},
        "yv", yv_doc
    );
    // 将 yv 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "yv", yv);

    // 创建名为 yve 的特殊函数对象，使用 cyl_bessel_ye 函数处理不同类型的参数
    PyObject *yve = SpecFun_NewUFunc(
        {static_cast<func_ff_f_t>(special::cyl_bessel_ye), static_cast<func_dd_d_t>(special::cyl_bessel_ye),
         static_cast<func_fF_F_t>(special::cyl_bessel_ye), static_cast<func_dD_D_t>(special::cyl_bessel_ye)},
        "yve", yve_doc
    );
    // 将 yve 对象添加到 _special_ufuncs 模块中，增加其引用计数
    PyModule_AddObjectRef(_special_ufuncs, "yve", yve);

    // 返回 _special_ufuncs 模块对象，其中包含了所有特殊函数对象的引用
    return _special_ufuncs;
}



# 这行代码关闭了一个代码块。在编程中，大括号通常用于表示代码块的开始和结束。
# 这里的大括号可能是用于类、函数、循环或条件语句的结束标记之一。
# 在这种情况下，它可能是用于结束一个函数或类定义。
```