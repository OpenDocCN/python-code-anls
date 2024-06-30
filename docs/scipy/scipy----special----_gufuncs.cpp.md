# `D:\src\scipysrc\scipy\scipy\special\_gufuncs.cpp`

```
// 包含自定义头文件 "ufunc.h"
#include "ufunc.h"

// 包含特殊函数相关的头文件
#include "special.h"
#include "special/bessel.h"
#include "special/legendre.h"
#include "special/sph_harm.h"

// 使用标准命名空间 std
using namespace std;

// 定义 float 到 float 函数指针类型
using func_f_f1f1_t =
    void (*)(float, mdspan<float, dextents<ptrdiff_t, 1>, layout_stride>, mdspan<float, dextents<ptrdiff_t, 1>, layout_stride>);
// 定义 double 到 double 函数指针类型
using func_d_d1d1_t =
    void (*)(double, mdspan<double, dextents<ptrdiff_t, 1>, layout_stride>, mdspan<double, dextents<ptrdiff_t, 1>, layout_stride>);
// 定义 complex<float> 到 complex<float> 函数指针类型
using func_F_F1F1_t =
    void (*)(complex<float>, mdspan<complex<float>, dextents<ptrdiff_t, 1>, layout_stride>, mdspan<complex<float>, dextents<ptrdiff_t, 1>, layout_stride>);
// 定义 complex<double> 到 complex<double> 函数指针类型
using func_D_D1D1_t =
    void (*)(complex<double>, mdspan<complex<double>, dextents<ptrdiff_t, 1>, layout_stride>, mdspan<complex<double>, dextents<ptrdiff_t, 1>, layout_stride>);

// 定义 float 到 float 的二维函数指针类型
using func_f_f2f2_t =
    void (*)(float, mdspan<float, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<float, dextents<ptrdiff_t, 2>, layout_stride>);
// 定义 double 到 double 的二维函数指针类型
using func_d_d2d2_t =
    void (*)(double, mdspan<double, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<double, dextents<ptrdiff_t, 2>, layout_stride>);
// 定义 complex<float> 到 complex<float> 的二维函数指针类型
using func_F_F2F2_t =
    void (*)(complex<float>, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>);
// 定义 complex<double> 到 complex<double> 的二维函数指针类型
using func_D_D2D2_t =
    void (*)(complex<double>, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>);

// 定义带布尔参数的 float 到 float 的二维函数指针类型
using func_fb_f2f2_t =
    void (*)(float, bool, mdspan<float, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<float, dextents<ptrdiff_t, 2>, layout_stride>);
// 定义带布尔参数的 double 到 double 的二维函数指针类型
using func_db_d2d2_t =
    void (*)(double, bool, mdspan<double, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<double, dextents<ptrdiff_t, 2>, layout_stride>);

// 定义带长整型和布尔参数的 complex<float> 到 complex<float> 的二维函数指针类型
using func_Flb_F2F2_t =
    void (*)(complex<float>, long, bool, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>);
// 定义带长整型和布尔参数的 complex<double> 到 complex<double> 的二维函数指针类型
using func_Dlb_D2D2_t =
    void (*)(complex<double>, long, bool, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>);

// 定义 float 到 float 到复数的二维函数指针类型
using func_ff_F2_t = void (*)(float, float, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>);
// 定义 double 到 double 到复数的二维函数指针类型
using func_dd_D2_t = void (*)(double, double, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>);

// 声明外部变量，用于文档字符串
extern const char *lpn_doc;
extern const char *lpmn_doc;
extern const char *clpmn_doc;
extern const char *lqn_doc;
extern const char *lqmn_doc;
extern const char *rctj_doc;
extern const char *rcty_doc;
extern const char *sph_harm_all_doc;

// 声明外部 C 函数，用于获取浮点错误
// 它在 Cython "_ufuncs_extra_code_common.pxi" 中为 "_generate_pyx.py" 的 sf_error 定义
// 该函数用于在 PyUFunc_API 数组已初始化的情况下调用 PyUFunc_getfperr
extern "C" int wrap_PyUFunc_getfperr() { return PyUFunc_getfperr(); }
// 定义 Python 模块的基本结构和信息
static PyModuleDef _gufuncs_def = {
    PyModuleDef_HEAD_INIT,  // 使用默认的模块头初始化
    "_gufuncs",              // 模块名为 "_gufuncs"
    NULL,                    // 模块文档字符串为 NULL
    -1,                      // 全局状态为 -1
    NULL,                    // 模块方法集合为空
    NULL,                    // 模块的自定义槽为空
    NULL,                    // 模块的清理函数为空
    NULL,                    // 模块的自定义内存分配器为空
    NULL                     // 模块的状态为空
};

// Python 模块的初始化函数，用于创建并返回一个新的模块对象
PyMODINIT_FUNC PyInit__gufuncs() {
    // 导入 NumPy 的数组处理模块
    import_array();
    // 导入 NumPy 的数学函数模块
    import_umath();
    
    // 检查是否有 Python 异常发生，如果有则返回空指针
    if (PyErr_Occurred()) {
        return NULL;
    }

    // 创建名为 "_gufuncs" 的 Python 模块对象，如果创建失败则返回空指针
    PyObject *_gufuncs = PyModule_Create(&_gufuncs_def);
    if (_gufuncs == nullptr) {
        return NULL;
    }

    // 创建并添加名为 "_lpn" 的通用函数对象到模块 "_gufuncs"
    PyObject *_lpn = SpecFun_NewGUFunc(
        {static_cast<func_f_f1f1_t>(::lpn), static_cast<func_d_d1d1_t>(::lpn), static_cast<func_F_F1F1_t>(::lpn),
         static_cast<func_D_D1D1_t>(::lpn)},
        2, "_lpn", lpn_doc, "()->(np1),(np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_lpn", _lpn);

    // 创建并添加名为 "_lpmn" 的通用函数对象到模块 "_gufuncs"
    PyObject *_lpmn = SpecFun_NewGUFunc(
        {static_cast<func_fb_f2f2_t>(::lpmn), static_cast<func_db_d2d2_t>(::lpmn)}, 2, "_lpmn", lpmn_doc,
        "(),()->(mp1,np1),(mp1,np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_lpmn", _lpmn);

    // 创建并添加名为 "_clpmn" 的通用函数对象到模块 "_gufuncs"
    PyObject *_clpmn = SpecFun_NewGUFunc(
        {static_cast<func_Flb_F2F2_t>(special::clpmn), static_cast<func_Dlb_D2D2_t>(special::clpmn)}, 2, "_clpmn",
        clpmn_doc, "(),(),()->(mp1,np1),(mp1,np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_clpmn", _clpmn);

    // 创建并添加名为 "_lqn" 的通用函数对象到模块 "_gufuncs"
    PyObject *_lqn = SpecFun_NewGUFunc(
        {static_cast<func_f_f1f1_t>(special::lqn), static_cast<func_d_d1d1_t>(special::lqn),
         static_cast<func_F_F1F1_t>(special::lqn), static_cast<func_D_D1D1_t>(special::lqn)},
        2, "_lqn", lqn_doc, "()->(np1),(np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_lqn", _lqn);

    // 创建并添加名为 "_lqmn" 的通用函数对象到模块 "_gufuncs"
    PyObject *_lqmn = SpecFun_NewGUFunc(
        {static_cast<func_f_f2f2_t>(special::lqmn), static_cast<func_d_d2d2_t>(special::lqmn),
         static_cast<func_F_F2F2_t>(special::lqmn), static_cast<func_D_D2D2_t>(special::lqmn)},
        2, "_lqmn", lqmn_doc, "()->(mp1,np1),(mp1,np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_lqmn", _lqmn);

    // 创建并添加名为 "_rctj" 的通用函数对象到模块 "_gufuncs"
    PyObject *_rctj = SpecFun_NewGUFunc(
        {static_cast<func_f_f1f1_t>(special::rctj), static_cast<func_d_d1d1_t>(special::rctj)}, 2, "_rctj", rctj_doc,
        "()->(np1),(np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_rctj", _rctj);

    // 创建并添加名为 "_rcty" 的通用函数对象到模块 "_gufuncs"
    PyObject *_rcty = SpecFun_NewGUFunc(
        {static_cast<func_f_f1f1_t>(special::rcty), static_cast<func_d_d1d1_t>(special::rcty)}, 2, "_rcty", rcty_doc,
        "()->(np1),(np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_rcty", _rcty);

    // 创建并添加名为 "_sph_harm_all" 的通用函数对象到模块 "_gufuncs"
    PyObject *_sph_harm_all = SpecFun_NewGUFunc(
        {static_cast<func_dd_D2_t>(special::sph_harm_all), static_cast<func_ff_F2_t>(special::sph_harm_all)}, 1,
        "_sph_harm_all", sph_harm_all_doc, "(),()->(mp1,np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_sph_harm_all", _sph_harm_all);

    // 返回创建的 Python 模块对象 "_gufuncs"
    return _gufuncs;
}
```