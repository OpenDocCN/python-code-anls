# `D:\src\scipysrc\scipy\scipy\stats\_unuran\unuran_callback.h`

```
#include <Python.h>
#include "ccallback.h"
#include "unuran.h"

// 定义一个宏，用于创建一个 Python 函数回调的包装
#define UNURAN_THUNK(CAST_FUNC, FUNCNAME, LEN)                                              \
    // 保护 Python 全局解释器锁的状态
    PyGILState_STATE gstate = PyGILState_Ensure();                                          \
    // 如果发生错误，返回无穷大
    if (PyErr_Occurred()) return UNUR_INFINITY;                                             \
    // 获取 C 回调函数对象
    ccallback_t *callback = ccallback_obtain();                                             \
                                                                                            \
    // 初始化变量
    PyObject *arg1 = NULL, *argobj = NULL, *res = NULL, *funcname = NULL;                   \
    double result = 0.;                                                                     \
    int error = 0;                                                                          \
                                                                                            \
    // 将输入参数转换为 Python 对象
    argobj = CAST_FUNC(x);                                                                  \
    if (argobj == NULL) {                                                                   \
        error = 1;                                                                          \
        goto done;                                                                          \
    }                                                                                       \
                                                                                            \
    // 创建字符串对象，表示函数名
    funcname = Py_BuildValue("s#", FUNCNAME, LEN);                                          \
    if (funcname == NULL) {                                                                 \
        error = 1;                                                                          \
        goto done;                                                                          \
    }                                                                                       \
                                                                                            \
    // 创建一个元组，包含函数参数
    arg1 = PyTuple_New(2);                                                                  \
    if (arg1 == NULL) {                                                                     \
        error = 1;                                                                          \
        goto done;                                                                          \
    }                                                                                       \
                                                                                            \
    // 设置元组中的第一个和第二个元素为相应的 Python 对象
    PyTuple_SET_ITEM(arg1, 0, argobj);                                                      \
    PyTuple_SET_ITEM(arg1, 1, funcname);                                                    \
    argobj = NULL; funcname = NULL;                                                         \
    // 初始化两个指针变量argobj和funcname为NULL

    res = PyObject_CallObject(callback->py_function, arg1);                                 \
    // 调用Python对象callback->py_function，使用arg1作为参数列表，并返回结果到res

    if (res == NULL) {                                                                      \
        // 如果调用返回NULL，表示调用失败
        error = 1;                                                                          \
        // 设置错误标志为1
        goto done;                                                                          \
        // 跳转到标签done处执行后续处理
    }                                                                                       \
                                                                                            \
    result = PyFloat_AsDouble(res);                                                         \
    // 将Python对象res转换为double类型，并将结果存储在变量result中

    if (PyErr_Occurred()) {                                                                 \
        // 检查Python解释器中是否发生错误
        error = 1;                                                                          \
        // 设置错误标志为1
        goto done;                                                                          \
        // 跳转到标签done处执行后续处理
    }                                                                                       \
                                                                                            \
done:                                                                                       \
    PyGILState_Release(gstate);                                                             \
    Py_XDECREF(arg1);                                                                       \
    Py_XDECREF(argobj);                                                                     \
    Py_XDECREF(funcname);                                                                   \
    Py_XDECREF(res);                                                                        \
                                                                                            \
    if (error) {                                                                            \
        /* 如果发生错误，则返回 INFINITY。这会导致 UNU.RAN 报错并返回错误码或空值。
           在 Cython 中，我们可以检查是否设置了 Python 错误变量来引发错误。 */             \
        return UNUR_INFINITY;                                                               \
    }                                                                                       \
                                                                                            \
    return result

void error_handler(const char *objid, const char *file, int line, const char *errortype, int unur_errno, const char *reason)
{
    // 如果 UNU.RAN 返回错误码不是 UNUR_SUCCESS，则处理错误
    if ( unur_errno != UNUR_SUCCESS ) {
        // 如果 Python 中已经有异常发生，直接返回
        if (PyErr_Occurred()) {
            return;
        }
        // 获取日志流
        FILE *LOG = unur_get_stream();
        char objid_[256], reason_[256];
        // 设置对象ID和原因的默认值
        (objid == NULL || strcmp(objid, "") == 0) ? strcpy(objid_, "unknown") : strcpy(objid_, objid);
        (reason == NULL || strcmp(reason, "") == 0) ? strcpy(reason_, "unknown error!") : strcpy(reason_, reason);
        // 获取错误码对应的错误消息
        const char *errno_msg = unur_get_strerror(unur_errno);
        // 如果错误类型为 "error"，则使用 fprintf 记录日志
        if ( strcmp(errortype, "error") == 0 ) {
            fprintf(LOG, "[objid: %s] %d : %s => %s", objid_, unur_errno, reason_, errno_msg);
        }
        // 否则，使用 PyErr_WarnFormat 发出运行时警告
        else {
            PyErr_WarnFormat(PyExc_RuntimeWarning, 1, "[objid: %s] %d : %s => %s", objid_, unur_errno, reason_, errno_msg);
        }
    }
}

static ccallback_signature_t unuran_call_signatures[] = {
    {NULL}
};

int init_unuran_callback(ccallback_t *callback, PyObject *fcn)
{
    int ret;
    int flags = CCALLBACK_OBTAIN;

    // 准备回调函数，并检查返回值
    ret = ccallback_prepare(callback, unuran_call_signatures, fcn, flags);
    if (ret == -1) {
        // 如果准备失败，返回错误
        return -1;
    }

    // 初始化回调信息指针为空
    callback->info_p = NULL;

    return 0;
}

int release_unuran_callback(ccallback_t *callback) {
    // 释放回调函数资源，并返回释放结果
    int ret = ccallback_release(callback);
    return ret;
}
/* ********************************* UNU.RAN Thunks ********************************* */
/* ********************************************************************************** */

// 定义一个函数 pmf_thunk，计算离散概率质量函数值
double pmf_thunk(int x, const struct unur_distr *distr)
{
    // 使用宏 UNURAN_THUNK 调用 PyLong_FromLong，传递 "pmf" 和参数个数 3
    UNURAN_THUNK(PyLong_FromLong, "pmf", 3);
}

// 定义一个函数 pdf_thunk，计算概率密度函数值
double pdf_thunk(double x, const struct unur_distr *distr)
{
    // 使用宏 UNURAN_THUNK 调用 PyFloat_FromDouble，传递 "pdf" 和参数个数 3
    UNURAN_THUNK(PyFloat_FromDouble, "pdf", 3);
}

// 定义一个函数 dpdf_thunk，计算概率密度函数的导数值
double dpdf_thunk(double x, const struct unur_distr *distr)
{
    // 使用宏 UNURAN_THUNK 调用 PyFloat_FromDouble，传递 "dpdf" 和参数个数 4
    UNURAN_THUNK(PyFloat_FromDouble, "dpdf", 4);
}

// 定义一个函数 logpdf_thunk，计算对数概率密度函数值
double logpdf_thunk(double x, const struct unur_distr *distr)
{
    // 使用宏 UNURAN_THUNK 调用 PyFloat_FromDouble，传递 "logpdf" 和参数个数 6
    UNURAN_THUNK(PyFloat_FromDouble, "logpdf", 6);
}

// 定义一个函数 cont_cdf_thunk，计算连续分布函数值
double cont_cdf_thunk(double x, const struct unur_distr *distr)
{
    // 使用宏 UNURAN_THUNK 调用 PyFloat_FromDouble，传递 "cdf" 和参数个数 3
    UNURAN_THUNK(PyFloat_FromDouble, "cdf", 3);
}

// 定义一个函数 discr_cdf_thunk，计算离散分布函数值
double discr_cdf_thunk(int x, const struct unur_distr *distr)
{
    // 使用宏 UNURAN_THUNK 调用 PyLong_FromLong，传递 "cdf" 和参数个数 3
    UNURAN_THUNK(PyLong_FromLong, "cdf", 3);
}
```