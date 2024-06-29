# `.\numpy\numpy\_core\src\multiarray\temp_elide.c`

```
/*
 * 定义宏以确保使用最新的 NumPy API 版本，并声明 _MULTIARRAYMODULE 宏
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T_CLEAN 宏定义，确保 Python.h 之前不会定义 ssize_t
 * 包含 Python.h 头文件，这是 Python C API 的主头文件
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
 * 包含 NumPy 的配置信息头文件和数组对象头文件
 * npy_config.h 包含了 NumPy 库的配置信息
 * numpy/arrayobject.h 包含了数组对象的定义和操作函数声明
 */
#include "npy_config.h"
#include "numpy/arrayobject.h"

/*
 * 定义宏 NPY_NUMBER_MAX，用于返回两个数中的较大值
 * 宏 ARRAY_SIZE 用于计算数组的元素个数
 */
#define NPY_NUMBER_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ARRAY_SIZE(a) (sizeof(a)/sizeof(a[0]))

/*
 * 以下是用于尝试在 Python 表达式中避免或简化临时变量的函数
 * 以便将某些操作转换为原地操作，避免创建不必要的临时变量
 * 详细解释见注释块中的英文说明
 */

/*
 * 如果定义了 HAVE_BACKTRACE 和 HAVE_DLFCN_H，并且未定义 PYPY_VERSION
 * 则包含 feature_detection_misc.h 头文件
 */
#if defined HAVE_BACKTRACE && defined HAVE_DLFCN_H && ! defined PYPY_VERSION

/*
 * 定义 NPY_ELIDE_DEBUG 宏为 0，不打印简化的操作
 * 定义 NPY_MAX_STACKSIZE 宏为 10，指定最大的堆栈大小
 */

/* TODO can pep523 be used to somehow? */

#endif
/*
 * 以上代码段主要用于处理 Python 和 NumPy C 扩展中的一些宏定义和功能声明，
 * 包括 API 版本、数组操作宏、操作优化的功能说明等。
 */
#define PYFRAMEEVAL_FUNC "_PyEval_EvalFrameDefault"
/*
 * 根据 Python 中的调用帧默认评估函数名称定义常量 "_PyEval_EvalFrameDefault"
 * 这是 C 宏的定义，用于在代码中引用这个函数名。
 */

/*
 * 在以下条件下，回溯开销小于通过原地操作获得的速度增益的启发性数组大小，这取决于被检查的堆栈深度。
 * 通过对 10 个堆栈的测量表明，大约在 100KiB 时开始变得有价值，但为了保守起见，设置更高的值，接近 L2 缓存的溢出处。
 */
#ifndef Py_DEBUG
#define NPY_MIN_ELIDE_BYTES (256 * 1024)
#else
/*
 * 在调试模式下总是省略，但跳过标量因为它们可以在原地操作期间转换为 0 维数组。
 */
#define NPY_MIN_ELIDE_BYTES (32)
#endif

#include <dlfcn.h>

#if defined HAVE_EXECINFO_H
#include <execinfo.h>
#elif defined HAVE_LIBUNWIND_H
#include <libunwind.h>
#endif

/*
 * 在表中进行指针的线性搜索，指针的数量通常很少，但如果可以测量到性能影响，这可以转换为二分搜索。
 */
static int
find_addr(void * addresses[], npy_intp naddr, void * addr)
{
    npy_intp j;
    for (j = 0; j < naddr; j++) {
        if (addr == addresses[j]) {
            return 1;
        }
    }
    return 0;
}

/*
 * 检查调用者，获取 multiarray 和 Python 的基地址，并检查回溯是否仅在这些库中调用 dladdr，
 * 仅在初始的 multiarray 堆栈之后，所有内容都在 Python 内部时，我们可以省略，因为没有 C-API 用户可能会搞乱引用计数。
 * 仅在找到 Python 帧评估函数之前检查，大约每个堆栈大小为 10 时的开销为 10 微秒。
 *
 * TODO: 有些调用经过 umath 中的 scalarmath，但我们无法从 multiarraymodule 中获取它的基地址，因为它没有与其链接。
 */
static int
check_callers(int * cannot)
{
    NPY_TLS static int init = 0;
    /*
     * 测量 DSO（动态共享对象）对象的内存起始和结束，如果一个地址位于这些边界内部，那么它属于该库的一部分，
     * 因此我们不需要对其调用 dladdr（假设是线性内存）。
     */
    NPY_TLS static void * pos_python_start;
    NPY_TLS static void * pos_python_end;
    NPY_TLS static void * pos_ma_start;
    NPY_TLS static void * pos_ma_end;

    /* 存储已知地址以节省 dladdr 调用 */
    NPY_TLS static void * py_addr[64];
    NPY_TLS static void * pyeval_addr[64];
    NPY_TLS static npy_intp n_py_addr = 0;
    NPY_TLS static npy_intp n_pyeval = 0;

    void *buffer[NPY_MAX_STACKSIZE];
    int i, nptrs;
    int ok = 0;
    /* 无法确定调用者 */
    if (init == -1) {
        *cannot = 1;
        return 0;
    }

    nptrs = backtrace(buffer, NPY_MAX_STACKSIZE);
    if (nptrs == 0) {
        /* 完全失败，禁用省略 */
        init = -1;
        *cannot = 1;
        return 0;
    }

    /* 设置 DSO 基地址，结束后更新 */
    // 检查初始化是否已经完成，如果未完成则执行以下逻辑
    if (NPY_UNLIKELY(init == 0)) {
        Dl_info info;
        /* get python base address */
        // 获取 PyNumber_Or 函数的基地址信息
        if (dladdr(&PyNumber_Or, &info)) {
            // 设置 Python 起始和结束地址为获取到的基地址
            pos_python_start = info.dli_fbase;
            pos_python_end = info.dli_fbase;
        }
        else {
            // 如果获取失败，标记初始化为失败状态并返回 0
            init = -1;
            return 0;
        }
        /* get multiarray base address */
        // 获取 PyArray_INCREF 函数的基地址信息
        if (dladdr(&PyArray_INCREF, &info)) {
            // 设置多维数组起始和结束地址为获取到的基地址
            pos_ma_start = info.dli_fbase;
            pos_ma_end = info.dli_fbase;
        }
        else {
            // 如果获取失败，标记初始化为失败状态并返回 0
            init = -1;
            return 0;
        }
        // 初始化成功标志置为 1
        init = 1;
    }

    /* loop over callstack addresses to check if they leave numpy or cpython */
    // 循环遍历调用堆栈地址，检查是否离开了 numpy 或者 cpython
    for (i = 0; i < nptrs; i++) {
        Dl_info info;
        int in_python = 0;
        int in_multiarray = 0;
#if NPY_ELIDE_DEBUG >= 2
        // 如果调试级别大于等于2，使用dladdr函数获取缓冲区中每个地址对应的符号信息，并打印到标准输出
        dladdr(buffer[i], &info);
        printf("%s(%p) %s(%p)\n", info.dli_fname, info.dli_fbase,
               info.dli_sname, info.dli_saddr);
#endif

        /* check stored DSO boundaries first */
        // 首先检查存储的动态共享对象（DSO）边界
        if (buffer[i] >= pos_python_start && buffer[i] <= pos_python_end) {
            // 如果当前缓冲区中的地址在Python模块的范围内，设置in_python为1
            in_python = 1;
        }
        else if (buffer[i] >= pos_ma_start && buffer[i] <= pos_ma_end) {
            // 如果当前缓冲区中的地址在多维数组模块的范围内，设置in_multiarray为1
            in_multiarray = 1;
        }

        /* update DSO boundaries via dladdr if necessary */
        // 如果不在Python模块和多维数组模块内，通过dladdr更新DSO边界
        if (!in_python && !in_multiarray) {
            // 如果dladdr无法获取地址信息，初始化为-1，标记为不可用，并跳出循环
            if (dladdr(buffer[i], &info) == 0) {
                init = -1;
                ok = 0;
                break;
            }
            /* update DSO end */
            // 更新DSO的结束地址
            if (info.dli_fbase == pos_python_start) {
                // 如果地址信息中的基地址与Python模块的起始地址相同，更新Python模块的结束地址
                pos_python_end = NPY_NUMBER_MAX(buffer[i], pos_python_end);
                in_python = 1;
            }
            else if (info.dli_fbase == pos_ma_start) {
                // 如果地址信息中的基地址与多维数组模块的起始地址相同，更新多维数组模块的结束地址
                pos_ma_end = NPY_NUMBER_MAX(buffer[i], pos_ma_end);
                in_multiarray = 1;
            }
        }

        /* no longer in ok libraries and not reached PyEval -> no elide */
        // 如果既不在Python模块也不在多维数组模块内，标记为不可用，并跳出循环
        if (!in_python && !in_multiarray) {
            ok = 0;
            break;
        }

        /* in python check if the frame eval function was reached */
        // 在Python模块内，检查是否达到了帧评估函数
        if (in_python) {
            /* if reached eval we are done */
            // 如果达到了评估函数，标记为可用，并跳出循环
            if (find_addr(pyeval_addr, n_pyeval, buffer[i])) {
                ok = 1;
                break;
            }
            /*
             * check if its some other function, use pointer lookup table to
             * save expensive dladdr calls
             */
            // 检查是否为其他函数，使用指针查找表避免昂贵的dladdr调用
            if (find_addr(py_addr, n_py_addr, buffer[i])) {
                continue;
            }

            /* new python address, check for PyEvalFrame */
            // 新的Python地址，检查是否为PyEvalFrame函数
            if (dladdr(buffer[i], &info) == 0) {
                init = -1;
                ok = 0;
                break;
            }
            if (info.dli_sname &&
                    strcmp(info.dli_sname, PYFRAMEEVAL_FUNC) == 0) {
                // 如果地址信息中的符号名与PYFRAMEEVAL_FUNC相同，将地址存储到pyeval_addr数组中，标记为可用，并跳出循环
                if (n_pyeval < (npy_intp)ARRAY_SIZE(pyeval_addr)) {
                    pyeval_addr[n_pyeval++] = buffer[i];
                }
                ok = 1;
                break;
            }
            else if (n_py_addr < (npy_intp)ARRAY_SIZE(py_addr)) {
                // 否则将地址存储到py_addr数组中，以避免再次进行dladdr调用
                py_addr[n_py_addr++] = buffer[i];
            }
        }
    }

    /* all stacks after numpy are from python, we can elide */
    // 如果ok为真，则所有numpy之后的堆栈都来自Python，可以删除
    if (ok) {
        *cannot = 0;
        return 1;
    }
    else {
#if NPY_ELIDE_DEBUG != 0
        // 如果不允许删除（NPY_ELIDE_DEBUG不为0），输出信息提示不可删除
        puts("cannot elide due to c-api usage");
#endif
        *cannot = 1;
        return 0;
    }
}
/*
 * 检查在 "alhs @op@ orhs" 中，如果 alhs 是一个临时对象（refcnt == 1），
 * 则可以进行原地操作而不是创建一个新的临时对象。
 * 如果即使交换了参数也无法进行原地操作，则设置 cannot 为 true。
 */
static int
can_elide_temp(PyObject *olhs, PyObject *orhs, int *cannot)
{
    /*
     * 要成为候选对象，数组需要满足以下条件：
     * - 引用计数为 1
     * - 是一个精确的基本类型数组
     * - 拥有自己的数据
     * - 可写
     * - 不是写入时复制的数组
     * - 数据大小大于阈值 NPY_MIN_ELIDE_BYTES
     */
    PyArrayObject *alhs = (PyArrayObject *)olhs;
    if (Py_REFCNT(olhs) != 1 || !PyArray_CheckExact(olhs) ||
            !PyArray_ISNUMBER(alhs) ||
            !PyArray_CHKFLAGS(alhs, NPY_ARRAY_OWNDATA) ||
            !PyArray_ISWRITEABLE(alhs) ||
            PyArray_CHKFLAGS(alhs, NPY_ARRAY_WRITEBACKIFCOPY) ||
            PyArray_NBYTES(alhs) < NPY_MIN_ELIDE_BYTES) {
        return 0;
    }
    if (PyArray_CheckExact(orhs) ||
        PyArray_CheckAnyScalar(orhs)) {
        PyArrayObject * arhs;

        /* 从右操作数创建数组 */
        Py_INCREF(orhs);
        arhs = (PyArrayObject *)PyArray_EnsureArray(orhs);
        if (arhs == NULL) {
            return 0;
        }

        /*
         * 如果右操作数不是标量，维度必须匹配
         * TODO: 可以考虑在相同类型下进行广播
         */
        if (!(PyArray_NDIM(arhs) == 0 ||
              (PyArray_NDIM(arhs) == PyArray_NDIM(alhs) &&
               PyArray_CompareLists(PyArray_DIMS(alhs), PyArray_DIMS(arhs),
                                    PyArray_NDIM(arhs))))) {
                Py_DECREF(arhs);
                return 0;
        }

        /* 必须能够安全地转换（检查右操作数中的标量值） */
        if (PyArray_CanCastArrayTo(arhs, PyArray_DESCR(alhs),
                                   NPY_SAFE_CASTING)) {
            Py_DECREF(arhs);
            return check_callers(cannot);
        }
        Py_DECREF(arhs);
    }

    return 0;
}

/*
 * 尝试消除二元操作中的临时对象，如果 commutative 为 true，则尝试交换参数
 */
NPY_NO_EXPORT int
try_binary_elide(PyObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative)
{
    /* 当不能独立于参数顺序进行消除时设置为 true */
    int cannot = 0;
    if (can_elide_temp(m1, m2, &cannot)) {
        *res = inplace_op((PyArrayObject *)m1, m2);
#if NPY_ELIDE_DEBUG != 0
        puts("elided temporary in binary op");
#endif
        return 1;
    }
    else if (commutative && !cannot) {
        if (can_elide_temp(m2, m1, &cannot)) {
            *res = inplace_op((PyArrayObject *)m2, m1);
#if NPY_ELIDE_DEBUG != 0
            puts("elided temporary in commutative binary op");
#endif
            return 1;
        }
    }
    *res = NULL;
    return 0;
}

/* 尝试消除一元操作中的临时对象 */
NPY_NO_EXPORT int
can_elide_temp_unary(PyArrayObject * m1)
{
    int cannot;
    # 检查对象 m1 的引用计数是否为 1，且确保 m1 是精确的 NumPy 数组
    # 同时检查 m1 是否为数值类型的 NumPy 数组，拥有自己的数据，并且可写
    if (Py_REFCNT(m1) != 1 || !PyArray_CheckExact(m1) ||
            !PyArray_ISNUMBER(m1) ||
            !PyArray_CHKFLAGS(m1, NPY_ARRAY_OWNDATA) ||
            !PyArray_ISWRITEABLE(m1) ||
            PyArray_NBYTES(m1) < NPY_MIN_ELIDE_BYTES) {
        # 如果上述条件有任何一条不满足，则返回 0
        return 0;
    }
    # 检查调用者是否符合特定要求，将结果保存在 cannot 变量中
    if (check_callers(&cannot)) {
#if NPY_ELIDE_DEBUG != 0
        puts("elided temporary in unary op");
#endif
        // 如果 NPY_ELIDE_DEBUG 宏不为 0，则打印调试信息到标准输出
        return 1;
    }
    else {
        // 否则返回 0
        return 0;
    }
}
#else /* unsupported interpreter or missing backtrace */
NPY_NO_EXPORT int
can_elide_temp_unary(PyArrayObject * m1)
{
    // 返回 0，表示不支持的解释器或者缺少回溯功能
    return 0;
}

NPY_NO_EXPORT int
try_binary_elide(PyArrayObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative)
{
    // 将 res 指针设为 NULL
    *res = NULL;
    // 返回 0，表示未成功进行二元操作的优化
    return 0;
}
#endif
```