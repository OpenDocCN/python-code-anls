# `.\numpy\numpy\_core\src\umath\_rational_tests.c`

```
/* Fixed size rational numbers exposed to Python */

/* 包含必要的头文件：Python.h 和 structmember.h */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

/* 定义 NPY_NO_DEPRECATED_API，使用最新的 NumPy API 版本 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "common.h"  /* 引入 common.h 头文件，用于错误处理 */

/* 包含 math.h 头文件 */
#include <math.h>


/* Relevant arithmetic exceptions */

/* Uncomment the following line to work around a bug in numpy */
/* #define ACQUIRE_GIL */

/* 定义设置溢出异常的函数 */
static void
set_overflow(void) {
#ifdef ACQUIRE_GIL
    /* 如果定义了 ACQUIRE_GIL，则获取全局解释器锁 */
    PyGILState_STATE state = PyGILState_Ensure();
#endif
    /* 如果没有发生异常，则设置 OverflowError 异常 */
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_OverflowError,
                "overflow in rational arithmetic");
    }
#ifdef ACQUIRE_GIL
    /* 释放全局解释器锁 */
    PyGILState_Release(state);
#endif
}

/* 定义设置零除异常的函数 */
static void
set_zero_divide(void) {
#ifdef ACQUIRE_GIL
    /* 如果定义了 ACQUIRE_GIL，则获取全局解释器锁 */
    PyGILState_STATE state = PyGILState_Ensure();
#endif
    /* 如果没有发生异常，则设置 ZeroDivisionError 异常 */
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_ZeroDivisionError,
                "zero divide in rational arithmetic");
    }
#ifdef ACQUIRE_GIL
    /* 释放全局解释器锁 */
    PyGILState_Release(state);
#endif
}

/* Integer arithmetic utilities */

/* 定义安全的取负数函数 */
static inline npy_int32
safe_neg(npy_int32 x) {
    /* 如果 x 是最小的负数，设置溢出异常 */
    if (x==(npy_int32)1<<31) {
        set_overflow();
    }
    return -x;
}

/* 定义安全的取绝对值函数（32位） */
static inline npy_int32
safe_abs32(npy_int32 x) {
    npy_int32 nx;
    /* 如果 x 大于等于 0，直接返回 x */
    if (x>=0) {
        return x;
    }
    /* 否则计算绝对值，如果溢出则设置异常 */
    nx = -x;
    if (nx<0) {
        set_overflow();
    }
    return nx;
}

/* 定义安全的取绝对值函数（64位） */
static inline npy_int64
safe_abs64(npy_int64 x) {
    npy_int64 nx;
    /* 如果 x 大于等于 0，直接返回 x */
    if (x>=0) {
        return x;
    }
    /* 否则计算绝对值，如果溢出则设置异常 */
    nx = -x;
    if (nx<0) {
        set_overflow();
    }
    return nx;
}

/* 计算最大公约数函数 */
static inline npy_int64
gcd(npy_int64 x, npy_int64 y) {
    /* 取两数的绝对值 */
    x = safe_abs64(x);
    y = safe_abs64(y);
    /* 辗转相除法计算最大公约数 */
    if (x < y) {
        npy_int64 t = x;
        x = y;
        y = t;
    }
    while (y) {
        npy_int64 t;
        x = x%y;
        t = x;
        x = y;
        y = t;
    }
    return x;
}

/* 计算最小公倍数函数 */
static inline npy_int64
lcm(npy_int64 x, npy_int64 y) {
    npy_int64 lcm;
    /* 如果 x 或 y 为 0，直接返回 0 */
    if (!x || !y) {
        return 0;
    }
    /* 计算最小公倍数，并检查是否溢出 */
    x /= gcd(x,y);
    lcm = x*y;
    if (lcm/y!=x) {
        set_overflow();
    }
    return safe_abs64(lcm);
}

/* Fixed precision rational numbers */

/* 定义有理数结构体 */
typedef struct {
    /* 分子 */
    npy_int32 n;
    /*
     * 分母减一: numpy.zeros() 为非对象类型使用 memset(0)，确保 rational(0) 全为零字节
     */
    npy_int32 dmm;
} rational;

/* 定义创建整数有理数的函数 */
static inline rational
make_rational_int(npy_int64 n) {
    rational r = {(npy_int32)n,0};
    /* 如果分子不等于原整数 n，设置溢出异常 */
    if (r.n != n) {
        set_overflow();
    }
    return r;
}

/* 定义创建有理数的函数（慢速版本） */
static rational
make_rational_slow(npy_int64 n_, npy_int64 d_) {
    rational r = {0};
    /* 如果分母为零，设置零除异常 */
    if (!d_) {
        set_zero_divide();
    }
    /* 省略部分代码，未完整给出 */
}
    // 否则分支：处理分数的规范化
    else {
        // 计算分子和分母的最大公约数
        npy_int64 g = gcd(n_, d_);
        // 声明整型变量 d
        npy_int32 d;
        // 简化分数：分子除以最大公约数，分母除以最大公约数
        n_ /= g;
        d_ /= g;
        // 将简化后的分子转换为 npy_int32 类型，并赋给结果结构体的分子成员
        r.n = (npy_int32)n_;
        // 将简化后的分母赋给变量 d
        d = (npy_int32)d_;
        // 如果简化后的分子或分母与原始值不相等，设置溢出标志
        if (r.n != n_ || d != d_) {
            set_overflow();
        }
        // 否则，处理正常情况
        else {
            // 如果分母小于等于 0，取其绝对值并将结果结构体的分子成员取反
            if (d <= 0) {
                d = -d;
                r.n = safe_neg(r.n);
            }
            // 将分母减 1 的结果赋给结果结构体的 dmm 成员
            r.dmm = d - 1;
        }
    }
    // 返回处理后的结果结构体
    return r;
}

static inline npy_int32
d(rational r) {
    return r.dmm+1;
}

/* Assumes d_ > 0 */
// 快速创建有理数的函数，假设分母大于0
static rational
make_rational_fast(npy_int64 n_, npy_int64 d_) {
    // 计算最大公约数
    npy_int64 g = gcd(n_,d_);
    rational r;
    // 简化分数
    n_ /= g;
    d_ /= g;
    r.n = (npy_int32)n_;
    r.dmm = (npy_int32)(d_-1);
    // 如果简化后分数不等于原始分数，则设置溢出标志
    if (r.n!=n_ || r.dmm+1!=d_) {
        set_overflow();
    }
    return r;
}

static inline rational
rational_negative(rational r) {
    rational x;
    // 对有理数取负
    x.n = safe_neg(r.n);
    x.dmm = r.dmm;
    return x;
}

static inline rational
rational_add(rational x, rational y) {
    /*
     * Note that the numerator computation can never overflow int128_t,
     * since each term is strictly under 2**128/4 (since d > 0).
     */
    // 有理数加法，注意分子计算不会溢出，因为每项严格小于2**128/4
    return make_rational_fast((npy_int64)x.n*d(y)+(npy_int64)d(x)*y.n,
        (npy_int64)d(x)*d(y));
}

static inline rational
rational_subtract(rational x, rational y) {
    /* We're safe from overflow as with + */
    // 有理数减法，与加法类似，不会溢出
    return make_rational_fast((npy_int64)x.n*d(y)-(npy_int64)d(x)*y.n,
        (npy_int64)d(x)*d(y));
}

static inline rational
rational_multiply(rational x, rational y) {
    /* We're safe from overflow as with + */
    // 有理数乘法，与加法类似，不会溢出
    return make_rational_fast((npy_int64)x.n*y.n,(npy_int64)d(x)*d(y));
}

static inline rational
rational_divide(rational x, rational y) {
    // 有理数除法，使用慢速的方式进行计算
    return make_rational_slow((npy_int64)x.n*d(y),(npy_int64)d(x)*y.n);
}

static inline npy_int64
rational_floor(rational x) {
    /* Always round down */
    // 有理数向下取整
    if (x.n>=0) {
        return x.n/d(x);
    }
    /*
     * This can be done without casting up to 64 bits, but it requires
     * working out all the sign cases
     */
    // 处理负数情况的向下取整
    return -((-(npy_int64)x.n+d(x)-1)/d(x));
}

static inline npy_int64
rational_ceil(rational x) {
    // 有理数向上取整
    return -rational_floor(rational_negative(x));
}

static inline rational
rational_remainder(rational x, rational y) {
    // 计算有理数的余数
    return rational_subtract(x, rational_multiply(y,make_rational_int(
                    rational_floor(rational_divide(x,y)))));
}

static inline rational
rational_abs(rational x) {
    rational y;
    // 计算有理数的绝对值
    y.n = safe_abs32(x.n);
    y.dmm = x.dmm;
    return y;
}

static inline npy_int64
rational_rint(rational x) {
    /*
     * Round towards nearest integer, moving exact half integers towards
     * zero
     */
    // 四舍五入到最近的整数，对半整数向零舍入
    npy_int32 d_ = d(x);
    return (2*(npy_int64)x.n+(x.n<0?-d_:d_))/(2*(npy_int64)d_);
}

static inline int
rational_sign(rational x) {
    // 返回有理数的符号
    return x.n<0?-1:x.n==0?0:1;
}

static inline rational
rational_inverse(rational x) {
    rational y = {0};
    if (!x.n) {
        // 如果有理数为零，则设置零除错误
        set_zero_divide();
    }
    else {
        npy_int32 d_;
        y.n = d(x);
        d_ = x.n;
        if (d_ <= 0) {
            d_ = safe_neg(d_);
            y.n = -y.n;
        }
        y.dmm = d_-1;
    }
    return y;
}

static inline int
rational_eq(rational x, rational y) {
    /*
     * Since we enforce d > 0, and store fractions in reduced form,
     * equality is easy.
     */
    // 判断两个有理数是否相等，由于我们要求 d > 0 并且以最简分数形式存储，因此判断相等很简单
    return x.n==y.n && x.dmm==y.dmm;
}

static inline int
static int
scan_rational(const char** s, rational* x) {
    // 定义变量以保存分子、分母及偏移量
    long n,d;
    int offset;
    // 定义指针 ss，指向字符串起始位置
    const char* ss;
    
    // 尝试从字符串 *s 中读取长整型数值 n，同时记录偏移量到 offset
    if (sscanf(*s,"%ld%n",&n,&offset)<=0) {
        // 若未成功读取，返回 0 表示失败
        return 0;
    }
    // 更新 ss 为 *s 加上偏移量 offset 的位置
    ss = *s+offset;
    
    // 如果 ss 指向的字符不是 '/'，说明只有分子，构造一个整数有理数并返回 1
    if (*ss!='/') {
        *s = ss;
        *x = make_rational_int(n);
        return 1;
    }
    
    // 否则，ss 指向 '/'，接着解析分母 d
    ss++;
    // 尝试从 ss 中读取长整型数值 d，同时记录偏移量到 offset
    if (sscanf(ss,"%ld%n",&d,&offset)<=0 || d<=0) {
        // 如果未能成功读取或分母 d 小于等于 0，返回 0 表示失败
        return 0;
    }
    // 更新 *s 为 ss 加上偏移量 offset 的位置
    *s = ss+offset;
    // 使用分子 n 和分母 d 构造一个有理数并保存到 *x
    *x = make_rational_slow(n,d);
    // 返回 1 表示成功解析有理数
    return 1;
}


这段代码主要用于解析字符串表示的有理数，包括处理只有分子或分子分母都有的情况，返回对应的有理数对象或者指示解析失败。
    // 循环遍历参数元组中的每个元素，i 是循环变量，从 0 开始到 size-1
    for (i=0; i<size; i++) {
        PyObject* y;
        int eq;
        // 将参数元组中第 i 个元素赋值给 x[i]
        x[i] = PyTuple_GET_ITEM(args, i);
        // 将 x[i] 转换为长整型并赋值给 n[i]
        n[i] = PyLong_AsLong(x[i]);
        // 检查是否在转换过程中发生错误
        if (error_converting(n[i])) {
            // 如果是类型错误，则设置异常并返回 0
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError,
                        "expected integer %s, got %s",
                        // 根据 i 的值选择错误信息中的描述
                        (i ? "denominator" : "numerator"),
                        // 获取参数类型的名称
                        x[i]->ob_type->tp_name);
            }
            return 0;
        }
        /* 检查是否为精确整数 */
        // 创建一个长整型对象 y，用于比较
        y = PyLong_FromLong(n[i]);
        // 如果创建对象失败，则返回 0
        if (!y) {
            return 0;
        }
        // 使用 RichCompare 检查 x[i] 和 y 是否相等
        eq = PyObject_RichCompareBool(x[i],y,Py_EQ);
        // 释放对象 y
        Py_DECREF(y);
        // 如果比较出错，则返回 0
        if (eq<0) {
            return 0;
        }
        // 如果不相等，则设置类型错误异常并返回 0
        if (!eq) {
            PyErr_Format(PyExc_TypeError,
                    "expected integer %s, got %s",
                    // 根据 i 的值选择错误信息中的描述
                    (i ? "denominator" : "numerator"),
                    // 获取参数类型的名称
                    x[i]->ob_type->tp_name);
            return 0;
        }
    }
    // 调用 make_rational_slow 函数生成一个有理数对象 r
    r = make_rational_slow(n[0],n[1]);
    // 如果在生成有理数对象过程中发生异常，则返回 0
    if (PyErr_Occurred()) {
        return 0;
    }
    // 返回 Python 中的有理数对象 PyRational
    return PyRational_FromRational(r);
/*
 * 宏定义 AS_RATIONAL(dst,object) 将对象转换为 rational 结构体，赋给 dst。
 * 如果对象是 PyRational 类型，则直接获取其值。
 * 如果对象是整数类型，尝试将其转换为 long 型，然后生成对应的 PyLong 对象进行比较和处理。
 * 若转换失败，根据情况返回 Py_NotImplemented 或 0，或者抛出异常。
 */
#define AS_RATIONAL(dst,object) \
    { \
        dst.n = 0; \  // 初始化 dst 结构体的分子部分为 0
        if (PyRational_Check(object)) { \  // 检查 object 是否为 PyRational 类型
            dst = ((PyRational*)object)->r; \  // 若是，直接获取其 rational 结构体赋给 dst
        } \
        else { \  // 如果不是 PyRational 类型
            PyObject* y_; \  // 定义一个 PyObject 指针 y_
            int eq_; \  // 定义一个整型变量 eq_
            long n_ = PyLong_AsLong(object); \  // 尝试将 object 转换为 long 型
            if (error_converting(n_)) { \  // 如果转换失败
                if (PyErr_ExceptionMatches(PyExc_TypeError)) { \  // 如果是类型错误异常
                    PyErr_Clear(); \  // 清除异常状态
                    Py_INCREF(Py_NotImplemented); \  // 增加 Py_NotImplemented 的引用计数
                    return Py_NotImplemented; \  // 返回 Py_NotImplemented 对象
                } \
                return 0; \  // 转换出错，返回 0 表示失败
            } \
            y_ = PyLong_FromLong(n_); \  // 将 n_ 转换为 PyLong 对象 y_
            if (!y_) { \  // 如果转换失败（y_ 为 NULL）
                return 0; \  // 返回 0 表示失败
            } \
            eq_ = PyObject_RichCompareBool(object,y_,Py_EQ); \  // 使用 PyObject_RichCompareBool 比较 object 和 y_ 是否相等
            Py_DECREF(y_); \  // 减少 PyLong 对象 y_ 的引用计数
            if (eq_<0) { \  // 如果比较出错
                return 0; \  // 返回 0 表示失败
            } \
            if (!eq_) { \  // 如果不相等
                Py_INCREF(Py_NotImplemented); \  // 增加 Py_NotImplemented 的引用计数
                return Py_NotImplemented; \  // 返回 Py_NotImplemented 对象
            } \
            dst = make_rational_int(n_); \  // 将 n_ 转换为 rational 结构体赋给 dst
        } \
    }

/*
 * 定义了一个函数 pyrational_richcompare，用于比较两个 PyRational 对象的大小关系。
 * 根据传入的操作符 op，调用不同的比较函数进行比较。
 * 返回一个 PyBool 对象表示比较结果。
 */
static PyObject*
pyrational_richcompare(PyObject* a, PyObject* b, int op) {
    rational x, y; \  // 定义 rational 结构体变量 x 和 y
    int result = 0; \  // 定义整型变量 result 并初始化为 0
    AS_RATIONAL(x,a); \  // 将 a 转换为 rational 结构体赋给 x
    AS_RATIONAL(y,b); \  // 将 b 转换为 rational 结构体赋给 y
    #define OP(py,op) case py: result = rational_##op(x,y); break; \  // 根据操作符 op 定义不同的比较操作，并将结果赋给 result
    switch (op) { \  // 根据操作符 op 执行不同的比较
        OP(Py_LT,lt) \  // 小于操作
        OP(Py_LE,le) \  // 小于等于操作
        OP(Py_EQ,eq) \  // 等于操作
        OP(Py_NE,ne) \  // 不等于操作
        OP(Py_GT,gt) \  // 大于操作
        OP(Py_GE,ge) \  // 大于等于操作
    }; \  // 结束 switch
    #undef OP \  // 取消宏定义 OP
    return PyBool_FromLong(result); \  // 返回比较结果的 PyBool 对象
}

/*
 * 定义了一个函数 pyrational_repr，用于返回 PyRational 对象的字符串表示形式。
 * 如果 rational 结构体 x 的分母不为 1，则返回形如 "rational(分子,分母)" 的字符串。
 * 否则，返回形如 "rational(分子)" 的字符串。
 */
static PyObject*
pyrational_repr(PyObject* self) {
    rational x = ((PyRational*)self)->r; \  // 将 self 转换为 PyRational 对象，并获取其 rational 结构体赋给 x
    if (d(x)!=1) { \  // 如果 x 的分母不为 1
        return PyUnicode_FromFormat(
                "rational(%ld,%ld)",(long)x.n,(long)d(x)); \  // 返回带分子和分母的格式化字符串对象
    }
    else { \  // 如果 x 的分母为 1
        return PyUnicode_FromFormat(
                "rational(%ld)",(long)x.n); \  // 返回只带分子的格式化字符串对象
    }
}

/*
 * 定义了一个函数 pyrational_str，用于返回 PyRational 对象的简化字符串表示形式。
 * 如果 rational 结构体 x 的分母不为 1，则返回形如 "分子/分母" 的字符串。
 * 否则，返回形如 "分子" 的字符串。
 */
static PyObject*
pyrational_str(PyObject* self) {
    rational x = ((PyRational*)self)->r; \  // 将 self 转换为 PyRational 对象，并获取其 rational 结构体赋给 x
    if (d(x)!=1) { \  // 如果 x 的分母不为 1
        return PyUnicode_FromFormat(
                "%ld/%ld",(long)x.n,(long)d(x)); \  // 返回带分子和分母的格式化字符串对象
    }
    else { \  // 如果 x 的分母为 1
        return PyUnicode_FromFormat(
                "%ld",(long)x.n); \  // 返回只带分子的格式化字符串对象
    }
}

/*
 * 定义了一个函数 pyrational_hash，用于计算 PyRational 对象的哈希值。
 * 使用较弱的哈希算法，计算结果为 h。
 * 如果计算结果为 -1，则返回特定的哈希值 2，以避免返回 -1。
 */
static npy_hash_t
pyrational_hash(PyObject* self) {
    rational x = ((PyRational*)self)->r; \  // 将 self 转换为 PyRational 对象，并获取其 rational 结构体赋给 x
    /* 使用 Python 期望的较弱哈希算法 */
    long h = 131071*x.n+524287*x.dmm; \  // 计算哈希值 h
    /* 不返回特定的错误值 -1 */
    return h==-1?2:h; \  // 如果 h 为 -1，则返回 2；否则返回 h 本身
}

/*
 * 宏定义 RATIONAL_BINOP_2(name,exp) 定义了一组二元操作函数。
 * 对于给定的操作名 name 和表达式 exp，定义了一个函数 pyrational_##name。
 * 该函数接受两个 PyObject 对象 a 和 b，将它们转换为 rational 结构体 x 和 y，然后计算表达式 exp 得到 z。
 * 如果在计算过程中出现错误，则返回 0 表示失败；否则返回 z 的 PyRational 表示。
 */
#define RATIONAL_BINOP_2(name,exp) \
    static PyObject* \
    pyrational_##name(PyObject* a, PyObject* b) { \
        rational x, y, z; \  // 定义 rational 结构体变量 x, y 和 z
        AS_RATIONAL(x,a); \  // 将 a 转换为 rational 结构体赋给 x
        AS_RATIONAL(y,b); \  // 将 b 转换为 rational 结构体赋给 y
        z = exp; \  // 计算 exp 并将
#define RATIONAL_BINOP_2(floor_divide,
    make_rational_int(rational_floor(rational_divide(x,y))))

#define RATIONAL_UNOP(name,type,exp,convert) \
    static PyObject* \
    pyrational_##name(PyObject* self) { \
        rational x = ((PyRational*)self)->r; \
        type y = exp; \
        // 检查是否有异常发生
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        // 转换并返回结果
        return convert(y); \
    }

// 定义名为 negative 的一元运算函数
RATIONAL_UNOP(negative,rational,rational_negative(x),PyRational_FromRational)
// 定义名为 absolute 的一元运算函数
RATIONAL_UNOP(absolute,rational,rational_abs(x),PyRational_FromRational)
// 定义名为 int 的一元运算函数
RATIONAL_UNOP(int,long,rational_int(x),PyLong_FromLong)
// 定义名为 float 的一元运算函数
RATIONAL_UNOP(float,double,rational_double(x),PyFloat_FromDouble)

// 定义名为 positive 的一元运算函数
static PyObject*
pyrational_positive(PyObject* self) {
    // 增加对象的引用计数并返回对象本身
    Py_INCREF(self);
    return self;
}

// 定义名为 nonzero 的一元运算函数
static int
pyrational_nonzero(PyObject* self) {
    // 获取对象中的 rational 结构体并检查是否非零
    rational x = ((PyRational*)self)->r;
    return rational_nonzero(x);
}

// 定义 PyNumberMethods 结构体及其成员函数指针
static PyNumberMethods pyrational_as_number = {
    pyrational_add,          /* nb_add */
    pyrational_subtract,     /* nb_subtract */
    pyrational_multiply,     /* nb_multiply */
    pyrational_remainder,    /* nb_remainder */
    0,                       /* nb_divmod */
    0,                       /* nb_power */
    pyrational_negative,     /* nb_negative */
    pyrational_positive,     /* nb_positive */
    pyrational_absolute,     /* nb_absolute */
    pyrational_nonzero,      /* nb_nonzero */
    0,                       /* nb_invert */
    0,                       /* nb_lshift */
    0,                       /* nb_rshift */
    0,                       /* nb_and */
    0,                       /* nb_xor */
    0,                       /* nb_or */
    pyrational_int,          /* nb_int */
    0,                       /* reserved */
    pyrational_float,        /* nb_float */

    0,                       /* nb_inplace_add */
    0,                       /* nb_inplace_subtract */
    0,                       /* nb_inplace_multiply */
    0,                       /* nb_inplace_remainder */
    0,                       /* nb_inplace_power */
    0,                       /* nb_inplace_lshift */
    0,                       /* nb_inplace_rshift */
    0,                       /* nb_inplace_and */
    0,                       /* nb_inplace_xor */
    0,                       /* nb_inplace_or */

    pyrational_floor_divide, /* nb_floor_divide */
    pyrational_divide,       /* nb_true_divide */
    0,                       /* nb_inplace_floor_divide */
    0,                       /* nb_inplace_true_divide */
    0,                       /* nb_index */
};

// 定义返回 numerator 的函数
static PyObject*
pyrational_n(PyObject* self, void* closure) {
    return PyLong_FromLong(((PyRational*)self)->r.n);
}

// 定义返回 denominator 的函数
static PyObject*
pyrational_d(PyObject* self, void* closure) {
    return PyLong_FromLong(d(((PyRational*)self)->r));
}

// 定义属性的获取函数列表
static PyGetSetDef pyrational_getset[] = {
    {(char*)"n",pyrational_n,0,(char*)"numerator",0},
    {(char*)"d",pyrational_d,0,(char*)"denominator",0},
    {0} /* sentinel */
};
static PyTypeObject PyRational_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)               /* 初始化基类对象，无基类，初始大小为0 */
    "numpy._core._rational_tests.rational",      /* 类型对象的名称 */
    sizeof(PyRational),                          /* 类型对象的基本大小 */
    0,                                           /* 每个对象的附加大小 */
    0,                                           /* 对象释放函数 */
    0,                                           /* 对象打印函数 */
    0,                                           /* 获取对象属性的方法 */
    0,                                           /* 设置对象属性的方法 */
    0,                                           /* 保留字段 */
    pyrational_repr,                             /* 对象的字符串表示形式的函数 */
    &pyrational_as_number,                       /* 数字协议的实现 */
    0,                                           /* 序列协议的实现 */
    0,                                           /* 映射协议的实现 */
    pyrational_hash,                             /* 哈希函数 */
    0,                                           /* 调用对象的函数 */
    pyrational_str,                              /* 对象的字符串表示形式的函数 */
    0,                                           /* 获取对象属性的函数 */
    0,                                           /* 设置对象属性的函数 */
    0,                                           /* 缓冲区协议的实现 */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* 类型标志 */
    "Fixed precision rational numbers",          /* 类型的文档字符串 */
    0,                                           /* 遍历对象的函数 */
    0,                                           /* 清除对象的函数 */
    pyrational_richcompare,                      /* 对象之间的比较函数 */
    0,                                           /* 弱引用列表偏移量 */
    0,                                           /* 迭代器协议的实现 */
    0,                                           /* 迭代器的下一个元素的函数 */
    0,                                           /* 类型的方法 */
    0,                                           /* 类型的成员 */
    pyrational_getset,                           /* 获取和设置对象属性的函数集 */
    0,                                           /* 基类对象 */
    0,                                           /* 类型的字典 */
    0,                                           /* 描述符的获取函数 */
    0,                                           /* 描述符的设置函数 */
    0,                                           /* 字典偏移量 */
    0,                                           /* 初始化对象的函数 */
    0,                                           /* 分配对象的函数 */
    pyrational_new,                              /* 创建新对象的函数 */
    0,                                           /* 释放对象的函数 */
    0,                                           /* 垃圾回收标志 */
    0,                                           /* 基类元组 */
    0,                                           /* 方法解析顺序 */
    0,                                           /* 类型缓存 */
    0,                                           /* 子类 */
    0,                                           /* 弱引用列表 */
    0,                                           /* 对象销毁的函数 */
    0,                                           /* 版本标签 */
};

/* NumPy support */

static PyObject*
npyrational_getitem(void* data, void* arr) {
    // 声明一个有理数变量 r
    rational r;
    // 将 data 指向的数据复制到 r 中
    memcpy(&r,data,sizeof(rational));
    // 调用 PyRational_FromRational 函数，将有理数 r 转换为 Python 对象并返回
    return PyRational_FromRational(r);
}

static int
npyrational_setitem(PyObject* item, void* data, void* arr) {
    // 声明一个有理数变量 r
    rational r;
    // 如果 item 是 PyRational 类型的对象
    if (PyRational_Check(item)) {
        // 从 PyRational 对象中获取有理数 r，并赋值给 r
        r = ((PyRational*)item)->r;
    }
    else {
        // 将 item 转换为 long long 类型
        long long n = PyLong_AsLongLong(item);
        PyObject* y;
        int eq;
        // 如果转换出错，返回 -1
        if (error_converting(n)) {
            return -1;
        }
        // 创建一个新的 PyLong 对象 y
        y = PyLong_FromLongLong(n);
        if (!y) {
            return -1;
        }
        // 比较 item 和 y 是否相等
        eq = PyObject_RichCompareBool(item, y, Py_EQ);
        Py_DECREF(y);
        // 如果比较失败，返回 -1
        if (eq<0) {
            return -1;
        }
        // 如果不相等，抛出 TypeError 异常
        if (!eq) {
            PyErr_Format(PyExc_TypeError,
                    "expected rational, got %s", item->ob_type->tp_name);
            return -1;
        }
        // 根据整数 n 创建有理数 r
        r = make_rational_int(n);
    }
    // 将有理数 r 复制到 data 指向的内存中
    memcpy(data, &r, sizeof(rational));
    return 0;
}

static inline void
byteswap(npy_int32* x) {
    // 将 x 指向的 npy_int32 类型数据进行字节交换
    char* p = (char*)x;
    size_t i;
    for (i = 0; i < sizeof(*x)/2; i++) {
        size_t j = sizeof(*x)-1-i;
        char t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
}

static void
npyrational_copyswapn(void* dst_, npy_intp dstride, void* src_,
        npy_intp sstride, npy_intp n, int swap, void* arr) {
    // 将 dst_ 和 src_ 指向的数据进行复制和交换
    char *dst = (char*)dst_, *src = (char*)src_;
    npy_intp i;
    // 如果 src 为 NULL，则直接返回
    if (!src) {
        return;
    }
    // 如果 swap 为真，则进行数据交换
    if (swap) {
        for (i = 0; i < n; i++) {
            // 获取目标地址 dst+dstride*i 处的有理数 r
            rational* r = (rational*)(dst+dstride*i);
            // 将源地址 src+sstride*i 处的数据复制到 r 中
            memcpy(r,src+sstride*i,sizeof(rational));
            // 对 r 中的 n 和 dmm 进行字节交换
            byteswap(&r->n);
            byteswap(&r->dmm);
        }
    }
    // 如果不需要交换，并且步长相等，则直接进行数据复制
    else if (dstride == sizeof(rational) && sstride == sizeof(rational)) {
        memcpy(dst, src, n*sizeof(rational));
    }
    else {
        // 否则，按照指定的步长复制数据
        for (i = 0; i < n; i++) {
            memcpy(dst + dstride*i, src + sstride*i, sizeof(rational));
        }
    }
}

static void
npyrational_copyswap(void* dst, void* src, int swap, void* arr) {
    // 将 src 指向的有理数数据复制到 dst 指向的内存中，并进行字节交换
    rational* r;
    // 如果 src 为 NULL，则直接返回
    if (!src) {
        return;
    }
    // 将 src 强制转换为有理数类型，并复制到 dst 指向的内存中
    r = (rational*)dst;
    memcpy(r,src,sizeof(rational));
    // 如果 swap 为真，则对 r 中的 n 和 dmm 进行字节交换
    if (swap) {
        byteswap(&r->n);
        byteswap(&r->dmm);
    }
}

static int
npyrational_compare(const void* d0, const void* d1, void* arr) {
    // 比较两个有理数对象的大小关系
    rational x = *(rational*)d0,
             y = *(rational*)d1;
    // 如果 x 小于 y，则返回 -1；如果 x 等于 y，则返回 0；否则返回 1
    return rational_lt(x,y)?-1:rational_eq(x,y)?0:1;
}

#define FIND_EXTREME(name,op) \
    // 定义一个名为 name 的宏，它接受一个操作符 op
    static int \
    # 定义一个名为 npyrational_##name 的函数，接受四个参数：void* data_, npy_intp n, npy_intp* max_ind, void* arr
    npyrational_##name(void* data_, npy_intp n, \
            npy_intp* max_ind, void* arr) { \
        # 声明一个指向 rational 类型的常量指针 data
        const rational* data; \
        # 声明一个 npy_intp 类型的变量 best_i
        npy_intp best_i; \
        # 声明一个 rational 类型的变量 best_r
        rational best_r; \
        # 声明一个 npy_intp 类型的变量 i
        npy_intp i; \
        # 如果 n 为 0，则直接返回 0
        if (!n) { \
            return 0; \
        } \
        # 将 data_ 转换为 rational* 类型，并赋值给 data
        data = (rational*)data_; \
        # 初始化 best_i 为 0
        best_i = 0; \
        # 初始化 best_r 为 data[0]
        best_r = data[0]; \
        # 循环遍历从 1 到 n-1 的索引 i
        for (i = 1; i < n; i++) { \
            # 如果 data[i] 比 best_r 满足 rational_##op 的条件
            if (rational_##op(data[i],best_r)) { \
                # 更新 best_i 为 i
                best_i = i; \
                # 更新 best_r 为 data[i]
                best_r = data[i]; \
            } \
        } \
        # 将 best_i 的值赋给 *max_ind
        *max_ind = best_i; \
        # 返回 0
        return 0; \
    }
FIND_EXTREME(argmin,lt)
# 定义了一个宏 FIND_EXTREME，用于找到序列中的最小值，使用 lt 操作符进行比较

FIND_EXTREME(argmax,gt)
# 定义了一个宏 FIND_EXTREME，用于找到序列中的最大值，使用 gt 操作符进行比较

static void
npyrational_dot(void* ip0_, npy_intp is0, void* ip1_, npy_intp is1,
        void* op, npy_intp n, void* arr) {
    rational r = {0};
    const char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
    npy_intp i;
    for (i = 0; i < n; i++) {
        r = rational_add(r,rational_multiply(*(rational*)ip0,*(rational*)ip1));
        ip0 += is0;
        ip1 += is1;
    }
    *(rational*)op = r;
}
# 计算两个有理数数组的点积，并将结果存储在 op 中

static npy_bool
npyrational_nonzero(void* data, void* arr) {
    rational r;
    memcpy(&r,data,sizeof(r));
    return rational_nonzero(r)?NPY_TRUE:NPY_FALSE;
}
# 检查给定的有理数是否非零，返回相应的布尔值

static int
npyrational_fill(void* data_, npy_intp length, void* arr) {
    rational* data = (rational*)data_;
    rational delta = rational_subtract(data[1],data[0]);
    rational r = data[1];
    npy_intp i;
    for (i = 2; i < length; i++) {
        r = rational_add(r,delta);
        data[i] = r;
    }
    return 0;
}
# 用等差数列填充有理数数组，起始值为 data[0] 和 data[1]，公差为 delta

static int
npyrational_fillwithscalar(void* buffer_, npy_intp length,
        void* value, void* arr) {
    rational r = *(rational*)value;
    rational* buffer = (rational*)buffer_;
    npy_intp i;
    for (i = 0; i < length; i++) {
        buffer[i] = r;
    }
    return 0;
}
# 用给定的有理数值填充有理数数组

static PyArray_ArrFuncs npyrational_arrfuncs;
# 定义了一个 PyArray_ArrFuncs 结构体变量 npyrational_arrfuncs

typedef struct { char c; rational r; } align_test;
# 定义了一个结构体 align_test，包含一个字符和一个有理数成员

PyArray_DescrProto npyrational_descr_proto = {
    PyObject_HEAD_INIT(0)
    &PyRational_Type,       /* typeobj */
    'V',                    /* kind */
    'r',                    /* type */
    '=',                    /* byteorder */
    /*
     * For now, we need NPY_NEEDS_PYAPI in order to make numpy detect our
     * exceptions.  This isn't technically necessary,
     * since we're careful about thread safety, and hopefully future
     * versions of numpy will recognize that.
     */
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM, /* hasobject */
    0,                      /* type_num */
    sizeof(rational),       /* elsize */
    offsetof(align_test,r), /* alignment */
    0,                      /* subarray */
    0,                      /* fields */
    0,                      /* names */
    &npyrational_arrfuncs,  /* f */
};
# 定义了一个 PyArray_DescrProto 结构体变量 npyrational_descr_proto，描述有理数类型的数组的属性

#define DEFINE_CAST(From,To,statement) \
    static void \
    npycast_##From##_##To(void* from_, void* to_, npy_intp n, \
                          void* fromarr, void* toarr) { \
        const From* from = (From*)from_; \
        To* to = (To*)to_; \
        npy_intp i; \
        for (i = 0; i < n; i++) { \
            From x = from[i]; \
            statement \
            to[i] = y; \
        } \
    }
# 定义一个宏 DEFINE_CAST，用于生成类型转换函数的模板

#define DEFINE_INT_CAST(bits) \
    DEFINE_CAST(npy_int##bits,rational,rational y = make_rational_int(x);) \
    DEFINE_CAST(rational,npy_int##bits,npy_int32 z = rational_int(x); \
                npy_int##bits y = z; if (y != z) set_overflow();)
# 定义一个宏 DEFINE_INT_CAST，用于生成整数类型和有理数之间的转换函数模板
DEFINE_INT_CAST(8)
DEFINE_INT_CAST(16)
DEFINE_INT_CAST(32)
DEFINE_INT_CAST(64)
DEFINE_CAST(rational,float,double y = rational_double(x);)
# 调用 DEFINE_INT_CAST 和 DEFINE_CAST 宏，分别生成不同位数整数和有理数，以及有理数和浮点数、双精度浮点数之间的转换函数
#define BINARY_UFUNC(name,intype0,intype1,outtype,exp) \
    // 定义二元通用函数的宏，参数包括函数名、输入类型0、输入类型1、输出类型以及表达式
    void name(char** args, npy_intp const *dimensions, \
              npy_intp const *steps, void* data) { \
        // 定义函数，接受字符指针数组args、维度dimensions、步长steps以及数据指针data作为参数
        npy_intp is0 = steps[0], is1 = steps[1], \
            os = steps[2], n = *dimensions; \
        // 定义输入步长is0、is1，输出步长os以及数据点数n
        char *i0 = args[0], *i1 = args[1], *o = args[2]; \
        // 初始化输入指针i0、i1和输出指针o
        int k; \
        // 定义循环计数器k
        for (k = 0; k < n; k++) { \
            // 循环遍历每个数据点
            intype0 x = *(intype0*)i0; \
            // 将i0解引用为intype0类型，并赋值给x
            intype1 y = *(intype1*)i1; \
            // 将i1解引用为intype1类型，并赋值给y
            *(outtype*)o = exp; \
            // 将exp计算结果转换为outtype类型，并存储到o指向的位置
            i0 += is0; i1 += is1; o += os; \
            // 更新指针，移动到下一个数据点
        } \
    }

#define RATIONAL_BINARY_UFUNC(name,type,exp) \
    // 定义有理数二元通用函数的宏，参数包括函数名、类型以及表达式
    BINARY_UFUNC(rational_ufunc_##name,rational,rational,type,exp)
    // 使用BINARY_UFUNC宏生成有理数版本的二元通用函数

RATIONAL_BINARY_UFUNC(add,rational,rational_add(x,y))
// 定义有理数加法函数
RATIONAL_BINARY_UFUNC(subtract,rational,rational_subtract(x,y))
// 定义有理数减法函数
RATIONAL_BINARY_UFUNC(multiply,rational,rational_multiply(x,y))
// 定义有理数乘法函数
RATIONAL_BINARY_UFUNC(divide,rational,rational_divide(x,y))
// 定义有理数除法函数
RATIONAL_BINARY_UFUNC(remainder,rational,rational_remainder(x,y))
// 定义有理数求余函数
RATIONAL_BINARY_UFUNC(floor_divide,rational,
    make_rational_int(rational_floor(rational_divide(x,y))))
// 定义有理数向下整除函数

PyUFuncGenericFunction rational_ufunc_true_divide = rational_ufunc_divide;
// 设置有理数真除函数为有理数除法函数

RATIONAL_BINARY_UFUNC(minimum,rational,rational_lt(x,y)?x:y)
// 定义有理数最小值函数
RATIONAL_BINARY_UFUNC(maximum,rational,rational_lt(x,y)?y:x)
// 定义有理数最大值函数
RATIONAL_BINARY_UFUNC(equal,npy_bool,rational_eq(x,y))
// 定义有理数相等判断函数
RATIONAL_BINARY_UFUNC(not_equal,npy_bool,rational_ne(x,y))
// 定义有理数不相等判断函数
RATIONAL_BINARY_UFUNC(less,npy_bool,rational_lt(x,y))
// 定义有理数小于判断函数
RATIONAL_BINARY_UFUNC(greater,npy_bool,rational_gt(x,y))
// 定义有理数大于判断函数
RATIONAL_BINARY_UFUNC(less_equal,npy_bool,rational_le(x,y))
// 定义有理数小于等于判断函数
RATIONAL_BINARY_UFUNC(greater_equal,npy_bool,rational_ge(x,y))
// 定义有理数大于等于判断函数

BINARY_UFUNC(gcd_ufunc,npy_int64,npy_int64,npy_int64,gcd(x,y))
// 定义最大公约数函数
BINARY_UFUNC(lcm_ufunc,npy_int64,npy_int64,npy_int64,lcm(x,y))
// 定义最小公倍数函数

#define UNARY_UFUNC(name,type,exp) \
    // 定义一元通用函数的宏，参数包括函数名、类型以及表达式
    void rational_ufunc_##name(char** args, npy_intp const *dimensions, \
                               npy_intp const *steps, void* data) { \
        // 定义函数，接受字符指针数组args、维度dimensions、步长steps以及数据指针data作为参数
        npy_intp is = steps[0], os = steps[1], n = *dimensions; \
        // 定义输入步长is、输出步长os以及数据点数n
        char *i = args[0], *o = args[1]; \
        // 初始化输入指针i和输出指针o
        int k; \
        // 定义循环计数器k
        for (k = 0; k < n; k++) { \
            // 循环遍历每个数据点
            rational x = *(rational*)i; \
            // 将i解引用为rational类型，并赋值给x
            *(type*)o = exp; \
            // 将exp计算结果转换为type类型，并存储到o指向的位置
            i += is; o += os; \
            // 更新指针，移动到下一个数据点
        } \
    }

UNARY_UFUNC(negative,rational,rational_negative(x))
// 定义有理数负数函数
UNARY_UFUNC(absolute,rational,rational_abs(x))
// 定义有理数绝对值函数
UNARY_UFUNC(floor,rational,make_rational_int(rational_floor(x)))
// 定义有理数向下取整函数
UNARY_UFUNC(ceil,rational,make_rational_int(rational_ceil(x)))
// 定义有理数向上取整函数
UNARY_UFUNC(trunc,rational,make_rational_int(x.n/d(x)))
// 定义有理数截断函数
UNARY_UFUNC(square,rational,rational_multiply(x,x))
// 定义有理数平方函数
UNARY_UFUNC(rint,rational,make_rational_int(rational_rint(x)))
// 定义有理数四舍五入函数
UNARY_UFUNC(sign,rational,make_rational_int(rational_sign(x)))
// 定义有理数符号函数
UNARY_UFUNC(reciprocal,rational,rational_inverse(x))
// 定义有理数倒数函数
UNARY_UFUNC(numerator,npy_int64,x.n)
// 定义有理数分子函数
    /* 定义 UNARY_UFUNC 宏，接受三个参数：denominator、npy_int64 类型、d(x) 参数 */
    UNARY_UFUNC(denominator,npy_int64,d(x))

    /* 内联函数，用于有理数矩阵相乘操作 */
    static inline void
    rational_matrix_multiply(char **args, npy_intp const *dimensions, npy_intp const *steps)
    {
        /* 指向输入和输出数组数据的指针 */
        char *ip1 = args[0];  // 第一个输入数组指针
        char *ip2 = args[1];  // 第二个输入数组指针
        char *op = args[2];   // 输出数组指针

        /* 核心维度的长度 */
        npy_intp dm = dimensions[0];  // 第一个维度的长度
        npy_intp dn = dimensions[1];  // 第二个维度的长度
        npy_intp dp = dimensions[2];  // 第三个维度的长度

        /* 核心维度的步长 */
        npy_intp is1_m = steps[0];  // 第一个输入数组在第一个维度上的步长
        npy_intp is1_n = steps[1];  // 第一个输入数组在第二个维度上的步长
        npy_intp is2_n = steps[2];  // 第二个输入数组在第二个维度上的步长
        npy_intp is2_p = steps[3];  // 第二个输入数组在第三个维度上的步长
        npy_intp os_m = steps[4];   // 输出数组在第一个维度上的步长
        npy_intp os_p = steps[5];   // 输出数组在第三个维度上的步长

        /* 核心维度的计数器 */
        npy_intp m, p;

        /* 计算每行/列向量对的点积 */
        for (m = 0; m < dm; m++) {
            for (p = 0; p < dp; p++) {
                npyrational_dot(ip1, is1_n, ip2, is2_n, op, dn, NULL);

                /* 移动到第二个输入数组和输出数组的下一列 */
                ip2 += is2_p;
                op  += os_p;
            }

            /* 重置到第二个输入数组和输出数组的第一列 */
            ip2 -= is2_p * p;
            op -= os_p * p;

            /* 移动到第一个输入数组和输出数组的下一行 */
            ip1 += is1_m;
            op += os_m;
        }
    }


    /* 通用函数，用于有理数矩阵相乘操作，处理多维数组 */
    static void
    rational_gufunc_matrix_multiply(char **args, npy_intp const *dimensions,
                                    npy_intp const *steps, void *NPY_UNUSED(func))
    {
        /* 外部维度的计数器 */
        npy_intp N_;

        /* 扁平化外部维度的长度 */
        npy_intp dN = dimensions[0];

        /* 扁平化外部维度的步长，用于输入和输出数组 */
        npy_intp s0 = steps[0];
        npy_intp s1 = steps[1];
        npy_intp s2 = steps[2];

        /*
         * 循环遍历外部维度，在每次循环中对核心维度执行矩阵相乘操作
         */
        for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
            rational_matrix_multiply(args, dimensions+1, steps+3);
        }
    }


    /* 测试函数，用于有理数加法操作，处理一维数组 */
    static void
    rational_ufunc_test_add(char** args, npy_intp const *dimensions,
                            npy_intp const *steps, void* data) {
        npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
        char *i0 = args[0], *i1 = args[1], *o = args[2];
        int k;
        for (k = 0; k < n; k++) {
            npy_int64 x = *(npy_int64*)i0;
            npy_int64 y = *(npy_int64*)i1;
            *(rational*)o = rational_add(make_rational_fast(x, 1),
                                         make_rational_fast(y, 1));
            i0 += is0; i1 += is1; o += os;
        }
    }


    /* 测试函数，用于有理数加法操作，处理多维数组 */
    static void
    rational_ufunc_test_add_rationals(char** args, npy_intp const *dimensions,
                                      npy_intp const *steps, void* data) {
        npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
        char *i0 = args[0], *i1 = args[1], *o = args[2];
        int k;
        /* 循环遍历多维数组，对每个元素执行有理数加法操作 */
        for (k = 0; k < n; k++) {
            npy_int64 x = *(npy_int64*)i0;
            npy_int64 y = *(npy_int64*)i1;
            *(rational*)o = rational_add(make_rational_fast(x, 1),
                                         make_rational_fast(y, 1));
            i0 += is0; i1 += is1; o += os;
        }
    }
    # 遍历循环，从 k=0 开始，直到 k<n 结束
    for (k = 0; k < n; k++) {
        # 从指针 i0 处获取 rational 类型的数据，存入变量 x
        rational x = *(rational*)i0;
        # 从指针 i1 处获取 rational 类型的数据，存入变量 y
        rational y = *(rational*)i1;
        # 计算 x 和 y 的和，将结果存入指针 o 指向的位置
        *(rational*)o = rational_add(x, y);
        # 更新指针 i0，移动 is0 个字节，指向下一个输入数据
        i0 += is0;
        # 更新指针 i1，移动 is1 个字节，指向下一个输入数据
        i1 += is1;
        # 更新指针 o，移动 os 个字节，指向下一个输出位置
        o += os;
    }
}

PyMethodDef module_methods[] = {
    {0} /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,       // 定义Python模块的头信息初始化
    "_rational_tests",           // 模块名为 "_rational_tests"
    NULL,                        // 模块的文档字符串为空
    -1,                          // 模块状态为 -1，表示使用子解释器支持
    module_methods,              // 模块方法定义，这里为空，用于定义模块中的函数
    NULL,                        // 模块的全局状态，这里为空
    NULL,                        // 模块的内存分配器，这里为空
    NULL,                        // 模块的清理函数，这里为空
    NULL                         // 模块的析构函数，这里为空
};

PyMODINIT_FUNC PyInit__rational_tests(void) {
    PyObject *m = NULL;          // Python对象指针m，初始化为空
    PyObject* numpy_str;         // Python对象指针numpy_str
    PyObject* numpy;             // Python对象指针numpy
    int npy_rational;            // 整型变量np_rational

    import_array();              // 导入NumPy的数组接口
    if (PyErr_Occurred()) {      // 如果发生了Python错误
        goto fail;               // 跳转到失败标签处
    }
    import_umath();              // 导入NumPy的数学函数
    if (PyErr_Occurred()) {      // 如果发生了Python错误
        goto fail;               // 跳转到失败标签处
    }
    numpy_str = PyUnicode_FromString("numpy");  // 创建字符串对象"numpy"
    if (!numpy_str) {            // 如果字符串对象创建失败
        goto fail;               // 跳转到失败标签处
    }
    numpy = PyImport_Import(numpy_str);  // 导入numpy模块
    Py_DECREF(numpy_str);        // 释放numpy_str的引用计数
    if (!numpy) {                // 如果numpy模块导入失败
        goto fail;               // 跳转到失败标签处
    }

    /* Can't set this until we import numpy */
    PyRational_Type.tp_base = &PyGenericArrType_Type;  // 设置PyRational_Type的基类为PyGenericArrType_Type

    /* Initialize rational type object */
    if (PyType_Ready(&PyRational_Type) < 0) {  // 准备PyRational_Type类型对象
        goto fail;               // 跳转到失败标签处
    }

    /* Initialize rational descriptor */
    PyArray_InitArrFuncs(&npyrational_arrfuncs);  // 初始化npyrational_arrfuncs数组函数集
    npyrational_arrfuncs.getitem = npyrational_getitem;  // 设置npyrational_arrfuncs的getitem函数
    npyrational_arrfuncs.setitem = npyrational_setitem;  // 设置npyrational_arrfuncs的setitem函数
    npyrational_arrfuncs.copyswapn = npyrational_copyswapn;  // 设置npyrational_arrfuncs的copyswapn函数
    npyrational_arrfuncs.copyswap = npyrational_copyswap;  // 设置npyrational_arrfuncs的copyswap函数
    npyrational_arrfuncs.compare = npyrational_compare;  // 设置npyrational_arrfuncs的compare函数
    npyrational_arrfuncs.argmin = npyrational_argmin;  // 设置npyrational_arrfuncs的argmin函数
    npyrational_arrfuncs.argmax = npyrational_argmax;  // 设置npyrational_arrfuncs的argmax函数
    npyrational_arrfuncs.dotfunc = npyrational_dot;  // 设置npyrational_arrfuncs的dotfunc函数
    npyrational_arrfuncs.nonzero = npyrational_nonzero;  // 设置npyrational_arrfuncs的nonzero函数
    npyrational_arrfuncs.fill = npyrational_fill;  // 设置npyrational_arrfuncs的fill函数
    npyrational_arrfuncs.fillwithscalar = npyrational_fillwithscalar;  // 设置npyrational_arrfuncs的fillwithscalar函数
    /* Left undefined: scanfunc, fromstr, sort, argsort */
    Py_SET_TYPE(&npyrational_descr_proto, &PyArrayDescr_Type);  // 设置npyrational_descr_proto的类型为PyArrayDescr_Type
    npy_rational = PyArray_RegisterDataType(&npyrational_descr_proto);  // 注册npyrational_descr_proto作为一种数据类型
    if (npy_rational < 0) {      // 如果注册失败
        goto fail;               // 跳转到失败标签处
    }
    PyArray_Descr *npyrational_descr = PyArray_DescrFromType(npy_rational);  // 根据数据类型创建描述符对象

    /* Support dtype(rational) syntax */
    if (PyDict_SetItemString(PyRational_Type.tp_dict, "dtype",
                             (PyObject*)npyrational_descr) < 0) {  // 将dtype(rational)语法支持添加到PyRational_Type的字典中
        goto fail;               // 跳转到失败标签处
    }

    /* Register casts to and from rational */
    #define REGISTER_CAST(From,To,from_descr,to_typenum,safe) { \
            PyArray_Descr* from_descr_##From##_##To = (from_descr); \
            if (PyArray_RegisterCastFunc(from_descr_##From##_##To, \
                                         (to_typenum), \
                                         npycast_##From##_##To) < 0) { \
                goto fail; \
            } \
            if (safe && PyArray_RegisterCanCast(from_descr_##From##_##To, \
                                                (to_typenum), \
                                                NPY_NOSCALAR) < 0) { \
                goto fail; \
            } \
        }
    #define REGISTER_INT_CASTS(bits) \
        REGISTER_CAST(npy_int##bits, rational, \
                      PyArray_DescrFromType(NPY_INT##bits), npy_rational, 1) \
        REGISTER_CAST(rational, npy_int##bits, npyrational_descr, \
                      NPY_INT##bits, 0)
    注册整数类型的类型转换宏，包括从整数到有理数的转换和从有理数到整数的转换。
    REGISTER_INT_CASTS(8)
    注册8位整数类型的类型转换。
    REGISTER_INT_CASTS(16)
    注册16位整数类型的类型转换。
    REGISTER_INT_CASTS(32)
    注册32位整数类型的类型转换。
    REGISTER_INT_CASTS(64)
    注册64位整数类型的类型转换。
    REGISTER_CAST(rational,float,npyrational_descr,NPY_FLOAT,0)
    注册从有理数到单精度浮点数的类型转换。
    REGISTER_CAST(rational,double,npyrational_descr,NPY_DOUBLE,1)
    注册从有理数到双精度浮点数的类型转换。
    REGISTER_CAST(npy_bool,rational, PyArray_DescrFromType(NPY_BOOL),
                  npy_rational,1)
    注册从布尔类型到有理数的类型转换。
    REGISTER_CAST(rational,npy_bool,npyrational_descr,NPY_BOOL,0)
    注册从有理数到布尔类型的类型转换。
    
    /* Register ufuncs */
    #define REGISTER_UFUNC(name,...) { \
        PyUFuncObject* ufunc = \
            (PyUFuncObject*)PyObject_GetAttrString(numpy, #name); \
        int _types[] = __VA_ARGS__; \
        if (!ufunc) { \
            goto fail; \
        } \
        if (sizeof(_types)/sizeof(int)!=ufunc->nargs) { \
            PyErr_Format(PyExc_AssertionError, \
                         "ufunc %s takes %d arguments, our loop takes %lu", \
                         #name, ufunc->nargs, (unsigned long) \
                         (sizeof(_types)/sizeof(int))); \
            Py_DECREF(ufunc); \
            goto fail; \
        } \
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc, npy_rational, \
                rational_ufunc_##name, _types, 0) < 0) { \
            Py_DECREF(ufunc); \
            goto fail; \
        } \
        Py_DECREF(ufunc); \
    }
    注册通用函数（ufunc），并为其注册有理数类型的循环执行函数。
    
    #define REGISTER_UFUNC_BINARY_RATIONAL(name) \
        REGISTER_UFUNC(name, {npy_rational, npy_rational, npy_rational})
    注册二元有理数通用函数。
    
    #define REGISTER_UFUNC_BINARY_COMPARE(name) \
        REGISTER_UFUNC(name, {npy_rational, npy_rational, NPY_BOOL})
    注册二元比较有理数通用函数。
    
    #define REGISTER_UFUNC_UNARY(name) \
        REGISTER_UFUNC(name, {npy_rational, npy_rational})
    注册一元有理数通用函数。
    
    /* Binary */
    REGISTER_UFUNC_BINARY_RATIONAL(add)
    注册二元有理数加法通用函数。
    REGISTER_UFUNC_BINARY_RATIONAL(subtract)
    注册二元有理数减法通用函数。
    REGISTER_UFUNC_BINARY_RATIONAL(multiply)
    注册二元有理数乘法通用函数。
    REGISTER_UFUNC_BINARY_RATIONAL(divide)
    注册二元有理数除法通用函数。
    REGISTER_UFUNC_BINARY_RATIONAL(remainder)
    注册二元有理数取余通用函数。
    REGISTER_UFUNC_BINARY_RATIONAL(true_divide)
    注册二元有理数真除法通用函数。
    REGISTER_UFUNC_BINARY_RATIONAL(floor_divide)
    注册二元有理数向下取整除法通用函数。
    REGISTER_UFUNC_BINARY_RATIONAL(minimum)
    注册二元有理数最小值通用函数。
    REGISTER_UFUNC_BINARY_RATIONAL(maximum)
    注册二元有理数最大值通用函数。
    
    /* Comparisons */
    REGISTER_UFUNC_BINARY_COMPARE(equal)
    注册二元有理数等于比较通用函数。
    REGISTER_UFUNC_BINARY_COMPARE(not_equal)
    注册二元有理数不等于比较通用函数。
    REGISTER_UFUNC_BINARY_COMPARE(less)
    注册二元有理数小于比较通用函数。
    REGISTER_UFUNC_BINARY_COMPARE(greater)
    注册二元有理数大于比较通用函数。
    REGISTER_UFUNC_BINARY_COMPARE(less_equal)
    注册二元有理数小于等于比较通用函数。
    REGISTER_UFUNC_BINARY_COMPARE(greater_equal)
    注册二元有理数大于等于比较通用函数。
    
    /* Unary */
    REGISTER_UFUNC_UNARY(negative)
    注册一元有理数负数通用函数。
    REGISTER_UFUNC_UNARY(absolute)
    注册一元有理数绝对值通用函数。
    REGISTER_UFUNC_UNARY(floor)
    注册一元有理数向下取整通用函数。
    REGISTER_UFUNC_UNARY(ceil)
    注册一元有理数向上取整通用函数。
    REGISTER_UFUNC_UNARY(trunc)
    注册一元有理数截断通用函数。
    REGISTER_UFUNC_UNARY(rint)
    注册一元有理数四舍五入通用函数。
    REGISTER_UFUNC_UNARY(square)
    注册一元有理数平方通用函数。
    REGISTER_UFUNC_UNARY(reciprocal)
    注册一元有理数倒数通用函数。
    REGISTER_UFUNC_UNARY(sign)
    注册一元有理数符号通用函数。
    
    /* Create module */
    m = PyModule_Create(&moduledef);
    创建 Python 模块对象并使用指定的模块定义初始化它。
    if (!m) {
        // 如果模块指针 m 为 NULL，则跳转到失败标签
        goto fail;
    }

    /* Add rational type */
    // 增加有理数类型到模块 m 中
    Py_INCREF(&PyRational_Type);
    PyModule_AddObject(m,"rational",(PyObject*)&PyRational_Type);

    /* Create matrix multiply generalized ufunc */
    // 创建矩阵乘法通用函数
    {
        // 定义通用函数支持的数据类型数组
        int types2[3] = {npy_rational,npy_rational,npy_rational};
        // 创建矩阵乘法通用函数对象 gufunc
        PyObject* gufunc = PyUFunc_FromFuncAndDataAndSignature(0,0,0,0,2,1,
            PyUFunc_None,"matrix_multiply",
            "return result of multiplying two matrices of rationals",
            0,"(m,n),(n,p)->(m,p)");
        // 如果创建失败，则跳转到失败标签
        if (!gufunc) {
            goto fail;
        }
        // 将有理数类型的循环注册到 gufunc 中
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)gufunc, npy_rational,
                rational_gufunc_matrix_multiply, types2, 0) < 0) {
            goto fail;
        }
        // 将 gufunc 对象添加到模块 m 中
        PyModule_AddObject(m,"matrix_multiply",(PyObject*)gufunc);
    }

    /* Create test ufunc with built in input types and rational output type */
    // 创建具有内置输入类型和有理数输出类型的测试通用函数
    {
        // 定义测试通用函数支持的数据类型数组
        int types3[3] = {NPY_INT64,NPY_INT64,npy_rational};
        // 创建测试通用函数对象 ufunc
        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,2,1,
                PyUFunc_None,"test_add",
                "add two matrices of int64 and return rational matrix",0);
        // 如果创建失败，则跳转到失败标签
        if (!ufunc) {
            goto fail;
        }
        // 将有理数类型的循环注册到 ufunc 中
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc, npy_rational,
                rational_ufunc_test_add, types3, 0) < 0) {
            goto fail;
        }
        // 将 ufunc 对象添加到模块 m 中
        PyModule_AddObject(m,"test_add",(PyObject*)ufunc);
    }

    /* Create test ufunc with rational types using RegisterLoopForDescr */
    // 使用 RegisterLoopForDescr 创建具有有理数类型的测试通用函数
    {
        // 创建测试通用函数对象 ufunc
        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,2,1,
                PyUFunc_None,"test_add_rationals",
                "add two matrices of rationals and return rational matrix",0);
        // 定义用于描述器的数据类型数组
        PyArray_Descr* types[3] = {npyrational_descr,
                                    npyrational_descr,
                                    npyrational_descr};
        // 如果创建失败，则跳转到失败标签
        if (!ufunc) {
            goto fail;
        }
        // 将有理数类型的循环注册到 ufunc 中
        if (PyUFunc_RegisterLoopForDescr((PyUFuncObject*)ufunc, npyrational_descr,
                rational_ufunc_test_add_rationals, types, 0) < 0) {
            goto fail;
        }
        // 将 ufunc 对象添加到模块 m 中
        PyModule_AddObject(m,"test_add_rationals",(PyObject*)ufunc);
    }

    /* Create numerator and denominator ufuncs */
    // 创建分子和分母的通用函数
    #define NEW_UNARY_UFUNC(name,type,doc) { \
        // 定义通用函数支持的数据类型数组
        int types[2] = {npy_rational,type}; \
        // 创建通用函数对象 ufunc
        PyObject* ufunc = PyUFunc_FromFuncAndData(0,0,0,0,1,1, \
            PyUFunc_None,#name,doc,0); \
        // 如果创建失败，则跳转到失败标签
        if (!ufunc) { \
            goto fail; \
        } \
        // 将有理数类型的循环注册到 ufunc 中
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc, \
                npy_rational,rational_ufunc_##name,types,0)<0) { \
            goto fail; \
        } \
        // 将 ufunc 对象添加到模块 m 中
        PyModule_AddObject(m,#name,(PyObject*)ufunc); \
    }
    // 创建分子通用函数对象并添加到模块 m 中
    NEW_UNARY_UFUNC(numerator,NPY_INT64,"rational number numerator");
    // 创建分母通用函数对象并添加到模块 m 中
    NEW_UNARY_UFUNC(denominator,NPY_INT64,"rational number denominator");

    /* Create gcd and lcm ufuncs */
    # 定义宏 GCD_LCM_UFUNC，用于创建并注册名为 gcd 和 lcm 的通用函数对象到模块 m 中
    # 这些函数处理整数类型 NPY_INT64，执行特定功能并提供相关文档说明

    # 宏的展开部分，创建函数指针数组 func 和类型数组 types
    # func 包含名为 gcd_ufunc 或 lcm_ufunc 的函数指针
    # types 指定了函数接受和返回的参数类型都是 NPY_INT64

    # 初始化数据指针数组 data，这里设为 NULL

    # 使用 PyUFunc_FromFuncAndData 创建通用函数对象 ufunc
    # 参数依次是 func、data、types、1、2、1、PyUFunc_One、#name、doc、0
    # 如果创建失败，则跳转到标签 fail 处理错误

    # 将创建的 ufunc 对象添加到模块 m 中，名称分别为 "gcd" 和 "lcm"

    GCD_LCM_UFUNC(gcd,NPY_INT64,"greatest common denominator of two integers");
    GCD_LCM_UFUNC(lcm,NPY_INT64,"least common multiple of two integers");

    # 返回创建好的模块对象 m
    return m;
fail:
    # 如果当前没有 Python 异常发生
    if (!PyErr_Occurred()) {
        # 设置一个运行时错误的异常字符串，说明无法加载 _rational_tests 模块
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load _rational_tests module.");
    }
    # 如果 m 非空（即 m 模块对象存在）
    if (m) {
        # 释放 m 模块对象的引用计数，并将 m 设置为 NULL
        Py_DECREF(m);
        m = NULL;
    }
    # 返回 m 模块对象（可能为 NULL）
    return m;
}
```