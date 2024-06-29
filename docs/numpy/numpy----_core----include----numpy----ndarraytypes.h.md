# `.\numpy\numpy\_core\include\numpy\ndarraytypes.h`

```py
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NDARRAYTYPES_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NDARRAYTYPES_H_

// 包含必要的头文件：npy_common.h, npy_endian.h, npy_cpu.h, utils.h
#include "npy_common.h"
#include "npy_endian.h"
#include "npy_cpu.h"
#include "utils.h"

// 定义一个不导出的宏，用于内部使用
#define NPY_NO_EXPORT NPY_VISIBILITY_HIDDEN

// 根据编译时是否启用多线程（NPY_NO_SMP），确定是否允许使用线程
#if !NPY_NO_SMP
        #define NPY_ALLOW_THREADS 1
#else
        #define NPY_ALLOW_THREADS 0
#endif

// 如果编译环境不支持 __has_extension 宏，则定义为 0
#ifndef __has_extension
#define __has_extension(x) 0
#endif

/*
 * There are several places in the code where an array of dimensions
 * is allocated statically.  This is the size of that static
 * allocation.
 *
 * The array creation itself could have arbitrary dimensions but all
 * the places where static allocation is used would need to be changed
 * to dynamic (including inside of several structures)
 *
 * As of NumPy 2.0, we strongly discourage the downstream use of NPY_MAXDIMS,
 * but since auditing everything seems a big ask, define it as 64.
 * A future version could:
 * - Increase or remove the limit and require recompilation (like 2.0 did)
 * - Deprecate or remove the macro but keep the limit (at basically any time)
 */
// 定义静态分配的维度数组大小上限
#define NPY_MAXDIMS 64
// 不可改变的遗留迭代器的最大维度限制
#define NPY_MAXDIMS_LEGACY_ITERS 32
// NPY_MAXARGS 是版本相关的，定义在 npy_2_compat.h 中

/* Used for Converter Functions "O&" code in ParseTuple */
// 用于解析元组中 "O&" 类型参数的转换器函数返回状态
#define NPY_FAIL 0
#define NPY_SUCCEED 1

#endif  // NUMPY_CORE_INCLUDE_NUMPY_NDARRAYTYPES_H_
/* 枚举定义不同的 NumPy 数据类型 */
enum NPY_TYPES {
    NPY_BOOL=0,             /* 布尔型 */
    NPY_BYTE, NPY_UBYTE,    /* 有符号和无符号字节型 */
    NPY_SHORT, NPY_USHORT,  /* 有符号和无符号短整型 */
    NPY_INT, NPY_UINT,      /* 有符号和无符号整型 */
    NPY_LONG, NPY_ULONG,    /* 有符号和无符号长整型 */
    NPY_LONGLONG, NPY_ULONGLONG,  /* 有符号和无符号长长整型 */
    NPY_FLOAT, NPY_DOUBLE,  /* 单精度和双精度浮点型 */
    NPY_LONGDOUBLE,         /* 长双精度浮点型 */
    NPY_CFLOAT, NPY_CDOUBLE,/* 单复数和双复数浮点型 */
    NPY_CLONGDOUBLE,        /* 长双复数浮点型 */
    NPY_OBJECT=17,          /* Python 对象 */
    NPY_STRING, NPY_UNICODE,/* 字符串和Unicode类型 */
    NPY_VOID,               /* Void（空）类型 */

    /* 以下为1.6版后添加的新类型，可能在2.0版中整合到以上类型中 */
    NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,

    NPY_CHAR,               /* 不推荐使用，如果使用会报错 */

    /* *legacy* 类型的数量 */
    NPY_NTYPES_LEGACY=24,

    /* 避免将来添加新类型时改变 */
    NPY_NOTYPE=25,

    NPY_USERDEF=256,        /* 留出空间用于用户定义类型 */

    /* 不包括1.6版后添加的新类型的类型数量 */
    NPY_NTYPES_ABI_COMPATIBLE=21,

    /*
     * 2.0版后添加的新 DType，不共享传统布局
     * VSTRING 是这些类型中的第一个，将来可能为用户定义 DType 开辟一个块
     */
    NPY_VSTRING=2056,
};

/* 基本类型数组优先级 */
#define NPY_PRIORITY 0.0

/* 默认子类型优先级 */
#define NPY_SUBTYPE_PRIORITY 1.0

/* 默认标量优先级 */
#define NPY_SCALAR_PRIORITY -1000000.0

/* 浮点类型的数量（不包括 half） */
#define NPY_NUM_FLOATTYPE 3

/*
 * 这些字符对应于数组类型和 struct 模块
 */
/*
 * 定义了 Numpy 数组数据类型的字符表示和对应的枚举值
 */
enum NPY_TYPECHAR {
        NPY_BOOLLTR = '?',              // 布尔类型
        NPY_BYTELTR = 'b',              // 字节类型
        NPY_UBYTELTR = 'B',             // 无符号字节类型
        NPY_SHORTLTR = 'h',             // 短整型
        NPY_USHORTLTR = 'H',            // 无符号短整型
        NPY_INTLTR = 'i',               // 整型
        NPY_UINTLTR = 'I',              // 无符号整型
        NPY_LONGLTR = 'l',              // 长整型
        NPY_ULONGLTR = 'L',             // 无符号长整型
        NPY_LONGLONGLTR = 'q',          // 长长整型
        NPY_ULONGLONGLTR = 'Q',         // 无符号长长整型
        NPY_HALFLTR = 'e',              // 半精度浮点型
        NPY_FLOATLTR = 'f',             // 单精度浮点型
        NPY_DOUBLELTR = 'd',            // 双精度浮点型
        NPY_LONGDOUBLELTR = 'g',        // 长双精度浮点型
        NPY_CFLOATLTR = 'F',            // 复数类型（单精度）
        NPY_CDOUBLELTR = 'D',           // 复数类型（双精度）
        NPY_CLONGDOUBLELTR = 'G',       // 复数类型（长双精度）
        NPY_OBJECTLTR = 'O',            // Python 对象
        NPY_STRINGLTR = 'S',            // 字符串
        NPY_DEPRECATED_STRINGLTR2 = 'a',// 废弃的字符串类型
        NPY_UNICODELTR = 'U',           // Unicode 字符串
        NPY_VOIDLTR = 'V',              // 任意数据类型（void）
        NPY_DATETIMELTR = 'M',          // 日期时间类型
        NPY_TIMEDELTALTR = 'm',         // 时间增量类型
        NPY_CHARLTR = 'c',              // 字符类型

        /*
         * 新的非遗留数据类型
         */
        NPY_VSTRINGLTR = 'T',           // 可变长度字符串

        /*
         * 注意，我们移除了 `NPY_INTPLTR`，因为我们将其定义
         * 改为 'n'，而不再使用 'p'。在大多数平台上，这是相同的整数。
         * 对于与 `size_t` 相同大小的 `np.intp`，应该使用 'n'，
         * 而 'p' 仍然表示指针大小。
         *
         * 'p', 'P', 'n', 和 'N' 都是有效的，并在 `arraytypes.c.src` 中明确定义。
         */

        /*
         * 这些用于 dtype 的 '种类'，而不是上面的 '类型码'。
         */
        NPY_GENBOOLLTR ='b',            // 通用布尔类型
        NPY_SIGNEDLTR = 'i',            // 有符号类型
        NPY_UNSIGNEDLTR = 'u',          // 无符号类型
        NPY_FLOATINGLTR = 'f',          // 浮点类型
        NPY_COMPLEXLTR = 'c',           // 复数类型

};

/*
 * 改动可能会破坏 Numpy API 的兼容性，
 * 因为会改变 PyArray_ArrFuncs 中的偏移量，因此需要小心。
 * 在这里，我们重复使用 mergesort 插槽来实现任何类型的稳定排序，
 * 实际实现将依赖于数据类型。
 */
typedef enum {
        _NPY_SORT_UNDEFINED=-1,         // 未定义的排序类型
        NPY_QUICKSORT=0,                // 快速排序
        NPY_HEAPSORT=1,                 // 堆排序
        NPY_MERGESORT=2,                // 归并排序
        NPY_STABLESORT=2,               // 稳定排序
} NPY_SORTKIND;
#define NPY_NSORTS (NPY_STABLESORT + 1)


typedef enum {
        NPY_INTROSELECT=0               // 引导选择算法
} NPY_SELECTKIND;
#define NPY_NSELECTS (NPY_INTROSELECT + 1)


typedef enum {
        NPY_SEARCHLEFT=0,               // 向左搜索
        NPY_SEARCHRIGHT=1               // 向右搜索
} NPY_SEARCHSIDE;
#define NPY_NSEARCHSIDES (NPY_SEARCHRIGHT + 1)


typedef enum {
        NPY_NOSCALAR=-1,                // 非标量
        NPY_BOOL_SCALAR,                // 布尔标量
        NPY_INTPOS_SCALAR,              // 正整数标量
        NPY_INTNEG_SCALAR,              // 负整数标量
        NPY_FLOAT_SCALAR,               // 浮点数标量
        NPY_COMPLEX_SCALAR,             // 复数标量
        NPY_OBJECT_SCALAR               // 对象标量
} NPY_SCALARKIND;
#define NPY_NSCALARKINDS (NPY_OBJECT_SCALAR + 1)

/* 用于指定数组内存布局或迭代顺序 */
typedef enum {
        /* 如果输入都是 Fortran，则使用 Fortran 顺序，否则使用 C 顺序 */
        NPY_ANYORDER=-1,                // 任意顺序
        /* C 顺序 */
        NPY_CORDER=0,                   // C 顺序
        /* Fortran 顺序 */
        NPY_FORTRANORDER=1,             // Fortran 顺序
        /* 尽可能接近输入的顺序 */
        NPY_KEEPORDER=2                 // 保持原顺序
} NPY_ORDER;

/* 用于指定在支持的操作中允许的类型转换 */
/*
 * 定义枚举类型 NPY_CASTING，用于指定数组类型转换时的转换规则
 */
typedef enum {
        _NPY_ERROR_OCCURRED_IN_CAST = -1,
        /* 只允许相同类型的转换 */
        NPY_NO_CASTING=0,
        /* 允许相同和字节交换类型的转换 */
        NPY_EQUIV_CASTING=1,
        /* 只允许安全的转换 */
        NPY_SAFE_CASTING=2,
        /* 允许安全转换或者同种类型的转换 */
        NPY_SAME_KIND_CASTING=3,
        /* 允许任意类型的转换 */
        NPY_UNSAFE_CASTING=4,
} NPY_CASTING;

/*
 * 定义枚举类型 NPY_CLIPMODE，用于指定数组运算中的截断模式
 */
typedef enum {
        NPY_CLIP=0,
        NPY_WRAP=1,
        NPY_RAISE=2
} NPY_CLIPMODE;

/*
 * 定义枚举类型 NPY_CORRELATEMODE，用于指定相关运算的模式
 */
typedef enum {
        NPY_VALID=0,
        NPY_SAME=1,
        NPY_FULL=2
} NPY_CORRELATEMODE;

/* DATETIME 类型的特殊值，表示非时间值 (NaT) */
#define NPY_DATETIME_NAT NPY_MIN_INT64

/*
 * DATETIME ISO 8601 字符串的最大长度上限
 *   YEAR: 21 (64-bit year)
 *   MONTH: 3
 *   DAY: 3
 *   HOURS: 3
 *   MINUTES: 3
 *   SECONDS: 3
 *   ATTOSECONDS: 1 + 3*6
 *   TIMEZONE: 5
 *   NULL TERMINATOR: 1
 */
#define NPY_DATETIME_MAX_ISO8601_STRLEN (21 + 3*5 + 1 + 3*6 + 6 + 1)

/*
 * 枚举类型 NPY_DATETIMEUNIT，定义了日期时间单位
 *   NPY_FR_ERROR = -1: 错误或未确定
 *   NPY_FR_Y = 0: 年
 *   NPY_FR_M = 1: 月
 *   NPY_FR_W = 2: 周
 *   NPY_FR_D = 4: 日
 *   NPY_FR_h = 5: 小时
 *   NPY_FR_m = 6: 分钟
 *   NPY_FR_s = 7: 秒
 *   NPY_FR_ms = 8: 毫秒
 *   NPY_FR_us = 9: 微秒
 *   NPY_FR_ns = 10: 纳秒
 *   NPY_FR_ps = 11: 皮秒
 *   NPY_FR_fs = 12: 飞秒
 *   NPY_FR_as = 13: 阿托秒
 *   NPY_FR_GENERIC = 14: 未限定单位，可以转换为任何单位
 */
typedef enum {
        /* 强制为有符号的枚举类型，必须为 -1 以保持代码兼容性 */
        NPY_FR_ERROR = -1,      /* 错误或未确定 */
        /* 有效单位开始 */
        NPY_FR_Y = 0,           /* 年 */
        NPY_FR_M = 1,           /* 月 */
        NPY_FR_W = 2,           /* 周 */
        /* 1.6 版本中的 NPY_FR_B (值为 3) 的间隔 */
        NPY_FR_D = 4,           /* 日 */
        NPY_FR_h = 5,           /* 小时 */
        NPY_FR_m = 6,           /* 分钟 */
        NPY_FR_s = 7,           /* 秒 */
        NPY_FR_ms = 8,          /* 毫秒 */
        NPY_FR_us = 9,          /* 微秒 */
        NPY_FR_ns = 10,         /* 纳秒 */
        NPY_FR_ps = 11,         /* 皮秒 */
        NPY_FR_fs = 12,         /* 飞秒 */
        NPY_FR_as = 13,         /* 阿托秒 */
        NPY_FR_GENERIC = 14     /* 未限定单位，可以转换为任何单位 */
} NPY_DATETIMEUNIT;

/*
 * NPY_DATETIME_NUMUNITS 的定义，为了与 1.6 ABI 兼容性，实际单位数比此值少一个
 */
#define NPY_DATETIME_NUMUNITS (NPY_FR_GENERIC + 1)
#define NPY_DATETIME_DEFAULTUNIT NPY_FR_GENERIC

/*
 * 工作日约定，用于将无效的工作日映射为有效的工作日
 */
typedef enum {
    /* 前进到下一个工作日 */
    NPY_BUSDAY_FORWARD,
    NPY_BUSDAY_FOLLOWING = NPY_BUSDAY_FORWARD,
    /* 后退到前一个工作日 */
    NPY_BUSDAY_BACKWARD,
    NPY_BUSDAY_PRECEDING = NPY_BUSDAY_BACKWARD,
    /*
     * 前进到下一个工作日，除非跨越月边界，否则后退
     */
    NPY_BUSDAY_MODIFIEDFOLLOWING,
    /*
     * 后退到前一个工作日，除非跨越月边界，否则前进
     */
    NPY_BUSDAY_MODIFIEDPRECEDING,
    /* 在非工作日返回 NaT（Not a Time）值。 */
    NPY_BUSDAY_NAT,
    /* 在非工作日抛出异常。 */
    NPY_BUSDAY_RAISE
} NPY_BUSDAY_ROLL;



/************************************************************
 * NumPy Auxiliary Data for inner loops, sort functions, etc.
 ************************************************************/



/*
 * When creating an auxiliary data struct, this should always appear
 * as the first member, like this:
 *
 * typedef struct {
 *     NpyAuxData base;
 *     double constant;
 * } constant_multiplier_aux_data;
 */
typedef struct NpyAuxData_tag NpyAuxData;



/* Function pointers for freeing or cloning auxiliary data */
typedef void (NpyAuxData_FreeFunc) (NpyAuxData *);
typedef NpyAuxData *(NpyAuxData_CloneFunc) (NpyAuxData *);



struct NpyAuxData_tag {
    NpyAuxData_FreeFunc *free;
    NpyAuxData_CloneFunc *clone;
    /* To allow for a bit of expansion without breaking the ABI */
    void *reserved[2];
};



/* Macros to use for freeing and cloning auxiliary data */
#define NPY_AUXDATA_FREE(auxdata) \
    do { \
        if ((auxdata) != NULL) { \
            (auxdata)->free(auxdata); \
        } \
    } while(0)

#define NPY_AUXDATA_CLONE(auxdata) \
    ((auxdata)->clone(auxdata))



#define NPY_ERR(str) fprintf(stderr, #str); fflush(stderr);
#define NPY_ERR2(str) fprintf(stderr, str); fflush(stderr);



/*
* Macros to define how array, and dimension/strides data is
* allocated. These should be made private
*/

#define NPY_USE_PYMEM 1



#if NPY_USE_PYMEM == 1
/* use the Raw versions which are safe to call with the GIL released */
#define PyArray_malloc PyMem_RawMalloc
#define PyArray_free PyMem_RawFree
#define PyArray_realloc PyMem_RawRealloc
#else
#define PyArray_malloc malloc
#define PyArray_free free
#define PyArray_realloc realloc
#endif



/* Dimensions and strides */
#define PyDimMem_NEW(size)                                         \
    ((npy_intp *)PyArray_malloc(size*sizeof(npy_intp)))

#define PyDimMem_FREE(ptr) PyArray_free(ptr)

#define PyDimMem_RENEW(ptr,size)                                   \
        ((npy_intp *)PyArray_realloc(ptr,size*sizeof(npy_intp)))



/* forward declaration */
struct _PyArray_Descr;



/* These must deal with unaligned and swapped data if necessary */
typedef PyObject * (PyArray_GetItemFunc) (void *, void *);
typedef int (PyArray_SetItemFunc)(PyObject *, void *, void *);

typedef void (PyArray_CopySwapNFunc)(void *, npy_intp, void *, npy_intp,
                                     npy_intp, int, void *);

typedef void (PyArray_CopySwapFunc)(void *, void *, int, void *);
typedef npy_bool (PyArray_NonzeroFunc)(void *, void *);



/*
 * These assume aligned and notswapped data -- a buffer will be used
 * before or contiguous data will be obtained
 */

typedef int (PyArray_CompareFunc)(const void *, const void *, void *);
typedef int (PyArray_ArgFunc)(void*, npy_intp, npy_intp*, void *);

typedef void (PyArray_DotFunc)(void *, npy_intp, void *, npy_intp, void *,
                               npy_intp, void *);
/*
 * 声明一个函数指针类型 PyArray_VectorUnaryFunc，接受四个参数:
 *   - void *: 指向输入数组的指针
 *   - void *: 指向输出数组的指针
 *   - npy_intp: 表示数组的长度或者维度
 *   - void *: 其他可能的参数
 */
typedef void (PyArray_VectorUnaryFunc)(void *, void *, npy_intp, void *,
                                       void *);

/*
 * 声明一个函数指针类型 PyArray_ScanFunc，接受四个参数:
 *   - FILE *: 指向文件的指针
 *   - void *: 指向数据的指针
 *   - char *: 分隔符参数（不再使用，保留为了向后兼容）
 *   - struct _PyArray_Descr *: 数组描述符结构体的指针
 */
typedef int (PyArray_ScanFunc)(FILE *fp, void *dptr,
                               char *ignore, struct _PyArray_Descr *);

/*
 * 声明一个函数指针类型 PyArray_FromStrFunc，接受四个参数:
 *   - char *: 指向输入字符串的指针
 *   - void *: 指向数据的指针
 *   - char **: 指向结束位置的指针，用于指示解析的结束位置
 *   - struct _PyArray_Descr *: 数组描述符结构体的指针
 */
typedef int (PyArray_FromStrFunc)(char *s, void *dptr, char **endptr,
                                  struct _PyArray_Descr *);

/*
 * 声明一个函数指针类型 PyArray_FillFunc，接受三个参数:
 *   - void *: 指向数据的指针，用于填充数组
 *   - npy_intp: 表示数组的长度或者维度
 *   - void *: 其他可能的参数
 */
typedef int (PyArray_FillFunc)(void *, npy_intp, void *);

/*
 * 声明一个函数指针类型 PyArray_SortFunc，接受三个参数:
 *   - void *: 指向数组的指针，表示要排序的数组
 *   - npy_intp: 表示数组的长度或者维度
 *   - void *: 其他可能的参数
 */
typedef int (PyArray_SortFunc)(void *, npy_intp, void *);

/*
 * 声明一个函数指针类型 PyArray_ArgSortFunc，接受四个参数:
 *   - void *: 指向数组的指针，表示要排序的数组
 *   - npy_intp *: 指向整数数组的指针，用于存储排序后的索引
 *   - npy_intp: 表示数组的长度或者维度
 *   - void *: 其他可能的参数
 */
typedef int (PyArray_ArgSortFunc)(void *, npy_intp *, npy_intp, void *);

/*
 * 声明一个函数指针类型 PyArray_FillWithScalarFunc，接受四个参数:
 *   - void *: 指向数据的指针，用于填充数组
 *   - npy_intp: 表示数组的长度或者维度
 *   - void *: 指向标量数据的指针，用于填充数组
 *   - void *: 其他可能的参数
 */
typedef int (PyArray_FillWithScalarFunc)(void *, npy_intp, void *, void *);

/*
 * 声明一个函数指针类型 PyArray_ScalarKindFunc，接受一个参数:
 *   - void *: 指向数据的指针，表示要查询的标量类型
 */
typedef int (PyArray_ScalarKindFunc)(void *);

/*
 * 定义一个结构体 PyArray_Dims，包含两个成员变量:
 *   - npy_intp *: 指向整数数组的指针，用于存储维度信息
 *   - int: 表示数组的维度数量
 */
typedef struct {
        npy_intp *ptr;
        int len;
} PyArray_Dims;
typedef struct {
        /*
         * Functions to cast to most other standard types
         * Can have some NULL entries. The types
         * DATETIME, TIMEDELTA, and HALF go into the castdict
         * even though they are built-in.
         */
        PyArray_VectorUnaryFunc *cast[NPY_NTYPES_ABI_COMPATIBLE];

        /* The next four functions *cannot* be NULL */

        /*
         * Functions to get and set items with standard Python types
         * -- not array scalars
         */
        PyArray_GetItemFunc *getitem;
        PyArray_SetItemFunc *setitem;

        /*
         * Copy and/or swap data.  Memory areas may not overlap
         * Use memmove first if they might
         */
        PyArray_CopySwapNFunc *copyswapn;
        PyArray_CopySwapFunc *copyswap;

        /*
         * Function to compare items
         * Can be NULL
         */
        PyArray_CompareFunc *compare;

        /*
         * Function to select largest
         * Can be NULL
         */
        PyArray_ArgFunc *argmax;

        /*
         * Function to compute dot product
         * Can be NULL
         */
        PyArray_DotFunc *dotfunc;

        /*
         * Function to scan an ASCII file and
         * place a single value plus possible separator
         * Can be NULL
         */
        PyArray_ScanFunc *scanfunc;

        /*
         * Function to read a single value from a string
         * and adjust the pointer; Can be NULL
         */
        PyArray_FromStrFunc *fromstr;

        /*
         * Function to determine if data is zero or not
         * If NULL a default version is
         * used at Registration time.
         */
        PyArray_NonzeroFunc *nonzero;

        /*
         * Used for arange. Should return 0 on success
         * and -1 on failure.
         * Can be NULL.
         */
        PyArray_FillFunc *fill;

        /*
         * Function to fill arrays with scalar values
         * Can be NULL
         */
        PyArray_FillWithScalarFunc *fillwithscalar;

        /*
         * Sorting functions
         * Can be NULL
         */
        PyArray_SortFunc *sort[NPY_NSORTS];
        PyArray_ArgSortFunc *argsort[NPY_NSORTS];

        /*
         * Dictionary of additional casting functions
         * PyArray_VectorUnaryFuncs
         * which can be populated to support casting
         * to other registered types. Can be NULL
         */
        PyObject *castdict;

        /*
         * Functions useful for generalizing
         * the casting rules.
         * Can be NULL;
         */
        PyArray_ScalarKindFunc *scalarkind;
        int **cancastscalarkindto;
        int *cancastto;

        void *_unused1;
        void *_unused2;
        void *_unused3;

        /*
         * Function to select smallest
         * Can be NULL
         */
        PyArray_ArgFunc *argmin;

} PyArray_ArrFuncs;


/* The item must be reference counted when it is inserted or extracted. */
#define NPY_ITEM_REFCOUNT   0x01
/*
 * 定义了一些常量，用于描述数据类型的特征标志位。
 * 这些标志位用于指示数据类型的特性，如是否包含对象、是否需要初始化等。
 */
#define NPY_ITEM_HASOBJECT  0x01        /* Same as needing REFCOUNT */
#define NPY_LIST_PICKLE     0x02        /* Convert to list for pickling */
#define NPY_ITEM_IS_POINTER 0x04        /* The item is a POINTER  */
#define NPY_NEEDS_INIT      0x08        /* memory needs to be initialized for this data-type */
#define NPY_NEEDS_PYAPI     0x10        /* operations need Python C-API so don't give-up thread. */
#define NPY_USE_GETITEM     0x20        /* Use f.getitem when extracting elements of this data-type */
#define NPY_USE_SETITEM     0x40        /* Use f.setitem when setting creating 0-d array from this data-type.*/
#define NPY_ALIGNED_STRUCT  0x80        /* A sticky flag specifically for structured arrays */

/*
 * 这些标志位是针对全局数据类型的继承标志位，如果字段中的任何数据类型具有这些标志位，则其继承。
 */
#define NPY_FROM_FIELDS    (NPY_NEEDS_INIT | NPY_LIST_PICKLE | \
                            NPY_ITEM_REFCOUNT | NPY_NEEDS_PYAPI)

#define NPY_OBJECT_DTYPE_FLAGS (NPY_LIST_PICKLE | NPY_USE_GETITEM | \
                                NPY_ITEM_IS_POINTER | NPY_ITEM_REFCOUNT | \
                                NPY_NEEDS_INIT | NPY_NEEDS_PYAPI)

#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
/*
 * 如果 NumPy 版本符合 2.x 及以上，使用公共的 PyArray_Descr 结构体。
 */
typedef struct _PyArray_Descr {
        PyObject_HEAD
        /*
         * typeobj 表示该类型的实例的类型对象，不应该有两个 type_number 指向相同的类型对象。
         */
        PyTypeObject *typeobj;
        /* 表示该类型的种类 */
        char kind;
        /* 表示该类型的唯一字符 */
        char type;
        /*
         * '>' (big), '<' (little), '|' (not-applicable), 或 '=' (native)，
         * 表示字节顺序。
         */
        char byteorder;
        /* 旧的标志位字段（未使用），用于确保 type_num 是稳定的。 */
        char _former_flags;
        /* 表示该类型的数字 */
        int type_num;
        /* 用于特定 dtype 实例的标志位 */
        npy_uint64 flags;
        /* 该类型的元素大小（itemsize） */
        npy_intp elsize;
        /* 该类型所需的对齐方式 */
        npy_intp alignment;
        /* 元数据字典或 NULL */
        PyObject *metadata;
        /* 缓存的哈希值（如果尚未计算则为 -1） */
        npy_hash_t hash;
        /* 未使用的插槽（必须初始化为 NULL），用于将来的扩展 */
        void *reserved_null[2];
} PyArray_Descr;

#else  /* 兼容 1.x 和 2.x 的版本（仅共享字段）: */

typedef struct _PyArray_Descr {
        PyObject_HEAD
        PyTypeObject *typeobj;
        char kind;
        char type;
        char byteorder;
        char _former_flags;
        int type_num;
} PyArray_Descr;

/* 要访问修改后的字段，请定义完整的 2.0 版本结构体： */

#endif
/*
 * 定义了一个名为 _PyArray_DescrNumPy2 的结构体，表示 NumPy 2.x 中的数组描述符。
 * 此结构体包含了用于描述数组数据类型的各种字段和元数据。
 * 是 NumPy 1.x 和 2.x 兼容版本的一部分。
 */
typedef struct {
        PyObject_HEAD
        PyTypeObject *typeobj;   // 指向数组数据类型对象的指针
        char kind;               // 数据类型的种类
        char type;               // 数据类型的具体类型
        char byteorder;          // 数据的字节顺序
        char _former_flags;      // 先前的标志位
        int type_num;            // 数据类型的编号
        npy_uint64 flags;        // 数组描述符的标志
        npy_intp elsize;         // 每个数组元素的大小
        npy_intp alignment;      // 数组数据的对齐方式
        PyObject *metadata;      // 数组描述符的元数据
        npy_hash_t hash;         // 数组描述符的哈希值
        void *reserved_null[2];  // 保留字段，暂未使用
} _PyArray_DescrNumPy2;

#endif  /* 1.x and 2.x compatible version */

/*
 * 半私有结构体 _PyArray_LegacyDescr，包含了遗留描述符的额外字段
 * 在进行类型转换或访问之前必须检查 NPY_DT_is_legacy。
 * 当运行在 1.x 版本时，结构体不是有效的公共 API 使用。
 */
typedef struct {
        PyObject_HEAD
        PyTypeObject *typeobj;   // 指向数组数据类型对象的指针
        char kind;               // 数据类型的种类
        char type;               // 数据类型的具体类型
        char byteorder;          // 数据的字节顺序
        char _former_flags;      // 先前的标志位
        int type_num;            // 数据类型的编号
        npy_uint64 flags;        // 数组描述符的标志
        npy_intp elsize;         // 每个数组元素的大小
        npy_intp alignment;      // 数组数据的对齐方式
        PyObject *metadata;      // 数组描述符的元数据
        npy_hash_t hash;         // 数组描述符的哈希值
        void *reserved_null[2];  // 保留字段，暂未使用
        struct _arr_descr *subarray;  // 子数组的描述符
        PyObject *fields;        // 数组字段的 Python 对象
        PyObject *names;         // 数组字段的名称
        NpyAuxData *c_metadata;  // C 风格的元数据
} _PyArray_LegacyDescr;


/*
 * PyArray_DescrProto 结构体，未修改的 PyArray_Descr 结构体，与 NumPy 1.x 中的版本完全相同。
 * 用作注册新的遗留数据类型的原型。
 * 在 1.x 版本中，也用于访问用户代码中的字段。
 */
typedef struct {
        PyObject_HEAD
        PyTypeObject *typeobj;   // 指向数组数据类型对象的指针
        char kind;               // 数据类型的种类
        char type;               // 数据类型的具体类型
        char byteorder;          // 数据的字节顺序
        char flags;              // 数组描述符的标志
        int type_num;            // 数据类型的编号
        int elsize;              // 每个数组元素的大小
        int alignment;           // 数组数据的对齐方式
        struct _arr_descr *subarray;  // 子数组的描述符
        PyObject *fields;        // 数组字段的 Python 对象
        PyObject *names;         // 数组字段的名称
        PyArray_ArrFuncs *f;     // 数组函数的指针
        PyObject *metadata;      // 数组描述符的元数据
        NpyAuxData *c_metadata;  // C 风格的元数据
        npy_hash_t hash;         // 数组描述符的哈希值
} PyArray_DescrProto;


typedef struct _arr_descr {
        PyArray_Descr *base;     // 数组的基本描述符
        PyObject *shape;         // 数组的形状，一个元组
} PyArray_ArrayDescr;

/*
 * 数组数据的内存处理器结构体。
 * free 函数的声明与 PyMemAllocatorEx 不同。
 */
typedef struct {
    void *ctx;                  // 上下文指针
    void* (*malloc) (void *ctx, size_t size);        // 分配内存的函数指针
    void* (*calloc) (void *ctx, size_t nelem, size_t elsize);  // 分配并清零内存的函数指针
    void* (*realloc) (void *ctx, void *ptr, size_t new_size);  // 重新分配内存的函数指针
    void (*free) (void *ctx, void *ptr, size_t size);  // 释放内存的函数指针
    /*
     * 这是版本 1 的结尾。只能在此行之后添加新的字段。
     */
} PyDataMemAllocator;

typedef struct {
    char name[127];             // 名称字符串，长度为 127，用于保持结构体对齐
    uint8_t version;            // 版本号，当前为 1
    PyDataMemAllocator allocator;  // 内存分配器结构体
} PyDataMem_Handler;


/*
 * 主数组对象的结构体。
 *
 * 推荐使用下面定义的内联函数（如 PyArray_DATA 等）来访问此处的字段，
 * 因为在多个版本中直接访问成员本身已经被弃用。
 * 为了确保代码不使用已弃用的访问方式，
 * 需要定义 NPY_NO_DEPRECATED_API 为 NPY_1_7_API_VERSION
 * （或更高版本，如 NPY_1_8_API_VERSION）。
 */
/* This struct defines the fields of a PyArrayObject, which represents an array object in NumPy */

typedef struct tagPyArrayObject_fields {
    PyObject_HEAD
    /* Pointer to the raw data buffer */
    char *data;
    /* The number of dimensions, also called 'ndim' */
    int nd;
    /* The size in each dimension, also called 'shape' */
    npy_intp *dimensions;
    /*
     * Number of bytes to jump to get to the
     * next element in each dimension
     */
    npy_intp *strides;
    /*
     * This object is decref'd upon
     * deletion of array. Except in the
     * case of WRITEBACKIFCOPY which has
     * special handling.
     *
     * For views it points to the original
     * array, collapsed so no chains of
     * views occur.
     *
     * For creation from buffer object it
     * points to an object that should be
     * decref'd on deletion
     *
     * For WRITEBACKIFCOPY flag this is an
     * array to-be-updated upon calling
     * PyArray_ResolveWritebackIfCopy
     */
    PyObject *base;
    /* Pointer to type structure */
    PyArray_Descr *descr;
    /* Flags describing array -- see below */
    int flags;
    /* For weak references */
    PyObject *weakreflist;
#if NPY_FEATURE_VERSION >= NPY_1_20_API_VERSION
    void *_buffer_info;  /* private buffer info, tagged to allow warning */
#endif
    /*
     * For malloc/calloc/realloc/free per object
     */
#if NPY_FEATURE_VERSION >= NPY_1_22_API_VERSION
    PyObject *mem_handler;
#endif
} PyArrayObject_fields;

/*
 * To hide the implementation details, we only expose
 * the Python struct HEAD.
 */
#if !defined(NPY_NO_DEPRECATED_API) || \
    (NPY_NO_DEPRECATED_API < NPY_1_7_API_VERSION)
/*
 * Can't put this in npy_deprecated_api.h like the others.
 * PyArrayObject field access is deprecated as of NumPy 1.7.
 */
typedef PyArrayObject_fields PyArrayObject;
#else
typedef struct tagPyArrayObject {
        PyObject_HEAD
} PyArrayObject;
#endif

/*
 * Removed 2020-Nov-25, NumPy 1.20
 * #define NPY_SIZEOF_PYARRAYOBJECT (sizeof(PyArrayObject_fields))
 *
 * The above macro was removed as it gave a false sense of a stable ABI
 * with respect to the structures size.  If you require a runtime constant,
 * you can use `PyArray_Type.tp_basicsize` instead.  Otherwise, please
 * see the PyArrayObject documentation or ask the NumPy developers for
 * information on how to correctly replace the macro in a way that is
 * compatible with multiple NumPy versions.
 */

/* Mirrors buffer object to ptr */

typedef struct {
        PyObject_HEAD
        PyObject *base;
        void *ptr;
        npy_intp len;
        int flags;
} PyArray_Chunk;

typedef struct {
    NPY_DATETIMEUNIT base;
    int num;
} PyArray_DatetimeMetaData;

typedef struct {
    NpyAuxData base;
    PyArray_DatetimeMetaData meta;
} PyArray_DatetimeDTypeMetaData;

/*
 * This structure contains an exploded view of a date-time value.
 * NaT is represented by year == NPY_DATETIME_NAT.
 */
/*
 * 定义了表示日期时间的结构体，包括年、月、日、时、分、秒、微秒、皮秒和阿秒
 */
typedef struct {
        npy_int64 year;
        npy_int32 month, day, hour, min, sec, us, ps, as;
} npy_datetimestruct;

/*
 * 这个结构体在内部没有使用
 */
typedef struct {
        npy_int64 day;
        npy_int32 sec, us, ps, as;
} npy_timedeltastruct;

/*
 * PyArray_FinalizeFunc 的类型定义，是一个指向函数的指针，接受一个 PyArrayObject 和一个 PyObject* 参数并返回 int
 */
typedef int (PyArray_FinalizeFunc)(PyArrayObject *, PyObject *);

/*
 * 表示 C 风格连续的数组，即最后一个索引变化最快。数据元素紧挨着存储在一起。
 * 可以在构造函数中请求此标志。
 * 可以通过 PyArray_FLAGS(arr) 函数测试此标志。
 */
#define NPY_ARRAY_C_CONTIGUOUS    0x0001

/*
 * 表示 Fortran 风格连续的数组，即第一个索引在内存中变化最快（strides 数组与 C 连续数组相反）。
 * 可以在构造函数中请求此标志。
 * 可以通过 PyArray_FLAGS(arr) 函数测试此标志。
 */
#define NPY_ARRAY_F_CONTIGUOUS    0x0002

/*
 * 注意：所有的零维数组都是 C_CONTIGUOUS 和 F_CONTIGUOUS。
 * 如果一个一维数组是 C_CONTIGUOUS，那么它也是 F_CONTIGUOUS。
 * 多于一维的数组如果有零个或一个元素，则可以同时是 C_CONTIGUOUS 和 F_CONTIGUOUS。
 * 高维数组的连续性标志与 `array.squeeze()` 相同；
 * 当检查连续性时，具有 `array.shape[dimension] == 1` 的维度实际上被忽略。
 */

/*
 * 如果设置了该标志，数组拥有其数据：在删除数组时会释放数据。
 * 可以通过 PyArray_FLAGS(arr) 函数测试此标志。
 */
#define NPY_ARRAY_OWNDATA         0x0004

/*
 * 从任意类型转换到数组时，无论是否安全都进行强制转换。
 * 仅在各种 FromAny 函数的参数标志中使用。
 */
#define NPY_ARRAY_FORCECAST       0x0010

/*
 * 总是复制数组。返回的数组总是连续的、对齐的和可写的。
 * 参见：NPY_ARRAY_ENSURENOCOPY = 0x4000。
 * 可以在构造函数中请求此标志。
 */
#define NPY_ARRAY_ENSURECOPY      0x0020

/*
 * 确保返回的数组是基类 ndarray。
 * 可以在构造函数中请求此标志。
 */
#define NPY_ARRAY_ENSUREARRAY     0x0040

/*
 * 确保 strides 以元素大小为单位。对于某些记录数组的操作是必需的。
 * 可以在构造函数中请求此标志。
 */
#define NPY_ARRAY_ELEMENTSTRIDES  0x0080

/*
 * 数组数据按照类型存储的适当内存地址对齐。例如，整数数组（每个4字节）从4的倍数的内存地址开始。
 * 可以在构造函数中请求此标志。
 * 可以通过 PyArray_FLAGS(arr) 函数测试此标志。
 */
#define NPY_ARRAY_ALIGNED         0x0100

/*
 * 数组数据具有本机字节顺序。
 * 可以在构造函数中请求此标志。
 */
/*
 * 定义了一系列 NumPy 数组的属性标志位
 */

#define NPY_ARRAY_NOTSWAPPED      0x0200

/*
 * 数组数据可写
 *
 * 可以在构造函数中请求此标志位。
 * 可以通过 PyArray_FLAGS(arr) 测试此标志位。
 */
#define NPY_ARRAY_WRITEABLE       0x0400

/*
 * 如果设置了此标志位，那么 base 包含一个指向相同大小数组的指针，
 * 在调用 PyArray_ResolveWritebackIfCopy 时应更新为当前数组的内容。
 *
 * 可以在构造函数中请求此标志位。
 * 可以通过 PyArray_FLAGS(arr) 测试此标志位。
 */
#define NPY_ARRAY_WRITEBACKIFCOPY 0x2000

/*
 * 在从对象/数组转换时禁止复制（结果是一个视图）
 *
 * 可以在构造函数中请求此标志位。
 */
#define NPY_ARRAY_ENSURENOCOPY 0x4000

/*
 * 注意：在 multiarray/arrayobject.h 中还定义了从第 31 位开始的内部标志位。
 */

#define NPY_ARRAY_BEHAVED      (NPY_ARRAY_ALIGNED | \
                                NPY_ARRAY_WRITEABLE)
#define NPY_ARRAY_BEHAVED_NS   (NPY_ARRAY_ALIGNED | \
                                NPY_ARRAY_WRITEABLE | \
                                NPY_ARRAY_NOTSWAPPED)
#define NPY_ARRAY_CARRAY       (NPY_ARRAY_C_CONTIGUOUS | \
                                NPY_ARRAY_BEHAVED)
#define NPY_ARRAY_CARRAY_RO    (NPY_ARRAY_C_CONTIGUOUS | \
                                NPY_ARRAY_ALIGNED)
#define NPY_ARRAY_FARRAY       (NPY_ARRAY_F_CONTIGUOUS | \
                                NPY_ARRAY_BEHAVED)
#define NPY_ARRAY_FARRAY_RO    (NPY_ARRAY_F_CONTIGUOUS | \
                                NPY_ARRAY_ALIGNED)
#define NPY_ARRAY_DEFAULT      (NPY_ARRAY_CARRAY)
#define NPY_ARRAY_IN_ARRAY     (NPY_ARRAY_CARRAY_RO)
#define NPY_ARRAY_OUT_ARRAY    (NPY_ARRAY_CARRAY)
#define NPY_ARRAY_INOUT_ARRAY  (NPY_ARRAY_CARRAY)
#define NPY_ARRAY_INOUT_ARRAY2 (NPY_ARRAY_CARRAY | \
                                NPY_ARRAY_WRITEBACKIFCOPY)
#define NPY_ARRAY_IN_FARRAY    (NPY_ARRAY_FARRAY_RO)
#define NPY_ARRAY_OUT_FARRAY   (NPY_ARRAY_FARRAY)
#define NPY_ARRAY_INOUT_FARRAY (NPY_ARRAY_FARRAY)
#define NPY_ARRAY_INOUT_FARRAY2 (NPY_ARRAY_FARRAY | \
                                NPY_ARRAY_WRITEBACKIFCOPY)

#define NPY_ARRAY_UPDATE_ALL   (NPY_ARRAY_C_CONTIGUOUS | \
                                NPY_ARRAY_F_CONTIGUOUS | \
                                NPY_ARRAY_ALIGNED)

/* 此标志位是数组接口的，不是 PyArrayObject 的 */
#define NPY_ARR_HAS_DESCR  0x0800




/*
 * 内部缓冲区的大小，用于对齐。将 BUFSIZE 设为 npy_cdouble 的倍数 —— 通常为 16，以便对齐 ufunc 缓冲区
 */
#define NPY_MIN_BUFSIZE ((int)sizeof(npy_cdouble))
#define NPY_MAX_BUFSIZE (((int)sizeof(npy_cdouble))*1000000)
#define NPY_BUFSIZE 8192
/* 缓冲区压力测试大小： */
/*#define NPY_BUFSIZE 17*/

/*
 * C API：包括宏和函数。这里定义了宏。
 */


#define PyArray_ISCONTIGUOUS(m) PyArray_CHKFLAGS((m), NPY_ARRAY_C_CONTIGUOUS)
/* 定义宏PyArray_ISWRITEABLE，检查数组是否可写 */
#define PyArray_ISWRITEABLE(m) PyArray_CHKFLAGS((m), NPY_ARRAY_WRITEABLE)
/* 定义宏PyArray_ISALIGNED，检查数组是否按对齐要求对齐 */
#define PyArray_ISALIGNED(m) PyArray_CHKFLAGS((m), NPY_ARRAY_ALIGNED)

/* 定义宏PyArray_IS_C_CONTIGUOUS，检查数组是否C顺序连续 */
#define PyArray_IS_C_CONTIGUOUS(m) PyArray_CHKFLAGS((m), NPY_ARRAY_C_CONTIGUOUS)
/* 定义宏PyArray_IS_F_CONTIGUOUS，检查数组是否Fortran顺序连续 */
#define PyArray_IS_F_CONTIGUOUS(m) PyArray_CHKFLAGS((m), NPY_ARRAY_F_CONTIGUOUS)

/* NPY_BEGIN_THREADS_DEF在某些地方使用，因此始终定义它 */
#define NPY_BEGIN_THREADS_DEF PyThreadState *_save=NULL;
#if NPY_ALLOW_THREADS
/* 如果允许线程，则定义以下线程相关宏 */
#define NPY_BEGIN_ALLOW_THREADS Py_BEGIN_ALLOW_THREADS
#define NPY_END_ALLOW_THREADS Py_END_ALLOW_THREADS
#define NPY_BEGIN_THREADS do {_save = PyEval_SaveThread();} while (0);
#define NPY_END_THREADS   do { if (_save) \
                { PyEval_RestoreThread(_save); _save = NULL;} } while (0);
/* 根据循环大小阈值决定是否启用线程 */
#define NPY_BEGIN_THREADS_THRESHOLDED(loop_size) do { if ((loop_size) > 500) \
                { _save = PyEval_SaveThread();} } while (0);

/* 定义宏NPY_ALLOW_C_API_DEF，保护C API访问状态 */
#define NPY_ALLOW_C_API_DEF  PyGILState_STATE __save__;
/* 进入C API访问状态 */
#define NPY_ALLOW_C_API      do {__save__ = PyGILState_Ensure();} while (0);
/* 退出C API访问状态 */
#define NPY_DISABLE_C_API    do {PyGILState_Release(__save__);} while (0);
#else
/* 如果不允许线程，则将所有线程相关宏置空 */
#define NPY_BEGIN_ALLOW_THREADS
#define NPY_END_ALLOW_THREADS
#define NPY_BEGIN_THREADS
#define NPY_END_THREADS
#define NPY_BEGIN_THREADS_THRESHOLDED(loop_size)
#define NPY_BEGIN_THREADS_DESCR(dtype)
#define NPY_END_THREADS_DESCR(dtype)
#define NPY_ALLOW_C_API_DEF
#define NPY_ALLOW_C_API
#define NPY_DISABLE_C_API
#endif

/**********************************
 * The nditer object, added in 1.6
 **********************************/

/* nditer对象的实际结构是内部细节 */
typedef struct NpyIter_InternalOnly NpyIter;

/* 可能被特殊化的迭代器函数指针 */
typedef int (NpyIter_IterNextFunc)(NpyIter *iter);  /* 迭代到下一个函数 */
typedef void (NpyIter_GetMultiIndexFunc)(NpyIter *iter,
                                      npy_intp *outcoords);  /* 获取多索引 */

/*** 全局标志，可能被传递给迭代器构造函数 ***/

/* 跟踪代表C顺序的索引 */
#define NPY_ITER_C_INDEX                    0x00000001
/* 跟踪代表Fortran顺序的索引 */
#define NPY_ITER_F_INDEX                    0x00000002
/* 跟踪多索引 */
#define NPY_ITER_MULTI_INDEX                0x00000004
/* 外部用户代码执行1维最内层循环 */
#define NPY_ITER_EXTERNAL_LOOP              0x00000008
/* 将所有操作数转换为共同的数据类型 */
#define NPY_ITER_COMMON_DTYPE               0x00000010
/* 操作数可能持有引用，在迭代期间需要API访问 */
#define NPY_ITER_REFS_OK                    0x00000020
/* 允许零大小的操作数，迭代检查IterSize为0 */
#define NPY_ITER_ZEROSIZE_OK                0x00000040
/* 允许归约（大小为0的步幅，但维度大小>1） */
#define NPY_ITER_REDUCE_OK                  0x00000080
/* 启用子范围迭代 */
#define NPY_ITER_RANGED                     0x00000100
/* 启用缓冲 */
/* 定义了一系列的常量，用于 NumPy 迭代器对象的配置 */

/* 全局标志位，指定迭代器的基本行为特征 */
#define NPY_ITER_BUFFERED                   0x00000200
/* 当启用缓冲时，尽可能扩展内部循环 */
#define NPY_ITER_GROWINNER                  0x00000400
/* 延迟分配缓冲区直到第一次 Reset* 调用 */
#define NPY_ITER_DELAY_BUFALLOC             0x00000800
/* 当指定 NPY_KEEPORDER 时，禁止反转负步幅轴 */
#define NPY_ITER_DONT_NEGATE_STRIDES        0x00001000
/*
 * 如果输出操作数与其他操作数重叠（基于启发式方法，可能有误报但不会漏报），则进行临时复制以消除重叠。
 */
#define NPY_ITER_COPY_IF_OVERLAP            0x00002000

/*** 可传递给迭代器构造函数的每个操作数的标志 ***/

/* 操作数将被读取和写入 */
#define NPY_ITER_READWRITE                  0x00010000
/* 操作数将只被读取 */
#define NPY_ITER_READONLY                   0x00020000
/* 操作数将只被写入 */
#define NPY_ITER_WRITEONLY                  0x00040000
/* 操作数的数据必须是本机字节顺序 */
#define NPY_ITER_NBO                        0x00080000
/* 操作数的数据必须对齐 */
#define NPY_ITER_ALIGNED                    0x00100000
/* 操作数的数据必须是连续的（在内部循环内） */
#define NPY_ITER_CONTIG                     0x00200000
/* 可以复制操作数以满足要求 */
#define NPY_ITER_COPY                       0x00400000
/* 可以使用 WRITEBACKIFCOPY 复制操作数以满足要求 */
#define NPY_ITER_UPDATEIFCOPY               0x00800000
/* 如果操作数为 NULL，则分配操作数 */
#define NPY_ITER_ALLOCATE                   0x01000000
/* 如果分配了操作数，则不使用任何子类型 */
#define NPY_ITER_NO_SUBTYPE                 0x02000000
/* 这是一个虚拟数组插槽，操作数为 NULL，但临时数据存在 */
#define NPY_ITER_VIRTUAL                    0x04000000
/* 要求维度与迭代器维度完全匹配 */
#define NPY_ITER_NO_BROADCAST               0x08000000
/* 在此数组上使用掩码，影响缓冲区到数组的复制 */
#define NPY_ITER_WRITEMASKED                0x10000000
/* 此数组是所有 WRITEMASKED 操作数的掩码 */
#define NPY_ITER_ARRAYMASK                  0x20000000
/* 假设迭代器顺序数据访问以应对 COPY_IF_OVERLAP */
#define NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE 0x40000000

/* 全局标志的掩码 */
#define NPY_ITER_GLOBAL_FLAGS               0x0000ffff
/* 每个操作数的标志的掩码 */
#define NPY_ITER_PER_OP_FLAGS               0xffff0000


/*****************************
 * 基本迭代器对象
 *****************************/

/* 前向声明 */
typedef struct PyArrayIterObject_tag PyArrayIterObject;

/*
 * 一个函数类型，用于将一组坐标转换为指向数据的指针
 */
typedef char* (*npy_iter_get_dataptr_t)(
        PyArrayIterObject* iter, const npy_intp*);
/* 定义一个结构体 PyArrayIterObject_tag，用于实现数组迭代器 */
struct PyArrayIterObject_tag {
        PyObject_HEAD  /* Python 对象头部 */

        int               nd_m1;            /* 数组维度数减一 */
        npy_intp          index, size;      /* 当前索引和数组大小 */
        npy_intp          coordinates[NPY_MAXDIMS_LEGACY_ITERS];/* NPY_MAXDIMS_LEGACY_ITERS 维度的坐标 */
        npy_intp          dims_m1[NPY_MAXDIMS_LEGACY_ITERS];    /* 数组的维度数组减一 */
        npy_intp          strides[NPY_MAXDIMS_LEGACY_ITERS];    /* 数组的步幅或者虚拟的步幅 */
        npy_intp          backstrides[NPY_MAXDIMS_LEGACY_ITERS];/* 后退的步幅 */
        npy_intp          factors[NPY_MAXDIMS_LEGACY_ITERS];     /* 形状因子 */
        PyArrayObject     *ao;               /* 指向 PyArrayObject 的指针 */
        char              *dataptr;          /* 当前项的指针 */
        npy_bool          contiguous;        /* 是否连续存储 */

        npy_intp          bounds[NPY_MAXDIMS_LEGACY_ITERS][2];   /* 边界数组 */
        npy_intp          limits[NPY_MAXDIMS_LEGACY_ITERS][2];   /* 限制数组 */
        npy_intp          limits_sizes[NPY_MAXDIMS_LEGACY_ITERS];/* 限制大小数组 */
        npy_iter_get_dataptr_t translate;   /* 获取数据指针的函数 */
} ;


/* 迭代器 API */
#define PyArrayIter_Check(op) PyObject_TypeCheck((op), &PyArrayIter_Type)

#define _PyAIT(it) ((PyArrayIterObject *)(it))  /* 将迭代器类型转换为 PyArrayIterObject 指针类型 */
#define PyArray_ITER_RESET(it) do { \
        _PyAIT(it)->index = 0; \  /* 重置迭代器索引为 0 */
        _PyAIT(it)->dataptr = PyArray_BYTES(_PyAIT(it)->ao); \  /* 重置数据指针为数组数据的起始位置 */
        memset(_PyAIT(it)->coordinates, 0, \
               (_PyAIT(it)->nd_m1+1)*sizeof(npy_intp)); \  /* 将坐标数组清零 */
} while (0)

#define _PyArray_ITER_NEXT1(it) do { \
        (it)->dataptr += _PyAIT(it)->strides[0]; \  /* 移动到下一个元素的位置 */
        (it)->coordinates[0]++; \  /* 更新第一个维度的坐标 */
} while (0)

#define _PyArray_ITER_NEXT2(it) do { \
        if ((it)->coordinates[1] < (it)->dims_m1[1]) { \  /* 如果第二个维度的坐标小于 dims_m1 中的值 */
                (it)->coordinates[1]++; \  /* 更新第二个维度的坐标 */
                (it)->dataptr += (it)->strides[1]; \  /* 移动到下一个元素的位置 */
        } \
        else { \
                (it)->coordinates[1] = 0; \  /* 第二个维度的坐标重置为 0 */
                (it)->coordinates[0]++; \  /* 更新第一个维度的坐标 */
                (it)->dataptr += (it)->strides[0] - \
                        (it)->backstrides[1]; \  /* 移动到下一个元素的位置，考虑到第二个维度的步幅 */
        } \
} while (0)


这些注释解释了给定的 C 语言结构体定义和宏定义，以及它们的功能和作用。
// 宏定义：PyArray_ITER_NEXT(it)
#define PyArray_ITER_NEXT(it) do { \
        // 增加迭代器的索引
        _PyAIT(it)->index++; \
        // 如果维度数为 0，调用单维度迭代器的下一步函数
        if (_PyAIT(it)->nd_m1 == 0) { \
                _PyArray_ITER_NEXT1(_PyAIT(it)); \
        } \
        // 如果是连续存储的多维数组，增加数据指针以移动到下一个元素
        else if (_PyAIT(it)->contiguous) \
                _PyAIT(it)->dataptr += PyArray_ITEMSIZE(_PyAIT(it)->ao); \
        // 如果是二维数组，调用双维度迭代器的下一步函数
        else if (_PyAIT(it)->nd_m1 == 1) { \
                _PyArray_ITER_NEXT2(_PyAIT(it)); \
        } \
        // 对于更高维度的数组，逐个维度增加坐标和数据指针
        else { \
                int __npy_i; \
                for (__npy_i=_PyAIT(it)->nd_m1; __npy_i >= 0; __npy_i--) { \
                        if (_PyAIT(it)->coordinates[__npy_i] < \
                            _PyAIT(it)->dims_m1[__npy_i]) { \
                                _PyAIT(it)->coordinates[__npy_i]++; \
                                _PyAIT(it)->dataptr += \
                                        _PyAIT(it)->strides[__npy_i]; \
                                break; \
                        } \
                        else { \
                                _PyAIT(it)->coordinates[__npy_i] = 0; \
                                _PyAIT(it)->dataptr -= \
                                        _PyAIT(it)->backstrides[__npy_i]; \
                        } \
                } \
        } \
} while (0)

// 宏定义：PyArray_ITER_GOTO(it, destination)
#define PyArray_ITER_GOTO(it, destination) do { \
        int __npy_i; \
        // 将迭代器的索引重置为 0
        _PyAIT(it)->index = 0; \
        // 将数据指针定位到数组的起始位置
        _PyAIT(it)->dataptr = PyArray_BYTES(_PyAIT(it)->ao); \
        // 遍历各维度，根据目标坐标调整数据指针和迭代器的坐标
        for (__npy_i = _PyAIT(it)->nd_m1; __npy_i>=0; __npy_i--) { \
                // 如果目标坐标为负数，转换为对应的非负坐标
                if (destination[__npy_i] < 0) { \
                        destination[__npy_i] += \
                                _PyAIT(it)->dims_m1[__npy_i]+1; \
                } \
                // 根据目标坐标和步长调整数据指针
                _PyAIT(it)->dataptr += destination[__npy_i] * \
                        _PyAIT(it)->strides[__npy_i]; \
                // 设置迭代器的坐标为目标坐标
                _PyAIT(it)->coordinates[__npy_i] = \
                        destination[__npy_i]; \
                // 计算迭代器的索引，考虑各维度的步长和边界
                _PyAIT(it)->index += destination[__npy_i] * \
                        ( __npy_i==_PyAIT(it)->nd_m1 ? 1 : \
                          _PyAIT(it)->dims_m1[__npy_i+1]+1) ; \
        } \
} while (0)
/*
 * 宏定义：PyArray_ITER_GOTO1D(it, ind)
 * 功能：将迭代器移动到一维数组中的指定索引位置
 */
#define PyArray_ITER_GOTO1D(it, ind) do { \
        int __npy_i; \
        npy_intp __npy_ind = (npy_intp)(ind); \  // 将输入的索引转换为 npy_intp 类型
        if (__npy_ind < 0) __npy_ind += _PyAIT(it)->size; \  // 如果索引为负数，将其转换为正数
        _PyAIT(it)->index = __npy_ind; \  // 设置迭代器的当前索引为 __npy_ind
        if (_PyAIT(it)->nd_m1 == 0) { \  // 如果数组的维度为1
                _PyAIT(it)->dataptr = PyArray_BYTES(_PyAIT(it)->ao) + \  // 设置数据指针为数组的起始地址 + 索引 * 步长
                        __npy_ind * _PyAIT(it)->strides[0]; \
        } \
        else if (_PyAIT(it)->contiguous) \  // 如果数组是连续存储的
                _PyAIT(it)->dataptr = PyArray_BYTES(_PyAIT(it)->ao) + \  // 设置数据指针为数组的起始地址 + 索引 * 每个元素的字节大小
                        __npy_ind * PyArray_ITEMSIZE(_PyAIT(it)->ao); \
        else { \  // 否则，数组不是连续存储的
                _PyAIT(it)->dataptr = PyArray_BYTES(_PyAIT(it)->ao); \  // 设置数据指针为数组的起始地址
                for (__npy_i = 0; __npy_i<=_PyAIT(it)->nd_m1; \
                     __npy_i++) { \  // 遍历数组的各维度
                        _PyAIT(it)->coordinates[__npy_i] = \  // 计算当前维度的坐标值
                                (__npy_ind / _PyAIT(it)->factors[__npy_i]); \
                        _PyAIT(it)->dataptr += \  // 根据坐标值和步长，计算数据指针的偏移量
                                (__npy_ind / _PyAIT(it)->factors[__npy_i]) \
                                * _PyAIT(it)->strides[__npy_i]; \
                        __npy_ind %= _PyAIT(it)->factors[__npy_i]; \  // 更新索引值以处理下一个维度
                } \
        } \
} while (0)

/*
 * 宏定义：PyArray_ITER_DATA(it)
 * 功能：获取迭代器当前指向的数据指针
 */
#define PyArray_ITER_DATA(it) ((void *)(_PyAIT(it)->dataptr))

/*
 * 宏定义：PyArray_ITER_NOTDONE(it)
 * 功能：检查迭代器是否未完成遍历
 */
#define PyArray_ITER_NOTDONE(it) (_PyAIT(it)->index < _PyAIT(it)->size)


/*
 * 结构体定义：PyArrayMultiIterObject
 * 功能：多数组迭代器对象
 */
typedef struct {
        PyObject_HEAD
        int                  numiter;  // 迭代器的数量
        npy_intp             size;  // 广播后的总大小
        npy_intp             index;  // 当前索引
        int                  nd;  // 维度数
        npy_intp             dimensions[NPY_MAXDIMS_LEGACY_ITERS];  // 维度数组
        /*
         * Space for the individual iterators, do not specify size publicly
         * to allow changing it more easily.
         * One reason is that Cython uses this for checks and only allows
         * growing structs (as of Cython 3.0.6).  It also allows NPY_MAXARGS
         * to be runtime dependent.
         */
#if (defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD)
        PyArrayIterObject    *iters[64];  // 迭代器对象数组，最大容量为64
#elif defined(__cplusplus)
        /*
         * C++ doesn't stricly support flexible members and gives compilers
         * warnings (pedantic only), so we lie.  We can't make it 64 because
         * then Cython is unhappy (larger struct at runtime is OK smaller not).
         */
        PyArrayIterObject    *iters[32];  // 迭代器对象数组，最大容量为32（为了避免C++编译器警告）
#else
        PyArrayIterObject    *iters[];  // 迭代器对象数组，长度可变
#endif
} PyArrayMultiIterObject;

#define _PyMIT(m) ((PyArrayMultiIterObject *)(m))
#define PyArray_MultiIter_RESET(multi) do {                                   \
        int __npy_mi;                                                         \
        _PyMIT(multi)->index = 0;                                             \
        for (__npy_mi=0; __npy_mi < _PyMIT(multi)->numiter;  __npy_mi++) {    \
                PyArray_ITER_RESET(_PyMIT(multi)->iters[__npy_mi]);           \
        }                                                                     \
} while (0)


# 重置多重迭代器 `multi` 的状态，将其索引置为0，并逐个重置其包含的每个迭代器
#define PyArray_MultiIter_RESET(multi) do {                                   \
        int __npy_mi;                                                         \
        _PyMIT(multi)->index = 0;                                             \
        for (__npy_mi=0; __npy_mi < _PyMIT(multi)->numiter;  __npy_mi++) {    \
                PyArray_ITER_RESET(_PyMIT(multi)->iters[__npy_mi]);           \
        }                                                                     \
} while (0)



#define PyArray_MultiIter_NEXT(multi) do {                                    \
        int __npy_mi;                                                         \
        _PyMIT(multi)->index++;                                               \
        for (__npy_mi=0; __npy_mi < _PyMIT(multi)->numiter;   __npy_mi++) {   \
                PyArray_ITER_NEXT(_PyMIT(multi)->iters[__npy_mi]);            \
        }                                                                     \
} while (0)


# 将多重迭代器 `multi` 的索引递增，并逐个移动其包含的每个迭代器到下一个位置
#define PyArray_MultiIter_NEXT(multi) do {                                    \
        int __npy_mi;                                                         \
        _PyMIT(multi)->index++;                                               \
        for (__npy_mi=0; __npy_mi < _PyMIT(multi)->numiter;   __npy_mi++) {   \
                PyArray_ITER_NEXT(_PyMIT(multi)->iters[__npy_mi]);            \
        }                                                                     \
} while (0)



#define PyArray_MultiIter_GOTO(multi, dest) do {                            \
        int __npy_mi;                                                       \
        for (__npy_mi=0; __npy_mi < _PyMIT(multi)->numiter; __npy_mi++) {   \
                PyArray_ITER_GOTO(_PyMIT(multi)->iters[__npy_mi], dest);    \
        }                                                                   \
        _PyMIT(multi)->index = _PyMIT(multi)->iters[0]->index;              \
} while (0)


# 将多重迭代器 `multi` 中所有迭代器移动到目标位置 `dest`，并将多重迭代器的索引设置为第一个迭代器的当前索引
#define PyArray_MultiIter_GOTO(multi, dest) do {                            \
        int __npy_mi;                                                       \
        for (__npy_mi=0; __npy_mi < _PyMIT(multi)->numiter; __npy_mi++) {   \
                PyArray_ITER_GOTO(_PyMIT(multi)->iters[__npy_mi], dest);    \
        }                                                                   \
        _PyMIT(multi)->index = _PyMIT(multi)->iters[0]->index;              \
} while (0)



#define PyArray_MultiIter_GOTO1D(multi, ind) do {                          \
        int __npy_mi;                                                      \
        for (__npy_mi=0; __npy_mi < _PyMIT(multi)->numiter; __npy_mi++) {  \
                PyArray_ITER_GOTO1D(_PyMIT(multi)->iters[__npy_mi], ind);  \
        }                                                                  \
        _PyMIT(multi)->index = _PyMIT(multi)->iters[0]->index;             \
} while (0)


# 将多重迭代器 `multi` 中所有迭代器移动到一维索引 `ind` 指定的位置，并将多重迭代器的索引设置为第一个迭代器的当前索引
#define PyArray_MultiIter_GOTO1D(multi, ind) do {                          \
        int __npy_mi;                                                      \
        for (__npy_mi=0; __npy_mi < _PyMIT(multi)->numiter; __npy_mi++) {  \
                PyArray_ITER_GOTO1D(_PyMIT(multi)->iters[__npy_mi], ind);  \
        }                                                                  \
        _PyMIT(multi)->index = _PyMIT(multi)->iters[0]->index;             \
} while (0)



#define PyArray_MultiIter_DATA(multi, i)                \
        ((void *)(_PyMIT(multi)->iters[i]->dataptr))


# 返回多重迭代器 `multi` 中第 `i` 个迭代器当前指向的数据指针
#define PyArray_MultiIter_DATA(multi, i)                \
        ((void *)(_PyMIT(multi)->iters[i]->dataptr))



#define PyArray_MultiIter_NEXTi(multi, i)               \
        PyArray_ITER_NEXT(_PyMIT(multi)->iters[i])


# 将多重迭代器 `multi` 中第 `i` 个迭代器移动到下一个位置
#define PyArray_MultiIter_NEXTi(multi, i)               \
        PyArray_ITER_NEXT(_PyMIT(multi)->iters[i])



#define PyArray_MultiIter_NOTDONE(multi)                \
        (_PyMIT(multi)->index < _PyMIT(multi)->size)


# 检查多重迭代器 `multi` 是否完成迭代，即索引是否小于其总大小
#define PyArray_MultiIter_NOTDONE(multi)                \
        (_PyMIT(multi)->index < _PyMIT(multi)->size)



static NPY_INLINE int
PyArray_MultiIter_NUMITER(PyArrayMultiIterObject *multi)
{
    return multi->numiter;
}


# 返回多重迭代器 `multi` 中包含的迭代器数量
static NPY_INLINE int
PyArray_MultiIter_NUMITER(PyArrayMultiIterObject *multi)
{
    return multi->numiter;
}



static NPY_INLINE npy_intp
PyArray_MultiIter_SIZE(PyArrayMultiIterObject *multi)
{
    return multi->size;
}


# 返回多重迭代器 `multi` 的总大小
static NPY_INLINE npy_intp
PyArray_MultiIter_SIZE(PyArrayMultiIterObject *multi)
{
    return multi->size;
}



static NPY_INLINE npy_intp
PyArray_MultiIter_INDEX(PyArrayMultiIterObject *multi)
{
    return multi->index;
}


# 返回多重迭代器 `multi` 的当前索引
static NPY_INLINE npy_intp
PyArray_MultiIter_INDEX(PyArrayMultiIterObject *multi)
{
    return multi->index;
}



static NPY_INLINE int
PyArray_MultiIter_NDIM(PyArrayMultiIterObject *multi)
{
    return multi->nd;
}


# 返回多重迭代器 `multi` 涉及的数组维度数目
static NPY_INLINE int
PyArray_MultiIter_NDIM(PyArrayMultiIterObject *multi)
{
    return multi->nd;
}



static NPY_INLINE npy_intp *
PyArray_MultiIter_DIMS(PyArrayMultiIterObject *multi)
{
    return multi->dimensions;
}


# 返回多重迭代器 `multi` 涉及的数组维度
static NPY_INLINE npy_intp *
PyArray_MultiIter_DIMS(PyArrayMultiIterObject *multi)
{
    return multi->dimensions;
}



static NPY_INLINE void **
PyArray_MultiIter_ITERS(PyArrayMultiIterObject *multi)
{
    return (void**)multi->iters;
}


# 返回多重迭代器 `multi` 中迭代器对象数组的指针
static NPY_INLINE void **
PyArray_MultiIter_ITERS(PyArrayMultiIterObject *multi)
{
    return (void**)multi->iters;
}



enum {


# 枚举的开始标记
enum {
    # 定义常量：邻域迭代器使用零填充方式
    NPY_NEIGHBORHOOD_ITER_ZERO_PADDING,
    # 定义常量：邻域迭代器使用一填充方式
    NPY_NEIGHBORHOOD_ITER_ONE_PADDING,
    # 定义常量：邻域迭代器使用常量填充方式
    NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING,
    # 定义常量：邻域迭代器使用循环填充方式
    NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING,
    # 定义常量：邻域迭代器使用镜像填充方式
    NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING
};

typedef struct {
    PyObject_HEAD

    /*
     * PyArrayIterObject part: keep this in this exact order
     */
    int               nd_m1;            /* number of dimensions - 1 */
    npy_intp          index, size;      /* current index and total size */
    npy_intp          coordinates[NPY_MAXDIMS_LEGACY_ITERS]; /* N-dimensional loop coordinates */
    npy_intp          dims_m1[NPY_MAXDIMS_LEGACY_ITERS];     /* ao->dimensions - 1 */
    npy_intp          strides[NPY_MAXDIMS_LEGACY_ITERS];     /* ao->strides or fake */
    npy_intp          backstrides[NPY_MAXDIMS_LEGACY_ITERS]; /* how far to jump back */
    npy_intp          factors[NPY_MAXDIMS_LEGACY_ITERS];      /* shape factors */
    PyArrayObject     *ao;              /* pointer to the PyArrayObject */
    char              *dataptr;         /* pointer to current item */
    npy_bool          contiguous;       /* flag indicating contiguous data */

    npy_intp          bounds[NPY_MAXDIMS_LEGACY_ITERS][2];   /* bounds for each dimension */
    npy_intp          limits[NPY_MAXDIMS_LEGACY_ITERS][2];   /* limits for each dimension */
    npy_intp          limits_sizes[NPY_MAXDIMS_LEGACY_ITERS]; /* size of limits for each dimension */
    npy_iter_get_dataptr_t translate;   /* function pointer to translate data */

    /*
     * New members
     */
    npy_intp nd;                       /* number of dimensions */

    /* Dimensions is the dimension of the array */
    npy_intp dimensions[NPY_MAXDIMS_LEGACY_ITERS]; /* dimensions of the array */

    /*
     * Neighborhood points coordinates are computed relatively to the
     * point pointed by _internal_iter
     */
    PyArrayIterObject* _internal_iter;  /* internal iterator object */

    /*
     * To keep a reference to the representation of the constant value
     * for constant padding
     */
    char* constant;                    /* constant padding representation */

    int mode;                          /* mode of operation */
} PyArrayNeighborhoodIterObject;

/*
 * Neighborhood iterator API
 */

/* General: those work for any mode */

/* Reset the neighborhood iterator */
static inline int
PyArrayNeighborhoodIter_Reset(PyArrayNeighborhoodIterObject* iter);

/* Move to the next neighborhood */
static inline int
PyArrayNeighborhoodIter_Next(PyArrayNeighborhoodIterObject* iter);

#if 0
/* Move to the next neighborhood (2D version) */
static inline int
PyArrayNeighborhoodIter_Next2D(PyArrayNeighborhoodIterObject* iter);
#endif

/*
 * Include inline implementations - functions defined there are not
 * considered public API
 */
#define NUMPY_CORE_INCLUDE_NUMPY__NEIGHBORHOOD_IMP_H_
#include "_neighborhood_iterator_imp.h"
#undef NUMPY_CORE_INCLUDE_NUMPY__NEIGHBORHOOD_IMP_H_


/* The default array type */
#define NPY_DEFAULT_TYPE NPY_DOUBLE
/* default integer type defined in npy_2_compat header */

/*
 * All sorts of useful ways to look into a PyArrayObject. It is recommended
 * to use PyArrayObject * objects instead of always casting from PyObject *,
 * for improved type checking.
 *
 * In many cases here the macro versions of the accessors are deprecated,
 * but can't be immediately changed to inline functions because the
 * preexisting macros accept PyObject * and do automatic casts. Inline
 * functions accepting PyArrayObject * provides for some compile-time
 * checking of correctness when working with these objects in C.
 */

/* Check if array is one segment (contiguous) */
#define PyArray_ISONESEGMENT(m) (PyArray_CHKFLAGS(m, NPY_ARRAY_C_CONTIGUOUS) || \
                                 PyArray_CHKFLAGS(m, NPY_ARRAY_F_CONTIGUOUS))


注释：  

/* 结构体定义结束 */

/* 定义 PyArrayNeighborhoodIterObject 结构体，继承自 PyObject */
/* PyArrayIterObject 的一部分：保持此精确顺序 */

/* 数组迭代器结构体的维度减一 */
/* 当前索引和总大小 */
/* N 维循环的坐标 */
/* ao->dimensions - 1 */
/* ao->strides 或虚拟值 */
/* 后退跨度 */
/* 形状因子 */
/* 指向 PyArrayObject 的指针 */
/* 指向当前项的指针 */
/* 表示数据是否连续的标志 */

/* 每个维度的边界 */
/* 每个维度的限制 */
/* 每个维度限制的大小 */
/* 数据转换函数指针 */

/* 新成员 */

/* 数组的维度 */
/* 数组的维度 */

/* 邻域点坐标相对于 _internal_iter 指向的点计算 */

/* 保留对常量填充值表示的引用 */

/* 操作模式 */
/*
 * 检查数组是否按 Fortran 风格存储（列优先），同时不应该按 C 风格存储（行优先）
 * 返回值：如果数组按 Fortran 风格存储则返回非零值，否则返回零
 */
#define PyArray_ISFORTRAN(m) (PyArray_CHKFLAGS(m, NPY_ARRAY_F_CONTIGUOUS) && \
                             (!PyArray_CHKFLAGS(m, NPY_ARRAY_C_CONTIGUOUS)))

/*
 * 如果数组按 Fortran 风格存储，返回 NPY_ARRAY_F_CONTIGUOUS，否则返回 0
 */
#define PyArray_FORTRAN_IF(m) ((PyArray_CHKFLAGS(m, NPY_ARRAY_F_CONTIGUOUS) ? \
                               NPY_ARRAY_F_CONTIGUOUS : 0))

/*
 * 返回数组的维度数
 */
static inline int
PyArray_NDIM(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->nd;
}

/*
 * 返回数组的数据指针
 */
static inline void *
PyArray_DATA(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->data;
}

/*
 * 返回数组的数据指针（以字符形式）
 */
static inline char *
PyArray_BYTES(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->data;
}

/*
 * 返回数组的维度数组指针
 */
static inline npy_intp *
PyArray_DIMS(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->dimensions;
}

/*
 * 返回数组的步幅数组指针
 */
static inline npy_intp *
PyArray_STRIDES(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->strides;
}

/*
 * 返回数组的指定维度的大小
 */
static inline npy_intp
PyArray_DIM(const PyArrayObject *arr, int idim)
{
    return ((PyArrayObject_fields *)arr)->dimensions[idim];
}

/*
 * 返回数组的指定步幅
 */
static inline npy_intp
PyArray_STRIDE(const PyArrayObject *arr, int istride)
{
    return ((PyArrayObject_fields *)arr)->strides[istride];
}

/*
 * 返回数组的 base 对象
 */
static inline NPY_RETURNS_BORROWED_REF PyObject *
PyArray_BASE(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->base;
}

/*
 * 返回数组的数据类型描述符
 */
static inline NPY_RETURNS_BORROWED_REF PyArray_Descr *
PyArray_DESCR(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->descr;
}

/*
 * 返回数组的 flags 属性
 */
static inline int
PyArray_FLAGS(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->flags;
}

/*
 * 返回数组的数据类型编号
 */
static inline int
PyArray_TYPE(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->descr->type_num;
}

/*
 * 检查数组的 flags 是否包含指定的 flags
 * 返回值：如果数组 flags 包含所有指定 flags，则返回非零值，否则返回零
 */
static inline int
PyArray_CHKFLAGS(const PyArrayObject *arr, int flags)
{
    return (PyArray_FLAGS(arr) & flags) == flags;
}

/*
 * 返回数组的数据类型描述符
 */
static inline PyArray_Descr *
PyArray_DTYPE(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->descr;
}

/*
 * 返回数组的维度数组指针
 */
static inline npy_intp *
PyArray_SHAPE(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->dimensions;
}

/*
 * 设置数组的指定 flags，不进行检查，假定用户知道自己在做什么
 */
static inline void
PyArray_ENABLEFLAGS(PyArrayObject *arr, int flags)
{
    ((PyArrayObject_fields *)arr)->flags |= flags;
}

/*
 * 清除数组的指定 flags，不进行检查，假定用户知道自己在做什么
 */
static inline void
PyArray_CLEARFLAGS(PyArrayObject *arr, int flags)
{
    ((PyArrayObject_fields *)arr)->flags &= ~flags;
}

#if NPY_FEATURE_VERSION >= NPY_1_22_API_VERSION
    /*
     * 返回数组的内存处理器对象（如果支持）
     */
    static inline NPY_RETURNS_BORROWED_REF PyObject *
    PyArray_HANDLER(PyArrayObject *arr)
    {
        return ((PyArrayObject_fields *)arr)->mem_handler;
    }
#endif

/*
 * 检查数组的数据类型是否为布尔型
 * 返回值：如果数组的数据类型为 NPY_BOOL，则返回非零值，否则返回零
 */
#define PyTypeNum_ISBOOL(type) ((type) == NPY_BOOL)
# 定义宏：检查数据类型是否为无符号整数类型
#define PyTypeNum_ISUNSIGNED(type) (((type) == NPY_UBYTE) ||   \
                                 ((type) == NPY_USHORT) ||     \
                                 ((type) == NPY_UINT) ||       \
                                 ((type) == NPY_ULONG) ||      \
                                 ((type) == NPY_ULONGLONG))

# 定义宏：检查数据类型是否为有符号整数类型
#define PyTypeNum_ISSIGNED(type) (((type) == NPY_BYTE) ||      \
                               ((type) == NPY_SHORT) ||        \
                               ((type) == NPY_INT) ||          \
                               ((type) == NPY_LONG) ||         \
                               ((type) == NPY_LONGLONG))

# 定义宏：检查数据类型是否为整数类型
#define PyTypeNum_ISINTEGER(type) (((type) >= NPY_BYTE) &&     \
                                ((type) <= NPY_ULONGLONG))

# 定义宏：检查数据类型是否为浮点数类型
#define PyTypeNum_ISFLOAT(type) ((((type) >= NPY_FLOAT) && \
                              ((type) <= NPY_LONGDOUBLE)) || \
                              ((type) == NPY_HALF))

# 定义宏：检查数据类型是否为数字类型（包括整数和浮点数）
#define PyTypeNum_ISNUMBER(type) (((type) <= NPY_CLONGDOUBLE) || \
                                  ((type) == NPY_HALF))

# 定义宏：检查数据类型是否为字符串类型（包括字节字符串和Unicode字符串）
#define PyTypeNum_ISSTRING(type) (((type) == NPY_STRING) ||    \
                                  ((type) == NPY_UNICODE))

# 定义宏：检查数据类型是否为复数类型
#define PyTypeNum_ISCOMPLEX(type) (((type) >= NPY_CFLOAT) &&   \
                                ((type) <= NPY_CLONGDOUBLE))

# 定义宏：检查数据类型是否为灵活类型（如字符串或Void类型）
#define PyTypeNum_ISFLEXIBLE(type) (((type) >=NPY_STRING) &&  \
                                    ((type) <=NPY_VOID))

# 定义宏：检查数据类型是否为日期时间类型
#define PyTypeNum_ISDATETIME(type) (((type) >=NPY_DATETIME) &&  \
                                    ((type) <=NPY_TIMEDELTA))

# 定义宏：检查数据类型是否为用户自定义类型
#define PyTypeNum_ISUSERDEF(type) (((type) >= NPY_USERDEF) && \
                                   ((type) < NPY_USERDEF+     \
                                    NPY_NUMUSERTYPES))

# 定义宏：检查数据类型是否为扩展类型（包括灵活类型和用户自定义类型）
#define PyTypeNum_ISEXTENDED(type) (PyTypeNum_ISFLEXIBLE(type) ||  \
                                    PyTypeNum_ISUSERDEF(type))

# 定义宏：检查数据类型是否为对象类型
#define PyTypeNum_ISOBJECT(type) ((type) == NPY_OBJECT)


# 定义宏：检查数据类型是否为传统（遗留）数据类型
#define PyDataType_ISLEGACY(dtype) ((dtype)->type_num < NPY_VSTRING && ((dtype)->type_num >= 0))

# 定义宏：检查数据类型是否为布尔类型
#define PyDataType_ISBOOL(obj) PyTypeNum_ISBOOL(((PyArray_Descr*)(obj))->type_num)

# 定义宏：检查数据类型是否为无符号整数类型
#define PyDataType_ISUNSIGNED(obj) PyTypeNum_ISUNSIGNED(((PyArray_Descr*)(obj))->type_num)

# 定义宏：检查数据类型是否为有符号整数类型
#define PyDataType_ISSIGNED(obj) PyTypeNum_ISSIGNED(((PyArray_Descr*)(obj))->type_num)

# 定义宏：检查数据类型是否为整数类型
#define PyDataType_ISINTEGER(obj) PyTypeNum_ISINTEGER(((PyArray_Descr*)(obj))->type_num )

# 定义宏：检查数据类型是否为浮点数类型
#define PyDataType_ISFLOAT(obj) PyTypeNum_ISFLOAT(((PyArray_Descr*)(obj))->type_num)

# 定义宏：检查数据类型是否为数字类型（包括整数和浮点数）
#define PyDataType_ISNUMBER(obj) PyTypeNum_ISNUMBER(((PyArray_Descr*)(obj))->type_num)

# 定义宏：检查数据类型是否为字符串类型（包括字节字符串和Unicode字符串）
#define PyDataType_ISSTRING(obj) PyTypeNum_ISSTRING(((PyArray_Descr*)(obj))->type_num)

# 定义宏：检查数据类型是否为复数类型
#define PyDataType_ISCOMPLEX(obj) PyTypeNum_ISCOMPLEX(((PyArray_Descr*)(obj))->type_num)

# 定义宏：检查数据类型是否为灵活类型（如字符串或Void类型）
#define PyDataType_ISFLEXIBLE(obj) PyTypeNum_ISFLEXIBLE(((PyArray_Descr*)(obj))->type_num)

# 定义宏：检查数据类型是否为日期时间类型
#define PyDataType_ISDATETIME(obj) PyTypeNum_ISDATETIME(((PyArray_Descr*)(obj))->type_num)
/*
 * 定义了一系列用于检查和操作 NumPy 数组数据类型和标志的宏。
 */

#define PyDataType_ISUSERDEF(obj) PyTypeNum_ISUSERDEF(((PyArray_Descr*)(obj))->type_num)
// 检查给定数据类型对象是否为用户定义类型

#define PyDataType_ISEXTENDED(obj) PyTypeNum_ISEXTENDED(((PyArray_Descr*)(obj))->type_num)
// 检查给定数据类型对象是否为扩展类型

#define PyDataType_ISOBJECT(obj) PyTypeNum_ISOBJECT(((PyArray_Descr*)(obj))->type_num)
// 检查给定数据类型对象是否为对象类型

#define PyDataType_MAKEUNSIZED(dtype) ((dtype)->elsize = 0)
// 将给定数据类型对象标记为无大小，通常用于灵活数据类型

/*
 * PyDataType_* FLAGS, FLACHK, REFCHK, HASFIELDS, HASSUBARRAY, UNSIZED,
 * SUBARRAY, NAMES, FIELDS, C_METADATA, and METADATA require version specific
 * lookup and are defined in npy_2_compat.h.
 */
// 这些宏定义需要特定版本的查找，并在 npy_2_compat.h 中定义

#define PyArray_ISBOOL(obj) PyTypeNum_ISBOOL(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为布尔类型

#define PyArray_ISUNSIGNED(obj) PyTypeNum_ISUNSIGNED(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为无符号整数类型

#define PyArray_ISSIGNED(obj) PyTypeNum_ISSIGNED(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为有符号整数类型

#define PyArray_ISINTEGER(obj) PyTypeNum_ISINTEGER(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为整数类型

#define PyArray_ISFLOAT(obj) PyTypeNum_ISFLOAT(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为浮点数类型

#define PyArray_ISNUMBER(obj) PyTypeNum_ISNUMBER(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为数字类型

#define PyArray_ISSTRING(obj) PyTypeNum_ISSTRING(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为字符串类型

#define PyArray_ISCOMPLEX(obj) PyTypeNum_ISCOMPLEX(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为复数类型

#define PyArray_ISFLEXIBLE(obj) PyTypeNum_ISFLEXIBLE(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为灵活数据类型

#define PyArray_ISDATETIME(obj) PyTypeNum_ISDATETIME(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为日期时间类型

#define PyArray_ISUSERDEF(obj) PyTypeNum_ISUSERDEF(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为用户定义类型

#define PyArray_ISEXTENDED(obj) PyTypeNum_ISEXTENDED(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为扩展类型

#define PyArray_ISOBJECT(obj) PyTypeNum_ISOBJECT(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为对象类型

#define PyArray_HASFIELDS(obj) PyDataType_HASFIELDS(PyArray_DESCR(obj))
// 检查给定数组对象是否具有字段描述符

/*
 * FIXME: This should check for a flag on the data-type that
 * states whether or not it is variable length.  Because the
 * ISFLEXIBLE check is hard-coded to the built-in data-types.
 */
#define PyArray_ISVARIABLE(obj) PyTypeNum_ISFLEXIBLE(PyArray_TYPE(obj))
// 检查给定数组对象的类型是否为可变长度类型

#define PyArray_SAFEALIGNEDCOPY(obj) (PyArray_ISALIGNED(obj) && !PyArray_ISVARIABLE(obj))
// 检查给定数组对象是否是对齐的且不是可变长度类型

#define NPY_LITTLE '<'
// 定义小端字节顺序符号

#define NPY_BIG '>'
// 定义大端字节顺序符号

#define NPY_NATIVE '='
// 定义本地字节顺序符号

#define NPY_SWAP 's'
// 定义交换字节顺序符号

#define NPY_IGNORE '|'
// 定义忽略字节顺序符号

#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN
#define NPY_NATBYTE NPY_BIG
#define NPY_OPPBYTE NPY_LITTLE
#else
#define NPY_NATBYTE NPY_LITTLE
#define NPY_OPPBYTE NPY_BIG
#endif

#define PyArray_ISNBO(arg) ((arg) != NPY_OPPBYTE)
// 检查参数是否与本地字节顺序匹配

#define PyArray_IsNativeByteOrder PyArray_ISNBO
// 别名，检查参数是否与本地字节顺序匹配

#define PyArray_ISNOTSWAPPED(m) PyArray_ISNBO(PyArray_DESCR(m)->byteorder)
// 检查给定数组对象的描述符的字节顺序是否与本地字节顺序匹配

#define PyArray_ISBYTESWAPPED(m) (!PyArray_ISNOTSWAPPED(m))
// 检查给定数组对象是否具有交换字节顺序的描述符

#define PyArray_FLAGSWAP(m, flags) (PyArray_CHKFLAGS(m, flags) &&       \
                                    PyArray_ISNOTSWAPPED(m))
// 检查给定数组对象是否符合特定标志并且没有交换字节顺序

#define PyArray_ISCARRAY(m) PyArray_FLAGSWAP(m, NPY_ARRAY_CARRAY)
// 检查给定数组对象是否具有 C 风格数组标志

#define PyArray_ISCARRAY_RO(m) PyArray_FLAGSWAP(m, NPY_ARRAY_CARRAY_RO)
// 检查给定数组对象是否具有只读 C 风格数组标志

#define PyArray_ISFARRAY(m) PyArray_FLAGSWAP(m, NPY_ARRAY_FARRAY)
// 检查给定数组对象是否具有 Fortran 风格数组标志

#define PyArray_ISFARRAY_RO(m) PyArray_FLAGSWAP(m, NPY_ARRAY_FARRAY_RO)
// 检查给定数组对象是否具有只读 Fortran 风格数组标志

#define PyArray_ISBEHAVED(m) PyArray_FLAGSWAP(m, NPY_ARRAY_BEHAVED)
// 检查给定数组对象是否具有规范行为标志

#define PyArray_ISBEHAVED_RO(m) PyArray_FLAGSWAP(m, NPY_ARRAY_ALIGNED)
// 检查给定数组对象是否具有只读规范行为标志
/*
 * 宏定义：检查数据类型是否未交换字节序
 */
#define PyDataType_ISNOTSWAPPED(d) PyArray_ISNBO(((PyArray_Descr *)(d))->byteorder)

/*
 * 宏定义：检查数据类型是否已交换字节序
 */
#define PyDataType_ISBYTESWAPPED(d) (!PyDataType_ISNOTSWAPPED(d))

/************************************************************
 * PyArray_CreateSortedStridePerm 中使用的结构体，从 1.7 版本开始引入。
 ************************************************************/

typedef struct {
    npy_intp perm, stride;
} npy_stride_sort_item;

/************************************************************
 * 存储在数组的 __array_struct__ 属性返回的 PyCapsule 中的结构体形式。
 * 查看完整文档请参考 https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
 ************************************************************/
typedef struct {
    int two;              /*
                           * 作为健全性检查，包含整数 2
                           */

    int nd;               /* 维度数量 */

    char typekind;        /*
                           * 数组中的数据类型种类代码，对应于 typestr 的字符代码
                           */

    int itemsize;         /* 每个元素的大小 */

    int flags;            /*
                           * 数据解释方式的标志位。有效的标志有 CONTIGUOUS (1),
                           * F_CONTIGUOUS (2), ALIGNED (0x100), NOTSWAPPED (0x200),
                           * WRITEABLE (0x400)。ARR_HAS_DESCR (0x800) 表示结构中有 arrdescr 字段
                           */

    npy_intp *shape;       /*
                            * 长度为 nd 的形状信息数组
                            */

    npy_intp *strides;    /* 长度为 nd 的步幅信息数组 */

    void *data;           /* 数组第一个元素的指针 */

    PyObject *descr;      /*
                           * 字段列表或 NULL（如果 flags 没有设置 ARR_HAS_DESCR 标志则忽略）
                           */
} PyArrayInterface;

/****************************************
 * NpyString
 *
 * NpyString API 使用的类型。
 ****************************************/

/*
 * "packed" 编码字符串。访问字符串数据时必须先解包字符串。
 */
typedef struct npy_packed_static_string npy_packed_static_string;

/*
 * 对打包字符串中数据的无修改只读视图。
 */
typedef struct npy_unpacked_static_string {
    size_t size;
    const char *buf;
} npy_static_string;

/*
 * 处理静态字符串的堆分配。
 */
typedef struct npy_string_allocator npy_string_allocator;

typedef struct {
    PyArray_Descr base;
    // 表示空值的对象
    PyObject *na_object;
    // 标志，指示是否将任意对象强制转换为字符串
    char coerce;
    // 表示 na 对象是否类似于 NaN
    char has_nan_na;
    // 表示 na 对象是否为字符串
    char has_string_na;
    // 若非零，则表示此实例已被数组所拥有
    char array_owned;
    // 当需要默认字符串时使用的字符串数据
    npy_static_string default_string;
    // 若存在，表示缺失数据对象的名称
    npy_static_string na_name;
    // 分配器应当仅在获取 allocator_lock 后直接访问，
    // 并在 allocator 不再需要时立即释放锁
    npy_string_allocator *allocator;
/*
 * PyArray_StringDTypeObject: This structure definition likely represents
 *                           a data type object for string arrays in NumPy.
 */

/*
 * PyArray_DTypeMeta related definitions:
 *
 * As of now, this API is preliminary and will be extended as necessary.
 */
#if defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD
    /*
     * The Structures defined in this block are currently considered
     * private API and may change without warning!
     * Part of this (at least the size) is expected to be public API without
     * further modifications.
     */
    /* TODO: Make this definition public in the API, as soon as it's settled */

    /*
     * PyArrayDTypeMeta_Type: This variable likely defines the type object
     *                        for the meta information related to data types
     *                        in NumPy arrays.
     */

    /*
     * PyArray_DTypeMeta: Structure defining metadata for NumPy data types.
     *                    It extends PyHeapTypeObject and includes fields
     *                    such as singleton, type_num, scalar_type, flags,
     *                    dt_slots, and reserved, aimed at providing a stable
     *                    ABI for static and opaque API usage.
     */
#endif  /* NPY_INTERNAL_BUILD */


/*
 * Use the keyword NPY_DEPRECATED_INCLUDES to ensure that the header files
 * npy_*_*_deprecated_api.h are only included from here and nowhere else.
 */
#ifdef NPY_DEPRECATED_INCLUDES
#error "Do not use the reserved keyword NPY_DEPRECATED_INCLUDES."
#endif
#define NPY_DEPRECATED_INCLUDES

/*
 * Conditional inclusion of deprecated API headers based on NumPy version.
 * Includes npy_1_7_deprecated_api.h for versions below 1.7 unless deprecated
 * APIs are explicitly disabled (NPY_NO_DEPRECATED_API).
 */
#if !defined(NPY_NO_DEPRECATED_API) || \
    (NPY_NO_DEPRECATED_API < NPY_1_7_API_VERSION)
#include "npy_1_7_deprecated_api.h"
#endif

/*
 * There is no file npy_1_8_deprecated_api.h since there are no additional
 * deprecated API features in NumPy 1.8.
 *
 * Note to maintainers: insert code like the following in future NumPy
 * versions.
 *
 * #if !defined(NPY_NO_DEPRECATED_API) || \
 *     (NPY_NO_DEPRECATED_API < NPY_1_9_API_VERSION)
 * #include "npy_1_9_deprecated_api.h"
 * #endif
 */
#undef NPY_DEPRECATED_INCLUDES
取消预处理器定义 NPY_DEPRECATED_INCLUDES，如果之前定义过的话。
```