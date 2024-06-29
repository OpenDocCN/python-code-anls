# `.\numpy\numpy\_core\src\multiarray\nditer_impl.h`

```py
/*
 * This is a PRIVATE INTERNAL NumPy header, intended to be used *ONLY*
 * by the iterator implementation code. All other internal NumPy code
 * should use the exposed iterator API.
 */
#ifndef NPY_ITERATOR_IMPLEMENTATION_CODE
#error This header is intended for use ONLY by iterator implementation code.
#endif

#ifndef NUMPY_CORE_SRC_MULTIARRAY_NDITER_IMPL_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NDITER_IMPL_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"

#include "convert_datatype.h"

#include "lowlevel_strided_loops.h"
#include "dtype_transfer.h"
#include "dtype_traversal.h"


/********** ITERATOR CONSTRUCTION TIMING **************/
#define NPY_IT_CONSTRUCTION_TIMING 0

#if NPY_IT_CONSTRUCTION_TIMING
#define NPY_IT_TIME_POINT(var) { \
            unsigned int hi, lo; \
            __asm__ __volatile__ ( \
                "rdtsc" \
                : "=d" (hi), "=a" (lo)); \
            var = (((unsigned long long)hi) << 32) | lo; \
        }
#define NPY_IT_PRINT_TIME_START(var) { \
            printf("%30s: start\n", #var); \
            c_temp = var; \
        }
#define NPY_IT_PRINT_TIME_VAR(var) { \
            printf("%30s: %6.0f clocks\n", #var, \
                    ((double)(var-c_temp))); \
            c_temp = var; \
        }
#else
#define NPY_IT_TIME_POINT(var)
#endif

/******************************************************/

/********** PRINTF DEBUG TRACING **************/
#define NPY_IT_DBG_TRACING 0

#if NPY_IT_DBG_TRACING
#define NPY_IT_DBG_PRINT(s) printf("%s", s)
#define NPY_IT_DBG_PRINT1(s, p1) printf(s, p1)
#define NPY_IT_DBG_PRINT2(s, p1, p2) printf(s, p1, p2)
#define NPY_IT_DBG_PRINT3(s, p1, p2, p3) printf(s, p1, p2, p3)
#else
#define NPY_IT_DBG_PRINT(s)
#define NPY_IT_DBG_PRINT1(s, p1)
#define NPY_IT_DBG_PRINT2(s, p1, p2)
#define NPY_IT_DBG_PRINT3(s, p1, p2, p3)
#endif
/**********************************************/

/* Rounds up a number of bytes to be divisible by sizeof intptr_t */
#if NPY_SIZEOF_PY_INTPTR_T == 4
#define NPY_PTR_ALIGNED(size) ((size + 0x3)&(-0x4))
#else
#define NPY_PTR_ALIGNED(size) ((size + 0x7)&(-0x8))
#endif

/* Internal iterator flags */

/* The perm is the identity */
#define NPY_ITFLAG_IDENTPERM    (1 << 0)
/* The perm has negative entries (indicating flipped axes) */
#define NPY_ITFLAG_NEGPERM      (1 << 1)
/* The iterator is tracking an index */
#define NPY_ITFLAG_HASINDEX     (1 << 2)
/* The iterator is tracking a multi-index */
#define NPY_ITFLAG_HASMULTIINDEX    (1 << 3)
/* The iteration order was forced on construction */
#define NPY_ITFLAG_FORCEDORDER  (1 << 4)
/* The inner loop is handled outside the iterator */
#define NPY_ITFLAG_EXLOOP       (1 << 5)
/* The iterator is ranged */
#define NPY_ITFLAG_RANGE        (1 << 6)
/* The iterator is buffered */
#define NPY_ITFLAG_BUFFER       (1 << 7)
/* The iterator should grow the buffered inner loop when possible */
/* 定义一个位掩码，用于表示迭代器需要的各种特性 */
#define NPY_ITFLAG_GROWINNER    (1 << 8)
/* 仅有一个迭代，可以为其特化iternext */
#define NPY_ITFLAG_ONEITERATION (1 << 9)
/* 延迟缓冲区分配，直到第一次调用Reset* */
#define NPY_ITFLAG_DELAYBUF     (1 << 10)
/* 迭代过程中需要 API 访问 */
#define NPY_ITFLAG_NEEDSAPI     (1 << 11)
/* 迭代包括一个或多个操作数的缩减 */
#define NPY_ITFLAG_REDUCE       (1 << 12)
/* 缩减迭代下次不需要重新计算缩减循环 */
#define NPY_ITFLAG_REUSE_REDUCE_LOOPS (1 << 13)

/*
 * 用于所有传输函数的(组合)ArrayMethod标志的偏移量。
 * 目前，我们使用最高8位。
 */
#define NPY_ITFLAG_TRANSFERFLAGS_SHIFT 24

/* 内部迭代器每操作数的迭代器标志 */

/* 操作数将被写入 */
#define NPY_OP_ITFLAG_WRITE        0x0001
/* 操作数将被读取 */
#define NPY_OP_ITFLAG_READ         0x0002
/* 需要类型转换/字节交换/对齐 */
#define NPY_OP_ITFLAG_CAST         0x0004
/* 操作数从不需要缓冲 */
#define NPY_OP_ITFLAG_BUFNEVER     0x0008
/* 操作数已对齐 */
#define NPY_OP_ITFLAG_ALIGNED      0x0010
/* 操作数正在被缩减 */
#define NPY_OP_ITFLAG_REDUCE       0x0020
/* 操作数用于临时使用，没有后备数组 */
#define NPY_OP_ITFLAG_VIRTUAL      0x0040
/* 复制缓冲区到数组时需要掩码 */
#define NPY_OP_ITFLAG_WRITEMASKED  0x0080
/* 操作数的数据指针指向其缓冲区 */
#define NPY_OP_ITFLAG_USINGBUFFER  0x0100
/* 必须复制操作数（如果还有ITFLAG_WRITE，则使用UPDATEIFCOPY） */
#define NPY_OP_ITFLAG_FORCECOPY    0x0200
/* 操作数具有临时数据，在dealloc时写回 */
#define NPY_OP_ITFLAG_HAS_WRITEBACK 0x0400

/*
 * 迭代器的数据布局由三元组(itflags, ndim, nop)完全指定。
 * 这三个变量预期在调用这些宏的所有函数中存在，
 * 要么作为从迭代器初始化的真实变量，要么作为专门函数（如各种iternext函数）中的常量。
 */

struct NpyIter_InternalOnly {
    /* 初始固定位置数据 */
    npy_uint32 itflags;
    npy_uint8 ndim, nop;
    npy_int8 maskop;
    npy_intp itersize, iterstart, iterend;
    /* 只有在设置了RANGED或BUFFERED时才使用iterindex */
    npy_intp iterindex;
    /* 剩余的是变量数据 */
    char iter_flexdata[];
};

typedef struct NpyIter_AxisData_tag NpyIter_AxisData;
typedef struct NpyIter_TransferInfo_tag NpyIter_TransferInfo;
typedef struct NpyIter_BufferData_tag NpyIter_BufferData;

typedef npy_int16 npyiter_opitflags;

/* 迭代器成员的字节大小 */
#define NIT_PERM_SIZEOF(itflags, ndim, nop) \
        NPY_PTR_ALIGNED(NPY_MAXDIMS)
#define NIT_DTYPES_SIZEOF(itflags, ndim, nop) \
        ((NPY_SIZEOF_PY_INTPTR_T)*(nop))
/* 计算并返回重置数据指针所需的字节数，考虑指针的数量和大小 */
#define NIT_RESETDATAPTR_SIZEOF(itflags, ndim, nop) \
        ((NPY_SIZEOF_PY_INTPTR_T)*(nop+1))

/* 计算并返回基础偏移量所需的字节数，考虑指针的数量和大小 */
#define NIT_BASEOFFSETS_SIZEOF(itflags, ndim, nop) \
        ((NPY_SIZEOF_PY_INTPTR_T)*(nop+1))  /* 可能是 intp 的大小 */

/* 计算并返回操作数数组所需的字节数，考虑指针的数量和大小 */
#define NIT_OPERANDS_SIZEOF(itflags, ndim, nop) \
        ((NPY_SIZEOF_PY_INTPTR_T)*(nop))

/* 计算并返回操作标志数组所需的字节数，确保对齐到指针的大小 */
#define NIT_OPITFLAGS_SIZEOF(itflags, ndim, nop) \
        (NPY_PTR_ALIGNED(sizeof(npyiter_opitflags) * nop))

/* 计算并返回缓冲区数据结构所需的字节数，考虑缓冲区标志位和相关指针数量 */
#define NIT_BUFFERDATA_SIZEOF(itflags, ndim, nop) \
        ((itflags&NPY_ITFLAG_BUFFER) ? ( \
            (NPY_SIZEOF_PY_INTPTR_T)*(6 + 5*nop) + sizeof(NpyIter_TransferInfo) * nop) : 0)

/* 从迭代器的灵活数据开始计算，返回排列数据的字节偏移量 */
#define NIT_PERM_OFFSET() \
        (0)

/* 计算并返回数据类型数组的字节偏移量，考虑迭代器标志、维度和操作数指针的数量 */
#define NIT_DTYPES_OFFSET(itflags, ndim, nop) \
        (NIT_PERM_OFFSET() + \
         NIT_PERM_SIZEOF(itflags, ndim, nop))

/* 计算并返回重置数据指针的字节偏移量，考虑迭代器标志、维度和操作数指针的数量 */
#define NIT_RESETDATAPTR_OFFSET(itflags, ndim, nop) \
        (NIT_DTYPES_OFFSET(itflags, ndim, nop) + \
         NIT_DTYPES_SIZEOF(itflags, ndim, nop))

/* 计算并返回基础偏移量的字节偏移量，考虑迭代器标志、维度和操作数指针的数量 */
#define NIT_BASEOFFSETS_OFFSET(itflags, ndim, nop) \
        (NIT_RESETDATAPTR_OFFSET(itflags, ndim, nop) + \
         NIT_RESETDATAPTR_SIZEOF(itflags, ndim, nop))

/* 计算并返回操作数数组的字节偏移量，考虑迭代器标志、维度和操作数指针的数量 */
#define NIT_OPERANDS_OFFSET(itflags, ndim, nop) \
        (NIT_BASEOFFSETS_OFFSET(itflags, ndim, nop) + \
         NIT_BASEOFFSETS_SIZEOF(itflags, ndim, nop))

/* 计算并返回操作标志数组的字节偏移量，确保对齐到指针的大小，考虑迭代器标志、维度和操作数指针的数量 */
#define NIT_OPITFLAGS_OFFSET(itflags, ndim, nop) \
        (NIT_OPERANDS_OFFSET(itflags, ndim, nop) + \
         NIT_OPERANDS_SIZEOF(itflags, ndim, nop))

/* 计算并返回缓冲区数据结构的字节偏移量，考虑迭代器标志、维度和操作数指针的数量 */
#define NIT_BUFFERDATA_OFFSET(itflags, ndim, nop) \
        (NIT_OPITFLAGS_OFFSET(itflags, ndim, nop) + \
         NIT_OPITFLAGS_SIZEOF(itflags, ndim, nop))

/* 计算并返回轴数据结构的字节偏移量，考虑迭代器标志、维度和操作数指针的数量 */
#define NIT_AXISDATA_OFFSET(itflags, ndim, nop) \
        (NIT_BUFFERDATA_OFFSET(itflags, ndim, nop) + \
         NIT_BUFFERDATA_SIZEOF(itflags, ndim, nop))

/* 返回迭代器中 itflags 成员的值 */
#define NIT_ITFLAGS(iter) \
        ((iter)->itflags)

/* 返回迭代器中 ndim 成员的值 */
#define NIT_NDIM(iter) \
        ((iter)->ndim)

/* 返回迭代器中 nop 成员的值 */
#define NIT_NOP(iter) \
        ((iter)->nop)

/* 返回迭代器中 maskop 成员的值 */
#define NIT_MASKOP(iter) \
        ((iter)->maskop)

/* 返回迭代器中 itersize 成员的值 */
#define NIT_ITERSIZE(iter) \
        (iter->itersize)

/* 返回迭代器中 iterstart 成员的值 */
#define NIT_ITERSTART(iter) \
        (iter->iterstart)

/* 返回迭代器中 iterend 成员的值 */
#define NIT_ITEREND(iter) \
        (iter->iterend)

/* 返回迭代器中 iterindex 成员的值 */
#define NIT_ITERINDEX(iter) \
        (iter->iterindex)

/* 返回迭代器中排列数据的指针 */
#define NIT_PERM(iter)  ((npy_int8 *)( \
        iter->iter_flexdata + NIT_PERM_OFFSET()))

/* 返回迭代器中数据类型数组的指针 */
#define NIT_DTYPES(iter) ((PyArray_Descr **)( \
        iter->iter_flexdata + NIT_DTYPES_OFFSET(itflags, ndim, nop)))

/* 返回迭代器中重置数据指针数组的指针 */
#define NIT_RESETDATAPTR(iter) ((char **)( \
        iter->iter_flexdata + NIT_RESETDATAPTR_OFFSET(itflags, ndim, nop)))

/* 返回迭代器中基础偏移量数组的指针 */
#define NIT_BASEOFFSETS(iter) ((npy_intp *)( \
        iter->iter_flexdata + NIT_BASEOFFSETS_OFFSET(itflags, ndim, nop)))

/* 返回迭代器中操作数数组的指针 */
#define NIT_OPERANDS(iter) ((PyArrayObject **)( \
        iter->iter_flexdata + NIT_OPERANDS_OFFSET(itflags, ndim, nop)))

/* 返回迭代器中操作标志数组的指针 */
#define NIT_OPITFLAGS(iter) ((npyiter_opitflags *)( \
        iter->iter_flexdata + NIT_OPITFLAGS_OFFSET(itflags, ndim, nop)))
#define NIT_BUFFERDATA(iter) ((NpyIter_BufferData *)( \
        iter->iter_flexdata + NIT_BUFFERDATA_OFFSET(itflags, ndim, nop)))
# 宏定义：根据迭代器的指针，返回对应的NpyIter_BufferData结构体指针

#define NIT_AXISDATA(iter) ((NpyIter_AxisData *)( \
        iter->iter_flexdata + NIT_AXISDATA_OFFSET(itflags, ndim, nop)))
# 宏定义：根据迭代器的指针，返回对应的NpyIter_AxisData结构体指针

/* Internal-only BUFFERDATA MEMBER ACCESS */

struct NpyIter_TransferInfo_tag {
    NPY_cast_info read;
    NPY_cast_info write;
    NPY_traverse_info clear;
    /* Probably unnecessary, but make sure what follows is intptr aligned: */
    Py_intptr_t _unused_ensure_alignment[];
};
# 结构体定义：定义了NpyIter_TransferInfo结构体，包含了三个成员read、write和clear，以及用于对齐的无用成员数组

struct NpyIter_BufferData_tag {
    npy_intp buffersize, size, bufiterend,
             reduce_pos, reduce_outersize, reduce_outerdim;
    Py_intptr_t bd_flexdata;
};
# 结构体定义：定义了NpyIter_BufferData结构体，包含了多个整型成员和一个灵活数据区域的指针

#define NBF_BUFFERSIZE(bufferdata) ((bufferdata)->buffersize)
# 宏定义：返回给定NpyIter_BufferData结构体指针的buffersize成员

#define NBF_SIZE(bufferdata) ((bufferdata)->size)
# 宏定义：返回给定NpyIter_BufferData结构体指针的size成员

#define NBF_BUFITEREND(bufferdata) ((bufferdata)->bufiterend)
# 宏定义：返回给定NpyIter_BufferData结构体指针的bufiterend成员

#define NBF_REDUCE_POS(bufferdata) ((bufferdata)->reduce_pos)
# 宏定义：返回给定NpyIter_BufferData结构体指针的reduce_pos成员

#define NBF_REDUCE_OUTERSIZE(bufferdata) ((bufferdata)->reduce_outersize)
# 宏定义：返回给定NpyIter_BufferData结构体指针的reduce_outersize成员

#define NBF_REDUCE_OUTERDIM(bufferdata) ((bufferdata)->reduce_outerdim)
# 宏定义：返回给定NpyIter_BufferData结构体指针的reduce_outerdim成员

#define NBF_STRIDES(bufferdata) ( \
        &(bufferdata)->bd_flexdata + 0)
# 宏定义：返回给定NpyIter_BufferData结构体指针的bd_flexdata成员指针的地址

#define NBF_PTRS(bufferdata) ((char **) \
        (&(bufferdata)->bd_flexdata + 1*(nop)))
# 宏定义：返回给定NpyIter_BufferData结构体指针的bd_flexdata成员指针数组的地址

#define NBF_REDUCE_OUTERSTRIDES(bufferdata) ( \
        (&(bufferdata)->bd_flexdata + 2*(nop)))
# 宏定义：返回给定NpyIter_BufferData结构体指针的bd_flexdata成员指针数组的地址

#define NBF_REDUCE_OUTERPTRS(bufferdata) ((char **) \
        (&(bufferdata)->bd_flexdata + 3*(nop)))
# 宏定义：返回给定NpyIter_BufferData结构体指针的bd_flexdata成员指针数组的地址

#define NBF_BUFFERS(bufferdata) ((char **) \
        (&(bufferdata)->bd_flexdata + 4*(nop)))
# 宏定义：返回给定NpyIter_BufferData结构体指针的bd_flexdata成员指针数组的地址

#define NBF_TRANSFERINFO(bufferdata) ((NpyIter_TransferInfo *) \
        (&(bufferdata)->bd_flexdata + 5*(nop)))
# 宏定义：返回给定NpyIter_BufferData结构体指针的bd_flexdata成员指针的地址转换为NpyIter_TransferInfo类型的指针

/* Internal-only AXISDATA MEMBER ACCESS. */
struct NpyIter_AxisData_tag {
    npy_intp shape, index;
    Py_intptr_t ad_flexdata;
};
# 结构体定义：定义了NpyIter_AxisData结构体，包含了shape、index和灵活数据区域的指针

#define NAD_SHAPE(axisdata) ((axisdata)->shape)
# 宏定义：返回给定NpyIter_AxisData结构体指针的shape成员

#define NAD_INDEX(axisdata) ((axisdata)->index)
# 宏定义：返回给定NpyIter_AxisData结构体指针的index成员

#define NAD_STRIDES(axisdata) ( \
        &(axisdata)->ad_flexdata + 0)
# 宏定义：返回给定NpyIter_AxisData结构体指针的ad_flexdata成员指针的地址

#define NAD_PTRS(axisdata) ((char **) \
        (&(axisdata)->ad_flexdata + 1*(nop+1)))
# 宏定义：返回给定NpyIter_AxisData结构体指针的ad_flexdata成员指针数组的地址

#define NAD_NSTRIDES() \
        ((nop) + ((itflags&NPY_ITFLAG_HASINDEX) ? 1 : 0))
# 宏定义：返回NpyIter_AxisData结构体的stride数组的大小

/* Size of one AXISDATA struct within the iterator */
#define NIT_AXISDATA_SIZEOF(itflags, ndim, nop) (( \
        /* intp shape */ \
        1 + \
        /* intp index */ \
        1 + \
        /* intp stride[nop+1] AND char* ptr[nop+1] */ \
        2*((nop)+1) \
        )*(size_t)NPY_SIZEOF_PY_INTPTR_T)
# 宏定义：返回NpyIter_AxisData结构体在迭代器中的大小，依赖于itflags、ndim和nop参数

/*
 * Macro to advance an AXISDATA pointer by a specified count.
 * Requires that sizeof_axisdata be previously initialized
 * to NIT_AXISDATA_SIZEOF(itflags, ndim, nop).
 */
#define NIT_INDEX_AXISDATA(axisdata, index) ((NpyIter_AxisData *) \
        (((char *)(axisdata)) + (index)*sizeof_axisdata))
# 宏定义：将给定的NpyIter_AxisData指针按照索引index向前移动一定步数，依赖于sizeof_axisdata的初始化

#define NIT_ADVANCE_AXISDATA(axisdata, count) \
        axisdata = NIT_INDEX_AXISDATA(axisdata, count)
# 宏定义：将给定的NpyIter_AxisData指针按照指定的count数目向前移动

/* Size of the whole iterator */
/* 定义一个宏，用于计算迭代器的大小，包括内部结构和轴数据大小 */
#define NIT_SIZEOF_ITERATOR(itflags, ndim, nop) ( \
        sizeof(struct NpyIter_InternalOnly) + \
        NIT_AXISDATA_OFFSET(itflags, ndim, nop) + \
        NIT_AXISDATA_SIZEOF(itflags, ndim, nop)*(ndim ? ndim : 1))

/* 内部辅助函数，在实现文件间共享 */

/**
 * 撤销迭代器的轴置换。当操作数的维度少于迭代器时，这可能会返回插入的（广播）维度的负值。
 *
 * @param axis 要撤销迭代器轴置换的轴。
 * @param ndim 如果使用 `op_axes`，则为迭代器的维度，否则为操作数的维度。
 * @param perm 迭代器轴置换 NIT_PERM(iter)
 * @param axis_flipped 如果这是一个翻转的轴（即以相反顺序迭代），则设置为 true，否则为 false。
 *                    如果不需要该信息，则可以为 NULL。
 * @return 未置换的轴。如果没有 `op_axes`，则是正确的；如果有 `op_axes`，则这是对 `op_axes` 的索引（未置换的迭代器轴）。
 */
static inline int
npyiter_undo_iter_axis_perm(
        int axis, int ndim, const npy_int8 *perm, npy_bool *axis_flipped)
{
    npy_int8 p = perm[axis];
    /* 迭代器以相反顺序处理轴，因此根据 ndim 进行调整 */
    npy_bool flipped = p < 0;
    if (axis_flipped != NULL) {
        *axis_flipped = flipped;
    }
    if (flipped) {
        axis = ndim + p;
    }
    else {
        axis = ndim - p - 1;
    }
    return axis;
}

/* 下面是一些不导出的函数声明，用于操作 NpyIter 结构 */

NPY_NO_EXPORT void
npyiter_coalesce_axes(NpyIter *iter);

NPY_NO_EXPORT int
npyiter_allocate_buffers(NpyIter *iter, char **errmsg);

NPY_NO_EXPORT void
npyiter_goto_iterindex(NpyIter *iter, npy_intp iterindex);

NPY_NO_EXPORT int
npyiter_copy_from_buffers(NpyIter *iter);

NPY_NO_EXPORT int
npyiter_copy_to_buffers(NpyIter *iter, char **prev_dataptrs);

NPY_NO_EXPORT void
npyiter_clear_buffers(NpyIter *iter);

/*
 * 获取传输函数的 ArrayMethod 标志位。
 * TODO: 此函数应该是公共的，并且应该从 `nditer_impl.h` 中删除，但这要求首先将 ArrayMethod 标志位公开为公共 API。
 */
NPY_NO_EXPORT int
NpyIter_GetTransferFlags(NpyIter *iter);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NDITER_IMPL_H_ */
```