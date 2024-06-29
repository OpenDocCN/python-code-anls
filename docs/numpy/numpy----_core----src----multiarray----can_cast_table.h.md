# `.\numpy\numpy\_core\src\multiarray\can_cast_table.h`

```py
/*
 * This file defines a compile time constant casting table for use in
 * a few situations:
 * 1. As a fast-path in can-cast (untested how much it helps).
 * 2. To define the actual cast safety stored on the CastingImpl/ArrayMethod
 * 3. For scalar math, since it also needs cast safety information.
 *
 * It is useful to have this constant to allow writing compile time generic
 * code based on cast safety in the scalar math code.
 */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_CAN_CAST_TABLE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CAN_CAST_TABLE_H_

#include "numpy/ndarraytypes.h"

/* The from type fits into to (it has a smaller or equal number of bits) */
#define FITS(FROM, TO) (NPY_SIZEOF_##FROM <= NPY_SIZEOF_##TO)
/* Unsigned "from" fits a signed integer if it is truly smaller */
#define UFITS(FROM, TO) (NPY_SIZEOF_##FROM < NPY_SIZEOF_##TO)
/* Integer "from" only fits a float if it is truly smaller or double... */
#define IFITS(FROM, TO) (  \
    NPY_SIZEOF_##FROM < NPY_SIZEOF_##TO || (  \
            NPY_SIZEOF_##FROM == NPY_SIZEOF_##TO  \
            && NPY_SIZEOF_##FROM >= NPY_SIZEOF_DOUBLE))

/*
 * NOTE: The Order is bool, integers (signed, unsigned) tuples, float, cfloat,
 *       then 6 fixed ones (object, string, unicode, void, datetime, timedelta),
 *       and finally half.
 *       Note that in the future we may only need the numeric casts here, but
 *       currently it fills in the others as well.
 */
#define CASTS_SAFELY_FROM_UINT(FROM)  \
    {0,  \
     UFITS(FROM, BYTE), FITS(FROM, BYTE), UFITS(FROM, SHORT), FITS(FROM, SHORT),  \
     UFITS(FROM, INT), FITS(FROM, INT), UFITS(FROM, LONG), FITS(FROM, LONG),  \
     UFITS(FROM, LONGLONG), FITS(FROM, LONGLONG),  \
     IFITS(FROM, FLOAT), IFITS(FROM, DOUBLE), IFITS(FROM, LONGDOUBLE),  \
     IFITS(FROM, FLOAT), IFITS(FROM, DOUBLE), IFITS(FROM, LONGDOUBLE),  \
     1, 1, 1, 1, 0, NPY_SIZEOF_##FROM < NPY_SIZEOF_TIMEDELTA, IFITS(FROM, HALF)}

#define CASTS_SAFELY_FROM_INT(FROM)  \
    {0,  \
     FITS(FROM, BYTE), 0, FITS(FROM, SHORT), 0,  \
     FITS(FROM, INT), 0, FITS(FROM, LONG), 0,  \
     FITS(FROM, LONGLONG), 0,  \
     IFITS(FROM, FLOAT), IFITS(FROM, DOUBLE), IFITS(FROM, LONGDOUBLE),  \
     IFITS(FROM, FLOAT), IFITS(FROM, DOUBLE), IFITS(FROM, LONGDOUBLE),  \
     1, 1, 1, 1, 0, NPY_SIZEOF_##FROM <= NPY_SIZEOF_TIMEDELTA, IFITS(FROM, HALF)}

/* Floats are similar to ints, but cap at double */
#define CASTS_SAFELY_FROM_FLOAT(FROM)  \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  \
     FITS(FROM, FLOAT), FITS(FROM, DOUBLE), FITS(FROM, LONGDOUBLE),  \
     FITS(FROM, FLOAT), FITS(FROM, DOUBLE), FITS(FROM, LONGDOUBLE),  \
     1, 1, 1, 1, 0, 0, FITS(FROM, HALF)}

#define CASTS_SAFELY_FROM_CFLOAT(FROM)  \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  \
     0, 0, 0,  \
     FITS(FROM, FLOAT), FITS(FROM, DOUBLE), FITS(FROM, LONGDOUBLE),  \
     1, 1, 1, 1, 0, 0, 0}

#endif  // NUMPY_CORE_SRC_MULTIARRAY_CAN_CAST_TABLE_H_


注释：
/*
 * 安全类型转换表格 `_npy_can_cast_safely_table`
 * 这个表格描述了NumPy中每种数据类型之间的安全转换关系。
 * 表格的行和列代表不同的数据类型，每个元素指示从行类型到列类型的转换是否安全。
 */

static const npy_bool _npy_can_cast_safely_table[NPY_NTYPES_LEGACY][NPY_NTYPES_LEGACY] = {
        /* Bool 安全转换到除了 datetime（没有零值）之外的任何类型 */
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 0, 1, 1},
        /* 整数类型，分为有符号和无符号 */
        CASTS_SAFELY_FROM_INT(BYTE), CASTS_SAFELY_FROM_UINT(BYTE),
        CASTS_SAFELY_FROM_INT(SHORT), CASTS_SAFELY_FROM_UINT(SHORT),
        CASTS_SAFELY_FROM_INT(INT), CASTS_SAFELY_FROM_UINT(INT),
        CASTS_SAFELY_FROM_INT(LONG), CASTS_SAFELY_FROM_UINT(LONG),
        CASTS_SAFELY_FROM_INT(LONGLONG), CASTS_SAFELY_FROM_UINT(LONGLONG),
        /* 浮点数和复数 */
        CASTS_SAFELY_FROM_FLOAT(FLOAT),
        CASTS_SAFELY_FROM_FLOAT(DOUBLE),
        CASTS_SAFELY_FROM_FLOAT(LONGDOUBLE),
        CASTS_SAFELY_FROM_CFLOAT(FLOAT),
        CASTS_SAFELY_FROM_CFLOAT(DOUBLE),
        CASTS_SAFELY_FROM_CFLOAT(LONGDOUBLE),
        /*
         * 主要的数值类型后面是：
         * object, string, unicode, void, datetime, timedelta (以及 half)
         */
        /* object 类型只能安全转换到它自己 */
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* bool + ints */
         0, 0, 0, 0, 0, 0,  /* floats (without half) */
         1, 0, 0, 0, 0, 0, 0},
        /* string 类型可以安全转换到 object, unicode 和 void */
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* bool + ints */
         0, 0, 0, 0, 0, 0,  /* floats (without half) */
         1, 1, 1, 1, 0, 0, 0},
        /* unicode 类型可以安全转换到 object 和 void */
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* bool + ints */
         0, 0, 0, 0, 0, 0,  /* floats (without half) */
         1, 0, 1, 1, 0, 0, 0},
        /* void 类型可以安全转换到 object */
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* bool + ints */
         0, 0, 0, 0, 0, 0,  /* floats (without half) */
         1, 0, 0, 1, 0, 0, 0},
        /* datetime 类型可以安全转换到 object, string, unicode, void */
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* bool + ints */
         0, 0, 0, 0, 0, 0,  /* floats (without half) */
         1, 1, 1, 1, 1, 0, 0},
        /* timedelta 类型可以安全转换到 object, string, unicode, void */
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* bool + ints */
         0, 0, 0, 0, 0, 0,  /* floats (without half) */
         1, 1, 1, 1, 0, 1, 0},
        /* half 类型 */
        CASTS_SAFELY_FROM_FLOAT(HALF),
};

#undef FITS
#undef UFITS
#undef IFITS
#undef CASTS_SAFELY_TO_UINT
#undef CASTS_SAFELY_TO_INT
#undef CASTS_SAFELY_TO_FLOAT
#undef CASTS_SAFELY_TO_CFLOAT

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CAN_CAST_TABLE_H_ */
```