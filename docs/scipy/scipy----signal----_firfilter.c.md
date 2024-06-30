# `D:\src\scipysrc\scipy\scipy\signal\_firfilter.c`

```
// 定义一个宏，指示不导入数组，包括 "numpy/ndarrayobject.h" 和 "_sigtools.h"
#define NO_IMPORT_ARRAY
// 导入必要的头文件
#include "numpy/ndarrayobject.h"
#include "_sigtools.h"
// 导入标准库头文件
#include <stdbool.h>
#include <stdint.h>

// 定义一个静态整型数组，包含各种数据类型的大小
static int elsizes[] = {sizeof(npy_bool),
                        sizeof(npy_byte),
                        sizeof(npy_ubyte),
                        sizeof(npy_short),
                        sizeof(npy_ushort),
                        sizeof(int),
                        sizeof(npy_uint),
                        sizeof(long),
                        sizeof(npy_ulong),
                        sizeof(npy_longlong),
                        sizeof(npy_ulonglong),
                        sizeof(float),
                        sizeof(double),
                        sizeof(npy_longdouble),
                        sizeof(npy_cfloat),
                        sizeof(npy_cdouble),
                        sizeof(npy_clongdouble),
                        sizeof(void *),
            0,0,0,0};

// 定义一个函数指针类型 OneMultAddFunction，接受参数类型为 (char *, char *, int64_t, char **, int64_t)
typedef void (OneMultAddFunction) (char *, char *, int64_t, char **, int64_t);

// 定义宏 MAKE_ONEMULTADD，用于生成特定数据类型的函数
#define MAKE_ONEMULTADD(fname, type) \
static void fname ## _onemultadd(char *sum, char *term1, int64_t str, char **pvals, int64_t n) { \
        type dsum = *(type*)sum; \
        for (int64_t k=0; k < n; k++) { \
          type tmp = *(type*)(term1 + k * str); \
          dsum += tmp * *(type*)pvals[k]; \
        } \
        *(type*)(sum) = dsum; \
}

// 生成多个函数，每个函数针对不同的数据类型
MAKE_ONEMULTADD(UBYTE, npy_ubyte)
MAKE_ONEMULTADD(USHORT, npy_ushort)
MAKE_ONEMULTADD(UINT, npy_uint)
MAKE_ONEMULTADD(ULONG, npy_ulong)
MAKE_ONEMULTADD(ULONGLONG, npy_ulonglong)

MAKE_ONEMULTADD(BYTE, npy_byte)
MAKE_ONEMULTADD(SHORT, short)
MAKE_ONEMULTADD(INT, int)
MAKE_ONEMULTADD(LONG, long)
MAKE_ONEMULTADD(LONGLONG, npy_longlong)

MAKE_ONEMULTADD(FLOAT, float)
MAKE_ONEMULTADD(DOUBLE, double)
MAKE_ONEMULTADD(LONGDOUBLE, npy_longdouble)

#ifdef __GNUC__
// 对于 GNU 编译器，生成复数类型的函数
MAKE_ONEMULTADD(CFLOAT, __complex__ float)
MAKE_ONEMULTADD(CDOUBLE, __complex__ double)
MAKE_ONEMULTADD(CLONGDOUBLE, __complex__ long double)
#else
// 对于非 GNU 编译器，定义复数类型的函数，分为两步计算
#define MAKE_C_ONEMULTADD(fname, type) \
static void fname ## _onemultadd2(char *sum, char *term1, char *term2) { \
  ((type *) sum)[0] += ((type *) term1)[0] * ((type *) term2)[0] \
    - ((type *) term1)[1] * ((type *) term2)[1]; \
  ((type *) sum)[1] += ((type *) term1)[0] * ((type *) term2)[1] \
    + ((type *) term1)[1] * ((type *) term2)[0]; \
  return; }

// 定义复数类型的函数，接受多个参数
#define MAKE_C_ONEMULTADD2(fname, type) \
static void fname ## _onemultadd(char *sum, char *term1, int64_t str, \
                                 char **pvals, int64_t n) { \
        for (int64_t k=0; k < n; k++) { \
          fname ## _onemultadd2(sum, term1 + k * str, pvals[k]); \
        } \
}

// 生成复数类型的函数，分为单步和两步计算两种情况
MAKE_C_ONEMULTADD(CFLOAT, float)
MAKE_C_ONEMULTADD(CDOUBLE, double)
MAKE_C_ONEMULTADD(CLONGDOUBLE, npy_longdouble)
MAKE_C_ONEMULTADD2(CFLOAT, float)
MAKE_C_ONEMULTADD2(CDOUBLE, double)
MAKE_C_ONEMULTADD2(CLONGDOUBLE, npy_longdouble)
#endif /* __GNUC__ */
static OneMultAddFunction *OneMultAdd[]={NULL,
                     BYTE_onemultadd,    // Index 1: Pointer to BYTE_onemultadd function
                     UBYTE_onemultadd,   // Index 2: Pointer to UBYTE_onemultadd function
                     SHORT_onemultadd,   // Index 3: Pointer to SHORT_onemultadd function
                                         USHORT_onemultadd,  // Index 4: Pointer to USHORT_onemultadd function
                     INT_onemultadd,     // Index 5: Pointer to INT_onemultadd function
                                         UINT_onemultadd,   // Index 6: Pointer to UINT_onemultadd function
                     LONG_onemultadd,    // Index 7: Pointer to LONG_onemultadd function
                     ULONG_onemultadd,   // Index 8: Pointer to ULONG_onemultadd function
                     LONGLONG_onemultadd,// Index 9: Pointer to LONGLONG_onemultadd function
                     ULONGLONG_onemultadd,// Index 10: Pointer to ULONGLONG_onemultadd function
                     FLOAT_onemultadd,   // Index 11: Pointer to FLOAT_onemultadd function
                     DOUBLE_onemultadd,  // Index 12: Pointer to DOUBLE_onemultadd function
                     LONGDOUBLE_onemultadd,// Index 13: Pointer to LONGDOUBLE_onemultadd function
                     CFLOAT_onemultadd,  // Index 14: Pointer to CFLOAT_onemultadd function
                     CDOUBLE_onemultadd, // Index 15: Pointer to CDOUBLE_onemultadd function
                     CLONGDOUBLE_onemultadd,// Index 16: Pointer to CLONGDOUBLE_onemultadd function
                                         NULL, NULL, NULL, NULL};  // Indices 17 to 20: NULL pointers

//
// reflect_symm_index(j, m) maps an arbitrary integer j to the interval [0, m).
//
// * j can have any value.
// * m is assumed to be positive.
//
// The mapping from j to [0, m) is via reflection about the edges of the array.
// That is, the "base" array is [0, 1, 2, ..., m-1].  To continue to the right,
// the indices count down: [m-1, m-2, ... 0], and then count up again
// [0, 1, ..., m-1], and so on. The same extension pattern is followed on the
// left.
//
// Example, with m = 5:
//                           ----extension--------|-----base----|----extension----
//                        j: -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 10
// reflect_symm_index(j, 5):  3  4  4  3  2  1  0  0  1  2  3  4  4  3  2  1  0  0
//
static int64_t
reflect_symm_index(int64_t j, int64_t m)
{
    // First map j to k in the interval [0, 2*m-1).
    // Then flip the k values that are greater than or equal to m.
    int64_t k = (j >= 0) ? (j % (2*m)) : (llabs(j + 1) % (2*m));
    return (k >= m) ? (2*m - k - 1) : k;
}

//
// circular_wrap_index(j, m) maps an arbitrary integer j to the interval [0, m).
// The mapping makes the indices periodic.
//
// * j can have any value.
// * m is assumed to be positive.
//
// Example, with m = 5:
//                            ----extension--------|-----base----|----extension----
//                         j: -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 10
// circular_wrap_index(j, 5):  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0
//
static int64_t
circular_wrap_index(int64_t j, int64_t m)
{
    // About the negative case: in C, -3 % 5 is -3, so that explains
    // the " + m" after j % m.  But -5 % 5 is 0, and so -5 % 5 + 5 is 5,
    // which we want to wrap around to 0.  That's why the second " % m" is
    // included in the expression.
    return (j >= 0) ? (j % m) : ((j % m + m) % m);
}


/* This could definitely be more optimized... */
  /* Input data Ns[0] x Ns[1] */
  int pylab_convolve_2d (char  *in,
               /* Input strides */
               npy_intp   *instr,
               /* Output data */
               char  *out,
               /* Output strides */
               npy_intp   *outstr,
               /* coefficients in filter */
               char  *hvals,
               /* coefficients strides */
               npy_intp   *hstr,
               /* Size of kernel Nwin[0] x Nwin[1] */
               npy_intp   *Nwin,
               /* Size of image Ns[0] x Ns[1] */
               npy_intp   *Ns,
               /* convolution parameters */
               int   flag,
               /* fill value */
               char  *fillvalue) /* fill value */
{
  const int boundary = flag & BOUNDARY_MASK;  /* flag can be fill, reflecting, circular */
  const int outsize = flag & OUTSIZE_MASK;
  const int convolve = flag & FLIP_MASK;
  const int type_num = (flag & TYPE_MASK) >> TYPE_SHIFT;
  /*type_size*/

  OneMultAddFunction *mult_and_add = OneMultAdd[type_num];
  if (mult_and_add == NULL) return -5;  /* Not available for this type */

  if (type_num < 0 || type_num > MAXTYPES) return -4;  /* Invalid type */
  const int type_size = elsizes[type_num];

  int64_t Os[2];
  if (outsize == FULL) {Os[0] = Ns[0]+Nwin[0]-1; Os[1] = Ns[1]+Nwin[1]-1;}
  else if (outsize == SAME) {Os[0] = Ns[0]; Os[1] = Ns[1];}
  else if (outsize == VALID) {Os[0] = Ns[0]-Nwin[0]+1; Os[1] = Ns[1]-Nwin[1]+1;}
  else return -1; /* Invalid output flag */

  if ((boundary != PAD) && (boundary != REFLECT) && (boundary != CIRCULAR))
    return -2; /* Invalid boundary flag */

  char **indices = malloc(Nwin[1] * sizeof(indices[0]));
  if (indices == NULL) return -3; /* No memory */

  /* Speed this up by not doing any if statements in the for loop.  Need 3*3*2=18 different
     loops executed for different conditions */

  for (int64_t m=0; m < Os[0]; m++) {
    /* Reposition index into input image based on requested output size */
    int64_t new_m;
    if (outsize == FULL) new_m = convolve ? m : (m-Nwin[0]+1);
    else if (outsize == SAME) new_m = convolve ? (m+((Nwin[0]-1)>>1)) : (m-((Nwin[0]-1) >> 1));
    else new_m = convolve ? (m+Nwin[0]-1) : m; /* VALID */

    for (int64_t n=0; n < Os[1]; n++) {  /* loop over columns */
      char * sum = out+m*outstr[0]+n*outstr[1];
      memset(sum, 0, type_size); /* sum = 0.0; */

      int64_t new_n;
      if (outsize == FULL) new_n = convolve ? n : (n-Nwin[1]+1);
      else if (outsize == SAME) new_n = convolve ? (n+((Nwin[1]-1)>>1)) : (n-((Nwin[1]-1) >> 1));
      else new_n = convolve ? (n+Nwin[1]-1) : n;

      /* Sum over kernel, if index into image is out of bounds
     handle it according to boundary flag */
      for (int64_t j=0; j < Nwin[0]; j++) {
    int64_t ind0 = convolve ? (new_m-j): (new_m+j);
    bool bounds_pad_flag = false;

    if ((ind0 < 0) || (ind0 >= Ns[0])) {
      if (boundary == REFLECT) ind0 = reflect_symm_index(ind0, Ns[0]);
      else if (boundary == CIRCULAR) ind0 = circular_wrap_index(ind0, Ns[0]);
      else bounds_pad_flag = true;
    }


**注释：**


  /* Input data Ns[0] x Ns[1] */
  int pylab_convolve_2d (char  *in,
               /* Input strides */
               npy_intp   *instr,
               /* Output data */
               char  *out,
               /* Output strides */
               npy_intp   *outstr,
               /* coefficients in filter */
               char  *hvals,
               /* coefficients strides */
               npy_intp   *hstr,
               /* Size of kernel Nwin[0] x Nwin[1] */
               npy_intp   *Nwin,
               /* Size of image Ns[0] x Ns[1] */
               npy_intp   *Ns,
               /* convolution parameters */
               int   flag,
               /* fill value */
               char  *fillvalue) /* fill value */
{
  const int boundary = flag & BOUNDARY_MASK;  /* flag can be fill, reflecting, circular */
  const int outsize = flag & OUTSIZE_MASK;
  const int convolve = flag & FLIP_MASK;
  const int type_num = (flag & TYPE_MASK) >> TYPE_SHIFT;
  /*type_size*/

  OneMultAddFunction *mult_and_add = OneMultAdd[type_num];
  if (mult_and_add == NULL) return -5;  /* Not available for this type */

  if (type_num < 0 || type_num > MAXTYPES) return -4;  /* Invalid type */
  const int type_size = elsizes[type_num];

  int64_t Os[2];
  if (outsize == FULL) {Os[0] = Ns[0]+Nwin[0]-1; Os[1] = Ns[1]+Nwin[1]-1;}
  else if (outsize == SAME) {Os[0] = Ns[0]; Os[1] = Ns[1];}
  else if (outsize == VALID) {Os[0] = Ns[0]-Nwin[0]+1; Os[1] = Ns[1]-Nwin[1]+1;}
  else return -1; /* Invalid output flag */

  if ((boundary != PAD) && (boundary != REFLECT) && (boundary != CIRCULAR))
    return -2; /* Invalid boundary flag */

  char **indices = malloc(Nwin[1] * sizeof(indices[0]));
  if (indices == NULL) return -3; /* No memory */

  /* Speed this up by not doing any if statements in the for loop.  Need 3*3*2=18 different
     loops executed for different conditions */

  for (int64_t m=0; m < Os[0]; m++) {
    /* Reposition index into input image based on requested output size */
    int64_t new_m;
    if (outsize == FULL) new_m = convolve ? m : (m-Nwin[0]+1);
    else if (outsize == SAME) new_m = convolve ? (m+((Nwin[0]-1)>>1)) : (m-((Nwin[0]-1) >> 1));
    else new_m = convolve ? (m+Nwin[0]-1) : m; /* VALID */

    for (int64_t n=0; n < Os[1]; n++) {  /* loop over columns */
      char * sum = out+m*outstr[0]+n*outstr[1];
      memset(sum, 0, type_size); /* sum = 0.0; */

      int64_t new_n;
      if (outsize == FULL) new_n = convolve ? n : (n-Nwin[1]+1);
      else if (outsize == SAME) new_n = convolve ? (n+((Nwin[1]-1)>>1)) : (n-((Nwin[1]-1) >> 1));
      else new_n = convolve ? (n+Nwin[1]-1) : n;

      /* Sum over kernel, if index into image is out of bounds
     handle it according to boundary flag */
      for (int64_t j=0; j < Nwin[0]; j++) {
    int64_t ind0 = convolve ? (new_m-j): (new_m+j);
    bool bounds_pad_flag = false;

    if ((ind0 < 0) || (ind0 >= Ns[0])) {
      if (boundary == REFLECT) ind0 = reflect_symm_index(ind0, Ns[0]);
      else if (boundary == CIRCULAR) ind0 = circular_wrap_index(ind0, Ns[0]);
      else bounds_pad_flag = true;
    }
    // 计算ind0与instr[0]的乘积，存储在ind0_memory中
    const int64_t ind0_memory = ind0 * instr[0];

    // 如果bounds_pad_flag为真
    if (bounds_pad_flag) {
      // 将indices数组中的每个元素设置为fillvalue
      for (int64_t k = 0; k < Nwin[1]; k++) {
          indices[k] = fillvalue;
      }
    }
    // 如果bounds_pad_flag为假
    else  {
      // 遍历Nwin[1]个元素的循环
      for (int64_t k = 0; k < Nwin[1]; k++) {
        // 根据convolve的值选择不同的计算方式，将结果存储在ind1中
        int64_t ind1 = convolve ? (new_n - k) : (new_n + k);
        
        // 检查ind1是否超出边界Ns[1]
        if ((ind1 < 0) || (ind1 >= Ns[1])) {
          // 根据boundary的不同类型进行边界处理
          if (boundary == REFLECT) ind1 = reflect_symm_index(ind1, Ns[1]);
          else if (boundary == CIRCULAR) ind1 = circular_wrap_index(ind1, Ns[1]);
          else bounds_pad_flag = true;
        }

        // 如果bounds_pad_flag为真
        if (bounds_pad_flag) {
          // 将indices数组中的每个元素设置为fillvalue
          indices[k] = fillvalue;
        }
        // 如果bounds_pad_flag为假
        else {
          // 计算indices[k]的值，通过in、ind0_memory和ind1*instr[1]的计算得到
          indices[k] = in + ind0_memory + ind1 * instr[1];
        }
        // 将bounds_pad_flag重置为假
        bounds_pad_flag = false;
      }
    }
    // 调用mult_and_add函数，用sum、hvals+j*hstr[0]、hstr[1]、indices和Nwin[1]作为参数
    mult_and_add(sum, hvals + j * hstr[0], hstr[1], indices, Nwin[1]);
  }
  // 释放indices数组的内存
  free(indices);
  // 返回0作为函数执行的结果
  return 0;
}


注释：


# 这行代码表示一个函数的结束或者一个代码块的结束，通常与函数或者控制结构的开始部分对应。
# 在大多数编程语言中，闭合符号 } 用于标记代码块的结束。
# 在此处的上下文中，这可能是一个函数定义的结尾，或者是一个条件语句、循环或类定义的结尾。
# 它表示前面的代码段的逻辑范围的结束，是代码结构的一部分。
```