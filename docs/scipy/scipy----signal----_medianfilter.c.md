# `D:\src\scipysrc\scipy\scipy\signal\_medianfilter.c`

```
/*--------------------------------------------------------------------*/

#include "Python.h"
#define NO_IMPORT_ARRAY
#include "numpy/ndarrayobject.h"


/* defined below */
void f_medfilt2(float*,float*,npy_intp*,npy_intp*);
void d_medfilt2(double*,double*,npy_intp*,npy_intp*);
void b_medfilt2(unsigned char*,unsigned char*,npy_intp*,npy_intp*);
extern char *check_malloc (size_t);


/* The QUICK_SELECT routine is based on Hoare's Quickselect algorithm,
 * with unrolled recursion.
 * Author: Thouis R. Jones, 2008
 */

// 宏定义：交换数组中的两个元素
#define ELEM_SWAP(t, a, x, y) {register t temp = (a)[x]; (a)[x] = (a)[y]; (a)[y] = temp;}
// 宏定义：比较三个值，判断第一个是否是最小值
#define FIRST_LOWEST(x, y, z) (((x) < (y)) && ((x) < (z)))
// 宏定义：比较三个值，判断第一个是否是最大值
#define FIRST_HIGHEST(x, y, z) (((x) > (y)) && ((x) > (z)))
// 宏定义：返回两个值中较小的索引
#define LOWEST_IDX(a, x, y) (((a)[x] < (a)[y]) ? (x) : (y))
// 宏定义：返回两个值中较大的索引
#define HIGHEST_IDX(a, x, y) (((a)[x] > (a)[y]) ? (x) : (y))

// 宏定义：根据数组中的三个值，返回中间值的索引
#define MEDIAN_IDX(a, l, m, h) (FIRST_LOWEST((a)[l], (a)[m], (a)[h]) ? LOWEST_IDX(a, m, h) : (FIRST_HIGHEST((a)[l], (a)[m], (a)[h]) ? HIGHEST_IDX(a, m, h) : (l)))

// 宏定义：快速选择算法，用于从数组中找到第n小的元素
#define QUICK_SELECT(NAME, TYPE)                                        \
TYPE NAME(TYPE arr[], int n)                                            \
{                                                                       \
    int lo, hi, mid, md;                                                \
    int median_idx;                                                     \
    int ll, hh;                                                         \
    TYPE piv;                                                           \
                                                                        \
    lo = 0; hi = n-1;                                                   \
    // 计算中位数的索引，对于偶数长度的数组，取较小的中间值
    median_idx = (n - 1) / 2;                                           
    while (1) {                                                         \
        // 循环直到找到中位数或者返回结果
        if ((hi - lo) < 2) {                                            \
            // 如果剩余元素个数小于等于2，直接比较并返回中间值
            if (arr[hi] < arr[lo]) ELEM_SWAP(TYPE, arr, lo, hi);        \
            return arr[median_idx];                                     \
        }                                                               \
                                                                        \
        mid = (hi + lo) / 2;                                            \
        // 计算中间位置的索引
        /* put the median of lo,mid,hi at position lo - this will be the pivot */ \
        // 将 lo、mid、hi 三者的中位数放到 lo 位置，作为枢轴元素
        md = MEDIAN_IDX(arr, lo, mid, hi);                              \
        ELEM_SWAP(TYPE, arr, lo, md);                                   \
                                                                        \
        /* Nibble from each end towards middle, swapping misordered items */ \
        // 从两端向中间移动，交换顺序不正确的元素
        piv = arr[lo];                                                  \
        for (ll = lo+1, hh = hi;; ll++, hh--) {                         \
            // 从左到右找到第一个大于等于枢轴元素的位置 ll
            while (arr[ll] < piv) ll++;                    
            // 从右到左找到第一个小于等于枢轴元素的位置 hh
            while (arr[hh] > piv) hh--;                    
            // 如果 hh <= ll，说明左右两边已经分区完毕
            if (hh <= ll) break;                    
            // 交换 ll 和 hh 处的元素，使得左边的元素小于等于枢轴，右边的元素大于等于枢轴
            ELEM_SWAP(TYPE, arr, ll, hh);                
        }                                                               \
        /* move pivot to top of lower partition */                      \
        // 将枢轴元素放到左分区的顶部
        ELEM_SWAP(TYPE, arr, hh, lo);                                   
        /* set lo, hi for new range to search */                        \
        // 根据枢轴的位置调整 lo 和 hi 的范围
        if (hh < median_idx) /* search upper partition */               \
            lo = hh+1;                                                  \
        else if (hh > median_idx) /* search lower partition */          \
            hi = hh-1;                                                  \
        else                                                            \
            return piv;                                                 \
    }                                                                   \
/* 
   定义一个二维中值滤波器宏，对边缘进行零填充。

   参数说明：
   NAME: 定义的函数名
   TYPE: 输入和输出数据的类型
   SELECT: 选择函数（可能是一个条件或方法）

   函数实现：
   NAME(TYPE* in, TYPE* out, npy_intp* Nwin, npy_intp* Ns)
   - in: 输入数据指针
   - out: 输出数据指针
   - Nwin: 包含滤波窗口大小的数组
   - Ns: 暂时未使用的参数

   函数内部变量说明：
   - nx, ny: 输入数据的尺寸
   - hN[2]: 每个维度的一半窗口大小
   - pre_x, pre_y, pos_x, pos_y: 边缘填充时的坐标计算
   - subx, suby, k, totN: 辅助计算变量
   - myvals, fptr1, fptr2, ptr1, ptr2: 用于存储和操作数据的指针
*/
#define MEDIAN_FILTER_2D(NAME, TYPE, SELECT)                            \
void NAME(TYPE* in, TYPE* out, npy_intp* Nwin, npy_intp* Ns)            \
{                                                                       \
    int nx, ny, hN[2];                                                  \
    int pre_x, pre_y, pos_x, pos_y;                                     \
    int subx, suby, k, totN;                                            \
    TYPE *myvals, *fptr1, *fptr2, *ptr1, *ptr2;                         \
                                                                        \
    totN = Nwin[0] * Nwin[1];                                           \
    myvals = (TYPE *) check_malloc( totN * sizeof(TYPE));               \
                                                                        \
    Py_BEGIN_ALLOW_THREADS                                              \
                                                                        \
    hN[0] = Nwin[0] >> 1;                                               \
    hN[1] = Nwin[1] >> 1;                                               \
    ptr1 = in;                                                          \
    fptr1 = out;                                                        \
    for (ny = 0; ny < Ns[0]; ny++)                                      \
        // 循环遍历输入数组的第一维度
        for (nx = 0; nx < Ns[1]; nx++) {                                \
            // 循环遍历输入数组的第二维度
            pre_x = hN[1];                                              \
            // 初始化前向 x 方向的边界值
            pre_y = hN[0];                                              \
            // 初始化前向 y 方向的边界值
            pos_x = hN[1];                                              \
            // 初始化后向 x 方向的边界值
            pos_y = hN[0];                                              \
            // 初始化后向 y 方向的边界值
            if (nx < hN[1]) pre_x = nx;                                 \
            // 更新前向 x 方向的边界值
            if (nx >= Ns[1] - hN[1]) pos_x = Ns[1] - nx - 1;            \
            // 更新后向 x 方向的边界值
            if (ny < hN[0]) pre_y = ny;                                 \
            // 更新前向 y 方向的边界值
            if (ny >= Ns[0] - hN[0]) pos_y = Ns[0] - ny - 1;            \
            // 更新后向 y 方向的边界值
            fptr2 = myvals;                                             \
            // 指向输出数组的指针
            ptr2 = ptr1 - pre_x - pre_y*Ns[1];                          \
            // 指向输入数组的指针，根据前向边界计算偏移
            for (suby = -pre_y; suby <= pos_y; suby++) {                \
                // 循环遍历 y 方向的窗口
                for (subx = -pre_x; subx <= pos_x; subx++)              \
                    // 循环遍历 x 方向的窗口，复制数据到输出数组
                    *fptr2++ = *ptr2++;                                 \
                // 跳过窗口末尾的填充数据
                ptr2 += Ns[1] - (pre_x + pos_x + 1);                    \
            }                                                           \
            // 移动到下一个输入数组元素
            ptr1++;                                                     \
                                                                        \
            /* Zero pad */                                              \
            // 对输出数组剩余部分进行零填充
            for (k = (pre_x + pos_x + 1)*(pre_y + pos_y + 1); k < totN; k++) \
                *fptr2++ = 0.0;                                         \
                                                                        \
            /*      *fptr1++ = median(myvals,totN); */                  \
            // 计算并存储输出数组的中位数值
            *fptr1++ = SELECT(myvals,totN);                             \
        }                                                               \
                                                                        \
    Py_END_ALLOW_THREADS                                                \
                                                                        \
    free(myvals);                                                       \
/* 定义用于浮点数、双精度数和无符号字符的快速选择函数 */
QUICK_SELECT(f_quick_select, float)
/* 定义用于双精度数的快速选择函数 */
QUICK_SELECT(d_quick_select, double)
/* 定义用于无符号字符的快速选择函数 */
QUICK_SELECT(b_quick_select, unsigned char)

/* 定义用于浮点数、双精度数和无符号字符的二维中值滤波函数 */
MEDIAN_FILTER_2D(f_medfilt2, float, f_quick_select)
/* 定义用于双精度数的二维中值滤波函数 */
MEDIAN_FILTER_2D(d_medfilt2, double, d_quick_select)
/* 定义用于无符号字符的二维中值滤波函数 */
MEDIAN_FILTER_2D(b_medfilt2, unsigned char, b_quick_select)
```