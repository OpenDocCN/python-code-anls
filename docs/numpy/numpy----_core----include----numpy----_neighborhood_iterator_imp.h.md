# `.\numpy\numpy\_core\include\numpy\_neighborhood_iterator_imp.h`

```
#ifndef NUMPY_CORE_INCLUDE_NUMPY__NEIGHBORHOOD_IMP_H_
#error You should not include this header directly
#endif
/*
 * Private API (here for inline)
 */
// 定义一个静态内联函数，用于增加邻域迭代器的坐标
static inline int
_PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter);

/*
 * Update to next item of the iterator
 *
 * Note: this simply increment the coordinates vector, last dimension
 * incremented first , i.e, for dimension 3
 * ...
 * -1, -1, -1
 * -1, -1,  0
 * -1, -1,  1
 *  ....
 * -1,  0, -1
 * -1,  0,  0
 *  ....
 * 0,  -1, -1
 * 0,  -1,  0
 *  ....
 */
// 定义宏用于更新迭代器的坐标，实现按照特定顺序递增坐标
#define _UPDATE_COORD_ITER(c) \
    wb = iter->coordinates[c] < iter->bounds[c][1]; \
    if (wb) { \
        iter->coordinates[c] += 1; \
        return 0; \
    } \
    else { \
        iter->coordinates[c] = iter->bounds[c][0]; \
    }

// 针对二维数组进行优化的版本，手动展开循环
static inline int
_PyArrayNeighborhoodIter_IncrCoord2D(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp wb;

    _UPDATE_COORD_ITER(1)  // 优化后的增加坐标函数调用，针对第二维
    _UPDATE_COORD_ITER(0)  // 优化后的增加坐标函数调用，针对第一维

    return 0;
}
#undef _UPDATE_COORD_ITER

/*
 * Advance to the next neighbour
 */
// 前进到下一个邻域
static inline int
PyArrayNeighborhoodIter_Next(PyArrayNeighborhoodIterObject* iter)
{
    _PyArrayNeighborhoodIter_IncrCoord (iter);  // 调用增加坐标的函数
    iter->dataptr = iter->translate((PyArrayIterObject*)iter, iter->coordinates);  // 更新数据指针

    return 0;
}

/*
 * Reset functions
 */
// 重置函数
static inline int
PyArrayNeighborhoodIter_Reset(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp i;

    for (i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];  // 将所有维度的坐标重置为最小值
    }
    iter->dataptr = iter->translate((PyArrayIterObject*)iter, iter->coordinates);  // 更新数据指针

    return 0;
}
```