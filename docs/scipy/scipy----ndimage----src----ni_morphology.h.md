# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_morphology.h`

```
/*
 * 版权声明和许可条款
 *
 * 1. 源代码的再分发和使用需保留上述版权声明、本条件列表和以下免责声明。
 *
 * 2. 二进制形式的再分发需在文件中包含上述版权声明、本条件列表和以下免责声明。
 *
 * 3. 未经特定事前书面许可，不得使用作者的名字来支持或推广由此软件派生的产品。
 *
 * 此软件由作者"按原样"提供，不提供任何明示或暗示的担保，包括但不限于适销性和特定用途的适用性。
 * 作者不对任何直接、间接、偶发、特殊、惩罚性或后果性损害负责，即使在使用此软件的可能性下亦然。
 */

#ifndef NI_MORPHOLOGY_H
#define NI_MORPHOLOGY_H

// 函数声明：二值图像的腐蚀操作
int NI_BinaryErosion(PyArrayObject*, PyArrayObject*, PyArrayObject*,
         PyArrayObject*, int, npy_intp*, int, int, int*, NI_CoordinateList**);

// 函数声明：二值图像的腐蚀操作（简化参数列表的版本）
int NI_BinaryErosion2(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                      int, npy_intp*, int, NI_CoordinateList**);

// 函数声明：基于暴力法的距离变换
int NI_DistanceTransformBruteForce(PyArrayObject*, int, PyArrayObject*,
                                   PyArrayObject*, PyArrayObject*);

// 函数声明：一次遍历的距离变换
int NI_DistanceTransformOnePass(PyArrayObject*, PyArrayObject*,
                                PyArrayObject*);

// 函数声明：欧几里得特征变换
int NI_EuclideanFeatureTransform(PyArrayObject*, PyArrayObject*,
                                 PyArrayObject*);

#endif
```