# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_interpolation.h`

```
/*
 * 版权所有 (C) 2003-2005 Peter J. Verveer
 *
 * 在源代码和二进制形式下重新分发及使用，无论是否修改，均可允许，但需符合以下条件：
 *
 * 1. 源代码的再分发必须保留上述版权声明、此条件列表及以下免责声明。
 *
 * 2. 在二进制形式下的再分发，必须在文档和/或其他提供的材料中重现上述版权声明、此条件列表及以下免责声明。
 *
 * 3. 未经特定书面许可，不得使用作者的名字来支持或推广从此软件衍生的产品。
 *
 * 此软件由作者"按现状"提供，任何明示或暗示的保证，包括但不限于适销性和特定用途的适用性保证，均不做出。无论在何种情况下，作者均不对任何直接的、间接的、偶发的、特殊的、惩罚性的或后果性的损害（包括但不限于替代物品或服务的采购；使用数据、利润或业务中断造成的损失）承担责任，无论是合同责任、严格责任或因其他方式产生的任何责任。
 */

#ifndef NI_INTERPOLATION_H
#define NI_INTERPOLATION_H

// 声明一维样条滤波函数，接收输入和输出数组以及一些参数
int NI_SplineFilter1D(PyArrayObject*, int, int, NI_ExtendMode, PyArrayObject*);

// 声明几何变换函数，接收输入数组、变换函数指针及其参数以及若干输出数组
int NI_GeometricTransform(PyArrayObject*, int (*)(npy_intp*, double*, int, int, void*),
                          void*, PyArrayObject*, PyArrayObject*, PyArrayObject*,
                          PyArrayObject*, int, int, double, int);

// 声明缩放平移函数，接收若干输入和输出数组以及一些参数
int NI_ZoomShift(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                 PyArrayObject*, int, int, double, int, int);

#endif
```