# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_fourier.h`

```
/*
 * 版权所有 (C) 2003-2005 Peter J. Verveer
 *
 * 在源代码和二进制形式中重新分发和使用，无论是否经过修改，都是允许的，
 * 前提是满足以下条件：
 *
 * 1. 源代码的再分发必须保留上述版权声明、本条件列表和以下免责声明。
 *
 * 2. 以二进制形式再分发时，必须在文档和/或其他提供的材料中复制上述版权声明、
 *    本条件列表和以下免责声明。
 *
 * 3. 未经特定书面许可，不得使用作者的名字来认可或推广从本软件衍生的产品。
 *
 * 本软件由作者"按原样"提供，不提供任何明示或暗示的保证，
 * 包括但不限于适销性和特定用途的暗示保证。
 * 无论在任何情况下，作者都不对任何直接、间接、附带、特殊、惩罚性或
 * 后果性损害（包括但不限于替代商品或服务的采购、使用数据或利润的损失，
 * 或业务中断）承担责任，无论是合同责任、严格责任或侵权（包括疏忽或其他方式）的理论，
 * 即使事先已告知发生此类损害的可能性。
 */

#ifndef NI_FOURIER_H
#define NI_FOURIER_H

// 定义函数NI_FourierFilter，接受多个参数并返回整数
int NI_FourierFilter(PyArrayObject*, PyArrayObject*, npy_intp, int,
                                         PyArrayObject*, int);

// 定义函数NI_FourierShift，接受多个参数并返回整数
int NI_FourierShift(PyArrayObject*, PyArrayObject*, npy_intp, int,
                                        PyArrayObject*);

#endif
```