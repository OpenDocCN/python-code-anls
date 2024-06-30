# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\libsvm\svm.cpp`

```
/*
   Copyright (c) 2000-2009 Chih-Chung Chang and Chih-Jen Lin
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

   3. Neither name of copyright holders nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
   Modified 2010:

   - Support for dense data by Ming-Fang Weng

   - Return indices for support vectors, Fabian Pedregosa
     <fabian.pedregosa@inria.fr>

   - Fixes to avoid name collision, Fabian Pedregosa

   - Add support for instance weights, Fabian Pedregosa based on work
     by Ming-Wei Chang, Hsuan-Tien Lin, Ming-Hen Tsai, Chia-Hua Ho and
     Hsiang-Fu Yu,
     <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances>.

   - Make labels sorted in svm_group_classes, Fabian Pedregosa.

   Modified 2020:

   - Improved random number generator by using a mersenne twister + tweaked
     lemire postprocessor. This fixed a convergence issue on windows targets.
     Sylvain Marie, Schneider Electric
     see <https://github.com/scikit-learn/scikit-learn/pull/13511#issuecomment-481729756>

   Modified 2021:

   - Exposed number of iterations run in optimization, Juan Martín Loyola.
     See <https://github.com/scikit-learn/scikit-learn/pull/21408/>
 */

#include <math.h>                // 导入数学函数库，例如 sqrt() 和 sin()
#include <stdio.h>               // 导入标准输入输出函数库，例如 printf() 和 scanf()
#include <stdlib.h>              // 导入标准库函数库，例如 malloc() 和 free()
#include <ctype.h>               // 导入字符处理函数库，例如 isdigit() 和 toupper()
#include <float.h>               // 导入浮点数处理函数库，例如 FLT_MAX 和 DBL_MIN
#include <string.h>              // 导入字符串处理函数库，例如 strcpy() 和 strcmp()
#include <stdarg.h>              // 导入可变参数列表处理函数库，例如 va_start() 和 va_end()
#include <climits>               // 导入 C++ 标准库中的整数限制宏定义
#include <random>                // 导入随机数生成器类，用于生成随机数
#include "svm.h"                 // 导入自定义的 SVM 相关头文件
#include "_svm_cython_blas_helpers.h"  // 导入帮助实现 BLAS 的辅助函数头文件
#include "../newrand/newrand.h"  // 导入新随机数生成器头文件

#ifndef _LIBSVM_CPP
typedef float Qfloat;            // 定义 Qfloat 类型为 float
typedef signed char schar;       // 定义 schar 类型为 signed char

#ifndef min
// 定义模板函数，返回两个值中的较小值
template <class T> static inline T min(T x, T y) { return (x < y) ? x : y; }
#endif

#ifndef max
// 定义模板函数，返回两个值中的较大值
template <class T> static inline T max(T x, T y) { return (x > y) ? x : y; }
#endif
//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
    // Constructor for Cache class
    Cache(int l, long int size);
    
    // Destructor for Cache class
    ~Cache();

    // Request data [0,len)
    // Return some position p where [p,len) need to be filled
    // (p >= len if nothing needs to be filled)
    int get_data(const int index, Qfloat **data, int len);

    // Swap index i and j in the cache
    void swap_index(int i, int j);

private:
    int l;                  // Number of total data items
    long int size;          // Cache size limit in bytes

    struct head_t
    {
        head_t *prev, *next;    // Circular list pointers
        Qfloat *data;           // Cached data
        int len;                // Length of cached data
    };

    head_t *head;           // Array of cache entries
    head_t lru_head;        // Head of the LRU list

    // Delete entry h from the LRU list
    void lru_delete(head_t *h);

    // Insert entry h into the LRU list
    void lru_insert(head_t *h);
};

// Constructor initializes Cache with given parameters
Cache::Cache(int l_, long int size_) : l(l_), size(size_)
{
    head = (head_t *)calloc(l, sizeof(head_t));    // Allocate and initialize head array
    size /= sizeof(Qfloat);                        // Convert size to number of Qfloat elements
    size -= l * sizeof(head_t) / sizeof(Qfloat);   // Adjust size for head structure
    size = max(size, 2 * (long int) l);            // Ensure cache is large enough for two columns
    lru_head.next = lru_head.prev = &lru_head;     // Initialize LRU list
}

// Destructor frees allocated memory for Cache
Cache::~Cache()
{
    for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
        free(h->data);      // Free cached data
    free(head);             // Free head array
}

// Delete entry h from the LRU list
void Cache::lru_delete(head_t *h)
{
    h->prev->next = h->next;    // Update pointers to remove h from the list
    h->next->prev = h->prev;
}

// Insert entry h into the LRU list
void Cache::lru_insert(head_t *h)
{
    h->next = &lru_head;        // Insert h at the end of the list
    h->prev = lru_head.prev;
    h->prev->next = h;
    h->next->prev = h;
}

// Request data [0,len) from the cache for a given index
// Return the position p where [p,len) need to be filled
// (p >= len if nothing needs to be filled)
int Cache::get_data(const int index, Qfloat **data, int len)
{
    # 获取指向头部数组中指定索引的指针
    head_t *h = &head[index];
    # 如果头部元素长度不为零，则从LRU缓存中删除该元素
    if(h->len) lru_delete(h);
    # 计算需要额外分配或释放的空间量
    int more = len - h->len;

    if(more > 0)
    {
        # 释放旧空间
        while(size < more)
        {
            # 获取LRU链表中的第一个元素
            head_t *old = lru_head.next;
            # 从LRU缓存中删除该元素
            lru_delete(old);
            # 释放该元素的数据空间
            free(old->data);
            # 更新已释放空间的大小
            size += old->len;
            # 将元素的数据指针和长度重置为零
            old->data = 0;
            old->len = 0;
        }

        # 分配新空间
        h->data = (Qfloat *)realloc(h->data, sizeof(Qfloat) * len);
        # 减少已分配但未使用空间的大小
        size -= more;
        # 交换头部元素的长度和新长度
        swap(h->len, len);
    }

    # 将头部元素插入LRU缓存
    lru_insert(h);
    # 将数据指针指向头部元素的数据
    *data = h->data;
    # 返回新长度
    return len;
}

// 交换 Cache 中索引为 i 和 j 的数据项
void Cache::swap_index(int i, int j)
{
    // 如果 i 和 j 相等，直接返回
    if(i==j) return;

    // 如果 head[i] 中有数据，从 LRU 缓存中删除
    if(head[i].len) lru_delete(&head[i]);
    // 如果 head[j] 中有数据，从 LRU 缓存中删除
    if(head[j].len) lru_delete(&head[j]);

    // 交换 head[i] 和 head[j] 中的数据和长度
    swap(head[i].data, head[j].data);
    swap(head[i].len, head[j].len);

    // 如果交换后 head[i] 中有数据，重新插入到 LRU 缓存中
    if(head[i].len) lru_insert(&head[i]);
    // 如果交换后 head[j] 中有数据，重新插入到 LRU 缓存中
    if(head[j].len) lru_insert(&head[j]);

    // 确保 i < j
    if(i > j) swap(i, j);

    // 遍历 LRU 链表，处理索引大于 i 的节点
    for(head_t *h = lru_head.next; h != &lru_head; h = h->next)
    {
        // 如果节点 h 的长度大于 i
        if(h->len > i)
        {
            // 如果节点 h 的长度大于 j，交换数据
            if(h->len > j)
                swap(h->data[i], h->data[j]);
            else
            {
                // 否则，放弃该节点的数据
                lru_delete(h);
                free(h->data);
                size += h->len;
                h->data = 0;
                h->len = 0;
            }
        }
    }
}

//
// Kernel evaluation
//

// k_function 方法用于计算单个核函数的值
// Kernel 的构造函数用于准备计算 l*l 的核矩阵
// get_Q 方法用于从 Q 矩阵中获取一列数据
//
class QMatrix {
public:
    virtual Qfloat *get_Q(int column, int len) const = 0;
    virtual double *get_QD() const = 0;
    virtual void swap_index(int i, int j) const = 0;
    virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
    // 构造函数，根据参数构造 Kernel 对象
#ifdef _DENSE_REP
    Kernel(int l, PREFIX(node) * x, const svm_parameter& param, BlasFunctions *blas_functions);
#else
    Kernel(int l, PREFIX(node) * const * x, const svm_parameter& param, BlasFunctions *blas_functions);
#endif
    virtual ~Kernel();

    // 静态方法 k_function，用于计算核函数值
    static double k_function(const PREFIX(node) *x, const PREFIX(node) *y,
                 const svm_parameter& param, BlasFunctions *blas_functions);
    
    // 从 Q 矩阵中获取一列数据的虚函数
    virtual Qfloat *get_Q(int column, int len) const = 0;
    // 获取 QD 数组的虚函数
    virtual double *get_QD() const = 0;
    
    // 交换索引 i 和 j 对应的数据项，不是 const
    virtual void swap_index(int i, int j) const
    {
        swap(x[i], x[j]);
        if(x_square) swap(x_square[i], x_square[j]);
    }
protected:

    // 核函数指针成员
    double (Kernel::*kernel_function)(int i, int j) const;

private:
#ifdef _DENSE_REP
    PREFIX(node) *x;  // 节点数据数组指针
#else
    const PREFIX(node) **x;  // 节点数据数组指针的指针
#endif
    double *x_square;  // 节点数据平方和数组指针
    BlasFunctions *m_blas;  // BLAS 函数指针

    // SVM 参数
    const int kernel_type;  // 核函数类型
    const int degree;  // 多项式核函数的阶数
    const double gamma;  // 核函数参数 gamma
    const double coef0;  // 核函数参数 coef0

    // 计算两个节点数据的内积
    static double dot(const PREFIX(node) *px, const PREFIX(node) *py, BlasFunctions *blas_functions);
#ifdef _DENSE_REP
    static double dot(const PREFIX(node) &px, const PREFIX(node) &py, BlasFunctions *blas_functions);
#endif

    // 线性核函数
    double kernel_linear(int i, int j) const
    {
        return dot(x[i], x[j], m_blas);
    }
    // 多项式核函数
    double kernel_poly(int i, int j) const
    {
        return powi(gamma * dot(x[i], x[j], m_blas) + coef0, degree);
    }
    // RBF 核函数
    double kernel_rbf(int i, int j) const
    {
        return exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j], m_blas)));
    }
    // Sigmoid 核函数
    double kernel_sigmoid(int i, int j) const
    {
        return tanh(gamma * dot(x[i], x[j], m_blas) + coef0);
    }
    // 预计算核函数
    double kernel_precomputed(int i, int j) const
    {
#ifdef _DENSE_REP
// 如果定义了 _DENSE_REP 宏，则使用稠密表示方式计算内积
double Kernel::dot(const PREFIX(node) *px, const PREFIX(node) *py, BlasFunctions *blas_functions)
{
    // 初始化内积的总和为 0
    double sum = 0;

    // 取节点 px 和 py 中维度较小的一个作为内积的维度
    int dim = min(px->dim, py->dim);

    // 调用 BLAS 库计算稠密向量的内积
    sum = blas_functions->dot(dim, px->values, 1, py->values, 1);

    // 返回计算得到的内积结果
    return sum;
}

// 如果定义了 _DENSE_REP 宏，则使用稠密表示方式计算单个节点的内积
double Kernel::dot(const PREFIX(node) &px, const PREFIX(node) &py, BlasFunctions *blas_functions)
{
    // 初始化内积的总和为 0
    double sum = 0;

    // 取节点 px 和 py 中维度较小的一个作为内积的维度
    int dim = min(px.dim, py.dim);

    // 调用 BLAS 库计算稠密向量的内积
    sum = blas_functions->dot(dim, px.values, 1, py.values, 1);

    // 返回计算得到的内积结果
    return sum;
}
#else
// 如果未定义 _DENSE_REP 宏，则使用稀疏表示方式计算内积
double Kernel::dot(const PREFIX(node) *px, const PREFIX(node) *py, BlasFunctions *blas_functions)
{
    // 初始化内积的总和为 0
    double sum = 0;

    // 当 px 和 py 的索引均不为 -1 时，执行循环
    while(px->index != -1 && py->index != -1)
    {
        // 若 px 和 py 的索引相等，则将其值乘积加到内积总和中
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;  // 移动 px 指针至下一个节点
            ++py;  // 移动 py 指针至下一个节点
        }
        else
        {
            // 若 px 的索引大于 py 的索引，则移动 py 指针至下一个节点
            if(px->index > py->index)
                ++py;
            else  // 否则移动 px 指针至下一个节点
                ++px;
        }
    }

    // 返回计算得到的稀疏表示方式的内积结果
    return sum;
}
#endif

// Kernel 类的构造函数，根据传入的参数初始化对象
#ifdef _DENSE_REP
Kernel::Kernel(int l, PREFIX(node) * x_, const svm_parameter& param, BlasFunctions *blas_functions)
#else
Kernel::Kernel(int l, PREFIX(node) * const * x_, const svm_parameter& param, BlasFunctions *blas_functions)
#endif
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
    m_blas = blas_functions;  // 将 BLAS 函数对象赋值给成员变量 m_blas

    // 根据内核类型选择相应的内核函数
    switch(kernel_type)
    {
        case LINEAR:
            kernel_function = &Kernel::kernel_linear;  // 线性内核函数
            break;
        case POLY:
            kernel_function = &Kernel::kernel_poly;    // 多项式内核函数
            break;
        case RBF:
            kernel_function = &Kernel::kernel_rbf;     // 径向基函数（RBF）内核函数
            break;
        case SIGMOID:
            kernel_function = &Kernel::kernel_sigmoid; // sigmoid 内核函数
            break;
        case PRECOMPUTED:
            kernel_function = &Kernel::kernel_precomputed;  // 预先计算的内核函数
            break;
    }

    clone(x,x_,l);  // 克隆节点数据 x_

    // 若内核类型为 RBF，则计算 x_square 数组
    if(kernel_type == RBF)
    {
        x_square = new double[l];  // 分配存储空间

        // 计算每个节点的平方和并存储在 x_square 数组中
        for(int i=0;i<l;i++)
            x_square[i] = dot(x[i],x[i],blas_functions);
    }
    else
        x_square = 0;  // 否则将 x_square 设置为 nullptr
}

// Kernel 类的析构函数，释放动态分配的内存
Kernel::~Kernel()
{
    delete[] x;         // 释放节点数据数组 x
    delete[] x_square;  // 释放存储节点平方和的数组 x_square
}

// Kernel 类中稠密表示方式的节点内积计算函数
#ifdef _DENSE_REP
double Kernel::dot(const PREFIX(node) *px, const PREFIX(node) *py, BlasFunctions *blas_functions)
{
    // 初始化内积的总和为 0
    double sum = 0;

    // 取节点 px 和 py 中维度较小的一个作为内积的维度
    int dim = min(px->dim, py->dim);

    // 调用 BLAS 库计算稠密向量的内积
    sum = blas_functions->dot(dim, px->values, 1, py->values, 1);

    // 返回计算得到的内积结果
    return sum;
}

// Kernel 类中稠密表示方式的单个节点内积计算函数
double Kernel::dot(const PREFIX(node) &px, const PREFIX(node) &py, BlasFunctions *blas_functions)
{
    // 初始化内积的总和为 0
    double sum = 0;

    // 取节点 px 和 py 中维度较小的一个作为内积的维度
    int dim = min(px.dim, py.dim);

    // 调用 BLAS 库计算稠密向量的内积
    sum = blas_functions->dot(dim, px.values, 1, py.values, 1);

    // 返回计算得到的内积结果
    return sum;
}
#else
// Kernel 类中稀疏表示方式的节点内积计算函数
double Kernel::dot(const PREFIX(node) *px, const PREFIX(node) *py, BlasFunctions *blas_functions)
{
    // 初始化内积的总和为 0
    double sum = 0;

    // 当 px 和 py 的索引均不为 -1 时，执行循环
    while(px->index != -1 && py->index != -1)
    {
        // 若 px 和 py 的索引相等，则将其值乘积加到内积总和中
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;  // 移动 px 指针至下一个节点
            ++py;  // 移动 py 指针至下一个节点
        }
        else
        {
            // 若 px 的索引大于 py 的索引，则移动 py 指针至下一个节点
            if(px->index > py->index)
                ++py;
            else  // 否则移动 px 指针至下一个节点
                ++px;
        }
    }

    // 返回计算得到的稀疏表示方式的内积结果
    return sum;
}
#endif

// Kernel 类中的核函数 k_function 实现
double Kernel::k_function(const PREFIX(node) *x, const PREFIX(node) *y,
              const svm_parameter& param, BlasFunctions *blas_functions)
{
    // 根据参数中的内核类型选择相应的核函数计算方式
    switch(param.kernel_type)
    {
        case LINEAR:
            return dot(x,y,blas_functions);  // 线性核函数，直接计算内积
        case POLY:
            return powi(param.gamma*dot(x,y,blas_functions)+param.coef0,param.degree);  // 多项式核函数
        case RBF:
        {
            double sum = 0;
#ifdef _DENSE_REP
            // 如果定义了 _DENSE_REP 宏，则执行以下代码块
            int dim = min(x->dim, y->dim), i;
            // 计算维度的最小值，并初始化变量 i
            double* m_array = (double*)malloc(sizeof(double)*dim);
            // 分配内存以存储大小为 dim 的 double 类型数组
            for (i = 0; i < dim; i++)
            {
                // 遍历数组，计算 x 和 y 对应索引处值的差，并存入 m_array
                m_array[i] = x->values[i] - y->values[i];
            }
            // 计算 m_array 的点积，结果存入 sum
            sum = blas_functions->dot(dim, m_array, 1, m_array, 1);
            // 释放 m_array 分配的内存
            free(m_array);
            // 继续计算 x 余下部分的平方和，加入 sum
            for (; i < x->dim; i++)
                sum += x->values[i] * x->values[i];
            // 继续计算 y 余下部分的平方和，加入 sum
            for (; i < y->dim; i++)
                sum += y->values[i] * y->values[i];
#else
            // 如果未定义 _DENSE_REP 宏，则执行以下代码块
            while(x->index != -1 && y->index != -1)
            {
                // 当 x 和 y 的索引均不为 -1 时执行循环
                if(x->index == y->index)
                {
                    // 如果 x 和 y 的索引相同，则计算它们值的差的平方，并加入 sum
                    double d = x->value - y->value;
                    sum += d*d;
                    // 指向下一个元素
                    ++x;
                    ++y;
                }
                else
                {
                    // 如果 x 和 y 的索引不同，则根据索引大小比较，分别计算并加入 sum
                    if(x->index > y->index)
                    {
                        sum += y->value * y->value;
                        // 指向 y 的下一个元素
                        ++y;
                    }
                    else
                    {
                        sum += x->value * x->value;
                        // 指向 x 的下一个元素
                        ++x;
                    }
                }
            }

            // 处理 x 中剩余的元素，计算它们的平方和并加入 sum
            while(x->index != -1)
            {
                sum += x->value * x->value;
                // 指向下一个元素
                ++x;
            }

            // 处理 y 中剩余的元素，计算它们的平方和并加入 sum
            while(y->index != -1)
            {
                sum += y->value * y->value;
                // 指向下一个元素
                ++y;
            }
#endif
            // 返回经过指数函数处理的结果，参数为 sum 的相反数乘以 param.gamma
            return exp(-param.gamma*sum);
        }
        // 对于 case 为 SIGMOID 的情况，执行以下代码块
        case SIGMOID:
            // 返回经过双曲正切函数处理的结果，参数为 dot(x,y,blas_functions) 乘以 param.gamma 和 param.coef0 的和
            return tanh(param.gamma*dot(x,y,blas_functions)+param.coef0);
        // 对于 case 为 PRECOMPUTED 的情况，执行以下代码块
        case PRECOMPUTED:
                    {
#ifdef _DENSE_REP
            // 如果定义了 _DENSE_REP 宏，则返回 x 中索引为 y->ind 处的值
            return x->values[y->ind];
#else
            // 如果未定义 _DENSE_REP 宏，则返回 x 数组中索引为 (int)(y->value) 处的值
            return x[(int)(y->value)].value;
#endif
                    }
        // 默认情况下，返回 0，表示不可达
        default:
            return 0;  // Unreachable
    }
}
// SMO 算法参考 Fan et al., JMLR 6(2005), p. 1889--1918
// 解决问题:
//
//    min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//        y^T \alpha = \delta
//        y_i = +1 或 -1
//        0 <= alpha_i <= Cp，如果 y_i = 1
//        0 <= alpha_i <= Cn，如果 y_i = -1
//
// 给定:
//
//    Q, p, y, Cp, Cn 和初始可行点 \alpha
//    l 是向量和矩阵的大小
//    eps 是停止容差
//
// 解将存入 \alpha，目标值将存入 obj
//

class Solver {
public:
    // 默认构造函数
    Solver() {};
    // 虚析构函数
    virtual ~Solver() {};

    // 解的信息结构体
    struct SolutionInfo {
        double obj;         // 目标函数值
        double rho;         // 
                double *upper_bound;  // 上界
        double r;            // 对于 Solver_NU
                bool solve_timed_out;  // 解决超时
        int n_iter;         // 迭代次数
    };

    // 解算法函数
    void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
           double *alpha_, const double *C_, double eps,
           SolutionInfo* si, int shrinking, int max_iter);
protected:
    int active_size;  // 活跃大小
    schar *y;         // y 向量
    double *G;        // 目标函数的梯度
    enum { LOWER_BOUND, UPPER_BOUND, FREE };  // 枚举类型：下界、上界、自由
    // 定义指向字符串的指针，用于表示每个变量的状态（下界、上界或自由）
    char *alpha_status;    // LOWER_BOUND, UPPER_BOUND, FREE
    
    // 指向双精度浮点数的指针，存储每个变量的 alpha 值
    double *alpha;
    
    // 指向常量 QMatrix 对象的指针，用于存储 Q 矩阵
    const QMatrix *Q;
    
    // 指向双精度浮点数的指针，存储每个变量对应的 QD 值
    const double *QD;
    
    // 双精度浮点数，表示训练中使用的精度值
    double eps;
    
    // 双精度浮点数，表示正类和负类的惩罚参数
    double Cp,Cn;
    
    // 指向双精度浮点数的指针，存储每个变量的惩罚参数
    double *C;
    
    // 指向双精度浮点数的指针，存储每个变量的梯度 p
    double *p;
    
    // 指向整型数组的指针，存储当前活跃集的索引
    int *active_set;
    
    // 指向双精度浮点数的指针，存储梯度 G_bar，当自由变量视为 0 时
    double *G_bar;        // gradient, if we treat free variables as 0
    
    // 整型变量，表示训练数据的样本数量
    int l;
    
    // 布尔变量，表示是否需要进行展开操作（未指明具体用途）
    bool unshrink;    // XXX

    // 获取第 i 个变量的惩罚参数 C
    double get_C(int i)
    {
        return C[i];
    }
    
    // 更新第 i 个变量的状态（下界、上界或自由），根据当前的 alpha 值和 C 值
    void update_alpha_status(int i)
    {
        if(alpha[i] >= get_C(i))
            alpha_status[i] = UPPER_BOUND;
        else if(alpha[i] <= 0)
            alpha_status[i] = LOWER_BOUND;
        else
            alpha_status[i] = FREE;
    }
    
    // 判断第 i 个变量是否为上界
    bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
    
    // 判断第 i 个变量是否为下界
    bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
    
    // 判断第 i 个变量是否自由
    bool is_free(int i) { return alpha_status[i] == FREE; }
    
    // 交换索引 i 和 j 对应的变量
    void swap_index(int i, int j);
    
    // 重构梯度
    void reconstruct_gradient();
    
    // 选择工作集中的变量对 i 和 j 进行设置
    virtual int select_working_set(int &i, int &j);
    
    // 计算 rho 值
    virtual double calculate_rho();
    
    // 执行收缩操作
    virtual void do_shrinking();
private:
    bool be_shrunk(int i, double Gmax1, double Gmax2);
};

// 实现 swap_index 方法，交换索引 i 和 j 处的变量及其相关数据
void Solver::swap_index(int i, int j)
{
    // 交换 Q 对象中索引 i 和 j 处的数据
    Q->swap_index(i,j);
    // 交换标签 y 中索引 i 和 j 处的数据
    swap(y[i],y[j]);
    // 交换梯度 G 中索引 i 和 j 处的数据
    swap(G[i],G[j]);
    // 交换 alpha_status 中索引 i 和 j 处的数据
    swap(alpha_status[i],alpha_status[j]);
    // 交换 alpha 中索引 i 和 j 处的数据
    swap(alpha[i],alpha[j]);
    // 交换 p 中索引 i 和 j 处的数据
    swap(p[i],p[j]);
    // 交换 active_set 中索引 i 和 j 处的数据
    swap(active_set[i],active_set[j]);
    // 交换 G_bar 中索引 i 和 j 处的数据
    swap(G_bar[i],G_bar[j]);
    // 交换 C 中索引 i 和 j 处的数据
    swap(C[i], C[j]);
}

// 实现 reconstruct_gradient 方法，重新构建梯度 G 的非活跃元素
void Solver::reconstruct_gradient()
{
    // 如果活跃集的大小等于总样本数 l，直接返回
    if(active_size == l) return;

    int i,j;
    int nr_free = 0;

    // 重新构建 G 的非活跃元素
    for(j=active_size;j<l;j++)
        G[j] = G_bar[j] + p[j];

    // 统计当前活跃集中的自由变量个数
    for(j=0;j<active_size;j++)
        if(is_free(j))
            nr_free++;

    // 如果自由变量的数量小于活跃集大小的一半，发出警告
    if(2*nr_free < active_size)
        info("\nWarning: using -h 0 may be faster\n");

    // 根据 nr_free 和活跃集大小计算条件，选择不同的计算方式重构梯度 G
    if (nr_free*l > 2*active_size*(l-active_size))
    {
        // 使用 l - active_size 范围内的数据重新构建 G
        for(i=active_size;i<l;i++)
        {
            const Qfloat *Q_i = Q->get_Q(i,active_size);
            for(j=0;j<active_size;j++)
                if(is_free(j))
                    G[i] += alpha[j] * Q_i[j];
        }
    }
    else
    {
        // 使用活跃集中的自由变量重新构建 G
        for(i=0;i<active_size;i++)
            if(is_free(i))
            {
                const Qfloat *Q_i = Q->get_Q(i,l);
                double alpha_i = alpha[i];
                for(j=active_size;j<l;j++)
                    G[j] += alpha_i * Q_i[j];
            }
    }
}

// 实现 Solve 方法，解决 SVM 优化问题
void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
           double *alpha_, const double *C_, double eps,
           SolutionInfo* si, int shrinking, int max_iter)
{
    this->l = l;
    this->Q = &Q;
    QD=Q.get_QD();
    clone(p, p_,l);
    clone(y, y_,l);
    clone(alpha,alpha_,l);
    clone(C, C_, l);
    this->eps = eps;
    unshrink = false;
    si->solve_timed_out = false;

    // 初始化 alpha_status 数组
    {
        alpha_status = new char[l];
        for(int i=0;i<l;i++)
            update_alpha_status(i);
    }

    // 初始化活跃集 (用于收缩)
    {
        active_set = new int[l];
        for(int i=0;i<l;i++)
            active_set[i] = i;
        active_size = l;
    }

    // 初始化梯度 G 和 G_bar
    {
        G = new double[l];
        G_bar = new double[l];
        int i;
        for(i=0;i<l;i++)
        {
            G[i] = p[i];
            G_bar[i] = 0;
        }
        for(i=0;i<l;i++)
            if(!is_lower_bound(i))
            {
                const Qfloat *Q_i = Q.get_Q(i,l);
                double alpha_i = alpha[i];
                int j;
                for(j=0;j<l;j++)
                    G[j] += alpha_i*Q_i[j];
                if(is_upper_bound(i))
                    for(j=0;j<l;j++)
                        G_bar[j] += get_C(i) * Q_i[j];
            }
    }

    // 优化步骤
    int iter = 0;
    int counter = min(l,1000)+1;

    while(1)
    {
        // 计算 rho
        si->rho = calculate_rho();

        // 计算目标函数值
    {
        double v = 0;  // 初始化一个双精度变量 v，用于存储计算结果
        int i;  // 声明整型变量 i，用于循环索引
        for(i=0;i<l;i++)
            v += alpha[i] * (G[i] + p[i]);  // 计算 v 的值，累加 alpha[i] 乘以 (G[i] + p[i])

        si->obj = v/2;  // 将 v 的一半赋值给 si 结构体的 obj 成员
    }

    // 将解决方案放回
    {
        for(int i=0;i<l;i++)
            alpha_[active_set[i]] = alpha[i];  // 将 alpha 数组中的值复制回 alpha_ 数组对应的索引位置
    }

    // 将所有东西归位
    /*{
        for(int i=0;i<l;i++)
            while(active_set[i] != i)
                swap_index(i,active_set[i]);
                // 或者 Q.swap_index(i,active_set[i]);
    }*/

    // 将每个元素的上界存储到 si 结构体的 upper_bound 数组中
    for(int i=0;i<l;i++)
        si->upper_bound[i] = C[i];

    // 存储迭代次数
    si->n_iter = iter;

    // 输出优化完成的信息，并打印迭代次数
    info("\noptimization finished, #iter = %d\n",iter);

    // 释放动态分配的内存空间
    delete[] p;
    delete[] y;
    delete[] alpha;
    delete[] alpha_status;
    delete[] active_set;
    delete[] G;
    delete[] G_bar;
    delete[] C;
// 结束 Solver 类的一个私有成员函数的定义

// 返回1如果已经是最优解，否则返回0
int Solver::select_working_set(int &out_i, int &out_j)
{
    // 返回满足以下条件的 i, j：
    // i: 最大化 -y_i * grad(f)_i，其中 i 属于 I_up(\alpha)
    // j: 最小化目标值的减少量
    //    （如果二次系数 <= 0，则用 tau 替换）
    //    -y_j * grad(f)_j < -y_i * grad(f)_i，其中 j 属于 I_low(\alpha)

    double Gmax = -INF; // 最大的 G 值，初始为负无穷
    double Gmax2 = -INF; // 第二大的 G 值，初始为负无穷
    int Gmax_idx = -1; // Gmax 对应的索引
    int Gmin_idx = -1; // 最小目标值减少量对应的索引
    double obj_diff_min = INF; // 最小的目标值差异，初始为正无穷大

    for(int t=0; t<active_size; t++)
        if(y[t]==+1) // 对于 y[t] 等于 +1 的情况
        {
            if(!is_upper_bound(t)) // 如果 t 不是上界
                if(-G[t] >= Gmax) // 如果 -G[t] 大于等于当前的 Gmax
                {
                    Gmax = -G[t]; // 更新 Gmax
                    Gmax_idx = t; // 更新 Gmax 对应的索引
                }
        }
        else // 对于 y[t] 不等于 +1 的情况
        {
            if(!is_lower_bound(t)) // 如果 t 不是下界
                if(G[t] >= Gmax) // 如果 G[t] 大于等于当前的 Gmax
                {
                    Gmax = G[t]; // 更新 Gmax
                    Gmax_idx = t; // 更新 Gmax 对应的索引
                }
        }

    int i = Gmax_idx; // 获取最大 G 值对应的索引
    const Qfloat *Q_i = NULL; // 初始化 Q_i 为 NULL
    if(i != -1) // 如果 i 不等于 -1，说明 Q_i 不会被访问（因为 Gmax=-INF 当 i=-1 时）
        Q_i = Q->get_Q(i, active_size); // 获取 Q_i 的值

    for(int j=0; j<active_size; j++)
    {
        if(y[j]==+1) // 对于 y[j] 等于 +1 的情况
        {
            if (!is_lower_bound(j)) // 如果 j 不是下界
            {
                double grad_diff = Gmax + G[j]; // 计算 grad_diff
                if (G[j] >= Gmax2) // 如果 G[j] 大于等于当前的 Gmax2
                    Gmax2 = G[j]; // 更新 Gmax2
                if (grad_diff > 0) // 如果 grad_diff 大于 0
                {
                    double obj_diff;
                    double quad_coef = QD[i] + QD[j] - 2.0 * y[i] * Q_i[j]; // 计算二次系数
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff * grad_diff) / quad_coef; // 计算目标值差异
                    else
                        obj_diff = -(grad_diff * grad_diff) / TAU; // 如果二次系数 <= 0，用 TAU 替代

                    if (obj_diff <= obj_diff_min) // 如果目标值差异小于等于当前的 obj_diff_min
                    {
                        Gmin_idx = j; // 更新 Gmin_idx
                        obj_diff_min = obj_diff; // 更新 obj_diff_min
                    }
                }
            }
        }
        else // 对于 y[j] 不等于 +1 的情况
        {
            if (!is_upper_bound(j)) // 如果 j 不是上界
            {
                double grad_diff = Gmax - G[j]; // 计算 grad_diff
                if (-G[j] >= Gmax2) // 如果 -G[j] 大于等于当前的 Gmax2
                    Gmax2 = -G[j]; // 更新 Gmax2
                if (grad_diff > 0) // 如果 grad_diff 大于 0
                {
                    double obj_diff;
                    double quad_coef = QD[i] + QD[j] + 2.0 * y[i] * Q_i[j]; // 计算二次系数
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff * grad_diff) / quad_coef; // 计算目标值差异
                    else
                        obj_diff = -(grad_diff * grad_diff) / TAU; // 如果二次系数 <= 0，用 TAU 替代

                    if (obj_diff <= obj_diff_min) // 如果目标值差异小于等于当前的 obj_diff_min
                    {
                        Gmin_idx = j; // 更新 Gmin_idx
                        obj_diff_min = obj_diff; // 更新 obj_diff_min
                    }
                }
            }
        }
    }

    if (Gmax + Gmax2 < eps || Gmin_idx == -1) // 如果 Gmax + Gmax2 小于 eps 或者 Gmin_idx 等于 -1
        return 1; // 返回1，表示已经是最优解

    out_i = Gmax_idx; // 将最大的 G 值对应的索引赋给 out_i
    out_j = Gmin_idx; // 将最小目标值减少量对应的索引赋给 out_j
    return 0; // 返回0，表示不是最优解
}

// 返回是否需要收缩
bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
    if(is_upper_bound(i)) // 如果 i 是上界
        ```
    {
        # 如果样本 i 的标签 y[i] 为 +1
        if(y[i]==+1)
            # 返回 -G[i] 是否大于 Gmax1
            return(-G[i] > Gmax1);
        # 如果样本 i 的标签 y[i] 不为 +1
        else
            # 返回 -G[i] 是否大于 Gmax2
            return(-G[i] > Gmax2);
    }
    # 如果样本 i 是一个下界（支持向量的一种情况）
    else if(is_lower_bound(i))
    {
        # 如果样本 i 的标签 y[i] 为 +1
        if(y[i]==+1)
            # 返回 G[i] 是否大于 Gmax2
            return(G[i] > Gmax2);
        # 如果样本 i 的标签 y[i] 不为 +1
        else
            # 返回 G[i] 是否大于 Gmax1
            return(G[i] > Gmax1);
    }
    else
        # 其他情况下返回 false
        return(false);
//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver
{
public:
    // 默认构造函数
    Solver_NU() {}

    // 解决 NU-SVM 分类和回归问题
    void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
           double *alpha, const double *C_, double eps,
           SolutionInfo* si, int shrinking, int max_iter)
    {
        // 将解决结果存储在 si 中
        this->si = si;
        // 调用基类 Solver 的 Solve 方法解决问题
        Solver::Solve(l, Q, p, y, alpha, C_, eps, si, shrinking, max_iter);
    }

private:
    // 解决过程中存储解决方案信息的指针
    SolutionInfo *si;

    // 选择工作集的索引 i, j
    int select_working_set(int &i, int &j);

    // 计算 rho 值
    double calculate_rho();

    // 判断是否需要收缩变量
    bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);

    // 执行变量收缩
    void do_shrinking();
};

// 如果已经是最优解返回 1，否则返回 0
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
    // 返回 i, j，使得 y_i = y_j 并且
    // 初始化变量，用于记录正类样本中梯度乘积最大的值及其索引
    double Gmaxp = -INF;
    // 初始化变量，用于记录正类样本中第二大的梯度值
    double Gmaxp2 = -INF;
    // 初始化变量，记录正类样本中梯度乘积最大的样本索引
    int Gmaxp_idx = -1;

    // 初始化变量，用于记录负类样本中梯度乘积最大的值及其索引
    double Gmaxn = -INF;
    // 初始化变量，用于记录负类样本中第二大的梯度值
    double Gmaxn2 = -INF;
    // 初始化变量，记录负类样本中梯度乘积最大的样本索引
    int Gmaxn_idx = -1;

    // 初始化变量，记录最小的目标函数变化量
    int Gmin_idx = -1;
    // 初始化变量，用于记录目标函数变化量的最小值
    double obj_diff_min = INF;

    // 遍历活跃样本集合
    for(int t=0; t<active_size; t++)
        // 对于正类样本
        if(y[t] == +1)
        {
            // 如果该样本不是上界支持向量
            if(!is_upper_bound(t))
                // 如果该样本的梯度的负值大于等于当前记录的最大正梯度
                if(-G[t] >= Gmaxp)
                {
                    // 更新最大正梯度及其索引
                    Gmaxp = -G[t];
                    Gmaxp_idx = t;
                }
        }
        // 对于负类样本
        else
        {
            // 如果该样本不是下界支持向量
            if(!is_lower_bound(t))
                // 如果该样本的梯度值大于等于当前记录的最大负梯度
                if(G[t] >= Gmaxn)
                {
                    // 更新最大负梯度及其索引
                    Gmaxn = G[t];
                    Gmaxn_idx = t;
                }
        }

    // 记录最大正梯度样本索引和最大负梯度样本索引
    int ip = Gmaxp_idx;
    int in = Gmaxn_idx;
    // 用于存储Q矩阵中的指针变量，初始化为NULL
    const Qfloat *Q_ip = NULL;
    const Qfloat *Q_in = NULL;
    // 如果最大正梯度索引不为-1，则获取其对应的Q向量
    if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
        Q_ip = Q->get_Q(ip, active_size);
    // 如果最大负梯度索引不为-1，则获取其对应的Q向量
    if(in != -1)
        Q_in = Q->get_Q(in, active_size);

    // 遍历活跃样本集合
    for(int j=0; j<active_size; j++)
    {
        // 对于正类样本
        if(y[j] == +1)
        {
            // 如果该样本不是下界支持向量
            if (!is_lower_bound(j))
            {
                // 计算当前样本与最大正梯度样本的梯度差
                double grad_diff = Gmaxp + G[j];
                // 更新最大正梯度样本的第二大梯度值
                if (G[j] >= Gmaxp2)
                    Gmaxp2 = G[j];
                // 如果梯度差大于0
                if (grad_diff > 0)
                {
                    double obj_diff;
                    // 计算二次项系数
                    double quad_coef = QD[ip] + QD[j] - 2 * Q_ip[j];
                    // 如果二次项系数大于0，则计算目标函数变化量
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    else
                        obj_diff = -(grad_diff * grad_diff) / TAU;

                    // 更新最小目标函数变化量及其对应的样本索引
                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
        // 对于负类样本
        else
        {
            // 如果该样本不是上界支持向量
            if (!is_upper_bound(j))
            {
                // 计算当前样本与最大负梯度样本的梯度差
                double grad_diff = Gmaxn - G[j];
                // 更新最大负梯度样本的第二大梯度值
                if (-G[j] >= Gmaxn2)
                    Gmaxn2 = -G[j];
                // 如果梯度差大于0
                if (grad_diff > 0)
                {
                    double obj_diff;
                    // 计算二次项系数
                    double quad_coef = QD[in] + QD[j] - 2 * Q_in[j];
                    // 如果二次项系数大于0，则计算目标函数变化量
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    else
                        obj_diff = -(grad_diff * grad_diff) / TAU;

                    // 更新最小目标函数变化量及其对应的样本索引
                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    // 如果最大正梯度乘积及其第二大梯度值的和或者最大负梯度乘积及其第二大梯度值的和小于eps，
    // 或者最小目标函数变化量的索引为-1，则返回1
    if(max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < eps || Gmin_idx == -1)
        return 1;

    // 如果最小目标函数变化量对应的样本为正类，则设置输出的i为最大正梯度样本的索引
    if (y[Gmin_idx] == +1)
        out_i = Gmaxp_idx;
    // 否则，设置输出的i为最大负梯度样本的索引
    else
        out_i = Gmaxn_idx;
    // 设置输出的j为最小目标函数变化量的样本索引
    out_j = Gmin_idx;

    // 返回0表示成功执行
    return 0;
}

// 判断是否是上界
bool Solver_NU::is_upper_bound(int i)
{
    // 返回是否是上界
    return (status[i] == UPPER_BOUND);
}

// 判断是否是下界
bool Solver_NU::is_lower_bound(int i)
{
    // 返回是否是下界
    return (status[i] == LOWER_BOUND);
}

// 判断是否应该被缩小
bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
    if(is_upper_bound(i))
    {
        if(y[i]==+1)
            return (-G[i] > Gmax1); // 检查是否应该被缩小，基于Gmax1
        else
            return (-G[i] > Gmax4); // 检查是否应该被缩小，基于Gmax4
    }
    else if(is_lower_bound(i))
    {
        if(y[i]==+1)
            return (G[i] > Gmax2); // 检查是否应该被缩小，基于Gmax2
        else
            return (G[i] > Gmax3); // 检查是否应该被缩小，基于Gmax3
    }
    else
        return false; // 默认情况下不应该被缩小
}

// 执行缩小操作
void Solver_NU::do_shrinking()
{
    double Gmax1 = -INF;    // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
    double Gmax2 = -INF;    // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
    double Gmax3 = -INF;    // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
    double Gmax4 = -INF;    // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

    // 找到最大的违反对首先
    int i;
    for(i = 0; i < active_size; i++)
    {
        if(!is_upper_bound(i))
        {
            if(y[i] == +1)
            {
                if(-G[i] > Gmax1) Gmax1 = -G[i]; // 更新Gmax1
            }
            else if(-G[i] > Gmax4) Gmax4 = -G[i]; // 更新Gmax4
        }
        if(!is_lower_bound(i))
        {
            if(y[i] == +1)
            {
                if(G[i] > Gmax2) Gmax2 = G[i]; // 更新Gmax2
            }
            else if(G[i] > Gmax3) Gmax3 = G[i]; // 更新Gmax3
        }
    }

    // 如果不需要缩小并且最大的违反度对小于eps*10
    if(unshrink == false && max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= eps * 10)
    {
        unshrink = true;
        reconstruct_gradient();
        active_size = l;
    }

    // 开始缩小
    for(i = 0; i < active_size; i++)
    {
        if(be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
        {
            active_size--;
            while(active_size > i)
            {
                if(!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
                {
                    swap_index(i, active_size);
                    break;
                }
                active_size--;
            }
        }
    }
}

// 计算 rho 值
double Solver_NU::calculate_rho()
{
    int nr_free1 = 0, nr_free2 = 0;
    double ub1 = INF, ub2 = INF;
    double lb1 = -INF, lb2 = -INF;
    double sum_free1 = 0, sum_free2 = 0;

    // 统计自由项
    for(int i = 0; i < active_size; i++)
    {
        if(y[i] == +1)
        {
            if(is_upper_bound(i))
                lb1 = max(lb1, G[i]); // 更新 lb1
            else if(is_lower_bound(i))
                ub1 = min(ub1, G[i]); // 更新 ub1
            else
            {
                ++nr_free1;
                sum_free1 += G[i]; // 累加自由项
            }
        }
        else
        {
            if(is_upper_bound(i))
                lb2 = max(lb2, G[i]); // 更新 lb2
            else if(is_lower_bound(i))
                ub2 = min(ub2, G[i]); // 更新 ub2
            else
            {
                ++nr_free2;
                sum_free2 += G[i]; // 累加自由项
            }
        }
    }

    double r1, r2;
    if(nr_free1 > 0)
        r1 = sum_free1 / nr_free1; // 计算 r1
    else
        r1 = (ub1 + lb1) / 2; // 使用上下界的中间值作为 r1

    if(nr_free2 > 0)
        r2 = sum_free2 / nr_free2; // 计算 r2
    else
        r2 = (ub2 + lb2) / 2; // 使用上下界的中间值作为 r2

    si->r = (r1 + r2) / 2; // 更新 si 的 r 值
    return (r1 - r2) / 2; // 返回 rho 值
}

//
// 各种形式的 Q 矩阵
//
class SVC_Q: public Kernel
{
public:
    // SVC_Q 类的构造函数，接受问题对象 prob、参数对象 param、标签数组 y_ 和 BLAS 函数对象
    SVC_Q(const PREFIX(problem)& prob, const svm_parameter& param, const schar *y_, BlasFunctions *blas_functions)
    :Kernel(prob.l, prob.x, param, blas_functions)
    {
        // 复制标签数组 y_
        clone(y,y_,prob.l);
        // 创建缓存对象，设置缓存大小为 param.cache_size MB
        cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
        // 分配内存以保存 QD 数组
        QD = new double[prob.l];
        // 计算并存储 QD 数组的值
        for(int i=0;i<prob.l;i++)
            QD[i] = (this->*kernel_function)(i,i);
    }

    // 返回第 i 行起始的 Q 矩阵数据，长度为 len
    Qfloat *get_Q(int i, int len) const
    {
        Qfloat *data;
        int start, j;
        // 从缓存中获取数据，如果未能从缓存中获取全部数据，则计算其余部分并存入缓存
        if((start = cache->get_data(i,&data,len)) < len)
        {
            for(j=start;j<len;j++)
                // 计算并填充 data 数组
                data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
        }
        return data;
    }

    // 返回 QD 数组的指针
    double *get_QD() const
    {
        return QD;
    }

    // 交换索引 i 和 j 对应的数据
    void swap_index(int i, int j) const
    {
        // 交换缓存中的数据
        cache->swap_index(i,j);
        // 调用基类的 swap_index 方法
        Kernel::swap_index(i,j);
        // 交换标签数组中的元素
        swap(y[i],y[j]);
        // 交换 QD 数组中的元素
        swap(QD[i],QD[j]);
    }

    // SVC_Q 类的析构函数，释放动态分配的内存
    ~SVC_Q()
    {
        delete[] y;
        delete cache;
        delete[] QD;
    }
private:
    // 存储标签数组
    schar *y;
    // 缓存对象指针
    Cache *cache;
    // 保存 QD 数组的指针
    double *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
    // ONE_CLASS_Q 类的构造函数，接受问题对象 prob、参数对象 param 和 BLAS 函数对象
    ONE_CLASS_Q(const PREFIX(problem)& prob, const svm_parameter& param, BlasFunctions *blas_functions)
    :Kernel(prob.l, prob.x, param, blas_functions)
    {
        // 创建缓存对象，设置缓存大小为 param.cache_size MB
        cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
        // 分配内存以保存 QD 数组
        QD = new double[prob.l];
        // 计算并存储 QD 数组的值
        for(int i=0;i<prob.l;i++)
            QD[i] = (this->*kernel_function)(i,i);
    }

    // 返回第 i 行起始的 Q 矩阵数据，长度为 len
    Qfloat *get_Q(int i, int len) const
    {
        Qfloat *data;
        int start, j;
        // 从缓存中获取数据，如果未能从缓存中获取全部数据，则计算其余部分并存入缓存
        if((start = cache->get_data(i,&data,len)) < len)
        {
            for(j=start;j<len;j++)
                // 计算并填充 data 数组
                data[j] = (Qfloat)(this->*kernel_function)(i,j);
        }
        return data;
    }

    // 返回 QD 数组的指针
    double *get_QD() const
    {
        return QD;
    }

    // 交换索引 i 和 j 对应的数据
    void swap_index(int i, int j) const
    {
        // 交换缓存中的数据
        cache->swap_index(i,j);
        // 调用基类的 swap_index 方法
        Kernel::swap_index(i,j);
        // 交换 QD 数组中的元素
        swap(QD[i],QD[j]);
    }

    // ONE_CLASS_Q 类的析构函数，释放动态分配的内存
    ~ONE_CLASS_Q()
    {
        delete cache;
        delete[] QD;
    }
private:
    // 缓存对象指针
    Cache *cache;
    // 保存 QD 数组的指针
    double *QD;
};

class SVR_Q: public Kernel
{
public:
    // SVR_Q 类的构造函数，接受问题对象 prob、参数对象 param 和 BLAS 函数对象
    SVR_Q(const PREFIX(problem)& prob, const svm_parameter& param, BlasFunctions *blas_functions)
    :Kernel(prob.l, prob.x, param, blas_functions)
    {
        // 将 prob.l 赋值给 l
        l = prob.l;
        // 创建缓存对象，设置缓存大小为 param.cache_size MB
        cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
        // 分配内存以保存 QD 数组、sign 数组和 index 数组
        QD = new double[2*l];
        sign = new schar[2*l];
        index = new int[2*l];
        // 计算并存储 QD 数组的值，初始化 sign 和 index 数组
        for(int k=0;k<l;k++)
        {
            sign[k] = 1;
            sign[k+l] = -1;
            index[k] = k;
            index[k+l] = k;
            QD[k] = (this->*kernel_function)(k,k);
            QD[k+l] = QD[k];
        }
        // 分配内存以保存 buffer 数组
        buffer[0] = new Qfloat[2*l];
        buffer[1] = new Qfloat[2*l];
        // 初始化 next_buffer
        next_buffer = 0;
    }

    // 交换索引 i 和 j 对应的数据
    void swap_index(int i, int j) const
    {
        // 交换 sign 和 index 数组中的元素
        swap(sign[i],sign[j]);
        swap(index[i],index[j]);
        // 交换 QD 数组中的元素
        swap(QD[i],QD[j]);
    }


**注释：**


// SVR_Q 类的构造函数，接受问题对象 prob、参数对象 param 和 BLAS 函数对象
SVR_Q(const PREFIX(problem)& prob, const svm_parameter& param, BlasFunctions *blas_functions)
:Kernel(prob.l, prob.x, param, blas_functions)
{
    // 将 prob.l 赋值给 l
    l = prob.l;
    // 创建缓存对象，设置缓存大小为 param.cache_size MB
    cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
    // 分配内存以保存 QD 数组、sign 数组和 index 数组
    QD = new double[2*l];
    sign = new schar[2*l];
    index = new int[2*l];
    // 计算并存储 QD 数组的值，初始化 sign 和 index 数组
    for(int k=0;k<l;k++)
    {
        sign[k] = 1;
        sign[k+l] = -1;
        index[k] = k;
        index[k+l] = k;
        QD[k] = (this->*kernel_function)(k,k);
        QD[k+l] = QD[k];
    }
    // 分配内存以保存 buffer 数组
    buffer[0] = new Qfloat[2*l];
    buffer[1] = new Qfloat[2*l];
    // 初始化 next_buffer
    next_buffer = 0;
}

// 交换索引 i 和 j 对应的数据
void swap_index(int i, int j) const
{
    // 交换 sign 和 index 数组中的元素
    swap(sign[i],sign[j]);
    swap(index[i],index[j]);
    // 交换 QD 数组中的元素
    swap(QD[i],QD[j]);
}
    # 返回指定索引 i 对应的 Q 值数组指针，长度为 len
    Qfloat *get_Q(int i, int len) const
    {
        Qfloat *data;  // 定义 Q 值数据数组的指针
        int j, real_i = index[i];  // 声明循环变量 j 和真实索引 real_i
        // 如果缓存中没有索引为 real_i 的数据，则重新计算
        if(cache->get_data(real_i,&data,l) < l)
        {
            // 使用 kernel_function 计算索引为 real_i 和 j 的 Q 值，并存储到 data 数组中
            for(j=0;j<l;j++)
                data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
        }

        // 重新排序并复制数据到 buf 数组
        Qfloat *buf = buffer[next_buffer];  // 获取下一个可用的缓冲区
        next_buffer = 1 - next_buffer;  // 切换到另一个缓冲区
        schar si = sign[i];  // 获取索引 i 对应的符号
        // 计算并存储 si * sign[j] * data[index[j]] 到 buf 数组中
        for(j=0;j<len;j++)
            buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
        return buf;  // 返回存储了重新排序后数据的 buf 数组的指针
    }

    // 返回 QD 数组的指针，其中存储了样本的 QD 值
    double *get_QD() const
    {
        return QD;
    }

    // SVR_Q 类的析构函数，释放动态分配的内存
    ~SVR_Q()
    {
        delete cache;  // 释放缓存对象的内存
        delete[] sign;  // 释放符号数组的内存
        delete[] index;  // 释放索引数组的内存
        delete[] buffer[0];  // 释放缓冲区数组的第一个缓冲区的内存
        delete[] buffer[1];  // 释放缓冲区数组的第二个缓冲区的内存
        delete[] QD;  // 释放 QD 数组的内存
    }
private:
    int l;                          // 存储问题中的样本数量
    Cache *cache;                   // 缓存对象指针
    schar *sign;                    // 标签数组指针
    int *index;                     // 索引数组指针
    mutable int next_buffer;        // 下一个缓冲区的索引（可变）
    Qfloat *buffer[2];              // 两个缓冲区的 Q 值数组指针
    double *QD;                     // 双精度 QD 数组指针
};

//
// 构造和解决各种问题形式
//
static void solve_c_svc(
    const PREFIX(problem) *prob, const svm_parameter* param,
    double *alpha, Solver::SolutionInfo* si, double Cp, double Cn, BlasFunctions *blas_functions)
{
    int l = prob->l;                // 获取问题中的样本数量
    double *minus_ones = new double[l];     // 创建一个长度为 l 的双精度数组，初始化为 -1
    schar *y = new schar[l];        // 创建一个长度为 l 的有符号字符数组
    double *C = new double[l];      // 创建一个长度为 l 的双精度数组

    int i;

    for(i=0;i<l;i++)
    {
        alpha[i] = 0;               // 初始化 alpha 数组为 0
        minus_ones[i] = -1;         // 初始化 minus_ones 数组为 -1
        if(prob->y[i] > 0)
        {
            y[i] = +1;              // 如果标签为正类，则设置 y[i] 为 +1
            C[i] = prob->W[i] * Cp; // 根据 Cp 设置 C[i] 的值
        }
        else
        {
            y[i] = -1;              // 如果标签为负类，则设置 y[i] 为 -1
            C[i] = prob->W[i] * Cn; // 根据 Cn 设置 C[i] 的值
        }
    }

    Solver s;                       // 创建 Solver 对象
    s.Solve(l, SVC_Q(*prob,*param,y, blas_functions), minus_ones, y,
        alpha, C, param->eps, si, param->shrinking,
                param->max_iter);

        /*
    double sum_alpha=0;
    for(i=0;i<l;i++)
        sum_alpha += alpha[i];

    if (Cp==Cn)
        info("nu = %f\n", sum_alpha/(Cp*prob->l));
        */

    for(i=0;i<l;i++)
        alpha[i] *= y[i];           // 将 alpha 数组中的每个元素乘以相应的标签 y[i]

    delete[] C;                     // 释放 C 数组的内存
    delete[] minus_ones;            // 释放 minus_ones 数组的内存
    delete[] y;                     // 释放 y 数组的内存
}

static void solve_nu_svc(
    const PREFIX(problem) *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si, BlasFunctions *blas_functions)
{
    int i;
    int l = prob->l;                // 获取问题中的样本数量
    double nu = param->nu;          // 获取参数中的 nu 值

    schar *y = new schar[l];        // 创建一个长度为 l 的有符号字符数组
    double *C = new double[l];      // 创建一个长度为 l 的双精度数组

    for(i=0;i<l;i++)
    {
        if(prob->y[i] > 0)
            y[i] = +1;              // 如果标签为正类，则设置 y[i] 为 +1
        else
            y[i] = -1;              // 如果标签为负类，则设置 y[i] 为 -1

        C[i] = prob->W[i];          // 设置 C[i] 的值为 prob->W[i]
    }

    double nu_l = 0;
    for(i=0;i<l;i++) nu_l += nu * C[i];  // 计算 nu_l 的值

    double sum_pos = nu_l / 2;
    double sum_neg = nu_l / 2;

    for(i=0;i<l;i++)
    {
        if(y[i] == +1)
        {
            alpha[i] = std::min(C[i], sum_pos);   // 计算 alpha[i] 的值
            sum_pos -= alpha[i];            // 更新 sum_pos
        }
        else
        {
            alpha[i] = std::min(C[i], sum_neg);   // 计算 alpha[i] 的值
            sum_neg -= alpha[i];            // 更新 sum_neg
        }
    }

    double *zeros = new double[l];      // 创建一个长度为 l 的双精度数组，初始化为 0

    for(i=0;i<l;i++)
        zeros[i] = 0;                   // 初始化 zeros 数组为 0

    Solver_NU s;                        // 创建 Solver_NU 对象
    s.Solve(l, SVC_Q(*prob,*param,y,blas_functions), zeros, y,
        alpha, C, param->eps, si,  param->shrinking, param->max_iter);
    double r = si->r;

    info("C = %f\n",1/r);               // 打印信息，计算 C 的值

    for(i=0;i<l;i++)
    {
        alpha[i] *= y[i] / r;           // 更新 alpha[i] 的值
        si->upper_bound[i] /= r;        // 更新 si->upper_bound[i] 的值
    }

    si->rho /= r;                       // 更新 si->rho 的值
    si->obj /= (r * r);                 // 更新 si->obj 的值

    delete[] C;                         // 释放 C 数组的内存
    delete[] y;                         // 释放 y 数组的内存
    delete[] zeros;                     // 释放 zeros 数组的内存
}

static void solve_one_class(
    const PREFIX(problem) *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si, BlasFunctions *blas_functions)
{
    int l = prob->l;                    // 获取问题中的样本数量
    double *zeros = new double[l];      // 创建一个长度为 l 的双精度数组，初始化为 0
    schar *ones = new schar[l];         // 创建一个长度为 l 的有符号字符数组，初始化为 1
    double *C = new double[l];          // 创建一个长度为 l 的双精度数组
    int i;

    double nu_l = 0;

    for(i=0;i<l;i++)
    {
        C[i] = prob->W[i];              // 设置 C[i] 的值为 prob->W[i]
        nu_l += C[i] * param->nu;       // 计算 nu_l 的值
    }

    i = 0;
    while(nu_l > 0)
    {
        alpha[i] = min(C[i],nu_l);
        // 计算 alpha[i]，取 C[i] 和 nu_l 中较小的值赋给 alpha[i]
        nu_l -= alpha[i];
        // 更新 nu_l，减去当前计算得到的 alpha[i]
        ++i;
        // 增加索引 i，进入下一个循环迭代
    }
    for(;i<l;i++)
        alpha[i] = 0;
    // 将剩余的 alpha[i]（i >= i）设为 0，确保所有 alpha 数组元素都被赋值
    
    for(i=0;i<l;i++)
    {
        zeros[i] = 0;
        // 初始化 zeros 数组，所有元素设为 0
        ones[i] = 1;
        // 初始化 ones 数组，所有元素设为 1
    }
    
    Solver s;
    // 创建 Solver 对象 s
    
    s.Solve(l, ONE_CLASS_Q(*prob,*param,blas_functions), zeros, ones,
        alpha, C, param->eps, si, param->shrinking, param->max_iter);
    // 调用 Solver 对象 s 的 Solve 方法进行求解，传入参数包括数据长度 l、问题类型 ONE_CLASS_Q、数据数组 zeros 和 ones、alpha 数组、C 数组、参数 param 的 epsilon、si、是否收缩和最大迭代次数
    
    delete[] C;
    // 释放数组 C 占用的内存空间
    delete[] zeros;
    // 释放数组 zeros 占用的内存空间
    delete[] ones;
    // 释放数组 ones 占用的内存空间
}

// 解决 epsilon-SVR 的函数
static void solve_epsilon_svr(
    const PREFIX(problem) *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si, BlasFunctions *blas_functions)
{
    // 获取问题中的样本数量
    int l = prob->l;

    // 分配额外的空间存储相关变量
    double *alpha2 = new double[2*l];
    double *linear_term = new double[2*l];
    schar *y = new schar[2*l];
    double *C = new double[2*l];
    int i;

    // 初始化数组
    for(i=0;i<l;i++)
    {
        alpha2[i] = 0;
        // 计算线性项
        linear_term[i] = param->p - prob->y[i];
        y[i] = 1;
        // 计算惩罚因子
        C[i] = prob->W[i] * param->C;

        alpha2[i+l] = 0;
        // 计算线性项
        linear_term[i+l] = param->p + prob->y[i];
        y[i+l] = -1;
        // 计算惩罚因子
        C[i+l] = prob->W[i] * param->C;
    }

    // 创建 Solver 对象并解决问题
    Solver s;
    s.Solve(2*l, SVR_Q(*prob, *param, blas_functions), linear_term, y,
        alpha2, C, param->eps, si, param->shrinking, param->max_iter);

    // 计算 alpha
    double sum_alpha = 0;
    for(i=0;i<l;i++)
    {
        alpha[i] = alpha2[i] - alpha2[i+l];
        sum_alpha += fabs(alpha[i]);
    }

    // 释放内存
    delete[] alpha2;
    delete[] linear_term;
    delete[] C;
    delete[] y;
}

// 解决 nu-SVR 的函数
static void solve_nu_svr(
    const PREFIX(problem) *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si, BlasFunctions *blas_functions)
{
    // 获取问题中的样本数量
    int l = prob->l;

    // 分配额外的空间存储相关变量
    double *C = new double[2*l];
    double *alpha2 = new double[2*l];
    double *linear_term = new double[2*l];
    schar *y = new schar[2*l];
    int i;

    // 计算参数的总和
    double sum = 0;
    for(i=0;i<l;i++)
    {
        C[i] = C[i+l] = prob->W[i] * param->C;
        sum += C[i] * param->nu;
    }
    sum /= 2;

    // 初始化数组
    for(i=0;i<l;i++)
    {
        alpha2[i] = alpha2[i+l] = std::min(sum, C[i]);
        sum -= alpha2[i];

        linear_term[i] = -prob->y[i];
        y[i] = 1;

        linear_term[i+l] = prob->y[i];
        y[i+l] = -1;
    }

    // 创建 Solver_NU 对象并解决问题
    Solver_NU s;
    s.Solve(2*l, SVR_Q(*prob, *param, blas_functions), linear_term, y,
        alpha2, C, param->eps, si, param->shrinking, param->max_iter);

    // 输出求解结果的信息
    info("epsilon = %f\n", -si->r);

    // 计算 alpha
    for(i=0;i<l;i++)
        alpha[i] = alpha2[i] - alpha2[i+l];

    // 释放内存
    delete[] alpha2;
    delete[] linear_term;
    delete[] C;
    delete[] y;
}

//
// decision_function 结构体
//
struct decision_function
{
    double *alpha;  // 支持向量的权重数组
    double rho;     // 决策函数的偏置项
    int n_iter;     // 迭代次数
};

// 训练单个 SVM 模型的函数
static decision_function svm_train_one(
    const PREFIX(problem) *prob, const svm_parameter *param,
    double Cp, double Cn, int *status, BlasFunctions *blas_functions)
{
    // 分配空间存储支持向量的权重
    double *alpha = Malloc(double, prob->l);
    // 创建 SolutionInfo 结构体
    Solver::SolutionInfo si;
    
    // 根据 SVM 类型选择不同的求解方法
    switch(param->svm_type)
    {
        // 根据不同的 SVM 类型分配上界数组
        case C_SVC:
            si.upper_bound = Malloc(double,prob->l);
            // 解决 C_SVC 类型的 SVM 问题
            solve_c_svc(prob,param,alpha,&si,Cp,Cn,blas_functions);
            break;
        case NU_SVC:
            si.upper_bound = Malloc(double,prob->l);
            // 解决 NU_SVC 类型的 SVM 问题
            solve_nu_svc(prob,param,alpha,&si,blas_functions);
            break;
        case ONE_CLASS:
            si.upper_bound = Malloc(double,prob->l);
            // 解决 ONE_CLASS 类型的 SVM 问题
            solve_one_class(prob,param,alpha,&si,blas_functions);
            break;
        case EPSILON_SVR:
            si.upper_bound = Malloc(double,2*prob->l);
            // 解决 EPSILON_SVR 类型的 SVM 问题
            solve_epsilon_svr(prob,param,alpha,&si,blas_functions);
            break;
        case NU_SVR:
            si.upper_bound = Malloc(double,2*prob->l);
            // 解决 NU_SVR 类型的 SVM 问题
            solve_nu_svr(prob,param,alpha,&si,blas_functions);
            break;
    }

    // 更新状态，包括是否超时
    *status |= si.solve_timed_out;

    // 输出优化问题的目标值和 rho 值
    info("obj = %f, rho = %f\n",si.obj,si.rho);

    // 输出支持向量的信息

    int nSV = 0;    // 支持向量的数量
    int nBSV = 0;   // 边界支持向量的数量
    for(int i=0;i<prob->l;i++)
    {
        if(fabs(alpha[i]) > 0)
        {
            ++nSV;
            if(prob->y[i] > 0)
            {
                // 正类支持向量
                if(fabs(alpha[i]) >= si.upper_bound[i])
                    ++nBSV; // 边界支持向量数加一
            }
            else
            {
                // 负类支持向量
                if(fabs(alpha[i]) >= si.upper_bound[i])
                    ++nBSV; // 边界支持向量数加一
            }
        }
    }

    // 释放 si.upper_bound 的内存
    free(si.upper_bound);

    // 输出支持向量和边界支持向量的数量
    info("nSV = %d, nBSV = %d\n",nSV,nBSV);

    // 构建决策函数对象并返回
    decision_function f;
    f.alpha = alpha;
    f.rho = si.rho;
    f.n_iter = si.n_iter;
    return f;
}

// Platt's binary SVM Probabilistic Output: an improvement from Lin et al.
// 计算 sigmoid 函数的参数 A 和 B，用于概率输出
static void sigmoid_train(
    int l, const double *dec_values, const double *labels,
    double& A, double& B)
{
    // 计算正类和负类的先验概率
    double prior1=0, prior0 = 0;
    int i;

    for (i=0;i<l;i++)
        if (labels[i] > 0) prior1+=1;
        else prior0+=1;

    // 最大迭代次数
    int max_iter=100;    // Maximal number of iterations
    // 线搜索中的最小步长
    double min_step=1e-10;    // Minimal step taken in line search
    // 数值上确保 Hessian 矩阵是正定的的参数
    double sigma=1e-12;    // For numerically strict PD of Hessian
    // 精度阈值
    double eps=1e-5;
    // 计算类别的目标概率
    double hiTarget=(prior1+1.0)/(prior1+2.0);
    double loTarget=1/(prior0+2.0);
    double *t=Malloc(double,l);  // 分配 t 数组的内存空间
    double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
    double newA,newB,newf,d1,d2;
    int iter;

    // 初始化点和初始函数值
    A=0.0; B=log((prior0+1.0)/(prior1+1.0));
    double fval = 0.0;

    // 计算初始的函数值 fval 和 t 数组
    for (i=0;i<l;i++)
    {
        if (labels[i]>0) t[i]=hiTarget;
        else t[i]=loTarget;
        fApB = dec_values[i]*A+B;
        if (fApB>=0)
            fval += t[i]*fApB + log(1+exp(-fApB));
        else
            fval += (t[i] - 1)*fApB +log(1+exp(fApB));
    }

    // 开始迭代优化 A 和 B 参数
    for (iter=0;iter<max_iter;iter++)
    {
        // 更新梯度和黑塞矩阵（使用 H' = H + sigma I）
        h11=sigma; // 数值上确保是严格正定的
        h22=sigma;
        h21=0.0;g1=0.0;g2=0.0;
        for (i=0;i<l;i++)
        {
            // 计算 f(x) = wx + b
            fApB = dec_values[i]*A+B;
            if (fApB >= 0)
            {
                // 计算 logistic loss 中的 p 和 q
                p=exp(-fApB)/(1.0+exp(-fApB));
                q=1.0/(1.0+exp(-fApB));
            }
            else
            {
                p=1.0/(1.0+exp(fApB));
                q=exp(fApB)/(1.0+exp(fApB));
            }
            // 计算二阶导数 d2 = p * (1 - p)
            d2=p*q;
            // 更新 Hessian 矩阵的元素
            h11+=dec_values[i]*dec_values[i]*d2;
            h22+=d2;
            h21+=dec_values[i]*d2;
            // 计算梯度 g = g + (-gradient of logistic loss)
            d1=t[i]-p;
            g1+=dec_values[i]*d1;
            g2+=d1;
        }

        // 停止条件
        if (fabs(g1)<eps && fabs(g2)<eps)
            break;

        // 寻找牛顿方向：-inv(H') * g
        det=h11*h22-h21*h21;
        dA=-(h22*g1 - h21 * g2) / det;
        dB=-(-h21*g1+ h11 * g2) / det;
        gd=g1*dA+g2*dB;

        stepsize = 1;        // 线搜索
        while (stepsize >= min_step)
        {
            // 计算新的 A 和 B
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            // 计算新的函数值
            newf = 0.0;
            for (i=0;i<l;i++)
            {
                // 计算更新后的 f(x) = w'x + b'
                fApB = dec_values[i]*newA+newB;
                if (fApB >= 0)
                    newf += t[i]*fApB + log(1+exp(-fApB));
                else
                    newf += (t[i] - 1)*fApB +log(1+exp(fApB));
            }
            // 检查足够的减少条件
            if (newf<fval+0.0001*stepsize*gd)
            {
                // 更新参数 A, B 和函数值 fval
                A=newA;B=newB;fval=newf;
                break;
            }
            else
                // 减小步长
                stepsize = stepsize / 2.0;
        }

        // 如果步长小于最小步长，表示线搜索失败
        if (stepsize < min_step)
        {
            info("Line search fails in two-class probability estimates\n");
            break;
        }
    }

    // 如果迭代次数超过最大允许迭代次数，输出警告信息
    if (iter>=max_iter)
        info("Reaching maximal iterations in two-class probability estimates\n");
    // 释放临时变量 t 的内存空间
    free(t);
}

// 预测时使用的 sigmoid 函数，用于将决策值映射到概率值
static double sigmoid_predict(double decision_value, double A, double B)
{
    double fApB = decision_value*A+B;
    // 避免数值溢出，特别处理当 fApB 较大时的情况
    if (fApB >= 0)
        return exp(-fApB)/(1.0+exp(-fApB));
    else
        return 1.0/(1+exp(fApB)) ;
}

// 使用多类别概率方法计算每个类别的概率
static void multiclass_probability(int k, double **r, double *p)
{
    int t,j;
    int iter = 0, max_iter=max(100,k);
    double **Q=Malloc(double *,k);
    double *Qp=Malloc(double,k);
    double pQp, eps=0.005/k;

    for (t=0;t<k;t++)
    {
        p[t]=1.0/k;  // 如果 k = 1 则有效
        Q[t]=Malloc(double,k);
        Q[t][t]=0;
        for (j=0;j<t;j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=Q[j][t];
        }
        for (j=t+1;j<k;j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=-r[j][t]*r[t][j];
        }
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // 停止条件，重新计算 QP 和 pQP 以提高数值精度
        pQp=0;
        for (t=0;t<k;t++)
        {
            Qp[t]=0;
            for (j=0;j<k;j++)
                Qp[t]+=Q[t][j]*p[j];
            pQp+=p[t]*Qp[t];
        }
        double max_error=0;
        for (t=0;t<k;t++)
        {
            double error=fabs(Qp[t]-pQp);
            if (error>max_error)
                max_error=error;
        }
        if (max_error<eps) break;

        for (t=0;t<k;t++)
        {
            double diff=(-Qp[t]+pQp)/Q[t][t];
            p[t]+=diff;
            pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
            for (j=0;j<k;j++)
            {
                Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
                p[j]/=(1+diff);
            }
        }
    }
    if (iter>=max_iter)
        info("Exceeds max_iter in multiclass_prob\n");
    for(t=0;t<k;t++) free(Q[t]);
    free(Q);
    free(Qp);
}

// 计算二分类 SVM 的概率估计的交叉验证决策值
static void svm_binary_svc_probability(
    const PREFIX(problem) *prob, const svm_parameter *param,
    double Cp, double Cn, double& probA, double& probB, int * status, BlasFunctions *blas_functions)
{
    int i;
    int nr_fold = 5;
    int *perm = Malloc(int,prob->l);
    double *dec_values = Malloc(double,prob->l);

    // 随机打乱数据顺序
    for(i=0;i<prob->l;i++) perm[i]=i;
    for(i=0;i<prob->l;i++)
    {
        int j = i+bounded_rand_int(prob->l-i);
        swap(perm[i],perm[j]);
    }
    for(i=0;i<nr_fold;i++)
    {
        int begin = i*prob->l/nr_fold;
        int end = (i+1)*prob->l/nr_fold;
        int j,k;
        struct PREFIX(problem) subprob;

        subprob.l = prob->l-(end-begin);
#ifdef _DENSE_REP
        subprob.x = Malloc(struct PREFIX(node),subprob.l);
#else
        subprob.x = Malloc(struct PREFIX(node)*,subprob.l);
#endif
        subprob.y = Malloc(double,subprob.l);

        k=0;
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k++] = prob->y[perm[j]];
        }
        for(j=end;j<prob->l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k++] = prob->y[perm[j]];
        }

        int sub_status = svm_svm_binary_svc(
            &subprob, param, Cp, Cn, probA, probB, blas_functions);
        if (sub_status)
        {
            *status = 1;
            break;
        }
    }
    free(perm);
    free(dec_values);
}
#endif
        // 分配内存给 subprob.y 和 subprob.W
        subprob.y = Malloc(double,subprob.l);
        subprob.W = Malloc(double,subprob.l);

        k=0;
        // 复制 perm 数组中的一部分数据到 subprob 结构中
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            subprob.W[k] = prob->W[perm[j]];
            ++k;
        }
        // 复制 perm 数组中的另一部分数据到 subprob 结构中
        for(j=end;j<prob->l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            subprob.W[k] = prob->W[perm[j]];
            ++k;
        }
        int p_count=0,n_count=0;
        // 计算 subprob.y 中正负样本的数量
        for(j=0;j<k;j++)
            if(subprob.y[j]>0)
                p_count++;
            else
                n_count++;

        // 根据正负样本数量设定 dec_values[perm[j]] 的值
        if(p_count==0 && n_count==0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = 0;
        else if(p_count > 0 && n_count == 0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = 1;
        else if(p_count == 0 && n_count > 0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = -1;
        else
        {
            // 创建子问题的参数设置
            svm_parameter subparam = *param;
            subparam.probability=0;
            subparam.C=1.0;
            subparam.nr_weight=2;
            subparam.weight_label = Malloc(int,2);
            subparam.weight = Malloc(double,2);
            subparam.weight_label[0]=+1;
            subparam.weight_label[1]=-1;
            subparam.weight[0]=Cp;
            subparam.weight[1]=Cn;
            // 训练 SVM 模型得到 submodel
            struct PREFIX(model) *submodel = PREFIX(train)(&subprob,&subparam, status, blas_functions);
            // 根据不同情况预测并修正 dec_values[perm[j]] 的值
            for(j=begin;j<end;j++)
            {
#ifdef _DENSE_REP
                // 使用 submodel 预测值，更新 dec_values[perm[j]]
                PREFIX(predict_values)(submodel,(prob->x+perm[j]),&(dec_values[perm[j]]), blas_functions);
#else
                // 使用 submodel 预测值，更新 dec_values[perm[j]]
                PREFIX(predict_values)(submodel,prob->x[perm[j]],&(dec_values[perm[j]]), blas_functions);
#endif
                // 确保 dec_values[perm[j]] 的正负符号与 submodel 的标签一致
                dec_values[perm[j]] *= submodel->label[0];
            }
            // 释放并销毁 submodel 相关资源
            PREFIX(free_and_destroy_model)(&submodel);
            PREFIX(destroy_param)(&subparam);
        }
        // 释放 subprob 中分配的内存
        free(subprob.x);
        free(subprob.y);
        free(subprob.W);
    }
    // 使用 sigmoid_train 训练后的 dec_values 进行概率估计
    sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
    // 释放 dec_values 和 perm 数组的内存
    free(dec_values);
    free(perm);
}

// 返回 Laplace 分布的参数
static double svm_svr_probability(
    const PREFIX(problem) *prob, const svm_parameter *param, BlasFunctions *blas_functions)
{
    int i;
    int nr_fold = 5;
    // 分配内存给 ymv 数组
    double *ymv = Malloc(double,prob->l);
    double mae = 0;

    // 复制参数 param，并设置 probability 和 random_seed 属性
    svm_parameter newparam = *param;
    newparam.probability = 0;
    newparam.random_seed = -1; // 这里调用的是 train 函数，它已经设置了随机种子

    // 使用交叉验证计算 ymv 数组
    PREFIX(cross_validation)(prob,&newparam,nr_fold,ymv, blas_functions);
    // 计算平均绝对误差
    for(i=0;i<prob->l;i++)
    {
        ymv[i]=prob->y[i]-ymv[i];
        mae += fabs(ymv[i]);
    }
    mae /= prob->l;


这段代码涉及 SVM 模型训练和交叉验证过程中的数据处理和模型操作，详细注释了每一行代码的作用和意图。
    // 计算标准差为sqrt(2*mae*mae)
    double std=sqrt(2*mae*mae);
    // 初始化计数器count为0
    int count=0;
    // 将mae重置为0
    mae=0;
    // 循环遍历prob->l个数据点
    for(i=0;i<prob->l;i++)
        // 如果ymv[i]的绝对值大于5倍std
        if (fabs(ymv[i]) > 5*std)
            // 计数器count加1
            count=count+1;
        else
            // 否则将ymv[i]的绝对值加到mae中
            mae+=fabs(ymv[i]);
    // 将mae除以(prob->l-count)得到平均绝对误差
    mae /= (prob->l-count);
    // 输出关于测试数据的概率模型信息，其中z服从拉普拉斯分布，sigma为mae
    info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
    // 释放ymv数组的内存空间
    free(ymv);
    // 返回计算得到的平均绝对误差mae
    return mae;
// label: 标签名称，start: 每个类别的起始索引，count: 每个类别的数据数量，perm: 原始数据的索引排列
// perm 数组必须在调用此子程序之前分配内存空间
static void svm_group_classes(const PREFIX(problem) *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
    int l = prob->l;                    // 获取问题 prob 的数据数量 l
    int max_nr_class = 16;              // 初始最大类别数设为 16
    int nr_class = 0;                   // 类别数量初始化为 0
    int *label = Malloc(int,max_nr_class);    // 分配初始大小的 label 数组内存空间
    int *count = Malloc(int,max_nr_class);    // 分配初始大小的 count 数组内存空间
    int *data_label = Malloc(int,l);    // 分配大小为 l 的数据标签数组内存空间
    int i, j, this_label, this_count;

    for(i=0;i<l;i++)
    {
        this_label = (int)prob->y[i];   // 获取第 i 个数据点的标签
        for(j=0;j<nr_class;j++)
        {
            if(this_label == label[j])   // 如果标签已经存在于 label 数组中
            {
                ++count[j];             // 对应类别的计数加一
                break;
            }
        }
        if(j == nr_class)                // 如果标签是新的
        {
            if(nr_class == max_nr_class) // 如果类别数量达到最大容量
            {
                max_nr_class *= 2;      // 扩大最大类别容量
                label = (int *)realloc(label,max_nr_class*sizeof(int));  // 重新分配更大的 label 数组空间
                count = (int *)realloc(count,max_nr_class*sizeof(int));  // 重新分配更大的 count 数组空间
            }
            label[nr_class] = this_label;   // 将新标签加入 label 数组
            count[nr_class] = 1;            // 对应类别的计数设为 1
            ++nr_class;                     // 类别数量加一
        }
    }

    /*
     * 对标签按直接插入法进行排序，并应用相同的转换到 count 数组。
     */
    for(j=1; j<nr_class; j++)
    {
        i = j-1;
        this_label = label[j];
        this_count = count[j];
        while(i>=0 && label[i] > this_label)
        {
            label[i+1] = label[i];
            count[i+1] = count[i];
            i--;
        }
        label[i+1] = this_label;
        count[i+1] = this_count;
    }

    for (i=0; i<l; i++)
    {
        j = 0;
        this_label = (int)prob->y[i];
        while(this_label != label[j]){
            j ++;
        }
        data_label[i] = j;                  // 将数据点的标签转换为类别索引并存储在 data_label 数组中
    }

    int *start = Malloc(int,nr_class);      // 分配大小为 nr_class 的 start 数组内存空间
    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+count[i-1];   // 计算每个类别在 perm 数组中的起始位置
    for(i=0;i<l;i++)
    {
        perm[start[data_label[i]]] = i;     // 根据类别索引将数据点索引写入 perm 数组
        ++start[data_label[i]];             // 更新 start 数组中对应类别的起始位置
    }

    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+count[i-1];   // 再次计算每个类别在 perm 数组中的起始位置

    *nr_class_ret = nr_class;               // 返回类别数量
    *label_ret = label;                     // 返回标签数组
    *start_ret = start;                     // 返回起始位置数组
    *count_ret = count;                     // 返回每个类别数据数量数组
    free(data_label);                       // 释放 data_label 数组内存空间
}
    # 遍历 prob 结构体中的元素，其中 prob->l 表示元素个数
    for(i=0;i<prob->l;i++)
        # 检查 prob->W[i] 是否大于 0
        if(prob->W[i] > 0)
        {
            # 将 prob->x[i] 复制给 newprob->x[j]
            newprob->x[j] = prob->x[i];
            # 将 prob->y[i] 复制给 newprob->y[j]
            newprob->y[j] = prob->y[i];
            # 将 prob->W[i] 复制给 newprob->W[j]
            newprob->W[j] = prob->W[i];
            # j 自增，用于指向下一个新元素的位置
            j++;
        }
    // 接口函数：训练 SVM 模型
    // 参数：
    //   prob: SVM 训练问题的指针
    //   param: SVM 参数
    //   status: 指示训练状态的整数指针
    //   blas_functions: BLAS 函数集合指针
    // 返回值：
    //   训练得到的 SVM 模型指针
    PREFIX(model) *PREFIX(train)(const PREFIX(problem) *prob, const svm_parameter *param,
            int *status, BlasFunctions *blas_functions)
    {
        // 移除权重为零的样本
        PREFIX(problem) newprob;
        remove_zero_weight(&newprob, prob);
        prob = &newprob;

        // 分配内存以存储模型
        PREFIX(model) *model = Malloc(PREFIX(model), 1);
        model->param = *param;
        model->free_sv = 0;    // XXX

        // 如果随机种子非负，设置随机种子
        if (param->random_seed >= 0)
        {
            set_seed(param->random_seed);
        }

        // 如果是回归或者单类别 SVM
        if (param->svm_type == ONE_CLASS ||
           param->svm_type == EPSILON_SVR ||
           param->svm_type == NU_SVR)
        {
            // 设置模型的类别数为 2
            model->nr_class = 2;
            model->label = NULL;
            model->nSV = NULL;
            model->probA = NULL; model->probB = NULL;
            model->sv_coef = Malloc(double *, 1);

            // 如果需要概率估计，并且是 epsilon-SVR 或者 nu-SVR
            if (param->probability &&
               (param->svm_type == EPSILON_SVR ||
                param->svm_type == NU_SVR))
            {
                // 计算 SVR 模型的概率 A 值
                model->probA = Malloc(double, 1);
                model->probA[0] = NAMESPACE::svm_svr_probability(prob, param, blas_functions);
            }

            // 使用 SVM 训练一个决策函数
            NAMESPACE::decision_function f = NAMESPACE::svm_train_one(prob, param, 0, 0, status, blas_functions);
            model->rho = Malloc(double, 1);
            model->rho[0] = f.rho;
            model->n_iter = Malloc(int, 1);
            model->n_iter[0] = f.n_iter;

            // 计算支持向量的数量
            int nSV = 0;
            int i;
            for (i = 0; i < prob->l; i++)
                if (fabs(f.alpha[i]) > 0) ++nSV;
            model->l = nSV;

            // 分配内存存储支持向量
    #ifdef _DENSE_REP
            model->SV = Malloc(PREFIX(node), nSV);
    #else
            model->SV = Malloc(PREFIX(node) *, nSV);
    #endif
            model->sv_ind = Malloc(int, nSV);
            model->sv_coef[0] = Malloc(double, nSV);
            int j = 0;
            for (i = 0; i < prob->l; i++)
                if (fabs(f.alpha[i]) > 0)
                {
                    // 存储支持向量和相关信息
                    model->SV[j] = prob->x[i];
                    model->sv_ind[j] = i;
                    model->sv_coef[0][j] = f.alpha[i];
                    ++j;
                }

            // 释放决策函数的 alpha 内存
            free(f.alpha);
        }
        else
        {
            // 分类问题
            int l = prob->l;
            int nr_class;
            int *label = NULL;
            int *start = NULL;
            int *count = NULL;
            int *perm = Malloc(int, l);

            // 分组同一类别的训练数据
            NAMESPACE::svm_group_classes(prob, &nr_class, &label, &start, &count, perm);
    #ifdef _DENSE_REP
            PREFIX(node) *x = Malloc(PREFIX(node), l);
    #else
            PREFIX(node) **x = Malloc(PREFIX(node) *, l);
    #endif
#endif
                // 分配双精度数组 W，用于存储数据权重
                double *W = Malloc(double, l);

        // 循环遍历数据集进行处理
        int i;
        for(i=0;i<l;i++)
                {
            // 根据排列顺序 perm[i] 选择相应的输入向量和权重
            x[i] = prob->x[perm[i]];
            W[i] = prob->W[perm[i]];
                }

        // 计算加权的 C 值

        // 分配并初始化加权 C 数组
        double *weighted_C = Malloc(double, nr_class);
        for(i=0;i<nr_class;i++)
            weighted_C[i] = param->C;
        // 根据参数中指定的权重对相应的类别进行加权处理
        for(i=0;i<param->nr_weight;i++)
        {
            int j;
            for(j=0;j<nr_class;j++)
                if(param->weight_label[i] == label[j])
                    break;
            // 如果指定的类别标签未找到，输出警告信息
            if(j == nr_class)
                fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
            else
                weighted_C[j] *= param->weight[i];
        }

        // 训练 k*(k-1)/2 个模型

        // 分配用于存储是否非零的标记数组
        bool *nonzero = Malloc(bool,l);
        for(i=0;i<l;i++)
            nonzero[i] = false;
                // 分配存储决策函数的数组
                NAMESPACE::decision_function *f = Malloc(NAMESPACE::decision_function,nr_class*(nr_class-1)/2);

        // 分配用于存储概率估计的数组
        double *probA=NULL,*probB=NULL;
        if (param->probability)
        {
            probA=Malloc(double,nr_class*(nr_class-1)/2);
            probB=Malloc(double,nr_class*(nr_class-1)/2);
        }

        // 初始化索引变量 p，并循环处理每一对类别
        int p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                // 初始化子问题 sub_prob
                PREFIX(problem) sub_prob;
                int si = start[i], sj = start[j];
                int ci = count[i], cj = count[j];
                sub_prob.l = ci+cj;
#ifdef _DENSE_REP
                // 分配稠密表示下的节点数组
                sub_prob.x = Malloc(PREFIX(node),sub_prob.l);
#else
                // 分配非稠密表示下的节点指针数组
                sub_prob.x = Malloc(PREFIX(node) *,sub_prob.l);
`
#endif
                # 分配内存给 sub_prob 的 W 数组，大小为 sub_prob.l
                sub_prob.W = Malloc(double, sub_prob.l);
                # 分配内存给 sub_prob 的 y 数组，大小为 sub_prob.l
                sub_prob.y = Malloc(double, sub_prob.l);
                int k;
                # 遍历 ci 个样本，将它们赋值给 sub_prob 的 x 和 y 数组，并初始化权重 W
                for(k=0; k<ci; k++)
                {
                    sub_prob.x[k] = x[si+k];  # 从 x 数组中复制 si+k 的值到 sub_prob 的 x 数组
                    sub_prob.y[k] = +1;  # 设置 y 的标签为 +1
                    sub_prob.W[k] = W[si+k];  # 从 W 数组中复制 si+k 的值到 sub_prob 的 W 数组
                }
                # 遍历 cj 个样本，将它们赋值给 sub_prob 的 x 和 y 数组，并初始化权重 W
                for(k=0; k<cj; k++)
                {
                    sub_prob.x[ci+k] = x[sj+k];  # 从 x 数组中复制 sj+k 的值到 sub_prob 的 x 数组
                    sub_prob.y[ci+k] = -1;  # 设置 y 的标签为 -1
                    sub_prob.W[ci+k] = W[sj+k];  # 从 W 数组中复制 sj+k 的值到 sub_prob 的 W 数组
                }

                # 如果参数设定了概率估计，调用 svm_binary_svc_probability 函数进行概率估计
                if(param->probability)
                                    NAMESPACE::svm_binary_svc_probability(&sub_prob, param, weighted_C[i], weighted_C[j], probA[p], probB[p], status, blas_functions);

                # 使用训练函数训练模型，返回结果赋值给 f[p]
                f[p] = NAMESPACE::svm_train_one(&sub_prob, param, weighted_C[i], weighted_C[j], status, blas_functions);
                # 检查 ci 样本的 alpha 值，如果是非零且 alpha 的绝对值大于 0，则将 nonzero 标记为 true
                for(k=0; k<ci; k++)
                    if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
                        nonzero[si+k] = true;
                # 检查 cj 样本的 alpha 值，如果是非零且 alpha 的绝对值大于 0，则将 nonzero 标记为 true
                for(k=0; k<cj; k++)
                    if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
                        nonzero[sj+k] = true;
                # 释放 sub_prob 的 x 数组内存
                free(sub_prob.x);
                # 释放 sub_prob 的 y 数组内存
                free(sub_prob.y);
                # 释放 sub_prob 的 W 数组内存
                free(sub_prob.W);
                ++p;
            }

        // 构建输出模型

        # 设置模型的类别数
        model->nr_class = nr_class;

        # 分配内存给模型的标签数组，大小为 nr_class
        model->label = Malloc(int, nr_class);
        # 将 label 数组的内容复制到模型的标签数组中
        for(i=0; i<nr_class; i++)
            model->label[i] = label[i];

        # 分配内存给模型的 rho 数组，大小为 nr_class*(nr_class-1)/2
        model->rho = Malloc(double, nr_class*(nr_class-1)/2);
        # 分配内存给模型的迭代次数数组，大小为 nr_class*(nr_class-1)/2
        model->n_iter = Malloc(int, nr_class*(nr_class-1)/2);
        # 将 f 数组的 rho 和 n_iter 值复制到模型的 rho 和 n_iter 数组中
        for(i=0; i<nr_class*(nr_class-1)/2; i++)
        {
            model->rho[i] = f[i].rho;
            model->n_iter[i] = f[i].n_iter;
        }

        # 如果参数设定了概率估计，分配内存给模型的概率数组 probA 和 probB
        if(param->probability)
        {
            model->probA = Malloc(double, nr_class*(nr_class-1)/2);
            model->probB = Malloc(double, nr_class*(nr_class-1)/2);
            # 将 probA 和 probB 数组的值复制到模型的 probA 和 probB 数组中
            for(i=0; i<nr_class*(nr_class-1)/2; i++)
            {
                model->probA[i] = probA[i];
                model->probB[i] = probB[i];
            }
        }
        else
        {
            # 如果没有概率估计，设置模型的 probA 和 probB 为 NULL
            model->probA = NULL;
            model->probB = NULL;
        }

        int total_sv = 0;
        # 分配内存给 nz_count 数组，大小为 nr_class
        int *nz_count = Malloc(int, nr_class);
        # 分配内存给模型的支持向量计数数组 nSV，大小为 nr_class
        model->nSV = Malloc(int, nr_class);
        for(i=0; i<nr_class; i++)
        {
            int nSV = 0;
            # 遍历每个类别的样本，计算非零支持向量的数量
            for(int j=0; j<count[i]; j++)
                if(nonzero[start[i]+j])
                {
                    ++nSV;
                    ++total_sv;
                }
            model->nSV[i] = nSV;
            nz_count[i] = nSV;
        }

                # 输出总的支持向量数量
                info("Total nSV = %d\n", total_sv);

        # 设置模型的总支持向量数量
        model->l = total_sv;
                # 分配内存给模型的支持向量索引数组 sv_ind，大小为 total_sv
                model->sv_ind = Malloc(int, total_sv);
#ifdef _DENSE_REP
        # 如果使用稠密表示，分配内存给模型的支持向量数组 SV，大小为 total_sv
        model->SV = Malloc(PREFIX(node), total_sv);
#else
        # 否则，使用稀疏表示，分配内存给模型的支持向量数组 SV，大小为 total_sv
        model->SV = Malloc(PREFIX(node) *, total_sv);
#endif
        p = 0;  // 初始化 p 为 0，用于追踪支持向量的索引
        for(i=0;i<l;i++) {  // 循环遍历每个样本
            if(nonzero[i]) {  // 如果第 i 个样本是非零的
                model->SV[p] = x[i];  // 将第 i 个样本的特征向量存入模型的支持向量中
                model->sv_ind[p] = perm[i];  // 将第 i 个样本在原始数据中的索引存入模型的支持向量索引中
                ++p;  // 更新支持向量的索引
            }
        }

        int *nz_start = Malloc(int,nr_class);  // 分配内存用于存储每个类别中非零元素的起始位置
        nz_start[0] = 0;  // 第一个类别的起始位置为 0
        for(i=1;i<nr_class;i++)
            nz_start[i] = nz_start[i-1]+nz_count[i-1];  // 计算每个类别的起始位置

        model->sv_coef = Malloc(double *,nr_class-1);  // 为每个类别间的分类器分配内存
        for(i=0;i<nr_class-1;i++)
            model->sv_coef[i] = Malloc(double,total_sv);  // 为每个类别间的分类器的系数分配内存

        p = 0;  // 重新初始化 p
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]

                int si = start[i];  // 类别 i 的起始位置
                int sj = start[j];  // 类别 j 的起始位置
                int ci = count[i];  // 类别 i 的样本数
                int cj = count[j];  // 类别 j 的样本数

                int q = nz_start[i];  // 类别 i 非零元素的起始位置
                int k;
                for(k=0;k<ci;k++)
                    if(nonzero[si+k])
                        model->sv_coef[j-1][q++] = f[p].alpha[k];  // 存储分类器 (i,j) 中类别 i 的系数
                q = nz_start[j];  // 类别 j 非零元素的起始位置
                for(k=0;k<cj;k++)
                    if(nonzero[sj+k])
                        model->sv_coef[i][q++] = f[p].alpha[ci+k];  // 存储分类器 (i,j) 中类别 j 的系数
                ++p;  // 更新分类器索引
            }

        free(label);  // 释放分配的标签内存
        free(probA);  // 释放分配的 probA 内存
        free(probB);  // 释放分配的 probB 内存
        free(count);  // 释放分配的每个类别的样本数内存
        free(perm);   // 释放分配的排列索引内存
        free(start);  // 释放分配的每个类别的起始位置内存
        free(W);      // 释放分配的权重内存
        free(x);      // 释放分配的样本特征向量内存
        free(weighted_C);  // 释放分配的加权 C 参数内存
        free(nonzero);  // 释放分配的非零元素标记内存
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            free(f[i].alpha);  // 释放分配的每个分类器的系数内存
        free(f);  // 释放分配的分类器结构数组内存
        free(nz_count);  // 释放分配的每个类别的非零元素数内存
        free(nz_start);  // 释放分配的每个类别的非零元素起始位置内存
    }
    free(newprob.x);  // 释放分配的新问题的样本特征向量内存
    free(newprob.y);  // 释放分配的新问题的标签内存
    free(newprob.W);  // 释放分配的新问题的权重内存
    return model;  // 返回训练好的模型
}

// Stratified cross validation
void PREFIX(cross_validation)(const PREFIX(problem) *prob, const svm_parameter *param, int nr_fold, double *target, BlasFunctions *blas_functions)
{
    int i;
    int *fold_start = Malloc(int,nr_fold+1);  // 分配内存用于存储每个折叠的起始位置
    int l = prob->l;  // 获取问题的样本数
    int *perm = Malloc(int,l);  // 分配内存用于存储排列索引
    int nr_class;  // 类别数
    if(param->random_seed >= 0)
    {
        set_seed(param->random_seed);  // 设置随机种子
    }

    // stratified cv may not give leave-one-out rate
    // Each class to l folds -> some folds may have zero elements
    if((param->svm_type == C_SVC ||
        param->svm_type == NU_SVC) && nr_fold < l)
    {
        // 定义指针变量，并初始化为空指针
        int *start = NULL;
        int *label = NULL;
        int *count = NULL;
    
        // 调用指定命名空间下的 svm_group_classes 函数，传入参数，获取类别数量、类别标签、起始位置、计数和排列数组
        NAMESPACE::svm_group_classes(prob, &nr_class, &label, &start, &count, perm);
    
        // 对排列数组进行随机打乱，并根据折叠方式分组数据
        int *fold_count = Malloc(int, nr_fold); // 分配内存，存储每个折叠中的数据数量
        int c;
        int *index = Malloc(int, l); // 分配内存，存储索引数组
        for (i = 0; i < l; i++)
            index[i] = perm[i]; // 复制排列数组到索引数组
    
        // 随机打乱每个类别中的数据
        for (c = 0; c < nr_class; c++)
            for (i = 0; i < count[c]; i++)
            {
                int j = i + bounded_rand_int(count[c] - i); // 生成随机数 j
                swap(index[start[c] + j], index[start[c] + i]); // 交换索引数组中的元素
            }
    
        // 计算每个折叠中的数据数量
        for (i = 0; i < nr_fold; i++)
        {
            fold_count[i] = 0;
            for (c = 0; c < nr_class; c++)
                fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c] / nr_fold;
        }
    
        // 初始化折叠的起始位置
        fold_start[0] = 0;
        for (i = 1; i <= nr_fold; i++)
            fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
    
        // 将数据按折叠顺序重新排列
        for (c = 0; c < nr_class; c++)
            for (i = 0; i < nr_fold; i++)
            {
                int begin = start[c] + i * count[c] / nr_fold;
                int end = start[c] + (i + 1) * count[c] / nr_fold;
                for (int j = begin; j < end; j++)
                {
                    perm[fold_start[i]] = index[j]; // 更新排列数组中的元素
                    fold_start[i]++;
                }
            }
    
        // 释放内存
        fold_start[0] = 0;
        for (i = 1; i <= nr_fold; i++)
            fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
        free(start);
        free(label);
        free(count);
        free(index);
        free(fold_count);
    }
    else
    {
        // 如果不需要分组，直接对排列数组进行随机打乱
        for (i = 0; i < l; i++)
            perm[i] = i; // 初始化排列数组
        for (i = 0; i < l; i++)
        {
            int j = i + bounded_rand_int(l - i); // 生成随机数 j
            swap(perm[i], perm[j]); // 交换排列数组中的元素
        }
    
        // 计算折叠的起始位置
        for (i = 0; i <= nr_fold; i++)
            fold_start[i] = i * l / nr_fold;
    }
    
    // 计算每个折叠的起始和结束位置，并处理子问题
    for (i = 0; i < nr_fold; i++)
    {
        int begin = fold_start[i];
        int end = fold_start[i + 1];
        int j, k;
        struct PREFIX(problem) subprob;
    
        subprob.l = l - (end - begin);
#ifdef _DENSE_REP
    // 如果定义了 _DENSE_REP 宏，则分配一个节点结构的数组
    subprob.x = Malloc(struct PREFIX(node),subprob.l);
#else
    // 如果未定义 _DENSE_REP 宏，则分配一个节点结构的指针数组
    subprob.x = Malloc(struct PREFIX(node)*,subprob.l);
#endif

// 分配一个长度为 subprob.l 的 double 类型数组给 subprob.y
subprob.y = Malloc(double,subprob.l);

// 分配一个长度为 subprob.l 的 double 类型数组给 subprob.W
subprob.W = Malloc(double,subprob.l);

// 初始化 k 为 0
k=0;
// 复制 perm 数组的前 begin 个元素到 subprob 结构中对应的字段
for(j=0;j<begin;j++)
{
    subprob.x[k] = prob->x[perm[j]];
    subprob.y[k] = prob->y[perm[j]];
    subprob.W[k] = prob->W[perm[j]];
    ++k;
}
// 复制 perm 数组的从 end 到 l-1 的元素到 subprob 结构中对应的字段
for(j=end;j<l;j++)
{
    subprob.x[k] = prob->x[perm[j]];
    subprob.y[k] = prob->y[perm[j]];
    subprob.W[k] = prob->W[perm[j]];
    ++k;
}

// 定义一个整型变量 dummy_status，并初始化为 0，用于忽略超时错误
int dummy_status = 0; // IGNORES TIMEOUT ERRORS

// 使用 subprob 数据训练一个模型，结果保存在 submodel 中
struct PREFIX(model) *submodel = PREFIX(train)(&subprob,param, &dummy_status, blas_functions);

// 如果需要概率估计，并且模型类型是 C_SVC 或 NU_SVC
if(param->probability &&
   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
{
    // 分配一个长度为 PREFIX(get_nr_class)(submodel) 的 double 数组给 prob_estimates
    double *prob_estimates=Malloc(double, PREFIX(get_nr_class)(submodel));
    // 对于 perm 数组中从 begin 到 end-1 的元素，预测其类别的概率，并将结果保存到 target 数组中
    for(j=begin;j<end;j++)
#ifdef _DENSE_REP
        target[perm[j]] = PREFIX(predict_probability)(submodel,(prob->x + perm[j]),prob_estimates, blas_functions);
#else
        target[perm[j]] = PREFIX(predict_probability)(submodel,prob->x[perm[j]],prob_estimates, blas_functions);
#endif
    // 释放 prob_estimates 数组的内存空间
    free(prob_estimates);
}
else
    // 对于 perm 数组中从 begin 到 end-1 的元素，预测其类别，并将结果保存到 target 数组中
    for(j=begin;j<end;j++)
#ifdef _DENSE_REP
        target[perm[j]] = PREFIX(predict)(submodel,prob->x+perm[j],blas_functions);
#else
        target[perm[j]] = PREFIX(predict)(submodel,prob->x[perm[j]],blas_functions);
#endif

// 释放并销毁 submodel 模型的内存空间
PREFIX(free_and_destroy_model)(&submodel);

// 释放 subprob 结构中 x、y 和 W 字段的内存空间
free(subprob.x);
free(subprob.y);
free(subprob.W);
}

// 释放 fold_start 数组的内存空间
free(fold_start);

// 释放 perm 数组的内存空间
free(perm);
}
#ifdef _DENSE_REP
                    // 如果定义了 _DENSE_REP，使用稠密表示计算支持向量机的决策函数
                    sum += sv_coef[i] * NAMESPACE::Kernel::k_function(x,model->SV+i,model->param,blas_functions);
#else
                // 否则，使用普通表示计算支持向量机的决策函数
                sum += sv_coef[i] * NAMESPACE::Kernel::k_function(x,model->SV[i],model->param,blas_functions);
#endif
        // 减去模型的 rho 值，得到最终的决策函数值
        sum -= model->rho[0];
        // 将计算出的决策函数值存储到 dec_values 中
        *dec_values = sum;

        // 根据 SVM 类型返回预测结果
        if(model->param.svm_type == ONE_CLASS)
            return (sum>0)?1:-1;
        else
            return sum;
    }
    else
    {
        // 多类分类时的预测过程
        int nr_class = model->nr_class;
        int l = model->l;

        // 分配内存以存储每个样本点到决策函数值的映射
        double *kvalue = Malloc(double,l);
        // 计算样本点 x 与支持向量的核函数值
        for(i=0;i<l;i++)
#ifdef _DENSE_REP
                    kvalue[i] = NAMESPACE::Kernel::k_function(x,model->SV+i,model->param,blas_functions);
#else
                kvalue[i] = NAMESPACE::Kernel::k_function(x,model->SV[i],model->param,blas_functions);
#endif

        // 计算每个类别的起始点
        int *start = Malloc(int,nr_class);
        start[0] = 0;
        for(i=1;i<nr_class;i++)
            start[i] = start[i-1]+model->nSV[i-1];

        // 初始化每个类别的投票数
        int *vote = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            vote[i] = 0;

        // 开始投票过程
        int p=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];

                // 获取当前类别间的支持向量系数
                int k;
                double *coef1 = model->sv_coef[j-1];
                double *coef2 = model->sv_coef[i];
                // 计算当前类别间的决策函数值
                for(k=0;k<ci;k++)
                    sum += coef1[si+k] * kvalue[si+k];
                for(k=0;k<cj;k++)
                    sum += coef2[sj+k] * kvalue[sj+k];
                sum -= model->rho[p];
                dec_values[p] = sum;

                // 根据决策函数值决定投票
                if(dec_values[p] > 0)
                    ++vote[i];
                else
                    ++vote[j];
                p++;
            }

        // 选择票数最多的类别作为最终预测结果
        int vote_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;

        // 释放内存并返回预测标签
        free(kvalue);
        free(start);
        free(vote);
        return model->label[vote_max_idx];
    }
}

// 预测单个样本点 x 的类别，并返回决策函数值
double PREFIX(predict)(const PREFIX(model) *model, const PREFIX(node) *x, BlasFunctions *blas_functions)
{
    int nr_class = model->nr_class;
    double *dec_values;

    // 分配内存以存储决策函数值
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
        dec_values = Malloc(double, 1);
    else
        dec_values = Malloc(double, nr_class*(nr_class-1)/2);

    // 调用 predict_values 函数计算预测结果并得到决策函数值
    double pred_result = PREFIX(predict_values)(model, x, dec_values, blas_functions);

    // 释放内存并返回预测结果
    free(dec_values);
    return pred_result;
}

// 预测单个样本点 x 的类别，并计算类别概率估计值
double PREFIX(predict_probability)(
    const PREFIX(model) *model, const PREFIX(node) *x, double *prob_estimates, BlasFunctions *blas_functions)
{
    // 如果是 C_SVC 或 NU_SVC，并且模型中有概率估计参数
    if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
        model->probA!=NULL && model->probB!=NULL)
    {
        // 定义整型变量 i
        int i;
        // 获取模型的类别数
        int nr_class = model->nr_class;
        // 分配内存以存储决策值的数组
        double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
        // 使用模型进行预测，将决策值存储在 dec_values 中
        PREFIX(predict_values)(model, x, dec_values, blas_functions);
    
        // 设置最小概率值
        double min_prob=1e-7;
        // 分配内存以存储两两类别之间的概率
        double **pairwise_prob = Malloc(double *, nr_class);
        // 为每对类别分配内存
        for(i = 0; i < nr_class; i++)
            pairwise_prob[i] = Malloc(double, nr_class);
    
        // 计数变量 k
        int k = 0;
        // 遍历类别对，计算每对的概率
        for(i = 0; i < nr_class; i++)
            for(int j = i + 1; j < nr_class; j++)
            {
                // 计算两类别之间的概率，并将其限制在 min_prob 和 1-min_prob 之间
                pairwise_prob[i][j] = min(max(NAMESPACE::sigmoid_predict(dec_values[k], model->probA[k], model->probB[k]), min_prob), 1 - min_prob);
                // 根据对称性设置另一半的概率
                pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
                // 更新决策值索引
                k++;
            }
    
        // 计算多类别问题的概率估计
        NAMESPACE::multiclass_probability(nr_class, pairwise_prob, prob_estimates);
    
        // 初始化概率最大值的索引
        int prob_max_idx = 0;
        // 寻找概率估计中最大值的索引
        for(i = 1; i < nr_class; i++)
            if(prob_estimates[i] > prob_estimates[prob_max_idx])
                prob_max_idx = i;
    
        // 释放分配的内存：每对类别的概率数组
        for(i = 0; i < nr_class; i++)
            free(pairwise_prob[i]);
        // 释放分配的内存：决策值数组
        free(dec_values);
        // 释放分配的内存：两两类别概率数组的指针
        free(pairwise_prob);
    
        // 返回具有最大概率估计的类别标签
        return model->label[prob_max_idx];
    }
    else
        // 如果不是多类别情况，则调用单类别预测函数
        return PREFIX(predict)(model, x, blas_functions);
    }
}

// 释放模型内容的函数，包括支持向量和相关数据
void PREFIX(free_model_content)(PREFIX(model)* model_ptr)
{
    // 如果需要释放支持向量并且支持向量的数量大于0且支持向量数组不为空
    if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
#ifdef _DENSE_REP
        // 如果是密集表示，释放每个支持向量的值数组
        for (int i = 0; i < model_ptr->l; i++)
            free(model_ptr->SV[i].values);
#else
        // 如果是其他表示，释放整体支持向量数组
        free((void *)(model_ptr->SV[0]));
#endif

    // 释放支持向量系数数组
    if(model_ptr->sv_coef)
    {
        for(int i=0;i<model_ptr->nr_class-1;i++)
            free(model_ptr->sv_coef[i]);
    }

    // 释放支持向量数组并将指针设为NULL
    free(model_ptr->SV);
    model_ptr->SV = NULL;

    // 释放支持向量系数数组并将指针设为NULL
    free(model_ptr->sv_coef);
    model_ptr->sv_coef = NULL;

    // 释放支持向量索引数组并将指针设为NULL
    free(model_ptr->sv_ind);
    model_ptr->sv_ind = NULL;

    // 释放rho数组并将指针设为NULL
    free(model_ptr->rho);
    model_ptr->rho = NULL;

    // 释放标签数组并将指针设为NULL
    free(model_ptr->label);
    model_ptr->label= NULL;

    // 释放probA数组并将指针设为NULL
    free(model_ptr->probA);
    model_ptr->probA = NULL;

    // 释放probB数组并将指针设为NULL
    free(model_ptr->probB);
    model_ptr->probB= NULL;

    // 释放nSV数组并将指针设为NULL
    free(model_ptr->nSV);
    model_ptr->nSV = NULL;

    // 释放n_iter并将指针设为NULL
    free(model_ptr->n_iter);
    model_ptr->n_iter = NULL;
}

// 释放并销毁模型的函数
void PREFIX(free_and_destroy_model)(PREFIX(model)** model_ptr_ptr)
{
    // 如果模型指针不为空且模型对象不为空
    if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
    {
        // 调用释放模型内容的函数
        PREFIX(free_model_content)(*model_ptr_ptr);
        // 释放模型对象内存并将指针设为NULL
        free(*model_ptr_ptr);
        *model_ptr_ptr = NULL;
    }
}

// 销毁参数对象的函数
void PREFIX(destroy_param)(svm_parameter* param)
{
    // 释放权重标签数组并将指针设为NULL
    free(param->weight_label);
    // 释放权重数组并将指针设为NULL
    free(param->weight);
}

// 检查参数的函数，返回参数问题的描述或NULL（无问题）
const char *PREFIX(check_parameter)(const PREFIX(problem) *prob, const svm_parameter *param)
{
    // 检查svm_type参数的合法性
    int svm_type = param->svm_type;
    if(svm_type != C_SVC &&
       svm_type != NU_SVC &&
       svm_type != ONE_CLASS &&
       svm_type != EPSILON_SVR &&
       svm_type != NU_SVR)
        return "unknown svm type";

    // 检查kernel_type参数的合法性
    int kernel_type = param->kernel_type;
    if(kernel_type != LINEAR &&
       kernel_type != POLY &&
       kernel_type != RBF &&
       kernel_type != SIGMOID &&
       kernel_type != PRECOMPUTED)
        return "unknown kernel type";

    // 检查gamma参数是否非负
    if(param->gamma < 0)
        return "gamma < 0";

    // 检查多项式核函数的次数是否非负
    if(param->degree < 0)
        return "degree of polynomial kernel < 0";

    // 检查cache_size参数是否大于0
    if(param->cache_size <= 0)
        return "cache_size <= 0";

    // 检查eps参数是否大于0
    if(param->eps <= 0)
        return "eps <= 0";

    // 根据svm_type参数检查C参数的合法性
    if(svm_type == C_SVC ||
       svm_type == EPSILON_SVR ||
       svm_type == NU_SVR)
        if(param->C <= 0)
            return "C <= 0";

    // 根据svm_type参数检查nu参数的合法性
    if(svm_type == NU_SVC ||
       svm_type == ONE_CLASS ||
       svm_type == NU_SVR)
        if(param->nu <= 0 || param->nu > 1)
            return "nu <= 0 or nu > 1";

    // 根据svm_type参数检查p参数的合法性
    if(svm_type == EPSILON_SVR)
        if(param->p < 0)
            return "p < 0";

    // 检查shrinking参数是否为0或1
    if(param->shrinking != 0 &&
       param->shrinking != 1)
        return "shrinking != 0 and shrinking != 1";

    // 检查probability参数是否为0或1
    if(param->probability != 0 &&
       param->probability != 1)
        return "probability != 0 and probability != 1";

    // 如果probability为1且svm_type为ONE_CLASS，则不支持一类SVM的概率输出
    if(param->probability == 1 &&
       svm_type == ONE_CLASS)
        return "one-class SVM probability output not supported yet";

    // 参数检查通过，返回NULL
    // 检查是否使用了 NU_SVC 类型的支持向量机模型
    if(svm_type == NU_SVC)
    {
        int l = prob->l;  // 获取样本数目
        int max_nr_class = 16;  // 最大类别数
        int nr_class = 0;  // 实际类别数目
        int *label = Malloc(int,max_nr_class);  // 分配类别数组内存
        double *count = Malloc(double,max_nr_class);  // 分配类别权重数组内存

        int i;
        for(i=0;i<l;i++)
        {
            int this_label = (int)prob->y[i];  // 获取当前样本的类别标签
            int j;
            for(j=0;j<nr_class;j++)
                if(this_label == label[j])
                {
                    count[j] += prob->W[i];  // 累加同类别的权重
                    break;
                }
            if(j == nr_class)
            {
                if(nr_class == max_nr_class)
                {
                    max_nr_class *= 2;
                    label = (int *)realloc(label,max_nr_class*sizeof(int));  // 扩展类别数组
                    count = (double *)realloc(count,max_nr_class*sizeof(double));  // 扩展权重数组

                }
                label[nr_class] = this_label;  // 记录新类别
                count[nr_class] = prob->W[i];  // 记录新类别的权重
                ++nr_class;  // 类别数目加一
            }
        }

        for(i=0;i<nr_class;i++)
        {
            double n1 = count[i];
            for(int j=i+1;j<nr_class;j++)
            {
                double n2 = count[j];
                // 检查是否给定的 nu 值导致某些类别的权重不可行
                if(param->nu*(n1+n2)/2 > min(n1,n2))
                {
                    free(label);
                    free(count);
                    return "specified nu is infeasible";  // 返回指定的 nu 值不可行的错误信息
                }
            }
        }
        free(label);  // 释放类别数组内存
        free(count);  // 释放权重数组内存
    }

    // 检查其他类型的支持向量机模型是否需要调整样本集
    if(svm_type == C_SVC ||
       svm_type == EPSILON_SVR ||
       svm_type == NU_SVR ||
       svm_type == ONE_CLASS)
    {
        PREFIX(problem) newprob;
        // 过滤掉权重为零或负数的样本
        remove_zero_weight(&newprob, prob);

        // 如果所有样本都被移除
        if(newprob.l == 0) {
            free(newprob.x);
            free(newprob.y);
            free(newprob.W);
            return "Invalid input - all samples have zero or negative weights.";  // 返回所有样本权重为零或负数的错误信息
        }
        // 如果新样本集的大小与原始样本集不同，并且是 C_SVC 类型的模型
        else if(prob->l != newprob.l &&
                svm_type == C_SVC)
        {
            bool only_one_label = true;
            int first_label = newprob.y[0];
            for(int i=1;i<newprob.l;i++)
            {
                if(newprob.y[i] != first_label)
                {
                    only_one_label = false;
                    break;
                }
            }
            // 如果所有具有正权重的样本属于同一个类别
            if(only_one_label) {
                free(newprob.x);
                free(newprob.y);
                free(newprob.W);
                return "Invalid input - all samples with positive weights belong to the same class.";  // 返回所有正权重样本属于同一类别的错误信息
            }
        }

        free(newprob.x);  // 释放新样本集特征数组内存
        free(newprob.y);  // 释放新样本集标签数组内存
        free(newprob.W);  // 释放新样本集权重数组内存
    }
    return NULL;  // 返回空指针，表示处理成功
}

void PREFIX(set_print_string_function)(void (*print_func)(const char *))
{
    // 检查传入的打印函数是否为 NULL
    if(print_func == NULL)
        // 如果为 NULL，将默认的打印函数指针指向标准输出打印函数
        svm_print_string = &print_string_stdout;
    else
        // 如果不为 NULL，则直接使用传入的打印函数
        svm_print_string = print_func;
}
```