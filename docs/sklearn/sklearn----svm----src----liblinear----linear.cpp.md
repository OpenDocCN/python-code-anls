# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\liblinear\linear.cpp`

```
/*
   Modified 2011:

   - Make labels sorted in group_classes, Dan Yamins.

   Modified 2012:

   - Changes roles of +1 and -1 to match scikit API, Andreas Mueller
        See issue 546: https://github.com/scikit-learn/scikit-learn/pull/546
   - Also changed roles for pairwise class weights, Andreas Mueller
        See issue 1491: https://github.com/scikit-learn/scikit-learn/pull/1491

   Modified 2014:

   - Remove the hard-coded value of max_iter (1000), that allows max_iter
     to be passed as a parameter from the classes LogisticRegression and
     LinearSVC, Manoj Kumar
   - Added function get_n_iter that exposes the number of iterations.
        See issue 3499: https://github.com/scikit-learn/scikit-learn/issues/3499
        See pull 3501: https://github.com/scikit-learn/scikit-learn/pull/3501

   Modified 2015:
   - Patched liblinear for sample_weights - Manoj Kumar
     See https://github.com/scikit-learn/scikit-learn/pull/5274

   Modified 2020:
   - Improved random number generator by using a mersenne twister + tweaked
     lemire postprocessor. This fixed a convergence issue on windows targets.
     Sylvain Marie, Schneider Electric
     See <https://github.com/scikit-learn/scikit-learn/pull/13511#issuecomment-481729756>

 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
#include <climits>
#include <random>
#include "../newrand/newrand.h"

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
    fputs(s,stdout);
    fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
/*
   定义一个函数info，该函数类似于printf，用于将格式化的字符串输出到标准输出。
   如果编译时宏为1，则函数info可用；如果为0，则函数info为空函数。
 */
static void info(const char *fmt,...)
{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf,fmt,ap);
    va_end(ap);
    (*liblinear_print_string)(buf);
}
#else
/*
   如果编译时宏为0，则定义info为空函数。
 */
static void info(const char *fmt,...) {}
#endif

class l2r_lr_fun: public function
{
public:
    /*
       l2r_lr_fun类的构造函数，接受问题prob和参数C。
       在构造函数中，初始化了z和D数组，这些数组与问题的规模相关。
     */
    l2r_lr_fun(const problem *prob, double *C);
    /*
       l2r_lr_fun类的析构函数，用于释放动态分配的内存。
     */
    ~l2r_lr_fun();

    /*
       计算目标函数的值，接受当前权重向量w作为参数。
     */
    double fun(double *w);
    /*
       计算目标函数关于权重向量w的梯度，将结果存储在g中。
     */
    void grad(double *w, double *g);
    /*
       计算Hessian矩阵与向量s的乘积，将结果存储在Hs中。
     */
    void Hv(double *s, double *Hs);

    /*
       返回问题中变量的数量。
     */
    int get_nr_variable(void);

private:
    /*
       计算矩阵X与向量v的乘积，将结果存储在Xv中。
     */
    void Xv(double *v, double *Xv);
    /*
       计算矩阵X的转置与向量v的乘积，将结果存储在XTv中。
     */
    void XTv(double *v, double *XTv);

    double *C;           // 正则化参数
    double *z;           // 临时向量
    double *D;           // 临时向量
    const problem *prob; // 指向问题的指针
};

/*
   l2r_lr_fun类的构造函数实现。
   在构造函数中，分配了z和D数组的内存，并初始化了指向问题的指针prob。
 */
l2r_lr_fun::l2r_lr_fun(const problem *prob, double *C)
{
    int l=prob->l;

    this->prob = prob;

    z = new double[l];
    D = new double[l];
    this->C = C;
}
# 析构函数，释放对象的资源，删除动态分配的内存数组
l2r_lr_fun::~l2r_lr_fun()
{
    delete[] z;  // 删除 z 数组
    delete[] D;  // 删除 D 数组
}

# 计算逻辑回归的目标函数值
double l2r_lr_fun::fun(double *w)
{
    int i;
    double f=0;
    double *y=prob->y;  // 获取训练样本的标签
    int l=prob->l;      // 获取训练样本的数量
    int w_size=get_nr_variable();  // 获取模型的变量个数

    Xv(w, z);  // 计算 X*w，结果存储在 z 中

    # 计算正则化项
    for(i=0;i<w_size;i++)
        f += w[i]*w[i];
    f /= 2.0;

    # 计算损失项
    for(i=0;i<l;i++)
    {
        double yz = y[i]*z[i];
        if (yz >= 0)
            f += C[i]*log(1 + exp(-yz));
        else
            f += C[i]*(-yz+log(1 + exp(yz)));
    }

    return(f);  // 返回目标函数值
}

# 计算逻辑回归的梯度
void l2r_lr_fun::grad(double *w, double *g)
{
    int i;
    double *y=prob->y;  // 获取训练样本的标签
    int l=prob->l;      // 获取训练样本的数量
    int w_size=get_nr_variable();  // 获取模型的变量个数

    # 计算 z 和 D
    for(i=0;i<l;i++)
    {
        z[i] = 1/(1 + exp(-y[i]*z[i]));  // 计算 z[i]
        D[i] = z[i]*(1-z[i]);            // 计算 D[i]
        z[i] = C[i]*(z[i]-1)*y[i];       // 修改 z[i]
    }

    XTv(z, g);  // 计算 X^T*z，结果存储在 g 中

    # 计算梯度
    for(i=0;i<w_size;i++)
        g[i] = w[i] + g[i];  // g[i] = w[i] + X^T*z[i]
}

# 获取变量个数
int l2r_lr_fun::get_nr_variable(void)
{
    return prob->n;  // 返回问题的变量个数
}

# 计算 Hessian 矩阵乘以向量的结果
void l2r_lr_fun::Hv(double *s, double *Hs)
{
    int i;
    int l=prob->l;      // 获取训练样本的数量
    int w_size=get_nr_variable();  // 获取模型的变量个数
    double *wa = new double[l];    // 分配临时数组

    Xv(s, wa);  // 计算 X*s，结果存储在 wa 中

    # 计算 Hessian 矩阵与向量的乘积
    for(i=0;i<l;i++)
        wa[i] = C[i]*D[i]*wa[i];

    XTv(wa, Hs);  // 计算 X^T*wa，结果存储在 Hs 中

    # 更新 Hs
    for(i=0;i<w_size;i++)
        Hs[i] = s[i] + Hs[i];  // Hs[i] = s[i] + X^T*wa[i]

    delete[] wa;  // 释放临时数组的内存
}

# 计算 X*v，结果存储在 Xv 中
void l2r_lr_fun::Xv(double *v, double *Xv)
{
    int i;
    int l=prob->l;            // 获取训练样本的数量
    feature_node **x=prob->x;  // 获取训练样本的特征节点数组

    for(i=0;i<l;i++)
    {
        feature_node *s=x[i];  // 获取第 i 个样本的特征节点数组
        Xv[i]=0;
        while(s->index!=-1)
        {
            Xv[i]+=v[s->index-1]*s->value;  // 计算 X*v 的第 i 行
            s++;
        }
    }
}

# 计算 X^T*v，结果存储在 XTv 中
void l2r_lr_fun::XTv(double *v, double *XTv)
{
    int i;
    int l=prob->l;            // 获取训练样本的数量
    int w_size=get_nr_variable();  // 获取模型的变量个数
    feature_node **x=prob->x;  // 获取训练样本的特征节点数组

    # 初始化 XTv
    for(i=0;i<w_size;i++)
        XTv[i]=0;

    # 计算 X^T*v
    for(i=0;i<l;i++)
    {
        feature_node *s=x[i];  // 获取第 i 个样本的特征节点数组
        while(s->index!=-1)
        {
            XTv[s->index-1]+=v[i]*s->value;  // 计算 X^T*v 的第 s->index-1 行
            s++;
        }
    }
}

# 构造函数，初始化 l2r_l2_svc_fun 类的对象
l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
    int l=prob->l;  // 获取训练样本的数量

    this->prob = prob;  // 初始化 prob 成员变量

    z = new double[l];  // 分配 z 数组的内存
    D = new double[l];  // 分配 D 数组的内存
    I = new int[l];     // 分配 I 数组的内存
    this->C = C;        // 初始化 C 成员变量
}

# 析构函数，释放 l2r_l2_svc_fun 类的对象资源，删除动态分配的内存数组
l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
    delete[] z;  // 删除 z 数组
    delete[] D;  // 删除 D 数组
    delete[] I;  // 删除 I 数组
}

# 计算 l2r_l2_svc_fun 类的目标函数值
double l2r_l2_svc_fun::fun(double *w)
{
    int i;
    double f=0;
    double *y=prob->y;  // 获取训练样本的标签
    int l=prob->l;      // 获取训练样本的数量
    int w_size=get_nr_variable();  // 获取模型的变量个数

    Xv(w, z);  // 计算 X*w，结果存储在 z 中

    # 计算正则化项
    for(i=0;i<w_size;i++)
        f += w[i]*w[i];
    f /= 2.0;

    # 计算损失项
    for(i=0;i<l;i++)
    {
        z[i] = y[i]*z[i];
        double d = 1-z[i];
        if (d > 0)
            f += C[i]*d*d;
    }

    return(f);  // 返回目标函数值
}
void l2r_l2_svc_fun::grad(double *w, double *g)
{
    int i;
    double *y=prob->y;  // 指向问题实例prob中的标签数据
    int l=prob->l;  // 获取问题实例prob中的样本数量
    int w_size=get_nr_variable();  // 获取变量的维度大小

    sizeI = 0;  // 初始化索引集合的大小为0
    for (i=0;i<l;i++)  // 遍历每个样本
        if (z[i] < 1)  // 如果z[i]小于1
        {
            z[sizeI] = C[i]*y[i]*(z[i]-1);  // 计算并存储z[i]
            I[sizeI] = i;  // 记录索引i
            sizeI++;  // 索引集合大小加一
        }
    subXTv(z, g);  // 调用subXTv函数，计算z和g的内积

    for(i=0;i<w_size;i++)  // 遍历变量的维度
        g[i] = w[i] + 2*g[i];  // 计算梯度g
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
    return prob->n;  // 返回问题实例prob中的变量数量
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
    int i;
    int w_size=get_nr_variable();  // 获取变量的维度大小
    double *wa = new double[sizeI];  // 创建大小为sizeI的新数组wa

    subXv(s, wa);  // 调用subXv函数，计算s和wa的内积
    for(i=0;i<sizeI;i++)  // 遍历索引集合
        wa[i] = C[I[i]]*wa[i];  // 计算wa[i]
    
    subXTv(wa, Hs);  // 调用subXTv函数，计算wa和Hs的内积
    for(i=0;i<w_size;i++)  // 遍历变量的维度
        Hs[i] = s[i] + 2*Hs[i];  // 计算Hs[i]
    delete[] wa;  // 释放wa数组的内存空间
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
    int i;
    int l=prob->l;  // 获取问题实例prob中的样本数量
    feature_node **x=prob->x;  // 指向问题实例prob中的特征节点数据

    for(i=0;i<l;i++)  // 遍历每个样本
    {
        feature_node *s=x[i];  // 指向当前样本的特征节点
        Xv[i]=0;  // 初始化Xv[i]为0
        while(s->index!=-1)  // 循环直到特征节点的索引为-1
        {
            Xv[i]+=v[s->index-1]*s->value;  // 计算Xv[i]
            s++;  // 移动到下一个特征节点
        }
    }
}

void l2r_l2_svc_fun::subXv(double *v, double *Xv)
{
    int i;
    feature_node **x=prob->x;  // 指向问题实例prob中的特征节点数据

    for(i=0;i<sizeI;i++)  // 遍历索引集合
    {
        feature_node *s=x[I[i]];  // 指向当前索引对应的特征节点
        Xv[i]=0;  // 初始化Xv[i]为0
        while(s->index!=-1)  // 循环直到特征节点的索引为-1
        {
            Xv[i]+=v[s->index-1]*s->value;  // 计算Xv[i]
            s++;  // 移动到下一个特征节点
        }
    }
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
    int i;
    int w_size=get_nr_variable();  // 获取变量的维度大小
    feature_node **x=prob->x;  // 指向问题实例prob中的特征节点数据

    for(i=0;i<w_size;i++)  // 遍历变量的维度
        XTv[i]=0;  // 初始化XTv[i]为0
    for(i=0;i<sizeI;i++)  // 遍历索引集合
    {
        feature_node *s=x[I[i]];  // 指向当前索引对应的特征节点
        while(s->index!=-1)  // 循环直到特征节点的索引为-1
        {
            XTv[s->index-1]+=v[i]*s->value;  // 计算XTv
            s++;  // 移动到下一个特征节点
        }
    }
}

class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
public:
    l2r_l2_svr_fun(const problem *prob, double *C, double p);

    double fun(double *w);
    void grad(double *w, double *g);

private:
    double p;  // 存储参数p
};

l2r_l2_svr_fun::l2r_l2_svr_fun(const problem *prob, double *C, double p):
    l2r_l2_svc_fun(prob, C)  // 调用基类构造函数初始化
{
    this->p = p;  // 存储参数p
}

double l2r_l2_svr_fun::fun(double *w)
{
    int i;
    double f=0;  // 初始化目标函数值为0
    double *y=prob->y;  // 指向问题实例prob中的标签数据
    int l=prob->l;  // 获取问题实例prob中的样本数量
    int w_size=get_nr_variable();  // 获取变量的维度大小
    double d;

    Xv(w, z);  // 计算Xv

    for(i=0;i<w_size;i++)  // 遍历变量的维度
        f += w[i]*w[i];  // 计算目标函数的第一部分
    f /= 2;  // 目标函数的第一部分除以2
    for(i=0;i<l;i++)  // 遍历每个样本
    {
        d = z[i] - y[i];  // 计算z[i]和y[i]的差值
        if(d < -p)  // 如果差值小于-p
            f += C[i]*(d+p)*(d+p);  // 更新目标函数值
        else if(d > p)  // 如果差值大于p
            f += C[i]*(d-p)*(d-p);  // 更新目标函数值
    }

    return(f);  // 返回目标函数值
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
    int i;
    double *y=prob->y;  // 指向问题实例prob中的标签数据
    int l=prob->l;  // 获取问题实例prob中的样本数量
    int w_size=get_nr_variable();  // 获取变量的维度大小
    double d;

    sizeI = 0;  // 初始化索引集合的大小为0
    for(i=0;i<l;i++)  // 遍历每个样本
    {
        d = z[i] - y[i];  // 计算z[i]和y[i]的差值

        // generate index set I
        if(d < -p)  // 如果差值小于-p
        {
            z[sizeI] = C[i]*(d+p);  // 计算并存储z[i]
            I[sizeI] = i;  // 记录索引i
            sizeI++;  // 索引集合大小加一
        }
        else if(d > p)  // 如果差值大于p
        {
            z[sizeI] = C[i]*(d-p);  // 计算并存储z[i]
            I[sizeI] = i;  // 记录索引i
            sizeI++;  // 索引集合大小加一
        }
    }
    subXTv(z, g);  // 调用subXTv函数，计算z和g的内积
}
    # 遍历索引从 0 到 w_size 的范围
    for(i=0;i<w_size;i++)
        # 将 g 数组的第 i 个元素更新为 w 数组的第 i 个元素加上 2 倍的 g 数组的第 i 个元素的值
        g[i] = w[i] + 2*g[i];
}

// A coordinate descent algorithm for
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
//
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i,
//  C^m_i = 0 if m != y_i,
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i
//
// Given:
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

class Solver_MCSVM_CS
{
    public:
        // Constructor for Solver_MCSVM_CS class
        Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
        
        // Destructor for Solver_MCSVM_CS class
        ~Solver_MCSVM_CS();
        
        // Solve function to solve the SVM problem
        int Solve(double *w);
        
    private:
        // Helper function to solve a sub-problem
        void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
        
        // Helper function to determine if a variable should be shrunk
        bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
        
        // Array to store B values
        double *B;
        
        // Array to store C values
        double *C;
        
        // Array to store G values
        double *G;
        
        // Size of the weight vector
        int w_size;
        
        // Number of instances
        int l;
        
        // Number of classes
        int nr_class;
        
        // Maximum number of iterations
        int max_iter;
        
        // Stopping tolerance
        double eps;
        
        // Pointer to the problem structure
        const problem *prob;
};

// Constructor for Solver_MCSVM_CS class
Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *prob, int nr_class, double *weighted_C, double eps, int max_iter)
{
    // Initialize class variables with parameters
    this->w_size = prob->n;
    this->l = prob->l;
    this->nr_class = nr_class;
    this->eps = eps;
    this->max_iter = max_iter;
    this->prob = prob;
    
    // Allocate memory for B, G, and C arrays
    this->B = new double[nr_class];
    this->G = new double[nr_class];
    this->C = new double[prob->l];
    
    // Compute weighted C values
    for(int i = 0; i < prob->l; i++)
        this->C[i] = prob->W[i] * weighted_C[(int)prob->y[i]];
}

// Destructor for Solver_MCSVM_CS class
Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
    // Free allocated memory for B, G, and C arrays
    delete[] B;
    delete[] G;
    delete[] C;
}

// Function to compare two doubles for qsort
int compare_double(const void *a, const void *b)
{
    if(*(double *)a > *(double *)b)
        return -1;
    if(*(double *)a < *(double *)b)
        return 1;
    return 0;
}

// Function to solve a sub-problem of the SVM
void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
    int r;
    double *D;

    // Clone B array into D array
    clone(D, B, active_i);

    // Update D[yi] if yi < active_i
    if(yi < active_i)
        D[yi] += A_i * C_yi;

    // Sort D in descending order
    qsort(D, active_i, sizeof(double), compare_double);

    // Calculate beta based on sorted D
    double beta = D[0] - A_i * C_yi;
    for(r = 1; r < active_i && beta < r * D[r]; r++)
        beta += D[r];
    beta /= r;

    // Update alpha_new based on beta and B
    for(r = 0; r < active_i; r++)
    {
        if(r == yi)
            alpha_new[r] = min(C_yi, (beta - B[r]) / A_i);
        else
            alpha_new[r] = min((double)0, (beta - B[r]) / A_i);
    }

    // Free allocated memory for D array
    delete[] D;
}

// Function to determine if a variable should be shrunk
bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
    double bound = 0;

    // Determine bound based on m and yi
    if(m == yi)
        bound = C[GETI(i)];

    // Return true if alpha_i equals bound and G[m] is less than minG
    if(alpha_i == bound && G[m] < minG)
        return true;

    // Otherwise, return false
    return false;
}

// Function to solve the SVM problem
int Solver_MCSVM_CS::Solve(double *w)
{
    // Initialize variables
    int i, m, s;
    int iter = 0;
    double *alpha = new double[l * nr_class];
    double *alpha_new = new double[nr_class];
    int *index = new int[l];
    double *QD = new double[l];
    // 分配一个长度为 nr_class 的整型数组，用于存储 d_ind
    int *d_ind = new int[nr_class];
    // 分配一个长度为 nr_class 的双精度浮点数数组，用于存储 d_val
    double *d_val = new double[nr_class];
    // 分配一个长度为 nr_class*l 的整型数组，用于存储 alpha_index
    int *alpha_index = new int[nr_class*l];
    // 分配一个长度为 l 的整型数组，用于存储 y_index
    int *y_index = new int[l];
    // 将变量 l 的值赋给 active_size
    int active_size = l;
    // 分配一个长度为 l 的整型数组，用于存储 active_size_i，并将 active_size 赋给每个元素
    int *active_size_i = new int[l];
    // 计算 eps_shrink，设置为 max(10.0*eps, 1.0)，作为停止收缩的容差
    double eps_shrink = max(10.0*eps, 1.0);
    // 设置 start_from_all 为 true

    // 初始化 alpha 数组，将其所有元素设置为 0
    for(i=0;i<l*nr_class;i++)
        alpha[i] = 0;

    // 初始化 w 数组，将其所有元素设置为 0
    for(i=0;i<w_size*nr_class;i++)
        w[i] = 0;
    
    // 遍历每个样本
    for(i=0;i<l;i++)
    {
        // 对每个类别 m，为 alpha_index[i*nr_class+m] 赋值 m
        for(m=0;m<nr_class;m++)
            alpha_index[i*nr_class+m] = m;
        // 获取第 i 个样本的特征向量 xi
        feature_node *xi = prob->x[i];
        // 初始化 QD[i] 为 0
        QD[i] = 0;
        // 遍历该样本的每个特征
        while(xi->index != -1)
        {
            double val = xi->value;
            // 计算 QD[i] 的值，累加 val 的平方
            QD[i] += val*val;

            // 如果初始化 alpha 不是零，取消下面的 for 循环的注释来初始化 w
            // for(m=0; m<nr_class; m++)
            //    w[(xi->index-1)*nr_class+m] += alpha[i*nr_class+m]*val;
            xi++;
        }
        // 将 nr_class 赋给 active_size_i[i]
        active_size_i[i] = nr_class;
        // 将 prob->y[i] 转换为整数后赋给 y_index[i]
        y_index[i] = (int)prob->y[i];
        // 将 i 赋给 index[i]
        index[i] = i;
    }

    // 迭代更新过程
    while(iter < max_iter)
    {
        // 迭代代码...
    }

    // 输出优化完成信息和迭代次数
    info("\noptimization finished, #iter = %d\n",iter);
    // 如果达到最大迭代次数，输出警告信息
    if (iter >= max_iter)
        info("\nWARNING: reaching max number of iterations\n");

    // 计算目标函数值 v
    double v = 0;
    // 计算支持向量的数量 nSV
    int nSV = 0;
    // 计算 v 的第一部分，遍历 w 数组
    for(i=0;i<w_size*nr_class;i++)
        v += w[i]*w[i];
    v = 0.5*v;
    // 计算 v 的第二部分，遍历 alpha 数组
    for(i=0;i<l*nr_class;i++)
    {
        v += alpha[i];
        if(fabs(alpha[i]) > 0)
            nSV++;
    }
    // 计算 v 的第三部分，遍历每个样本
    for(i=0;i<l;i++)
        v -= alpha[i*nr_class+(int)prob->y[i]];
    // 输出目标函数值 v 和支持向量数量 nSV
    info("Objective value = %lf\n",v);
    info("nSV = %d\n",nSV);

    // 释放动态分配的内存
    delete [] alpha;
    delete [] alpha_new;
    delete [] index;
    delete [] QD;
    delete [] d_ind;
    delete [] d_val;
    delete [] alpha_index;
    delete [] y_index;
    delete [] active_size_i;
    // 返回迭代次数 iter
    return iter;
// }

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//         upper_bound_i = Cp if y_i = 1
//         upper_bound_i = Cn if y_i = -1
//         D_ii = 0
// In L2-SVM case:
//         upper_bound_i = INF
//         D_ii = 1/(2*Cp)    if y_i = 1
//         D_ii = 1/(2*Cn)    if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

static int solve_l2r_l1l2_svc(
    const problem *prob, double *w, double eps,
    double Cp, double Cn, int solver_type, int max_iter)
{
    int l = prob->l;
    int w_size = prob->n;
    int i, s, iter = 0;
    double C, d, G;
    double *QD = new double[l];
    int *index = new int[l];
    double *alpha = new double[l];
    schar *y = new schar[l];
    int active_size = l;

    // PG: projected gradient, for shrinking and stopping
    double PG;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double PGmax_new, PGmin_new;

    // default solver_type: L2R_L2LOSS_SVC_DUAL
    double *diag = new double[l];
    double *upper_bound = new double[l];
    double *C_ = new double[l];
    for(i=0; i<l; i++)
    {
        if(prob->y[i]>0)
            C_[i] = prob->W[i] * Cp;
        else
            C_[i] = prob->W[i] * Cn;
        diag[i] = 0.5/C_[i];
        upper_bound[i] = INF;
    }
    if(solver_type == L2R_L1LOSS_SVC_DUAL)
    {
        for(i=0; i<l; i++)
        {
            diag[i] = 0;
            upper_bound[i] = C_[i];
        }
    }

    for(i=0; i<l; i++)
    {
        if(prob->y[i] > 0)
        {
            y[i] = +1;
        }
        else
        {
            y[i] = -1;
        }
    }

    // Initial alpha can be set here. Note that
    // 0 <= alpha[i] <= upper_bound[GETI(i)]
    for(i=0; i<l; i++)
        alpha[i] = 0;

    for(i=0; i<w_size; i++)
        w[i] = 0;
    for(i=0; i<l; i++)
    {
        QD[i] = diag[GETI(i)];

        feature_node *xi = prob->x[i];
        while (xi->index != -1)
        {
            double val = xi->value;
            QD[i] += val*val;
            w[xi->index-1] += y[i]*alpha[i]*val;
            xi++;
        }
        index[i] = i;
    }

    while (iter < max_iter)
    {
        // 初始化 PGmax_new 为负无穷大，用于存储最大的 PG 值
        PGmax_new = -INF;
        // 初始化 PGmin_new 为正无穷大，用于存储最小的 PG 值
        PGmin_new = INF;
    
        // 随机打乱 index 数组中的元素顺序
        for (i=0; i<active_size; i++)
        {
            int j = i+bounded_rand_int(active_size-i);
            swap(index[i], index[j]);
        }
    
        // 遍历每个样本
        for (s=0; s<active_size; s++)
        {
            i = index[s];
            // 计算当前样本的梯度 G
            G = 0;
            // 获取样本 i 的类别标签 yi
            schar yi = y[i];
    
            // 获取样本 i 的特征节点 xi
            feature_node *xi = prob->x[i];
            // 计算 G，累加所有特征的权重 w 乘以特征值 xi->value
            while(xi->index!= -1)
            {
                G += w[xi->index-1]*(xi->value);
                xi++;
            }
            G = G*yi-1;
    
            // 获取样本 i 的上界 C
            C = upper_bound[GETI(i)];
            // 更新 G，加上 alpha[i] 乘以 diag[GETI(i)]
            G += alpha[i]*diag[GETI(i)];
    
            // 初始化 PG 为 0
            PG = 0;
            // 根据 alpha[i] 的值进行判断
            if (alpha[i] == 0)
            {
                // 如果 alpha[i] == 0，并且 G 大于旧的 PGmax_old，则删除当前样本
                if (G > PGmax_old)
                {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                // 否则，如果 G 小于 0，则设置 PG 为 G
                else if (G < 0)
                    PG = G;
            }
            else if (alpha[i] == C)
            {
                // 如果 alpha[i] == C，并且 G 小于旧的 PGmin_old，则删除当前样本
                if (G < PGmin_old)
                {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                // 否则，如果 G 大于 0，则设置 PG 为 G
                else if (G > 0)
                    PG = G;
            }
            else
                // 如果 alpha[i] 不等于 0 或者 C，则直接设置 PG 为 G
                PG = G;
    
            // 更新 PGmax_new 和 PGmin_new
            PGmax_new = max(PGmax_new, PG);
            PGmin_new = min(PGmin_new, PG);
    
            // 如果 PG 的绝对值大于 1.0e-12，则进行参数更新
            if(fabs(PG) > 1.0e-12)
            {
                // 记录旧的 alpha[i]
                double alpha_old = alpha[i];
                // 更新 alpha[i]
                alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
                // 计算 d
                d = (alpha[i] - alpha_old)*yi;
                // 更新 w 向量
                xi = prob->x[i];
                while (xi->index != -1)
                {
                    w[xi->index-1] += d*xi->value;
                    xi++;
                }
            }
        }
    
        // 更新迭代次数
        iter++;
        // 每 10 次迭代输出一个点表示进展
        if(iter % 10 == 0)
            info(".");
    
        // 如果 PGmax_new 和 PGmin_new 的差值小于等于 eps，则结束优化
        if(PGmax_new - PGmin_new <= eps)
        {
            // 如果 active_size 等于 l，则结束优化
            if(active_size == l)
                break;
            // 否则，重置 active_size，输出一个星号，重置 PGmax_old 和 PGmin_old
            else
            {
                active_size = l;
                info("*");
                PGmax_old = INF;
                PGmin_old = -INF;
                continue;
            }
        }
        // 更新 PGmax_old 和 PGmin_old
        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        // 如果 PGmax_old 小于等于 0，则将其设为正无穷大
        if (PGmax_old <= 0)
            PGmax_old = INF;
        // 如果 PGmin_old 大于等于 0，则将其设为负无穷大
        if (PGmin_old >= 0)
            PGmin_old = -INF;
    }
    
    // 输出优化完成信息和迭代次数
    info("\noptimization finished, #iter = %d\n",iter);
    // 如果迭代次数达到最大值，则输出警告信息
    if (iter >= max_iter)
        info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");
    
    // 计算目标函数值
    double v = 0;
    // 统计支持向量的个数
    int nSV = 0;
    // 计算 ||w||^2
    for(i=0; i<w_size; i++)
        v += w[i]*w[i];
    // 计算目标函数值 v
    for(i=0; i<l; i++)
    {
        v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
        if(alpha[i] > 0)
            ++nSV;
    }
    // 输出目标函数值和支持向量个数
    info("Objective value = %lf\n",v/2);
    info("nSV = %d\n",nSV);
    
    // 释放动态分配的内存
    delete [] QD;
    delete [] alpha;
    delete [] y;
    // 释放动态分配的整型数组 index 所占内存
    delete [] index;
    // 释放动态分配的浮点型数组 diag 所占内存
    delete [] diag;
    // 释放动态分配的浮点型数组 upper_bound 所占内存
    delete [] upper_bound;
    // 释放动态分配的双精度浮点型数组 C_ 所占内存
    delete [] C_;
    // 返回函数迭代器 iter
    return iter;
// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//         upper_bound_i = C
//         lambda_i = 0
// In L2-SVM case:
//         upper_bound_i = INF
//         lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#undef GETI
#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

// Function definition for solving the L2-loss epsilon-SVR dual problem using coordinate descent
static int solve_l2r_l1l2_svr(
    const problem *prob, double *w, const parameter *param,
    int solver_type, int max_iter)
{
    int l = prob->l;                // Number of instances
    double C = param->C;            // Regularization parameter C
    double p = param->p;            // Epsilon in SVR
    int w_size = prob->n;           // Size of the feature vector
    double eps = param->eps;        // Stopping tolerance
    int i, s, iter = 0;             // Loop variables and iteration count
    int active_size = l;            // Size of active set
    int *index = new int[l];        // Array to store instance indices

    double d, G, H;                 // Temporary variables
    double Gmax_old = INF;          // Maximum gradient magnitude in previous iteration
    double Gmax_new, Gnorm1_new;    // New values of maximum gradient magnitude and L1-norm
    double Gnorm1_init = -1.0;      // Initial value of L1-norm (to be initialized)
    double *beta = new double[l];   // Coefficient vector beta
    double *QD = new double[l];     // Diagonal of the Q matrix
    double *y = prob->y;            // Array of target values

    // L2R_L2LOSS_SVR_DUAL specific variables
    double *lambda = new double[l];         // Array for lambda values
    double *upper_bound = new double[l];    // Array for upper bound values
    double *C_ = new double[l];             // Weighted C values

    // Initialize lambda, upper_bound, and C_ based on solver_type
    for (i=0; i<l; i++)
    {
        C_[i] = prob->W[i] * C;
        lambda[i] = 0.5 / C_[i];
        upper_bound[i] = INF;
    }
    if (solver_type == L2R_L1LOSS_SVR_DUAL)
    {
        for (i=0; i<l; i++)
        {
            lambda[i] = 0;
            upper_bound[i] = C_[i];
        }
    }

    // Initialize beta to zero
    for (i=0; i<l; i++)
        beta[i] = 0;

    // Initialize w to zero
    for (i=0; i<w_size; i++)
        w[i] = 0;

    // Compute QD and initialize w using beta
    for (i=0; i<l; i++)
    {
        QD[i] = 0;
        feature_node *xi = prob->x[i];
        while (xi->index != -1)
        {
            double val = xi->value;
            QD[i] += val * val;
            w[xi->index - 1] += beta[i] * val;
            xi++;
        }
        index[i] = i;
    }

    // Main loop of the coordinate descent algorithm
    while (iter < max_iter)
    {
        // Iteration-specific calculations

        iter++;
    }

    // Output optimization summary
    info("\noptimization finished, #iter = %d\n", iter);
    if (iter >= max_iter)
        info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

    // Calculate objective value and count support vectors
    double v = 0;
    int nSV = 0;
    for (i=0; i<w_size; i++)
        v += w[i] * w[i];
    v = 0.5 * v;
    for (i=0; i<l; i++)
    {
        v += p * fabs(beta[i]) - y[i] * beta[i] + 0.5 * lambda[GETI(i)] * beta[i] * beta[i];
        if (beta[i] != 0)
            nSV++;
    }

    // Output objective value and number of support vectors
    info("Objective value = %lf\n", v);
    info("nSV = %d\n", nSV);

    // Clean up allocated memory
    delete [] beta;
    delete [] QD;
    delete [] index;
    delete [] lambda;
    delete [] upper_bound;
    delete [] C_;

    // Return number of iterations performed
    return iter;
}
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (i)
// To support weights for instances, use GETI(i) (i)

// 定义函数 solve_l2r_lr_dual，用于解决 L2 正则化逻辑回归的对偶问题
int solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn,
                       int max_iter)
{
    // l 是问题中实例的数量，w_size 是特征数
    int l = prob->l;
    int w_size = prob->n;
    int i, s, iter = 0;

    // xTx 存储每个实例的特征向量的平方和
    double *xTx = new double[l];
    // index 用于存储实例的索引
    int *index = new int[l];
    // alpha 存储 alpha 和 C - alpha 的值
    double *alpha = new double[2*l]; // store alpha and C - alpha
    // y 是标签向量
    schar *y = new schar[l];
    // max_inner_iter 是内部牛顿法的最大迭代次数
    int max_inner_iter = 100; // for inner Newton
    // innereps 是内部迭代的停止容许度
    double innereps = 1e-2;
    // innereps_min 是内部迭代的最小停止容许度
    double innereps_min = min(1e-8, eps);
    // upper_bound 存储每个实例的上界
    double *upper_bound = new double[l];

    // 根据标签 y_i 的正负性设置每个实例的上界和对应的标签
    for(i=0; i<l; i++)
    {
        if(prob->y[i] > 0)
        {
            upper_bound[i] = prob->W[i] * Cp;
            y[i] = +1;
        }
        else
        {
            upper_bound[i] = prob->W[i] * Cn;
            y[i] = -1;
        }
    }

    // 初始化 alpha
    for(i=0; i<l; i++)
    {
        alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
        alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
    }

    // 初始化 w 向量
    for(i=0; i<w_size; i++)
        w[i] = 0;

    // 计算 xTx 和部分 w 向量的初始化
    for(i=0; i<l; i++)
    {
        xTx[i] = 0;
        feature_node *xi = prob->x[i];
        while (xi->index != -1)
        {
            double val = xi->value;
            xTx[i] += val*val;
            w[xi->index-1] += y[i]*alpha[2*i]*val;
            xi++;
        }
        index[i] = i;
    }

    // 主循环，执行最大迭代次数 max_iter
    while (iter < max_iter)
    {
        // 随机重排索引数组，使用 Fisher-Yates 算法
        for (i=0; i<l; i++)
        {
            // 生成介于 i 和 l 之间的随机整数 j，并交换 index[i] 和 index[j] 的值
            int j = i + bounded_rand_int(l-i);
            swap(index[i], index[j]);
        }
        // 初始化牛顿法迭代次数和最大梯度 Gmax
        int newton_iter = 0;
        double Gmax = 0;
        // 遍历所有样本
        for (s=0; s<l; s++)
        {
            // 获取当前样本索引
            i = index[s];
            // 获取当前样本标签 y[i]
            schar yi = y[i];
            // 获取当前样本的上界 C
            double C = upper_bound[GETI(i)];
            // 计算 y_i * w^T * x_i 和 x_i^T * x_i
            double ywTx = 0, xisq = xTx[i];
            // 获取当前样本的特征向量
            feature_node *xi = prob->x[i];
            // 计算 y_i * w^T * x_i
            while (xi->index != -1)
            {
                ywTx += w[xi->index-1]*xi->value;
                xi++;
            }
            ywTx *= y[i];
            // 计算子问题的两个参数 a 和 b
            double a = xisq, b = ywTx;

            // 决定是最小化 g_1(z) 还是 g_2(z)
            int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
            if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
            {
                ind1 = 2*i+1;
                ind2 = 2*i;
                sign = -1;
            }

            // 计算 g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
            double alpha_old = alpha[ind1];
            double z = alpha_old;
            // 根据 z 是否接近上界 C 调整 z 的值
            if(C - z < 0.5 * C)
                z = 0.1*z;
            // 计算 g_t(z) 的梯度 gp
            double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
            // 更新最大梯度 Gmax
            Gmax = max(Gmax, fabs(gp));

            // 子问题的牛顿法迭代过程
            const double eta = 0.1; // xi in the paper
            int inner_iter = 0;
            while (inner_iter <= max_inner_iter)
            {
                // 如果梯度 gp 足够小则退出迭代
                if(fabs(gp) < innereps)
                    break;
                // 计算 g_t(z) 的二阶导数 gpp，并更新 z
                double gpp = a + C/(C-z)/z;
                double tmpz = z - gp/gpp;
                if(tmpz <= 0)
                    z *= eta;
                else // tmpz 在 (0, C) 内
                    z = tmpz;
                // 更新 gp
                gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
                // 更新牛顿法迭代次数
                newton_iter++;
                inner_iter++;
            }

            // 如果进行了内部迭代，则更新 w
            if(inner_iter > 0)
            {
                alpha[ind1] = z;
                alpha[ind2] = C-z;
                xi = prob->x[i];
                while (xi->index != -1)
                {
                    w[xi->index-1] += sign*(z-alpha_old)*yi*xi->value;
                    xi++;
                }
            }
        }

        // 更新总迭代次数并输出进度信息
        iter++;
        if(iter % 10 == 0)
            info(".");

        // 如果最大梯度 Gmax 小于设定的阈值 eps，则结束优化
        if(Gmax < eps)
            break;

        // 根据牛顿法迭代次数调整内部迭代精度
        if(newton_iter <= l/10)
            innereps = max(innereps_min, 0.1*innereps);

    }

    // 输出优化结束信息和总迭代次数
    info("\noptimization finished, #iter = %d\n",iter);
    // 如果达到最大迭代次数则输出警告信息
    if (iter >= max_iter)
        info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

    // 计算目标函数值
    double v = 0;
    for(i=0; i<w_size; i++)
        v += w[i] * w[i];
    v *= 0.5;
    for(i=0; i<l; i++)
        v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
            - upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
    // 输出目标函数值
    info("Objective value = %lf\n", v);

    // 释放内存
    delete [] xTx;
    delete [] alpha;
    delete [] y;
    delete [] index;
}
    释放动态分配的 upper_bound 数组内存
    delete [] upper_bound;
    返回 iter 变量的值
    return iter;
}

// 结束了 solve_l1r_l2_svc 函数的定义

// 定义了宏 GETI，用于获取索引 i 的值
#undef GETI
#define GETI(i) (i)
// 如果需要支持实例权重，可以使用 GETI(i) (i)

// 实现了 L1-正则化 L2-损失支持向量分类的坐标下降算法
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// 输入:
// prob_col - 包含问题数据的结构体指针
// w - 存放解的数组
// eps - 停止迭代的容差
// Cp, Cn - 正负样本的惩罚参数
// max_iter - 最大迭代次数
//
// 输出:
// 返回迭代次数 iter
//
// 参考文献:
// Yuan et al. (2010) 和 LIBLINEAR 论文附录, Fan et al. (2008)
static int solve_l1r_l2_svc(
    problem *prob_col, double *w, double eps,
    double Cp, double Cn, int max_iter)
{
    int l = prob_col->l;        // 样本数
    int w_size = prob_col->n;   // 特征维度数
    int j, s, iter = 0;         // 迭代计数器和临时变量
    int active_size = w_size;   // 活跃集的大小
    int max_num_linesearch = 20;// 最大线性搜索次数

    double sigma = 0.01;        // 步长参数
    double d, G_loss, G, H;     // 临时变量
    double Gmax_old = INF;      // 旧的最大梯度
    double Gmax_new, Gnorm1_new;// 新的最大梯度和 L1 范数
    double Gnorm1_init = -1.0;  // 初始的 L1 范数，在第一次迭代时初始化
    double d_old, d_diff;       // 临时变量
    double loss_old, loss_new;  // 损失函数的旧值和新值
    double appxcond, cond;      // 近似条件数和条件数

    int *index = new int[w_size];   // 特征索引数组
    schar *y = new schar[l];        // 标签数组
    double *b = new double[l];      // b = 1 - yw^Txi
    double *xj_sq = new double[w_size]; // 存放 xj^2 的数组
    feature_node *x;                // 特征节点指针

    double *C = new double[l];      // 惩罚参数数组

    // 可在此处设置初始 w
    for(j=0; j<w_size; j++)
        w[j] = 0;

    // 初始化 b, y, C
    for(j=0; j<l; j++)
    {
        b[j] = 1;
        if(prob_col->y[j] > 0)
        {
            y[j] = 1;
            C[j] = prob_col->W[j] * Cp;
        }
        else
        {
            y[j] = -1;
            C[j] = prob_col->W[j] * Cn;
        }
    }

    // 初始化 index, xj_sq
    for(j=0; j<w_size; j++)
    {
        index[j] = j;
        xj_sq[j] = 0;
        x = prob_col->x[j];
        while(x->index != -1)
        {
            int ind = x->index-1;
            x->value *= y[ind]; // x->value 存储 yi*xij
            double val = x->value;
            b[ind] -= w[j]*val;
            xj_sq[j] += C[GETI(ind)]*val*val;
            x++;
        }
    }

    // 开始迭代
    while(iter < max_iter)
    {
        // 在这里执行迭代步骤...

        iter++;
    }

    // 输出优化结果
    info("\noptimization finished, #iter = %d\n", iter);
    if(iter >= max_iter)
        info("\nWARNING: reaching max number of iterations\n");

    // 计算目标函数值
    double v = 0;
    int nnz = 0;
    for(j=0; j<w_size; j++)
    {
        x = prob_col->x[j];
        while(x->index != -1)
        {
            x->value *= prob_col->y[x->index-1]; // 恢复 x->value
            x++;
        }
        if(w[j] != 0)
        {
            v += fabs(w[j]);
            nnz++;
        }
    }
    for(j=0; j<l; j++)
        if(b[j] > 0)
            v += C[GETI(j)]*b[j]*b[j];

    // 输出目标函数值和稀疏性
    info("Objective value = %lf\n", v);
    info("#nonzeros/#features = %d/%d\n", nnz, w_size);

    // 释放内存
    delete [] index;
    delete [] y;
    delete [] b;
    delete [] xj_sq;
    delete [] C;
    return iter;
}

// 结束了 solve_l1r_l2_svc 函数的定义

// 定义了坐标下降算法用于 L1-正则化的逻辑回归问题
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// 输入:
// x, y, Cp, Cn
// eps - 停止迭代的容差
//
// solution will be put in w
//
// 此处省略了具体的实现代码
// 定义宏GETI(i)，返回参数i本身
#undef GETI
#define GETI(i) (i)
// 支持实例权重，使用GETI(i) (i)

// 解决 L1 正则化逻辑回归问题的函数
static int solve_l1r_lr(
    const problem *prob_col, double *w, double eps,
    double Cp, double Cn, int max_newton_iter)
{
    int l = prob_col->l;                    // 获取问题中样本数目
    int w_size = prob_col->n;               // 获取问题中特征数目
    int j, s, newton_iter=0, iter=0;        // 初始化迭代器和计数器
    int max_iter = 1000;                    // 最大迭代次数
    int max_num_linesearch = 20;            // 最大线搜索次数
    int active_size;                        // 活跃集的大小
    int QP_active_size;                     // QP问题的活跃集大小
    int QP_no_change = 0;                   // QP问题的无变化次数计数器

    double nu = 1e-12;                      // 公差值
    double inner_eps = 1;                   // 内部公差
    double sigma = 0.01;                    // 梯度步长
    double w_norm, w_norm_new;              // 当前和更新后的w的L1范数
    double z, G, H;                         // 临时变量
    double Gnorm1_init = -1.0;              // Gnorm1_init在第一次迭代时初始化
    double Gmax_old = INF;                  // 旧的最大梯度
    double Gmax_new, Gnorm1_new;            // 新的最大梯度和L1范数
    double QP_Gmax_old = INF;               // QP问题的旧的最大梯度
    double QP_Gmax_new, QP_Gnorm1_new;      // QP问题的新的最大梯度和L1范数
    double delta, negsum_xTd, cond;         // 临时变量

    int *index = new int[w_size];           // 特征索引数组
    schar *y = new schar[l];                // 标签数组
    double *Hdiag = new double[w_size];     // 对角线Hessian矩阵数组
    double *Grad = new double[w_size];      // 梯度数组
    double *wpd = new double[w_size];       // w的临时数组
    double *xjneg_sum = new double[w_size]; // 负样本特征值之和数组
    double *xTd = new double[l];            // 特征值与梯度乘积之和数组
    double *exp_wTx = new double[l];        // exp(w^T * x)数组
    double *exp_wTx_new = new double[l];    // 更新后的exp(w^T * x)数组
    double *tau = new double[l];            // 辅助变量数组
    double *D = new double[l];              // 对角线Hessian矩阵的对应值数组
    feature_node *x;                        // 特征节点

    double *C = new double[l];              // 惩罚系数数组

    // 初始化w向量
    for(j=0; j<w_size; j++)
        w[j] = 0;

    // 初始化y数组和C数组
    for(j=0; j<l; j++)
    {
        if(prob_col->y[j] > 0)
        {
            y[j] = 1;
            C[j] = prob_col->W[j] * Cp;
        }
        else
        {
            y[j] = -1;
            C[j] = prob_col->W[j] * Cn;
        }

        exp_wTx[j] = 0;
    }

    // 计算初始化时的w的L1范数和相关数组
    w_norm = 0;
    for(j=0; j<w_size; j++)
    {
        w_norm += fabs(w[j]);
        wpd[j] = w[j];
        index[j] = j;
        xjneg_sum[j] = 0;
        x = prob_col->x[j];
        while(x->index != -1)
        {
            int ind = x->index-1;
            double val = x->value;
            exp_wTx[ind] += w[j]*val;
            if(y[ind] == -1)
                xjneg_sum[j] += C[GETI(ind)]*val;
            x++;
        }
    }

    // 计算exp(w^T * x)和相关数组
    for(j=0; j<l; j++)
    {
        exp_wTx[j] = exp(exp_wTx[j]);
        double tau_tmp = 1/(1+exp_wTx[j]);
        tau[j] = C[GETI(j)]*tau_tmp;
        D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
    }

    // 主要的Newton迭代循环
    while(newton_iter < max_newton_iter)
    {
        // 迭代主体部分未提供，可以在此添加代码
    }

    // 输出优化结果信息
    info("=========================\n");
    info("optimization finished, #iter = %d\n", newton_iter);
    if(newton_iter >= max_newton_iter)
        info("WARNING: reaching max number of iterations\n");

    // 计算目标函数值
    double v = 0;
    int nnz = 0;
    for(j=0; j<w_size; j++)
        if(w[j] != 0)
        {
            v += fabs(w[j]);
            nnz++;
        }
    for(j=0; j<l; j++)
        if(y[j] == 1)
            v += C[GETI(j)]*log(1+1/exp_wTx[j]);
        else
            v += C[GETI(j)]*log(1+exp_wTx[j]);

    info("Objective value = %lf\n", v);
}
    # 打印非零数目和特征数目的比例
    info("#nonzeros/#features = %d/%d\n", nnz, w_size);

    # 释放动态分配的索引数组内存
    delete [] index;
    # 释放动态分配的标签数组内存
    delete [] y;
    # 释放动态分配的对角线Hessian矩阵数组内存
    delete [] Hdiag;
    # 释放动态分配的梯度数组内存
    delete [] Grad;
    # 释放动态分配的w*p*d数组内存
    delete [] wpd;
    # 释放动态分配的xjneg_sum数组内存
    delete [] xjneg_sum;
    # 释放动态分配的xTd数组内存
    delete [] xTd;
    # 释放动态分配的exp_wTx数组内存
    delete [] exp_wTx;
    # 释放动态分配的exp_wTx_new数组内存
    delete [] exp_wTx_new;
    # 释放动态分配的tau数组内存
    delete [] tau;
    # 释放动态分配的D数组内存
    delete [] D;
    # 释放动态分配的C数组内存
    delete [] C;
    # 返回牛顿法迭代的结果
    return newton_iter;
// 转置矩阵 X，从行格式到列格式
static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
    int i;
    int l = prob->l;  // 获取问题的样本数
    int n = prob->n;  // 获取问题的特征数
    size_t nnz = 0;   // 非零元素的数量
    size_t *col_ptr = new size_t [n+1];  // 列指针数组，用于记录每列的起始位置
    feature_node *x_space;  // 存储转置后的特征节点数组
    prob_col->l = l;  // 设置转置后问题的样本数
    prob_col->n = n;  // 设置转置后问题的特征数
    prob_col->y = new double[l];  // 分配内存存储转置后的标签
    prob_col->x = new feature_node*[n];  // 分配内存存储转置后的特征节点指针数组
    prob_col->W = new double[l];  // 分配内存存储样本权重

    // 复制原问题的标签和权重
    for(i=0; i<l; i++)
    {
        prob_col->y[i] = prob->y[i];  // 复制标签
        prob_col->W[i] = prob->W[i];  // 复制权重
    }

    // 初始化列指针数组为0
    for(i=0; i<n+1; i++)
        col_ptr[i] = 0;

    // 计算每列非零元素的数量并更新列指针数组
    for(i=0; i<l; i++)
    {
        feature_node *x = prob->x[i];  // 获取原问题的第i个样本的特征节点数组
        while(x->index != -1)  // 遍历该样本的特征节点直到遇到结尾标志
        {
            nnz++;  // 非零元素数量加一
            col_ptr[x->index]++;  // 更新对应列的非零元素计数
            x++;  // 移动到下一个特征节点
        }
    }

    // 根据每列非零元素的数量更新列指针数组，转换为每列的起始位置
    for(i=1; i<n+1; i++)
        col_ptr[i] += col_ptr[i-1] + 1;

    // 分配内存存储转置后的特征节点数据
    x_space = new feature_node[nnz+n];

    // 填充转置后问题的特征节点指针数组
    for(i=0; i<n; i++)
        prob_col->x[i] = &x_space[col_ptr[i]];

    // 将原问题的特征数据转置到新的特征节点数组中
    for(i=0; i<l; i++)
    {
        feature_node *x = prob->x[i];  // 获取原问题的第i个样本的特征节点数组
        while(x->index != -1)  // 遍历该样本的特征节点直到遇到结尾标志
        {
            int ind = x->index-1;  // 计算特征索引
            x_space[col_ptr[ind]].index = i+1;  // 将特征节点索引存储为样本索引（从1开始）
            x_space[col_ptr[ind]].value = x->value;  // 存储特征节点的值
            col_ptr[ind]++;  // 更新列指针数组
            x++;  // 移动到下一个特征节点
        }
    }

    // 结束每列的特征节点数据存储
    for(i=0; i<n; i++)
        x_space[col_ptr[i]].index = -1;  // 标记每列结束

    *x_space_ret = x_space;  // 返回转置后的特征节点数组

    delete [] col_ptr;  // 释放列指针数组的内存
}
    /* START MOD: Sort labels and apply to array count --dyamins */

    int j;
    // 遍历每个类别标签，进行排序
    for (j = 1; j < nr_class; j++)
    {
        // 初始化变量i为当前j的前一个索引
        i = j - 1;
        // 当前类别标签及其对应的计数
        int this_label = label[j];
        int this_count = count[j];
        // 插入排序：将当前标签和计数与已排序的部分进行比较并插入正确的位置
        while (i >= 0 && label[i] > this_label)
        {
            label[i + 1] = label[i];
            count[i + 1] = count[i];
            i--;
        }
        label[i + 1] = this_label;
        count[i + 1] = this_count;
    }

    // 将数据标签映射到排序后的标签数组
    for (i = 0; i < l; i++)
    {
        j = 0;
        // 获取当前样本的标签，并在排序后的标签数组中查找对应的索引
        int this_label = (int)prob->y[i];
        while (this_label != label[j])
        {
            j++;
        }
        data_label[i] = j;
    }

    /* END MOD */
//
// Labels are ordered by their first occurrence in the training set.
// However, for two-class sets with -1/+1 labels and -1 appears first,
// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
//
if (nr_class == 2 && label[0] == -1 && label[1] == 1)
{
    // Swap the labels and corresponding counts
    swap(label[0], label[1]);
    swap(count[0], count[1]);
    // Adjust data labels to match the swapped class order
    for (i = 0; i < l; i++)
    {
        if (data_label[i] == 0)
            data_label[i] = 1;
        else
            data_label[i] = 0;
    }
}

// Allocate memory for the start indices of each class in the permutation array
int *start = Malloc(int, nr_class);
start[0] = 0;
for (i = 1; i < nr_class; i++)
    start[i] = start[i - 1] + count[i - 1];

// Populate the permutation array based on class start indices
for (i = 0; i < l; i++)
{
    perm[start[data_label[i]]] = i;
    ++start[data_label[i]];
}

// Reset start indices for reuse in other computations
start[0] = 0;
for (i = 1; i < nr_class; i++)
    start[i] = start[i - 1] + count[i - 1];

// Assign values to return pointers
*nr_class_ret = nr_class;
*label_ret = label;
*start_ret = start;
*count_ret = count;

// Free allocated memory for data_label
free(data_label);
}

static int train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn, BlasFunctions *blas_functions)
{
    double eps = param->eps;
    int max_iter = param->max_iter;
    int pos = 0;
    int neg = 0;
    int n_iter = -1;

    // Count positive and negative instances
    for (int i = 0; i < prob->l; i++)
        if (prob->y[i] > 0)
            pos++;
    neg = prob->l - pos;

    // Calculate tolerance for the solver
    double primal_solver_tol = eps * max(min(pos, neg), 1) / prob->l;

    function *fun_obj = NULL;
    switch (param->solver_type)
    {
        // Case handling for different solver types (missing implementation).
    }
    return n_iter;
}

//
// Remove zero weighed data as libsvm and some liblinear solvers require C > 0.
//
static void remove_zero_weight(problem *newprob, const problem *prob)
{
    int i;
    int l = 0;

    // Count non-zero weighted instances
    for (i = 0; i < prob->l; i++)
        if (prob->W[i] > 0)
            l++;

    // Allocate memory for reduced problem instance
    *newprob = *prob;
    newprob->l = l;
    newprob->x = Malloc(feature_node *, l);
    newprob->y = Malloc(double, l);
    newprob->W = Malloc(double, l);

    int j = 0;
    // Copy non-zero weighted instances to the new problem
    for (i = 0; i < prob->l; i++)
        if (prob->W[i] > 0)
        {
            newprob->x[j] = prob->x[i];
            newprob->y[j] = prob->y[i];
            newprob->W[j] = prob->W[i];
            j++;
        }
}

//
// Interface functions
//
model *train(const problem *prob, const parameter *param, BlasFunctions *blas_functions)
{
    problem newprob;

    // Remove instances with zero weight
    remove_zero_weight(&newprob, prob);
    prob = &newprob;

    int i, j;
    int l = prob->l;
    int n = prob->n;
    int w_size = prob->n;
    model *model_ = Malloc(model, 1);

    // Determine number of features in the model
    if (prob->bias >= 0)
        model_->nr_feature = n - 1;
    else
        model_->nr_feature = n;

    // Initialize model parameters
    model_->param = *param;
    model_->bias = prob->bias;

    // Check if it's a regression model; if so, train with regression-specific method
    if (check_regression_model(model_))
    {
        model_->w = Malloc(double, w_size);
        model_->n_iter = Malloc(int, 1);
        model_->nr_class = 2;
        model_->label = NULL;
        model_->n_iter[0] = train_one(prob, param, &model_->w[0], 0, 0, blas_functions);
    }
    else
    {
        // Handling for non-regression models (missing implementation).
    }
    return model_;
}
// 执行交叉验证的函数，用于评估模型性能
void cross_validation(const problem *prob, const parameter *param, int nr_fold, double *target)
{
    int i;
    int *fold_start;  // 每个折叠（fold）的起始索引数组
    int l = prob->l;  // 数据集中样本的数量
    int *perm = Malloc(int,l);  // 分配存储样本索引的内存空间

    // 如果折叠数大于样本数，发出警告并将折叠数设置为样本数（使用留一法交叉验证）
    if (nr_fold > l)
    {
        nr_fold = l;
        fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
    }
    
    // 分配存储折叠起始索引的内存空间
    fold_start = Malloc(int,nr_fold+1);

    // 随机排列样本索引
    for(i=0;i<l;i++)
        perm[i]=i;

    // 对样本索引进行随机重排
    for(i=0;i<l;i++)
    {
        int j = i+bounded_rand_int(l-i);
        swap(perm[i],perm[j]);
    }

    // 计算每个折叠的起始索引
    for(i=0;i<=nr_fold;i++)
        fold_start[i]=i*l/nr_fold;

    // 对每个折叠进行交叉验证
    for(i=0;i<nr_fold;i++)
    {
        int begin = fold_start[i];  // 当前折叠的起始索引
        int end = fold_start[i+1];  // 当前折叠的结束索引
        int j,k;
        struct problem subprob;

        // 设置子问题的偏置、特征数和样本数
        subprob.bias = prob->bias;
        subprob.n = prob->n;
        subprob.l = l-(end-begin);

        // 分配子问题的特征节点和标签内存空间
        subprob.x = Malloc(struct feature_node*,subprob.l);
        subprob.y = Malloc(double,subprob.l);

        k=0;
        // 复制属于当前折叠之外的样本到子问题中
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for(j=end;j<l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }

        // 训练子问题的模型
        struct model *submodel = train(&subprob,param);

        // 预测当前折叠中的样本，并存储预测结果到目标数组中
        for(j=begin;j<end;j++)
            target[perm[j]] = predict(submodel,prob->x[perm[j]]);

        // 释放子问题的内存空间
        free_and_destroy_model(&submodel);
        free(subprob.x);
        free(subprob.y);
    }

    // 释放折叠起始索引和样本索引的内存空间
    free(fold_start);
    free(perm);
}

// 对测试数据进行预测，返回预测的标签值
double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
    int idx;
    int n;
    // 根据模型的偏置值判断特征的维度是否需要加一
    if(model_->bias>=0)
        n=model_->nr_feature+1;
    else
        n=model_->nr_feature;

    // 获取模型的权重向量和类别数
    double *w=model_->w;
    int nr_class=model_->nr_class;
    int i;
    int nr_w;

    // 根据模型类型和解法类型确定权重向量的数目
    if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
        nr_w = 1;
    else
        nr_w = nr_class;

    // 遍历测试数据的特征节点，计算每个类别的决策值
    const feature_node *lx=x;
    for(i=0;i<nr_w;i++)
        dec_values[i] = 0;
    for(; (idx=lx->index)!=-1; lx++)
    {
        // 如果特征维度不超过训练数据的维度，则更新决策值
        if(idx<=n)
            for(i=0;i<nr_w;i++)
                dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
    }

    // 如果只有两类，根据模型类型决定返回的标签值
    if(nr_class==2)
    {
        if(check_regression_model(model_))
            return dec_values[0];
        else
            return (dec_values[0]>0)?model_->label[0]:model_->label[1];
    }
    else
    {
        // 对多类分类问题，选择具有最大决策值的类别标签返回
        int dec_max_idx = 0;
        for(i=1;i<nr_class;i++)
        {
            if(dec_values[i] > dec_values[dec_max_idx])
                dec_max_idx = i;
        }
        return model_->label[dec_max_idx];
    }
}

// 对单个样本进行预测，返回预测的标签值
double predict(const model *model_, const feature_node *x)
{
    // 分配存储类别决策值的内存空间
    double *dec_values = Malloc(double, model_->nr_class);

    // 调用预测值函数进行预测，并释放决策值内存空间后返回预测的标签值
    double label=predict_values(model_, x, dec_values);
    free(dec_values);
    return label;
}
// 预测函数，计算分类器的预测概率
double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
    // 检查模型是否支持概率预测
    if(check_probability_model(model_))
    {
        int i;
        int nr_class=model_->nr_class; // 获取类别数
        int nr_w;
        if(nr_class==2)
            nr_w = 1;
        else
            nr_w = nr_class;

        // 预测数据的标签值
        double label=predict_values(model_, x, prob_estimates);

        // 将预测的值转换为概率
        for(i=0;i<nr_w;i++)
            prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

        // 对于二元分类，调整概率估计
        if(nr_class==2) // for binary classification
            prob_estimates[1]=1.-prob_estimates[0];
        else
        {
            // 对多类分类进行归一化概率估计
            double sum=0;
            for(i=0; i<nr_class; i++)
                sum+=prob_estimates[i];

            for(i=0; i<nr_class; i++)
                prob_estimates[i]=prob_estimates[i]/sum;
        }

        // 返回预测的标签值
        return label;
    }
    else
        return 0; // 如果不支持概率预测，则返回0
}

// 支持的求解器类型表格
static const char *solver_type_table[]=
{
    "L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
    "L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
    "", "", "",
    "L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

// 将模型保存到文件
int save_model(const char *model_file_name, const struct model *model_)
{
    int i;
    int nr_feature=model_->nr_feature; // 获取特征数
    int n;
    const parameter& param = model_->param; // 获取模型参数

    // 计算特征向量的大小
    if(model_->bias>=0)
        n=nr_feature+1;
    else
        n=nr_feature;
    int w_size = n;

    // 打开模型文件，准备写入模型数据
    FILE *fp = fopen(model_file_name,"w");
    if(fp==NULL) return -1; // 如果打开文件失败，则返回错误码-1

    // 保存当前的本地化设置，并设置为"C"语言环境
    char *old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    int nr_w;
    // 确定需要保存的类别数
    if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
        nr_w=1;
    else
        nr_w=model_->nr_class;

    // 写入求解器类型到模型文件
    fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
    // 写入类别数到模型文件
    fprintf(fp, "nr_class %d\n", model_->nr_class);

    // 如果模型有标签信息，写入到模型文件
    if(model_->label)
    {
        fprintf(fp, "label");
        for(i=0; i<model_->nr_class; i++)
            fprintf(fp, " %d", model_->label[i]);
        fprintf(fp, "\n");
    }

    // 写入特征数到模型文件
    fprintf(fp, "nr_feature %d\n", nr_feature);

    // 写入偏置项到模型文件
    fprintf(fp, "bias %.16g\n", model_->bias);

    // 写入权重矩阵到模型文件
    fprintf(fp, "w\n");
    for(i=0; i<w_size; i++)
    {
        int j;
        for(j=0; j<nr_w; j++)
            fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
        fprintf(fp, "\n");
    }

    // 恢复原始的本地化设置
    setlocale(LC_ALL, old_locale);
    free(old_locale);

    // 检查文件写入是否出错，或者关闭文件是否成功，返回相应的错误码
    if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
    else return 0; // 如果操作成功完成，则返回0
}

// 加载模型文件
struct model *load_model(const char *model_file_name)
{
    FILE *fp = fopen(model_file_name,"r");
    if(fp==NULL) return NULL; // 如果打开文件失败，则返回空指针

    int i;
    int nr_feature;
    int n;
    int nr_class;
    double bias;
    model *model_ = Malloc(model,1); // 分配模型内存空间
    parameter& param = model_->param; // 获取模型参数

    model_->label = NULL; // 初始化模型标签为NULL

    // 保存当前的本地化设置，并设置为"C"语言环境
    char *old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    char cmd[81];
    while(1)
    {
        // 从文件流中读取最多80个字符到cmd变量中
        fscanf(fp,"%80s",cmd);
        // 如果cmd等于"solver_type"
        if(strcmp(cmd,"solver_type")==0)
        {
            // 再次从文件流中读取最多80个字符到cmd变量中
            fscanf(fp,"%80s",cmd);
            int i;
            // 遍历solver_type_table数组，直到遇到空指针为止
            for(i=0;solver_type_table[i];i++)
            {
                // 如果cmd与solver_type_table[i]相等
                if(strcmp(solver_type_table[i],cmd)==0)
                {
                    // 将i赋值给param.solver_type，并跳出循环
                    param.solver_type=i;
                    break;
                }
            }
            // 如果solver_type_table[i]为NULL
            if(solver_type_table[i] == NULL)
            {
                // 打印错误信息到标准错误输出
                fprintf(stderr,"unknown solver type.\n");
    
                // 恢复旧的locale设置
                setlocale(LC_ALL, old_locale);
                // 释放内存
                free(model_->label);
                free(model_);
                free(old_locale);
                // 返回NULL指针
                return NULL;
            }
        }
        // 如果cmd等于"nr_class"
        else if(strcmp(cmd,"nr_class")==0)
        {
            // 从文件流中读取一个整数到nr_class变量中
            fscanf(fp,"%d",&nr_class);
            // 将nr_class赋值给model_->nr_class
            model_->nr_class=nr_class;
        }
        // 如果cmd等于"nr_feature"
        else if(strcmp(cmd,"nr_feature")==0)
        {
            // 从文件流中读取一个整数到nr_feature变量中
            fscanf(fp,"%d",&nr_feature);
            // 将nr_feature赋值给model_->nr_feature
            model_->nr_feature=nr_feature;
        }
        // 如果cmd等于"bias"
        else if(strcmp(cmd,"bias")==0)
        {
            // 从文件流中读取一个双精度浮点数到bias变量中
            fscanf(fp,"%lf",&bias);
            // 将bias赋值给model_->bias
            model_->bias=bias;
        }
        // 如果cmd等于"w"
        else if(strcmp(cmd,"w")==0)
        {
            // 跳出循环
            break;
        }
        // 如果cmd等于"label"
        else if(strcmp(cmd,"label")==0)
        {
            // 从model_->nr_class指定数量的整数中读取数据到model_->label数组中
            int nr_class = model_->nr_class;
            model_->label = Malloc(int,nr_class);
            for(int i=0;i<nr_class;i++)
                fscanf(fp,"%d",&model_->label[i]);
        }
        // 如果cmd不匹配上述任何情况
        else
        {
            // 打印未知文本错误信息到标准错误输出，包括cmd内容
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            // 恢复旧的locale设置
            setlocale(LC_ALL, old_locale);
            // 释放内存
            free(model_->label);
            free(model_);
            free(old_locale);
            // 返回NULL指针
            return NULL;
        }
    }
    
    // 将model_->nr_feature赋值给nr_feature变量
    nr_feature=model_->nr_feature;
    // 根据model_->bias的值确定n的值
    if(model_->bias>=0)
        n=nr_feature+1;
    else
        n=nr_feature;
    // 计算w_size的值
    int w_size = n;
    int nr_w;
    // 根据nr_class和param.solver_type的值确定nr_w的值
    if(nr_class==2 && param.solver_type != MCSVM_CS)
        nr_w = 1;
    else
        nr_w = nr_class;
    
    // 分配内存给model_->w数组，大小为w_size*nr_w*sizeof(double)字节
    model_->w=Malloc(double, w_size*nr_w);
    for(i=0; i<w_size; i++)
    {
        int j;
        // 从文件流中读取w_size*nr_w个双精度浮点数到model_->w数组中
        for(j=0; j<nr_w; j++)
            fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
        // 从文件流中读取换行符
        fscanf(fp, "\n");
    }
    
    // 恢复旧的locale设置
    setlocale(LC_ALL, old_locale);
    // 释放内存
    free(old_locale);
    
    // 如果文件流fp发生错误或关闭失败，则返回NULL指针
    if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;
    
    // 返回model_指针，函数结束
    return model_;
#endif

// 返回模型中特征数目
int get_nr_feature(const model *model_)
{
    return model_->nr_feature;
}

// 返回模型中类别数目
int get_nr_class(const model *model_)
{
    return model_->nr_class;
}

// 获取模型中的类别标签
void get_labels(const model *model_, int* label)
{
    if (model_->label != NULL)
        for(int i=0;i<model_->nr_class;i++)
            label[i] = model_->label[i];
}

// 获取模型中的迭代次数
void get_n_iter(const model *model_, int* n_iter)
{
    int labels;
    labels = model_->nr_class;
    if (labels == 2)
        labels = 1;

    if (model_->n_iter != NULL)
        for(int i=0;i<labels;i++)
            n_iter[i] = model_->n_iter[i];
}

#if 0
// 在此处使用内联以提高性能（比非内联版本快约20%）
static inline double get_w_value(const struct model *model_, int idx, int label_idx)
{
    int nr_class = model_->nr_class;
    int solver_type = model_->param.solver_type;
    const double *w = model_->w;

    // 检查索引是否有效，若无效则返回0
    if(idx < 0 || idx > model_->nr_feature)
        return 0;
    
    // 对于回归模型，直接返回权重值
    if(check_regression_model(model_))
        return w[idx];
    else
    {
        // 对于分类模型，根据类别和解法类型返回相应的权重值
        if(label_idx < 0 || label_idx >= nr_class)
            return 0;
        if(nr_class == 2 && solver_type != MCSVM_CS)
        {
            if(label_idx == 0)
                return w[idx];
            else
                return -w[idx];
        }
        else
            return w[idx*nr_class+label_idx];
    }
}

// 获取决策函数中特征的系数
// feat_idx: 从1到nr_feature的特征索引
// label_idx: 对于分类模型，从0到nr_class-1；对于回归模型，忽略label_idx
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx)
{
    // 检查特征索引是否有效，若无效则返回0
    if(feat_idx > model_->nr_feature)
        return 0;
    return get_w_value(model_, feat_idx-1, label_idx);
}

// 获取决策函数的偏置
double get_decfun_bias(const struct model *model_, int label_idx)
{
    int bias_idx = model_->nr_feature;
    double bias = model_->bias;
    
    // 若偏置无效或小于等于0，则返回0
    if(bias <= 0)
        return 0;
    else
        // 返回偏置乘以权重值
        return bias * get_w_value(model_, bias_idx, label_idx);
}
#endif

// 释放模型结构体中的内容
void free_model_content(struct model *model_ptr)
{
    if(model_ptr->w != NULL)
        free(model_ptr->w);
    if(model_ptr->label != NULL)
        free(model_ptr->label);
    if(model_ptr->n_iter != NULL)
        free(model_ptr->n_iter);
}

// 释放并销毁模型结构体
void free_and_destroy_model(struct model **model_ptr_ptr)
{
    struct model *model_ptr = *model_ptr_ptr;
    if(model_ptr != NULL)
    {
        free_model_content(model_ptr);
        free(model_ptr);
    }
}

// 销毁参数结构体
void destroy_param(parameter* param)
{
    if(param->weight_label != NULL)
        free(param->weight_label);
    if(param->weight != NULL)
        free(param->weight);
}

// 检查问题参数的有效性
const char *check_parameter(const problem *prob, const parameter *param)
{
    if(param->eps <= 0)
        return "eps <= 0";

    if(param->C <= 0)
        return "C <= 0";

    if(param->p < 0)
        return "p < 0";
    # 检查参数指定的求解器类型是否属于已知的类型之一
    if(param->solver_type != L2R_LR
        && param->solver_type != L2R_L2LOSS_SVC_DUAL
        && param->solver_type != L2R_L2LOSS_SVC
        && param->solver_type != L2R_L1LOSS_SVC_DUAL
        && param->solver_type != MCSVM_CS
        && param->solver_type != L1R_L2LOSS_SVC
        && param->solver_type != L1R_LR
        && param->solver_type != L2R_LR_DUAL
        && param->solver_type != L2R_L2LOSS_SVR
        && param->solver_type != L2R_L2LOSS_SVR_DUAL
        && param->solver_type != L2R_L1LOSS_SVR_DUAL)
        # 如果求解器类型未知，返回字符串 "unknown solver type"
        return "unknown solver type";
    
    # 如果求解器类型已知，返回空指针
    return NULL;
}

#if 0
// 检查概率模型的类型，返回是否为逻辑回归类型的判断结果
int check_probability_model(const struct model *model_)
{
    return (model_->param.solver_type==L2R_LR ||
            model_->param.solver_type==L2R_LR_DUAL ||
            model_->param.solver_type==L1R_LR);
}
#endif

// 检查回归模型的类型，返回是否为支持向量回归类型的判断结果
int check_regression_model(const struct model *model_)
{
    return (model_->param.solver_type==L2R_L2LOSS_SVR ||
            model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
            model_->param.solver_type==L2R_L2LOSS_SVR_DUAL);
}

// 设置打印字符串的函数，允许指定用户自定义的打印函数或者默认使用标准输出
void set_print_string_function(void (*print_func)(const char*))
{
    // 如果传入的打印函数为空，则使用默认的标准输出打印函数
    if (print_func == NULL)
        liblinear_print_string = &print_string_stdout;
    else
        // 否则，使用传入的打印函数
        liblinear_print_string = print_func;
}
```