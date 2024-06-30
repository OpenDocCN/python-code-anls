# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\libsvm\svm.h`

```
#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 310

#ifdef __cplusplus
extern "C" {
#endif
#include "_svm_cython_blas_helpers.h"

// 定义 svm_node 结构体，表示 SVM 中的节点
struct svm_node
{
    int dim;            // 维度
    int ind;            // 索引。如果使用预先计算的核函数，则此字段是必要的，虽然有些冗余
    double *values;     // 值数组的指针
};

// 定义 svm_problem 结构体，表示 SVM 的问题
struct svm_problem
{
    int l;              // 样本数量
    double *y;          // 类别标签数组的指针
    struct svm_node *x; // 数据节点数组的指针
    double *W;          // 实例权重
};

// 定义 svm_csr_node 结构体，表示 SVM 的稀疏节点
struct svm_csr_node
{
    int index;          // 索引
    double value;       // 值
};

// 定义 svm_csr_problem 结构体，表示 SVM 的稀疏问题
struct svm_csr_problem
{
    int l;                      // 样本数量
    double *y;                  // 类别标签数组的指针
    struct svm_csr_node **x;    // 稀疏节点数组的指针
    double *W;                  // 实例权重
};

// 枚举 svm_type 表示 SVM 的类型
enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };

// 枚举 kernel_type 表示 SVM 的核函数类型
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };

// 定义 svm_parameter 结构体，表示 SVM 的参数
struct svm_parameter
{
    int svm_type;       // SVM 类型
    int kernel_type;    // 核函数类型
    int degree;         // 多项式核函数的次数
    double gamma;       // 多项式/RBF/Sigmoid 核函数的 gamma 参数
    double coef0;       // 多项式/Sigmoid 核函数的 coef0 参数

    // 以下参数仅用于训练
    double cache_size;  // 缓存大小（单位 MB）
    double eps;         // 停止训练的 epsilon 参数
    double C;           // C_SVC、EPSILON_SVR 和 NU_SVR 的参数
    int nr_weight;      // C_SVC 的参数
    int *weight_label;  // C_SVC 的参数
    double *weight;     // C_SVC 的参数
    double nu;          // NU_SVC、ONE_CLASS 和 NU_SVR 的参数
    double p;           // EPSILON_SVR 的参数
    int shrinking;      // 是否使用收缩启发式
    int probability;    // 是否进行概率估计
    int max_iter;       // 求解器运行时的最大迭代次数
    int random_seed;    // 随机数生成器的种子
};

//
// svm_model
//
// 定义 svm_model 结构体，表示 SVM 模型
struct svm_model
{
    struct svm_parameter param;    // 参数
    int nr_class;                  // 类别数，在回归/单类 SVM 中为 2
    int l;                         // 支持向量的总数
    struct svm_node *SV;           // 支持向量数组
    double **sv_coef;              // 决策函数中支持向量的系数数组
    int *n_iter;                   // 优化过程中运行的迭代次数

    int *sv_ind;                   // 支持向量的索引

    double *rho;                   // 决策函数中的常数（rho[k*(k-1)/2]）
    double *probA;                 // 类别对概率信息
    double *probB;

    /* 仅用于分类 */

    int *label;                    // 每个类别的标签
    int *nSV;                      // 每个类别的支持向量数量
                                   // nSV[0] + nSV[1] + ... + nSV[k-1] = l
    /* XXX */
    int free_sv;                   // 如果 svm_model 是由 svm_load_model 创建的，则为 1
                                   // 如果是由 svm_train 创建的，则为 0
};

// 定义 svm_csr_model 结构体，表示 SVM 的稀疏模型
struct svm_csr_model
{
    struct svm_parameter param;    // 参数
    int nr_class;                  // 类别数，在回归/单类 SVM 中为 2
    int l;                         // 支持向量的总数
    struct svm_csr_node **SV;      // 支持向量数组
    double **sv_coef;              // 决策函数中支持向量的系数数组
    # 模型拟合过程中的迭代次数
    int *n_iter;        /* number of iterations run by the optimization routine to fit the model */

    # 支持向量的索引
    int *sv_ind;            /* index of support vectors */

    # 决策函数中的常数项 (rho[k*(k-1)/2])
    double *rho;        /* constants in decision functions (rho[k*(k-1)/2]) */
    
    # 用于存储配对概率信息的数组
    double *probA;        /* pairwise probability information */
    double *probB;

    /* 仅用于分类 */

    # 每个类别的标签 (label[k])
    int *label;        /* label of each class (label[k]) */
    
    # 每个类别的支持向量数量 (nSV[k])
    int *nSV;        /* number of SVs for each class (nSV[k]) */
                /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
    
    /* XXX */
    
    # 表示 svm_model 是由 svm_load_model 创建的 (1)
    # 或是由 svm_train 创建的 (0)
    int free_sv;        /* 1 if svm_model is created by svm_load_model*/
                /* 0 if svm_model is created by svm_train */
};

/* svm_ functions are defined by libsvm_template.cpp from generic versions in svm.cpp */
// 定义了一系列 SVM 相关函数，这些函数是在 svm.cpp 中的通用版本由 libsvm_template.cpp 实现的

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param, int *status, BlasFunctions *blas_functions);
// SVM 模型训练函数，使用给定的问题 prob 和参数 param 进行训练，blas_functions 为 BLAS 函数集

void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target, BlasFunctions *blas_functions);
// SVM 模型的交叉验证函数，用于评估模型性能

int svm_save_model(const char *model_file_name, const struct svm_model *model);
// 将 SVM 模型保存到文件中，model_file_name 是保存文件名，model 是要保存的模型结构体指针

struct svm_model *svm_load_model(const char *model_file_name);
// 从文件中加载 SVM 模型，model_file_name 是加载的文件名

int svm_get_svm_type(const struct svm_model *model);
// 获取 SVM 模型的类型

int svm_get_nr_class(const struct svm_model *model);
// 获取 SVM 模型中的类别数目

void svm_get_labels(const struct svm_model *model, int *label);
// 获取 SVM 模型中的类别标签

double svm_get_svr_probability(const struct svm_model *model);
// 获取 SVM 模型用于回归时的概率估计

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values, BlasFunctions *blas_functions);
// 对给定的数据 x 进行预测，并返回决策值，dec_values 是决策值数组，blas_functions 是 BLAS 函数集

double svm_predict(const struct svm_model *model, const struct svm_node *x, BlasFunctions *blas_functions);
// 对给定的数据 x 进行预测，并返回预测结果，blas_functions 是 BLAS 函数集

double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates, BlasFunctions *blas_functions);
// 对给定的数据 x 进行预测，并返回每个类别的概率估计值，prob_estimates 是概率数组，blas_functions 是 BLAS 函数集

void svm_free_model_content(struct svm_model *model_ptr);
// 释放 SVM 模型结构体中的内容

void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
// 释放并销毁 SVM 模型结构体

void svm_destroy_param(struct svm_parameter *param);
// 销毁 SVM 参数结构体

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
// 检查 SVM 训练参数的有效性，并返回错误信息，如果参数有效返回 NULL

void svm_set_print_string_function(void (*print_func)(const char *));
// 设置打印字符串的函数指针，用于 SVM 内部的打印输出

/* sparse version */

/* svm_csr_ functions are defined by libsvm_template.cpp from generic versions in svm.cpp */
// sparse 版本的 SVM 函数，这些函数是在 svm.cpp 中的通用版本由 libsvm_template.cpp 实现的

struct svm_csr_model *svm_csr_train(const struct svm_csr_problem *prob, const struct svm_parameter *param, int *status, BlasFunctions *blas_functions);
// sparse 版本的 SVM 模型训练函数，使用给定的问题 prob 和参数 param 进行训练，blas_functions 为 BLAS 函数集

void svm_csr_cross_validation(const struct svm_csr_problem *prob, const struct svm_parameter *param, int nr_fold, double *target, BlasFunctions *blas_functions);
// sparse 版本的 SVM 模型交叉验证函数，用于评估模型性能

int svm_csr_get_svm_type(const struct svm_csr_model *model);
// 获取 sparse 版本的 SVM 模型类型

int svm_csr_get_nr_class(const struct svm_csr_model *model);
// 获取 sparse 版本的 SVM 模型中的类别数目

void svm_csr_get_labels(const struct svm_csr_model *model, int *label);
// 获取 sparse 版本的 SVM 模型中的类别标签

double svm_csr_get_svr_probability(const struct svm_csr_model *model);
// 获取 sparse 版本的 SVM 模型用于回归时的概率估计

double svm_csr_predict_values(const struct svm_csr_model *model, const struct svm_csr_node *x, double* dec_values, BlasFunctions *blas_functions);
// 对给定的数据 x 进行预测，并返回决策值，dec_values 是决策值数组，blas_functions 是 BLAS 函数集

double svm_csr_predict(const struct svm_csr_model *model, const struct svm_csr_node *x, BlasFunctions *blas_functions);
// 对给定的数据 x 进行预测，并返回预测结果，blas_functions 是 BLAS 函数集

double svm_csr_predict_probability(const struct svm_csr_model *model, const struct svm_csr_node *x, double* prob_estimates, BlasFunctions *blas_functions);
// 对给定的数据 x 进行预测，并返回每个类别的概率估计值，prob_estimates 是概率数组，blas_functions 是 BLAS 函数集

void svm_csr_free_model_content(struct svm_csr_model *model_ptr);
// 释放 sparse 版本的 SVM 模型结构体中的内容

void svm_csr_free_and_destroy_model(struct svm_csr_model **model_ptr_ptr);
// 释放并销毁 sparse 版本的 SVM 模型结构体

void svm_csr_destroy_param(struct svm_parameter *param);
// 销毁 sparse 版本的 SVM 参数结构体

const char *svm_csr_check_parameter(const struct svm_csr_problem *prob, const struct svm_parameter *param);
// 检查 sparse 版本的 SVM 训练参数的有效性，并返回错误信息，如果参数有效返回 NULL

/* end sparse version */

#ifdef __cplusplus
}
#endif
#endif /* _LIBSVM_H */



// 如果未定义 _LIBSVM_H 宏，则结束当前头文件的条件编译
#endif /* _LIBSVM_H */


这段代码是用于C/C++中的条件编译。`#endif` 用于结束之前的 `#ifdef` 或 `#ifndef` 条件编译块，它会匹配最近的 `#ifdef` 或 `#ifndef`。在这里，`#endif` 后面跟着的注释 `/* _LIBSVM_H */` 是为了标识这个条件编译块所对应的宏名 `_LIBSVM_H`。条件编译用来根据不同的宏定义来选择性地包含或排除代码段，以便在编译时根据不同的条件进行控制。
```