# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\liblinear\linear.h`

```
#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "_cython_blas_helpers.h"

// 结构体，表示一个特征节点，包括特征索引和特征值
struct feature_node
{
    int index;          // 特征索引
    double value;       // 特征值
};

// 结构体，表示问题，包括样本数l、特征数n、标签y、特征节点x数组、偏置项bias和权重W
struct problem
{
    int l, n;               // 样本数和特征数
    double *y;              // 标签数组
    struct feature_node **x; // 特征节点的二级指针，表示特征数据
    double bias;            // 偏置项，如果<0则表示没有偏置项
    double *W;              // 权重数组
};

// 枚举类型，表示不同的求解器类型
enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */

// 结构体，表示模型参数，包括求解器类型solver_type、停止准则eps、惩罚参数C、权重相关参数等
struct parameter
{
    int solver_type;        // 求解器类型

    /* these are for training only */
    double eps;             // 停止准则
    double C;               // 惩罚参数
    int nr_weight;          // 权重数目
    int *weight_label;      // 权重标签数组
    double* weight;         // 权重数组
    int max_iter;           // 最大迭代次数
    double p;               // 用于某些求解器的参数
};

// 结构体，表示模型，包括模型参数param、类别数nr_class、特征数nr_feature、权重数组w等
struct model
{
    struct parameter param; // 模型参数
    int nr_class;           // 类别数
    int nr_feature;         // 特征数
    double *w;              // 权重数组
    int *label;             // 每个类别的标签
    double bias;            // 偏置项
    int *n_iter;            // 每个类别的迭代次数数组
};

void set_seed(unsigned seed);

// 训练模型的函数声明
struct model* train(const struct problem *prob, const struct parameter *param, BlasFunctions *blas_functions);
void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, double *target);

// 根据模型预测的函数声明
double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);
double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

// 保存和加载模型的函数声明
int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

// 获取模型特征数、类别数、标签数组、迭代次数等函数声明
int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);
void get_n_iter(const struct model *model_, int* n_iter);

// 释放模型内容的函数声明
void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

// 检查模型参数、模型类型的函数声明
const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
int check_regression_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */
```