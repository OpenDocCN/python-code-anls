# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\libsvm\libsvm_helper.c`

```
/*
 * 包含必要的头文件：stdlib.h、Python.h 和 svm.h
 */
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "svm.h"
#include "_svm_cython_blas_helpers.h"

/*
 * 定义一个宏 MAX，返回两个数中较大的那个
 */
#ifndef MAX
    #define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

/*
 * Some helper methods for libsvm bindings.
 *
 * We need to access from python some parameters stored in svm_model
 * but libsvm does not expose this structure, so we define it here
 * along some utilities to convert from numpy arrays.
 *
 * License: BSD 3 clause
 *
 * Author: 2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
 */

/*
 * 将密集矩阵转换为适合于 libsvm 的稀疏表示形式。x 是长度为 nrow*ncol 的数组。
 *
 * 通常矩阵是密集的，因此我们加速这个过程。我们创建一个临时数组 temp 收集非零元素，
 * 然后将其 memcpy 到正确的数组中。
 *
 * 对于索引值，需要特别小心，因为 libsvm 的索引从 1 开始而不是从 0 开始。
 *
 * 严格来说，C 标准不要求结构体是连续的，但在实践中这是一个合理的假设。
 */
struct svm_node *dense_to_libsvm(double *x, Py_ssize_t *dims)
{
    struct svm_node *node;
    Py_ssize_t len_row = dims[1];
    double *tx = x;
    int i;

    node = malloc(dims[0] * sizeof(struct svm_node));

    if (node == NULL) return NULL;
    for (i = 0; i < dims[0]; ++i) {
        node[i].values = tx;
        node[i].dim = (int) len_row;
        node[i].ind = i; /* 只在 kernel=precomputed 时使用，但开销不大 */
        tx += len_row;
    }

    return node;
}

/*
 * 填充一个 svm_parameter 结构体。
 */
void set_parameter(struct svm_parameter *param, int svm_type, int kernel_type, int degree,
        double gamma, double coef0, double nu, double cache_size, double C,
        double eps, double p, int shrinking, int probability, int nr_weight,
        char *weight_label, char *weight, int max_iter, int random_seed)
{
    param->svm_type = svm_type;
    param->kernel_type = kernel_type;
    param->degree = degree;
    param->coef0 = coef0;
    param->nu = nu;
    param->cache_size = cache_size;
    param->C = C;
    param->eps = eps;
    param->p = p;
    param->shrinking = shrinking;
    param->probability = probability;
    param->nr_weight = nr_weight;
    param->weight_label = (int *) weight_label;
    param->weight = (double *) weight;
    param->gamma = gamma;
    param->max_iter = max_iter;
    param->random_seed = random_seed;
}

/*
 * 填充一个 svm_problem 结构体。problem->x 将被 malloc。
 */
void set_problem(struct svm_problem *problem, char *X, char *Y, char *sample_weight, Py_ssize_t *dims, int kernel_type)
{
    if (problem == NULL) return;
    problem->l = (int) dims[0]; /* 样本数 */
    problem->y = (double *) Y;
    problem->x = dense_to_libsvm((double *) X, dims); /* 隐含调用 malloc */
    problem->W = (double *) sample_weight;
}
/*
 * 创建并返回一个 svm_model 实例。
 *
 * 复制 model->sv_coef 应该很直接，但不幸的是，numpy 和 libsvm 用于表示矩阵的方法不同，
 * 因此需要一些迭代来处理。
 *
 * 可能的问题：在64位系统上，numpy 可以存储的列数是一个 long，但是 libsvm 强制 model->l 
 * 必须是一个 int，因此可能会出现 numpy 矩阵无法适应 libsvm 数据结构的情况。
 */
struct svm_model *set_model(struct svm_parameter *param, int nr_class,
                            char *SV, Py_ssize_t *SV_dims,
                            char *support, Py_ssize_t *support_dims,
                            Py_ssize_t *sv_coef_strides,
                            char *sv_coef, char *rho, char *nSV,
                            char *probA, char *probB)
{
    struct svm_model *model;    // 定义 svm_model 结构体指针变量 model
    double *dsv_coef = (double *) sv_coef;   // 将 sv_coef 转换为 double 类型指针

    int i, m;   // 定义整型变量 i 和 m

    m = nr_class * (nr_class-1)/2;   // 计算 m 的值，用于存储 model->rho 数组的长度

    // 分配内存给 svm_model 结构体指针变量 model
    if ((model = malloc(sizeof(struct svm_model))) == NULL)
        goto model_error;
    
    // 分配内存给 model->nSV 数组
    if ((model->nSV = malloc(nr_class * sizeof(int))) == NULL)
        goto nsv_error;
    
    // 分配内存给 model->label 数组
    if ((model->label = malloc(nr_class * sizeof(int))) == NULL)
        goto label_error;
    
    // 分配内存给 model->sv_coef 数组
    if ((model->sv_coef = malloc((nr_class-1)*sizeof(double *))) == NULL)
        goto sv_coef_error;
    
    // 分配内存给 model->rho 数组
    if ((model->rho = malloc( m * sizeof(double))) == NULL)
        goto rho_error;

    // 在训练时，model->n_iter 只分配在动态内存中
    model->n_iter = NULL;

    model->nr_class = nr_class;   // 设置 model 的类别数量
    model->param = *param;        // 复制 svm_parameter 结构体参数到 model->param
    model->l = (int) support_dims[0];   // 设置 model 的 l 属性为 support_dims 的第一个元素

    // 根据 kernel_type 类型选择分配 model->SV 数组内存或者调用 dense_to_libsvm 函数转换
    if (param->kernel_type == PRECOMPUTED) {
        if ((model->SV = malloc ((model->l) * sizeof(struct svm_node))) == NULL)
            goto SV_error;
        for (i=0; i<model->l; ++i) {
            model->SV[i].ind = ((int *) support)[i];
            model->SV[i].values = NULL;
        }
    } else {
        model->SV = dense_to_libsvm((double *) SV, SV_dims);
    }

    // 对于回归和单类分类器，不使用 nSV 和 label
    if (param->svm_type < 2) {
        memcpy(model->nSV, nSV,     model->nr_class * sizeof(int));
        for(i=0; i < model->nr_class; i++)
            model->label[i] = i;
    }

    // 将 sv_coef 赋值给 model->sv_coef 数组
    for (i=0; i < model->nr_class-1; i++) {
        model->sv_coef[i] = dsv_coef + i*(model->l);
    }

    // 将 rho 赋值给 model->rho 数组
    for (i=0; i<m; ++i) {
        (model->rho)[i] = -((double *) rho)[i];
    }

    // 为了避免段错误，这些特性不被包装，但是 svm_destroy_model 将尝试释放它们。
    
    // 如果 param->probability 为真，分配 model->probA 和 model->probB 数组内存并复制数据
    if (param->probability) {
        if ((model->probA = malloc(m * sizeof(double))) == NULL)
            goto probA_error;
        memcpy(model->probA, probA, m * sizeof(double));
        if ((model->probB = malloc(m * sizeof(double))) == NULL)
            goto probB_error;
        memcpy(model->probB, probB, m * sizeof(double));

            }
        }

        // 返回构建好的 svm_model 结构体指针变量 model
        return model;

probB_error:
        free(model->probA);
probA_error:
        free(model->rho);
rho_error:
        free(model->sv_coef);
sv_coef_error:
        free(model->label);
label_error:
        free(model->nSV);
nsv_error:
        free(model);
model_error:
        return NULL;
}
    } else {
        // 如果不是概率模型，将模型的probA和probB设为NULL
        model->probA = NULL;
        model->probB = NULL;
    }

    /* We'll free SV ourselves */
    // 设置free_sv为0，表示我们将手动释放SV（支持向量）
    model->free_sv = 0;
    // 返回模型对象
    return model;
probB_error:
    // 释放模型中的probA数组内存
    free(model->probA);
probA_error:
    // 释放模型中的SV数组内存
    free(model->SV);
SV_error:
    // 释放模型中的rho数组内存
    free(model->rho);
rho_error:
    // 释放模型中的sv_coef数组内存
    free(model->sv_coef);
sv_coef_error:
    // 释放模型中的label数组内存
    free(model->label);
label_error:
    // 释放模型中的nSV数组内存
    free(model->nSV);
nsv_error:
    // 释放整个模型结构体内存
    free(model);
model_error:
    // 返回空指针，表示内存释放后的模型无效
    return NULL;
}



/*
 * Get the number of support vectors in a model.
 */
Py_ssize_t get_l(struct svm_model *model)
{
    // 返回模型中支持向量的数量
    return (Py_ssize_t) model->l;
}

/*
 * Get the number of classes in a model, = 2 in regression/one class
 * svm.
 */
Py_ssize_t get_nr(struct svm_model *model)
{
    // 返回模型中的类别数量，在回归或单类别SVM中应为2
    return (Py_ssize_t) model->nr_class;
}

/*
 * Get the number of iterations run in optimization
 */
void copy_n_iter(char *data, struct svm_model *model)
{
    // 计算优化中运行的迭代次数，并将其拷贝到data中
    const int n_models = MAX(1, model->nr_class * (model->nr_class-1) / 2);
    memcpy(data, model->n_iter, n_models * sizeof(int));
}

/*
 * Some helpers to convert from libsvm sparse data structures
 * model->sv_coef is a double **, whereas data is just a double *,
 * so we have to do some stupid copying.
 */
void copy_sv_coef(char *data, struct svm_model *model)
{
    // 将模型中的支持向量系数拷贝到data中
    int i, len = model->nr_class-1;
    double *temp = (double *) data;
    for(i=0; i<len; ++i) {
        memcpy(temp, model->sv_coef[i], sizeof(double) * model->l);
        temp += model->l;
    }
}

void copy_intercept(char *data, struct svm_model *model, Py_ssize_t *dims)
{
    /* intercept = -rho */
    // 拷贝模型中的截距（intercept），由于intercept = -rho
    Py_ssize_t i, n = dims[0];
    double t, *ddata = (double *) data;
    for (i=0; i<n; ++i) {
        t = model->rho[i];
        /* we do this to avoid ugly -0.0 */
        *ddata = (t != 0) ? -t : 0;
        ++ddata;
    }
}

/*
 * This is a bit more complex since SV are stored as sparse
 * structures, so we have to do the conversion on the fly and also
 * iterate fast over data.
 */
void copy_SV(char *data, struct svm_model *model, Py_ssize_t *dims)
{
    // 将模型中的支持向量（SV）拷贝到data中
    int i, n = model->l;
    double *tdata = (double *) data;
    int dim = model->SV[0].dim;
    for (i=0; i<n; ++i) {
        memcpy (tdata, model->SV[i].values, dim * sizeof(double));
        tdata += dim;
    }
}

void copy_support (char *data, struct svm_model *model)
{
    // 将模型中的支持向量索引拷贝到data中
    memcpy (data, model->sv_ind, (model->l) * sizeof(int));
}

/*
 * copy svm_model.nSV, an array with the number of SV for each class
 * will be NULL in the case of SVR, OneClass
 */
void copy_nSV(char *data, struct svm_model *model)
{
    // 将模型中的每个类别的支持向量数目拷贝到data中
    if (model->label == NULL) return;
    memcpy(data, model->nSV, model->nr_class * sizeof(int));
}

void copy_probA(char *data, struct svm_model *model, Py_ssize_t * dims)
{
    // 将模型中的probA数组拷贝到data中
    memcpy(data, model->probA, dims[0] * sizeof(double));
}

void copy_probB(char *data, struct svm_model *model, Py_ssize_t * dims)
{
    // 将模型中的probB数组拷贝到data中
    memcpy(data, model->probB, dims[0] * sizeof(double));
}

/*
 * Predict using model.
 *
 *  It will return -1 if we run out of memory.
 */
int copy_predict(char *predict, struct svm_model *model, Py_ssize_t *predict_dims,
                 char *dec_values, BlasFunctions *blas_functions)
{
    // 使用模型进行预测，并将预测结果存储在predict中，将决策值存储在dec_values中
    double *t = (double *) dec_values;
    // 定义一个指向 svm_node 结构体的指针变量 predict_nodes，用于存储转换后的预测数据
    struct svm_node *predict_nodes;
    // Py_ssize_t 类型的变量 i，用于循环迭代
    Py_ssize_t i;

    // 调用 dense_to_libsvm 函数将 predict 数组转换为 libsvm 需要的数据格式，并将结果赋给 predict_nodes
    predict_nodes = dense_to_libsvm((double *) predict, predict_dims);

    // 如果 predict_nodes 为 NULL，表示转换失败，返回 -1
    if (predict_nodes == NULL)
        return -1;

    // 遍历 predict_nodes 数组，进行预测操作，并将结果存储到 t 指向的位置
    for(i=0; i<predict_dims[0]; ++i) {
        *t = svm_predict(model, &predict_nodes[i], blas_functions);
        ++t;
    }

    // 释放 predict_nodes 数组所占用的内存
    free(predict_nodes);

    // 返回预测成功的标志，即返回 0
    return 0;
/*
 * 复制预测值函数
 * 将预测结果复制到dec_values中，同时进行blas函数计算
 */
int copy_predict_values(char *predict, struct svm_model *model,
                        Py_ssize_t *predict_dims, char *dec_values, int nr_class, BlasFunctions *blas_functions)
{
    Py_ssize_t i;
    struct svm_node *predict_nodes;

    // 将预测数据转换为libsvm格式的节点数组
    predict_nodes = dense_to_libsvm((double *) predict, predict_dims);
    if (predict_nodes == NULL)
        return -1;

    // 遍历预测节点并进行预测值计算
    for(i=0; i<predict_dims[0]; ++i) {
        svm_predict_values(model, &predict_nodes[i],
                           ((double *) dec_values) + i*nr_class,
                           blas_functions);
    }

    // 释放预测节点数组的内存
    free(predict_nodes);
    return 0;
}



/*
 * 复制预测概率函数
 * 将预测结果复制到dec_values中，同时进行blas函数计算
 */
int copy_predict_proba(char *predict, struct svm_model *model, Py_ssize_t *predict_dims,
                       char *dec_values, BlasFunctions *blas_functions)
{
    Py_ssize_t i, n, m;
    struct svm_node *predict_nodes;

    // 获取预测数据的维度信息
    n = predict_dims[0];
    m = (Py_ssize_t) model->nr_class;

    // 将预测数据转换为libsvm格式的节点数组
    predict_nodes = dense_to_libsvm((double *) predict, predict_dims);
    if (predict_nodes == NULL)
        return -1;

    // 遍历预测节点并进行预测概率计算
    for(i=0; i<n; ++i) {
        svm_predict_probability(model, &predict_nodes[i],
                                ((double *) dec_values) + i*m,
                                blas_functions);
    }

    // 释放预测节点数组的内存
    free(predict_nodes);
    return 0;
}


/*
 * 释放模型内存函数
 * 释放svm_model结构体及其内部分配的内存空间
 */
int free_model(struct svm_model *model)
{
    /* 类似于svm_free_and_destroy_model，但不释放sv_coef[i] */
    if (model == NULL) return -1;

    // 释放SV数组的内存
    free(model->SV);

    // 不释放sv_ind和n_iter，因为它们在set_model函数中未被创建
    /* free(model->sv_ind);
     * free(model->n_iter);
     */

    // 释放sv_coef数组的内存
    free(model->sv_coef);

    // 释放rho数组的内存
    free(model->rho);

    // 释放label数组的内存
    free(model->label);

    // 释放probA数组的内存
    free(model->probA);

    // 释放probB数组的内存
    free(model->probB);

    // 释放nSV数组的内存
    free(model->nSV);

    // 最后释放svm_model结构体的内存
    free(model);

    return 0;
}


/*
 * 释放参数内存函数
 * 释放svm_parameter结构体及其内部分配的内存空间
 */
int free_param(struct svm_parameter *param)
{
    if (param == NULL) return -1;

    // 释放svm_parameter结构体的内存
    free(param);

    return 0;
}


/* 从原始libsvm代码中借用的静态函数 */
static void print_null(const char *s) {}

/*
 * 将字符串输出到标准输出函数
 * 将字符串s输出到标准输出，并刷新输出流
 */
static void print_string_stdout(const char *s)
{
    // 将字符串s输出到标准输出
    fputs(s, stdout);

    // 刷新标准输出流
    fflush(stdout);
}


/*
 * 设置打印详细信息函数
 * 根据verbosity_flag设置输出函数为print_string_stdout或print_null
 */
void set_verbosity(int verbosity_flag){
    if (verbosity_flag)
        // 如果verbosity_flag为真，设置输出函数为print_string_stdout
        svm_set_print_string_function(&print_string_stdout);
    else
        // 如果verbosity_flag为假，设置输出函数为print_null
        svm_set_print_string_function(&print_null);
}
```