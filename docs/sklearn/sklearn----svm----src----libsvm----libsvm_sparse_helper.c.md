# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\libsvm\libsvm_sparse_helper.c`

```
/*
 * 包含标准库头文件
 */
#include <stdlib.h>

/*
 * 定义宏，清除 Python.h 中可能定义的 PY_SSIZE_T_CLEAN 标记
 */
#define PY_SSIZE_T_CLEAN
/*
 * 包含 Python C API 头文件
 */
#include <Python.h>
/*
 * 包含 libsvm 头文件
 */
#include "svm.h"
/*
 * 包含 Cython 生成的头文件，提供对 BLAS 的辅助函数支持
 */
#include "_svm_cython_blas_helpers.h"

/*
 * 如果未定义 MAX 宏，则定义 MAX 宏为取两数中的较大值
 */
#ifndef MAX
    #define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

/*
 * 将 scipy.sparse.csr 转换为 libsvm 的稀疏数据结构
 */
struct svm_csr_node **csr_to_libsvm (double *values, int* indices, int* indptr, int n_samples)
{
    struct svm_csr_node **sparse, *temp;
    int i, j=0, k=0, n;
    /*
     * 分配稀疏结构体指针数组的内存空间
     */
    sparse = malloc (n_samples * sizeof(struct svm_csr_node *));
    if (sparse == NULL)
        return NULL;

    for (i=0; i<n_samples; ++i) {
        /*
         * 计算第 i 行中的元素个数
         */
        n = indptr[i+1] - indptr[i]; /* count elements in row i */
        /*
         * 分配临时节点数组的内存空间
         */
        temp = malloc ((n+1) * sizeof(struct svm_csr_node));
        if (temp == NULL) {
            /*
             * 如果分配失败，则释放之前分配的内存空间并返回 NULL
             */
            for (j=0; j<i; j++)
                free(sparse[j]);
            free(sparse);
            return NULL;
        }

        for (j=0; j<n; ++j) {
            /*
             * 将值和索引填充到临时节点数组中
             */
            temp[j].value = values[k];
            temp[j].index = indices[k] + 1; /* libsvm uses 1-based indexing */
            ++k;
        }
        /* 设置哨兵值 */
        temp[n].index = -1;
        sparse[i] = temp;
    }

    return sparse;
}

/*
 * 设置 SVM 参数结构体
 */
struct svm_parameter * set_parameter(int svm_type, int kernel_type, int degree,
        double gamma, double coef0, double nu, double cache_size, double C,
        double eps, double p, int shrinking, int probability, int nr_weight,
        char *weight_label, char *weight, int max_iter, int random_seed)
{
    struct svm_parameter *param;
    /*
     * 分配 SVM 参数结构体的内存空间
     */
    param = malloc(sizeof(struct svm_parameter));
    if (param == NULL) return NULL;
    /*
     * 设置 SVM 参数的各个字段
     */
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
    return param;
}

/*
 * 从 scipy.sparse.csr 矩阵创建并返回 svm_csr_problem 结构体。返回的结构体需要用户手动释放。
 */
struct svm_csr_problem * csr_set_problem (char *values, Py_ssize_t *n_indices,
        char *indices, Py_ssize_t *n_indptr, char *indptr, char *Y,
                char *sample_weight, int kernel_type) {

    struct svm_csr_problem *problem;
    /*
     * 分配 svm_csr_problem 结构体的内存空间
     */
    problem = malloc (sizeof (struct svm_csr_problem));
    if (problem == NULL) return NULL;
    /*
     * 设置问题结构体的字段
     */
    problem->l = (int) n_indptr[0] - 1;
    problem->y = (double *) Y;
    /*
     * 将 scipy.sparse.csr 转换为 libsvm 的稀疏数据结构
     */
    problem->x = csr_to_libsvm((double *) values, (int *) indices,
                               (int *) indptr, problem->l);
    /* 
     * 一旦实现了加权样本，应该移除这部分代码
     */
    problem->W = (double *) sample_weight;

    if (problem->x == NULL) {
        /*
         * 如果转换失败，释放问题结构体的内存空间并返回 NULL
         */
        free(problem);
        return NULL;
    }
    # 返回函数的结果变量 problem
    return problem;
    // 定义一个函数 csr_set_model，返回类型为 struct svm_csr_model*，接受多个参数
    struct svm_csr_model *csr_set_model(struct svm_parameter *param, int nr_class,
                                        char *SV_data, Py_ssize_t *SV_indices_dims,
                                        char *SV_indices, Py_ssize_t *SV_indptr_dims,
                                        char *SV_intptr,
                                        char *sv_coef, char *rho, char *nSV,
                                        char *probA, char *probB)
    {
        // 声明结构体指针 model
        struct svm_csr_model *model;
        // 将 sv_coef 强制转换为 double* 类型，赋值给 dsv_coef
        double *dsv_coef = (double *) sv_coef;
        // 声明整型变量 i 和 m
        int i, m;

        // 计算 m 的值，用于后续内存分配
        m = nr_class * (nr_class-1)/2;

        // 分配 model 的内存空间，并检查是否成功
        if ((model = malloc(sizeof(struct svm_csr_model))) == NULL)
            goto model_error;
        // 分配 model->nSV 的内存空间，并检查是否成功
        if ((model->nSV = malloc(nr_class * sizeof(int))) == NULL)
            goto nsv_error;
        // 分配 model->label 的内存空间，并检查是否成功
        if ((model->label = malloc(nr_class * sizeof(int))) == NULL)
            goto label_error;
        // 分配 model->sv_coef 的内存空间，并检查是否成功
        if ((model->sv_coef = malloc((nr_class-1)*sizeof(double *))) == NULL)
            goto sv_coef_error;
        // 分配 model->rho 的内存空间，并检查是否成功
        if ((model->rho = malloc( m * sizeof(double))) == NULL)
            goto rho_error;

        // 在训练时只分配 model->n_iter 的内存空间，并设置为 NULL
        model->n_iter = NULL;

        /* 对于预先计算的核函数，不使用 dense_to_precomputed，
           因为我们不希望有前导的 0。由于索引从 1 开始（而不是从 0 开始），这将起作用 */
        // 设置 model->l 的值为 SV_indptr_dims[0] - 1
        model->l = (int) SV_indptr_dims[0] - 1;
        // 将 CSR 格式的数据转换成 libsvm 格式的数据，并赋给 model->SV
        model->SV = csr_to_libsvm((double *) SV_data, (int *) SV_indices,
                                  (int *) SV_intptr, model->l);
        // 设置 model 的 nr_class 属性
        model->nr_class = nr_class;
        // 复制 param 参数的值到 model->param
        model->param = *param;

        /*
         * 对于回归和单类分类，不使用 nSV 和 label。
         */
        // 如果 param->svm_type 小于 2，则执行以下操作
        if (param->svm_type < 2) {
            // 复制 nSV 数组的值到 model->nSV
            memcpy(model->nSV,   nSV,   model->nr_class * sizeof(int));
            // 初始化 model->label 数组
            for(i=0; i < model->nr_class; i++)
                model->label[i] = i;
        }

        // 遍历 model->sv_coef 数组，并为每个元素分配内存空间并复制数据
        for (i=0; i < model->nr_class-1; i++) {
            /*
             * 由于 svm_destroy_model 将释放数组的每个元素，所以不能将所有这些 malloc 收拢在一次调用中。
             */
            // 分配 model->sv_coef[i] 的内存空间，并检查是否成功
            if ((model->sv_coef[i] = malloc((model->l) * sizeof(double))) == NULL) {
                int j;
                // 如果分配失败，释放之前已分配的内存空间
                for (j=0; j<i; j++)
                    free(model->sv_coef[j]);
                goto sv_coef_i_error;
            }
            // 复制 dsv_coef 数组的值到 model->sv_coef[i]
            memcpy(model->sv_coef[i], dsv_coef, (model->l) * sizeof(double));
            dsv_coef += model->l;
        }

        // 将 rho 数组的值取负数后复制到 model->rho 数组
        for (i=0; i<m; ++i) {
            (model->rho)[i] = -((double *) rho)[i];
        }

        /*
         * 为了避免段错误，这些特性没有被包装，但是 svm_destroy_model 将尝试释放它们。
         */
        
        // 如果 param->probability 为 true，则分配 model->probA 和 model->probB 的内存空间，并复制数据
        if (param->probability) {
            if ((model->probA = malloc(m * sizeof(double))) == NULL)
                goto probA_error;
            memcpy(model->probA, probA, m * sizeof(double));
            if ((model->probB = malloc(m * sizeof(double))) == NULL)
                goto probB_error;
            memcpy(model->probB, probB, m * sizeof(double));
        } else {
            // 否则将 model->probA 和 model->probB 设置为 NULL
            model->probA = NULL;
            model->probB = NULL;
        }

        // 设置 model->free_sv 为 0，表示不释放 SV 数据
        model->free_sv = 0;
        // 返回构建好的 model 结构体指针
        return model;

    probB_error:
        // 错误处理标签
        ```
    }
    // 释放存储在模型结构体中probA指针指向的内存空间
    free(model->probA);
probA_error:
    for (i=0; i < model->nr_class-1; i++)
        free(model->sv_coef[i]);
    # 释放每个类别支持向量系数的内存
sv_coef_i_error:
    free(model->rho);
    # 释放模型的 rho（偏置）的内存
rho_error:
    free(model->sv_coef);
    # 释放所有类别的支持向量系数数组的内存
sv_coef_error:
    free(model->label);
    # 释放模型的类别标签数组的内存
label_error:
    free(model->nSV);
    # 释放模型的支持向量数目数组的内存
nsv_error:
    free(model);
    # 释放整个模型结构体的内存
model_error:
    return NULL;
    # 返回空指针，表示出现了错误


/*
 * Copy support vectors into a scipy.sparse.csr matrix
 */
int csr_copy_SV (char *data, Py_ssize_t *n_indices,
        char *indices, Py_ssize_t *n_indptr, char *indptr,
        struct svm_csr_model *model, int n_features)
{
    int i, j, k=0, index;
    double *dvalues = (double *) data;
    int *iindices = (int *) indices;
    int *iindptr  = (int *) indptr;
    iindptr[0] = 0;
    for (i=0; i<model->l; ++i) { /* iterate over support vectors */
        index = model->SV[i][0].index;
        for(j=0; index >=0 ; ++j) {
            iindices[k] = index - 1;
            dvalues[k] = model->SV[i][j].value;
            index = model->SV[i][j+1].index;
            ++k;
        }
        iindptr[i+1] = k;
    }
    # 将支持向量数据复制到 scipy.sparse.csr 矩阵中
    return 0;
}


/* get number of nonzero coefficients in support vectors */
Py_ssize_t get_nonzero_SV (struct svm_csr_model *model) {
    int i, j;
    Py_ssize_t count=0;
    for (i=0; i<model->l; ++i) {
        j = 0;
        while (model->SV[i][j].index != -1) {
            ++j;
            ++count;
        }
    }
    # 获取支持向量中非零系数的数目
    return count;
}


/*
 * Predict using a model, where data is expected to be encoded into a csr matrix.
 */
int csr_copy_predict (Py_ssize_t *data_size, char *data, Py_ssize_t *index_size,
        char *index, Py_ssize_t *intptr_size, char *intptr, struct svm_csr_model *model,
        char *dec_values, BlasFunctions *blas_functions) {
    double *t = (double *) dec_values;
    struct svm_csr_node **predict_nodes;
    Py_ssize_t i;

    predict_nodes = csr_to_libsvm((double *) data, (int *) index,
                                  (int *) intptr, intptr_size[0]-1);

    if (predict_nodes == NULL)
        return -1;
    for(i=0; i < intptr_size[0] - 1; ++i) {
        *t = svm_csr_predict(model, predict_nodes[i], blas_functions);
        free(predict_nodes[i]);
        ++t;
    }
    free(predict_nodes);
    # 使用 csr 矩阵预测结果，并释放相关的内存
    return 0;
}

int csr_copy_predict_values (Py_ssize_t *data_size, char *data, Py_ssize_t *index_size,
                char *index, Py_ssize_t *intptr_size, char *intptr, struct svm_csr_model *model,
                char *dec_values, int nr_class, BlasFunctions *blas_functions) {
    struct svm_csr_node **predict_nodes;
    Py_ssize_t i;

    predict_nodes = csr_to_libsvm((double *) data, (int *) index,
                                  (int *) intptr, intptr_size[0]-1);

    if (predict_nodes == NULL)
        return -1;
    for(i=0; i < intptr_size[0] - 1; ++i) {
        svm_csr_predict_values(model, predict_nodes[i],
                               ((double *) dec_values) + i*nr_class,
                   blas_functions);
        free(predict_nodes[i]);
    }
    free(predict_nodes);
    # 使用 csr 矩阵预测结果的概率值，并释放相关的内存
    return 0;
}
/*
 * Copy predicted probabilities from the CSR format data to the output array.
 * This function computes predictions using a CSR model and stores them in dec_values.
 * It returns 0 on success and -1 if prediction nodes are NULL.
 */
int csr_copy_predict_proba (Py_ssize_t *data_size, char *data, Py_ssize_t *index_size,
        char *index, Py_ssize_t *intptr_size, char *intptr, struct svm_csr_model *model,
        char *dec_values, BlasFunctions *blas_functions) {

    struct svm_csr_node **predict_nodes;
    Py_ssize_t i;
    int m = model->nr_class;

    // Convert CSR format data to libsvm format
    predict_nodes = csr_to_libsvm((double *) data, (int *) index,
                                  (int *) intptr, intptr_size[0]-1);

    if (predict_nodes == NULL)
        return -1;

    // Predict probabilities for each node
    for(i=0; i < intptr_size[0] - 1; ++i) {
        svm_csr_predict_probability(
        model, predict_nodes[i], ((double *) dec_values) + i*m, blas_functions);
        free(predict_nodes[i]);
    }
    free(predict_nodes);
    return 0;
}

/*
 * Retrieve the number of classes in the SVM CSR model.
 */
Py_ssize_t get_nr(struct svm_csr_model *model)
{
    return (Py_ssize_t) model->nr_class;
}

/*
 * Copy intercept values (-rho) from the model to the data array.
 * This function avoids setting -0.0 by converting 0 to 0 explicitly.
 */
void copy_intercept(char *data, struct svm_csr_model *model, Py_ssize_t *dims)
{
    Py_ssize_t i, n = dims[0];
    double t, *ddata = (double *) data;
    for (i=0; i<n; ++i) {
        t = model->rho[i];
        *ddata = (t != 0) ? -t : 0;  // Set -rho[i], avoiding -0.0
        ++ddata;
    }
}

/*
 * Copy support vector indices from the model to the data array.
 */
void copy_support (char *data, struct svm_csr_model *model)
{
    memcpy (data, model->sv_ind, (model->l) * sizeof(int));
}

/*
 * Copy support vector coefficients from the model to the data array.
 * This function handles the conversion from double ** to double *.
 */
void copy_sv_coef(char *data, struct svm_csr_model *model)
{
    int i, len = model->nr_class-1;
    double *temp = (double *) data;
    for(i=0; i<len; ++i) {
        memcpy(temp, model->sv_coef[i], sizeof(double) * model->l);
        temp += model->l;
    }
}

/*
 * Copy the number of iterations from the model to the data array.
 * Computes the number of models and copies the iterations accordingly.
 */
void copy_n_iter(char *data, struct svm_csr_model *model)
{
    const int n_models = MAX(1, model->nr_class * (model->nr_class-1) / 2);
    memcpy(data, model->n_iter, n_models * sizeof(int));
}

/*
 * Get the number of support vectors in the model.
 */
Py_ssize_t get_l(struct svm_csr_model *model)
{
    return (Py_ssize_t) model->l;
}

/*
 * Copy the number of support vectors per class from the model to the data array.
 * This function checks if labels exist before copying.
 */
void copy_nSV(char *data, struct svm_csr_model *model)
{
    if (model->label == NULL) return;
    memcpy(data, model->nSV, model->nr_class * sizeof(int));
}

/*
 * Copy labels from the model to the data array.
 * This function checks if labels exist before copying.
 */
void copy_label(char *data, struct svm_csr_model *model)
{
    if (model->label == NULL) return;
    memcpy(data, model->label, model->nr_class * sizeof(int));
}

/*
 * Copy probA values from the model to the data array.
 */
void copy_probA(char *data, struct svm_csr_model *model, Py_ssize_t * dims)
{
    memcpy(data, model->probA, dims[0] * sizeof(double));
}

/*
 * Copy probB values from the model to the data array.
 */
void copy_probB(char *data, struct svm_csr_model *model, Py_ssize_t * dims)
{
    memcpy(data, model->probB, dims[0] * sizeof(double));
}
/* 释放 svm_csr_problem 结构体指针所指向的内存空间 */
int free_problem(struct svm_csr_problem *problem)
{
    int i;
    /* 如果 problem 为空指针，则直接返回 -1 */
    if (problem == NULL) return -1;
    /* 释放 problem 结构体中 x 数组中每个元素指向的内存空间 */
    for (i=0; i<problem->l; ++i)
        free (problem->x[i]);
    /* 释放 problem 结构体中 x 数组的内存空间 */
    free (problem->x);
    /* 释放 problem 结构体指针指向的内存空间 */
    free (problem);
    return 0;
}

/* 释放 svm_csr_model 结构体指针所指向的内存空间 */
int free_model(struct svm_csr_model *model)
{
    /* 类似于 svm_free_and_destroy_model，但不释放 sv_coef[i] */
    /* 不释放 n_iter，因为它们在 set_model 中未被创建 */
    /* 如果 model 为空指针，则直接返回 -1 */
    if (model == NULL) return -1;
    /* 释放 model 结构体中 SV 数组的内存空间 */
    free(model->SV);
    /* 释放 model 结构体中 sv_coef 数组的内存空间 */
    free(model->sv_coef);
    /* 释放 model 结构体中 rho 数组的内存空间 */
    free(model->rho);
    /* 释放 model 结构体中 label 数组的内存空间 */
    free(model->label);
    /* 释放 model 结构体中 probA 数组的内存空间 */
    free(model->probA);
    /* 释放 model 结构体中 probB 数组的内存空间 */
    free(model->probB);
    /* 释放 model 结构体中 nSV 数组的内存空间 */
    free(model->nSV);
    /* 释放 model 结构体指针指向的内存空间 */
    free(model);

    return 0;
}

/* 释放 svm_parameter 结构体指针所指向的内存空间 */
int free_param(struct svm_parameter *param)
{
    /* 如果 param 为空指针，则直接返回 -1 */
    if (param == NULL) return -1;
    /* 释放 param 结构体指针指向的内存空间 */
    free(param);
    return 0;
}

/* 释放 svm_csr_model 结构体中 SV 和 sv_coef 数组的内存空间 */
int free_model_SV(struct svm_csr_model *model)
{
    int i;
    /* 逆序释放 model 结构体中 SV 数组中每个元素指向的内存空间 */
    for (i=model->l-1; i>=0; --i) free(model->SV[i]);
    /* svm_destroy_model 会释放 model 结构体中 SV 数组的内存空间 */
    /* 释放 model 结构体中 sv_coef 数组中每个元素指向的内存空间 */
    for (i=0; i < model->nr_class-1 ; ++i) free(model->sv_coef[i]);
    /* svm_destroy_model 会释放 model 结构体中 sv_coef 数组的内存空间 */
    return 0;
}

/* 从原始 libsvm 代码中借用的静态函数，打印空内容 */
static void print_null(const char *s) {}

/* 将字符串 s 输出到标准输出 */
static void print_string_stdout(const char *s)
{
    /* 将字符串 s 输出到标准输出 */
    fputs(s,stdout);
    /* 刷新标准输出缓冲区 */
    fflush(stdout);
}

/* 提供便利包装函数，设置输出的详细程度 */
void set_verbosity(int verbosity_flag){
    /* 如果 verbosity_flag 为真，则设置输出函数为 print_string_stdout */
    if (verbosity_flag)
        svm_set_print_string_function(&print_string_stdout);
    /* 如果 verbosity_flag 为假，则设置输出函数为 print_null */
    else
        svm_set_print_string_function(&print_null);
}
```