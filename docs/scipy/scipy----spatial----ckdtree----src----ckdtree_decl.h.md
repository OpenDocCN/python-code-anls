# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\ckdtree_decl.h`

```
#ifndef CKDTREE_CPP_DECL
#define CKDTREE_CPP_DECL

/*
 * Use numpy to provide some platform independency.
 * Define these functions for your platform
 * */
#include <numpy/npy_common.h>  // 包含 numpy 的通用头文件
#include <cmath>  // 包含数学函数库

#define CKDTREE_LIKELY(x) NPY_LIKELY(x)  // 定义概率大的情况
#define CKDTREE_UNLIKELY(x)  NPY_UNLIKELY(x)  // 定义概率小的情况
#define CKDTREE_PREFETCH(x, rw, loc)  NPY_PREFETCH(x, rw, loc)  // 定义预取数据的宏

#define ckdtree_intp_t npy_intp  // 定义 ckdtree_intp_t 为 numpy 的整数类型
#define ckdtree_fmin(x, y)   fmin(x, y)  // 定义 ckdtree_fmin 为 fmin 函数的别名
#define ckdtree_fmax(x, y)   fmax(x, y)  // 定义 ckdtree_fmax 为 fmax 函数的别名
#define ckdtree_fabs(x)   fabs(x)  // 定义 ckdtree_fabs 为 fabs 函数的别名

#include "ordered_pair.h"  // 包含 ordered_pair.h 头文件
#include "coo_entries.h"  // 包含 coo_entries.h 头文件

struct ckdtreenode {
    ckdtree_intp_t      split_dim;  // 分割维度
    ckdtree_intp_t      children;  // 子节点数
    double   split;  // 分割值
    ckdtree_intp_t      start_idx;  // 起始索引
    ckdtree_intp_t      end_idx;  // 结束索引
    ckdtreenode   *less;  // 指向比当前节点小的子节点的指针
    ckdtreenode   *greater;  // 指向比当前节点大的子节点的指针
    ckdtree_intp_t      _less;  // 小于分割值的节点个数
    ckdtree_intp_t      _greater;  // 大于等于分割值的节点个数
};

struct ckdtree {
    // tree structure
    std::vector<ckdtreenode>  *tree_buffer;  // 树的缓冲区
    ckdtreenode   *ctree;  // 树的根节点
    // meta data
    double   *raw_data;  // 原始数据指针
    ckdtree_intp_t      n;  // 数据集大小
    ckdtree_intp_t      m;  // 维度大小
    ckdtree_intp_t      leafsize;  // 叶子节点大小
    double   *raw_maxes;  // 数据最大值指针
    double   *raw_mins;  // 数据最小值指针
    ckdtree_intp_t      *raw_indices;  // 原始索引指针
    double   *raw_boxsize_data;  // 包围盒大小数据指针
    ckdtree_intp_t size;  // 树的大小
};

/* Build methods in C++ for better speed and GIL release */

int
build_ckdtree(ckdtree *self, ckdtree_intp_t start_idx, intptr_t end_idx,
              double *maxes, double *mins, int _median, int _compact);
// 构建 ckdtree 的方法声明

int
build_weights (ckdtree *self, double *node_weights, double *weights);
// 构建权重的方法声明

/* Query methods in C++ for better speed and GIL release */

int
query_knn(const ckdtree     *self,
          double       *dd,
          ckdtree_intp_t          *ii,
          const double *xx,
          const ckdtree_intp_t     n,
          const ckdtree_intp_t     *k,
          const ckdtree_intp_t     nk,
          const ckdtree_intp_t     kmax,
          const double  eps,
          const double  p,
          const double  distance_upper_bound);
// k 最近邻查询方法声明

int
query_pairs(const ckdtree *self,
            const double r,
            const double p,
            const double eps,
            std::vector<ordered_pair> *results);
// 查询对方法声明

int
count_neighbors_unweighted(const ckdtree *self,
                const ckdtree *other,
                ckdtree_intp_t n_queries,
                double *real_r,
                ckdtree_intp_t *results,
                const double p,
                int cumulative);
// 计算未加权邻居个数方法声明

int
count_neighbors_weighted(const ckdtree *self,
                const ckdtree *other,
                double *self_weights,
                double *other_weights,
                double *self_node_weights,
                double *other_node_weights,
                ckdtree_intp_t n_queries,
                double *real_r,
                double *results,
                const double p,
                int cumulative);
// 计算加权邻居个数方法声明

int
#endif
// 定义函数 query_ball_point，用于在给定的 CKD 树中查找球形范围内的点
query_ball_point(const ckdtree *self,  // CKD 树对象指针，表示要查询的树
                 const double *x,      // 查询点的坐标数组指针
                 const double *r,      // 球形范围的半径数组指针
                 const double p,       // 距离度量的 p 值
                 const double eps,     // 查询中使用的容差值
                 const ckdtree_intp_t n_queries,  // 查询点的数量
                 std::vector<ckdtree_intp_t> *results,  // 存储查询结果的向量指针
                 const bool return_length,  // 是否返回结果长度的标志
                 const bool sort_output);  // 是否对输出进行排序的标志

// 定义函数 query_ball_tree，用于在两棵 CKD 树之间查询球形范围内的点
int
query_ball_tree(const ckdtree *self,      // 第一棵 CKD 树对象指针，表示查询的主树
                const ckdtree *other,     // 第二棵 CKD 树对象指针，表示查询的目标树
                const double r,           // 球形范围的半径
                const double p,           // 距离度量的 p 值
                const double eps,         // 查询中使用的容差值
                std::vector<ckdtree_intp_t> *results  // 存储查询结果的向量指针
                );

// 定义函数 sparse_distance_matrix，用于计算两棵 CKD 树之间稀疏距离矩阵
int
sparse_distance_matrix(const ckdtree *self,        // 第一棵 CKD 树对象指针，表示计算距离的主树
                       const ckdtree *other,       // 第二棵 CKD 树对象指针，表示计算距离的目标树
                       const double p,             // 距离度量的 p 值
                       const double max_distance,  // 允许的最大距离
                       std::vector<coo_entry> *results  // 存储计算结果的 COO 格式向量指针
                       );

// 结束文件的条件预处理指令
#endif
```