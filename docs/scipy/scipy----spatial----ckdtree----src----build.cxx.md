# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\build.cxx`

```
#include "ckdtree_decl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <new>
#include <typeinfo>
#include <stdexcept>
#include <ios>

#define tree_buffer_root(buf) (&(buf)[0][0])

static ckdtree_intp_t
build(ckdtree *self, ckdtree_intp_t start_idx, intptr_t end_idx,
      double *maxes, double *mins,
      const int _median, const int _compact)
{
    // 获取数据集维度
    const ckdtree_intp_t m = self->m;
    // 获取原始数据指针
    const double *data = self->raw_data;
    // 获取索引数组指针
    ckdtree_intp_t *indices = (intptr_t *)(self->raw_indices);

    // 新建一个节点，并初始化为零
    ckdtreenode new_node = {}, *n, *root;
    // 节点索引、左子节点索引、右子节点索引
    ckdtree_intp_t node_index, _less, _greater;
    // 循环变量和临时变量
    ckdtree_intp_t i, j, p, d;
    // 节点大小、分割值、最小值、最大值
    double size, split, minval, maxval;

    /* 将一个新节点推入节点堆栈 */
    self->tree_buffer->push_back(new_node);
    // 计算新节点在节点堆栈中的索引
    node_index = self->tree_buffer->size() - 1;
    // 获取根节点指针
    root = tree_buffer_root(self->tree_buffer);
    // 获取当前节点指针
    n = root + node_index;
    // 将当前节点的内存初始化为零
    memset(n, 0, sizeof(n[0]));

    // 设置当前节点的起始和结束索引，以及子节点数量
    n->start_idx = start_idx;
    n->end_idx = end_idx;
    n->children = end_idx - start_idx;

    if (end_idx - start_idx <= self->leafsize) {
        /* 如果节点包含的数据量小于等于叶子节点限制，则返回叶子节点 */
        n->split_dim = -1;
        return node_index;
    }
}

int build_ckdtree(ckdtree *self, ckdtree_intp_t start_idx, intptr_t end_idx,
                  double *maxes, double *mins, int _median, int _compact)
{
    // 调用build函数构建kd树
    build(self, start_idx, end_idx, maxes, mins, _median, _compact);
    return 0;
}

static double
add_weights(ckdtree *self,
            double *node_weights,
            ckdtree_intp_t node_index,
            double *weights)
{
    // 获取索引数组指针
    ckdtree_intp_t *indices = (intptr_t *)(self->raw_indices);

    // 获取根节点指针
    ckdtreenode *n, *root;
    root = tree_buffer_root(self->tree_buffer);
    // 获取当前节点指针
    n = root + node_index;

    double sum = 0;

    if (n->split_dim != -1) {
        /* 如果当前节点不是叶子节点，则递归计算左右子节点的权重总和 */
        double left, right;
        left = add_weights(self, node_weights, n->_less, weights);
        right = add_weights(self, node_weights, n->_greater, weights);
        sum = left + right;
    } else {
        ckdtree_intp_t i;

        /* 如果当前节点是叶子节点，则计算叶子节点中索引对应权重的总和 */
        for (i = n->start_idx; i < n->end_idx; ++i) {
            sum += weights[indices[i]];
        }
    }

    // 将当前节点的权重总和存入节点权重数组
    node_weights[node_index] = sum;
    return sum;
}

int build_weights(ckdtree *self, double *node_weights, double *weights)
{
    // 调用add_weights函数计算所有节点的权重总和
    add_weights(self, node_weights, 0, weights);
    return 0;
}
```