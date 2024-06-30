# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\query_ball_point.cxx`

```
// 包含头文件 "ckdtree_decl.h" 和 "rectangle.h"
#include "ckdtree_decl.h"
#include "rectangle.h"

// 包含 C++ 标准库的头文件
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

// 定义静态函数 traverse_no_checking，用于深度优先遍历 kd 树节点，不进行距离检查
static void
traverse_no_checking(const ckdtree *self,
                     const int return_length,
                     std::vector<ckdtree_intp_t> &results,
                     const ckdtreenode *node)
{
    // 获取 kd 树的原始索引数组
    const ckdtree_intp_t *indices = self->raw_indices;
    const ckdtreenode *lnode;
    ckdtree_intp_t i;

    // 如果当前节点是叶子节点
    if (node->split_dim == -1) {  /* leaf node */
        lnode = node;
        const ckdtree_intp_t start = lnode->start_idx;
        const ckdtree_intp_t end = lnode->end_idx;
        // 遍历叶子节点中的索引范围
        for (i = start; i < end; ++i) {
            // 如果需要返回长度信息
            if (return_length) {
                results[0] ++;
            } else {  // 否则将索引值加入结果向量
                results.push_back(indices[i]);
            }
        }
    }
    else {
        // 递归遍历左右子节点
        traverse_no_checking(self, return_length, results, node->less);
        traverse_no_checking(self, return_length, results, node->greater);
    }
}

// 定义模板函数 traverse_checking，用于检查距离后深度优先遍历 kd 树节点
template <typename MinMaxDist> static void
traverse_checking(const ckdtree *self,
                  const int return_length,
                  std::vector<ckdtree_intp_t> &results,
                  const ckdtreenode *node,
                  RectRectDistanceTracker<MinMaxDist> *tracker
)
{
    const ckdtreenode *lnode;
    double d;
    ckdtree_intp_t i;

    // 如果当前节点的最小距离大于上限乘以容差因子，直接返回
    if (tracker->min_distance > tracker->upper_bound * tracker->epsfac) {
        return;
    }
    // 如果当前节点的最大距离小于上限除以容差因子，直接遍历该节点
    else if (tracker->max_distance < tracker->upper_bound / tracker->epsfac) {
        traverse_no_checking(self, return_length, results, node);
    }
    // 如果当前节点是叶子节点
    else if (node->split_dim == -1)  { /* leaf node */

        /* brute-force */
        lnode = node;
        const double p = tracker->p;
        const double tub = tracker->upper_bound;
        const double *tpt = tracker->rect1.mins();
        const double *data = self->raw_data;
        const ckdtree_intp_t *indices = self->raw_indices;
        const ckdtree_intp_t m = self->m;
        const ckdtree_intp_t start = lnode->start_idx;
        const ckdtree_intp_t end = lnode->end_idx;

        // 预取叶子节点中的数据，提高访存效率
        CKDTREE_PREFETCH(data + indices[start] * m, 0, m);
        if (start < end - 1)
            CKDTREE_PREFETCH(data + indices[start+1] * m, 0, m);

        // 遍历叶子节点中的索引范围
        for (i = start; i < end; ++i) {

            if (i < end - 2)
                CKDTREE_PREFETCH(data + indices[i+2] * m, 0, m);

            // 计算当前点到目标点的距离
            d = MinMaxDist::point_point_p(self, data + indices[i] * m, tpt, p, m, tub);

            // 如果距离小于等于上限，根据需要返回长度信息或者将索引值加入结果向量
            if (d <= tub) {
                if(return_length) {
                    results[0] ++;
                } else {
                    results.push_back((ckdtree_intp_t) indices[i]);
                }
            }
        }
    }
}
    // 如果当前节点不是叶子节点，则执行以下操作
    else {
        // 将当前节点标记为“小于”的路径，深度为2，并推入路径跟踪器
        tracker->push_less_of(2, node);
        // 递归遍历左子树，检查是否有符合条件的节点，并更新结果集
        traverse_checking(self, return_length, results, node->less, tracker);
        // 弹出“小于”的路径标记，恢复路径跟踪器状态

        tracker->pop();

        // 将当前节点标记为“大于”的路径，深度为2，并推入路径跟踪器
        tracker->push_greater_of(2, node);
        // 递归遍历右子树，检查是否有符合条件的节点，并更新结果集
        traverse_checking(self, return_length, results, node->greater, tracker);
        // 弹出“大于”的路径标记，恢复路径跟踪器状态
        tracker->pop();
    }
}



int
query_ball_point(const ckdtree *self, const double *x,
                 const double *r, const double p, const double eps,
                 const ckdtree_intp_t n_queries,
                 std::vector<ckdtree_intp_t> *results,
                 const bool return_length,
                 const bool sort_output)
{
#define HANDLE(cond, kls) \
    if(cond) { \
        if(return_length) results[i].push_back(0); \
        RectRectDistanceTracker<kls> tracker(self, point, rect, p, eps, r[i]); \
        traverse_checking(self, return_length, results[i], self->ctree, &tracker); \
    } else

    for (ckdtree_intp_t i=0; i < n_queries; ++i) {
        const ckdtree_intp_t m = self->m;
        Rectangle rect(m, self->raw_mins, self->raw_maxes);
        if (CKDTREE_LIKELY(self->raw_boxsize_data == NULL)) {
            Rectangle point(m, x + i * m, x + i * m);
            // 处理不同距离度量的情况
            HANDLE(CKDTREE_LIKELY(p == 2), MinkowskiDistP2)
            HANDLE(p == 1, MinkowskiDistP1)
            HANDLE(std::isinf(p), MinkowskiDistPinf)
            HANDLE(1, MinkowskiDistPp)
            {}  // 结束 HANDLE 宏的代码块
        } else {
            Rectangle point(m, x + i * m, x + i * m);
            int j;
            // 处理带盒子尺寸的情况，调整点的边界
            for(j=0; j<m; ++j) {
                point.maxes()[j] = point.mins()[j] = BoxDist1D::wrap_position(point.mins()[j], self->raw_boxsize_data[j]);
            }
            // 处理不同距离度量的情况
            HANDLE(CKDTREE_LIKELY(p == 2), BoxMinkowskiDistP2)
            HANDLE(p == 1, BoxMinkowskiDistP1)
            HANDLE(std::isinf(p), BoxMinkowskiDistPinf)
            HANDLE(1, BoxMinkowskiDistPp)
            {}  // 结束 HANDLE 宏的代码块
        }

        // 如果不需要返回长度但需要排序输出结果，则对结果向量进行排序
        if (!return_length && sort_output) {
            std::sort(results[i].begin(), results[i].end());
        }
    }
    return 0;
}
```