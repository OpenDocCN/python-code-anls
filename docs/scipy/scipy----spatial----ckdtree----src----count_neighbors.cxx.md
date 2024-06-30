# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\count_neighbors.cxx`

```
/*
 * 包含必要的头文件声明
 */
#include "ckdtree_decl.h"
#include "rectangle.h"

/*
 * 包含标准库头文件
 */
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <new>
#include <typeinfo>
#include <stdexcept>
#include <ios>

/*
 * 定义结构体 WeightedTree，包含指向 ckdtree 的指针和权重数组
 */
struct WeightedTree {
    const ckdtree *tree;   // 指向 ckdtree 的指针
    double *weights;       // 权重数组
    double *node_weights;  // 节点权重数组
};

/*
 * 定义结构体 CNBParams，包含参数和 WeightedTree 结构体的实例
 */
struct CNBParams
{
    double *r;             // double 类型指针
    void * results;        /* 将在内部强制转换 */
    WeightedTree self, other;  // WeightedTree 结构体实例
    int cumulative;        // 累积计数器
};

/*
 * 模板函数 traverse，用于遍历节点并处理距离计算
 */
template <typename MinMaxDist, typename WeightType, typename ResultType> static void
traverse(
    RectRectDistanceTracker<MinMaxDist> *tracker,   // RectRectDistanceTracker 类型指针
    const CNBParams *params,                       // CNBParams 类型指针
    double *start, double *end,                    // 双精度浮点数指针
    const ckdtreenode *node1,                      // ckdtreenode 类型指针
    const ckdtreenode *node2)                      // ckdtreenode 类型指针
{
    static void (* const next)(RectRectDistanceTracker<MinMaxDist> *tracker,
            const CNBParams *params,
            double *start, double *end,
            const ckdtreenode *node1,
            const ckdtreenode *node2) = traverse<MinMaxDist, WeightType, ResultType>;

    ResultType *results = (ResultType*) params->results;  // 将 params->results 强制转换为 ResultType 指针

    /*
     * 通过子节点距离追踪器加速处理距离较近的节点对，查看是否有未完成的工作
     */
    double * new_start = std::lower_bound(start, end, tracker->min_distance);  // 在[start, end)范围内查找第一个不小于 tracker->min_distance 的元素
    double * new_end = std::lower_bound(start, end, tracker->max_distance);    // 在[start, end)范围内查找第一个不小于 tracker->max_distance 的元素

    /* 
     * 由于 max_distance >= min_distance，end < start 永不发生
     */
    if (params->cumulative) {
        double * i;
        if (new_end != end) {
            ResultType nn = WeightType::get_weight(&params->self, node1)
                          * WeightType::get_weight(&params->other, node2);

            for (i = new_end; i < end; ++i) {
                results[i - params->r] += nn;
            }
        }
        /*
         * 任何大于 end 的箱子已经被正确计算，因此我们可以在此分支的遍历未来截断查询
         */
        start = new_start;
        end = new_end;
    } else {
        start = new_start;
        end = new_end;

        if (end == start) {
            ResultType nn = WeightType::get_weight(&params->self, node1)
                          * WeightType::get_weight(&params->other, node2);
            results[start - params->r] += nn;
        }
    }

    if (end == start) {
        /*
         * 此对节点恰好落入一个箱子，无需进一步探测
         */
        return;
    }

    /*
     * 需要深入探测一些
     */
}
    else { /* 1 is an inner node */
        // 如果节点1是内部节点

        if (node2->split_dim == -1) {
            // 如果节点2是叶子节点

            /* 1 is an inner node, 2 is a leaf node */
            // 节点1是内部节点，节点2是叶子节点

            // 将节点1的“小于”路径推入跟踪器
            tracker->push_less_of(1, node1);
            // 递归处理，找出满足条件的节点组合
            next(tracker, params, start, end, node1->less, node2);
            // 弹出节点1的路径信息
            tracker->pop();

            // 将节点1的“大于”路径推入跟踪器
            tracker->push_greater_of(1, node1);
            // 递归处理，找出满足条件的节点组合
            next(tracker, params, start, end, node1->greater, node2);
            // 弹出节点1的路径信息
            tracker->pop();
        }
        else { /* 1 and 2 are inner nodes */
            // 否则，节点1和节点2都是内部节点

            // 将节点1的“小于”路径推入跟踪器
            tracker->push_less_of(1, node1);
            // 将节点2的“小于”路径推入跟踪器
            tracker->push_less_of(2, node2);
            // 递归处理，找出满足条件的节点组合
            next(tracker, params, start, end, node1->less, node2->less);
            // 弹出节点2的路径信息
            tracker->pop();

            // 将节点2的“大于”路径推入跟踪器
            tracker->push_greater_of(2, node2);
            // 递归处理，找出满足条件的节点组合
            next(tracker, params, start, end, node1->less, node2->greater);
            // 弹出节点2的路径信息
            tracker->pop();
            // 弹出节点1的路径信息
            tracker->pop();

            // 将节点1的“大于”路径推入跟踪器
            tracker->push_greater_of(1, node1);
            // 将节点2的“小于”路径推入跟踪器
            tracker->push_less_of(2, node2);
            // 递归处理，找出满足条件的节点组合
            next(tracker, params, start, end, node1->greater, node2->less);
            // 弹出节点2的路径信息
            tracker->pop();

            // 将节点2的“大于”路径推入跟踪器
            tracker->push_greater_of(2, node2);
            // 递归处理，找出满足条件的节点组合
            next(tracker, params, start, end, node1->greater, node2->greater);
            // 弹出节点2的路径信息
            tracker->pop();
            // 弹出节点1的路径信息
            tracker->pop();
        }
    }
}

template <typename WeightType, typename ResultType> void
count_neighbors(struct CNBParams *params,
                ckdtree_intp_t n_queries, const double p)
{
    // 获取当前对象的 k-d 树
    const ckdtree *self = params->self.tree;
    // 获取其他对象的 k-d 树
    const ckdtree *other = params->other.tree;

#define HANDLE(cond, kls) \
    if (cond) { \
        // 创建距离追踪器对象
        RectRectDistanceTracker<kls> tracker(self, r1, r2, p, 0.0, 0.0);\
        // 遍历并计算结果
        traverse<kls, WeightType, ResultType>(&tracker, params, params->r, params->r+n_queries, \
                 self->ctree, other->ctree); \
    } else

    // 创建两个矩形对象
    Rectangle r1(self->m, self->raw_mins, self->raw_maxes);
    Rectangle r2(other->m, other->raw_mins, other->raw_maxes);

    // 检查是否需要使用原始距离或者盒子化距离
    if (CKDTREE_LIKELY(self->raw_boxsize_data == NULL)) {
        // 处理欧氏距离的情况
        HANDLE(CKDTREE_LIKELY(p == 2), MinkowskiDistP2)
        // 处理曼哈顿距离的情况
        HANDLE(p == 1, MinkowskiDistP1)
        // 处理无穷范数距离的情况
        HANDLE(std::isinf(p), MinkowskiDistPinf)
        // 处理通用 Minkowski 距离的情况
        HANDLE(1, MinkowskiDistPp)
        {}
    } else {
        // 处理盒子化情况下的欧氏距离
        HANDLE(CKDTREE_LIKELY(p == 2), BoxMinkowskiDistP2)
        // 处理盒子化情况下的曼哈顿距离
        HANDLE(p == 1, BoxMinkowskiDistP1)
        // 处理盒子化情况下的无穷范数距离
        HANDLE(std::isinf(p), BoxMinkowskiDistPinf)
        // 处理盒子化情况下的通用 Minkowski 距离
        HANDLE(1, BoxMinkowskiDistPp)
        {}
    }
}

struct Unweighted {
    /* the interface for accessing weights of unweighted data. */
    static inline ckdtree_intp_t
    get_weight(const WeightedTree *wt, const ckdtreenode * node)
    {
        // 返回节点的子节点数量
        return node->children;
    }
    static inline ckdtree_intp_t
    get_weight(const WeightedTree *wt, const ckdtree_intp_t i)
    {
        // 返回权重为 1
        return 1;
    }
};

int
count_neighbors_unweighted(const ckdtree *self, const ckdtree *other,
                ckdtree_intp_t n_queries, double *real_r, intptr_t *results,
                const double p, int cumulative) {

    CNBParams params = {0};

    // 设置参数对象
    params.r = real_r;
    params.results = (void*) results;
    params.self.tree = self;
    params.other.tree = other;
    params.cumulative = cumulative;

    // 调用通用计算邻居数量的函数
    count_neighbors<Unweighted, ckdtree_intp_t>(&params, n_queries, p);

    // 返回 0 表示执行成功
    return 0;
}

struct Weighted {
    /* the interface for accessing weights of weighted data. */
    static inline double
    get_weight(const WeightedTree *wt, const ckdtreenode * node)
    {
        // 如果权重数组存在，则返回节点对应的权重，否则返回子节点数量
        return (wt->weights != NULL)
           ? wt->node_weights[node - wt->tree->ctree]
           : node->children;
    }
    static inline double
    get_weight(const WeightedTree *wt, const ckdtree_intp_t i)
    {
        // 如果权重数组存在，则返回第 i 个节点的权重，否则返回 1
        return (wt->weights != NULL)? wt->weights[i] : 1;
    }
};

int
count_neighbors_weighted(const ckdtree *self, const ckdtree *other,
                double *self_weights, double *other_weights,
                double *self_node_weights, double *other_node_weights,
                ckdtree_intp_t n_queries, double *real_r, double *results,
                const double p, int cumulative)
{

    CNBParams params = {0};

    // 设置参数对象
    params.r = real_r;
    params.results = (void*) results;
    params.cumulative = cumulative;

    params.self.tree = self;
    params.other.tree = other;
    # 如果 self_weights 参数不为空，则将其赋值给 params.self.weights
    if (self_weights) {
        params.self.weights = self_weights;
        # 如果 self_node_weights 参数不为空，则将其赋值给 params.self.node_weights
        params.self.node_weights = self_node_weights;
    }
    
    # 如果 other_weights 参数不为空，则将其赋值给 params.other.weights
    if (other_weights) {
        params.other.weights = other_weights;
        # 如果 other_node_weights 参数不为空，则将其赋值给 params.other.node_weights
        params.other.node_weights = other_node_weights;
    }

    # 调用 count_neighbors 函数，传入参数 params、n_queries、p，返回结果
    count_neighbors<Weighted, double>(&params, n_queries, p);

    # 返回整数 0 表示函数正常结束
    return 0;
}



# 这行代码表示一个代码块的结束，结束了一个函数或者条件语句的定义或执行。
```