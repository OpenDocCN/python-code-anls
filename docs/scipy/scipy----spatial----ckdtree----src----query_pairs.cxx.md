# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\query_pairs.cxx`

```
static void
traverse_no_checking(const ckdtree *self,
                     std::vector<ordered_pair> *results,
                     const ckdtreenode *node1, const ckdtreenode *node2)
{
    // 声明局部变量
    const ckdtreenode *lnode1;
    const ckdtreenode *lnode2;
    ckdtree_intp_t i, j, min_j;
    // 获取 KD 树节点的原始索引数组
    const ckdtree_intp_t *indices = self->raw_indices;

    // 如果 node1 是叶子节点
    if (node1->split_dim == -1) { /* leaf node */
        lnode1 = node1;

        // 如果 node2 也是叶子节点
        if (node2->split_dim == -1) { /* leaf node */
            lnode2 = node2;

            // 获取叶子节点的起始和结束索引
            const ckdtree_intp_t start1 = lnode1->start_idx;
            const ckdtree_intp_t start2 = lnode2->start_idx;
            const ckdtree_intp_t end1 = lnode1->end_idx;
            const ckdtree_intp_t end2 = lnode2->end_idx;

            // 遍历节点1和节点2之间的索引对
            for (i = start1; i < end1; ++i) {

                /* Special care here to avoid duplicate pairs */
                // 避免重复的索引对
                if (node1 == node2)
                    min_j = i + 1;
                else
                    min_j = start2;

                // 添加有序索引对到结果向量中
                for (j = min_j; j < end2; ++j)
                    add_ordered_pair(results, indices[i], indices[j]);
            }
        }
        else {
            // 递归遍历节点1和节点2的子节点
            traverse_no_checking(self, results, node1, node2->less);
            traverse_no_checking(self, results, node1, node2->greater);
        }
    }
    else {
        // 如果 node1 不是叶子节点
        if (node1 == node2) {
            /*
             * Avoid traversing (node1->less, node2->greater) and
             * (node1->greater, node2->less) (it's the same node pair twice
             * over, which is the source of the complication in the
             * original KDTree.query_pairs)
             */
            // 避免遍历相同的节点对，避免查询中的复杂性源
            traverse_no_checking(self, results, node1->less, node2->less);
            traverse_no_checking(self, results, node1->less, node2->greater);
            traverse_no_checking(self, results, node1->greater, node2->greater);
        }
        else {
            // 递归遍历节点1和节点2的子节点
            traverse_no_checking(self, results, node1->less, node2);
            traverse_no_checking(self, results, node1->greater, node2);
        }
    }
}


template <typename MinMaxDist> static void
traverse_checking(const ckdtree *self,
                  std::vector<ordered_pair> *results,
                  const ckdtreenode *node1, const ckdtreenode *node2,
                  RectRectDistanceTracker<MinMaxDist> *tracker)
{
    // 声明局部变量
    const ckdtreenode *lnode1;
    const ckdtreenode *lnode2;
    double d;
    ckdtree_intp_t i, j, min_j;

    // 如果当前节点对的最小距离大于阈值，则直接返回
    if (tracker->min_distance > tracker->upper_bound * tracker->epsfac)
        return;
    // 如果当前节点对的最大距离小于阈值，则调用无检查遍历函数
    else if (tracker->max_distance < tracker->upper_bound / tracker->epsfac)
        traverse_no_checking(self, results, node1, node2);
    else if (node1->split_dim == -1) { /* 1 is leaf node */
        lnode1 = node1;

        if (node2->split_dim == -1) {  /* 1 & 2 are leaves */
            lnode2 = node2;

            /* brute-force */
            const double p = tracker->p;  // 获取跟踪器中的参数 p
            const double tub = tracker->upper_bound;  // 获取跟踪器中的上界 tub
            const double *data = self->raw_data;  // 获取原始数据数组指针
            const double epsfac = tracker->epsfac;  // 获取跟踪器中的 epsfac
            const ckdtree_intp_t *indices = self->raw_indices;  // 获取原始索引数组指针
            const ckdtree_intp_t m = self->m;  // 获取数据向量的维度
            const ckdtree_intp_t start1 = lnode1->start_idx;  // 获取第一个叶子节点起始索引
            const ckdtree_intp_t start2 = lnode2->start_idx;  // 获取第二个叶子节点起始索引
            const ckdtree_intp_t end1 = lnode1->end_idx;  // 获取第一个叶子节点结束索引
            const ckdtree_intp_t end2 = lnode2->end_idx;  // 获取第二个叶子节点结束索引

            CKDTREE_PREFETCH(data+indices[start1]*m, 0, m);  // 预取第一个叶子节点的第一个数据点
            if (start1 < end1 - 1)
               CKDTREE_PREFETCH(data+indices[start1+1]*m, 0, m);  // 如果有下一个数据点，预取

            for(i = start1; i < end1; ++i) {  // 循环遍历第一个叶子节点的索引范围

                if (i < end1 - 2)
                     CKDTREE_PREFETCH(data+indices[i+2]*m, 0, m);  // 如果有下两个数据点，预取

                /* Special care here to avoid duplicate pairs */
                if (node1 == node2)
                    min_j = i + 1;  // 如果节点1和节点2相同，从当前 i 后面开始
                else
                    min_j = start2;  // 否则从节点2的起始索引开始

                if (min_j < end2)
                    CKDTREE_PREFETCH(data+indices[min_j]*m, 0, m);  // 预取 min_j 索引对应的数据点
                if (min_j < end2 - 1)
                    CKDTREE_PREFETCH(data+indices[min_j+1]*m, 0, m);  // 预取 min_j+1 索引对应的数据点

                for (j = min_j; j < end2; ++j) {  // 循环遍历第二个叶子节点的索引范围

                    if (j < end2 - 2)
                        CKDTREE_PREFETCH(data+indices[j+2]*m, 0, m);  // 如果有下两个数据点，预取

                    d = MinMaxDist::point_point_p(
                            self,
                            data + indices[i] * m,
                            data + indices[j] * m,
                            p, m, tub);  // 计算数据点 i 和 j 之间的距离

                    if (d <= tub/epsfac)  // 如果距离小于等于 tub/epsfac，则将这对索引有序地添加到结果集中
                        add_ordered_pair(results, indices[i], indices[j]);
                }
            }
        }
        else {  /* 1 is a leaf node, 2 is inner node */
            tracker->push_less_of(2, node2);  // 将节点2的 less 分支推入跟踪器
            traverse_checking(self, results, node1, node2->less, tracker);  // 递归遍历检查节点1和节点2的 less 分支
            tracker->pop();  // 弹出跟踪器中的最后一个操作

            tracker->push_greater_of(2, node2);  // 将节点2的 greater 分支推入跟踪器
            traverse_checking(self, results, node1, node2->greater, tracker);  // 递归遍历检查节点1和节点2的 greater 分支
            tracker->pop();  // 弹出跟踪器中的最后一个操作
        }
    }
    else {  /* 1 is an inner node */
        // 如果节点 1 是内部节点
        if (node2->split_dim == -1) { /* 1 is an inner node, 2 is a leaf node */
            // 如果节点 1 是内部节点，节点 2 是叶子节点
            // 将节点 1 的 less 分支推入追踪器
            tracker->push_less_of(1, node1);
            // 递归遍历检查节点 1 的 less 分支和节点 2
            traverse_checking(self, results, node1->less, node2, tracker);
            // 从追踪器中弹出操作
            tracker->pop();

            // 将节点 1 的 greater 分支推入追踪器
            tracker->push_greater_of(1, node1);
            // 递归遍历检查节点 1 的 greater 分支和节点 2
            traverse_checking(self, results, node1->greater, node2, tracker);
            // 从追踪器中弹出操作
            tracker->pop();
        }
        else { /* 1 and 2 are inner nodes */
            // 如果节点 1 和节点 2 都是内部节点
            // 将节点 1 的 less 分支推入追踪器
            tracker->push_less_of(1, node1);
            // 将节点 2 的 less 分支推入追踪器
            tracker->push_less_of(2, node2);
            // 递归遍历检查节点 1 的 less 分支和节点 2 的 less 分支
            traverse_checking(self, results, node1->less, node2->less, tracker);
            // 从追踪器中弹出操作
            tracker->pop();

            // 将节点 2 的 greater 分支推入追踪器
            tracker->push_greater_of(2, node2);
            // 递归遍历检查节点 1 的 less 分支和节点 2 的 greater 分支
            traverse_checking(self, results, node1->less, node2->greater,
                tracker);
            // 从追踪器中弹出操作
            tracker->pop();
            // 从追踪器中弹出操作
            tracker->pop();

            // 将节点 1 的 greater 分支推入追踪器
            tracker->push_greater_of(1, node1);
            // 如果节点 1 不等于节点 2
            if (node1 != node2) {
                /*
                 * 避免遍历 (node1->less, node2->greater) 和 (node1->greater, node2->less)
                 * 这是同一个节点对两次遍历的源头，这是在原始 KDTree.query_pairs 中复杂性的来源
                 */
                // 将节点 2 的 less 分支推入追踪器
                tracker->push_less_of(2, node2);
                // 递归遍历检查节点 1 的 greater 分支和节点 2 的 less 分支
                traverse_checking(self, results, node1->greater, node2->less,
                    tracker);
                // 从追踪器中弹出操作
                tracker->pop();
            }
            // 将节点 2 的 greater 分支推入追踪器
            tracker->push_greater_of(2, node2);
            // 递归遍历检查节点 1 的 greater 分支和节点 2 的 greater 分支
            traverse_checking(self, results, node1->greater, node2->greater,
                tracker);
            // 从追踪器中弹出操作
            tracker->pop();
            // 从追踪器中弹出操作
            tracker->pop();
        }
    }
}



#include <iostream>

// 查询在给定条件下的 k-d 树中的配对，将结果存储在指定的向量中
int
query_pairs(const ckdtree *self,            // 给定的 k-d 树指针
            const double r,                 // 范围参数 r
            const double p,                 // 距离度量参数 p
            const double eps,               // 容忍度参数 eps
            std::vector<ordered_pair> *results)  // 存储查询结果的向量指针
{

#define HANDLE(cond, kls) \                // 定义处理宏，根据条件调用指定的距离跟踪类
    if(cond) { \                           // 如果条件成立
        RectRectDistanceTracker<kls> tracker(self, r1, r2, p, eps, r);\  // 创建距离跟踪器对象
        traverse_checking(self, results, self->ctree, self->ctree, \    // 调用遍历函数
            &tracker); \                    // 传入距离跟踪器对象
    } else                                 // 如果条件不成立
                                            // 什么都不做
    {}

    Rectangle r1(self->m, self->raw_mins, self->raw_maxes);  // 创建矩形 r1 对象
    Rectangle r2(self->m, self->raw_mins, self->raw_maxes);  // 创建矩形 r2 对象

    if(CKDTREE_LIKELY(self->raw_boxsize_data == NULL)) {    // 如果原始盒子大小数据为空
        HANDLE(CKDTREE_LIKELY(p == 2), MinkowskiDistP2)     // 处理欧几里得距离条件
        HANDLE(p == 1, MinkowskiDistP1)                    // 处理曼哈顿距离条件
        HANDLE(std::isinf(p), MinkowskiDistPinf)           // 处理无穷范数距离条件
        HANDLE(1, MinkowskiDistPp)                         // 处理一般 Minkowski 距离条件
        {}                                                 // 什么都不做
    } else {                                                // 如果原始盒子大小数据不为空
        HANDLE(CKDTREE_LIKELY(p == 2), BoxMinkowskiDistP2)  // 处理盒子内欧几里得距离条件
        HANDLE(p == 1, BoxMinkowskiDistP1)                 // 处理盒子内曼哈顿距离条件
        HANDLE(std::isinf(p), BoxMinkowskiDistPinf)        // 处理盒子内无穷范数距离条件
        HANDLE(1, BoxMinkowskiDistPp)                      // 处理盒子内一般 Minkowski 距离条件
        {}                                                 // 什么都不做
    }

    return 0;  // 返回成功状态
}
```