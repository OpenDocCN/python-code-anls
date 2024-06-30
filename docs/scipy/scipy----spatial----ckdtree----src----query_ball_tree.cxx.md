# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\query_ball_tree.cxx`

```
static void
traverse_no_checking(const ckdtree *self, const ckdtree *other,
                     std::vector<ckdtree_intp_t> *results,
                     const ckdtreenode *node1, const ckdtreenode *node2)
{
    // 定义局部变量，用于存储左右子节点
    const ckdtreenode *lnode1;
    const ckdtreenode *lnode2;
    // 获取 self 和 other 的原始索引数组
    const ckdtree_intp_t *sindices = self->raw_indices;
    const ckdtree_intp_t *oindices = other->raw_indices;
    ckdtree_intp_t i, j;

    // 如果 node1 是叶子节点
    if (node1->split_dim == -1) {   /* leaf node */
        lnode1 = node1;

        // 如果 node2 也是叶子节点
        if (node2->split_dim == -1) {  /* leaf node */
            lnode2 = node2;

            // 获取叶子节点的起始和结束索引
            const ckdtree_intp_t start1 = lnode1->start_idx;
            const ckdtree_intp_t start2 = lnode2->start_idx;
            const ckdtree_intp_t end1 = lnode1->end_idx;
            const ckdtree_intp_t end2 = lnode2->end_idx;

            // 遍历叶子节点之间的索引范围，将结果存储在 results 中
            for (i = start1; i < end1; ++i) {
                auto &results_i = results[sindices[i]];
                for (j = start2; j < end2; ++j)
                    results_i.push_back(oindices[j]);
            }
        }
        else {
            // 如果 node2 不是叶子节点，则递归调用 traverse_no_checking 处理左右子节点
            traverse_no_checking(self, other, results, node1, node2->less);
            traverse_no_checking(self, other, results, node1, node2->greater);
        }
    }
    else {
        // 如果 node1 不是叶子节点，则递归调用 traverse_no_checking 处理左右子节点
        traverse_no_checking(self, other, results, node1->less, node2);
        traverse_no_checking(self, other, results, node1->greater, node2);
    }
}


template <typename MinMaxDist> static void
traverse_checking(const ckdtree *self, const ckdtree *other,
                  std::vector<ckdtree_intp_t> *results,
                  const ckdtreenode *node1, const ckdtreenode *node2,
                  RectRectDistanceTracker<MinMaxDist> *tracker)
{
    const ckdtreenode *lnode1;
    const ckdtreenode *lnode2;
    double d;
    ckdtree_intp_t i, j;

    // 如果当前距离小于上限的一定比例，则直接返回，不继续遍历
    if (tracker->min_distance > tracker->upper_bound * tracker->epsfac)
        return;
    // 如果当前距离大于上限的一定比例，则直接返回，不继续遍历
    else if (tracker->max_distance < tracker->upper_bound / tracker->epsfac)
        traverse_no_checking(self, other, results, node1, node2);
    else if (node1->split_dim == -1) { /* 如果 node1 是叶子节点 */
        lnode1 = node1;  // 将 node1 赋值给 lnode1

        if (node2->split_dim == -1) {  /* 如果 node1 和 node2 都是叶子节点 */

            /* 暴力搜索 */
            lnode2 = node2;  // 将 node2 赋值给 lnode2
            const double p = tracker->p;  // 获取 tracker 中的 p 值
            const double tub = tracker->upper_bound;  // 获取 tracker 中的上界值
            const double tmd = tracker->max_distance;  // 获取 tracker 中的最大距离值
            const double *sdata = self->raw_data;  // 获取 self 对象的原始数据数组指针
            const ckdtree_intp_t *sindices = self->raw_indices;  // 获取 self 对象的原始索引数组指针
            const double *odata = other->raw_data;  // 获取 other 对象的原始数据数组指针
            const ckdtree_intp_t *oindices = other->raw_indices;  // 获取 other 对象的原始索引数组指针
            const ckdtree_intp_t m = self->m;  // 获取 self 对象的维度数
            const ckdtree_intp_t start1 = lnode1->start_idx;  // 获取 lnode1 的起始索引
            const ckdtree_intp_t start2 = lnode2->start_idx;  // 获取 lnode2 的起始索引
            const ckdtree_intp_t end1 = lnode1->end_idx;  // 获取 lnode1 的结束索引
            const ckdtree_intp_t end2 = lnode2->end_idx;  // 获取 lnode2 的结束索引

            CKDTREE_PREFETCH(sdata + sindices[start1] * m, 0, m);  // 预取 self 数据的一部分到缓存

            if (start1 < end1 - 1)
                CKDTREE_PREFETCH(sdata + sindices[start1+1] * m, 0, m);  // 如果条件满足，预取 self 数据的另一部分到缓存

            for (i = start1; i < end1; ++i) {  // 遍历 self 中的数据点

                if (i < end1 - 2)
                    CKDTREE_PREFETCH(sdata + sindices[i+2] * m, 0, m);  // 如果条件满足，预取 self 数据的另一部分到缓存

                CKDTREE_PREFETCH(odata + oindices[start2] * m, 0, m);  // 预取 other 数据的一部分到缓存

                if (start2 < end2 - 1)
                    CKDTREE_PREFETCH(odata + oindices[start2+1] * m, 0, m);  // 如果条件满足，预取 other 数据的另一部分到缓存

                auto &results_i = results[sindices[i]];  // 获取 results 中的第 i 个结果向量的引用

                for (j = start2; j < end2; ++j) {  // 遍历 other 中的数据点

                    if (j < end2 - 2)
                        CKDTREE_PREFETCH(odata + oindices[j+2] * m, 0, m);  // 如果条件满足，预取 other 数据的另一部分到缓存

                    d = MinMaxDist::point_point_p(
                            self,
                            sdata + sindices[i] * m,
                            odata + oindices[j] * m,
                            p, m, tmd);  // 计算 self 和 other 中数据点之间的距离

                    if (d <= tub)
                        results_i.push_back(other->raw_indices[j]);  // 如果距离小于等于上界值 tub，则将 other 的索引 j 添加到 results_i 中
                }
            }

        }
        else { /* 如果 node1 是叶子节点，而 node2 是内部节点 */

            tracker->push_less_of(2, node2);  // 将 node2 的 less 子节点推入 tracker 中
            traverse_checking(
                self, other, results, node1, node2->less, tracker);  // 递归遍历 self 和 other，使用 node2 的 less 子节点
            tracker->pop();  // 弹出 tracker 中的最后一个节点

            tracker->push_greater_of(2, node2);  // 将 node2 的 greater 子节点推入 tracker 中
            traverse_checking(
                self, other, results, node1, node2->greater, tracker);  // 递归遍历 self 和 other，使用 node2 的 greater 子节点
            tracker->pop();  // 弹出 tracker 中的最后一个节点
        }
    }
    else {  /* 1 is an inner node */
        // 如果节点1是内部节点
        if (node2->split_dim == -1) { /* 1 is an inner node, 2 is a leaf node */
            // 如果节点1是内部节点，节点2是叶子节点
            // 将节点1的"小于"路径推入跟踪器
            tracker->push_less_of(1, node1);
            // 递归遍历左子树
            traverse_checking(
                self, other, results, node1->less, node2, tracker);
            // 弹出跟踪器栈顶元素
            tracker->pop();

            // 将节点1的"大于"路径推入跟踪器
            tracker->push_greater_of(1, node1);
            // 递归遍历右子树
            traverse_checking(
                self, other, results, node1->greater, node2, tracker);
            // 弹出跟踪器栈顶元素
            tracker->pop();
        }
        else { /* 1 & 2 are inner nodes */
            // 如果节点1和节点2都是内部节点

            // 将节点1的"小于"路径推入跟踪器
            tracker->push_less_of(1, node1);
            // 将节点2的"小于"路径推入跟踪器
            tracker->push_less_of(2, node2);
            // 递归遍历左子树
            traverse_checking(
                self, other, results, node1->less, node2->less, tracker);
            // 弹出跟踪器栈顶元素
            tracker->pop();

            // 将节点2的"大于"路径推入跟踪器
            tracker->push_greater_of(2, node2);
            // 递归遍历左子树和右子树的组合
            traverse_checking(
                self, other, results, node1->less, node2->greater, tracker);
            // 弹出跟踪器栈顶元素
            tracker->pop();
            // 再次弹出跟踪器栈顶元素

            // 将节点1的"大于"路径推入跟踪器
            tracker->push_greater_of(1, node1);
            // 将节点2的"小于"路径推入跟踪器
            tracker->push_less_of(2, node2);
            // 递归遍历右子树和左子树的组合
            traverse_checking(
                self, other, results, node1->greater, node2->less, tracker);
            // 弹出跟踪器栈顶元素
            tracker->pop();

            // 将节点2的"大于"路径推入跟踪器
            tracker->push_greater_of(2, node2);
            // 递归遍历右子树
            traverse_checking(
                self, other, results, node1->greater, node2->greater, tracker);
            // 弹出跟踪器栈顶元素
            tracker->pop();
            // 再次弹出跟踪器栈顶元素
        }
    }
}

int
query_ball_tree(const ckdtree *self, const ckdtree *other,
                const double r, const double p, const double eps,
                std::vector<ckdtree_intp_t> *results)
{
    // 定义宏 HANDLE，根据条件选择不同的距离计算方式并进行遍历检查
#define HANDLE(cond, kls) \
    if(cond) { \
        // 创建距离计算跟踪器对象，用于记录距离计算过程
        RectRectDistanceTracker<kls> tracker(self, r1, r2, p, eps, r); \
        // 调用遍历检查函数，比较两棵树的节点并更新结果集
        traverse_checking(self, other, results, self->ctree, other->ctree, \
            &tracker); \
    } else

    // 创建两个矩形对象，分别对应 self 和 other 的边界矩形
    Rectangle r1(self->m, self->raw_mins, self->raw_maxes);
    Rectangle r2(other->m, other->raw_mins, other->raw_maxes);

    // 根据 self 的原始盒子数据是否为空选择不同的距离计算方式
    if(CKDTREE_LIKELY(self->raw_boxsize_data == NULL)) {
        // 根据不同的距离度量条件选择相应的距离计算类，并调用 HANDLE 宏
        HANDLE(CKDTREE_LIKELY(p == 2), MinkowskiDistP2)
        HANDLE(p == 1, MinkowskiDistP1)
        HANDLE(std::isinf(p), MinkowskiDistPinf)
        HANDLE(1, MinkowskiDistPp)
        {}
    } else {
        // 根据不同的距离度量条件选择相应的距离计算类，并调用 HANDLE 宏
        HANDLE(CKDTREE_LIKELY(p == 2), BoxMinkowskiDistP2)
        HANDLE(p == 1, BoxMinkowskiDistP1)
        HANDLE(std::isinf(p), BoxMinkowskiDistPinf)
        HANDLE(1, BoxMinkowskiDistPp)
        {}
    }

    // 对每个查询结果进行排序
    for (ckdtree_intp_t i = 0; i < self->n; ++i) {
        std::sort(results[i].begin(), results[i].end());
    }

    // 返回状态值 0 表示函数执行成功
    return 0;
}
```