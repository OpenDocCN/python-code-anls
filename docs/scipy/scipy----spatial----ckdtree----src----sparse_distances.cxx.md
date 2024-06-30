# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\sparse_distances.cxx`

```
// 包含头文件 "ckdtree_decl.h"，声明了 ckdtree 结构体或类的相关内容
#include "ckdtree_decl.h"
// 包含头文件 "rectangle.h"，声明了与矩形相关的结构体或类
#include "rectangle.h"
// 包含头文件 "coo_entries.h"，声明了与 COO 格式条目相关的结构体或类
#include "coo_entries.h"

// 包含数学函数库
#include <cmath>
// 包含 C 标准库函数
#include <cstdlib>
// 包含 C 字符串操作函数
#include <cstring>

// 包含标准向量容器类模板
#include <vector>
// 包含字符串类模板
#include <string>
// 包含字符串流类模板
#include <sstream>
// 包含标准异常类模板
#include <stdexcept>
// 包含输入输出流类模板
#include <ios>

// 定义模板函数 traverse，参数类型为 MinMaxDist 类型
template <typename MinMaxDist> static void
traverse(const ckdtree *self, const ckdtree *other,
         std::vector<coo_entry> *results,
         const ckdtreenode *node1, const ckdtreenode *node2,
         RectRectDistanceTracker<MinMaxDist> *tracker)
{
    // 如果当前最小距离大于上界，直接返回，不继续遍历
    if (tracker->min_distance > tracker->upper_bound)
        return;
}
    else if (node1->split_dim == -1) {  /* 1 is leaf node */

        if (node2->split_dim == -1) {  /* 1 & 2 are leaves */
            /* brute-force */
            const double p = tracker->p;  // 获取跟踪器的距离度量参数 p
            const double tub = tracker->upper_bound;  // 获取跟踪器的上界阈值
            const double *sdata = self->raw_data;  // 获取自身数据数组的指针
            const ckdtree_intp_t *sindices = self->raw_indices;  // 获取自身数据索引数组的指针
            const double *odata = other->raw_data;  // 获取另一个数据结构的数据数组指针
            const ckdtree_intp_t *oindices = other->raw_indices;  // 获取另一个数据结构的数据索引数组指针
            const ckdtree_intp_t m = self->m;  // 获取数据维度
            const ckdtree_intp_t start1 = node1->start_idx;  // 获取节点1的起始索引
            const ckdtree_intp_t start2 = node2->start_idx;  // 获取节点2的起始索引
            const ckdtree_intp_t end1 = node1->end_idx;  // 获取节点1的结束索引
            const ckdtree_intp_t end2 = node2->end_idx;  // 获取节点2的结束索引

            CKDTREE_PREFETCH(sdata + sindices[start1] * m, 0, m);  // 预取自身数据的起始位置
            if (start1 < end1 - 1)
               CKDTREE_PREFETCH(sdata + sindices[start1+1] * m, 0, m);  // 如果存在下一个数据点，预取其位置

            for (ckdtree_intp_t i = start1; i < end1; ++i) {  // 遍历节点1的索引范围

                if (i < end1 - 2)
                     CKDTREE_PREFETCH(sdata + sindices[i+2] * m, 0, m);  // 预取未来可能用到的自身数据点位置

                CKDTREE_PREFETCH(odata + oindices[start2] * m, 0, m);  // 预取另一个数据结构的起始位置
                if (start2 < end2 - 1)
                    CKDTREE_PREFETCH(sdata + oindices[start2+1] * m, 0, m);  // 如果存在下一个数据点，预取其位置

                for (ckdtree_intp_t j = start2; j < end2; ++j) {  // 遍历节点2的索引范围

                    if (j < end2 - 2)
                        CKDTREE_PREFETCH(odata + oindices[j+2] * m, 0, m);  // 预取未来可能用到的另一个数据结构数据点位置

                    double d = MinMaxDist::point_point_p(  // 计算节点1和节点2对应数据点之间的距离
                            self,
                            sdata + sindices[i] * m,
                            odata + oindices[j] * m,
                            p, m, tub);

                    if (d <= tub) {  // 如果距离小于等于上界阈值

                        if (CKDTREE_LIKELY(p == 2.0))  // 如果 p 等于 2.0，使用平方根计算距离
                            d = std::sqrt(d);
                        else if ((p != 1) && (!std::isinf(p)))  // 如果 p 不等于 1 且不是无穷大，使用幂函数计算距离
                            d = std::pow(d, 1. / p);

                        coo_entry e = {sindices[i], oindices[j], d};  // 创建坐标条目，包含索引和距离
                        results->push_back(e);  // 将坐标条目添加到结果向量中
                    }
                }
            }
        }
        else {  /* 1 is a leaf node, 2 is inner node */
            tracker->push_less_of(2, node2);  // 将节点2的较小子节点推入跟踪器堆栈
            traverse(self, other, results, node1, node2->less, tracker);  // 递归遍历节点1和节点2的较小子节点
            tracker->pop();  // 弹出堆栈，恢复堆栈状态

            tracker->push_greater_of(2, node2);  // 将节点2的较大子节点推入跟踪器堆栈
            traverse(self, other, results, node1, node2->greater, tracker);  // 递归遍历节点1和节点2的较大子节点
            tracker->pop();  // 弹出堆栈，恢复堆栈状态
        }
    }
    else {  /* 1 is an inner node */
        // 如果节点 1 是内部节点
        if (node2->split_dim == -1) {
            // 如果节点 2 是叶子节点
            // 将节点 1 的"小于"路径压入追踪器
            tracker->push_less_of(1, node1);
            // 递归遍历左子树
            traverse(self, other, results, node1->less, node2, tracker);
            // 弹出追踪器栈顶元素
            tracker->pop();

            // 将节点 1 的"大于"路径压入追踪器
            tracker->push_greater_of(1, node1);
            // 递归遍历右子树
            traverse(self, other, results, node1->greater, node2, tracker);
            // 弹出追踪器栈顶元素
            tracker->pop();
        }
        else { /* 1 and 2 are inner nodes */
            // 如果节点 1 和节点 2 都是内部节点
            // 将节点 1 的"小于"路径压入追踪器
            tracker->push_less_of(1, node1);
            // 将节点 2 的"小于"路径压入追踪器
            tracker->push_less_of(2, node2);
            // 递归遍历左子树
            traverse(self, other, results, node1->less, node2->less, tracker);
            // 弹出追踪器栈顶元素
            tracker->pop();

            // 将节点 2 的"大于"路径压入追踪器
            tracker->push_greater_of(2, node2);
            // 递归遍历左子树的右子树
            traverse(self, other, results, node1->less, node2->greater, tracker);
            // 弹出追踪器栈顶元素
            tracker->pop();
            // 弹出追踪器栈顶元素，恢复到只包含节点 1 的路径
            tracker->pop();

            // 将节点 1 的"大于"路径压入追踪器
            tracker->push_greater_of(1, node1);
            // 将节点 2 的"小于"路径压入追踪器
            tracker->push_less_of(2, node2);
            // 递归遍历右子树的左子树
            traverse(self, other, results, node1->greater, node2->less, tracker);
            // 弹出追踪器栈顶元素
            tracker->pop();

            // 将节点 2 的"大于"路径压入追踪器
            tracker->push_greater_of(2, node2);
            // 递归遍历右子树的右子树
            traverse(self, other, results, node1->greater, node2->greater, tracker);
            // 弹出追踪器栈顶元素
            tracker->pop();
            // 弹出追踪器栈顶元素，恢复到初始状态
            tracker->pop();
        }
    }
}
```