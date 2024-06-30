# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\ordered_pair.h`

```
#ifndef CKDTREE_ORDERED_PAIR
#define CKDTREE_ORDERED_PAIR

#include <vector>

// 定义一个结构体，表示有序的一对整数
struct ordered_pair {
    ckdtree_intp_t i;  // 第一个整数
    ckdtree_intp_t j;  // 第二个整数
};

// 定义一个内联函数，用于将有序对添加到结果向量中
inline void
add_ordered_pair(std::vector<ordered_pair> *results,
                       const ckdtree_intp_t i, const intptr_t j)
{
    // 如果第一个整数大于第二个整数，创建一个有序对（j,i）并将其添加到结果向量中
    if (i > j) {
        ordered_pair p = {j,i};
        results->push_back(p);
    }
    // 否则创建一个有序对（i,j）并将其添加到结果向量中
    else {
        ordered_pair p = {i,j};
        results->push_back(p);;
    }
}

#endif
```