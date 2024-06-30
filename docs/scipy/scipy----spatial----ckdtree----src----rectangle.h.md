# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\rectangle.h`

```
#ifndef CKDTREE_CPP_RECTANGLE
#define CKDTREE_CPP_RECTANGLE

#include <new>
#include <typeinfo>
#include <stdexcept>
#include <ios>
#include <cmath>
#include <cstring>

/* Interval arithmetic
 * ===================
 */

// 定义一个结构体 Rectangle，用于处理超矩形的几何运算
struct Rectangle {

    const ckdtree_intp_t m; // 成员变量 m，表示维度数

    // 访问器函数，返回最大值数组的指针，允许 const 对象调用此函数
    double * const maxes() const { return &buf[0]; }
    // 访问器函数，返回最小值数组的指针，允许 const 对象调用此函数
    double * const mins() const { return &buf[0] + m; }

    // 构造函数，初始化 Rectangle 对象，复制输入的最小值和最大值数组
    Rectangle(const ckdtree_intp_t _m,
              const double *_mins,
              const double *_maxes) : m(_m), buf(2 * m) {

        /* copy array data */
        /* FIXME: use std::vector ? */
        // 使用 memcpy 将输入数组的数据复制到内部缓冲区中
        std::memcpy((void*)mins(), (void*)_mins, m*sizeof(double));
        std::memcpy((void*)maxes(), (void*)_maxes, m*sizeof(double));
    };

    // 拷贝构造函数，复制另一个 Rectangle 对象
    Rectangle(const Rectangle& rect) : m(rect.m), buf(rect.buf) {};

    private:
        mutable std::vector<double> buf; // 可变的内部缓冲区，存储最小值和最大值数组
};

#include "distance.h"

/*
 * Rectangle-to-rectangle distance tracker
 * =======================================
 *
 * The logical unit that repeats over and over is to keep track of the
 * maximum and minimum distances between points in two hyperrectangles
 * as these rectangles are successively split.
 *
 * Example
 * -------
 * node1 encloses points in rect1, node2 encloses those in rect2
 *
 * cdef RectRectDistanceTracker dist_tracker
 * dist_tracker = RectRectDistanceTracker(rect1, rect2, p)
 *
 * ...
 *
 * if dist_tracker.min_distance < ...:
 *     ...
 *
 * dist_tracker.push_less_of(1, node1)
 * do_something(node1.less, dist_tracker)
 * dist_tracker.pop()
 *
 * dist_tracker.push_greater_of(1, node1)
 * do_something(node1.greater, dist_tracker)
 * dist_tracker.pop()
 *
 * Notice that Point is just a reduced case of Rectangle where
 * mins == maxes.
 *
 */

// 定义结构体 RR_stack_item，用于跟踪超矩形之间的距离
struct RR_stack_item {
    ckdtree_intp_t    which;
    ckdtree_intp_t    split_dim;
    double min_along_dim;
    double max_along_dim;
    double min_distance;
    double max_distance;
};

// 常量定义：表示小于关系和大于关系
const ckdtree_intp_t LESS = 1;
const ckdtree_intp_t GREATER = 2;

// 模板类 RectRectDistanceTracker，用于跟踪超矩形之间的距离
template<typename MinMaxDist>
    struct RectRectDistanceTracker {

    const ckdtree * tree;
    Rectangle rect1;
    Rectangle rect2;
    double p;
    double epsfac;
    double upper_bound;
    double min_distance;
    double max_distance;

    ckdtree_intp_t stack_size;
    ckdtree_intp_t stack_max_size;
    std::vector<RR_stack_item> stack_arr;
    RR_stack_item *stack;

    /* if min/max distance / adjustment is less than this,
     * we believe the incremental tracking is inaccurate */
    // 变量：当距离调整小于此值时，认为增量跟踪不准确
    double inaccurate_distance_limit;

    // 调整栈大小的私有函数
    void _resize_stack(const ckdtree_intp_t new_max_size) {
        stack_arr.resize(new_max_size);
        stack = &stack_arr[0];
        stack_max_size = new_max_size;
    };
    // 定义 RectRectDistanceTracker 类的构造函数，初始化成员变量并进行必要的异常检查
    RectRectDistanceTracker(const ckdtree *_tree,
                 const Rectangle& _rect1, const Rectangle& _rect2,
                 const double _p, const double eps,
                 const double _upper_bound)
        : tree(_tree), rect1(_rect1), rect2(_rect2), stack_arr(8) {

        // 检查两个矩形的维度是否相同，若不同则抛出异常
        if (rect1.m != rect2.m) {
            const char *msg = "rect1 and rect2 have different dimensions";
            throw std::invalid_argument(msg); // raises ValueError
        }

        p = _p;

        /* internally we represent all distances as distance ** p */
        // 将所有距离表示为 distance ** p 的形式进行内部处理
        if (CKDTREE_LIKELY(p == 2.0))
            upper_bound = _upper_bound * _upper_bound;
        else if ((!std::isinf(p)) && (!std::isinf(_upper_bound)))
            upper_bound = std::pow(_upper_bound,p);
        else
            upper_bound = _upper_bound;

        /* fiddle approximation factor */
        // 调整近似因子 epsfac
        if (CKDTREE_LIKELY(p == 2.0)) {
            double tmp = 1. + eps;
            epsfac = 1. / (tmp*tmp);
        }
        else if (eps == 0.)
            epsfac = 1.;
        else if (std::isinf(p))
            epsfac = 1. / (1. + eps);
        else
            epsfac = 1. / std::pow((1. + eps), p);

        // 设置堆栈初始值
        stack = &stack_arr[0];
        stack_max_size = 8;
        stack_size = 0;

        /* Compute initial min and max distances */
        // 计算初始的最小和最大距离
        MinMaxDist::rect_rect_p(tree, rect1, rect2, p, &min_distance, &max_distance);
        // 若最大距离为无穷大，则抛出异常
        if(std::isinf(max_distance)) {
            const char *msg = "Encountering floating point overflow. "
                              "The value of p too large for this dataset; "
                              "For such large p, consider using the special case p=np.inf . ";
            throw std::invalid_argument(msg); // raises ValueError
        }
        // 设置不精确距离限制为最大距离
        inaccurate_distance_limit = max_distance;
    };
    void push(const ckdtree_intp_t which, const intptr_t direction,
              const ckdtree_intp_t split_dim, const double split_val) {
        // 拷贝成员变量 p 到本地变量 p
        const double p = this->p;
        
        /* 如果误差可能会影响增量距离追踪，则 subnomial 等于 1。
         * 在这种情况下，我们总是重新计算距离。
         * 重新计算会增加 pow 的调用次数，因此如果舍入误差看起来不会抹掉值，
         * 我们仍然执行增量更新。
         */
        int subnomial = 0;

        Rectangle *rect;
        // 根据 which 的值选择相应的矩形
        if (which == 1)
            rect = &rect1;
        else
            rect = &rect2;

        /* 将数据压入堆栈 */
        if (stack_size == stack_max_size)
            _resize_stack(stack_max_size * 2);

        // 获取堆栈中的下一个项
        RR_stack_item *item = &stack[stack_size];
        ++stack_size;
        // 设置堆栈项的属性
        item->which = which;
        item->split_dim = split_dim;
        item->min_distance = min_distance;
        item->max_distance = max_distance;
        item->min_along_dim = rect->mins()[split_dim];
        item->max_along_dim = rect->maxes()[split_dim];

        /* 更新最小/最大距离 */
        double min1, max1;
        double min2, max2;

        // 计算两个矩形之间的距离范围
        MinMaxDist::interval_interval_p(tree, rect1, rect2, split_dim, p, &min1, &max1);

        // 根据 direction 更新矩形的边界值
        if (direction == LESS)
            rect->maxes()[split_dim] = split_val;
        else
            rect->mins()[split_dim] = split_val;

        // 再次计算更新后的距离范围
        MinMaxDist::interval_interval_p(tree, rect1, rect2, split_dim, p, &min2, &max2);

        // 检查是否需要重新计算距离
        subnomial = subnomial || (min_distance < inaccurate_distance_limit || max_distance < inaccurate_distance_limit);

        subnomial = subnomial || ((min1 != 0 && min1 < inaccurate_distance_limit) || max1 < inaccurate_distance_limit);
        subnomial = subnomial || ((min2 != 0 && min2 < inaccurate_distance_limit) || max2 < inaccurate_distance_limit);
        subnomial = subnomial || (min_distance < inaccurate_distance_limit || max_distance < inaccurate_distance_limit);

        // 如果可能的话，根据 subnomial 的值重新计算距离
        if (CKDTREE_UNLIKELY(subnomial)) {
            MinMaxDist::rect_rect_p(tree, rect1, rect2, p, &min_distance, &max_distance);
        } else {
            // 否则，通过增量更新距离值
            min_distance += (min2 - min1);
            max_distance += (max2 - max1);
        }
    };

    inline void push_less_of(const ckdtree_intp_t which,
                                 const ckdtreenode *node) {
        // 调用 push 函数，传递 LESS 方向
        push(which, LESS, node->split_dim, node->split);
    };

    inline void push_greater_of(const ckdtree_intp_t which,
                                    const ckdtreenode *node) {
        // 调用 push 函数，传递 GREATER 方向
        push(which, GREATER, node->split_dim, node->split);
    };
    // 从栈中弹出一个元素
    inline void pop() {
        // 减少栈大小
        --stack_size;

        // 断言栈大小不会小于0
        if (CKDTREE_UNLIKELY(stack_size < 0)) {
            // 如果出现栈大小小于0的情况，抛出逻辑错误异常
            const char *msg = "Bad stack size. This error should never occur.";
            throw std::logic_error(msg);
        }

        // 获取栈顶元素的引用
        RR_stack_item* item = &stack[stack_size];

        // 更新最小和最大距离
        min_distance = item->min_distance;
        max_distance = item->max_distance;

        // 根据栈顶元素的which属性选择更新rect1或rect2的边界
        if (item->which == 1) {
            rect1.mins()[item->split_dim] = item->min_along_dim;
            rect1.maxes()[item->split_dim] = item->max_along_dim;
        }
        else {
            rect2.mins()[item->split_dim] = item->min_along_dim;
            rect2.maxes()[item->split_dim] = item->max_along_dim;
        }
    };
};


#endif
```