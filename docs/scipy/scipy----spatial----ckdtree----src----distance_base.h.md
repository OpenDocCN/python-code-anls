# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\distance_base.h`

```
/*
 * 结构体 `BaseMinkowskiDistPp`，用于计算 Minkowski 距离的基本操作
 */
template <typename Dist1D>
struct BaseMinkowskiDistPp {

    /*
     * 计算两个超矩形之间在第 k 维上的最小/最大距离
     * 只有在 p 不是无穷大时才能使用这些函数
     */
    static inline void
    interval_interval_p(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const ckdtree_intp_t k, const double p,
                        double *min, double *max)
    {
        /* Compute the minimum/maximum distance along dimension k between points in
         * two hyperrectangles.
         */
        Dist1D::interval_interval(tree, rect1, rect2, k, min, max);
        *min = std::pow(*min, p);  // 计算最小距离的 p 次幂
        *max = std::pow(*max, p);  // 计算最大距离的 p 次幂
    }

    /*
     * 计算两个超矩形之间在所有维度上的最小/最大距离
     */
    static inline void
    rect_rect_p(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const double p,
                        double *min, double *max)
    {
        *min = 0.;
        *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            double min_, max_;

            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);

            *min += std::pow(min_, p);  // 累加每个维度的最小距离的 p 次幂
            *max += std::pow(max_, p);  // 累加每个维度的最大距离的 p 次幂
        }
    }

    /*
     * 计算两个点之间的 Minkowski p-距离的 p 次幂
     *
     * 如果距离的 p 次幂大于 upperbound，则可能返回大于 upperbound 的任何数
     * （计算结果被截断）
     */
    static inline double
    point_point_p(const ckdtree * tree,
               const double *x, const double *y,
               const double p, const ckdtree_intp_t k,
               const double upperbound)
    {
       /*
        * Compute the distance between x and y
        *
        * Computes the Minkowski p-distance to the power p between two points.
        * If the distance**p is larger than upperbound, then any number larger
        * than upperbound may be returned (the calculation is truncated).
        */

        ckdtree_intp_t i;
        double r, r1;
        r = 0;
        for (i=0; i<k; ++i) {
            r1 = Dist1D::point_point(tree, x, y, i);
            r += std::pow(r1, p);  // 累加每个维度上的距离的 p 次幂
            if (r>upperbound)
                 return r;
        }
        return r;
    }

    /*
     * 计算 s 的 p 次幂
     */
    static inline double
    distance_p(const double s, const double p)
    {
        return std::pow(s,p);  // 返回 s 的 p 次幂
    }
};
    {
        // 初始化最小值和最大值为0
        *min = 0.;
        *max = 0.;
        // 遍历矩形 rect1 的每一维度
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            // 定义局部变量存储距离的最小值和最大值
            double min_, max_;
            // 调用 Dist1D::interval_interval 函数计算矩形 rect1 和 rect2 在第 i 维度上的距离区间
            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);
            // 将每个维度上的最小距离累加到 *min，将最大距离累加到 *max
            *min += min_;
            *max += max_;
        }
    }
    
    static inline double
    point_point_p(const ckdtree * tree,
               const double *x, const double *y,
               const double p, const ckdtree_intp_t k,
               const double upperbound)
    {
        // 定义变量 r 存储点到点之间的距离的累加和
        double r;
        r = 0;
        // 遍历 k 维度，计算点 x 和点 y 之间的距离并累加
        for (ckdtree_intp_t i=0; i<k; ++i) {
            r += Dist1D::point_point(tree, x, y, i);
            // 如果累加的距离超过了上限 upperbound，则立即返回
            if (r > upperbound)
                return r;
        }
        // 返回累加的距离总和
        return r;
    }
    
    static inline double
    distance_p(const double s, const double p)
    {
        // 直接返回参数 s，即距离函数 distance_p 的实现
        return s;
    }
    };



template <typename Dist1D>
struct BaseMinkowskiDistPinf : public BaseMinkowskiDistPp<Dist1D> {
    static inline void
    interval_interval_p(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const ckdtree_intp_t k, double p,
                        double *min, double *max)
    {
        // 调用 rect_rect_p 函数计算两个矩形之间的距离范围
        return rect_rect_p(tree, rect1, rect2, p, min, max);
    }

    static inline void
    rect_rect_p(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const double p,
                        double *min, double *max)
    {
        // 初始化最小距离和最大距离为 0
        *min = 0.;
        *max = 0.;
        // 遍历矩形 rect1 中的维度
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            double min_, max_;

            // 调用 Dist1D::interval_interval 函数计算矩形 rect1 和 rect2 在第 i 维度上的距离范围
            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);

            // 更新整体的最小距离和最大距离
            *min = ckdtree_fmax(*min, min_);
            *max = ckdtree_fmax(*max, max_);
        }
    }

    static inline double
    point_point_p(const ckdtree * tree,
               const double *x, const double *y,
               const double p, const ckdtree_intp_t k,
               const double upperbound)
    {
        // 初始化距离 r 为 0
        ckdtree_intp_t i;
        double r;
        r = 0;
        // 遍历点 x 和点 y 的前 k 维度
        for (i=0; i<k; ++i) {
            // 计算点 x 和点 y 在第 i 维度上的距离，更新 r
            r = ckdtree_fmax(r, Dist1D::point_point(tree, x, y, i));
            // 如果 r 超过了上界 upperbound，则直接返回 r
            if (r > upperbound)
                return r;
        }
        return r;
    }

    static inline double
    distance_p(const double s, const double p)
    {
        // 距离函数，返回距离 s
        return s;
    }
};

template <typename Dist1D>
struct BaseMinkowskiDistP2 : public BaseMinkowskiDistPp<Dist1D> {
    static inline void
    interval_interval_p(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const ckdtree_intp_t k, const double p,
                        double *min, double *max)
    {
        /* Compute the minimum/maximum distance along dimension k between points in
         * two hyperrectangles.
         */
        // 调用 Dist1D::interval_interval 函数计算两个矩形之间在第 k 维度上的距离范围
        Dist1D::interval_interval(tree, rect1, rect2, k, min, max);
        // 对最小距离和最大距离分别进行平方
        *min *= *min;
        *max *= *max;
    }

    static inline void
    rect_rect_p(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const double p,
                        double *min, double *max)
    {
        // 初始化最小距离和最大距离为 0
        *min = 0.;
        *max = 0.;
        // 遍历矩形 rect1 中的维度
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            double min_, max_;

            // 调用 Dist1D::interval_interval 函数计算矩形 rect1 和 rect2 在第 i 维度上的距离范围
            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);
            // 将计算结果平方
            min_ *= min_;
            max_ *= max_;

            // 更新整体的最小距离平方和最大距离平方
            *min += min_;
            *max += max_;
        }
    }
    static inline double
    {
        // 声明整型变量 i，用于循环计数
        ckdtree_intp_t i;
        // 声明双精度浮点数变量 r，并初始化为 0
        double r;
        r = 0;
        // 循环计算每个点到目标点的距离的平方并累加
        for (i=0; i<k; ++i) {
            // 计算并获取第 i 个点到目标点的距离的平方
            double r1 = Dist1D::point_point(tree, x, y, i);
            // 将计算得到的距离的平方加到总距离的平方和 r 中
            r += r1 * r1;
            // 如果总距离的平方超过了上限，直接返回当前总距离的平方
            if (r > upperbound)
                return r;
        }
        // 循环结束后返回总距离的平方
        return r;
    }
    
    static inline double
    distance_p(const double s, const double p)
    {
        // 返回参数 s 的平方
        return s * s;
    }
};


注释：


// 结束当前的代码块或语句，该分号表示语句结束
```