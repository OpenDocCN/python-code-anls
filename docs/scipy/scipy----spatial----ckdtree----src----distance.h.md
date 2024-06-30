# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\distance.h`

```
/*
 * 包含距离计算的基本头文件
 */
#include "distance_base.h"

/*
 * 表示一维空间中的平面距离计算
 */
struct PlainDist1D {
    /*
     * 计算给定点到指定最小值和最大值的距离
     */
    static inline const double side_distance_from_min_max(
        const ckdtree * tree, const double x,
        const double min,
        const double max,
        const ckdtree_intp_t k
        )
    {
        double s, t;
        s = 0;
        t = x - max;
        if (t > s) {
            s = t;
        } else {
            t = min - x;
            if (t > s) s = t;
        }
        return s;
    }

    /*
     * 计算两个矩形之间在指定维度 k 上的最小和最大距离
     */
    static inline void
    interval_interval(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const ckdtree_intp_t k,
                        double *min, double *max)
    {
        /* 计算两个超矩形之间在维度 k 上的最小/最大距离 */
        *min = ckdtree_fmax(0., fmax(rect1.mins()[k] - rect2.maxes()[k],
                              rect2.mins()[k] - rect1.maxes()[k]));
        *max = ckdtree_fmax(rect1.maxes()[k] - rect2.mins()[k],
                              rect2.maxes()[k] - rect1.mins()[k]);
    }

    /*
     * 计算两个点之间在指定维度 k 上的距离
     */
    static inline double
    point_point(const ckdtree * tree,
               const double *x, const double *y,
                 const ckdtree_intp_t k) {
        return ckdtree_fabs(x[k] - y[k]);
    }
};

/*
 * 定义基于 PlainDist1D 的 Minkowski 距离计算类型
 */
typedef BaseMinkowskiDistPp<PlainDist1D> MinkowskiDistPp;
typedef BaseMinkowskiDistPinf<PlainDist1D> MinkowskiDistPinf;
typedef BaseMinkowskiDistP1<PlainDist1D> MinkowskiDistP1;

/*
 * 定义未优化的 Minkowski P2 距离计算类型
 */
typedef BaseMinkowskiDistP2<PlainDist1D> NonOptimizedMinkowskiDistP2;

/*
 * 距离测量函数
 * ==============
 */

/*
 * 计算两个向量 u 和 v 之间的平方欧氏距离
 */
inline double
sqeuclidean_distance_double(const double *u, const double *v, ckdtree_intp_t n)
{
    double s;
    ckdtree_intp_t i = 0;
    // 手动展开的循环，可以进行向量化处理
    double acc[4] = {0., 0., 0., 0.};
    for (; i + 4 <= n; i += 4) {
        double _u[4] = {u[i], u[i + 1], u[i + 2], u[i + 3]};
        double _v[4] = {v[i], v[i + 1], v[i + 2], v[i + 3]};
        double diff[4] = {_u[0] - _v[0],
                               _u[1] - _v[1],
                               _u[2] - _v[2],
                               _u[3] - _v[3]};
        acc[0] += diff[0] * diff[0];
        acc[1] += diff[1] * diff[1];
        acc[2] += diff[2] * diff[2];
        acc[3] += diff[3] * diff[3];
    }
    s = acc[0] + acc[1] + acc[2] + acc[3];
    if (i < n) {
        for(; i<n; ++i) {
            double d = u[i] - v[i];
            s += d * d;
        }
    }
    return s;
}

/*
 * 定义基于未优化 Minkowski P2 距离计算的结构体
 */
struct MinkowskiDistP2: NonOptimizedMinkowskiDistP2 {
    /*
     * 计算两个点之间的 Minkowski P2 距离
     */
    static inline double
    point_point_p(const ckdtree * tree,
               const double *x, const double *y,
               const double p, const ckdtree_intp_t k,
               const double upperbound)
    {
        return sqeuclidean_distance_double(x, y, k);
    }
};

/*
 * 表示一维空间中的盒子距离计算
 */
struct BoxDist1D {
    static inline void _interval_interval_1d (
        double min, double max,
        double *realmin, double *realmax,
        const double full, const double half
    )
    {
        /* Minimum and maximum distance of two intervals in a periodic box
         *
         * min and max is the nonperiodic distance between the near
         * and far edges.
         *
         * full and half are the box size and 0.5 * box size.
         *
         * value is returned in realmin and realmax;
         *
         * This function is copied from kdcount, and the convention
         * of is that
         *
         * min = rect1.min - rect2.max
         * max = rect1.max - rect2.min = - (rect2.min - rect1.max)
         *
         * We will fix the convention later.
         * */
        
        if (CKDTREE_UNLIKELY(full <= 0)) {
            /* A non-periodic dimension */
            /* \/     */
            
            if(max <= 0 || min >= 0) {
                /* do not pass though 0 */
                min = ckdtree_fabs(min);
                max = ckdtree_fabs(max);
                if(min < max) {
                    *realmin = min;
                    *realmax = max;
                } else {
                    *realmin = max;
                    *realmax = min;
                }
            } else {
                min = ckdtree_fabs(min);
                max = ckdtree_fabs(max);
                *realmax = ckdtree_fmax(max, min);
                *realmin = 0;
            }
            
            /* done with non-periodic dimension */
            return;
        }
        
        if(max <= 0 || min >= 0) {
            /* do not pass through 0 */
            min = ckdtree_fabs(min);
            max = ckdtree_fabs(max);
            if(min > max) {
                double t = min;
                min = max;
                max = t;
            }
            if(max < half) {
                /* all below half*/
                *realmin = min;
                *realmax = max;
            } else if(min > half) {
                /* all above half */
                *realmax = full - min;
                *realmin = full - max;
            } else {
                /* min below, max above */
                *realmax = half;
                *realmin = ckdtree_fmin(min, full - max);
            }
        } else {
            /* pass though 0 */
            min = -min;
            if(min > max) max = min;
            if(max > half) max = half;
            *realmax = max;
            *realmin = 0;
        }
    }
    
    static inline void
    interval_interval(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const ckdtree_intp_t k,
                        double *min, double *max)
    {
        /* 计算两个超矩形在第 k 维上的点之间的最小/最大距离 */
        _interval_interval_1d(rect1.mins()[k] - rect2.maxes()[k],
                    rect1.maxes()[k] - rect2.mins()[k], min, max,
                    tree->raw_boxsize_data[k], tree->raw_boxsize_data[k + rect1.m]);
    }
    
    static inline double
    point_point(const ckdtree * tree,
               const double *x, const double *y,
               const ckdtree_intp_t k)
    {
        double r1;
        // 计算点 x 和点 y 在第 k 维上的距离，并进行周期性包装
        r1 = wrap_distance(x[k] - y[k], tree->raw_boxsize_data[k + tree->m], tree->raw_boxsize_data[k]);
        r1 = ckdtree_fabs(r1);  // 取距离的绝对值
        return r1;
    }
    
    static inline const double
    wrap_position(const double x, const double boxsize)
    {
        if (boxsize <= 0) return x;
        const double r = std::floor(x / boxsize);  // 计算 x 在周期边界下的位置
        double x1 = x - r * boxsize;
        /* 确保结果在边界盒内。*/
        while(x1 >= boxsize) x1 -= boxsize;  // 处理超出边界的情况
        while(x1 < 0) x1 += boxsize;  // 处理负数的情况
        return x1;
    }
    
    static inline const double side_distance_from_min_max(
        const ckdtree * tree, const double x,
        const double min,
        const double max,
        const ckdtree_intp_t k
        )
    {
        double s, t, tmin, tmax;
        double fb = tree->raw_boxsize_data[k];
        double hb = tree->raw_boxsize_data[k + tree->m];
    
        if (fb <= 0) {
            /* 非周期性维度 */
            s = PlainDist1D::side_distance_from_min_max(tree, x, min, max, k);  // 使用非周期性距离计算
            return s;
        }
    
        /* 周期性维度 */
        s = 0;
        tmax = x - max;
        tmin = x - min;
        /* 测试点是否在这个范围内 */
        if(CKDTREE_LIKELY(tmax < 0 && tmin > 0)) {
            /* 是的，最小距离为0 */
            return 0;
        }
    
        /* 不是 */
        tmax = ckdtree_fabs(tmax);
        tmin = ckdtree_fabs(tmin);
    
        /* 让 tmin 成为更近的边缘 */
        if(tmin > tmax) { t = tmin; tmin = tmax; tmax = t; }
    
        /* 两个边缘都小于半个盒子。 */
        /* 没有包装，使用更近的边缘 */
        if(tmax < hb) return tmin;
    
        /* 两个边缘都大于半个盒子。 */
        /* 在两个边缘都有包装，使用包装后的更远边缘 */
        if(tmin > hb) return fb - tmax;
    
        /* 更远的边缘有包装 */
        tmax = fb - tmax;
        if(tmin > tmax) return tmax;
        return tmin;
    }
    
    private:
    static inline double
    wrap_distance(const double x, const double hb, const double fb)
    {
        double x1;
        if (CKDTREE_UNLIKELY(x < -hb)) x1 = fb + x;  // 如果 x 小于负半边界，则进行包装
        else if (CKDTREE_UNLIKELY(x > hb)) x1 = x - fb;  // 如果 x 大于正半边界，则进行包装
        else x1 = x;
    #if 0
        printf("ckdtree_fabs_b x : %g x1 %g\n", x, x1);
    #endif
        return x1;
    }
};



// 定义 BoxDist1D 为基础的 Minkowski 距离 Pp 版本的类型
typedef BaseMinkowskiDistPp<BoxDist1D> BoxMinkowskiDistPp;
// 定义 BoxDist1D 为基础的 Minkowski 距离 Pinf 版本的类型
typedef BaseMinkowskiDistPinf<BoxDist1D> BoxMinkowskiDistPinf;
// 定义 BoxDist1D 为基础的 Minkowski 距离 P1 版本的类型
typedef BaseMinkowskiDistP1<BoxDist1D> BoxMinkowskiDistP1;
// 定义 BoxDist1D 为基础的 Minkowski 距离 P2 版本的类型
typedef BaseMinkowskiDistP2<BoxDist1D> BoxMinkowskiDistP2;



// 上述代码段为定义几种不同的 Minkowski 距离类型的 C++ 类型别名。
// 每种类型以 BoxDist1D 作为基础类型，并分别对应于不同的 P 范数版本。
```