# `D:\src\scipysrc\scipy\scipy\spatial\src\distance_metrics.h`

```
#pragma once

#include <cmath>
#include "views.h"

#ifdef __GNUC__
#define ALWAYS_INLINE inline __attribute__((always_inline))
#define INLINE_LAMBDA __attribute__((always_inline))
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#define INLINE_LAMBDA
#else
#define ALWAYS_INLINE inline
#define INLINE_LAMBDA
#endif

// 结构体定义：Identity，用于返回其参数的值
struct Identity {
    // 模板成员函数：operator()，始终内联，返回原始值
    template <typename T>
    ALWAYS_INLINE T operator() (T && val) const {
        return std::forward<T>(val);
    }
};

// 结构体定义：Plus，用于返回两个参数的加法结果
struct Plus {
    // 模板成员函数：operator()，始终内联，返回两数之和
    template <typename T>
    ALWAYS_INLINE T operator()(T a, T b) const {
        return a + b;
    }
};

// 辅助结构体：ForceUnroll，用于强制完全展开固定边界的循环
template <int unroll>
struct ForceUnroll{
    // 模板成员函数：operator()，始终内联，递归展开循环直到 unroll == 1
    template <typename Func>
    ALWAYS_INLINE void operator()(const Func& f) const {
        ForceUnroll<unroll - 1>{}(f); // 递归调用，减少 unroll 的值
        f(unroll - 1); // 调用函数对象 f，传入当前 unroll 的值
    }
};

// ForceUnroll 特化版本：当 unroll 等于 1 时，直接调用函数对象 f
template <>
struct ForceUnroll<1> {
    template <typename Func>
    ALWAYS_INLINE void operator()(const Func& f) const {
        f(0);
    }
};

// transform_reduce_2d_ 函数模板：执行二维变换和规约操作
template <int ilp_factor=4,
          typename T,
          typename TransformFunc,
          typename ProjectFunc = Identity,
          typename ReduceFunc = Plus>
void transform_reduce_2d_(
    StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y,
    const TransformFunc& map,
    const ProjectFunc& project = Identity{},
    const ReduceFunc& reduce = Plus{}) {
    // 定义 map 函数返回值的累积类型
    using AccumulateType = typename std::decay<decltype(
        map(std::declval<T>(), std::declval<T>()))>::type;
    // 定义 x 和 y 的步幅
    intptr_t xs = x.strides[1], ys = y.strides[1];

    // 初始化循环计数器 i
    intptr_t i = 0;
    // 如果 x 和 y 的步幅都为 1，则使用 ILP 并行化
    if (xs == 1 && ys == 1) {
        // 外层循环，每次增加 ilp_factor，直到 x.shape[0] - (ilp_factor - 1)
        for (; i + (ilp_factor - 1) < x.shape[0]; i += ilp_factor) {
            // 声明并初始化指向 x 和 y 对应行的指针数组
            const T* x_rows[ilp_factor];
            const T* y_rows[ilp_factor];
            // 使用 ForceUnroll 展开循环，为每个 k 填充 x_rows 和 y_rows
            ForceUnroll<ilp_factor>{}([&](int k) {
                x_rows[k] = &x(i + k, 0);
                y_rows[k] = &y(i + k, 0);
            });

            // 声明并初始化距离数组 dist
            AccumulateType dist[ilp_factor] = {};
            // 内层循环，遍历 x 的列
            for (intptr_t j = 0; j < x.shape[1]; ++j) {
                // 使用 ForceUnroll 展开循环，计算每个 k 对应的 map 值，并规约到 dist
                ForceUnroll<ilp_factor>{}([&](int k) {
                    auto val = map(x_rows[k][j], y_rows[k][j]);
                    dist[k] = reduce(dist[k], val);
                });
            }

            // 使用 ForceUnroll 展开循环，将计算结果映射到 out 的相应位置
            ForceUnroll<ilp_factor>{}([&](int k) {
                out(i + k, 0) = project(dist[k]);
            });
        }


注释：以上代码块完整地为给定的 C++ 代码添加了详细的注释，解释了每一行代码的作用和功能。
    } else {
        // 对于剩余不足 ilp_factor 的部分，使用普通循环处理
        for (; i + (ilp_factor - 1) < x.shape[0]; i += ilp_factor) {
            // 定义指向 x 和 y 中 ilp_factor 行的指针数组
            const T* x_rows[ilp_factor];
            const T* y_rows[ilp_factor];
            // 强制展开 ilp_factor 次循环，填充指针数组
            ForceUnroll<ilp_factor>{}([&](int k) {
                x_rows[k] = &x(i + k, 0);
                y_rows[k] = &y(i + k, 0);
            });

            // 初始化距离累积数组，每个元素对应一个行的距离
            AccumulateType dist[ilp_factor] = {};
            // 遍历行中的每个列
            for (intptr_t j = 0; j < x.shape[1]; ++j) {
                // 计算每个行的偏移量
                auto x_offset = j * xs;
                auto y_offset = j * ys;
                // 强制展开 ilp_factor 次循环，计算每个行的距离
                ForceUnroll<ilp_factor>{}([&](int k) {
                    auto val = map(x_rows[k][x_offset], y_rows[k][y_offset]);
                    // 更新距离累积值
                    dist[k] = reduce(dist[k], val);
                });
            }

            // 强制展开 ilp_factor 次循环，将计算结果写入输出
            ForceUnroll<ilp_factor>{}([&](int k) {
                out(i + k, 0) = project(dist[k]);
            });
        }
    }
    // 处理剩余的不足 ilp_factor 的行
    for (; i < x.shape[0]; ++i) {
        // 获取当前行的指针
        const T* x_row = &x(i, 0);
        const T* y_row = &y(i, 0);
        // 初始化距离累积值
        AccumulateType dist = {};
        // 遍历行中的每个列
        for (intptr_t j = 0; j < x.shape[1]; ++j) {
            // 计算每个列的偏移量
            auto val = map(x_row[j * xs], y_row[j * ys]);
            // 更新距离累积值
            dist = reduce(dist, val);
        }
        // 将计算结果写入输出
        out(i, 0) = project(dist);
    }
}

template <int ilp_factor=2, typename T,
          typename TransformFunc,
          typename ProjectFunc = Identity,
          typename ReduceFunc = Plus>
void transform_reduce_2d_(
    StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y,
    StridedView2D<const T> w, const TransformFunc& map,
    const ProjectFunc& project = Identity{},
    const ReduceFunc& reduce = Plus{}) {
    intptr_t i = 0;
    intptr_t xs = x.strides[1], ys = y.strides[1], ws = w.strides[1];
    // Result type of calling map
    using AccumulateType = typename std::decay<decltype(
        map(std::declval<T>(), std::declval<T>(), std::declval<T>()))>::type;

    // Loop over rows of matrices x, y, w with ILP (Inner Loop Parallelism) optimization
    for (; i + (ilp_factor - 1) < x.shape[0]; i += ilp_factor) {
        const T* x_rows[ilp_factor];
        const T* y_rows[ilp_factor];
        const T* w_rows[ilp_factor];

        // Fetch pointers to rows i, i+1, ..., i+ilp_factor-1 from matrices x, y, w
        ForceUnroll<ilp_factor>{}([&](int k) {
            x_rows[k] = &x(i + k, 0);
            y_rows[k] = &y(i + k, 0);
            w_rows[k] = &w(i + k, 0);
        });

        // Array to store accumulated values after applying map function
        AccumulateType dist[ilp_factor] = {};

        // Loop over columns of matrices x, y, w
        for (intptr_t j = 0; j < x.shape[1]; ++j) {
            // Apply map function to elements of matrices x, y, w and accumulate results
            ForceUnroll<ilp_factor>{}([&](int k) {
                auto val = map(x_rows[k][j * xs], y_rows[k][j * ys], w_rows[k][j * ws]);
                dist[k] = reduce(dist[k], val);
            });
        }

        // Apply project function to each accumulated value and store in output matrix out
        ForceUnroll<ilp_factor>{}([&](int k) {
            out(i + k, 0) = project(dist[k]);
        });
    }

    // Handle remaining rows that couldn't be processed with ILP
    for (; i < x.shape[0]; ++i) {
        const T* x_row = &x(i, 0);
        const T* y_row = &y(i, 0);
        const T* w_row = &w(i, 0);
        AccumulateType dist = {};

        // Loop over columns of matrices x, y, w
        for (intptr_t j = 0; j < x.shape[1]; ++j) {
            // Apply map function to elements of matrices x_row, y_row, w_row and accumulate results
            auto val = map(x_row[j * xs], y_row[j * ys], w_row[j * ws]);
            dist = reduce(dist, val);
        }

        // Apply project function to the accumulated value and store in output matrix out
        out(i, 0) = project(dist);
    }
}

struct MinkowskiDistance {
    double p_;

    // Operator function for 2D transformation with Minkowski distance calculation
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        const T p = static_cast<T>(p_);
        const T invp = static_cast<T>(1.0 / p_);

        // Call transform_reduce_2d_ function with map and project lambda functions for Minkowski distance
        transform_reduce_2d_(out, x, y, [p](T x, T y) INLINE_LAMBDA {
            auto diff = std::abs(x - y);
            return std::pow(diff, p);
        },
        [invp](T x) { return std::pow(x, invp); });
    }

    // Operator function for 2D transformation with weighted Minkowski distance calculation
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        const T p = static_cast<T>(p_);
        const T invp = static_cast<T>(1.0 / p_);

        // Call transform_reduce_2d_ function with map and project lambda functions for weighted Minkowski distance
        transform_reduce_2d_(out, x, y, w, [p](T x, T y, T w) INLINE_LAMBDA {
            auto diff = std::abs(x - y);
            return w * std::pow(diff, p);
        },
        [invp](T x) { return std::pow(x, invp); });
    }
};

struct EuclideanDistance {
    // Operator function for 2D transformation with Euclidean distance calculation
    template <typename T>
    // 定义函数调用操作符，用于计算二维数组 x 和 y 的转换归约操作，并将结果存储到 out 中
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 调用内部函数 transform_reduce_2d_ 进行二维转换归约操作，使用 lambda 表达式定义转换和归约逻辑
        transform_reduce_2d_(out, x, y, [](T x, T y) INLINE_LAMBDA {
            // 计算两个元素之间的差的平方
            auto diff = std::abs(x - y);
            return diff * diff;
        },
        // lambda 函数，用于对单个元素进行操作，在此处用于计算平方根
        [](T x) { return std::sqrt(x); });
    }

    // 定义模板函数调用操作符，用于计算带权重的二维数组 x、y 和 w 的转换归约操作，并将结果存储到 out 中
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 调用内部函数 transform_reduce_2d_ 进行带权重的二维转换归约操作，使用 lambda 表达式定义转换和归约逻辑
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            // 计算两个元素之间的差的平方，并乘以权重 w
            auto diff = std::abs(x - y);
            return w * (diff * diff);
        },
        // lambda 函数，用于对单个元素进行操作，在此处用于计算平方根
        [](T x) { return std::sqrt(x); });
    }
};

// 结构体定义：ChebyshevDistance，实现了两个重载的函数调用运算符
struct ChebyshevDistance {
    // 模板函数，计算两个二维数组 x 和 y 的切比雪夫距离，并存储在 out 中
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用 transform_reduce_2d_ 函数，将 x 和 y 二维数组元素进行转换和归约操作
        transform_reduce_2d_(out, x, y, [](T x, T y) INLINE_LAMBDA {
            // 内部 lambda 函数，计算两个元素 x 和 y 的差的绝对值
            return std::abs(x - y);
        },
        Identity{},  // 归约初始值的身份元素
        [](T x, T y) { return std::max(x, y); });  // 归约操作，返回两个值的最大值
    }

    // 模板函数，计算带权重的两个二维数组 x 和 y 的切比雪夫距离，并存储在 out 中
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 循环遍历 x 的第一维度
        for (intptr_t i = 0; i < x.shape[0]; ++i) {
            T dist = 0;  // 初始化距离为 0
            // 循环遍历 x 的第二维度
            for (intptr_t j = 0; j < x.shape[1]; ++j) {
                auto diff = std::abs(x(i, j) - y(i, j));  // 计算当前位置上 x 和 y 的差的绝对值
                if (w(i, j) > 0 && diff > dist) {  // 如果权重 w 大于 0 并且差的绝对值大于当前距离
                    dist = diff;  // 更新距离为差的绝对值
                }
            }
            out(i, 0) = dist;  // 将计算得到的距离存储在输出数组 out 中的第 i 行第 0 列
        }
    }
};

// 结构体定义：CityBlockDistance，实现了两个重载的函数调用运算符
struct CityBlockDistance {
    // 模板函数，计算两个二维数组 x 和 y 的曼哈顿距离，并存储在 out 中
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用 transform_reduce_2d_ 函数，将 x 和 y 二维数组元素进行转换和归约操作
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            // 内部 lambda 函数，计算两个元素 x 和 y 的差的绝对值
            return std::abs(x - y);
        });
    }

    // 模板函数，计算带权重的两个二维数组 x 和 y 的曼哈顿距离，并存储在 out 中
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 使用 transform_reduce_2d_ 函数，将 x 和 y 二维数组元素和权重 w 进行转换和归约操作
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            // 内部 lambda 函数，计算带权重的两个元素 x 和 y 的差的绝对值乘以权重 w
            return w * std::abs(x - y);
        });
    }
};

// 结构体定义：SquareEuclideanDistance，实现了两个重载的函数调用运算符
struct SquareEuclideanDistance {
    // 模板函数，计算两个二维数组 x 和 y 的平方欧几里得距离，并存储在 out 中
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用 transform_reduce_2d_ 函数，将 x 和 y 二维数组元素进行转换和归约操作
        transform_reduce_2d_(out, x, y, [](T x, T y) INLINE_LAMBDA {
            auto diff = x - y;  // 计算两个元素 x 和 y 的差
            return diff * diff;  // 返回差的平方
        });
    }

    // 模板函数，计算带权重的两个二维数组 x 和 y 的平方欧几里得距离，并存储在 out 中
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 使用 transform_reduce_2d_ 函数，将 x 和 y 二维数组元素和权重 w 进行转换和归约操作
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            auto diff = x - y;  // 计算两个元素 x 和 y 的差
            return w * diff * diff;  // 返回带权重的差的平方
        });
    }
};

// 结构体定义：BraycurtisDistance，实现了两个重载的函数调用运算符
struct BraycurtisDistance {
    // 内部结构定义：Acc，用于累加差和和的结构体
    template <typename T>
    struct Acc {
        Acc(): diff(0), sum(0) {}  // 默认构造函数，初始化 diff 和 sum 为 0
        T diff, sum;  // 成员变量，用于存储差和和
    };

    // 模板函数，计算两个二维数组 x 和 y 的布雷柯蒂斯距离，并存储在 out 中
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用 transform_reduce_2d_<2> 函数，将 x 和 y 二维数组元素进行转换和归约操作
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;  // 创建累加器对象 acc
            acc.diff = std::abs(x - y);  // 计算当前位置上 x 和 y 的差的绝对值
            acc.sum = std::abs(x + y);  // 计算当前位置上 x 和 y 的和的绝对值
            return acc;  // 返回累加器对象 acc
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.diff / acc.sum;  // 内部 lambda 函数，计算差的绝对值和和的绝对值的比值
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;  // 创建累加器对象 acc
            acc.diff = a.diff + b.diff;  // 累加两个累加器对象的差的绝对值
            acc.sum = a.sum + b.sum;  // 累加两个累加器对象的和的绝对值
            return acc;  // 返回累加器对象 acc
        });
    }

    // 模板函数，计算带权重的两个二维数组 x 和 y 的布雷柯蒂斯距离，并存储在 out 中
    template <typename T>


这段代码主要定义了几个结构体，每
    // 定义函数调用符重载，计算二维视图中的加权平均距离
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 计算加权平均距离的公式为 dist = (w * abs(x - y)).sum() / (w * abs(x + y)).sum()
        
        // 调用 transform_reduce_2d_ 函数，对输入的二维视图进行变换和归约操作
        transform_reduce_2d_(out, x, y, w,
            // 第一个 lambda 函数：计算每个元素的差值和加权值之积以及和之积
            [](T x, T y, T w) INLINE_LAMBDA {
                Acc<T> acc;
                acc.diff = w * std::abs(x - y);  // 计算加权差的绝对值
                acc.sum = w * std::abs(x + y);   // 计算加权和的绝对值
                return acc;
            },
            // 第二个 lambda 函数：根据每个元素的累积结果计算最终的距离值
            [](const Acc<T>& acc) INLINE_LAMBDA {
                return acc.diff / acc.sum;  // 计算最终的加权平均距离
            },
            // 第三个 lambda 函数：将两个累积器的结果合并
            [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
                Acc<T> acc;
                acc.diff = a.diff + b.diff;  // 合并差的累积结果
                acc.sum = a.sum + b.sum;     // 合并和的累积结果
                return acc;
            });
    }
};

// 结构体定义：CanberraDistance
struct CanberraDistance {
    // 模板方法：计算Canberra距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 计算Canberra距离公式：dist = (abs(x - y) / (abs(x) + abs(y))).sum()
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            auto num = std::abs(x - y);
            auto denom = std::abs(x) + std::abs(y);
            // 无分支替代方案：(denom == 0) ? 0 : num / denom;
            return num / (denom + (denom == 0));
        });
    }

    // 模板方法：计算带权重的Canberra距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 计算带权重的Canberra距离公式：dist = (w * abs(x - y) / (abs(x) + abs(y))).sum()
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            auto num = w * std::abs(x - y);
            auto denom = std::abs(x) + std::abs(y);
            // 无分支替代方案：(denom == 0) ? 0 : num / denom;
            return num / (denom + (denom == 0));
        });
    }
};

// 结构体定义：HammingDistance
struct HammingDistance {
    // 结构体：Acc用于累计非匹配和总数
    template <typename T>
    struct Acc {
        Acc(): nonmatches(0), total(0) {}
        T nonmatches, total;
    };

    // 模板方法：计算Hamming距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 计算Hamming距离公式：dist = (x != y) ? 1 : 0;
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = x != y;
            acc.total = 1;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.nonmatches / acc.total;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = a.nonmatches + b.nonmatches;
            acc.total = a.total + b.total;
            return acc;
        });
    }

    // 模板方法：计算带权重的Hamming距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 计算带权重的Hamming距离公式：dist = w * (x != y);
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = w * (x != y);
            acc.total = w;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.nonmatches / acc.total;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = a.nonmatches + b.nonmatches;
            acc.total = a.total + b.total;
            return acc;
        });
    }
};

// 结构体定义：DiceDistance
struct DiceDistance {
    // 结构体：Acc用于累计非匹配和总数
    template <typename T>
    struct Acc {
        Acc(): nonmatches(0), tt_matches(0) {}
        T nonmatches, tt_matches;
    };

    // 模板方法：计算Dice距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 计算Dice距离公式
        // Numerator: 2 * (x != y)
        // Denominator: x + y
        transform_reduce_2d_<5>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = 2 * (x != y);
            acc.tt_matches = x + y;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.nonmatches / acc.tt_matches;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = a.nonmatches + b.nonmatches;
            acc.tt_matches = a.tt_matches + b.tt_matches;
            return acc;
        });
    }
};
    // 定义一个函数对象，接受三个参数：输出视图 out，输入视图 x 和 y
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 调用 transform_reduce_2d_ 函数处理二维数据
        transform_reduce_2d_<2>(
            out,                                // 输出视图
            x,                                  // 输入视图 x
            y,                                  // 输入视图 y
            [](T x, T y) INLINE_LAMBDA {         // Lambda 函数：计算每个元素的累积值
                Acc<T> acc;
                acc.nonmatches = x * (1.0 - y) + y * (1.0 - x);  // 计算非匹配项
                acc.tt_matches = x * y;                         // 计算全局匹配项
                return acc;
            },
            [](const Acc<T>& acc) INLINE_LAMBDA {  // Lambda 函数：合并累积值到最终结果
                return acc.nonmatches / (2*acc.tt_matches + acc.nonmatches);  // 计算最终输出值
            },
            [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {  // Lambda 函数：合并两个累积值
                Acc<T> acc;
                acc.nonmatches = a.nonmatches + b.nonmatches;      // 累加非匹配项
                acc.tt_matches = a.tt_matches + b.tt_matches;      // 累加全局匹配项
                return acc;
            }
        );
    }
    
    // 定义一个模板函数对象，接受四个参数：输出视图 out，输入视图 x，y 和权重视图 w
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 调用 transform_reduce_2d_ 函数处理带权重的二维数据
        transform_reduce_2d_(
            out,                                    // 输出视图
            x,                                      // 输入视图 x
            y,                                      // 输入视图 y
            w,                                      // 权重视图 w
            [](T x, T y, T w) INLINE_LAMBDA {        // Lambda 函数：计算每个元素的累积值，考虑权重
                Acc<T> acc;
                acc.nonmatches = w * (x != y);       // 计算带权重的非匹配项
                acc.tt_matches = w * ((x != 0) & (y != 0));  // 计算带权重的全局匹配项
                return acc;
            },
            [](const Acc<T>& acc) INLINE_LAMBDA {    // Lambda 函数：合并累积值到最终结果
                return acc.nonmatches / (2*acc.tt_matches + acc.nonmatches);  // 计算最终输出值
            },
            [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {  // Lambda 函数：合并两个累积值
                Acc<T> acc;
                acc.nonmatches = a.nonmatches + b.nonmatches;      // 累加带权重的非匹配项
                acc.tt_matches = a.tt_matches + b.tt_matches;      // 累加带权重的全局匹配项
                return acc;
            }
        );
    }
};

// 结构体 JaccardDistance 开始
struct JaccardDistance {
    // 内部模板结构 Acc 开始
    template <typename T>
    struct Acc {
        Acc(): num(0), denom(0) {}
        T num, denom;
    };
    // 内部模板结构 Acc 结束

    // 重载运算符 () 开始，计算 Jaccard 距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用 transform_reduce_2d_ 函数进行双向遍历和归约，定义 lambda 函数以计算距离度量
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            // 计算 Jaccard 距离的分子和分母
            acc.num = (x != y) & ((x != 0) | (y != 0));
            acc.denom = (x != 0) | (y != 0);
            return acc;
        },
        // 第二个 lambda 函数，计算最终的距离值
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (acc.denom != 0) * (acc.num / (1 * (acc.denom == 0) + acc.denom));
        },
        // 第三个 lambda 函数，用于归约多个结果
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.num = a.num + b.num;
            acc.denom = a.denom + b.denom;
            return acc;
        });
    }
    // 重载运算符 () 结束

    // 重载运算符 ()，支持带权重的 Jaccard 距离计算 开始
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 使用 transform_reduce_2d_ 函数进行带权重的遍历和归约，定义 lambda 函数以计算带权重的距离度量
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            // 计算带权重的 Jaccard 距离的分子和分母
            acc.num = w * ((x != y) & ((x != 0) | (y != 0)));
            acc.denom = w * ((x != 0) | (y != 0));
            return acc;
        },
        // 第二个 lambda 函数，计算最终的带权重的距离值
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (acc.denom != 0) * (acc.num / (1 * (acc.denom == 0) + acc.denom));
        },
        // 第三个 lambda 函数，用于归约多个结果
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.num = a.num + b.num;
            acc.denom = a.denom + b.denom;
            return acc;
        });
    }
    // 重载运算符 ()，支持带权重的 Jaccard 距离计算 结束
};
// 结构体 JaccardDistance 结束

// 结构体 RogerstanimotoDistance 开始
struct RogerstanimotoDistance {
    // 内部模板结构 Acc 开始
    template <typename T>
    struct Acc {
        Acc(): ndiff(0), n(0) {}
        T ndiff, n;
    };
    // 内部模板结构 Acc 结束

    // 重载运算符 () 开始，计算 Rogerstanimoto 距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用 transform_reduce_2d_ 函数进行双向遍历和归约，定义 lambda 函数以计算距离度量
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            // 计算 Rogerstanimoto 距离的分子和分母
            acc.ndiff = (x != 0) != (y != 0);
            acc.n = 1;
            return acc;
        },
        // 第二个 lambda 函数，计算最终的距离值
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (acc.n + acc.ndiff);
        },
        // 第三个 lambda 函数，用于归约多个结果
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ndiff = a.ndiff + b.ndiff;
            acc.n = a.n + b.n;
            return acc;
        });
    }
    // 重载运算符 () 结束
};
// 结构体 RogerstanimotoDistance 结束
    // 定义函数对象的调用运算符，用于计算二维视图的变换和规约操作
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 调用内部函数 transform_reduce_2d_，传入四个二维视图和三个 lambda 表达式
        transform_reduce_2d_(out, x, y, w, 
            // 第一个 lambda 表达式：计算每个元素的累加器
            [](T x, T y, T w) INLINE_LAMBDA {
                Acc<T> acc;
                // 计算不同位（bitwise）的数量差异
                acc.ndiff = w * ((x != 0) != (y != 0));
                // 设置权重
                acc.n = w;
                return acc;
            },
            // 第二个 lambda 表达式：计算累加器的最终结果
            [](const Acc<T>& acc) INLINE_LAMBDA {
                // 根据累加器计算权重和数量差异的比例
                return (2 * acc.ndiff) / (acc.n + acc.ndiff);
            },
            // 第三个 lambda 表达式：合并两个累加器的操作
            [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
                Acc<T> acc;
                // 合并数量差异和权重
                acc.ndiff = a.ndiff + b.ndiff;
                acc.n = a.n + b.n;
                return acc;
            });
    }
};

// 定义 Kulczynski1Distance 结构体
struct Kulczynski1Distance {
    // 定义模板类 Acc，用于存储计数值 ntt 和 ndiff
    template <typename T>
    struct Acc {
        Acc(): ntt(0), ndiff(0) {}  // 默认构造函数初始化 ntt 和 ndiff
        T ntt, ndiff;  // 存储 ntt 和 ndiff 的变量
    };

    // 重载 () 运算符，计算两个二维数组 x 和 y 的 Kulczynski1 距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 调用 transform_reduce_2d_ 函数，处理 x 和 y 数组的每个元素
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 对象 acc，用于存储计算结果
            acc.ntt = (x != 0) & (y != 0);  // 计算 x 和 y 不为零的交集数量
            acc.ndiff = (x != 0) != (y != 0);  // 计算 x 和 y 不相等的数量
            return acc;  // 返回计算结果 acc
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.ntt / acc.ndiff;  // 计算 Kulczynski1 距离
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 对象 acc，用于存储合并后的计算结果
            acc.ntt = a.ntt + b.ntt;  // 合并 ntt 计数
            acc.ndiff = a.ndiff + b.ndiff;  // 合并 ndiff 计数
            return acc;  // 返回合并后的计算结果 acc
        });
    }

    // 重载 () 运算符，计算带权重的两个二维数组 x、y 的 Kulczynski1 距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 调用 transform_reduce_2d_ 函数，处理 x、y、w 数组的每个元素
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 对象 acc，用于存储计算结果
            acc.ntt = w * ((x != 0) & (y != 0));  // 带权重计算 x 和 y 不为零的交集数量
            acc.ndiff = w * ((x != 0) != (y != 0));  // 带权重计算 x 和 y 不相等的数量
            return acc;  // 返回计算结果 acc
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.ntt / acc.ndiff;  // 计算 Kulczynski1 距离
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 对象 acc，用于存储合并后的计算结果
            acc.ntt = a.ntt + b.ntt;  // 合并 ntt 计数
            acc.ndiff = a.ndiff + b.ndiff;  // 合并 ndiff 计数
            return acc;  // 返回合并后的计算结果 acc
        });
    }
};

// 定义 RussellRaoDistance 结构体
struct RussellRaoDistance {
    // 定义模板类 Acc，用于存储计数值 ntt 和 n
    template <typename T>
    struct Acc {
        Acc(): ntt(0), n(0) {}  // 默认构造函数初始化 ntt 和 n
        T ntt, n;  // 存储 ntt 和 n 的变量
    };

    // 重载 () 运算符，计算两个二维数组 x 和 y 的 Russell-Rao 距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 调用 transform_reduce_2d_ 函数，处理 x 和 y 数组的每个元素
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 对象 acc，用于存储计算结果
            acc.ntt = (x != 0) & (y != 0);  // 计算 x 和 y 不为零的交集数量
            acc.n = 1;  // 设置 n 的初始值为 1
            return acc;  // 返回计算结果 acc
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (acc.n - acc.ntt) / acc.n;  // 计算 Russell-Rao 距离
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 对象 acc，用于存储合并后的计算结果
            acc.ntt = a.ntt + b.ntt;  // 合并 ntt 计数
            acc.n = a.n + b.n;  // 合并 n 计数
            return acc;  // 返回合并后的计算结果 acc
        });
    }

    // 重载 () 运算符，计算带权重的两个二维数组 x、y 的 Russell-Rao 距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 调用 transform_reduce_2d_ 函数，处理 x、y、w 数组的每个元素
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 对象 acc，用于存储计算结果
            acc.ntt = w * ((x != 0) & (y != 0));  // 带权重计算 x 和 y 不为零的交集数量
            acc.n = w;  // 设置 n 的值为权重 w
            return acc;  // 返回计算结果 acc
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (acc.n - acc.ntt) / acc.n;  // 计算 Russell-Rao 距离
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 对象 acc，用于存储合并后的计算结果
            acc.ntt = a.ntt + b.ntt;  // 合并 ntt 计数
            acc.n = a.n + b.n;  // 合并 n 计数
            return acc;  // 返回合并后的计算结果 acc
        });
    }
};

// 定义 SokalmichenerDistance 结构体
struct SokalmichenerDistance {
    template <typename T>
    struct Acc {
        Acc(): ntt(0), ndiff(0), n(0) {}
        T ntt, ndiff, n;
    };
    
    // 定义了一个模板函数，计算两个二维数组的转换和归约结果
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用4作为模板参数调用了transform_reduce_2d_函数，对x和y进行转换和归约操作
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            // 初始化累加器Acc，并计算非零值情况下的ntt、ndiff和n值
            Acc<T> acc;
            acc.ntt = (x != 0) & (y != 0);
            acc.ndiff = (x != 0) != (y != 0);
            acc.n = 1;
            return acc;
        },
        // 计算累加器Acc的归约结果，返回ndiff和n的加权比例
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (acc.ndiff + acc.n);
        },
        // 将两个累加器Acc合并，计算它们的ntt、ndiff和n的总和
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.ndiff = a.ndiff + b.ndiff;
            acc.n = a.n + b.n;
            return acc;
        });
    }
    
    // 定义了一个模板函数，计算三个二维数组的转换和归约结果，考虑了权重w
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 使用transform_reduce_2d_函数，对x、y和w进行转换和归约操作
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            // 初始化累加器Acc，并考虑权重w，计算加权后的ntt、ndiff和n值
            Acc<T> acc;
            acc.ntt = w * ((x != 0) & (y != 0));
            acc.ndiff = w * ((x != 0) != (y != 0));
            acc.n = w;
            return acc;
        },
        // 计算累加器Acc的归约结果，返回ndiff和n的加权比例
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (acc.ndiff + acc.n);
        },
        // 将两个累加器Acc合并，计算它们的ntt、ndiff和n的总和
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.ndiff = a.ndiff + b.ndiff;
            acc.n = a.n + b.n;
            return acc;
        });
    }
};

// SokalsneathDistance 结构体定义
struct SokalsneathDistance {
    // Acc 结构模板定义
    template <typename T>
    struct Acc {
        Acc(): ntt(0), ndiff(0) {}  // Acc 结构体构造函数初始化 ntt 和 ndiff
        T ntt, ndiff;  // 成员变量 ntt 和 ndiff
    };

    // operator() 函数模板，计算 Sokalsneath 距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用 transform_reduce_2d_ 函数模板进行计算
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 结构体对象 acc
            // 计算 ntt 和 ndiff
            acc.ntt = (x != 0) & (y != 0);  // 如果 x 和 y 都不为零，则 ntt 为真
            acc.ndiff = (x != 0) != (y != 0);  // 计算 x 和 y 是否不相同
            return acc;  // 返回 Acc 结构体对象
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            // 计算 Sokalsneath 距离
            return (2 * acc.ndiff) / (2 * acc.ndiff + acc.ntt);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            // 合并两个 Acc 结构体对象的结果
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;  // 合并 ntt
            acc.ndiff = a.ndiff + b.ndiff;  // 合并 ndiff
            return acc;  // 返回合并后的 Acc 结构体对象
        });
    }

    // operator() 函数模板的重载，计算 Sokalsneath 距离，带权重
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 使用 transform_reduce_2d_ 函数模板进行计算
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 结构体对象 acc
            // 计算带权重的 ntt 和 ndiff
            acc.ntt = w * ((x != 0) & (y != 0));  // 如果 x 和 y 都不为零，则 ntt 乘以权重 w
            acc.ndiff = w * ((x != 0) != (y != 0));  // 计算带权重的 ndiff
            return acc;  // 返回 Acc 结构体对象
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            // 计算 Sokalsneath 距离
            return (2 * acc.ndiff) / (2 * acc.ndiff + acc.ntt);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            // 合并两个 Acc 结构体对象的结果
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;  // 合并带权重的 ntt
            acc.ndiff = a.ndiff + b.ndiff;  // 合并带权重的 ndiff
            return acc;  // 返回合并后的 Acc 结构体对象
        });
    }
};

// YuleDistance 结构体定义
struct YuleDistance {
    // Acc 结构模板定义
    template <typename T>
    struct Acc {
        Acc(): ntt(0), nft(0), nff(0), ntf(0) {}  // Acc 结构体构造函数初始化各成员变量
        intptr_t ntt, nft, nff, ntf;  // 成员变量 ntt, nft, nff, ntf
    };

    // operator() 函数模板，计算 Yule 距离
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // 使用 transform_reduce_2d_<2> 函数模板进行计算
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;  // 创建 Acc 结构体对象 acc
            // 计算 Yule 距离中的各项
            acc.ntt = (x != 0) & (y != 0);  // x 和 y 都不为零
            acc.ntf = (x != 0) & (y == 0);  // x 不为零，y 为零
            acc.nft = (x == 0) & (y != 0);  // x 为零，y 不为零
            acc.nff = (x == 0) & (y == 0);  // x 和 y 都为零
            return acc;  // 返回 Acc 结构体对象
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            // 计算 Yule 距离
            intptr_t half_R = acc.ntf * acc.nft;
            return (2. * half_R) / (acc.ntt * acc.nff + half_R + (half_R == 0));
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            // 合并两个 Acc 结构体对象的结果
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;  // 合并 ntt
            acc.nft = a.nft + b.nft;  // 合并 nft
            acc.nff = a.nff + b.nff;  // 合并 nff
            acc.ntf = a.ntf + b.ntf;  // 合并 ntf
            return acc;  // 返回合并后的 Acc 结构体对象
        });
    }

    // 下面还有未完成的代码，未提供完整部分的注释
    // 定义一个函数调用运算符，接受四个 StridedView2D 类型的参数 out, x, y, w，并执行以下操作
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // 调用 transform_reduce_2d_ 模板函数，处理二维数据，使用 lambda 表达式计算每个元素的 Acc 结构
        transform_reduce_2d_<2>(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            // 定义并初始化 Acc 结构
            Acc<T> acc;
            // 计算 acc 的 ntt 成员
            acc.ntt = w * ((x != 0) & (y != 0));
            // 计算 acc 的 ntf 成员
            acc.ntf = w * ((x != 0) & (!(y != 0)));
            // 计算 acc 的 nft 成员
            acc.nft = w * ((!(x != 0)) & (y != 0));
            // 计算 acc 的 nff 成员
            acc.nff = w * ((!(x != 0)) & (!(y != 0)));
            // 返回计算结果的 acc 结构
            return acc;
        },
        // 第二个参数：处理每个 Acc 结构并返回一个 double 值的 lambda 表达式
        [](const Acc<T>& acc) INLINE_LAMBDA {
            // 计算 half_R 值
            intptr_t half_R = acc.ntf * acc.nft;
            // 返回计算结果
            return (2. * half_R) / (acc.ntt * acc.nff + half_R + (half_R == 0));
        },
        // 第三个参数：接受两个 Acc 结构并返回一个合并后的 Acc 结构的 lambda 表达式
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            // 定义并初始化 Acc 结构
            Acc<T> acc;
            // 合并 a 和 b 的 ntt 成员
            acc.ntt = a.ntt + b.ntt;
            // 合并 a 和 b 的 nft 成员
            acc.nft = a.nft + b.nft;
            // 合并 a 和 b 的 nff 成员
            acc.nff = a.nff + b.nff;
            // 合并 a 和 b 的 ntf 成员
            acc.ntf = a.ntf + b.ntf;
            // 返回合并后的 acc 结构
            return acc;
        });
    }
};



# 这行代码是一个单独的分号，通常用于结束语句或代码块，但在这里没有任何上下文，是一个语法错误。
```