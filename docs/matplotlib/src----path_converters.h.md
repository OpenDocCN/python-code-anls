# `D:\src\scipysrc\matplotlib\src\path_converters.h`

```
/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PATH_CONVERTERS_H
#define MPL_PATH_CONVERTERS_H

#include <cmath>
#include <cstdint>

#include "agg_clip_liang_barsky.h"
#include "mplutils.h"
#include "agg_conv_segmentator.h"

/*
 This file contains a number of vertex converters that modify
 paths. They all work as iterators, where the output is generated
 on-the-fly, and don't require a copy of the full data.

 Each class represents a discrete step in a "path-cleansing" pipeline.
 They are currently applied in the following order in the Agg backend:

   1. Affine transformation (implemented in Agg, not here)

   2. PathNanRemover: skips over segments containing non-finite numbers
      by inserting MOVETO commands

   3. PathClipper: Clips line segments to a given rectangle.  This is
      helpful for data reduction, and also to avoid a limitation in
      Agg where coordinates cannot be larger than 24-bit signed
      integers.

   4. PathSnapper: Rounds the path to the nearest center-pixels.
      This makes rectilinear curves look much better.

   5. PathSimplifier: Removes line segments from highly dense paths
      that would not have an impact on their appearance.  Speeds up
      rendering and reduces file sizes.

   6. curve-to-line-segment conversion (implemented in Agg, not here)

   7. stroking (implemented in Agg, not here)
 */

/************************************************************
 This is a base class for vertex converters that need to queue their
 output.  It is designed to be as fast as possible vs. the STL's queue
 which is more flexible.
 */
template <int QueueSize>
class EmbeddedQueue
{
  protected:
    // Constructor initializing queue indices
    EmbeddedQueue() : m_queue_read(0), m_queue_write(0)
    {
        // empty
    }

    struct item
    {
        item()
        {
        }

        // Set method for setting command and coordinates
        inline void set(const unsigned cmd_, const double x_, const double y_)
        {
            cmd = cmd_;
            x = x_;
            y = y_;
        }
        unsigned cmd; // Command type
        double x;     // X-coordinate
        double y;     // Y-coordinate
    };
    int m_queue_read;    // Index to read from queue
    int m_queue_write;   // Index to write to queue
    item m_queue[QueueSize]; // Array holding queued items

    // Push method to add a new item to the queue
    inline void queue_push(const unsigned cmd, const double x, const double y)
    {
        m_queue[m_queue_write++].set(cmd, x, y);
    }

    // Check if queue is non-empty
    inline bool queue_nonempty()
    {
        return m_queue_read < m_queue_write;
    }

    // Pop method to retrieve an item from the queue
    inline bool queue_pop(unsigned *cmd, double *x, double *y)
    {
        if (queue_nonempty()) {
            const item &front = m_queue[m_queue_read++];
            *cmd = front.cmd;
            *x = front.x;
            *y = front.y;

            return true;
        }

        // Reset queue indices when queue is empty
        m_queue_read = 0;
        m_queue_write = 0;

        return false;
    }

    // Clear method to reset queue indices
    inline void queue_clear()
    {
        m_queue_read = 0;
        m_queue_write = 0;
    }
};

/* Defines when path segment types have more than one vertex */
static const size_t num_extra_points_map[] =
    {0, 0, 0, 1,
     2, 0, 0, 0,
     0, 0, 0, 0,
     0, 0, 0, 0
    };

#endif // MPL_PATH_CONVERTERS_H
/* 一个简单的线性同余随机数生成器的实现。
   这是一个经典且快速的随机数生成器，适用于我们画线的目的，但不应用于像加密这样重要的事务。
   我们自己实现而不使用 C 标准库，以确保种子状态不与其他第三方代码共享。
   对于兼容性原因，我们要求不使用晚于 C++98 的功能。 */
class RandomNumberGenerator
{
private:
    /* 这些是来自 MS Visual C++ 的常数，其模数为 2^32，因此可以节省显式的模运算 */
    static const uint32_t a = 214013;
    static const uint32_t c = 2531011;
    uint32_t m_seed;

public:
    RandomNumberGenerator() : m_seed(0) {}
    RandomNumberGenerator(int seed) : m_seed(seed) {}

    /* 设置随机数生成器的种子值 */
    void seed(int seed)
    {
        m_seed = seed;
    }

    /* 返回一个 [0, 1) 之间的随机双精度浮点数 */
    double get_double()
    {
        m_seed = (a * m_seed + c);
        return (double)m_seed / (double)(1LL << 32);
    }
};

/*
  PathNanRemover 是一个顶点转换器，用于从顶点列表中移除非有限值，并在必要时插入 MOVETO 命令以跳过它们。
  如果曲线段中包含至少一个非有限值，则整个曲线段将被跳过。
 */
template <class VertexSource>
class PathNanRemover : protected EmbeddedQueue<4>
{
    VertexSource *m_source;
    bool m_remove_nans;
    bool m_has_codes;
    bool valid_segment_exists;
    bool m_last_segment_valid;
    bool m_was_broken;
    double m_initX;
    double m_initY;

  public:
    /* 如果路径包含贝塞尔曲线段或闭合循环，则 has_codes 应设置为 true，
     * 因为这需要一个较慢的算法来移除 NaN 值。当不确定时，设置为 true。 */
    PathNanRemover(VertexSource &source, bool remove_nans, bool has_codes)
        : m_source(&source), m_remove_nans(remove_nans), m_has_codes(has_codes),
          m_last_segment_valid(false), m_was_broken(false),
          m_initX(nan("")), m_initY(nan(""))
    {
        // 在遇到第一个有效（无 NaN）命令之前，忽略所有 close/end_poly 命令
        valid_segment_exists = false;
    }

    /* 将路径回绕到指定的路径 ID */
    inline void rewind(unsigned path_id)
    {
        queue_clear();
        m_source->rewind(path_id);
    }

    inline unsigned vertex(double *x, double *y)
    {
        // ...
    }
};

/************************************************************
 PathClipper 使用 Liang-Barsky 线段裁剪算法（在 Agg 中实现）来将路径裁剪到给定的矩形。
 线段永远不会延伸到矩形外部。曲线段不被裁剪，而是始终完整包含在内。
 */
template <class VertexSource>
class PathClipper : public EmbeddedQueue<3>
{
    VertexSource *m_source;
    bool m_do_clipping;
    agg::rect_base<double> m_cliprect;
    double m_lastX;
    double m_lastY;
    bool m_moveto;
    // 初始 X 坐标
    double m_initX;
    // 初始 Y 坐标
    double m_initY;
    // 是否已初始化的标志
    bool m_has_init;
    // 是否已被裁剪的标志

  public:
    // 构造函数，初始化 PathClipper 实例
    PathClipper(VertexSource &source, bool do_clipping, double width, double height)
        : m_source(&source),
          // 是否执行裁剪操作
          m_do_clipping(do_clipping),
          // 裁剪矩形，初始化为 (-1.0, -1.0, width + 1.0, height + 1.0)
          m_cliprect(-1.0, -1.0, width + 1.0, height + 1.0),
          // 上一个 X 坐标，初始化为 NaN
          m_lastX(nan("")),
          // 上一个 Y 坐标，初始化为 NaN
          m_lastY(nan("")),
          // 是否为移动到新点操作
          m_moveto(true),
          // 初始 X 坐标，初始化为 NaN
          m_initX(nan("")),
          // 初始 Y 坐标，初始化为 NaN
          m_initY(nan("")),
          // 是否已初始化的标志，初始化为 false
          m_has_init(false),
          // 是否已被裁剪的标志，初始化为 false
          m_was_clipped(false)
    {
        // 构造函数为空
    }

    // 构造函数，初始化 PathClipper 实例
    PathClipper(VertexSource &source, bool do_clipping, const agg::rect_base<double> &rect)
        : m_source(&source),
          // 是否执行裁剪操作
          m_do_clipping(do_clipping),
          // 裁剪矩形，初始化为提供的 rect，并进行边界扩展
          m_cliprect(rect),
          // 上一个 X 坐标，初始化为 NaN
          m_lastX(nan("")),
          // 上一个 Y 坐标，初始化为 NaN
          m_lastY(nan("")),
          // 是否为移动到新点操作
          m_moveto(true),
          // 初始 X 坐标，初始化为 NaN
          m_initX(nan("")),
          // 初始 Y 坐标，初始化为 NaN
          m_initY(nan("")),
          // 是否已初始化的标志，初始化为 false
          m_has_init(false),
          // 是否已被裁剪的标志，初始化为 false
          m_was_clipped(false)
    {
        // 调整裁剪矩形的边界，扩展 1.0 单位
        m_cliprect.x1 -= 1.0;
        m_cliprect.y1 -= 1.0;
        m_cliprect.x2 += 1.0;
        m_cliprect.y2 += 1.0;
    }

    // 重置路径处理状态
    inline void rewind(unsigned path_id)
    {
        // 重置初始化标志为 false
        m_has_init = false;
        // 重置是否被裁剪的标志为 false
        m_was_clipped = false;
        // 设置为移动到新点的状态
        m_moveto = true;
        // 调用顶点源的重置方法
        m_source->rewind(path_id);
    }

    // 绘制裁剪后的线段
    int draw_clipped_line(double x0, double y0, double x1, double y1,
                          bool closed=false)
    {
        // 对线段进行裁剪，返回裁剪状态
        unsigned moved = agg::clip_line_segment(&x0, &y0, &x1, &y1, m_cliprect);
        // 更新是否被裁剪的状态
        m_was_clipped = m_was_clipped || (moved != 0);
        // 如果未完全被裁剪
        if (moved < 4) {
            // 如果第一个点被移动或者是移动到新点操作
            if (moved & 1 || m_moveto) {
                // 将移动到命令推入队列
                queue_push(agg::path_cmd_move_to, x0, y0);
            }
            // 将线段到命令推入队列
            queue_push(agg::path_cmd_line_to, x1, y1);
            // 如果是闭合路径且未被裁剪
            if (closed && !m_was_clipped) {
                // 如果终点未移动，则关闭路径
                queue_push(agg::path_cmd_end_poly | agg::path_flags_close,
                           x1, y1);
            }

            // 设置非移动到新点状态
            m_moveto = false;
            return 1;
        }

        return 0;
    }

    // 获取顶点
    unsigned vertex(double *x, double *y)
    {
        // 待实现的方法，用于获取顶点
    }
};

/************************************************************
 PathSnapper rounds vertices to their nearest center-pixels.  This
 makes rectilinear paths (rectangles, horizontal and vertical lines
 etc.) look much cleaner.
*/
enum e_snap_mode {
    SNAP_AUTO,   // 自动模式：根据路径特性决定是否进行像素对齐
    SNAP_FALSE,  // 关闭模式：不进行像素对齐
    SNAP_TRUE    // 强制模式：始终进行像素对齐
};

template <class VertexSource>
class PathSnapper
{
  private:
    VertexSource *m_source;
    bool m_snap;         // 是否执行像素对齐
    double m_snap_value; // 像素对齐的偏移值

    static bool should_snap(VertexSource &path, e_snap_mode snap_mode, unsigned total_vertices)
    {
        // 如果路径仅包含直线（水平或垂直），则应该进行像素对齐
        double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        unsigned code;

        switch (snap_mode) {
        case SNAP_AUTO:
            // 如果顶点数超过1024个，则不进行像素对齐
            if (total_vertices > 1024) {
                return false;
            }

            // 获取第一个顶点的坐标
            code = path.vertex(&x0, &y0);
            if (code == agg::path_cmd_stop) {
                return false;
            }

            // 遍历路径中的每个顶点，检查是否仅包含直线段
            while ((code = path.vertex(&x1, &y1)) != agg::path_cmd_stop) {
                switch (code) {
                case agg::path_cmd_curve3:
                case agg::path_cmd_curve4:
                    // 如果路径包含曲线，则不进行像素对齐
                    return false;
                case agg::path_cmd_line_to:
                    // 如果直线段不是水平或垂直，则不进行像素对齐
                    if (fabs(x0 - x1) >= 1e-4 && fabs(y0 - y1) >= 1e-4) {
                        return false;
                    }
                }
                x0 = x1;
                y0 = y1;
            }

            // 路径符合条件，进行像素对齐
            return true;
        case SNAP_FALSE:
            // 关闭模式，不进行像素对齐
            return false;
        case SNAP_TRUE:
            // 强制模式，始终进行像素对齐
            return true;
        }

        return false;
    }

  public:
    /*
      snap_mode should be one of:
        - SNAP_AUTO: Examine the path to determine if it should be snapped
        - SNAP_TRUE: Force snapping
        - SNAP_FALSE: No snapping
    */
    PathSnapper(VertexSource &source,
                e_snap_mode snap_mode,
                unsigned total_vertices = 15,
                double stroke_width = 0.0)
        : m_source(&source)
    {
        // 根据路径特性确定是否进行像素对齐
        m_snap = should_snap(source, snap_mode, total_vertices);

        // 如果需要像素对齐，则计算像素对齐的偏移值
        if (m_snap) {
            int is_odd = mpl_round_to_int(stroke_width) % 2;
            m_snap_value = (is_odd) ? 0.5 : 0.0;
        }

        // 重置路径源的状态
        source.rewind(0);
    }

    inline void rewind(unsigned path_id)
    {
        // 重置路径源的状态
        m_source->rewind(path_id);
    }

    inline unsigned vertex(double *x, double *y)
    {
        unsigned code;
        // 获取路径中的顶点，并根据需要进行像素对齐
        code = m_source->vertex(x, y);
        if (m_snap && agg::is_vertex(code)) {
            // 执行像素对齐操作
            *x = floor(*x + 0.5) + m_snap_value;
            *y = floor(*y + 0.5) + m_snap_value;
        }
        return code;
    }

    inline bool is_snapping()
    {
        // 返回当前是否正在执行像素对齐
        return m_snap;
    }
};

/************************************************************
 PathSimplifier reduces the number of vertices in a dense path without
 changing its appearance.
*/
template <class VertexSource>
class PathSimplifier : protected EmbeddedQueue<9>
{
  public:
    /* Set simplify to true to perform simplification */
    // 构造函数，初始化 PathSimplifier 对象
    PathSimplifier(VertexSource &source, bool do_simplify, double simplify_threshold)
        : m_source(&source),
          m_simplify(do_simplify),
          /* we square simplify_threshold so that we can compute
             norms without doing the square root every step. */
          // 将 simplify_threshold 的平方存储起来，避免每步都计算平方根
          m_simplify_threshold(simplify_threshold * simplify_threshold),

          m_moveto(true),
          m_after_moveto(false),
          m_clipped(false),

          // the x, y values from last iteration
          // 上一次迭代中的 x, y 值
          m_lastx(0.0),
          m_lasty(0.0),

          // the dx, dy comprising the original vector, used in conjunction
          // with m_currVecStart* to define the original vector.
          // 原始向量的 dx, dy，与 m_currVecStart* 结合使用来定义原始向量
          m_origdx(0.0),
          m_origdy(0.0),

          // the squared norm of the original vector
          // 原始向量的平方范数
          m_origdNorm2(0.0),

          // maximum squared norm of vector in forward (parallel) direction
          // 正向（平行）方向向量的最大平方范数
          m_dnorm2ForwardMax(0.0),
          // maximum squared norm of vector in backward (anti-parallel) direction
          // 反向（反平行）方向向量的最大平方范数
          m_dnorm2BackwardMax(0.0),

          // was the last point the furthest from lastWritten in the
          // forward (parallel) direction?
          // 上一个点在正向（平行）方向上是否距离 lastWritten 最远？
          m_lastForwardMax(false),
          // was the last point the furthest from lastWritten in the
          // backward (anti-parallel) direction?
          // 上一个点在反向（反平行）方向上是否距离 lastWritten 最远？
          m_lastBackwardMax(false),

          // added to queue when _push is called
          // 调用 _push 时添加到队列中的 x, y 值
          m_nextX(0.0),
          m_nextY(0.0),

          // added to queue when _push is called if any backwards
          // (anti-parallel) vectors were observed
          // 如果观察到任何反向（反平行）向量，则调用 _push 时添加到队列中的 x, y 值
          m_nextBackwardX(0.0),
          m_nextBackwardY(0.0),

          // start of the current vector that is being simplified
          // 正在简化的当前向量的起点
          m_currVecStartX(0.0),
          m_currVecStartY(0.0)
    {
        // empty
    }

    // 重置到指定路径 ID 的起始点
    inline void rewind(unsigned path_id)
    {
        queue_clear();
        m_moveto = true;
        m_source->rewind(path_id);
    }

    // 获取顶点坐标
    unsigned vertex(double *x, double *y)
    }

  private:
    VertexSource *m_source;
    bool m_simplify;
    double m_simplify_threshold;

    bool m_moveto;
    bool m_after_moveto;
    bool m_clipped;
    double m_lastx, m_lasty;

    double m_origdx;
    double m_origdy;
    double m_origdNorm2;
    double m_dnorm2ForwardMax;
    double m_dnorm2BackwardMax;
    bool m_lastForwardMax;
    bool m_lastBackwardMax;
    double m_nextX;
    double m_nextY;
    double m_nextBackwardX;
    double m_nextBackwardY;
    double m_currVecStartX;
    double m_currVecStartY;

    // 将 x, y 值推入队列
    inline void _push(double *x, double *y)
    {
        // 检查是否需要推回向量，如果最大反向向量大于零则需要推回
        bool needToPushBack = (m_dnorm2BackwardMax > 0.0);
    
        /* 如果观察到任何反向（反平行）向量，
           那么我们需要推送前向和后向向量。 */
        if (needToPushBack) {
            /* 如果最后一个向量是在前向方向上的最大值，
               那么我们需要先推后向再推前向。否则，
               最后一个向量是在后向方向上的最大值，
               或者介于两者之间，无论如何，我们安全地先推前向再推后向。 */
            if (m_lastForwardMax) {
                queue_push(agg::path_cmd_line_to, m_nextBackwardX, m_nextBackwardY);
                queue_push(agg::path_cmd_line_to, m_nextX, m_nextY);
            } else {
                queue_push(agg::path_cmd_line_to, m_nextX, m_nextY);
                queue_push(agg::path_cmd_line_to, m_nextBackwardX, m_nextBackwardY);
            }
        } else {
            /* 如果我们没有观察到反向向量，只需推前向。 */
            queue_push(agg::path_cmd_line_to, m_nextX, m_nextY);
        }
    
        /* 如果我们在当前线段和下一条线段之间裁剪了一些段，
           我们还需要移动到最后一个点。 */
        if (m_clipped) {
            queue_push(agg::path_cmd_move_to, m_lastx, m_lasty);
        } else if ((!m_lastForwardMax) && (!m_lastBackwardMax)) {
            /* 如果最后一条线段不是最长的线段，那么移动到序列中上一条线段的终点。
               只有在未裁剪时才执行此操作，因为在裁剪的情况下，lastx，lasty 不是刚刚绘制的线的一部分。 */
    
            /* 如果没有伪影，这将是 move_to */
            queue_push(agg::path_cmd_line_to, m_lastx, m_lasty);
        }
    
        /* 现在重置所有变量以准备处理下一条线段 */
        m_origdx = *x - m_lastx;
        m_origdy = *y - m_lasty;
        m_origdNorm2 = m_origdx * m_origdx + m_origdy * m_origdy;
    
        m_dnorm2ForwardMax = m_origdNorm2;
        m_lastForwardMax = true;
        m_currVecStartX = m_queue[m_queue_write - 1].x;
        m_currVecStartY = m_queue[m_queue_write - 1].y;
        m_lastx = m_nextX = *x;
        m_lasty = m_nextY = *y;
        m_dnorm2BackwardMax = 0.0;
        m_lastBackwardMax = false;
    
        m_clipped = false;
    }
};

template <class VertexSource>
class Sketch
{
  public:
    /*
       scale: 每条线段垂直于原始线的摆动比例（以像素为单位）

       length: 摆动沿原始线的基础波长（以像素为单位）

       randomness: 决定摆动长度随机收缩和扩展的因子
    */
    Sketch(VertexSource &source, double scale, double length, double randomness)
        : m_source(&source),
          m_scale(scale),
          m_length(length),
          m_randomness(randomness),
          m_segmented(source),
          m_last_x(0.0),
          m_last_y(0.0),
          m_has_last(false),
          m_p(0.0),
          m_rand(0)
    {
        // 将“游标”重置为初始状态
        rewind(0);
        const double d_M_PI = 3.14159265358979323846;
        // 计算用于参数 p 的比例因子
        m_p_scale = (2.0 * d_M_PI) / (m_length * m_randomness);
        // 预先计算对数值，用于随机长度计算
        m_log_randomness = 2.0 * log(m_randomness);
    }

    unsigned vertex(double *x, double *y)
    {
        if (m_scale == 0.0) {
            // 若缩放比例为零，直接从源顶点获取坐标
            return m_source->vertex(x, y);
        }

        // 从分段路径获取顶点坐标
        unsigned code = m_segmented.vertex(x, y);

        if (code == agg::path_cmd_move_to) {
            // 若为移动命令，重置“游标”状态和参数 p
            m_has_last = false;
            m_p = 0.0;
        }

        if (m_has_last) {
            // 希望“游标”沿正弦波进行随机移动
            double d_rand = m_rand.get_double();
            // 计算新的参数 p，根据随机值和预计算的对数值
            m_p += exp(d_rand * m_log_randomness);
            // 计算两点之间的距离
            double den = m_last_x - *x;
            double num = m_last_y - *y;
            double len = num * num + den * den;
            m_last_x = *x;
            m_last_y = *y;
            if (len != 0) {
                len = sqrt(len);
                // 计算正弦波对应的摆动值
                double r = sin(m_p * m_p_scale) * m_scale;
                // 计算摆动后的新坐标
                double roverlen = r / len;
                *x += roverlen * num;
                *y -= roverlen * den;
            }
        } else {
            // 若为第一个点，设置初始坐标
            m_last_x = *x;
            m_last_y = *y;
        }

        // 标记已经有了上一个点
        m_has_last = true;

        return code;
    }

    inline void rewind(unsigned path_id)
    {
        // 重置“游标”状态和参数 p
        m_has_last = false;
        m_p = 0.0;
        if (m_scale != 0.0) {
            // 若缩放比例不为零，重置随机数种子并重新设置分段路径
            m_rand.seed(0);
            m_segmented.rewind(path_id);
        } else {
            // 若缩放比例为零，直接重置源顶点的路径
            m_source->rewind(path_id);
        }
    }

  private:
    VertexSource *m_source; // 顶点源
    double m_scale;         // 摆动比例
    double m_length;        // 基础波长
    double m_randomness;    // 随机性因子
    agg::conv_segmentator<VertexSource> m_segmented; // 分段器
    // 上一个 x 值，用于记录上一次的坐标位置
    double m_last_x;
    // 上一个 y 值，用于记录上一次的坐标位置
    double m_last_y;
    // 记录是否存在上一个坐标位置的标志
    bool m_has_last;
    // 用于存储某个数学概率的变量
    double m_p;
    // 随机数生成器对象，用于生成随机数
    RandomNumberGenerator m_rand;
    // 用于缩放概率值的比例因子
    double m_p_scale;
    // 控制生成随机数的随机性程度的参数
    double m_log_randomness;
};

#endif // MPL_PATH_CONVERTERS_H


注释：


// 结束了路径转换器的声明
};
// 结束了 ifdef 预处理指令，检查 MPL_PATH_CONVERTERS_H 是否被定义，避免重复包含
#endif // MPL_PATH_CONVERTERS_H
```