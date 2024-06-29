# `D:\src\scipysrc\matplotlib\src\_backend_agg_basic_types.h`

```py
#ifndef MPL_BACKEND_AGG_BASIC_TYPES_H
#define MPL_BACKEND_AGG_BASIC_TYPES_H

/* Contains some simple types from the Agg backend that are also used
   by other modules */

#include <vector>

#include "agg_color_rgba.h"
#include "agg_math_stroke.h"
#include "agg_trans_affine.h"
#include "path_converters.h"

#include "py_adaptors.h"

// 定义剪切路径结构体，包含路径迭代器和仿射变换
struct ClipPath
{
    mpl::PathIterator path;  // 路径迭代器对象
    agg::trans_affine trans;  // 仿射变换对象
};

// 定义草图参数结构体，包含缩放比例、长度和随机性
struct SketchParams
{
    double scale;  // 缩放比例
    double length;  // 长度
    double randomness;  // 随机性
};

// 定义虚线类
class Dashes
{
    typedef std::vector<std::pair<double, double> > dash_t;  // 双精度浮点数对的向量类型
    double dash_offset;  // 虚线偏移量
    dash_t dashes;  // 存储虚线长度和跳过长度的向量

  public:
    double get_dash_offset() const  // 获取虚线偏移量的方法
    {
        return dash_offset;
    }
    void set_dash_offset(double x)  // 设置虚线偏移量的方法
    {
        dash_offset = x;
    }
    void add_dash_pair(double length, double skip)  // 添加虚线长度和跳过长度的方法
    {
        dashes.push_back(std::make_pair(length, skip));
    }
    size_t size() const  // 获取虚线对向量大小的方法
    {
        return dashes.size();
    }

    // 将虚线转换为Stroke对象的模板方法，考虑分辨率和是否启用抗锯齿
    template <class T>
    void dash_to_stroke(T &stroke, double dpi, bool isaa)
    {
        double scaleddpi = dpi / 72.0;  // 计算缩放后的分辨率
        for (dash_t::const_iterator i = dashes.begin(); i != dashes.end(); ++i) {
            double val0 = i->first;  // 虚线长度
            double val1 = i->second;  // 跳过长度
            val0 = val0 * scaleddpi;  // 根据分辨率缩放虚线长度
            val1 = val1 * scaleddpi;  // 根据分辨率缩放跳过长度
            if (!isaa) {
                val0 = (int)val0 + 0.5;  // 如果未启用抗锯齿，对长度进行四舍五入
                val1 = (int)val1 + 0.5;  // 如果未启用抗锯齿，对跳过长度进行四舍五入
            }
            stroke.add_dash(val0, val1);  // 添加虚线模式到Stroke对象
        }
        stroke.dash_start(get_dash_offset() * scaleddpi);  // 设置虚线起始位置
    }
};

typedef std::vector<Dashes> DashesVector;  // 定义Dashes对象的向量类型

// 定义GCAgg类
class GCAgg
{
  public:
    GCAgg()  // 构造函数，初始化默认值
        : linewidth(1.0),
          alpha(1.0),
          cap(agg::butt_cap),
          join(agg::round_join),
          snap_mode(SNAP_FALSE)
    {
    }

    ~GCAgg()  // 析构函数，无特殊操作
    {
    }

    double linewidth;  // 线宽度
    double alpha;  // 透明度
    bool forced_alpha;  // 强制透明度
    agg::rgba color;  // 颜色
    bool isaa;  // 是否抗锯齿

    agg::line_cap_e cap;  // 线帽风格
    agg::line_join_e join;  // 线连接风格

    agg::rect_d cliprect;  // 剪切矩形

    ClipPath clippath;  // 剪切路径对象

    Dashes dashes;  // 虚线对象

    e_snap_mode snap_mode;  // 捕捉模式

    mpl::PathIterator hatchpath;  // 等高线路径迭代器
    agg::rgba hatch_color;  // 等高线颜色
    double hatch_linewidth;  // 等高线线宽

    bool has_hatchpath()  // 检查是否存在等高线路径的方法
    {
        return hatchpath.total_vertices() != 0;
    }

  private:
    // 防止拷贝构造和赋值操作
    GCAgg(const GCAgg &);
    GCAgg &operator=(const GCAgg &);
};

#endif
```