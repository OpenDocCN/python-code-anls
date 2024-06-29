# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_trans_affine.cpp`

```py
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// Affine transformations
//
//----------------------------------------------------------------------------

#include "agg_trans_affine.h"

// 命名空间 agg 开始
namespace agg
{

    //------------------------------------------------------------------------
    // 通过源平行四边形(src)和目标平行四边形(dst)的顶点坐标，构造仿射变换
    const trans_affine& trans_affine::parl_to_parl(const double* src, 
                                                   const double* dst)
    {
        // 计算水平和垂直缩放因子
        sx  = src[2] - src[0];   // 水平缩放因子计算
        shy = src[3] - src[1];   // 垂直错切因子计算
        shx = src[4] - src[0];   // 水平错切因子计算
        sy  = src[5] - src[1];   // 垂直缩放因子计算
        tx  = src[0];            // 水平平移量计算
        ty  = src[1];            // 垂直平移量计算
        // 执行反转变换
        invert();
        // 与目标平行四边形(dst)的仿射变换相乘
        multiply(trans_affine(dst[2] - dst[0], dst[3] - dst[1], 
                              dst[4] - dst[0], dst[5] - dst[1],
                              dst[0], dst[1]));
        return *this;
    }

    //------------------------------------------------------------------------
    // 将矩形(x1, y1)-(x2, y2)变换为平行四边形(parl)
    const trans_affine& trans_affine::rect_to_parl(double x1, double y1, 
                                                   double x2, double y2, 
                                                   const double* parl)
    {
        double src[6];
        // 设置源平行四边形的顶点坐标
        src[0] = x1; src[1] = y1;        // 左下角
        src[2] = x2; src[3] = y1;        // 右下角
        src[4] = x2; src[5] = y2;        // 右上角
        // 调用 parl_to_parl 方法进行变换
        parl_to_parl(src, parl);
        return *this;
    }

    //------------------------------------------------------------------------
    // 将平行四边形(parl)变换为矩形(x1, y1)-(x2, y2)
    const trans_affine& trans_affine::parl_to_rect(const double* parl, 
                                                   double x1, double y1, 
                                                   double x2, double y2)
    {
        double dst[6];
        // 设置目标矩形的顶点坐标
        dst[0] = x1; dst[1] = y1;        // 左下角
        dst[2] = x2; dst[3] = y1;        // 右下角
        dst[4] = x2; dst[5] = y2;        // 右上角
        // 调用 parl_to_parl 方法进行变换
        parl_to_parl(parl, dst);
        return *this;
    }

    //------------------------------------------------------------------------
    // 将当前仿射变换与给定的变换矩阵(m)相乘
    const trans_affine& trans_affine::multiply(const trans_affine& m)
    {
        // 计算新的仿射变换参数
        double t0 = sx  * m.sx + shy * m.shx;
        double t2 = shx * m.sx + sy  * m.shx;
        double t4 = tx  * m.sx + ty  * m.shx + m.tx;
        shy = sx  * m.shy + shy * m.sy;
        sy  = shx * m.shy + sy  * m.sy;
        ty  = tx  * m.shy + ty  * m.sy + m.ty;
        sx  = t0;
        shx = t2;
        tx  = t4;
        return *this;
    }

} // 命名空间 agg 结束
    //------------------------------------------------------------------------
    const trans_affine& trans_affine::invert()
    {
        // 计算矩阵的行列式的倒数，用于矩阵求逆
        double d  = determinant_reciprocal();
    
        // 临时变量保存变换后的值
        double t0  =  sy  * d;
               sy  =  sx  * d;
               shy = -shy * d;
               shx = -shx * d;
    
        // 计算平移后的值
        double t4 = -tx * t0  - ty * shx;
               ty = -tx * shy - ty * sy;
    
        // 更新矩阵的缩放和平移
        sx = t0;
        tx = t4;
        return *this;
    }
    //------------------------------------------------------------------------
    const trans_affine& trans_affine::flip_x()
    {
        // 沿着 X 轴翻转矩阵
        sx  = -sx;
        shy = -shy;
        tx  = -tx;
        return *this;
    }
    //------------------------------------------------------------------------
    const trans_affine& trans_affine::flip_y()
    {
        // 沿着 Y 轴翻转矩阵
        shx = -shx;
        sy  = -sy;
        ty  = -ty;
        return *this;
    }
    //------------------------------------------------------------------------
    const trans_affine& trans_affine::reset()
    {
        // 重置矩阵为单位矩阵
        sx  = sy  = 1.0; 
        shy = shx = tx = ty = 0.0;
        return *this;
    }
    //------------------------------------------------------------------------
    bool trans_affine::is_identity(double epsilon) const
    {
        // 判断矩阵是否为单位矩阵
        return is_equal_eps(sx,  1.0, epsilon) &&
               is_equal_eps(shy, 0.0, epsilon) &&
               is_equal_eps(shx, 0.0, epsilon) && 
               is_equal_eps(sy,  1.0, epsilon) &&
               is_equal_eps(tx,  0.0, epsilon) &&
               is_equal_eps(ty,  0.0, epsilon);
    }
    //------------------------------------------------------------------------
    bool trans_affine::is_valid(double epsilon) const
    {
        // 判断矩阵是否有效（非零缩放因子）
        return fabs(sx) > epsilon && fabs(sy) > epsilon;
    }
    //------------------------------------------------------------------------
    bool trans_affine::is_equal(const trans_affine& m, double epsilon) const
    {
        // 判断矩阵是否与另一个矩阵相等
        return is_equal_eps(sx,  m.sx,  epsilon) &&
               is_equal_eps(shy, m.shy, epsilon) &&
               is_equal_eps(shx, m.shx, epsilon) && 
               is_equal_eps(sy,  m.sy,  epsilon) &&
               is_equal_eps(tx,  m.tx,  epsilon) &&
               is_equal_eps(ty,  m.ty,  epsilon);
    }
    //------------------------------------------------------------------------
    double trans_affine::rotation() const
    {
        // 计算矩阵的旋转角度
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 1.0;
        double y2 = 0.0;
        transform(&x1, &y1);
        transform(&x2, &y2);
        return atan2(y2-y1, x2-x1);
    }
    //------------------------------------------------------------------------
    void trans_affine::translation(double* dx, double* dy) const
    {
        // 获取矩阵的平移分量
        *dx = tx;
        *dy = ty;
    }
    //------------------------------------------------------------------------
    void trans_affine::scaling(double* x, double* y) const
    {
        // 获取矩阵的缩放因子
        *x = sx;
        *y = sy;
    }
    {
        // 定义起点和终点的初始坐标
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 1.0;
        double y2 = 1.0;
    
        // 创建当前对象的仿射变换副本
        trans_affine t(*this);
    
        // 将仿射变换副本按照当前对象的逆时针旋转角度进行旋转
        t *= trans_affine_rotation(-rotation());
    
        // 对起点 (x1, y1) 和终点 (x2, y2) 应用仿射变换
        t.transform(&x1, &y1);
        t.transform(&x2, &y2);
    
        // 计算变换后的终点与起点的坐标差，存入指定的变量中
        *x = x2 - x1;
        *y = y2 - y1;
    }
}



# 这行代码关闭了一个代码块，通常是函数或类定义的末尾处，确保代码逻辑的完整性
```