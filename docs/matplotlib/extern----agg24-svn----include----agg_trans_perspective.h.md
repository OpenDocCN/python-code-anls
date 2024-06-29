# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_trans_perspective.h`

```
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
// Perspective 2D transformations
//
//----------------------------------------------------------------------------
#ifndef AGG_TRANS_PERSPECTIVE_INCLUDED
#define AGG_TRANS_PERSPECTIVE_INCLUDED

#include "agg_trans_affine.h"

namespace agg
{
    //=======================================================trans_perspective
    // 透视变换类，继承自仿射变换类
    struct trans_perspective
    {
        //------------------------------------------------------------------------
        // 将四边形坐标转换为四边形的透视变换
        inline bool square_to_quad(const double* q)
        {
            // 计算两个对角线向量的差值
            double dx = q[0] - q[2] + q[4] - q[6];
            double dy = q[1] - q[3] + q[5] - q[7];
            
            // 检查是否为仿射情况（平行四边形）
            if(dx == 0.0 && dy == 0.0)
            {   
                // 仿射变换情况（平行四边形）
                //---------------
                sx  = q[2] - q[0];  // x 方向缩放参数
                shy = q[3] - q[1];  // y 方向的斜切参数
                w0  = 0.0;           // 透视参数
                shx = q[4] - q[2];  // x 方向的斜切参数
                sy  = q[5] - q[3];  // y 方向缩放参数
                w1  = 0.0;           // 透视参数
                tx  = q[0];          // x 方向的平移参数
                ty  = q[1];          // y 方向的平移参数
                w2  = 1.0;           // 透视参数
            }
            else
            {
                // 一般情况
                double dx1 = q[2] - q[4];   // 对角线向量的差值
                double dy1 = q[3] - q[5];   // 对角线向量的差值
                double dx2 = q[6] - q[4];   // 对角线向量的差值
                double dy2 = q[7] - q[5];   // 对角线向量的差值
                double den = dx1 * dy2 - dx2 * dy1;  // 计算行列式的值

                // 检查行列式是否为零（奇异情况）
                if(den == 0.0)
                {
                    // 奇异情况
                    //---------------
                    sx = shy = w0 = shx = sy = w1 = tx = ty = w2 = 0.0;  // 所有参数清零
                    return false;  // 返回失败
                }
                
                // 一般情况
                //---------------
                double u = (dx * dy2 - dy * dx2) / den;  // 计算 u 参数
                double v = (dy * dx1 - dx * dy1) / den;  // 计算 v 参数
                sx  = q[2] - q[0] + u * q[2];  // x 方向的缩放和透视参数
                shy = q[3] - q[1] + u * q[3];  // y 方向的斜切和透视参数
                w0  = u;                       // 第一个透视参数
                shx = q[6] - q[0] + v * q[6];  // x 方向的斜切和透视参数
                sy  = q[7] - q[1] + v * q[7];  // y 方向的缩放和透视参数
                w1  = v;                       // 第二个透视参数
                tx  = q[0];                    // x 方向的平移参数
                ty  = q[1];                    // y 方向的平移参数
                w2  = 1.0;                     // 第三个透视参数
            }
            return true;  // 返回成功
        }
        
        //------------------------------------------------------------------------
        // 反转透视变换
        inline bool invert()
    {
        // 计算变换矩阵的第一行分量
        double d0 = sy  * w2 - w1  * ty;
        // 计算变换矩阵的第二行分量
        double d1 = w0  * ty - shy * w2;
        // 计算变换矩阵的第三行分量
        double d2 = shy * w1 - w0  * sy;
        // 计算变换矩阵的分母，用于后续的逆运算
        double d  = sx  * d0 + shx * d1 + tx * d2;
        // 如果分母为零，则重置所有变换参数并返回 false
        if(d == 0.0) 
        {
            sx = shy = w0 = shx = sy = w1 = tx = ty = w2 = 0.0;
            return false;
        }
        // 计算分母的倒数，用于计算逆变换
        d = 1.0 / d;
        // 保存当前变换矩阵副本
        trans_perspective a = *this;
        // 计算变换矩阵的各个元素，实现逆变换
        sx  = d * d0;
        shy = d * d1;
        w0  = d * d2;
        shx = d * (a.w1 * a.tx  - a.shx * a.w2);
        sy  = d * (a.sx * a.w2  - a.w0 * a.tx);
        w1  = d * (a.w0 * a.shx - a.sx * a.w1);
        tx  = d * (a.shx * a.ty  - a.sy * a.tx);
        ty  = d * (a.shy * a.tx  - a.sx * a.ty);
        w2  = d * (a.sx * a.sy  - a.shy * a.shx);
        // 返回 true 表示逆变换成功
        return true;
    }
    
    //------------------------------------------------------------------------
    inline bool trans_perspective::quad_to_square(const double* q)
    {
        // 如果无法将四边形 q 转换为单位正方形，则返回 false
        if(!square_to_quad(q)) return false;
        // 对转换后的正方形进行逆变换，使其恢复为原来的四边形
        invert();
        // 返回 true 表示逆变换成功
        return true;
    }
    
    //------------------------------------------------------------------------
    inline bool trans_perspective::quad_to_quad(const double* qs, 
                                                const double* qd)
    {
        // 创建一个新的透视变换对象 p
        trans_perspective p;
        // 将源四边形 qs 转换为单位正方形，若失败则返回 false
        if(!quad_to_square(qs)) return false;
        // 将单位正方形转换为目标四边形 qd
        if(!p.square_to_quad(qd)) return false;
        // 将两个透视变换对象相乘，得到从源四边形到目标四边形的变换
        multiply(p);
        // 返回 true 表示变换成功
        return true;
    }
    
    //------------------------------------------------------------------------
    inline bool trans_perspective::rect_to_quad(double x1, double y1, 
                                                double x2, double y2,
                                                const double* q)
    {
        // 创建一个存储矩形坐标的数组 r
        double r[8];
        // 根据矩形的左上角 (x1, y1) 和右下角 (x2, y2) 设置数组 r 的坐标
        r[0] = r[6] = x1;
        r[2] = r[4] = x2;
        r[1] = r[3] = y1;
        r[5] = r[7] = y2;
        // 将矩形 r 转换为目标四边形 q
        return quad_to_quad(r, q);
    }
    
    //------------------------------------------------------------------------
    inline bool trans_perspective::quad_to_rect(const double* q,
                                                double x1, double y1, 
                                                double x2, double y2)
    {
        // 创建一个存储矩形坐标的数组 r
        double r[8];
        // 根据矩形的左上角 (x1, y1) 和右下角 (x2, y2) 设置数组 r 的坐标
        r[0] = r[6] = x1;
        r[2] = r[4] = x2;
        r[1] = r[3] = y1;
        r[5] = r[7] = y2;
        // 将源四边形 q 转换为目标矩形 r
        return quad_to_quad(q, r);
    }
    
    //------------------------------------------------------------------------
    inline trans_perspective::trans_perspective(double x1, double y1, 
                                                double x2, double y2, 
                                                const double* quad)
    {
        // 将矩形 (x1, y1)-(x2, y2) 映射到目标四边形 quad
        rect_to_quad(x1, y1, x2, y2, quad);
    }
    
    //------------------------------------------------------------------------
    inline trans_perspective::trans_perspective(const double* quad, 
                                                double x1, double y1, 
                                                double x2, double y2)
    {
        // 将源四边形 quad 映射到矩形 (x1, y1)-(x2, y2)
        quad_to_rect(quad, x1, y1, x2, y2);
    }
    //------------------------------------------------------------------------
    // 构造函数：使用源点集和目标点集初始化透视变换
    inline trans_perspective::trans_perspective(const double* src, 
                                                const double* dst) 
    {
        quad_to_quad(src, dst); // 调用 quad_to_quad 函数，根据源点集和目标点集计算透视变换
    }
    
    //------------------------------------------------------------------------
    // 方法：重置透视变换为单位矩阵
    inline const trans_perspective& trans_perspective::reset()
    {
        sx  = 1; shy = 0; w0 = 0;   // 设置透视变换的初始参数
        shx = 0; sy  = 1; w1 = 0;
        tx  = 0; ty  = 0; w2 = 1;
        return *this;  // 返回当前对象的引用
    }
    
    //------------------------------------------------------------------------
    // 方法：将当前透视变换与给定透视变换相乘
    inline const trans_perspective& 
    trans_perspective::multiply(const trans_perspective& a)
    {
        trans_perspective b = *this;  // 复制当前对象
        // 计算矩阵相乘结果并更新当前对象的参数
        sx  = a.sx *b.sx  + a.shx*b.shy + a.tx*b.w0;
        shx = a.sx *b.shx + a.shx*b.sy  + a.tx*b.w1;
        tx  = a.sx *b.tx  + a.shx*b.ty  + a.tx*b.w2;
        shy = a.shy*b.sx  + a.sy *b.shy + a.ty*b.w0;
        sy  = a.shy*b.shx + a.sy *b.sy  + a.ty*b.w1;
        ty  = a.shy*b.tx  + a.sy *b.ty  + a.ty*b.w2;
        w0  = a.w0 *b.sx  + a.w1 *b.shy + a.w2*b.w0;
        w1  = a.w0 *b.shx + a.w1 *b.sy  + a.w2*b.w1;
        w2  = a.w0 *b.tx  + a.w1 *b.ty  + a.w2*b.w2;
        return *this;  // 返回当前对象的引用
    }
    
    //------------------------------------------------------------------------
    // 方法：将当前透视变换与给定仿射变换相乘
    inline const trans_perspective& 
    trans_perspective::multiply(const trans_affine& a)
    {
        trans_perspective b = *this;  // 复制当前对象
        // 计算透视变换与仿射变换相乘后的结果并更新当前对象的参数
        sx  = a.sx *b.sx  + a.shx*b.shy + a.tx*b.w0;
        shx = a.sx *b.shx + a.shx*b.sy  + a.tx*b.w1;
        tx  = a.sx *b.tx  + a.shx*b.ty  + a.tx*b.w2;
        shy = a.shy*b.sx  + a.sy *b.shy + a.ty*b.w0;
        sy  = a.shy*b.shx + a.sy *b.sy  + a.ty*b.w1;
        ty  = a.shy*b.tx  + a.sy *b.ty  + a.ty*b.w2;
        return *this;  // 返回当前对象的引用
    }
    
    //------------------------------------------------------------------------
    // 方法：使用给定透视变换在当前透视变换之前进行预乘
    inline const trans_perspective& 
    trans_perspective::premultiply(const trans_perspective& b)
    {
        trans_perspective a = *this;  // 复制当前对象
        // 计算给定透视变换与当前透视变换相乘后的结果并更新当前对象的参数
        sx  = a.sx *b.sx  + a.shx*b.shy + a.tx*b.w0;
        shx = a.sx *b.shx + a.shx*b.sy  + a.tx*b.w1;
        tx  = a.sx *b.tx  + a.shx*b.ty  + a.tx*b.w2;
        shy = a.shy*b.sx  + a.sy *b.shy + a.ty*b.w0;
        sy  = a.shy*b.shx + a.sy *b.sy  + a.ty*b.w1;
        ty  = a.shy*b.tx  + a.sy *b.ty  + a.ty*b.w2;
        w0  = a.w0 *b.sx  + a.w1 *b.shy + a.w2*b.w0;
        w1  = a.w0 *b.shx + a.w1 *b.sy  + a.w2*b.w1;
        w2  = a.w0 *b.tx  + a.w1 *b.ty  + a.w2*b.w2;
        return *this;  // 返回当前对象的引用
    }
    
    //------------------------------------------------------------------------
    // 方法：使用给定仿射变换在当前透视变换之前进行预乘
    inline const trans_perspective& 
    trans_perspective::premultiply(const trans_affine& b)
    {
        // 创建一个新的 trans_perspective 对象 a，并复制当前对象 (*this) 的值
        trans_perspective a = *this;
        // 计算新对象的 sx 分量
        sx  = a.sx *b.sx  + a.shx*b.shy;
        // 计算新对象的 shx 分量
        shx = a.sx *b.shx + a.shx*b.sy;
        // 计算新对象的 tx 分量
        tx  = a.sx *b.tx  + a.shx*b.ty  + a.tx;
        // 计算新对象的 shy 分量
        shy = a.shy*b.sx  + a.sy *b.shy;
        // 计算新对象的 sy 分量
        sy  = a.shy*b.shx + a.sy *b.sy;
        // 计算新对象的 ty 分量
        ty  = a.shy*b.tx  + a.sy *b.ty  + a.ty;
        // 计算新对象的 w0 分量
        w0  = a.w0 *b.sx  + a.w1 *b.shy;
        // 计算新对象的 w1 分量
        w1  = a.w0 *b.shx + a.w1 *b.sy;
        // 计算新对象的 w2 分量
        w2  = a.w0 *b.tx  + a.w1 *b.ty  + a.w2;
        // 返回当前对象 (*this)
        return *this;
    }
    
    //------------------------------------------------------------------------
    // 对 trans_perspective 对象进行反转并乘以当前对象
    const trans_perspective& 
    trans_perspective::multiply_inv(const trans_perspective& m)
    {
        // 创建一个新的 trans_perspective 对象 t，并使用参数对象 m 的值初始化它
        trans_perspective t = m;
        // 对 t 进行反转操作
        t.invert();
        // 调用 multiply 方法将反转后的对象 t 乘以当前对象，并返回结果
        return multiply(t);
    }
    
    //------------------------------------------------------------------------
    // 对 trans_affine 对象进行反转并乘以当前对象
    const trans_perspective&
    trans_perspective::multiply_inv(const trans_affine& m)
    {
        // 创建一个新的 trans_affine 对象 t，并使用参数对象 m 的值初始化它
        trans_affine t = m;
        // 对 t 进行反转操作
        t.invert();
        // 调用 multiply 方法将反转后的对象 t 乘以当前对象，并返回结果
        return multiply(t);
    }
    
    //------------------------------------------------------------------------
    // 对 trans_perspective 对象进行前置反转并乘以当前对象
    const trans_perspective&
    trans_perspective::premultiply_inv(const trans_perspective& m)
    {
        // 创建一个新的 trans_perspective 对象 t，并使用参数对象 m 的值初始化它
        trans_perspective t = m;
        // 对 t 进行反转操作
        t.invert();
        // 将当前对象 (*this) 设置为 t 乘以当前对象的结果，并返回当前对象的引用
        return *this = t.multiply(*this);
    }
    
    //------------------------------------------------------------------------
    // 对 trans_affine 对象进行前置反转并乘以当前对象
    const trans_perspective&
    trans_perspective::premultiply_inv(const trans_affine& m)
    {
        // 创建一个新的 trans_perspective 对象 t，并使用参数对象 m 初始化它
        trans_perspective t(m);
        // 对 t 进行反转操作
        t.invert();
        // 将当前对象 (*this) 设置为 t 乘以当前对象的结果，并返回当前对象的引用
        return *this = t.multiply(*this);
    }
    
    //------------------------------------------------------------------------
    // 将当前对象沿 x 和 y 轴平移
    inline const trans_perspective& 
    trans_perspective::translate(double x, double y)
    {
        // 更新 tx 和 ty 分量以实现平移
        tx += x;
        ty += y;
        // 返回当前对象的引用
        return *this;
    }
    
    //------------------------------------------------------------------------
    // 将当前对象绕原点旋转指定角度 a
    inline const trans_perspective& trans_perspective::rotate(double a)
    {
        // 调用 multiply 方法，将当前对象乘以旋转角度 a 的 trans_affine_rotation 对象
        multiply(trans_affine_rotation(a));
        // 返回当前对象的引用
        return *this;
    }
    
    //------------------------------------------------------------------------
    // 将当前对象沿 x 和 y 方向按比例缩放
    inline const trans_perspective& trans_perspective::scale(double s)
    {
        // 调用 multiply 方法，将当前对象乘以按比例缩放的 trans_affine_scaling 对象
        multiply(trans_affine_scaling(s));
        // 返回当前对象的引用
        return *this;
    }
    
    //------------------------------------------------------------------------
    // 将当前对象沿 x 和 y 方向按不同比例缩放
    inline const trans_perspective& trans_perspective::scale(double x, double y)
    {
        // 调用 multiply 方法，将当前对象乘以按指定比例缩放的 trans_affine_scaling 对象
        multiply(trans_affine_scaling(x, y));
        // 返回当前对象的引用
        return *this;
    }
    
    //------------------------------------------------------------------------
    // 对输入坐标进行透视变换
    inline void trans_perspective::transform(double* px, double* py) const
    {
        // 从指针中读取坐标值，赋给局部变量 x 和 y
        double x = *px;
        double y = *py;
        // 计算变换系数 m，用于将透视变换应用到 x 和 y 坐标上
        double m = 1.0 / (x*w0 + y*w1 + w2);
        // 更新 px 和 py 指向的坐标值，实现透视变换
        *px = m * (x*sx  + y*shx + tx);
        *py = m * (x*shy + y*sy  + ty);
    }
    
    //------------------------------------------------------------------------
    // 对输入坐标进行仿射变换（仅平移和旋转）
    inline void trans_perspective::transform_affine(double* x, double* y) const
    {
        // 保存变量 *x 的临时副本
        double tmp = *x;
        // 应用透视变换的 2x2 矩阵变换到 (x, y)
        *x = tmp * sx  + *y * shx + tx;
        *y = tmp * shy + *y * sy  + ty;
    }

    //------------------------------------------------------------------------
    inline void trans_perspective::transform_2x2(double* x, double* y) const
    {
        // 保存变量 *x 的临时副本
        double tmp = *x;
        // 应用透视变换的 2x2 矩阵变换到 (x, y)
        *x = tmp * sx  + *y * shx;
        *y = tmp * shy + *y * sy;
    }

    //------------------------------------------------------------------------
    inline void trans_perspective::inverse_transform(double* x, double* y) const
    {
        // 创建当前透视变换的副本 t
        trans_perspective t(*this);
        // 如果 t 可逆，则对 (x, y) 应用逆变换
        if(t.invert()) t.transform(x, y);
    }

    //------------------------------------------------------------------------
    inline void trans_perspective::store_to(double* m) const
    {
        // 将透视变换的参数存储到数组 m 中
        *m++ = sx;  *m++ = shy; *m++ = w0; 
        *m++ = shx; *m++ = sy;  *m++ = w1;
        *m++ = tx;  *m++ = ty;  *m++ = w2;
    }

    //------------------------------------------------------------------------
    inline const trans_perspective& trans_perspective::load_from(const double* m)
    {
        // 从数组 m 中加载透视变换的参数到当前对象中
        sx  = *m++; shy = *m++; w0 = *m++; 
        shx = *m++; sy  = *m++; w1 = *m++;
        tx  = *m++; ty  = *m++; w2 = *m++;
        return *this;
    }

    //------------------------------------------------------------------------
    inline const trans_perspective& 
    trans_perspective::from_affine(const trans_affine& a)
    {
        // 从仿射变换 a 转换到透视变换的参数
        sx  = a.sx;  shy = a.shy; w0 = 0; 
        shx = a.shx; sy  = a.sy;  w1 = 0;
        tx  = a.tx;  ty  = a.ty;  w2 = 1;
        return *this;
    }

    //------------------------------------------------------------------------
    inline double trans_perspective::determinant() const
    {
        // 计算透视变换矩阵的行列式
        return sx  * (sy  * w2 - ty  * w1) +
               shx * (ty  * w0 - shy * w2) +
               tx  * (shy * w1 - sy  * w0);
    }
  
    //------------------------------------------------------------------------
    inline double trans_perspective::determinant_reciprocal() const
    {
        // 计算透视变换矩阵行列式的倒数
        return 1.0 / determinant();
    }

    //------------------------------------------------------------------------
    inline bool trans_perspective::is_valid(double epsilon) const
    {
        // 检查透视变换是否有效，即各参数的绝对值是否大于 epsilon
        return fabs(sx) > epsilon && fabs(sy) > epsilon && fabs(w2) > epsilon;
    }

    //------------------------------------------------------------------------
    inline bool trans_perspective::is_identity(double epsilon) const
    {
        // 检查透视变换是否为单位矩阵，各参数与单位矩阵元素的比较，精度为 epsilon
        return is_equal_eps(sx,  1.0, epsilon) &&
               is_equal_eps(shy, 0.0, epsilon) &&
               is_equal_eps(w0,  0.0, epsilon) &&
               is_equal_eps(shx, 0.0, epsilon) && 
               is_equal_eps(sy,  1.0, epsilon) &&
               is_equal_eps(w1,  0.0, epsilon) &&
               is_equal_eps(tx,  0.0, epsilon) &&
               is_equal_eps(ty,  0.0, epsilon) &&
               is_equal_eps(w2,  1.0, epsilon);
    }

    //------------------------------------------------------------------------
    // 比较当前透视变换对象与给定透视变换对象的各个成员变量是否相等，使用给定的精度阈值
    inline bool trans_perspective::is_equal(const trans_perspective& m, 
                                            double epsilon) const
    {
        // 逐个比较各个成员变量是否在给定精度范围内相等
        return is_equal_eps(sx,  m.sx,  epsilon) &&
               is_equal_eps(shy, m.shy, epsilon) &&
               is_equal_eps(w0,  m.w0,  epsilon) &&
               is_equal_eps(shx, m.shx, epsilon) && 
               is_equal_eps(sy,  m.sy,  epsilon) &&
               is_equal_eps(w1,  m.w1,  epsilon) &&
               is_equal_eps(tx,  m.tx,  epsilon) &&
               is_equal_eps(ty,  m.ty,  epsilon) &&
               is_equal_eps(w2,  m.w2,  epsilon);
    }

    //------------------------------------------------------------------------
    // 计算当前透视变换对象的比例因子
    inline double trans_perspective::scale() const
    {
        // 使用预定义的公式计算比例因子
        double x = 0.707106781 * sx  + 0.707106781 * shx;
        double y = 0.707106781 * shy + 0.707106781 * sy;
        // 返回计算得到的比例因子的平方根
        return sqrt(x*x + y*y);
    }

    //------------------------------------------------------------------------
    // 计算当前透视变换对象的旋转角度
    inline double trans_perspective::rotation() const
    {
        // 定义两个点并进行当前透视变换后的坐标变换
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 1.0;
        double y2 = 0.0;
        transform(&x1, &y1);  // 对第一个点进行透视变换
        transform(&x2, &y2);  // 对第二个点进行透视变换
        // 返回两点的变换后坐标计算得到的旋转角度
        return atan2(y2-y1, x2-x1);
    }

    //------------------------------------------------------------------------
    // 获取当前透视变换对象的平移量
    void trans_perspective::translation(double* dx, double* dy) const
    {
        // 直接返回当前对象的平移量
        *dx = tx;
        *dy = ty;
    }

    //------------------------------------------------------------------------
    // 计算当前透视变换对象的缩放量
    void trans_perspective::scaling(double* x, double* y) const
    {
        // 定义两个点并进行旋转调整后的坐标变换
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 1.0;
        double y2 = 1.0;
        trans_perspective t(*this);  // 创建当前对象的拷贝
        t *= trans_affine_rotation(-rotation());  // 对拷贝对象进行逆时针旋转变换
        t.transform(&x1, &y1);  // 对第一个点进行变换
        t.transform(&x2, &y2);  // 对第二个点进行变换
        // 计算变换后两点坐标差，作为缩放量
        *x = x2 - x1;
        *y = y2 - y1;
    }

    //------------------------------------------------------------------------
    // 获取当前透视变换对象的绝对缩放量
    void trans_perspective::scaling_abs(double* x, double* y) const
    {
        // 直接返回当前对象的绝对缩放量
        *x = sqrt(sx  * sx  + shx * shx);
        *y = sqrt(shy * shy + sy  * sy);
    }
}
// 结束条件预处理指令，表示结束某个条件编译段落

#endif
// 结束条件编译指令，指示编译器结束处理与给定条件相关联的代码段
```