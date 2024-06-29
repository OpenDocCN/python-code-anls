# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_trans_affine.h`

```py
#ifndef AGG_TRANS_AFFINE_INCLUDED
#define AGG_TRANS_AFFINE_INCLUDED

// 如果未定义 AGG_TRANS_AFFINE_INCLUDED 宏，则包含本文件内容，避免重复包含


#include <math.h>
#include "agg_basics.h"

// 包含 math.h 标准数学库和 agg_basics.h 头文件


namespace agg
{

// 进入 agg 命名空间


const double affine_epsilon = 1e-14; 

// 定义精度常量 affine_epsilon，用于比较浮点数是否相等的阈值


//============================================================trans_affine
//
// See Implementation agg_trans_affine.cpp
//
// Affine transformation are linear transformations in Cartesian coordinates
// (strictly speaking not only in Cartesian, but for the beginning we will 
// think so). They are rotation, scaling, translation and skewing.  
// After any affine transformation a line segment remains a line segment 
// and it will never become a curve. 
//
// There will be no math about matrix calculations, since it has been 
// described many times. Ask yourself a very simple question:
// "why do we need to understand and use some matrix stuff instead of just 
// rotating, scaling and so on". The answers are:
//
// 1. Any combination of transformations can be done by only 4 multiplications
//    and 4 additions in floating point.
// 2. One matrix transformation is equivalent to the number of consecutive
//    discrete transformations, i.e. the matrix "accumulates" all transformations 
//    in the order of their settings. Suppose we have 4 transformations: 
//       * rotate by 30 degrees,
//       * scale X to 2.0, 
//       * scale Y to 1.5, 
//       * move to (100, 100). 
//    The result will depend on the order of these transformations, 
//    and the advantage of matrix is that the sequence of discret calls:
//    rotate(30), scaleX(2.0), scaleY(1.5), move(100,100) 
//    will have exactly the same result as the following matrix transformations:
//   
//    affine_matrix m;
//    m *= rotate_matrix(30); 
//    m *= scaleX_matrix(2.0);
//    m *= scaleY_matrix(1.5);
//    m *= move_matrix(100,100);
//
//    m.transform_my_point_at_last(x, y);
//
// What is the good of it? In real life we will set-up the matrix only once

// 对于 trans_affine 类的详细说明，这是一个仿射变换类的声明和定义，实现位于 agg_trans_affine.cpp 文件中。仿射变换是在笛卡尔坐标系中的线性变换，包括旋转、缩放、平移和倾斜等。任何仿射变换后，线段仍然是线段，不会变成曲线。该注释详细介绍了为什么需要使用矩阵来实现这些变换，以及矩阵在连续变换中的优势。


#endif

// 结束 AGG_TRANS_AFFINE_INCLUDED 宏的定义条件
    //----------------------------------------------------------------------
    // 结构体定义：trans_affine
    //----------------------------------------------------------------------
    struct trans_affine
    {
    };
    
    //------------------------------------------------------------------------
    // 二维坐标变换函数，根据当前仿射矩阵进行坐标变换
    //------------------------------------------------------------------------
    inline void trans_affine::transform(double* x, double* y) const
    {
        double tmp = *x;
        *x = tmp * sx  + *y * shx + tx;
        *y = tmp * shy + *y * sy  + ty;
    }
    
    //------------------------------------------------------------------------
    // 二维坐标变换函数（无平移部分），只使用仿射矩阵的旋转和缩放部分进行坐标变换
    //------------------------------------------------------------------------
    inline void trans_affine::transform_2x2(double* x, double* y) const
    {
        double tmp = *x;
        *x = tmp * sx  + *y * shx;
        *y = tmp * shy + *y * sy;
    }
    
    //------------------------------------------------------------------------
    // 二维坐标反向变换函数，根据当前仿射矩阵进行坐标反向变换
    //------------------------------------------------------------------------
    inline void trans_affine::inverse_transform(double* x, double* y) const
    {
        double d = determinant_reciprocal();
        double a = (*x - tx) * d;
        double b = (*y - ty) * d;
        *x = a * sy - b * shx;
        *y = b * sx - a * shy;
    }
    
    //------------------------------------------------------------------------
    // 计算仿射矩阵的缩放系数，用于确定仿射矩阵的缩放量
    //------------------------------------------------------------------------
    inline double trans_affine::scale() const
    {
        double x = 0.707106781 * sx  + 0.707106781 * shx;
        double y = 0.707106781 * shy + 0.707106781 * sy;
        return sqrt(x*x + y*y);
    }
    
    //------------------------------------------------------------------------
    // 对仿射矩阵进行平移变换
    //------------------------------------------------------------------------
    inline const trans_affine& trans_affine::translate(double x, double y) 
    { 
        tx += x;
        ty += y; 
        return *this;
    }
    
    //------------------------------------------------------------------------
    // 对仿射矩阵进行旋转变换
    //------------------------------------------------------------------------
    inline const trans_affine& trans_affine::rotate(double a) 
    {
        double ca = cos(a); 
        double sa = sin(a);
        double t0 = sx  * ca - shy * sa;
        double t2 = shx * ca - sy * sa;
        double t4 = tx  * ca - ty * sa;
        shy = sx  * sa + shy * ca;
        sy  = shx * sa + sy * ca; 
        ty  = tx  * sa + ty * ca;
        sx  = t0;
        shx = t2;
        tx  = t4;
        return *this;
    }
    //------------------------------------------------------------------------
    // 在当前仿射变换基础上按照给定的 x 和 y 缩放系数进行缩放操作
    inline const trans_affine& trans_affine::scale(double x, double y) 
    {
        double mm0 = x; // 可能是优化器的提示
        double mm3 = y; 
        sx  *= mm0;      // 缩放系数 sx 更新为原值乘以 x
        shx *= mm0;      // 剪切系数 shx 更新为原值乘以 x
        tx  *= mm0;      // 平移系数 tx 更新为原值乘以 x
        shy *= mm3;      // 剪切系数 shy 更新为原值乘以 y
        sy  *= mm3;      // 缩放系数 sy 更新为原值乘以 y
        ty  *= mm3;      // 平移系数 ty 更新为原值乘以 y
        return *this;    // 返回当前的仿射变换对象
    }
    
    //------------------------------------------------------------------------
    // 在当前仿射变换基础上按照给定的缩放系数 s 进行统一缩放操作
    inline const trans_affine& trans_affine::scale(double s) 
    {
        double m = s;    // 可能是优化器的提示
        sx  *= m;        // 缩放系数 sx 更新为原值乘以 s
        shx *= m;        // 剪切系数 shx 更新为原值乘以 s
        tx  *= m;        // 平移系数 tx 更新为原值乘以 s
        shy *= m;        // 剪切系数 shy 更新为原值乘以 s
        sy  *= m;        // 缩放系数 sy 更新为原值乘以 s
        ty  *= m;        // 平移系数 ty 更新为原值乘以 s
        return *this;    // 返回当前的仿射变换对象
    }
    
    //------------------------------------------------------------------------
    // 将当前仿射变换对象与给定的仿射变换对象进行前置乘法组合
    inline const trans_affine& trans_affine::premultiply(const trans_affine& m)
    {
        trans_affine t = m;       // 创建给定仿射变换对象的副本
        return *this = t.multiply(*this); // 返回当前对象与副本对象相乘的结果赋值给当前对象
    }
    
    //------------------------------------------------------------------------
    // 将当前仿射变换对象与给定的仿射变换对象的逆矩阵进行乘法组合
    inline const trans_affine& trans_affine::multiply_inv(const trans_affine& m)
    {
        trans_affine t = m;       // 创建给定仿射变换对象的副本
        t.invert();               // 对副本对象进行求逆操作
        return multiply(t);       // 返回当前对象与逆对象相乘的结果
    }
    
    //------------------------------------------------------------------------
    // 将当前仿射变换对象的逆矩阵与给定的仿射变换对象进行乘法组合
    inline const trans_affine& trans_affine::premultiply_inv(const trans_affine& m)
    {
        trans_affine t = m;       // 创建给定仿射变换对象的副本
        t.invert();               // 对副本对象进行求逆操作
        return *this = t.multiply(*this); // 返回当前对象与逆对象相乘的结果赋值给当前对象
    }
    
    //------------------------------------------------------------------------
    // 计算当前仿射变换对象的绝对缩放系数，用于图像重采样中的缩放系数估计
    void trans_affine::scaling_abs(double* x, double* y) const
    {
        // 用于计算图像重采样中的缩放系数。
        // 当存在相当大的剪切时，该方法比单纯使用 sx, sy 更能给出较好的估计。
        *x = sqrt(sx  * sx  + shx * shx); // 计算 x 方向的绝对缩放系数
        *y = sqrt(shy * shy + sy  * sy); // 计算 y 方向的绝对缩放系数
    }
    
    //====================================================trans_affine_rotation
    // 旋转矩阵。sin() 和 cos() 函数对于同一角度计算了两次。
    // 这没有坏处，因为现代处理器上 sin()/cos() 的性能非常好。
    // 而且这个操作不会经常调用。
    class trans_affine_rotation : public trans_affine
    {
    public:
        trans_affine_rotation(double a) : 
          trans_affine(cos(a), sin(a), -sin(a), cos(a), 0.0, 0.0)
        {}
    };
    
    //====================================================trans_affine_scaling
    // 缩放矩阵。x, y - 分别表示 X 和 Y 方向的缩放系数
    class trans_affine_scaling : public trans_affine
    {
    public:
        trans_affine_scaling(double x, double y) : 
          trans_affine(x, 0.0, 0.0, y, 0.0, 0.0)
        {}
    
        trans_affine_scaling(double s) : 
          trans_affine(s, 0.0, 0.0, s, 0.0, 0.0)
        {}
    };
    
    //================================================trans_affine_translation
    // 翻译矩阵
    class trans_affine_translation : public trans_affine
    {
    public:
        // 构造函数：传入 x 和 y 偏移量，初始化平移变换矩阵
        trans_affine_translation(double x, double y) : 
          trans_affine(1.0, 0.0, 0.0, 1.0, x, y)
        {}
    };
    
    //====================================================trans_affine_skewing
    // 倾斜矩阵
    class trans_affine_skewing : public trans_affine
    {
    public:
        // 构造函数：传入 x 和 y 的倾斜角度，初始化倾斜变换矩阵
        trans_affine_skewing(double x, double y) : 
          trans_affine(1.0, tan(y), tan(x), 1.0, 0.0, 0.0)
        {}
    };
    
    
    //================================================trans_affine_line_segment
    // 线段变换矩阵
    // 旋转、缩放和平移，关联 0...dist 范围内的线段 x1,y1,x2,y2
    class trans_affine_line_segment : public trans_affine
    {
    public:
        // 构造函数：传入线段两端点坐标 x1, y1, x2, y2 和距离 dist，计算变换矩阵
        trans_affine_line_segment(double x1, double y1, double x2, double y2, 
                                  double dist)
        {
            // 计算线段的方向向量
            double dx = x2 - x1;
            double dy = y2 - y1;
            // 如果距离 dist 大于 0，则进行缩放变换
            if(dist > 0.0)
            {
                multiply(trans_affine_scaling(sqrt(dx * dx + dy * dy) / dist));
            }
            // 添加旋转变换
            multiply(trans_affine_rotation(atan2(dy, dx)));
            // 添加平移变换
            multiply(trans_affine_translation(x1, y1));
        }
    };
    
    
    //================================================trans_affine_reflection_unit
    // 单位反射矩阵
    // 将坐标沿过原点的单位向量 (ux, uy) 的线段反射
    // 由 John Horigan 贡献
    class trans_affine_reflection_unit : public trans_affine
    {
    public:
        // 构造函数：传入单位向量 ux, uy，初始化单位反射变换矩阵
        trans_affine_reflection_unit(double ux, double uy) :
          trans_affine(2.0 * ux * ux - 1.0, 
                       2.0 * ux * uy, 
                       2.0 * ux * uy, 
                       2.0 * uy * uy - 1.0, 
                       0.0, 0.0)
        {}
    };
    
    
    //=================================================trans_affine_reflection
    // 反射矩阵
    // 将坐标沿过原点的角度 a 或包含非单位向量 (x, y) 的线段反射
    // 由 John Horigan 贡献
    class trans_affine_reflection : public trans_affine_reflection_unit
    {
    public:
        // 构造函数：传入角度 a，初始化反射变换矩阵
        trans_affine_reflection(double a) :
          trans_affine_reflection_unit(cos(a), sin(a))
        {}
    
        // 构造函数：传入非单位向量 x, y，初始化反射变换矩阵
        trans_affine_reflection(double x, double y) :
          trans_affine_reflection_unit(x / sqrt(x * x + y * y), y / sqrt(x * x + y * y))
        {}
    };
}


注释：


// 结束一个 C++ 的条件编译指令块，对应于 #ifdef 或 #ifndef



#endif


注释：


// 结束一个 C++ 的条件编译指令块，对应于 #ifdef 或 #ifndef 的另一种条件
```