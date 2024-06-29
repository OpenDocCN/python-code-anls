# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_basics.h`

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

#ifndef AGG_BASICS_INCLUDED
#define AGG_BASICS_INCLUDED

#include <math.h>
#include "agg_config.h"

//---------------------------------------------------------AGG_CUSTOM_ALLOCATOR
#ifdef AGG_CUSTOM_ALLOCATOR
#include "agg_allocator.h"
#else
namespace agg
{
    // The policy of all AGG containers and memory allocation strategy 
    // in general is that no allocated data requires explicit construction.
    // It means that the allocator can be really simple; you can even
    // replace new/delete to malloc/free. The constructors and destructors 
    // won't be called in this case, however everything will remain working. 
    // The second argument of deallocate() is the size of the allocated 
    // block. You can use this information if you wish.
    //------------------------------------------------------------pod_allocator
    // 简单的 POD（Plain Old Data）分配器，用于分配数组
    template<class T> struct pod_allocator
    {
        // 分配 num 个 T 类型的对象并返回指针
        static T*   allocate(unsigned num)       { return new T [num]; }
        // 释放指针 ptr 指向的内存块，num 为分配的对象数目（在此处未使用）
        static void deallocate(T* ptr, unsigned) { delete [] ptr;      }
    };

    // Single object allocator. It's also can be replaced with your custom
    // allocator. The difference is that it can only allocate a single 
    // object and the constructor and destructor must be called. 
    // In AGG there is no need to allocate an array of objects with
    // calling their constructors (only single ones). So that, if you
    // replace these new/delete to malloc/free make sure that the in-place
    // new is called and take care of calling the destructor too.
    //------------------------------------------------------------obj_allocator
    // 单个对象分配器，用于分配单个对象
    template<class T> struct obj_allocator
    {
        // 分配一个 T 类型的对象并返回指针
        static T*   allocate()         { return new T; }
        // 释放指针 ptr 指向的单个对象内存块
        static void deallocate(T* ptr) { delete ptr;   }
    };
}
#endif


//-------------------------------------------------------- Default basic types
//
// If the compiler has different capacity of the basic types you can redefine
// them via the compiler command line or by generating agg_config.h that is
// empty by default.
//
#ifndef AGG_INT8
// 定义有符号 8 位整数类型
#define AGG_INT8 signed char
#endif

#ifndef AGG_INT8U
// 定义无符号 8 位整数类型
#define AGG_INT8U unsigned char
#endif

#ifndef AGG_INT16
// 定义有符号 16 位整数类型
#define AGG_INT16 short
#endif

#ifndef AGG_INT16U
// 定义无符号 16 位整数类型
#define AGG_INT16U unsigned short
#endif

#endif // AGG_BASICS_INCLUDED
#endif

#ifndef AGG_INT32
#define AGG_INT32 int
#endif

#ifndef AGG_INT32U
#define AGG_INT32U unsigned
#endif

#ifndef AGG_INT64
#if defined(_MSC_VER) || defined(__BORLANDC__)
#define AGG_INT64 signed __int64
#else
#define AGG_INT64 signed long long
#endif
#endif

#ifndef AGG_INT64U
#if defined(_MSC_VER) || defined(__BORLANDC__)
#define AGG_INT64U unsigned __int64
#else
#define AGG_INT64U unsigned long long
#endif
#endif

//------------------------------------------------ Some fixes for MS Visual C++
#if defined(_MSC_VER)
#pragma warning(disable:4786) // Identifier was truncated...
#endif

#if defined(_MSC_VER)
#define AGG_INLINE __forceinline
#else
#define AGG_INLINE inline
#endif

namespace agg
{
    //-------------------------------------------------------------------------
    typedef AGG_INT8   int8;         // Define a typedef for 8-bit signed integer
    typedef AGG_INT8U  int8u;        // Define a typedef for 8-bit unsigned integer
    typedef AGG_INT16  int16;        // Define a typedef for 16-bit signed integer
    typedef AGG_INT16U int16u;       // Define a typedef for 16-bit unsigned integer
    typedef AGG_INT32  int32;        // Define a typedef for 32-bit signed integer
    typedef AGG_INT32U int32u;       // Define a typedef for 32-bit unsigned integer
    typedef AGG_INT64  int64;        // Define a typedef for 64-bit signed integer
    typedef AGG_INT64U int64u;       // Define a typedef for 64-bit unsigned integer

#if defined(AGG_FISTP)
#pragma warning(push)
#pragma warning(disable : 4035) // Disable warning "no return value"
    AGG_INLINE int iround(double v)              // Inline function to round a double to the nearest integer and return as int
    {
        int t;
        __asm fld   qword ptr [v]                // Load v into FPU register
        __asm fistp dword ptr [t]                // Store rounded integer value from FPU to t
        __asm mov eax, dword ptr [t]             // Move the integer value to eax (return register)
    }
    AGG_INLINE unsigned uround(double v)         // Inline function to round a double to the nearest integer and return as unsigned
    {
        unsigned t;
        __asm fld   qword ptr [v]                // Load v into FPU register
        __asm fistp dword ptr [t]                // Store rounded integer value from FPU to t
        __asm mov eax, dword ptr [t]             // Move the integer value to eax (return register)
    }
#pragma warning(pop)
    AGG_INLINE int ifloor(double v)              // Inline function to floor a double and return as int
    {
        return int(floor(v));                   // Return floored value as int
    }
    AGG_INLINE unsigned ufloor(double v)         // Inline function to floor a double and return as unsigned
    {
        return unsigned(floor(v));              // Return floored value as unsigned
    }
    AGG_INLINE int iceil(double v)               // Inline function to ceil a double and return as int
    {
        return int(ceil(v));                    // Return ceiled value as int
    }
    AGG_INLINE unsigned uceil(double v)          // Inline function to ceil a double and return as unsigned
    {
        return unsigned(ceil(v));               // Return ceiled value as unsigned
    }
#elif defined(AGG_QIFIST)
    AGG_INLINE int iround(double v)              // Inline function to round a double and return as int
    {
        return int(v);                          // Return rounded value as int
    }
    AGG_INLINE int uround(double v)              // Inline function to round a double and return as unsigned
    {
        return unsigned(v);                     // Return rounded value as unsigned
    }
    AGG_INLINE int ifloor(double v)              // Inline function to floor a double and return as int
    {
        return int(floor(v));                   // Return floored value as int
    }
    AGG_INLINE unsigned ufloor(double v)         // Inline function to floor a double and return as unsigned
    {
        return unsigned(floor(v));              // Return floored value as unsigned
    }
    AGG_INLINE int iceil(double v)               // Inline function to ceil a double and return as int
    {
        return int(ceil(v));                    // Return ceiled value as int
    }
    AGG_INLINE unsigned uceil(double v)          // Inline function to ceil a double and return as unsigned
    {
        return unsigned(ceil(v));               // Return ceiled value as unsigned
    }
#else
    AGG_INLINE int iround(double v)              // Inline function to round a double and return as int
    {
        return int((v < 0.0) ? v - 0.5 : v + 0.5); // Return rounded value based on sign of v
    }
    AGG_INLINE int uround(double v)              // Inline function to round a double and return as int
    {
        return unsigned(v + 0.5);               // Return rounded value as unsigned
    }
    AGG_INLINE int ifloor(double v)              // Inline function to floor a double and return as int
    {
        int i = int(v);                         // Convert v to int
        return i - (i > v);                     // Return floored value as int
    }
    AGG_INLINE unsigned ufloor(double v)         // Inline function to floor a double and return as unsigned
    {
        return unsigned(v);                     // Return floored value as unsigned
    }
    AGG_INLINE int iceil(double v)               // Inline function to ceil a double and return as int
    {
        return int(ceil(v));                    // Return ceiled value as int
    }
    AGG_INLINE unsigned uceil(double v)          // Inline function to ceil a double and return as unsigned
    {
        return unsigned(ceil(v));               // Return ceiled value as unsigned
    }
#endif
}
    {
        # 返回给定浮点数 v 向上取整后的整数值
        return int(ceil(v));
    }
    AGG_INLINE unsigned uceil(double v)
    {
        # 返回给定浮点数 v 向上取整后的无符号整数值
        return unsigned(ceil(v));
    }
//----------------------------------------------------------------saturation
template<int Limit> struct saturation
{
    // 返回将双精度浮点数 v 饱和到 [-Limit, Limit] 范围内的整数
    AGG_INLINE static int iround(double v)
    {
        if(v < double(-Limit)) return -Limit;
        if(v > double( Limit)) return  Limit;
        return agg::iround(v);
    }
};

//---------------------------------------------------------------mul_one
template<unsigned Shift> struct mul_one
{
    // 使用指定的位移量 Shift 对两个无符号整数进行乘法运算，实现一种带有位移量的乘法
    AGG_INLINE static unsigned mul(unsigned a, unsigned b)
    {
        unsigned q = a * b + (1 << (Shift-1));
        return (q + (q >> Shift)) >> Shift;
    }
};

//-------------------------------------------------------------------------
typedef unsigned char cover_type;    //----cover_type
enum cover_scale_e
{
    cover_shift = 8,                 //----cover_shift
    cover_size  = 1 << cover_shift,  //----cover_size 
    cover_mask  = cover_size - 1,    //----cover_mask 
    cover_none  = 0,                 //----cover_none 
    cover_full  = cover_mask         //----cover_full 
};

//----------------------------------------------------poly_subpixel_scale_e
// 这些常量确定亚像素精度，更精确地说，是坐标小数部分的位数。
// 可能的坐标容量位数可以通过以下公式计算：
// sizeof(int) * 8 - poly_subpixel_shift，例如，对于32位整数和8位小数部分，容量为24位。
enum poly_subpixel_scale_e
{
    poly_subpixel_shift = 8,                      //----poly_subpixel_shift
    poly_subpixel_scale = 1<<poly_subpixel_shift, //----poly_subpixel_scale 
    poly_subpixel_mask  = poly_subpixel_scale-1   //----poly_subpixel_mask 
};

//----------------------------------------------------------filling_rule_e
enum filling_rule_e
{
    fill_non_zero,   // 非零填充规则
    fill_even_odd    // 奇偶填充规则
};

//-----------------------------------------------------------------------pi
const double pi = 3.14159265358979323846;

//------------------------------------------------------------------deg2rad
// 将角度转换为弧度
inline double deg2rad(double deg)
{
    return deg * pi / 180.0;
}

//------------------------------------------------------------------rad2deg
// 将弧度转换为角度
inline double rad2deg(double rad)
{
    return rad * 180.0 / pi;
}
    {
        // 定义类型别名 `value_type` 和 `self_type` 分别为模板参数 `T` 和 `rect_base<T>`
        typedef T            value_type;
        typedef rect_base<T> self_type;
        
        // 定义矩形的四个顶点坐标
        T x1, y1, x2, y2;
    
        // 默认构造函数，无参数
        rect_base() {}
    
        // 带参数的构造函数，初始化矩形的四个顶点坐标
        rect_base(T x1_, T y1_, T x2_, T y2_) :
            x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
    
        // 初始化函数，用于更新矩形的四个顶点坐标
        void init(T x1_, T y1_, T x2_, T y2_) 
        {
            x1 = x1_; y1 = y1_; x2 = x2_; y2 = y2_; 
        }
    
        // 归一化函数，确保矩形左上角在左下角的左上方，右下角在右上角的右下方
        const self_type& normalize()
        {
            T t;
            if(x1 > x2) { t = x1; x1 = x2; x2 = t; }
            if(y1 > y2) { t = y1; y1 = y2; y2 = t; }
            return *this;
        }
    
        // 裁剪函数，使得当前矩形与给定矩形 `r` 相交，并更新当前矩形的坐标
        bool clip(const self_type& r)
        {
            if(x2 > r.x2) x2 = r.x2;
            if(y2 > r.y2) y2 = r.y2;
            if(x1 < r.x1) x1 = r.x1;
            if(y1 < r.y1) y1 = r.y1;
            return x1 <= x2 && y1 <= y2;
        }
    
        // 判断矩形是否有效，即左上角坐标小于等于右下角坐标
        bool is_valid() const
        {
            return x1 <= x2 && y1 <= y2;
        }
    
        // 点击测试函数，判断给定的点 `(x, y)` 是否在矩形内部
        bool hit_test(T x, T y) const
        {
            return (x >= x1 && x <= x2 && y >= y1 && y <= y2);
        }
        
        // 判断当前矩形是否与给定的矩形 `r` 重叠
        bool overlaps(const self_type& r) const
        {
            return !(r.x1 > x2 || r.x2 < x1
                  || r.y1 > y2 || r.y2 < y1);
        }
    };
    
    //-----------------------------------------------------intersect_rectangles
    // 模板函数，计算两个矩形 `r1` 和 `r2` 的交集，并返回结果矩形
    template<class Rect> 
    inline Rect intersect_rectangles(const Rect& r1, const Rect& r2)
    {
        Rect r = r1;
    
        // 优先处理右下角坐标，因为反向处理可能导致编译器在某些情况下出错
        if(r.x2 > r2.x2) r.x2 = r2.x2; 
        if(r.y2 > r2.y2) r.y2 = r2.y2;
        if(r.x1 < r2.x1) r.x1 = r2.x1;
        if(r.y1 < r2.y1) r.y1 = r2.y1;
        return r;
    }
    
    
    //---------------------------------------------------------unite_rectangles
    // 模板函数，计算两个矩形 `r1` 和 `r2` 的并集，并返回结果矩形
    template<class Rect> 
    inline Rect unite_rectangles(const Rect& r1, const Rect& r2)
    {
        Rect r = r1;
        if(r.x2 < r2.x2) r.x2 = r2.x2;
        if(r.y2 < r2.y2) r.y2 = r2.y2;
        if(r.x1 > r2.x1) r.x1 = r2.x1;
        if(r.y1 > r2.y1) r.y1 = r2.y1;
        return r;
    }
    
    // 定义整数、浮点数和双精度浮点数类型的矩形别名
    typedef rect_base<int>    rect_i; //----rect_i
    typedef rect_base<float>  rect_f; //----rect_f
    typedef rect_base<double> rect_d; //----rect_d
    
    //---------------------------------------------------------path_commands_e
    // 枚举类型，表示路径的各种命令
    enum path_commands_e
    {
        path_cmd_stop     = 0,        // 定义停止路径指令的常量，值为0
        path_cmd_move_to  = 1,        // 定义移动到指令的常量，值为1
        path_cmd_line_to  = 2,        // 定义直线到指令的常量，值为2
        path_cmd_curve3   = 3,        // 定义三次贝塞尔曲线指令的常量，值为3
        path_cmd_curve4   = 4,        // 定义四次贝塞尔曲线指令的常量，值为4
        path_cmd_curveN   = 5,        // 定义多次贝塞尔曲线指令的常量，值为5
        path_cmd_catrom   = 6,        // 定义Catmull-Rom样条曲线指令的常量，值为6
        path_cmd_ubspline = 7,        // 定义均匀B样条曲线指令的常量，值为7
        path_cmd_end_poly = 0x0F,     // 定义闭合多边形结束指令的常量，值为0x0F
        path_cmd_mask     = 0x0F      // 路径指令掩码，用于提取路径指令类型
    };
    
    //------------------------------------------------------------path_flags_e
    enum path_flags_e
    {
        path_flags_none  = 0,         // 不设置任何路径标志的常量，值为0
        path_flags_ccw   = 0x10,      // 逆时针路径标志的常量，值为0x10
        path_flags_cw    = 0x20,      // 顺时针路径标志的常量，值为0x20
        path_flags_close = 0x40,      // 闭合路径标志的常量，值为0x40
        path_flags_mask  = 0xF0       // 路径标志掩码，用于提取路径标志类型
    };
    
    //---------------------------------------------------------------is_vertex
    inline bool is_vertex(unsigned c)
    {
        // 检查指令是否表示顶点，即在移动到指令和闭合多边形结束指令之间
        return c >= path_cmd_move_to && c < path_cmd_end_poly;
    }
    
    //--------------------------------------------------------------is_drawing
    inline bool is_drawing(unsigned c)
    {
        // 检查指令是否表示绘图操作，即在直线到指令和闭合多边形结束指令之间
        return c >= path_cmd_line_to && c < path_cmd_end_poly;
    }
    
    //-----------------------------------------------------------------is_stop
    inline bool is_stop(unsigned c)
    { 
        // 检查指令是否为停止路径指令
        return c == path_cmd_stop;
    }
    
    //--------------------------------------------------------------is_move_to
    inline bool is_move_to(unsigned c)
    {
        // 检查指令是否为移动到指令
        return c == path_cmd_move_to;
    }
    
    //--------------------------------------------------------------is_line_to
    inline bool is_line_to(unsigned c)
    {
        // 检查指令是否为直线到指令
        return c == path_cmd_line_to;
    }
    
    //----------------------------------------------------------------is_curve
    inline bool is_curve(unsigned c)
    {
        // 检查指令是否为任何类型的贝塞尔曲线指令
        return c == path_cmd_curve3 || c == path_cmd_curve4;
    }
    
    //---------------------------------------------------------------is_curve3
    inline bool is_curve3(unsigned c)
    {
        // 检查指令是否为三次贝塞尔曲线指令
        return c == path_cmd_curve3;
    }
    
    //---------------------------------------------------------------is_curve4
    inline bool is_curve4(unsigned c)
    {
        // 检查指令是否为四次贝塞尔曲线指令
        return c == path_cmd_curve4;
    }
    
    //-------------------------------------------------------------is_end_poly
    inline bool is_end_poly(unsigned c)
    {
        // 检查指令是否为闭合多边形结束指令
        return (c & path_cmd_mask) == path_cmd_end_poly;
    }
    
    //----------------------------------------------------------------is_close
    inline bool is_close(unsigned c)
    {
        // 检查指令是否为闭合路径，即结合了路径结束和闭合标志
        return (c & ~(path_flags_cw | path_flags_ccw)) ==
               (path_cmd_end_poly | path_flags_close); 
    }
    
    //------------------------------------------------------------is_next_poly
    // 检查给定命令是否为下一个多边形命令（停止、移动到或结束多边形）
    inline bool is_next_poly(unsigned c)
    {
        return is_stop(c) || is_move_to(c) || is_end_poly(c);
    }

    //-------------------------------------------------------------------is_cw
    // 检查命令是否包含顺时针方向标志位
    inline bool is_cw(unsigned c)
    {
        return (c & path_flags_cw) != 0;
    }

    //------------------------------------------------------------------is_ccw
    // 检查命令是否包含逆时针方向标志位
    inline bool is_ccw(unsigned c)
    {
        return (c & path_flags_ccw) != 0;
    }

    //-------------------------------------------------------------is_oriented
    // 检查命令是否包含方向标志位（顺时针或逆时针）
    inline bool is_oriented(unsigned c)
    {
        return (c & (path_flags_cw | path_flags_ccw)) != 0; 
    }

    //---------------------------------------------------------------is_closed
    // 检查命令是否包含闭合标志位
    inline bool is_closed(unsigned c)
    {
        return (c & path_flags_close) != 0; 
    }

    //----------------------------------------------------------get_close_flag
    // 获取命令的闭合标志位
    inline unsigned get_close_flag(unsigned c)
    {
        return c & path_flags_close; 
    }

    //-------------------------------------------------------clear_orientation
    // 清除命令中的方向标志位（顺时针和逆时针）
    inline unsigned clear_orientation(unsigned c)
    {
        return c & ~(path_flags_cw | path_flags_ccw);
    }

    //---------------------------------------------------------get_orientation
    // 获取命令中的方向标志位（顺时针或逆时针）
    inline unsigned get_orientation(unsigned c)
    {
        return c & (path_flags_cw | path_flags_ccw);
    }

    //---------------------------------------------------------set_orientation
    // 设置命令的方向标志位（顺时针或逆时针），先清除原有的标志位，再设置新的标志位
    inline unsigned set_orientation(unsigned c, unsigned o)
    {
        return clear_orientation(c) | o;
    }

    //--------------------------------------------------------------point_base
    // 点基类模板，表示具有类型 T 的点，包含 x 和 y 坐标
    template<class T> struct point_base
    {
        typedef T value_type;
        T x,y;
        point_base() {}
        point_base(T x_, T y_) : x(x_), y(y_) {}
    };
    typedef point_base<int>    point_i; //-----point_i，整数型点
    typedef point_base<float>  point_f; //-----point_f，单精度浮点型点
    typedef point_base<double> point_d; //-----point_d，双精度浮点型点

    //-------------------------------------------------------------vertex_base
    // 顶点基类模板，表示具有类型 T 的顶点，包含 x 和 y 坐标以及命令信息
    template<class T> struct vertex_base
    {
        typedef T value_type;
        T x,y;
        unsigned cmd;
        vertex_base() {}
        vertex_base(T x_, T y_, unsigned cmd_) : x(x_), y(y_), cmd(cmd_) {}
    };
    typedef vertex_base<int>    vertex_i; //-----vertex_i，整数型顶点
    typedef vertex_base<float>  vertex_f; //-----vertex_f，单精度浮点型顶点
    typedef vertex_base<double> vertex_d; //-----vertex_d，双精度浮点型顶点

    //----------------------------------------------------------------row_info
    // 行信息模板，表示具有类型 T 的行信息，包含起始和结束 x 坐标以及指向数据的指针
    template<class T> struct row_info
    {
        int x1, x2;
        T* ptr;
        row_info() {}
        row_info(int x1_, int x2_, T* ptr_) : x1(x1_), x2(x2_), ptr(ptr_) {}
    };

    //----------------------------------------------------------const_row_info
    // 常量行信息模板，表示具有类型 T 的常量行信息，包含起始和结束 x 坐标以及指向数据的指针
    template<class T> struct const_row_info
    {
        // 定义一个结构体 const_row_info，包含两个整型成员 x1 和 x2，以及一个指向常量 T 类型的指针 ptr
        int x1, x2;
        const T* ptr;
    
        // 默认构造函数 const_row_info，不执行任何操作
        const_row_info() {}
    
        // 带参构造函数 const_row_info，用来初始化 x1、x2 和 ptr 成员变量
        const_row_info(int x1_, int x2_, const T* ptr_) :
            x1(x1_), x2(x2_), ptr(ptr_) {}
    };
    
    //------------------------------------------------------------is_equal_eps
    // 模板函数，比较两个值 v1 和 v2 是否在给定的 epsilon 精度内相等
    template<class T> inline bool is_equal_eps(T v1, T v2, T epsilon)
    {
        // 使用 fabs 函数计算 v1 和 v2 之间的绝对差，并与 epsilon 比较
        return fabs(v1 - v2) <= double(epsilon);
    }
}


注释：

// 关闭一个条件编译指令块的结束标志



#endif


注释：

// 如果当前的条件编译指令与之前的 #if、#ifdef 或 #ifndef 匹配，则结束该条件编译块
```