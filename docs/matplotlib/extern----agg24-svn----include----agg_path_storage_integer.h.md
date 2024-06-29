# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_path_storage_integer.h`

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

#ifndef AGG_PATH_STORAGE_INTEGER_INCLUDED
#define AGG_PATH_STORAGE_INTEGER_INCLUDED

#include <string.h>
#include "agg_array.h"

namespace agg
{
    //---------------------------------------------------------vertex_integer
    // 整数顶点结构体，用于表示路径中的顶点坐标
    template<class T, unsigned CoordShift=6> struct vertex_integer
    {
        // 路径指令类型枚举
        enum path_cmd
        {
            cmd_move_to = 0,  // 移动到指令
            cmd_line_to = 1,  // 直线到指令
            cmd_curve3  = 2,  // 三次贝塞尔曲线指令
            cmd_curve4  = 3   // 四次贝塞尔曲线指令
        };

        // 坐标缩放常量枚举
        enum coord_scale_e
        {
            coord_shift = CoordShift,         // 坐标移位数
            coord_scale  = 1 << coord_shift   // 坐标缩放比例
        };

        T x,y;  // 整数类型的坐标值 x 和 y

        // 默认构造函数
        vertex_integer() {}

        // 构造函数，将传入的浮点坐标值 x_, y_ 转换成整数类型的坐标，使用 flag 进行标志
        vertex_integer(T x_, T y_, unsigned flag) :
            x(((x_ << 1) & ~1) | (flag &  1)),   // 对 x 进行位移和标志位设置
            y(((y_ << 1) & ~1) | (flag >> 1)) {} // 对 y 进行位移和标志位设置

        // 将整数坐标转换为浮点坐标，并根据其标志位返回相应路径指令类型
        unsigned vertex(double* x_, double* y_, 
                        double dx=0, double dy=0,
                        double scale=1.0) const
        {
            *x_ = dx + (double(x >> 1) / coord_scale) * scale;  // 计算 x 坐标的浮点值
            *y_ = dy + (double(y >> 1) / coord_scale) * scale;  // 计算 y 坐标的浮点值
            switch(((y & 1) << 1) | (x & 1))  // 根据 x 和 y 的最低位组合判断路径指令类型
            {
                case cmd_move_to: return path_cmd_move_to;  // 移动到指令
                case cmd_line_to: return path_cmd_line_to;  // 直线到指令
                case cmd_curve3:  return path_cmd_curve3;   // 三次贝塞尔曲线指令
                case cmd_curve4:  return path_cmd_curve4;   // 四次贝塞尔曲线指令
            }
            return path_cmd_stop;  // 停止指令
        }
    };


    //---------------------------------------------------path_storage_integer
    // 整数路径存储类模板
    template<class T, unsigned CoordShift=6> class path_storage_integer
    {
    private:
        pod_bvector<vertex_integer_type, 6> m_storage;  // 使用 pod_bvector 存储整数顶点
        unsigned                            m_vertex_idx;  // 当前顶点索引
        bool                                m_closed;  // 路径是否封闭标志
    };




    //-----------------------------------------serialized_integer_path_adaptor
    // 序列化整数路径适配器类模板
    template<class T, unsigned CoordShift=6> class serialized_integer_path_adaptor
    {
    // 定义一个公共类，其中包含顶点整数类型的别名vertex_integer_type
    public:
        typedef vertex_integer<T, CoordShift> vertex_integer_type;
    
        //--------------------------------------------------------------------
        // 默认构造函数，初始化各成员变量为默认值
        serialized_integer_path_adaptor() :
            m_data(0),                // 数据指针初始化为0
            m_end(0),                 // 结束指针初始化为0
            m_ptr(0),                 // 当前指针初始化为0
            m_dx(0.0),                // dx初始化为0.0
            m_dy(0.0),                // dy初始化为0.0
            m_scale(1.0),             // scale初始化为1.0
            m_vertices(0)             // 顶点数初始化为0
        {}
    
        //--------------------------------------------------------------------
        // 带参数的构造函数，初始化各成员变量
        serialized_integer_path_adaptor(const int8u* data, unsigned size,
                                        double dx, double dy) :
            m_data(data),             // 数据指针初始化为给定的data
            m_end(data + size),       // 结束指针初始化为data + size
            m_ptr(data),              // 当前指针初始化为data
            m_dx(dx),                 // dx初始化为给定的dx
            m_dy(dy),                 // dy初始化为给定的dy
            m_vertices(0)             // 顶点数初始化为0
        {}
    
        //--------------------------------------------------------------------
        // 初始化函数，用于更新各成员变量的值
        void init(const int8u* data, unsigned size, 
                  double dx, double dy, double scale=1.0)
        {
            m_data     = data;        // 更新数据指针
            m_end      = data + size; // 更新结束指针
            m_ptr      = data;        // 更新当前指针
            m_dx       = dx;          // 更新dx
            m_dy       = dy;          // 更新dy
            m_scale    = scale;       // 更新scale
            m_vertices = 0;           // 重置顶点数为0
        }
    
        //--------------------------------------------------------------------
        // 将当前指针重置为数据起始位置，并重置顶点数为0
        void rewind(unsigned) 
        { 
            m_ptr      = m_data;      // 重置当前指针为数据起始位置
            m_vertices = 0;           // 重置顶点数为0
        }
    
        //--------------------------------------------------------------------
        // 获取当前顶点的坐标，并返回路径命令
        unsigned vertex(double* x, double* y)
        {
            // 如果数据指针为空或者当前指针超出结束指针，则返回停止路径命令
            if(m_data == 0 || m_ptr > m_end) 
            {
                *x = 0;
                *y = 0;
                return path_cmd_stop;
            }
    
            // 如果当前指针等于结束指针，则返回结束多边形路径命令
            if(m_ptr == m_end)
            {
                *x = 0;
                *y = 0;
                m_ptr += sizeof(vertex_integer_type);  // 移动当前指针到下一个顶点
                return path_cmd_end_poly | path_flags_close;
            }
    
            // 从当前指针位置读取顶点整数类型的数据，并调用其方法计算顶点坐标及路径命令
            vertex_integer_type v;
            memcpy(&v, m_ptr, sizeof(vertex_integer_type));
            unsigned cmd = v.vertex(x, y, m_dx, m_dy, m_scale);
    
            // 如果是移动到命令并且顶点数大于2，则返回结束多边形路径命令
            if(is_move_to(cmd) && m_vertices > 2)
            {
                *x = 0;
                *y = 0;
                m_vertices = 0;  // 重置顶点数为0
                return path_cmd_end_poly | path_flags_close;
            }
            ++m_vertices;         // 增加顶点数计数
            m_ptr += sizeof(vertex_integer_type);  // 移动当前指针到下一个顶点
            return cmd;           // 返回计算得到的路径命令
        }
    
    private:
        const int8u* m_data;        // 数据指针，指向序列化数据的起始位置
        const int8u* m_end;         // 结束指针，指向序列化数据的末尾位置
        const int8u* m_ptr;         // 当前指针，指向当前处理的位置
        double       m_dx;          // x轴偏移量
        double       m_dy;          // y轴偏移量
        double       m_scale;       // 缩放比例
        unsigned     m_vertices;    // 顶点数计数
    };
}


注释：

// 结束一个函数的定义。在 C/C++ 中，"}" 表示函数或代码块的结束。



#endif


注释：

// 如果定义了条件编译指令中使用的宏，关闭条件编译块。在 C/C++ 中，"#endif" 用于结束 "#ifdef" 或 "#ifndef" 的条件编译段落。
```