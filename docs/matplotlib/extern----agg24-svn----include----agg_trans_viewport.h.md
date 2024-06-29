# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_trans_viewport.h`

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
// Viewport transformer - simple orthogonal conversions from world coordinates
//                        to screen (device) ones.
//
//----------------------------------------------------------------------------

#ifndef AGG_TRANS_VIEWPORT_INCLUDED
#define AGG_TRANS_VIEWPORT_INCLUDED

#include <string.h>
#include "agg_trans_affine.h"

// 命名空间 agg 中定义的几种纵横比模式
namespace agg
{

    enum aspect_ratio_e
    {
        aspect_ratio_stretch,  // 拉伸模式
        aspect_ratio_meet,     // 适应模式
        aspect_ratio_slice     // 切片模式
    };


    //----------------------------------------------------------trans_viewport
    class trans_viewport
    {
    private:
        void update();  // 更新视口参数的私有方法

        double         m_world_x1;     // 世界坐标系的左下角 X 坐标
        double         m_world_y1;     // 世界坐标系的左下角 Y 坐标
        double         m_world_x2;     // 世界坐标系的右上角 X 坐标
        double         m_world_y2;     // 世界坐标系的右上角 Y 坐标
        double         m_device_x1;    // 设备坐标系的左下角 X 坐标
        double         m_device_y1;    // 设备坐标系的左下角 Y 坐标
        double         m_device_x2;    // 设备坐标系的右上角 X 坐标
        double         m_device_y2;    // 设备坐标系的右上角 Y 坐标
        aspect_ratio_e m_aspect;       // 当前视口的纵横比模式
        bool           m_is_valid;     // 视口参数是否有效的标志
        double         m_align_x;      // X 方向的对齐参数
        double         m_align_y;      // Y 方向的对齐参数
        double         m_wx1;          // 内部使用的世界坐标系左下角 X 坐标
        double         m_wy1;          // 内部使用的世界坐标系左下角 Y 坐标
        double         m_wx2;          // 内部使用的世界坐标系右上角 X 坐标
        double         m_wy2;          // 内部使用的世界坐标系右上角 Y 坐标
        double         m_dx1;          // 内部使用的设备坐标系左下角 X 坐标
        double         m_dy1;          // 内部使用的设备坐标系左下角 Y 坐标
        double         m_kx;           // 内部使用的 X 方向缩放因子
        double         m_ky;           // 内部使用的 Y 方向缩放因子
    };



    //-----------------------------------------------------------------------
    inline void trans_viewport::update()
    {
        // 定义一个极小的值 epsilon，用于比较浮点数是否接近于零
        const double epsilon = 1e-30;
        
        // 如果任何两个坐标之间的差值小于 epsilon，表示几何信息无效，进行初始化
        if(fabs(m_world_x1  - m_world_x2)  < epsilon ||
           fabs(m_world_y1  - m_world_y2)  < epsilon ||
           fabs(m_device_x1 - m_device_x2) < epsilon ||
           fabs(m_device_y1 - m_device_y2) < epsilon)
        {
            // 初始化世界坐标和设备坐标
            m_wx1 = m_world_x1;
            m_wy1 = m_world_y1;
            m_wx2 = m_world_x1 + 1.0;
            m_wy2 = m_world_y2 + 1.0;
            m_dx1 = m_device_x1;
            m_dy1 = m_device_y1;
            m_kx  = 1.0;
            m_ky  = 1.0;
            m_is_valid = false;
            return;
        }
    
        // 保存当前世界坐标和设备坐标
        double world_x1  = m_world_x1;
        double world_y1  = m_world_y1;
        double world_x2  = m_world_x2;
        double world_y2  = m_world_y2;
        double device_x1 = m_device_x1;
        double device_y1 = m_device_y1;
        double device_x2 = m_device_x2;
        double device_y2 = m_device_y2;
    
        // 如果缩放模式不是拉伸模式
        if(m_aspect != aspect_ratio_stretch)
        {
            double d;
            // 计算水平和垂直方向的缩放比例
            m_kx = (device_x2 - device_x1) / (world_x2 - world_x1);
            m_ky = (device_y2 - device_y1) / (world_y2 - world_y1);
    
            // 如果缩放模式是 "meet" 且水平缩放比例小于垂直缩放比例，或者相反
            if((m_aspect == aspect_ratio_meet) == (m_kx < m_ky))
            {
                // 根据垂直方向的缩放比例调整世界坐标范围，以及对齐偏移量
                d         = (world_y2 - world_y1) * m_ky / m_kx;
                world_y1 += (world_y2 - world_y1 - d) * m_align_y;
                world_y2  =  world_y1 + d;
            }
            else
            {
                // 根据水平方向的缩放比例调整世界坐标范围，以及对齐偏移量
                d         = (world_x2 - world_x1) * m_kx / m_ky;
                world_x1 += (world_x2 - world_x1 - d) * m_align_x;
                world_x2  =  world_x1 + d;
            }
        }
    
        // 更新世界坐标和设备坐标
        m_wx1 = world_x1;
        m_wy1 = world_y1;
        m_wx2 = world_x2;
        m_wy2 = world_y2;
        m_dx1 = device_x1;
        m_dy1 = device_y1;
        m_kx  = (device_x2 - device_x1) / (world_x2 - world_x1);
        m_ky  = (device_y2 - device_y1) / (world_y2 - world_y1);
        m_is_valid = true;
    }
}


注释：


// 结束一个代码块或函数定义的标准 C/C++ 语法，用于匹配相应的开头



#endif


注释：


// 在条件编译中，用于结束一个条件块的标准 C/C++ 预处理指令，匹配相应的 #ifdef 或 #ifndef
```