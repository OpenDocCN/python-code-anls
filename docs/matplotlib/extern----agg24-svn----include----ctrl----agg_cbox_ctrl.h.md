# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_cbox_ctrl.h`

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
// classes cbox_ctrl_impl, cbox_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_CBOX_CTRL_INCLUDED
#define AGG_CBOX_CTRL_INCLUDED

#include "agg_basics.h"
#include "agg_conv_stroke.h"
#include "agg_gsv_text.h"
#include "agg_trans_affine.h"
#include "agg_color_rgba.h"
#include "agg_ctrl.h"

// 引入AGG库的基础组件和控件相关的头文件

namespace agg
{

    //----------------------------------------------------------cbox_ctrl_impl
    // 控件实现类 cbox_ctrl_impl，继承自控件基类 ctrl
    class cbox_ctrl_impl : public ctrl
    {
    public:
        // 构造函数，初始化控件位置和标签，可选是否翻转Y轴
        cbox_ctrl_impl(double x, double y, const char* label, bool flip_y=false);

        // 设置文本厚度
        void text_thickness(double t)  { m_text_thickness = t; }
        
        // 设置文本大小
        void text_size(double h, double w=0.0);

        // 获取标签内容
        const char* label() { return m_label; }
        
        // 设置标签内容
        void label(const char* l);

        // 获取控件状态
        bool status() const { return m_status; }
        
        // 设置控件状态
        void status(bool st) { m_status = st; }

        // 检查指定坐标是否在控件的矩形范围内
        virtual bool in_rect(double x, double y) const;
        
        // 处理鼠标按下事件
        virtual bool on_mouse_button_down(double x, double y);
        
        // 处理鼠标释放事件
        virtual bool on_mouse_button_up(double x, double y);
        
        // 处理鼠标移动事件
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        
        // 处理箭头键事件
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        // 顶点源接口函数，返回路径数
        unsigned num_paths() { return 3; };
        
        // 准备迭代指定路径的顶点
        void     rewind(unsigned path_id);
        
        // 返回当前路径的顶点坐标
        unsigned vertex(double* x, double* y);

    private:
        double   m_text_thickness;          // 文本厚度
        double   m_text_height;             // 文本高度
        double   m_text_width;              // 文本宽度
        char     m_label[128];              // 标签字符串
        bool     m_status;                  // 控件状态
        double   m_vx[32];                  // X坐标数组
        double   m_vy[32];                  // Y坐标数组

        gsv_text              m_text;       // 文本对象
        conv_stroke<gsv_text> m_text_poly;  // 描边文本对象

        unsigned m_idx;                     // 索引
        unsigned m_vertex;                  // 顶点计数
    };


    //----------------------------------------------------------cbox_ctrl_impl
    // 模板类 cbox_ctrl，继承自 cbox_ctrl_impl，使用模板参数 ColorT
    template<class ColorT> class cbox_ctrl : public cbox_ctrl_impl
    {
    // 公有构造函数，初始化复选框控件
    public:
        cbox_ctrl(double x, double y, const char* label, bool flip_y=false) :
            // 调用基类的构造函数初始化位置和标签
            cbox_ctrl_impl(x, y, label, flip_y),
            // 初始化文本颜色为黑色
            m_text_color(rgba(0.0, 0.0, 0.0)),
            // 初始化非活动状态颜色为黑色
            m_inactive_color(rgba(0.0, 0.0, 0.0)),
            // 初始化活动状态颜色为深红色
            m_active_color(rgba(0.4, 0.0, 0.0))
        {
            // 将颜色指针数组设置为指向对应颜色的指针
            m_colors[0] = &m_inactive_color;
            m_colors[1] = &m_text_color;
            m_colors[2] = &m_active_color;
        }
          
        // 设置文本颜色
        void text_color(const ColorT& c) { m_text_color = c; }
        
        // 设置非活动状态颜色
        void inactive_color(const ColorT& c) { m_inactive_color = c; }
        
        // 设置活动状态颜色
        void active_color(const ColorT& c) { m_active_color = c; }

        // 获取指定索引处的颜色
        const ColorT& color(unsigned i) const { return *m_colors[i]; } 

    // 私有成员变量
    private:
        // 复制构造函数私有化
        cbox_ctrl(const cbox_ctrl<ColorT>&);
        // 赋值运算符私有化
        const cbox_ctrl<ColorT>& operator = (const cbox_ctrl<ColorT>&);

        // 文本颜色
        ColorT m_text_color;
        // 非活动状态颜色
        ColorT m_inactive_color;
        // 活动状态颜色
        ColorT m_active_color;
        // 颜色指针数组
        ColorT* m_colors[3];
    };
}


注释：


// 这行是 C/C++ 中的预处理器指令，用于结束一个条件编译区块



#endif


注释：


// 这行是 C/C++ 中的预处理器指令，用于结束一个条件编译指令，对应于 #ifdef 或 #ifndef
```