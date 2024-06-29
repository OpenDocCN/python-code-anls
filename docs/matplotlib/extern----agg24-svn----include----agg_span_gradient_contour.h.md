# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_gradient_contour.h`

```py
//----------------------------------------------------------------------------
// AGG Contribution Pack - Gradients 1 (AGG CP - Gradients 1)
// http://milan.marusinec.sk/aggcp
//
// For Anti-Grain Geometry - Version 2.4
// http://www.antigrain.org
//
// Contribution Created By:
//  Milan Marusinec alias Milano
//  milan@marusinec.sk
//  Copyright (c) 2007-2008
//
// Permission to copy, use, modify, sell and distribute this software
// is granted provided this copyright notice appears in all copies.
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
// [History] -----------------------------------------------------------------
//
// 02.02.2008-Milano: Ported from Object Pascal code of AggPas
//

#ifndef AGG_SPAN_GRADIENT_CONTOUR_INCLUDED
#define AGG_SPAN_GRADIENT_CONTOUR_INCLUDED

#include "agg_basics.h"
#include "agg_trans_affine.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_gray.h"
#include "agg_conv_transform.h"
#include "agg_conv_curve.h"
#include "agg_bounding_rect.h"
#include "agg_renderer_base.h"
#include "agg_renderer_primitives.h"
#include "agg_rasterizer_outline.h"
#include "agg_span_gradient.h"

#define infinity 1E20

namespace agg
{

    //==========================================================gradient_contour
    // 定义渐变轮廓类
    class gradient_contour
    {
    private:
        // 渐变轮廓类的私有成员变量
        int8u* m_buffer;    // 缓冲区指针，指向渐变轮廓数据
        int       m_width;    // 缓冲区宽度
        int    m_height;   // 缓冲区高度
        int    m_frame;    // 帧数

        double m_d1;        // 双精度浮点数变量1
        double m_d2;        // 双精度浮点数变量2
    // 定义公共部分和默认构造函数，初始化成员变量
    public:
        gradient_contour() :
            m_buffer(NULL),                              // 初始化缓冲区指针为空
            m_width(0),                                  // 初始化宽度为0
            m_height(0),                                 // 初始化高度为0
            m_frame(10),                                 // 初始化帧为10
            m_d1(0),                                     // 初始化d1为0
            m_d2(100)                                    // 初始化d2为100
        {
        }

        // 带参数的构造函数，初始化成员变量
        gradient_contour(double d1, double d2) :
            m_buffer(NULL),                              // 初始化缓冲区指针为空
            m_width(0),                                  // 初始化宽度为0
            m_height(0),                                 // 初始化高度为0
            m_frame(10),                                 // 初始化帧为10
            m_d1(d1),                                    // 初始化d1为给定值
            m_d2(d2)                                     // 初始化d2为给定值
        {
        }

        // 析构函数，释放动态分配的缓冲区
        ~gradient_contour()
        {
            if (m_buffer)
            {
                delete [] m_buffer;                     // 删除缓冲区数组
            }
        }

        // 创建轮廓路径的函数声明，参数为路径存储对象指针
        int8u* contour_create(path_storage* ps );

        // 返回轮廓宽度的函数
        int    contour_width() { return m_width; }

        // 返回轮廓高度的函数
        int    contour_height() { return m_height; }

        // 设置d1的函数
        void   d1(double d ) { m_d1 = d; }

        // 设置d2的函数
        void   d2(double d ) { m_d2 = d; }

        // 设置帧大小的函数
        void   frame(int f ) { m_frame = f; }

        // 返回帧大小的函数
        int    frame() { return m_frame; }

        // 计算梯度的函数
        int calculate(int x, int y, int d) const
        {
            if (m_buffer)
            {
                int px = x >> agg::gradient_subpixel_shift;  // 右移操作
                int py = y >> agg::gradient_subpixel_shift;  // 右移操作

                px %= m_width;                              // 求模运算

                if (px < 0)
                {
                    px += m_width;                          // 调整负数情况
                }

                py %= m_height;                             // 求模运算

                if (py < 0 )
                {
                    py += m_height;                         // 调整负数情况
                }

                // 返回梯度计算结果，使用缓冲区数据和给定的d1和d2值
                return iround(m_buffer[py * m_width + px ] * (m_d2 / 256 ) + m_d1 ) << agg::gradient_subpixel_shift;
            }
            else
            {
                return 0;                                   // 缓冲区为空时返回0
            }
        }
    };

    // 内联函数，计算给定整数的平方
    static AGG_INLINE int square(int x ) { return x * x; }

    // DT算法，作者为Pedro Felzenszwalb
    void dt(float* spanf, float* spang, float* spanr, int* spann ,int length )
    {
        int k = 0;
        float s;

        spann[0 ] = 0;                                      // 初始化spann数组第一个元素为0
        spang[0 ] = float(-infinity );                      // 初始化spang数组第一个元素为负无穷
        spang[1 ] = float(+infinity );                      // 初始化spang数组第二个元素为正无穷

        // 开始DT算法的主循环
        for (int q = 1; q <= length - 1; q++)
        {
            s = ((spanf[q ] + square(q ) ) - (spanf[spann[k ] ] + square(spann[k ] ) ) ) / (2 * q - 2 * spann[k ] );

            while (s <= spang[k ])
            {
                k--;
                s = ((spanf[q ] + square(q ) ) - (spanf[spann[k ] ] + square(spann[k ] ) ) ) / (2 * q - 2 * spann[k ] );
            }

            k++;
            spann[k ] = q;
            spang[k ] = s;
            spang[k + 1 ] = float(+infinity);
        }

        k = 0;

        // 完成DT算法的第二阶段
        for (int q = 0; q <= length - 1; q++)
        {
            while (spang[k + 1 ] < q )
            {
                k++;
            }

            spanr[q ] = square(q - spann[k ] ) + spanf[spann[k ] ];
        }
    }

    // DT算法，作者为Pedro Felzenszwalb
    int8u* gradient_contour::contour_create(path_storage* ps )
    {
}
// 关闭条件编译指令，结束预处理器条件编译区块
#endif
// 表示条件编译的结束，对应于预处理器指令 #ifdef 或 #if 的结束
```