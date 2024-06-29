# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_interpolator_adaptor.h`

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

#ifndef AGG_SPAN_INTERPOLATOR_ADAPTOR_INCLUDED
#define AGG_SPAN_INTERPOLATOR_ADAPTOR_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //===============================================span_interpolator_adaptor
    template<class Interpolator, class Distortion>
    class span_interpolator_adaptor : public Interpolator
    {
    public:
        typedef Interpolator base_type;
        typedef typename base_type::trans_type trans_type;
        typedef Distortion distortion_type;

        //--------------------------------------------------------------------
        // 默认构造函数，不执行任何操作
        span_interpolator_adaptor() {}

        // 构造函数，接受变换和失真类型作为参数，初始化基类和失真对象指针
        span_interpolator_adaptor(trans_type& trans, 
                                  distortion_type& dist) :
            base_type(trans),
            m_distortion(&dist)
        {   
        }

        // 构造函数，接受变换、失真类型以及坐标和长度作为参数，初始化基类和失真对象指针
        span_interpolator_adaptor(trans_type& trans,
                                  distortion_type& dist,
                                  double x, double y, unsigned len) :
            base_type(trans, x, y, len),
            m_distortion(&dist)
        {
        }

        //--------------------------------------------------------------------
        // 返回失真对象的引用
        distortion_type& distortion() const
        {
            return *m_distortion;
        }

        // 设置失真对象的新引用
        void distortion(distortion_type& dist)
        {
            m_distortion = dist;
        }

        // 获取坐标，并通过失真对象计算失真后的坐标
        void coordinates(int* x, int* y) const
        {
            base_type::coordinates(x, y); // 调用基类的坐标获取方法
            m_distortion->calculate(x, y); // 使用失真对象计算坐标的失真效果
        }

    private:
        //--------------------------------------------------------------------
        distortion_type* m_distortion; // 指向失真对象的指针
    };
}

#endif
```