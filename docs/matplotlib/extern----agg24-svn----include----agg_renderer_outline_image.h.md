# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_renderer_outline_image.h`

```
// 定义了一个模板类 line_image_scale，用于对图像进行线性缩放处理
template<class Source> class line_image_scale
{
public:
    typedef typename Source::color_type color_type; // 定义颜色类型为 Source 类中的 color_type

    // 构造函数，接受源图像和目标高度参数
    line_image_scale(const Source& src, double height) :
        m_source(src), // 初始化成员变量 m_source，表示源图像
        m_height(height), // 初始化成员变量 m_height，表示目标高度
        m_scale(src.height() / height), // 计算缩放比例，源图像高度与目标高度的比值
        m_scale_inv(height / src.height()) // 计算缩放比例的倒数
    {
    }

    // 返回源图像的宽度
    double width()  const { return m_source.width(); }
    
    // 返回目标高度
    double height() const { return m_height; }

    // 获取缩放后的像素颜色值
    color_type pixel(int x, int y) const 
    { 
        // 如果缩放比例小于1.0，执行以下操作
        if (m_scale < 1.0)
        {
            // 在最近的源像素之间进行插值
            double src_y = (y + 0.5) * m_scale - 0.5; // 计算插值位置在源图像中的纵坐标
            int h  = m_source.height() - 1; // 源图像高度减1
            int y1 = ifloor(src_y); // 取插值位置的下限
            int y2 = y1 + 1; // 取插值位置的上限
            rgba pix1 = (y1 < 0) ? rgba::no_color() : m_source.pixel(x, y1); // 获取下限像素颜色
            rgba pix2 = (y2 > h) ? rgba::no_color() : m_source.pixel(x, y2); // 获取上限像素颜色
            return pix1.gradient(pix2, src_y - y1); // 返回插值结果
        }
        else // 如果缩放比例大于等于1.0，执行以下操作
        {
            // 在 y 和 y+1 之间的源像素取平均值
            double src_y1 = (y + 0.5) * m_scale - 0.5; // 计算平均范围的下限
            double src_y2 = src_y1 + m_scale; // 计算平均范围的上限
            int h  = m_source.height() - 1; // 源图像高度减1
            int y1 = ifloor(src_y1); // 取下限
            int y2 = ifloor(src_y2); // 取上限
            rgba c = rgba::no_color(); // 初始化颜色对象
            if (y1 >= 0) c += rgba(m_source.pixel(x, y1)) *= y1 + 1 - src_y1; // 添加下限像素颜色的加权和
            while (++y1 < y2) // 遍历区间内的所有像素
            {
                if (y1 <= h) c += m_source.pixel(x, y1); // 添加每个像素的颜色
            }
            if (y2 <= h) c += rgba(m_source.pixel(x, y2)) *= src_y2 - y2; // 添加上限像素颜色的加权和
            return c *= m_scale_inv; // 返回加权和乘以缩放比例的倒数
        }
    }
    //======================================================line_image_pattern
    template<class Filter> class line_image_pattern
    {
    private:
        // 复制构造函数的私有化声明，防止对象的拷贝
        line_image_pattern(const line_image_pattern<filter_type>&);
        // 赋值运算符重载的私有化声明，防止对象的赋值
        const line_image_pattern<filter_type>& 
            operator = (const line_image_pattern<filter_type>&);
    
    protected:
        // 缓存行指针，存储颜色类型的缓存
        row_ptr_cache<color_type> m_buf;
        // 滤波器指针，指向滤波器对象
        const filter_type*        m_filter;
        // 膨胀值，用于图像模式处理的膨胀值设定
        unsigned                  m_dilation;
        // 高分辨率膨胀值，用于图像模式处理的高分辨率膨胀值设定
        int                       m_dilation_hr;
        // 数据存储，用于存储颜色类型数据的数组
        pod_array<color_type>     m_data;
        // 宽度值，图像模式的宽度
        unsigned                  m_width;
        // 高度值，图像模式的高度
        unsigned                  m_height;
        // 高分辨率宽度值，用于图像模式处理的高分辨率宽度设定
        int                       m_width_hr;
        // 高分辨率半高度值，用于图像模式处理的高分辨率半高度设定
        int                       m_half_height_hr;
        // 垂直偏移量值，用于图像模式处理的垂直偏移量设定
        int                       m_offset_y_hr;
    };
    
    //=================================================line_image_pattern_pow2
    template<class Filter> class line_image_pattern_pow2 : 
    public line_image_pattern<Filter>
    {
    public:
        typedef Filter filter_type;
        typedef typename filter_type::color_type color_type;
        typedef line_image_pattern<Filter> base_type;
    
        //--------------------------------------------------------------------
        // 构造函数，初始化基类和成员变量
        line_image_pattern_pow2(Filter& filter) :
            line_image_pattern<Filter>(filter), m_mask(line_subpixel_mask) {}
    
        //--------------------------------------------------------------------
        // 模板构造函数，初始化基类和成员变量，并根据源数据创建模式
        template<class Source> 
        line_image_pattern_pow2(Filter& filter, const Source& src) :
            line_image_pattern<Filter>(filter), m_mask(line_subpixel_mask)
        {
            create(src);
        }
            
        //--------------------------------------------------------------------
        // 根据源数据创建模式，计算掩码值和高分辨率宽度
        template<class Source> void create(const Source& src)
        {
            line_image_pattern<Filter>::create(src); // 调用基类方法创建模式
            m_mask = 1;
            // 计算掩码值，确保其大于等于模式宽度
            while(m_mask < base_type::m_width) 
            {
                m_mask <<= 1;
                m_mask |= 1;
            }
            // 将掩码左移半个子像素位，并设置为子像素掩码
            m_mask <<= line_subpixel_shift - 1;
            m_mask |=  line_subpixel_mask;
            // 设置高分辨率宽度为掩码值加一
            base_type::m_width_hr = m_mask + 1;
        }
    
        //--------------------------------------------------------------------
        // 获取像素值，调用基类滤波器对象的高分辨率像素函数
        void pixel(color_type* p, int x, int y) const
        {
            base_type::m_filter->pixel_high_res(
                    base_type::m_buf.rows(), 
                    p,
                    (x & m_mask) + base_type::m_dilation_hr,
                    y + base_type::m_offset_y_hr);
        }
    private:
        // 子像素掩码值，用于子像素位计算
        unsigned m_mask;
    };
    //===================================================distance_interpolator4
    class distance_interpolator4
    {
    private:
        //---------------------------------------------------------------------
        int m_dx;               // 增量 dx
        int m_dy;               // 增量 dy
        int m_dx_start;         // 开始时的增量 dx
        int m_dy_start;         // 开始时的增量 dy
        int m_dx_pict;          // 图片中间的增量 dx
        int m_dy_pict;          // 图片中间的增量 dy
        int m_dx_end;           // 结束时的增量 dx
        int m_dy_end;           // 结束时的增量 dy
    
        int m_dist;             // 距离
        int m_dist_start;       // 开始时的距离
        int m_dist_pict;        // 图片中间的距离
        int m_dist_end;         // 结束时的距离
        int m_len;              // 长度
    };
    
    
    
    
    
    //==================================================line_interpolator_image
    template<class Renderer> class line_interpolator_image
    {
    private:
        line_interpolator_image(const line_interpolator_image<Renderer>&);  // 私有拷贝构造函数
        const line_interpolator_image<Renderer>&
            operator = (const line_interpolator_image<Renderer>&);           // 私有赋值运算符重载
    
    protected:
        const line_parameters& m_lp;        // 线条参数的常量引用
        dda2_line_interpolator m_li;        // DDA2 线条插值器对象
        distance_interpolator4 m_di;        // 距离插值器对象
        renderer_type&         m_ren;       // 渲染器类型对象的引用
        int m_plen;                         // 线条长度
        int m_x;                            // 当前 x 坐标
        int m_y;                            // 当前 y 坐标
        int m_old_x;                        // 旧的 x 坐标
        int m_old_y;                        // 旧的 y 坐标
        int m_count;                        // 计数器
        int m_width;                        // 宽度
        int m_max_extent;                   // 最大扩展
        int m_start;                        // 起始位置
        int m_step;                         // 步长
        int m_dist_pos[max_half_width + 1]; // 距离位置数组
        color_type m_colors[max_half_width * 2 + 4];  // 颜色类型数组
    };
    
    
    
    
    
    //===================================================renderer_outline_image
    template<class BaseRenderer, class ImagePattern> 
    class renderer_outline_image
    {
    private:
        base_ren_type*      m_ren;      // 基础渲染器类型指针
        pattern_type* m_pattern;        // 图案类型指针
        int                 m_start;    // 起始位置
        double              m_scale_x;  // 缩放比例 x
        rect_i              m_clip_box; // 裁剪框矩形
        bool                m_clipping; // 是否裁剪
    };
}


注释：


// 关闭一个 #ifdef 或 #ifndef 预处理指令的块



#endif


注释：


// 结束一个 #ifdef 或 #ifndef 预处理指令块的条件部分
```