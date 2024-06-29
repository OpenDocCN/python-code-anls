# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_pixfmt_rgb_packed.h`

```
// 定义一个命名空间agg，用于封装Anti-Grain Geometry图形库的相关功能和结构
namespace agg
{
    //=========================================================blender_rgb555
    // 定义一个结构体blender_rgb555，用于处理RGB555像素格式的混合操作
    struct blender_rgb555
    {
        typedef rgba8 color_type;                    // 定义颜色类型为rgba8，即包含红、绿、蓝和透明度四个通道的颜色
        typedef color_type::value_type value_type;   // 像素值类型
        typedef color_type::calc_type calc_type;     // 计算类型
        typedef int16u pixel_type;                   // 像素类型为16位无符号整数

        // 混合像素操作，根据给定的颜色分量和透明度进行混合
        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            pixel_type rgb = *p;                    // 获取当前像素值
            calc_type r = (rgb >> 7) & 0xF8;        // 提取当前像素的红色分量并扩展到8位
            calc_type g = (rgb >> 2) & 0xF8;        // 提取当前像素的绿色分量并扩展到8位
            calc_type b = (rgb << 3) & 0xF8;        // 提取当前像素的蓝色分量并扩展到8位
            // 根据当前像素值的颜色分量和给定的目标颜色分量、透明度，计算新的像素值并进行混合
            *p = (pixel_type)
               (((((cr - r) * alpha + (r << 8)) >> 1)  & 0x7C00) |
                ((((cg - g) * alpha + (g << 8)) >> 6)  & 0x03E0) |
                 (((cb - b) * alpha + (b << 8)) >> 11) | 0x8000);
        }

        // 创建一个RGB555像素值，根据给定的红、绿、蓝分量
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 7) | 
                                ((g & 0xF8) << 2) | 
                                 (b >> 3) | 0x8000);
        }

        // 从RGB555像素值创建一个rgba8颜色对象
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 7) & 0xF8, 
                              (p >> 2) & 0xF8, 
                              (p << 3) & 0xF8);
        }
    };
    {
        // 定义颜色类型为 rgba8
        typedef rgba8 color_type;
        // 定义值类型为颜色类型的值类型
        typedef color_type::value_type value_type;
        // 定义计算类型为颜色类型的计算类型
        typedef color_type::calc_type calc_type;
        // 定义像素类型为 int16u
        typedef int16u pixel_type;
    
        // 定义静态内联函数，用于混合像素
        static AGG_INLINE void blend_pix(pixel_type* p,
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha,
                                         unsigned cover)
        {
            // 计算反向的 alpha 值
            alpha = color_type::base_mask - alpha;
            // 读取当前像素值
            pixel_type rgb = *p;
            // 提取并转换红色分量
            calc_type r = (rgb >> 7) & 0xF8;
            // 提取并转换绿色分量
            calc_type g = (rgb >> 2) & 0xF8;
            // 提取并转换蓝色分量
            calc_type b = (rgb << 3) & 0xF8;
            // 执行颜色混合并更新像素值
            *p = (pixel_type)
               ((((r * alpha + cr * cover) >> 1)  & 0x7C00) |
                (((g * alpha + cg * cover) >> 6)  & 0x03E0) |
                 ((b * alpha + cb * cover) >> 11) | 0x8000);
        }
    
        // 定义静态内联函数，用于创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 组合并返回像素值
            return (pixel_type)(((r & 0xF8) << 7) |
                                ((g & 0xF8) << 2) |
                                 (b >> 3) | 0x8000);
        }
    
        // 定义静态内联函数，用于从像素值创建颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 提取并返回颜色值
            return color_type((p >> 7) & 0xF8,
                              (p >> 2) & 0xF8,
                              (p << 3) & 0xF8);
        }
    };
    //=========================================================blender_rgb555_gamma
    struct blender_rgb555_gamma
    {
        // 定义颜色类型和计算类型
        typedef rgba8 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int16u pixel_type;
        typedef Gamma gamma_type;

        // 构造函数，初始化 gamma 为 0
        blender_rgb555_gamma() : m_gamma(0) {}

        // 设置 gamma 值的方法
        void gamma(const gamma_type& g) { m_gamma = &g; }

        // 混合像素的方法
        AGG_INLINE void blend_pix(pixel_type* p,
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha,
                                  unsigned)
        {
            // 读取像素值
            pixel_type rgb = *p;
            // 计算红色通道的混合值
            calc_type r = m_gamma->dir((rgb >> 7) & 0xF8);
            // 计算绿色通道的混合值
            calc_type g = m_gamma->dir((rgb >> 2) & 0xF8);
            // 计算蓝色通道的混合值
            calc_type b = m_gamma->dir((rgb << 3) & 0xF8);
            // 合成新的像素值
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 8)) >> 8) << 7) & 0x7C00) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 8)) >> 8) << 2) & 0x03E0) |
                 (m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 8)) >> 8) >> 3) | 0x8000);
        }

        // 创建像素值的静态方法
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 7) |
                                ((g & 0xF8) << 2) |
                                 (b >> 3) | 0x8000);
        }

        // 创建颜色类型的静态方法
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 7) & 0xF8,
                              (p >> 2) & 0xF8,
                              (p << 3) & 0xF8);
        }

    private:
        const Gamma* m_gamma;
    };





    //=========================================================blender_rgb565
    struct blender_rgb565
    {
        // 定义颜色类型和计算类型
        typedef rgba8 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int16u pixel_type;

        // 混合像素的方法
        static AGG_INLINE void blend_pix(pixel_type* p,
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha,
                                         unsigned)
        {
            // 读取像素值
            pixel_type rgb = *p;
            // 计算红色通道的混合值
            calc_type r = (rgb >> 8) & 0xF8;
            // 计算绿色通道的混合值
            calc_type g = (rgb >> 3) & 0xFC;
            // 计算蓝色通道的混合值
            calc_type b = (rgb << 3) & 0xF8;
            // 合成新的像素值
            *p = (pixel_type)
               (((((cr - r) * alpha + (r << 8))) & 0xF800) |
                ((((cg - g) * alpha + (g << 8)) >> 5) & 0x07E0) |
                 (((cb - b) * alpha + (b << 8)) >> 11));
        }

        // 创建像素值的静态方法
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
        }

        // 创建颜色类型的静态方法
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 8) & 0xF8,
                              (p >> 3) & 0xFC,
                              (p << 3) & 0xF8);
        }
    };
    //=====================================================blender_rgb565_pre

    // 定义一个结构体 blender_rgb565_pre，用于预处理 RGB565 格式的颜色
    struct blender_rgb565_pre
    {
        // 定义颜色类型为 rgba8
        typedef rgba8 color_type;
        // 值类型为颜色值类型的值类型
        typedef color_type::value_type value_type;
        // 计算类型为颜色值类型的计算类型
        typedef color_type::calc_type calc_type;
        // 像素类型为 16 位无符号整数
        typedef int16u pixel_type;

        // 内联函数，用于混合像素
        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            // 计算透明度的补码
            alpha = color_type::base_mask - alpha;
            // 获取像素值
            pixel_type rgb = *p;
            // 分离像素中的红色分量
            calc_type r = (rgb >> 8) & 0xF8;
            // 分离像素中的绿色分量
            calc_type g = (rgb >> 3) & 0xFC;
            // 分离像素中的蓝色分量
            calc_type b = (rgb << 3) & 0xF8;
            // 混合新的颜色值并更新像素
            *p = (pixel_type)
               ((((r * alpha + cr * cover)      ) & 0xF800) |
                (((g * alpha + cg * cover) >> 5 ) & 0x07E0) |
                 ((b * alpha + cb * cover) >> 11));
        }

        // 内联函数，创建一个 RGB565 格式的像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
        }

        // 内联函数，从像素值创建颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 8) & 0xF8, 
                              (p >> 3) & 0xFC, 
                              (p << 3) & 0xF8);
        }
    };

    //=====================================================blender_rgb565_gamma

    // 模板类 blender_rgb565_gamma，用于处理带 gamma 校正的 RGB565 颜色混合
    template<class Gamma> class blender_rgb565_gamma
    {
    # 公共部分开始，定义了几种类型别名和成员变量
    public:
        # 定义了颜色类型为 rgba8
        typedef rgba8 color_type;
        # 颜色类型的数值类型
        typedef color_type::value_type value_type;
        # 颜色类型的计算类型
        typedef color_type::calc_type calc_type;
        # 像素类型为 int16u
        typedef int16u pixel_type;
        # Gamma 类型的别名
        typedef Gamma gamma_type;
    
        # 默认构造函数，初始化 Gamma 指针为 nullptr
        blender_rgb565_gamma() : m_gamma(0) {}
        
        # 设置 Gamma 值的函数
        void gamma(const gamma_type& g) { m_gamma = &g; }
    
        # 内联函数，用于像素混合操作
        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            # 读取原像素值
            pixel_type rgb = *p;
            # 对红色通道进行 Gamma 校正
            calc_type r = m_gamma->dir((rgb >> 8) & 0xF8);
            # 对绿色通道进行 Gamma 校正
            calc_type g = m_gamma->dir((rgb >> 3) & 0xFC);
            # 对蓝色通道进行 Gamma 校正
            calc_type b = m_gamma->dir((rgb << 3) & 0xF8);
            # 计算混合后的新像素值
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 8)) >> 8) << 8) & 0xF800) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 8)) >> 8) << 3) & 0x07E0) |
                 (m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 8)) >> 8) >> 3));
        }
    
        # 静态内联函数，创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
        }
    
        # 静态内联函数，从像素值创建颜色对象
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 8) & 0xF8, 
                              (p >> 3) & 0xFC, 
                              (p << 3) & 0xF8);
        }
    
    # 私有部分开始，定义了 Gamma 指针成员变量
    private:
        # 指向 Gamma 类型的常量指针
        const Gamma* m_gamma;
    };
    
    
    这段代码定义了一个名为 `blender_rgb565_gamma` 的结构体，实现了一些颜色混合和像素操作的函数。
    {
        // 定义颜色类型为 rgba16
        typedef rgba16 color_type;
        // 定义值类型为颜色类型的值类型
        typedef color_type::value_type value_type;
        // 定义计算类型为颜色类型的计算类型
        typedef color_type::calc_type calc_type;
        // 定义像素类型为 32 位无符号整数
        typedef int32u pixel_type;
    
        // 内联函数：混合像素值
        static AGG_INLINE void blend_pix(pixel_type* p,
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha,
                                         unsigned cover)
        {
            // 计算透明度的反向值
            alpha = color_type::base_mask - alpha;
            // 计算覆盖率加权值
            cover = (cover + 1) << (color_type::base_shift - 8);
            // 提取当前像素的 RGB 值
            pixel_type rgb = *p;
            // 提取当前像素的红色分量并扩展到高位
            calc_type r = (rgb >> 14) & 0xFFC0;
            // 提取当前像素的绿色分量并扩展到高位
            calc_type g = (rgb >> 4)  & 0xFFC0;
            // 提取当前像素的蓝色分量并扩展到高位
            calc_type b = (rgb << 6)  & 0xFFC0;
            // 计算混合后的像素值并存储回原像素
            *p = (pixel_type)
               ((((r * alpha + cr * cover) >> 2)  & 0x3FF00000) |
                (((g * alpha + cg * cover) >> 12) & 0x000FFC00) |
                 ((b * alpha + cb * cover) >> 22) | 0xC0000000);
        }
    
        // 内联函数：生成像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 构造并返回像素值
            return (pixel_type)(((r & 0xFFC0) << 14) |
                                ((g & 0xFFC0) << 4) |
                                 (b >> 6) | 0xC0000000);
        }
    
        // 内联函数：从像素值生成颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 根据像素值提取红、绿、蓝分量并生成颜色类型对象
            return color_type((p >> 14) & 0xFFC0,
                              (p >> 4)  & 0xFFC0,
                              (p << 6)  & 0xFFC0);
        }
    };
    
    //==================================================blender_rgbAAA_pre
    // 结构体：预乘混合器，RGBAAA
    struct blender_rgbAAA_pre
    {
        // 定义颜色类型为 rgba16
        typedef rgba16 color_type;
        // 定义值类型为颜色类型的值类型
        typedef color_type::value_type value_type;
        // 定义计算类型为颜色类型的计算类型
        typedef color_type::calc_type calc_type;
        // 定义像素类型为 32 位无符号整数
        typedef int32u pixel_type;
    
        // 内联函数：混合像素值
        static AGG_INLINE void blend_pix(pixel_type* p,
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha,
                                         unsigned cover)
        {
            // 计算透明度的反向值
            alpha = color_type::base_mask - alpha;
            // 计算覆盖率加权值
            cover = (cover + 1) << (color_type::base_shift - 8);
            // 提取当前像素的 RGB 值
            pixel_type rgb = *p;
            // 提取当前像素的红色分量并扩展到高位
            calc_type r = (rgb >> 14) & 0xFFC0;
            // 提取当前像素的绿色分量并扩展到高位
            calc_type g = (rgb >> 4)  & 0xFFC0;
            // 提取当前像素的蓝色分量并扩展到高位
            calc_type b = (rgb << 6)  & 0xFFC0;
            // 计算混合后的像素值并存储回原像素
            *p = (pixel_type)
               ((((r * alpha + cr * cover) >> 2)  & 0x3FF00000) |
                (((g * alpha + cg * cover) >> 12) & 0x000FFC00) |
                 ((b * alpha + cb * cover) >> 22) | 0xC0000000);
        }
    
        // 内联函数：生成像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 构造并返回像素值
            return (pixel_type)(((r & 0xFFC0) << 14) |
                                ((g & 0xFFC0) << 4) |
                                 (b >> 6) | 0xC0000000);
        }
    
        // 内联函数：从像素值生成颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 根据像素值提取红、绿、蓝分量并生成颜色类型对象
            return color_type((p >> 14) & 0xFFC0,
                              (p >> 4)  & 0xFFC0,
                              (p << 6)  & 0xFFC0);
        }
    };
    //=================================================blender_rgbAAA_gamma
    template<class Gamma> class blender_rgbAAA_gamma
    {
    public:
        typedef rgba16 color_type;  // 定义颜色类型为 rgba16
        typedef color_type::value_type value_type;  // 定义值类型为 color_type 的值类型
        typedef color_type::calc_type calc_type;  // 定义计算类型为 color_type 的计算类型
        typedef int32u pixel_type;  // 定义像素类型为 int32u
        typedef Gamma gamma_type;  // 定义伽马类型为 Gamma
    
        blender_rgbAAA_gamma() : m_gamma(0) {}  // 构造函数，默认初始化 m_gamma 为 0
        void gamma(const gamma_type& g) { m_gamma = &g; }  // 设置伽马校正函数
    
        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            pixel_type rgb = *p;  // 获取当前像素的值
            calc_type r = m_gamma->dir((rgb >> 14) & 0xFFC0);  // 对红色通道进行伽马校正
            calc_type g = m_gamma->dir((rgb >> 4)  & 0xFFC0);  // 对绿色通道进行伽马校正
            calc_type b = m_gamma->dir((rgb << 6)  & 0xFFC0);  // 对蓝色通道进行伽马校正
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 16)) >> 16) << 14) & 0x3FF00000) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 16)) >> 16) << 4 ) & 0x000FFC00) |
                 (m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 16)) >> 16) >> 6 ) | 0xC0000000);
            // 混合像素，对每个通道进行伽马校正和混合，最后重新组合为新的像素值
        }
    
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (b >> 6) | 0xC0000000);
            // 创建像素值，将 RGB 值组合为一个像素
        }
    
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 14) & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p << 6)  & 0xFFC0);
            // 根据像素值创建颜色
        }
    private:
        const Gamma* m_gamma;  // 伽马校正对象的指针
    };
    {
        // 定义 RGBA16 格式的颜色类型
        typedef rgba16 color_type;
        // 定义颜色值的数据类型
        typedef color_type::value_type value_type;
        // 定义计算过程中使用的数据类型
        typedef color_type::calc_type calc_type;
        // 定义像素数据的类型
        typedef int32u pixel_type;
    
        // 内联函数：混合像素颜色值
        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            // 从像素值中提取当前的 B、G、R 成分
            pixel_type bgr = *p;
            calc_type b = (bgr >> 14) & 0xFFC0;
            calc_type g = (bgr >> 4)  & 0xFFC0;
            calc_type r = (bgr << 6)  & 0xFFC0;
    
            // 计算新的混合后的像素值并存储回原位置
            *p = (pixel_type)
               (((((cb - b) * alpha + (b << 16)) >> 2)  & 0x3FF00000) |
                ((((cg - g) * alpha + (g << 16)) >> 12) & 0x000FFC00) |
                 (((cr - r) * alpha + (r << 16)) >> 22) | 0xC0000000);
        }
    
        // 内联函数：创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (r >> 6) | 0xC0000000);
        }
    
        // 内联函数：从像素值创建颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 6)  & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p >> 14) & 0xFFC0);
        }
    };
    
    //=================================================blender_bgrAAA_pre
    // 结构体：预乘 alpha 之后的 BGRAAA 混合器
    struct blender_bgrAAA_pre
    {
        // 定义 RGBA16 格式的颜色类型
        typedef rgba16 color_type;
        // 定义颜色值的数据类型
        typedef color_type::value_type value_type;
        // 定义计算过程中使用的数据类型
        typedef color_type::calc_type calc_type;
        // 定义像素数据的类型
        typedef int32u pixel_type;
    
        // 内联函数：混合像素颜色值（预乘 alpha 版本）
        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            // 计算 alpha 的补码
            alpha = color_type::base_mask - alpha;
            // 计算覆盖值（cover）的调整值
            cover = (cover + 1) << (color_type::base_shift - 8);
    
            // 从像素值中提取当前的 B、G、R 成分
            pixel_type bgr = *p;
            calc_type b = (bgr >> 14) & 0xFFC0;
            calc_type g = (bgr >> 4)  & 0xFFC0;
            calc_type r = (bgr << 6)  & 0xFFC0;
    
            // 计算新的混合后的像素值并存储回原位置
            *p = (pixel_type)
               ((((b * alpha + cb * cover) >> 2)  & 0x3FF00000) |
                (((g * alpha + cg * cover) >> 12) & 0x000FFC00) |
                 ((r * alpha + cr * cover) >> 22) | 0xC0000000);
        }
    
        // 内联函数：创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (r >> 6) | 0xC0000000);
        }
    
        // 内联函数：从像素值创建颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 6)  & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p >> 14) & 0xFFC0);
        }
    };
    //=================================================blender_bgrAAA_gamma
    // 定义一个模板类 blender_bgrAAA_gamma，用于颜色混合操作，支持 gamma 校正
    template<class Gamma> class blender_bgrAAA_gamma
    {
    public:
        typedef rgba16 color_type;            // 定义颜色类型为 rgba16
        typedef color_type::value_type value_type;  // 颜色值类型
        typedef color_type::calc_type calc_type;    // 计算类型
        typedef int32u pixel_type;            // 像素类型
        typedef Gamma gamma_type;             // gamma 类型
    
        blender_bgrAAA_gamma() : m_gamma(0) {}    // 构造函数，初始化 gamma 为 0
        void gamma(const gamma_type& g) { m_gamma = &g; }  // 设置 gamma 校正对象的方法
    
        // 内联函数，用于像素混合操作，支持 alpha 通道
        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            pixel_type bgr = *p;    // 获取像素值
            calc_type b = m_gamma->dir((bgr >> 14) & 0xFFC0);   // 对蓝色通道进行 gamma 校正
            calc_type g = m_gamma->dir((bgr >> 4)  & 0xFFC0);   // 对绿色通道进行 gamma 校正
            calc_type r = m_gamma->dir((bgr << 6)  & 0xFFC0);   // 对红色通道进行 gamma 校正
            // 计算混合后的像素值，考虑 alpha 通道和 gamma 校正的影响
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 16)) >> 16) << 14) & 0x3FF00000) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 16)) >> 16) << 4 ) & 0x000FFC00) |
                 (m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 16)) >> 16) >> 6 ) | 0xC0000000);
        }
    
        // 静态内联函数，用于创建像素值，支持 gamma 校正
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (r >> 6) | 0xC0000000);
        }
    
        // 静态内联函数，将像素值转换为颜色类型，支持 gamma 校正
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 6)  & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p >> 14) & 0xFFC0);
        }
    
    private:
        const Gamma* m_gamma;   // 私有成员变量，指向 gamma 校正对象的指针
    };
    
    
    
    //=====================================================blender_rgbBBA
    // 定义一个结构体 blender_rgbBBA
    struct blender_rgbBBA
    {
        // 定义 rgba16 类型的别名
        typedef rgba16 color_type;
        // 从 color_type 中提取 value_type 类型的别名
        typedef color_type::value_type value_type;
        // 从 color_type 中提取 calc_type 类型的别名
        typedef color_type::calc_type calc_type;
        // 定义 int32u 类型的像素类型别名
        typedef int32u pixel_type;
    
        // 定义静态内联函数 blend_pix，用于混合像素
        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            // 获取当前像素的 RGB 值
            pixel_type rgb = *p;
            // 提取当前像素的红色分量，并进行处理
            calc_type r = (rgb >> 16) & 0xFFE0;
            // 提取当前像素的绿色分量，并进行处理
            calc_type g = (rgb >> 5)  & 0xFFE0;
            // 提取当前像素的蓝色分量，并进行处理
            calc_type b = (rgb << 6)  & 0xFFC0;
            // 计算混合后的像素值，并存入当前像素的指针位置
            *p = (pixel_type)
               (((((cr - r) * alpha + (r << 16))      ) & 0xFFE00000) |
                ((((cg - g) * alpha + (g << 16)) >> 11) & 0x001FFC00) |
                 (((cb - b) * alpha + (b << 16)) >> 22));
        }
    
        // 定义静态内联函数 make_pix，用于创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 将给定的 RGB 分量合成为一个像素值并返回
            return (pixel_type)(((r & 0xFFE0) << 16) | ((g & 0xFFE0) << 5) | (b >> 6));
        }
    
        // 定义静态内联函数 make_color，用于创建颜色对象
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 从给定的像素值中提取红、绿、蓝分量并返回对应的颜色对象
            return color_type((p >> 16) & 0xFFE0, 
                              (p >> 5)  & 0xFFE0, 
                              (p << 6)  & 0xFFC0);
        }
    };
    
    
    //=================================================blender_rgbBBA_pre
    // 定义结构体 blender_rgbBBA_pre，用于 RGB 带透明度预乘的混合
    struct blender_rgbBBA_pre
    {
        // 定义 rgba16 类型的别名
        typedef rgba16 color_type;
        // 从 color_type 中提取 value_type 类型的别名
        typedef color_type::value_type value_type;
        // 从 color_type 中提取 calc_type 类型的别名
        typedef color_type::calc_type calc_type;
        // 定义 int32u 类型的像素类型别名
        typedef int32u pixel_type;
    
        // 定义静态内联函数 blend_pix，用于进行带透明度预乘的像素混合
        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            // 计算透明度的补码值
            alpha = color_type::base_mask - alpha;
            // 计算覆盖度的增量，并调整为适合像素格式的值
            cover = (cover + 1) << (color_type::base_shift - 8);
            // 获取当前像素的 RGB 值
            pixel_type rgb = *p;
            // 提取当前像素的红色分量，并进行处理
            calc_type r = (rgb >> 16) & 0xFFE0;
            // 提取当前像素的绿色分量，并进行处理
            calc_type g = (rgb >> 5)  & 0xFFE0;
            // 提取当前像素的蓝色分量，并进行处理
            calc_type b = (rgb << 6)  & 0xFFC0;
            // 计算混合后的像素值，并存入当前像素的指针位置
            *p = (pixel_type)
               ((((r * alpha + cr * cover)      ) & 0xFFE00000) |
                (((g * alpha + cg * cover) >> 11) & 0x001FFC00) |
                 ((b * alpha + cb * cover) >> 22));
        }
    
        // 定义静态内联函数 make_pix，用于创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 将给定的 RGB 分量合成为一个像素值并返回
            return (pixel_type)(((r & 0xFFE0) << 16) | ((g & 0xFFE0) << 5) | (b >> 6));
        }
    
        // 定义静态内联函数 make_color，用于创建颜色对象
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 从给定的像素值中提取红、绿、蓝分量并返回对应的颜色对象
            return color_type((p >> 16) & 0xFFE0, 
                              (p >> 5)  & 0xFFE0, 
                              (p << 6)  & 0xFFC0);
        }
    };
    //=====================================================blender_rgbBBA_gamma
    // 定义一个结构体 blender_rgbBBA_gamma，用于颜色混合操作，支持 gamma 校正
    struct blender_rgbBBA_gamma
    {
        // 定义颜色类型为 rgba16
        typedef rgba16 color_type;
        // 定义值类型为 color_type 的值类型
        typedef color_type::value_type value_type;
        // 定义计算类型为 color_type 的计算类型
        typedef color_type::calc_type calc_type;
        // 定义像素类型为 32 位无符号整数
        typedef int32u pixel_type;
        // 定义 gamma 类型为 Gamma
        typedef Gamma gamma_type;

        // 构造函数，初始化 gamma 为 0
        blender_rgbBBA_gamma() : m_gamma(0) {}
        
        // 设置 gamma 值的方法
        void gamma(const gamma_type& g) { m_gamma = &g; }

        // 内联函数，用于混合像素
        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            // 读取当前像素值
            pixel_type rgb = *p;
            // 根据 gamma 校正后的值计算红色分量 r
            calc_type r = m_gamma->dir((rgb >> 16) & 0xFFE0);
            // 根据 gamma 校正后的值计算绿色分量 g
            calc_type g = m_gamma->dir((rgb >> 5)  & 0xFFE0);
            // 根据 gamma 校正后的值计算蓝色分量 b
            calc_type b = m_gamma->dir((rgb << 6)  & 0xFFC0);
            // 合成新的像素值，并进行 gamma 逆校正
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 16)) >> 16) << 16) & 0xFFE00000) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 16)) >> 16) << 5 ) & 0x001FFC00) |
                 (m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 16)) >> 16) >> 6 ));
        }

        // 静态内联函数，创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 组装像素值，注意颜色分量的位移和掩码
            return (pixel_type)(((r & 0xFFE0) << 16) | ((g & 0xFFE0) << 5) | (b >> 6));
        }

        // 静态内联函数，根据像素值创建颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 根据像素值反向解析出颜色分量并返回
            return color_type((p >> 16) & 0xFFE0, 
                              (p >> 5)  & 0xFFE0, 
                              (p << 6)  & 0xFFC0);
        }

    private:
        const Gamma* m_gamma;
    };


    //=====================================================blender_bgrABB
    // 定义一个结构体 blender_bgrABB，用于 BGR 颜色混合操作
    struct blender_bgrABB
    {
        // 定义颜色类型为 rgba16
        typedef rgba16 color_type;
        // 定义值类型为 color_type 的值类型
        typedef color_type::value_type value_type;
        // 定义计算类型为 color_type 的计算类型
        typedef color_type::calc_type calc_type;
        // 定义像素类型为 32 位无符号整数
        typedef int32u pixel_type;

        // 静态内联函数，用于 BGR 混合像素
        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            // 读取当前像素值
            pixel_type bgr = *p;
            // 提取当前像素值中的蓝色分量 b
            calc_type b = (bgr >> 16) & 0xFFC0;
            // 提取当前像素值中的绿色分量 g
            calc_type g = (bgr >> 6)  & 0xFFE0;
            // 提取当前像素值中的红色分量 r
            calc_type r = (bgr << 5)  & 0xFFE0;
            // 合成新的像素值
            *p = (pixel_type)
               (((((cb - b) * alpha + (b << 16))      ) & 0xFFC00000) |
                ((((cg - g) * alpha + (g << 16)) >> 10) & 0x003FF800) |
                 (((cr - r) * alpha + (r << 16)) >> 21));
        }

        // 静态内联函数，根据颜色分量创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 组装像素值，注意颜色分量的位移和掩码
            return (pixel_type)(((b & 0xFFC0) << 16) | ((g & 0xFFE0) << 6) | (r >> 5));
        }

        // 静态内联函数，根据像素值创建颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 根据像素值解析出颜色分量并返回
            return color_type((p << 5)  & 0xFFE0,
                              (p >> 6)  & 0xFFE0, 
                              (p >> 16) & 0xFFC0);
        }
    };
    //=================================================blender_bgrABB_pre
    // 定义一个结构体 blender_bgrABB_pre，用于处理像素的颜色混合和转换
    struct blender_bgrABB_pre
    {
        // 声明颜色类型为 rgba16
        typedef rgba16 color_type;
        // 定义值类型为颜色值类型的值类型
        typedef color_type::value_type value_type;
        // 定义计算类型为颜色类型的计算类型
        typedef color_type::calc_type calc_type;
        // 定义像素类型为 int32u
        typedef int32u pixel_type;

        // 声明静态内联函数，用于混合像素
        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            // 计算透明度的反向值
            alpha = color_type::base_mask - alpha;
            // 计算覆盖度的偏移和缩放
            cover = (cover + 1) << (color_type::base_shift - 8);
            // 获取当前像素的 BGR 值
            pixel_type bgr = *p;
            // 分别提取并计算 B、G、R 值
            calc_type b = (bgr >> 16) & 0xFFC0;
            calc_type g = (bgr >> 6)  & 0xFFE0;
            calc_type r = (bgr << 5)  & 0xFFE0;
            // 混合并更新像素的 BGR 值
            *p = (pixel_type)
               ((((b * alpha + cb * cover)      ) & 0xFFC00000) |
                (((g * alpha + cg * cover) >> 10) & 0x003FF800) |
                 ((r * alpha + cr * cover) >> 21));
        }

        // 声明静态内联函数，创建像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 组装并返回像素值
            return (pixel_type)(((b & 0xFFC0) << 16) | ((g & 0xFFE0) << 6) | (r >> 5));
        }

        // 声明静态内联函数，将像素值转换为颜色类型
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 从像素值提取并返回颜色类型
            return color_type((p << 5)  & 0xFFE0,
                              (p >> 6)  & 0xFFE0, 
                              (p >> 16) & 0xFFC0);
        }
    };



    //=================================================blender_bgrABB_gamma
    // 定义模板类 blender_bgrABB_gamma，基于 Gamma 进行颜色转换
    template<class Gamma> class blender_bgrABB_gamma
    {
    public:
        // 声明颜色类型为 rgba16
        typedef rgba16 color_type;
        // 定义值类型为 color_type 的 value_type
        typedef color_type::value_type value_type;
        // 定义计算类型为 color_type 的 calc_type
        typedef color_type::calc_type calc_type;
        // 声明像素类型为 int32u
        typedef int32u pixel_type;
        // 声明 gamma 类型为 Gamma
        typedef Gamma gamma_type;

        // 默认构造函数，初始化 m_gamma 为 nullptr
        blender_bgrABB_gamma() : m_gamma(0) {}
        
        // 设置 gamma 方法，接受 gamma 类型的引用参数 g
        void gamma(const gamma_type& g) { m_gamma = &g; }

        // 内联函数，用于混合像素
        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            // 保存当前像素值到 bgr
            pixel_type bgr = *p;
            // 通过 gamma 对象处理 bgr 中的蓝色分量
            calc_type b = m_gamma->dir((bgr >> 16) & 0xFFC0);
            // 通过 gamma 对象处理 bgr 中的绿色分量
            calc_type g = m_gamma->dir((bgr >> 6)  & 0xFFE0);
            // 通过 gamma 对象处理 bgr 中的红色分量
            calc_type r = m_gamma->dir((bgr << 5)  & 0xFFE0);
            // 按照混合公式重新计算像素值，并应用 gamma 逆处理
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 16)) >> 16) << 16) & 0xFFC00000) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 16)) >> 16) << 6 ) & 0x003FF800) |
                 (m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 16)) >> 16) >> 5 ));
        }

        // 静态内联函数，生成像素值
        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            // 根据给定的 RGB 分量生成像素值
            return (pixel_type)(((b & 0xFFC0) << 16) | ((g & 0xFFE0) << 6) | (r >> 5));
        }

        // 静态内联函数，从像素值创建颜色
        static AGG_INLINE color_type make_color(pixel_type p)
        {
            // 从给定的像素值提取并创建颜色对象
            return color_type((p << 5)  & 0xFFE0,
                              (p >> 6)  & 0xFFE0, 
                              (p >> 16) & 0xFFC0);
        }

    private:
        // 指向 Gamma 对象的常量指针
        const Gamma* m_gamma;
    };


    
    //===========================================pixfmt_alpha_blend_rgb_packed
    // 像素格式模板，实现 alpha 混合和 RGB 打包
    template<class Blender,  class RenBuf> class pixfmt_alpha_blend_rgb_packed
    {
    public:
        // 定义渲染缓冲类型为 RenBuf
        typedef RenBuf   rbuf_type;
        // 定义行数据类型为 rbuf_type 的 row_data
        typedef typename rbuf_type::row_data row_data;
        // 定义混合器类型为 Blender
        typedef Blender  blender_type;
        // 定义颜色类型为 blender_type 的 color_type
        typedef typename blender_type::color_type color_type;
        // 定义像素类型为 blender_type 的 pixel_type
        typedef typename blender_type::pixel_type pixel_type;
        // 声明 order_type 为整数类型，仅作为占位符使用
        typedef int                               order_type; // A fake one
        // 定义值类型为 color_type 的 value_type
        typedef typename color_type::value_type   value_type;
        // 定义计算类型为 color_type 的 calc_type
        typedef typename color_type::calc_type    calc_type;
        // 枚举，基本比例参数
        enum base_scale_e 
        {
            // 颜色类型的位移量
            base_shift = color_type::base_shift,
            // 颜色类型的基本比例
            base_scale = color_type::base_scale,
            // 颜色类型的掩码
            base_mask  = color_type::base_mask,
            // 像素类型的字节宽度
            pix_width  = sizeof(pixel_type),
        };
    private:
        //--------------------------------------------------------------------
        // 像素复制或混合函数，根据给定的颜色和覆盖度处理像素
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            // 如果颜色的 alpha 值不为零
            if (c.a)
            {
                // 计算实际使用的 alpha 值，基于颜色的 alpha 和覆盖度
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                // 如果计算得到的 alpha 等于基础掩码（base_mask）
                if(alpha == base_mask)
                {
                    // 直接使用颜色创建像素值
                    *p = m_blender.make_pix(c.r, c.g, c.b);
                }
                else
                {
                    // 使用混合器混合颜色到像素值
                    m_blender.blend_pix(p, c.r, c.g, c.b, alpha, cover);
                }
            }
        }

    private:
        // 渲染缓冲区指针
        rbuf_type* m_rbuf;
        // 混合器对象
        Blender    m_blender;
    };

    // 声明各种像素格式的别名模板，使用不同的混合器和渲染缓冲区类型
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgb555, rendering_buffer> pixfmt_rgb555; //----pixfmt_rgb555
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgb565, rendering_buffer> pixfmt_rgb565; //----pixfmt_rgb565

    typedef pixfmt_alpha_blend_rgb_packed<blender_rgb555_pre, rendering_buffer> pixfmt_rgb555_pre; //----pixfmt_rgb555_pre
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgb565_pre, rendering_buffer> pixfmt_rgb565_pre; //----pixfmt_rgb565_pre

    typedef pixfmt_alpha_blend_rgb_packed<blender_rgbAAA, rendering_buffer> pixfmt_rgbAAA; //----pixfmt_rgbAAA
    typedef pixfmt_alpha_blend_rgb_packed<blender_bgrAAA, rendering_buffer> pixfmt_bgrAAA; //----pixfmt_bgrAAA
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgbBBA, rendering_buffer> pixfmt_rgbBBA; //----pixfmt_rgbBBA
    typedef pixfmt_alpha_blend_rgb_packed<blender_bgrABB, rendering_buffer> pixfmt_bgrABB; //----pixfmt_bgrABB

    typedef pixfmt_alpha_blend_rgb_packed<blender_rgbAAA_pre, rendering_buffer> pixfmt_rgbAAA_pre; //----pixfmt_rgbAAA_pre
    typedef pixfmt_alpha_blend_rgb_packed<blender_bgrAAA_pre, rendering_buffer> pixfmt_bgrAAA_pre; //----pixfmt_bgrAAA_pre
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgbBBA_pre, rendering_buffer> pixfmt_rgbBBA_pre; //----pixfmt_rgbBBA_pre
    typedef pixfmt_alpha_blend_rgb_packed<blender_bgrABB_pre, rendering_buffer> pixfmt_bgrABB_pre; //----pixfmt_bgrABB_pre


    //-----------------------------------------------------pixfmt_rgb555_gamma
    // 像素格式模板，带有 Gamma 矫正的 RGB555 格式
    template<class Gamma> class pixfmt_rgb555_gamma : 
    public pixfmt_alpha_blend_rgb_packed<blender_rgb555_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        // 构造函数，初始化时设置 Gamma 矫正
        pixfmt_rgb555_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb_packed<blender_rgb555_gamma<Gamma>, 
                                          rendering_buffer>(rb) 
        {
            this->blender().gamma(g);
        }
    };


    //-----------------------------------------------------pixfmt_rgb565_gamma
    // 像素格式模板，带有 Gamma 矫正的 RGB565 格式
    template<class Gamma> class pixfmt_rgb565_gamma : 
    public pixfmt_alpha_blend_rgb_packed<blender_rgb565_gamma<Gamma>, rendering_buffer>
    {
    // 定义一个模板类 pixfmt_rgb565_gamma，继承自 pixfmt_alpha_blend_rgb_packed，用于处理 RGB565 格式的像素数据并应用 Gamma 校正
    template<class Gamma> class pixfmt_rgb565_gamma :
    public pixfmt_alpha_blend_rgb_packed<blender_rgb565_gamma<Gamma>, rendering_buffer>
    {
    public:
        // 构造函数，接受一个渲染缓冲区和一个 Gamma 校正对象作为参数
        pixfmt_rgb565_gamma(rendering_buffer& rb, const Gamma& g) :
            // 调用基类构造函数，传入渲染缓冲区，设置像素格式为 RGB565，同时指定使用 Gamma 校正
            pixfmt_alpha_blend_rgb_packed<blender_rgb565_gamma<Gamma>, rendering_buffer>(rb)
        {
            // 设置当前对象的混合器（blender）的 Gamma 校正参数
            this->blender().gamma(g);
        }
    };
    
    
    //-----------------------------------------------------pixfmt_rgbAAA_gamma
    // 定义一个模板类 pixfmt_rgbAAA_gamma，继承自 pixfmt_alpha_blend_rgb_packed，用于处理 RGBAAA 格式的像素数据并应用 Gamma 校正
    template<class Gamma> class pixfmt_rgbAAA_gamma :
    public pixfmt_alpha_blend_rgb_packed<blender_rgbAAA_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        // 构造函数，接受一个渲染缓冲区和一个 Gamma 校正对象作为参数
        pixfmt_rgbAAA_gamma(rendering_buffer& rb, const Gamma& g) :
            // 调用基类构造函数，传入渲染缓冲区，设置像素格式为 RGBAAA，同时指定使用 Gamma 校正
            pixfmt_alpha_blend_rgb_packed<blender_rgbAAA_gamma<Gamma>, 
                                          rendering_buffer>(rb)
        {
            // 设置当前对象的混合器（blender）的 Gamma 校正参数
            this->blender().gamma(g);
        }
    };
    
    
    //-----------------------------------------------------pixfmt_bgrAAA_gamma
    // 定义一个模板类 pixfmt_bgrAAA_gamma，继承自 pixfmt_alpha_blend_rgb_packed，用于处理 BGRAAA 格式的像素数据并应用 Gamma 校正
    template<class Gamma> class pixfmt_bgrAAA_gamma :
    public pixfmt_alpha_blend_rgb_packed<blender_bgrAAA_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        // 构造函数，接受一个渲染缓冲区和一个 Gamma 校正对象作为参数
        pixfmt_bgrAAA_gamma(rendering_buffer& rb, const Gamma& g) :
            // 调用基类构造函数，传入渲染缓冲区，设置像素格式为 BGRAAA，同时指定使用 Gamma 校正
            pixfmt_alpha_blend_rgb_packed<blender_bgrAAA_gamma<Gamma>, 
                                          rendering_buffer>(rb)
        {
            // 设置当前对象的混合器（blender）的 Gamma 校正参数
            this->blender().gamma(g);
        }
    };
    
    
    //-----------------------------------------------------pixfmt_rgbBBA_gamma
    // 定义一个模板类 pixfmt_rgbBBA_gamma，继承自 pixfmt_alpha_blend_rgb_packed，用于处理 RGBBBA 格式的像素数据并应用 Gamma 校正
    template<class Gamma> class pixfmt_rgbBBA_gamma :
    public pixfmt_alpha_blend_rgb_packed<blender_rgbBBA_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        // 构造函数，接受一个渲染缓冲区和一个 Gamma 校正对象作为参数
        pixfmt_rgbBBA_gamma(rendering_buffer& rb, const Gamma& g) :
            // 调用基类构造函数，传入渲染缓冲区，设置像素格式为 RGBBBA，同时指定使用 Gamma 校正
            pixfmt_alpha_blend_rgb_packed<blender_rgbBBA_gamma<Gamma>, 
                                          rendering_buffer>(rb)
        {
            // 设置当前对象的混合器（blender）的 Gamma 校正参数
            this->blender().gamma(g);
        }
    };
    
    
    //-----------------------------------------------------pixfmt_bgrABB_gamma
    // 定义一个模板类 pixfmt_bgrABB_gamma，继承自 pixfmt_alpha_blend_rgb_packed，用于处理 BGRABB 格式的像素数据并应用 Gamma 校正
    template<class Gamma> class pixfmt_bgrABB_gamma :
    public pixfmt_alpha_blend_rgb_packed<blender_bgrABB_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        // 构造函数，接受一个渲染缓冲区和一个 Gamma 校正对象作为参数
        pixfmt_bgrABB_gamma(rendering_buffer& rb, const Gamma& g) :
            // 调用基类构造函数，传入渲染缓冲区，设置像素格式为 BGRABB，同时指定使用 Gamma 校正
            pixfmt_alpha_blend_rgb_packed<blender_bgrABB_gamma<Gamma>, 
                                          rendering_buffer>(rb)
        {
            // 设置当前对象的混合器（blender）的 Gamma 校正参数
            this->blender().gamma(g);
        }
    };
}


注释：


// 这是 C/C++ 中的预处理器指令，用于结束一个条件编译区块



#endif


注释：


// 这是 C/C++ 中的预处理器指令，用于结束一个条件编译指令块的控制，与 #ifdef 或 #if 相对应
```