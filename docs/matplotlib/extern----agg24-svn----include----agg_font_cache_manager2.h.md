# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_font_cache_manager2.h`

```py
// 定义一个命名空间agg，用于组织AGG库的相关代码
namespace agg {

  // 声明一个命名空间fman，用于管理字体相关的数据结构和功能
  namespace fman {

    // 枚举类型，表示字形数据的类型
    enum glyph_data_type
    {
      glyph_data_invalid = 0,   // 无效数据
      glyph_data_mono    = 1,   // 单色数据
      glyph_data_gray8   = 2,   // 灰度8位数据
      glyph_data_outline = 3    // 轮廓数据
    };


    // 表示缓存中的单个字形数据结构
    struct cached_glyph
    {
      void *            cached_font;    // 缓存的字体对象指针
      unsigned        glyph_code;     // 字形代码
      unsigned        glyph_index;    // 字形索引
      int8u*          data;           // 字形数据指针
      unsigned        data_size;      // 数据大小
      glyph_data_type data_type;      // 数据类型
      rect_i          bounds;         // 边界框
      double          advance_x;      // X轴进步值
      double          advance_y;      // Y轴进步值
    };


    // 缓存字形集合类，管理多个cached_glyph对象
    class cached_glyphs
    {
    public:
      enum block_size_e { block_size = 16384-16 };  // 块大小常量定义为16368

      //--------------------------------------------------------------------
      cached_glyphs()
        : m_allocator(block_size)  // 使用指定大小初始化内存分配器
      { memset(m_glyphs, 0, sizeof(m_glyphs)); }  // 初始化m_glyphs数组为0

      //--------------------------------------------------------------------
      // 查找指定字形代码的缓存字形对象
      const cached_glyph* find_glyph(unsigned glyph_code) const
      {
        unsigned msb = (glyph_code >> 8) & 0xFF;  // 计算字形代码的高8位
        if(m_glyphs[msb])  // 如果m_glyphs中该索引位置不为空
        {
          return m_glyphs[msb][glyph_code & 0xFF];  // 返回指定索引的缓存字形对象
        }
        return 0;  // 否则返回空指针
      }

      //--------------------------------------------------------------------
      // 缓存指定字形数据
      cached_glyph* cache_glyph(
        void *            cached_font,
        unsigned        glyph_code,
        unsigned        glyph_index,
        unsigned        data_size,
        glyph_data_type data_type,
        const rect_i&   bounds,
        double          advance_x,
        double          advance_y)
    {
      // 提取字形代码中的最高有效字节
      unsigned msb = (glyph_code >> 8) & 0xFF;
      // 如果指定的字形代码最高有效字节对应的数组位置为空
      if(m_glyphs[msb] == 0)
      {
        // 分配内存给256个cached_glyph*的数组，并初始化为0
        m_glyphs[msb] =
          (cached_glyph**)m_allocator.allocate(sizeof(cached_glyph*) * 256,
          sizeof(cached_glyph*));
        memset(m_glyphs[msb], 0, sizeof(cached_glyph*) * 256);
      }

      // 提取字形代码中的最低有效字节
      unsigned lsb = glyph_code & 0xFF;
      // 如果在msb和lsb位置的cached_glyph*已经存在，返回0，不覆盖已存在的值
      if(m_glyphs[msb][lsb]) return 0;

      // 分配内存给cached_glyph对象
      cached_glyph* glyph =
        (cached_glyph*)m_allocator.allocate(sizeof(cached_glyph),
        sizeof(double));

      // 填充cached_glyph对象的字段
      glyph->cached_font          = cached_font;
      glyph->glyph_code           = glyph_code;
      glyph->glyph_index          = glyph_index;
      glyph->data                = m_allocator.allocate(data_size);
      glyph->data_size           = data_size;
      glyph->data_type           = data_type;
      glyph->bounds              = bounds;
      glyph->advance_x           = advance_x;
      glyph->advance_y           = advance_y;
      // 将新建的cached_glyph*保存在m_glyphs中的对应位置，并返回这个指针
      return m_glyphs[msb][lsb] = glyph;
    }

  private:
    block_allocator m_allocator;
    cached_glyph**   m_glyphs[256];
  };
    {
      cached_font(
        font_engine_type& engine,
        typename FontEngine::loaded_face *face,
        double height,
        double width,
        bool hinting,
        glyph_rendering rendering )
        : m_engine( engine )  // 初始化字体引擎
        , m_face( face )  // 初始化字体面对象指针
        , m_height( height )  // 设置字体高度
        , m_width( width )  // 设置字体宽度
        , m_hinting( hinting )  // 设置是否使用hinting
        , m_rendering( rendering )  // 设置字形渲染方式
      {
        select_face();  // 选择当前面对象实例
        m_face_height=m_face->height();  // 获取面对象的高度
        m_face_width=m_face->width();  // 获取面对象的宽度
        m_face_ascent=m_face->ascent();  // 获取面对象的ascent（上升部分）
        m_face_descent=m_face->descent();  // 获取面对象的descent（下降部分）
        m_face_ascent_b=m_face->ascent_b();  // 获取面对象的ascent_b（上升部分的baseline）
        m_face_descent_b=m_face->descent_b();  // 获取面对象的descent_b（下降部分的baseline）
      }
    
      double height() const
      {
        return m_face_height;  // 返回字体高度
      }
    
      double width() const
      {
        return m_face_width;  // 返回字体宽度
      }
    
      double ascent() const
      {
        return m_face_ascent;  // 返回字体ascent（上升部分）
      }
    
      double descent() const
      {
        return m_face_descent;  // 返回字体descent（下降部分）
      }
    
      double ascent_b() const
      {
        return m_face_ascent_b;  // 返回字体ascent_b（上升部分的baseline）
      }
    
      double descent_b() const
      {
        return m_face_descent_b;  // 返回字体descent_b（下降部分的baseline）
      }
    
      bool add_kerning( const cached_glyph *first, const cached_glyph *second, double* x, double* y)
      {
        if( !first || !second )
          return false;  // 如果first或second为空指针，则返回false
    
        select_face();  // 选择当前面对象实例
        return m_face->add_kerning(
          first->glyph_index, second->glyph_index, x, y );  // 调用面对象的add_kerning方法，添加字符间距信息
      }
    
      void select_face()
      {
        m_face->select_instance( m_height, m_width, m_hinting, m_rendering );  // 选择面对象的实例，使用当前设置的高度、宽度、hinting和渲染方式
      }
    
      const cached_glyph *get_glyph(unsigned cp)
      {
        const cached_glyph *glyph=m_glyphs.find_glyph(cp);  // 在缓存的字形中查找给定代码点cp对应的字形
        if( glyph==0 )
        {
          typename FontEngine::prepared_glyph prepared;
          select_face();  // 选择当前面对象实例
          bool success=m_face->prepare_glyph(cp, &prepared);  // 准备给定代码点cp的字形
          if( success )
          {
            glyph=m_glyphs.cache_glyph(
              this,
              prepared.glyph_code,
              prepared.glyph_index,
              prepared.data_size,
              prepared.data_type,
              prepared.bounds,
              prepared.advance_x,
              prepared.advance_y );  // 缓存字形到字形缓存中
            assert( glyph!=0 );  // 断言确保glyph非空
            m_face->write_glyph_to(&prepared,glyph->data);  // 将准备好的字形数据写入到glyph的数据中
          }
        }
        return glyph;  // 返回找到或缓存的字形
      }
    
      font_engine_type&   m_engine;  // 字体引擎对象的引用
      typename FontEngine::loaded_face *m_face;  // 加载的字体面对象指针
      double                m_height;  // 字体高度
      double                m_width;  // 字体宽度
      bool                m_hinting;  // 是否使用hinting
      glyph_rendering        m_rendering;  // 字形渲染方式
      double                m_face_height;  // 字体面对象的高度
      double                m_face_width;  // 字体面对象的宽度
      double                m_face_ascent;  // 字体面对象的ascent（上升部分）
      double                m_face_descent;  // 字体面对象的descent（下降部分）
      double                m_face_ascent_b;  // 字体面对象的ascent_b（上升部分的baseline）
      double                m_face_descent_b;  // 字体面对象的descent_b（下降部分的baseline）
      cached_glyphs        m_glyphs;  // 缓存的字形集合
    };
    // 定义字体缓存管理器构造函数，初始化字体引擎和最大字体数
    font_cache_manager(font_engine_type& engine, unsigned max_fonts=32)
      :m_engine(engine)
    { }
    
    //--------------------------------------------------------------------
    // 初始化内嵌适配器函数，根据不同的字形数据类型初始化相应的适配器
    void init_embedded_adaptors(const cached_glyph* gl,
      double x, double y,
      double scale=1.0)
    {
      // 如果传入的字形数据指针不为空
      if(gl)
      {
        // 根据字形数据类型进行不同的初始化
        switch(gl->data_type)
        {
        default: return; // 默认情况下返回
        case glyph_data_mono:
          m_mono_adaptor.init(gl->data, gl->data_size, x, y); // 初始化单色适配器
          break;
    
        case glyph_data_gray8:
          m_gray8_adaptor.init(gl->data, gl->data_size, x, y); // 初始化灰度8位适配器
          break;
    
        case glyph_data_outline:
          m_path_adaptor.init(gl->data, gl->data_size, x, y, scale); // 初始化轮廓适配器
          break;
        }
      }
    }
    
    
    //--------------------------------------------------------------------
    // 返回路径适配器的引用
    path_adaptor_type&   path_adaptor()   { return m_path_adaptor;   }
    // 返回灰度8位适配器的引用
    gray8_adaptor_type&  gray8_adaptor()  { return m_gray8_adaptor;  }
    // 返回灰度8位扫描线的引用
    gray8_scanline_type& gray8_scanline() { return m_gray8_scanline; }
    // 返回单色适配器的引用
    mono_adaptor_type&   mono_adaptor()   { return m_mono_adaptor;   }
    // 返回单色扫描线的引用
    mono_scanline_type&  mono_scanline()  { return m_mono_scanline;  }
    
    
    private:
    //--------------------------------------------------------------------
    // 私有复制构造函数，防止对象复制
    font_cache_manager(const self_type&);
    // 私有赋值运算符重载，防止对象赋值
    const self_type& operator = (const self_type&);
    
    // 字体引擎的引用
    font_engine_type&   m_engine;
    // 路径适配器对象
    path_adaptor_type   m_path_adaptor;
    // 灰度8位适配器对象
    gray8_adaptor_type  m_gray8_adaptor;
    // 灰度8位扫描线对象
    gray8_scanline_type m_gray8_scanline;
    // 单色适配器对象
    mono_adaptor_type   m_mono_adaptor;
    // 单色扫描线对象
    mono_scanline_type  m_mono_scanline;
    };
}
// 这是一个预处理器指令，表示结束一个条件编译段，与#ifdef匹配使用

}
// 结束了前一个的预处理器指令段，用于条件编译

#endif
// 结束一个条件编译指令，与#ifdef匹配使用
```