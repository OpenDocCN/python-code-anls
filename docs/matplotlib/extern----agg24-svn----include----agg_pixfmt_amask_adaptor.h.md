# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_pixfmt_amask_adaptor.h`

```py
// 定义了一个名为 pixfmt_amask_adaptor 的模板类，作为 PixFmt 和 AlphaMask 之间的适配器
template<class PixFmt, class AlphaMask> class pixfmt_amask_adaptor
{
public:
    // 定义类型别名，PixFmt 的颜色类型为 color_type，行数据为 row_data
    typedef PixFmt pixfmt_type;
    typedef typename pixfmt_type::color_type color_type;
    typedef typename pixfmt_type::row_data row_data;
    // 定义 AlphaMask 的类型别名，cover_type 表示覆盖类型
    typedef AlphaMask amask_type;
    typedef typename amask_type::cover_type cover_type;

private:
    // 枚举 span_extra_tail_e 定义为 256，用于扩展 span 数组长度
    enum span_extra_tail_e { span_extra_tail = 256 };

    // 重新分配 span 数组的大小，确保足够存储长度为 len 的元素
    void realloc_span(unsigned len)
    {
        if(len > m_span.size())
        {
            m_span.resize(len + span_extra_tail);
        }
    }

    // 初始化 span 数组为长度为 len，并用 amask_type::cover_full 初始化每个元素
    void init_span(unsigned len)
    {
        realloc_span(len);
        memset(&m_span[0], amask_type::cover_full, len * sizeof(cover_type));
    }

    // 初始化 span 数组为长度为 len，并使用 covers 数组的值进行初始化
    void init_span(unsigned len, const cover_type* covers)
    {
        realloc_span(len);
        memcpy(&m_span[0], covers, len * sizeof(cover_type));
    }


private:
    pixfmt_type*          m_pixf;         // PixFmt 类型的指针 m_pixf
    const amask_type*     m_mask;         // 指向常量 AlphaMask 类型对象的指针 m_mask
    pod_array<cover_type> m_span;         // 使用 pod_array 存储 cover_type 类型的 span 数组
};
```