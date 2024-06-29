# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_rbox_ctrl.h`

```
// 定义 AGG 几何图形控件的矩形框控件 rbox_ctrl_impl
class rbox_ctrl_impl : public ctrl
{
public:
    // 构造函数，初始化矩形框的位置和大小，可选择是否翻转 y 轴
    rbox_ctrl_impl(double x1, double y1, double x2, double y2, bool flip_y=false);

    // 设置边框宽度及额外边框宽度
    void border_width(double t, double extra=0.0);

    // 设置文本的线条粗细
    void text_thickness(double t);

    // 设置文本的尺寸，高度 h 和宽度 w（默认为 0）
    void text_size(double h, double w=0.0);

    // 添加条目到控件中
    void add_item(const char* text);

    // 返回当前选中的条目索引
    int cur_item() const;

    // 设置当前选中的条目索引
    void cur_item(int i);

    // 判断给定的坐标 (x, y) 是否在矩形框内
    virtual bool in_rect(double x, double y) const;

    // 处理鼠标按下事件，返回是否处理成功
    virtual bool on_mouse_button_down(double x, double y);

    // 处理鼠标释放事件，返回是否处理成功
    virtual bool on_mouse_button_up(double x, double y);

    // 处理鼠标移动事件，返回是否处理成功
    virtual bool on_mouse_move(double x, double y, bool button_flag);

    // 处理方向键事件，返回是否处理成功
    virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

    // 顶点源接口，返回路径数目
    unsigned num_paths();

    // 准备重置路径
    void rewind(unsigned path_id);

    // 返回顶点坐标，存储在 x 和 y 中
    unsigned vertex(double* x, double* y);

private:
    // 计算矩形框的几何形状
    void calc_rbox();

    // 边框宽度
    double          m_border_width;

    // 额外边框宽度
    double          m_border_extra;

    // 文本线条粗细
    double          m_text_thickness;

    // 文本高度
    double          m_text_height;

    // 文本宽度
    double          m_text_width;

    // 条目数组，最多 32 个
    pod_array<char> m_items[32];

    // 条目数量
    unsigned        m_num_items;

    // 当前选中的条目索引
    int             m_cur_item;

    // 矩形框的左上角和右下角坐标
    double   m_xs1;
    double   m_ys1;
    double   m_xs2;
    double   m_ys2;

    // 条目顶点数组
    double   m_vx[32];
    double   m_vy[32];

    // 绘制的条目索引
    unsigned m_draw_item;

    // 垂直方向的间距
    double   m_dy;

    // 椭圆对象
    ellipse               m_ellipse;

    // 椭圆的描边转换器
    conv_stroke<ellipse>  m_ellipse_poly;

    // 文本对象
    gsv_text              m_text;

    // 文本的描边转换器
    conv_stroke<gsv_text> m_text_poly;

    // 索引和顶点数量
    unsigned m_idx;
    unsigned m_vertex;
};
    //------------------------------------------------------------------------
    template<class ColorT> class rbox_ctrl : public rbox_ctrl_impl
    {
    public:
        // 构造函数，初始化矩形控件，可选择是否翻转 y 轴
        rbox_ctrl(double x1, double y1, double x2, double y2, bool flip_y=false) :
            rbox_ctrl_impl(x1, y1, x2, y2, flip_y),
            // 初始化背景颜色为浅黄色
            m_background_color(rgba(1.0, 1.0, 0.9)),
            // 初始化边框颜色为黑色
            m_border_color(rgba(0.0, 0.0, 0.0)),
            // 初始化文本颜色为黑色
            m_text_color(rgba(0.0, 0.0, 0.0)),
            // 初始化非活动状态颜色为黑色
            m_inactive_color(rgba(0.0, 0.0, 0.0)),
            // 初始化活动状态颜色为深红色
            m_active_color(rgba(0.4, 0.0, 0.0))
        {
            // 将颜色指针数组指向各个颜色成员
            m_colors[0] = &m_background_color;
            m_colors[1] = &m_border_color;
            m_colors[2] = &m_text_color;
            m_colors[3] = &m_inactive_color;
            m_colors[4] = &m_active_color;
        }
          
    
        // 设置背景颜色
        void background_color(const ColorT& c) { m_background_color = c; }
        // 设置边框颜色
        void border_color(const ColorT& c) { m_border_color = c; }
        // 设置文本颜色
        void text_color(const ColorT& c) { m_text_color = c; }
        // 设置非活动状态颜色
        void inactive_color(const ColorT& c) { m_inactive_color = c; }
        // 设置活动状态颜色
        void active_color(const ColorT& c) { m_active_color = c; }
    
        // 返回指定索引的颜色对象的引用
        const ColorT& color(unsigned i) const { return *m_colors[i]; } 
    
    private:
        // 禁止拷贝构造函数
        rbox_ctrl(const rbox_ctrl<ColorT>&);
        // 禁止赋值运算符重载
        const rbox_ctrl<ColorT>& operator = (const rbox_ctrl<ColorT>&);
       
        // 成员变量：背景颜色
        ColorT m_background_color;
        // 成员变量：边框颜色
        ColorT m_border_color;
        // 成员变量：文本颜色
        ColorT m_text_color;
        // 成员变量：非活动状态颜色
        ColorT m_inactive_color;
        // 成员变量：活动状态颜色
        ColorT m_active_color;
        // 成员变量：颜色指针数组
        ColorT* m_colors[5];
    };
}


这行代码表示一个 C/C++ 程序的条件编译指令结束。通常情况下，它用于结束一个条件编译区块，对应于 `#ifdef` 或 `#ifndef` 等条件编译指令的开始。


#endif


这行代码用于结束一个条件编译区块，对应于 `#ifdef` 或 `#ifndef` 的开始部分。`#endif` 指令表明条件编译的结束，其中的条件通常是与 `#ifdef` 或 `#ifndef` 相对应的预处理器条件。
```