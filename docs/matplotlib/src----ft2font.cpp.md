# `D:\src\scipysrc\matplotlib\src\ft2font.cpp`

```
/**
 * -*- mode: c++; c-basic-offset: 4 -*-
 */

// 定义宏，禁止导入数组
#define NO_IMPORT_ARRAY

// 包含必要的头文件
#include <algorithm>    // 提供算法
#include <iterator>     // 提供迭代器
#include <set>          // 提供集合容器
#include <sstream>      // 提供字符串流
#include <stdexcept>    // 提供标准异常类
#include <string>       // 提供字符串操作

// 包含自定义头文件
#include "ft2font.h"        // FreeType 2 字体渲染相关
#include "mplutils.h"       // Matplotlib 工具
#include "numpy_cpp.h"      // NumPy C++ 接口
#include "py_exceptions.h"  // Python 异常处理

// 如果未定义 M_PI 宏，定义为 π 的值
#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

/**
 * 为了改善字体的提示效果，此代码使用了一个技巧，
 * 参见 http://agg.sourceforge.net/antigrain.com/research/font_rasterization/index.html
 * 
 * 思路是在 x 方向限制提示效果，同时保留 y 方向的提示。
 * 由于 freetype 不直接支持这一点，因此在 x 方向设置的 dpi 高于 y 方向，
 * 这会影响提示网格。然后，在字体上放置全局变换以缩小它回到期望的大小。
 * 虽然 dpi 设置影响提示，而全局变换不影响这一点有些奇怪，
 * 但这是 FreeType 的文档行为，并且希望不太可能改变。
 * FreeType 2 教程中提到：
 * 
 * 注意：该变换应用于通过 FT_Load_Glyph 加载的每个字形，
 * 完全独立于任何提示过程。这意味着，如果你加载一个大小为 24 像素的字形，
 * 或者通过变换按比例为 2 加载一个大小为 12 像素的字形，你将不会得到相同的结果，
 * 因为提示将以不同的方式计算（除非你禁用提示）。
 */

// FreeType 2 库的全局变量
FT_Library _ft2Library;

// 根据 FreeType 错误代码加载错误字符串
static char const* ft_error_string(FT_Error error) {
#undef __FTERRORS_H__
#define FT_ERROR_START_LIST     switch (error) {
#define FT_ERRORDEF( e, v, s )    case v: return s;
#define FT_ERROR_END_LIST         default: return NULL; }
#include FT_ERRORS_H
}

// 抛出 FreeType 错误的异常处理函数
void throw_ft_error(std::string message, FT_Error error) {
    char const* s = ft_error_string(error);
    std::ostringstream os("");
    if (s) {
        os << message << " (" << s << "; error code 0x" << std::hex << error << ")";
    } else {  // 不应该发生，但不会因查找失败而添加其他错误。
        os << message << " (error code 0x" << std::hex << error << ")";
    }
    throw std::runtime_error(os.str());
}

// 默认构造函数，初始化成员变量
FT2Image::FT2Image() : m_dirty(true), m_buffer(NULL), m_width(0), m_height(0)
{
}

// 带参数的构造函数，调用 resize() 函数初始化图像大小
FT2Image::FT2Image(unsigned long width, unsigned long height)
    : m_dirty(true), m_buffer(NULL), m_width(0), m_height(0)
{
    resize(width, height);
}

// 析构函数，释放 m_buffer 所指向的内存
FT2Image::~FT2Image()
{
    delete[] m_buffer;
}

// 调整图像大小的函数实现
void FT2Image::resize(long width, long height)
{
    // 如果宽度或高度小于等于 0，则设置为 1
    if (width <= 0) {
        width = 1;
    }
    if (height <= 0) {
        height = 1;
    }
    // 计算需要的字节数
    size_t numBytes = width * height;
    # 检查传入的宽度和高度是否与当前对象的宽度和高度不同
    if ((unsigned long)width != m_width || (unsigned long)height != m_height) {
        # 如果新的像素数据大小大于当前缓冲区大小，需要重新分配缓冲区
        if (numBytes > m_width * m_height) {
            # 删除当前的缓冲区
            delete[] m_buffer;
            # 将缓冲区指针置为空
            m_buffer = NULL;
            # 分配新的缓冲区，大小为新像素数据的大小
            m_buffer = new unsigned char[numBytes];
        }

        # 更新对象的宽度和高度为新传入的宽度和高度
        m_width = (unsigned long)width;
        m_height = (unsigned long)height;
    }

    # 如果传入的像素数据大小不为0并且缓冲区存在，则将缓冲区清零
    if (numBytes && m_buffer) {
        # 使用 memset 函数将缓冲区的内容全部置为0
        memset(m_buffer, 0, numBytes);
    }

    # 将对象的 dirty 标志设置为true，表示数据已经被修改
    m_dirty = true;
// 绘制位图到图像缓冲区中指定的位置
void FT2Image::draw_bitmap(FT_Bitmap *bitmap, FT_Int x, FT_Int y)
{
    FT_Int image_width = (FT_Int)m_width;  // 图像宽度
    FT_Int image_height = (FT_Int)m_height;  // 图像高度
    FT_Int char_width = bitmap->width;  // 字符宽度
    FT_Int char_height = bitmap->rows;  // 字符高度

    FT_Int x1 = std::min(std::max(x, 0), image_width);  // 起始x坐标
    FT_Int y1 = std::min(std::max(y, 0), image_height);  // 起始y坐标
    FT_Int x2 = std::min(std::max(x + char_width, 0), image_width);  // 结束x坐标
    FT_Int y2 = std::min(std::max(y + char_height, 0), image_height);  // 结束y坐标

    FT_Int x_start = std::max(0, -x);  // x方向开始绘制位置
    FT_Int y_offset = y1 - std::max(0, -y);  // y方向偏移量

    if (bitmap->pixel_mode == FT_PIXEL_MODE_GRAY) {
        // 灰度像素模式处理
        for (FT_Int i = y1; i < y2; ++i) {
            unsigned char *dst = m_buffer + (i * image_width + x1);  // 目标缓冲区指针
            unsigned char *src = bitmap->buffer + (((i - y_offset) * bitmap->pitch) + x_start);  // 源位图数据指针
            for (FT_Int j = x1; j < x2; ++j, ++dst, ++src)
                *dst |= *src;  // 位图数据与目标缓冲区做或操作
        }
    } else if (bitmap->pixel_mode == FT_PIXEL_MODE_MONO) {
        // 单色像素模式处理
        for (FT_Int i = y1; i < y2; ++i) {
            unsigned char *dst = m_buffer + (i * image_width + x1);  // 目标缓冲区指针
            unsigned char *src = bitmap->buffer + ((i - y_offset) * bitmap->pitch);  // 源位图数据指针
            for (FT_Int j = x1; j < x2; ++j, ++dst) {
                int x = (j - x1 + x_start);  // 计算位图数据偏移量
                int val = *(src + (x >> 3)) & (1 << (7 - (x & 0x7)));  // 获取位图像素值
                *dst = val ? 255 : *dst;  // 根据像素值设置目标缓冲区值
            }
        }
    } else {
        throw std::runtime_error("Unknown pixel mode");  // 未知的像素模式异常
    }

    m_dirty = true;  // 标记图像缓冲区已修改
}

// 在图像缓冲区中绘制一个空心矩形
void FT2Image::draw_rect(unsigned long x0, unsigned long y0, unsigned long x1, unsigned long y1)
{
    if (x0 > m_width || x1 > m_width || y0 > m_height || y1 > m_height) {
        throw std::runtime_error("Rect coords outside image bounds");  // 矩形坐标超出图像边界异常
    }

    size_t top = y0 * m_width;  // 矩形顶部在缓冲区中的位置
    size_t bottom = y1 * m_width;  // 矩形底部在缓冲区中的位置
    for (size_t i = x0; i < x1 + 1; ++i) {
        m_buffer[i + top] = 255;  // 设置顶部和底部边界的像素值为255
        m_buffer[i + bottom] = 255;
    }

    for (size_t j = y0 + 1; j < y1; ++j) {
        m_buffer[x0 + j * m_width] = 255;  // 设置左右边界的像素值为255
        m_buffer[x1 + j * m_width] = 255;
    }

    m_dirty = true;  // 标记图像缓冲区已修改
}

// 在图像缓冲区中绘制一个填充的矩形
void FT2Image::draw_rect_filled(unsigned long x0, unsigned long y0, unsigned long x1, unsigned long y1)
{
    x0 = std::min(x0, m_width);  // 限制矩形坐标在图像范围内
    y0 = std::min(y0, m_height);
    x1 = std::min(x1 + 1, m_width);
    y1 = std::min(y1 + 1, m_height);

    for (size_t j = y0; j < y1; j++) {
        for (size_t i = x0; i < x1; i++) {
            m_buffer[i + j * m_width] = 255;  // 设置矩形区域内的所有像素值为255
        }
    }

    m_dirty = true;  // 标记图像缓冲区已修改
}

// 警告：绘制字形时的异常处理，输出一组字体名称到流中
static void ft_glyph_warn(FT_ULong charcode, std::set<FT_String*> family_names)
{
    PyObject *text_helpers = NULL, *tmp = NULL;  // Python对象的初始化
    std::set<FT_String*>::iterator it = family_names.begin();  // 迭代器初始化为字体名称集合的起始位置
    std::stringstream ss;  // 字符串流
    ss<<*it;  // 将第一个字体名称写入字符串流
    while(++it != family_names.end()){  // 迭代输出所有字体名称到字符串流
        ss<<", "<<*it;
    }
    // 继续...
}
    // 导入 matplotlib._text_helpers 模块，若导入失败或返回值为空，则进入条件判断
    if (!(text_helpers = PyImport_ImportModule("matplotlib._text_helpers")) ||
        // 调用 text_helpers 模块的方法 "warn_on_missing_glyph"，传入参数 "(k, s)"
        !(tmp = PyObject_CallMethod(text_helpers,
                "warn_on_missing_glyph", "(k, s)",
                charcode, ss.str().c_str()))) {
        // 如果导入或调用失败，则跳转到 exit 标签处
        goto exit;
    }
exit:
    // 释放text_helpers指向的Python对象
    Py_XDECREF(text_helpers);
    // 释放tmp指向的Python对象
    Py_XDECREF(tmp);
    // 如果发生了Python异常，则抛出自定义异常mpl::exception()
    if (PyErr_Occurred()) {
        throw mpl::exception();
    }
}

// ft_outline_decomposer应该传递给FT_Outline_Decompose函数。在第一遍循环中，vertices和codes被设置为NULL，
// 并且index被简单地增加，以便于插入每个顶点，因此在结束时，index被设置为总顶点数。
// 在第二遍循环中，vertices和codes应该指向正确大小的数组，并且index再次设置为零，
// 以便获取轮廓分解的填充顶点和代码。
struct ft_outline_decomposer
{
    int index;          // 当前处理的顶点索引
    double* vertices;   // 指向顶点数组的指针
    unsigned char* codes; // 指向代码数组的指针
};

static int
ft_outline_move_to(FT_Vector const* to, void* user)
{
    ft_outline_decomposer* d = reinterpret_cast<ft_outline_decomposer*>(user);
    if (d->codes) {
        if (d->index) {
            // 添加CLOSEPOLY非常重要，以使路径效果(patheffects)工作。
            *(d->vertices++) = 0;
            *(d->vertices++) = 0;
            *(d->codes++) = CLOSEPOLY;
        }
        *(d->vertices++) = to->x * (1. / 64.);
        *(d->vertices++) = to->y * (1. / 64.);
        *(d->codes++) = MOVETO;
    }
    d->index += d->index ? 2 : 1;  // 更新顶点索引，考虑是否需要加2
    return 0;
}

static int
ft_outline_line_to(FT_Vector const* to, void* user)
{
    ft_outline_decomposer* d = reinterpret_cast<ft_outline_decomposer*>(user);
    if (d->codes) {
        *(d->vertices++) = to->x * (1. / 64.);
        *(d->vertices++) = to->y * (1. / 64.);
        *(d->codes++) = LINETO;
    }
    d->index++;  // 更新顶点索引
    return 0;
}

static int
ft_outline_conic_to(FT_Vector const* control, FT_Vector const* to, void* user)
{
    ft_outline_decomposer* d = reinterpret_cast<ft_outline_decomposer*>(user);
    if (d->codes) {
        *(d->vertices++) = control->x * (1. / 64.);
        *(d->vertices++) = control->y * (1. / 64.);
        *(d->vertices++) = to->x * (1. / 64.);
        *(d->vertices++) = to->y * (1. / 64.);
        *(d->codes++) = CURVE3;
        *(d->codes++) = CURVE3;
    }
    d->index += 2;  // 更新顶点索引
    return 0;
}

static int
ft_outline_cubic_to(
  FT_Vector const* c1, FT_Vector const* c2, FT_Vector const* to, void* user)
{
    ft_outline_decomposer* d = reinterpret_cast<ft_outline_decomposer*>(user);
    if (d->codes) {
        *(d->vertices++) = c1->x * (1. / 64.);
        *(d->vertices++) = c1->y * (1. / 64.);
        *(d->vertices++) = c2->x * (1. / 64.);
        *(d->vertices++) = c2->y * (1. / 64.);
        *(d->vertices++) = to->x * (1. / 64.);
        *(d->vertices++) = to->y * (1. / 64.);
        *(d->codes++) = CURVE4;
        *(d->codes++) = CURVE4;
        *(d->codes++) = CURVE4;
    }
    d->index += 3;  // 更新顶点索引
    return 0;
}

static FT_Outline_Funcs ft_outline_funcs = {
    ft_outline_move_to,
    ft_outline_line_to,
    ft_outline_conic_to,
    ft_outline_cubic_to};

PyObject*
FT2Font::get_path()
{
    if (!face->glyph) {
        PyErr_SetString(PyExc_RuntimeError, "No glyph loaded");
        return NULL;
    }
    # 创建一个名为 decomposer 的 ft_outline_decomposer 结构体实例
    ft_outline_decomposer decomposer = {};
    
    # 调用 FT_Outline_Decompose 函数，将字形的轮廓信息分解为简单的轮廓段
    if (FT_Error error =
        FT_Outline_Decompose(
          &face->glyph->outline, &ft_outline_funcs, &decomposer)) {
        # 如果分解失败，则抛出运行时错误，并返回空指针
        PyErr_Format(PyExc_RuntimeError,
                     "FT_Outline_Decompose failed with error 0x%x", error);
        return NULL;
    }
    
    # 如果 decomposer.index 为零，表示没有轮廓数据（空轮廓），不附加 CLOSEPOLY
    if (!decomposer.index) {  // Don't append CLOSEPOLY to null glyphs.
        # 创建一个空的 vertices 数组视图，表示没有顶点
        npy_intp vertices_dims[2] = { 0, 2 };
        numpy::array_view<double, 2> vertices(vertices_dims);
        
        # 创建一个空的 codes 数组视图，表示没有路径代码
        npy_intp codes_dims[1] = { 0 };
        numpy::array_view<unsigned char, 1> codes(codes_dims);
        
        # 返回一个 Python 对象，表示空的顶点数组和路径代码数组
        return Py_BuildValue("NN", vertices.pyobj(), codes.pyobj());
    }
    
    # 根据 decomposer.index 大小创建 vertices 和 codes 数组视图
    npy_intp vertices_dims[2] = { decomposer.index + 1, 2 };
    numpy::array_view<double, 2> vertices(vertices_dims);
    npy_intp codes_dims[1] = { decomposer.index + 1 };
    numpy::array_view<unsigned char, 1> codes(codes_dims);
    
    # 重置 decomposer 的索引，并设置 vertices 和 codes 的数据指针
    decomposer.index = 0;
    decomposer.vertices = vertices.data();
    decomposer.codes = codes.data();
    
    # 再次调用 FT_Outline_Decompose 函数，将轮廓分解为简单的轮廓段
    if (FT_Error error =
        FT_Outline_Decompose(
          &face->glyph->outline, &ft_outline_funcs, &decomposer)) {
        # 如果分解失败，则抛出运行时错误，并返回空指针
        PyErr_Format(PyExc_RuntimeError,
                     "FT_Outline_Decompose failed with error 0x%x", error);
        return NULL;
    }
    
    # 将一个点 (0, 0) 添加到 vertices 数组中
    *(decomposer.vertices++) = 0;
    *(decomposer.vertices++) = 0;
    
    # 将 CLOSEPOLY 添加到 codes 数组中
    *(decomposer.codes++) = CLOSEPOLY;
    
    # 返回包含 vertices 和 codes 数组对象的 Python 元组
    return Py_BuildValue("NN", vertices.pyobj(), codes.pyobj());
}

FT2Font::FT2Font(FT_Open_Args &open_args,
                 long hinting_factor_,
                 std::vector<FT2Font *> &fallback_list)
    : image(), face(NULL)
{
    // 初始化对象状态
    clear();

    // 使用 FreeType 库打开字体文件，并将结果保存在 face 中
    FT_Error error = FT_Open_Face(_ft2Library, &open_args, 0, &face);
    if (error) {
        throw_ft_error("Can not load face", error);
    }

    // 设置默认的字距因子为 0，即不进行字距调整
    kerning_factor = 0;

    // 设置默认的字体大小为 12 点，分辨率为 72dpi
    hinting_factor = hinting_factor_;

    // 设置字体大小和分辨率
    error = FT_Set_Char_Size(face, 12 * 64, 0, 72 * (unsigned int)hinting_factor, 72);
    if (error) {
        FT_Done_Face(face);
        throw_ft_error("Could not set the fontsize", error);
    }

    // 如果使用了外部流，则设置相应的标志位
    if (open_args.stream != NULL) {
        face->face_flags |= FT_FACE_FLAG_EXTERNAL_STREAM;
    }

    // 设置变换矩阵，用于调整字形的渲染
    FT_Matrix transform = { 65536 / hinting_factor, 0, 0, 65536 };
    FT_Set_Transform(face, &transform, 0);

    // 设置备用字体列表
    std::copy(fallback_list.begin(), fallback_list.end(), std::back_inserter(fallbacks));
}

FT2Font::~FT2Font()
{
    // 释放所有已加载的字形对象
    for (size_t i = 0; i < glyphs.size(); i++) {
        FT_Done_Glyph(glyphs[i]);
    }

    // 如果字体对象存在，释放其资源
    if (face) {
        FT_Done_Face(face);
    }
}

void FT2Font::clear()
{
    // 重置笔的位置
    pen.x = 0;
    pen.y = 0;

    // 释放所有已加载的字形对象
    for (size_t i = 0; i < glyphs.size(); i++) {
        FT_Done_Glyph(glyphs[i]);
    }

    // 清空字形和字体映射表
    glyphs.clear();
    glyph_to_font.clear();
    char_to_font.clear();

    // 清空备用字体列表中每个字体的状态
    for (size_t i = 0; i < fallbacks.size(); i++) {
        fallbacks[i]->clear();
    }
}

void FT2Font::set_size(double ptsize, double dpi)
{
    // 设置字体大小和分辨率
    FT_Error error = FT_Set_Char_Size(
        face, (FT_F26Dot6)(ptsize * 64), 0, (FT_UInt)(dpi * hinting_factor), (FT_UInt)dpi);
    if (error) {
        throw_ft_error("Could not set the fontsize", error);
    }

    // 设置变换矩阵，用于调整字形的渲染
    FT_Matrix transform = { 65536 / hinting_factor, 0, 0, 65536 };
    FT_Set_Transform(face, &transform, 0);

    // 设置备用字体列表中每个字体的大小和分辨率
    for (size_t i = 0; i < fallbacks.size(); i++) {
        fallbacks[i]->set_size(ptsize, dpi);
    }
}

void FT2Font::set_charmap(int i)
{
    // 设置当前使用的字符映射表
    if (i >= face->num_charmaps) {
        throw std::runtime_error("i exceeds the available number of char maps");
    }
    FT_CharMap charmap = face->charmaps[i];
    if (FT_Error error = FT_Set_Charmap(face, charmap)) {
        throw_ft_error("Could not set the charmap", error);
    }
}

void FT2Font::select_charmap(unsigned long i)
{
    // 选择指定编码方式的字符映射表
    if (FT_Error error = FT_Select_Charmap(face, (FT_Encoding)i)) {
        throw_ft_error("Could not set the charmap", error);
    }
}

int FT2Font::get_kerning(FT_UInt left, FT_UInt right, FT_UInt mode, bool fallback /*= false*/)
{
    // 如果 fallback 为真，并且 glyph_to_font 中能找到 left 和 right 字形对应的字体对象
    // 则执行以下条件语句
    if (fallback && glyph_to_font.find(left) != glyph_to_font.end() &&
        glyph_to_font.find(right) != glyph_to_font.end()) {
        
        // 获取 left 和 right 字形对应的字体对象
        FT2Font *left_ft_object = glyph_to_font[left];
        FT2Font *right_ft_object = glyph_to_font[right];
        
        // 如果 left 和 right 字形对应的字体对象不同
        if (left_ft_object != right_ft_object) {
            // 我们不知道如何在不同的字体之间进行字距调整
            return 0;
        }
        
        // 如果 left_ft_object 等于 right_ft_object，则执行与 set_text 相同的操作
        return right_ft_object->get_kerning(left, right, mode, false);
    }
    else
    {
        // 如果 fallback 不为真，或者 glyph_to_font 中找不到 left 或 right 字形对应的字体对象
        FT_Vector delta;
        
        // 调用当前对象的 get_kerning 方法，返回 left 和 right 字形的字距值
        return get_kerning(left, right, mode, delta);
    }
}

// 获取字符间距调整值
int FT2Font::get_kerning(FT_UInt left, FT_UInt right, FT_UInt mode, FT_Vector &delta)
{
    // 检查字体是否支持字符间距调整
    if (!FT_HAS_KERNING(face)) {
        return 0;
    }

    // 获取字符间距信息并计算调整值
    if (!FT_Get_Kerning(face, left, right, mode, &delta)) {
        // 根据提示因子和间距因子计算最终的字符间距调整值
        return (int)(delta.x) / (hinting_factor << kerning_factor);
    } else {
        return 0;
    }
}

// 设置字符间距因子
void FT2Font::set_kerning_factor(int factor)
{
    // 更新字符间距因子
    kerning_factor = factor;
    // 递归更新所有回退字体对象的字符间距因子
    for (size_t i = 0; i < fallbacks.size(); i++) {
        fallbacks[i]->set_kerning_factor(factor);
    }
}

// 设置文本参数和变换矩阵
void FT2Font::set_text(
    size_t N, uint32_t *codepoints, double angle, FT_Int32 flags, std::vector<double> &xys)
{
    FT_Matrix matrix; /* transformation matrix */

    // 计算角度对应的旋转矩阵
    angle = angle * (2 * M_PI / 360.0);
    double cosangle = cos(angle) * 0x10000L;
    double sinangle = sin(angle) * 0x10000L;

    matrix.xx = (FT_Fixed)cosangle;
    matrix.xy = (FT_Fixed)-sinangle;
    matrix.yx = (FT_Fixed)sinangle;
    matrix.yy = (FT_Fixed)cosangle;

    // 清除先前的设置
    clear();

    // 初始化边界框的初始值
    bbox.xMin = bbox.yMin = 32000;
    bbox.xMax = bbox.yMax = -32000;

    FT_UInt previous = 0;
    FT2Font *previous_ft_object = NULL;
    // 遍历每个字符的索引，从0到N-1
    for (size_t n = 0; n < N; n++) {
        // 初始化字形索引、字形边界框和上一个字符的笔画位置
        FT_UInt glyph_index = 0;
        FT_BBox glyph_bbox;
        FT_Pos last_advance;

        // 初始化字符码错误、字形错误以及已经检查过的字体集合
        FT_Error charcode_error, glyph_error;
        std::set<FT_String*> glyph_seen_fonts;
        // 设置当前字体对象为该对象本身
        FT2Font *ft_object_with_glyph = this;
        
        // 调用加载字符的函数，尝试加载字符及其对应的字形信息
        bool was_found = load_char_with_fallback(ft_object_with_glyph, glyph_index, glyphs,
                                                 char_to_font, glyph_to_font, codepoints[n], flags,
                                                 charcode_error, glyph_error, glyph_seen_fonts, false);
        // 如果字符未找到
        if (!was_found) {
            // 发出警告，指出未找到的字符和检查过的字体集合
            ft_glyph_warn((FT_ULong)codepoints[n], glyph_seen_fonts);
            // 渲染缺失的字形为豆腐块
            // 回到最顶层的字体对象
            ft_object_with_glyph = this;
            // 将字符码映射到当前字体对象
            char_to_font[codepoints[n]] = ft_object_with_glyph;
            // 将字形索引映射到当前字体对象
            glyph_to_font[glyph_index] = ft_object_with_glyph;
            // 加载缺失的字形
            ft_object_with_glyph->load_glyph(glyph_index, flags, ft_object_with_glyph, false);
        }

        // 检索字距和移动笔画位置
        if ((ft_object_with_glyph == previous_ft_object) &&  // 如果两个字体对象相同
            ft_object_with_glyph->has_kerning() &&           // 如果字体支持字距调整
            previous && glyph_index                          // 并且确实存在两个字形
            ) {
            // 计算字距并更新笔画位置
            FT_Vector delta;
            pen.x += ft_object_with_glyph->get_kerning(previous, glyph_index, FT_KERNING_DEFAULT, delta);
        }

        // 提取字形图像并存储到字形表中
        FT_Glyph &thisGlyph = glyphs[glyphs.size() - 1];

        // 获取当前字形的水平进度值
        last_advance = ft_object_with_glyph->get_face()->glyph->advance.x;
        // 对字形进行笔画变换
        FT_Glyph_Transform(thisGlyph, 0, &pen);
        // 对字形进行矩阵变换
        FT_Glyph_Transform(thisGlyph, &matrix, 0);
        // 将当前笔画位置的X、Y坐标添加到向量xys中
        xys.push_back(pen.x);
        xys.push_back(pen.y);

        // 获取字形边界框信息
        FT_Glyph_Get_CBox(thisGlyph, FT_GLYPH_BBOX_SUBPIXELS, &glyph_bbox);

        // 更新整体边界框的最小和最大X、Y值
        bbox.xMin = std::min(bbox.xMin, glyph_bbox.xMin);
        bbox.xMax = std::max(bbox.xMax, glyph_bbox.xMax);
        bbox.yMin = std::min(bbox.yMin, glyph_bbox.yMin);
        bbox.yMax = std::max(bbox.yMax, glyph_bbox.yMax);

        // 更新笔画位置的X坐标，加上上一个字形的水平进度
        pen.x += last_advance;

        // 更新上一个字形的索引和字体对象
        previous = glyph_index;
        previous_ft_object = ft_object_with_glyph;

    }

    // 对笔画位置进行矩阵变换
    FT_Vector_Transform(&pen, &matrix);
    // 更新进度值为笔画位置的X坐标
    advance = pen.x;

    // 如果边界框的最小X值大于最大X值，则将边界框设置为0
    if (bbox.xMin > bbox.xMax) {
        bbox.xMin = bbox.yMin = bbox.xMax = bbox.yMax = 0;
    }
}

void FT2Font::load_char(long charcode, FT_Int32 flags, FT2Font *&ft_object, bool fallback = false)
{
    // 如果这是父级 FT2Font，则缓存将以两种方式填充：
    // 1. 先前调用过 set_text 方法
    // 2. 没有调用过 set_text 方法并且启用了回退（fallback）
    std::set <FT_String *> glyph_seen_fonts;
    
    // 如果启用了回退且 char_to_font 中存在 charcode 对应的对象
    if (fallback && char_to_font.find(charcode) != char_to_font.end()) {
        ft_object = char_to_font[charcode];
        // 由于无论如何都将分配给 ft_object
        FT2Font *throwaway = NULL;
        ft_object->load_char(charcode, flags, throwaway, false);
    } 
    // 如果启用了回退但 char_to_font 中不存在 charcode 对应的对象
    else if (fallback) {
        FT_UInt final_glyph_index;
        FT_Error charcode_error, glyph_error;
        FT2Font *ft_object_with_glyph = this;
        // 尝试使用回退加载字符
        bool was_found = load_char_with_fallback(ft_object_with_glyph, final_glyph_index,
                                                 glyphs, char_to_font, glyph_to_font,
                                                 charcode, flags, charcode_error, glyph_error,
                                                 glyph_seen_fonts, true);
        // 如果未找到字符
        if (!was_found) {
            // 发出字符警告
            ft_glyph_warn(charcode, glyph_seen_fonts);
            // 如果 charcode_error 存在，则抛出字符加载错误
            if (charcode_error) {
                throw_ft_error("Could not load charcode", charcode_error);
            }
            // 否则，如果 glyph_error 存在，则抛出字符加载错误
            else if (glyph_error) {
                throw_ft_error("Could not load charcode", glyph_error);
            }
        }
        ft_object = ft_object_with_glyph;
    } 
    // 如果未启用回退
    else {
        // 没有回退的情况
        ft_object = this;
        // 获取字符的字形索引
        FT_UInt glyph_index = FT_Get_Char_Index(face, (FT_ULong) charcode);
        // 如果没有找到字形索引
        if (!glyph_index){
            // 将见过的字体名称插入 glyph_seen_fonts 中
            glyph_seen_fonts.insert((face != NULL)?face->family_name: NULL);
            // 发出字符警告
            ft_glyph_warn((FT_ULong)charcode, glyph_seen_fonts);
        }
        // 如果加载字形出错
        if (FT_Error error = FT_Load_Glyph(face, glyph_index, flags)) {
            // 抛出字符加载错误
            throw_ft_error("Could not load charcode", error);
        }
        // 获取此字形的 FT_Glyph 对象
        FT_Glyph thisGlyph;
        if (FT_Error error = FT_Get_Glyph(face->glyph, &thisGlyph)) {
            // 抛出获取字形错误
            throw_ft_error("Could not get glyph", error);
        }
        // 将字形对象添加到 glyphs 中
        glyphs.push_back(thisGlyph);
    }
}


bool FT2Font::get_char_fallback_index(FT_ULong charcode, int& index) const
{
    // 获取字符的字形索引
    FT_UInt glyph_index = FT_Get_Char_Index(face, charcode);
    // 如果存在字形索引
    if (glyph_index) {
        // -1 表示主机拥有该字符，我们无需回退
        index = -1;
        return true;
    } 
    // 如果不存在字形索引
    else {
        int inner_index = 0;
        bool was_found;
        
        // 遍历回退列表
        for (size_t i = 0; i < fallbacks.size(); ++i) {
            // TODO 处理递归
            // 尝试从回退中获取字符的索引
            was_found = fallbacks[i]->get_char_fallback_index(charcode, inner_index);
            // 如果找到了字符
            if (was_found) {
                index = i;
                return true;
            }
        }
    }
    // 如果未找到字符，则返回 false
    return false;
}
// 加载具有备用方案的字符，根据需要选择字体对象，处理加载和缓存操作
bool FT2Font::load_char_with_fallback(FT2Font *&ft_object_with_glyph,
                                      FT_UInt &final_glyph_index,
                                      std::vector<FT_Glyph> &parent_glyphs,
                                      std::unordered_map<long, FT2Font *> &parent_char_to_font,
                                      std::unordered_map<FT_UInt, FT2Font *> &parent_glyph_to_font,
                                      long charcode,
                                      FT_Int32 flags,
                                      FT_Error &charcode_error,
                                      FT_Error &glyph_error,
                                      std::set<FT_String*> &glyph_seen_fonts,
                                      bool override = false)
{
    // 获取字符在字体中的字形索引
    FT_UInt glyph_index = FT_Get_Char_Index(face, charcode);
    // 将字体名称添加到已见过字体集合中
    glyph_seen_fonts.insert(face->family_name);

    // 如果找到了字符的字形索引或者要求覆盖，则继续处理
    if (glyph_index || override) {
        // 加载字符的字形数据
        charcode_error = FT_Load_Glyph(face, glyph_index, flags);
        if (charcode_error) {
            return false;
        }
        FT_Glyph thisGlyph;
        // 获取字符的字形对象
        glyph_error = FT_Get_Glyph(face->glyph, &thisGlyph);
        if (glyph_error) {
            return false;
        }

        // 设置最终使用的字形索引
        final_glyph_index = glyph_index;

        // 缓存结果以供将来使用
        ft_object_with_glyph = this;
        parent_glyph_to_font[final_glyph_index] = this;
        parent_char_to_font[charcode] = this;
        parent_glyphs.push_back(thisGlyph);
        return true;
    }
    else {
        // 如果没有找到字符的字形索引且有备用字体，尝试备用字体加载
        for (size_t i = 0; i < fallbacks.size(); ++i) {
            bool was_found = fallbacks[i]->load_char_with_fallback(
                ft_object_with_glyph, final_glyph_index, parent_glyphs,
                parent_char_to_font, parent_glyph_to_font, charcode, flags,
                charcode_error, glyph_error, glyph_seen_fonts, override);
            if (was_found) {
                return true;
            }
        }
        return false;
    }
}

// 加载给定字形的数据，可选地使用备用字体
void FT2Font::load_glyph(FT_UInt glyph_index,
                         FT_Int32 flags,
                         FT2Font *&ft_object,
                         bool fallback = false)
{
    // 如果启用备用字体并且已经缓存了该字形对应的字体对象，则直接使用缓存
    if (fallback && glyph_to_font.find(glyph_index) != glyph_to_font.end()) {
        ft_object = glyph_to_font[glyph_index];
    } else {
        // 否则使用当前字体对象
        ft_object = this;
    }

    // 调用当前字体对象的加载字形方法
    ft_object->load_glyph(glyph_index, flags);
}

// 加载指定字形的数据，并将其存储在字形集合中
void FT2Font::load_glyph(FT_UInt glyph_index, FT_Int32 flags)
{
    // 加载字形数据到当前字体对象中
    if (FT_Error error = FT_Load_Glyph(face, glyph_index, flags)) {
        throw_ft_error("Could not load glyph", error);
    }
    FT_Glyph thisGlyph;
    // 获取字形对象
    if (FT_Error error = FT_Get_Glyph(face->glyph, &thisGlyph)) {
        throw_ft_error("Could not get glyph", error);
    }
    // 将字形对象添加到字形集合中
    glyphs.push_back(thisGlyph);
}
FT_UInt FT2Font::get_char_index(FT_ULong charcode, bool fallback = false)
{
    // 声明一个 FT2Font 指针对象 ft_object，初始化为 NULL
    FT2Font *ft_object = NULL;
    // 如果 fallback 为 true，并且 char_to_font 中存在 charcode 键
    if (fallback && char_to_font.find(charcode) != char_to_font.end()) {
        // fallback 表示是否要搜索回退列表。
        // 在此之前应调用 set_text/load_char_with_fallback 到父 FT2Font，
        // 以便在这里使用回退列表（因为它会填充缓存）
        // 将 ft_object 设置为 char_to_font 中对应 charcode 的值
        ft_object = char_to_font[charcode];
    } else {
        // 否则将 ft_object 设置为当前对象的指针 this
        ft_object = this;
    }

    // 调用 FT_Get_Char_Index 获取字符 charcode 在 ft_object 字体面的索引
    return FT_Get_Char_Index(ft_object->get_face(), charcode);
}

void FT2Font::get_width_height(long *width, long *height)
{
    // 将 advance 的值赋给传入指针 *width
    *width = advance;
    // 计算并将 bbox 的高度赋给传入指针 *height
    *height = bbox.yMax - bbox.yMin;
}

long FT2Font::get_descent()
{
    // 返回负的 bbox.yMin，即字体的下降值
    return -bbox.yMin;
}

void FT2Font::get_bitmap_offset(long *x, long *y)
{
    // 将 bbox 的 xMin 赋给传入指针 *x
    *x = bbox.xMin;
    // 将 0 赋给传入指针 *y
    *y = 0;
}

void FT2Font::draw_glyphs_to_bitmap(bool antialiased)
{
    // 计算图像的宽度和高度
    long width = (bbox.xMax - bbox.xMin) / 64 + 2;
    long height = (bbox.yMax - bbox.yMin) / 64 + 2;

    // 调整图像大小为计算出的宽度和高度
    image.resize(width, height);

    // 遍历所有字形
    for (size_t n = 0; n < glyphs.size(); n++) {
        // 将字形转换为位图，根据 antialiased 决定是否使用抗锯齿渲染模式
        FT_Error error = FT_Glyph_To_Bitmap(
            &glyphs[n], antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO, 0, 1);
        // 如果转换出错，抛出异常
        if (error) {
            throw_ft_error("Could not convert glyph to bitmap", error);
        }

        // 将当前字形转换为位图字形
        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[n];
        // 计算位图左上角在目标表面上的坐标位置

        // 位图左侧和顶部在像素中，字符串 bbox 在子像素中
        FT_Int x = (FT_Int)(bitmap->left - (bbox.xMin * (1. / 64.)));
        FT_Int y = (FT_Int)((bbox.yMax * (1. / 64.)) - bitmap->top + 1);

        // 将位图绘制到目标表面上的指定位置
        image.draw_bitmap(&bitmap->bitmap, x, y);
    }
}

void FT2Font::get_xys(bool antialiased, std::vector<double> &xys)
{
    // 遍历所有字形
    for (size_t n = 0; n < glyphs.size(); n++) {

        // 将字形转换为位图，根据 antialiased 决定是否使用抗锯齿渲染模式
        FT_Error error = FT_Glyph_To_Bitmap(
            &glyphs[n], antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO, 0, 1);
        // 如果转换出错，抛出异常
        if (error) {
            throw_ft_error("Could not convert glyph to bitmap", error);
        }

        // 将当前字形转换为位图字形
        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[n];

        // 位图左侧和顶部在像素中，字符串 bbox 在子像素中
        FT_Int x = (FT_Int)(bitmap->left - bbox.xMin * (1. / 64.));
        FT_Int y = (FT_Int)(bbox.yMax * (1. / 64.) - bitmap->top + 1);
        // 确保索引为非负数
        x = x < 0 ? 0 : x;
        y = y < 0 ? 0 : y;

        // 将计算得到的 x 和 y 坐标添加到 xys 向量中
        xys.push_back(x);
        xys.push_back(y);
    }
}

void FT2Font::draw_glyph_to_bitmap(FT2Image &im, int x, int y, size_t glyphInd, bool antialiased)
{
    FT_Vector sub_offset;
    sub_offset.x = 0; // int((xd - (double)x) * 64.0);
    sub_offset.y = 0; // int((yd - (double)y) * 64.0);

    // 如果 glyphInd 超出了 glyphs 的范围，抛出运行时错误
    if (glyphInd >= glyphs.size()) {
        throw std::runtime_error("glyph num is out of range");
    }
}
    // 将字形转换为位图
    FT_Error error = FT_Glyph_To_Bitmap(
      &glyphs[glyphInd],  // 要转换为位图的字形对象
      antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO,  // 渲染模式，是否抗锯齿
      &sub_offset,  // 附加的平移偏移量
      1  // 是否销毁原始图像
      );

    // 检查转换过程中是否出现错误
    if (error) {
        throw_ft_error("Could not convert glyph to bitmap", error);  // 如果有错误，抛出异常
    }

    // 将字形对象转换为位图字形对象
    FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[glyphInd];

    // 在指定位置绘制位图
    im.draw_bitmap(&bitmap->bitmap, x + bitmap->left, y);
}

void FT2Font::get_glyph_name(unsigned int glyph_number, char *buffer, bool fallback = false)
{
    // 如果开启了回退并且该字形编号在字形到字体映射中存在
    if (fallback && glyph_to_font.find(glyph_number) != glyph_to_font.end()) {
        // 只有父级 FT2Font 才有缓存
        FT2Font *ft_object = glyph_to_font[glyph_number];
        // 调用父级 FT2Font 的 get_glyph_name 方法，不再回退
        ft_object->get_glyph_name(glyph_number, buffer, false);
        return;
    }
    // 如果字体没有字形名称
    if (!FT_HAS_GLYPH_NAMES(face)) {
        /* 注意，此生成的名称必须与 ttconv 中 ttfont_CharStrings_getname 生成的名称匹配 */
        // 将字形编号格式化为 "uni%08x" 形式的名称
        PyOS_snprintf(buffer, 128, "uni%08x", glyph_number);
    } else {
        // 否则，从 FreeType 获取字形名称
        if (FT_Error error = FT_Get_Glyph_Name(face, glyph_number, buffer, 128)) {
            // 如果获取失败，抛出 FreeType 错误
            throw_ft_error("Could not get glyph names", error);
        }
    }
}

long FT2Font::get_name_index(char *name)
{
    // 返回给定名称在字体中的索引
    return FT_Get_Name_Index(face, (FT_String *)name);
}
```