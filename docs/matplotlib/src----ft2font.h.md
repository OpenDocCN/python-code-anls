# `D:\src\scipysrc\matplotlib\src\ft2font.h`

```
/* -*- mode: c++; c-basic-offset: 4 -*- */

/* A python interface to FreeType */
#pragma once

#ifndef MPL_FT2FONT_H
#define MPL_FT2FONT_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

extern "C" {
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include FT_OUTLINE_H
#include FT_SFNT_NAMES_H
#include FT_TYPE1_TABLES_H
#include FT_TRUETYPE_TABLES_H
}

/*
 By definition, FT_FIXED as 2 16bit values stored in a single long.
 */
#define FIXED_MAJOR(val) (signed short)((val & 0xffff0000) >> 16)
#define FIXED_MINOR(val) (unsigned short)(val & 0xffff)

// the FreeType string rendered into a width, height buffer
class FT2Image
{
  public:
    // 默认构造函数
    FT2Image();
    // 指定宽高的构造函数
    FT2Image(unsigned long width, unsigned long height);
    virtual ~FT2Image();

    // 调整图像大小
    void resize(long width, long height);
    // 在指定位置绘制位图
    void draw_bitmap(FT_Bitmap *bitmap, FT_Int x, FT_Int y);
    // 绘制空心矩形
    void draw_rect(unsigned long x0, unsigned long y0, unsigned long x1, unsigned long y1);
    // 绘制填充矩形
    void draw_rect_filled(unsigned long x0, unsigned long y0, unsigned long x1, unsigned long y1);

    // 返回图像缓冲区
    unsigned char *get_buffer()
    {
        return m_buffer;
    }
    // 返回图像宽度
    unsigned long get_width()
    {
        return m_width;
    }
    // 返回图像高度
    unsigned long get_height()
    {
        return m_height;
    }

  private:
    bool m_dirty;             // 标记图像是否脏了
    unsigned char *m_buffer;  // 图像缓冲区指针
    unsigned long m_width;    // 图像宽度
    unsigned long m_height;   // 图像高度

    // 防止复制
    FT2Image(const FT2Image &);
    FT2Image &operator=(const FT2Image &);
};

extern FT_Library _ft2Library;  // FreeType 库对象声明

class FT2Font
{

  public:
    // 构造函数，接受打开参数、提示因子和备用字体列表
    FT2Font(FT_Open_Args &open_args, long hinting_factor, std::vector<FT2Font *> &fallback_list);
    virtual ~FT2Font();
    // 清空字体对象
    void clear();
    // 设置字体大小和 DPI
    void set_size(double ptsize, double dpi);
    // 设置字符映射
    void set_charmap(int i);
    // 选择字符映射
    void select_charmap(unsigned long i);
    // 设置文本及相关参数
    void set_text(
        size_t N, uint32_t *codepoints, double angle, FT_Int32 flags, std::vector<double> &xys);
    // 获取字符间距
    int get_kerning(FT_UInt left, FT_UInt right, FT_UInt mode, bool fallback);
    // 获取字符间距
    int get_kerning(FT_UInt left, FT_UInt right, FT_UInt mode, FT_Vector &delta);
    // 设置字符间距因子
    void set_kerning_factor(int factor);
    // 加载字符
    void load_char(long charcode, FT_Int32 flags, FT2Font *&ft_object, bool fallback);
    // 加载字符并处理备用字体
    bool load_char_with_fallback(FT2Font *&ft_object_with_glyph,
                                 FT_UInt &final_glyph_index,
                                 std::vector<FT_Glyph> &parent_glyphs,
                                 std::unordered_map<long, FT2Font *> &parent_char_to_font,
                                 std::unordered_map<FT_UInt, FT2Font *> &parent_glyph_to_font,
                                 long charcode,
                                 FT_Int32 flags,
                                 FT_Error &charcode_error,
                                 FT_Error &glyph_error,
                                 std::set<FT_String*> &glyph_seen_fonts,
                                 bool override);
    // 声明一个函数，加载指定字形的数据，并可选指定是否使用后备字体
    void load_glyph(FT_UInt glyph_index, FT_Int32 flags, FT2Font *&ft_object, bool fallback);

    // 声明一个函数，加载指定字形的数据，不使用后备字体
    void load_glyph(FT_UInt glyph_index, FT_Int32 flags);

    // 声明一个函数，获取字形的宽度和高度
    void get_width_height(long *width, long *height);

    // 声明一个函数，获取位图偏移量
    void get_bitmap_offset(long *x, long *y);

    // 声明一个函数，获取字形的下降值
    long get_descent();

    // 声明一个函数，根据抗锯齿选项获取一组坐标值
    // 使用引用参数 xys 来返回结果
    void get_xys(bool antialiased, std::vector<double> &xys);

    // 声明一个函数，将多个字形绘制到位图中
    void draw_glyphs_to_bitmap(bool antialiased);

    // 声明一个函数，将单个字形绘制到位图中
    void draw_glyph_to_bitmap(FT2Image &im, int x, int y, size_t glyphInd, bool antialiased);

    // 声明一个函数，根据字形编号获取字形名称
    void get_glyph_name(unsigned int glyph_number, char *buffer, bool fallback);

    // 声明一个函数，根据字体名称获取名称索引
    long get_name_index(char *name);

    // 声明一个函数，根据字符编码获取字符索引
    FT_UInt get_char_index(FT_ULong charcode, bool fallback);

    // 声明一个函数，获取字体路径
    PyObject* get_path();

    // 声明一个函数，根据字符编码获取字符的后备索引
    bool get_char_fallback_index(FT_ULong charcode, int& index) const;

    // 返回当前字体对象的引用
    FT_Face const &get_face() const
    {
        return face;
    }

    // 返回当前图像对象的引用
    FT2Image &get_image()
    {
        return image;
    }

    // 返回最后一个字形对象的引用
    FT_Glyph const &get_last_glyph() const
    {
        return glyphs.back();
    }

    // 返回最后一个字形在数组中的索引
    size_t get_last_glyph_index() const
    {
        return glyphs.size() - 1;
    }

    // 返回字形数组的大小
    size_t get_num_glyphs() const
    {
        return glyphs.size();
    }

    // 返回提示因子
    long get_hinting_factor() const
    {
        return hinting_factor;
    }

    // 返回字体是否具有字距调整
    FT_Bool has_kerning() const
    {
        return FT_HAS_KERNING(face);
    }

  private:
    FT2Image image;                          // 字体图像对象
    FT_Face face;                            // FreeType 字体对象
    FT_Vector pen;                           // 未转换的原点位置
    std::vector<FT_Glyph> glyphs;            // 存储字形数据的数组
    std::vector<FT2Font *> fallbacks;        // 后备字体对象数组
    std::unordered_map<FT_UInt, FT2Font *> glyph_to_font;  // 字形到字体对象的映射表
    std::unordered_map<long, FT2Font *> char_to_font;       // 字符到字体对象的映射表
    FT_BBox bbox;                            // 字体包围盒
    FT_Pos advance;                          // 字形的推进值
    long hinting_factor;                     // 提示因子
    int kerning_factor;                      // 字距因子

    // 禁止复制构造函数
    FT2Font(const FT2Font &);

    // 禁止赋值运算符重载
    FT2Font &operator=(const FT2Font &);
};

#endif



// 结束了一个 C++ 的类定义的尾部

#endif
// 结束了条件编译指令的尾部
```