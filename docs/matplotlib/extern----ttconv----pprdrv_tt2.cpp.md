# `D:\src\scipysrc\matplotlib\extern\ttconv\pprdrv_tt2.cpp`

```
/*
 * Modified for use within matplotlib
 * 5 July 2007
 * Michael Droettboom
 */
/*
** ~ppr/src/pprdrv/pprdrv_tt2.c
** Copyright 1995, Trinity College Computing Center.
** Written by David Chappell.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
** TrueType font support.  These functions allow PPR to generate
** PostScript fonts from Microsoft compatible TrueType font files.
**
** The functions in this file do most of the work to convert a
** TrueType font to a type 3 PostScript font.
**
** Most of the material in this file is derived from a program called
** "ttf2ps" which L. S. Ng posted to the usenet news group
** "comp.sources.postscript".  The author did not provide a copyright
** notice or indicate any restrictions on use.
**
** Last revised 11 July 1995.
*/

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <memory>
#include "pprdrv.h"
#include "truetype.h"
#include <algorithm>
#include <stack>
#include <list>

// Class definition for converting TrueType glyphs to Type 3 PostScript fonts
class GlyphToType3
{
private:
    GlyphToType3& operator=(const GlyphToType3& other);   // Private assignment operator to prevent copying
    GlyphToType3(const GlyphToType3& other);              // Private copy constructor

    /* The PostScript bounding box. */
    int llx,lly,urx,ury;        // Coordinates of the PostScript bounding box
    int advance_width;          // Width of the character advance

    /* Variables to hold the character data. */
    int *epts_ctr;              // Array of contour endpoints
    int num_pts, num_ctr;       // Number of points and contours
    FWord *xcoor, *ycoor;       // Arrays of x and y coordinates
    BYTE *tt_flags;             // Array of TrueType flags

    int stack_depth;            // Stack depth for tracking PS stack state

    // Private methods for handling PS operations
    void load_char(TTFONT* font, BYTE *glyph);                 // Load character data from TrueType font
    void stack(TTStreamWriter& stream, int new_elem);         // Stack operation for PS
    void stack_end(TTStreamWriter& stream);                   // End stack operation
    void PSConvert(TTStreamWriter& stream);                    // Convert to Type 3 PS font
    void PSCurveto(TTStreamWriter& stream,
                   FWord x0, FWord y0,
                   FWord x1, FWord y1,
                   FWord x2, FWord y2);                       // PS curve operation
    void PSMoveto(TTStreamWriter& stream, int x, int y);      // PS move operation
    void PSLineto(TTStreamWriter& stream, int x, int y);      // PS line operation
    void do_composite(TTStreamWriter& stream, struct TTFONT *font, BYTE *glyph);  // Handle composite glyphs

public:
    // Constructor for GlyphToType3 class
    GlyphToType3(TTStreamWriter& stream, struct TTFONT *font, int charindex, bool embedded = false);
    // Destructor for GlyphToType3 class
    ~GlyphToType3();
};

// Each point on a TrueType contour is either on the path or off it (a
// control point); here's a simple representation for building such
// contours. Added by Jouni Seppänen 2012-05-27.
enum Flag { ON_PATH, OFF_PATH };
struct FlaggedPoint
{
    enum Flag flag;     // Flag indicating if point is on path or off path
    FWord x;            // x-coordinate of the point
    FWord y;            // y-coordinate of the point
    # 定义带有标志、x 坐标和 y 坐标的构造函数，用于初始化 FlaggedPoint 对象
    FlaggedPoint(Flag flag_, FWord x_, FWord y_): flag(flag_), x(x_), y(y_) {};
};

/*
** This routine is used to break the character
** procedure up into a number of smaller
** procedures.  This is necessary so as not to
** overflow the stack on certain level 1 interpreters.
**
** Prepare to push another item onto the stack,
** starting a new procedure if necessary.
**
** Not all the stack depth calculations in this routine
** are perfectly accurate, but they do the job.
*/

void GlyphToType3::stack(TTStreamWriter& stream, int new_elem)
{
    // 如果点数较多才执行下面的操作
    if ( num_pts > 25 )  /* Only do something of we will have a log of points. */
    {
        // 如果栈深度为0，表示当前没有开始过程，需要开始一个新过程
        if (stack_depth == 0)
        {
            stream.put_char('{');
            stack_depth=1;
        }

        stack_depth += new_elem;                /* Account for what we propose to add */

        // 如果栈深度超过100，重新开始一个新过程，并估算新的栈深度
        if (stack_depth > 100)
        {
            stream.puts("}_e{");
            stack_depth = 3 + new_elem; /* A rough estimate */
        }
    }
} /* end of stack() */

void GlyphToType3::stack_end(TTStreamWriter& stream)                    /* called at end */
{
    // 如果栈深度不为0，表示有未结束的过程，需要结束最后的过程
    if ( stack_depth )
    {
        stream.puts("}_e");
        stack_depth=0;
    }
} /* end of stack_end() */

/*
** We call this routine to emit the PostScript code
** for the character we have loaded with load_char().
*/
void GlyphToType3::PSConvert(TTStreamWriter& stream)
{
    int j, k;

    /* Step thru the contours.
     * j = index to xcoor, ycoor, tt_flags (point data)
     * k = index to epts_ctr (which points belong to the same contour) */
    for(j = k = 0; k < num_ctr; k++)
    }

    /* Now, we can fill the whole thing. */
    stack(stream, 1);
    stream.puts("_cl");
} /* end of PSConvert() */

void GlyphToType3::PSMoveto(TTStreamWriter& stream, int x, int y)
{
    // 输出移动到指定坐标的PostScript命令
    stream.printf("%d %d _m\n", x, y);
}

void GlyphToType3::PSLineto(TTStreamWriter& stream, int x, int y)
{
    // 输出从当前点到指定坐标的PostScript命令
    stream.printf("%d %d _l\n", x, y);
}

/*
** Emit a PostScript "curveto" command, assuming the current point
** is (x0, y0), the control point of a quadratic spline is (x1, y1),
** and the endpoint is (x2, y2). Note that this requires a conversion,
** since PostScript splines are cubic.
*/
void GlyphToType3::PSCurveto(TTStreamWriter& stream,
                             FWord x0, FWord y0,
                             FWord x1, FWord y1,
                             FWord x2, FWord y2)
{
    double sx[3], sy[3], cx[3], cy[3];

    // 设置坐标参数
    sx[0] = x0;
    sy[0] = y0;
    sx[1] = x1;
    sy[1] = y1;
    sx[2] = x2;
    sy[2] = y2;

    // 计算控制点和终点的坐标
    cx[0] = (2*sx[1]+sx[0])/3;
    cy[0] = (2*sy[1]+sy[0])/3;
    cx[1] = (sx[2]+2*sx[1])/3;
    cy[1] = (sy[2]+2*sy[1])/3;
    cx[2] = sx[2];
    cy[2] = sy[2];

    // 输出PostScript的"curveto"命令
    stream.printf("%d %d %d %d %d %d _c\n",
                  (int)cx[0], (int)cy[0], (int)cx[1], (int)cy[1],
                  (int)cx[2], (int)cy[2]);
}

/*
** Deallocate the structures which stored
** the data for the last simple glyph.
*/
GlyphToType3::~GlyphToType3()
{
    // 释放存储最后一个简单字形数据的结构体
    free(tt_flags);            /* The flags array */
}
    # 释放内存：释放存储X坐标的数组
    free(xcoor);               /* The X coordinates */
    # 释放内存：释放存储Y坐标的数组
    free(ycoor);               /* The Y coordinates */
    # 释放内存：释放存储轮廓终点的数组
    free(epts_ctr);            /* The array of contour endpoints */
/*
** Load the simple glyph data pointed to by glyph.
** The pointer "glyph" should point 10 bytes into
** the glyph data.
*/
void GlyphToType3::load_char(TTFONT* font, BYTE *glyph)
{
    int x;
    BYTE c, ct;

    /* Read the contour endpoints list. */
    // 分配内存以存储轮廓端点列表
    epts_ctr = (int *)calloc(num_ctr,sizeof(int));
    for (x = 0; x < num_ctr; x++)
    {
        // 从字节流中读取轮廓端点并存储
        epts_ctr[x] = getUSHORT(glyph);
        glyph += 2;
    }

    /* From the endpoint of the last contour, we can */
    /* determine the number of points. */
    // 根据最后一个轮廓的端点确定点的总数
    num_pts = epts_ctr[num_ctr-1]+1;
#ifdef DEBUG_TRUETYPE
    // 调试信息，打印点的总数
    debug("num_pts=%d",num_pts);
    stream.printf("%% num_pts=%d\n",num_pts);
#endif

    /* Skip the instructions. */
    // 跳过指令部分
    x = getUSHORT(glyph);
    glyph += 2;
    glyph += x;

    /* Allocate space to hold the data. */
    // 分配空间以存储数据
    tt_flags = (BYTE *)calloc(num_pts,sizeof(BYTE));
    xcoor = (FWord *)calloc(num_pts,sizeof(FWord));
    ycoor = (FWord *)calloc(num_pts,sizeof(FWord));

    /* Read the flags array, uncompressing it as we go. */
    /* There is danger of overflow here. */
    // 读取标志数组，同时解压缩它们
    for (x = 0; x < num_pts; )
    {
        tt_flags[x++] = c = *(glyph++);

        if (c&8)                /* If next byte is repeat count, */
        {
            ct = *(glyph++);

            if ( (x + ct) > num_pts )
            {
                // 如果溢出，抛出异常
                throw TTException("Error in TT flags");
            }

            while (ct--)
            {
                // 复制标志直到重复计数为零
                tt_flags[x++] = c;
            }
        }
    }

    /* Read the x coordinates */
    // 读取 x 坐标
    for (x = 0; x < num_pts; x++)
    {
        if (tt_flags[x] & 2)            /* one byte value with */
        {
            /* external sign */
            // 如果使用外部符号，读取一个字节值
            c = *(glyph++);
            xcoor[x] = (tt_flags[x] & 0x10) ? c : (-1 * (int)c);
        }
        else if (tt_flags[x] & 0x10)    /* repeat last */
        {
            // 如果重复最后一个值，x 坐标设为零
            xcoor[x] = 0;
        }
        else                            /* two byte signed value */
        {
            // 否则读取两个字节的有符号值
            xcoor[x] = getFWord(glyph);
            glyph+=2;
        }
    }

    /* Read the y coordinates */
    // 读取 y 坐标
    for (x = 0; x < num_pts; x++)
    {
        if (tt_flags[x] & 4)            /* one byte value with */
        {
            /* external sign */
            // 如果使用外部符号，读取一个字节值
            c = *(glyph++);
            ycoor[x] = (tt_flags[x] & 0x20) ? c : (-1 * (int)c);
        }
        else if (tt_flags[x] & 0x20)    /* repeat last value */
        {
            // 如果重复最后一个值，y 坐标设为零
            ycoor[x] = 0;
        }
        else                            /* two byte signed value */
        {
            // 否则读取两个字节的有符号值
            ycoor[x] = getUSHORT(glyph);
            glyph+=2;
        }
    }

    /* Convert delta values to absolute values. */
    // 将增量值转换为绝对值
    for (x = 1; x < num_pts; x++)
    {
        xcoor[x] += xcoor[x-1];
        ycoor[x] += ycoor[x-1];
    }

    // 调用 topost 函数将坐标转换为 PostScript 单位
    for (x=0; x < num_pts; x++)
    {
        xcoor[x] = topost(xcoor[x]);
        ycoor[x] = topost(ycoor[x]);
    }

} /* end of load_char() */

/*
** Emmit PostScript code for a composite character.
*/
/*
** 处理复合字形中的每个组件。
*/
void GlyphToType3::do_composite(TTStreamWriter& stream, struct TTFONT *font, BYTE *glyph)
{
    USHORT flags;
    USHORT glyphIndex;
    int arg1;
    int arg2;

    /* 对于每个组件执行一次循环。*/
    do
    {
        flags = getUSHORT(glyph);       /* 读取标志字 */
        glyph += 2;

        glyphIndex = getUSHORT(glyph);  /* 读取字形索引字 */
        glyph += 2;

        if (flags & ARG_1_AND_2_ARE_WORDS)
        {
            /* tt 规范似乎表明这些值是有符号的。*/
            arg1 = getSHORT(glyph);
            glyph += 2;
            arg2 = getSHORT(glyph);
            glyph += 2;
        }
        else                    /* tt 规范并未明确指出 */
        {
            /* 这些值是否为有符号的。*/
            arg1 = *(signed char *)(glyph++);
            arg2 = *(signed char *)(glyph++);
        }

        if (flags & WE_HAVE_A_SCALE)
        {
            glyph += 2;
        }
        else if (flags & WE_HAVE_AN_X_AND_Y_SCALE)
        {
            glyph += 4;
        }
        else if (flags & WE_HAVE_A_TWO_BY_TWO)
        {
            glyph += 8;
        }
        else
        {
            // 如果没有上述标志，则不做任何操作。
        }

        /* 调试输出 */
#ifdef DEBUG_TRUETYPE
        stream.printf("%% flags=%d, arg1=%d, arg2=%d\n",
                      (int)flags,arg1,arg2);
#endif

        /* 如果有 (X,Y) 偏移且非零，翻译坐标系。*/
        if ( flags & ARGS_ARE_XY_VALUES )
        {
            if ( arg1 != 0 || arg2 != 0 )
                stream.printf("gsave %d %d translate\n", topost(arg1), topost(arg2) );
        }
        else
        {
            stream.printf("%% 未实现的偏移，arg1=%d, arg2=%d\n",arg1,arg2);
        }

        /* 调用 CharStrings 程序打印组件。*/
        stream.printf("false CharStrings /%s get exec\n",
                      ttfont_CharStrings_getname(font, glyphIndex));

        /* 如果我们翻译了坐标系，则恢复到原来的状态。*/
        if ( flags & ARGS_ARE_XY_VALUES && (arg1 != 0 || arg2 != 0) )
        {
            stream.puts("grestore ");
        }

    }
    while (flags & MORE_COMPONENTS);

} /* end of do_composite() */

/*
** 返回指定字形数据的指针。
*/
BYTE *find_glyph_data(struct TTFONT *font, int charindex)
{
    ULONG off;
    ULONG length;

    /* 从索引到位置表中读取字形的偏移量。*/
    if (font->indexToLocFormat == 0)
    {
        off = getUSHORT( font->loca_table + (charindex * 2) );
        off *= 2;
        length = getUSHORT( font->loca_table + ((charindex+1) * 2) );
        length *= 2;
        length -= off;
    }
    else
    {
        off = getULONG( font->loca_table + (charindex * 4) );
        length = getULONG( font->loca_table + ((charindex+1) * 4) );
        length -= off;
    }

    if (length > 0)
    {
        return font->glyf_table + off;
    }
    else
    {
        返回一个空指针 (NULL)，类型为 BYTE*。
        这是一个简单的返回语句，用于在某些情况下返回空指针。
    }
} /* end of find_glyph_data() */



GlyphToType3::GlyphToType3(TTStreamWriter& stream, struct TTFONT *font, int charindex, bool embedded /* = false */)
{
    BYTE *glyph;

    // 初始化成员变量
    tt_flags = NULL;
    xcoor = NULL;
    ycoor = NULL;
    epts_ctr = NULL;
    stack_depth = 0;

    /* 获取字形数据的指针。*/
    glyph = find_glyph_data(font, charindex);

    /* 如果字符为空白，则没有边界框，否则读取边界框。*/
    if (glyph == (BYTE*)NULL)
    {
        llx = lly = urx = ury = 0;   /* 空白字符的边界框全为零 */
        num_ctr = 0;                 /* 设置为后续的 if() */
    }
    else
    {
        /* 读取轮廓数。*/
        num_ctr = getSHORT(glyph);

        /* 读取后期处理的边界框。*/
        llx = getFWord(glyph + 2);
        lly = getFWord(glyph + 4);
        urx = getFWord(glyph + 6);
        ury = getFWord(glyph + 8);

        /* 推进指针。*/
        glyph += 10;
    }

    /* 如果是简单字符，加载其数据。*/
    if (num_ctr > 0)
    {
        load_char(font, glyph);
    }
    else
    {
        num_pts = 0;
    }

    /* 根据水平度量表确定字符的宽度。*/
    if (charindex < font->numberOfHMetrics)
    {
        advance_width = getuFWord(font->hmtx_table + (charindex * 4));
    }
    else
    {
        advance_width = getuFWord(font->hmtx_table + ((font->numberOfHMetrics - 1) * 4));
    }

    /* 执行 setcachedevice 以通知字体机制字符的边界框和前进宽度。*/
    stack(stream, 7);
    if (font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.printf("pop gsave .001 .001 scale %d 0 %d %d %d %d setcachedevice\n",
                      topost(advance_width),
                      topost(llx), topost(lly), topost(urx), topost(ury));
    }
    else
    {
        stream.printf("%d 0 %d %d %d %d _sc\n",
                      topost(advance_width),
                      topost(llx), topost(lly), topost(urx), topost(ury));
    }

    /* 如果是简单字形，则转换它，否则完成堆栈操作。*/
    if (num_ctr > 0)          /* 简单 */
    {
        PSConvert(stream);
    }
    else if (num_ctr < 0)     /* 复合 */
    {
        do_composite(stream, font, glyph);
    }

    /* 如果是混合类型字体，恢复初始状态。*/
    if (font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.printf("\ngrestore\n");
    }

    stack_end(stream);
}



/*
** This is the routine which is called from pprdrv_tt.c.
*/
void tt_type3_charproc(TTStreamWriter& stream, struct TTFONT *font, int charindex)
{
    GlyphToType3 glyph(stream, font, charindex);
} /* end of tt_type3_charproc() */

/*
** Some of the given glyph ids may refer to composite glyphs.
** This function adds all of the dependencies of those composite
** glyphs to the glyph id vector.  Michael Droettboom [06-07-07]
*/
// 对传入的 glyph_ids 向量进行排序，以便后续的处理
std::sort(glyph_ids.begin(), glyph_ids.end());

// 创建一个存放 glyph_id 的栈，初始放入所有的 glyph_id
std::stack<int> glyph_stack;
for (std::vector<int>::iterator i = glyph_ids.begin();
        i != glyph_ids.end(); ++i)
{
    glyph_stack.push(*i);
}

// 当栈非空时，持续处理栈顶的 glyph_id
while (glyph_stack.size())
{
    // 弹出栈顶的 glyph_id
    int gind = glyph_stack.top();
    glyph_stack.pop();

    // 查找对应的字形数据
    BYTE* glyph = find_glyph_data(font, gind);
    if (glyph != (BYTE*)NULL)
    {
        // 获取字形控制器的数量
        int num_ctr = getSHORT(glyph);
        // 如果数量小于等于 0，则这是一个复合字形
        if (num_ctr <= 0)
        {
            // 移动到复合字形数据的开始处
            glyph += 10;
            USHORT flags = 0;

            // 处理复合字形中的组成部分
            do
            {
                // 获取组成部分的标志
                flags = getUSHORT(glyph);
                glyph += 2;
                // 获取组成部分使用的字形索引
                gind = (int)getUSHORT(glyph);
                glyph += 2;

                // 在排序后的 glyph_ids 向量中查找 gind 的插入位置
                std::vector<int>::iterator insertion =
                    std::lower_bound(glyph_ids.begin(), glyph_ids.end(), gind);
                // 如果找不到 gind，则将其插入到向量中，并将其添加到栈中继续处理
                if (insertion == glyph_ids.end() || *insertion != gind)
                {
                    glyph_ids.insert(insertion, gind);
                    glyph_stack.push(gind);
                }

                // 根据标志位进行不同类型的数据跳过
                if (flags & ARG_1_AND_2_ARE_WORDS)
                {
                    glyph += 4;
                }
                else
                {
                    glyph += 2;
                }

                if (flags & WE_HAVE_A_SCALE)
                {
                    glyph += 2;
                }
                else if (flags & WE_HAVE_AN_X_AND_Y_SCALE)
                {
                    glyph += 4;
                }
                else if (flags & WE_HAVE_A_TWO_BY_TWO)
                {
                    glyph += 8;
                }
            }
            while (flags & MORE_COMPONENTS);
        }
    }
}
```