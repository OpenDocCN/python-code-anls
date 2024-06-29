# `D:\src\scipysrc\matplotlib\extern\ttconv\truetype.h`

```py
/*
 * -*- mode: c; c-basic-offset: 4 -*-
 */

/*
 * Modified for use within matplotlib
 * 5 July 2007
 * Michael Droettboom
 */

#include <stdio.h>

/*
** ~ppr/src/include/typetype.h
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
** This include file is shared by the source files
** "pprdrv/pprdrv_tt.c" and "pprdrv/pprdrv_tt2.c".
**
** Last modified 19 April 1995.
*/

/* Types used in TrueType font files. */

// 定义字节类型
#define BYTE unsigned char
// 定义无符号短整型
#define USHORT unsigned short int
// 定义有符号短整型
#define SHORT short signed int
// 定义无符号长整型
#define ULONG unsigned int
// 定义有符号长整型
#define FIXED long signed int
// 定义有符号短整型
#define FWord short signed int
// 定义无符号短整型
#define uFWord short unsigned int

/* This structure stores a 16.16 bit fixed */
/* point number. */
// 定义存储16.16位固定点数的结构体
typedef struct
    {
    short int whole;                    /* 整数部分 */
    unsigned short int fraction;        /* 小数部分 */
    } Fixed;

/* This structure tells what we have found out about */
/* the current font. */
// 描述当前字体信息的结构体
struct TTFONT
    {
    // 用于实现最低级别的异常安全性的快速简单方法
    // Michael Droettboom 添加
    TTFONT();                           /* 构造函数 */
    ~TTFONT();                          /* 析构函数 */

    const char *filename;               /* TrueType 字体文件名 */
    FILE *file;                         /* 打开的 TrueType 文件 */
    font_type_enum  target_type;        /* 目标类型：42 或 3 用于 PS，-3 用于 PDF */

    ULONG numTables;                    /* 表的数量 */
    char *PostName;                     /* 字体的 PostScript 名称 */
    char *FullName;                     /* 字体的全名 */
    char *FamilyName;                   /* 字体的家族名称 */
    char *Style;                        /* 字体的风格字符串 */
    char *Copyright;                    /* 字体的版权信息 */
    char *Version;                      /* 字体的版本信息 */
    char *Trademark;                    /* 字体的商标信息 */
    int llx, lly, urx, ury;             /* 包围框 */

    Fixed TTVersion;                    /* TrueType 版本号 */
    Fixed MfrRevision;                  /* 字体的制造商修订号 */

    BYTE *offset_table;                 /* 内存中的偏移表 */
    BYTE *post_table;                   /* 内存中的 'post' 表 */

    BYTE *loca_table;                   /* 内存中的 'loca' 表 */
    BYTE *glyf_table;                   /* 内存中的 'glyf' 表 */
    BYTE *hmtx_table;                   /* 内存中的 'hmtx' 表 */

    USHORT numberOfHMetrics;            /* HMetrics 数量 */
    int unitsPerEm;                     /* 每 EM 单位数值 */
    int HUPM;                           /* 上述值的一半 */

    int numGlyphs;                      /* 'post' 表中的字形数量 */

    int indexToLocFormat;               /* 索引到位置的格式：短或长偏移 */
};
/* 定义一个函数，用于从字节序列中获取一个无符号长整型数 */
ULONG getULONG(BYTE *p);

/* 定义一个函数，用于从字节序列中获取一个无符号短整型数 */
USHORT getUSHORT(BYTE *p);

/* 定义一个函数，用于从字节序列中获取一个固定点数 */
Fixed getFixed(BYTE *p);

/*
** 获取一个 funits 字，因为它是 16 位长，所以可以使用 getUSHORT() 来实现实际工作。
*/
#define getFWord(x) (FWord)getUSHORT(x)
#define getuFWord(x) (uFWord)getUSHORT(x)

/*
** 通过将 USHORT 强制转换为有符号 SHORT，可以获取一个 SHORT。
*/
#define getSHORT(x) (SHORT)getUSHORT(x)

/* 这是 pprdrv_tt.c 中唯一一个从 pprdrv_tt.c 被调用的函数。 */
const char *ttfont_CharStrings_getname(struct TTFONT *font, int charindex);

/* 处理类型 3 字符的函数 */
void tt_type3_charproc(TTStreamWriter& stream, struct TTFONT *font, int charindex);

/* 添加于 06-07-07 Michael Droettboom */
/* 向字体中添加字形依赖项 */
void ttfont_add_glyph_dependencies(struct TTFONT *font, std::vector<int>& glypy_ids);

/* 该函数将字体字符坐标系中的数值转换为 1000 单位字符系统中的数值。 */
#define topost(x) (int)( ((int)(x) * 1000 + font->HUPM) / font->unitsPerEm )
#define topost2(x) (int)( ((int)(x) * 1000 + font.HUPM) / font.unitsPerEm )

/* 复合字形的标志值 */
#define ARG_1_AND_2_ARE_WORDS 1
#define ARGS_ARE_XY_VALUES 2
#define ROUND_XY_TO_GRID 4
#define WE_HAVE_A_SCALE 8
/* 保留 16 */
#define MORE_COMPONENTS 32
#define WE_HAVE_AN_X_AND_Y_SCALE 64
#define WE_HAVE_A_TWO_BY_TWO 128
#define WE_HAVE_INSTRUCTIONS 256
#define USE_MY_METRICS 512

/* 文件结尾 */
```