# `D:\src\scipysrc\matplotlib\extern\ttconv\pprdrv_tt.cpp`

```
/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
 * Modified for use within matplotlib
 * 5 July 2007
 * Michael Droettboom
 */

/*
** ~ppr/src/pprdrv/pprdrv_tt.c
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
** Last revised 19 December 1995.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pprdrv.h"
#include "truetype.h"
#include <sstream>
#ifdef _POSIX_C_SOURCE
#    undef _POSIX_C_SOURCE
#endif
#ifndef _AIX
#ifdef _XOPEN_SOURCE
#    undef _XOPEN_SOURCE
#endif
#endif
#include <Python.h>

/*==========================================================================
** Convert the indicated Truetype font file to a type 42 or type 3
** PostScript font and insert it in the output stream.
**
** All the routines from here to the end of file file are involved
** in this process.
==========================================================================*/

/*---------------------------------------
** Endian conversion routines.
** These routines take a BYTE pointer
** and return a value formed by reading
** bytes starting at that point.
**
** These routines read the big-endian
** values which are used in TrueType
** font files.
---------------------------------------*/

/*
** Get an Unsigned 32 bit number.
*/
ULONG getULONG(BYTE *p)
{
    int x;
    ULONG val=0;

    for (x=0; x<4; x++)
    {
        val *= 0x100;
        val += p[x];
    }

    return val;
} /* end of getULONG() */

/*
** Get an unsigned 16 bit number.
*/
USHORT getUSHORT(BYTE *p)
{
    int x;
    USHORT val=0;

    for (x=0; x<2; x++)
    {
        val *= 0x100;
        val += p[x];
    }

    return val;
} /* end of getUSHORT() */

/*
** Get a 32 bit fixed point (16.16) number.
** A special structure is used to return the value.
*/
Fixed getFixed(BYTE *s)
{
    Fixed val={0,0};

    val.whole = ((s[0] * 256) + s[1]);
    val.fraction = ((s[2] * 256) + s[3]);

    return val;
} /* end of getFixed() */

/*-----------------------------------------------------------------------
** Load a TrueType font table into memory and return a pointer to it.
** The font's "file" and "offset_table" fields must be set before this
** routine is called.
**
** This first argument is a TrueType font structure, the second
** argument is the name of the table to retrieve.  A table name
** is always 4 characters, though the last characters may be
** padding spaces.
*/
/*-----------------------------------------------------------------------*/
/* 获取字体文件中指定名称的表格数据指针 */
BYTE *GetTable(struct TTFONT *font, const char *name)
{
    BYTE *ptr;                  // 指向表格目录项的指针
    ULONG x;                    // 循环计数器
    debug("GetTable(file,font,\"%s\")",name);

    /* 必须搜索表格目录 */
    ptr = font->offset_table + 12;  // 定位到表格目录的起始位置偏移12字节处
    x = 0;
    while (true)
    {
        if ( strncmp((const char*)ptr, name, 4) == 0 )
        {
            ULONG offset, length;   // 表格的偏移量和长度
            BYTE *table;            // 指向表格数据的指针

            offset = getULONG( ptr + 8 );   // 获取表格数据在文件中的偏移量
            length = getULONG( ptr + 12 );  // 获取表格数据的长度
            table = (BYTE*)calloc( sizeof(BYTE), length + 2 );   // 分配足够空间存放表格数据

            try
            {
                debug("Loading table \"%s\" from offset %d, %d bytes",name,offset,length);

                // 定位到文件中表格数据的偏移量处并读取数据
                if ( fseek( font->file, (long)offset, SEEK_SET ) )
                {
                    throw TTException("TrueType font may be corrupt (reason 3)");
                }

                // 读取表格数据到table缓冲区中
                if ( fread(table, sizeof(BYTE), length, font->file) != (sizeof(BYTE) * length) )
                {
                    throw TTException("TrueType font may be corrupt (reason 4)");
                }
            }
            catch (TTException&)
            {
                free(table);    // 释放表格数据的内存空间
                throw;          // 抛出异常
            }
            /* 总是添加NUL终止符；如果是UTF16字符串，则再添加一个字节 */
            table[length] = '\0';
            table[length + 1] = '\0';
            return table;    // 返回表格数据的指针
        }

        x++;
        ptr += 16;          // 移动到下一个表格目录项的起始位置
        if (x == font->numTables)
        {
            throw TTException("TrueType font is missing table");
        }
    }

} /* end of GetTable() */

static void utf16be_to_ascii(char *dst, char *src, size_t length) {
    ++src;
    for (; *src != 0 && length; dst++, src += 2, --length) {
        *dst = *src;
    }
}

/*--------------------------------------------------------------------*/
/* 读取 'name' 表格，获取其中的信息，并将信息存储在字体结构中 */
/*                                                                    */
/* 'name' 表格包含字体的名称和PostScript名称等信息。                  */
/*--------------------------------------------------------------------*/
void Read_name(struct TTFONT *font)
{
    BYTE *table_ptr, *ptr2;
    int numrecords;         // 此表格中的字符串数量
    BYTE *strings;          // 指向字符串存储区的指针
    int x;
    int platform;           // 当前平台ID
    int nameid;             // 名称ID
    int offset, length;     // 字符串的偏移量和长度
    debug("Read_name()");

    table_ptr = NULL;

    /* 设置默认值以避免将来对未定义指针的引用。为PostName、FullName、FamilyName、Version和Style分配内存，以便安全释放。 */
    for (char **ptr = &(font->PostName); ptr != NULL; )
    {
        // 为指针 *ptr 分配内存，大小为 "unknown" 字符串长度加上结束符号 '\0' 的空间
        *ptr = (char*) calloc(sizeof(char), strlen("unknown")+1);
        // 将 "unknown" 字符串复制到 *ptr 指向的内存空间中
        strcpy(*ptr, "unknown");
        // 检查当前 ptr 是否指向 font 结构体中的 PostName 字段的地址，如果是，则将 ptr 指向 FullName 字段的地址
        if (ptr == &(font->PostName)) ptr = &(font->FullName);
        // 如果 ptr 指向 FullName 字段的地址，则将 ptr 指向 FamilyName 字段的地址
        else if (ptr == &(font->FullName)) ptr = &(font->FamilyName);
        // 如果 ptr 指向 FamilyName 字段的地址，则将 ptr 指向 Version 字段的地址
        else if (ptr == &(font->FamilyName)) ptr = &(font->Version);
        // 如果 ptr 指向 Version 字段的地址，则将 ptr 指向 Style 字段的地址
        else if (ptr == &(font->Version)) ptr = &(font->Style);
        // 如果 ptr 不指向上述任何字段的地址，则将 ptr 设置为 NULL
        else ptr = NULL;
    }
    // 将 font 的 Copyright 和 Trademark 字段都设置为 NULL
    font->Copyright = font->Trademark = (char*)NULL;

    // 调用 GetTable 函数获取 font 结构体中 "name" 表的指针，table_ptr 指向该表的内容
    table_ptr = GetTable(font, "name");         /* pointer to table */
    try
    {
        // 如果发生 TTException 异常，则释放 table_ptr 指向的内存空间并重新抛出异常
        free(table_ptr);
        throw;
    }
    // 无论是否发生异常，最终都释放 table_ptr 指向的内存空间
    free(table_ptr);
} /* end of Read_name() */
/*---------------------------------------------------------------------
** Write the header for a PostScript font.
---------------------------------------------------------------------*/
void ttfont_header(TTStreamWriter& stream, struct TTFONT *font)
{
    int VMMin;
    int VMMax;

    /*
    ** To show that it is a TrueType font in PostScript format,
    ** we will begin the file with a specific string.
    ** This string also indicates the version of the TrueType
    ** specification on which the font is based and the
    ** font manufacturer's revision number for the font.
    */
    // 如果字体类型为 PS_TYPE_42 或 PS_TYPE_42_3_HYBRID，则输出特定的 TrueType 字体头部信息
    if ( font->target_type == PS_TYPE_42 ||
            font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.printf("%%!PS-TrueTypeFont-%d.%d-%d.%d\n",
                      font->TTVersion.whole, font->TTVersion.fraction,
                      font->MfrRevision.whole, font->MfrRevision.fraction);
    }

    /* If it is not a Type 42 font, we will use a different format. */
    // 如果不是 Type 42 字体，则使用另一种格式
    else
    {
        stream.putline("%!PS-Adobe-3.0 Resource-Font");
    }       /* See RBIIp 641 */

    /* We will make the title the name of the font. */
    // 将字体名称作为标题
    stream.printf("%%%%Title: %s\n",font->FullName);

    /* If there is a Copyright notice, put it here too. */
    // 如果有版权声明，则在此处输出
    if ( font->Copyright != (char*)NULL )
    {
        stream.printf("%%%%Copyright: %s\n",font->Copyright);
    }

    /* We created this file. */
    // 根据字体类型输出创建者信息
    if ( font->target_type == PS_TYPE_42 )
    {
        stream.putline("%%Creator: Converted from TrueType to type 42 by PPR");
    }
    else if (font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.putline("%%Creator: Converted from TypeType to type 42/type 3 hybrid by PPR");
    }
    else
    {
        stream.putline("%%Creator: Converted from TrueType to type 3 by PPR");
    }

    /* If VM usage information is available, print it. */
    // 如果有可用的 VM 使用信息，则输出
    if ( font->target_type == PS_TYPE_42 || font->target_type == PS_TYPE_42_3_HYBRID)
    {
        VMMin = (int)getULONG( font->post_table + 16 );
        VMMax = (int)getULONG( font->post_table + 20 );
        if ( VMMin > 0 && VMMax > 0 )
            stream.printf("%%%%VMUsage: %d %d\n",VMMin,VMMax);
    }

    /* Start the dictionary which will eventually */
    /* become the font. */
    // 根据字体类型开始定义字典
    if (font->target_type == PS_TYPE_42)
    {
        stream.putline("15 dict begin");
    }
    else
    {
        stream.putline("25 dict begin");

        /* Type 3 fonts will need some subroutines here. */
        // Type 3 字体在此处需要一些子程序
        stream.putline("/_d{bind def}bind def");
        stream.putline("/_m{moveto}_d");
        stream.putline("/_l{lineto}_d");
        stream.putline("/_cl{closepath eofill}_d");
        stream.putline("/_c{curveto}_d");
        stream.putline("/_sc{7 -1 roll{setcachedevice}{pop pop pop pop pop pop}ifelse}_d");
        stream.putline("/_e{exec}_d");
    }

    // 设置字体名称和绘制类型
    stream.printf("/FontName /%s def\n",font->PostName);
    stream.putline("/PaintType 0 def");
    // 如果字体的目标类型是PS_TYPE_42或PS_TYPE_42_3_HYBRID，则设置字体矩阵为[1 0 0 1 0 0]
    if (font->target_type == PS_TYPE_42 || font->target_type == PS_TYPE_42_3_HYBRID)
    {
        // 在输出流中写入字体矩阵的定义
        stream.putline("/FontMatrix[1 0 0 1 0 0]def");
    }
    else
    {
        // 否则，设置字体矩阵为[.001 0 0 .001 0 0]
        stream.putline("/FontMatrix[.001 0 0 .001 0 0]def");
    }

    // 在输出流中写入字体包围框的定义，使用字体对象的边界值进行格式化输出
    stream.printf("/FontBBox[%d %d %d %d]def\n",font->llx-1,font->lly-1,font->urx,font->ury);

    // 如果字体的目标类型是PS_TYPE_42或PS_TYPE_42_3_HYBRID，则设置字体类型为42
    if (font->target_type == PS_TYPE_42 || font->target_type == PS_TYPE_42_3_HYBRID)
    {
        // 在输出流中写入字体类型的定义为42
        stream.printf("/FontType 42 def\n", font->target_type );
    }
    else
    {
        // 否则，设置字体类型为3
        stream.printf("/FontType 3 def\n", font->target_type );
    }
} /* end of ttfont_header() */

/*-------------------------------------------------------------
** Define the encoding array for this font.
** Since we don't really want to deal with converting all of
** the possible font encodings in the wild to a standard PS
** one, we just explicitly create one for each font.
-------------------------------------------------------------*/
void ttfont_encoding(TTStreamWriter& stream, struct TTFONT *font, std::vector<int>& glyph_ids, font_type_enum target_type)
{
    if (target_type == PS_TYPE_3 || target_type == PS_TYPE_42_3_HYBRID)
    {
        // 开始定义字体的编码数组
        stream.printf("/Encoding [ ");

        // 遍历字形标识符列表
        for (std::vector<int>::const_iterator i = glyph_ids.begin();
                i != glyph_ids.end(); ++i)
        {
            // 获取字形名称
            const char* name = ttfont_CharStrings_getname(font, *i);
            // 输出字形名称到编码数组
            stream.printf("/%s ", name);
        }

        // 完成编码数组定义
        stream.printf("] def\n");
    }
    else
    {
        // 对于非 Type 3 或混合 Type 42/3 字体，采用标准编码
        stream.putline("/Encoding StandardEncoding def");
    }
} /* end of ttfont_encoding() */

/*-----------------------------------------------------------
** Create the optional "FontInfo" sub-dictionary.
-----------------------------------------------------------*/
void ttfont_FontInfo(TTStreamWriter& stream, struct TTFONT *font)
{
    Fixed ItalicAngle;

    /* We create a sub dictionary named "FontInfo" where we */
    /* store information which though it is not used by the */
    /* interpreter, is useful to some programs which will */
    /* be printing with the font. */
    // 创建名为 "FontInfo" 的子字典，存储与字体打印相关但不由解释器使用的信息
    stream.putline("/FontInfo 10 dict dup begin");

    /* These names come from the TrueType font's "name" table. */
    // 设置字体族名
    stream.printf("/FamilyName (%s) def\n",font->FamilyName);
    // 设置字体全名
    stream.printf("/FullName (%s) def\n",font->FullName);

    if ( font->Copyright != (char*)NULL || font->Trademark != (char*)NULL )
    {
        // 设置版权信息
        stream.printf("/Notice (%s",
                      font->Copyright != (char*)NULL ? font->Copyright : "");
        stream.printf("%s%s) def\n",
                      font->Trademark != (char*)NULL ? " " : "",
                      font->Trademark != (char*)NULL ? font->Trademark : "");
    }

    /* This information is not quite correct. */
    // 设置字体粗细信息
    stream.printf("/Weight (%s) def\n",font->Style);

    /* Some fonts have this as "version". */
    // 设置字体版本信息
    stream.printf("/Version (%s) def\n",font->Version);

    /* Some information from the "post" table. */
    // 设置斜体角度信息
    ItalicAngle = getFixed( font->post_table + 4 );
    stream.printf("/ItalicAngle %d.%d def\n",ItalicAngle.whole,ItalicAngle.fraction);
    // 设置是否固定间距
    stream.printf("/isFixedPitch %s def\n", getULONG( font->post_table + 12 ) ? "true" : "false" );
    // 设置下划线位置
    stream.printf("/UnderlinePosition %d def\n", (int)getFWord( font->post_table + 8 ) );
    // 设置下划线粗细
    stream.printf("/UnderlineThickness %d def\n", (int)getFWord( font->post_table + 10 ) );
    // 完成 "FontInfo" 子字典定义
    stream.putline("end readonly def");
} /* end of ttfont_FontInfo() */

/*-------------------------------------------------------------------
** sfnts routines
/*
** These routines generate the PostScript "sfnts" array which
** contains one or more strings which contain a reduced version
** of the TrueType font.
**
** A number of functions are required to accomplish this rather
** complicated task.
-------------------------------------------------------------------*/
// 定义变量：存储当前字符串长度、当前行长度、当前是否在字符串中
int string_len;
int line_len;
bool in_string;

/*
** This is called once at the start.
*/
// 开始 sfnts 数组的生成，输出起始标记 "/sfnts[<" 到流中
void sfnts_start(TTStreamWriter& stream)
{
    stream.puts("/sfnts[<");
    in_string=true;
    string_len=0;
    line_len=8;
} /* end of sfnts_start() */

/*
** Write a BYTE as a hexadecimal value as part of the sfnts array.
*/
// 将一个 BYTE 以十六进制形式写入 sfnts 数组
void sfnts_pputBYTE(TTStreamWriter& stream, BYTE n)
{
    static const char hexdigits[]="0123456789ABCDEF";

    if (!in_string)
    {
        stream.put_char('<');
        string_len=0;
        line_len++;
        in_string=true;
    }

    stream.put_char( hexdigits[ n / 16 ] );
    stream.put_char( hexdigits[ n % 16 ] );
    string_len++;
    line_len+=2;

    if (line_len > 70)
    {
        stream.put_char('\n');
        line_len=0;
    }

} /* end of sfnts_pputBYTE() */

/*
** Write a USHORT as a hexadecimal value as part of the sfnts array.
*/
// 将一个 USHORT 以十六进制形式写入 sfnts 数组
void sfnts_pputUSHORT(TTStreamWriter& stream, USHORT n)
{
    sfnts_pputBYTE(stream, n / 256);
    sfnts_pputBYTE(stream, n % 256);
} /* end of sfnts_pputUSHORT() */

/*
** Write a ULONG as part of the sfnts array.
*/
// 将一个 ULONG 写入 sfnts 数组
void sfnts_pputULONG(TTStreamWriter& stream, ULONG n)
{
    int x1,x2,x3;

    x1 = n % 256;
    n /= 256;
    x2 = n % 256;
    n /= 256;
    x3 = n % 256;
    n /= 256;

    sfnts_pputBYTE(stream, n);
    sfnts_pputBYTE(stream, x3);
    sfnts_pputBYTE(stream, x2);
    sfnts_pputBYTE(stream, x1);
} /* end of sfnts_pputULONG() */

/*
** This is called whenever it is
** necessary to end a string in the sfnts array.
**
** (The array must be broken into strings which are
** no longer than 64K characters.)
*/
// 在 sfnts 数组中结束当前字符串的写入
void sfnts_end_string(TTStreamWriter& stream)
{
    if (in_string)
    {
        string_len=0;           /* fool sfnts_pputBYTE() */

#ifdef DEBUG_TRUETYPE_INLINE
        puts("\n% dummy byte:\n");
#endif

        sfnts_pputBYTE(stream, 0);      /* extra byte for pre-2013 compatibility */
        stream.put_char('>');
        line_len++;
    }
    in_string=false;
} /* end of sfnts_end_string() */

/*
** This is called at the start of each new table.
** The argement is the length in bytes of the table
** which will follow.  If the new table will not fit
** in the current string, a new one is started.
*/
// 在开始每个新表格时调用，参数是即将跟随的表格的字节长度，
// 如果新表格无法放入当前字符串中，则开始新字符串
void sfnts_new_table(TTStreamWriter& stream, ULONG length)
{
    if ( (string_len + length) > 65528 )
        sfnts_end_string(stream);
} /* end of sfnts_new_table() */

/*
** We may have to break up the 'glyf' table.  That is the reason
** why we provide this special routine to copy it into the sfnts
** array.
*/
// 可能需要拆分 'glyf' 表，为此提供特殊的例程将其复制到 sfnts 数组中
void sfnts_glyf_table(TTStreamWriter& stream, struct TTFONT *font, ULONG oldoffset, ULONG correct_total_length)
{
    ULONG off;
    ULONG length;
    int c;
    ULONG total=0;              /* 记录写入表格的总字节数 */
    int x;
    bool loca_is_local=false;
    debug("sfnts_glyf_table(font,%d)", (int)correct_total_length);

    if (font->loca_table == NULL)
    {
        font->loca_table = GetTable(font,"loca");
        loca_is_local = true;
    }

    /* 将文件指针移动到正确的位置 */
    fseek( font->file, oldoffset, SEEK_SET );

    /* 逐个复制字形数据 */
    for (x=0; x < font->numGlyphs; x++)
    {
        /* 从索引到位置表中读取字形偏移量 */
        if (font->indexToLocFormat == 0)
        {
            off = getUSHORT( font->loca_table + (x * 2) );
            off *= 2;
            length = getUSHORT( font->loca_table + ((x+1) * 2) );
            length *= 2;
            length -= off;
        }
        else
        {
            off = getULONG( font->loca_table + (x * 4) );
            length = getULONG( font->loca_table + ((x+1) * 4) );
            length -= off;
        }
        debug("glyph length=%d",(int)length);

        /* 如果需要，开始新的字符串 */
        sfnts_new_table( stream, (int)length );

        /*
        ** 确保字形数据填充到两字节边界
        */
        if ( length % 2 ) {
            throw TTException("TrueType 字体的 'glyf' 表没有2字节填充");
        }

        /* 复制字形的字节数据 */
        while ( length-- )
        {
            if ( (c = fgetc(font->file)) == EOF ) {
                throw TTException("TrueType 字体可能损坏（原因 6）");
            }

            sfnts_pputBYTE(stream, c);
            total++;            /* 增加到总字节数 */
        }

    }

    if (loca_is_local)
    {
        free(font->loca_table);
        font->loca_table = NULL;
    }

    /* 填充到从表格目录中获取的正确总长度 */
    while ( total < correct_total_length )
    {
        sfnts_pputBYTE(stream, 0);
        total++;
    }
} /* end of sfnts_glyf_table() */

/*
** Here is the routine which ties it all together.
**
** Create the array called "sfnts" which
** holds the actual TrueType data.
*/
void ttfont_sfnts(TTStreamWriter& stream, struct TTFONT *font)
{
    static const char *table_names[] =  /* The names of all tables */
    {
        /* which it is worth while */
        "cvt ",                         /* to include in a Type 42 */
        "fpgm",                         /* PostScript font. */
        "glyf",
        "head",
        "hhea",
        "hmtx",
        "loca",
        "maxp",
        "prep"
    } ;

    struct                      /* The location of each of */
    {
        ULONG oldoffset;        /* the above tables. */
        ULONG newoffset;
        ULONG length;
        ULONG checksum;
    } tables[9];

    BYTE *ptr;                  /* A pointer into the origional table directory. */
    ULONG x,y;                  /* General use loop countes. */
    int c;                      /* Input character. */
    int diff;
    ULONG nextoffset;
    int count;                  /* How many `important' tables did we find? */

    ptr = font->offset_table + 12;
    nextoffset=0;
    count=0;

    /*
    ** Find the tables we want and store there vital
    ** statistics in tables[].
    */
    ULONG num_tables_read = 0;  /* Number of tables read from the directory */
    for (x = 0; x < 9; x++) {
        do {
          if (num_tables_read < font->numTables) {
              /* There are still tables to read from ptr */
              diff = strncmp((char*)ptr, table_names[x], 4);

              if (diff > 0) {           /* If we are past it. */
                  tables[x].length = 0;
                  diff = 0;
              } else if (diff < 0) {      /* If we haven't hit it yet. */
                  ptr += 16;
                  num_tables_read++;
              } else if (diff == 0) {     /* Here it is! */
                  tables[x].newoffset = nextoffset;
                  tables[x].checksum = getULONG( ptr + 4 );
                  tables[x].oldoffset = getULONG( ptr + 8 );
                  tables[x].length = getULONG( ptr + 12 );
                  nextoffset += ( ((tables[x].length + 3) / 4) * 4 );
                  count++;
                  ptr += 16;
                  num_tables_read++;
              }
          } else {
            /* We've read the whole table directory already */
            /* Some tables couldn't be found */
            tables[x].length = 0;
            break;  /* Proceed to next tables[x] */
          }
        } while (diff != 0);

    } /* end of for loop which passes over the table directory */

    /* Begin the sfnts array. */
    sfnts_start(stream);

    /* Generate the offset table header */
    /* Start by copying the TrueType version number. */
    ptr = font->offset_table;
    for (x=0; x < 4; x++)
    {
        sfnts_pputBYTE( stream,  *(ptr++) );
    }

    /* Now, generate those silly numTables numbers. */
    sfnts_pputUSHORT(stream, count);            /* 将表的数量写入流 */

    int search_range = 1;                       /* 初始化搜索范围 */
    int entry_sel = 0;                          /* 初始化条目选择器 */

    while (search_range <= count) {             /* 计算搜索范围和条目选择器 */
        search_range <<= 1;                     /* 左移一位，相当于乘以2 */
        entry_sel++;
    }
    entry_sel = entry_sel > 0 ? entry_sel - 1 : 0;  /* 根据条件更新条目选择器 */
    search_range = (search_range >> 1) * 16;    /* 计算搜索范围 */
    int range_shift = count * 16 - search_range; /* 计算范围偏移 */

    sfnts_pputUSHORT(stream, search_range);      /* 将搜索范围写入流 */
    sfnts_pputUSHORT(stream, entry_sel);         /* 将条目选择器写入流 */
    sfnts_pputUSHORT(stream, range_shift);       /* 将范围偏移写入流 */

    debug("only %d tables selected",count);      /* 调试信息，显示选择的表的数量 */

    /* 现在，生成表目录 */
    for (x=0; x < 9; x++)                       /* 遍历每个表 */
    {
        if ( tables[x].length == 0 )            /* 跳过缺失的表 */
        {
            continue;
        }

        /* 名称 */
        sfnts_pputBYTE( stream, table_names[x][0] );
        sfnts_pputBYTE( stream, table_names[x][1] );
        sfnts_pputBYTE( stream, table_names[x][2] );
        sfnts_pputBYTE( stream, table_names[x][3] );

        /* 校验和 */
        sfnts_pputULONG( stream, tables[x].checksum );

        /* 偏移 */
        sfnts_pputULONG( stream, tables[x].newoffset + 12 + (count * 16) );

        /* 长度 */
        sfnts_pputULONG( stream, tables[x].length );
    }

    /* 现在，发送表数据 */
    for (x=0; x < 9; x++)                       /* 遍历每个表 */
    {
        if ( tables[x].length == 0 )            /* 跳过不存在的表 */
        {
            continue;
        }
        debug("emmiting table '%s'",table_names[x]);   /* 调试信息，显示正在发送的表的名称 */

        /* 'glyf' 表需要特殊处理 */
        if ( strcmp(table_names[x],"glyf")==0 )
        {
            sfnts_glyf_table(stream,font,tables[x].oldoffset,tables[x].length);  /* 发送 'glyf' 表的数据 */
        }
        else                    /* 其他表不能超过65535字节的长度 */
        {
            /* 如果表长度超过65535字节，则抛出异常 */
            if ( tables[x].length > 65535 )
            {
                throw TTException("TrueType font has a table which is too long");
            }

            /* 如有必要，开始新的字符串 */
            sfnts_new_table(stream, tables[x].length);

            /* 在文件中定位到正确的位置 */
            fseek( font->file, tables[x].oldoffset, SEEK_SET );

            /* 复制表的字节数据 */
            for ( y=0; y < tables[x].length; y++ )
            {
                if ( (c = fgetc(font->file)) == EOF )   /* 如果文件结束，抛出异常 */
                {
                    throw TTException("TrueType font may be corrupt (reason 7)");
                }

                sfnts_pputBYTE(stream, c);       /* 将字节写入流 */
            }
        }

        /* 填充到四字节边界 */
        y=tables[x].length;
        while ( (y % 4) != 0 )
        {
            sfnts_pputBYTE(stream, 0);          /* 填充0字节到四字节边界 */
            y++;
        }
    }
#ifdef DEBUG_TRUETYPE_INLINE
            puts("\n% pad byte:\n");
#endif
        }
    # 下面是一个长字符串，包含大量的特殊字符和符号
    "Yacute","yacute","Thorn","thorn","minus","multiply","onesuperior",
    "twosuperior","threesuperior","onehalf","onequarter","threequarters","franc",
    "Gbreve","gbreve","Idot","Scedilla","scedilla","Cacute","cacute","Ccaron",
    "ccaron","dmacron","markingspace","capslock","shift","propeller","enter",
    "markingtabrtol","markingtabltor","control","markingdeleteltor",
    "markingdeletertol","option","escape","parbreakltor","parbreakrtol",
    "newpage","checkmark","linebreakltor","linebreakrtol","markingnobreakspace",
    "diamond","appleoutline"
/*
** This routine is called by the one below.
** It is also called from pprdrv_tt2.c
*/
const char *ttfont_CharStrings_getname(struct TTFONT *font, int charindex)
{
    int GlyphIndex;
    static char temp[80];   // 用于存储生成的字符名的静态缓冲区
    char *ptr;
    ULONG len;

    Fixed post_format;      // 存储 'post' 表的格式号

    /* The 'post' table format number. */
    post_format = getFixed( font->post_table );

    if ( post_format.whole != 2 || post_format.fraction != 0 )
    {
        /* We don't have a glyph name table, so generate a name.
           This generated name must match exactly the name that is
           generated by FT2Font in get_glyph_name */
        PyOS_snprintf(temp, 80, "uni%08x", charindex);   // 生成一个默认的字符名
        return temp;    // 返回生成的字符名
    }

    GlyphIndex = (int)getUSHORT( font->post_table + 34 + (charindex * 2) );

    if ( GlyphIndex <= 257 )            /* If a standard Apple name, */
    {
        return Apple_CharStrings[GlyphIndex];    // 返回预定义的苹果标准字符名
    }
    else                                /* Otherwise, use one */
    {
        /* of the pascal strings. */
        GlyphIndex -= 258;

        /* Set pointer to start of Pascal strings. */
        ptr = (char*)(font->post_table + 34 + (font->numGlyphs * 2));

        len = (ULONG)*(ptr++);  /* Step thru the strings */
        while (GlyphIndex--)            /* until we get to the one */
        {
            /* that we want. */
            ptr += len;
            len = (ULONG)*(ptr++);
        }

        if ( len >= sizeof(temp) )
        {
            throw TTException("TrueType font file contains a very long PostScript name");   // 如果名称过长则抛出异常
        }

        strncpy(temp,ptr,len);  /* Copy the pascal string into */   // 将 Pascal 字符串复制到缓冲区
        temp[len]='\0';   /* a buffer and make it ASCIIz. */   // 确保字符串以 '\0' 结尾，变成 ASCII 字符串

        return temp;    // 返回生成的字符名
    }
} /* end of ttfont_CharStrings_getname() */

/*
** This is the central routine of this section.
*/
void ttfont_CharStrings(TTStreamWriter& stream, struct TTFONT *font, std::vector<int>& glyph_ids)
{
    Fixed post_format;

    /* The 'post' table format number. */
    post_format = getFixed( font->post_table );

    /* Emmit the start of the PostScript code to define the dictionary. */
    stream.printf("/CharStrings %d dict dup begin\n", glyph_ids.size()+1);
    /* Section 5.8.2 table 5.7 of the PS Language Ref says a CharStrings dictionary must contain an entry for .notdef */
    stream.printf("/.notdef 0 def\n");

    /* Emmit one key-value pair for each glyph. */
    for (std::vector<int>::const_iterator i = glyph_ids.begin();
            i != glyph_ids.end(); ++i)
    {
        if ((font->target_type == PS_TYPE_42 ||
             font->target_type == PS_TYPE_42_3_HYBRID)
            && *i < 256) /* type 42 */
        {
            stream.printf("/%s %d def\n",ttfont_CharStrings_getname(font, *i), *i);   // 定义 Type 42 字体的字符名和对应的编码
        }
        else                            /* type 3 */
        {
            stream.printf("/%s{",ttfont_CharStrings_getname(font, *i));   // 定义 Type 3 字体的字符名和字符处理过程的开始

            tt_type3_charproc(stream, font, *i);   // 调用 Type 3 字体的字符处理过程函数

            stream.putline("}_d");      /* "} bind def" */   // 结束 Type 3 字体的字符处理过程定义
        }
    }



    # 这里是一个代码块的结束，可能是一个函数、循环、条件语句或其他代码块的结尾。
    # 在此处可能需要进行资源释放、状态更新或其他清理工作。



    stream.putline("end readonly def");



    # 向流对象 `stream` 中写入一行文本 "end readonly def"
    # 这里的行为可能是在某种特定情况下向流中添加结尾信息或标记。
    # `putline` 方法的具体实现取决于 `stream` 对象的类型和定义。
/*----------------------------------------------------------------
** Emmit the code to finish up the dictionary and turn
** it into a font.
----------------------------------------------------------------*/
void ttfont_trailer(TTStreamWriter& stream, struct TTFONT *font)
{
    /* 如果生成的是类型 3 字体，则需要提供 BuildGlyph 和 BuildChar 过程。 */
    if (font->target_type == PS_TYPE_3 ||
        font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.put_char('\n');

        stream.putline("/BuildGlyph");
        stream.putline(" {exch begin");         /* 开始字体字典 */
        stream.putline(" CharStrings exch");
        stream.putline(" 2 copy known not{pop /.notdef}if");
        stream.putline(" true 3 1 roll get exec");
        stream.putline(" end}_d");

        stream.put_char('\n');

        /* 此过程用于与 Level 1 解释器兼容。 */
        stream.putline("/BuildChar {");
        stream.putline(" 1 index /Encoding get exch get");
        stream.putline(" 1 index /BuildGlyph get exec");
        stream.putline("}_d");

        stream.put_char('\n');
    }

    /* 如果生成的是类型 42 字体，需要检查 PostScript 解释器是否理解类型 42 字体。 */
    /* 如果不理解，我们假设 Apple TrueType 栅格化器已加载，并相应调整字体。 */
    /* 这些设置信息和 BuildGlyph 的部分来自于 Macintosh 生成的 TrueType 字体。 */
    if (font->target_type == PS_TYPE_42 ||
        font->target_type == PS_TYPE_42_3_HYBRID)
    } /* end of if Type 42 not understood. */

    stream.putline("FontName currentdict end definefont pop");
    /* stream.putline("%%EOF"); */
} /* end of ttfont_trailer() */

/*------------------------------------------------------------------
** This is the externally callable routine which inserts the font.
------------------------------------------------------------------*/

void read_font(const char *filename, font_type_enum target_type, std::vector<int>& glyph_ids, TTFONT& font)
{
    BYTE *ptr;

    /* 决定要生成的 PostScript 字体类型。 */
    font.target_type = target_type;

    if (font.target_type == PS_TYPE_42)
    {
        bool has_low = false;  // 初始化标志，用于检测是否存在低编码字形
        bool has_high = false;  // 初始化标志，用于检测是否存在高编码字形

        for (std::vector<int>::const_iterator i = glyph_ids.begin();
                i != glyph_ids.end(); ++i)
        {
            if (*i > 255)  // 如果字形编码大于255
            {
                has_high = true;  // 设置高编码字形存在标志
                if (has_low) break;  // 如果已经存在低编码字形，则退出循环
            }
            else
            {
                has_low = true;  // 设置低编码字形存在标志
                if (has_high) break;  // 如果已经存在高编码字形，则退出循环
            }
        }

        if (has_high && has_low)  // 如果同时存在高编码和低编码字形
        {
            font.target_type = PS_TYPE_42_3_HYBRID;  // 设置字体目标类型为 PS_TYPE_42_3_HYBRID
        }
        else if (has_high && !has_low)  // 如果只存在高编码字形
        {
            font.target_type = PS_TYPE_3;  // 设置字体目标类型为 PS_TYPE_3
        }
    }

    /* Save the file name for error messages. */
    font.filename=filename;  // 保存文件名以备错误消息使用

    /* Open the font file */
    if ( (font.file = fopen(filename,"rb")) == (FILE*)NULL )  // 打开字体文件
    {
        throw TTException("Failed to open TrueType font");  // 如果打开失败，抛出异常
    }

    /* Allocate space for the unvarying part of the offset table. */
    assert(font.offset_table == NULL);  // 确保偏移表为空
    font.offset_table = (BYTE*)calloc( 12, sizeof(BYTE) );  // 分配偏移表的空间

    /* Read the first part of the offset table. */
    if ( fread( font.offset_table, sizeof(BYTE), 12, font.file ) != 12 )  // 读取偏移表的前12个字节
    {
        throw TTException("TrueType font may be corrupt (reason 1)");  // 如果读取失败，抛出异常
    }

    /* Determine how many directory entries there are. */
    font.numTables = getUSHORT( font.offset_table + 4 );  // 获取目录项的数量
    debug("numTables=%d",(int)font.numTables);  // 输出目录项数量的调试信息

    /* Expand the memory block to hold the whole thing. */
    font.offset_table = (BYTE*)realloc( font.offset_table, sizeof(BYTE) * (12 + font.numTables * 16) );  // 扩展偏移表以容纳完整内容

    /* Read the rest of the table directory. */
    if ( fread( font.offset_table + 12, sizeof(BYTE), (font.numTables*16), font.file ) != (font.numTables*16) )  // 读取剩余的表目录
    {
        throw TTException("TrueType font may be corrupt (reason 2)");  // 如果读取失败，抛出异常
    }

    /* Extract information from the "Offset" table. */
    font.TTVersion = getFixed( font.offset_table );  // 从“Offset”表中提取信息

    /* Load the "head" table and extract information from it. */
    ptr = GetTable(&font, "head");  // 加载“head”表
    try
    {
        font.MfrRevision = getFixed( ptr + 4 );           /* font revision number */  // 获取字体修订版本号
        font.unitsPerEm = getUSHORT( ptr + 18 );  // 获取每EM单位数
        font.HUPM = font.unitsPerEm / 2;  // 计算半EM单位数
        debug("unitsPerEm=%d",(int)font.unitsPerEm);  // 输出每EM单位数的调试信息
        font.llx = topost2( getFWord( ptr + 36 ) );               /* bounding box info */  // 提取边界框信息
        font.lly = topost2( getFWord( ptr + 38 ) );
        font.urx = topost2( getFWord( ptr + 40 ) );
        font.ury = topost2( getFWord( ptr + 42 ) );
        font.indexToLocFormat = getSHORT( ptr + 50 );     /* size of 'loca' data */  // 获取'loca'数据的大小
        if (font.indexToLocFormat != 0 && font.indexToLocFormat != 1)  // 检查indexToLocFormat的有效性
        {
            throw TTException("TrueType font is unusable because indexToLocFormat != 0");  // 如果无效，抛出异常
        }
        if ( getSHORT(ptr+52) != 0 )  // 检查glyphDataFormat的有效性
        {
            throw TTException("TrueType font is unusable because glyphDataFormat != 0");  // 如果无效，抛出异常
        }
    }
    catch (TTException& )  // 捕获可能抛出的异常
    {
        free(ptr);
        throw;
    }
    /* 如果发生异常，释放内存并重新抛出异常 */

    free(ptr);
    /* 释放指针所指向的内存 */

    /* 从 "name" 表中加载信息 */
    Read_name(&font);

    /* 我们需要有 PostScript 表存在 */
    assert(font.post_table == NULL);
    font.post_table = GetTable(&font, "post");
    font.numGlyphs = getUSHORT( font.post_table + 32 );
    /* 获取字体的字符数 */

    /* 如果正在生成 Type 3 字体，需要保留 'loca' 和 'glyf' 表 */
    /* 在生成 CharStrings 期间需要这些表 */
    if (font.target_type == PS_TYPE_3 || font.target_type == PS_TYPE_42_3_HYBRID)
    {
        BYTE *ptr;                      /* 我们只需要一个值 */
        ptr = GetTable(&font, "hhea");
        font.numberOfHMetrics = getUSHORT(ptr + 34);
        free(ptr);
        /* 获取 'hhea' 表并释放内存 */

        assert(font.loca_table == NULL);
        font.loca_table = GetTable(&font,"loca");
        assert(font.glyf_table == NULL);
        font.glyf_table = GetTable(&font,"glyf");
        assert(font.hmtx_table == NULL);
        font.hmtx_table = GetTable(&font,"hmtx");
        /* 获取 'loca', 'glyf', 'hmtx' 表并确保未分配内存 */
    }

    if (glyph_ids.size() == 0)
    {
        glyph_ids.clear();
        glyph_ids.reserve(font.numGlyphs);
        for (int x = 0; x < font.numGlyphs; ++x)
        {
            glyph_ids.push_back(x);
        }
        /* 如果字形 ID 列表为空，清除并准备存储 font.numGlyphs 个字形 ID */
    }
    else if (font.target_type == PS_TYPE_3 ||
             font.target_type == PS_TYPE_42_3_HYBRID)
    {
        ttfont_add_glyph_dependencies(&font, glyph_ids);
        /* 如果正在生成 Type 3 或混合字体，添加字形依赖关系 */
    }
} /* end of insert_ttfont() */

/* 
   向字体流中插入TrueType字体数据
   filename: 字体文件名
   stream: TTStreamWriter对象，用于写入字体流数据
   target_type: 目标字体类型，枚举值
   glyph_ids: 包含字形ID的整数向量
*/
void insert_ttfont(const char *filename, TTStreamWriter& stream,
                   font_type_enum target_type, std::vector<int>& glyph_ids)
{
    /* 定义TTFONT结构体实例 */
    struct TTFONT font;

    /* 读取字体文件数据并填充font结构体 */
    read_font(filename, target_type, glyph_ids, font);

    /* 写入PostScript字体的头部信息 */
    ttfont_header(stream, &font);

    /* 定义字体的编码 */
    ttfont_encoding(stream, &font, glyph_ids, target_type);

    /* 插入FontInfo字典 */
    ttfont_FontInfo(stream, &font);

    /* 如果生成的是Type 42字体，发射sfnts数组 */
    if (font.target_type == PS_TYPE_42 ||
        font.target_type == PS_TYPE_42_3_HYBRID)
    {
        ttfont_sfnts(stream, &font);
    }

    /* 发射CharStrings数组 */
    ttfont_CharStrings(stream, &font, glyph_ids);

    /* 发送字体的尾部信息 */
    ttfont_trailer(stream, &font);

} /* end of insert_ttfont() */

/* 
   TTFONT类的构造函数，初始化所有成员为NULL
*/
TTFONT::TTFONT() :
    file(NULL),
    PostName(NULL),
    FullName(NULL),
    FamilyName(NULL),
    Style(NULL),
    Copyright(NULL),
    Version(NULL),
    Trademark(NULL),
    offset_table(NULL),
    post_table(NULL),
    loca_table(NULL),
    glyf_table(NULL),
    hmtx_table(NULL)
{

}

/* 
   TTFONT类的析构函数，释放动态分配的内存并关闭文件
*/
TTFONT::~TTFONT()
{
    if (file)
    {
        fclose(file);
    }
    free(PostName);
    free(FullName);
    free(FamilyName);
    free(Style);
    free(Copyright);
    free(Version);
    free(Trademark);
    free(offset_table);
    free(post_table);
    free(loca_table);
    free(glyf_table);
    free(hmtx_table);
}

/* end of file */
```