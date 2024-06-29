# `D:\src\scipysrc\matplotlib\src\_tkmini.h`

```
/* Small excerpts from the Tcl / Tk 8.6 headers
 *
 * License terms copied from:
 * http://www.tcl.tk/software/tcltk/license.html
 * as of 20 May 2016.
 *
 * Copyright (c) 1987-1994 The Regents of the University of California.
 * Copyright (c) 1993-1996 Lucent Technologies.
 * Copyright (c) 1994-1998 Sun Microsystems, Inc.
 * Copyright (c) 1998-2000 by Scriptics Corporation.
 * Copyright (c) 2002 by Kevin B. Kenny.  All rights reserved.
 *
 * This software is copyrighted by the Regents of the University
 * of California, Sun Microsystems, Inc., Scriptics Corporation,
 * and other parties. The following terms apply to all files
 * associated with the software unless explicitly disclaimed in
 * individual files.
 *
 * The authors hereby grant permission to use, copy, modify,
 * distribute, and license this software and its documentation
 * for any purpose, provided that existing copyright notices are
 * retained in all copies and that this notice is included
 * verbatim in any distributions. No written agreement, license,
 * or royalty fee is required for any of the authorized uses.
 * Modifications to this software may be copyrighted by their
 * authors and need not follow the licensing terms described
 * here, provided that the new terms are clearly indicated on
 * the first page of each file where they apply.
 *
 * IN NO EVENT SHALL THE AUTHORS OR DISTRIBUTORS BE LIABLE TO
 * ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
 * CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS
 * SOFTWARE, ITS DOCUMENTATION, OR ANY DERIVATIVES THEREOF, EVEN
 * IF THE AUTHORS HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE AUTHORS AND DISTRIBUTORS SPECIFICALLY DISCLAIM ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, AND NON-INFRINGEMENT. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, AND THE AUTHORS AND DISTRIBUTORS HAVE NO
 * OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 * ENHANCEMENTS, OR MODIFICATIONS.
 *
 * GOVERNMENT USE: If you are acquiring this software on behalf
 * of the U.S. government, the Government shall have only
 * "Restricted Rights" in the software and related documentation
 * as defined in the Federal Acquisition Regulations (FARs) in
 * Clause 52.227.19 (c) (2). If you are acquiring the software
 * on behalf of the Department of Defense, the software shall be
 * classified as "Commercial Computer Software" and the
 * Government shall have only "Restricted Rights" as defined in
 * Clause 252.227-7013 (c) (1) of DFARs. Notwithstanding the
 * foregoing, the authors grant the U.S. Government and others
 * acting in its behalf permission to use and distribute the
 * software in accordance with the terms specified in this
 * license
 */

/*
 * Unless otherwise noted, these definitions are stable from Tcl / Tk 8.5
 * through Tck / Tk master as of 21 May 2016
 */

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Users of versions of Tcl >= 8.6 encouraged to treat Tcl_Interp as an opaque
 * pointer.  The following definition results when TCL_NO_DEPRECATED defined.
 */
// 定义 Tcl_Interp 结构体，用于表示 Tcl 解释器，版本 >= 8.6 建议将其视为不透明指针
typedef struct Tcl_Interp Tcl_Interp;

/* Tk header excerpts */

// 定义 Tk_PhotoHandle 为 void 指针类型
typedef void *Tk_PhotoHandle;

// 定义 Tk_PhotoImageBlock 结构体，描述图像块信息
typedef struct Tk_PhotoImageBlock
{
    unsigned char *pixelPtr;  // 指向像素数据的指针
    int width;                // 图像宽度
    int height;               // 图像高度
    int pitch;                // 行距
    int pixelSize;            // 像素大小
    int offset[4];            // 偏移数组
} Tk_PhotoImageBlock;

// 定义图像合成规则的常量
#define TK_PHOTO_COMPOSITE_OVERLAY  0  // 像素级透明度规则
#define TK_PHOTO_COMPOSITE_SET      1  // 直接设置图像缓冲

// 定义 Tcl 函数返回值的常量
#define TCL_OK     0   // 成功返回值
#define TCL_ERROR  1   // 错误返回值

/* Typedefs derived from function signatures in Tk header */
/* Tk_FindPhoto typedef */
// 定义 Tk_FindPhoto_t 类型，表示查找图像处理句柄的函数指针类型
typedef Tk_PhotoHandle (*Tk_FindPhoto_t) (Tcl_Interp *interp, const char *imageName);

/* Tk_PhotoPutBLock typedef */
// 定义 Tk_PhotoPutBlock_t 类型，表示将图像块放置到图像处理句柄的函数指针类型
typedef int (*Tk_PhotoPutBlock_t) (Tcl_Interp *interp, Tk_PhotoHandle handle,
                                   Tk_PhotoImageBlock *blockPtr, int x, int y,
                                   int width, int height, int compRule);

/* Typedefs derived from function signatures in Tcl header */
/* Tcl_SetVar typedef */
// 定义 Tcl_SetVar_t 类型，表示设置 Tcl 变量的函数指针类型
typedef const char *(*Tcl_SetVar_t)(Tcl_Interp *interp, const char *varName,
                                    const char *newValue, int flags);

#ifdef __cplusplus
}
#endif
```