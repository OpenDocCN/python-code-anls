# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\global_r.c`

```
/*<html><pre>  -<a                             href="qh-globa_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   global_r.c
   initializes all the globals of the qhull application

   see README

   see libqhull_r.h for qh.globals and function prototypes

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/global_r.c#12 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
 */

#include "qhull_ra.h"

/*========= qh->definition -- globals defined in libqhull_r.h =======================*/

/*-<a                             href  ="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="version">-</a>

  qh_version
    version string by year and date
    qh_version2 for Unix users and -V

    the revision increases on code changes only

  notes:
    change date:    Changes.txt, Announce.txt, index.htm, README.txt,
                    qhull-news.html, Eudora signatures, CMakeLists.txt
    change version: README.txt, qh-get.htm, File_id.diz, Makefile.txt, CMakeLists.txt
    check that CmakeLists @version is the same as qh_version2
    change year:    Copying.txt
    check download size
    recompile user_eg_r.c, rbox_r.c, libqhull_r.c, qconvex_r.c, qdelaun_r.c qvoronoi_r.c, qhalf_r.c, testqset_r.c
*/

const char qh_version[]= "2019.1.r 2019/06/21";
const char qh_version2[]= "qhull_r 7.3.2 (2019.1.r 2019/06/21)";

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="appendprint">-</a>

  qh_appendprint(qh, printFormat )
    append printFormat to qh.PRINTout unless already defined
*/
void qh_appendprint(qhT *qh, qh_PRINT format) {
  int i;

  for (i=0; i < qh_PRINTEND; i++) {
    if (qh->PRINTout[i] == format && format != qh_PRINTqhull)
      break;
    if (!qh->PRINTout[i]) {
      qh->PRINTout[i]= format;
      break;
    }
  }
} /* appendprint */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="checkflags">-</a>

  qh_checkflags(qh, commandStr, hiddenFlags )
    errors if commandStr contains hiddenFlags
    hiddenFlags starts and ends with a space and is space delimited (checked)

  notes:
    ignores first word (e.g., "qconvex i")
    use qh_strtol/strtod since strtol/strtod may or may not skip trailing spaces

  see:
    qh_initflags() initializes Qhull according to commandStr
*/
void qh_checkflags(qhT *qh, char *command, char *hiddenflags) {
  char *s= command, *t, *chkerr; /* qh_skipfilename is non-const */
  char key, opt, prevopt;
  char chkkey[]=  "   ";    /* check one character options ('s') */
  char chkopt[]=  "    ";   /* check two character options ('Ta') */
  char chkopt2[]= "     ";  /* check three character options ('Q12') */
  boolT waserr= False;

  if (*hiddenflags != ' ' || hiddenflags[strlen(hiddenflags)-1] != ' ') {
    qh_fprintf(qh, qh->ferr, 6026, "qhull internal error (qh_checkflags): hiddenflags must start and end with a space: \"%s\"\n", hiddenflags);
    // 输出错误消息到 qh->ferr 流，指明 hiddenflags 必须以空格开头和结尾
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    // 调用 qh_errexit 函数，终止程序执行，指示出现 qh_ERRqhull 类型的错误

  }
  if (strpbrk(hiddenflags, ",\n\r\t")) {
    // 检查 hiddenflags 是否包含逗号、换行符、回车符或制表符
    qh_fprintf(qh, qh->ferr, 6027, "qhull internal error (qh_checkflags): hiddenflags contains commas, newlines, or tabs: \"%s\"\n", hiddenflags);
    // 输出错误消息到 qh->ferr 流，指明 hiddenflags 包含不允许的字符
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    // 调用 qh_errexit 函数，终止程序执行，指示出现 qh_ERRqhull 类型的错误
  }
  while (*s && !isspace(*s))  /* skip program name */
    s++;
  // 跳过程序名，直到遇到空白字符为止

  while (*s) {
    // 遍历字符串 s
    while (*s && isspace(*s))
      s++;
    // 跳过 s 中的空白字符

    if (*s == '-')
      s++;
    // 如果当前字符是 '-'，则跳过

    if (!*s)
      break;
    // 如果 s 指向字符串结尾，则跳出循环

    key= *s++;
    // 将当前字符赋值给 key，并移动 s 到下一个字符

    chkerr= NULL;
    // 初始化 chkerr 为 NULL

    if (key == 'T' && (*s == 'I' || *s == 'O')) {  /* TI or TO 'file name' */
      // 如果 key 是 'T'，且下一个字符是 'I' 或 'O'
      s= qh_skipfilename(qh, ++s);
      // 调用 qh_skipfilename 函数，跳过文件名
      continue;
      // 继续下一次循环
    }

    chkkey[1]= key;
    // 将 key 赋值给 chkkey 的第二个元素

    if (strstr(hiddenflags, chkkey)) {
      // 如果 chkkey 是 hiddenflags 的子字符串
      chkerr= chkkey;
      // 将 chkkey 赋值给 chkerr
    } else if (isupper(key)) {
      // 如果 key 是大写字母
      opt= ' ';
      prevopt= ' ';
      chkopt[1]= key;
      chkopt2[1]= key;

      while (!chkerr && *s && !isspace(*s)) {
        // 当 chkerr 为假且 s 不为空且当前字符不是空白字符时执行循环
        opt= *s++;
        // 将当前字符赋值给 opt，并移动 s 到下一个字符

        if (isalpha(opt)) {
          // 如果 opt 是字母
          chkopt[2]= opt;
          // 将 opt 赋值给 chkopt 的第三个元素
          if (strstr(hiddenflags, chkopt))
            chkerr= chkopt;
          // 如果 chkopt 是 hiddenflags 的子字符串，则将 chkopt 赋值给 chkerr
          
          if (prevopt != ' ') {
            chkopt2[2]= prevopt;
            chkopt2[3]= opt;
            if (strstr(hiddenflags, chkopt2))
              chkerr= chkopt2;
            // 如果 chkopt2 是 hiddenflags 的子字符串，则将 chkopt2 赋值给 chkerr
          }
        } else if (key == 'Q' && isdigit(opt) && prevopt != 'b'
              && (prevopt == ' ' || islower(prevopt))) {
            // 如果 key 是 'Q'，opt 是数字且 prevopt 不是 'b'，并且 prevopt 是空格或小写字母
            if (isdigit(*s)) {  /* Q12 */
              chkopt2[2]= opt;
              chkopt2[3]= *s++;
              if (strstr(hiddenflags, chkopt2))
                chkerr= chkopt2;
              // 如果 chkopt2 是 hiddenflags 的子字符串，则将 chkopt2 赋值给 chkerr
            } else {
              chkopt[2]= opt;
              // 将 opt 赋值给 chkopt 的第三个元素
              if (strstr(hiddenflags, chkopt))
                chkerr= chkopt;
              // 如果 chkopt 是 hiddenflags 的子字符串，则将 chkopt 赋值给 chkerr
            }
        } else {
          qh_strtod(s-1, &t);
          // 调用 qh_strtod 函数，将 s-1 转换为双精度数，结果保存在 t 中
          if (s < t)
            s= t;
          // 如果 s 小于 t，则将 s 移动到 t
        }
        prevopt= opt;
        // 将 opt 赋值给 prevopt
      }
    }

    if (chkerr) {
      *chkerr= '\'';
      // 在 chkerr 的首字符前插入 '\''
      chkerr[strlen(chkerr)-1]=  '\'';
      // 将 chkerr 的末字符替换为 '\''
      qh_fprintf(qh, qh->ferr, 6029, "qhull option error: option %s is not used with this program.\n             It may be used with qhull.\n", chkerr);
      // 输出错误消息到 qh->ferr 流，指明选项 chkerr 在当前程序中不可用，可能用于 qhull
      waserr= True;
      // 设置 waserr 为 True
    }
  }

  if (waserr)
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
    // 如果 waserr 是真，调用 qh_errexit 函数，终止程序执行，指示出现 qh_ERRinput 类型的错误
/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="clear_outputflags">-</a>

  qh_clear_outputflags(qh)
    Clear output flags for QhullPoints
*/
void qh_clear_outputflags(qhT *qh) {
  int i,k;

  qh->ANNOTATEoutput= False;          /* 设置 ANNOTATEoutput 标志为 False */
  qh->DOintersections= False;         /* 设置 DOintersections 标志为 False */
  qh->DROPdim= -1;                    /* 将 DROPdim 设置为 -1 */
  qh->FORCEoutput= False;             /* 设置 FORCEoutput 标志为 False */
  qh->GETarea= False;                 /* 设置 GETarea 标志为 False */
  qh->GOODpoint= 0;                   /* 将 GOODpoint 设置为 0 */
  qh->GOODpointp= NULL;               /* 将 GOODpointp 指针设置为 NULL */
  qh->GOODthreshold= False;           /* 设置 GOODthreshold 标志为 False */
  qh->GOODvertex= 0;                  /* 将 GOODvertex 设置为 0 */
  qh->GOODvertexp= NULL;              /* 将 GOODvertexp 指针设置为 NULL */
  qh->IStracing= 0;                   /* 将 IStracing 设置为 0 */
  qh->KEEParea= False;                /* 设置 KEEParea 标志为 False */
  qh->KEEPmerge= False;               /* 设置 KEEPmerge 标志为 False */
  qh->KEEPminArea= REALmax;           /* 将 KEEPminArea 设置为 REALmax */
  qh->PRINTcentrums= False;           /* 设置 PRINTcentrums 标志为 False */
  qh->PRINTcoplanar= False;           /* 设置 PRINTcoplanar 标志为 False */
  qh->PRINTdots= False;               /* 设置 PRINTdots 标志为 False */
  qh->PRINTgood= False;               /* 设置 PRINTgood 标志为 False */
  qh->PRINTinner= False;              /* 设置 PRINTinner 标志为 False */
  qh->PRINTneighbors= False;          /* 设置 PRINTneighbors 标志为 False */
  qh->PRINTnoplanes= False;           /* 设置 PRINTnoplanes 标志为 False */
  qh->PRINToptions1st= False;         /* 设置 PRINToptions1st 标志为 False */
  qh->PRINTouter= False;              /* 设置 PRINTouter 标志为 False */
  qh->PRINTprecision= True;           /* 设置 PRINTprecision 标志为 True */
  qh->PRINTridges= False;             /* 设置 PRINTridges 标志为 False */
  qh->PRINTspheres= False;            /* 设置 PRINTspheres 标志为 False */
  qh->PRINTstatistics= False;         /* 设置 PRINTstatistics 标志为 False */
  qh->PRINTsummary= False;            /* 设置 PRINTsummary 标志为 False */
  qh->PRINTtransparent= False;        /* 设置 PRINTtransparent 标志为 False */
  qh->SPLITthresholds= False;         /* 设置 SPLITthresholds 标志为 False */
  qh->TRACElevel= 0;                  /* 将 TRACElevel 设置为 0 */
  qh->TRInormals= False;              /* 设置 TRInormals 标志为 False */
  qh->USEstdout= False;               /* 设置 USEstdout 标志为 False */
  qh->VERIFYoutput= False;            /* 设置 VERIFYoutput 标志为 False */
  
  for (k=qh->input_dim+1; k--; ) {    /* 对于 k 从 input_dim+1 递减到 0 */
    qh->lower_threshold[k]= -REALmax; /* 将 lower_threshold[k] 设置为 -REALmax */
    qh->upper_threshold[k]= REALmax;  /* 将 upper_threshold[k] 设置为 REALmax */
    qh->lower_bound[k]= -REALmax;     /* 将 lower_bound[k] 设置为 -REALmax */
    qh->upper_bound[k]= REALmax;      /* 将 upper_bound[k] 设置为 REALmax */
  }

  for (i=0; i < qh_PRINTEND; i++) {    /* 对于 i 从 0 到 qh_PRINTEND-1 */
    qh->PRINTout[i]= qh_PRINTnone;    /* 将 PRINTout[i] 设置为 qh_PRINTnone */
  }

  if (!qh->qhull_commandsiz2)
      qh->qhull_commandsiz2= (int)strlen(qh->qhull_command); /* 如果 qhull_commandsiz2 为 0，则设置为 qhull_command 的长度 */
  else {
      qh->qhull_command[qh->qhull_commandsiz2]= '\0';  /* 否则将 qhull_command 的第 qhull_commandsiz2 位置设置为 '\0' */
  }
  if (!qh->qhull_optionsiz2)
    qh->qhull_optionsiz2= (int)strlen(qh->qhull_options);  /* 如果 qhull_optionsiz2 为 0，则设置为 qhull_options 的长度 */
  else {
    qh->qhull_options[qh->qhull_optionsiz2]= '\0';        /* 否则将 qhull_options 的第 qhull_optionsiz2 位置设置为 '\0' */
    qh->qhull_optionlen= qh_OPTIONline;  /* 设置 qhull_optionlen 为 qh_OPTIONline，开始新的一行 */
  }
} /* clear_outputflags */
/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="freebuffers">-</a>

  qh_freebuffers()
    释放全局内存缓冲区

  notes:
    必须与 qh_initbuffers() 匹配
*/
void qh_freebuffers(qhT *qh) {

  trace5((qh, qh->ferr, 5001, "qh_freebuffers: freeing up global memory buffers\n"));
  /* 由 qh_initqhull_buffers 分配 */
  qh_setfree(qh, &qh->other_points);   // 释放其他点集合
  qh_setfree(qh, &qh->del_vertices);   // 释放删除的顶点集合
  qh_setfree(qh, &qh->coplanarfacetset); // 释放共面面集合
  qh_memfree(qh, qh->NEARzero, qh->hull_dim * (int)sizeof(realT));  // 释放 NEARzero 数组
  qh_memfree(qh, qh->lower_threshold, (qh->input_dim+1) * (int)sizeof(realT));  // 释放 lower_threshold 数组
  qh_memfree(qh, qh->upper_threshold, (qh->input_dim+1) * (int)sizeof(realT));  // 释放 upper_threshold 数组
  qh_memfree(qh, qh->lower_bound, (qh->input_dim+1) * (int)sizeof(realT));  // 释放 lower_bound 数组
  qh_memfree(qh, qh->upper_bound, (qh->input_dim+1) * (int)sizeof(realT));  // 释放 upper_bound 数组
  qh_memfree(qh, qh->gm_matrix, (qh->hull_dim+1) * qh->hull_dim * (int)sizeof(coordT));  // 释放 gm_matrix 数组
  qh_memfree(qh, qh->gm_row, (qh->hull_dim+1) * (int)sizeof(coordT *));  // 释放 gm_row 数组
  qh->NEARzero = qh->lower_threshold = qh->upper_threshold = NULL;  // 置空 NEARzero, lower_threshold, upper_threshold 指针
  qh->lower_bound = qh->upper_bound = NULL;  // 置空 lower_bound, upper_bound 指针
  qh->gm_matrix = NULL;  // 置空 gm_matrix 指针
  qh->gm_row = NULL;  // 置空 gm_row 指针

  if (qh->line)                /* 由 qh_readinput 分配，如果没有错误则释放 */
    qh_free(qh->line);
  if (qh->half_space)
    qh_free(qh->half_space);
  if (qh->temp_malloc)
    qh_free(qh->temp_malloc);
  if (qh->feasible_point)      /* 由 qh_readfeasible 分配 */
    qh_free(qh->feasible_point);
  if (qh->feasible_string)     /* 由 qh_initflags 分配 */
    qh_free(qh->feasible_string);
  qh->line = qh->feasible_string = NULL;  // 置空 line, feasible_string 指针
  qh->half_space = qh->feasible_point = qh->temp_malloc = NULL;  // 置空 half_space, feasible_point, temp_malloc 指针
  /* 通常由 qh_readinput 分配 */
  if (qh->first_point && qh->POINTSmalloc) {
    qh_free(qh->first_point);
    qh->first_point = NULL;
  }
  if (qh->input_points && qh->input_malloc) { /* 由 qh_joggleinput 设置 */
    qh_free(qh->input_points);
    qh->input_points = NULL;
  }
  trace5((qh, qh->ferr, 5002, "qh_freebuffers: finished\n"));
} /* freebuffers */


/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="freebuild">-</a>

  qh_freebuild(qh, allmem )
    释放由 qh_initbuild 和 qh_buildhull 使用的全局内存
    如果 !allmem，
      不释放短期内存（例如，facetT，由 qh_memfreeshort 释放）

  design:
    释放重心
    释放每个顶点
    对于每个面
      释放棱
      释放外部集合、共面集合、邻居集合、棱集合、顶点集合
      释放面
    释放哈希表
    释放内部点
    释放合并集合
    释放临时集合
*/
/* 释放构建过程中分配的内存和临时数据结构 */
void qh_freebuild(qhT *qh, boolT allmem) {
    facetT *facet, *previousfacet= NULL;  // 声明面元指针和前一个面元指针
    vertexT *vertex, *previousvertex= NULL;  // 声明顶点指针和前一个顶点指针
    ridgeT *ridge, **ridgep, *previousridge= NULL;  // 声明棱指针、棱指针数组和前一个棱指针
    mergeT *merge, **mergep;  // 声明合并结构体指针和合并结构体指针数组
    int newsize;  // 声明整型变量 newsize
    boolT freeall;  // 声明布尔变量 freeall

    /* 释放全局集合，包括来自 qh_buildhull 的引用 */
    trace5((qh, qh->ferr, 5004, "qh_freebuild: free global sets\n"));
    FOREACHmerge_(qh->facet_mergeset)  // 遍历面元合并集合（通常为空）
        qh_memfree(qh, merge, (int)sizeof(mergeT));  // 释放每个合并结构体内存
    FOREACHmerge_(qh->degen_mergeset)  // 遍历退化面元合并集合（通常为空）
        qh_memfree(qh, merge, (int)sizeof(mergeT));  // 释放每个合并结构体内存
    FOREACHmerge_(qh->vertex_mergeset)  // 遍历顶点合并集合（通常为空）
        qh_memfree(qh, merge, (int)sizeof(mergeT));  // 释放每个合并结构体内存
    qh->facet_mergeset= NULL;  // 将面元合并集合设为 NULL，临时集合由 qh_settempfree_all 释放
    qh->degen_mergeset= NULL;  // 将退化面元合并集合设为 NULL，临时集合由 qh_settempfree_all 释放
    qh->vertex_mergeset= NULL;  // 将顶点合并集合设为 NULL，临时集合由 qh_settempfree_all 释放
    qh_setfree(qh, &(qh->hash_table));  // 释放哈希表集合
    trace5((qh, qh->ferr, 5003, "qh_freebuild: free temporary sets (qh_settempfree_all)\n"));
    qh_settempfree_all(qh);  // 释放所有临时集合
    trace1((qh, qh->ferr, 1005, "qh_freebuild: free memory from qh_inithull and qh_buildhull\n"));

    if (qh->del_vertices)
        qh_settruncate(qh, qh->del_vertices, 0);  // 如果有要删除的顶点集合，则清空

    if (allmem) {
        while ((vertex= qh->vertex_list)) {  // 循环删除顶点列表中的所有顶点
            if (vertex->next)
                qh_delvertex(qh, vertex);  // 删除顶点
            else {
                qh_memfree(qh, vertex, (int)sizeof(vertexT));  // 释放顶点内存（哨兵）
                qh->newvertex_list= qh->vertex_list= NULL;
                break;
            }
            previousvertex= vertex;  // 保存前一个顶点（内存故障时使用）
            QHULL_UNUSED(previousvertex)
        }
    } else if (qh->VERTEXneighbors) {  // 如果不释放所有内存但有顶点邻居集合
        FORALLvertices
            qh_setfreelong(qh, &(vertex->neighbors));  // 释放所有顶点的邻居集合
    }

    qh->VERTEXneighbors= False;  // 置顶点邻居标志为假
    qh->GOODclosest= NULL;  // 将 GOODclosest 置为 NULL

    if (allmem) {
        FORALLfacets {
            FOREACHridge_(facet->ridges)
                ridge->seen= False;  // 标记面元的所有棱为未看见
        }
        while ((facet= qh->facet_list)) {  // 循环删除面元列表中的所有面元
            if (!facet->newfacet || !qh->NEWtentative || qh_setsize(qh, facet->ridges) > 1) {  // 跳过新面元或新的试探性面元或有多于一个棱的面元
                trace4((qh, qh->ferr, 4095, "qh_freebuild: delete the previously-seen ridges of f%d\n", facet->id));
                FOREACHridge_(facet->ridges) {
                    if (ridge->seen)
                        qh_delridge(qh, ridge);  // 删除已看见的棱
                    else
                        ridge->seen= True;  // 将未看见的棱标记为已看见
                    previousridge= ridge;  // 保存前一个棱（内存故障时使用）
                    QHULL_UNUSED(previousridge)
                }
            }
            qh_setfree(qh, &(facet->outsideset));  // 释放面元的外部集合
            qh_setfree(qh, &(facet->coplanarset));  // 释放面元的共面集合
            qh_setfree(qh, &(facet->neighbors));  // 释放面元的邻居集合
            qh_setfree(qh, &(facet->ridges));  // 释放面元的棱集合
            qh_setfree(qh, &(facet->vertices));  // 释放面元的顶点集合
            if (facet->next)
                qh_delfacet(qh, facet);  // 删除面元
            else {
                qh_memfree(qh, facet, (int)sizeof(facetT));  // 释放面元内存
                qh->visible_list= qh->newfacet_list= qh->facet_list= NULL;
            }
            previousfacet= facet;  // 保存前一个面元（内存故障时使用）
            QHULL_UNUSED(previousfacet)
        }
    } else {
        freeall= True;
    // 如果扩展 Qhull 的维度成功，则不需要释放内存
    if (qh_setlarger_quick(qh, qh->hull_dim + 1, &newsize))
      freeall= False;
    
    // 遍历所有的凸包面元
    FORALLfacets {
      // 释放面元的外部点集合内存
      qh_setfreelong(qh, &(facet->outsideset));
      // 释放面元的共面点集合内存
      qh_setfreelong(qh, &(facet->coplanarset));
      
      // 如果面元不是简单形式或者需要全部释放，则进一步处理
      if (!facet->simplicial || freeall) {
        // 释放面元的邻居集合内存
        qh_setfreelong(qh, &(facet->neighbors));
        // 释放面元的棱集合内存
        qh_setfreelong(qh, &(facet->ridges));
        // 释放面元的顶点集合内存
        qh_setfreelong(qh, &(facet->vertices));
      }
    }
  }
  /* qh 内部常量的处理 */
  
  // 释放 Qhull 中的内部点和法向量所占用的内存
  qh_memfree(qh, qh->interior_point, qh->normal_size);
  // 将 Qhull 内部点指针置空
  qh->interior_point= NULL;
/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="freeqhull">-</a>

  qh_freeqhull(qh, allmem )

  free global memory and set qhT to 0
  if !allmem,
    does not free short memory (freed by qh_memfreeshort unless qh_NOmem)

notes:
  sets qh.NOerrexit in case caller forgets to
  Does not throw errors

see:
  see qh_initqhull_start2()
  For libqhull_r, qhstatT is part of qhT

design:
  free global and temporary memory from qh_initbuild and qh_buildhull
  free buffers
*/
void qh_freeqhull(qhT *qh, boolT allmem) {
  // 禁止在此处使用 setjmp，因为该函数在退出时和 ~QhullQh 时调用
  qh->NOerrexit= True;
  // 输出跟踪消息，释放全局内存
  trace1((qh, qh->ferr, 1006, "qh_freeqhull: free global memory\n"));
  // 调用函数释放构建的内存块
  qh_freebuild(qh, allmem);
  // 释放缓冲区
  qh_freebuffers(qh);
  // 清除 qhT 的内容，保留 qh.qhmem 和 qh.qhstat
  trace1((qh, qh->ferr, 1061, "qh_freeqhull: clear qhT except for qh.qhmem and qh.qhstat\n"));
  /* memset is the same in qh_freeqhull() and qh_initqhull_start2() */
  memset((char *)qh, 0, sizeof(qhT)-sizeof(qhmemT)-sizeof(qhstatT));
  // 再次设置 qh.NOerrexit，防止调用者忘记设置
  qh->NOerrexit= True;
} /* freeqhull */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="init_A">-</a>

  qh_init_A(qh, infile, outfile, errfile, argc, argv )
    initialize memory and stdio files
    convert input options to option string (qh.qhull_command)

  notes:
    infile may be NULL if qh_readpoints() is not called

    errfile should always be defined.  It is used for reporting
    errors.  outfile is used for output and format options.

    argc/argv may be 0/NULL

    called before error handling initialized
    qh_errexit() may not be used
*/
void qh_init_A(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile, int argc, char *argv[]) {
  // 初始化内存管理
  qh_meminit(qh, errfile);
  // 初始化 Qhull 运行
  qh_initqhull_start(qh, infile, outfile, errfile);
  // 将命令行参数转换为选项字符串
  qh_init_qhull_command(qh, argc, argv);
} /* init_A */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="init_B">-</a>

  qh_init_B(qh, points, numpoints, dim, ismalloc )
    initialize globals for points array

    points has numpoints dim-dimensional points
      points[0] is the first coordinate of the first point
      points[1] is the second coordinate of the first point
      points[dim] is the first coordinate of the second point

    ismalloc=True
      Qhull will call qh_free(points) on exit or input transformation
    ismalloc=False
      Qhull will allocate a new point array if needed for input transformation

    qh.qhull_command
      is the option string.
      It is defined by qh_init_B(), qh_qhull_command(), or qh_initflags

  returns:
    if qh.PROJECTinput or (qh.DELAUNAY and qh.PROJECTdelaunay)
      projects the input to a new point array

        if qh.DELAUNAY,
          qh.hull_dim is increased by one
        if qh.ATinfinity,
          qh_projectinput adds point-at-infinity for Delaunay tri.
*/
    # 如果 qh.SCALEinput 为真，则改变输入的上下界限，参见 qh_scaleinput 函数
    if qh.SCALEinput:
        # 如果 qh.ROTATEinput 为真，则对输入进行随机旋转，参见 qh_rotateinput 函数
        if qh.ROTATEinput:
            # 如果 qh.DELAUNAY 也为真，则围绕最后一个坐标轴进行旋转
            if qh.DELAUNAY:
                rotates about the last coordinate
    
    # 注意：
    # 在定义完点之后调用此代码块
    # 可能会使用 qh_errexit() 函数来处理错误退出
/*
void qh_init_B(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc) {
  // 初始化 Qhull 全局变量
  qh_initqhull_globals(qh, points, numpoints, dim, ismalloc);
  
  // 如果内存大小为0，则初始化 Qhull 内存
  if (qh->qhmem.LASTsize == 0)
    qh_initqhull_mem(qh);
  
  // 初始化 Qhull 缓冲区
  qh_initqhull_buffers(qh);
  
  // 初始化阈值
  qh_initthresholds(qh, qh->qhull_command);
  
  // 如果设置了投影输入或者是 Delaunay 且设置了投影 Delaunay，则投影输入
  if (qh->PROJECTinput || (qh->DELAUNAY && qh->PROJECTdelaunay))
    qh_projectinput(qh);
  
  // 如果设置了缩放输入，则缩放输入
  if (qh->SCALEinput)
    qh_scaleinput(qh);
  
  // 如果设置了随机旋转角度
  if (qh->ROTATErandom >= 0) {
    // 生成随机旋转矩阵
    qh_randommatrix(qh, qh->gm_matrix, qh->hull_dim, qh->gm_row);
    
    // 如果是 Delaunay 情况下，调整旋转矩阵
    if (qh->DELAUNAY) {
      int k, lastk= qh->hull_dim-1;
      for (k=0; k < lastk; k++) {
        qh->gm_row[k][lastk]= 0.0;
        qh->gm_row[lastk][k]= 0.0;
      }
      qh->gm_row[lastk][lastk]= 1.0;
    }
    
    // 对旋转矩阵进行 Gram-Schmidt 正交化处理
    qh_gram_schmidt(qh, qh->hull_dim, qh->gm_row);
    
    // 根据旋转矩阵旋转输入数据
    qh_rotateinput(qh, qh->gm_row);
  }
} /* init_B */
*/

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="init_qhull_command">-</a>

  qh_init_qhull_command(qh, argc, argv )
    build qh.qhull_command from argc/argv
    Calls qh_exit if qhull_command is too short

  returns:
    a space-delimited string of options (just as typed)

  notes:
    makes option string easy to input and output

    argc/argv may be 0/NULL
*/
void qh_init_qhull_command(qhT *qh, int argc, char *argv[]) {

  // 将 argc 和 argv 转换成 qhull_command 字符串
  if (!qh_argv_to_command(argc, argv, qh->qhull_command, (int)sizeof(qh->qhull_command))){
    /* Assumes qh.ferr is defined. */
    // 输出错误信息到 qh.ferr，如果命令行超过设定的最大长度
    qh_fprintf(qh, qh->ferr, 6033, "qhull input error: more than %d characters in command line.\n",
          (int)sizeof(qh->qhull_command));
    // 退出程序，报告错误
    qh_exit(qh_ERRinput);  /* error reported, can not use qh_errexit */
  }
} /* init_qhull_command */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="initflags">-</a>

  qh_initflags(qh, commandStr )
    set flags and initialized constants from commandStr
    calls qh_exit() if qh.NOerrexit

  returns:
    sets qh.qhull_command to command if needed

  notes:
    ignores first word (e.g., 'qhull' in "qhull d")
    use qh_strtol/strtod since strtol/strtod may or may not skip trailing spaces

  see:
    qh_initthresholds() continues processing of 'Pdn' and 'PDn'
    'prompt' in unix_r.c for documentation

  design:
    for each space-delimited option group
      if top-level option
        check syntax
        append appropriate option to option string
        set appropriate global variable or append printFormat to print options
      else
        for each sub-option
          check syntax
          append appropriate option to option string
          set appropriate global variable or append printFormat to print options
*/
void qh_initflags(qhT *qh, char *command) {
  int k, i, lastproject;
  char *s= command, *t, *prev_s, *start, key, *lastwarning= NULL;
  boolT isgeom= False, wasproject;
  realT r;

  // 如果禁用错误退出
  if(qh->NOerrexit){
    # 使用 qh_fprintf 函数将错误消息输出到 qh 对象的 ferr 流中，编号为 6245
    qh_fprintf(qh, qh->ferr, 6245, "qhull internal error (qh_initflags): qh.NOerrexit was not cleared before calling qh_initflags().  It should be cleared after setjmp().  Exit qhull.\n");
    # 调用 qh_exit 函数，退出程序并指定错误码为 qh_ERRqhull
    qh_exit(qh_ERRqhull);
#ifdef qh_RANDOMdist
  // 如果定义了 qh_RANDOMdist 宏，则设置 qh->RANDOMfactor 为 qh_RANDOMdist 的值
  qh->RANDOMfactor= qh_RANDOMdist;
  // 设置 qh->RANDOMfactor 作为 Random-qh_RANDOMdist 的选项，并存储在 qh->RANDOMfactor 中
  qh_option(qh, "Random-qh_RANDOMdist", NULL, &qh->RANDOMfactor);
  // 设置 qh->RANDOMdist 为 True，表示启用了随机分布
  qh->RANDOMdist= True;
#endif

// 检查命令指针是否在 qh->qhull_command 数组的有效范围内
if (command <= &qh->qhull_command[0] || command > &qh->qhull_command[0] + sizeof(qh->qhull_command)) {
  // 如果 command 不是指向 qh->qhull_command 的起始地址，则执行以下操作
  if (command != &qh->qhull_command[0]) {
    // 清空 qh->qhull_command 字符串
    *qh->qhull_command= '\0';
    // 将 command 拼接到 qh->qhull_command 后面，确保不超过 sizeof(qh->qhull_command) 的长度
    strncat(qh->qhull_command, command, sizeof(qh->qhull_command)-strlen(qh->qhull_command)-1);
  }
  // 跳过程序名称部分，直到遇到空白字符为止
  while (*s && !isspace(*s))  /* skip program name */
    s++;
}

// 处理剩余的命令字符串
while (*s) {
  // 跳过空白字符
  while (*s && isspace(*s))
    s++;
  // 如果遇到 '-' 字符，则跳过
  if (*s == '-')
    s++;
  // 如果到达字符串末尾，则结束循环
  if (!*s)
    break;
  // 记录当前的 s 指针位置
  prev_s= s;
  // 根据 '-' 后面的字符执行不同的操作
  switch (*s++) {
  case 'd':
    // 设置 qh->DELAUNAY 为 True，表示启用 Delaunay 选项
    qh_option(qh, "delaunay", NULL, NULL);
    qh->DELAUNAY= True;
    break;
  case 'f':
    // 设置 qh_PRINTfacets 为输出 facets 的选项
    qh_option(qh, "facets", NULL, NULL);
    qh_appendprint(qh, qh_PRINTfacets);
    break;
  case 'i':
    // 设置 qh_PRINTincidences 为输出 incidences 的选项
    qh_option(qh, "incidence", NULL, NULL);
    qh_appendprint(qh, qh_PRINTincidences);
    break;
  case 'm':
    // 设置 qh_PRINTmathematica 为输出 mathematica 格式的选项
    qh_option(qh, "mathematica", NULL, NULL);
    qh_appendprint(qh, qh_PRINTmathematica);
    break;
  case 'n':
    // 设置 qh_PRINTnormals 为输出 normals 的选项
    qh_option(qh, "normals", NULL, NULL);
    qh_appendprint(qh, qh_PRINTnormals);
    break;
  case 'o':
    // 设置 qh_PRINToff 为输出 offFile 格式的选项
    qh_option(qh, "offFile", NULL, NULL);
    qh_appendprint(qh, qh_PRINToff);
    break;
  case 'p':
    // 设置 qh_PRINTpoints 为输出 points 的选项
    qh_option(qh, "points", NULL, NULL);
    qh_appendprint(qh, qh_PRINTpoints);
    break;
  case 's':
    // 设置 qh->PRINTsummary 为 True，表示启用 summary 输出选项
    qh_option(qh, "summary", NULL, NULL);
    qh->PRINTsummary= True;
    break;
  case 'v':
    // 设置 qh->VORONOI 和 qh->DELAUNAY 为 True，表示启用 Voronoi 和 Delaunay 选项
    qh_option(qh, "voronoi", NULL, NULL);
    qh->VORONOI= True;
    qh->DELAUNAY= True;
    break;
  case 'A':
    // 如果后面的字符不是数字、'.' 或者 '-'，输出警告信息
    if (!isdigit(*s) && *s != '.' && *s != '-') {
      qh_fprintf(qh, qh->ferr, 7002, "qhull input warning: no maximum cosine angle given for option 'An'.  A1.0 is coplanar\n");
      lastwarning= s-1;
    }else {
      // 如果是 '-' 字符，则设置 qh->premerge_cos 为负数
      if (*s == '-') {
        qh->premerge_cos= -qh_strtod(s, &s);
        qh_option(qh, "Angle-premerge-", NULL, &qh->premerge_cos);
        qh->PREmerge= True;
      }else {
        // 否则设置 qh->postmerge_cos 为对应的值
        qh->postmerge_cos= qh_strtod(s, &s);
        qh_option(qh, "Angle-postmerge", NULL, &qh->postmerge_cos);
        qh->POSTmerge= True;
      }
      // 启用 MERGING 标志
      qh->MERGING= True;
    }
    break;
  case 'C':
    // 如果后面的字符不是数字、'.' 或者 '-'，输出警告信息
    if (!isdigit(*s) && *s != '.' && *s != '-') {
      qh_fprintf(qh, qh->ferr, 7003, "qhull input warning: no centrum radius given for option 'Cn'\n");
      lastwarning= s-1;
    }else {
      // 如果是 '-' 字符，则设置 qh->premerge_centrum 为负数
      if (*s == '-') {
        qh->premerge_centrum= -qh_strtod(s, &s);
        qh_option(qh, "Centrum-premerge-", NULL, &qh->premerge_centrum);
        qh->PREmerge= True;
      }else {
        // 否则设置 qh->postmerge_centrum 为对应的值
        qh->postmerge_centrum= qh_strtod(s, &s);
        qh_option(qh, "Centrum-postmerge", NULL, &qh->postmerge_centrum);
        qh->POSTmerge= True;
      }
      // 启用 MERGING 标志
      qh->MERGING= True;
    }
    break;
    case 'E':
      // 如果下一个字符是减号，报错，因为期望的是正数
      if (*s == '-') {
        qh_fprintf(qh, qh->ferr, 6363, "qhull option error: expecting a positive number for maximum roundoff 'En'.  Got '%s'\n", s-1);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      // 如果下一个字符不是数字，发出警告，表示未指定最大舍入值
      } else if (!isdigit(*s)) {
        qh_fprintf(qh, qh->ferr, 7005, "qhull option warning: no maximum roundoff given for option 'En'\n");
        lastwarning = s-1;
      // 否则将字符串转换为双精度浮点数，设置为 DISTround，并标记已设置舍入值
      } else {
        qh->DISTround = qh_strtod(s, &s);
        qh_option(qh, "Distance-roundoff", NULL, &qh->DISTround);
        qh->SETroundoff = True;
      }
      break;
    case 'H':
      // 记录起始位置
      start = s;
      // 标记为半空间设置
      qh->HALFspace = True;
      // 尝试将字符串转换为双精度浮点数，直到转换失败
      qh_strtod(s, &t);
      while (t > s)  {
        // 检查每个字符是否为空格或逗号，如果不是，报错
        if (*t && !isspace(*t)) {
          if (*t == ',')
            t++;
          else {
            qh_fprintf(qh, qh->ferr, 6364, "qhull option error: expecting 'Hn,n,n,...' for feasible point of halfspace intersection. Got '%s'\n", start-1);
            qh_errexit(qh, qh_ERRinput, NULL, NULL);
          }
        }
        s = t;
        qh_strtod(s, &t);
      }
      // 如果成功转换，分配内存存储半空间描述字符串，并设置相关选项
      if (start < t) {
        if (!(qh->feasible_string = (char *)calloc((size_t)(t-start+1), (size_t)1))) {
          qh_fprintf(qh, qh->ferr, 6034, "qhull error: insufficient memory for 'Hn,n,n'\n");
          qh_errexit(qh, qh_ERRmem, NULL, NULL);
        }
        strncpy(qh->feasible_string, start, (size_t)(t-start));
        qh_option(qh, "Halfspace-about", NULL, NULL);
        qh_option(qh, qh->feasible_string, NULL, NULL);
      } else
        // 如果没有有效数据，设置半空间选项
        qh_option(qh, "Halfspace", NULL, NULL);
      break;
    case 'R':
      // 如果下一个字符不是数字，发出警告，表示未指定随机扰动值
      if (!isdigit(*s)) {
        qh_fprintf(qh, qh->ferr, 7007, "qhull option warning: missing random perturbation for option 'Rn'\n");
        lastwarning = s-1;
      // 否则将字符串转换为双精度浮点数，设置为 RANDOMfactor，并标记为已设置随机扰动值
      } else {
        qh->RANDOMfactor = qh_strtod(s, &s);
        qh_option(qh, "Random-perturb", NULL, &qh->RANDOMfactor);
        qh->RANDOMdist = True;
      }
      break;
    case 'V':
      // 如果下一个字符不是数字或减号，发出警告，表示未指定可见距离
      if (!isdigit(*s) && *s != '-') {
        qh_fprintf(qh, qh->ferr, 7008, "qhull option warning: missing visible distance for option 'Vn'\n");
        lastwarning = s-1;
      // 否则将字符串转换为双精度浮点数，设置为 MINvisible
      } else {
        qh->MINvisible = qh_strtod(s, &s);
        qh_option(qh, "Visible", NULL, &qh->MINvisible);
      }
      break;
    case 'U':
      // 如果下一个字符不是数字或减号，发出警告，表示未指定共面距离
      if (!isdigit(*s) && *s != '-') {
        qh_fprintf(qh, qh->ferr, 7009, "qhull option warning: missing coplanar distance for option 'Un'\n");
        lastwarning = s-1;
      // 否则将字符串转换为双精度浮点数，设置为 MAXcoplanar
      } else {
        qh->MAXcoplanar = qh_strtod(s, &s);
        qh_option(qh, "U-coplanar", NULL, &qh->MAXcoplanar);
      }
      break;
    // 如果选项字符为 'W'
    case 'W':
      // 如果下一个字符是 '-'，表示出错，输出错误信息并退出
      if (*s == '-') {
        qh_fprintf(qh, qh->ferr, 6365, "qhull option error: expecting a positive number for outside width 'Wn'.  Got '%s'\n", s-1);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
      // 如果下一个字符不是数字，发出警告，表示缺少外部宽度选项
      else if (!isdigit(*s)) {
        qh_fprintf(qh, qh->ferr, 7011, "qhull option warning: missing outside width for option 'Wn'\n");
        lastwarning= s-1;
      }
      // 否则，解析外部宽度的数值
      else {
        qh->MINoutside= qh_strtod(s, &s);
        // 设置 'W-outside' 选项并标记为使用近似凸壳算法
        qh_option(qh, "W-outside", NULL, &qh->MINoutside);
        qh->APPROXhull= True;
      }
      break;
    /************  子菜单 ***************/
    // 如果选项字符为 'G'
    case 'G':
      // 标记为几何视图选项
      isgeom= True;
      // 添加几何视图打印标记
      qh_appendprint(qh, qh_PRINTgeom);
      // 遍历选项字符串，直到遇到空白字符为止
      while (*s && !isspace(*s)) {
        switch (*s++) {
        // 如果字符为 'a'，打印所有点选项并设置打印点标志
        case 'a':
          qh_option(qh, "Gall-points", NULL, NULL);
          qh->PRINTdots= True;
          break;
        // 如果字符为 'c'，打印中心选项并设置打印中心标志
        case 'c':
          qh_option(qh, "Gcentrums", NULL, NULL);
          qh->PRINTcentrums= True;
          break;
        // 如果字符为 'h'，打印交点选项并设置计算交点标志
        case 'h':
          qh_option(qh, "Gintersections", NULL, NULL);
          qh->DOintersections= True;
          break;
        // 如果字符为 'i'，打印内部选项并设置打印内部标志
        case 'i':
          qh_option(qh, "Ginner", NULL, NULL);
          qh->PRINTinner= True;
          break;
        // 如果字符为 'n'，打印无平面选项并设置打印无平面标志
        case 'n':
          qh_option(qh, "Gno-planes", NULL, NULL);
          qh->PRINTnoplanes= True;
          break;
        // 如果字符为 'o'，打印外部选项并设置打印外部标志
        case 'o':
          qh_option(qh, "Gouter", NULL, NULL);
          qh->PRINTouter= True;
          break;
        // 如果字符为 'p'，打印点选项并设置打印共面点标志
        case 'p':
          qh_option(qh, "Gpoints", NULL, NULL);
          qh->PRINTcoplanar= True;
          break;
        // 如果字符为 'r'，打印边界选项并设置打印边界标志
        case 'r':
          qh_option(qh, "Gridges", NULL, NULL);
          qh->PRINTridges= True;
          break;
        // 如果字符为 't'，打印透明选项并设置打印透明标志
        case 't':
          qh_option(qh, "Gtransparent", NULL, NULL);
          qh->PRINTtransparent= True;
          break;
        // 如果字符为 'v'，打印顶点选项并设置打印球体标志
        case 'v':
          qh_option(qh, "Gvertices", NULL, NULL);
          qh->PRINTspheres= True;
          break;
        // 如果字符为 'D'
        case 'D':
          // 如果下一个字符不是数字，发出警告，表示缺少维度选项
          if (!isdigit(*s)) {
            qh_fprintf(qh, qh->ferr, 7004, "qhull option warning: missing dimension for option 'GDn'\n");
            lastwarning= s-2;
          }
          // 否则，解析要丢弃的维度数值
          else {
            // 如果已经设置过要丢弃的维度，发出警告
            if (qh->DROPdim >= 0) {
              qh_fprintf(qh, qh->ferr, 7013, "qhull option warning: can only drop one dimension.  Previous 'GD%d' ignored\n",
                   qh->DROPdim);
              lastwarning= s-2;
            }
            // 设置要丢弃的维度数值，并设置选项
            qh->DROPdim= qh_strtol(s, &s);
            qh_option(qh, "GDrop-dim", &qh->DROPdim, NULL);
          }
          break;
        // 对于未知选项字符，发出警告并跳过至下一个空白字符
        default:
          s--;
          qh_fprintf(qh, qh->ferr, 7014, "qhull option warning: unknown 'G' geomview option 'G%c', skip to next space\n", (int)s[0]);
          lastwarning= s-1;
          while (*++s && !isspace(*s));
          break;
        }
      }
      break;
    case 'P':
      // 当前字符为 'P'，处理以 'P' 开头的打印选项
      while (*s && !isspace(*s)) {
        // 循环处理直到遇到空格为止
        switch (*s++) {
          // 开始处理当前字符，并移动到下一个字符
        case 'd': case 'D':  /* see qh_initthresholds() */
          // 对应选项 'd' 或 'D'，初始化阈值相关设置
          key= s[-1];  // 记录选项字符
          i= qh_strtol(s, &s);  // 解析整数参数
          r= 0;
          if (*s == ':') {
            // 如果遇到冒号，解析浮点数参数
            s++;
            r= qh_strtod(s, &s);
          }
          if (key == 'd')
            qh_option(qh, "Pdrop-facets-dim-less", &i, &r);  // 设置对应选项参数
          else
            qh_option(qh, "PDrop-facets-dim-more", &i, &r);
          break;
        case 'g':
          // 对应选项 'g'，设置好的面打印选项
          qh_option(qh, "Pgood-facets", NULL, NULL);
          qh->PRINTgood= True;  // 设置打印好的面为真
          break;
        case 'G':
          // 对应选项 'G'，设置好的面的邻居打印选项
          qh_option(qh, "PGood-facet-neighbors", NULL, NULL);
          qh->PRINTneighbors= True;  // 设置打印面的邻居为真
          break;
        case 'o':
          // 对应选项 'o'，设置输出强制打印选项
          qh_option(qh, "Poutput-forced", NULL, NULL);
          qh->FORCEoutput= True;  // 设置强制输出为真
          break;
        case 'p':
          // 对应选项 'p'，设置忽略精度打印选项
          qh_option(qh, "Pprecision-ignore", NULL, NULL);
          qh->PRINTprecision= False;  // 设置打印精度为假
          break;
        case 'A':
          // 对应选项 'A'，处理保留面积选项
          if (!isdigit(*s)) {
            // 如果参数不是数字，输出警告并记录上一个字符位置
            qh_fprintf(qh, qh->ferr, 7006, "qhull option warning: missing facet count for keep area option 'PAn'\n");
            lastwarning= s-2;
          } else {
            // 否则解析参数，并设置保留面积选项
            qh->KEEParea= qh_strtol(s, &s);
            qh_option(qh, "PArea-keep", &qh->KEEParea, NULL);
            qh->GETarea= True;  // 设置获取面积为真
          }
          break;
        case 'F':
          // 对应选项 'F'，处理保留面积选项
          if (!isdigit(*s)) {
            // 如果参数不是数字，输出警告并记录上一个字符位置
            qh_fprintf(qh, qh->ferr, 7010, "qhull option warning: missing facet area for option 'PFn'\n");
            lastwarning= s-2;
          } else {
            // 否则解析参数，并设置最小保留面积选项
            qh->KEEPminArea= qh_strtod(s, &s);
            qh_option(qh, "PFacet-area-keep", NULL, &qh->KEEPminArea);
            qh->GETarea= True;  // 设置获取面积为真
          }
          break;
        case 'M':
          // 对应选项 'M'，处理保留合并选项
          if (!isdigit(*s)) {
            // 如果参数不是数字，输出警告并记录上一个字符位置
            qh_fprintf(qh, qh->ferr, 7090, "qhull option warning: missing merge count for option 'PMn'\n");
            lastwarning= s-2;
          } else {
            // 否则解析参数，并设置保留合并选项
            qh->KEEPmerge= qh_strtol(s, &s);
            qh_option(qh, "PMerge-keep", &qh->KEEPmerge, NULL);
          }
          break;
        default:
          // 对于未知选项字符，输出警告并跳过到下一个空格
          s--;
          qh_fprintf(qh, qh->ferr, 7015, "qhull option warning: unknown 'P' print option 'P%c', skip to next space\n", (int)s[0]);
          lastwarning= s-1;
          while (*++s && !isspace(*s));
          break;
        }
      }
      break;
#ifndef qh_NOmerge
            // 如果未定义 qh_NOmerge 宏，则设置 Q14-merge-pinched-vertices 选项为 True
            qh_option(qh, "Q14-merge-pinched-vertices", NULL, NULL);
            qh->MERGEpinched= True;
#else
            /* ignore 'Q14' for q_benchmark testing of difficult cases for Qhull */
            // 如果定义了 qh_NOmerge 宏，则在 q_benchmark 测试中忽略 'Q14-merge-pinched' 选项，输出警告信息
            qh_fprintf(qh, qh->ferr, 7099, "qhull option warning: option 'Q14-merge-pinched' disabled due to qh_NOmerge\n");
#endif
    default:
      // 对于未知选项输出警告信息，显示未知选项的字符及其 ASCII 码
      qh_fprintf(qh, qh->ferr, 7094, "qhull option warning: unknown option '%c'(%x)\n",
        (int)s[-1], (int)s[-1]);
      lastwarning= s-2;
      break;
    }
    // 如果前一个选项字符与当前字符相同，并且当前字符存在且不是空格，则输出警告信息，忽略到下一个空格之前的字符
    if (s-1 == prev_s && *s && !isspace(*s)) {
      qh_fprintf(qh, qh->ferr, 7036, "qhull option warning: missing space after option '%c'(%x), reserved for sub-options, ignoring '%c' options to next space\n",
               (int)*prev_s, (int)*prev_s, (int)*prev_s);
      lastwarning= s-1;
      while (*s && !isspace(*s))
        s++;
    }
  }
  // 如果 STOPcone 存在且 JOGGLEmax 小于 REALmax 的一半，则输出警告信息，指出 'TCn' 选项在 'QJn' 选项使用时被忽略
  if (qh->STOPcone && qh->JOGGLEmax < REALmax/2) {
    qh_fprintf(qh, qh->ferr, 7078, "qhull option warning: 'TCn' (stopCone) ignored when used with 'QJn' (joggle)\n");
    lastwarning= command;
  }
  // 如果是几何输出且未强制输出，并且 PRINTout[1] 存在，则输出警告信息，指出附加输出格式与 Geomview 不兼容
  if (isgeom && !qh->FORCEoutput && qh->PRINTout[1]) {
    qh_fprintf(qh, qh->ferr, 7037, "qhull option warning: additional output formats ('Fc',etc.) are not compatible with Geomview ('G').  Use option 'Po' to override\n");
    lastwarning= command;
  }
  // 如果存在上一个警告信息且未允许警告，则输出错误信息，显示先前的警告信息，建议使用 'Qw' 选项来覆盖
  if (lastwarning && !qh->ALLOWwarning) {
    qh_fprintf(qh, qh->ferr, 6035, "qhull option error: see previous warnings, use 'Qw' to override: '%s' (last offset %d)\n", 
          command, (int)(lastwarning-command));
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  // 输出跟踪信息，指示选项标志已初始化
  trace4((qh, qh->ferr, 4093, "qh_initflags: option flags initialized\n"));
  /* set derived values in qh_initqhull_globals */
} /* initflags */


/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="initqhull_buffers">-</a>

  qh_initqhull_buffers(qh)
    initialize global memory buffers

  notes:
    must match qh_freebuffers()
*/
void qh_initqhull_buffers(qhT *qh) {
  int k;

  // 计算临时空间大小，以便于初始化全局内存缓冲区
  qh->TEMPsize= (qh->qhmem.LASTsize - (int)sizeof(setT))/SETelemsize;
  if (qh->TEMPsize <= 0 || qh->TEMPsize > qh->qhmem.LASTsize)
    qh->TEMPsize= 8;  /* e.g., if qh_NOmem */
  // 初始化各种全局内存缓冲区
  qh->other_points= qh_setnew(qh, qh->TEMPsize);
  qh->del_vertices= qh_setnew(qh, qh->TEMPsize);
  qh->coplanarfacetset= qh_setnew(qh, qh->TEMPsize);
  qh->NEARzero= (realT *)qh_memalloc(qh, qh->hull_dim * (int)sizeof(realT));
  qh->lower_threshold= (realT *)qh_memalloc(qh, (qh->input_dim+1) * (int)sizeof(realT));
  qh->upper_threshold= (realT *)qh_memalloc(qh, (qh->input_dim+1) * (int)sizeof(realT));
  qh->lower_bound= (realT *)qh_memalloc(qh, (qh->input_dim+1) * (int)sizeof(realT));
  qh->upper_bound= (realT *)qh_memalloc(qh, (qh->input_dim+1) * (int)sizeof(realT));
  for (k=qh->input_dim+1; k--; ) {  /* duplicated in qh_initqhull_buffers and qh_clear_outputflags */
    // 初始化阈值数组
    qh->lower_threshold[k]= -REALmax;
    qh->upper_threshold[k]= REALmax;
    qh->lower_bound[k]= -REALmax;
    qh->upper_bound[k]= REALmax;

设置 `qh` 结构体中 `upper_bound` 数组的第 `k` 个元素为 `REALmax` 的值。


  }
  qh->gm_matrix= (coordT *)qh_memalloc(qh, (qh->hull_dim+1) * qh->hull_dim * (int)sizeof(coordT));

使用 `qh_memalloc` 函数为 `qh` 结构体中的 `gm_matrix` 分配内存空间，空间大小为 `(qh->hull_dim+1) * qh->hull_dim` 个 `coordT` 类型的元素所占空间大小。


  qh->gm_row= (coordT **)qh_memalloc(qh, (qh->hull_dim+1) * (int)sizeof(coordT *));

使用 `qh_memalloc` 函数为 `qh` 结构体中的 `gm_row` 分配内存空间，空间大小为 `(qh->hull_dim+1)` 个指向 `coordT` 类型指针的空间大小。
} /* initqhull_buffers */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="initqhull_globals">-</a>

  qh_initqhull_globals(qh, points, numpoints, dim, ismalloc )
    initialize globals
    if ismalloc
      points were malloc'd and qhull should free at end

  returns:
    sets qh.first_point, num_points, input_dim, hull_dim and others
    seeds random number generator (seed=1 if tracing)
    modifies qh.hull_dim if ((qh.DELAUNAY and qh.PROJECTdelaunay) or qh.PROJECTinput)
    adjust user flags as needed
    also checks DIM3 dependencies and constants

  notes:
    do not use qh_point() since an input transformation may move them elsewhere
    qh_initqhull_start() sets default values for non-zero globals
    consider duplicate error checks in qh_readpoints.  It is called before qh_initqhull_globals

  design:
    initialize points array from input arguments
    test for qh.ZEROcentrum
      (i.e., use opposite vertex instead of cetrum for convexity testing)
    initialize qh.CENTERtype, qh.normal_size,
      qh.center_size, qh.TRACEpoint/level,
    initialize and test random numbers
    qh_initqhull_outputflags() -- adjust and test output flags
*/
void qh_initqhull_globals(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc) {
  int seed, pointsneeded, extra= 0, i, randi, k;
  realT randr;
  realT factorial;

  time_t timedata;

  trace0((qh, qh->ferr, 13, "qh_initqhull_globals: for %s | %s\n", qh->rbox_command,
      qh->qhull_command));
  // 检查输入点的数量是否合法，若不合法则输出错误信息并退出
  if (numpoints < 1 || numpoints > qh_POINTSmax) {
    qh_fprintf(qh, qh->ferr, 6412, "qhull input error (qh_initqhull_globals): expecting between 1 and %d points.  Got %d %d-d points\n",
      qh_POINTSmax, numpoints, dim);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
    /* same error message in qh_readpoints */
  }
  // 设置是否使用malloc分配的点，并初始化相关全局变量
  qh->POINTSmalloc= ismalloc;
  qh->first_point= points;
  qh->num_points= numpoints;
  qh->hull_dim= qh->input_dim= dim;
  // 根据条件设置是否启用合并选项，并相应地调整参数
  if (!qh->NOpremerge && !qh->MERGEexact && !qh->PREmerge && qh->JOGGLEmax > REALmax/2) {
    qh->MERGING= True;
    if (qh->hull_dim <= 4) {
      qh->PREmerge= True;
      qh_option(qh, "_pre-merge", NULL, NULL);
    }else {
      qh->MERGEexact= True;
      qh_option(qh, "Qxact-merge", NULL, NULL);
    }
  }else if (qh->MERGEexact)
    qh->MERGING= True;
  // 若禁用预合并选项但启用了精确合并选项，则输出警告信息
  if (qh->NOpremerge && (qh->MERGEexact || qh->PREmerge))
    qh_fprintf(qh, qh->ferr, 7095, "qhull option warning: 'Q0-no-premerge' ignored due to exact merge ('Qx') or pre-merge ('C-n' or 'A-n')\n");
  // 若禁用预合并选项且joggle参数设置异常，则将joggle参数置为0
  if (!qh->NOpremerge && qh->JOGGLEmax > REALmax/2) {
#ifdef qh_NOmerge
    qh->JOGGLEmax= 0.0;
    //...
#ifdef qh_NOmerge
  // 如果定义了 qh_NOmerge 宏，则执行以下代码块
  {
    qh_fprintf(qh, qh->ferr, 6045, "qhull option error: merging not installed (qh_NOmerge) for 'Qx', 'Cn' or 'An')\n");
    // 输出错误信息到 qh->ferr 流，指示 qh_NOmerge 宏未安装用于 'Qx', 'Cn' 或 'An' 的合并操作
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
    // 退出程序，报告输入错误
  }
#endif
  // 如果未定义 qh_NOmerge 宏，则跳过此代码块

if (qh->DELAUNAY && qh->KEEPcoplanar && !qh->KEEPinside) {
  // 如果进行 DELAUNAY 计算，并且保持共面点，并且不保持内部点
  {
    qh->KEEPinside= True;
    // 设置 KEEPinside 为 True，保持内部点
    qh_option(qh, "Qinterior-keep", NULL, NULL);
    // 设置 Qinterior-keep 选项
  }
}

if (qh->VORONOI && !qh->DELAUNAY) {
  // 如果进行 VORONOI 计算，并且不进行 DELAUNAY 计算
  {
    qh_fprintf(qh, qh->ferr, 6038, "qhull internal error (qh_initqhull_globals): if qh.VORONOI is set, qh.DELAUNAY must be set.  Qhull constructs the Delaunay triangulation in order to compute the Voronoi diagram\n");
    // 输出内部错误信息到 qh->ferr 流，指示必须设置 qh.DELAUNAY 才能进行 qh.VORONOI 计算
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    // 退出程序，报告 Qhull 内部错误
  }
}

if (qh->DELAUNAY && qh->HALFspace) {
  // 如果进行 DELAUNAY 计算，并且使用 HALFspace
  {
    qh_fprintf(qh, qh->ferr, 6046, "qhull option error: can not use Delaunay('d') or Voronoi('v') with halfspace intersection('H')\n");
    // 输出错误信息到 qh->ferr 流，指示不能在半空间交集 ('H') 中使用 Delaunay('d') 或 Voronoi('v')
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
    // 退出程序，报告输入错误
    /* same error message in qh_readpoints */
    // 在 qh_readpoints 中也输出相同的错误信息
  }
}

if (!qh->DELAUNAY && (qh->UPPERdelaunay || qh->ATinfinity)) {
  // 如果不进行 DELAUNAY 计算，并且使用 UPPERdelaunay 或 ATinfinity
  {
    qh_fprintf(qh, qh->ferr, 6047, "qhull option error: use upper-Delaunay('Qu') or infinity-point('Qz') with Delaunay('d') or Voronoi('v')\n");
    // 输出错误信息到 qh->ferr 流，指示在 Delaunay('d') 或 Voronoi('v') 中使用 upper-Delaunay('Qu') 或 infinity-point('Qz')
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
    // 退出程序，报告输入错误
  }
}

if (qh->UPPERdelaunay && qh->ATinfinity) {
  // 如果使用 UPPERdelaunay 和 ATinfinity
  {
    qh_fprintf(qh, qh->ferr, 6048, "qhull option error: can not use infinity-point('Qz') with upper-Delaunay('Qu')\n");
    // 输出错误信息到 qh->ferr 流，指示不能在 upper-Delaunay('Qu') 中使用 infinity-point('Qz')
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
    // 退出程序，报告输入错误
  }
}

if (qh->MERGEpinched && qh->ONLYgood) {
  // 如果使用 MERGEpinched 和 ONLYgood
  {
    qh_fprintf(qh, qh->ferr, 6362, "qhull option error: can not use merge-pinched-vertices ('Q14') with good-facets-only ('Qg')\n");
    // 输出错误信息到 qh->ferr 流，指示不能在仅用 good-facets-only ('Qg') 的情况下使用 merge-pinched-vertices ('Q14')
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
    // 退出程序，报告输入错误
  }
}

if (qh->MERGEpinched && qh->hull_dim == 2) {
  // 如果使用 MERGEpinched 并且凸壳维度为 2
  {
    trace2((qh, qh->ferr, 2108, "qh_initqhull_globals: disable qh.MERGEpinched for 2-d.  It has no effect"))
    // 输出跟踪信息到 ferr 流，指示在 2 维情况下禁用 qh.MERGEpinched，因为它没有效果
    qh->MERGEpinched= False;
    // 设置 MERGEpinched 为 False
  }
}

if (qh->SCALElast && !qh->DELAUNAY && qh->PRINTprecision) {
  // 如果使用 SCALElast 并且不进行 DELAUNAY 计算，并且打印精度
  {
    // 输出警告信息到 qh 对象的 ferr 流，编号为 7040，提示选项 'Qbb' 通常与 'd' 或 'v' 一起使用
    qh_fprintf(qh, qh->ferr, 7040, "qhull option warning: option 'Qbb' (scale-last-coordinate) is normally used with 'd' or 'v'\n");

    // 根据条件设置 DOcheckmax 标志，若不跳过检查且处于合并或近似凸壳状态，则设置为真
    qh->DOcheckmax= (!qh->SKIPcheckmax && (qh->MERGING || qh->APPROXhull));

    // 根据 DOcheckmax 标志设置 KEEPnearinside 标志，需满足未保留内部点且未保留共面点，且非禁用近内部点选项
    qh->KEEPnearinside= (qh->DOcheckmax && !(qh->KEEPinside && qh->KEEPcoplanar)
                          && !qh->NOnearinside);

    // 若处于合并状态，设置 CENTERtype 为 qh_AScentrum
    if (qh->MERGING)
        qh->CENTERtype= qh_AScentrum;
    // 若处于 Voronoi 图计算状态，设置 CENTERtype 为 qh_ASvoronoi
    else if (qh->VORONOI)
        qh->CENTERtype= qh_ASvoronoi;

    // 若启用测试顶点邻居选项且非合并状态，输出错误信息到 qh 对象的 ferr 流，编号为 6049，并退出程序
    if (qh->TESTvneighbors && !qh->MERGING) {
        qh_fprintf(qh, qh->ferr, 6049, "qhull option error: test vertex neighbors('Qv') needs a merge option\n");
        qh_errexit(qh, qh_ERRinput, NULL ,NULL);
    }

    // 若启用投影输入或处于 Delaunay 且启用 Delaunay 投影选项，调整凸壳维度
    if (qh->PROJECTinput || (qh->DELAUNAY && qh->PROJECTdelaunay)) {
        qh->hull_dim -= qh->PROJECTinput;
        if (qh->DELAUNAY) {
            qh->hull_dim++;
            // 若处于无限点状态，额外维度加一
            if (qh->ATinfinity)
                extra= 1;
        }
    }

    // 若凸壳维度小于等于 1，输出错误信息到 qh 对象的 ferr 流，编号为 6050，并退出程序
    if (qh->hull_dim <= 1) {
        qh_fprintf(qh, qh->ferr, 6050, "qhull error: dimension %d must be > 1\n", qh->hull_dim);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }

    // 计算阶乘并存储到 factorial 中，k 从 2 到 hull_dim - 1
    for (k=2, factorial=1.0; k < qh->hull_dim; k++)
        factorial *= k;
    qh->AREAfactor= 1.0 / factorial;

    // 输出跟踪信息到 qh 对象的 ferr 流，编号为 2005，描述全局变量的初始化情况
    trace2((qh, qh->ferr, 2005, "qh_initqhull_globals: initialize globals.  input_dim %d, numpoints %d, malloc? %d, projected %d to hull_dim %d\n",
            qh->input_dim, numpoints, ismalloc, qh->PROJECTinput, qh->hull_dim));

    // 计算 normal_size 和 center_size 的大小，以字节为单位
    qh->normal_size= qh->hull_dim * (int)sizeof(coordT);
    qh->center_size= qh->normal_size - (int)sizeof(coordT);

    // 计算所需的点数，为 hull_dim + 1
    pointsneeded= qh->hull_dim+1;

    // 若凸壳维度大于 DIMmergeVertex，则禁用 MERGEvertices 并设置相关选项
    if (qh->hull_dim > qh_DIMmergeVertex) {
        qh->MERGEvertices= False;
        qh_option(qh, "Q3-no-merge-vertices-dim-high", NULL, NULL);
    }

    // 若存在 GOODpoint，所需点数增加一个
    if (qh->GOODpoint)
        pointsneeded++;
#ifdef qh_NOtrace
  // 如果定义了 qh_NOtrace 宏，则检查是否需要输出跟踪信息
  if (qh->IStracing || qh->TRACEmerge || qh->TRACEpoint != qh_IDnone || qh->TRACEdist < REALmax/2) {
      // 输出错误信息表明跟踪功能未安装
      qh_fprintf(qh, qh->ferr, 6051, "qhull option error: tracing is not installed (qh_NOtrace in user_r.h).  Trace options 'Tn', 'TMn', 'TPn' and 'TWn' mostly removed.  Continue with 'Qw' (allow warning)\n");
      // 如果不允许警告，则退出
      if (!qh->ALLOWwarning)
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
#endif
// 如果 RERUN 大于 1，设置跟踪上次运行的标志
if (qh->RERUN > 1) {
  qh->TRACElastrun= qh->IStracing; /* qh_build_withrestart duplicates next conditional */
  // 如果正在跟踪且不为 -1，则输出跟踪信息并清除跟踪标志
  if (qh->IStracing && qh->IStracing != -1) {
    qh_fprintf(qh, qh->ferr, 8162, "qh_initqhull_globals: trace last of TR%d runs at level %d\n", qh->RERUN, qh->IStracing);
    qh->IStracing= 0;
  }
}else if (qh->TRACEpoint != qh_IDnone || qh->TRACEdist < REALmax/2 || qh->TRACEmerge) {
  // 否则如果设置了某些跟踪选项，则设置跟踪级别
  qh->TRACElevel= (qh->IStracing ? qh->IStracing : 3);
  qh->IStracing= 0;
}
// 如果 ROTATErandom 等于 0 或者 -1，则根据时间种子设置旋转随机数种子
if (qh->ROTATErandom == 0 || qh->ROTATErandom == -1) {
  seed= (int)time(&timedata);
  // 如果 ROTATErandom 等于 -1，则取反种子并设置 QRandom-seed 选项
  if (qh->ROTATErandom  == -1) {
    seed= -seed;
    qh_option(qh, "QRandom-seed", &seed, NULL );
  }else
    qh_option(qh, "QRotate-random", &seed, NULL);
  qh->ROTATErandom= seed;
}
seed= qh->ROTATErandom;
// 如果种子为 INT_MIN，则设为默认值 1；否则如果为负数，取绝对值
if (seed == INT_MIN)    /* default value */
  seed= 1;
else if (seed < 0)
  seed= -seed;
// 使用种子设置随机数生成器
qh_RANDOMseed_(qh, seed);
randr= 0.0;
// 计算 1000 个随机数的平均值
for (i=1000; i--; ) {
  randi= qh_RANDOMint;
  randr += randi;
  // 如果随机数大于最大允许值，则输出错误信息并退出
  if (randi > qh_RANDOMmax) {
    qh_fprintf(qh, qh->ferr, 8036, "\
qhull configuration error (qh_RANDOMmax in user_r.h): random integer %d > qh_RANDOMmax (%.8g)\n",
             randi, qh_RANDOMmax);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
}
// 使用相同的种子重新设置随机数生成器
qh_RANDOMseed_(qh, seed);
// 计算平均随机数值
randr= randr/1000;
// 如果平均值远离期望值的一半，则输出警告信息
if (randr < qh_RANDOMmax * 0.1
|| randr > qh_RANDOMmax * 0.9)
  qh_fprintf(qh, qh->ferr, 8037, "\
qhull configuration warning (qh_RANDOMmax in user_r.h): average of 1000 random integers (%.2g) is much different than expected (%.2g).  Is qh_RANDOMmax (%.2g) wrong?\n",
           randr, qh_RANDOMmax * 0.5, qh_RANDOMmax);
// 设置随机数参数
qh->RANDOMa= 2.0 * qh->RANDOMfactor/qh_RANDOMmax;
qh->RANDOMb= 1.0 - qh->RANDOMfactor;
// 如果 HASHfactor 小于 1.1，则输出内部错误信息并退出
if (qh_HASHfactor < 1.1) {
  qh_fprintf(qh, qh->ferr, 6052, "qhull internal error (qh_initqhull_globals): qh_HASHfactor %d must be at least 1.1.  Qhull uses linear hash probing\n",
    qh_HASHfactor);
  qh_errexit(qh, qh_ERRqhull, NULL, NULL);
}
// 如果点数加上额外点数小于所需点数，则输出错误信息并退出
if (numpoints+extra < pointsneeded) {
  qh_fprintf(qh, qh->ferr, 6214, "qhull input error: not enough points(%d) to construct initial simplex (need %d)\n",
          numpoints, pointsneeded);
  qh_errexit(qh, qh_ERRinput, NULL, NULL);
}
// 初始化 Qhull 的输出标志
qh_initqhull_outputflags(qh);
} /* initqhull_globals */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="initqhull_mem">-</a>

  qh_initqhull_mem(qh )
    initialize mem_r.c for qhull
    qh.hull_dim and qh.normal_size determine some of the allocation sizes
    # 如果 qh.MERGING 为真，则执行以下操作
    if qh.MERGING:
        # 在这里包含 ridgeT
    
        # 调用 qh_user_memsizes 函数（位于 user_r.c 文件），用于为快速分配添加最多 10 种额外大小
        # （参见下面的 numsizes）
    
    # 返回:
    # mem_r.c 已准备好供 qh_memalloc/qh_memfree 使用（如果在此之前调用则会报错）
    
    # 注意:
    # qh_produceoutput() 函数打印 memsizes
/*
void qh_initqhull_mem(qhT *qh) {
  int numsizes;
  int i;

  numsizes= 8+10;  // 初始化 numsizes 变量为 18
  // 初始化内存缓冲区，设置大小和对齐方式
  qh_meminitbuffers(qh, qh->IStracing, qh_MEMalign, numsizes,
                     qh_MEMbufsize, qh_MEMinitbuf);
  // 分配内存给 vertexT 结构体
  qh_memsize(qh, (int)sizeof(vertexT));
  if (qh->MERGING) {
    // 如果启用了 MERGING，分配内存给 ridgeT 和 mergeT 结构体
    qh_memsize(qh, (int)sizeof(ridgeT));
    qh_memsize(qh, (int)sizeof(mergeT));
  }
  // 分配内存给 facetT 结构体
  qh_memsize(qh, (int)sizeof(facetT));
  i= (int)sizeof(setT) + (qh->hull_dim - 1) * SETelemsize;  /* ridge.vertices */
  // 计算并分配内存给 ridge.vertices
  qh_memsize(qh, i);
  // 分配内存给 normal
  qh_memsize(qh, qh->normal_size);
  i += SETelemsize;                 /* facet.vertices, .ridges, .neighbors */
  // 计算并分配内存给 facet.vertices, .ridges, .neighbors
  qh_memsize(qh, i);
  // 用户自定义内存大小设置
  qh_user_memsizes(qh);
  // 设置完成内存管理
  qh_memsetup(qh);
} * initqhull_mem */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="initqhull_outputflags">-</a>

  qh_initqhull_outputflags
    initialize flags concerned with output

  returns:
    adjust user flags as needed

  see:
    qh_clear_outputflags() resets the flags

  design:
    test for qh.PRINTgood (i.e., only print 'good' facets)
    check for conflicting print output options
*/
void qh_initqhull_outputflags(qhT *qh) {
  boolT printgeom= False, printmath= False, printcoplanar= False;
  int i;

  trace3((qh, qh->ferr, 3024, "qh_initqhull_outputflags: %s\n", qh->qhull_command));
  // 如果 PRINTgood 或 PRINTneighbors 没有设置，则根据条件设置 PRINTgood
  if (!(qh->PRINTgood || qh->PRINTneighbors)) {
    if (qh->DELAUNAY || qh->KEEParea || qh->KEEPminArea < REALmax/2 || qh->KEEPmerge
        || (!qh->ONLYgood && (qh->GOODvertex || qh->GOODpoint))) {
      qh->PRINTgood= True;
      qh_option(qh, "Pgood", NULL, NULL);
    }
  }
  // 如果 PRINTtransparent 设置为 True，则进行相关设置
  if (qh->PRINTtransparent) {
    // 如果 hull_dim 不为 4，或者不是 DELAUNAY，或者是 VORONOI，或者 DROPdim >= 0，则报错退出
    if (qh->hull_dim != 4 || !qh->DELAUNAY || qh->VORONOI || qh->DROPdim >= 0) {
      qh_fprintf(qh, qh->ferr, 6215, "qhull option error: transparent Delaunay('Gt') needs 3-d Delaunay('d') w/o 'GDn'\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    // 设置 DROPdim 和 PRINTridges
    qh->DROPdim= 3;
    qh->PRINTridges= True;
  }
  // 遍历 PRINTout 数组，设置相关的打印标志
  for (i=qh_PRINTEND; i--; ) {
    if (qh->PRINTout[i] == qh_PRINTgeom)
      printgeom= True;
    else if (qh->PRINTout[i] == qh_PRINTmathematica || qh->PRINTout[i] == qh_PRINTmaple)
      printmath= True;
    else if (qh->PRINTout[i] == qh_PRINTcoplanars)
      printcoplanar= True;
    else if (qh->PRINTout[i] == qh_PRINTpointnearest)
      printcoplanar= True;
    else if (qh->PRINTout[i] == qh_PRINTpointintersect && !qh->HALFspace) {
      // 如果 PRINTout[i] 为 qh_PRINTpointintersect 且没有设置 HALFspace，则报错退出
      qh_fprintf(qh, qh->ferr, 6053, "qhull option error: option 'Fp' is only used for \nhalfspace intersection('Hn,n,n').\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }else if (qh->PRINTout[i] == qh_PRINTtriangles && (qh->HALFspace || qh->VORONOI)) {
      // 如果 PRINTout[i] 为 qh_PRINTtriangles 且是 HALFspace 或 VORONOI，则报错退出
      qh_fprintf(qh, qh->ferr, 6054, "qhull option error: option 'Ft' is not available for Voronoi vertices ('v') or halfspace intersection ('H')\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    }else if (qh->PRINTout[i] == qh_PRINTcentrums && qh->VORONOI) {
      // 如果当前输出类型是 centrums（中心）且正在计算 Voronoi 图，则报错并退出
      qh_fprintf(qh, qh->ferr, 6055, "qhull option error: option 'FC' is not available for Voronoi vertices('v')\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }else if (qh->PRINTout[i] == qh_PRINTvertices) {
      // 如果当前输出类型是 vertices（顶点）
      if (qh->VORONOI)
        // 如果正在计算 Voronoi 图，则设置选项 Fvoronoi
        qh_option(qh, "Fvoronoi", NULL, NULL);
      else
        // 否则设置选项 Fvertices
        qh_option(qh, "Fvertices", NULL, NULL);
    }
  }
  if (printcoplanar && qh->DELAUNAY && qh->JOGGLEmax < REALmax/2) {
    // 如果设置了 printcoplanar 选项，且正在进行 Delaunay 三角化，同时 joggle 参数小于 REALmax 的一半
    if (qh->PRINTprecision)
      // 如果打印精度选项被设置，则发出警告
      qh_fprintf(qh, qh->ferr, 7041, "qhull option warning: 'QJ' (joggle) will usually prevent coincident input sites for options 'Fc' and 'FP'\n");
  }
  if (printmath && (qh->hull_dim > 3 || qh->VORONOI)) {
    // 如果设置了 printmath 选项，并且维度大于 3 或者正在计算 Voronoi 图
    qh_fprintf(qh, qh->ferr, 6056, "qhull option error: Mathematica and Maple output is only available for 2-d and 3-d convex hulls and 2-d Delaunay triangulations\n");
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  if (printgeom) {
    // 如果设置了 printgeom 选项
    if (qh->hull_dim > 4) {
      // 如果维度大于 4，则输出错误信息并退出
      qh_fprintf(qh, qh->ferr, 6057, "qhull option error: Geomview output is only available for 2-d, 3-d and 4-d\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    if (qh->PRINTnoplanes && !(qh->PRINTcoplanar + qh->PRINTcentrums
     + qh->PRINTdots + qh->PRINTspheres + qh->DOintersections + qh->PRINTridges)) {
      // 如果 PRINTnoplanes 被设置，并且没有为 Geomview 输出指定任何输出选项
      qh_fprintf(qh, qh->ferr, 6058, "qhull option error: no output specified for Geomview\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    if (qh->VORONOI && (qh->hull_dim > 3 || qh->DROPdim >= 0)) {
      // 如果正在计算 Voronoi 图，且维度大于 3 或者 DROPdim 大于等于 0
      qh_fprintf(qh, qh->ferr, 6059, "qhull option error: Geomview output for Voronoi diagrams only for 2-d\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    /* can not warn about furthest-site Geomview output: no lower_threshold */
    // 不能警告关于最远点 Geomview 输出：没有 lower_threshold
    if (qh->hull_dim == 4 && qh->DROPdim == -1 &&
        (qh->PRINTcoplanar || qh->PRINTspheres || qh->PRINTcentrums)) {
      // 如果维度为 4，并且 DROPdim 为 -1，并且 PRINTcoplanar、PRINTspheres 或 PRINTcentrums 中有一个被设置
      qh_fprintf(qh, qh->ferr, 7042, "qhull option warning: coplanars, vertices, and centrums output not available for 4-d output(ignored).  Could use 'GDn' instead.\n");
      // 清除 PRINTcoplanar、PRINTspheres 和 PRINTcentrums 的设置
      qh->PRINTcoplanar= qh->PRINTspheres= qh->PRINTcentrums= False;
    }
  }
  if (!qh->KEEPcoplanar && !qh->KEEPinside && !qh->ONLYgood) {
    // 如果没有设置 KEEPcoplanar、KEEPinside 和 ONLYgood 选项
    if ((qh->PRINTcoplanar && qh->PRINTspheres) || printcoplanar) {
      // 如果设置了 PRINTcoplanar 和 PRINTspheres 选项，或者 printcoplanar 被设置
      if (qh->QHULLfinished) {
        // 如果 QHULL 完成标志被设置，则发出警告
        qh_fprintf(qh, qh->ferr, 7072, "qhull output warning: ignoring coplanar points, option 'Qc' was not set for the first run of qhull.\n");
      }else {
        // 否则设置 KEEPcoplanar 并设置选项 Qcoplanar
        qh->KEEPcoplanar= True;
        qh_option(qh, "Qcoplanar", NULL, NULL);
      }
    }
  }
  qh->PRINTdim= qh->hull_dim;
  if (qh->DROPdim >=0) {    /* after Geomview checks */
    // 如果 DROPdim 大于等于 0（在 Geomview 检查之后）
    if (qh->DROPdim < qh->hull_dim) {
      // 如果 DROPdim 小于 hull_dim，则减少 PRINTdim，并发出警告
      qh->PRINTdim--;
      if (!printgeom || qh->hull_dim < 3)
        qh_fprintf(qh, qh->ferr, 7043, "qhull option warning: drop dimension 'GD%d' is only available for 3-d/4-d Geomview\n", qh->DROPdim);
    }else
      // 否则将 DROPdim 设置为 -1
      qh->DROPdim= -1;
  }else if (qh->VORONOI) {
    // 如果正在计算 Voronoi 图
    qh->DROPdim= qh->hull_dim-1;
    # 设置 qh 结构体中的 PRINTdim 字段为 hull_dim-1
    qh->PRINTdim= qh->hull_dim-1;
/* qh_initqhull_outputflags */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="initqhull_start">-</a>

  qh_initqhull_start(qh, infile, outfile, errfile )
    allocate memory if needed and call qh_initqhull_start2()
*/
void qh_initqhull_start(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile) {

  qh_initstatistics(qh);  /* 调用初始化统计信息函数 */
  qh_initqhull_start2(qh, infile, outfile, errfile);  /* 调用初始化 qhull 的第二步函数 */
} /* initqhull_start */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="initqhull_start2">-</a>

  qh_initqhull_start2(qh, infile, outfile, errfile )
    start initialization of qhull
    initialize statistics, stdio, default values for global variables
    assumes qh is allocated
  notes:
    report errors elsewhere, error handling and g_qhull_output [Qhull.cpp, QhullQh()] not in initialized
  see:
    qh_maxmin() determines the precision constants
    qh_freeqhull()
*/
void qh_initqhull_start2(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile) {
  time_t timedata;
  int seed;

  qh_CPUclock; /* start the clock(for qh_clock).  One-shot. */
  /* memset is the same in qh_freeqhull() and qh_initqhull_start2() */
  memset((char *)qh, 0, sizeof(qhT)-sizeof(qhmemT)-sizeof(qhstatT));   /* 将 qhT 结构体内存空间清零 */
  qh->NOerrexit= True;  /* 设置 NOerrexit 标志为真 */
  qh->DROPdim= -1;  /* 初始化 DROPdim 为 -1 */
  qh->ferr= errfile;  /* 设置 ferr 指向错误输出文件 */
  qh->fin= infile;  /* 设置 fin 指向输入文件 */
  qh->fout= outfile;  /* 设置 fout 指向输出文件 */
  qh->furthest_id= qh_IDunknown;  /* 初始化 furthest_id 为 qh_IDunknown */
#ifndef qh_NOmerge
  qh->JOGGLEmax= REALmax;  /* 如果不定义 qh_NOmerge，则初始化 JOGGLEmax 为 REALmax */
#else
  qh->JOGGLEmax= 0.0;  /* 如果定义了 qh_NOmerge，则初始化 JOGGLEmax 为 0.0 */

  /* 如果定义了 qh_NOmerge，则初始化 JOGGLEmax 为 0.0 */
#endif
#endif
  qh->KEEPminArea= REALmax;  // 设置 qh 结构体的 KEEPminArea 字段为最大实数值
  qh->last_low= REALmax;     // 设置 qh 结构体的 last_low 字段为最大实数值
  qh->last_high= REALmax;    // 设置 qh 结构体的 last_high 字段为最大实数值
  qh->last_newhigh= REALmax; // 设置 qh 结构体的 last_newhigh 字段为最大实数值
  qh->last_random= 1;        // 设置 qh 结构体的 last_random 字段为 1（仅用于可重入）
  qh->lastcpu= 0.0;          // 设置 qh 结构体的 lastcpu 字段为 0.0
  qh->max_outside= 0.0;      // 设置 qh 结构体的 max_outside 字段为 0.0
  qh->max_vertex= 0.0;       // 设置 qh 结构体的 max_vertex 字段为 0.0
  qh->MAXabs_coord= 0.0;     // 设置 qh 结构体的 MAXabs_coord 字段为 0.0
  qh->MAXsumcoord= 0.0;      // 设置 qh 结构体的 MAXsumcoord 字段为 0.0
  qh->MAXwidth= -REALmax;    // 设置 qh 结构体的 MAXwidth 字段为负的最大实数值
  qh->MERGEindependent= True; // 设置 qh 结构体的 MERGEindependent 字段为 True
  qh->MINdenom_1= fmax_(1.0/REALmax, REALmin); // 设置 qh 结构体的 MINdenom_1 字段为 1.0/REALmax 和 REALmin 中的较大值（用于 qh_scalepoints）
  qh->MINoutside= 0.0;       // 设置 qh 结构体的 MINoutside 字段为 0.0
  qh->MINvisible= REALmax;   // 设置 qh 结构体的 MINvisible 字段为最大实数值
  qh->MAXcoplanar= REALmax;  // 设置 qh 结构体的 MAXcoplanar 字段为最大实数值
  qh->outside_err= REALmax;  // 设置 qh 结构体的 outside_err 字段为最大实数值
  qh->premerge_centrum= 0.0; // 设置 qh 结构体的 premerge_centrum 字段为 0.0
  qh->premerge_cos= REALmax; // 设置 qh 结构体的 premerge_cos 字段为最大实数值
  qh->PRINTprecision= True;  // 设置 qh 结构体的 PRINTprecision 字段为 True
  qh->PRINTradius= 0.0;      // 设置 qh 结构体的 PRINTradius 字段为 0.0
  qh->postmerge_cos= REALmax;// 设置 qh 结构体的 postmerge_cos 字段为最大实数值
  qh->postmerge_centrum= 0.0;// 设置 qh 结构体的 postmerge_centrum 字段为 0.0
  qh->ROTATErandom= INT_MIN; // 设置 qh 结构体的 ROTATErandom 字段为 INT_MIN
  qh->MERGEvertices= True;   // 设置 qh 结构体的 MERGEvertices 字段为 True
  qh->totarea= 0.0;          // 设置 qh 结构体的 totarea 字段为 0.0
  qh->totvol= 0.0;           // 设置 qh 结构体的 totvol 字段为 0.0
  qh->TRACEdist= REALmax;    // 设置 qh 结构体的 TRACEdist 字段为最大实数值
  qh->TRACEpoint= qh_IDnone; // 设置 qh 结构体的 TRACEpoint 字段为 qh_IDnone（用于追踪一个点）
  qh->tracefacet_id= UINT_MAX;  // 设置 qh 结构体的 tracefacet_id 字段为 UINT_MAX（用于追踪一个 facet，追踪完成后设置为 UINT_MAX）
  qh->traceridge_id= UINT_MAX;  // 设置 qh 结构体的 traceridge_id 字段为 UINT_MAX（用于追踪一个 ridge，追踪完成后设置为 UINT_MAX）
  qh->tracevertex_id= UINT_MAX; // 设置 qh 结构体的 tracevertex_id 字段为 UINT_MAX（用于追踪一个 vertex，追踪完成后设置为 UINT_MAX）
  seed= (int)time(&timedata);  // 获取当前时间作为种子
  qh_RANDOMseed_(qh, seed);    // 使用种子初始化 qh 结构体的随机数生成器
  qh->run_id= qh_RANDOMint;    // 设置 qh 结构体的 run_id 字段为随机数生成器生成的随机数
  if(!qh->run_id)
      qh->run_id++;  // 如果 run_id 为零，则增加其值以确保非零
  qh_option(qh, "run-id", &qh->run_id, NULL);  // 设置 qh 结构体的 run-id 选项
  strcat(qh->qhull, "qhull");  // 将字符串 "qhull" 连接到 qh 结构体的 qhull 字段
} /* initqhull_start2 */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="initthresholds">-</a>

  qh_initthresholds(qh, commandString )
    set thresholds for printing and scaling from commandString

  returns:
    sets qh.GOODthreshold or qh.SPLITthreshold if 'Pd0D1' used

  see:
    qh_initflags(), 'Qbk' 'QBk' 'Pdk' and 'PDk'
    qh_inthresholds()

  design:
    for each 'Pdn' or 'PDn' option
      check syntax
      set qh.lower_threshold or qh.upper_threshold
    set qh.GOODthreshold if an unbounded threshold is used
    set qh.SPLITthreshold if a bounded threshold is used
*/
void qh_initthresholds(qhT *qh, char *command) {
  realT value;  // 用于存储阈值的实数值
  int idx, maxdim, k;  // 用于循环和维度的变量
  char *s= command; /* non-const due to strtol */  // 指向命令字符串的指针，因为要使用 strtol，所以是非常量指针
  char *lastoption, *lastwarning= NULL;  // 用于记录最后一个选项和最后一个警告的字符串指针
  char key;  // 用于存储命令中的键值

  maxdim= qh->input_dim;  // 获取输入维度的最大值
  if (qh->DELAUNAY && (qh->PROJECTdelaunay || qh->PROJECTinput))
    maxdim++;  // 如果是 Delaunay 三角剖分并且进行了投影，则增加最大维度

  while (*s) {  // 循环处理命令字符串
    if (*s == '-')  // 跳过连字符
      s++;
    if (*s == 'P') {
      // 检查是否是以 'P' 开头的选项
      lastoption= s++;
      // 记录上一个选项，并将指针移到下一个字符
      while (*s && !isspace(key= *s++)) {
        // 遍历直到遇到空白字符，key 是当前字符
        if (key == 'd' || key == 'D') {
          // 如果 key 是 'd' 或 'D'
          if (!isdigit(*s)) {
            // 如果下一个字符不是数字
            qh_fprintf(qh, qh->ferr, 7044, "qhull option warning: no dimension given for Print option 'P%c' at: %s.  Ignored\n",
                    key, s-1);
            // 输出警告信息，表示没有给出维度信息
            lastwarning= lastoption;
            // 记录最近的警告选项
            continue;
          }
          // 解析维度信息
          idx= qh_strtol(s, &s);
          // 将 s 解析为整数，更新 s 指针
          if (idx >= qh->hull_dim) {
            // 如果解析的维度超过了 qhull 的维度
            qh_fprintf(qh, qh->ferr, 7045, "qhull option warning: dimension %d for Print option 'P%c' is >= %d.  Ignored\n",
                idx, key, qh->hull_dim);
            // 输出警告信息，表示维度超出范围
            lastwarning= lastoption;
            // 记录最近的警告选项
            continue;
          }
          if (*s == ':') {
            // 如果下一个字符是 ':'
            s++;
            // 移动到下一个字符
            value= qh_strtod(s, &s);
            // 解析数值信息，更新 s 指针
            if (fabs((double)value) > 1.0) {
              // 如果值的绝对值超过了 1.0
              qh_fprintf(qh, qh->ferr, 7046, "qhull option warning: value %2.4g for Print option 'P%c' is > +1 or < -1.  Ignored\n",
                      value, key);
              // 输出警告信息，表示值超出范围
              lastwarning= lastoption;
              // 记录最近的警告选项
              continue;
            }
          }else
            value= 0.0;
          // 如果没有 ':'，则默认值为 0.0
          if (key == 'd')
            qh->lower_threshold[idx]= value;
          else
            qh->upper_threshold[idx]= value;
          // 根据 key 更新阈值数组中的值
        }
      }
    }else if (*s == 'Q') {
      // 检查是否是以 'Q' 开头的选项
      lastoption= s++;
      // 记录上一个选项，并将指针移到下一个字符
      while (*s && !isspace(key= *s++)) {
        // 遍历直到遇到空白字符，key 是当前字符
        if (key == 'b' && *s == 'B') {
          // 如果 key 是 'b' 并且下一个字符是 'B'
          s++;
          // 跳过 'B'
          for (k=maxdim; k--; ) {
            // 遍历维度的上下界数组
            qh->lower_bound[k]= -qh_DEFAULTbox;
            // 设置下界为默认值的负值
            qh->upper_bound[k]= qh_DEFAULTbox;
            // 设置上界为默认值
          }
        }else if (key == 'b' && *s == 'b')
          // 如果 key 是 'b' 并且下一个字符是 'b'
          s++;
          // 跳过 'b'
        else if (key == 'b' || key == 'B') {
          // 如果 key 是 'b' 或 'B'
          if (!isdigit(*s)) {
            // 如果下一个字符不是数字
            qh_fprintf(qh, qh->ferr, 7047, "qhull option warning: no dimension given for Qhull option 'Q%c'\n",
                    key);
            // 输出警告信息，表示没有给出维度信息
            lastwarning= lastoption;            
            // 记录最近的警告选项
            continue;
          }
          // 解析维度信息
          idx= qh_strtol(s, &s);
          // 将 s 解析为整数，更新 s 指针
          if (idx >= maxdim) {
            // 如果解析的维度超过了最大维度
            qh_fprintf(qh, qh->ferr, 7048, "qhull option warning: dimension %d for Qhull option 'Q%c' is >= %d.  Ignored\n",
                idx, key, maxdim);
            // 输出警告信息，表示维度超出范围
            lastwarning= lastoption;            
            // 记录最近的警告选项
            continue;
          }
          if (*s == ':') {
            // 如果下一个字符是 ':'
            s++;
            // 移动到下一个字符
            value= qh_strtod(s, &s);
          }else if (key == 'b')
            value= -qh_DEFAULTbox;
          else
            value= qh_DEFAULTbox;
          // 如果没有 ':'，则根据 key 设置默认值
          if (key == 'b')
            qh->lower_bound[idx]= value;
          else
            qh->upper_bound[idx]= value;
          // 根据 key 更新维度上下界数组中的值
        }
      }
    }else {
      // 如果不是以 'P' 或 'Q' 开头的选项
      while (*s && !isspace(*s))
        // 跳过非空白字符
        s++;
    }
    // 跳过空白字符
    while (isspace(*s))
      s++;
  }
  // 遍历完成后更新一些阈值信息
  for (k=qh->hull_dim; k--; ) {
    // 遍历 qhull 的维度
    if (qh->lower_threshold[k] > -REALmax/2) {
      // 如果下界大于 -REALmax/2
      qh->GOODthreshold= True;
      // 设置 GOODthreshold 为 True
      if (qh->upper_threshold[k] < REALmax/2) {
        // 如果上界小于 REALmax/2
        qh->SPLITthresholds= True;
        // 设置 SPLITthresholds 为 True
        qh->GOODthreshold= False;
        // 设置 GOODthreshold 为 False
        break;
        // 结束循环
      }
    }
  }
    }else if (qh->upper_threshold[k] < REALmax/2)
      qh->GOODthreshold= True;


    // 如果当前处理的阈值小于 REALmax 的一半，则将 GOODthreshold 设置为 True
    } else if (qh->upper_threshold[k] < REALmax/2)
        qh->GOODthreshold = True;



  }


  // 结束当前的 for 循环
  }



  if (lastwarning && !qh->ALLOWwarning) {


  // 如果存在上一个警告并且不允许警告（ALLOWwarning 为假），则执行以下操作
  if (lastwarning && !qh->ALLOWwarning) {



    qh_fprintf(qh, qh->ferr, 6036, "qhull option error: see previous warnings, use 'Qw' to override: '%s' (last offset %d)\n", 
      command, (int)(lastwarning-command));


    // 使用 qh_fprintf 将错误消息输出到 qh->ferr 流中，包括警告信息和命令行位置偏移
    qh_fprintf(qh, qh->ferr, 6036, "qhull option error: see previous warnings, use 'Qw' to override: '%s' (last offset %d)\n", 
      command, (int)(lastwarning-command));



    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }


    // 使用 qh_errexit 终止程序执行，以 qh_ERRinput 错误类型退出
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
} /* initthresholds */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="lib_check">-</a>

  qh_lib_check( qhullLibraryType, qhTsize, vertexTsize, ridgeTsize, facetTsize, setTsize, qhmemTsize )
    Report error if library does not agree with caller

  notes:
    NOerrors -- qh_lib_check can not call qh_errexit()
*/
void qh_lib_check(int qhullLibraryType, int qhTsize, int vertexTsize, int ridgeTsize, int facetTsize, int setTsize, int qhmemTsize) {
    int last_errcode= qh_ERRnone;

#if defined(_MSC_VER) && defined(_DEBUG) && defined(QHULL_CRTDBG)  /* user_r.h */
    /*_CrtSetBreakAlloc(744);*/  /* Break at memalloc {744}, or 'watch' _crtBreakAlloc */
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) );
    _CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );
    _CrtSetReportFile( _CRT_ERROR, _CRTDBG_FILE_STDERR );
    _CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );
    _CrtSetReportFile( _CRT_WARN, _CRTDBG_FILE_STDERR );
    _CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );
    _CrtSetReportFile( _CRT_ASSERT, _CRTDBG_FILE_STDERR );
#endif

    if (qhullLibraryType==QHULL_NON_REENTRANT) { /* 0 */
      // 如果调用者使用不支持重新进入的 Qhull 库，报告错误
      qh_fprintf_stderr(6257, "qh_lib_check: Incorrect qhull library called.  Caller uses non-reentrant Qhull with a static qhT.  Qhull library is reentrant.\n");
      last_errcode= 6257;
    }else if (qhullLibraryType==QHULL_QH_POINTER) { /* 1 */
      // 如果调用者使用不支持重新进入的 Qhull 库，但使用了动态 qhT，报告错误
      qh_fprintf_stderr(6258, "qh_lib_check: Incorrect qhull library called.  Caller uses non-reentrant Qhull with a dynamic qhT via qh_QHpointer.  Qhull library is reentrant.\n");
      last_errcode= 6258;
    }else if (qhullLibraryType != QHULL_REENTRANT) { /* 2 */
      // 如果 qhullLibraryType 不是预期的 QHULL_NON_REENTRANT(0), QHULL_QH_POINTER(1), 或 QHULL_REENTRANT(2)，报告错误
      qh_fprintf_stderr(6262, "qh_lib_check: Expecting qhullLibraryType QHULL_NON_REENTRANT(0), QHULL_QH_POINTER(1), or QHULL_REENTRANT(2).  Got %d\n", qhullLibraryType);
      last_errcode= 6262;
    }
    // 检查 qhT 的大小是否正确
    if (qhTsize != (int)sizeof(qhT)) {
      qh_fprintf_stderr(6249, "qh_lib_check: Incorrect qhull library called.  Size of qhT for caller is %d, but for qhull library is %d.\n", qhTsize, (int)sizeof(qhT));
      last_errcode= 6249;
    }
    // 检查 vertexT 的大小是否正确
    if (vertexTsize != (int)sizeof(vertexT)) {
      qh_fprintf_stderr(6250, "qh_lib_check: Incorrect qhull library called.  Size of vertexT for caller is %d, but for qhull library is %d.\n", vertexTsize, (int)sizeof(vertexT));
      last_errcode= 6250;
    }
    // 检查 ridgeT 的大小是否正确
    if (ridgeTsize != (int)sizeof(ridgeT)) {
      qh_fprintf_stderr(6251, "qh_lib_check: Incorrect qhull library called.  Size of ridgeT for caller is %d, but for qhull library is %d.\n", ridgeTsize, (int)sizeof(ridgeT));
      last_errcode= 6251;
    }
    # 检查 facetT 的大小是否与当前 sizeof(facetT) 的大小相符，如果不符则输出错误信息并记录错误码
    if (facetTsize != (int)sizeof(facetT)) {
      qh_fprintf_stderr(6252, "qh_lib_check: Incorrect qhull library called.  Size of facetT for caller is %d, but for qhull library is %d.\n", facetTsize, (int)sizeof(facetT));
      last_errcode= 6252;
    }
    
    # 如果 setTsize 非零且与当前 sizeof(setT) 的大小不符，则输出错误信息并记录错误码
    if (setTsize && setTsize != (int)sizeof(setT)) {
      qh_fprintf_stderr(6253, "qh_lib_check: Incorrect qhull library called.  Size of setT for caller is %d, but for qhull library is %d.\n", setTsize, (int)sizeof(setT));
      last_errcode= 6253;
    }
    
    # 如果 qhmemTsize 非零且与当前 sizeof(qhmemT) 的大小不符，则输出错误信息并记录错误码
    if (qhmemTsize && qhmemTsize != sizeof(qhmemT)) {
      qh_fprintf_stderr(6254, "qh_lib_check: Incorrect qhull library called.  Size of qhmemT for caller is %d, but for qhull library is %d.\n", qhmemTsize, sizeof(qhmemT));
      last_errcode= 6254;
    }
    
    # 如果存在上一个错误码，则输出错误信息并退出程序
    if (last_errcode) {
      qh_fprintf_stderr(6259, "qhull internal error (qh_lib_check): Cannot continue due to QH%d.  '%s' is not reentrant (e.g., qhull.so) or out-of-date.  Exit with %d\n", 
            last_errcode, qh_version2, last_errcode - 6200);
      qh_exit(last_errcode - 6200);  /* can not use qh_errexit(), must be less than 255 */
    }
/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="option">-</a>

  qh_option(qh, option, intVal, realVal )
    将选项描述添加到 qh.qhull_options 中

  notes:
    NOerrors -- qh_option 不能调用 qh_errexit() [qh_initqhull_start2]
    选项将会与统计数据 ('Ts') 和错误一起打印
    strlen(option) < 40
*/
void qh_option(qhT *qh, const char *option, int *i, realT *r) {
  char buf[200];
  int buflen, remainder;

  if (strlen(option) > sizeof(buf)-30-30) {
    qh_fprintf(qh, qh->ferr, 6408, "qhull internal error (qh_option): option (%d chars) has more than %d chars.  May overflow temporary buffer.  Option '%s'\n",
        (int)strlen(option), (int)sizeof(buf)-30-30, option);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  sprintf(buf, "  %s", option);
  if (i)
    sprintf(buf+strlen(buf), " %d", *i);
  if (r)
    sprintf(buf+strlen(buf), " %2.2g", *r);
  buflen= (int)strlen(buf);   /* WARN64 */
  qh->qhull_optionlen += buflen;
  remainder= (int)(sizeof(qh->qhull_options) - strlen(qh->qhull_options)) - 1;    /* WARN64 */
  maximize_(remainder, 0);
  if (qh->qhull_optionlen >= qh_OPTIONline && remainder > 0) {
    strncat(qh->qhull_options, "\n", (unsigned int)remainder);
    --remainder;
    qh->qhull_optionlen= buflen;
  }
  if (buflen > remainder) {
    trace1((qh, qh->ferr, 1058, "qh_option: option would overflow qh.qhull_options. Truncated '%s'\n", buf));
  }
  strncat(qh->qhull_options, buf, (unsigned int)remainder);
} /* option */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="zero">-</a>

  qh_zero( qh, errfile )
    初始化和清零 Qhull 的内存，用于 qh_new_qhull()

  notes:
    在 global_r.c 中不需要，因为静态变量会被初始化为零
*/
void qh_zero(qhT *qh, FILE *errfile) {
    memset((char *)qh, 0, sizeof(qhT));   /* 每个字段都被设为 0, FALSE, NULL */
    qh->NOerrexit= True;
    qh_meminit(qh, errfile);
} /* zero */
```