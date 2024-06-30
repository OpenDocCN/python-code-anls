# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\user_r.c`

```
/*
   user_r.c
   user redefinable functions

   see user2_r.c for qh_fprintf, qh_malloc, qh_free

   see README.txt  see COPYING.txt for copyright information.

   see libqhull_r.h for data structures, macros, and user-callable functions.

   see user_eg_r.c, user_eg2_r.c, and unix_r.c for examples.

   see user_r.h for user-definable constants

      use qh_NOmem in mem_r.h to turn off memory management
      use qh_NOmerge in user_r.h to turn off facet merging
      set qh_KEEPstatistics in user_r.h to 0 to turn off statistics

   This is unsupported software.  You're welcome to make changes,
   but you're on your own if something goes wrong.  Use 'Tc' to
   check frequently.  Usually qhull will report an error if
   a data structure becomes inconsistent.  If so, it also reports
   the last point added to the hull, e.g., 102.  You can then trace
   the execution of qhull with "T4P102".

   Please report any errors that you fix to qhull@qhull.org

   Qhull-template is a template for calling qhull from within your application

   if you recompile and load this module, then user.o will not be loaded
   from qhull.a

   you can add additional quick allocation sizes in qh_user_memsizes

   if the other functions here are redefined to not use qh_print...,
   then io.o will not be loaded from qhull.a.  See user_eg_r.c for an
   example.  We recommend keeping io.o for the extra debugging
   information it supplies.
*/

#include "qhull_ra.h"  // 包含 qhull 的 reentrant 接口头文件

#include <stdarg.h>  // 包含标准可变参数头文件

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="qhull_template">-</a>

  Qhull-template
    Template for calling qhull from inside your program

  returns:
    exit code(see qh_ERR... in libqhull_r.h)
    all memory freed

  notes:
    This can be called any number of times.
*/
#if 0
{
  int dim;                  /* points的维度 */
  int numpoints;            /* 点的数量 */
  coordT *points;           /* 每个点的坐标数组 */
  boolT ismalloc;           /* 如果qhull应在qh_freeqhull()或重新分配中释放points，则为True */
  char flags[]= "qhull Tv"; /* qhull的选项标志，参见html/qh-quick.htm */
  FILE *outfile= stdout;    /* qh_produce_output的输出流
                               使用NULL跳过qh_produce_output */
  FILE *errfile= stderr;    /* qhull代码的错误消息 */
  int exitcode;             /* 如果qhull没有错误，则为0 */
  facetT *facet;            /* 由FORALLfacets设置 */
  int curlong, totlong;     /* qh_memfreeshort后的剩余内存 */

  qhT qh_qh;                /* Qhull的数据结构。大多数调用的第一个参数 */
  qhT *qh= &qh_qh;          /* 或者 -- qhT *qh= (qhT *)malloc(sizeof(qhT)) */

  QHULL_LIB_CHECK /* 检查兼容的库 */

  qh_zero(qh, errfile);

  /* 初始化dim, numpoints, points[], ismalloc */
  exitcode= qh_new_qhull(qh, dim, numpoints, points, ismalloc,
                      flags, outfile, errfile);
  if (!exitcode) {                  /* 如果没有错误 */
    /* 'qh->facet_list' 包含凸包 */
    FORALLfacets {
       /* ... 你的代码 ... */
    }
  }
  qh_freeqhull(qh, !qh_ALL);
  qh_memfreeshort(qh, &curlong, &totlong);
  if (curlong || totlong)
    qh_fprintf(qh, errfile, 7079, "qhull internal warning (main): did not free %d bytes of long memory(%d pieces)\n", totlong, curlong);
}
#endif

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="new_qhull">-</a>

  qh_new_qhull(qh, dim, numpoints, points, ismalloc, qhull_cmd, outfile, errfile )
    运行qhull
    在第一次调用之前，要么调用qh_zero(qh, errfile)，要么将qh设置为全零状态。

  返回:
    结果存储在qh中
    exitcode (如果没有错误则为0)

  注意:
    在完成结果之前不要修改points。
      qhull数据结构包含指向points数组的指针。
    在调用qh_new_qhull()之前不要调用qhull函数。
      qhull数据结构在调用qh_new_qhull()之前未初始化。
    不要调用qh_init_A (global_r.c)

    默认的errfile是stderr，outfile可以为null。
    qhull_cmd必须以"qhull "开头。
    对于Delaunay三角剖分 ('d' 和 'v')，将points投影到一个新的点数组。
    对于半空间交集 ('H')，将points转换为一个新的点数组。

  参见:
    本文件开头的Qhull-template。
    使用qh_new_qhull的示例在user_eg_r.c中。
*/
/* 
   Qhull library function for creating a new convex hull using specified parameters and input points.

   Parameters:
   - qh: Pointer to Qhull data structure
   - dim: Dimension of the points (number of coordinates per point)
   - numpoints: Number of points
   - points: Array of coordinates of input points
   - ismalloc: Boolean indicating if 'points' array is dynamically allocated
   - qhull_cmd: Command string for qhull
   - outfile: File pointer for output
   - errfile: File pointer for error messages

   Returns:
   - exitcode: Integer exit status of the qhull operation
*/

int qh_new_qhull(qhT *qh, int dim, int numpoints, coordT *points, boolT ismalloc,
                char *qhull_cmd, FILE *outfile, FILE *errfile) {
  /* gcc may issue a "might be clobbered" warning for dim, points, and ismalloc [-Wclobbered].
     These parameters are not referenced after a longjmp() and hence not clobbered.
     See http://stackoverflow.com/questions/7721854/what-sense-do-these-clobbered-variable-warnings-make */
  
  int exitcode, hulldim;   /* Declaration of integer variables 'exitcode' and 'hulldim' */
  boolT new_ismalloc;      /* Declaration of boolean variable 'new_ismalloc' */
  coordT *new_points;      /* Declaration of pointer to coordinates 'new_points' */

  if(!errfile){
    errfile= stderr;   /* Assign stderr to errfile if it is NULL */
  }
  if (!qh->qhmem.ferr) {
    qh_meminit(qh, errfile);   /* Initialize qhull memory if not already initialized */
  } else {
    qh_memcheck(qh);   /* Check qhull memory */
  }
  if (strncmp(qhull_cmd, "qhull ", (size_t)6) && strcmp(qhull_cmd, "qhull") != 0) {
    qh_fprintf(qh, errfile, 6186, "qhull error (qh_new_qhull): start qhull_cmd argument with \"qhull \" or set to \"qhull\"\n");
    return qh_ERRinput;   /* Error message if qhull_cmd does not start with "qhull " or "qhull" */
  }
  qh_initqhull_start(qh, NULL, outfile, errfile);   /* Initialize qhull */
  if(numpoints==0 && points==NULL){
      trace1((qh, qh->ferr, 1047, "qh_new_qhull: initialize Qhull\n"));
      return 0;   /* Initialize Qhull if no points are provided */
  }
  trace1((qh, qh->ferr, 1044, "qh_new_qhull: build new Qhull for %d %d-d points with %s\n", numpoints, dim, qhull_cmd));
  exitcode= setjmp(qh->errexit);   /* Set jump point for error handling */
  if (!exitcode){
    qh->NOerrexit= False;
    qh_initflags(qh, qhull_cmd);   /* Initialize qhull flags */
    if (qh->DELAUNAY)
      qh->PROJECTdelaunay= True;   /* Set PROJECTdelaunay flag for Delaunay triangulation */
    if (qh->HALFspace) {
      /* points is an array of halfspaces,
         the last coordinate of each halfspace is its offset */
      hulldim= dim-1;   /* Dimension for halfspaces */
      qh_setfeasible(qh, hulldim);   /* Set feasible point for halfspaces */
      new_points= qh_sethalfspace_all(qh, dim, numpoints, points, qh->feasible_point);   /* Set halfspace points */
      new_ismalloc= True;   /* Set new_ismalloc to true for newly allocated memory */
      if (ismalloc)
        qh_free(points);   /* Free memory if points were dynamically allocated */
    }else {
      hulldim= dim;   /* Dimension for regular points */
      new_points= points;   /* Use existing points */
      new_ismalloc= ismalloc;   /* Use existing memory allocation status */
    }
    qh_init_B(qh, new_points, numpoints, hulldim, new_ismalloc);   /* Initialize B data structure */
    qh_qhull(qh);   /* Perform qhull computation */
    qh_check_output(qh);   /* Check qhull output consistency */
    if (outfile) {
      qh_produce_output(qh);   /* Produce output if outfile is provided */
    }else {
      qh_prepare_output(qh);   /* Prepare output if outfile is not provided */
    }
    if (qh->VERIFYoutput && !qh->FORCEoutput && !qh->STOPadd && !qh->STOPcone && !qh->STOPpoint)
      qh_check_points(qh);   /* Verify output points */
  }
  qh->NOerrexit= True;   /* Set NOerrexit flag to true */
  return exitcode;   /* Return exit code from qhull operation */
} /* new_qhull */
void qh_errexit(qhT *qh, int exitcode, facetT *facet, ridgeT *ridge) {
    // 清空跟踪变量，避免通过 qh_fprintf 陷入无限递归
    qh->tracefacet= NULL;
    qh->traceridge= NULL;
    qh->tracevertex= NULL;
    // 如果 ERREXITcalled 已经为 True，处理之前的错误并退出程序
    if (qh->ERREXITcalled) {
        qh_fprintf(qh, qh->ferr, 8126, "\nqhull error while handling previous error in qh_errexit.  Exit program\n");
        qh_exit(qh_ERRother);
    }
    qh->ERREXITcalled= True;  // 标记 ERREXITcalled 为 True
    // 如果 Qhull 还未完成，计算执行时间
    if (!qh->QHULLfinished)
        qh->hulltime= qh_CPUclock - qh->hulltime;
    // 打印错误信息
    qh_errprint(qh, "ERRONEOUS", facet, NULL, ridge, NULL);
    // 设置 _maxoutside 选项，并打印相关信息
    qh_option(qh, "_maxoutside", NULL, &qh->MAXoutside);
    qh_fprintf(qh, qh->ferr, 8127, "\nWhile executing: %s | %s\n", qh->rbox_command, qh->qhull_command);
    qh_fprintf(qh, qh->ferr, 8128, "Options selected for Qhull %s:\n%s\n", qh_version, qh->qhull_options);
    // 如果 furthest_id >= 0，打印相关信息
    if (qh->furthest_id >= 0) {
        qh_fprintf(qh, qh->ferr, 8129, "Last point added to hull was p%d.", qh->furthest_id);
        // 如果存在 zzval_(Ztotmerge)，打印相关信息
        if (zzval_(Ztotmerge))
            qh_fprintf(qh, qh->ferr, 8130, "  Last merge was #%d.", zzval_(Ztotmerge));
        // 根据 QHULLfinished 和 POSTmerging 打印相关状态信息
        if (qh->QHULLfinished)
            qh_fprintf(qh, qh->ferr, 8131, "\nQhull has finished constructing the hull.");
        else if (qh->POSTmerging)
            qh_fprintf(qh, qh->ferr, 8132, "\nQhull has started post-merging.");
        qh_fprintf(qh, qh->ferr, 8133, "\n");
    }
    // 如果 FORCEoutput 为真，并且 QHULLfinished 为真，或者 facet 和 ridge 都为 NULL，生成输出
    if (qh->FORCEoutput && (qh->QHULLfinished || (!facet && !ridge)))
        qh_produce_output(qh);
    // 如果 exitcode 不等于 qh_ERRinput
    else if (exitcode != qh_ERRinput) {
        // 如果 exitcode 不等于 qh_ERRsingular，并且 zzval_(Zsetplane) 大于 hull_dim+1，打印摘要信息
        if (exitcode != qh_ERRsingular && zzval_(Zsetplane) > qh->hull_dim+1) {
            qh_fprintf(qh, qh->ferr, 8134, "\nAt error exit:\n");
            qh_printsummary(qh, qh->ferr);
            // 如果 PRINTstatistics 为真，收集和打印统计信息
            if (qh->PRINTstatistics) {
                qh_collectstatistics(qh);
                qh_allstatistics(qh);
                qh_printstatistics(qh, qh->ferr, "at error exit");
                qh_memstatistics(qh, qh->ferr);
            }
        }
        // 如果 PRINTprecision 为真，打印精度统计信息
        if (qh->PRINTprecision)
            qh_printstats(qh, qh->ferr, qh->qhstat.precision, NULL);
    }
    // 如果 exitcode 为 0，则将 exitcode 设置为 qh_ERRother
    if (!exitcode)
        exitcode= qh_ERRother;
    // 根据 exitcode 的不同值，选择打印帮助信息
    else if (exitcode == qh_ERRprec && !qh->PREmerge)
        qh_printhelp_degenerate(qh, qh->ferr);
    else if (exitcode == qh_ERRqhull)
        qh_printhelp_internal(qh, qh->ferr);
    else if (exitcode == qh_ERRsingular)
        qh_printhelp_singular(qh, qh->ferr);
    else if (exitcode == qh_ERRdebug)
        qh_fprintf(qh, qh->ferr, 8016, "qhull exit due to qh_ERRdebug\n");
    else if (exitcode == qh_ERRtopology || exitcode == qh_ERRwide || exitcode == qh_ERRprec) {
        // 如果 NOpremerge 为真，并且没有在进行 MERGING，打印帮助信息
        if (qh->NOpremerge && !qh->MERGING)
            qh_printhelp_degenerate(qh, qh->ferr);
        // 根据 exitcode 的值，打印不同的帮助信息
        else if (exitcode == qh_ERRtopology)
            qh_printhelp_topology(qh, qh->ferr);
        else if (exitcode == qh_ERRwide)
            qh_printhelp_wide(qh, qh->ferr);
    } else if (exitcode > 255) {
        // 如果 exitcode 大于 255，打印错误信息并将 exitcode 设置为 255
        qh_fprintf(qh, qh->ferr, 6426, "qhull internal error (qh_errexit): exit code %d is greater than 255.  Invalid argument for exit().  Replaced with 255\n", exitcode);
        exitcode= 255;
    }
    // 如果 NOerrexit 为真，则不执行退出操作
    # 使用 qh_fprintf 函数将错误消息输出到 qh 对象的 ferr 流中，错误码为 6187
    # 错误消息包括详细的错误描述和退出状态码
    qh_fprintf(qh, qh->ferr, 6187, "qhull internal error (qh_errexit): either error while reporting error QH%d, or qh.NOerrexit not cleared after setjmp(). Exit program with error status %d\n",
         qh->last_errcode, exitcode);
    # 调用 qh_exit 函数，退出程序并返回指定的退出状态码
    qh_exit(exitcode);
  }
  # 将 ERREXITcalled 标志设置为 False，表示没有调用过 ERREXIT 函数
  qh->ERREXITcalled= False;
  # 将 NOerrexit 标志设置为 True，表示禁止退出操作
  qh->NOerrexit= True;
  # 将 ALLOWrestart 标志设置为 False，表示不允许重新启动操作，longjmp 将取消 qh_build_withrestart
  qh->ALLOWrestart= False;  /* longjmp will undo qh_build_withrestart */
  # 使用 longjmp 函数跳转到 qh->errexit 处，并传递退出状态码作为参数
  longjmp(qh->errexit, exitcode);
/* errprint */

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="errprint">-</a>

  qh_errprint(qh, fp, string, atfacet, otherfacet, atridge, atvertex )
    prints out the information of facets and ridges to fp
    also prints neighbors and geomview output

  notes:
    except for string, any parameter may be NULL
*/
void qh_errprint(qhT *qh, const char *string, facetT *atfacet, facetT *otherfacet, ridgeT *atridge, vertexT *atvertex) {
  int i;

  if (atvertex) {
    qh_fprintf(qh, qh->ferr, 8138, "%s VERTEX:\n", string);  /* 输出顶点信息到错误流 */
    qh_printvertex(qh, qh->ferr, atvertex);  /* 打印顶点信息 */
  }
  if (atridge) {
    qh_fprintf(qh, qh->ferr, 8137, "%s RIDGE:\n", string);  /* 输出棱信息到错误流 */
    qh_printridge(qh, qh->ferr, atridge);  /* 打印棱信息 */
    if (!atfacet)
      atfacet= atridge->top;  /* 设置当前面为棱的顶面 */
    if (!otherfacet)
      otherfacet= otherfacet_(atridge, atfacet);  /* 设置另一个面为与当前棱相邻的面 */
    if (atridge->top && atridge->top != atfacet && atridge->top != otherfacet)
      qh_printfacet(qh, qh->ferr, atridge->top);  /* 如果顶面存在且不是当前面或相邻面，则打印顶面 */
    if (atridge->bottom && atridge->bottom != atfacet && atridge->bottom != otherfacet)
      qh_printfacet(qh, qh->ferr, atridge->bottom);  /* 如果底面存在且不是当前面或相邻面，则打印底面 */
  }
  if (atfacet) {
    qh_fprintf(qh, qh->ferr, 8135, "%s FACET:\n", string);  /* 输出面信息到错误流 */
    qh_printfacet(qh, qh->ferr, atfacet);  /* 打印面信息 */
  }
  if (otherfacet) {
    qh_fprintf(qh, qh->ferr, 8136, "%s OTHER FACET:\n", string);  /* 输出另一个面信息到错误流 */
    qh_printfacet(qh, qh->ferr, otherfacet);  /* 打印另一个面信息 */
  }
  if (qh->fout && qh->FORCEoutput && atfacet && !qh->QHULLfinished && !qh->IStracing) {
    qh_fprintf(qh, qh->ferr, 8139, "ERRONEOUS and NEIGHBORING FACETS to output\n");  /* 输出错误和相邻面到错误流 */
    for (i=0; i < qh_PRINTEND; i++)  /* 遍历所有打印选项 */
      qh_printneighborhood(qh, qh->fout, qh->PRINTout[i], atfacet, otherfacet,
                            !qh_ALL);  /* 打印面的邻域信息 */
  }
} /* errprint */


/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printfacetlist">-</a>

  qh_printfacetlist(qh, fp, facetlist, facets, printall )
    print all fields for a facet list and/or set of facets to fp
    if !printall,
      only prints good facets

  notes:
    also prints all vertices
*/
void qh_printfacetlist(qhT *qh, facetT *facetlist, setT *facets, boolT printall) {
  facetT *facet, **facetp;

  if (facetlist)
    qh_checklists(qh, facetlist);  /* 检查面列表的有效性 */
  qh_fprintf(qh, qh->ferr, 9424, "printfacetlist: vertices\n");  /* 输出顶点信息到错误流 */
  qh_printbegin(qh, qh->ferr, qh_PRINTfacets, facetlist, facets, printall);  /* 打印面的开始信息 */
  if (facetlist) {
    qh_fprintf(qh, qh->ferr, 9413, "printfacetlist: facetlist\n");  /* 输出面列表信息到错误流 */
    FORALLfacet_(facetlist)
      qh_printafacet(qh, qh->ferr, qh_PRINTfacets, facet, printall);  /* 打印单个面的信息 */
  }
  if (facets) {
    qh_fprintf(qh, qh->ferr, 9414, "printfacetlist: %d facets\n", qh_setsize(qh, facets));  /* 输出面集合的大小到错误流 */
    FOREACHfacet_(facets)
      qh_printafacet(qh, qh->ferr, qh_PRINTfacets, facet, printall);  /* 打印单个面的信息 */
  }
  qh_fprintf(qh, qh->ferr, 9412, "printfacetlist: end\n");  /* 输出结束信息到错误流 */
  qh_printend(qh, qh->ferr, qh_PRINTfacets, facetlist, facets, printall);  /* 打印面的结束信息 */
} /* printfacetlist */
/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printhelp_degenerate">-</a>

  qh_printhelp_degenerate(qh, fp )
    prints descriptive message for precision error with qh_ERRprec

  notes:
    no message if qh_QUICKhelp
*/
void qh_printhelp_degenerate(qhT *qh, FILE *fp) {

  // 检查是否满足任一精度错误条件，然后输出相应的错误信息到指定文件流
  if (qh->MERGEexact || qh->PREmerge || qh->JOGGLEmax < REALmax/2)
    qh_fprintf(qh, fp, 9368, "\n\
A Qhull error has occurred.  Qhull should have corrected the above\n\
precision error.  Please send the input and all of the output to\n\
qhull_bug@qhull.org\n");
  // 如果非快速帮助模式(qh_QUICKhelp为假)，输出详细的精度问题解决方案和建议
  else if (!qh_QUICKhelp) {
    qh_fprintf(qh, fp, 9369, "\n\
Precision problems were detected during construction of the convex hull.\n\
This occurs because convex hull algorithms assume that calculations are\n\
exact, but floating-point arithmetic has roundoff errors.\n\
\n\
To correct for precision problems, do not use 'Q0'.  By default, Qhull\n\
selects 'C-0' or 'Qx' and merges non-convex facets.  With option 'QJ',\n\
Qhull joggles the input to prevent precision problems.  See \"Imprecision\n\
in Qhull\" (qh-impre.htm).\n\
\n\
If you use 'Q0', the output may include\n\
coplanar ridges, concave ridges, and flipped facets.  In 4-d and higher,\n\
Qhull may produce a ridge with four neighbors or two facets with the same \n\
vertices.  Qhull reports these events when they occur.  It stops when a\n\
concave ridge, flipped facet, or duplicate facet occurs.\n");
#if REALfloat
    qh_fprintf(qh, fp, 9370, "\
\n\
Qhull is currently using single precision arithmetic.  The following\n\
will probably remove the precision problems:\n\
  - recompile qhull for realT precision(#define REALfloat 0 in user_r.h).\n");
#endif
    // 如果正在计算 Delaunay 三角化且满足特定条件，输出相关建议
    if (qh->DELAUNAY && !qh->SCALElast && qh->MAXabs_coord > 1e4)
      qh_fprintf(qh, fp, 9371, "\
\n\
When computing the Delaunay triangulation of coordinates > 1.0,\n\
  - use 'Qbb' to scale the last coordinate to [0,m] (max previous coordinate)\n");
    // 如果正在计算 Delaunay 三角化且未添加无限远点，输出相关建议
    if (qh->DELAUNAY && !qh->ATinfinity)
      qh_fprintf(qh, fp, 9372, "\
When computing the Delaunay triangulation:\n\
  - use 'Qz' to add a point at-infinity.  This reduces precision problems.\n");

    // 输出关于三角形输出选项的建议
    qh_fprintf(qh, fp, 9373, "\
\n\
If you need triangular output:\n\
  - use option 'Qt' to triangulate the output\n\
  - use option 'QJ' to joggle the input points and remove precision errors\n\
  - use option 'Ft'.  It triangulates non-simplicial facets with added points.\n\
\n\
If you must use 'Q0',\n\
try one or more of the following options.  They can not guarantee an output.\n\
  - use 'QbB' to scale the input to a cube.\n\
  - use 'Po' to produce output and prevent partitioning for flipped facets\n\
  - use 'V0' to set min. distance to visible facet as 0 instead of roundoff\n\
  - use 'En' to specify a maximum roundoff error less than %2.2g.\n\
  - options 'Qf', 'Qbb', and 'QR0' may also help\n",
               qh->DISTround);
    // 输出关于精度问题的单精度警告（如果是单精度编译）
    qh_fprintf(qh, fp, 9374, "\
\n\
Qhull is currently using single precision arithmetic.  The following\n\
will probably remove the precision problems:\n\
  - recompile qhull for realT precision(#define REALfloat 0 in user_r.h).\n\
\n\
/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printhelp_degenerate">-</a>

  qh_printhelp_degenerate(qh, fp )
    prints descriptive message for qhull degenerate input

  notes:
    no message if qh_QUICKhelp
*/
void qh_printhelp_degenerate(qhT *qh, FILE *fp) {

  if (!qh_QUICKhelp) {
    qh_fprintf(qh, fp, 9035, "\n\
To guarantee simplicial output:\n\
  - use option 'Qt' to triangulate the output\n\
  - use option 'QJ' to joggle the input points and remove precision errors\n\
  - use option 'Ft' to triangulate the output by adding points\n\
  - use exact arithmetic (see \"Imprecision in Qhull\", qh-impre.htm)\n\
");
  }
} /* printhelp_degenerate */

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printhelp_internal">-</a>

  qh_printhelp_internal(qh, fp )
    prints descriptive message for qhull internal error with qh_ERRqhull

  notes:
    no message if qh_QUICKhelp
*/
void qh_printhelp_internal(qhT *qh, FILE *fp) {

  if (!qh_QUICKhelp) {
    qh_fprintf(qh, fp, 9426, "\n\
A Qhull internal error has occurred.  Please send the input and output to\n\
qhull_bug@qhull.org. If you can duplicate the error with logging ('T4z'), please\n\
include the log file.\n");
  }
} /* printhelp_internal */

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printhelp_narrowhull">-</a>

  qh_printhelp_narrowhull(qh, fp, minangle )
    Warn about a narrow hull

  notes:
    Alternatively, reduce qh_WARNnarrow in user_r.h

*/
void qh_printhelp_narrowhull(qhT *qh, FILE *fp, realT minangle) {

    qh_fprintf(qh, fp, 7089, "qhull precision warning: The initial hull is narrow.  Is the input lower\n\
dimensional (e.g., a square in 3-d instead of a cube)?  Cosine of the minimum\n\
angle is %.16f.  If so, Qhull may produce a wide facet.\n\
Options 'Qs' (search all points), 'Qbb' (scale last coordinate), or\n\
'QbB' (scale to unit box) may remove this warning.\n\
See 'Limitations' in qh-impre.htm.  Use 'Pp' to skip this warning.\n",
          -minangle);   /* convert from angle between normals to angle between facets */
} /* printhelp_narrowhull */

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printhelp_singular">-</a>

  qh_printhelp_singular(qh, fp )
    prints descriptive message for singular input
*/
void qh_printhelp_singular(qhT *qh, FILE *fp) {
  facetT *facet;
  vertexT *vertex, **vertexp;
  realT min, max, *coord, dist;
  int i,k;

  qh_fprintf(qh, fp, 9376, "\n\
The input to qhull appears to be less than %d dimensional, or a\n\
computation has overflowed.\n\n\
Qhull could not construct a clearly convex simplex from points:\n",
           qh->hull_dim);
  qh_printvertexlist(qh, fp, "", qh->facet_list, NULL, qh_ALL);
  if (!qh_QUICKhelp)
    qh_fprintf(qh, fp, 9377, "\n\
The center point is coplanar with a facet, or a vertex is coplanar\n\
with a neighboring facet.  The maximum round off error for\n\
computing distances is %2.2g.  The center point, facets and distances\n\
to the center point are as follows:\n\n", qh->DISTround);
  qh_printpointid(qh, fp, "center point", qh->hull_dim, qh->interior_point, qh_IDunknown);
  qh_fprintf(qh, fp, 9378, "\n");
  FORALLfacets {
    qh_fprintf(qh, fp, 9379, "facet");
    // 对于每个面元的顶点列表中的每个顶点，依次执行以下操作
    FOREACHvertex_(facet->vertices)
      // 使用 qh_pointid 函数获取顶点对应的点的标识号，并写入文件流 fp
      qh_fprintf(qh, fp, 9380, " p%d", qh_pointid(qh, vertex->point));
    // 增加 Zdistio 计数器的值
    zinc_(Zdistio);
    // 计算 qh->interior_point 到当前面元 facet 的距离，并将结果保存到 dist 变量中
    qh_distplane(qh, qh->interior_point, facet, &dist);
    // 将 dist 的值格式化写入文件流 fp，以 " distance= %4.2g\n" 的格式
    qh_fprintf(qh, fp, 9381, " distance= %4.2g\n", dist);
  }
  // 如果非快速帮助模式
  if (!qh_QUICKhelp) {
    // 如果存在 qh->HALFspace
    if (qh->HALFspace)
      // 在文件流 fp 中打印额外的空行和换行符
      qh_fprintf(qh, fp, 9382, "\n\
These points are the dual of the given halfspaces.  They indicate that\n\
the intersection is degenerate.\n");
    qh_fprintf(qh, fp, 9383,"\n\
These points either have a maximum or minimum x-coordinate, or\n\
they maximize the determinant for k coordinates.  Trial points\n\
are first selected from points that maximize a coordinate.\n");


// 打印关于点集合的说明，这些点是给定半空间的对偶。它们表明交集是退化的。
    qh_fprintf(qh, fp, 9383,"\n\
// 打印关于点集合的说明，这些点要么具有最大或最小的 x 坐标，要么最大化 k 个坐标的行列式。
// 试验点首先从最大化某个坐标的点中选择。
    if (qh->hull_dim >= qh_INITIALmax)
      qh_fprintf(qh, fp, 9384, "\n\
// 如果维度较高，则使用最小和最大 x 坐标的点（如果行列式非零）。选项 'Qs' 将做得更好，尽管速度要慢得多。
// 可以通过随机旋转输入来更改点集，使用 'QR0' 替代 'Qs'。
  }
  qh_fprintf(qh, fp, 9385, "\nThe min and max coordinates for each dimension are:\n");
  for (k=0; k < qh->hull_dim; k++) {
    min= REALmax;
    max= -REALmin;
    for (i=qh->num_points, coord= qh->first_point+k; i--; coord += qh->hull_dim) {
      maximize_(max, *coord);
      minimize_(min, *coord);
    }
    qh_fprintf(qh, fp, 9386, "  %d:  %8.4g  %8.4g  difference= %4.4g\n", k, min, max, max-min);
  }
  if (!qh_QUICKhelp) {
    qh_fprintf(qh, fp, 9387, "\n\
// 如果输入应该是全维度的，您有几个选择来确定初始单纯形：
//  - 使用 'QJ' 来摇动输入使其全维度
//  - 使用 'QbB' 将点缩放到单位立方体
//  - 使用 'QR0' 随机旋转输入以获得不同的最大点
//  - 使用 'Qs' 搜索所有点以获取初始单纯形
//  - 使用 'En' 指定一个小于 %2.2g 的最大舍入误差。
//  - 使用 'T3' 跟踪执行以查看每个点的行列式。
                     qh->DISTround);
#if REALfloat
    qh_fprintf(qh, fp, 9388, "\
  - 重新编译 qhull 以获得 realT 精度（在 libqhull_r.h 中 #define REALfloat 0）。
#endif
    qh_fprintf(qh, fp, 9389, "\n\
// 如果输入是低维度的：
//  - 使用 'QJ' 来摇动输入使其全维度
//  - 使用 'Qbk:0Bk:0' 从输入中删除坐标 k。应选择范围最小的坐标。凸壳将具有正确的拓扑结构。
//  - 确定包含点的平面，将点旋转到一个坐标平面中，并删除其他坐标。
//  - 添加一个或多个点以使输入全维度。
");
  }
} /* printhelp_singular */

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printhelp_topology">-</a>

  qh_printhelp_topology(qh, fp )
    prints descriptive message for qhull topology error with qh_ERRtopology

  notes:
    no message if qh_QUICKhelp
*/
void qh_printhelp_topology(qhT *qh, FILE *fp) {

  if (!qh_QUICKhelp) {
    qh_fprintf(qh, fp, 9427, "\n\
// Qhull 拓扑错误已发生。Qhull 未能从面合并和顶点合并中恢复。
// 这通常发生在输入几乎退化且进行了大量合并时。
/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printhelp_topology">-</a>

  printhelp_topology(qh, fp )
    prints descriptive message for qhull topology facets

  notes:
    no message if qh_QUICKhelp
*/
void printhelp_topology(qhT *qh, FILE *fp) {

  if (!qh_QUICKhelp) {
    qh_fprintf(qh, fp, 9428, "\n\
A topology facet has been identified. This facet typically arises due to topological constraints or requirements in the input data set.\n\
See http://www.qhull.org/html/qh-impre.htm#limit\n");
  }
} /* printhelp_topology */

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="printhelp_wide">-</a>

  qh_printhelp_wide(qh, fp )
    prints descriptive message for qhull wide facet with qh_ERRwide

  notes:
    no message if qh_QUICKhelp
*/
void qh_printhelp_wide(qhT *qh, FILE *fp) {

  if (!qh_QUICKhelp) {
    qh_fprintf(qh, fp, 9428, "\n\
A wide merge error has occurred.  Qhull has produced a wide facet due to facet merges and vertex merges.\n\
This usually occurs when the input is nearly degenerate and substantial merging has occurred.\n\
See http://www.qhull.org/html/qh-impre.htm#limit\n");
  }
} /* printhelp_wide */

/*-<a                             href="qh-user_r.htm#TOC"
  >-------------------------------</a><a name="user_memsizes">-</a>

  qh_user_memsizes(qh)
    allocate up to 10 additional, quick allocation sizes

  notes:
    increase maximum number of allocations in qh_initqhull_mem()
*/
void qh_user_memsizes(qhT *qh) {

  QHULL_UNUSED(qh)
  /* qh_memsize(qh, size); */
} /* user_memsizes */
```