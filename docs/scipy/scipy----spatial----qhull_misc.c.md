# `D:\src\scipysrc\scipy\scipy\spatial\qhull_misc.c`

```
/*
   这是 qhull_src/src/user_r.c:qh_new_qhull 的一个补丁版本，
   增加了一个名为 "feaspoint" 的额外参数。

   详见 qhull_src/README.scipy
*/
int qh_new_qhull_scipy(qhT *qh, int dim, int numpoints, coordT *points, boolT ismalloc,
                char *qhull_cmd, FILE *outfile, FILE *errfile, coordT* feaspoint) {
  /* gcc 可能会针对 dim、points 和 ismalloc 发出 "可能被污染" 的警告 [-Wclobbered]。
     这些参数在 longjmp() 后不再被引用，因此不会被污染。
     参见 http://stackoverflow.com/questions/7721854/what-sense-do-these-clobbered-variable-warnings-make */
  int exitcode, hulldim;
  boolT new_ismalloc;
  coordT *new_points;

  // 如果 errfile 为空，则使用标准错误流 stderr
  if(!errfile){
    errfile= stderr;
  }
  // 如果 qh->qhmem.ferr 为空，则初始化 qhull 内存
  if (!qh->qhmem.ferr) {
    qh_meminit(qh, errfile);
  } else {
    qh_memcheck(qh);
  }
  // 如果 qhull_cmd 不以 "qhull " 开头，或者不等于 "qhull"，则输出错误信息并返回错误代码
  if (strncmp(qhull_cmd, "qhull ", (size_t)6) && strcmp(qhull_cmd, "qhull") != 0) {
    qh_fprintf(qh, errfile, 6186, "qhull error (qh_new_qhull): start qhull_cmd argument with \"qhull \" or set to \"qhull\"\n");
    return qh_ERRinput;
  }
  // 初始化 qhull
  qh_initqhull_start(qh, NULL, outfile, errfile);
  // 如果点数为 0 且 points 为空，则初始化 Qhull 并返回成功代码
  if(numpoints==0 && points==NULL){
      trace1((qh, qh->ferr, 1047, "qh_new_qhull: initialize Qhull\n"));
      return 0;
  }
  // 输出构建 Qhull 的跟踪信息
  trace1((qh, qh->ferr, 1044, "qh_new_qhull: build new Qhull for %d %d-d points with %s\n", numpoints, dim, qhull_cmd));
  // 设置错误跳转点并执行 Qhull 运算
  exitcode= setjmp(qh->errexit);
  if (!exitcode){
    qh->NOerrexit= False;
    // 初始化 Qhull 的标志
    qh_initflags(qh, qhull_cmd);
    // 如果是计算 Delaunay 三角化，则设置 PROJECTdelaunay 标志
    if (qh->DELAUNAY)
      qh->PROJECTdelaunay= True;
    // 如果是半空间运算，则设置适用点（feasible_point）
    if (qh->HALFspace) {
      // points 是一个半空间数组，每个半空间的最后一个坐标是其偏移量
      hulldim= dim-1;
      // 如果有给定的 feaspoint 参数，则设置可行点
      if(feaspoint)
      {
        coordT* coords;
        coordT* value;
        int i;
        // 分配并复制 feaspoint 到 qh->feasible_point
        if (!(qh->feasible_point= (pointT*)qh_malloc(hulldim * sizeof(coordT)))) {
          qh_fprintf(qh, qh->ferr, 6079, "qhull error: insufficient memory for 'Hn,n,n'\n");
          qh_errexit(qh, qh_ERRmem, NULL, NULL);
        }
        coords = qh->feasible_point;
        value = feaspoint;
        for(i = 0; i < hulldim; ++i)
        {
          *(coords++) = *(value++);
        }
      }
      else
      {
        // 否则使用 qh_setfeasible 设置可行点
        qh_setfeasible(qh, hulldim);
      }
      // 设置所有半空间并返回新的点集合
      new_points= qh_sethalfspace_all(qh, dim, numpoints, points, qh->feasible_point);
      new_ismalloc= True;
      // 如果原始点集合是动态分配的，则释放之
      if (ismalloc)
        qh_free(points);
    }else {
      // 否则使用原始的点集合和标志初始化 Qhull
      hulldim= dim;
      new_points= points;
      new_ismalloc= ismalloc;
    }
    // 初始化 Qhull 结构并进行 Qhull 计算
    qh_init_B(qh, new_points, numpoints, hulldim, new_ismalloc);
    qh_qhull(qh);
    // 检查 Qhull 输出
    qh_check_output(qh);
    // 根据情况输出结果
    if (outfile) {
      qh_produce_output(qh);
    }else {
      qh_prepare_output(qh);
    }
    // 如果启用了 VERIFYoutput 并且没有强制停止，则检查计算点
    if (qh->VERIFYoutput && !qh->FORCEoutput && !qh->STOPadd && !qh->STOPcone && !qh->STOPpoint)
      qh_check_points(qh);
  }
  // 恢复错误处理设置
  qh->NOerrexit= True;
  // 返回退出代码
  return exitcode;
} /* new_qhull */
```