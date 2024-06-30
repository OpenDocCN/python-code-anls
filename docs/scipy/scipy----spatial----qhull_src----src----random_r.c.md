# `D:\src\scipysrc\scipy\scipy\spatial\qhull_src\src\random_r.c`

```
/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="argv_to_command">-</a>

  qh_argv_to_command( argc, argv, command, max_size )

    build command from argc/argv
    max_size is at least

  returns:
    a space-delimited string of options (just as typed)
    returns false if max_size is too short

  notes:
    silently removes
    makes option string easy to input and output
    matches qh_argv_to_command_size
    argc may be 0
*/
int qh_argv_to_command(int argc, char *argv[], char* command, int max_size) {
  int i, remaining;
  char *s;
  *command= '\0';  /* max_size > 0 */

  // 如果有参数
  if (argc) {
    // 获取程序文件名（去除路径和扩展名）
    if ((s= strrchr( argv[0], '\\')) /* get filename w/o .exe extension */
    || (s= strrchr( argv[0], '/')))
        s++;
    else
        s= argv[0];
    // 将文件名复制到command中，如果长度符合要求
    if ((int)strlen(s) < max_size)   /* WARN64 */
        strcpy(command, s);
    else
        goto error_argv; // 如果长度超过max_size，跳转到错误处理
    // 如果文件名包含扩展名".EXE"或".exe"，则去除扩展名
    if ((s= strstr(command, ".EXE"))
    ||  (s= strstr(command, ".exe")))
        *s= '\0';
  }
  // 处理每一个参数
  for (i=1; i < argc; i++) {
    s= argv[i];
    remaining= max_size - (int)strlen(command) - (int)strlen(s) - 2;   /* WARN64 */
    // 如果参数为空或者包含空格，需要进行特殊处理
    if (!*s || strchr(s, ' ')) {
      char *t= command + strlen(command);
      remaining -= 2;
      // 如果剩余空间不足，跳转到错误处理
      if (remaining < 0) {
        goto error_argv;
      }
      *t++= ' ';
      *t++= '"';
      // 处理参数中的特殊字符和引号
      while (*s) {
        if (*s == '"') {
          if (--remaining < 0)
            goto error_argv;
          *t++= '\\';
        }
        *t++= *s++;
      }
      *t++= '"';
      *t= '\0';
    }else if (remaining < 0) {
      goto error_argv; // 如果剩余空间不足，跳转到错误处理
    }else {
      // 将参数添加到命令字符串中
      strcat(command, " ");
      strcat(command, s);
    }
  }
  return 1; // 成功构建命令，返回1

error_argv:
  return 0; // 构建命令失败，返回0
} /* argv_to_command */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="argv_to_command_size">-</a>

  qh_argv_to_command_size( argc, argv )

    return size to allocate for qh_argv_to_command()

  notes:
    only called from rbox with qh_errexit not enabled
    caller should report error if returned size is less than 1
    argc may be 0
    actual size is usually shorter
*/
int qh_argv_to_command_size(int argc, char *argv[]) {
    int count= 1; /* null-terminator if argc==0 */
    int i;
    char *s;

    // 计算命令字符串的最小大小
    for (i = 0; i < argc; i++) {
        count += strlen(argv[i]) + 1; // 每个参数的长度加上一个空格
    }
    return count; // 返回计算出的大小
} /* argv_to_command_size */
    # 初始化计数器，用于统计所有参数及其长度总和，每个参数长度加1（包括字符串末尾的空字符）
    for (i=0; i<argc; i++){
      count += (int)strlen(argv[i]) + 1;   /* WARN64 */
      # 如果当前参数的索引大于0且参数中包含空格，则需要额外计算加上两个引号的长度
      if (i>0 && strchr(argv[i], ' ')) {
        count += 2;  /* quote delimiters */
        # 遍历当前参数字符串，检查是否包含双引号，如果有则需要额外增加一个字符的长度
        for (s=argv[i]; *s; s++) {
          if (*s == '"') {
            count++;
          }
        }
      }
    }
    # 返回最终计算得到的总字符数
    return count;
} /* argv_to_command_size */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="rand">-</a>

  qh_rand()
  qh_srand(qh, seed )
    generate pseudo-random number between 1 and 2^31 -2

  notes:
    For qhull and rbox, called from qh_RANDOMint(),etc. [user_r.h]

    From Park & Miller's minimal standard random number generator
      Communications of the ACM, 31:1192-1201, 1988.
    Does not use 0 or 2^31 -1
      this is silently enforced by qh_srand()
    Can make 'Rn' much faster by moving qh_rand to qh_distplane
*/

/* Global variables and constants */

#define qh_rand_a 16807
#define qh_rand_m 2147483647
#define qh_rand_q 127773  /* m div a */
#define qh_rand_r 2836    /* m mod a */

/* 
   qh_rand(qh)
   Generate a pseudo-random number using Park & Miller's minimal standard algorithm.
   This function is used for generating random numbers in the range [1, 2^31 - 2].

   Parameters:
     qh: Pointer to qhT structure containing last_random as seed

   Returns:
     Pseudo-random number in the specified range

   Notes:
     - Called from qh_RANDOMint() in user_r.h
     - Ensures generated numbers are within specified range by silently enforcing constraints
     - Optimization note: Moving qh_rand to qh_distplane can greatly speed up 'Rn'
*/
int qh_rand(qhT *qh) {
    int lo, hi, test;
    int seed= qh->last_random;

    hi= seed / qh_rand_q;  /* seed div q */
    lo= seed % qh_rand_q;  /* seed mod q */
    test= qh_rand_a * lo - qh_rand_r * hi;
    if (test > 0)
        seed= test;
    else
        seed= test + qh_rand_m;
    qh->last_random= seed;
    /* seed= seed < qh_RANDOMmax/2 ? 0 : qh_RANDOMmax;  for testing */
    /* seed= qh_RANDOMmax;  for testing */
    return seed;
} /* rand */

/*
   qh_srand(qh, seed)
   Initialize the seed for the pseudo-random number generator.

   Parameters:
     qh: Pointer to qhT structure
     seed: Seed value to initialize the generator

   Notes:
     - Ensures the seed is within valid range [1, qh_rand_m - 1]
*/
void qh_srand(qhT *qh, int seed) {
    if (seed < 1)
        qh->last_random= 1;
    else if (seed >= qh_rand_m)
        qh->last_random= qh_rand_m - 1;
    else
        qh->last_random= seed;
} /* qh_srand */

/*-<a                             href="qh-geom_r.htm#TOC"
>-------------------------------</a><a name="randomfactor">-</a>

qh_randomfactor(qh, scale, offset )
  return a random factor r * scale + offset

notes:
  qh.RANDOMa/b are defined in global_r.c
  qh_RANDOMint requires 'qh'
*/
/*
   qh_randomfactor(qh, scale, offset)
   Generate a random factor r * scale + offset.

   Parameters:
     qh: Pointer to qhT structure
     scale: Scaling factor
     offset: Offset value

   Returns:
     Random factor in the range [offset, offset + scale]
*/
realT qh_randomfactor(qhT *qh, realT scale, realT offset) {
    realT randr;

    randr= qh_RANDOMint;
    return randr * scale + offset;
} /* randomfactor */

/*-<a                             href="qh-geom_r.htm#TOC"
>-------------------------------</a><a name="randommatrix">-</a>

  qh_randommatrix(qh, buffer, dim, rows )
    generate a random dim X dim matrix in range [-1,1]
    assumes buffer is [dim+1, dim]

  returns:
    sets buffer to random numbers
    sets rows to rows of buffer
    sets row[dim] as scratch row

  notes:
    qh_RANDOMint requires 'qh'
*/
/*
   qh_randommatrix(qh, buffer, dim, rows)
   Generate a random dim X dim matrix with elements in the range [-1, 1].

   Parameters:
     qh: Pointer to qhT structure
     buffer: Buffer to store the matrix
     dim: Dimension of the matrix (dim X dim)
     rows: Pointer to array of row pointers

   Notes:
     - Assumes buffer is allocated as [dim+1, dim]
     - Uses qh_RANDOMint to generate random numbers
*/
void qh_randommatrix(qhT *qh, realT *buffer, int dim, realT **rows) {
    int i, k;
    realT **rowi, *coord, realr;

    coord= buffer;
    rowi= rows;
    for (i=0; i < dim; i++) {
        *(rowi++)= coord;
        for (k=0; k < dim; k++) {
            realr= qh_RANDOMint;
            *(coord++)= 2.0 * realr/(qh_RANDOMmax+1) - 1.0;
        }
    }
    *rowi= coord;
} /* randommatrix */

/*-<a                             href="qh-globa_r.htm#TOC"
  >-------------------------------</a><a name="strtol">-</a>

  qh_strtol( s, endp) qh_strtod( s, endp)
    internal versions of strtol() and strtod()
    does not skip trailing spaces
  notes:
    some implementations of strtol()/strtod() skip trailing spaces
*/
# 将字符串转换为双精度浮点数
double qh_strtod(const char *s, char **endp) {
  double result;

  result= strtod(s, endp);  // 使用 strtod 函数将字符串 s 转换为双精度浮点数，endp 用于存储转换结束的位置
  // 如果转换后的字符串位置小于 endp，并且 endp 的前一个字符是空格，则将 endp 向前移动一位
  if (s < (*endp) && (*endp)[-1] == ' ')
    (*endp)--;
  return result;  // 返回转换后的双精度浮点数结果
} /* strtod */

# 将字符串转换为长整型整数
int qh_strtol(const char *s, char **endp) {
  int result;

  result= (int) strtol(s, endp, 10);     /* WARN64 */  // 使用 strtol 函数将字符串 s 转换为十进制长整型整数，忽略任何 WARN64 注释
  // 如果转换后的字符串位置小于 endp，并且 endp 的前一个字符是空格，则将 endp 向前移动一位
  if (s< (*endp) && (*endp)[-1] == ' ')
    (*endp)--;
  return result;  // 返回转换后的长整型整数结果
} /* strtol */
```