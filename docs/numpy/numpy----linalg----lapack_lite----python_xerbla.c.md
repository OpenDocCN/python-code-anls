# `.\numpy\numpy\linalg\lapack_lite\python_xerbla.c`

```
/*
  定义宏以清除 Py_ssize_t 的定义
  包含 Python.h 头文件
  包含 numpy 库的通用头文件和 nmpy_cblas.h 头文件
*/

/*
  从原始手册页：
  --------------------------
  XERBLA 是 LAPACK 程序的错误处理程序。
  如果输入参数具有无效值，则由 LAPACK 程序调用它。
  将打印一条消息并停止执行。

  不打印消息和停止执行，而是引发一个带有消息的 ValueError 异常。

  参数:
  -----------
  srname: 错误消息中使用的子程序名称，最多六个字符。
          结尾的空格会被跳过。
  info: 无效参数的编号。
*/

CBLAS_INT BLAS_FUNC(xerbla)(char *srname, CBLAS_INT *info)
{
        static const char format[] = "On entry to %.*s" \
                " parameter number %d had an illegal value";
        char buf[sizeof(format) + 6 + 4];   /* 6 for name, 4 for param. num. */

        int len = 0; /* 子程序名称的长度 */
        PyGILState_STATE save;

        while( len<6 && srname[len]!='\0' )
                len++;
        while( len && srname[len-1]==' ' )
                len--;
        
        // 保证全局解释器锁（GIL）被获取
        save = PyGILState_Ensure();
        
        // 使用给定的格式化字符串生成错误消息
        PyOS_snprintf(buf, sizeof(buf), format, len, srname, (int)*info);
        
        // 设置 ValueError 异常并传递错误消息
        PyErr_SetString(PyExc_ValueError, buf);
        
        // 释放全局解释器锁（GIL）
        PyGILState_Release(save);

        // 返回 0，表示出错处理函数的返回值
        return 0;
}
```