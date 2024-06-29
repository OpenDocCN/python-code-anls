# `.\numpy\numpy\_core\src\common\python_xerbla.c`

```
/*
  From the original manpage:
  --------------------------
  XERBLA is an error handler for the LAPACK routines.
  It is called by an LAPACK routine if an input parameter has an invalid value.
  A message is printed and execution stops.

  Instead of printing a message and stopping the execution, a
  ValueError is raised with the message.

  Parameters:
  -----------
  srname: Subroutine name to use in error message, maximum six characters.
          Spaces at the end are skipped.
  info: Number of the invalid parameter.
*/

// 定义函数 BLAS_FUNC(xerbla)，用于处理 LAPACK 函数的错误
CBLAS_INT BLAS_FUNC(xerbla)(char *srname, CBLAS_INT *info)
{
        static const char format[] = "On entry to %.*s" \
                " parameter number %d had an illegal value";
        char buf[sizeof(format) + 6 + 4];   /* 6 for name, 4 for param. num. */

        int len = 0; /* subroutine name 的长度 */
        PyGILState_STATE save;

        // 计算 subroutine name 的实际长度，跳过末尾的空格
        while( len<6 && srname[len]!='\0' )
                len++;
        while( len && srname[len-1]==' ' )
                len--;

        // 确保全局解释器锁（GIL）已获取
        save = PyGILState_Ensure();
        // 格式化错误信息并设置为 ValueError 异常的内容
        PyOS_snprintf(buf, sizeof(buf), format, len, srname, (int)*info);
        PyErr_SetString(PyExc_ValueError, buf);
        // 释放全局解释器锁（GIL）
        PyGILState_Release(save);

        // 返回 0 表示处理完成，无返回值
        return 0;
}
```