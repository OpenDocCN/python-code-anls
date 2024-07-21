# `.\pytorch\torch\lib\libshm\err.h`

```
// 宏定义 `SYSCHECK_ERR_RETURN_NEG1(expr)`，用于检查表达式 `expr` 的返回值
// 如果 `expr` 返回 `-1`，则进入循环体
while (true) {
    // 检查表达式 `expr` 的返回值是否为 `-1`
    if ((expr) == -1) {
        // 如果 `errno` 为 `EINTR`，表示被中断，继续循环
        if (errno == EINTR) {
            continue;
        } else {
            // 抛出 `std::system_error` 异常，传递 `errno` 和 `std::system_category()` 作为参数
            throw std::system_error(errno, std::system_category());
        }
    } else {
        // 如果 `expr` 返回值不为 `-1`，跳出循环
        break;
    }
}
// 注释结束
```