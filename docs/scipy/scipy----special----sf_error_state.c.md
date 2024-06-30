# `D:\src\scipysrc\scipy\scipy\special\sf_error_state.c`

```
#include <stdlib.h>

#include "sf_error_state.h"


/* 如果不使用 volatile，Clang 可能会尝试优化掉这个数组 */
/* 定义静态的 volatile 数组 sf_error_actions，存储了对应错误码的处理动作 */
static volatile sf_action_t sf_error_actions[] = {
    SF_ERROR_IGNORE, /* SF_ERROR_OK */
    SF_ERROR_IGNORE, /* SF_ERROR_SINGULAR */
    SF_ERROR_IGNORE, /* SF_ERROR_UNDERFLOW */
    SF_ERROR_IGNORE, /* SF_ERROR_OVERFLOW */
    SF_ERROR_IGNORE, /* SF_ERROR_SLOW */
    SF_ERROR_IGNORE, /* SF_ERROR_LOSS */
    SF_ERROR_IGNORE, /* SF_ERROR_NO_RESULT */
    SF_ERROR_IGNORE, /* SF_ERROR_DOMAIN */
    SF_ERROR_IGNORE, /* SF_ERROR_ARG */
    SF_ERROR_IGNORE, /* SF_ERROR_OTHER */
    SF_ERROR_IGNORE  /* SF_ERROR__LAST */
};

/* 定义设置特定错误码处理动作的函数 */
SCIPY_DLL void scipy_sf_error_set_action(sf_error_t code, sf_action_t action)
{
    /* 根据错误码将其对应位置的处理动作设定为指定的 action */
    sf_error_actions[(int)code] = action;
}

/* 定义获取特定错误码处理动作的函数 */
SCIPY_DLL sf_action_t scipy_sf_error_get_action(sf_error_t code)
{
    /* 返回指定错误码对应的处理动作 */
    return sf_error_actions[(int)code];
}
```