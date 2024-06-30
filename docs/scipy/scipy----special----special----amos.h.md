# `D:\src\scipysrc\scipy\scipy\special\special\amos.h`

```
#pragma once

#include "amos/amos.h"  // 包含 AMOS 库头文件
#include "error.h"      // 包含错误处理相关头文件

namespace special {

// 将 AMOS 库的 ierr 值映射为 sf_error 的等价值
inline sf_error_t ierr_to_sferr(int nz, int ierr) {
    /* Return sf_error equivalents for amos ierr values */
    
    // 如果 nz 不为零，返回 SF_ERROR_UNDERFLOW
    if (nz != 0) {
        return SF_ERROR_UNDERFLOW;
    }

    // 根据 ierr 的不同值，返回相应的 sf_error 错误码
    switch (ierr) {
    case 1:
        return SF_ERROR_DOMAIN;     // ierr 为 1，返回 SF_ERROR_DOMAIN
    case 2:
        return SF_ERROR_OVERFLOW;   // ierr 为 2，返回 SF_ERROR_OVERFLOW
    case 3:
        return SF_ERROR_LOSS;       // ierr 为 3，返回 SF_ERROR_LOSS
    case 4:
        return SF_ERROR_NO_RESULT;  // ierr 为 4，返回 SF_ERROR_NO_RESULT
    case 5:                         // ierr 为 5，算法未满足终止条件
        return SF_ERROR_NO_RESULT;  // 返回 SF_ERROR_NO_RESULT
    }

    // 默认情况下返回 SF_ERROR_OK
    return SF_ERROR_OK;
}

} // namespace special
```