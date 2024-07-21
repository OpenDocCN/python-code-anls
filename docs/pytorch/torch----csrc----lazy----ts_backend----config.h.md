# `.\pytorch\torch\csrc\lazy\ts_backend\config.h`

```py
#pragma once
// 包含<c10/util/Flags.h>头文件，该头文件提供了处理标志的功能

// 定义一个布尔类型的标志，名称为torch_lazy_ts_tensor_update_sync，声明在这里，实际定义在其他地方
C10_DECLARE_bool(torch_lazy_ts_tensor_update_sync);

// 定义另一个布尔类型的标志，名称为torch_lazy_ts_cuda，声明在这里，实际定义在其他地方
C10_DECLARE_bool(torch_lazy_ts_cuda);
```