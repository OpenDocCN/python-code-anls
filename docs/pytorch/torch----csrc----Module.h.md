# `.\pytorch\torch\csrc\Module.h`

```py
#ifndef THP_MODULE_INC
#define THP_MODULE_INC

定义了一个宏 `THP_MODULE_INC`，用于条件编译防止重复包含。


#define THP_STATELESS_ATTRIBUTE_NAME "_torch"

定义了另一个宏 `THP_STATELESS_ATTRIBUTE_NAME`，其值为 `"_torch"`，这个宏可能在代码中用于标识某个状态属性的名称。


#endif

结束条件编译指令 `#ifndef`，确保在同一个文件中这些宏的定义只会包含一次。
```