# `.\numpy\numpy\_core\src\multiarray\sequence.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_SEQUENCE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_SEQUENCE_H_

// 定义条件编译指令，防止头文件重复包含
// 如果 NUMPY_CORE_SRC_MULTIARRAY_SEQUENCE_H_ 宏未定义，则执行下面的内容

// 声明外部链接的 PySequenceMethods 结构体变量 array_as_sequence
extern NPY_NO_EXPORT PySequenceMethods array_as_sequence;

// 结束条件编译指令块
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_SEQUENCE_H_ */
```