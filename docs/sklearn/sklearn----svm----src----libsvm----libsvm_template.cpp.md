# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\libsvm\libsvm_template.cpp`

```
/* this is a hack to generate libsvm with both sparse and dense
   methods in the same binary*/
#define _DENSE_REP
#include "svm.cpp"
#undef _DENSE_REP
#include "svm.cpp"


注释：


/* this is a hack to generate libsvm with both sparse and dense
   methods in the same binary*/
// 定义宏_DENSE_REP，用于启用稠密表示方法
#define _DENSE_REP
// 包含"sparse"方法的svm.cpp文件，编译时会使用稠密表示方法
#include "svm.cpp"
// 取消定义宏_DENSE_REP，以便后续能够包含"sparse"方法的svm.cpp文件
#undef _DENSE_REP
// 再次包含"svm.cpp"，此时编译时不使用稠密表示方法
#include "svm.cpp"
```