# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\coo_entries.h`

```
#ifndef CKDTREE_COO_ENTRIES
#define CKDTREE_COO_ENTRIES



// 如果 CKDTREE_COO_ENTRIES 宏未定义，则定义它，用于避免重复包含结构体定义
struct coo_entry {
    // 结构体成员 i，表示索引 i
    ckdtree_intp_t i;
    // 结构体成员 j，表示索引 j
    ckdtree_intp_t j;
    // 结构体成员 v，表示值 v，为双精度浮点数
    double v;
};



#endif


这段代码是一个 C/C++ 头文件中的预处理指令和结构体定义。头文件使用 `#ifndef`、`#define`、`#endif` 来防止多重包含，确保结构体只在第一次引入时被定义。结构体 `coo_entry` 定义了三个成员变量：`i` 和 `j` 都是 `ckdtree_intp_t` 类型的整数，`v` 是双精度浮点数，用于表示一个 COO 格式的条目（即稀疏矩阵中的非零元素）。
```