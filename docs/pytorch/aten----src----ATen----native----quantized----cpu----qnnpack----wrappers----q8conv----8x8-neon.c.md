# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\wrappers\q8conv\8x8-neon.c`

```py
/* Auto-generated by generate-wrappers.py script. Do not modify */

#if defined(__arm__) || defined(__aarch64__)
// 如果当前编译环境是 ARM 或者 AArch64 架构，则包含相应的 NEON 指令集优化代码
#include <q8conv/8x8-neon.c>
#endif /* defined(__arm__) || defined(__aarch64__) */
```