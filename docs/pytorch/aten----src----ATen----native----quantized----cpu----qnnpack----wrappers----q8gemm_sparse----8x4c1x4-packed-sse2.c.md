# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\wrappers\q8gemm_sparse\8x4c1x4-packed-sse2.c`

```
#if defined(__i386__) || defined(__i686__) || defined(__x86_64__)
    # 如果编译目标是 __i386__、__i686__ 或者 __x86_64__ 中的任何一个，则编译以下代码片段

    #include <q8gemm_sparse/8x4-packA-sse2.c>
    # 包含 SSE2 指令集优化的 q8gemm_sparse/8x4-packA-sse2.c 文件

    #include <q8gemm_sparse/8x4c1x4-dq-packedA-sse2.c>
    # 包含 SSE2 指令集优化的 q8gemm_sparse/8x4c1x4-dq-packedA-sse2.c 文件

#endif /* defined(__i386__) || defined(__i686__) || defined(__x86_64__) */
    # 结束条件编译指令，说明这段代码只在上述定义的架构中进行编译
```